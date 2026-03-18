"""
agent.py - The Core ReAct Loop for KernelForge RL

This module runs the autonomous optimization loop for a given PyTorch program.
It prompts an LLM to generate optimized CUDA C++ code, wraps it in a load_inline
Python script, evaluates it in the sandbox, runs the profiler if it passes,
calculates the speedup reward, and feeds diagnostics back to the LLM.
"""

import sys
import re
from typing import List, Dict, Any, Tuple
from sandbox import evaluate
from profiler import profile_kernel
from reward import calculate_reward
from sys_prompt import get_system_prompt


def _fix_cuda_api(cuda_code: str) -> str:
    """
    Fix common PyTorch C++ API mismatches the SFT model learned incorrectly.
    Applied before every sandbox compilation.
    """
    # current_device(): torch::cuda:: and at::cuda:: namespaces don't have it
    cuda_code = re.sub(
        r'(?:torch|at)::cuda::current_device\(\)',
        'c10::cuda::current_device()',
        cuda_code,
    )

    # getCurrentCUDAStream(): remove from <<<>>> launches (default stream is fine)
    cuda_code = re.sub(
        r',\s*(?:at|c10|torch)::cuda::getCurrentCUDAStream\([^)]*\)\s*(>>>)',
        r'\1',
        cuda_code,
    )
    # Any remaining getCurrentCUDAStream() call → replace with 0 (null/default stream)
    cuda_code = re.sub(
        r'(?:at|c10|torch)::cuda::getCurrentCUDAStream\([^)]*\)',
        '0',
        cuda_code,
    )

    # __fabsf / __fabs / __sqrtf are host-only intrinsics; device code uses plain versions
    for fn in ('fabsf', 'fabs', 'sqrtf', 'sqrt', 'expf', 'exp', 'logf', 'log'):
        cuda_code = cuda_code.replace(f'__{fn}(', f'{fn}(')

    # std:: math functions are not available in device code
    cuda_code = re.sub(r'std::(max|min|abs|fabs)\s*\(', lambda m: {
        'max': 'fmaxf(', 'min': 'fminf(', 'abs': 'fabsf(', 'fabs': 'fabsf('
    }[m.group(1)], cuda_code)

    # __host__ or __device__ on torch::Tensor binding functions is invalid.
    # The binding function must be a plain host function callable from Python.
    # Remove __host__, __device__, __forceinline__ prefixes before torch::Tensor returns.
    cuda_code = re.sub(
        r'(?:__host__|__device__|__forceinline__)\s+(?=(?:inline\s+)?(?:std::vector<torch::Tensor>|torch::Tensor)\s+\w+\s*\()',
        '',
        cuda_code,
    )
    # Also strip __host__ __device__ combos (can appear in either order)
    cuda_code = re.sub(
        r'(?:__host__\s+__device__|__device__\s+__host__)\s+(?=(?:inline\s+)?(?:std::vector<torch::Tensor>|torch::Tensor)\s+\w+\s*\()',
        '',
        cuda_code,
    )

    return cuda_code


import hashlib


def _split_args(raw: str) -> list:
    """Split a function argument string by commas, respecting nested parens/brackets.

    Fixes the bug where stride=(1,1,1) gets shredded into ['stride=(1','1','1)'].
    """
    args, depth, current = [], 0, []
    for ch in raw:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            a = ''.join(current).strip()
            if a:
                args.append(a)
            current = []
        else:
            current.append(ch)
    if current:
        a = ''.join(current).strip()
        if a:
            args.append(a)
    return args


def build_load_inline_wrapper(cuda_code: str, ref_code: str) -> str:
    """
    Wrap raw CUDA C++ code in a full load_inline Python script.
    
    Parses the C++ to extract binding function signatures, and parses the
    reference PyTorch code to get forward() arguments, then generates the
    complete Python wrapper with ModelNew class.
    """
    # 0. Strip PYBIND11_MODULE block — load_inline generates its own, having two causes linker errors.
    lines = cuda_code.split('\n')
    result, skip, brace_depth = [], False, 0
    for line in lines:
        if not skip and 'PYBIND11_MODULE' in line:
            skip = True
            brace_depth = 0
        if skip:
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0 and '{' in cuda_code:
                skip = False
            continue
        result.append(line)
    cuda_code = '\n'.join(result)

    # 1. Normalise all Tensor type spellings to torch::Tensor before any parsing.
    cuda_code = re.sub(r'\bat::Tensor\b', 'torch::Tensor', cuda_code)
    # Bare 'Tensor' (no namespace): lookbehind prevents double-expanding torch::Tensor.
    cuda_code = re.sub(r'(?<!:)\bTensor\b', 'torch::Tensor', cuda_code)

    # Find all torch::Tensor binding function signatures
    sig_pattern = r'(torch::Tensor\s+(\w+)\s*\([^)]*\))\s*\{'
    sig_matches = re.findall(sig_pattern, cuda_code)

    if not sig_matches:
        # Fallback 1: semicolon-terminated declarations
        sig_matches = re.findall(r'(torch::Tensor\s+(\w+)\s*\([^)]*\))\s*;', cuda_code)

    if not sig_matches:
        # Fallback 2: vector<torch::Tensor> return (multi-output kernels)
        vec_matches = re.findall(
            r'(std::vector<torch::Tensor>\s+(\w+)\s*\([^)]*\))\s*\{', cuda_code
        )
        if vec_matches:
            sig_matches = vec_matches

    if not sig_matches:
        # Fallback 3: any non-kernel, non-device C++ function — take the first one
        # that is NOT __global__ or __device__ and has recognisable tensor args
        for m in re.finditer(
            r'(?<!__global__\s)(?<!__device__\s)'
            r'(\w[\w:<>*&\s]+\s+(\w+)\s*\((?:[^)]*torch::[^)]*)\))\s*\{',
            cuda_code,
        ):
            full_sig, fname = m.group(1).strip(), m.group(2)
            # Rewrite return type to torch::Tensor
            fixed = re.sub(r'^[\w:<>*&\s]+(?=\s+\w+\s*\()', 'torch::Tensor', full_sig)
            cuda_code = cuda_code[:m.start(1)] + fixed + cuda_code[m.start(1) + len(full_sig):]
            sig_matches = [(fixed, fname)]
            break

    if not sig_matches:
        return None

    func_signatures = [m[0] for m in sig_matches]
    func_names = [m[1] for m in sig_matches]

    # Build cpp_source from signatures
    cpp_source = "; ".join(func_signatures) + ";"
    
    # 2. Parse reference code's Model.forward() and __init__() arg names
    # Use balanced-paren extraction so multiline/nested signatures work correctly.
    def _extract_def_args(code: str, def_name: str) -> str:
        """Return the raw arg string inside def <def_name>(self, ...) handling nested parens."""
        m = re.search(rf'def {def_name}\s*\(', code)
        if not m:
            return ""
        start = m.end() - 1  # points at '('
        depth = 0
        i = start
        while i < len(code):
            if code[i] == '(':
                depth += 1
            elif code[i] == ')':
                depth -= 1
                if depth == 0:
                    inner = code[start + 1:i]  # everything between outer parens
                    # strip leading 'self,' or 'self'
                    inner = re.sub(r'^\s*self\s*,?\s*', '', inner)
                    return inner.strip()
            i += 1
        return ""

    fwd_raw = _extract_def_args(ref_code, "forward")
    fwd_args_clean = ", ".join(
        arg.split(":")[0].split("=")[0].strip()
        for arg in _split_args(fwd_raw) if arg.strip()
    ) if fwd_raw else "x"

    init_raw = _extract_def_args(ref_code, "__init__")
    init_args_clean = ", ".join(
        arg.split(":")[0].split("=")[0].strip()
        for arg in _split_args(init_raw) if arg.strip()
    ) if init_raw else ""

    # Use the last binding function as the one to call from forward()
    binding_func = func_names[-1]

    # 3. Parse binding function arg names to detect stateful models
    last_sig = func_signatures[-1]
    param_list_match = re.search(r'\w+\s*\(([^)]*)\)\s*$', last_sig.split('{')[0])
    binding_arg_names = []
    if param_list_match:
        for param in param_list_match.group(1).split(','):
            words = re.findall(r'\b\w+\b', param.strip())
            if words:
                binding_arg_names.append(words[-1])

    forward_args_list = [a.strip() for a in fwd_args_clean.split(',') if a.strip()]
    forward_set = set(forward_args_list)
    extra_args = [a for a in binding_arg_names if a not in forward_set]

    # 4. Fix common PyTorch C++ API mismatches before compilation
    cuda_code = _fix_cuda_api(cuda_code)

    cuda_source_expr = repr(cuda_code)
    cpp_source_expr  = repr(cpp_source)

    # 5. Generate unique module name
    code_hash = hashlib.md5((cuda_code + cpp_source).encode("utf-8")).hexdigest()[:12]
    mod_name = f"kf_ext_{code_hash}"

    init_sig = ", " + init_args_clean if init_args_clean else ""

    # 6. For stateful models, copy __init__ body from reference and map C++ arg
    #    names to the correct self.* attribute paths.
    def _extract_init_body(code: str) -> str:
        """Return the body lines of Model.__init__, stripped of super().__init__()."""
        m = re.search(r'^([ \t]*)def __init__\s*\(', code, re.MULTILINE)
        if not m:
            return ""
        method_indent = m.group(1)
        body_indent   = method_indent + "    "
        # Skip past the signature parens to the colon
        pos = m.end() - 1
        depth = 0
        i = pos
        while i < len(code):
            if code[i] == '(':   depth += 1
            elif code[i] == ')':
                depth -= 1
                if depth == 0: i += 1; break
            i += 1
        colon = code.index(':', i)
        lines = code[colon + 1:].split('\n')
        body_lines = []
        for line in lines:
            if not line.strip():
                body_lines.append(''); continue
            if re.match(rf'^{re.escape(method_indent)}(?:def|class)\s', line):
                break
            if 'super(' in line and '__init__' in line:
                continue
            # Re-indent to 8 spaces
            if line.startswith(body_indent):
                body_lines.append('        ' + line[len(body_indent):])
            else:
                body_lines.append('        ' + line.lstrip())
        return '\n'.join(body_lines)

    def _map_extra_arg(arg: str, init_body: str) -> str:
        """Map a C++ binding arg name to the correct Python self.* accessor."""
        # Direct match: self.bias, self.weight, etc.
        if re.search(rf'\bself\.{re.escape(arg)}\b', init_body):
            return f'self.{arg}'
        # Pattern: conv_weight -> self.conv.weight, bn_running_mean -> self.bn.running_mean
        known_attrs = ('weight', 'bias', 'running_mean', 'running_var',
                       'weight_g', 'weight_v', 'scale', 'gamma', 'beta')
        for attr in known_attrs:
            if arg.endswith('_' + attr):
                module = arg[:-len(attr) - 1]
                if re.search(rf'\bself\.{re.escape(module)}\b', init_body):
                    return f'self.{module}.{attr}'
        return f'self.{arg}'

    # 7. Build the full Python wrapper
    ext_block = f'''import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = {cuda_source_expr}

cpp_source = {cpp_source_expr}

ext = load_inline(
    name="{mod_name}",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions={func_names},
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)
'''

    if extra_args:
        # Stateful: copy __init__ body from reference, map extra args to self.* paths
        init_body = _extract_init_body(ref_code)
        extra_accessors = [_map_extra_arg(a, init_body) for a in extra_args]
        all_fwd_args = fwd_args_clean + (', ' + ', '.join(extra_accessors) if extra_accessors else '')
        wrapper = ext_block + f'''
class ModelNew(torch.nn.Module):
    def __init__(self{init_sig}):
        super().__init__()
{init_body}
    def forward(self, {fwd_args_clean}):
        return ext.{binding_func}({all_fwd_args})
'''
    else:
        # Stateless
        wrapper = ext_block + f'''
class ModelNew(torch.nn.Module):
    def __init__(self{init_sig}):
        super().__init__()
    def forward(self, {fwd_args_clean}):
        return ext.{binding_func}({fwd_args_clean})
'''
    return wrapper


def _extract_cuda_code(response: str) -> str:
    """Extract the CUDA C++ code block from the LLM's response."""
    # Try cpp, cuda, c++ block markers
    for lang in ["cpp", "cuda", "c\\+\\+"]:
        match = re.search(rf"```{lang}(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # If no markdown tags are present, strip out <think> blocks 
    clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    clean_response = re.sub(r"<think>.*", "", clean_response, flags=re.DOTALL)
    
    # Check if remaining text looks like CUDA C++
    stripped = clean_response.strip()
    if any(kw in stripped for kw in ["__global__", "torch::Tensor", "#include"]):
        return stripped
        
    return ""


class KernelForgeAgent:
    def __init__(self, model_name: str = "Qwen/Qwen3-14B", mock_mode: bool = False):
        """Initialize the agent with a local HF model.
        If mock_mode is True, bypasses loading the literal HuggingFace model.
        """
        self.system_prompt = get_system_prompt()
        self.mock_mode = mock_mode
        
        if not mock_mode:
            print(f"Loading Model: {model_name}...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype="auto"
            )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM to generate the next response based on conversation history."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Pre-fill the assistant response to force C++ output (matches SFT training format)
        prefill = "```cpp\n#include <torch/extension.h>\n"
        text += prefill
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.7,   # Slight creativity for exploration
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract only the newly generated text
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def run_react_loop(self, target_program: str, max_steps: int = 5) -> Tuple[str, float]:
        """
        Execute the iterative Reasoning + Acting (ReAct) optimization loop.
        Args:
            target_program: The reference PyTorch program to compile against.
            max_steps: Maximum number of generation attempts.
        Returns:
            Tuple of (Best Source Code, Highest Reward)
        """
        print(f"\n🚀 Starting Optimization Loop (Max Steps: {max_steps})")
        
        # Initialize conversation state
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user", 
                "content": f"Write an optimized CUDA C++ kernel to replace this PyTorch implementation. Output only the C++ code.\n\nReference Program:\n```python\n{target_program}\n```"
            }
        ]

        best_code = ""
        best_reward = 0.0

        for step in range(1, max_steps + 1):
            print(f"\n--- Step {step}/{max_steps} ---")
            
            # Debug: print full message history at step 3 to verify model sees errors
            if step == 3:
                print("\n" + "="*60)
                print("🔍 DEBUG: Full Message History at Step 3")
                print("="*60)
                for i, msg in enumerate(messages):
                    role = msg["role"].upper()
                    content = msg["content"]
                    # Truncate long content but show enough to verify
                    if len(content) > 300:
                        content = content[:300] + f"\n... [{len(content)} chars total]"
                    print(f"\n[MSG {i}] {role}:")
                    print(content)
                print("="*60 + "\n")
            
            # 1. Generation
            print("🧠 Generating Kernel...")
            response = self.generate(messages)
            # Restore the prefill we forced it to start with
            full_response = "```cpp\n#include <torch/extension.h>\n" + response
            messages.append({"role": "assistant", "content": full_response})
            
            # 2. Extract CUDA C++ Code
            cuda_code = _extract_cuda_code(full_response)
            if not cuda_code:
                print("❌ Failed to extract C++ code block. Requesting fix...")
                print(f"--- FAILED GENERATED TEXT ---\n{response[:500]}\n-----------------------------")
                messages.append({"role": "user", "content": "Error: Could not find ```cpp block in your response. Please output the CUDA C++ code properly in a ```cpp code block."})
                continue
            
            # 3. Wrap in load_inline Python
            candidate_code = build_load_inline_wrapper(cuda_code, target_program)
            if not candidate_code:
                print("❌ Could not parse binding function from C++ code.")
                print(f"--- FAILED GENERATED CODE ---\n{cuda_code[:500]}\n-----------------------------")
                messages.append({"role": "user", "content": "Error: Could not find a `torch::Tensor` binding function in your code. You must include a C++ function that returns `torch::Tensor` and calls your CUDA kernel."})
                continue
                
            # 4. Sandbox Evaluation
            print("🛠️  Compiling and Evaluating in Sandbox...")
            eval_result = evaluate(candidate_code, target_program)
            
            if not eval_result["correct"]:
                # Compilation failed or output was wrong
                error_msg = eval_result.get("compiler_error") or "Outputs do not match the reference implementation exactly (Correctness Failed)."
                print(f"❌ Evaluation Failed: {error_msg.strip()[:100]}...\n")
                print(f"--- FAILED GENERATED CODE ---\n{cuda_code}\n-----------------------------")
                
                # Feed error back to LLM (show the C++ error, not Python wrapper errors)
                feedback = f"Your CUDA C++ code failed during compilation or evaluation.\n\nError Log:\n```\n{error_msg}\n```\n\nAnalyze the root cause carefully. If this is a CUDA runtime error (illegal memory access), check your shared memory sizing, indexing bounds, and output write offsets. Remember: use float* and data_ptr<float>() for float32 tensors. Fix the bug and output the corrected C++ code."
                messages.append({"role": "user", "content": feedback})
                continue
                
            runtime_ms = eval_result["runtime_ms"]
            print(f"✅ Sandbox Passed! Latency: {runtime_ms:.3f} ms")
            
            # 5. Profiling and Reward
            print("🔬 Profiling Hardware Metrics...")
            profiler_feedback = profile_kernel(candidate_code, target_program)
            
            reward = calculate_reward(eval_result)
            print(f"🏆 Reward: {reward:.2f}x Speedup")
            
            # Track best performance
            if reward > best_reward:
                best_reward = reward
                best_code = candidate_code
                
            # 6. Iteration Feedback
            if step < max_steps:
                print("🔄 Sending profiler feedback back to LLM for next iteration...")
                feedback = f"Success! Your kernel ran in {runtime_ms:.3f} ms, achieving a {reward:.2f}x speedup over the baseline.\n\nHere is the hardware profiling report:\n{profiler_feedback}\n\nCan you optimize the C++ kernel further to resolve the bottleneck? Output the improved ```cpp code."
                messages.append({"role": "user", "content": feedback})

        print(f"\n🏁 Optimization Completed. Best Reward: {best_reward:.2f}x")
        return best_code, best_reward
