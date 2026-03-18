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

    # C++17 structured bindings (auto [a, b, c] = x.sizes()) not supported by nvcc.
    # Rewrite to explicit indexed access.
    def _replace_structured_binding(m):
        names = [n.strip() for n in m.group(1).split(',')]
        rhs = m.group(2).strip()
        lines = [f"auto _sb_ = {rhs};"]
        for i, name in enumerate(names):
            lines.append(f"auto {name} = _sb_[{i}];")
        return '\n'.join(lines)
    cuda_code = re.sub(
        r'auto\s*\[([^\]]+)\]\s*=\s*([^;]+);',
        _replace_structured_binding,
        cuda_code,
    )

    # Bug 1 — extra closing paren in stacked math intrinsic expressions.
    # The model produces e.g. "return x * tanhf(logf(1.0f + expf(x))));"
    # Only fix lines with `return` that end with ); to avoid corrupting
    # legitimately unbalanced lines inside multi-line function calls.
    def _fix_math_line(line: str) -> str:
        stripped = line.rstrip()
        if not (stripped.endswith(';') and 'return' in stripped):
            return line
        excess = stripped.count(')') - stripped.count('(')
        if excess <= 0:
            return line
        core = stripped[:-1]  # drop ;
        for _ in range(excess):
            if core.endswith(')'):
                core = core[:-1]
        return core + ';'
    cuda_code = '\n'.join(_fix_math_line(l) for l in cuda_code.splitlines())

    # Bug 2 — extra closing paren after TORCH_CHECK / AT_CHECK macro calls
    # e.g. TORCH_CHECK(x.is_contiguous(), "msg"))  → remove the extra )
    cuda_code = re.sub(
        r'((?:TORCH_CHECK|AT_CHECK)\s*\([^;]+\))\)',
        r'\1',
        cuda_code,
    )

    # Bug 3 — .ptr<T>() is not a PyTorch C++ API method; correct is .data_ptr<T>()
    cuda_code = re.sub(r'\.ptr\s*<', '.data_ptr<', cuda_code)

    # Bug 4 — tensor.type() returns DeprecatedTypeProperties, not ScalarType.
    # AT_DISPATCH_FLOATING_TYPES and similar macros need .scalar_type().
    cuda_code = re.sub(r'\b(\w+)\.type\(\)', r'\1.scalar_type()', cuda_code)

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

    # Find all binding function signatures using balanced-paren matching.
    # This handles multi-line signatures and nested parens in default args
    # (e.g. stride=std::vector<int64_t>(), n=static_cast<int>(x.size(0))).
    def _find_binding_functions(code: str):
        """
        Scan for torch::Tensor / std::vector<torch::Tensor> / std::tuple<...>
        binding functions. Uses balanced-paren matching so nested parens in
        argument lists (default values, casts) don't truncate the signature.
        Returns list of (declaration_str, func_name).
        """
        found = []
        ret_pat = re.compile(
            r'(?:std::(?:vector|tuple)\s*<[^>]+>\s*|torch::Tensor\s+)'
            r'(\w+)\s*\('
        )
        for m in ret_pat.finditer(code):
            func_name = m.group(1)
            # Skip CUDA qualifiers / well-known non-binding names
            if func_name in ('__global__', '__device__', '__host__',
                             'TORCH_CHECK', 'AT_CHECK', 'AT_ASSERTM',
                             'torch', 'at', 'std'):
                continue
            paren_start = m.end() - 1  # points at the opening '('
            depth, i = 0, paren_start
            while i < len(code):
                if code[i] == '(':
                    depth += 1
                elif code[i] == ')':
                    depth -= 1
                    if depth == 0:
                        # Only accept definitions '{' or declarations ';'
                        rest = code[i + 1:i + 80].lstrip()
                        if rest[:1] in ('{', ';') or re.match(r'(?:const|noexcept|override)\s*[{;]', rest):
                            decl = code[m.start():i + 1].strip()
                            found.append((decl, func_name))
                        break
                i += 1
        return found

    sig_matches = _find_binding_functions(cuda_code)

    if not sig_matches:
        print(f"[WRAPPER DEBUG] No torch::Tensor binding found. First 300 chars of cuda_code:\n{cuda_code[:300]}")
        return None

    func_signatures = [m[0] for m in sig_matches]
    func_names = [m[1] for m in sig_matches]
    print(f"[WRAPPER DEBUG] Found {len(func_names)} binding(s): {func_names}")

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

    print(f"[WRAPPER DEBUG] fwd_raw={repr(fwd_raw[:120])} → fwd_args_clean={repr(fwd_args_clean)}")
    print(f"[WRAPPER DEBUG] init_raw={repr(init_raw[:120])} → init_args_clean={repr(init_args_clean)}")

    # Use the last binding function as the one to call from forward()
    binding_func = func_names[-1]

    # 3. Parse binding function arg names to detect stateful models.
    # Use balanced-paren extraction on the last signature to handle nested parens
    # in default values (e.g. stride=std::vector<int64_t>()).
    last_sig = func_signatures[-1]
    binding_arg_names = []
    paren_m = re.search(r'\w+\s*\(', last_sig)
    if paren_m:
        ps = paren_m.end() - 1
        depth, i = 0, ps
        while i < len(last_sig):
            if last_sig[i] == '(':  depth += 1
            elif last_sig[i] == ')':
                depth -= 1
                if depth == 0:
                    raw_params = last_sig[ps + 1:i]
                    for param in _split_args(raw_params):
                        words = re.findall(r'\b\w+\b', param.split('=')[0].strip())
                        if words:
                            binding_arg_names.append(words[-1])
                    break
            i += 1

    forward_args_list = [a.strip() for a in fwd_args_clean.split(',') if a.strip()]
    forward_set = set(forward_args_list)
    extra_args = [a for a in binding_arg_names if a not in forward_set]

    # 4. Fix common PyTorch C++ API mismatches before compilation
    cuda_code = _fix_cuda_api(cuda_code)

    cuda_source_expr = repr(cuda_code)
    cpp_source_expr  = repr(cpp_source)

    # 5. Generate unique module name
    code_hash = hashlib.md5((cuda_code + cpp_source).encode("utf-8")).hexdigest()[:12]
    import random as _random
    mod_name = f"kf_ext_{code_hash}_{_random.randint(0, 0xFFFFFF):06x}"

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
        # Strip _obj suffix (SakanaAI naming: weight_obj, bias_obj, x_obj, ...)
        effective = arg[:-4] if arg.endswith('_obj') else arg

        # If effective name is a forward arg, pass it through directly (not self.*)
        if effective in forward_set:
            return effective

        # Direct match: self.bias, self.weight, etc.
        if re.search(rf'\bself\.{re.escape(effective)}\b', init_body):
            return f'self.{effective}'

        # Plural → singular: weights→weight, biases→bias
        if effective.endswith('s'):
            singular = effective[:-1]
            if re.search(rf'\bself\.{re.escape(singular)}\b', init_body):
                return f'self.{singular}'

        # nn.Sequential / nn.ModuleList — weights/biases live inside child layers.
        seq_match = re.search(
            r'\bself\.(\w+)\s*=\s*nn\.(?:Sequential|ModuleList)\b', init_body
        )
        if seq_match:
            container = f'self.{seq_match.group(1)}'
            if effective in ('weights', 'weight'):
                return f'[l.weight for l in {container} if hasattr(l, "weight")]'
            if effective in ('biases', 'bias'):
                return f'[l.bias for l in {container} if hasattr(l, "bias")]'

        # Find first nn.Module attribute (Conv, Linear, BN, etc.) to pull attrs from.
        nn_mod = re.search(r'\bself\.(\w+)\s*=\s*nn\.', init_body)
        mod_name = nn_mod.group(1) if nn_mod else None

        # Tensor attributes on the nn module (weight, bias, running_mean, ...)
        tensor_attrs = ('weight', 'bias', 'running_mean', 'running_var',
                        'weight_g', 'weight_v', 'scale', 'gamma', 'beta')
        if effective in tensor_attrs and mod_name:
            return f'self.{mod_name}.{effective}'

        # 'input' / 'inp' are common C++ names for the primary input tensor — pass through
        if effective in ('input', 'inp', 'in_tensor') and forward_args_list:
            return forward_args_list[0]

        # Scalar/tuple attributes stored in nn module.
        # stride/padding/dilation/kernel_size are tuples → index [0].
        # groups is a plain int → no index.
        tuple_attrs = ('stride', 'padding', 'dilation', 'output_padding', 'kernel_size')
        if effective in tuple_attrs and mod_name:
            return f'self.{mod_name}.{effective}[0]'
        if effective == 'groups' and mod_name:
            return f'self.{mod_name}.groups'

        # Decomposed 2D/3D params: stride_h→stride[0], stride_w→stride[1], stride_d→stride[2]
        dim_map = {'_h': 0, '_w': 1, '_d': 2}
        for suffix, idx in dim_map.items():
            if effective.endswith(suffix):
                base_attr = effective[:-len(suffix)]
                if base_attr in tuple_attrs and mod_name:
                    return f'self.{mod_name}.{base_attr}[{idx}]'
                if base_attr == 'kernel_size' and mod_name:
                    return f'self.{mod_name}.kernel_size[{idx}]'

        # Pattern: conv_weight -> self.conv.weight, bn_running_mean -> self.bn.running_mean
        known_attrs = tensor_attrs
        for attr in known_attrs:
            if effective.endswith('_' + attr):
                module = effective[:-len(attr) - 1]
                if re.search(rf'\bself\.{re.escape(module)}\b', init_body):
                    return f'self.{module}.{attr}'

        return f'self.{effective}'

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
    model_new_start = wrapper.find("class ModelNew")
    print(f"[WRAPPER DEBUG] Generated ModelNew:\n{wrapper[model_new_start:]}")
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
