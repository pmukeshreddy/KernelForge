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


def build_load_inline_wrapper(cuda_code: str, ref_code: str) -> str:
    """
    Wrap raw CUDA C++ code in a full load_inline Python script.
    
    Parses the C++ to extract binding function signatures, and parses the
    reference PyTorch code to get forward() arguments, then generates the
    complete Python wrapper with ModelNew class.
    """
    # 1. Find all torch::Tensor binding function signatures (definitions, not declarations)
    sig_pattern = r'(torch::Tensor\s+(\w+)\s*\([^)]*\))\s*\{'
    sig_matches = re.findall(sig_pattern, cuda_code)
    
    if not sig_matches:
        # Fallback: try to find any function returning torch::Tensor
        sig_pattern_decl = r'(torch::Tensor\s+(\w+)\s*\([^)]*\))\s*;'
        sig_matches = re.findall(sig_pattern_decl, cuda_code)
    
    if not sig_matches:
        return None  # Can't parse - signal extraction failure
    
    # sig_matches is [(full_sig, func_name), ...]
    func_signatures = [m[0] for m in sig_matches]
    func_names = [m[1] for m in sig_matches]
    
    # Build cpp_source from signatures
    cpp_source = "; ".join(func_signatures) + ";"
    
    # 2. Parse reference code's Model.forward() and __init__() arg names
    fwd_match = re.search(r'def forward\(self,\s*(.*?)\)', ref_code)
    fwd_args = fwd_match.group(1).strip() if fwd_match else "x"
    fwd_args_clean = ", ".join(
        arg.split(":")[0].strip() for arg in fwd_args.split(",")
    )

    # Mirror Model.__init__ args so get_init_inputs() can construct ModelNew
    init_match = re.search(r'def __init__\(self,\s*(.*?)\)', ref_code, re.DOTALL)
    raw_init = init_match.group(1).strip() if (init_match and init_match.group(1).strip()) else ""
    init_args_clean = ", ".join(
        arg.split(":")[0].strip() for arg in raw_init.split(",") if arg.strip()
    ) if raw_init else ""

    # Use the last binding function as the one to call from forward()
    binding_func = func_names[-1]
    
    # 3. Fix common API version mismatches before compilation

    # getCurrentCUDAStream: safest fix is to drop the stream arg from kernel launches
    # so the kernel runs on the default CUDA stream (always works, zero overhead).
    # Covers: <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>
    #     and <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>
    cuda_code = re.sub(
        r',\s*(?:at|c10)::cuda::getCurrentCUDAStream\([^)]*\)\s*(>>>)',
        r'\1',
        cuda_code,
    )
    # If the stream call appears elsewhere (e.g., stored in a variable), add include
    if 'getCurrentCUDAStream' in cuda_code:
        if '#include <ATen/cuda/CUDAContext.h>' not in cuda_code:
            cuda_code = '#include <ATen/cuda/CUDAContext.h>\n' + cuda_code
        cuda_code = re.sub(
            r'(?:at|c10)::cuda::getCurrentCUDAStream\(\)',
            'at::cuda::getCurrentCUDAStream()',
            cuda_code,
        )

    # __fabsf / __fabs are host-only in modern CUDA; device code must use fabsf / fabs
    cuda_code = cuda_code.replace('__fabsf(', 'fabsf(')
    cuda_code = cuda_code.replace('__fabs(', 'fabs(')

    # Escape CUDA code for safe embedding in Python triple-quoted string
    safe_cuda = cuda_code.replace('\\', '\\\\').replace('"""', "'''")
    # Escape cpp_source for single-quoted Python string
    safe_cpp = cpp_source.replace('\\', '\\\\').replace('"', '\\"')
    
    # 4. Build the full Python wrapper
    wrapper = f'''import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """{safe_cuda}"""

cpp_source = "{safe_cpp}"

ext = load_inline(
    name="custom_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions={func_names},
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(torch.nn.Module):
    def __init__(self{", " + init_args_clean if init_args_clean else ""}):
        super().__init__()
    def forward(self, {fwd_args_clean}):
        return ext.{binding_func}({fwd_args_clean})
'''
    return wrapper


class KernelForgeAgent:
    def __init__(self, model_name: str = "mukeshreddy/kernelforge-sft-qwen3-8b", mock_mode: bool = False):
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

    def extract_cuda_code(self, response: str) -> str:
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
            cuda_code = self.extract_cuda_code(full_response)
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
