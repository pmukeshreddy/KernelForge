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

    # Deprecated warp shuffle intrinsics → _sync variants (required on sm_70+, H100)
    # __shfl_down(val, delta) → __shfl_down_sync(0xffffffff, val, delta)
    for shfl in ('__shfl_down', '__shfl_up', '__shfl_xor', '__shfl'):
        sync_ver = f'{shfl}_sync'
        if shfl in cuda_code and sync_ver not in cuda_code:
            cuda_code = re.sub(
                rf'{re.escape(shfl)}\s*\(',
                f'{sync_ver}(0xffffffff, ',
                cuda_code,
            )

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

    # Bug 2 — extra closing paren after TORCH_CHECK / AT_CHECK macro calls.
    # Use balanced-paren scanner so we never cross into adjacent calls.
    def _fix_torch_check_parens(code: str) -> str:
        out, i = [], 0
        for macro in ('TORCH_CHECK', 'AT_CHECK'):
            pass  # processed in loop below
        pat = re.compile(r'(?:TORCH_CHECK|AT_CHECK)\s*\(')
        while i < len(code):
            m = pat.search(code, i)
            if not m:
                out.append(code[i:]); break
            out.append(code[i:m.start()])
            depth, j = 0, m.start()
            while j < len(code):
                if code[j] == '(':   depth += 1
                elif code[j] == ')':
                    depth -= 1
                    if depth == 0:
                        # Include the matching ')'
                        out.append(code[m.start():j + 1])
                        j += 1
                        # Skip exactly one spurious extra ')'
                        while j < len(code) and code[j] in ' \t':
                            j += 1
                        if j < len(code) and code[j] == ')':
                            j += 1  # drop it
                        i = j
                        break
                j += 1
            else:
                out.append(code[m.start():]); i = len(code)
        return ''.join(out)
    cuda_code = _fix_torch_check_parens(cuda_code)

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

    # Bug 5 — Output tensors allocated on CPU instead of CUDA.
    # Model writes: torch::empty({...}, torch::kInt32)  → creates on CPU!
    # Fix: torch::empty({...}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA))
    # Process line-by-line so we only touch lines with torch::empty/zeros/ones/full.
    _DTYPES = (
        'kFloat', 'kFloat32', 'kFloat64', 'kDouble', 'kHalf', 'kBFloat16',
        'kInt', 'kInt32', 'kInt64', 'kLong', 'kByte', 'kBool', 'kShort',
    )
    _dtype_pat = '|'.join(_DTYPES)
    _alloc_line_pat = re.compile(r'torch::(?:empty|zeros|ones|full)\s*\(')
    _bare_dtype_pat = re.compile(
        rf',\s*(torch::(?:{_dtype_pat}))\s*\)'
    )
    def _fix_cpu_line(line: str) -> str:
        if not _alloc_line_pat.search(line):
            return line
        # Don't touch lines that already have device() or TensorOptions
        if 'device(' in line or 'TensorOptions' in line or '.options()' in line:
            return line
        return _bare_dtype_pat.sub(
            r', torch::TensorOptions().dtype(\1).device(torch::kCUDA))',
            line,
        )
    cuda_code = '\n'.join(_fix_cpu_line(l) for l in cuda_code.splitlines())

    # Bug 6 — torch::empty/zeros/ones with NO options at all → CPU float32.
    # Detect: torch::empty({...}) where ) immediately closes the call after }.
    # Fix: append .to(input.device()) — but we don't know input name, so
    # instead use device(torch::kCUDA) since this is always a CUDA extension.
    def _fix_no_options_alloc(line: str) -> str:
        if not _alloc_line_pat.search(line):
            return line
        if 'device(' in line or 'TensorOptions' in line or '.options()' in line:
            return line
        # Match torch::empty({...}) — closing } immediately followed by )
        return re.sub(
            r'(torch::(?:empty|zeros|ones)\s*\(\s*\{[^}]*\})\s*\)',
            r'\1, torch::TensorOptions().device(torch::kCUDA))',
            line,
        )
    cuda_code = '\n'.join(_fix_no_options_alloc(l) for l in cuda_code.splitlines())

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
    Wrap raw CUDA C++ code in a Python script that compiles via load().

    Strategy:
      - Write the .cu file to disk unchanged (PYBIND11_MODULE preserved).
      - Compile with torch.utils.cpp_extension.load() — no cpp_sources to
        generate, no load_inline declaration hacks.  The PYBIND11_MODULE the
        model/SakanaAI wrote IS the binding; no parsing needed for compilation.
      - Parse PYBIND11_MODULE only to learn which function to call from
        ModelNew.forward() and what typed args it takes.
    """

    # ── Step 1: Parse PYBIND11_MODULE for exported function names ────────────
    pybind_exports = []   # [(python_name, cpp_func_name)]
    pyb_m = re.search(r'PYBIND11_MODULE\s*\(', cuda_code)
    if pyb_m:
        try:
            brace_pos = cuda_code.index('{', pyb_m.end())
        except ValueError:
            brace_pos = -1
        if brace_pos >= 0:
            depth, i = 0, brace_pos
            while i < len(cuda_code):
                if cuda_code[i] == '{':
                    depth += 1
                elif cuda_code[i] == '}':
                    depth -= 1
                    if depth == 0:
                        block = cuda_code[brace_pos:i + 1]
                        for dm in re.finditer(
                            r'm\.def\s*\(\s*"(\w+)"\s*,\s*&\s*(\w+)', block
                        ):
                            pybind_exports.append((dm.group(1), dm.group(2)))
                        break
                i += 1

    if not pybind_exports:
        print(f"[WRAPPER DEBUG] No PYBIND11_MODULE found. First 300 chars:\n{cuda_code[:300]}")
        return None

    binding_func = pybind_exports[-1][1]
    print(f"[WRAPPER DEBUG] PYBIND11_MODULE exports: {pybind_exports}, calling: {binding_func}")

    # ── Step 2: Fix CUDA API issues ───────────────────────────────────────────
    cuda_code = _fix_cuda_api(cuda_code)

    # ── Step 3: Parse ref_code for forward() / __init__() signatures ─────────
    def _extract_def_args(code: str, def_name: str) -> str:
        m = re.search(rf'def {def_name}\s*\(', code)
        if not m:
            return ""
        start = m.end() - 1
        depth, i = 0, start
        while i < len(code):
            if code[i] == '(':
                depth += 1
            elif code[i] == ')':
                depth -= 1
                if depth == 0:
                    inner = code[start + 1:i]
                    inner = re.sub(r'^\s*self\s*,?\s*', '', inner)
                    return inner.strip()
            i += 1
        return ""

    fwd_raw = _extract_def_args(ref_code, "forward")
    fwd_args_clean = ", ".join(
        a.split(":")[0].split("=")[0].strip()
        for a in _split_args(fwd_raw) if a.strip()
    ) if fwd_raw else "x"

    init_raw = _extract_def_args(ref_code, "__init__")
    init_args_clean = ", ".join(
        a.split(":")[0].split("=")[0].strip()
        for a in _split_args(init_raw) if a.strip()
    ) if init_raw else ""

    forward_args_list = [a.strip() for a in fwd_args_clean.split(',') if a.strip()]
    forward_set       = set(forward_args_list)
    print(f"[WRAPPER DEBUG] forward args: {repr(fwd_args_clean)}")

    # ── Step 4: Parse typed args from the binding function signature ──────────
    cuda_code_norm = re.sub(r'\bat::Tensor\b', 'torch::Tensor', cuda_code)
    cuda_code_norm = re.sub(r'(?<!:)\bTensor\b', 'torch::Tensor', cuda_code_norm)

    typed_args = []
    sig_pat = re.compile(
        r'(?:std::(?:vector|tuple)\s*<[^>]*torch::Tensor[^>]*>\s*'
        r'|torch::Tensor\s+)'
        + rf'{re.escape(binding_func)}\s*\('
    )
    sig_m = sig_pat.search(cuda_code_norm)
    if sig_m:
        ps, depth, i = sig_m.end() - 1, 0, sig_m.end() - 1
        while i < len(cuda_code_norm):
            if cuda_code_norm[i] == '(':
                depth += 1
            elif cuda_code_norm[i] == ')':
                depth -= 1
                if depth == 0:
                    for param in _split_args(cuda_code_norm[ps + 1:i]):
                        param = param.split('=')[0].strip()
                        toks  = re.findall(r'\b\w[\w:]*\b', param)
                        if not toks:
                            continue
                        name = toks[-1]
                        if name in ('const', 'int', 'float', 'double', 'bool', 'void',
                                    'int64_t', 'size_t', 'uint32_t', 'int32_t',
                                    'unsigned', 'long', 'short', 'auto'):
                            continue
                        type_part = param[:param.rfind(name)].strip().rstrip('*& ')
                        typed_args.append((type_part, name))
                    break
            i += 1

    # ── Step 5: Build __init__ body and arg resolver ──────────────────────────
    def _extract_init_body(code: str) -> str:
        m = re.search(r'^([ \t]*)def __init__\s*\(', code, re.MULTILINE)
        if not m:
            return ""
        method_indent = m.group(1)
        body_indent   = method_indent + "    "
        pos = m.end() - 1
        depth, i = 0, pos
        while i < len(code):
            if code[i] == '(':
                depth += 1
            elif code[i] == ')':
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        colon = code.index(':', i)
        body_lines = []
        for line in code[colon + 1:].split('\n'):
            if not line.strip():
                body_lines.append('')
                continue
            if re.match(rf'^{re.escape(method_indent)}(?:def|class)\s', line):
                break
            if 'super(' in line and '__init__' in line:
                continue
            if line.startswith(body_indent):
                body_lines.append('        ' + line[len(body_indent):])
            else:
                body_lines.append('        ' + line.lstrip())
        return '\n'.join(body_lines)

    def _resolve_arg(type_str: str, name: str, init_body: str) -> str:
        t   = type_str.lower().replace(' ', '')
        eff = name
        for sfx in ('_obj', '_opt', '_tensor'):
            if eff.endswith(sfx):
                eff = eff[:-len(sfx)]
                break

        if 'cudastream' in t:
            return '0'
        if eff in ('nullptr', 'null', 'none'):
            return 'None'

        is_tensor = 'torch::tensor' in t or 'at::tensor' in t
        canonical = forward_args_list[0] if forward_args_list else 'x'

        if is_tensor and eff in forward_set:
            return eff
        if is_tensor and eff in ('input', 'inp', 'in_tensor'):
            return canonical
        if is_tensor and eff in ('output', 'out', 'result', 'out_tensor', 'output_tensor'):
            return f'torch.empty_like({canonical})'

        if re.match(r'^num_(streams?|threads?|blocks?|warps?)$', eff):
            dm = re.search(rf'\b{re.escape(name)}\s*=\s*(\d+)', cuda_code_norm)
            return dm.group(1) if dm else '4'
        if eff.startswith('log_') and eff[4:] in forward_set:
            return f'torch.log({eff[4:]})'

        if re.search(rf'\bself\.{re.escape(eff)}\b', init_body):
            return f'self.{eff}'
        if eff.endswith('s') and re.search(rf'\bself\.{re.escape(eff[:-1])}\b', init_body):
            return f'self.{eff[:-1]}'

        seq = re.search(r'\bself\.(\w+)\s*=\s*nn\.(?:Sequential|ModuleList)\b', init_body)
        if seq:
            container = f'self.{seq.group(1)}'
            if eff in ('weights', 'weight'):
                return f'[l.weight for l in {container} if hasattr(l, "weight")]'
            if eff in ('biases', 'bias'):
                return f'[l.bias for l in {container} if hasattr(l, "bias")]'

        nn_m = re.search(r'\bself\.(\w+)\s*=\s*nn\.', init_body)
        mod  = nn_m.group(1) if nn_m else None

        TENSOR_ATTRS = ('weight', 'bias', 'running_mean', 'running_var',
                        'weight_g', 'weight_v', 'scale', 'gamma', 'beta')
        TUPLE_ATTRS  = ('stride', 'padding', 'dilation', 'output_padding', 'kernel_size')
        SCALAR_ATTRS = ('eps', 'momentum', 'num_features', 'num_groups',
                        'in_features', 'out_features', 'num_heads', 'p', 'groups')

        if mod:
            if eff in TENSOR_ATTRS:
                return f'self.{mod}.{eff}'
            if eff in TUPLE_ATTRS:
                return f'_si(self.{mod}.{eff})'
            if eff in SCALAR_ATTRS:
                return f'self.{mod}.{eff}'
            for sfx, idx in (('_h', 0), ('_w', 1), ('_d', 2)):
                if eff.endswith(sfx) and eff[:-len(sfx)] in TUPLE_ATTRS:
                    return f'_si(self.{mod}.{eff[:-len(sfx)]}, {idx})'
            for attr in TENSOR_ATTRS:
                if eff.endswith('_' + attr):
                    mod2 = eff[:-len(attr) - 1]
                    if re.search(rf'\bself\.{re.escape(mod2)}\b', init_body):
                        return f'self.{mod2}.{attr}'

        return f'self.{eff}'

    # ── Step 6: Build call args ───────────────────────────────────────────────
    init_body = _extract_init_body(ref_code)
    call_args = [
        name if name in forward_set else _resolve_arg(t, name, init_body)
        for t, name in typed_args
    ]
    has_extra = any(n not in forward_set for _, n in typed_args)

    print(f"[WRAPPER DEBUG] typed_args: {[(t[:30], n) for t, n in typed_args]}")
    print(f"[WRAPPER DEBUG] call_args:  {call_args}")

    # ── Step 7: Emit wrapper — write .cu to disk, compile with load() ─────────
    code_hash        = hashlib.md5(cuda_code.encode()).hexdigest()[:12]
    import random as _random
    mod_name         = f"kf_ext_{code_hash}_{_random.randint(0, 0xFFFFFF):06x}"
    cuda_source_expr = repr(cuda_code)
    init_sig         = (", " + init_args_clean) if init_args_clean else ""
    call_str         = ", ".join(call_args)

    ext_block = f'''import torch
import torch.nn as nn
import os
from torch.utils.cpp_extension import load

def _si(v, i=0):
    return v[i] if isinstance(v, (tuple, list)) else v

_cuda_source = {cuda_source_expr}
_cu_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".kf_cu_cache")
os.makedirs(_cu_dir, exist_ok=True)
_cu_path = os.path.join(_cu_dir, "{mod_name}.cu")
with open(_cu_path, "w") as _f:
    _f.write(_cuda_source)

ext = load(
    name="{mod_name}",
    sources=[_cu_path],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)
'''

    if has_extra:
        model_block = f'''
class ModelNew(torch.nn.Module):
    def __init__(self{init_sig}):
        super().__init__()
{init_body}
    def forward(self, {fwd_args_clean}):
        return ext.{binding_func}({call_str})
'''
    else:
        model_block = f'''
class ModelNew(torch.nn.Module):
    def __init__(self{init_sig}):
        super().__init__()
    def forward(self, {fwd_args_clean}):
        return ext.{binding_func}({call_str})
'''

    wrapper = ext_block + model_block
    print(f"[WRAPPER DEBUG] ModelNew:\n{wrapper[wrapper.find('class ModelNew'):]}")
    return wrapper

def _extract_python_code(response: str) -> str:
    """Extract the Python model_new.py block from the LLM's response."""
    # Strip <think> blocks first
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"<think>.*", "", response, flags=re.DOTALL)
    m = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Handle unclosed block
    m = re.search(r"```python\s*(.*)", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


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
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Detect if model_name is a LoRA adapter (has adapter_config.json on HF)
            adapter_path = None
            base_model_name = model_name
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo_id=model_name, filename="adapter_config.json")
                # If we get here, it's a LoRA adapter — read base model from config
                import json
                from huggingface_hub import hf_hub_download
                cfg_path = hf_hub_download(repo_id=model_name, filename="adapter_config.json")
                with open(cfg_path) as f:
                    adapter_cfg = json.load(f)
                base_model_name = adapter_cfg.get("base_model_name_or_path", model_name)
                adapter_path = model_name
                print(f"Detected LoRA adapter. Base model: {base_model_name}")
            except Exception:
                pass  # Not a LoRA adapter, load as full model

            print(f"Loading base model: {base_model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )

            if adapter_path:
                print(f"Loading LoRA adapter: {adapter_path}...")
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter_path)

            self.model.eval()
            print("Model ready.")

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM to generate the next response based on conversation history."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Pre-fill with python block to match SFT training format
        prefill = "```python\n"
        text += prefill

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.7,
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
                "content": (
                    "Write the complete model_new.py for the following operation.\n\n"
                    "Output EXACTLY ONE ```python code block containing the complete model_new.py "
                    "with load_inline and a ModelNew(nn.Module) class.\n\n"
                    f"Reference Program:\n```python\n{target_program}\n```"
                )
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
            full_response = "```python\n" + response
            messages.append({"role": "assistant", "content": full_response})
            
            # 2. Extract Python model_new.py block (matches SFT training format)
            candidate_code = _extract_python_code(full_response)
            if not candidate_code or "ModelNew" not in candidate_code:
                print("❌ Failed to extract Python model_new.py block. Requesting fix...")
                print(f"--- FAILED GENERATED TEXT ---\n{response[:500]}\n-----------------------------")
                messages.append({"role": "user", "content": "Error: Could not find a ```python code block with a ModelNew class in your response. Output EXACTLY ONE ```python block containing the complete model_new.py file with load_inline and a ModelNew(nn.Module) class."})
                continue
                
            # 4. Sandbox Evaluation
            print("🛠️  Compiling and Evaluating in Sandbox...")
            eval_result = evaluate(candidate_code, target_program)
            
            if not eval_result["correct"]:
                # Compilation failed or output was wrong
                error_msg = eval_result.get("compiler_error") or "Outputs do not match the reference implementation exactly (Correctness Failed)."
                print(f"❌ Evaluation Failed: {error_msg.strip()[:100]}...\n")
                print(f"--- FAILED GENERATED CODE ---\n{candidate_code}\n-----------------------------")
                
                # Feed error back to LLM (show the C++ error, not Python wrapper errors)
                feedback = f"Your previous answer was incorrect. Here is the error message:\n{error_msg}\n\nRestart your reasoning process and generate new, complete code."
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
                feedback = f"Your previous answer was correct at {reward:.2f}x speedup over PyTorch.\n\nHere is the hardware profiling report:\n{profiler_feedback}\n\nRestart your reasoning process and generate new, complete code."
                messages.append({"role": "user", "content": feedback})

        print(f"\n🏁 Optimization Completed. Best Reward: {best_reward:.2f}x")
        return best_code, best_reward
