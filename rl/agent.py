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

    Strategy:
      1. Parse PYBIND11_MODULE BEFORE stripping it — m.def() gives us the
         exact C++ function names SakanaAI already verified work.
      2. Strip the block (load_inline generates its own pybind11 bindings).
      3. Find the signature of each exported function by name to extract
         typed arg lists.
      4. Type-aware argument mapping: torch::Tensor args vs scalars vs streams
         are handled differently rather than guessing from names alone.
    """

    # ── Step 1: Extract exports from PYBIND11_MODULE before stripping ────────
    # m.def("python_name", &cpp_func, ...) → we want cpp_func names (exact)
    pybind_exports = []   # ordered list of cpp function names
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
                            pybind_exports.append(dm.group(2))  # C++ func name
                        break
                i += 1

    # ── Step 2: Strip PYBIND11_MODULE block ──────────────────────────────────
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

    # ── Step 3: Normalise tensor type spellings ───────────────────────────────
    cuda_code = re.sub(r'\bat::Tensor\b', 'torch::Tensor', cuda_code)
    cuda_code = re.sub(r'(?<!:)\bTensor\b', 'torch::Tensor', cuda_code)

    # ── Step 4: Find function signatures ─────────────────────────────────────
    # Helper: extract a balanced-paren signature given the opening '(' position
    def _extract_sig(code: str, paren_start: int) -> str | None:
        depth, i = 0, paren_start
        while i < len(code):
            if code[i] == '(':
                depth += 1
            elif code[i] == ')':
                depth -= 1
                if depth == 0:
                    rest = code[i + 1:i + 80].lstrip()
                    if rest[:1] in ('{', ';') or re.match(
                        r'(?:const|noexcept|override)\s*[{;]', rest
                    ):
                        return code[paren_start:i + 1]
                    return None
            i += 1
        return None

    # Find the start of a declaration for func_name: walk back from the '('
    # to pick up the return type (everything since last ';' / '}' / newline).
    def _find_sig_by_name(code: str, func_name: str):
        pat = re.compile(rf'\b{re.escape(func_name)}\s*\(')
        for m in pat.finditer(code):
            paren_start = m.end() - 1
            sig_inner = _extract_sig(code, paren_start)
            if sig_inner is None:
                continue
            # Walk back from m.start() to find the start of the return type.
            # Stop at ';' or '}' only — NOT '\n', so multi-line return types like
            #   std::vector<torch::Tensor>\nforward(...) are captured correctly.
            k = m.start() - 1
            while k >= 0 and code[k] not in (';', '}'):
                k -= 1
            decl_start = k + 1
            decl = code[decl_start:m.end() - 1 + len(sig_inner)].strip()
            # Skip CUDA kernel definitions (__global__ / __device__ functions)
            if re.match(r'__(?:global|device|host)__', decl):
                continue
            return decl
        return None

    if pybind_exports:
        # Use the exact names SakanaAI already verified
        sig_matches = []
        for fn in pybind_exports:
            decl = _find_sig_by_name(cuda_code, fn)
            if decl:
                sig_matches.append((decl, fn))
        if not sig_matches:
            # Names found in PYBIND11_MODULE but definitions not found — fall through
            pybind_exports = []

    if not pybind_exports:
        # Fallback: scan for torch::Tensor return-type functions
        def _scan_tensor_returns(code: str):
            found = []
            ret_pat = re.compile(
                r'(?:std::(?:vector|tuple)\s*<[^>]*(?:torch|at)::Tensor[^>]*>\s*'
                r'|torch::Tensor\s+)'
                r'(\w+)\s*\('
            )
            skip_names = {'__global__', '__device__', '__host__',
                          'TORCH_CHECK', 'AT_CHECK', 'AT_ASSERTM', 'torch', 'at', 'std'}
            for m in ret_pat.finditer(code):
                fn = m.group(1)
                if fn in skip_names:
                    continue
                sig_inner = _extract_sig(code, m.end() - 1)
                if sig_inner:
                    decl = code[m.start():m.end() - 1 + len(sig_inner)].strip()
                    found.append((decl, fn))
            return found
        sig_matches = _scan_tensor_returns(cuda_code)

    if not sig_matches:
        print(f"[WRAPPER DEBUG] No binding found. First 300 chars:\n{cuda_code[:300]}")
        return None

    func_signatures = [s[0] for s in sig_matches]
    func_names      = [s[1] for s in sig_matches]
    print(f"[WRAPPER DEBUG] Bindings: {func_names} (via {'pybind11 exports' if pybind_exports else 'return-type scan'})")

    cpp_source = "; ".join(func_signatures) + ";"

    # ── Step 5: Parse ref_code for forward() / __init__() signatures ─────────
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

    print(f"[WRAPPER DEBUG] forward args: {repr(fwd_args_clean)}")

    forward_args_list = [a.strip() for a in fwd_args_clean.split(',') if a.strip()]
    forward_set       = set(forward_args_list)
    binding_func      = func_names[-1]

    # ── Step 6: Parse typed args from the chosen binding signature ───────────
    # Returns list of (type_str, name_str) for the last (primary) binding.
    last_sig = func_signatures[-1]
    typed_args = []   # [(type_str, name_str)]
    pm = re.search(r'\w+\s*\(', last_sig)
    if pm:
        ps = pm.end() - 1
        depth, i = 0, ps
        while i < len(last_sig):
            if last_sig[i] == '(':
                depth += 1
            elif last_sig[i] == ')':
                depth -= 1
                if depth == 0:
                    for param in _split_args(last_sig[ps + 1:i]):
                        param = param.split('=')[0].strip()
                        toks  = re.findall(r'\b\w[\w:]*\b', param)
                        if not toks:
                            continue
                        name = toks[-1]
                        if name in ('const', 'int', 'float', 'double', 'bool', 'void',
                                    'int64_t', 'size_t', 'uint32_t', 'int32_t',
                                    'unsigned', 'long', 'short', 'auto'):
                            continue
                        idx       = param.rfind(name)
                        type_part = param[:idx].strip().rstrip('*& ')
                        typed_args.append((type_part, name))
                    break
            i += 1

    # ── Step 7: Fix CUDA API before compilation ───────────────────────────────
    cuda_code = _fix_cuda_api(cuda_code)

    # ── Step 8: Type-aware argument resolver ──────────────────────────────────
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
        """Map one C++ arg (with its type) to a Python expression."""
        t = type_str.lower().replace(' ', '')

        # ── Type: CUDA stream → default stream (0) ───────────────────────────
        if 'cudastream' in t:
            return '0'

        # ── Type: null / optional → None ────────────────────────────────────
        if name in ('nullptr', 'null', 'none'):
            return 'None'

        # ── Determine if this is a tensor type ───────────────────────────────
        is_tensor = 'torch::tensor' in t or 'at::tensor' in t

        # Strip decorative suffixes to get the semantic name
        eff = name
        for sfx in ('_obj', '_opt', '_tensor'):
            if eff.endswith(sfx):
                eff = eff[:-len(sfx)]
                break

        # ── Tensor: forward arg passthrough ──────────────────────────────────
        # 'input'/'inp' are conventional C++ aliases for the first forward arg.
        canonical = forward_args_list[0] if forward_args_list else 'x'
        if is_tensor and eff in forward_set:
            return eff
        if is_tensor and eff in ('input', 'inp', 'in_tensor'):
            return canonical

        # ── Tensor: output buffer → allocate from first forward arg ──────────
        OUTPUT_NAMES = {'output', 'out', 'result', 'out_tensor', 'output_tensor'}
        if is_tensor and eff in OUTPUT_NAMES:
            return f'torch.empty_like({canonical})'

        # ── Scalar: CUDA impl details that should never be self.* ─────────────
        if re.match(r'^num_(streams?|threads?|blocks?|warps?)$', eff):
            m = re.search(rf'\b{re.escape(name)}\s*=\s*(\d+)', last_sig)
            return m.group(1) if m else '4'

        # ── log_X where X is a forward arg ───────────────────────────────────
        if eff.startswith('log_') and eff[4:] in forward_set:
            return f'torch.log({eff[4:]})'

        # ── Attribute lookups against init_body ──────────────────────────────
        # Direct self.eff match
        if re.search(rf'\bself\.{re.escape(eff)}\b', init_body):
            return f'self.{eff}'

        # Plural → singular
        if eff.endswith('s'):
            s = eff[:-1]
            if re.search(rf'\bself\.{re.escape(s)}\b', init_body):
                return f'self.{s}'

        # nn.Sequential / ModuleList → list comprehension for weights/biases
        seq = re.search(r'\bself\.(\w+)\s*=\s*nn\.(?:Sequential|ModuleList)\b', init_body)
        if seq:
            container = f'self.{seq.group(1)}'
            if eff in ('weights', 'weight'):
                return f'[l.weight for l in {container} if hasattr(l, "weight")]'
            if eff in ('biases', 'bias'):
                return f'[l.bias for l in {container} if hasattr(l, "bias")]'

        # First nn module in init for attribute access
        nn_m = re.search(r'\bself\.(\w+)\s*=\s*nn\.', init_body)
        mod  = nn_m.group(1) if nn_m else None

        TENSOR_ATTRS = ('weight', 'bias', 'running_mean', 'running_var',
                        'weight_g', 'weight_v', 'scale', 'gamma', 'beta')
        TUPLE_ATTRS  = ('stride', 'padding', 'dilation', 'output_padding', 'kernel_size')

        if mod:
            # Tensor attribute on nn module
            if eff in TENSOR_ATTRS:
                return f'self.{mod}.{eff}'

            # Scalar/tuple attribute (stride, padding, …) — _si handles int-or-tuple
            if eff in TUPLE_ATTRS:
                return f'_si(self.{mod}.{eff})'
            if eff == 'groups':
                return f'self.{mod}.groups'

            # Decomposed 2D/3D: stride_h → _si(self.conv.stride, 0)
            for sfx, idx in (('_h', 0), ('_w', 1), ('_d', 2)):
                if eff.endswith(sfx):
                    base = eff[:-len(sfx)]
                    if base in TUPLE_ATTRS:
                        return f'_si(self.{mod}.{base}, {idx})'

            # Pattern: conv_weight → self.conv.weight
            for attr in TENSOR_ATTRS:
                if eff.endswith('_' + attr):
                    module = eff[:-len(attr) - 1]
                    if re.search(rf'\bself\.{re.escape(module)}\b', init_body):
                        return f'self.{module}.{attr}'

        # Fallback — best-effort self.eff
        return f'self.{eff}'

    # ── Step 9: Build the call-site argument list ─────────────────────────────
    # Extract init body once (needed for resolving extra args)
    init_body = _extract_init_body(ref_code)

    call_args = []
    for type_str, name in typed_args:
        if name in forward_set:
            call_args.append(name)
        else:
            call_args.append(_resolve_arg(type_str, name, init_body))

    extra_args = [n for _, n in typed_args if n not in forward_set]
    has_extra  = bool(extra_args)

    print(f"[WRAPPER DEBUG] typed_args: {[(t[:30], n) for t, n in typed_args]}")
    print(f"[WRAPPER DEBUG] call_args:  {call_args}")

    # ── Step 10: Emit the wrapper ─────────────────────────────────────────────
    code_hash = hashlib.md5((cuda_code + cpp_source).encode()).hexdigest()[:12]
    import random as _random
    mod_name = f"kf_ext_{code_hash}_{_random.randint(0, 0xFFFFFF):06x}"

    cuda_source_expr = repr(cuda_code)
    cpp_source_expr  = repr(cpp_source)
    init_sig         = (", " + init_args_clean) if init_args_clean else ""
    call_str         = ", ".join(call_args)

    ext_block = f'''import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# _si: safely index a value that may be int or tuple/list
def _si(v, i=0):
    return v[i] if isinstance(v, (tuple, list)) else v

cuda_source = {cuda_source_expr}
cpp_source  = {cpp_source_expr}

ext = load_inline(
    name="{mod_name}",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions={func_names},
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
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
