"""
test_antihack.py - Verification tests for SP3 Anti-Hacking Defenses.
"""
import sys
from antihack import check_security


def run_tests():
    passed = 0
    failed = 0
    
    # 1. Banned Import: subprocess
    code_subprocess = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import subprocess

class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""

    # 2. Banned ImportFrom: os.system
    code_os = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from os import system

class ModelNew(nn.Module):
    def forward(self, x): pass
"""

    # 3. Banned Function Call: torch.nn.functional
    code_fn_call = """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def forward(self, x):
        return F.relu(x)
"""

    # 4. Banned Function Call: Direct torch.nn.functional
    code_fn_direct = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def forward(self, x):
        return torch.nn.functional.relu(x)
"""

    # 5. Missing load_inline
    code_no_inline = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""

    # 6. Legitimate Kernel (Should Pass)
    code_legit = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = "kernel code here"
cpp_source = "cpp code here"

ext = load_inline(name="ext", cpp_sources=cpp_source, cuda_sources=cuda_source, functions=["run"])

class ModelNew(nn.Module):
    def forward(self, x):
        return ext.run(x)
"""

    tests = [
        ("Banned Import (subprocess)", code_subprocess, False),
        ("Banned ImportFrom (os)", code_os, False),
        ("Banned Call (F.relu)", code_fn_call, False),
        ("Banned Call (torch.nn.functional.relu)", code_fn_direct, False),
        ("Missing load_inline", code_no_inline, False),
        ("Legitimate Kernel", code_legit, True),
    ]
    
    print("Running SP3 Anti-Hacking Static Analysis Tests...\n")
    
    for name, code, expected_safe in tests:
        is_safe, err = check_security(code)
        
        if is_safe == expected_safe:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
            
        print(f"{status:8} | {name:<40} | Safe: {is_safe} (Expected: {expected_safe})")
        if not is_safe and expected_safe:
            print(f"           Error: {err}")
            
    print(f"\nResults: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
