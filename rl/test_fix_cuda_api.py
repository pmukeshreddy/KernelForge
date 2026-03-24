"""
test_fix_cuda_api.py - Unit tests for _fix_cuda_api() auto-fix patterns.

Only tests patterns we KEEP (silent bugs, type system, environment fixes).
Removed patterns (std::max, .type(), VLA, __shared__ VLA, __fminf, __clamp)
are now learned by the model through RL error feedback.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import _fix_cuda_api


def test_bug9_stride_and_size_no_arg():
    """Bug 9: .stride() → .strides(), .size() → .sizes()"""
    code = "auto s = input.stride();"
    fixed = _fix_cuda_api(code)
    assert ".strides()" in fixed, f"Expected .strides(), got: {fixed}"
    # .stride(dim) should NOT be changed
    code2 = "auto s = input.stride(0);"
    fixed2 = _fix_cuda_api(code2)
    assert ".stride(0)" in fixed2, f".stride(dim) should be preserved, got: {fixed2}"
    # .size() (no-arg) → .sizes()
    code3 = 'TORCH_CHECK(predictions.size() == targets.size(), "mismatch");'
    fixed3 = _fix_cuda_api(code3)
    assert ".sizes()" in fixed3, f"Expected .sizes(), got: {fixed3}"
    assert ".size()" not in fixed3, f".size() should be fixed, got: {fixed3}"
    # .size(dim) should NOT be changed
    code4 = "int n = input.size(0);"
    fixed4 = _fix_cuda_api(code4)
    assert ".size(0)" in fixed4, f".size(dim) should be preserved, got: {fixed4}"
    print("✅ Bug 9: .stride()/.size() → .strides()/.sizes()")


def test_bug10_intarrayref_data():
    """Bug 10: .sizes()/.strides() → .sizes().data() in kernel arg contexts"""
    code = "kernel<<<b,t>>>(input.sizes(), input.strides(), n);"
    fixed = _fix_cuda_api(code)
    assert "input.sizes().data()" in fixed, f"Expected .data(), got: {fixed}"
    assert "input.strides().data()" in fixed, f"Expected .data() on strides, got: {fixed}"
    print("✅ Bug 10a: kernel arg context gets .data()")


def test_bug10_no_double_data():
    """Bug 10: don't double-apply .data().data()"""
    code = "kernel<<<b,t>>>(input.sizes().data(), n);"
    fixed = _fix_cuda_api(code)
    assert ".data().data()" not in fixed, f"Double .data() applied: {fixed}"
    assert "input.sizes().data()" in fixed, f"Expected single .data(), got: {fixed}"
    print("✅ Bug 10b: no double .data()")


def test_bug10_no_apply_on_subscript():
    """Bug 10: don't apply when subscripted (e.g., .sizes()[0])"""
    code = "int64_t dim = input.sizes()[0];"
    fixed = _fix_cuda_api(code)
    assert ".sizes()[0]" in fixed, f"Should not add .data() on subscript, got: {fixed}"
    print("✅ Bug 10c: subscript access preserved")


def test_bug12_const_pointer():
    """Bug 12: int64_t* shape = input.sizes() → const int64_t* shape = ..."""
    code = "int64_t* shape = input.sizes().data();"
    fixed = _fix_cuda_api(code)
    assert "const int64_t* shape" in fixed, f"Expected const, got: {fixed}"
    print("✅ Bug 12: const pointer from sizes()/strides()")


def test_regression_kept_patterns():
    """Ensure kept patterns still work."""
    # Bug 3: .ptr<T> → .data_ptr<T>
    code = "input.ptr<float>()"
    fixed = _fix_cuda_api(code)
    assert ".data_ptr<float>" in fixed, f"Bug 3 regression: {fixed}"

    print("✅ Kept patterns still work")


def test_removed_patterns_no_longer_fixed():
    """Verify removed auto-fixes are actually gone (model learns these via RL)."""
    # std::max should NOT be auto-fixed anymore
    code = "float r = std::max(a, b);"
    fixed = _fix_cuda_api(code)
    assert "std::max" in fixed, f"std::max should NOT be auto-fixed, got: {fixed}"

    # .type() should NOT be auto-fixed anymore
    code2 = "auto t = input.type();"
    fixed2 = _fix_cuda_api(code2)
    assert ".type()" in fixed2, f".type() should NOT be auto-fixed, got: {fixed2}"

    # __fminf should NOT be auto-fixed anymore
    code3 = "__fminf(a, b)"
    fixed3 = _fix_cuda_api(code3)
    assert "__fminf" in fixed3, f"__fminf should NOT be auto-fixed, got: {fixed3}"

    print("✅ Removed patterns are gone (model learns via RL)")


if __name__ == "__main__":
    tests = [
        test_bug9_stride_and_size_no_arg,
        test_bug10_intarrayref_data,
        test_bug10_no_double_data,
        test_bug10_no_apply_on_subscript,
        test_bug12_const_pointer,
        test_regression_kept_patterns,
        test_removed_patterns_no_longer_fixed,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"❌ {t.__name__}: {e}")
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    print(f"{'='*50}")
    sys.exit(0 if passed == len(tests) else 1)
