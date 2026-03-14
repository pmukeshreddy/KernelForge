"""
test_reward.py - Fast mock tests verifying the SP2 reward formula.
"""
import sys
from reward import calculate_reward


def run_tests():
    passed = 0
    failed = 0
    
    # Format: (name, mock_sandbox_result, expected_reward)
    tests = [
        ("Compile Fail", 
         {"compiles": False, "correct": False}, 
         0.0),
        
        ("Wrong Output", 
         {"compiles": True, "correct": False}, 
         0.0),
         
        ("Missing Baseline Time", 
         {"compiles": True, "correct": True, "runtime_ms": 5.0}, 
         0.0),
         
        ("Missing Kernel Time", 
         {"compiles": True, "correct": True, "baseline_runtime_ms": 5.0}, 
         0.0),
        
        ("Negative Time Anomaly", 
         {"compiles": True, "correct": True, "baseline_runtime_ms": 5.0, "runtime_ms": -1.0}, 
         0.0),
        
        ("Correct, 2x slower", 
         {"compiles": True, "correct": True, "baseline_runtime_ms": 5.0, "runtime_ms": 10.0}, 
         0.5),
        
        ("Correct, same speed", 
         {"compiles": True, "correct": True, "baseline_runtime_ms": 5.0, "runtime_ms": 5.0}, 
         1.0),
        
        ("Correct, 2x faster", 
         {"compiles": True, "correct": True, "baseline_runtime_ms": 5.0, "runtime_ms": 2.5}, 
         2.0),
        
        ("Correct, 3x faster", 
         {"compiles": True, "correct": True, "baseline_runtime_ms": 6.0, "runtime_ms": 2.0}, 
         3.0),
         
        ("Correct, 15x faster (capped)", 
         {"compiles": True, "correct": True, "baseline_runtime_ms": 15.0, "runtime_ms": 1.0}, 
         10.0),  # Max reward cap
    ]
    
    print("Running SP2 Reward Function Tests...\n")
    
    for name, mock_res, expected in tests:
        actual = calculate_reward(mock_res)
        
        # Float comparison with tolerance
        if abs(actual - expected) < 1e-6:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
            
        print(f"{status:8} | {name:<30} | Expected: {expected:>5.1f} | Actual: {actual:>5.1f}")
        
    print(f"\nResults: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
