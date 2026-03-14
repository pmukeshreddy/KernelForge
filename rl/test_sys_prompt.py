"""
test_sys_prompt.py - Verifies the SP5 System Prompt loads correctly.
"""
from sys_prompt import get_system_prompt

def test_system_prompt_loading():
    prompt = get_system_prompt()
    
    # Verify it loaded as a string
    assert isinstance(prompt, str)
    assert len(prompt) > 500
    
    # Verify key constraints are present
    assert "NVIDIA A100 GPU" in prompt
    assert "<torch/extension.h>" in prompt
    assert "__global__" in prompt
    assert "bottleneck" in prompt.lower()
    
    print("✅ PASS | System Prompt Content Constraints Verified")
    print(f"Total tokens (approx): {len(prompt.split())}")
    
    # Preview the prompt
    print("\nPreview of the prompt snippet:")
    print("-" * 50)
    print(prompt[:300] + "...\n[...]\n..." + prompt[-300:])
    print("-" * 50)

if __name__ == "__main__":
    test_system_prompt_loading()
