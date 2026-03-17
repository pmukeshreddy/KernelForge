import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "Qwen/Qwen3-14B"
adapter = "./sft_qwen3_14b_lora"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(adapter)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto",
                                              attn_implementation="sdpa")
model = PeftModel.from_pretrained(model, adapter)
model.eval()

prompt = """<|im_start|>system
You are an expert NVIDIA CUDA Systems Engineer.
Your objective is to write optimized CUDA C++ kernels to replace PyTorch operations.
<|im_end|>
<|im_start|>user
Write a CUDA kernel for element-wise ReLU on a float32 tensor.

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)
```
<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
print("Generating...")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=1.0)

generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n=== Generated Kernel ===")
print(generated)
