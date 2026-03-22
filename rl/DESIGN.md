# KernelForge RL — Design Document

## What This Is

Multi-turn GRPO training on Qwen3-14B to write optimized CUDA kernels.
Each training step: sample G=8 trajectories over T=4 turns per prompt,
compute discounted advantages, update the model with GRPO+DAPO loss.

---

## Model

- **Base**: Qwen3-14B with SFT LoRA adapter (`../sft/sft_qwen3_14b_lora`)
- **Training**: LoRA fine-tuning via GRPO (full weights frozen, adapter updated)
- **Inference**: SGLang server with live LoRA hot-swap between rollout and update

---

## Training Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| Group size G | 8 (default 16) | trajectories per prompt |
| Turns T | 4 | Kevin's recipe |
| Discount γ | 0.4 | Kevin ablation (best of 0.1/0.4/0.7/1.0) |
| Learning rate | 2e-6 | |
| Warmup steps | 10 | cosine schedule |
| Batch size | 4 | prompts per gradient step |
| GRPO epochs | 2 | gradient updates per rollout |
| Clip low | 0.2 | standard PPO |
| Clip high | 0.28 | DAPO Clip-Higher (asymmetric) |
| Max grad norm | 0.05 | |
| Max new tokens | 6000 | thinking + code budget |
| Think budget | 2000 | tokens before forcing code output |
| Length penalty | 0.0001/token | ~-0.3 at 3000 tokens |

---

## Reward Structure

| Outcome | Reward |
|---|---|
| Correct | `0.3 + speedup_reward` |
| Nearly correct (<30% wrong) | `+0.05` |
| Partially wrong (30–90%) | `-0.05` |
| Mostly wrong (>90%) | `-0.10` |
| Shape mismatch | `-0.25` |
| Compile fail | `-0.50` |
| No code block | `-1.00` |
| Overlong (≥95% token budget) | extra `-0.50` |
| Regression (correct→wrong) | extra `-0.30` (SCoRe) |

Speedup reward: `speedup` if ≤ 2×, `2 + log₂(speedup/2)` above 2×, capped at 3.0.
All rewards include a per-token length penalty of -0.0001.

**Graduated rewards** create gradient signal at every failure level — the model
isn't stuck at all-zero reward on hard problems.

**SCoRe regression penalty**: when a previously-correct trajectory becomes wrong,
apply an extra -0.30 to specifically discourage breaking working code.

---

## Multi-Turn Context

All prior turns kept in context (full history), matching Kevin, Dr. Kernel, MURPHY.
`<think>` blocks stripped from prior assistant turns — model sees its own code + feedback only.

Discounted returns: `R_t = Σ_{k≥t} γ^(k-t) * r_k` with γ=0.4.
This means a turn-1 correct kernel still receives credit even if turn 2 breaks it.

---

## Feedback Format

Exactly Kevin (arXiv:2507.11948 Appendix D) — raw result + error, nothing else:

```
Your previous answer failed to compile. Here is the error message:
<raw compiler output>

Restart your reasoning process and generate new, complete code.
```

```
Your previous answer was incorrect. Here is the error message:
<sandbox correctness detail with wrong element samples>

Restart your reasoning process and generate new, complete code.
```

```
Your previous answer was correct at 1.43x speedup over PyTorch.

Restart your reasoning process and generate new, complete code.
```

No routing on wrong_frac. No "rewrite from scratch" vs "fix the issue."
No "try to optimize further." The model reasons autonomously from context + reward.

**Why**: Prescriptive feedback caused 7/8 correct turn-1 → 0/8 correct turn-2.
"Try to optimize further" triggered risky tiled matmul attempts that broke everything.

---

## Sandbox Diagnostics

`sandbox.py` evaluates each kernel and returns:

| Field | Description |
|---|---|
| `compiles` | bool — JIT compilation succeeded |
| `correct` | bool — all outputs match within atol=rtol=1e-3 |
| `compiler_error` | compiler output OR correctness failure detail |
| `wrong_frac` | fraction of elements exceeding tolerance |
| `shape_ok` | False if output shape mismatches reference |
| `runtime_ms` | median of 10 timed CUDA runs |
| `baseline_runtime_ms` | same for reference PyTorch model |

Correctness failure message includes: error pattern, max/mean abs error, bias direction,
worst position (coords, expected, got), and first 4 wrong elements with coordinates.

Memory corruption detection: `max_err > 1e5` while expected scale `< 10` → emits
"MEMORY CORRUPTION" message pointing to CPU tensor / plain-list weight as likely cause.

---

## System Prompt

Sections (in order):
1. **Role**: expert NVIDIA CUDA Systems Engineer
2. **Output Format**: one ```python block, complete `model_new.py`, `load_inline`
3. **Constraints**: includes, binding, float32, no cuBLAS/cuDNN/CUTLASS
4. **Correctness Checklist** (CUDA Agent SKILL.md): thread bounds, syncthreads, data types, memory safety
5. **Common Bugs**: fmaxf, max threads, shared memory declaration, no --use_fast_math, nn.Parameter
6. **Iteration Strategy**: diagnose first, never risk correctness for speed, verify index formulas

---

## What Didn't Work (and Why)

| Thing tried | Why removed |
|---|---|
| `"likely indexing error or off-by-one"` hint | Caused model to change correct weight indexing → broke it (wrong diagnosis) |
| `"try shared memory, vectorized loads"` in feedback | Triggered risky tiling on correct kernels → 7/8→0/8 collapse |
| CUDA DSA template message injection | Pattern-matching hack; replaced actual error |
| `"rewrite from scratch"` vs `"fix the issue"` routing on wrong_frac | Prescriptive routing — Kevin doesn't do this, minimal is better |
| γ=0.7 | Too high — Kevin ablation shows 0.4 is best |
| Last-turn-only context | Dr. Kernel, MURPHY, Kevin all use full history |
| Best-of-turn context injection | Cherry-picking hack |
| Tiling-specific bugs in system prompt | Derived from one training run, not general knowledge |

---

## Key Research

| Paper | What we use |
|---|---|
| Kevin (arXiv:2507.11948) | γ=0.4, full history, minimal feedback format, "Restart your reasoning" |
| Dr. Kernel (arXiv:2602.05885) | full history context confirmed |
| CUDA Agent (arXiv:2602.24286) | correctness checklist in system prompt |
| SCoRe (Google, ICLR 2025) | regression penalty (-0.30) on correct→wrong |
| DAPO | asymmetric clip (0.2/0.28) |
| KernelBench (arXiv:2502.10517) | benchmark + prompt format baseline |

---

## Anti-Hack Policy

A change is principled if it is backed by a published paper or established CUDA domain knowledge,
and not derived specifically from observing a pattern in our training logs.

**Principled:**
- Kevin γ=0.4 and minimal feedback (paper ablation)
- CUDA Agent correctness checklist (general CUDA engineering, in SKILL.md)
- `nn.Parameter` rule (factual Python/PyTorch behavior)
- `--use_fast_math` rule (provably fails atol=1e-3 with chained transcendentals)
- Raw error messages + sample wrong elements (ground truth, zero interpretation)
- SCoRe regression penalty (ICLR 2025 paper)

**Hack:**
- Injecting canned messages when specific error strings are matched
- Adding system prompt rules derived from one specific training failure
- Routing feedback differently based on wrong_frac thresholds
- Any message that steers toward a specific fix rather than reporting facts
