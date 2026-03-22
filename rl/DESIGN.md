# KernelForge RL — Design Decisions

## Overview

Multi-turn GRPO training where a model iteratively writes and refines CUDA kernels.
Each prompt runs for T=4 turns. G=8 trajectories sampled per prompt.

---

## Reward

| Outcome | Reward |
|---|---|
| Correct | `0.3 + speedup_reward` (log-scaled above 2×) |
| Shape mismatch | `-0.25` |
| Mostly wrong (>90%) | `-0.10` |
| Partially wrong (30-90%) | `-0.05` |
| Nearly correct (<30%) | `+0.05` |
| Compile fail | `-0.50` |
| No code | `-0.80` |
| Regression (was correct, now wrong) | extra `-0.30` (SCoRe) |

Speedup reward: linear below 2×, log₂-scaled above, capped at 3.0.

---

## Multi-Turn Context

Full history kept across all turns (Kevin, Dr. Kernel, MURPHY all confirm this).
`<think>` blocks stripped from prior turns — model sees its own code + feedback only.

Discount factor γ=0.4 (Kevin ablation: best among γ ∈ {0.1, 0.4, 0.7, 1.0}).

---

## Feedback Format

Exactly Kevin (arXiv:2507.11948 Appendix D) — minimal, no prescriptive instructions:

```
Your previous answer failed to compile. Here is the error message:
<error>

Restart your reasoning process and generate new, complete code.
```

```
Your previous answer was incorrect. Here is the error message:
<error>

Restart your reasoning process and generate new, complete code.
```

```
Your previous answer was correct at X.XXx speedup over PyTorch.

Restart your reasoning process and generate new, complete code.
```

**Why minimal?** Prescriptive feedback ("try shared memory", "rewrite from scratch") was
causing the model to attempt risky optimizations on already-correct kernels.
Observed: 7/8 correct turn 1 → 0/8 correct turn 2 after feedback said "try to optimize."
Kevin's approach: give the model the raw result and let it reason autonomously.

---

## System Prompt

Based on KernelBench format + CUDA Agent SKILL.md correctness checklist.

Key sections:
- **Output Format**: one ```python block, complete model_new.py, load_inline
- **Correctness Checklist** (CUDA Agent): thread bounds, syncthreads, data types, memory safety
- **Common Bugs**: fmaxf not std::max, no --use_fast_math, nn.Parameter not plain lists
- **Iteration Strategy**: diagnose first, never risk correctness for speed

---

## Sandbox Diagnostics

`sandbox.py` returns per-evaluation:
- `compiles`: bool
- `compiler_error`: compiler output or correctness failure detail
- `correct`: bool
- `wrong_frac`: fraction of elements exceeding atol=1e-3
- `shape_ok`: False if output shape mismatches reference
- `runtime_ms` / `baseline_runtime_ms`: median of 10 timed runs

Correctness failure includes: pattern (fundamental/partial), max/mean error, bias,
worst position with expected vs got, and first 4 wrong elements with coordinates.

Memory corruption detection: when max_err > 1e5 and expected scale < 10,
emits MEMORY CORRUPTION diagnostic pointing to plain Python list weights.

---

## Key Research

| Paper | What we use |
|---|---|
| Kevin (arXiv:2507.11948) | γ=0.4, full history context, minimal feedback format |
| Dr. Kernel (arXiv:2602.05885) | full history context confirmed |
| CUDA Agent (arXiv:2602.24286) | correctness checklist in system prompt |
| SCoRe (Google, ICLR 2025) | regression penalty on correct→wrong transitions |
| KernelBench (arXiv:2502.10517) | benchmark + prompt format baseline |

---

## Anti-Hack Policy

Changes must be principled (backed by research or established domain knowledge),
not derived from observing specific training failures.

**Not a hack:**
- Kevin's γ=0.4 and minimal feedback (research ablation)
- CUDA Agent's correctness checklist (general CUDA engineering)
- `nn.Parameter` rule (factual Python/PyTorch behavior)
- `--use_fast_math` rule (proven to fail atol=1e-3 in our setup)
- Raw error messages + sample wrong elements (ground truth, no interpretation)

**Is a hack:**
- "likely indexing error or off-by-one" hint in error messages
- Injecting CUDA DSA template message when specific error patterns match
- Adding rules to system prompt derived from one specific training run failure
- "try shared memory, vectorized loads" in feedback for correct kernels
