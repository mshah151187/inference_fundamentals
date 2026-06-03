# Nsight Systems — Batch=1 Kernel Findings

**Setup:** GPT-2 (124M), batch=1, 10 prompts, max_tokens=50, A100-SXM4-40GB  
**Script:** `nsys_profiler/script/inference_nsys.py` (no torch.profiler — avoids CUPTI conflict)  
**File:** `nsys_batch1_clean.nsys-rep`  
**Screenshots:** `batch_1/screenshots/`

---

## Kernel Row Observations

### 1. Kernel Types Fired

Three dominant kernel variants visible in the timeline:

| Kernel | % of GPU time | What it is |
|---|---|---|
| `std::enable_if<T7...` (attention) | 29.8% | Flash Attention efficient_attention backend (FP32) |
| `gemv2T_kernel_val` | 17.5% | GEMV — transposed matrix-vector multiply |
| `gemvNSP_kernel` | 15.0% | GEMV — non-square panel variant |

### 2. GEMV, Not GEMM — batch=1 degenerates matmul

At batch=1, linear layer matmul `(seq_len, hidden) × (hidden, hidden)` has seq_len=1 per decode step — the input is effectively a vector. cuBLAS detects this and dispatches GEMV (matrix-vector) kernels instead of GEMM (matrix-matrix).

Evidence from tooltip:
```
gemv2T_kernel_val
  Grid: <<<4281, 1, 1>>>   ← 1D grid = GEMV (GEMM would be 2D: <<<X, Y, 1>>>)
  Block: <<<128, 1, 1>>>
  Num Launch from thread: 2888   ← launched 2888 times across 10 prompts × 50 tokens
```

GEMV uses far fewer SMs than GEMM — most of the A100's 108 SMs sit idle during each kernel.

### 3. White Gaps = GPU Idle, Waiting for CPU

The CUDA HW Kernel row shows alternating colored blocks (GPU executing) and white gaps (GPU idle). The gaps are roughly equal to or larger than the kernel blocks at this zoom level.

```
GPU:  [kernel─][gap────][kernel─][gap────][kernel─][gap────]
CPU:           ← dispatching next cudaLaunchKernel (~28μs) →
```

**The GPU is waiting for the CPU** — not the other way around. CPU dispatch overhead (~28μs per kernel) exceeds kernel execution time (~10-30μs for GEMV at batch=1). This is the CPU-bottleneck diagnosis confirmed visually.

Kernel launch latency from tooltip: `+8.548 μs` — time from CPU submission to GPU starting execution (queue wait time).

### 4. Pageable H2D Memory Copy

Two green memory blocks visible around t=4s — input tokens copied from CPU RAM to GPU via **pageable memory** (not pinned).

```
Memcpy H2D (pageable) — visible as green block in CUDA HW Memory row
```

Pageable transfer requires an extra staging step: CPU → OS staging buffer → GPU. Pinned memory eliminates the staging buffer and transfers directly. This is an optimization opportunity — `torch.zeros(..., pin_memory=True)` would make this transfer faster. (See `tensor_pin_memory/` for deep dive.)

### 5. Memory Row Between Kernels

The CUDA HW Memory row is flat (empty) between compute kernels. `aten::cat` (KV cache concatenation, 292ms CPU time in torch.profiler) does not produce visible GPU memory operations — the actual copy is tiny (few KB per layer at seq_len=50). The cost of `aten::cat` is CPU setup overhead, not GPU memory bandwidth. This is why PagedAttention's benefit is eliminating CPU overhead, not GPU bandwidth.

---

## Summary Diagnosis

| Observation | Evidence |
|---|---|
| CPU-dispatch bound | White gaps between every kernel = GPU idle waiting for CPU |
| GEMV not GEMM | `gemv2T`, `gemvNSP` kernel names; 1D grid dims |
| Low SM utilization | 1D grid, few blocks per SM — most SMs idle each kernel |
| Pageable input transfer | Green H2D block labeled "pageable" at t≈4s |
| Attention using efficient_attention (not Flash Attn v2) | `std::enable_if` kernel = CUTLASS FMHA, not `flash_fwd_kernel` (FP32 cannot use FA v2) |

**Core finding: batch=1 is simultaneously CPU-dispatch bound AND compute-underutilized. The GPU is faster than the CPU can feed it.**

---

## Next — Threads Section

- CUDA API row: dense stream of `cudaLaunchKernel` calls confirming CPU dispatch rate
- Python thread: shows Python/PyTorch overhead between kernel launches
- cuBLAS row: algorithm selection per kernel call
