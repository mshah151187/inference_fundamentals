# CUDA Kernel Launch Process — CPU to GPU Pipeline

Reference: observed in profiling_result.txt — `aten::addmm` Self CPU 674ms vs Self CUDA 329ms.
The 2× gap is explained entirely by this pipeline.

---

## The Full Pipeline — `nn.Linear` to Tensor Core Execution

### Step 1 — Python: `nn.Linear.forward()`

```python
output = self.c_attn(x)   # GPT-2 QKV projection — nn.Linear
```

`nn.Linear.forward()` calls `F.linear(input, weight, bias)` which resolves to:

```python
torch.addmm(bias, input, weight.T)
# = bias + (input @ weight.T)
# This is the aten::addmm you see in the profiler
```

---

### Step 2 — PyTorch Dispatcher (Python → C++, ~8–10μs)

```
torch.addmm()
    ↓
pybind11 bridge          — Python object → C++ ATen Tensor
    ↓
PyTorch Dispatcher
  - reads dispatch key: device=CUDA, dtype=float32, layout=strided
  - looks up dispatch table → selects at::cuda::addmm_cuda()
    ↓
C++ kernel implementation
```

The dispatcher is a multi-level lookup table keyed on (device, dtype, layout).
Every PyTorch op goes through it in eager mode. At 140K kernel launches (our run),
even 8μs per dispatch = 1.12s of pure table-lookup overhead — matches the
`cudaLaunchKernel` 1.088s we see in the profiler.

---

### Step 3 — C++ CUDA Implementation (~10μs)

```
at::cuda::addmm_cuda():
  1. Validate tensor shapes — (M,K) × (K,N) compatible, dtypes match
  2. Compute output shape (M, N)
  3. Allocate output tensor  ← this is the aten::empty call in the profiler
  4. Retrieve cuBLAS handle for current CUDA stream
  5. Call cublasSgemm(handle, ..., M, N, K, alpha=1, A, lda, B, ldb, beta=1, C, ldc)
  6. Return immediately — submission is async
```

Steps 1–4 are pure CPU work. Step 5 submits the kernel asynchronously and returns.
The CPU does not wait for the GPU — it immediately moves on to dispatch the next op.

---

### Step 4 — What is cuBLAS?

cuBLAS = **CUDA Basic Linear Algebra Subroutines** — NVIDIA's GPU-accelerated linear
algebra library. GPU equivalent of CPU BLAS (OpenBLAS, MKL).

Core function used for `addmm`:
```
cublasSgemm:  C = alpha × (A @ B) + beta × C    (single precision)
cublasGemmEx: same but supports mixed precision (FP16, BF16, INT8)
```

The `addmm` operation maps directly:
```
output = bias + input @ weight.T
       = 1 × (input @ weight.T) + 1 × bias
         ↑ alpha                  ↑ beta × C (C initialized to bias)
```

The bias add is **fused inside the GEMM kernel** — computed in registers alongside
the matrix multiply, with no separate HBM round-trip. This is the "fused" in `addmm`
(add + mm). Happens on every call, not just warmup.

---

### Step 5 — cuBLAS Algorithm Selection (~5μs after warmup, ~50–200μs cold)

For matrix multiply `(M, K) × (K, N)`, cuBLAS has dozens of internal kernel variants —
different tile sizes, warp configurations, tensor core layouts optimized for different
matrix shapes and GPU architectures:

```
Variant A: tile 16×16,  2 warps/SM  — best for tiny M, N
Variant B: tile 32×32,  4 warps/SM  — best for medium M, N
Variant C: tile 64×64,  8 warps/SM  — best for large M, N
Variant D: tile 128×128, split-K   — best for very small K
...
```

Selection depends on `(M, K, N, dtype, transposition, GPU arch)`.

**Cold (first call with this shape):**
```
cuBLAS runs heuristic search over variant table
Selects best variant → stores in internal cache
Cost: 50–200μs (one-time per shape)
```

**Warm (subsequent calls, same shape):**
```
Cache hit → direct lookup → launch selected variant
Cost: ~2–5μs
```

**This is what warmup accomplishes.** 3 warmup passes cycle through every
`(M, K, N)` shape the model uses — Q/K/V projection, output projection, MLP fc1,
MLP fc2, lm_head. After warmup every production call hits the cache.

Without warmup: first real request pays 50–200μs per unique shape = 10–100× slower
than steady state. Kubernetes readiness probe must wait for warmup to complete before
routing traffic for exactly this reason.

---

### Step 6 — `cudaLaunchKernel()` — Async Submission (~5μs)

```
cublasSgemm() internally calls cudaLaunchKernel():

  CPU side:
    - Packages kernel function pointer + grid dims + block dims + args
    - Pushes command descriptor onto GPU stream (FIFO command queue in driver)
    - Returns immediately — CPU does NOT wait for GPU

  GPU side (happens later, asynchronously):
    - HW command processor dequeues the command
    - Schedules thread blocks onto available SMs
    - Executes
```

```
CPU timeline:  [dispatch][dispatch][dispatch][dispatch] → moves on immediately
GPU timeline:         [kernel1──][kernel2──][kernel3──] → executes sequentially
```

The `cudaLaunchKernel` entry at 1.088s in the profiler is the **cumulative CPU cost
of 140K push-to-queue operations** — not GPU time. Each push costs ~7.8μs on CPU.

---

### Step 7 — GPU Execution — Tensor Core GEMM (~13.7μs for batch=1)

```
GPU HW Scheduler receives the kernel:

1. Assign thread blocks to SMs
   batch=1, shape (50, 768) × (768, 768):
   → ~12 thread blocks generated
   → 12 of 108 SMs occupied, 96 SMs idle

2. Each active SM:
   - Load tile of A (input rows) from HBM → L2 cache → shared memory
   - Load tile of B (weight cols) from HBM → L2 cache → shared memory
   - Tensor Core WMMA instruction: 16×16×16 tile multiply-accumulate
     (one WMMA = 16×16×16 = 4,096 MACs in a single warp instruction)
   - Accumulate partial results in registers
   - Add bias (beta × C) in registers — no extra HBM load (fused)
   - Write output tile back to HBM

3. Kernel completes, SM signals done to scheduler
```

At batch=1, the matrix is too small to fill all SMs. This is **low occupancy** —
GPU finishes in 13.7μs but most of its hardware sat idle the entire time.

---

## Diagnosis — Why CPU Time > GPU Time in Our Profiling Run

From profiling_result.txt (batch=1, 10 prompts, max_tokens=50):
```
Self CPU total:  4.075s
Self CUDA total: 823ms
Ratio: ~5x
```

The CPU is not slow in absolute terms — 28μs dispatch per kernel is normal eager-mode
overhead. The root cause is that **GPU kernels are too small to amortize that fixed cost.**

At batch=1, each `addmm` produces a matrix `(50, 768) × (768, 768)` — too small to
occupy more than 12 of 108 SMs. The GPU finishes in 13.7μs and sits idle while the CPU
spends 28μs dispatching the next kernel. The GPU is faster than the CPU can feed it.

Two ways to frame the same observation:
- **GPU perspective:** not enough data per kernel, most SMs idle, execution too short
- **CPU perspective:** kernel launch overhead dominates because GPU execution time is
  shorter than dispatch time

Across 140K launches: 140K × 28μs = ~3.9s of dispatch cost — matches the observed
4.075s CPU total. The GPU was actually busy for only 823ms of that wall time.

This is a **batch size problem**, not a kernel efficiency problem. Batching fixes it
by making GPU execution time (500μs+ per call at batch=64) large enough that
dispatch overhead (still ~28μs) becomes negligible.

---

## Per-Call Cost Breakdown (batch=1, addmm on A100)

```
Python dispatch (torch.addmm)          ~2μs
Dispatcher traversal                   ~8μs
Shape validation + output allocation   ~8μs
cuBLAS handle + algorithm cache lookup ~5μs
cudaLaunchKernel (async submit)        ~5μs
─────────────────────────────────────────────
Total Self CPU per call:              ~28μs

GPU stream queue wait                  varies
Tensor core GEMM execution            ~13.7μs
─────────────────────────────────────────────
Total Self CUDA per call:             ~13.7μs

CPU is on the critical path.
GPU cannot start until CPU finishes dispatch.
At batch=1: CPU is 2× slower than GPU execution itself.
```

---

## Why Batching Fixes This

```
batch=1:   matrix (50, 768) × (768, 768)   → GPU time ~13.7μs
batch=64:  matrix (3200, 768) × (768, 768) → GPU time ~500μs

CPU dispatch cost stays constant: ~28μs regardless of batch size.

batch=1:  dispatch(28μs) / gpu(13.7μs)  = 2.0  → CPU is bottleneck
batch=64: dispatch(28μs) / gpu(500μs)   = 0.056 → dispatch is 5% overhead, negligible
```

Batching makes the GPU kernel large enough that dispatch overhead becomes negligible.
The CPU stays busy dispatching the next kernel while the GPU works on the current one.

---

## How torch.compile + CUDA Graphs Eliminate This

In eager mode: 140K separate `cudaLaunchKernel` calls = 1.088s CPU overhead.

`torch.compile` with CUDA graphs records all kernel launches for one forward pass
into a graph, then replays the entire forward pass with a single `cudaGraphLaunch()`
call — CPU overhead collapses from 1.088s to near zero.

### What changes per step — eager vs CUDA graph replay

The 7-step pipeline runs fully during **capture** (once). During **replay** it is
almost entirely bypassed:

| Step | Eager mode | CUDA graph replay |
|---|---|---|
| 1. Python dispatch | `torch.addmm()` → pybind11 per op | `g.replay()` — one call, no op dispatch |
| 2. PyTorch Dispatcher | table lookup per op | skipped entirely |
| 3. C++ implementation | shape validation + output allocation | skipped — shapes baked in, buffers pre-allocated |
| 4. cuBLAS algorithm selection | cache lookup per shape | skipped — variant baked into graph node at capture |
| 5. cudaLaunchKernel | called N times (140K in our run) | replaced by single `cudaGraphLaunch()` |
| 6. GPU execution | tensor core GEMM | **identical** — same kernels, same grid dims |

Steps 1–5 are pure CPU pipeline — completely eliminated during replay.
Step 6 (GPU execution) is identical in both modes.

**Key mental model:** CUDA graphs don't make the GPU faster. They make the CPU get
out of the way faster. The GPU was always capable of running those kernels efficiently
— the bottleneck was the CPU spending 44ms dispatching them one by one. Graph replay
hands the GPU the full work order upfront in one 5μs call.

→ Full explanation, bucketing, and piecewise graphs: **`CUDA_Graph.md`**

---

## Relationship to Other Knowledge Base Docs

- `CUDA_Graph.md` — how CUDA graphs work, shape constraints, bucketing, piecewise execution
- `profiling_guide.md` — how to measure and interpret CPU vs CUDA time ratios
- `kv_cache.md` — aten::cat is a separate kernel launch per decode step (same pipeline)
- `PagedAttention.md` — eliminates aten::cat kernel launches entirely
