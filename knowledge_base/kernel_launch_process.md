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

## How torch.compile Eliminates This

In eager mode: 140K separate `cudaLaunchKernel` calls = 1.088s CPU overhead.

`torch.compile` with CUDA graphs:
```
First call (trace):
  PyTorch records all kernel launches for one forward pass into a CUDA Graph
  Cost: same as eager + graph capture overhead

Subsequent calls (replay):
  Single cudaGraphLaunch() replays all kernels
  CPU cost: ~1 cudaGraphLaunch call instead of 140K cudaLaunchKernel calls
  Dispatch overhead: ~microseconds instead of ~1 second
```

The GPU executes the exact same kernels — but the CPU overhead collapses from
1.088s to near zero. This is Phase 3 of the project.

---

## CUDA Graph — How It Works

### The problem it solves

Every kernel launch in eager mode follows the full CPU pipeline:
Python → Dispatcher → C++ → cudaLaunchKernel → GPU stream queue.
At batch=1 this overhead (~28μs per op) exceeds GPU execution time (~13.7μs).
For a 32-layer transformer with ~50 ops per layer that's 1,600 kernel launches per
forward pass — over 44ms of pure CPU dispatch cost on every decode step.

### Capture phase — record, don't execute

```python
# Warm up first (cuBLAS algorithm cache, allocations)
for _ in range(3):
    output = model(input_ids)

# Capture: GPU goes into record mode
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(input_ids)   # no GPU execution happens here
                                # CPU records the sequence of kernel launches
                                # into a graph data structure
```

During capture, CUDA intercepts every `cudaLaunchKernel` call and logs it as a
node in a DAG (directed acyclic graph) with edges representing dependencies:

```
Graph nodes (each = one kernel + its args):
  [embedding_lookup] → [layernorm_0] → [addmm_qkv] → [attention] → [addmm_out]
                                ↓
                          [addmm_mlp1] → [gelu] → [addmm_mlp2]
                                                        ↓
                                                  [layernorm_32] → [lm_head]
```

Tensor pointers (addresses) are baked into the graph nodes at capture time.
The input and output tensors must be pre-allocated and reused across replays.

### Replay phase — single call, full forward pass

```python
# Update input in-place (same memory address, new values)
input_ids.copy_(new_token_ids)

# Replay: entire forward pass in one API call
g.replay()
# CPU cost: one cudaGraphLaunch (~5μs) instead of 1,600 cudaLaunchKernel calls
# GPU executes the exact same kernels in the exact same order
```

```
Eager mode:
  CPU: [launch][launch][launch]...[launch]  ← 1,600 calls × 28μs = 44ms CPU
  GPU:         [k1][k2][k3]...[k1600]

CUDA Graph:
  CPU: [graphLaunch]                         ← 1 call × 5μs = 5μs CPU
  GPU: [k1][k2][k3]...[k1600]               ← identical GPU execution
```

GPU execution is identical. CPU overhead collapses from 44ms to 5μs.

### Why decode works but prefill doesn't

CUDA Graph requires the computation graph to be **structurally identical** across
replays — same kernels, same tensor shapes, same data flow.

```
Decode (one token per step):
  Input shape: [batch, 1]          ← fixed every step
  KV cache shape: grows, but attention kernel shape is fixed if batch is fixed
  Computation graph: identical every step
  → CUDA Graph works ✓

Prefill (process full prompt):
  Input shape: [batch, prompt_len]  ← different per request
  prompt_len = 47? 512? 1024?       ← changes every request
  Computation graph: different shapes → different kernel variants selected
  → CUDA Graph cannot be reused ✗
```

### How vLLM uses CUDA Graph

vLLM captures **multiple graphs** — one per batch size — to handle variable
incoming request counts during decode:

```
At startup, vLLM captures graphs for batch sizes: [1, 2, 4, 8, 16, 32, ...]

Incoming decode step with 5 requests:
  → pad to batch=8 (next captured size)
  → replay graph captured for batch=8
  → ignore padded output rows

Prefill: always runs in eager mode (variable prompt lengths)
Decode:  always runs via CUDA Graph (fixed shape per captured batch size)
```

The padding overhead (running batch=8 instead of batch=5) is small compared to
the dispatch savings from graph replay.

### Constraint — tensors must be pre-allocated

Because tensor addresses are baked into graph nodes at capture time, you cannot
allocate new tensors during replay. All inputs, outputs, and intermediate buffers
must be allocated before capture and reused across replays.

```python
# Pre-allocate before capture
static_input  = torch.zeros(batch, seq_len, dtype=torch.long, device='cuda')
static_output = torch.zeros(batch, vocab_size, dtype=torch.float, device='cuda')

# Capture with these static tensors
with torch.cuda.graph(g):
    static_output = model(static_input)

# Each decode step: copy new data into pre-allocated buffers, then replay
static_input.copy_(new_token_ids)
g.replay()
logits = static_output  # read result from pre-allocated buffer
```

This is why vLLM's decode loop looks different from a naive inference loop —
it manages a pool of static buffers rather than allocating tensors per request.

---

## Why Shape Matters and Why Bucketing Helps

### Shape is baked into three things simultaneously

When a CUDA Graph node is captured for an `addmm` kernel, the capture records:

```
addmm at capture time — seq_len=50, batch=1, hidden=768:
  (M, K, N) = (50, 768, 768)

  1. Kernel variant  — cuBLAS selects Variant B (tile 32×32) for this shape
  2. Grid dimensions — ceil(50/32) × ceil(768/32) = 2 × 24 = 48 thread blocks
  3. Tensor args     — M=50, N=768, K=768 baked into kernel arguments

All three are recorded in the graph node at capture time.
```

If you replay this graph with `seq_len=100` (M=100):
```
Grid dims replayed: (2, 24, 1)  ← still 48 thread blocks for only rows 0..49
Rows 50..99: never computed — no thread blocks assigned to them
Output: silently wrong, no error raised
```

The GPU doesn't validate shapes at replay time. It just executes what was recorded.

### Why every unique shape needs its own graph

For a transformer layer, every `addmm` shape is a function of `(batch_size, seq_len)`:

```
QKV projection:    (batch × seq_len,  hidden) × (hidden, 3×hidden)
Output projection: (batch × seq_len,  hidden) × (hidden,   hidden)
MLP fc1:           (batch × seq_len,  hidden) × (hidden, 4×hidden)
MLP fc2:           (batch × seq_len, 4×hidden) × (4×hidden, hidden)
```

Change either `batch_size` or `seq_len` → `M` changes in every kernel →
different cuBLAS variant selected → different grid dims → need a new capture.

One captured graph is valid for exactly one `(batch_size, seq_len)` pair.

### Bucketing — fix the shape space so graphs can be pre-captured

Without bucketing, seq_len can be anything from 1 to max_context (e.g., 8192).
You can't pre-capture 8192 different graphs at startup — too much memory and
capture time.

Bucketing collapses the continuous shape space into a small fixed set:

```
Bucket sizes (example):  [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

Incoming request with seq_len=100:
  → round UP to nearest bucket: 128
  → pad input to seq_len=128 (fill with dummy tokens)
  → replay graph captured for seq_len=128
  → discard padded output positions
```

At startup, vLLM captures one CUDA Graph per bucket size — 10 captures instead
of 8192. Each capture pre-runs cuBLAS algorithm selection and locks in the right
kernel variant and grid dims for that bucket's shape.

```
Startup (once):
  capture graph for seq_len=1    → graph_1
  capture graph for seq_len=2    → graph_2
  ...
  capture graph for seq_len=512  → graph_512

Per decode step (seq_len=100):
  → select graph_128
  → copy padded input into static buffer
  → graph_128.replay()           ← 5μs CPU, correct grid dims for M=128
```

### The padding waste tradeoff

Bucketing trades compute waste for dispatch efficiency:

```
seq_len=100 padded to bucket 128:
  Wasted compute: 28 extra token positions processed = 22% overhead
  Dispatch saving: 44ms eager CPU overhead → 5μs graph replay

At inference scale (thousands of requests/sec), the dispatch saving dominates.
Padding waste is bounded by the bucket granularity — finer buckets = less waste
but more graphs to capture and more GPU memory for static buffers.
```

Coarser buckets (powers of 2) are common because the wasted compute is at most
2× (worst case: seq_len just above the previous bucket) and the number of graphs
stays small (log2(max_seq_len) graphs total).

---

## Relationship to Other Knowledge Base Docs

- `profiling_guide.md` — how to measure and interpret CPU vs CUDA time ratios
- `kv_cache.md` — aten::cat is a separate kernel launch per decode step (same pipeline)
- `PagedAttention.md` — eliminates aten::cat kernel launches entirely
