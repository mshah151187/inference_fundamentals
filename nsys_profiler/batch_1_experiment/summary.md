# Nsight Systems — Batch=1 Experiment Summary

**Setup:** GPT-2 (124M), batch=1, 10 prompts, max_new_tokens=50, A100-SXM4-40GB  
**Script:** `nsys_profiler/script/inference_nsys.py`  
**Profile file:** `nsys_batch1_clean.nsys-rep`

---

## What Each Row Tells Us

### 1. CUDA HW — Kernel Execution vs Memory

Hardware-level view of how the GPU was used across the full execution window.

- **94% Kernel execution, ~6% Memory operations** at batch=1
- Kernel row expands to show time distribution across kernel types:
  - 17.5% `gemv2T_kernel_val` — GEMV (transposed), linear layer matmuls
  - 13.9% `gemvNSP_kernel` — GEMV (non-square panel), another matmul variant
  - 11.5% `vectorized_elementwise_kernel` — pointwise ops (GELU, dropout, add)
  -  9.1% `fmha_cutlassF_f32_aligned_64x64_rf_sm80` — attention kernel (FP32)
- White gaps visible between kernel blocks = GPU idle, waiting for CPU dispatch

→ Details: `nsys_profiling_kernel_findings.md`

### 2. NVTX Row

PyTorch-annotated GPU-side operation names. NVTX block duration = GPU execution time
of that operation. The annotation is written on the CPU at dispatch time; the GPU block
starts a few microseconds later (stream queue wait).

- Operation name tells you what is executing: cuBLAS (matmul), attention, elementwise
- Aligns with CUDA HW kernel row — same operation, two vantage points:
  ```
  CUDA HW:  [── gemv2T_kernel ──]    ← GPU hardware executing
  NVTX:        [── cuBLAS ──]        ← PyTorch's name for same op
  ```
- If NVTX block for op N+1 starts while CUDA HW still shows op N → pipelining:
  CPU dispatching next op while GPU runs current one

→ Details: `nsys_profiling_cublas_findings.md`

### 3. OS Runtime Libraries

Where the OS is involved — system calls made during the run. Reveals what the CPU is
doing (or blocked on) between GPU kernel launches.

Key syscalls observed:

| Syscall | What it means |
|---|---|
| `ioctl` | CPU submitted work to GPU driver |
| `nanosleep` | CPU voluntarily sleeping (explicit wait) |
| `poll` | CPU blocked, waiting on a file descriptor to become ready |
| `pthread_cond_wait` | CPU thread blocked on a condition variable |
| `mmap64` | Memory mapping (tokenizer cache, model weights) |

**Key finding — `poll` blocks:** between prompts, the CPU spends 74ms in a single
`poll` call (20+ frame call stack leading from `model.generate()` → PyTorch scheduler
→ CUDA sync → OS poll). GPU is idle the entire time. This is 30-40% of total runtime
wasted — CPU not overlapping next prompt's tokenization with GPU execution.

```
Timeline at prompt boundary:
  GPU: [──── last decode kernel ────][ idle 74ms ][ next prompt kernels... ]
  CPU:                               [── poll ──────────────────────────────]
```

→ Details: `nsys_profiling_os_runtime_findings.md`

### 4. cuBLAS Row (CPU side)

CPU-side view of kernel dispatching. Two blocks visible per matmul operation:

```
Block 1: cublasLtMatMulAlgoGetHeuristic  ← pure CPU: selects algorithm from cache
Block 2: cublasLtMatmul                  ← dispatches kernel to GPU stream
```

Sequential by nature — CPU must finish preparing one dispatch before starting the next.
No pipelining on the CPU dispatch side itself. The white gaps BETWEEN cuBLAS blocks on
the CPU row = CPU idle between consecutive dispatch sequences (waiting on sync or
scheduling overhead).

→ Details: `nsys_profiling_cublas_findings.md`

### 5. CUDA API Row

Runtime-level CUDA calls between cuBLAS (CPU) and NVTX (GPU). Reveals which
underlying kernels and optimization tactics cuBLAS chose.

Key calls observed:

| CUDA API call | What cuBLAS chose |
|---|---|
| `cudaLaunchKernel` → `gemv2T_kernel_val` | GEMV dispatch (batch=1 degenerates matmul to vector op) |
| `cudaLaunchKernel` → `splitKreduc_kernel` | Split-K: large K dimension split across SMs, partial sums reduced |
| `cudaLaunchKernel` → `vectorized_elementwise_kernel` | Float4 128-bit vectorized loads for pointwise ops |
| `CatArrayBatchedCopy_aligned16_Config` | KV cache `torch.cat` — cost grows linearly with seq position |
| `cudaMemcpyAsync` | Async H2D/D2D transfers |
| `cudaDeviceSynchronize` | CPU forced to wait for GPU — prompt boundary sync point |

→ Details: `nsys_profiling_cuda_api_findings.md`

---

## Two Major Findings

### Finding 1 — GPU is Starved: White Patches = GPU Waiting for CPU

At batch=1, GEMV kernels complete in 10–30μs. CPU dispatch overhead (~28μs per kernel)
exceeds or matches kernel execution time. The GPU finishes its work and sits idle while
the CPU prepares the next dispatch.

```
GPU: [kernel─10μs][─ idle 28μs ─][kernel─10μs][─ idle 28μs ─]
CPU:               [dispatch~28μs]              [dispatch~28μs]
```

The GPU is faster than the CPU can feed it. A100 has 108 SMs — GEMV at batch=1 uses
a fraction of them. Most SMs are idle during every kernel.

### Finding 2 — CPU Work is Sequential: No Overlap Between Prompts

The `poll` observation from OS Runtime confirms it: `model.generate()` is synchronous.
After the last decode step for prompt N, the CPU blocks in `poll` waiting for GPU to
finish, then begins tokenizing prompt N+1. There is zero overlap:

```
Prompt N:  [─── GPU kernels ───][sync/poll]
Prompt N+1:                               [tokenize][─── GPU kernels ───][sync/poll]
```

Both CPU and GPU are idle at different times — they take turns rather than pipelining.

---

## Unified Diagnosis

**Batch=1 is simultaneously:**
- **CPU-dispatch bound** — GPU faster than CPU can feed it
- **Compute-underutilized** — GEMV uses few SMs; most A100 SMs idle each kernel
- **Sequentially bottlenecked** — prompts processed one at a time, no overlap

The GPU is never the limiting factor at batch=1. The bottleneck is entirely on the
CPU side — dispatch overhead, synchronization, and sequential prompt processing.

---

## What to Expect With Larger Batch Sizes

| Observation at batch=1 | Expected change at batch=N |
|---|---|
| GEMV kernels (1D grid, few SMs) | Switches to GEMM (2D grid, all 108 SMs utilized) |
| White patches dominate GPU row | Gaps shrink — larger kernels take longer, CPU dispatches while GPU still busy |
| 94% kernel, ~6% memory | Memory bandwidth utilization increases — larger weight loads per kernel |
| `poll` blocks at prompt boundary | With dynamic batching, boundaries overlap — less idle |
| CPU dispatch overhead dominates | Compute time grows faster than dispatch overhead → GPU becomes bottleneck |
| Low SM utilization | SM occupancy improves — larger matrices fill more SMs |
| KV cache `torch.cat` cost small | Cat cost grows linearly with seq_len — will become visible at high batch×seq |

**The transition point:** as batch size increases, the system shifts from
CPU-dispatch bound → memory-bandwidth bound (weight loading dominates) →
potentially compute bound (if batch large enough to saturate SMs + FP8/quantization).

At batch=1 we are far left on this curve. The Nsight Systems profile at larger batch
sizes should show:
- GEMM kernels replacing GEMV (2D grid dims in tooltip)
- Fewer white gaps in GPU kernel row
- Memory row more active (weight matrix loads visible)
- CPU `poll` blocks shorter or absent (batch collects multiple prompts)
- `torch.compile` eliminating many small kernel launches → fewer CUDA API calls

---

## Files Reference

| File | What it covers |
|---|---|
| `nsys_profiling_kernel_findings.md` | CUDA HW kernel row — types, GEMV diagnosis, SM utilization |
| `nsys_profiling_cublas_findings.md` | cuBLAS row — algorithm selection, pipelining observations |
| `nsys_profiling_cuda_api_findings.md` | CUDA API row — kernel tactics, Split-K, KV cache cat |
| `nsys_profiling_os_runtime_findings.md` | OS Runtime — poll blocks, mmap, prompt boundary waste |
| `screenshots/` | All annotated Nsight screenshots |
