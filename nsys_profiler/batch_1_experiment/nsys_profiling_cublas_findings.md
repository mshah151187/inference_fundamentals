# Nsight Systems — Batch=1 cuBLAS Findings

**Screenshots:** `screenshots/cuBLAS_1.png`, `screenshots/cuBLAS_2.png`, `screenshots/cuBLAS_3.png`

---

## What is the cuBLAS Row?

The cuBLAS row shows **CPU-side cuBLAS API call durations** — when the Python thread called into the cuBLAS library to dispatch a matrix multiplication. Every linear layer in GPT-2 (attention projections Q/K/V/out, MLP fc1/fc2) goes through this row.

There are two cuBLAS rows in the Nsight timeline — they show the same operation from opposite sides:

```
NVTX (cuBLAS)        — GPU-side: how long the GPU actually executed the matmul kernel
cuBLAS (under thread) — CPU-side: how long the CPU spent inside the cuBLAS API call
```

---

## Connection to Kernel Launch Process

The cuBLAS rows map directly to the steps in `knowledge_base/kernel_launch_process.md`:

```
cuBLAS row (CPU side):
  = kernel_launch_process.md Step 3 — C++ CUDA Implementation
    1. Validate tensor shapes
    2. Compute output shape
    3. Allocate output tensor       (aten::empty in torch profiler)
    4. Retrieve cuBLAS handle       ← cuBLAS handle lookup for current CUDA stream
    5. Call cublasLtMatmul(...)     ← submits kernel asynchronously, returns immediately
  Total CPU-side cost: ~21μs

NVTX (cuBLAS) row (GPU side):
  = Actual GPU kernel executing after async dispatch
    cuBLAS selected kernel variant (gemv2T_kernel_val or gemvNSP_kernel for batch=1)
    Tensor Cores / CUDA Cores performing the matrix multiply
  Total GPU execution: ~13μs
```

`cublasLtMatmul` (Lt = "Light") is the flexible cuBLAS API that supports different data types, layouts, and algorithm selection — used in place of the older `cublasSgemm` for modern PyTorch.

---

## Observation 1 — Two CPU Blocks Per Matmul: GetHeuristic + Dispatch

**Screenshot:** `cuBLAS_2.png` — two consecutive blocks on the cuBLAS CPU row:

```
Block 1: cublasLtMatMulAlgoGetHeuristic   ← CPU only, no kernel launched
Block 2: cublasLtMatmul [21.313 μs]       ← actual kernel dispatch to GPU
```

**`cublasLtMatMulAlgoGetHeuristic`** is a pure CPU call. cuBLAS asks internally: "given this matrix shape (M, K, N) and data type, which of my kernel variants is fastest for this GPU?" It returns an algorithm descriptor. No GPU work happens here.

**`cublasLtMatmul`** uses that descriptor to submit the kernel to the GPU stream and returns immediately (async). This is the 21.3μs CPU dispatch cost.

---

## Observation 2 — NVTX GPU Block Starts Before the Second CPU Block: Pipelining

**Screenshot:** `cuBLAS_2.png` — the NVTX `cublasLtMatmul [13.088 μs]` GPU block starts visibly earlier than the second CPU block (`cublasLtMatmul [21.313 μs]`).

These are NOT the same operation. They are two consecutive matmuls:

```
Earlier CPU call (left of visible window):
  cublasLtMatmul (matmul N) → submitted to GPU stream → CPU returns immediately

GPU stream (command queue):
  [executing matmul N, 13.088μs]   ← this is the NVTX block you see

CPU (simultaneously, preparing matmul N+1):
  cublasLtMatMulAlgoGetHeuristic   ← select algorithm for matmul N+1
  cublasLtMatmul (matmul N+1)      ← dispatch matmul N+1 to GPU stream
```

The CUDA stream is a command queue. CPU writes into it and moves on immediately. GPU drains the queue independently. So GPU executes matmul N while CPU is simultaneously preparing and dispatching matmul N+1 — natural pipelining within a single decode step.

**GPU did not start without a dispatch** — it started because an earlier CPU call (matmul N's dispatch) already submitted it. The current visible CPU blocks are for the NEXT operation.

---

## Observation 3 — CPU Dispatch Cost Exceeds GPU Execution Time

```
NVTX (cuBLAS):  cublasLtMatmul  [13.088 μs]   ← GPU execution time
cuBLAS row:     cublasLtMatmul  [21.313 μs]   ← CPU dispatch duration
```

CPU overhead (21.3μs) > GPU execution (13.1μs).

**What the 21.3μs CPU dispatch includes:**
- cuBLAS handle retrieval for current CUDA stream
- Use algorithm descriptor from GetHeuristic
- Workspace validation
- Kernel launch via CUDA driver (ioctl to submit)
- Return to caller

**What the 13.1μs GPU execution is:**
- Actual matrix multiply on CUDA cores / Tensor Cores
- For batch=1: GEMV (matrix-vector), not full GEMM — weight matrix × single input vector
- Loads weight row from HBM → multiply → accumulate → write output

**Why CPU > GPU for batch=1:**
Matrices are tiny at batch=1. GPU finishes the math in 13μs but CPU spent 21μs just setting it up. This is the **kernel launch overhead dominates** problem — directly visible in these two rows.

---

## Observation 2 — The Gap Between CPU Dispatch and GPU Execution

In `cuBLAS_1.png`, you can see the rhythm: a `cublasLtMatmul` block appears in the cuBLAS (CPU) row, then slightly later the corresponding kernel fires in the CUDA HW row above. The gap between them is the **async dispatch latency** — time from CPU submitting the kernel to GPU actually starting it.

```
CPU:  [cublasLtMatmul API call, 21μs] → returns
                                         ↓  async gap (~few μs)
GPU:                                    [kernel executes, 13μs]
```

This is why CUDA kernel submission is called asynchronous — the CPU call returns before the GPU starts. The GPU works from a command queue; the CPU writes to the queue and moves on.

---

## Observation 3 — Each Linear Layer Is a Separate cublasLtMatmul Call

In `cuBLAS_1.png`, multiple `cublasLtMatmul` blocks are visible in sequence. Each one corresponds to one linear layer in one token generation step:

```
One transformer layer, one decode step:
  cublasLtMatmul → Q projection     (768 × 768)
  cublasLtMatmul → K projection     (768 × 768)
  cublasLtMatmul → V projection     (768 × 768)
  cublasLtMatmul → attention output (768 × 768)
  cublasLtMatmul → MLP fc1          (768 × 3072)
  cublasLtMatmul → MLP fc2          (3072 × 768)
```

GPT-2 has 12 layers × 6 matmuls = 72 cublasLtMatmul calls per decode step. At 50 tokens generated per prompt, that's 3,600 cuBLAS calls per prompt.

Each call pays the 21μs CPU overhead. The GPU executes each in ~13μs. The CPU dispatch cost accumulates and directly contributes to the white gaps (GPU idle) visible in the CUDA HW kernel row.

---

## Observation 4 — GPU Execution Exceeds CPU Dispatch for MLP Layers

**Screenshot:** `cuBLAS_3.png`

```
CPU dispatch:  cublasLtMatmul  [17.901 μs]
GPU execution: cublasLtMatmul  [22.336 μs]   (NVTX row)
CUDA HW:       void gemvNSP_kernel<float...>  ← MLP layer kernel variant
```

GPU (22.3μs) > CPU (17.9μs) — the opposite ratio from Observation 3.

**Why: larger weight matrix = more HBM loads = longer GPU time.**

At batch=1 all GEMVs are memory-bound — GPU execution time is proportional to how many weight bytes are loaded from HBM:

```
Attention projection (768×768 weight):
  HBM load: 768×768 × 4B = 2.4MB
  GPU: 13.1μs   CPU: 21.3μs   → CPU dispatch dominates

MLP layer (768×3072 weight, fc1):
  HBM load: 768×3072 × 4B = 9.4MB
  GPU: 22.3μs   CPU: 17.9μs   → GPU execution dominates
```

The MLP weight matrix is 4× larger than attention projections → GPU spends 4× longer loading weights from HBM → GPU execution time crosses above CPU dispatch cost.

**The sequential pattern (CPU first, then GPU):**

In cuBLAS_2, the GPU was already running a previous matmul while CPU dispatched the next one (pipelined). Here, CPU dispatch finishes before GPU starts — the stream had nothing queued at this point. This happens at layer boundaries where the previous kernel has already completed and the stream is empty waiting for the next dispatch.

**Overall picture — uneven bottleneck across layer types:**

```
Attention layers (768×768):   CPU dispatch (21μs) > GPU (13μs)  → GPU starves waiting for CPU
MLP layers (768×3072):        GPU (22μs) > CPU dispatch (18μs)  → CPU gets breathing room
```

Not all layers are equally launch-overhead dominated at batch=1. MLP layers are closer to balanced because the weight matrices are 4× larger. But the fix is the same — at larger batch sizes GPU execution time grows with batch size while CPU dispatch stays flat at ~18–21μs regardless of batch, making GPU time dominate across all layer types.

---

## Summary

| Row | What it shows | Duration seen |
|---|---|---|
| cuBLAS (CPU) | CPU time inside cublasLtMatmul API call | ~21μs |
| NVTX (cuBLAS) | GPU time executing the matmul kernel | ~13μs |

**Key takeaway:** For batch=1, CPU dispatch cost exceeds GPU execution time per matmul. This is the root cause of GPU idle gaps visible in the CUDA HW row. The fix is batching — at larger batch sizes the GPU execution time grows (more data to process) while CPU dispatch cost stays flat (~21μs regardless of batch size), making GPU time dominate and utilization rise.
