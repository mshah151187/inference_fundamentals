# Nsight Systems — Batch=1 CUDA API Findings

**Screenshots:** *(add as captured)*

---

## What is the CUDA API Row?

The CUDA API row shows **CPU-side CUDA runtime API calls** — the actual function calls the CPU makes into the CUDA runtime library to submit work to the GPU. This sits one layer below cuBLAS in the call stack:

```
Python
  → PyTorch op (addmm, layernorm, ...)
      → cuBLAS row  (cublasLtMatmul)         ← cuBLAS API call
          → CUDA API row (cudaLaunchKernel)   ← CUDA runtime call
              → CUDA HW row                   ← GPU executes
```

For non-cuBLAS ops (elementwise, layernorm, attention), PyTorch calls `cudaLaunchKernel` directly — those appear in the CUDA API row without a corresponding cuBLAS entry.

**Duration shown:** CPU time spent inside each CUDA runtime call. For `cudaLaunchKernel` this is the dispatch overhead (~2–5μs). For sync calls (`cudaDeviceSynchronize`) this can be tens to hundreds of milliseconds.

---

## Functions Visible in the CUDA API Row

### `cudaLaunchKernel` — labeled as `kernel` or `globalKernel`

The fundamental kernel submission call. CPU writes the kernel launch parameters (grid dims, block dims, shared memory size, arguments) into the GPU command queue and returns immediately.

```
cudaLaunchKernel(
    kernel_fn,           ← which kernel to run
    gridDim=(4281,1,1),  ← how many thread blocks
    blockDim=(128,1,1),  ← threads per block
    args=[A, B, C, M, N, K],
    sharedMem=0,
    stream=stream0
)
→ returns in ~2–5μs, GPU executes asynchronously
```

One `cublasLtMatmul` CPU call typically produces **multiple** `cudaLaunchKernel` entries — one for the main compute kernel and one or more for reduction kernels (e.g., `splitKreduc`).

**What to look for:** duration of each entry (~2–5μs is healthy). Gaps between consecutive entries = CPU overhead between kernel submissions = directly causes GPU idle in the CUDA HW row.

---

### `splitKreduc` — Split-K Reduction Kernel

When a matrix dimension (K) is large, cuBLAS splits the work across multiple SMs in parallel (Split-K strategy) and then runs a separate reduction kernel to sum the partial results.

```
cublasLtMatmul (C = A @ B, K=3072):
  cudaLaunchKernel → main compute kernel   (each SM handles K/n chunk)
  cudaLaunchKernel → splitKreduc           (sum partial results across SMs)
```

Visible as a second `cudaLaunchKernel` entry immediately after the main kernel launch for the same matmul. The `splitKreduc` kernel itself is short — it only sums partial outputs, not the full matmul.

**When it appears:** MLP layers with large K dimensions (fc1: K=768→3072, fc2: K=3072→768). Not present for smaller attention projection matmuls.

---

### `vectorized_elementwise_kernel` — labeled as `vectorized_e...`

PyTorch's fused elementwise kernel for pointwise operations: GELU activation, dropout, residual add, bias add (when not fused inside cuBLAS). Launched directly via `cudaLaunchKernel` — no cuBLAS involved.

```
# After MLP fc1:
hidden = F.gelu(hidden)   → cudaLaunchKernel(vectorized_elementwise_kernel, ...)

# Residual add:
x = x + residual          → cudaLaunchKernel(vectorized_elementwise_kernel, ...)
```

**Why "vectorized":** Each thread loads 4 floats at once using a 128-bit float4 instruction instead of 4 separate 32-bit loads — the vectorized loads tactic. This maximizes HBM bandwidth per thread for memory-bound pointwise ops:

```
Non-vectorized thread:          Vectorized thread:
  load x[i]      (32-bit)         load x[i:i+4]   (128-bit, 1 instruction)
  compute f(x[i])                  compute f on all 4
  store y[i]     (32-bit)         store y[i:i+4]  (128-bit, 1 instruction)
                                → 4× elements per thread per instruction
```

The kernel name itself encodes this — "vectorized" = float4 loads, "elementwise" = one independent op per element, "kernel" = single GPU kernel covering the full op chain (GELU, add, scale all fused into one pass).

**Not a cuBLAS call:** PyTorch launches this directly via `cudaLaunchKernel`. No entry appears in the cuBLAS row — only in the CUDA API row and CUDA HW row.

**Where it appears in the timeline:**
```
cublasLtMatmul  (Q projection)
vectorized_e... (attention score scaling: S / sqrt(d_k))
cublasLtMatmul  (attention output projection)
vectorized_e... (residual add: x + attention_out)
cublasLtMatmul  (MLP fc1)
vectorized_e... (GELU activation)
cublasLtMatmul  (MLP fc2)
vectorized_e... (residual add: x + mlp_out)
```

**What to look for:** appears between cuBLAS matmul launches. Should be fast (~2–10μs at batch=1 — tensors are small vectors). If slow, indicates large intermediate tensors or missing fusion.

---

### `CatArrayBatchedCopy_aligned16_Config` — Tensor Concatenation

PyTorch's internal CUDA kernel for `torch.cat()`. Copies multiple source tensors into a single pre-allocated output tensor in one kernel launch.

**Name breakdown:**
```
Cat          → torch.cat() operation
ArrayBatched → processes multiple source tensors in one kernel launch
Copy         → copies src tensors into pre-allocated destination
aligned16    → 16-byte (128-bit) aligned loads — float4 vectorized access
Config       → template parameter (dtype, alignment configuration)
```

**Where it appears in GPT-2 — KV cache concatenation:**

Every decode step, the new K and V vectors are appended to the growing KV cache:

```python
# Inside GPT-2 attention, every decode step:
key   = torch.cat([past_key,   new_key],   dim=-2)  → CatArrayBatchedCopy
value = torch.cat([past_value, new_value], dim=-2)  → CatArrayBatchedCopy
```

**Why ArrayBatched — one kernel launch for N inputs:**

```
One kernel launch receives array of descriptors:
  src[0] = past_key ptr,  dst_offset=0,         size=step × head_dim
  src[1] = new_key ptr,   dst_offset=step×dim,  size=1 × head_dim

Each thread block handles one src tensor's copy in parallel.
→ N inputs = 1 kernel launch instead of N launches
```

**Growing cost per decode step:**

```
Step  1: cat([1 token],  [1 new]) → copy  2 × head_dim bytes
Step 10: cat([10 tokens],[1 new]) → copy 11 × head_dim bytes
Step 49: cat([49 tokens],[1 new]) → copy 50 × head_dim bytes
```

KV cache copy grows linearly with sequence position — by token 50 each attention layer copies 50× more than at step 1. Across 12 GPT-2 layers × 2 (K and V) = 24 cat operations per decode step, each growing. This is one reason long sequences get slower per token.

**Not a cuBLAS call:** launched directly via `cudaLaunchKernel` — no entry in cuBLAS row, only in CUDA API and CUDA HW rows.

**PagedAttention eliminates this:** vLLM keeps KV cache in fixed non-contiguous memory pages — no cat needed, new K/V written directly into the next free page slot. The `CatArrayBatchedCopy` entries disappear entirely in vLLM profiles.

**What to look for:** appears twice per transformer layer per decode step (once for K, once for V). Duration grows with step number — if you zoom into late decode steps vs early ones, this kernel takes noticeably longer.

---

### `cudaMemcpyAsync` — Async Memory Copy

Non-blocking memory transfer between CPU and GPU. Returns immediately; transfer happens in background on a DMA engine.

```
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream)
→ DMA engine handles transfer, CPU continues
→ GPU kernels in same stream wait for transfer to complete before running
```

**H2D (Host→Device):** input token IDs transferred to GPU at the start of each prompt. Small (few KB for a handful of tokens).

**D2H (Device→Host):** output token IDs transferred back to CPU after generation completes. This is what triggers the inter-prompt poll/sync we saw in the OS Runtime row.

**What to look for:** H2D entries at the start of each prompt, D2H entries at the end. Duration should be tiny for token ID tensors (few KB). Long D2H = large output being copied back.

---

### `cudaStreamSynchronize` / `cudaDeviceSynchronize` — Hard Sync

Blocks the CPU thread until all previously submitted work in the stream (or all streams) has completed on the GPU.

```
cudaDeviceSynchronize()
→ CPU blocks here until GPU finishes all queued work
→ manifests as ioctl + nanosleep + poll in OS Runtime row
→ visible as a long gap in the CUDA API row itself
```

**When it appears:** at the end of `model.generate()` — PyTorch must synchronize before returning output tensors to Python. This is the root cause of the 74ms poll blocks observed in the OS Runtime row (Observation 7 in os_runtime findings).

**What to look for:** long duration entries (tens to hundreds of ms). Each one = a prompt boundary where CPU waited for GPU. Should appear exactly N times for N prompts.

---

## Row Relationships and Rules of Thumb

### cuBLAS row → CUDA API row (containment)

The `cudaLaunchKernel` entry in the CUDA API row is nested INSIDE the cuBLAS block. cuBLAS does setup first, then calls `cudaLaunchKernel` as its last step before returning:

```
cuBLAS row:  [GetHeuristic][----------cublasLtMatmul----------]
                                 setup...        [cudaLaunchKernel]
CUDA API row:                                    ↑ nested here, near end of cuBLAS block
```

- cuBLAS block is **wider** — includes algorithm selection, workspace setup, then launch
- CUDA API block is **narrower** — just the `cudaLaunchKernel` call (~2–5μs), starting near the end of the cuBLAS block
- Rule: **CUDA API block always falls within the cuBLAS block time span (parent → child)**

---

### CUDA API row → NVTX row (temporal ordering)

NVTX (GPU execution) always starts AFTER `cudaLaunchKernel` completes. CPU submits the kernel to the command queue and returns; GPU picks it up asynchronously:

```
CUDA API row:  [cudaLaunchKernel, ~3μs]
                                   ↓  async gap (few μs — GPU picks up from command queue)
NVTX row:                            [kernel executes, 13–22μs]
```

- Rule: **NVTX block always starts after its corresponding CUDA API block ends (cause → effect)**
- The gap between CUDA API end and NVTX start = async dispatch latency (command queue propagation)

---

### When NVTX Appears Before the Current CUDA API Block

This does NOT break the rule — it means the NVTX block belongs to a **different (earlier) operation**. This is the pipelined case:

```
Matmul N:    earlier CUDA API [cudaLaunchKernel] ──────→ NVTX [GPU executing matmul N]
                                                                 ↑ still running
Matmul N+1:  cuBLAS [GetHeuristic][cublasLtMatmul → cudaLaunchKernel]
                                                              ↓
                                                   NVTX [will execute matmul N+1 next]
```

GPU is executing matmul N (from a previous dispatch) while CPU is simultaneously preparing and dispatching matmul N+1. The CUDA stream is a command queue — GPU drains it independently. This is natural pipelining within a decode step.

---

### Rules of Thumb — Reading the Three Rows Together

| Observation | What it means |
|---|---|
| CUDA API block falls inside cuBLAS block | Normal — cudaLaunchKernel is the last step of cuBLAS dispatch |
| NVTX starts just after CUDA API ends | Normal — async gap between CPU submit and GPU start |
| NVTX overlaps with the NEXT cuBLAS/CUDA API block | Pipelining — GPU running op N while CPU dispatches op N+1 |
| CUDA API block with no corresponding NVTX nearby | Kernel queued but GPU hasn't started yet — command queue backed up |
| Long gap between consecutive CUDA API blocks | CPU overhead between dispatches → GPU idle in CUDA HW row |
| `cudaDeviceSynchronize` long block in CUDA API | Prompt boundary — CPU blocked until GPU finishes all queued work |

---

## Layer Stack — How All Rows Connect

```
One linear layer (e.g., Q projection: 768×768):

cuBLAS row:   [cublasLtMatMulAlgoGetHeuristic][cublasLtMatmul, 17–21μs]
                                                      ↓ calls
CUDA API row:                              [cudaLaunchKernel, ~3μs][splitKreduc, ~3μs]
                                                      ↓ async submit
CUDA HW row:                                                [gemv2T_kernel, 13–22μs]

One elementwise op (GELU after fc1):

cuBLAS row:   (no entry — not a matmul)
CUDA API row: [cudaLaunchKernel(vectorized_elementwise), ~2μs]
                    ↓ async submit
CUDA HW row:                   [vectorized_elementwise_kernel, ~3μs]

Prompt boundary sync:

CUDA API row: [cudaDeviceSynchronize, 74ms]   ← CPU blocked
OS Runtime:   [poll + ioctl + nanosleep, 74ms] ← same event, OS level view
CUDA HW row:  [GPU executing last kernels of prompt...]
```

---

## Key Takeaways

| What you see | What it means |
|---|---|
| Multiple `cudaLaunchKernel` per cuBLAS call | cuBLAS decomposes one matmul into compute + reduction kernels |
| Gaps between `cudaLaunchKernel` entries | CPU overhead between dispatches → GPU idle in CUDA HW row |
| `splitKreduc` after main kernel | MLP layer (large K) — Split-K strategy with separate reduction pass |
| `vectorized_elementwise` entries | Pointwise ops (GELU, residual add) — short, memory-bound |
| `cudaDeviceSynchronize` long block | Prompt boundary — CPU waiting for GPU, same event as OS Runtime poll block |
| H2D `cudaMemcpyAsync` at prompt start | Input token IDs transferred to GPU (tiny, few KB) |
| D2H `cudaMemcpyAsync` at prompt end | Output tokens copied back to CPU — triggers sync |
