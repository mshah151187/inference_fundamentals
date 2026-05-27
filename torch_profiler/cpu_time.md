# CPU Time in torch.profiler — How It's Measured and What It Means

## The Core Question

When you see `CPU time: 5ms` for `aten::mm` in the profiler output, the natural reaction is:
"But the matmul runs on the GPU — what is the CPU doing for 5ms?"

The answer: the CPU never computes the matmul. CPU time measures **dispatch overhead** — the time
the CPU spends submitting work to the GPU, not executing it.

---

## What Happens When Python Calls `torch.mm(A, B)`

```
Python: torch.mm(A, B)
            │
            ▼
   PyTorch Dispatcher (C++ layer on CPU)    ← profiler hook fires: record start time
   ┌─────────────────────────────────────┐
   │ 1. Resolve op: mm + CUDA backend    │
   │ 2. Allocate output tensor C in HBM  │
   │ 3. Call cudaLaunchKernel()          │  ← submits kernel to GPU command queue
   │ 4. cudaLaunchKernel() returns       │  ← immediately, without waiting for GPU
   └─────────────────────────────────────┘
            │
            ▼                               ← profiler hook fires: record end time
   Back to Python                           = CPU time for aten::mm (microseconds)

   Meanwhile, asynchronously on GPU:
   ┌─────────────────────────────────────────────────────────┐
   │ CUDA event START                                        │
   │   kernel executes (actual matmul on tensor cores)       │
   │ CUDA event END                                          │
   │ cudaEventElapsedTime(start, end) = CUDA time            │
   └─────────────────────────────────────────────────────────┘
```

**Key point:** `cudaLaunchKernel()` is asynchronous. It drops the work into the GPU's command queue
and returns immediately. The CPU has already moved on before the GPU even starts the kernel.

---

## How torch.profiler Captures Each

**CPU time:**
- Measured by hooks installed at the PyTorch dispatcher level
- Pre-hook fires when the op enters the dispatcher: `start = std::chrono::high_resolution_clock::now()`
- Post-hook fires when `cudaLaunchKernel()` returns: `end = now()`
- `CPU time = end - start` — wall-clock time on the CPU side

**CUDA time:**
- Measured by CUDA events inserted into the GPU stream around the kernel
- `cudaEventRecord(start_event, stream)` before kernel launch
- `cudaEventRecord(end_event, stream)` after kernel launch
- GPU timestamps these events with its own hardware clock as they pass through the stream
- `cudaEventSynchronize(end_event)` + `cudaEventElapsedTime(start, end)` = GPU execution time
- The CPU never knows the CUDA time until it explicitly queries the events

---

## The Two Timelines Can Overlap

Because `cudaLaunchKernel()` is async, the CPU dispatches kernel N+1 while the GPU is still
executing kernel N:

```
Time →
CPU:  [dispatch mm1][dispatch mm2][dispatch softmax][dispatch mm3] ...
GPU:       [──────── mm1 ─────────][──── mm2 ────][── softmax ──][── mm3 ──]
```

This is the ideal case — CPU stays ahead of GPU, GPU is never idle.
torch.profiler reports CPU time and CUDA time as separate measurements, not as a combined wall time.

---

## What CPU Time >> CUDA Time Means (Question 2 in torch_profiler.md)

```
aten::mm   CPU time: 5.00ms   CUDA time: 0.10ms   → CPU is the bottleneck
aten::mm   CPU time: 0.05ms   CUDA time: 34.5ms   → GPU is the bottleneck (expected)
```

When CPU time >> CUDA time:
- The GPU finishes kernel N in 0.1ms
- Then sits idle waiting for the CPU to dispatch kernel N+1
- The CPU is spending 5ms working through Python, the dispatcher, and all overhead before
  the next `cudaLaunchKernel()` call
- GPU is starving — not from lack of data, but lack of new work orders

```
Time →
CPU:  [──── dispatch mm1 (5ms) ────][── dispatch mm2 ──]
GPU:       [mm1 0.1ms][idle 4.9ms]   [mm2][idle]
```

The gap between CUDA kernels is GPU idle time caused by CPU dispatch latency.

---

## Why This Happens at Batch=1

GPT-2 at batch=1 generates ~2400 `aten::mm` calls per generate() call. Each kernel is tiny
(~0.01ms of actual GPU work at batch=1, seq=50, d=768). The Python → dispatcher → cudaLaunchKernel
roundtrip is ~5-50μs per op depending on Python overhead.

```
Kernel compute time:  0.01ms
Dispatch overhead:    0.05ms (5x the actual work!)
```

Most of the "wall time" is the CPU talking to itself, not the GPU computing.

---

## Common Causes of High CPU Time

| Cause | What happens | Fix |
|---|---|---|
| Python overhead per op | Each Python function call adds ~1-10μs | `torch.compile` — traces and eliminates Python dispatch |
| Pageable memory transfer | `inputs.to("cuda")` blocks CPU until transfer done | Pinned memory + `non_blocking=True` |
| Python GIL | Only one Python thread can dispatch at a time | Use C++ serving or multiple processes |
| Eager mode dispatch | Every op hits the dispatcher individually | `torch.compile` + CUDA graphs — fuse into one launch |

---

## CUDA Graphs — Eliminating CPU Dispatch Entirely

With CUDA graphs, the dispatch sequence is recorded once and replayed on the GPU:

```python
# Record phase (one-time):
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(inputs)   # CPU dispatches once, GPU records all kernels

# Replay phase (every request):
g.replay()   # GPU replays the entire sequence with zero CPU involvement
```

After `g.replay()`, the CPU time for all ops drops to near zero — no dispatcher, no Python,
no `cudaLaunchKernel()` calls. The GPU runs the full sequence from its own recorded command buffer.

Limitation: input tensor shapes must be identical across replays (graph is compiled for fixed shapes).
This is why CUDA graphs work well for fixed-batch inference but need special handling for
variable-length sequences (padding to fixed length or bucketing by length).

---

## Summary

| Metric | Measures | How captured | CPU or GPU clock |
|---|---|---|---|
| CPU time | Dispatch overhead (Python → dispatcher → cudaLaunchKernel return) | Dispatcher hooks (`std::chrono`) | CPU wall clock |
| CUDA time | Actual kernel execution on GPU | CUDA events in GPU stream | GPU hardware clock |

**Your data loading intuition is correct — but only for weights.** Weights are loaded once into HBM
and stay there. But for every forward pass, every single op still requires a CPU dispatch call.
The GPU holds the data; the CPU still has to issue the work order for each kernel.
