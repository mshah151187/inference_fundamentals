# GPU Memory Usage Metric in torch.profiler

## What `profile_memory=True` Actually Measures

When you enable `profile_memory=True` in `torch.profiler`, it tracks **HBM allocation and deallocation events** — specifically, how many bytes PyTorch's caching allocator carved out of GPU VRAM (HBM) for each op's output and intermediate tensors.

It does **not** measure:
- How much data was moved from HBM → SM during kernel execution (that's memory bandwidth)
- L1/L2 cache usage
- Register pressure

---

## GPU Memory Hierarchy (Where Things Actually Live)

```
┌─────────────────────────────────────────────────────────┐
│                  GPU CHIP                               │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │   SM 0   │  │   SM 1   │  │   SM N   │  ...        │
│  │ L1 / SMEM│  │ L1 / SMEM│  │ L1 / SMEM│             │
│  │ Registers│  │ Registers│  │ Registers│             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
│       └──────────────┴─────────────┘                   │
│                      │                                  │
│               ┌──────┴──────┐                          │
│               │   L2 Cache  │  (shared across SMs)     │
│               └──────┬──────┘                          │
│                      │                                  │
└──────────────────────┼──────────────────────────────────┘
                       │  memory bus (~2 TB/s on A100)
               ┌───────┴──────┐
               │     HBM      │  ← PyTorch tensors live here
               │  (40GB VRAM) │  ← nvidia-smi shows usage here
               └──────────────┘
```

| Layer | Size (A100) | Speed | What lives here |
|---|---|---|---|
| Registers | ~256KB/SM | ~20 TB/s | Active computation values, loop counters |
| L1 / Shared Memory | ~192KB/SM | ~10 TB/s | Tiles of matrices being multiplied |
| L2 Cache | 40MB total | ~5 TB/s | Recently accessed HBM data |
| HBM | 40GB | ~2 TB/s | All `torch.Tensor` objects — weights, activations, KV cache |

**Key point:** every `torch.Tensor` you create lives in HBM. When a CUDA kernel runs, it reads from HBM into SM registers/shared memory, computes, and writes results back to HBM.

---

## The Sequence of Events During Inference

```
model.to("cuda")
  → PyTorch allocates HBM blocks for all weight tensors
  → DMA copy: CPU DRAM → GPU HBM
  → Weights now sit idle in HBM (548MB for GPT-2)

tokenizer(prompt)
  → Pure CPU operation
  → Returns input_ids tensor in CPU DRAM

inputs.to("cuda")
  → DMA copy: CPU DRAM → GPU HBM
  → input_ids now in HBM (tiny — just token ids, e.g. 50 × 4 bytes)

model.generate(**inputs)
  → For each forward pass step:
       1. CUDA kernel launched (e.g., aten::mm for a linear layer)
       2. Kernel reads weight matrix from HBM → SM tiled blocks
       3. Kernel reads input activation from HBM → SM tiled blocks
       4. Tensor cores compute in registers
       5. Result written back to HBM as a NEW tensor (output activation)
       6. PyTorch allocator records: "aten::mm allocated X bytes in HBM"
```

Step 6 is what `profile_memory=True` captures.

---

## Concrete Example: What Counts as "Memory Allocated" for aten::mm

`aten::mm(A, B)` computes `C = A @ B`.

```
A: shape (seq_len, d_model)  = (50, 768)   → 50*768*2 bytes = 76,800 bytes  (FP16)
B: shape (d_model, d_model)  = (768, 768)  → 768*768*2 bytes = 1,179,648 bytes
C: shape (seq_len, d_model)  = (50, 768)   → 50*768*2 bytes = 76,800 bytes  ← OUTPUT
```

**What profile_memory counts as allocated by aten::mm: only C — 76,800 bytes.**

A and B are inputs — they were allocated by whoever created them (a previous op, or model init). The op that *creates* a tensor is charged for it. `aten::mm` creates C, so C's allocation is attributed to `aten::mm`.

**A and B are not "allocated" by this op** — they're already in HBM, passed in as arguments.

---

## Where "Intermediate Tensors" Actually Come From

In a full transformer forward pass (GPT-2 self-attention block), each op creates one or more new tensors in HBM:

```
Input activation X: (seq_len, d_model) = (50, 768)
Already in HBM from previous layer.

── Q projection: aten::mm(X, W_Q) ──────────────────────────
   Allocates: Q = (50, 768) = 76,800 bytes        ← intermediate

── K projection: aten::mm(X, W_K) ──────────────────────────
   Allocates: K = (50, 768) = 76,800 bytes        ← intermediate

── V projection: aten::mm(X, W_V) ──────────────────────────
   Allocates: V = (50, 768) = 76,800 bytes        ← intermediate

── Attention scores: aten::baddbmm(Q, K^T) ─────────────────
   Allocates: scores = (n_heads, seq_len, seq_len)
            = (12, 50, 50) = 60,000 bytes         ← intermediate

── aten::softmax(scores) ────────────────────────────────────
   Allocates: attn_weights = (12, 50, 50) = 60,000 bytes  ← intermediate

── Context: aten::bmm(attn_weights, V) ─────────────────────
   Allocates: context = (50, 768) = 76,800 bytes  ← intermediate

── Output projection: aten::mm(context, W_O) ───────────────
   Allocates: out = (50, 768) = 76,800 bytes      ← output of attention block
```

Every arrow above is a new HBM allocation. `profile_memory` attributes each one to the op that created it.

**At peak (during aten::baddbmm):** Q, K, V, scores are all live in HBM simultaneously. Once scores are consumed by softmax, Q and K can be freed. This is why peak memory > sum of individual allocations.

---

## What profile_memory Does NOT Measure: Memory Bandwidth

When `aten::mm` executes, the kernel must physically move A and B from HBM into SM before it can compute. For GPT-2 at batch=1:

```
aten::mm(X, W_Q):
  X:   50 × 768 × 2 bytes =   76,800 bytes read from HBM
  W_Q: 768 × 768 × 2 bytes = 1,179,648 bytes read from HBM
  C:   50 × 768 × 2 bytes =    76,800 bytes written to HBM
  ─────────────────────────────────────────────────────────
  Total bandwidth: ~1.33 MB per call
  Multiply by 2400 calls (all linear layers, all steps) = ~3.2 GB moved
```

This movement is **memory bandwidth** — measured in GB/s, not bytes allocated.
`profile_memory` doesn't see this. It only sees the 76,800-byte allocation of C.

To see bandwidth: use **Nsight Compute** (`ncu`) — it reports `dram__bytes_read` and `dram__bytes_write` per kernel with actual GB/s figures.

---

## The Two Concepts Side by Side

| Concept | What it measures | Unit | Why it matters | Tool |
|---|---|---|---|---|
| Memory allocation | New HBM tensors created by an op | bytes | VRAM pressure, OOM risk | `profile_memory=True` in torch.profiler |
| Memory bandwidth | HBM ↔ SM data movement during kernel | GB/s | Memory-bound vs compute-bound | Nsight Compute |

These terms describe which resource becomes the bottleneck *as you scale work up* — not necessarily what's happening at your current operating point. The right framework is the **roofline model**:

```
Performance
(TFLOPS)     compute ceiling (312 TFLOPS FP16 on A100)
             ─────────────────────────────────────────  ← compute-bound above ridge
            /
           /  ← memory-bound below ridge (slope = bandwidth)
          /
         /─────── ridge point = 156 FLOPs/byte (312T / 2T/s)
        /
       ──────────────────────────────────────────────→
                                      Arithmetic Intensity (FLOPs/byte)
```

**Memory-bound** (arithmetic intensity < ridge point, AND near the bandwidth ceiling):
- Memory bus is saturated feeding data to tensor cores
- Tensor cores finish quickly and wait for the next load
- Example: large embedding lookups, elementwise ops (softmax, gelu, layer_norm) at large batch
- Nsight Compute: `dram__bytes_read` near 2 TB/s; low FLOP utilization

**Compute-bound** (arithmetic intensity > ridge point, AND near the compute ceiling):
- Tensor cores are the bottleneck — enough data, kernel is running at near-peak TFLOPS
- Example: large matmuls (batch 64+, large d_model) — weights loaded once, reused across many batch rows
- Nsight Compute: tensor core utilization near 100%; memory bandwidth well below ceiling

**Low occupancy / insufficient parallelism** (batch=1 small models — NEITHER ceiling hit):
- The issue is not that the memory bus is saturated, and not that tensor cores are starving for data
- There simply isn't enough parallel work to fill all 108 SMs on A100
- Matrix tiles are too small (seq_len=50, d_model=768) — most SMs sit idle for lack of warps
- Kernel launch overhead becomes a significant fraction of total wall time
- Arithmetic intensity for aten::mm at batch=1: ~44 FLOPs/byte → below ridge point (156)
  → *would be* memory-bound if you scaled up, but right now you're far below both ceilings
- Fix: increase batch size until SMs are occupied, then you hit the actual memory or compute wall

**Practical implication for profiling:**
`profile_memory=True` tells you which ops allocate the most HBM — useful for OOM debugging and understanding peak VRAM. It cannot tell you whether an op is memory-bound or compute-bound. For that: Nsight Compute, which reports arithmetic intensity and roofline position per kernel.

---

## PyTorch Caching Allocator — Why Freed ≠ Returned to OS

PyTorch does not call `cudaFree` every time a tensor is deleted. Instead it maintains a **memory pool**:

```
First run:   cudaMalloc(500MB) ← actual OS allocation
             split into blocks → Q(76KB), K(76KB), scores(60KB) ...
After op:    tensor freed → block returned to pool (NOT to OS)
Next op:     reuses block from pool → no cudaMalloc needed
```

This is why:
- `nvidia-smi` shows VRAM usage stays high even after tensors are freed
- `profile_memory` reports "freed" when tensors return to pool, not when OS reclaims memory
- Actual VRAM used by the process = the high-water mark of the pool

`torch.cuda.memory_allocated()` = bytes currently in use by live tensors
`torch.cuda.memory_reserved()` = bytes held by the pool (always >= allocated)
