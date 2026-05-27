# Profiling Guide — Hardware Baselines and Heuristics

This guide is a living reference. Hardware specs are authoritative NVIDIA numbers.
Experiment baselines are filled in from actual runs on Lambda Labs instances.

---

## FLOPS

FLOPS = Floating Point Operations Per Second — how many floating point math operations the hardware can execute per second
T = Tera = 10¹² = 1 trillion
312 TFLOPS = 312 trillion floating point operations per second

  And the precision qualifier matters because different precisions use different hardware paths:

  FP32  →  CUDA cores        →  19.5 TFLOPS  (A100)
  TF32  →  Tensor Cores      →  156 TFLOPS   (A100)
  FP16  →  Tensor Cores      →  312 TFLOPS   (A100)
  INT8  →  Tensor Cores      →  624 TOPS     (A100)

## Hardware Specs — A100 vs H100

### NVIDIA A100 SXM4 40GB (Ampere, GA100)
*Lambda Labs: $1.99/hr — this is our experiment hardware*

| Property | Value |
|---|---|
| Architecture | Ampere (GA100) |
| SMs | 108 |
| CUDA Cores | 6,912 |
| Tensor Cores | 432 (3rd gen) |
| **FP16 Tensor Core** | **312 TFLOPS** (no sparsity) / 624 TFLOPS (structured sparsity) |
| BF16 Tensor Core | 312 TFLOPS (no sparsity) |
| TF32 Tensor Core | 156 TFLOPS (no sparsity) |
| FP8 | Not supported (Ampere) |
| INT8 | 624 TOPS (no sparsity) |
| INT4 | 1,248 TOPS (no sparsity) |
| FP64 | 19.5 TFLOPS |
| **HBM** | **40 GB** HBM2 |
| **Memory Bandwidth** | **1,555 GB/s (~1.6 TB/s)** |
| L2 Cache | 40 MB |
| Shared Mem / SM | 192 KB (configurable L1 + shared) |
| NVLink | 3.0 — 600 GB/s bidirectional |
| PCIe | Gen 4 — 64 GB/s |
| TDP | 400W |

### NVIDIA H100 SXM5 80GB (Hopper, GH100)
*Lambda Labs: ~$3.99/hr — next target hardware*

| Property | Value |
|---|---|
| Architecture | Hopper (GH100) |
| SMs | 132 |
| CUDA Cores | 16,896 |
| Tensor Cores | 528 (4th gen) |
| **FP16 Tensor Core** | **989 TFLOPS** (no sparsity) / 1,979 TFLOPS (structured sparsity) |
| BF16 Tensor Core | 989 TFLOPS (no sparsity) |
| TF32 Tensor Core | 494 TFLOPS (no sparsity) |
| **FP8 Tensor Core** | **1,979 TFLOPS** (no sparsity) — new in Hopper, the inference standard |
| INT8 | 1,979 TOPS (no sparsity) |
| FP64 | 67 TFLOPS |
| **HBM** | **80 GB** HBM3 |
| **Memory Bandwidth** | **3,350 GB/s (~3.35 TB/s)** |
| L2 Cache | 50 MB |
| Shared Mem / SM | 256 KB |
| NVLink | 4.0 — 900 GB/s bidirectional |
| PCIe | Gen 5 — 128 GB/s |
| TDP | 700W |

---

## A100 vs H100 — Side by Side

| Metric | A100 SXM4 40GB | H100 SXM5 80GB | H100 / A100 ratio |
|---|---|---|---|
| FP16 TFLOPS | 312 | 989 | **3.2x** |
| FP8 TFLOPS | — | 1,979 | ∞ (new capability) |
| HBM capacity | 40 GB | 80 GB | **2x** |
| Memory bandwidth | 1.6 TB/s | 3.35 TB/s | **2.1x** |
| SMs | 108 | 132 | 1.2x |
| NVLink bandwidth | 600 GB/s | 900 GB/s | 1.5x |
| TDP | 400W | 700W | 1.75x |

**Key insight:** H100 is 3.2x faster on compute but only 2.1x faster on memory bandwidth.
The ridge point shifts right — kernels need higher arithmetic intensity to be compute-bound on H100.

---

## Ridge Point — Roofline Threshold

The ridge point is the arithmetic intensity (FLOPs/byte) above which a kernel becomes compute-bound.
Below it, scaling up compute doesn't help — memory bandwidth is the ceiling.

```
Ridge point = Peak TFLOPS / Memory Bandwidth

A100:  312 TFLOPS / 1.555 TB/s = ~200 FLOPs/byte
H100:  989 TFLOPS / 3.350 TB/s = ~295 FLOPs/byte
```

**What this means in practice:**

| Op | Arithmetic Intensity | A100 | H100 |
|---|---|---|---|
| aten::mm (batch=1, d=768) | ~44 FLOPs/byte | memory-bound | memory-bound |
| aten::mm (batch=64, d=768) | ~384 FLOPs/byte | compute-bound | compute-bound |
| aten::softmax | ~2 FLOPs/byte | memory-bound | memory-bound |
| aten::layer_norm | ~4 FLOPs/byte | memory-bound | memory-bound |

Elementwise ops are always far below the ridge point — they do 1-4 FLOPs per byte loaded.
Large matmuls at batch=64+ exceed the ridge point on both GPUs.

---

## Profiling Heuristics — What to Look For

### CPU Dispatch Time per Op

| Value | Verdict |
|---|---|
| < 50μs (0.05ms) | Normal eager-mode dispatch overhead |
| 50μs – 500μs | Investigate — Python loop overhead, slow host-side prep |
| > 500μs (0.5ms) | Red flag — GIL contention, pageable memory blocking |

CPU dispatch for `aten::mm` should be ~5–20μs in healthy eager mode.

### CUDA Kernel Time per Call (A100, GPT-2 scale)

| Op | batch=1 | batch=16 | batch=64 |
|---|---|---|---|
| `aten::mm` | 0.01–0.05ms | 0.1–0.5ms | 0.5–3ms |
| `aten::softmax` | 0.005–0.02ms | 0.02–0.1ms | 0.1–0.5ms |
| `aten::layer_norm` | 0.005–0.02ms | 0.02–0.1ms | 0.1–0.5ms |

Elementwise ops should be 5–20x shorter than matmul at the same batch size.
If they approach matmul CUDA time, memory bandwidth is saturated.

### CPU time / CUDA time Ratio

```
Ratio > 1  (CPU > CUDA): dispatch overhead dominates — batch too small, kernel too tiny
Ratio ~0.01: healthy — GPU is doing real work, CPU dispatch is negligible
Ratio >> 1 with large absolute CPU time: severe CPU bottleneck — investigate dispatch path
```

### CUDA % Distribution (sorted by cuda_time_total)

| Pattern | Verdict |
|---|---|
| mm + addmm + baddbmm > 70% CUDA % | Compute-dominated — healthy at large batch |
| Elementwise (softmax, gelu, layer_norm) > 30% CUDA % | Memory bandwidth pressure — consider op fusion |
| No single op > 20%, everything flat | Overhead-dominated — batch too small, kernel launch noise |

### Master Utilization Heuristic

```
GPU utilization % = (tokens/sec × FLOPs/token) / GPU peak TFLOPS × 100

GPT-2 117M on A100 at batch=1:
  ~400 tokens/sec × 0.24 GFLOP/token = 96 GFLOP/sec
  96 GFLOP/sec / 312,000 GFLOP/sec = 0.03%  ← almost nothing

GPT-2 117M on A100 at batch=64 (target):
  ~8,000 tokens/sec × 0.24 GFLOP/token = 1,920 GFLOP/sec
  1,920 / 312,000 = 0.6%  ← still very low for a 117M model
  (GPT-2 is too small to stress A100 even at large batch)
```

If utilization < 1% → fix pipeline (batching, memory transfer) before touching kernels.
If utilization 1–30% → batching helps, consider continuous batching.
If utilization > 60% → well-utilized, optimize at kernel level (fusion, quantization).

---

## Why Batching Improves Throughput

### What batching actually does to the matrix

The GPU doesn't see "10 sequences running in parallel." It sees one bigger matrix:

```
batch=1:  matmul (50, 768) × (768, 768)  →  38,400 output elements
batch=10: matmul (500, 768) × (768, 768) →  384,000 output elements
```

The 10 sequences are stacked into more rows. The weight matrix `(768, 768)` is loaded
from HBM once and reused across all 500 rows instead of 50. More compute per byte loaded
= higher arithmetic intensity = closer to the ridge point.

### Unit of parallelism — it's not one thread per sequence

The GPU tiles the output matrix into small blocks and assigns each block to an SM:

| Unit | Size | What it does |
|---|---|---|
| Thread | 1 | Computes a few multiply-accumulates |
| Warp | 32 threads | All execute the same instruction on different data (SIMT) |
| Thread Block | multiple warps | Assigned to one SM, shares L1/shared memory |
| SM | runs thread blocks | 108 on A100 — the real parallel workers |
| Grid | all thread blocks | The full kernel launch |

No thread "knows" it's working on sequence 3 vs sequence 7 — it's computing a tile of
the output matrix that might span rows from multiple sequences. The GPU just sees rows.

```
A100: 108 SMs

batch=1:  38,400 output elements → ~12 thread blocks → 12 SMs busy, 96 SMs idle
batch=10: 384,000 output elements → ~120 thread blocks → all 108 SMs occupied
```

This is **SIMT** — Single Instruction, Multiple Threads. Same instruction (multiply-accumulate),
different data (different rows). Batching gives more rows → more tiles → more SMs occupied.

### Autoregressive generation with batching

```
batch=1, 10 prompts sequentially:
  ~50 forward passes per prompt × 10 prompts = ~500 forward passes
  Each pass: matmul (seq_len, 768) × (768, 768)  — small matrix

batch=10, all prompts together:
  ~50 forward passes total (all 10 sequences advance in lockstep)
  Each pass: matmul (10×seq_len, 768) × (768, 768)  — 10x larger matrix
  Same 500 tokens generated, but in 50 passes instead of 500
```

Weight matrix `(768, 768)` loaded from HBM once per step regardless of batch size.
At batch=10 it serves 10x more rows for the same HBM load → 10x better reuse.

### The padding cost

Variable-length sequences must be padded to the longest in the batch:
```
["hi", "the quick brown fox jumped over"] → pad "hi" to match length
```
Padded tokens still go through the matmul — wasted compute. At large batch with diverse
lengths, padding waste can be 20-40%. This is why continuous batching (vLLM) improves
on static batching — it groups sequences of similar length and never pads across requests.

---

## GPT-2 Experiment Baselines — Fill In After Runs

### Phase 1: Baseline (batch=1, max_new_tokens=50, 50 prompts)

| Metric | Expected | Observed (A100) |
|---|---|---|
| Throughput (tokens/sec) | 300–600 | |
| Avg latency (ms/request) | 80–150ms | |
| Top op by CUDA % | aten::mm | |
| aten::mm CUDA time per call | 0.01–0.05ms | |
| aten::mm CPU time per call | 0.005–0.02ms | |
| Self Mem (aten::mm) | ~9 MB total | |
| GPU utilization % | ~0.03% | |

### Phase 2: Batching (batch = 1, 4, 16, 32, 64)

| Batch | Throughput (tok/s) | Latency (ms/req) | aten::mm CUDA/call | CUDA % mm+addmm+baddbmm |
|---|---|---|---|---|
| 1 | | | | |
| 4 | | | | |
| 16 | | | | |
| 32 | | | | |
| 64 | | | | |

Expected: throughput scales super-linearly with batch (batching efficiency), latency grows sub-linearly.
Compute % of matmuls should rise with batch — approaching 70%+ at batch=64.

---

## Warmup Principle

**Warmup data must match production data in the dimensions that affect GPU behavior.**

For LLMs those dimensions are: sequence length and batch size.

### Why a dummy 1-token warmup is insufficient

```python
# BAD — "warmup" = 1 BPE token, production prompts = 50+ tokens
dummy = tokenizer("warmup", return_tensors="pt").to(DEVICE)
_ = model.generate(**dummy, max_new_tokens=10)
```

The GPU reaches a different steady state at seq_len=1 vs seq_len=50:
- **cuBLAS algorithm selection**: matmul tile configurations are chosen per matrix dimensions.
  seq_len=1 → tiny matrices → different algorithm than seq_len=50.
  First real profiled request re-triggers selection → anomalous latency spike.
- **PyTorch allocator pool**: sized to the warmup workload. First real request may trigger
  new `cudaMalloc` calls not seen during warmup → allocation cost leaks into profiled results.
- **CUDA kernel JIT**: kernel variants are compiled per (dtype, shape class). A 1-token warmup
  may not compile all variants used by longer sequences.

### Correct warmup

```python
# GOOD — representative prompt, 3 passes
warmup_prompt = prompts[0]   # same length distribution as actual profiling data
with torch.no_grad():
    for _ in range(3):       # 3 passes to reach full steady state
        dummy = tokenizer(warmup_prompt, return_tensors="pt").to(DEVICE)
        _ = model.generate(**dummy, max_new_tokens=MAX_NEW_TOKENS)
```

**Why 3 passes:**
- Pass 1: CUDA JIT kernel compilation
- Pass 2: cuBLAS algorithm selection settles + allocator pool fills to steady size
- Pass 3: fully steady state — this is what the profiler will measure

### Applies beyond profiling

In production serving, the same principle applies when loading a model:
warm up with representative traffic before routing real requests. A cold model
pays JIT/algorithm-selection costs on the first few requests → latency spikes
that don't reflect actual steady-state throughput.

---

## What Each Phase Teaches

| Phase | What you're measuring | Key question |
|---|---|---|
| Phase 1 — Baseline | Raw single-request performance | Where does time go? Which ops dominate? |
| Phase 2 — Batching | Throughput vs latency tradeoff | At what batch does GPU become utilized? |
| Phase 3 — Nsight | Kernel-level occupancy, arithmetic intensity | Are we hitting the roofline? SM occupancy? |
| Phase 4 — Compilers | torch.compile + CUDA graphs impact | How much does dispatch overhead cost? |
| Phase 5 — Quantization | INT8/FP8 throughput improvement | Does halving dtype double throughput? |
| Phase 6 — Kubernetes | Serving at scale | Does theoretical throughput hold under load? |
