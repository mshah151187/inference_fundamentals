# Roofline Model — GPU Performance Analysis

Reference hardware: A100 SXM4 80GB  
Related: `nvidia_gpu_hardware.md`, `nsight_session_plan.md`

---

## The One Question Roofline Answers

**For this kernel — what is the hardware bottleneck?**

Every kernel is limited by one of two hardware ceilings:
1. **Memory bandwidth ceiling** — HBM can only move data so fast (1,555 GB/s on A100)
2. **Compute ceiling** — CUDA/Tensor cores can only do so many FLOPs/sec (19.5 or 312 TFLOPS)

Roofline places your kernel on a chart and tells you which ceiling (if any) you are near.

---

## Arithmetic Intensity — The Key Number

```
Arithmetic Intensity (AI) = FLOPs performed / Bytes moved from HBM
                            units: FLOPs/byte
```

This single number tells you how "compute-rich" a kernel is relative to its memory traffic.

**How to compute it for a matmul:**

```
addmm at batch=1, shape (1, 768) × (768, 768):

FLOPs = 2 × M × N × K  (2 because FMA = multiply + add)
      = 2 × 1 × 768 × 768
      = 1.18M FLOPs

Bytes moved from HBM:
  A matrix: M×K × 4 bytes = 1×768×4     = 3,072 bytes
  B matrix: K×N × 4 bytes = 768×768×4   = 2,359,296 bytes
  C output: M×N × 4 bytes = 1×768×4     = 3,072 bytes
  Total ≈ 2.37MB

AI = 1.18M / 2.37M ≈ 0.5 FLOPs/byte
```

**Intuition:** low AI = kernel mostly moves data and does little math per byte.
High AI = kernel does a lot of math per byte loaded.

---

## The Roofline Chart

```
 Achieved
 TFLOPS
   │
   │                                    ══════════════════ Compute ceiling
   │                          ══════════  312 TFLOPS (Tensor, FP16)
   │                ══════════            19.5 TFLOPS (CUDA, FP32)  ←─ our GPT-2 run
   │      ══════════
   │══════  ← slope = memory bandwidth (1,555 GB/s)
   │
   └─────────────────────────────────────────────── FLOPs/byte
   0    1   4   8  24                    156   200
                                          ↑
                                    ridge point
```

**The ridge point** is where the two ceilings meet:
```
Ridge point = Peak TFLOPS / Peak Memory Bandwidth

FP32 (CUDA cores):   19.5 TFLOPS / 1,555 GB/s ≈  13 FLOPs/byte
FP16 (Tensor cores): 312  TFLOPS / 1,555 GB/s ≈ 200 FLOPs/byte
```

Two ridge points because FP32 vs FP16 use different compute ceilings.
Our GPT-2 run is FP32 → ridge point ≈ 13 FLOPs/byte.

---

## Three Regions — Three Diagnoses

### Region 1 — Memory-Bandwidth Bound (left of ridge)

Kernel is below the memory bandwidth roofline slope.
Adding more compute units won't help — memory bus is the bottleneck.

```
Fix: reduce bytes moved
  → Fuse operations (fewer passes over HBM)
  → Quantize (INT8/FP8 = 2-4× fewer bytes per weight)
  → Tiling (keep data in SRAM, avoid re-loading from HBM)
  → FlashAttention (tiles QKV in SRAM, O(N) HBM vs O(N²) naive)
```

### Region 2 — Compute Bound (right of ridge)

Kernel is below the compute ceiling.
Adding memory bandwidth won't help — CUDA/Tensor cores are the bottleneck.

```
Fix: reduce FLOPs or increase compute throughput
  → Switch FP32 → FP16 (16× tensor core throughput)
  → Prune model (fewer weights = fewer FLOPs)
  → Knowledge distillation (smaller model)
  → Algorithmic improvements (sparse attention, linear attention)
```

### Region 3 — Occupancy Limited (below both ceilings)

Kernel is far below both the memory and compute ceilings.
Neither bus is saturated — the kernel is simply too small to fill the GPU pipeline.

```
Fix: increase work per kernel launch
  → Increase batch size (more rows in matmul → more threads → more SM utilization)
  → Fuse kernels (combine multiple small ops into one larger kernel)
  → torch.compile + CUDA graphs (eliminate launch overhead, keep GPU busy)
```

**This is exactly where batch=1 lives.** The matmul matrices are so small that
the GPU never ramps up before the kernel finishes.

---

## Where Our GPT-2 Kernels Land (batch=1, FP32)

```
Kernel                    FLOPs    Bytes     AI (FLOPs/byte)   Region
─────────────────────────────────────────────────────────────────────
addmm (1,768)×(768,768)   1.18M    2.37MB    0.5               Occupancy limited
native_layer_norm         ~0.1M    ~24KB     4                 Memory-bound (if at scale)
fmha_cutlass (attn)       ~1.2M    ~150KB    8                 Memory-bound (if at scale)
vectorized_elementwise    ~0.05M   ~12KB     4                 Memory-bound (if at scale)
```

FP32 ridge point ≈ 13 FLOPs/byte. All kernels are left of it — and at batch=1,
none saturate the memory bus either. All are occupancy-limited.

---

## How Roofline Shifts With Batch Size

```
                 FP32 ridge      FP16 ridge
                     ↓               ↓
batch=1:  ●  (0.5)                               ← occupancy limited
batch=32: ●●● (16)                               ← near FP32 ridge
batch=128:  ●●●●● (64)            ←              ← memory-bound (FP32)
batch=512:         ●●●●●●●● (256)       ←        ← compute-bound (FP16 only)
```

As batch increases:
- M dimension grows → AI grows linearly (more FLOPs, same weight bytes)
- Eventually hits memory bandwidth ceiling (batch=32–128 for FP32)
- FP16/BF16 + Tensor Cores push the compute ceiling 16× higher → need larger
  batches or sequence length to hit compute-bound

**The transition we expect to observe in Phase 2 batching experiments:**

| Batch size | Kernel type | Expected region |
|---|---|---|
| 1 | GEMV (1D grid) | Occupancy limited |
| 4–8 | GEMM (2D grid starts) | Approaching memory-bound |
| 32–64 | GEMM | Memory-bandwidth bound |
| 256+ (FP16) | GEMM | Near compute-bound |

---

## What Nsight Compute Reports to Confirm Region

| Metric | Occupancy limited | Memory-bound | Compute-bound |
|---|---|---|---|
| SM utilization % | < 5% | 20–60% | > 80% |
| HBM bandwidth % | < 15% | 70–90% | 20–40% |
| Warp occupancy % | < 10% | 30–60% | > 70% |
| Achieved TFLOPS | < 1 | 5–50 | > 200 (FP16) |
| Arithmetic intensity | < ridge | < ridge | > ridge |

For batch=1 we expect all metrics in the "occupancy limited" column — confirming
what torch.profiler and Nsight Systems told us, now with hardware-level evidence.

---

## The Roofline in Nsight Compute GUI

When you open an `.ncu-rep` file, the **Speed of Light** section shows:
- A roofline chart with your kernel's position plotted
- Two lines: memory roofline (diagonal) and compute roofline (horizontal)
- Your kernel plotted as a dot — you immediately see which ceiling is closer

The **Memory Workload Analysis** section shows HBM bandwidth utilization %.
The **Compute Workload Analysis** section shows SM throughput %.

Together they confirm which region your kernel is in and by how much.

---

## Connection to Optimization Tactics

Each region maps to a set of tactics (see `cuda_optimization_tactics.md`):

| Region | Primary tactics |
|---|---|
| Occupancy limited (batch=1) | Batching, kernel fusion, persistent kernels, CUDA graphs |
| Memory-bandwidth bound | Quantization, tiling, double buffering, vectorized loads, Flash Attention |
| Compute bound | Tensor Core WMMA, FP8, Split-K, Stream-K, register blocking |

The roofline tells you which tactics to apply. Applying memory tactics to a
compute-bound kernel (or vice versa) wastes engineering effort.
