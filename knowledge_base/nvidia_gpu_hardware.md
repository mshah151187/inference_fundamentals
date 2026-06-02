# NVIDIA GPU Hardware — CUDA Cores, Tensor Cores, Memory Hierarchy

---

## 1. The Two Types of Compute Units

Every NVIDIA GPU SM (Streaming Multiprocessor) has two distinct types of compute hardware:

| Unit | Purpose | Operates on |
|---|---|---|
| **CUDA Core** | General-purpose scalar FP/INT arithmetic | One multiply-add per clock, any dtype |
| **Tensor Core** | Matrix multiply only — D = A×B + C | FP16, BF16, INT8, FP8 (not FP32) |

They are physically separate silicon on the chip. A workload uses one or the other, not both simultaneously.

---

## 2. CUDA Cores

A CUDA core is a simple floating-point unit. Each clock cycle it executes one fused multiply-add (FMA):

```
result = a * b + c    ← one FMA = 2 FLOPs (one multiply + one add)
```

CUDA cores handle:
- FP32 (the default dtype for PyTorch models)
- FP64 (double precision, used in scientific computing)
- Integer arithmetic (INT32)
- Any computation that doesn't map to a matrix multiply

**A100 CUDA core throughput:** 19.5 TFLOPS (FP32)

This is the path your GPT-2 profiling run used — FP32 operations went through CUDA cores, not Tensor Cores.

---

## 3. Tensor Cores

Tensor Cores are specialized hardware units that compute a small matrix multiply-accumulate in a single clock cycle:

```
D = A × B + C

Where:
  A ∈ [16×16], dtype FP16/BF16
  B ∈ [16×16], dtype FP16/BF16
  C, D ∈ [16×16], dtype FP16/FP32 (accumulator)
```

One Tensor Core instruction does 16×16×16 = 4096 multiply-adds in the time a CUDA core does 1. That's where the throughput multiplier comes from.

**A100 Tensor Core throughput:** 312 TFLOPS (FP16/BF16)
**Ratio vs CUDA cores:** 312 / 19.5 = **16× more FLOPs per second**

This is why switching from FP32 to FP16 in your model unlocks a 16× compute advantage for matrix multiplications — and why Flash Attention V2 requires FP16/BF16 (see `flash_attention.md`).

### What Tensor Cores support by generation

| GPU Architecture | Tensor Core Gen | Supported dtypes |
|---|---|---|
| Volta (V100) | 1st gen | FP16 |
| Turing (T4) | 2nd gen | FP16, INT8, INT4 |
| Ampere (A100) | 3rd gen | FP16, BF16, INT8, TF32 |
| Hopper (H100) | 4th gen | FP16, BF16, INT8, FP8 |

**FP32 is never on this list.** FP32 always runs on CUDA cores.

### TF32 (Tensor Float 32) — the A100 special case

A100 introduced TF32: it takes FP32 inputs, rounds the mantissa to 10 bits (same as FP16), computes on Tensor Cores, and returns FP32 output. This gives ~8× speedup over FP32 CUDA cores with no code changes and minimal accuracy loss. PyTorch enables this by default on A100 for matmul.

---

## 4. SM — The Basic Compute Unit

A GPU is composed of many SMs (Streaming Multiprocessors). Each SM is an independent compute engine with its own:
- CUDA cores
- Tensor cores
- Shared memory (SRAM) — the fast on-chip memory FlashAttention tiles into
- Register file
- Warp schedulers

```
A100 GPU
├── 108 SMs
│   ├── 64 CUDA cores (FP32) per SM   → 108 × 64 = 6,912 total
│   ├── 4 Tensor Core units per SM    → 108 × 4  = 432 total
│   ├── 96 KB shared memory (SRAM)    ← FlashAttention's tile buffer lives here
│   └── 256 KB register file
└── Connected to HBM via memory controller
```

SM
  ├── Thread Block 1 (e.g., 256 threads)
  │   ├── warp 0 (32 threads) ─┐
  │   ├── warp 1 (32 threads)   ├── all can read/write the same 96 KB shared memory
  │   ├── warp 2 (32 threads)   │
  │   └── warp 3 (32 threads) ─┘
  └── Shared Memory (96 KB) ← visible to all warps in the block

All threads running on an SM share its SRAM. Threads are grouped into **warps** of 32, and all threads in a warp execute the same instruction in lockstep (SIMT — Single Instruction Multiple Threads).

---

## 5. Memory Hierarchy

This is the most important thing to understand for inference performance — nearly all bottlenecks are memory bottlenecks, not compute bottlenecks.

```
Fastest / Smallest                                   Slowest / Largest
    │                                                      │
    ▼                                                      ▼

Registers     Shared Mem (SRAM)    L2 Cache         HBM
~256 KB/SM       ~96 KB/SM         40 MB          80 GB (A100)
per thread      shared by SM     shared by GPU    off-chip DRAM

~10 TB/s        ~10 TB/s         ~5 TB/s        ~1.6 TB/s (A100)
                                               ~3.35 TB/s (H100)
```

### Registers
- Fastest storage — sub-nanosecond access
- Private to each thread — no sharing
- Where `rf` in `fmha_cutlassF_f32_aligned_64x64_rf_sm80` accumulates partial outputs

### Shared Memory (SRAM)
- On-chip, shared within an SM
- ~96 KB per SM on A100 — tiny but ~10× faster than HBM
- The key resource FlashAttention exploits: tiles Q, K, V into SRAM so the NxN attention matrix never goes to HBM
- Also called L1 cache (configurable split between L1 and shared memory)

### HBM (High Bandwidth Memory)
- The "GPU RAM" — where model weights, activations, and KV cache live
- Off-chip DRAM stacked near the GPU die
- 1.6 TB/s on A100, 3.35 TB/s on H100 — fast vs CPU RAM but slow vs SRAM
- **Most inference kernels are HBM-bandwidth-bound, not compute-bound**

---

## 6. A100 vs H100 — Key Numbers

| Spec | A100 (80GB) | H100 (80GB SXM) |
|---|---|---|
| Architecture | Ampere (sm_80) | Hopper (sm_90) |
| SMs | 108 | 132 |
| CUDA cores (FP32) | 6,912 | 16,896 |
| Tensor Core gen | 3rd | 4th |
| FP16 TFLOPS (Tensor) | 312 | 989 |
| FP32 TFLOPS (CUDA) | 19.5 | 67 |
| HBM capacity | 80 GB | 80 GB |
| HBM bandwidth | 1,555 GB/s | 3,350 GB/s |
| Min supported dtype (Tensor) | BF16/FP16 | FP8 (new on H100) |

H100's 4th gen Tensor Cores add **FP8** support — enabling even more aggressive quantization (see quantization.md when written).

---

## 7. How This Connects to Our Profiling Work

```
GPT-2 FP32 run:
  fmha_cutlassF_f32_aligned_64x64_rf_sm80
  └── f32 → cannot use Tensor Cores
  └── uses CUDA cores at 19.5 TFLOPS
  └── efficient_attention backend (xFormers/CUTLASS FMHA)
  └── still tiled (SRAM-based), just not tensor-core accelerated

After FP16 conversion:
  flash_fwd_kernel_...
  └── fp16 → Tensor Cores active at 312 TFLOPS (16× more)
  └── Flash Attention V2 backend selected
  └── additionally parallelizes across sequence length dimension
```

The 93ms attention kernel in the FP32 profiling run is largely irreducible at FP32 — the bottleneck is CUDA core compute throughput. FP16 is the lever.

---

## 8. Why Most Inference is Memory-Bandwidth Bound, Not Compute Bound

During decode (generating one token at a time):
- The model has billions of parameters sitting in HBM
- Each forward pass reads all weights from HBM once
- Computation per weight is tiny (one multiply-add)
- Arithmetic intensity = FLOPs / bytes read ≈ 1 FLOP / 2 bytes (FP16)

A100's peak arithmetic intensity to fully utilize compute:
```
312 TFLOPS / 1,555 GB/s = ~200 FLOPs per byte needed to be compute-bound
```

Decode has ~1 FLOP/byte. We're 200× below the compute-bound threshold — **pure memory bandwidth bound.**

This is why:
- Larger batch sizes help (more tokens share the same weight reads)
- Quantization helps (smaller weights = fewer bytes to read from HBM)
- KV cache compression helps (reduces HBM bandwidth for attention)
- PagedAttention helps (reduces KV cache memory waste, fits more in HBM)

Reference: `flash_attention.md`, `kv_cache.md`, `PagedAttention.md`
