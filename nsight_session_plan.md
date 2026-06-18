# Nsight Session Plan — Batch=1 Baseline Diagnosis

**Goal:** Confirm with hardware-level evidence what we derived analytically from torch.profiler.
One session on the A100 baseline (batch=1, GPT-2, max_tokens=50).

**Hardware:** Lambda Labs A100 SXM4 40GB
**Cost estimate:** ~1.5–2 hrs → ~$3–4

---

## What We Expect to Confirm

From torch.profiler analysis:

| Prediction | Expected Nsight Evidence |
|---|---|
| batch=1 is low-occupancy (not memory/compute bound) | SM occupancy < 10% for addmm |
| GPU idle between kernels (CPU dispatch gap) | Timeline gaps between GPU kernels in nsys |
| addmm is GEMV not GEMM at batch=1 | Roofline: below memory ridge point |
| Elementwise ops (layernorm, add, mul) are memory-bound | Very low arithmetic intensity in ncu |
| FlashAttention efficient_attention backend active | fmha_cutlass kernel visible in nsys timeline |

---

## Tool 1 — Nsight Systems (nsys)

**What it gives:** System-wide timeline — CPU threads, CUDA API calls, GPU kernels,
memory transfers all on one timeline. Answers "where are the gaps and what causes them?"

### Command

```bash
nsys profile \
  --trace=cuda,cudnn,cublas,osrt \
  --output=/home/ubuntu/inference_fundamentals/nsight/nsys_batch1 \
  --force-overwrite true \
  python /home/ubuntu/inference_fundamentals/torch_profiler/script/inference_profile.py
```

Output: `nsys_batch1.nsys-rep` — open in Nsight Systems GUI on your Mac.

### What to look for in the timeline

**1. CPU→GPU gap pattern:**
```
CPU row:  [dispatch─────][gap][dispatch─────][gap][dispatch─────]
GPU row:       [kernel──]          [kernel──]          [kernel──]
                    ↑ gap = CPU preparing next launch
```
At batch=1 we expect visible white space on the GPU row between kernels.

**2. Kernel duration vs gap duration:**
- addmm kernel: should be ~13.7μs
- Gap after addmm: should be ~28μs (CPU dispatch for next op)
- Ratio confirms CPU is the bottleneck

**3. cuBLAS kernel names visible:**
- `volta_sgemm_...` or `ampere_sgemm_...` for addmm — confirms GEMM variant selected
- `gemvN_kernel` or `gemvx` — confirms GEMV dispatch at batch=1 (degenerate matmul)
- `fmha_cutlassF_f32_...` — FlashAttention kernel

**4. Memory transfers:**
- Should see H2D transfers at start (input tokens to GPU)
- No D2H during inference (output stays on GPU until generate() finishes)
- Large H2D = tokenization not overlapped → optimization opportunity

### Download for local analysis

```bash
# On Lambda instance — compress for download
gzip nsys_batch1.nsys-rep

# Download to Mac (run locally)
scp ubuntu@<instance-ip>:/home/ubuntu/inference_fundamentals/nsight/nsys_batch1.nsys-rep.gz .
gunzip nsys_batch1.nsys-rep
```

Open in Nsight Systems GUI (free download from NVIDIA).

---

## Tool 2 — Nsight Compute (ncu)

**What it gives:** Per-kernel hardware metrics — SM occupancy, warp efficiency,
memory bandwidth, arithmetic intensity, roofline position.
Answers "why is this specific kernel slow at the hardware level?"

**Note:** ncu is much slower than normal execution (10–100× overhead per kernel).
Two flags are critical to avoid hour-long runs:
- `--launch-count N` — stop after profiling N matching kernel invocations (without this, ncu profiles every invocation in the entire script — hundreds of them)
- `--kernel-name regex` — only profile kernels whose name matches the pattern (skip unrelated kernels entirely)
- `--set basic` — collect only the core counter set (SM throughput, memory throughput, occupancy); much faster than `--set full`

### Step 1 — Capture inference kernels (gemv matmuls + attention + layer_norm)

```bash
sudo env "PATH=$PATH" ncu \
  --kernel-name "gemv2T_kernel_val|gemvNSP_kernel|enable_if|layer_norm" \
  --launch-count 10 \
  --set full \
  -o /home/ubuntu/inference_fundamentals/nsight/ncu_gemv_batch1 \
  --force-overwrite \
  python3 /home/ubuntu/inference_fundamentals/nsys_profiler/script/inference_nsys.py
```

**Critical correctness notes:**
- `sudo env "PATH=$PATH"` — required; sudo resets PATH losing the custom ncu install
- `|` not `\|` — ncu uses ERE alternation; backslash causes zero matches (no output file)
- `enable_if` not `fmha_cutlass` — at batch=1 FP32 the attention kernel is a CUTLASS template (`std::enable_if<...>`); `fmha_cutlass` only appears for FP16
- Use `inference_nsys.py` not `inference_profile.py` — torch.profiler causes CUPTI conflict
- `--set full` gives roofline data, warp stall reasons, arithmetic intensity (~15-20 min with kernel filter)

### Step 2 — Download to Mac

```bash
scp ubuntu@<ip>:/home/ubuntu/inference_fundamentals/nsight/ncu_gemv_batch1.ncu-rep ~/Downloads/
```

Open in Nsight Compute GUI → analyze SM throughput %, memory throughput %, roofline position per kernel.

### Key metrics explained

| Metric | What it measures | Expected at batch=1 |
|---|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM compute utilization % of peak | < 5% — confirms low occupancy |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Active warps / max warps | < 10% — most warps idle |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | HBM bandwidth utilization % | 10–30% — below bandwidth ceiling |
| `l1tex__t_bytes.sum.per_second` | L1/shared memory throughput | Low — small tiles |
| `sm__sass_thread_inst_executed_op_ffma_pred_on.sum` | FMA instructions executed | Low count — confirms tiny matmul |

### Roofline interpretation

```
A100 ridge point: 200 FLOPs/byte

addmm at batch=1, shape (50, 768) × (768, 768):
  FLOPs: 2 × 50 × 768 × 768 = 59M
  Bytes: (50×768 + 768×768 + 50×768) × 4 = ~2.5 MB
  Arithmetic intensity: 59M / 2.5M = ~24 FLOPs/byte

24 << 200 (ridge point) → deep in memory-bound region on roofline
But HBM bandwidth also not saturated at batch=1 (matrix too small)
→ Neither ceiling is hit → low-occupancy (underutilized hardware)
```

Expected roofline position for key kernels:
```
addmm (batch=1):          ~24 FLOPs/byte  → far left of ridge → low occupancy
native_layer_norm:         ~4 FLOPs/byte  → always memory-bound
fmha_cutlass (attention):  ~8 FLOPs/byte  → memory-bound at short seq_len
```

---

## Session Checklist

```
□ 1. Spin up Lambda Labs A100 instance
□ 2. Restore from filesystem snapshot (avoids re-downloading GPT-2)
□ 3. mkdir -p /home/ubuntu/inference_fundamentals/nsight
□ 4. Run nsys profile command → nsys_batch1.nsys-rep
□ 5. Run ncu summary command (--launch-count 3, --set basic) → ncu_summary_batch1.ncu-rep  [~5 min]
□ 6. Run ncu deep dive command (--launch-count 1, --set full) → ncu_deep_batch1.ncu-rep    [~10 min]
□ 7. Download all three .rep files to Mac
□ 8. Open nsys_batch1.nsys-rep in Nsight Systems GUI
□ 9. Open ncu_deep_batch1.ncu-rep in Nsight Compute GUI
□ 10. Fill in observed values in results table below
□ 11. Shut down instance (don't forget — $1.99/hr)
```

---

## Results Table (fill in during session)

### nsys findings

| Observation | Expected | Observed |
|---|---|---|
| addmm kernel duration | ~13.7μs | |
| Gap after addmm (CPU dispatch) | ~28μs | |
| GPU idle % of total wall time | > 70% | |
| H2D transfer visible at start | Yes | |
| fmha_cutlass kernel visible | Yes | |
| GEMV kernel visible (not GEMM) | Yes at batch=1 | |

### ncu findings

| Kernel | SM throughput % | HBM bandwidth % | Arithmetic intensity | Warp occupancy % |
|---|---|---|---|---|
| addmm (sgemm) | | | | |
| native_layer_norm | | | | |
| fmha_cutlass (attention) | | | | |
| aten::add / aten::mul | | | | |

---

## What to Document After the Session

Create `knowledge_base/nsight_results_batch1.md` with:
1. Annotated screenshots of nsys timeline (gaps, kernel durations)
2. Roofline chart from ncu for addmm
3. SM occupancy numbers confirming low-occupancy diagnosis
4. Comparison table: torch.profiler prediction vs ncu measured
5. One-paragraph summary: "batch=1 is neither memory-bound nor compute-bound — it is occupancy-limited"

This becomes the baseline everything else is measured against in Phase 2 batching.

---

## Reference

- torch.profiler analysis: `torch_profiler/torch_profiler.md`
- Kernel launch pipeline: `knowledge_base/kernel_launch_process.md`
- Hardware specs (A100 ridge point, peak TFLOPS): `knowledge_base/profiling_guide.md`
- FlashAttention kernel details: `knowledge_base/flash_attention.md`
