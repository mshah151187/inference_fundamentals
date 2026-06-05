# Inference Fundamentals

**Goal:** Learn GPU inference from first principles — profiling every optimization step by step to see its real impact.  
**Model:** GPT-2 (117M) — decoder-only transformer, no auth token, fast iteration.  
**Hardware:** NVIDIA A100 SXM4 40GB on Lambda Labs ($1.99/hr — spin up only when actively working).

**Core principle:** PyTorch profiler runs at every phase. Each new technique is measured before and after — you see the delta in the profiler output, not just in theory.

---

## Files

| File | Description |
|------|-------------|
| [setup.md](setup.md) | Lambda Labs A100 setup, Python env, GPT-2 download, tmux, snapshots |
| [run_gpt2_inference.md](run_gpt2_inference.md) | Phase 1 — baseline inference + PyTorch profiler |
| [working_log.md](working_log.md) | Session-by-session log of observations and results |
| [torch_profiler/torch_profiler.md](torch_profiler/torch_profiler.md) | PyTorch profiler reference — how it works, how to read output |
| [knowledge_base/profiler.md](knowledge_base/profiler.md) | Overview of all profiling tools (Kineto, Nsight, HTA, Perfetto) |
| [tensor_pin_memory/script/pinned_memory_transfer.py](tensor_pin_memory/script/pinned_memory_transfer.py) | Pageable vs pinned memory transfer deep dive |
| *(coming)* | Phase 2 — batching (naive → bucketed → dynamic) |
| *(coming)* | Phase 3 — warm-up + torch.compile |
| *(coming)* | Phase 4 — KV cache implementation |
| *(coming)* | Phase 5 — quantization (INT8, INT4, FP8, KV cache) |
| *(coming)* | Phase 6 — Nsight Systems + Nsight Compute deep dive |
| *(coming)* | Phase 7 — FastAPI server + Kubernetes deployment |

---

## Phase Roadmap

Profiling is not a one-time activity — it is the measuring instrument at every phase.
Each phase adds one technique, profiles before and after, and records the delta.

---

### Phase 1 — Baseline Inference + PyTorch Profiler

Run GPT-2 on synthetic prompts with no optimizations. Establish the baseline numbers everything else is measured against.

Profile output answers:
- What is baseline throughput (tokens/sec) and avg latency (ms/request)?
- Which ops dominate CUDA time? (aten::mm, aten::baddbmm, aten::softmax)
- Is GPT-2 compute-bound or memory-bound at batch size 1?
- How much of wall time is the GPU actually doing work vs idle?

→ [run_gpt2_inference.md](run_gpt2_inference.md)

---

### Phase 2 — Request Batching

Add batching and profile the impact at each step.

**Step 1 — Naive batching:** fixed batch size, pad all sequences to the longest in the batch.  
**Step 2 — Bucketed batching:** group requests by sequence length into predefined bins (e.g. 64, 128, 256, 512, 1024). Pad only to bin boundary — reduces wasted compute.  
**Step 3 — Dynamic batching:** collect requests over a short time window, batch whatever arrived. Maximizes GPU utilization without waiting for a full fixed batch.

Profile after each step: throughput increase, GPU idle gaps shrinking, padding waste.

Key insight: bucketing and dynamic batching solve different problems.
- Bucketing controls **shape** → finite compiled graphs, bounded padding waste.
- Dynamic batching controls **batch size** → GPU utilization, latency vs throughput tradeoff.
- Production uses both together.

---

### Phase 3 — Warm-up + Compilation

**Warm-up:** before opening to traffic, run N dummy inference passes with each expected bucket shape. Absorbs CUDA kernel JIT compilation, memory allocator warm-up, and torch.compile tracing. Without this, first real requests are 10-100× slower than steady state.

Profile: compare first-request latency vs warm request latency. See the cliff.

**torch.compile:** wraps the model with the Inductor backend. On first traced shape, generates fused CUDA kernels — multiple small ops merged into one kernel launch. Subsequent calls hit the compiled path.

Profile after compile: kernel count drops, CUDA time for hot ops decreases, fewer gaps on GPU track.

Key insight: warm-up and compile interact — warm-up triggers compilation for each bucket shape upfront so no real request ever pays the compilation cost.

---

### Phase 4 — KV Cache Implementation

Understand and implement KV cache from first principles, profiling the impact at each step.

**Step 1 — Observe the problem (done):**
Profiler shows `aten::cat` as a top CUDA time op with 1.12 GB memory footprint.
Root cause: dynamic KV cache growth via `torch.cat` — allocates new tensor + copies all
existing K/V vectors on every decode step. Cumulative cost is O(seq_len²).

**Step 2 — Study vLLM source:**
- `vllm/attention/backends/` — PagedAttention CUDA kernel
- `vllm/core/block_manager.py` — block table and free block pool management
- `vllm/worker/cache_engine.py` — HBM pool initialization at startup

**Step 3 — Implement fixed pre-allocation:**
Pre-allocate KV cache buffer for `max_seq_len` upfront. Write new K/V in-place at the
current position — no `torch.cat`, no copying.
Profile: `aten::cat` memory footprint drops to zero. Measure throughput improvement.
Downside: memory bubble — HBM reserved for max_seq_len even when actual sequence is short.

**Step 4 — Implement block-based allocation (PagedAttention mechanics):**
Pre-allocate a shared HBM block pool at startup (block_size = 16 tokens).
Each request gets a block table mapping logical → physical block.
Assign blocks from pool on demand — one block per 16 tokens, no contiguous reservation.
Write K/V in-place to current block; claim new block when full.
Profile: `aten::cat` gone + HBM waste bounded to (block_size - 1) tokens per request.
Measure: max concurrent requests vs fixed pre-allocation.

Key references: [knowledge_base/kv_cache.md](knowledge_base/kv_cache.md), [knowledge_base/PagedAttention.md](knowledge_base/PagedAttention.md)

---

### Phase 5 — Quantization

Apply quantization techniques and profile the impact on each.

- **PTQ INT8** via bitsandbytes — `load_in_8bit`; compare throughput and perplexity vs FP32 baseline
- **PTQ INT4** via bitsandbytes — `load_in_4bit`; measure memory vs accuracy tradeoff
- **GPTQ** — weight-only quantization with calibration data; profile CUDA ops before/after
- **AWQ** — activation-aware weight quantization; compare to GPTQ on same prompts
- **FP8** — H100/A100 native FP8; measure roofline shift (memory-bound → compute-bound)
- **KV cache quantization** — INT8 KV cache; measure max batch size increase from VRAM savings

Profile after each: op table changes (aten::mm bytes moved decreases → memory-bound ops speed up), VRAM usage drops, throughput increases.

Key question: at what level does accuracy degrade noticeably vs throughput gain?  
Deliverable: comparison table — method → tokens/sec → perplexity → VRAM.

---

### Phase 6 — Nsight Deep Dive

By Phase 5 the stack is well-optimized. Now use Nsight to understand what's happening at the hardware level.

**Nsight Systems (nsys):** system-wide timeline — CPU threads, GPU kernels, H2D transfers, gaps. Answers: where are the remaining idle periods? What's the kernel distribution?

**Nsight Compute (ncu):** per-kernel hardware metrics — roofline position, SM utilization, memory bandwidth, occupancy. Answers: why is this specific kernel still slow?

Compare roofline position before and after quantization — see the model shift from memory-bound toward compute-bound as INT8/FP8 reduces bytes moved per op.

---

### vllm-tuner Integrations (by Phase)

We studied the [`vllm-tuner`](../Github_Repos/vllm-tuner/) repo (Bayesian optimization for vLLM via Optuna). Most of it is too high-level for our educational goals, but three components map cleanly onto our phases:

| Component | Source file | Where it fits | What it adds |
|---|---|---|---|
| **GPU monitor** | `vllm_tuner/profiling/gpu_collector.py` | Phase 2+ | Real-time NVML metrics — GPU utilization, memory used, power draw, SM/memory clocks — alongside torch.profiler op tables |
| **Async request generator** | `vllm_tuner/benchmarks/request_generator.py` | Phase 2 | AsyncIO pattern for concurrent in-process `model.generate()` calls + latency/throughput tracking |
| **HTML comparison reports** | `vllm_tuner/reporting/html.py` | Phase 5 | Interactive Plotly dashboards for comparing quantization configs (INT4 vs FP8 vs FP32) |

**What we skip and why:**
- **Optuna study manager / optimizer** — Bayesian tuning adds abstraction distance from hardware. We want to see *why* each change moves the needle, not just find optimal values.
- **vLLM server launcher** — we run GPT-2 in-process, not via HTTP server.
- **Pydantic config models** — overkill at our current experiment scale.

**Immediate next action (Phase 2):** write `scripts/gpu_monitor.py` wrapping `pynvml` — collects utilization, memory, power, clocks — and wire it into batch size sweep experiments.

---

### Phase 7 — Containerization + Kubernetes

FastAPI inference server → Dockerfile with CUDA base image → Kubernetes Deployment.

Key engineering: readiness probe gates traffic until warm-up completes. Without this, Kubernetes routes requests to a pod that hasn't finished compilation + warm-up yet → first-request latency spike in production.

```
Startup sequence inside pod:
  Model Manager downloads from Artifactory (name + group + version)
      ↓
  torch.compile(model)
      ↓
  Warm-up loop over all bucket shapes
      ↓
  Readiness probe passes → Kubernetes routes traffic
```

---

## What You Learn (Interview Summary)

| Phase | Core skill | Talking point |
|-------|-----------|---------------|
| 1 | Read profiler output, identify hot ops | "torch.profiler showed 70% of CUDA time in aten::mm — memory-bound at batch size 1" |
| 2 | Batching tradeoffs | "Bucketing controls shape for compilation stability; dynamic batching controls batch size for GPU utilization — production needs both" |
| 3 | Warm-up + compilation | "Warm-up over all bucket shapes triggers torch.compile tracing upfront — no real request ever pays the compilation cost" |
| 4 | KV cache from scratch | "Replaced aten::cat with pre-allocated buffer — eliminated 1.12 GB cumulative copy cost; then implemented block-based allocation to remove memory bubble" |
| 5 | Quantization hands-on | "AWQ INT4 gave 3x memory reduction with <1% perplexity increase; KV cache INT8 let us double max batch size" |
| 6 | Nsight roofline | "Nsight Compute placed GPT-2 below the memory roofline at batch size 1; FP8 shifted it toward compute-bound" |
| 7 | GPU workloads on Kubernetes | "Readiness probe gates traffic until warm-up completes — prevents first-request latency spikes in production" |

---

## Cost Estimate (Lambda Labs)

| Phase | Estimated time | Cost |
|-------|---------------|------|
| Setup | 0.5 hr | ~$1 |
| Phase 1 | 2 hr | ~$4 |
| Phase 2 | 2 hr | ~$4 |
| Phase 3 | 2 hr | ~$4 |
| Phase 4 | 3 hr | ~$6 |
| Phase 5 | 3 hr | ~$6 |
| Phase 6 | 3 hr | ~$6 |
| Phase 7 | 2 hr | ~$4 |
| **Total** | **~17.5 hr** | **~$35** |

> Filesystem snapshot preserves your env between sessions (~$0.20/GB/month for storage).
