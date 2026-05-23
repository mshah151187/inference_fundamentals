# Inference Fundamentals

**Goal:** Learn GPU inference from first principles — profiling, batching, compilers, quantization, and deployment.  
**Model:** GPT-2 (117M) — decoder-only transformer, no auth token, fast iteration.  
**Hardware:** NVIDIA H100 SXM5 on Lambda Labs ($4.29/hr — spin up only when actively working).

---

## Files

| File | Description |
|------|-------------|
| [setup.md](setup.md) | Lambda Labs H100 setup, Python env, GPT-2 download, tmux, snapshots |
| [run_gpt2_inference.md](run_gpt2_inference.md) | Phase 1 — baseline inference + PyTorch profiler |
| *(coming)* | Phase 2 — request batching (naive, bucketed, dynamic) |
| *(coming)* | Phase 3 — Nsight Systems + Nsight Compute |
| *(coming)* | Phase 4 — torch.compile + TensorRT |
| *(coming)* | Phase 5 — Quantization experiments (PTQ, GPTQ, AWQ, FP8, KV cache) |
| *(coming)* | Phase 6 — FastAPI server + Kubernetes deployment |

---

## Phase Roadmap

### Phase 1 — Baseline Inference + PyTorch Profiling
Run GPT-2 on synthetic prompts. Use `torch.profiler` to find the top ops by CUDA time, classify the model as compute-bound or memory-bound, and export a Chrome trace.  
→ [run_gpt2_inference.md](run_gpt2_inference.md)

### Phase 2 — Request Batching Strategies
Compare naive batching (fixed batch + pad to max), bucketing (group by sequence length), and dynamic batching (fill by total token budget). Measure throughput and padding waste for each.

### Phase 3 — Nsight Systems + Nsight Compute
System-level timeline with Nsight Systems (gaps, H2D transfers, kernel distribution). Kernel-level deep dive with Nsight Compute (roofline, SM utilization, memory bandwidth, occupancy).

### Phase 4 — Compilers
`torch.compile` with `inductor` backend vs TensorRT via `torch-tensorrt`. Measure speedup, observe op fusion reducing kernel count, compare profile output before and after.

### Phase 5 — Quantization Experiments
Apply the quantization techniques you know from theory to real GPT-2 inference on H100. Measure the accuracy vs speed tradeoff for each method.

- **PTQ with bitsandbytes** — INT8 and INT4 via `load_in_8bit` / `load_in_4bit`; compare throughput and perplexity vs FP32 baseline
- **GPTQ** — weight-only quantization with calibration data; run with `auto-gptq`; profile CUDA ops before/after
- **AWQ** — activation-aware weight quantization; run with `autoawq`; compare to GPTQ on same prompts
- **FP8 on H100** — H100's native FP8 Transformer Engine; enable via `transformer_engine` library; measure roofline shift
- **KV cache quantization** — quantize the KV cache to INT8 to reduce memory footprint during generation; measure max batch size increase
- **Profile each variant** — use PyTorch profiler on every method; see how the op table changes (fewer bytes moved = memory-bound ops get faster)

Key question to answer: at what quantization level does accuracy degrade noticeably vs the throughput gain? Build a comparison table: method → tokens/sec → perplexity → VRAM usage.

### Phase 6 — Containerization + Kubernetes
FastAPI inference server → Dockerfile with CUDA base image → Kubernetes Deployment with GPU resource limits, liveness/readiness probes, and Service. Practice locally with minikube.

---

## What You Learn (Interview Summary)

| Phase | Core skill | Talking point |
|-------|-----------|---------------|
| 1 | Read profiler output, identify hot ops | "torch.profiler showed 70% of CUDA time in aten::mm — memory-bound at batch size 1" |
| 2 | Batching tradeoffs | "Dynamic batching by token budget gave 2x+ throughput vs naive fixed-size batching" |
| 3 | Nsight tools, roofline model | "Nsight Compute placed the GEMM kernel below the memory roofline at batch size 1" |
| 4 | Compiler internals, op fusion | "torch.compile reduced kernel count by ~40% through operator fusion" |
| 5 | Quantization methods hands-on | "AWQ at INT4 gave 3x memory reduction with <1% perplexity increase vs FP32 on GPT-2" |
| 6 | GPU workloads on Kubernetes | "GPU pods need requests==limits; readiness probes gate traffic until model is loaded" |

---

## Cost Estimate (Lambda Labs)

| Phase | Estimated time | Cost |
|-------|---------------|------|
| Setup | 0.5 hr | ~$2 |
| Phase 1 | 2 hr | ~$9 |
| Phase 2 | 2 hr | ~$9 |
| Phase 3 | 3 hr | ~$13 |
| Phase 4 | 3 hr | ~$13 |
| Phase 5 | 3 hr | ~$13 |
| Phase 6 | 2 hr | ~$9 |
| **Total** | **~15.5 hr** | **~$68** |

> Filesystem snapshot preserves your env between sessions (~$0.20/GB/month for storage).
