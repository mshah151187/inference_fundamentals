# Working Log — Inference Fundamentals

Daily log of what was done, what was observed, and what's next.

---

## 2026-05-22

### Instance
- **Hardware:** NVIDIA A100-SXM4-40GB (H100 was out of capacity)
- **Price:** $1.99/hr
- **Base image:** Lambda Stack 22.04
- **Driver:** 580.105.08 | **CUDA:** 13.0 | **PyTorch:** 2.7.0 (pre-installed by Lambda Stack)

### What We Did
- Spun up Lambda Labs A100 instance
- Generated SSH key on Mac (`~/.ssh/id_ed25519`) and added to Lambda Labs as `monil_lambda_ai_key`
- SSH'd in, verified GPU with `nvidia-smi` — A100 40GB, 0MiB used, idle
- Verified PyTorch: `2.7.0`, `torch.cuda.is_available() = True`
- Confirmed Lambda Stack pre-installs: PyTorch, CUDA, tmux, build-essential — no manual install needed
- Confirmed `transformers` is NOT pre-installed — need to `pip install transformers accelerate`

### Key Learnings
- Lambda Stack 22.04 is the right base image — saves a lot of setup time
- PyTorch install step in setup.md is skippable — already at 2.7.0
- `nvidia-smi` fields: GPU name, driver, CUDA version, VRAM usage, power draw, GPU utilization
- Three storage layers: instance local SSD (lost on terminate) vs persistent filesystem vs snapshot

### Status
- [x] `pip install transformers accelerate` — transformers 5.9.0, accelerate 1.13.0
- [x] Fixed Pillow version (`pip install --upgrade Pillow`) — system Pillow too old for transformers 5.x
- [x] Set up GitHub repo (`inference_fundamentals`) and cloned on instance
- [x] Downloaded GPT-2 — 124M params, 548MB, cached at ~/.cache/huggingface/hub/
- [x] Ran verify_setup.py — full end-to-end inference working
- [ ] Take snapshot

### Observations from verify_setup.py
- VRAM total: 42.4 GB (A100 SXM4 40GB reports ~42.4GB due to ECC overhead)
- input_ids: tensor([[15496, 11, 11362, 0]]) — "Hello, GPU!" splits into 4 BPE tokens
- attention_mask: tensor([[1, 1, 1, 1]]) — all 1s, no padding on single sequence
- output shape: (1, 24) = 4 input tokens + 20 generated tokens
- Generated: "Hello, GPU!\n\nI'm not sure if you've heard of the GPU, but it's a very popular"

### Next Session
Take snapshot, then start Phase 1 (`run_gpt2_inference.md`) — create dataset.py, inference_baseline.py, inference_profile.py.

---

## 2026-05-25

### What We Did
- Created `scripts/dataset.py` — synthetic prompt dataset (SHORT/MEDIUM/LONG mix, n=50)
- Created `scripts/inference_baseline.py` — Phase 1 baseline: 50 prompts, measures throughput (tokens/sec) and avg latency (ms/request), no profiling overhead
- Created `torch_profiler/inference_profile.py` — Phase 1 profiling: 10 prompts, torch.profiler with CPU+CUDA activities, exports top ops tables + Perfetto trace
- Added `knowledge_base/profiler.md` and `torch_profiler/torch_profiler.md` — reference docs
- Added `tensor_pin_memory/` — pageable vs pinned memory deep dive

### Phase 1 Results — TO BE FILLED IN
Run on A100 instance, then record here:

| Metric | Value |
|--------|-------|
| Hardware | A100-SXM4-40GB |
| Prompts | 50 |
| Total tokens | |
| Elapsed (s) | |
| Throughput (tokens/sec) | |
| Avg latency (ms/req) | |

**Top ops by CUDA time (from inference_profile.py):**

| Op | CUDA % | CUDA time | # Calls |
|----|--------|-----------|---------|
| | | | |
| | | | |
| | | | |

**Compute-bound or memory-bound?** (circle one + brief reason):

### Next Session
- Spin up Lambda Labs A100 (`bash scripts/setup_instance.sh`)
- Run `python3 scripts/inference_baseline.py` → fill in results table above
- Run `python3 torch_profiler/inference_profile.py` → fill in top ops table
- Download `torch_profiler/inference_trace.json` → open in Perfetto UI
- Phase 2: batching experiments (batch_size = 1, 4, 16, 32)
