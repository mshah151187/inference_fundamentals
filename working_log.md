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

---

## 2026-06-02

### What We Did
- Added knowledge base docs: `flash_attention.md`, `kernel_launch_process.md`, `kv_cache.md`, `PagedAttention.md`, `nvidia_gpu_hardware.md`, `nsight_session_plan.md`
- Attempted Nsight Systems session on A100 — ran into two issues:
  1. **CUPTI conflict:** `inference_profile.py` uses `torch.profiler` which registers its own CUPTI subscriber — nsys can't register a second one. Created `nsys_profiler/script/inference_nsys.py` (plain inference, no torch.profiler) to fix this.
  2. **Version mismatch:** Instance had nsys **2024.6.2** (via apt), Mac GUI is **2026.3.1**. `.qdstrm` files are NOT cross-version compatible — file was unreadable.
- Clean nsys capture completed (no CUPTI error) but `.qdstrm` unreadable due to version mismatch.
- Instance terminated.

### Key Learnings
- Never run nsys against a script that uses `torch.profiler` — CUPTI conflict drops all CUDA kernel data
- nsys `.qdstrm` format is NOT backward compatible across major versions — GUI version must match CLI version
- Mac GUI is **2026.3.1** — instance apt only has 2024.6.2

### Next Session — Nsight Systems
1. Spin up A100 instance: `bash ~/inference_fundamentals/scripts/setup_instance.sh`
2. **Install matching nsys version FIRST:**
   ```bash
   apt-cache show nsight-systems | grep Version   # check available
   sudo apt-get install -y nsight-systems         # install
   nsys --version                                 # must match Mac GUI 2026.3.x
   ```
   If apt doesn't have 2026.3.x, download matching CLI from developer.nvidia.com/nsight-systems
3. `mkdir -p ~/inference_fundamentals/nsight`
4. Run nsys with the clean script (no torch.profiler):
   ```bash
   nsys profile \
     --trace=cuda,cudnn,cublas,osrt \
     --output=/home/ubuntu/inference_fundamentals/nsight/nsys_batch1_clean \
     --force-overwrite true \
     python ~/inference_fundamentals/nsys_profiler/script/inference_nsys.py
   ```
5. Download `.qdstrm` to Mac and open in Nsight Systems GUI

---

## 2026-06-05

### What We Did
- Installed Nsight Compute GUI (ARM64) on Mac — version **2026.2.0.0**
- Installed Nsight Compute CLI on Lambda A100 instance:
  - Downloaded `nsight_compute-linux-x86_64-2026.2.0.8.run` from developer.nvidia.com
  - Installed via: `sudo ~/nsight_compute-linux-x86_64-2026.2.0.8.run -- -noprompt`
  - Added to PATH: `export PATH=/usr/local/NVIDIA-Nsight-Compute-2026.2/target/linux-desktop-glibc_2_11_3-x64:$PATH`
  - Confirmed `ncu --version` = 2026.2.0.0 — matches Mac GUI ✓
- Pushed all pending docs to GitHub: roofline_model.md, cuda_optimization_tactics.md, nsys batch_1_experiment findings, summary.md
- Attempted ncu capture — hit two issues:
  1. `ERR_NVGPUCTRPERM` — GPU perf counters require sudo
  2. CUPTI conflict — `inference_profile.py` uses torch.profiler, must use `inference_nsys.py`
  3. `sudo pip3 install transformers accelerate Pillow --upgrade` needed for root environment
- Kill with `--launch-count 30` fix identified but instance terminated before completing

### Key Learnings
- ncu requires `sudo` on Lambda instances for GPU performance counter access
- Run `sudo env "PATH=$PATH" ncu ...` to preserve the custom PATH under sudo
- Root's Python env is separate — must `sudo pip3 install transformers accelerate` + `sudo pip3 install --upgrade Pillow`
- `--launch-count N` limits capture to first N kernels — essential to avoid 30+ min runs
- Never use `inference_profile.py` with ncu — CUPTI conflict (same rule as nsys)
- ncu PATH: `/usr/local/NVIDIA-Nsight-Compute-2026.2/target/linux-desktop-glibc_2_11_3-x64`

---

## 2026-06-14

### What We Did
- Spun up A100 instance, cloned repo, ran setup_instance.sh
- Launched the continuous batching pipeline for the first time (scheduler + gpu_worker + tokenizer + detokenizer + generator)
- Hit **two separate CUDA OOM crashes** — root-caused and fixed both

### OOM #1 — KV Store Too Large

**Symptom:** OOM at seq ~584 shortly after pipeline started filling slots.

**Root cause:** `MAX_SLOTS = 910` allocated a KV store of `910 × 1024 × 36 MB ≈ 32 GB`. A100 total is 42.4 GB. That left only ~7.5 GB for model weights (548 MB) + decode activations — nowhere near enough headroom.

**Fix:** Reduced `MAX_SLOTS: 910 → 256`.
- New KV store: `256 × 1024 × 36 MB ≈ 9.2 GB`
- Headroom for activations + weights: ~28 GB

**Files changed:**
- `batching/script/scheduler.py` — `MAX_SLOTS = 256`
- `batching/script/gpu_worker.py` — `MAX_SLOTS = 256`

---

### OOM #2 — Decode Activation Memory Grows with Sequence Length

**Symptom:** OOM again at seq ~736 even with 256 slots. Error:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 626 MiB (GPU 0; 39.56 GiB total; 624.56 MiB free)
```

**Root cause:** In `_decode_batch`, the padded KV tensors are allocated per step:
```python
k_batch = torch.zeros(batch_size, NUM_KV_HEADS, max_seq, HEAD_DIM, device=self.device)
```
These are **float32** and scale as `O(batch × seq_len)`. At seq=736 with batch=256, 12 layers:
- Per layer: `256 × 12 × 736 × 64 × 4 bytes ≈ 555 MB`
- All 12 layers assembled per step: ~13.3 GB
- Plus 9.2 GB KV store → OOM

**Key insight:** Decode activation memory is NOT constant — it grows with `seq_len`. The longer sequences accumulate in the KV store, the heavier each decode step becomes.

**Fix:** Cap the maximum total sequence length (prompt + generation):
- `MAX_NEW_TOKENS: 500 → 150`
- Prompt length: 35-40 sentences → 27-30 sentences (~330-360 tokens)
- Total max: ~360 + 150 = ~510 tokens → well within budget

**File changed:** `batching/script/generator.py`

---

### Two Options We Considered
1. **Reduce KV store** → fewer concurrent slots → less memory for the store itself
2. **Reduce computation memory** → model compression / quantization (float16/int8)

We applied option 1 (slot reduction). Option 2 was deferred — plan to study model compression techniques (quantization, pruning) before applying. Float16 changes were explicitly **not applied** this session.

### What We Observed
- Decode step time grows with seq_len (O(seq) behavior confirmed empirically)
  - seq ~519 → ~1 s/step
  - seq ~718 → ~2.7 s/step
- Pipeline was accumulating ~110K waiting requests with 0 finished before OOM — decode throughput was too low to drain the queue
- `MAX_SLOTS=256 + MAX_NEW_TOKENS=150` is the stable configuration for float32 on 40GB A100

### Key Learnings
- KV store is a fixed allocation; decode activations are a variable allocation that grows every step
- Both must fit in HBM simultaneously — slot budget must account for peak activation size, not just the KV store
- KV slot formula: `slots = HBM_budget / (max_seq_len × per_token_KV_cost)`; halving max_seq_len doubles available slots
- Real production systems (OpenAI, Anthropic, vLLM, TGI) hard-reject requests exceeding prompt/output caps before the GPU ever sees them — covered in `inference_optimizations.md §4.6`

### Changes in Repo
- `batching/script/scheduler.py`: `MAX_SLOTS = 256`
- `batching/script/gpu_worker.py`: `MAX_SLOTS = 256`
- `batching/script/generator.py`: `MAX_NEW_TOKENS = 150`, prompts 27-30 sentences
- `knowledge_base/inference_optimizations.md`: added §4.6 "Knowing Your Workload"

### Next Session
- Spin up A100, push latest changes, restart pipeline with `MAX_SLOTS=256` + `MAX_NEW_TOKENS=150`
- Verify requests complete and `[Detokenizer]` lines fire (finished > 0)
- Watch decode step time — should stay under ~1s/step with seq capped at ~510
- Once stable: study model compression (quantization/pruning) before applying float16

---

## 2026-06-08

### What We Did
- Spun up fresh A100 instance (no snapshot — reinstall from scratch each time)
- Cloned repo, ran setup_instance.sh, SCP'd ncu installer, chmod +x, installed ncu
- Installed root packages: `sudo pip3 install transformers accelerate && sudo pip3 install --upgrade Pillow`
- Discovered two bugs in the ncu command from previous session:
  1. **Wrong regex alternation**: `\|` does not work in ncu `--kernel-name` — use `|` (no backslash)
  2. **Wrong attention kernel name**: `fmha_cutlass` doesn't exist at batch=1 FP32 — actual name is `std::enable_if<...>` (CUTLASS template); match with `enable_if`
- Without `--kernel-name`, first 3 kernels captured are `vectorized_elementwise_kernel` (warmup elementwise ops) — not the matmul kernels we care about
- Ran two captures successfully:
  1. `ncu_summary_batch1.ncu-rep` — no kernel filter, `--launch-count 3`, `--set full` → captured `vectorized_elementwise_kernel` (warmup)
  2. `ncu_gemv_batch1.ncu-rep` — `--kernel-name "gemv2T_kernel_val|gemvNSP_kernel|enable_if|layer_norm"`, `--launch-count 10`, `--set full` → captured actual inference kernels
- Downloaded both files to Mac, opened in Nsight Compute GUI 2026.2.0.0

### Key Learnings
- `--kernel-name` regex uses `|` for alternation (ERE), NOT `\|` — backslash causes no matches
- At batch=1 FP32, attention kernel is CUTLASS `std::enable_if<...>` — match with `enable_if`; `fmha_cutlass` only appears for FP16
- Without `--kernel-name`, ncu captures first N kernels from warmup — not representative inference kernels
- `--launch-count 10` with `--set full` + kernel filter completes in ~15-20 min (acceptable)
- No snapshot needed — reinstall from scratch is fast enough (~10 min total setup)

### Next Session — Analyze ncu Results in GUI

Files already downloaded to `~/Downloads/`:
- `ncu_summary_batch1.ncu-rep` — 3× `vectorized_elementwise_kernel` (memory-bound baseline)
- `ncu_gemv_batch1.ncu-rep` — 10× gemv/attention/layer_norm kernels (main analysis target)

Open `ncu_gemv_batch1.ncu-rep` in Nsight Compute GUI and analyze:
1. SM throughput % for gemv2T_kernel_val — expect < 5% (occupancy-limited)
2. Memory throughput % — expect 10-30% (not even bandwidth-saturated at batch=1)
3. Roofline position — expect far left of ridge point (~24 FLOPs/byte vs ridge at 200)
4. Warp stall reasons — expect memory dependency stalls (waiting on HBM loads)

**Corrected ncu commands for next instance:**

```bash
# Setup
git clone https://github.com/mshah151187/inference_fundamentals.git ~/inference_fundamentals
bash ~/inference_fundamentals/scripts/setup_instance.sh
# SCP ncu installer from Mac: scp ~/Downloads/nsight_compute-linux-x86_64-2026.2.0.8.run ubuntu@<ip>:/home/ubuntu/
chmod +x ~/nsight_compute-linux-x86_64-2026.2.0.8.run
sudo ~/nsight_compute-linux-x86_64-2026.2.0.8.run -- -noprompt
export PATH=/usr/local/NVIDIA-Nsight-Compute-2026.2/target/linux-desktop-glibc_2_11_3-x64:$PATH
sudo pip3 install transformers accelerate && sudo pip3 install --upgrade Pillow
mkdir -p ~/inference_fundamentals/nsight

# Capture inference kernels (gemv matmuls + attention + layer_norm)
sudo env "PATH=$PATH" ncu \
  --kernel-name "gemv2T_kernel_val|gemvNSP_kernel|enable_if|layer_norm" \
  --launch-count 10 \
  --set full \
  -o /home/ubuntu/inference_fundamentals/nsight/ncu_gemv_batch1 \
  --force-overwrite \
  python3 /home/ubuntu/inference_fundamentals/nsys_profiler/script/inference_nsys.py

# Download to Mac
scp ubuntu@<ip>:/home/ubuntu/inference_fundamentals/nsight/ncu_gemv_batch1.ncu-rep ~/Downloads/
```
