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
- [ ] `pip install transformers accelerate` — pending
- [ ] Download GPT-2
- [ ] Create `~/inference_fundamentals/` directory
- [ ] Run end-to-end verify script
- [ ] Take snapshot

### Next Session
Continue setup: install transformers, download GPT-2, run verify script, take snapshot. Then start Phase 1 (`run_gpt2_inference.md`).
