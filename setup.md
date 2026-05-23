# Environment Setup — Lambda Labs A100

Everything you need before running any phase. Do this once and take a snapshot.

> **Hardware note:** H100 SXM5 was out of capacity — using **A100 SXM4 40GB** at $1.99/hr.  
> All phases work identically except Phase 5 FP8 experiment (H100-only) — swap for INT8 instead.

---

## Step 1 — Spin Up the Instance

1. Go to [lambda.ai](https://lambda.ai) → **Instances** → **Launch**
2. Select **A100 SXM4 (1x GPU)** — $1.99/hr (use H100 SXM5 if available)
3. Base image: **Lambda Stack 22.04** — pre-installs PyTorch, CUDA, cuDNN, tmux
4. Filesystem: **Don't Attach** — GPT-2 is 548MB, snapshot is sufficient
5. Firewall: **Global Rules** (default) — SSH on port 22 is all we need
6. Add SSH key — generate on Mac if needed:
```bash
# On Mac — generate key if ~/.ssh/id_ed25519 doesn't exist:
ssh-keygen -t ed25519 -C "monil_lambda_ai_key"
cat ~/.ssh/id_ed25519.pub   # copy this into Lambda Labs
```
7. Launch → wait ~60 seconds → SSH in from Mac terminal:
```bash
ssh ubuntu@<instance-ip>    # IP shown on dashboard after status = Running
```

> **Cost tip:** Shut down the instance when not actively working.  
> Take a filesystem snapshot before shutdown — saves venv and downloaded models.

---

## Step 2 — Verify the GPU

```bash
nvidia-smi
```

**Actual output (2026-05-22, A100 instance):**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08    Driver Version: 580.105.08    CUDA Version: 13.0              |
+-----------------------------------------------------------------------------------------+
|   0  NVIDIA A100-SXM4-40GB    On  |  0MiB / 40960MiB  |  0%  Default                  |
+-----------------------------------------------------------------------------------------+
```

Key things to confirm:
- GPU name shows A100 (or H100 if available)
- Memory-Usage shows `0MiB` — nothing loaded yet
- GPU-Util shows `0%` — idle and ready

---

## Step 3 — System Packages

> **Lambda Stack 22.04 already includes:** build-essential, git, curl, htop, tmux, Python 3.10  
> Skip this step unless something is missing.

```bash
# Only run if needed:
sudo apt update && sudo apt install -y python3-pip python3-venv
```

---

## Step 4 — Python Virtual Environment

```bash
python3 -m venv ~/venv
source ~/venv/bin/activate        # run this every time you SSH in
pip install --upgrade pip
```

Add to `~/.bashrc` so the venv activates automatically on login:
```bash
echo "source ~/venv/bin/activate" >> ~/.bashrc
```

---

## Step 5 — Install Python Packages

> **Lambda Stack 22.04 already includes:** PyTorch 2.7.0, CUDA 13.0  
> Only need to install HuggingFace packages and later phase tools.

```bash
# Verify PyTorch is already there (Lambda Stack pre-installs it):
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Expected: 2.7.0 / True

# Install only what Lambda Stack does NOT include:
pip install transformers accelerate   # HuggingFace model hub + device management

# Install later when needed (don't install now):
# Phase 5 (inference server): pip install fastapi uvicorn pydantic
# Phase 4 (compilers):        pip install torch-tensorrt tensorrt
# Phase 5 (quantization):     pip install bitsandbytes auto-gptq autoawq
```

---

## Step 6 — Download GPT-2 Model

### Where does the model live?

There are three distinct storage layers on Lambda Labs:

| Storage | Size | Persists after shutdown? | Use for |
|---------|------|--------------------------|---------|
| Instance local SSD | ~1.4TB NVMe | No — lost when instance terminates | Model cache, scripts, venv |
| Persistent filesystem | Configurable | Yes | Large models / datasets you reuse across sessions |
| Filesystem snapshot | Copy of local SSD | Yes (manually taken) | Cheap way to preserve local disk state |

**How it flows:**
```
HuggingFace download  →  Instance local disk          →  GPU VRAM (when script runs)
     (internet)           ~/.cache/huggingface/hub/        model.to("cuda")
```
- `from_pretrained("gpt2")` saves weights + config to **local disk**
- `.to("cuda")` copies them into **A100 VRAM (40GB)** — only while your process runs
- VRAM is cleared when the Python process exits; disk cache stays until instance termination

**For this project (GPT-2 = 548MB):** local disk + snapshot is all you need.  
**For larger models later:** Llama-3.1-8B = ~16GB, Llama-3.1-70B = ~140GB — attach a persistent filesystem or use quantized versions.

---

Download once, reuse every session (cached in `~/.cache/huggingface`):

```bash
python3 ~/inference_fundamentals/scripts/download_gpt2_model.py
```

---

## Step 7 — tmux (Keep Jobs Alive Across SSH Disconnects)

> **Lambda Stack pre-installs tmux** — no install needed.

Always start a tmux session before running anything long:

```bash
tmux new -s main          # create session named 'main'
# ... run your scripts ...

# If SSH disconnects:
tmux attach -t main       # re-attach to the session
```

Useful tmux keys: `Ctrl+B D` = detach, `Ctrl+B %` = split pane, `Ctrl+B [` = scroll mode.

---

## Step 8 — Project Directory

```bash
mkdir -p ~/inference_fundamentals
cd ~/inference_fundamentals
```

All phase scripts live here.

---

## Verify Everything Works

```bash
python3 ~/inference_fundamentals/scripts/verify_setup.py
```

If you see generated text and "Setup complete." — you're ready for Phase 1.

---

## Take a Snapshot

After setup is confirmed working:
1. Lambda Labs dashboard → **Instances** → your instance → **Take Snapshot**
2. Name it `inference-fundamentals-base`
3. Future sessions: restore from snapshot instead of reinstalling
