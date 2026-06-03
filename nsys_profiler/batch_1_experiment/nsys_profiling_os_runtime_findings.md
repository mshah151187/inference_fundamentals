# Nsight Systems — Batch=1 OS Runtime Libraries Findings

**Screenshots:** `screenshots/os_runtime_libraries_1.png`, `screenshots/os_runtime_libraries_2.png`, `screenshots/os_runtime_libraries_3.png`

---

## What is the OS Runtime Libraries Row?

This row shows **system calls** made by the process to the OS kernel — file I/O, memory allocation, threading, etc. These are CPU-side operations, not GPU operations.

Common system calls visible here:
| System Call | What it does |
|---|---|
| `stat64` | Check file attributes (size, permissions, modification time) |
| `lstat64` | Same as stat64 but on symlink targets — used heavily by HuggingFace cache |
| `open` | Open a file descriptor |
| `read` / `mmap` | Read file contents into memory |

---

## Observation 1 — Big Grey Patch at Start (0s → ~2.3s)

A dense grey block is visible at the beginning of the timeline, corresponding to **model loading from disk**.

```
from_pretrained("gpt2").to("cuda"):
  → HuggingFace checks cache: lstat64() / stat64() on every file in ~/.cache/huggingface/hub/
  → open() + mmap() weight files into CPU RAM
  → .to("cuda") copies weights from CPU RAM → GPU HBM
```

This burst of OS system calls = the dense grey block. All 148 weight shards being located, verified, and loaded.

**Screenshot:** `os_runtime_libraries_2.png` — tooltip shows `lstat64, +1.495 μs` at 2.34s, right at the tail end of the loading phase.

---

## Observation 2 — Sparse Marks After ~2.3s (Inference Phase)

Once the model is fully loaded into GPU memory, OS runtime activity drops to near zero. Inference only reads from GPU HBM — no disk, no CPU RAM, no file I/O.

```
Model loading (0 → 2.3s):   dense OS calls (file I/O heavy)
Warmup + inference (2.3s+):  sparse OS marks (GPU memory only)
```

The tiny remaining marks during inference are Python's internal bookkeeping — memory allocator calls (`malloc`/`free`), occasional file checks, not anything on the critical path.

---

## Observation 3 — Individual Call Duration

Each system call is extremely short:
- `stat64`: +1.125 μs
- `lstat64`: +1.495 μs

These are not a bottleneck individually. The grey block looks dense because **thousands of them fire in rapid succession** during model loading — HuggingFace checks every cached file for every weight shard.

---

## Observation 4 — ioctl + nanosleep Blocks During Inference

**Screenshot:** `os_runtime_libraries_3.png` — large `ioctl → ioctl → nanosleep` blocks visible at ~4s+49ms, each spanning hundreds of milliseconds. These appear at boundaries between prompts.

These are completely different from the tiny stat64/lstat64 calls — they are **CUDA GPU synchronization**.

**`ioctl`** — the NVIDIA driver communicates with GPU hardware through `/dev/nvidia*` device files using `ioctl`. Two roles:
1. **Submit work** — CPU sends kernel launch commands down to GPU driver
2. **Poll completion** — CPU checks whether GPU has finished executing

**`nanosleep`** — CPU thread intentionally sleeps briefly. The CUDA runtime uses a poll + sleep strategy while waiting for GPU completion:
```
while GPU not done:
    ioctl(check status)   ← poll
    nanosleep(few μs)     ← back off to avoid burning 100% CPU
    ioctl(check status)   ← poll again
```

**When does this happen?** At the boundary between prompts — when `model.generate()` returns and PyTorch needs to read the output tokens back from GPU. The CPU must wait for all GPU kernels to finish before proceeding to the next prompt.

**What the Python thread shows simultaneously:** Small green activity spikes just before the ioctl blocks — Python dispatching the last kernels of a prompt, then the thread drops to 0% (blocked in OS waiting for GPU) during the ioctl/nanosleep loop.

---

## Summary

| Phase | OS Runtime Activity | Cause |
|---|---|---|
| Model loading (0 → 2.3s) | Dense grey block | File I/O: open/mmap/stat64/lstat64 on weight files |
| Warmup + Inference (2.3s+) | Nearly silent | No file I/O — all data in GPU HBM |
| Between prompts | Large ioctl + nanosleep blocks | CPU waiting for GPU to finish (CUDA synchronization) |

**Key takeaway:** OS Runtime Libraries is not a bottleneck during inference. The file I/O burst at startup is one-time. The ioctl/nanosleep blocks are synchronization overhead — unavoidable when CPU must wait for GPU results before starting the next prompt.
