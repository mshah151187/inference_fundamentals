# Nsight Systems — Batch=1 OS Runtime Libraries Findings

**Screenshots:** `screenshots/os_runtime_libraries_1.png`, `screenshots/os_runtime_libraries_2.png`

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

## Summary

| Phase | OS Runtime Activity | Cause |
|---|---|---|
| Model loading (0 → 2.3s) | Dense grey block | File I/O: open/mmap/stat64/lstat64 on weight files |
| Warmup + Inference (2.3s+) | Nearly silent | No file I/O — all data in GPU HBM |

**Key takeaway:** OS Runtime Libraries is not a bottleneck during inference. The burst at startup is expected and one-time. In production serving (model pre-loaded), this phase doesn't exist — the model is already in GPU memory when requests arrive.
