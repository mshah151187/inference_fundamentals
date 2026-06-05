# Nsight Systems — Batch=1 OS Runtime Libraries Findings

**Screenshots:** `screenshots/os_runtime_libraries_1.png`, `screenshots/os_runtime_libraries_2.png`, `screenshots/os_runtime_libraries_3.png`, `screenshots/os_runtime_libraries_4.png`, `screenshots/os_runtime_libraries_5.png`

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
| `ioctl` | CPU→GPU driver communication via `/dev/nvidia*` — submit work or poll completion |
| `nanosleep` | CPU backs off briefly between ioctl polls to avoid burning 100% CPU |
| `poll` | OS file descriptor wait on `/dev/nvidia*` — blocks until driver signals GPU completion |
| `pthread_cond_wait` | Thread-to-thread signal — CUDA internal thread pool coordination |

**CPU-blocking syscalls compared:**

All four of `ioctl` (poll mode), `nanosleep`, `poll`, and `pthread_cond_wait` result in the CPU blocked waiting for GPU — but through different layers:

| Syscall | Layer | Mechanism |
|---|---|---|
| `ioctl` + `nanosleep` | Driver level | CPU actively spinning, checking GPU status in a loop |
| `poll` | Driver level | OS blocks thread on `/dev/nvidia*` fd until driver signals |
| `pthread_cond_wait` | CUDA runtime level | CUDA internal thread pool — worker thread signals main thread on GPU completion |

```
pthread_cond_wait pattern:
  Python thread:        pthread_cond_wait(&cond, &mutex)  ← sleeps, releases mutex
  CUDA worker thread:   GPU done → pthread_cond_signal()  ← wakes Python thread

poll pattern:
  Python thread:        poll(/dev/nvidia*)  ← OS blocks until driver fires event

ioctl + nanosleep pattern:
  Python thread:        while GPU not done:
                            ioctl(check status)
                            nanosleep(few μs)
```

You will often see all three in the same inter-prompt gap — on different threads in the Nsight timeline. They are all symptoms of the same root cause: CPU has nothing to do while GPU executes the autoregressive decode loop.

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

**Note on `poll` syscall:** Alongside `ioctl` and `nanosleep`, you may also see a `poll` syscall in this block. `poll` is another OS mechanism the CUDA runtime uses to wait on a file descriptor — in this case `/dev/nvidia*`. It blocks the thread until the driver signals GPU completion, same purpose as the ioctl check loop but used at different points in the CUDA runtime's wait strategy.

---

## Observation 5 — The CPU Idle Gap Is What We Want to Exploit

The ioctl/nanosleep/poll blocks represent **CPU doing nothing useful** while the GPU is running. The script is written synchronously:

```python
for i, prompt in enumerate(prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)  # ← CPU blocks here
    # next prompt tokenization only starts AFTER generate() returns
```

`model.generate()` does not return until GPU finishes AND output tokens are copied back to CPU. So the actual timeline looks like:

```
Prompt 1:
  CPU: tokenize → dispatch kernels → [IDLE: ioctl/nanosleep/poll] → read output
                                       ↑ GPU executing here

Prompt 2:
  CPU: tokenize → dispatch kernels → [IDLE: ioctl/nanosleep/poll] → read output
                                       ↑ GPU executing here
```

During the entire GPU execution window, the CPU is blocked in the synchronization loop — wasted. What we want instead:

```
Prompt 1:
  CPU: tokenize → dispatch kernels → [tokenize prompt 2 here] → read output
                                       ↑ GPU executing here (overlapped)
Prompt 2:
  CPU: dispatch kernels → [tokenize prompt 3 here] → read output
                           ↑ GPU executing here (overlapped)
```

CPU preprocessing (tokenization, KV cache allocation, request scheduling) overlapped with GPU compute. The GPU never stalls waiting for the CPU to prepare the next batch.

**This is exactly what vLLM's continuous batching solves.** The scheduler runs on CPU and prepares the next iteration's batch while the GPU is executing the current one. The ioctl/nanosleep idle gap we see here is the inefficiency vLLM closes — and it is visible proof, in the OS Runtime row, of why naive HuggingFace `generate()` leaves throughput on the table.

---

## Observation 6 — Warmup Phase Has a Distinct OS Runtime Signature

**Screenshot:** `os_runtime_libraries_4.png` — timeline around 3s+407ms to 3s+409ms.

Before the first `poll` at ~408ms, the OS Runtime row shows mostly **white space** with small clusters of `lstat64 → open64 → read` in between. After 408ms, hard `poll` blocks appear.

**What white space means:** No OS system calls firing. The CPU thread is active — dispatching CUDA kernels, running Python, executing the tokenizer — but none of that requires OS-level syscalls visible in this row. Pure userspace work. Crucially, the GPU is running in parallel during this time (visible in the CUDA HW row above).

**The lstat64/open64/read clusters:** The HuggingFace tokenizer stat-checks its vocabulary cache files at each `tokenizer()` call. Same file-check pattern as model loading but much smaller bursts — one cluster per prompt tokenization.

**Why the first poll appears at exactly 408ms:** This marks the end of the 3 warmup passes. During warmup, `model.generate()` dispatches kernels and GPU runs, but Python hasn't hit a hard sync barrier yet. At 408ms the warmup loop completes and the first real synchronization point fires — CPU needs to read output tokens back from GPU.

```
0 → 408ms  (warmup phase):
  OS Runtime: white + tiny lstat clusters
  CPU: tokenizing, dispatching kernels — active, no blocking wait
  GPU: running warmup kernels in parallel

408ms+ (inference phase):
  OS Runtime: hard poll blocks between prompts
  CPU: blocked waiting for GPU to return output tokens
  GPU: executing autoregressive decode kernels
```

This contrast makes the warmup vs inference boundary clearly visible in the OS Runtime row — warmup looks mostly white (CPU working), inference looks like repeating poll blocks (CPU blocked at each prompt boundary).

---

## Observation 7 — 74ms Poll Block: Quantifying the CPU Idle Waste

**Screenshot:** `os_runtime_libraries_5.png` — tooltip on a poll block in the inference phase.

```
poll
Begins: 3.40869s
Ends:   3.48229s  (+74.209 ms)
```

**What this 74ms represents:** the full autoregressive decode of one prompt — 50 tokens × ~1.5ms per token step. The CPU is completely blocked for the entire GPU execution window of a single prompt.

**Call stack at poll entry:**
```
libc.so → poll
_ssl.cpython3-10 → [2 Frames]      ← Python C extension using poll()
_PyEval_EvalFrameDefault            ← Python interpreter loop
_PyFunction_Vectorcall              ← Python function dispatch
... (20+ repeating PyEval/PyFunction frames)
                                    ← model.generate() call stack depth
```

The poll is buried 20+ frames deep inside Python's interpreter — `model.generate()` → Python eval loop → CUDA runtime → libc `poll()`. The CPU thread is completely blocked at the OS level, not just yielding. No Python work can happen on this thread during those 74ms.

**Scale of waste across the full run:**
```
10 prompts × ~74ms per poll = ~740ms of pure CPU idle
Total inference run ≈ 2–3 seconds
CPU idle from polling alone ≈ 30–40% of total runtime
```

**What the CPU could have done in 74ms:**
- Tokenized the next 5–10 prompts (tokenization takes microseconds)
- Allocated KV cache slots for the next batch
- Run the vLLM scheduler for the next iteration
- Handled incoming HTTP requests

This single screenshot is the clearest evidence of why naive sequential `model.generate()` leaves throughput on the table. Every prompt pays a full GPU-execution-length CPU tax at the OS poll level, with zero overlap.

---

## Summary

| Phase | OS Runtime Activity | Cause |
|---|---|---|
| Model loading (0 → 2.3s) | Dense grey block | File I/O: open/mmap/stat64/lstat64 on weight files |
| Warmup + Inference (2.3s+) | Nearly silent | No file I/O — all data in GPU HBM |
| Between prompts | Large ioctl + nanosleep + poll blocks | CPU waiting for GPU to finish (CUDA synchronization) |

**Key takeaway:** The ioctl/nanosleep/poll blocks are not just synchronization overhead — they are wasted CPU time. During those blocks the GPU is running and the CPU could be tokenizing the next prompt, allocating KV cache, and scheduling the next batch. Production inference engines (vLLM) overlap this CPU work with GPU execution, eliminating the idle gap and maximizing GPU utilization.
