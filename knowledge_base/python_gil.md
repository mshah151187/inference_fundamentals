# Python GIL (Global Interpreter Lock)

## What is the GIL

A mutex inside CPython (the standard Python interpreter) that allows only **one thread to execute Python bytecode at a time**, regardless of how many CPU cores are available.

Every Python process has exactly one GIL. All threads within that process compete for it.

---

## Concurrency vs Parallelism

**Parallelism** — two things literally executing at the same instant on different CPU cores.  
**Concurrency** — two things making progress over a time window, but not necessarily at the same instant (interleaved).

**Without GIL — true parallelism:**
```
Core 1: [Thread 1 running ──────────────────────]
Core 2: [Thread 2 running ──────────────────────]
         both executing simultaneously
```

**With GIL — only concurrency:**
```
Core 1: [Thread 1 ──────][waiting][Thread 1 ────]
Core 2: [waiting][Thread 2 ──────][waiting]
         only one thread holds GIL at a time
```

Even with 8 cores, only one Python thread runs at any moment. The rest are blocked waiting for the GIL.

---

## When the GIL is Released

The GIL is released when a thread is **waiting on I/O** (network, disk, sleep). Another thread can immediately acquire it and run.

```
Thread 1: [run][waiting for network ──────────────][run]
Thread 2:      [GIL acquired → run ───────────────][wait]
```

Both threads make progress — Thread 1 waits on I/O, Thread 2 computes. Wall-clock time is reduced. This is **concurrency**, not parallelism.

---

## I/O-bound vs CPU-bound

| Work type | GIL released? | Multi-threading helps? |
|---|---|---|
| Network call, disk read, sleep | Yes — while waiting | Yes — other threads run during wait |
| Tokenization, tensor ops, math | No — pure computation | No — other threads just wait |

**CPU-bound with threads — no benefit:**
```
Thread 1: [tokenize ──────────────────────────────]
Thread 2: [waiting for GIL ──────────────────────]
          same wall-clock time as single-threaded
          + context-switch overhead on top
```

---

## GIL is Per Process

GIL lives inside the interpreter. Each process has its own interpreter, therefore its own GIL. They are completely independent.

```
Process 1: [Python interpreter] → [GIL_1]
Process 2: [Python interpreter] → [GIL_2]
Process 3: [Python interpreter] → [GIL_3]

GIL_1 has no effect on GIL_2 or GIL_3.
```

Contrast with threads — all threads within a process share the same interpreter and the same GIL:

```
Process 1: [Python interpreter] → [GIL_1]
               ↑         ↑
           Thread A   Thread B   ← both competing for GIL_1
```

---

## Multiprocessing Bypasses the GIL

Since each process has its own GIL, multiple processes on multiple cores = true parallelism:

```
Process 1 (Core 1): [tokenize 62 candidates ──]
Process 2 (Core 2): [tokenize 62 candidates ──]
Process 3 (Core 3): [tokenize 62 candidates ──]
Process 4 (Core 4): [tokenize 62 candidates ──]
                     all 250 done in ~1/4 the time
```

Cost: processes don't share memory — data must be serialized and sent via IPC (pipes, queues, shared memory, or gRPC).

---

## Production Example — LinkedIn Semantic Search

At 1300 items/s/GPU (after scoring-only prefill), the GPU was well-utilized. The bottleneck shifted to **CPU preprocessing**: tokenizing 250 candidate prompts, building input tensors, encoding numerical features. Pure CPU-bound work — multi-threading gave zero benefit.

**Fix: multiprocess gRPC design**

```
Query + 250 candidates arrive
        ↓
Main serving process
    ↓ routes batches via gRPC to N worker processes
  [Worker 1]   [Worker 2]   [Worker 3]   [Worker 4]
  tokenize     tokenize     tokenize     tokenize
  1..62        63..125      126..187     188..250
        ↓
  results aggregated → single GPU forward pass
```

Each worker is a separate OS process — separate interpreter, separate GIL. gRPC handles IPC with Protocol Buffer serialization, connection pooling, and cross-machine compatibility.

**Result: 1300 → 1600 items/s/GPU (+23%)**

---

## gc.freeze() — Eliminating GC Tail Latency

Python's garbage collector periodically scans all tracked heap objects for reference cycles. Scan cost is O(n) where n = number of Python objects. For an LLM serving process the heap includes model weights, tokenizer vocab tables, compiled CUDA graphs — objects that are loaded once and never freed.

These periodic scans cause unpredictable pauses → p99 latency spikes.

`gc.freeze()` moves all currently tracked objects to a **permanent generation** the GC never scans again:

```python
# after warmup — model loaded, CUDA graphs captured, tokenizer ready
gc.freeze()   # permanent objects frozen → GC ignores them forever

# GC now only scans per-request allocations:
# intermediate tensors, tokenized inputs, response buffers → tiny heap, no spikes
```

**Why post-warmup:** during warmup, objects are in flux (model loading, JIT, CUDA graph capture). After warmup, everything permanent is stable. Freezing captures exactly the right set.
