# PyTorch Profiler

## What Is It

`torch.profiler` is PyTorch's built-in tool to measure **where time and memory go** during a training or inference run. It is built on top of **Kineto** — an open-source C++ library that handles low-level CUDA event capture and NVTX range alignment between CPU and GPU timelines. It hooks into the PyTorch dispatcher and CUDA runtime to record every op that fires — on CPU and on GPU — with precise timestamps.

Without a profiler you can measure total wall time, but you cannot answer:
- Which op is eating 70% of GPU time?
- Is my GPU sitting idle while CPU prepares data?
- Is this model compute-bound (GEMM dominates) or memory-bound (elementwise ops dominate)?
- How much GPU memory does each op allocate?

The profiler answers all of these.

---

## How It Works Under the Hood

PyTorch has a **dispatcher** — every call like `torch.mm(a, b)` goes through it before hitting the actual CUDA kernel. The profiler installs hooks at the dispatcher level:

```
Python: model(inputs)
           │
           ▼
    PyTorch Dispatcher        ← profiler hook fires HERE (record start time)
           │
           ▼
    CUDA kernel launches      ← GPU starts executing
           │
           ▼
    kernel finishes           ← profiler hook fires HERE (record end time)
           │
           ▼
    back to Python
```

For CPU ops, start/end times are recorded directly. For GPU ops, PyTorch uses **CUDA events** — lightweight markers inserted into the CUDA stream that the GPU timestamps as they pass through. This is important: GPU time is measured by the GPU's own clock, not the CPU clock. You cannot measure GPU time with `time.perf_counter()` — that only measures wall time on the CPU side.

When you exit the `with profile(...) as prof:` block, PyTorch collects all the recorded events and builds the summary tables and trace file.

---

## The API

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # what to record
    record_shapes=True,      # record input tensor shapes per op
    profile_memory=True,     # track GPU memory allocated per op
    with_stack=False,        # record Python call stack (True = slower but deeper debugging)
) as prof:
    # everything inside here is profiled
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            _ = model.generate(**inputs, max_new_tokens=50)
```

### Key Parameters

| Parameter | What it does | When to use |
|-----------|-------------|-------------|
| `ProfilerActivity.CPU` | Records CPU-side op times | Always |
| `ProfilerActivity.CUDA` | Records GPU kernel times via CUDA events | Always when on GPU |
| `record_shapes=True` | Adds input tensor shape to each op | Helps identify which layer fired (e.g. large vs small matmul) |
| `profile_memory=True` | Tracks GPU memory allocated and freed per op | When debugging VRAM usage |
| `with_stack=False` | Records Python call stack per op | Set True only for deep debugging — makes profiling 3-5x slower |

### Important: Warm Up Before Profiling

```python
# Do this OUTSIDE the with profile(...) block
with torch.no_grad():
    dummy = tokenizer("warmup", return_tensors="pt").to(device)
    _ = model.generate(**dummy, max_new_tokens=10)

# NOW start profiling
with profile(...) as prof:
    ...
```

First-ever inference is slower because CUDA JIT-compiles kernels on first use and caches them. If you profile the first run, your results are polluted with compilation overhead. The warm-up run triggers all compilations so profiling sees steady-state numbers.

---

## Reading the Output

### The Summary Table

```python
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

Sample output:
```
---------------------------------  --------  --------  --------  --------  --------
Name                               CPU %     CPU time  CUDA %    CUDA time  # Calls
---------------------------------  --------  --------  --------  --------  --------
aten::mm                            2.34%    1.200ms   68.12%   34.500ms      2400
aten::baddbmm                       1.10%    0.560ms   15.30%    7.750ms       480
aten::softmax                       3.20%    1.640ms    6.10%    3.090ms       480
aten::addmm                         1.80%    0.920ms    4.20%    2.130ms      1440
aten::gelu                          2.10%    1.080ms    2.50%    1.270ms       480
aten::layer_norm                    1.90%    0.970ms    1.80%    0.910ms       960
---------------------------------  --------  --------  --------  --------  --------
```

### Column Meanings

| Column | What it measures |
|--------|-----------------|
| `Name` | PyTorch op name. `aten::mm` = matrix multiply. `aten::baddbmm` = batched matmul (attention). `aten::addmm` = matmul + bias. `aten::gelu` = activation. `aten::layer_norm` = layer normalization. |
| `Self CPU %` | CPU time spent *in this op only*, not counting child ops it calls. |
| `CPU total %` | CPU time including all child ops. For a high-level op like `model.forward`, this includes everything underneath. |
| `CUDA total %` | GPU time as % of total GPU time. **This is the primary column for inference analysis.** |
| `CUDA time` | Absolute GPU time for this op across all calls. |
| `# of Calls` | How many times this op fired total. `aten::mm` fires once per linear layer per token step — so many calls is expected. |
| `Self CUDA Mem` | GPU memory allocated by this op itself. |

### Three Sort Keys You'll Use

```python
# What dominates GPU compute?
prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)

# What's slow on CPU? (tokenization, data prep overhead)
prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)

# What's allocating GPU memory?
prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
```

---

## How to Interpret the Results

### Question 1: Is the model compute-bound or memory-bound?

Look at the top ops by CUDA time:

**Compute-bound** (enough work to keep tensor cores busy):
- `aten::mm`, `aten::addmm`, `aten::baddbmm` together > 70% of CUDA time
- Means GEMM (matrix multiply) dominates — GPU tensor cores are the bottleneck
- Typical at large batch sizes (batch 32+)
- To go faster: operator fusion, quantization (INT8/FP8 = 2x tensor core throughput)

**Memory-bound** (not enough parallelism, GPU waits on data):
- Elementwise ops (`aten::softmax`, `aten::gelu`, `aten::layer_norm`) are a significant share
- Or GEMM % is low even though it's the top op
- Typical at batch size 1 — each matrix is small, tensor cores finish quickly and wait for next load
- To go faster: batching (more work per kernel launch), pinned memory + async transfer, KV cache optimization

**GPT-2 at batch size 1 is memory-bound.** The weight matrices are loaded from VRAM for each forward pass but the compute is tiny (batch=1, seq_len=~50). The GPU loads 548MB of weights, does a small matmul, then waits for the next load.

### Question 2: Is my GPU sitting idle while CPU works?

Compare CPU time vs CUDA time for the same op:

```
aten::mm   CPU time: 0.05ms   CUDA time: 34.5ms   → GPU-bound (expected for matmul)
aten::mm   CPU time: 5.00ms   CUDA time: 0.1ms    → CPU-bound (GPU waiting on CPU)
```

If CPU time >> CUDA time across many ops: CPU is the bottleneck — GPU is starved. Common causes:
- Tokenization not overlapped with transfer (pageable memory, no non_blocking)
- Python GIL holding up kernel launches
- Data loading slower than GPU compute

### Question 3: What are the hot ops (where to optimize)?

Sort by `cuda_time_total`. The top 3 ops by CUDA time are where optimization effort pays off. For GPT-2:
1. `aten::mm` / `aten::addmm` — linear projections (Q, K, V, output, MLP)
2. `aten::baddbmm` — attention score computation (QK^T)
3. `aten::softmax` — attention weights

These are the ops that `torch.compile` fuses, quantization accelerates, and FlashAttention replaces.

---

## The Chrome Trace

```python
prof.export_chrome_trace("trace_baseline.json")
```

Open in **Perfetto UI** at `https://ui.perfetto.dev` → Open trace file → select the JSON.
(`chrome://tracing` is the legacy viewer and is deprecated — Perfetto is the current replacement.)

```
Timeline view:

CPU  ──[mm]──[baddbmm]──[softmax]──[mm]──────[mm]──────...
              ↕ kernel launch gap
GPU  ────────────[mm kernel]──────[baddbmm]──[softmax]──[mm]──...
```

**What to look for:**

| What you see | What it means |
|--------------|---------------|
| Gaps on GPU track | GPU is idle — CPU hasn't launched the next kernel yet. Kernel launch overhead or slow data prep. |
| Gaps on CPU track | CPU waiting on GPU sync (e.g. `.item()` calls that pull a scalar back to CPU). Avoid in hot loops. |
| Wide GPU bars | Hot kernels — hover to see name, duration, SM utilization. |
| H2D transfer bars | `cudaMemcpyH2DAsync` — input tensors moving CPU → GPU. For pageable: shows as a blocking CPU event. For pinned: short CPU event + async GPU bar. |
| Many narrow GPU bars | Many small kernels — high kernel launch overhead. `torch.compile` fuses these into fewer wider kernels. |

---

## Expected GPT-2 Profiler Output (Reference)

When you run `inference_profile.py` on A100, you should see approximately:

```
Op                  CUDA time %    Interpretation
aten::mm            ~55%           Linear projections (Q,K,V,O,MLP) — dominant
aten::baddbmm       ~15%           Attention QK^T computation
aten::softmax       ~8%            Attention weight normalization
aten::addmm         ~7%            Linear with bias
aten::gelu          ~5%            MLP activation
aten::layer_norm    ~4%            Layer normalization
cudaMemcpyH2D       ~3%            Input token IDs CPU → GPU
other               ~3%            Misc
```

`aten::mm` + `aten::baddbmm` + `aten::addmm` together ≈ 77% → mostly compute ops, but at batch size 1 the GPU is still memory-bound because each matmul is tiny and the GPU finishes quickly then waits.

The Chrome trace will show many short GPU bars with small gaps between them — those gaps shrink dramatically once you add batching (Phase 2) or use `torch.compile` (Phase 4).

---

## Profiling Script

See `scripts/inference_profile.py` — runs GPT-2 on 20 prompts with profiler attached, prints the three tables (CUDA time, CPU time, memory), and exports the Chrome trace to `trace_baseline.json`.
