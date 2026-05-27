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

Sample output (sorted by `cuda_time_total`, with `profile_memory=True`):
```
-----------------------------------------------  --------  ----------  --------  ----------  --------  ------------  ------------
Name                                             CPU %     CPU time    CUDA %    CUDA time   # Calls   Self Mem      Total Mem
-----------------------------------------------  --------  ----------  --------  ----------  --------  ------------  ------------
aten::mm                                          2.34%     1.200ms    68.12%    34.500ms      2400      9.18 Mb       9.18 Mb
aten::baddbmm                                     1.10%     0.560ms    15.30%     7.750ms       480      2.29 Mb       2.29 Mb
aten::softmax                                     3.20%     1.640ms     6.10%     3.090ms       480      2.29 Mb       2.29 Mb
aten::addmm                                       1.80%     0.920ms     4.20%     2.130ms      1440      9.18 Mb       9.18 Mb
aten::gelu                                        2.10%     1.080ms     2.50%     1.270ms       480      3.07 Mb       3.07 Mb
aten::layer_norm                                  1.90%     0.970ms     1.80%     0.910ms       960      3.07 Mb       3.07 Mb
-----------------------------------------------  --------  ----------  --------  ----------  --------  ------------  ------------
```

- **Self Mem** — HBM bytes allocated by this op for its own output tensor (not counting children)
- **Total Mem** — HBM bytes allocated by this op + all child ops it calls internally

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

**Memory-bound** (low arithmetic intensity, memory bandwidth is the first ceiling you'd hit at scale):
- Elementwise ops (`aten::softmax`, `aten::gelu`, `aten::layer_norm`) are a significant share of CUDA time
- Or GEMM % is low even though it's the top op
- Memory bus is saturated — tensor cores finish quickly and wait for the next HBM load
- Typical at moderate-to-large batch: enough work to stress the bus, not enough to max tensor cores
- To go faster: operator fusion (reduces round-trips to HBM), quantization (smaller dtypes = less bandwidth)

**Note on batch=1:** batch=1 is NOT memory-bound in the traditional sense. It is **low-occupancy / underutilized** — neither the memory bus NOR the tensor cores are stressed. The matrix tiles are too small to fill all SMs. See Question 4 below.

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

### Question 4: How can I tell if the GPU is underutilized — and should I push more work through it?

This question is not answered by a single column. You read a pattern across three signals:

**Signal 1 — Low Self Mem per op:**
```
aten::mm    Self Mem: 9.18 Mb    (batch=1, seq=50, d=768)
aten::mm    Self Mem: 587 Mb     (batch=64, seq=50, d=768)
```
Low allocation means small output tensors — small matrices. HBM has headroom to hold activations
from many more requests simultaneously. Room in memory = room to add more requests.

**Signal 2 — Short absolute CUDA time per op:**
```
aten::mm    CUDA time: 34.5ms total / 2400 calls = 0.014ms per call   ← very short kernel
aten::mm    CUDA time: 890ms total / 2400 calls = 0.37ms per call     ← kernel doing real work
```
Short per-call CUDA time means small matrix tiles — kernel launches, does a tiny matmul,
finishes in microseconds. Most SMs sit idle for the duration. Tensor cores are not occupied.

**Signal 3 — Low total throughput vs theoretical:**
```
Observed:    350 tokens/sec   (batch=1, GPT-2 on A100)
A100 peak:   ~312 TFLOPS FP16
GPT-2 needs: ~0.6 TFLOPS per token at batch=1
Utilization: 0.6T / 312T = 0.2%   ← GPU is doing almost nothing
```
If your tokens/sec × FLOPs-per-token is a tiny fraction of the GPU's rated TFLOPS, the hardware
is underutilized regardless of what the profiler says about individual ops.

**What to do when all three signals are low:**

Increase batch size — pack more requests together per forward pass:
```
batch=1  → aten::mm tiles are (50, 768) — tiny, SMs idle
batch=16 → aten::mm tiles are (800, 768) — SMs start filling up
batch=64 → aten::mm tiles are (3200, 768) — approaching full SM occupancy
```
As batch grows: Self Mem increases, CUDA time per op increases, throughput (tokens/sec) increases
faster than latency — you get more tokens per GPU-second. This is why dynamic batching and
continuous batching exist in production serving (vLLM, Triton, TensorRT-LLM).

**The limit:** keep increasing batch until either:
- Self Mem hits HBM capacity → OOM (or CUDA out of memory error)
- CUDA time stops scaling linearly → you've hit the memory bandwidth ceiling (memory-bound)
- CUDA time plateaus at high utilization → you've hit the compute ceiling (compute-bound)

**torch.profiler gives you the economic signal. Nsight Compute gives you the physical diagnosis:**
```
torch.profiler:   "low memory + short kernels → GPU has room for more work"
Nsight Compute:   "SM occupancy 4%, tensor core active 1% → confirmed underutilized"
```

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
