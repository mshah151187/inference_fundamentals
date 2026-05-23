# Phase 1 — Baseline GPT-2 Inference + PyTorch Profiling

**Prerequisite:** `setup.md` complete — venv active, GPT-2 downloaded, GPU verified.

**Goal:** Run GPT-2 on a synthetic dataset, use PyTorch profiler to identify which ops dominate CUDA time, and determine whether the model is compute-bound or memory-bound at inference.

---

## Step 1 — Create the Synthetic Dataset

Save as `~/inference_fundamentals/dataset.py`:

```python
# dataset.py
import random

SHORT = [
    "Tell me a joke.",
    "What is 2+2?",
    "Hello!",
    "What day is it?",
    "Name a color.",
]
MEDIUM = [
    "Explain how transformers work in simple terms.",
    "Write a short story about a robot.",
    "What are the main causes of climate change?",
    "How does a CPU differ from a GPU?",
    "Describe the water cycle.",
]
LONG = [
    "Describe in detail the architecture of the original transformer model "
    "as introduced in the 'Attention is All You Need' paper, including "
    "the encoder, decoder, multi-head attention, and positional encoding.",
    "Write a detailed essay on the history of machine learning from "
    "the perceptron in the 1950s through deep learning in the 2020s.",
    "Explain how NVIDIA H100 GPUs differ from previous generations, covering "
    "HBM3 memory, NVLink 4.0, Transformer Engine, and FP8 compute.",
]

def get_dataset(n=100, seed=42):
    random.seed(seed)
    # Mix: ~40% short, ~40% medium, ~20% long — realistic inference workload
    pool = SHORT * 8 + MEDIUM * 8 + LONG * 4
    return random.choices(pool, k=n)

if __name__ == "__main__":
    data = get_dataset(n=10)
    for i, p in enumerate(data):
        print(f"{i+1}. [{len(p):3d} chars] {p[:60]}...")
```

Run it to verify:
```bash
python3 dataset.py
```

---

## Step 2 — Baseline Inference (No Profiling)

Save as `~/inference_fundamentals/inference_baseline.py`:

```python
# inference_baseline.py
import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import get_dataset

device = "cuda"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

prompts = get_dataset(n=50)

# Warm-up: first inference is slower due to CUDA kernel JIT compilation
print("Warming up...")
with torch.no_grad():
    dummy = tokenizer("warmup", return_tensors="pt").to(device)
    _ = model.generate(**dummy, max_new_tokens=10)

# Timed run
print("Running inference...")
total_output_tokens = 0
start = time.perf_counter()

with torch.no_grad():
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=50)
        total_output_tokens += outputs.shape[-1]
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(prompts)} done")

elapsed = time.perf_counter() - start
print(f"\nResults:")
print(f"  Prompts:       {len(prompts)}")
print(f"  Total tokens:  {total_output_tokens}")
print(f"  Elapsed:       {elapsed:.2f}s")
print(f"  Throughput:    {total_output_tokens / elapsed:.1f} tokens/sec")
print(f"  Latency avg:   {elapsed / len(prompts) * 1000:.1f} ms/request")
```

Run it:
```bash
python3 inference_baseline.py
```

Note down the throughput and avg latency — this is your **baseline** to beat in later phases.

---

## Step 3 — Add PyTorch Profiler

Save as `~/inference_fundamentals/inference_profile.py`:

```python
# inference_profile.py
import torch
from torch.profiler import profile, ProfilerActivity
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import get_dataset

device = "cuda"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

prompts = get_dataset(n=20)   # smaller set — profiling adds overhead

# Warm-up outside the profiler so compilation noise doesn't pollute results
with torch.no_grad():
    dummy = tokenizer("warmup", return_tensors="pt").to(device)
    _ = model.generate(**dummy, max_new_tokens=10)

print("Profiling...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,      # record input tensor shapes per op
    profile_memory=True,     # track GPU memory allocations
    with_stack=False,        # call stack — set True for deeper debugging (slower)
) as prof:
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            _ = model.generate(**inputs, max_new_tokens=50)

# --- Report 1: Top ops by CUDA time ---
print("\n=== Top 20 ops by CUDA time ===")
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))

# --- Report 2: Top ops by CPU time ---
print("\n=== Top 10 ops by CPU time ===")
print(prof.key_averages().table(
    sort_by="cpu_time_total",
    row_limit=10
))

# --- Report 3: Memory ---
print("\n=== Top 10 ops by GPU memory ===")
print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage",
    row_limit=10
))

# --- Chrome trace: visualize in browser ---
prof.export_chrome_trace("trace_baseline.json")
print("\nChrome trace saved: trace_baseline.json")
print("To view: scp to local machine, open chrome://tracing, load the file")
```

Run it:
```bash
python3 inference_profile.py 2>&1 | tee profile_output.txt
```

---

## Step 4 — Read the Profiler Output

### What the table columns mean

| Column | Meaning |
|--------|---------|
| `Name` | Op name — `aten::mm` = matrix multiply, `aten::softmax` = softmax |
| `Self CPU %` | Time in this op itself (excluding children) |
| `CPU total %` | Time including child ops |
| `CUDA total %` | GPU time — **this is what matters for inference speed** |
| `# of Calls` | How many times this op fired |
| `Self CPU Mem` | CPU memory allocated |
| `CUDA Mem` | GPU memory allocated |

### What to look for

**Expected top CUDA ops for GPT-2:**
```
aten::mm           — matrix multiply (linear layers: Q, K, V, output projections)
aten::baddbmm      — batched matrix multiply (attention scores: QK^T)
aten::softmax      — attention weights normalization
aten::addmm        — linear layer with bias (same family as mm)
aten::gelu         — activation function in MLP
aten::layer_norm   — layer normalization
```

**What the split tells you:**
- If `aten::mm` / `aten::addmm` / `aten::baddbmm` together are > 70% of CUDA time → **compute-bound** (GEMM dominates)
- If elementwise ops (`softmax`, `gelu`, `layer_norm`) are a significant share → **memory-bound** at this batch size
- For batch size 1 (single requests), GPT-2 is typically **memory-bound** — not enough parallelism to saturate tensor cores
- For larger batches, it shifts toward **compute-bound**

**CPU vs CUDA gap:**
- If an op has high CPU time but low CUDA time → kernel launch overhead (CPU submitting work faster than GPU executes)
- Large gap between CPU time and CUDA time across all ops → GPU is underutilized

### Fill in this table after running

| Op | CUDA time % | # calls | Bound type |
|----|-------------|---------|------------|
| aten::mm | | | |
| aten::baddbmm | | | |
| aten::softmax | | | |
| aten::gelu | | | |
| aten::layer_norm | | | |

---

## Step 5 — View the Chrome Trace (Optional but Recommended)

```bash
# Copy to your local machine
scp ubuntu@<instance-ip>:~/inference_fundamentals/trace_baseline.json .
```

1. Open Chrome → go to `chrome://tracing`
2. Click **Load** → select `trace_baseline.json`
3. Use `W/S` to zoom in/out, `A/D` to pan

**What to look for in the trace:**
- **CPU track:** Python/C++ ops running on CPU
- **GPU track:** CUDA kernels executing on GPU
- **Gaps on GPU track:** idle GPU — means CPU is the bottleneck (data prep, tokenization)
- **Wide kernels:** these are your hot ops — hover to see kernel name and duration
- **H2D/D2H transfers:** `cudaMemcpy` bars — input tensors moving CPU → GPU

---

## Deliverable

Before moving to Phase 2, answer these questions (write them in a comment at the top of `inference_profile.py` or in a notes file):

1. What is your baseline throughput (tokens/sec) from Step 2?
2. What are the top 3 ops by CUDA time and their percentages?
3. Is GPT-2 compute-bound or memory-bound at batch size 1?
4. What is the ratio of CPU time to CUDA time for `aten::mm`? (Large ratio = GPU is waiting on CPU)
5. What % of wall time is the GPU actually doing work (vs idle)?

These answers become your reference point for every optimization in Phases 2-4.
