# CUDA Graphs — How They Work, Bucketing, and Piecewise Execution

Related: `kernel_launch_process.md` (why dispatch overhead exists), `kv_cache.md` (PagedAttention eliminates dynamic allocations that break graphs)

---

## 1. The Problem CUDA Graphs Solve

Every kernel launch in eager mode follows the full CPU pipeline:
Python → Dispatcher → C++ → cudaLaunchKernel → GPU stream queue.
At batch=1 this overhead (~28μs per op) exceeds GPU execution time (~13.7μs).
For a 32-layer transformer with ~50 ops per layer: 1,600 kernel launches per forward
pass — over 44ms of pure CPU dispatch cost on every decode step.

After optimizations like shorter sequences, smaller models, and prefix caching, each
remaining kernel is small and fast. The GPU finishes quickly then **waits for the CPU
to dispatch the next kernel**. Bottleneck shifts from GPU compute to CPU dispatch.

---

## 2. How CUDA Graphs Work

### Capture phase — record, don't execute

```python
# Warm up first (cuBLAS algorithm cache, allocations)
for _ in range(3):
    output = model(input_ids)

# Capture: GPU goes into record mode
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(input_ids)   # no GPU execution happens here
                                # CPU records the sequence of kernel launches
                                # into a graph data structure
```

During capture, CUDA intercepts every `cudaLaunchKernel` call and logs it as a node
in a DAG (directed acyclic graph) with edges representing dependencies:

```
Graph nodes (each = one kernel + its args):
  [embedding_lookup] → [layernorm_0] → [addmm_qkv] → [attention] → [addmm_out]
                                ↓
                          [addmm_mlp1] → [gelu] → [addmm_mlp2]
                                                        ↓
                                                  [layernorm_32] → [lm_head]
```

Tensor pointers (addresses) are baked into graph nodes at capture time.
Input and output tensors must be pre-allocated and reused across replays.

### Replay phase — single call, full forward pass

```python
# Update input in-place (same memory address, new values)
input_ids.copy_(new_token_ids)

# Replay: entire forward pass in one API call
g.replay()
# CPU cost: one cudaGraphLaunch (~5μs) instead of 1,600 cudaLaunchKernel calls
```

```
Eager mode:
  CPU: [launch][launch][launch]...[launch]  ← 1,600 calls × 28μs = 44ms CPU
  GPU:         [k1][k2][k3]...[k1600]

CUDA Graph:
  CPU: [graphLaunch]                         ← 1 call × 5μs = 5μs CPU
  GPU: [k1][k2][k3]...[k1600]               ← identical GPU execution
```

GPU execution is identical. CPU overhead collapses from 44ms to 5μs.

### Constraint — tensors must be pre-allocated

Tensor addresses are baked into graph nodes at capture time — you cannot allocate
new tensors during replay. All inputs, outputs, and intermediates must be allocated
before capture and reused across replays.

```python
# Pre-allocate before capture
static_input  = torch.zeros(batch, seq_len, dtype=torch.long, device='cuda')
static_output = torch.zeros(batch, vocab_size, dtype=torch.float, device='cuda')

# Capture
with torch.cuda.graph(g):
    static_output = model(static_input)

# Each step: copy new data into pre-allocated buffers, then replay
static_input.copy_(new_token_ids)
g.replay()
logits = static_output
```

### How address baking works — concrete example

After `model.to('cuda')` and before capture, every tensor already has a fixed HBM address:

```
HBM layout after model load:

  Address 0x1234:  weight matrix W  (768×768, loaded once, never moves)
  Address 0x3456:  static_input     (pre-allocated input buffer, shape fixed)
  Address 0x7890:  static_output    (pre-allocated output buffer, shape fixed)
  Address 0xABCD:  intermediate_0   (activations, pre-allocated before capture)
  ...
```

During capture, every kernel launch is recorded with these exact addresses baked in:

```
Graph node 0 — addmm kernel:
  instruction: "multiply input × weight, write result to intermediate"
  src_A = 0x3456   ← input buffer address, frozen at capture time
  src_B = 0x1234   ← weight address, frozen at capture time
  dst   = 0xABCD   ← intermediate address, frozen at capture time

Graph node 1 — layernorm kernel:
  src   = 0xABCD
  dst   = 0x7890
  ...
```

At replay time, the graph re-fires these exact instructions — same addresses, no
re-evaluation of anything. To handle a new request:

```
Request 1 arrives — tokens [42, 17, 8]:
  Step 1: CPU writes [42, 17, 8] into HBM address 0x3456  ← overwrite input buffer
  Step 2: graph.replay()
          GPU executes: "read 0x3456, read 0x1234, compute, write 0x7890"
  Step 3: CPU reads result from 0x7890

Request 2 arrives — tokens [5, 99, 23]:
  Step 1: CPU writes [5, 99, 23] into HBM address 0x3456  ← same address, new data
  Step 2: graph.replay()                                   ← identical graph execution
  Step 3: CPU reads result from 0x7890
```

Weight at 0x1234 is **never touched between requests** — it sits in HBM permanently.
Only the input buffer at 0x3456 is overwritten before each replay. The graph itself
is completely unchanged — same nodes, same addresses, same kernel variants.

---

## 3. The Shape Constraint

CUDA Graph requires the computation to be **structurally identical** across replays —
same kernels, same tensor shapes, same data flow.

### Why shape is baked in at three levels

When a graph node is captured for an `addmm` kernel at seq_len=50:

```
addmm capture — seq_len=50, batch=1, hidden=768:
  (M, K, N) = (50, 768, 768)

  1. Kernel variant  — cuBLAS selects Variant B (tile 32×32) for this shape
  2. Grid dimensions — ceil(50/32) × ceil(768/32) = 2 × 24 = 48 thread blocks
  3. Tensor args     — M=50, N=768, K=768 baked into kernel arguments
```

If you replay with seq_len=100 (M=100):
```
Grid dims replayed: (2, 24, 1)  ← still 48 thread blocks for only rows 0..49
Rows 50..99: never computed — no thread blocks assigned to them
Output: silently wrong, no error raised
```

The GPU doesn't validate shapes at replay time. It just executes what was recorded.

### One unique shape = one unique graph

For a transformer layer, every `addmm` shape is a function of `(batch_size, seq_len)`:
```
QKV projection:    (batch × seq_len,  hidden) × (hidden, 3×hidden)
Output projection: (batch × seq_len,  hidden) × (hidden,   hidden)
MLP fc1:           (batch × seq_len,  hidden) × (hidden, 4×hidden)
MLP fc2:           (batch × seq_len, 4×hidden) × (4×hidden, hidden)
```

Change either `batch_size` or `seq_len` → M changes in every kernel → different
cuBLAS variant + different grid dims → need a new capture.

---

## 4. Bucketing — Collapsing the Shape Space

Without bucketing, seq_len can be anything from 1 to max_context (e.g., 8192).
You can't pre-capture 8192 different graphs at startup.

Bucketing collapses the continuous shape space into a small fixed set:

```
Bucket sizes: [32, 64, 128, 256, 512]

Incoming request with seq_len=100:
  → round UP to nearest bucket: 128
  → pad input to seq_len=128 (dummy tokens, masked in attention)
  → replay graph captured for seq_len=128
  → discard padded output positions
```

At startup, capture one CUDA Graph per bucket — 5 captures instead of 8192.

```
Startup (once):
  capture graph for (batch=1, seq_len=32)  → graph_32
  capture graph for (batch=1, seq_len=64)  → graph_64
  ...
  capture graph for (batch=1, seq_len=512) → graph_512

Per inference call (seq_len=100):
  → select graph_128
  → copy padded input into static buffer
  → graph_128.replay()   ← 5μs CPU, correct grid dims
```

### The padding tradeoff

```
seq_len=100 padded to bucket 128:
  Wasted compute: 28 extra token positions = 22% overhead
  Dispatch saving: 44ms eager CPU overhead → 5μs graph replay

At scale, dispatch saving dominates.
Padding waste is bounded by bucket granularity — finer buckets = less waste
but more graphs to capture and more GPU memory for static buffers.
```

Powers-of-2 buckets are common: worst-case padding is 2× (just above a bucket boundary),
number of graphs stays small (log2(max_seq_len) total).

### How vLLM uses bucketing

```
vLLM captures graphs for batch sizes: [1, 2, 4, 8, 16, 32, ...]

Incoming decode step with 5 requests:
  → pad to batch=8 (next captured size)
  → replay graph_batch8
  → ignore padded output rows

Prefill: always runs in eager mode (variable prompt lengths)
Decode:  always runs via CUDA Graph (fixed shape per captured batch size)
```

---

## 5. Piecewise CUDA Graphs — Handling Mixed Static/Dynamic Computation

### The problem with full-graph capture for inference

In LLM ranking (e.g., LinkedIn's SLM reranker), a prompt has distinct segments with
different shape characteristics:

```
[system prompt (fixed)] [query (variable)] [candidate suffix (variable)]
      ↑ same every call      ↑ varies per user   ↑ varies per candidate
```

You can't capture one full-sequence graph — the variable portions break it.

### Piecewise approach: segment by dynamism

Split the computation into **stable segments** and **dynamic segments**. Capture CUDA
graphs for stable segments, fall back to eager execution for dynamic segments, stitch
them together at runtime.

```
Segment 1: system prompt KV    → fixed shape always        → CUDA graph ✓
Segment 2: query tokens        → variable, but bucketed    → one graph per bucket ✓
Segment 3: candidate suffix    → variable                  → eager execution
Segment 4: scoring head        → fixed shape (1 logit out) → CUDA graph ✓
```

Execution at inference time:
```
graph_system_prompt.replay()        ← 5μs, fixed
graph_query_bucket128.replay()      ← 5μs, bucketed
eager_forward(candidate_tokens)     ← normal dispatch, pays kernel overhead
graph_scoring_head.replay()         ← 5μs, fixed
```

The stable segments get the full dispatch-elimination benefit. Only the dynamic segment
pays eager overhead — and that segment is now a fraction of total compute.

### What "piecewise" means in the CUDA API

Between graph segments, PyTorch uses **conditional graph nodes** or **stream
synchronization points** to hand off between graph replay and eager execution:

```
[graph_A replay] → sync point → [eager ops] → sync point → [graph_B replay]
```

The sync point flushes the GPU stream so the eager ops see the outputs of graph_A,
and graph_B sees the outputs of the eager ops. No tensor copying — they share HBM,
the sync point is purely a scheduling barrier.

### Bucketing applies to both query and candidate

The same bucketing logic applies to candidate suffixes too:

```
Query buckets:     [32, 64, 128, 256]     → 3-4 buckets cover 95% of query lengths
Candidate buckets: [32, 64, 128, 256, 512] → wider range needed for raw text
```

**Why candidates are harder to bucket than queries:**
- Query length variance is low (5–50 words) — 3-4 buckets sufficient
- Raw candidate variance is high (50–2,100 tokens) — many buckets needed, short
  candidates padded to 256 waste 80% of compute
- Continuous batching mixes candidates from multiple queries — needs length-sorting
  before batching (adds latency) or heavy padding within the batch

### Stable segment summary

```
Segment              | Shape varies? | Strategy
─────────────────────|───────────────|──────────────────────────
System prompt KV     | No            | Full CUDA graph
Query tokens         | Yes, small    | Bucketed graphs (3-4)
Candidate suffix     | Yes, large    | Eager OR bucketed (many graphs)
Scoring head         | No            | Full CUDA graph
```

---

## 6. Why MixLM Eliminates the Dynamic Shape Problem for Candidates

MixLM compresses every candidate to exactly T_S embedding tokens (e.g., 16). The
candidate suffix is **always** `[16, dim]` — no bucketing needed:

```
Without MixLM: [prefix (bucketed)] | [candidate (variable 50–2100 tok)]
               → piecewise graphs, eager fallback for candidate

With MixLM:    [prefix (bucketed)] | [16 embedding tokens]
               → single full graph per bucket, no eager fallback
```

With MixLM, every segment has a fixed or bucketed shape. The entire sequence becomes
graphable — no dynamic segments, no eager execution, no stitching overhead.

This is why MixLM and CUDA graphs compound: MixLM eliminates the shape variability that
forced piecewise execution in the first place.

---

## 7. When to Use Each Approach

| Situation | Approach |
|---|---|
| Fully fixed shapes (decode step in vLLM) | Single full graph per batch size |
| Variable input length, bounded range | Bucketed graphs |
| Mixed static/dynamic computation | Piecewise graphs |
| MixLM / fixed embedding tokens | Single full graph (no dynamic segments) |
| Highly variable shapes, hard to bucket | Eager mode (accept dispatch overhead) |

---

## Related Docs

- `kernel_launch_process.md` — full CPU→GPU dispatch pipeline, why 28μs per kernel
- `kv_cache.md` — KV cache fundamentals; PagedAttention eliminates aten::cat which would require dynamic allocation incompatible with graph capture
- `PagedAttention.md` — block-based KV cache; pre-allocated fixed buffers are what make graph capture viable for vLLM decode
