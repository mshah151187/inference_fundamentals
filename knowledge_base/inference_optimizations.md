# Inference Serving Optimizations

**Interview question:** "How would you improve inference metrics?"

The right answer starts with: "Which metric — latency, throughput, or cost?"
Each has different levers. Conflating them leads to wrong answers.

---

## 1. Inference Metrics

### Latency
Time from request received to response returned.

```
p50 latency  — median experience (most users)
p99 latency  — tail experience (worst 1% of users, drives SLA violations)
```

For LLMs specifically, latency splits into two sub-metrics:
```
TTFT (Time to First Token)     — how long until streaming starts
                                 = prefill time + queue wait
ITL  (Inter-Token Latency)     — time between consecutive output tokens
                                 = one decode step duration
```

### LLM Metrics in Depth

LLM serving has a richer metric space than traditional ML because generation is
autoregressive — the user waits twice: once for the first word, then again between
each word.

```
arrival          slot allocated   first token              last token
   │                   │               │                       │
   ▼                   ▼               ▼                       ▼
───┬───────────────────┬───────────────┬───────────────────────┤
   │◄── queue_wait ───►│◄── prefill ──►│◄──── decode steps ───►│
   │                                   │                       │
   │◄─────────── TTFT ────────────────►│                       │
   │                                   │◄── ITL ──►│           │
   │◄──────────────────── total latency ───────────────────────►│
```

**Queue wait** — time in WAITING queue before a KV slot is allocated.
```
Driven by: HBM pressure — all slots occupied, new requests backlog.
High queue_wait = system at memory capacity.
Good: < 50ms.  Bad: > 500ms (system saturated).
```

**TTFT (Time to First Token)** — queue wait + prefill time.
```
= queue_wait + prefill_time

Drives: perceived responsiveness. User sees a blank screen until TTFT elapses.
TTFT > 1s feels broken. TTFT < 300ms feels instant.

queue_wait large  → memory pressure, too few KV slots for incoming rate
prefill_time large → long prompt, compute-bound forward pass
```

**ITL (Inter-Token Latency)** — time between successive output tokens = one decode step.
```
Drives: streaming smoothness. Tokens arrive at 1/ITL per second.
Humans read ~250ms/word → ITL < 50ms feels instant.

Decode is memory-bandwidth bound (GEMV at batch=1, GEMM at large batch).
ITL is determined primarily by HBM bandwidth and decode batch size.
```

**Throughput (TPS — tokens per second)** — total output tokens across all requests per second.
```
= total_tokens_generated / elapsed_time

Driven by: decode batch size, model size, hardware bandwidth.
Higher batch → each decode step produces more tokens → TPS increases.
```

**Capacity planning formula (Little's Law):**
```
concurrent_requests = QPS × request_lifetime
KV_budget           = concurrent_requests × KV_per_request

Example: 200 QPS, 5s avg lifetime, 36 MB per request
  concurrent = 200 × 5 = 1,000 requests
  KV_budget  = 1,000 × 36 MB = 36 GB

This is the primary sizing question for LLM infra.
Compute is rarely the constraint — HBM capacity is.
```

### Throughput
How much work the system completes per unit time.
```
Requests per second (RPS)    — how many users served per second
Tokens per second (TPS)      — output tokens generated per second across all requests
```

**Key tension: latency vs throughput**
```
Higher batching → more requests processed together → better throughput
               → each request waits longer in the queue → worse latency

These two metrics pull in opposite directions.
You cannot optimize both simultaneously without more hardware.
```

### GPU Utilization
What fraction of GPU compute capacity is being used.
```
SM Utilization %     — are CUDA/Tensor cores busy?
HBM Bandwidth %      — is memory bandwidth saturated?
```

Low GPU utilization with high request volume = wasted hardware = high cost.
High GPU utilization with good latency = efficient serving.

### Memory Utilization
How much HBM is occupied.
```
Model weights:    fixed (e.g. 14 GB for 7B model in FP16)
KV cache:         grows with batch size × sequence length
Activations:      proportional to batch size
```

Memory limits maximum batch size — if HBM is full, you cannot batch more requests.
Memory utilization directly caps throughput.

### Cost
GPU hours consumed per unit of useful work.
```
$ per 1000 tokens       — standard LLM cost metric
GPU hours per request   — infra-level cost metric
```

Cost = f(GPU utilization, throughput). Higher utilization at same throughput = lower cost.

---

## 2. Understanding Request Flow and Decomposing Latency

### Request Flow

Every inference request passes through the same logical stages in sequence:

```
Request arrives
      ↓
┌─────────────────────────────┐
│ Task 1: Feature Fetching     │  ← fetch candidate features + user history
│         from Feature Store   │    from KV store, feature store, etc.
└─────────────────────────────┘
      ↓
┌─────────────────────────────┐
│ Task 2: Last-Mile            │  ← feature joining, normalization,
│         Transformation       │    tensor construction, or a pre-processing
└─────────────────────────────┘    ML model (e.g. embedding lookup)
      ↓
┌─────────────────────────────┐
│ Task 3: Queue Wait           │  ← waiting for a GPU worker to be free
│                              │    often overlooked, dominates under load
└─────────────────────────────┘
      ↓
┌─────────────────────────────┐
│ Task 4: Model Inference      │  ← GPU forward pass
│         (GPU)                │
└─────────────────────────────┘
      ↓
┌─────────────────────────────┐
│ Task 5: Post-Processing      │  ← score assembly, ranking, business rules
└─────────────────────────────┘
      ↓
Result returned

Total latency for one request = sum of all task times (strict sequential dependency —
each task needs the output of the previous one)
```

### Two Goals for Any Optimization Technique

**Goal 1 — Within-task parallelization**

Find operations inside a single task that have no dependency on each other and run
them in parallel.

```
Example — Task 1 (Feature Fetching):
  Fetch candidate features  ─┐
                              ├─→ both fetches in parallel → total = max(50ms, 30ms)
  Fetch user history        ─┘    instead of 50ms + 30ms = 80ms
```

Reduces the duration of that task without touching other tasks.

**Goal 2 — Pipeline parallelism across requests**

For a single request, cross-task parallelization is impossible — data must flow
sequentially through the pipeline. But when serving multiple requests, different
requests can be at different stages simultaneously:

```
Time →       T1           T2           T3           T4
Req 1:   [Fetch]      [Transform]  [Inference]  [PostProc]
Req 2:               [Fetch]      [Transform]  [Inference]
Req 3:                            [Fetch]      [Transform]
Req 4:                                         [Fetch]
```

GPU never idles waiting for features. CPU never idles waiting for GPU.
Each hardware resource is occupied at every time slot.

### The Bottleneck Rule

**Throughput of the pipeline = throughput of the slowest task.**

Speeding up a non-bottleneck task does not improve throughput.

```
Scenario 1 — Inference is the bottleneck:

  Feature fetch:   20ms  ← fast
  Transform:        5ms
  Queue wait:       0ms  (GPU always free)
  Inference:      100ms  ← bottleneck
  Post-process:    5ms

  Pipeline throughput = 1 request per 100ms
  Pipelining helps: fetch for req 2 happens during req 1's inference → no idle GPU
  Speeding up fetch from 20ms → 5ms saves nothing on throughput
  To improve: reduce inference time (quantization, batching, faster GPU)

Scenario 2 — Feature fetch is the bottleneck:

  Feature fetch:   90ms  ← bottleneck (slow feature store, many features)
  Transform:        5ms
  Queue wait:       0ms
  Inference:       30ms  ← GPU sitting idle for 60ms waiting for features
  Post-process:    5ms

  Pipeline throughput = 1 request per 90ms
  GPU utilization is low despite fast inference — the upstream is the problem
  To improve: parallelize fetches within Task 1, cache features, reduce feature count
```

Identifying which task is the bottleneck is the first step before applying any technique.
Distributed tracing across all tasks gives you this answer directly.

---

## Concept of streaming pipeline: pipeline parallelism

Instead of one process doing all five tasks sequentially for every request, divide the
work across multiple dedicated processes and connect them via a queue (ZMQ, Kafka, etc.).
Each process owns one stage. While it works on request N, the previous stage has already
moved on to request N+1.

```
Stage 1 Process          Stage 2 Process          Stage 3 Process
(Tokenization /          (Scheduler)              (GPU Worker)
 Feature Fetch)                │                        │
      │                        │                        │
  req_N ──ZMQ──────────────▶  │                        │
      │                    req_N ──ZMQ──────────────▶  │
  req_N+1 ──ZMQ────────────▶  │                    req_N executing on GPU
      │                    req_N+1 ──ZMQ────────────▶  │
  req_N+2 ──ZMQ────────────▶  │                    req_N+1 executing on GPU
```

Each stage is always busy. No stage waits for the next one to finish — it hands off
via ZMQ and immediately picks up the next request.

### Key properties

**For a single request — latency does not improve:**
```
Total latency = Stage 1 time + Stage 2 time + Stage 3 time
```
The pipeline doesn't shorten a single request's journey. Each request still passes
through all stages sequentially.

**For the system — throughput improves:**
```
System throughput = throughput of the slowest stage  (bottleneck rule)
```
All stages run concurrently across different requests. If Stage 3 (GPU) takes 100ms
and the other stages take 10ms each, throughput = 1 request per 100ms — upstream
stages are never the bottleneck.

**Queues between stages absorb speed mismatches:**
If Stage 2 is faster than Stage 3, ZMQ's internal queue buffers the excess.
Stage 2 keeps producing; Stage 3 drains at its own pace. No stage blocks.
If the queue fills up (HWM), Stage 2 naturally slows — built-in backpressure.

### Overlap within Stage 3 — the GPU Worker

The same pipeline principle applies inside a single stage:

```
GPU Worker (Stage 3):

  Build tensors on CPU  (batch N+1)  ─────────────────────┐
  Async PCIe transfer   (batch N+1)  ──────────────────┐  │
  GPU executes model    (batch N)    ←── runs while  ──┘  │
                                         transfer and  ────┘
                                         build happen
```

Pinned memory + `non_blocking=True` enables this overlap within the GPU worker.
Without it, `.to("cuda")` blocks the CPU and the three steps collapse back to sequential.

### Why separate processes instead of threads?

Python's GIL allows only one thread to execute Python bytecode at a time.
Separate processes have separate GIL domains — true parallelism.
ZMQ is the natural IPC glue: low latency (2-5μs over `ipc://`), lock-free internal
queues, and built-in patterns (PUSH/PULL for pipeline, ROUTER for fan-out).

### The critical nuance: overlap is across requests, not within one

The scheduler cannot start KV block allocation for request N until tokenization is
done — it needs sequence length to know how many blocks to allocate. So for a single
request, Stage 1 → Stage 2 is strictly sequential.

The overlap is: Stage 1 working on req N+1 while Stage 2 works on req N while
Stage 3 executes req N-1. Goal 2 — pipeline parallelism across requests.

### This pattern appears at every scale

```
vLLM inference       Tokenizer → Scheduler → GPU Worker
Video ML pipeline    Ingest → Feature Extraction → Embedding → KNN Index
Kafka streaming      Producer → Broker → Consumer
CPU instruction      Fetch → Decode → Execute → Writeback
execution
```

Same idea at every level — split sequential work into stages, connect with a queue,
run stages in parallel across items. The queue is ZMQ in vLLM, Kafka in data
pipelines, a 4-entry buffer in a CPU pipeline.

---

## 3. Generic Inference Optimization Techniques

These apply to **any model** on GPU — not LLM-specific.
For each technique: which goal it addresses, what metrics it improves, what it hurts.

---

### 3.0 IPC Transport Selection — ipc:// over tcp:// for Same-Node Communication

**What it does:**
When multiple processes on the same machine communicate (e.g. Tokenizer → Scheduler →
GPU Worker), choose Unix domain sockets (`ipc://`) over TCP (`tcp://localhost`) for
inter-process message passing.

**Why it matters:**
Every request crosses multiple process boundaries in a streaming pipeline. Each crossing
is on the critical path of that request's latency.

```
tcp://localhost (same machine):          ipc:// (Unix domain socket):
  IP header construction                   No IP layer
  Port lookup + routing table check        No port allocation
  Loopback interface traversal             No routing
  TCP sequence numbers, ACK overhead       Kernel copies directly between
                                           process buffers via socket file
  ~10–20μs per send/recv                   ~2–5μs per send/recv
```

**Where it applies:**
Any multi-process architecture on a single node — inference serving pipelines,
data preprocessing pipelines, microservices on the same host. Not limited to ML.

```
Within one node:     always prefer ipc:// over tcp://
Across nodes:        tcp:// is the only option — no unix socket across machines
Across threads       use inproc:// (shared memory, ~50ns, no kernel involvement)
(same process):
```

**Metric improved:** Latency ↓
Per-boundary saving of ~10-15μs × number of pipeline stages × requests per second.
At 100 RPS with 4 ZMQ boundaries per request: 100 × 4 × 15μs = 6ms saved per second
in pure IPC overhead — before any model-level optimization.

**Which goal:** Goal 2 — reduces the handoff cost between pipeline stages, making
the streaming pipeline tighter.

**Pros:**
- Zero code change beyond the address string (`ipc://` vs `tcp://`)
- No external dependency — Unix domain sockets are OS primitives

**Cons:**
- Only works on the same host — must switch to `tcp://` for multi-node deployments
- Socket files need cleanup on crash (stale `.ipc` files can cause bind failures)

---

### 3.1 Batching

See detailed doc: `batching.md`

---

### 3.2 Bucketing

**What it does:** Maps every request's input length to one of N pre-defined bucket
sizes by padding up to the nearest bucket. All requests in a bucket have the
same fixed tensor shape.

**Which goal:** Bucketing serves two goals simultaneously:

**Goal 1 (within-task):** Reduces padding waste compared to naive batching.
Without bucketing, one long sequence in a batch pads all others to its length.
With bucketing, requests are grouped by similar lengths so padding is minimal.

```
Without bucketing — batch [12, 15, 11, 400], all padded to 400:
  GPU runs 4 × 400 = 1,600 token steps
  Useful: 438 steps → 73% wasted

With bucketing (buckets: 16, 32, 64, 512):
  [12, 15, 11] → bucket 16 → 3 × 16 = 48 steps
  [400]        → bucket 512 → 1 × 512 = 512 steps
  Total: 560 steps vs 1,600 — far less waste
```

**Goal 2 (pipeline) — enables CUDA Graph reuse:**
This is the more important benefit. CUDA Graphs require fixed input tensor shapes.
A graph captured for shape `(batch=8, seq_len=512)` can only replay for exactly that shape.

Without bucketing, every request has a unique sequence length → unique shape →
CUDA Graph cannot be reused → either re-capture every time (expensive) or skip graphs.

With bucketing, there are only N distinct input shapes — one per bucket.
Capture N CUDA Graphs once at warm-up. Every request thereafter hits one of the
N known shapes and replays the pre-captured graph at zero capture cost.

```
Without bucketing:
  Request 73 tokens  → shape (batch, 73)  → unique → no graph reuse
  Request 89 tokens  → shape (batch, 89)  → unique → no graph reuse
  Request 134 tokens → shape (batch, 134) → unique → no graph reuse

With bucketing (buckets: 64, 128, 256, 512):
  Request 73 tokens  → padded to 128 → shape (batch, 128) → reuses Graph_128
  Request 89 tokens  → padded to 128 → shape (batch, 128) → reuses Graph_128
  Request 134 tokens → padded to 256 → shape (batch, 256) → reuses Graph_256
```

Bucketing is what makes CUDA Graphs practical for variable-length inference workloads.

**Metrics improved:** Throughput ↑, Cost ↓, Latency ↓ (via enabling CUDA Graph reuse)

**Metrics hurt:** Slightly more padding than optimal (padded to bucket ceiling, not
exact length). Tradeoff is almost always worth it — CUDA Graph savings dominate.

---

### 3.3 Compilation (torch.compile)

**What it does:** JIT-compiles the model's forward pass into optimized CUDA kernels.
Fuses adjacent operations (e.g. LayerNorm + add + activation into one kernel),
removes redundant memory reads/writes, and applies GPU-specific code generation.

**Which goal:** Goal 1 — within-task parallelization (within the inference task).
Reduces the number of kernel launches and HBM round-trips for intermediate tensors.

```
Without compile:
  LayerNorm kernel  → write output to HBM
  Add kernel        → read from HBM, write to HBM
  Activation kernel → read from HBM, write to HBM
  3 kernel launches, 4 HBM read/write passes

With compile (fused):
  LayerNorm + Add + Activation → single fused kernel
  1 kernel launch, 1 HBM write pass
  Intermediate results stay in registers/SRAM — never hit HBM
```

**Metrics improved:** Latency ↓, Throughput ↑

**Metrics hurt:** Warm-up time on first call (compilation happens at runtime).
Breaks on highly dynamic input shapes — compile assumes fixed or near-fixed shapes.

---

### 3.4 CUDA Graphs

**What it does:** Captures the entire forward pass as a static graph during a warm-up
run. Subsequent calls replay the graph with a single CPU call instead of launching
each kernel individually.

**Which goal:** Goal 1 — within-task parallelization (within the inference task).
Eliminates CPU-side kernel dispatch overhead, which dominates at small batch sizes.

```
Without CUDA Graphs:
  ~100 kernels per forward pass × 21μs CPU dispatch each = 2.1ms CPU overhead
  At batch=1, GPU compute ≈ 3ms → CPU overhead is 40% of total inference time

With CUDA Graphs:
  1 graph replay call ≈ 10μs total CPU overhead
  CPU overhead: 2.1ms → 0.01ms
```

**Metrics improved:** Latency ↓ (most impactful at small batch where GPU time is short
and CPU overhead is proportionally large)

**Metrics hurt:** Requires fixed input shapes — graph is compiled for specific tensor
dimensions. Any shape change requires re-capturing the graph.

See: `CUDA_Graph.md`

---

### 3.5 Multi-Process Serving

**What it does:** Runs one Python process per GPU. Each process owns one GPU
exclusively and handles requests independently with no shared state.

**Which goal:** Goal 2 — pipeline parallelism. Multiple GPUs serve different requests
simultaneously without the Python GIL serializing execution.

```
Python GIL: only one thread can execute Python bytecode at a time.

1 process, 8 GPU threads:
  Thread 1 running → Thread 2-8 blocked by GIL
  Effective GPU utilization: 1 out of 8

8 processes, 1 GPU each:
  Each process has its own GIL — they run truly in parallel
  Effective GPU utilization: 8 out of 8
```

A load balancer distributes incoming requests across processes.
Each process: receives request → fetches features → runs inference → returns result.

**Metrics improved:** Throughput ↑, GPU Utilization ↑

**Metrics hurt:** Memory — model weights loaded once per process. 8 GPUs = model
loaded 8 times. Not a problem when each process owns one GPU (model lives on that GPU's HBM).

See: `python_gil.md`

---

### 3.6 Async Execution / Pipelining

**What it does:** Overlaps CPU work (feature fetch, preprocessing) for the next
request with GPU compute for the current request.

**Which goal:** Goal 2 — pipeline parallelism across requests. While GPU executes
Task 4 (inference) for request N, CPU executes Task 1+2 (fetch + transform) for
request N+1 in parallel.

```
Without async:
  [Fetch N] → [Transform N] → [Inference N] → [Fetch N+1] → [Transform N+1] → ...
  GPU idle during fetch + transform of every request

With async:
  [Fetch N] → [Transform N] → [Inference N]
                               [Fetch N+1]  → [Transform N+1] → [Inference N+1]
  CPU and GPU overlap → GPU never idles waiting for features
```

**Metrics improved:** Latency ↓ (eliminates GPU idle time), Throughput ↑

**Metrics hurt:** Complexity — requires async I/O and careful coordination between
CPU and GPU work. Prefetch logic must handle cancellation if requests are dropped.

---

### 3.7 Quantization

**What it does:** Reduces the numeric precision of model weights (FP32 → FP16 → INT8 → INT4).
Fewer bytes per weight = less HBM bandwidth needed to load the model per forward pass.

**Which goal:** Goal 1 — within-task parallelization (within inference task).
Directly reduces the HBM bottleneck that limits decode throughput.

```
FP32 → FP16:  4 bytes → 2 bytes  = 2× fewer bytes read from HBM per weight
FP16 → INT8:  2 bytes → 1 byte   = 2× fewer bytes (4× vs FP32)
FP16 → INT4:  2 bytes → 0.5 byte = 4× fewer bytes (8× vs FP32)
```

Since decode is HBM-bandwidth bound, halving bytes ≈ halving decode latency.

**Metrics improved:** Throughput ↑, Latency ↓ (decode), Memory ↓, Cost ↓

**Metrics hurt:** Accuracy — lower precision introduces quantization error.
INT4 requires careful calibration. Negligible for INT8 on most models.

---

### 3.8 Producer-Side Batching for Backpressure Resilience

**What it does:** Coalesce multiple requests into a single message at the producer
before sending downstream, rather than emitting one message per request.

**Which goal:** Goal 2 — pipeline parallelism. Reduces the number of messages in
flight between stages, which directly improves resilience when a downstream stage
is slow.

**Why message count matters — ZMQ HWM:**

ZMQ (and most message queue systems — Kafka, RabbitMQ) enforce backpressure via
a High Water Mark (HWM) counted in **messages, not bytes**. When the downstream
consumer is slow, the upstream buffer fills up message by message. Once HWM is
reached, the producer's `send()` blocks.

```
ZMQ default HWM = 1000 messages

Single-request messages:  1000 messages × 1 request  = 1000 requests buffered
Batched messages (N=16):  1000 messages × 16 requests = 16,000 requests buffered
```

Batching at the producer multiplies the effective buffer capacity by N — the same
number of ZMQ messages now absorbs N× more requests before backpressure kicks in.

**What this does and does not fix:**

Batching delays backpressure — it does not eliminate it. If the downstream stage
is fundamentally slow (e.g. GPU Worker blocked on a long batch), the queue will
eventually fill regardless of message size. The root fix is keeping downstream step
time bounded. Producer batching is a complementary resilience layer that absorbs
short bursts without propagating stalls upstream.

```
Downstream slow for 5s, upstream emits at 180 QPS:

Without batching (N=1):  180 × 5 = 900 messages → hits HWM=1000 in ~5.5s → stall
With batching (N=16):    900/16 = 57 messages  → far from HWM=1000 → no stall

Downstream slow for 60s:
Without batching:  180 × 60 = 10,800 → stall at ~5.5s
With batching:     10,800/16 = 675  → stall at ~88s  ← longer grace period
```

**Throughput bonus — batch tokenization:**

Tokenizers (HuggingFace, SentencePiece) have vectorized batch mode that is
significantly faster than tokenizing one string at a time:

```python
# Single (current):
for request in requests:
    tokens = tokenizer(request.prompt)   # Python loop, no parallelism

# Batched:
batch_tokens = tokenizer(
    [r.prompt for r in requests],
    padding=True, truncation=True
)                                        # vectorized C++ kernel, 5-10× faster
```

The same CPU call tokenizes N requests in roughly the time of 1. This frees the
tokenizer stage to keep up with higher QPS without becoming a bottleneck.

**Where it applies:**

Any pipeline stage where the producer is faster than the consumer and the queue
is message-count bounded. Not limited to ML inference:

```
LLM pipeline:     Generator → Tokenizer batch → Scheduler
Video pipeline:   Frame reader → Feature extractor batch → Embedding model
Kafka pipeline:   Event producer → batch → Consumer group
```

**Tradeoffs:**

```
Pro:  N× more buffer before backpressure
Pro:  Batch processing at consumer is more efficient (one recv() per N requests)
Pro:  Tokenizer throughput improves via vectorized batch mode
Con:  Adds micro-batching latency — producer waits to fill a batch before sending
      (mitigated by a timeout: send when batch full OR after T ms, whichever first)
Con:  Larger individual messages — if one message is dropped, N requests are lost
```

**Metric improved:** Resilience under burst load ↑, Throughput ↑, Latency ↑ (slight,
due to batching wait)

---

## 4. LLM-Specific Optimizations

These apply specifically to autoregressive transformer inference.
Sub-categorized by scope — what hardware boundary the optimization operates within.

---

### 4.1 Node-Level Optimizations

Optimizations that run within a single machine. Processes communicate via ZMQ (`ipc://`)
or shared memory. A single GPU or multiple GPUs on the same node.

| Optimization | What it does | Metric improved |
|---|---|---|
| **Streaming Pipeline** | Tokenizer → Scheduler → GPU Worker as separate processes connected via ZMQ — each stage runs concurrently across requests | Throughput ↑, GPU Utilization ↑ |
| **Continuous Batching** | Scheduler makes batching decisions every decode iteration — EOS frees slot, next waiting request promoted immediately | Throughput ↑, GPU Utilization ↑ |
| **KV Cache** | Cache K,V tensors for past tokens — avoid recomputing attention every decode step | Latency ↓, Throughput ↑ |
| **PagedAttention** | Store KV cache in non-contiguous pages — eliminate fragmentation, enable prefix sharing | Memory Util ↑, Throughput ↑ |
| **Prefix Caching** | Cache KV of shared prefix (system prompt) across requests — skip prefill for repeated prefix | TTFT ↓, Throughput ↑ |
| **FlashAttention** | Tile Q/K/V into SRAM — never materialize N×N attention matrix in HBM | Latency ↓, Memory ↓ |
| **Speculative Decoding** | Draft model generates K tokens speculatively, large model verifies in parallel | ITL ↓, Latency ↓ |
| **GQA / MQA** | Fewer K/V heads — reduces KV cache size per request | Memory ↓, Throughput ↑ |

See detailed doc on scheduler internals, three-queue structure, and all three
memory-optimization techniques: `vllm_scheduler.md`

---

### 4.2 Cluster-Level Optimizations

Optimizations that require multiple machines or dedicated GPU pools.
Communication between pools via RDMA (inter-node) or NVLink (intra-node).
Cannot be applied on a single GPU — the whole point is hardware specialization
across separate pools.

| Optimization | What it does | Metric improved |
|---|---|---|
| **Disaggregated Prefill/Decode** | Separate prefill (compute-bound) and decode (memory-bandwidth-bound) onto dedicated GPU pools — eliminate interference, scale each pool independently | TTFT ↓, ITL ↓, GPU Utilization ↑ |
| **Tensor Parallelism (TP)** | Split model weight matrices across GPUs — each GPU holds a shard, all-reduce after each layer | Latency ↓ (larger models fit, faster per step) |
| **Pipeline Parallelism (PP)** | Split model layers across GPUs/nodes — each GPU holds a layer range, passes activations to next | Throughput ↑ (more model capacity) |

**Why disaggregated prefill/decode is cluster-level:**

Prefill is compute-bound (GEMM, high arithmetic intensity).
Decode is memory-bandwidth-bound (GEMV, low arithmetic intensity).
On a single GPU they share the same compute units and HBM — a long prefill
blocks all decode requests for that entire step (ITL spike, TTFT spike).

Separating them requires two dedicated GPU pools:
```
P-GPU pool (Prefill):    optimized for compute throughput — large GEMM
                         few powerful GPUs, process prompts fast
                              ↓
                    KV cache transfer (RDMA / NVLink)
                              ↓
D-GPU pool (Decode):     optimized for HBM bandwidth — continuous GEMV
                         many GPUs, run large decode batches continuously
```

Scaling is now independent:
- Traffic has long prompts → add P-GPUs
- Traffic has long outputs → add D-GPUs

---

### 4.3 TTFT Optimization Techniques

TTFT = queue_wait + prefill_time. Every technique below attacks one or both components.

---

#### PagedAttention — Reduce queue_wait by fitting more requests in HBM

**Root cause it fixes:** With fixed-size KV slots, each request reserves `max_seq_len`
worth of HBM upfront — even if it only generates 10 tokens. Slots fill up fast.
New requests pile up in the waiting queue. queue_wait grows.

**How it works:** Divide HBM into fixed-size pages (e.g. 16 tokens each). A request
is allocated pages one at a time, only as it generates tokens. Short requests use few
pages and return them immediately on completion.

**Concrete example:**

```
Without PagedAttention — fixed slots (max_seq_len=1024):
  A100 40GB, GPT-2 small
  per slot = 36 MB, available KV HBM = 32 GB
  max concurrent requests = 32,000 / 36 ≈ 910 slots

  At 180 QPS, avg request lifetime = 5s:
  steady-state concurrent = 180 × 5 = 900 requests
  → slots near-full → every new request waits in queue
  → queue_wait = 2–4s for requests arriving at peak

  If avg request uses only 200 tokens (not 1024):
  each slot 80% empty → 29 GB of HBM wasted on padding
  actual useful KV data = 900 × 200/1024 × 36 MB ≈ 6.3 GB out of 32 GB

With PagedAttention — 16-token pages:
  page size = 16 × 12 × 2 × 12 × 64 × 2 = 576 KB
  32 GB / 576 KB ≈ 58,000 pages in free pool

  same 900 requests at 200 tokens avg:
  pages needed = 900 × ceil(200/16) = 900 × 13 = 11,700 pages (6.5 GB)
  free pages remaining = 46,300 → can admit 3,500 more requests
  → queue_wait drops to near 0 at same QPS
  → TTFT = just prefill_time (queue_wait eliminated)
```

---

#### Disaggregated Prefill / Decode — Eliminate prefill-decode interference at cluster level

**Root cause it fixes:** Prefill (GEMM, compute-bound) and decode (GEMV,
memory-bandwidth-bound) have completely different resource profiles but share the
same GPU. A prefill step starves decode requests. Decode steps waste SM compute.
Both TTFT and ITL suffer from this interference.

**How it works:** Separate two GPU pools. P-GPU pool runs prefill only. D-GPU pool
runs decode only. After P-GPU completes prefill for a request, it ships the KV cache
to D-GPU via RDMA/NVLink. D-GPU never pauses for prefill.

**Concrete example:**

```
Shared GPU (standard serving):
  Assume: 900 decode requests running, new request arrives

  Step N:   [900 decode]                       10ms  — decode normal
  Step N+1: [1 prefill = 500 tokens]          200ms  — ALL decode stalls
  Step N+2: [900 decode + 1 new decode]        10ms  — back to normal

  For 900 existing requests: ITL at step N+1 = 200ms instead of 10ms → spike
  For new request:           TTFT = queue_wait + 200ms prefill

Disaggregated P/D:
  P-GPU pool:  receives new request → runs 500-token prefill → 200ms
               ships KV cache to D-GPU via RDMA (~1ms at 400 GB/s)
               ready for next prefill immediately

  D-GPU pool:  Step N:   [900 decode]            10ms  — uninterrupted
               Step N+1: [900 decode]            10ms  — still uninterrupted
               (KV arrives from P-GPU after 200ms)
               Step N+k: [901 decode]            10ms  — new request joins

  For 900 existing requests: ITL = 10ms every step, no spikes
  For new request:           TTFT = prefill_time (200ms) + KV transfer (~1ms)
                                  — queue_wait on D-GPU eliminated if D-GPU
                                    has free capacity

  P-GPU and D-GPU scale independently:
    many long prompts → add P-GPUs
    many long outputs → add D-GPUs
```

**Trade-off:** KV transfer over RDMA adds ~1–5ms to TTFT. Requires high-bandwidth
interconnect. Adds operational complexity (two pools, KV routing). Only worthwhile
at scale where prefill-decode interference is measurable.

---

**TTFT technique summary:**

```
Technique            Attacks         Mechanism
───────────────────  ──────────────  ──────────────────────────────────────────
PagedAttention       queue_wait      Fit more requests per GPU → queue drains faster
Disagg. P/D          queue_wait      D-GPU never stalls for prefill → no queue backup
                     + prefill_time  P-GPU runs dedicated prefill fast (cluster-level)
```
- On a single GPU both pools collapse into one — no isolation possible

Detailed docs: `kv_cache.md`, `PagedAttention.md`, `flash_attention.md`

---

### 4.4 ITL Optimization Techniques

ITL = time per decode step. High ITL means tokens trickle out slowly — streaming
feels choppy even if TTFT was fine.

---

#### Chunked Prefill — Protect existing requests from prefill-induced ITL spikes

**Root cause it fixes:** When a new request is admitted, its full prefill runs in
one dedicated step. A 500-token prefill takes ~200ms. Every decode request in the
running batch misses its 10ms ITL window for that step — a visible stall in their
token stream.

Chunked prefill does NOT reduce the new request's TTFT. Total prefill compute is
the same regardless of how it is chunked. What changes is how that compute is
distributed across steps — protecting existing requests' ITL.

**How it works:** Split prefill into small chunks (e.g. 128 tokens each) and
process one chunk per decode step alongside the existing decode batch. No single
step is monopolized by a full prefill.

**Concrete example:**

```
500-token prefill, no chunked prefill:
  Step N:   [900 decode]              10ms  ← normal ITL
  Step N+1: [1 full prefill]         200ms  ← 900 requests stall, ITL = 200ms
  Step N+2: [901 decode]              10ms  ← back to normal
  New request TTFT = queue_wait + 200ms

500-token prefill, chunked prefill (128 tokens per chunk):
  128 tokens = 200ms × (128/500) ≈ 51ms per chunk

  Step N:   [900 decode + chunk 1/4]   ~51ms  ← ITL bump, not spike
  Step N+1: [900 decode + chunk 2/4]   ~51ms
  Step N+2: [900 decode + chunk 3/4]   ~51ms
  Step N+3: [900 decode + chunk 4/4]   ~51ms → first token for new request
  New request TTFT = queue_wait + 4 × 51ms = queue_wait + 204ms  ← same as before

Without chunked prefill: ITL p99 = 200ms (one big spike every new admission)
With chunked prefill:    ITL p99 = 51ms  (smaller bumps, distributed)

TTFT for the new request is unchanged.
The 900 existing requests are what benefit — their worst-case ITL drops 4×.
```

**Why total prefill time stays the same:**
128-token chunk takes 51ms ≠ "free." Prefill is compute-bound (GEMM). Decode is
memory-bandwidth-bound (GEMV). On a single GPU they share SMs — a chunk of 128
prefill tokens costs 51ms of SM time whether chunked or not. You are just
spreading that 200ms cost across 4 steps of 51ms instead of 1 step of 200ms.

**ITL technique summary:**

```
Technique        Attacks    Mechanism
───────────────  ─────────  ────────────────────────────────────────────────
Chunked Prefill  ITL p99    Spread prefill cost across steps → no single
                            step monopolized → worst-case ITL drops
GQA / MQA        ITL avg    Fewer KV heads → less HBM read per decode step
                            → each step faster → lower average ITL
Speculative dec  ITL avg    Draft model generates k tokens, large model
                            verifies all k in one step → effective ITL / k
```

---

### 4.5 Memory Management in LLM Serving

**The core tension:** KV cache allocation and prefill activation memory compete for the
same HBM. The scheduler is what enforces the tradeoff.

---

#### Two types of GPU memory

```
Model weights       — fixed, loaded once at startup (~500 MB for GPT-2, ~14 GB for 7B)

KV cache            — persistent, pre-allocated at startup, grows with concurrent requests
                      lives in HBM between decode steps
                      per slot = seq_len × layers × 2 × heads × head_dim × dtype
                      GPT-2: 1024 × 12 × 2 × 12 × 64 × 2 bytes = 36 MB per slot
                      910 slots = 32 GB  (our A100 40GB budget)

Activation memory   — temporary, allocated during forward pass, freed immediately after
                      spikes during prefill, tiny during decode
```

---

#### Why prefill and decode have completely different activation footprints

**Prefill** — all input tokens processed at once. Each request contributes `seq_len` tokens:

```
GPT-2 small, batch=910 requests, seq=430 tokens each:

  Q, K, V projections:  [910, 430, 768]  × 3  =  3.6 GB
  Attention score matrix: [910, 12, 430, 430]  =  9.6 GB   ← O(batch × seq²)
  MLP hidden layer:    [910, 430, 3072]       =  14.4 GB
  ─────────────────────────────────────────────────────────
  Total activation peak:                       ~ 18 GB
```

**Decode** — one new query token per request per step. Each request contributes 1 token:

```
GPT-2 small, batch=910 requests, 1 new token each:

  Q, K, V projections:  [910, 1, 768]  × 3  =   2 MB
  Attention scores:     [910, 12, 1, 430]   =   23 MB   ← O(batch × 1 × seq)
  MLP hidden layer:    [910, 1, 3072]       =   11 MB
  ─────────────────────────────────────────────────────────
  Total activation peak:                       ~ 36 MB   ← negligible
```

The difference: **prefill is O(seq) in activation memory per token**, decode is O(1) per token.
KV cache is read from HBM but was already allocated — no new allocation per decode step.

---

#### The burst scenario

This problem surfaces under extreme load — when many requests arrive simultaneously,
fill all KV slots, and all need prefill at once:

```
180 QPS, 910 KV slots, no prefill admission control:

  t=0s:    requests arrive → slots fill in 5s → 910 requests all in WAITING
  t=5s:    scheduler promotes all 910 to RUNNING in one _schedule() call
  t=5s:    GPU Worker receives batch with 910 prefill slots
  t=5s:    GPU attempts 910 × seq=430 token prefill:

    KV cache (already allocated): 32 GB
    Activation memory needed:     18 GB
    Model weights:                 2 GB
    ──────────────────────────────────
    Total:                        52 GB  → OOM on A100 40GB
```

Even without OOM, processing 910 prefills sequentially (one forward pass each):
```
910 requests × 300ms per prefill = 4.5 hours for one batch step
```
Scheduler blocks waiting for result → Tokenizer's ZMQ buffer fills → backpressure
cascades up the entire pipeline.

---

#### FlashAttention's role

FlashAttention tiles the Q×K^T computation into SRAM blocks — the full attention
score matrix is never materialized in HBM:

```
Standard attention:   [910, 12, 430, 430] × 4 bytes = 9.6 GB   ← peak HBM spike
FlashAttention:       [910, 12, 64, 430]  × 4 bytes = 1.1 GB   ← per tile, reused
                      (tile size = 64, reused across tiles, not accumulated)
```

FlashAttention reduces activation from O(seq²) to O(seq × tile_size). But it does not
touch Q/K/V projections or MLP activations, which remain O(batch × seq × d_model).
After FlashAttention:

```
With FlashAttention, batch=910 prefills:
  Q, K, V:        3.6 GB  (unchanged)
  Attention:     ~0 GB    (tiled, never materialized)
  MLP:           14.4 GB  (unchanged)
  KV cache:      32 GB    (unchanged)
  ──────────────────────────────────
  Total:         ~50 GB   → still OOM on A100 40GB
```

FlashAttention helps significantly (eliminates 9.6 GB spike) but does not make
arbitrarily large prefill batches free. MLP activations dominate at large batch sizes.

---

#### Scheduling as the memory enforcer

Since you cannot batch all N prefills at once, the scheduler must limit how many
prefills enter each step. Two mechanisms:

**MAX_NEW_PER_STEP** — limit requests admitted per step:

```python
MAX_NEW_PER_STEP = 1   # at most 1 new prefill per decode step

def _schedule(self):
    promoted = []
    while self.waiting and len(promoted) < MAX_NEW_PER_STEP:
        slot_id = self.block_pool.allocate(request.request_id)
        if slot_id is None:
            break
        ...
        promoted.append(request.request_id)
```

Effect on each step:
```
Step with MAX_NEW_PER_STEP=1:
  1 prefill  × 430 tokens → activation: 4 MB    ← tiny
  909 decodes × 1 token   → activation: 36 MB   ← tiny
  Total: 40 MB on top of 32 GB KV → fits comfortably

  GPU step time: ~300ms prefill + ~10ms decode = ~310ms
  Scheduler unblocks every 310ms → pipeline stays alive
```

Requests ramp up one per step. Steady state (all 910 slots decoding) is reached
slowly, but no OOM and no pipeline stall.

**Chunked prefill** — limit tokens per step instead of requests:

Rather than blocking a full 430-token prefill in one step, break it into 128-token
chunks spread across 4 steps. Finer control over per-step activation budget:

```
Activation per chunk step: 1 request × 128 tokens × d_model = 1.2 MB  ← even smaller
vs full prefill step:       1 request × 430 tokens × d_model = 4 MB
```

Chunked prefill also protects decode ITL: a 128-token chunk costs ~51ms vs 200ms
for a full 430-token prefill — less disruption to the ongoing decode batch.

**vLLM's approach — dynamic admission control:**

Rather than a fixed N, vLLM measures remaining HBM before every step:

```python
# conceptual — not vLLM source
remaining_hbm = total_hbm - kv_allocated - weights
activation_budget = remaining_hbm * SAFETY_FACTOR
tokens_this_step  = activation_budget / activation_per_token
# admit prefill chunks up to tokens_this_step
```

As KV fills up (more concurrent requests), remaining HBM shrinks, fewer prefill
tokens are admitted per step automatically.

---

#### The unified mental model

```
More KV slots allocated (more concurrent requests)
  → less HBM headroom for activation memory
    → must limit prefill tokens per step
      → scheduler admission control enforces this
        → MAX_NEW_PER_STEP or chunked prefill are the levers

Decode at steady state (all slots occupied) is fine:
  activation is O(batch × 1 × d_model) — negligible regardless of slot count.
  The constraint is exclusively during prefill steps.
```

This is why KV cache sizing and scheduling policy are inseparable.
You cannot set `max_slots` without also setting the prefill admission policy —
otherwise a burst fills all slots and the first batch step OOMs or stalls for hours.

---

### 4.6 Knowing Your Workload: How Real Production Systems Handle Limits

**The core insight:** Your KV slot capacity is directly determined by your expected
maximum sequence length. The formula is fixed:

```
KV slots = HBM_budget / (max_seq_len × per_token_KV_cost)

GPT-2 small example:
  32 GB HBM / (1024 tokens × 36 KB/token) = 910 slots    ← our experiment
  32 GB HBM /  (512 tokens × 36 KB/token) = 1820 slots   ← 2× more concurrency
  32 GB HBM /  (256 tokens × 36 KB/token) = 3641 slots   ← 4× more concurrency
```

Halving your max sequence length doubles the number of concurrent users you can
serve from the same hardware — without any code change. This is why workload
knowledge is a first-class infrastructure decision.

---

#### What we learned from our pipeline experiment

We set `max_new_tokens=500` with prompts of ~450 tokens. Max seq grew to ~950.
As seq grew past 736, the float32 padded KV tensors in `_decode_batch` exhausted
remaining HBM and OOMed:

```
KV store (256 slots × 36 MB):            9.2 GB  (fixed)
Decode activation (float32, batch=256):
  24 × [256, 12, 736, 64] × 4 bytes  =  13.3 GB  (grows with seq)
──────────────────────────────────────────────────────────
Total:                                   38.8 GB  → OOM at 39.5 GB A100
```

The fix: cap `max_new_tokens=150`, total seq stays under ~510. Activation stays
under 9 GB. Pipeline runs to completion without OOM.

**The lesson:** knowing that your workload has p99 output length = 150 tokens lets
you set a hard cap that prevents the system from running into a class of memory
problems at all — before the GPU ever sees the request.

---

#### How every major production system handles this

| System | Input cap | Output cap | Where enforced |
|---|---|---|---|
| OpenAI GPT-4o | 128K context window | `max_tokens` param (required) | API gateway |
| Anthropic Claude | 200K context window | `max_tokens` param (required) | API gateway |
| vLLM deployment | `--max-model-len` flag | `--max-new-tokens` flag | Startup config |
| TGI (HuggingFace) | `--max-input-length` | `--max-new-tokens` | Startup config |

**Key pattern:** these caps are enforced at the API gateway or startup config —
requests that exceed the limit are rejected with HTTP 400 before touching the GPU.
The GPU never sees an oversized request.

---

#### What "knowing your workload" means operationally

Before setting any of these limits, production teams profile their traffic:

```
Workload analysis questions:
  What is p50 / p99 input token length?
  What is p50 / p99 output token length?
  What is peak QPS? What is bursty vs steady?
  What is avg request lifetime?

These drive the capacity sizing formula:
  concurrent_requests = QPS × avg_request_lifetime       ← Little's Law
  KV_budget           = concurrent_requests × KV_per_request
  max_seq_len         = p99_input + p99_output + safety_margin
  MAX_SLOTS           = KV_budget / (max_seq_len × per_token_cost)
```

A chatbot workload (short prompts, short replies) and a document summarization
workload (long prompts, medium replies) need completely different hardware configurations
and slot counts — even on the same model.

---

#### Hard reject vs soft cap

Two strategies for handling requests that exceed limits:

**Hard reject (stateless, simple):**
```
Request arrives with input_tokens=600, max_new_tokens=300 → total=900
Server checks: 900 > max_seq_len=512 → return HTTP 400 immediately
GPU never involved. Latency = 1ms (pure API layer check).
```

**Soft cap / truncation (user-transparent, dangerous):**
```
Request arrives with 600 tokens → silently truncate to 512
Risk: model loses context from truncated portion
      user gets wrong answer without knowing why
```

Production systems almost always prefer hard reject over silent truncation.
Failing loudly is easier to debug than wrong answers.

---

#### Why this matters more at larger scale

For GPT-2 small (117M params), the numbers are modest. For production models:

```
Llama-3 70B in FP16:
  Weights: 140 GB  (requires 2× A100 80GB just for weights)
  KV per slot at seq=4096: 4096 × 80 layers × 2 × 8 heads × 128 dim × 2 bytes = 671 MB

  Available for KV on 2× A100 (160 GB total):
    160 GB - 140 GB weights = 20 GB for KV
    20 GB / 671 MB per slot = 29 slots

  With p99 output = 512 tokens instead of 4096:
    KV per slot at seq=512+prompt: ~84 MB
    20 GB / 84 MB = 238 slots ← 8× more concurrency, same hardware
```

Knowing that your users rarely need 4096-token outputs and capping at 512 gives
you 8× more concurrent capacity at no cost. This is the most direct lever an
infra team has before reaching for more GPUs.

---

## 5. Metric → Optimization Map

When the interviewer names a specific problem, map to the right lever:

```
"Latency too high, GPU is underutilized"
  → Batching: more requests per forward pass
  → CUDA Graphs: eliminate CPU dispatch overhead
  → Compilation: fuse kernels, better codegen

"Latency too high, GPU is already at capacity"
  → Quantization: reduce bytes → faster weight reads
  → Speculative decoding: convert sequential decode to parallel verify
  → FlashAttention: faster attention kernel

"Throughput too low"
  → Batching + Bucketing: more work per GPU cycle
  → Continuous batching: no idle GPU slots between sequences
  → Multi-process serving: bypass Python GIL

"Memory OOM / can't increase batch"
  → Quantization: smaller weights
  → PagedAttention: eliminate KV cache fragmentation
  → GQA/MQA: smaller KV cache

"Cost too high"
  → All of the above (cost = GPU time × hourly rate)
  → Traffic shaping: fill idle GPU windows with deferred work
  → Result caching: skip GPU for repeated queries
```

See also: `ml_serving_infra_optimizations.md` for system-level patterns
(result caching, retry policy, PID depth control, traffic shaping).

---

## 6. What Phase 2 Experiments Cover

Phase 2 validates batching empirically — the highest-leverage generic optimization:

```
Experiment: GPT-2 forward pass at batch = 1, 4, 8, 32, 64, 128
Measure:    latency (ms), throughput (tokens/sec), NCU roofline position

Expected transitions:
  batch=1:    occupancy-limited — GEMV, 1D grid, SM utilization < 5%
  batch=4-8:  transitioning — GEMM starts, 2D grid
  batch=32:   memory-bandwidth bound — HBM saturating
  batch=128+: approaching compute-bound (FP16 Tensor Cores)
```

Then: bucketing, compilation (torch.compile), CUDA graphs — each measured
against the same latency/throughput baseline to quantify their individual impact.
