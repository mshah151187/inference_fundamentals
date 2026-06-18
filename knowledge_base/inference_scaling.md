# Inference Scaling

**Interview question:** "Your inference system is hitting its throughput ceiling. How do you scale it?"

The right answer starts with: "Which phase is the bottleneck, and is it stateful or stateless?"
Scaling the wrong phase wastes engineering effort. Scaling a stateful phase without understanding
its state model breaks correctness.

---

## 1. Generic Principles

### Step 1 — Find the bottleneck phase

A pipeline has multiple stages. The slowest stage determines system throughput (Little's Law).
Throwing resources at a non-bottleneck stage does nothing.

```
Profile first:
  - Add per-stage latency logging (what we do with shm_write_ms, shm_read_ms)
  - Look at queue depths — a growing queue upstream of a stage = that stage is the bottleneck
  - GPU: Nsight Systems → is the GPU idle waiting for CPU? or CPU idle waiting for GPU?

Common signals:
  Queue depth growing at scheduler    → GPU worker is the bottleneck
  Tokenizer backlog                   → tokenizer is the bottleneck (rare)
  GPU utilization < 50%               → scheduler not batching fast enough
  HBM full, requests queued           → memory is the bottleneck (not compute)
```

### Step 2 — Is the phase stateful or stateless?

```
Stateless phase:
  Every request is self-contained. The process holds no per-request state
  between calls. Restarting the process loses nothing important.
  Example: tokenizer — given a prompt string, returns token IDs. No memory of past requests.

Stateful phase:
  The process holds mutable state shared across requests.
  Scaling requires either partitioning state or replicating it consistently.
  Example: scheduler — owns the KV block pool, request queue, and in-flight request map.
           Two schedulers without coordination would double-allocate KV slots.
```

### Step 3 — Stateless? Try multi-process on the same node first

```
Scale-up (more processes, same node)   →   try this first
Scale-out (more nodes, load balancer)  →   try this when node is saturated
```

**Why same-node first:**

```
Communication cost:
  Intra-node IPC (Unix socket):   ~1–5 µs
  Intra-node SHM:                 ~0.01–0.05 µs  (no syscall on hot path)
  Inter-node 100GbE:              ~100–200 µs
  Inter-node InfiniBand HDR:      ~1–2 µs (but requires expensive hardware)

For inference, adding 200 µs network hop to every tokenization request
adds directly to TTFT for every user.

Resource sharing:
  Processes on same node share L3 cache, NUMA memory, PCIe bus to GPU.
  Model weights loaded once per GPU — not duplicated per node.
  A 70B model (140 GB BF16) on 2 nodes = 280 GB just for weights.
  Same node, tensor parallel across 2 GPUs = 140 GB total.

Operational simplicity:
  Same node → one process manager, one failure domain, no network partitions.
  Multi-node → health checks, load balancer as new SPOF, distributed tracing.
```

### Step 4 — Scale out when node is saturated

```
Trigger                               Action
────────────────────────────────────  ─────────────────────────────────────────
CPU cores exhausted (tokenizer)       Add tokenizer nodes behind L4 LB
GPU memory full (KV cache, weights)   Add GPU nodes; LB distributes new requests
GPU compute saturated                 Tensor parallel (same node) or pipeline parallel (multi-node)
Need fault tolerance / HA             Multi-node with health checks and failover
Traffic exceeds single node capacity  Fleet of engine instances behind API gateway
```

**Load balancer consideration for stateful phases:**

```
Stateless (tokenizer fleet):
  Any request → any tokenizer node → correct
  Round-robin, least-connections, random — all work

Stateful (GPU engine with KV cache):
  Prefill creates KV cache on Node A.
  Decode MUST go to Node A — KV cache is there.
  If LB sends decode step to Node B → cache miss → full re-prefill → latency spike.

  Solution: sticky sessions (LB pins request_id → node)
  Or: disaggregated KV cache (shared KV store — expensive, but vLLM Proxy + Mooncake does this)
```

### Decision framework summary

```
Bottleneck identified
        │
        ▼
  Stateless?
  ├── No  → Optimize single process (batching, async, algorithmic)
  │          Cannot horizontally scale without state partitioning
  └── Yes → Node resources available?
             ├── Yes → Scale up: add processes on same node
             │          ZMQ PUSH/PULL fan-out + fan-in (no code change in other stages)
             │          SHM: one ring buffer per producer (keep SPSC guarantee)
             └── No  → Scale out: add nodes behind load balancer
                        Stateless: simple round-robin LB
                        Stateful: sticky sessions or disaggregated state store
```

---

## 2. Scaling LLM Inference — Phase by Phase

### Phase overview

```
Request
  │
  ▼
┌────────────────────┐
│  1. Tokenizer      │  CPU — converts prompt string to token IDs
└────────────────────┘
  │
  ▼
┌────────────────────┐
│  2. Scheduler      │  CPU — manages KV block pool, batches requests, routes to GPU
└────────────────────┘
  │
  ▼
┌────────────────────┐
│  3. GPU Worker     │  GPU — runs forward pass (prefill + decode)
│     Prefill        │       prefill: parallel over all input tokens
│     Decode         │       decode: one token at a time, memory-bandwidth bound
└────────────────────┘
  │
  ▼
┌────────────────────┐
│  4. Detokenizer    │  CPU — converts output token IDs back to string
└────────────────────┘
```

---

### Phase 1 — Tokenizer

**What it does:**
Converts a raw prompt string into a list of integer token IDs using a vocabulary
lookup (BPE merge rules). For GPT-2: ~50k vocab, merge rules applied greedily.

```
Input:  "Hello world"
Output: [15496, 995]     ← integer token IDs
```

**Stateful?** No. Each request is fully independent. No shared state between requests.
The tokenizer vocabulary is read-only after loading.

**Bottleneck?** Almost never. GPT-2 tokenization: ~50–200 µs per request.
At 1000 QPS you need < 1 ms/request — easily handled by one process.
HuggingFace's fast tokenizer (Rust-backed) is even faster.

**Scaling techniques:**

```
Technique              When to apply                   How
─────────────────────  ──────────────────────────────  ────────────────────────────────
Multiple processes     If tokenizer IS the bottleneck  N processes, ZMQ PUSH fan-out
(same node)            (rare — very high QPS)          Each PULL connects to generator's PUSH
                                                        ZMQ round-robins automatically

Batch tokenization     Multiple prompts at once        tokenizer(batch_of_prompts)
                                                        HuggingFace supports batched encode()

Fast tokenizer         Always                          tokenizer = AutoTokenizer(use_fast=True)
                                                        Rust-backed, 5–10× faster than Python

Multi-node tokenizer   >100k QPS or special models     Independent fleet behind L4 LB
fleet                  (image+text tokenizer is heavy)  Round-robin — fully stateless
```

**Fan-out + fan-in with ZMQ (same node, N tokenizers):**

```
Generator                     Tokenizer 0 (PULL connect)
PUSH bind  ──round-robin──►   Tokenizer 1 (PULL connect)
                              Tokenizer 2 (PULL connect)

Tokenizer N                   Scheduler
PUSH connect  ──fan-in────►   PULL bind

Generator: unchanged (ZMQ round-robins to available PULLs automatically)
Scheduler: unchanged (PULL bind receives from any connected PUSH)
SHM ring buffer: one per tokenizer (SPSC guarantee preserved)
```

---

### Phase 2 — Scheduler

**What it does:**
- Maintains the request queue (waiting → running → finished)
- Allocates and frees KV cache blocks (PagedAttention block pool)
- Builds batch metadata for the GPU worker each step
- Collects GPU output, updates request state

**Stateful?** Yes — deeply stateful.

```
State it owns:
  block_pool    KV slot allocation map — which GPU memory blocks are free/taken
  waiting       queue of requests pending KV slot allocation
  running       in-flight requests with their KV slot assignments
  request_map   request_id → Request object (tracks generated tokens, latency timers)

Why this prevents naive horizontal scaling:
  Two schedulers each see a private block_pool.
  Both allocate KV slot 5 for different requests.
  GPU worker receives two requests claiming slot 5 → one overwrites the other's KV cache.
  Silent correctness corruption.
```

**Bottleneck?** Can become the bottleneck at high QPS in Python due to:
- Python GIL — only one thread runs at a time
- O(N) scheduling loop over all running requests each step
- Proto serialization for every GPU dispatch

**Scaling techniques:**

```
Technique              When to apply                   How
─────────────────────  ──────────────────────────────  ────────────────────────────────
Larger batch size      GPU utilization < 80%           Increase MAX_NEW_PER_STEP
                                                        More requests per GPU step → fewer
                                                        scheduler→GPU round trips

Async scheduler        Python GIL is the bottleneck    asyncio event loop — overlap
                                                        scheduling logic with GPU execution

Continuous batching    Static batching wastes GPU       Decode step: add new requests mid-flight
(iteration-level)      cycles on padding                vLLM default — new request joins
                                                        next decode step without waiting
                                                        for current batch to finish

Chunked prefill        Long prefill monopolizes GPU     Split prefill into chunks
                       blocking short decode requests   interleaved with decode steps

State partitioning     Extreme scale (rare)             Shard request space by hash(request_id)
(multi-scheduler)                                       Each scheduler owns disjoint KV blocks
                                                        Requires coordination layer
```

**Why you optimize before you scale:**

The scheduler is the central coordinator. Adding more schedulers without state partitioning
breaks correctness. The right path is to make the single scheduler faster:
- Batch larger (more tokens per GPU step)
- Move to async (overlap CPU scheduling with GPU compute)
- Profile Python hot path — serialization and list operations dominate

---

### Phase 3 — GPU Worker (Prefill)

**What it does:**
Runs the forward pass over all input tokens in parallel. Every token attends to every
previous token (full attention). Compute-intensive — scales with sequence length squared.

```
Input:  batch of prompts, each with up to 900 tokens
Output: KV cache written for all input tokens + logit for the last token (next token prediction)
```

**Stateful?** Yes — KV cache written into pre-allocated GPU HBM blocks.
The KV cache persists across the prefill→decode handoff.

**Bottleneck?** Yes, at long prompts or high batch sizes.
Prefill is compute-bound: many tokens, full attention matrix = high arithmetic intensity.

**Scaling techniques:**

```
Technique              When to apply                   How
─────────────────────  ──────────────────────────────  ────────────────────────────────
Tensor Parallelism     Model too large for one GPU     Split attention heads across GPUs
(same node)            or prefill compute-bound        Q,K,V projections sharded column-wise
                                                        AllReduce after each layer
                                                        NVLink: ~600 GB/s → low overhead

FlashAttention         Always                          Fused attention kernel: reads Q,K,V
                                                        once, computes attention in SRAM,
                                                        writes O once — avoids O(seq²) HBM writes

Chunked prefill        Prefill starving decode          Break long prompt into 512-token chunks
                       (TTFT/ITL tradeoff)              Each chunk is one GPU step
                                                        Decode requests interleave between chunks

Prefill/Decode         Prefill and decode interfere     Separate fleets: prefill nodes + decode nodes
Disaggregation         with each other's batching       KV cache transferred via RDMA after prefill
(multi-node)           (different arithmetic intensity) DistServe, Mooncake implement this

Speculative Decoding   Decode is the bottleneck,        Small draft model generates N tokens,
                       not prefill                      large model verifies in one parallel pass
                                                        Effectively parallelizes sequential decode
```

**Tensor Parallelism detail:**

```
Model: 4096-dim, 32 attention heads
Split across 4 GPUs (TP=4):

  GPU 0: heads 0–7    Q,K,V columns 0–1023
  GPU 1: heads 8–15   Q,K,V columns 1024–2047
  GPU 2: heads 16–23  Q,K,V columns 2048–3071
  GPU 3: heads 24–31  Q,K,V columns 3072–4095

Each GPU computes its shard of attention independently.
AllReduce combines partial sums → each GPU has full output.

Communication: AllReduce at each layer = 2 × hidden_dim × 4 bytes per token
NVLink bandwidth: ~600 GB/s → AllReduce for one token ~microseconds
Network (100GbE): ~10 GB/s → AllReduce ~milliseconds → too slow for TP across nodes
→ Tensor Parallelism only works efficiently within one node (NVLink required)
```

---

### Phase 3 — GPU Worker (Decode)

**What it does:**
Generates one token per step, autoregressively. Each step:
reads the full KV cache for all in-flight requests, computes attention, predicts next token.

**Stateful?** Yes — KV cache grows by one row per step per request.
Each decode step reads ALL previous KV vectors (seq_len grows over time).

**Bottleneck?** Yes — the dominant bottleneck in LLM inference.

```
Why decode is memory-bandwidth bound:
  batch=1, one decode step:
    Compute:   2 × seq_len × d_model multiply-adds  = ~234M FLOPs (GPT-2)
    Memory:    read all model weights                = ~234 MB
    Arithmetic intensity: 234M / 234M = 1 FLOP/byte

  A100 roofline: 312 TFLOPS / 2 TB/s = 156 FLOPs/byte
  GPU is 156× underutilized — spending all time waiting for HBM reads.
```

**Scaling techniques:**

```
Technique              When to apply                   How
─────────────────────  ──────────────────────────────  ────────────────────────────────
Larger decode batch    ITL acceptable, want higher TPS  Batch more requests together
                                                         Each weight byte read serves N requests
                                                         Arithmetic intensity × N → compute-bound

Quantization           Memory-bandwidth bound           W4A16 (AWQ/GPTQ): 4× less HBM reads
                                                         W8A8 FP8: 2× less + 2× faster matmul
                                                         See quantization.md

KV cache quantization  HBM full, requests queuing       INT8/FP8 KV cache: 2× more concurrent
                                                         requests in same HBM

PagedAttention         KV cache fragmentation           Non-contiguous KV blocks like virtual
                       wastes HBM                       memory pages — vLLM default

Speculative Decoding   Small batch, latency critical    Draft model proposes K tokens
                       (memory-bound, can't batch more) Large model verifies all K in parallel
                                                         If all accepted: K tokens for price of 1 step

Pipeline Parallelism   Model too large even after TP    Layer 0–15 on Node A, 16–31 on Node B
(multi-node)           Node A's GPU memory insufficient Micro-batching hides inter-node latency
                                                         Requires fast interconnect (InfiniBand)
```

**Why larger batch is the primary lever:**

```
batch=1:   read 234 MB weights, compute 234M FLOPs = 1 FLOP/byte   → memory-bound
batch=32:  read 234 MB weights, compute 7.5G FLOPs = 32 FLOP/byte  → less memory-bound
batch=156: read 234 MB weights, compute 36.5G FLOPs= 156 FLOP/byte → at compute roofline

Same HBM read, proportionally more compute. Throughput scales linearly with batch
until you hit the compute roofline or run out of HBM for KV cache.
```

---

### Phase 4 — Detokenizer

**What it does:**
Converts output token IDs back to a string. Inverse of tokenization.
Typically runs once per generated token (streaming) or once at end of generation.

**Stateful?** No. Each token ID → string lookup is independent.

**Bottleneck?** Almost never. Detokenization is a simple vocabulary lookup — microseconds.

**Scaling techniques:**

```
Technique              When to apply                   How
─────────────────────  ──────────────────────────────  ────────────────────────────────
Fuse with scheduler    Always                          Run detokenization in same process
                                                        as scheduler — avoids IPC overhead
                                                        for a sub-microsecond operation

Streaming output       Long generations                Send each token immediately after decode
                                                        Client sees output as it's generated
                                                        Reduces perceived latency

Multiple processes     Extremely high QPS              Same as tokenizer — ZMQ fan-out
(same node)            (rarely needed)                 Fully stateless
```

---

## 3. Scaling Techniques Summary

```
Phase          Stateful?   Primary bottleneck      Scale-up technique              Scale-out
─────────────  ─────────   ────────────────────    ──────────────────────────────  ──────────────────
Tokenizer      No          Rarely bottleneck       N processes + ZMQ fan-out/in    Stateless fleet + LB
Scheduler      Yes         Python, batching        Async, larger batch, cont batch  State partition (hard)
Prefill        Yes (KV)    Compute (long prompts)  Tensor parallel, FlashAttention  Prefill/decode disagg
Decode         Yes (KV)    Memory bandwidth        Larger batch, quantization       Pipeline parallel
Detokenizer    No          Rarely bottleneck       Fuse with scheduler             Stateless fleet + LB
```

```
General rule for where to invest effort:
  1. Profile → confirm the bottleneck before optimizing anything
  2. Algorithmic first (batching, FlashAttention, quantization) — free gains
  3. Scale up (more processes same node) — cheap, low latency IPC
  4. Scale out (more nodes) — when node is saturated, add LB complexity only then
```
