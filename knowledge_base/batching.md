#### Batching at core

Groups N requests into a single forward pass. Instead of running
the model N times serially, run it once with N inputs stacked into a batch tensor.

#### Goal of batching in overall request serving

Batching aims overall system's throughput improvement. It does NOT focus on latency improvement of a request.
Increasing througput means reducing GPU hours, reducing cost.

```
At batch=1 (GEMV):
  GPU reads all model weights from HBM → produces 1 output
  Arithmetic intensity ≈ 1 FLOP/byte → far below compute-bound threshold
  → paid full HBM bandwidth cost, got 1 result back

At batch=32 (GEMM):
  GPU reads same model weights from HBM → produces 32 outputs
  Arithmetic intensity ≈ 32 FLOPs/byte → approaching memory-bound ridge
  → same HBM cost, 32× the useful work
```

#### Where does batching happen, who does it, and what is the aim?

Batching happens at the boundary between CPU preprocessing and GPU execution.
A dedicated **Scheduler** process sits at this boundary. It receives requests
that have already completed all CPU-side work — in traditional ML serving that
means features fetched and transformed; in LLM serving that means tokenization done.

Important: **the GPU has no process of its own.** Every process in the system is a
CPU process. "GPU worker" means a CPU process whose job is to build tensors and
dispatch CUDA kernels to the GPU. The GPU itself is just a device that executes
those kernels — it has no scheduling or decision-making logic.

```
┌─────────────────────────────────────────────────────────────────┐
│  CPU                                                            │
│                                                                 │
│  [Preprocessing processes]     one per request, short-lived     │
│   Feature Fetch + Transform                                     │
│   or Tokenization                                               │
│          ↓  (request metadata: token IDs, features)            │
│  [Scheduler process]           one, always running             │
│   - accumulates requests                                        │
│   - decides batch composition                                   │
│   - for LLM: tracks KV block state, promotes from waiting queue │
│          ↓  (batch metadata: which requests, block assignments) │
│  [GPU Worker process]          one per GPU, always running      │
│   - receives batch metadata from scheduler                      │
│   - constructs PyTorch tensors (stacks token IDs, block table)  │
│   - calls model.forward() → dispatches CUDA kernels             │
│          ↓  (CUDA kernel launch)                                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  GPU                                                            │
│   - executes CUDA kernels (matrix multiplies, attention, etc.)  │
│   - writes outputs back to HBM                                  │
│   - no process, no scheduling — just a kernel executor          │
└─────────────────────────────────────────────────────────────────┘
```

The scheduler's dual goal — and why it is hard:

```
Goal A: Serve each request as fast as possible   → minimize latency
Goal B: Keep GPU fully utilized at all times      → maximize throughput

These two goals conflict:
  Fast individual request  →  dispatch immediately, small batch, GPU underutilized
  High GPU utilization     →  wait for more requests, large batch, requests stall in queue

Every scheduling decision is a point on this tradeoff curve.
The scheduler's job is to navigate it — never starve the GPU,
never make a single request wait longer than necessary.
```

#### How schedulers can batch requests?

For traditional ML models (rankers, encoders, classifiers), the scheduler's job is simple:
accumulate requests, dispatch a batch, wait for the batch to finish, repeat. It has no
visibility into what happens inside the GPU — it just fires and forgets.

For LLM serving, the scheduler is far more involved. It must:
- Track KV cache block availability — each request holds GPU memory proportional to its sequence length
- Make per-iteration decisions — which waiting requests can be promoted given current free blocks
- Handle preemption — evict running requests if memory pressure is too high
- Balance prefill vs decode — new requests (prefill-heavy) vs existing requests (decode-heavy)
  have very different compute profiles and cannot always be naively mixed

The scheduler is no longer a simple gate. It is a resource manager that makes intelligent
decisions every few milliseconds based on live GPU memory state.

##### Static Batching

Fixed batch size N. Dispatch only when N requests are accumulated. Run all N to
completion together. Simple to implement.

For fixed-computation models (ranking, encoding, classification), the amount of work
per request is deterministic at dispatch time — same architecture, same input shape,
all requests finish at the same step. Static batching works perfectly for these.

```
Sequential Transformer Ranker — batch of 3 requests:
  Req1: 1000 interactions → score   ┐
  Req2: 1000 interactions → score   ├─ all finish at the same forward pass step
  Req3: 1000 interactions → score   ┘

Batch dispatched → runs to completion → all 3 results returned → next batch starts.
No idle slots. No wasted GPU cycles.
```

**Pros:**
- Simple to implement — fixed batch size, no dynamic scheduling logic
- Predictable memory usage — batch size is constant, easy to pre-allocate
- Works perfectly for fixed-computation models — all requests finish together

**Cons:**
- New requests must wait until the full batch of N is accumulated before dispatching
- Under low traffic, GPU sits idle waiting for N requests to arrive

**Why static batching fails for autoregressive LLM generation:**

For text generation, output length is unknown at dispatch time — it depends on the
content of the response. Requests start together but finish at completely different steps.

```
Batch = [Req1 (generates 50 tokens), Req2 (generates 10 tokens), Req3 (generates 30 tokens)]
        ↑ unknown at dispatch time — only discovered when EOS token is generated

After 10 steps: Req2 hits EOS → done, but slot kept alive with padding
                GPU still runs batch of 3 — paying compute for 2 useful + 1 wasted slot
After 30 steps: Req3 hits EOS → done, slot padded
                GPU still runs batch of 3 — 1 useful + 2 wasted slots
After 50 steps: Req1 hits EOS → batch complete → next batch starts

New requests wait until the entire batch finishes — even if most slots are idle.
```

This idle-slot waste is specific to generation. Static and dynamic batching cannot
solve it — they have no mechanism to swap a finished sequence out mid-batch.

##### Dynamic Batching

Accumulate requests up to a timeout OR a max batch size, whichever comes first.
Dispatch as soon as either threshold is hit.

```
Max batch = 8, Timeout = 10ms

t=0ms:  Req1 arrives → start timer
t=3ms:  Req2 arrives
t=7ms:  Req3 arrives
t=10ms: Timeout fires → dispatch [Req1, Req2, Req3] even though batch not full

OR:

t=0ms:  Req1 arrives → start timer
...
t=4ms:  8th request arrives → max batch hit → dispatch immediately, skip timeout
```

**Pros:**
- Adapts to traffic bursts — dispatches quickly under high load (max batch hit), waits briefly under low load (timeout)
- Better latency than static when traffic is sparse — doesn't wait for a fixed N
- Still simple to reason about — two knobs (max batch size, timeout)

**Cons:**
- Timeout adds latency under low traffic — requests wait up to timeout ms even when GPU is free
- Tuning timeout + max batch is workload-dependent; wrong values hurt both latency and throughput
- For fixed-computation models: once dispatched, all requests finish together — no idle-slot issue
- For autoregressive LLM generation: once dispatched, batch runs to completion together —
  inherits the same idle-slot problem as static. Dynamic batching only improves the accumulation
  phase (how you gather requests), not the execution phase (what happens after dispatch)

##### Continuous Batching: Why it is a special requirement for LLM Serving

###### The scheduler is a CPU process — how can it know the state of KV cache on GPU?**

The scheduler never touches actual KV cache tensors. It maintains a **logical mirror** of
GPU memory state purely in CPU RAM.

```
GPU HBM                              CPU RAM (Scheduler process)
─────────────────────────────        ──────────────────────────────────
KV Cache Block 0  [actual tensors]   BlockPool: block_0 → free
KV Cache Block 1  [actual tensors]   BlockPool: block_1 → req_A
KV Cache Block 2  [actual tensors]   BlockPool: block_2 → req_A
KV Cache Block 3  [actual tensors]   BlockPool: block_3 → free
...                                  free_count = 2
```

At startup, the KV cache is pre-allocated on GPU as one large tensor and divided into
fixed-size blocks. The scheduler tracks only metadata — which block IDs are free,
which are assigned to which request. It never reads tensor values.

When the scheduler promotes a waiting request:
  1. Check free_count — a CPU integer, no GPU round-trip
  2. Assign block IDs to the request — update a CPU dict
  3. Send block ID assignments to the GPU worker
  4. GPU worker writes KV tensors into those HBM locations using the IDs

When a request hits EOS:
  1. GPU worker signals completion back to CPU
  2. Scheduler marks those block IDs as free in its CPU metadata
  3. free_count goes up — next waiting request can be promoted immediately

In vLLM source this is:
  - block_pool.py       — BlockPool, FreeKVCacheBlockQueue — pure CPU data structures, integer block IDs only
  - kv_cache_manager.py — allocate_slots(), free()         — CPU methods that update that metadata

KV cache data stays on GPU the entire time. The scheduler has a cheap CPU-side accounting
model that mirrors allocation state. Querying "how many free blocks?" is just reading a
CPU integer — no GPU round-trip needed.


###### What the scheduler manages — three queues

```
Waiting     — request arrived, no KV cache blocks allocated yet, not in any GPU batch
Running     — KV cache allocated, request is actively being decoded in the current batch
Preempted   — was running, evicted because KV cache filled up (swapped to CPU or recomputed later)
```

**What happens when a request arrives:**

```
Request arrives
      ↓
Scheduler puts it in Waiting queue
      ↓
Each step(): scheduler checks if free KV cache blocks exist
      ↓
If yes → allocate blocks → promote request from Waiting → Running
      ↓
Request joins the active batch for the next decode iteration
```

**What happens each decode step (iteration-level scheduling):**

```
decode step N:
  - Run one forward pass over the entire Running batch
  - Each request generates one token
  - Check outputs: did any request generate EOS?
        ↓ yes
  - Free that request's KV cache blocks → return to pool
  - Promote next request from Waiting → Running (blocks now available)

decode step N+1:
  - New request is already in the batch
  - GPU never saw an idle step between the two requests
```

This is the core difference from static/dynamic batching:

```
Static batching:
  [Req1(50 tokens), Req2(10 tokens), Req3(30 tokens)]
  → run all 50 iterations → all done → load next batch
  GPU slots for Req2 idle for iterations 11-50
  GPU slots for Req3 idle for iterations 31-50

Continuous batching:
  Iteration 10: Req2 hits EOS → blocks freed → Req4 promoted immediately
  Iteration 30: Req3 hits EOS → blocks freed → Req5 promoted immediately
  GPU batch always full — no idle slots between sequences
```

The promotion happens **between two consecutive decode steps**, not between two batches.
That is what makes the GPU never idle — the scheduling granularity matches the generation granularity.

Continuous batching is the standard in production LLM serving (vLLM, TGI).
It is specifically a Goal 2 technique — maximizing pipeline utilization across requests.

**Pros:**
- GPU never idles between sequences — slot freed at EOS is filled before next iteration
- Throughput scales with load — more waiting requests → batch stays full continuously
- Lower p99 latency under high load — no request waits for an entire batch to finish

**Cons:**
- Scheduler runs every iteration (every ~few ms) — CPU overhead vs per-batch scheduling
- KV cache management is complex — block allocation/free must be correct at every step
- Prefill and decode have very different compute profiles — mixing them in one batch
  can underutilize GPU (chunked prefill is a further refinement to handle this)

---