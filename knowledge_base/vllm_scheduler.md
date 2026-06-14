# vLLM Scheduler

The scheduler is the brain of an LLM serving system. It decides which requests
run, when they run, and what GPU memory they're allowed to hold. In vLLM, the
scheduler is not a simple queue — it is a **memory resource manager** that makes
per-iteration decisions about every in-flight request.

---

## 1. Why LLM Scheduling is Different

In traditional ML inference (image classification, ranking, embedding), the
scheduler's job is trivial:

```
request arrives → accumulate in batch → dispatch to GPU → GPU done → memory freed
                                                               ↑
                                                  stateless, one-shot forward pass
                                                  memory held for ~milliseconds
```

After the forward pass, all activations and intermediate tensors are freed. The
next batch starts with a clean slate. The scheduler is a traffic cop: batch and
dispatch.

LLM inference breaks this model completely:

```
request arrives → prefill → decode step 1 → decode step 2 → ... → EOS
                     ↑             ↑               ↑
               KV cache       KV grows        KV still in HBM
               allocated      still held      still held

               memory held for SECONDS, across hundreds of steps
```

The KV cache for each request is **persistent state that accumulates for the
entire generation lifetime**. 8 in-flight requests = 8 KV caches permanently
occupying HBM until EOS.

| | Traditional ML | LLM Serving |
|---|---|---|
| Per-request HBM | Transient (one pass) | Persistent (entire generation) |
| Memory pressure | Resets after every batch | Accumulates with new arrivals |
| Scheduler role | Traffic cop — batch and dispatch | Hotel manager — rooms held until checkout |
| Admission check | Does batch fit in memory? | How many rooms currently occupied? |

The scheduler cannot simply "send more requests to GPU" when compute is free.
It must first ask: is there HBM available to hold this request's KV cache for
however long it takes to generate its output?

---

## 2. Three-Queue Structure

vLLM's scheduler maintains three queues at all times:

```
┌─────────────────────────────────────────────────────────────┐
│  WAITING        RUNNING              PREEMPTED              │
│  ─────────      ─────────────────    ───────────            │
│  req_5          req_1 (decode)       req_3 (paused)         │
│  req_6          req_2 (decode)       req_4 (paused)         │
│  req_7          req_8 (prefill)                             │
│                 req_9 (decode)                              │
│                                                             │
│  no KV slot     KV slot allocated    KV slot freed          │
│  not started    actively running     evicted from HBM       │
└─────────────────────────────────────────────────────────────┘
```

**WAITING** — tokenized requests with no KV slot yet. Arrival order preserved.
New requests join here and wait until HBM is available.

**RUNNING** — requests with an allocated KV slot, actively generating tokens.
Each step, every running request contributes one entry to the dispatch batch
(decode) or a full prefill pass.

**PREEMPTED** — requests that were running but got evicted. Their KV blocks
are freed from HBM to make room for higher-priority requests. They can be
resumed later (KV rebuilt via recompute or swap to CPU RAM).

---

## 3. Per-Step Scheduling Decisions

The scheduler runs a tight loop. Every iteration:

```
step():
  1. _drain_incoming()        ← pull new tokenized requests → WAITING

  2. _schedule()              ← try to promote WAITING → RUNNING
                                 for each waiting request (FIFO):
                                   if block_pool.allocate() succeeds → RUNNING
                                   else → stop (HBM full, rest stays in WAITING)

  3. _build_batch_metadata()  ← pack all RUNNING requests:
                                   prefill requests → full token_ids, is_prefill=True
                                   decode requests  → last generated token only

  4. dispatch to GPU Worker   ← GPU Worker runs forward pass, returns next tokens

  5. _update_from_outputs()   ← for each result:
                                   append generated token
                                   update kv_seq_len
                                   if is_finished:
                                     block_pool.free(slot_id)   ← slot released NOW
                                     move to FINISHED
                                     → next step's _schedule() can fill that slot
```

The slot freed in step 5 is visible to step 2 of the NEXT iteration. This is
what makes the system "continuous" — a new request can join the running batch
within one decode step of a previous request finishing.

### Preemption

When a new high-priority request needs a slot but none are free, the scheduler
can preempt a running request:

```
1. Pick victim request (e.g. lowest priority or longest running)
2. Free its KV blocks from HBM
3. Move it to PREEMPTED queue
4. Allocate freed slot to new request
5. When slot becomes available again, resume preempted request
   (recompute KV from scratch, or swap back from CPU RAM)
```

Recompute is wasteful (redo all prefill work). CPU swap avoids recompute but
requires PCIe bandwidth. Neither is free — preemption is a last resort.

---

## 4. The Core Constraint: Admission is Memory-Gated

```python
slot_id = self.block_pool.allocate(request.request_id)
if slot_id is None:
    break   # waiting queue stalls regardless of GPU compute state
```

Even if the GPU has 90% SM utilization headroom, no new request is admitted
without a free KV slot. In a steady-state decode workload (all requests in
decode phase), the GPU is:

```
SM compute:       ~5–10% utilized   (GEMV is memory-bound, barely touches SMs)
HBM bandwidth:    saturated          (reading 8 KV caches per step)
Admission gate:   CLOSED             (no free slots)
→ SMs idle while the queue grows
```

The system is bounded by memory, not by compute. This is the fundamental
inefficiency the techniques below address.

---

## 5. Techniques to Improve Utilization and Latency

### 5.1 PagedAttention — Shrink Per-Request HBM Footprint

**Problem:** In naive KV slot allocation, each request reserves `max_seq_len`
worth of KV cache at admission time, even if it only generates 10 tokens.

```
naive slot:   req holds 2048-token KV from start → 90% of slot wasted early on
              8 slots × 2048 × layers × heads × head_dim → large HBM chunk always reserved
```

**PagedAttention** divides KV cache into fixed-size pages (e.g. 16 tokens each).
A request only holds pages proportional to tokens generated so far:

```
page-based:   req at step 50 → holds 4 pages (50 tokens ÷ 16 per page)
              same HBM fits many more in-flight requests
              pages returned to free pool as soon as request finishes

              more requests fit → queue drains faster → lower TTFT
              slots turn over faster → higher throughput
```

Pages are non-contiguous in physical HBM (virtual → physical mapping). This also
eliminates internal fragmentation from variable-length sequences.

### 5.2 Chunked Prefill — Fill Idle SM Cycles

**Problem:** Waiting requests contribute zero GPU work until a slot is free.
Meanwhile, decode steps leave SMs largely idle (GEMV is memory-bound).

**Chunked prefill** splits a request's prefill into small chunks and interleaves
them into decode steps:

```
without chunked prefill:
  decode step 1: [req1 decode, req2 decode, req3 decode]   SM: 5%
  decode step 2: [req1 decode, req2 decode, req3 decode]   SM: 5%
  slot frees →
  decode step 3: [req4 FULL PREFILL]                       SM: 80% spike, then done

with chunked prefill:
  decode step 1: [req1 decode, req2 decode, req4 prefill chunk 1/4]   SM: 25%
  decode step 2: [req1 decode, req2 decode, req4 prefill chunk 2/4]   SM: 25%
  decode step 3: [req1 decode, req2 decode, req4 prefill chunk 3/4]   SM: 25%
  decode step 4: [req1 decode, req2 decode, req4 prefill chunk 4/4]   SM: 25%
  → req4's first token ready sooner, SMs kept busier throughout
```

Effect: lower TTFT for waiting requests + higher sustained SM utilization.
Tradeoff: slightly longer individual decode steps (sharing with prefill chunk).

### 5.3 Disaggregated Prefill / Decode — Separate Compute Profiles

**Problem:** Prefill and decode have completely different compute profiles but
share the same GPU in standard serving.

```
Prefill:  process full prompt at once → GEMM (matrix × matrix) → compute-bound
Decode:   generate one token → GEMV (matrix × vector) → memory-bandwidth-bound

Running both on the same GPU:
  - prefill step hogs SMs → decode requests stall → EOS delayed → slots held longer
  - decode steps waste SMs → SM idles while waiting for HBM
```

**Disaggregated serving** separates these into two pools:

```
┌──────────────────────────────────────────────────────────┐
│  P-GPU pool (Prefill)        D-GPU pool (Decode)         │
│  ────────────────────        ─────────────────           │
│  compute-bound GEMM          memory-bandwidth GEMV        │
│  runs full prefill           batches many decode steps    │
│  generates KV cache          receives KV via RDMA/NVLink  │
│                                                           │
│  sized for compute           sized for HBM bandwidth      │
└──────────────────────────────────────────────────────────┘
```

When P-GPU completes prefill, KV cache is transferred to D-GPU via RDMA/NVLink.
D-GPU runs decode independently without interference from new arrivals.

Effect:
- Decode throughput unaffected by new prefill requests
- Slots on D-GPU turn over faster (no prefill spikes stalling decode)
- Each pool can be independently scaled for its bottleneck

Tradeoff: KV transfer latency over network adds to TTFT. Requires high-bandwidth
interconnect (NVLink or InfiniBand) to keep transfer cost low.

---

## 6. Summary

```
Root cause: LLM requests hold HBM for seconds (KV cache persists across steps)
            → admission is memory-gated, not compute-gated
            → SMs idle while queue grows

Technique           Attacks                          Mechanism
─────────────────   ──────────────────────────────   ──────────────────────────
PagedAttention      HBM waste per request            Pages instead of slots;
                                                     only hold what's generated
Chunked Prefill     SM idleness during decode        Interleave prefill chunks
                                                     into decode steps
Disagg. P/D         Prefill-decode interference      Separate GPU pools per
                    on shared GPU                    compute profile
```

All three reduce time a request spends waiting before its first token (TTFT) and
improve GPU utilization — from different angles, attacking the same root problem.
