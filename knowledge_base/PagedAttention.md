# PagedAttention

Reference: kv_cache.md for KV cache fundamentals.

---

## Two Problems with Naive KV Cache Management

### Problem 1 — Dynamic Growth via `aten::cat` (Copying)

The simplest PyTorch implementation grows the KV cache by concatenating the new
token's K/V onto the existing tensor after each decode step:

```python
# After computing K_new, V_new for the current token:
k_cache = torch.cat([k_cache, K_new], dim=1)   # allocate + copy
v_cache = torch.cat([v_cache, V_new], dim=1)
```

`torch.cat` is **not in-place**. For each decode step at sequence position `t`:
1. Allocates a new tensor of size `t+1`
2. Copies all `t` existing K/V vectors into it
3. Writes the new K/V at position `t`
4. Frees the old tensor

Copy cost grows with sequence length — step 10 copies 10 rows, step 49 copies 49 rows.
Cumulative allocation is **O(seq_len²)** across the full decode loop.

**Observed in profiling (GPT-2 medium, 10 prompts, max_tokens=50):**
```
aten::cat  CUDA memory: 1.12 GB
           = cumulative allocation from copying across all 50 steps × 24 layers × 10 prompts
```

**Fix direction:** eliminate copying by pre-allocating the KV cache buffer upfront
and writing new K/V in-place. No allocation, no copy — O(1) per step.

---

### Problem 2 — Memory Bubble from max_seq_len Pre-allocation

Pre-allocation eliminates copying, but naive pre-allocation introduces a new problem.

Since sequence length is unknown at request start (user controls when to stop generating),
the system must reserve space for the worst-case length — `max_seq_len`:

```python
# Pre-allocate for worst case
k_cache = torch.zeros(n_layers, max_seq_len, n_heads, head_dim)
v_cache = torch.zeros(n_layers, max_seq_len, n_heads, head_dim)
```

For GPT-2 medium with max_seq_len=2048:
```
Per request: 24 layers × 2048 tokens × 16 heads × 64 dim × 2 bytes × 2 (K+V)
           = 24 × 2048 × 16 × 64 × 2 × 2 = ~201 MB reserved

Actual usage if request generates 50 tokens:
           = 24 × 50 × 16 × 64 × 2 × 2 = ~4.9 MB used

Memory wasted: 201 MB - 4.9 MB = ~196 MB (97.5% waste)
```

This wasted reserved-but-unused HBM is the **memory bubble**.

At scale with 100 concurrent requests:
```
Naive pre-allocation: 100 × 201 MB = ~20 GB reserved
Actual usage:        100 × 4.9 MB  = ~490 MB used
~19.5 GB of HBM locked and idle
```

The memory bubble directly limits how many requests can run concurrently —
a GPU with 40 GB HBM can only serve ~200 requests instead of potentially thousands.

---

## The Core Tension

```
Dynamic cat  →  no wasted memory, but O(seq_len²) copying cost
Pre-allocate →  O(1) append, but wastes HBM proportional to (max_seq_len - actual_len)
```

The question becomes: **how can we pre-allocate effectively without knowing sequence
length in advance?**

Answer: don't pre-allocate per-request. Pre-allocate a shared pool and assign chunks
on demand — exactly how OS virtual memory manages RAM.

---

## PagedAttention — Block-Based Pre-allocation

PagedAttention (vLLM, 2023) solves both problems simultaneously by borrowing the
OS paging model:

- HBM is divided into fixed-size **blocks** (e.g., 16 tokens per block) — pre-allocated once at startup
- Each request is assigned blocks from the pool **as needed** — one block at a time
- Blocks for a single request need not be contiguous in physical HBM
- A per-request **block table** maps logical block number → physical block address
- New K/V written **in-place** into the current block's next free slot — O(1), no copy
- When a block fills up, one new block is claimed from the pool — no reallocation of existing data

```
Pre-allocated HBM pool (startup, once):
  [ Block 0 ][ Block 1 ][ Block 2 ][ Block 3 ][ Block 4 ][ Block 5 ] ...
    free        free       free       free       free       free

Request A arrives (generates tokens one by one):
  token 0-15:  assigned Block 0  → block table: { 0 → Block 0 }
  token 16-31: assigned Block 1  → block table: { 0 → Block 0, 1 → Block 1 }
  token 32:    assigned Block 3  → block table: { 0 → Block 0, 1 → Block 1, 2 → Block 3 }

Request B arrives concurrently:
  token 0-15:  assigned Block 2  → block table: { 0 → Block 2 }
  (Block 2 is non-contiguous with Request A's blocks — that is fine)
```

**Memory waste bounded:** at most `block_size - 1` tokens wasted in the last block
per request (15 tokens for block_size=16), regardless of max_seq_len.

**Append cost:** O(1) — write in-place to next free slot, no copy.

For a detailed walkthrough of the block table and attention kernel, see the
PagedAttention paper: vLLM (Kwon et al., 2023).

---

## What This Eliminates from the Profiler

```
Before (naive cat):   aten::cat  1.12 GB CUDA memory, top-3 CUDA time op
After  (PagedAttention): aten::cat absent — replaced by in-place block writes
                         KV cache ops drop from top-20 CUDA time ops
```

---

## Future Work (This Project)

1. Download vLLM source code
   - Study `vllm/attention/backends/` — PagedAttention CUDA kernel
   - Study `vllm/core/block_manager.py` — block table and pool management
   - Study `vllm/worker/cache_engine.py` — HBM pool initialization

2. Implement a simple KV cache in Inference_Fundamentals:
   - Phase 1: fixed pre-allocation (eliminate cat, accept memory bubble)
   - Phase 2: block-based allocation (PagedAttention mechanics, no custom CUDA kernel)
   - Profile each phase and compare `aten::cat` memory footprint across implementations

3. Measure the impact:
   - CUDA memory allocated by KV ops
   - Max concurrent requests per GPU at fixed HBM budget
   - Throughput (tokens/sec) vs naive implementation
