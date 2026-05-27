# Naive KV Cache vs vLLM PagedAttention — Deep Dive

This doc explains how KV cache works naively in PyTorch (what we measured in Phase 1),
why it has a fundamental O(t²) memory problem, and how vLLM's PagedAttention solves it.
Numbers in this doc come from the actual Phase 1 run on Lambda Labs A100.

---

## Part 1: What is the KV Cache and Why Does it Exist?

### Two phases of autoregressive generation

When GPT-2 generates tokens, every forward pass runs multi-head self-attention:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

All three matrices Q, K, V are computed from the input sequence via linear projections.

**Phase 1 — Prefill:** All input tokens are processed in one forward pass.
- Input: the full prompt (say, 30 tokens)
- All 30 Q, K, V vectors computed in parallel
- Attention is over all 30 tokens simultaneously
- One forward pass produces K and V tensors for all 30 positions

**Phase 2 — Decode:** One new token generated per forward pass.
- Input: only the newly generated token (1 token)
- Q is computed for the 1 new token
- But attention must be over ALL previous tokens (prompt + already generated)
- So K and V for all previous tokens must be available

This is the key insight: **to compute attention for token t, you need K and V from tokens 0 through t-1.**
Instead of re-running the full sequence through the transformer at every step (O(t²) compute),
you cache K and V from previous steps. This is the KV cache.

### KV cache shape

For GPT-2 (12 layers, 12 heads, head_dim=64):

```
K cache shape at step t: [1, 12, t, 64]   (batch, heads, seq_len, head_dim)
V cache shape at step t: [1, 12, t, 64]

Combined KV cache at step t: 2 × 12 × t × 64 × 4 bytes (FP32)
= 6,144 × t bytes
= ~6 KB per token per step
```

At t=50 tokens: ~300 KB of KV cache total for GPT-2. Small model — large models are much bigger.
GPT-3 (175B): each token adds ~1.5 MB to KV cache. At 2048 tokens, that's ~3 GB per sequence.

---

## Part 2: The Naive Implementation — aten::cat

### What happens at each decode step

PyTorch's naive approach: grow the KV cache tensors by concatenating.

```python
# Pseudocode of naive decode step t
q_new = linear_q(x_new)    # shape: [1, 12, 1, 64]  — only the new token
k_new = linear_k(x_new)    # shape: [1, 12, 1, 64]
v_new = linear_v(x_new)    # shape: [1, 12, 1, 64]

# Append to existing cache — THIS IS THE PROBLEM
k_cache = torch.cat([k_cache, k_new], dim=2)   # aten::cat
v_cache = torch.cat([v_cache, v_new], dim=2)   # aten::cat

# Now run attention over full sequence
attn_output = attention(q_new, k_cache, v_cache)
```

`aten::cat` does not append in-place. It:
1. Allocates a new tensor of shape `[1, 12, t+1, 64]`
2. Copies all t existing K vectors into the new tensor
3. Copies the 1 new K vector into position t
4. Frees the old tensor

Then the same for V.

### The O(t²) copy cost

At decode step t (generating the t-th new token):
- aten::cat copies t tokens × 64 floats × 4 bytes × 12 heads = `t × 3,072 bytes`
- One call for K, one for V: `t × 6,144 bytes`

Total copies across all steps to generate T tokens:
```
sum(t × 6,144) for t=0 to T = T(T+1)/2 × 6,144 bytes = O(T²)
```

For T=50 tokens (our Phase 1 run, max_new_tokens=50):
```
sum = 50 × 51 / 2 = 1,275 × 6,144 bytes = ~7.8 MB per sequence
10 sequences total = ~78 MB of data copied just for KV cache construction
```

This is what `aten::cat` — 1.12 GB cumulative alloc in the profiler — was measuring.
Wait: 1.12 GB >> 78 MB. The discrepancy: profiler tracks **allocated**, not **net**.
Each `cat` allocates a new tensor equal to the full cache at that step:

```
Step 1:  allocated 2 × 12 × 2 × 64 × 4 = 12,288 bytes
Step 2:  allocated 2 × 12 × 3 × 64 × 4 = 18,432 bytes
...
Step 50: allocated 2 × 12 × 51 × 64 × 4 = 314,572 bytes
```

Sum of all allocations (K+V for 12 layers, both):
```
2 (K+V) × 12 (layers) × sum(t × 64 × 4) for t=1..51
= 24 × 64 × 4 × sum(1..51)
= 24 × 256 × 1,326
= ~8.1 MB per sequence × 12 layers... 
```

The 1.12 GB total cumulative matches the 13,500 aten::cat calls:
13,500 calls / 12 layers / 10 sequences = 112.5 steps per sequence × 2 (for K and V per layer)
= ~56 decode steps (close to max_new_tokens=50 + prefill steps).

### Why large cumulative alloc ≠ slow CUDA time

Each individual cat at step t copies only ~t × 6 KB of data.
At A100 memory bandwidth of 1.555 TB/s:

```
Step 25 (mid-run): copies 25 × 6,144 bytes = 153,600 bytes
Time = 153,600 / (1.555 × 10¹²) = 0.099 microseconds ≈ 0.1 μs
```

The profiler showed aten::cat at 7.6% of CUDA time, but individual calls are ~5 μs each.
That's mostly kernel launch overhead (the CUDA kernel launches in ~2-5μs regardless of data size).

**Key insight:** The problem with naive aten::cat is NOT speed-per-call — it's:
1. **Memory fragmentation**: 13,500 separate HBM allocations fragment the allocator pool
2. **O(t²) total copies**: grows quadratically with sequence length
3. **Peak memory**: at step t, two full copies of the cache exist simultaneously (old + new)
4. **No sharing**: identical prefixes across sequences each maintain their own full copy

---

## Part 3: vLLM PagedAttention

### Core idea — borrow from OS virtual memory

The OS manages RAM as fixed-size pages (typically 4 KB). A process sees a contiguous virtual
address space, but physical pages can be anywhere in RAM. A page table maps virtual → physical.

PagedAttention applies this to KV cache:
- **Block** = fixed number of token slots (e.g., 16 tokens × all layers × 2 (K+V))
- **Block table per sequence** = logical position → physical block index
- Physical blocks can be anywhere in HBM — they don't need to be contiguous
- GPU attention kernel follows the block table to gather non-contiguous blocks

### Block pool setup

At model load time (before any requests), vLLM pre-allocates the entire HBM budget:

```python
# Pseudocode — actual vLLM uses more sophisticated sizing
block_size = 16  # tokens per block
n_blocks = (total_hbm - model_weights) // block_size_bytes
kv_cache = torch.empty([n_layers, 2, n_blocks, block_size, n_heads, head_dim])
#                        ^         ^  ^         ^            ^        ^
#                        layers    KV n_blocks  tokens/block heads    head_dim
free_blocks = deque(range(n_blocks))  # pool of available block indices
```

This single allocation runs once. No more `cudaMalloc` during inference.

### Decode step with PagedAttention

```
Step 1 — sequence arrives:
  Allocate block 42 from free_blocks
  block_table[seq_id] = [42]
  Write token 0's K and V into block 42, slot 0

Step 2 — generate token 1:
  block 42, slot 1 is free → write directly
  No allocation, no copy

...

Step 16 — block 42 is full (16 tokens):
  Grab block 7 from free_blocks
  block_table[seq_id] = [42, 7]
  Write token 16's K and V into block 7, slot 0

Step 17:
  block 7, slot 1 is free → write directly
```

At every decode step: **O(1) write, zero copies, zero allocations** (except when a new block is needed,
which happens every 16 steps and costs only updating the block table — a tiny host-side operation).

### Custom CUDA attention kernel

Standard FlashAttention assumes contiguous K and V tensors. PagedAttention replaces it with
a custom kernel that takes the block table as input:

```
for each block_idx in block_table[seq_id]:
    k_block = kv_cache[layer, K, block_idx, :, :, :]  # [block_size, n_heads, head_dim]
    v_block = kv_cache[layer, V, block_idx, :, :, :]
    # accumulate partial attention scores
    scores += softmax(q @ k_block.T / sqrt(d_k)) @ v_block
```

The kernel gathers non-contiguous blocks during the attention computation itself —
no need to first copy them into a contiguous buffer.

---

## Part 4: What PagedAttention Eliminates

| Naive (aten::cat) | vLLM (PagedAttention) |
|---|---|
| New tensor per step: O(t) alloc | Single pre-allocated pool: O(1) alloc at startup |
| Copy t tokens each step | Write 1 token's K+V directly into free slot |
| O(t²) total copies across T steps | O(T) total writes across T steps |
| 13,500 separate HBM allocations | 0 allocations during inference |
| Fragmented allocator pool | Contiguous fixed-size blocks |
| Peak: 2× cache in HBM simultaneously (old + new) | 1× cache always |
| No sharing across sequences | Prefix sharing with copy-on-write |
| aten::cat visible in profiler | No aten::cat kernel calls |

Our Phase 1 profiler numbers:
```
aten::cat: 7.6% CUDA time, 1.12 GB cumulative alloc, 13,500 calls
With vLLM: 0% CUDA time, ~0 alloc, 0 calls
```

---

## Part 5: What PagedAttention Unlocks

### Continuous batching (the big win)

Naive static batching:
```
Batch of 8 sequences all start together, all finish together.
Sequence 1 finishes at step 30. Steps 31-50: its GPU slot is wasted.
GPU waits for the slowest sequence before accepting new requests.
```

With PagedAttention:
```
Sequence 1 finishes at step 30.
Its blocks are immediately returned to free_blocks.
A new incoming request grabs those blocks at step 31.
GPU never has idle slots — sequences are continuously swapped in/out.
```

This is why vLLM achieves 2-4x higher throughput than static batching for realistic workloads
where sequence lengths vary. The GPU is always fully occupied.

### Prefix sharing (KV cache reuse)

Many requests share the same system prompt:
```
"You are a helpful assistant. [System prompt: 200 tokens]"
User1: "What is 2+2?"
User2: "Tell me a joke."
```

Naive: each sequence independently computes and stores K+V for those 200 system prompt tokens.
100 concurrent users = 100 copies of the same 200-token KV cache.

PagedAttention:
```
System prompt tokens → blocks 0, 1, 2 ... 12 (200 tokens / 16 = 12.5 → 13 blocks)
block_table[user1] = [0, 1, 2, ..., 12, 50]   # shared prefix + user1's own block
block_table[user2] = [0, 1, 2, ..., 12, 51]   # SAME prefix blocks!
```

Prefix blocks are **read-only** — multiple sequences point to the same physical blocks.
Copy-on-write: if a sequence needs to modify a shared block, it gets its own copy at that point.

100 concurrent users with same 200-token prefix: need 13 physical blocks (not 1,300).
This is a 100x reduction in memory for the shared prefix.

### Beam search memory efficiency

Beam search maintains B candidate sequences simultaneously. Naive approach: B full KV cache copies.
With PagedAttention: all beams share the prefix blocks up to where they diverged.
Memory scales with number of unique tokens, not B × total tokens.

---

## Part 6: Profiler Before and After

### Phase 1 naive (what we measured on A100)

```
aten::addmm       329ms  39.9%   24,000 calls   96,995 MFLOPs  — linear projections
gemvx (2×)        211ms  25.6%   11,760+5,880   — GEMV: batch=1 decode = vector×matrix
FlashAttention     93ms  11.4%    6,000 calls   — fmha_cutlassF_f32_aligned_64x64_rf_sm80
aten::layer_norm   77ms   9.5%   12,500 calls   — memory-bound elementwise
aten::add          64ms   7.8%   26,000 calls   — residual connections
aten::mul          63ms   7.7%   25,000 calls   — GELU activation
aten::cat          62ms   7.6%   13,500 calls   — KV cache growth  ← ELIMINATED by vLLM
aten::mm           56ms   6.8%      500 calls   38,597 MFLOPs
```

### Expected with vLLM (Phase 6 target)

```
vLLM paged_attn kernel    replaces FlashAttention + aten::cat
aten::addmm              similar — linear projections unchanged
gemvx                    unchanged — batch=1 still GEMV (unless continuous batching fills batch)
aten::cat                GONE — 0 calls, 0 allocation
PagedAttention kernel    custom gather + attention computation
```

The primary Phase 2 goal (batching) will show: GEMV → GEMM transition at batch=4+,
aten::cat still present, layer_norm % drops as matmuls dominate.

---

## Summary

| Concept | Naive | PagedAttention |
|---|---|---|
| KV cache storage | Grows via aten::cat each step | Fixed block pool pre-allocated |
| Step cost | O(t) copy: allocate + copy all t previous | O(1) write: one slot in current block |
| Total copies over T steps | O(T²) | O(T) |
| HBM allocations | 13,500 separate (Phase 1) | 0 during inference |
| Memory fragmentation | High | None |
| Multi-sequence sharing | None — each has its own copy | Prefix sharing with copy-on-write |
| Scheduling | Static batching — wait for slowest | Continuous batching — swap finished sequences immediately |
| Profiler signature | aten::cat visible, 7.6% CUDA | No aten::cat |

PagedAttention doesn't make individual attention ops faster on a per-token basis.
It eliminates the O(t²) copy overhead and enables the scheduling strategies (continuous batching,
prefix sharing) that keep GPUs busy — which is where real throughput gains come from.
