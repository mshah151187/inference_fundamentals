# FlashAttention — Basics, Efficiency, Kernel Launch, and Backend Comparison

Observed in profiling_result.txt:
```
fmha_cutlassF_f32_aligned_64x64_rf_sm80   93.856ms   11.39% CUDA   6000 calls
```

---

## 1. Standard (Naive) Attention — The Problem

For each transformer layer, attention computes:

```
Q, K, V ∈ [batch, n_heads, seq_len, head_dim]

Step 1: S = Q @ K.T / sqrt(d_k)   → [batch, n_heads, seq_len, seq_len]  ← NxN matrix
Step 2: P = softmax(S)             → [batch, n_heads, seq_len, seq_len]  ← NxN matrix
Step 3: O = P @ V                  → [batch, n_heads, seq_len, head_dim]
```

**Memory problem:** Steps 1 and 2 materialize the full `seq_len × seq_len` attention
matrix in HBM. For seq_len=1024, n_heads=12, FP32:
```
1024 × 1024 × 12 × 4 bytes = 50 MB per layer
```

**HBM round-trip problem:** Each step reads/writes the NxN matrix to HBM:
```
1. Write S (NxN) to HBM after Q @ K.T
2. Read S from HBM to compute softmax
3. Write P (NxN) to HBM after softmax
4. Read P from HBM to compute P @ V
= 4 HBM accesses for the NxN matrix — O(N²) IO cost
```

HBM bandwidth is the bottleneck (~1.6 TB/s on A100). Writing and reading 50 MB × 4
times = 200 MB of HBM traffic per layer, per forward pass — purely for intermediate
buffers that are thrown away.

---

## 2. FlashAttention — The Key Insight

**Observation:** GPU SRAM (shared memory, ~96 KB/SM on A100) is 10–20× faster than HBM
but tiny. If we can tile the attention computation to fit inside SRAM, we never need to
write the NxN matrix to HBM.

**Idea:** Process attention in tiles. For each tile of Q, iterate over tiles of K and V,
accumulate the output incrementally — and keep everything in SRAM registers.

```
Tile sizes:  Br = 64 (query block), Bc = 64 (key/value block)

For each Q tile Q_i (64 rows):
    For each K,V tile K_j, V_j (64 rows):
        S_ij = Q_i @ K_j.T / sqrt(d_k)    ← 64×64 tile, fits in SRAM
        Update running softmax (online trick — see below)
        O_i += softmax(S_ij) @ V_j         ← accumulate in registers

Write O_i to HBM once, move to next Q tile
```

The full NxN matrix is **never written to HBM.** Only tile-sized buffers live in SRAM.

---

## 3. Concrete Tile-by-Tile Example

**Setup:** seq_len = 4, head_dim = 2, tile size Br = Bc = 2 (2 query rows, 2 key rows per tile)

```
Q (4×2):          K (4×2):          V (4×2):
q0 = [2, 0]       k0 = [1, 0]       v0 = [1, 0]
q1 = [0, 2]       k1 = [0, 1]       v1 = [0, 1]
q2 = [1, 1]       k2 = [1, 1]       v2 = [1, 1]
q3 = [0, 0]       k3 = [0, 0]       v3 = [0, 0]
```

The full 4×4 attention score matrix S = Q @ K.T (what naive attention materializes in HBM):

```
         k0   k1   k2   k3
q0  S = [ 2    0    2    0 ]   ← dot products of q0 with every key
q1      [ 0    2    2    0 ]   ← dot products of q1 with every key
q2      [ 1    1    2    0 ]
q3      [ 0    0    0    0 ]
```

FlashAttention NEVER builds this full matrix. Instead it computes 2×2 tiles one at a time in SRAM.

---

### Tile iteration map

```
Full S matrix divided into 2×2 tiles:

         k0,k1      k2,k3
       ┌──────────┬──────────┐
q0,q1  │  S_00    │  S_01    │   ← OUTER i=0
       │ (i=0,j=0)│ (i=0,j=1)│
       ├──────────┼──────────┤
q2,q3  │  S_10    │  S_11    │   ← OUTER i=1
       │ (i=1,j=0)│ (i=1,j=1)│
       └──────────┴──────────┘
```

Each tile is computed in SRAM and immediately used to update the running output. The tile is then discarded — never written to HBM.

---

### OUTER i=0: Load Q_0 = [q0, q1] into SRAM. Process all K,V tiles for these 2 queries.

**INNER j=0: Load K_0=[k0,k1], V_0=[v0,v1] into SRAM**

```
SRAM now holds: Q_0, K_0, V_0

S_00 = Q_0 @ K_0.T =                     ← top-left block of full S
  q0·k0  q0·k1       2   0
  q1·k0  q1·k1  =    0   2

This is scores for queries {q0,q1} vs keys {k0,k1} only.
We don't yet know scores against k2,k3 — so softmax can't be finalized.

Initialize running statistics per query row:
  Row q0:  m = max(2, 0) = 2
           l = exp(2-2) + exp(0-2) = 1 + 0.135 = 1.135
           O_q0 = exp(2-2)·v0 + exp(0-2)·v1
                = 1·[1,0] + 0.135·[0,1] = [1.000, 0.135]

  Row q1:  m = max(0, 2) = 2
           l = exp(0-2) + exp(2-2) = 0.135 + 1 = 1.135
           O_q1 = exp(0-2)·v0 + exp(2-2)·v1
                = 0.135·[1,0] + 1·[0,1] = [0.135, 1.000]
```

**INNER j=1: Evict K_0,V_0. Load K_1=[k2,k3], V_1=[v2,v3] into SRAM**

```
SRAM now holds: Q_0, K_1, V_1   (K_0, V_0 gone — never needed again)

S_01 = Q_0 @ K_1.T =                     ← top-right block of full S
  q0·k2  q0·k3       2   0
  q1·k2  q1·k3  =    2   0

New scores for queries {q0,q1} vs keys {k2,k3}.

Update running statistics — rescale old values to account for possibly larger max:

  Row q0:  m_new = max(m_old=2, max(2,0)=2) = 2    ← max unchanged
           rescale = exp(m_old - m_new) = exp(0) = 1
           l_new = 1·1.135 + exp(2-2) + exp(0-2) = 1.135 + 1 + 0.135 = 2.270
           O_q0  = 1·[1.000, 0.135]                ← rescaled old output
                 + exp(2-2)·v2 + exp(0-2)·v3       ← new tile contribution
                 = [1.000, 0.135] + 1·[1,1] + 0.135·[0,0]
                 = [2.000, 1.135]

  Row q1:  m_new = max(2, max(2,0)) = 2             ← max unchanged
           rescale = 1
           l_new = 1·1.135 + exp(2-2) + exp(0-2) = 2.270
           O_q1  = 1·[0.135, 1.000] + exp(2-2)·v2 + exp(0-2)·v3
                 = [0.135, 1.000] + [1, 1] + [0, 0]
                 = [1.135, 2.000]

No more K,V tiles. Normalize and write output rows q0,q1 to HBM:
  O[q0] = O_q0 / l = [2.000/2.270, 1.135/2.270] = [0.881, 0.500]
  O[q1] = O_q1 / l = [1.135/2.270, 2.000/2.270] = [0.500, 0.881]
```

Q_0 tile is done. **Evict Q_0 from SRAM. Two output rows written to HBM once.**

---

### OUTER i=1: Load Q_1 = [q2, q3] into SRAM. Same pattern.

```
INNER j=0: Load K_0, V_0  → compute S_10 (bottom-left block), init m/l/O for q2,q3
INNER j=1: Load K_1, V_1  → compute S_11 (bottom-right block), rescale + accumulate
           Normalize → write O[q2], O[q3] to HBM
```

---

### What was in SRAM at each step

```
Step          SRAM contents              HBM writes
──────────    ──────────────────────     ──────────────────
i=0, j=0      Q_0, K_0, V_0, m/l/O       nothing
i=0, j=1      Q_0, K_1, V_1, m/l/O       nothing
end of i=0    (empty)                     O[q0], O[q1]  ← written ONCE
i=1, j=0      Q_1, K_0, V_0, m/l/O       nothing
i=1, j=1      Q_1, K_1, V_1, m/l/O       nothing
end of i=1    (empty)                     O[q2], O[q3]  ← written ONCE
```

The full 4×4 S matrix (16 values) was computed tile by tile — at most 4 values lived in SRAM at any one time. For seq_len=4096 this would be 4096×4096 = 16M values that naive attention writes to HBM, vs FlashAttention which writes only the 4096×head_dim output once.

---

## 4. Online Softmax — What Makes Tiling Work

Normal softmax needs the full row to compute the denominator:
```
softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
```

You can't tile this naively — you don't know the full sum until you've seen all tiles.

**Online softmax** maintains running statistics across tiles:
```
For each new tile j:
    m_new = max(m_old, max(S_ij))         ← running row maximum (for numerical stability)
    l_new = exp(m_old - m_new) × l_old    ← rescale old sum
          + sum(exp(S_ij - m_new))        ← add new tile's contribution
    O_i   = (exp(m_old - m_new) × O_i    ← rescale old output
          + exp(S_ij - m_new) @ V_j)     ← add new tile's contribution

After all tiles: O_i = O_i / l_final     ← normalize once
```

The running max `m` and sum `l` are scalars per query row — they live in registers.
No NxN buffer needed anywhere.

---

## 4. Memory and IO Complexity

| | Naive Attention | FlashAttention |
|---|---|---|
| Memory for attention matrix | O(N²) | O(1) — tile in SRAM |
| HBM reads/writes | O(N²) | O(N²/M) — M = SRAM size |
| Passes over K, V | 1 | N/Bc passes — more HBM reads of K,V |
| Net HBM traffic | 4 × NxN writes/reads | ~(N/Bc) × tile loads of K,V |

FlashAttention trades more passes over K and V (re-reads K,V tiles from HBM each outer
loop iteration) for eliminating the NxN intermediate buffers entirely. At realistic
sequence lengths, eliminating the NxN buffer wins — especially as seq_len grows.

---

## 5. Kernel Launch Process (from our profiling run)

Dispatch chain observed in profiling_result.txt:

```
aten::scaled_dot_product_attention          Self CPU: 59ms   Self CUDA: 0
    ↓ checks available backends, selects efficient_attention for f32
aten::_scaled_dot_product_efficient_attention  Self CPU: 61ms   Self CUDA: 0
    ↓
aten::_efficient_attention_forward          Self CPU: 77ms   Self CUDA: 93ms  ← launches kernel
    ↓
fmha_cutlassF_f32_aligned_64x64_rf_sm80    Self CPU: 0      Self CUDA: 93ms  ← GPU execution
```

Same pattern as addmm — only the innermost op has non-zero Self CUDA.
Wrapper ops add CPU dispatch overhead (Self CUDA = 0) but no GPU time.

**Decoding the kernel name:**
```
fmha       = Flash Multi-Head Attention (tiled, SRAM-based)
cutlass    = built on NVIDIA CUTLASS (CUDA Templates for Linear Algebra Subroutines)
F          = Forward pass
f32        = float32 precision
aligned    = input tensors are memory-aligned → enables vectorized HBM loads
64x64      = tile size: 64 query tokens × 64 key tokens per tile
             fits in SRAM: 64 × 64 × 4 bytes × 2 (Q and K tiles) = 32 KB
rf         = accumulates partial output in register file (not shared memory) — fastest path
sm80       = compiled for Ampere architecture (A100 = sm_80)
```

---

## 6. Backend Comparison — PyTorch's scaled_dot_product_attention

PyTorch automatically selects among three backends at runtime:

| Backend | Kernel | Memory | dtype | Hardware |
|---|---|---|---|---|
| `flash_attention` | FlashAttention v2 (Tri Dao) | O(N) | FP16 / BF16 only | A100+ / H100 |
| `efficient_attention` | xFormers / CUTLASS FMHA | O(N) | any incl. FP32 | A100+ |
| `math` | Naive PyTorch (no tiling) | O(N²) | any | any GPU |

Selection logic (simplified):
```python
if flash_attention available and dtype in (fp16, bf16) and no padding mask:
    → flash_attention backend
elif efficient_attention available and hardware supports it:
    → efficient_attention backend   ← our run landed here (f32)
else:
    → math backend (naive, fallback)
```

**Why our run uses `efficient_attention` and not `flash_attention`:**

GPT-2 runs in float32 by default. FlashAttention v2 requires FP16 or BF16 to use
tensor cores efficiently — float32 triggers fallback to `efficient_attention`.

After Phase 5 quantization (FP16 weights + activations), the backend will switch:
```
Before (FP32):  fmha_cutlassF_f32_aligned_64x64_rf_sm80   ← efficient_attention
After  (FP16):  flash_fwd_kernel_... (FlashAttention v2)  ← flash_attention
```

FlashAttention v2 additionally parallelizes across the sequence length dimension
(not just batch/heads), giving better SM utilization at long sequences.

---

## 7. Why FlashAttention is Already Optimal in Our Run

```
fmha_cutlassF_f32_aligned_64x64_rf_sm80   93ms   11.39% CUDA
```

- Tiled computation — no NxN buffer in HBM
- Register file accumulation (rf) — output accumulated in registers, fastest path
- Memory-aligned loads — vectorized HBM reads
- CUTLASS-based — hand-tuned tile configs for sm80

There is nothing to optimize inside the attention kernel itself for FP32. The 93ms is
largely irreducible at batch=1 / seq_len=50 in float32.

Two levers to reduce it:
1. **Switch to FP16** — unlocks FlashAttention v2 backend, tensor core acceleration
2. **Increase batch size** — amortizes kernel launch overhead, same dispatch cost for more work

---

## 8. Relationship to KV Cache

FlashAttention is a **prefill optimization** — it processes the full prompt in one pass
with tiled computation. It does not change the decode phase.

During decode, each step generates one token. Attention is computed against the KV cache
(all previous tokens). At seq_len=50, the attention matrix is tiny — tiling provides
little benefit. The decode bottleneck is KV cache management (aten::cat, memory bandwidth)
not the attention computation itself.

Reference: `kv_cache.md`, `PagedAttention.md`
