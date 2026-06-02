# KV Cache in Transformer Architecture

## Why Attention Needs K and V

In the self-attention mechanism, for each token the model computes three vectors:
- **Q (Query)** — what this token is looking for
- **K (Key)**   — what this token offers as context
- **V (Value)** — the content this token contributes

Attention score for token `t` against all previous tokens `0..t-1`:

```
Attention(Q_t, K_0..t-1, V_0..t-1) = softmax(Q_t · K^T / sqrt(d_k)) · V
```

For a sequence of length `t`, token `t` must attend to all `t` previous tokens.
This requires K and V for every prior token to be available at compute time.

---

## The Problem Without KV Cache

In autoregressive generation (decode phase), tokens are generated one at a time.
Each new token `t` is fed through the full transformer to produce the next token `t+1`.

**Without KV cache — naive recomputation:**

```
Generating token 3:
  Forward pass inputs: [token_0, token_1, token_2, token_3]
  Recomputes K and V for tokens 0, 1, 2 from scratch
  Then computes attention for token_3 against all

Generating token 4:
  Forward pass inputs: [token_0, token_1, token_2, token_3, token_4]
  Recomputes K and V for tokens 0, 1, 2, 3 from scratch again
  ...
```

Total compute: O(seq_len²) — each new token recomputes all previous K/V vectors.
For a 1000-token sequence, token 999 recomputes 999 K/V pairs it already computed before.

---

## KV Cache — Store and Reuse

**Key insight:** K and V for token `t` depend only on token `t`'s embedding and the
layer weights — not on any future tokens. Once computed, they never change.

KV cache stores the K and V tensors for every token seen so far, per layer:

```
kv_cache[layer][token_index] = (K_vector, V_vector)
```

With KV cache — decode step for token `t`:

```
Step 1: Run forward pass for token t only (single token, not full sequence)
Step 2: Compute K_t, V_t for each layer
Step 3: Append K_t, V_t to kv_cache[layer]
Step 4: Compute attention using Q_t (new) against K_0..t, V_0..t (from cache)
Step 5: Output logits → sample next token
```

Compute per step: O(seq_len) for attention (dot product against all cached K/V),
but no recomputation of previous tokens. Total decode compute: O(seq_len²) → O(n · seq_len).

---

## KV Cache Shape and Memory

For GPT-2 medium (24 layers, 16 heads, 64 head_dim, FP16):

```
Per token per layer:
  K: [n_heads, head_dim] = [16, 64] = 1,024 float16 values = 2 KB
  V: [n_heads, head_dim] = [16, 64] = 1,024 float16 values = 2 KB
  Total per layer: 4 KB

All 24 layers:
  4 KB × 24 = 96 KB per token

For a 512-token sequence:
  96 KB × 512 = 49 MB per request

For 100 concurrent requests:
  49 MB × 100 = ~4.9 GB of KV cache in HBM
```

KV cache is the dominant HBM consumer during inference at scale —
larger than model weights for high-concurrency serving.

---

## Prefill vs Decode Phase

| Phase | Input | KV Cache | Compute pattern |
|---|---|---|---|
| Prefill | Full prompt (all tokens at once) | Built from scratch — all K/V computed in parallel | Compute-bound (GEMM, matrix × matrix) |
| Decode | One new token per step | Append one K/V per layer, attend against full cache | Memory-bound (GEMV, matrix × vector) |

Prefill is fast — parallelism across all prompt tokens (like training).
Decode is slow — sequential, one token at a time, bottlenecked by HBM bandwidth
(reading the full KV cache on every step).

This is the ridge point distinction from profiling: decode is bandwidth-bound,
not compute-bound, regardless of how fast the GPU's FP16 TFLOPS are.
