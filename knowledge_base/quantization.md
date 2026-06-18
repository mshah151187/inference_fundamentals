# Quantization

---

## 1. What is it?

Quantization maps floating-point numbers to a lower-precision representation.

A weight stored in BF16 occupies 2 bytes. The same weight in INT8 occupies 1 byte.
In INT4, it occupies half a byte (packed as two 4-bit values per byte).

The mapping is:

```
quantize:   float_val = round(float_val / scale) + zero_point   → stored as INT
dequantize: float_val = (int_val - zero_point) × scale          → used in compute
```

`scale` and `zero_point` are per-tensor, per-channel, or per-group constants computed
once at quantization time. They are stored alongside the weights.

**Three axes of quantization — from an optimization standpoint:**

| Axis | What | Primary goal |
|---|---|---|
| **Weights (W)** | Model parameter tensors (static, learned at training time) | Reduce VRAM + memory bandwidth reading weights each forward pass |
| **Activations (A)** | Tensors produced and consumed at runtime (per request) | Enable low-precision tensor core compute (W8A8, FP8) |
| **Data** | Input representations fed into the model | Reduce storage and transfer cost of the input itself |

The weight and activation axes are well-known in LLM serving.
The data axis is most prominent in recommendation systems.

**Data quantization — what it means:**

In a transformer-based LLM, the "data" entering the model is a sequence of token IDs →
looked up in a small embedding table → trivial size (50k vocab × 768 dim = 75 MB BF16 for GPT-2).

In a recommendation model (LinkedIn HSTU, Meta DLRM), the "data" is item/user embeddings
looked up from tables that scale with the entire catalogue:

```
Item embedding table:
  1 billion items × 256-dim × 2 bytes (BF16) = 512 GB

Transformer weights on top: ~1-2 GB
                                 ↑
                         the data IS the memory problem here
```

Quantizing the embedding table to INT8 → 256 GB. INT4 → 128 GB.
The transformer weights barely matter; the data does.

This is what the LinkedIn GR paper was doing when they discussed embedding quantization —
it is data compression, not model compression. The downstream HSTU layers still see BF16
(dequantized at lookup time). The quantization just reduces how much storage and
HBM bandwidth is consumed by the input representation.

**Scale granularity for data (per-row):**

Each item's embedding is a separate row with its own magnitude.
Per-row scale (one scale per item) is the natural fit — equivalent to per-channel for weight matrices.
A popular item with large embedding norm and a cold item with near-zero norm each get
their own scale so neither wastes INT8 precision on the other's range.

---

Notation: **W8A16** = 8-bit weights, 16-bit activations. **W4A16** = 4-bit weights.
**W8A8** = both weights and activations in 8-bit (fully low-precision compute).

---

### Worked example — same weight matrix, three granularities

Weight matrix W, shape [2, 4] (2 output channels, 4 input features):

```
W = [
  row 0:  0.1,  2.5, -1.2,  0.3
  row 1: -0.5,  0.1,  0.2, -3.0
]
```

Using **symmetric INT8** (zero_point = 0, range [-127, 127]).
Formula: `scale = max(|values|) / 127`

---

**Per-tensor — one scale for the entire matrix:**

```
global max |value| = 3.0   (the -3.0 in row 1)
scale = 3.0 / 127 = 0.02362

quantize each value: q = round(value / scale)

row 0:  0.1 → round( 0.1 / 0.02362) =   4
        2.5 → round( 2.5 / 0.02362) = 106   ← only 106 of 127 steps used
       -1.2 → round(-1.2 / 0.02362) = -51
        0.3 → round( 0.3 / 0.02362) =  13

row 1: -0.5 → -21
        0.1 →   4
        0.2 →   8
       -3.0 →-127

ONE scale = 0.02362 covers the entire matrix.
```

Row 0's largest value is 2.5, but the scale was set by row 1's 3.0.
Row 0 uses only steps up to 106 out of 127 — **17% of the INT8 range wasted**.
That wasted range = lower precision for row 0.

---

**Per-channel — one scale per output channel (per row):**

```
row 0 max |value| = 2.5  →  scale_0 = 2.5 / 127 = 0.01969
row 1 max |value| = 3.0  →  scale_1 = 3.0 / 127 = 0.02362

row 0 (uses scale_0):
  0.1 → round( 0.1 / 0.01969) =   5
  2.5 → round( 2.5 / 0.01969) = 127   ← hits the ceiling, full range used
 -1.2 → round(-1.2 / 0.01969) = -61
  0.3 → round( 0.3 / 0.01969) =  15

row 1 (uses scale_1):
 -0.5 → -21
  0.1 →   4
  0.2 →   8
 -3.0 →-127   ← also hits ceiling

TWO scales: [0.01969,  0.02362]
```

Each row now uses the full [-127, 127] range. More bins = finer resolution = lower error.

Dequantize row 0, value 2.5:  `127 × 0.01969 = 2.501`  error = 0.001
Dequantize row 0, value 2.5 under per-tensor: `106 × 0.02362 = 2.504`  error = 0.004

Small difference for INT8. For INT4 (only 16 steps) it becomes critical.

---

**Per-group — one scale per G consecutive weights (shown with INT4, G=2):**

INT4 symmetric range: [-7, 7] (15 steps).
`scale = max(|values in group|) / 7`

```
row 0 split into groups of 2:
  group [0.1,  2.5]: max = 2.5  → scale_00 = 2.5 / 7 = 0.357
  group [-1.2, 0.3]: max = 1.2  → scale_01 = 1.2 / 7 = 0.171

  0.1 → round( 0.1 / 0.357) =  0
  2.5 → round( 2.5 / 0.357) =  7   ← full range
 -1.2 → round(-1.2 / 0.171) = -7   ← full range
  0.3 → round( 0.3 / 0.171) =  2

row 1 split into groups of 2:
  group [-0.5, 0.1]: max = 0.5  → scale_10 = 0.5 / 7 = 0.071
  group [ 0.2,-3.0]: max = 3.0  → scale_11 = 3.0 / 7 = 0.429

 -0.5 → round(-0.5 / 0.071) = -7   ← full range
  0.1 → round( 0.1 / 0.071) =  1
  0.2 → round( 0.2 / 0.429) =  0
 -3.0 → round(-3.0 / 0.429) = -7   ← full range

FOUR scales: [0.357, 0.171, 0.071, 0.429]  (one per group of 2 weights)
```

Every group independently uses the full INT4 range. Each group's large outlier
gets 7 steps dedicated to it rather than being squeezed by another group's outlier.

---

**Why per-group is essential for INT4 but not INT8:**

INT8 has 256 steps. Wasting 17% from a bad scale still leaves ~210 effective steps.
INT4 has only 16 steps. Wasting 17% leaves ~13 steps — resolution degrades sharply.
Per-group (G=128 in practice) guarantees every 128-weight slice uses all 16 steps.

The cost: `(num_weights / G)` extra FP16 scale values stored alongside the INT4 weights.
For G=128: 1 FP16 per 128 weights = 1/128 overhead on top of 4 bits ≈ 4.125 effective bits/weight.

---

## 2. Data Types

Every number stored in a GPU lives in one of these formats. Understanding what
bits are used for what tells you exactly why each format exists and when to use it.

### Floating point basics

A floating point number has three fields:

```
FP32 (32 bits):
  [S] [EEEEEEEE] [MMMMMMMMMMMMMMMMMMMMMMM]
   1      8                23

value = (-1)^S  ×  2^(E - bias)  ×  (1 + M/2^23)
         sign       exponent          mantissa

Exponent bits → range  (how large/small the number can be)
Mantissa bits → precision  (how finely you can distinguish two nearby values)
```

More exponent bits = wider range. More mantissa bits = finer resolution.
This is the core tradeoff between all float formats.

---

### Format breakdown

```
Format   Bits   Sign   Exp   Mantissa   Max value       Smallest step near 1.0
───────  ─────  ────   ───   ────────   ─────────       ──────────────────────
FP32      32     1      8      23       ~3.4 × 10^38     ~1.2 × 10^-7
BF16      16     1      8       7       ~3.4 × 10^38     ~7.8 × 10^-3
FP16      16     1      5      10        65504            ~9.8 × 10^-4
FP8 E4M3   8     1      4       3          448            ~0.125
FP8 E5M2   8     1      5       2        57344            ~0.25
INT8       8    (sign bit embedded in value, no exp/mantissa)
INT4       4    (sign bit embedded in value, no exp/mantissa)
NF4        4    (not a hardware dtype — software lookup table, 16 hardcoded values)
```

---

### FP32 — the baseline

```
Range: ±3.4 × 10^38   Precision: ~7 decimal digits
```

Used for: parameter updates during training, loss scaling, optimizer states (Adam momentum).
NOT used for inference weights — too large (2× BF16 = 2× memory, no accuracy gain for serving).

---

### BF16 — the LLM workhorse

```
Same 8 exponent bits as FP32 → same range → no overflow when casting from FP32
7 mantissa bits instead of 23 → coarser precision (fewer significant digits)
```

Key insight: BF16 is FP32 with the bottom 16 mantissa bits truncated.

```
FP32: [S][EEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMM]
BF16: [S][EEEEEEEE][MMMMMMM]
                    ← just truncate here
```

This means FP32 → BF16 conversion is just a bit-shift — no special hardware.
No overflow risk (same exponent range as FP32).

Used for: LLM weight storage, activations, inference compute on A100/H100.
Hardware: BF16 tensor cores on A100 (312 TFLOPS), H100 (989 TFLOPS BF16).

---

### FP16 — older sibling of BF16

```
5 exponent bits → max value 65504  (much smaller range than FP32)
10 mantissa bits → finer precision than BF16
```

The problem: training gradients can exceed 65504 → overflow to inf → NaN.
Solution was loss scaling (multiply loss by 2^N before backward to keep gradients in range).
BF16 made this largely unnecessary.

FP16 is still used in some contexts:
- Older GPUs that predate BF16 tensor core support (pre-Ampere)
- CUDA kernels where FP16's extra mantissa bits improve accuracy in small computations
- Scale factors in quantized models are often stored as FP16

---

### FP8 — the H100 novelty

Two sub-formats, trading range for precision:

```
E4M3 (weights, forward activations):
  4 exponent bits → max ±448
  3 mantissa bits → finer resolution within that range
  safe for weight values which are small and centered

E5M2 (gradients, backward pass):
  5 exponent bits → max ±57344  (wider range needed for gradient magnitudes)
  2 mantissa bits → coarser within range
```

Why two variants? Weights and activations have small, well-behaved values → E4M3.
Gradients can spike to large values → need wider range → E5M2.

Hardware: H100 FP8 tensor cores at 1,979 TFLOPS — 2× faster than BF16.
NOT supported on A100. On A100, FP8 models fall back to BF16 compute.

---

### INT8 and INT4 — integers, not floats

Integers have no exponent or mantissa fields. They are just a fixed-width count.

```
INT8:  -128  to  127   (signed),   0 to 255 (unsigned)
INT4:    -8  to    7   (signed),   0 to  15 (unsigned)
```

There is no concept of "range" or "precision" built into the type.
You supply both externally via `scale` (and optionally `zero_point`):

```
value = (integer - zero_point) × scale
```

This is why integer quantization always needs a companion scale tensor.
The scale tensor is stored in FP16 — it IS a floating point number.

Advantages:
- Fast: INT8 matmuls run on all modern GPU tensor cores
- Compact: 1 byte (INT8) or 0.5 bytes packed (INT4) per value
- Deterministic: no floating point rounding in the stored representation

Disadvantages:
- No self-contained meaning — scale must travel with the data
- Dequantize before compute (for W8A16, W4A16) — adds overhead unless fused

---

### NF4 — not a hardware type

NF4 is NOT a GPU dtype. There is no NF4 tensor core.

It is a **quantization scheme** that happens to use 4 bits as indices:

```
16 hardcoded float values at quantile positions of a standard normal distribution:
  index 0  → -1.0000
  index 1  → -0.6962
  index 2  → -0.5251
  ...
  index 7  →  0.0000
  ...
  index 15 →  1.0000

Store a 4-bit index (0-15) per weight.
To dequantize: look up index in this table, multiply by the group scale.
```

Why quantile positions? Neural network weights are approximately normal.
Quantile-spaced bins = equal number of weights per bin = lowest average error.

At compute time: indices are looked up → BF16 values → fed into BF16 tensor cores.
NF4 is always W4A16 — the compute always happens in BF16.

---

### Summary: when to use what

```
Format     When to use it
─────────  ─────────────────────────────────────────────────────────────────
FP32       Training: optimizer states, master weights, loss
BF16       Inference: default weight and activation dtype for LLMs
FP16       Legacy inference; some scale factor storage
FP8 E4M3   H100 inference: weights + forward activations (W8A8 FP8)
FP8 E5M2   H100 training: backward pass gradients
INT8       Weight-only quantization (W8A16) or full W8A8 with SmoothQuant
INT4       Weight-only quantization (W4A16) via GPTQ/AWQ — needs per-group scale
NF4        bitsandbytes load_in_4bit — better accuracy than uniform INT4 for weights
```

```
Precision ranking (high → low):
  FP32 > FP16 > BF16 > FP8 E4M3 > FP8 E5M2 > INT8 > INT4 / NF4

Range ranking (wide → narrow):
  FP32 ≈ BF16 >> FP8 E5M2 > FP16 > FP8 E4M3 >> INT8* >> INT4*
  (* INT range is determined by scale, not the type itself)

Memory cost:
  FP32=4B  BF16=FP16=2B  FP8=1B  INT8=1B  INT4=0.5B(packed)  NF4=0.5B(packed)

Compute speed on H100 (tensor core TFLOPS):
  FP8: 1,979  >  BF16/FP16: 989  >  INT8: 1,979  >  INT4: not native
```

---

## 3. Why used?

### Primary reason: transformer matmuls are memory-bandwidth bound at small batch sizes

A single forward pass of GPT-2 (117M params in BF16) = 234 MB of weight reads.
For a decode step (batch=1, 1 token), compute = 234M multiply-adds.
Arithmetic intensity = 234M FLOPs / 234 MB = ~1 FLOP/byte.

A100's ratio: 312 TFLOPS (BF16) / 2 TB/s (HBM bandwidth) = 156 FLOPs/byte.

Your workload at 1 FLOP/byte is 156× below the compute roofline.
The GPU is spending almost all its time waiting for weights to arrive from HBM,
not actually computing. You are **memory-bandwidth bound**.

Quantization to INT8 halves the bytes read per weight → arithmetic intensity doubles.
Quantization to INT4 quarters bytes → arithmetic intensity 4×. The model shifts
rightward on the roofline toward the compute-bound region.

### Secondary reason: VRAM is the hard constraint

More VRAM freed by quantization → larger batch → more requests in flight → higher throughput.

A 70B model in BF16 = 140 GB. Does not fit on one A100 (40 GB).
In INT4 = 35 GB. Fits on one A100.

For GPT-2 (234 MB BF16), the weight savings are modest but the KV cache savings
(when quantizing KV cache separately) directly increase max concurrent requests.

---

## 4. What inference metric does it target?

| Metric | How quantization helps |
|---|---|
| **Throughput (tokens/sec)** | Primary target. Fewer bytes per weight → faster matmuls for memory-bound ops. Larger batch from freed VRAM. |
| **TTFT** | Indirect. Larger batch means more requests served per prefill step. Also faster attention if KV cache is quantized. |
| **ITL** | Improves when decode matmuls are memory-bandwidth bound (common at small batch). Less so when batch is large and GPU is already compute-bound. |
| **Max concurrent requests** | Direct. Every 2× reduction in weight dtype frees proportional VRAM → 2× more KV cache slots. |
| **Accuracy (perplexity)** | Degrades. This is the tradeoff. INT8 ≈ no visible loss. INT4 ≈ small loss. Lower than INT4 = noticeable. |

**The roofline shift (critical insight):**

```
                    compute roofline
                   /
          ITL    /
BF16 ●──────────/──────────────────  memory roofline
                                  \
INT8    ●───────────────────────────●  (shifted right, same compute ceiling)
INT4         ●──────────────────────●  (shifted further right)
                                    ↑
             at some batch size you cross from memory-bound to compute-bound
```

At large batch sizes, the model is already compute-bound — quantization still helps
memory but the throughput gain is smaller. The biggest gains are at small-to-medium
batch where memory bandwidth is the bottleneck.

---

## 5. Integer Quantization — Unpacking the Terminology

### The core intuition: mapping floats to integers step by step

Forget the word quantization for now. Here is the raw problem:

You have a row of floats. Each float takes 4 bytes (FP32) or 2 bytes (BF16).
You want to store them as integers — each integer takes 1 byte (INT8).
Goal: save space but recover values as accurately as possible later.

```
floats:  -3.0,  -1.2,   0.0,   1.5,   2.5,   3.0      (2 bytes each in BF16)
store as integers                                         (1 byte each in INT8)
recover approximate floats when needed
```

How would you do this?

---

**Step 1 — Look at what integers are available.**

INT8 can hold -127 to +127. That is **255 distinct values**.

```
INT8 number line:

  -127  -126  -125  ...  -1    0    1   ...  125   126   127
    |─────|─────|──── ... ─|────|────|─ ... ──|─────|─────|

255 slots. Each slot will represent one float value.
```

---

**Step 2 — Look at your float range.**

Your floats go from -3.0 to +3.0. That is a **span of 6.0**.

```
Float number line (continuous — infinite values between any two points):

  -3.0 ──────────────────── 0.0 ──────────────────── +3.0
  continuous, no gaps, infinite precision
```

---

**Step 3 — Divide the float range into 255 equal slices.**

You have 255 integer slots and a float span of 6.0.
To use every slot, divide the span equally:

```
width of each slice = float span / number of slots
                    = 6.0 / 254  (254 gaps between 255 boundary points)
                    = 0.02362 float units per slot
```

This width is the **bin width** — or **scale**. Every integer represents a
float window of exactly 0.02362 wide.

```
Each slot on the integer line owns a 0.02362-wide strip of the float line:

slot -127  owns floats from  -3.000  to  -2.976
slot -126  owns floats from  -2.976  to  -2.953
slot -125  owns floats from  -2.953  to  -2.929
...
slot    0  owns floats from  -0.012  to  +0.012
...
slot  126  owns floats from  +2.953  to  +2.976
slot  127  owns floats from  +2.976  to  +3.000
```

255 slots × 0.02362 = 6.0  ✓ — they tile the float range exactly.

---

**Step 4 — Map a float to its slot (quantize).**

Which slot does -1.2 fall into?

```
slot number = float value / bin width
            = -1.2 / 0.02362
            = -50.8

round to nearest whole slot → slot -51
```

Done. Store the integer **-51** instead of the float **-1.2**. 1 byte instead of 2.

---

**Step 5 — Recover the float from the slot (dequantize).**

Slot -51 owns the strip from -51 × 0.02362 to -50 × 0.02362:

```
-51 × 0.02362 = -1.2046
-50 × 0.02362 = -1.1810
```

The original -1.2 was somewhere in this strip. We stored only the slot number,
so we recover the slot's representative value (its center ≈ left edge):

```
recovered float = slot × bin width = -51 × 0.02362 = -1.2046

error = |-1.2 - (-1.2046)| = 0.0046
```

This error is unavoidable — you compressed 6.0 units of float range into 255 slots.
Each slot is 0.02362 wide, so the maximum possible error on any value is 0.02362 / 2 = 0.0118.

---

**Step 6 — Where does bin width = 0.02362 come from? The key formula.**

You want the largest float (3.0) to map to the largest slot (127).
Set them equal:

```
3.0 / bin_width = 127

bin_width = 3.0 / 127 = 0.02362
```

This single equation sets the bin width. It comes entirely from two things:
1. The largest float value (3.0) — determines how wide the range must be
2. The largest integer value (127) — how many slots you have on the positive side

This is why **max sets the scale**. If you used a smaller number than 3.0,
some floats would fall outside slot range and get clipped. If you used a larger
number, you'd be leaving slots unused (wasted precision).

---

**Clarification on the number line drawing:**

Earlier number lines showed tick marks at -3.0, -2.0, -1.0, 0.0, ... Those
are NOT bin boundaries. The actual bin boundaries are every 0.02362 apart —
255 of them between -3.0 and +3.0. Impossible to draw all 255, so the drawing
showed reference marks at every 1.0 float unit (= every 42 bins). The actual
bins are far finer than those tick marks suggest:

```
Between -1.0 and 0.0 alone, there are ~42 bins:
  -1.000  -0.976  -0.953  -0.929  -0.906  ...  -0.024   0.000
     |───────|───────|───────|───────|── ... ──|───────|
   slot    slot    slot    slot    slot       slot    slot
   -42     -41     -40     -39     -38         -1       0
   ← these 42 slots all live between the -1.0 and 0.0 tick marks →
```



### Asymmetric case (zero_point ≠ 0)

Now consider activations after a non-centered layer. Range [-0.5, +4.0].
Using uint8 [0, 255] — all non-negative integers, no sign bit.

The float line and int line no longer share the same zero:

```
Float:
  -0.5     0.0     1.0     2.0     3.0     4.0
    |-------|-------|-------|-------|-------|
    │       │                               │
    │       │  scale = 4.5/255 = 0.01765   │
    │       │  float 0.0 ≠ integer 0        │
    ▼       ▼                               ▼
    0      28      85     142     198      255
    |-------|-------|-------|-------|-------|
uint8:
```

Float -0.5 maps to integer 0 (the uint8 minimum).
Float +4.0 maps to integer 255 (the uint8 maximum).
Float 0.0 lands at integer **28** — not at 0.

`zero_point = 28` records this misalignment. It answers:
**"which integer represents the float value 0.0?"**

Quantize:   `q = round(x / 0.01765) + 28`
Dequantize: `x = (q - 28) × 0.01765`

Check: float 0.0 → `round(0.0 / 0.01765) + 28` = 0 + 28 = **28** ✓
Check: integer 28 → `(28 - 28) × 0.01765` = **0.0** ✓

Without zero_point, float 0.0 would map to integer 0, but integer 0 represents
float -0.5 in this range. Every zero would be decoded as -0.5 — a systematic bias
injected into every activation that was actually 0.

---

### Why zero_point matters: the bias argument

Imagine a sparse activation with many exact 0.0 values (common after ReLU or gating).

With zero_point = 0 on an asymmetric range:
- Float 0.0 stored as integer 0
- Dequantized: 0 × 0.01765 = 0.0  ← correct
- But integer 0 in this range SHOULD represent float -0.5
- So all your "zeros" decode as 0.0 but your -0.5 values also decode as 0.0
- You have collapsed two distinct float values onto one integer → bias

With zero_point = 28:
- Float -0.5 → integer 0 (correctly at the edge)
- Float  0.0 → integer 28 (correctly at the center-left)
- No collapsed values, no bias

For **weights**, zero_point is almost always 0 because weight distributions are
symmetric around zero — the float range IS centered, so symmetric quantization
wastes nothing.

For **activations**, zero_point is often non-zero because activation distributions
are skewed (ReLU is all positive, layer norms can shift the mean).

---

### Uniform vs Non-Uniform Quantization

So far all examples used **uniform** quantization: the integer steps are equally
spaced in float space. Step 0→1 covers the same float range as step 126→127.

```
Uniform INT4 bins (equally spaced between -1.0 and +1.0):
  -1.0  -0.87  -0.73  -0.60  ...  +0.73  +0.87  +1.0
    |-----|-----|-----|-----|       |-----|-----|
    each gap = 2.0/15 = 0.133 — same everywhere
```

**Problem:** neural network weights are NOT uniformly distributed. They follow an
approximately normal (bell-curve) distribution — most weights cluster near zero,
very few weights are near the extremes.

With uniform bins:
- 7 bins cover [-1.0, 0.0] where very few weights live
- 7 bins cover [0.0, +1.0] where most weights live
- Many bins wasted on the tails, crowded bins in the center → high error there

**Non-Uniform quantization** places bins where data is dense:

```
NF4 bins (placed at quantiles of a standard normal distribution):
  -1.0  -0.69  -0.52  -0.40  -0.28  -0.17  -0.09   0.0
    |------|------|------|------|------|------|------|
    gaps are SMALLER near zero (where most weights are)
    gaps are LARGER at the tails (where few weights are)

  0.0   0.09   0.17   0.28   0.40   0.52   0.69   1.0
    |------|------|------|------|------|------|------|
```

Each bin covers the same NUMBER of weights (by construction — quantiles).
More bins near zero = lower quantization error where weights actually live.

**NF4 (NormalFloat4)** used in bitsandbytes `bnb_4bit_quant_type="nf4"` works
exactly this way: 16 bin boundaries placed at the 1/16, 2/16, ..., 15/16 quantiles
of a standard normal distribution. These 16 values are hardcoded as a lookup table.

**Quantization with non-uniform bins** is a table lookup:
```
q = argmin_i |x - lookup_table[i]|   (find the nearest bin)
x ≈ lookup_table[q]                   (reconstruct from table)
```
No multiply/divide by scale — but requires an argmin search (or binary search)
at quantization time. Dequantization is just a table lookup: fast.

**Uniform vs Non-Uniform summary:**

| | Uniform (INT8/INT4) | Non-Uniform (NF4) |
|---|---|---|
| Bin spacing | Equal everywhere | Denser where data is dense |
| Quantize cost | divide + round | table lookup (argmin) |
| Dequantize cost | multiply | table lookup |
| Best when | data is uniform | data is known to be normal |
| Used in | INT8/GPTQ/AWQ | bitsandbytes NF4 |

---

### Why use max to set the scale?

**The hard constraint:** an integer type has a fixed range — INT8 is [-127, 127],
16 steps total. Every float value must land inside this range. A float that falls
outside gets **clipped** to the boundary — clamped to -127 or +127 — and that
error cannot be recovered. The value is permanently destroyed.

To guarantee zero clipping, scale must satisfy:

```
scale ≥ max(|values|) / q_max

→ using scale = max(|values|) / q_max gives:
  - full range used (every step gets assigned)
  - zero clipping error (no value falls outside)
  - finest possible bins for the given range
```

This is why max is the natural default: it is the minimum safe scale that
guarantees no clipping.

---

**But max is not always optimal — the clipping vs rounding tradeoff:**

Every quantization error comes from one of two sources:

```
Total error = rounding error + clipping error

rounding error:  float lands between two bins → rounded to the nearest one
                 → happens to every value, proportional to bin width (= scale)

clipping error:  float falls outside the integer range → clamped to boundary
                 → only happens to outliers, but error = |true_value - boundary|
```

When data has an outlier, using max forces ALL bins to be wide just to accommodate
that one extreme value. Everything else gets coarser bins. You pay rounding error
on EVERY value to avoid clipping ONE value.

**Example:** 6 values, one outlier.

```
values = [0.1, 0.2, 0.3, 0.2, 0.1, 8.0]
```

**Option A: scale = max / 7 = 8.0 / 7 = 1.143  (INT4, range -7 to 7, for simplicity)**

```
Float number line (full range -8.0 to +8.0):

  -8.0   -5.7   -3.4   -1.1    1.1    3.4    5.7    8.0
    |──────|──────|──────|──────|──────|──────|──────|
   -7     -5     -3     -1     +1     +3     +5     +7
  INT4:    each step = 1.143 float units   (very wide bins)

0.1 → round(0.1 / 1.143) = round(0.09) = 0,  dequant = 0.0,   error = 0.10
0.2 → round(0.2 / 1.143) = round(0.17) = 0,  dequant = 0.0,   error = 0.20
0.3 → round(0.3 / 1.143) = round(0.26) = 0,  dequant = 0.0,   error = 0.30
8.0 → round(8.0 / 1.143) = round(7.0)  = 7,  dequant = 8.0,   error = 0.00

All five small values collapse to integer 0.
The outlier 8.0 is perfect. Everyone else is terrible.
```

**Option B: clip 8.0, scale = 0.3 / 7 = 0.043  (calibrated to the majority)**

```
Float number line (clipped range -0.3 to +0.3):

  -0.3  -0.21  -0.13  -0.04   0.04   0.13   0.21   0.3
    |──────|──────|──────|──────|──────|──────|──────|
   -7     -5     -3     -1     +1     +3     +5     +7
  INT4:    each step = 0.043 float units   (narrow bins)

0.1 → round(0.1 / 0.043) = round(2.3) = 2,  dequant = 0.086,  error = 0.014
0.2 → round(0.2 / 0.043) = round(4.7) = 5,  dequant = 0.215,  error = 0.015
0.3 → round(0.3 / 0.043) = round(7.0) = 7,  dequant = 0.300,  error = 0.000
8.0 → clipped to 7,                          dequant = 0.300,  error = 7.70  ← big

Total error across all 6 values:
  Option A (max): 0.10 + 0.20 + 0.30 + 0.20 + 0.10 + 0.00 = 0.90
  Option B (clip): 0.014 + 0.015 + 0.000 + 0.015 + 0.014 + 7.70 = 7.76
```

Wait — in this example option A wins because the clipping error on 8.0 dominates.
That changes when there are MANY small values and VERY few outliers:

```
values = [0.1, 0.2, 0.3, ... × 1000 values ..., 8.0]  ← one outlier in 1000

Option A total error ≈ 1000 × 0.20 + 0 = 200   (rounding on everyone)
Option B total error ≈ 1000 × 0.014 + 7.7 = 21.7   (small rounding + one clip)
```

At scale, clipping one outlier and getting tight bins for the majority wins.
This is exactly what calibration-based quantization does.

---

**When max is the right choice:**

- Weight tensors: normally distributed (bell curve), no extreme outliers.
  Max is ~3σ from the mean. Using max = using 3σ = negligible clipping probability.
  Bins cover the distribution well. Max is near-optimal.

**When max is the wrong choice:**

- Activation tensors: after certain layers (attention softmax, LayerNorm output),
  some channels develop large outliers — values 10-100× larger than the median.
  LLM.int8() paper (Tim Dettmers, 2022) found that in 6.7B+ parameter models,
  ~0.1% of activation values are outliers but setting max to cover them degrades
  precision for the remaining 99.9%.

**What calibration does instead:**

GPTQ, AWQ, SmoothQuant run 128 representative samples through the model,
collect actual activation distributions per layer, then pick a clip threshold
(e.g. 99.9th percentile) that minimizes total quantization error — accepting a
tiny clipping error on 0.1% of values to gain much finer bins for 99.9%.

```
 using max (safe):         |──────────────────────────────●|
                                                           ↑ outlier forces wide bins

 using calibrated clip:    |──────────────────|     ●clipped
                                               ↑ tight bins for the majority
```

This tradeoff between rounding error and clipping error is the central engineering
problem of quantization. Max is the conservative starting point; calibration
finds the better balance.

---

### Per-tensor vs Per-channel: number line comparison

Same matrix. Same element: **row 0, 3rd element = -1.2**.

```
W = [
  row 0:  0.1,  2.5, -1.2*,  0.3     ← we track this element
  row 1: -0.5,  0.1,  0.2,  -3.0
]
```

---

**Case 1 — Per-tensor scale (one scale for the whole matrix):**

```
global max |value| = 3.0   (from row 1's -3.0)
scale_tensor = 3.0 / 127 = 0.02362
```

The full float number line maps to INT8:

```
Float:
  -3.0              -1.2       0.0               3.0
    |─────────────────●─────────|─────────────────|
    │                 │         │
    ▼                 ▼         ▼
  -127              -51         0               +127
    |─────────────────●─────────|─────────────────|
INT8:         scale = 0.02362 per step
```

Zoom in around -1.2 to see the bins:

```
         ← one INT8 step = 0.02362 float units →

Float:
  -1.2282    -1.2046   -1.1810    -1.1574
     |──────────|──────────|──────────|
                   ↑
                  -1.2 lands here (between -1.2282 and -1.2046)
                  round(-1.2 / 0.02362) = round(-50.8) = -51

INT8:
    -52         -51        -50        -49
     |──────────|──────────|──────────|

Quantized:    -51
Dequantized:  -51 × 0.02362 = -1.2046
Error:        |-1.2 - (-1.2046)| = 0.0046
```

---

**Case 2 — Per-channel scale (one scale per row):**

```
row 0 max |value| = 2.5   (the 2.5 in row 0 — NOT driven by row 1 anymore)
scale_row0 = 2.5 / 127 = 0.01969
```

Row 0's float range shrinks to [-2.5, +2.5]. The same INT8 range [-127, +127] now
covers a narrower float span → each integer step is a smaller float unit:

```
Float:
  -2.5              -1.2       0.0               2.5
    |─────────────────●─────────|─────────────────|
    │                 │         │
    ▼                 ▼         ▼
  -127              -61         0               +127
    |─────────────────●─────────|─────────────────|
INT8:         scale = 0.01969 per step   (narrower bins!)
```

Zoom in:

```
         ← one INT8 step = 0.01969 float units (narrower than 0.02362) →

Float:
  -1.2207    -1.2011   -1.1814    -1.1617
     |──────────|──────────|──────────|
                   ↑
                  -1.2 lands here (between -1.2207 and -1.2011)
                  round(-1.2 / 0.01969) = round(-60.9) = -61

INT8:
    -62         -61        -60        -59
     |──────────|──────────|──────────|

Quantized:    -61
Dequantized:  -61 × 0.01969 = -1.2011
Error:        |-1.2 - (-1.2011)| = 0.0011
```

---

**Side-by-side comparison:**

```
                      Per-tensor          Per-channel (row 0)
                      ──────────────────  ──────────────────────
Scale set by:         row 1's -3.0        row 0's 2.5
scale value:          0.02362             0.01969
Bin width (float):    0.0236              0.0197  ← 20% narrower
-1.2 maps to int:     -51                 -61
Dequantized:          -1.2046             -1.2011
Error:                0.0046              0.0011  ← 4× lower
```

**Why is the error 4× lower?** Row 0's largest value is 2.5, not 3.0.
Per-tensor stretches the INT8 ruler to reach 3.0 even though row 0 never needs
to go that far. That makes each step 20% coarser than it needs to be.
Per-channel gives row 0 its own ruler — calibrated to 2.5 — so each step is tighter
and -1.2 lands in a narrower bin closer to the true value.

**The general rule:** whenever a row/channel has a smaller max than the global max,
per-tensor wastes integer steps on a range that row never uses. Per-channel reclaims
those steps, giving that row finer resolution for the values it actually holds.

---

### Terminology map

| Term | Meaning |
|---|---|
| **Symmetric** | Float range centered at 0 → zero_point = 0 always |
| **Asymmetric** | Float range can be anywhere → zero_point shifts to align 0.0 |
| **Uniform** | Integer bins equally spaced in float space |
| **Non-Uniform** | Integer bins at custom positions (e.g. quantile-spaced) |
| **scale** | Float units per integer step (the stretch factor) |
| **zero_point** | Integer that represents float 0.0 (the alignment offset) |
| **Per-tensor** | One scale (and zero_point) for the whole tensor |
| **Per-channel** | One scale per output channel (row of weight matrix) |
| **Per-group** | One scale per G consecutive weights (G=128 typical for INT4) |

---

## 6. Granularity of scale factors

How `scale` (and optionally `zero_point`) is shared across values matters a lot
for accuracy vs overhead:

```
per-tensor:   one scale for the entire weight matrix    → cheapest, least accurate
per-channel:  one scale per output channel (row)        → good balance
per-group:    one scale per G consecutive weights       → expensive, most accurate
              (typical G = 128 for INT4)
```

INT8 typically uses per-channel scales (one per row of weight matrix).
INT4 typically uses per-group scales (one per 128 weights) — needed because
4-bit range is so narrow that per-channel alone loses too much accuracy.

---

## 7. Techniques

### 5.1 PTQ INT8 — Post-Training Quantization, 8-bit weights

**Type:** W8A16 (weights INT8, compute in FP16/BF16)
**How:** after training, round weights to INT8 with per-channel scale. No retraining.
**Runtime:** dequantize weight slice to BF16 just before each matmul.
**Tool:** bitsandbytes `load_in_8bit=True`

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained("gpt2", quantization_config=bnb_config)
```

**What happens inside bitsandbytes:**
- Linear layer weights stored as INT8
- `bnb.nn.Linear8bitLt.forward()` calls a custom CUDA kernel that dequantizes
  the weight block on-the-fly into a BF16 register tile, then does the matmul
- Activations remain BF16 throughout

**When to use:** quick, no calibration data needed, minimal accuracy loss.
**Limit:** compute still in BF16. Savings = memory only (not tensor core speedup).

---

### 5.2 PTQ INT4 — Post-Training Quantization, 4-bit weights

**Type:** W4A16
**How:** same as INT8 but quantize to 4 bits with per-group scales (G=128 typical).
**Tool:** bitsandbytes `load_in_4bit=True` with NF4 (NormalFloat4) dtype

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # NF4 dtype — designed for normally-distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # quantize the scale factors too (saves ~0.4 bits/param)
)
model = AutoModelForCausalLM.from_pretrained("gpt2", quantization_config=bnb_config)
```

**NF4 (NormalFloat4):** weights in transformer models are approximately normally
distributed. NF4 places 16 quantization bins at the 1/16, 2/16, ... quantiles of
a standard normal distribution — so bins are densest where weights are densest.
This gives better accuracy than uniform INT4 at the same bit width.

**Double quantization:** the per-group scale factors (32-bit floats, one per 128
weights) are themselves quantized to 8-bit. Saves an extra ~0.37 bits per param.

**Limit:** 4-bit range is very narrow. Per-group scales are essential to recover accuracy.

---

### 5.3 GPTQ — Gradient-based Post-Training Quantization

**Type:** W4A16 (or W3A16)
**Key idea:** use second-order information (Hessian) to find the INT4 weights that
minimize the output error of each layer — not just round to nearest INT4.

**How it works:**
1. Run calibration data (128 samples typical) through the model, capture activations.
2. For each linear layer, compute the Hessian H = E[X^T X] where X is the layer input.
3. Quantize weights column by column. After quantizing column i, update remaining
   unquantized columns to compensate for the error introduced (Cholesky of H^{-1}).
4. Result: INT4 weights that minimize the squared output error for this specific layer,
   given the calibration distribution.

**Runtime:** same as INT4 PTQ — dequantize block to BF16 before matmul.

**vs naive INT4:** GPTQ weights have noticeably lower perplexity because quantization
error is redistributed (not just truncated). The Hessian tells you which weights matter
most — errors in high-Hessian weights hurt output more, so the algorithm compensates there.

**Tool:** `auto-gptq` library, huggingface supports GPTQ checkpoints natively.

**Cost:** calibration + optimization takes ~30 min on 7B model. One-time offline step.

---

### 5.4 AWQ — Activation-Aware Weight Quantization

**Type:** W4A16
**Key idea:** not all weights are equally important. Weights that get multiplied by
large activations cause large quantization errors. Scale those weights BEFORE
quantization so they land in a better INT4 bin.

**How it works:**
1. Run calibration data, find per-channel activation magnitudes (the `s` vector).
2. Identify salient channels — those with the largest activation values.
3. Scale salient weight columns UP by `s` (making them larger → more INT4 bins
   used → lower quantization error for the important channels).
4. Scale the corresponding input activation DOWN by `1/s` (to cancel out the scaling —
   the math output is unchanged: `(s × W) × (X / s) = W × X`).
5. Quantize the scaled weights to INT4.

**Runtime:** during inference, the `1/s` scaling of activations is fused into the
preceding operation. Quantized weights are dequantized per-group before matmul.

**vs GPTQ:**
- AWQ is faster to apply (no Hessian computation)
- AWQ tends to perform better on zero-shot tasks
- GPTQ can push to lower bit widths (W3, W2) more gracefully
- Both achieve similar perplexity at W4A16

**Tool:** `autoawq` library. vLLM natively supports AWQ checkpoints.

```python
# vLLM loading AWQ model
from vllm import LLM
llm = LLM(model="TheBloke/Mistral-7B-AWQ", quantization="awq")
```

---

### 5.5 FP8 W8A8 — Float8, Both Weights and Activations

**Type:** W8A8 (both quantized, compute in FP8 tensor cores)
**Key idea:** unlike INT8 (which dequantizes back to BF16 before compute), FP8
does the entire matmul in FP8 on H100/A100 tensor cores.

**FP8 formats:**
- `E4M3` — 4 exponent bits, 3 mantissa bits, range ±448. Used for weights (narrow range OK).
- `E5M2` — 5 exponent bits, 2 mantissa bits, range ±57344. Used for gradients/activations (wide range needed).

**Why FP8 matters for throughput:**
H100 tensor core throughput: 989 TFLOPS (FP8) vs 494 TFLOPS (BF16/FP16).
FP8 is exactly 2× faster on H100 compute. PLUS half the bytes moved from HBM.
Combined effect: both the compute ceiling AND the memory bandwidth benefit.

**Challenge: activation ranges vary per token**
Weights are static → calibrate scale once.
Activations vary at runtime → need dynamic per-token or per-tensor scaling.
Two approaches:
- **Static scaling:** run calibration data, fix scale per-tensor. Fast, but clips
  outlier activations that exceed the calibrated range.
- **Dynamic scaling (per-token):** compute abs-max of each token's activation vector
  at runtime, scale it to FP8 range. Accurate but adds a small reduction kernel.

**vLLM FP8 implementation:**
`vllm/model_executor/layers/quantization/fp8.py` — `Fp8LinearMethod` class.
Stores weight as FP8, weight scale as FP32. Forward: call
`torch._scaled_mm(input_fp8, weight_fp8, scale_a, scale_b, out_dtype=bfloat16)` —
this is the H100 native FP8 GEMM.

**Accuracy:** FP8 E4M3 has 8× finer resolution than INT8 in relative terms (floating
point vs integer). Accuracy loss is typically negligible vs BF16.

---

### 5.6 SmoothQuant — Smooth the Quantization Difficulty

**Type:** W8A8 (enables full INT8 compute)
**Problem:** activations in transformers have outliers — a few channels are 100×
larger than the median. Quantizing these to INT8 wastes most bins on the outliers
and loses precision everywhere else.

**Solution:** migrate the quantization difficulty from activations to weights.
For each channel c, compute a per-channel smooth factor `s_c = max(|X_c|)^α / max(|W_c|)^(1-α)`.

Then:
- Scale input X column c DOWN: `X_smoothed[:, c] = X[:, c] / s_c`
- Scale weight W row c UP:     `W_smoothed[c, :] = W[c, :] × s_c`

The product is unchanged: `X_smoothed × W_smoothed = X × W`.
But now X_smoothed has no outliers (divided them away) and W_smoothed's outliers
are mild (weights are well-distributed in transformers).

Both X_smoothed and W_smoothed can now be cleanly quantized to INT8.
The `1/s_c` division is fused into the LayerNorm/RMSNorm before each linear.

**α parameter:** controls how much difficulty shifts to weights. α=0.5 is typical.
Higher α = more difficulty on weights = easier activations.

**vs AWQ:** AWQ is weight-only (W4A16), handles activations differently.
SmoothQuant enables W8A8 — both sides in INT8 → full INT8 tensor core compute.

---

### 5.7 KV Cache Quantization — INT8/FP8 KV

**Type:** quantize K,V tensors stored in the KV cache (not model weights)
**Metric target:** VRAM → max batch size and max sequence length

**Why the KV cache is large:**
GPT-2 (12 layers, 12 heads, 64 head_dim, BF16):
Per token per layer: 2 × 12 × 64 × 2 bytes = 3072 bytes
At seq_len=1024: 3072 × 1024 × 12 = 37.7 MB per request

At 256 concurrent requests: 37.7 × 256 = 9.7 GB — just for KV cache.

INT8 KV cache: halves that to 4.8 GB. FP8: also halves it.

**Challenge:** K and V have different distributions than weights.
- Keys have high variance across heads (some heads have very large K norms).
- Values have different statistics from Keys.
- Per-token dynamic scaling is common: at write time, compute abs-max for the token's
  K vector, store (K_quant, k_scale). At read time, dequantize.

**vLLM FP8 KV cache:**
`vllm/model_executor/layers/quantization/kv_cache.py`
`BaseKVCacheMethod.get_kv_cache_dtype()` → returns `"fp8_e5m2"` or `"int8"`.
Scale factors stored alongside KV blocks. Flash attention kernels accept a
`kv_scale` argument — they dequantize inside the attention CUDA kernel,
so no separate dequant kernel is needed.

```python
# Enable FP8 KV cache in vLLM
llm = LLM(model="meta-llama/Llama-3-8B", kv_cache_dtype="fp8_e5m2")
```

---

### 5.8 NVFP4 / MXFP4 — 4-bit Float (Blackwell/H100)

**Type:** W4A4 (Blackwell-native) or W4A8
**Format:** 4-bit float with per-group FP8 scale factor (one FP8 scale per 16 weights).
Group size 16 → scale overhead = 1 FP8 byte / 16 weights = 0.5 bits/weight overhead.
Net: 4.5 effective bits per weight.

**MX (Microscaling) format:** NVIDIA/Microsoft standard. The FP8 group scale is
fused into the tensor core block — dequantization happens inside the hardware,
not as a separate kernel.

**For MoE:** token dispatch volume to experts is proportional to weight dtype.
NVFP4 vs BF16 = 4× less data to route to each expert → 4× less AllToAll traffic.

**Hardware requirement:** Blackwell (B200) for native W4A4. H100/A100 can do W4A16
with software dequantization.

---

### 5.9 GGUF (llama.cpp / Ollama)

**Type:** mixed per-layer bit depth
**Key idea:** not all layers need the same precision. Attention layers and early/late
layers are often more sensitive — keep them at higher precision. MLP middle layers
can often be compressed more aggressively.

**GGUF quantization types:**
- `Q4_K_M`: 4-bit with k-quant block float. Most weights 4-bit, some 6-bit for sensitive layers.
- `Q8_0`: 8-bit uniform. Near-lossless.
- `Q4_0`: 4-bit uniform. More aggressive, some quality loss.

**K-quant block float:** groups of 256 weights share one scale (FP16) and one min
value. Within the group, each weight is stored as its distance from `min` in 4 bits.
Different from per-group scales: block float uses a min-value anchor, not zero-point.

**Partial GPU offload:** GGUF allows loading some layers on GPU, rest on CPU.
`n_gpu_layers=32` means first 32 transformer blocks on GPU, rest on CPU.
Useful when model is too large for VRAM but you want GPU acceleration for the hot path.

**Not relevant for A100 serving** — GGUF is designed for consumer hardware (Apple Silicon,
RTX 3090). On A100 you'd use vLLM with AWQ/GPTQ/FP8 instead.

---

## 8. How Weight Quantization Works End-to-End

### Libraries

| Library | Role | Methods supported |
|---|---|---|
| **bitsandbytes** | On-the-fly PTQ at load time — no offline step needed | INT8 (LLM.int8()), INT4 NF4 |
| **auto-gptq** | Offline GPTQ calibration, saves pre-quantized checkpoint | GPTQ W4A16, W3A16 |
| **autoawq** | Offline AWQ calibration, saves pre-quantized checkpoint | AWQ W4A16 |
| **vLLM (built-in)** | Loads pre-quantized checkpoints, provides fused kernels | FP8, AWQ, GPTQ, bitsandbytes |

bitsandbytes and vLLM are the serving libraries. auto-gptq and autoawq are the
offline tools that produce the checkpoints vLLM then loads.

---

### The two moments: quantization vs dequantization

These happen at completely different times:

```
Quantization:    happens ONCE — at load time (bitsandbytes) or offline before deployment (GPTQ/AWQ)
                 input: BF16 weights
                 output: INT8/INT4 weights + scale factors stored in HBM

Dequantization:  happens at EVERY forward pass, per tile, inside the kernel
                 input: INT8/INT4 tile from HBM + scale
                 output: BF16 tile in registers → fed immediately into tensor cores
                 the BF16 tile never touches HBM
```

---

### bitsandbytes INT8 (W8A16) — step by step

**At load time:**

```python
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)
```

```
Step 1 — load BF16 weights from disk to CPU RAM:
  q_proj.weight  [768, 768]  bfloat16  = 1.1 MB on CPU

Step 2 — replace_with_bnb_linear() walks the model tree:
  every nn.Linear  →  bnb.nn.Linear8bitLt
  (the class itself changes — forward() now points to the INT8 kernel)

Step 3 — quantize weights inside Linear8bitLt.__init__():
  for each output row (per-channel):
    scale  = max(|row|) / 127          ← one FP16 scalar per row
    q_row  = round(row / scale)        ← INT8 vector, one byte per weight
    clamp  q_row to [-127, 127]

Step 4 — store and free:
  .weight        INT8   [768, 768]  = 590 KB
  .weight_scale  FP16   [768]       =   1.5 KB   (one scale per output row)
  original BF16 freed from CPU RAM

Step 5 — move to GPU HBM:
  590 KB + 1.5 KB land in HBM (vs 1.1 MB for BF16)
  BF16 weights never exist in HBM at any point
```

**At forward pass (every request):**

```
x arrives: [batch, seq, 768]  bfloat16  (from previous layer, already in HBM)

bnb.nn.Linear8bitLt.forward(x)
  → dispatches to bnb.matmul() → custom CUDA kernel

kernel (per tile, e.g. 128 output rows at a time):

  ① load INT8 tile from HBM:    [128, 768] × 1 byte  =  98 KB  ← HBM read
  ② load scales for this tile:  [128]      × 2 bytes =  0.25 KB ← HBM read
  ③ dequantize in registers:
       w_bf16 = int8_tile.to(bfloat16) * scales[:, None]
       w_bf16 lives ONLY in registers — never written to HBM
  ④ matmul in tensor cores:
       y_tile = x_bf16 @ w_bf16.T        ← BF16 compute
  ⑤ accumulate y_tile into output buffer in HBM
  ⑥ advance to next tile — w_bf16 discarded from registers

return y: [batch, seq, 768]  bfloat16
```

---

### bitsandbytes INT4 NF4 (W4A16) — differences from INT8

```
Group size:  G=64 (default in bnb, not per-channel like INT8)
             each group of 64 consecutive weights shares one scale

Quantization dtype:  NF4 (not round(x/scale))
  Step 1: normalize group:  x_norm = x / scale   (scale = abs-max of group)
  Step 2: table lookup:     q = argmin_i |x_norm - NF4_TABLE[i]|
          NF4_TABLE = 16 values at quantile positions of standard normal:
          [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
            0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7229, 1.0]
  Result: q is 4-bit index (0-15), 2 weights packed per byte

Double quantization (optional, bnb_4bit_use_double_quant=True):
  The per-group FP32 scales are themselves quantized to INT8
  One more scale (FP32) per 256 scale values
  Saves an extra ~0.37 bits per weight

Storage (GPT-2 q_proj [768, 768]):
  qweight:  [768, 768] × 0.5 bytes  = 295 KB   (packed INT4)
  scales:   [768, 12]  × 2 bytes    =  18 KB    (one FP16 per 64 weights)
  Total:                              313 KB  (vs 1.1 MB BF16 = ~3.5× compression)

Dequantization in kernel:
  unpack INT4 pair → index into NF4_TABLE → multiply by group scale
  table lookup replaces the integer multiply
```

---

### AWQ / GPTQ — offline pre-quantized checkpoints

These are quantized once by the model publisher, NOT at your load time.

```
OFFLINE (done once before serving):

  GPTQ:
  ① run 128 calibration samples through the model, capture activations X per layer
  ② compute Hessian H = E[X^T X]  — tells you which weights matter most
  ③ quantize column by column:
       after quantizing column i, update remaining columns to compensate
       for the error introduced (uses Cholesky of H^{-1})
  ④ result: INT4 weights that minimize layer output error on the calibration set

  AWQ:
  ① run calibration, compute per-channel activation magnitudes s_c
  ② scale salient weight columns UP by s_c:
       W_scaled[:, c] = W[:, c] × s_c
  ③ scale input activations DOWN by 1/s_c  (absorbed into preceding LayerNorm)
       — math output unchanged: (s_c × W) × (X / s_c) = W × X
  ④ quantize the scaled weights to INT4 per-group

  Both save to disk:
    qweight   [out, in/8]  int32    (INT4 packed, 8 weights per int32)
    scales    [out, in/G]  float16  (one per group of 128 weights)
    qzeros    [out, in/G]  int32    (zero_points packed)

AT LOAD TIME (your serving machine):

  config.json already contains:
    {"quantization_config": {"quant_type": "awq", "bits": 4, "group_size": 128}}

  from_pretrained reads this → instantiates AwqConfig
  replaces nn.Linear with awq.QuantLinear (or vLLM's AwqLinearMethod)
  loads qweight/scales/qzeros directly to HBM — NO quantization step needed

AT FORWARD PASS:

  awq_gemm_cuda(x_bf16, qweight_int4, scales_fp16, qzeros_int4, group_size=128)
  → single fused kernel: unpack INT4 + dequantize + GEMM in one launch
  → output: y_bf16
```

---

### Fused vs unfused kernel — why it matters

```
bitsandbytes INT8 (two separate steps):

  HBM(INT8)  →  dequant kernel  →  HBM(BF16)  →  GEMM kernel  →  HBM(output)
                                        ↑
                              intermediate BF16 write to HBM
                              (costs HBM bandwidth — the thing we're trying to save)

AWQ/GPTQ fused kernel (one step):

  HBM(INT4)  →  dequant+GEMM kernel  →  HBM(output)
                      ↑
              dequant happens in registers during the GEMM tile loop
              BF16 intermediate never reaches HBM
              one kernel launch, one pass over the weights
```

fused kernels are one reason AWQ/GPTQ are preferred for production serving
over bitsandbytes, even when both achieve the same INT4 compression ratio.

---

### Memory at each stage (GPT-2 q_proj [768, 768])

```
Stage                    Location    Format           Size
──────────────────────── ─────────── ──────────────── ────────
Saved checkpoint (BF16)  disk        bfloat16         1.1 MB
load_in_8bit             HBM         INT8 + FP16 sc   592 KB   (2× reduction)
load_in_4bit NF4         HBM         INT4 + FP16 sc   313 KB   (3.5× reduction)
AWQ/GPTQ INT4            HBM         INT4 + FP16 sc   ~300 KB  (similar to NF4)
During forward (W4A16):
  active tile             registers   BF16             ~50 KB   (ephemeral, per tile)
  output                  HBM         BF16             batch × seq × 768
After forward            HBM         INT4 + scales    unchanged — weights stay compressed
```

---

## 9. How Activation Quantization Works End-to-End

**First: why are activations quantized at all?**

This is different from weight quantization. Weights are quantized to reduce model
footprint — fewer bytes in HBM, less bandwidth consumed reading them each forward pass.
Activations are NOT stored persistently. They are computed, used immediately, and
discarded. There is nothing to compress for storage.

Activations are quantized for exactly one reason: **to unlock faster tensor cores.**

```
INT8 and FP8 tensor cores are separate hardware units on the SM.
They only fire when BOTH operands are in low precision simultaneously.

A linear layer computes:   Y = X · W^T + b

  X  — input data (token embeddings or output of previous layer)
       shape [batch × seq, d_model]                  ← operand 1
       this IS the activation — the name "activation" just means
       "whatever is flowing through the network at this point"

  W  — weight matrix, shape [d_out, d_model]         ← operand 2
       static, lives in HBM, loaded at startup

  b  — bias vector, shape [d_out], stays in BF16

  Y  — output, shape [batch × seq, d_out]
       becomes the input X of the next layer

The input data flows through the network as X.
At every linear layer X is the "activation" — it is the data at that point in the graph.
First layer: X = token embeddings (your actual input).
Later layers: X = output of the previous layer's matmul.

Both X and W must be INT8 for the INT8 tensor core to execute this matmul.
Quantizing only W (W8A16) leaves X in BF16 → the INT8 unit cannot be used.
Quantizing both X and W (W8A8) → INT8 unit fires → 2× faster.

W8A16 (only weights quantized):
  weight:     INT8  ──► dequantize to BF16 ──► BF16 matmul on BF16 tensor cores
  activation: BF16
  benefit:    2× less HBM bandwidth reading weights
              compute still runs at BF16 speed

W8A8 (weights AND activations quantized):
  weight:     INT8
  activation: INT8  ──► INT8 @ INT8 on INT8 tensor cores
  benefit:    2× less HBM bandwidth  PLUS  2× faster compute

A100 tensor core throughput:
  BF16 tensor cores: 312 TFLOPS
  INT8 tensor cores: 624 TFLOPS   ← 2× faster, same silicon, different units

The activation quantization step costs almost nothing:
  one abs-max reduction + one multiply + round = microseconds per token per layer
  this unlocks 2× compute speedup on every matmul in the network
```

The second runtime tensor that gets quantized is the KV cache — but for a completely
different reason. KV cache IS stored persistently in HBM between decode steps, so
quantizing it reduces its memory footprint directly.

```
Linear layer activations:  quantized to unlock INT8/FP8 tensor cores (faster compute)
                           NOT for storage — discarded immediately after use

KV cache:                  quantized to reduce HBM footprint (storage compression)
                           stored in HBM, grows with sequence length, survives across steps
```

---

### 9.1 How Linear Layer Activation Quantization Works

**Why activations cannot be quantized offline like weights:**

```
Weights:     fixed after training. abs-max is the same for every inference call.
             Scale computed once and stored permanently.

Activations: depend on the input. The activation range after layer 5 for
             prompt "Hello" is completely different from prompt "Solve this integral".
             You cannot know the scale until the input arrives.
```

**Two approaches to get the scale at runtime:**

**Static scaling (SmoothQuant / calibration-based):**

```
Offline (done once):
  Run 128 representative inputs through the model.
  For each linear layer, collect activation distributions.
  Compute a scale that covers the 99.9th percentile of observed values:
    scale_a = percentile_99_9(abs(X)) / 127
  Store scale_a as a FP16 parameter alongside the quantized weight.

At inference (every forward pass):
  x_int8  = round(x_bf16 / scale_a).clamp(-127, 127)   ← one multiply + round
  y_int32 = x_int8 @ w_int8.T                           ← INT8 tensor cores
  y_bf16  = y_int32.float() * (scale_a * scale_w)       ← dequantize output
```

Scale is fixed — cheap to apply, but can clip unexpected outliers in new inputs.

**Dynamic scaling (per-token, vLLM FP8):**

```
At inference, for each token's activation vector independently:
  token_scale = abs_max(x_bf16[token, :]) / 448       ← small reduction per token
  x_fp8       = round(x_bf16[token, :] / token_scale) ← quantize to FP8

  y = torch._scaled_mm(x_fp8, w_fp8,
                        scale_a=token_scale,
                        scale_b=w_scale,
                        out_dtype=bfloat16)            ← H100 FP8 tensor cores
```

Scale adapts to each token's actual range. More accurate than static, but costs a small
abs-max reduction kernel per layer per token.

**Where this happens — inside the layer's forward():**

The quantization step is NOT a preprocessing step you write in the pipeline.
It lives inside the custom linear class that replaced nn.Linear at load time:

```python
class Fp8LinearMethod:
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # x arrives in BF16 from the previous layer

        # Step 1: quantize activation to FP8 dynamically
        x_scale = x.abs().max() / 448               # per-tensor scale
        x_fp8   = (x / x_scale).to(torch.float8_e4m3fn)

        # Step 2: FP8 matmul on H100 tensor cores
        # self.weight (FP8) and self.weight_scale (FP16) loaded at startup
        y = torch._scaled_mm(
            x_fp8, self.weight,
            scale_a=x_scale,
            scale_b=self.weight_scale,
            out_dtype=torch.bfloat16,
        )
        return y   # BF16 — next layer sees no difference
```

The previous layer outputs BF16. The next layer receives BF16. The quantization
and dequantization are invisible to everything outside this single class.

**Static vs dynamic scaling — tradeoff:**

```
                Static (SmoothQuant)        Dynamic (FP8 per-token)
                ─────────────────────       ───────────────────────
Scale source:   calibration data (offline)  abs-max at runtime
Scale varies:   never (fixed per layer)     every token
Clip risk:      yes (outliers outside range) no (scale adapts)
Runtime cost:   negligible (one multiply)   small reduction per token per layer
Calibration:    128 samples needed          none needed
Used in:        SmoothQuant, static FP8     vLLM dynamic FP8
```

**The full sequence through one transformer layer (W8A8 INT8):**

```
x_bf16  [batch, seq, d_model]   ← arrives from previous layer

  ① Q projection (QuantLinear):
       x_int8  = round(x_bf16 / scale_a).clamp(-127, 127)
       Wq_int8 = loaded at startup (pre-quantized weight)
       Q_int32 = x_int8 @ Wq_int8.T        ← INT8 tensor cores
       Q_bf16  = Q_int32 * (scale_a * scale_wq)

  ② K projection (QuantLinear):   same pattern → K_bf16
  ③ V projection (QuantLinear):   same pattern → V_bf16

  ④ Attention:                     Q_bf16 @ K_bf16.T (BF16 — attention stays in BF16)

  ⑤ O projection (QuantLinear):
       attn_out_int8 = round(attn_out_bf16 / scale_a_out).clamp(-127, 127)
       y_int32       = attn_out_int8 @ Wo_int8.T
       y_bf16        = y_int32 * (scale_a_out * scale_wo)

  ⑥ MLP (same pattern: two QuantLinear layers)

x_bf16  [batch, seq, d_model]   → passed to next layer
```

Attention itself stays in BF16 — the softmax and score normalization need float precision.
Only the projection matmuls (where arithmetic intensity is low and weight reads dominate) run in INT8.

---

### 9.2 How Attention Layer KV Quantization Works

**Why KV cache is a different problem from activations:**

```
Regular activations:  produced and consumed in the SAME forward pass
                      no need to store them between steps

KV cache:             produced at prefill (or each decode step)
                      stored in HBM
                      read on EVERY future decode step for this request
                      grows by one row per token per step per layer

At seq_len=1024, GPT-2, BF16:
  Per request KV = 12 layers × 2 × 12 heads × 64 head_dim × 2 bytes = 37.7 MB
  256 concurrent requests = 9.7 GB just for KV cache

INT8 KV cache: halves this to 4.8 GB → room for ~512 concurrent requests
```

**Quantization at write time (prefill or each decode step):**

```
After K and V projections produce K_bf16, V_bf16:

  For each token t, each attention head h:
    k_scale[t,h] = abs_max(K_bf16[t, h, :]) / 127   ← per-token-per-head scale
    K_int8[t,h,:]= round(K_bf16[t,h,:] / k_scale[t,h]).clamp(-127, 127)

  Write to HBM KV cache block:
    block[layer][token] = {K_int8: 64 bytes, k_scale: 2 bytes}
                                                 vs {K_bf16: 128 bytes}
    → 2× less HBM written per token per head
```

Scale granularity is per-token-per-head because:

```
Attention heads specialize during training. Head 3 might track positional patterns
(small K values) while head 9 tracks semantic similarity (large K values).
Per-head scale lets each head use its full INT8 range independently.
Per-tensor scale would be dominated by the highest-norm head and waste precision
on all other heads.
```

**Dequantization at read time — fused inside the attention kernel:**

```
Each decode step reads the full KV cache for this request:

  Read K_int8 + k_scales from HBM   ← 2× less bytes than K_bf16
  
  Inside the Flash Attention kernel (no separate dequant kernel launch):
    for each KV block:
      K_bf16 = K_int8.to(bfloat16) * k_scales   ← in registers, during memory fetch
      scores = Q_bf16 @ K_bf16.T                 ← BF16 attention score
      ...

  The dequantization runs in the SM's ALUs while the next KV block is being
  fetched from HBM — it fills the memory latency gap rather than adding to it.
```

**vLLM implementation:**

```python
# Attention kernel call in vllm/model_executor/layers/attention/backends/flash_attn.py
flash_attn_with_kvcache(
    q,
    k_cache,          # INT8 or FP8 tensor in the KV block
    v_cache,          # INT8 or FP8 tensor in the KV block
    kv_scale=kv_scale,  # passed to kernel — dequantization fused inside
    ...
)

# The kv_scale is set in vllm/model_executor/layers/quantization/kv_cache.py
# Each model layer stores one scale per attention head, updated at each prefill step
```

**Why per-token-per-head and NOT per-group for KV:**

```
Weight quantization uses per-group (G=128) because:
  weights are large matrices — many elements per group → fine resolution
  the group is a contiguous block of the weight matrix

KV quantization uses per-token-per-head because:
  one KV vector = head_dim=64 elements — already small
  splitting into groups of 64 = one group per vector anyway
  a "group" would just be the whole vector → per-token-per-head IS per-group for KV
```

**The full picture — where each quantization lives in the forward pass:**

```
Input tokens
      │
      ▼
Embedding (BF16)
      │
      ▼  ── for each transformer layer ──────────────────────────────────
      │
      ├── Q proj (QuantLinear):
      │     quantize x_bf16 → x_int8         ← activation quantization
      │     int8_matmul → Q_bf16
      │
      ├── K proj (QuantLinear):
      │     quantize x_bf16 → x_int8         ← activation quantization
      │     int8_matmul → K_bf16
      │     quantize K_bf16 → K_int8         ← KV cache quantization (write)
      │     store (K_int8, k_scale) in HBM block
      │
      ├── V proj (QuantLinear):   same → store (V_int8, v_scale) in HBM block
      │
      ├── Attention:
      │     read (K_int8, k_scale) from HBM  ← 2× less HBM traffic
      │     dequantize K inside kernel        ← fused, in registers
      │     softmax(Q @ K.T) @ V in BF16
      │
      ├── O proj (QuantLinear):   activation quantize → int8_matmul → BF16
      │
      └── MLP (two QuantLinear):  activation quantize → int8_matmul → BF16
      │
      ▼  ── end of layer ─────────────────────────────────────────────────
      │
LM head → logits (BF16) → next token
```

**Summary — weight vs activation vs KV cache quantization:**

```
                  Weights              Activations          KV Cache
                  ──────────────────   ──────────────────   ──────────────────────
When quantized:   load time (or        every forward pass   every token written
                  offline)             (runtime)            (runtime)
Scale computed:   once (offline)       per-token or         per-token-per-head
                                       per-tensor at        at write time
                                       runtime
Scale stored:     with weight tensor   not stored           with KV block in HBM
                  in HBM              (recomputed each time) (must survive across steps)
Dequantize when:  before matmul        output of matmul     inside attention kernel
                  (in registers)       (rescale to BF16)    (fused, in registers)
Primary goal:     bandwidth + VRAM     enable INT8/FP8      VRAM (more concurrent
                                       tensor core compute  requests)
```

---

## 10. Comparison Table

| Method | Type | Memory savings | Accuracy loss | Calibration needed | Hardware requirement |
|---|---|---|---|---|---|
| INT8 PTQ (bnb) | W8A16 | 2× weights | near-zero | no | any GPU |
| INT4 PTQ NF4 | W4A16 | 4× weights | small | no | any GPU |
| GPTQ | W4A16 | 4× weights | small | yes (128 samples) | any GPU |
| AWQ | W4A16 | 4× weights | small | yes (128 samples) | any GPU |
| SmoothQuant | W8A8 | 2× weights | near-zero | yes | any GPU with INT8 TCs |
| FP8 W8A8 | W8A8 | 2× weights + 2× compute | near-zero | yes (static) or no (dynamic) | H100 / A100 |
| KV FP8/INT8 | KV only | 2× KV cache | near-zero | no | any GPU |
| NVFP4 | W4A4 | 4× weights | small | yes | Blackwell |
| GGUF Q4_K_M | W4 mixed | ~4× weights | small-medium | no | any (CPU+GPU) |

---

## 11. What to implement (Phase 5 plan)

| Step | What | Tool | Profile signal |
|---|---|---|---|
| 1 | INT8 weights | bitsandbytes `load_in_8bit` | VRAM: 234 MB → 117 MB weights; throughput delta |
| 2 | INT4 NF4 weights | bitsandbytes `load_in_4bit` | VRAM: → ~60 MB weights; perplexity delta |
| 3 | GPTQ offline | auto-gptq | Same as INT4 but better perplexity — compare |
| 4 | AWQ offline | autoawq | Compare perplexity to GPTQ at same bit width |
| 5 | FP8 (if A100 supports) | vLLM + FP8 | Roofline shift in Nsight — memory → compute bound |
| 6 | KV cache INT8 | vLLM `kv_cache_dtype="int8"` | Max batch size increase before OOM |

**Deliverable:** table with method → VRAM → tokens/sec → perplexity (ppl on wikitext-2).
