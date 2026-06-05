# CUDA Optimization Tactics

Every tactic below solves a specific bottleneck. CUDA/cuBLAS applies them based on matrix shape, data type, and GPU architecture — detailed in the "When applied" section for each.

---

## Category 1 — Parallelism Tactics

How to split work across SMs to maximize utilization.

---

### 1.1 Split-K

**Problem:** GEMV (batch=1) produces few output elements — too few to keep all SMs busy.

**Toy example:** output = weight @ input, weight=(4×10), input=(10,)

Without Split-K only 4 SMs work (one per output row). A100 has 108 SMs — 104 idle.

**Step-by-step with Split-K=2:**

```
weight (4×10), input=[1,1,1,1,1,1,1,1,1,1]

Step 1 — main kernel splits K=10 into 2 chunks of 5:

  SM group 0 (row 0, K[0:5]):   partial[0,0] = 1+2+3+4+5   = 15
  SM group 1 (row 0, K[5:10]):  partial[0,1] = 6+7+8+9+10  = 40
  SM group 2 (row 1, K[0:5]):   partial[1,0] = 2+3+4+5+6   = 20
  SM group 3 (row 1, K[5:10]):  partial[1,1] = 7+8+9+10+11 = 45
  SM group 4 (row 2, K[0:5]):   partial[2,0] = 5
  SM group 5 (row 2, K[5:10]):  partial[2,1] = 5
  SM group 6 (row 3, K[0:5]):   partial[3,0] = 40
  SM group 7 (row 3, K[5:10]):  partial[3,1] = 15

  → 8 SM groups running in parallel vs 4 before
  → partial sums written to HBM workspace buffer

Step 2 — split_k_reduce_kernel sums partials:

  output[0] = 15 + 40 = 55
  output[1] = 20 + 45 = 65
  output[2] =  5 +  5 = 10
  output[3] = 40 + 15 = 55
```

**Tradeoff:** Extra HBM round-trip to write/read partial sums. Worth it when SM utilization gain exceeds memory cost.

**When applied:** Large K, small M×N (batch=1 MLP layers: K=3072, M=1). Not applied for large batch (M×N already large enough to saturate SMs).

---

### 1.2 Stream-K

**Problem:** Split-K assigns a fixed K-chunk per SM — uneven work if K is not divisible evenly. Some SMs finish early and sit idle.

**Toy example:** K=10, split=3 (uneven: chunks of 4, 3, 3)

```
Split-K (uneven):
  SM 0: K[0:4]  → 4 multiplies
  SM 1: K[4:7]  → 3 multiplies   ← finishes early, waits
  SM 2: K[7:10] → 3 multiplies   ← finishes early, waits

Stream-K (work stealing):
  Total multiply-adds = 10
  3 SMs → each claims ~3-4 units dynamically:
  SM 0: processes units 0,1,2,3   (claims from global counter)
  SM 1: processes units 4,5,6     (claims next available)
  SM 2: processes units 7,8,9     (claims next available)
  → all 3 SMs finish at roughly the same time
```

**Step-by-step:**
```
1. Global atomic counter starts at 0
2. Each SM atomically increments counter to claim its next work unit
3. SM processes claimed unit, then immediately claims the next
4. No SM waits — if one finishes early it claims more work
5. Final reduction same as Split-K
```

**Tradeoff:** Atomic counter overhead. Reduces tail latency from SM imbalance.

**When applied:** cuBLAS uses Stream-K for irregular matrix shapes where K % split_factor != 0. Available from CUDA 12+ / cuBLAS 12+.

---

### 1.3 Persistent Kernels

**Problem:** Each tile of output requires a kernel launch → launch overhead (~5μs) repeated for every tile. For small tiles this overhead dominates.

**Toy example:** Matrix C = A @ B, output has 16 tiles of 32×32 each.

```
Without persistent kernels (16 launches):
  Launch kernel → SM computes tile 0 → kernel exits
  Launch kernel → SM computes tile 1 → kernel exits
  ...×16
  Total launch overhead: 16 × 5μs = 80μs

With persistent kernel (1 launch):
  Launch kernel once → SM computes tile 0
                     → SM computes tile 1   ← SM stays alive, loops internally
                     → SM computes tile 2
                     ...
                     → SM computes tile 15 → kernel exits
  Total launch overhead: 1 × 5μs = 5μs
```

**Step-by-step:**
```
1. Launch ONE kernel with enough thread blocks to fill all SMs
2. Each thread block runs a loop internally:
     while (tile_id < total_tiles):
         tile_id = atomicAdd(global_counter, 1)   ← claim next tile
         compute output tile[tile_id]
         write result to C
3. SMs stay alive until all tiles are done — no re-launch
```

**Tradeoff:** Kernel is more complex (internal loop + atomic). Cannot be interrupted mid-kernel.

**When applied:** Flash Attention uses this. cuBLAS uses it for very large matrices where tile count is high and launch overhead would otherwise accumulate.

---

## Category 2 — Memory Hierarchy Tactics

How to move data efficiently through the memory hierarchy (HBM → L2 → SRAM → registers).

---

### 2.1 Tiling (Shared Memory Blocking)

**Problem:** Matrix multiply naively reads each element from HBM many times. HBM is slow (1.6 TB/s). SRAM is fast (10+ TB/s, 96KB/SM). Load once into SRAM, reuse many times.

**Toy example:** C = A @ B, all 4×4 matrices, tile size 2×2.

```
A (4×4):          B (4×4):
[1  2  3  4]      [1  2  3  4]
[5  6  7  8]      [5  6  7  8]
[9  10 11 12]     [9  10 11 12]
[13 14 15 16]     [13 14 15 16]
```

**Without tiling — naive (HBM reads per output element):**
```
C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
       = 1×1 + 2×5 + 3×9 + 4×13 = 1+10+27+52 = 90

To compute C[0,0]: read row 0 of A (4 reads) + col 0 of B (4 reads) = 8 HBM reads
Total for all 16 outputs: 16 × 8 = 128 HBM reads
```

**With tiling (tile=2×2) — load into SRAM, reuse:**
```
Tile iteration 0: load A[0:2, 0:2] and B[0:2, 0:2] into SRAM (8 HBM reads total)
  SRAM_A = [[1,2],[5,6]]     SRAM_B = [[1,2],[5,6]]
  Compute partial C[0:2, 0:2]:
    C[0,0] += 1×1 + 2×5 = 11   (using SRAM — no HBM)
    C[0,1] += 1×2 + 2×6 = 14
    C[1,0] += 5×1 + 6×5 = 35
    C[1,1] += 5×2 + 6×6 = 46

Tile iteration 1: load A[0:2, 2:4] and B[2:4, 0:2] into SRAM (8 HBM reads)
  SRAM_A = [[3,4],[7,8]]     SRAM_B = [[9,10],[13,14]]
  Accumulate into C[0:2, 0:2]:
    C[0,0] += 3×9 + 4×13 = 27+52 = 79  → C[0,0] = 11+79 = 90  ✓
    C[0,1] += 3×10 + 4×14 = 30+56 = 86 → C[0,1] = 14+86 = 100
    C[1,0] += 7×9 + 8×13 = 63+104 = 167 → C[1,0] = 35+167 = 202
    C[1,1] += 7×10 + 8×14 = 70+112 = 182 → C[1,1] = 46+182 = 228

... repeat for remaining 3 tile blocks of C
```

**HBM reads comparison:**
```
Naive:  128 HBM reads
Tiled:  32 HBM reads (each tile loaded once, reused for all outputs in that tile)
        4× reduction in HBM traffic
```

**When applied:** Always — tiling is the foundation of every GEMM kernel. Tile size (32×32, 64×64, 128×128) chosen based on available SRAM (96KB/SM on A100) and register pressure.

---

### 2.2 Vectorized Loads

**Problem:** Each HBM load instruction fetches 32 bits (one float). HBM bandwidth is maximized when each transaction is 128 bits (4 floats). Loading one float at a time wastes 75% of available bandwidth per transaction.

**Toy example:** Load 8 floats from HBM into registers.

```
Array in HBM: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
Address:        0x00 0x04 0x08 0x0C 0x10 0x14 0x18 0x1C  (4 bytes each)
```

**Without vectorized loads (scalar, 32-bit):**
```
Thread 0: LD.F32 r0, [addr+0x00]   → 32-bit load → 1 float
Thread 0: LD.F32 r1, [addr+0x04]   → 32-bit load → 1 float
Thread 0: LD.F32 r2, [addr+0x08]   → 32-bit load → 1 float
Thread 0: LD.F32 r3, [addr+0x0C]   → 32-bit load → 1 float
→ 4 separate load instructions, 4 separate cache transactions
```

**With vectorized loads (float4, 128-bit):**
```
Thread 0: LD.F32x4 {r0,r1,r2,r3}, [addr+0x00]  → 128-bit load → 4 floats in ONE instruction
→ 1 load instruction, 1 cache transaction, same result
```

**In CUDA C++:**
```cpp
// Scalar
float a = ptr[0];
float b = ptr[1];
float c = ptr[2];
float d = ptr[3];

// Vectorized (float4 = 128-bit)
float4 vec = *reinterpret_cast<float4*>(ptr);  // a=vec.x, b=vec.y, c=vec.z, d=vec.w
```

**Bandwidth impact:**
```
Scalar:      4 instructions × 32-bit = 128 bits loaded but 4 instruction slots used
Vectorized:  1 instruction  × 128-bit = 128 bits loaded in 1 instruction slot
→ 4× more data per instruction → same bandwidth, fewer instruction cycles wasted
```

**When applied:** Always when data is 128-bit aligned. cuBLAS aligns all weight matrices to 128-bit boundaries. PyTorch tensor allocations are aligned by default.

---

### 2.3 Memory Coalescing

**Problem:** A warp has 32 threads. If all 32 threads load from scattered (non-consecutive) addresses, the GPU issues 32 separate HBM transactions. If they load from consecutive addresses, it issues ONE transaction for all 32 threads.

**Toy example:** 32 threads loading from an array of 32 floats.

**Coalesced access (consecutive addresses):**
```
Thread  0: load addr[0]   ┐
Thread  1: load addr[1]   │
Thread  2: load addr[2]   │ → GPU issues ONE 128-byte HBM transaction
...                        │   covering addresses [0, 128)
Thread 31: load addr[31]  ┘   all 32 threads satisfied in 1 transaction
```

**Non-coalesced access (strided — e.g., column-major matrix read):**
```
Thread  0: load addr[0]    → transaction 1: fetch [0,128)
Thread  1: load addr[32]   → transaction 2: fetch [128,256)  ← different cache line
Thread  2: load addr[64]   → transaction 3: fetch [256,384)
...
Thread 31: load addr[992]  → transaction 32: fetch [3968,4096)
→ 32 separate HBM transactions for 32 floats — 32× worse bandwidth utilization
```

**Concrete matrix example:**
```
Matrix A (4×4), stored row-major in HBM:
  [a00, a01, a02, a03, a10, a11, a12, a13, a20, ...]
   addr:0   4   8   12  16  20  24   28   32

Reading row 0 (coalesced):
  4 threads load a00, a01, a02, a03 → consecutive → 1 transaction ✓

Reading column 0 (non-coalesced, stride=4):
  4 threads load a00, a10, a20, a30 → stride 4 → 4 transactions ✗
```

**How cuBLAS fixes this:** Transposes one matrix operand before the kernel so both reads are row-major (coalesced). The `cublasSgemm` `transa`/`transb` flags control this.

**When applied:** Always considered in kernel design. cuBLAS kernel variants exist for different transposition combinations (NN, NT, TN, TT) precisely to ensure coalesced access.

---

### 2.4 Double Buffering (Async Prefetch)

**Problem:** While the SM is computing with tile N in SRAM, it is idle waiting for tile N+1 to load from HBM. Memory load and compute happen sequentially.

**Toy example:** 4 tiles to process, each tile takes 10μs to load, 10μs to compute.

**Without double buffering:**
```
Time:  0     10    20    30    40    50    60    70    80
       [load0][comp0][load1][comp1][load2][comp2][load3][comp3]
       Total: 80μs  (load and compute strictly sequential)
```

**With double buffering (two SRAM buffers, async prefetch):**
```
Time:  0     10    20    30    40    50
       [load0]
              [comp0 + load1 in background]
                    [comp1 + load2 in background]
                          [comp2 + load3 in background]
                                [comp3]
       Total: 50μs  (compute and memory overlapped)
```

**Step-by-step:**
```
1. Allocate two SRAM buffers: buf[0] and buf[1]
2. Load tile 0 into buf[0] synchronously (prime the pipeline)
3. Loop:
     issue async load of tile i+1 into buf[(i+1)%2]   ← fires DMA, CPU continues
     compute with tile i from buf[i%2]                 ← SM computes while DMA runs
     wait for async load to complete                    ← sync point (minimal wait)
     swap active buffer
4. Compute last tile with no prefetch
```

**In CUDA (cp.async instruction):**
```cpp
// Issue async copy from HBM to SRAM — returns immediately
__pipeline_memcpy_async(sram_buf_next, hbm_ptr_next, tile_bytes);

// Compute with current tile while async copy runs
compute(sram_buf_current);

// Wait for async copy to complete before using it
__pipeline_commit();
__pipeline_wait_prior(0);
```

**When applied:** All high-performance GEMM kernels in cuBLAS use double buffering. Requires `cp.async` (Ampere A100+). Essential for hiding HBM latency (~400 cycles) behind compute.

---

### 2.5 Register Blocking

**Problem:** Shared memory has limited bandwidth and latency (~20 cycles). For the innermost compute loop, even SRAM is too slow — registers are zero-latency.

**Toy example:** Thread computes a 2×2 block of output C.

**Without register blocking (one output at a time):**
```
for k in range(K):
    C[0,0] += A[0,k] * B[k,0]   ← load A, load B, compute, write C — each step hits SRAM
    C[0,1] += A[0,k] * B[k,1]
    C[1,0] += A[1,k] * B[k,0]
    C[1,1] += A[1,k] * B[k,1]
→ C values read from SRAM every iteration
```

**With register blocking (accumulate in registers):**
```
# Initialize registers (zero-latency access)
reg_c00, reg_c01, reg_c10, reg_c11 = 0.0, 0.0, 0.0, 0.0

for k in range(K):
    reg_a0 = SRAM_A[0, k]   # load once
    reg_a1 = SRAM_A[1, k]   # load once
    reg_b0 = SRAM_B[k, 0]   # load once
    reg_b1 = SRAM_B[k, 1]   # load once

    reg_c00 += reg_a0 * reg_b0   # pure register ops — zero memory access
    reg_c01 += reg_a0 * reg_b1
    reg_c10 += reg_a1 * reg_b0
    reg_c11 += reg_a1 * reg_b1

# Write final accumulated values to SRAM/HBM only once
C[0,0] = reg_c00
C[0,1] = reg_c01
C[1,0] = reg_c10
C[1,1] = reg_c11
```

**Savings:**
```
Without: K × 4 SRAM reads for C (read-modify-write each iteration)
With:    K × 4 register ops (zero latency) + 4 SRAM writes at end
→ eliminates K × 4 SRAM accesses from the innermost loop
```

**When applied:** Always in cuBLAS GEMM kernels. Typical register block size: 8×8 per thread (64 accumulator registers). Tradeoff: more registers per thread → lower occupancy (fewer threads per SM). cuBLAS tunes this per architecture.

---

## Category 3 — Compute Tactics

How to execute arithmetic faster using specialized hardware.

---

### 3.1 Tensor Core WMMA (Warp Matrix Multiply Accumulate)

**Problem:** CUDA cores do one FP32 multiply-add per cycle. Tensor Cores do a 16×16×16 matrix multiply in one instruction — 16× more compute per cycle.

**Toy example (conceptual — hardware operates at 16×16×16):**

```
# Using 4×4 to show the concept:
A (4×4, FP16):              B (4×4, FP16):
[[1, 2, 3, 4],              [[1, 0, 0, 0],
 [5, 6, 7, 8],               [0, 1, 0, 0],
 [1, 1, 1, 1],               [0, 0, 1, 0],
 [2, 2, 2, 2]]               [0, 0, 0, 1]]

CUDA Core approach (scalar FMA):
  C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
         = 1×1 + 2×0 + 3×0 + 4×0 = 1
  → 4 multiply-adds for ONE output element
  → 16 output elements × 4 = 64 FMA instructions total

Tensor Core approach (WMMA instruction):
  wmma::mma_sync(C_frag, A_frag, B_frag, C_frag)
  → ONE instruction computes ALL 16 output elements simultaneously
  → 64 FMAs done in hardware in 1 clock cycle by the Tensor Core unit
```

**Real hardware — 16×16×16:**
```
One Tensor Core instruction:
  D (16×16, FP32) = A (16×16, FP16) × B (16×16, FP16) + C (16×16, FP32)
  = 16×16×16 = 4096 multiply-adds in ONE instruction

One CUDA core instruction:
  d = a * b + c    ← 1 multiply-add

Ratio: 4096× more compute per instruction
```

**In CUDA C++:**
```cpp
// Load matrix fragments into registers
wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16,16,16, float> c_frag;

wmma::load_matrix_sync(a_frag, A_ptr, 16);
wmma::load_matrix_sync(b_frag, B_ptr, 16);
wmma::fill_fragment(c_frag, 0.0f);

// ONE instruction = 16×16×16 FMAs
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

wmma::store_matrix_sync(C_ptr, c_frag, 16, wmma::mem_row_major);
```

**When applied:** Only for FP16 or BF16 inputs. M, N, K must be multiples of 16. cuBLAS uses Tensor Cores automatically when dtype=FP16/BF16. For FP32, falls back to CUDA cores (19.5 TFLOPS vs 312 TFLOPS — 16× difference).

---

### 3.2 FMA (Fused Multiply-Add)

**Problem:** Computing `a*b + c` as two separate instructions introduces intermediate rounding error and uses two instruction slots.

**Toy example:**
```
a = 1.0000001  (float32)
b = 2.0
c = -2.0000002

# Without FMA (two separate ops):
tmp = a * b        → rounds to nearest float32: 2.0000002
result = tmp + c   → 2.0000002 + (-2.0000002) = 0.0   ← correct by luck

# Another case:
a = 1.0000001
b = 3.0
c = -3.0000004

tmp = a * b        → rounds: 3.0000002  (rounding error introduced here)
result = tmp + c   → 3.0000002 - 3.0000004 = -0.0000002  ← wrong

# With FMA (one instruction, intermediate stored at full precision):
result = fma(a, b, c) = a*b + c computed at 64-bit internally, rounded once at end
       = 3.0000003 - 3.0000004 = -0.0000001  ← more accurate
```

**Performance benefit:**
```
Without FMA:  MUL instruction + ADD instruction = 2 instruction slots
With FMA:     FMADD instruction = 1 instruction slot
→ 2× throughput for multiply-accumulate chains
```

**When applied:** Always. Modern NVIDIA GPUs automatically fuse `a*b + c` patterns into FMADD at the hardware level. Compilers (`nvcc`) emit FMA instructions by default. The innermost GEMM loop is entirely FMA instructions.

---

### 3.3 Warp Shuffle Reduction

**Problem:** Reducing 32 values (one per thread in a warp) to a single sum. Naive approach writes all values to shared memory, then reads back — multiple SRAM accesses and sync barriers needed.

**Toy example:** 8 threads (simplified warp), each holds one value. Compute the sum.

```
Initial state:
  Thread 0: val=1
  Thread 1: val=2
  Thread 2: val=3
  Thread 3: val=4
  Thread 4: val=5
  Thread 5: val=6
  Thread 6: val=7
  Thread 7: val=8
  Goal: all threads know sum=36
```

**Without warp shuffle (shared memory approach):**
```
1. All threads write val to shmem[thread_id]        ← 8 SRAM writes
2. __syncthreads()                                   ← sync barrier
3. Thread 0 reads all 8 values and sums:             ← 8 SRAM reads
   sum = 1+2+3+4+5+6+7+8 = 36
4. Thread 0 writes sum to shmem[0]                   ← 1 SRAM write
5. __syncthreads()
6. All threads read shmem[0]                         ← 8 SRAM reads
→ 25 SRAM ops + 2 sync barriers
```

**With warp shuffle (__shfl_xor_sync):**
```
Round 1 (offset=4): each thread XORs its value with thread (id XOR 4)
  Thread 0 ↔ Thread 4:  val[0] = 1+5 = 6,   val[4] = 5+1 = 6
  Thread 1 ↔ Thread 5:  val[1] = 2+6 = 8,   val[5] = 6+2 = 8
  Thread 2 ↔ Thread 6:  val[2] = 3+7 = 10,  val[6] = 7+3 = 10
  Thread 3 ↔ Thread 7:  val[3] = 4+8 = 12,  val[7] = 8+4 = 12

Round 2 (offset=2):
  Thread 0 ↔ Thread 2:  val[0] = 6+10 = 16,  val[2] = 10+6 = 16
  Thread 1 ↔ Thread 3:  val[1] = 8+12 = 20,  val[3] = 12+8 = 20
  Thread 4 ↔ Thread 6:  val[4] = 6+10 = 16,  val[6] = 10+6 = 16
  Thread 5 ↔ Thread 7:  val[5] = 8+12 = 20,  val[7] = 12+8 = 20

Round 3 (offset=1):
  Thread 0 ↔ Thread 1:  val[0] = 16+20 = 36  ✓
  Thread 2 ↔ Thread 3:  val[2] = 16+20 = 36  ✓
  ... all threads now hold 36

→ 0 SRAM ops, 0 sync barriers — data moves directly between thread registers
```

**In CUDA C++:**
```cpp
// Warp reduction in 5 lines
int val = thread_value;
for (int offset = warpSize/2; offset > 0; offset >>= 1)
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
// val now contains the sum across all 32 threads in the warp
```

**When applied:** All reduction operations in CUDA — softmax, layernorm, attention row-sum, Split-K final reduction. cuBLAS uses warp shuffles for the epilogue reduction step. Faster than shared memory for reductions within a warp.

---

## Category 4 — Fusion Tactics

Eliminating kernel launch boundaries to avoid unnecessary HBM round-trips.

---

### 4.1 Epilogue Fusion (Bias Add Inside GEMM)

**Problem:** After a linear layer, bias is added to every output element. Naively this is a separate kernel — reads the GEMM output from HBM, adds bias, writes back. Unnecessary HBM round-trip.

**Toy example:** Linear layer: output = input @ weight + bias
```
weight (4×4), input (4,), bias=[10, 20, 30, 40]
GEMM result (before bias): [90, 100, 40, 55]   (made-up values)
```

**Without epilogue fusion (2 kernels):**
```
Kernel 1 (GEMM):
  SM computes output[i] = sum(input × weight[i, :])
  Writes [90, 100, 40, 55] to HBM                     ← HBM write

Kernel 2 (bias add):
  Reads [90, 100, 40, 55] from HBM                    ← HBM read (wasted)
  Adds bias: [90+10, 100+20, 40+30, 55+40]
  Writes [100, 120, 70, 95] to HBM                    ← HBM write
```

**With epilogue fusion (1 kernel):**
```
Kernel 1 (GEMM + bias fused):
  SM computes output[i] = sum(input × weight[i, :])
  Before writing to HBM, adds bias while value is still in register:
    reg = 90 → reg + bias[0] = 100  → write 100 to HBM
    reg = 100 → reg + bias[1] = 120 → write 120 to HBM
    ...
  Writes [100, 120, 70, 95] directly to HBM           ← 1 HBM write, no intermediate read
```

**Savings:**
```
Without: 2 kernel launches + 1 extra HBM read + 1 extra HBM write per element
With:    1 kernel launch + bias computed in registers for free
→ eliminates 2N HBM ops (N = output size)
```

**When applied:** Always for `addmm` (PyTorch's fused linear layer). The `cublasLtMatmul` epilogue parameter specifies the fusion: `CUBLASLT_EPILOGUE_BIAS`, `CUBLASLT_EPILOGUE_RELU`, `CUBLASLT_EPILOGUE_GELU`. This is why you don't see a separate bias kernel in the CUDA API row.

---

### 4.2 Flash Attention (Attention Kernel Fusion)

**Problem:** Standard attention computes S = Q @ K.T (N×N matrix), applies softmax, then O = S @ V. The N×N attention matrix (N=sequence length) must be written to HBM and read back — O(N²) HBM traffic.

**Toy example:** seq_len=4, head_dim=2. Attention matrix S is 4×4 = 16 elements.

**Without Flash Attention (3 kernels):**
```
Kernel 1: S = Q @ K.T        → writes 4×4=16 floats to HBM
Kernel 2: P = softmax(S)     → reads 16 floats from HBM, writes 16 back
Kernel 3: O = P @ V          → reads 16 floats from HBM

HBM traffic for attention matrix: 16 × 3 = 48 float reads/writes
```

**With Flash Attention (1 kernel, tiled):**
```
Process Q in tiles of Br=2 rows, K/V in tiles of Bc=2 columns.
Keep running softmax state (m=running max, l=running sum) in registers.

Tile (Q[0:2], K[0:2]):
  S_00 = Q[0:2] @ K[0:2].T     ← computed in SRAM, never in HBM
  m_0 = rowmax(S_00) = [3, 7]
  l_0 = rowsum(exp(S_00 - m_0))
  O_0 = softmax(S_00) @ V[0:2]  ← partial output, in SRAM

Tile (Q[0:2], K[2:4]):
  S_01 = Q[0:2] @ K[2:4].T     ← in SRAM
  m_1 = max(m_0, rowmax(S_01))  ← update running max
  rescale O_0 and l_0 for new max
  O_0 += softmax(S_01) @ V[2:4] ← accumulate into same output register
  l_1 = rescaled_l_0 + rowsum(exp(S_01 - m_1))

Finalize:
  O[0:2] = O_0 / l_1            ← normalize, write to HBM ONCE
```

**HBM traffic:**
```
Without Flash Attention: O(N²) — N×N attention matrix written/read 3 times
With Flash Attention:    O(N)  — only Q, K, V (input) and O (output) touch HBM
                                 attention matrix never leaves SRAM

For seq_len=1024, head_dim=64:
  Without: 1024×1024 × 3 = 3M float HBM ops just for attention weights
  With:    eliminated entirely — stays in SRAM tile-by-tile
```

**When applied:** PyTorch 2.0+ uses Flash Attention automatically via `F.scaled_dot_product_attention`. Required for long sequences where N² HBM traffic would dominate. Not needed for very short sequences (N<64) where the attention matrix fits in SRAM anyway.

---

### 4.3 Elementwise Chain Fusion

**Problem:** After attention, transformers apply several elementwise ops: LayerNorm → Residual Add → GELU → Dropout. Each as a separate kernel reads and writes the full tensor from/to HBM.

**Toy example:** tensor x of 8 elements, apply: y = GELU(x + residual) / norm_factor

```
x        = [1.0, -1.0, 2.0, 0.5, -0.5, 3.0, 1.5, -2.0]
residual  = [0.1,  0.1, 0.1, 0.1,  0.1, 0.1, 0.1,  0.1]
norm      = 1.5
```

**Without fusion (3 separate kernels):**
```
Kernel 1 (residual add):
  reads x (8 floats) from HBM
  reads residual (8 floats) from HBM
  tmp = x + residual = [1.1, -0.9, 2.1, 0.6, -0.4, 3.1, 1.6, -1.9]
  writes tmp (8 floats) to HBM                 ← 24 HBM ops

Kernel 2 (GELU):
  reads tmp (8 floats) from HBM
  tmp2 = GELU(tmp) = [0.91, -0.16, 1.97, 0.40, -0.15, 3.07, 1.40, -0.14]
  writes tmp2 (8 floats) to HBM                ← 16 HBM ops

Kernel 3 (normalize):
  reads tmp2 (8 floats) from HBM
  y = tmp2 / norm_factor
  writes y (8 floats) to HBM                   ← 16 HBM ops

Total: 56 HBM ops for 8 elements
```

**With fusion (1 kernel):**
```
Kernel 1 (fused: add + GELU + normalize):
  Thread i:
    reads x[i] from HBM       ← 1 read
    reads residual[i] from HBM ← 1 read
    tmp  = x[i] + residual[i]  ← in register, no HBM
    tmp2 = GELU(tmp)            ← in register, no HBM
    y[i] = tmp2 / norm_factor   ← in register, no HBM
    writes y[i] to HBM          ← 1 write

Total: 3 HBM ops per element × 8 = 24 HBM ops
→ 56 → 24: 2.3× reduction in HBM traffic
```

**When applied:** PyTorch's `torch.compile` and `triton` auto-fuse elementwise chains. cuBLAS epilogue fusion handles the first elementwise op after GEMM. Subsequent chains are fused by the compiler. The rule: if an op only reads and writes the same tensor (no reduction), it can be fused with adjacent elementwise ops — all computed in registers in a single pass over the data.

---

## How cuBLAS Chooses Which Tactics to Apply

`cublasLtMatMulAlgoGetHeuristic` takes the operation parameters and returns the best algorithm:

```
Inputs:
  M, N, K         ← matrix dimensions
  dtype            ← FP32 / FP16 / BF16 / INT8
  GPU architecture ← A100, H100, ...
  available SRAM   ← 96KB/SM on A100

Decision table (simplified):
  dtype = FP16/BF16  → use Tensor Cores  (else CUDA cores)
  M×N small, K large → apply Split-K     (more SM parallelism)
  M×N large          → no Split-K        (already enough tiles)
  tile size           → 64×64 for medium M/N, 128×128 for large
  vectorized loads    → always (alignment check)
  double buffering    → always (Ampere+)
  epilogue            → fuse bias if bias ptr provided
```

**Cold vs warm selection:**
```
Cold (first call for this shape):
  cuBLAS runs a quick benchmark of top-N candidate algorithms
  picks the fastest one
  stores result in algorithm cache

Warm (same shape seen before):
  cache lookup: O(1), ~5μs (the GetHeuristic call you see in CUDA API row)
  no benchmarking needed
```

This is why the first inference pass (warmup) is slower — algorithm selection runs in benchmarking mode. Subsequent passes hit the cache.
