# Warp and Thread Execution — From Grid Launch to Instruction Issue

---

## Step 1 — Kernel Launch: Grid of Blocks

When the CPU calls a CUDA kernel, it specifies a **grid** — a 3D arrangement of thread blocks:

```
cudaLaunchKernel(kernel, grid=(4281, 1, 1), block=(128, 1, 1))

Grid:
  Total blocks = 4281 × 1 × 1 = 4281 blocks
  Each block has 128 × 1 × 1 = 128 threads
  Total threads = 4281 × 128 = 548,368 threads
```

The grid describes the full problem. The block describes the unit of work assigned to one SM.

**Hardware limits — these are hard constraints, not conventions:**

```
Max threads per block:  1,024
  blockDim.x × blockDim.y × blockDim.z must be ≤ 1,024
  Exceeding this → cudaErrorInvalidConfiguration at launch (immediate failure)
  Max warps per block = 1,024 / 32 = 32

Max grid dimensions (compute capability ≥ 2.0, covers all modern GPUs):
  gridDim.x:  2^31 − 1  (~2.1 billion)
  gridDim.y:  65,535
  gridDim.z:  65,535
```

Grid limits are never a practical constraint — you can launch billions of blocks in x.
Total blocks in a grid >> blocks running concurrently; the excess queues in GigaThread
Engine and is dispatched as SMs free up. The real constraint is always SM resources.

---

## Step 2 — Block Assignment to SMs

The GPU's **GigaThread Engine** (hardware block scheduler) distributes blocks across SMs.

- A block is assigned to an SM **in its entirety** — it never splits across two SMs
- Multiple blocks can co-reside on the same SM if resources allow (registers, shared memory)
- The SM runs its assigned blocks until they complete, then picks up new ones from the queue

```
A100: 108 SMs

Grid has 4281 blocks → GigaThread distributes:
  SM 0  → blocks 0, 1, ..., (as many as fit)
  SM 1  → next batch of blocks
  ...
  SM 107 → last batch

As blocks finish, the SM pulls the next unassigned block from the queue.
```

**Resource check before assignment:** Before placing a block on an SM, the scheduler
verifies the SM has enough:
1. **Registers** — block's threads × registers/thread ≤ SM's remaining register file
2. **Shared memory** — block's shared memory request ≤ SM's remaining shared memory
3. **Warp slots** — block's warps ≤ SM's remaining warp slot capacity

If any check fails, the block waits until a running block finishes and frees resources.

---

## Step 3 — Threads Divided Into Warps

Once a block lands on an SM, its threads are divided into **warps of 32 threads**:

```
Block with 128 threads:
  Warp 0 → threads  0–31
  Warp 1 → threads 32–63
  Warp 2 → threads 64–95
  Warp 3 → threads 96–127
  Total: 4 warps per block
```

The warp is the fundamental unit of execution on the GPU. The SM never executes a single
thread — it always executes 32 threads together as a warp.

**Block size should be a multiple of 32.** A block of 100 threads creates 4 warps, but
the last warp (threads 96–99) has only 4 active threads — the other 28 slots are wasted.
The hardware still pays the cost of executing a full warp.

---

## Step 4 — Register Allocation

Before any warp executes, the SM allocates registers from its register file.

**A100 SM register file: 65,536 × 32-bit registers** — shared across all co-resident warps.

```
Kernel compiled with 16 registers/thread:
  16 registers × 32 threads = 512 registers per warp
  65,536 / 512 = 128 warps could fit (A100 SM caps at 64 warps — register file not limiting)

Kernel compiled with 64 registers/thread:
  64 × 32 = 2,048 registers per warp
  65,536 / 2,048 = 32 warps maximum → register file IS the bottleneck

Kernel compiled with 128 registers/thread:
  128 × 32 = 4,096 registers per warp
  65,536 / 4,096 = 16 warps maximum → severely limited
```

Registers are allocated **statically at compile time** — the compiler analyzes the kernel
and determines exactly how many registers each thread needs. This count is fixed in the
binary. ncu reads it from the compiled kernel and shows it in the Registers/Thread column.

Once allocated, each thread has exclusive access to its registers — no sharing between
threads. This is what makes GPU registers faster than shared memory (no synchronization
needed).

**What happens when threads × registers exceeds the SM's register budget:**

GigaThread Engine can only assign a block if the SM has enough registers for it. If a
block would require more registers than the SM has remaining, it waits. If it would
require more than the SM has *in total* (even alone), it can never run — but the
compiler intervenes before this happens:

```
threads × registers_per_thread > 65,536 (total SM registers)
        ↓
Compiler detects at compile time → spills excess to local memory

Register file (fast):  sub-nanosecond, private to thread, on-chip
Local memory (slow):   per-thread private section of HBM,
                       L2-cached but orders of magnitude slower

→ Kernel launches and runs correctly
→ Spilled variables cause HBM traffic on every access
→ Visible in Nsight Compute as local memory load/store traffic
→ Fix: reduce register pressure or use __launch_bounds__

Rare hard-fail case (compiler couldn't resolve):
→ cudaErrorLaunchOutOfResources at runtime
```

**`__launch_bounds__` — help the compiler allocate registers correctly:**

```cuda
__global__ __launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)
void my_kernel(...) { ... }
```

Without this hint, the compiler conservatively assumes 1,024 threads per block and
under-allocates registers per thread to leave room for worst case → more spilling.

With the hint: compiler knows fewer threads will share the register file → can assign
more registers per thread → less spilling → better performance.

---

## Step 5 — Warp Scheduling (the Warp Scheduler)

Each SM has **4 warp schedulers** (on A100). Each scheduler independently picks a
**ready warp** — one with no outstanding memory dependency or barrier — and issues its
next instruction every cycle.

```
SM on A100:
  4 warp schedulers running in parallel
  Each scheduler issues 1 instruction per cycle to its selected warp
  = up to 4 instructions issued simultaneously per SM per cycle
```

All 32 threads in the selected warp execute that instruction in lockstep — **SIMT
(Single Instruction, Multiple Threads)**. Same instruction, different data (different
thread IDs, different memory addresses).

The scheduler maintains a pool of all warps resident on the SM and tracks which are
**ready** vs **stalled**:

```
Ready warp:   all operands available → scheduler can issue next instruction immediately
Stalled warp: waiting on HBM load, barrier, or dependency → scheduler skips it
```

---

## Step 6 — Instruction Issue and Execution

For a selected warp, the scheduler reads the next instruction from the warp's program
counter and issues it to the execution units:

```
Instruction types → execution units:
  FMA (multiply-accumulate) → Tensor Cores or CUDA Cores
  Memory load/store         → LSU (Load Store Unit) → L1 → L2 → HBM
  Integer arithmetic        → INT units
  Special functions (sin, exp, sqrt) → SFU (Special Function Unit)
```

All 32 threads execute simultaneously. The FMA units are 32-wide — one clock cycle
computes 32 multiply-accumulates at once.

---

## Latency Hiding — Why Occupancy Matters

HBM memory access takes **300–400 cycles**. If a warp issues a load and the SM waited
for it, 400 cycles of compute time would be wasted. Instead, the SM **switches to
another ready warp** instantly:

```
Cycle 0:   Warp A issues: load x from HBM → stalled (result arrives in ~400 cycles)
Cycle 1:   Scheduler skips Warp A → switches to Warp B → issues FMA instruction
Cycle 2:   Warp B: FMA → issues next instruction
...
Cycle 400: Warp A's HBM load completes → Warp A becomes ready again
Cycle 401: Scheduler picks Warp A → issues next instruction

Result: SM was doing useful work during all 400 cycles
```

This is **latency hiding** — the SM fills memory latency gaps with compute from other
warps. It only works if there are enough resident warps to cover the latency:

```
HBM latency: ~400 cycles
Warp issues 1 instruction every ~4 cycles (rough average with mix of fast/slow ops)

Warps needed to fully hide latency:
  400 cycles / 4 cycles per instruction = ~100 warps needed

A100 SM max: 64 warps
→ Even at full occupancy, some latency is visible. But 64 >> 16, so more warps = better.
```

**The occupancy-latency hiding connection:**

| Warps resident | Latency hiding | Likely bottleneck |
|---|---|---|
| 64 (max) | Good — many warps to switch between | Memory bandwidth or compute |
| 32 (register-limited) | Moderate — some idle cycles | Occupancy |
| 8–16 (severely limited) | Poor — SM stalls visibly | Occupancy |
| 1–4 (tiny kernel) | None — SM stalls for every load | Occupancy (fix: larger batch) |

---

## Warp Divergence

Within a warp, all 32 threads execute the same instruction. If the kernel has a
conditional branch and different threads take different paths, the hardware cannot split
the warp — it must **serialize the paths**:

```python
# Kernel with branch:
if thread_id % 2 == 0:
    result = compute_A(x)   # threads 0, 2, 4, ... take this path
else:
    result = compute_B(x)   # threads 1, 3, 5, ... take this path
```

**How the hardware handles it:**

```
Pass 1: Execute compute_A with threads [0, 2, 4, ...] active
        Threads [1, 3, 5, ...] are masked off — they consume the cycle, do no work

Pass 2: Execute compute_B with threads [1, 3, 5, ...] active
        Threads [0, 2, 4, ...] are masked off — consume the cycle, do no work

Total cost: 2× the cycles of a non-divergent warp
Wasted work: 50% of thread-cycles in each pass
```

**Divergence is not caused by registers.** It is caused by conditional code where
threads within the same warp take different branches. Registers cause fewer warps per
SM; divergence causes wasted cycles within a warp.

**When divergence is unavoidable:** boundary conditions in tiled algorithms — the last
tile of a matrix may have fewer elements than a full tile, so some threads check `if
(col < N)` and branch. cuBLAS and CUTLASS handle this with predication (compiler
generates masked instructions rather than explicit branches) to reduce the overhead.

**When divergence is bad practice:**
```python
# Bad — alternating threads diverge every warp
if thread_id % 2 == 0: ...

# Also bad — branch depends on runtime data with no spatial locality
if data[thread_id] > threshold: ...   # if data is random, 50% divergence

# Good — branch at warp boundary (no divergence within a warp)
if thread_id >= 32: ...   # all threads in warp 0 take same path
                           # all threads in warp 1 take same path
```

---

## Summary: What Limits Warp Execution

| Limiter | Effect | Caused by | Fix |
|---|---|---|---|
| Registers/thread | Fewer warps per SM → less latency hiding | Complex kernel, many variables | Reduce register use, split kernel |
| Shared memory | Fewer blocks per SM | Large tile sizes | Reduce tile size |
| Block size not multiple of 32 | Wasted thread slots in last warp | Poor launch config | Round up to next multiple of 32 |
| Warp divergence | Wasted cycles within a warp | Conditional branches | Restructure branches to warp boundaries |
| Small grid | Few SMs occupied | Small problem size | Larger batch, larger input |
| Memory latency (HBM) | Warps stall waiting for data | Memory-bound kernel | Fusion, tiling, quantization |

**The fundamental constraint:** the GPU has abundant parallelism but every resource
(registers, shared memory, warp slots) is finite and shared. Occupancy is the measure
of how well the kernel uses what's available. Low occupancy means the hardware is idle
not because it's doing hard work, but because it can't find enough work to do.
