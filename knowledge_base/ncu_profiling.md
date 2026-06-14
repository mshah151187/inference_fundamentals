# Nsight Compute — Report Column Reference

How to read the summary table in the Nsight Compute GUI. Each row in the table is one
captured kernel invocation. The columns tell you what that kernel did and where it was slow.

---

## Summary Table Columns

### # (Row Index)
Sequential counter of captured kernel invocations, in launch order.
Not a kernel ID — just the position in this report. If you captured 10 invocations of
`gemv2T_kernel_val`, they appear as rows 0–9.

---

### Estimated Speedup %
ncu's estimate of how much faster this kernel *could* run if the top bottleneck is fixed.
Computed from the bottleneck metric — e.g., if SM occupancy is 7.8% vs theoretical 100%,
the speedup estimate is ~92%. Higher % = more headroom = more broken kernel.

This is ncu's way of ranking which kernels most need attention. It does NOT mean the fix
is easy — it means the hardware is underutilized by that much.

```
Est. Speedup 99.61% → kernel using ~0.4% of available hardware → severely underutilized
Est. Speedup 10%    → kernel fairly well tuned, small room left
```

---

### Function Name
Short (possibly truncated) kernel name as it appears in the binary. For CUTLASS template
kernels this is usually mangled C++ — unreadable. Use Demangled Name instead.

---

### Demangled Name
Full C++ demangled template kernel name. For cuBLAS kernels this is human-readable
(`gemv2T_kernel_val`, `gemvNSP_kernel`). For CUTLASS kernels it's a long template
instantiation (`std::enable_if<...>`) — the template parameters encode tile sizes,
data types, and pipeline stages chosen at compile time.

---

### Duration (μs)
Actual GPU wall-clock time this kernel invocation took to execute. This is the GPU-side
time only — does not include CPU dispatch overhead or time waiting in the CUDA stream queue.

```
Duration 1.49 μs  → tiny kernel, very fast
Duration 13.1 μs  → gemv2T at batch=1 (from nsys findings)
Duration 22.3 μs  → gemvNSP at batch=1 for larger MLP weights
```

Compare against CPU dispatch time (~21 μs from nsys) to see whether GPU or CPU is the bottleneck.

---

### Runtime Improvement (μs)
Estimated time saved per invocation if the speedup is realized:
```
Runtime Improvement = Duration × (1 - 1 / speedup_factor)
```
Multiply by number of invocations to get total time saved across the run. Helps prioritize
which kernel is worth optimizing — a 50% speedup on a 1 μs kernel saves less than a 5%
speedup on a 1 ms kernel.

---

### Compute Throughput % (SM Throughput)
What fraction of the GPU's peak SM (Streaming Multiprocessor) compute capacity this kernel
used, averaged over its duration.

```
~0%      → kernel barely touching compute — either occupancy-limited or memory-bound
~50%     → moderate compute utilization
~100%    → fully compute-bound — tensor cores saturated
```

At batch=1, expect < 5% for gemv kernels. The matrix is too small to fill 108 SMs.

**Full metric name:** `sm__throughput.avg.pct_of_peak_sustained_elapsed`

---

### Memory Throughput % (HBM Bandwidth)
What fraction of the GPU's peak HBM memory bandwidth this kernel used.

```
~0–10%   → barely reading from HBM — either very small data or compute-bound
~50–80%  → approaching memory bandwidth ceiling
~100%    → fully memory-bound — HBM is the bottleneck
```

At batch=1 for gemv, expect 10–30% — the weight matrices are small enough that even a
memory-bound kernel doesn't saturate HBM bandwidth. This is occupancy-limited territory:
neither ceiling is hit.

**Full metric name:** `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`

---

### Registers / Thread
Number of GPU registers allocated per thread for this kernel. Registers are a **shared
resource per SM** — more registers per thread → fewer threads can co-reside on the SM →
lower occupancy.

```
16 registers/thread  → low register pressure → can fit many warps per SM
64 registers/thread  → high register pressure → limits warps per SM → lower occupancy
255 registers/thread → maximum, severely limits occupancy
```

Register pressure is one of the three main occupancy limiters (others: shared memory
usage, block size). ncu's occupancy section shows which one is the binding constraint.

---

### Grid Dim (Grid Size)
The 3D grid of thread blocks launched for this kernel: `(X, Y, Z)`.
Total thread blocks = X × Y × Z. Each block runs on one SM (blocks can share an SM if
resources allow, but one SM can't run blocks from different kernels simultaneously).

```
Grid (1, 1, 1)       → 1 block total → only 1 SM has any work → 107 SMs idle on A100
Grid (4281, 1, 1)    → 4281 blocks → can fill all 108 SMs with ~39 blocks each (good)
Grid (X, Y, 1)       → 2D grid → typical GEMM (tiles in row and column dimensions)
Grid (X, 1, 1)       → 1D grid → typical GEMV (vector output, one dimension)
```

**1D grid = GEMV signal.** cuBLAS dispatches GEMV (not GEMM) when one matrix dimension
is 1, which happens at batch=1 decode steps. GEMM uses a 2D grid.

---

### Block Size (Thread Block Dim)
Threads per block: `(X, Y, Z)`. Total threads per block = X × Y × Z.
The GPU schedules threads in groups of 32 called warps. Block size should be a multiple
of 32 to avoid partial (wasted) warps.

```
Block (128, 1, 1)  → 128 threads = 4 warps per block
Block (256, 1, 1)  → 256 threads = 8 warps per block
Block (32, 1, 1)   → 32 threads = 1 warp per block (minimal — often low occupancy)
```

Block size × Grid size = total threads launched. For a 768×768 weight matrix GEMV:
total output elements = 768 → needs ~768 threads → small grid, few blocks, few SMs used.

---

### Result Type
Label for the type of profiling result. `Kernel: SMT` means this is a standard GPU kernel
result (SMT = the profiling mode used). Other types you may see: `Kernel: Replay` (when
ncu had to re-run the kernel multiple times to collect all counters), `API` (for CUDA API
calls, not kernels).

---

## How to Read a Row Together

Example from the screenshot — `vectorized_elementwise_kernel`:

```
Est. Speedup: 99.61%   → kernel almost entirely wasted hardware
Duration:     1.49 μs  → tiny kernel
Compute %:    0.01%    → no compute happening
Memory %:     1.44%    → barely reading HBM
Grid:         (1,1,1)  → 1 block → 1 SM used out of 72
Block:        128      → 4 warps
Registers:    16       → low register pressure (not the occupancy limiter here)
```

**Reading:** This kernel launched 1 block (1 SM), does almost no compute, barely touches
memory, and takes 1.5 μs. 71 SMs sit idle. The 99.61% speedup estimate is because the
kernel could theoretically use all 72 SMs — but that would require a larger problem size,
not a code change. This is an inherently tiny utility kernel; the speedup estimate is
misleading here.

**The key insight:** high Estimated Speedup on a tiny Duration kernel is usually not
worth chasing. Sort by `Runtime Improvement (μs)` not `Estimated Speedup %` to find
kernels actually worth optimizing.

---

## Details Page — 

### Lingo

Elapsed Cycle: 

The total number of GPU clock cycles that passed from the moment the kernel started executing on a given SM until it finished on that SM. It counts all cycles – work‑doing cycles plus cycles when the SM was idle, stalled, or waiting for other resources

NCU multiplies the kernel run‑time (in nanoseconds) by the SM clock frequency (SM Frequency). 
Elapsed Cycles = Duration (ns) × SM Frequency (Hz) / 1 e9

Active Cycle:

Sub‑set of elapsed cycles during which the indicated unit actually performed work (issued at least one instruction, transferred data, etc.

Counted by hardware event counters that fire only when the unit is busy.

### GPU Speed of Light section

This section computes theoretically maximum limit of hardware vs observed statistics. The results are in %.
What % of maximum of a metric you achieved in your run?

### Speed of Light: L1/TEX Cache Throughput %

#### What it measures

L1/TEX Cache Throughput % is bytes transferred **through the L1 pipeline** per active
L1 cycle, expressed as a percentage of L1's peak sustained bandwidth.

```
L1/TEX Throughput % = (bytes through L1 pipeline per active cycle) / L1 peak bandwidth
```

Key: it measures L1 **pipeline activity**, not whether data was served FROM L1.

---

#### The critical distinction — always pair with L1 Hit Rate

A miss still travels through the L1 pipeline:

```
Hit path:   Warp → L1 (hit)  → data served from L1 cache → Warp
Miss path:  Warp → L1 (miss detected) → L2/HBM → data returns THROUGH L1 → Warp
                                                   ↑ L1 pipeline active here too
                                                   ↑ L1 gets populated for future requests
```

Because misses also activate the L1 pipeline, L1 Throughput % can be high even when
the cache is serving nothing useful. **Never read L1 Throughput % alone:**

```
L1 Throughput % high + L1 Hit Rate high  → data served FROM L1 (fast, ideal)

L1 Throughput % high + L1 Hit Rate ~0%  → L1 pipeline busy passing misses through
                                           actual data came from L2 or HBM
                                           high throughput % is misleading here
```

---

#### Why throughput % can be high even for a tiny kernel

NCU computes L1 Throughput % over **active L1 cycles only** — not total kernel duration:

```
vectorized_elementwise_kernel example:
  Total kernel duration:  3,117 elapsed cycles
  SM active cycles:       15.86 cycles  (L1 active for only ~0.5% of kernel time)
  L1 Throughput %:        75%           (during those 15.86 cycles, pipe ran at 75% rate)
```

The 75% reflects intense but brief bursts. Overall contribution to performance is
negligible — the kernel barely ran. This is why L1 Throughput % must be read alongside
total kernel duration and DRAM throughput % for full context.

---

#### Reading all memory metrics together

```
Metric                    What it tells you
──────────────────────────────────────────────────────────────────
L1/TEX Throughput %       How hard the L1 pipeline worked during active cycles
L1 Hit Rate %             What fraction of requests were actually served from L1
L2 Throughput %           How hard L2 worked — data came from here on L1 misses
DRAM Throughput %         How hard HBM worked — data came from here on L2 misses
```

Example interpretation (vectorized_elementwise_kernel):
```
L1 Throughput:  75%   L1 Hit Rate: 0%   → L1 pipeline active but serving no hits
L2 Throughput:  1.48%                   → actual data came from L2
DRAM Throughput: 0.47%                  → almost nothing went to HBM
SM Active Cycles: 15.86 / 3117 total   → L1 was active for 0.5% of kernel time

Conclusion: all data came from L2 (L1 miss → L2 hit). Kernel is too small to matter —
1 block, 4 warps, 71 SMs idle. L1 metric alone would have been misleading.
```

---

#### The Three Bottleneck Regimes

After reading the columns, place the kernel in one of three regimes:

```
Compute %  high, Memory % low   → compute-bound   → tensor cores are the ceiling
                                                      fix: reduce FLOPs (quantization, pruning)

Compute % low, Memory % high    → memory-bound     → HBM bandwidth is the ceiling
                                                      fix: reduce data movement (fusion, tiling)

Compute % low, Memory % low     → occupancy-limited → neither ceiling hit
                                                      fix: larger batch (more tiles per SM)
                                                      or kernel fusion (fewer launches)
```

At batch=1, gemv kernels are expected to be **occupancy-limited**: SM throughput < 5%,
HBM bandwidth 10–30%, because the matrices are too small to fill the GPU's SMs.


### PM Sampling Metrics

### Lingo

PM: Stands for Performance‑Monitor. These are the hardware counters inside the GPU that can be sampled periodically (e.g., every 1 µs or every 10 k cycles).

PM Metric: A metric whose value is obtained from PM sampling. The tool records a series of instances – each instance consists of a timestamp (ns) and the counter value at that moment.

Instanced vs Aggregated: 
Instanced value – the raw sample series, plotted on the timeline view.
Aggregate value – the non‑instanced entry shown in the regular tables (sum, average, etc., depending on the metric).

### How many PM Metrics exist?

The exact count depends on the GPU architecture and the version of Nsight Compute, because each new generation adds or removes counters.
You can obtain the definitive list for your system with the CLI command:

**ncu --query-metrics-collection pmsampling**

---

### Warp Stall Reasons

Nsight Compute classifies every cycle a warp cannot issue an instruction into a stall reason.
The tool samples the hardware stall-status registers and aggregates time spent in each category.
Warp Stall Reasons appear in the PM Sampling section as a bar chart — the dominant bar tells you what to fix.

| Stall Reason | What blocks the warp | Typical symptom | Common ways to reduce it |
|---|---|---|---|
| **IMC Miss** | Load from constant memory not present in the constant cache | Long stall cycles/warp (e.g. ~41 cycles), high % in Warp Stall Sampling table | Make constant-memory accesses uniform within a warp (all threads read the same address). Cache hot data in shared memory or registers. |
| **L1 Miss / L1 Throttling** | Load/store misses L1/TEX and must go to L2/DRAM | Low L1 Hit Rate, high L2 Hit Rate, stall entry named L1 Miss | Coalesce global memory accesses. Increase reuse so lines stay in L1. |
| **L2 Miss / L2 Throttling** | Load/store misses L2 and must go to DRAM | Low L2 Hit Rate, high DRAM Busy, stall reason L2 Miss | Improve spatial/temporal locality. Use shared memory to buffer data. |
| **DRAM Latency** | Accesses reach device memory and incur long round-trip latency | High DRAM Active Cycles, stall reason DRAM Latency | Reduce number of global memory accesses. Prefetch data or overlap memory ops with computation. |
| **Instruction Dependency** | Warp's next instruction depends on result of a previous instruction that has not completed | Stalls labeled Inst Dep or Dep-Reg | Re-order independent work to hide latency. Use more warps (higher occupancy) so another warp can issue while one is waiting. |
| **Memory Dependency** | Load/store must wait for a previous memory operation (e.g. store to same address) | Stall reason Mem Dep | Avoid read-after-write hazards. Split into separate kernels if possible. |
| **Barrier / Sync** | Threads in a block hit `__syncthreads()` and must wait for others | Stall reason Sync | Reduce unnecessary barriers. Balance work so all warps in a block reach the barrier together. |
| **Divergent Branch** | Threads in a warp take different execution paths — warp must serialize the paths | Stall reason Divergence | Structure code so branches are taken uniformly. Move divergent work to separate kernels. |
| **SMEM Bank Conflict** | Shared-memory accesses hit the same memory bank within a half-warp | Stall reason SMEM Bank Conflict | Pad shared-memory arrays or use warp-shuffle instructions. |
| **Texture Cache Miss** | Texture fetch misses texture cache and goes to L2/DRAM | Stall reason Tex Miss | Use 2D/3D spatial locality. Bind textures with appropriate format. |
| **Warp Scheduler Stall** | No warp on the scheduler is eligible to issue (all waiting on stalls) | High No Eligible % in Scheduler Statistics, Issue Slot Utilization warning | Increase number of resident warps (raise occupancy, reduce register pressure). Eliminate long stalls (e.g. constant-cache misses). |

---

## Compute Workload Analysis — Pipe Utilization

This section shows how hard the SM's compute pipelines worked during the kernel.
It answers: *which pipeline is the bottleneck, and how close to peak are we?*

### Three Headline Metrics

| Metric | What it means | `vectorized_elementwise_kernel` value |
|---|---|---|
| **SM Busy %** | % of elapsed cycles the SM was active (issuing any instruction) | 0.01% — SM nearly always idle |
| **Executed IPC** | Instructions issued per clock cycle (peak = 4 on A100, one per warp scheduler) | 0.04 — 1% of peak throughput |
| **Issue Slots Busy %** | % of scheduler issue slots that dispatched an instruction | 0.97% — 99% of slots empty |

**IPC context:** A100 has 4 warp schedulers per SM → theoretical peak = 4 instructions/cycle.
Executed IPC = 0.04 means the SM issued instructions for only 1% of its capacity.

### Pipeline Utilization Charts

Two chart pairs are shown — one for **Aggregate Pipes** (logical), one for **Physical Pipes**:

```
Aggregate Pipes (logical grouping):
  ALU     — integer and logic operations
  FMA     — float multiply-add (CUDA cores, FP32/FP16)
  Tensor  — tensor core matrix multiply operations

Physical Pipes (hardware execution units):
  FMA Heavy / FMA Lite  — different FP throughput tiers
  ADU                   — address calculation unit
  LSU                   — load/store unit
```

Each pipeline is shown in two %:
```
% of active cycles              → how often this pipeline fired during SM-active cycles
% of peak instructions executed → instructions issued vs theoretical peak throughput
  over active cycles
```

Both metrics near zero = pipeline barely used, consistent answer from two angles.

### NCU Warning

> "All compute pipelines are under-utilized. Either this workload is very small
> or it doesn't issue enough warps per scheduler. Check the Occupancy and
> Scheduler Statistics sections for further details."

This warning fires when SM Busy and IPC are both very low. It points you to
Occupancy (are enough warps resident?) and Scheduler Statistics (are resident
warps actually issuing instructions?).

### Reading Compute Workload Analysis Together

```
SM Busy high, IPC high, FMA pipeline ~100%   → compute-bound (CUDA cores saturated)
SM Busy high, IPC high, Tensor pipeline ~100% → compute-bound (Tensor cores saturated)
SM Busy low,  IPC low,  all pipelines ~0%    → occupancy-limited (not enough warps)
SM Busy high, IPC low,  all pipelines low    → memory-bound (warps stalled on memory,
                                               SM active but not issuing instructions)
```

`vectorized_elementwise_kernel`: all pipelines ~0%, SM Busy 0.01% → occupancy-limited.
1 block, 4 warps, 107 SMs idle — the problem is not enough work, not the pipeline.