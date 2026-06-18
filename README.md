# Inference Fundamentals

**Goal:** Learn GPU inference from first principles — profiling every optimization step by step to see its real impact.  
**Model:** GPT-2 (117M) — decoder-only transformer, no auth token, fast iteration.  
**Hardware:** NVIDIA A100 SXM4 40GB on Lambda Labs ($1.99/hr — spin up only when actively working).

**Core principle:** PyTorch profiler runs at every phase. Each new technique is measured before and after — you see the delta in the profiler output, not just in theory.

---

## Files

| File | Description |
|------|-------------|
| [setup.md](setup.md) | Lambda Labs A100 setup, Python env, GPT-2 download, tmux, snapshots |
| [run_gpt2_inference.md](run_gpt2_inference.md) | Phase 1 — baseline inference + PyTorch profiler |
| [working_log.md](working_log.md) | Session-by-session log of observations and results |
| [torch_profiler/torch_profiler.md](torch_profiler/torch_profiler.md) | PyTorch profiler reference — how it works, how to read output |
| [knowledge_base/profiler.md](knowledge_base/profiler.md) | Overview of all profiling tools (Kineto, Nsight, HTA, Perfetto) |
| [tensor_pin_memory/script/pinned_memory_transfer.py](tensor_pin_memory/script/pinned_memory_transfer.py) | Pageable vs pinned memory transfer deep dive |
| *(coming)* | Phase 2 — batching (naive → bucketed → dynamic) |
| *(coming)* | Phase 3 — multiprocess serving (FastAPI + SO_REUSEPORT + process-per-GPU) |
| *(coming)* | Phase 4 — warm-up + torch.compile + CUDA graphs |
| *(coming)* | Phase 5 — KV cache implementation |
| *(coming)* | Phase 6 — quantization (INT8, INT4, FP8, KV cache) |
| *(coming)* | Phase 7 — Nsight Systems + Nsight Compute deep dive |
| *(coming)* | Phase 8 — Kubernetes deployment + piecewise CUDA graphs |

---

## Phase Roadmap

Profiling is not a one-time activity — it is the measuring instrument at every phase.
Each phase adds one technique, profiles before and after, and records the delta.

---

### Phase 1 — Baseline Inference + PyTorch Profiler

Run GPT-2 on synthetic prompts with no optimizations. Establish the baseline numbers everything else is measured against.

Profile output answers:
- What is baseline throughput (tokens/sec) and avg latency (ms/request)?
- Which ops dominate CUDA time? (aten::mm, aten::baddbmm, aten::softmax)
- Is GPT-2 compute-bound or memory-bound at batch size 1?
- How much of wall time is the GPU actually doing work vs idle?

→ [run_gpt2_inference.md](run_gpt2_inference.md)

---

### Phase 2 — Request Batching

Add batching and profile the impact at each step.

**Step 1 — Naive batching:** fixed batch size, pad all sequences to the longest in the batch.  
**Step 2 — Bucketed batching:** group requests by sequence length into predefined bins (e.g. 64, 128, 256, 512, 1024). Pad only to bin boundary — reduces wasted compute.  
**Step 3 — Dynamic batching:** collect requests over a short time window, batch whatever arrived. Maximizes GPU utilization without waiting for a full fixed batch.

Profile after each step: throughput increase, GPU idle gaps shrinking, padding waste.

Key insight: bucketing and dynamic batching solve different problems.
- Bucketing controls **shape** → finite compiled graphs, bounded padding waste.
- Dynamic batching controls **batch size** → GPU utilization, latency vs throughput tradeoff.
- Production uses both together.

**CUDA Graph prerequisite exercise (Step 2):**
After implementing bucketed batching, observe that each bucket now produces a fixed
`(batch_size, seq_len)` shape on every call. Verify this is the shape contract CUDA
graphs require:
- Log the `(batch, seq_len)` of every forward call across 100 requests — confirm only
  bucket boundary values appear (64, 128, 256, 512, 1024), never arbitrary lengths.
- Intentionally pass a shape outside a bucket boundary and observe the output corruption
  that would happen if a graph were replayed with the wrong shape (demonstrates *why*
  bucketing is a prerequisite, not just an optimization).

→ Reference: `knowledge_base/CUDA_Graph.md` §3 (shape constraint), §4 (bucketing)

---

### Phase 3 — Multiprocess Serving

Build a real serving layer and implement multiprocess serving from scratch. Goal: understand how CPU and GPU work is split across processes, how the GIL limits single-process throughput, and how SO_REUSEPORT enables true parallelism.

**Step 1 — Single-process FastAPI server (baseline):**
Wrap `model.generate()` in a FastAPI endpoint. One process, one GPU.
- POST `/generate` with prompt → returns generated text + latency
- Measure: requests/sec, p50/p99 latency, CPU utilization (single core), GPU utilization
- Profile with torch.profiler: how much wall time is CPU preprocessing vs GPU compute?

**Step 2 — Observe the GIL bottleneck:**
Add multi-threading (`ThreadPoolExecutor`) to handle concurrent requests.
- Send 10 concurrent requests, measure wall-clock time vs sequential
- Observe: threads don't help for tokenization (CPU-bound) — same wall time as sequential
- CPU monitor shows one core at 100%, rest idle → GIL in action

**Step 3 — Multiprocess with SO_REUSEPORT:**
Spawn N worker processes, each listening on the same port via `SO_REUSEPORT`.
OS kernel distributes incoming connections across them — no application-level router needed.
Each process has its own Python interpreter (own GIL) → true CPU parallelism.
- N=4 workers, each preprocessing independently on its own core
- All 4 share the single GPU (requests serialized at GPU level)
- Measure: CPU utilization across all cores, requests/sec vs single-process
- Observe: preprocessing parallelizes; GPU becomes the serialization point

**Step 4 — Process-per-GPU (if multi-GPU available on Lambda):**
One gRPC/FastAPI process paired with one GPU. Each process owns its GPU exclusively.
- N processes = N GPUs, each process handles a full request end-to-end
- No GPU serialization — each request gets a dedicated GPU
- Measure: per-GPU throughput vs Step 3; observe GPU utilization improves
- Load balancing: SO_REUSEPORT at OS level, or explicit round-robin router process

**Step 5 — gc.freeze() tail latency:**
After warmup (Step 3 or 4), call `gc.freeze()` and reload traffic.
- Compare p99 latency before/after freeze
- Observe: GC pause spikes disappear from the latency distribution

**Step 6 (optional) — Split into gRPC servicer + model worker (LinkedIn architecture):**
Separate CPU preprocessing and GPU inference into distinct process types:
```
Client → [gRPC servicer process]  ← handles HTTP, tokenization, feature encoding
              ↓ gRPC call (protobuf)
         [Model worker process]   ← owns GPU, runs model.generate(), returns logits
              ↓
         [gRPC servicer process]  ← decodes response, returns HTTP to client
```
- N gRPC servicer processes (CPU-only, no GPU) + M model worker processes (GPU-only)
- Scale them independently: more servicers if preprocessing bottlenecks, more workers if GPU bottlenecks
- gRPC between them: binary protobuf serialization, faster than HTTP for internal IPC
- Implement with Python `grpc` library: define `.proto` schema (GenerateRequest, GenerateResponse),
  compile with `protoc`, implement servicer and worker as separate scripts
- Measure: can you saturate the GPU more fully by tuning N servicers vs M workers?
- Compare to Step 3 (FastAPI all-in-one): same multiprocess concept, different IPC and separation of concerns

**Key concepts implemented:**
- `SO_REUSEPORT` — multiple processes bind the same port; OS kernel load-balances
- `multiprocessing` vs `threading` — GIL bypass via separate processes
- Process pinning (`os.sched_setaffinity`) — pin each worker to a dedicated CPU core
- `gc.freeze()` — freeze permanent heap objects to eliminate GC scan pauses
- GPU ownership per process — each worker calls `torch.cuda.set_device(rank)`

**What to measure at each step:**

| Config | CPU cores active | Requests/sec | p99 latency | GPU util |
|---|---|---|---|---|
| Single process | 1 | baseline | baseline | X% |
| 4 threads | 1 (GIL) | ≈ baseline | ≈ baseline | X% |
| 4 processes (SO_REUSEPORT, 1 GPU) | 4 | ~2–3× | lower | higher |
| N processes (1 per GPU) | N | N× | lower | ~same per GPU |
| + gc.freeze() | N | same | p99 drops | same |

→ Reference: `knowledge_base/python_gil.md`

---

### Phase 4 — Warm-up + Compilation + CUDA Graphs

**Warm-up:** before opening to traffic, run N dummy inference passes with each expected bucket shape. Absorbs CUDA kernel JIT compilation, memory allocator warm-up, and torch.compile tracing. Without this, first real requests are 10-100× slower than steady state.

Profile: compare first-request latency vs warm request latency. See the cliff.

**torch.compile:** wraps the model with the Inductor backend. On first traced shape, generates fused CUDA kernels — multiple small ops merged into one kernel launch. Subsequent calls hit the compiled path.

Profile after compile: kernel count drops, CUDA time for hot ops decreases, fewer gaps on GPU track.

Key insight: warm-up and compile interact — warm-up triggers compilation for each bucket shape upfront so no real request ever pays the compilation cost.

**CUDA Graph exercise:**

Step 1 — Capture and replay a single graph:
- Pre-allocate static input/output buffers for one bucket shape (e.g. seq_len=128)
- Capture the full forward pass into a `torch.cuda.CUDAGraph`
- Replay it 1000 times, measure CPU dispatch time vs eager mode
- Expected: CPU overhead collapses from ~44ms/pass to ~5μs/pass

Step 2 — Capture one graph per bucket:
- For each bucket size (64, 128, 256, 512, 1024), capture one graph at startup
- Route incoming requests to the correct graph by padding to nearest bucket
- Profile: verify `cudaLaunchKernel` CPU time in profiler drops to near zero

Step 3 — Verify shape safety:
- Attempt to replay a graph with an input that was NOT pre-allocated in the static buffer
  (allocate a new tensor and pass it) — observe the wrong-address failure
- Confirms why `input.copy_(new_data)` is the correct pattern, not `input = new_data`

Profile signal to watch: `cudaLaunchKernel` Self CPU (was 1.088s in Phase 1 baseline)
should drop dramatically. Kernel count in profiler table should collapse to near 1.

→ Reference: `knowledge_base/CUDA_Graph.md` §2 (capture/replay), §4 (bucketing)

---

### Phase 4 — KV Cache Implementation

Understand and implement KV cache from first principles, profiling the impact at each step.

**Step 1 — Observe the problem (done):**
Profiler shows `aten::cat` as a top CUDA time op with 1.12 GB memory footprint.
Root cause: dynamic KV cache growth via `torch.cat` — allocates new tensor + copies all
existing K/V vectors on every decode step. Cumulative cost is O(seq_len²).

**Step 2 — Study vLLM source:**
- `vllm/attention/backends/` — PagedAttention CUDA kernel
- `vllm/core/block_manager.py` — block table and free block pool management
- `vllm/worker/cache_engine.py` — HBM pool initialization at startup

**Step 3 — Implement fixed pre-allocation:**
Pre-allocate KV cache buffer for `max_seq_len` upfront. Write new K/V in-place at the
current position — no `torch.cat`, no copying.
Profile: `aten::cat` memory footprint drops to zero. Measure throughput improvement.
Downside: memory bubble — HBM reserved for max_seq_len even when actual sequence is short.

**Step 4 — Implement block-based allocation (PagedAttention mechanics):**
Pre-allocate a shared HBM block pool at startup (block_size = 16 tokens).
Each request gets a block table mapping logical → physical block.
Assign blocks from pool on demand — one block per 16 tokens, no contiguous reservation.
Write K/V in-place to current block; claim new block when full.
Profile: `aten::cat` gone + HBM waste bounded to (block_size - 1) tokens per request.
Measure: max concurrent requests vs fixed pre-allocation.

Key references: [knowledge_base/kv_cache.md](knowledge_base/kv_cache.md), [knowledge_base/PagedAttention.md](knowledge_base/PagedAttention.md)

---

### Phase 5 — Quantization

Apply quantization techniques and profile the impact on each.

- **PTQ INT8** via bitsandbytes — `load_in_8bit`; compare throughput and perplexity vs FP32 baseline
- **PTQ INT4** via bitsandbytes — `load_in_4bit`; measure memory vs accuracy tradeoff
- **GPTQ** — weight-only quantization with calibration data; profile CUDA ops before/after
- **AWQ** — activation-aware weight quantization; compare to GPTQ on same prompts
- **FP8** — H100/A100 native FP8; measure roofline shift (memory-bound → compute-bound)
- **KV cache quantization** — INT8 KV cache; measure max batch size increase from VRAM savings

Profile after each: op table changes (aten::mm bytes moved decreases → memory-bound ops speed up), VRAM usage drops, throughput increases.

Key question: at what level does accuracy degrade noticeably vs throughput gain?  
Deliverable: comparison table — method → tokens/sec → perplexity → VRAM.

---

### Phase 6 — Nsight Deep Dive

By Phase 5 the stack is well-optimized. Now use Nsight to understand what's happening at the hardware level.

**Nsight Systems (nsys):** system-wide timeline — CPU threads, GPU kernels, H2D transfers, gaps. Answers: where are the remaining idle periods? What's the kernel distribution?

**Nsight Compute (ncu):** per-kernel hardware metrics — roofline position, SM utilization, memory bandwidth, occupancy. Answers: why is this specific kernel still slow?

Compare roofline position before and after quantization — see the model shift from memory-bound toward compute-bound as INT8/FP8 reduces bytes moved per op.

---

### vllm-tuner Integrations (by Phase)

We studied the [`vllm-tuner`](../Github_Repos/vllm-tuner/) repo (Bayesian optimization for vLLM via Optuna). Most of it is too high-level for our educational goals, but three components map cleanly onto our phases:

| Component | Source file | Where it fits | What it adds |
|---|---|---|---|
| **GPU monitor** | `vllm_tuner/profiling/gpu_collector.py` | Phase 2+ | Real-time NVML metrics — GPU utilization, memory used, power draw, SM/memory clocks — alongside torch.profiler op tables |
| **Async request generator** | `vllm_tuner/benchmarks/request_generator.py` | Phase 2 | AsyncIO pattern for concurrent in-process `model.generate()` calls + latency/throughput tracking |
| **HTML comparison reports** | `vllm_tuner/reporting/html.py` | Phase 5 | Interactive Plotly dashboards for comparing quantization configs (INT4 vs FP8 vs FP32) |

**What we skip and why:**
- **Optuna study manager / optimizer** — Bayesian tuning adds abstraction distance from hardware. We want to see *why* each change moves the needle, not just find optimal values.
- **vLLM server launcher** — we run GPT-2 in-process, not via HTTP server.
- **Pydantic config models** — overkill at our current experiment scale.

**Immediate next action (Phase 2):** write `scripts/gpu_monitor.py` wrapping `pynvml` — collects utilization, memory, power, clocks — and wire it into batch size sweep experiments.

---

### Phase 7 — Containerization + Kubernetes + Piecewise CUDA Graphs

FastAPI inference server → Dockerfile with CUDA base image → Kubernetes Deployment.

Key engineering: readiness probe gates traffic until warm-up completes. Without this, Kubernetes routes requests to a pod that hasn't finished compilation + warm-up yet → first-request latency spike in production.

```
Startup sequence inside pod:
  Model Manager downloads from Artifactory (name + group + version)
      ↓
  torch.compile(model)
      ↓
  Warm-up loop over all bucket shapes
      ↓
  Readiness probe passes → Kubernetes routes traffic
```

**Piecewise CUDA Graph exercise:**

GPT-2 inference has a prompt structure with mixed static/dynamic segments — exactly
what piecewise graphs are designed for:

```
[system prompt (fixed)] | [user query (variable)] | [generation (dynamic)]
      ↑ same every call         ↑ bucketed              ↑ autoregressive
```

Step 1 — Identify stable vs dynamic segments:
- Measure what fraction of compute each segment contributes (torch.profiler with NVTX ranges)
- Confirm system prompt KV is recomputable identically every call

Step 2 — Capture segment graphs separately:
- Graph A: system prompt KV computation (fixed shape, capture once)
- Graph B: per bucket query + scoring (one graph per bucket size)
- Eager: generation loop (autoregressive, shape changes per step — keep eager)

Step 3 — Stitch with stream sync points:
- `graph_A.replay()` → `torch.cuda.synchronize()` → feed Graph A output into Graph B
- `graph_B.replay()` → eager generation loop
- Measure: dispatch overhead on stable segments drops to ~5μs; eager segment unchanged

Step 4 — Compare to Phase 3 full-graph approach:
- Full graph (Phase 3): works only when entire sequence is fixed shape
- Piecewise (Phase 7): handles mixed static/dynamic — stable segments get graph speedup,
  dynamic segments fall back to eager without breaking the stable ones

Profile signal: per-segment CPU time breakdown (use NVTX range labels to separate
system prompt, query, generation in the profiler output).

→ Reference: `knowledge_base/CUDA_Graph.md` §5 (piecewise), §6 (MixLM interaction)

---

## What You Learn (Interview Summary)

| Phase | Core skill | Talking point |
|-------|-----------|---------------|
| 1 | Read profiler output, identify hot ops | "torch.profiler showed 70% of CUDA time in aten::mm — memory-bound at batch size 1" |
| 2 | Batching + shape contract | "Bucketing fixes the shape space — confirmed only bucket boundary values (64,128,256…) appear across 100 requests; this is the prerequisite for CUDA graph capture" |
| 3 | CUDA graphs + warm-up | "Captured one graph per bucket at startup — CPU dispatch overhead collapsed from 44ms to 5μs per forward pass; warm-up triggers capture so no live request pays the compilation cost" |
| 4 | KV cache from scratch | "Replaced aten::cat with pre-allocated buffer — eliminated 1.12 GB cumulative copy cost; then implemented block-based allocation to remove memory bubble" |
| 5 | Quantization hands-on | "AWQ INT4 gave 3x memory reduction with <1% perplexity increase; KV cache INT8 let us double max batch size" |
| 6 | Nsight roofline | "Nsight Compute placed GPT-2 below the memory roofline at batch size 1; FP8 shifted it toward compute-bound" |
| 7 | Piecewise CUDA graphs + Kubernetes | "Split prompt into stable (system prompt) and dynamic (query, generation) segments — captured graphs for stable segments, stitched with stream sync points; readiness probe gates traffic until all bucket graphs are captured and warmed up" |

---

## Cost Estimate (Lambda Labs)

| Phase | Estimated time | Cost |
|-------|---------------|------|
| Setup | 0.5 hr | ~$1 |
| Phase 1 | 2 hr | ~$4 |
| Phase 2 | 2 hr | ~$4 |
| Phase 3 | 2 hr | ~$4 |
| Phase 4 | 3 hr | ~$6 |
| Phase 5 | 3 hr | ~$6 |
| Phase 6 | 3 hr | ~$6 |
| Phase 7 | 2 hr | ~$4 |
| **Total** | **~17.5 hr** | **~$35** |

> Filesystem snapshot preserves your env between sessions (~$0.20/GB/month for storage).
