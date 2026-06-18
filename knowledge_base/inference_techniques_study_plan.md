# LLM Inference Optimization Techniques — Study Plan

Sourced from: vLLM blog, SGLang, TGI, TensorRT-LLM, DeepSpeed-FastGen, research papers.

Novelty ratings:
- **[B]** = well-known, beginner knows it
- **[I]** = intermediate, requires reading papers
- **[A]** = advanced, requires reading source code or blog posts
- **[X]** = expert-level, almost no documentation, you'd miss it without deep diving

Status: [ ] = not started, [~] = in progress, [x] = done

---

## Category 1 — KV Cache Optimizations

| # | Technique | Metric | Novelty | Status |
|---|-----------|--------|---------|--------|
| 1.1 | **PagedAttention** — non-contiguous KV blocks, free pool, block table per sequence | Throughput, Memory | [B] | [x] |
| 1.2 | **Automatic Prefix Caching (APC)** — hash-per-block (not per-tree), LRU with refcount, multi-tenant isolation via cache salt | TTFT, Throughput | [I] | [ ] |
| 1.3 | **Distributed KV Cache Pool** (Mooncake/LMCache) — GPUDirect RDMA between GPU memory regions, 3-level hierarchy (local DRAM / remote DRAM / SSD), MLA deduplication | TTFT (46×), Throughput (3.8×) | [X] | [ ] |
| 1.4 | **KV Cache Offloading to CPU/SSD** — cudaMemcpyAsync on separate stream, block consolidation for DMA efficiency (10×), FlexGen LP placement | Throughput (9×), TTFT | [I/A] | [ ] |
| 1.5 | **KV Cache Quantization (FP8/INT8)** — per-head scales vs. per-tensor, two-level accumulation for long-context accuracy | Memory (54% of BF16) | [A] | [ ] |
| 1.6 | **MLA — Multi-Head Latent Attention** (DeepSeek) — low-rank KV compression, absorption trick at inference time, store only latent `c_KV` | Memory (4–14% of MHA) | [A/X] | [ ] |
| 1.7 | **External KV Cache Daemon** (PegaFlow) — Rust daemon survives vLLM restarts, CUDA IPC + gRPC, GIL-free | Startup latency | [X] | [ ] |
| 1.8 | **Encoder Cache** (Multimodal) — vision embeddings cached by image hash, decouples vision encoder from text prefill scheduling | TTFT, Throughput | [A] | [ ] |
| 1.9 | **Shared Memory IPC Object Store** — ring buffer in shm, pointer broadcast (not data copy), order-independent reader/writer refcount eviction | TTFT (−40.5%), Throughput (+69.9%) | [X] | [~] |

---

## Category 2 — Attention Computation

| # | Technique | Metric | Novelty | Status |
|---|-----------|--------|---------|--------|
| 2.1 | **FlashAttention 1/2/3** — tiled SRAM computation, fused QK^T+softmax+output, FA3 TMA + warp specialization + FP8 | TTFT, ITL, Memory | [B]/[A] | [x] |
| 2.2 | **FlashInfer** — JIT-compiled kernels per workload (batch size, page size, head dim, dtype), paged KV-aware, GQA/MLA/sliding window | ITL (29–69% lower) | [A] | [ ] |
| 2.3 | **FlashMLA** — ETAP (Efficient Transpose Attention Pipeline), warp-based L2 prefetch for concat-K, fused RoPE+Quant+Q-write | TTFT, Throughput | [X] | [ ] |
| 2.4 | **GQA / MQA** — grouped/multi-query attention, KV bandwidth reduced by n_heads/n_kv_heads | Memory, ITL | [I] | [ ] |
| 2.5 | **Triton Attention Backend** — portable paged attention, persistent kernel with fixed SM grid (CUDA graph compatible), Q-side blocking | Cross-platform ITL | [A/X] | [ ] |
| 2.6 | **Context Parallelism / Ring Attention** — Q,K,V partitioned across GPUs, ring KV pass, blockwise online softmax | TTFT for long context | [I/A] | [ ] |
| 2.7 | **DeepSeek V4 NSA** — 4-level hybrid sparse attention (local, c4a, c128a, top-k), inverse RoPE trick, 256-token logical block | TTFT, Memory (8.7× KV reduction) | [X] | [ ] |

---

## Category 3 — Batching and Scheduling

| # | Technique | Metric | Novelty | Status |
|---|-----------|--------|---------|--------|
| 3.1 | **Continuous Batching** (Orca) — iteration-level scheduling, evict finished / admit new every step | Throughput (23×) | [B] | [x] |
| 3.2 | **Chunked Prefill** (Sarathi) — split prefill into chunks, interleave with decode tokens in same forward pass, token budget | ITL stability, TTFT fairness | [I] | [x] |
| 3.3 | **Dynamic SplitFuse** (DeepSpeed) — fuse short prompts to fill fixed token budget, split long prompts across iterations | Throughput | [I] | [ ] |
| 3.4 | **Token Budget + Decode Priority** — decode requests scheduled first (1 token each), remaining budget for prefill chunks | Balanced TTFT/ITL | [A] | [ ] |
| 3.5 | **Multi-Step Scheduling** — schedule multiple decode steps before returning to Python, reduces scheduling overhead per token | Throughput (+28%) | [A] | [ ] |
| 3.6 | **Preemption: Swap vs. Recompute** — decision policy (swap when swap_cost < recompute_cost), all-or-nothing per request | Fairness, Memory util | [A] | [ ] |
| 3.7 | **Cache-Aware Load Balancing** — route requests to replica most likely to have prefix cache hit (hash → instance mapping) | TTFT, Hit rate (1.7%→92%) | [A/X] | [ ] |
| 3.8 | **KV Sharing for Parallel Sampling / Beam Search** — reference-counted physical blocks, copy-on-write on divergence | Memory | [I/A] | [ ] |

---

## Category 4 — Memory Management

| # | Technique | Metric | Novelty | Status |
|---|-----------|--------|---------|--------|
| 4.1 | **Sleep Mode** — offload weights to CPU, preserve CUDA graphs + JIT + allocator; wake = reload weights only | Switch latency (18–200×) | [X] | [ ] |
| 4.2 | **Persistent Batch / Incremental Input Updates** — InputBatch tensor updated in-place with NumPy, copy only diff rows to GPU | Throughput | [A/X] | [ ] |
| 4.3 | **GPU-Side Input Preparation (MRV2)** — Triton kernels build input_ids/positions on GPU, zero CPU-GPU sync, Gumbel-Max sampling | Throughput (+56% on GB200) | [X] | [ ] |
| 4.4 | **Elastic Expert Parallelism** — runtime scale MoE GPU count via API call, EPLB weight shuffle, atomic switch | Cost efficiency | [X] | [ ] |
| 4.5 | **EPLB / Hot Expert Replication** — sliding-window token counts, replicate hot experts, live weight shuffle | Throughput, ITL for MoE | [X] | [ ] |
| 4.6 | **Weight Offloading V2 (NVLink-C2C)** — async weight prefetch on separate CUDA stream, 400 GB/s NVLink-C2C vs 64 GB/s PCIe | Throughput (memory-constrained) | [A/X] | [ ] |

---

## Category 5A — Speculative Decoding

| # | Technique | Metric | Novelty | Status |
|---|-----------|--------|---------|--------|
| 5.1 | **Classic Spec Decoding** — small draft proposes K tokens, large model verifies in one pass, rejection sampling preserves distribution | ITL (2–3×) | [B/I] | [ ] |
| 5.2 | **EAGLE / EAGLE-2 / EAGLE-3** — single layer draft sharing embedding+LM head with target, last hidden state as input, dynamic tree | ITL (1.5–3×) | [I/A] | [ ] |
| 5.3 | **P-EAGLE** — all K draft positions in ONE forward pass using learnable placeholder tokens | ITL (+1.69× over EAGLE-3) | [X] | [ ] |
| 5.4 | **SpecInfer — Token Tree Verification** — multiple draft models generate token tree, all verified in one forward pass with tree attention mask | ITL (1.5–2.8×) | [A] | [ ] |
| 5.5 | **DFlash Spec Decoding** — 5-layer 0.6B draft predicts K tokens in single parallel forward pass (not autoregressive) | ITL (2–3×) | [A] | [ ] |
| 5.6 | **Prompt Lookup Decoding** — n-gram matching against prompt for zero-cost draft proposals | ITL (prompt-overlap tasks) | [I] | [ ] |
| 5.7 | **Medusa** — K MLP heads on last hidden state, each predicts position +i, tree attention verification | ITL (2.2–3.6×) | [I/A] | [ ] |

---

## Category 5B — Quantization

| # | Technique | Metric | Novelty | Status |
|---|-----------|--------|---------|--------|
| 5.8 | **FP8 W8A8** — per-channel weight scales, per-token activation scales, H100 tensor cores 2× vs BF16 | Throughput, Compute | [I] | [ ] |
| 5.9 | **AWQ** — activation-aware: scale salient channels before INT4 quantization, per-group dequant at runtime, W4A16 | Memory (4×) | [I] | [ ] |
| 5.10 | **GPTQ** — second-order (Hessian) optimal INT4 weights, layer-wise minimization, W4A16 | Memory, Throughput | [I] | [ ] |
| 5.11 | **NVFP4 / MXFP4** — 4-bit float with per-group FP8 scales, fused dequant in tensor core block, MoE dispatch volume −4× | Throughput (Blackwell) | [A/X] | [ ] |
| 5.12 | **GGUF** (llama.cpp/Ollama) — hybrid per-layer bit depth (Q4_K_M, Q8_0), k-quant block float, partial GPU offload | Memory, CPU/Apple Silicon | [I/A] | [ ] |
| 5.13 | **SmoothQuant** — migrate quantization difficulty from activations to weights via per-channel smooth factor, W8A8 | Accuracy under INT8 | [I] | [ ] |

---

## Category 5C — Architecture Serving Tricks

| # | Technique | Metric | Novelty | Status |
|---|-----------|--------|---------|--------|
| 5.14 | **Multi-LoRA Serving (S-LoRA)** — paged adapter weights, batched GEMM applies different LoRA per request in same batch | Throughput (4×), Adapter count | [A] | [ ] |
| 5.15 | **Structured Output / XGrammar** — PDA (not FSM) compiled to C++, valid-token bit-mask per step, mask pre-computed off critical path | Correctness, TTFT overhead | [I/A] | [ ] |

---

## Category 6 — Communication and IPC

| # | Technique | Metric | Novelty | Status |
|---|-----------|--------|---------|--------|
| 6.1 | **ZMQ API Server / Engine Decoupling** — separate processes for HTTP handling and inference loop, GIL contention eliminated | Throughput (2.7×) | [A] | [x] |
| 6.2 | **Tensor Parallelism + NCCL AllReduce** — ring/tree AllReduce for partial matmul results, NVLink 600 GB/s | ITL | [B/I] | [ ] |
| 6.3 | **Fused AllReduce + RMSNorm** — torch.compile fuses collective directly into norm kernel, no HBM round-trip between them | Throughput (+15%) | [X] | [ ] |
| 6.4 | **Async Tensor Parallelism** — pipelined AllGather-compute-ReduceScatter per layer | Throughput (+10%) | [X] | [ ] |
| 6.5 | **DeepEP All-to-All Kernels** — custom kernels for MoE token dispatch, NVFP4 reduces volume 4× | Throughput for MoE | [X] | [ ] |
| 6.6 | **Dual-Batch Overlap (DBO)** — two microbatches interleaved so MoE collective for A overlaps attention compute for B | Throughput (high-EP MoE) | [X] | [ ] |
| 6.7 | **RDMA KV Transfer (NIXL)** — GPU-GPU RDMA bypassing CPU, write-mode pushes KV layer-by-layer during prefill (not after) | TTFT overhead | [A/X] | [ ] |
| 6.8 | **Stream Interval Buffering** — buffer N tokens before SSE send (except first), reduces socket syscalls | Throughput (+57%) | [A/X] | [ ] |
| 6.9 | **ShmRingBuffer** (our implementation) — slot-indexed ring buffer in POSIX shm, ZMQ PUSH sends 1-byte slot index as wake signal | IPC latency | [X] | [~] |

---

## Category 7 — Hardware-Level / Kernel Optimizations

| # | Technique | Metric | Novelty | Status |
|---|-----------|--------|---------|--------|
| 7.1 | **CUDA Graph Capture + Replay** — record GPU DAG per batch size, replay skips all CPU kernel-launch overhead | ITL at small batch | [I/A] | [ ] |
| 7.2 | **Piecewise CUDA Graphs** — graph-safe subgraphs captured, graph-unsafe ops (dynamic control flow) run eager, TorchDynamo break analysis | Latency | [A/X] | [ ] |
| 7.3 | **torch.compile Integration** — TorchDynamo + TorchInductor, specific fusions: SiLU+Quant, AllReduce+Norm, Attn+Quant, async TP | Throughput (1.8–2× geomean) | [A] | [ ] |
| 7.4 | **Fused MoE Kernel** — single kernel for SiLU(gate) × activation × quantize_scale, includes expert dispatch/combine | Throughput | [A] | [ ] |
| 7.5 | **Persistent Kernel Fixed Grid** — fixed SM-count launch grid enables CUDA graph for attention kernels with variable-length sequences | Enables CUDA graphs | [X] | [ ] |
| 7.6 | **Gumbel-Max Sampling Kernel** — argmax(logits + Gumbel noise) = categorical sample, avoids materializing full softmax | ITL (sampling) | [X] | [ ] |
| 7.7 | **Concat-K Optimization (MLA)** — warp-based 128-bit vectorized access + L2 prefetch for MLA key concatenation | TTFT/ITL for MLA | [X] | [ ] |

---

## Category 8 — Serving Infrastructure

| # | Technique | Metric | Novelty | Status |
|---|-----------|--------|---------|--------|
| 8.1 | **Prefill-Decode Disaggregation** — separate GPU pools, KV transferred via RDMA, write-mode pushes layers during prefill | ITL stability, Goodput (2.5×) | [I/A] | [ ] |
| 8.2 | **Encoder-Prefill-Decode (EPD) Three-Way Split** — separate GPU pools for vision encoder, text prefill, and decode | Throughput (2.5× multimodal) | [X] | [ ] |
| 8.3 | **SSM Disaggregation** — dual NIXL descriptors for same memory, DS layout for contiguous TP slicing of SSM state | TTFT/ITL for hybrid models | [X] | [ ] |
| 8.4 | **Wave Coordination (DP+EP)** — dummy steps keep DP replicas in lockstep for MoE expert collectives | Enables multi-node MoE | [X] | [ ] |
| 8.5 | **Async Output Processing** — detokenization for step N runs concurrently with GPU compute for step N+1 | Throughput (+8.7%) | [A] | [ ] |
| 8.6 | **Streaming Input / Anchor Request** — KV blocks pinned during live session, new chunks extend prompt without recompute | TTFT for streaming input | [X] | [ ] |
| 8.7 | **Pipeline Parallelism Bubble Avoidance** — token throttling per stage, micro-batch interleaving, chunked prefill for variance reduction | Throughput | [A] | [ ] |
| 8.8 | **Multi-Process Engine (EngineCore isolation)** — scheduling+KV+workers in one process, HTTP+tokenize+detokenize in another | Latency tail, Throughput | [A] | [x] |
| 8.9 | **Object Caching / GC Avoidance** — reuse Python dicts/lists/tensors across steps, pre-allocate fixed containers, Python GC is measurable | Throughput (+24%) | [A] | [ ] |
| 8.10 | **KV-Aware Load Balancer** — prefix hash → instance routing, P/D-aware routing for prefill-heavy requests | Cache hit rate, TTFT | [A] | [ ] |
| 8.11 | **torch.compile Compilation Caching** — compiled artifacts cached to disk, symbolic shapes compile once for range of batch sizes | Startup latency | [A] | [ ] |

---

## Study Priority

**Tier 1 — Foundations (do first):**
PagedAttention (1.1), Continuous Batching (3.1), FlashAttention (2.1), Classic Spec Decoding (5.1),
CUDA Graphs (7.1), Tensor Parallelism + AllReduce (6.2), GQA/MQA (2.4)

**Tier 2 — Production realities:**
Prefix Caching APC (1.2), Chunked Prefill (3.2), FP8 KV Quant (1.5), Async Output Processing (8.5),
Persistent Batch / NumPy diff (4.2), P/D Disaggregation (8.1), Elastic EP + EPLB (4.4/4.5),
Multi-Process Engine (8.8), AWQ/GPTQ/FP8 (5.8–5.10)

**Tier 3 — Expert / cutting-edge (after Tier 2):**
SHM IPC Object Store (1.9), GPU-Side Input Prep MRV2 (4.3), P-EAGLE (5.3), DBO (6.6),
PegaFlow External KV Daemon (1.7), Sleep Mode (4.1), Fused AllReduce+Norm (6.3),
Async TP (6.4), NVFP4 for MoE (5.11), Stream Interval Buffering (6.8),
EPD Three-Way Split (8.2), Wave Coordination DP (8.4), NSA Sparse Attention (2.7),
Weight Offloading NVLink-C2C (4.6), MLA + FlashMLA (1.6/2.3)

---

*Sources: vLLM blog (vllm.ai/blog), SGLang, TGI, TensorRT-LLM, DeepSpeed-FastGen,
FlashAttention, PagedAttention, Sarathi-Serve, SpecInfer, EAGLE-3, DeepSeek V2/V3/V4 technical reports.*
