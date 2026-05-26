"""
Topic: CPU → GPU Memory Transfer — Pageable vs Pinned Memory
=============================================================

When you call .to("cuda"), PyTorch must copy tensor data from CPU RAM into GPU VRAM.
The speed of this transfer depends on whether the source memory is pageable or pinned.

--- SLOW PATH: Pageable Memory (default) ---

  CPU RAM (pageable)  →  pinned staging buffer  →  GPU VRAM
       (OS-managed)         (CUDA-managed)          (HBM)
          Step 1                Step 2

  Step 1: CUDA driver allocates a temporary page-locked buffer internally and
          copies your tensor data into it. This is done because DMA (Direct Memory
          Access) hardware can only read from page-locked memory — it cannot safely
          read from pageable memory because the OS might move that page at any moment.
  Step 2: DMA engine transfers from the staging buffer to GPU VRAM.

  Downside: two copies, and the .to("cuda") call BLOCKS the CPU until the transfer
  is fully complete — CPU and GPU cannot overlap work.

--- FAST PATH: Pinned (Page-Locked) Memory ---

  CPU RAM (pinned)  →  GPU VRAM
   (page-locked)       (HBM)
      Single DMA hop

  .pin_memory() tells the OS: "lock this memory page — never swap it to disk."
  Because it can never move, the DMA engine can transfer directly to GPU VRAM
  without a staging copy. One hop instead of two.

  With non_blocking=True: the transfer is ASYNCHRONOUS — .to("cuda") returns
  immediately, CPU continues to the next line, while the DMA engine transfers
  in the background. This lets you overlap CPU work (tokenizing next prompt)
  with GPU transfer of the current prompt.

--- WHY IT MATTERS FOR INFERENCE THROUGHPUT ---

  Pageable:  CPU blocks during transfer → sequential: tokenize → wait → inference
  Pinned:    CPU returns immediately    → overlapped: tokenize next while transferring current

  For short prompts (few tokens), the H2D transfer time can be comparable to
  the GPU compute time. At batch size 1, pinned + non_blocking is the main tool
  to reduce the CPU-side bottleneck.

--- CAVEAT ---

  Pinned memory is scarce — the OS cannot reclaim it for other processes.
  Pinning too much RAM can starve other processes or cause OOM on CPU.
  Use it for tensors you transfer repeatedly (model inputs), not everything.

--- HOW THIS APPEARS IN PyTorch PROFILER ---

  Pageable transfer: shows as a synchronous cudaMemcpyH2D event. CPU time ≈ transfer time.
  Pinned transfer:   cudaMemcpyAsync — shows as a short CPU event, long async GPU event.
  In the Perfetto trace (ui.perfetto.dev): pinned transfers leave a GPU-side bar that
  overlaps with CPU activity on the next prompt, while pageable transfers leave a gap.

Run: python3 pinned_memory_transfer.py
(Runs on CPU only for illustration — no GPU required. On a GPU machine the timings
 at the bottom will show the actual speedup.)
"""

import torch
import time

# ─── SLOW PATH: pageable tensor ──────────────────────────────────────────────

# torch.randn() allocates memory through Python/PyTorch's normal allocator.
# The OS manages this memory as "pageable" — it can swap pages to disk under
# memory pressure, and the physical address of each page can change at any time.
# Shape (10000, 10000) = 100M floats × 4 bytes = ~400MB — large enough to see
# transfer time differences.
pageable_tensor = torch.randn(10_000, 10_000)

# tensor.is_pinned() returns False — this is regular pageable CPU memory.
print(f"pageable_tensor.is_pinned() : {pageable_tensor.is_pinned()}")
# Expected: False

# ─── FAST PATH: pinned tensor ────────────────────────────────────────────────

# .pin_memory() calls mlock() (Linux) / VirtualLock() (Windows) under the hood.
# This tells the OS kernel: "do not page this memory out to disk, keep it
# physically resident in RAM at a fixed address."
# The CUDA DMA engine requires fixed physical addresses — pinned memory
# satisfies this, pageable memory does not.
pinned_tensor = torch.randn(10_000, 10_000).pin_memory()

# Now is_pinned() returns True — this tensor lives in page-locked CPU RAM.
print(f"pinned_tensor.is_pinned()   : {pinned_tensor.is_pinned()}")
# Expected: True

print()

# ─── Transfer comparison (only meaningful on a real GPU machine) ──────────────

if torch.cuda.is_available():

    # --- Pageable transfer (blocking) ---
    # .to("cuda") with a pageable tensor:
    #   1. CUDA runtime internally allocates a temporary pinned staging buffer
    #   2. Copies pageable_tensor → staging buffer (CPU-side copy)
    #   3. DMA: staging buffer → GPU VRAM
    #   4. Frees staging buffer
    # This call BLOCKS until step 4 is complete — the CPU sits idle waiting.
    torch.cuda.synchronize()   # flush any prior GPU work so timing is clean
    t0 = time.perf_counter()
    gpu_tensor_slow = pageable_tensor.to("cuda")
    torch.cuda.synchronize()   # wait for GPU to finish before stopping the clock
    slow_ms = (time.perf_counter() - t0) * 1000
    print(f"Pageable → GPU transfer : {slow_ms:.2f} ms  (blocking, 2-hop)")

    # --- Pinned transfer (async, non-blocking) ---
    # non_blocking=True is only meaningful with pinned memory.
    # With pageable memory, non_blocking=True is silently ignored — CUDA cannot
    # do an async DMA from pageable memory because the page address might change.
    #
    # With pinned memory + non_blocking=True:
    #   1. DMA engine is told: "copy pinned_tensor → GPU VRAM" (single hop)
    #   2. .to("cuda", non_blocking=True) returns IMMEDIATELY to the CPU
    #   3. The DMA transfer happens concurrently on the CUDA copy engine
    #   4. torch.cuda.synchronize() is needed before you USE the tensor on GPU —
    #      it waits for the async transfer to actually complete
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu_tensor_fast = pinned_tensor.to("cuda", non_blocking=True)
    
    # At this point the CPU is back — transfer may still be in flight.
    # We call synchronize() only to measure the true wall time for comparison.
    # In real code you would NOT synchronize here — you would do CPU work instead
    # (e.g. tokenize the next prompt), then use gpu_tensor_fast later after sync.
    torch.cuda.synchronize()
    fast_ms = (time.perf_counter() - t0) * 1000
    print(f"Pinned   → GPU transfer : {fast_ms:.2f} ms  (async, 1-hop)")

    print(f"\nSpeedup: {slow_ms / fast_ms:.2f}x")
    # On A100 40GB with a 400MB tensor, typical: pageable ~8ms, pinned ~4ms → ~2x

else:
    print("No GPU available — transfer timing skipped.")
    print("On a GPU machine you would see ~2x speedup for the pinned path.")

print()

# ─── Realistic inference pattern: overlap tokenization with transfer ──────────

# This is the pattern inference servers use.
# Without pinned memory (sequential):
#   tokenize prompt_1 → transfer → GPU runs → tokenize prompt_2 → transfer → GPU runs
#
# With pinned memory + non_blocking (overlapped):
#   tokenize prompt_1 → start async transfer ──────────────────────────────┐
#                        tokenize prompt_2 → start async transfer ──────┐  │
#                                            GPU runs prompt_1 ←────────┘  │
#                                                        GPU runs prompt_2 ←┘

prompts = ["Tell me a joke.", "What is 2+2?", "Hello!"]

# Simulate: pre-allocate pinned input buffers (in practice, done once at server startup)
# For illustration we pin a small tensor per prompt — real code reuses buffers.
print("Simulated async transfer pattern (CPU print order shows overlap):")
for i, prompt in enumerate(prompts):
    # Step 1: tokenize on CPU (returns pageable tensor from HuggingFace tokenizer by default)
    token_ids = torch.tensor([ord(c) for c in prompt])  # toy tokenization

    # Step 2: copy to pinned buffer
    # In a real inference server you would pre-allocate a pinned tensor pool
    # and write into it rather than calling .pin_memory() per request.
    pinned_ids = token_ids.pin_memory()

    if torch.cuda.is_available():
        # Step 3: async transfer — returns immediately, DMA runs in background
        gpu_ids = pinned_ids.to("cuda", non_blocking=True)
        # CPU is free here — could tokenize next prompt now
        print(f"  Prompt {i+1}: transfer started (CPU free to work on next prompt)")
        # Step 4: synchronize before the model actually reads the tensor
        torch.cuda.synchronize()
        print(f"  Prompt {i+1}: transfer complete, model can read gpu_ids")
    else:
        print(f"  Prompt {i+1}: [{pinned_ids.is_pinned()}] pinned, would transfer async on GPU machine")
