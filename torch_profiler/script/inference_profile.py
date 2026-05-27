# inference_profile.py
# Phase 1, Step 3: Profile GPT-2 inference with torch.profiler.
# Answers: which ops dominate GPU time? Is the model compute-bound or memory-bound?
#
# Run from repo root: python3 torch_profiler/inference_profile.py
# Or from torch_profiler/: python3 inference_profile.py
#
# Output: top ops table (CUDA time, CPU time, memory) + inference_trace.json for Perfetto UI

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import torch
from torch.profiler import profile, ProfilerActivity
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import get_dataset

DEVICE = "cuda"
N_PROMPTS = 10       # smaller than baseline — profiling adds overhead per op
MAX_NEW_TOKENS = 50
TRACE_FILE = os.path.join(os.path.dirname(__file__), "inference_trace.json")

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

prompts = get_dataset(n=N_PROMPTS)

# Warm up: bring GPU to steady state before the profiling window opens.
# Three passes on a representative-length prompt (not a dummy 1-token string):
#   Pass 1: CUDA JIT kernel compilation
#   Pass 2: cuBLAS algorithm selection stabilizes + allocator pool fills
#   Pass 3: fully steady state — matches what profiler will measure
# Key principle: warmup data must match production data in sequence length and batch size.
# A 1-token dummy reaches a different cuBLAS/allocator steady state than 50-token prompts,
# so the first real profiled request would still pay a one-time cost.
print("Warming up (3 passes on representative prompt)...")
warmup_prompt = prompts[0]
with torch.no_grad():
    for _ in range(3):
        dummy = tokenizer(warmup_prompt, return_tensors="pt").to(DEVICE)
        _ = model.generate(**dummy, max_new_tokens=MAX_NEW_TOKENS)

print(f"Profiling {N_PROMPTS} prompts...")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,     # lets you see which layer fired (large vs small matmul)
    profile_memory=True,    # tracks GPU memory allocated per op
    with_stack=True,        # records Python call stack per op — 3-5x slower but first run: want full detail
    with_flops=True,        # estimates FLOPs per op using heuristics — enables arithmetic intensity calc
) as prof:
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{N_PROMPTS} done")

print("\n" + "="*70)
print("TOP 20 OPS BY CUDA TIME  (primary view: where does GPU time go?)")
print("="*70)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

print("="*70)
print("TOP 10 OPS BY CPU TIME  (is CPU a bottleneck? tokenization overhead?)")
print("="*70)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print("="*70)
print("TOP 10 OPS BY GPU MEMORY  (what allocates the most VRAM?)")
print("="*70)
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

print("="*70)
print("TOP 10 OPS BY FLOPS  (which ops do the most compute work?)")
print("Note: with_flops=True uses heuristics — only aten::mm/addmm/bmm/baddbmm estimated")
print("="*70)
print(prof.key_averages().table(sort_by="flops", row_limit=10))

prof.export_chrome_trace(TRACE_FILE)
print(f"\nTrace saved to: {TRACE_FILE}")
print("Open at: https://ui.perfetto.dev  (drag and drop the file)")

# Quick interpretation guide printed at runtime for easy reference
print("\n" + "="*70)
print("INTERPRETATION GUIDE")
print("="*70)
print("Compute-bound:   aten::mm + aten::addmm + aten::baddbmm > 70% CUDA time")
print("Memory-bound:    elementwise ops (softmax, gelu, layer_norm) are large share")
print("Low-occupancy:   batch=1 — neither bus nor tensor cores stressed, just small work")
print("CPU bottleneck:  CPU time >> CUDA time across many ops — GPU starved")
print("Arithmetic intensity = flops / (self_cuda_memory_usage bytes)")
print("  < 200 FLOPs/byte on A100 → memory-bound at scale (ridge point = 200)")
print("  > 200 FLOPs/byte on A100 → compute-bound at scale")
