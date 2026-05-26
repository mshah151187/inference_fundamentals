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

# Warm up: trigger CUDA JIT compilation before the profiling window opens.
# Without this, first-request kernel compilation noise inflates op times.
print("Warming up...")
with torch.no_grad():
    dummy = tokenizer("warmup", return_tensors="pt").to(DEVICE)
    _ = model.generate(**dummy, max_new_tokens=10)

print(f"Profiling {N_PROMPTS} prompts...")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,     # lets you see which layer fired (large vs small matmul)
    profile_memory=True,    # tracks GPU memory allocated per op
    with_stack=False,       # True only for deep debugging — 3-5x slower
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

prof.export_chrome_trace(TRACE_FILE)
print(f"\nTrace saved to: {TRACE_FILE}")
print("Open at: https://ui.perfetto.dev  (drag and drop the file)")

# Quick interpretation guide printed at runtime for easy reference
print("\n" + "="*70)
print("INTERPRETATION GUIDE")
print("="*70)
print("Compute-bound:  aten::mm + aten::addmm + aten::baddbmm > 70% CUDA time")
print("Memory-bound:   elementwise ops (softmax, gelu, layer_norm) are large share")
print("                OR GEMM % is low despite being top op")
print("GPT-2 batch=1 is EXPECTED to be memory-bound — weights loaded for tiny compute")
print("CPU bottleneck: CPU time >> CUDA time across many ops — GPU starved")
