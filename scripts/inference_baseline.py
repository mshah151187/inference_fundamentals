# inference_baseline.py
# Phase 1, Step 2: Run GPT-2 on 50 synthetic prompts, measure throughput and latency.
# No profiling here — this is the clean baseline number to beat in later phases.
#
# Run: python3 inference_baseline.py
# Note down: throughput (tokens/sec) and avg latency (ms/request)

import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import get_dataset

device = "cuda"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

prompts = get_dataset(n=50)

# Warm-up: first inference is slower due to CUDA kernel JIT compilation.
# Run one dummy request outside the timed window so compilation noise doesn't
# inflate our baseline numbers.
print("Warming up...")
with torch.no_grad():
    dummy = tokenizer("warmup", return_tensors="pt").to(device)
    _ = model.generate(**dummy, max_new_tokens=10)

print("Running inference...")
total_output_tokens = 0
start = time.perf_counter()

with torch.no_grad():
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=50)
        # outputs.shape = (1, input_len + 50) — we count all tokens including input
        total_output_tokens += outputs.shape[-1]
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(prompts)} done")

elapsed = time.perf_counter() - start

print(f"\nResults:")
print(f"  Prompts:       {len(prompts)}")
print(f"  Total tokens:  {total_output_tokens}")
print(f"  Elapsed:       {elapsed:.2f}s")
print(f"  Throughput:    {total_output_tokens / elapsed:.1f} tokens/sec")
print(f"  Latency avg:   {elapsed / len(prompts) * 1000:.1f} ms/request")
