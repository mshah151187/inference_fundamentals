import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import get_dataset

# Plain inference — no torch.profiler context manager.
# torch.profiler registers its own CUPTI subscriber which conflicts with nsys.
# nsys wraps this script from outside and observes all CUDA API calls at driver level.

DEVICE = "cuda"
N_PROMPTS = 10
MAX_NEW_TOKENS = 50

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

prompts = get_dataset(n=N_PROMPTS)

# Warmup — 3 passes to reach steady state before nsys starts capturing
print("Warming up...")
warmup_prompt = prompts[0]
with torch.no_grad():
    for _ in range(3):
        dummy = tokenizer(warmup_prompt, return_tensors="pt").to(DEVICE)
        _ = model.generate(**dummy, max_new_tokens=MAX_NEW_TOKENS)

print(f"Running {N_PROMPTS} prompts (nsys is observing)...")
with torch.no_grad():
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{N_PROMPTS} done")

print("Done.")
