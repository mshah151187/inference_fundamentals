"""
Verify the full setup — GPU visible, PyTorch working, GPT-2 can run inference.
Run after setup.md is complete.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"GPU             : {torch.cuda.get_device_name(0)}")
print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

device = "cuda"
print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Running inference...")
inputs = tokenizer("Hello, GPU!", return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20)

print(f"Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")
print()
print("Setup complete. Ready for Phase 1.")
