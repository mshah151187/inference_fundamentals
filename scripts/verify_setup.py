"""
Verify the full setup — GPU visible, PyTorch working, GPT-2 can run inference.
Run after setup.md is complete.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# torch.__version__ is a string attribute on the torch module — just prints what's installed
print(f"PyTorch version : {torch.__version__}")

# torch.cuda.is_available() checks if NVIDIA driver + CUDA toolkit are reachable from PyTorch.
# Returns True if GPU is usable, False if CPU-only. Should be True on the A100 instance.
print(f"CUDA available  : {torch.cuda.is_available()}")

# torch.cuda.get_device_name(0) returns the name of GPU at index 0.
# "0" = first GPU. On a single-GPU instance, only index 0 exists.
print(f"GPU             : {torch.cuda.get_device_name(0)}")

# get_device_properties(0).total_memory returns total VRAM in bytes.
# Dividing by 1e9 converts bytes → GB. A100 SXM4 = 40GB, H100 SXM5 = 80GB.
print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# "cuda" tells PyTorch to use the GPU. Alternative is "cpu".
# We store it in a variable so all .to(device) calls are consistent — easy to switch later.
device = "cuda"

print("Loading GPT-2...")

# from_pretrained("gpt2") loads model weights from ~/.cache/huggingface/hub/ (downloaded earlier).
# It reads config.json to build the architecture, then loads weights from model.safetensors.
# At this point the model (548MB of weights) lives on CPU RAM.
#
# .to(device) moves ALL model tensors (weights, buffers) from CPU RAM → GPU VRAM.
# After this call, ~548MB is sitting in the A100's 40GB HBM memory.
#
# .eval() switches the model from training mode to evaluation mode. Two effects:
#   1. Dropout layers are DISABLED — in training, dropout randomly zeros out neurons
#      to prevent overfitting. In inference you want deterministic, full output.
#   2. BatchNorm layers use running statistics instead of batch statistics.
# Important: .eval() does NOT disable gradient computation — that's what no_grad() does below.
# .eval() returns the model itself, so you can chain .to(device).eval() on one line.
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

# from_pretrained loads the tokenizer files: vocab.json (token→ID map) and merges.txt (BPE rules).
# The tokenizer runs entirely on CPU — it's pure Python text processing, no GPU needed.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# GPT-2's tokenizer has no dedicated padding token by default (unlike BERT which has [PAD]).
# eos_token = end-of-sequence token = "<|endoftext|>" = token ID 50256.
# Setting pad_token = eos_token tells the tokenizer: "use <|endoftext|> as padding when
# you need to pad sequences to the same length in a batch."
# Without this, batched inference throws a warning or error.
tokenizer.pad_token = tokenizer.eos_token

print("Running inference...")

# tokenizer("Hello, GPU!") converts the string to token IDs using BPE (Byte Pair Encoding):
#   "Hello, GPU!"  →  [15496, 11, 29140, 0]  (approximate token IDs)
#
# return_tensors="pt" means return PyTorch tensors instead of plain Python lists.
#   "pt" = PyTorch   "tf" = TensorFlow   "np" = NumPy
#
# The tokenizer returns a dict with two keys:
#   input_ids      : tensor of shape (1, seq_len) — the token IDs
#   attention_mask : tensor of shape (1, seq_len) — 1 for real tokens, 0 for padding.
#                    All 1s here since we have no padding (single sequence, no batch).
#
# .to(device) moves both tensors from CPU → GPU so they're on the same device as the model.
# Model and inputs must be on the same device — mixing CPU/GPU throws a RuntimeError.
inputs = tokenizer("Hello, GPU!", return_tensors="pt").to(device)

# --- Inspect tokenizer output ---
print(f"input_ids      : {inputs['input_ids']}")
# e.g. tensor([[15496,    11, 29140,     0]]) — shape (1, 4): batch=1, seq_len=4
# Each number is a token ID mapping to a word/subword in GPT-2's 50,257-token vocabulary.

print(f"attention_mask : {inputs['attention_mask']}")
# e.g. tensor([[1, 1, 1, 1]]) — all 1s: no padding, every token is real.
# Would show 0s at padding positions if we batched sequences of different lengths.

print(f"input_ids shape: {inputs['input_ids'].shape}")
# torch.Size([1, 4]) — (batch_size=1, sequence_length=4)

# Decode each token individually to see how the string was split
tokens = [tokenizer.decode([t]) for t in inputs['input_ids'][0]]
print(f"tokens         : {tokens}")
# e.g. ['Hello', ',', ' GPU', '!'] — shows how BPE splits the string into subwords

print()

# torch.no_grad() is a context manager that disables gradient tracking for everything inside.
# During training, PyTorch records every operation to build a computation graph for backprop.
# During inference we don't need gradients — disabling them:
#   1. Saves GPU memory — no gradient buffers allocated (can be 2-3x the model size)
#   2. Speeds up computation — no overhead tracking ops
# Always use no_grad() during inference.
with torch.no_grad():
    # **inputs unpacks the dict: input_ids=tensor(...), attention_mask=tensor(...)
    # model.generate() runs autoregressive decoding — generates one token at a time:
    #   Step 1: forward pass → get logits (50,257 scores, one per vocab token)
    #   Step 2: pick the highest scoring token (greedy decoding by default)
    #   Step 3: append that token to the input sequence
    #   Step 4: repeat until max_new_tokens is reached or <|endoftext|> is generated
    # max_new_tokens=20 means generate up to 20 tokens beyond the input.
    # Returns tensor of shape (batch_size=1, input_length + generated_length).
    out = model.generate(**inputs, max_new_tokens=20)

# out has shape (1, N) — batch of 1 sequence.
# out[0] gets the first (and only) sequence as a 1D tensor of token IDs.
print(f"output tensor  : {out}")
# e.g. tensor([[15496, 11, 29140, 0, 318, 257, ...]]) — input tokens + generated tokens combined

print(f"output shape   : {out.shape}")
# torch.Size([1, 24]) — input was 4 tokens + 20 new tokens generated = 24 total

# tokenizer.decode() converts token IDs back to a human-readable string.
# skip_special_tokens=True removes <|endoftext|> and other special tokens from the output.
print(f"Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")
print()
print("Setup complete. Ready for Phase 1.")
