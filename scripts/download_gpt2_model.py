"""
Download GPT-2 model and tokenizer from HuggingFace.
Run once — files cached at ~/.cache/huggingface/hub/ on the instance.
"""

"""
GPT2Tokenizer

  Converts between raw text and token IDs — numbers the model actually understands.

  "Hello GPU"  →  [15496, 29140]  →  "Hello GPU"
     text            token IDs          text back

  GPT-2 has a vocabulary of 50,257 tokens. Each token is a word, subword, or character. tokenizer("Hello GPU", return_tensors="pt") gives you a PyTorch tensor of IDs ready to
  feed into the model.

  ---
  GPT2LMHeadModel

  GPT-2 with a language model head on top. Two parts:

  GPT2Model          — the transformer stack (12 layers of attention + MLP)
      +
  LM Head            — a linear layer that maps hidden states → vocab logits (50,257 scores)

  The LM head is what lets the model predict the next token — it takes the last hidden state (768-dim vector) and projects it to a score for every word in the vocabulary.
  Highest score = predicted next token.

  ---
  What from_pretrained("gpt2") does — step by step:

  1. Check ~/.cache/huggingface/hub/
     └── If cached → load from disk (fast)
     └── If not → download from huggingface.co (one time)

  2. Downloads these files:
     ├── config.json        — architecture params (12 layers, 768 hidden dim, 12 heads)
     ├── model.safetensors  — actual trained weights (~548MB)
     ├── vocab.json         — token → ID mapping
     └── merges.txt         — BPE merge rules for tokenizer

  3. Builds the model architecture from config.json

  4. Loads the weights from model.safetensors into the architecture

  5. Returns a Python object ready for inference (on CPU at this point)

  After from_pretrained, the model sits on CPU. Only after .to("cuda") do the weights move into GPU VRAM.

  ---
  The "LMHead" distinction matters:

  HuggingFace has multiple GPT-2 classes:

  ┌───────────────────────────────┬──────────────────────────────────────────────────────┐
  │             Class             │                       Use for                        │
  ├───────────────────────────────┼──────────────────────────────────────────────────────┤
  │ GPT2Model                     │ Raw hidden states — embeddings, feature extraction   │
  ├───────────────────────────────┼──────────────────────────────────────────────────────┤
  │ GPT2LMHeadModel               │ Next token prediction, text generation ← what we use │
  ├───────────────────────────────┼──────────────────────────────────────────────────────┤
  │ GPT2ForSequenceClassification │ Text classification with a classifier head           │
  └───────────────────────────────┴──────────────────────────────────────────────────────┘

  For inference (model.generate()), always use GPT2LMHeadModel.

"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("Downloading GPT-2 model...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

print("Downloading GPT-2 tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print(f"  Vocab size: {tokenizer.vocab_size:,}")

print("Done. Files cached at ~/.cache/huggingface/hub/")
