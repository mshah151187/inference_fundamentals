# dataset.py
# Synthetic prompt dataset for GPT-2 inference experiments.
# Mix of short/medium/long prompts approximates a real inference workload.

import random

SHORT = [
    "Tell me a joke.",
    "What is 2+2?",
    "Hello!",
    "What day is it?",
    "Name a color.",
]
MEDIUM = [
    "Explain how transformers work in simple terms.",
    "Write a short story about a robot.",
    "What are the main causes of climate change?",
    "How does a CPU differ from a GPU?",
    "Describe the water cycle.",
]
LONG = [
    "Describe in detail the architecture of the original transformer model "
    "as introduced in the 'Attention is All You Need' paper, including "
    "the encoder, decoder, multi-head attention, and positional encoding.",
    "Write a detailed essay on the history of machine learning from "
    "the perceptron in the 1950s through deep learning in the 2020s.",
    "Explain how NVIDIA H100 GPUs differ from previous generations, covering "
    "HBM3 memory, NVLink 4.0, Transformer Engine, and FP8 compute.",
]

def get_dataset(n=100, seed=42):
    random.seed(seed)
    # ~40% short, ~40% medium, ~20% long
    pool = SHORT * 8 + MEDIUM * 8 + LONG * 4
    return random.choices(pool, k=n)

if __name__ == "__main__":
    data = get_dataset(n=10)
    for i, p in enumerate(data):
        print(f"{i+1}. [{len(p):3d} chars] {p[:60]}...")
