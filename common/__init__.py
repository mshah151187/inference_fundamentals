# common — library of base classes and concrete variants for the inference pipeline.
#
# Pipeline stages (in order):
#   BaseGenerator → BaseTokenizer → BaseScheduler → BaseWorker
#
# Base classes define the contract (abstract methods) and shared run() logic.
# Concrete variants in subdirectories implement the abstract methods.
#
# Variants:
#   common/request_generators/prompt_generator.py  — PromptGenerator (sentence pool, 180 QPS)
#   common/tokenizers/gpt2_tokenizer.py            — GPT2Tokenizer (BPE, max 900 tokens)
#   common/schedulers/continuous_scheduler.py      — ContinuousScheduler (with metrics)
#   common/gpu_workers/gpt2_worker.py              — GPT2Worker  (12L 12H, bnb quant support)
#   common/gpu_workers/llama_worker.py             — LlamaWorker (32L 8KV-H GQA, bnb quant)
#
# Shared utilities:
#   common/request.py     — Request dataclass with latency helpers (ttft, itl, queue_wait)
#   common/block_pool.py  — BlockPoolMetadata (CPU slot accounting) + KVStore (GPU HBM)
#
# Each project (batching/, model_compression/) is a thin experiment runner
# that picks variants from common and wires them together.
