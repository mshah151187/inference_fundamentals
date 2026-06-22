"""
PromptGenerator — concrete variant of BaseGenerator.

Simulates client traffic by emitting random prompts drawn from a fixed
sentence pool. Wire format: protobuf RawRequest (includes arrival_time
for downstream latency metrics).
"""

import os
import random
import sys
import time

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'batching', 'generated'))

from common.base_generator import BaseGenerator, RawRequest
import messages_pb2

SENTENCE_POOL = [
    "The transformer architecture revolutionized natural language processing.",
    "Attention mechanisms allow models to focus on relevant parts of the input.",
    "Large language models are trained on vast amounts of text data.",
    "GPU memory bandwidth is often the bottleneck during inference.",
    "Batching multiple requests together improves hardware utilization.",
    "The key-value cache avoids recomputing attention for previous tokens.",
    "Continuous batching allows new requests to join mid-generation.",
    "Quantization reduces model size by using lower precision weights.",
    "The scheduler decides which requests to process in each iteration.",
    "Pipeline parallelism splits work across multiple processes.",
    "PagedAttention manages KV cache memory in fixed-size blocks.",
    "Flash attention tiles the computation to avoid materializing the full attention matrix.",
]

MAX_NEW_TOKENS = 150


class PromptGenerator(BaseGenerator):

    def __init__(self, push_addr: str, target_qps: float = 180.0,
                 duration: float = float('inf')):
        super().__init__(push_addr)
        self._target_qps  = target_qps
        self._duration    = duration
        self._req_count   = 0

    # ── abstract implementations ──────────────────────────────────────────────

    def build_request(self) -> RawRequest:
        rid    = f"req_{self._req_count:04d}"
        prompt = self._random_prompt()
        self._req_count += 1
        return RawRequest(request_id=rid, prompt=prompt, max_new_tokens=MAX_NEW_TOKENS)

    def target_qps(self) -> float:
        return self._target_qps

    def duration_seconds(self) -> float:
        return self._duration

    # ── protobuf serialization ────────────────────────────────────────────────

    def _serialize(self, req: RawRequest) -> bytes:
        return messages_pb2.RawRequest(
            request_id=req.request_id,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            arrival_time=time.time(),
        ).SerializeToString()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _random_prompt(self, min_sentences: int = 27, max_sentences: int = 30) -> str:
        n = random.randint(min_sentences, max_sentences)
        return " ".join(random.choices(SENTENCE_POOL, k=n))
