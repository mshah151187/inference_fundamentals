"""
Process 1 — RequestGenerator

Simulates external client traffic: emits one request every EMIT_INTERVAL seconds
with varied prompt lengths. Sends RawRequest protobuf messages to TokenizerProcess
via ZMQ PUSH.
"""

import os
import random
import sys
import time
import uuid

import zmq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'generated'))
import messages_pb2

EMIT_INTERVAL  = 1 / 180   # 180 QPS → one request every ~5.6ms
MAX_NEW_TOKENS = 150
ZMQ_ADDR = "ipc:///tmp/gen_to_tok.ipc"

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


def random_prompt(min_sentences: int = 27, max_sentences: int = 30) -> str:
    # 27-30 sentences × ~12 tokens each ≈ 324-360 tokens prompt
    # + 150 max_new_tokens → max total ~510 tokens, well within max_seq_len=1024
    n = random.randint(min_sentences, max_sentences)
    return " ".join(random.choices(SENTENCE_POOL, k=n))


def run():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUSH)
    socket.bind(ZMQ_ADDR)

    print(f"[Generator] started, emitting every {EMIT_INTERVAL}s → {ZMQ_ADDR}")

    req_count = 0
    try:
        while True:
            prompt = random_prompt()
            request = messages_pb2.RawRequest(
                request_id=f"req_{req_count:04d}",
                prompt=prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                arrival_time=time.time(),
            )
            socket.send(request.SerializeToString())
            print(f"[Generator] sent {request.request_id} "
                  f"(prompt words={len(prompt.split())})")
            req_count += 1
            time.sleep(EMIT_INTERVAL)
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        ctx.term()


if __name__ == "__main__":
    run()
