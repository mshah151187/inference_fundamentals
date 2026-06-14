"""
Process 2 — TokenizerProcess

Receives RawRequest protobuf messages from Generator.
Tokenizes the prompt, produces TokenizedRequest protobuf messages.
Forwards to Scheduler via ZMQ PUSH.
"""

import os
import sys

import zmq
from transformers import GPT2Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'generated'))
import messages_pb2

IN_ADDR  = "ipc:///tmp/gen_to_tok.ipc"
OUT_ADDR = "ipc:///tmp/tok_to_sched.ipc"
MAX_INPUT_TOKENS = 900


def run():
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ctx = zmq.Context()

    in_socket  = ctx.socket(zmq.PULL)
    in_socket.connect(IN_ADDR)

    out_socket = ctx.socket(zmq.PUSH)
    out_socket.bind(OUT_ADDR)

    print(f"[Tokenizer] started | {IN_ADDR} → {OUT_ADDR}")

    try:
        while True:
            raw = messages_pb2.RawRequest.FromString(in_socket.recv())

            token_ids = tokenizer.encode(
                raw.prompt, truncation=True, max_length=MAX_INPUT_TOKENS
            )

            tokenized = messages_pb2.TokenizedRequest(
                request_id=raw.request_id,
                prompt=raw.prompt,
                max_new_tokens=raw.max_new_tokens,
                arrival_time=raw.arrival_time,
                token_ids=token_ids,
                num_input_tokens=len(token_ids),
            )
            out_socket.send(tokenized.SerializeToString())
            print(f"[Tokenizer] {tokenized.request_id} → {tokenized.num_input_tokens} tokens")
    except KeyboardInterrupt:
        pass
    finally:
        in_socket.close()
        out_socket.close()
        ctx.term()


if __name__ == "__main__":
    run()
