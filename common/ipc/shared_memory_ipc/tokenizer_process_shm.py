"""
Process 2 — TokenizerProcess (SHM variant)

Identical to batching/script/tokenizer_process.py with ONE change:
  BEFORE: out_socket.send(tokenized.SerializeToString())   ← ZMQ PUSH (4 copies)
  AFTER:  ring_writer.write(data) + notify_socket.send()   ← SHM write + 1-byte notify (1 copy)

ZMQ socket type: PUB (not PUSH).
  PUSH delivers each message to exactly one connected PULL socket (round-robin).
  PUB delivers each message to ALL connected SUB sockets simultaneously.
  Since shm_ring_buffer now supports N_READERS, every reader must receive the
  slot_idx notification — PUB is the correct primitive for broadcast.

Everything else — receiving from Generator, tokenizing, building the proto — is unchanged.
"""

import os
import sys
import time

import zmq
from transformers import GPT2Tokenizer

# Reuse proto from parent project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'generated'))
import messages_pb2

# SHM ring buffer
sys.path.insert(0, os.path.dirname(__file__))
from shm_ring_buffer import ShmRingBufferWriter, NOTIFY_ADDR

IN_ADDR          = "ipc:///tmp/gen_to_tok.ipc"   # unchanged — same Generator feeds both pipelines
MAX_INPUT_TOKENS = 900


def run():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ctx = zmq.Context()

    # ── Unchanged: receive from Generator via ZMQ ─────────────────────────────
    in_socket = ctx.socket(zmq.PULL)
    in_socket.connect(IN_ADDR)

    # ── Changed: ZMQ PUB broadcasts slot_idx to all connected SUB readers ──────
    # PUB binds; every SUB that connects receives every published message.
    # notify_socket sends only a 1-byte slot index (~30 B over IPC).
    # The actual TokenizedRequest bytes go directly into shared memory.
    notify_socket = ctx.socket(zmq.PUB)
    notify_socket.bind(NOTIFY_ADDR)

    ring_writer = ShmRingBufferWriter(notify_socket)

    print(f"[Tokenizer-SHM] started | recv={IN_ADDR} | data=shm | notify(PUB)={NOTIFY_ADDR}")

    try:
        while True:
            # ── Unchanged: receive and tokenize ──────────────────────────────
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

            # ── Changed: write into SHM ring buffer instead of ZMQ send ──────
            data      = tokenized.SerializeToString()
            write_ms  = ring_writer.write(data)

            print(f"[Tokenizer-SHM] {tokenized.request_id} "
                  f"→ {tokenized.num_input_tokens} tokens "
                  f"| shm_write={write_ms:.3f}ms "
                  f"| payload={len(data)}B")

    except KeyboardInterrupt:
        pass
    finally:
        ring_writer.close()
        in_socket.close()
        notify_socket.close()
        ctx.term()


if __name__ == "__main__":
    run()
