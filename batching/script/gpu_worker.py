"""
Process 4 — GPUWorker

Owns KVStore (actual GPU HBM allocation).
Receives BatchMetadata protobuf messages from Scheduler.
Sends BatchOutput protobuf messages back to Scheduler.
"""

import os
import sys
from typing import List

import torch
import zmq
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from block_pool import KVStore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'generated'))
import messages_pb2

DISPATCH_ADDR = "ipc:///tmp/sched_to_gpu.ipc"
RESULT_ADDR   = "ipc:///tmp/gpu_to_sched.ipc"

MAX_SLOTS    = 910   # 910 × 36 MB = 32 GB KV cache on A100 40GB
MAX_SEQ_LEN  = 1024
NUM_LAYERS   = 12
NUM_KV_HEADS = 12
HEAD_DIM     = 64
EOS_TOKEN_ID = 50256


class GPUWorker:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[GPUWorker] loading GPT-2 on {self.device}...")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device).eval()

        self.kv_store = KVStore(
            max_slots=MAX_SLOTS,
            num_layers=NUM_LAYERS,
            max_seq_len=MAX_SEQ_LEN,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            device=self.device,
        )

        self._pinned_buf = torch.zeros(
            MAX_SLOTS, MAX_SEQ_LEN, dtype=torch.long
        ).pin_memory()

        ctx = zmq.Context()
        self.in_socket  = ctx.socket(zmq.PULL)
        self.out_socket = ctx.socket(zmq.PUSH)
        self.in_socket.connect(DISPATCH_ADDR)
        self.out_socket.bind(RESULT_ADDR)

    def _to_gpu(self, token_ids: List[int]) -> torch.Tensor:
        n = len(token_ids)
        t = torch.tensor(token_ids, dtype=torch.long)
        self._pinned_buf[0, :n].copy_(t)
        gpu_t = self._pinned_buf[0, :n].to(self.device, non_blocking=True)
        torch.cuda.synchronize()
        return gpu_t.unsqueeze(0)  # [1, seq_len]

    def _prefill(self, slot: messages_pb2.RequestSlot) -> messages_pb2.RequestOutput:
        input_ids = self._to_gpu(list(slot.token_ids))
        with torch.no_grad():
            out = self.model(input_ids, use_cache=True)

        next_token_id = int(out.logits[0, -1, :].argmax())
        self.kv_store.write_slot(slot.kv_slot_id, out.past_key_values,
                                 slot.seq_length)

        is_finished = (next_token_id == EOS_TOKEN_ID or slot.max_new_tokens <= 1)
        print(f"[GPUWorker] prefill {slot.request_id} "
              f"input_len={slot.seq_length} next_token={next_token_id}")

        return messages_pb2.RequestOutput(
            request_id=slot.request_id,
            next_token_id=next_token_id,
            is_finished=is_finished,
        )

    def _decode_batch(self, slots: List[messages_pb2.RequestSlot]
                      ) -> List[messages_pb2.RequestOutput]:
        batch_size = len(slots)
        seq_lengths = [s.seq_length for s in slots]
        max_seq = max(seq_lengths)

        # ── load and pad KV caches ──────────────────────────────────────
        padded_kvs = []
        for layer_idx in range(NUM_LAYERS):
            k_batch = torch.zeros(batch_size, NUM_KV_HEADS, max_seq, HEAD_DIM,
                                  device=self.device)
            v_batch = torch.zeros(batch_size, NUM_KV_HEADS, max_seq, HEAD_DIM,
                                  device=self.device)
            for i, s in enumerate(slots):
                kv = self.kv_store.read_slot(s.kv_slot_id, s.seq_length)
                k_i, v_i = kv[layer_idx]
                k_batch[i, :, max_seq - s.seq_length:, :] = k_i.squeeze(0)
                v_batch[i, :, max_seq - s.seq_length:, :] = v_i.squeeze(0)
            padded_kvs.append((k_batch, v_batch))
        past_key_values = tuple(padded_kvs)

        # ── attention mask ──────────────────────────────────────────────
        attention_mask = torch.zeros(batch_size, max_seq + 1, device=self.device)
        for i, seq_len in enumerate(seq_lengths):
            attention_mask[i, max_seq - seq_len:] = 1
        attention_mask[:, -1] = 1

        # ── last token per request as input ────────────────────────────
        last_tokens = [list(s.token_ids)[-1] for s in slots]
        input_ids = torch.tensor(last_tokens, dtype=torch.long,
                                 device=self.device).unsqueeze(1)

        # ── batched forward pass ────────────────────────────────────────
        with torch.no_grad():
            out = self.model(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
            )

        next_token_ids = out.logits[:, -1, :].argmax(dim=-1).tolist()

        # ── write updated KV back to slots ──────────────────────────────
        for i, s in enumerate(slots):
            new_seq_len = s.seq_length + 1
            single_kv = tuple(
                (out.past_key_values[l][0][i:i+1, :, max_seq - s.seq_length:, :],
                 out.past_key_values[l][1][i:i+1, :, max_seq - s.seq_length:, :])
                for l in range(NUM_LAYERS)
            )
            self.kv_store.write_slot(s.kv_slot_id, single_kv, new_seq_len)

        outputs = []
        for i, s in enumerate(slots):
            next_tok = next_token_ids[i]
            # num_generated_tokens is tokens generated BEFORE this step.
            # After this step we have num_generated_tokens + 1 total.
            # seq_length is kv_seq_len (input tokens + generated so far),
            # NOT the count of generated tokens — comparing it to max_new_tokens
            # would fire immediately for long prompts.
            is_finished = (next_tok == EOS_TOKEN_ID or
                           (s.num_generated_tokens + 1) >= s.max_new_tokens)
            outputs.append(messages_pb2.RequestOutput(
                request_id=s.request_id,
                next_token_id=next_tok,
                is_finished=is_finished,
            ))

        print(f"[GPUWorker] decode batch={batch_size} "
              f"seq_lens={seq_lengths} → tokens={next_token_ids}")
        return outputs

    def execute(self, batch: messages_pb2.BatchMetadata) -> messages_pb2.BatchOutput:
        outputs = []

        prefill_slots = [s for s in batch.slots if s.is_prefill]
        decode_slots  = [s for s in batch.slots if not s.is_prefill]

        for slot in prefill_slots:
            outputs.append(self._prefill(slot))

        if decode_slots:
            outputs.extend(self._decode_batch(decode_slots))

        return messages_pb2.BatchOutput(outputs=outputs)

    def run(self):
        print("[GPUWorker] started, waiting for batches...")
        try:
            while True:
                batch = messages_pb2.BatchMetadata.FromString(self.in_socket.recv())
                result = self.execute(batch)
                self.out_socket.send(result.SerializeToString())
        except KeyboardInterrupt:
            pass
        finally:
            self.in_socket.close()
            self.out_socket.close()


def run():
    GPUWorker().run()


if __name__ == "__main__":
    run()
