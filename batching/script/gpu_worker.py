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


def _extract_kv_slice(past_key_values, batch_idx: int, seq_start: int,
                      num_layers: int) -> tuple:
    """Extract KV for one request from a batched past_key_values.
    Normalizes all Cache formats to legacy tuple-of-tuples before slicing.
      k shape returned: [1, num_heads, seq_len, head_dim]
    """
    # Normalize to legacy tuple-of-tuples. Try most universal path first.
    if hasattr(past_key_values, 'to_legacy_cache'):
        past_key_values = past_key_values.to_legacy_cache()
    elif hasattr(past_key_values, 'key_cache'):
        past_key_values = tuple(
            (past_key_values.key_cache[l], past_key_values.value_cache[l])
            for l in range(num_layers)
        )
    else:
        # Unknown format — dump all instance attributes for diagnosis
        print(f"[DEBUG] past_key_values type={type(past_key_values).__name__}")
        print(f"[DEBUG] instance attrs: {list(vars(past_key_values).keys())}")

    return tuple(
        (past_key_values[l][0][batch_idx:batch_idx+1, :, seq_start:, :],
         past_key_values[l][1][batch_idx:batch_idx+1, :, seq_start:, :])
        for l in range(num_layers)
    )


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

    def _prefill_batch(self, slots: List[messages_pb2.RequestSlot]
                       ) -> List[messages_pb2.RequestOutput]:
        batch_size  = len(slots)
        seq_lengths = [s.seq_length for s in slots]
        max_len     = max(seq_lengths)

        # left-pad so every sequence ends at position max_len-1
        input_ids      = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        position_ids   = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)

        for i, s in enumerate(slots):
            toks = list(s.token_ids)
            slen = len(toks)
            pad  = max_len - slen
            input_ids[i,      pad:] = torch.tensor(toks, dtype=torch.long, device=self.device)
            attention_mask[i, pad:] = 1
            position_ids[i,   pad:] = torch.arange(slen, device=self.device)

        with torch.no_grad():
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )

        outputs = []
        for i, s in enumerate(slots):
            slen          = s.seq_length
            next_token_id = int(out.logits[i, -1, :].argmax())

            single_kv = _extract_kv_slice(out.past_key_values, i, max_len - slen, NUM_LAYERS)
            self.kv_store.write_slot(s.kv_slot_id, single_kv, slen)

            is_finished = (next_token_id == EOS_TOKEN_ID or s.max_new_tokens <= 1)
            print(f"[GPUWorker] prefill {s.request_id} "
                  f"input_len={slen} next_token={next_token_id}")

            outputs.append(messages_pb2.RequestOutput(
                request_id=s.request_id,
                next_token_id=next_token_id,
                is_finished=is_finished,
            ))

        return outputs

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
            single_kv = _extract_kv_slice(out.past_key_values, i,
                                          max_seq - s.seq_length, NUM_LAYERS)
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

        if prefill_slots:
            outputs.extend(self._prefill_batch(prefill_slots))

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
