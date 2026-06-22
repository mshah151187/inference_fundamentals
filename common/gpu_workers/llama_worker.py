"""
LlamaWorker — concrete variant of BaseWorker for Llama 3.1 8B.

Llama 3.1 8B architecture:
  num_layers   = 32
  num_kv_heads = 8   (GQA: 32 query heads share 8 KV heads → 4× smaller KV cache)
  head_dim     = 128
  EOS token    = 128009  (<|eot_id|>)

Quantization modes:
  "none"      — BF16 (~16 GB on A100)
  "int8"      — W8A16 bitsandbytes (~8 GB)
  "int4_nf4"  — W4A16 NF4 bitsandbytes (~4 GB)

Requires HF_TOKEN env var for gated model access.
"""

import os
import sys
from typing import List

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'batching', 'generated'))

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from common.base_worker import BaseWorker
from common.block_pool import KVStore
import messages_pb2

MODEL_ID     = "meta-llama/Meta-Llama-3.1-8B"
MAX_SLOTS    = 64
MAX_SEQ_LEN  = 512
NUM_LAYERS   = 32
NUM_KV_HEADS = 8
HEAD_DIM     = 128
EOS_TOKEN_ID = 128009

QUANT_MODE = "none"


def _load_llama(device: str) -> AutoModelForCausalLM:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN env var not set. Required for Llama access.")

    if QUANT_MODE == "none":
        return AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=hf_token,
        ).to(device).eval()

    if QUANT_MODE == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif QUANT_MODE == "int4_nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        raise ValueError(f"Unknown QUANT_MODE: {QUANT_MODE!r}")

    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": device},
        token=hf_token,
    ).eval()


def _extract_kv_slice(past_key_values, batch_idx: int, seq_start: int,
                      num_layers: int) -> tuple:
    if hasattr(past_key_values, 'to_legacy_cache'):
        past_key_values = past_key_values.to_legacy_cache()
    elif hasattr(past_key_values, 'key_cache'):
        past_key_values = tuple(
            (past_key_values.key_cache[l], past_key_values.value_cache[l])
            for l in range(num_layers)
        )
    return tuple(
        (past_key_values[l][0][batch_idx:batch_idx+1, :, seq_start:, :],
         past_key_values[l][1][batch_idx:batch_idx+1, :, seq_start:, :])
        for l in range(num_layers)
    )


class LlamaWorker(BaseWorker):

    def __init__(self, dispatch_addr: str, result_addr: str,
                 quant_mode: str = "none"):
        global QUANT_MODE
        QUANT_MODE = quant_mode
        super().__init__(dispatch_addr, result_addr)  # calls load_model
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
        print(f"[{self.__class__.__name__}] KV store: "
              f"{MAX_SLOTS} slots × {MAX_SEQ_LEN} tokens")

    # ── abstract implementations ──────────────────────────────────────────────

    def load_model(self):
        return _load_llama(self.device)

    def prefill(self, slots: List[messages_pb2.RequestSlot]
                ) -> List[messages_pb2.RequestOutput]:
        batch_size  = len(slots)
        seq_lengths = [s.seq_length for s in slots]
        max_len     = max(seq_lengths)

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
            single_kv     = _extract_kv_slice(out.past_key_values, i,
                                              max_len - slen, NUM_LAYERS)
            self.kv_store.write_slot(s.kv_slot_id, single_kv, slen)
            is_finished = (next_token_id == EOS_TOKEN_ID or s.max_new_tokens <= 1)
            print(f"[{self.__class__.__name__}] prefill {s.request_id} "
                  f"input_len={slen} next_token={next_token_id}")
            outputs.append(messages_pb2.RequestOutput(
                request_id=s.request_id,
                next_token_id=next_token_id,
                is_finished=is_finished,
            ))
        return outputs

    def decode(self, slots: List[messages_pb2.RequestSlot]
               ) -> List[messages_pb2.RequestOutput]:
        batch_size  = len(slots)
        seq_lengths = [s.seq_length for s in slots]
        max_seq     = max(seq_lengths)

        # Llama GQA: KV tensors are bfloat16
        padded_kvs = []
        for layer_idx in range(NUM_LAYERS):
            k_batch = torch.zeros(batch_size, NUM_KV_HEADS, max_seq, HEAD_DIM,
                                  device=self.device, dtype=torch.bfloat16)
            v_batch = torch.zeros(batch_size, NUM_KV_HEADS, max_seq, HEAD_DIM,
                                  device=self.device, dtype=torch.bfloat16)
            for i, s in enumerate(slots):
                kv = self.kv_store.read_slot(s.kv_slot_id, s.seq_length)
                k_i, v_i = kv[layer_idx]
                k_batch[i, :, max_seq - s.seq_length:, :] = k_i.squeeze(0)
                v_batch[i, :, max_seq - s.seq_length:, :] = v_i.squeeze(0)
            padded_kvs.append((k_batch, v_batch))
        past_key_values = tuple(padded_kvs)

        attention_mask = torch.zeros(batch_size, max_seq + 1, device=self.device)
        for i, seq_len in enumerate(seq_lengths):
            attention_mask[i, max_seq - seq_len:] = 1
        attention_mask[:, -1] = 1

        last_tokens = [list(s.token_ids)[-1] for s in slots]
        input_ids   = torch.tensor(last_tokens, dtype=torch.long,
                                   device=self.device).unsqueeze(1)

        with torch.no_grad():
            out = self.model(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
            )

        next_token_ids = out.logits[:, -1, :].argmax(dim=-1).tolist()

        for i, s in enumerate(slots):
            single_kv = _extract_kv_slice(out.past_key_values, i,
                                          max_seq - s.seq_length, NUM_LAYERS)
            self.kv_store.write_slot(s.kv_slot_id, single_kv, s.seq_length + 1)

        outputs = []
        for i, s in enumerate(slots):
            next_tok    = next_token_ids[i]
            is_finished = (next_tok == EOS_TOKEN_ID or
                           (s.num_generated_tokens + 1) >= s.max_new_tokens)
            outputs.append(messages_pb2.RequestOutput(
                request_id=s.request_id,
                next_token_id=next_tok,
                is_finished=is_finished,
            ))

        print(f"[{self.__class__.__name__}] decode batch={batch_size} "
              f"seq_lens={seq_lengths} → tokens={next_token_ids}")
        return outputs
