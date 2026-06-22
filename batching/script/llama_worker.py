"""
LlamaWorker — same role as GPUWorker but runs Llama 3.1 8B.

Llama 3.1 8B architecture constants:
  num_layers   = 32
  num_kv_heads = 8   (GQA: 32 query heads, 8 KV heads)
  head_dim     = 128
  vocab_size   = 128256
  EOS token    = 128009  (<|eot_id|>) — used by instruct; 128001 for base

Why Llama instead of GPT-2 for quantization experiments:
  GPT-2 (117M params, FP32 ~0.5GB) — savings from INT8/INT4 are negligible.
  Llama 3.1 8B (8B params, BF16 ~16GB) — INT8 → ~8GB, INT4 → ~4GB.
  Measurable memory and throughput differences justify the experiment.

KV cache sizing for Llama 3.1 8B on A100 80GB:
  Per slot: 32 layers × 2 (K+V) × 8 heads × 128 head_dim × 2 bytes (BF16)
          = 32 × 2 × 8 × 128 × 2 = 131072 bytes = 128 KB per token per slot
  64 slots × 512 seq_len × 128 KB = ~4 GB for KV cache
  Leaves ~60 GB for model weights (16 GB BF16) + activations.

Access: requires HuggingFace token for gated LLaMA repo.
  export HF_TOKEN=<your_token>   before running.
"""

import os
import sys
from typing import List

import torch
import zmq
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from block_pool import KVStore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'generated'))
import messages_pb2

DISPATCH_ADDR = "ipc:///tmp/sched_to_gpu.ipc"
RESULT_ADDR   = "ipc:///tmp/gpu_to_sched.ipc"

MODEL_ID     = "meta-llama/Meta-Llama-3.1-8B"

# Llama 3.1 8B architecture
MAX_SLOTS    = 64     # 64 × 128KB/token × 512 tokens = ~4 GB KV cache
MAX_SEQ_LEN  = 512
NUM_LAYERS   = 32
NUM_KV_HEADS = 8      # GQA: 32 query heads share 8 KV heads
HEAD_DIM     = 128
EOS_TOKEN_ID = 128009  # <|eot_id|> for instruct; 128001 for base

# ── Quantization mode ──────────────────────────────────────────────────────────
# "none"     — BF16, no quantization (~16 GB on A100)
# "int8"     — bitsandbytes W8A16: weights INT8, activations BF16 (~8 GB)
# "int4_nf4" — bitsandbytes W4A16: weights NF4 4-bit (~4 GB)
QUANT_MODE = "none"


def _load_model(model_id: str, device: str) -> AutoModelForCausalLM:
    """
    Load Llama with the quantization mode set by QUANT_MODE.
    Requires HF_TOKEN env var for gated model access.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN env var not set. Required for Llama access.")

    if QUANT_MODE == "none":
        return AutoModelForCausalLM.from_pretrained(
            model_id,
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
        model_id,
        quantization_config=bnb_config,
        device_map={"": device},
        token=hf_token,
    ).eval()


def _extract_kv_slice(past_key_values, batch_idx: int, seq_start: int,
                      num_layers: int) -> tuple:
    """
    Extract KV for one request from batched past_key_values.
    Llama uses GQA — KV heads (8) != query heads (32).
    Shape returned per layer: k=[1, num_kv_heads, seq_len, head_dim]
    """
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


class LlamaWorker:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[LlamaWorker] loading {MODEL_ID} on {self.device} | quant={QUANT_MODE}...")
        self.model = _load_model(MODEL_ID, self.device)

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

        print(f"[LlamaWorker] ready. KV store: {MAX_SLOTS} slots × {MAX_SEQ_LEN} tokens")

    def _to_gpu(self, token_ids: List[int]) -> torch.Tensor:
        n = len(token_ids)
        t = torch.tensor(token_ids, dtype=torch.long)
        self._pinned_buf[0, :n].copy_(t)
        gpu_t = self._pinned_buf[0, :n].to(self.device, non_blocking=True)
        torch.cuda.synchronize()
        return gpu_t.unsqueeze(0)

    def _prefill_batch(self, slots: List[messages_pb2.RequestSlot]
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

            single_kv = _extract_kv_slice(out.past_key_values, i, max_len - slen, NUM_LAYERS)
            self.kv_store.write_slot(s.kv_slot_id, single_kv, slen)

            is_finished = (next_token_id == EOS_TOKEN_ID or s.max_new_tokens <= 1)
            print(f"[LlamaWorker] prefill {s.request_id} "
                  f"input_len={slen} next_token={next_token_id}")

            outputs.append(messages_pb2.RequestOutput(
                request_id=s.request_id,
                next_token_id=next_token_id,
                is_finished=is_finished,
            ))

        return outputs

    def _decode_batch(self, slots: List[messages_pb2.RequestSlot]
                      ) -> List[messages_pb2.RequestOutput]:
        batch_size  = len(slots)
        seq_lengths = [s.seq_length for s in slots]
        max_seq     = max(seq_lengths)

        # load and pad KV caches — num_kv_heads=8 for Llama GQA
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
            new_seq_len = s.seq_length + 1
            single_kv   = _extract_kv_slice(out.past_key_values, i,
                                             max_seq - s.seq_length, NUM_LAYERS)
            self.kv_store.write_slot(s.kv_slot_id, single_kv, new_seq_len)

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

        print(f"[LlamaWorker] decode batch={batch_size} "
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
        print("[LlamaWorker] started, waiting for batches...")
        try:
            while True:
                batch  = messages_pb2.BatchMetadata.FromString(self.in_socket.recv())
                result = self.execute(batch)
                self.out_socket.send(result.SerializeToString())
        except KeyboardInterrupt:
            pass
        finally:
            self.in_socket.close()
            self.out_socket.close()


def run():
    LlamaWorker().run()


if __name__ == "__main__":
    run()
