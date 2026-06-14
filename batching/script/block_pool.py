"""
BlockPool is split across two processes:

  BlockPoolMetadata  — lives in Scheduler process (pure Python, no GPU)
                       owns: free_slots list, allocated dict
                       does: allocate(), free(), free_count()

  KVStore            — lives in GPU Worker process (GPU tensor)
                       owns: pre-allocated kv_store tensor on HBM
                       does: read_slot(), write_slot()

The only data that crosses the process boundary is slot_id integers,
passed in batch_metadata over ZMQ.
"""

from typing import Dict, List, Optional
import torch


class BlockPoolMetadata:
    """
    Owned by Scheduler. Tracks which slots are free and which request owns which slot.
    No GPU memory involved — pure CPU metadata.
    """

    def __init__(self, max_slots: int):
        self.max_slots = max_slots
        self.free_slots: List[int] = list(range(max_slots))
        self.allocated: Dict[str, int] = {}  # request_id -> slot_id

    def allocate(self, request_id: str) -> Optional[int]:
        if not self.free_slots:
            return None
        slot_id = self.free_slots.pop(0)
        self.allocated[request_id] = slot_id
        return slot_id

    def free(self, request_id: str):
        slot_id = self.allocated.pop(request_id, None)
        if slot_id is not None:
            self.free_slots.append(slot_id)

    def free_count(self) -> int:
        return len(self.free_slots)

    def status_str(self) -> str:
        return (f"slots free={self.free_count()}/{self.max_slots} "
                f"allocated={list(self.allocated.keys())}")


class KVStore:
    """
    Owned by GPU Worker. Holds the actual KV cache tensors in HBM.

    Shape: [max_slots, num_layers, 2, max_seq_len, num_kv_heads, head_dim]
            2 = K and V

    For GPT-2 small (num_layers=12, num_kv_heads=12, head_dim=64, FP16=2 bytes):
      per slot = 1024 × 12 × 2 × 12 × 64 × 2 = 36 MB
      910 slots = 910 × 36 MB = 32 GB  (target KV budget on A100 40GB)
      remaining 8 GB: model weights (~500 MB) + CUDA overhead (~1 GB) + safety buffer
    """

    def __init__(self, max_slots: int, num_layers: int, max_seq_len: int,
                 num_kv_heads: int, head_dim: int, device: str = 'cuda'):
        self.max_slots   = max_slots
        self.num_layers  = num_layers
        self.max_seq_len = max_seq_len
        self.device      = device

        # pre-allocate entire KV cache on GPU at startup — never reallocated
        self.kv_store = torch.zeros(
            max_slots, num_layers, 2, max_seq_len, num_kv_heads, head_dim,
            device=device, dtype=torch.float16
        )

        mem_mb = self.kv_store.numel() * 2 / (1024 ** 2)
        print(f"[KVStore] pre-allocated {mem_mb:.1f} MB on {device} "
              f"({max_slots} slots × {num_layers} layers × 2 × {max_seq_len} tokens)")

    def write_slot(self, slot_id: int, past_key_values, seq_len: int):
        """
        Write HuggingFace past_key_values into slot.

        Accepts either legacy tuple-of-tuples (transformers < 4.36) or DynamicCache
        (transformers >= 4.36). DynamicCache exposes .key_cache / .value_cache lists.

        K shape per layer: [batch=1, num_heads, seq_len, head_dim]
        Stored transposed:  [seq_len, num_heads, head_dim] to match kv_store layout.
        """
        use_cache_obj = hasattr(past_key_values, 'key_cache')
        for layer_idx in range(self.num_layers):
            if use_cache_obj:
                k = past_key_values.key_cache[layer_idx]
                v = past_key_values.value_cache[layer_idx]
            else:
                k, v = past_key_values[layer_idx]
            # k: [1, num_heads, seq_len, head_dim] → [seq_len, num_heads, head_dim]
            k = k.squeeze(0).permute(1, 0, 2).to(torch.float16)
            v = v.squeeze(0).permute(1, 0, 2).to(torch.float16)
            self.kv_store[slot_id, layer_idx, 0, :seq_len] = k
            self.kv_store[slot_id, layer_idx, 1, :seq_len] = v

    def read_slot(self, slot_id: int, seq_len: int) -> tuple:
        """
        Read KV cache from slot and return as HuggingFace past_key_values format.
        Returns tuple of num_layers tuples, each (K, V).
          K shape: [1, num_heads, seq_len, head_dim]
        """
        past_key_values = []
        for layer_idx in range(self.num_layers):
            # [seq_len, num_heads, head_dim] → [1, num_heads, seq_len, head_dim]
            k = self.kv_store[slot_id, layer_idx, 0, :seq_len].permute(1, 0, 2).unsqueeze(0).float()
            v = self.kv_store[slot_id, layer_idx, 1, :seq_len].permute(1, 0, 2).unsqueeze(0).float()
            past_key_values.append((k, v))
        return tuple(past_key_values)
