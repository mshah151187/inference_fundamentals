"""
NcuProfilingWorker — BaseWorker mixin that adds NVTX range annotations.

Overrides execute() to push/pop named NVTX ranges around prefill and decode.
In the Nsight Systems / Nsight Compute timeline you will see:
  [prefill_bN]  — one range per prefill batch (N = batch size)
  [decode_bN]   — one range per decode batch

Safe to use with nsys and ncu — no CUPTI conflict (no torch.profiler).

Usage (combine with any concrete worker via multiple inheritance):

    class ProfiledLlamaWorker(NcuProfilingWorker, LlamaWorker):
        pass

    worker = ProfiledLlamaWorker(dispatch_addr, result_addr, quant_mode="none")

MRO: NcuProfilingWorker.execute() shadows BaseWorker.execute().
     LlamaWorker provides load_model / prefill / decode.

Wrap the entire run command with nsys or ncu — do NOT pass --profile torch
at the same time (CUPTI conflict):

    nsys profile --trace=cuda,nvtx,osrt \\
      --output=<traces_dir>/nsys_capture \\
      python3 <experiment>.py --profile nsys

    sudo env "PATH=$PATH" ncu \\
      --kernel-name "gemv2T_kernel_val|enable_if|layer_norm" \\
      --launch-count 10 --set full \\
      -o <traces_dir>/ncu_capture \\
      python3 <experiment>.py --profile ncu
"""

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'batching', 'generated'))

import torch
import messages_pb2
from common.base_worker import BaseWorker


class NcuProfilingWorker(BaseWorker):
    """
    Mixin: overrides execute() with NVTX annotations.
    Subclass alongside a concrete worker (LlamaWorker, GPT2Worker, …).
    Does not implement load_model / prefill / decode — those come from
    the concrete worker via MRO.
    """

    def execute(self, batch: messages_pb2.BatchMetadata) -> messages_pb2.BatchOutput:
        prefill_slots = [s for s in batch.slots if s.is_prefill]
        decode_slots  = [s for s in batch.slots if not s.is_prefill]
        outputs = []

        if prefill_slots:
            torch.cuda.nvtx.range_push(f"prefill_b{len(prefill_slots)}")
            outputs.extend(self.prefill(prefill_slots))
            torch.cuda.nvtx.range_pop()

        if decode_slots:
            torch.cuda.nvtx.range_push(f"decode_b{len(decode_slots)}")
            outputs.extend(self.decode(decode_slots))
            torch.cuda.nvtx.range_pop()

        return messages_pb2.BatchOutput(outputs=outputs)
