"""
TorchProfilingWorker — BaseWorker mixin that wraps run() with torch.profiler.

Overrides run() to capture CPU + CUDA activity using torch.profiler's
schedule-based capture. Exports a Perfetto-compatible JSON trace to
traces_dir (passed at construction time).

Profiler schedule:
  wait=2    — skip first 2 batches (pipeline ramp-up noise)
  warmup=3  — warm the profiler for 3 batches (CUDA lazy init)
  active=20 — capture 20 batches of real inference
  repeat=1  — one capture window then stop

CUPTI conflict — DO NOT wrap with nsys or ncu simultaneously.
Use NcuProfilingWorker instead for nsys/ncu captures.

Usage (combine with any concrete worker via multiple inheritance):

    class ProfiledLlamaWorker(TorchProfilingWorker, LlamaWorker):
        pass

    worker = ProfiledLlamaWorker(
        dispatch_addr, result_addr,
        quant_mode="none",
        traces_dir="/path/to/quantization/traces",
    )

MRO: TorchProfilingWorker.run() shadows BaseWorker.run().
     LlamaWorker provides load_model / prefill / decode / execute.

Open the exported trace:
  Perfetto UI : https://ui.perfetto.dev  (drag-and-drop the .json file)
  TensorBoard : tensorboard --logdir=<traces_dir>
"""

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'batching', 'generated'))

import torch
import messages_pb2
from common.base_worker import BaseWorker


class TorchProfilingWorker(BaseWorker):
    """
    Mixin: overrides run() with torch.profiler schedule-based capture.
    Subclass alongside a concrete worker (LlamaWorker, GPT2Worker, …).
    Does not implement load_model / prefill / decode — those come from
    the concrete worker via MRO.

    traces_dir is passed as a keyword arg at construction; all other args
    are forwarded to the concrete worker's __init__ via super().
    """

    def __init__(self, *args, traces_dir: str = "./traces", **kwargs):
        # pop traces_dir before forwarding — concrete workers don't expect it
        self._traces_dir = traces_dir
        super().__init__(*args, **kwargs)

    def run(self) -> None:
        os.makedirs(self._traces_dir, exist_ok=True)

        schedule = torch.profiler.schedule(wait=2, warmup=3, active=20, repeat=1)
        handler  = torch.profiler.tensorboard_trace_handler(self._traces_dir)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=schedule,
            on_trace_ready=handler,
            record_shapes=True,
            with_stack=False,
        ) as prof:
            print(f"[{self.__class__.__name__}] torch.profiler active → {self._traces_dir}")
            try:
                while True:
                    batch = messages_pb2.BatchMetadata.FromString(self._in.recv())
                    with torch.profiler.record_function("execute"):
                        result = self.execute(batch)
                    self._out.send(result.SerializeToString())
                    prof.step()
            except KeyboardInterrupt:
                pass
            finally:
                self._in.close()
                self._out.close()
                print(f"[{self.__class__.__name__}] trace saved → {self._traces_dir}")
