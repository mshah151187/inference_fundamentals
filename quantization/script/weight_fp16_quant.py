"""
quantization/script/weight_fp16_quant.py — Llama 3.1 8B FP16 weight quantization.

Weight dtype: float16 (vs bfloat16 in base.py).
FP16 and BF16 use the same memory per parameter (2 bytes) so VRAM and
MAX_SLOTS are unchanged vs the baseline. The difference is numeric range:
  BF16 : 8-bit exponent → same dynamic range as FP32, lower precision
  FP16 : 5-bit exponent → narrower range, higher precision in [0,1]
For LLM inference FP16 can cause overflow on large logits — BF16 is the
preferred half-precision format for Llama. This experiment confirms whether
any throughput or stability difference is observable.

Compare against: base.py (BF16 baseline)

Run:
  python3 quantization/script/weight_fp16_quant.py
  python3 quantization/script/weight_fp16_quant.py --profile torch
  python3 quantization/script/weight_fp16_quant.py --profile ncu
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

from common.request_generators.prompt_generator import PromptGenerator
from common.tokenizers.llama_tokenizer import LlamaTokenizer
from common.schedulers.continuous_scheduler import ContinuousScheduler
from common.gpu_workers.llama_worker import LlamaWorker
from common.profiling_workers.ncu_profiling_worker import NcuProfilingWorker
from common.profiling_workers.torch_profiling_worker import TorchProfilingWorker

# ── experiment config ──────────────────────────────────────────────────────────
QUANT_MODE = "fp16"    # float16 weights
MAX_SLOTS  = 64        # same as BF16 — weight dtype doesn't change KV store size
BATCH_SIZE = 64

TRACES_DIR = os.path.join(_root, "quantization", "traces")

# ── IPC addresses ──────────────────────────────────────────────────────────────
GEN_TOK_ADDR   = "ipc:///tmp/quant_gen_to_tok.ipc"
TOK_SCHED_ADDR = "ipc:///tmp/quant_tok_to_sched.ipc"
SCHED_GPU_ADDR = "ipc:///tmp/quant_sched_to_gpu.ipc"
GPU_SCHED_ADDR = "ipc:///tmp/quant_gpu_to_sched.ipc"

# ── combined worker classes ────────────────────────────────────────────────────
class _TorchProfiledLlamaWorker(TorchProfilingWorker, LlamaWorker): pass
class _NcuProfiledLlamaWorker(NcuProfilingWorker, LlamaWorker):     pass

_WORKER_CLS = {
    "none":  LlamaWorker,
    "torch": _TorchProfiledLlamaWorker,
    "nsys":  _NcuProfiledLlamaWorker,
    "ncu":   _NcuProfiledLlamaWorker,
}

# ── process entry points ───────────────────────────────────────────────────────
def _run_generator(duration: float):
    PromptGenerator(GEN_TOK_ADDR, duration=duration).run()

def _run_tokenizer():
    LlamaTokenizer(GEN_TOK_ADDR, TOK_SCHED_ADDR).run()

def _run_scheduler():
    ContinuousScheduler(
        TOK_SCHED_ADDR, SCHED_GPU_ADDR, GPU_SCHED_ADDR,
        max_slots=MAX_SLOTS, batch_size=BATCH_SIZE,
    ).run()

def _run_worker(profile_mode: str):
    cls   = _WORKER_CLS[profile_mode]
    extra = {"traces_dir": TRACES_DIR} if profile_mode == "torch" else {}
    cls(SCHED_GPU_ADDR, GPU_SCHED_ADDR, quant_mode=QUANT_MODE, **extra).run()

# ── orchestrator ───────────────────────────────────────────────────────────────
def main(duration: float, profile_mode: str):
    print(f"[Main] model=Llama-3.1-8B  quant={QUANT_MODE}  "
          f"max_slots={MAX_SLOTS}  duration={duration}s  profile={profile_mode}")

    processes = [
        mp.Process(target=_run_worker,    args=(profile_mode,), name="Worker",    daemon=True),
        mp.Process(target=_run_scheduler,                        name="Scheduler", daemon=True),
        mp.Process(target=_run_tokenizer,                        name="Tokenizer", daemon=True),
        mp.Process(target=_run_generator, args=(duration,),      name="Generator", daemon=True),
    ]

    print("[Main] starting processes...")
    for p in processes:
        p.start()
        time.sleep(1.0)

    print("[Main] all processes running. Ctrl+C to stop.\n")
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n[Main] shutting down...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        print("[Main] done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--profile", choices=list(_WORKER_CLS.keys()), default="none")
    args = parser.parse_args()
    main(args.duration, args.profile)
