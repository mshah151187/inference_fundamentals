"""
quantization/script/weight_int4_quant.py — Llama 3.1 8B W4A16 NF4 quantization.

Weight quantization: INT4 NF4 via bitsandbytes (W4A16 — weights NF4, activations BF16).
  - NF4 (NormalFloat4): 4-bit format optimized for normally distributed weights
  - Double quantization: quantizes the quantization constants themselves (~0.5 GB extra savings)
  - compute_dtype=bfloat16: dequantizes to BF16 before matmul (not INT4 compute)
  - 4× weight memory reduction vs BF16 → ~4 GB VRAM for weights alone

Expected vs baseline (BF16):
  VRAM      : ~4 GB   (vs ~16 GB)  — 4× reduction
  MAX_SLOTS : 256     (vs 64)      — significantly more concurrent requests
  Throughput: can increase due to more batching headroom
  Accuracy  : slight degradation vs INT8 — NF4 minimizes this for LLMs

Compare against: base.py (BF16), weight_int8_quant.py (INT8)

Run:
  python3 quantization/script/weight_int4_quant.py
  python3 quantization/script/weight_int4_quant.py --profile torch
  python3 quantization/script/weight_int4_quant.py --profile ncu
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
QUANT_MODE = "int4_nf4"  # W4A16 NF4 + double quantization
MAX_SLOTS  = 256          # 4× baseline — large freed VRAM goes to KV slots
BATCH_SIZE = 256

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
