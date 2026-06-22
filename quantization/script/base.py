"""
quantization/script/base.py — Llama 3.1 8B BF16 baseline (no quantization).

Wires together components from common/ into the 4-process pipeline.
All implementation lives in common/; this file only reflects the experiment
configuration (model, quant mode, slot budget, profiling mode).

Process topology:
  Generator ──[ipc]──▶ Tokenizer ──[ipc]──▶ Scheduler ──[ipc]──▶ Worker
                                                  ▲                   │
                                                  └──────[ipc]────────┘

Profiling:
  --profile none   plain run, maximum throughput (default)
  --profile torch  torch.profiler inside Worker → quantization/traces/
  --profile nsys   NVTX annotations → wrap with:
                     nsys profile --trace=cuda,nvtx,osrt \\
                       --output=quantization/traces/llama_base_nsys \\
                       python3 quantization/script/base.py --profile nsys
  --profile ncu    NVTX annotations → wrap with:
                     sudo env "PATH=$PATH" ncu \\
                       --kernel-name "gemv2T_kernel_val|enable_if|layer_norm" \\
                       --launch-count 10 --set full \\
                       -o quantization/traces/llama_base_ncu \\
                       python3 quantization/script/base.py --profile ncu

Requires:
  export HF_TOKEN=<your_token>

Run:
  python3 quantization/script/base.py                    # 2 min, no profiling
  python3 quantization/script/base.py --profile torch    # torch profiler
  python3 quantization/script/base.py --profile nsys     # wrap with nsys
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
QUANT_MODE = "none"    # BF16 baseline; int8.py → "int8", int4.py → "int4_nf4"
MAX_SLOTS  = 64        # KV slots; int8 → 128, int4 → 256
BATCH_SIZE = 64

TRACES_DIR = os.path.join(_root, "quantization", "traces")

# ── IPC addresses ──────────────────────────────────────────────────────────────
GEN_TOK_ADDR   = "ipc:///tmp/quant_gen_to_tok.ipc"
TOK_SCHED_ADDR = "ipc:///tmp/quant_tok_to_sched.ipc"
SCHED_GPU_ADDR = "ipc:///tmp/quant_sched_to_gpu.ipc"
GPU_SCHED_ADDR = "ipc:///tmp/quant_gpu_to_sched.ipc"

# ── combined worker classes ────────────────────────────────────────────────────
# Thin combinations of a profiling mixin (from common/) + LlamaWorker.
# MRO: profiling mixin's run()/execute() shadows BaseWorker's;
#      LlamaWorker provides load_model / prefill / decode.

class _TorchProfiledLlamaWorker(TorchProfilingWorker, LlamaWorker):
    pass

class _NcuProfiledLlamaWorker(NcuProfilingWorker, LlamaWorker):
    pass

_WORKER_CLS = {
    "none":  LlamaWorker,
    "torch": _TorchProfiledLlamaWorker,
    "nsys":  _NcuProfiledLlamaWorker,
    "ncu":   _NcuProfiledLlamaWorker,
}

# ── process entry points ───────────────────────────────────────────────────────
# Top-level functions required for mp.Process pickling under spawn.
# profile_mode passed explicitly — spawn gives each child a fresh interpreter.

def _run_generator(duration: float):
    PromptGenerator(GEN_TOK_ADDR, duration=duration).run()

def _run_tokenizer():
    LlamaTokenizer(GEN_TOK_ADDR, TOK_SCHED_ADDR).run()

def _run_scheduler():
    ContinuousScheduler(
        TOK_SCHED_ADDR, SCHED_GPU_ADDR, GPU_SCHED_ADDR,
        max_slots=MAX_SLOTS,
        batch_size=BATCH_SIZE,
    ).run()

def _run_worker(profile_mode: str):
    cls = _WORKER_CLS[profile_mode]
    extra = {"traces_dir": TRACES_DIR} if profile_mode == "torch" else {}
    cls(SCHED_GPU_ADDR, GPU_SCHED_ADDR, quant_mode=QUANT_MODE, **extra).run()

# ── orchestrator ───────────────────────────────────────────────────────────────

def main(duration: float, profile_mode: str):
    print(f"[Main] model=Llama-3.1-8B  quant={QUANT_MODE}  "
          f"max_slots={MAX_SLOTS}  duration={duration}s  profile={profile_mode}")

    processes = [
        mp.Process(target=_run_worker,    args=(profile_mode,),
                   name="Worker",    daemon=True),
        mp.Process(target=_run_scheduler, name="Scheduler", daemon=True),
        mp.Process(target=_run_tokenizer, name="Tokenizer", daemon=True),
        mp.Process(target=_run_generator, args=(duration,),
                   name="Generator", daemon=True),
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
    parser.add_argument("--duration", type=float, default=120.0,
                        help="run duration in seconds (default: 120)")
    parser.add_argument("--profile", choices=list(_WORKER_CLS.keys()),
                        default="none")
    args = parser.parse_args()

    main(args.duration, args.profile)
