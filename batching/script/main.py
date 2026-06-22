"""
batching/script/main.py — experiment runner for continuous batching.

Wires together variants from common/ and launches all 4 pipeline stages
as separate processes. Addresses are centralized here.

Process topology:
  Generator ──[ipc]──▶ Tokenizer ──[ipc]──▶ Scheduler ──[ipc]──▶ Worker
                                                  ▲                   │
                                                  └──────[ipc]────────┘

Run:
  python main.py                  # GPT-2, no quantization
  python main.py --worker llama   # Llama 3.1 8B (requires HF_TOKEN)
  python main.py --quant int8     # GPT-2 with W8A16 quantization
  python main.py --worker llama --quant int4_nf4  # Llama W4A16
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

from common.request_generators.prompt_generator import PromptGenerator
from common.tokenizers.gpt2_tokenizer import GPT2Tokenizer
from common.schedulers.continuous_scheduler import ContinuousScheduler
from common.gpu_workers.gpt2_worker import GPT2Worker
from common.gpu_workers.llama_worker import LlamaWorker

# ── IPC addresses ─────────────────────────────────────────────────────────────
GEN_TOK_ADDR   = "ipc:///tmp/gen_to_tok.ipc"
TOK_SCHED_ADDR = "ipc:///tmp/tok_to_sched.ipc"
SCHED_GPU_ADDR = "ipc:///tmp/sched_to_gpu.ipc"
GPU_SCHED_ADDR = "ipc:///tmp/gpu_to_sched.ipc"

# ── Scheduler config per worker ───────────────────────────────────────────────
WORKER_CONFIG = {
    "gpt2":  {"cls": GPT2Worker,   "max_slots": 256, "batch_size": 256},
    "llama": {"cls": LlamaWorker,  "max_slots": 64,  "batch_size": 64},
}


# ── process entry points (top-level for mp.Process pickling) ──────────────────

def _run_generator():
    PromptGenerator(GEN_TOK_ADDR).run()


def _run_tokenizer():
    GPT2Tokenizer(GEN_TOK_ADDR, TOK_SCHED_ADDR).run()


def _run_scheduler(max_slots: int, batch_size: int):
    ContinuousScheduler(
        TOK_SCHED_ADDR, SCHED_GPU_ADDR, GPU_SCHED_ADDR,
        max_slots=max_slots, batch_size=batch_size,
    ).run()


def _run_worker(worker_cls, quant_mode: str):
    worker_cls(SCHED_GPU_ADDR, GPU_SCHED_ADDR, quant_mode=quant_mode).run()


# ── orchestrator ──────────────────────────────────────────────────────────────

def main(worker_name: str, quant_mode: str):
    cfg = WORKER_CONFIG[worker_name]
    print(f"[Main] worker={worker_name}  quant={quant_mode}  "
          f"max_slots={cfg['max_slots']}")

    # Worker must start first: binds result socket + pre-allocates GPU memory.
    # Give it time before Scheduler connects.
    processes = [
        mp.Process(target=_run_worker,
                   args=(cfg["cls"], quant_mode),
                   name="Worker",    daemon=True),
        mp.Process(target=_run_scheduler,
                   args=(cfg["max_slots"], cfg["batch_size"]),
                   name="Scheduler", daemon=True),
        mp.Process(target=_run_tokenizer,
                   name="Tokenizer", daemon=True),
        mp.Process(target=_run_generator,
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
    parser.add_argument("--worker", choices=list(WORKER_CONFIG.keys()),
                        default="gpt2")
    parser.add_argument("--quant",  choices=["none", "int8", "int4_nf4"],
                        default="none",
                        help="quantization mode for the model worker")
    args = parser.parse_args()
    main(args.worker, args.quant)
