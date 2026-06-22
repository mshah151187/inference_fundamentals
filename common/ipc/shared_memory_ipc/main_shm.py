"""
main_shm.py — SHM variant of the batching pipeline.

What changed vs. main.py:
  - Creates shared memory region BEFORE spawning processes (both tokenizer and
    scheduler attach to it by name; main owns the lifetime of the shm object)
  - Spawns tokenizer_process_shm instead of tokenizer_process
  - Spawns scheduler_shm instead of scheduler

What is unchanged:
  - generator.py     — same, feeds tokenizer on the same gen_to_tok.ipc address
  - gpu_worker.py    — same, sched_to_gpu.ipc / gpu_to_sched.ipc unchanged
  - block_pool.py    — same
  - request.py       — same

HOW TO RUN (on Lambda, in two separate tmux panes):

  Pane 1 — ZMQ baseline:
    cd ~/inference_fundamentals/batching/script
    python main.py 2>&1 | tee /tmp/zmq_pipeline.log

  Pane 2 — SHM variant (after ZMQ run completes, or on a fresh instance):
    cd ~/inference_fundamentals/batching/shared_memory_ipc/script
    python main_shm.py 2>&1 | tee /tmp/shm_pipeline.log

WHAT TO COMPARE:
  Both pipelines print the same [Metrics] block every 30s. Compare:
    - TTFT p50/p99
    - ITL  p50/p99
    - Throughput (tokens/sec)

  Additionally, SHM pipeline prints per-request:
    [Tokenizer-SHM] req_0001 → 347 tokens | shm_write=0.021ms | payload=1412B
    [Scheduler-SHM] queued req_0001 ... shm_read=0.008ms ...

  Compare shm_write + shm_read against equivalent ZMQ socket overhead.
"""

import multiprocessing as mp
import sys
import os
import time

# Resolve script paths so child processes can import correctly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_SCRIPT_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'script')

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PARENT_SCRIPT_DIR)

from shm_ring_buffer import create_shm, SHM_NAME

import generator
import gpu_worker
import tokenizer_process_shm
import scheduler_shm


def main():
    # ── Create shared memory BEFORE spawning child processes ──────────────────
    # Both tokenizer_process_shm and scheduler_shm attach to this region by name.
    # Main process owns the SharedMemory object — it stays alive until main exits.
    shm = create_shm()
    print(f"[Main-SHM] shared memory '{SHM_NAME}' ready")

    # GPU Worker must start first — it binds the result socket and
    # pre-allocates GPU memory. Give it time before Scheduler connects.
    processes = [
        mp.Process(target=gpu_worker.run,               name="GPUWorker",     daemon=True),
        mp.Process(target=scheduler_shm.run,            name="Scheduler-SHM", daemon=True),
        mp.Process(target=tokenizer_process_shm.run,    name="Tokenizer-SHM", daemon=True),
        mp.Process(target=generator.run,                name="Generator",     daemon=True),
    ]

    print("[Main-SHM] starting processes...")
    for p in processes:
        p.start()
        time.sleep(1.0)   # stagger startup so sockets bind before connect

    print("[Main-SHM] all processes running. Ctrl+C to stop.\n")
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n[Main-SHM] shutting down...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
    finally:
        # Clean up shared memory
        shm.close()
        shm.unlink()   # remove the /dev/shm/tok_to_sched_shm file
        print("[Main-SHM] shared memory released. done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # required for CUDA in child processes
    main()
