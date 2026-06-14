"""
main.py — launches all 4 processes and wires them together.

Process topology:

  Generator ──[ipc]──▶ Tokenizer ──[ipc]──▶ Scheduler ──[ipc]──▶ GPUWorker
                                                  ▲                    │
                                                  └──────[ipc]─────────┘

Run:
  python main.py

Stop:
  Ctrl+C — sends KeyboardInterrupt to all child processes.
"""

import multiprocessing as mp
import time

import generator
import tokenizer_process
import scheduler
import gpu_worker


def main():
    # GPU Worker must start first — it binds the result socket and
    # pre-allocates GPU memory. Give it time before Scheduler connects.
    processes = [
        mp.Process(target=gpu_worker.run,          name="GPUWorker",  daemon=True),
        mp.Process(target=scheduler.run,           name="Scheduler",  daemon=True),
        mp.Process(target=tokenizer_process.run,   name="Tokenizer",  daemon=True),
        mp.Process(target=generator.run,           name="Generator",  daemon=True),
    ]

    print("[Main] starting processes...")
    for p in processes:
        p.start()
        time.sleep(1.0)  # stagger startup so sockets bind before connect

    print("[Main] all processes running. Ctrl+C to stop.\n")
    try:
        # wait on all processes
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
    mp.set_start_method("spawn", force=True)  # required for CUDA in child processes
    main()
