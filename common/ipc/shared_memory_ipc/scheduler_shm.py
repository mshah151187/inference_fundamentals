"""
Process 3 — Scheduler (SHM variant)

Identical to batching/script/scheduler.py with TWO changes:

  1. __init__:
     BEFORE: self.in_socket (ZMQ PULL on tok_to_sched.ipc)
     AFTER:  self.notify_socket (ZMQ SUB, receives 1-byte slot index broadcast from tokenizer)
             self.ring_reader   (ShmRingBufferReader(reader_id=0), reads payload from shm)

     ZMQ SUB requires setsockopt(SUBSCRIBE, b"") to receive any messages.
     Without this, ZMQ SUB silently drops everything.

     reader_id=0 means this process owns flag byte 0 in each slot header.
     A second consumer (e.g. a logger process) would use reader_id=1.

  2. _drain_incoming:
     BEFORE: self.in_socket.recv() → deserialize proto
     AFTER:  self.notify_socket.recv() → slot_idx → ring_reader.read(slot_idx) → deserialize proto

Everything else — scheduling logic, block pool, GPU dispatch, metrics — is unchanged.
"""

import os
import sys
import time
from collections import deque
from typing import Dict, List, Optional

import zmq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'script'))
from block_pool import BlockPoolMetadata
from request import Request, RequestStatus

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'generated'))
import messages_pb2

sys.path.insert(0, os.path.dirname(__file__))
from shm_ring_buffer import ShmRingBufferReader, NOTIFY_ADDR

# ── Unchanged addresses ───────────────────────────────────────────────────────
DISPATCH_ADDR = "ipc:///tmp/sched_to_gpu.ipc"
RESULT_ADDR   = "ipc:///tmp/gpu_to_sched.ipc"

MAX_SLOTS = 256


class SchedulerShm:

    def __init__(self):
        self.waiting: deque[Request]         = deque()
        self.running: List[Request]          = []
        self.block_pool = BlockPoolMetadata(max_slots=MAX_SLOTS)
        self.request_map: Dict[str, Request] = {}

        self.finished_count   = 0
        self.total_tokens_out = 0
        self._metrics_start   = time.time()
        self._last_report     = time.time()
        self._metrics_window: List[Request] = []
        self._REPORT_INTERVAL = 30.0

        ctx = zmq.Context()

        # ── Changed: ZMQ SUB replaces PULL; ShmRingBufferReader takes reader_id ──
        # SUB connects to the tokenizer's PUB socket.
        # SUBSCRIBE b"" = subscribe to all messages (no topic filter).
        # Without setsockopt(SUBSCRIBE), ZMQ SUB receives nothing.
        self.notify_socket = ctx.socket(zmq.SUB)
        self.notify_socket.connect(NOTIFY_ADDR)
        self.notify_socket.setsockopt(zmq.SUBSCRIBE, b"")

        # reader_id=0: this process owns flag[0] in each slot.
        # A second reader process would use ShmRingBufferReader(reader_id=1).
        self.ring_reader   = ShmRingBufferReader(reader_id=0)

        # ── Unchanged: GPU dispatch and result sockets ────────────────────────
        self.dispatch_socket = ctx.socket(zmq.PUSH)
        self.result_socket   = ctx.socket(zmq.PULL)
        self.dispatch_socket.bind(DISPATCH_ADDR)
        self.result_socket.connect(RESULT_ADDR)

        # Poller watches the SUB socket.
        # poll(timeout=0) = non-blocking check; blocks on result_socket when no new requests.
        self.poller = zmq.Poller()
        self.poller.register(self.notify_socket, zmq.POLLIN)

    # ── Changed: drain reads from SHM instead of ZMQ socket ─────────────────

    def _drain_incoming(self):
        total_recv_ms = 0.0
        count = 0
        while True:
            ready = dict(self.poller.poll(timeout=0))
            if self.notify_socket not in ready:
                break

            # Receive the 1-byte slot index notification (tiny ZMQ message)
            slot_idx  = int.from_bytes(self.notify_socket.recv(), 'little')

            # Read payload directly from shared memory — no kernel copy
            t0            = time.perf_counter()
            data, read_ms = self.ring_reader.read(slot_idx)
            total_recv_ms += read_ms

            tokenized = messages_pb2.TokenizedRequest.FromString(data)
            request   = Request(
                request_id=tokenized.request_id,
                prompt=tokenized.prompt,
                max_new_tokens=tokenized.max_new_tokens,
                token_ids=list(tokenized.token_ids),
                num_input_tokens=tokenized.num_input_tokens,
                arrival_time=tokenized.arrival_time,
                status=RequestStatus.WAITING,
            )
            self.waiting.append(request)
            self.request_map[request.request_id] = request
            count += 1
            print(f"[Scheduler-SHM] queued {request.request_id} "
                  f"(input_tokens={request.num_input_tokens}) "
                  f"shm_read={read_ms:.3f}ms "
                  f"waiting={len(self.waiting)}")

        if count > 0:
            print(f"[Scheduler-SHM] drained {count} requests "
                  f"| avg_shm_read={total_recv_ms/count:.3f}ms")

    # ── Everything below is UNCHANGED from scheduler.py ──────────────────────

    def _schedule(self):
        MAX_NEW_PER_STEP = 4
        promoted = []
        while self.waiting and len(promoted) < MAX_NEW_PER_STEP:
            request = self.waiting[0]
            slot_id = self.block_pool.allocate(request.request_id)
            if slot_id is None:
                break
            self.waiting.popleft()
            request.kv_slot_id = slot_id
            request.status     = RequestStatus.RUNNING
            request.start_time = time.time()
            self.running.append(request)
            promoted.append(request.request_id)
        if promoted:
            print(f"[Scheduler-SHM] promoted {promoted} | {self.block_pool.status_str()}")

    def _build_batch_metadata(self) -> Optional[messages_pb2.BatchMetadata]:
        if not self.running:
            return None

        batch = messages_pb2.BatchMetadata()
        for r in self.running:
            is_prefill = (r.num_generated_tokens == 0)
            slot = batch.slots.add()
            slot.request_id            = r.request_id
            slot.token_ids[:]          = (r.token_ids if is_prefill
                                          else [r.generated_token_ids[-1]])
            slot.kv_slot_id            = r.kv_slot_id
            slot.seq_length            = r.num_input_tokens if is_prefill else r.kv_seq_len
            slot.is_prefill            = is_prefill
            slot.max_new_tokens        = r.max_new_tokens
            slot.num_generated_tokens  = r.num_generated_tokens
        return batch

    def _update_from_outputs(self, batch_output: messages_pb2.BatchOutput):
        now = time.time()
        for out in batch_output.outputs:
            request = self.request_map[out.request_id]
            request.generated_token_ids.append(out.next_token_id)
            request.num_generated_tokens += 1

            if request.num_generated_tokens == 1:
                request.kv_seq_len = request.num_input_tokens
            else:
                request.kv_seq_len += 1

            if request.first_token_time is None:
                request.first_token_time = now

            if out.is_finished:
                request.status      = RequestStatus.FINISHED
                request.finish_time = now
                self.block_pool.free(out.request_id)
                self.running = [r for r in self.running
                                if r.request_id != out.request_id]
                self.finished_count   += 1
                self.total_tokens_out += request.num_generated_tokens
                self._metrics_window.append(request)
                print(f"[Scheduler-SHM] FINISHED {out.request_id} | "
                      f"tokens={request.num_generated_tokens} | "
                      f"queue_wait={request.queue_wait():.3f}s | "
                      f"TTFT={request.ttft():.3f}s | "
                      f"ITL={request.itl()*1000:.1f}ms | "
                      f"total={request.total_latency():.3f}s")

    def _log_state(self):
        print(f"[Scheduler-SHM] step | "
              f"waiting={len(self.waiting)} "
              f"running={len(self.running)} "
              f"finished={self.finished_count} | "
              f"{self.block_pool.status_str()}")

    def _report_metrics(self):
        now = time.time()
        if now - self._last_report < self._REPORT_INTERVAL:
            return
        window = self._metrics_window
        if not window:
            self._last_report = now
            return

        elapsed = now - self._metrics_start
        tps     = self.total_tokens_out / elapsed

        def p(vals, pct):
            vals = sorted(vals)
            idx  = int(len(vals) * pct / 100)
            return vals[min(idx, len(vals) - 1)]

        ttfts = [r.ttft()          for r in window if r.ttft()          is not None]
        waits = [r.queue_wait()    for r in window if r.queue_wait()    is not None]
        lats  = [r.total_latency() for r in window if r.total_latency() is not None]
        itls  = [r.itl() * 1000    for r in window if r.itl()           is not None]

        print(
            f"\n{'='*60}\n"
            f"[Metrics-SHM] last {self._REPORT_INTERVAL:.0f}s window "
            f"({len(window)} requests completed)\n"
            f"  Throughput : {tps:.1f} tokens/sec (system lifetime)\n"
            f"  Queue wait : p50={p(waits,50):.3f}s  p99={p(waits,99):.3f}s\n"
            f"  TTFT       : p50={p(ttfts,50):.3f}s  p99={p(ttfts,99):.3f}s\n"
            f"  ITL        : p50={p(itls,50):.1f}ms  p99={p(itls,99):.1f}ms\n"
            f"  Total lat  : p50={p(lats,50):.3f}s   p99={p(lats,99):.3f}s\n"
            f"{'='*60}\n"
        )
        self._metrics_window = []
        self._last_report    = now

    def step(self):
        self._drain_incoming()
        self._schedule()

        batch = self._build_batch_metadata()
        if batch is None:
            time.sleep(0.05)
            return

        self.dispatch_socket.send(batch.SerializeToString())
        batch_output = messages_pb2.BatchOutput.FromString(
            self.result_socket.recv()
        )
        self._update_from_outputs(batch_output)
        self._log_state()
        self._report_metrics()

    def run(self):
        print(f"[Scheduler-SHM] started | max_slots={MAX_SLOTS} | recv=shm+notify")
        try:
            while True:
                self.step()
        except KeyboardInterrupt:
            pass
        finally:
            self.ring_reader.close()
            self.notify_socket.close()
            self.dispatch_socket.close()
            self.result_socket.close()


def run():
    SchedulerShm().run()


if __name__ == "__main__":
    run()
