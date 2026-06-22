"""
ContinuousScheduler — concrete variant of BaseScheduler.

Implements continuous batching: drains incoming requests without blocking,
admits up to MAX_NEW_PER_STEP new prefills per step to keep activation
memory bounded, runs the synchronous send→recv→update loop with the GPU
Worker, and reports latency metrics (TTFT, ITL, queue_wait) every 30s.

Wire format: protobuf TokenizedRequest in, BatchMetadata/BatchOutput to/from Worker.
"""

import os
import sys
import time
from collections import deque
from typing import Dict, List, Optional

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'batching', 'generated'))

from common.base_scheduler import BaseScheduler
from common.request import Request, RequestStatus
from common.block_pool import BlockPoolMetadata
import messages_pb2
import zmq

MAX_NEW_PER_STEP = 4
REPORT_INTERVAL  = 30.0


class ContinuousScheduler(BaseScheduler):

    def __init__(self, tok_pull_addr: str, dispatch_addr: str, result_addr: str,
                 max_slots: int = 256, batch_size: int = 256):
        # set before super().__init__ because __init__ calls max_kv_slots()
        self._max_slots   = max_slots
        self._batch_size  = batch_size
        super().__init__(tok_pull_addr, dispatch_addr, result_addr)

        # replace base-class queues with Request-aware structures
        self.waiting: deque         = deque()
        self.running: List[Request] = []
        self.request_map: Dict[str, Request] = {}
        self.block_pool = BlockPoolMetadata(max_slots=max_slots)

        # metrics
        self.finished_count   = 0
        self.total_tokens_out = 0
        self._metrics_start   = time.time()
        self._last_report     = time.time()
        self._metrics_window: List[Request] = []

    # ── abstract implementations ──────────────────────────────────────────────

    def max_batch_size(self) -> int:
        return self._batch_size

    def max_kv_slots(self) -> int:
        return self._max_slots

    def on_request_complete(self, request_id: str, token_ids: List[int]) -> None:
        # base hook — metrics are printed in _update_from_outputs; nothing extra here
        pass

    # ── override run() with synchronous step loop ─────────────────────────────

    def run(self) -> None:
        print(f"[{self.__class__.__name__}] started | max_slots={self._max_slots}")
        # re-register poller on the correct sockets (super().__init__ registers them)
        poller = zmq.Poller()
        poller.register(self._tok_in, zmq.POLLIN)
        try:
            while True:
                self._drain_incoming(poller)
                self._schedule()

                batch = self._build_batch()
                if batch is None:
                    time.sleep(0.05)
                    continue

                self._worker_out.send(batch.SerializeToString())
                batch_output = messages_pb2.BatchOutput.FromString(
                    self._worker_in.recv()
                )
                self._update_from_outputs(batch_output)
                self._log_state()
                self._report_metrics()

        except KeyboardInterrupt:
            pass
        finally:
            self._tok_in.close()
            self._worker_out.close()
            self._worker_in.close()
            print(f"[{self.__class__.__name__}] exiting")

    # ── step helpers ──────────────────────────────────────────────────────────

    def _drain_incoming(self, poller: zmq.Poller) -> None:
        while True:
            ready = dict(poller.poll(timeout=0))
            if self._tok_in not in ready:
                break
            proto = messages_pb2.TokenizedRequest.FromString(self._tok_in.recv())
            req = Request(
                request_id=proto.request_id,
                prompt=proto.prompt,
                max_new_tokens=proto.max_new_tokens,
                token_ids=list(proto.token_ids),
                num_input_tokens=proto.num_input_tokens,
                arrival_time=proto.arrival_time,
                status=RequestStatus.WAITING,
            )
            self.waiting.append(req)
            self.request_map[req.request_id] = req
            print(f"[{self.__class__.__name__}] queued {req.request_id} "
                  f"(input_tokens={req.num_input_tokens}) waiting={len(self.waiting)}")

    def _schedule(self) -> None:
        promoted = []
        while self.waiting and len(promoted) < MAX_NEW_PER_STEP:
            req = self.waiting[0]
            slot_id = self.block_pool.allocate(req.request_id)
            if slot_id is None:
                break
            self.waiting.popleft()
            req.kv_slot_id = slot_id
            req.status     = RequestStatus.RUNNING
            req.start_time = time.time()
            self.running.append(req)
            promoted.append(req.request_id)
        if promoted:
            print(f"[{self.__class__.__name__}] promoted {promoted} | "
                  f"{self.block_pool.status_str()}")

    def _build_batch(self) -> Optional[messages_pb2.BatchMetadata]:
        if not self.running:
            return None
        batch = messages_pb2.BatchMetadata()
        for r in self.running:
            is_prefill = (r.num_generated_tokens == 0)
            slot = batch.slots.add()
            slot.request_id   = r.request_id
            slot.token_ids[:] = (r.token_ids if is_prefill
                                 else [r.generated_token_ids[-1]])
            slot.kv_slot_id            = r.kv_slot_id
            slot.seq_length            = r.num_input_tokens if is_prefill else r.kv_seq_len
            slot.is_prefill            = is_prefill
            slot.max_new_tokens        = r.max_new_tokens
            slot.num_generated_tokens  = r.num_generated_tokens
        return batch

    def _update_from_outputs(self, batch_output: messages_pb2.BatchOutput) -> None:
        now = time.time()
        for out in batch_output.outputs:
            req = self.request_map.get(out.request_id)
            if req is None:
                continue
            req.generated_token_ids.append(out.next_token_id)
            req.num_generated_tokens += 1

            if req.num_generated_tokens == 1:
                req.kv_seq_len = req.num_input_tokens
            else:
                req.kv_seq_len += 1

            if req.first_token_time is None:
                req.first_token_time = now

            if out.is_finished:
                req.status      = RequestStatus.FINISHED
                req.finish_time = now
                self.block_pool.free(out.request_id)
                self.running = [r for r in self.running
                                if r.request_id != out.request_id]
                self.finished_count   += 1
                self.total_tokens_out += req.num_generated_tokens
                self._metrics_window.append(req)
                print(f"[{self.__class__.__name__}] FINISHED {out.request_id} | "
                      f"tokens={req.num_generated_tokens} | "
                      f"queue_wait={req.queue_wait():.3f}s | "
                      f"TTFT={req.ttft():.3f}s | "
                      f"ITL={req.itl()*1000:.1f}ms | "
                      f"total={req.total_latency():.3f}s")

    def _log_state(self) -> None:
        print(f"[{self.__class__.__name__}] step | "
              f"waiting={len(self.waiting)} running={len(self.running)} "
              f"finished={self.finished_count} | {self.block_pool.status_str()}")

    def _report_metrics(self) -> None:
        now = time.time()
        if now - self._last_report < REPORT_INTERVAL:
            return
        window = self._metrics_window
        if not window:
            self._last_report = now
            return

        elapsed = now - self._metrics_start
        tps     = self.total_tokens_out / elapsed if elapsed > 0 else 0

        def p(vals, pct):
            vals = sorted(vals)
            idx  = int(len(vals) * pct / 100)
            return vals[min(idx, len(vals) - 1)]

        ttfts = [r.ttft()          for r in window if r.ttft()          is not None]
        waits = [r.queue_wait()    for r in window if r.queue_wait()    is not None]
        lats  = [r.total_latency() for r in window if r.total_latency() is not None]
        itls  = [r.itl() * 1000   for r in window if r.itl()           is not None]

        print(
            f"\n{'='*60}\n"
            f"[Metrics] last {REPORT_INTERVAL:.0f}s window "
            f"({len(window)} requests completed)\n"
            f"  Throughput : {tps:.1f} tokens/sec (lifetime)\n"
            f"  Queue wait : p50={p(waits,50):.3f}s  p99={p(waits,99):.3f}s\n"
            f"  TTFT       : p50={p(ttfts,50):.3f}s  p99={p(ttfts,99):.3f}s\n"
            f"  ITL        : p50={p(itls,50):.1f}ms  p99={p(itls,99):.1f}ms\n"
            f"  Total lat  : p50={p(lats,50):.3f}s   p99={p(lats,99):.3f}s\n"
            f"{'='*60}\n"
        )
        self._metrics_window = []
        self._last_report    = now
