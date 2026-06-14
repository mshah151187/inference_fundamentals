"""
Process 3 — Scheduler

Owns BlockPoolMetadata (slot accounting, no GPU).
Manages two queues: waiting and running.

Receives TokenizedRequest protobuf messages from Tokenizer.
Sends BatchMetadata protobuf messages to GPU Worker.
Receives BatchOutput protobuf messages from GPU Worker.
"""

import os
import sys
import time
from collections import deque
from typing import Dict, List, Optional

import zmq

from block_pool import BlockPoolMetadata
from request import Request, RequestStatus

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'generated'))
import messages_pb2

IN_ADDR       = "ipc:///tmp/tok_to_sched.ipc"
DISPATCH_ADDR = "ipc:///tmp/sched_to_gpu.ipc"
RESULT_ADDR   = "ipc:///tmp/gpu_to_sched.ipc"

MAX_SLOTS = 256   # 256 × 36 MB = 9.2 GB; leaves ~28 GB for activations + weights


class Scheduler:

    def __init__(self):
        self.waiting: deque[Request]         = deque()
        self.running: List[Request]          = []
        self.block_pool = BlockPoolMetadata(max_slots=MAX_SLOTS)
        self.request_map: Dict[str, Request] = {}

        self.finished_count   = 0
        self.total_tokens_out = 0       # cumulative output tokens across all requests
        self._metrics_start   = time.time()
        self._last_report     = time.time()
        self._metrics_window: List[Request] = []   # completed requests since last report
        self._REPORT_INTERVAL = 30.0               # print aggregate stats every 30s

        ctx = zmq.Context()
        self.in_socket       = ctx.socket(zmq.PULL)
        self.dispatch_socket = ctx.socket(zmq.PUSH)
        self.result_socket   = ctx.socket(zmq.PULL)

        self.in_socket.connect(IN_ADDR)
        self.dispatch_socket.bind(DISPATCH_ADDR)
        self.result_socket.connect(RESULT_ADDR)

        # Poller enables non-blocking check on in_socket.
        # socket.recv() blocks forever until a message arrives — the thread
        # would stall there and never proceed to schedule or dispatch.
        # Poller with timeout=0 lets us peek instantly: if a message is waiting
        # take it, if not move on. This way step() always drains all pending
        # incoming requests first, then proceeds with scheduling regardless of
        # whether new requests arrived or not.
        self.poller = zmq.Poller()
        self.poller.register(self.in_socket, zmq.POLLIN)

    def _drain_incoming(self):
        while True:
            ready = dict(self.poller.poll(timeout=0))
            if self.in_socket not in ready:
                break
            tokenized = messages_pb2.TokenizedRequest.FromString(
                self.in_socket.recv()
            )
            request = Request(
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
            print(f"[Scheduler] queued {request.request_id} "
                  f"(input_tokens={request.num_input_tokens}) "
                  f"waiting={len(self.waiting)}")

    def _schedule(self):
        # Limit new prefills per step so GPU activation memory stays bounded.
        # Each prefill costs O(seq_len × d_model) activation memory vs O(1 × d_model)
        # for decode. Admitting too many prefills at once spikes activation memory
        # on top of the pre-allocated KV cache budget.
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
            print(f"[Scheduler] promoted {promoted} | {self.block_pool.status_str()}")

    def _build_batch_metadata(self) -> Optional[messages_pb2.BatchMetadata]:
        if not self.running:
            return None

        batch = messages_pb2.BatchMetadata()
        for r in self.running:
            is_prefill = (r.num_generated_tokens == 0)
            slot = batch.slots.add()
            slot.request_id     = r.request_id
            slot.token_ids[:]   = (r.token_ids if is_prefill
                                   else [r.generated_token_ids[-1]])
            slot.kv_slot_id     = r.kv_slot_id
            # for prefill: KV store is empty, GPU Worker writes num_input_tokens entries
            # for decode:  use kv_seq_len — the actual entries currently in the KV store,
            #              NOT current_seq_len which is always 1 ahead of the KV store
            slot.seq_length             = r.num_input_tokens if is_prefill else r.kv_seq_len
            slot.is_prefill             = is_prefill
            slot.max_new_tokens         = r.max_new_tokens
            slot.num_generated_tokens   = r.num_generated_tokens
        return batch

    def _update_from_outputs(self, batch_output: messages_pb2.BatchOutput):
        now = time.time()
        for out in batch_output.outputs:
            request = self.request_map[out.request_id]
            request.generated_token_ids.append(out.next_token_id)
            request.num_generated_tokens += 1

            # update kv_seq_len to reflect what the GPU Worker just wrote:
            #   prefill (num_generated_tokens just became 1): wrote num_input_tokens entries
            #   decode  (num_generated_tokens > 1):           extended by 1 entry
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
                print(f"[Scheduler] FINISHED {out.request_id} | "
                      f"tokens={request.num_generated_tokens} | "
                      f"queue_wait={request.queue_wait():.3f}s | "
                      f"TTFT={request.ttft():.3f}s | "
                      f"ITL={request.itl()*1000:.1f}ms | "
                      f"total={request.total_latency():.3f}s")

    def _log_state(self):
        print(f"[Scheduler] step | "
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
        tps     = self.total_tokens_out / elapsed   # tokens/sec since start

        def p(vals, pct):
            vals = sorted(vals)
            idx  = int(len(vals) * pct / 100)
            return vals[min(idx, len(vals) - 1)]

        ttfts   = [r.ttft()         for r in window if r.ttft()         is not None]
        waits   = [r.queue_wait()   for r in window if r.queue_wait()   is not None]
        lats    = [r.total_latency() for r in window if r.total_latency() is not None]
        itls    = [r.itl() * 1000   for r in window if r.itl()          is not None]

        print(
            f"\n{'='*60}\n"
            f"[Metrics] last {self._REPORT_INTERVAL:.0f}s window "
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
        print(f"[Scheduler] started | max_slots={MAX_SLOTS}")
        try:
            while True:
                self.step()
        except KeyboardInterrupt:
            pass
        finally:
            self.in_socket.close()
            self.dispatch_socket.close()
            self.result_socket.close()


def run():
    Scheduler().run()


if __name__ == "__main__":
    run()
