"""
Stage 3 — BaseScheduler

Responsibility: receive TokenizedRequests from the Tokenizer, manage KV slot
assignment, form batches, and dispatch BatchMetadata to the Worker over ZMQ.
Receive BatchOutput from the Worker and route results back to callers.

Subclass contract:
  - Override max_batch_size() to set the batch capacity.
  - Override max_kv_slots() to set the KV store capacity.
  - Override on_request_complete() to handle finished requests (log, write, etc.)
  - run() loop is provided.

Slot management:
  The scheduler owns a free-list of integer slot ids (0..max_kv_slots-1).
  Each active request holds one slot for the duration of its decode phase.
  Slots are released back to the free list when a request finishes.

ZMQ topology:
  Tokenizer (PUSH) ──[ipc]──▶ Scheduler (PULL)
  Scheduler (PUSH) ──[ipc]──▶ Worker    (PULL)
  Worker    (PUSH) ──[ipc]──▶ Scheduler (PULL)
"""

import sys
import os
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List

import zmq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'batching', 'generated'))
import messages_pb2


class BaseScheduler(ABC):

    def __init__(self, tok_pull_addr: str, dispatch_addr: str, result_addr: str):
        """
        tok_pull_addr : PULL from Tokenizer.
        dispatch_addr : PUSH batches to Worker.
        result_addr   : PULL results back from Worker.
        """
        self._free_slots: deque = deque(range(self.max_kv_slots()))
        self._active: Dict[str, messages_pb2.RequestSlot] = {}  # request_id → slot
        self._pending: deque = deque()                           # TokenizedRequests waiting for a slot

        ctx = zmq.Context()
        self._tok_in    = ctx.socket(zmq.PULL)
        self._worker_out = ctx.socket(zmq.PUSH)
        self._worker_in  = ctx.socket(zmq.PULL)
        self._tok_in.connect(tok_pull_addr)
        self._worker_out.bind(dispatch_addr)
        self._worker_in.connect(result_addr)

        self._poller = zmq.Poller()
        self._poller.register(self._tok_in,   zmq.POLLIN)
        self._poller.register(self._worker_in, zmq.POLLIN)

    # ── abstract API ──────────────────────────────────────────────────────────

    @abstractmethod
    def max_batch_size(self) -> int:
        """
        Maximum number of requests in one batch sent to the Worker.
        Constrained by GPU memory (active KV caches must all fit in HBM).
        Example: return 8
        """

    @abstractmethod
    def max_kv_slots(self) -> int:
        """
        Total KV cache slots pre-allocated in the Worker's KVStore.
        Must match KVStore(max_slots=...) in the Worker.
        Example: return 256   # for GPT-2
                 return 64    # for Llama 3.1 8B
        """

    @abstractmethod
    def on_request_complete(self, request_id: str, token_ids: List[int]) -> None:
        """
        Called when a request has finished generating (is_finished=True).

        Implementations decide what to do with the output:
          - log to stdout
          - write to HDFS / database
          - push to a results queue
          - decode token_ids back to text and store

        token_ids : full sequence of generated token ids (not including prompt).
        """

    # ── shared slot management ────────────────────────────────────────────────

    def _try_admit(self) -> None:
        """Admit pending requests into active if slots are available."""
        while self._pending and self._free_slots:
            req = self._pending.popleft()
            slot_id = self._free_slots.popleft()
            self._active[req.request_id] = messages_pb2.RequestSlot(
                request_id=req.request_id,
                kv_slot_id=slot_id,
                token_ids=req.token_ids,
                seq_length=len(req.token_ids),
                max_new_tokens=req.max_new_tokens,
                num_generated_tokens=0,
                is_prefill=True,
            )

    def _form_batch(self) -> messages_pb2.BatchMetadata:
        """Pack up to max_batch_size active slots into one BatchMetadata."""
        slots = list(self._active.values())[:self.max_batch_size()]
        return messages_pb2.BatchMetadata(slots=slots)

    def _handle_result(self, result: messages_pb2.BatchOutput) -> None:
        """Process Worker output: update slot state or release finished slots."""
        for output in result.outputs:
            rid = output.request_id
            if rid not in self._active:
                continue
            slot = self._active[rid]

            if output.is_finished:
                self.on_request_complete(rid, list(slot.token_ids))
                self._free_slots.append(slot.kv_slot_id)
                del self._active[rid]
            else:
                # append new token, mark as decode phase
                updated_tokens = list(slot.token_ids) + [output.next_token_id]
                self._active[rid] = messages_pb2.RequestSlot(
                    request_id=slot.request_id,
                    kv_slot_id=slot.kv_slot_id,
                    token_ids=updated_tokens,
                    seq_length=slot.seq_length + 1,
                    max_new_tokens=slot.max_new_tokens,
                    num_generated_tokens=slot.num_generated_tokens + 1,
                    is_prefill=False,
                )

    # ── shared run loop ───────────────────────────────────────────────────────

    def _deserialize_tokenized(self, raw: bytes):
        """Reverse of BaseTokenizer._serialize."""
        parts = raw.decode().split("|", 2)
        from base_tokenizer import TokenizedRequest
        return TokenizedRequest(
            request_id=parts[0],
            max_new_tokens=int(parts[1]),
            token_ids=[int(t) for t in parts[2].split(",") if t],
        )

    def run(self) -> None:
        print(f"[{self.__class__.__name__}] started")
        try:
            while True:
                socks = dict(self._poller.poll(timeout=10))

                if self._tok_in in socks:
                    raw = self._tok_in.recv()
                    req = self._deserialize_tokenized(raw)
                    self._pending.append(req)

                if self._worker_in in socks:
                    result = messages_pb2.BatchOutput.FromString(self._worker_in.recv())
                    self._handle_result(result)

                self._try_admit()

                if self._active:
                    batch = self._form_batch()
                    self._worker_out.send(batch.SerializeToString())

        except KeyboardInterrupt:
            pass
        finally:
            self._tok_in.close()
            self._worker_out.close()
            self._worker_in.close()
            print(f"[{self.__class__.__name__}] exiting")
