"""
Stage 1 — BaseGenerator

Responsibility: produce a stream of raw inference requests and push them
to the tokenizer stage over ZMQ.

Subclass contract:
  - Override build_request() to produce one request at a time.
  - Override inter_request_delay() to control request arrival rate.
  - run() loop is provided — do not override unless you need custom transport.

ZMQ topology:
  Generator (PUSH) ──[ipc]──▶ Tokenizer (PULL)
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import zmq


@dataclass
class RawRequest:
    """
    The unit passed from Generator → Tokenizer.

    request_id     : unique identifier, carried through the whole pipeline.
    prompt         : raw text to be tokenized and run through the model.
    max_new_tokens : how many tokens to generate for this request.
    """
    request_id:     str
    prompt:         str
    max_new_tokens: int


class BaseGenerator(ABC):

    def __init__(self, push_addr: str):
        """
        push_addr : ZMQ address this generator PUSHes requests to.
                    e.g. "ipc:///tmp/gen_to_tok.ipc"
        """
        ctx = zmq.Context()
        self._socket = ctx.socket(zmq.PUSH)
        self._socket.bind(push_addr)

    # ── abstract API ──────────────────────────────────────────────────────────

    @abstractmethod
    def build_request(self) -> RawRequest:
        """
        Produce the next RawRequest to send downstream.

        Called in a tight loop by run(). Each call should return one request.
        Implementations decide:
          - what prompts to generate (random, from file, from dataset, ...)
          - what max_new_tokens to assign
          - how to assign request_ids (counter, uuid, ...)
        """

    @abstractmethod
    def target_qps(self) -> float:
        """
        Target request arrival rate in queries per second.
        The run() loop sleeps 1/target_qps() seconds between requests
        to maintain this rate on average.

        Examples:
          return 150.0   # 150 QPS stress test
          return 10.0    # 10 QPS steady-state simulation
          return 0.0     # no sleep — maximum throughput
        """

    @abstractmethod
    def duration_seconds(self) -> float:
        """
        How long to generate requests before stopping.

        Examples:
          return 120.0   # run for 2 minutes
          return 30.0    # short smoke test
          return float('inf')  # run until Ctrl+C
        """

    # ── shared run loop ───────────────────────────────────────────────────────

    def _serialize(self, req: RawRequest) -> bytes:
        """Default serialization: pipe-separated plain text."""
        return f"{req.request_id}|{req.max_new_tokens}|{req.prompt}".encode()

    def run(self) -> None:
        qps      = self.target_qps()
        duration = self.duration_seconds()
        delay    = 1.0 / qps if qps > 0 else 0.0

        count    = 0
        deadline = time.monotonic() + duration

        print(f"[{self.__class__.__name__}] started — "
              f"target={qps} QPS, duration={duration}s")
        try:
            while time.monotonic() < deadline:
                req = self.build_request()
                self._socket.send(self._serialize(req))
                count += 1
                if delay > 0:
                    time.sleep(delay)
        except KeyboardInterrupt:
            pass
        finally:
            self._socket.close()
            elapsed = duration - max(0.0, deadline - time.monotonic())
            actual_qps = count / elapsed if elapsed > 0 else 0
            print(f"[{self.__class__.__name__}] sent {count} requests "
                  f"in {elapsed:.1f}s ({actual_qps:.1f} QPS actual), exiting")
