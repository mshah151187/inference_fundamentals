import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class RequestStatus(Enum):
    WAITING  = "waiting"
    RUNNING  = "running"
    FINISHED = "finished"


@dataclass
class Request:
    request_id: str
    prompt: str
    max_new_tokens: int = 30

    # filled by TokenizerProcess
    token_ids: List[int] = field(default_factory=list)
    num_input_tokens: int = 0

    # filled by Scheduler on promotion
    kv_slot_id: Optional[int] = None
    status: RequestStatus = RequestStatus.WAITING

    # generation state — updated each decode step
    num_generated_tokens: int = 0
    generated_token_ids: List[int] = field(default_factory=list)

    # actual number of token entries written in the KV store for this request.
    # this is NOT the same as current_seq_len:
    #   after prefill:        kv_seq_len = num_input_tokens
    #                         num_generated_tokens = 1  (first token produced)
    #                         current_seq_len = num_input_tokens + 1
    #   after decode step k:  kv_seq_len = num_input_tokens + k
    #                         num_generated_tokens = k + 1
    # the gap exists because the scheduler increments num_generated_tokens when
    # it receives the output, but the KV write happened one step earlier in the
    # GPU Worker. always use kv_seq_len when telling GPU Worker how many entries
    # to read from the KV store.
    kv_seq_len: int = 0

    # timing — for TTFT and total latency metrics
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None    # when slot was allocated (WAITING → RUNNING)
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    @property
    def current_seq_len(self) -> int:
        return self.num_input_tokens + self.num_generated_tokens

    @property
    def is_finished(self) -> bool:
        return self.num_generated_tokens >= self.max_new_tokens

    def queue_wait(self) -> Optional[float]:
        """Time spent in WAITING queue before slot was allocated."""
        if self.start_time is None:
            return None
        return self.start_time - self.arrival_time

    def ttft(self) -> Optional[float]:
        """Arrival → first token. Includes queue wait + prefill time."""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    def itl(self) -> Optional[float]:
        """Average inter-token latency = decode duration / (tokens - 1)."""
        if self.finish_time is None or self.first_token_time is None:
            return None
        decode_steps = self.num_generated_tokens - 1
        if decode_steps <= 0:
            return None
        return (self.finish_time - self.first_token_time) / decode_steps

    def total_latency(self) -> Optional[float]:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time
