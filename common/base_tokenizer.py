"""
Stage 2 — BaseTokenizer

Responsibility: receive raw text requests from the Generator, tokenize them,
and push TokenizedRequests to the Scheduler over ZMQ.

Subclass contract:
  - Override load_tokenizer() to return the tokenizer for your model.
  - Override tokenize() to convert text → token id list.
  - run() loop is provided — do not override unless you need custom transport.

ZMQ topology:
  Generator (PUSH) ──[ipc]──▶ Tokenizer (PULL)
  Tokenizer (PUSH) ──[ipc]──▶ Scheduler (PULL)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

import zmq


@dataclass
class TokenizedRequest:
    """
    The unit passed from Tokenizer → Scheduler.

    request_id     : carried from Generator unchanged.
    token_ids      : list of integer token ids produced by the tokenizer.
    max_new_tokens : carried from Generator unchanged.
    """
    request_id:     str
    token_ids:      List[int]
    max_new_tokens: int


class BaseTokenizer(ABC):

    def __init__(self, pull_addr: str, push_addr: str):
        """
        pull_addr : ZMQ address to PULL raw requests from Generator.
        push_addr : ZMQ address to PUSH tokenized requests to Scheduler.
        """
        self._tokenizer = self.load_tokenizer()

        ctx = zmq.Context()
        self._in  = ctx.socket(zmq.PULL)
        self._out = ctx.socket(zmq.PUSH)
        self._in.connect(pull_addr)
        self._out.bind(push_addr)

    # ── abstract API ──────────────────────────────────────────────────────────

    @abstractmethod
    def load_tokenizer(self) -> Any:
        """
        Load and return the tokenizer for your model.
        Called once during __init__.

        Examples:
          return GPT2Tokenizer.from_pretrained("gpt2")
          return AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        """

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """
        Convert a text prompt into a list of integer token ids.

        Implementations decide:
          - truncation strategy (max input length)
          - special tokens (BOS, system prompt wrapping for instruct models)
          - padding (none here — Scheduler handles batch padding)

        Example:
          return self._tokenizer.encode(text, add_special_tokens=True)
        """

    # ── shared run loop ───────────────────────────────────────────────────────

    def _deserialize(self, raw: bytes) -> tuple:
        """Reverse of BaseGenerator._serialize."""
        parts = raw.decode().split("|", 2)
        request_id, max_new_tokens, prompt = parts[0], int(parts[1]), parts[2]
        return request_id, prompt, max_new_tokens

    def _serialize(self, req: TokenizedRequest) -> bytes:
        """Serialize TokenizedRequest for Scheduler."""
        token_str = ",".join(str(t) for t in req.token_ids)
        return f"{req.request_id}|{req.max_new_tokens}|{token_str}".encode()

    def run(self) -> None:
        print(f"[{self.__class__.__name__}] started")
        try:
            while True:
                raw = self._in.recv()
                request_id, prompt, max_new_tokens = self._deserialize(raw)
                token_ids = self.tokenize(prompt)
                req = TokenizedRequest(
                    request_id=request_id,
                    token_ids=token_ids,
                    max_new_tokens=max_new_tokens,
                )
                self._out.send(self._serialize(req))
        except KeyboardInterrupt:
            pass
        finally:
            self._in.close()
            self._out.close()
            print(f"[{self.__class__.__name__}] exiting")
