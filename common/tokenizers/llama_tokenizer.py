"""
LlamaTokenizer — concrete variant of BaseTokenizer for Llama 3.1 8B.

Uses AutoTokenizer with the Llama 3.1 chat template.
Requires HF_TOKEN env var for gated model access.
"""

import os
import sys
import time

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'batching', 'generated'))

from common.base_tokenizer import BaseTokenizer, TokenizedRequest
import messages_pb2
import zmq
from transformers import AutoTokenizer

MODEL_ID         = "meta-llama/Meta-Llama-3.1-8B"
MAX_INPUT_TOKENS = 512


class LlamaTokenizer(BaseTokenizer):

    # ── abstract implementations ──────────────────────────────────────────────

    def load_tokenizer(self):
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError("HF_TOKEN env var not set. Required for Llama access.")
        tok = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
        tok.pad_token = tok.eos_token
        return tok

    def tokenize(self, text: str):
        return self._tokenizer.encode(
            text, truncation=True, max_length=MAX_INPUT_TOKENS,
            add_special_tokens=True,
        )

    # ── override run() to carry arrival_time through protobuf ────────────────

    def run(self) -> None:
        print(f"[{self.__class__.__name__}] started")
        try:
            while True:
                raw_proto = messages_pb2.RawRequest.FromString(self._in.recv())

                token_ids = self.tokenize(raw_proto.prompt)

                out = messages_pb2.TokenizedRequest(
                    request_id=raw_proto.request_id,
                    prompt=raw_proto.prompt,
                    max_new_tokens=raw_proto.max_new_tokens,
                    arrival_time=raw_proto.arrival_time,
                    token_ids=token_ids,
                    num_input_tokens=len(token_ids),
                )
                self._out.send(out.SerializeToString())
                print(f"[{self.__class__.__name__}] "
                      f"{raw_proto.request_id} → {len(token_ids)} tokens")
        except KeyboardInterrupt:
            pass
        finally:
            self._in.close()
            self._out.close()
            print(f"[{self.__class__.__name__}] exiting")
