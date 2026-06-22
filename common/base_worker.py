"""
Stage 4 — BaseWorker

Responsibility: own the model and KV cache. Receive BatchMetadata from
Scheduler, run prefill and/or decode, return BatchOutput.

Subclass contract:
  - Override load_model() to load your specific model (GPT-2, Llama, etc.)
  - Override prefill() to run the prefill forward pass for a list of slots.
  - Override decode() to run the decode forward pass for a list of slots.
  - execute() and run() are provided — do not override.

Prefill vs Decode:
  Prefill : first pass for a new request. Processes the full prompt in one
            forward pass. Writes KV cache for all prompt tokens.
  Decode  : subsequent passes. Takes one token as input, reads KV cache for
            all prior tokens, writes one new KV entry, returns next token.

ZMQ topology:
  Scheduler (PUSH) ──[ipc]──▶ Worker (PULL)
  Worker    (PUSH) ──[ipc]──▶ Scheduler (PULL)
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Any, List

import torch
import zmq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'batching', 'generated'))
import messages_pb2


class BaseWorker(ABC):

    def __init__(self, dispatch_addr: str, result_addr: str):
        """
        dispatch_addr : PULL batches from Scheduler (Worker connects).
        result_addr   : PUSH results to Scheduler   (Worker binds).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{self.__class__.__name__}] loading model on {self.device}...")
        self.model = self.load_model()
        print(f"[{self.__class__.__name__}] model loaded")

        ctx = zmq.Context()
        self._in  = ctx.socket(zmq.PULL)
        self._out = ctx.socket(zmq.PUSH)
        self._in.connect(dispatch_addr)
        self._out.bind(result_addr)

    # ── abstract API ──────────────────────────────────────────────────────────

    @abstractmethod
    def load_model(self) -> Any:
        """
        Load and return the model placed on self.device.
        Called once during __init__ before ZMQ sockets are opened.

        Implementations decide:
          - which model to load (GPT-2, Llama, quantized variant, ...)
          - quantization config (BitsAndBytesConfig, GPTQ, ...)
          - dtype (FP32, BF16, ...)

        Example (GPT-2):
          return GPT2LMHeadModel.from_pretrained("gpt2").to(self.device).eval()

        Example (Llama INT8):
          return AutoModelForCausalLM.from_pretrained(
              "meta-llama/Meta-Llama-3.1-8B",
              quantization_config=BitsAndBytesConfig(load_in_8bit=True),
              device_map={"": self.device},
          ).eval()
        """

    @abstractmethod
    def prefill(
        self,
        slots: List[messages_pb2.RequestSlot],
    ) -> List[messages_pb2.RequestOutput]:
        """
        Run the prefill forward pass for a batch of new requests.

        Each slot in `slots` has is_prefill=True. The full prompt token ids
        are in slot.token_ids. Implementations must:
          1. Build a padded input_ids tensor [batch, max_seq_len]
          2. Run model forward pass with use_cache=True
          3. Extract and store KV cache per slot into self.kv_store
          4. Return one RequestOutput per slot with:
               - next_token_id : argmax of logits at the last prompt position
               - is_finished   : True if EOS or max_new_tokens <= 1

        Padding: left-pad so all sequences end at the same position.
        This ensures the last logit (used for next token) is always at index -1.
        """

    @abstractmethod
    def decode(
        self,
        slots: List[messages_pb2.RequestSlot],
    ) -> List[messages_pb2.RequestOutput]:
        """
        Run the decode forward pass for a batch of in-progress requests.

        Each slot in `slots` has is_prefill=False. slot.token_ids contains
        the full sequence so far (prompt + generated tokens). Implementations must:
          1. Load KV cache from self.kv_store for each slot
          2. Use only the last token as input_ids [batch, 1]
          3. Run model forward pass with past_key_values
          4. Write updated KV cache back to self.kv_store
          5. Return one RequestOutput per slot with:
               - next_token_id : argmax of output logits
               - is_finished   : True if EOS or num_generated_tokens+1 >= max_new_tokens
        """

    # ── shared execute + run ──────────────────────────────────────────────────

    def execute(
        self,
        batch: messages_pb2.BatchMetadata,
    ) -> messages_pb2.BatchOutput:
        """
        Split batch into prefill and decode slots, dispatch to abstract methods,
        merge results. Subclasses do not override this.
        """
        outputs = []
        prefill_slots = [s for s in batch.slots if s.is_prefill]
        decode_slots  = [s for s in batch.slots if not s.is_prefill]

        if prefill_slots:
            outputs.extend(self.prefill(prefill_slots))
        if decode_slots:
            outputs.extend(self.decode(decode_slots))

        return messages_pb2.BatchOutput(outputs=outputs)

    def run(self) -> None:
        print(f"[{self.__class__.__name__}] started, waiting for batches...")
        try:
            while True:
                batch  = messages_pb2.BatchMetadata.FromString(self._in.recv())
                result = self.execute(batch)
                self._out.send(result.SerializeToString())
        except KeyboardInterrupt:
            pass
        finally:
            self._in.close()
            self._out.close()
            print(f"[{self.__class__.__name__}] exiting")
