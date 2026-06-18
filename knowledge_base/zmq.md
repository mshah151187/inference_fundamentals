# ZeroMQ (ZMQ)

ZMQ is a **messaging library**, not a message broker. There is no central server,
no daemon, no queue process sitting between sender and receiver. The library itself
implements the socket, the queue, the transport, and the framing — embedded directly
into your process.

This is the first thing to get right: ZMQ ≠ RabbitMQ, Kafka, Redis Pub/Sub.
Those are external services your process connects to. ZMQ is a library your process
links against. The queue lives inside your process's memory.

---

## 1. The Problem ZMQ Solves

Raw OS sockets (BSD sockets, `socket()` syscall) give you byte streams (TCP) or
datagrams (UDP). They do not give you:
- Message framing — you must define where one message ends and the next begins
- Reconnection — if the other side dies, you handle it
- Queuing — if the receiver is slow, you handle backpressure
- Patterns — request/reply, fan-out, load balancing — all manual

ZMQ wraps raw sockets and adds all of the above. You work with **messages** (framed,
atomic units), not byte streams. The library handles reconnection, buffering,
and pattern enforcement.

---

## 2. Core Abstraction: The ZMQ Socket

A ZMQ socket is not a BSD socket. It is a higher-level object that:
- Can manage multiple underlying OS connections internally
- Has a type that enforces a communication pattern
- Has an internal send/receive queue (configurable size)
- Handles framing, reconnection, and flow control automatically

```python
import zmq

ctx = zmq.Context()
socket = ctx.socket(zmq.PUSH)   # socket type determines the pattern
socket.bind("ipc:///tmp/my_queue.ipc")
socket.send(b"hello")
```

---

## 3. Transport Types

ZMQ supports four transports. The choice determines where messages travel.

```
Transport     Address format              What it uses under the hood
──────────────────────────────────────────────────────────────────────
inproc://     inproc://my-queue           Shared memory within one process
                                          Zero syscalls — just a memory copy
                                          Threads in same process only

ipc://        ipc:///tmp/my.ipc           Unix domain socket
                                          Kernel-mediated, stays on same machine
                                          No network stack — very fast
                                          Separate processes on same host

tcp://        tcp://127.0.0.1:5555        TCP socket
              tcp://*:5555                Works across machines
                                          Full network stack overhead

pgm://        pgm://eth0;239.0.0.1:5555   Multicast (rare, specialized)
```

**For inference serving (vLLM):**
- Scheduler → GPU Worker: `ipc://` (separate processes, same machine)
- Within a single process (threads): `inproc://`
- Multi-node inference: `tcp://`

**Why `ipc://` over `tcp://` for same-machine IPC:**
Unix domain sockets bypass the full network stack. No IP header processing,
no port allocation overhead, no loopback interface. Data goes through the kernel
buffer directly. Latency is roughly 2-5x lower than localhost TCP.

---

## 4. Socket Patterns (Types)

The socket type determines who can talk to whom and in what pattern.
This is ZMQ's killer feature — patterns are first-class, not bolted on.

### PUSH / PULL — Pipeline / Fan-out

```
Producer (PUSH) ──→ [ZMQ Queue] ──→ Consumer (PULL)
```

- PUSH sends, PULL receives
- If multiple PULLs: ZMQ round-robins messages across them automatically
- One-way — no reply
- **vLLM uses this**: EngineCore PUSH → GPU Worker PULL for batch dispatch

```python
# sender
s = ctx.socket(zmq.PUSH)
s.bind("ipc:///tmp/work.ipc")
s.send(msgpack.encode(batch_metadata))

# receiver (GPU worker)
r = ctx.socket(zmq.PULL)
r.connect("ipc:///tmp/work.ipc")
batch = msgpack.decode(r.recv())
```

### REQ / REP — Request / Reply

```
Client (REQ) ←──→ Server (REP)
```

- Strict alternating: send then recv, send then recv — enforced by the library
- Simplest pattern but fragile — if either side crashes mid-exchange, socket is stuck
- Good for simple RPC, bad for high-throughput serving

### DEALER / ROUTER — Async Request / Reply

```
Client (DEALER) ──→ Router ──→ Worker (DEALER)
```

- DEALER is REQ without the strict alternation — can send multiple requests in flight
- ROUTER adds identity frames — can route replies back to specific senders
- Used in vLLM for the front-end ↔ EngineCore path where multiple clients talk to one engine

### PUB / SUB — Broadcast

```
Publisher (PUB) ──→ [all subscribers simultaneously]
                 ├──→ Subscriber 1 (SUB)
                 ├──→ Subscriber 2 (SUB)
                 └──→ Subscriber 3 (SUB)
```

- PUB sends each message to ALL connected SUBs simultaneously
- One-way, no acknowledgement, no reply
- Good for: slot-index notification to multiple readers, metrics streaming, config broadcast

**vs PUSH/PULL:**

| | PUSH / PULL | PUB / SUB |
|---|---|---|
| Delivery | round-robin — one receiver per message | broadcast — every subscriber per message |
| Use when | work queue (load balance tasks) | broadcast (every consumer needs every message) |

**The mandatory SUBSCRIBE call:**

```python
# Publisher (binds)
pub = ctx.socket(zmq.PUB)
pub.bind("ipc:///tmp/notify.ipc")

# Subscriber (connects) — MUST call setsockopt before any recv()
sub = ctx.socket(zmq.SUB)
sub.connect("ipc:///tmp/notify.ipc")
sub.setsockopt(zmq.SUBSCRIBE, b"")   # b"" = subscribe to ALL messages
                                      # b"metrics" = only messages starting with "metrics"
# Without setsockopt(SUBSCRIBE), the SUB socket receives NOTHING — silent drop
```

**Topic filtering:**

```python
# Publisher sends with a topic prefix
pub.send(b"metrics " + data)
pub.send(b"alert "   + data)

# Subscriber A — only wants metrics
sub_a.setsockopt(zmq.SUBSCRIBE, b"metrics")

# Subscriber B — wants everything
sub_b.setsockopt(zmq.SUBSCRIBE, b"")
```

Filtering happens inside the PUB socket before sending — subscribers that don't
match the prefix never receive the bytes. In our SHM pipeline, we use `b""` (no filter)
because the slot_idx notification is always relevant to every reader.

**The slow joiner problem:**

```
Timeline:
  t=0  PUB binds and starts sending
  t=1  SUB connects
  t=2  SUB calls setsockopt(SUBSCRIBE, b"")

Messages sent at t=0..1 are DROPPED — the SUB was not connected yet.
ZMQ PUB has no message queue for late-joining subscribers.
```

Fix: ensure SUB connects and subscribes BEFORE PUB starts sending. In our
pipeline, startup order in `main_shm.py` is: Scheduler (SUB) first, then
Tokenizer (PUB). With a 1-second stagger between each process start, the SUB
is subscribed well before the PUB sends its first message.

### PAIR — Exclusive Pair

```
Process A (PAIR) ←──→ Process B (PAIR)
```

- Exactly two endpoints, bidirectional
- No routing, no fan-out
- Used for shutdown signals in vLLM (EngineCore sends shutdown token to itself via PAIR)

---

## 5. Message Framing

ZMQ sends and receives **messages**, not byte streams. A message is an atomic unit —
either the whole message arrives or nothing does (no partial reads unlike TCP).

Messages can be **multipart** — multiple frames sent as an atomic group:

```python
socket.send_multipart([b"frame1", b"frame2", b"frame3"])

# receiver gets all frames or none
frames = socket.recv_multipart()
```

This is used in vLLM's ROUTER pattern — the first frame is the identity (which client
sent this), subsequent frames are the actual payload. ROUTER uses the identity frame
to route replies back to the correct client.

---

## 6. Serialization: What Goes Inside the Message

ZMQ is payload-agnostic — it sends bytes. You choose the serialization format.

```
Format        Library           Typical use
──────────────────────────────────────────────────────────
msgpack       msgspec.msgpack   vLLM — compact binary, 2-10x faster than JSON
pickle        Python stdlib     Quick prototyping — not safe across languages
protobuf      google.protobuf   Cross-language, schema-enforced, large ecosystems
JSON          stdlib json       Human-readable, slow, verbose
```

**Why vLLM uses msgpack over pickle:**
- pickle is Python-only and can execute arbitrary code on deserialization (security risk)
- msgpack is compact, fast, and language-agnostic
- msgspec's msgpack implementation is ~10x faster than the standard msgpack library
  because it uses Rust under the hood and avoids intermediate Python object creation

**What vLLM actually serializes over ZMQ:**
```
Scheduler → GPU Worker:
  {
    request_ids:   ["req_42", "req_43"],
    token_ids:     [[1024, 892], [341, 77]],   # Python int lists, not tensors
    block_table:   [[3, 7], [1, 5]],           # KV block assignments
    is_prefill:    [False, False]
  }

GPU Worker → Scheduler (outputs):
  {
    request_id:    "req_42",
    output_token:  8921,
    finish_reason: None   # or "stop", "length"
  }
```

Tensors never cross the ZMQ boundary. Only metadata — integer IDs, small lists.
The KV cache tensors live on GPU and are referenced by block IDs, not copied.

---

## 7. Internals: What Makes ZMQ Fast

### Lock-free queues
ZMQ's internal send/receive queues use lock-free ring buffers (based on the LMAX
Disruptor pattern). No mutex contention between producer and consumer threads.

### Batch sending (I/O threads)
ZMQ has a background I/O thread pool (configurable via `zmq.Context(io_threads=N)`).
Your application thread enqueues the message and returns immediately. The I/O thread
drains the queue and calls `sendmsg()` asynchronously. For high-frequency small
messages (like per-iteration scheduling decisions), this matters.

### Zero-copy for large messages
For messages above a threshold, ZMQ can avoid copying the payload — it passes a
reference to the buffer to the OS. vLLM has a separate `tensor_ipc.py` for
true zero-copy tensor sharing using shared memory, bypassing ZMQ for the tensor data
itself (see Section 9).

---

## 8. HWM — High Water Mark (Backpressure)

Every ZMQ socket has a **high water mark** — the maximum number of messages it will
queue before applying backpressure.

```python
socket.setsockopt(zmq.SNDHWM, 1000)   # max 1000 messages in send queue
socket.setsockopt(zmq.RCVHWM, 1000)   # max 1000 messages in receive queue
```

When the send queue is full:
- PUSH socket: `send()` blocks (or raises if NOBLOCK flag set)
- PUB socket: drops the message silently

This is ZMQ's flow control mechanism. Setting HWM=0 means unlimited queuing —
dangerous, can OOM if producer is faster than consumer.

**In inference serving context:**
If the GPU worker is slower than the scheduler (e.g. during a heavy prefill),
the ZMQ queue absorbs the burst. If it fills up, the scheduler blocks — which is
actually correct behavior (stop scheduling if GPU can't keep up).

---

## 9. ZMQ vs Alternatives — When to Use What

```
Mechanism          Latency    Throughput   Complexity   When to use
──────────────────────────────────────────────────────────────────────────
inproc (threads)   ~50ns      very high    low          same process, threads
                               (no copy)
ipc (ZMQ)          ~2-5μs     high         low          same host, processes
                               (kernel buf)
TCP localhost       ~10-20μs   high         low          same host, need TCP compat
Shared memory      ~100ns     very high    high         same host, large data
(mmap/POSIX shm)               (zero copy)               tensors, buffers
TCP remote         ~100μs+    medium       medium       multi-node
Kafka/RabbitMQ     ~1ms+      high         high         persistence, replay needed
```

**ZMQ sweet spot:** same-host inter-process messaging where you need patterns
(fan-out, routing) and don't want to run a broker. Latency in the low microseconds,
no external dependency.

**ZMQ is wrong when:**
- You need message persistence / replay → use Kafka
- You need cross-language service mesh → use gRPC
- You're passing large tensors → use shared memory + ZMQ just for the signal

---

## 10. vLLM Architecture — ZMQ Topology

```
                    [Client / API Server]
                           │
                     zmq.ROUTER (tcp://)
                           │
              ┌────────────▼────────────┐
              │      EngineCore         │  CPU process
              │      (Scheduler)        │
              │                         │
              │  input_socket  (ROUTER) │◀── client requests
              │  output_socket (ROUTER) │──▶ client responses
              │  coord_socket  (PAIR)   │◀──▶ coordination
              └────────────┬────────────┘
                           │
                    zmq PUSH (ipc://)
                           │
              ┌────────────▼────────────┐
              │     GPU Worker          │  CPU process, one per GPU
              │                         │
              │  builds tensors         │
              │  dispatches CUDA kernels│
              │           ↓             │
              │         [GPU]           │
              └─────────────────────────┘
```

EngineCore (`core.py`) uses `msgspec.msgpack` to encode the schedule output,
sends it via PUSH socket over `ipc://`. GPU worker decodes it, builds
`torch.tensor()` objects, calls `model.forward()`.

Results (output token IDs) come back via a separate PULL socket in the other direction.

---

## 11. Poller — Watching Multiple Sockets Simultaneously

A ZMQ socket's `.recv()` blocks until a message arrives. When a process owns
multiple sockets, blocking on one means missing messages from the others.

`zmq.Poller` solves this — it asks the OS: "which of my registered sockets
have data ready right now?" and returns immediately.

```python
poller = zmq.Poller()
poller.register(notify_socket, zmq.POLLIN)   # watch for incoming messages
poller.register(result_socket, zmq.POLLIN)

ready = dict(poller.poll(timeout=0))         # timeout=0 = non-blocking

if notify_socket in ready:
    slot_idx = notify_socket.recv()          # guaranteed not to block
if result_socket in ready:
    result  = result_socket.recv()
```

**`timeout` values:**

| Value | Behaviour |
|---|---|
| `0` | Return immediately — non-blocking check |
| `N` (ms) | Wait up to N milliseconds |
| `-1` / `None` | Block forever until at least one socket is ready |

**How it works under the hood:**

`poller.poll()` calls the OS `select()` / `epoll()` syscall on all registered
file descriptors simultaneously. The OS wakes the process when any one of them
has data — no busy-waiting.

**Usage pattern in our scheduler:**

```
step():
  1. poll(timeout=0) on notify_socket  ← non-blocking drain of all tokenizer messages
  2. _schedule()                        ← decide what to run
  3. dispatch_socket.send(batch)        ← send to GPU
  4. result_socket.recv()               ← block here — always need GPU result before next step
```

Step 1 is non-blocking (drain everything queued from tokenizer).
Step 4 is blocking (no point proceeding without GPU output).
The Poller only appears in step 1 — where we need "check without blocking."

**Without Poller — the deadlock:**

```python
# WRONG: blocks on notify_socket even if GPU responded first
data   = notify_socket.recv()
result = result_socket.recv()   # never reached if GPU responds first
```

Poller turns that into a safe, ordered check.

---

## 12. Staff+ Interview Angles

**Q: Why not use a database or Redis as the queue between scheduler and GPU worker?**

Redis adds a network hop (even on localhost), serialization overhead, and a
dependency on an external service. ZMQ is in-process, microsecond latency, zero
dependencies. For a system making scheduling decisions every ~5ms (200 decode
steps/second), the difference between 2μs ZMQ and 500μs Redis round-trip is material.

**Q: Why not just use Python multiprocessing.Queue?**

`multiprocessing.Queue` uses a pipe under the hood with pickle serialization and
acquires locks on every put/get. ZMQ's lock-free queues and msgpack are faster.
More importantly, ZMQ gives you socket patterns (fan-out to multiple workers,
ROUTER identity-based routing) that multiprocessing.Queue doesn't have.

**Q: How does vLLM avoid copying tensor data over ZMQ?**

It doesn't send tensors over ZMQ at all. KV cache is pre-allocated on GPU at startup.
The scheduler sends block IDs (integers) over ZMQ. The GPU worker uses those IDs
as indices into the pre-allocated HBM region. For CPU tensors that do need sharing
(e.g. input embeddings in some configurations), vLLM uses `tensor_ipc.py` which
uses POSIX shared memory (`shm_open`) — ZMQ just sends the shared memory handle
(a file descriptor or name), not the tensor bytes.

**Q: What happens if the GPU worker crashes?**

ZMQ's PUSH socket detects the broken connection and buffers messages up to HWM.
When the worker restarts and reconnects, ZMQ automatically re-delivers buffered
messages. vLLM handles this at the application level by restarting the worker
process and re-initializing the GPU state — ZMQ's reconnection handles the
transport layer transparently.

**Q: inproc vs ipc — when does the choice matter?**

`inproc://` requires both endpoints to be in the same process — threads, not processes.
No kernel involvement, just a memory copy. `ipc://` crosses a process boundary via
a Unix domain socket — goes through the kernel but stays on the same host.
vLLM uses `ipc://` because Scheduler and GPU Worker are separate processes (separate
address spaces, separate GIL domains — that's the whole point of multi-process serving).
