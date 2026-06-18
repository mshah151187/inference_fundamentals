# IPC Data Transfer — What Exactly Happens

---

## 1. Process Isolation: Why Transfer Is Non-Trivial

The OS gives every process its own **virtual address space**. A 64-bit Linux process sees
addresses from `0x0` to `0x7fffffffffff` — but these are virtual, not physical.

The CPU's MMU (Memory Management Unit) translates virtual → physical on every memory access
using a **page table** owned by the OS. Two processes have completely separate page tables.
Process A's virtual address `0x7f000000` and Process B's virtual address `0x7f000000` point
to different physical RAM pages. The hardware enforces this — there is no way for process A
to read process B's memory by guessing addresses.

```
Process A                        Physical RAM                  Process B
─────────────────────            ─────────────                 ─────────────────────
virtual: 0x7f000000 ──► MMU ──► page at 0x3a000  (A's data)
virtual: 0x7f010000 ──► MMU ──► page at 0x3b000

                                 page at 0x4c000  (B's data)  ◄── MMU ◄── virtual: 0x7f000000
                                 page at 0x4d000               ◄── MMU ◄── virtual: 0x7f010000
```

This isolation is the core reason data transfer between processes is expensive — you have to
explicitly cross the boundary.

---

## 2. The Kernel as the Bridge

The only way to cross the process boundary is through the **kernel** — the privileged layer
that owns all physical memory and manages the page tables.

When a process wants to send data to another process, it calls a system call (`send`, `write`,
`sendmsg`). This triggers a **CPU mode switch**: user mode → kernel mode. The kernel now runs
with full access to all physical memory. It can read from process A's pages and write to
process B's pages — or into a shared kernel buffer that process B will later read from.

```
User mode (Process A)    │    Kernel mode         │    User mode (Process B)
─────────────────────    │    ────────────         │    ─────────────────────
data in heap             │                         │
  ↓                      │                         │
socket.send()   ─────────┼──► kernel copies ───────┼──► socket.recv()
  [syscall]              │    data into             │      [syscall]
                         │    kernel buffer         │
                         │         ↓                │
                         │    kernel copies ─────────┼──► data in heap
                         │    to recv buffer        │
```

This mode switch is not free — there is overhead every time you cross the user/kernel boundary.

---

## 3. Normal ZMQ Transfer — Step by Step (Tokenizer → Scheduler)

Let's trace what happens when the tokenizer sends token IDs to the scheduler over a ZMQ
PUSH/PULL socket using IPC transport (`ipc://`).

---

### Step 1 — Serialize: Python Object → Bytes (COPY 1)

**Where:** Tokenizer process, user space.

`token_ids` is a Python list — a high-level Python structure with type metadata, reference
counts, and object pointers. The kernel has no concept of any of this. It only understands
raw bytes. So before anything can cross the user→kernel boundary, the data must be flattened
into a plain byte stream.

This serialization step is NOT a ZMQ requirement — it is a Python-object cost. If the data
were already a contiguous buffer (numpy array, bytearray), ZMQ could read it directly via
the buffer protocol and this copy would be skipped entirely. It happens here because we have
a Python list.

```
token_ids = [15496, 11, 338, ...]     ← Python list in tokenizer's heap
                                         (objects, pointers, refcounts — not raw bytes)

data = pickle.dumps(token_ids)        ← COPY 1
                                         new bytes object created in tokenizer's heap
                                         data now exists twice in memory
```

---

### Step 2 — System Call: Hand Control to the Kernel

**Where:** User space → kernel mode transition.

`socket.send(data)` is a system call. The CPU switches from **user mode** (restricted) to
**kernel mode** (privileged). The tokenizer process is now paused — the kernel is running
on its behalf. The kernel has full visibility into all physical memory, including the bytes
object sitting in the tokenizer's heap.

```
socket.send(data)     ← system call boundary
                         user mode → kernel mode
                         tokenizer process pauses here
```

---

### Step 3 — Kernel Copies Bytes into Socket Buffer (COPY 2)

**Where:** Inside the kernel.

The kernel reads the bytes from the tokenizer's user-space pages and copies them into a
kernel-owned socket buffer (a region of memory the kernel manages that neither process
can touch directly). This crossing from user pages → kernel pages is the fundamental cost
of IPC — the kernel must make a private copy so it can safely transfer ownership.

```
┌──────── KERNEL ─────────────────────────────────────────────────┐
│  memcpy(kernel_socket_buf, data, len)    ← COPY 2               │
│  (tokenizer's user pages → kernel-owned socket buffer pages)    │
└─────────────────────────────────────────────────────────────────┘
```

After this copy, the kernel has everything it needs. It returns to user mode — the
tokenizer's `send()` call returns.

---

### Step 4 — Kernel Transfers Buffer to Receiver Side

**Where:** Inside the kernel (IPC / unix socket path).

For a ZMQ IPC socket (unix domain socket on the same machine), the kernel does not need
to go over a network. It moves the pointer to the socket buffer from the sender's receive
queue to the receiver's receive queue. On Linux, this can avoid a second kernel-to-kernel
copy — the receiver will read from the same kernel buffer pages the sender wrote into.

```
┌──────── KERNEL ─────────────────────────────────────────────────┐
│  kernel_socket_buf pointer moved to scheduler's recv queue      │
│  (no extra copy on same-machine unix socket — pointer transfer) │
└─────────────────────────────────────────────────────────────────┘
```

---

### Step 5 — System Call: Scheduler Asks for Data

**Where:** Scheduler process, user space → kernel mode transition.

The scheduler calls `socket.recv()`, which is another system call. CPU switches to kernel
mode again. The kernel checks the receive queue — the data is ready. It now needs to copy
the bytes from the kernel socket buffer into the scheduler's user-space memory, because the
scheduler cannot access kernel pages directly.

```
raw = socket.recv()   ← system call boundary
                         user mode → kernel mode
                         scheduler process pauses here
```

---

### Step 6 — Kernel Copies Bytes into Scheduler's Heap (COPY 3)

**Where:** Inside the kernel, crossing back to user space.

The kernel copies bytes from the kernel socket buffer into a new buffer in the scheduler's
user-space heap. This is the mirror of COPY 2 — the data crosses the kernel→user boundary.
After this copy, the scheduler's `recv()` call returns with the raw bytes.

```
┌──────── KERNEL ─────────────────────────────────────────────────┐
│  memcpy(scheduler_user_buf, kernel_socket_buf, len)  ← COPY 3  │
│  (kernel-owned socket buffer pages → scheduler's user pages)    │
└─────────────────────────────────────────────────────────────────┘

raw = bytes(...)      ← raw bytes now in scheduler's heap
```

---

### Step 7 — Deserialize: Bytes → Python Object (COPY 4)

**Where:** Scheduler process, user space.

The scheduler has raw bytes but needs a usable Python structure. `pickle.loads` reconstructs
the Python list — allocating new Python objects and copying the raw values into them. This is
the mirror of Step 1. Again, not a ZMQ cost — a Python-object cost. If the receiver only
needed raw bytes (e.g., to forward to the GPU), this step could be skipped.

```
token_ids = pickle.loads(raw)    ← COPY 4
                                    new Python list created in scheduler's heap
                                    raw bytes and Python list now both in memory
```

---

**Total: 4 copies of the data.**
- Copy 1: Python list → pickle bytes (user space — Python object cost, skippable with buffer types)
- Copy 2: pickle bytes → kernel socket buffer (user → kernel — fundamental IPC cost)
- Copy 3: kernel socket buffer → scheduler heap (kernel → user — fundamental IPC cost)
- Copy 4: bytes → Python list (user space — Python object cost, skippable if receiver uses raw bytes)

Plus 2 syscalls (mode switches). Copies 2 and 3 are unavoidable in normal IPC — only shared
memory eliminates them.

For small data (1–4 KB token ID list) this is microseconds. For large data (9 MB image
tensor, 100 KB batch metadata) the copies add up and become measurable latency.

---

## 4. What Shared Memory Changes

Shared memory bypasses the kernel-buffer step entirely. The OS maps the **same physical pages**
into both processes' virtual address spaces. After the initial setup, neither process needs the
kernel to move data — both read and write directly to the same RAM.

```
Setup (done once at startup):
  shm = SharedMemory(name="batch_data", create=True, size=1_000_000)

TOKENIZER                        Physical RAM                 SCHEDULER
──────────────────              ─────────────                ──────────────────
virtual: 0x7f000000 ──► MMU ──► [shm pages] ◄── MMU ◄── virtual: 0x7e000000
```

---

### Step 1 — Setup: Map Shared Memory into Both Processes (One-Time)

**Where:** Both processes, user space, at startup.

Before any transfer can happen, both processes must ask the OS to map the same physical
pages into their virtual address spaces. The writer creates the shared memory region; the
reader attaches to it by name. After this, both processes have a direct virtual address
that points to the same RAM — no kernel involvement needed for any subsequent read or write.

```
# Tokenizer (writer) — creates the region
shm = SharedMemory(name="batch_data", create=True, size=1_000_000)

# Scheduler (reader) — attaches to the same region by name
shm = SharedMemory(name="batch_data", create=False)
```

This setup happens once. Every transfer after this costs nothing in kernel setup overhead.

---

### Step 2 — Serialize into Shared Memory: Python Object → Bytes (COPY 1)

**Where:** Tokenizer process, user space, writing directly into shm pages.

Just like ZMQ transfer, `token_ids` is a Python list — it must be converted to bytes before
it can be stored in the raw shared memory buffer. The key difference from ZMQ: instead of
serializing into a private heap buffer first and then copying to the kernel, we serialize
**directly into the shared memory pages**. The destination of this copy is already visible
to the scheduler.

This is still 1 copy, but it is the **only copy** in the entire transfer. There is no
second copy into a kernel buffer, and no third copy out of a kernel buffer.

```
token_ids = [15496, 11, 338, ...]    ← in tokenizer's private heap

pickle.dumps_into(shm.buf, token_ids)   ← COPY 1
                                           serialize directly into shm pages
                                           shm.buf is a memoryview of shared RAM
                                           scheduler can already see these pages
```

---

### Step 3 — Send Handle over ZMQ: Notify the Scheduler (Control Plane)

**Where:** Tokenizer process, user space → kernel mode (briefly, for the tiny handle only).

The scheduler is sitting in `socket.recv()` waiting. It has no way to know the tokenizer
finished writing into shared memory. We still need a notification signal. We send a tiny
handle — just the shm name, offset, and byte count — over ZMQ. This goes through the normal
ZMQ copy path, but the payload is ~30 bytes, not the actual data.

This is the **control plane**. The data plane already happened in Step 2.

```
socket.send(b"batch_42:offset=0:size=1024")
# ~30 bytes over ZMQ — fast, the kernel copies are negligible at this size
# this is purely a "data is ready, here is where to find it" signal
```

---

### Step 4 — Recv Handle: Scheduler Wakes Up

**Where:** Scheduler process, user space, receiving the tiny ZMQ control message.

The scheduler's `recv()` returns with the handle. No data has moved yet — the scheduler
only has the address of the data. The actual token IDs are still sitting in shared memory
where the tokenizer wrote them.

```
handle = socket.recv()    ← returns b"batch_42:offset=0:size=1024"
                             ~30 bytes, not the payload
                             CPU mode switch happened but for negligible data
```

---

### Step 5 — Read Directly from Shared Memory (NO COPY)

**Where:** Scheduler process, user space, reading from shm pages.

The scheduler parses the handle and reads directly from `shm.buf` at the given offset.
Because the tokenizer and scheduler share the same physical pages, this is a normal memory
read — no system call, no kernel involvement, no copy. The scheduler's CPU is reading the
exact bytes the tokenizer wrote in Step 2.

```
offset, size = 0, 1024
token_ids = pickle.loads(shm.buf[offset:offset+size])
#                        ↑
#                        reading from the same physical RAM the tokenizer wrote to
#                        no copy — MMU resolves this virtual address to the same pages
```

The `pickle.loads` here reconstructs the Python list from the shared bytes. This is the
same deserialization cost as ZMQ Step 7 — unavoidable if you need a Python object on the
other end. If the scheduler only needed raw bytes (e.g., to forward to the GPU), even this
could be skipped.

---

**Total: 1 copy of the data** (write into shm in Step 2). Reader accesses the same bytes in place.

Copies 2 and 3 from normal ZMQ (user→kernel, kernel→user) are completely gone. The kernel
is only involved for the tiny 30-byte handle in Steps 3 and 4, which is negligible.

The ZMQ socket still exists but its role changes: it carries the **control plane** (notification
+ address), not the **data plane**. Data moves through shared memory; coordination moves
through ZMQ.

---

## 5. Copy Count Comparison

```
Mechanism              Copies    Syscalls    Serde    Notes
─────────────────────  ──────    ────────    ─────    ─────────────────────────────────
ZMQ IPC (current)        4          2          2      serialize, user→kernel, kernel→user, deserialize
ZMQ TCP (localhost)      4–5        2          2      extra loopback network stack copies
Pipe                     4          2          2      same as ZMQ IPC
Shared memory            1          0*         1      write once; reader reads in place
                                                      * 0 kernel copies in data path
                                                        (2 syscalls for the tiny handle)
```

---

## 6. The Synchronization Problem Shared Memory Introduces

ZMQ gives you synchronization for free: `recv()` blocks until data is ready. With shared
memory, both processes see the same bytes — but there's no automatic signal for "data is
written, you can read now" or "reader is done, you can overwrite."

You must add synchronization yourself:

**Option A — Flag byte (simplest)**
```
shm layout: [1 byte flag][payload bytes...]

Writer:
  write payload into shm
  shm.buf[0] = 1          ← set flag: data ready

Reader:
  while shm.buf[0] == 0:  ← spin until flag set
      pass
  read payload from shm
  shm.buf[0] = 0          ← clear flag: slot free
```

**Option B — Ring buffer (vLLM ShmRingBuffer style)**
```
shm layout: [N slots × (written_flag + per_reader_flags + data)]

Writer: writes into slot N, sets written_flag=1
Reader: spins on written_flag; on see=1, reads data, sets my_reader_flag=1
Writer: slot free when written_flag=1 AND all reader_flags=1
ZMQ PUB: sends 1-byte wake signal so readers can sleep instead of spin
```

**Option C — File lock (vLLM-Omni SharedMemoryConnector)**
```
fcntl.flock(lock_file, LOCK_EX)   ← exclusive lock on /dev/shm/req_42.lock
write to shm
fcntl.flock(lock_file, LOCK_UN)   ← release

Reader:
fcntl.flock(lock_file, LOCK_EX)   ← blocks until writer releases
read from shm
```

The choice is a latency vs simplicity tradeoff:
- Flag byte: lowest latency, only works for one reader
- Ring buffer: works for N readers, vLLM production choice
- File lock: simplest to implement, highest latency (kernel involvement on every lock)

---

## 7. Why This Matters for Inference

In an inference pipeline, every scheduler step involves:
1. Scheduler → GPU Worker: batch metadata (which slots, seq lens, token IDs)
2. GPU Worker → Detokenizer: generated token IDs
3. These transfers are ON THE CRITICAL PATH — they add directly to step latency (ITL)

Eliminating copies 2 and 3 (kernel round-trip) for the Scheduler → GPU Worker path removes
kernel involvement from the data path entirely. For a 50 KB batch metadata payload:
- Normal ZMQ: ~5 μs kernel copy in + ~5 μs kernel copy out = ~10 μs
- Shared memory: ~5 μs write into shm + ~0 μs read (same pages) = ~5 μs

Small at our scale. Large at production scale (bigger batches, larger metadata, N workers all
needing the same data — 1 write vs N copies).
