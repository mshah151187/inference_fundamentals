"""
ShmRingBuffer — broadcast single-producer multi-consumer ring buffer
over POSIX shared memory.

WHY THIS EXISTS
───────────────
Normal ZMQ IPC transfer (tokenizer → scheduler) involves 4 copies:
  1. Python object → pickle bytes           (user space, serialization)
  2. pickle bytes  → kernel socket buffer   (user → kernel, syscall)
  3. kernel buffer → scheduler heap         (kernel → user, syscall)
  4. bytes         → Python object          (user space, deserialization)

With this ring buffer, step 2 and 3 disappear. The tokenizer writes
serialized bytes directly into shared physical RAM pages. The readers
read from those same pages — no kernel socket buffer in between.

SHARED MEMORY LAYOUT
────────────────────
N_SLOTS slots packed sequentially. Each slot:

  bytes 0 .. N_READERS-1 : one flag byte per reader  (0=consumed, 1=data ready)
  bytes N_READERS .. N_READERS+3 : uint32 payload size, little-endian
  bytes N_READERS+4 .. end : payload (up to MAX_DATA_BYTES)

Example with N_READERS=2:

  byte 0  : flag_reader0
  byte 1  : flag_reader1
  bytes 2-5 : uint32 size
  bytes 6+  : payload

SYNCHRONIZATION
───────────────
Single producer (tokenizer), N consumers (schedulers/workers). No locks needed.

Each reader owns exactly one flag byte. No two processes ever write the same byte,
so no atomic operation or mutex is required.

  Writer:
    1. Spin until ALL N reader flags == 0   (every reader has consumed the slot)
    2. Write size + payload into slot
    3. Set ALL N reader flags = 1           ← flags set BEFORE ZMQ notify
    4. ZMQ PUB send(slot_idx)              ← syscall acts as memory barrier
                                             all subscribers receive the same msg

  Reader (reader_id = r):
    1. ZMQ SUB recv()                      ← wakes when writer publishes
    2. Read size + payload from slot
    3. Set flag[r] = 0                     ← release only MY flag; no other reader
                                             touches this byte → no atomic needed

The ZMQ send/recv syscall is a full memory barrier on x86: any write before
send() (steps 2+3) is visible to the reader after recv() returns.

WHY NO LOCKS:
  flag[r] is written by exactly two entities:
    - Writer sets it 0→1 (after writing payload)
    - Reader r sets it 1→0 (after reading payload)
  These two directions never race: writer waits until flag[r]==0 before writing,
  reader waits for the ZMQ notification (which comes after all flags are set to 1).
  No two processes ever write flag[r] at the same time.
"""

import struct
import time
from multiprocessing.shared_memory import SharedMemory

# ── Ring buffer parameters ────────────────────────────────────────────────────

N_READERS      = 2             # number of consumer processes (broadcast fan-out)
N_SLOTS        = 128           # number of slots; 128 slots at 180 QPS = ~0.7s of buffer
MAX_DATA_BYTES = 16_384        # 16 KB per slot; our TokenizedRequest is ~1.5 KB

# Header: N_READERS flag bytes + 4 size bytes
_HEADER_SIZE   = N_READERS + 4
SLOT_SIZE      = _HEADER_SIZE + MAX_DATA_BYTES
SHM_TOTAL_SIZE = N_SLOTS * SLOT_SIZE

# ── Shared addresses (used by both writer and readers) ────────────────────────

SHM_NAME    = "tok_to_sched_shm"
NOTIFY_ADDR = "ipc:///tmp/shm_tok_notify.ipc"   # writer PUB, readers SUB

# ── Slot field offsets ────────────────────────────────────────────────────────

# flag for reader r is at slot_off + r   (r in 0..N_READERS-1)
_SIZE_OFF = N_READERS          # uint32 payload size starts right after flag bytes
_DATA_OFF = _HEADER_SIZE       # payload starts after all header bytes


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

def create_shm() -> SharedMemory:
    """
    Create and zero-initialize the shared memory region.
    Must be called once from main.py BEFORE spawning any child processes
    so all children can attach by name.
    All flag bytes start at 0 (free) after zero-init.
    """
    shm = SharedMemory(name=SHM_NAME, create=True, size=SHM_TOTAL_SIZE)
    shm.buf[:SHM_TOTAL_SIZE] = b'\x00' * SHM_TOTAL_SIZE
    print(f"[ShmRingBuffer] created '{SHM_NAME}' "
          f"({N_SLOTS} slots × {SLOT_SIZE} B = {SHM_TOTAL_SIZE / 1024:.1f} KB), "
          f"broadcast to {N_READERS} readers")
    return shm


# ─────────────────────────────────────────────────────────────────────────────
# Writer (runs in Tokenizer process)
# ─────────────────────────────────────────────────────────────────────────────

class ShmRingBufferWriter:
    """
    Tokenizer uses this to write serialized TokenizedRequest bytes into the
    next free ring buffer slot, then PUBs the slot index so all N_READERS
    subscribers wake up simultaneously.

    The notify_socket must be a ZMQ PUB socket (not PUSH — PUB broadcasts
    to all connected SUB sockets, PUSH would only deliver to one).
    """

    def __init__(self, notify_socket):
        self.shm       = SharedMemory(name=SHM_NAME, create=False)
        self.buf       = self.shm.buf
        self.write_idx = 0
        self._notify   = notify_socket   # ZMQ PUB socket owned by caller

    def write(self, data: bytes) -> float:
        """
        Write data into the next slot and broadcast slot_idx to all readers.
        Spins if ANY reader has not yet consumed the slot (back-pressure).
        Returns write duration in milliseconds.
        """
        assert len(data) <= MAX_DATA_BYTES, (
            f"payload {len(data)} B exceeds MAX_DATA_BYTES {MAX_DATA_BYTES} B"
        )
        t0       = time.perf_counter()
        slot_off = self.write_idx * SLOT_SIZE

        # Spin until ALL reader flags == 0 (every reader has consumed this slot).
        # The slowest reader determines back-pressure — writer cannot advance
        # past a slot until every reader has cleared its flag.
        spins = 0
        while any(self.buf[slot_off + r] == 1 for r in range(N_READERS)):
            time.sleep(0.0001)
            spins += 1
            if spins % 1000 == 0:
                laggards = [r for r in range(N_READERS) if self.buf[slot_off + r] == 1]
                print(f"[ShmWriter] slot {self.write_idx} blocked on readers "
                      f"{laggards} ({spins} spins)")

        # Write size then payload into the slot body.
        struct.pack_into('<I', self.buf, slot_off + _SIZE_OFF, len(data))
        self.buf[slot_off + _DATA_OFF : slot_off + _DATA_OFF + len(data)] = data

        # Set ALL reader flags = 1 BEFORE the ZMQ send.
        # The ZMQ PUB send() syscall is the memory barrier: readers are
        # guaranteed to see flag[r]==1 and the valid payload after SUB recv().
        for r in range(N_READERS):
            self.buf[slot_off + r] = 1

        # PUB broadcasts slot_idx to every connected SUB socket simultaneously.
        self._notify.send(bytes([self.write_idx]))

        self.write_idx = (self.write_idx + 1) % N_SLOTS
        return (time.perf_counter() - t0) * 1000   # ms

    def close(self):
        self.buf.release()
        self.shm.close()


# ─────────────────────────────────────────────────────────────────────────────
# Reader (runs in each consumer process — one instance per reader_id)
# ─────────────────────────────────────────────────────────────────────────────

class ShmRingBufferReader:
    """
    Each consumer process creates one ShmRingBufferReader with its own reader_id
    (0 to N_READERS-1). The notify_socket must be a ZMQ SUB socket subscribed
    to all messages ("").

    The reader_id determines which flag byte this reader owns. Only this reader
    ever clears flag[reader_id] — so no lock or atomic is needed.

    Usage in consumer loop:
        slot_idx  = int.from_bytes(notify_socket.recv(), 'little')
        data, ms  = ring_reader.read(slot_idx)
        request   = TokenizedRequest.FromString(data)
    """

    def __init__(self, reader_id: int):
        assert 0 <= reader_id < N_READERS, (
            f"reader_id {reader_id} out of range [0, {N_READERS})"
        )
        self.reader_id = reader_id
        self.shm       = SharedMemory(name=SHM_NAME, create=False)
        self.buf       = self.shm.buf

    def read(self, slot_idx: int) -> tuple[bytes, float]:
        """
        Read and consume slot_idx.
        The ZMQ SUB recv() guarantees flag[reader_id]==1 and payload is valid.
        After reading, clears only flag[reader_id] — does not touch other readers' flags.
        Returns (data, read_duration_ms).
        """
        t0       = time.perf_counter()
        slot_off = slot_idx * SLOT_SIZE

        size = struct.unpack_from('<I', self.buf, slot_off + _SIZE_OFF)[0]
        data = bytes(self.buf[slot_off + _DATA_OFF : slot_off + _DATA_OFF + size])

        # Clear only MY flag. Other readers' flags are untouched.
        # Writer will not reuse this slot until all N flags reach 0.
        self.buf[slot_off + self.reader_id] = 0

        return data, (time.perf_counter() - t0) * 1000   # data + read_ms

    def close(self):
        self.buf.release()
        self.shm.close()
