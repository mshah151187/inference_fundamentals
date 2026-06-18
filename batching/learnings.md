# Batching Project — Learnings

---

## 1. Assuming max_seq_len is bad

- Every slot reserves `max_seq_len × N bytes` upfront
- Real requests use 200 tokens, slot holds 1024 → 80% wasted HBM per slot
- Fewer concurrent slots → queue fills → queue_wait grows → TTFT suffers
- Wasted HBM = wasted money (you're paying for capacity that does nothing)
- Lesson: allocate for actual usage, not worst-case; release memory as soon as the request finishes

---

## 2. Looking for contiguous memory is bad

- Naive KV cache needs one contiguous block of `max_seq_len × N bytes`
- HBM fragments over time as requests complete and return slots of varying sizes
- Even with 30 GB free, no single contiguous 36 MB chunk → new request waits
- Solution: pages (PagedAttention) — `ceil(prompt_len / page_size)` scattered pages, any layout works
- Fragmentation becomes irrelevant; free pages are a pool, not a map

---

## 3. Max 1 prefill per schedule step

- Prefill is compute-bound (GEMM); decode is memory-bandwidth-bound (GEMV)
- Both share the same GPU in each scheduler step
- One large prefill (e.g. 430 tokens) dominates the step → decode requests stall for that full duration → ITL spike
- Limiting to 1 prefill per step caps the worst-case interference
- Prefill still completes in one step; decode requests resume next step
- Better fix: chunked prefill (split 430 tokens into 128-token chunks across steps) — smooths interference without sacrificing prefill latency
