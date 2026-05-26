# Profiling Tools Overview

---

## PyTorch Profiler (Kineto)

Within PyTorch, the `torch.profiler`, based on the Kineto open source project, provides operator-level breakdowns of CPU and CUDA/GPU runtimes. In addition, it can record input shapes and take memory snapshots using simple Python context managers. The PyTorch profiler can capture detailed timeline traces and hardware counters across training and inference workloads using NVTX ranges to align the events. It provides end-to-end observability from Python code down to the CUDA kernels — and even provides performance tips for common issues like data-loading stalls and inefficient CUDA code.

---

## Nsight Systems (nsys)

For system-wide correlation, including CPU threads, GPU kernels, OS events, I/O, and interconnect traffic, NVIDIA Nsight Systems produces a unified timeline view. Its GUI and CLI reports can merge NVTX zones, Python call stacks, and CUDA streams across multiprocess and multinode runs. This makes it easy to spot where I/O and synchronization stalls might be impacting compute performance.

---

## Nsight Compute (ncu)

Complementing Nsight Systems is NVIDIA Nsight Compute for per-kernel analysis. Nsight Compute collects detailed hardware metrics such as occupancy, memory bandwidth, and SM utilization. It can even generate roofline charts mapped to source code. Nsight Compute helps answer *why* a particular kernel is slow (e.g., memory bound, low occupancy) after other higher-level tools identify *which* kernels are the hotspots.

---

## PyTorch Memory Profiler

PyTorch also includes a memory profiler, which you can enable with `profile_memory=True` in `torch.profiler`. The PyTorch memory profiler breaks down peak and cumulative GPU memory allocations per operation. This reveals memory usage hotspots that might otherwise go unnoticed.

---

## Linux perf

On the host side, Linux's `perf` tool can sample CPU hardware counters, including cycles, instructions, and cache misses — and unwind full C/C++ and Python call graphs. Starting with `perf sched`, you can see when CPU threads sit idle due to I/O or thread scheduling/synchronizing. This uncovers bottlenecks in data preprocessing loops, Python's GIL, or synchronization that can starve the GPU.

---

## Holistic Trace Analysis (HTA)

Meta's open source Holistic Trace Analysis (HTA) tool ingests PyTorch profiler traces to help diagnose multi-GPU bottlenecks. With HTA, one can visualize distributed training timelines with NVTX ranges alongside CUDA kernel traces. By drilling into memory allocation patterns over time, you can identify periods of idle GPU — including when GPUs are waiting on each other.

> Note: TensorBoard's PyTorch trace visualization plugin is deprecated. Instead, use Perfetto for timeline viewing and Meta's HTA for distributed trace analysis.

---

## Chrome Trace and Perfetto Viewer

For web-based exploration of large PyTorch profiler trace files, use the **Perfetto UI** (`https://ui.perfetto.dev`). It loads JSON traces and lets you interactively explore timeline views and flame charts, with fine-grained filtering and SQL queries on the trace data — down to the submillisecond level. Perfetto is ideal for sharing profile results between members of your organization for cross-team analysis.

> Note: Chrome's legacy trace viewer (`chrome://tracing`) is deprecated — prefer the Perfetto web UI.

---

## TorchEval

TorchEval lets you log and monitor model throughput, latency, and quality metrics alongside training and evaluation metrics — all within a unified interface. It is PyTorch's official metrics library and provides a simple API for end-to-end performance and quality metrics, making it easy to plug into training loops and integrate across distributed environments.

---

## ExecuTorch

For embedded, mobile, and edge devices, the ExecuTorch project allows profiling, visualizing, and debugging PyTorch models in lightweight runtime environments like Meta glasses. ExecuTorch has a small, dynamic memory footprint and supports Linux, iOS, Android, and embedded systems. Hugging Face supports ExecuTorch through its Optimum ExecuTorch project, which makes this environment easy to integrate if you're already using the Hugging Face ecosystem.

---

## Tool Hierarchy (Quick Reference)

| Tool | Level | Primary Question |
|------|-------|-----------------|
| torch.profiler | Operator | Which ops are hot? Where does GPU time go? |
| Nsight Systems | System | Where are the gaps, stalls, H2D transfers? |
| Nsight Compute | Kernel | Why is this specific kernel slow? (roofline) |
| PyTorch Memory Profiler | Memory | Which ops allocate the most GPU memory? |
| Linux perf | CPU | Is the CPU itself the bottleneck? (GIL, cache misses) |
| HTA | Distributed | Which GPU is the straggler in multi-GPU runs? |
| Perfetto | Visualization | Visual timeline exploration of profiler traces |
