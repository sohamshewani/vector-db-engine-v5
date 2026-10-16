# vector-db-engine
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![CI](https://github.com/sohamshewani/vector-db-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/sohamshewani/vector-db-engine/actions) [![Security: Hardened](https://img.shields.io/badge/Security-Hardened-blue.svg)](https://github.com/sohamshewani/vector-db-engine) [![Stars](https://img.shields.io/github/stars/sohamshewani/vector-db-engine?style=flat)](https://github.com/sohamshewani/vector-db-engine)

## Executive Summary & Problem Space

`vector-db-engine` is a high-throughput, low-latency embedded vector database engine engineered from the ground up in modern C (C11). Designed for mission-critical systems requiring sub-millisecond similarity search combined with strict transactional guarantees, it eliminates the architectural bottlenecks typical of managed or garbage-collected runtimes. 

Modern high-dimensional vector search workloads face severe performance cliffs due to cache misses during distance calculations, lock contention on index updates, and memory fragmentation under heavy churn. `vector-db-engine` solves these challenges through:
- **Cache-conscious columnar layouts** (`src/core/column.c`) optimized for SIMD vector execution pipelines (`src/core/vector_exec.c`).
- **Zero-copy memory management** backed by arena and pool allocators (`include/vector_db/allocator.h`) that eliminate system allocator overhead (`malloc`/`free`) on hot paths.
- **Lock-free concurrent bitmap indexing** (`src/core/bitmap_index.c`) allowing real-time filtering without read-side blocking.
- **Adaptive hardware-level compression** (`src/core/compression.c`) scaling from raw FP32 storage to aggressive scalar quantization variants.

---

## Structural ASCII Architecture / Dataflow Diagram

```
 +-------------------------------------------------------------------+
 |                        Client / Application                       |
 +-------------------------------------------------------------------+
                                   |
                                   v
 +-------------------------------------------------------------------+
 |                    Public API Layer (C11)                         |
 |  - include/vector_db/types.h     - include/vector_db/allocator.h  |
 +-------------------------------------------------------------------+
                                   |
         +-------------------------+-------------------------+
         | (Write Path)                                      | (Read / Query Path)
         v                                                   v
 +-------------------------------+   +-------------------------------+
 |      Vector Execution         |   |      Bitmap Index Layer       |
 |  - src/core/vector_exec.c     |   |  - src/core/bitmap_index.c    |
 +-------------------------------+   +-------------------------------+
         |                                                   |
         v                                                   v
 +-------------------------------+   +-------------------------------+
 |       Column Storage          |   |     Compression Engine        |
 |  - src/core/column.c          |   |  - src/core/compression.c     |
 +-------------------------------+   +-------------------------------+
         |                                                   |
         +-------------------------+-------------------------+
                                   |
                                   v
 +-------------------------------------------------------------------+
 |                 Hardware Abstraction & OS Memory                  |
 |       Arena/Pool Allocators & SIMD Vector Instructions            |
 +-------------------------------------------------------------------+
```

---

## Concurrency Model & Memory Invariants

### Thread Safety & Lock Contention Prevention
- **Reader-Writer Isolation:** Read paths execute completely lock-free using epoch-based reclamation and immutable memory views. Write operations serialize through fine-grained partition locks, preventing global engine stalls.
- **Atomic State Transitions:** Index mutations and allocation tracking utilize C11 atomic primitives (`stdatomic.h`) with strict sequential consistency (`memory_order_seq_cst`) where synchronization boundaries are required, dropping to `memory_order_relaxed` for high-frequency internal metrics.

### Memory Ownership & Invariants
1. **Explicit Allocation Boundaries:** Every allocation request is routed through arena or segregated-fit pool allocators (`include/vector_db/allocator.h`). Raw system calls are isolated to arena initialization phases.
2. **Immutability After Commit:** Once a vector block is committed to the columnar storage engine (`src/core/column.c`), its memory layout is immutable. Updates occur via append-only delta logs merged asynchronously by background compaction workers.
3. **Zero-Leak Guarantee:** All resources adhere strictly to RAII-equivalent deterministic cleanup patterns (`vector_arena_destroy()`, `column_free()`). Static analyzers and test suites (`tests/test_invariants.c`) enforce zero-tolerance for buffer overruns and uninitialized reads.

---

## Performance Characteristics & Algorithmic Complexity Table

| Operation | Target Subsystem | Time Complexity | Space Complexity | P99 Latency Est. (1M Vectors) |
| :--- | :--- | :--- | :--- | :--- |
| **Exact Vector Insert** | `src/core/column.c` | $O(1)$ amortized | $O(D)$ where $D =$ dimensions | $< 15\,\mu\text{s}$ |
| **SIMD Distance Calc** | `src/core/vector_exec.c` | $O(D)$ vectorized | $O(1)$ | $< 120\,\mu\text{s}$ |
| **Bitmap Filter Scan** | `src/core/bitmap_index.c` | $O(N / 64)$ | $O(N / 8)$ bytes | $< 50\,\mu\text{s}$ |
| **Compressed Read** | `src/core/compression.c` | $O(D)$ | $O(D / k)$ | $< 200\,\mu\text{s}$ |

---

## Security Architecture & Threat Model

### Input Boundaries & Sanitization
- **Strict Bounds Checking:** All incoming vectors and metadata payloads are validated against strict schema limits defined in `include/vector_db/types.h` before hitting internal memory allocators.
- **Integer Overflow Protection:** Dimension calculations and buffer length derivations incorporate safe-math overflow checks, rejecting malformed allocation requests instantly.

### Memory Corruption Protections
- **Canary Stamping:** Allocator blocks utilize internal boundary canaries to detect stack/heap smashing attacks during fuzz testing.
- **No Unsafe Formatting:** String parsing and log pipelines strictly prohibit unsafe format strings (`sprintf`, unconstrained `strcpy`), enforcing bounded buffers (`snprintf`) across all logging and telemetry paths.

---

## Build, Verification & Benchmarking Recipes

The engine relies on a clean POSIX-compliant `Makefile` with zero external dependencies.

### 1. Compile the Engine, Tests, and Benchmarks
```bash
make clean && make all
```

### 2. Execute Invariant and Unit Test Suites
```bash
make test
```
*(Executes strict memory boundary, concurrency, and invariant tests defined in `tests/test_invariants.c`)*

### 3. Run Micro-Benchmarks
```bash
make bench
```
*(Executes high-performance throughput and latency profiles defined in `benchmarks/bench_engine.c`)*