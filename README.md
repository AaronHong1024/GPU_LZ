# **GZLZ** — GPU-Accelerated LZ77 Compression

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](#license)
[![libCUBWT](https://img.shields.io/badge/libCUBWT-GPU--BWT-blue.svg)](https://github.com/kobolabs/libcubwt)
[![SDSL-lite](https://img.shields.io/badge/SDSL--lite-succinct--data--structures-lightgrey.svg)](https://github.com/simongog/sdsl-lite)

> **Primary implementation and system design by Aaron Hong.**

GZLZ is an advanced GPU-accelerated toolkit designed specifically for efficient suffix array (SA) construction and LZ77 compression algorithms. Leveraging cutting-edge CUDA implementations, GZLZ significantly enhances speed and memory efficiency, making it ideal for bioinformatics and large-scale data compression tasks.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Key Highlights](#key-highlights)
- [Performance Benchmarks](#performance-benchmarks)
- [Understanding the Algorithms](#understanding-the-algorithms)
- [Dependencies](#dependencies)
- [Build Instructions](#build-instructions)
- [Usage](#usage)
- [License](#license)

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/yourusername/GPU_LZ.git
cd GPU_LZ
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Run suffix array profiling
./sa_profiler <input_file>

# Run LZ77 compression
./lz77 <input_file>

```

---

## Key Highlights

* **Optimized GPU Algorithms:** Achieves superior speed over traditional CPU libraries and superiour memory efficiency over existing GPU libraries.
* **Tailored for Bioinformatics:** Specifically optimized for massive genomic datasets.
* **Memory-Efficient Compression:** Stream-based processing suitable for extremely large datasets.

---

## Performance Benchmarks

Tested on NVIDIA A100 GPU (80 GB VRAM):

| Dataset          | Method              | Runtime (ms)    | GPU Peak Memory (MiB) | Speedup vs CPU |
| ---------------- | ------------------- | --------------- | --------------------- | -------------- |
| **32× (1.8 GB)** | libCUBWT (GPU)      | 10,449          | 55,252                | **25.83×**     |
| **32× (1.8 GB)** | **GZLZ (GPU)**      | **96,886**      | **29,356**            | **2.78×**      |
| **32× (1.8 GB)** | SDSL (CPU, 32-core) | 269,883         | —                     | 1.00×          |
| **64× (3.7 GB)** | libCUBWT (GPU)      | —               | —                     | —              |
| **64× (3.7 GB)** | **GZLZ (GPU)**      | **294,324**     | **58,806**            | **2.05×**      |
| **64× (3.7 GB)** | SDSL (CPU, 32-core) | 604,863         | —                     | 1.00×          |

---

## Understanding the Algorithms

### What is a Suffix Array?

A suffix array is a data structure that stores the starting indices of all suffixes of a given string, sorted in lexicographic (dictionary) order. By organizing the string's suffixes this way, it enables highly efficient solutions to common string-processing tasks such as substring search, longest repeated substring detection, and pattern matching. **We compute the Suffix Array using Prefix Doubling**

**Example:**

```
String: "BANANA"
Sorted Suffixes:
0: A
1: ANA
2: ANANA
3: BANANA
4: NA
5: NANA
```

### Prefix Doubling Architecture & Kernel Flow

GZLZ implements an efficient **Prefix Doubling Suffix Array** construction on the GPU, orchestrating multiple CUDA kernels and memory operations to maximize parallel throughput while minimizing memory footprint.
Each iteration doubles the prefix length and re-ranks suffixes using a series of compute, sort, and scan operations. The flow is fully adaptive, monitoring device memory to select either full-GPU or streaming mode.

#### **GPU Kernel/Operation Pipeline**

```mermaid
flowchart TD
    INIT["Init: rank & index arrays (kernel)"] --> PREP["Prep: (rank, next rank) pairs (kernel)"]
    PREP --> SORT["Sort: index[] by (rank, next rank) (thrust::sort_by_key)"]
    SORT --> DIFF["Diff: Compute group boundaries (kernel)"]
    DIFF --> SCAN["Scan: Inclusive scan (thrust::inclusive_scan)"]
    SCAN --> UPDATE["Update: Assign new ranks (kernel)"]
    UPDATE --> EXIT{"All groups unique?"}
    EXIT -- No --> PREP
    EXIT -- Yes --> DONE["Done: Copy back Suffix Array"]

    classDef op fill:#f0f6ff,stroke:#6fa1f2,stroke-width:2px;
    classDef cond fill:#fff8e1,stroke:#ebba5b,stroke-width:2px,stroke-dasharray: 6 2;
    classDef done fill:#e7ffe7,stroke:#45c264,stroke-width:2px;

    class INIT,PREP,SORT,DIFF,SCAN,UPDATE op
    class EXIT cond
    class DONE done
```

####  **Key Optimizations**

* **In-place Memory Reuse:**
  Intermediate buffers are reused, minimizing allocations.
* **Streaming & Adaptive Modes:**
  If input size exceeds memory, GZLZ splits into batches and pipelines the doubling.
* **Profiling Hooks:**
  Each step is profiled for fine-grained analysis and bottleneck detection.

---


### What is LZ77 Compression?

LZ77 is a foundational algorithm for lossless data compression. It replaces repeated substrings with references to their previous occurrences, representing data as a sequence of **(offset, length, next character)** triples.
This drastically reduces storage for highly repetitive or structured data, making it crucial in genomic compression and other large-scale data applications.

#### **Factorization Pipeline in GZLZ**

GZLZ accelerates LZ77 factorization on the GPU by exploiting suffix arrays for rapid substring matching and utilizing parallel primitives to compute candidate matches efficiently.

1. **Suffix Array Construction:** Build the suffix array for the input string (using GPU prefix doubling as above).
2. **PSV/NSV Computation:** For each position, compute the Previous Smaller Value (PSV) and Next Smaller Value (NSV) in the suffix array, representing lexicographically closer neighbors.
3. **Factor Extraction:** For each position, select the neighbor (PSV or NSV) yielding the longest match, and encode the (offset, length, next character) triple.
4. **Streaming/Batching:** For datasets exceeding GPU memory, the pipeline dynamically streams suffix array segments and merges matches across batch boundaries.

**Example:**
```mermaid
graph LR
A["BANANA_BANANA"] -- "BANANA → (0,6)" --> B["(0,6)_BANANA"]
B -- "BANANA → (7,6)" --> C["(0,6)_(7,6)"]
```
* First `"BANANA"`: Encoded as literal.
* Second `"BANANA"`: Encoded as `(offset=0, length=6, next='_')`, etc.

#### **GPU Kernel Flow for LZ77 (Batch & Streaming)**

```mermaid
flowchart TD
    BUILD_SA["Build Suffix Array<br/>(Prefix Doubling)"] --> COMPUTE_PSV["Compute PSV (kernel)"]
    BUILD_SA --> COMPUTE_NSV["Compute NSV (kernel)"]

    COMPUTE_PSV --> DECIDE{"Is memory limited?"}
    COMPUTE_NSV --> DECIDE

    DECIDE -- "Yes<br/><sub>Next&nbsp;batch</sub>" --> COMPUTE_PSV
    DECIDE -- "Yes<br/><sub>Next&nbsp;batch</sub>" --> COMPUTE_NSV
    DECIDE -- "No" --> KKP3["KKP3 Factorization"]

    classDef step fill:#f0f6ff,stroke:#6fa1f2,stroke-width:2px;
    classDef cond fill:#fff8e1,stroke:#ebba5b,stroke-width:2px,stroke-dasharray:6 2;
    class BUILD_SA,COMPUTE_PSV,COMPUTE_NSV,KKP3 step
    class DECIDE cond

```

#### **Why PSV & NSV Matter**

Let \$SA\$ be the suffix array for \$T\$, and \$i\$ an index in \$SA\$:

* **PSV(i):** The closest suffix before \$i\$ in SA, such that \$SA\[j] < SA\[i]\$ and \$j < i\$.
* **NSV(i):** The closest suffix after \$i\$ in SA, such that \$SA\[j] < SA\[i]\$ and \$j > i\$.

By comparing the match lengths for \$PSV(i)\$ and \$NSV(i)\$, we efficiently find the longest leftward match.

$$
\mathrm{LZ77\_factor}(i) = \max \{ \mathrm{lcp}(T[i..],\ T[\mathrm{PSV}(i)..]),\ \mathrm{lcp}(T[i..],\ T[\mathrm{NSV}(i)..]) \}
$$

#### **Key GZLZ LZ77 Optimizations**

* **Blockwise Kernels:** Compute PSV/NSV in parallel blocks with batch merging for ultra-large files.
* **Text Order Mapping:** GPU text-order transforms for faster extraction and output.
* **Memory Adaptivity:** Automatic switching to streaming when memory is tight, with seamless batch merging logic.
* **Profiling/Instrumentation:** Every LZ77 phase profiled for speed and bottleneck analysis.

---

## Visualizations

### GPU Memory Usage Comparison

```mermaid
pie title GPU Memory Usage (32× Dataset)
"GZLZ GPU" : 29
"libCUBWT GPU" : 55
```

---

## Dependencies

### System Requirements

- **CUDA Toolkit**: Version 11.0 or higher
- **CMake**: Version 3.18 or higher
- **C++ Compiler**: Supporting C++17 standard
- **NVIDIA GPU**: Compute Capability 8.0+ (Ampere or newer recommended)
  - Tested on: A100 (80 GB VRAM)
  - Supported architectures: SM 80, 86, 90

### Automatic Dependencies (via CMake FetchContent)

The following libraries are automatically downloaded and built during CMake configuration:

- **[libdivsufsort](https://github.com/simongog/libdivsufsort)** (v2.0.1) — Suffix array construction library
- **[libsais](https://github.com/IlyaGrebnov/libsais)** (v2.8.6) — Fast linear-time suffix array construction
- **[SDSL-lite](https://github.com/simongog/sdsl-lite)** (v2.1.1) — Succinct data structure library
- **[CLI11](https://github.com/CLIUtils/CLI11)** (v1.9.0) — Command line parser
- **[malloc_count](https://github.com/bingmann/malloc_count)** (v0.7.1) — Memory profiling
- **[spdlog](https://github.com/gabime/spdlog)** (v1.15.0) — Async logging library
- **[Catch2](https://github.com/catchorg/Catch2)** (v3.6.0) — Unit testing framework

### Optional Dependencies

- **OpenMP**: For parallel CPU suffix array construction (recommended)

---

## Build Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/GPU_LZ.git
cd GPU_LZ
```

### 2. Create Build Directory

```bash
mkdir build
cd build
```

### 3. Configure with CMake

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**Build Types:**
- `Release`: Optimized build with `-O3` flag (recommended for benchmarks)
- `Debug`: Debug symbols with `-g -G -O0` flags

### 4. Build the Project

```bash
cmake --build . -j$(nproc)
```

This will create three executables:
- **`sa_profiler`** — Suffix array construction profiling tool
- **`lz77`** — LZ77 compression tool
- **`tests`** — Unit test suite

---

## Usage

### Suffix Array Profiling

Profile GPU suffix array construction on your input file:

```bash
./sa_profiler <input_file>
```

**Example:**
```bash
./sa_profiler ../data/genome.fasta
```

**Output:**
- Profiling summary with detailed timing breakdown
- Initialization, sorting, kernel execution times
- Memory usage statistics

### LZ77 Compression

Compress data using GPU-accelerated LZ77:

```bash
./lz77 <input_file>
```

**Example:**
```bash
./lz77 ../data/genome.fasta
```

**Features:**
- Automatic GPU/CPU mode selection based on available memory
- Streaming mode for files exceeding GPU VRAM
- Detailed compression statistics and timing

### Running Unit Tests

```bash
./tests
```

---

## Bioinformatics & Compression Applications

GZLZ directly targets critical needs in bioinformatics, where genomic data volumes explode exponentially. Optimized GPU implementations allow researchers and practitioners to handle larger datasets efficiently, accelerating genome assembly, sequence alignment, and high-throughput data compression workflows.

---
