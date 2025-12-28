// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

//
// Standalone Benchmark: Original (Linear Seek) vs Optimized (Galloping Search)
//
// Compile with:
//   clang++ -std=c++17 -O3 -I../include -o benchmark_zipper_merge benchmark_zipper_merge_standalone.cpp
//
// Run:
//   ./benchmark_zipper_merge
//

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <vector>

#include "knowhere/bitsetview.h"

using namespace knowhere;

namespace {

constexpr int64_t kSeed = 42;
constexpr int kWarmupIterations = 5;
constexpr int kBenchmarkIterations = 20;

using table_t = uint32_t;

// 64-byte aligned allocator for cache-line aligned memory access
template <typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    constexpr AlignedAllocator() noexcept = default;

    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {
    }

    [[nodiscard]] T*
    allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void
    deallocate(T* p, [[maybe_unused]] std::size_t n) noexcept {
        std::free(p);
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template <typename T, typename U, std::size_t Alignment>
bool
operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) noexcept {
    return true;
}

template <typename T, typename U, std::size_t Alignment>
bool
operator!=(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) noexcept {
    return false;
}

// Cache-line aligned vector
template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, 64>>;

// Generate a sorted posting list with random doc IDs (standard allocation)
std::vector<table_t>
GeneratePostingList(size_t size, size_t max_doc_id, int seed = kSeed) {
    std::mt19937 rng(seed);
    std::vector<table_t> posting_list;
    posting_list.reserve(size);

    std::vector<table_t> all_ids(max_doc_id);
    std::iota(all_ids.begin(), all_ids.end(), 0);
    std::shuffle(all_ids.begin(), all_ids.end(), rng);

    posting_list.assign(all_ids.begin(), all_ids.begin() + size);
    std::sort(posting_list.begin(), posting_list.end());

    return posting_list;
}

// Generate a sorted posting list with 64-byte aligned allocation
AlignedVector<table_t>
GenerateAlignedPostingList(size_t size, size_t max_doc_id, int seed = kSeed) {
    std::mt19937 rng(seed);
    AlignedVector<table_t> posting_list;
    posting_list.reserve(size);

    std::vector<table_t> all_ids(max_doc_id);
    std::iota(all_ids.begin(), all_ids.end(), 0);
    std::shuffle(all_ids.begin(), all_ids.end(), rng);

    posting_list.assign(all_ids.begin(), all_ids.begin() + size);
    std::sort(posting_list.begin(), posting_list.end());

    return posting_list;
}

// Generate a bitset with specified percentage of bits set (filtered out)
std::vector<uint8_t>
GenerateBitset(size_t num_bits, float filter_percentage, int seed = kSeed) {
    std::mt19937 rng(seed);
    size_t num_filtered = static_cast<size_t>(num_bits * filter_percentage / 100.0f);

    std::vector<bool> bits(num_bits, false);
    for (size_t i = 0; i < num_filtered; ++i) {
        bits[i] = true;
    }
    std::shuffle(bits.begin(), bits.end(), rng);

    std::vector<uint8_t> bitset((num_bits + 7) / 8, 0);
    for (size_t i = 0; i < num_bits; ++i) {
        if (bits[i]) {
            bitset[i >> 3] |= (0x1 << (i & 0x7));
        }
    }

    return bitset;
}

// Original implementation (from main branch) - uses linear scan for seek()
class OriginalCursor {
 public:
    OriginalCursor(const std::vector<table_t>& posting_list, const BitsetView& filter)
        : posting_list_(posting_list), filter_(filter), loc_(0) {
        skip_filtered_ids();
    }

    bool
    has_next() const {
        return loc_ < posting_list_.size();
    }

    table_t
    current() const {
        return posting_list_[loc_];
    }

    void
    next() {
        ++loc_;
        skip_filtered_ids();
    }

    void
    seek(table_t target) {
        while (loc_ < posting_list_.size() && posting_list_[loc_] < target) {
            ++loc_;
        }
        skip_filtered_ids();
    }

 private:
    void
    skip_filtered_ids() {
        while (loc_ < posting_list_.size() && !filter_.empty() && filter_.test(posting_list_[loc_])) {
            ++loc_;
        }
    }

    const std::vector<table_t>& posting_list_;
    const BitsetView& filter_;
    size_t loc_;
};

// Optimized cursor: Uses galloping search for seek() operations
// Sequential next() and filter test remain unchanged (linear scan is CPU-efficient)
class OptimizedCursor {
 public:
    OptimizedCursor(const std::vector<table_t>& posting_list, const BitsetView& filter)
        : posting_list_(posting_list.data()),
          posting_size_(posting_list.size()),
          filter_(filter),
          loc_(0) {
        skip_filtered_linear();
    }

    bool
    has_next() const {
        return loc_ < posting_size_;
    }

    table_t
    current() const {
        return posting_list_[loc_];
    }

    void
    next() {
        ++loc_;
        skip_filtered_linear();
    }

    void
    seek(table_t target) {
        if (loc_ >= posting_size_) {
            return;
        }

        // Use galloping search to find position >= target
        // O(log d) where d = distance to target
        loc_ = gallop_ge(loc_, target);

        // Use linear scan for filter test (CPU-efficient)
        skip_filtered_linear();
    }

 private:
    // Fast linear scan - CPU-efficient: sequential memory access, predictable branches
    void
    skip_filtered_linear() {
        if (filter_.empty()) {
            return;
        }
        while (loc_ < posting_size_ && filter_.test(posting_list_[loc_])) {
            ++loc_;
        }
    }

    // Galloping (exponential) search: O(log d) where d = distance to target
    // CPU-efficient because:
    // 1. Exponential phase uses sequential doubling (cache-friendly)
    // 2. Binary search phase operates on small range that fits in L1 cache
    size_t
    gallop_ge(size_t start, table_t target) const {
        if (start >= posting_size_ || posting_list_[start] >= target) {
            return start;
        }

        // Exponential search: sequential memory access pattern
        size_t step = 1;
        size_t lo = start;
        size_t hi = start + step;

        while (hi < posting_size_ && posting_list_[hi] < target) {
            lo = hi;
            step *= 2;
            hi = lo + step;
        }
        hi = std::min(hi, posting_size_);

        // Binary search within small range (typically < 2*d elements)
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (posting_list_[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    const table_t* posting_list_;
    size_t posting_size_;
    const BitsetView& filter_;
    size_t loc_;
};

// Aligned cursor (no prefetch) - tests pure alignment benefit
// Uses 64-byte aligned memory for optimal cache line usage
class AlignedOptimizedCursor {
 public:
    AlignedOptimizedCursor(const AlignedVector<table_t>& posting_list, const BitsetView& filter)
        : posting_list_(posting_list.data()),
          posting_size_(posting_list.size()),
          filter_(filter),
          loc_(0) {
        skip_filtered_linear();
    }

    bool
    has_next() const {
        return loc_ < posting_size_;
    }

    table_t
    current() const {
        return posting_list_[loc_];
    }

    void
    next() {
        ++loc_;
        skip_filtered_linear();
    }

    void
    seek(table_t target) {
        if (loc_ >= posting_size_) {
            return;
        }
        loc_ = gallop_ge(loc_, target);
        skip_filtered_linear();
    }

 private:
    void
    skip_filtered_linear() {
        if (filter_.empty()) {
            return;
        }
        while (loc_ < posting_size_ && filter_.test(posting_list_[loc_])) {
            ++loc_;
        }
    }

    // Same galloping search as OptimizedCursor, but on aligned memory
    size_t
    gallop_ge(size_t start, table_t target) const {
        if (start >= posting_size_ || posting_list_[start] >= target) {
            return start;
        }

        size_t step = 1;
        size_t lo = start;
        size_t hi = start + step;

        while (hi < posting_size_ && posting_list_[hi] < target) {
            lo = hi;
            step *= 2;
            hi = lo + step;
        }
        hi = std::min(hi, posting_size_);

        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (posting_list_[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    const table_t* posting_list_;
    size_t posting_size_;
    const BitsetView& filter_;
    size_t loc_;
};

struct BenchmarkResult {
    size_t posting_list_size;
    float filter_percentage;
    double original_time_ms;
    double optimized_time_ms;
    double aligned_time_ms;
    double speedup;
    double aligned_speedup;
    size_t docs_visited;
};

BenchmarkResult
RunBenchmark(size_t posting_list_size, size_t max_doc_id, float filter_percentage) {
    auto posting_list = GeneratePostingList(posting_list_size, max_doc_id);
    auto aligned_posting_list = GenerateAlignedPostingList(posting_list_size, max_doc_id);
    auto bitset_data = GenerateBitset(max_doc_id, filter_percentage);
    BitsetView bitset(bitset_data.data(), max_doc_id);

    size_t docs_visited = 0;

    // Warmup original
    for (int i = 0; i < kWarmupIterations; ++i) {
        OriginalCursor cursor(posting_list, bitset);
        while (cursor.has_next()) {
            cursor.next();
        }
    }

    // Benchmark original
    volatile table_t original_sum = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        OriginalCursor cursor(posting_list, bitset);
        size_t count = 0;
        while (cursor.has_next()) {
            ++count;
            original_sum += cursor.current();
            cursor.next();
        }
        if (i == 0) {
            docs_visited = count;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    (void)original_sum;
    double original_time_ms =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0 / kBenchmarkIterations;

    // Warmup optimized (unaligned)
    for (int i = 0; i < kWarmupIterations; ++i) {
        OptimizedCursor cursor(posting_list, bitset);
        while (cursor.has_next()) {
            cursor.next();
        }
    }

    // Benchmark optimized (unaligned)
    volatile table_t optimized_sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        OptimizedCursor cursor(posting_list, bitset);
        while (cursor.has_next()) {
            optimized_sum += cursor.current();
            cursor.next();
        }
    }
    end = std::chrono::high_resolution_clock::now();
    (void)optimized_sum;
    double optimized_time_ms =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0 / kBenchmarkIterations;

    // Warmup aligned + prefetch
    for (int i = 0; i < kWarmupIterations; ++i) {
        AlignedOptimizedCursor cursor(aligned_posting_list, bitset);
        while (cursor.has_next()) {
            cursor.next();
        }
    }

    // Benchmark aligned + prefetch
    volatile table_t aligned_sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        AlignedOptimizedCursor cursor(aligned_posting_list, bitset);
        while (cursor.has_next()) {
            aligned_sum += cursor.current();
            cursor.next();
        }
    }
    end = std::chrono::high_resolution_clock::now();
    (void)aligned_sum;
    double aligned_time_ms =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0 / kBenchmarkIterations;

    double speedup = (optimized_time_ms > 0.0001) ? (original_time_ms / optimized_time_ms) : 999.0;
    double aligned_speedup = (aligned_time_ms > 0.0001) ? (original_time_ms / aligned_time_ms) : 999.0;

    return BenchmarkResult{
        .posting_list_size = posting_list_size,
        .filter_percentage = filter_percentage,
        .original_time_ms = original_time_ms,
        .optimized_time_ms = optimized_time_ms,
        .aligned_time_ms = aligned_time_ms,
        .speedup = speedup,
        .aligned_speedup = aligned_speedup,
        .docs_visited = docs_visited,
    };
}

BenchmarkResult
RunBenchmarkWithSeeks(size_t posting_list_size, size_t max_doc_id, float filter_percentage) {
    auto posting_list = GeneratePostingList(posting_list_size, max_doc_id);
    auto aligned_posting_list = GenerateAlignedPostingList(posting_list_size, max_doc_id);
    auto bitset_data = GenerateBitset(max_doc_id, filter_percentage);
    BitsetView bitset(bitset_data.data(), max_doc_id);

    std::mt19937 rng(kSeed + 1);
    std::uniform_int_distribution<table_t> dist(0, max_doc_id - 1);
    std::vector<table_t> seek_targets;
    // 10K seek targets to stress the seek path with large data
    for (int i = 0; i < 10000; ++i) {
        seek_targets.push_back(dist(rng));
    }
    std::sort(seek_targets.begin(), seek_targets.end());

    size_t docs_visited = 0;

    // Warmup original
    for (int i = 0; i < kWarmupIterations; ++i) {
        OriginalCursor cursor(posting_list, bitset);
        for (auto target : seek_targets) {
            cursor.seek(target);
            if (cursor.has_next()) {
                cursor.next();
            }
        }
    }

    // Benchmark original
    volatile table_t original_sum = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        OriginalCursor cursor(posting_list, bitset);
        size_t count = 0;
        for (auto target : seek_targets) {
            cursor.seek(target);
            if (cursor.has_next()) {
                ++count;
                original_sum += cursor.current();
                cursor.next();
            }
        }
        if (i == 0) {
            docs_visited = count;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    (void)original_sum;
    double original_time_ms =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0 / kBenchmarkIterations;

    // Warmup optimized (unaligned)
    for (int i = 0; i < kWarmupIterations; ++i) {
        OptimizedCursor cursor(posting_list, bitset);
        for (auto target : seek_targets) {
            cursor.seek(target);
            if (cursor.has_next()) {
                cursor.next();
            }
        }
    }

    // Benchmark optimized (unaligned)
    volatile table_t optimized_sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        OptimizedCursor cursor(posting_list, bitset);
        for (auto target : seek_targets) {
            cursor.seek(target);
            if (cursor.has_next()) {
                optimized_sum += cursor.current();
                cursor.next();
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    (void)optimized_sum;
    double optimized_time_ms =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0 / kBenchmarkIterations;

    // Warmup aligned + prefetch
    for (int i = 0; i < kWarmupIterations; ++i) {
        AlignedOptimizedCursor cursor(aligned_posting_list, bitset);
        for (auto target : seek_targets) {
            cursor.seek(target);
            if (cursor.has_next()) {
                cursor.next();
            }
        }
    }

    // Benchmark aligned + prefetch
    volatile table_t aligned_sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        AlignedOptimizedCursor cursor(aligned_posting_list, bitset);
        for (auto target : seek_targets) {
            cursor.seek(target);
            if (cursor.has_next()) {
                aligned_sum += cursor.current();
                cursor.next();
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    (void)aligned_sum;
    double aligned_time_ms =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0 / kBenchmarkIterations;

    double speedup = (optimized_time_ms > 0.0001) ? (original_time_ms / optimized_time_ms) : 999.0;
    double aligned_speedup = (aligned_time_ms > 0.0001) ? (original_time_ms / aligned_time_ms) : 999.0;

    return BenchmarkResult{
        .posting_list_size = posting_list_size,
        .filter_percentage = filter_percentage,
        .original_time_ms = original_time_ms,
        .optimized_time_ms = optimized_time_ms,
        .aligned_time_ms = aligned_time_ms,
        .speedup = speedup,
        .aligned_speedup = aligned_speedup,
        .docs_visited = docs_visited,
    };
}

void
PrintResults(const std::vector<BenchmarkResult>& results, const std::string& title) {
    std::cout << "\n" << title << "\n";
    std::cout << std::string(130, '=') << "\n";
    std::cout << std::setw(12) << "PostingSize" << std::setw(10) << "Filter%" << std::setw(12) << "DocsVisited"
              << std::setw(14) << "Original(ms)" << std::setw(14) << "Gallop(ms)" << std::setw(14) << "Aligned(ms)"
              << std::setw(12) << "Gallop" << std::setw(12) << "Aligned"
              << "\n";
    std::cout << std::string(130, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::setw(12) << r.posting_list_size << std::setw(9) << r.filter_percentage << "%"
                  << std::setw(12) << r.docs_visited << std::setw(14) << std::fixed << std::setprecision(3)
                  << r.original_time_ms << std::setw(14) << r.optimized_time_ms << std::setw(14) << r.aligned_time_ms
                  << std::setw(11) << std::setprecision(2) << r.speedup << "x"
                  << std::setw(11) << r.aligned_speedup << "x\n";
    }
    std::cout << std::string(130, '=') << "\n";
}

}  // namespace

int
main() {
    std::cout << "================================================================================\n";
    std::cout << "Benchmark: Original vs Galloping vs Aligned+Prefetch\n";
    std::cout << "================================================================================\n";
    std::cout << "Warmup iterations: " << kWarmupIterations << "\n";
    std::cout << "Benchmark iterations: " << kBenchmarkIterations << "\n";
    std::cout << "Aligned = 64-byte cache-line aligned memory (no prefetch)\n";

    // LARGE DATA: 50M docs, 5-10M posting list entries (~20-40MB per list)
    // This ensures data doesn't fit in L2/L3 cache, stressing memory subsystem
    const std::vector<size_t> posting_sizes = {5000000, 7500000, 10000000};
    const std::vector<float> filter_percentages = {5.0f, 10.0f, 20.0f, 50.0f};
    const size_t max_doc_id = 50000000;  // 50 million docs

    std::cout << "Data size: " << max_doc_id / 1000000 << "M docs, posting lists up to "
              << posting_sizes.back() / 1000000 << "M entries (~"
              << (posting_sizes.back() * sizeof(table_t)) / (1024 * 1024) << " MB)\n";

    // Sequential iteration benchmark
    {
        std::vector<BenchmarkResult> results;
        std::cout << "\nRunning sequential iteration benchmark...\n";

        for (auto posting_size : posting_sizes) {
            for (auto filter_pct : filter_percentages) {
                std::cout << "  posting=" << posting_size << ", filter=" << filter_pct << "%..." << std::flush;
                auto result = RunBenchmark(posting_size, max_doc_id, filter_pct);
                results.push_back(result);
                std::cout << " gallop: " << std::fixed << std::setprecision(2) << result.speedup << "x"
                          << " aligned: " << result.aligned_speedup << "x\n";
            }
        }

        PrintResults(results, "Sequential Iteration (full posting list traversal)");
    }

    // Seek operations benchmark
    {
        std::vector<BenchmarkResult> results;
        std::cout << "\nRunning seek operations benchmark...\n";

        for (auto posting_size : posting_sizes) {
            for (auto filter_pct : filter_percentages) {
                std::cout << "  posting=" << posting_size << ", filter=" << filter_pct << "%..." << std::flush;
                auto result = RunBenchmarkWithSeeks(posting_size, max_doc_id, filter_pct);
                results.push_back(result);
                std::cout << " gallop: " << std::fixed << std::setprecision(2) << result.speedup << "x"
                          << " aligned: " << result.aligned_speedup << "x\n";
            }
        }

        PrintResults(results, "Seek Operations (simulating WAND/MaxScore pivots)");
    }

    // High filter rates benchmark (use large posting list)
    {
        std::vector<BenchmarkResult> results;
        const std::vector<float> high_filter_pcts = {70.0f, 80.0f, 90.0f, 95.0f, 99.0f};
        std::cout << "\nRunning high filter rate benchmark...\n";

        for (auto filter_pct : high_filter_pcts) {
            std::cout << "  filter=" << filter_pct << "%..." << std::flush;
            auto result = RunBenchmark(5000000, max_doc_id, filter_pct);  // 5M posting list
            results.push_back(result);
            std::cout << " gallop: " << std::fixed << std::setprecision(2) << result.speedup << "x"
                      << " aligned: " << result.aligned_speedup << "x\n";
        }

        PrintResults(results, "High Filter Rates (posting_size=5M)");
    }

    std::cout << "\nBenchmark complete.\n";
    return 0;
}

