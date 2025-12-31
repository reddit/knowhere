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
// Benchmark: Zipper Merge vs Linear Scan for skip_filtered_ids
//
// This benchmark compares the performance of the new zipper merge optimization
// against the original linear scan approach for skipping filtered doc IDs in
// sparse inverted index search.
//
// Test parameters:
// - Posting list sizes: 200K, 225K, 250K (typical segment sizes)
// - Filter selectivity: 5%, 10%, 20%, 50% (percentage of docs filtered out)
//

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "knowhere/bitsetview.h"
#include "knowhere/sparse_utils.h"

using namespace knowhere;
using namespace knowhere::sparse;

namespace {

constexpr int64_t kSeed = 42;
constexpr int kWarmupIterations = 3;
constexpr int kBenchmarkIterations = 10;

// Generate a sorted posting list with random doc IDs
std::vector<table_t>
GeneratePostingList(size_t size, size_t max_doc_id, int seed = kSeed) {
    std::mt19937 rng(seed);
    std::vector<table_t> posting_list;
    posting_list.reserve(size);

    // Generate unique random doc IDs
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

// Linear scan implementation (original approach)
class LinearScanCursor {
 public:
    LinearScanCursor(const std::vector<table_t>& posting_list, const BitsetView& filter)
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
        // Original linear scan approach
        while (loc_ < posting_list_.size() && !filter_.empty() && filter_.test(posting_list_[loc_])) {
            ++loc_;
        }
    }

    const std::vector<table_t>& posting_list_;
    const BitsetView& filter_;
    size_t loc_;
};

// Zipper merge implementation (new optimized approach)
class ZipperMergeCursor {
 public:
    ZipperMergeCursor(const std::vector<table_t>& posting_list, const BitsetView& filter)
        : posting_list_(posting_list), filter_(filter), loc_(0) {
        if (!filter_.empty()) {
            filter_iter_ = filter_.valid_doc_iterator();
        }
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
        if (loc_ < posting_list_.size()) {
            auto it = std::lower_bound(posting_list_.begin() + loc_, posting_list_.end(), target);
            loc_ = it - posting_list_.begin();
        }
        skip_filtered_ids();
    }

 private:
    void
    skip_filtered_ids() {
        if (filter_.empty()) {
            return;
        }

        // Zipper merge approach
        while (loc_ < posting_list_.size() && filter_iter_.has_next()) {
            table_t posting_doc = posting_list_[loc_];
            size_t filter_doc = filter_iter_.current();

            if (posting_doc == static_cast<table_t>(filter_doc)) {
                return;  // Match found
            } else if (posting_doc < filter_doc) {
                // Posting list is behind - binary search forward
                auto it = std::lower_bound(posting_list_.begin() + loc_, posting_list_.end(),
                                           static_cast<table_t>(filter_doc));
                loc_ = it - posting_list_.begin();
            } else {
                // Filter is behind - advance filter iterator
                filter_iter_.advance_to_ge(posting_doc);
            }
        }
        if (!filter_iter_.has_next()) {
            loc_ = posting_list_.size();  // No more valid docs
        }
    }

    const std::vector<table_t>& posting_list_;
    const BitsetView& filter_;
    ValidDocIdIterator filter_iter_;
    size_t loc_;
};

// Benchmark result structure
struct BenchmarkResult {
    size_t posting_list_size;
    float filter_percentage;
    double linear_scan_time_ms;
    double zipper_merge_time_ms;
    double speedup;
    size_t docs_visited;
};

// Run a single benchmark configuration
BenchmarkResult
RunBenchmark(size_t posting_list_size, size_t max_doc_id, float filter_percentage) {
    auto posting_list = GeneratePostingList(posting_list_size, max_doc_id);
    auto bitset_data = GenerateBitset(max_doc_id, filter_percentage);
    BitsetView bitset(bitset_data.data(), max_doc_id);

    size_t docs_visited = 0;

    // Warmup linear scan
    for (int i = 0; i < kWarmupIterations; ++i) {
        LinearScanCursor cursor(posting_list, bitset);
        while (cursor.has_next()) {
            cursor.next();
        }
    }

    // Benchmark linear scan
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        LinearScanCursor cursor(posting_list, bitset);
        size_t count = 0;
        while (cursor.has_next()) {
            ++count;
            cursor.next();
        }
        if (i == 0) {
            docs_visited = count;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double linear_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / kBenchmarkIterations;

    // Warmup zipper merge
    for (int i = 0; i < kWarmupIterations; ++i) {
        ZipperMergeCursor cursor(posting_list, bitset);
        while (cursor.has_next()) {
            cursor.next();
        }
    }

    // Benchmark zipper merge
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        ZipperMergeCursor cursor(posting_list, bitset);
        while (cursor.has_next()) {
            cursor.next();
        }
    }
    end = std::chrono::high_resolution_clock::now();
    double zipper_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / kBenchmarkIterations;

    return BenchmarkResult{
        .posting_list_size = posting_list_size,
        .filter_percentage = filter_percentage,
        .linear_scan_time_ms = linear_time_ms,
        .zipper_merge_time_ms = zipper_time_ms,
        .speedup = linear_time_ms / zipper_time_ms,
        .docs_visited = docs_visited,
    };
}

// Run benchmark with seek operations (simulates WAND/MaxScore pivot jumping)
BenchmarkResult
RunBenchmarkWithSeeks(size_t posting_list_size, size_t max_doc_id, float filter_percentage) {
    auto posting_list = GeneratePostingList(posting_list_size, max_doc_id);
    auto bitset_data = GenerateBitset(max_doc_id, filter_percentage);
    BitsetView bitset(bitset_data.data(), max_doc_id);

    // Generate seek targets (simulating pivot doc IDs in WAND algorithm)
    std::mt19937 rng(kSeed + 1);
    std::uniform_int_distribution<table_t> dist(0, max_doc_id - 1);
    std::vector<table_t> seek_targets;
    for (int i = 0; i < 1000; ++i) {
        seek_targets.push_back(dist(rng));
    }
    std::sort(seek_targets.begin(), seek_targets.end());

    size_t docs_visited = 0;

    // Warmup linear scan
    for (int i = 0; i < kWarmupIterations; ++i) {
        LinearScanCursor cursor(posting_list, bitset);
        for (auto target : seek_targets) {
            cursor.seek(target);
            if (cursor.has_next()) {
                cursor.next();
            }
        }
    }

    // Benchmark linear scan with seeks
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        LinearScanCursor cursor(posting_list, bitset);
        size_t count = 0;
        for (auto target : seek_targets) {
            cursor.seek(target);
            if (cursor.has_next()) {
                ++count;
                cursor.next();
            }
        }
        if (i == 0) {
            docs_visited = count;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double linear_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / kBenchmarkIterations;

    // Warmup zipper merge
    for (int i = 0; i < kWarmupIterations; ++i) {
        ZipperMergeCursor cursor(posting_list, bitset);
        for (auto target : seek_targets) {
            cursor.seek(target);
            if (cursor.has_next()) {
                cursor.next();
            }
        }
    }

    // Benchmark zipper merge with seeks
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        ZipperMergeCursor cursor(posting_list, bitset);
        for (auto target : seek_targets) {
            cursor.seek(target);
            if (cursor.has_next()) {
                cursor.next();
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    double zipper_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / kBenchmarkIterations;

    return BenchmarkResult{
        .posting_list_size = posting_list_size,
        .filter_percentage = filter_percentage,
        .linear_scan_time_ms = linear_time_ms,
        .zipper_merge_time_ms = zipper_time_ms,
        .speedup = linear_time_ms / zipper_time_ms,
        .docs_visited = docs_visited,
    };
}

void
PrintResults(const std::vector<BenchmarkResult>& results, const std::string& title) {
    std::cout << "\n" << title << "\n";
    std::cout << std::string(100, '=') << "\n";
    std::cout << std::setw(15) << "PostingSize" << std::setw(15) << "FilterPct" << std::setw(15) << "DocsVisited"
              << std::setw(18) << "LinearScan(ms)" << std::setw(18) << "ZipperMerge(ms)" << std::setw(12) << "Speedup"
              << "\n";
    std::cout << std::string(100, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::setw(15) << r.posting_list_size << std::setw(14) << r.filter_percentage << "%"
                  << std::setw(15) << r.docs_visited << std::setw(18) << std::fixed << std::setprecision(3)
                  << r.linear_scan_time_ms << std::setw(18) << r.zipper_merge_time_ms << std::setw(11)
                  << std::setprecision(2) << r.speedup << "x\n";
    }
    std::cout << std::string(100, '=') << "\n";
}

}  // namespace

class Benchmark_ZipperMerge : public ::testing::Test {
 protected:
    void
    SetUp() override {
        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "Benchmark: Zipper Merge vs Linear Scan for skip_filtered_ids\n";
        std::cout << "================================================================\n";
        std::cout << "Warmup iterations: " << kWarmupIterations << "\n";
        std::cout << "Benchmark iterations: " << kBenchmarkIterations << "\n";
    }
};

TEST_F(Benchmark_ZipperMerge, SequentialIteration) {
    // Test configurations
    const std::vector<size_t> posting_sizes = {200000, 225000, 250000};
    const std::vector<float> filter_percentages = {5.0f, 10.0f, 20.0f, 50.0f};
    const size_t max_doc_id = 1000000;  // 1M total docs in segment

    std::vector<BenchmarkResult> results;

    for (auto posting_size : posting_sizes) {
        for (auto filter_pct : filter_percentages) {
            std::cout << "Running: posting_size=" << posting_size << ", filter=" << filter_pct << "%..." << std::flush;
            auto result = RunBenchmark(posting_size, max_doc_id, filter_pct);
            results.push_back(result);
            std::cout << " done (speedup: " << std::fixed << std::setprecision(2) << result.speedup << "x)\n";
        }
    }

    PrintResults(results, "Sequential Iteration (full posting list traversal)");
}

TEST_F(Benchmark_ZipperMerge, SeekOperations) {
    // Test configurations - simulates WAND/MaxScore algorithm with seeks
    const std::vector<size_t> posting_sizes = {200000, 225000, 250000};
    const std::vector<float> filter_percentages = {5.0f, 10.0f, 20.0f, 50.0f};
    const size_t max_doc_id = 1000000;  // 1M total docs in segment

    std::vector<BenchmarkResult> results;

    for (auto posting_size : posting_sizes) {
        for (auto filter_pct : filter_percentages) {
            std::cout << "Running (seeks): posting_size=" << posting_size << ", filter=" << filter_pct << "%..."
                      << std::flush;
            auto result = RunBenchmarkWithSeeks(posting_size, max_doc_id, filter_pct);
            results.push_back(result);
            std::cout << " done (speedup: " << std::fixed << std::setprecision(2) << result.speedup << "x)\n";
        }
    }

    PrintResults(results, "Seek Operations (simulating WAND/MaxScore pivots)");
}

TEST_F(Benchmark_ZipperMerge, HighFilterRates) {
    // Test with very high filter rates (common in filtered search)
    const size_t posting_size = 250000;
    const std::vector<float> filter_percentages = {70.0f, 80.0f, 90.0f, 95.0f, 99.0f};
    const size_t max_doc_id = 1000000;

    std::vector<BenchmarkResult> results;

    for (auto filter_pct : filter_percentages) {
        std::cout << "Running (high filter): filter=" << filter_pct << "%..." << std::flush;
        auto result = RunBenchmark(posting_size, max_doc_id, filter_pct);
        results.push_back(result);
        std::cout << " done (speedup: " << std::fixed << std::setprecision(2) << result.speedup << "x)\n";
    }

    PrintResults(results, "High Filter Rates (posting_size=250K)");
}

int
main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

