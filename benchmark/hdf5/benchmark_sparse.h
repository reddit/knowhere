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

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/dataset.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/version.h"

namespace sparse_benchmark {

// Timer utility for benchmarking
struct Timer {
    using clock_t = std::chrono::high_resolution_clock;
    using time_point_t = std::chrono::time_point<clock_t>;

    time_point_t start_;

    Timer() : start_(clock_t::now()) {
    }

    void
    reset() {
        start_ = clock_t::now();
    }

    double
    elapsed_seconds() const {
        return std::chrono::duration<double>(clock_t::now() - start_).count();
    }

    double
    elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(clock_t::now() - start_).count();
    }

    double
    elapsed_us() const {
        return std::chrono::duration<double, std::micro>(clock_t::now() - start_).count();
    }
};

// Statistics calculator for benchmark results
struct BenchmarkStats {
    std::vector<double> latencies_us;
    double total_time_s = 0.0;
    int64_t total_queries = 0;
    float recall = 0.0f;

    void
    add_latency(double latency_us) {
        latencies_us.push_back(latency_us);
    }

    double
    mean_latency_us() const {
        if (latencies_us.empty())
            return 0.0;
        return std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / latencies_us.size();
    }

    double
    percentile_latency_us(double p) const {
        if (latencies_us.empty())
            return 0.0;
        auto sorted = latencies_us;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = static_cast<size_t>(p / 100.0 * (sorted.size() - 1));
        return sorted[idx];
    }

    double
    p50_latency_us() const {
        return percentile_latency_us(50);
    }
    double
    p90_latency_us() const {
        return percentile_latency_us(90);
    }
    double
    p99_latency_us() const {
        return percentile_latency_us(99);
    }

    double
    qps() const {
        return total_time_s > 0 ? total_queries / total_time_s : 0;
    }
};

// Data distribution types for generating synthetic sparse data
enum class DataDistribution {
    UNIFORM,      // Uniform random distribution
    ZIPF,         // Zipf distribution (power law) - common in text/BM25
    CLUSTERED,    // Clustered data - groups of similar documents
    POWER_LAW,    // Power law for document lengths
    SKEWED_DIMS,  // Some dimensions appear much more frequently
};

// Filter distribution types
enum class FilterDistribution {
    NONE,                  // No filtering
    FIRST_N,               // First N documents are filtered
    LAST_N,                // Last N documents are filtered
    RANDOM,                // Random N documents filtered
    ALTERNATING,           // Every other document filtered
    CLUSTERED,             // Contiguous clusters of filtered documents
    REVERSE_POSTING_ORDER  // Filter documents that appear early in posting lists
};

// Configuration for data generation
struct DataGenConfig {
    int32_t num_docs = 100000;         // Number of documents
    int32_t num_dims = 30000;          // Number of dimensions (vocabulary size)
    float doc_sparsity = 0.99f;        // Document sparsity (1 - density)
    float query_sparsity = 0.995f;     // Query sparsity
    int32_t num_queries = 100;         // Number of queries
    DataDistribution distribution = DataDistribution::UNIFORM;
    float zipf_alpha = 1.2f;           // Zipf distribution parameter
    float max_value = 1.0f;            // Maximum value for elements
    bool use_integer_values = false;   // Use integer values (for BM25 term frequencies)
    int32_t max_tf = 256;              // Maximum term frequency for BM25
    int seed = 42;                     // Random seed
};

// Configuration for filter generation
struct FilterConfig {
    FilterDistribution distribution = FilterDistribution::NONE;
    float filter_ratio = 0.0f;  // Ratio of documents to filter (0.0 - 1.0)
    int seed = 42;
};

// Generate Zipf-distributed random values
class ZipfGenerator {
 public:
    ZipfGenerator(int32_t n, float alpha, int seed = 42) : n_(n), alpha_(alpha), rng_(seed) {
        // Precompute CDF
        double sum = 0.0;
        for (int32_t i = 1; i <= n_; ++i) {
            sum += 1.0 / std::pow(i, alpha_);
        }
        cdf_.resize(n_ + 1);
        cdf_[0] = 0.0;
        double cumsum = 0.0;
        for (int32_t i = 1; i <= n_; ++i) {
            cumsum += (1.0 / std::pow(i, alpha_)) / sum;
            cdf_[i] = cumsum;
        }
    }

    int32_t
    operator()() {
        double u = uniform_(rng_);
        auto it = std::lower_bound(cdf_.begin(), cdf_.end(), u);
        return std::max(1, static_cast<int32_t>(it - cdf_.begin()));
    }

 private:
    int32_t n_;
    float alpha_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};
    std::vector<double> cdf_;
};

// Generate sparse dataset with specified distribution
inline knowhere::DataSetPtr
GenSparseDataSetWithDistribution(const DataGenConfig& config) {
    std::mt19937 rng(config.seed);
    std::uniform_real_distribution<float> value_dist(0.0f, config.max_value);
    std::uniform_int_distribution<int32_t> dim_dist(0, config.num_dims - 1);
    std::uniform_int_distribution<int32_t> tf_dist(1, config.max_tf);

    // Calculate expected number of non-zero elements per document
    int32_t expected_nnz_per_doc =
        static_cast<int32_t>(config.num_dims * (1.0f - config.doc_sparsity));
    expected_nnz_per_doc = std::max(1, expected_nnz_per_doc);

    std::vector<std::map<int32_t, float>> data(config.num_docs);

    // Create Zipf generator for dimension selection if needed
    std::unique_ptr<ZipfGenerator> zipf_gen;
    if (config.distribution == DataDistribution::ZIPF ||
        config.distribution == DataDistribution::SKEWED_DIMS) {
        zipf_gen = std::make_unique<ZipfGenerator>(config.num_dims, config.zipf_alpha, config.seed);
    }

    // Generate documents based on distribution
    for (int32_t doc = 0; doc < config.num_docs; ++doc) {
        int32_t doc_nnz = expected_nnz_per_doc;

        // Vary document length based on distribution
        if (config.distribution == DataDistribution::POWER_LAW) {
            // Power law distribution for document lengths
            std::exponential_distribution<float> len_dist(1.0f / expected_nnz_per_doc);
            doc_nnz = std::max(1, std::min(config.num_dims, static_cast<int32_t>(len_dist(rng))));
        }

        // Select dimensions for this document
        std::set<int32_t> selected_dims;
        while (static_cast<int32_t>(selected_dims.size()) < doc_nnz) {
            int32_t dim;
            if (config.distribution == DataDistribution::ZIPF ||
                config.distribution == DataDistribution::SKEWED_DIMS) {
                dim = (*zipf_gen)() - 1;  // Zipf returns 1-indexed
            } else {
                dim = dim_dist(rng);
            }
            if (dim >= 0 && dim < config.num_dims) {
                selected_dims.insert(dim);
            }
        }

        // Assign values to selected dimensions
        for (int32_t dim : selected_dims) {
            float val;
            if (config.use_integer_values) {
                val = static_cast<float>(tf_dist(rng));
            } else {
                val = value_dist(rng);
            }
            data[doc][dim] = val;
        }
    }

    // Convert to SparseRow format
    auto tensor = std::make_unique<knowhere::sparse::SparseRow<float>[]>(config.num_docs);
    for (int32_t i = 0; i < config.num_docs; ++i) {
        if (data[i].empty()) {
            continue;
        }
        knowhere::sparse::SparseRow<float> row(data[i].size());
        size_t j = 0;
        for (auto& [idx, val] : data[i]) {
            row.set_at(j++, idx, val);
        }
        tensor[i] = std::move(row);
    }

    auto ds = knowhere::GenDataSet(config.num_docs, config.num_dims, tensor.release());
    ds->SetIsOwner(true);
    ds->SetIsSparse(true);
    return ds;
}

// Generate query dataset
inline knowhere::DataSetPtr
GenSparseQuerySet(const DataGenConfig& config) {
    DataGenConfig query_config = config;
    query_config.num_docs = config.num_queries;
    query_config.doc_sparsity = config.query_sparsity;
    query_config.seed = config.seed + 1000;  // Different seed for queries
    return GenSparseDataSetWithDistribution(query_config);
}

// Generate bitset filter based on configuration
inline std::vector<uint8_t>
GenFilterBitset(int32_t num_docs, const FilterConfig& config) {
    if (config.distribution == FilterDistribution::NONE || config.filter_ratio <= 0.0f) {
        return {};
    }

    int32_t num_filtered = static_cast<int32_t>(num_docs * config.filter_ratio);
    num_filtered = std::min(num_filtered, num_docs);

    std::vector<uint8_t> bitset((num_docs + 7) / 8, 0);

    auto set_bit = [&](int32_t idx) {
        if (idx >= 0 && idx < num_docs) {
            bitset[idx >> 3] |= (1 << (idx & 0x7));
        }
    };

    std::mt19937 rng(config.seed);

    switch (config.distribution) {
        case FilterDistribution::FIRST_N: {
            for (int32_t i = 0; i < num_filtered; ++i) {
                set_bit(i);
            }
            break;
        }

        case FilterDistribution::LAST_N: {
            for (int32_t i = num_docs - num_filtered; i < num_docs; ++i) {
                set_bit(i);
            }
            break;
        }

        case FilterDistribution::RANDOM: {
            std::vector<int32_t> indices(num_docs);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng);
            for (int32_t i = 0; i < num_filtered; ++i) {
                set_bit(indices[i]);
            }
            break;
        }

        case FilterDistribution::ALTERNATING: {
            // Filter every (1/filter_ratio)-th document
            int32_t step = std::max(1, static_cast<int32_t>(1.0f / config.filter_ratio));
            for (int32_t i = 0; i < num_docs; i += step) {
                set_bit(i);
            }
            break;
        }

        case FilterDistribution::CLUSTERED: {
            // Create clustered regions of filtered documents
            int32_t cluster_size = std::max(1, num_filtered / 10);
            int32_t remaining = num_filtered;
            std::uniform_int_distribution<int32_t> start_dist(0, num_docs - cluster_size);

            while (remaining > 0) {
                int32_t start = start_dist(rng);
                int32_t this_cluster = std::min(remaining, cluster_size);
                for (int32_t i = start; i < start + this_cluster && i < num_docs; ++i) {
                    set_bit(i);
                    remaining--;
                }
            }
            break;
        }

        case FilterDistribution::REVERSE_POSTING_ORDER: {
            // Filter documents with highest IDs (tend to appear later in posting lists)
            // This helps benchmark the seek optimization
            for (int32_t i = num_docs - 1; i >= num_docs - num_filtered && i >= 0; --i) {
                set_bit(i);
            }
            break;
        }

        default:
            break;
    }

    return bitset;
}

// Calculate recall between ground truth and results
// If debug_mismatches is true, print detailed info about mismatched results
inline float
CalcRecall(const knowhere::DataSet& ground_truth, const knowhere::DataSet& result, bool debug_mismatches = false) {
    auto nq = result.GetRows();
    auto k = result.GetDim();
    auto gt_k = ground_truth.GetDim();
    auto gt_ids = ground_truth.GetIds();
    auto res_ids = result.GetIds();
    auto gt_distances = ground_truth.GetDistance();
    auto res_distances = result.GetDistance();

    if (gt_ids == nullptr || res_ids == nullptr) {
        return 0.0f;
    }

    int32_t match_count = 0;
    int32_t mismatch_queries = 0;
    for (int64_t i = 0; i < nq; ++i) {
        std::set<int64_t> gt_set;
        std::set<int64_t> res_set;
        for (int64_t j = 0; j < gt_k && j < k; ++j) {
            if (gt_ids[i * gt_k + j] >= 0) {
                gt_set.insert(gt_ids[i * gt_k + j]);
            }
        }
        int32_t query_matches = 0;
        for (int64_t j = 0; j < k; ++j) {
            if (res_ids[i * k + j] >= 0) {
                res_set.insert(res_ids[i * k + j]);
                if (gt_set.count(res_ids[i * k + j]) > 0) {
                    query_matches++;
                }
            }
        }
        match_count += query_matches;

        // Debug: print mismatches for this query
        if (debug_mismatches && query_matches < static_cast<int32_t>(std::min(k, gt_k))) {
            mismatch_queries++;
            printf("\n[RECALL_MISMATCH] Query %ld: matched %d/%ld\n", i, query_matches, std::min(k, gt_k));

            // Find IDs in GT but not in result
            printf("  GT IDs not in result: ");
            for (auto gt_id : gt_set) {
                if (res_set.find(gt_id) == res_set.end()) {
                    printf("%ld ", gt_id);
                }
            }
            printf("\n");

            // Find IDs in result but not in GT
            printf("  Result IDs not in GT: ");
            for (auto res_id : res_set) {
                if (gt_set.find(res_id) == gt_set.end()) {
                    printf("%ld ", res_id);
                }
            }
            printf("\n");

            // Print GT IDs and distances
            printf("  GT top-%ld: ", std::min(k, gt_k));
            for (int64_t j = 0; j < std::min(k, gt_k); ++j) {
                if (gt_distances) {
                    printf("(%ld:%.6f) ", gt_ids[i * gt_k + j], gt_distances[i * gt_k + j]);
                } else {
                    printf("%ld ", gt_ids[i * gt_k + j]);
                }
            }
            printf("\n");

            // Print result IDs and distances
            printf("  Result top-%ld: ", k);
            for (int64_t j = 0; j < k; ++j) {
                if (res_distances) {
                    printf("(%ld:%.6f) ", res_ids[i * k + j], res_distances[i * k + j]);
                } else {
                    printf("%ld ", res_ids[i * k + j]);
                }
            }
            printf("\n");
            fflush(stdout);
        }
    }

    if (debug_mismatches && mismatch_queries > 0) {
        printf("\n[RECALL_SUMMARY] Total queries with mismatches: %d/%ld\n", mismatch_queries, nq);
        fflush(stdout);
    }

    return static_cast<float>(match_count) / (nq * std::min(k, gt_k));
}

// Calculate recall for GetAllDistances (compare top-k from full distances)
inline float
CalcRecallFromDistances(const std::vector<float>& gt_distances, const std::vector<float>& result_distances,
                        int32_t k) {
    if (gt_distances.size() != result_distances.size()) {
        return 0.0f;
    }

    // Get top-k indices from both
    auto get_topk_indices = [](const std::vector<float>& distances, int32_t k) {
        std::vector<std::pair<float, int32_t>> indexed;
        for (size_t i = 0; i < distances.size(); ++i) {
            if (distances[i] > 0) {
                indexed.emplace_back(distances[i], i);
            }
        }
        std::partial_sort(indexed.begin(),
                          indexed.begin() + std::min(static_cast<size_t>(k), indexed.size()),
                          indexed.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        std::set<int32_t> result;
        for (int32_t i = 0; i < k && i < static_cast<int32_t>(indexed.size()); ++i) {
            result.insert(indexed[i].second);
        }
        return result;
    };

    auto gt_topk = get_topk_indices(gt_distances, k);
    auto res_topk = get_topk_indices(result_distances, k);

    int32_t match_count = 0;
    for (auto id : res_topk) {
        if (gt_topk.count(id) > 0) {
            match_count++;
        }
    }

    return static_cast<float>(match_count) / k;
}

// Convert filter distribution to string
inline std::string
FilterDistributionToString(FilterDistribution dist) {
    switch (dist) {
        case FilterDistribution::NONE:
            return "NONE";
        case FilterDistribution::FIRST_N:
            return "FIRST_N";
        case FilterDistribution::LAST_N:
            return "LAST_N";
        case FilterDistribution::RANDOM:
            return "RANDOM";
        case FilterDistribution::ALTERNATING:
            return "ALTERNATING";
        case FilterDistribution::CLUSTERED:
            return "CLUSTERED";
        case FilterDistribution::REVERSE_POSTING_ORDER:
            return "REVERSE_POSTING";
        default:
            return "UNKNOWN";
    }
}

// Convert data distribution to string
inline std::string
DataDistributionToString(DataDistribution dist) {
    switch (dist) {
        case DataDistribution::UNIFORM:
            return "UNIFORM";
        case DataDistribution::ZIPF:
            return "ZIPF";
        case DataDistribution::CLUSTERED:
            return "CLUSTERED";
        case DataDistribution::POWER_LAW:
            return "POWER_LAW";
        case DataDistribution::SKEWED_DIMS:
            return "SKEWED_DIMS";
        default:
            return "UNKNOWN";
    }
}

// Print benchmark results
inline void
PrintBenchmarkResults(const std::string& test_name, const BenchmarkStats& stats) {
    printf("================================================================================\n");
    printf("Test: %s\n", test_name.c_str());
    printf("--------------------------------------------------------------------------------\n");
    printf("  Total Queries:    %ld\n", stats.total_queries);
    printf("  Total Time:       %.3f s\n", stats.total_time_s);
    printf("  QPS:              %.2f\n", stats.qps());
    printf("  Mean Latency:     %.2f us\n", stats.mean_latency_us());
    printf("  P50 Latency:      %.2f us\n", stats.p50_latency_us());
    printf("  P90 Latency:      %.2f us\n", stats.p90_latency_us());
    printf("  P99 Latency:      %.2f us\n", stats.p99_latency_us());
    printf("  Recall:           %.4f\n", stats.recall);
    printf("================================================================================\n\n");
    std::fflush(stdout);
}

}  // namespace sparse_benchmark
