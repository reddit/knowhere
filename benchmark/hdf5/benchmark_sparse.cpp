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

#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "benchmark_sparse.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/sparse_utils.h"

using namespace sparse_benchmark;

// Global timer for tracking total benchmark time
static Timer g_T0;

// Build index with specified algorithm
static knowhere::Index<knowhere::IndexNode>
BuildIndex(const knowhere::DataSetPtr& train_ds, const std::string& algo, const std::string& metric,
           float drop_ratio_build = 0.0f) {
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    std::string index_type = (algo == "TAAT_NAIVE") ? knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX
                                                    : knowhere::IndexEnum::INDEX_SPARSE_WAND;

    auto index = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(index_type, version);
    REQUIRE(index.has_value());

    knowhere::Json build_conf;
    build_conf[knowhere::meta::METRIC_TYPE] = metric;
    build_conf[knowhere::indexparam::DROP_RATIO_BUILD] = drop_ratio_build;
    build_conf[knowhere::indexparam::INVERTED_INDEX_ALGO] = algo;

    if (metric == knowhere::metric::BM25) {
        build_conf[knowhere::meta::BM25_K1] = 1.2f;
        build_conf[knowhere::meta::BM25_B] = 0.75f;
        build_conf[knowhere::meta::BM25_AVGDL] = 100.0f;
    }

    Timer build_timer;
    auto status = index.value().Build(train_ds, build_conf);
    REQUIRE(status == knowhere::Status::success);
    printf("[%.3f s] Built %s index with %s algo in %.3f s\n", g_T0.elapsed_seconds(), index_type.c_str(), algo.c_str(),
           build_timer.elapsed_seconds());

    return index.value();
}

// Run search benchmark
static BenchmarkStats
BenchmarkSearch(knowhere::Index<knowhere::IndexNode>& index, const knowhere::DataSetPtr& query_ds,
                const knowhere::DataSetPtr& gt, const std::string& metric, int32_t topk, float drop_ratio_search,
                const std::vector<uint8_t>& filter_data, int32_t num_docs, int32_t num_runs = 3) {
    BenchmarkStats stats;

    knowhere::Json search_conf;
    search_conf[knowhere::meta::METRIC_TYPE] = metric;
    search_conf[knowhere::meta::TOPK] = topk;
    search_conf[knowhere::indexparam::DROP_RATIO_SEARCH] = drop_ratio_search;
    search_conf["refine_factor"] = 1;
    search_conf[knowhere::meta::DIM_MAX_SCORE_RATIO] = 1.05f;

    if (metric == knowhere::metric::BM25) {
        search_conf[knowhere::meta::BM25_K1] = 1.2f;
        search_conf[knowhere::meta::BM25_B] = 0.75f;
        search_conf[knowhere::meta::BM25_AVGDL] = 100.0f;
    }

    knowhere::BitsetView bitset;
    if (!filter_data.empty()) {
        bitset = knowhere::BitsetView(filter_data.data(), num_docs);
    }

    int64_t nq = query_ds->GetRows();

    // Warmup run
    auto warmup_result = index.Search(query_ds, search_conf, bitset);

    // Benchmark runs
    Timer total_timer;
    for (int32_t run = 0; run < num_runs; ++run) {
        Timer run_timer;
        auto result = index.Search(query_ds, search_conf, bitset);
        stats.add_latency(run_timer.elapsed_us() / nq);  // Per-query latency
    }
    stats.total_time_s = total_timer.elapsed_seconds();
    stats.total_queries = nq * num_runs;

    // Calculate recall against ground truth
    if (gt) {
        auto final_result = index.Search(query_ds, search_conf, bitset);
        if (final_result.has_value()) {
            stats.recall = CalcRecall(*gt, *final_result.value());
        }
    }

    return stats;
}

// Run Iterator benchmark (which uses GetAllDistances internally)
// This benchmarks the AnnIterator API which calls GetAllDistances to compute all distances
static BenchmarkStats
BenchmarkIterator(const knowhere::DataSetPtr& train_ds, const knowhere::DataSetPtr& query_ds, const std::string& metric,
                  float drop_ratio_search, const std::vector<uint8_t>& filter_data, int32_t num_docs,
                  int32_t num_runs = 3) {
    BenchmarkStats stats;

    knowhere::Json conf;
    conf[knowhere::meta::METRIC_TYPE] = metric;
    conf[knowhere::indexparam::DROP_RATIO_SEARCH] = drop_ratio_search;

    if (metric == knowhere::metric::BM25) {
        conf[knowhere::meta::BM25_K1] = 1.2f;
        conf[knowhere::meta::BM25_B] = 0.75f;
        conf[knowhere::meta::BM25_AVGDL] = 100.0f;
    }

    knowhere::BitsetView bitset;
    if (!filter_data.empty()) {
        bitset = knowhere::BitsetView(filter_data.data(), num_docs);
    }

    int64_t nq = query_ds->GetRows();

    // Build a TAAT index for Iterator/GetAllDistances
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto index = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
        knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version);

    knowhere::Json build_conf;
    build_conf[knowhere::meta::METRIC_TYPE] = metric;
    build_conf[knowhere::indexparam::INVERTED_INDEX_ALGO] = "TAAT_NAIVE";
    if (metric == knowhere::metric::BM25) {
        build_conf[knowhere::meta::BM25_K1] = 1.2f;
        build_conf[knowhere::meta::BM25_B] = 0.75f;
        build_conf[knowhere::meta::BM25_AVGDL] = 100.0f;
    }

    auto status = index.value().Build(train_ds, build_conf);
    if (status != knowhere::Status::success) {
        return stats;
    }

    // Warmup - create iterators and retrieve first few results
    {
        auto warmup_result = index.value().AnnIterator(query_ds, conf, bitset);
        if (warmup_result.has_value()) {
            for (auto& iter : warmup_result.value()) {
                if (iter) {
                    // Trigger the lazy computation by calling Next a few times
                    for (int i = 0; i < 10 && iter->HasNext(); ++i) {
                        iter->Next();
                    }
                }
            }
        }
    }

    // Benchmark runs - we measure the time to create iterators and iterate through results
    Timer total_timer;
    for (int32_t run = 0; run < num_runs; ++run) {
        Timer run_timer;
        auto result = index.value().AnnIterator(query_ds, conf, bitset);
        if (result.has_value()) {
            for (auto& iter : result.value()) {
                if (iter) {
                    // Iterate through all results to fully exercise GetAllDistances
                    while (iter->HasNext()) {
                        iter->Next();
                    }
                }
            }
        }
        stats.add_latency(run_timer.elapsed_us() / nq);
    }
    stats.total_time_s = total_timer.elapsed_seconds();
    stats.total_queries = nq * num_runs;

    return stats;
}

// Generate ground truth using brute force search
static knowhere::DataSetPtr
GenerateGroundTruth(const knowhere::DataSetPtr& train_ds, const knowhere::DataSetPtr& query_ds,
                    const std::string& metric, int32_t topk, const std::vector<uint8_t>& filter_data,
                    int32_t num_docs) {
    knowhere::Json conf;
    conf[knowhere::meta::METRIC_TYPE] = metric;
    conf[knowhere::meta::TOPK] = topk;

    if (metric == knowhere::metric::BM25) {
        conf[knowhere::meta::BM25_K1] = 1.2f;
        conf[knowhere::meta::BM25_B] = 0.75f;
        conf[knowhere::meta::BM25_AVGDL] = 100.0f;
    }

    knowhere::BitsetView bitset;
    if (!filter_data.empty()) {
        bitset = knowhere::BitsetView(filter_data.data(), num_docs);
    }

    auto result = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, bitset);
    if (!result.has_value()) {
        return nullptr;
    }
    return result.value();
}

// ============================================================================
// Test: Search with different algorithms (TAAT, WAND, MaxScore)
// ============================================================================
TEST_CASE("Benchmark_sparse: TEST_SEARCH_ALGORITHMS", "[benchmark][sparse]") {
    g_T0.reset();
    knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);

    printf("\n[%.3f s] TEST_SEARCH_ALGORITHMS\n", g_T0.elapsed_seconds());
    printf("================================================================================\n");

    DataGenConfig data_config;
    data_config.num_docs = 100000;
    data_config.num_dims = 30000;
    data_config.doc_sparsity = 0.97f;  // ~900 non-zeros per doc
    data_config.query_sparsity = 0.99f;
    data_config.num_queries = 100;
    data_config.distribution = DataDistribution::UNIFORM;

    printf("[%.3f s] Generating data: %d docs, %d dims, sparsity=%.2f\n", g_T0.elapsed_seconds(), data_config.num_docs,
           data_config.num_dims, data_config.doc_sparsity);

    auto train_ds = GenSparseDataSetWithDistribution(data_config);
    auto query_ds = GenSparseQuerySet(data_config);

    std::vector<std::string> algorithms = {"TAAT_NAIVE", "DAAT_WAND", "DAAT_MAXSCORE"};
    std::vector<std::string> metrics = {knowhere::metric::IP};
    std::vector<int32_t> topks = {10, 100};

    for (const auto& metric : metrics) {
        printf("\n--- Metric: %s ---\n", metric.c_str());

        // Generate ground truth with no filter
        auto gt = GenerateGroundTruth(train_ds, query_ds, metric, 100, {}, data_config.num_docs);

        for (const auto& algo : algorithms) {
            auto index = BuildIndex(train_ds, algo, metric);

            for (int32_t topk : topks) {
                std::string test_name = algo + "_" + metric + "_k" + std::to_string(topk);
                auto stats = BenchmarkSearch(index, query_ds, gt, metric, topk, 0.0f, {}, data_config.num_docs);
                PrintBenchmarkResults(test_name, stats);
            }
        }
    }
}

// ============================================================================
// Test: BM25 Search with different algorithms
// ============================================================================
TEST_CASE("Benchmark_sparse: TEST_BM25_SEARCH", "[benchmark][sparse][bm25]") {
    g_T0.reset();
    knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);

    printf("\n[%.3f s] TEST_BM25_SEARCH\n", g_T0.elapsed_seconds());
    printf("================================================================================\n");

    DataGenConfig data_config;
    data_config.num_docs = 100000;
    data_config.num_dims = 30000;
    data_config.doc_sparsity = 0.97f;
    data_config.query_sparsity = 0.99f;
    data_config.num_queries = 100;
    data_config.distribution = DataDistribution::ZIPF;
    data_config.use_integer_values = true;
    data_config.max_tf = 256;

    printf("[%.3f s] Generating BM25 data: %d docs, %d dims, Zipf distribution\n", g_T0.elapsed_seconds(),
           data_config.num_docs, data_config.num_dims);

    auto train_ds = GenSparseDataSetWithDistribution(data_config);
    auto query_ds = GenSparseQuerySet(data_config);

    std::vector<std::string> algorithms = {"TAAT_NAIVE", "DAAT_WAND", "DAAT_MAXSCORE"};
    std::string metric = knowhere::metric::BM25;
    int32_t topk = 10;

    auto gt = GenerateGroundTruth(train_ds, query_ds, metric, topk, {}, data_config.num_docs);

    printf("\n--- BM25 Search WITHOUT Filters ---\n");
    for (const auto& algo : algorithms) {
        auto index = BuildIndex(train_ds, algo, metric);

        std::string test_name = algo + "_BM25_k" + std::to_string(topk);
        auto stats = BenchmarkSearch(index, query_ds, gt, metric, topk, 0.0f, {}, data_config.num_docs);
        PrintBenchmarkResults(test_name, stats);
    }

    // Test with filters to exercise seek() path
    printf("\n--- BM25 Search WITH Filters (exercises galloping search) ---\n");
    std::vector<float> filter_ratios = {0.3f, 0.5f, 0.7f};
    std::vector<FilterDistribution> filter_dists = {
        FilterDistribution::RANDOM,
        FilterDistribution::REVERSE_POSTING_ORDER
    };

    for (const auto& algo : algorithms) {
        if (algo == "TAAT_NAIVE") continue;  // Skip naive for filtered tests

        auto index = BuildIndex(train_ds, algo, metric);

        for (float filter_ratio : filter_ratios) {
            for (auto filter_dist : filter_dists) {
                FilterConfig filter_config;
                filter_config.distribution = filter_dist;
                filter_config.filter_ratio = filter_ratio;

                auto filter_data = GenFilterBitset(data_config.num_docs, filter_config);
                auto filtered_gt = GenerateGroundTruth(train_ds, query_ds, metric, topk, filter_data, data_config.num_docs);

                std::string test_name = algo + "_BM25_filter_" + FilterDistributionToString(filter_dist) + "_" +
                                        std::to_string(static_cast<int>(filter_ratio * 100)) + "pct";
                auto stats = BenchmarkSearch(index, query_ds, filtered_gt, metric, topk, 0.0f, filter_data, data_config.num_docs);
                PrintBenchmarkResults(test_name, stats);
            }
        }
    }
}

// ============================================================================
// Test: Quick BM25 Filter test with DAAT MaxScore (0.7f filter ratio only)
// ============================================================================
TEST_CASE("Benchmark_sparse: TEST_QUICK_BM25_FILTER_MAXSCORE", "[benchmark][sparse][bm25][quick]") {
    g_T0.reset();
    knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);

    printf("\n[%.3f s] TEST_QUICK_BM25_FILTER_MAXSCORE\n", g_T0.elapsed_seconds());
    printf("================================================================================\n");

    DataGenConfig data_config;
    data_config.num_docs = 50000;  // Smaller dataset for quick test
    data_config.num_dims = 30000;
    data_config.doc_sparsity = 0.97f;
    data_config.query_sparsity = 0.99f;
    data_config.num_queries = 50;  // Fewer queries for quick test
    data_config.distribution = DataDistribution::ZIPF;
    data_config.use_integer_values = true;
    data_config.max_tf = 256;

    printf("[%.3f s] Generating BM25 data: %d docs, %d dims, Zipf distribution\n", g_T0.elapsed_seconds(),
           data_config.num_docs, data_config.num_dims);

    auto train_ds = GenSparseDataSetWithDistribution(data_config);
    auto query_ds = GenSparseQuerySet(data_config);

    std::string algorithm = "DAAT_MAXSCORE";
    std::string metric = knowhere::metric::BM25;
    int32_t topk = 10;
    float filter_ratio = 0.7f;  // Only test 0.7f filter ratio

    // Test with filters to exercise seek() path
    printf("\n--- BM25 Search WITH 70%% Filters (DAAT_MAXSCORE only) ---\n");

    auto index = BuildIndex(train_ds, algorithm, metric);

    std::vector<FilterDistribution> filter_dists = {
        FilterDistribution::RANDOM,
        FilterDistribution::REVERSE_POSTING_ORDER
    };

    for (auto filter_dist : filter_dists) {
        FilterConfig filter_config;
        filter_config.distribution = filter_dist;
        filter_config.filter_ratio = filter_ratio;

        auto filter_data = GenFilterBitset(data_config.num_docs, filter_config);
        auto filtered_gt = GenerateGroundTruth(train_ds, query_ds, metric, topk, filter_data, data_config.num_docs);

        std::string test_name = algorithm + "_BM25_filter_" + FilterDistributionToString(filter_dist) + "_70pct";
        auto stats = BenchmarkSearch(index, query_ds, filtered_gt, metric, topk, 0.0f, filter_data, data_config.num_docs);
        PrintBenchmarkResults(test_name, stats);
    }
}

// ============================================================================
// Test: Search with different filter ratios and distributions
// This is critical for evaluating seek-fix and filter-aware-wand optimizations
// ============================================================================
TEST_CASE("Benchmark_sparse: TEST_FILTERED_SEARCH", "[benchmark][sparse][filter]") {
    g_T0.reset();
    knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);

    printf("\n[%.3f s] TEST_FILTERED_SEARCH\n", g_T0.elapsed_seconds());
    printf("================================================================================\n");

    DataGenConfig data_config;
    data_config.num_docs = 100000;
    data_config.num_dims = 30000;
    data_config.doc_sparsity = 0.97f;
    data_config.query_sparsity = 0.99f;
    data_config.num_queries = 100;
    data_config.distribution = DataDistribution::UNIFORM;

    printf("[%.3f s] Generating data: %d docs, %d dims\n", g_T0.elapsed_seconds(), data_config.num_docs,
           data_config.num_dims);

    auto train_ds = GenSparseDataSetWithDistribution(data_config);
    auto query_ds = GenSparseQuerySet(data_config);

    std::vector<std::string> algorithms = {"DAAT_WAND", "DAAT_MAXSCORE"};
    std::string metric = knowhere::metric::IP;
    int32_t topk = 10;

    std::vector<FilterDistribution> filter_distributions = {
        FilterDistribution::FIRST_N,     FilterDistribution::LAST_N,
        FilterDistribution::RANDOM,      FilterDistribution::CLUSTERED,
        FilterDistribution::REVERSE_POSTING_ORDER,
    };

    std::vector<float> filter_ratios = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};

    for (const auto& algo : algorithms) {
        printf("\n=== Algorithm: %s ===\n", algo.c_str());
        auto index = BuildIndex(train_ds, algo, metric);

        for (float filter_ratio : filter_ratios) {
            for (auto filter_dist : filter_distributions) {
                FilterConfig filter_config;
                filter_config.distribution = filter_dist;
                filter_config.filter_ratio = filter_ratio;

                auto filter_data = GenFilterBitset(data_config.num_docs, filter_config);
                auto gt = GenerateGroundTruth(train_ds, query_ds, metric, topk, filter_data, data_config.num_docs);

                std::string test_name = algo + "_filter_" + FilterDistributionToString(filter_dist) + "_" +
                                        std::to_string(static_cast<int>(filter_ratio * 100)) + "pct";

                auto stats =
                    BenchmarkSearch(index, query_ds, gt, metric, topk, 0.0f, filter_data, data_config.num_docs);
                PrintBenchmarkResults(test_name, stats);
            }
        }
    }
}

// ============================================================================
// Test: Iterator/GetAllDistances performance
// This tests the SIMD optimization for compute_all_distances via the Iterator API
// ============================================================================
TEST_CASE("Benchmark_sparse: TEST_ITERATOR", "[benchmark][sparse][iterator]") {
    g_T0.reset();
    knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);

    printf("\n[%.3f s] TEST_ITERATOR (GetAllDistances)\n", g_T0.elapsed_seconds());
    printf("================================================================================\n");

    DataGenConfig data_config;
    data_config.num_docs = 50000;
    data_config.num_dims = 30000;
    data_config.doc_sparsity = 0.97f;
    data_config.query_sparsity = 0.99f;
    data_config.num_queries = 50;
    data_config.distribution = DataDistribution::UNIFORM;

    printf("[%.3f s] Generating data: %d docs, %d dims\n", g_T0.elapsed_seconds(), data_config.num_docs,
           data_config.num_dims);

    auto train_ds = GenSparseDataSetWithDistribution(data_config);
    auto query_ds = GenSparseQuerySet(data_config);

    std::vector<std::string> metrics = {knowhere::metric::IP, knowhere::metric::BM25};

    for (const auto& metric : metrics) {
        printf("\n--- Metric: %s ---\n", metric.c_str());

        // Without filter
        {
            std::string test_name = "Iterator_" + std::string(metric) + "_no_filter";
            auto stats = BenchmarkIterator(train_ds, query_ds, metric, 0.0f, {}, data_config.num_docs);
            PrintBenchmarkResults(test_name, stats);
        }

        // With different filter ratios
        std::vector<float> filter_ratios = {0.3f, 0.5f, 0.7f};
        for (float filter_ratio : filter_ratios) {
            FilterConfig filter_config;
            filter_config.distribution = FilterDistribution::RANDOM;
            filter_config.filter_ratio = filter_ratio;
            auto filter_data = GenFilterBitset(data_config.num_docs, filter_config);

            std::string test_name = "Iterator_" + std::string(metric) + "_filter_" +
                                    std::to_string(static_cast<int>(filter_ratio * 100)) + "pct";
            auto stats = BenchmarkIterator(train_ds, query_ds, metric, 0.0f, filter_data, data_config.num_docs);
            PrintBenchmarkResults(test_name, stats);
        }
    }
}

// ============================================================================
// Test: Cursor seek/next performance with different data distributions
// This helps evaluate the seek-fix optimization
// ============================================================================
TEST_CASE("Benchmark_sparse: TEST_CURSOR_SEEK_PATTERNS", "[benchmark][sparse][seek]") {
    g_T0.reset();
    knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);

    printf("\n[%.3f s] TEST_CURSOR_SEEK_PATTERNS\n", g_T0.elapsed_seconds());
    printf("================================================================================\n");

    std::vector<DataDistribution> distributions = {DataDistribution::UNIFORM, DataDistribution::ZIPF,
                                                   DataDistribution::SKEWED_DIMS};

    std::string metric = knowhere::metric::IP;
    int32_t topk = 10;

    for (auto dist : distributions) {
        DataGenConfig data_config;
        data_config.num_docs = 100000;
        data_config.num_dims = 30000;
        data_config.doc_sparsity = 0.97f;
        data_config.query_sparsity = 0.99f;
        data_config.num_queries = 100;
        data_config.distribution = dist;

        printf("\n=== Data Distribution: %s ===\n", DataDistributionToString(dist).c_str());
        printf("[%.3f s] Generating data\n", g_T0.elapsed_seconds());

        auto train_ds = GenSparseDataSetWithDistribution(data_config);
        auto query_ds = GenSparseQuerySet(data_config);
        auto gt = GenerateGroundTruth(train_ds, query_ds, metric, topk, {}, data_config.num_docs);

        std::vector<std::string> algorithms = {"DAAT_WAND", "DAAT_MAXSCORE"};

        for (const auto& algo : algorithms) {
            auto index = BuildIndex(train_ds, algo, metric);

            std::vector<std::pair<FilterDistribution, float>> filter_configs = {
                {FilterDistribution::NONE, 0.0f},
                {FilterDistribution::FIRST_N, 0.5f},
                {FilterDistribution::LAST_N, 0.5f},
                {FilterDistribution::ALTERNATING, 0.5f},
            };

            for (auto& [filter_dist, filter_ratio] : filter_configs) {
                FilterConfig filter_config;
                filter_config.distribution = filter_dist;
                filter_config.filter_ratio = filter_ratio;
                auto filter_data = GenFilterBitset(data_config.num_docs, filter_config);
                auto filtered_gt =
                    GenerateGroundTruth(train_ds, query_ds, metric, topk, filter_data, data_config.num_docs);

                std::string test_name =
                    algo + "_" + DataDistributionToString(dist) + "_" + FilterDistributionToString(filter_dist);

                auto stats =
                    BenchmarkSearch(index, query_ds, filtered_gt, metric, topk, 0.0f, filter_data, data_config.num_docs);
                PrintBenchmarkResults(test_name, stats);
            }
        }
    }
}

// ============================================================================
// Test: Impact of drop_ratio_search on performance and recall
// ============================================================================
TEST_CASE("Benchmark_sparse: TEST_DROP_RATIO_SEARCH", "[benchmark][sparse][drop_ratio]") {
    g_T0.reset();
    knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);

    printf("\n[%.3f s] TEST_DROP_RATIO_SEARCH\n", g_T0.elapsed_seconds());
    printf("================================================================================\n");

    DataGenConfig data_config;
    data_config.num_docs = 100000;
    data_config.num_dims = 30000;
    data_config.doc_sparsity = 0.97f;
    data_config.query_sparsity = 0.99f;
    data_config.num_queries = 100;
    data_config.distribution = DataDistribution::UNIFORM;

    printf("[%.3f s] Generating data\n", g_T0.elapsed_seconds());

    auto train_ds = GenSparseDataSetWithDistribution(data_config);
    auto query_ds = GenSparseQuerySet(data_config);

    std::string metric = knowhere::metric::IP;
    int32_t topk = 10;

    auto gt = GenerateGroundTruth(train_ds, query_ds, metric, topk, {}, data_config.num_docs);

    std::vector<std::string> algorithms = {"DAAT_WAND", "DAAT_MAXSCORE"};
    std::vector<float> drop_ratios = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

    for (const auto& algo : algorithms) {
        printf("\n=== Algorithm: %s ===\n", algo.c_str());
        auto index = BuildIndex(train_ds, algo, metric);

        for (float drop_ratio : drop_ratios) {
            std::string test_name = algo + "_drop_ratio_" + std::to_string(static_cast<int>(drop_ratio * 100)) + "pct";

            knowhere::Json search_conf;
            search_conf[knowhere::meta::METRIC_TYPE] = metric;
            search_conf[knowhere::meta::TOPK] = topk;
            search_conf[knowhere::indexparam::DROP_RATIO_SEARCH] = drop_ratio;
            search_conf["refine_factor"] = 1;
            search_conf[knowhere::meta::DIM_MAX_SCORE_RATIO] = 1.05f;

            BenchmarkStats stats;
            int32_t num_runs = 3;
            int64_t nq = query_ds->GetRows();

            // Warmup
            index.Search(query_ds, search_conf, nullptr);

            Timer total_timer;
            for (int32_t run = 0; run < num_runs; ++run) {
                Timer run_timer;
                auto result = index.Search(query_ds, search_conf, nullptr);
                stats.add_latency(run_timer.elapsed_us() / nq);
            }
            stats.total_time_s = total_timer.elapsed_seconds();
            stats.total_queries = nq * num_runs;

            // Calculate recall
            auto final_result = index.Search(query_ds, search_conf, nullptr);
            if (final_result.has_value() && gt) {
                stats.recall = CalcRecall(*gt, *final_result.value());
            }

            PrintBenchmarkResults(test_name, stats);
        }
    }
}

// ============================================================================
// Test: Scalability with different dataset sizes
// ============================================================================
TEST_CASE("Benchmark_sparse: TEST_SCALABILITY", "[benchmark][sparse][scalability]") {
    g_T0.reset();
    knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);

    printf("\n[%.3f s] TEST_SCALABILITY\n", g_T0.elapsed_seconds());
    printf("================================================================================\n");

    std::vector<int32_t> num_docs_list = {10000, 50000, 100000, 200000};
    std::string metric = knowhere::metric::IP;
    int32_t topk = 10;
    std::string algo = "DAAT_MAXSCORE";

    for (int32_t num_docs : num_docs_list) {
        DataGenConfig data_config;
        data_config.num_docs = num_docs;
        data_config.num_dims = 30000;
        data_config.doc_sparsity = 0.97f;
        data_config.query_sparsity = 0.99f;
        data_config.num_queries = 100;
        data_config.distribution = DataDistribution::UNIFORM;

        printf("\n=== Dataset Size: %d docs ===\n", num_docs);
        printf("[%.3f s] Generating data\n", g_T0.elapsed_seconds());

        auto train_ds = GenSparseDataSetWithDistribution(data_config);
        auto query_ds = GenSparseQuerySet(data_config);
        auto gt = GenerateGroundTruth(train_ds, query_ds, metric, topk, {}, data_config.num_docs);

        auto index = BuildIndex(train_ds, algo, metric);

        std::string test_name = algo + "_" + std::to_string(num_docs) + "_docs";
        auto stats = BenchmarkSearch(index, query_ds, gt, metric, topk, 0.0f, {}, data_config.num_docs);
        PrintBenchmarkResults(test_name, stats);

        // Also test with filter
        FilterConfig filter_config;
        filter_config.distribution = FilterDistribution::RANDOM;
        filter_config.filter_ratio = 0.5f;
        auto filter_data = GenFilterBitset(data_config.num_docs, filter_config);
        auto filtered_gt = GenerateGroundTruth(train_ds, query_ds, metric, topk, filter_data, data_config.num_docs);

        std::string filtered_test_name = algo + "_" + std::to_string(num_docs) + "_docs_filtered_50pct";
        auto filtered_stats =
            BenchmarkSearch(index, query_ds, filtered_gt, metric, topk, 0.0f, filter_data, data_config.num_docs);
        PrintBenchmarkResults(filtered_test_name, filtered_stats);
    }
}

// ============================================================================
// Test: Different sparsity levels
// ============================================================================
TEST_CASE("Benchmark_sparse: TEST_SPARSITY_LEVELS", "[benchmark][sparse][sparsity]") {
    g_T0.reset();
    knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);

    printf("\n[%.3f s] TEST_SPARSITY_LEVELS\n", g_T0.elapsed_seconds());
    printf("================================================================================\n");

    std::vector<float> sparsity_levels = {0.95f, 0.97f, 0.99f, 0.995f};
    std::string metric = knowhere::metric::IP;
    int32_t topk = 10;
    std::string algo = "DAAT_MAXSCORE";

    for (float sparsity : sparsity_levels) {
        DataGenConfig data_config;
        data_config.num_docs = 100000;
        data_config.num_dims = 30000;
        data_config.doc_sparsity = sparsity;
        data_config.query_sparsity = sparsity + 0.005f;
        data_config.num_queries = 100;
        data_config.distribution = DataDistribution::UNIFORM;

        int32_t avg_nnz = static_cast<int32_t>(data_config.num_dims * (1.0f - sparsity));
        printf("\n=== Sparsity: %.3f (avg nnz per doc: %d) ===\n", sparsity, avg_nnz);
        printf("[%.3f s] Generating data\n", g_T0.elapsed_seconds());

        auto train_ds = GenSparseDataSetWithDistribution(data_config);
        auto query_ds = GenSparseQuerySet(data_config);
        auto gt = GenerateGroundTruth(train_ds, query_ds, metric, topk, {}, data_config.num_docs);

        auto index = BuildIndex(train_ds, algo, metric);

        std::string test_name = algo + "_sparsity_" + std::to_string(static_cast<int>(sparsity * 1000)) + "ppt";
        auto stats = BenchmarkSearch(index, query_ds, gt, metric, topk, 0.0f, {}, data_config.num_docs);
        PrintBenchmarkResults(test_name, stats);
    }
}
