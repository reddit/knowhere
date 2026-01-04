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

#include <thread>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <nlohmann/json.hpp>

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

// Get current git branch name
static std::string
GetGitBranch() {
    FILE* fp = popen("git rev-parse --abbrev-ref HEAD 2>/dev/null", "r");
    if (!fp) return "unknown";

    char buffer[128];
    std::string result;
    while (fgets(buffer, sizeof(buffer), fp) != nullptr) {
        result += buffer;
    }
    pclose(fp);

    // Remove trailing newline
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    return result.empty() ? "unknown" : result;
}

// Get git commit hash
static std::string
GetGitCommit() {
    FILE* fp = popen("git rev-parse HEAD 2>/dev/null", "r");
    if (!fp) return "unknown";

    char buffer[128];
    std::string result;
    while (fgets(buffer, sizeof(buffer), fp) != nullptr) {
        result += buffer;
    }
    pclose(fp);

    // Remove trailing newline
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    return result.empty() ? "unknown" : result;
}

// Structure to hold complete benchmark result information
struct BenchmarkResult {
    // Git information
    std::string git_branch;
    std::string git_commit;

    // Test metadata
    std::string test_name;
    std::string algorithm;
    std::string metric;
    int32_t topk = 0;
    float drop_ratio_search = 0.0f;
    float drop_ratio_build = 0.0f;

    // Data configuration
    DataGenConfig data_config;

    // Filter configuration
    FilterConfig filter_config;
    bool has_filter = false;

    // Performance results
    BenchmarkStats stats;

    // Timing and system information
    std::string timestamp;
    std::string start_time;
    std::string end_time;
    size_t num_search_threads = 0;

    nlohmann::json to_json() const {
        nlohmann::json j;

        // Git information
        j["git"]["branch"] = git_branch;
        j["git"]["commit"] = git_commit;

        // Test metadata
        j["test"]["name"] = test_name;
        j["test"]["algorithm"] = algorithm;
        j["test"]["metric"] = metric;
        j["test"]["topk"] = topk;
        j["test"]["drop_ratio_search"] = drop_ratio_search;
        j["test"]["drop_ratio_build"] = drop_ratio_build;
        j["test"]["timestamp"] = timestamp;
        j["test"]["start_time"] = start_time;
        j["test"]["end_time"] = end_time;
        j["test"]["num_search_threads"] = num_search_threads;

        // Data configuration
        j["data"]["num_docs"] = data_config.num_docs;
        j["data"]["num_dims"] = data_config.num_dims;
        j["data"]["doc_sparsity"] = data_config.doc_sparsity;
        j["data"]["query_sparsity"] = data_config.query_sparsity;
        j["data"]["num_queries"] = data_config.num_queries;
        j["data"]["distribution"] = DataDistributionToString(data_config.distribution);
        j["data"]["zipf_alpha"] = data_config.zipf_alpha;
        j["data"]["max_value"] = data_config.max_value;
        j["data"]["use_integer_values"] = data_config.use_integer_values;
        j["data"]["max_tf"] = data_config.max_tf;
        j["data"]["seed"] = data_config.seed;

        // Filter configuration
        j["filter"]["has_filter"] = has_filter;
        if (has_filter) {
            j["filter"]["distribution"] = FilterDistributionToString(filter_config.distribution);
            j["filter"]["filter_ratio"] = filter_config.filter_ratio;
            j["filter"]["seed"] = filter_config.seed;
        }

        // Performance results
        j["performance"]["total_queries"] = stats.total_queries;
        j["performance"]["total_time_s"] = stats.total_time_s;
        j["performance"]["qps"] = stats.qps();
        j["performance"]["mean_latency_us"] = stats.mean_latency_us();
        j["performance"]["p50_latency_us"] = stats.p50_latency_us();
        j["performance"]["p90_latency_us"] = stats.p90_latency_us();
        j["performance"]["p99_latency_us"] = stats.p99_latency_us();
        j["performance"]["recall"] = stats.recall;

        return j;
    }
};

// Write benchmark result to JSON file
static void
WriteBenchmarkResult(const BenchmarkResult& result) {
    // Create results directory if it doesn't exist
    std::filesystem::create_directories("benchmark_results");

    // Generate filename with timestamp and test name
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y%m%d_%H%M%S");
    std::string timestamp_str = ss.str();

    std::string filename = "benchmark_results/" + result.git_branch + "_" + result.test_name + "_" + timestamp_str + ".json";

    // Write JSON to file
    std::ofstream file(filename);
    if (file.is_open()) {
        file << result.to_json().dump(2);
        file.close();
        printf("JSON result written to: %s\n", filename.c_str());
    } else {
        printf("Failed to write JSON result to: %s\n", filename.c_str());
    }
}

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

// Helper to create a single-query dataset from a sparse query dataset
static knowhere::DataSetPtr
CreateSingleQueryDataset(const knowhere::DataSetPtr& query_ds, int64_t query_idx) {
    auto rows = query_ds->GetTensor();
    auto sparse_rows = static_cast<const knowhere::sparse::SparseRow<float>*>(rows);

    // Copy the single query row
    auto single_row = std::make_unique<knowhere::sparse::SparseRow<float>[]>(1);
    single_row[0] = sparse_rows[query_idx];  // Copy constructor

    auto ds = knowhere::GenDataSet(1, query_ds->GetDim(), single_row.release());
    ds->SetIsOwner(true);
    ds->SetIsSparse(true);
    return ds;
}

// Run search benchmark with individual query latency tracking
static BenchmarkStats
BenchmarkSearch(knowhere::Index<knowhere::IndexNode>& index, const knowhere::DataSetPtr& query_ds,
                const knowhere::DataSetPtr& gt, const std::string& metric, int32_t topk, float drop_ratio_search,
                const std::vector<uint8_t>& filter_data, int32_t num_docs, int32_t num_runs = 3,
                const std::string& test_name = "", const std::string& algorithm = "",
                const DataGenConfig& data_config = {}, const FilterConfig& filter_config = {}, bool has_filter = false,
                const std::string& start_time_str = "") {
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

    // Warmup run (batch)
    auto warmup_result = index.Search(query_ds, search_conf, bitset);

    // Benchmark runs - search queries individually to get per-query latencies
    Timer total_timer;
    for (int32_t run = 0; run < num_runs; ++run) {
        for (int64_t q = 0; q < nq; ++q) {
            auto single_query = CreateSingleQueryDataset(query_ds, q);
            Timer query_timer;
            auto result = index.Search(single_query, search_conf, bitset);
            stats.add_latency(query_timer.elapsed_us());
        }
    }
    stats.total_time_s = total_timer.elapsed_seconds();
    stats.total_queries = nq * num_runs;

    // Calculate recall against ground truth (batch search is fine for recall)
    if (gt) {
        auto final_result = index.Search(query_ds, search_conf, bitset);
        if (final_result.has_value()) {
            stats.recall = CalcRecall(*gt, *final_result.value());
        }
    }

    // Write JSON result if test metadata is provided
    if (!test_name.empty()) {
        // Capture end time
        auto end_time_point = std::chrono::system_clock::now();
        auto end_time_t = std::chrono::system_clock::to_time_t(end_time_point);
        std::stringstream end_ss;
        end_ss << std::put_time(std::gmtime(&end_time_t), "%Y-%m-%dT%H:%M:%SZ");
        std::string end_time_str = end_ss.str();

        BenchmarkResult result;
        result.git_branch = GetGitBranch();
        result.git_commit = GetGitCommit();
        result.test_name = test_name;
        result.algorithm = algorithm;
        result.metric = metric;
        result.topk = topk;
        result.drop_ratio_search = drop_ratio_search;
        result.data_config = data_config;
        result.filter_config = filter_config;
        result.has_filter = has_filter;
        result.stats = stats;

        // Set timing information
        result.start_time = start_time_str;
        result.end_time = end_time_str;
        result.num_search_threads = knowhere::KnowhereConfig::GetSearchThreadPoolSize();

        // Set timestamp (same as end time for backward compatibility)
        result.timestamp = end_time_str;

        WriteBenchmarkResult(result);
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

    // Benchmark runs - measure per-query latency by processing queries individually
    Timer total_timer;
    for (int32_t run = 0; run < num_runs; ++run) {
        for (int64_t q = 0; q < nq; ++q) {
            auto single_query = CreateSingleQueryDataset(query_ds, q);
            Timer query_timer;
            auto result = index.value().AnnIterator(single_query, conf, bitset);
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
            stats.add_latency(query_timer.elapsed_us());
        }
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
    data_config.query_sparsity = 0.9998f;  // ~6 terms per query instead of 300
    data_config.num_queries = 500;
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

                // Capture start time
                auto start_time_point = std::chrono::system_clock::now();
                auto start_time_t = std::chrono::system_clock::to_time_t(start_time_point);
                std::stringstream start_ss;
                start_ss << std::put_time(std::gmtime(&start_time_t), "%Y-%m-%dT%H:%M:%SZ");
                std::string start_time_str = start_ss.str();

                auto stats = BenchmarkSearch(index, query_ds, gt, metric, topk, 0.0f, {}, data_config.num_docs, 3,
                                           test_name, algo, data_config, {}, false, start_time_str);
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
    data_config.query_sparsity = 0.9998f;  // ~6 terms per query instead of 300
    data_config.num_queries = 500;
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

        // Capture start time
        auto start_time_point = std::chrono::system_clock::now();
        auto start_time_t = std::chrono::system_clock::to_time_t(start_time_point);
        std::stringstream start_ss;
        start_ss << std::put_time(std::gmtime(&start_time_t), "%Y-%m-%dT%H:%M:%SZ");
        std::string start_time_str = start_ss.str();

        auto stats = BenchmarkSearch(index, query_ds, gt, metric, topk, 0.0f, {}, data_config.num_docs, 3,
                                   test_name, algo, data_config, {}, false, start_time_str);
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
                FilterConfig current_filter_config = filter_config;
                current_filter_config.distribution = filter_dist;
                current_filter_config.filter_ratio = filter_ratio;

                // Capture start time
                auto start_time_point = std::chrono::system_clock::now();
                auto start_time_t = std::chrono::system_clock::to_time_t(start_time_point);
                std::stringstream start_ss;
                start_ss << std::put_time(std::gmtime(&start_time_t), "%Y-%m-%dT%H:%M:%SZ");
                std::string start_time_str = start_ss.str();

                auto stats = BenchmarkSearch(index, query_ds, filtered_gt, metric, topk, 0.0f, filter_data, data_config.num_docs, 3,
                                           test_name, algo, data_config, current_filter_config, true, start_time_str);
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
    // Use all available CPU cores for maximum QPS
    knowhere::KnowhereConfig::SetSearchThreadPoolSize(std::thread::hardware_concurrency());

    printf("\n[%.3f s] TEST_QUICK_BM25_FILTER_MAXSCORE\n", g_T0.elapsed_seconds());
    printf("================================================================================\n");

    DataGenConfig data_config;
    data_config.num_docs = 50000;  // Smaller dataset for quick test
    data_config.num_dims = 30000;
    data_config.doc_sparsity = 0.97f;
    data_config.query_sparsity = 0.9998f;  // ~6 terms per query instead of 300
    data_config.num_queries = 100;  // Fewer queries for quick test
    data_config.distribution = DataDistribution::ZIPF;
    data_config.use_integer_values = true;
    data_config.max_tf = 256;

    printf("[%.3f s] Generating BM25 data: %d docs, %d dims, Zipf distribution (~6 terms/query)\n", g_T0.elapsed_seconds(),
           data_config.num_docs, data_config.num_dims);

    auto train_ds = GenSparseDataSetWithDistribution(data_config);
    auto query_ds = GenSparseQuerySet(data_config);

    std::string algorithm = "DAAT_MAXSCORE";
    std::string metric = knowhere::metric::BM25;
    int32_t topk = 10;
    std::vector<float> filter_ratios = {0.7f, 0.9f};  // Test both 70% and 90% filter ratios

    // Test with filters to exercise seek() path
    printf("\n--- BM25 Search WITH 70%% and 90%% Filters (DAAT_MAXSCORE only) ---\n");

    auto index = BuildIndex(train_ds, algorithm, metric);

    std::vector<FilterDistribution> filter_dists = {
        FilterDistribution::RANDOM,
        FilterDistribution::REVERSE_POSTING_ORDER
    };

    for (float filter_ratio : filter_ratios) {
        for (auto filter_dist : filter_dists) {
            FilterConfig filter_config;
            filter_config.distribution = filter_dist;
            filter_config.filter_ratio = filter_ratio;

            auto filter_data = GenFilterBitset(data_config.num_docs, filter_config);
            auto filtered_gt = GenerateGroundTruth(train_ds, query_ds, metric, topk, filter_data, data_config.num_docs);

            std::string test_name = algorithm + "_BM25_filter_" + FilterDistributionToString(filter_dist) + "_" +
                                    std::to_string(static_cast<int>(filter_ratio * 100)) + "pct";
            FilterConfig current_filter_config;
            current_filter_config.distribution = filter_dist;
            current_filter_config.filter_ratio = filter_ratio;

            // Capture start time
            auto start_time_point = std::chrono::system_clock::now();
            auto start_time_t = std::chrono::system_clock::to_time_t(start_time_point);
            std::stringstream start_ss;
            start_ss << std::put_time(std::gmtime(&start_time_t), "%Y-%m-%dT%H:%M:%SZ");
            std::string start_time_str = start_ss.str();

            auto stats = BenchmarkSearch(index, query_ds, filtered_gt, metric, topk, 0.0f, filter_data, data_config.num_docs, 3,
                                       test_name, algorithm, data_config, current_filter_config, true, start_time_str);
            PrintBenchmarkResults(test_name, stats);
        }
    }
}

// ============================================================================
// Test: Search with different filter ratios and distributions
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
    data_config.query_sparsity = 0.9998f;  // ~6 terms per query instead of 300
    data_config.num_queries = 500;
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

                // Capture start time
                auto start_time_point = std::chrono::system_clock::now();
                auto start_time_t = std::chrono::system_clock::to_time_t(start_time_point);
                std::stringstream start_ss;
                start_ss << std::put_time(std::gmtime(&start_time_t), "%Y-%m-%dT%H:%M:%SZ");
                std::string start_time_str = start_ss.str();

                auto stats =
                    BenchmarkSearch(index, query_ds, gt, metric, topk, 0.0f, filter_data, data_config.num_docs, 3,
                                   test_name, algo, data_config, filter_config, true, start_time_str);
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
    data_config.query_sparsity = 0.9998f;  // ~6 terms per query instead of 300
    data_config.num_queries = 500;
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
        data_config.query_sparsity = 0.9998f;  // ~6 terms per query instead of 300
        data_config.num_queries = 500;
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

                FilterConfig current_filter_config;
                current_filter_config.distribution = filter_dist;
                current_filter_config.filter_ratio = filter_ratio;

                // Capture start time
                auto start_time_point = std::chrono::system_clock::now();
                auto start_time_t = std::chrono::system_clock::to_time_t(start_time_point);
                std::stringstream start_ss;
                start_ss << std::put_time(std::gmtime(&start_time_t), "%Y-%m-%dT%H:%M:%SZ");
                std::string start_time_str = start_ss.str();

                auto stats =
                    BenchmarkSearch(index, query_ds, filtered_gt, metric, topk, 0.0f, filter_data, data_config.num_docs, 3,
                                   test_name, algo, data_config, current_filter_config, filter_dist != FilterDistribution::NONE, start_time_str);
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
    data_config.query_sparsity = 0.9998f;  // ~6 terms per query instead of 300
    data_config.num_queries = 500;
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

            // Benchmark with per-query latency tracking
            Timer total_timer;
            for (int32_t run = 0; run < num_runs; ++run) {
                for (int64_t q = 0; q < nq; ++q) {
                    auto single_query = CreateSingleQueryDataset(query_ds, q);
                    Timer query_timer;
                    auto result = index.Search(single_query, search_conf, nullptr);
                    stats.add_latency(query_timer.elapsed_us());
                }
            }
            stats.total_time_s = total_timer.elapsed_seconds();
            stats.total_queries = nq * num_runs;

            // Calculate recall
            auto final_result = index.Search(query_ds, search_conf, nullptr);
            if (final_result.has_value() && gt) {
                stats.recall = CalcRecall(*gt, *final_result.value());
            }

            // Write JSON result for drop ratio test
            BenchmarkResult result;
            result.git_branch = GetGitBranch();
            result.git_commit = GetGitCommit();
            result.test_name = test_name;
            result.algorithm = algo;
            result.metric = metric;
            result.topk = topk;
            result.drop_ratio_search = drop_ratio;
            result.data_config = data_config;
            result.filter_config = {};
            result.has_filter = false;
            result.stats = stats;

            // Set timing information
            result.start_time = "";  // Drop ratio test doesn't capture start time separately
            auto end_time_point = std::chrono::system_clock::now();
            auto end_time_t = std::chrono::system_clock::to_time_t(end_time_point);
            std::stringstream end_ss;
            end_ss << std::put_time(std::gmtime(&end_time_t), "%Y-%m-%dT%H:%M:%SZ");
            result.end_time = end_ss.str();
            result.num_search_threads = knowhere::KnowhereConfig::GetSearchThreadPoolSize();

            // Set timestamp (same as end time for backward compatibility)
            result.timestamp = result.end_time;

            WriteBenchmarkResult(result);

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
        data_config.query_sparsity = 0.9998f;  // ~6 terms per query instead of 300
        data_config.num_queries = 500;
        data_config.distribution = DataDistribution::UNIFORM;

        printf("\n=== Dataset Size: %d docs ===\n", num_docs);
        printf("[%.3f s] Generating data\n", g_T0.elapsed_seconds());

        auto train_ds = GenSparseDataSetWithDistribution(data_config);
        auto query_ds = GenSparseQuerySet(data_config);
        auto gt = GenerateGroundTruth(train_ds, query_ds, metric, topk, {}, data_config.num_docs);

        auto index = BuildIndex(train_ds, algo, metric);

        std::string test_name = algo + "_" + std::to_string(num_docs) + "_docs";

        // Capture start time
        auto start_time_point = std::chrono::system_clock::now();
        auto start_time_t = std::chrono::system_clock::to_time_t(start_time_point);
        std::stringstream start_ss;
        start_ss << std::put_time(std::gmtime(&start_time_t), "%Y-%m-%dT%H:%M:%SZ");
        std::string start_time_str = start_ss.str();

        auto stats = BenchmarkSearch(index, query_ds, gt, metric, topk, 0.0f, {}, data_config.num_docs, 3,
                                   test_name, algo, data_config, {}, false, start_time_str);
        PrintBenchmarkResults(test_name, stats);

        // Also test with filter
        {
            FilterConfig filter_config;
            filter_config.distribution = FilterDistribution::RANDOM;
            filter_config.filter_ratio = 0.5f;
            auto filter_data = GenFilterBitset(data_config.num_docs, filter_config);
            auto filtered_gt = GenerateGroundTruth(train_ds, query_ds, metric, topk, filter_data, data_config.num_docs);

            std::string filtered_test_name = algo + "_" + std::to_string(num_docs) + "_docs_filtered_50pct";
            FilterConfig current_filter_config;
            current_filter_config.distribution = FilterDistribution::RANDOM;
            current_filter_config.filter_ratio = 0.5f;

            // Capture start time
            auto start_time_point = std::chrono::system_clock::now();
            auto start_time_t = std::chrono::system_clock::to_time_t(start_time_point);
            std::stringstream start_ss;
            start_ss << std::put_time(std::gmtime(&start_time_t), "%Y-%m-%dT%H:%M:%SZ");
            std::string start_time_str = start_ss.str();

            auto filtered_stats =
                BenchmarkSearch(index, query_ds, filtered_gt, metric, topk, 0.0f, filter_data, data_config.num_docs, 3,
                               filtered_test_name, algo, data_config, current_filter_config, true, start_time_str);
            PrintBenchmarkResults(filtered_test_name, filtered_stats);
        }
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
        data_config.num_queries = 500;
        data_config.distribution = DataDistribution::UNIFORM;

        int32_t avg_nnz = static_cast<int32_t>(data_config.num_dims * (1.0f - sparsity));
        printf("\n=== Sparsity: %.3f (avg nnz per doc: %d) ===\n", sparsity, avg_nnz);
        printf("[%.3f s] Generating data\n", g_T0.elapsed_seconds());

        auto train_ds = GenSparseDataSetWithDistribution(data_config);
        auto query_ds = GenSparseQuerySet(data_config);
        auto gt = GenerateGroundTruth(train_ds, query_ds, metric, topk, {}, data_config.num_docs);

        auto index = BuildIndex(train_ds, algo, metric);

        std::string test_name = algo + "_sparsity_" + std::to_string(static_cast<int>(sparsity * 1000)) + "ppt";

        // Capture start time
        auto start_time_point = std::chrono::system_clock::now();
        auto start_time_t = std::chrono::system_clock::to_time_t(start_time_point);
        std::stringstream start_ss;
        start_ss << std::put_time(std::gmtime(&start_time_t), "%Y-%m-%dT%H:%M:%SZ");
        std::string start_time_str = start_ss.str();

        auto stats = BenchmarkSearch(index, query_ds, gt, metric, topk, 0.0f, {}, data_config.num_docs, 3,
                                   test_name, algo, data_config, {}, false, start_time_str);
        PrintBenchmarkResults(test_name, stats);
    }
}
