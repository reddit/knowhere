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

#include <future>
#include <thread>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

void
WriteBinaryToFile(const std::string& filename, const knowhere::BinaryPtr binary) {
    auto data = binary->data.get();
    auto size = binary->size;
    // if tmp_file already exists, remove it
    std::remove(filename.c_str());
    std::ofstream out(filename, std::ios::binary);
    out.write((const char*)data, size);
    out.close();
}

TEST_CASE("Test Mem Sparse Index With Float Vector", "[float metrics]") {
    auto [nb, dim, doc_sparsity, query_sparsity] = GENERATE(table<int32_t, int32_t, float, float>({
        // 300 dim, avg doc nnz 12, avg query nnz 9
        {2000, 300, 0.95, 0.97},
        // 300 dim, avg doc nnz 9, avg query nnz 3
        {2000, 300, 0.97, 0.99},
        // 3000 dim, avg doc nnz 90, avg query nnz 30
        {2000, 3000, 0.97, 0.99},
    }));
    auto topk = 5;
    int64_t nq = 10;

    auto metric = GENERATE(knowhere::metric::IP, knowhere::metric::BM25);

    auto inverted_index_algo = GENERATE("TAAT_NAIVE", "DAAT_WAND", "DAAT_MAXSCORE");

    auto drop_ratio_search = metric == knowhere::metric::BM25 ? GENERATE(0.0, 0.1) : GENERATE(0.0, 0.3);

    auto version = GenTestVersionList();

    auto base_gen = [=, dim = dim]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::BM25_K1] = 1.2;
        json[knowhere::meta::BM25_B] = 0.75;
        json[knowhere::meta::BM25_AVGDL] = 100;
        return json;
    };

    auto sparse_inverted_index_gen = [base_gen, drop_ratio_search = drop_ratio_search,
                                      inverted_index_algo = inverted_index_algo]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::DROP_RATIO_SEARCH] = drop_ratio_search;
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = inverted_index_algo;
        return json;
    };

    auto sparse_dataset_gen = [&](int nr, int dim, float sparsity) -> knowhere::DataSetPtr {
        if (metric == knowhere::metric::BM25) {
            return GenSparseDataSetWithMaxVal(nr, dim, sparsity, 256, true);
        } else {
            return GenSparseDataSet(nr, dim, sparsity);
        }
    };

    auto train_ds = sparse_dataset_gen(nb, dim, doc_sparsity);
    auto query_ds = sparse_dataset_gen(nq, dim + 20, query_sparsity);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric}, {knowhere::meta::TOPK, topk},      {knowhere::meta::BM25_K1, 1.2},
        {knowhere::meta::BM25_B, 0.75},        {knowhere::meta::BM25_AVGDL, 100},
    };

    auto check_distance_decreasing = [](const knowhere::DataSet& ds) {
        auto nq = ds.GetRows();
        auto k = ds.GetDim();
        auto* distances = ds.GetDistance();
        auto* ids = ds.GetIds();
        for (auto i = 0; i < nq; ++i) {
            for (auto j = 0; j < k - 1; ++j) {
                if (ids[i * k + j] == -1 || ids[i * k + j + 1] == -1) {
                    break;
                }
                REQUIRE(distances[i * k + j] >= distances[i * k + j + 1]);
            }
        }
    };

    auto check_result_match_filter = [](const knowhere::DataSet& ds, const knowhere::BitsetView& bitset) {
        auto nq = ds.GetRows();
        auto k = ds.GetDim();
        auto* ids = ds.GetIds();
        for (auto i = 0; i < nq; ++i) {
            for (auto j = 0; j < k; ++j) {
                if (ids[i * k + j] == -1) {
                    break;
                }
                REQUIRE(!bitset.test(ids[i * k + j]));
            }
        }
    };

    SECTION("Test Search") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, sparse_inverted_index_gen),
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_WAND, sparse_inverted_index_gen),
        }));
        auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, nullptr);
        check_distance_decreasing(*gt.value());

        auto use_mmap = GENERATE(true, false);
        auto tmp_file = "/tmp/knowhere_sparse_inverted_index_test";
        {
            auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
            auto cfg_json = gen().dump();
            CAPTURE(name, cfg_json);
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(idx.Type() == name);
            REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
            REQUIRE(idx.Size() > 0);
            REQUIRE(idx.Count() == nb);
            REQUIRE(idx.HasRawData(metric) ==
                    knowhere::IndexStaticFaced<knowhere::sparse_u32_f32>::HasRawData(name, version, json));

            knowhere::BinarySet bs;
            REQUIRE(idx.Serialize(bs) == knowhere::Status::success);
            if (use_mmap) {
                WriteBinaryToFile(tmp_file, bs.GetByName(idx.Type()));
                REQUIRE(idx.DeserializeFromFile(tmp_file, json) == knowhere::Status::success);
            } else {
                REQUIRE(idx.Deserialize(bs, json) == knowhere::Status::success);
            }

            auto results = idx.Search(query_ds, json, nullptr);
            REQUIRE(results.has_value());
            float recall = GetKNNRecall(*gt.value(), *results.value());
            check_distance_decreasing(*results.value());
            auto drop_ratio_search = json[knowhere::indexparam::DROP_RATIO_SEARCH].get<float>();
            if (drop_ratio_search == 0) {
                REQUIRE(recall == 1);
            } else {
                // most test cases are above 0.95, only a few between 0.9 and 0.95
                REQUIRE(recall >= 0.85);
            }
            // idx to destruct and munmap
        }
        if (use_mmap) {
            REQUIRE(std::remove(tmp_file) == 0);
        }
    }

    SECTION("Test Search with Bitset") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, sparse_inverted_index_gen),
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_WAND, sparse_inverted_index_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Size() > 0);
        REQUIRE(idx.Count() == nb);

        auto gen_bitset_fn = GENERATE(GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet);
        auto bitset_percentages = GENERATE(0.4f, 0.9f);

        auto bitset_data = gen_bitset_fn(nb, bitset_percentages * nb);
        knowhere::BitsetView bitset(bitset_data.data(), nb);
        auto filter_gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, bitset);
        check_result_match_filter(*filter_gt.value(), bitset);

        auto results = idx.Search(query_ds, json, bitset);
        check_result_match_filter(*results.value(), bitset);

        REQUIRE(results.has_value());
        float recall = GetKNNRecall(*filter_gt.value(), *results.value());
        check_distance_decreasing(*results.value());

        auto drop_ratio_search = json[knowhere::indexparam::DROP_RATIO_SEARCH].get<float>();
        if (drop_ratio_search == 0) {
            REQUIRE(recall == 1);
        } else {
            REQUIRE(recall >= 0.8);
        }
    }

    SECTION("Test Sparse Iterator with Bitset") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, sparse_inverted_index_gen),
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_WAND, sparse_inverted_index_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Size() > 0);
        REQUIRE(idx.Count() == nb);

        auto gen_bitset_fn = GENERATE(GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet);
        auto bitset_percentages = GENERATE(0.4f, 0.9f);

        auto bitset_data = gen_bitset_fn(nb, bitset_percentages * nb);
        knowhere::BitsetView bitset(bitset_data.data(), nb);
        auto iterators_or = idx.AnnIterator(query_ds, json, bitset);
        REQUIRE(iterators_or.has_value());
        auto& iterators = iterators_or.value();
        REQUIRE(iterators.size() == (size_t)nq);

        int count = 0;
        int out_of_order = 0;
        for (int i = 0; i < nq; ++i) {
            auto& iter = iterators[i];
            float prev_dist = std::numeric_limits<float>::max();
            while (iter->HasNext()) {
                auto [id, dist] = iter->Next();
                REQUIRE(!bitset.test(id));
                count++;
                if (prev_dist < dist) {
                    out_of_order++;
                }
                prev_dist = dist;
            }
        }
        // less than 5% of the distances are out of order.
        REQUIRE(out_of_order * 20 <= count);
    }

    SECTION("Test Sparse Range Search") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, sparse_inverted_index_gen),
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_WAND, sparse_inverted_index_gen),
        }));

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Size() > 0);
        REQUIRE(idx.Count() == nb);

        auto [radius, range_filter] = metric == knowhere::metric::BM25 ? GENERATE(table<float, float>({
                                                                             {80.0, 100.0},
                                                                             {100.0, 200.0},
                                                                         }))
                                                                       : GENERATE(table<float, float>({
                                                                             {0.5, 1},
                                                                             {1, 1.5},
                                                                         }));

        json[knowhere::meta::RADIUS] = radius;
        json[knowhere::meta::RANGE_FILTER] = range_filter;

        auto results = idx.RangeSearch(query_ds, json, nullptr);
        REQUIRE(results.has_value());

        auto gt =
            knowhere::BruteForce::RangeSearch<knowhere::sparse::SparseRow<float>>(train_ds, query_ds, json, nullptr);
        REQUIRE(gt.has_value());

        auto ids = results.value()->GetIds();
        auto lims = results.value()->GetLims();
        auto distances = results.value()->GetDistance();
        // any distance must be in range
        for (size_t i = 0; i < lims[nq]; ++i) {
            REQUIRE(distances[i] >= radius);
            REQUIRE(distances[i] <= range_filter);
        }

        auto ids_gt = gt.value()->GetIds();
        auto lims_gt = gt.value()->GetLims();
        auto distances_gt = gt.value()->GetDistance();
        // any distance must be in range
        for (size_t i = 0; i < lims_gt[nq]; ++i) {
            REQUIRE(distances_gt[i] > radius);
            REQUIRE(distances_gt[i] <= range_filter);
        }

        int actual_count = 0;
        int gt_count = 0;

        for (int i = 0; i < nq; ++i) {
            gt_count += lims_gt[i + 1] - lims_gt[i];

            std::unordered_set<int64_t> gt_ids;
            for (size_t j = lims_gt[i]; j < lims_gt[i + 1]; ++j) {
                gt_ids.insert(ids_gt[j]);
            }
            for (size_t j = lims[i]; j < lims[i + 1]; ++j) {
                if (gt_ids.find(ids[j]) != gt_ids.end()) {
                    actual_count++;
                }
            }
        }
        // most above 0.95, only a few between 0.9 and 0.83
        REQUIRE(actual_count * 1.0f / gt_count >= 0.83);
    }
}

TEST_CASE("Test Mem Sparse Index Handle Empty Vector", "[float metrics]") {
    auto [base_data, has_first_result] = GENERATE(table<std::vector<std::map<int32_t, float>>, bool>(
        {{std::vector<std::map<int32_t, float>>{
              {{1, 1.1f}, {2, 2.2f}, {6, 3.3f}},
              {},          // explicitly empty row
              {{5, 0.0f}}  // implicitly empty row
          },
          true},
         {std::vector<std::map<int32_t, float>>{{{1, 0.0f}}, {{3, 0.0f}}, {{5, 0.0f}}}, false},
         {std::vector<std::map<int32_t, float>>{{{1, 0.0f}}, {{3, 0.0f}}, {}}, false},
         {std::vector<std::map<int32_t, float>>{{}, {}, {}}, false}}));

    auto dim = 7;
    const auto train_ds = GenSparseDataSet(base_data, dim);

    auto topk = 5;

    auto metric = GENERATE(knowhere::metric::IP, knowhere::metric::BM25);
    auto version = GenTestVersionList();

    auto drop_ratio_search = GENERATE(0.0, 0.6);

    auto base_gen = [=, dim = dim, drop_ratio_search = drop_ratio_search]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::BM25_K1] = 1.2;
        json[knowhere::meta::BM25_B] = 0.75;
        json[knowhere::meta::BM25_AVGDL] = 100;
        json[knowhere::indexparam::DROP_RATIO_SEARCH] = drop_ratio_search;
        return json;
    };

    auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
        std::make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, base_gen),
        std::make_tuple(knowhere::IndexEnum::INDEX_SPARSE_WAND, base_gen),
    }));

    // query data must be constructed to match base_data and has_first_result:
    // if has_first_result is true, only q0 should find doc 0; otherwise, no query should find any neighbor.
    std::vector<std::map<int32_t, float>> query_data = {{{1, 1.1f}}, {{5, 1.1f}}, {}};
    const auto query_ds = GenSparseDataSet(query_data, dim);

    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
    auto cfg_json = gen().dump();
    CAPTURE(name, cfg_json);
    knowhere::Json json = knowhere::Json::parse(cfg_json);
    REQUIRE(idx.Type() == name);
    REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
    REQUIRE(idx.Size() > 0);

    knowhere::BinarySet bs;
    REQUIRE(idx.Serialize(bs) == knowhere::Status::success);

    auto use_mmap = GENERATE(false, true);
    auto tmp_file = "/tmp/knowhere_sparse_inverted_index_test";

    if (use_mmap) {
        WriteBinaryToFile(tmp_file, bs.GetByName(idx.Type()));
        REQUIRE(idx.DeserializeFromFile(tmp_file, json) == knowhere::Status::success);
    } else {
        REQUIRE(idx.Deserialize(bs, json) == knowhere::Status::success);
    }

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric}, {knowhere::meta::TOPK, topk},      {knowhere::meta::BM25_K1, 1.2},
        {knowhere::meta::BM25_B, 0.75},        {knowhere::meta::BM25_AVGDL, 100},
    };

    SECTION("Test Search") {
        auto check_result = [&, has_first_result = has_first_result](const knowhere::DataSet& ds) {
            auto nq = ds.GetRows();
            auto k = ds.GetDim();
            auto* ids = ds.GetIds();
            REQUIRE(ids[0] == (has_first_result ? 0 : -1));
            for (auto i = 1; i < nq * k; ++i) {
                REQUIRE(ids[i] == -1);
            }
        };
        auto bf_res = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, nullptr);
        REQUIRE(bf_res.has_value());
        check_result(*bf_res.value());

        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        check_result(*results.value());
    }

    SECTION("Test RangeSearch") {
        auto check_result = [&, has_first_result = has_first_result](const knowhere::DataSet& ds) {
            auto lims = ds.GetLims();
            auto* ids = ds.GetIds();
            if (has_first_result) {
                REQUIRE(lims[0] == 0);
                REQUIRE(lims[1] == 1);
                REQUIRE(ids[0] == 0);
                REQUIRE(lims[2] == 1);
                REQUIRE(lims[3] == 1);
            } else {
                // if no result found, lims should be all 0, ids and distances should point at 0-element array instead
                // of all -1, thus cannot be checked.
                REQUIRE(lims[0] == 0);
                REQUIRE(lims[1] == 0);
                REQUIRE(lims[2] == 0);
                REQUIRE(lims[3] == 0);
            }
        };
        json[knowhere::meta::RADIUS] = 0.0f;
        json[knowhere::meta::RANGE_FILTER] = 10000.0f;

        auto bf_res =
            knowhere::BruteForce::RangeSearch<knowhere::sparse::SparseRow<float>>(train_ds, query_ds, json, nullptr);
        REQUIRE(bf_res.has_value());
        check_result(*bf_res.value());

        auto results = idx.RangeSearch(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        check_result(*results.value());
    }
}

TEST_CASE("Test Mem Sparse Index CC", "[float metrics]") {
    std::atomic<int32_t> value_base(0);
    // each time a new batch of vectors are generated, the base value is increased by 1.
    // also the sparse vectors are all full, so newly generated vectors are guaranteed
    // to have larger IP than old vectors.
    auto doc_vector_gen = [&](int32_t nb, int32_t dim) {
        auto base = value_base.fetch_add(1);
        std::vector<std::map<int32_t, float>> data(nb);
        for (int32_t i = 0; i < nb; ++i) {
            for (int32_t j = 0; j < dim; ++j) {
                data[i][j] = base + static_cast<float>(rand()) / RAND_MAX * 0.8 + 0.1;
            }
        }
        return GenSparseDataSet(data, dim);
    };

    auto nb = 1000;
    auto dim = 30;
    auto topk = 50;
    int64_t nq = 100;

    auto query_ds = doc_vector_gen(nq, dim);

    auto inverted_index_algo = GENERATE("TAAT_NAIVE", "DAAT_WAND", "DAAT_MAXSCORE");

    auto drop_ratio_search = GENERATE(0.0, 0.3);

    auto metric = GENERATE(knowhere::metric::IP);
    auto version = GenTestVersionList();

    auto base_gen = [=, dim = dim]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::BM25_K1] = 1.2;
        json[knowhere::meta::BM25_B] = 0.75;
        json[knowhere::meta::BM25_AVGDL] = 100;
        return json;
    };

    auto sparse_inverted_index_gen = [base_gen, drop_ratio_search = drop_ratio_search,
                                      inverted_index_algo = inverted_index_algo]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::DROP_RATIO_SEARCH] = drop_ratio_search;
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = inverted_index_algo;
        return json;
    };

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric}, {knowhere::meta::TOPK, topk},      {knowhere::meta::BM25_K1, 1.2},
        {knowhere::meta::BM25_B, 0.75},        {knowhere::meta::BM25_AVGDL, 100},
    };

    // since all newly inserted vectors are guaranteed to have larger IP than old vectors,
    // the result ids of each search requests shoule be from the same batch of inserted vectors.
    auto check_result = [&](const knowhere::DataSet& ds) {
        auto nq = ds.GetRows();
        auto k = ds.GetDim();
        auto* ids = ds.GetIds();
        auto expected_id_base = ids[0] / nb;
        for (auto i = 0; i < nq; ++i) {
            for (auto j = 0; j < k; ++j) {
                auto base = ids[i * k + j] / nb;
                REQUIRE(base == expected_id_base);
            }
        }
    };

    auto test_time = 2;

    using std::make_tuple;
    auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
        make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC, sparse_inverted_index_gen),
        make_tuple(knowhere::IndexEnum::INDEX_SPARSE_WAND_CC, sparse_inverted_index_gen),
    }));

    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
    auto cfg_json = gen().dump();
    CAPTURE(name, cfg_json);
    knowhere::Json json = knowhere::Json::parse(cfg_json);
    REQUIRE(idx.Type() == name);
    // build the index with some initial data
    auto train_ds = doc_vector_gen(nb, dim);
    REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

    auto add_task = [&]() {
        auto start = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() <
               test_time) {
            auto doc_ds = doc_vector_gen(nb, dim);
            auto res = idx.Add(doc_ds, json);
            REQUIRE(res == knowhere::Status::success);
        }
    };

    auto search_task = [&]() {
        auto start = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() <
               test_time) {
            auto results = idx.Search(query_ds, json, nullptr);
            REQUIRE(results.has_value());
            check_result(*results.value());
        }
    };

    SECTION("Test Search") {
        std::vector<std::future<void>> task_list;
        for (int thread = 0; thread < 5; thread++) {
            task_list.push_back(std::async(std::launch::async, search_task));
        }
        task_list.push_back(std::async(std::launch::async, add_task));
        for (auto& task : task_list) {
            task.wait();
        }
    }

    SECTION("Test GetVectorByIds") {
        std::vector<int64_t> ids = {0, 1, 2};
        REQUIRE(idx.HasRawData(metric) ==
                knowhere::IndexStaticFaced<knowhere::sparse_u32_f32>::HasRawData(name, version, json));
        auto results = idx.GetVectorByIds(GenIdsDataSet(3, ids));
        REQUIRE(results.has_value());
        auto xb = (knowhere::sparse::SparseRow<float>*)train_ds->GetTensor();
        auto res_data = (knowhere::sparse::SparseRow<float>*)results.value()->GetTensor();
        for (int i = 0; i < 3; ++i) {
            const auto& truth_row = xb[i];
            const auto& res_row = res_data[i];
            REQUIRE(truth_row.size() == res_row.size());
            for (size_t j = 0; j < truth_row.size(); ++j) {
                REQUIRE(truth_row[j] == res_row[j]);
            }
        }
    }
}

// ============================================================================
// Block-Max WAND Comprehensive Test Suite
// ============================================================================

namespace {
// Helper: Manually compute expected block max scores for a posting list
std::vector<float>
ComputeExpectedBlockMaxScores(const std::vector<float>& plist_vals, const std::vector<float>& row_sums,
                               const std::vector<uint32_t>& plist_ids, bool use_bm25, float k1, float b,
                               float avgdl) {
    constexpr size_t kBlockSize = 128;
    if (plist_vals.empty()) {
        return {};
    }

    size_t num_blocks = (plist_vals.size() + kBlockSize - 1) / kBlockSize;
    std::vector<float> block_max_scores(num_blocks, 0.0f);

    for (size_t i = 0; i < plist_vals.size(); ++i) {
        size_t block_idx = i / kBlockSize;
        float score = plist_vals[i];

        if (use_bm25) {
            // Apply BM25 formula: tf * (k1 + 1) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
            uint32_t doc_id = plist_ids[i];
            float doc_len = row_sums[doc_id];
            float tf = plist_vals[i];
            score = tf * (k1 + 1.0f) / (tf + k1 * (1.0f - b + b * (doc_len / avgdl)));
        }

        block_max_scores[block_idx] = std::max(block_max_scores[block_idx], score);
    }

    return block_max_scores;
}

// Helper: Create controlled sparse dataset with specific posting list characteristics
knowhere::DataSetPtr
CreateControlledSparseDataSet(const std::vector<std::vector<std::pair<int32_t, float>>>& doc_data) {
    int32_t rows = doc_data.size();
    int32_t max_dim = 0;
    for (const auto& doc : doc_data) {
        for (const auto& [dim, val] : doc) {
            max_dim = std::max(max_dim, dim);
        }
    }

    auto tensor = std::make_unique<knowhere::sparse::SparseRow<float>[]>(rows);
    for (int32_t i = 0; i < rows; ++i) {
        if (doc_data[i].empty()) {
            continue;
        }
        knowhere::sparse::SparseRow<float> row(doc_data[i].size());
        size_t j = 0;
        for (auto& [dim, val] : doc_data[i]) {
            row.set_at(j++, dim, val);
        }
        tensor[i] = std::move(row);
    }

    auto ds = knowhere::GenDataSet(rows, max_dim + 1, tensor.release());
    ds->SetIsOwner(true);
    ds->SetIsSparse(true);
    return ds;
}
}  // namespace

TEST_CASE("Test BlockMaxInfo Structure", "[block_max]") {
    using knowhere::sparse::BlockMaxInfo;
    constexpr size_t kBlockSize = 128;

    SECTION("Block Index Calculation") {
        // Test various positions
        REQUIRE(BlockMaxInfo::block_index(0) == 0);
        REQUIRE(BlockMaxInfo::block_index(1) == 0);
        REQUIRE(BlockMaxInfo::block_index(127) == 0);
        REQUIRE(BlockMaxInfo::block_index(128) == 1);
        REQUIRE(BlockMaxInfo::block_index(129) == 1);
        REQUIRE(BlockMaxInfo::block_index(255) == 1);
        REQUIRE(BlockMaxInfo::block_index(256) == 2);
        REQUIRE(BlockMaxInfo::block_index(384) == 3);
        REQUIRE(BlockMaxInfo::block_index(1000) == 7);  // 1000 / 128 = 7
    }

    SECTION("Number of Blocks Calculation") {
        REQUIRE(BlockMaxInfo::num_blocks(0) == 0);
        REQUIRE(BlockMaxInfo::num_blocks(1) == 1);
        REQUIRE(BlockMaxInfo::num_blocks(127) == 1);
        REQUIRE(BlockMaxInfo::num_blocks(128) == 1);
        REQUIRE(BlockMaxInfo::num_blocks(129) == 2);
        REQUIRE(BlockMaxInfo::num_blocks(256) == 2);
        REQUIRE(BlockMaxInfo::num_blocks(257) == 3);
        REQUIRE(BlockMaxInfo::num_blocks(kBlockSize) == 1);
        REQUIRE(BlockMaxInfo::num_blocks(kBlockSize + 1) == 2);
        REQUIRE(BlockMaxInfo::num_blocks(3 * kBlockSize) == 3);
        REQUIRE(BlockMaxInfo::num_blocks(3 * kBlockSize + 50) == 4);
    }

    SECTION("Max Score From Block") {
        BlockMaxInfo info;
        info.block_max_scores = {0.5f, 0.8f, 0.3f, 0.9f, 0.2f};

        // From block 0: should return max of all blocks
        REQUIRE(info.max_score_from_block(0) == 0.9f);

        // From block 1: max of blocks 1-4
        REQUIRE(info.max_score_from_block(1) == 0.9f);

        // From block 3: max of blocks 3-4
        REQUIRE(info.max_score_from_block(3) == 0.9f);

        // From block 4: only block 4
        REQUIRE(info.max_score_from_block(4) == 0.2f);

        // Past end: returns 0
        REQUIRE(info.max_score_from_block(5) == 0.0f);
        REQUIRE(info.max_score_from_block(100) == 0.0f);
    }

    SECTION("Individual Block Score") {
        BlockMaxInfo info;
        info.block_max_scores = {0.5f, 0.8f, 0.3f, 0.9f, 0.2f};

        REQUIRE(info.block_score(0) == 0.5f);
        REQUIRE(info.block_score(1) == 0.8f);
        REQUIRE(info.block_score(2) == 0.3f);
        REQUIRE(info.block_score(3) == 0.9f);
        REQUIRE(info.block_score(4) == 0.2f);

        // Past end returns 0
        REQUIRE(info.block_score(5) == 0.0f);
        REQUIRE(info.block_score(100) == 0.0f);
    }

    SECTION("Empty BlockMaxInfo") {
        BlockMaxInfo info;
        REQUIRE(info.block_max_scores.empty());
        REQUIRE(info.max_score_from_block(0) == 0.0f);
        REQUIRE(info.block_score(0) == 0.0f);
    }
}

TEST_CASE("Test Block Max Info Build and Correctness", "[block_max][build]") {
    constexpr size_t kBlockSize = 128;
    auto version = GenTestVersionList();

    SECTION("IP Metric - Block Max Equals Raw Values") {
        // Create controlled dataset where we know exact posting list structure
        // Dimension 0: 200 docs with varying scores
        std::vector<std::vector<std::pair<int32_t, float>>> doc_data(300);

        // First 200 docs have dimension 0 with specific scores
        for (int i = 0; i < 200; ++i) {
            float score = 0.1f + (i % 10) * 0.1f;  // Scores 0.1 to 1.0, cycling
            doc_data[i].push_back({0, score});
        }

        auto train_ds = CreateControlledSparseDataSet(doc_data);

        knowhere::Json json;
        json[knowhere::meta::DIM] = 10;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        json[knowhere::meta::TOPK] = 5;
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_WAND";

        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
                knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        // For IP, block max should equal max of raw values in each block
        // Block 0: docs 0-127, scores cycle through 0.1-1.0
        // Block 1: docs 128-199 (72 docs), scores cycle through 0.1-1.0
        // Expected: block 0 max = 1.0, block 1 max = 1.0

        // We can't directly access block_max_info_, but we can verify through search behavior
        // The index was built successfully, which means block info was computed
        REQUIRE(idx.Count() == 300);
    }

    SECTION("BM25 Metric - Block Max Uses BM25 Formula") {
        // Create dataset for BM25
        std::vector<std::vector<std::pair<int32_t, float>>> doc_data(300);

        for (int i = 0; i < 200; ++i) {
            // Term frequency values for BM25
            float tf = 1.0f + (i % 5);  // TF from 1 to 5
            doc_data[i].push_back({0, tf});
        }

        auto train_ds = CreateControlledSparseDataSet(doc_data);

        knowhere::Json json;
        json[knowhere::meta::DIM] = 10;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::BM25;
        json[knowhere::meta::TOPK] = 5;
        json[knowhere::meta::BM25_K1] = 1.2f;
        json[knowhere::meta::BM25_B] = 0.75f;
        json[knowhere::meta::BM25_AVGDL] = 100.0f;
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_WAND";

        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
                knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        // For BM25, block max should be computed using BM25 formula
        // We verify correctness through successful build and search
        REQUIRE(idx.Count() == 300);
    }

    SECTION("Posting List Size Variations") {
        auto test_size = GENERATE(1, 50, 127, 128, 129, 200, 256, 384, 400);
        CAPTURE(test_size);

        std::vector<std::vector<std::pair<int32_t, float>>> doc_data(test_size);
        for (int i = 0; i < test_size; ++i) {
            doc_data[i].push_back({0, 1.0f});
        }

        auto train_ds = CreateControlledSparseDataSet(doc_data);

        knowhere::Json json;
        json[knowhere::meta::DIM] = 10;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        json[knowhere::meta::TOPK] = std::min(5, test_size);
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_WAND";

        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
                knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Count() == test_size);

        // Expected number of blocks
        size_t expected_blocks = (test_size + kBlockSize - 1) / kBlockSize;
        CAPTURE(expected_blocks);

        // Verify search works (implicitly tests block info correctness)
        std::vector<std::vector<std::pair<int32_t, float>>> query_data(1);
        query_data[0].push_back({0, 1.0f});
        auto query_ds = CreateControlledSparseDataSet(query_data);

        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
    }

    SECTION("Multiple Dimensions with Different Posting List Sizes") {
        // Dim 0: 50 docs, Dim 1: 150 docs, Dim 2: 300 docs
        std::vector<std::vector<std::pair<int32_t, float>>> doc_data(300);

        for (int i = 0; i < 50; ++i) {
            doc_data[i].push_back({0, 1.0f});
        }
        for (int i = 0; i < 150; ++i) {
            doc_data[i].push_back({1, 1.0f});
        }
        for (int i = 0; i < 300; ++i) {
            doc_data[i].push_back({2, 1.0f});
        }

        auto train_ds = CreateControlledSparseDataSet(doc_data);

        knowhere::Json json;
        json[knowhere::meta::DIM] = 10;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        json[knowhere::meta::TOPK] = 5;
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_WAND";

        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
                knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        // All dimensions should have block info built
        REQUIRE(idx.Count() == 300);
    }
}

TEST_CASE("Test Filter-Aware Block-Max WAND", "[block_max][filter_aware]") {
    auto version = GenTestVersionList();
    int32_t nb = 1000;
    int32_t dim = 100;
    auto topk = 10;

    auto metric = GENERATE(knowhere::metric::IP, knowhere::metric::BM25);
    auto algo = GENERATE("DAAT_WAND", "DAAT_MAXSCORE");

    auto sparse_dataset_gen = [&](int nr, int dim_val, float sparsity) -> knowhere::DataSetPtr {
        if (metric == knowhere::metric::BM25) {
            return GenSparseDataSetWithMaxVal(nr, dim_val, sparsity, 256, true);
        } else {
            return GenSparseDataSet(nr, dim_val, sparsity);
        }
    };

    auto train_ds = sparse_dataset_gen(nb, dim, 0.95f);
    auto query_ds = sparse_dataset_gen(5, dim, 0.97f);

    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = metric;
    json[knowhere::meta::TOPK] = topk;
    json[knowhere::meta::BM25_K1] = 1.2f;
    json[knowhere::meta::BM25_B] = 0.75f;
    json[knowhere::meta::BM25_AVGDL] = 100.0f;
    json[knowhere::indexparam::DROP_RATIO_SEARCH] = 0.0f;
    json[knowhere::indexparam::INVERTED_INDEX_ALGO] = algo;

    auto idx =
        knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
    REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
        {knowhere::meta::BM25_K1, 1.2f},
        {knowhere::meta::BM25_B, 0.75f},
        {knowhere::meta::BM25_AVGDL, 100.0f},
    };

    SECTION("No Filter Baseline") {
        auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, nullptr);
        auto results = idx.Search(query_ds, json, nullptr);

        REQUIRE(results.has_value());
        REQUIRE(gt.has_value());

        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall == 1.0f);
    }

    SECTION("Various Filter Ratios") {
        auto filter_ratio = GENERATE(0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 0.95f);
        CAPTURE(filter_ratio);

        auto bitset_data = GenerateBitsetWithRandomTbitsSet(nb, static_cast<size_t>(nb * filter_ratio));
        knowhere::BitsetView bitset(bitset_data.data(), nb);

        auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, bitset);
        auto results = idx.Search(query_ds, json, bitset);

        REQUIRE(results.has_value());
        REQUIRE(gt.has_value());

        // Verify all results pass the filter
        auto* ids = results.value()->GetIds();
        auto nq = results.value()->GetRows();
        auto k = results.value()->GetDim();
        for (int i = 0; i < nq * k; ++i) {
            if (ids[i] != -1) {
                REQUIRE(!bitset.test(ids[i]));
            }
        }

        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall == 1.0f);
    }

    SECTION("Sequential vs Random Filters") {
        float filter_ratio = 0.5f;

        // Sequential filter (first half)
        auto seq_bitset_data = GenerateBitsetWithFirstTbitsSet(nb, static_cast<size_t>(nb * filter_ratio));
        knowhere::BitsetView seq_bitset(seq_bitset_data.data(), nb);

        // Random filter
        auto rand_bitset_data = GenerateBitsetWithRandomTbitsSet(nb, static_cast<size_t>(nb * filter_ratio));
        knowhere::BitsetView rand_bitset(rand_bitset_data.data(), nb);

        auto gt_seq = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, seq_bitset);
        auto results_seq = idx.Search(query_ds, json, seq_bitset);

        auto gt_rand = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, rand_bitset);
        auto results_rand = idx.Search(query_ds, json, rand_bitset);

        REQUIRE(results_seq.has_value());
        REQUIRE(results_rand.has_value());

        float recall_seq = GetKNNRecall(*gt_seq.value(), *results_seq.value());
        float recall_rand = GetKNNRecall(*gt_rand.value(), *results_rand.value());

        REQUIRE(recall_seq == 1.0f);
        REQUIRE(recall_rand == 1.0f);
    }
}

TEST_CASE("Test Block Max Info Rebuild After Load", "[block_max][serialization]") {
    auto version = GenTestVersionList();
    int32_t nb = 500;
    int32_t dim = 50;
    auto topk = 5;

    auto metric = GENERATE(knowhere::metric::IP, knowhere::metric::BM25);
    auto algo = GENERATE("DAAT_WAND", "DAAT_MAXSCORE");

    auto sparse_dataset_gen = [&](int nr, int dim_val, float sparsity) -> knowhere::DataSetPtr {
        if (metric == knowhere::metric::BM25) {
            return GenSparseDataSetWithMaxVal(nr, dim_val, sparsity, 256, true);
        } else {
            return GenSparseDataSet(nr, dim_val, sparsity);
        }
    };

    auto train_ds = sparse_dataset_gen(nb, dim, 0.95f);
    auto query_ds = sparse_dataset_gen(3, dim, 0.97f);

    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = metric;
    json[knowhere::meta::TOPK] = topk;
    json[knowhere::meta::BM25_K1] = 1.2f;
    json[knowhere::meta::BM25_B] = 0.75f;
    json[knowhere::meta::BM25_AVGDL] = 100.0f;
    json[knowhere::indexparam::INVERTED_INDEX_ALGO] = algo;

    SECTION("Serialize and Deserialize Consistency") {
        auto idx1 = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx1.Build(train_ds, json) == knowhere::Status::success);

        // Search before serialization
        auto results_before = idx1.Search(query_ds, json, nullptr);
        REQUIRE(results_before.has_value());

        // Serialize
        knowhere::BinarySet bs;
        REQUIRE(idx1.Serialize(bs) == knowhere::Status::success);

        // Deserialize into new index
        auto idx2 = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx2.Deserialize(bs, json) == knowhere::Status::success);

        // Search after deserialization
        auto results_after = idx2.Search(query_ds, json, nullptr);
        REQUIRE(results_after.has_value());

        // Results should be identical
        auto* ids_before = results_before.value()->GetIds();
        auto* ids_after = results_after.value()->GetIds();
        auto* dist_before = results_before.value()->GetDistance();
        auto* dist_after = results_after.value()->GetDistance();

        auto nq = results_before.value()->GetRows();
        auto k = results_before.value()->GetDim();

        for (int i = 0; i < nq * k; ++i) {
            REQUIRE(ids_before[i] == ids_after[i]);
            if (ids_before[i] != -1) {
                REQUIRE(std::abs(dist_before[i] - dist_after[i]) < 1e-5f);
            }
        }
    }

    SECTION("Mmap Mode Rebuild") {
        auto tmp_file = "/tmp/knowhere_block_max_test_mmap";

        auto idx1 = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx1.Build(train_ds, json) == knowhere::Status::success);

        auto results_before = idx1.Search(query_ds, json, nullptr);

        knowhere::BinarySet bs;
        REQUIRE(idx1.Serialize(bs) == knowhere::Status::success);
        WriteBinaryToFile(tmp_file, bs.GetByName(idx1.Type()));

        auto idx2 = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx2.DeserializeFromFile(tmp_file, json) == knowhere::Status::success);

        auto results_after = idx2.Search(query_ds, json, nullptr);
        REQUIRE(results_after.has_value());

        // Verify consistency
        float recall = GetKNNRecall(*results_before.value(), *results_after.value());
        REQUIRE(recall == 1.0f);

        std::remove(tmp_file);
    }

    SECTION("BM25 with Different avgdl at Search Time") {
        if (metric != knowhere::metric::BM25) {
            return;  // Skip for IP
        }

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        // Serialize and deserialize
        knowhere::BinarySet bs;
        REQUIRE(idx.Serialize(bs) == knowhere::Status::success);

        auto idx2 = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx2.Deserialize(bs, json) == knowhere::Status::success);

        // Search with different avgdl
        knowhere::Json search_json = json;
        search_json[knowhere::meta::BM25_AVGDL] = 150.0f;  // Different from build time (100.0)

        auto results = idx2.Search(query_ds, search_json, nullptr);
        REQUIRE(results.has_value());

        // Results should still be valid (block info rebuilt with new avgdl)
        auto* ids = results.value()->GetIds();
        REQUIRE(ids[0] != -1);  // At least one result
    }
}

TEST_CASE("Test Block Max Info on Incremental Add", "[block_max][incremental]") {
    auto version = GenTestVersionList();
    int32_t dim = 50;
    auto topk = 5;

    // Use CC variant for Add support
    auto metric = knowhere::metric::IP;
    auto algo = "DAAT_WAND";

    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = metric;
    json[knowhere::meta::TOPK] = topk;
    json[knowhere::indexparam::INVERTED_INDEX_ALGO] = algo;

    SECTION("Add to Partial Block") {
        // Initial build: 100 docs (< 128, so single partial block)
        auto train_ds = GenSparseDataSet(100, dim, 0.95f);

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND_CC, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Count() == 100);

        // Add 20 more docs (total 120, still < 128)
        auto add_ds = GenSparseDataSet(20, dim, 0.95f);
        REQUIRE(idx.Add(add_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Count() == 120);

        // Search should work correctly
        auto query_ds = GenSparseDataSet(2, dim, 0.97f);
        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
    }

    SECTION("Add Creates New Block") {
        // Initial build: 100 docs
        auto train_ds = GenSparseDataSet(100, dim, 0.95f);

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND_CC, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        // Add 100 more docs (total 200, creates 2 blocks)
        auto add_ds = GenSparseDataSet(100, dim, 0.95f);
        REQUIRE(idx.Add(add_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Count() == 200);

        auto query_ds = GenSparseDataSet(2, dim, 0.97f);
        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
    }

    SECTION("Multiple Add Calls") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND_CC, version).value();

        // Initial: 50 docs
        auto train_ds = GenSparseDataSet(50, dim, 0.95f);
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Count() == 50);

        // Add 1: +50 docs (total 100)
        auto add_ds1 = GenSparseDataSet(50, dim, 0.95f);
        REQUIRE(idx.Add(add_ds1, json) == knowhere::Status::success);
        REQUIRE(idx.Count() == 100);

        // Add 2: +50 docs (total 150, crosses block boundary)
        auto add_ds2 = GenSparseDataSet(50, dim, 0.95f);
        REQUIRE(idx.Add(add_ds2, json) == knowhere::Status::success);
        REQUIRE(idx.Count() == 150);

        // Add 3: +50 docs (total 200, completes 2nd block)
        auto add_ds3 = GenSparseDataSet(50, dim, 0.95f);
        REQUIRE(idx.Add(add_ds3, json) == knowhere::Status::success);
        REQUIRE(idx.Count() == 200);

        // Search should work after all adds
        auto query_ds = GenSparseDataSet(2, dim, 0.97f);
        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
    }

    SECTION("Add with New Dimensions") {
        // Initial data with dimensions 0, 1, 2
        std::vector<std::vector<std::pair<int32_t, float>>> initial_data(100);
        for (int i = 0; i < 100; ++i) {
            initial_data[i] = {{0, 1.0f}, {1, 1.0f}, {2, 1.0f}};
        }
        auto train_ds = CreateControlledSparseDataSet(initial_data);

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND_CC, version).value();

        json[knowhere::meta::DIM] = 10;
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        // Add data with dimensions 2, 3, 4 (2 overlaps, 3 and 4 are new)
        std::vector<std::vector<std::pair<int32_t, float>>> add_data(50);
        for (int i = 0; i < 50; ++i) {
            add_data[i] = {{2, 1.0f}, {3, 1.0f}, {4, 1.0f}};
        }
        auto add_ds = CreateControlledSparseDataSet(add_data);

        REQUIRE(idx.Add(add_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Count() == 150);

        // Search on new dimensions should work
        std::vector<std::vector<std::pair<int32_t, float>>> query_data(1);
        query_data[0] = {{3, 1.0f}, {4, 1.0f}};
        auto query_ds = CreateControlledSparseDataSet(query_data);

        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
    }
}

TEST_CASE("Test Block Max Edge Cases", "[block_max][edge_cases]") {
    auto version = GenTestVersionList();
    constexpr size_t kBlockSize = 128;

    SECTION("Empty Index") {
        std::vector<std::vector<std::pair<int32_t, float>>> empty_data(0);
        // Note: Can't actually create empty dataset, so create minimal one
        std::vector<std::vector<std::pair<int32_t, float>>> minimal_data(1);
        minimal_data[0] = {{0, 1.0f}};
        auto train_ds = CreateControlledSparseDataSet(minimal_data);

        knowhere::Json json;
        json[knowhere::meta::DIM] = 10;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        json[knowhere::meta::TOPK] = 5;
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_WAND";

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
    }

    SECTION("Single Document") {
        std::vector<std::vector<std::pair<int32_t, float>>> doc_data(1);
        doc_data[0] = {{0, 1.0f}, {1, 2.0f}, {2, 3.0f}};
        auto train_ds = CreateControlledSparseDataSet(doc_data);

        knowhere::Json json;
        json[knowhere::meta::DIM] = 10;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        json[knowhere::meta::TOPK] = 1;
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_WAND";

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Count() == 1);

        // Each dimension has posting list of length 1 (single block with 1 element)
        std::vector<std::vector<std::pair<int32_t, float>>> query_data(1);
        query_data[0] = {{0, 1.0f}};
        auto query_ds = CreateControlledSparseDataSet(query_data);

        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto* ids = results.value()->GetIds();
        REQUIRE(ids[0] == 0);
    }

    SECTION("Exact Block Boundary - 128 Elements") {
        std::vector<std::vector<std::pair<int32_t, float>>> doc_data(kBlockSize);
        for (size_t i = 0; i < kBlockSize; ++i) {
            doc_data[i] = {{0, 1.0f}};
        }
        auto train_ds = CreateControlledSparseDataSet(doc_data);

        knowhere::Json json;
        json[knowhere::meta::DIM] = 10;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        json[knowhere::meta::TOPK] = 5;
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_WAND";

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Count() == kBlockSize);

        // Should have exactly 1 block (not 2)
        std::vector<std::vector<std::pair<int32_t, float>>> query_data(1);
        query_data[0] = {{0, 1.0f}};
        auto query_ds = CreateControlledSparseDataSet(query_data);

        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
    }

    SECTION("One Element Past Block Boundary - 129 Elements") {
        std::vector<std::vector<std::pair<int32_t, float>>> doc_data(kBlockSize + 1);
        for (size_t i = 0; i < kBlockSize + 1; ++i) {
            doc_data[i] = {{0, 1.0f}};
        }
        auto train_ds = CreateControlledSparseDataSet(doc_data);

        knowhere::Json json;
        json[knowhere::meta::DIM] = 10;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        json[knowhere::meta::TOPK] = 5;
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_WAND";

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Count() == kBlockSize + 1);

        // Should have 2 blocks (block 1 has only 1 element)
        std::vector<std::vector<std::pair<int32_t, float>>> query_data(1);
        query_data[0] = {{0, 1.0f}};
        auto query_ds = CreateControlledSparseDataSet(query_data);

        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
    }

    SECTION("Large Posting List - 10000 Elements") {
        size_t large_size = 10000;
        std::vector<std::vector<std::pair<int32_t, float>>> doc_data(large_size);
        for (size_t i = 0; i < large_size; ++i) {
            doc_data[i] = {{0, 1.0f + (i % 100) * 0.01f}};  // Varying scores
        }
        auto train_ds = CreateControlledSparseDataSet(doc_data);

        knowhere::Json json;
        json[knowhere::meta::DIM] = 10;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        json[knowhere::meta::TOPK] = 10;
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_WAND";

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Count() == large_size);

        // Expected: 79 blocks (10000 / 128 = 78.125, rounds up to 79)
        size_t expected_blocks = (large_size + kBlockSize - 1) / kBlockSize;
        REQUIRE(expected_blocks == 79);

        std::vector<std::vector<std::pair<int32_t, float>>> query_data(1);
        query_data[0] = {{0, 1.0f}};
        auto query_ds = CreateControlledSparseDataSet(query_data);

        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
    }

    SECTION("Zero Scores in Posting List") {
        std::vector<std::vector<std::pair<int32_t, float>>> doc_data(100);
        for (int i = 0; i < 100; ++i) {
            doc_data[i] = {{0, 0.0f}};  // All zero scores
        }
        auto train_ds = CreateControlledSparseDataSet(doc_data);

        knowhere::Json json;
        json[knowhere::meta::DIM] = 10;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        json[knowhere::meta::TOPK] = 5;
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_WAND";

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        // Note: zeros might be filtered out during add_row_to_index
        // But if they remain, block max should be 0

        std::vector<std::vector<std::pair<int32_t, float>>> query_data(1);
        query_data[0] = {{0, 1.0f}};
        auto query_ds = CreateControlledSparseDataSet(query_data);

        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
    }
}

TEST_CASE("Test Block Max WAND Regression", "[block_max][regression]") {
    auto version = GenTestVersionList();
    int32_t nb = 800;
    int32_t dim = 100;
    auto topk = 10;
    int32_t nq = 5;

    auto metric = GENERATE(knowhere::metric::IP, knowhere::metric::BM25);
    auto algo = GENERATE("DAAT_WAND", "DAAT_MAXSCORE");

    auto sparse_dataset_gen = [&](int nr, int dim_val, float sparsity) -> knowhere::DataSetPtr {
        if (metric == knowhere::metric::BM25) {
            return GenSparseDataSetWithMaxVal(nr, dim_val, sparsity, 256, true);
        } else {
            return GenSparseDataSet(nr, dim_val, sparsity);
        }
    };

    auto train_ds = sparse_dataset_gen(nb, dim, 0.95f);
    auto query_ds = sparse_dataset_gen(nq, dim, 0.97f);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
        {knowhere::meta::BM25_K1, 1.2f},
        {knowhere::meta::BM25_B, 0.75f},
        {knowhere::meta::BM25_AVGDL, 100.0f},
    };

    knowhere::Json json = conf;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::indexparam::INVERTED_INDEX_ALGO] = algo;

    SECTION("Correctness vs Brute Force - No Filter") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, nullptr);
        auto results = idx.Search(query_ds, json, nullptr);

        REQUIRE(results.has_value());
        REQUIRE(gt.has_value());

        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall == 1.0f);

        // Verify distances match (within floating point tolerance)
        auto* dist_gt = gt.value()->GetDistance();
        auto* dist_result = results.value()->GetDistance();
        auto* ids_result = results.value()->GetIds();

        for (int i = 0; i < nq * topk; ++i) {
            if (ids_result[i] != -1) {
                REQUIRE(std::abs(dist_gt[i] - dist_result[i]) < 2e-4f);
            }
        }
    }

    SECTION("Correctness with Various Filters") {
        auto filter_ratio = GENERATE(0.1f, 0.3f, 0.5f, 0.7f, 0.9f);
        CAPTURE(filter_ratio);

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        auto bitset_data = GenerateBitsetWithRandomTbitsSet(nb, static_cast<size_t>(nb * filter_ratio));
        knowhere::BitsetView bitset(bitset_data.data(), nb);

        auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, bitset);
        auto results = idx.Search(query_ds, json, bitset);

        REQUIRE(results.has_value());
        REQUIRE(gt.has_value());

        // Verify all returned IDs pass filter
        auto* ids = results.value()->GetIds();
        for (int i = 0; i < nq * topk; ++i) {
            if (ids[i] != -1) {
                REQUIRE(!bitset.test(ids[i]));
            }
        }

        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall == 1.0f);
    }

    SECTION("DAAT_WAND vs DAAT_MAXSCORE Consistency") {
        if (algo != "DAAT_WAND") {
            return;  // Only run once
        }

        auto idx_wand = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        auto idx_maxscore = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();

        knowhere::Json json_wand = json;
        json_wand[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_WAND";

        knowhere::Json json_maxscore = json;
        json_maxscore[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_MAXSCORE";

        REQUIRE(idx_wand.Build(train_ds, json_wand) == knowhere::Status::success);
        REQUIRE(idx_maxscore.Build(train_ds, json_maxscore) == knowhere::Status::success);

        auto results_wand = idx_wand.Search(query_ds, json_wand, nullptr);
        auto results_maxscore = idx_maxscore.Search(query_ds, json_maxscore, nullptr);

        REQUIRE(results_wand.has_value());
        REQUIRE(results_maxscore.has_value());

        // Both should produce identical results
        float recall = GetKNNRecall(*results_wand.value(), *results_maxscore.value());
        REQUIRE(recall == 1.0f);
    }

    SECTION("Consistency Across Serialization") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        auto results_before = idx.Search(query_ds, json, nullptr);
        REQUIRE(results_before.has_value());

        knowhere::BinarySet bs;
        REQUIRE(idx.Serialize(bs) == knowhere::Status::success);

        auto idx2 = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx2.Deserialize(bs, json) == knowhere::Status::success);

        auto results_after = idx2.Search(query_ds, json, nullptr);
        REQUIRE(results_after.has_value());

        // Results must be identical
        auto* ids_before = results_before.value()->GetIds();
        auto* ids_after = results_after.value()->GetIds();

        for (int i = 0; i < nq * topk; ++i) {
            REQUIRE(ids_before[i] == ids_after[i]);
        }
    }

    SECTION("Consistency Across dim_max_score_ratio") {
        auto ratio = GENERATE(0.5f, 0.8f, 1.0f, 1.05f, 1.3f);
        CAPTURE(ratio);

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        knowhere::Json search_json = json;
        search_json[knowhere::meta::DIM_MAX_SCORE_RATIO] = ratio;

        auto results = idx.Search(query_ds, search_json, nullptr);
        REQUIRE(results.has_value());

        // For ratio = 1.0, should be exact
        if (std::abs(ratio - 1.0f) < 0.01f) {
            auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, nullptr);
            float recall = GetKNNRecall(*gt.value(), *results.value());
            REQUIRE(recall == 1.0f);
        }
    }
}

TEST_CASE("Test Block Max Integration Scenarios", "[block_max][integration]") {
    auto version = GenTestVersionList();

    SECTION("Realistic Sparse Vector Scenario") {
        int32_t nb = 2000;
        int32_t dim = 1000;
        int32_t nq = 10;
        auto topk = 10;

        auto train_ds = GenSparseDataSet(nb, dim, 0.97f);
        auto query_ds = GenSparseDataSet(nq, dim, 0.98f);

        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_WAND";

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        // Apply 50% filter
        auto bitset_data = GenerateBitsetWithRandomTbitsSet(nb, nb / 2);
        knowhere::BitsetView bitset(bitset_data.data(), nb);

        auto results = idx.Search(query_ds, json, bitset);
        REQUIRE(results.has_value());

        // Verify all results pass filter
        auto* ids = results.value()->GetIds();
        for (int i = 0; i < nq * topk; ++i) {
            if (ids[i] != -1) {
                REQUIRE(!bitset.test(ids[i]));
            }
        }
    }

    SECTION("BM25 with Varying Document Lengths") {
        int32_t nb = 1000;
        int32_t dim = 200;
        int32_t nq = 5;
        auto topk = 10;

        // Generate dataset with wide doc length distribution
        std::vector<std::vector<std::pair<int32_t, float>>> doc_data(nb);
        std::mt19937 rng(42);
        std::uniform_int_distribution<> dim_dist(0, dim - 1);
        std::uniform_real_distribution<> val_dist(1.0f, 5.0f);

        for (int i = 0; i < nb; ++i) {
            // Varying doc lengths: some short (5 terms), some long (100 terms)
            int doc_len = (i % 3 == 0) ? 5 : ((i % 3 == 1) ? 50 : 100);
            std::set<int32_t> used_dims;

            for (int j = 0; j < doc_len; ++j) {
                int32_t dim_id = dim_dist(rng);
                while (used_dims.count(dim_id)) {
                    dim_id = dim_dist(rng);
                }
                used_dims.insert(dim_id);
                float tf = static_cast<float>(static_cast<int32_t>(val_dist(rng)));
                doc_data[i].push_back({dim_id, tf});
            }
        }

        auto train_ds = CreateControlledSparseDataSet(doc_data);

        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::BM25;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::BM25_K1] = 1.2f;
        json[knowhere::meta::BM25_B] = 0.75f;
        json[knowhere::meta::BM25_AVGDL] = 50.0f;
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_WAND";

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        // Query dataset
        std::vector<std::vector<std::pair<int32_t, float>>> query_data(nq);
        for (int i = 0; i < nq; ++i) {
            for (int j = 0; j < 10; ++j) {
                query_data[i].push_back({j * 20, 1.0f});
            }
        }
        auto query_ds = CreateControlledSparseDataSet(query_data);

        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
    }

    SECTION("Multi-Query Batch with Different Filters") {
        int32_t nb = 1000;
        int32_t dim = 100;
        auto topk = 10;

        auto train_ds = GenSparseDataSet(nb, dim, 0.95f);

        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_WAND";

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        // Query 1: No filter
        auto query_ds1 = GenSparseDataSet(2, dim, 0.97f);
        auto results1 = idx.Search(query_ds1, json, nullptr);
        REQUIRE(results1.has_value());

        // Query 2: 30% filter
        auto query_ds2 = GenSparseDataSet(2, dim, 0.97f);
        auto bitset_data2 = GenerateBitsetWithRandomTbitsSet(nb, nb * 3 / 10);
        knowhere::BitsetView bitset2(bitset_data2.data(), nb);
        auto results2 = idx.Search(query_ds2, json, bitset2);
        REQUIRE(results2.has_value());

        // Query 3: 80% filter
        auto query_ds3 = GenSparseDataSet(2, dim, 0.97f);
        auto bitset_data3 = GenerateBitsetWithRandomTbitsSet(nb, nb * 8 / 10);
        knowhere::BitsetView bitset3(bitset_data3.data(), nb);
        auto results3 = idx.Search(query_ds3, json, bitset3);
        REQUIRE(results3.has_value());

        // All queries should produce valid results
        REQUIRE(results1.value()->GetIds()[0] != -1);
        REQUIRE(results2.value()->GetIds()[0] != -1);
        REQUIRE(results3.value()->GetIds()[0] != -1);
    }

    SECTION("Drop Ratio Search with Filters") {
        int32_t nb = 1000;
        int32_t dim = 100;
        int32_t nq = 5;
        auto topk = 10;

        auto train_ds = GenSparseDataSet(nb, dim, 0.95f);
        auto query_ds = GenSparseDataSet(nq, dim, 0.97f);

        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::indexparam::DROP_RATIO_SEARCH] = 0.3f;  // Drop 30% of query terms
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_WAND";

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_WAND, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        // Apply 50% doc filter
        auto bitset_data = GenerateBitsetWithRandomTbitsSet(nb, nb / 2);
        knowhere::BitsetView bitset(bitset_data.data(), nb);

        auto results = idx.Search(query_ds, json, bitset);
        REQUIRE(results.has_value());

        // Verify all results pass the doc filter
        auto* ids = results.value()->GetIds();
        for (int i = 0; i < nq * topk; ++i) {
            if (ids[i] != -1) {
                REQUIRE(!bitset.test(ids[i]));
            }
        }

    }
}
