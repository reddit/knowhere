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

#include <chrono>
#include <thread>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "knowhere/comp/task.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/expected.h"
#include "knowhere/heap.h"
#include "knowhere/utils.h"
#include "knowhere/version.h"
#include "roaring/roaring.h"
#include "utils.h"

namespace {
const std::vector<size_t> kBitsetSizes{4, 8, 10, 64, 100, 500, 1024};
}

template <typename T>
void
CheckNormalizeDataset(int rows, int dim, float diff) {
    auto ds = GenDataSet(rows, dim);
    auto type_ds = knowhere::ConvertToDataTypeIfNeeded<T>(ds);
    auto data = (T*)type_ds->GetTensor();
    knowhere::NormalizeDataset<T>(type_ds);
    for (int i = 0; i < rows; ++i) {
        float sum = 0.0;
        for (int j = 0; j < dim; ++j) {
            auto val = data[i * dim + j];
            sum += val * val;
        }
        CHECK(std::abs(1.0f - sum) <= diff);
    }
}

template <typename T>
void
CheckCopyAndNormalizeVecs(int rows, int dim, float diff) {
    auto ds = GenDataSet(rows, dim);
    auto type_ds = knowhere::ConvertToDataTypeIfNeeded<T>(ds);
    auto data = (T*)type_ds->GetTensor();

    auto data_copy = knowhere::CopyAndNormalizeVecs<T>(data, rows, dim);

    for (int i = 0; i < rows; ++i) {
        float sum = 0.0;
        for (int j = 0; j < dim; ++j) {
            auto val = data_copy[i * dim + j];
            sum += val * val;
        }
        CHECK(std::abs(1.0f - sum) <= diff);
    }
}

TEST_CASE("Test Vector Normalization", "[normalize]") {
    using Catch::Approx;

    uint64_t rows = 100;
    uint64_t dim = 128;

    SECTION("Test Normalize Dataset") {
        CheckNormalizeDataset<knowhere::fp32>(rows, dim, 0.00001);
        CheckNormalizeDataset<knowhere::fp16>(rows, dim, 0.001);
        CheckNormalizeDataset<knowhere::bf16>(rows, dim, 0.01);
    }

    SECTION("Test Copy and Normalize Vectors") {
        CheckCopyAndNormalizeVecs<knowhere::fp32>(rows, dim, 0.00001);
        CheckCopyAndNormalizeVecs<knowhere::fp16>(rows, dim, 0.001);
        CheckCopyAndNormalizeVecs<knowhere::bf16>(rows, dim, 0.01);
    }
}

TEST_CASE("Test Bitset Generation", "[utils]") {
    SECTION("Sequential") {
        for (const auto size : kBitsetSizes) {
            for (size_t i = 0; i <= size; ++i) {
                auto bitset_data = GenerateBitsetWithFirstTbitsSet(size, i);
                knowhere::BitsetView bitset(bitset_data.data(), size);
                for (size_t j = 0; j < i; ++j) {
                    REQUIRE(bitset.test(j));
                }
                for (size_t j = i; j < size; ++j) {
                    REQUIRE(!bitset.test(j));
                }
            }
        }
    }

    SECTION("Random") {
        for (const auto size : kBitsetSizes) {
            for (size_t i = 0; i <= size; ++i) {
                auto bitset_data = GenerateBitsetWithRandomTbitsSet(size, i);
                knowhere::BitsetView bitset(bitset_data.data(), size);
                size_t cnt = 0;
                for (size_t j = 0; j < size; ++j) {
                    cnt += bitset.test(j);
                }
                REQUIRE(cnt == i);
            }
        }
    }
}

TEST_CASE("Test BitsetView Roaring", "[utils]") {
    std::vector<uint8_t> dense(2, 0);
    for (auto id : {1, 3, 8, 13}) {
        dense[id >> 3] |= 0x1 << (id & 0x7);
    }

    auto* roaring = roaring_bitmap_create();
    for (auto id : {1, 3, 8, 13}) {
        roaring_bitmap_add(roaring, id);
    }

    knowhere::BitsetView dense_view(dense.data(), 16);
    knowhere::BitsetView roaring_view(roaring, 16);
    REQUIRE(roaring_view.is_roaring());
    REQUIRE(roaring_view.count() == 0);
    REQUIRE(roaring_view.get_filtered_out_num_() == dense_view.get_filtered_out_num_());
    REQUIRE(roaring_view.filter_ratio() == dense_view.filter_ratio());
    REQUIRE(roaring_view.get_first_valid_index() == dense_view.get_first_valid_index());
    for (size_t i = 0; i < 16; ++i) {
        REQUIRE(roaring_view.test(i) == dense_view.test(i));
    }
    REQUIRE(roaring_view.ToDense() == dense);

    roaring_view.set_id_offset(2);
    dense_view.set_id_offset(2);
    for (size_t i = 0; i < 14; ++i) {
        REQUIRE(roaring_view.test(i) == dense_view.test(i));
    }

    std::vector<uint32_t> out_ids{0, 1, 3, 7, 8, 13};
    knowhere::BitsetView roaring_out_ids_view(roaring, 16);
    knowhere::BitsetView dense_out_ids_view(dense.data(), 16);
    roaring_out_ids_view.set_out_ids(out_ids.data(), out_ids.size());
    dense_out_ids_view.set_out_ids(out_ids.data(), out_ids.size());
    REQUIRE(roaring_out_ids_view.count() == dense_out_ids_view.count());
    for (size_t i = 0; i < out_ids.size(); ++i) {
        REQUIRE(roaring_out_ids_view.test(i) == dense_out_ids_view.test(i));
    }

    roaring_bitmap_free(roaring);
}

TEST_CASE("Test BitsetView Frozen Roaring", "[utils]") {
    auto* roaring = roaring_bitmap_create();
    for (auto id : {0, 2, 4, 10}) {
        roaring_bitmap_add(roaring, id);
    }
    const auto frozen_size = roaring_bitmap_frozen_size_in_bytes(roaring);
    std::vector<char> frozen(frozen_size);
    roaring_bitmap_frozen_serialize(roaring, frozen.data());

    auto frozen_view = knowhere::BitsetView::FromFrozenRoaring(frozen.data(), frozen.size(), 12);
    REQUIRE(frozen_view.is_roaring());
    REQUIRE(frozen_view.get_filtered_out_num_() == 4);
    REQUIRE(frozen_view.get_first_valid_index() == 1);
    REQUIRE(frozen_view.test(0));
    REQUIRE(!frozen_view.test(1));
    REQUIRE(frozen_view.test(10));
    REQUIRE(frozen_view.test(12));

    roaring_bitmap_free(roaring);
}

namespace {
constexpr size_t kHeapSize = 10;
constexpr size_t kElementCount = 10000;
}  // namespace

TEST_CASE("ResultMaxHeap") {
    knowhere::ResultMaxHeap<float, size_t> heap(kHeapSize);
    auto pairs = GenerateRandomDistanceIdPair(kElementCount);
    for (const auto& [dist, id] : pairs) {
        heap.Push(dist, id);
    }
    REQUIRE(heap.Size() == kHeapSize);
    std::sort(pairs.begin(), pairs.end());
    for (int i = kHeapSize - 1; i >= 0; --i) {
        auto op = heap.Pop();
        REQUIRE(op.has_value());
        REQUIRE(op.value().second == pairs[i].second);
    }
    REQUIRE(heap.Size() == 0);
}

TEST_CASE("Test Time Recorder") {
    knowhere::TimeRecorder tr("test", 2);
    int64_t sum = 0;
    for (int i = 0; i < 10000; i++) {
        sum += i * i;
    }
    auto span = tr.ElapseFromBegin("done");
    REQUIRE(span > 0);
    REQUIRE(sum > 0);
}

TEST_CASE("Test Version") {
    REQUIRE(knowhere::Version::VersionSupport(knowhere::Version::GetDefaultVersion()));
    REQUIRE(knowhere::Version::VersionSupport(knowhere::Version::GetMinimalVersion()));
    REQUIRE(knowhere::Version::VersionSupport(knowhere::Version::GetCurrentVersion()));
    REQUIRE(knowhere::Version::VersionSupport(knowhere::Version::GetMaximumVersion()));
    REQUIRE(knowhere::Version::GetMinimalVersion() <= knowhere::Version::GetCurrentVersion());
    REQUIRE(knowhere::Version::GetCurrentVersion() <= knowhere::Version::GetMaximumVersion());
}

TEST_CASE("Test DiskLoad") {
    REQUIRE(knowhere::UseDiskLoad(knowhere::IndexEnum::INDEX_DISKANN,
                                  knowhere::Version::GetCurrentVersion().VersionNumber()));
#ifdef KNOWHERE_WITH_CARDINAL
    REQUIRE(
        knowhere::UseDiskLoad(knowhere::IndexEnum::INDEX_HNSW, knowhere::Version::GetCurrentVersion().VersionNumber()));
#else
    REQUIRE(!knowhere::UseDiskLoad(knowhere::IndexEnum::INDEX_HNSW,
                                   knowhere::Version::GetCurrentVersion().VersionNumber()));
#endif
}

TEST_CASE("Test ThreadPool") {
    SECTION("Build thread pool") {
        knowhere::ThreadPool::InitGlobalBuildThreadPool(0);
        auto prev_build_thread_num = knowhere::ThreadPool::GetGlobalBuildThreadPoolSize();
        knowhere::ThreadPool::InitGlobalBuildThreadPool(prev_build_thread_num);

        knowhere::ThreadPool::SetGlobalBuildThreadPoolSize(2);
        REQUIRE(knowhere::ThreadPool::GetGlobalBuildThreadPoolSize() == 2);
        knowhere::ThreadPool::SetGlobalBuildThreadPoolSize(4);
        REQUIRE(knowhere::ThreadPool::GetGlobalBuildThreadPoolSize() == 4);
        knowhere::ThreadPool::SetGlobalBuildThreadPoolSize(0);
        REQUIRE(knowhere::ThreadPool::GetGlobalBuildThreadPoolSize() == 4);

        REQUIRE(knowhere::ThreadPool::GetBuildThreadPoolPendingTaskCount() == 0);

        if (prev_build_thread_num > 0) {
            knowhere::ThreadPool::SetGlobalBuildThreadPoolSize(prev_build_thread_num);
        }
    }

    SECTION("Search thread pool") {
        knowhere::ThreadPool::InitGlobalSearchThreadPool(0);
        auto prev_search_thread_num = knowhere::ThreadPool::GetGlobalSearchThreadPoolSize();
        knowhere::ThreadPool::InitGlobalSearchThreadPool(prev_search_thread_num);

        knowhere::ThreadPool::SetGlobalSearchThreadPoolSize(2);
        REQUIRE(knowhere::ThreadPool::GetGlobalSearchThreadPoolSize() == 2);
        knowhere::ThreadPool::SetGlobalSearchThreadPoolSize(4);
        REQUIRE(knowhere::ThreadPool::GetGlobalSearchThreadPoolSize() == 4);
        knowhere::ThreadPool::SetGlobalSearchThreadPoolSize(0);
        REQUIRE(knowhere::ThreadPool::GetGlobalSearchThreadPoolSize() == 4);

        REQUIRE(knowhere::ThreadPool::GetSearchThreadPoolPendingTaskCount() == 0);

        if (prev_search_thread_num > 0) {
            knowhere::ThreadPool::SetGlobalSearchThreadPoolSize(prev_search_thread_num);
        }
    }

    SECTION("ScopedBuildOmpSetter") {
        int prev_num_threads = knowhere::ThreadPool::GetGlobalBuildThreadPoolSize();
        {
            int target_num_threads = (prev_num_threads / 2) > 0 ? (prev_num_threads / 2) : 1;
            knowhere::ThreadPool::ScopedBuildOmpSetter setter(target_num_threads);
            auto thread_num_1 = omp_get_max_threads();
            REQUIRE(thread_num_1 == target_num_threads);
        }
        auto thread_num_2 = omp_get_max_threads();
        REQUIRE(thread_num_2 == prev_num_threads);
    }

    SECTION("ScopedSearchOmpSetter") {
        int prev_num_threads = knowhere::ThreadPool::GetGlobalSearchThreadPoolSize();
        {
            int target_num_threads = (prev_num_threads / 2) > 0 ? (prev_num_threads / 2) : 1;
            knowhere::ThreadPool::ScopedSearchOmpSetter setter(target_num_threads);
            auto thread_num_1 = omp_get_max_threads();
            REQUIRE(thread_num_1 == target_num_threads);
        }
        auto thread_num_2 = omp_get_max_threads();
        REQUIRE(thread_num_2 == prev_num_threads);
    }
}

TEST_CASE("Test WaitAllSuccess with folly::Unit futures") {
    auto pool = knowhere::ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<folly::Unit>> futures;

    SECTION("All futures succeed") {
        for (size_t i = 0; i < 10; ++i) {
            futures.emplace_back(pool->push([]() { return folly::Unit(); }));
        }
        REQUIRE(knowhere::WaitAllSuccess(futures) == knowhere::Status::success);
    }

    SECTION("One future throws an exception") {
        for (size_t i = 0; i < 10; ++i) {
            futures.emplace_back(pool->push([i]() {
                if (i == 5) {
                    throw std::runtime_error("Task failed");
                }
                return folly::Unit();
            }));
        }
        REQUIRE_THROWS_AS(knowhere::WaitAllSuccess(futures), std::runtime_error);
    }

    SECTION("WaitAllSuccess should wait until all tasks finish even if any throws exception") {
        std::atomic<int> externalValue{0};

        futures.emplace_back(pool->push([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            REQUIRE(externalValue.load() == 1);
            externalValue.store(2);
            return folly::Unit();
        }));

        futures.emplace_back(pool->push([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            externalValue.store(1);
            throw std::runtime_error("Task failed");
        }));

        REQUIRE_THROWS_AS(knowhere::WaitAllSuccess(futures), std::runtime_error);
        REQUIRE(externalValue.load() == 2);
    }
}

TEST_CASE("Test WaitAllSuccess with knowhere::Status futures") {
    auto pool = knowhere::ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<knowhere::Status>> futures;

    SECTION("All futures succeed with Status::success") {
        for (size_t i = 0; i < 10; ++i) {
            futures.emplace_back(pool->push([]() { return knowhere::Status::success; }));
        }
        REQUIRE(knowhere::WaitAllSuccess(futures) == knowhere::Status::success);
    }

    SECTION("One future returns Status::invalid_args") {
        for (size_t i = 0; i < 10; ++i) {
            futures.emplace_back(pool->push([i]() {
                if (i == 5) {
                    return knowhere::Status::invalid_args;
                }
                return knowhere::Status::success;
            }));
        }
        REQUIRE(knowhere::WaitAllSuccess(futures) == knowhere::Status::invalid_args);
    }

    SECTION("One future throws an exception") {
        for (size_t i = 0; i < 10; ++i) {
            futures.emplace_back(pool->push([i]() {
                if (i == 5) {
                    throw std::runtime_error("Task failed");
                }
                return knowhere::Status::success;
            }));
        }
        REQUIRE_THROWS_AS(knowhere::WaitAllSuccess(futures), std::runtime_error);
    }
}
