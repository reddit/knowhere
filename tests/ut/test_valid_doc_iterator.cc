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

#include <algorithm>
#include <random>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/bitsetview.h"
#include "knowhere/sparse_utils.h"

using namespace knowhere;

namespace {
constexpr int64_t kSeed = 42;

// Generate a bitset with specific bits set to 1 (filtered out)
std::vector<uint8_t>
GenerateBitset(size_t n, const std::vector<size_t>& filtered_ids) {
    std::vector<uint8_t> data((n + 7) / 8, 0);
    for (size_t id : filtered_ids) {
        if (id < n) {
            data[id >> 3] |= (0x1 << (id & 0x7));
        }
    }
    return data;
}

// Generate a bitset with first t bits set to 1 (filtered out)
std::vector<uint8_t>
GenerateBitsetWithFirstTbitsSet(size_t n, size_t t) {
    std::vector<uint8_t> data((n + 7) / 8, 0);
    for (size_t i = 0; i < t && i < n; ++i) {
        data[i >> 3] |= (0x1 << (i & 0x7));
    }
    return data;
}

// Generate a bitset with random t bits set to 1 (filtered out)
std::vector<uint8_t>
GenerateBitsetWithRandomTbitsSet(size_t n, size_t t) {
    std::vector<bool> bits_shuffle(n, false);
    for (size_t i = 0; i < t && i < n; ++i) {
        bits_shuffle[i] = true;
    }
    std::mt19937 g(kSeed);
    std::shuffle(bits_shuffle.begin(), bits_shuffle.end(), g);
    std::vector<uint8_t> data((n + 7) / 8, 0);
    for (size_t i = 0; i < n; ++i) {
        if (bits_shuffle[i]) {
            data[i >> 3] |= (0x1 << (i & 0x7));
        }
    }
    return data;
}

// Get all valid (non-filtered) doc IDs by linear scan
std::vector<size_t>
GetValidDocIds(const BitsetView& bitset, size_t n) {
    std::vector<size_t> valid_ids;
    for (size_t i = 0; i < n; ++i) {
        if (!bitset.test(i)) {
            valid_ids.push_back(i);
        }
    }
    return valid_ids;
}

}  // namespace

TEST_CASE("Test ValidDocIdIterator Basic Operations", "[valid_doc_iterator]") {
    SECTION("Empty bitset - all docs valid") {
        BitsetView empty_bitset;
        REQUIRE(empty_bitset.empty());

        // When bitset is empty, iterator behavior: all docs pass
        // But ValidDocIdIterator requires non-null bits, so this is a special case
        // In practice, empty bitset means no filtering
    }

    SECTION("All bits set - no valid docs") {
        const size_t n = 100;
        std::vector<uint8_t> data((n + 7) / 8, 0xFF);
        BitsetView bitset(data.data(), n);

        auto iter = bitset.valid_doc_iterator();
        REQUIRE_FALSE(iter.has_next());
    }

    SECTION("No bits set - all docs valid") {
        const size_t n = 100;
        std::vector<uint8_t> data((n + 7) / 8, 0);
        BitsetView bitset(data.data(), n);

        auto iter = bitset.valid_doc_iterator();

        std::vector<size_t> collected;
        while (iter.has_next()) {
            collected.push_back(iter.current());
            iter.advance_to_ge(iter.current() + 1);
        }

        REQUIRE(collected.size() == n);
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(collected[i] == i);
        }
    }

    SECTION("First half filtered") {
        const size_t n = 100;
        auto data = GenerateBitsetWithFirstTbitsSet(n, n / 2);
        BitsetView bitset(data.data(), n);

        auto iter = bitset.valid_doc_iterator();

        std::vector<size_t> collected;
        while (iter.has_next()) {
            collected.push_back(iter.current());
            iter.advance_to_ge(iter.current() + 1);
        }

        REQUIRE(collected.size() == n / 2);
        for (size_t i = 0; i < collected.size(); ++i) {
            REQUIRE(collected[i] == n / 2 + i);
        }
    }

    SECTION("Every other doc filtered") {
        const size_t n = 100;
        std::vector<size_t> filtered;
        for (size_t i = 0; i < n; i += 2) {
            filtered.push_back(i);
        }
        auto data = GenerateBitset(n, filtered);
        BitsetView bitset(data.data(), n);

        auto iter = bitset.valid_doc_iterator();

        std::vector<size_t> collected;
        while (iter.has_next()) {
            collected.push_back(iter.current());
            iter.advance_to_ge(iter.current() + 1);
        }

        REQUIRE(collected.size() == n / 2);
        for (size_t i = 0; i < collected.size(); ++i) {
            REQUIRE(collected[i] == 2 * i + 1);
        }
    }
}

TEST_CASE("Test ValidDocIdIterator advance_to_ge", "[valid_doc_iterator]") {
    SECTION("Advance skips filtered docs") {
        const size_t n = 1000;
        // Filter out docs 0-499, leaving 500-999 valid
        auto data = GenerateBitsetWithFirstTbitsSet(n, 500);
        BitsetView bitset(data.data(), n);

        auto iter = bitset.valid_doc_iterator();

        // First valid doc should be 500
        REQUIRE(iter.has_next());
        REQUIRE(iter.current() == 500);

        // Advance to 600
        iter.advance_to_ge(600);
        REQUIRE(iter.has_next());
        REQUIRE(iter.current() == 600);

        // Advance to 999
        iter.advance_to_ge(999);
        REQUIRE(iter.has_next());
        REQUIRE(iter.current() == 999);

        // Advance past end
        iter.advance_to_ge(1000);
        REQUIRE_FALSE(iter.has_next());
    }

    SECTION("Advance to exact match") {
        const size_t n = 100;
        std::vector<size_t> filtered = {0, 1, 2, 5, 6, 7, 10, 11};
        auto data = GenerateBitset(n, filtered);
        BitsetView bitset(data.data(), n);

        auto iter = bitset.valid_doc_iterator();

        // First valid is 3
        REQUIRE(iter.current() == 3);

        // Advance to 8 (valid)
        iter.advance_to_ge(8);
        REQUIRE(iter.current() == 8);

        // Advance to 10 (filtered) - should land on 12
        iter.advance_to_ge(10);
        REQUIRE(iter.current() == 12);
    }

    SECTION("Advance with large gaps") {
        const size_t n = 10000;
        // Filter all except every 100th doc
        std::vector<size_t> filtered;
        for (size_t i = 0; i < n; ++i) {
            if (i % 100 != 0) {
                filtered.push_back(i);
            }
        }
        auto data = GenerateBitset(n, filtered);
        BitsetView bitset(data.data(), n);

        auto iter = bitset.valid_doc_iterator();

        // Should start at 0
        REQUIRE(iter.current() == 0);

        // Advance to 50 - should land on 100
        iter.advance_to_ge(50);
        REQUIRE(iter.current() == 100);

        // Advance to 5050 - should land on 5100
        iter.advance_to_ge(5050);
        REQUIRE(iter.current() == 5100);
    }
}

TEST_CASE("Test ValidDocIdIterator Word-Level Optimization", "[valid_doc_iterator]") {
    SECTION("Large bitset with dense filtering") {
        const size_t n = 10000;
        // Filter 90% of docs
        auto data = GenerateBitsetWithRandomTbitsSet(n, n * 9 / 10);
        BitsetView bitset(data.data(), n);

        auto expected = GetValidDocIds(bitset, n);
        REQUIRE(expected.size() > 0);

        auto iter = bitset.valid_doc_iterator();

        std::vector<size_t> collected;
        while (iter.has_next()) {
            collected.push_back(iter.current());
            iter.advance_to_ge(iter.current() + 1);
        }

        REQUIRE(collected == expected);
    }

    SECTION("Large bitset with sparse filtering") {
        const size_t n = 10000;
        // Filter 10% of docs
        auto data = GenerateBitsetWithRandomTbitsSet(n, n / 10);
        BitsetView bitset(data.data(), n);

        auto expected = GetValidDocIds(bitset, n);

        auto iter = bitset.valid_doc_iterator();

        std::vector<size_t> collected;
        while (iter.has_next()) {
            collected.push_back(iter.current());
            iter.advance_to_ge(iter.current() + 1);
        }

        REQUIRE(collected == expected);
    }

    SECTION("Bitset aligned to 64-bit boundaries") {
        const size_t n = 640;  // 10 * 64 bits
        // Filter every 64th doc to test word boundary handling
        std::vector<size_t> filtered;
        for (size_t i = 0; i < n; i += 64) {
            filtered.push_back(i);
        }
        auto data = GenerateBitset(n, filtered);
        BitsetView bitset(data.data(), n);

        auto expected = GetValidDocIds(bitset, n);
        auto iter = bitset.valid_doc_iterator();

        std::vector<size_t> collected;
        while (iter.has_next()) {
            collected.push_back(iter.current());
            iter.advance_to_ge(iter.current() + 1);
        }

        REQUIRE(collected == expected);
    }
}

TEST_CASE("Test ValidVectorDocIdIterator", "[valid_doc_iterator]") {
    using namespace knowhere::sparse;

    SECTION("Empty vector") {
        std::vector<table_t> docids;
        DocIdFilterByVector filter(std::move(docids));

        auto iter = filter.valid_doc_iterator();
        REQUIRE_FALSE(iter.has_next());
    }

    SECTION("Single element") {
        std::vector<table_t> docids = {42};
        DocIdFilterByVector filter(std::move(docids));

        auto iter = filter.valid_doc_iterator();
        REQUIRE(iter.has_next());
        REQUIRE(iter.current() == 42);

        iter.advance_to_ge(43);
        REQUIRE_FALSE(iter.has_next());
    }

    SECTION("Sequential elements") {
        std::vector<table_t> docids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        DocIdFilterByVector filter(std::move(docids));

        auto iter = filter.valid_doc_iterator();

        std::vector<size_t> collected;
        while (iter.has_next()) {
            collected.push_back(iter.current());
            iter.advance_to_ge(iter.current() + 1);
        }

        REQUIRE(collected.size() == 10);
        for (size_t i = 0; i < 10; ++i) {
            REQUIRE(collected[i] == i);
        }
    }

    SECTION("Sparse elements with gaps") {
        std::vector<table_t> docids = {10, 50, 100, 500, 1000};
        DocIdFilterByVector filter(std::move(docids));

        auto iter = filter.valid_doc_iterator();

        // Start at 10
        REQUIRE(iter.current() == 10);

        // Advance to 30 - should land on 50
        iter.advance_to_ge(30);
        REQUIRE(iter.current() == 50);

        // Advance to 100 - exact match
        iter.advance_to_ge(100);
        REQUIRE(iter.current() == 100);

        // Advance to 200 - should land on 500
        iter.advance_to_ge(200);
        REQUIRE(iter.current() == 500);

        // Advance past end
        iter.advance_to_ge(2000);
        REQUIRE_FALSE(iter.has_next());
    }

    SECTION("Unsorted input gets sorted") {
        std::vector<table_t> docids = {50, 10, 100, 30, 20};
        DocIdFilterByVector filter(std::move(docids));

        auto iter = filter.valid_doc_iterator();

        std::vector<size_t> collected;
        while (iter.has_next()) {
            collected.push_back(iter.current());
            iter.advance_to_ge(iter.current() + 1);
        }

        // Should be sorted
        REQUIRE(collected == std::vector<size_t>{10, 20, 30, 50, 100});
    }
}

TEST_CASE("Test next_valid_doc_ge helper", "[valid_doc_iterator]") {
    SECTION("Empty bitset returns target") {
        BitsetView empty_bitset;
        REQUIRE(empty_bitset.next_valid_doc_ge(50) == 50);
    }

    SECTION("Find next valid doc") {
        const size_t n = 100;
        auto data = GenerateBitsetWithFirstTbitsSet(n, 60);
        BitsetView bitset(data.data(), n);

        // First 60 are filtered, so next valid >= 50 is 60
        REQUIRE(bitset.next_valid_doc_ge(50) == 60);

        // Next valid >= 60 is 60
        REQUIRE(bitset.next_valid_doc_ge(60) == 60);

        // Next valid >= 70 is 70
        REQUIRE(bitset.next_valid_doc_ge(70) == 70);

        // Past end returns size()
        REQUIRE(bitset.next_valid_doc_ge(100) == bitset.size());
    }
}

TEST_CASE("Test Iterator Interface Compatibility", "[valid_doc_iterator]") {
    // Both BitsetView and DocIdFilterByVector should provide the same interface
    // for valid_doc_iterator(). This test verifies the interface is consistent.

    SECTION("BitsetView iterator interface") {
        const size_t n = 100;
        std::vector<uint8_t> data((n + 7) / 8, 0);
        // Set bits 0-49 as filtered
        for (size_t i = 0; i < 50; ++i) {
            data[i >> 3] |= (0x1 << (i & 0x7));
        }
        BitsetView bitset(data.data(), n);

        auto iter = bitset.valid_doc_iterator();

        // Test has_next()
        REQUIRE(iter.has_next() == true);

        // Test current()
        REQUIRE(iter.current() == 50);

        // Test advance_to_ge()
        iter.advance_to_ge(75);
        REQUIRE(iter.current() == 75);
    }

    SECTION("DocIdFilterByVector iterator interface") {
        using namespace knowhere::sparse;

        std::vector<table_t> docids = {50, 51, 52, 75, 76, 99};
        DocIdFilterByVector filter(std::move(docids));

        auto iter = filter.valid_doc_iterator();

        // Test has_next()
        REQUIRE(iter.has_next() == true);

        // Test current()
        REQUIRE(iter.current() == 50);

        // Test advance_to_ge()
        iter.advance_to_ge(75);
        REQUIRE(iter.current() == 75);
    }
}

