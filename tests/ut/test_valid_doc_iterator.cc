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
