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

#include "simd/block_seek.h"

namespace knowhere::sparse {

// Reference implementation using branchless binary search
// This is the fallback for CPUs without SIMD support
size_t
simd_seek_in_block_ref(const uint32_t* block_ids, uint32_t target, size_t block_len) {
    if (block_len == 0) {
        return 0;
    }

    // Early exit: target is already >= first element or <= last element checks
    if (block_ids[0] >= target) {
        return 0;
    }
    if (block_ids[block_len - 1] < target) {
        return block_len;
    }

    // Branchless binary search for first element >= target
    size_t lo = 0;
    size_t n = block_len;
    while (n > 1) {
        size_t half = n >> 1;
        size_t mid = lo + half;
        // Branchless: lo = (block_ids[mid] < target) ? mid : lo
        size_t mask = -static_cast<size_t>(block_ids[mid] < target);
        lo += (half & mask);
        // n = (block_ids[mid] < target) ? n - half : half + (n & 1)
        n = half + ((n - 2 * half) & mask);
    }

    // lo now points to first element >= target, or block_len if not found
    return lo + static_cast<size_t>(block_ids[lo] < target);
}

}  // namespace knowhere::sparse

