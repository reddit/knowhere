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

#if defined(__x86_64__) && defined(__AVX2__)
#include <immintrin.h>

namespace knowhere::sparse {

// AVX2 implementation: processes 8 x 32-bit doc IDs per iteration
// Uses unsigned comparison by XORing with sign bit to convert to signed domain
size_t
simd_seek_in_block_avx2(const uint32_t* block_ids, uint32_t target, size_t block_len) {
    if (block_len == 0) {
        return 0;
    }

    // Early exit checks
    if (block_ids[0] >= target) {
        return 0;
    }
    if (block_ids[block_len - 1] < target) {
        return block_len;
    }

    // Broadcast target to all lanes
    __m256i target_vec = _mm256_set1_epi32(static_cast<int32_t>(target));

    // For unsigned comparison, we XOR with 0x80000000 to flip sign bit
    // This converts unsigned comparison to signed comparison domain
    // (Only needed if doc IDs can exceed INT32_MAX, but included for correctness)
    __m256i sign_bit = _mm256_set1_epi32(static_cast<int32_t>(0x80000000u));
    __m256i target_signed = _mm256_xor_si256(target_vec, sign_bit);

    size_t i = 0;
    // Process 8 elements at a time
    for (; i + 8 <= block_len; i += 8) {
        // Load 8 doc IDs (unaligned load for safety - block starts may not be aligned)
        __m256i data_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&block_ids[i]));

        // Convert to signed domain for comparison
        __m256i data_signed = _mm256_xor_si256(data_vec, sign_bit);

        // Compare: data >= target is equivalent to !(data < target) = !(target > data)
        // We want first position where data >= target
        // cmpgt gives us data > target - 1 when we want data >= target
        // So we use: cmp_ge = ~(target > data) = (data > target) | (data == target)
        __m256i cmp_gt = _mm256_cmpgt_epi32(data_signed, target_signed);  // data > target
        __m256i cmp_eq = _mm256_cmpeq_epi32(data_vec, target_vec);        // data == target
        __m256i cmp_ge = _mm256_or_si256(cmp_gt, cmp_eq);                 // data >= target

        // Extract mask: each bit corresponds to one 32-bit lane
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp_ge));

        if (mask != 0) {
            // Found at least one element >= target
            // Count trailing zeros to find first matching lane
            return i + __builtin_ctz(static_cast<unsigned int>(mask));
        }
    }

    // Handle remaining elements (0-7)
    for (; i < block_len; ++i) {
        if (block_ids[i] >= target) {
            return i;
        }
    }

    return block_len;
}

}  // namespace knowhere::sparse

#else
// Stub for non-x86_64 or non-AVX2 builds
namespace knowhere::sparse {
size_t
simd_seek_in_block_avx2(const uint32_t* block_ids, uint32_t target, size_t block_len) {
    return simd_seek_in_block_ref(block_ids, target, block_len);
}
}  // namespace knowhere::sparse
#endif

