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

#if defined(__x86_64__) && defined(__SSE4_2__)
#include <emmintrin.h>  // SSE2
#include <smmintrin.h>  // SSE4.1

namespace knowhere::sparse {

// SSE4.2 implementation: processes 4 x 32-bit doc IDs per iteration
// Uses unsigned comparison by XORing with sign bit to convert to signed domain
size_t
simd_seek_in_block_sse(const uint32_t* block_ids, uint32_t target, size_t block_len) {
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

    // Broadcast target to all 4 lanes
    __m128i target_vec = _mm_set1_epi32(static_cast<int32_t>(target));

    // For unsigned comparison, XOR with 0x80000000 to flip sign bit
    __m128i sign_bit = _mm_set1_epi32(static_cast<int32_t>(0x80000000u));
    __m128i target_signed = _mm_xor_si128(target_vec, sign_bit);

    size_t i = 0;
    // Process 4 elements at a time
    for (; i + 4 <= block_len; i += 4) {
        // Load 4 doc IDs (unaligned load)
        __m128i data_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&block_ids[i]));

        // Convert to signed domain for comparison
        __m128i data_signed = _mm_xor_si128(data_vec, sign_bit);

        // data >= target = (data > target) | (data == target)
        __m128i cmp_gt = _mm_cmpgt_epi32(data_signed, target_signed);
        __m128i cmp_eq = _mm_cmpeq_epi32(data_vec, target_vec);
        __m128i cmp_ge = _mm_or_si128(cmp_gt, cmp_eq);

        // Extract mask
        int mask = _mm_movemask_ps(_mm_castsi128_ps(cmp_ge));

        if (mask != 0) {
            return i + __builtin_ctz(static_cast<unsigned int>(mask));
        }
    }

    // Handle remaining elements (0-3)
    for (; i < block_len; ++i) {
        if (block_ids[i] >= target) {
            return i;
        }
    }

    return block_len;
}

}  // namespace knowhere::sparse

#else
// Stub for non-SSE builds
namespace knowhere::sparse {
size_t
simd_seek_in_block_sse(const uint32_t* block_ids, uint32_t target, size_t block_len) {
    return simd_seek_in_block_ref(block_ids, target, block_len);
}
}  // namespace knowhere::sparse
#endif

