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

#if defined(__x86_64__) && defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>

namespace knowhere::sparse {

// AVX-512 implementation: processes 16 x 32-bit doc IDs per iteration
// Uses native unsigned comparison intrinsics available in AVX-512
size_t
simd_seek_in_block_avx512(const uint32_t* block_ids, uint32_t target, size_t block_len) {
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

    // Broadcast target to all 16 lanes
    __m512i target_vec = _mm512_set1_epi32(static_cast<int32_t>(target));

    size_t i = 0;
    // Process 16 elements at a time
    for (; i + 16 <= block_len; i += 16) {
        // Load 16 doc IDs (unaligned load for safety)
        __m512i data_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&block_ids[i]));

        // AVX-512 has native unsigned comparison!
        // _mm512_cmpge_epu32_mask returns a 16-bit mask where each bit
        // indicates if the corresponding lane satisfies data >= target
        __mmask16 mask = _mm512_cmpge_epu32_mask(data_vec, target_vec);

        if (mask != 0) {
            // Found at least one element >= target
            // Count trailing zeros to find first matching lane
            return i + __builtin_ctz(static_cast<unsigned int>(mask));
        }
    }

    // Handle remaining elements (0-15)
    for (; i < block_len; ++i) {
        if (block_ids[i] >= target) {
            return i;
        }
    }

    return block_len;
}

}  // namespace knowhere::sparse

#else
// Stub for non-AVX512 builds - fall back to AVX2 or reference
namespace knowhere::sparse {
size_t
simd_seek_in_block_avx512(const uint32_t* block_ids, uint32_t target, size_t block_len) {
#if defined(__x86_64__) && defined(__AVX2__)
    return simd_seek_in_block_avx2(block_ids, target, block_len);
#else
    return simd_seek_in_block_ref(block_ids, target, block_len);
#endif
}
}  // namespace knowhere::sparse
#endif

