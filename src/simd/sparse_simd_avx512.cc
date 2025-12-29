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

#if defined(__x86_64__)

#include "sparse_simd_avx512.h"

#include <immintrin.h>

#include <algorithm>

namespace knowhere {
namespace sparse {

namespace {
// Helper: Horizontal max reduction for __m512
inline float
horizontal_max_ps_512(__m512 v) {
    // Reduce 512 bits to 256 bits
    __m256 max_low = _mm512_castps512_ps256(v);
    __m256 max_high = _mm512_extractf32x8_ps(v, 1);
    __m256 max256 = _mm256_max_ps(max_low, max_high);

    // Reduce 256 bits to 128 bits
    __m128 max_high128 = _mm256_extractf128_ps(max256, 1);
    __m128 max_low128 = _mm256_castps256_ps128(max256);
    __m128 max128 = _mm_max_ps(max_low128, max_high128);

    // Horizontal max within 128 bits
    __m128 shuf = _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1));
    max128 = _mm_max_ps(max128, shuf);
    shuf = _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2));
    max128 = _mm_max_ps(max128, shuf);

    return _mm_cvtss_f32(max128);
}
}  // namespace

float
sparse_find_max_float_avx512(const float* data, size_t len) {
    if (len == 0) {
        return 0.0f;
    }

    __m512 max_vec = _mm512_set1_ps(data[0]);

    size_t i = 0;
    // Process 16 floats at a time with AVX-512
    for (; i + 16 <= len; i += 16) {
        __m512 vals = _mm512_loadu_ps(data + i);
        max_vec = _mm512_max_ps(max_vec, vals);
    }

    // Reduce to scalar max
    float max_val = horizontal_max_ps_512(max_vec);

    // Handle remaining elements
    for (; i < len; ++i) {
        max_val = std::max(max_val, data[i]);
    }

    return max_val;
}

void
sparse_accumulate_scores_ip_avx512(float* scores, const uint32_t* doc_ids, const float* vals, size_t len,
                                    float q_weight) {
    // AVX-512 has scatter store instruction _mm512_i32scatter_ps
    // This is much more efficient than AVX2 for scatter operations
    //
    // NOTE: This implementation assumes all doc_ids within a batch of 16 are UNIQUE.
    // This is guaranteed by the sparse inverted index structure where each document
    // appears at most once per dimension in a posting list. If duplicates exist in
    // the same batch, the gather-add-scatter pattern would lose intermediate updates.

    __m512 q_weight_vec = _mm512_set1_ps(q_weight);

    size_t i = 0;
    // Process 16 elements at a time
    for (; i + 16 <= len; i += 16) {
        // Load 16 values
        __m512 vals_vec = _mm512_loadu_ps(vals + i);
        // Multiply by query weight
        __m512 weighted_vals = _mm512_mul_ps(vals_vec, q_weight_vec);

        // Load 16 doc_ids
        __m512i doc_ids_vec = _mm512_loadu_si512((__m512i*)(doc_ids + i));

        // Gather current scores
        __m512 current_scores = _mm512_i32gather_ps(doc_ids_vec, scores, 4);

        // Add weighted values
        __m512 new_scores = _mm512_add_ps(current_scores, weighted_vals);

        // Scatter back to scores array
        _mm512_i32scatter_ps(scores, doc_ids_vec, new_scores, 4);
    }

    // Handle remaining elements
    for (; i < len; ++i) {
        scores[doc_ids[i]] += q_weight * vals[i];
    }
}

void
sparse_accumulate_scores_bm25_avx512(float* scores, const uint32_t* doc_ids, const float* vals, const float* row_sums,
                                      size_t len, float q_weight, float k1, float b, float avgdl) {
    // BM25 formula: score = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    //
    // NOTE: This implementation assumes all doc_ids within a batch of 16 are UNIQUE.
    // See sparse_accumulate_scores_ip_avx512 for details.
    const __m512 k1_vec = _mm512_set1_ps(k1);
    const __m512 k1_plus_1_vec = _mm512_set1_ps(k1 + 1.0f);
    const __m512 one_minus_b_vec = _mm512_set1_ps(1.0f - b);
    const __m512 b_over_avgdl_vec = _mm512_set1_ps(b / avgdl);
    const __m512 q_weight_vec = _mm512_set1_ps(q_weight);

    size_t i = 0;
    // Process 16 elements at a time
    for (; i + 16 <= len; i += 16) {
        // Load term frequencies (tf)
        __m512 tf_vec = _mm512_loadu_ps(vals + i);

        // Load document IDs
        __m512i doc_ids_vec = _mm512_loadu_si512((__m512i*)(doc_ids + i));

        // Gather document lengths (dl)
        __m512 dl_vec = _mm512_i32gather_ps(doc_ids_vec, row_sums, 4);

        // Compute: k1 * (1 - b + b * dl / avgdl)
        __m512 dl_factor = _mm512_mul_ps(b_over_avgdl_vec, dl_vec);      // b * dl / avgdl
        dl_factor = _mm512_add_ps(one_minus_b_vec, dl_factor);           // 1 - b + b * dl / avgdl
        __m512 k1_factor = _mm512_mul_ps(k1_vec, dl_factor);             // k1 * (...)

        // Compute denominator: tf + k1 * (1 - b + b * dl / avgdl)
        __m512 denominator = _mm512_add_ps(tf_vec, k1_factor);

        // Compute numerator: tf * (k1 + 1)
        __m512 numerator = _mm512_mul_ps(tf_vec, k1_plus_1_vec);

        // Compute BM25 score: numerator / denominator
        __m512 bm25_score = _mm512_div_ps(numerator, denominator);

        // Multiply by query weight
        __m512 weighted_score = _mm512_mul_ps(bm25_score, q_weight_vec);

        // Gather current scores
        __m512 current_scores = _mm512_i32gather_ps(doc_ids_vec, scores, 4);

        // Add new scores
        __m512 new_scores = _mm512_add_ps(current_scores, weighted_score);

        // Scatter back to scores array
        _mm512_i32scatter_ps(scores, doc_ids_vec, new_scores, 4);
    }

    // Handle remaining elements (scalar)
    const float k1_plus_1 = k1 + 1.0f;
    const float one_minus_b = 1.0f - b;
    const float b_over_avgdl = b / avgdl;

    for (; i < len; ++i) {
        uint32_t doc_id = doc_ids[i];
        float tf = vals[i];
        float dl = row_sums[doc_id];

        float denominator = tf + k1 * (one_minus_b + b_over_avgdl * dl);
        float bm25_score = (tf * k1_plus_1) / denominator;

        scores[doc_id] += q_weight * bm25_score;
    }
}

void
sparse_accumulate_scores_ip_u16_avx512(float* scores, const uint32_t* doc_ids, const uint16_t* vals, size_t len,
                                        float q_weight) {
    // AVX-512 implementation for uint16_t quantized values
    // Process 16 uint16_t values at a time, convert to float in-register
    //
    // NOTE: This implementation assumes all doc_ids within a batch of 16 are UNIQUE.
    // This is guaranteed by the sparse inverted index structure where each document
    // appears at most once per dimension in a posting list.

    __m512 q_weight_vec = _mm512_set1_ps(q_weight);

    size_t i = 0;
    // Process 16 elements at a time
    for (; i + 16 <= len; i += 16) {
        // Load 16 uint16_t values (256 bits)
        __m256i vals_u16 = _mm256_loadu_si256((__m256i*)(vals + i));

        // Convert uint16 to int32 (expands to 512 bits)
        __m512i vals_i32 = _mm512_cvtepu16_epi32(vals_u16);

        // Convert int32 to float
        __m512 vals_f32 = _mm512_cvtepi32_ps(vals_i32);

        // Multiply by query weight
        __m512 weighted_vals = _mm512_mul_ps(vals_f32, q_weight_vec);

        // Load 16 doc_ids
        __m512i doc_ids_vec = _mm512_loadu_si512((__m512i*)(doc_ids + i));

        // Gather current scores
        __m512 current_scores = _mm512_i32gather_ps(doc_ids_vec, scores, 4);

        // Add weighted values
        __m512 new_scores = _mm512_add_ps(current_scores, weighted_vals);

        // Scatter back to scores array
        _mm512_i32scatter_ps(scores, doc_ids_vec, new_scores, 4);
    }

    // Handle remaining elements
    for (; i < len; ++i) {
        scores[doc_ids[i]] += q_weight * static_cast<float>(vals[i]);
    }
}

void
sparse_accumulate_scores_bm25_u16_avx512(float* scores, const uint32_t* doc_ids, const uint16_t* vals,
                                          const float* row_sums, size_t len, float q_weight, float k1, float b,
                                          float avgdl) {
    // BM25 formula: score = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    // AVX-512 implementation for uint16_t quantized term frequencies
    //
    // NOTE: This implementation assumes all doc_ids within a batch of 16 are UNIQUE.
    const __m512 k1_vec = _mm512_set1_ps(k1);
    const __m512 k1_plus_1_vec = _mm512_set1_ps(k1 + 1.0f);
    const __m512 one_minus_b_vec = _mm512_set1_ps(1.0f - b);
    const __m512 b_over_avgdl_vec = _mm512_set1_ps(b / avgdl);
    const __m512 q_weight_vec = _mm512_set1_ps(q_weight);

    size_t i = 0;
    // Process 16 elements at a time
    for (; i + 16 <= len; i += 16) {
        // Load 16 uint16_t term frequencies
        __m256i tf_u16 = _mm256_loadu_si256((__m256i*)(vals + i));

        // Convert uint16 to int32, then to float
        __m512i tf_i32 = _mm512_cvtepu16_epi32(tf_u16);
        __m512 tf_vec = _mm512_cvtepi32_ps(tf_i32);

        // Load document IDs
        __m512i doc_ids_vec = _mm512_loadu_si512((__m512i*)(doc_ids + i));

        // Gather document lengths (dl)
        __m512 dl_vec = _mm512_i32gather_ps(doc_ids_vec, row_sums, 4);

        // Compute: k1 * (1 - b + b * dl / avgdl)
        __m512 dl_factor = _mm512_mul_ps(b_over_avgdl_vec, dl_vec);      // b * dl / avgdl
        dl_factor = _mm512_add_ps(one_minus_b_vec, dl_factor);           // 1 - b + b * dl / avgdl
        __m512 k1_factor = _mm512_mul_ps(k1_vec, dl_factor);             // k1 * (...)

        // Compute denominator: tf + k1 * (1 - b + b * dl / avgdl)
        __m512 denominator = _mm512_add_ps(tf_vec, k1_factor);

        // Compute numerator: tf * (k1 + 1)
        __m512 numerator = _mm512_mul_ps(tf_vec, k1_plus_1_vec);

        // Compute BM25 score: numerator / denominator
        __m512 bm25_score = _mm512_div_ps(numerator, denominator);

        // Multiply by query weight
        __m512 weighted_score = _mm512_mul_ps(bm25_score, q_weight_vec);

        // Gather current scores
        __m512 current_scores = _mm512_i32gather_ps(doc_ids_vec, scores, 4);

        // Add new scores
        __m512 new_scores = _mm512_add_ps(current_scores, weighted_score);

        // Scatter back to scores array
        _mm512_i32scatter_ps(scores, doc_ids_vec, new_scores, 4);
    }

    // Handle remaining elements (scalar)
    const float k1_plus_1 = k1 + 1.0f;
    const float one_minus_b = 1.0f - b;
    const float b_over_avgdl = b / avgdl;

    for (; i < len; ++i) {
        uint32_t doc_id = doc_ids[i];
        float tf = static_cast<float>(vals[i]);
        float dl = row_sums[doc_id];

        float denominator = tf + k1 * (one_minus_b + b_over_avgdl * dl);
        float bm25_score = (tf * k1_plus_1) / denominator;

        scores[doc_id] += q_weight * bm25_score;
    }
}

}  // namespace sparse
}  // namespace knowhere

#endif  // __x86_64__
