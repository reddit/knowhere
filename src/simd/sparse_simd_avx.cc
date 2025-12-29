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

#include "sparse_simd_avx.h"

#include <immintrin.h>

#include <algorithm>

namespace knowhere {
namespace sparse {

namespace {
// Helper: Horizontal max reduction for __m256
inline float
horizontal_max_ps(__m256 v) {
    // Extract high 128 bits and low 128 bits, then max
    __m128 max_high = _mm256_extractf128_ps(v, 1);
    __m128 max_low = _mm256_castps256_ps128(v);
    __m128 max128 = _mm_max_ps(max_low, max_high);

    // Horizontal max within 128 bits
    __m128 shuf = _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1));
    max128 = _mm_max_ps(max128, shuf);
    shuf = _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2));
    max128 = _mm_max_ps(max128, shuf);

    return _mm_cvtss_f32(max128);
}
}  // namespace

float
sparse_find_max_float_avx(const float* data, size_t len) {
    if (len == 0) {
        return 0.0f;
    }

    __m256 max_vec = _mm256_set1_ps(data[0]);

    size_t i = 0;
    // Process 8 floats at a time with AVX
    for (; i + 8 <= len; i += 8) {
        __m256 vals = _mm256_loadu_ps(data + i);
        max_vec = _mm256_max_ps(max_vec, vals);
    }

    // Reduce to scalar max
    float max_val = horizontal_max_ps(max_vec);

    // Handle remaining elements
    for (; i < len; ++i) {
        max_val = std::max(max_val, data[i]);
    }

    return max_val;
}

void
sparse_accumulate_scores_ip_avx(float* scores, const uint32_t* doc_ids, const float* vals, size_t len,
                                 float q_weight) {
    // For scatter operations, SIMD benefit is limited due to random writes
    // However, we can still vectorize the multiply operation

    __m256 q_weight_vec = _mm256_set1_ps(q_weight);

    size_t i = 0;
    // Process 8 elements at a time
    for (; i + 8 <= len; i += 8) {
        // Load 8 values
        __m256 vals_vec = _mm256_loadu_ps(vals + i);
        // Multiply by query weight
        __m256 weighted_vals = _mm256_mul_ps(vals_vec, q_weight_vec);

        // Unfortunately, scatter writes are still sequential
        // AVX2 doesn't have efficient scatter store for variable indices
        // Store to temporary buffer and scatter manually
        float temp[8];
        _mm256_storeu_ps(temp, weighted_vals);

        // Scatter to scores array
        for (int j = 0; j < 8; ++j) {
            scores[doc_ids[i + j]] += temp[j];
        }
    }

    // Handle remaining elements
    for (; i < len; ++i) {
        scores[doc_ids[i]] += q_weight * vals[i];
    }
}

void
sparse_accumulate_scores_bm25_avx(float* scores, const uint32_t* doc_ids, const float* vals, const float* row_sums,
                                   size_t len, float q_weight, float k1, float b, float avgdl) {
    // BM25 formula: score = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    const __m256 k1_vec = _mm256_set1_ps(k1);
    const __m256 k1_plus_1_vec = _mm256_set1_ps(k1 + 1.0f);
    const __m256 one_minus_b_vec = _mm256_set1_ps(1.0f - b);
    const __m256 b_over_avgdl_vec = _mm256_set1_ps(b / avgdl);
    const __m256 q_weight_vec = _mm256_set1_ps(q_weight);

    size_t i = 0;
    // Process 8 elements at a time
    for (; i + 8 <= len; i += 8) {
        // Load term frequencies (tf)
        __m256 tf_vec = _mm256_loadu_ps(vals + i);

        // Load document lengths (dl) - requires gather
        // Since AVX2 has _mm256_i32gather_ps, we can use it efficiently
        __m256i doc_ids_vec = _mm256_loadu_si256((__m256i*)(doc_ids + i));
        __m256 dl_vec = _mm256_i32gather_ps(row_sums, doc_ids_vec, 4);  // scale=4 for float (4 bytes)

        // Compute: k1 * (1 - b + b * dl / avgdl)
        __m256 dl_factor = _mm256_mul_ps(b_over_avgdl_vec, dl_vec);                 // b * dl / avgdl
        dl_factor = _mm256_add_ps(one_minus_b_vec, dl_factor);                      // 1 - b + b * dl / avgdl
        __m256 k1_factor = _mm256_mul_ps(k1_vec, dl_factor);                        // k1 * (...)

        // Compute denominator: tf + k1 * (1 - b + b * dl / avgdl)
        __m256 denominator = _mm256_add_ps(tf_vec, k1_factor);

        // Compute numerator: tf * (k1 + 1)
        __m256 numerator = _mm256_mul_ps(tf_vec, k1_plus_1_vec);

        // Compute BM25 score: numerator / denominator
        __m256 bm25_score = _mm256_div_ps(numerator, denominator);

        // Multiply by query weight
        __m256 weighted_score = _mm256_mul_ps(bm25_score, q_weight_vec);

        // Store to temporary buffer for scatter
        float temp[8];
        _mm256_storeu_ps(temp, weighted_score);

        // Scatter to scores array
        for (int j = 0; j < 8; ++j) {
            scores[doc_ids[i + j]] += temp[j];
        }
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
sparse_accumulate_scores_ip_u16_avx(float* scores, const uint32_t* doc_ids, const uint16_t* vals, size_t len,
                                     float q_weight) {
    // AVX2 implementation for uint16_t quantized values
    // Process 8 uint16_t values at a time, convert to float in-register

    __m256 q_weight_vec = _mm256_set1_ps(q_weight);

    size_t i = 0;
    // Process 8 elements at a time
    for (; i + 8 <= len; i += 8) {
        // Load 8 uint16_t values (128 bits)
        __m128i vals_u16 = _mm_loadu_si128((__m128i*)(vals + i));

        // Convert uint16 to int32 (expands to 256 bits)
        __m256i vals_i32 = _mm256_cvtepu16_epi32(vals_u16);

        // Convert int32 to float
        __m256 vals_f32 = _mm256_cvtepi32_ps(vals_i32);

        // Multiply by query weight
        __m256 weighted_vals = _mm256_mul_ps(vals_f32, q_weight_vec);

        // Store to temporary buffer and scatter manually (AVX2 lacks scatter)
        float temp[8];
        _mm256_storeu_ps(temp, weighted_vals);

        // Scatter to scores array
        for (int j = 0; j < 8; ++j) {
            scores[doc_ids[i + j]] += temp[j];
        }
    }

    // Handle remaining elements
    for (; i < len; ++i) {
        scores[doc_ids[i]] += q_weight * static_cast<float>(vals[i]);
    }
}

void
sparse_accumulate_scores_bm25_u16_avx(float* scores, const uint32_t* doc_ids, const uint16_t* vals,
                                       const float* row_sums, size_t len, float q_weight, float k1, float b,
                                       float avgdl) {
    // BM25 formula: score = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    // AVX2 implementation for uint16_t quantized term frequencies
    const __m256 k1_vec = _mm256_set1_ps(k1);
    const __m256 k1_plus_1_vec = _mm256_set1_ps(k1 + 1.0f);
    const __m256 one_minus_b_vec = _mm256_set1_ps(1.0f - b);
    const __m256 b_over_avgdl_vec = _mm256_set1_ps(b / avgdl);
    const __m256 q_weight_vec = _mm256_set1_ps(q_weight);

    size_t i = 0;
    // Process 8 elements at a time
    for (; i + 8 <= len; i += 8) {
        // Load 8 uint16_t term frequencies
        __m128i tf_u16 = _mm_loadu_si128((__m128i*)(vals + i));

        // Convert uint16 to int32, then to float
        __m256i tf_i32 = _mm256_cvtepu16_epi32(tf_u16);
        __m256 tf_vec = _mm256_cvtepi32_ps(tf_i32);

        // Load document IDs for gather
        __m256i doc_ids_vec = _mm256_loadu_si256((__m256i*)(doc_ids + i));

        // Gather document lengths (dl)
        __m256 dl_vec = _mm256_i32gather_ps(row_sums, doc_ids_vec, 4);

        // Compute: k1 * (1 - b + b * dl / avgdl)
        __m256 dl_factor = _mm256_mul_ps(b_over_avgdl_vec, dl_vec);      // b * dl / avgdl
        dl_factor = _mm256_add_ps(one_minus_b_vec, dl_factor);           // 1 - b + b * dl / avgdl
        __m256 k1_factor = _mm256_mul_ps(k1_vec, dl_factor);             // k1 * (...)

        // Compute denominator: tf + k1 * (1 - b + b * dl / avgdl)
        __m256 denominator = _mm256_add_ps(tf_vec, k1_factor);

        // Compute numerator: tf * (k1 + 1)
        __m256 numerator = _mm256_mul_ps(tf_vec, k1_plus_1_vec);

        // Compute BM25 score: numerator / denominator
        __m256 bm25_score = _mm256_div_ps(numerator, denominator);

        // Multiply by query weight
        __m256 weighted_score = _mm256_mul_ps(bm25_score, q_weight_vec);

        // Store to temporary buffer for scatter
        float temp[8];
        _mm256_storeu_ps(temp, weighted_score);

        // Scatter to scores array
        for (int j = 0; j < 8; ++j) {
            scores[doc_ids[i + j]] += temp[j];
        }
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
