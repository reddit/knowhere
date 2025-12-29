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

#if defined(__aarch64__)

#include "sparse_simd_neon.h"

#include <arm_neon.h>

#include <algorithm>

namespace knowhere {
namespace sparse {

namespace {
// Helper: Horizontal max reduction for float32x4_t
inline float
horizontal_max_f32(float32x4_t v) {
    // Pairwise max
    float32x2_t max_pair = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
    max_pair = vpmax_f32(max_pair, max_pair);
    return vget_lane_f32(max_pair, 0);
}
}  // namespace

float
sparse_find_max_float_neon(const float* data, size_t len) {
    if (len == 0) {
        return 0.0f;
    }

    float32x4_t max_vec = vdupq_n_f32(data[0]);

    size_t i = 0;
    // Process 4 floats at a time with NEON
    for (; i + 4 <= len; i += 4) {
        float32x4_t vals = vld1q_f32(data + i);
        max_vec = vmaxq_f32(max_vec, vals);
    }

    // Reduce to scalar max
    float max_val = horizontal_max_f32(max_vec);

    // Handle remaining elements
    for (; i < len; ++i) {
        max_val = std::max(max_val, data[i]);
    }

    return max_val;
}

void
sparse_accumulate_scores_ip_neon(float* scores, const uint32_t* doc_ids, const float* vals, size_t len,
                                  float q_weight) {
    // NEON doesn't have efficient scatter operations
    // Vectorize the multiply, then scatter manually

    float32x4_t q_weight_vec = vdupq_n_f32(q_weight);

    size_t i = 0;
    // Process 4 elements at a time
    for (; i + 4 <= len; i += 4) {
        // Load 4 values
        float32x4_t vals_vec = vld1q_f32(vals + i);
        // Multiply by query weight
        float32x4_t weighted_vals = vmulq_f32(vals_vec, q_weight_vec);

        // Store to temporary buffer
        float temp[4];
        vst1q_f32(temp, weighted_vals);

        // Scatter to scores array
        for (int j = 0; j < 4; ++j) {
            scores[doc_ids[i + j]] += temp[j];
        }
    }

    // Handle remaining elements
    for (; i < len; ++i) {
        scores[doc_ids[i]] += q_weight * vals[i];
    }
}

void
sparse_accumulate_scores_bm25_neon(float* scores, const uint32_t* doc_ids, const float* vals, const float* row_sums,
                                    size_t len, float q_weight, float k1, float b, float avgdl) {
    // BM25 formula: score = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    const float32x4_t k1_vec = vdupq_n_f32(k1);
    const float32x4_t k1_plus_1_vec = vdupq_n_f32(k1 + 1.0f);
    const float32x4_t one_minus_b_vec = vdupq_n_f32(1.0f - b);
    const float32x4_t b_over_avgdl_vec = vdupq_n_f32(b / avgdl);
    const float32x4_t q_weight_vec = vdupq_n_f32(q_weight);

    size_t i = 0;
    // Process 4 elements at a time
    for (; i + 4 <= len; i += 4) {
        // Load term frequencies (tf)
        float32x4_t tf_vec = vld1q_f32(vals + i);

        // Load document lengths (dl) - manual gather
        float dl_array[4];
        for (int j = 0; j < 4; ++j) {
            dl_array[j] = row_sums[doc_ids[i + j]];
        }
        float32x4_t dl_vec = vld1q_f32(dl_array);

        // Compute: k1 * (1 - b + b * dl / avgdl)
        float32x4_t dl_factor = vmulq_f32(b_over_avgdl_vec, dl_vec);      // b * dl / avgdl
        dl_factor = vaddq_f32(one_minus_b_vec, dl_factor);                 // 1 - b + b * dl / avgdl
        float32x4_t k1_factor = vmulq_f32(k1_vec, dl_factor);              // k1 * (...)

        // Compute denominator: tf + k1 * (1 - b + b * dl / avgdl)
        float32x4_t denominator = vaddq_f32(tf_vec, k1_factor);

        // Compute numerator: tf * (k1 + 1)
        float32x4_t numerator = vmulq_f32(tf_vec, k1_plus_1_vec);

        // Compute BM25 score: numerator / denominator
        // NEON doesn't have direct division, use reciprocal estimate + Newton-Raphson
        // Two iterations for better precision (~0.01% error instead of ~0.4%)
        float32x4_t recip = vrecpeq_f32(denominator);
        recip = vmulq_f32(vrecpsq_f32(denominator, recip), recip);  // First iteration
        recip = vmulq_f32(vrecpsq_f32(denominator, recip), recip);  // Second iteration
        float32x4_t bm25_score = vmulq_f32(numerator, recip);

        // Multiply by query weight
        float32x4_t weighted_score = vmulq_f32(bm25_score, q_weight_vec);

        // Store to temporary buffer
        float temp[4];
        vst1q_f32(temp, weighted_score);

        // Scatter to scores array
        for (int j = 0; j < 4; ++j) {
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
sparse_accumulate_scores_ip_u16_neon(float* scores, const uint32_t* doc_ids, const uint16_t* vals, size_t len,
                                      float q_weight) {
    // NEON implementation for uint16_t quantized values
    // Process 4 uint16_t values at a time, convert to float in-register

    float32x4_t q_weight_vec = vdupq_n_f32(q_weight);

    size_t i = 0;
    // Process 4 elements at a time
    for (; i + 4 <= len; i += 4) {
        // Load 4 uint16_t values (64 bits)
        uint16x4_t vals_u16 = vld1_u16(vals + i);

        // Convert uint16 to uint32
        uint32x4_t vals_u32 = vmovl_u16(vals_u16);

        // Convert uint32 to float
        float32x4_t vals_f32 = vcvtq_f32_u32(vals_u32);

        // Multiply by query weight
        float32x4_t weighted_vals = vmulq_f32(vals_f32, q_weight_vec);

        // Store to temporary buffer
        float temp[4];
        vst1q_f32(temp, weighted_vals);

        // Scatter to scores array
        for (int j = 0; j < 4; ++j) {
            scores[doc_ids[i + j]] += temp[j];
        }
    }

    // Handle remaining elements
    for (; i < len; ++i) {
        scores[doc_ids[i]] += q_weight * static_cast<float>(vals[i]);
    }
}

void
sparse_accumulate_scores_bm25_u16_neon(float* scores, const uint32_t* doc_ids, const uint16_t* vals,
                                        const float* row_sums, size_t len, float q_weight, float k1, float b,
                                        float avgdl) {
    // BM25 formula: score = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    // NEON implementation for uint16_t quantized term frequencies
    const float32x4_t k1_vec = vdupq_n_f32(k1);
    const float32x4_t k1_plus_1_vec = vdupq_n_f32(k1 + 1.0f);
    const float32x4_t one_minus_b_vec = vdupq_n_f32(1.0f - b);
    const float32x4_t b_over_avgdl_vec = vdupq_n_f32(b / avgdl);
    const float32x4_t q_weight_vec = vdupq_n_f32(q_weight);

    size_t i = 0;
    // Process 4 elements at a time
    for (; i + 4 <= len; i += 4) {
        // Load 4 uint16_t term frequencies
        uint16x4_t tf_u16 = vld1_u16(vals + i);

        // Convert uint16 to uint32, then to float
        uint32x4_t tf_u32 = vmovl_u16(tf_u16);
        float32x4_t tf_vec = vcvtq_f32_u32(tf_u32);

        // Load document lengths (dl) - manual gather
        float dl_array[4];
        for (int j = 0; j < 4; ++j) {
            dl_array[j] = row_sums[doc_ids[i + j]];
        }
        float32x4_t dl_vec = vld1q_f32(dl_array);

        // Compute: k1 * (1 - b + b * dl / avgdl)
        float32x4_t dl_factor = vmulq_f32(b_over_avgdl_vec, dl_vec);      // b * dl / avgdl
        dl_factor = vaddq_f32(one_minus_b_vec, dl_factor);                 // 1 - b + b * dl / avgdl
        float32x4_t k1_factor = vmulq_f32(k1_vec, dl_factor);              // k1 * (...)

        // Compute denominator: tf + k1 * (1 - b + b * dl / avgdl)
        float32x4_t denominator = vaddq_f32(tf_vec, k1_factor);

        // Compute numerator: tf * (k1 + 1)
        float32x4_t numerator = vmulq_f32(tf_vec, k1_plus_1_vec);

        // Compute BM25 score: numerator / denominator
        // Use reciprocal estimate + Newton-Raphson iterations
        float32x4_t recip = vrecpeq_f32(denominator);
        recip = vmulq_f32(vrecpsq_f32(denominator, recip), recip);  // First iteration
        recip = vmulq_f32(vrecpsq_f32(denominator, recip), recip);  // Second iteration
        float32x4_t bm25_score = vmulq_f32(numerator, recip);

        // Multiply by query weight
        float32x4_t weighted_score = vmulq_f32(bm25_score, q_weight_vec);

        // Store to temporary buffer
        float temp[4];
        vst1q_f32(temp, weighted_score);

        // Scatter to scores array
        for (int j = 0; j < 4; ++j) {
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

#endif  // __aarch64__
