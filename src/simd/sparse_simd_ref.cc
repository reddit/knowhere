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

#include "sparse_simd_ref.h"

#include <algorithm>
#include <cmath>

namespace knowhere {
namespace sparse {

float
sparse_find_max_float_ref(const float* data, size_t len) {
    if (len == 0) {
        return 0.0f;
    }
    float max_val = data[0];
    for (size_t i = 1; i < len; ++i) {
        max_val = std::max(max_val, data[i]);
    }
    return max_val;
}

void
sparse_accumulate_scores_ip_ref(float* scores, const uint32_t* doc_ids, const float* vals, size_t len,
                                 float q_weight) {
    for (size_t i = 0; i < len; ++i) {
        scores[doc_ids[i]] += q_weight * vals[i];
    }
}

void
sparse_accumulate_scores_bm25_ref(float* scores, const uint32_t* doc_ids, const float* vals, const float* row_sums,
                                   size_t len, float q_weight, float k1, float b, float avgdl) {
    // BM25 formula: score = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    const float k1_plus_1 = k1 + 1.0f;
    const float one_minus_b = 1.0f - b;
    const float b_over_avgdl = b / avgdl;

    for (size_t i = 0; i < len; ++i) {
        uint32_t doc_id = doc_ids[i];
        float tf = vals[i];
        float dl = row_sums[doc_id];

        // BM25 score component for this term
        float denominator = tf + k1 * (one_minus_b + b_over_avgdl * dl);
        float bm25_score = (tf * k1_plus_1) / denominator;

        scores[doc_id] += q_weight * bm25_score;
    }
}

void
sparse_accumulate_scores_ip_u16_ref(float* scores, const uint32_t* doc_ids, const uint16_t* vals, size_t len,
                                     float q_weight) {
    for (size_t i = 0; i < len; ++i) {
        scores[doc_ids[i]] += q_weight * static_cast<float>(vals[i]);
    }
}

void
sparse_accumulate_scores_bm25_u16_ref(float* scores, const uint32_t* doc_ids, const uint16_t* vals,
                                       const float* row_sums, size_t len, float q_weight, float k1, float b,
                                       float avgdl) {
    // BM25 formula: score = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    const float k1_plus_1 = k1 + 1.0f;
    const float one_minus_b = 1.0f - b;
    const float b_over_avgdl = b / avgdl;

    for (size_t i = 0; i < len; ++i) {
        uint32_t doc_id = doc_ids[i];
        float tf = static_cast<float>(vals[i]);
        float dl = row_sums[doc_id];

        // BM25 score component for this term
        float denominator = tf + k1 * (one_minus_b + b_over_avgdl * dl);
        float bm25_score = (tf * k1_plus_1) / denominator;

        scores[doc_id] += q_weight * bm25_score;
    }
}

}  // namespace sparse
}  // namespace knowhere
