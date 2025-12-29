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

#pragma once

#include <cstddef>
#include <cstdint>

namespace knowhere {
namespace sparse {

/// Find maximum value in a float array using SIMD
/// Used for efficient block max score computation in filter-aware WAND
extern float (*sparse_find_max_float)(const float* data, size_t len);

/// Accumulate scores with scatter pattern for sparse vector search
/// scores[doc_ids[i]] += vals[i] * q_weight
/// Used in compute_all_distances for TAAT search
extern void (*sparse_accumulate_scores_ip)(float* scores, const uint32_t* doc_ids, const float* vals, size_t len,
                                            float q_weight);

/// Accumulate scores with BM25 computation
/// scores[doc_ids[i]] += q_weight * BM25(vals[i], row_sums[doc_ids[i]], k1, b, avgdl)
/// Used in compute_all_distances for BM25 metric
extern void (*sparse_accumulate_scores_bm25)(float* scores, const uint32_t* doc_ids, const float* vals,
                                              const float* row_sums, size_t len, float q_weight, float k1, float b,
                                              float avgdl);

/// Accumulate scores with scatter pattern for sparse vector search (uint16_t quantized values)
/// scores[doc_ids[i]] += vals[i] * q_weight
/// Used in compute_all_distances for TAAT search with quantized posting lists
extern void (*sparse_accumulate_scores_ip_u16)(float* scores, const uint32_t* doc_ids, const uint16_t* vals, size_t len,
                                                float q_weight);

/// Accumulate scores with BM25 computation (uint16_t quantized values)
/// scores[doc_ids[i]] += q_weight * BM25(vals[i], row_sums[doc_ids[i]], k1, b, avgdl)
/// Used in compute_all_distances for BM25 metric with quantized posting lists
extern void (*sparse_accumulate_scores_bm25_u16)(float* scores, const uint32_t* doc_ids, const uint16_t* vals,
                                                  const float* row_sums, size_t len, float q_weight, float k1, float b,
                                                  float avgdl);

}  // namespace sparse
}  // namespace knowhere
