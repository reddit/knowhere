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

/// NEON implementation: Find maximum value in float array
float
sparse_find_max_float_neon(const float* data, size_t len);

/// NEON implementation: Accumulate scores with scatter (IP metric)
void
sparse_accumulate_scores_ip_neon(float* scores, const uint32_t* doc_ids, const float* vals, size_t len,
                                  float q_weight);

/// NEON implementation: Accumulate scores with BM25 computation
void
sparse_accumulate_scores_bm25_neon(float* scores, const uint32_t* doc_ids, const float* vals, const float* row_sums,
                                    size_t len, float q_weight, float k1, float b, float avgdl);

/// NEON implementation: Accumulate scores with scatter (IP metric, uint16_t quantized)
void
sparse_accumulate_scores_ip_u16_neon(float* scores, const uint32_t* doc_ids, const uint16_t* vals, size_t len,
                                      float q_weight);

/// NEON implementation: Accumulate scores with BM25 computation (uint16_t quantized)
void
sparse_accumulate_scores_bm25_u16_neon(float* scores, const uint32_t* doc_ids, const uint16_t* vals,
                                        const float* row_sums, size_t len, float q_weight, float k1, float b,
                                        float avgdl);

}  // namespace sparse
}  // namespace knowhere
