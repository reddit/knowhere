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

#include "knowhere/sparse_utils.h"

#include "simd/sparse_simd.h"

namespace knowhere {
namespace sparse {

float
BlockMaxInfo::max_score_from_block(size_t block_idx) const {
    if (block_idx >= block_max_scores.size()) {
        return 0.0f;
    }

    // Use SIMD-optimized function for finding max
    // The function pointer is initialized by the hook system at startup
    return sparse_find_max_float(block_max_scores.data() + block_idx, block_max_scores.size() - block_idx);
}

}  // namespace sparse
}  // namespace knowhere
