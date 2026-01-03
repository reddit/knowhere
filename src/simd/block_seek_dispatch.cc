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
#include "simd/hook.h"

namespace knowhere::sparse {

// Global function pointer - initialized to reference implementation
BlockSeekFn simd_seek_in_block = simd_seek_in_block_ref;

// Initialize dispatch based on CPU capabilities
void
init_block_seek_dispatch() {
#if defined(__x86_64__)
    if (faiss::cpu_support_avx512()) {
        simd_seek_in_block = simd_seek_in_block_avx512;
    } else if (faiss::cpu_support_avx2()) {
        simd_seek_in_block = simd_seek_in_block_avx2;
    } else if (faiss::cpu_support_sse4_2()) {
        simd_seek_in_block = simd_seek_in_block_sse;
    } else {
        simd_seek_in_block = simd_seek_in_block_ref;
    }
#else
    // Non-x86_64 platforms use reference implementation
    simd_seek_in_block = simd_seek_in_block_ref;
#endif
}

// Auto-initialize on library load using static initializer
namespace {
struct BlockSeekInitializer {
    BlockSeekInitializer() {
        init_block_seek_dispatch();
    }
};
static BlockSeekInitializer g_block_seek_init;
}  // namespace

}  // namespace knowhere::sparse

