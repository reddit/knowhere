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

#ifndef BLOCK_SEEK_H
#define BLOCK_SEEK_H

#include <cstddef>
#include <cstdint>

namespace knowhere::sparse {

// Block size for sparse index posting lists
// Must match kBlockSize in sparse_utils.h
constexpr size_t kSimdBlockSize = 128;

// Function pointer type for runtime dispatch
using BlockSeekFn = size_t (*)(const uint32_t* block_ids, uint32_t target, size_t block_len);

// Reference implementation (fallback)
size_t simd_seek_in_block_ref(const uint32_t* block_ids, uint32_t target, size_t block_len);

// SSE4.2 implementation (4 x 32-bit per iteration)
size_t simd_seek_in_block_sse(const uint32_t* block_ids, uint32_t target, size_t block_len);

// AVX2 implementation (8 x 32-bit per iteration)
size_t simd_seek_in_block_avx2(const uint32_t* block_ids, uint32_t target, size_t block_len);

// AVX-512 implementation (16 x 32-bit per iteration)
size_t simd_seek_in_block_avx512(const uint32_t* block_ids, uint32_t target, size_t block_len);

// Runtime-dispatched function pointer (set during initialization)
extern BlockSeekFn simd_seek_in_block;

// Initialize the function pointer based on CPU capabilities
// Called automatically on first use, but can be called explicitly
void init_block_seek_dispatch();

}  // namespace knowhere::sparse

#endif  // BLOCK_SEEK_H

