// Copyright (C) 2019-2024 Zilliz. All rights reserved.
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

namespace knowhere {

// Env: KNOWHERE_DISABLE_HNSW_BRUTE_FORCE
// Truthy values: 1, true, yes, on (case-insensitive).
//
// When enabled, sealed HNSW segments never use IndexBruteForceWrapper:
//   - upfront BF for selective filters / large topk
//   - post-HNSW fallback when graph search returns too few results
//
// Evaluated once per process on first call (intended for query-node pod env).
bool
IsHnswBruteForceDisabledByEnv();

}  // namespace knowhere
