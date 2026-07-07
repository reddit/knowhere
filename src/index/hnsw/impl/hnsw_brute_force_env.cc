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

#include "index/hnsw/impl/hnsw_brute_force_env.h"

#include <strings.h>

#include <cstdlib>

namespace knowhere {

namespace {

constexpr const char* kDisableHnswBruteForceEnv = "KNOWHERE_DISABLE_HNSW_BRUTE_FORCE";

bool
ParseTruthyEnv(const char* value) {
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return strcasecmp(value, "1") == 0 || strcasecmp(value, "true") == 0 || strcasecmp(value, "yes") == 0 ||
           strcasecmp(value, "on") == 0;
}

}  // namespace

bool
IsHnswBruteForceDisabledByEnv() {
    static const bool disabled = ParseTruthyEnv(std::getenv(kDisableHnswBruteForceEnv));
    return disabled;
}

}  // namespace knowhere
