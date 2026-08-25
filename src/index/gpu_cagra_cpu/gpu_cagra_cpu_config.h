// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <string>

#include "knowhere/config.h"

namespace knowhere {

// CPU-only GPU_CAGRA config. Declares the fields QueryNode/DataNode pass through
// on load/search without pulling in CUDA/cuVS headers.
struct GpuCagraCpuConfig : public BaseConfig {
    CFG_BOOL adapt_for_cpu;
    CFG_INT ef;

    KNOHWERE_DECLARE_CONFIG(GpuCagraCpuConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(adapt_for_cpu)
            .description("train on GPU, search on CPU")
            .set_default(true)
            .for_train()
            .for_deserialize();
        KNOWHERE_CONFIG_DECLARE_FIELD(ef)
            .description("hnsw ef")
            .allow_empty_without_default()
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_search();
    }

    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        if (param_type == PARAM_TYPE::SEARCH && !ef.has_value()) {
            ef = std::max(k.value(), CFG_INT::value_type{64});
        }
        if (param_type == PARAM_TYPE::TRAIN) {
            constexpr std::array<std::string_view, 3> legal_metric_list{"L2", "IP", "COSINE"};
            std::string metric = metric_type.value();
            if (std::find(legal_metric_list.begin(), legal_metric_list.end(), metric) == legal_metric_list.end()) {
                std::string msg = "metric type " + metric + " not found or not supported, supported: [L2 IP COSINE]";
                return HandleError(err_msg, msg, Status::invalid_metric_type);
            }
        }
        return Status::success;
    }
};

}  // namespace knowhere
