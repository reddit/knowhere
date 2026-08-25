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

#include "catch2/catch_test_macros.hpp"
#include "knowhere/binaryset.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/version.h"
#include "utils.h"

using namespace knowhere;

#ifndef KNOWHERE_WITH_CUVS
TEST_CASE("GPU_CAGRA cpu factory", "[gpu_cagra_cpu]") {
    auto ver = Version::GetCurrentVersion().VersionNumber();
    REQUIRE(IndexFactory::Instance().FeatureCheck(IndexEnum::INDEX_GPU_CAGRA, feature::FLOAT32));
    REQUIRE_FALSE(IndexFactory::Instance().FeatureCheck(IndexEnum::INDEX_GPU_CAGRA, feature::GPU));

    auto index = IndexFactory::Instance().Create<fp32>(IndexEnum::INDEX_GPU_CAGRA, ver);
    REQUIRE(index.has_value());
    REQUIRE(index.value().Type() == IndexEnum::INDEX_GPU_CAGRA);

    auto ds = GenDataSet(8, 4);
    REQUIRE(index.value().Train(ds, Json::object()) == Status::not_implemented);

    BinarySet binset;
    REQUIRE(index.value().Deserialize(binset, Json::object()) == Status::invalid_binary_set);
}
#endif
