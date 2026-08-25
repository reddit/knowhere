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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "hnswlib/hnswalg.h"
#include "index/gpu_cagra_cpu/gpu_cagra_cpu_config.h"
#include "io/memory_io.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node.h"
#include "knowhere/index/index_node_thread_pool_wrapper.h"
#include "knowhere/log.h"

namespace knowhere {

// Loads GPU_CAGRA blobs that were serialized with adapt_for_cpu (hnswlib,
// key GPU_CUVS_CAGRA_cpu). Build/serialize stay on the CUDA GPU_CAGRA node.
template <typename DataType>
class GpuCagraCpuIndexNode : public IndexNode {
 public:
    using DistType = float;

    GpuCagraCpuIndexNode(const int32_t& /*version*/, const Object& /*object*/) {
    }

    Status
    Train(const DataSetPtr, std::shared_ptr<Config>, bool) override {
        LOG_KNOWHERE_ERROR_ << "GPU_CAGRA CPU node cannot train; build on a GPU DataNode with adapt_for_cpu=true";
        return Status::not_implemented;
    }

    Status
    Add(const DataSetPtr, std::shared_ptr<Config>, bool) override {
        return Status::not_implemented;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext*) const override {
        if (hnsw_index_ == nullptr) {
            return expected<DataSetPtr>::Err(Status::empty_index, "GPU_CAGRA cpu index not loaded");
        }
        const auto& cagra_cfg = static_cast<const GpuCagraCpuConfig&>(*cfg);
        auto nq = dataset->GetRows();
        auto xq = dataset->GetTensor();
        auto k = cagra_cfg.k.value();

        auto p_id = std::make_unique<int64_t[]>(k * nq);
        auto p_dist = std::make_unique<DistType[]>(k * nq);

        hnswlib::SearchParam param{(size_t)cagra_cfg.ef.value()};
        bool transform = (hnsw_index_->metric_type_ == hnswlib::Metric::INNER_PRODUCT ||
                          hnsw_index_->metric_type_ == hnswlib::Metric::COSINE);

        for (int i = 0; i < nq; ++i) {
            auto p_id_ptr = p_id.get();
            auto p_dist_ptr = p_dist.get();
            auto single_query = (const char*)xq + i * hnsw_index_->data_size_;
            auto rst = hnsw_index_->searchKnn(single_query, k, bitset, &param);
            size_t rst_size = rst.size();
            auto p_single_dis = p_dist_ptr + i * k;
            auto p_single_id = p_id_ptr + i * k;
            for (size_t idx = 0; idx < rst_size; ++idx) {
                const auto& [dist, id] = rst[idx];
                p_single_dis[idx] = transform ? (-dist) : dist;
                p_single_id[idx] = id;
            }
            for (size_t idx = rst_size; idx < (size_t)k; idx++) {
                p_single_dis[idx] = DistType(1.0 / 0.0);
                p_single_id[idx] = -1;
            }
        }

        return GenResultDataSet(nq, k, p_id.release(), p_dist.release());
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext*) const override {
        if (!hnsw_index_) {
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }
        auto dim = Dim();
        auto rows = dataset->GetRows();
        auto ids = dataset->GetIds();
        try {
            auto data = std::make_unique<uint8_t[]>(hnsw_index_->data_size_ * rows);
            for (int64_t i = 0; i < rows; i++) {
                int64_t id = ids[i];
                std::copy_n(hnsw_index_->getDataByInternalId(id), hnsw_index_->data_size_,
                            data.get() + i * hnsw_index_->data_size_);
            }
            return GenResultDataSet(rows, dim, std::move(data));
        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "hnsw inner error: " << e.what();
            return expected<DataSetPtr>::Err(Status::hnsw_inner_error, e.what());
        }
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig&, const IndexVersion&) {
        return true;
    }

    bool
    HasRawData(const std::string&) const override {
        return true;
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config>) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
    }

    Status
    Serialize(BinarySet&) const override {
        LOG_KNOWHERE_ERROR_ << "GPU_CAGRA CPU node cannot serialize; build on a GPU DataNode";
        return Status::not_implemented;
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config>) override {
        if constexpr (std::is_same_v<DataType, std::int8_t>) {
            LOG_KNOWHERE_ERROR_ << "CAGRA+HNSW does not support INT8 data.";
            return Status::invalid_binary_set;
        }
        BinaryPtr binary = binset.GetByNames({
            std::string(IndexEnum::INDEX_CUVS_CAGRA) + "_cpu",
            std::string(IndexEnum::INDEX_GPU_CAGRA) + "_cpu",
        });
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "GPU_CAGRA CPU load needs a GPU_CUVS_CAGRA_cpu (adapt_for_cpu) binary; native CAGRA "
                                   "cannot be deserialized without CUDA. Rebuild the index with adapt_for_cpu=true.";
            return Status::invalid_binary_set;
        }
        hnswlib::SpaceInterface<float>* space = nullptr;
        hnsw_index_.reset(new (std::nothrow) hnswlib::HierarchicalNSW<DataType, float, hnswlib::None>(space));
        try {
            MemoryIOReader reader(binary->data.get(), binary->size);
            hnsw_index_->loadIndex(reader);
            hnsw_index_->base_layer_only = true;
        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "hnsw inner error: " << e.what();
            hnsw_index_.reset();
            return Status::hnsw_inner_error;
        }
        return Status::success;
    }

    Status
    DeserializeFromFile(const std::string&, std::shared_ptr<Config>) override {
        return Status::not_implemented;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<GpuCagraCpuConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    int64_t
    Dim() const override {
        if (!hnsw_index_) {
            return 0;
        }
        return (*static_cast<size_t*>(hnsw_index_->dist_func_param_));
    }

    int64_t
    Size() const override {
        if (!hnsw_index_) {
            return 0;
        }
        return hnsw_index_->cal_size();
    }

    int64_t
    Count() const override {
        if (!hnsw_index_) {
            return 0;
        }
        return hnsw_index_->cur_element_count;
    }

    std::string
    Type() const override {
        return IndexEnum::INDEX_GPU_CAGRA;
    }

 private:
    std::unique_ptr<hnswlib::HierarchicalNSW<DataType, float, hnswlib::None>> hnsw_index_ = nullptr;
};

KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_CAGRA, GpuCagraCpuIndexNode, fp32, knowhere::feature::FLOAT32, 8);
KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_CUVS_CAGRA, GpuCagraCpuIndexNode, fp32, knowhere::feature::FLOAT32, 8);
KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_CAGRA, GpuCagraCpuIndexNode, fp16, knowhere::feature::FP16, 8);
KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_CUVS_CAGRA, GpuCagraCpuIndexNode, fp16, knowhere::feature::FP16, 8);

}  // namespace knowhere
