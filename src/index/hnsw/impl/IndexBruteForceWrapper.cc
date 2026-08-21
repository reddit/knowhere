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

#include "index/hnsw/impl/IndexBruteForceWrapper.h"

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/cppcontrib/knowhere/impl/Bruteforce.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>

#include "knowhere/bitsetview.h"
#include "knowhere/bitsetview_idselector.h"

namespace knowhere {

using idx_t = faiss::idx_t;

// the following structure is a hack, because GCC cannot properly
//   de-virtualize a plain BitsetViewIDSelector.
struct BitsetViewIDSelectorWrapper final {
    const BitsetView bitset_view;

    inline BitsetViewIDSelectorWrapper(BitsetView bitset_view) : bitset_view{bitset_view} {
    }

    [[nodiscard]] inline bool
    is_member(faiss::idx_t id) const {
        // it is by design that bitset_view.empty() is not tested here
        return (!bitset_view.test(id));
    }
};

using RoaringBitmapPtr = std::unique_ptr<roaring_bitmap_t, decltype(&roaring_bitmap_free)>;

RoaringBitmapPtr
MakeValidRoaringBitmap(const BitsetView& bitset_view, idx_t ntotal) {
    if (!bitset_view.can_iterate_roaring_without_mapping() || ntotal < 0) {
        return {nullptr, roaring_bitmap_free};
    }

    const uint64_t range_start = bitset_view.id_offset();
    const uint64_t range_end = range_start + static_cast<uint64_t>(ntotal);
    constexpr uint64_t kRoaringUniverseEnd = uint64_t{std::numeric_limits<uint32_t>::max()} + 1;
    if (range_end < range_start || range_end > kRoaringUniverseEnd) {
        return {nullptr, roaring_bitmap_free};
    }

    RoaringBitmapPtr valid_ids(roaring_bitmap_from_range(range_start, range_end, 1), roaring_bitmap_free);
    if (valid_ids != nullptr) {
        roaring_bitmap_andnot_inplace(valid_ids.get(), bitset_view.roaring());
    }
    return valid_ids;
}

template<typename C>
void
SearchValidRoaringIds(const roaring_bitmap_t* valid_ids, uint64_t id_offset, faiss::DistanceComputer& dis, idx_t k,
                      float* distances, idx_t* labels) {
    faiss::cppcontrib::knowhere::brute_force_search_candidates_impl<C>(
        dis,
        [&](auto&& visit) {
            roaring_uint32_iterator_t iterator;
            roaring_iterator_init(valid_ids, &iterator);
            while (iterator.has_value) {
                visit(static_cast<idx_t>(static_cast<uint64_t>(iterator.current_value) - id_offset));
                roaring_uint32_iterator_advance(&iterator);
            }
        },
        k,
        distances,
        labels);
}

//
IndexBruteForceWrapper::IndexBruteForceWrapper(faiss::Index* underlying_index)
    : faiss::cppcontrib::knowhere::IndexWrapper{underlying_index} {
}

void
IndexBruteForceWrapper::search(faiss::idx_t n, const float* __restrict x, faiss::idx_t k, float* __restrict distances,
                               faiss::idx_t* __restrict labels,
                               const faiss::SearchParameters* __restrict params) const {
    FAISS_THROW_IF_NOT(k > 0);

    std::unique_ptr<faiss::DistanceComputer> dis(index->get_distance_computer());

    faiss::IDSelector* sel = (params == nullptr) ? nullptr : params->sel;
    const auto* bw_idselector = dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel);
    RoaringBitmapPtr valid_roaring_ids(nullptr, roaring_bitmap_free);
    if (bw_idselector && !bw_idselector->bitset_view.empty()) {
        valid_roaring_ids = MakeValidRoaringBitmap(bw_idselector->bitset_view, index->ntotal);
    }

    // no parallelism by design
    for (idx_t i = 0; i < n; i++) {
        // prepare the query
        dis->set_query(x + i * index->d);

        // allocate heap
        idx_t* const __restrict local_ids = labels + i * index->d;
        float* const __restrict local_distances = distances + i * index->d;

        if (is_similarity_metric(index->metric_type)) {
            using C = faiss::CMin<float, idx_t>;

            if (valid_roaring_ids != nullptr) {
                SearchValidRoaringIds<C>(valid_roaring_ids.get(), bw_idselector->bitset_view.id_offset(), *dis, k,
                                         local_distances, local_ids);
                continue;
            }

            // try knowhere-specific filter
            if (bw_idselector && !bw_idselector->bitset_view.empty()) {
                BitsetViewIDSelectorWrapper bw_idselector_w(bw_idselector->bitset_view);

                faiss::cppcontrib::knowhere::brute_force_search_impl<C, faiss::DistanceComputer,
                                                                     BitsetViewIDSelectorWrapper>(
                    index->ntotal, *dis, bw_idselector_w, k, local_distances, local_ids);
            } else {
                faiss::IDSelectorAll sel_all;
                faiss::cppcontrib::knowhere::brute_force_search_impl<C, faiss::DistanceComputer, faiss::IDSelectorAll>(
                    index->ntotal, *dis, sel_all, k, local_distances, local_ids);
            }
        } else {
            using C = faiss::CMax<float, idx_t>;

            if (valid_roaring_ids != nullptr) {
                SearchValidRoaringIds<C>(valid_roaring_ids.get(), bw_idselector->bitset_view.id_offset(), *dis, k,
                                         local_distances, local_ids);
                continue;
            }

            // try knowhere-specific filter
            if (bw_idselector && !bw_idselector->bitset_view.empty()) {
                BitsetViewIDSelectorWrapper bw_idselector_w(bw_idselector->bitset_view);

                faiss::cppcontrib::knowhere::brute_force_search_impl<C, faiss::DistanceComputer,
                                                                     BitsetViewIDSelectorWrapper>(
                    index->ntotal, *dis, bw_idselector_w, k, local_distances, local_ids);
            } else {
                faiss::IDSelectorAll sel_all;
                faiss::cppcontrib::knowhere::brute_force_search_impl<C, faiss::DistanceComputer, faiss::IDSelectorAll>(
                    index->ntotal, *dis, sel_all, k, local_distances, local_ids);
            }
        }
    }
}

void
IndexBruteForceWrapper::range_search(faiss::idx_t n, const float* x, float radius, faiss::RangeSearchResult* result,
                                     const faiss::SearchParameters* params) const {
    using RH_min = faiss::RangeSearchBlockResultHandler<faiss::CMax<float, int64_t>>;
    using RH_max = faiss::RangeSearchBlockResultHandler<faiss::CMin<float, int64_t>>;
    RH_min bres_min(result, radius);
    RH_max bres_max(result, radius);

    std::unique_ptr<faiss::DistanceComputer> dis(index->get_distance_computer());

    // no parallelism by design
    for (idx_t i = 0; i < n; i++) {
        // prepare the query
        dis->set_query(x + i * index->d);

        // set up a filter
        faiss::IDSelector* __restrict sel = (params == nullptr) ? nullptr : params->sel;

        // If `sel` is a knowhere BitsetViewIDSelector wrapping an empty bitset, BitsetView::test
        // returns true for every id (out_id >= num_bits_=0 short-circuits), so is_member returns
        // false for every id and the BF range_search would emit zero results. Fall back to
        // IDSelectorAll in that case, mirroring the guard already present in
        // IndexBruteForceWrapper::search above.
        const knowhere::BitsetViewIDSelector* __restrict bw_idselector =
            dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel);
        const bool sel_accepts_all = (sel == nullptr) || (bw_idselector && bw_idselector->bitset_view.empty());

        if (is_similarity_metric(index->metric_type)) {
            typename RH_max::SingleResultHandler res_max(bres_max);
            res_max.begin(i);

            if (sel_accepts_all) {
                // Compiler is expected to de-virtualize virtual method calls
                faiss::IDSelectorAll sel_all;

                faiss::cppcontrib::knowhere::brute_force_range_search_impl<
                    typename RH_max::SingleResultHandler, faiss::DistanceComputer, faiss::IDSelectorAll>(
                    index->ntotal, *dis, sel_all, res_max);
            } else {
                faiss::cppcontrib::knowhere::brute_force_range_search_impl<typename RH_max::SingleResultHandler,
                                                                           faiss::DistanceComputer, faiss::IDSelector>(
                    index->ntotal, *dis, *sel, res_max);
            }

            res_max.end();
        } else {
            typename RH_min::SingleResultHandler res_min(bres_min);
            res_min.begin(i);

            if (sel_accepts_all) {
                // Compiler is expected to de-virtualize virtual method calls
                faiss::IDSelectorAll sel_all;

                faiss::cppcontrib::knowhere::brute_force_range_search_impl<
                    typename RH_min::SingleResultHandler, faiss::DistanceComputer, faiss::IDSelectorAll>(
                    index->ntotal, *dis, sel_all, res_min);
            } else {
                faiss::cppcontrib::knowhere::brute_force_range_search_impl<typename RH_min::SingleResultHandler,
                                                                           faiss::DistanceComputer, faiss::IDSelector>(
                    index->ntotal, *dis, *sel, res_min);
            }

            res_min.end();
        }
    }
}

}  // namespace knowhere
