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

#ifndef BITSET_H
#define BITSET_H

#include <cassert>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>

namespace knowhere {

// Iterator over valid (non-filtered) doc IDs in a BitsetView.
// Efficiently skips over filtered-out ranges using word-level operations.
// Used for zipper merge in sparse inverted index search.
class ValidDocIdIterator {
 public:
    ValidDocIdIterator() = default;

    ValidDocIdIterator(const uint8_t* bits, size_t num_bits, const uint32_t* out_ids = nullptr,
                         size_t num_internal_ids = 0, size_t id_offset = 0)
        : bits_(bits),
          num_bits_(num_bits),
          out_ids_(out_ids),
          num_internal_ids_(num_internal_ids),
          id_offset_(id_offset) {
        if (out_ids_ != nullptr) {
            max_id_ = num_internal_ids_;
        } else {
            max_id_ = num_bits_;
        }
        // Initialize to first passing doc
        if (max_id_ > 0) {
            advance_to_ge(0);
        } else {
            current_ = max_id_;  // exhausted
        }
    }

    // Returns true if there are more passing doc IDs
    [[nodiscard]] bool
    has_next() const {
        return current_ < max_id_;
    }

    // Returns the current passing doc ID (undefined if !has_next())
    [[nodiscard]] size_t
    current() const {
        return current_;
    }

    // Advance to the next passing doc ID
    void
    next() {
        if (current_ < max_id_) {
            advance_to_ge(current_ + 1);
        }
    }

    // Advance to the first passing doc ID >= target.
    void
    advance_to_ge(size_t target) {
        if (target >= max_id_) {
            current_ = max_id_;
            return;
        }

        current_ = target;

        // Use word-level operations for efficiency when no id mapping
        if (out_ids_ == nullptr && bits_ != nullptr) {
            advance_to_ge_optimized();
        } else {
            // Fallback for id-mapped case: linear scan with test()
            advance_to_ge_linear();
        }
    }

 private:
    // Test if a doc ID is filtered out (should be skipped)
    [[nodiscard]] bool
    is_filtered(size_t index) const {
        int64_t out_id = static_cast<int64_t>(index) + static_cast<int64_t>(id_offset_);
        if (out_ids_ != nullptr) {
            out_id = out_ids_[out_id];
        }
        if (out_id >= static_cast<int64_t>(num_bits_)) {
            return true;
        }
        return (bits_[out_id >> 3] & (0x1 << (out_id & 0x7))) != 0;
    }

    // Linear scan fallback (used when id mapping is present)
    void
    advance_to_ge_linear() {
        while (current_ < max_id_ && is_filtered(current_)) {
            ++current_;
        }
    }

    // Optimized word-level scan (no id mapping case)
    // Skips 64-bit chunks where all bits are set (all filtered)
    void
    advance_to_ge_optimized() {
        // First, check the current position
        size_t pos = current_ + id_offset_;
        if (pos >= num_bits_) {
            current_ = max_id_;
            return;
        }

        // Check if current position passes
        if ((bits_[pos >> 3] & (0x1 << (pos & 0x7))) == 0) {
            return;  // Found a passing doc at current_
        }

        // Move to next position
        ++pos;
        if (pos >= num_bits_) {
            current_ = max_id_;
            return;
        }

        // Align to 64-bit boundary for fast scanning
        size_t word_idx = pos >> 6;  // pos / 64
        size_t bit_in_word = pos & 63;
        size_t num_words = (num_bits_ + 63) >> 6;

        const uint64_t* words = reinterpret_cast<const uint64_t*>(bits_);

        // Check remaining bits in current word
        if (bit_in_word != 0 && word_idx < num_words) {
            uint64_t word = words[word_idx];
            // Mask out bits before our position
            uint64_t mask = ~((1ULL << bit_in_word) - 1);
            uint64_t masked = ~word & mask;  // Invert: 1 = passing, then mask

            if (masked != 0) {
                size_t offset = __builtin_ctzll(masked);
                size_t found = (word_idx << 6) + offset;
                if (found < num_bits_) {
                    current_ = found - id_offset_;
                    if (current_ < max_id_) {
                        return;
                    }
                }
            }
            ++word_idx;
        }

        // Scan full words
        while (word_idx < num_words) {
            uint64_t word = words[word_idx];
            if (word != 0xFFFFFFFFFFFFFFFFULL) {
                // There's at least one unset bit
                uint64_t inverted = ~word;
                size_t offset = __builtin_ctzll(inverted);
                size_t found = (word_idx << 6) + offset;
                if (found < num_bits_) {
                    current_ = found - id_offset_;
                    if (current_ < max_id_) {
                        return;
                    }
                }
            }
            ++word_idx;
        }

        // No more valid docs found
        current_ = max_id_;
    }

    const uint8_t* bits_ = nullptr;
    size_t num_bits_ = 0;
    const uint32_t* out_ids_ = nullptr;
    size_t num_internal_ids_ = 0;
    size_t id_offset_ = 0;
    size_t max_id_ = 0;
    size_t current_ = 0;
};

class BitsetView {
 public:
    BitsetView() = default;
    ~BitsetView() = default;

    BitsetView(const uint8_t* data, size_t num_bits, size_t num_filtered_out_bits = 0, size_t id_offset = 0)
        : bits_(data), num_bits_(num_bits), num_filtered_out_bits_(num_filtered_out_bits), id_offset_(id_offset) {
    }

    BitsetView(const std::nullptr_t) : BitsetView() {
    }

    bool
    empty() const {
        return num_bits_ == 0;
    }

    // return the number of the bits. if with id mapping, return the number of the internal ids.
    size_t
    size() const {
        if (out_ids_ != nullptr) {
            return num_internal_ids_;
        }
        return num_bits_;
    }

    // return the number of filtered out bits. if with id mapping, return the number of filtered out ids.
    size_t
    count() const {
        if (out_ids_ != nullptr) {
            return num_filtered_out_ids_;
        }
        return num_filtered_out_bits_;
    }

    size_t
    byte_size() const {
        return (num_bits_ + 8 - 1) >> 3;
    }

    const uint8_t*
    data() const {
        return bits_;
    }

    bool
    has_out_ids() const {
        return out_ids_ != nullptr;
    }

    void
    set_out_ids(const uint32_t* out_ids, size_t num_internal_ids,
                std::optional<size_t> num_filtered_out_ids = std::nullopt) {
        out_ids_ = out_ids;
        num_internal_ids_ = num_internal_ids;
        if (num_filtered_out_ids.has_value()) {
            num_filtered_out_ids_ = num_filtered_out_ids.value();
        } else {
            // auto calculate num_filtered_out_ids if not provided
            num_filtered_out_ids_ = get_filtered_out_num_();
        }
    }

    const uint32_t*
    out_ids_data() const {
        if (out_ids_ == nullptr) {
            return nullptr;
        }
        return out_ids_;
    }

    void
    set_id_offset(size_t id_offset) {
        id_offset_ = id_offset;
    }

    // if the test succeeds, then the index should be skipped during search; otherwise, it should be included.
    bool
    test(int64_t index) const {
        int64_t out_id = index + id_offset_;
        if (out_ids_ != nullptr) {
            out_id = out_ids_[out_id];
        }
        // when index is larger than the max_offset, ignore it
        return (out_id >= static_cast<int64_t>(num_bits_)) || (bits_[out_id >> 3] & (0x1 << (out_id & 0x7)));
    }
    // return the filtered ratio. if with id mapping, calculated by internal_ids rather than bits.
    float
    filter_ratio() const {
        return empty() ? 0.0f : ((float)count() / size());
    }

    size_t
    get_filtered_out_num_() const {
        if (empty()) {
            return 0;
        }
        if (out_ids_ != nullptr) {
            // if with id mapping, there is no optimization for the traversal.
            size_t count = 0;
            for (size_t i = 0; i < num_internal_ids_; i++) {
                if (test(i)) {
                    count++;
                }
            }
            return count;
        }
        // if without id mapping, use a better algorithm to calculate the number of filtered out bits.
        size_t ret = 0;
        auto len_uint8 = byte_size();
        auto len_uint64 = len_uint8 >> 3;

        auto popcount8 = [&](uint8_t x) -> int {
            x = (x & 0x55) + ((x >> 1) & 0x55);
            x = (x & 0x33) + ((x >> 2) & 0x33);
            x = (x & 0x0F) + ((x >> 4) & 0x0F);
            return x;
        };

        uint64_t* p_uint64 = (uint64_t*)bits_;
        for (size_t i = 0; i < len_uint64; i++) {
            ret += __builtin_popcountll(*p_uint64);
            p_uint64++;
        }

        // calculate remainder
        uint8_t* p_uint8 = (uint8_t*)bits_ + (len_uint64 << 3);
        for (size_t i = (len_uint64 << 3); i < len_uint8; i++) {
            ret += popcount8(*p_uint8);
            p_uint8++;
        }

        return ret;
    }

    // return the first valid idx. if with id mapping, return the first valid internal_id.
    size_t
    get_first_valid_index() const {
        if (out_ids_ != nullptr) {
            // if with id mapping, there is no optimization for the traversal.
            for (size_t i = 0; i < num_internal_ids_; i++) {
                if (!test(i)) {
                    return i;
                }
            }
            return num_internal_ids_;
        }
        // if without id mapping, use a better algorithm to find the first valid index.
        size_t ret = 0;
        auto len_uint8 = byte_size();
        auto len_uint64 = len_uint8 >> 3;

        uint64_t* p_uint64 = (uint64_t*)bits_;
        for (size_t i = 0; i < len_uint64; i++) {
            uint64_t value = (~(*p_uint64));
            if (value == 0) {
                p_uint64++;
                continue;
            }
            ret = __builtin_ctzll(value);
            return i * 64 + ret;
        }

        // calculate remainder
        uint8_t* p_uint8 = (uint8_t*)bits_ + (len_uint64 << 3);
        for (size_t i = 0; i < len_uint8 - (len_uint64 << 3); i++) {
            uint8_t value = (~(*p_uint8));
            if (value == 0) {
                p_uint8++;
                continue;
            }
            ret = __builtin_ctz(value);
            return len_uint64 * 64 + i * 8 + ret;
        }

        return num_bits_;
    }

    std::string
    to_string(size_t from, size_t to) const {
        if (empty()) {
            return "";
        }
        std::stringbuf buf;
        to = std::min<size_t>(to, num_bits_);
        for (size_t i = from; i < to; i++) {
            buf.sputc(test(i) ? '1' : '0');
        }
        return buf.str();
    }


 private:
    const uint8_t* bits_ = nullptr;
    size_t num_bits_ = 0;
    size_t num_filtered_out_bits_ = 0;

    // optional. many indexes will share one bitset, requiring offset to distinguish between them.
    //  like multi-chunk brute-force in /src/common/comp/brute_force.cc, or mv-only in /src/index/hnsw/faiss_hnsw.cc
    size_t id_offset_ = 0;  // offset of the internal ids

    // optional. bitset supports id mapping.
    // Even allows multiple ids to map to the same bit, so the number of internal ids and bits may be not equal.
    const uint32_t* out_ids_ = nullptr;
    size_t num_internal_ids_ = 0;
    size_t num_filtered_out_ids_ = 0;
};
}  // namespace knowhere

#endif /* BITSET_H */
