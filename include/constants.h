/*
* Copyright (c) 2018
* Markus Goetz
*
* This software may be modified and distributed under the terms of MIT-style license.
*
* Description: Type definitions and constants for HPDBSCAN
*
* Maintainer: m.goetz
*
* Email: markus.goetz@kit.edu
*/

#ifndef CONSTANTS_H
#define	CONSTANTS_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include "dynamic_aligned_allocator.hpp"

using ssize_t       = ptrdiff_t;
using Locator       = std::pair<size_t, size_t>;
template <typename index_type>
using Clusters = typename std::vector<index_type>;
template <typename index_type>
using Cluster = index_type;
using Cell          = size_t;
using Cells         = std::vector<Cell>;
using CellHistogram = std::map<Cell, size_t>;
using CellIndex     = std::map<Cell, Locator>;
using CellBounds    = std::array<size_t, 4>;
using ComputeBounds = std::array<size_t, 2>;
using Cuts          = std::vector<Locator>;

constexpr size_t  BITS_PER_BYTE = 8;
template<typename index_type>
inline constexpr index_type NOT_VISITED = std::numeric_limits<index_type>::max();
template<typename index_type>
inline constexpr index_type NOISE = std::numeric_limits<index_type>::max() -1;

constexpr std::array RADIX_POWERS {
        1LLU,
        10LLU,
        100LLU,
        1000LLU,
        10000LLU,
        100000LLU,
        1000000LLU,
        10000000LLU,
        100000000LLU,
        10000000000LLU,
        100000000000LLU,
        1000000000000LLU,
        10000000000000LLU,
        100000000000000LLU,
        1000000000000000LLU,
        10000000000000000LLU,
        100000000000000000LLU,
        1000000000000000000LLU,
};
constexpr size_t RADIX_BUCKETS = 10; // radix sort buckets

#endif // CONSTANTS_H
