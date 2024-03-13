/*
* Copyright (c) 2018
* Markus Goetz
*
* This software may be modified and distributed under the terms of MIT-style license.
*
* Description: Highly parallel DBSCAN algorithm implementation
*
* Maintainer: m.goetz
*
* Email: markus.goetz@kit.edu
*/

#include <cstdint>

#include "hpdbscan.h"

// explicit template instantiation
template Clusters<std::int16_t> HPDBSCAN<std::int16_t>::cluster<std::uint8_t >(Dataset<std::uint8_t >&, int);
template Clusters<std::int16_t> HPDBSCAN<std::int16_t>::cluster<std::uint16_t>(Dataset<std::uint16_t>&, int);
template Clusters<std::int16_t> HPDBSCAN<std::int16_t>::cluster<std::uint32_t>(Dataset<std::uint32_t>&, int);
template Clusters<std::int16_t> HPDBSCAN<std::int16_t>::cluster<std::uint64_t>(Dataset<std::uint64_t>&, int);

template Clusters<std::int16_t> HPDBSCAN<std::int16_t>::cluster<std::int8_t >(Dataset<std::int8_t >&, int);
template Clusters<std::int16_t> HPDBSCAN<std::int16_t>::cluster<std::int16_t>(Dataset<std::int16_t>&, int);
template Clusters<std::int16_t> HPDBSCAN<std::int16_t>::cluster<std::int32_t>(Dataset<std::int32_t>&, int);
template Clusters<std::int16_t> HPDBSCAN<std::int16_t>::cluster<std::int64_t>(Dataset<std::int64_t>&, int);

template Clusters<std::int16_t> HPDBSCAN<std::int16_t>::cluster<float >(Dataset<float >&, int);
template Clusters<std::int16_t> HPDBSCAN<std::int16_t>::cluster<double>(Dataset<double>&, int);

template Clusters<std::int16_t> HPDBSCAN<std::int16_t>::cluster(const std::string& path, const std::string& dataset);
template Clusters<std::int16_t> HPDBSCAN<std::int16_t>::cluster(const std::string& path, const std::string& dataset, int threads);

template Clusters<std::int32_t> HPDBSCAN<std::int32_t>::cluster<std::uint8_t >(Dataset<std::uint8_t >&, int);
template Clusters<std::int32_t> HPDBSCAN<std::int32_t>::cluster<std::uint16_t>(Dataset<std::uint16_t>&, int);
template Clusters<std::int32_t> HPDBSCAN<std::int32_t>::cluster<std::uint32_t>(Dataset<std::uint32_t>&, int);
template Clusters<std::int32_t> HPDBSCAN<std::int32_t>::cluster<std::uint64_t>(Dataset<std::uint64_t>&, int);

template Clusters<std::int32_t> HPDBSCAN<std::int32_t>::cluster<std::int8_t >(Dataset<std::int8_t >&, int);
template Clusters<std::int32_t> HPDBSCAN<std::int32_t>::cluster<std::int16_t>(Dataset<std::int16_t>&, int);
template Clusters<std::int32_t> HPDBSCAN<std::int32_t>::cluster<std::int32_t>(Dataset<std::int32_t>&, int);
template Clusters<std::int32_t> HPDBSCAN<std::int32_t>::cluster<std::int64_t>(Dataset<std::int64_t>&, int);

template Clusters<std::int32_t> HPDBSCAN<std::int32_t>::cluster<float >(Dataset<float >&, int);
template Clusters<std::int32_t> HPDBSCAN<std::int32_t>::cluster<double>(Dataset<double>&, int);

template Clusters<std::int32_t> HPDBSCAN<std::int32_t>::cluster(const std::string& path, const std::string& dataset);
template Clusters<std::int32_t> HPDBSCAN<std::int32_t>::cluster(const std::string& path, const std::string& dataset, int threads);

template Clusters<std::int64_t> HPDBSCAN<std::int64_t>::cluster<std::uint8_t >(Dataset<std::uint8_t >&, int);
template Clusters<std::int64_t> HPDBSCAN<std::int64_t>::cluster<std::uint16_t>(Dataset<std::uint16_t>&, int);
template Clusters<std::int64_t> HPDBSCAN<std::int64_t>::cluster<std::uint32_t>(Dataset<std::uint32_t>&, int);
template Clusters<std::int64_t> HPDBSCAN<std::int64_t>::cluster<std::uint64_t>(Dataset<std::uint64_t>&, int);

template Clusters<std::int64_t> HPDBSCAN<std::int64_t>::cluster<std::int8_t >(Dataset<std::int8_t >&, int);
template Clusters<std::int64_t> HPDBSCAN<std::int64_t>::cluster<std::int16_t>(Dataset<std::int16_t>&, int);
template Clusters<std::int64_t> HPDBSCAN<std::int64_t>::cluster<std::int32_t>(Dataset<std::int32_t>&, int);
template Clusters<std::int64_t> HPDBSCAN<std::int64_t>::cluster<std::int64_t>(Dataset<std::int64_t>&, int);

template Clusters<std::int64_t> HPDBSCAN<std::int64_t>::cluster<float >(Dataset<float >&, int);
template Clusters<std::int64_t> HPDBSCAN<std::int64_t>::cluster<double>(Dataset<double>&, int);

template Clusters<std::int64_t> HPDBSCAN<std::int64_t>::cluster(const std::string& path, const std::string& dataset);
template Clusters<std::int64_t> HPDBSCAN<std::int64_t>::cluster(const std::string& path, const std::string& dataset, int threads);
