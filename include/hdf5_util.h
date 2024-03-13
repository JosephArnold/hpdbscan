/*
* Copyright (c) 2018
* Markus Goetz
*
* This software may be modified and distributed under the terms of MIT-style license.
*
* Description: MPI type selection utility
*
* Maintainer: m.goetz
*
* Email: markus.goetz@kit.edu
*/

#ifndef HDF5_UTIL_H
#define HDF5_UTIL_H

#include <cstdint>
#include <type_traits>

#include <hdf5.h>

// HDF5 Type Case
#define H5TC(type, h5_type) \
    if(std::is_same_v<type, index_type>) \
    { \
        return h5_type; \
    } 
#define H5TCe(type, h5_type) \
    else if(std::is_same_v<type, index_type>) \
    { \
        return h5_type; \
    } 

    template<typename index_type>
    inline constexpr hid_t get_hdf5_type()
    {
        H5TC(std::int8_t,   H5T_NATIVE_SCHAR)
        H5TCe(std::int16_t,  H5T_NATIVE_SHORT)
        H5TCe(std::int32_t,  H5T_NATIVE_INT)
        H5TCe(std::int64_t,  H5T_NATIVE_LONG)
        H5TCe(std::uint8_t,  H5T_NATIVE_UCHAR)
        H5TCe(std::uint16_t, H5T_NATIVE_USHORT)
        H5TCe(std::uint32_t, H5T_NATIVE_UINT)
        H5TCe(std::uint64_t, H5T_NATIVE_ULONG)
        H5TCe(float,    H5T_NATIVE_FLOAT)
        H5TCe(double,    H5T_NATIVE_DOUBLE)

        // gotta noexcept for constexpr, so just return int
        return H5T_NATIVE_INT;
    }

// Don't pollute compiler with macros
#undef H5TCe
#undef H5TC

#endif // MPI_UTIL_H
