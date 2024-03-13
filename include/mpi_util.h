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

#ifndef MPI_UTIL_H
#define MPI_UTIL_H

#include <cstdint>
#include <type_traits>

#include <mpi.h>

// Mpi Type Case
#define MTC(type, mpi_type) \
    if(std::is_same_v<type, index_type>) \
    { \
        return mpi_type; \
    } 
#define MTCe(type, mpi_type) \
    else if(std::is_same_v<type, index_type>) \
    { \
        return mpi_type; \
    } 

    template<typename index_type>
    MPI_Datatype get_mpi_type()
    {
        MTC(float, MPI_FLOAT)
        MTCe(double, MPI_DOUBLE)
        MTCe(std::int8_t, MPI_INT8_T)
        MTCe(std::int16_t, MPI_INT16_T)
        MTCe(std::int32_t, MPI_INT32_T)
        MTCe(std::int64_t, MPI_INT64_T)
        MTCe(std::uint8_t, MPI_UINT8_T)
        MTCe(std::uint16_t, MPI_UINT16_T)
        MTCe(std::uint32_t, MPI_UINT32_T)
        MTCe(std::uint64_t, MPI_UINT64_T)

        throw std::invalid_argument("No MPI type mapped to specified type");
    }

// Don't pollute compiler with macros
#undef MTCe
#undef MTC

#if defined(WITH_OUTPUT)
//#include <vector>
//#include <cstddef>
//#include <iostream>
//
//inline void print_atoav_params(std::string label, std::vector<int> counts, std::vector<int> displs, std::size_t ranks, std::size_t this_rank)
//{
//    std::cout << label << " (rank " << this_rank << "):\n";
//    std::cout << "Counts: {";
//    for(std::size_t rank = 0; rank< ranks; rank++)
//    {
//        std::cout << counts[rank] << ", ";
//    }
//    std::cout << "}\n";
//    std::cout << "Displs: {";
//    for(std::size_t rank = 0; rank< ranks; rank++)
//    {
//        std::cout << displs[rank] << ", ";
//    }
//    std::cout << "}\n";
//}
#endif

#endif // MPI_UTIL_H
