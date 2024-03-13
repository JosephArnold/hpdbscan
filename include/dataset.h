/*
* Copyright (c) 2018
* Markus Goetz
*
* This software may be modified and distributed under the terms of MIT-style license.
*
* Description: Dataset abstraction
*
* Maintainer: m.goetz
*
* Email: markus.goetz@kit.edu
*/

#ifndef DATASET_H
#define DATASET_H

#include <cstdlib>
#include <algorithm>
#include <vector>

#include <hdf5.h>
#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "dynamic_aligned_allocator.hpp"
#include "constants.h"
#include "hdf5_util.h"

template<typename data_type>
struct Dataset {
    hsize_t m_shape[2];
    hsize_t m_chunk[2];
    hsize_t m_offset[2] = {0, 0};
    std::vector<data_type, dynamic_aligned_allocator<data_type>> m_elements;

    Dataset(const hsize_t shape[2])
        : m_elements(shape[0]*shape[1], 0, dynamic_aligned_allocator<data_type>(64))
    {
        std::copy(shape, shape + 2, m_shape);
        std::copy(shape, shape + 2, m_chunk);

        //m_p = std::aligned_alloc(64,shape[0] * shape[1] * H5Tget_precision(type) / BITS_PER_BYTE);
    }

    Dataset(data_type* data, const hsize_t shape[2]) 
        : Dataset(shape)
    {
        std::copy(data, data + shape[0] * shape[1], m_elements.begin());

        #ifdef WITH_MPI
        // determine the global shape...
        MPI_Allreduce(MPI_IN_PLACE, m_shape, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        // ... and the chunk offset
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Exscan(m_chunk, m_offset, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0) m_offset[0] = 0;
        #endif
    }

    ~Dataset() {
        //H5Tclose(m_type);
    }
};

#endif // DATASET_H
