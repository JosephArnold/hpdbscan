#include <iostream>
#include <cassert>

#include "mpi_util.h"

int main()
{

    assert(MPI_INT8_T   == get_mpi_type<std::int8_t>()   && "wrong type returned for std::int8_t");
    assert(MPI_INT16_T  == get_mpi_type<std::int16_t>()  && "wrong type returned for std::int16_t");
    assert(MPI_INT32_T  == get_mpi_type<std::int32_t>()  && "wrong type returned for std::int32_t");
    assert(MPI_INT64_T  == get_mpi_type<std::int64_t>()  && "wrong type returned for std::int64_t");
    assert(MPI_UINT8_T  == get_mpi_type<std::uint8_t>()  && "wrong type returned for std::uint8_t");
    assert(MPI_UINT16_T == get_mpi_type<std::uint16_t>() && "wrong type returned for std::uint16_t");
    assert(MPI_UINT32_T == get_mpi_type<std::uint32_t>() && "wrong type returned for std::uint32_t");
    assert(MPI_UINT64_T == get_mpi_type<std::uint64_t>() && "wrong type returned for std::uint64_t");

    assert(MPI_INT32_T  == get_mpi_type<int>()      && "wrong type returned for int");
    assert(MPI_INT64_T == get_mpi_type<long>()     && "wrong type returned for long");

    assert(MPI_INT8_T   == get_mpi_type<int8_t>()   && "wrong type returned for int8_t");
    assert(MPI_INT16_T  == get_mpi_type<int16_t>()  && "wrong type returned for int16_t");
    assert(MPI_INT32_T  == get_mpi_type<int32_t>()  && "wrong type returned for int32_t");
    assert(MPI_INT64_T  == get_mpi_type<int64_t>()  && "wrong type returned for int64_t");
    assert(MPI_UINT8_T  == get_mpi_type<uint8_t>()  && "wrong type returned for uint8_t");
    assert(MPI_UINT16_T == get_mpi_type<uint16_t>() && "wrong type returned for uint16_t");
    assert(MPI_UINT32_T == get_mpi_type<uint32_t>() && "wrong type returned for uint32_t");
    assert(MPI_UINT64_T == get_mpi_type<uint64_t>() && "wrong type returned for uint64_t");
    assert(MPI_FLOAT    == get_mpi_type<float>()    && "wrong type returned for float");
    assert(MPI_DOUBLE   == get_mpi_type<double>()   && "wrong type returned for double");


    return 0;
}
