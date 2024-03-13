%module hpdbscan

%{
    #define SWIG_FILE_WITH_INIT
    #include <cstdint>
    #include "hpdbscan.h"

    #ifdef WITH_MPI
    #include <mpi.h>
    #endif
%}

%include "typemaps.i"
%include "stdint.i"
%include "std_string.i"
%include "std_vector.i"
%include "numpy.i"

namespace std {
    %template(ClusterVector) vector<ptrdiff_t>;
}

%init %{
    import_array();
%}

%apply(double* IN_ARRAY2, int DIM1, int DIM2){(double* data, int dim0, int dim1)};

 class HPDBSCAN {
 public:
     HPDBSCAN(float epsilon, std::size_t min_points);
     template<typename index_type>
     std::vector<index_type> cluster(const std::string& path,
                                     const std::string& dataset);
     template<typename index_type>
     std::vector<index_type> cluster(const std::string& path,
                                     const std::string& dataset,
                                     int threads);
     template<typename data_type, typename index_type>
     std::vector<index_type> cluster(data_type* data, int dim0, int dim1);
 };

%extend HPDBSCAN {
    %template(cluster64) cluster<ptrdiff_t>;
    %template(cluster64FP64) cluster<double, ptrdiff_t>;
    %template(cluster64FP32) cluster<float, ptrdiff_t>;
}
