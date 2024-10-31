/*For index_type=int64_t */
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/count.h>
#include <iostream>
#include <cmath>
#include <limits>
#include <arm_sve.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <hdf5.h>
#include <limits>
#include <numeric>
#include <omp.h>
#include <parallel/algorithm>
#include <vector>
#include <set>
#define IMPLEMENTING_OPTIMIZATION

void copyDatatoDevice(const float*, size_t, 
		  thrust::device_vector<float>&);

void copyDatatoDevice(const int32_t*, size_t,
                  thrust::device_vector<int32_t>&);


int32_t compute_distance(const int32_t index,
                       const size_t n, const float EPS2,
                       const std::vector<int32_t>& neighboring_points,
                      const float* dataset, size_t dimensions,
                      std::vector<int32_t>& min_points_area,
                      int32_t* clusters,
                      int32_t& cluster_label
                      );

void compute_neighbours(const int32_t*,
                                const int32_t*,
                                int32_t*,
                                const size_t,
                                const float,
                                int32_t*,
                                int32_t*,
				int32_t*,
                                const float*,
                                const size_t,
				const size_t,
				const size_t,
				int32_t, cudaStream_t*);

template <typename data_type, typename index_type>
class RegionQuery {

    
    thrust::device_vector<data_type> thrust_dataset;
    
    const data_type* dataset;
    const float EPS2;
    const size_t dimensions;
    index_type* clusters;
    const size_t m_min_points;
    const int32_t m_global_point_offset;
    const size_t dataset_size;

    public:
    RegionQuery(const data_type* data, const float EPS2, const size_t dimensions, 
		index_type* clusters, 
		const size_t m_min_points,
                const int32_t m_global_point_offset, size_t dataset_size) : EPS2(EPS2), 
	          dimensions(dimensions), dataset(data), 
	          clusters(clusters), m_min_points(m_min_points),
                  m_global_point_offset(m_global_point_offset),
	          dataset_size(dataset_size){


    }


     void compute_neighbours_gpu(const std::vector<index_type>& pts_common_nb,
                   index_type* neighboring_points,
		   size_t n,
                   std::vector<index_type>& min_points_area,
                   std::vector<index_type>& count,
		   std::vector< Cluster<index_type>>& cluster_labels,
		   size_t num_of_Points_with_common_nb, cudaStream_t* stream
		   ) {}
#if 0
     std::vector<index_type> get_neighbors(const Cell cell, size_t* n){

	      // allocate some space for the neighboring cells, be pessimistic and reserve 3^dims for possibly all neighbors
          int32_t* neighboring_cells = new int32_t[int(std::pow(3, dimensions))];
	   //cudaMalloc(&neighboring_cells, std::pow(3, dimensions) * sizeof(size_t));
	  int32_t idx = 0;
        //neighboring_cells.reserve(std::pow(3, dimensions));
          neighboring_cells[idx++] = cell;

          size_t cells_in_lower_space = 1;
          size_t cells_in_current_space = 1;
	  size_t number_of_points = 0;
          for (int i = 0; i < m_cells_size; i++) {
              if (m_cell_keys[i] == cell) {
                  number_of_points = m_cell_values_second[i];
                  break;
              }
          }

	  int32_t first, second;
          // fetch all existing neighboring cells
          for (size_t d = 0; d < dimensions; d++) {
          
	      cells_in_current_space *= m_cell_dimensions[m_swapped_dimensions[d]];

              for (size_t i = 0, end = idx; i < idx; ++i) {
              
	          const Cell current_cell = neighboring_cells[i];

                  // check "left" neighbor - a.k.a the cell in the current dimension that has a lower number
                  const Cell left = current_cell - cells_in_lower_space;
                  //const auto found_left = m_cell_index.find(left);
		  const int32_t found_left = find(left, &first, &second);
                  if (current_cell % cells_in_current_space >= cells_in_lower_space) {
                      neighboring_cells[idx++] = left;
                      number_of_points += found_left != -1 ? second : 0;
                  }
                // check "right" neighbor - a.k.a the cell in the current dimension that has a higher number
                  const Cell right = current_cell + cells_in_lower_space;
                  const auto found_right = find(right,  &first, &second);
                  if (current_cell % cells_in_current_space < cells_in_current_space - cells_in_lower_space) {
                      neighboring_cells[idx++] = right;
                      number_of_points += found_right != -1 ? second : 0;
                  }
               }
               cells_in_lower_space = cells_in_current_space;
          }

          // copy the points from the neighboring cells over
          std::vector<index_type> neighboring_points;
          //neighboring_points = new index_type[number_of_points];

	  *n = 0;
          for (int32_t i = 0; i < idx; i++) {
              const int32_t found = find(neighboring_cells[i], &first, &second);
              // skip empty cells
              if (found == -1) {
                  continue;
              }
              // ... otherwise copy the points over
              //const std::pair<size_t, size_t>& locator = found->second;
	      for(int32_t j = 0; j < second;j++) {

	        //  neighboring_points[*n++] = first + j;

	      }
           }

           return neighboring_points;

       }
#endif
     
     };

template <>
class RegionQuery<float, int32_t> {

    float* d_dataset;
    thrust::device_vector<int32_t> t_clusters;

    const float* dataset;
    const float EPS2;
    const size_t dimensions;
    int32_t* clusters;
    const size_t m_min_points;
    const int32_t m_global_point_offset;
    const size_t dataset_size;

    public:
    RegionQuery(const float* data, const float EPS2, const size_t dimensions,
                int32_t* clusters,
                const size_t m_min_points,
                const int32_t m_global_point_offset,
		const size_t dataset_size) : EPS2(EPS2),
                  dimensions(dimensions), dataset(data),
                  clusters(clusters), m_min_points(m_min_points),
                  m_global_point_offset(m_global_point_offset),
                  dataset_size(dataset_size){
                

    }


    void compute_neighbours_gpu(const std::vector<int32_t>& pts_common_nb,
                   const int32_t* neighboring_points,
		   const size_t n,
                   std::vector<int32_t>& min_points_area,
                   std::vector<int32_t>& count,
		   std::vector<Cluster<int32_t>>& cluster_labels,
                   const size_t num_of_Points_with_common_nb,
		   cudaStream_t* stream
                   ) {

         //const std::size_t n = neighboring_points.size();

	 //keep all the GPU code in the cuda file 
	 compute_neighbours(pts_common_nb.data(),
                                neighboring_points,
                                clusters,
                                n,
                                EPS2,
                                min_points_area.data(),
                                count.data(),
                                cluster_labels.data(),
                                dataset,
				num_of_Points_with_common_nb,
                                dimensions,
				m_min_points,
				m_global_point_offset, stream);

    }
#if 0
      inline int32_t find(Cell cell, int32_t* first, int32_t* second){
        for (int i = 0; i < m_cells_size; i++) {
            if (m_cell_keys[i] == cell) {
                *first = m_cell_values_first[i];
                *second = m_cell_values_second[i];
                return 1;
            }
        }
        return -1;
     }

      std::vector<int32_t> get_neighbors(const Cell cell, size_t* n) {

	  //std::cout<<"Inside get_neighbours function"<<std::endl;
              // allocate some space for the neighboring cells, be pessimistic and reserve 3^dims for possibly all neighbors
          //int32_t* neighboring_cells = new int32_t[int(std::pow(3, dimensions))];
	  Cells neighboring_cells;
          neighboring_cells.reserve(std::pow(3, dimensions));
          neighboring_cells.push_back(cell);
           //cudaMalloc(&neighboring_cells, std::pow(3, dimensions) * sizeof(size_t));
          //int32_t idx = 0;
          //neighboring_cells[idx++] = cell;

          size_t cells_in_lower_space = 1;
          size_t cells_in_current_space = 1;
          size_t number_of_points = 0;

	  for (uint32_t i = 0; i < m_cells_size; i++) {
              if (m_cell_keys[i] == cell) {
                  number_of_points = m_cell_values_second[i];
                  break;
              }
          }
	  
          int32_t first, second;
          // fetch all existing neighboring cells
	   for (size_t d = 0; d < dimensions; d++) {

              cells_in_current_space *= m_cell_dimensions[m_swapped_dimensions[d]];

              for (size_t i = 0, end =  neighboring_cells.size(); i < end; ++i) {

                  const Cell current_cell = neighboring_cells[i];

                  // check "left" neighbor - a.k.a the cell in the current dimension that has a lower number
                  const Cell left = current_cell - cells_in_lower_space;
                  //const auto found_left = m_cell_index.find(left);
                  const int32_t found_left = find(left, &first, &second);
                  if (current_cell % cells_in_current_space >= cells_in_lower_space) {
                      neighboring_cells.push_back(left);
                      number_of_points += found_left !=  -1 ? second : 0;
                  }
                // check "right" neighbor - a.k.a the cell in the current dimension that has a higher number
                  const Cell right = current_cell + cells_in_lower_space;
                  const auto found_right = find(right,  &first, &second);
		  //const auto found_right = m_cell_index.find(right);
                  if (current_cell % cells_in_current_space < cells_in_current_space - cells_in_lower_space) {
                      neighboring_cells.push_back(right);
                      number_of_points += found_right != -1 ? second : 0;
                  }
               }
               cells_in_lower_space = cells_in_current_space;
          }
	   // copy the points from the neighboring cells over
	  std::vector<int32_t> neighboring_points;
          //neighboring_points = new int32_t[number_of_points];
	  neighboring_points.reserve(number_of_points);
          *n = 0;
          for (int32_t i = 0; i < neighboring_cells.size(); i++) {
              const int32_t found = find(neighboring_cells[i], &first, &second);
              // skip empty cells
              if (found == -1) {
                  continue;
              }
              // ... otherwise copy the points over
	      neighboring_points.resize(neighboring_points.size() + second);
              //std::iota(neighboring_points.end() - second, neighboring_points.end(), first);
              
	      for(int32_t j = 0; j < second;j++) {

                  neighboring_points[*n++] = first + j;

              }
	      
           }
           //*n = neighboring_points.size();
           return neighboring_points;

       }
#endif
};
/*
template<>
void RegionQuery<float, std::int32_t>::template compute_neighbours_gpu(const std::vector<int32_t>& pts_common_nb,
                   const std::vector<int32_t>& neighboring_points,
                   std::vector<std::vector<int32_t>>& min_points_area,
                   std::vector<int32_t>& count,
		   std::vector<Cluster<int32_t>>& cluster_labels,
		   const size_t num_of_Points_with_common_nb
		   ) {


	std::size_t n = neighboring_points.size();

	for(size_t i = 0; i < num_of_Points_with_common_nb; i++) {

	    int32_t point_index = pts_common_nb[i];

	    min_points_area[i] = std::vector<int32_t>(n, NOT_VISITED<int32_t>);

	    uint32_t index = point_index * dimensions;

	    cluster_labels[i] = m_global_point_offset + point_index + 1;

	    count[i] =  compute_distance(index,
                                         n, EPS2,
                                         neighboring_points,
                                         dataset, dimensions,
                                         min_points_area[i],
                                         clusters,
                                         cluster_labels[i]);

	   if(count[i] >= m_min_points) {
	     
	        atomic_min(clusters + point_index, static_cast<Cluster<int32_t>>(-cluster_labels[i]));

		for(auto nb : min_points_area[i]) {

		    if(nb != NOT_VISITED<int32_t>) {
		        
		        atomic_min(clusters + nb, cluster_labels[i]);

		    }
		
		}

	    }
	    else if (clusters[point_index] == NOT_VISITED<int32_t>) {
                        // mark as noise
                atomic_min(clusters + point_index, NOISE<int32_t>);
            }

	}
}
*/
