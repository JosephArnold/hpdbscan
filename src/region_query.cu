#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include "atomic.h"
#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/count.h>

#define NOT_VISITED INT_MAX
#define NOISE INT_MAX - 1


void copyDatatoDevice(const float* dataset, size_t n, 
					  thrust::device_vector<float>& d_vector) {

    d_vector =  thrust::device_vector<float>(n);

    thrust::copy(dataset, dataset + n, d_vector.begin());

}

void copyDatatoDevice(const int32_t* dataset, size_t n,
                                          thrust::device_vector<int32_t>& d_vector) {

    d_vector =  thrust::device_vector<int32_t>(n);

    thrust::copy(dataset, dataset + n, d_vector.begin());

}


struct Point{

    int32_t index;
    float coordinates[3];

};


__device__ void compute_euclidean_dist(const int32_t index,
	                               const int32_t i,	
				       const int32_t* neighboring_points,
		                       const int32_t* clusters, 
				       const float EPS2,  
				       int32_t* min_points_area,
				       int32_t* count, 
				       const float* dataset,
				       const size_t dimensions,
		                       const size_t n,
				       int32_t* cluster_label) {

    for( uint32_t tid = 0; tid < n; tid++) {

	const int32_t nb_index = neighboring_points[tid];
        float offset = 0.0f;
                for (size_t d = 0; d < dimensions; ++d) {
                    float coord = dataset[index + d];
                    float ocoord = dataset[nb_index * dimensions + d];
                    float distance = coord - ocoord;
                    offset += distance * distance;
                }

                if (offset <= EPS2) {

                    atomicAdd(count, 1);

                    min_points_area[i * n + tid] = nb_index;

                    if (clusters[nb_index] < 0) {

                        atomicMin(cluster_label, abs(clusters[nb_index]));
                    }

                }
    }
    
}

__device__ void compute_euclidean_dist_hybrid(const int32_t index,
                                       const int32_t i,
				       const int32_t* neighboring_points,
				       const int32_t shared_elements_size,
				       const struct Point* s_neighboring_points,
                                       const int32_t* clusters,
                                       const float EPS2,
                                       int32_t* min_points_area,
                                       int32_t* count,
                                       const float* dataset,
                                       const size_t dimensions,
                                       const size_t n,
                                       int32_t* cluster_label) {

    
    for( uint32_t tid = 0; tid < shared_elements_size; tid++) {

        const int32_t nb_index = s_neighboring_points[tid].index;
        float offset = 0.0f;
                for (size_t d = 0; d < dimensions; ++d) {
                    float coord = dataset[index + d];
                    float ocoord =  s_neighboring_points[tid].coordinates[d];
                    float distance = coord - ocoord;
                    offset += distance * distance;
                }

                if (offset <= EPS2) {

                    atomicAdd(count, 1);

                    min_points_area[i * n + tid] = nb_index;

                    if (clusters[nb_index] < 0) {

                        atomicMin(cluster_label, abs(clusters[nb_index]));
                    }

                }
    }

    for( uint32_t tid = shared_elements_size; tid < n; tid++) {

        const int32_t nb_index = neighboring_points[tid];
        float offset = 0.0f;
                for (size_t d = 0; d < dimensions; ++d) {
                    float coord = dataset[index + d];
                    float ocoord =  dataset[nb_index * dimensions + d];
                    float distance = coord - ocoord;
                    offset += distance * distance;
                }

                if (offset <= EPS2) {

                    atomicAdd(count, 1);

                    min_points_area[i * n + tid] = nb_index;

                    if (clusters[nb_index] < 0) {

                        atomicMin(cluster_label, abs(clusters[nb_index]));
                    }

                }
    }

}

__global__ void compute_neighbours_cuda_hybrid(
                                const int32_t* pts_common_nb,
                                const int32_t* neighboring_points,
				const int32_t shared_size,
                                int32_t* clusters,
                                const size_t n,
                                const float EPS2,
                                int32_t* min_points_area,
                                int32_t* count,
                                int32_t* cluster_labels,
                                const float* dataset,
                                const size_t num_of_Points_with_common_nb,
                                const size_t dimensions,
                                const size_t m_min_points,
                                int32_t m_global_point_offset) {

         extern  __shared__ struct Point s_neighboring_points_data[];
         
	 for (int i = threadIdx.x; i < shared_size; i += blockDim.x)  {

                 const int32_t index = neighboring_points[i]  * dimensions;
             
	         s_neighboring_points_data[i].index =  neighboring_points[i];
             
	         for(int32_t d = 0; d < dimensions;d++) {
                     s_neighboring_points_data[i].coordinates[d] = dataset[index + d];
                 }

         }
	 
         __syncthreads();

         const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	 if(tid < num_of_Points_with_common_nb) {

              const int32_t point_index = pts_common_nb[tid];

	      const int32_t index = point_index * dimensions;

              const uint32_t limit  = tid * n + n;

              cluster_labels[tid] =  m_global_point_offset + point_index + 1;

              compute_euclidean_dist_hybrid(index,
                                   tid,
                                   neighboring_points,
				   shared_size,
                                   s_neighboring_points_data,
                                   clusters,
                                   EPS2,
                                   min_points_area,
                                   &count[tid],
                                   dataset,
                                   dimensions,
                                   n,
                                   &cluster_labels[tid]);

              if(count[tid] >= m_min_points) {

                  atomicMin(clusters + point_index, -cluster_labels[tid]);

                  for(uint32_t k  = tid * n; k < limit; k++) {

                      if(min_points_area[k] != NOT_VISITED) {

                          atomicMin(clusters + min_points_area[k], cluster_labels[tid]);

                      }

                  }

              }
              else if (clusters[point_index] == NOT_VISITED) {
                        // mark as noise
                  atomicMin(clusters + point_index, NOISE);
              }
        }

}

__global__ void compute_neighbours_cuda(
                                const int32_t* pts_common_nb,
                                const int32_t* neighboring_points,
				int32_t* clusters,
                                const size_t n,
                                const float EPS2,
                                int32_t* min_points_area,
				int32_t* cluster_labels,
                                int32_t* count,
                                const float* dataset,
                                const size_t num_of_Points_with_common_nb,
                                const size_t dimensions,
				int32_t m_global_point_offset
                                ) {


         const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	 extern __shared__ int min_cluster_label[];  // Shared memory for index of minimum distance

	 uint32_t pts_common_nb_id = tid / n;

	 int neighboring_points_id = tid % n;  // Thread within the block
   

	 if((pts_common_nb_id < num_of_Points_with_common_nb) && (neighboring_points_id < n)) {

              //if (threadIdx.x == 0) {
	      //    min_cluster_label[0] =  m_global_point_offset + pts_common_nb_id + 1;
	      //}

	       //__syncthreads();

	      const int32_t point_index = pts_common_nb[pts_common_nb_id];

              const int32_t index = point_index * dimensions;


              const int32_t nb_index = neighboring_points[neighboring_points_id] * dimensions;
        
	      float offset = 0.0f;
              
	      for (size_t d = 0; d < dimensions; ++d) {
                    float coord = dataset[index + d];
                    float ocoord = dataset[nb_index + d];
                    float distance = coord - ocoord;
                    offset += distance * distance;
               }

               if (offset <= EPS2) {

                    atomicAdd(&count[pts_common_nb_id], 1);

                    min_points_area[pts_common_nb_id * n + neighboring_points_id] = neighboring_points[neighboring_points_id];

		    //if (clusters[neighboring_points[neighboring_points_id]] < 0) {

                      //  atomicMin(&min_cluster_label[0], abs(clusters[neighboring_points[neighboring_points_id]]));

                    //}


               }

	        //__syncthreads();//Each thread has calculated the distance and min_cluster_label[0] 
		/*Each thread would have performed the atomic min operation on &min_cluster_label*/
		                

	       //if (threadIdx.x == 0) {
                 //  cluster_labels[pts_common_nb_id] = min_cluster_label[0];
            
               //}

         }
 }




void compute_neighbours(const int32_t* pts_common_nb,
                                const int32_t* neighboring_points,
			        int32_t* clusters,
			        const size_t n,
			        const float EPS2,
                                int32_t* min_points_area,
				int32_t* count,
                                int32_t* cluster_labels,
			        const float* dataset,
				const size_t num_of_Points_with_common_nb,
			        const size_t dimensions,
				const size_t m_min_points,
				int32_t m_global_point_offset,
				cudaStream_t* stream) {



    int blockSize;   // The launch configurator returned block size 
    int minGridSize; // The minimum grid size needed to achieve the 
                   // maximum occupancy for a full device launch 
 
    size_t max_points = 1000;
    size_t points_per_block = min(max_points, n);
    size_t sharedMemSize = sizeof(struct Point) * points_per_block;  // Shared memory size per block
        // Round up according to array size
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                                       compute_neighbours_cuda_hybrid, sharedMemSize, 0);

    int gridSize = (num_of_Points_with_common_nb + blockSize - 1) / blockSize;
    //dim3 gridSize(num_of_Points_with_common_nb, (n + blockSize - 1) / blockSize);  // 2D grid: n blocks for A, blocks for B

    compute_neighbours_cuda_hybrid<<<gridSize, blockSize, sharedMemSize, *stream>>>(
		                pts_common_nb,
                                neighboring_points,
				points_per_block,
				clusters,
                                n,
                                EPS2,
                                min_points_area,
                                count,
				cluster_labels,
                                dataset,
                                num_of_Points_with_common_nb,
                                dimensions,
				m_min_points,
				m_global_point_offset
				);

    cudaDeviceSynchronize();

}

