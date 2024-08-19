#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

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


/*
__global__ void regional_query_gpu((const int32_t* points_with_common_nb,
                           size_t dimensions,
                           int32_t* cluster_label,
                           const size_t n,
			   const size_t m,
                           const float* dataset,			  
                           const int32_t* neighboring_points,
                           const float EPS2,
                           int32_t* clusters,
                           int32_t* d_min_points_area,
			   int32_t* d_border_points_area,
                           int32_t* d_count, int32_t min_points) {


	int32_t NOT_VISITED = INT_MAX;
       
        // iterate through all neighboring points and check whether they are in range
	int i = threadIdx.x;

        //for (size_t i = 0; i < n; i++) {
            float offset = 0.0;

            // determine euclidean distance to other point
            for (size_t d = 0; d < dimensions; ++d) {
                float coord = dataset[point_index*dimensions+d];
                float ocoord = dataset[neighboring_points[i]*dimensions+d];

                const float distance = coord - ocoord;
                offset += distance * distance;
            }
            // .. if in range, add it to the vector with in range points
            if (offset <= EPS2) {
                const int32_t neighbor_label = clusters[neighboring_points[i]];
                 atomicAdd(d_count, 1);
                // if neighbor point has an assigned label and it is a core, determine what label to take
                if (neighbor_label < 0) {
                    *cluster_label = min(*cluster_label, (abs(neighbor_label)));
                    d_min_points_area[i] = neighboring_points[i]; //store only core points
                }
                else {
                    d_border_points_area[i] = neighboring_points[i];
                }
            }
        //}

	 __syncthreads();

        if(*d_count >= min_points) {

	    printf("count after computation is  %d", *d_count);
            for(int i = 0; i < n; ++i) {
                if(d_border_points_area[i] !=  NOT_VISITED)
                    atomicMin((clusters + d_border_points_area[i]), *cluster_label);
            }
	     
	    atomicMin((clusters + point_index), -(*cluster_label));
            //clusters[point_index] =  static_cast<Cluster<int32_t>>(-cluster_label);

        }

}
*/

struct is_not_int_max {
    __host__ __device__
    bool operator()(const int x) const {
        return x != INT_MAX;
    }
};

struct is_valid {
    __device__ bool operator()(const int x) const {
        return x != -1;
    }
};

__global__ void compute_neighbours_cuda(
                                const int32_t* pts_common_nb,
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
                                int32_t m_global_point_offset) {

	 const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	 const uint32_t limit  = tid * n + n;

	 if(tid < num_of_Points_with_common_nb) {

            const int32_t point_index = pts_common_nb[tid];

            const int32_t index = point_index * dimensions;

            cluster_labels[tid] =  m_global_point_offset + point_index + 1;

            for(int32_t i = 0 ; i < n; i++) {

                float offset = 0.0;
                for (size_t d = 0; d < dimensions; ++d) {
                    float coord = dataset[index + d];
                    float ocoord = dataset[neighboring_points[i] * dimensions + d];
                    float distance = coord - ocoord;
                    offset += distance * distance;
                }

                if (offset <= EPS2) {

		    //border_points_area[tid * n + i] = neighboring_points[i]; //will contain all neighbours

                    count[tid]++;

		    min_points_area[tid * n + i] = neighboring_points[i];
                    
		    if (clusters[neighboring_points[i]] < 0) {

		        cluster_labels[tid] = min(cluster_labels[tid], abs(clusters[neighboring_points[i]]));
                       //atomicMin(&cluster_labels[tid],
                       //                          abs(clusters[neighboring_points[i]]));
                    }
		    
                }
            }
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
				int32_t m_global_point_offset) {


    int blockSize = 256;
    int numBlocks = (num_of_Points_with_common_nb + blockSize - 1) / blockSize;

    // Launch the kernel to compute distances and identify valid indices
     compute_neighbours_cuda<<<numBlocks, blockSize>>>(pts_common_nb,
                                neighboring_points,
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
                                m_global_point_offset);

      cudaDeviceSynchronize();

}
	

/*
void region_query_cuda(const int32_t* points_with_common_nb,,
                           size_t dimensions,
                           int32_t* cluster_label,
                           const size_t n,
			   const size_t sizeofdataset,
        		   const float* dataset,                   
                           const int32_t* neighboring_points,
                           const float EPS2,
                           int32_t* clusters,
                           int32_t* min_points_area,
                           int32_t* count, int32_t min_points) {

   
    int32_t *d_min_points_area, *d_border_points_area, *d_count;
    float* d_dataset;
    int32_t *d_neighbouring_points, *d_clusters;

    cudaMalloc(&d_min_points_area, n * sizeof(int32_t));
    cudaMalloc(&d_border_points_area, n * sizeof(int32_t));
    cudaMalloc(&d_dataset, sizeofdataset * dimensions * sizeof(float));
    cudaMalloc(&d_clusters, sizeofdataset * sizeof(int32_t));
    cudaMalloc(&d_neighbouring_points, n * sizeof(int32_t));
    cudaMalloc(&d_count, sizeof(int32_t));



    cudaMemcpy(d_dataset, dataset, dimensions * sizeofdataset * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clusters, clusters, sizeofdataset * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbouring_points, neighboring_points, n * sizeof(int32_t), cudaMemcpyHostToDevice);

    regional_query_gpu<<<1, n>>>(point_index, dimensions, cluster_label, n, d_dataset, d_neighbouring_points, EPS2, d_clusters, d_min_points_area, d_border_points_area, d_count, min_points);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    cudaMemcpy(min_points_area, d_min_points_area, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(count, d_count, sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_min_points_area);
    cudaFree(d_border_points_area);

}
*/

/*
void say_hello(const int32_t point_index,
               const int32_t* neighboring_points,
               const float EPS2,
               const int32_t* clusters,
               int32_t* min_points_area,
               int32_t count) {

	 cuda_hello<<<1,1>>>();
}
*/

