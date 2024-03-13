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

#ifndef HPDBSCAN_H
#define HPDBSCAN_H

#include <cmath>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_set>

#include <omp.h>
#include <hdf5.h>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#ifdef WITH_OUTPUT
#include <iostream>
#endif

#include "atomic.h"
#include "constants.h"
#include "dataset.h"
#include "hdf5_util.h"
#include "io.h"
#include "rules.h"
#include "spatial_index.h"

template<typename index_type>
class HPDBSCAN {
    float m_epsilon;
    std::size_t m_min_points;

    #ifdef WITH_MPI
    int m_rank;
    int m_size;
    #endif


    template <typename data_type>
    Rules<index_type> local_dbscan(Clusters<index_type>& clusters, 
            const SpatialIndex<data_type, index_type>& index) {
        const float EPS2 = m_epsilon * m_epsilon;
        const std::size_t lower = index.lower_halo_bound();
        const std::size_t upper = index.upper_halo_bound();


        Rules<index_type> rules;
        Cell previous_cell = NOT_VISITED<index_type>;
        std::vector<index_type> neighboring_points;

#if 0
        int retval;

	int numEvents = 4;

        long long values[4];

        int events[4] = {PAPI_L1_DCH, PAPI_L1_DCA, PAPI_L2_DCH, PAPI_L2_DCA};

        if (PAPI_start_counters(events, numEvents) != PAPI_OK )  // !=PAPI_OK
	    printf("PAPI error: %d\n", 1);
#endif
        // local DBSCAN run
        #pragma omp parallel for schedule(dynamic, 2048) private(neighboring_points) firstprivate(previous_cell) reduction(merge: rules)
        for (index_type point = lower; point < static_cast<index_type>(upper); ++point) {
            // small optimization, we only perform a neighborhood query if it is a new cell
            Cell current_cell = index.cell_of(point);

            if (current_cell != previous_cell) {
                neighboring_points = index.template get_neighbors(current_cell);
                previous_cell = current_cell;
            }

            std::vector<index_type> min_points_area;
	        index_type count = 0;
            Cluster<index_type> cluster_label = NOISE<index_type>;
            if (neighboring_points.size() >= m_min_points) {
                cluster_label = index.region_query(point, neighboring_points, EPS2, clusters, min_points_area, count);
            }

            if (static_cast<std::size_t>(count) >= m_min_points) {
                // set the label to be negative as to mark it as core point
                // NOTE: unary operator promotes shorter types to int, so explicitly cast
                atomic_min(clusters.data() + point, 
                        static_cast<Cluster<index_type>>(-cluster_label));

                for (auto& other : min_points_area) {

                    if(other != NOT_VISITED<index_type>) {
                        // get the absolute value here, we are only interested what cluster it is not in the core property
                        // check whether the other point is a cluster
                        if (clusters[other] < 0) {
                            rules.update(std::abs(clusters[other]), cluster_label);
                        }
                        // mark as a border point
                        atomic_min(clusters.data() + other, cluster_label);
                    }
                }
            }
            else if (clusters[point] == NOT_VISITED<index_type>) {
                // mark as noise
                atomic_min(clusters.data() + point, NOISE<index_type>);
            }
        }

#if 0
        if ((retval = PAPI_read_counters(values, numEvents)) != PAPI_OK) {
            fprintf(stderr, "PAPI failed to read counters: %s\n", PAPI_strerror(retval));
            exit(1);
        }

	std::cout<<"Level 1 data cache hits "<<values[0]<<std::endl;
	std::cout<<"Level 1 data cache accesses "<<values[1]<<std::endl; 
	double cache_miss_rate = static_cast<double>(values[0]) / static_cast<double>(values[1]);
	std::cout<<"L1 cache hit rate "<<cache_miss_rate<<std::endl;

	std::cout<<"L2 data cache hits "<<values[2]<<std::endl;
        std::cout<<"L2 data cache accesses "<<values[3]<<std::endl;
        cache_miss_rate = static_cast<double>(values[2]) / static_cast<double>(values[3]);
        std::cout<<"L2 cache hit rate "<<cache_miss_rate<<std::endl;
#endif

        return rules;
    }

    #ifdef WITH_MPI
    template <typename data_type>
    void merge_halos(Clusters<index_type>& clusters, Rules<index_type>& rules,
            const SpatialIndex<data_type, index_type>& index) {
        Cuts cuts = index.compute_cuts();

        // exchange the number of points in the halos
        std::vector<int> send_counts(m_size, 0);
        std::vector<int> recv_counts(m_size, 0);
        for (std::size_t i = 0; i < cuts.size(); ++i) {
            send_counts[i] = static_cast<int>(cuts[i].second - cuts[i].first);
        }
        MPI_Alltoall(send_counts.data(), 1, MPI_INT32_T,
                     recv_counts.data(), 1, MPI_INT32_T, MPI_COMM_WORLD);

        // accumulate the numbers of points from each node
        std::vector<int> send_displs(m_size, 0);
        std::vector<int> recv_displs(m_size, 0);
        std::size_t total_items_to_receive = 0;
        for (int i = 0; i < m_size; ++i) {
            send_displs[i] = cuts[i].first;
            recv_displs[i] = total_items_to_receive;
            total_items_to_receive += static_cast<std::size_t>(recv_counts[i]);
        }

        // create a buffer for the incoming cluster labels and exchange them
        const std::size_t upper_halo_bound = index.upper_halo_bound();
        const std::size_t lower_halo_bound = index.lower_halo_bound();
        Clusters<index_type> halo_labels(total_items_to_receive);

        MPI_Alltoallv(
            clusters.data(),    send_counts.data(), send_displs.data(), get_mpi_type<index_type>(),
            halo_labels.data(), recv_counts.data(), recv_displs.data(), get_mpi_type<index_type>(), MPI_COMM_WORLD
        );

        // update the local clusters with the received information
        for (std::size_t i = 0; i < m_size; ++i) {
            std::size_t offset = (i < m_rank ? lower_halo_bound : upper_halo_bound - recv_counts[i]);

            for (std::size_t j = 0; j < recv_counts[i]; ++j) {
                const std::size_t index = j + offset;
                const Cluster<index_type> own_cluster = clusters[index];
                const Cluster<index_type> halo_cluster = halo_labels[j + recv_displs[i]];

                // incoming cluster label is core point, update it
                if (own_cluster < 0) {
                    //own_cluster = std::abs(own_cluster);
                    index_type a_own_cluster = -own_cluster;
                    const auto [min,max] = std::minmax(a_own_cluster, halo_cluster);
                    rules.update(max, min);
                } else {
                    atomic_min(&clusters[index], halo_cluster);
                }
            }
        }
    }

    void distribute_rules(Rules<index_type>& rules) {
        const int number_of_rules = static_cast<int>(rules.size());

        // determine how many rules each rank will send
        std::vector<int> send_counts(m_size);
        std::vector<int> send_displs(m_size);
        std::vector<int> recv_counts(m_size);
        std::vector<int> recv_displs(m_size);

        for (int i = 0; i < m_size; ++i) {
            send_counts[i] = 2 * number_of_rules;
            send_displs[i] = 0;
        }
        MPI_Alltoall(send_counts.data(), 1, MPI_INT32_T,
                     recv_counts.data(), 1, MPI_INT32_T, MPI_COMM_WORLD);

        // ... based on that calculate the displacements into the receive buffer
        std::size_t total = 0;
        for (int i = 0; i < m_size; ++i) {
            recv_displs[i] = total;
            total += recv_counts[i];
        }

        // serialize the rules
        Clusters<index_type> serialized_rules(send_counts[m_rank]);
        std::size_t index = 0;
        for (const auto& rule : rules) {
            serialized_rules[index++] = rule.first;
            serialized_rules[index++] = rule.second;
        }

        // exchange the rules and update the own rules
        Clusters<index_type> incoming_rules(total);
        MPI_Alltoallv(
            serialized_rules.data(), send_counts.data(), send_displs.data(), get_mpi_type<index_type>(),
            incoming_rules.data(),   recv_counts.data(), recv_displs.data(), get_mpi_type<index_type>(), MPI_COMM_WORLD
        );
        for (std::size_t i = 0; i < total; i += 2) {
            rules.update(incoming_rules[i], incoming_rules[i + 1]);
        }
    }
    #endif

    void apply_rules(Clusters<index_type>& clusters, const Rules<index_type>& rules) {
        #pragma omp parallel for
        for (std::size_t i = 0; i < clusters.size(); ++i) {
            const bool is_core = clusters[i] < 0;
            Cluster<index_type> cluster = std::abs(clusters[i]);
            Cluster<index_type> matching_rule = rules.rule(cluster);

            while (matching_rule < NOISE<index_type>) {
                cluster = matching_rule;
                matching_rule = rules.rule(matching_rule);
            }
            clusters[i] = is_core ? -cluster : cluster;
        }
    }

    #ifdef WITH_OUTPUT
    template<typename data_type>
    void summarize(const Dataset<data_type>& dataset, const Clusters<index_type>& clusters) const {
        std::unordered_set<Cluster<index_type>> unique_clusters;
        std::size_t cluster_points = 0;
        std::size_t core_points = 0;
        std::size_t noise_points = 0;

        // iterate through the points and sum up the
        for (std::size_t i = 0; i < dataset.m_chunk[0]; ++i) {
            const Cluster<index_type> cluster = clusters[i];
            unique_clusters.insert(std::abs(cluster));

            if (cluster == 0) {
                ++noise_points;
            } else {
                ++cluster_points;
            }
            if (cluster < 0) {
                ++core_points;
            }
        }
        std::size_t metrics[] = {cluster_points, noise_points, core_points};

        #ifdef WITH_MPI
        int number_of_unique_clusters = static_cast<int>(unique_clusters.size());
        std::vector<int> set_counts(m_size);
        std::vector<int> set_displs(m_size);

        if (m_rank == 0) {
        #endif
        std::cout << "Summary..." << std::endl;
        #ifdef WITH_MPI
        }
        MPI_Gather(&number_of_unique_clusters, 1, MPI_INT32_T, 
                   set_counts.data(), 1, MPI_INT32_T, 0, MPI_COMM_WORLD);

        // allocate the buffers for the serialized sets
        Clusters<index_type> global_buffer;
        Clusters<index_type> local_buffer(number_of_unique_clusters);
        std::copy(unique_clusters.begin(), unique_clusters.end(), local_buffer.begin());

        // sum up the total number of elements on the MPI root to determine the global buffer size
        std::size_t buffer_size = 0;
        if (m_rank == 0) {
            for (int i = 0; i < m_size; ++i) {
                set_displs[i] = buffer_size;
                buffer_size += set_counts[i];
            }
            global_buffer.resize(buffer_size);
        }

        // collect the individual unique clusters on the MPI root into a global buffer
        MPI_Gatherv(
            local_buffer.data(), number_of_unique_clusters, get_mpi_type<index_type>(),
            global_buffer.data(), set_counts.data(), set_displs.data(), get_mpi_type<index_type>(), 0, MPI_COMM_WORLD
        );
        // accumulate the metrics of each node
        MPI_Reduce(
            m_rank == 0 ? MPI_IN_PLACE : metrics, metrics, sizeof(metrics) / sizeof(std::size_t),
            MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD
        );

        if (m_rank == 0) {
            unique_clusters.insert(global_buffer.begin(), global_buffer.end());
        #endif
        std::cout << "\tClusters:       " << (metrics[1] ? unique_clusters.size() - 1 : unique_clusters.size()) << std::endl
                  << "\tCluster points: " << metrics[0] << std::endl
                  << "\tNoise points:   " << metrics[1] << std::endl
                  << "\tCore points:    " << metrics[2] << std::endl;
        #ifdef WITH_MPI
        }
        #endif
    }
    #endif

public:
    HPDBSCAN(float epsilon, std::size_t min_points) : m_epsilon(epsilon), m_min_points(min_points) {
        #ifdef WITH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &m_size);
        #endif

        // sanitize values
        if (epsilon <= 0.0) {
            throw std::invalid_argument("epsilon needs to be positive");
        }
    }

    Clusters<index_type> cluster(const std::string& path, const std::string& dataset) {
        return cluster(path, dataset, omp_get_max_threads());
    }

    template<typename data_type>
    Clusters<index_type> cluster(const std::string& path, const std::string& dataset, int threads) {
        // read in the data
        Dataset<data_type> data = IO::read_hdf5<data_type>(path, dataset);

        return cluster<data_type>(data, threads);
    }

    Clusters<index_type> cluster(const std::string& path, const std::string& dataset, int threads) {
        // get H5 data type
        hid_t hdf5_type = IO::get_hdf5_datatype(path, dataset);

#define H5TC(native, h5)\
        if (hdf5_type == h5)\
        {\
            return cluster<native>(path, dataset, threads);\
        }
#define H5TCe(native, h5)\
        else if (hdf5_type == h5)\
        {\
            return cluster<native>(path, dataset, threads);\
        }
        H5TC(std::int8_t,    H5T_NATIVE_SCHAR)
        H5TCe(std::int16_t,  H5T_NATIVE_SHORT)
        H5TCe(std::int32_t,  H5T_NATIVE_INT)
        H5TCe(std::int64_t,  H5T_NATIVE_LONG)
        H5TCe(std::uint8_t,  H5T_NATIVE_UCHAR)
        H5TCe(std::uint16_t, H5T_NATIVE_USHORT)
        H5TCe(std::uint32_t, H5T_NATIVE_UINT)
        H5TCe(std::uint64_t, H5T_NATIVE_ULONG)
        H5TCe(float,         H5T_NATIVE_FLOAT)
        H5TCe(double,        H5T_NATIVE_DOUBLE)

#undef H5TC
#undef H5TCe

        throw std::invalid_argument("Unsupported data set type");
    }

    template <typename data_type>
    Clusters<index_type> cluster(Dataset<data_type>& dataset, int threads=omp_get_max_threads()) {
        #ifdef WITH_OUTPUT
        double execution_start = omp_get_wtime();
        #endif
        // set the number of threads
        omp_set_num_threads(threads);

        // set default number formatting
        #ifdef WITH_OUTPUT
        std::cout << std::fixed << std::setw(11) << std::setprecision(6) << std::setfill(' ');
        #endif

        // initialize the feature indexer
        SpatialIndex<data_type, index_type> index(dataset, m_epsilon);
        // initialize the clusters array
        Clusters<index_type> clusters(dataset.m_chunk[0], NOT_VISITED<index_type>);

        // run the first local clustering round
        #ifdef WITH_OUTPUT
            double start = omp_get_wtime();

            #ifdef WITH_MPI
            if (m_rank == 0) {
            #endif
            std::cout << "Clustering..." << std::endl;
            std::cout << "\tDBSCAN...              " << std::flush;
            #ifdef WITH_MPI
            }
            #endif
            start = omp_get_wtime();
        #endif
        Rules rules = local_dbscan(clusters, index);
        #ifdef WITH_OUTPUT
            #ifdef WITH_MPI
            if (m_rank == 0) {
            #endif
            std::cout << "[OK] in " << omp_get_wtime() - start << std::endl;
            #ifdef WITH_MPI
            }
            #endif
            start = omp_get_wtime();
        #endif

        #ifdef WITH_MPI
            #ifdef WITH_OUTPUT
            if (m_rank == 0) {
                std::cout << "\tMerging halos...       " << std::flush;
            }
            #endif
            merge_halos(clusters, rules, index);
            // MPI_Barrier(MPI_COMM_WORLD);
            distribute_rules(rules);
            #ifdef WITH_OUTPUT
            if (m_rank == 0) {
                std::cout << "[OK] in " << omp_get_wtime() - start << std::endl;
                start = omp_get_wtime();
            }
            #endif
        #endif

        #ifdef WITH_OUTPUT
            #ifdef WITH_MPI
            if (m_rank == 0) {
            #endif
            std::cout << "\tAppyling rules...      " << std::flush;
            #ifdef WITH_MPI
            }
            #endif
        #endif
        apply_rules(clusters, rules);
        #ifdef WITH_OUTPUT
            #ifdef WITH_MPI
            if (m_rank == 0) {
            #endif
            std::cout << "[OK] in " << omp_get_wtime() - start << std::endl;
            #ifdef WITH_MPI
            }
            #endif
        #endif

        #ifdef WITH_OUTPUT
            #ifdef WITH_MPI
            if (m_rank == 0) {
            #endif
                std::cout << "\tRecovering order:    " << std::endl;
                start = omp_get_wtime();
            #ifdef WITH_MPI
            }
            #endif
        #endif
        index.recover_initial_order(clusters);
        #ifdef WITH_OUTPUT
            #ifdef WITH_MPI
            if (m_rank == 0) {
            #endif
            std::cout << "\tRecovering order finished in " << omp_get_wtime() - start << std::endl;
            #ifdef WITH_MPI
            }
            #endif
            start = omp_get_wtime();
        #endif

        #ifdef WITH_OUTPUT
        summarize(dataset, clusters);
            #ifdef WITH_MPI
            if (m_rank == 0) {
            #endif
            std::cout << "Total time: " << omp_get_wtime() - execution_start << std::endl;
            #ifdef WITH_MPI
            }
            #endif
        #endif

        // return the results
        return clusters;
    }

    template <typename data_type>
    Clusters<index_type> cluster(data_type* data, int dim0, int dim1, int threads) {
        std::size_t chunk[2] = {static_cast<std::size_t>(dim0), static_cast<std::size_t>(dim1)};
        Dataset<data_type> dataset(data, chunk, get_hdf5_type<data_type>());

        return cluster<data_type, index_type>(dataset, threads);
    }

    template <typename data_type>
    Clusters<index_type> cluster(data_type* data, int dim0, int dim1) {
        return cluster<data_type, index_type>(data, dim0, dim1, omp_get_max_threads());
    }
};

#endif // HPDBSCAN_H
