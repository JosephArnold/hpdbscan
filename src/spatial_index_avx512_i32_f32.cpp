#include <immintrin.h>

#define IMPLEMENTING_OPTIMIZATION

#include "spatial_index.h"

typedef std::int32_t index_type;

#if defined(USE_ND_OPTIMIZATIONS)
template<>
template<>
Cluster<index_type> SpatialIndex<float, index_type>::region_query_optimized_nd<3>(
        const index_type point_index,
        const std::vector<index_type>& neighboring_points,
        const float EPS2,
        const Clusters<index_type>& clusters,
        std::vector<index_type>& min_points_area,
        index_type& count) const {
    const std::size_t dimensions = static_cast<std::size_t>(m_data.m_chunk[1]);
    const float* point = &m_data.m_elements[point_index * dimensions];
    const float* np_ptr = m_data.m_elements.data();
    const std::size_t precision = sizeof(float);
    Cluster<index_type> cluster_label = m_global_point_offset + point_index + 1;
    const std::size_t ElementsPerAVX = sizeof(__m512) / precision;
    std::size_t elements_in_vector = ElementsPerAVX;

    //  Align memory (This has to be done while data loading, not here. )
    // Initialize array pointers and masks
    std::size_t n = neighboring_points.size();
    min_points_area = std::vector<index_type>(n, NOT_VISITED<index_type>);

    __m512 v_eps = _mm512_set1_ps(EPS2);
    __m512 v_zero_ps = _mm512_setzero_ps();
    __m512i v_zero_epi32 = _mm512_setzero_epi32();
    __m512i v_one = _mm512_set1_epi32(1);
    __m512i v_dims = _mm512_set1_epi32(dimensions);

    __m512 v_point_x = _mm512_set1_ps(point[0]);
    __m512 v_point_y = _mm512_set1_ps(point[1]);
    __m512 v_point_z = _mm512_set1_ps(point[2]);

#if defined(USE_8X_NEIGHBOUR_LOOP_UNROLL)
    constexpr std::size_t vectors_per_loop = 8;
    std::size_t loop_elements = elements_in_vector*vectors_per_loop;
    std::size_t rest = n % loop_elements;
    std::size_t chunks = n/loop_elements;

    for (std::size_t j = 0; j < chunks; j++) {
        std::size_t i = j*loop_elements;
        // No need for masks in unrolled loop 
        __m512i v_indices1 = _mm512_loadu_epi32(&neighboring_points[i+elements_in_vector*0]);
        __m512i v_indices2 = _mm512_loadu_epi32(&neighboring_points[i+elements_in_vector*1]);
        __m512i v_indices3 = _mm512_loadu_epi32(&neighboring_points[i+elements_in_vector*2]);
        __m512i v_indices4 = _mm512_loadu_epi32(&neighboring_points[i+elements_in_vector*3]);
        __m512i v_indices5 = _mm512_loadu_epi32(&neighboring_points[i+elements_in_vector*4]);
        __m512i v_indices6 = _mm512_loadu_epi32(&neighboring_points[i+elements_in_vector*5]);
        __m512i v_indices7 = _mm512_loadu_epi32(&neighboring_points[i+elements_in_vector*6]);
        __m512i v_indices8 = _mm512_loadu_epi32(&neighboring_points[i+elements_in_vector*7]);

        __m512i v_indices_scaled1 = _mm512_mullo_epi32(v_indices1, v_dims);
        __m512i v_indices_scaled2 = _mm512_mullo_epi32(v_indices2, v_dims);
        __m512i v_indices_scaled3 = _mm512_mullo_epi32(v_indices3, v_dims);
        __m512i v_indices_scaled4 = _mm512_mullo_epi32(v_indices4, v_dims);
        __m512i v_indices_scaled5 = _mm512_mullo_epi32(v_indices5, v_dims);
        __m512i v_indices_scaled6 = _mm512_mullo_epi32(v_indices6, v_dims);
        __m512i v_indices_scaled7 = _mm512_mullo_epi32(v_indices7, v_dims);
        __m512i v_indices_scaled8 = _mm512_mullo_epi32(v_indices8, v_dims);

        __m512 v_neighbour_x1 = _mm512_i32gather_ps(v_indices_scaled1, np_ptr+0, 4);
        __m512 v_neighbour_y1 = _mm512_i32gather_ps(v_indices_scaled1, np_ptr+1, 4);
        __m512 v_neighbour_z1 = _mm512_i32gather_ps(v_indices_scaled1, np_ptr+2, 4);

        __m512 v_neighbour_x2 = _mm512_i32gather_ps(v_indices_scaled2, np_ptr+0, 4);
        __m512 v_neighbour_y2 = _mm512_i32gather_ps(v_indices_scaled2, np_ptr+1, 4);
        __m512 v_neighbour_z2 = _mm512_i32gather_ps(v_indices_scaled2, np_ptr+2, 4);

        __m512 v_neighbour_x3 = _mm512_i32gather_ps(v_indices_scaled3, np_ptr+0, 4);
        __m512 v_neighbour_y3 = _mm512_i32gather_ps(v_indices_scaled3, np_ptr+1, 4);
        __m512 v_neighbour_z3 = _mm512_i32gather_ps(v_indices_scaled3, np_ptr+2, 4);

        __m512 v_neighbour_x4 = _mm512_i32gather_ps(v_indices_scaled4, np_ptr+0, 4);
        __m512 v_neighbour_y4 = _mm512_i32gather_ps(v_indices_scaled4, np_ptr+1, 4);
        __m512 v_neighbour_z4 = _mm512_i32gather_ps(v_indices_scaled4, np_ptr+2, 4);

        __m512 v_neighbour_x5 = _mm512_i32gather_ps(v_indices_scaled5, np_ptr+0, 4);
        __m512 v_neighbour_y5 = _mm512_i32gather_ps(v_indices_scaled5, np_ptr+1, 4);
        __m512 v_neighbour_z5 = _mm512_i32gather_ps(v_indices_scaled5, np_ptr+2, 4);

        __m512 v_neighbour_x6 = _mm512_i32gather_ps(v_indices_scaled6, np_ptr+0, 4);
        __m512 v_neighbour_y6 = _mm512_i32gather_ps(v_indices_scaled6, np_ptr+1, 4);
        __m512 v_neighbour_z6 = _mm512_i32gather_ps(v_indices_scaled6, np_ptr+2, 4);

        __m512 v_neighbour_x7 = _mm512_i32gather_ps(v_indices_scaled7, np_ptr+0, 4);
        __m512 v_neighbour_y7 = _mm512_i32gather_ps(v_indices_scaled7, np_ptr+1, 4);
        __m512 v_neighbour_z7 = _mm512_i32gather_ps(v_indices_scaled7, np_ptr+2, 4);

        __m512 v_neighbour_x8 = _mm512_i32gather_ps(v_indices_scaled8, np_ptr+0, 4);
        __m512 v_neighbour_y8 = _mm512_i32gather_ps(v_indices_scaled8, np_ptr+1, 4);
        __m512 v_neighbour_z8 = _mm512_i32gather_ps(v_indices_scaled8, np_ptr+2, 4);

        __m512 v_diff_x1 = _mm512_sub_ps(v_point_x, v_neighbour_x1);
        __m512 v_results1 = _mm512_mul_ps(v_diff_x1, v_diff_x1);
        __m512 v_diff_x2 = _mm512_sub_ps(v_point_x, v_neighbour_x2);
        __m512 v_results2 = _mm512_mul_ps(v_diff_x2, v_diff_x2);
        __m512 v_diff_x3 = _mm512_sub_ps(v_point_x, v_neighbour_x3);
        __m512 v_results3 = _mm512_mul_ps(v_diff_x3, v_diff_x3);
        __m512 v_diff_x4 = _mm512_sub_ps(v_point_x, v_neighbour_x4);
        __m512 v_results4 = _mm512_mul_ps(v_diff_x4, v_diff_x4);
        __m512 v_diff_x5 = _mm512_sub_ps(v_point_x, v_neighbour_x5);
        __m512 v_results5 = _mm512_mul_ps(v_diff_x5, v_diff_x5);
        __m512 v_diff_x6 = _mm512_sub_ps(v_point_x, v_neighbour_x6);
        __m512 v_results6 = _mm512_mul_ps(v_diff_x6, v_diff_x6);
        __m512 v_diff_x7 = _mm512_sub_ps(v_point_x, v_neighbour_x7);
        __m512 v_results7 = _mm512_mul_ps(v_diff_x7, v_diff_x7);
        __m512 v_diff_x8 = _mm512_sub_ps(v_point_x, v_neighbour_x8);
        __m512 v_results8 = _mm512_mul_ps(v_diff_x8, v_diff_x8);

        __m512 v_diff_y1 = _mm512_sub_ps(v_point_y, v_neighbour_y1);
        __m512 v_results1_1 = _mm512_mul_ps(v_diff_y1, v_diff_y1);
        __m512 v_diff_y2 = _mm512_sub_ps(v_point_y, v_neighbour_y2);
        __m512 v_results2_1 = _mm512_mul_ps(v_diff_y2, v_diff_y2);
        __m512 v_diff_y3 = _mm512_sub_ps(v_point_y, v_neighbour_y3);
        __m512 v_results3_1 = _mm512_mul_ps(v_diff_y3, v_diff_y3);
        __m512 v_diff_y4 = _mm512_sub_ps(v_point_y, v_neighbour_y4);
        __m512 v_results4_1 = _mm512_mul_ps(v_diff_y4, v_diff_y4);
        __m512 v_diff_y5 = _mm512_sub_ps(v_point_y, v_neighbour_y5);
        __m512 v_results5_1 = _mm512_mul_ps(v_diff_y5, v_diff_y5);
        __m512 v_diff_y6 = _mm512_sub_ps(v_point_y, v_neighbour_y6);
        __m512 v_results6_1 = _mm512_mul_ps(v_diff_y6, v_diff_y6);
        __m512 v_diff_y7 = _mm512_sub_ps(v_point_y, v_neighbour_y7);
        __m512 v_results7_1 = _mm512_mul_ps(v_diff_y7, v_diff_y7);
        __m512 v_diff_y8 = _mm512_sub_ps(v_point_y, v_neighbour_y8);
        __m512 v_results8_1 = _mm512_mul_ps(v_diff_y8, v_diff_y8);

        __m512 v_diff_z1 = _mm512_sub_ps(v_point_z, v_neighbour_z1);
        __m512 v_results1_2 = _mm512_fmadd_ps(v_diff_z1, v_diff_z1, v_results1);
        __m512 v_diff_z2 = _mm512_sub_ps(v_point_z, v_neighbour_z2);
        __m512 v_results2_2 = _mm512_fmadd_ps(v_diff_z2, v_diff_z2, v_results2);
        __m512 v_diff_z3 = _mm512_sub_ps(v_point_z, v_neighbour_z3);
        __m512 v_results3_2 = _mm512_fmadd_ps(v_diff_z3, v_diff_z3, v_results3);
        __m512 v_diff_z4 = _mm512_sub_ps(v_point_z, v_neighbour_z4);
        __m512 v_results4_2 = _mm512_fmadd_ps(v_diff_z4, v_diff_z4, v_results4);
        __m512 v_diff_z5 = _mm512_sub_ps(v_point_z, v_neighbour_z5);
        __m512 v_results5_2 = _mm512_fmadd_ps(v_diff_z5, v_diff_z5, v_results5);
        __m512 v_diff_z6 = _mm512_sub_ps(v_point_z, v_neighbour_z6);
        __m512 v_results6_2 = _mm512_fmadd_ps(v_diff_z6, v_diff_z6, v_results6);
        __m512 v_diff_z7 = _mm512_sub_ps(v_point_z, v_neighbour_z7);
        __m512 v_results7_2 = _mm512_fmadd_ps(v_diff_z7, v_diff_z7, v_results7);
        __m512 v_diff_z8 = _mm512_sub_ps(v_point_z, v_neighbour_z8);
        __m512 v_results8_2 = _mm512_fmadd_ps(v_diff_z8, v_diff_z8, v_results8);

        v_results1 = _mm512_add_ps(v_results1_2, v_results1_1);
        v_results2 = _mm512_add_ps(v_results2_2, v_results2_1);
        v_results3 = _mm512_add_ps(v_results3_2, v_results3_1);
        v_results4 = _mm512_add_ps(v_results4_2, v_results4_1);
        v_results5 = _mm512_add_ps(v_results5_2, v_results5_1);
        v_results6 = _mm512_add_ps(v_results6_2, v_results6_1);
        v_results7 = _mm512_add_ps(v_results7_2, v_results7_1);
        v_results8 = _mm512_add_ps(v_results8_2, v_results8_1);

        // Mark entries where the result is less than epsilon^2
        __mmask16 v_eps_mask1 = _mm512_cmple_ps_mask( v_results1, v_eps);
        __mmask16 v_eps_mask2 = _mm512_cmple_ps_mask( v_results2, v_eps);
        __mmask16 v_eps_mask3 = _mm512_cmple_ps_mask( v_results3, v_eps);
        __mmask16 v_eps_mask4 = _mm512_cmple_ps_mask( v_results4, v_eps);
        __mmask16 v_eps_mask5 = _mm512_cmple_ps_mask( v_results5, v_eps);
        __mmask16 v_eps_mask6 = _mm512_cmple_ps_mask( v_results6, v_eps);
        __mmask16 v_eps_mask7 = _mm512_cmple_ps_mask( v_results7, v_eps);
        __mmask16 v_eps_mask8 = _mm512_cmple_ps_mask( v_results8, v_eps);

        count += _mm512_mask_reduce_add_epi32(v_eps_mask1, v_one);
        count += _mm512_mask_reduce_add_epi32(v_eps_mask2, v_one);
        count += _mm512_mask_reduce_add_epi32(v_eps_mask3, v_one);
        count += _mm512_mask_reduce_add_epi32(v_eps_mask4, v_one);
        count += _mm512_mask_reduce_add_epi32(v_eps_mask5, v_one);
        count += _mm512_mask_reduce_add_epi32(v_eps_mask6, v_one);
        count += _mm512_mask_reduce_add_epi32(v_eps_mask7, v_one);
        count += _mm512_mask_reduce_add_epi32(v_eps_mask8, v_one);
        // std::cout << "Count: " << count << std::endl;

        // Load cluster labels of close entries
        __m512i v_cluster_labels1 = _mm512_mask_i32gather_epi32(v_zero_epi32, v_eps_mask1, v_indices1, &clusters[0], 4);
        __m512i v_cluster_labels2 = _mm512_mask_i32gather_epi32(v_zero_epi32, v_eps_mask2, v_indices2, &clusters[0], 4);
        __m512i v_cluster_labels3 = _mm512_mask_i32gather_epi32(v_zero_epi32, v_eps_mask3, v_indices3, &clusters[0], 4);
        __m512i v_cluster_labels4 = _mm512_mask_i32gather_epi32(v_zero_epi32, v_eps_mask4, v_indices4, &clusters[0], 4);
        __m512i v_cluster_labels5 = _mm512_mask_i32gather_epi32(v_zero_epi32, v_eps_mask5, v_indices5, &clusters[0], 4);
        __m512i v_cluster_labels6 = _mm512_mask_i32gather_epi32(v_zero_epi32, v_eps_mask6, v_indices6, &clusters[0], 4);
        __m512i v_cluster_labels7 = _mm512_mask_i32gather_epi32(v_zero_epi32, v_eps_mask7, v_indices7, &clusters[0], 4);
        __m512i v_cluster_labels8 = _mm512_mask_i32gather_epi32(v_zero_epi32, v_eps_mask8, v_indices8, &clusters[0], 4);

        // filter labels that are not visited and that are less than zero
        __mmask16 v_less_than_zero1 = _mm512_cmplt_epi32_mask(v_cluster_labels1, v_zero_epi32);
        __mmask16 v_less_than_zero2 = _mm512_cmplt_epi32_mask(v_cluster_labels2, v_zero_epi32);
        __mmask16 v_less_than_zero3 = _mm512_cmplt_epi32_mask(v_cluster_labels3, v_zero_epi32);
        __mmask16 v_less_than_zero4 = _mm512_cmplt_epi32_mask(v_cluster_labels4, v_zero_epi32);
        __mmask16 v_less_than_zero5 = _mm512_cmplt_epi32_mask(v_cluster_labels5, v_zero_epi32);
        __mmask16 v_less_than_zero6 = _mm512_cmplt_epi32_mask(v_cluster_labels6, v_zero_epi32);
        __mmask16 v_less_than_zero7 = _mm512_cmplt_epi32_mask(v_cluster_labels7, v_zero_epi32);
        __mmask16 v_less_than_zero8 = _mm512_cmplt_epi32_mask(v_cluster_labels8, v_zero_epi32);

        v_cluster_labels1 = _mm512_abs_epi32(v_cluster_labels1);
        v_cluster_labels2 = _mm512_abs_epi32(v_cluster_labels2);
        v_cluster_labels3 = _mm512_abs_epi32(v_cluster_labels3);
        v_cluster_labels4 = _mm512_abs_epi32(v_cluster_labels4);
        v_cluster_labels5 = _mm512_abs_epi32(v_cluster_labels5);
        v_cluster_labels6 = _mm512_abs_epi32(v_cluster_labels6);
        v_cluster_labels7 = _mm512_abs_epi32(v_cluster_labels7);
        v_cluster_labels8 = _mm512_abs_epi32(v_cluster_labels8);

        // set min cluster label
        cluster_label = std::min(cluster_label, _mm512_mask_reduce_min_epi32(v_less_than_zero1, v_cluster_labels1));
        cluster_label = std::min(cluster_label, _mm512_mask_reduce_min_epi32(v_less_than_zero2, v_cluster_labels2));
        cluster_label = std::min(cluster_label, _mm512_mask_reduce_min_epi32(v_less_than_zero3, v_cluster_labels3));
        cluster_label = std::min(cluster_label, _mm512_mask_reduce_min_epi32(v_less_than_zero4, v_cluster_labels4));
        cluster_label = std::min(cluster_label, _mm512_mask_reduce_min_epi32(v_less_than_zero5, v_cluster_labels5));
        cluster_label = std::min(cluster_label, _mm512_mask_reduce_min_epi32(v_less_than_zero6, v_cluster_labels6));
        cluster_label = std::min(cluster_label, _mm512_mask_reduce_min_epi32(v_less_than_zero7, v_cluster_labels7));
        cluster_label = std::min(cluster_label, _mm512_mask_reduce_min_epi32(v_less_than_zero8, v_cluster_labels8));
        // std::cout << "cluster label: " << cluster_label << std::endl;

        // store min_points_area
        _mm512_mask_storeu_epi32(&min_points_area[i+elements_in_vector*0], v_eps_mask1, v_indices1);
        _mm512_mask_storeu_epi32(&min_points_area[i+elements_in_vector*1], v_eps_mask2, v_indices2);
        _mm512_mask_storeu_epi32(&min_points_area[i+elements_in_vector*2], v_eps_mask3, v_indices3);
        _mm512_mask_storeu_epi32(&min_points_area[i+elements_in_vector*3], v_eps_mask4, v_indices4);
        _mm512_mask_storeu_epi32(&min_points_area[i+elements_in_vector*4], v_eps_mask5, v_indices5);
        _mm512_mask_storeu_epi32(&min_points_area[i+elements_in_vector*5], v_eps_mask6, v_indices6);
        _mm512_mask_storeu_epi32(&min_points_area[i+elements_in_vector*6], v_eps_mask7, v_indices7);
        _mm512_mask_storeu_epi32(&min_points_area[i+elements_in_vector*7], v_eps_mask8, v_indices8);
    }
    for (std::size_t i = n-rest; i < n; i += elements_in_vector) {
#elif defined(USE_4X_NEIGHBOUR_LOOP_UNROLL)
    constexpr std::size_t vectors_per_loop = 4;
    std::size_t loop_elements = elements_in_vector*vectors_per_loop;
    std::size_t rest = n % loop_elements;
    std::size_t chunks = n/loop_elements;

    for (std::size_t j = 0; j < chunks; j++) {
        std::size_t i = j*loop_elements;
        // No need for masks in unrolled loop 
        __m512i v_indices1 = _mm512_loadu_epi32(&neighboring_points[i+elements_in_vector*0]);
        __m512i v_indices2 = _mm512_loadu_epi32(&neighboring_points[i+elements_in_vector*1]);
        __m512i v_indices3 = _mm512_loadu_epi32(&neighboring_points[i+elements_in_vector*2]);
        __m512i v_indices4 = _mm512_loadu_epi32(&neighboring_points[i+elements_in_vector*3]);

        __m512i v_indices_scaled1 = _mm512_mullo_epi32(v_indices1, v_dims);
        __m512i v_indices_scaled2 = _mm512_mullo_epi32(v_indices2, v_dims);
        __m512i v_indices_scaled3 = _mm512_mullo_epi32(v_indices3, v_dims);
        __m512i v_indices_scaled4 = _mm512_mullo_epi32(v_indices4, v_dims);

        __m512 v_neighbour_x1 = _mm512_i32gather_ps(v_indices_scaled1, np_ptr+0, 4);
        __m512 v_neighbour_y1 = _mm512_i32gather_ps(v_indices_scaled1, np_ptr+1, 4);
        __m512 v_neighbour_z1 = _mm512_i32gather_ps(v_indices_scaled1, np_ptr+2, 4);

        __m512 v_neighbour_x2 = _mm512_i32gather_ps(v_indices_scaled2, np_ptr+0, 4);
        __m512 v_neighbour_y2 = _mm512_i32gather_ps(v_indices_scaled2, np_ptr+1, 4);
        __m512 v_neighbour_z2 = _mm512_i32gather_ps(v_indices_scaled2, np_ptr+2, 4);

        __m512 v_neighbour_x3 = _mm512_i32gather_ps(v_indices_scaled3, np_ptr+0, 4);
        __m512 v_neighbour_y3 = _mm512_i32gather_ps(v_indices_scaled3, np_ptr+1, 4);
        __m512 v_neighbour_z3 = _mm512_i32gather_ps(v_indices_scaled3, np_ptr+2, 4);

        __m512 v_neighbour_x4 = _mm512_i32gather_ps(v_indices_scaled4, np_ptr+0, 4);
        __m512 v_neighbour_y4 = _mm512_i32gather_ps(v_indices_scaled4, np_ptr+1, 4);
        __m512 v_neighbour_z4 = _mm512_i32gather_ps(v_indices_scaled4, np_ptr+2, 4);

        __m512 v_diff_x1 = _mm512_sub_ps(v_point_x, v_neighbour_x1);
        __m512 v_results1 = _mm512_mul_ps(v_diff_x1, v_diff_x1);
        __m512 v_diff_x2 = _mm512_sub_ps(v_point_x, v_neighbour_x2);
        __m512 v_results2 = _mm512_mul_ps(v_diff_x2, v_diff_x2);
        __m512 v_diff_x3 = _mm512_sub_ps(v_point_x, v_neighbour_x3);
        __m512 v_results3 = _mm512_mul_ps(v_diff_x3, v_diff_x3);
        __m512 v_diff_x4 = _mm512_sub_ps(v_point_x, v_neighbour_x4);
        __m512 v_results4 = _mm512_mul_ps(v_diff_x4, v_diff_x4);

        __m512 v_diff_y1 = _mm512_sub_ps(v_point_y, v_neighbour_y1);
        __m512 v_results1_1 = _mm512_mul_ps(v_diff_y1, v_diff_y1);
        __m512 v_diff_y2 = _mm512_sub_ps(v_point_y, v_neighbour_y2);
        __m512 v_results2_1 = _mm512_mul_ps(v_diff_y2, v_diff_y2);
        __m512 v_diff_y3 = _mm512_sub_ps(v_point_y, v_neighbour_y3);
        __m512 v_results3_1 = _mm512_mul_ps(v_diff_y3, v_diff_y3);
        __m512 v_diff_y4 = _mm512_sub_ps(v_point_y, v_neighbour_y4);
        __m512 v_results4_1 = _mm512_mul_ps(v_diff_y4, v_diff_y4);

        __m512 v_diff_z1 = _mm512_sub_ps(v_point_z, v_neighbour_z1);
        __m512 v_results1_2 = _mm512_fmadd_ps(v_diff_z1, v_diff_z1, v_results1);
        __m512 v_diff_z2 = _mm512_sub_ps(v_point_z, v_neighbour_z2);
        __m512 v_results2_2 = _mm512_fmadd_ps(v_diff_z2, v_diff_z2, v_results2);
        __m512 v_diff_z3 = _mm512_sub_ps(v_point_z, v_neighbour_z3);
        __m512 v_results3_2 = _mm512_fmadd_ps(v_diff_z3, v_diff_z3, v_results3);
        __m512 v_diff_z4 = _mm512_sub_ps(v_point_z, v_neighbour_z4);
        __m512 v_results4_2 = _mm512_fmadd_ps(v_diff_z4, v_diff_z4, v_results4);

        v_results1 = _mm512_add_ps(v_results1_2, v_results1_1);
        v_results2 = _mm512_add_ps(v_results2_2, v_results2_1);
        v_results3 = _mm512_add_ps(v_results3_2, v_results3_1);
        v_results4 = _mm512_add_ps(v_results4_2, v_results4_1);

        // Mark entries where the result is less than epsilon^2
        __mmask16 v_eps_mask1 = _mm512_cmple_ps_mask( v_results1, v_eps);
        __mmask16 v_eps_mask2 = _mm512_cmple_ps_mask( v_results2, v_eps);
        __mmask16 v_eps_mask3 = _mm512_cmple_ps_mask( v_results3, v_eps);
        __mmask16 v_eps_mask4 = _mm512_cmple_ps_mask( v_results4, v_eps);

        count += _mm512_mask_reduce_add_epi32(v_eps_mask1, v_one);
        count += _mm512_mask_reduce_add_epi32(v_eps_mask2, v_one);
        count += _mm512_mask_reduce_add_epi32(v_eps_mask3, v_one);
        count += _mm512_mask_reduce_add_epi32(v_eps_mask4, v_one);
        // std::cout << "Count: " << count << std::endl;

        // Load cluster labels of close entries
        __m512i v_cluster_labels1 = _mm512_mask_i32gather_epi32(v_zero_epi32, v_eps_mask1, v_indices1, &clusters[0], 4);
        __m512i v_cluster_labels2 = _mm512_mask_i32gather_epi32(v_zero_epi32, v_eps_mask2, v_indices2, &clusters[0], 4);
        __m512i v_cluster_labels3 = _mm512_mask_i32gather_epi32(v_zero_epi32, v_eps_mask3, v_indices3, &clusters[0], 4);
        __m512i v_cluster_labels4 = _mm512_mask_i32gather_epi32(v_zero_epi32, v_eps_mask4, v_indices4, &clusters[0], 4);

        // filter labels that are not visited and that are less than zero
        __mmask16 v_less_than_zero1 = _mm512_cmplt_epi32_mask(v_cluster_labels1, v_zero_epi32);
        __mmask16 v_less_than_zero2 = _mm512_cmplt_epi32_mask(v_cluster_labels2, v_zero_epi32);
        __mmask16 v_less_than_zero3 = _mm512_cmplt_epi32_mask(v_cluster_labels3, v_zero_epi32);
        __mmask16 v_less_than_zero4 = _mm512_cmplt_epi32_mask(v_cluster_labels4, v_zero_epi32);

        v_cluster_labels1 = _mm512_abs_epi32(v_cluster_labels1);
        v_cluster_labels2 = _mm512_abs_epi32(v_cluster_labels2);
        v_cluster_labels3 = _mm512_abs_epi32(v_cluster_labels3);
        v_cluster_labels4 = _mm512_abs_epi32(v_cluster_labels4);

        // set min cluster label
        cluster_label = std::min(cluster_label, _mm512_mask_reduce_min_epi32(v_less_than_zero1, v_cluster_labels1));
        cluster_label = std::min(cluster_label, _mm512_mask_reduce_min_epi32(v_less_than_zero2, v_cluster_labels2));
        cluster_label = std::min(cluster_label, _mm512_mask_reduce_min_epi32(v_less_than_zero3, v_cluster_labels3));
        cluster_label = std::min(cluster_label, _mm512_mask_reduce_min_epi32(v_less_than_zero4, v_cluster_labels4));
        // std::cout << "cluster label: " << cluster_label << std::endl;

        // store min_points_area
        _mm512_mask_storeu_epi32(&min_points_area[i+elements_in_vector*0], v_eps_mask1, v_indices1);
        _mm512_mask_storeu_epi32(&min_points_area[i+elements_in_vector*1], v_eps_mask2, v_indices2);
        _mm512_mask_storeu_epi32(&min_points_area[i+elements_in_vector*2], v_eps_mask3, v_indices3);
        _mm512_mask_storeu_epi32(&min_points_area[i+elements_in_vector*3], v_eps_mask4, v_indices4);
    }
    for (std::size_t i = n-rest; i < n; i += elements_in_vector) {
#else
    for (std::size_t i = 0; i < n; i += elements_in_vector) {
#endif
        // Mask
        __mmask16 mask = ( n - i > ElementsPerAVX ) ? (1 << 16) - 1 : ((1 << ( n - i )) - 1);
        __m512i v_indices = _mm512_maskz_loadu_epi32(mask, &neighboring_points[i]);
        __m512i v_indices_scaled = _mm512_mullo_epi32(v_indices, v_dims);

        // ******* UNROLLED DIMENSION LOOP ************
        __m512 v_current_point_x = _mm512_set1_ps(point[0]);
        __m512 v_current_point_y = _mm512_set1_ps(point[1]);
        __m512 v_current_point_z = _mm512_set1_ps(point[2]);

        __m512 v_other_points_x = _mm512_mask_i32gather_ps(v_zero_ps, mask, v_indices_scaled, np_ptr+0, 4);
        __m512 v_other_points_y = _mm512_mask_i32gather_ps(v_zero_ps, mask, v_indices_scaled, np_ptr+1, 4);
        __m512 v_other_points_z = _mm512_mask_i32gather_ps(v_zero_ps, mask, v_indices_scaled, np_ptr+2, 4);
        
        __m512 v_diff_x = _mm512_sub_ps(v_current_point_x, v_other_points_x);
        __m512 v_results = _mm512_mul_ps(v_diff_x, v_diff_x);
        __m512 v_diff_y = _mm512_sub_ps(v_current_point_y, v_other_points_y);
        __m512 v_results1 = _mm512_mul_ps(v_diff_y, v_diff_y);
        __m512 v_diff_z = _mm512_sub_ps(v_current_point_z, v_other_points_z);
        __m512 v_results2 = _mm512_fmadd_ps(v_diff_z, v_diff_z, v_results);
        v_results = _mm512_add_ps(v_results2, v_results1);
        // ********************************************


        // Mark entries where the result is less than epsilon^2
        __mmask16 v_eps_mask = _mm512_mask_cmple_ps_mask(mask, v_results, v_eps);
        count += _mm512_mask_reduce_add_epi32(v_eps_mask, v_one);
        // std::cout << "Count: " << count << std::endl;

        // Load cluster labels of close entries
        __m512i v_cluster_labels = _mm512_mask_i32gather_epi32(v_zero_epi32, v_eps_mask, v_indices, &clusters[0], 4);

        // filter labels that are not visited and that are less than zero
        __mmask16 v_less_than_zero = _mm512_cmplt_epi32_mask(v_cluster_labels, v_zero_epi32);

        v_cluster_labels = _mm512_abs_epi32(v_cluster_labels);

        // set min cluster label
        cluster_label = std::min(cluster_label, _mm512_mask_reduce_min_epi32(v_less_than_zero, v_cluster_labels));
        // std::cout << "cluster label: " << cluster_label << std::endl;

        // store min_points_area
        _mm512_mask_storeu_epi32(&min_points_area[i], v_eps_mask, v_indices);
    }
    return cluster_label;
}

#endif

template<>
Cluster<index_type> SpatialIndex<float, index_type>::region_query_optimized(
        const index_type point_index,
        const std::vector<index_type>& neighboring_points,
        const float EPS2,
        const Clusters<index_type>& clusters,
        std::vector<index_type>& min_points_area,
        index_type& count) const {
    const std::size_t dimensions = static_cast<std::size_t>(m_data.m_chunk[1]);
#if defined(USE_ND_OPTIMIZATIONS)
    if (3 == dimensions)
    {
        return region_query_optimized_nd<3>(
                point_index,
                neighboring_points,
                EPS2,
                clusters,
                min_points_area,
                count
                );
    }
#endif
    const float* point = &m_data.m_elements[point_index * dimensions];
    const float* np_ptr = m_data.m_elements.data();
    const std::size_t precision = sizeof(float);
    Cluster<index_type> cluster_label = m_global_point_offset + point_index + 1;
    const std::size_t ElementsPerAVX = sizeof(__m512) / precision;

    //  Align memory (This has to be done while data loading, not here. )
    // Initialize array pointers and masks
    std::size_t npoints = neighboring_points.size();
    min_points_area = std::vector<index_type>(npoints, NOT_VISITED<index_type>);

    __m512 v_eps = _mm512_set1_ps(EPS2);
    // std::cout << "V_EPS: ";
    // print_m512(v_eps);

    __m512 v_zero_ps = _mm512_setzero_ps();
    // std::cout << "V_ZERO_PS: ";
    // print_m512(v_zero_ps);

    __m512i v_zero_epi32 = _mm512_setzero_epi32();
    // std::cout << "V_ZERO_EPI: ";
    // print_m512i(v_zero_epi32);

    __m512i v_one = _mm512_set1_epi32(1);
    // std::cout << "V_ONE: ";
    // print_m512i(v_one);

    __m512i v_dims = _mm512_set1_epi32(dimensions);
    // std::cout << "V_DIMS: ";
    // print_m512i(v_dims);

    for (std::size_t i = 0; i < npoints; i += ElementsPerAVX){
        // Mask
        __mmask16 mask = ( npoints - i > ElementsPerAVX ) ? (1 << 16) - 1 : ((1 << ( npoints - i )) - 1);
        // Load index into mm512
        __m512i v_indices = _mm512_maskz_loadu_epi32(mask, &neighboring_points[i]);
        // std::cout << "V_INDICES: ";
        // print_m512i(v_indices);
        __m512i v_indices_scaled = _mm512_mullo_epi32(v_indices, v_dims);
        // std::cout << "V_INDICES_SCALED: ";
        // print_m512i(v_indices_scaled);

        // Create result
        __m512 v_results = _mm512_setzero_ps();

        // Loop over dimensions
        for (std::size_t d = 0; d < dimensions; d++){
            // Load dimension d of current point
            __m512 v_current_point_dim = _mm512_set1_ps(point[d]);
            // std::cout << "V_CURRENT_POINT_DIM: ";
            // print_m512(v_current_point_dim);
            // Scale indices
            // Load dimension d of other points (shifting base address to match current dimension, and scaling the indices by number of dimensions)
            __m512 v_other_points_dim = _mm512_mask_i32gather_ps(v_zero_ps, mask, v_indices_scaled, np_ptr + d, 4);

            // Get the diff
            __m512 v_diff = _mm512_sub_ps(v_current_point_dim, v_other_points_dim);
            // std::cout << "V_DIFF: ";
            // print_m512(v_diff);

            // Square that diff
            __m512 v_sqrd = _mm512_mul_ps(v_diff, v_diff);
            // std::cout << "V_SQRD: ";
            // print_m512(v_sqrd);

            // Add to results
            v_results = _mm512_add_ps(v_results, v_sqrd);   
        }

        // Mark entries where the result is less than epsilon^2
        __mmask16 v_eps_mask = _mm512_mask_cmple_ps_mask(mask, v_results, v_eps);
        count += _mm512_mask_reduce_add_epi32(v_eps_mask, v_one);
        // std::cout << "Count: " << count << std::endl;

        // Load cluster labels of close entries
        __m512i v_cluster_labels = _mm512_mask_i32gather_epi32(v_zero_epi32, v_eps_mask, v_indices, &clusters[0], 4);

        // filter labels that are not visited and that are less than zero
        __mmask16 v_less_than_zero = _mm512_cmplt_epi32_mask(v_cluster_labels, v_zero_epi32);

        v_cluster_labels = _mm512_abs_epi32(v_cluster_labels);

        // set min cluster label
        cluster_label = std::min(cluster_label, _mm512_mask_reduce_min_epi32(v_less_than_zero, v_cluster_labels));
        // std::cout << "cluster label: " << cluster_label << std::endl;

        // store min_points_area
        _mm512_mask_storeu_epi32(&min_points_area[i], v_eps_mask, v_indices);
    }
    return cluster_label;
}
