#include <immintrin.h>

#define IMPLEMENTING_OPTIMIZATION

#include "spatial_index.h"

#include <bit>

typedef std::int64_t index_type;

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
    Cluster<index_type> cluster_label = m_global_point_offset + point_index + 1;
    constexpr std::size_t elements_in_vector = sizeof(__m512)/sizeof(float);
    constexpr std::size_t indices_in_vector = sizeof(__m512i)/sizeof(index_type);

    //  Align memory (This has to be done while data loading, not here. )
    // Initialize array pointers and masks
    std::size_t n = neighboring_points.size();
    min_points_area = std::vector<index_type>(n, NOT_VISITED<index_type>);

    __m512 v_eps = _mm512_set1_ps(EPS2);
    __m512 v_zero_ps = _mm512_setzero_ps();
    __m256 v_zero_ps256 = _mm256_setzero_ps();
    __m512i v_zero_epi64 = _mm512_setzero_si512();
    __m512i v_one_epi32 = _mm512_set1_epi32(1);
    __m512i v_dims = _mm512_set1_epi64(dimensions);

    __m512 v_point_x = _mm512_set1_ps(point[0]);
    __m512 v_point_y = _mm512_set1_ps(point[1]);
    __m512 v_point_z = _mm512_set1_ps(point[2]);

    #if defined(USE_4X_NEIGHBOUR_LOOP_UNROLL)
    constexpr std::size_t vectors_per_loop = 4;
    std::size_t loop_elements = elements_in_vector*vectors_per_loop;
    std::size_t rest = n % loop_elements;
    std::size_t chunks = n/loop_elements;

    for (std::size_t j = 0; j < chunks; j++) {
        std::size_t i = j*loop_elements;
        // No need for masks in unrolled loop 
        __m512i v_indices1_1 = _mm512_loadu_epi64(&neighboring_points[i+indices_in_vector*0]);
        __m512i v_indices1_2 = _mm512_loadu_epi64(&neighboring_points[i+indices_in_vector*1]);
        __m512i v_indices2_1 = _mm512_loadu_epi64(&neighboring_points[i+indices_in_vector*2]);
        __m512i v_indices2_2 = _mm512_loadu_epi64(&neighboring_points[i+indices_in_vector*3]);
        __m512i v_indices3_1 = _mm512_loadu_epi64(&neighboring_points[i+indices_in_vector*4]);
        __m512i v_indices3_2 = _mm512_loadu_epi64(&neighboring_points[i+indices_in_vector*5]);
        __m512i v_indices4_1 = _mm512_loadu_epi64(&neighboring_points[i+indices_in_vector*6]);
        __m512i v_indices4_2 = _mm512_loadu_epi64(&neighboring_points[i+indices_in_vector*7]);

        __m512i v_indices_scaled1_1 = _mm512_mullo_epi64(v_indices1_1, v_dims);
        __m512i v_indices_scaled1_2 = _mm512_mullo_epi64(v_indices1_2, v_dims);
        __m512i v_indices_scaled2_1 = _mm512_mullo_epi64(v_indices2_1, v_dims);
        __m512i v_indices_scaled2_2 = _mm512_mullo_epi64(v_indices2_2, v_dims);
        __m512i v_indices_scaled3_1 = _mm512_mullo_epi64(v_indices3_1, v_dims);
        __m512i v_indices_scaled3_2 = _mm512_mullo_epi64(v_indices3_2, v_dims);
        __m512i v_indices_scaled4_1 = _mm512_mullo_epi64(v_indices4_1, v_dims);
        __m512i v_indices_scaled4_2 = _mm512_mullo_epi64(v_indices4_2, v_dims);

        __m512 v_neighbour_x1 = _mm512_castps256_ps512(_mm512_i64gather_ps(v_indices_scaled1_1, np_ptr+0, 4));
        v_neighbour_x1 = _mm512_insertf32x8(v_neighbour_x1,_mm512_i64gather_ps(v_indices_scaled1_2, np_ptr+0, 4),1);
        __m512 v_neighbour_y1 = _mm512_castps256_ps512(_mm512_i64gather_ps(v_indices_scaled1_1, np_ptr+1, 4));
        v_neighbour_y1 = _mm512_insertf32x8(v_neighbour_y1,_mm512_i64gather_ps(v_indices_scaled1_2, np_ptr+1, 4),1);
        __m512 v_neighbour_z1 = _mm512_castps256_ps512(_mm512_i64gather_ps(v_indices_scaled1_1, np_ptr+2, 4));
        v_neighbour_z1 = _mm512_insertf32x8(v_neighbour_z1,_mm512_i64gather_ps(v_indices_scaled1_2, np_ptr+2, 4),1);

        __m512 v_neighbour_x2 = _mm512_castps256_ps512(_mm512_i64gather_ps(v_indices_scaled2_1, np_ptr+0, 4));
        v_neighbour_x2 = _mm512_insertf32x8(v_neighbour_x2,_mm512_i64gather_ps(v_indices_scaled2_2, np_ptr+0, 4),1);
        __m512 v_neighbour_y2 = _mm512_castps256_ps512(_mm512_i64gather_ps(v_indices_scaled2_1, np_ptr+1, 4));
        v_neighbour_y2 = _mm512_insertf32x8(v_neighbour_y2,_mm512_i64gather_ps(v_indices_scaled2_2, np_ptr+1, 4),1);
        __m512 v_neighbour_z2 = _mm512_castps256_ps512(_mm512_i64gather_ps(v_indices_scaled2_1, np_ptr+2, 4));
        v_neighbour_z2 = _mm512_insertf32x8(v_neighbour_z2,_mm512_i64gather_ps(v_indices_scaled2_2, np_ptr+2, 4),1);

        __m512 v_neighbour_x3 = _mm512_castps256_ps512(_mm512_i64gather_ps(v_indices_scaled3_1, np_ptr+0, 4));
        v_neighbour_x3 = _mm512_insertf32x8(v_neighbour_x3,_mm512_i64gather_ps(v_indices_scaled3_2, np_ptr+0, 4),1);
        __m512 v_neighbour_y3 = _mm512_castps256_ps512(_mm512_i64gather_ps(v_indices_scaled3_1, np_ptr+1, 4));
        v_neighbour_y3 = _mm512_insertf32x8(v_neighbour_y3,_mm512_i64gather_ps(v_indices_scaled3_2, np_ptr+1, 4),1);
        __m512 v_neighbour_z3 = _mm512_castps256_ps512(_mm512_i64gather_ps(v_indices_scaled3_1, np_ptr+2, 4));
        v_neighbour_z3 = _mm512_insertf32x8(v_neighbour_z3,_mm512_i64gather_ps(v_indices_scaled3_2, np_ptr+2, 4),1);

        __m512 v_neighbour_x4 = _mm512_castps256_ps512(_mm512_i64gather_ps(v_indices_scaled4_1, np_ptr+0, 4));
        v_neighbour_x4 = _mm512_insertf32x8(v_neighbour_x4,_mm512_i64gather_ps(v_indices_scaled4_2, np_ptr+0, 4),1);
        __m512 v_neighbour_y4 = _mm512_castps256_ps512(_mm512_i64gather_ps(v_indices_scaled4_1, np_ptr+1, 4));
        v_neighbour_y4 = _mm512_insertf32x8(v_neighbour_y4,_mm512_i64gather_ps(v_indices_scaled4_2, np_ptr+1, 4),1);
        __m512 v_neighbour_z4 = _mm512_castps256_ps512(_mm512_i64gather_ps(v_indices_scaled4_1, np_ptr+2, 4));
        v_neighbour_z4 = _mm512_insertf32x8(v_neighbour_z4,_mm512_i64gather_ps(v_indices_scaled4_2, np_ptr+2, 4),1);

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

        count += _mm_popcnt_u32(_cvtmask16_u32(v_eps_mask1));
        count += _mm_popcnt_u32(_cvtmask16_u32(v_eps_mask2));
        count += _mm_popcnt_u32(_cvtmask16_u32(v_eps_mask3));
        count += _mm_popcnt_u32(_cvtmask16_u32(v_eps_mask4));
        // std::cout << "Count: " << count << std::endl;

        // Load cluster labels of close entries
        __m512i v_cluster_labels1_1 = _mm512_mask_i64gather_epi64(
                v_zero_epi64, v_eps_mask1, v_indices1_1, &clusters[0], 8);
        __m512i v_cluster_labels1_2 = _mm512_mask_i64gather_epi64(
                v_zero_epi64, _kshiftri_mask16(v_eps_mask1,8), v_indices1_2, &clusters[0], 8);

        __m512i v_cluster_labels2_1 = _mm512_mask_i64gather_epi64(
                v_zero_epi64, v_eps_mask2, v_indices2_1, &clusters[0], 8);
        __m512i v_cluster_labels2_2 = _mm512_mask_i64gather_epi64(
                v_zero_epi64, _kshiftri_mask16(v_eps_mask2,8), v_indices2_2, &clusters[0], 8);

        __m512i v_cluster_labels3_1 = _mm512_mask_i64gather_epi64(
                v_zero_epi64, v_eps_mask3, v_indices3_1, &clusters[0], 8);
        __m512i v_cluster_labels3_2 = _mm512_mask_i64gather_epi64(
                v_zero_epi64, _kshiftri_mask16(v_eps_mask3,8), v_indices3_2, &clusters[0], 8);

        __m512i v_cluster_labels4_1 = _mm512_mask_i64gather_epi64(
                v_zero_epi64, v_eps_mask4, v_indices4_1, &clusters[0], 8);
        __m512i v_cluster_labels4_2 = _mm512_mask_i64gather_epi64(
                v_zero_epi64, _kshiftri_mask16(v_eps_mask4,8), v_indices4_2, &clusters[0], 8);

        // filter labels that are not visited and that are less than zero
        __mmask8 v_less_than_zero1_1 = _mm512_cmplt_epi64_mask(v_cluster_labels1_1, v_zero_epi64);
        __mmask8 v_less_than_zero1_2 = _mm512_cmplt_epi64_mask(v_cluster_labels1_2, v_zero_epi64);
        __mmask8 v_less_than_zero2_1 = _mm512_cmplt_epi64_mask(v_cluster_labels2_1, v_zero_epi64);
        __mmask8 v_less_than_zero2_2 = _mm512_cmplt_epi64_mask(v_cluster_labels2_2, v_zero_epi64);
        __mmask8 v_less_than_zero3_1 = _mm512_cmplt_epi64_mask(v_cluster_labels3_1, v_zero_epi64);
        __mmask8 v_less_than_zero3_2 = _mm512_cmplt_epi64_mask(v_cluster_labels3_2, v_zero_epi64);
        __mmask8 v_less_than_zero4_1 = _mm512_cmplt_epi64_mask(v_cluster_labels4_1, v_zero_epi64);
        __mmask8 v_less_than_zero4_2 = _mm512_cmplt_epi64_mask(v_cluster_labels4_2, v_zero_epi64);

        v_cluster_labels1_1 = _mm512_abs_epi64(v_cluster_labels1_1);
        v_cluster_labels1_2 = _mm512_abs_epi64(v_cluster_labels1_2);
        v_cluster_labels2_1 = _mm512_abs_epi64(v_cluster_labels2_1);
        v_cluster_labels2_2 = _mm512_abs_epi64(v_cluster_labels2_2);
        v_cluster_labels3_1 = _mm512_abs_epi64(v_cluster_labels3_1);
        v_cluster_labels3_2 = _mm512_abs_epi64(v_cluster_labels3_2);
        v_cluster_labels4_1 = _mm512_abs_epi64(v_cluster_labels4_1);
        v_cluster_labels4_2 = _mm512_abs_epi64(v_cluster_labels4_2);

        // set min cluster label
        cluster_label = std::min(cluster_label, static_cast<index_type>(_mm512_mask_reduce_min_epi64(v_less_than_zero1_1, v_cluster_labels1_1)));
        cluster_label = std::min(cluster_label, static_cast<index_type>(_mm512_mask_reduce_min_epi64(v_less_than_zero1_2, v_cluster_labels1_2)));
        cluster_label = std::min(cluster_label, static_cast<index_type>(_mm512_mask_reduce_min_epi64(v_less_than_zero2_1, v_cluster_labels2_1)));
        cluster_label = std::min(cluster_label, static_cast<index_type>(_mm512_mask_reduce_min_epi64(v_less_than_zero2_2, v_cluster_labels2_2)));
        cluster_label = std::min(cluster_label, static_cast<index_type>(_mm512_mask_reduce_min_epi64(v_less_than_zero3_1, v_cluster_labels3_1)));
        cluster_label = std::min(cluster_label, static_cast<index_type>(_mm512_mask_reduce_min_epi64(v_less_than_zero3_2, v_cluster_labels3_2)));
        cluster_label = std::min(cluster_label, static_cast<index_type>(_mm512_mask_reduce_min_epi64(v_less_than_zero4_1, v_cluster_labels4_1)));
        cluster_label = std::min(cluster_label, static_cast<index_type>(_mm512_mask_reduce_min_epi64(v_less_than_zero4_2, v_cluster_labels4_2)));
        // std::cout << "cluster label: " << cluster_label << std::endl;

        // store min_points_area
        _mm512_mask_storeu_epi64(&min_points_area[i+indices_in_vector*0], v_eps_mask1, v_indices1_1);
        _mm512_mask_storeu_epi64(&min_points_area[i+indices_in_vector*1], _kshiftri_mask16(v_eps_mask1,8), v_indices1_2);
        _mm512_mask_storeu_epi64(&min_points_area[i+indices_in_vector*2], v_eps_mask2, v_indices2_1);
        _mm512_mask_storeu_epi64(&min_points_area[i+indices_in_vector*3], _kshiftri_mask16(v_eps_mask2,8), v_indices2_2);
        _mm512_mask_storeu_epi64(&min_points_area[i+indices_in_vector*4], v_eps_mask3, v_indices3_1);
        _mm512_mask_storeu_epi64(&min_points_area[i+indices_in_vector*5], _kshiftri_mask16(v_eps_mask3,8), v_indices3_2);
        _mm512_mask_storeu_epi64(&min_points_area[i+indices_in_vector*6], v_eps_mask4, v_indices4_1);
        _mm512_mask_storeu_epi64(&min_points_area[i+indices_in_vector*7], _kshiftri_mask16(v_eps_mask4,8), v_indices4_2);
    }
    for (std::size_t i = n-rest; i < n; i += elements_in_vector) {
#else
    for (std::size_t i = 0; i < n; i += elements_in_vector) {
#endif
        // Example:
        // 16 elements(float) in vector (512/32)
        // 8 indices(int64) in vector (512/64)
        //
        // 
        // 13 elements left, then
        // emask  = 0b0001111111111111
        // imask1 = 0b11111111
        // imask2 = 0b00011111
        __mmask16 emask = ( n - i > elements_in_vector ) ? (1 << 16) - 1 : ((1 << ( n - i )) - 1);
        __mmask8 imask1 = emask;
        __mmask8 imask2 = _kshiftri_mask16(emask,8);

        __m512i v_indices1_1 = _mm512_maskz_loadu_epi64(imask1,&neighboring_points[i+indices_in_vector*0]);
        __m512i v_indices1_2 = _mm512_maskz_loadu_epi64(imask2,&neighboring_points[i+indices_in_vector*1]);

        __m512i v_indices_scaled1_1 = _mm512_mullo_epi64(v_indices1_1, v_dims);
        __m512i v_indices_scaled1_2 = _mm512_mullo_epi64(v_indices1_2, v_dims);

        __m512 v_neighbour_x1 = _mm512_castps256_ps512(_mm512_mask_i64gather_ps(v_zero_ps256, imask1, v_indices_scaled1_1, np_ptr+0, 4));
        v_neighbour_x1 = _mm512_insertf32x8(v_neighbour_x1,_mm512_mask_i64gather_ps(v_zero_ps256, imask2, v_indices_scaled1_2, np_ptr+0, 4),1);
        __m512 v_neighbour_y1 = _mm512_castps256_ps512(_mm512_mask_i64gather_ps(v_zero_ps256, imask1, v_indices_scaled1_1, np_ptr+1, 4));
        v_neighbour_y1 = _mm512_insertf32x8(v_neighbour_y1,_mm512_mask_i64gather_ps(v_zero_ps256, imask2, v_indices_scaled1_2, np_ptr+1, 4),1);
        __m512 v_neighbour_z1 = _mm512_castps256_ps512(_mm512_mask_i64gather_ps(v_zero_ps256, imask1, v_indices_scaled1_1, np_ptr+2, 4));
        v_neighbour_z1 = _mm512_insertf32x8(v_neighbour_z1,_mm512_mask_i64gather_ps(v_zero_ps256, imask2, v_indices_scaled1_2, np_ptr+2, 4),1);

        __m512 v_diff_x1 = _mm512_sub_ps(v_point_x, v_neighbour_x1);
        __m512 v_results1 = _mm512_mul_ps(v_diff_x1, v_diff_x1);

        __m512 v_diff_y1 = _mm512_sub_ps(v_point_y, v_neighbour_y1);
        __m512 v_results1_1 = _mm512_mul_ps(v_diff_y1, v_diff_y1);

        __m512 v_diff_z1 = _mm512_sub_ps(v_point_z, v_neighbour_z1);
        __m512 v_results1_2 = _mm512_fmadd_ps(v_diff_z1, v_diff_z1, v_results1);

        v_results1 = _mm512_add_ps(v_results1_2, v_results1_1);

        // Mark entries where the result is less than epsilon^2
        __mmask16 v_eps_mask1 = _mm512_mask_cmple_ps_mask(emask, v_results1, v_eps);

        //count += _mm512_mask_reduce_add_epi32(v_eps_mask1, v_one_epi32);
        // std::cout << "Count: " << count << std::endl;
        count += _mm_popcnt_u32(_cvtmask16_u32(v_eps_mask1));

        // Load cluster labels of close entries
        __m512i v_cluster_labels1_1 = _mm512_mask_i64gather_epi64(
                v_zero_epi64, v_eps_mask1, v_indices1_1, &clusters[0], 8);
        __m512i v_cluster_labels1_2 = _mm512_mask_i64gather_epi64(
                v_zero_epi64, _kshiftri_mask16(v_eps_mask1,8), v_indices1_2, &clusters[0], 8);

        // filter labels that are not visited and that are less than zero
        __mmask8 v_less_than_zero1_1 = _mm512_cmplt_epi64_mask(v_cluster_labels1_1, v_zero_epi64);
        __mmask8 v_less_than_zero1_2 = _mm512_cmplt_epi64_mask(v_cluster_labels1_2, v_zero_epi64);

        v_cluster_labels1_1 = _mm512_abs_epi64(v_cluster_labels1_1);
        v_cluster_labels1_2 = _mm512_abs_epi64(v_cluster_labels1_2);

        // (intrinsic returns a long long int, while std::int64_t is a long int)
        cluster_label = std::min(cluster_label, static_cast<index_type>(_mm512_mask_reduce_min_epi64(v_less_than_zero1_1, v_cluster_labels1_1)));
        cluster_label = std::min(cluster_label, static_cast<index_type>(_mm512_mask_reduce_min_epi64(v_less_than_zero1_2, v_cluster_labels1_2)));
        // std::cout << "cluster label: " << cluster_label << std::endl;

        // store min_points_area
        _mm512_mask_storeu_epi64(&min_points_area[i+indices_in_vector*0], v_eps_mask1, v_indices1_1);
        _mm512_mask_storeu_epi64(&min_points_area[i+indices_in_vector*1], _kshiftri_mask16(v_eps_mask1,8), v_indices1_2);

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
    constexpr std::size_t elements_in_vector = sizeof(__m512)/sizeof(float);
    constexpr std::size_t indices_in_vector = sizeof(__m512i)/sizeof(index_type);

    //  Align memory (This has to be done while data loading, not here. )
    // Initialize array pointers and masks
    std::size_t n = neighboring_points.size();
    min_points_area = std::vector<index_type>(n, NOT_VISITED<index_type>);

    __m512 v_eps = _mm512_set1_ps(EPS2);
    __m512 v_zero_ps = _mm512_setzero_ps();
    __m256 v_zero_ps256 = _mm256_setzero_ps();
    __m512i v_zero_epi64 = _mm512_setzero_si512();
    __m512i v_one_epi32 = _mm512_set1_epi32(1);
    __m512i v_dims = _mm512_set1_epi64(dimensions);

    __m512 v_point_x = _mm512_set1_ps(point[0]);
    __m512 v_point_y = _mm512_set1_ps(point[1]);
    __m512 v_point_z = _mm512_set1_ps(point[2]);
    for (std::size_t i = 0; i < n; i += elements_in_vector) {
        // Example:
        // 16 elements(float) in vector (512/32)
        // 8 indices(int64) in vector (512/64)
        //
        // 
        // 13 elements left, then
        // emask  = 0b0001111111111111
        // imask1 = 0b11111111
        // imask2 = 0b00011111
        __mmask16 emask = ( n - i > elements_in_vector ) ? (1 << 16) - 1 : ((1 << ( n - i )) - 1);
        __mmask8 imask1 = emask;
        __mmask8 imask2 = _kshiftri_mask16(emask, 8);

        __m512i v_indices1_1 = _mm512_maskz_loadu_epi64(imask1,&neighboring_points[i+indices_in_vector*0]);
        __m512i v_indices1_2 = _mm512_maskz_loadu_epi64(imask2,&neighboring_points[i+indices_in_vector*1]);

        __m512i v_indices_scaled1_1 = _mm512_mullo_epi64(v_indices1_1, v_dims);
        __m512i v_indices_scaled1_2 = _mm512_mullo_epi64(v_indices1_2, v_dims);

        __m512 v_results = _mm512_setzero_ps();
        // Loop over dimensions
        for (std::size_t d = 0; d < dimensions; d++){
            __m512 v_current_point_dim = _mm512_set1_ps(point[d]);

            __m512 v_neighbour_dim = _mm512_castps256_ps512(
                    _mm512_mask_i64gather_ps(
                        v_zero_ps256,
                        imask1,
                        v_indices_scaled1_1,
                        np_ptr+d, 4));
            v_neighbour_dim = _mm512_insertf32x8(v_neighbour_dim,
                    _mm512_mask_i64gather_ps(
                        v_zero_ps256,
                        imask2,
                        v_indices_scaled1_2,
                        np_ptr+d, 4),1);

            // Get the diff
            __m512 v_diff = _mm512_sub_ps(v_current_point_dim, v_neighbour_dim);
            v_results = _mm512_fmadd_ps(v_diff, v_diff, v_results);   
        }

        // Mark entries where the result is less than epsilon^2
        __mmask16 v_eps_mask1 = _mm512_mask_cmple_ps_mask(emask, v_results, v_eps);

        //count += _mm512_mask_reduce_add_epi32(v_eps_mask1, v_one_epi32);
        // std::cout << "Count: " << count << std::endl;
        count += _mm_popcnt_u32(_cvtmask16_u32(v_eps_mask1));

        // Load cluster labels of close entries
        __m512i v_cluster_labels1_1 = _mm512_mask_i64gather_epi64(
                v_zero_epi64, v_eps_mask1, v_indices1_1, &clusters[0], 8);
        __m512i v_cluster_labels1_2 = _mm512_mask_i64gather_epi64(
                v_zero_epi64, _kshiftri_mask16(v_eps_mask1,8), v_indices1_2, &clusters[0], 8);

        // filter labels that are not visited and that are less than zero
        __mmask8 v_less_than_zero1_1 = _mm512_cmplt_epi64_mask(v_cluster_labels1_1, v_zero_epi64);
        __mmask8 v_less_than_zero1_2 = _mm512_cmplt_epi64_mask(v_cluster_labels1_2, v_zero_epi64);

        v_cluster_labels1_1 = _mm512_abs_epi64(v_cluster_labels1_1);
        v_cluster_labels1_2 = _mm512_abs_epi64(v_cluster_labels1_2);

        // set min cluster label
        cluster_label = std::min(cluster_label, static_cast<index_type>(_mm512_mask_reduce_min_epi64(v_less_than_zero1_1, v_cluster_labels1_1)));
        cluster_label = std::min(cluster_label, static_cast<index_type>(_mm512_mask_reduce_min_epi64(v_less_than_zero1_2, v_cluster_labels1_2)));
        // std::cout << "cluster label: " << cluster_label << std::endl;

        // store min_points_area
        _mm512_mask_storeu_epi64(&min_points_area[i+indices_in_vector*0], v_eps_mask1, v_indices1_1);
        _mm512_mask_storeu_epi64(&min_points_area[i+indices_in_vector*1], _kshiftri_mask16(v_eps_mask1,8), v_indices1_2);

    }
    return cluster_label;
}
