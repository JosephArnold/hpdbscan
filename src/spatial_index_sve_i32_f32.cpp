#ifdef __ARM_FEATURE_SVE

#include <arm_sve.h>

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

    constexpr std::size_t dimensions = 3;

    const float* point = &m_data.m_elements[point_index * dimensions];
    const float* np_ptr = m_data.m_elements.data();

    Cluster<index_type> cluster_label = m_global_point_offset + point_index + 1;

    size_t n = neighboring_points.size();

    min_points_area = std::vector<index_type>(n, NOT_VISITED<index_type>);


#if defined(USE_8X_NEIGHBOUR_LOOP_UNROLL)
    constexpr size_t vectors_per_loop = 8;
    size_t elements_in_vector = svcntw();
    size_t loop_elements = elements_in_vector*vectors_per_loop;
    size_t rest = n % loop_elements;
    size_t chunks = n/loop_elements;

    for (size_t j = 0; j < chunks; j++) {
        size_t i = j*loop_elements;

        // no masking for unrolled loop
        // svbool_t pg = svwhilelt_b32(i, n);
        svbool_t pg = svptrue_b32();

        svint32_t sv_indices1 = svld1_s32(pg, &neighboring_points[i]);
        svint32_t sv_indices2 = svld1_s32(pg, &neighboring_points[i+1*elements_in_vector]);
        svint32_t sv_indices3 = svld1_s32(pg, &neighboring_points[i+2*elements_in_vector]);
        svint32_t sv_indices4 = svld1_s32(pg, &neighboring_points[i+3*elements_in_vector]);
        svint32_t sv_indices5 = svld1_s32(pg, &neighboring_points[i+4*elements_in_vector]);
        svint32_t sv_indices6 = svld1_s32(pg, &neighboring_points[i+5*elements_in_vector]);
        svint32_t sv_indices7 = svld1_s32(pg, &neighboring_points[i+6*elements_in_vector]);
        svint32_t sv_indices8 = svld1_s32(pg, &neighboring_points[i+7*elements_in_vector]);

        svint32_t sv_indices_scaled1 = svmul_n_s32_z(pg, sv_indices1, dimensions);
        svint32_t sv_indices_scaled2 = svmul_n_s32_z(pg, sv_indices2, dimensions);
        svint32_t sv_indices_scaled3 = svmul_n_s32_z(pg, sv_indices3, dimensions);
        svint32_t sv_indices_scaled4 = svmul_n_s32_z(pg, sv_indices4, dimensions);
        svint32_t sv_indices_scaled5 = svmul_n_s32_z(pg, sv_indices5, dimensions);
        svint32_t sv_indices_scaled6 = svmul_n_s32_z(pg, sv_indices6, dimensions);
        svint32_t sv_indices_scaled7 = svmul_n_s32_z(pg, sv_indices7, dimensions);
        svint32_t sv_indices_scaled8 = svmul_n_s32_z(pg, sv_indices8, dimensions);

        svfloat32_t v_point_x = svdup_n_f32(point[0]);
        svfloat32_t v_point_y = svdup_n_f32(point[1]);
        svfloat32_t v_point_z = svdup_n_f32(point[2]);

        svint32_t other_point_index1 = svadd_n_s32_z(pg, sv_indices_scaled1, 0);
        svint32_t other_point_index2 = svadd_n_s32_z(pg, sv_indices_scaled2, 0);
        svint32_t other_point_index3 = svadd_n_s32_z(pg, sv_indices_scaled3, 0);
        svint32_t other_point_index4 = svadd_n_s32_z(pg, sv_indices_scaled4, 0);
        svint32_t other_point_index5 = svadd_n_s32_z(pg, sv_indices_scaled5, 0);
        svint32_t other_point_index6 = svadd_n_s32_z(pg, sv_indices_scaled6, 0);
        svint32_t other_point_index7 = svadd_n_s32_z(pg, sv_indices_scaled7, 0);
        svint32_t other_point_index8 = svadd_n_s32_z(pg, sv_indices_scaled8, 0);

        svfloat32_t other_point_x1 = svld1_gather_s32index_f32(pg, &np_ptr[0]+0, other_point_index1);
        svfloat32_t other_point_x2 = svld1_gather_s32index_f32(pg, &np_ptr[0]+0, other_point_index2);
        svfloat32_t other_point_x3 = svld1_gather_s32index_f32(pg, &np_ptr[0]+0, other_point_index3);
        svfloat32_t other_point_x4 = svld1_gather_s32index_f32(pg, &np_ptr[0]+0, other_point_index4);
        svfloat32_t other_point_x5 = svld1_gather_s32index_f32(pg, &np_ptr[0]+0, other_point_index5);
        svfloat32_t other_point_x6 = svld1_gather_s32index_f32(pg, &np_ptr[0]+0, other_point_index6);
        svfloat32_t other_point_x7 = svld1_gather_s32index_f32(pg, &np_ptr[0]+0, other_point_index7);
        svfloat32_t other_point_x8 = svld1_gather_s32index_f32(pg, &np_ptr[0]+0, other_point_index8);

        svfloat32_t other_point_y1 = svld1_gather_s32index_f32(pg, &np_ptr[0]+1, other_point_index1);
        svfloat32_t other_point_y2 = svld1_gather_s32index_f32(pg, &np_ptr[0]+1, other_point_index2);
        svfloat32_t other_point_y3 = svld1_gather_s32index_f32(pg, &np_ptr[0]+1, other_point_index3);
        svfloat32_t other_point_y4 = svld1_gather_s32index_f32(pg, &np_ptr[0]+1, other_point_index4);
        svfloat32_t other_point_y5 = svld1_gather_s32index_f32(pg, &np_ptr[0]+1, other_point_index5);
        svfloat32_t other_point_y6 = svld1_gather_s32index_f32(pg, &np_ptr[0]+1, other_point_index6);
        svfloat32_t other_point_y7 = svld1_gather_s32index_f32(pg, &np_ptr[0]+1, other_point_index7);
        svfloat32_t other_point_y8 = svld1_gather_s32index_f32(pg, &np_ptr[0]+1, other_point_index8);

        svfloat32_t other_point_z1 = svld1_gather_s32index_f32(pg, &np_ptr[0]+2, other_point_index1);
        svfloat32_t other_point_z2 = svld1_gather_s32index_f32(pg, &np_ptr[0]+2, other_point_index2);
        svfloat32_t other_point_z3 = svld1_gather_s32index_f32(pg, &np_ptr[0]+2, other_point_index3);
        svfloat32_t other_point_z4 = svld1_gather_s32index_f32(pg, &np_ptr[0]+2, other_point_index4);
        svfloat32_t other_point_z5 = svld1_gather_s32index_f32(pg, &np_ptr[0]+2, other_point_index5);
        svfloat32_t other_point_z6 = svld1_gather_s32index_f32(pg, &np_ptr[0]+2, other_point_index6);
        svfloat32_t other_point_z7 = svld1_gather_s32index_f32(pg, &np_ptr[0]+2, other_point_index7);
        svfloat32_t other_point_z8 = svld1_gather_s32index_f32(pg, &np_ptr[0]+2, other_point_index8);

        svfloat32_t v_diff_x1 = svsub_f32_x(pg, other_point_x1, v_point_x);
        svfloat32_t v_result1 = svmul_f32_x(pg, v_diff_x1, v_diff_x1);
        svfloat32_t v_diff_x2 = svsub_f32_x(pg, other_point_x2, v_point_x);
        svfloat32_t v_result2 = svmul_f32_x(pg, v_diff_x2, v_diff_x2);
        svfloat32_t v_diff_x3 = svsub_f32_x(pg, other_point_x3, v_point_x);
        svfloat32_t v_result3 = svmul_f32_x(pg, v_diff_x3, v_diff_x3);
        svfloat32_t v_diff_x4 = svsub_f32_x(pg, other_point_x4, v_point_x);
        svfloat32_t v_result4 = svmul_f32_x(pg, v_diff_x4, v_diff_x4);
        svfloat32_t v_diff_x5 = svsub_f32_x(pg, other_point_x5, v_point_x);
        svfloat32_t v_result5 = svmul_f32_x(pg, v_diff_x5, v_diff_x5);
        svfloat32_t v_diff_x6 = svsub_f32_x(pg, other_point_x6, v_point_x);
        svfloat32_t v_result6 = svmul_f32_x(pg, v_diff_x6, v_diff_x6);
        svfloat32_t v_diff_x7 = svsub_f32_x(pg, other_point_x7, v_point_x);
        svfloat32_t v_result7 = svmul_f32_x(pg, v_diff_x7, v_diff_x7);
        svfloat32_t v_diff_x8 = svsub_f32_x(pg, other_point_x8, v_point_x);
        svfloat32_t v_result8 = svmul_f32_x(pg, v_diff_x8, v_diff_x8);

        svfloat32_t v_diff_y1 = svsub_f32_x(pg, other_point_y1, v_point_y);
        svfloat32_t v_result1_1 = svmul_f32_x(pg, v_diff_y1, v_diff_y1);
        svfloat32_t v_diff_y2 = svsub_f32_x(pg, other_point_y2, v_point_y);
        svfloat32_t v_result2_1 = svmul_f32_x(pg, v_diff_y2, v_diff_y2);
        svfloat32_t v_diff_y3 = svsub_f32_x(pg, other_point_y3, v_point_y);
        svfloat32_t v_result3_1 = svmul_f32_x(pg, v_diff_y3, v_diff_y3);
        svfloat32_t v_diff_y4 = svsub_f32_x(pg, other_point_y4, v_point_y);
        svfloat32_t v_result4_1 = svmul_f32_x(pg, v_diff_y4, v_diff_y4);
        svfloat32_t v_diff_y5 = svsub_f32_x(pg, other_point_y5, v_point_y);
        svfloat32_t v_result5_1 = svmul_f32_x(pg, v_diff_y5, v_diff_y5);
        svfloat32_t v_diff_y6 = svsub_f32_x(pg, other_point_y6, v_point_y);
        svfloat32_t v_result6_1 = svmul_f32_x(pg, v_diff_y6, v_diff_y6);
        svfloat32_t v_diff_y7 = svsub_f32_x(pg, other_point_y7, v_point_y);
        svfloat32_t v_result7_1 = svmul_f32_x(pg, v_diff_y7, v_diff_y7);
        svfloat32_t v_diff_y8 = svsub_f32_x(pg, other_point_y8, v_point_y);
        svfloat32_t v_result8_1 = svmul_f32_x(pg, v_diff_y8, v_diff_y8);

        svfloat32_t v_diff_z1 = svsub_f32_x(pg, other_point_z1, v_point_z);
        svfloat32_t v_result1_2 = svmla_f32_x(pg, v_result1, v_diff_z1, v_diff_z1);
        svfloat32_t v_diff_z2 = svsub_f32_x(pg, other_point_z2, v_point_z);
        svfloat32_t v_result2_2 = svmla_f32_x(pg, v_result2, v_diff_z2, v_diff_z2);
        svfloat32_t v_diff_z3 = svsub_f32_x(pg, other_point_z3, v_point_z);
        svfloat32_t v_result3_2 = svmla_f32_x(pg, v_result3, v_diff_z3, v_diff_z3);
        svfloat32_t v_diff_z4 = svsub_f32_x(pg, other_point_z4, v_point_z);
        svfloat32_t v_result4_2 = svmla_f32_x(pg, v_result4, v_diff_z4, v_diff_z4);
        svfloat32_t v_diff_z5 = svsub_f32_x(pg, other_point_z5, v_point_z);
        svfloat32_t v_result5_2 = svmla_f32_x(pg, v_result5, v_diff_z5, v_diff_z5);
        svfloat32_t v_diff_z6 = svsub_f32_x(pg, other_point_z6, v_point_z);
        svfloat32_t v_result6_2 = svmla_f32_x(pg, v_result6, v_diff_z6, v_diff_z6);
        svfloat32_t v_diff_z7 = svsub_f32_x(pg, other_point_z7, v_point_z);
        svfloat32_t v_result7_2 = svmla_f32_x(pg, v_result7, v_diff_z7, v_diff_z7);
        svfloat32_t v_diff_z8 = svsub_f32_x(pg, other_point_z8, v_point_z);
        svfloat32_t v_result8_2 = svmla_f32_x(pg, v_result8, v_diff_z8, v_diff_z8);

        v_result1 = svadd_f32_x(pg, v_result1_2, v_result1_1);
        v_result2 = svadd_f32_x(pg, v_result2_2, v_result2_1);
        v_result3 = svadd_f32_x(pg, v_result3_2, v_result3_1);
        v_result4 = svadd_f32_x(pg, v_result4_2, v_result4_1);
        v_result5 = svadd_f32_x(pg, v_result5_2, v_result5_1);
        v_result6 = svadd_f32_x(pg, v_result6_2, v_result6_1);
        v_result7 = svadd_f32_x(pg, v_result7_2, v_result7_1);
        v_result8 = svadd_f32_x(pg, v_result8_2, v_result8_1);


        svbool_t mask1 = svcmple_n_f32(pg, v_result1, EPS2);
        svbool_t mask2 = svcmple_n_f32(pg, v_result2, EPS2);
        svbool_t mask3 = svcmple_n_f32(pg, v_result3, EPS2);
        svbool_t mask4 = svcmple_n_f32(pg, v_result4, EPS2);
        svbool_t mask5 = svcmple_n_f32(pg, v_result5, EPS2);
        svbool_t mask6 = svcmple_n_f32(pg, v_result6, EPS2);
        svbool_t mask7 = svcmple_n_f32(pg, v_result7, EPS2);
        svbool_t mask8 = svcmple_n_f32(pg, v_result8, EPS2);

        count += svcntp_b32(pg, mask1);
        count += svcntp_b32(pg, mask2);
        count += svcntp_b32(pg, mask3);
        count += svcntp_b32(pg, mask4);
        count += svcntp_b32(pg, mask5);
        count += svcntp_b32(pg, mask6);
        count += svcntp_b32(pg, mask7);
        count += svcntp_b32(pg, mask8);

         //load only cluster labels of distances less than ESP2
        svint32_t cluster_labels_of_neighbours1 = svld1_gather_s32index_s32(mask1, &clusters[0], sv_indices1);
        svint32_t cluster_labels_of_neighbours2 = svld1_gather_s32index_s32(mask2, &clusters[0], sv_indices2);
        svint32_t cluster_labels_of_neighbours3 = svld1_gather_s32index_s32(mask3, &clusters[0], sv_indices3);
        svint32_t cluster_labels_of_neighbours4 = svld1_gather_s32index_s32(mask4, &clusters[0], sv_indices4);
        svint32_t cluster_labels_of_neighbours5 = svld1_gather_s32index_s32(mask5, &clusters[0], sv_indices5);
        svint32_t cluster_labels_of_neighbours6 = svld1_gather_s32index_s32(mask6, &clusters[0], sv_indices6);
        svint32_t cluster_labels_of_neighbours7 = svld1_gather_s32index_s32(mask7, &clusters[0], sv_indices7);
        svint32_t cluster_labels_of_neighbours8 = svld1_gather_s32index_s32(mask8, &clusters[0], sv_indices8);

        svbool_t not_visited1 = svcmpne_n_s32(mask1, cluster_labels_of_neighbours1, NOT_VISITED<index_type>);
        svbool_t not_visited2 = svcmpne_n_s32(mask2, cluster_labels_of_neighbours2, NOT_VISITED<index_type>);
        svbool_t not_visited3 = svcmpne_n_s32(mask3, cluster_labels_of_neighbours3, NOT_VISITED<index_type>);
        svbool_t not_visited4 = svcmpne_n_s32(mask4, cluster_labels_of_neighbours4, NOT_VISITED<index_type>);
        svbool_t not_visited5 = svcmpne_n_s32(mask5, cluster_labels_of_neighbours5, NOT_VISITED<index_type>);
        svbool_t not_visited6 = svcmpne_n_s32(mask6, cluster_labels_of_neighbours6, NOT_VISITED<index_type>);
        svbool_t not_visited7 = svcmpne_n_s32(mask7, cluster_labels_of_neighbours7, NOT_VISITED<index_type>);
        svbool_t not_visited8 = svcmpne_n_s32(mask8, cluster_labels_of_neighbours8, NOT_VISITED<index_type>);

        svbool_t less_than_zero1 = svcmplt_n_s32(mask1, cluster_labels_of_neighbours1, 0);
        svbool_t less_than_zero2 = svcmplt_n_s32(mask2, cluster_labels_of_neighbours2, 0);
        svbool_t less_than_zero3 = svcmplt_n_s32(mask3, cluster_labels_of_neighbours3, 0);
        svbool_t less_than_zero4 = svcmplt_n_s32(mask4, cluster_labels_of_neighbours4, 0);
        svbool_t less_than_zero5 = svcmplt_n_s32(mask5, cluster_labels_of_neighbours5, 0);
        svbool_t less_than_zero6 = svcmplt_n_s32(mask6, cluster_labels_of_neighbours6, 0);
        svbool_t less_than_zero7 = svcmplt_n_s32(mask7, cluster_labels_of_neighbours7, 0);
        svbool_t less_than_zero8 = svcmplt_n_s32(mask8, cluster_labels_of_neighbours8, 0);

        cluster_labels_of_neighbours1 = svabs_s32_z(less_than_zero1, cluster_labels_of_neighbours1);
        cluster_labels_of_neighbours2 = svabs_s32_z(less_than_zero2, cluster_labels_of_neighbours2);
        cluster_labels_of_neighbours3 = svabs_s32_z(less_than_zero3, cluster_labels_of_neighbours3);
        cluster_labels_of_neighbours4 = svabs_s32_z(less_than_zero4, cluster_labels_of_neighbours4);
        cluster_labels_of_neighbours5 = svabs_s32_z(less_than_zero5, cluster_labels_of_neighbours5);
        cluster_labels_of_neighbours6 = svabs_s32_z(less_than_zero6, cluster_labels_of_neighbours6);
        cluster_labels_of_neighbours7 = svabs_s32_z(less_than_zero7, cluster_labels_of_neighbours7);
        cluster_labels_of_neighbours8 = svabs_s32_z(less_than_zero8, cluster_labels_of_neighbours8);

        cluster_label = std::min(cluster_label, svminv_s32(less_than_zero1, cluster_labels_of_neighbours1));
        cluster_label = std::min(cluster_label, svminv_s32(less_than_zero2, cluster_labels_of_neighbours2));
        cluster_label = std::min(cluster_label, svminv_s32(less_than_zero3, cluster_labels_of_neighbours3));
        cluster_label = std::min(cluster_label, svminv_s32(less_than_zero4, cluster_labels_of_neighbours4));
        cluster_label = std::min(cluster_label, svminv_s32(less_than_zero5, cluster_labels_of_neighbours5));
        cluster_label = std::min(cluster_label, svminv_s32(less_than_zero6, cluster_labels_of_neighbours6));
        cluster_label = std::min(cluster_label, svminv_s32(less_than_zero7, cluster_labels_of_neighbours7));
        cluster_label = std::min(cluster_label, svminv_s32(less_than_zero8, cluster_labels_of_neighbours8));

        svst1_s32(mask1, &min_points_area[i], sv_indices1);
        svst1_s32(mask2, &min_points_area[i+1*elements_in_vector], sv_indices2);
        svst1_s32(mask3, &min_points_area[i+2*elements_in_vector], sv_indices3);
        svst1_s32(mask4, &min_points_area[i+3*elements_in_vector], sv_indices4);
        svst1_s32(mask5, &min_points_area[i+4*elements_in_vector], sv_indices5);
        svst1_s32(mask6, &min_points_area[i+5*elements_in_vector], sv_indices6);
        svst1_s32(mask7, &min_points_area[i+6*elements_in_vector], sv_indices7);
        svst1_s32(mask8, &min_points_area[i+7*elements_in_vector], sv_indices8);

    }
    for (size_t i = n-rest; i < n; i += svcntw()) {
#elif defined(USE_4X_NEIGHBOUR_LOOP_UNROLL)
    constexpr size_t vectors_per_loop = 4;
    size_t elements_in_vector = svcntw();
    size_t loop_elements = elements_in_vector*vectors_per_loop;
    size_t rest = n % loop_elements;
    size_t chunks = n/loop_elements;

    for (size_t j = 0; j < chunks; j++) {
        size_t i = j*loop_elements;

        // no masking for unrolled loop
        // svbool_t pg = svwhilelt_b32(i, n);
        svbool_t pg = svptrue_b32();

        svint32_t sv_indices1 = svld1_s32(pg, &neighboring_points[i]);
        svint32_t sv_indices2 = svld1_s32(pg, &neighboring_points[i+1*elements_in_vector]);
        svint32_t sv_indices3 = svld1_s32(pg, &neighboring_points[i+2*elements_in_vector]);
        svint32_t sv_indices4 = svld1_s32(pg, &neighboring_points[i+3*elements_in_vector]);
        svint32_t sv_indices5 = svld1_s32(pg, &neighboring_points[i+4*elements_in_vector]);
        svint32_t sv_indices6 = svld1_s32(pg, &neighboring_points[i+1*elements_in_vector]);
        svint32_t sv_indices7 = svld1_s32(pg, &neighboring_points[i+2*elements_in_vector]);
        svint32_t sv_indices8 = svld1_s32(pg, &neighboring_points[i+3*elements_in_vector]);

        svint32_t sv_indices_scaled1 = svmul_n_s32_z(pg, sv_indices1, dimensions);
        svint32_t sv_indices_scaled2 = svmul_n_s32_z(pg, sv_indices2, dimensions);
        svint32_t sv_indices_scaled3 = svmul_n_s32_z(pg, sv_indices3, dimensions);
        svint32_t sv_indices_scaled4 = svmul_n_s32_z(pg, sv_indices4, dimensions);

        svfloat32_t v_point_x = svdup_n_f32(point[0]);
        svfloat32_t v_point_y = svdup_n_f32(point[1]);
        svfloat32_t v_point_z = svdup_n_f32(point[2]);

        svint32_t other_point_index1 = svadd_n_s32_z(pg, sv_indices_scaled1, 0);
        svint32_t other_point_index2 = svadd_n_s32_z(pg, sv_indices_scaled2, 0);
        svint32_t other_point_index3 = svadd_n_s32_z(pg, sv_indices_scaled3, 0);
        svint32_t other_point_index4 = svadd_n_s32_z(pg, sv_indices_scaled4, 0);

        svfloat32_t other_point_x1 = svld1_gather_s32index_f32(pg, &np_ptr[0]+0, other_point_index1);
        svfloat32_t other_point_x2 = svld1_gather_s32index_f32(pg, &np_ptr[0]+0, other_point_index2);
        svfloat32_t other_point_x3 = svld1_gather_s32index_f32(pg, &np_ptr[0]+0, other_point_index3);
        svfloat32_t other_point_x4 = svld1_gather_s32index_f32(pg, &np_ptr[0]+0, other_point_index4);

        svfloat32_t other_point_y1 = svld1_gather_s32index_f32(pg, &np_ptr[0]+1, other_point_index1);
        svfloat32_t other_point_y2 = svld1_gather_s32index_f32(pg, &np_ptr[0]+1, other_point_index2);
        svfloat32_t other_point_y3 = svld1_gather_s32index_f32(pg, &np_ptr[0]+1, other_point_index3);
        svfloat32_t other_point_y4 = svld1_gather_s32index_f32(pg, &np_ptr[0]+1, other_point_index4);

        svfloat32_t other_point_z1 = svld1_gather_s32index_f32(pg, &np_ptr[0]+2, other_point_index1);
        svfloat32_t other_point_z2 = svld1_gather_s32index_f32(pg, &np_ptr[0]+2, other_point_index2);
        svfloat32_t other_point_z3 = svld1_gather_s32index_f32(pg, &np_ptr[0]+2, other_point_index3);
        svfloat32_t other_point_z4 = svld1_gather_s32index_f32(pg, &np_ptr[0]+2, other_point_index4);

        svfloat32_t v_diff_x1 = svsub_f32_x(pg, other_point_x1, v_point_x);
        svfloat32_t v_result1 = svmul_f32_x(pg, v_diff_x1, v_diff_x1);

        svfloat32_t v_diff_x2 = svsub_f32_x(pg, other_point_x2, v_point_x);
        svfloat32_t v_result2 = svmul_f32_x(pg, v_diff_x2, v_diff_x2);

        svfloat32_t v_diff_x3 = svsub_f32_x(pg, other_point_x3, v_point_x);
        svfloat32_t v_result3 = svmul_f32_x(pg, v_diff_x3, v_diff_x3);

        svfloat32_t v_diff_x4 = svsub_f32_x(pg, other_point_x4, v_point_x);
        svfloat32_t v_result4 = svmul_f32_x(pg, v_diff_x4, v_diff_x4);

        svfloat32_t v_diff_y1 = svsub_f32_x(pg, other_point_y1, v_point_y);
        svfloat32_t v_result1_1 = svmul_f32_x(pg, v_diff_y1, v_diff_y1);

        svfloat32_t v_diff_y2 = svsub_f32_x(pg, other_point_y2, v_point_y);
        svfloat32_t v_result2_1 = svmul_f32_x(pg, v_diff_y2, v_diff_y2);

        svfloat32_t v_diff_y3 = svsub_f32_x(pg, other_point_y3, v_point_y);
        svfloat32_t v_result3_1 = svmul_f32_x(pg, v_diff_y3, v_diff_y3);

        svfloat32_t v_diff_y4 = svsub_f32_x(pg, other_point_y4, v_point_y);
        svfloat32_t v_result4_1 = svmul_f32_x(pg, v_diff_y4, v_diff_y4);

        svfloat32_t v_diff_z1 = svsub_f32_x(pg, other_point_z1, v_point_z);
        svfloat32_t v_result1_2 = svmla_f32_x(pg, v_result1, v_diff_z1, v_diff_z1);

        svfloat32_t v_diff_z2 = svsub_f32_x(pg, other_point_z2, v_point_z);
        svfloat32_t v_result2_2 = svmla_f32_x(pg, v_result2, v_diff_z2, v_diff_z2);

        svfloat32_t v_diff_z3 = svsub_f32_x(pg, other_point_z3, v_point_z);
        svfloat32_t v_result3_2 = svmla_f32_x(pg, v_result3, v_diff_z3, v_diff_z3);

        svfloat32_t v_diff_z4 = svsub_f32_x(pg, other_point_z4, v_point_z);
        svfloat32_t v_result4_2 = svmla_f32_x(pg, v_result4, v_diff_z4, v_diff_z4);

        v_result1 = svadd_f32_x(pg, v_result1_2, v_result1_1);
        v_result2 = svadd_f32_x(pg, v_result2_2, v_result2_1);
        v_result3 = svadd_f32_x(pg, v_result3_2, v_result3_1);
        v_result4 = svadd_f32_x(pg, v_result4_2, v_result4_1);


        svbool_t mask1 = svcmple_n_f32(pg, v_result1, EPS2);
        svbool_t mask2 = svcmple_n_f32(pg, v_result2, EPS2);
        svbool_t mask3 = svcmple_n_f32(pg, v_result3, EPS2);
        svbool_t mask4 = svcmple_n_f32(pg, v_result4, EPS2);

        count += svcntp_b32(pg, mask1);
        count += svcntp_b32(pg, mask2);
        count += svcntp_b32(pg, mask3);
        count += svcntp_b32(pg, mask4);

         //load only cluster labels of distances less than ESP2
        svint32_t cluster_labels_of_neighbours1 = svld1_gather_s32index_s32(mask1, &clusters[0], sv_indices1);
        svint32_t cluster_labels_of_neighbours2 = svld1_gather_s32index_s32(mask2, &clusters[0], sv_indices2);
        svint32_t cluster_labels_of_neighbours3 = svld1_gather_s32index_s32(mask3, &clusters[0], sv_indices3);
        svint32_t cluster_labels_of_neighbours4 = svld1_gather_s32index_s32(mask4, &clusters[0], sv_indices4);

        svbool_t not_visited1 = svcmpne_n_s32(mask1, cluster_labels_of_neighbours1, NOT_VISITED<index_type>);
        svbool_t not_visited2 = svcmpne_n_s32(mask2, cluster_labels_of_neighbours2, NOT_VISITED<index_type>);
        svbool_t not_visited3 = svcmpne_n_s32(mask3, cluster_labels_of_neighbours3, NOT_VISITED<index_type>);
        svbool_t not_visited4 = svcmpne_n_s32(mask4, cluster_labels_of_neighbours4, NOT_VISITED<index_type>);

        svbool_t less_than_zero1 = svcmplt_n_s32(mask1, cluster_labels_of_neighbours1, 0);
        svbool_t less_than_zero2 = svcmplt_n_s32(mask2, cluster_labels_of_neighbours2, 0);
        svbool_t less_than_zero3 = svcmplt_n_s32(mask3, cluster_labels_of_neighbours3, 0);
        svbool_t less_than_zero4 = svcmplt_n_s32(mask4, cluster_labels_of_neighbours4, 0);

        cluster_labels_of_neighbours1 = svabs_s32_z(less_than_zero1, cluster_labels_of_neighbours1);
        cluster_labels_of_neighbours2 = svabs_s32_z(less_than_zero2, cluster_labels_of_neighbours2);
        cluster_labels_of_neighbours3 = svabs_s32_z(less_than_zero3, cluster_labels_of_neighbours3);
        cluster_labels_of_neighbours4 = svabs_s32_z(less_than_zero4, cluster_labels_of_neighbours4);

        cluster_label = std::min(cluster_label, svminv_s32(less_than_zero1, cluster_labels_of_neighbours1));
        cluster_label = std::min(cluster_label, svminv_s32(less_than_zero2, cluster_labels_of_neighbours2));
        cluster_label = std::min(cluster_label, svminv_s32(less_than_zero3, cluster_labels_of_neighbours3));
        cluster_label = std::min(cluster_label, svminv_s32(less_than_zero4, cluster_labels_of_neighbours4));

        svst1_s32(mask1, &min_points_area[i], sv_indices1);
        svst1_s32(mask2, &min_points_area[i+1*elements_in_vector], sv_indices2);
        svst1_s32(mask3, &min_points_area[i+2*elements_in_vector], sv_indices3);
        svst1_s32(mask4, &min_points_area[i+3*elements_in_vector], sv_indices4);

    }
    for (size_t i = n-rest; i < n; i += svcntw()) {
#else
    for (size_t i = 0; i < n; i += svcntw()) {
#endif

        svbool_t pg = svwhilelt_b32(i, n);

        svint32_t sv_indices = svld1_s32(pg, &neighboring_points[i]);

        svint32_t sv_indices_scaled = svmul_n_s32_z(pg, sv_indices, dimensions);

        //for(size_t d = 0; d < dimensions; d++) {

        //    svfloat32_t point_coordinate_v = svdup_n_f32(point[d]); 

        //    svint32_t  other_point_index = svadd_n_s32_z(pg, sv_indices_scaled, d);

        //    svfloat32_t other_point_coordinate_v = svld1_gather_s32index_f32(pg, &np_ptr[0], other_point_index);

        //    svfloat32_t diff_v = svsub_f32_x(pg, other_point_coordinate_v, point_coordinate_v);

        //    svfloat32_t diff_square = svmul_f32_x(pg, diff_v, diff_v);

        //    results_v = svadd_x(pg, results_v, diff_square);

        //}

        svfloat32_t v_current_point_x = svdup_n_f32(point[0]);
        svfloat32_t v_current_point_y = svdup_n_f32(point[1]);
        svfloat32_t v_current_point_z = svdup_n_f32(point[2]);

        svint32_t other_point_index = svadd_n_s32_z(pg, sv_indices_scaled, 0);

        svfloat32_t other_point_x = svld1_gather_s32index_f32(pg, &np_ptr[0]+0, other_point_index);
        svfloat32_t other_point_y = svld1_gather_s32index_f32(pg, &np_ptr[0]+1, other_point_index);
        svfloat32_t other_point_z = svld1_gather_s32index_f32(pg, &np_ptr[0]+2, other_point_index);

        svfloat32_t v_diff_x = svsub_f32_x(pg, other_point_x, v_current_point_x);
        svfloat32_t v_results = svmul_f32_x(pg, v_diff_x, v_diff_x);
        svfloat32_t v_diff_y = svsub_f32_x(pg, other_point_y, v_current_point_y);
        svfloat32_t v_results1 = svmul_f32_x(pg, v_diff_y, v_diff_y);
        svfloat32_t v_diff_z = svsub_f32_x(pg, other_point_z, v_current_point_z);
        svfloat32_t v_results2 = svmla_f32_x(pg, v_results, v_diff_z, v_diff_z);
        v_results = svadd_f32_x(pg, v_results2, v_results1);


        svbool_t mask = svcmple_n_f32(pg, v_results, EPS2);

        count += svcntp_b32(pg, mask);

        svint32_t cluster_labels_of_neighbours = svld1_gather_s32index_s32(mask, &clusters[0], sv_indices); //load only cluster labels of distances less than ESP2

        svbool_t not_visited = svcmpne_n_s32(mask, cluster_labels_of_neighbours, NOT_VISITED<index_type>); //NOT_VISITED_s32 is equal to INT_MAX

        svbool_t less_than_zero = svcmplt_n_s32(mask, cluster_labels_of_neighbours, 0);

        cluster_labels_of_neighbours = svabs_s32_z(less_than_zero, cluster_labels_of_neighbours);

        cluster_label = std::min(cluster_label, svminv_s32(less_than_zero, cluster_labels_of_neighbours));

        svst1_s32(mask, &min_points_area[i], sv_indices);

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

    const std::int32_t dimensions = static_cast<std::int32_t>(m_data.m_chunk[1]);
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

    Cluster<index_type> cluster_label = m_global_point_offset + point_index + 1;

    size_t n = neighboring_points.size();

    min_points_area = std::vector<index_type>(n, NOT_VISITED<index_type>);


    for (size_t i = 0; i < n; i += svcntw()) {

        svbool_t pg = svwhilelt_b32(i, n);

        svint32_t sv_indices = svld1_s32(pg, &neighboring_points[i]);

        svint32_t sv_indices_scaled = svmul_n_s32_z(pg, sv_indices, dimensions);

        svfloat32_t results_v = svdup_n_f32(0.0f);

        for(size_t d = 0; d < dimensions; d++) {

            svfloat32_t point_coordinate_v = svdup_n_f32(point[d]); 

            svint32_t  other_point_index = svadd_n_s32_z(pg, sv_indices_scaled, d);

            svfloat32_t other_point_coordinate_v = svld1_gather_s32index_f32(pg, &np_ptr[0], other_point_index);

            svfloat32_t diff_v = svsub_f32_x(pg, other_point_coordinate_v, point_coordinate_v);

            svfloat32_t diff_square = svmul_f32_x(pg, diff_v, diff_v);

            results_v = svadd_x(pg, results_v, diff_square);

        }

        svbool_t mask = svcmple_n_f32(pg, results_v, EPS2);

        count += svcntp_b32(pg, mask);

        svint32_t cluster_labels_of_neighbours = svld1_gather_s32index_s32(mask, &clusters[0], sv_indices); //load only cluster labels of distances less than ESP2

        svbool_t not_visited = svcmpne_n_s32(mask, cluster_labels_of_neighbours, NOT_VISITED<index_type>); //NOT_VISITED_s32 is equal to INT_MAX

        svbool_t less_than_zero = svcmplt_n_s32(mask, cluster_labels_of_neighbours, 0);

        cluster_labels_of_neighbours = svabs_s32_z(less_than_zero, cluster_labels_of_neighbours);

        cluster_label = std::min(cluster_label, svminv_s32(less_than_zero, cluster_labels_of_neighbours));

        svst1_s32(mask, &min_points_area[i], sv_indices);

    }

    return cluster_label;

}

#endif

