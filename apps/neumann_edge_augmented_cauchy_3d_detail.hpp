#pragma once

#include "neumann_edge_augmented_cauchy_3d.hpp"

#include <vector>

namespace kfbim::app3d::detail {

struct NeumannEdgeDistanceCandidate3D {
    double squared_distance = 0.0;
    int sample_index = -1;
    int edge_sample_index = -1;
    int certified_symmetric_partner = -1;
};

struct NeumannEdgeSampleCountPlan3D {
    int connection_sample_count = 0;
    int cumulative_sample_count = 0;
};

NeumannEdgeSampleCountPlan3D plan_neumann_edge_sample_count_3d(
    double requested_sample_count,
    int minimum_edge_samples,
    int cumulative_sample_count,
    int reserved_factorization_units,
    int connection_index);

std::vector<int> select_neumann_edge_distance_candidates_3d(
    std::vector<NeumannEdgeDistanceCandidate3D> candidates,
    double radius_squared,
    int count);

void validate_neumann_edge_local_map_3d(
    const NeumannEdgeLocalMap3D& local,
    int surface_size,
    int edge_sample_count);

int certified_l_prism_symmetric_partner_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const SurfaceDof3D& center,
    const geometry3d::NurbsPatchEdgeConnection3D& connection,
    const NeumannEdgeAuxiliarySample3D& sample);

} // namespace kfbim::app3d::detail
