#pragma once

#include "nurbs_surface_model_3d.hpp"
#include "rational_bezier_element_3d.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

namespace kfbim::geometry3d {

struct NurbsSurfaceCrossing3D {
    int patch_index = -1;
    int component = -1;
    double u = 0.0;
    double v = 0.0;
    double edge_parameter = 0.0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double residual = 0.0;
    double transversality = 0.0;
};

struct NurbsSurfaceIntersectionDiagnostics3D {
    int candidate_elements = 0;
    int triangle_seed_hits = 0;
    int triangle_seed_misses_recovered = 0;
    int subdivision_boxes = 0;
    int newton_attempts = 0;
    int newton_iterations = 0;
    int same_patch_deduplications = 0;
    int seam_deduplications = 0;
    int unresolved_candidates = 0;
    int maximum_subdivision_depth_reached = 0;
    int terminal_certificate_boxes = 0;
    int maximum_terminal_certificate_depth_reached = 0;
    int closest_point_attempts = 0;
    int closest_point_iterations = 0;
    int roots_recovered_by_closest_point = 0;
    int terminal_misses_by_closest_point = 0;
    int closest_point_failures = 0;
};

struct NurbsSurfaceIntersectionResult3D {
    std::vector<NurbsSurfaceCrossing3D> crossings;
    NurbsSurfaceIntersectionDiagnostics3D diagnostics;
    bool overlap_detected = false;
};

struct NurbsCartesianEdgeQuery3D {
    int axis = -1;
    int i = -1;
    int j = -1;
    int k = -1;
    Eigen::Vector3d start = Eigen::Vector3d::Zero();
    Eigen::Vector3d end = Eigen::Vector3d::Zero();
};

struct NurbsSurfaceIntersectorOptions3D {
    bool use_triangle_seeds = true;
    int bvh_leaf_size = 8;
    double maximum_element_extent =
        std::numeric_limits<double>::infinity();
    int local_max_subdivision_depth = 36;
};

class NurbsSurfaceIntersector3D {
public:
    explicit NurbsSurfaceIntersector3D(
        NurbsSurfaceModel3D model,
        NurbsSurfaceIntersectorOptions3D options = {});

    const NurbsSurfaceModel3D& model() const;
    const NurbsAabb3D& bounds() const;
    double geometry_tolerance() const;
    std::size_t query_element_count() const noexcept;
    double maximum_query_element_extent() const noexcept;

    NurbsSurfaceIntersectionResult3D intersect_segment(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end) const;
    std::optional<NurbsSurfaceCrossing3D> intersect_cartesian_edge(
        const NurbsCartesianEdgeQuery3D& edge,
        NurbsSurfaceIntersectionDiagnostics3D* diagnostics = nullptr) const;
    std::vector<int> containing_components(
        const Eigen::Vector3d& point) const;
    std::vector<RationalBezierElement3D> acceleration_leaves(
        double maximum_extent) const;

private:
    struct BvhNode {
        NurbsAabb3D bounds;
        int left = -1;
        int right = -1;
        int begin = 0;
        int end = 0;
    };

    int build_bvh_node(int begin, int end);
    void collect_candidate_elements(
        int node,
        const NurbsAabb3D& segment_bounds,
        std::vector<int>& candidates) const;
    NurbsSurfaceIntersectionResult3D intersect_segment_impl(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end,
        const NurbsCartesianEdgeQuery3D* cartesian_edge) const;

    NurbsSurfaceModel3D model_;
    NurbsSurfaceIntersectorOptions3D options_;
    NurbsAabb3D bounds_;
    double geometry_tolerance_ = 1e-14;
    double maximum_query_element_extent_ = 0.0;
    std::vector<RationalBezierElement3D> elements_;
    std::vector<int> element_order_;
    std::vector<BvhNode> bvh_nodes_;
    int bvh_root_ = -1;
};

} // namespace kfbim::geometry3d
