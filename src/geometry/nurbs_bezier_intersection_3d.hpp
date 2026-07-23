#pragma once

#include "rational_bezier_element_3d.hpp"

#include <Eigen/Dense>
#include <stdexcept>

#include <vector>

namespace kfbim::geometry3d {

struct NurbsElementRoot3D {
    int patch_index = -1;
    int component = -1;
    double u = 0.0;
    double v = 0.0;
    double t = 0.0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double residual = 0.0;
    double transversality = 0.0;
};

struct NurbsElementParameterSeed3D {
    double u = 0.0;
    double v = 0.0;
    double t = 0.0;
};

struct NurbsElementIntersectionDiagnostics3D {
    int subdivision_boxes = 0;
    int conservative_rejections = 0;
    int triangle_seed_hits = 0;
    int newton_attempts = 0;
    int newton_iterations = 0;
    int roots_recovered_without_triangle_seed = 0;
    int unresolved_boxes = 0;
    int maximum_subdivision_depth_reached = 0;
    int terminal_certificate_boxes = 0;
    int maximum_terminal_certificate_depth_reached = 0;
    int closest_point_attempts = 0;
    int closest_point_iterations = 0;
    int roots_recovered_by_closest_point = 0;
    int terminal_misses_by_closest_point = 0;
    int closest_point_failures = 0;
    int supplied_seed_attempts = 0;
    int roots_recovered_by_supplied_seed = 0;
    int maximum_supplied_seed_count = 0;
    int high_degree_control_hull_fallbacks = 0;
};

struct NurbsElementIntersectionResult3D {
    std::vector<NurbsElementRoot3D> roots;
    NurbsElementIntersectionDiagnostics3D diagnostics;
    bool overlap_detected = false;
};

class UnresolvedNurbsIntersectionCandidate3D : public std::runtime_error {
public:
    explicit UnresolvedNurbsIntersectionCandidate3D(
        NurbsElementIntersectionResult3D partial_result);

    const NurbsElementIntersectionResult3D& partial_result() const noexcept;

private:
    NurbsElementIntersectionResult3D partial_result_;
};

struct NurbsElementIntersectionOptions3D {
    double geometry_tolerance = 1e-12;
    double parameter_tolerance = 1e-12;
    int max_subdivision_depth = 36;
    int max_newton_iterations = 24;
    bool use_triangle_seed = true;
    std::vector<NurbsElementParameterSeed3D> parameter_seeds;
};

NurbsElementIntersectionResult3D intersect_nurbs_bezier_element_3d(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end,
    const NurbsElementIntersectionOptions3D& options);

} // namespace kfbim::geometry3d
