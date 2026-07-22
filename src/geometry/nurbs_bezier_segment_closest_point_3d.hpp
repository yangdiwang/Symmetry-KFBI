#pragma once

#include "rational_bezier_element_3d.hpp"

#include <Eigen/Dense>

#include <limits>
#include <optional>

namespace kfbim::geometry3d {

struct NurbsElementSegmentClosestPointOptions3D {
    double distance_tolerance = 1.0e-12;
    double parameter_tolerance = 1.0e-12;
    int max_iterations = 36;
};

struct NurbsElementSegmentClosestPointResult3D {
    bool converged = false;
    double u = 0.0;
    double v = 0.0;
    double t = 0.0;
    Eigen::Vector3d surface_point = Eigen::Vector3d::Zero();
    Eigen::Vector3d segment_point = Eigen::Vector3d::Zero();
    double distance = std::numeric_limits<double>::infinity();
    double projected_gradient_norm = std::numeric_limits<double>::infinity();
    int iterations = 0;
};

NurbsElementSegmentClosestPointResult3D
closest_point_nurbs_bezier_element_to_segment_3d(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end,
    const NurbsElementSegmentClosestPointOptions3D& options,
    std::optional<Eigen::Vector2d> preferred_seed = std::nullopt);

} // namespace kfbim::geometry3d
