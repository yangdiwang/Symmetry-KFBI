#pragma once

#include "rational_bezier_element_3d.hpp"

#include <Eigen/Dense>

#include <limits>
#include <vector>

namespace kfbim::geometry3d {

struct NurbsParameterBox2D {
    double u0 = 0.0;
    double u1 = 0.0;
    double v0 = 0.0;
    double v1 = 0.0;
};

struct NurbsLineSurfaceStationaryPointOptions3D {
    double geometry_tolerance = 1.0e-12;
    double parameter_tolerance = 1.0e-12;
    int max_iterations = 32;
};

struct NurbsLineSurfaceStationaryPoint3D {
    bool converged = false;
    double u = 0.0;
    double v = 0.0;
    double t = 0.0;
    Eigen::Vector3d surface_point = Eigen::Vector3d::Zero();
    double distance = std::numeric_limits<double>::infinity();
    double tangent_measure = std::numeric_limits<double>::infinity();
    int iterations = 0;
};

NurbsLineSurfaceStationaryPoint3D
solve_nurbs_bezier_segment_stationary_point_3d(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end,
    const NurbsParameterBox2D& parameter_box,
    const Eigen::Vector2d& initial_parameter,
    const NurbsLineSurfaceStationaryPointOptions3D& options = {});
std::vector<NurbsLineSurfaceStationaryPoint3D>
find_nurbs_bezier_segment_stationary_points_3d(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end,
    const NurbsParameterBox2D& parameter_box,
    const std::vector<Eigen::Vector2d>& initial_parameters,
    const NurbsLineSurfaceStationaryPointOptions3D& options = {});

} // namespace kfbim::geometry3d
