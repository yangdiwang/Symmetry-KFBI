#pragma once

#include "nurbs_surface_model_3d.hpp"

#include <Eigen/Dense>

#include <array>
#include <vector>

namespace kfbim::geometry3d {

struct RationalBezierElement3D {
    int patch_index = -1;
    int component = -1;
    int degree_u = 0;
    int degree_v = 0;
    double parameter_u0 = 0.0;
    double parameter_u1 = 0.0;
    double parameter_v0 = 0.0;
    double parameter_v1 = 0.0;
    std::vector<Eigen::Vector4d> homogeneous_controls;

    double u0() const;
    double u1() const;
    double v0() const;
    double v1() const;
    const Eigen::Vector4d& control(int i, int j) const;
    Eigen::Vector3d evaluate(double u, double v) const;
    NurbsAabb3D bounds() const;
    std::array<RationalBezierElement3D, 2> split_u() const;
    std::array<RationalBezierElement3D, 2> split_v() const;
};

} // namespace kfbim::geometry3d
