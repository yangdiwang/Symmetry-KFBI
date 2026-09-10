#pragma once

#include "src/support/geometry/native_nurbs_surface_transform_3d.hpp"
#include <array>
#include <string>

namespace kfbim::app3d {

// Mirrors tests/cases/trace_first_3d/python_torus.json. Opt-in, not the legacy torus.
struct TraceFirstHarmonicJet3D {
    double value = 0.0;
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
    Eigen::Matrix3d hessian = Eigen::Matrix3d::Zero();
    // third[k](i,j) = d_i d_j d_k u, world coordinates.
    std::array<Eigen::Matrix3d, 3> third{
        Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
};

struct TraceFirstCase3D {
    std::string id = "python_torus_20260909";
    std::string transform_name;
    NativeNurbsSurface3D surface;
    RigidTransform3D transform;
    double boundary_mean_shift = 0.0;
    int density_base_N = 32;
    std::array<int, 2> density_base_spans{{4, 2}};
    int trace_gauss_order = 4;
    int mean_gauss_order = 8;
    TraceFirstHarmonicJet3D evaluate(const Eigen::Vector3d& world) const;
    std::array<int, 2> density_spans(int N) const;
};

TraceFirstCase3D make_trace_first_python_torus_case_3d(
    const std::string& transform_name = "rotate");

} // namespace kfbim::app3d
