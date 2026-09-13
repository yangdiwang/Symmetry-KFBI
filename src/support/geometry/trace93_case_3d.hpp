#pragma once

#include "src/support/geometry/trace_first_case_3d.hpp"
#include "src/geometry/models3d/bicubic_nurbs_models_3d.hpp"

namespace kfbim::app3d {

struct Trace93GeometryJet3D {
    // d[i][j] = world-coordinate partial_u^i partial_v^j X.
    std::array<std::array<Eigen::Vector3d, 5>, 5> d;
    Trace93GeometryJet3D();
};

// Exact native geometry and Python analysis charts are deliberately distinct.
// In particular a rational circle parameter is not a linear angle parameter.
struct Trace93Case3D : TraceFirstCase3D {
    std::string geometry_name;
    std::vector<SurfaceAnalysisChart3D> analysis_patches;
    bool native_parameters_match_analysis = true;

    Trace93GeometryJet3D analysis_at(
        int pid, double u, double v, int order = 3) const;
    Eigen::Vector2d world_to_analysis_uv(
        int pid, const Eigen::Vector3d& world,
        const Eigen::Vector2d& initial = Eigen::Vector2d(0.5, 0.5)) const;
    Eigen::Vector2d world_to_native_uv(
        int pid, const Eigen::Vector3d& world,
        const Eigen::Vector2d& initial = Eigen::Vector2d(0.5, 0.5)) const;
    Eigen::Vector2d analysis_to_native_uv(int pid, double u, double v) const;
};

// Python 93 benchmark patch IDs, dimensions, field and mean quadrature.
// All native geometry patches are exact bicubic NURBS, without fitting.
Trace93Case3D make_trace93_case_3d(
    const std::string& geometry, const std::string& transform = "rotate");

} // namespace kfbim::app3d
