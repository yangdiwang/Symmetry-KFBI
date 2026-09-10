#pragma once

#include "src/support/cauchy/direct_coefficient_cubic_cauchy_3d.hpp"

namespace kfbim::app3d {

// Normalized [0,1]^2 patch coordinates. Extended means that the first/last
// rational knot span is evaluated beyond its domain, never clamped. This is a
// local normal projection, not a globally closest-point or branch certificate.
struct ExtendedTubularProjection3D {
    double u = 0.0, v = 0.0, r = 0.0;
    double tangential_residual = 0.0;
    int iterations = 0;
    bool converged = false;
};

[[nodiscard]] NativeSurfaceCubicParameterJet3D extended_nurbs_parameter_jet_3d(
    const geometry3d::NurbsSurfacePatch3D& patch, double u, double v);

[[nodiscard]] ExtendedTubularProjection3D project_extended_tubular_3d(
    const geometry3d::NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& target, double initial_u, double initial_v,
    int maximum_iterations = 20, double step_tolerance = 2.0e-12);

struct TubularCauchyWeights3D {
    Eigen::Matrix<double, 1, 10> value;
    Eigen::Matrix<double, 1, 6> normal;
    double projection_residual = 0.0;
    double projected_u = 0.0, projected_v = 0.0;
    int projection_iterations = 0;
    bool projection_converged = false;
};

struct ExtendedTubularCauchyPlan3D {
    // Non-owning; the immutable geometry must outlive this setup-only plan.
    const geometry3d::NurbsSurfacePatch3D* surface = nullptr;
    double u = 0.0, v = 0.0;
    double normal_orientation = 1.0;
    Eigen::Matrix<double, 20, 10> value_map;
    Eigen::Matrix<double, 20, 6> normal_map;

    // Rows index cubic_cauchy_powers_3d(), now interpreted as (du,dv,r).
    // Inputs are the SAME physical tangent-graph jets as the existing direct
    // cubic plan. No sampled unknown-density fitting is introduced.
    [[nodiscard]] TubularCauchyWeights3D weights(
        const Eigen::Vector3d& world_target, int degree = 3) const;
};

// Geometry and physical.u/v must use the same normalized parameter chart.
// Compose the physical polynomial with X(u+du,v+dv)+r*n(u+du,v+dv), truncated
// to total order 3. P2 uses the order <=2 part, not a separate fitted center.
[[nodiscard]] ExtendedTubularCauchyPlan3D build_extended_tubular_cauchy_plan_3d(
    const geometry3d::NurbsSurfacePatch3D& patch,
    const NativeSurfaceCubicParameterJet3D& geometry,
    const DirectCoefficientCubicCauchyPlan3D& physical);

} // namespace kfbim::app3d
