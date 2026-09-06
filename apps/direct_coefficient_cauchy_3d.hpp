#pragma once

#include "crossing_cauchy_plan_3d.hpp"
#include "native_nurbs_density_space_3d.hpp"

#include <Eigen/Core>

#include <array>
#include <functional>

namespace kfbim::app3d {

// Geometry derivatives with respect to the normalized patch parameters used
// by NativeNurbsDensitySpace3D.  The implementation evaluates the rational
// NURBS quotient analytically through second order; no geometry samples or
// finite differences enter this jet.
struct NativeSurfaceParameterJet3D {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d x_u = Eigen::Vector3d::Zero();
    Eigen::Vector3d x_v = Eigen::Vector3d::Zero();
    Eigen::Vector3d x_uu = Eigen::Vector3d::Zero();
    Eigen::Vector3d x_uv = Eigen::Vector3d::Zero();
    Eigen::Vector3d x_vv = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
};

struct NativeSurfaceNormalParameterJet3D {
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal_u = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal_v = Eigen::Vector3d::Zero();
};

[[nodiscard]] NativeSurfaceParameterJet3D
native_surface_parameter_jet_3d(
    const NativeNurbsDensitySpace3D& density,
    int patch,
    double u,
    double v);

[[nodiscard]] NativeSurfaceNormalParameterJet3D
surface_normal_parameter_jet_3d(
    const NativeSurfaceParameterJet3D& geometry);

[[nodiscard]] Eigen::Vector2d parameter_gradient_to_tangent_gradient_3d(
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame,
    const Eigen::Vector2d& parameter_gradient);

// Map (mu,mu_u,mu_v,mu_uu,mu_uv,mu_vv) to
// (J0,J0_s,J0_t,J0_ss,J0_st,J0_tt), where (s,t) are the graph coordinates
// induced by frame.  This includes the second-order inverse-coordinate terms;
// merely rotating the parameter Hessian is not sufficient on a nonlinear
// patch parameterization.
[[nodiscard]] Eigen::Matrix<double, 6, 6>
parameter_to_cauchy_value_jet_matrix_3d(
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame);

[[nodiscard]] TangentGraphHessian3D
tangent_graph_hessian_from_parameter_jet_3d(
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame);

struct DirectCoefficientJetDiagnostics3D {
    double parameter_to_tangent_determinant = 0.0;
    double parameter_to_tangent_condition = 0.0;
    double tangent_plane_residual = 0.0;
};

// Immutable crossing plan for the unknown density.  Every output row has at
// most the same 4x4 active tensor support as the cubic density itself.  It can
// therefore be retained per crossing and applied in GMRES without sampling,
// least-squares recovery, or a dense global coefficient map.
struct DirectCoefficientValueJetPlan3D {
    int patch = -1;
    double u = 0.0;
    double v = 0.0;
    LocalOrthonormalFrame3D frame;
    TangentGraphHessian3D graph_hessian;
    std::array<NativeDensityC0Stencil3D, 6> cauchy_rows;
    DirectCoefficientJetDiagnostics3D diagnostics;

    [[nodiscard]] ValueJet3D evaluate(
        const Eigen::Ref<const Eigen::VectorXd>& c0_coefficients) const;

    // Precompose a Cauchy P2 value row with the analytic coefficient jet.
    // The returned sparse row acts directly on Base/C0 coefficients.
    [[nodiscard]] NativeDensityC0Stencil3D compose_value_row(
        const ValueCauchyWeightRow3D& cauchy_weights) const;
};

[[nodiscard]] DirectCoefficientValueJetPlan3D
build_direct_coefficient_value_jet_plan_3d(
    const NativeNurbsDensitySpace3D& density,
    int patch,
    double u,
    double v,
    const LocalOrthonormalFrame3D& frame);

// Reuse the geometry jet already evaluated for this exact (patch,u,v).
// The caller must supply the jet in the density surface's normalized
// parameter coordinates; coefficient rows and diagnostics are unchanged.
[[nodiscard]] DirectCoefficientValueJetPlan3D
build_direct_coefficient_value_jet_plan_3d(
    const NativeNurbsDensitySpace3D& density,
    int patch,
    double u,
    double v,
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame);

// Dirichlet counterpart: the unknown normal jump J1 only needs its value and
// two first tangential derivatives.  Rows remain on the same local 4x4 cubic
// support and may be precomposed with CauchyPolynomialWeights3D::w1.
struct DirectCoefficientNormalJetPlan3D {
    int patch = -1;
    double u = 0.0;
    double v = 0.0;
    LocalOrthonormalFrame3D frame;
    TangentGraphHessian3D graph_hessian;
    std::array<NativeDensityC0Stencil3D, 3> cauchy_rows;
    DirectCoefficientJetDiagnostics3D diagnostics;

    [[nodiscard]] NormalJet3D evaluate(
        const Eigen::Ref<const Eigen::VectorXd>& c0_coefficients) const;

    [[nodiscard]] NativeDensityC0Stencil3D compose_normal_row(
        const NormalCauchyWeightRow3D& cauchy_weights) const;
};

[[nodiscard]] DirectCoefficientNormalJetPlan3D
build_direct_coefficient_normal_jet_plan_3d(
    const NativeNurbsDensitySpace3D& density,
    int patch,
    double u,
    double v,
    const LocalOrthonormalFrame3D& frame);

// Geometry-reusing counterpart of the value-jet overload above.
[[nodiscard]] DirectCoefficientNormalJetPlan3D
build_direct_coefficient_normal_jet_plan_3d(
    const NativeNurbsDensitySpace3D& density,
    int patch,
    double u,
    double v,
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame);

// Analytic known Dirichlet data for the normal-density formulation.  The
// ambient gradient/Hessian describe any smooth ambient extension of g_D;
// the conversion below retains exactly the value, tangential gradient and
// tangential Hessian of its surface restriction, including curvature terms.
struct KnownDirichletValueGradientHessian3D {
    double value = 0.0;
    Eigen::Vector3d ambient_gradient = Eigen::Vector3d::Zero();
    Eigen::Matrix3d ambient_hessian = Eigen::Matrix3d::Zero();
};

using KnownDirichletValueGradientHessianCallback3D = std::function<
    KnownDirichletValueGradientHessian3D(
        int patch,
        double u,
        double v,
        const Eigen::Vector3d& point,
        const Eigen::Vector3d& normal)>;

using KnownDirichletJetCallback3D = std::function<
    ValueJet3D(
        int patch,
        double u,
        double v,
        const NativeSurfaceParameterJet3D& geometry,
        const LocalOrthonormalFrame3D& frame)>;

[[nodiscard]] ValueJet3D
known_dirichlet_jet_from_ambient_derivatives_3d(
    const KnownDirichletValueGradientHessian3D& data,
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame);

[[nodiscard]] ValueJet3D evaluate_known_dirichlet_jet_3d(
    const KnownDirichletValueGradientHessianCallback3D& callback,
    int patch,
    double u,
    double v,
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame);

// Convenient callback contract for known Neumann data.  ambient_gradient may
// be any smooth ambient extension: only its two tangential projections are
// retained.  This gives (J1,J1_s,J1_t) analytically when manufactured or
// application data expose a gradient callback.
struct KnownNeumannValueGradient3D {
    double value = 0.0;
    Eigen::Vector3d ambient_gradient = Eigen::Vector3d::Zero();
};

using KnownNeumannValueGradientCallback3D = std::function<
    KnownNeumannValueGradient3D(
        int patch,
        double u,
        double v,
        const Eigen::Vector3d& point,
        const Eigen::Vector3d& normal)>;

using KnownNeumannJetCallback3D = std::function<
    NormalJet3D(
        int patch,
        double u,
        double v,
        const NativeSurfaceParameterJet3D& geometry,
        const LocalOrthonormalFrame3D& frame)>;

[[nodiscard]] NormalJet3D known_neumann_jet_from_ambient_gradient_3d(
    const KnownNeumannValueGradient3D& data,
    const LocalOrthonormalFrame3D& frame);

[[nodiscard]] NormalJet3D evaluate_known_neumann_jet_3d(
    const KnownNeumannValueGradientCallback3D& callback,
    int patch,
    double u,
    double v,
    const NativeSurfaceParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame);

struct DirectCoefficientCauchyRow3D {
    NativeDensityC0Stencil3D unknown_value_row;
    double known_normal_offset = 0.0;
};

[[nodiscard]] DirectCoefficientCauchyRow3D
compose_direct_coefficient_cauchy_row_3d(
    const DirectCoefficientValueJetPlan3D& value_plan,
    const CauchyPolynomialWeights3D& weights,
    const NormalJet3D& known_normal_jet);

} // namespace kfbim::app3d
