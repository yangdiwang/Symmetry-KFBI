#pragma once

#include "src/support/cauchy/direct_coefficient_cauchy_3d.hpp"

namespace kfbim::app3d {

// Surface derivative ordering (not factorial-normalized):
// value,s,t,ss,st,tt,sss,sst,stt,ttt and value,s,t,ss,st,tt.
using CubicValueJet3D = Eigen::Matrix<double, 10, 1>;
using CubicNormalJet3D = Eigen::Matrix<double, 6, 1>;

struct NativeSurfaceCubicParameterJet3D {
    NativeSurfaceParameterJet3D lower;
    Eigen::Vector3d x_uuu = Eigen::Vector3d::Zero();
    Eigen::Vector3d x_uuv = Eigen::Vector3d::Zero();
    Eigen::Vector3d x_uvv = Eigen::Vector3d::Zero();
    Eigen::Vector3d x_vvv = Eigen::Vector3d::Zero();
};

struct TangentGraphThirdJet3D {
    TangentGraphHessian3D hessian;
    double sss = 0.0;
    double sst = 0.0;
    double stt = 0.0;
    double ttt = 0.0;
};

struct CauchyMonomial3D {
    int s = 0, t = 0, r = 0;
    [[nodiscard]] int degree() const noexcept { return s + t + r; }
};

// Lexicographic exponent ordering deliberately differs between P2 and P3.
// Never take head<10>() of the cubic coefficient vector.
[[nodiscard]] const std::array<CauchyMonomial3D, 20>& cubic_cauchy_powers_3d();
[[nodiscard]] const std::array<CauchyMonomial3D, 10>& quadratic_cauchy_powers_3d();
[[nodiscard]] Eigen::Matrix<double, 10, 20> cubic_to_quadratic_selection_3d();

struct CubicCauchyPolynomial3D {
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    LocalOrthonormalFrame3D frame;
    // Coefficient of s^a t^b r^c (derivative / (a! b! c!)).
    Eigen::Matrix<double, 20, 1> coefficients = Eigen::Matrix<double, 20, 1>::Zero();
    [[nodiscard]] double evaluate(const Eigen::Vector3d& target, int degree = 3) const;
    [[nodiscard]] Eigen::Matrix<double, 10, 1> quadratic_coefficients() const;
};

[[nodiscard]] NativeSurfaceCubicParameterJet3D native_surface_cubic_parameter_jet_3d(
    const NativeNurbsDensitySpace3D& density, int patch, double u, double v);
// A geometry-only patch query does not require a closed density topology.
// Parameters still use the normalized [0,1]^2 patch chart.
[[nodiscard]] NativeSurfaceCubicParameterJet3D native_surface_cubic_parameter_jet_3d(
    const geometry3d::NurbsSurfacePatch3D& patch, double u, double v);
[[nodiscard]] Eigen::Matrix<double, 10, 10> parameter_to_cubic_cauchy_jet_matrix_3d(
    const NativeSurfaceCubicParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame);
[[nodiscard]] TangentGraphThirdJet3D tangent_graph_third_jet_3d(
    const NativeSurfaceCubicParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame);

// Immutable geometry-local maps: no density fitting and no solve at apply.
// Laplace closure is appropriate for the harmonic jump used by this driver.
struct CubicCauchyClosure3D {
    Eigen::Matrix<double, 20, 10> value_map;
    Eigen::Matrix<double, 20, 6> normal_map;
    [[nodiscard]] CubicCauchyPolynomial3D polynomial(
        const Eigen::Vector3d& center, const LocalOrthonormalFrame3D& frame,
        const CubicValueJet3D& value, const CubicNormalJet3D& normal) const;
};
[[nodiscard]] CubicCauchyClosure3D build_cubic_cauchy_closure_3d(
    const TangentGraphThirdJet3D& graph);

struct DirectCoefficientCubicCauchyPlan3D {
    int patch = -1;
    double u = 0.0, v = 0.0;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    LocalOrthonormalFrame3D frame;
    TangentGraphThirdJet3D graph;
    CubicCauchyClosure3D closure;
    std::array<NativeDensityC0Stencil3D, 10> value_rows;
    std::array<NativeDensityC0Stencil3D, 6> normal_rows;
    DirectCoefficientJetDiagnostics3D diagnostics;
    [[nodiscard]] DirectCoefficientValueJetPlan3D lower_value_plan() const;
    [[nodiscard]] DirectCoefficientNormalJetPlan3D lower_normal_plan() const;
    [[nodiscard]] CubicValueJet3D evaluate_value_jet(
        const Eigen::Ref<const Eigen::VectorXd>& coefficients) const;
    [[nodiscard]] CubicNormalJet3D evaluate_normal_jet(
        const Eigen::Ref<const Eigen::VectorXd>& coefficients) const;
    // displacement is in world coordinates, measured from this plan's center.
    // degree=2 reuses the same map and only removes total-degree-three terms.
    [[nodiscard]] Eigen::Matrix<double, 1, 10> value_weights(
        const Eigen::Vector3d& displacement, int degree = 3) const;
    [[nodiscard]] Eigen::Matrix<double, 1, 6> normal_weights(
        const Eigen::Vector3d& displacement, int degree = 3) const;
    [[nodiscard]] NativeDensityC0Stencil3D compose_value_row(
        const Eigen::Vector3d& displacement, int degree = 3) const;
    [[nodiscard]] NativeDensityC0Stencil3D compose_normal_row(
        const Eigen::Vector3d& displacement, int degree = 3) const;
};

[[nodiscard]] DirectCoefficientCubicCauchyPlan3D build_direct_coefficient_cubic_cauchy_plan_3d(
    const NativeNurbsDensitySpace3D& density, int patch, double u, double v,
    const LocalOrthonormalFrame3D& frame);
[[nodiscard]] DirectCoefficientCubicCauchyPlan3D build_direct_coefficient_cubic_cauchy_plan_3d(
    const NativeNurbsDensitySpace3D& density, int patch, double u, double v,
    const NativeSurfaceCubicParameterJet3D& geometry,
    const LocalOrthonormalFrame3D& frame);

// Ambient derivatives of a smooth known extension U.  third[k](i,j) is
// d_i d_j d_k U, symmetric in all three indices.  No fitting is performed.
struct KnownAmbientThird3D {
    double value = 0.0;
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
    Eigen::Matrix3d hessian = Eigen::Matrix3d::Zero();
    std::array<Eigen::Matrix3d, 3> third{{Eigen::Matrix3d::Zero(),
        Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()}};
};
// J0 = U|Gamma.  J1 = n dot grad U, including the derivatives of n.
[[nodiscard]] CubicValueJet3D known_cubic_dirichlet_jet_3d(
    const KnownAmbientThird3D& data, const TangentGraphThirdJet3D& graph,
    const LocalOrthonormalFrame3D& frame);
[[nodiscard]] CubicNormalJet3D known_cubic_neumann_jet_3d(
    const KnownAmbientThird3D& data, const TangentGraphThirdJet3D& graph,
    const LocalOrthonormalFrame3D& frame);

} // namespace kfbim::app3d
