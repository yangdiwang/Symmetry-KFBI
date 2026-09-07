#pragma once

#include <Eigen/Core>

#include <vector>

namespace kfbim::app3d {

using SurfacePointMatrix3D = Eigen::Matrix<double, Eigen::Dynamic, 3>;
using TangentCoordinateMatrix3D = Eigen::Matrix<double, Eigen::Dynamic, 2>;
using ValueJet3D = Eigen::Matrix<double, 6, 1>;
using NormalJet3D = Eigen::Matrix<double, 3, 1>;
using ValueCauchyWeightRow3D = Eigen::Matrix<double, 1, 6>;
using NormalCauchyWeightRow3D = Eigen::Matrix<double, 1, 3>;

// The frame is right handed: tangent1 x tangent2 = normal.  The normal is
// the positive graph-height direction used by TangentGraphHessian3D.
struct LocalOrthonormalFrame3D {
    Eigen::Vector3d tangent1 = Eigen::Vector3d::UnitX();
    Eigen::Vector3d tangent2 = Eigen::Vector3d::UnitY();
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
};

[[nodiscard]] LocalOrthonormalFrame3D
make_local_orthonormal_frame_3d(const Eigen::Vector3d& normal);

[[nodiscard]] LocalOrthonormalFrame3D
make_local_orthonormal_frame_3d(
    const Eigen::Vector3d& normal,
    const Eigen::Vector3d& tangent1_hint);

[[nodiscard]] bool is_local_orthonormal_frame_3d(
    const LocalOrthonormalFrame3D& frame,
    double tolerance = 1.0e-10);

// Throws std::invalid_argument when the frame is not finite, orthonormal and
// right handed.
void validate_local_orthonormal_frame_3d(
    const LocalOrthonormalFrame3D& frame,
    double tolerance = 1.0e-10);

// Local surface convention:
//
//   x(s,t) = center + s*tangent1 + t*tangent2 + r(s,t)*normal,
//   r(s,t) = 0.5*(h11*s^2 + 2*h12*s*t + h22*t^2) + O(|(s,t)|^3).
struct TangentGraphHessian3D {
    double h11 = 0.0;
    double h12 = 0.0;
    double h22 = 0.0;

    [[nodiscard]] Eigen::Matrix2d matrix() const;
    [[nodiscard]] bool all_finite() const;
};

// w0 multiplies (J0, J0_s, J0_t, J0_ss, J0_st, J0_tt), while w1
// multiplies (J1, J1_s, J1_t).  local_displacement stores (s,t,r).
struct CauchyPolynomialWeights3D {
    Eigen::Vector3d local_displacement = Eigen::Vector3d::Zero();
    ValueCauchyWeightRow3D w0 = ValueCauchyWeightRow3D::Zero();
    NormalCauchyWeightRow3D w1 = NormalCauchyWeightRow3D::Zero();

    [[nodiscard]] double apply_value_jet(const ValueJet3D& jet) const;
    [[nodiscard]] double apply_normal_jet(const NormalJet3D& jet) const;
};

// Direct C++ counterpart of pweights_general in
// docs/kfbi_general_cap_exterior_trace.py.  displacement is target-center.
[[nodiscard]] CauchyPolynomialWeights3D pweights_general_3d(
    const Eigen::Vector3d& displacement,
    const LocalOrthonormalFrame3D& frame,
    const TangentGraphHessian3D& graph_hessian);

[[nodiscard]] CauchyPolynomialWeights3D
cauchy_polynomial_weights_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const TangentGraphHessian3D& graph_hessian,
    const Eigen::Vector3d& target);

struct JetRecoveryOptions3D {
    // Positive least-squares weights.  An empty vector means unit weights.
    Eigen::VectorXd sample_weights;

    // Physical tangent length used to nondimensionalize s and t.  A
    // non-positive value selects the maximum sample tangent radius.
    double coordinate_scale = 0.0;

    // Singular values <= tolerance*sigma_max are discarded.
    double relative_singular_tolerance = 3.0e-12;

    // A truncated recovery is occasionally useful for diagnostics.  Normal
    // KFBI construction should retain the default and reject deficient sets.
    bool require_full_rank = true;
};

struct JetRecoveryDiagnostics3D {
    int sample_count = 0;
    int jet_dimension = 0;
    int numerical_rank = 0;
    double coordinate_scale = 0.0;
    double relative_singular_tolerance = 0.0;
    double largest_singular_value = 0.0;
    double smallest_singular_value = 0.0;
    double smallest_retained_singular_value = 0.0;
    double condition_number = 0.0;
    double retained_condition_number = 0.0;
    double polynomial_reproduction_error = 0.0;
    Eigen::VectorXd singular_values;

    [[nodiscard]] bool full_rank() const noexcept;
};

// matrix maps sample values to the physical, crossing-centered jet.  For a
// value recovery it has 6 rows; for a normal recovery it has 3 rows.
struct JetRecoveryMatrix3D {
    Eigen::MatrixXd matrix;
    JetRecoveryDiagnostics3D diagnostics;

    [[nodiscard]] Eigen::VectorXd recover(
        const Eigen::VectorXd& sample_values) const;
};

// Convert physical surface samples to crossing-centered graph coordinates.
// The normal coordinate is deliberately omitted: a surface density is fitted
// as a function of graph coordinates (s,t), not ambient Cartesian position.
[[nodiscard]] TangentCoordinateMatrix3D centered_tangent_coordinates_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const SurfacePointMatrix3D& sample_points);

[[nodiscard]] TangentCoordinateMatrix3D centered_tangent_coordinates_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const std::vector<Eigen::Vector3d>& sample_points);

// Second-order Taylor recovery with coefficient ordering
// (f, f_s, f_t, f_ss, f_st, f_tt).
[[nodiscard]] JetRecoveryMatrix3D build_value_jet_recovery_3d(
    const TangentCoordinateMatrix3D& centered_tangent_coordinates,
    const JetRecoveryOptions3D& options = {});

[[nodiscard]] JetRecoveryMatrix3D build_value_jet_recovery_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const SurfacePointMatrix3D& sample_points,
    const JetRecoveryOptions3D& options = {});

[[nodiscard]] JetRecoveryMatrix3D build_value_jet_recovery_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const std::vector<Eigen::Vector3d>& sample_points,
    const JetRecoveryOptions3D& options = {});

// First-order Taylor recovery with coefficient ordering (g, g_s, g_t).
[[nodiscard]] JetRecoveryMatrix3D build_normal_jet_recovery_3d(
    const TangentCoordinateMatrix3D& centered_tangent_coordinates,
    const JetRecoveryOptions3D& options = {});

[[nodiscard]] JetRecoveryMatrix3D build_normal_jet_recovery_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const SurfacePointMatrix3D& sample_points,
    const JetRecoveryOptions3D& options = {});

[[nodiscard]] JetRecoveryMatrix3D build_normal_jet_recovery_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const std::vector<Eigen::Vector3d>& sample_points,
    const JetRecoveryOptions3D& options = {});

// Geometry and the two density-independent sample recovery maps cached for one
// Cartesian-grid/surface crossing.
struct CrossingCauchyPlan3D {
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    LocalOrthonormalFrame3D frame;
    TangentGraphHessian3D graph_hessian;
    JetRecoveryMatrix3D value_jet_recovery;
    JetRecoveryMatrix3D normal_jet_recovery;

    [[nodiscard]] CauchyPolynomialWeights3D weights_at(
        const Eigen::Vector3d& target) const;
};

[[nodiscard]] CrossingCauchyPlan3D build_crossing_cauchy_plan_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const TangentGraphHessian3D& graph_hessian,
    const SurfacePointMatrix3D& value_sample_points,
    const SurfacePointMatrix3D& normal_sample_points,
    const JetRecoveryOptions3D& value_options = {},
    const JetRecoveryOptions3D& normal_options = {});

[[nodiscard]] CrossingCauchyPlan3D build_crossing_cauchy_plan_3d(
    const Eigen::Vector3d& center,
    const LocalOrthonormalFrame3D& frame,
    const TangentGraphHessian3D& graph_hessian,
    const std::vector<Eigen::Vector3d>& value_sample_points,
    const std::vector<Eigen::Vector3d>& normal_sample_points,
    const JetRecoveryOptions3D& value_options = {},
    const JetRecoveryOptions3D& normal_options = {});

} // namespace kfbim::app3d
