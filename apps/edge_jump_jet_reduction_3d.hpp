#pragma once

#include <Eigen/Core>
#include <Eigen/QR>

namespace kfbim::app3d {

struct EdgeJumpJetTargetProjection3D {
    // Compatible targets in the original (unscaled) mortar-row convention.
    Eigen::MatrixXd targets;
    // Mean-zero coordinate lifts whose images are targets.
    Eigen::MatrixXd lifts;
    // Projection defect after normalizing every mortar row by its 2-norm.
    double normalized_projection_linf = 0.0;
    double mean_null_denominator = 0.0;
    double mean_null_residual_linf = 0.0;
    double mean_null_identity_residual = 0.0;
    double lift_mean_residual_linf = 0.0;
};

// Projects one or more possibly inconsistent affine mortar targets onto the
// range of D restricted to the intrinsic mean-zero coordinate space.  This is
// needed when algebraically dependent patch-pair equations receive slightly
// inconsistent right-hand sides from a broken normal-density fit.
[[nodiscard]] EdgeJumpJetTargetProjection3D
project_edge_jump_jet_targets_to_mean_zero_range_3d(
    const Eigen::MatrixXd& edge_constraint_matrix,
    const Eigen::MatrixXd& affine_targets,
    const Eigen::VectorXd& mean_row,
    double relative_rank_tolerance = 1.0e-11,
    double minimum_mean_null_denominator = 1.0e-8);

// Canonical sharp-edge frame and the two independent compatibility data for
//
//   grad_Gamma1(phi_1) + psi_1 n_1
//       = grad_Gamma2(phi_2) + psi_2 n_2.
//
// The common edge-tangent component is omitted: when phi_1 == phi_2 as a
// function on the complete edge, equality of that component is already the
// derivative of the C0 constraint.  The returned conormals use the canonical
// orientation induced by the two supplied normals; a caller using outward
// patch-edge conormals must apply the corresponding signs to its derivative
// rows.
struct EdgeJumpJetCompatibility3D {
    Eigen::Vector3d tangent = Eigen::Vector3d::Zero();
    Eigen::Vector3d first_conormal = Eigen::Vector3d::Zero();
    Eigen::Vector3d second_conormal = Eigen::Vector3d::Zero();
    double normal_dot = 0.0;
    double normal_cross_norm = 0.0;
    double first_conormal_derivative = 0.0;
    double second_conormal_derivative = 0.0;
};

[[nodiscard]] EdgeJumpJetCompatibility3D
make_edge_jump_jet_compatibility_3d(
    const Eigen::Vector3d& first_normal,
    const Eigen::Vector3d& second_normal,
    double first_normal_jump,
    double second_normal_jump,
    double minimum_normal_cross_norm = 1.0e-8);

struct EdgeJumpJetReductionOptions3D {
    // Applied after every constraint row has been normalized.  The same
    // threshold is used by the rank-revealing QR and minimum-norm solve.
    double relative_rank_tolerance = 1.0e-11;
    double constraint_residual_tolerance = 2.0e-10;
    double minimum_border_norm = 1.0e-13;
};

// Algebraic reduction for affine edge-jump constraints coupled to the
// operator-flux multiplier.
//
// Let x be the full value-density coordinate vector and lambda the flux
// correction used by the Neumann equation.  The input convention is
//
//       D x = d_g + lambda d_lambda,     m^T x = prescribed_mean.
//
// The class constructs
//
//       x = x_g + Q y + lambda x_lambda,
//
// where Q has orthonormal columns and exactly spans null([D; m^T]).  Thus the
// mean constraint is intrinsic to every Krylov vector; it is not appended as
// a bordered row.
//
// If the unreduced equation is A x + b lambda = rhs, substitution gives
//
//   A Q y + (b + A x_lambda) lambda = rhs - A x_g.
//
// project_and_test() applies the orthogonal range projection that eliminates
// this effective border, followed by the Galerkin test Q^T.  It produces the
// square reduced equation
//
//   Q^T P_b A Q y = Q^T P_b (rhs - A x_g).
//
// The discarded border component determines lambda through recover_lambda().
// Callers must still audit the complete unreduced residual: additional edge
// constraints deliberately discard equations orthogonal to range(Q) as a
// Galerkin choice.
class EdgeJumpJetAffineReduction3D {
public:
    EdgeJumpJetAffineReduction3D(
        const Eigen::MatrixXd& edge_constraint_matrix,
        const Eigen::VectorXd& data_constraint_target,
        const Eigen::VectorXd& lambda_constraint_target,
        const Eigen::VectorXd& mean_row,
        double prescribed_mean = 0.0,
        EdgeJumpJetReductionOptions3D options = {});

    [[nodiscard]] int full_size() const noexcept;
    [[nodiscard]] int reduced_size() const noexcept;
    [[nodiscard]] int constraint_rank() const noexcept;

    // Materializes Q only for diagnostics/tests.  Solver paths should use
    // homogeneous_coordinates() and project_and_test(), which apply the
    // Householder representation without forming the dense nullspace basis.
    [[nodiscard]] const Eigen::MatrixXd& homogeneous_basis() const;
    [[nodiscard]] const Eigen::VectorXd& data_lift() const noexcept;
    [[nodiscard]] const Eigen::VectorXd& lambda_lift() const noexcept;

    [[nodiscard]] Eigen::VectorXd homogeneous_coordinates(
        const Eigen::Ref<const Eigen::VectorXd>& reduced) const;
    [[nodiscard]] Eigen::VectorXd expand(
        const Eigen::Ref<const Eigen::VectorXd>& reduced,
        double lambda) const;

    // raw_border is b and applied_lambda_lift is A*x_lambda.
    [[nodiscard]] Eigen::VectorXd effective_border(
        const Eigen::Ref<const Eigen::VectorXd>& raw_border,
        const Eigen::Ref<const Eigen::VectorXd>& applied_lambda_lift) const;

    // Returns Q^T (I - bb^T/(b^Tb)) values.
    [[nodiscard]] Eigen::VectorXd project_and_test(
        const Eigen::Ref<const Eigen::VectorXd>& values,
        const Eigen::Ref<const Eigen::VectorXd>& effective_border) const;

    // Input is rhs - A*x_g - A*Q*y.  The result is the least-squares scalar
    // coefficient along the effective border.
    [[nodiscard]] double recover_lambda(
        const Eigen::Ref<const Eigen::VectorXd>& remaining_range_vector,
        const Eigen::Ref<const Eigen::VectorXd>& effective_border) const;

    [[nodiscard]] double data_constraint_residual() const noexcept;
    [[nodiscard]] double lambda_constraint_residual() const noexcept;
    [[nodiscard]] double nullspace_constraint_residual() const noexcept;
    [[nodiscard]] double nullspace_orthogonality_residual() const noexcept;

private:
    void require_full_size(Eigen::Index size, const char* name) const;
    void require_reduced_size(Eigen::Index size, const char* name) const;
    Eigen::VectorXd to_householder_coordinates(
        const Eigen::Ref<const Eigen::VectorXd>& values) const;
    Eigen::VectorXd from_householder_coordinates(
        const Eigen::Ref<const Eigen::VectorXd>& values) const;
    double checked_border_squared_norm(
        const Eigen::Ref<const Eigen::VectorXd>& border) const;

    EdgeJumpJetReductionOptions3D options_;
    Eigen::MatrixXd constraints_;
    Eigen::VectorXd data_target_;
    Eigen::VectorXd lambda_target_;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> row_space_factor_;
    mutable Eigen::MatrixXd homogeneous_basis_;
    Eigen::VectorXd data_lift_;
    Eigen::VectorXd lambda_lift_;
    int constraint_rank_ = 0;
    double data_constraint_residual_ = 0.0;
    double lambda_constraint_residual_ = 0.0;
    double nullspace_constraint_residual_ = 0.0;
    double nullspace_orthogonality_residual_ = 0.0;
};

} // namespace kfbim::app3d
