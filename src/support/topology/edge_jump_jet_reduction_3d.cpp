#include "src/support/topology/edge_jump_jet_reduction_3d.hpp"

#include <Eigen/Geometry>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace kfbim::app3d {
namespace {

double infinity_norm(const Eigen::VectorXd& values)
{
    return values.size() == 0
        ? 0.0
        : values.lpNorm<Eigen::Infinity>();
}

void validate_options(const EdgeJumpJetReductionOptions3D& options)
{
    if (!(options.relative_rank_tolerance > 0.0)
        || !(options.relative_rank_tolerance < 1.0)
        || !std::isfinite(options.relative_rank_tolerance)) {
        throw std::invalid_argument(
            "edge-jump rank tolerance must be finite and in (0,1)");
    }
    if (!(options.constraint_residual_tolerance > 0.0)
        || !std::isfinite(options.constraint_residual_tolerance)) {
        throw std::invalid_argument(
            "edge-jump constraint tolerance must be finite and positive");
    }
    if (!(options.minimum_border_norm > 0.0)
        || !std::isfinite(options.minimum_border_norm)) {
        throw std::invalid_argument(
            "edge-jump minimum border norm must be finite and positive");
    }
}

Eigen::VectorXd minimum_norm_lift(
    const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>& factor,
    const Eigen::MatrixXd& normalized_constraints,
    const Eigen::VectorXd& normalized_target,
    double residual_tolerance,
    const char* name)
{
    Eigen::VectorXd lift = factor.solve(normalized_target);
    if (!lift.allFinite())
        throw std::runtime_error(std::string(name) + " lift is not finite");

    const double residual = infinity_norm(
        normalized_constraints * lift - normalized_target);
    const double scale = std::max(1.0, infinity_norm(normalized_target));
    if (!(residual <= residual_tolerance * scale)) {
        throw std::runtime_error(
            std::string(name) + " edge-jump constraints are inconsistent");
    }
    return lift;
}

} // namespace

EdgeJumpJetTargetProjection3D
project_edge_jump_jet_targets_to_mean_zero_range_3d(
    const Eigen::MatrixXd& edge_constraint_matrix,
    const Eigen::MatrixXd& affine_targets,
    const Eigen::VectorXd& mean_row,
    double relative_rank_tolerance,
    double minimum_mean_null_denominator)
{
    const Eigen::Index row_count = edge_constraint_matrix.rows();
    const Eigen::Index coordinate_count = edge_constraint_matrix.cols();
    if (row_count <= 0 || coordinate_count <= 1
        || affine_targets.rows() != row_count
        || affine_targets.cols() <= 0
        || mean_row.size() != coordinate_count) {
        throw std::invalid_argument(
            "edge-jump target projection dimensions are inconsistent");
    }
    if (!edge_constraint_matrix.allFinite()
        || !affine_targets.allFinite() || !mean_row.allFinite()) {
        throw std::invalid_argument(
            "edge-jump target projection inputs must be finite");
    }
    if (!(relative_rank_tolerance > 0.0)
        || !(relative_rank_tolerance < 1.0)
        || !std::isfinite(relative_rank_tolerance)
        || !(minimum_mean_null_denominator > 0.0)
        || !(minimum_mean_null_denominator < 1.0)
        || !std::isfinite(minimum_mean_null_denominator)) {
        throw std::invalid_argument(
            "edge-jump target projection tolerances are invalid");
    }

    Eigen::MatrixXd normalized_matrix = edge_constraint_matrix;
    Eigen::MatrixXd normalized_targets = affine_targets;
    Eigen::VectorXd row_norms(row_count);
    for (Eigen::Index row = 0; row < row_count; ++row) {
        const double norm = edge_constraint_matrix.row(row).norm();
        if (!(norm > 128.0 * std::numeric_limits<double>::epsilon())) {
            throw std::invalid_argument(
                "edge-jump target projection contains a zero row");
        }
        row_norms[row] = norm;
        normalized_matrix.row(row) /= norm;
        normalized_targets.row(row) /= norm;
    }
    const double mean_norm = mean_row.norm();
    if (!(mean_norm > 128.0 * std::numeric_limits<double>::epsilon())) {
        throw std::invalid_argument(
            "edge-jump target projection contains a zero mean row");
    }
    const Eigen::VectorXd mean_unit = mean_row / mean_norm;

    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> factor;
    factor.setThreshold(relative_rank_tolerance);
    factor.compute(normalized_matrix);
    Eigen::MatrixXd lifts = factor.solve(normalized_targets);
    const Eigen::VectorXd mean_null_direction = mean_unit
        - factor.solve(normalized_matrix * mean_unit);
    const double mean_denominator =
        mean_unit.dot(mean_null_direction);
    if (!(mean_denominator > minimum_mean_null_denominator)
        || !std::isfinite(mean_denominator)) {
        throw std::runtime_error(
            "edge-jump derivative constraints do not retain a stable "
            "constant mean-null direction");
    }
    for (Eigen::Index column = 0; column < lifts.cols(); ++column) {
        lifts.col(column) -= mean_null_direction
            * (mean_unit.dot(lifts.col(column)) / mean_denominator);
    }
    const Eigen::MatrixXd projected_normalized_targets =
        normalized_matrix * lifts;

    EdgeJumpJetTargetProjection3D result;
    result.targets = projected_normalized_targets;
    for (Eigen::Index row = 0; row < row_count; ++row)
        result.targets.row(row) *= row_norms[row];
    result.lifts = std::move(lifts);
    result.normalized_projection_linf =
        (projected_normalized_targets - normalized_targets)
            .cwiseAbs().maxCoeff();
    result.mean_null_denominator = mean_denominator;
    result.mean_null_residual_linf =
        (normalized_matrix * mean_null_direction)
            .lpNorm<Eigen::Infinity>();
    result.mean_null_identity_residual = std::abs(
        mean_denominator - mean_null_direction.squaredNorm());
    result.lift_mean_residual_linf =
        (mean_unit.transpose() * result.lifts)
            .cwiseAbs().maxCoeff();
    if (!result.targets.allFinite() || !result.lifts.allFinite()
        || !std::isfinite(result.normalized_projection_linf)
        || !std::isfinite(result.mean_null_residual_linf)
        || !std::isfinite(result.mean_null_identity_residual)
        || !std::isfinite(result.lift_mean_residual_linf)) {
        throw std::runtime_error(
            "edge-jump target projection produced non-finite data");
    }
    return result;
}

EdgeJumpJetCompatibility3D make_edge_jump_jet_compatibility_3d(
    const Eigen::Vector3d& first_normal,
    const Eigen::Vector3d& second_normal,
    double first_normal_jump,
    double second_normal_jump,
    double minimum_normal_cross_norm)
{
    if (!first_normal.allFinite() || !second_normal.allFinite()
        || !std::isfinite(first_normal_jump)
        || !std::isfinite(second_normal_jump)) {
        throw std::invalid_argument(
            "edge-jump compatibility data must be finite");
    }
    if (!(minimum_normal_cross_norm > 0.0)
        || !(minimum_normal_cross_norm < 1.0)
        || !std::isfinite(minimum_normal_cross_norm)) {
        throw std::invalid_argument(
            "edge-jump minimum normal cross norm must lie in (0,1)");
    }
    const double first_norm = first_normal.norm();
    const double second_norm = second_normal.norm();
    if (!(first_norm > 0.0) || !(second_norm > 0.0))
        throw std::invalid_argument("edge-jump normal is zero");

    const Eigen::Vector3d n1 = first_normal / first_norm;
    const Eigen::Vector3d n2 = second_normal / second_norm;
    const Eigen::Vector3d cross = n1.cross(n2);
    const double sine = cross.norm();
    if (!(sine >= minimum_normal_cross_norm)) {
        throw std::invalid_argument(
            "edge-jump compatibility is singular at a G1 or nearly G1 seam");
    }

    EdgeJumpJetCompatibility3D result;
    result.normal_dot = std::clamp(n1.dot(n2), -1.0, 1.0);
    result.normal_cross_norm = sine;
    result.tangent = cross / sine;
    result.first_conormal =
        (n2 - result.normal_dot * n1) / sine;
    result.second_conormal =
        (n1 - result.normal_dot * n2) / sine;
    result.first_conormal_derivative =
        (second_normal_jump
         - result.normal_dot * first_normal_jump) / sine;
    result.second_conormal_derivative =
        (first_normal_jump
         - result.normal_dot * second_normal_jump) / sine;
    return result;
}

EdgeJumpJetAffineReduction3D::EdgeJumpJetAffineReduction3D(
    const Eigen::MatrixXd& edge_constraint_matrix,
    const Eigen::VectorXd& data_constraint_target,
    const Eigen::VectorXd& lambda_constraint_target,
    const Eigen::VectorXd& mean_row,
    double prescribed_mean,
    EdgeJumpJetReductionOptions3D options)
    : options_(options)
{
    validate_options(options_);
    const Eigen::Index size = mean_row.size();
    if (size <= 1)
        throw std::invalid_argument(
            "edge-jump reduction requires at least two full coordinates");
    if (edge_constraint_matrix.cols() != size
        || data_constraint_target.size() != edge_constraint_matrix.rows()
        || lambda_constraint_target.size()
               != edge_constraint_matrix.rows()) {
        throw std::invalid_argument(
            "edge-jump reduction input dimensions are inconsistent");
    }
    if (!edge_constraint_matrix.allFinite()
        || !data_constraint_target.allFinite()
        || !lambda_constraint_target.allFinite()
        || !mean_row.allFinite() || !std::isfinite(prescribed_mean)) {
        throw std::invalid_argument(
            "edge-jump reduction inputs must be finite");
    }

    constraints_.resize(edge_constraint_matrix.rows() + 1, size);
    constraints_.topRows(edge_constraint_matrix.rows()) =
        edge_constraint_matrix;
    constraints_.bottomRows(1) = mean_row.transpose();
    data_target_.resize(constraints_.rows());
    data_target_.head(data_constraint_target.size()) =
        data_constraint_target;
    data_target_[data_target_.size() - 1] = prescribed_mean;
    lambda_target_.resize(constraints_.rows());
    lambda_target_.head(lambda_constraint_target.size()) =
        lambda_constraint_target;
    lambda_target_[lambda_target_.size() - 1] = 0.0;

    Eigen::MatrixXd normalized_constraints = constraints_;
    Eigen::VectorXd normalized_data = data_target_;
    Eigen::VectorXd normalized_lambda = lambda_target_;
    for (Eigen::Index row = 0; row < constraints_.rows(); ++row) {
        const double norm = constraints_.row(row).norm();
        if (!(norm > 128.0 * std::numeric_limits<double>::epsilon())) {
            throw std::invalid_argument(
                "edge-jump reduction contains a zero constraint row");
        }
        normalized_constraints.row(row) /= norm;
        normalized_data[row] /= norm;
        normalized_lambda[row] /= norm;
    }

    row_space_factor_.setThreshold(options_.relative_rank_tolerance);
    row_space_factor_.compute(normalized_constraints.transpose());
    constraint_rank_ = static_cast<int>(row_space_factor_.rank());
    const int nullity = static_cast<int>(size) - constraint_rank_;
    if (nullity <= 0)
        throw std::runtime_error(
            "edge-jump and mean constraints eliminate every coordinate");
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> factor;
    factor.setThreshold(options_.relative_rank_tolerance);
    factor.compute(normalized_constraints);
    data_lift_ = minimum_norm_lift(
        factor, normalized_constraints, normalized_data,
        options_.constraint_residual_tolerance,
        "data");
    lambda_lift_ = minimum_norm_lift(
        factor, normalized_constraints, normalized_lambda,
        options_.constraint_residual_tolerance,
        "lambda");

    // Remove the last roundoff-sized nullspace component without ever
    // materializing Q.  The first rank entries are the row-space coordinates
    // of the Householder factorization of C^T.
    const auto retain_row_space = [&](Eigen::VectorXd values) {
        Eigen::VectorXd transformed = to_householder_coordinates(values);
        transformed.tail(nullity).setZero();
        return from_householder_coordinates(transformed);
    };
    data_lift_ = retain_row_space(std::move(data_lift_));
    lambda_lift_ = retain_row_space(std::move(lambda_lift_));

    data_constraint_residual_ = infinity_norm(
        constraints_ * data_lift_ - data_target_);
    lambda_constraint_residual_ = infinity_norm(
        constraints_ * lambda_lift_ - lambda_target_);
    const int audit_columns = std::min(nullity, 8);
    Eigen::MatrixXd audit_basis(size, audit_columns);
    for (int column = 0; column < audit_columns; ++column) {
        Eigen::VectorXd reduced = Eigen::VectorXd::Zero(nullity);
        const int index = audit_columns == 1
            ? 0
            : column * (nullity - 1) / (audit_columns - 1);
        reduced[index] = 1.0;
        audit_basis.col(column) = homogeneous_coordinates(reduced);
    }
    nullspace_constraint_residual_ =
        (constraints_ * audit_basis).cwiseAbs().maxCoeff();
    const Eigen::MatrixXd gram = audit_basis.transpose() * audit_basis;
    nullspace_orthogonality_residual_ =
        (gram - Eigen::MatrixXd::Identity(audit_columns, audit_columns))
            .cwiseAbs().maxCoeff();
}

int EdgeJumpJetAffineReduction3D::full_size() const noexcept
{
    return static_cast<int>(constraints_.cols());
}

int EdgeJumpJetAffineReduction3D::reduced_size() const noexcept
{
    return full_size() - constraint_rank_;
}

int EdgeJumpJetAffineReduction3D::constraint_rank() const noexcept
{
    return constraint_rank_;
}

const Eigen::MatrixXd&
EdgeJumpJetAffineReduction3D::homogeneous_basis() const
{
    if (homogeneous_basis_.rows() != full_size()
        || homogeneous_basis_.cols() != reduced_size()) {
        Eigen::MatrixXd selector = Eigen::MatrixXd::Zero(
            full_size(), reduced_size());
        selector.bottomRows(reduced_size()).setIdentity();
        homogeneous_basis_ = row_space_factor_.householderQ() * selector;
    }
    return homogeneous_basis_;
}

const Eigen::VectorXd& EdgeJumpJetAffineReduction3D::data_lift() const noexcept
{
    return data_lift_;
}

const Eigen::VectorXd&
EdgeJumpJetAffineReduction3D::lambda_lift() const noexcept
{
    return lambda_lift_;
}

Eigen::VectorXd EdgeJumpJetAffineReduction3D::homogeneous_coordinates(
    const Eigen::Ref<const Eigen::VectorXd>& reduced) const
{
    require_reduced_size(reduced.size(), "reduced coordinates");
    Eigen::VectorXd embedded = Eigen::VectorXd::Zero(full_size());
    embedded.tail(reduced_size()) = reduced;
    return from_householder_coordinates(embedded);
}

Eigen::VectorXd EdgeJumpJetAffineReduction3D::expand(
    const Eigen::Ref<const Eigen::VectorXd>& reduced,
    double lambda) const
{
    if (!std::isfinite(lambda))
        throw std::invalid_argument("edge-jump multiplier is not finite");
    return data_lift_ + homogeneous_coordinates(reduced)
         + lambda * lambda_lift_;
}

Eigen::VectorXd EdgeJumpJetAffineReduction3D::effective_border(
    const Eigen::Ref<const Eigen::VectorXd>& raw_border,
    const Eigen::Ref<const Eigen::VectorXd>& applied_lambda_lift) const
{
    require_full_size(raw_border.size(), "raw border");
    require_full_size(applied_lambda_lift.size(), "applied lambda lift");
    if (!raw_border.allFinite() || !applied_lambda_lift.allFinite())
        throw std::invalid_argument("edge-jump border data are not finite");
    Eigen::VectorXd result = raw_border + applied_lambda_lift;
    checked_border_squared_norm(result);
    return result;
}

Eigen::VectorXd EdgeJumpJetAffineReduction3D::project_and_test(
    const Eigen::Ref<const Eigen::VectorXd>& values,
    const Eigen::Ref<const Eigen::VectorXd>& border) const
{
    require_full_size(values.size(), "range vector");
    require_full_size(border.size(), "effective border");
    if (!values.allFinite() || !border.allFinite())
        throw std::invalid_argument(
            "edge-jump projected range data are not finite");
    const double squared_norm = checked_border_squared_norm(border);
    const Eigen::VectorXd projected =
        values - border * (border.dot(values) / squared_norm);
    return to_householder_coordinates(projected).tail(reduced_size());
}

double EdgeJumpJetAffineReduction3D::recover_lambda(
    const Eigen::Ref<const Eigen::VectorXd>& remaining_range_vector,
    const Eigen::Ref<const Eigen::VectorXd>& border) const
{
    require_full_size(remaining_range_vector.size(), "remaining range vector");
    require_full_size(border.size(), "effective border");
    if (!remaining_range_vector.allFinite() || !border.allFinite())
        throw std::invalid_argument(
            "edge-jump multiplier recovery data are not finite");
    return border.dot(remaining_range_vector)
         / checked_border_squared_norm(border);
}

double EdgeJumpJetAffineReduction3D::data_constraint_residual() const noexcept
{
    return data_constraint_residual_;
}

double EdgeJumpJetAffineReduction3D::lambda_constraint_residual() const noexcept
{
    return lambda_constraint_residual_;
}

double
EdgeJumpJetAffineReduction3D::nullspace_constraint_residual() const noexcept
{
    return nullspace_constraint_residual_;
}

double
EdgeJumpJetAffineReduction3D::nullspace_orthogonality_residual() const noexcept
{
    return nullspace_orthogonality_residual_;
}

void EdgeJumpJetAffineReduction3D::require_full_size(
    Eigen::Index size,
    const char* name) const
{
    if (size != full_size())
        throw std::invalid_argument(
            std::string("edge-jump ") + name + " has the wrong size");
}

void EdgeJumpJetAffineReduction3D::require_reduced_size(
    Eigen::Index size,
    const char* name) const
{
    if (size != reduced_size())
        throw std::invalid_argument(
            std::string("edge-jump ") + name + " has the wrong size");
}

Eigen::VectorXd EdgeJumpJetAffineReduction3D::to_householder_coordinates(
    const Eigen::Ref<const Eigen::VectorXd>& values) const
{
    require_full_size(values.size(), "Householder input");
    return row_space_factor_.householderQ().adjoint() * values;
}

Eigen::VectorXd EdgeJumpJetAffineReduction3D::from_householder_coordinates(
    const Eigen::Ref<const Eigen::VectorXd>& values) const
{
    require_full_size(values.size(), "Householder coordinates");
    return row_space_factor_.householderQ() * values;
}

double EdgeJumpJetAffineReduction3D::checked_border_squared_norm(
    const Eigen::Ref<const Eigen::VectorXd>& border) const
{
    const double squared_norm = border.squaredNorm();
    const double minimum_squared =
        options_.minimum_border_norm * options_.minimum_border_norm;
    if (!(squared_norm > minimum_squared) || !std::isfinite(squared_norm))
        throw std::runtime_error(
            "edge-jump effective border is singular");
    return squared_norm;
}

} // namespace kfbim::app3d
