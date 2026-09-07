#include "src/support/trace/mean_free_trace_mass_coordinates_3d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kfbim::app3d {
namespace {

void validate_options(
    const MeanFreeTraceMassCoordinateOptions3D& options)
{
    if (!(options.relative_observability_tolerance > 0.0)
        || !(options.relative_observability_tolerance < 1.0)
        || !std::isfinite(options.relative_observability_tolerance)) {
        throw std::invalid_argument(
            "mean-free trace-mass observability tolerance must be finite "
            "and in (0,1)");
    }
    if (!(options.invariant_tolerance > 0.0)
        || !std::isfinite(options.invariant_tolerance)) {
        throw std::invalid_argument(
            "mean-free trace-mass invariant tolerance must be finite and "
            "positive");
    }
}

double infinity_norm(const Eigen::VectorXd& values)
{
    return values.size() == 0
        ? 0.0
        : values.lpNorm<Eigen::Infinity>();
}

} // namespace

MeanFreeTraceMassCoordinates3D::MeanFreeTraceMassCoordinates3D(
    const Eigen::Ref<const Eigen::VectorXd>& mean_dual,
    MeanFreeTraceMassCoordinateOptions3D options)
    : mean_dual_(mean_dual)
{
    validate_options(options);
    if (mean_dual_.size() < 2) {
        throw std::invalid_argument(
            "mean-free trace-mass reduction requires at least two "
            "coordinates");
    }
    if (!mean_dual_.allFinite()) {
        throw std::invalid_argument(
            "mean-free trace-mass dual must be finite");
    }
    const double moment_norm = mean_dual_.norm();
    if (!(moment_norm > 0.0) || !std::isfinite(moment_norm)) {
        throw std::invalid_argument(
            "mean-free trace-mass dual must be nonzero");
    }

    const double pivot_abs = mean_dual_.cwiseAbs().maxCoeff(&pivot_);
    const double relative_observability = pivot_abs / moment_norm;
    if (!(relative_observability
          > options.relative_observability_tolerance)) {
        throw std::runtime_error(
            "mean-free trace-mass direction has no observable pivot");
    }

    multipliers_.resize(mean_dual_.size() - 1);
    Eigen::Index reduced_index = 0;
    for (Eigen::Index source = 0; source < mean_dual_.size(); ++source) {
        if (source == pivot_)
            continue;
        multipliers_[reduced_index++] =
            mean_dual_[source] / mean_dual_[pivot_];
    }
    const double multiplier_squared_norm = multipliers_.squaredNorm();
    const double root = std::sqrt(1.0 + multiplier_squared_norm);
    // Stable evaluation of (1/sqrt(1+s)-1)/s, including the s=0 limit.
    inverse_square_root_rank_one_coefficient_ =
        -1.0 / (root * (1.0 + root));

    // d^T H before the symmetric inverse-square-root update.
    Eigen::VectorXd mean_residual(multipliers_.size());
    reduced_index = 0;
    for (Eigen::Index source = 0; source < mean_dual_.size(); ++source) {
        if (source == pivot_)
            continue;
        mean_residual[reduced_index] =
            mean_dual_[source]
            - mean_dual_[pivot_] * multipliers_[reduced_index];
        ++reduced_index;
    }
    mean_residual = apply_inverse_square_root(mean_residual);

    // Q^T Q-I is analytically a scalar multiple of r r^T.  Evaluating that
    // scalar is an O(K) certification and avoids materializing Q.
    const double beta = inverse_square_root_rank_one_coefficient_;
    const double one_plus_beta_s =
        1.0 + beta * multiplier_squared_norm;
    const double gram_rank_one_coefficient =
        2.0 * beta + beta * beta * multiplier_squared_norm
        + one_plus_beta_s * one_plus_beta_s;
    const double maximum_multiplier = multipliers_.size() == 0
        ? 0.0
        : multipliers_.cwiseAbs().maxCoeff();
    const double orthonormality_residual =
        std::abs(gram_rank_one_coefficient)
        * maximum_multiplier * maximum_multiplier;

    diagnostics_.source_coordinates = mean_dual_.size();
    diagnostics_.reduced_coordinates = mean_dual_.size() - 1;
    diagnostics_.pivot_coordinate = pivot_;
    diagnostics_.pivot_moment = mean_dual_[pivot_];
    diagnostics_.moment_norm = moment_norm;
    diagnostics_.relative_observability = relative_observability;
    diagnostics_.maximum_elimination_multiplier = maximum_multiplier;
    diagnostics_.coordinate_map_condition = multipliers_.size() <= 1
        ? 1.0
        : root;
    diagnostics_.mean_orthogonality_residual =
        infinity_norm(mean_residual);
    diagnostics_.orthonormality_residual = orthonormality_residual;

    const double mean_scale = std::max(
        std::numeric_limits<double>::min(), moment_norm);
    if (diagnostics_.mean_orthogonality_residual
            > options.invariant_tolerance * mean_scale
        || diagnostics_.orthonormality_residual
               > options.invariant_tolerance) {
        throw std::runtime_error(
            "mean-free trace-mass basis failed its invariants");
    }
}

Eigen::Index
MeanFreeTraceMassCoordinates3D::source_coordinate_count() const noexcept
{
    return mean_dual_.size();
}

Eigen::Index
MeanFreeTraceMassCoordinates3D::reduced_coordinate_count() const noexcept
{
    return multipliers_.size();
}

Eigen::VectorXd MeanFreeTraceMassCoordinates3D::apply_inverse_square_root(
    const Eigen::Ref<const Eigen::VectorXd>& values) const
{
    if (values.size() != reduced_coordinate_count()) {
        throw std::invalid_argument(
            "mean-free inverse-square-root vector has the wrong size");
    }
    Eigen::VectorXd result = values;
    result.noalias() += inverse_square_root_rank_one_coefficient_
        * multipliers_ * multipliers_.dot(values);
    return result;
}

Eigen::VectorXd MeanFreeTraceMassCoordinates3D::lift(
    const Eigen::Ref<const Eigen::VectorXd>& reduced) const
{
    if (reduced.size() != reduced_coordinate_count()) {
        throw std::invalid_argument(
            "mean-free trace-mass lift has the wrong size");
    }
    const Eigen::VectorXd transformed =
        apply_inverse_square_root(reduced);
    Eigen::VectorXd full(source_coordinate_count());
    Eigen::Index reduced_index = 0;
    for (Eigen::Index source = 0; source < full.size(); ++source) {
        if (source == pivot_) {
            full[source] = -multipliers_.dot(transformed);
        } else {
            full[source] = transformed[reduced_index++];
        }
    }
    if (!full.allFinite())
        throw std::runtime_error("mean-free trace-mass lift is not finite");
    return full;
}

Eigen::VectorXd MeanFreeTraceMassCoordinates3D::restrict(
    const Eigen::Ref<const Eigen::VectorXd>& full) const
{
    if (full.size() != source_coordinate_count()) {
        throw std::invalid_argument(
            "mean-free trace-mass restriction has the wrong size");
    }
    Eigen::VectorXd reduced(reduced_coordinate_count());
    Eigen::Index reduced_index = 0;
    for (Eigen::Index source = 0; source < full.size(); ++source) {
        if (source == pivot_)
            continue;
        reduced[reduced_index] =
            full[source] - multipliers_[reduced_index] * full[pivot_];
        ++reduced_index;
    }
    reduced = apply_inverse_square_root(reduced);
    if (!reduced.allFinite()) {
        throw std::runtime_error(
            "mean-free trace-mass restriction is not finite");
    }
    return reduced;
}

Eigen::VectorXd MeanFreeTraceMassCoordinates3D::project(
    const Eigen::Ref<const Eigen::VectorXd>& full) const
{
    return lift(restrict(full));
}

const Eigen::VectorXd&
MeanFreeTraceMassCoordinates3D::mean_dual() const noexcept
{
    return mean_dual_;
}

const MeanFreeTraceMassCoordinateDiagnostics3D&
MeanFreeTraceMassCoordinates3D::diagnostics() const noexcept
{
    return diagnostics_;
}

} // namespace kfbim::app3d
