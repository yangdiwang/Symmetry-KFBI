#include "src/support/topology/topology_mean_free_reduction_3d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace kfbim::app3d {
namespace {

bool sparse_is_finite(const SparseMatrixCSR3D& matrix)
{
    for (Eigen::Index row = 0; row < matrix.outerSize(); ++row) {
        for (SparseMatrixCSR3D::InnerIterator entry(matrix, row); entry;
             ++entry) {
            if (!std::isfinite(entry.value()))
                return false;
        }
    }
    return true;
}

double vector_linf(const Eigen::VectorXd& values)
{
    return values.size() == 0
        ? 0.0
        : values.lpNorm<Eigen::Infinity>();
}

std::vector<double> sparse_column_norms(const SparseMatrixCSR3D& matrix)
{
    std::vector<double> squared(
        static_cast<std::size_t>(matrix.cols()), 0.0);
    for (Eigen::Index row = 0; row < matrix.outerSize(); ++row) {
        for (SparseMatrixCSR3D::InnerIterator entry(matrix, row); entry;
             ++entry) {
            squared[static_cast<std::size_t>(entry.col())] +=
                entry.value() * entry.value();
        }
    }
    for (double& value : squared)
        value = std::sqrt(value);
    return squared;
}

double maximum(const std::vector<double>& values)
{
    return values.empty()
        ? 0.0
        : *std::max_element(values.begin(), values.end());
}

void validate_options(const TopologyMeanFreeReductionOptions3D& options)
{
    if (!(options.relative_observability_tolerance > 0.0)
        || !(options.relative_observability_tolerance < 1.0)
        || !std::isfinite(options.relative_observability_tolerance)) {
        throw std::invalid_argument(
            "topology mean relative observability tolerance must be "
            "finite and in (0,1)");
    }
    if (!(options.invariant_tolerance > 0.0)
        || !std::isfinite(options.invariant_tolerance)) {
        throw std::invalid_argument(
            "topology mean invariant tolerance must be finite and positive");
    }
}

double positive_scale(double value)
{
    return std::max(value, std::numeric_limits<double>::min());
}

} // namespace

Eigen::Index TopologyMeanFreeReduction3D::coefficient_count() const noexcept
{
    return particular_.size();
}

Eigen::Index
TopologyMeanFreeReduction3D::source_coordinate_count() const noexcept
{
    return coordinate_particular_.size();
}

Eigen::Index
TopologyMeanFreeReduction3D::reduced_coordinate_count() const noexcept
{
    return homogeneous_.cols();
}

const Eigen::VectorXd&
TopologyMeanFreeReduction3D::particular() const noexcept
{
    return particular_;
}

const SparseMatrixCSR3D&
TopologyMeanFreeReduction3D::homogeneous() const noexcept
{
    return homogeneous_;
}

const Eigen::VectorXd&
TopologyMeanFreeReduction3D::coordinate_particular() const noexcept
{
    return coordinate_particular_;
}

const SparseMatrixCSR3D&
TopologyMeanFreeReduction3D::coordinate_homogeneous() const noexcept
{
    return coordinate_homogeneous_;
}

Eigen::VectorXd TopologyMeanFreeReduction3D::lift_homogeneous(
    const Eigen::Ref<const Eigen::VectorXd>& reduced) const
{
    if (reduced.size() != reduced_coordinate_count()) {
        throw std::invalid_argument(
            "topology mean-free lift has the wrong coordinate count");
    }
    Eigen::VectorXd lifted = homogeneous_ * reduced;
    if (!lifted.allFinite()) {
        throw std::runtime_error(
            "topology mean-free homogeneous lift is not finite");
    }
    return lifted;
}

Eigen::VectorXd TopologyMeanFreeReduction3D::lift(
    const Eigen::Ref<const Eigen::VectorXd>& reduced) const
{
    Eigen::VectorXd lifted = lift_homogeneous(reduced);
    lifted += particular_;
    return lifted;
}

const TopologyMeanFreeReductionDiagnostics3D&
TopologyMeanFreeReduction3D::diagnostics() const noexcept
{
    return diagnostics_;
}

TopologyMeanFreeReduction3D eliminate_topology_mean_3d(
    const Eigen::Ref<const Eigen::VectorXd>& particular,
    const SparseMatrixCSR3D& homogeneous,
    const Eigen::Ref<const Eigen::VectorXd>& mean_dual,
    double target_mean,
    TopologyMeanFreeReductionOptions3D options)
{
    validate_options(options);
    if (particular.size() <= 0) {
        throw std::invalid_argument(
            "topology mean elimination requires positive coefficient count");
    }
    if (homogeneous.rows() != particular.size()
        || mean_dual.size() != particular.size()) {
        throw std::invalid_argument(
            "topology mean elimination coefficient dimensions disagree");
    }
    if (homogeneous.cols() <= 0) {
        throw std::invalid_argument(
            "topology mean elimination has no coordinate to eliminate");
    }
    if (!particular.allFinite() || !mean_dual.allFinite()
        || !sparse_is_finite(homogeneous)
        || !std::isfinite(target_mean)) {
        throw std::invalid_argument(
            "topology mean elimination inputs must be finite");
    }
    if (!(mean_dual.norm() > 0.0)) {
        throw std::invalid_argument(
            "topology mean elimination requires a nonzero mean covector");
    }

    const Eigen::Index source_coordinates = homogeneous.cols();
    const Eigen::VectorXd moment = homogeneous.transpose() * mean_dual;
    Eigen::Index pivot = 0;
    const double pivot_abs = moment.cwiseAbs().maxCoeff(&pivot);
    const std::vector<double> input_column_norms =
        sparse_column_norms(homogeneous);
    const double observability_scale =
        mean_dual.norm() * maximum(input_column_norms);
    if (!(observability_scale > 0.0)) {
        throw std::invalid_argument(
            "topology mean elimination homogeneous map is zero");
    }
    const double relative_observability = pivot_abs / observability_scale;
    if (!(pivot_abs
          > options.relative_observability_tolerance
                * observability_scale)) {
        throw std::runtime_error(
            "topology mean constraint is not observable in the homogeneous "
            "coordinate space");
    }

    const double mean_before = mean_dual.dot(particular);
    const double correction =
        (target_mean - mean_before) / moment[pivot];
    if (!std::isfinite(correction)) {
        throw std::runtime_error(
            "topology mean particular correction is not finite");
    }

    TopologyMeanFreeReduction3D result;
    result.coordinate_particular_ =
        Eigen::VectorXd::Zero(source_coordinates);
    result.coordinate_particular_[pivot] = correction;
    result.particular_.noalias() =
        particular + homogeneous * result.coordinate_particular_;

    const Eigen::Index reduced_coordinates = source_coordinates - 1;
    std::vector<Eigen::Triplet<double>> coordinate_entries;
    coordinate_entries.reserve(static_cast<std::size_t>(
        std::max<Eigen::Index>(0, 2 * reduced_coordinates)));
    double maximum_multiplier = 0.0;
    double multiplier_squared_norm = 0.0;
    Eigen::Index reduced_column = 0;
    for (Eigen::Index source_column = 0;
         source_column < source_coordinates; ++source_column) {
        if (source_column == pivot)
            continue;
        coordinate_entries.emplace_back(
            source_column, reduced_column, 1.0);
        const double multiplier = moment[source_column] / moment[pivot];
        if (multiplier != 0.0) {
            coordinate_entries.emplace_back(
                pivot, reduced_column, -multiplier);
        }
        maximum_multiplier =
            std::max(maximum_multiplier, std::abs(multiplier));
        multiplier_squared_norm += multiplier * multiplier;
        ++reduced_column;
    }
    result.coordinate_homogeneous_.resize(
        source_coordinates, reduced_coordinates);
    result.coordinate_homogeneous_.setFromTriplets(
        coordinate_entries.begin(), coordinate_entries.end());
    result.coordinate_homogeneous_.makeCompressed();

    result.homogeneous_ =
        homogeneous * result.coordinate_homogeneous_;
    result.homogeneous_.prune(0.0);
    result.homogeneous_.makeCompressed();
    if (!result.particular_.allFinite()
        || !sparse_is_finite(result.homogeneous_)) {
        throw std::runtime_error(
            "topology mean-free affine result contains a non-finite entry");
    }

    const double mean_after = mean_dual.dot(result.particular_);
    const Eigen::VectorXd coordinate_mean_residual =
        result.coordinate_homogeneous_.transpose() * moment;
    const Eigen::VectorXd homogeneous_mean_residual =
        result.homogeneous_.transpose() * mean_dual;

    result.diagnostics_.source_coordinates = source_coordinates;
    result.diagnostics_.reduced_coordinates = reduced_coordinates;
    result.diagnostics_.coordinate_rank = reduced_coordinates;
    result.diagnostics_.pivot_coordinate = pivot;
    result.diagnostics_.pivot_moment = moment[pivot];
    result.diagnostics_.moment_norm = moment.norm();
    result.diagnostics_.relative_observability = relative_observability;
    result.diagnostics_.maximum_elimination_multiplier = maximum_multiplier;
    result.diagnostics_.coordinate_map_condition =
        reduced_coordinates <= 1
        ? 1.0
        : std::sqrt(1.0 + multiplier_squared_norm);
    result.diagnostics_.particular_mean_before = mean_before;
    result.diagnostics_.target_mean = target_mean;
    result.diagnostics_.particular_mean_after = mean_after;
    result.diagnostics_.particular_mean_residual =
        std::abs(mean_after - target_mean);
    result.diagnostics_.coordinate_mean_orthogonality_residual =
        vector_linf(coordinate_mean_residual);
    result.diagnostics_.homogeneous_mean_residual =
        vector_linf(homogeneous_mean_residual);

    const double particular_scale = positive_scale(
        std::max({std::abs(target_mean), std::abs(mean_before),
                  std::abs(correction * moment[pivot]),
                  mean_dual.norm() * result.particular_.norm()}));
    const std::vector<double> output_column_norms =
        sparse_column_norms(result.homogeneous_);
    const double homogeneous_scale = positive_scale(
        mean_dual.norm() * maximum(output_column_norms));
    const double coordinate_scale = positive_scale(
        moment.norm() * std::sqrt(1.0 + maximum_multiplier
                                             * maximum_multiplier));
    if (result.diagnostics_.particular_mean_residual
            > options.invariant_tolerance * particular_scale
        || result.diagnostics_.coordinate_mean_orthogonality_residual
            > options.invariant_tolerance * coordinate_scale
        || result.diagnostics_.homogeneous_mean_residual
            > options.invariant_tolerance * homogeneous_scale) {
        throw std::runtime_error(
            "topology mean-free construction failed its mean invariants");
    }
    return result;
}

} // namespace kfbim::app3d
