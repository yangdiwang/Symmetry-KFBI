#include "src/support/topology/topology_trace_projector_3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace kfbim::app3d {
namespace {

bool sparse_is_finite(const SparseMatrixCSR3D& matrix)
{
    for (int outer = 0; outer < matrix.outerSize(); ++outer) {
        for (SparseMatrixCSR3D::InnerIterator entry(matrix, outer);
             entry; ++entry) {
            if (!std::isfinite(entry.value()))
                return false;
        }
    }
    return true;
}

double deterministic_sign(Eigen::Index index, int probe)
{
    std::uint64_t value =
        (static_cast<std::uint64_t>(index) + 1ULL)
            * 0x9e3779b97f4a7c15ULL
        + (static_cast<std::uint64_t>(probe) + 1ULL)
              * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return (value & 1ULL) == 0ULL ? -1.0 : 1.0;
}

constexpr int exact_roundtrip_limit = 256;
constexpr int large_roundtrip_probe_count = 8;

} // namespace

TopologyTraceProjector3D::TopologyTraceProjector3D(
    const SparseMatrixCSR3D& c0_test_design,
    const SparseMatrixCSR3D& base_expansion,
    const AffineReduction3D& reduction,
    const Eigen::VectorXd& weights,
    double relative_rank_tolerance,
    double maximum_condition)
    : maximum_condition_(maximum_condition)
{
    if (c0_test_design.rows() <= 0 || c0_test_design.cols() <= 0) {
        throw std::invalid_argument(
            "topology trace test design must have positive dimensions");
    }
    if (base_expansion.rows() != c0_test_design.cols()
        || base_expansion.cols() != reduction.base_size()) {
        throw std::invalid_argument(
            "topology trace dimensions do not satisfy "
            "Bz = B_test A0 E");
    }
    if (reduction.reduced_size() <= 0) {
        throw std::invalid_argument(
            "topology trace reduction has no homogeneous coordinates");
    }
    if (weights.size() != c0_test_design.rows()) {
        throw std::invalid_argument(
            "topology trace weight count does not match test samples");
    }
    if (!sparse_is_finite(c0_test_design)
        || !sparse_is_finite(base_expansion)
        || !weights.allFinite()
        || !(weights.array() > 0.0).all()) {
        throw std::invalid_argument(
            "topology trace inputs must be finite and weights positive");
    }
    if (!(maximum_condition_ >= 1.0)
        || !std::isfinite(maximum_condition_)) {
        throw std::invalid_argument(
            "topology trace maximum condition must be finite and at least one");
    }

    particular_c0_.noalias() =
        base_expansion * reduction.particular_base();
    homogeneous_c0_ =
        base_expansion * reduction.homogeneous_base();
    initialize_projector(
        c0_test_design, weights, relative_rank_tolerance);
}

TopologyTraceProjector3D::TopologyTraceProjector3D(
    const SparseMatrixCSR3D& c0_test_design,
    Eigen::VectorXd particular_c0,
    SparseMatrixCSR3D homogeneous_c0,
    const Eigen::VectorXd& weights,
    double relative_rank_tolerance,
    double maximum_condition)
    : particular_c0_(std::move(particular_c0))
    , homogeneous_c0_(std::move(homogeneous_c0))
    , maximum_condition_(maximum_condition)
{
    if (c0_test_design.rows() <= 0 || c0_test_design.cols() <= 0) {
        throw std::invalid_argument(
            "topology trace test design must have positive dimensions");
    }
    if (particular_c0_.size() != c0_test_design.cols()
        || homogeneous_c0_.rows() != c0_test_design.cols()) {
        throw std::invalid_argument(
            "explicit topology trace affine coefficient dimensions disagree");
    }
    if (homogeneous_c0_.cols() <= 0) {
        throw std::invalid_argument(
            "explicit topology trace affine space has no homogeneous "
            "coordinates");
    }
    if (weights.size() != c0_test_design.rows()) {
        throw std::invalid_argument(
            "topology trace weight count does not match test samples");
    }
    if (!sparse_is_finite(c0_test_design)
        || !weights.allFinite()
        || !(weights.array() > 0.0).all()) {
        throw std::invalid_argument(
            "topology trace inputs must be finite and weights positive");
    }
    if (!(maximum_condition_ >= 1.0)
        || !std::isfinite(maximum_condition_)) {
        throw std::invalid_argument(
            "topology trace maximum condition must be finite and at least one");
    }
    initialize_projector(
        c0_test_design, weights, relative_rank_tolerance);
}

void TopologyTraceProjector3D::initialize_projector(
    const SparseMatrixCSR3D& c0_test_design,
    const Eigen::VectorXd& weights,
    double relative_rank_tolerance)
{
    homogeneous_c0_.prune(0.0);
    homogeneous_c0_.makeCompressed();
    if (!particular_c0_.allFinite()
        || !sparse_is_finite(homogeneous_c0_)) {
        throw std::runtime_error(
            "topology trace affine lift contains a non-finite entry");
    }

    // At this point every weight has already been certified finite and
    // strictly positive by the public constructors.  Consequently the
    // effective observation count is weights.size().  Check it against the
    // *final* affine homogeneous dimension (after topology and mean
    // elimination), before ReducedTraceProjection3D forms any factorization.
    if (weights.size() <= homogeneous_c0_.cols()) {
        throw std::invalid_argument(
            "topology trace observability requires strictly more "
            "positive-weight samples than final reduced coordinates "
            "(samples="
            + std::to_string(static_cast<long long>(weights.size()))
            + ", coordinates="
            + std::to_string(
                static_cast<long long>(homogeneous_c0_.cols()))
            + ")");
    }

    // ReducedTraceProjection3D owns both required specification backends:
    // weighted QR below the sparse threshold and sparse SPD above it.
    const Eigen::SparseMatrix<double> test_design_csc(c0_test_design);
    const Eigen::SparseMatrix<double> homogeneous_c0_csc(homogeneous_c0_);
    native_projector_ = std::make_unique<ReducedTraceProjection3D>(
        test_design_csc,
        homogeneous_c0_csc,
        weights,
        relative_rank_tolerance,
        true);

    if (native_projector_->rank()
            != static_cast<int>(homogeneous_c0_.cols())) {
        throw std::runtime_error(
            "topology trace projector does not have full reduced rank");
    }
    if (native_projector_->condition() > maximum_condition_) {
        throw std::runtime_error(
            "topology trace projector exceeds the configured condition limit");
    }

    diagnostics_.samples = native_projector_->sample_count();
    diagnostics_.reduced_coordinates =
        native_projector_->coefficient_count();
    diagnostics_.rank = native_projector_->rank();
    diagnostics_.condition = native_projector_->condition();
    diagnostics_.pb_identity_residual =
        native_projector_->pb_identity_residual();
    diagnostics_.pb_max_error = native_projector_->pb_max_error();
    diagnostics_.backend = native_projector_->backend();
    diagnostics_.gram_nonzeros = native_projector_->gram_nonzeros();
    diagnostics_.factor_nonzeros = native_projector_->factor_nonzeros();
    certify_full_roundtrip(c0_test_design);
}

Eigen::VectorXd TopologyTraceProjector3D::project(
    const Eigen::Ref<const Eigen::VectorXd>& full_trace) const
{
    const Eigen::VectorXd native = native_projector_->apply(full_trace);
    Eigen::VectorXd trace_mass =
        native_projector_->native_to_trace_mass(native);
    if (!trace_mass.allFinite()) {
        throw std::runtime_error(
            "topology trace projection produced non-finite mass coordinates");
    }
    return trace_mass;
}

Eigen::VectorXd TopologyTraceProjector3D::native_to_trace_mass(
    const Eigen::Ref<const Eigen::VectorXd>& native_reduced) const
{
    return native_projector_->native_to_trace_mass(native_reduced);
}

Eigen::VectorXd TopologyTraceProjector3D::trace_mass_to_native(
    const Eigen::Ref<const Eigen::VectorXd>& trace_mass) const
{
    return native_projector_->trace_mass_to_native(trace_mass);
}

Eigen::VectorXd TopologyTraceProjector3D::lift_homogeneous_c0(
    const Eigen::Ref<const Eigen::VectorXd>& trace_mass) const
{
    const Eigen::VectorXd native = trace_mass_to_native(trace_mass);
    Eigen::VectorXd lifted = homogeneous_c0_ * native;
    if (!lifted.allFinite()) {
        throw std::runtime_error(
            "topology homogeneous lift produced non-finite coefficients");
    }
    return lifted;
}

Eigen::VectorXd TopologyTraceProjector3D::lift_full_c0(
    const Eigen::Ref<const Eigen::VectorXd>& trace_mass) const
{
    Eigen::VectorXd lifted = lift_homogeneous_c0(trace_mass);
    lifted += particular_c0_;
    return lifted;
}

Eigen::VectorXd TopologyTraceProjector3D::mean_dual_to_trace_mass(
    const Eigen::Ref<const Eigen::VectorXd>& c0_mean_dual) const
{
    require_c0_size(c0_mean_dual.size());
    if (!c0_mean_dual.allFinite()) {
        throw std::invalid_argument(
            "topology trace mean dual contains a non-finite entry");
    }
    const Eigen::VectorXd native_dual =
        homogeneous_c0_.transpose() * c0_mean_dual;
    return native_projector_->native_dual_to_trace_mass(native_dual);
}

double TopologyTraceProjector3D::particular_mean(
    const Eigen::Ref<const Eigen::VectorXd>& c0_mean_dual) const
{
    require_c0_size(c0_mean_dual.size());
    if (!c0_mean_dual.allFinite()) {
        throw std::invalid_argument(
            "topology trace mean dual contains a non-finite entry");
    }
    const double value = c0_mean_dual.dot(particular_c0_);
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            "topology trace particular mean is not finite");
    }
    return value;
}

const Eigen::VectorXd&
TopologyTraceProjector3D::particular_c0() const noexcept
{
    return particular_c0_;
}

const SparseMatrixCSR3D&
TopologyTraceProjector3D::homogeneous_c0() const noexcept
{
    return homogeneous_c0_;
}

int TopologyTraceProjector3D::sample_count() const noexcept
{
    return diagnostics_.samples;
}

int TopologyTraceProjector3D::c0_coefficient_count() const noexcept
{
    return static_cast<int>(particular_c0_.size());
}

int TopologyTraceProjector3D::reduced_coordinate_count() const noexcept
{
    return diagnostics_.reduced_coordinates;
}

int TopologyTraceProjector3D::rank() const noexcept
{
    return diagnostics_.rank;
}

double TopologyTraceProjector3D::condition() const noexcept
{
    return diagnostics_.condition;
}

double TopologyTraceProjector3D::maximum_condition() const noexcept
{
    return maximum_condition_;
}

double TopologyTraceProjector3D::pb_identity_residual() const noexcept
{
    return diagnostics_.pb_identity_residual;
}

double TopologyTraceProjector3D::pb_max_error() const noexcept
{
    return diagnostics_.pb_max_error;
}

double TopologyTraceProjector3D::full_roundtrip_residual() const noexcept
{
    return diagnostics_.full_roundtrip_residual;
}

double TopologyTraceProjector3D::full_roundtrip_max_error() const noexcept
{
    return diagnostics_.full_roundtrip_max_error;
}

ReducedTraceProjectionBackend3D
TopologyTraceProjector3D::backend() const noexcept
{
    return diagnostics_.backend;
}

bool TopologyTraceProjector3D::uses_sparse_cholesky() const noexcept
{
    return diagnostics_.backend
        == ReducedTraceProjectionBackend3D::SparseSimplicialLlt;
}

std::int64_t TopologyTraceProjector3D::gram_nonzeros() const noexcept
{
    return diagnostics_.gram_nonzeros;
}

std::int64_t TopologyTraceProjector3D::factor_nonzeros() const noexcept
{
    return diagnostics_.factor_nonzeros;
}

const TopologyTraceProjectorDiagnostics3D&
TopologyTraceProjector3D::diagnostics() const noexcept
{
    return diagnostics_;
}

const Eigen::MatrixXd&
TopologyTraceProjector3D::native_projection_matrix() const
{
    return native_projector_->projection_matrix();
}

void TopologyTraceProjector3D::certify_full_roundtrip(
    const SparseMatrixCSR3D& c0_test_design)
{
    const int reduced = reduced_coordinate_count();
    const int probes = reduced <= exact_roundtrip_limit
        ? reduced
        : large_roundtrip_probe_count;
    diagnostics_.full_roundtrip_probes = probes;
    double squared_error_sum = 0.0;
    double max_error = 0.0;
    Eigen::VectorXd target(reduced);
    for (int probe = 0; probe < probes; ++probe) {
        if (reduced <= exact_roundtrip_limit) {
            target.setZero();
            target[probe] = 1.0;
        } else {
            for (int row = 0; row < reduced; ++row)
                target[row] = deterministic_sign(row, probe);
        }
        const Eigen::VectorXd coefficients =
            lift_homogeneous_c0(target);
        const Eigen::VectorXd trace = c0_test_design * coefficients;
        Eigen::VectorXd error = project(trace);
        error -= target;
        squared_error_sum += error.squaredNorm();
        max_error = std::max(max_error, error.cwiseAbs().maxCoeff());
    }
    diagnostics_.full_roundtrip_residual = std::sqrt(
        squared_error_sum
        / (static_cast<double>(probes)
           * (reduced <= exact_roundtrip_limit
                  ? 1.0
                  : static_cast<double>(reduced))));
    diagnostics_.full_roundtrip_max_error = max_error;
    if (!std::isfinite(diagnostics_.full_roundtrip_residual)
        || !std::isfinite(diagnostics_.full_roundtrip_max_error)) {
        throw std::runtime_error(
            "topology trace full roundtrip certification is not finite");
    }
}

void TopologyTraceProjector3D::require_c0_size(Eigen::Index size) const
{
    if (size != particular_c0_.size()) {
        throw std::invalid_argument(
            "topology trace coefficient-space vector has the wrong size");
    }
}

} // namespace kfbim::app3d
