#include "reduced_trace_projection_3d.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d {
namespace {

void validate_common_inputs(
    Eigen::Index rows,
    Eigen::Index columns,
    bool design_is_finite,
    const Eigen::VectorXd& weights,
    bool require_left_inverse)
{
    if (rows <= 0 || columns <= 0) {
        throw std::invalid_argument(
            "reduced trace design must have positive dimensions");
    }
    if (rows > static_cast<Eigen::Index>(std::numeric_limits<int>::max())
        || columns
            > static_cast<Eigen::Index>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "reduced trace design dimensions exceed the supported range");
    }
    if (weights.size() != rows) {
        throw std::invalid_argument(
            "reduced trace weight count does not match the design rows");
    }
    if (!design_is_finite) {
        throw std::invalid_argument(
            "reduced trace design contains a non-finite entry");
    }
    if (!weights.allFinite()
        || !(weights.array() > 0.0).all()) {
        throw std::invalid_argument(
            "reduced trace weights must be finite and strictly positive");
    }
    if (require_left_inverse && rows < columns) {
        throw std::invalid_argument(
            "reduced trace design has fewer samples than coefficients");
    }
}

bool sparse_is_finite(const Eigen::SparseMatrix<double>& matrix)
{
    for (int outer = 0; outer < matrix.outerSize(); ++outer) {
        for (Eigen::SparseMatrix<double>::InnerIterator entry(matrix, outer);
             entry; ++entry) {
            if (!std::isfinite(entry.value()))
                return false;
        }
    }
    return true;
}

void validate_rank_tolerance(double relative_rank_tolerance)
{
    if (!(relative_rank_tolerance > 0.0)
        || !(relative_rank_tolerance < 1.0)
        || !std::isfinite(relative_rank_tolerance)) {
        throw std::invalid_argument(
            "reduced trace rank tolerance must be finite and in (0,1)");
    }
}

template <class RightHandSide>
void solve_cholesky_in_place(
    const Eigen::LLT<Eigen::MatrixXd, Eigen::Lower>& factor,
    RightHandSide& right_hand_side)
{
    factor.matrixL().solveInPlace(right_hand_side);
    factor.matrixU().solveInPlace(right_hand_side);
}

constexpr int exact_gram_diagnostic_limit = 1000;
constexpr int large_gram_probe_count = 8;

using SparseGramFactor = Eigen::SimplicialLLT<
    Eigen::SparseMatrix<double>,
    Eigen::Lower,
    Eigen::AMDOrdering<int>>;

void symmetrize_from_lower(Eigen::MatrixXd& matrix)
{
    for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
        for (Eigen::Index row = column + 1; row < matrix.rows(); ++row)
            matrix(column, row) = matrix(row, column);
    }
}

double effective_relative_gram_tolerance(
    int sample_count,
    double relative_rank_tolerance)
{
    const double requested =
        relative_rank_tolerance * relative_rank_tolerance;
    const double accumulation =
        8.0 * std::numeric_limits<double>::epsilon()
        * static_cast<double>(std::max(1, sample_count));
    return std::max(requested, accumulation);
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

void certify_large_gram(
    const Eigen::MatrixXd& gram,
    const Eigen::LLT<Eigen::MatrixXd, Eigen::Lower>& gram_factor,
    double& pb_identity_residual,
    double& pb_max_error)
{
    const Eigen::Index size = gram.rows();
    double squared_frobenius_sum = 0.0;
    pb_max_error = 0.0;
    Eigen::VectorXd probe(size);
    Eigen::VectorXd image(size);
    for (int p = 0; p < large_gram_probe_count; ++p) {
        for (Eigen::Index row = 0; row < size; ++row)
            probe[row] = deterministic_sign(row, p);
        image.noalias() = gram * probe;
        solve_cholesky_in_place(gram_factor, image);
        image -= probe;
        squared_frobenius_sum += image.squaredNorm();
    }
    pb_identity_residual = std::sqrt(
        squared_frobenius_sum
        / (static_cast<double>(large_gram_probe_count)
           * static_cast<double>(size)));

    // Canonical columns give a directly interpretable sampled max norm.  The
    // full K-column solve is intentionally avoided for large surface spaces.
    for (int p = 0; p < large_gram_probe_count; ++p) {
        const Eigen::Index column = large_gram_probe_count == 1
            ? 0
            : static_cast<Eigen::Index>(p) * (size - 1)
                  / (large_gram_probe_count - 1);
        image = gram.col(column);
        solve_cholesky_in_place(gram_factor, image);
        image[column] -= 1.0;
        pb_max_error = std::max(
            pb_max_error, image.cwiseAbs().maxCoeff());
    }
}

Eigen::SparseMatrix<double> symmetric_sparse_gram(
    const Eigen::SparseMatrix<double>& raw_gram)
{
    if (raw_gram.rows() != raw_gram.cols()
        || !sparse_is_finite(raw_gram)) {
        throw std::runtime_error(
            "weighted sparse trace Gram matrix is invalid");
    }
    // Sparse products may accumulate the two triangles in a slightly
    // different order.  Cholesky and all diagnostics use the same canonical
    // lower triangle so that the represented SPD operator is exactly
    // self-adjoint without ever materializing a dense matrix.
    Eigen::SparseMatrix<double> gram =
        raw_gram.template selfadjointView<Eigen::Lower>();
    gram.prune(0.0);
    gram.makeCompressed();
    return gram;
}

double sparse_one_norm(const Eigen::SparseMatrix<double>& matrix)
{
    double result = 0.0;
    for (int column = 0; column < matrix.outerSize(); ++column) {
        double column_sum = 0.0;
        for (Eigen::SparseMatrix<double>::InnerIterator entry(
                 matrix, column);
             entry; ++entry) {
            column_sum += std::abs(entry.value());
        }
        result = std::max(result, column_sum);
    }
    return result;
}

Eigen::VectorXd solve_sparse_gram(
    const SparseGramFactor& factor,
    const Eigen::Ref<const Eigen::VectorXd>& right_hand_side)
{
    Eigen::VectorXd solution = factor.solve(right_hand_side);
    if (factor.info() != Eigen::Success || !solution.allFinite()) {
        throw std::runtime_error(
            "weighted sparse trace Gram solve failed");
    }
    return solution;
}

double estimate_sparse_inverse_one_norm(
    const SparseGramFactor& factor,
    int size)
{
    // Hager's one-norm estimator specialized to the symmetric inverse.  It
    // requires only sparse triangular solves and O(K) work vectors.  The
    // alternating-vector safeguard is the standard protection against a
    // sign-vector cycle.
    Eigen::VectorXd direction = Eigen::VectorXd::Constant(
        size, 1.0 / static_cast<double>(size));
    double estimate = 0.0;
    Eigen::Index previous_index = -1;
    for (int iteration = 0; iteration < 8; ++iteration) {
        const Eigen::VectorXd image =
            solve_sparse_gram(factor, direction);
        const double candidate = image.lpNorm<1>();
        if (iteration > 0 && candidate <= estimate)
            break;
        estimate = std::max(estimate, candidate);

        Eigen::VectorXd signs(image.size());
        for (Eigen::Index row = 0; row < image.size(); ++row)
            signs[row] = image[row] < 0.0 ? -1.0 : 1.0;
        const Eigen::VectorXd dual_image =
            solve_sparse_gram(factor, signs);
        Eigen::Index next_index = 0;
        const double dual_max =
            dual_image.cwiseAbs().maxCoeff(&next_index);
        const double dual_pairing = dual_image.dot(direction);
        if (dual_max <= dual_pairing
                + 8.0 * std::numeric_limits<double>::epsilon()
                      * std::max(1.0, dual_max)
            || next_index == previous_index) {
            break;
        }
        direction.setZero();
        direction[next_index] = 1.0;
        previous_index = next_index;
    }

    Eigen::VectorXd alternating(size);
    if (size == 1) {
        alternating[0] = 1.0;
    } else {
        for (int row = 0; row < size; ++row) {
            const double magnitude = 1.0
                + static_cast<double>(row)
                      / static_cast<double>(size - 1);
            alternating[row] = (row & 1) == 0
                ? magnitude : -magnitude;
        }
    }
    const double alternating_estimate =
        2.0
        * solve_sparse_gram(factor, alternating).lpNorm<1>()
        / (3.0 * static_cast<double>(size));
    estimate = std::max(estimate, alternating_estimate);
    if (!(estimate > 0.0) || !std::isfinite(estimate)) {
        throw std::runtime_error(
            "weighted sparse trace inverse norm estimate failed");
    }
    return estimate;
}

void certify_sparse_gram(
    const Eigen::SparseMatrix<double>& gram,
    const SparseGramFactor& factor,
    double& pb_identity_residual,
    double& pb_max_error)
{
    const Eigen::Index size = gram.rows();
    double squared_error_sum = 0.0;
    pb_max_error = 0.0;
    Eigen::VectorXd probe(size);
    for (int p = 0; p < large_gram_probe_count; ++p) {
        for (Eigen::Index row = 0; row < size; ++row)
            probe[row] = deterministic_sign(row, p);
        Eigen::VectorXd error = solve_sparse_gram(
            factor, Eigen::VectorXd(gram * probe));
        error -= probe;
        squared_error_sum += error.squaredNorm();
    }
    pb_identity_residual = std::sqrt(
        squared_error_sum
        / (static_cast<double>(large_gram_probe_count)
           * static_cast<double>(size)));

    for (int p = 0; p < large_gram_probe_count; ++p) {
        const Eigen::Index column = large_gram_probe_count == 1
            ? 0
            : static_cast<Eigen::Index>(p) * (size - 1)
                  / (large_gram_probe_count - 1);
        Eigen::VectorXd gram_column = Eigen::VectorXd::Zero(size);
        for (Eigen::SparseMatrix<double>::InnerIterator entry(
                 gram, static_cast<int>(column));
             entry; ++entry) {
            gram_column[entry.row()] = entry.value();
        }
        Eigen::VectorXd error = solve_sparse_gram(factor, gram_column);
        error[column] -= 1.0;
        pb_max_error = std::max(
            pb_max_error, error.cwiseAbs().maxCoeff());
    }
}

void finalize_sparse_gram_projection(
    Eigen::SparseMatrix<double> raw_gram,
    int sample_count,
    int coefficient_count,
    double relative_rank_tolerance,
    SparseGramFactor& gram_factor,
    int& rank,
    double& condition,
    double& pb_identity_residual,
    double& pb_max_error,
    std::int64_t& gram_nonzeros,
    std::int64_t& factor_nonzeros)
{
    if (raw_gram.rows() != coefficient_count
        || raw_gram.cols() != coefficient_count) {
        throw std::runtime_error(
            "weighted sparse trace Gram dimensions are invalid");
    }
    Eigen::SparseMatrix<double> gram =
        symmetric_sparse_gram(raw_gram);
    gram_nonzeros = static_cast<std::int64_t>(gram.nonZeros());

    gram_factor.compute(gram);
    if (gram_factor.info() != Eigen::Success) {
        throw std::runtime_error(
            "weighted sparse trace Gram Cholesky factorization failed");
    }
    const Eigen::SparseMatrix<double> lower_factor =
        gram_factor.matrixL();
    factor_nonzeros =
        static_cast<std::int64_t>(lower_factor.nonZeros());
    const Eigen::VectorXd diagonal = lower_factor.diagonal();
    if (!diagonal.allFinite()
        || !(diagonal.array() > 0.0).all()) {
        throw std::runtime_error(
            "weighted sparse trace Gram factor has an invalid diagonal");
    }

    const double gram_norm = sparse_one_norm(gram);
    const double inverse_norm = estimate_sparse_inverse_one_norm(
        gram_factor, coefficient_count);
    const double gram_condition = gram_norm * inverse_norm;
    const double reciprocal_gram_condition = 1.0 / gram_condition;
    const double relative_gram_tolerance =
        effective_relative_gram_tolerance(
            sample_count, relative_rank_tolerance);
    if (!(gram_norm > 0.0) || !std::isfinite(gram_condition)
        || !(reciprocal_gram_condition > relative_gram_tolerance)) {
        throw std::runtime_error(
            "weighted reduced trace design is numerically rank deficient "
            "for the sparse Gram factorization");
    }

    rank = coefficient_count;
    condition = std::sqrt(gram_condition);
    certify_sparse_gram(
        gram, gram_factor, pb_identity_residual, pb_max_error);
    if (!std::isfinite(condition)
        || !std::isfinite(pb_identity_residual)
        || !std::isfinite(pb_max_error)) {
        throw std::runtime_error(
            "sparse reduced trace projection certification is not finite");
    }
}

void finalize_gram_projection(
    Eigen::MatrixXd gram,
    int sample_count,
    int coefficient_count,
    double relative_rank_tolerance,
    Eigen::LLT<Eigen::MatrixXd, Eigen::Lower>& gram_factor,
    int& rank,
    double& condition,
    double& pb_identity_residual,
    double& pb_max_error)
{
    if (gram.rows() != coefficient_count
        || gram.cols() != coefficient_count || !gram.allFinite()) {
        throw std::runtime_error(
            "weighted reduced trace Gram matrix is invalid");
    }
    symmetrize_from_lower(gram);

    gram_factor.compute(gram);
    if (gram_factor.info() != Eigen::Success) {
        throw std::runtime_error(
            "weighted reduced trace Gram Cholesky factorization failed");
    }

    const double relative_gram_tolerance =
        effective_relative_gram_tolerance(
            sample_count, relative_rank_tolerance);
    if (coefficient_count <= exact_gram_diagnostic_limit) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver;
        eigen_solver.compute(gram, Eigen::EigenvaluesOnly);
        if (eigen_solver.info() != Eigen::Success
            || !eigen_solver.eigenvalues().allFinite()) {
            throw std::runtime_error(
                "weighted reduced trace Gram eigensolve failed");
        }
        const Eigen::VectorXd& eigenvalues = eigen_solver.eigenvalues();
        const double largest = eigenvalues[eigenvalues.size() - 1];
        if (!(largest > 0.0) || !std::isfinite(largest)) {
            throw std::runtime_error(
                "weighted reduced trace Gram matrix is not positive definite");
        }
        const double threshold = relative_gram_tolerance * largest;
        rank = static_cast<int>((eigenvalues.array() > threshold).count());
        if (rank != coefficient_count) {
            throw std::runtime_error(
                "weighted reduced trace design is numerically rank deficient "
                "for the Gram factorization");
        }
        condition = std::sqrt(largest / eigenvalues[0]);

        Eigen::MatrixXd pb_error = gram;
        solve_cholesky_in_place(gram_factor, pb_error);
        pb_error.diagonal().array() -= 1.0;
        pb_identity_residual = pb_error.norm()
            / std::sqrt(static_cast<double>(coefficient_count));
        pb_max_error = pb_error.cwiseAbs().maxCoeff();
    } else {
        const double reciprocal_gram_condition = gram_factor.rcond();
        if (!(reciprocal_gram_condition > relative_gram_tolerance)
            || !std::isfinite(reciprocal_gram_condition)) {
            throw std::runtime_error(
                "weighted reduced trace design is numerically rank deficient "
                "for the large Gram factorization");
        }
        rank = coefficient_count;
        condition = std::sqrt(1.0 / reciprocal_gram_condition);
        certify_large_gram(
            gram, gram_factor, pb_identity_residual, pb_max_error);
    }

    if (!std::isfinite(condition)) {
        throw std::runtime_error(
            "weighted reduced trace condition number is not finite");
    }
    if (!std::isfinite(pb_identity_residual)
        || !std::isfinite(pb_max_error)) {
        throw std::runtime_error(
            "reduced trace projection certification is not finite");
    }
}

void finalize_weighted_qr_projection(
    Eigen::MatrixXd reduced_basis_design,
    const Eigen::VectorXd& weights,
    double relative_rank_tolerance,
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd>& factor,
    Eigen::VectorXd& sqrt_weights,
    int& rank,
    double& condition,
    double& pb_identity_residual,
    double& pb_max_error,
    std::int64_t& gram_nonzeros,
    std::int64_t& factor_nonzeros)
{
    if (reduced_basis_design.rows() != weights.size()
        || reduced_basis_design.cols() <= 0
        || !reduced_basis_design.allFinite()) {
        throw std::runtime_error(
            "weighted reduced trace QR design is invalid");
    }
    sqrt_weights = weights.array().sqrt().matrix();
    for (Eigen::Index row = 0; row < reduced_basis_design.rows(); ++row)
        reduced_basis_design.row(row) *= sqrt_weights[row];

    // Set the threshold before compute so every rank-dependent operation and
    // solve uses one numerical-rank convention.
    factor.setThreshold(relative_rank_tolerance);
    factor.compute(reduced_basis_design);
    if (factor.info() != Eigen::Success) {
        throw std::runtime_error(
            "weighted reduced trace QR factorization failed");
    }
    rank = static_cast<int>(factor.rank());
    const int coefficient_count = static_cast<int>(
        reduced_basis_design.cols());
    if (rank != coefficient_count) {
        throw std::runtime_error(
            "weighted reduced trace design is numerically rank deficient "
            "for the QR factorization");
    }

    const Eigen::MatrixXd upper = factor.matrixR()
        .topLeftCorner(coefficient_count, coefficient_count)
        .template triangularView<Eigen::Upper>();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        upper, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd singular_values = svd.singularValues();
    if (singular_values.size() != coefficient_count
        || !singular_values.allFinite()
        || !(singular_values[0] > 0.0)) {
        throw std::runtime_error(
            "weighted reduced trace QR condition estimate failed");
    }
    const double threshold =
        relative_rank_tolerance * singular_values[0];
    const int svd_rank = static_cast<int>(
        (singular_values.array() > threshold).count());
    if (svd_rank != coefficient_count
        || !(singular_values[coefficient_count - 1] > 0.0)) {
        throw std::runtime_error(
            "weighted reduced trace design is numerically rank deficient "
            "for the QR singular-value audit");
    }
    condition = singular_values[0]
        / singular_values[coefficient_count - 1];

    Eigen::MatrixXd pb_error = factor.solve(reduced_basis_design);
    if (factor.info() != Eigen::Success || !pb_error.allFinite()) {
        throw std::runtime_error(
            "weighted reduced trace QR certification solve failed");
    }
    pb_error.diagonal().array() -= 1.0;
    pb_identity_residual = pb_error.norm()
        / std::sqrt(static_cast<double>(coefficient_count));
    pb_max_error = pb_error.cwiseAbs().maxCoeff();
    gram_nonzeros = 0;
    factor_nonzeros = static_cast<std::int64_t>(coefficient_count)
        * static_cast<std::int64_t>(coefficient_count + 1) / 2;
    if (!std::isfinite(condition)
        || !std::isfinite(pb_identity_residual)
        || !std::isfinite(pb_max_error)) {
        throw std::runtime_error(
            "weighted reduced trace QR diagnostics are not finite");
    }
}

Eigen::SparseMatrix<double> weighted_sparse_transpose(
    const Eigen::SparseMatrix<double>& design,
    const Eigen::VectorXd& weights)
{
    Eigen::SparseMatrix<double> result = design.transpose();
    result.makeCompressed();
    for (int sample = 0; sample < result.outerSize(); ++sample) {
        for (Eigen::SparseMatrix<double>::InnerIterator entry(result, sample);
             entry; ++entry) {
            entry.valueRef() *= weights[sample];
        }
    }
    return result;
}

} // namespace

ReducedTraceProjection3D::ReducedTraceProjection3D(
    const Eigen::MatrixXd& reduced_basis_design,
    const Eigen::VectorXd& weights,
    double relative_rank_tolerance,
    bool use_weighted_qr)
{
    validate_common_inputs(
        reduced_basis_design.rows(), reduced_basis_design.cols(),
        reduced_basis_design.allFinite(), weights, true);
    validate_rank_tolerance(relative_rank_tolerance);
    sample_count_ = static_cast<int>(reduced_basis_design.rows());
    coefficient_count_ = static_cast<int>(reduced_basis_design.cols());
    relative_rank_tolerance_ = relative_rank_tolerance;
    if (use_weighted_qr) {
        storage_mode_ = StorageMode::WeightedQr;
        backend_ = ReducedTraceProjectionBackend3D::DenseWeightedQr;
        finalize_weighted_qr_projection(
            reduced_basis_design, weights, relative_rank_tolerance_,
            weighted_qr_factor_, sqrt_weights_, rank_, condition_,
            pb_identity_residual_, pb_max_error_, gram_nonzeros_,
            factor_nonzeros_);
    } else {
        storage_mode_ = StorageMode::DenseReduced;
        backend_ = ReducedTraceProjectionBackend3D::DenseLlt;
        weighted_design_transpose_ = reduced_basis_design.transpose();
        for (Eigen::Index sample = 0;
             sample < weighted_design_transpose_.cols(); ++sample) {
            weighted_design_transpose_.col(sample) *= weights[sample];
        }
        Eigen::MatrixXd gram(coefficient_count_, coefficient_count_);
        gram.noalias() = weighted_design_transpose_ * reduced_basis_design;
        gram_nonzeros_ = static_cast<std::int64_t>(coefficient_count_)
            * static_cast<std::int64_t>(coefficient_count_);
        factor_nonzeros_ = static_cast<std::int64_t>(coefficient_count_)
            * static_cast<std::int64_t>(coefficient_count_ + 1) / 2;
        finalize_gram_projection(
            std::move(gram), sample_count_, coefficient_count_,
            relative_rank_tolerance_, gram_factor_, rank_, condition_,
            pb_identity_residual_, pb_max_error_);
    }
}

ReducedTraceProjection3D::ReducedTraceProjection3D(
    const Eigen::SparseMatrix<double>& reduced_basis_design,
    const Eigen::VectorXd& weights,
    double relative_rank_tolerance,
    bool use_weighted_qr)
{
    validate_common_inputs(
        reduced_basis_design.rows(), reduced_basis_design.cols(),
        sparse_is_finite(reduced_basis_design), weights, true);
    validate_rank_tolerance(relative_rank_tolerance);
    sample_count_ = static_cast<int>(reduced_basis_design.rows());
    coefficient_count_ = static_cast<int>(reduced_basis_design.cols());
    relative_rank_tolerance_ = relative_rank_tolerance;
    storage_mode_ = StorageMode::SparseReduced;

    if (coefficient_count_
        >= sparse_cholesky_coordinate_threshold()) {
        weighted_sparse_design_transpose_ =
            weighted_sparse_transpose(reduced_basis_design, weights);
        Eigen::SparseMatrix<double> sparse_gram =
            weighted_sparse_design_transpose_ * reduced_basis_design;
        backend_ =
            ReducedTraceProjectionBackend3D::SparseSimplicialLlt;
        finalize_sparse_gram_projection(
            std::move(sparse_gram),
            sample_count_,
            coefficient_count_,
            relative_rank_tolerance_,
            sparse_gram_factor_,
            rank_,
            condition_,
            pb_identity_residual_,
            pb_max_error_,
            gram_nonzeros_,
            factor_nonzeros_);
    } else if (use_weighted_qr) {
        storage_mode_ = StorageMode::WeightedQr;
        backend_ = ReducedTraceProjectionBackend3D::DenseWeightedQr;
        finalize_weighted_qr_projection(
            Eigen::MatrixXd(reduced_basis_design),
            weights,
            relative_rank_tolerance_,
            weighted_qr_factor_,
            sqrt_weights_,
            rank_,
            condition_,
            pb_identity_residual_,
            pb_max_error_,
            gram_nonzeros_,
            factor_nonzeros_);
    } else {
        backend_ = ReducedTraceProjectionBackend3D::DenseLlt;
        weighted_sparse_design_transpose_ =
            weighted_sparse_transpose(reduced_basis_design, weights);
        Eigen::MatrixXd gram(
            weighted_sparse_design_transpose_ * reduced_basis_design);
        gram_nonzeros_ = static_cast<std::int64_t>(coefficient_count_)
            * static_cast<std::int64_t>(coefficient_count_);
        factor_nonzeros_ = static_cast<std::int64_t>(coefficient_count_)
            * static_cast<std::int64_t>(coefficient_count_ + 1) / 2;
        finalize_gram_projection(
            std::move(gram), sample_count_, coefficient_count_,
            relative_rank_tolerance_, gram_factor_, rank_, condition_,
            pb_identity_residual_, pb_max_error_);
    }
}

ReducedTraceProjection3D::ReducedTraceProjection3D(
    const Eigen::SparseMatrix<double>& c0_basis_design,
    const Eigen::MatrixXd& reduction_matrix,
    const Eigen::VectorXd& weights,
    double relative_rank_tolerance,
    bool use_weighted_qr)
{
    if (c0_basis_design.cols() <= 0
        || reduction_matrix.rows() != c0_basis_design.cols()
        || reduction_matrix.cols() <= 0) {
        throw std::invalid_argument(
            "composed trace design dimensions do not satisfy B = B0 R");
    }
    validate_common_inputs(
        c0_basis_design.rows(), reduction_matrix.cols(),
        sparse_is_finite(c0_basis_design)
            && reduction_matrix.allFinite(),
        weights,
        true);
    validate_rank_tolerance(relative_rank_tolerance);
    sample_count_ = static_cast<int>(c0_basis_design.rows());
    coefficient_count_ = static_cast<int>(reduction_matrix.cols());
    relative_rank_tolerance_ = relative_rank_tolerance;
    if (use_weighted_qr) {
        storage_mode_ = StorageMode::WeightedQr;
        backend_ = ReducedTraceProjectionBackend3D::DenseWeightedQr;
        Eigen::MatrixXd reduced_design = c0_basis_design * reduction_matrix;
        finalize_weighted_qr_projection(
            std::move(reduced_design), weights, relative_rank_tolerance_,
            weighted_qr_factor_, sqrt_weights_, rank_, condition_,
            pb_identity_residual_, pb_max_error_, gram_nonzeros_,
            factor_nonzeros_);
    } else {
        storage_mode_ = StorageMode::SparseComposed;
        backend_ = ReducedTraceProjectionBackend3D::DenseLlt;
        weighted_sparse_design_transpose_ =
            weighted_sparse_transpose(c0_basis_design, weights);
        reduction_matrix_ = reduction_matrix;
        const Eigen::SparseMatrix<double> c0_gram =
            weighted_sparse_design_transpose_ * c0_basis_design;
        const Eigen::MatrixXd gram_times_reduction =
            c0_gram * reduction_matrix_;
        Eigen::MatrixXd gram(coefficient_count_, coefficient_count_);
        gram.noalias() = reduction_matrix_.transpose()
            * gram_times_reduction;
        gram_nonzeros_ = static_cast<std::int64_t>(coefficient_count_)
            * static_cast<std::int64_t>(coefficient_count_);
        factor_nonzeros_ = static_cast<std::int64_t>(coefficient_count_)
            * static_cast<std::int64_t>(coefficient_count_ + 1) / 2;
        finalize_gram_projection(
            std::move(gram), sample_count_, coefficient_count_,
            relative_rank_tolerance_, gram_factor_, rank_, condition_,
            pb_identity_residual_, pb_max_error_);
    }
}

ReducedTraceProjection3D::ReducedTraceProjection3D(
    const Eigen::SparseMatrix<double>& c0_basis_design,
    const Eigen::SparseMatrix<double>& reduction_matrix,
    const Eigen::VectorXd& weights,
    double relative_rank_tolerance,
    bool use_weighted_qr)
{
    if (c0_basis_design.cols() <= 0
        || reduction_matrix.rows() != c0_basis_design.cols()
        || reduction_matrix.cols() <= 0) {
        throw std::invalid_argument(
            "composed trace design dimensions do not satisfy B = B0 R");
    }
    validate_common_inputs(
        c0_basis_design.rows(), reduction_matrix.cols(),
        sparse_is_finite(c0_basis_design)
            && sparse_is_finite(reduction_matrix),
        weights,
        true);
    validate_rank_tolerance(relative_rank_tolerance);
    sample_count_ = static_cast<int>(c0_basis_design.rows());
    coefficient_count_ = static_cast<int>(reduction_matrix.cols());
    relative_rank_tolerance_ = relative_rank_tolerance;
    storage_mode_ = StorageMode::SparseComposedSparseReduction;

    if (coefficient_count_
        >= sparse_cholesky_coordinate_threshold()) {
        weighted_sparse_design_transpose_ =
            weighted_sparse_transpose(c0_basis_design, weights);
        sparse_reduction_matrix_ = reduction_matrix;
        sparse_reduction_matrix_.makeCompressed();
        Eigen::SparseMatrix<double> sparse_gram;
        {
            // Limit the lifetime of the two intermediate sparse products. At
            // factorization time only the final reduced Gram remains resident.
            const Eigen::SparseMatrix<double> c0_gram =
                weighted_sparse_design_transpose_ * c0_basis_design;
            const Eigen::SparseMatrix<double> gram_times_reduction =
                c0_gram * sparse_reduction_matrix_;
            sparse_gram = sparse_reduction_matrix_.transpose()
                * gram_times_reduction;
        }
        backend_ =
            ReducedTraceProjectionBackend3D::SparseSimplicialLlt;
        finalize_sparse_gram_projection(
            std::move(sparse_gram),
            sample_count_,
            coefficient_count_,
            relative_rank_tolerance_,
            sparse_gram_factor_,
            rank_,
            condition_,
            pb_identity_residual_,
            pb_max_error_,
            gram_nonzeros_,
            factor_nonzeros_);
    } else if (use_weighted_qr) {
        storage_mode_ = StorageMode::WeightedQr;
        backend_ = ReducedTraceProjectionBackend3D::DenseWeightedQr;
        const Eigen::SparseMatrix<double> reduced_sparse_design =
            c0_basis_design * reduction_matrix;
        finalize_weighted_qr_projection(
            Eigen::MatrixXd(reduced_sparse_design),
            weights,
            relative_rank_tolerance_,
            weighted_qr_factor_,
            sqrt_weights_,
            rank_,
            condition_,
            pb_identity_residual_,
            pb_max_error_,
            gram_nonzeros_,
            factor_nonzeros_);
    } else {
        storage_mode_ = StorageMode::SparseComposedSparseReduction;
        backend_ = ReducedTraceProjectionBackend3D::DenseLlt;
        weighted_sparse_design_transpose_ =
            weighted_sparse_transpose(c0_basis_design, weights);
        sparse_reduction_matrix_ = reduction_matrix;
        sparse_reduction_matrix_.makeCompressed();
        const Eigen::SparseMatrix<double> c0_gram =
            weighted_sparse_design_transpose_ * c0_basis_design;
        const Eigen::SparseMatrix<double> gram_times_reduction =
            c0_gram * sparse_reduction_matrix_;
        Eigen::MatrixXd gram(
            sparse_reduction_matrix_.transpose()
            * gram_times_reduction);
        gram_nonzeros_ = static_cast<std::int64_t>(coefficient_count_)
            * static_cast<std::int64_t>(coefficient_count_);
        factor_nonzeros_ = static_cast<std::int64_t>(coefficient_count_)
            * static_cast<std::int64_t>(coefficient_count_ + 1) / 2;
        finalize_gram_projection(
            std::move(gram), sample_count_, coefficient_count_,
            relative_rank_tolerance_, gram_factor_, rank_, condition_,
            pb_identity_residual_, pb_max_error_);
    }
}

Eigen::VectorXd ReducedTraceProjection3D::apply(
    const Eigen::Ref<const Eigen::VectorXd>& trace) const
{
    if (trace.size() != sample_count_) {
        throw std::invalid_argument(
            "exterior trace size does not match the cached projection");
    }
    if (!trace.allFinite()) {
        throw std::invalid_argument(
            "exterior trace contains a non-finite entry");
    }
    Eigen::VectorXd coefficients;
    if (storage_mode_ == StorageMode::MaterializedProjection) {
        coefficients.noalias() = projection_cache_ * trace;
    } else if (backend_
               == ReducedTraceProjectionBackend3D::DenseWeightedQr) {
        const Eigen::VectorXd weighted_trace =
            (sqrt_weights_.array() * trace.array()).matrix();
        coefficients = weighted_qr_factor_.solve(weighted_trace);
    } else if (storage_mode_ == StorageMode::DenseReduced) {
        coefficients.noalias() = weighted_design_transpose_ * trace;
        solve_gram_in_place(coefficients);
    } else if (storage_mode_ == StorageMode::SparseReduced) {
        coefficients.noalias() = weighted_sparse_design_transpose_ * trace;
        solve_gram_in_place(coefficients);
    } else if (storage_mode_ == StorageMode::SparseComposed) {
        const Eigen::VectorXd c0_right_hand_side =
            weighted_sparse_design_transpose_ * trace;
        coefficients.noalias() =
            reduction_matrix_.transpose() * c0_right_hand_side;
        solve_gram_in_place(coefficients);
    } else {
        const Eigen::VectorXd c0_right_hand_side =
            weighted_sparse_design_transpose_ * trace;
        coefficients.noalias() =
            sparse_reduction_matrix_.transpose() * c0_right_hand_side;
        solve_gram_in_place(coefficients);
    }
    if (!coefficients.allFinite()) {
        throw std::runtime_error(
            "reduced trace projection produced non-finite coefficients");
    }
    return coefficients;
}

Eigen::VectorXd ReducedTraceProjection3D::native_to_trace_mass(
    const Eigen::Ref<const Eigen::VectorXd>& native) const
{
    if (native.size() != coefficient_count_) {
        throw std::invalid_argument(
            "native trace coordinate vector has the wrong size");
    }
    if (backend_
        == ReducedTraceProjectionBackend3D::SparseSimplicialLlt) {
        // SimplicialLLT factors P M P^-1 = L L^T.  Consequently
        // y=L^T P a is the mass coordinate paired with native a.
        const Eigen::VectorXd permuted =
            sparse_gram_factor_.permutationP() * native;
        return sparse_gram_factor_.matrixU() * permuted;
    }
    if (backend_ == ReducedTraceProjectionBackend3D::DenseWeightedQr) {
        const Eigen::VectorXd permuted =
            weighted_qr_factor_.colsPermutation().transpose() * native;
        return weighted_qr_factor_.matrixR()
            .topLeftCorner(coefficient_count_, coefficient_count_)
            .template triangularView<Eigen::Upper>() * permuted;
    }
    return gram_factor_.matrixU() * native;
}

Eigen::VectorXd ReducedTraceProjection3D::trace_mass_to_native(
    const Eigen::Ref<const Eigen::VectorXd>& mass_coordinates) const
{
    if (mass_coordinates.size() != coefficient_count_) {
        throw std::invalid_argument(
            "trace-mass coordinate vector has the wrong size");
    }
    Eigen::VectorXd native = mass_coordinates;
    if (backend_
        == ReducedTraceProjectionBackend3D::SparseSimplicialLlt) {
        sparse_gram_factor_.matrixU().solveInPlace(native);
        native = sparse_gram_factor_.permutationPinv() * native;
    } else if (backend_
               == ReducedTraceProjectionBackend3D::DenseWeightedQr) {
        weighted_qr_factor_.matrixR()
            .topLeftCorner(coefficient_count_, coefficient_count_)
            .template triangularView<Eigen::Upper>()
            .solveInPlace(native);
        native = weighted_qr_factor_.colsPermutation() * native;
    } else {
        gram_factor_.matrixU().solveInPlace(native);
    }
    return native;
}

Eigen::VectorXd ReducedTraceProjection3D::native_dual_to_trace_mass(
    const Eigen::Ref<const Eigen::VectorXd>& native_dual) const
{
    if (native_dual.size() != coefficient_count_) {
        throw std::invalid_argument(
            "native trace dual vector has the wrong size");
    }
    Eigen::VectorXd mass_dual;
    if (backend_
        == ReducedTraceProjectionBackend3D::SparseSimplicialLlt) {
        mass_dual = sparse_gram_factor_.permutationP() * native_dual;
        sparse_gram_factor_.matrixL().solveInPlace(mass_dual);
    } else if (backend_
               == ReducedTraceProjectionBackend3D::DenseWeightedQr) {
        mass_dual =
            weighted_qr_factor_.colsPermutation().transpose()
            * native_dual;
        weighted_qr_factor_.matrixR()
            .topLeftCorner(coefficient_count_, coefficient_count_)
            .transpose()
            .template triangularView<Eigen::Lower>()
            .solveInPlace(mass_dual);
    } else {
        mass_dual = native_dual;
        gram_factor_.matrixL().solveInPlace(mass_dual);
    }
    return mass_dual;
}

int ReducedTraceProjection3D::sample_count() const noexcept
{
    return sample_count_;
}

int ReducedTraceProjection3D::coefficient_count() const noexcept
{
    return coefficient_count_;
}

int ReducedTraceProjection3D::rank() const noexcept
{
    return rank_;
}

double ReducedTraceProjection3D::condition() const noexcept
{
    return condition_;
}

double ReducedTraceProjection3D::relative_rank_tolerance() const noexcept
{
    return relative_rank_tolerance_;
}

ReducedTraceProjectionBackend3D
ReducedTraceProjection3D::backend() const noexcept
{
    return backend_;
}

bool ReducedTraceProjection3D::uses_sparse_cholesky() const noexcept
{
    return backend_
        == ReducedTraceProjectionBackend3D::SparseSimplicialLlt;
}

std::int64_t ReducedTraceProjection3D::gram_nonzeros() const noexcept
{
    return gram_nonzeros_;
}

std::int64_t ReducedTraceProjection3D::factor_nonzeros() const noexcept
{
    return factor_nonzeros_;
}

double ReducedTraceProjection3D::pb_identity_residual() const noexcept
{
    return pb_identity_residual_;
}

double ReducedTraceProjection3D::pb_max_error() const noexcept
{
    return pb_max_error_;
}

const Eigen::MatrixXd&
ReducedTraceProjection3D::projection_matrix() const
{
    if (storage_mode_ != StorageMode::MaterializedProjection) {
        if (backend_
            == ReducedTraceProjectionBackend3D::DenseWeightedQr) {
            Eigen::MatrixXd thin_q = Eigen::MatrixXd::Zero(
                sample_count_, coefficient_count_);
            thin_q.topRows(coefficient_count_).setIdentity();
            thin_q = weighted_qr_factor_.householderQ() * thin_q;
            Eigen::MatrixXd reduced_rhs = thin_q.transpose();
            for (Eigen::Index sample = 0;
                 sample < reduced_rhs.cols(); ++sample) {
                reduced_rhs.col(sample) *= sqrt_weights_[sample];
            }
            weighted_qr_factor_.matrixR()
                .topLeftCorner(coefficient_count_, coefficient_count_)
                .template triangularView<Eigen::Upper>()
                .solveInPlace(reduced_rhs);
            projection_cache_.noalias() =
                weighted_qr_factor_.colsPermutation() * reduced_rhs;
        } else if (storage_mode_ == StorageMode::DenseReduced) {
            projection_cache_ = weighted_design_transpose_;
        } else if (storage_mode_ == StorageMode::SparseReduced) {
            projection_cache_ =
                Eigen::MatrixXd(weighted_sparse_design_transpose_);
        } else if (storage_mode_ == StorageMode::SparseComposed) {
            projection_cache_ = Eigen::MatrixXd::Zero(
                coefficient_count_, sample_count_);
            // Form R^T (B0^T W) directly from the sparse columns.  This lazy
            // compatibility path is intentionally not used by apply().
            for (int sample = 0;
                 sample < weighted_sparse_design_transpose_.outerSize();
                 ++sample) {
                for (Eigen::SparseMatrix<double>::InnerIterator entry(
                         weighted_sparse_design_transpose_, sample);
                     entry; ++entry) {
                    projection_cache_.col(sample).noalias() +=
                        entry.value()
                        * reduction_matrix_.row(entry.row()).transpose();
                }
            }
        } else {
            // This compatibility-only path is allowed to materialize the
            // reduced weighted transpose.  The normal apply() route above
            // retains the sparse factors and performs R^T (B0^T W t).
            const Eigen::SparseMatrix<double> reduced_weighted_transpose =
                sparse_reduction_matrix_.transpose()
                * weighted_sparse_design_transpose_;
            projection_cache_ =
                Eigen::MatrixXd(reduced_weighted_transpose);
        }
        if (backend_
            != ReducedTraceProjectionBackend3D::DenseWeightedQr) {
            solve_gram_in_place(projection_cache_);
        }
        weighted_design_transpose_.resize(0, 0);
        Eigen::SparseMatrix<double> empty_sparse;
        weighted_sparse_design_transpose_.swap(empty_sparse);
        reduction_matrix_.resize(0, 0);
        sparse_reduction_matrix_.resize(0, 0);
        storage_mode_ = StorageMode::MaterializedProjection;
    }
    return projection_cache_;
}

void ReducedTraceProjection3D::solve_gram_in_place(
    Eigen::MatrixXd& right_hand_sides) const
{
    if (backend_
        == ReducedTraceProjectionBackend3D::SparseSimplicialLlt) {
        right_hand_sides = sparse_gram_factor_.solve(right_hand_sides);
    } else if (backend_
               == ReducedTraceProjectionBackend3D::DenseLlt) {
        solve_cholesky_in_place(gram_factor_, right_hand_sides);
    } else {
        throw std::logic_error(
            "dense weighted-QR projection does not solve a Gram system");
    }
}

void ReducedTraceProjection3D::solve_gram_in_place(
    Eigen::VectorXd& right_hand_side) const
{
    if (backend_
        == ReducedTraceProjectionBackend3D::SparseSimplicialLlt) {
        right_hand_side = sparse_gram_factor_.solve(right_hand_side);
    } else if (backend_
               == ReducedTraceProjectionBackend3D::DenseLlt) {
        solve_cholesky_in_place(gram_factor_, right_hand_side);
    } else {
        throw std::logic_error(
            "dense weighted-QR projection does not solve a Gram system");
    }
}

Eigen::VectorXd mass_row(
    const Eigen::MatrixXd& reduced_basis_design,
    const Eigen::VectorXd& weights)
{
    validate_common_inputs(
        reduced_basis_design.rows(), reduced_basis_design.cols(),
        reduced_basis_design.allFinite(), weights, false);
    Eigen::VectorXd mass = reduced_basis_design.transpose() * weights;
    if (!mass.allFinite())
        throw std::runtime_error("reduced density mass row is not finite");
    return mass;
}

Eigen::VectorXd mass_row(
    const Eigen::SparseMatrix<double>& reduced_basis_design,
    const Eigen::VectorXd& weights)
{
    validate_common_inputs(
        reduced_basis_design.rows(), reduced_basis_design.cols(),
        sparse_is_finite(reduced_basis_design), weights, false);
    Eigen::VectorXd mass = reduced_basis_design.transpose() * weights;
    if (!mass.allFinite())
        throw std::runtime_error("reduced density mass row is not finite");
    return mass;
}

} // namespace kfbim::app3d
