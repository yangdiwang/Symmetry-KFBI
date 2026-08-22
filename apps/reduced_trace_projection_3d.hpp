#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/OrderingMethods>
#include <Eigen/QR>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <cstdint>

namespace kfbim::app3d {

enum class ReducedTraceProjectionBackend3D {
    DenseLlt,
    DenseWeightedQr,
    SparseSimplicialLlt
};

// Weighted least-squares map from exterior-trace samples to coefficients in a
// reduced surface-density basis.  For a design matrix B and positive physical
// quadrature weights w, the cached map is
//
//             P = (sqrt(W) B)^+ sqrt(W)
//               = (B^T W B)^-1 B^T W.
//
// Consequently apply(B * c) recovers c (up to floating-point error) whenever
// the weighted design has full column rank.  The legacy constructors retain
// their dense-LLT behavior by default.  The topology-affine route opts into
// direct column-pivoted QR of sqrt(W)B for small/medium systems, avoiding the
// squared condition number of a dense Gram matrix.  Large sparse systems keep
// the Gram operator sparse and use AMD-ordered SimplicialLLT.
class ReducedTraceProjection3D {
public:
    explicit ReducedTraceProjection3D(
        const Eigen::MatrixXd& reduced_basis_design,
        const Eigen::VectorXd& weights,
        double relative_rank_tolerance = 1.0e-12,
        bool use_weighted_qr = false);

    explicit ReducedTraceProjection3D(
        const Eigen::SparseMatrix<double>& reduced_basis_design,
        const Eigen::VectorXd& weights,
        double relative_rank_tolerance = 1.0e-12,
        bool use_weighted_qr = false);

    // Composition B = B0 R with a dense reduction.  B0 is the sparse C0
    // tensor-spline design and R maps reduced coefficients to C0
    // coefficients.  This small/medium compatibility path materializes B for
    // the numerically safer weighted QR factorization.
    ReducedTraceProjection3D(
        const Eigen::SparseMatrix<double>& c0_basis_design,
        const Eigen::MatrixXd& reduction_matrix,
        const Eigen::VectorXd& weights,
        double relative_rank_tolerance = 1.0e-12,
        bool use_weighted_qr = false);

    // Sparse counterpart of the composition above.  Small/medium systems
    // materialize B0 R only for weighted QR.  At
    // sparse_cholesky_coordinate_threshold() and above neither B0 R nor a
    // dense Gram is materialized; the reduced Gram remains sparse and is
    // factored by AMD-ordered sparse Cholesky.
    ReducedTraceProjection3D(
        const Eigen::SparseMatrix<double>& c0_basis_design,
        const Eigen::SparseMatrix<double>& reduction_matrix,
        const Eigen::VectorXd& weights,
        double relative_rank_tolerance = 1.0e-12,
        bool use_weighted_qr = false);

    [[nodiscard]] Eigen::VectorXd apply(
        const Eigen::Ref<const Eigen::VectorXd>& trace) const;

    // Trace-mass coordinates y=U a with U^T U=B^T W B.  Weighted QR uses
    // U=R P^T; sparse Cholesky uses the equivalent permuted triangular
    // factor.  In either case sqrt(W) B U^{-1} has orthonormal columns.
    [[nodiscard]] Eigen::VectorXd native_to_trace_mass(
        const Eigen::Ref<const Eigen::VectorXd>& native) const;
    [[nodiscard]] Eigen::VectorXd trace_mass_to_native(
        const Eigen::Ref<const Eigen::VectorXd>& mass_coordinates) const;

    // Covector transform paired with trace_mass_to_native().  If
    // y=U*a are trace-mass coordinates and ell^T*a is a native-coordinate
    // functional, this returns U^{-T}*ell so that the same functional is
    // represented by result^T*y.
    [[nodiscard]] Eigen::VectorXd native_dual_to_trace_mass(
        const Eigen::Ref<const Eigen::VectorXd>& native_dual) const;

    [[nodiscard]] int sample_count() const noexcept;
    [[nodiscard]] int coefficient_count() const noexcept;
    [[nodiscard]] int rank() const noexcept;
    [[nodiscard]] double condition() const noexcept;
    [[nodiscard]] double relative_rank_tolerance() const noexcept;
    [[nodiscard]] ReducedTraceProjectionBackend3D backend() const noexcept;
    [[nodiscard]] bool uses_sparse_cholesky() const noexcept;
    [[nodiscard]] std::int64_t gram_nonzeros() const noexcept;
    [[nodiscard]] std::int64_t factor_nonzeros() const noexcept;

    // The threshold is deliberately part of the public diagnostics contract:
    // callers and regression tests can tell when the scalable backend must be
    // active without duplicating an implementation constant.
    [[nodiscard]] static constexpr int
    sparse_cholesky_coordinate_threshold() noexcept
    {
        return 512;
    }

    // ||P B - I||_F / sqrt(K), where K is the coefficient count.  It is
    // evaluated exactly for moderate K and by deterministic probes for large
    // K so diagnostics do not require another cubic dense solve.
    [[nodiscard]] double pb_identity_residual() const noexcept;

    // max_ij |(P B - I)_ij| for moderate K; a deterministic sampled-column
    // maximum for large K.
    [[nodiscard]] double pb_max_error() const noexcept;

    // Materializes P on first use.  Normal operator applications do not call
    // this function and therefore do not retain a second K-by-M dense matrix.
    [[nodiscard]] const Eigen::MatrixXd& projection_matrix() const;

private:
    using SparseGramMatrix = Eigen::SparseMatrix<double>;
    using SparseGramFactor = Eigen::SimplicialLLT<
        SparseGramMatrix,
        Eigen::Lower,
        Eigen::AMDOrdering<int>>;

    enum class StorageMode {
        WeightedQr,
        DenseReduced,
        SparseReduced,
        SparseComposed,
        SparseComposedSparseReduction,
        MaterializedProjection
    };

    void solve_gram_in_place(Eigen::MatrixXd& right_hand_sides) const;
    void solve_gram_in_place(Eigen::VectorXd& right_hand_side) const;

    // Both matrices are mutable logical caches.  Initially only B^T W is
    // populated.  If projection_matrix() is requested, it is transformed into
    // P and the weighted transpose storage is released after the solve.
    mutable Eigen::MatrixXd weighted_design_transpose_;
    mutable Eigen::SparseMatrix<double> weighted_sparse_design_transpose_;
    mutable Eigen::MatrixXd reduction_matrix_;
    mutable Eigen::SparseMatrix<double> sparse_reduction_matrix_;
    mutable Eigen::MatrixXd projection_cache_;
    mutable StorageMode storage_mode_ = StorageMode::DenseReduced;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> weighted_qr_factor_;
    Eigen::VectorXd sqrt_weights_;
    Eigen::LLT<Eigen::MatrixXd, Eigen::Lower> gram_factor_;
    SparseGramFactor sparse_gram_factor_;
    ReducedTraceProjectionBackend3D backend_ =
        ReducedTraceProjectionBackend3D::DenseLlt;
    int sample_count_ = 0;
    int coefficient_count_ = 0;
    int rank_ = 0;
    double condition_ = 0.0;
    double relative_rank_tolerance_ = 0.0;
    double pb_identity_residual_ = 0.0;
    double pb_max_error_ = 0.0;
    std::int64_t gram_nonzeros_ = 0;
    std::int64_t factor_nonzeros_ = 0;
};

// Surface-mass functional for reduced coefficients: m = B^T w.
[[nodiscard]] Eigen::VectorXd mass_row(
    const Eigen::MatrixXd& reduced_basis_design,
    const Eigen::VectorXd& weights);

[[nodiscard]] Eigen::VectorXd mass_row(
    const Eigen::SparseMatrix<double>& reduced_basis_design,
    const Eigen::VectorXd& weights);

} // namespace kfbim::app3d
