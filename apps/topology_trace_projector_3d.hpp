#pragma once

#include "reduced_trace_projection_3d.hpp"
#include "topology_affine_reduction_3d.hpp"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cstdint>
#include <memory>

namespace kfbim::app3d {

// Diagnostics are kept separate from the operator API so drivers can report
// the projector quality without materializing Bz = B_test A0 E.
struct TopologyTraceProjectorDiagnostics3D {
    int samples = 0;
    int reduced_coordinates = 0;
    int rank = 0;
    double condition = 0.0;
    double pb_identity_residual = 0.0;
    double pb_max_error = 0.0;
    double full_roundtrip_residual = 0.0;
    double full_roundtrip_max_error = 0.0;
    int full_roundtrip_probes = 0;
    ReducedTraceProjectionBackend3D backend =
        ReducedTraceProjectionBackend3D::DenseWeightedQr;
    std::int64_t gram_nonzeros = 0;
    std::int64_t factor_nonzeros = 0;
};

// Unified density-space projector used by both Neumann value traces and
// Dirichlet normal traces.  The affine particular coefficient vector is
// cached for the fixed base field, but it is deliberately excluded from
// project() and lift_homogeneous_c0(): GMRES acts only on the homogeneous
// space G z, G=A0 E.
//
// project() returns trace-mass coordinates y=L^T z, where
// Bz^T W Bz=L L^T.  Thus the coordinate norm used by GMRES is the physical
// weighted trace norm of the represented density.
//
// Construction is intentionally overdetermined: the number of finite,
// positive-weight trace observations must be strictly greater than the final
// homogeneous coordinate count.  This observability gate is checked before
// the weighted QR/Cholesky factorization; full-rank and condition-limit checks
// are then applied independently.
class TopologyTraceProjector3D {
public:
    TopologyTraceProjector3D(
        const SparseMatrixCSR3D& c0_test_design,
        const SparseMatrixCSR3D& base_expansion,
        const AffineReduction3D& reduction,
        const Eigen::VectorXd& weights,
        double relative_rank_tolerance = 1.0e-12,
        double maximum_condition = 1.0e12);

    // Build directly from an already assembled affine C0 space
    //
    //   c = particular_c0 + homogeneous_c0 z.
    //
    // This overload is used after the final global Neumann mean constraint
    // has eliminated one topology coordinate.  The pivot elimination is
    // intentionally sparse but is not an Euclidean-orthogonal reduction, so
    // it must not be disguised as an AffineReduction3D.  The trace-mass
    // factorization below supplies the appropriate metric coordinates.
    TopologyTraceProjector3D(
        const SparseMatrixCSR3D& c0_test_design,
        Eigen::VectorXd particular_c0,
        SparseMatrixCSR3D homogeneous_c0,
        const Eigen::VectorXd& weights,
        double relative_rank_tolerance = 1.0e-12,
        double maximum_condition = 1.0e12);

    // Accept either Eigen sparse storage order while retaining CSR as the
    // canonical topology representation.
    template <int TestOptions, typename TestIndex,
              int ExpansionOptions, typename ExpansionIndex>
    TopologyTraceProjector3D(
        const Eigen::SparseMatrix<double, TestOptions, TestIndex>&
            c0_test_design,
        const Eigen::SparseMatrix<double, ExpansionOptions, ExpansionIndex>&
            base_expansion,
        const AffineReduction3D& reduction,
        const Eigen::VectorXd& weights,
        double relative_rank_tolerance = 1.0e-12,
        double maximum_condition = 1.0e12)
        : TopologyTraceProjector3D(
              SparseMatrixCSR3D(c0_test_design),
              SparseMatrixCSR3D(base_expansion),
              reduction,
              weights,
              relative_rank_tolerance,
              maximum_condition)
    {
    }

    [[nodiscard]] Eigen::VectorXd project(
        const Eigen::Ref<const Eigen::VectorXd>& full_trace) const;

    [[nodiscard]] Eigen::VectorXd native_to_trace_mass(
        const Eigen::Ref<const Eigen::VectorXd>& native_reduced) const;
    [[nodiscard]] Eigen::VectorXd trace_mass_to_native(
        const Eigen::Ref<const Eigen::VectorXd>& trace_mass) const;

    [[nodiscard]] Eigen::VectorXd lift_homogeneous_c0(
        const Eigen::Ref<const Eigen::VectorXd>& trace_mass) const;
    [[nodiscard]] Eigen::VectorXd lift_full_c0(
        const Eigen::Ref<const Eigen::VectorXd>& trace_mass) const;

    // For a coefficient-space functional m^T c, return the covector in
    // trace-mass coordinates.  Together with particular_mean(), this gives
    //   m^T(c_p + G z) = particular_mean(m) + result^T y.
    [[nodiscard]] Eigen::VectorXd mean_dual_to_trace_mass(
        const Eigen::Ref<const Eigen::VectorXd>& c0_mean_dual) const;
    [[nodiscard]] double particular_mean(
        const Eigen::Ref<const Eigen::VectorXd>& c0_mean_dual) const;

    [[nodiscard]] const Eigen::VectorXd& particular_c0() const noexcept;
    [[nodiscard]] const SparseMatrixCSR3D& homogeneous_c0() const noexcept;

    [[nodiscard]] int sample_count() const noexcept;
    [[nodiscard]] int c0_coefficient_count() const noexcept;
    [[nodiscard]] int reduced_coordinate_count() const noexcept;
    [[nodiscard]] int rank() const noexcept;
    [[nodiscard]] double condition() const noexcept;
    [[nodiscard]] double maximum_condition() const noexcept;
    [[nodiscard]] double pb_identity_residual() const noexcept;
    [[nodiscard]] double pb_max_error() const noexcept;
    [[nodiscard]] double full_roundtrip_residual() const noexcept;
    [[nodiscard]] double full_roundtrip_max_error() const noexcept;
    [[nodiscard]] ReducedTraceProjectionBackend3D backend() const noexcept;
    [[nodiscard]] bool uses_sparse_cholesky() const noexcept;
    [[nodiscard]] std::int64_t gram_nonzeros() const noexcept;
    [[nodiscard]] std::int64_t factor_nonzeros() const noexcept;
    [[nodiscard]] const TopologyTraceProjectorDiagnostics3D& diagnostics()
        const noexcept;

    // Compatibility/debugging only.  The returned matrix maps a trace to
    // native z coordinates, not to trace-mass coordinates, and is built only
    // when this method is called.
    [[nodiscard]] const Eigen::MatrixXd& native_projection_matrix() const;

private:
    void initialize_projector(
        const SparseMatrixCSR3D& c0_test_design,
        const Eigen::VectorXd& weights,
        double relative_rank_tolerance);
    void certify_full_roundtrip(
        const SparseMatrixCSR3D& c0_test_design);
    void require_c0_size(Eigen::Index size) const;

    Eigen::VectorXd particular_c0_;
    SparseMatrixCSR3D homogeneous_c0_;
    std::unique_ptr<ReducedTraceProjection3D> native_projector_;
    double maximum_condition_ = 0.0;
    TopologyTraceProjectorDiagnostics3D diagnostics_;
};

} // namespace kfbim::app3d
