#pragma once

#include "topology_affine_reduction_3d.hpp"

#include <Eigen/Core>

namespace kfbim::app3d {

// Options for the final, global rank-one Neumann gauge elimination.  This
// reduction is deliberately applied after all local topology constraints: the
// mean row has global support and must not merge otherwise-local blocks.
struct TopologyMeanFreeReductionOptions3D {
    // The mean constraint must be observable in range(G).  Observability is
    // measured relative to ||m||_2 max_j ||G_j||_2.
    double relative_observability_tolerance = 1.0e-13;
    // Post-construction invariants are checked at a scale-aware tolerance.
    double invariant_tolerance = 2.0e-11;
};

struct TopologyMeanFreeReductionDiagnostics3D {
    Eigen::Index source_coordinates = 0;
    Eigen::Index reduced_coordinates = 0;
    // H has an identity minor of this rank, independently of floating-point
    // rank decisions.  If the input G has full column rank, G H has this rank.
    Eigen::Index coordinate_rank = 0;
    Eigen::Index pivot_coordinate = -1;
    double pivot_moment = 0.0;
    double moment_norm = 0.0;
    double relative_observability = 0.0;
    double maximum_elimination_multiplier = 0.0;
    // Exact 2-norm condition of the sparse coordinate map H.  With at least
    // two output columns and the maximum-moment pivot it is
    // sqrt(1 + ||r||_2^2), r_j=a_j/a_p.  A zero/one-column map has condition
    // one by convention/the definition of its nonzero singular spectrum.
    double coordinate_map_condition = 1.0;
    double particular_mean_before = 0.0;
    double target_mean = 0.0;
    double particular_mean_after = 0.0;
    double particular_mean_residual = 0.0;
    // ||a^T H||_inf, where a=G^T m.  This certifies that the eliminated
    // coordinate directions are mean-free before applying G.
    double coordinate_mean_orthogonality_residual = 0.0;
    // ||m^T G H||_inf, the corresponding coefficient-space invariant.
    double homogeneous_mean_residual = 0.0;
};

// Affine result of eliminating one globally-supported mean degree of freedom:
//
//   c = c_p + G z,       m^T c = target
//
// becomes
//
//   c = c_tilde + G_tilde y,
//   c_tilde = c_p + G z_0,       G_tilde = G H,
//   m^T c_tilde = target,        m^T G_tilde = 0.
//
// H uses a maximum-moment pivot.  It has at most two nonzeros per column and
// an identity minor, so no dense global nullspace or SVD is materialized.
class TopologyMeanFreeReduction3D {
public:
    TopologyMeanFreeReduction3D() = default;

    [[nodiscard]] Eigen::Index coefficient_count() const noexcept;
    [[nodiscard]] Eigen::Index source_coordinate_count() const noexcept;
    [[nodiscard]] Eigen::Index reduced_coordinate_count() const noexcept;

    [[nodiscard]] const Eigen::VectorXd& particular() const noexcept;
    [[nodiscard]] const SparseMatrixCSR3D& homogeneous() const noexcept;

    // z = coordinate_particular() + coordinate_homogeneous() * y maps the
    // final mean-free coordinates back to the pre-mean topology coordinates.
    [[nodiscard]] const Eigen::VectorXd& coordinate_particular() const noexcept;
    [[nodiscard]] const SparseMatrixCSR3D& coordinate_homogeneous() const
        noexcept;

    [[nodiscard]] Eigen::VectorXd lift_homogeneous(
        const Eigen::Ref<const Eigen::VectorXd>& reduced) const;
    [[nodiscard]] Eigen::VectorXd lift(
        const Eigen::Ref<const Eigen::VectorXd>& reduced) const;

    [[nodiscard]] const TopologyMeanFreeReductionDiagnostics3D& diagnostics()
        const noexcept;

private:
    friend TopologyMeanFreeReduction3D eliminate_topology_mean_3d(
        const Eigen::Ref<const Eigen::VectorXd>&,
        const SparseMatrixCSR3D&,
        const Eigen::Ref<const Eigen::VectorXd>&,
        double,
        TopologyMeanFreeReductionOptions3D);

    Eigen::VectorXd particular_;
    SparseMatrixCSR3D homogeneous_;
    Eigen::VectorXd coordinate_particular_;
    SparseMatrixCSR3D coordinate_homogeneous_;
    TopologyMeanFreeReductionDiagnostics3D diagnostics_;
};

[[nodiscard]] TopologyMeanFreeReduction3D eliminate_topology_mean_3d(
    const Eigen::Ref<const Eigen::VectorXd>& particular,
    const SparseMatrixCSR3D& homogeneous,
    const Eigen::Ref<const Eigen::VectorXd>& mean_dual,
    double target_mean = 0.0,
    TopologyMeanFreeReductionOptions3D options = {});

} // namespace kfbim::app3d
