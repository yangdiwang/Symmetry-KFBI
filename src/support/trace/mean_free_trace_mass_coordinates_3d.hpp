#pragma once

#include <Eigen/Core>

namespace kfbim::app3d {

struct MeanFreeTraceMassCoordinateOptions3D {
    double relative_observability_tolerance = 1.0e-13;
    double invariant_tolerance = 2.0e-12;
};

struct MeanFreeTraceMassCoordinateDiagnostics3D {
    Eigen::Index source_coordinates = 0;
    Eigen::Index reduced_coordinates = 0;
    Eigen::Index pivot_coordinate = -1;
    double pivot_moment = 0.0;
    double moment_norm = 0.0;
    double relative_observability = 0.0;
    double maximum_elimination_multiplier = 0.0;
    double coordinate_map_condition = 1.0;
    double mean_orthogonality_residual = 0.0;
    double orthonormality_residual = 0.0;
};

// An implicit orthonormal basis Q for the codimension-one trace-mass space
//
//                       { z : d^T z = 0 }.
//
// A maximum-moment pivot first gives the sparse elimination map
//
//     H y = E y - e_p r^T y,       r_j = d_j / d_p,
//
// and Q=H(H^T H)^(-1/2).  Since H^T H=I+r r^T, both Q and Q^T are applied
// with one rank-one update.  No K-by-(K-1) dense matrix is stored.
class MeanFreeTraceMassCoordinates3D {
public:
    explicit MeanFreeTraceMassCoordinates3D(
        const Eigen::Ref<const Eigen::VectorXd>& mean_dual,
        MeanFreeTraceMassCoordinateOptions3D options = {});

    [[nodiscard]] Eigen::Index source_coordinate_count() const noexcept;
    [[nodiscard]] Eigen::Index reduced_coordinate_count() const noexcept;

    // Return Q*y in the full trace-mass coordinate space.
    [[nodiscard]] Eigen::VectorXd lift(
        const Eigen::Ref<const Eigen::VectorXd>& reduced) const;

    // Return Q^T*z.  This is both restriction to the final residual space
    // and the Euclidean adjoint of lift().
    [[nodiscard]] Eigen::VectorXd restrict(
        const Eigen::Ref<const Eigen::VectorXd>& full) const;

    // Return Q Q^T*z, the orthogonal projection onto the mean-free space.
    [[nodiscard]] Eigen::VectorXd project(
        const Eigen::Ref<const Eigen::VectorXd>& full) const;

    [[nodiscard]] const Eigen::VectorXd& mean_dual() const noexcept;
    [[nodiscard]] const MeanFreeTraceMassCoordinateDiagnostics3D&
    diagnostics() const noexcept;

private:
    [[nodiscard]] Eigen::VectorXd apply_inverse_square_root(
        const Eigen::Ref<const Eigen::VectorXd>& values) const;

    Eigen::VectorXd mean_dual_;
    Eigen::VectorXd multipliers_;
    Eigen::Index pivot_ = -1;
    double inverse_square_root_rank_one_coefficient_ = 0.0;
    MeanFreeTraceMassCoordinateDiagnostics3D diagnostics_;
};

} // namespace kfbim::app3d
