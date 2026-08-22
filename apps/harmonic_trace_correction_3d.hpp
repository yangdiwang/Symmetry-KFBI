#pragma once

#include <Eigen/Core>

#include <vector>

namespace kfbim::app3d {

enum class TraceCorrectionOwnerMode3D {
    CenterDof,
    CrossingOwner
};

struct HarmonicTraceCorrectionTermInput3D {
    int owner_dof = -1;
    Eigen::VectorXd evaluation;
};

[[nodiscard]] double apply_harmonic_trace_correction_3d(
    int center_dof,
    const Eigen::MatrixXd& coefficients,
    const Eigen::VectorXd& legacy_evaluation,
    const std::vector<HarmonicTraceCorrectionTermInput3D>& owner_terms,
    TraceCorrectionOwnerMode3D mode);

} // namespace kfbim::app3d
