#pragma once

#include <Eigen/Dense>

#include <vector>

namespace kfbim::app3d {

enum class HarmonicTraceCorrectionMode3D {
    CenterOwned,
    CrossingOwned,
};

struct HarmonicTraceOwnerTerm3D {
    int owner_dof = -1;
    Eigen::VectorXd evaluation;
};

[[nodiscard]] double apply_harmonic_trace_correction_3d(
    int center_dof,
    const Eigen::MatrixXd& coefficients,
    const Eigen::VectorXd& center_evaluation,
    const std::vector<HarmonicTraceOwnerTerm3D>& owner_terms,
    HarmonicTraceCorrectionMode3D mode);

} // namespace kfbim::app3d
