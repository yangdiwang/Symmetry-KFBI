#pragma once

#include "src/support/topology/topology_affine_reduction_3d.hpp"

#include <string>
#include <vector>

namespace kfbim::app3d {

struct ReachableTargetProjectionOptions3D {
    double relative_rank_tolerance = 1.0e-11;
    double exact_tolerance = 2.0e-10;
    double certification_tolerance = 2.0e-9;
};

struct ReachableTargetProjectionRecord3D {
    ConstraintBlockKind3D kind = ConstraintBlockKind3D::Edge;
    std::string key;
    std::vector<std::string> source_keys;
    std::vector<Eigen::Index> row_ids;
    Eigen::Index rows = 0;
    Eigen::Index support_cols = 0;
    Eigen::Index rank = 0;
    double requested_l2 = 0.0;
    double projection_l2 = 0.0;
    double projection_linf = 0.0;
};

// Projection of an affine constraint target onto the part reachable from a
// previously fixed topology space y=p+E*z.  Only rows in one connected
// component of target.C*E are converted to dense form; there is no global
// dense matrix and no least-squares relaxation of the projected constraints.
struct ReachableConstraintTargetProjection3D {
    ConstraintSystem3D projected_system;
    Eigen::VectorXd requested_rhs;
    Eigen::VectorXd projected_rhs;
    Eigen::VectorXd reachable_coordinates;
    std::vector<ReachableTargetProjectionRecord3D> records;
    double requested_linf = 0.0;
    double requested_l2 = 0.0;
    double projected_linf = 0.0;
    double projected_l2 = 0.0;
    double projection_linf = 0.0;
    double projection_l2 = 0.0;
    double relative_projection_l2 = 0.0;
    double reachable_certification_linf = 0.0;
    bool constraints_exact = true;
};

[[nodiscard]] ReachableConstraintTargetProjection3D
project_constraint_target_to_reachable_space_3d(
    const AffineReduction3D& fixed_topology,
    ConstraintSystem3D target_system,
    std::vector<ConstraintBlock3D> target_blocks,
    ReachableTargetProjectionOptions3D options = {});

} // namespace kfbim::app3d
