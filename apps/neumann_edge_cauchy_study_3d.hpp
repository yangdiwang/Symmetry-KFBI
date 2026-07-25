#pragma once

#include "dirichlet_rigid_transform_study_3d.hpp"
#include "neumann_edge_augmented_cauchy_3d.hpp"

#include <limits>
#include <string>
#include <vector>

namespace kfbim::app3d {

struct NeumannEdgeCauchyMeasurement3D {
    std::string case_id;
    int N = 0;
    double h = 0.0;
    NeumannEdgeCauchyMode3D mode = NeumannEdgeCauchyMode3D::None;
    bool finite_metrics = false;
    bool gmres_converged = false;
    int gmres_iterations = 0;
    double gmres_relative_residual = 0.0;
    double density_linf = 0.0;
    double density_l2 = 0.0;
    double interior_linf = 0.0;
    double interior_l2 = 0.0;
    double incident_edge_discrepancy_linf = 0.0;
    int expected_non_g1_connections = 0;
    int covered_non_g1_connections = 0;
    int edge_sample_count = 0;
    int affected_center_count = 0;
    int corner_center_count = 0;
    int unrelated_sample_or_attachment_count = 0;
    int rank_deficient_fit_count = 0;
    double harmonic_cubic_reproduction_defect = 0.0;
    double edge_condition_max = 0.0;
    double local_condition_max = 0.0;
    double shared_setup_seconds = 0.0;
    double mode_runtime_seconds = 0.0;
    double total_seconds = 0.0;
    bool far_centers_bitwise_legacy = false;
    bool geometry_diagnostics_pass = false;
    bool owner_invariants_pass = false;
    bool shared_preprocess_pass = false;
};

struct NeumannEdgeCauchyDerivedRow3D {
    NeumannEdgeCauchyMeasurement3D measurement;
    double density_linf_order = std::numeric_limits<double>::quiet_NaN();
    double density_l2_order = std::numeric_limits<double>::quiet_NaN();
    double interior_linf_order = std::numeric_limits<double>::quiet_NaN();
    double interior_l2_order = std::numeric_limits<double>::quiet_NaN();
    double density_linf_ratio_to_legacy = std::numeric_limits<double>::quiet_NaN();
    double density_l2_ratio_to_legacy = std::numeric_limits<double>::quiet_NaN();
    double interior_linf_ratio_to_legacy = std::numeric_limits<double>::quiet_NaN();
    double interior_l2_ratio_to_legacy = std::numeric_limits<double>::quiet_NaN();
    double edge_discrepancy_ratio_to_legacy = std::numeric_limits<double>::quiet_NaN();
    RigidStudyCriterionStatus3D row_pass = RigidStudyCriterionStatus3D::NotEvaluated;
};

struct NeumannEdgeCauchyAcceptance3D {
    RigidStudyCriterionStatus3D completeness_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D structure_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D reproduction_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D gmres_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D error_guard_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D order_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D rigid_spread_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D edge_discrepancy_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D geometry_owner_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D extended_evidence_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D overall_pass = RigidStudyCriterionStatus3D::NotEvaluated;
};

struct NeumannEdgeCauchyEvaluation3D {
    std::vector<NeumannEdgeCauchyDerivedRow3D> rows;
    NeumannEdgeCauchyAcceptance3D acceptance;
    bool all_pass = false;
};

std::vector<int> normalize_neumann_edge_cauchy_levels_3d(std::vector<int> levels);

NeumannEdgeCauchyEvaluation3D evaluate_neumann_edge_cauchy_study_3d(
    const std::vector<NeumannEdgeCauchyMeasurement3D>& measurements,
    const std::vector<std::string>& case_ids,
    bool require_complete_pilot);

bool neumann_edge_cauchy_study_exit_pass_3d(
    const NeumannEdgeCauchyEvaluation3D& evaluation,
    bool require_complete_pilot);

} // namespace kfbim::app3d