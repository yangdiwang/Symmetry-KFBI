#pragma once

#include "dirichlet_rigid_transform_study_3d.hpp"

#include <limits>
#include <string>
#include <vector>

namespace kfbim::app3d {

struct NeumannRigidStudyMeasurement3D {
    std::string case_id;
    int N = 0;
    double h = 0.0;
    bool finite_metrics = false;
    bool gmres_converged = false;
    int gmres_iterations = 0;
    double gmres_relative_residual = 0.0;
    double interior_linf = 0.0;
    bool geometry_diagnostics_pass = false;
    bool owner_invariants_pass = false;
};

struct NeumannRigidStudyDerivedRow3D {
    NeumannRigidStudyMeasurement3D measurement;
    double interior_order = std::numeric_limits<double>::quiet_NaN();
    double baseline_error_ratio =
        std::numeric_limits<double>::quiet_NaN();
    double baseline_iteration_ratio =
        std::numeric_limits<double>::quiet_NaN();
    RigidStudyCriterionStatus3D gmres_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D baseline_ratio_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D geometry_diagnostics_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D owner_invariants_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D row_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
};

struct NeumannRigidStudyAcceptance3D {
    std::string case_id;
    RigidStudyCriterionStatus3D completeness_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D row_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D gmres_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D monotone_error_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D order_64_128_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D baseline_ratio_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D geometry_diagnostics_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D owner_invariants_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D overall_pass =
        RigidStudyCriterionStatus3D::NotEvaluated;
};

struct NeumannRigidStudyEvaluation3D {
    std::vector<NeumannRigidStudyDerivedRow3D> rows;
    std::vector<NeumannRigidStudyAcceptance3D> cases;
    bool all_pass = false;
};

std::vector<int> normalize_neumann_rigid_levels_3d(
    std::vector<int> levels);

NeumannRigidStudyEvaluation3D evaluate_neumann_rigid_study_3d(
    const std::vector<NeumannRigidStudyMeasurement3D>& measurements,
    const std::vector<std::string>& case_ids,
    bool require_complete_acceptance);

} // namespace kfbim::app3d
