#pragma once

#include "src/support/diagnostics/dirichlet_rigid_transform_study_3d.hpp"
#include "src/support/geometry/native_nurbs_surface_3d.hpp"
#include "src/operators/i_kfbi_operator.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace kfbim::app3d {

enum class NeumannDensitySpace3D { PatchIndependent, NonG1EdgeProjected };

const char* neumann_density_space_name_3d(NeumannDensitySpace3D mode);

struct NeumannEdgeConstraintSample3D {
    int connection_index = -1;
    int sample_index = -1;
    int sample_count = 0;
    int first_patch = -1;
    int second_patch = -1;
    PatchEdge3D first_edge = PatchEdge3D::UMin;
    PatchEdge3D second_edge = PatchEdge3D::UMin;
    double normalized_parameter = 0.0;
    double first_parameter = 0.0;
    double second_parameter = 0.0;
    double quadrature_weight = 0.0;
    double mapped_point_gap = 0.0;
    bool first_reduced_order = false;
    bool second_reduced_order = false;
};

struct NeumannEdgeConstraintSet3D {
    int density_size = 0;
    int non_g1_connection_count = 0;
    int reduced_order_row_count = 0;
    Eigen::SparseMatrix<double, Eigen::RowMajor> matrix;
    Eigen::VectorXd quadrature_weights;
    std::vector<NeumannEdgeConstraintSample3D> samples;
};

struct NeumannEdgeConstraintAudit3D {
    int expected_non_g1_connections = 0;
    int covered_non_g1_connections = 0;
    int duplicate_connection_intervals = 0;
    int g1_constraint_rows = 0;
    int unrelated_constraint_rows = 0;
    int constraint_rows = 0;
    int reduced_order_rows = 0;
    bool pass = false;
};

NeumannEdgeConstraintAudit3D audit_neumann_edge_constraints_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double h,
    const NeumannEdgeConstraintSet3D& constraints);

NeumannEdgeConstraintSet3D build_neumann_edge_constraints_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double h);

Eigen::VectorXd apply_neumann_edge_constraints_3d(
    const NeumannEdgeConstraintSet3D& constraints,
    const Eigen::VectorXd& density);

double neumann_edge_mismatch_linf_3d(
    const NeumannEdgeConstraintSet3D& constraints,
    const Eigen::VectorXd& density);

double neumann_edge_mismatch_weighted_rms_3d(
    const NeumannEdgeConstraintSet3D& constraints,
    const Eigen::VectorXd& density);

struct NeumannEdgeProjectionDiagnostics3D {
    int input_constraint_count = 0;
    int retained_constraint_rank = 0;
    double rank_tolerance = 0.0;
    double factorization_seconds = 0.0;
    double constant_constraint_defect = 0.0;
};

class NeumannEdgeContinuityProjector3D {
public:
    NeumannEdgeContinuityProjector3D(
        const NeumannEdgeConstraintSet3D& constraints,
        const SurfaceDofCloud3D& cloud,
        double rank_tolerance = 1e-12);

    int density_size() const;
    int constraint_count() const;
    int retained_rank() const;

    Eigen::VectorXd project(const Eigen::VectorXd& density) const;
    Eigen::VectorXd complement(const Eigen::VectorXd& density) const;
    Eigen::VectorXd constraint_mismatch(const Eigen::VectorXd& density) const;
    double mismatch_linf(const Eigen::VectorXd& density) const;
    double mismatch_weighted_rms(const Eigen::VectorXd& density) const;

    const NeumannEdgeProjectionDiagnostics3D& diagnostics() const;

private:
    int density_size_ = 0;
    Eigen::SparseMatrix<double, Eigen::RowMajor> physical_constraints_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> retained_scaled_constraints_;
    Eigen::VectorXd inverse_surface_mass_;
    Eigen::VectorXd edge_quadrature_weights_;
    Eigen::LDLT<Eigen::MatrixXd> gram_factor_;
    std::vector<int> retained_original_rows_;
    NeumannEdgeProjectionDiagnostics3D diagnostics_;
};

class NeumannEdgeProjectedAugmentedOperator3D final
    : public kfbim::IKFBIOperator {
public:
    NeumannEdgeProjectedAugmentedOperator3D(
        const kfbim::IKFBIOperator& base,
        const NeumannEdgeContinuityProjector3D& projector);

    int problem_size() const override;
    void apply(
        const Eigen::VectorXd& unknown,
        Eigen::VectorXd& result) const override;

    Eigen::VectorXd project_right_hand_side(
        const Eigen::VectorXd& base_rhs) const;

private:
    const kfbim::IKFBIOperator& base_;
    const NeumannEdgeContinuityProjector3D& projector_;
};

struct NeumannEdgePreprocessInvariantSnapshot3D {
    std::uint64_t workload_fingerprint = 0;
    std::uint64_t output_digest = 0;
    std::uint64_t wrong_side_queries = 0;
    std::uint64_t geometry_queries = 0;
    bool diagnostics_match_reference = false;
};

bool neumann_edge_shared_preprocess_pass_3d(
    const std::vector<NeumannEdgePreprocessInvariantSnapshot3D>& snapshots);

struct NeumannEdgeContinuityMeasurement3D {
    std::string case_id;
    int N = 0;
    double h = 0.0;
    NeumannDensitySpace3D density_space = NeumannDensitySpace3D::PatchIndependent;
    bool finite_metrics = false;
    bool gmres_converged = false;
    int gmres_iterations = 0;
    double gmres_relative_residual = 0.0;
    double density_linf = 0.0;
    double density_l2 = 0.0;
    double interior_linf = 0.0;
    double interior_l2 = 0.0;
    double edge_mismatch_linf = 0.0;
    double edge_mismatch_weighted_rms = 0.0;
    double exact_edge_mismatch_linf = 0.0;
    int expected_non_g1_connections = 0;
    int covered_non_g1_connections = 0;
    int duplicate_connection_intervals = 0;
    int g1_constraint_rows = 0;
    int unrelated_constraint_rows = 0;
    int constraint_rows = 0;
    int constraint_rank = 0;
    int reduced_order_rows = 0;
    double constant_constraint_defect = 0.0;
    double projected_constraint_defect = 0.0;
    double projection_idempotence_defect = 0.0;
    double constant_projection_defect = 0.0;
    bool geometry_diagnostics_pass = false;
    bool owner_invariants_pass = false;
    bool shared_preprocess_pass = false;
};

struct NeumannEdgeContinuityDerivedRow3D {
    NeumannEdgeContinuityMeasurement3D measurement;
    double density_linf_order = std::numeric_limits<double>::quiet_NaN();
    double density_l2_order = std::numeric_limits<double>::quiet_NaN();
    double interior_linf_order = std::numeric_limits<double>::quiet_NaN();
    double interior_l2_order = std::numeric_limits<double>::quiet_NaN();
    double exact_edge_mismatch_order = std::numeric_limits<double>::quiet_NaN();
    double density_linf_ratio_to_unconstrained = std::numeric_limits<double>::quiet_NaN();
    double density_l2_ratio_to_unconstrained = std::numeric_limits<double>::quiet_NaN();
    double interior_linf_ratio_to_unconstrained = std::numeric_limits<double>::quiet_NaN();
    double interior_l2_ratio_to_unconstrained = std::numeric_limits<double>::quiet_NaN();
    double edge_reduction_ratio = std::numeric_limits<double>::quiet_NaN();
    RigidStudyCriterionStatus3D row_pass = RigidStudyCriterionStatus3D::NotEvaluated;
};

struct NeumannEdgeContinuityAcceptance3D {
    RigidStudyCriterionStatus3D completeness_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D topology_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D projector_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D exact_trace_order_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D gmres_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D error_guard_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D edge_reduction_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D trend_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D geometry_owner_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D extended_evidence_pass = RigidStudyCriterionStatus3D::NotEvaluated;
    RigidStudyCriterionStatus3D overall_pass = RigidStudyCriterionStatus3D::NotEvaluated;
};

struct NeumannEdgeContinuityEvaluation3D {
    std::vector<NeumannEdgeContinuityDerivedRow3D> rows;
    NeumannEdgeContinuityAcceptance3D acceptance;
    bool all_pass = false;
};

std::vector<int> normalize_neumann_edge_continuity_levels_3d(std::vector<int> levels);

NeumannEdgeContinuityEvaluation3D evaluate_neumann_edge_continuity_study_3d(
    const std::vector<NeumannEdgeContinuityMeasurement3D>& measurements,
    const std::vector<std::string>& case_ids,
    bool require_complete_pilot);

bool neumann_edge_continuity_study_exit_pass_3d(
    const NeumannEdgeContinuityEvaluation3D& evaluation,
    bool require_complete_pilot);

} // namespace kfbim::app3d