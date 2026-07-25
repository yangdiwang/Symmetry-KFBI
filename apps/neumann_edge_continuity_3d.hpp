#pragma once

#include "native_nurbs_surface_3d.hpp"
#include "../src/operators/i_kfbi_operator.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

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

} // namespace kfbim::app3d