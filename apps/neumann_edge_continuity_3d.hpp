#pragma once

#include "native_nurbs_surface_3d.hpp"

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

} // namespace kfbim::app3d