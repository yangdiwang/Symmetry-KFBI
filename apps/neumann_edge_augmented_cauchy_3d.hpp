#pragma once

#include "harmonic_polynomial_space_3d.hpp"
#include "native_nurbs_surface_3d.hpp"

#include <Eigen/Sparse>

#include <vector>

namespace kfbim::app3d {

enum class NeumannEdgeCauchyMode3D {
    None,
    NonG1AuxiliaryValues
};

const char* neumann_edge_cauchy_mode_name_3d(
    NeumannEdgeCauchyMode3D mode);

struct NeumannEdgeAuxiliaryOptions3D {
    int degree = 3;
    int value_samples_per_side = 24;
    int normal_samples_per_side = 14;
    int minimum_edge_samples = 4;
    double rank_relative_cutoff = 3.0e-12;
};

struct NeumannEdgeAuxiliarySample3D {
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
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d oriented_tangent = Eigen::Vector3d::Zero();
    Eigen::Vector3d first_normal = Eigen::Vector3d::Zero();
    Eigen::Vector3d second_normal = Eigen::Vector3d::Zero();
    Eigen::Matrix3d frame = Eigen::Matrix3d::Identity();
    double mapped_point_gap = 0.0;
    int first_owner_dof = -1;
    int second_owner_dof = -1;
    std::vector<int> first_value_dofs;
    std::vector<int> first_normal_dofs;
    std::vector<int> second_value_dofs;
    std::vector<int> second_normal_dofs;
    double condition = 0.0;
};

struct NeumannEdgeAuxiliaryDiagnostics3D {
    int expected_non_g1_connections = 0;
    int covered_non_g1_connections = 0;
    int edge_sample_count = 0;
    int unrelated_sample_count = 0;
    int asymmetric_sample_count = 0;
    int rank_deficient_fit_count = 0;
    double mapped_point_gap_max = 0.0;
    double frame_orthogonality_defect_max = 0.0;
    double harmonic_cubic_reproduction_defect_max = 0.0;
    double condition_max = 0.0;
    bool pass = false;
};

struct NeumannEdgeAuxiliaryValueMap3D {
    int surface_size = 0;
    Eigen::SparseMatrix<double, Eigen::RowMajor> value_map;
    Eigen::SparseMatrix<double, Eigen::RowMajor> normal_map;
    std::vector<NeumannEdgeAuxiliarySample3D> samples;
    NeumannEdgeAuxiliaryDiagnostics3D diagnostics;
};

NeumannEdgeAuxiliaryValueMap3D
build_neumann_edge_auxiliary_value_map_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double h,
    const NeumannEdgeAuxiliaryOptions3D& options = {});

Eigen::VectorXd evaluate_neumann_edge_values_3d(
    const NeumannEdgeAuxiliaryValueMap3D& map,
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump);

} // namespace kfbim::app3d
