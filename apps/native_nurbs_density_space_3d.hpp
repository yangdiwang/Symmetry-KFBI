#pragma once

#include "native_nurbs_surface_3d.hpp"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace kfbim::app3d {

// A value trace is continuous across a geometric feature but is differentiated
// only across G1 seams.  A normal trace is intentionally broken at a feature:
// even for a smooth ambient solution, grad(u).n changes when n jumps.
enum class NativeDensityField3D {
    ValueTrace,
    NormalTrace
};

// Legacy keeps the original global dense nullspace construction.  BaseOnly
// stops after the exact full-edge UnionFind merge; all partial-edge C0 and
// smooth-C1 equations are then supplied by the sparse topology layer.  The
// TopologyBase spelling documents that intended use while remaining an alias
// of the explicit BaseOnly mode.
enum class NativeDensityReductionBackend3D {
    Legacy,
    BaseOnly,
    TopologyBase = BaseOnly
};

enum class NativeDensitySeamCoupling3D {
    Broken,
    StrongC0,
    StrongC0WeakC1
};

struct NativeNurbsDensityOptions3D {
    NativeDensityField3D field = NativeDensityField3D::ValueTrace;
    NativeDensityReductionBackend3D reduction_backend =
        NativeDensityReductionBackend3D::Legacy;
    int coefficients_per_direction = 6;
    // At least degree+1 points are required on a one-span cubic seam.  With
    // only three points the ncoef=4 cylinder leaves one co-normal jump mode
    // invisible to the weak-C1 mortar matrix.
    int mortar_gauss_order = 4;
    int trace_gauss_order = 3;
    double rank_tolerance = 1.0e-10;
};

struct NativeDensitySeamInfo3D {
    int connection = -1;
    NativeDensitySeamCoupling3D coupling =
        NativeDensitySeamCoupling3D::Broken;
    bool geometry_g1 = false;
    bool full_edge_pair = false;
    bool strong_c0_merged = false;
    int partial_c0_first_row = 0;
    int partial_c0_row_count = 0;
    int weak_c1_first_row = 0;
    int weak_c1_row_count = 0;
    std::string label;
};

// A cubic tensor-product row has at most 4 x 4 active functions.  Indices are
// in the coefficient space obtained after exact full-edge UnionFind merges,
// but before partial-edge C0 and weak-C1 reductions.
struct NativeDensityC0Stencil3D {
    std::array<int, 16> indices{};
    std::array<double, 16> weights{};
    int count = 0;

    double dot(const Eigen::Ref<const Eigen::VectorXd>& coefficients) const;
};

struct NativeDensitySurfaceEvaluation3D {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, 2> tangents =
        Eigen::Matrix<double, 3, 2>::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double area_element = 0.0;
};

struct NativeDensityGaussPoint3D {
    int patch = -1;
    int element_u = -1;
    int element_v = -1;
    double u = 0.0;
    double v = 0.0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double surface_weight = 0.0;
};

// Optional affine compatibility data for a sharp feature edge.  The legacy
// and solved-conormal operators order two side equations per scalar mortar
// test function.  A direct ambient-gradient operator instead stores two
// transverse world-gradient components per test function.  The edge
// equations are weak line moments and do not include endpoint point values
// explicitly.
struct NativeFeatureEdgeJumpJetInfo3D {
    int connection = -1;
    int first_row = 0;
    int row_count = 0;
    int test_function_count = 0;
    bool full_edge_pair = false;
    bool reversed = false;
    double minimum_abs_dihedral_sine = 1.0;
    double maximum_abs_dihedral_sine = 0.0;
    double maximum_point_mismatch = 0.0;
    std::string label;
};

struct NativeFeatureEdgeJumpJetOptions3D {
    // The feature constraint is deliberately opt-in.  A general solution on
    // a polyhedron can have an edge singularity and need not possess the
    // single ambient gradient assumed by these equations.
    bool enabled = true;
    int mortar_gauss_order = 4;
    double minimum_abs_dihedral_sine = 1.0e-8;
    bool normalize_rows = true;
};

// Uniform cubic patch spaces are coupled in two stages:
//
//   raw patch coefficients --(full/full UnionFind)--> merged coefficients
//       -- R0: exact partial-edge C0 nullspace --> continuous coefficients
//       -- R1: smooth-seam weak-C1 nullspace --> reduced coefficients.
//
// reduction_matrix() is R0*R1 and maps reduced coefficients to the merged
// local space.  The sparse trace design therefore remains at most 16 entries
// per row and can be composed with this dense reduction without constructing
// a dense sample-by-reduced matrix.
class NativeNurbsDensitySpace3D {
public:
    NativeNurbsDensitySpace3D(
        NativeNurbsSurface3D surface,
        NativeNurbsDensityOptions3D options = {});
    NativeNurbsDensitySpace3D(
        NativeNurbsSurface3D surface,
        int coefficients_per_direction,
        NativeDensityField3D field);
    ~NativeNurbsDensitySpace3D();

    NativeNurbsDensitySpace3D(NativeNurbsDensitySpace3D&&) noexcept;
    NativeNurbsDensitySpace3D& operator=(
        NativeNurbsDensitySpace3D&&) noexcept;
    NativeNurbsDensitySpace3D(const NativeNurbsDensitySpace3D&) = delete;
    NativeNurbsDensitySpace3D& operator=(
        const NativeNurbsDensitySpace3D&) = delete;

    const NativeNurbsSurface3D& surface() const;
    const NativeNurbsDensityOptions3D& options() const;

    int patch_count() const;
    int coefficients_per_direction() const;
    int elements_per_direction() const;
    int raw_coefficient_count() const;
    int c0_coefficient_count() const;
    int continuous_coefficient_count() const;
    int reduced_coefficient_count() const;
    bool uses_identity_reduction() const noexcept;
    int partial_c0_constraint_count() const;
    int partial_c0_constraint_rank() const;
    int weak_c1_constraint_count() const;
    int weak_c1_constraint_rank() const;

    int broken_seam_count() const;
    int strong_c0_seam_count() const;
    int weak_c1_seam_count() const;
    int merged_c0_seam_count() const;
    int collocated_c0_seam_count() const;
    const std::vector<NativeDensitySeamInfo3D>& seams() const;

    NativeDensitySurfaceEvaluation3D geometry(
        int patch, double u, double v) const;
    NativeDensityC0Stencil3D c0_basis_stencil(
        int patch, double u, double v) const;
    Eigen::RowVectorXd c0_basis_row(
        int patch, double u, double v) const;
    // Exact tensor-product B-spline parameter derivatives in the Base/C0
    // coefficient space.  derivative_u + derivative_v may be at most two;
    // in particular these expose the six rows needed by direct coefficient
    // Cauchy closure without refitting the unknown density from samples.
    NativeDensityC0Stencil3D c0_parameter_derivative_stencil(
        int patch,
        double u,
        double v,
        int derivative_u,
        int derivative_v) const;
    // Share the two one-dimensional basis evaluations across all six P2
    // parameter-jet rows, ordered as value, u, v, uu, uv, vv.  Like the
    // single-row API, indices refer to this density space's C0 coordinates.
    std::array<NativeDensityC0Stencil3D, 6> c0_parameter_jet_stencils(
        int patch, double u, double v) const;
    Eigen::RowVectorXd c0_parameter_derivative_row(
        int patch,
        double u,
        double v,
        int derivative_u,
        int derivative_v) const;
    Eigen::RowVectorXd reduced_basis_row(
        int patch, double u, double v) const;
    Eigen::RowVectorXd reduced_parameter_derivative_row(
        int patch,
        double u,
        double v,
        int derivative_u,
        int derivative_v) const;
    Eigen::RowVectorXd c0_physical_directional_derivative_row(
        int patch,
        double u,
        double v,
        const Eigen::Vector3d& tangent_direction) const;
    NativeDensityC0Stencil3D c0_physical_directional_derivative_stencil(
        int patch,
        double u,
        double v,
        const Eigen::Vector3d& tangent_direction) const;
    Eigen::RowVectorXd reduced_physical_directional_derivative_row(
        int patch,
        double u,
        double v,
        const Eigen::Vector3d& tangent_direction) const;

    Eigen::VectorXd expand_reduced(
        const Eigen::Ref<const Eigen::VectorXd>& coefficients) const;
    Eigen::VectorXd expand_raw(
        const Eigen::Ref<const Eigen::VectorXd>& coefficients) const;
    double evaluate_c0(
        int patch,
        double u,
        double v,
        const Eigen::Ref<const Eigen::VectorXd>& c0_coefficients) const;
    double evaluate(
        int patch,
        double u,
        double v,
        const Eigen::Ref<const Eigen::VectorXd>& coefficients) const;

    const std::vector<int>& local_to_c0() const;
    const Eigen::MatrixXd& c0_reduction_matrix() const;
    const Eigen::MatrixXd& c1_reduction_matrix() const;
    const Eigen::MatrixXd& reduction_matrix() const;
    const Eigen::MatrixXd& partial_c0_constraint_matrix() const;
    const Eigen::MatrixXd& weak_c1_constraint_matrix() const;
    const Eigen::MatrixXd& continuous_weak_c1_constraint_matrix() const;

    const std::vector<NativeDensityGaussPoint3D>& trace_points() const;
    const Eigen::SparseMatrix<double>& trace_c0_design() const;
    const Eigen::VectorXd& trace_weights() const;
    const Eigen::RowVectorXd& c0_mass_row() const;
    const Eigen::RowVectorXd& mass_row() const;
    double surface_area() const;

    double partial_c0_constraint_residual() const;
    double weak_c1_constraint_residual() const;
    double reduction_constraint_residual() const;
    double max_c0_seam_jump(
        const Eigen::Ref<const Eigen::VectorXd>& coefficients,
        int samples_per_segment = 17) const;
    double max_smooth_conormal_jump(
        const Eigen::Ref<const Eigen::VectorXd>& coefficients,
        int samples_per_segment = 17) const;
    double constant_reproduction_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Geometry-only sharp-edge jet coupling between the already reduced value and
// broken-normal density spaces.  If av and an are their reduced coefficient
// vectors, respectively, the affine edge condition is
//
//             value_matrix() * av = normal_target_matrix() * an.
//
// The right-hand side is therefore obtained from known Neumann data after it
// has been represented in the NormalTrace space.  This object does not impose
// the constraint on NativeNurbsDensitySpace3D itself, so callers can enable it
// per solve and construct an affine lifting/nullspace without changing the
// baseline homogeneous density space.
class NativeFeatureEdgeJumpJetConstraints3D {
public:
    NativeFeatureEdgeJumpJetConstraints3D(
        const NativeNurbsDensitySpace3D& value_space,
        const NativeNurbsDensitySpace3D& normal_space,
        NativeFeatureEdgeJumpJetOptions3D options = {});

    const NativeFeatureEdgeJumpJetOptions3D& options() const noexcept;
    int feature_edge_count() const noexcept;
    int constraint_count() const noexcept;
    const std::vector<NativeFeatureEdgeJumpJetInfo3D>& edges() const noexcept;
    const Eigen::MatrixXd& value_matrix() const noexcept;
    const Eigen::MatrixXd& normal_target_matrix() const noexcept;
    const Eigen::VectorXd& row_scalings() const noexcept;

    Eigen::VectorXd normal_target(
        const Eigen::Ref<const Eigen::VectorXd>& normal_coefficients) const;
    Eigen::VectorXd residual(
        const Eigen::Ref<const Eigen::VectorXd>& value_coefficients,
        const Eigen::Ref<const Eigen::VectorXd>& normal_coefficients) const;

    double maximum_point_mismatch() const noexcept;
    double minimum_abs_dihedral_sine() const noexcept;
    double constant_value_residual() const noexcept;

private:
    NativeFeatureEdgeJumpJetOptions3D options_;
    std::vector<NativeFeatureEdgeJumpJetInfo3D> edges_;
    Eigen::MatrixXd value_matrix_;
    Eigen::MatrixXd normal_target_matrix_;
    Eigen::VectorXd row_scalings_;
    double maximum_point_mismatch_ = 0.0;
    double minimum_abs_dihedral_sine_ = 1.0;
    double constant_value_residual_ = 0.0;
};

} // namespace kfbim::app3d
