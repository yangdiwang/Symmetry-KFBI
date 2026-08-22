#pragma once

#include <Eigen/Core>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace kfbim::app3d {

// The analysis atlas is built on a sphere and is then mapped either to an
// ellipsoid or to a smooth radial flower.  In both cases the implicit normal
// is outward and the parameter domains of all patches are [0,1]^2.
enum class CapShape3D {
    Ellipsoid,
    Flower
};

enum class CapPatchKind3D {
    PolarCentral,
    PolarRing,
    MeridianBelt
};

enum class CapSide3D {
    None,
    North,
    South,
    East,
    West
};

enum class CapPatchEdge3D {
    U0,
    U1,
    V0,
    V1
};

struct CapAtlasDensityOptions3D {
    CapShape3D shape = CapShape3D::Ellipsoid;

    // Cubic open-uniform splines require at least four coefficients.  The
    // number of non-empty knot spans is coefficients_per_direction - 3.
    int coefficients_per_direction = 6;

    // Three point Gauss quadrature is the reference choice for both mortar
    // constraints and trace samples.  Orders 1 through 5 are supported.
    int mortar_gauss_order = 3;
    int trace_gauss_order = 3;

    // Reference-sphere atlas.  The central square is [-a,a]^2 and the polar
    // ring terminates on the circle rho=polar_radius.
    double square_half_width = 0.35;
    double polar_radius = 0.72;

    Eigen::Vector3d ellipsoid_axes = Eigen::Vector3d(1.20, 0.90, 0.72);
    double flower_epsilon = 0.16;
    double flower_eta = 0.035;

    // Optional proper rigid map from the untransformed analytic shape to
    // world coordinates:
    //
    //   x_world = rigid_center
    //             + rigid_rotation * (x_reference - rigid_center)
    //             + rigid_translation.
    //
    // The default is exactly the identity map.  Directions, tangents, and
    // normals are transformed by rigid_rotation only.
    Eigen::Matrix3d rigid_rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d rigid_center = Eigen::Vector3d::Zero();
    Eigen::Vector3d rigid_translation = Eigen::Vector3d::Zero();

    // Relative threshold used by the two rank-revealing QR factorizations:
    // first to remove redundant mortar rows, then to select eliminated C0
    // coefficients.
    double rank_tolerance = 1.0e-10;
};

struct CapPatchDescriptor3D {
    int index = -1;
    std::string name;
    CapPatchKind3D kind = CapPatchKind3D::PolarCentral;
    CapSide3D side = CapSide3D::None;
    // +1 for north-cap patches, -1 for south-cap patches, and 0 for belts.
    int hemisphere = 0;
};

struct CapSeamDescriptor3D {
    int patch_a = -1;
    CapPatchEdge3D edge_a = CapPatchEdge3D::U0;
    int patch_b = -1;
    CapPatchEdge3D edge_b = CapPatchEdge3D::U0;
    // If true, edge_b is evaluated at 1-t while edge_a is evaluated at t.
    bool reversed = false;
    std::string label;
};

struct CapSurfaceEvaluation3D {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, 2> tangents =
        Eigen::Matrix<double, 3, 2>::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double area_element = 0.0;
};

struct CapPatchLocation3D {
    int patch = -1;
    double u = 0.0;
    double v = 0.0;
    bool valid = false;
};

// A cubic tensor-product value row has at most 4 x 4 active functions.  The
// entries refer to merged C0 coefficients and can therefore be cached by
// matrix-free crossing/trace plans and reused throughout GMRES.
struct CapC0BasisStencil3D {
    std::array<int, 16> indices{};
    std::array<double, 16> weights{};
    int count = 0;

    double dot(const Eigen::Ref<const Eigen::VectorXd>& coefficients) const
    {
        double value = 0.0;
        for (int i = 0; i < count; ++i)
            value += weights[static_cast<std::size_t>(i)]
                     * coefficients[indices[static_cast<std::size_t>(i)]];
        return value;
    }
};

struct CapGaussTracePoint3D {
    int patch = -1;
    int element_u = -1;
    int element_v = -1;
    double u = 0.0;
    double v = 0.0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    // Tensor Gauss weight multiplied by |X_u x X_v|.
    double surface_weight = 0.0;
};

// Cubic tensor B-splines are first glued strongly at patch edges (C0), then
// constrained by a physical co-normal derivative mortar condition.  The
// actual unknown of this class is a vector of independent/reduced
// coefficients a, with merged C0 coefficients c = reduction_matrix() * a.
class CapAtlasDensitySpace3D {
public:
    explicit CapAtlasDensitySpace3D(
        CapAtlasDensityOptions3D options = {});
    ~CapAtlasDensitySpace3D();

    CapAtlasDensitySpace3D(CapAtlasDensitySpace3D&&) noexcept;
    CapAtlasDensitySpace3D& operator=(CapAtlasDensitySpace3D&&) noexcept;
    CapAtlasDensitySpace3D(const CapAtlasDensitySpace3D&) = delete;
    CapAtlasDensitySpace3D& operator=(const CapAtlasDensitySpace3D&) = delete;

    const CapAtlasDensityOptions3D& options() const;

    int patch_count() const;
    int coefficients_per_direction() const;
    int elements_per_direction() const;
    int local_coefficient_count() const;
    int c0_coefficient_count() const;
    int c1_constraint_count() const;
    int c1_constraint_rank() const;
    int reduced_coefficient_count() const;

    const std::vector<CapPatchDescriptor3D>& patches() const;
    const std::vector<CapSeamDescriptor3D>& seams() const;
    const std::vector<int>& local_to_c0() const;

    CapSurfaceEvaluation3D geometry(int patch, double u, double v) const;

    // Shape queries use the same parameters as the atlas.  They are exposed
    // so Cartesian-grid crossing plans cannot accidentally use a geometry
    // inconsistent with the density basis.
    double level_set(const Eigen::Vector3d& point) const;
    Eigen::Vector3d level_set_gradient(const Eigen::Vector3d& point) const;
    Eigen::Vector3d outward_normal(const Eigen::Vector3d& point) const;
    bool inside(const Eigen::Vector3d& point) const;
    // Map a reference unit-sphere direction (normalization is applied) to the
    // selected physical surface.
    Eigen::Vector3d surface_point_from_direction(
        const Eigen::Vector3d& reference_direction) const;

    // Locate by reference direction.  The input may be on the surface or a
    // nearby off-surface point (as needed by local crossing sample plans).
    CapPatchLocation3D locate(const Eigen::Vector3d& point) const;

    // Rows act directly on coefficient vectors, e.g.
    //     value = reduced_basis_row(...).dot(reduced_coefficients).
    Eigen::RowVectorXd c0_basis_row(int patch, double u, double v) const;
    CapC0BasisStencil3D c0_basis_stencil(int patch,
                                         double u,
                                         double v) const;
    Eigen::RowVectorXd reduced_basis_row(int patch,
                                         double u,
                                         double v) const;
    Eigen::RowVectorXd c0_basis_derivative_row(int patch,
                                               double u,
                                               double v,
                                               int derivative_u,
                                               int derivative_v) const;
    Eigen::RowVectorXd reduced_basis_derivative_row(int patch,
                                                    double u,
                                                    double v,
                                                    int derivative_u,
                                                    int derivative_v) const;

    Eigen::VectorXd expand_reduced(
        const Eigen::Ref<const Eigen::VectorXd>& reduced_coefficients) const;
    // Evaluate a density whose coefficients are already in the merged C0
    // space.  Matrix-free KFBI plans use this after one expand_reduced()
    // call, so each local sample touches only the at-most 16 active tensor
    // B-splines instead of forming a dense reduced row.
    double evaluate_c0(
        int patch,
        double u,
        double v,
        const Eigen::Ref<const Eigen::VectorXd>& c0_coefficients) const;
    double evaluate(int patch,
                    double u,
                    double v,
                    const Eigen::Ref<const Eigen::VectorXd>&
                        reduced_coefficients) const;

    const Eigen::MatrixXd& reduction_matrix() const;
    const Eigen::MatrixXd& c1_constraint_matrix() const;
    const Eigen::MatrixXd& independent_c1_constraint_matrix() const;

    const std::vector<CapGaussTracePoint3D>& trace_points() const;
    const Eigen::RowVectorXd& mass_row() const;
    double surface_area() const;

    // Construction diagnostics.  The first quantity should be near machine
    // precision; the latter two inspect a supplied reduced density.
    double reduction_constraint_residual() const;
    double max_c0_seam_jump(
        const Eigen::Ref<const Eigen::VectorXd>& reduced_coefficients,
        int samples_per_seam = 17) const;
    double max_physical_conormal_jump(
        const Eigen::Ref<const Eigen::VectorXd>& reduced_coefficients,
        int samples_per_seam = 17) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfbim::app3d
