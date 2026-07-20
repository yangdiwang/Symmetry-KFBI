#pragma once

#include <Eigen/Dense>

#include <functional>
#include <vector>

#include "../geometry/edge_patch_3d.hpp"
#include "../grid/cartesian_grid_3d.hpp"
#include "edge_singular_bspline_fit_3d.hpp"

namespace kfbim {

// Polar frame in the plane normal to a straight edge.  psi=0 points along
// radial_zero. Positive orientation uses tangent x radial_zero.
struct EdgePatchPolarFrame3D {
    Eigen::Vector3d radial_zero = Eigen::Vector3d::UnitX();
    double orientation = 1.0;
    double singular_sign = 1.0;
};

struct EdgePatchLaplacianOptions3D {
    int active_domain_label = 1;
    double geometry_tolerance = 1.0e-12;
    // At the first/last axial grid layer a centered strength difference is
    // unavailable. Vertex patches normally own that region; this fallback
    // keeps the edge operator defined when the regions overlap.
    bool allow_one_sided_strength_difference = true;
};

enum class EdgePatchLaplacianMethod3D {
    OutsidePatch,
    ArtificialExteriorEvent,
    SevenPointStencil,
    AnalyticPhysicalInterface
};

struct EdgePatchLaplacianResult3D {
    EdgePatchLaplacianMethod3D method =
        EdgePatchLaplacianMethod3D::OutsidePatch;
    // Value of -Delta A_edge at the requested grid node.
    double minus_laplacian = 0.0;
    // Contribution added to the regular-field equation
    //   (-Delta_h) w = C_h(mu) + rhs_correction.
    // It is exactly -minus_laplacian for this artificial edge field.
    double rhs_correction = 0.0;
    int physical_interface_neighbors = 0;
    int artificial_boundary_neighbors = 0;
};

// Hybrid straight-edge artificial-field Laplacian.
//
// * A stencil contained in one physical side uses the truncated artificial
//   field and the ordinary anisotropic seven-point stencil. Consequently a
//   crossing of the artificial cylinder is retained as a discrete event.
// * A stencil crossing the physical interface uses the wedge-harmonic model
//   analytically in the transverse plane. The only core volume term is
//   -a_m''(s) rho^lambda cos(lambda psi), with a_m'' evaluated by a centered
//   grid difference. Artificial-cylinder events are still added if the same
//   stencil also crosses the artificial boundary.
// * A center outside the artificial cylinder receives an exterior event when
//   a same-physical-side neighbor is inside: -A_inside/h_axis^2 in -Delta A,
//   hence +A_inside/h_axis^2 in the regular-field RHS correction.
//
// StrengthEvaluator returns the physical a_m(s) multiplying rho^lambda. Its
// s argument is physical arclength measured from patch.start.
class EdgePatchSingularLaplacian3D {
public:
    using StrengthEvaluator = std::function<double(double, int)>;

    EdgePatchSingularLaplacian3D(
        const CartesianGrid3D& grid,
        geometry3d::EdgePatch3D patch,
        EdgePatchPolarFrame3D frame,
        std::vector<double> exponents,
        StrengthEvaluator strength,
        Eigen::VectorXi physical_domain_labels,
        EdgePatchLaplacianOptions3D options = {});

    // Convenience overload backed by the fitted density samples. The fit's
    // [s_min,s_max] coordinate is mapped linearly onto the physical edge.
    // The fit object must outlive this Laplacian object.
    EdgePatchSingularLaplacian3D(
        const CartesianGrid3D& grid,
        geometry3d::EdgePatch3D patch,
        EdgePatchPolarFrame3D frame,
        const EdgeSingularBSplineFit3D& fit,
        Eigen::VectorXd fitted_sample_values,
        Eigen::VectorXi physical_domain_labels,
        EdgePatchLaplacianOptions3D options = {});

    [[nodiscard]] double field_value(int grid_node) const;
    // Singular contribution to the physical jump [u], [du/dn], and
    // [(-Delta+kappa)u] at an arbitrary interface point.  The sign is derived
    // from active_domain_label (interior/nonzero minus exterior/zero).
    [[nodiscard]] bool contains_point(const Eigen::Vector3d& point) const;
    // Physical (not jump-oriented) singular field used by restrict/add-back.
    // physical_field_* is truncated to the artificial cylinder;
    // continuation_field_value keeps the analytic radial continuation for
    // branch conversion of restrict stencil nodes just outside the wall.
    [[nodiscard]] double physical_field_value(
        const Eigen::Vector3d& point) const;
    [[nodiscard]] Eigen::Vector3d physical_field_gradient_at(
        const Eigen::Vector3d& point) const;
    [[nodiscard]] double continuation_field_value(
        const Eigen::Vector3d& point) const;
    [[nodiscard]] double singular_jump_value(
        const Eigen::Vector3d& point) const;
    [[nodiscard]] double singular_jump_normal_derivative(
        const Eigen::Vector3d& point,
        const Eigen::Vector3d& normal) const;
    [[nodiscard]] double singular_jump_operator_value(
        const Eigen::Vector3d& point,
        double kappa = 0.0) const;
    [[nodiscard]] EdgePatchLaplacianResult3D evaluate(int grid_node) const;
    [[nodiscard]] Eigen::VectorXd evaluate_all() const;
    [[nodiscard]] Eigen::VectorXd evaluate_rhs_correction_all() const;

    const geometry3d::EdgePatch3D& patch() const { return patch_; }
    const CartesianGrid3D& grid() const { return grid_; }
    int active_domain_label() const { return options_.active_domain_label; }
    int tangent_axis() const { return tangent_axis_; }

private:
    [[nodiscard]] Eigen::Vector3d point(int grid_node) const;
    [[nodiscard]] double untruncated_field_value(
        const Eigen::Vector3d& point,
        bool allow_one_layer_axial_extension = false) const;
    [[nodiscard]] double strength_with_axial_extension(
        double s,
        int exponent_index) const;
    [[nodiscard]] bool in_geometric_patch(
        const Eigen::Vector3d& point) const;
    [[nodiscard]] double strength_second_difference(
        double s,
        int exponent_index) const;
    [[nodiscard]] double strength_first_difference(
        double s,
        int exponent_index) const;
    [[nodiscard]] double physical_field_operator_value(
        const Eigen::Vector3d& point,
        double kappa) const;
    [[nodiscard]] Eigen::Vector3d physical_field_gradient(
        const Eigen::Vector3d& point) const;
    [[nodiscard]] double jump_orientation() const;
    [[nodiscard]] double analytic_minus_laplacian(
        const Eigen::Vector3d& center,
        const std::array<int, 6>& neighbors,
        int& artificial_boundary_neighbors) const;
    [[nodiscard]] double seven_point_minus_laplacian(
        int center,
        const std::array<int, 6>& neighbors) const;
    [[nodiscard]] EdgePatchLaplacianResult3D exterior_event(
        int center,
        const std::array<int, 6>& neighbors) const;

    const CartesianGrid3D& grid_;
    geometry3d::EdgePatch3D patch_;
    EdgePatchPolarFrame3D frame_;
    Eigen::Vector3d radial_zero_ = Eigen::Vector3d::UnitX();
    Eigen::Vector3d radial_quadrature_ = Eigen::Vector3d::UnitY();
    std::vector<double> exponents_;
    StrengthEvaluator strength_;
    Eigen::VectorXi physical_domain_labels_;
    EdgePatchLaplacianOptions3D options_;
    int tangent_axis_ = -1;
    double axial_spacing_ = 0.0;
};

} // namespace kfbim
