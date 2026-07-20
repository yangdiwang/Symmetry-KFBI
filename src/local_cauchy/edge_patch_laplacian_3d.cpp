#include "edge_patch_laplacian_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace kfbim {

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

bool finite_positive(double value)
{
    return std::isfinite(value) && value > 0.0;
}

Eigen::Vector3d normalize_or_throw(Eigen::Vector3d value,
                                   const char* message)
{
    const double norm = value.norm();
    if (!finite_positive(norm))
        throw std::invalid_argument(message);
    return value / norm;
}

int aligned_axis(const Eigen::Vector3d& tangent, double tolerance)
{
    int axis = 0;
    tangent.cwiseAbs().maxCoeff(&axis);
    if (std::abs(std::abs(tangent[axis]) - 1.0) > tolerance)
        return -1;
    for (int d = 0; d < 3; ++d) {
        if (d != axis && std::abs(tangent[d]) > tolerance)
            return -1;
    }
    return axis;
}

} // namespace

EdgePatchSingularLaplacian3D::EdgePatchSingularLaplacian3D(
    const CartesianGrid3D& grid,
    geometry3d::EdgePatch3D patch,
    EdgePatchPolarFrame3D frame,
    std::vector<double> exponents,
    StrengthEvaluator strength,
    Eigen::VectorXi physical_domain_labels,
    EdgePatchLaplacianOptions3D options)
    : grid_(grid)
    , patch_(std::move(patch))
    , frame_(std::move(frame))
    , exponents_(std::move(exponents))
    , strength_(std::move(strength))
    , physical_domain_labels_(std::move(physical_domain_labels))
    , options_(std::move(options))
{
    if (!finite_positive(patch_.length) || !finite_positive(patch_.radius))
        throw std::invalid_argument(
            "EdgePatchSingularLaplacian3D requires a valid edge patch");
    patch_.tangent = normalize_or_throw(
        patch_.tangent, "edge patch tangent must be nonzero");
    if (!std::isfinite(options_.geometry_tolerance)
        || options_.geometry_tolerance < 0.0) {
        throw std::invalid_argument(
            "edge patch Laplacian geometry tolerance must be nonnegative");
    }
    tangent_axis_ = aligned_axis(
        patch_.tangent, std::max(1.0e-12, options_.geometry_tolerance));
    if (tangent_axis_ < 0) {
        throw std::invalid_argument(
            "edge patch Laplacian currently requires a Cartesian-axis-aligned edge");
    }
    const auto spacing = grid_.spacing();
    axial_spacing_ = spacing[static_cast<std::size_t>(tangent_axis_)];
    for (double h : spacing) {
        if (!finite_positive(h))
            throw std::invalid_argument("grid spacing must be positive");
    }

    radial_zero_ = normalize_or_throw(
        frame_.radial_zero, "edge patch radial_zero must be nonzero");
    if (std::abs(radial_zero_.dot(patch_.tangent))
        > std::max(1.0e-10, options_.geometry_tolerance)) {
        throw std::invalid_argument(
            "edge patch radial_zero must be normal to the edge tangent");
    }
    if (!std::isfinite(frame_.orientation) || frame_.orientation == 0.0
        || !std::isfinite(frame_.singular_sign)) {
        throw std::invalid_argument("edge patch polar frame is invalid");
    }
    radial_quadrature_ = normalize_or_throw(
        (frame_.orientation > 0.0 ? 1.0 : -1.0)
            * patch_.tangent.cross(radial_zero_),
        "edge patch polar frame is degenerate");

    if (exponents_.empty() || !strength_)
        throw std::invalid_argument(
            "edge patch Laplacian needs exponents and a strength evaluator");
    for (double exponent : exponents_) {
        if (!finite_positive(exponent))
            throw std::invalid_argument(
                "edge patch Laplacian exponents must be positive");
    }
    if (physical_domain_labels_.size() != grid_.num_dofs())
        throw std::invalid_argument(
            "edge patch physical-domain labels must cover every grid node");
}

EdgePatchSingularLaplacian3D::EdgePatchSingularLaplacian3D(
    const CartesianGrid3D& grid,
    geometry3d::EdgePatch3D patch,
    EdgePatchPolarFrame3D frame,
    const EdgeSingularBSplineFit3D& fit,
    Eigen::VectorXd fitted_sample_values,
    Eigen::VectorXi physical_domain_labels,
    EdgePatchLaplacianOptions3D options)
    : EdgePatchSingularLaplacian3D(
          grid,
          patch,
          std::move(frame),
          fit.options().exponents,
          [&fit,
           coefficients = fit.singular_coefficients(fitted_sample_values),
           edge_length = patch.length,
           s_min = fit.options().s_min,
           s_max = fit.options().s_max](double physical_s, int exponent_index) {
              const double fit_s = s_min
                  + physical_s / edge_length * (s_max - s_min);
              return fit.singular_strength_from_coefficients(
                  coefficients, fit_s, exponent_index);
          },
          std::move(physical_domain_labels),
          std::move(options))
{
}

Eigen::Vector3d EdgePatchSingularLaplacian3D::point(int grid_node) const
{
    if (grid_node < 0 || grid_node >= grid_.num_dofs())
        throw std::out_of_range("edge patch grid-node index is out of range");
    const auto coordinate = grid_.coord(grid_node);
    return {coordinate[0], coordinate[1], coordinate[2]};
}

bool EdgePatchSingularLaplacian3D::in_geometric_patch(
    const Eigen::Vector3d& query) const
{
    return geometry3d::edge_patch_contains_point_3d(
        patch_, query, options_.geometry_tolerance);
}

double EdgePatchSingularLaplacian3D::untruncated_field_value(
    const Eigen::Vector3d& query,
    bool allow_one_layer_axial_extension) const
{
    const geometry3d::EdgePatchCoordinates3D coordinates =
        geometry3d::edge_patch_coordinates_3d(patch_, query);
    const double tolerance = options_.geometry_tolerance;
    const double axial_margin = allow_one_layer_axial_extension
        ? axial_spacing_ + tolerance : tolerance;
    if (coordinates.s < -axial_margin
        || coordinates.s > patch_.length + axial_margin
        || coordinates.rho <= tolerance) {
        return 0.0;
    }
    const Eigen::Vector3d radial = query - coordinates.axis_point;
    double psi = std::atan2(radial.dot(radial_quadrature_),
                            radial.dot(radial_zero_));
    if (psi < 0.0)
        psi += kTwoPi;

    double value = 0.0;
    for (int mode = 0; mode < static_cast<int>(exponents_.size()); ++mode) {
        const double exponent = exponents_[static_cast<std::size_t>(mode)];
        const double coefficient = allow_one_layer_axial_extension
            ? strength_with_axial_extension(coordinates.s, mode)
            : strength_(std::max(0.0,
                                 std::min(patch_.length, coordinates.s)),
                        mode);
        value += coefficient
               * std::pow(coordinates.rho, exponent)
               * std::cos(exponent * psi);
    }
    return frame_.singular_sign * value;
}

double EdgePatchSingularLaplacian3D::strength_with_axial_extension(
    double s,
    int exponent_index) const
{
    const double tolerance = options_.geometry_tolerance;
    if (s >= -tolerance && s <= patch_.length + tolerance) {
        return strength_(std::max(0.0, std::min(patch_.length, s)),
                         exponent_index);
    }
    const double h = axial_spacing_;
    if (patch_.length < 2.0 * h - tolerance) {
        throw std::runtime_error(
            "edge is too short for axial artificial-cap extension");
    }
    if (s >= -h - tolerance && s < 0.0) {
        // Quadratic extrapolation through s=0,h,2h.
        const double t = s / h;
        const double f0 = strength_(0.0, exponent_index);
        const double f1 = strength_(h, exponent_index);
        const double f2 = strength_(2.0 * h, exponent_index);
        return 0.5 * (t - 1.0) * (t - 2.0) * f0
             - t * (t - 2.0) * f1
             + 0.5 * t * (t - 1.0) * f2;
    }
    if (s <= patch_.length + h + tolerance && s > patch_.length) {
        // Symmetric quadratic extrapolation at the right artificial cap.
        const double t = (patch_.length - s) / h;
        const double f0 = strength_(patch_.length, exponent_index);
        const double f1 = strength_(patch_.length - h, exponent_index);
        const double f2 = strength_(patch_.length - 2.0 * h, exponent_index);
        return 0.5 * (t - 1.0) * (t - 2.0) * f0
             - t * (t - 2.0) * f1
             + 0.5 * t * (t - 1.0) * f2;
    }
    throw std::runtime_error(
        "artificial-cap query exceeds the one-layer axial extension");
}

double EdgePatchSingularLaplacian3D::field_value(int grid_node) const
{
    if (grid_node < 0 || grid_node >= grid_.num_dofs())
        throw std::out_of_range("edge patch grid-node index is out of range");
    if (physical_domain_labels_[grid_node] != options_.active_domain_label)
        return 0.0;
    const Eigen::Vector3d query = point(grid_node);
    if (!in_geometric_patch(query))
        return 0.0;
    return untruncated_field_value(query);
}

bool EdgePatchSingularLaplacian3D::contains_point(
    const Eigen::Vector3d& query) const
{
    return in_geometric_patch(query);
}

double EdgePatchSingularLaplacian3D::physical_field_value(
    const Eigen::Vector3d& query) const
{
    return in_geometric_patch(query)
        ? untruncated_field_value(query)
        : 0.0;
}

Eigen::Vector3d EdgePatchSingularLaplacian3D::physical_field_gradient_at(
    const Eigen::Vector3d& query) const
{
    return physical_field_gradient(query);
}

double EdgePatchSingularLaplacian3D::continuation_field_value(
    const Eigen::Vector3d& query) const
{
    // Deliberately do not test rho <= patch.radius here.  Restrict needs the
    // same analytic branch on stencil nodes immediately across the artificial
    // radial wall.  Axially, the edge/vertex split owns points beyond the two
    // end sections, so the ordinary untruncated evaluator returns zero there.
    return untruncated_field_value(query);
}

double EdgePatchSingularLaplacian3D::jump_orientation() const
{
    // All nonzero component labels are on the interior side used by the
    // correction-polynomial convention [u] = u_inside - u_outside.
    return options_.active_domain_label == 0 ? -1.0 : 1.0;
}

double EdgePatchSingularLaplacian3D::singular_jump_value(
    const Eigen::Vector3d& query) const
{
    if (!in_geometric_patch(query))
        return 0.0;
    return jump_orientation() * untruncated_field_value(query);
}

double EdgePatchSingularLaplacian3D::strength_first_difference(
    double s,
    int exponent_index) const
{
    const double h = axial_spacing_;
    const double tolerance = options_.geometry_tolerance;
    s = std::max(0.0, std::min(patch_.length, s));
    if (s - h >= -tolerance && s + h <= patch_.length + tolerance) {
        return (strength_(std::min(patch_.length, s + h), exponent_index)
                - strength_(std::max(0.0, s - h), exponent_index))
             / (2.0 * h);
    }
    if (s + 2.0 * h <= patch_.length + tolerance) {
        return (-3.0 * strength_(s, exponent_index)
                + 4.0 * strength_(s + h, exponent_index)
                - strength_(s + 2.0 * h, exponent_index))
             / (2.0 * h);
    }
    if (s - 2.0 * h >= -tolerance) {
        return (3.0 * strength_(s, exponent_index)
                - 4.0 * strength_(s - h, exponent_index)
                + strength_(s - 2.0 * h, exponent_index))
             / (2.0 * h);
    }
    throw std::runtime_error(
        "edge is too short for a centered or one-sided strength derivative");
}

Eigen::Vector3d EdgePatchSingularLaplacian3D::physical_field_gradient(
    const Eigen::Vector3d& query) const
{
    if (!in_geometric_patch(query))
        return Eigen::Vector3d::Zero();

    const geometry3d::EdgePatchCoordinates3D coordinates =
        geometry3d::edge_patch_coordinates_3d(patch_, query);
    const Eigen::Vector3d radial = query - coordinates.axis_point;
    double psi = std::atan2(radial.dot(radial_quadrature_),
                            radial.dot(radial_zero_));
    if (psi < 0.0)
        psi += kTwoPi;
    const Eigen::Vector3d e_r =
        std::cos(psi) * radial_zero_ + std::sin(psi) * radial_quadrature_;
    const Eigen::Vector3d e_psi =
        -std::sin(psi) * radial_zero_ + std::cos(psi) * radial_quadrature_;
    const double s = std::max(0.0,
                              std::min(patch_.length, coordinates.s));

    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
    for (int mode = 0; mode < static_cast<int>(exponents_.size()); ++mode) {
        const double exponent = exponents_[static_cast<std::size_t>(mode)];
        const double coefficient = strength_(s, mode);
        const double coefficient_s = strength_first_difference(s, mode);
        const double rho_power = std::pow(coordinates.rho, exponent);
        const double radial_factor =
            exponent * std::pow(coordinates.rho, exponent - 1.0);
        const double cos_mode = std::cos(exponent * psi);
        const double sin_mode = std::sin(exponent * psi);
        gradient += coefficient_s * rho_power * cos_mode * patch_.tangent;
        gradient += coefficient * radial_factor
                  * (cos_mode * e_r - sin_mode * e_psi);
    }
    return frame_.singular_sign * gradient;
}

double EdgePatchSingularLaplacian3D::singular_jump_normal_derivative(
    const Eigen::Vector3d& query,
    const Eigen::Vector3d& normal) const
{
    if (!in_geometric_patch(query))
        return 0.0;
    return jump_orientation() * physical_field_gradient(query).dot(normal);
}

double EdgePatchSingularLaplacian3D::physical_field_operator_value(
    const Eigen::Vector3d& query,
    double kappa) const
{
    if (!in_geometric_patch(query))
        return 0.0;
    const geometry3d::EdgePatchCoordinates3D coordinates =
        geometry3d::edge_patch_coordinates_3d(patch_, query);
    const Eigen::Vector3d radial = query - coordinates.axis_point;
    double psi = std::atan2(radial.dot(radial_quadrature_),
                            radial.dot(radial_zero_));
    if (psi < 0.0)
        psi += kTwoPi;
    const double s = std::max(0.0,
                              std::min(patch_.length, coordinates.s));

    double value = 0.0;
    for (int mode = 0; mode < static_cast<int>(exponents_.size()); ++mode) {
        const double exponent = exponents_[static_cast<std::size_t>(mode)];
        const double basis = std::pow(coordinates.rho, exponent)
                           * std::cos(exponent * psi);
        value += (-strength_second_difference(s, mode)
                  + kappa * strength_(s, mode))
               * basis;
    }
    return frame_.singular_sign * value;
}

double EdgePatchSingularLaplacian3D::singular_jump_operator_value(
    const Eigen::Vector3d& query,
    double kappa) const
{
    if (!std::isfinite(kappa))
        throw std::invalid_argument("edge singular kappa must be finite");
    return jump_orientation()
         * physical_field_operator_value(query, kappa);
}

double EdgePatchSingularLaplacian3D::strength_second_difference(
    double s,
    int exponent_index) const
{
    const double h = axial_spacing_;
    const double tolerance = options_.geometry_tolerance;
    if (s - h >= -tolerance && s + h <= patch_.length + tolerance) {
        return (strength_(std::max(0.0, s - h), exponent_index)
                - 2.0 * strength_(s, exponent_index)
                + strength_(std::min(patch_.length, s + h), exponent_index))
             / (h * h);
    }
    if (!options_.allow_one_sided_strength_difference) {
        throw std::runtime_error(
            "centered edge-strength difference reaches an edge endpoint");
    }
    if (s + 3.0 * h <= patch_.length + tolerance) {
        return (2.0 * strength_(s, exponent_index)
                - 5.0 * strength_(s + h, exponent_index)
                + 4.0 * strength_(s + 2.0 * h, exponent_index)
                - strength_(s + 3.0 * h, exponent_index))
             / (h * h);
    }
    if (s - 3.0 * h >= -tolerance) {
        return (2.0 * strength_(s, exponent_index)
                - 5.0 * strength_(s - h, exponent_index)
                + 4.0 * strength_(s - 2.0 * h, exponent_index)
                - strength_(s - 3.0 * h, exponent_index))
             / (h * h);
    }
    throw std::runtime_error(
        "edge is too short for a three-point or one-sided strength difference");
}

double EdgePatchSingularLaplacian3D::analytic_minus_laplacian(
    const Eigen::Vector3d& center,
    const std::array<int, 6>& neighbors,
    int& artificial_boundary_neighbors) const
{
    const geometry3d::EdgePatchCoordinates3D coordinates =
        geometry3d::edge_patch_coordinates_3d(patch_, center);
    const Eigen::Vector3d radial = center - coordinates.axis_point;
    double psi = std::atan2(radial.dot(radial_quadrature_),
                            radial.dot(radial_zero_));
    if (psi < 0.0)
        psi += kTwoPi;

    double result = 0.0;
    for (int mode = 0; mode < static_cast<int>(exponents_.size()); ++mode) {
        const double exponent = exponents_[static_cast<std::size_t>(mode)];
        const double second = strength_second_difference(coordinates.s, mode);
        result -= frame_.singular_sign * second
                * std::pow(coordinates.rho, exponent)
                * std::cos(exponent * psi);
    }

    // The analytic core cancels the transverse wedge Laplacian. If the same
    // stencil also leaves the artificial cylinder, restore the sharp-cutoff
    // discrete event relative to that untruncated analytic core.
    const auto spacing = grid_.spacing();
    for (int direction = 0; direction < 6; ++direction) {
        const Eigen::Vector3d neighbor = point(neighbors[direction]);
        if (in_geometric_patch(neighbor))
            continue;
        const geometry3d::EdgePatchCoordinates3D neighbor_coordinates =
            geometry3d::edge_patch_coordinates_3d(patch_, neighbor);
        const int axis = direction / 2;
        const double h = spacing[static_cast<std::size_t>(axis)];
        result += untruncated_field_value(neighbor, true) / (h * h);
        ++artificial_boundary_neighbors;
    }
    return result;
}

double EdgePatchSingularLaplacian3D::seven_point_minus_laplacian(
    int center,
    const std::array<int, 6>& neighbors) const
{
    const auto spacing = grid_.spacing();
    const double center_value = field_value(center);
    double result = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        const double h = spacing[static_cast<std::size_t>(axis)];
        result += (2.0 * center_value
                   - field_value(neighbors[2 * axis])
                   - field_value(neighbors[2 * axis + 1]))
                / (h * h);
    }
    return result;
}

EdgePatchLaplacianResult3D EdgePatchSingularLaplacian3D::exterior_event(
    int center,
    const std::array<int, 6>& neighbors) const
{
    EdgePatchLaplacianResult3D result;
    if (physical_domain_labels_[center] != options_.active_domain_label)
        return result;

    const auto spacing = grid_.spacing();
    for (int direction = 0; direction < 6; ++direction) {
        const int neighbor = neighbors[direction];
        if (neighbor < 0)
            continue;
        // Artificial-cylinder events are formed only within one physical
        // side. A neighbor across the physical interface belongs to the EIP
        // jump/analytic treatment instead.
        if (physical_domain_labels_[neighbor]
            != physical_domain_labels_[center]) {
            continue;
        }
        if (!in_geometric_patch(point(neighbor)))
            continue;

        const int axis = direction / 2;
        const double h = spacing[static_cast<std::size_t>(axis)];
        result.minus_laplacian -= field_value(neighbor) / (h * h);
        ++result.artificial_boundary_neighbors;
    }
    if (result.artificial_boundary_neighbors > 0) {
        result.method =
            EdgePatchLaplacianMethod3D::ArtificialExteriorEvent;
        result.rhs_correction = -result.minus_laplacian;
    }
    return result;
}

EdgePatchLaplacianResult3D EdgePatchSingularLaplacian3D::evaluate(
    int grid_node) const
{
    EdgePatchLaplacianResult3D result;
    const Eigen::Vector3d center_point = point(grid_node);
    if (physical_domain_labels_[grid_node] != options_.active_domain_label) {
        return result;
    }

    const std::array<int, 6> neighbors = grid_.neighbors(grid_node);
    if (!in_geometric_patch(center_point))
        return exterior_event(grid_node, neighbors);

    for (int neighbor : neighbors) {
        if (neighbor < 0) {
            throw std::runtime_error(
                "edge patch stencil reaches the Cartesian outer boundary");
        }
        if (physical_domain_labels_[neighbor]
            != options_.active_domain_label) {
            ++result.physical_interface_neighbors;
        }
        if (!in_geometric_patch(point(neighbor)))
            ++result.artificial_boundary_neighbors;
    }

    if (result.physical_interface_neighbors > 0) {
        result.method =
            EdgePatchLaplacianMethod3D::AnalyticPhysicalInterface;
        result.artificial_boundary_neighbors = 0;
        result.minus_laplacian = analytic_minus_laplacian(
            center_point, neighbors, result.artificial_boundary_neighbors);
    } else {
        result.method = EdgePatchLaplacianMethod3D::SevenPointStencil;
        result.minus_laplacian =
            seven_point_minus_laplacian(grid_node, neighbors);
    }
    result.rhs_correction = -result.minus_laplacian;
    return result;
}

Eigen::VectorXd EdgePatchSingularLaplacian3D::evaluate_all() const
{
    Eigen::VectorXd result = Eigen::VectorXd::Zero(grid_.num_dofs());
    for (int node = 0; node < grid_.num_dofs(); ++node) {
        if (physical_domain_labels_[node] == options_.active_domain_label)
            result[node] = evaluate(node).minus_laplacian;
    }
    return result;
}

Eigen::VectorXd
EdgePatchSingularLaplacian3D::evaluate_rhs_correction_all() const
{
    Eigen::VectorXd result = Eigen::VectorXd::Zero(grid_.num_dofs());
    for (int node = 0; node < grid_.num_dofs(); ++node) {
        if (physical_domain_labels_[node] == options_.active_domain_label)
            result[node] = evaluate(node).rhs_correction;
    }
    return result;
}

} // namespace kfbim
