#pragma once

#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "../interface/interface_2d.hpp"

namespace kfbim {

struct SdCornerBasis2D {
    int corner = -1;
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    Eigen::Vector2d tau = Eigen::Vector2d::Zero();
    double coeff = 0.0;
    // Converts the raw branch jump (right side minus left side of tau) to the
    // physical jump convention used by the interface data.
    double jump_sign = 1.0;
};

struct SdCornerLifting2D {
    std::vector<SdCornerBasis2D> bases;

    double value(const Eigen::Vector2d& x) const;
    Eigen::Vector2d gradient(const Eigen::Vector2d& x) const;
    double laplacian(const Eigen::Vector2d& x) const;
    double value_for_corner(int corner, const Eigen::Vector2d& x) const;
    Eigen::Vector2d gradient_for_corner(int corner,
                                        const Eigen::Vector2d& x) const;
    double laplacian_for_corner(int corner, const Eigen::Vector2d& x) const;
    double value_for_patch(const Interface2D& iface,
                           int patch,
                           const Eigen::Vector2d& x) const;
    Eigen::Vector2d gradient_for_patch(const Interface2D& iface,
                                       int patch,
                                       const Eigen::Vector2d& x) const;
    double laplacian_for_patch(const Interface2D& iface,
                               int patch,
                               const Eigen::Vector2d& x) const;
    double side_value_for_patch(const Interface2D& iface,
                                int patch,
                                const Eigen::Vector2d& point,
                                const Eigen::Vector2d& normal,
                                double eps = 1.0e-9) const;
    double side_normal_derivative_for_patch(const Interface2D& iface,
                                            int patch,
                                            const Eigen::Vector2d& point,
                                            const Eigen::Vector2d& normal,
                                            double eps = 1.0e-9) const;
    double jump_value_for_patch(const Interface2D& iface,
                                int patch,
                                const Eigen::Vector2d& point,
                                const Eigen::Vector2d& normal,
                                double eps = 1.0e-9) const;
    double jump_value(const Interface2D& iface,
                      int q,
                      double eps = 1.0e-9) const;
    double laplacian_jump(const Interface2D& iface,
                          int q,
                          double eps = 1.0e-9) const;
};

enum class SdCornerStrengthSource2D {
    Fixed,
    P2EndpointFromUJump
};

enum class SdCornerRegularizedJumpMode2D {
    InterpolateFullThenSubtract,
    InterpolateResidualAtP2Nodes
};

struct SdCornerLiftingOptions2D {
    bool enabled = false;
    SdCornerLifting2D lifting;
    double normal_probe = 1.0e-9;
    SdCornerStrengthSource2D strength_source =
        SdCornerStrengthSource2D::Fixed;
    SdCornerRegularizedJumpMode2D regularized_jump_mode =
        SdCornerRegularizedJumpMode2D::InterpolateFullThenSubtract;
};

namespace sd_corner_lifting_detail {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

inline Eigen::Vector2d normalized_or_throw(const Eigen::Vector2d& v,
                                           const char* context)
{
    const double n = v.norm();
    if (n <= 1.0e-14)
        throw std::invalid_argument(context);
    return v / n;
}

inline double raw_sd_basis(const Eigen::Vector2d& x,
                           const Eigen::Vector2d& point,
                           const Eigen::Vector2d& tau)
{
    const Eigen::Vector2d t =
        normalized_or_throw(tau, "S_D corner basis tau must be nonzero");
    const Eigen::Vector2d v = x - point;
    const double xi = t.dot(v);
    const double eta = -t[1] * v[0] + t[0] * v[1];
    const double r = std::hypot(xi, eta);
    if (r <= 0.0)
        return 0.0;

    double theta = std::atan2(eta, xi);
    if (theta < 0.0)
        theta += kTwoPi;

    return eta * std::log(r) + xi * theta;
}

inline Eigen::Vector2d raw_sd_basis_gradient(const Eigen::Vector2d& x,
                                             const Eigen::Vector2d& point,
                                             const Eigen::Vector2d& tau)
{
    const Eigen::Vector2d t =
        normalized_or_throw(tau, "S_D corner basis tau must be nonzero");
    const Eigen::Vector2d n(-t[1], t[0]);
    const Eigen::Vector2d v = x - point;
    const double xi = t.dot(v);
    const double eta = n.dot(v);
    const double r = std::hypot(xi, eta);
    if (r <= 1.0e-300)
        return Eigen::Vector2d::Zero();

    double theta = std::atan2(eta, xi);
    if (theta < 0.0)
        theta += kTwoPi;

    return theta * t + (std::log(r) + 1.0) * n;
}

inline double raw_sd_basis_jump_on_axis(const Eigen::Vector2d& x,
                                        const Eigen::Vector2d& point,
                                        const Eigen::Vector2d& tau,
                                        double axis_tolerance)
{
    const Eigen::Vector2d t =
        normalized_or_throw(tau, "S_D corner basis tau must be nonzero");
    const Eigen::Vector2d n(-t[1], t[0]);
    const Eigen::Vector2d v = x - point;
    const double xi = t.dot(v);
    const double eta = n.dot(v);
    const double tol = std::max(axis_tolerance, 1.0e-14);
    if (xi <= tol)
        return 0.0;
    if (std::abs(eta) > tol * std::max(1.0, std::abs(xi)))
        return 0.0;
    return kTwoPi * xi;
}

inline double raw_sd_basis_laplacian(const Eigen::Vector2d& x,
                                     const Eigen::Vector2d& point,
                                     const Eigen::Vector2d& tau)
{
    (void)x;
    (void)point;
    (void)tau;
    return 0.0;
}

inline void p2_shape_deriv(double s, double dN[3])
{
    dN[0] = s - 0.5;
    dN[1] = -2.0 * s;
    dN[2] = s + 0.5;
}

inline Eigen::Vector2d p2_panel_tangent(const Interface2D& iface,
                                        int panel,
                                        double s)
{
    double dN[3];
    p2_shape_deriv(s, dN);
    Eigen::Vector2d tangent = Eigen::Vector2d::Zero();
    for (int q = 0; q < 3; ++q) {
        tangent += dN[q]
                 * iface.points()
                       .row(iface.point_index(panel, q))
                       .transpose();
    }
    return tangent;
}

inline double p2_panel_scalar_deriv(const Interface2D& iface,
                                    int panel,
                                    const Eigen::VectorXd& values,
                                    double s)
{
    double dN[3];
    p2_shape_deriv(s, dN);
    double value = 0.0;
    for (int q = 0; q < 3; ++q)
        value += dN[q] * values[iface.point_index(panel, q)];
    return value;
}

inline double p2_panel_endpoint_chord_length(const Interface2D& iface,
                                             int panel)
{
    const Eigen::Vector2d x0 =
        iface.points().row(iface.point_index(panel, 0)).transpose();
    const Eigen::Vector2d x2 =
        iface.points().row(iface.point_index(panel, 2)).transpose();
    const double length = (x2 - x0).norm();
    if (length <= 1.0e-14)
        throw std::invalid_argument(
            "S_D P2 endpoint coefficient requires nondegenerate panel chord");
    return length;
}

inline double p2_endpoint_start_arclength_deriv(
    const Interface2D& iface,
    int panel,
    const Eigen::VectorXd& values)
{
    const int q0 = iface.point_index(panel, 0);
    const int q1 = iface.point_index(panel, 1);
    const int q2 = iface.point_index(panel, 2);
    return (-3.0 * values[q0] + 4.0 * values[q1] - values[q2])
         / p2_panel_endpoint_chord_length(iface, panel);
}

inline double p2_endpoint_end_arclength_deriv(
    const Interface2D& iface,
    int panel,
    const Eigen::VectorXd& values)
{
    const int q0 = iface.point_index(panel, 0);
    const int q1 = iface.point_index(panel, 1);
    const int q2 = iface.point_index(panel, 2);
    return (values[q0] - 4.0 * values[q1] + 3.0 * values[q2])
         / p2_panel_endpoint_chord_length(iface, panel);
}

} // namespace sd_corner_lifting_detail

inline double sd_corner_basis_jump_slope(const SdCornerBasis2D& basis,
                                         double axis_tolerance = 1.0e-9)
{
    const Eigen::Vector2d tau =
        sd_corner_lifting_detail::normalized_or_throw(
            basis.tau, "S_D corner basis tau must be nonzero");
    constexpr double kSlopeProbe = 1.0e-5;
    const Eigen::Vector2d pt = basis.point + kSlopeProbe * tau;
    return basis.jump_sign
         * sd_corner_lifting_detail::raw_sd_basis_jump_on_axis(
               pt, basis.point, basis.tau, axis_tolerance)
         / kSlopeProbe;
}

inline double sd_corner_physical_jump_sign(const SdCornerBasis2D& basis,
                                           const Eigen::Vector2d& outward_normal)
{
    const Eigen::Vector2d tau =
        sd_corner_lifting_detail::normalized_or_throw(
            basis.tau, "S_D corner basis tau must be nonzero");
    const Eigen::Vector2d left(-tau[1], tau[0]);
    const Eigen::Vector2d inside_dir =
        -sd_corner_lifting_detail::normalized_or_throw(
            outward_normal, "S_D corner normal must be nonzero");
    return left.dot(inside_dir) > 0.0 ? -1.0 : 1.0;
}

inline double sd_panel_endpoint_arclength_speed(const Interface2D& iface,
                                                int panel,
                                                double endpoint_s)
{
    const double speed =
        sd_corner_lifting_detail::p2_panel_tangent(
            iface, panel, endpoint_s).norm();
    if (speed <= 1.0e-14)
        throw std::invalid_argument(
            "S_D P2 endpoint coefficient requires nondegenerate panel speed");
    return speed;
}

inline SdCornerLifting2D build_p2_endpoint_sd_lifting_from_u_jump(
    const Interface2D& iface,
    const Eigen::VectorXd& u_jump,
    double normal_probe = 1.0e-9)
{
    if (iface.points_per_panel() != 3
        || iface.panel_node_layout() != PanelNodeLayout2D::QuadraticLagrange) {
        throw std::invalid_argument(
            "S_D P2 endpoint lifting requires P2 quadratic Lagrange panels");
    }
    if (u_jump.size() != iface.num_points()) {
        throw std::invalid_argument(
            "S_D P2 endpoint lifting u_jump size must equal interface point count");
    }

    SdCornerLifting2D lifting;
    lifting.bases.reserve(static_cast<std::size_t>(2 * iface.corners().size()));

    for (int c = 0; c < static_cast<int>(iface.corners().size()); ++c) {
        const CornerData2D& corner = iface.corner(c);
        const Eigen::Vector2d point =
            iface.points().row(corner.point).transpose();

        SdCornerBasis2D next;
        next.corner = c;
        next.point = point;
        next.tau =
            sd_corner_lifting_detail::normalized_or_throw(
                corner.tangent_plus,
                "S_D corner tangent_plus must be nonzero");
        next.jump_sign =
            sd_corner_physical_jump_sign(next, corner.normal_plus);
        const double d_next =
            sd_corner_lifting_detail::p2_endpoint_start_arclength_deriv(
                iface, corner.next_panel, u_jump);
        next.coeff =
            d_next / sd_corner_basis_jump_slope(next, normal_probe);
        lifting.bases.push_back(next);

        SdCornerBasis2D prev;
        prev.corner = c;
        prev.point = point;
        prev.tau =
            -sd_corner_lifting_detail::normalized_or_throw(
                corner.tangent_minus,
                "S_D corner tangent_minus must be nonzero");
        prev.jump_sign =
            sd_corner_physical_jump_sign(prev, corner.normal_minus);
        const double d_prev_global =
            sd_corner_lifting_detail::p2_endpoint_end_arclength_deriv(
                iface, corner.prev_panel, u_jump);
        prev.coeff =
            -d_prev_global / sd_corner_basis_jump_slope(prev, normal_probe);
        lifting.bases.push_back(prev);
    }
    return lifting;
}

inline double SdCornerLifting2D::value(const Eigen::Vector2d& x) const
{
    double total = 0.0;
    for (const SdCornerBasis2D& basis : bases) {
        total += basis.coeff
               * sd_corner_lifting_detail::raw_sd_basis(
                     x, basis.point, basis.tau);
    }
    return total;
}

inline Eigen::Vector2d SdCornerLifting2D::gradient(
    const Eigen::Vector2d& x) const
{
    Eigen::Vector2d total = Eigen::Vector2d::Zero();
    for (const SdCornerBasis2D& basis : bases) {
        total += basis.coeff
               * sd_corner_lifting_detail::raw_sd_basis_gradient(
                     x, basis.point, basis.tau);
    }
    return total;
}

inline double SdCornerLifting2D::laplacian(const Eigen::Vector2d& x) const
{
    double total = 0.0;
    for (const SdCornerBasis2D& basis : bases) {
        total += basis.coeff
               * sd_corner_lifting_detail::raw_sd_basis_laplacian(
                     x, basis.point, basis.tau);
    }
    return total;
}

inline double SdCornerLifting2D::value_for_corner(
    int corner,
    const Eigen::Vector2d& x) const
{
    double total = 0.0;
    for (const SdCornerBasis2D& basis : bases) {
        if (basis.corner != corner)
            continue;
        total += basis.coeff
               * sd_corner_lifting_detail::raw_sd_basis(
                     x, basis.point, basis.tau);
    }
    return total;
}

inline Eigen::Vector2d SdCornerLifting2D::gradient_for_corner(
    int corner,
    const Eigen::Vector2d& x) const
{
    Eigen::Vector2d total = Eigen::Vector2d::Zero();
    for (const SdCornerBasis2D& basis : bases) {
        if (basis.corner != corner)
            continue;
        total += basis.coeff
               * sd_corner_lifting_detail::raw_sd_basis_gradient(
                     x, basis.point, basis.tau);
    }
    return total;
}

inline double SdCornerLifting2D::laplacian_for_corner(
    int corner,
    const Eigen::Vector2d& x) const
{
    double total = 0.0;
    for (const SdCornerBasis2D& basis : bases) {
        if (basis.corner != corner)
            continue;
        total += basis.coeff
               * sd_corner_lifting_detail::raw_sd_basis_laplacian(
                     x, basis.point, basis.tau);
    }
    return total;
}

inline double SdCornerLifting2D::value_for_patch(
    const Interface2D& iface,
    int patch,
    const Eigen::Vector2d& x) const
{
    if (patch < 0 || patch >= static_cast<int>(iface.corner_patches().size()))
        return 0.0;
    return value_for_corner(
        iface.corner_patches()[static_cast<std::size_t>(patch)].corner, x);
}

inline Eigen::Vector2d SdCornerLifting2D::gradient_for_patch(
    const Interface2D& iface,
    int patch,
    const Eigen::Vector2d& x) const
{
    if (patch < 0 || patch >= static_cast<int>(iface.corner_patches().size()))
        return Eigen::Vector2d::Zero();
    return gradient_for_corner(
        iface.corner_patches()[static_cast<std::size_t>(patch)].corner, x);
}

inline double SdCornerLifting2D::laplacian_for_patch(
    const Interface2D& iface,
    int patch,
    const Eigen::Vector2d& x) const
{
    if (patch < 0 || patch >= static_cast<int>(iface.corner_patches().size()))
        return 0.0;
    return laplacian_for_corner(
        iface.corner_patches()[static_cast<std::size_t>(patch)].corner, x);
}

inline double SdCornerLifting2D::side_value_for_patch(
    const Interface2D& iface,
    int patch,
    const Eigen::Vector2d& point,
    const Eigen::Vector2d& normal,
    double eps) const
{
    if (patch < 0 || patch >= static_cast<int>(iface.corner_patches().size()))
        return 0.0;
    const Eigen::Vector2d n =
        sd_corner_lifting_detail::normalized_or_throw(
            normal, "S_D corner patch side normal must be nonzero");
    const CornerPatch2D& patch_data =
        iface.corner_patches()[static_cast<std::size_t>(patch)];
    Eigen::Vector2d side_point = point;
    if (patch_data.side == CornerPatchSide2D::Interior)
        side_point -= eps * n;
    else
        side_point += eps * n;
    return value_for_patch(iface, patch, side_point);
}

inline double SdCornerLifting2D::side_normal_derivative_for_patch(
    const Interface2D& iface,
    int patch,
    const Eigen::Vector2d& point,
    const Eigen::Vector2d& normal,
    double eps) const
{
    if (patch < 0 || patch >= static_cast<int>(iface.corner_patches().size()))
        return 0.0;
    const Eigen::Vector2d n =
        sd_corner_lifting_detail::normalized_or_throw(
            normal, "S_D corner patch side normal must be nonzero");
    const CornerPatch2D& patch_data =
        iface.corner_patches()[static_cast<std::size_t>(patch)];
    Eigen::Vector2d side_point = point;
    if (patch_data.side == CornerPatchSide2D::Interior)
        side_point -= eps * n;
    else
        side_point += eps * n;
    return gradient_for_patch(iface, patch, side_point).dot(n);
}

inline double SdCornerLifting2D::jump_value(
    const Interface2D& iface,
    int q,
    double eps) const
{
    if (q < 0 || q >= iface.num_points())
        throw std::invalid_argument("S_D corner jump point index is invalid");
    const Eigen::Vector2d point = iface.points().row(q).transpose();
    double total = 0.0;
    for (const SdCornerBasis2D& basis : bases) {
        total += basis.coeff
               * basis.jump_sign
               * sd_corner_lifting_detail::raw_sd_basis_jump_on_axis(
                     point, basis.point, basis.tau, eps);
    }
    return total;
}

inline double SdCornerLifting2D::jump_value_for_patch(
    const Interface2D& iface,
    int patch,
    const Eigen::Vector2d& point,
    const Eigen::Vector2d& normal,
    double eps) const
{
    (void)normal;
    if (patch < 0 || patch >= static_cast<int>(iface.corner_patches().size()))
        return 0.0;
    const int corner =
        iface.corner_patches()[static_cast<std::size_t>(patch)].corner;
    double total = 0.0;
    for (const SdCornerBasis2D& basis : bases) {
        if (basis.corner != corner)
            continue;
        total += basis.coeff
               * basis.jump_sign
               * sd_corner_lifting_detail::raw_sd_basis_jump_on_axis(
                     point, basis.point, basis.tau, eps);
    }
    return total;
}

inline double SdCornerLifting2D::laplacian_jump(
    const Interface2D& iface,
    int q,
    double eps) const
{
    if (q < 0 || q >= iface.num_points())
        throw std::invalid_argument("S_D corner jump point index is invalid");
    const Eigen::Vector2d point = iface.points().row(q).transpose();
    const Eigen::Vector2d normal =
        sd_corner_lifting_detail::normalized_or_throw(
            iface.normals().row(q).transpose(),
            "S_D corner jump normal must be nonzero");
    const Eigen::Vector2d inside = point - eps * normal;
    const Eigen::Vector2d outside = point + eps * normal;
    return laplacian(inside) - laplacian(outside);
}

} // namespace kfbim
