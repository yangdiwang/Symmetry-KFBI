#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../geometry/p2_curve_2d.hpp"
#include "../interface/interface_2d.hpp"
#include "local_poly.hpp"

namespace kfbim {

enum class CornerPatchSingularFamily2D {
    Sine = 0,
    Cosine = 1
};

enum class CornerPatchSingularFamilyPolicy2D {
    Sine = 0,
    ExteriorCosineInteriorSine = 1,
    InteriorDirichletTransmissionPair = 2,
    InteriorNeumannTransmissionPair = 3
};

enum class CornerPatchSingularModel2D {
    OneSidedWedge = 0,
    InteriorDirichletTransmissionPair = 1,
    InteriorNeumannTransmissionPair = 2
};

enum class CornerPatchSingularCoefficientFitMethod2D {
    ProjectedHarmonic = 0,
    EdgeP2Trace = 1
};

struct CornerPatchSingularMode2D {
    double lambda = 0.0;
    double amplitude = 0.0;
    // Radial scale in (r / radial_scale)^lambda.  The legacy value 1 keeps
    // the historical raw r^lambda basis.
    double radial_scale = 1.0;
    CornerPatchSingularFamily2D family = CornerPatchSingularFamily2D::Sine;
    CornerPatchSingularModel2D model =
        CornerPatchSingularModel2D::OneSidedWedge;
    double interior_cos_factor = 0.0;
    double interior_sine_factor = 1.0;
    double exterior_cos_factor = -1.0;
    double exterior_sine_factor = 0.0;
    double prev_jump_factor = 1.0;
    double next_jump_factor = 1.0;
};

struct CornerPatchFitDiagnostics2D {
    int rows = 0;
    int value_rows = 0;
    int normal_rows = 0;
    int rhs_rows = 0;
    int num_singular_modes = 0;
    int num_nuisance_modes = 0;
    int singular_rank = 0;
    int nuisance_rank = 0;
    double condition_number = 0.0;
    double relative_residual = 0.0;
    double coeff_norm = 0.0;
    int regular_rows = 0;
    int regular_rank = 0;
    double regular_condition_number = 0.0;
    double regular_relative_residual = 0.0;
    double regular_coeff_norm = 0.0;
};

struct CornerPatchCorrectionData2D {
    int patch = -1;
    int point = -1;
    CornerPatchSide2D side = CornerPatchSide2D::Interior;
    double lambda = 0.0;
    double amplitude = 0.0;
    double opening_angle = 0.0;
    double radius = 0.0;
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    Eigen::Vector2d ray_minus = Eigen::Vector2d::Zero();
    Eigen::Vector2d ray_plus = Eigen::Vector2d::Zero();
    LocalPoly2D regular_part;
    LocalPoly2D interface_regular_part;
    std::vector<int> interface_regular_panels;
    bool has_interface_regular_part = false;
    bool has_explicit_regular_part = false;
    std::vector<CornerPatchSingularMode2D> modes;
    CornerPatchFitDiagnostics2D diagnostics;
};

struct CornerPatchFittingSample2D {
    int patch = -1;
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    Eigen::Vector2d normal = Eigen::Vector2d::Zero();
    double value = 0.0;
    double normal_derivative = 0.0;
    double rhs_value = 0.0;
    bool has_value = true;
    bool has_normal_derivative = false;
    bool interface_jump_value = true;
    bool has_rhs_value = false;
    double confidence = 1.0;
};

struct CornerPatchArtificialDirichletSample2D {
    int patch = -1;
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    double value = 0.0;
};

using CornerPatchArtificialDirichletSampler2D = std::function<
    std::vector<CornerPatchArtificialDirichletSample2D>(
        const Interface2D& iface,
        const CornerPatch2D& patch,
        int patch_id,
        const Eigen::VectorXd& u_jump,
        const Eigen::VectorXd& un_jump,
        const Eigen::VectorXd& rhs_jump,
        double kappa)>;

using CornerPatchFittingSampler2D = std::function<
    std::vector<CornerPatchFittingSample2D>(
        const Interface2D& iface,
        const CornerPatch2D& patch,
        int patch_id,
        const Eigen::VectorXd& u_jump,
        const Eigen::VectorXd& un_jump,
        const Eigen::VectorXd& rhs_jump,
        double kappa)>;

struct CornerPatchSingularFitOptions2D {
    bool enabled = false;
    bool use_connected_panel_samples = true;
    int panels_per_direction = 2;
    int samples_per_panel = 2;
    int max_sine_modes = 1;
    CornerPatchSingularFamilyPolicy2D family_policy =
        CornerPatchSingularFamilyPolicy2D::Sine;
    int nuisance_degree = 3;
    bool use_normal_rows = false;
    double w_min = 0.25;
    double rank_tol_rel = 1.0e-12;
    bool throw_on_rank_deficient = true;
    bool fit_regular_after_singular = true;
    double basis_radius = 1.0;
    CornerPatchSingularCoefficientFitMethod2D coefficient_fit_method =
        CornerPatchSingularCoefficientFitMethod2D::ProjectedHarmonic;
    double edge_p2_fit_radius = 0.0;
    int edge_p2_max_points_per_direction = 4;
    double edge_p2_weight_shift = 0.15;
    bool edge_aligned_half_step_sampling = false;
    double edge_aligned_step = 0.0;
    bool edge_aligned_include_corner_row = true;
    // Optional per-patch singular amplitudes for projected fitting. Finite
    // entries are held fixed; non-finite entries are fitted from samples.
    Eigen::VectorXd fixed_amplitudes;
};

struct CornerPatchSolverOptions2D {
    // If empty, the first-stage solver uses u_jump at each corner point as the
    // singular amplitude. If provided, this vector is indexed by patch id and
    // gives the Dirichlet amplitude on the artificial boundary midpoint.
    Eigen::VectorXd artificial_dirichlet_amplitudes;

    // Optional P2 regular terms, indexed by patch id. If empty, the solver uses
    // a zero quadratic Taylor polynomial centered at each corner.
    std::vector<LocalPoly2D> regular_parts;

    // Optional artificial-boundary Dirichlet collocation samples. When samples
    // are supplied for a patch, any missing part of the model is fitted by
    // least squares:
    //   g(x) = a*S_corner(x) + P2_regular(x).
    // Explicit amplitudes and explicit regular_parts are held fixed.
    std::vector<CornerPatchArtificialDirichletSample2D> artificial_dirichlet_samples;

    // Optional dynamic sampler used by operator/potential paths. It is called
    // during every spread apply, so generated artificial BC data can depend on
    // the current jump vectors while preserving operator linearity when the
    // callback itself is linear.
    CornerPatchArtificialDirichletSampler2D artificial_dirichlet_sampler;

    CornerPatchSingularFitOptions2D singular_fit;
    std::vector<CornerPatchFittingSample2D> fitting_samples;
    CornerPatchFittingSampler2D fitting_sampler;
};

namespace laplace_corner_patch_detail {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

inline double cross2(const Eigen::Vector2d& a, const Eigen::Vector2d& b)
{
    return a[0] * b[1] - a[1] * b[0];
}

inline Eigen::Vector2d normalized_or_throw(const Eigen::Vector2d& v,
                                           const char* context)
{
    const double n = v.norm();
    if (n <= 1.0e-14)
        throw std::invalid_argument(context);
    return v / n;
}

inline double ccw_angle(const Eigen::Vector2d& from, const Eigen::Vector2d& to)
{
    double angle = std::atan2(cross2(from, to), from.dot(to));
    if (angle < 0.0)
        angle += kTwoPi;
    return angle;
}

inline double signed_angle(const Eigen::Vector2d& from,
                           const Eigen::Vector2d& to)
{
    return std::atan2(cross2(from, to), from.dot(to));
}

inline double corner_wedge_local_angle(const Eigen::Vector2d& ray_minus,
                                       const Eigen::Vector2d& ray_plus,
                                       double opening,
                                       const Eigen::Vector2d& dir)
{
    const double raw = signed_angle(ray_minus, dir);
    if (!(opening > kPi))
        return raw;

    const double end_signed = opening - kTwoPi;
    if (raw < 0.0) {
        const double dist_start = std::abs(raw);
        const double dist_end = std::abs(raw - end_signed);
        if (dist_end < dist_start)
            return opening + (raw - end_signed);
    }
    return raw;
}

inline bool direction_in_corner_wedge(const Eigen::Vector2d& ray_minus,
                                      double opening,
                                      const Eigen::Vector2d& dir,
                                      double tolerance = 1.0e-12)
{
    const double angle = ccw_angle(ray_minus, dir);
    return angle <= opening + tolerance;
}

inline double mode_radial_scale_power(const CornerPatchSingularMode2D& mode)
{
    return std::isfinite(mode.radial_scale) && mode.radial_scale > 0.0
        ? std::pow(mode.radial_scale, mode.lambda)
        : 1.0;
}

inline double polar_harmonic_value(double r,
                                   double theta,
                                   const CornerPatchSingularMode2D& mode,
                                   double cos_coeff,
                                   double sin_coeff)
{
    if (r <= 1.0e-14)
        return 0.0;
    const double angle = mode.lambda * theta;
    const double angular = cos_coeff * std::cos(angle)
                         + sin_coeff * std::sin(angle);
    return std::pow(r, mode.lambda) * angular
         / mode_radial_scale_power(mode);
}

inline Eigen::Vector2d polar_harmonic_gradient(
    const Eigen::Vector2d& er,
    double r,
    double theta,
    const CornerPatchSingularMode2D& mode,
    double cos_coeff,
    double sin_coeff)
{
    if (r <= 1.0e-14)
        return Eigen::Vector2d::Zero();
    const double angle = mode.lambda * theta;
    const double angular = cos_coeff * std::cos(angle)
                         + sin_coeff * std::sin(angle);
    const double dangular = mode.lambda
        * (-cos_coeff * std::sin(angle) + sin_coeff * std::cos(angle));
    const double scale = mode_radial_scale_power(mode);
    const Eigen::Vector2d et(-er[1], er[0]);
    const double radial =
        mode.lambda * std::pow(r, mode.lambda - 1.0) * angular / scale;
    const double tangential =
        std::pow(r, mode.lambda - 1.0) * dangular / scale;
    return radial * er + tangential * et;
}

inline Eigen::Matrix<double, 1, 6> p2_taylor_basis(
    const Eigen::Vector2d& center,
    const Eigen::Vector2d& x)
{
    const double dx = x[0] - center[0];
    const double dy = x[1] - center[1];
    Eigen::Matrix<double, 1, 6> basis;
    basis << 1.0, dx, dy, 0.5 * dx * dx, dx * dy, 0.5 * dy * dy;
    return basis;
}

inline Eigen::Matrix<double, 2, 6> p2_taylor_gradient_basis(
    const Eigen::Vector2d& center,
    const Eigen::Vector2d& x)
{
    const double dx = x[0] - center[0];
    const double dy = x[1] - center[1];
    Eigen::Matrix<double, 2, 6> basis;
    basis << 0.0, 1.0, 0.0, dx, dy, 0.0,
             0.0, 0.0, 1.0, 0.0, dx, dy;
    return basis;
}

inline Eigen::Matrix<double, 1, 10> p3_taylor_basis(
    const Eigen::Vector2d& center,
    const Eigen::Vector2d& x)
{
    const double dx = x[0] - center[0];
    const double dy = x[1] - center[1];
    Eigen::Matrix<double, 1, 10> basis;
    basis << 1.0,
             dx,
             dy,
             0.5 * dx * dx,
             dx * dy,
             0.5 * dy * dy,
             dx * dx * dx / 6.0,
             0.5 * dx * dx * dy,
             0.5 * dx * dy * dy,
             dy * dy * dy / 6.0;
    return basis;
}

inline Eigen::Matrix<double, 2, 10> p3_taylor_gradient_basis(
    const Eigen::Vector2d& center,
    const Eigen::Vector2d& x)
{
    const double dx = x[0] - center[0];
    const double dy = x[1] - center[1];
    Eigen::Matrix<double, 2, 10> basis;
    basis << 0.0,
             1.0,
             0.0,
             dx,
             dy,
             0.0,
             0.5 * dx * dx,
             dx * dy,
             0.5 * dy * dy,
             0.0,
             0.0,
             0.0,
             1.0,
             0.0,
             dx,
             dy,
             0.0,
             0.5 * dx * dx,
             dx * dy,
             0.5 * dy * dy;
    return basis;
}

inline Eigen::Matrix<double, 1, 10> p3_taylor_screened_operator_basis(
    const Eigen::Vector2d& center,
    const Eigen::Vector2d& x,
    double kappa)
{
    const double dx = x[0] - center[0];
    const double dy = x[1] - center[1];
    Eigen::Matrix<double, 1, 10> basis;
    basis << kappa,
             kappa * dx,
             kappa * dy,
             kappa * 0.5 * dx * dx - 1.0,
             kappa * dx * dy,
             kappa * 0.5 * dy * dy - 1.0,
             kappa * dx * dx * dx / 6.0 - dx,
             kappa * 0.5 * dx * dx * dy - dy,
             kappa * 0.5 * dx * dy * dy - dx,
             kappa * dy * dy * dy / 6.0 - dy;
    return basis;
}

inline int local_index_of_point(const Interface2D& iface, int panel, int point)
{
    for (int q = 0; q < iface.points_per_panel(); ++q) {
        if (iface.point_index(panel, q) == point)
            return q;
    }
    return -1;
}

inline int connected_panel_along_endpoint(const Interface2D& iface,
                                          int panel,
                                          int endpoint,
                                          bool moving_forward)
{
    const int component = iface.panel_components()[panel];
    for (int p = 0; p < iface.num_panels(); ++p) {
        if (p == panel || iface.panel_components()[p] != component)
            continue;
        if (moving_forward) {
            if (iface.point_index(p, 0) == endpoint)
                return p;
        } else {
            if (iface.point_index(p, iface.points_per_panel() - 1) == endpoint)
                return p;
        }
    }
    return -1;
}

inline void append_connected_panel_point_sample(
    const Interface2D& iface,
    int panel,
    int local,
    int point,
    int patch_id,
    const Eigen::VectorXd& u_jump,
    const Eigen::VectorXd& un_jump,
    const Eigen::VectorXd& rhs_jump,
    const CornerPatchSingularFitOptions2D& fit_options,
    std::vector<CornerPatchFittingSample2D>& samples)
{
    CornerPatchFittingSample2D sample;
    sample.patch = patch_id;
    sample.point = iface.points().row(point).transpose();
    sample.normal = iface.panel_normal(panel, local);
    sample.value = u_jump[point];
    sample.normal_derivative = un_jump[point];
    sample.rhs_value = rhs_jump[point];
    sample.has_value = true;
    sample.has_normal_derivative = fit_options.use_normal_rows;
    sample.has_rhs_value = true;
    samples.push_back(sample);
}

inline std::vector<CornerPatchFittingSample2D> collect_one_panel_direction_samples(
    const Interface2D& iface,
    int incident_panel,
    int corner_point,
    int patch_id,
    const Eigen::VectorXd& u_jump,
    const Eigen::VectorXd& un_jump,
    const Eigen::VectorXd& rhs_jump,
    const CornerPatchSingularFitOptions2D& fit_options)
{
    if (iface.points_per_panel() != 3
        || iface.panel_node_layout() != PanelNodeLayout2D::QuadraticLagrange) {
        throw std::invalid_argument(
            "projected corner patch fitting requires P2 quadratic panels");
    }
    if (fit_options.panels_per_direction < 1
        || fit_options.samples_per_panel != 2) {
        throw std::invalid_argument(
            "projected corner patch fitting requires at least one panel per direction and two samples per panel");
    }

    const int local_corner =
        local_index_of_point(iface, incident_panel, corner_point);
    if (local_corner != 0 && local_corner != 2) {
        throw std::invalid_argument(
            "corner incident panel must contain the corner as an endpoint");
    }

    const bool moving_forward = local_corner == 0;
    std::vector<CornerPatchFittingSample2D> samples;
    samples.reserve(static_cast<std::size_t>(
        2 * fit_options.panels_per_direction));
    int panel = incident_panel;
    for (int k = 0; k < fit_options.panels_per_direction; ++k) {
        const int outer_local = moving_forward ? 2 : 0;
        const int outer_point = iface.point_index(panel, outer_local);
        append_connected_panel_point_sample(iface,
                                            panel,
                                            1,
                                            iface.point_index(panel, 1),
                                            patch_id,
                                            u_jump,
                                            un_jump,
                                            rhs_jump,
                                            fit_options,
                                            samples);
        append_connected_panel_point_sample(iface,
                                            panel,
                                            outer_local,
                                            outer_point,
                                            patch_id,
                                            u_jump,
                                            un_jump,
                                            rhs_jump,
                                            fit_options,
                                            samples);
        if (k + 1 == fit_options.panels_per_direction)
            break;
        const int next_panel =
            connected_panel_along_endpoint(iface,
                                           panel,
                                           outer_point,
                                           moving_forward);
        if (next_panel < 0) {
            throw std::invalid_argument(
                "projected corner patch fitting could not find a connected panel in one corner direction");
        }
        const int local_shared = local_index_of_point(iface,
                                                      next_panel,
                                                      outer_point);
        if (moving_forward && local_shared != 0) {
            throw std::invalid_argument(
                "projected corner patch fitting expected a forward panel to start at the shared endpoint");
        }
        if (!moving_forward && local_shared != 2) {
            throw std::invalid_argument(
                "projected corner patch fitting expected a backward panel to end at the shared endpoint");
        }
        panel = next_panel;
    }
    return samples;
}

inline std::vector<int> collect_one_panel_direction_panel_ids(
    const Interface2D& iface,
    int incident_panel,
    int corner_point,
    const CornerPatchSingularFitOptions2D& fit_options)
{
    if (iface.points_per_panel() != 3
        || iface.panel_node_layout() != PanelNodeLayout2D::QuadraticLagrange) {
        throw std::invalid_argument(
            "projected corner patch fitting requires P2 quadratic panels");
    }
    if (fit_options.panels_per_direction < 1
        || fit_options.samples_per_panel != 2) {
        throw std::invalid_argument(
            "projected corner patch fitting requires at least one panel per direction and two samples per panel");
    }

    const int local_corner =
        local_index_of_point(iface, incident_panel, corner_point);
    if (local_corner != 0 && local_corner != 2) {
        throw std::invalid_argument(
            "corner incident panel must contain the corner as an endpoint");
    }

    const bool moving_forward = local_corner == 0;
    std::vector<int> panels;
    panels.reserve(static_cast<std::size_t>(fit_options.panels_per_direction));
    int panel = incident_panel;
    for (int k = 0; k < fit_options.panels_per_direction; ++k) {
        panels.push_back(panel);
        if (k + 1 == fit_options.panels_per_direction)
            break;
        const int outer_point =
            iface.point_index(panel, moving_forward ? 2 : 0);
        const int next_panel =
            connected_panel_along_endpoint(iface,
                                           panel,
                                           outer_point,
                                           moving_forward);
        if (next_panel < 0) {
            throw std::invalid_argument(
                "projected corner patch fitting could not find a connected panel in one corner direction");
        }
        const int local_shared = local_index_of_point(iface,
                                                      next_panel,
                                                      outer_point);
        if (moving_forward && local_shared != 0) {
            throw std::invalid_argument(
                "projected corner patch fitting expected a forward panel to start at the shared endpoint");
        }
        if (!moving_forward && local_shared != 2) {
            throw std::invalid_argument(
                "projected corner patch fitting expected a backward panel to end at the shared endpoint");
        }
        panel = next_panel;
    }
    return panels;
}

inline std::vector<int> collect_four_connected_panel_ids(
    const Interface2D& iface,
    const CornerData2D& corner,
    const CornerPatchSingularFitOptions2D& fit_options)
{
    std::vector<int> panels =
        collect_one_panel_direction_panel_ids(iface,
                                             corner.prev_panel,
                                             corner.point,
                                             fit_options);
    std::vector<int> plus =
        collect_one_panel_direction_panel_ids(iface,
                                             corner.next_panel,
                                             corner.point,
                                             fit_options);
    panels.insert(panels.end(), plus.begin(), plus.end());
    return panels;
}

inline std::vector<int> collect_incident_corner_panel_ids(
    const CornerData2D& corner)
{
    return {corner.prev_panel, corner.next_panel};
}

inline std::vector<int> collect_edge_panel_ids_within_radius(
    const Interface2D& iface,
    const CornerData2D& corner,
    double radius)
{
    if (!(radius > 0.0))
        return collect_incident_corner_panel_ids(corner);

    std::vector<int> panels;
    auto add_coordinate_aligned_panels = [&]() {
        const Eigen::Vector2d center =
            iface.points().row(corner.point).transpose();
        const Eigen::Vector2d dirs[2] = {
            normalized_or_throw(-corner.tangent_minus,
                                "corner tangent must be nonzero"),
            normalized_or_throw(corner.tangent_plus,
                                "corner tangent must be nonzero")
        };
        for (int panel = 0; panel < iface.num_panels(); ++panel) {
            bool keep = false;
            for (int local = 0; local < iface.points_per_panel(); ++local) {
                const int q = iface.point_index(panel, local);
                if (q == corner.point)
                    continue;
                const Eigen::Vector2d pt = iface.points().row(q).transpose();
                const Eigen::Vector2d v = pt - center;
                const double distance = v.norm();
                if (!(distance > 1.0e-13)
                    || distance > radius + 1.0e-13) {
                    continue;
                }
                const Eigen::Vector2d unit = v / distance;
                for (const Eigen::Vector2d& dir : dirs) {
                    if (unit.dot(dir) > 1.0 - 1.0e-10
                        && std::abs(cross2(unit, dir)) < 1.0e-8) {
                        keep = true;
                        break;
                    }
                }
                if (keep)
                    break;
            }
            if (keep)
                panels.push_back(panel);
        }
    };

    for (int side = 0; side < 2; ++side) {
        const bool previous_edge = side == 0;
        const int incident_panel =
            previous_edge ? corner.prev_panel : corner.next_panel;
        const int local_corner =
            local_index_of_point(iface, incident_panel, corner.point);
        if (local_corner != 0 && local_corner != 2) {
            panels.clear();
            add_coordinate_aligned_panels();
            if (panels.empty())
                return collect_incident_corner_panel_ids(corner);
            std::sort(panels.begin(), panels.end());
            panels.erase(std::unique(panels.begin(), panels.end()),
                         panels.end());
            return panels;
        }
        const bool moving_forward = local_corner == 0;
        int panel = incident_panel;
        double covered = 0.0;
        for (int guard = 0; guard < iface.num_panels(); ++guard) {
            if (covered > radius + 1.0e-13)
                break;
            panels.push_back(panel);
            const int start_local = moving_forward ? 0 : 2;
            const int end_local = moving_forward ? 2 : 0;
            const int start_point = iface.point_index(panel, start_local);
            const int end_point = iface.point_index(panel, end_local);
            const double panel_length =
                (iface.points().row(end_point).transpose()
                 - iface.points().row(start_point).transpose())
                    .norm();
            covered += panel_length;
            if (covered >= radius - 1.0e-13)
                break;
            const int outer_point = iface.point_index(panel, end_local);
            const int next_panel =
                connected_panel_along_endpoint(iface,
                                               panel,
                                               outer_point,
                                               moving_forward);
            if (next_panel < 0)
                break;
            panel = next_panel;
        }
    }
    std::sort(panels.begin(), panels.end());
    panels.erase(std::unique(panels.begin(), panels.end()), panels.end());
    return panels;
}

inline std::vector<CornerPatchFittingSample2D> collect_four_connected_panel_samples(
    const Interface2D& iface,
    const CornerData2D& corner,
    int patch_id,
    const Eigen::VectorXd& u_jump,
    const Eigen::VectorXd& un_jump,
    const Eigen::VectorXd& rhs_jump,
    const CornerPatchSingularFitOptions2D& fit_options)
{
    std::vector<CornerPatchFittingSample2D> samples;
    samples.reserve(8);
    std::vector<CornerPatchFittingSample2D> minus =
        collect_one_panel_direction_samples(iface,
                                            corner.prev_panel,
                                            corner.point,
                                            patch_id,
                                            u_jump,
                                            un_jump,
                                            rhs_jump,
                                            fit_options);
    std::vector<CornerPatchFittingSample2D> plus =
        collect_one_panel_direction_samples(iface,
                                            corner.next_panel,
                                            corner.point,
                                            patch_id,
                                            u_jump,
                                            un_jump,
                                            rhs_jump,
                                            fit_options);
    samples.insert(samples.end(), minus.begin(), minus.end());
    samples.insert(samples.end(), plus.begin(), plus.end());

    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(samples.size()); ++j) {
            if ((samples[i].point - samples[j].point).norm() <= 1.0e-14) {
                throw std::invalid_argument(
                    "projected corner patch fitting connected panel samples must be unique");
            }
        }
    }
    const std::size_t expected_samples =
        static_cast<std::size_t>(4 * fit_options.panels_per_direction);
    if (samples.size() != expected_samples) {
        throw std::invalid_argument(
            "projected corner patch fitting collected the wrong number of connected panel samples");
    }
    return samples;
}

inline double evaluate_singular_mode_basis(const CornerPatch2D& patch,
                                           const CornerPatchSingularMode2D& mode,
                                           const Eigen::Vector2d& x)
{
    const Eigen::Vector2d v = x - patch.center;
    const double r = v.norm();
    if (r <= 1.0e-14)
        return 0.0;
    const Eigen::Vector2d ray0 =
        normalized_or_throw(patch.ray_minus,
                            "corner patch ray_minus must be nonzero");
    const Eigen::Vector2d ray1 =
        normalized_or_throw(patch.ray_plus,
                            "corner patch ray_plus must be nonzero");
    if (mode.model
        == CornerPatchSingularModel2D::InteriorDirichletTransmissionPair) {
        const Eigen::Vector2d dir = v / r;
        const bool interior =
            direction_in_corner_wedge(ray0, patch.opening_angle, dir);
        if (interior) {
            const double theta = ccw_angle(ray0, dir);
            return polar_harmonic_value(
                r, theta, mode, 0.0, mode.interior_sine_factor);
        }
        const double theta = ccw_angle(ray1, dir);
        return polar_harmonic_value(
            r, theta, mode, -1.0, mode.exterior_sine_factor);
    }
    if (mode.model
        == CornerPatchSingularModel2D::InteriorNeumannTransmissionPair) {
        const Eigen::Vector2d dir = v / r;
        const bool interior =
            direction_in_corner_wedge(ray0, patch.opening_angle, dir);
        if (interior) {
            const double theta = ccw_angle(ray0, dir);
            return polar_harmonic_value(r,
                                        theta,
                                        mode,
                                        mode.interior_cos_factor,
                                        mode.interior_sine_factor);
        }
        const double theta = ccw_angle(ray1, dir);
        return polar_harmonic_value(r,
                                    theta,
                                    mode,
                                    mode.exterior_cos_factor,
                                    mode.exterior_sine_factor);
    }
    const double theta =
        corner_wedge_local_angle(ray0, ray1, patch.opening_angle, v / r);
    const double angular = mode.family == CornerPatchSingularFamily2D::Cosine
        ? std::cos(mode.lambda * theta)
        : std::sin(mode.lambda * theta);
    const double scale =
        std::isfinite(mode.radial_scale) && mode.radial_scale > 0.0
        ? mode.radial_scale
        : 1.0;
    return std::pow(r / scale, mode.lambda) * angular;
}

inline double evaluate_singular_modes_screened_operator(
    const CornerPatch2D& patch,
    const std::vector<CornerPatchSingularMode2D>& modes,
    Eigen::Vector2d x,
    double kappa)
{
    // Corner singular bases are harmonic; for (-Delta + kappa) only the
    // screened term remains.
    double value = 0.0;
    for (const CornerPatchSingularMode2D& mode : modes)
        value += mode.amplitude * evaluate_singular_mode_basis(patch, mode, x);
    return kappa * value;
}

inline Eigen::Vector2d evaluate_singular_mode_gradient(
    const CornerPatch2D& patch,
    const CornerPatchSingularMode2D& mode,
    const Eigen::Vector2d& x)
{
    const Eigen::Vector2d v = x - patch.center;
    const double r = v.norm();
    if (r <= 1.0e-14)
        return Eigen::Vector2d::Zero();

    const Eigen::Vector2d ray0 =
        normalized_or_throw(patch.ray_minus,
                            "corner patch ray_minus must be nonzero");
    const Eigen::Vector2d ray1 =
        normalized_or_throw(patch.ray_plus,
                            "corner patch ray_plus must be nonzero");
    const Eigen::Vector2d er = v / r;
    if (mode.model
        == CornerPatchSingularModel2D::InteriorDirichletTransmissionPair) {
        const bool interior =
            direction_in_corner_wedge(ray0, patch.opening_angle, er);
        if (interior) {
            const double theta = ccw_angle(ray0, er);
            return polar_harmonic_gradient(
                er, r, theta, mode, 0.0, mode.interior_sine_factor);
        }
        const double theta = ccw_angle(ray1, er);
        return polar_harmonic_gradient(
            er, r, theta, mode, -1.0, mode.exterior_sine_factor);
    }
    if (mode.model
        == CornerPatchSingularModel2D::InteriorNeumannTransmissionPair) {
        const bool interior =
            direction_in_corner_wedge(ray0, patch.opening_angle, er);
        if (interior) {
            const double theta = ccw_angle(ray0, er);
            return polar_harmonic_gradient(er,
                                           r,
                                           theta,
                                           mode,
                                           mode.interior_cos_factor,
                                           mode.interior_sine_factor);
        }
        const double theta = ccw_angle(ray1, er);
        return polar_harmonic_gradient(er,
                                       r,
                                       theta,
                                       mode,
                                       mode.exterior_cos_factor,
                                       mode.exterior_sine_factor);
    }
    const Eigen::Vector2d et(-er[1], er[0]);
    const double theta =
        corner_wedge_local_angle(ray0, ray1, patch.opening_angle, er);
    const double angle = mode.lambda * theta;
    double phi = 0.0;
    double dphi = 0.0;
    if (mode.family == CornerPatchSingularFamily2D::Cosine) {
        phi = std::cos(angle);
        dphi = -mode.lambda * std::sin(angle);
    } else {
        phi = std::sin(angle);
        dphi = mode.lambda * std::cos(angle);
    }
    const double scale =
        std::isfinite(mode.radial_scale) && mode.radial_scale > 0.0
        ? std::pow(mode.radial_scale, mode.lambda)
        : 1.0;
    const double radial =
        mode.lambda * std::pow(r, mode.lambda - 1.0) * phi / scale;
    const double angular = std::pow(r, mode.lambda) * dphi / (r * scale);
    return radial * er + angular * et;
}

inline CornerPatchSingularFamily2D singular_family_for_patch(
    const CornerPatch2D& patch,
    CornerPatchSingularFamilyPolicy2D policy)
{
    if ((policy == CornerPatchSingularFamilyPolicy2D::ExteriorCosineInteriorSine
         || policy == CornerPatchSingularFamilyPolicy2D::
                InteriorDirichletTransmissionPair)
        && patch.side == CornerPatchSide2D::Exterior) {
        return CornerPatchSingularFamily2D::Cosine;
    }
    return CornerPatchSingularFamily2D::Sine;
}

inline bool use_interior_dirichlet_transmission_pair(
    const CornerPatch2D& patch,
    CornerPatchSingularFamilyPolicy2D policy)
{
    return policy
               == CornerPatchSingularFamilyPolicy2D::
                      InteriorDirichletTransmissionPair
        && patch.side == CornerPatchSide2D::Interior
        && patch.opening_angle > kPi;
}

inline bool use_interior_neumann_transmission_pair(
    const CornerPatch2D& patch,
    CornerPatchSingularFamilyPolicy2D policy)
{
    return policy
               == CornerPatchSingularFamilyPolicy2D::
                      InteriorNeumannTransmissionPair
        && patch.side == CornerPatchSide2D::Interior
        && patch.opening_angle > kPi;
}

inline bool singular_fit_uses_normal_jump_trace(
    CornerPatchSingularFamilyPolicy2D policy)
{
    return policy
           == CornerPatchSingularFamilyPolicy2D::
                  InteriorNeumannTransmissionPair;
}

inline void configure_interior_dirichlet_transmission_pair(
    const CornerPatch2D& patch,
    CornerPatchSingularMode2D& mode)
{
    const double omega = patch.opening_angle;
    const double beta = kTwoPi - omega;
    if (!(omega > 0.0) || !(beta > 0.0)) {
        throw std::invalid_argument(
            "interior Dirichlet transmission pair requires an opening strictly between 0 and 2pi");
    }
    const int wedge_mode =
        static_cast<int>(std::llround(mode.lambda * omega / kPi));
    if (wedge_mode <= 0
        || std::abs(mode.lambda
                    - static_cast<double>(wedge_mode) * kPi / omega)
               > 2.0e-12) {
        throw std::invalid_argument(
            "interior Dirichlet transmission pair exponent is not a wedge eigenmode");
    }
    const double parity = (wedge_mode % 2 == 0) ? 1.0 : -1.0;
    const double denominator =
        1.0 - parity * std::cos(mode.lambda * beta);
    if (std::abs(denominator) < 1.0e-13) {
        throw std::invalid_argument(
            "degenerate interior Dirichlet transmission pair constants");
    }
    mode.model =
        CornerPatchSingularModel2D::InteriorDirichletTransmissionPair;
    mode.interior_sine_factor =
        std::sin(mode.lambda * beta) / denominator;
    mode.exterior_sine_factor = parity * mode.interior_sine_factor;
    mode.prev_jump_factor = 1.0;
    mode.next_jump_factor =
        std::cos(mode.lambda * beta)
        - mode.exterior_sine_factor * std::sin(mode.lambda * beta);
}

inline void configure_interior_neumann_transmission_pair(
    const CornerPatch2D& patch,
    CornerPatchSingularMode2D& mode)
{
    const double omega = patch.opening_angle;
    const double beta = kTwoPi - omega;
    if (!(omega > 0.0) || !(beta > 0.0)) {
        throw std::invalid_argument(
            "interior Neumann transmission pair requires an opening strictly between 0 and 2pi");
    }
    const int wedge_mode =
        static_cast<int>(std::llround(mode.lambda * omega / kPi));
    if (wedge_mode <= 0
        || std::abs(mode.lambda
                    - static_cast<double>(wedge_mode) * kPi / omega)
               > 2.0e-12) {
        throw std::invalid_argument(
            "interior Neumann transmission pair exponent is not a wedge eigenmode");
    }

    const double parity = (wedge_mode % 2 == 0) ? 1.0 : -1.0;
    const double sin_beta = std::sin(mode.lambda * beta);
    if (std::abs(sin_beta) < 1.0e-13) {
        throw std::invalid_argument(
            "degenerate interior Neumann transmission pair constants");
    }
    const double exterior_sine_over_interior_cos =
        (1.0 - parity * std::cos(mode.lambda * beta)) / sin_beta;
    if (std::abs(exterior_sine_over_interior_cos) < 1.0e-13) {
        throw std::invalid_argument(
            "zero previous-edge Neumann density normalization");
    }

    const double scale =
        std::isfinite(mode.radial_scale) && mode.radial_scale > 0.0
        ? mode.radial_scale
        : 1.0;
    const double interior_cos =
        -scale / (mode.lambda * exterior_sine_over_interior_cos);
    const double exterior_cos = parity * interior_cos;
    const double exterior_sine =
        interior_cos * exterior_sine_over_interior_cos;

    mode.model =
        CornerPatchSingularModel2D::InteriorNeumannTransmissionPair;
    mode.interior_cos_factor = interior_cos;
    mode.interior_sine_factor = 0.0;
    mode.exterior_cos_factor = exterior_cos;
    mode.exterior_sine_factor = exterior_sine;
    mode.prev_jump_factor = 1.0;
    mode.next_jump_factor =
        (mode.lambda / scale)
        * (-exterior_cos * std::sin(mode.lambda * beta)
           + exterior_sine * std::cos(mode.lambda * beta));
}

inline double corner_patch_singular_jump_factor(const CornerPatch2D& patch)
{
    return patch.side == CornerPatchSide2D::Exterior ? -1.0 : 1.0;
}

inline double fitting_sample_singular_factor(
    const CornerPatch2D& patch,
    const CornerPatchFittingSample2D& sample)
{
    return sample.interface_jump_value
        ? corner_patch_singular_jump_factor(patch)
        : 1.0;
}

inline bool point_is_on_previous_corner_edge(const CornerPatch2D& patch,
                                             const Eigen::Vector2d& x)
{
    const Eigen::Vector2d v = x - patch.center;
    const double r = v.norm();
    if (r <= 1.0e-14)
        return true;
    const Eigen::Vector2d dir = v / r;
    const Eigen::Vector2d next_ray =
        normalized_or_throw(patch.ray_minus,
                            "corner patch ray_minus must be nonzero");
    const Eigen::Vector2d prev_ray =
        normalized_or_throw(patch.ray_plus,
                            "corner patch ray_plus must be nonzero");
    const double next_distance =
        std::abs(std::atan2(cross2(next_ray, dir), next_ray.dot(dir)));
    const double prev_distance =
        std::abs(std::atan2(cross2(prev_ray, dir), prev_ray.dot(dir)));
    return prev_distance <= next_distance;
}

inline double singular_mode_jump_trace_basis(
    const CornerPatch2D& patch,
    const CornerPatchSingularMode2D& mode,
    bool previous_edge,
    double rho,
    const Eigen::Vector2d& x)
{
    if (mode.model
        == CornerPatchSingularModel2D::InteriorDirichletTransmissionPair) {
        const double factor =
            previous_edge ? mode.prev_jump_factor : mode.next_jump_factor;
        return factor * std::pow(std::max(0.0, rho), mode.lambda);
    }
    if (mode.model
        == CornerPatchSingularModel2D::InteriorNeumannTransmissionPair) {
        return 0.0;
    }
    return corner_patch_singular_jump_factor(patch)
         * evaluate_singular_mode_basis(patch, mode, x);
}

inline double singular_mode_normal_jump_trace_basis(
    const CornerPatch2D& patch,
    const CornerPatchSingularMode2D& mode,
    bool previous_edge,
    double rho,
    const Eigen::Vector2d& x,
    const Eigen::Vector2d& normal)
{
    if (mode.model
        == CornerPatchSingularModel2D::InteriorDirichletTransmissionPair) {
        return 0.0;
    }
    if (mode.model
        == CornerPatchSingularModel2D::InteriorNeumannTransmissionPair) {
        const double factor =
            previous_edge ? mode.prev_jump_factor : mode.next_jump_factor;
        return factor * std::pow(std::max(0.0, rho), mode.lambda - 1.0);
    }
    return corner_patch_singular_jump_factor(patch)
         * evaluate_singular_mode_gradient(patch, mode, x).dot(normal);
}

inline double singular_mode_normal_jump_trace_basis_at_point(
    const CornerPatch2D& patch,
    const CornerPatchSingularMode2D& mode,
    const Eigen::Vector2d& x,
    const Eigen::Vector2d& normal)
{
    const double r = (x - patch.center).norm();
    const double scale =
        std::isfinite(mode.radial_scale) && mode.radial_scale > 0.0
        ? mode.radial_scale
        : 1.0;
    const double rho = scale > 0.0 ? r / scale : r;
    return singular_mode_normal_jump_trace_basis(
        patch,
        mode,
        point_is_on_previous_corner_edge(patch, x),
        rho,
        x,
        normal);
}

inline double singular_mode_jump_trace_basis_at_point(
    const CornerPatch2D& patch,
    const CornerPatchSingularMode2D& mode,
    const Eigen::Vector2d& x)
{
    const double r = (x - patch.center).norm();
    const double scale =
        std::isfinite(mode.radial_scale) && mode.radial_scale > 0.0
        ? mode.radial_scale
        : 1.0;
    const double rho = scale > 0.0 ? r / scale : r;
    return singular_mode_jump_trace_basis(
        patch,
        mode,
        point_is_on_previous_corner_edge(patch, x),
        rho,
        x);
}

inline std::vector<CornerPatchSingularMode2D> build_singular_modes(
    const CornerPatch2D& patch,
    int max_sine_modes,
    CornerPatchSingularFamilyPolicy2D family_policy,
    double basis_radius = 1.0)
{
    if (max_sine_modes <= 0) {
        throw std::invalid_argument(
            "projected corner patch fitting requires at least one singular mode");
    }
    std::vector<CornerPatchSingularMode2D> modes;
    modes.reserve(static_cast<std::size_t>(max_sine_modes));
    const CornerPatchSingularFamily2D family =
        singular_family_for_patch(patch, family_policy);
    const bool dirichlet_transmission_pair =
        use_interior_dirichlet_transmission_pair(patch, family_policy);
    const bool neumann_transmission_pair =
        use_interior_neumann_transmission_pair(patch, family_policy);
    for (int m = 1; m <= max_sine_modes; ++m) {
        CornerPatchSingularMode2D mode;
        mode.lambda = static_cast<double>(m) * kPi / patch.opening_angle;
        mode.family = family;
        mode.radial_scale =
            std::isfinite(basis_radius) && basis_radius > 0.0 ? basis_radius
                                                              : 1.0;
        if (dirichlet_transmission_pair)
            configure_interior_dirichlet_transmission_pair(patch, mode);
        if (neumann_transmission_pair)
            configure_interior_neumann_transmission_pair(patch, mode);
        modes.push_back(mode);
    }
    return modes;
}

inline double safe_rank_tolerance(int rows,
                                  int cols,
                                  double sigma0,
                                  double rank_tol_rel);

struct EdgeP2TraceFitPoint2D {
    int point = -1;
    double distance = 0.0;
    bool previous_edge = false;
};

struct AlignedHalfStepTraceSample2D {
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    Eigen::Vector2d normal = Eigen::Vector2d::Zero();
    double distance = 0.0;
    bool previous_edge = false;
    double value = 0.0;
};

struct AlignedHalfStepTraceSamplePlan2D {
    int panel = -1;
    double local_s = 0.0;
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    Eigen::Vector2d normal = Eigen::Vector2d::Zero();
    double distance = 0.0;
    bool previous_edge = false;
};

struct AlignedHalfStepTraceGeometryPlan2D {
    std::vector<AlignedHalfStepTraceSamplePlan2D> samples;
    std::vector<int> interface_regular_panels;
};

struct AlignedHalfStepFitCache2D {
    int rows = 0;
    int cols = 0;
    int K = 0;
    int smooth_cols = 0;
    int rank = 0;
    bool normal_trace = false;
    double condition_number = 0.0;
    Eigen::MatrixXd A;
    Eigen::MatrixXd pinv;
    Eigen::VectorXd sigma;
};

inline std::string aligned_half_step_fit_cache_key_2d(
    const CornerPatch2D& patch,
    const CornerPatchCorrectionData2D& data,
    const std::vector<AlignedHalfStepTraceSample2D>& samples,
    const CornerPatchSingularFitOptions2D& fit_options,
    double fit_radius,
    double weight_shift,
    bool normal_trace,
    int rows,
    int cols,
    int smooth_cols)
{
    std::ostringstream out;
    out.precision(17);
    out << "patch=" << data.patch
        << "|corner=" << patch.corner
        << "|point=" << patch.point
        << "|center=" << patch.center[0] << "," << patch.center[1]
        << "|rays=" << patch.ray_minus[0] << "," << patch.ray_minus[1]
        << "," << patch.ray_plus[0] << "," << patch.ray_plus[1]
        << "|angle=" << patch.opening_angle
        << "|radius=" << patch.radius
        << "|fit_radius=" << fit_radius
        << "|weight_shift=" << weight_shift
        << "|normal=" << normal_trace
        << "|corner_row=" << fit_options.edge_aligned_include_corner_row
        << "|rank_tol=" << fit_options.rank_tol_rel
        << "|rows=" << rows
        << "|cols=" << cols
        << "|smooth_cols=" << smooth_cols
        << "|K=" << data.modes.size();
    for (const CornerPatchSingularMode2D& mode : data.modes) {
        out << "|mode=" << mode.lambda
            << "," << mode.radial_scale
            << "," << static_cast<int>(mode.family)
            << "," << static_cast<int>(mode.model)
            << "," << mode.interior_cos_factor
            << "," << mode.interior_sine_factor
            << "," << mode.exterior_cos_factor
            << "," << mode.exterior_sine_factor
            << "," << mode.prev_jump_factor
            << "," << mode.next_jump_factor;
    }
    for (const AlignedHalfStepTraceSample2D& sample : samples) {
        out << "|sample=" << sample.previous_edge
            << "," << sample.distance
            << "," << sample.point[0]
            << "," << sample.point[1]
            << "," << sample.normal[0]
            << "," << sample.normal[1];
    }
    return out.str();
}

inline AlignedHalfStepFitCache2D build_aligned_half_step_fit_cache_2d(
    const CornerPatch2D& patch,
    const CornerPatchCorrectionData2D& data,
    const std::vector<AlignedHalfStepTraceSample2D>& samples,
    const CornerPatchSingularFitOptions2D& fit_options,
    double fit_radius,
    double weight_shift,
    bool normal_trace,
    int rows,
    int cols,
    int smooth_cols)
{
    AlignedHalfStepFitCache2D cache;
    cache.rows = rows;
    cache.cols = cols;
    cache.K = static_cast<int>(data.modes.size());
    cache.smooth_cols = smooth_cols;
    cache.normal_trace = normal_trace;
    cache.A = Eigen::MatrixXd::Zero(rows, cols);

    int row = 0;
    for (const AlignedHalfStepTraceSample2D& item : samples) {
        const double rho = item.distance / fit_radius;
        const double w = 1.0 / ((weight_shift + rho) * (weight_shift + rho));
        const double sw = std::sqrt(w);
        if (normal_trace) {
            if (item.previous_edge) {
                cache.A(row, 0) = sw;
                cache.A(row, 1) = sw * rho;
            } else {
                cache.A(row, 2) = sw;
                cache.A(row, 3) = sw * rho;
            }
        } else {
            cache.A(row, 0) = sw;
            if (item.previous_edge) {
                cache.A(row, 1) = sw * rho;
                cache.A(row, 2) = sw * rho * rho;
            } else {
                cache.A(row, 3) = sw * rho;
                cache.A(row, 4) = sw * rho * rho;
            }
        }
        for (int k = 0; k < cache.K; ++k) {
            cache.A(row, smooth_cols + k) =
                sw
                * (normal_trace
                       ? singular_mode_normal_jump_trace_basis(
                             patch,
                             data.modes[static_cast<std::size_t>(k)],
                             item.previous_edge,
                             rho,
                             item.point,
                             item.normal)
                       : singular_mode_jump_trace_basis(
                             patch,
                             data.modes[static_cast<std::size_t>(k)],
                             item.previous_edge,
                             rho,
                             item.point));
        }
        ++row;
    }

    if (fit_options.edge_aligned_include_corner_row) {
        const double corner_sw = 1.0 / weight_shift;
        cache.A(row, 0) = corner_sw;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        cache.A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    cache.sigma = svd.singularValues();
    const double tol = cache.sigma.size() == 0
        ? 0.0
        : safe_rank_tolerance(rows,
                              cols,
                              cache.sigma[0],
                              fit_options.rank_tol_rel);
    for (int i = 0; i < cache.sigma.size(); ++i) {
        if (cache.sigma[i] > tol)
            ++cache.rank;
    }
    if (cache.rank < cols && fit_options.throw_on_rank_deficient) {
        std::ostringstream out;
        out << "aligned half-step singular coefficient fitting rank deficient"
            << " patch=" << data.patch
            << " corner=" << patch.corner
            << " rows=" << rows
            << " cols=" << cols
            << " rank=" << cache.rank;
        throw std::invalid_argument(out.str());
    }

    cache.pinv = Eigen::MatrixXd::Zero(cols, rows);
    for (int i = 0; i < cache.rank; ++i) {
        const double inv_sigma = cache.sigma[i] > tol
            ? 1.0 / cache.sigma[i]
            : 0.0;
        cache.pinv += inv_sigma
            * svd.matrixV().col(i)
            * svd.matrixU().col(i).transpose();
    }
    cache.condition_number =
        cache.rank > 0 ? cache.sigma[0] / cache.sigma[cache.rank - 1] : 0.0;
    return cache;
}

inline bool sample_p2_trace_geometry_at_edge_distance(
    const Interface2D& iface,
    const CornerData2D& corner,
    bool previous_edge,
    double target_distance,
    AlignedHalfStepTraceSamplePlan2D& sample)
{
    if (!(target_distance > 0.0))
        return false;

    const int incident_panel =
        previous_edge ? corner.prev_panel : corner.next_panel;
    int local_corner =
        local_index_of_point(iface, incident_panel, corner.point);
    const Eigen::Vector2d corner_point =
        iface.points().row(corner.point).transpose();
    if (local_corner < 0) {
        double best_distance = std::numeric_limits<double>::infinity();
        int best_local = -1;
        for (int local = 0; local < iface.points_per_panel(); ++local) {
            const int q = iface.point_index(incident_panel, local);
            const double distance =
                (iface.points().row(q).transpose() - corner_point).norm();
            if (distance < best_distance) {
                best_distance = distance;
                best_local = local;
            }
        }
        if (best_distance <= 1.0e-10)
            local_corner = best_local;
    }
    if (local_corner != 0 && local_corner != 1 && local_corner != 2) {
        throw std::invalid_argument(
            "aligned half-step edge fitting expected the corner on an incident panel");
    }

    auto set_sample = [&](int panel, double s) {
        sample.panel = panel;
        sample.local_s = s;
        sample.point = geometry2d::panel_point(iface, panel, s);
        sample.normal = geometry2d::panel_normal(iface, panel, s);
        sample.distance = target_distance;
        sample.previous_edge = previous_edge;
    };

    if (local_corner == 1) {
        const Eigen::Vector2d center = corner_point;
        const Eigen::Vector2d dir =
            normalized_or_throw(previous_edge ? -corner.tangent_minus
                                              :  corner.tangent_plus,
                                "aligned half-step edge fitting tangent must be nonzero");
        const int end_candidates[2] = {0, 2};
        int end_local = 0;
        double best_dot = -std::numeric_limits<double>::infinity();
        for (int candidate : end_candidates) {
            const int point = iface.point_index(incident_panel, candidate);
            const Eigen::Vector2d v =
                iface.points().row(point).transpose() - center;
            const double len = v.norm();
            const double dot = len > 1.0e-14 ? (v / len).dot(dir) : -1.0;
            if (dot > best_dot) {
                best_dot = dot;
                end_local = candidate;
            }
        }
        const int endpoint = iface.point_index(incident_panel, end_local);
        const Eigen::Vector2d endpoint_x =
            iface.points().row(endpoint).transpose();
        const double first_length = (endpoint_x - center).norm();
        if (!(first_length > 1.0e-14))
            return false;
        if (target_distance <= first_length + 1.0e-13) {
            const double t = std::min(
                1.0, std::max(0.0, target_distance / first_length));
            const double end_s = end_local == 0 ? -1.0 : 1.0;
            set_sample(incident_panel, end_s * t);
            return true;
        }

        const bool moving_forward = end_local == 2;
        int panel = connected_panel_along_endpoint(
            iface, incident_panel, endpoint, moving_forward);
        if (panel < 0)
            return false;
        double covered = first_length;
        for (int guard = 0; guard < iface.num_panels(); ++guard) {
            const int start_local = moving_forward ? 0 : 2;
            const int end_local_next = moving_forward ? 2 : 0;
            const int start_point = iface.point_index(panel, start_local);
            const int end_point = iface.point_index(panel, end_local_next);
            const double panel_length =
                (iface.points().row(end_point).transpose()
                 - iface.points().row(start_point).transpose())
                    .norm();
            if (!(panel_length > 1.0e-14))
                return false;
            const double remaining = target_distance - covered;
            if (remaining <= panel_length + 1.0e-13) {
                const double t =
                    std::min(1.0, std::max(0.0, remaining / panel_length));
                const double s = moving_forward ? -1.0 + 2.0 * t
                                                : 1.0 - 2.0 * t;
                set_sample(panel, s);
                return true;
            }
            covered += panel_length;
            const int outer_point = iface.point_index(panel, end_local_next);
            const int next_panel =
                connected_panel_along_endpoint(
                    iface, panel, outer_point, moving_forward);
            if (next_panel < 0)
                return false;
            panel = next_panel;
        }
        return false;
    }

    const bool moving_forward = local_corner == 0;
    int panel = incident_panel;
    double covered = 0.0;
    for (int guard = 0; guard < iface.num_panels(); ++guard) {
        const int start_local = moving_forward ? 0 : 2;
        const int end_local = moving_forward ? 2 : 0;
        const int start_point = iface.point_index(panel, start_local);
        const int end_point = iface.point_index(panel, end_local);
        const double panel_length =
            (iface.points().row(end_point).transpose()
             - iface.points().row(start_point).transpose())
                .norm();
        if (!(panel_length > 1.0e-14))
            return false;

        const double remaining = target_distance - covered;
        if (remaining <= panel_length + 1.0e-13) {
            const double t =
                std::min(1.0, std::max(0.0, remaining / panel_length));
            const double s = moving_forward ? -1.0 + 2.0 * t
                                            : 1.0 - 2.0 * t;
            set_sample(panel, s);
            return true;
        }

        covered += panel_length;
        const int outer_point = iface.point_index(panel, end_local);
        const int next_panel =
            connected_panel_along_endpoint(iface,
                                           panel,
                                           outer_point,
                                           moving_forward);
        if (next_panel < 0)
            return false;
        panel = next_panel;
    }
    return false;
}

inline std::string aligned_half_step_geometry_plan_cache_key_2d(
    const Interface2D& iface,
    const CornerData2D& corner,
    double fit_radius,
    int max_points,
    double step)
{
    const Eigen::Vector2d center =
        iface.points().row(corner.point).transpose();
    std::ostringstream out;
    out.precision(17);
    out << "iface=" << static_cast<const void*>(&iface)
        << "|points=" << iface.num_points()
        << "|panels=" << iface.num_panels()
        << "|ppp=" << iface.points_per_panel()
        << "|layout=" << static_cast<int>(iface.panel_node_layout())
        << "|corner=" << corner.point
        << "|prev=" << corner.prev_panel
        << "|next=" << corner.next_panel
        << "|center=" << center[0] << "," << center[1]
        << "|fit_radius=" << fit_radius
        << "|max_points=" << max_points
        << "|step=" << step;
    return out.str();
}

inline AlignedHalfStepTraceGeometryPlan2D
build_aligned_half_step_trace_geometry_plan(
    const Interface2D& iface,
    const CornerData2D& corner,
    double fit_radius,
    int max_points,
    double step)
{
    AlignedHalfStepTraceGeometryPlan2D plan;
    plan.samples.reserve(static_cast<std::size_t>(2 * max_points));
    const Eigen::Vector2d center =
        iface.points().row(corner.point).transpose();
    auto add_sample_at_point = [&](int point,
                                   double distance,
                                   bool previous_edge) -> bool {
        for (int panel = 0; panel < iface.num_panels(); ++panel) {
            for (int local = 0; local < iface.points_per_panel(); ++local) {
                if (iface.point_index(panel, local) != point)
                    continue;
                AlignedHalfStepTraceSamplePlan2D sample;
                sample.panel = panel;
                sample.local_s =
                    geometry2d::kP2NodeS[static_cast<std::size_t>(local)];
                sample.point = iface.points().row(point).transpose();
                sample.normal = iface.normals().row(point).transpose();
                sample.distance = distance;
                sample.previous_edge = previous_edge;
                plan.samples.push_back(sample);
                return true;
            }
        }
        return false;
    };
    for (int side = 0; side < 2; ++side) {
        const bool previous_edge = side == 0;
        const int incident_panel =
            previous_edge ? corner.prev_panel : corner.next_panel;
        const int local_corner =
            local_index_of_point(iface, incident_panel, corner.point);
        if (local_corner != 0 && local_corner != 1 && local_corner != 2) {
            const Eigen::Vector2d dir =
                normalized_or_throw(previous_edge ? -corner.tangent_minus
                                                  :  corner.tangent_plus,
                                    "aligned half-step edge fitting tangent must be nonzero");
            std::vector<std::pair<double, int>> side_points;
            for (int q = 0; q < iface.num_points(); ++q) {
                if (q == corner.point)
                    continue;
                const Eigen::Vector2d pt = iface.points().row(q).transpose();
                const Eigen::Vector2d v = pt - center;
                const double distance = v.norm();
                if (!(distance > 1.0e-13)
                    || distance > fit_radius + 1.0e-13) {
                    continue;
                }
                const Eigen::Vector2d unit = v / distance;
                if (unit.dot(dir) <= 1.0 - 1.0e-10)
                    continue;
                if (std::abs(cross2(unit, dir)) > 1.0e-8)
                    continue;
                const double half_step_index = distance / step - 0.5;
                if (std::abs(half_step_index - std::round(half_step_index))
                    > 1.0e-8) {
                    continue;
                }
                side_points.emplace_back(distance, q);
            }
            std::sort(side_points.begin(), side_points.end());
            if (static_cast<int>(side_points.size()) > max_points)
                side_points.resize(static_cast<std::size_t>(max_points));
            for (const auto& item : side_points)
                add_sample_at_point(item.second, item.first, previous_edge);
            continue;
        }
        for (int k = 0; k < max_points; ++k) {
            const double distance = (static_cast<double>(k) + 0.5) * step;
            if (distance > fit_radius + 1.0e-13)
                break;
            AlignedHalfStepTraceSamplePlan2D sample;
            if (!sample_p2_trace_geometry_at_edge_distance(
                    iface, corner, previous_edge, distance, sample)) {
                break;
            }
            plan.samples.push_back(sample);
        }
    }
    plan.interface_regular_panels =
        collect_edge_panel_ids_within_radius(iface, corner, fit_radius);
    return plan;
}

inline const AlignedHalfStepTraceGeometryPlan2D&
aligned_half_step_trace_geometry_plan_2d(
    const Interface2D& iface,
    const CornerData2D& corner,
    double fit_radius,
    int max_points,
    double step)
{
    static thread_local std::unordered_map<std::string,
                                           AlignedHalfStepTraceGeometryPlan2D>
        plan_cache_by_key;
    const std::string cache_key =
        aligned_half_step_geometry_plan_cache_key_2d(
            iface, corner, fit_radius, max_points, step);
    auto it = plan_cache_by_key.find(cache_key);
    if (it == plan_cache_by_key.end()) {
        it = plan_cache_by_key
                 .emplace(cache_key,
                          build_aligned_half_step_trace_geometry_plan(
                              iface, corner, fit_radius, max_points, step))
                 .first;
    }
    return it->second;
}

inline std::vector<AlignedHalfStepTraceSample2D>
instantiate_aligned_half_step_trace_samples(
    const Interface2D& iface,
    const AlignedHalfStepTraceGeometryPlan2D& plan,
    const Eigen::VectorXd& values)
{
    std::vector<AlignedHalfStepTraceSample2D> samples;
    samples.reserve(plan.samples.size());
    for (const AlignedHalfStepTraceSamplePlan2D& item : plan.samples) {
        AlignedHalfStepTraceSample2D sample;
        sample.point = item.point;
        sample.normal = item.normal;
        sample.distance = item.distance;
        sample.previous_edge = item.previous_edge;
        sample.value =
            geometry2d::panel_scalar(iface, item.panel, values, item.local_s);
        samples.push_back(sample);
    }
    return samples;
}

inline std::size_t prepare_aligned_half_step_trace_geometry_plan_2d(
    const Interface2D& iface,
    const CornerPatch2D& patch,
    const CornerPatchSingularFitOptions2D& fit_options)
{
    const CornerData2D& corner = iface.corner(patch.corner);
    const double fit_radius = fit_options.edge_p2_fit_radius > 0.0
        ? fit_options.edge_p2_fit_radius
        : (fit_options.basis_radius > 0.0 ? fit_options.basis_radius
                                          : patch.radius);
    const int max_points = fit_options.edge_p2_max_points_per_direction;
    const double step = fit_options.edge_aligned_step;
    if (!(fit_radius > 0.0) || !(step > 0.0) || max_points <= 0)
        return 0;
    const AlignedHalfStepTraceGeometryPlan2D& plan =
        aligned_half_step_trace_geometry_plan_2d(
            iface, corner, fit_radius, max_points, step);
    return plan.samples.size();
}

inline void fit_singular_modes_from_aligned_half_step_trace(
    const Interface2D& iface,
    const CornerPatch2D& patch,
    CornerPatchCorrectionData2D& data,
    const Eigen::VectorXd& trace_values,
    const CornerPatchSingularFitOptions2D& fit_options)
{
    const CornerData2D& corner = iface.corner(patch.corner);
    const double fit_radius = fit_options.edge_p2_fit_radius > 0.0
        ? fit_options.edge_p2_fit_radius
        : (fit_options.basis_radius > 0.0 ? fit_options.basis_radius
                                          : patch.radius);
    const int max_points = fit_options.edge_p2_max_points_per_direction;
    const double step = fit_options.edge_aligned_step;
    const double weight_shift = fit_options.edge_p2_weight_shift;
    if (!(fit_radius > 0.0) || !(step > 0.0) || max_points <= 0
        || !(weight_shift > 0.0)) {
        throw std::invalid_argument(
            "aligned half-step singular coefficient fitting requires positive radius, step, point count, and weight shift");
    }
    const bool normal_trace =
        singular_fit_uses_normal_jump_trace(fit_options.family_policy);
    if (normal_trace && fit_options.edge_aligned_include_corner_row) {
        throw std::invalid_argument(
            "Neumann singular edge fitting cannot use a corner row because the corner normal is multi-valued");
    }

    const AlignedHalfStepTraceGeometryPlan2D& geometry_plan =
        aligned_half_step_trace_geometry_plan_2d(
            iface, corner, fit_radius, max_points, step);
    std::vector<AlignedHalfStepTraceSample2D> samples =
        instantiate_aligned_half_step_trace_samples(
            iface, geometry_plan, trace_values);
    data.interface_regular_panels = geometry_plan.interface_regular_panels;

    const int K = static_cast<int>(data.modes.size());
    const int smooth_cols = normal_trace ? 4 : 5;
    const int cols = smooth_cols + K;
    const int corner_rows =
        fit_options.edge_aligned_include_corner_row ? 1 : 0;
    const int rows = static_cast<int>(samples.size()) + corner_rows;
    if (rows < cols) {
        throw std::invalid_argument(
            "aligned half-step singular coefficient fitting has too few boundary samples");
    }

    Eigen::VectorXd b = Eigen::VectorXd::Zero(rows);
    int row = 0;
    for (const AlignedHalfStepTraceSample2D& item : samples) {
        const double rho = item.distance / fit_radius;
        const double w = 1.0 / ((weight_shift + rho) * (weight_shift + rho));
        const double sw = std::sqrt(w);
        b[row] = sw * item.value;
        ++row;
    }

    if (fit_options.edge_aligned_include_corner_row) {
        const double corner_sw = 1.0 / weight_shift;
        b[row] = corner_sw * trace_values[corner.point];
    }

    static thread_local std::unordered_map<std::string,
                                           AlignedHalfStepFitCache2D>
        fit_cache_by_key;
    const std::string cache_key =
        aligned_half_step_fit_cache_key_2d(patch,
                                           data,
                                           samples,
                                           fit_options,
                                           fit_radius,
                                           weight_shift,
                                           normal_trace,
                                           rows,
                                           cols,
                                           smooth_cols);
    auto cache_it = fit_cache_by_key.find(cache_key);
    if (cache_it == fit_cache_by_key.end()) {
        cache_it = fit_cache_by_key
                       .emplace(cache_key,
                                build_aligned_half_step_fit_cache_2d(
                                    patch,
                                    data,
                                    samples,
                                    fit_options,
                                    fit_radius,
                                    weight_shift,
                                    normal_trace,
                                    rows,
                                    cols,
                                    smooth_cols))
                       .first;
    }
    const AlignedHalfStepFitCache2D& cache = cache_it->second;
    Eigen::VectorXd coeff = cache.pinv * b;
    for (int k = 0; k < K; ++k)
        data.modes[k].amplitude = coeff[smooth_cols + k];
    data.amplitude = K > 0 ? data.modes.front().amplitude : 0.0;
    data.lambda = K > 0 ? data.modes.front().lambda : 0.0;

    data.diagnostics.rows = rows;
    data.diagnostics.value_rows = normal_trace ? 0 : rows;
    data.diagnostics.normal_rows = normal_trace ? rows : 0;
    data.diagnostics.num_singular_modes = K;
    data.diagnostics.num_nuisance_modes = smooth_cols;
    data.diagnostics.singular_rank = cache.rank;
    data.diagnostics.nuisance_rank = smooth_cols;
    data.diagnostics.condition_number = cache.condition_number;
    data.diagnostics.relative_residual =
        (cache.A * coeff - b).norm() / std::max(b.norm(), 1.0e-30);
    double coeff_norm_sq = 0.0;
    for (const CornerPatchSingularMode2D& mode : data.modes)
        coeff_norm_sq += mode.amplitude * mode.amplitude;
    data.diagnostics.coeff_norm = std::sqrt(coeff_norm_sq);
}

inline std::vector<EdgeP2TraceFitPoint2D> collect_edge_p2_trace_fit_points(
    const Interface2D& iface,
    const CornerData2D& corner,
    bool previous_edge,
    double fit_radius,
    int max_points)
{
    if (iface.points_per_panel() != 3
        || iface.panel_node_layout() != PanelNodeLayout2D::QuadraticLagrange) {
        throw std::invalid_argument(
            "edge P2 singular coefficient fitting requires P2 quadratic panels");
    }
    if (!(fit_radius > 0.0) || max_points <= 0) {
        throw std::invalid_argument(
            "edge P2 singular coefficient fitting requires a positive radius and point count");
    }
    int extra_points = 0;
    if (const char* extra_env =
            std::getenv("KFBIM_EDGE_P2_FIT_EXTRA_POINTS")) {
        extra_points = std::max(0, std::atoi(extra_env));
    }
    const int target_points = max_points + extra_points;

    const int incident_panel =
        previous_edge ? corner.prev_panel : corner.next_panel;
    const int local_corner =
        local_index_of_point(iface, incident_panel, corner.point);
    if (local_corner != 0 && local_corner != 2) {
        throw std::invalid_argument(
            "edge P2 singular coefficient fitting expected the corner as a panel endpoint");
    }

    const bool moving_forward = local_corner == 0;
    const Eigen::Vector2d center =
        iface.points().row(corner.point).transpose();
    std::vector<EdgeP2TraceFitPoint2D> points;
    int panel = incident_panel;
    for (int guard = 0; guard < iface.num_panels(); ++guard) {
        const int outer_local = moving_forward ? 2 : 0;
        const int local_ids[2] = {1, outer_local};
        bool any_inside = false;
        for (int local : local_ids) {
            const int point = iface.point_index(panel, local);
            if (point == corner.point)
                continue;
            const double distance =
                (iface.points().row(point).transpose() - center).norm();
            const bool inside_fit_radius =
                distance > 1.0e-13 && distance <= fit_radius + 1.0e-13;
            const bool use_extra_sample =
                extra_points > 0 && distance > fit_radius + 1.0e-13
                && static_cast<int>(points.size()) < target_points;
            if (inside_fit_radius || use_extra_sample) {
                points.push_back({point, distance, previous_edge});
                any_inside = true;
            }
        }

        const int outer_point = iface.point_index(panel, outer_local);
        const double outer_distance =
            (iface.points().row(outer_point).transpose() - center).norm();
        if (outer_distance > fit_radius + 1.0e-13 && !any_inside)
            break;

        const int next_panel =
            connected_panel_along_endpoint(iface,
                                           panel,
                                           outer_point,
                                           moving_forward);
        if (next_panel < 0)
            break;
        panel = next_panel;
        if (outer_distance > fit_radius + 1.0e-13)
            break;
    }

    std::sort(points.begin(),
              points.end(),
              [](const EdgeP2TraceFitPoint2D& a,
                 const EdgeP2TraceFitPoint2D& b) {
                  if (a.distance != b.distance)
                      return a.distance < b.distance;
                  return a.point < b.point;
              });
    points.erase(std::unique(points.begin(),
                             points.end(),
                             [](const EdgeP2TraceFitPoint2D& a,
                                const EdgeP2TraceFitPoint2D& b) {
                                 return a.point == b.point;
                             }),
                 points.end());
    if (static_cast<int>(points.size()) > target_points)
        points.resize(static_cast<std::size_t>(target_points));
    return points;
}

inline void fit_singular_modes_from_edge_p2_trace(
    const Interface2D& iface,
    const CornerPatch2D& patch,
    CornerPatchCorrectionData2D& data,
    const Eigen::VectorXd& u_jump,
    const CornerPatchSingularFitOptions2D& fit_options)
{
    if (fit_options.edge_aligned_half_step_sampling) {
        fit_singular_modes_from_aligned_half_step_trace(
            iface, patch, data, u_jump, fit_options);
        return;
    }
    if (singular_fit_uses_normal_jump_trace(fit_options.family_policy)) {
        throw std::invalid_argument(
            "Neumann singular edge fitting requires edge_aligned_half_step_sampling=true");
    }

    const CornerData2D& corner = iface.corner(patch.corner);
    const double fit_radius = fit_options.edge_p2_fit_radius > 0.0
        ? fit_options.edge_p2_fit_radius
        : (fit_options.basis_radius > 0.0 ? fit_options.basis_radius
                                          : patch.radius);
    const int max_points = fit_options.edge_p2_max_points_per_direction;
    const double weight_shift = fit_options.edge_p2_weight_shift;
    if (!(weight_shift > 0.0)) {
        throw std::invalid_argument(
            "edge P2 singular coefficient fitting requires positive weight shift");
    }

    std::vector<EdgeP2TraceFitPoint2D> prev =
        collect_edge_p2_trace_fit_points(
            iface, corner, true, fit_radius, max_points);
    std::vector<EdgeP2TraceFitPoint2D> next =
        collect_edge_p2_trace_fit_points(
            iface, corner, false, fit_radius, max_points);
    data.interface_regular_panels =
        collect_edge_panel_ids_within_radius(iface, corner, fit_radius);
    std::vector<EdgeP2TraceFitPoint2D> points;
    points.reserve(prev.size() + next.size());
    points.insert(points.end(), prev.begin(), prev.end());
    points.insert(points.end(), next.begin(), next.end());

    const char* debug_patch_env =
        std::getenv("KFBIM_EDGE_P2_FIT_DEBUG_PATCH");
    const bool debug_patch =
        debug_patch_env != nullptr
        && std::atoi(debug_patch_env) == data.patch;

    const int K = static_cast<int>(data.modes.size());
    const int cols = 5 + K;
    const int rows = static_cast<int>(points.size()) + 1;
    if (rows < cols) {
        throw std::invalid_argument(
            "edge P2 singular coefficient fitting has too few boundary samples");
    }

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows, cols);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(rows);
    if (debug_patch) {
        const Eigen::Vector2d corner_x =
            iface.points().row(corner.point).transpose();
        std::printf("EDGE_P2_FIT_DEBUG patch=%d corner=%d corner_point=%d center=(%.16e, %.16e) fit_radius=%.16e patch_radius=%.16e opening=%.16e max_points=%d weight_shift=%.16e K=%d rows=%d cols=%d\n",
                    data.patch,
                    patch.corner,
                    corner.point,
                    corner_x[0],
                    corner_x[1],
                    fit_radius,
                    patch.radius,
                    patch.opening_angle,
                    max_points,
                    weight_shift,
                    K,
                    rows,
                    cols);
        std::printf("EDGE_P2_FIT_DEBUG patch=%d rays minus=(%.16e, %.16e) plus=(%.16e, %.16e)\n",
                    data.patch,
                    patch.ray_minus[0],
                    patch.ray_minus[1],
                    patch.ray_plus[0],
                    patch.ray_plus[1]);
    }
    int row = 0;
    for (const EdgeP2TraceFitPoint2D& item : points) {
        const Eigen::Vector2d x =
            iface.points().row(item.point).transpose();
        const double rho = item.distance / fit_radius;
        const double w = 1.0 / ((weight_shift + rho) * (weight_shift + rho));
        const double sw = std::sqrt(w);
        A(row, 0) = sw;
        if (item.previous_edge) {
            A(row, 1) = sw * rho;
            A(row, 2) = sw * rho * rho;
        } else {
            A(row, 3) = sw * rho;
            A(row, 4) = sw * rho * rho;
        }
        for (int k = 0; k < K; ++k) {
            A(row, 5 + k) =
                sw * singular_mode_jump_trace_basis(patch,
                                                    data.modes[k],
                                                    item.previous_edge,
                                                    rho,
                                                    x);
        }
        b[row] = sw * u_jump[item.point];
        if (debug_patch) {
            const Eigen::Vector2d center =
                iface.points().row(corner.point).transpose();
            const Eigen::Vector2d v = x - center;
            const double r = v.norm();
            const double theta = r > 1.0e-14
                ? corner_wedge_local_angle(patch.ray_minus.normalized(),
                                           patch.ray_plus.normalized(),
                                           patch.opening_angle,
                                           v / r)
                : 0.0;
            std::printf("EDGE_P2_FIT_DEBUG_ROW patch=%d row=%d side=%s point=%d x=(%.16e, %.16e) dist=%.16e rho=%.16e theta=%.16e weight=%.16e sw=%.16e u=%.16e A_unweighted=[",
                        data.patch,
                        row,
                        item.previous_edge ? "prev" : "next",
                        item.point,
                        x[0],
                        x[1],
                        item.distance,
                        rho,
                        theta,
                        w,
                        sw,
                        u_jump[item.point]);
            for (int c = 0; c < cols; ++c) {
                const double unweighted = sw != 0.0 ? A(row, c) / sw : A(row, c);
                std::printf("%s%.16e", c == 0 ? "" : " ", unweighted);
            }
            std::printf("] A_weighted=[");
            for (int c = 0; c < cols; ++c)
                std::printf("%s%.16e", c == 0 ? "" : " ", A(row, c));
            std::printf("] b_weighted=%.16e\n", b[row]);
        }
        ++row;
    }

    const double corner_weight =
        1.0 / (weight_shift * weight_shift);
    const double corner_sw = std::sqrt(corner_weight);
    A(row, 0) = corner_sw;
    b[row] = corner_sw * u_jump[corner.point];
    if (debug_patch) {
        const Eigen::Vector2d corner_x =
            iface.points().row(corner.point).transpose();
        std::printf("EDGE_P2_FIT_DEBUG_ROW patch=%d row=%d side=corner point=%d x=(%.16e, %.16e) dist=0.0000000000000000e+00 rho=0.0000000000000000e+00 theta=0.0000000000000000e+00 weight=%.16e sw=%.16e u=%.16e A_unweighted=[",
                    data.patch,
                    row,
                    corner.point,
                    corner_x[0],
                    corner_x[1],
                    corner_weight,
                    corner_sw,
                    u_jump[corner.point]);
        for (int c = 0; c < cols; ++c) {
            const double unweighted =
                corner_sw != 0.0 ? A(row, c) / corner_sw : A(row, c);
            std::printf("%s%.16e", c == 0 ? "" : " ", unweighted);
        }
        std::printf("] A_weighted=[");
        for (int c = 0; c < cols; ++c)
            std::printf("%s%.16e", c == 0 ? "" : " ", A(row, c));
        std::printf("] b_weighted=%.16e\n", b[row]);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd sigma = svd.singularValues();
    const double tol = sigma.size() == 0
        ? 0.0
        : safe_rank_tolerance(rows,
                              cols,
                              sigma[0],
                              fit_options.rank_tol_rel);
    int rank = 0;
    for (int i = 0; i < sigma.size(); ++i) {
        if (sigma[i] > tol)
            ++rank;
    }
    if (rank < cols && fit_options.throw_on_rank_deficient) {
        throw std::invalid_argument(
            "edge P2 singular coefficient fitting rank deficient");
    }

    Eigen::VectorXd coeff = Eigen::VectorXd::Zero(cols);
    for (int i = 0; i < rank; ++i) {
        const double inv_sigma = sigma[i] > tol ? 1.0 / sigma[i] : 0.0;
        coeff += inv_sigma * svd.matrixV().col(i)
               * (svd.matrixU().col(i).dot(b));
    }
    for (int k = 0; k < K; ++k)
        data.modes[k].amplitude = coeff[5 + k];
    data.amplitude = K > 0 ? data.modes.front().amplitude : 0.0;
    data.lambda = K > 0 ? data.modes.front().lambda : 0.0;

    data.diagnostics.rows = rows;
    data.diagnostics.value_rows = rows;
    data.diagnostics.normal_rows = 0;
    data.diagnostics.num_singular_modes = K;
    data.diagnostics.num_nuisance_modes = 5;
    data.diagnostics.singular_rank = rank;
    data.diagnostics.nuisance_rank = 5;
    data.diagnostics.condition_number =
        rank > 0 ? sigma[0] / sigma[rank - 1] : 0.0;
    data.diagnostics.relative_residual =
        (A * coeff - b).norm() / std::max(b.norm(), 1.0e-30);
    if (debug_patch) {
        std::printf("EDGE_P2_FIT_DEBUG_RESULT patch=%d rank=%d tol=%.16e sigma=[",
                    data.patch,
                    rank,
                    tol);
        for (int i = 0; i < sigma.size(); ++i)
            std::printf("%s%.16e", i == 0 ? "" : " ", sigma[i]);
        std::printf("] coeff=[");
        for (int c = 0; c < coeff.size(); ++c)
            std::printf("%s%.16e", c == 0 ? "" : " ", coeff[c]);
        std::printf("] relres=%.16e\n", data.diagnostics.relative_residual);
    }
    double coeff_norm_sq = 0.0;
    for (const CornerPatchSingularMode2D& mode : data.modes)
        coeff_norm_sq += mode.amplitude * mode.amplitude;
    data.diagnostics.coeff_norm = std::sqrt(coeff_norm_sq);
}

inline int harmonic_nuisance_count(int degree)
{
    if (degree < 0 || degree > 3) {
        throw std::invalid_argument(
            "projected corner patch fitting nuisance_degree must be in [0, 3]");
    }
    return degree == 0 ? 1 : 1 + 2 * degree;
}

inline Eigen::VectorXd harmonic_nuisance_values(const CornerPatch2D& patch,
                                                int degree,
                                                const Eigen::Vector2d& x)
{
    const int count = harmonic_nuisance_count(degree);
    Eigen::VectorXd values(count);
    const double X = (x[0] - patch.center[0]) / patch.radius;
    const double Y = (x[1] - patch.center[1]) / patch.radius;
    int col = 0;
    values[col++] = 1.0;
    if (degree >= 1) {
        values[col++] = X;
        values[col++] = Y;
    }
    if (degree >= 2) {
        values[col++] = X * X - Y * Y;
        values[col++] = 2.0 * X * Y;
    }
    if (degree >= 3) {
        values[col++] = X * X * X - 3.0 * X * Y * Y;
        values[col++] = 3.0 * X * X * Y - Y * Y * Y;
    }
    return values;
}

inline Eigen::MatrixX2d harmonic_nuisance_gradients(const CornerPatch2D& patch,
                                                    int degree,
                                                    const Eigen::Vector2d& x)
{
    const int count = harmonic_nuisance_count(degree);
    Eigen::MatrixX2d grads(count, 2);
    grads.setZero();
    const double X = (x[0] - patch.center[0]) / patch.radius;
    const double Y = (x[1] - patch.center[1]) / patch.radius;
    const double invR = 1.0 / patch.radius;
    int col = 0;
    grads.row(col++) = Eigen::RowVector2d(0.0, 0.0);
    if (degree >= 1) {
        grads.row(col++) = Eigen::RowVector2d(invR, 0.0);
        grads.row(col++) = Eigen::RowVector2d(0.0, invR);
    }
    if (degree >= 2) {
        grads.row(col++) = Eigen::RowVector2d(2.0 * X * invR,
                                              -2.0 * Y * invR);
        grads.row(col++) = Eigen::RowVector2d(2.0 * Y * invR,
                                              2.0 * X * invR);
    }
    if (degree >= 3) {
        grads.row(col++) = Eigen::RowVector2d(
            (3.0 * X * X - 3.0 * Y * Y) * invR,
            (-6.0 * X * Y) * invR);
        grads.row(col++) = Eigen::RowVector2d(
            (6.0 * X * Y) * invR,
            (3.0 * X * X - 3.0 * Y * Y) * invR);
    }
    return grads;
}

inline double safe_rank_tolerance(int rows,
                                  int cols,
                                  double sigma0,
                                  double rank_tol_rel)
{
    if (!(sigma0 > 0.0))
        return 0.0;
    return std::max(static_cast<double>(std::max(rows, cols))
                        * std::numeric_limits<double>::epsilon(),
                    rank_tol_rel)
           * sigma0;
}

} // namespace laplace_corner_patch_detail

inline double evaluate_corner_patch_singular_basis_2d(
    const CornerPatchCorrectionData2D& data,
    Eigen::Vector2d x)
{
    if (data.patch < 0)
        return 0.0;
    if (!(data.radius > 0.0) || !(data.opening_angle > 0.0)
        || !(data.lambda > 0.0)) {
        throw std::invalid_argument(
            "corner patch correction data has invalid geometry");
    }

    const Eigen::Vector2d v = x - data.center;
    const double r = v.norm();
    if (r <= 1.0e-14)
        return 0.0;

    const Eigen::Vector2d ray0 =
        laplace_corner_patch_detail::normalized_or_throw(
            data.ray_minus, "corner patch ray_minus must be nonzero");
    const Eigen::Vector2d ray1 =
        laplace_corner_patch_detail::normalized_or_throw(
            data.ray_plus, "corner patch ray_plus must be nonzero");
    const Eigen::Vector2d dir = v / r;
    const double theta =
        laplace_corner_patch_detail::corner_wedge_local_angle(
            ray0, ray1, data.opening_angle, dir);
    const double phi = std::sin(data.lambda * theta);

    return std::pow(r, data.lambda) * phi;
}

inline double evaluate_corner_patch_singular_2d(
    const CornerPatchCorrectionData2D& data,
    Eigen::Vector2d x)
{
    if (!data.modes.empty()) {
        CornerPatch2D patch;
        patch.corner = -1;
        patch.point = -1;
        patch.opening_angle = data.opening_angle;
        patch.radius = data.radius;
        patch.center = data.center;
        patch.ray_minus = data.ray_minus;
        patch.ray_plus = data.ray_plus;
        double value = 0.0;
        for (const CornerPatchSingularMode2D& mode : data.modes)
            value += mode.amplitude
                     * laplace_corner_patch_detail::evaluate_singular_mode_basis(
                         patch, mode, x);
        return value;
    }
    return data.amplitude * evaluate_corner_patch_singular_basis_2d(data, x);
}

inline Eigen::Vector2d evaluate_corner_patch_singular_gradient_2d(
    const CornerPatchCorrectionData2D& data,
    Eigen::Vector2d x)
{
    if (!data.modes.empty()) {
        CornerPatch2D patch;
        patch.corner = -1;
        patch.point = -1;
        patch.opening_angle = data.opening_angle;
        patch.radius = data.radius;
        patch.center = data.center;
        patch.ray_minus = data.ray_minus;
        patch.ray_plus = data.ray_plus;
        Eigen::Vector2d grad = Eigen::Vector2d::Zero();
        for (const CornerPatchSingularMode2D& mode : data.modes) {
            grad += mode.amplitude
                  * laplace_corner_patch_detail::evaluate_singular_mode_gradient(
                        patch, mode, x);
        }
        return grad;
    }

    CornerPatch2D patch;
    patch.corner = -1;
    patch.point = -1;
    patch.opening_angle = data.opening_angle;
    patch.radius = data.radius;
    patch.center = data.center;
    patch.ray_minus = data.ray_minus;
    patch.ray_plus = data.ray_plus;

    CornerPatchSingularMode2D mode;
    mode.lambda = data.lambda;
    mode.amplitude = data.amplitude;
    mode.family = CornerPatchSingularFamily2D::Sine;
    return data.amplitude
         * laplace_corner_patch_detail::evaluate_singular_mode_gradient(
               patch, mode, x);
}

inline CornerPatch2D corner_patch_data_geometry_2d(
    const CornerPatchCorrectionData2D& data)
{
    CornerPatch2D patch;
    patch.corner = -1;
    patch.point = data.point;
    patch.side = data.side;
    patch.shape = CornerPatchShape2D::Sector;
    patch.opening_angle = data.opening_angle;
    patch.radius = data.radius;
    patch.center = data.center;
    patch.ray_minus = data.ray_minus;
    patch.ray_plus = data.ray_plus;
    return patch;
}

inline double evaluate_corner_patch_singular_jump_trace_2d(
    const CornerPatchCorrectionData2D& data,
    Eigen::Vector2d x)
{
    if (!data.modes.empty()) {
        const CornerPatch2D patch = corner_patch_data_geometry_2d(data);
        double value = 0.0;
        for (const CornerPatchSingularMode2D& mode : data.modes) {
            if (mode.model
                == CornerPatchSingularModel2D::
                       InteriorNeumannTransmissionPair) {
                continue;
            }
            value += mode.amplitude
                   * laplace_corner_patch_detail::
                          singular_mode_jump_trace_basis_at_point(
                             patch, mode, x);
        }
        return value;
    }
    const double side_factor =
        data.side == CornerPatchSide2D::Exterior ? -1.0 : 1.0;
    return side_factor * evaluate_corner_patch_singular_2d(data, x);
}

inline double evaluate_corner_patch_singular_normal_jump_trace_2d(
    const CornerPatchCorrectionData2D& data,
    Eigen::Vector2d x,
    const Eigen::Vector2d& normal)
{
    bool all_transmission_pair = !data.modes.empty();
    for (const CornerPatchSingularMode2D& mode : data.modes) {
        all_transmission_pair =
            all_transmission_pair
            && mode.model
                   == CornerPatchSingularModel2D::
                          InteriorDirichletTransmissionPair;
    }
    if (all_transmission_pair)
        return 0.0;
    bool all_neumann_transmission_pair = !data.modes.empty();
    for (const CornerPatchSingularMode2D& mode : data.modes) {
        all_neumann_transmission_pair =
            all_neumann_transmission_pair
            && mode.model
                   == CornerPatchSingularModel2D::
                          InteriorNeumannTransmissionPair;
    }
    if (all_neumann_transmission_pair) {
        const CornerPatch2D patch = corner_patch_data_geometry_2d(data);
        double value = 0.0;
        for (const CornerPatchSingularMode2D& mode : data.modes) {
            value += mode.amplitude
                   * laplace_corner_patch_detail::
                         singular_mode_normal_jump_trace_basis_at_point(
                             patch, mode, x, normal);
        }
        return value;
    }
    const double side_factor =
        data.side == CornerPatchSide2D::Exterior ? -1.0 : 1.0;
    return side_factor
         * evaluate_corner_patch_singular_gradient_2d(data, x).dot(normal);
}

inline double evaluate_corner_patch_singular_average_trace_2d(
    const CornerPatchCorrectionData2D& data,
    Eigen::Vector2d x)
{
    bool all_transmission_pair = !data.modes.empty();
    for (const CornerPatchSingularMode2D& mode : data.modes) {
        all_transmission_pair =
            all_transmission_pair
            && mode.model
                   == CornerPatchSingularModel2D::
                          InteriorDirichletTransmissionPair;
    }
    if (all_transmission_pair)
        return -0.5 * evaluate_corner_patch_singular_jump_trace_2d(data, x);
    bool all_neumann_transmission_pair = !data.modes.empty();
    for (const CornerPatchSingularMode2D& mode : data.modes) {
        all_neumann_transmission_pair =
            all_neumann_transmission_pair
            && mode.model
                   == CornerPatchSingularModel2D::
                          InteriorNeumannTransmissionPair;
    }
    if (all_neumann_transmission_pair)
        return evaluate_corner_patch_singular_2d(data, x);
    return 0.5 * evaluate_corner_patch_singular_2d(data, x);
}

inline Eigen::Vector2d evaluate_corner_patch_singular_average_gradient_2d(
    const CornerPatchCorrectionData2D& data,
    Eigen::Vector2d x)
{
    bool all_transmission_pair = !data.modes.empty();
    for (const CornerPatchSingularMode2D& mode : data.modes) {
        all_transmission_pair =
            all_transmission_pair
            && mode.model
                   == CornerPatchSingularModel2D::
                          InteriorDirichletTransmissionPair;
    }
    if (!all_transmission_pair)
    {
        bool all_neumann_transmission_pair = !data.modes.empty();
        for (const CornerPatchSingularMode2D& mode : data.modes) {
            all_neumann_transmission_pair =
                all_neumann_transmission_pair
                && mode.model
                       == CornerPatchSingularModel2D::
                              InteriorNeumannTransmissionPair;
        }
        if (!all_neumann_transmission_pair)
            return 0.5 * evaluate_corner_patch_singular_gradient_2d(data, x);
    }

    const CornerPatch2D patch = corner_patch_data_geometry_2d(data);
    const Eigen::Vector2d v = x - patch.center;
    const double r = v.norm();
    if (r <= 1.0e-14)
        return Eigen::Vector2d::Zero();
    const Eigen::Vector2d er = v / r;
    const Eigen::Vector2d ray0 =
        laplace_corner_patch_detail::normalized_or_throw(
            patch.ray_minus, "corner patch ray_minus must be nonzero");
    const Eigen::Vector2d ray1 =
        laplace_corner_patch_detail::normalized_or_throw(
            patch.ray_plus, "corner patch ray_plus must be nonzero");
    Eigen::Vector2d gradient = Eigen::Vector2d::Zero();
    for (const CornerPatchSingularMode2D& mode : data.modes) {
        const bool neumann_pair =
            mode.model
            == CornerPatchSingularModel2D::InteriorNeumannTransmissionPair;
        const Eigen::Vector2d grad_in =
            laplace_corner_patch_detail::polar_harmonic_gradient(
                er,
                r,
                laplace_corner_patch_detail::ccw_angle(ray0, er),
                mode,
                neumann_pair ? mode.interior_cos_factor : 0.0,
                mode.interior_sine_factor);
        const Eigen::Vector2d grad_out =
            laplace_corner_patch_detail::polar_harmonic_gradient(
                er,
                r,
                laplace_corner_patch_detail::ccw_angle(ray1, er),
                mode,
                neumann_pair ? mode.exterior_cos_factor : -1.0,
                mode.exterior_sine_factor);
        gradient += 0.5 * mode.amplitude * (grad_in + grad_out);
    }
    return gradient;
}

inline double evaluate_corner_patch_correction_2d(
    const CornerPatchCorrectionData2D& data,
    Eigen::Vector2d x)
{
    double value = evaluate_corner_patch_singular_2d(data, x);
    if (data.regular_part.coeffs.size() > 0)
        value += evaluate_taylor_poly_2d(data.regular_part, x);
    return value;
}

inline double evaluate_corner_patch_interface_regular_2d(
    const CornerPatchCorrectionData2D& data,
    Eigen::Vector2d x)
{
    if (!data.has_interface_regular_part
        || data.interface_regular_part.coeffs.size() == 0)
        return 0.0;
    return evaluate_taylor_poly_2d(data.interface_regular_part, x);
}

inline std::vector<CornerPatchFittingSample2D> collect_projected_fit_samples_2d(
    const Interface2D& iface,
    const CornerPatch2D& patch,
    int patch_id,
    const Eigen::VectorXd& u_jump,
    const Eigen::VectorXd& un_jump,
    const Eigen::VectorXd& rhs_jump,
    double kappa,
    const CornerPatchSolverOptions2D& options)
{
    std::vector<CornerPatchFittingSample2D> samples;
    const bool has_explicit_fitting_input =
        !options.fitting_samples.empty()
        || static_cast<bool>(options.fitting_sampler);
    if (options.singular_fit.use_connected_panel_samples
        && !has_explicit_fitting_input) {
        const CornerData2D& corner = iface.corner(patch.corner);
        std::vector<CornerPatchFittingSample2D> connected =
            laplace_corner_patch_detail::collect_four_connected_panel_samples(
                iface,
                corner,
                patch_id,
                u_jump,
                un_jump,
                rhs_jump,
                options.singular_fit);
        samples.insert(samples.end(), connected.begin(), connected.end());
    }

    for (CornerPatchFittingSample2D sample : options.fitting_samples) {
        if (sample.patch == patch_id)
            samples.push_back(std::move(sample));
    }
    if (options.fitting_sampler) {
        std::vector<CornerPatchFittingSample2D> generated =
            options.fitting_sampler(iface,
                                    patch,
                                    patch_id,
                                    u_jump,
                                    un_jump,
                                    rhs_jump,
                                    kappa);
        for (CornerPatchFittingSample2D sample : generated) {
            if (sample.patch < 0)
                sample.patch = patch_id;
            if (sample.patch != patch_id) {
                throw std::invalid_argument(
                    "projected corner patch fitting sampler returned a sample for the wrong patch");
            }
            samples.push_back(std::move(sample));
        }
    }

    for (const CornerPatchArtificialDirichletSample2D& sample
         : options.artificial_dirichlet_samples) {
        if (sample.patch != patch_id)
            continue;
        CornerPatchFittingSample2D converted;
        converted.patch = patch_id;
        converted.point = sample.point;
        converted.value = sample.value;
        converted.has_value = true;
        converted.has_normal_derivative = false;
        converted.interface_jump_value = false;
        samples.push_back(converted);
    }
    if (options.artificial_dirichlet_sampler) {
        std::vector<CornerPatchArtificialDirichletSample2D> generated =
            options.artificial_dirichlet_sampler(iface,
                                                 patch,
                                                 patch_id,
                                                 u_jump,
                                                 un_jump,
                                                 rhs_jump,
                                                 kappa);
        for (CornerPatchArtificialDirichletSample2D sample : generated) {
            if (sample.patch < 0)
                sample.patch = patch_id;
            if (sample.patch != patch_id) {
                throw std::invalid_argument(
                    "projected corner patch fitting artificial sampler returned a sample for the wrong patch");
            }
            CornerPatchFittingSample2D converted;
            converted.patch = patch_id;
            converted.point = sample.point;
            converted.value = sample.value;
            converted.has_value = true;
            converted.has_normal_derivative = false;
            converted.interface_jump_value = false;
            samples.push_back(converted);
        }
    }
    return samples;
}

inline bool corner_patch_has_explicit_fitting_input_2d(
    const CornerPatchSolverOptions2D& options)
{
    return !options.fitting_samples.empty()
        || static_cast<bool>(options.fitting_sampler);
}

inline LocalPoly2D fit_corner_patch_regular_part_from_samples_2d(
    const CornerPatch2D& patch,
    const CornerPatchCorrectionData2D& data,
    const std::vector<CornerPatchFittingSample2D>& samples,
    const CornerPatchSingularFitOptions2D& fit_options,
    double kappa,
    CornerPatchFitDiagnostics2D* diagnostics = nullptr)
{
    using namespace laplace_corner_patch_detail;
    LocalPoly2D regular;
    regular.center = data.interface_regular_part.center;
    regular.coeffs = Eigen::VectorXd::Zero(10);

    int rows = 0;
    for (const CornerPatchFittingSample2D& sample : samples) {
        if (sample.has_value)
            ++rows;
        if (sample.has_normal_derivative)
            ++rows;
        if (sample.has_rhs_value)
            ++rows;
    }
    if (rows == 0)
        return regular;

    bool transmission_pair = !data.modes.empty();
    for (const CornerPatchSingularMode2D& mode : data.modes) {
        transmission_pair =
            transmission_pair
            && (mode.model
                    == CornerPatchSingularModel2D::
                           InteriorDirichletTransmissionPair
                || mode.model
                       == CornerPatchSingularModel2D::
                              InteriorNeumannTransmissionPair);
    }

    Eigen::MatrixXd A(rows, 10);
    Eigen::VectorXd rb(rows);
    int row = 0;
    for (const CornerPatchFittingSample2D& sample : samples) {
        const double radius_ratio =
            (sample.point - patch.center).norm() / patch.radius;
        const double w = sample.confidence
            * std::sqrt(std::max(fit_options.w_min,
                                 std::min(radius_ratio, 1.0)));
        if (sample.has_value) {
            const Eigen::Matrix<double, 1, 10> p3 =
                p3_taylor_basis(regular.center, sample.point);
            for (int k = 0; k < 10; ++k)
                A(row, k) = w * p3[k];
            const double singular_factor =
                fitting_sample_singular_factor(patch, sample);
            double singular = 0.0;
            if (transmission_pair) {
                singular =
                    evaluate_corner_patch_singular_jump_trace_2d(
                        data, sample.point);
            } else {
                for (const CornerPatchSingularMode2D& mode : data.modes) {
                    singular += singular_factor * mode.amplitude
                              * evaluate_singular_mode_basis(
                                  patch, mode, sample.point);
                }
            }
            rb[row] = w * (sample.value - singular);
            ++row;
        }
        if (sample.has_normal_derivative) {
            const Eigen::Vector2d n =
                normalized_or_throw(sample.normal,
                                    "corner patch fitting normal must be nonzero");
            const Eigen::Matrix<double, 2, 10> grad_basis =
                p3_taylor_gradient_basis(regular.center, sample.point);
            for (int k = 0; k < 10; ++k)
                A(row, k) = w * patch.radius * grad_basis.col(k).dot(n);
            const double singular_factor =
                fitting_sample_singular_factor(patch, sample);
            double singular_normal = 0.0;
            if (transmission_pair) {
                singular_normal =
                    evaluate_corner_patch_singular_normal_jump_trace_2d(
                        data, sample.point, n);
            } else {
                for (const CornerPatchSingularMode2D& mode : data.modes) {
                    singular_normal += singular_factor * mode.amplitude
                        * evaluate_singular_mode_gradient(
                              patch, mode, sample.point).dot(n);
                }
            }
            rb[row] = w * patch.radius
                    * (sample.normal_derivative - singular_normal);
            ++row;
        }
        if (sample.has_rhs_value) {
            const Eigen::Matrix<double, 1, 10> op_basis =
                p3_taylor_screened_operator_basis(
                    regular.center, sample.point, kappa);
            const double scale = patch.radius * patch.radius;
            for (int k = 0; k < 10; ++k)
                A(row, k) = w * scale * op_basis[k];
            const double singular_rhs =
                evaluate_singular_modes_screened_operator(
                    patch, data.modes, sample.point, kappa);
            rb[row] = w * scale * (sample.rhs_value - singular_rhs);
            ++row;
        }
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
    if (qr.rank() >= 1)
        regular.coeffs = qr.solve(rb);
    if (diagnostics != nullptr) {
        diagnostics->regular_rows = rows;
        diagnostics->rhs_rows = 0;
        for (const CornerPatchFittingSample2D& sample : samples) {
            if (sample.has_rhs_value)
                ++diagnostics->rhs_rows;
        }
        diagnostics->regular_rank = qr.rank();
        diagnostics->regular_coeff_norm = regular.coeffs.norm();
        const double rb_norm = rb.norm();
        diagnostics->regular_relative_residual =
            rb_norm > 0.0 ? (A * regular.coeffs - rb).norm() / rb_norm
                          : (A * regular.coeffs - rb).norm();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
        const Eigen::VectorXd sigma = svd.singularValues();
        if (sigma.size() > 0) {
            const double tol =
                safe_rank_tolerance(rows,
                                    A.cols(),
                                    sigma[0],
                                    fit_options.rank_tol_rel);
            int rank = 0;
            for (int i = 0; i < sigma.size(); ++i) {
                if (sigma[i] > tol)
                    ++rank;
            }
            diagnostics->regular_rank = rank;
            diagnostics->regular_condition_number =
                rank > 0 ? sigma[0] / sigma[rank - 1] : 0.0;
        }
    }
    return regular;
}

inline CornerPatchCorrectionData2D projected_corner_patch_fit_2d(
    const Interface2D& iface,
    int patch_id,
    const Eigen::VectorXd& u_jump,
    const Eigen::VectorXd& un_jump,
    const Eigen::VectorXd& rhs_jump,
    double kappa,
    const CornerPatchSolverOptions2D& options)
{
    using namespace laplace_corner_patch_detail;
    const CornerPatch2D& patch = iface.corner_patches()[patch_id];
    CornerPatch2D singular_patch = patch;
    if (options.singular_fit.family_policy
            == CornerPatchSingularFamilyPolicy2D::
                   InteriorDirichletTransmissionPair
        || options.singular_fit.family_policy
               == CornerPatchSingularFamilyPolicy2D::
                      InteriorNeumannTransmissionPair) {
        const CornerData2D& corner = iface.corner(patch.corner);
        const Eigen::Vector2d next_ray =
            normalized_or_throw(corner.tangent_plus,
                                "corner tangent_plus must be nonzero");
        const Eigen::Vector2d prev_ray =
            normalized_or_throw(-corner.tangent_minus,
                                "corner tangent_minus must be nonzero");
        const double interior_opening = ccw_angle(next_ray, prev_ray);
        singular_patch.shape = CornerPatchShape2D::Sector;
        if (interior_opening > kPi) {
            singular_patch.side = CornerPatchSide2D::Interior;
            singular_patch.ray_minus = next_ray;
            singular_patch.ray_plus = prev_ray;
            singular_patch.opening_angle = interior_opening;
        } else {
            singular_patch.side = CornerPatchSide2D::Exterior;
            singular_patch.ray_minus = prev_ray;
            singular_patch.ray_plus = next_ray;
            singular_patch.opening_angle = kTwoPi - interior_opening;
        }
        singular_patch.center = patch.center;
        singular_patch.radius = patch.radius;
    }
    CornerPatchCorrectionData2D data;
    data.patch = patch_id;
    data.point = patch.point;
    data.side = singular_patch.side;
    data.opening_angle = singular_patch.opening_angle;
    data.radius = singular_patch.radius;
    data.center = singular_patch.center;
    data.ray_minus = singular_patch.ray_minus;
    data.ray_plus = singular_patch.ray_plus;
    data.modes = build_singular_modes(
        singular_patch,
        options.singular_fit.max_sine_modes,
        options.singular_fit.family_policy,
        options.singular_fit.basis_radius);
    data.lambda = data.modes.front().lambda;
    const bool use_auto_connected_samples =
        options.singular_fit.use_connected_panel_samples
        && !corner_patch_has_explicit_fitting_input_2d(options);
    if (use_auto_connected_samples) {
        data.interface_regular_panels =
            collect_incident_corner_panel_ids(iface.corner(patch.corner));
    }
    const bool fixed_fit_amplitude =
        options.singular_fit.fixed_amplitudes.size() != 0
        && std::isfinite(options.singular_fit.fixed_amplitudes[patch_id]);
    const bool legacy_explicit_amplitude =
        options.artificial_dirichlet_amplitudes.size() != 0;
    data.amplitude = fixed_fit_amplitude
        ? options.singular_fit.fixed_amplitudes[patch_id]
        : (legacy_explicit_amplitude
              ? options.artificial_dirichlet_amplitudes[patch_id]
              : 0.0);
    data.modes.front().amplitude = data.amplitude;
    if (options.regular_parts.empty()) {
        data.regular_part.center = patch.center;
        data.regular_part.coeffs = Eigen::VectorXd::Zero(6);
        data.interface_regular_part.center = patch.center;
        data.interface_regular_part.coeffs = Eigen::VectorXd::Zero(6);
    } else {
        data.regular_part =
            options.regular_parts[static_cast<std::size_t>(patch_id)];
        if (data.regular_part.coeffs.size() != 6) {
            throw std::invalid_argument(
                "laplace_corner_patch_solver_2d regular_parts must be P2 quadratic polynomials");
        }
        data.interface_regular_part = data.regular_part;
        data.has_interface_regular_part = true;
        data.has_explicit_regular_part = true;
    }

    std::vector<CornerPatchFittingSample2D> samples =
        collect_projected_fit_samples_2d(iface,
                                         patch,
                                         patch_id,
                                         u_jump,
                                         un_jump,
                                         rhs_jump,
                                         kappa,
                                         options);
    std::vector<CornerPatchFittingSample2D> valid_samples;
    valid_samples.reserve(samples.size());
    const double r_min = std::max(1.0e-12 * patch.radius, 1.0e-14);
    for (CornerPatchFittingSample2D sample : samples) {
        if (sample.patch != patch_id)
            continue;
        const double r = (sample.point - patch.center).norm();
        if (r <= r_min)
            continue;
        if (!sample.has_value && !sample.has_normal_derivative
            && !sample.has_rhs_value)
            continue;
        if (!(sample.confidence > 0.0))
            continue;
        valid_samples.push_back(std::move(sample));
    }

    const bool explicit_amplitude =
        fixed_fit_amplitude || legacy_explicit_amplitude;
    const bool edge_p2_trace_fit =
        options.singular_fit.coefficient_fit_method
        == CornerPatchSingularCoefficientFitMethod2D::EdgeP2Trace;
    if (valid_samples.empty() && !explicit_amplitude
        && !edge_p2_trace_fit) {
        throw std::invalid_argument(
            "projected corner patch fitting needs fitting samples");
    }

    const int K = static_cast<int>(data.modes.size());
    const int H_cols =
        harmonic_nuisance_count(options.singular_fit.nuisance_degree);
    int rows = 0;
    int value_rows = 0;
    int normal_rows = 0;
    for (const CornerPatchFittingSample2D& sample : valid_samples) {
        if (sample.has_value) {
            ++rows;
            ++value_rows;
        }
        if (sample.has_normal_derivative) {
            ++rows;
            ++normal_rows;
        }
    }
    data.diagnostics.rows = rows;
    data.diagnostics.value_rows = value_rows;
    data.diagnostics.normal_rows = normal_rows;
    data.diagnostics.num_singular_modes = K;
    data.diagnostics.num_nuisance_modes = H_cols;

    if (!explicit_amplitude) {
        if (edge_p2_trace_fit) {
            const Eigen::VectorXd& trace_values =
                singular_fit_uses_normal_jump_trace(
                    options.singular_fit.family_policy)
                ? un_jump
                : u_jump;
            fit_singular_modes_from_edge_p2_trace(
                iface,
                singular_patch,
                data,
                trace_values,
                options.singular_fit);
        } else {
        if (rows <= H_cols) {
            throw std::invalid_argument(
                "projected corner patch fitting needs more rows than nuisance columns");
        }

        Eigen::MatrixXd S(rows, K);
        Eigen::MatrixXd H(rows, H_cols);
        Eigen::VectorXd b(rows);
        int row = 0;
        for (const CornerPatchFittingSample2D& sample : valid_samples) {
            const double radius_ratio =
                (sample.point - patch.center).norm() / patch.radius;
            const double w = sample.confidence
                * std::sqrt(std::max(options.singular_fit.w_min,
                                     std::min(radius_ratio, 1.0)));
            const Eigen::VectorXd h_values =
                harmonic_nuisance_values(patch,
                                         options.singular_fit.nuisance_degree,
                                         sample.point);
            const double singular_factor =
                fitting_sample_singular_factor(patch, sample);
            if (sample.has_value) {
                for (int k = 0; k < K; ++k) {
                    S(row, k) = w * singular_factor
                        * evaluate_singular_mode_basis(
                                        singular_patch, data.modes[k], sample.point);
                }
                for (int h = 0; h < H_cols; ++h)
                    H(row, h) = w * h_values[h];
                b[row] = w * sample.value;
                ++row;
            }
            if (sample.has_normal_derivative) {
                const Eigen::Vector2d n =
                    normalized_or_throw(sample.normal,
                                        "corner patch fitting normal must be nonzero");
                const Eigen::MatrixX2d h_grads =
                    harmonic_nuisance_gradients(
                        patch,
                        options.singular_fit.nuisance_degree,
                        sample.point);
                for (int k = 0; k < K; ++k) {
                    S(row, k) = w * patch.radius * singular_factor
                        * evaluate_singular_mode_gradient(
                              singular_patch, data.modes[k], sample.point).dot(n);
                }
                for (int h = 0; h < H_cols; ++h)
                    H(row, h) = w * patch.radius * h_grads.row(h).dot(n);
                b[row] = w * patch.radius * sample.normal_derivative;
                ++row;
            }
        }

        Eigen::VectorXd dS(K);
        Eigen::MatrixXd S0 = S;
        for (int k = 0; k < K; ++k) {
            dS[k] = S.col(k).norm();
            if (dS[k] <= 1.0e-14) {
                throw std::invalid_argument(
                    "projected corner patch fitting singular column has zero norm");
            }
            S0.col(k) /= dS[k];
        }

        std::vector<int> active_h_cols;
        active_h_cols.reserve(static_cast<std::size_t>(H_cols));
        for (int h = 0; h < H_cols; ++h) {
            if (H.col(h).norm() > 1.0e-14)
                active_h_cols.push_back(h);
        }
        Eigen::MatrixXd H0(rows, static_cast<int>(active_h_cols.size()));
        for (int j = 0; j < static_cast<int>(active_h_cols.size()); ++j)
            H0.col(j) = H.col(active_h_cols[static_cast<std::size_t>(j)])
                      / H.col(active_h_cols[static_cast<std::size_t>(j)]).norm();

        Eigen::MatrixXd QH(rows, 0);
        if (H0.cols() > 0) {
            Eigen::JacobiSVD<Eigen::MatrixXd> hsvd(
                H0, Eigen::ComputeThinU | Eigen::ComputeThinV);
            const Eigen::VectorXd hs = hsvd.singularValues();
            const double htol = hs.size() == 0
                ? 0.0
                : safe_rank_tolerance(rows,
                                      H0.cols(),
                                      hs[0],
                                      options.singular_fit.rank_tol_rel);
            int hrank = 0;
            for (int i = 0; i < hs.size(); ++i) {
                if (hs[i] > htol)
                    ++hrank;
            }
            data.diagnostics.nuisance_rank = hrank;
            QH = hsvd.matrixU().leftCols(hrank);
        }

        const Eigen::MatrixXd S_perp = QH.cols() == 0
            ? S0
            : S0 - QH * (QH.transpose() * S0);
        const Eigen::VectorXd b_perp = QH.cols() == 0
            ? b
            : b - QH * (QH.transpose() * b);

        Eigen::JacobiSVD<Eigen::MatrixXd> ssvd(
            S_perp, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::VectorXd sigma = ssvd.singularValues();
        const double stol = sigma.size() == 0
            ? 0.0
            : safe_rank_tolerance(rows,
                                  K,
                                  sigma[0],
                                  options.singular_fit.rank_tol_rel);
        int srank = 0;
        for (int i = 0; i < sigma.size(); ++i) {
            if (sigma[i] > stol)
                ++srank;
        }
        data.diagnostics.singular_rank = srank;
        if (srank < K && options.singular_fit.throw_on_rank_deficient) {
            std::ostringstream out;
            out << "projected corner patch fitting rank deficient"
                << " patch=" << patch_id
                << " corner=" << patch.corner
                << " rows=" << rows
                << " modes=" << K
                << " nuisance_degree="
                << options.singular_fit.nuisance_degree;
            throw std::invalid_argument(out.str());
        }
        if (srank == 0) {
            throw std::invalid_argument(
                "projected corner patch fitting has zero singular rank");
        }
        data.diagnostics.condition_number =
            sigma[0] / sigma[std::max(0, srank - 1)];

        Eigen::VectorXd scaled = Eigen::VectorXd::Zero(K);
        for (int i = 0; i < srank; ++i) {
            const double inv_sigma = sigma[i] > stol ? 1.0 / sigma[i] : 0.0;
            scaled += inv_sigma * ssvd.matrixV().col(i)
                    * (ssvd.matrixU().col(i).dot(b_perp));
        }
        for (int k = 0; k < K; ++k) {
            data.modes[k].amplitude = scaled[k] / dS[k];
        }
        data.lambda = data.modes.front().lambda;
        data.amplitude = data.modes.front().amplitude;
        data.diagnostics.relative_residual =
            (S_perp * scaled - b_perp).norm()
            / std::max(b_perp.norm(), 1.0e-30);
        double coeff_norm_sq = 0.0;
        for (const CornerPatchSingularMode2D& mode : data.modes)
            coeff_norm_sq += mode.amplitude * mode.amplitude;
        data.diagnostics.coeff_norm = std::sqrt(coeff_norm_sq);
        }
    }

    if (options.regular_parts.empty() && !valid_samples.empty()) {
        data.interface_regular_part =
            fit_corner_patch_regular_part_from_samples_2d(
                singular_patch, data, valid_samples, options.singular_fit, kappa,
                &data.diagnostics);
        data.has_interface_regular_part = true;
        if (options.singular_fit.fit_regular_after_singular)
            data.regular_part = data.interface_regular_part;
    }
    return data;
}

inline std::vector<CornerPatchCorrectionData2D> laplace_corner_patch_solver_2d(
    const Interface2D& iface,
    const Eigen::VectorXd& u_jump,
    const Eigen::VectorXd& un_jump,
    const Eigen::VectorXd& rhs_jump,
    double kappa,
    const CornerPatchSolverOptions2D& options)
{
    if (u_jump.size() != iface.num_points())
        throw std::invalid_argument(
            "laplace_corner_patch_solver_2d u_jump size must equal interface point count");
    if (un_jump.size() != iface.num_points())
        throw std::invalid_argument(
            "laplace_corner_patch_solver_2d un_jump size must equal interface point count");
    if (rhs_jump.size() != iface.num_points())
        throw std::invalid_argument(
            "laplace_corner_patch_solver_2d rhs_jump size must equal interface point count");
    if (options.artificial_dirichlet_amplitudes.size() != 0
        && options.artificial_dirichlet_amplitudes.size()
               != static_cast<Eigen::Index>(iface.corner_patches().size())) {
        throw std::invalid_argument(
            "laplace_corner_patch_solver_2d artificial_dirichlet_amplitudes size must equal corner patch count");
    }
    if (options.singular_fit.fixed_amplitudes.size() != 0
        && options.singular_fit.fixed_amplitudes.size()
               != static_cast<Eigen::Index>(iface.corner_patches().size())) {
        throw std::invalid_argument(
            "laplace_corner_patch_solver_2d singular_fit.fixed_amplitudes size must equal corner patch count");
    }
    if (!options.regular_parts.empty()
        && options.regular_parts.size() != iface.corner_patches().size()) {
        throw std::invalid_argument(
            "laplace_corner_patch_solver_2d regular_parts size must equal corner patch count");
    }
    for (const CornerPatchArtificialDirichletSample2D& sample
         : options.artificial_dirichlet_samples) {
        if (sample.patch < 0
            || sample.patch >= static_cast<int>(iface.corner_patches().size())) {
            throw std::invalid_argument(
                "laplace_corner_patch_solver_2d artificial sample references an invalid patch");
        }
    }
    for (const CornerPatchFittingSample2D& sample : options.fitting_samples) {
        if (sample.patch < 0
            || sample.patch >= static_cast<int>(iface.corner_patches().size())) {
            throw std::invalid_argument(
                "laplace_corner_patch_solver_2d fitting sample references an invalid patch");
        }
    }

    std::vector<CornerPatchCorrectionData2D> result;
    result.reserve(iface.corner_patches().size());
    for (int patch_id = 0;
         patch_id < static_cast<int>(iface.corner_patches().size());
         ++patch_id) {
        if (options.singular_fit.enabled) {
            result.push_back(projected_corner_patch_fit_2d(iface,
                                                           patch_id,
                                                           u_jump,
                                                           un_jump,
                                                           rhs_jump,
                                                           kappa,
                                                           options));
            continue;
        }
        const CornerPatch2D& patch = iface.corner_patches()[patch_id];
        CornerPatchCorrectionData2D data;
        data.patch = patch_id;
        data.point = patch.point;
        data.lambda = laplace_corner_patch_detail::kPi / patch.opening_angle;
        data.amplitude = options.artificial_dirichlet_amplitudes.size() == 0
            ? u_jump[patch.point]
            : options.artificial_dirichlet_amplitudes[patch_id];
        data.opening_angle = patch.opening_angle;
        data.radius = patch.radius;
        data.center = patch.center;
        data.ray_minus = patch.ray_minus;
        data.ray_plus = patch.ray_plus;
        if (options.regular_parts.empty()) {
            data.regular_part.center = patch.center;
            data.regular_part.coeffs = Eigen::VectorXd::Zero(6);
            data.interface_regular_part.center = patch.center;
            data.interface_regular_part.coeffs = Eigen::VectorXd::Zero(6);
        } else {
            data.regular_part = options.regular_parts[static_cast<std::size_t>(patch_id)];
            if (data.regular_part.coeffs.size() != 6) {
                throw std::invalid_argument(
                    "laplace_corner_patch_solver_2d regular_parts must be P2 quadratic polynomials");
            }
            data.interface_regular_part = data.regular_part;
            data.has_interface_regular_part = true;
            data.has_explicit_regular_part = true;
        }

        std::vector<CornerPatchArtificialDirichletSample2D> samples;
        for (const CornerPatchArtificialDirichletSample2D& sample
             : options.artificial_dirichlet_samples) {
            if (sample.patch == patch_id)
                samples.push_back(sample);
        }
        if (options.artificial_dirichlet_sampler) {
            std::vector<CornerPatchArtificialDirichletSample2D> generated =
                options.artificial_dirichlet_sampler(iface,
                                                     patch,
                                                     patch_id,
                                                     u_jump,
                                                     un_jump,
                                                     rhs_jump,
                                                     kappa);
            for (CornerPatchArtificialDirichletSample2D sample : generated) {
                if (sample.patch < 0)
                    sample.patch = patch_id;
                if (sample.patch != patch_id) {
                    throw std::invalid_argument(
                        "laplace_corner_patch_solver_2d artificial sampler returned a sample for the wrong patch");
                }
                samples.push_back(sample);
            }
        }

        const bool fit_amplitude =
            !samples.empty()
            && options.artificial_dirichlet_amplitudes.size() == 0;
        const bool fit_regular =
            !samples.empty() && options.regular_parts.empty();
        const int unknowns = (fit_amplitude ? 1 : 0) + (fit_regular ? 6 : 0);
        if (unknowns > 0) {
            if (static_cast<int>(samples.size()) < unknowns) {
                throw std::invalid_argument(
                    "laplace_corner_patch_solver_2d needs at least as many artificial samples as fitted unknowns");
            }

            Eigen::MatrixXd A(static_cast<int>(samples.size()), unknowns);
            Eigen::VectorXd b(static_cast<int>(samples.size()));
            for (int row = 0; row < static_cast<int>(samples.size()); ++row) {
                const auto& sample = samples[static_cast<std::size_t>(row)];
                int col = 0;
                double known = 0.0;
                double singular_basis = 0.0;
                if (!data.modes.empty()) {
                    CornerPatchCorrectionData2D unit_data = data;
                    unit_data.amplitude = 1.0;
                    for (CornerPatchSingularMode2D& mode : unit_data.modes)
                        mode.amplitude = 0.0;
                    unit_data.modes.front().amplitude = 1.0;
                    singular_basis =
                        evaluate_corner_patch_singular_2d(unit_data,
                                                          sample.point);
                } else {
                    singular_basis =
                        evaluate_corner_patch_singular_basis_2d(data,
                                                                sample.point);
                }
                if (fit_amplitude) {
                    A(row, col++) = singular_basis;
                } else {
                    known += !data.modes.empty()
                        ? evaluate_corner_patch_singular_2d(data,
                                                            sample.point)
                        : data.amplitude * singular_basis;
                }
                const Eigen::Matrix<double, 1, 6> regular_basis =
                    laplace_corner_patch_detail::p2_taylor_basis(
                        data.regular_part.center, sample.point);
                if (fit_regular) {
                    for (int k = 0; k < 6; ++k)
                        A(row, col++) = regular_basis[k];
                } else {
                    for (int k = 0; k < 6; ++k)
                        known += regular_basis[k] * data.regular_part.coeffs[k];
                }
                b[row] = sample.value - known;
            }

            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
            if (qr.rank() < unknowns) {
                throw std::invalid_argument(
                    "laplace_corner_patch_solver_2d artificial sample system is rank deficient");
            }
            const Eigen::VectorXd x = qr.solve(b);
            int col = 0;
            if (fit_amplitude) {
                data.amplitude = x[col++];
                if (!data.modes.empty())
                    data.modes.front().amplitude = data.amplitude;
            }
            if (fit_regular) {
                data.regular_part.coeffs.resize(6);
                for (int k = 0; k < 6; ++k)
                    data.regular_part.coeffs[k] = x[col++];
                data.interface_regular_part = data.regular_part;
                data.has_interface_regular_part = true;
            }
        }
        result.push_back(std::move(data));
    }
    return result;
}

inline std::vector<CornerPatchCorrectionData2D> laplace_corner_patch_solver_2d(
    const Interface2D& iface,
    const Eigen::VectorXd& u_jump,
    const Eigen::VectorXd& un_jump,
    const Eigen::VectorXd& rhs_jump,
    double kappa)
{
    return laplace_corner_patch_solver_2d(iface,
                                          u_jump,
                                          un_jump,
                                          rhs_jump,
                                          kappa,
                                          CornerPatchSolverOptions2D{});
}

} // namespace kfbim
