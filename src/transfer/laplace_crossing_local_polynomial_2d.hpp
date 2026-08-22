#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "../geometry/grid_pair_2d.hpp"
#include "../geometry/p2_curve_2d.hpp"
#include "../local_cauchy/local_poly.hpp"
#include "laplace_crossing_trace_stencil_2d.hpp"

namespace kfbim {

// Complete quadratic correction polynomial frozen at one smooth P2 crossing.
//
// The coefficients are stored in the orthonormal frame (tangent, normal) at
// center.  For d = x - center, xi = d.tangent and eta = d.normal,
//
//   P = value + tangent_derivative*xi + normal_derivative*eta
//       + 1/2*hessian_tt*xi^2 + hessian_tn*xi*eta
//       + 1/2*hessian_nn*eta^2.
//
// This is a spatial Taylor polynomial in a frozen frame, not a polynomial in
// the moving curve coordinates.  To retain O(h^2) accuracy in the highest
// trace derivatives, an active value jump phi is reconstructed in P3 and an
// active normal jump psi in P2.  The BVP/layer formulation selects which block
// is active; an inactive block is identically absent even if its input vector
// contains nonzero values.  The PDE contributes only its value at the
// crossing. Curvature terms convert those arclength derivatives into the
// frozen spatial Hessian.
struct LaplaceP2CrossingLocalPolynomial2D {
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    Eigen::Vector2d tangent = Eigen::Vector2d::UnitX();
    Eigen::Vector2d normal = Eigen::Vector2d::UnitY();

    double curvature = 0.0;
    double value = 0.0;
    double tangent_derivative = 0.0;
    double normal_derivative = 0.0;
    double hessian_tt = 0.0;
    double hessian_tn = 0.0;
    double hessian_nn = 0.0;

    double evaluate(Eigen::Vector2d point) const
    {
        const Eigen::Vector2d displacement = point - center;
        const double xi = displacement.dot(tangent);
        const double eta = displacement.dot(normal);
        return value
             + tangent_derivative * xi
             + normal_derivative * eta
             + 0.5 * hessian_tt * xi * xi
             + hessian_tn * xi * eta
             + 0.5 * hessian_nn * eta * eta;
    }

    double evaluate_on_normal(double signed_distance) const
    {
        return value
             + normal_derivative * signed_distance
             + 0.5 * hessian_nn * signed_distance * signed_distance;
    }

    LocalPoly2D cartesian_taylor() const
    {
        const Eigen::Vector2d gradient =
            tangent_derivative * tangent + normal_derivative * normal;
        const Eigen::Matrix2d hessian =
            hessian_tt * (tangent * tangent.transpose())
          + hessian_tn
                * (tangent * normal.transpose()
                   + normal * tangent.transpose())
          + hessian_nn * (normal * normal.transpose());

        LocalPoly2D poly;
        poly.center = center;
        poly.coeffs.resize(6);
        poly.coeffs << value,
                       gradient[0],
                       gradient[1],
                       hessian(0, 0),
                       hessian(0, 1),
                       hessian(1, 1);
        return poly;
    }
};

// Interface information required by the quadratic spatial closure at one
// crossing.  Derivatives are with respect to signed physical arclength, not a
// panel parameter.  Both the legacy local Lagrange fit and the formal ALS-CJ
// reconstruction feed this same geometry/PDE closure.
struct LaplaceCrossingTraceJet2D {
    double value = 0.0;
    double value_tangent_derivative = 0.0;
    double value_tangent_second_derivative = 0.0;
    double normal_value = 0.0;
    double normal_tangent_derivative = 0.0;
};

// Geometry/PDE data at the actual crossing.  Separating this from a P2 panel
// parameter lets an exact NURBS crossing (including a former corner-gap hit)
// close the same Cauchy equations without manufacturing a surrogate local_s.
struct LaplaceCrossingGeometryJet2D {
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    Eigen::Vector2d tangent = Eigen::Vector2d::UnitX();
    Eigen::Vector2d normal = Eigen::Vector2d::UnitY();
    double curvature = 0.0;
    double forcing = 0.0;
};

namespace crossing_local_detail {

struct TraceDofSample {
    int point = -1;
    double arclength = 0.0;
};

struct TraceJet {
    double value = 0.0;
    double first = 0.0;
    double second = 0.0;
};

inline double signed_panel_arclength(const Interface2D& iface,
                                     int panel,
                                     double from_s,
                                     double to_s)
{
    if (from_s == to_s)
        return 0.0;

    // Eight-point Gauss-Legendre integration is exact for constant-speed
    // retained arclength geometry and resolves positional P2 panel speeds to
    // far beyond the accuracy required by the P3/P2 trace fits.
    constexpr std::array<double, 8> nodes = {{
        -0.9602898564975363,
        -0.7966664774136267,
        -0.5255324099163290,
        -0.1834346424956498,
         0.1834346424956498,
         0.5255324099163290,
         0.7966664774136267,
         0.9602898564975363}};
    constexpr std::array<double, 8> weights = {{
        0.1012285362903763,
        0.2223810344533745,
        0.3137066458778873,
        0.3626837833783620,
        0.3626837833783620,
        0.3137066458778873,
        0.2223810344533745,
        0.1012285362903763}};

    const double midpoint = 0.5 * (from_s + to_s);
    const double half_span = 0.5 * (to_s - from_s);
    double integral = 0.0;
    for (std::size_t q = 0; q < nodes.size(); ++q) {
        const double sample_s = midpoint + half_span * nodes[q];
        const double speed =
            geometry2d::panel_tangent(iface, panel, sample_s).norm();
        if (!std::isfinite(speed) || !(speed > 1.0e-14)) {
            throw std::invalid_argument(
                "crossing-local trace fit encountered degenerate panel arclength");
        }
        integral += weights[q] * speed;
    }
    return half_span * integral;
}

inline int previous_oriented_panel(const Interface2D& iface, int panel)
{
    const int endpoint = iface.point_index(panel, 0);
    const int component = iface.panel_components()[panel];
    for (int candidate = 0; candidate < iface.num_panels(); ++candidate) {
        if (candidate != panel
            && iface.panel_components()[candidate] == component
            && iface.point_index(candidate, 2) == endpoint) {
            return candidate;
        }
    }
    return -1;
}

inline int next_oriented_panel(const Interface2D& iface, int panel)
{
    const int endpoint = iface.point_index(panel, 2);
    const int component = iface.panel_components()[panel];
    for (int candidate = 0; candidate < iface.num_panels(); ++candidate) {
        if (candidate != panel
            && iface.panel_components()[candidate] == component
            && iface.point_index(candidate, 0) == endpoint) {
            return candidate;
        }
    }
    return -1;
}

inline void add_trace_sample(std::vector<TraceDofSample>& samples,
                             int point,
                             double arclength)
{
    if (point < 0 || !std::isfinite(arclength))
        return;
    for (TraceDofSample& sample : samples) {
        if (sample.point != point)
            continue;
        if (std::abs(arclength) < std::abs(sample.arclength))
            sample.arclength = arclength;
        return;
    }
    samples.push_back({point, arclength});
}

inline std::vector<TraceDofSample> nearby_trace_dofs(
    const Interface2D& iface,
    int panel,
    double local_s)
{
    std::vector<TraceDofSample> samples;
    samples.reserve(5);
    for (int local = 0; local < 3; ++local) {
        const int point = iface.point_index(panel, local);
        if (iface.is_corner_point(point))
            continue;
        add_trace_sample(
            samples,
            point,
            signed_panel_arclength(
                iface,
                panel,
                local_s,
                geometry2d::kP2NodeS[static_cast<std::size_t>(local)]));
    }

    const int left_point = iface.point_index(panel, 0);
    const int previous = previous_oriented_panel(iface, panel);
    if (previous >= 0 && !iface.is_corner_point(left_point)) {
        add_trace_sample(
            samples,
            iface.point_index(previous, 1),
            signed_panel_arclength(iface, panel, local_s, -1.0)
                + signed_panel_arclength(iface, previous, 1.0, 0.0));
    }

    const int right_point = iface.point_index(panel, 2);
    const int next = next_oriented_panel(iface, panel);
    if (next >= 0 && !iface.is_corner_point(right_point)) {
        add_trace_sample(
            samples,
            iface.point_index(next, 1),
            signed_panel_arclength(iface, panel, local_s, 1.0)
                + signed_panel_arclength(iface, next, -1.0, 0.0));
    }

    std::sort(samples.begin(),
              samples.end(),
              [](const TraceDofSample& lhs, const TraceDofSample& rhs) {
                  const double lhs_distance = std::abs(lhs.arclength);
                  const double rhs_distance = std::abs(rhs.arclength);
                  if (lhs_distance != rhs_distance)
                      return lhs_distance < rhs_distance;
                  if (lhs.arclength != rhs.arclength)
                      return lhs.arclength < rhs.arclength;
                  return lhs.point < rhs.point;
              });
    return samples;
}

template <std::size_t SampleCount>
inline TraceJet interpolate_trace_jet(
    const std::vector<TraceDofSample>& available,
    const Eigen::VectorXd& values)
{
    if (available.size() < SampleCount) {
        throw std::invalid_argument(
            "crossing-local trace fit has too few smooth interface DOFs");
    }

    double scale = 0.0;
    for (std::size_t i = 0; i < SampleCount; ++i)
        scale = std::max(scale, std::abs(available[i].arclength));
    if (!(scale > 1.0e-14) || !std::isfinite(scale)) {
        throw std::invalid_argument(
            "crossing-local trace fit has a degenerate arclength stencil");
    }

    std::array<double, SampleCount> nodes{};
    for (std::size_t i = 0; i < SampleCount; ++i)
        nodes[i] = available[i].arclength / scale;

    TraceJet jet;
    for (std::size_t i = 0; i < SampleCount; ++i) {
        std::array<double, SampleCount> numerator{};
        numerator[0] = 1.0;
        std::size_t degree = 0;
        double denominator = 1.0;
        for (std::size_t j = 0; j < SampleCount; ++j) {
            if (j == i)
                continue;
            if (std::abs(nodes[i] - nodes[j]) <= 1.0e-13) {
                throw std::invalid_argument(
                    "crossing-local trace fit has repeated arclength nodes");
            }
            denominator *= nodes[i] - nodes[j];
            for (std::size_t k = degree + 1; k > 0; --k) {
                numerator[k] = numerator[k - 1]
                             - nodes[j] * numerator[k];
            }
            numerator[0] *= -nodes[j];
            ++degree;
        }

        const double sample_value = values[available[i].point];
        jet.value += sample_value * numerator[0] / denominator;
        if constexpr (SampleCount >= 2) {
            jet.first += sample_value * numerator[1]
                       / (denominator * scale);
        }
        if constexpr (SampleCount >= 3) {
            jet.second += 2.0 * sample_value * numerator[2]
                        / (denominator * scale * scale);
        }
    }
    return jet;
}

inline bool has_trace_stencil(const Interface2D& iface,
                              int panel,
                              double local_s,
                              std::size_t required_dofs)
{
    try {
        const std::vector<TraceDofSample> samples =
            nearby_trace_dofs(iface, panel, local_s);
        if (required_dofs < 1 || samples.size() < required_dofs)
            return false;
        double scale = 0.0;
        for (std::size_t i = 0; i < required_dofs; ++i)
            scale = std::max(scale, std::abs(samples[i].arclength));
        if (!(scale > 1.0e-14) || !std::isfinite(scale))
            return false;
        for (std::size_t i = 0; i < required_dofs; ++i) {
            for (std::size_t j = i + 1; j < required_dofs; ++j) {
                if (std::abs(samples[i].arclength
                             - samples[j].arclength)
                    <= 1.0e-13 * scale) {
                    return false;
                }
            }
        }
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

} // namespace crossing_local_detail

inline LaplaceP2CrossingLocalPolynomial2D
build_laplace_crossing_local_polynomial_from_geometry_trace_jet_2d(
    LaplaceCrossingGeometryJet2D geometry,
    const LaplaceCrossingTraceJet2D& trace,
    double alpha)
{
    if (!geometry.center.allFinite() || !geometry.tangent.allFinite()
        || !geometry.normal.allFinite()
        || !std::isfinite(geometry.curvature)
        || !std::isfinite(geometry.forcing) || !std::isfinite(alpha)) {
        throw std::invalid_argument(
            "crossing-local geometry/trace closure has nonfinite input");
    }
    const double tangent_norm = geometry.tangent.norm();
    const double normal_norm = geometry.normal.norm();
    if (!(tangent_norm > 1.0e-14) || !(normal_norm > 1.0e-14)) {
        throw std::invalid_argument(
            "crossing-local geometry/trace closure has a degenerate frame");
    }
    geometry.tangent /= tangent_norm;
    geometry.normal -= geometry.normal.dot(geometry.tangent)
                     * geometry.tangent;
    if (!(geometry.normal.norm() > 1.0e-14)) {
        throw std::invalid_argument(
            "crossing-local geometry/trace closure frame is collinear");
    }
    geometry.normal.normalize();

    LaplaceP2CrossingLocalPolynomial2D poly;
    poly.center = geometry.center;
    poly.tangent = geometry.tangent;
    poly.normal = geometry.normal;
    poly.curvature = geometry.curvature;
    poly.value = trace.value;
    poly.tangent_derivative = trace.value_tangent_derivative;
    poly.normal_derivative = trace.normal_value;
    poly.hessian_tt = trace.value_tangent_second_derivative
                    + geometry.curvature * trace.normal_value;
    poly.hessian_tn = trace.normal_tangent_derivative
                    - geometry.curvature
                        * trace.value_tangent_derivative;
    // Repo convention: (-Delta + alpha) C = rhs_jump.
    poly.hessian_nn = alpha * trace.value
                    - geometry.forcing
                    - trace.value_tangent_second_derivative
                    - geometry.curvature * trace.normal_value;

    const Eigen::Matrix<double, 6, 1> coefficients =
        (Eigen::Matrix<double, 6, 1>()
             << poly.value,
                poly.tangent_derivative,
                poly.normal_derivative,
                poly.hessian_tt,
                poly.hessian_tn,
                poly.hessian_nn)
            .finished();
    if (!coefficients.allFinite()) {
        throw std::invalid_argument(
            "crossing-local geometry/trace closure produced nonfinite coefficients");
    }
    return poly;
}

inline LaplaceP2CrossingLocalPolynomial2D
build_laplace_p2_crossing_local_polynomial_from_trace_jet_2d(
    const Interface2D& iface,
    int panel,
    double local_s,
    const LaplaceCrossingTraceJet2D& trace,
    const Eigen::VectorXd& rhs_jump,
    double alpha)
{
    if (!geometry2d::is_quadratic_lagrange_panel_layout(iface)) {
        throw std::invalid_argument(
            "crossing-local trace-jet closure requires P2 quadratic panels");
    }
    if (panel < 0 || panel >= iface.num_panels()
        || !std::isfinite(local_s)
        || local_s < -1.0 - 1.0e-10
        || local_s > 1.0 + 1.0e-10
        || rhs_jump.size() != iface.num_points()
        || !std::isfinite(alpha)) {
        throw std::invalid_argument(
            "crossing-local trace-jet closure has invalid input");
    }

    local_s = geometry2d::clamp_to_panel(local_s);
    const Eigen::Vector2d center =
        geometry2d::panel_point(iface, panel, local_s);
    const Eigen::Vector2d parameter_tangent =
        geometry2d::panel_tangent(iface, panel, local_s);
    const Eigen::Vector2d parameter_second =
        geometry2d::panel_second_derivative(iface, panel, local_s);
    const double speed = parameter_tangent.norm();
    if (!(speed > 1.0e-14) || !center.allFinite()
        || !parameter_tangent.allFinite()
        || !parameter_second.allFinite()) {
        throw std::invalid_argument(
            "crossing-local trace-jet closure encountered degenerate geometry");
    }

    const Eigen::Vector2d tangent = parameter_tangent / speed;
    Eigen::Vector2d normal =
        geometry2d::panel_normal(iface, panel, local_s);
    if (!normal.allFinite() || !(normal.norm() > 1.0e-14)) {
        throw std::invalid_argument(
            "crossing-local trace-jet closure encountered an invalid normal");
    }
    normal.normalize();
    normal -= normal.dot(tangent) * tangent;
    if (!(normal.norm() > 1.0e-14)) {
        throw std::invalid_argument(
            "crossing-local trace-jet closure frame is degenerate");
    }
    normal.normalize();

    const double curvature =
        -parameter_second.dot(normal) / (speed * speed);
    const double forcing =
        geometry2d::panel_scalar(iface, panel, rhs_jump, local_s);

    LaplaceCrossingGeometryJet2D geometry;
    geometry.center = center;
    geometry.tangent = tangent;
    geometry.normal = normal;
    geometry.curvature = curvature;
    geometry.forcing = forcing;
    return build_laplace_crossing_local_polynomial_from_geometry_trace_jet_2d(
        geometry, trace, alpha);
}

inline bool can_build_laplace_p2_crossing_local_polynomial_2d(
    const Interface2D& iface,
    const P2CrossingOwner2D& crossing,
    LaplaceCrossingTraceStencil2D trace_stencil =
        LaplaceCrossingTraceStencil2D::PhiP3PsiP2)
{
    const std::size_t required_dofs =
        crossing_trace_required_dofs(trace_stencil);
    if (!geometry2d::is_quadratic_lagrange_panel_layout(iface)
        || !is_valid_crossing_trace_stencil(trace_stencil)
        || crossing.status != P2CrossingOwnerStatus2D::ExactIntersection
        || crossing.exact_intersection_count != 1
        || crossing.panel_index < 0
        || crossing.panel_index >= iface.num_panels()
        || !std::isfinite(crossing.local_s)
        || crossing.local_s < -1.0 - 1.0e-10
        || crossing.local_s > 1.0 + 1.0e-10
        || !crossing.crossing_point.allFinite()) {
        return false;
    }

    // A physical corner has no unique tangent, normal, or curvature.  An
    // exact endpoint hit there must stay on the existing corner/fallback path.
    if (std::abs(crossing.local_s + 1.0) <= 1.0e-10
        && iface.is_corner_point(
            iface.point_index(crossing.panel_index, 0))) {
        return false;
    }
    if (std::abs(crossing.local_s - 1.0) <= 1.0e-10
        && iface.is_corner_point(
            iface.point_index(crossing.panel_index, 2))) {
        return false;
    }
    // Phi-P3 needs four smooth DOFs; a psi-only P2 stencil needs three.  The
    // required combination is prescribed by the BVP and is never inferred
    // from the sampled values.
    return required_dofs == 0
        || crossing_local_detail::has_trace_stencil(
               iface, crossing.panel_index, crossing.local_s, required_dofs);
}

inline LaplaceP2CrossingLocalPolynomial2D
build_laplace_p2_crossing_local_polynomial_2d(
    const Interface2D& iface,
    int panel,
    double local_s,
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump,
    const Eigen::VectorXd& rhs_jump,
    double alpha,
    LaplaceCrossingTraceStencil2D trace_stencil =
        LaplaceCrossingTraceStencil2D::PhiP3PsiP2)
{
    if (!geometry2d::is_quadratic_lagrange_panel_layout(iface)) {
        throw std::invalid_argument(
            "crossing-local polynomial requires P2 quadratic panels");
    }
    if (panel < 0 || panel >= iface.num_panels()
        || !std::isfinite(local_s)
        || local_s < -1.0 - 1.0e-10
        || local_s > 1.0 + 1.0e-10) {
        throw std::invalid_argument(
            "crossing-local polynomial has an invalid panel parameter");
    }
    if (value_jump.size() != iface.num_points()
        || normal_jump.size() != iface.num_points()
        || rhs_jump.size() != iface.num_points()) {
        throw std::invalid_argument(
            "crossing-local polynomial requires one value, normal, and RHS jump per interface point");
    }
    if (!std::isfinite(alpha)) {
        throw std::invalid_argument(
            "crossing-local polynomial requires a finite reaction coefficient");
    }

    const std::size_t required_dofs =
        crossing_trace_required_dofs(trace_stencil);
    if (!is_valid_crossing_trace_stencil(trace_stencil)) {
        throw std::invalid_argument(
            "crossing-local polynomial has an invalid trace stencil");
    }

    local_s = geometry2d::clamp_to_panel(local_s);
    std::vector<crossing_local_detail::TraceDofSample> trace_samples;
    if (required_dofs > 0) {
        trace_samples =
            crossing_local_detail::nearby_trace_dofs(iface, panel, local_s);
    }
    if (trace_samples.size() < required_dofs) {
        throw std::invalid_argument(
            "crossing-local polynomial has too few smooth DOFs for its BVP-selected trace stencil");
    }
    // First interpolate exactly the information requested by the local
    // equations in signed physical arclength.  The degrees are deliberately
    // different so the highest derivatives both have O(h^2) error:
    //
    //   phi=[u]   : P3 -> A, A_l, A_ll,
    //   psi=[u_n] : P2 -> B, B_l,
    //   [f]   : P0 -> one value at the crossing.
    //
    // In the layer-potential cases one of phi or psi is identically zero, so
    // its complete block vanishes.  Geometry conversion and PDE closure remain
    // separate from this trace interpolation.
    crossing_local_detail::TraceJet value_trace;
    if (crossing_trace_uses_phi_p3(trace_stencil)) {
        value_trace = crossing_local_detail::interpolate_trace_jet<4>(
            trace_samples, value_jump);
    }
    crossing_local_detail::TraceJet normal_trace;
    if (crossing_trace_uses_psi_p2(trace_stencil)) {
        normal_trace = crossing_local_detail::interpolate_trace_jet<3>(
            trace_samples, normal_jump);
    }
    LaplaceCrossingTraceJet2D trace;
    trace.value = value_trace.value;
    trace.value_tangent_derivative = value_trace.first;
    trace.value_tangent_second_derivative = value_trace.second;
    trace.normal_value = normal_trace.value;
    trace.normal_tangent_derivative = normal_trace.first;
    return build_laplace_p2_crossing_local_polynomial_from_trace_jet_2d(
        iface, panel, local_s, trace, rhs_jump, alpha);
}

inline LaplaceP2CrossingLocalPolynomial2D
build_laplace_p2_crossing_local_polynomial_2d(
    const Interface2D& iface,
    const P2CrossingOwner2D& crossing,
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump,
    const Eigen::VectorXd& rhs_jump,
    double alpha,
    LaplaceCrossingTraceStencil2D trace_stencil =
        LaplaceCrossingTraceStencil2D::PhiP3PsiP2)
{
    if (!can_build_laplace_p2_crossing_local_polynomial_2d(
            iface, crossing, trace_stencil)) {
        throw std::invalid_argument(
            "crossing-local polynomial requires one smooth exact crossing");
    }
    return build_laplace_p2_crossing_local_polynomial_2d(
        iface,
        crossing.panel_index,
        crossing.local_s,
        value_jump,
        normal_jump,
        rhs_jump,
        alpha,
        trace_stencil);
}

} // namespace kfbim
