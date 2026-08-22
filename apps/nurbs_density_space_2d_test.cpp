#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "src/geometry/nurbs_boundary_2d.hpp"
#include "src/transfer/nurbs_density_space_2d.hpp"

namespace {

using kfbim::NurbsDensityContinuity2D;
using kfbim::NurbsDensityCoordinate2D;
using kfbim::NurbsDensitySpace2D;
using kfbim::geometry2d::NurbsBoundaryPanelGeometry2D;
using kfbim::geometry2d::NurbsCurve2D;

constexpr double kPi = 3.141592653589793238462643383279502884;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void require_close(double actual,
                   double expected,
                   double tolerance,
                   const std::string& message)
{
    if (!std::isfinite(actual)
        || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual)
            + " expected=" + std::to_string(expected));
    }
}

NurbsBoundaryPanelGeometry2D make_circle_geometry(double axis_x = 1.0,
                                                   double axis_y = 1.0,
                                                   double angle = 0.0,
                                                   double parameter_start = 0.0,
                                                   double parameter_end = 1.0)
{
    if (!(parameter_end > parameter_start))
        throw std::invalid_argument("circle parameter interval is invalid");
    const auto map_parameter = [&](double unit) {
        return parameter_start
             + (parameter_end - parameter_start) * unit;
    };
    const double w = std::sqrt(0.5);
    NurbsCurve2D::ControlPointVector control_points;
    control_points.emplace_back(1.0, 0.0);
    control_points.emplace_back(1.0, 1.0);
    control_points.emplace_back(0.0, 1.0);
    control_points.emplace_back(-1.0, 1.0);
    control_points.emplace_back(-1.0, 0.0);
    control_points.emplace_back(-1.0, -1.0);
    control_points.emplace_back(0.0, -1.0);
    control_points.emplace_back(1.0, -1.0);
    control_points.emplace_back(1.0, 0.0);
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    for (Eigen::Vector2d& point : control_points) {
        const double x = axis_x * point[0];
        const double y = axis_y * point[1];
        point = {cosine * x - sine * y,
                 sine * x + cosine * y};
    }
    NurbsCurve2D curve(
        kfbim::geometry::NurbsBasis1D(
            2,
            {map_parameter(0.0), map_parameter(0.0), map_parameter(0.0),
             map_parameter(0.25), map_parameter(0.25),
             map_parameter(0.5), map_parameter(0.5),
             map_parameter(0.75), map_parameter(0.75),
             map_parameter(1.0), map_parameter(1.0), map_parameter(1.0)}),
        std::move(control_points),
        {1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0});

    std::vector<int> point_spans(41, 0);
    std::vector<double> point_parameters(41, 0.0);
    for (int i = 0; i < 41; ++i)
        point_parameters[static_cast<std::size_t>(i)] =
            map_parameter(static_cast<double>(i) / 40.0);

    return NurbsBoundaryPanelGeometry2D(
        std::move(curve),
        {parameter_start, parameter_end},
        {0, 0, 0, 0},
        {{{map_parameter(0.0), map_parameter(0.25)}},
         {{map_parameter(0.25), map_parameter(0.5)}},
         {{map_parameter(0.5), map_parameter(0.75)}},
         {{map_parameter(0.75), map_parameter(1.0)}}},
        std::move(point_spans),
        std::move(point_parameters),
        true,
        true);
}

NurbsBoundaryPanelGeometry2D make_l_geometry()
{
    NurbsCurve2D::ControlPointVector control_points;
    control_points.emplace_back(0.0, 0.0);
    control_points.emplace_back(2.0, 0.0);
    control_points.emplace_back(2.0, 1.0);
    control_points.emplace_back(1.0, 1.0);
    control_points.emplace_back(1.0, 2.0);
    control_points.emplace_back(0.0, 2.0);
    control_points.emplace_back(0.0, 0.0);
    NurbsCurve2D curve(
        kfbim::geometry::NurbsBasis1D(
            1, {0.0, 0.0, 1.0, 2.0, 3.0,
                4.0, 5.0, 6.0, 6.0}),
        std::move(control_points),
        std::vector<double>(7, 1.0));

    return NurbsBoundaryPanelGeometry2D(
        std::move(curve),
        {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
        {0, 1, 2, 3, 4, 5},
        {{{0.0, 1.0}},
         {{1.0, 2.0}},
         {{2.0, 3.0}},
         {{3.0, 4.0}},
         {{4.0, 5.0}},
         {{5.0, 6.0}}},
        {},
        {},
        true,
        true);
}

double circle_angle(const Eigen::Vector2d& point)
{
    double angle = std::atan2(point[1], point[0]);
    if (angle < 0.0)
        angle += 2.0 * kPi;
    return angle;
}

void test_smooth_periodic_circle()
{
    NurbsBoundaryPanelGeometry2D geometry = make_circle_geometry();
    NurbsDensitySpace2D phi(
        geometry,
        3,
        0.45,
        {NurbsDensityContinuity2D::C2});

    require(phi.branch_count() == 1,
            "circle must have one density branch");
    require_close(phi.branch_info(0).arclength,
                  2.0 * kPi,
                  2.0e-11,
                  "exact circle arclength");
    require(phi.coordinate()
                == NurbsDensityCoordinate2D::PhysicalArclength,
            "production density coordinate must be physical arclength");
    require(phi.branch_info(0).native_periodic,
            "smooth closed cubic must use a native periodic basis");
    require(phi.constraint_rank() == 0,
            "native periodic cubic must need no seam constraints");
    require(phi.raw_coefficient_count()
                == phi.branch_info(0).spline_span_count
            && phi.raw_coefficient_count()
                == phi.reduced_coefficient_count(),
            "native periodic cubic must have one direct coefficient per span");
    require(phi.constraint_matrix().rows() == 0
                && phi.reduction_matrix().isIdentity(1.0e-14),
            "native periodic cubic reduction must be identity");

    NurbsDensitySpace2D legacy_phi(
        geometry,
        3,
        0.45,
        {NurbsDensityContinuity2D::C2},
        NurbsDensityCoordinate2D::LegacyNurbsParameter);
    require(!legacy_phi.branch_info(0).native_periodic
                && legacy_phi.constraint_rank() == 9
                && legacy_phi.branch_info(0).parameter_chart_breaks.size() == 3
                && legacy_phi.raw_coefficient_count()
                     - legacy_phi.reduced_coefficient_count() == 9
                && legacy_phi.reduced_coefficient_count()
                     == legacy_phi.branch_info(0).spline_span_count,
            "parameter-chart P3 must preserve span-sized reduced coordinates");

    const auto& breaks = phi.branch_info(0).parameter_breaks;
    double minimum_chord = std::numeric_limits<double>::infinity();
    double maximum_chord = 0.0;
    for (std::size_t i = 1; i < breaks.size(); ++i) {
        const double chord =
            (geometry.curve().evaluate(breaks[i])
             - geometry.curve().evaluate(breaks[i - 1]))
                .norm();
        minimum_chord = std::min(minimum_chord, chord);
        maximum_chord = std::max(maximum_chord, chord);
    }
    require(maximum_chord - minimum_chord < 2.0e-11,
            "circle knot breaks must be equally spaced in arclength");
    require(std::abs(breaks[1]
                     - (breaks.back() - breaks.front())
                           / static_cast<double>(breaks.size() - 1))
                > 1.0e-3,
            "rational circle equal-arclength knots should not be uniform in xi");

    const NurbsDensitySpace2D::Function smooth_density =
        [](int, double, const Eigen::Vector2d& point) {
            const double theta = std::atan2(point[1], point[0]);
            return std::sin(2.0 * theta)
                 + 0.25 * std::cos(3.0 * theta);
        };
    const Eigen::VectorXd coefficients =
        phi.fit_function(smooth_density);
    const Eigen::VectorXd legacy_coefficients =
        legacy_phi.fit_function(smooth_density);

    // Parameter-Covariant Difference Crossing Jet (PCD-CJ): sample in xi,
    // then convert with the exact NURBS metric.  This identity test uses the
    // same value rows to independently assemble the chain rule, including the
    // J_xi term whose omission is a common parameterization error.
    const double covariant_xi = 0.10;
    const double parameter_step = 1.0e-3;
    const auto differential = geometry.evaluate_on_span(0, covariant_xi);
    const double mean_span = legacy_phi.branch_info(0).arclength
        / static_cast<double>(
            legacy_phi.branch_info(0).spline_span_count);
    const double step_over_span =
        parameter_step * differential.speed / mean_span;
    const auto covariant_rows =
        legacy_phi.covariant_parameter_finite_difference_rows(
            0, covariant_xi, step_over_span);
    const Eigen::RowVectorXd minus =
        legacy_phi.evaluation_rows(0, covariant_xi - parameter_step).value;
    const Eigen::RowVectorXd center_row =
        legacy_phi.evaluation_rows(0, covariant_xi).value;
    const Eigen::RowVectorXd plus =
        legacy_phi.evaluation_rows(0, covariant_xi + parameter_step).value;
    const Eigen::RowVectorXd parameter_first =
        (plus - minus) / (2.0 * parameter_step);
    const Eigen::RowVectorXd parameter_second =
        (plus - 2.0 * center_row + minus)
        / (parameter_step * parameter_step);
    const double speed_xi = differential.parameter_tangent.dot(
        differential.parameter_second) / differential.speed;
    const Eigen::RowVectorXd expected_first =
        parameter_first / differential.speed;
    const Eigen::RowVectorXd expected_second =
        parameter_second
            / (differential.speed * differential.speed)
        - parameter_first * speed_xi
            / std::pow(differential.speed, 3);
    require((covariant_rows.arclength_first - expected_first).norm()
                    < 2.0e-11
                && (covariant_rows.arclength_second - expected_second).norm()
                    < 2.0e-9,
            "parameter-covariant sampled rows must apply the exact metric chain rule");
    const Eigen::RowVectorXd naive_second = parameter_second
        / (differential.speed * differential.speed);
    require((naive_second - expected_second).norm() > 1.0e-3,
            "rational circle probe must expose the missing J_xi correction");

    const Eigen::VectorXd constant_parameter_coefficients =
        legacy_phi.fit_function(
            [](int, double, const Eigen::Vector2d&) { return 1.0; });
    for (double parameter : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        const auto sampled =
            legacy_phi.evaluate_covariant_parameter_finite_difference(
                0, parameter, constant_parameter_coefficients, 0.5);
        require_close(sampled.value, 1.0, 2.0e-11,
                      "covariant chart constant value");
        require_close(sampled.arclength_first, 0.0, 2.0e-9,
                      "covariant chart constant first derivative");
        require_close(sampled.arclength_second, 0.0, 2.0e-7,
                      "covariant chart constant second derivative");
    }

    for (double parameter : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        const auto analytic = legacy_phi.evaluation_rows(0, parameter);
        const auto sampled =
            legacy_phi.covariant_parameter_finite_difference_rows(
                0, parameter, 0.5);
        require((sampled.value - analytic.value).norm() < 2.0e-11
                    && (sampled.arclength_first
                        - analytic.arclength_first).norm() < 2.0e-7
                    && (sampled.arclength_second
                        - analytic.arclength_second).norm() < 2.0e-5,
                "one-sided covariant chart jet must reproduce its constrained analytic limit");
    }
    const auto left = phi.evaluate(0, 0.0, coefficients);
    const auto right = phi.evaluate(0, 1.0, coefficients);
    require_close(left.value, right.value, 5.0e-12,
                  "periodic fitted value continuity");
    require_close(left.arclength_first,
                  right.arclength_first,
                  5.0e-11,
                  "periodic fitted first continuity");
    require_close(left.arclength_second,
                  right.arclength_second,
                  5.0e-10,
                  "periodic fitted second continuity");

    // Recover a coefficient vector from Interface2D-compatible NURBS point
    // samples. The truth already lies in the reduced space, so this checks the
    // sample fitting path independently of function approximation error.
    Eigen::VectorXd truth = Eigen::VectorXd::LinSpaced(
        phi.reduced_coefficient_count(), -0.7, 0.9);

    const auto seam_left = phi.evaluation_rows(0, 0.0);
    const auto seam_right = phi.evaluation_rows(0, 1.0);
    require((seam_left.value - seam_right.value).norm() < 2.0e-13
                && (seam_left.arclength_first
                    - seam_right.arclength_first).norm() < 2.0e-12
                && (seam_left.arclength_second
                    - seam_right.arclength_second).norm() < 2.0e-11,
            "native periodic rows must agree at the geometric seam");
    const auto local_rows = phi.evaluation_rows(0, 0.03);
    int active_count = 0;
    for (int coefficient = 0;
         coefficient < local_rows.value.size();
         ++coefficient) {
        if (std::abs(local_rows.value[coefficient]) > 1.0e-14)
            ++active_count;
    }
    require(active_count <= phi.degree() + 1,
            "periodic cubic row must retain local cyclic support");

    const Eigen::VectorXd constant = Eigen::VectorXd::Ones(
        phi.reduced_coefficient_count());
    for (double parameter : {0.0, 0.07, 0.25, 0.61, 1.0}) {
        const auto jet = phi.evaluate(0, parameter, constant);
        require_close(jet.value, 1.0, 2.0e-13,
                      "periodic partition of unity");
        require_close(jet.arclength_first, 0.0, 2.0e-12,
                      "periodic first derivative partition");
        require_close(jet.arclength_second, 0.0, 2.0e-11,
                      "periodic second derivative partition");
        const auto sampled = phi.evaluate_sampled_finite_difference(
            0, parameter, constant, 0.25);
        require_close(sampled.value, 1.0, 2.0e-13,
                      "sampled periodic partition of unity");
        require_close(sampled.arclength_first, 0.0, 2.0e-11,
                      "sampled periodic first derivative partition at parameter "
                          + std::to_string(parameter));
        require_close(sampled.arclength_second, 0.0, 2.0e-9,
                      "sampled periodic second derivative partition at parameter "
                          + std::to_string(parameter));
    }

    const auto sampled_seam_left =
        phi.sampled_finite_difference_rows(0, 0.0, 0.25);
    const auto sampled_seam_right =
        phi.sampled_finite_difference_rows(0, 1.0, 0.25);
    require((sampled_seam_left.value - sampled_seam_right.value).norm()
                    < 2.0e-13
                && (sampled_seam_left.arclength_first
                    - sampled_seam_right.arclength_first).norm() < 2.0e-11
                && (sampled_seam_left.arclength_second
                    - sampled_seam_right.arclength_second).norm() < 2.0e-9,
            "sampled periodic finite differences must wrap at the seam");

    for (double quarter : {0.25, 0.5, 0.75}) {
        const double epsilon = 1.0e-9;
        const auto left = phi.evaluate(0, quarter - epsilon, truth);
        const auto right = phi.evaluate(0, quarter + epsilon, truth);
        require(std::abs(left.arclength_first
                         - right.arclength_first) < 2.0e-6
                    && std::abs(left.arclength_second
                                - right.arclength_second) < 2.0e-5,
                "physical density derivatives must cross CAD quarter knots continuously");
    }

    double periodic_second_jump = 0.0;
    double legacy_second_jump = 0.0;
    for (double quarter : {0.25, 0.5, 0.75}) {
        const double left_parameter = std::nextafter(quarter, 0.0);
        const double right_parameter = std::nextafter(quarter, 1.0);
        periodic_second_jump = std::max(
            periodic_second_jump,
            std::abs(
                phi.evaluate(0, left_parameter, coefficients)
                    .arclength_second
                - phi.evaluate(0, right_parameter, coefficients)
                    .arclength_second));
        legacy_second_jump = std::max(
            legacy_second_jump,
            std::abs(
                legacy_phi.evaluate(
                    0, left_parameter, legacy_coefficients)
                    .arclength_second
                - legacy_phi.evaluate(
                    0, right_parameter, legacy_coefficients)
                    .arclength_second));
    }
    require(periodic_second_jump < 2.0e-8,
            "periodic arclength P3 must remove CAD-knot second-derivative jumps");
    std::cout << "circle density phi_ss CAD-knot jump: xi="
              << legacy_second_jump
              << " arclength=" << periodic_second_jump << '\n';
    Eigen::VectorXd point_values(geometry.num_parameterized_points());
    for (int q = 0; q < point_values.size(); ++q) {
        point_values[q] = phi.evaluate(
            geometry.point_span(q), geometry.point_parameter(q), truth).value;
    }
    const Eigen::VectorXd recovered =
        phi.fit_parameterized_point_samples(point_values);
    require((recovered - truth).norm()
                <= 2.0e-10 * std::max(1.0, truth.norm()),
            "parameterized point samples must recover reduced coefficients");

    // Verify the physical derivative conversion on a non-uniform rational
    // parameter. Signed circle angle is exact physical arclength here.
    const double xi = 0.10;
    const double epsilon = 1.0e-4;
    const auto center = phi.evaluate(0, xi, truth);
    const double fm = phi.evaluate(0, xi - epsilon, truth).value;
    const double f0 = center.value;
    const double fp = phi.evaluate(0, xi + epsilon, truth).value;
    const double theta_m = circle_angle(
        geometry.curve().evaluate(xi - epsilon));
    const double theta_0 = circle_angle(
        geometry.curve().evaluate(xi));
    const double theta_p = circle_angle(
        geometry.curve().evaluate(xi + epsilon));
    const double hm = theta_0 - theta_m;
    const double hp = theta_p - theta_0;
    const double finite_first =
        -hp * fm / (hm * (hm + hp))
        + (hp - hm) * f0 / (hm * hp)
        + hm * fp / (hp * (hm + hp));
    const double finite_second = 2.0
        * (fm / (hm * (hm + hp))
           - f0 / (hm * hp)
           + fp / (hp * (hm + hp)));
    require_close(center.arclength_first,
                  finite_first,
                  2.0e-5,
                  "xi-to-arclength first derivative");
    require_close(center.arclength_second,
                  finite_second,
                  2.0e-4,
                  "xi-to-arclength second derivative");

    // Crossing convenience and direct coefficient Cauchy closure.
    kfbim::P2CrossingOwner2D crossing;
    crossing.panel_index = 0;
    crossing.local_s = -0.2;
    crossing.crossing_point = geometry.point(0, crossing.local_s);
    crossing.crossing_normal = geometry.normal(0, crossing.local_s);
    crossing.exact_intersection_count = 1;
    crossing.transverse_intersection_count = 1;
    crossing.status = kfbim::P2CrossingOwnerStatus2D::ExactIntersection;
    const auto location = phi.resolve_crossing(crossing);
    const auto crossing_phi = phi.evaluate_crossing(crossing, truth);
    const auto direct_phi = phi.evaluate(location, truth);
    require_close(crossing_phi.value, direct_phi.value, 1.0e-13,
                  "crossing value resolution");

    NurbsDensitySpace2D psi(
        geometry,
        2,
        0.45,
        {NurbsDensityContinuity2D::C1});
    require(psi.branch_info(0).native_periodic
                && psi.constraint_rank() == 0
                && psi.raw_coefficient_count()
                    == psi.reduced_coefficient_count(),
            "smooth closed quadratic must use direct periodic C1 coefficients");
    const Eigen::VectorXd psi_coefficients = psi.fit_function(
        [](int, double, const Eigen::Vector2d& point) {
            return 0.2 + 0.1 * point[0] - 0.05 * point[1];
        });
    const auto polynomial =
        kfbim::build_nurbs_density_local_polynomial_2d(
            phi,
            truth,
            psi,
            psi_coefficients,
            crossing,
            0.3,
            0.0);
    const auto crossing_psi = psi.evaluate_crossing(
        crossing, psi_coefficients);
    require_close(polynomial.value, crossing_phi.value, 1.0e-13,
                  "coefficient Cauchy value");
    require_close(polynomial.tangent_derivative,
                  crossing_phi.arclength_first,
                  1.0e-12,
                  "coefficient Cauchy tangent derivative");
    require_close(polynomial.normal_derivative,
                  crossing_psi.value,
                  1.0e-12,
                  "coefficient Cauchy normal derivative");
    require(std::isfinite(polynomial.hessian_tt)
                && std::isfinite(polynomial.hessian_tn)
                && std::isfinite(polynomial.hessian_nn),
            "coefficient Cauchy Hessian must be finite");
}

void test_ellipse_periodic_fixed_flux_fit()
{
    NurbsBoundaryPanelGeometry2D geometry =
        make_circle_geometry(0.92, 0.61, 0.37);
    const NurbsDensitySpace2D::Function flux =
        [&geometry](int branch,
                    double parameter,
                    const Eigen::Vector2d& point) {
            const Eigen::Vector2d gradient{
                3.0 * point[0] * point[0]
                    - 3.0 * point[1] * point[1]
                    + 0.70 * point[0] + 0.20,
                -6.0 * point[0] * point[1]
                    - 0.70 * point[1] - 0.15};
            return gradient.dot(
                geometry.evaluate_on_span(branch, parameter).normal);
        };

    std::vector<double> periodic_errors;
    std::vector<double> legacy_errors;
    for (int n : {32, 64, 128}) {
        const double target_spacing = 4.5 / static_cast<double>(n);
        NurbsDensitySpace2D periodic(
            geometry,
            2,
            target_spacing,
            {NurbsDensityContinuity2D::C1});
        NurbsDensitySpace2D legacy(
            geometry,
            2,
            target_spacing,
            {NurbsDensityContinuity2D::C1},
            NurbsDensityCoordinate2D::LegacyNurbsParameter);
        const Eigen::VectorXd periodic_coefficients =
            periodic.fit_function(flux);
        const Eigen::VectorXd legacy_coefficients =
            legacy.fit_function(flux);

        double periodic_error = 0.0;
        double legacy_error = 0.0;
        constexpr int sample_count = 2048;
        for (int sample = 0; sample < sample_count; ++sample) {
            const double parameter = static_cast<double>(sample)
                / static_cast<double>(sample_count);
            const Eigen::Vector2d point =
                geometry.curve().evaluate(parameter);
            const double exact = flux(0, parameter, point);
            periodic_error = std::max(
                periodic_error,
                std::abs(periodic.evaluate(
                    0, parameter, periodic_coefficients).value - exact));
            legacy_error = std::max(
                legacy_error,
                std::abs(legacy.evaluate(
                    0, parameter, legacy_coefficients).value - exact));
        }
        periodic_errors.push_back(periodic_error);
        legacy_errors.push_back(legacy_error);
        std::cout << "ellipse fixed psi fit N=" << n
                  << " xi=" << legacy_error
                  << " arclength=" << periodic_error << '\n';
    }
    require(periodic_errors[2] < periodic_errors[1]
                && periodic_errors[1] < periodic_errors[0]
                && periodic_errors[2] < 2.0e-4,
            "periodic arclength P2 must accurately fit smooth ellipse flux");
}

void test_affine_parameter_covariance()
{
    NurbsBoundaryPanelGeometry2D unit = make_circle_geometry();
    NurbsBoundaryPanelGeometry2D mapped = make_circle_geometry(
        1.0, 1.0, 0.0, 2.0, 7.0);
    NurbsDensitySpace2D unit_space(
        unit,
        3,
        0.45,
        {NurbsDensityContinuity2D::C2},
        NurbsDensityCoordinate2D::LegacyNurbsParameter);
    NurbsDensitySpace2D mapped_space(
        mapped,
        3,
        0.45,
        {NurbsDensityContinuity2D::C2},
        NurbsDensityCoordinate2D::LegacyNurbsParameter);
    require(unit_space.raw_coefficient_count()
                    == mapped_space.raw_coefficient_count()
                && unit_space.reduced_coefficient_count()
                    == mapped_space.reduced_coefficient_count(),
            "affine parameter maps must preserve density dimensions");

    const NurbsDensitySpace2D::Function density =
        [](int, double, const Eigen::Vector2d& point) {
            const double theta = std::atan2(point[1], point[0]);
            return 0.3 + std::sin(2.0 * theta)
                 - 0.2 * std::cos(3.0 * theta);
        };
    const Eigen::VectorXd unit_coefficients = unit_space.fit_function(density);
    const Eigen::VectorXd mapped_coefficients =
        mapped_space.fit_function(density);
    for (double unit_parameter : {0.0, 0.10, 0.25, 0.61, 1.0}) {
        const double mapped_parameter = 2.0 + 5.0 * unit_parameter;
        const auto unit_jet =
            unit_space.evaluate_covariant_parameter_finite_difference(
                0, unit_parameter, unit_coefficients, 0.5);
        const auto mapped_jet =
            mapped_space.evaluate_covariant_parameter_finite_difference(
                0, mapped_parameter, mapped_coefficients, 0.5);
        require_close(mapped_jet.value,
                      unit_jet.value,
                      2.0e-10,
                      "affine parameter covariant value");
        require_close(mapped_jet.arclength_first,
                      unit_jet.arclength_first,
                      2.0e-8,
                      "affine parameter covariant first derivative");
        require_close(mapped_jet.arclength_second,
                      unit_jet.arclength_second,
                      2.0e-6,
                      "affine parameter covariant second derivative");
    }
}

void test_parameter_chart_derivative_convergence()
{
    NurbsBoundaryPanelGeometry2D geometry = make_circle_geometry();
    std::vector<double> value_errors;
    std::vector<double> first_errors;
    std::vector<double> second_errors;
    for (double spacing : {0.18, 0.09, 0.045}) {
        NurbsDensitySpace2D space(
            geometry,
            3,
            spacing,
            {NurbsDensityContinuity2D::C2},
            NurbsDensityCoordinate2D::LegacyNurbsParameter);
        const Eigen::VectorXd coefficients = space.fit_function(
            [](int, double, const Eigen::Vector2d& point) {
                const double theta = std::atan2(point[1], point[0]);
                return std::sin(2.0 * theta)
                     + 0.25 * std::cos(3.0 * theta);
            });
        double value_error = 0.0;
        double first_error = 0.0;
        double second_error = 0.0;
        constexpr int sample_count = 2048;
        for (int sample = 0; sample < sample_count; ++sample) {
            const double parameter =
                (static_cast<double>(sample) + 0.37)
                / static_cast<double>(sample_count);
            const Eigen::Vector2d point =
                geometry.curve().evaluate(parameter);
            const double theta = circle_angle(point);
            const double exact_value =
                std::sin(2.0 * theta) + 0.25 * std::cos(3.0 * theta);
            const double exact_first =
                2.0 * std::cos(2.0 * theta)
                - 0.75 * std::sin(3.0 * theta);
            const double exact_second =
                -4.0 * std::sin(2.0 * theta)
                - 2.25 * std::cos(3.0 * theta);
            const auto jet = space.evaluate(0, parameter, coefficients);
            value_error = std::max(
                value_error, std::abs(jet.value - exact_value));
            first_error = std::max(
                first_error,
                std::abs(jet.arclength_first - exact_first));
            second_error = std::max(
                second_error,
                std::abs(jet.arclength_second - exact_second));
        }
        value_errors.push_back(value_error);
        first_errors.push_back(first_error);
        second_errors.push_back(second_error);
        std::cout << "parameter chart P3 spacing=" << spacing
                  << " errors=" << value_error << '/'
                  << first_error << '/' << second_error << '\n';
    }
    require(value_errors[2] < value_errors[1]
                && value_errors[1] < value_errors[0]
                && first_errors[2] < first_errors[1]
                && first_errors[1] < first_errors[0]
                && second_errors[2] < second_errors[1]
                && second_errors[1] < second_errors[0],
            "parameter chart P3 intrinsic jets must converge under refinement");
    require(value_errors.back() < 2.0e-5
                && first_errors.back() < 2.0e-3
                && second_errors.back() < 2.0e-1,
            "parameter chart P3 intrinsic fit is unexpectedly inaccurate");
}

void test_g0_l_shape_spaces()
{
    NurbsBoundaryPanelGeometry2D geometry = make_l_geometry();
    NurbsDensitySpace2D phi(
        geometry,
        3,
        0.4,
        std::vector<NurbsDensityContinuity2D>(
            6, NurbsDensityContinuity2D::C0));
    require(phi.branch_count() == 6,
            "L shape must retain six one-sided branches");
    require(phi.constraint_rank() == 6,
            "L-shape scalar C0 trace must impose one constraint per corner");
    require(phi.raw_coefficient_count()
                - phi.reduced_coefficient_count() == 6,
            "L-shape C0 reduced count");

    Eigen::VectorXd coefficients = Eigen::VectorXd::LinSpaced(
        phi.reduced_coefficient_count(), -1.0, 1.0);
    double maximum_derivative_jump = 0.0;
    for (int branch = 0; branch < 6; ++branch) {
        const int next = (branch + 1) % 6;
        const auto left = phi.evaluate(
            branch,
            phi.branch_info(branch).parameter_end,
            coefficients);
        const auto right = phi.evaluate(
            next,
            phi.branch_info(next).parameter_start,
            coefficients);
        require_close(left.value, right.value, 2.0e-12,
                      "L-shape one-sided C0 value");
        maximum_derivative_jump = std::max(
            maximum_derivative_jump,
            std::abs(left.arclength_first
                     - right.arclength_first));
    }
    require(maximum_derivative_jump > 1.0e-4,
            "L-shape C0 reduction must not impose derivative continuity");

    const Eigen::VectorXd polynomial_coefficients = phi.fit_function(
        [](int, double, const Eigen::Vector2d& point) {
            return point.squaredNorm();
        });
    for (const auto& sample : std::vector<std::pair<double, double>>{
             {0.0, 0.0}, {0.4, 1.6}, {1.0, 4.0}}) {
        const auto finite = phi.evaluate_sampled_finite_difference(
            0, sample.first, polynomial_coefficients, 0.25);
        require_close(finite.value,
                      4.0 * sample.first * sample.first,
                      2.0e-11,
                      "one-sided sampled polynomial value");
        require_close(finite.arclength_first,
                      sample.second,
                      2.0e-9,
                      "one-sided sampled polynomial first derivative");
        require_close(finite.arclength_second,
                      2.0,
                      2.0e-8,
                      "one-sided sampled polynomial second derivative");
    }

    NurbsDensitySpace2D parameter_phi(
        geometry,
        3,
        0.4,
        std::vector<NurbsDensityContinuity2D>(
            6, NurbsDensityContinuity2D::C0),
        NurbsDensityCoordinate2D::LegacyNurbsParameter);
    const Eigen::VectorXd parameter_polynomial_coefficients =
        parameter_phi.fit_function(
            [](int, double, const Eigen::Vector2d& point) {
                return point.squaredNorm();
            });
    for (const auto& sample : std::vector<std::pair<double, double>>{
             {0.0, 0.0}, {0.4, 1.6}, {1.0, 4.0}}) {
        const auto finite =
            parameter_phi.evaluate_covariant_parameter_finite_difference(
                0,
                sample.first,
                parameter_polynomial_coefficients,
                0.25);
        require_close(finite.value,
                      4.0 * sample.first * sample.first,
                      2.0e-11,
                      "one-sided covariant polynomial value");
        require_close(finite.arclength_first,
                      sample.second,
                      2.0e-8,
                      "one-sided covariant polynomial first derivative");
        require_close(finite.arclength_second,
                      2.0,
                      2.0e-7,
                      "one-sided covariant polynomial second derivative");
    }

    NurbsDensitySpace2D psi(
        geometry,
        2,
        0.4,
        std::vector<NurbsDensityContinuity2D>(
            6, NurbsDensityContinuity2D::Discontinuous));
    require(psi.constraint_rank() == 0,
            "L-shape one-sided psi must have no corner constraints");
    require(psi.raw_coefficient_count()
                == psi.reduced_coefficient_count(),
            "discontinuous L-shape coefficients remain raw coefficients");

    Eigen::VectorXd one_sided = Eigen::VectorXd::Zero(
        psi.reduced_coefficient_count());
    for (int branch = 0; branch < psi.branch_count(); ++branch) {
        const auto& info = psi.branch_info(branch);
        one_sided.segment(info.raw_coefficient_offset,
                          info.raw_coefficient_count)
            .setConstant(static_cast<double>(branch + 1));
    }
    const double branch0_right = psi.evaluate(
        0, psi.branch_info(0).parameter_end, one_sided).value;
    const double branch1_left = psi.evaluate(
        1, psi.branch_info(1).parameter_start, one_sided).value;
    require_close(branch0_right, 1.0, 1.0e-13,
                  "L-shape one-sided psi left value");
    require_close(branch1_left, 2.0, 1.0e-13,
                  "L-shape one-sided psi right value");
}

} // namespace

int main()
{
    try {
        test_smooth_periodic_circle();
        test_ellipse_periodic_fixed_flux_fit();
        test_affine_parameter_covariance();
        test_parameter_chart_derivative_convergence();
        test_g0_l_shape_spaces();
        std::cout << "nurbs_density_space_2d_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "nurbs_density_space_2d_test: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
