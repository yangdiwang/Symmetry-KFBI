#include "src/geometry/curve_2d.hpp"
#include "src/geometry/curve_resampler_2d.hpp"
#include "src/geometry/nurbs_boundary_2d.hpp"
#include "src/geometry/p2_curve_2d.hpp"
#include "src/grid/cartesian_grid_2d.hpp"
#include "src/interface/interface_2d.hpp"
#include "src/local_cauchy/jump_data.hpp"
#include "src/operators/laplace_bvp_2d.hpp"
#include "src/transfer/laplace_correction_support.hpp"
#include "src/transfer/laplace_arc_length_bspline_crossing_jet_2d.hpp"
#include "src/transfer/laplace_crossing_local_polynomial_2d.hpp"
#include "src/transfer/laplace_spread_2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace kfbim;

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
    const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    if (std::abs(actual - expected) > tolerance * scale) {
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual)
            + " expected=" + std::to_string(expected));
    }
}

class UnitCircleCurve final : public ICurve2D {
public:
    Eigen::Vector2d eval(double theta) const override
    {
        return {std::cos(theta), std::sin(theta)};
    }

    Eigen::Vector2d deriv(double theta) const override
    {
        return {-std::sin(theta), std::cos(theta)};
    }

    Eigen::Vector2d second_deriv(double theta) const override
    {
        return {-std::cos(theta), -std::sin(theta)};
    }

    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }
};

P2CrossingOwner2D exact_panel_owner(const Interface2D& iface,
                                    int panel,
                                    double local_s)
{
    P2CrossingOwner2D owner;
    owner.panel_index = panel;
    owner.local_s = local_s;
    owner.crossing_point = geometry2d::panel_point(iface, panel, local_s);
    owner.crossing_normal = geometry2d::panel_normal(iface, panel, local_s);
    owner.exact_intersection_count = 1;
    owner.status = P2CrossingOwnerStatus2D::ExactIntersection;
    return owner;
}

double periodic_angle(Eigen::Vector2d point)
{
    double theta = std::atan2(point[1], point[0]);
    if (theta < 0.0)
        theta += 2.0 * kPi;
    return theta;
}

double phi_trace(double theta)
{
    return std::sin(theta) + 0.2 * std::cos(2.0 * theta);
}

double phi_trace_first(double theta)
{
    return std::cos(theta) - 0.4 * std::sin(2.0 * theta);
}

double phi_trace_second(double theta)
{
    return -std::sin(theta) - 0.8 * std::cos(2.0 * theta);
}

double psi_trace(double theta)
{
    return 0.4 * std::cos(theta) + 0.15 * std::sin(2.0 * theta);
}

double psi_trace_first(double theta)
{
    return -0.4 * std::sin(theta) + 0.3 * std::cos(2.0 * theta);
}

struct AlsCjErrors {
    double phi_value = 0.0;
    double phi_first = 0.0;
    double phi_second = 0.0;
    double psi_value = 0.0;
    double psi_first = 0.0;
};

AlsCjErrors evaluate_als_cj_errors(double h)
{
    auto curve = std::make_shared<UnitCircleCurve>();
    const Interface2D iface =
        CurveResampler2D::discretize_quadratic_lagrange(curve, h, 1.6);
    const LaplaceArcLengthBSplineCrossingJetPlan2D plan(iface);
    require(plan.smooth_component_count() == 1,
            "ALS-CJ did not recognize the retained smooth circle");

    Eigen::VectorXd phi(iface.num_points());
    Eigen::VectorXd psi(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        const double theta = periodic_angle(
            iface.points().row(q).transpose());
        phi[q] = phi_trace(theta);
        psi[q] = psi_trace(theta);
    }
    const LaplaceArcLengthBSplineTraceState2D state = plan.fit(
        phi, psi, LaplaceCrossingTraceStencil2D::PhiP3PsiP2);

    AlsCjErrors errors;
    for (int panel = 0; panel < iface.num_panels(); ++panel) {
        for (int sample = 0; sample < 7; ++sample) {
            const double local_s = -0.91
                + 1.82 * static_cast<double>(sample) / 6.0;
            const P2CrossingOwner2D owner =
                exact_panel_owner(iface, panel, local_s);
            require(plan.can_evaluate(
                        owner,
                        LaplaceCrossingTraceStencil2D::PhiP3PsiP2),
                    "ALS-CJ rejected a smooth exact circle crossing");
            const LaplaceCrossingTraceJet2D jet = plan.evaluate(owner, state);
            const double theta = periodic_angle(owner.crossing_point);
            errors.phi_value = std::max(
                errors.phi_value,
                std::abs(jet.value - phi_trace(theta)));
            errors.phi_first = std::max(
                errors.phi_first,
                std::abs(jet.value_tangent_derivative
                         - phi_trace_first(theta)));
            errors.phi_second = std::max(
                errors.phi_second,
                std::abs(jet.value_tangent_second_derivative
                         - phi_trace_second(theta)));
            errors.psi_value = std::max(
                errors.psi_value,
                std::abs(jet.normal_value - psi_trace(theta)));
            errors.psi_first = std::max(
                errors.psi_first,
                std::abs(jet.normal_tangent_derivative
                         - psi_trace_first(theta)));
        }
    }
    return errors;
}

void test_als_cj_periodic_continuity_and_accuracy()
{
    auto curve = std::make_shared<UnitCircleCurve>();
    const Interface2D iface =
        CurveResampler2D::discretize_quadratic_lagrange(curve, 0.17, 1.6);
    const LaplaceArcLengthBSplineCrossingJetPlan2D plan(iface);
    require(plan.smooth_component_count() == 1
                && plan.total_component_count() == 1,
            "ALS-CJ circle component plan count");
    require(std::string(laplace_crossing_jet_scheme_name_2d(
                LaplaceCrossingJetScheme2D::ArcLengthBSplineCrossingJet))
                == "arc_length_bspline_crossing_jet",
            "ALS-CJ formal scheme name");

    Eigen::VectorXd phi(iface.num_points());
    Eigen::VectorXd psi(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        const double theta = periodic_angle(
            iface.points().row(q).transpose());
        phi[q] = phi_trace(theta);
        psi[q] = psi_trace(theta);
    }
    const LaplaceArcLengthBSplineTraceState2D state = plan.fit(
        phi, psi, LaplaceCrossingTraceStencil2D::PhiP3PsiP2);

    std::vector<int> panel_from_left(
        static_cast<std::size_t>(iface.num_points()), -1);
    for (int panel = 0; panel < iface.num_panels(); ++panel) {
        panel_from_left[static_cast<std::size_t>(
            iface.point_index(panel, 0))] = panel;
    }
    for (int panel = 0; panel < iface.num_panels(); ++panel) {
        const int endpoint = iface.point_index(panel, 2);
        const int next = panel_from_left[static_cast<std::size_t>(endpoint)];
        require(next >= 0, "ALS-CJ circle panel connectivity is not closed");
        const LaplaceCrossingTraceJet2D left = plan.evaluate(
            exact_panel_owner(iface, panel, 1.0), state);
        const LaplaceCrossingTraceJet2D right = plan.evaluate(
            exact_panel_owner(iface, next, -1.0), state);
        require_close(left.value, right.value, 2.0e-12,
                      "ALS-CJ phi is discontinuous across an owner panel");
        require_close(left.value_tangent_derivative,
                      right.value_tangent_derivative,
                      2.0e-11,
                      "ALS-CJ phi_l is discontinuous across an owner panel");
        require_close(left.value_tangent_second_derivative,
                      right.value_tangent_second_derivative,
                      2.0e-10,
                      "ALS-CJ phi_ll is discontinuous across an owner panel");
        require_close(left.normal_value,
                      right.normal_value,
                      2.0e-12,
                      "ALS-CJ psi is discontinuous across an owner panel");
        require_close(left.normal_tangent_derivative,
                      right.normal_tangent_derivative,
                      2.0e-11,
                      "ALS-CJ psi_l is discontinuous across an owner panel");
    }

    Eigen::VectorXd constant_phi = Eigen::VectorXd::Constant(
        iface.num_points(), 1.75);
    Eigen::VectorXd constant_psi = Eigen::VectorXd::Constant(
        iface.num_points(), -0.63);
    const auto constant_state = plan.fit(
        constant_phi,
        constant_psi,
        LaplaceCrossingTraceStencil2D::PhiP3PsiP2);
    const auto constant_jet = plan.evaluate(
        exact_panel_owner(iface, 0, 0.37), constant_state);
    require_close(constant_jet.value, 1.75, 2.0e-13,
                  "ALS-CJ cubic constant reproduction");
    require_close(constant_jet.normal_value, -0.63, 2.0e-13,
                  "ALS-CJ quadratic constant reproduction");
    require_close(constant_jet.value_tangent_derivative, 0.0, 2.0e-12,
                  "ALS-CJ cubic constant first derivative");
    require_close(constant_jet.value_tangent_second_derivative, 0.0, 2.0e-11,
                  "ALS-CJ cubic constant second derivative");
    require_close(constant_jet.normal_tangent_derivative, 0.0, 2.0e-12,
                  "ALS-CJ quadratic constant first derivative");

    const AlsCjErrors coarse = evaluate_als_cj_errors(0.24);
    const AlsCjErrors fine = evaluate_als_cj_errors(0.12);
    const double phi_second_order =
        std::log(coarse.phi_second / fine.phi_second) / std::log(2.0);
    const double psi_first_order =
        std::log(coarse.psi_first / fine.psi_first) / std::log(2.0);
    require(coarse.phi_second / fine.phi_second > 3.2,
            "ALS-CJ P3 phi_ll did not converge at second order");
    require(coarse.psi_first / fine.psi_first > 3.2,
            "ALS-CJ P2 psi_l did not converge at second order");
    require(fine.phi_value < 2.0e-4
                && fine.phi_first < 2.0e-3
                && fine.phi_second < 2.0e-2
                && fine.psi_value < 5.0e-4
                && fine.psi_first < 5.0e-3,
            "ALS-CJ periodic trace errors exceed the expected fine-grid scale");
    std::cout << "ALS-CJ trace interpolation: phi_ll order="
              << phi_second_order
              << ", psi_l order=" << psi_first_order
              << ", fine errors=" << fine.phi_second
              << '/' << fine.psi_first << '\n';
}

Interface2D make_open_straight_p2_chain()
{
    constexpr int point_count = 9;
    constexpr int panel_count = 4;
    Eigen::MatrixX2d points(point_count, 2);
    Eigen::MatrixX2d normals(point_count, 2);
    Eigen::VectorXd weights(point_count);
    for (int q = 0; q < point_count; ++q) {
        const double x = static_cast<double>(q)
                       / static_cast<double>(point_count - 1);
        points.row(q) << x, 0.0;
        normals.row(q) << 0.0, -1.0;
        weights[q] = 1.0 / static_cast<double>(point_count - 1);
    }
    Eigen::MatrixXi panel_points(panel_count, 3);
    for (int panel = 0; panel < panel_count; ++panel) {
        panel_points.row(panel)
            << 2 * panel, 2 * panel + 1, 2 * panel + 2;
    }
    return Interface2D(std::move(points),
                       std::move(normals),
                       std::move(weights),
                       3,
                       std::move(panel_points),
                       Eigen::VectorXi::Zero(panel_count),
                       PanelNodeLayout2D::QuadraticLagrange);
}

void test_als_cj_open_branch_polynomial_reproduction()
{
    const Interface2D iface = make_open_straight_p2_chain();
    const LaplaceArcLengthBSplineCrossingJetPlan2D plan(iface);
    require(plan.smooth_component_count() == 1
                && plan.periodic_component_count() == 0
                && plan.open_branch_count() == 1,
            "ALS-CJ did not build one open smooth-branch plan");

    const auto phi = [](double x) {
        return 1.2 - 0.4 * x + 0.7 * x * x - 0.3 * x * x * x;
    };
    const auto phi_first = [](double x) {
        return -0.4 + 1.4 * x - 0.9 * x * x;
    };
    const auto phi_second = [](double x) {
        return 1.4 - 1.8 * x;
    };
    const auto psi = [](double x) {
        return -0.6 + 0.8 * x + 0.25 * x * x;
    };
    const auto psi_first = [](double x) {
        return 0.8 + 0.5 * x;
    };

    Eigen::VectorXd phi_samples(iface.num_points());
    Eigen::VectorXd psi_samples(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        const double x = iface.points()(q, 0);
        phi_samples[q] = phi(x);
        psi_samples[q] = psi(x);
    }
    const auto state = plan.fit(
        phi_samples,
        psi_samples,
        LaplaceCrossingTraceStencil2D::PhiP3PsiP2);
    constexpr std::array<double, 5> local_samples{{
        -1.0, -0.73, -0.11, 0.58, 1.0}};
    for (int panel = 0; panel < iface.num_panels(); ++panel) {
        for (double local_s : local_samples) {
            const P2CrossingOwner2D owner =
                exact_panel_owner(iface, panel, local_s);
            require(plan.can_evaluate(
                        owner,
                        LaplaceCrossingTraceStencil2D::PhiP3PsiP2),
                    "ALS-CJ rejected an open-branch crossing");
            const auto jet = plan.evaluate(owner, state);
            const double x = owner.crossing_point[0];
            require_close(jet.value, phi(x), 2.0e-12,
                          "ALS-CJ open cubic value reproduction");
            require_close(jet.value_tangent_derivative,
                          phi_first(x), 3.0e-11,
                          "ALS-CJ open cubic first derivative");
            require_close(jet.value_tangent_second_derivative,
                          phi_second(x), 3.0e-10,
                          "ALS-CJ open cubic second derivative");
            require_close(jet.normal_value, psi(x), 2.0e-12,
                          "ALS-CJ open quadratic value reproduction");
            require_close(jet.normal_tangent_derivative,
                          psi_first(x), 3.0e-11,
                          "ALS-CJ open quadratic first derivative");
        }
    }
}

Interface2D make_nurbs_same_parameter_line_interface()
{
    constexpr int point_count = 9;
    constexpr int panel_count = 4;
    Eigen::MatrixX2d points(point_count, 2);
    Eigen::MatrixX2d normals(point_count, 2);
    Eigen::VectorXd weights(point_count);
    std::vector<int> point_spans(
        static_cast<std::size_t>(point_count), 0);
    std::vector<double> point_parameters(
        static_cast<std::size_t>(point_count));
    for (int q = 0; q < point_count; ++q) {
        const double parameter =
            (static_cast<double>(q) + 0.5)
            / static_cast<double>(point_count);
        point_parameters[static_cast<std::size_t>(q)] = parameter;
        points.row(q) << 2.0 * parameter, 0.0;
        normals.row(q) << 0.0, -1.0;
        weights[q] = 2.0 / static_cast<double>(point_count);
    }
    Eigen::MatrixXi panel_points(panel_count, 3);
    std::vector<int> panel_spans(
        static_cast<std::size_t>(panel_count), 0);
    std::vector<std::array<double, 2>> panel_parameters;
    panel_parameters.reserve(panel_count);
    for (int panel = 0; panel < panel_count; ++panel) {
        panel_points.row(panel)
            << 2 * panel, 2 * panel + 1, 2 * panel + 2;
        panel_parameters.push_back(
            {point_parameters[static_cast<std::size_t>(2 * panel)],
             point_parameters[static_cast<std::size_t>(2 * panel + 2)]});
    }
    std::shared_ptr<const IPanelGeometry2D> geometry =
        std::make_shared<geometry2d::NurbsBoundaryPanelGeometry2D>(
            geometry2d::NurbsCurve2D::make_line_segment(
                {0.0, 0.0}, {2.0, 0.0}),
            std::vector<double>{0.0, 1.0},
            std::move(panel_spans),
            std::move(panel_parameters),
            std::move(point_spans),
            std::move(point_parameters),
            true,
            false);
    return Interface2D(
        std::move(points),
        std::move(normals),
        std::move(weights),
        3,
        std::move(panel_points),
        Eigen::VectorXi::Zero(panel_count),
        PanelNodeLayout2D::QuadraticLagrange,
        std::move(geometry));
}

void test_nsp_cj_same_parameter_and_gap_crossing_reproduction()
{
    const Interface2D iface = make_nurbs_same_parameter_line_interface();
    const LaplaceArcLengthBSplineCrossingJetPlan2D plan(
        iface,
        LaplaceCrossingJetScheme2D::NurbsSameParameterCrossingJet);
    require(plan.uses_nurbs_same_parameter()
                && plan.nurbs_span_count() == 1,
            "NSP-CJ did not bind the exact line NURBS span");
    require(std::string(laplace_crossing_jet_scheme_name_2d(
                LaplaceCrossingJetScheme2D::
                    NurbsSameParameterCrossingJet))
                == "nurbs_same_parameter_crossing_jet",
            "NSP-CJ formal scheme name");

    const auto phi = [](double s) {
        return 0.7 - 0.2 * s + 0.4 * s * s - 0.15 * s * s * s;
    };
    const auto phi_s = [](double s) {
        return -0.2 + 0.8 * s - 0.45 * s * s;
    };
    const auto phi_ss = [](double s) {
        return 0.8 - 0.9 * s;
    };
    const auto psi = [](double s) {
        return -0.3 + 0.6 * s + 0.2 * s * s;
    };
    const auto psi_s = [](double s) {
        return 0.6 + 0.4 * s;
    };

    Eigen::VectorXd phi_samples(iface.num_points());
    Eigen::VectorXd psi_samples(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        const double s = iface.points()(q, 0);
        phi_samples[q] = phi(s);
        psi_samples[q] = psi(s);
    }
    const auto state = plan.fit(
        phi_samples,
        psi_samples,
        LaplaceCrossingTraceStencil2D::PhiP3PsiP2);

    std::vector<P2CrossingOwner2D> owners;
    owners.push_back(exact_panel_owner(iface, 2, 0.37));
    P2CrossingOwner2D gap;
    gap.panel_index = 0;
    gap.local_s = 0.0; // deliberately unrelated to the true crossing
    gap.crossing_point = {0.04, 0.0};
    gap.crossing_normal = {0.0, -1.0};
    gap.explicit_gap_intersection = true;
    gap.status = P2CrossingOwnerStatus2D::GapFallback;
    owners.push_back(gap);

    for (const P2CrossingOwner2D& owner : owners) {
        require(plan.can_evaluate(
                    owner,
                    LaplaceCrossingTraceStencil2D::PhiP3PsiP2),
                "NSP-CJ rejected an identified exact/gap crossing");
        const auto jet = plan.evaluate(owner, state);
        const double s = owner.crossing_point[0];
        require_close(jet.value, phi(s), 3.0e-12,
                      "NSP-CJ P3 value reproduction");
        require_close(jet.value_tangent_derivative, phi_s(s), 3.0e-11,
                      "NSP-CJ P3 physical first derivative");
        require_close(jet.value_tangent_second_derivative, phi_ss(s),
                      4.0e-10,
                      "NSP-CJ P3 physical second derivative");
        require_close(jet.normal_value, psi(s), 3.0e-12,
                      "NSP-CJ P2 normal value reproduction");
        require_close(jet.normal_tangent_derivative, psi_s(s), 3.0e-11,
                      "NSP-CJ P2 physical first derivative");

        const Eigen::VectorXd rhs =
            Eigen::VectorXd::Zero(iface.num_points());
        const auto polynomial =
            plan.build_local_polynomial(owner, state, rhs, 0.0);
        require((polynomial.center - owner.crossing_point).norm() <= 2.0e-12,
                "NSP-CJ local polynomial is not bound to the true crossing");
    }
}

void test_nurbs_boundary_analytic_curvature()
{
    geometry2d::NurbsBoundaryPanelGeometry2D geometry(
        geometry2d::NurbsCurve2D::make_quarter_circle(2.0),
        {0.0, 1.0},
        {0},
        {{{0.0, 1.0}}},
        {0},
        {0.5},
        true,
        false);
    for (double parameter : {0.13, 0.37, 0.79}) {
        const auto sample = geometry.evaluate_on_span(0, parameter);
        require_close(sample.point.norm(), 2.0, 3.0e-12,
                      "exact NURBS quarter-circle radius");
        require_close(sample.curvature, 0.5, 3.0e-11,
                      "analytic NURBS quarter-circle curvature");
        require((sample.normal - 0.5 * sample.point).norm() <= 3.0e-11,
                "NURBS boundary outward normal is not radial");
    }
}

Interface2D make_periodic_nurbs_circle_interface()
{
    constexpr int point_count = 9;
    constexpr int panel_count = 4;
    const double w = std::sqrt(0.5);
    geometry2d::NurbsCurve2D::ControlPointVector controls{
        {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0},
        {-1.0, 1.0}, {-1.0, 0.0}, {-1.0, -1.0},
        {0.0, -1.0}, {1.0, -1.0}, {1.0, 0.0}};
    geometry2d::NurbsCurve2D curve(
        geometry::NurbsBasis1D(
            2,
            {0.0, 0.0, 0.0,
             0.25, 0.25, 0.5, 0.5, 0.75, 0.75,
             1.0, 1.0, 1.0}),
        std::move(controls),
        {1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0});

    Eigen::MatrixX2d points(point_count, 2);
    Eigen::MatrixX2d normals(point_count, 2);
    Eigen::VectorXd weights = Eigen::VectorXd::Constant(
        point_count, 2.0 * kPi / static_cast<double>(point_count));
    std::vector<double> parameters(static_cast<std::size_t>(point_count));
    for (int q = 0; q < point_count; ++q) {
        const double parameter =
            (static_cast<double>(q) + 0.5)
            / static_cast<double>(point_count);
        parameters[static_cast<std::size_t>(q)] = parameter;
        const Eigen::Vector2d point = curve.evaluate(parameter);
        points.row(q) = point.transpose();
        normals.row(q) = point.normalized().transpose();
    }
    Eigen::MatrixXi panel_points(panel_count, 3);
    std::vector<std::array<double, 2>> panel_parameters;
    for (int panel = 0; panel < panel_count; ++panel) {
        panel_points.row(panel)
            << 2 * panel, 2 * panel + 1, 2 * panel + 2;
        panel_parameters.push_back(
            {parameters[static_cast<std::size_t>(2 * panel)],
             parameters[static_cast<std::size_t>(2 * panel + 2)]});
    }
    std::shared_ptr<const IPanelGeometry2D> provider =
        std::make_shared<geometry2d::NurbsBoundaryPanelGeometry2D>(
            std::move(curve),
            std::vector<double>{0.0, 1.0},
            std::vector<int>(panel_count, 0),
            std::move(panel_parameters),
            std::vector<int>(point_count, 0),
            std::move(parameters),
            true,
            true);
    return Interface2D(
        std::move(points),
        std::move(normals),
        std::move(weights),
        3,
        std::move(panel_points),
        Eigen::VectorXi::Zero(panel_count),
        PanelNodeLayout2D::QuadraticLagrange,
        std::move(provider));
}

void test_nsp_cj_periodic_nurbs_density_space()
{
    const Interface2D iface = make_periodic_nurbs_circle_interface();
    const LaplaceArcLengthBSplineCrossingJetPlan2D plan(
        iface,
        LaplaceCrossingJetScheme2D::NurbsSameParameterCrossingJet);
    require(plan.uses_nurbs_same_parameter()
                && plan.periodic_component_count() == 1
                && plan.open_branch_count() == 0,
            "NSP-CJ did not construct one periodic smooth NURBS branch");
    const Eigen::VectorXd phi =
        Eigen::VectorXd::Constant(iface.num_points(), 1.25);
    const Eigen::VectorXd psi =
        Eigen::VectorXd::Constant(iface.num_points(), -0.4);
    const auto state = plan.fit(
        phi, psi, LaplaceCrossingTraceStencil2D::PhiP3PsiP2);
    const auto jet = plan.evaluate(
        exact_panel_owner(iface, 1, 0.17), state);
    require_close(jet.value, 1.25, 3.0e-12,
                  "periodic NSP-CJ constant phi reproduction");
    require_close(jet.normal_value, -0.4, 3.0e-12,
                  "periodic NSP-CJ constant psi reproduction");
    require_close(jet.value_tangent_derivative, 0.0, 3.0e-11,
                  "periodic NSP-CJ constant phi_s");
    require_close(jet.value_tangent_second_derivative, 0.0, 3.0e-10,
                  "periodic NSP-CJ constant phi_ss");
    require_close(jet.normal_tangent_derivative, 0.0, 3.0e-11,
                  "periodic NSP-CJ constant psi_s");
}

void test_exterior_trace_fixed_density_is_already_a_jump()
{
    constexpr int n = 28;
    constexpr double box_min = -1.5;
    constexpr double box_length = 3.0;
    const double h = box_length / static_cast<double>(n);
    const CartesianGrid2D grid(
        {box_min, box_min}, {h, h}, {n, n}, DofLayout2D::Node);
    auto curve = std::make_shared<UnitCircleCurve>();
    const Interface2D iface =
        CurveResampler2D::discretize_quadratic_lagrange(curve, h, 1.6);

    LaplaceBvpOptions2D options;
    options.formulation = LaplaceBvpFormulation2D::ExteriorTraceCauchy;
    options.correction_method = LaplaceCorrectionMethod2D::CrossingOwner;
    options.crossing_jet_scheme =
        LaplaceCrossingJetScheme2D::ArcLengthBSplineCrossingJet;

    Eigen::VectorXd prescribed_jump(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        prescribed_jump[q] = 0.2 * x + 0.2 * x * y;
    }
    const Eigen::VectorXd bulk_rhs =
        Eigen::VectorXd::Zero(grid.num_dofs());
    const std::vector<Eigen::VectorXd> rhs_derivs(
        static_cast<std::size_t>(iface.num_points()),
        Eigen::VectorXd::Zero(1));

    const auto verify = [&](LaplaceBvpType2D type,
                            LaplaceBvpRestrictMethod2D restrict_method,
                            const LaplaceP2JointPolynomialRestrictOptions2D&
                                joint_options,
                            const std::string& route_name) {
        options.restrict_method = restrict_method;
        options.joint_restrict = joint_options;
        LaplaceBvp2D solver(grid, iface, type, options);
        const LaplaceBvpSolveResult2D result = solver.solve(
            prescribed_jump, bulk_rhs, rhs_derivs, 120, 1.0e-9, 50);
        require(result.converged,
                std::string("exterior-trace fixed-jump convention solve did not converge for ")
                    + route_name);
        // For an exterior physical BVP, the ghost trace is the interior trace,
        // hence ghost-physical is exactly the repository jump convention.
        const Eigen::VectorXd recovered_jump =
            type == LaplaceBvpType2D::ExteriorDirichlet
                ? result.ghost_trace - result.trace
                : result.ghost_normal_trace - result.normal_trace;
        require((recovered_jump - prescribed_jump)
                        .cwiseAbs().maxCoeff()
                    <= 2.0e-12,
                std::string("ExteriorTraceCauchy multiplied an already-oriented fixed density by the physical-side sign for ")
                    + route_name);
    };
    struct RestrictCase {
        std::string name;
        LaplaceBvpRestrictMethod2D method;
        LaplaceP2JointPolynomialRestrictOptions2D joint_options;
    };
    const std::vector<RestrictCase> restrict_cases = {
        {"six-point",
         LaplaceBvpRestrictMethod2D::SixPointQuadraticCrossingOwner,
         {}},
        {"joint-cubic",
         LaplaceBvpRestrictMethod2D::JointPolynomialCrossingOwner,
         {}},
        {"USN-P2",
         LaplaceBvpRestrictMethod2D::
             UnifiedSpatialNormalP2CrossingOwner,
         {}},
        {"USN-P2-DOF-EXT",
         LaplaceBvpRestrictMethod2D::
             UnifiedSpatialNormalP2DofCauchyExterior,
         {}}};
    for (const RestrictCase& restrict_case : restrict_cases) {
        verify(LaplaceBvpType2D::ExteriorDirichlet,
               restrict_case.method,
               restrict_case.joint_options,
               restrict_case.name);
        verify(LaplaceBvpType2D::ExteriorNeumann,
               restrict_case.method,
               restrict_case.joint_options,
               restrict_case.name);
    }
}

struct QuadraticField {
    double constant = 0.41;
    Eigen::Vector2d linear = Eigen::Vector2d(0.73, -0.58);
    Eigen::Matrix2d hessian =
        (Eigen::Matrix2d() << 0.64, -0.37, -0.37, -0.28).finished();

    double value(Eigen::Vector2d point) const
    {
        return constant + linear.dot(point)
             + 0.5 * point.dot(hessian * point);
    }

    Eigen::Vector2d gradient(Eigen::Vector2d point) const
    {
        return linear + hessian * point;
    }
};

Interface2D make_straight_panel(Eigen::Vector2d origin,
                                Eigen::Vector2d tangent,
                                double half_length,
                                bool reverse)
{
    tangent.normalize();
    const Eigen::Vector2d normal(tangent[1], -tangent[0]);
    const Eigen::Vector2d oriented_tangent = reverse ? -tangent : tangent;
    Eigen::MatrixX2d points(7, 2);
    Eigen::MatrixX2d normals(7, 2);
    for (int point = 0; point < 7; ++point) {
        const double coordinate = static_cast<double>(point - 3);
        points.row(point) =
            (origin + half_length * coordinate * oriented_tangent).transpose();
        normals.row(point) = normal.transpose();
    }
    Eigen::MatrixXi connectivity(3, 3);
    connectivity << 0, 1, 2,
                    2, 3, 4,
                    4, 5, 6;
    return Interface2D(
        std::move(points),
        std::move(normals),
        Eigen::VectorXd::Ones(7),
        3,
        std::move(connectivity),
        Eigen::VectorXi::Zero(3),
        PanelNodeLayout2D::QuadraticLagrange);
}

Interface2D make_single_straight_panel()
{
    Eigen::MatrixX2d points(3, 2);
    points << -1.0, 0.0,
               0.0, 0.0,
               1.0, 0.0;
    Eigen::MatrixX2d normals(3, 2);
    normals << 0.0, -1.0,
               0.0, -1.0,
               0.0, -1.0;
    Eigen::MatrixXi connectivity(1, 3);
    connectivity << 0, 1, 2;
    return Interface2D(
        std::move(points),
        std::move(normals),
        Eigen::VectorXd::Ones(3),
        3,
        std::move(connectivity),
        Eigen::VectorXi::Zero(1),
        PanelNodeLayout2D::QuadraticLagrange);
}

void fill_field_jumps(const Interface2D& iface,
                      const QuadraticField& field,
                      double alpha,
                      Eigen::VectorXd& value_jump,
                      Eigen::VectorXd& normal_jump,
                      Eigen::VectorXd& rhs_jump)
{
    value_jump.resize(iface.num_points());
    normal_jump.resize(iface.num_points());
    rhs_jump.resize(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        const Eigen::Vector2d point = iface.points().row(q).transpose();
        Eigen::Vector2d normal = iface.normals().row(q).transpose();
        normal.normalize();
        value_jump[q] = field.value(point);
        normal_jump[q] = field.gradient(point).dot(normal);
        rhs_jump[q] = -field.hessian.trace() + alpha * field.value(point);
    }
}

void test_unified_spatial_normal_p2_quadratic_reproduction()
{
    constexpr int n = 28;
    constexpr double box_min = -1.5;
    constexpr double box_length = 3.0;
    const double h = box_length / static_cast<double>(n);
    const CartesianGrid2D grid(
        {box_min, box_min}, {h, h}, {n, n}, DofLayout2D::Node);
    auto curve = std::make_shared<UnitCircleCurve>();
    const Interface2D iface =
        CurveResampler2D::discretize_quadratic_lagrange(curve, h, 1.6);
    const GridPair2D pair(grid, iface);

    std::vector<LaplaceJumpData2D> zero_jumps(
        static_cast<std::size_t>(iface.num_points()));
    for (LaplaceJumpData2D& jump : zero_jumps)
        jump.rhs_derivs = Eigen::VectorXd::Zero(1);
    LaplaceQuadraticPanelCenterSpread2D spread(
        pair, 0.0, LaplaceCorrectionMethod2D::CrossingOwner);
    Eigen::VectorXd rhs_correction =
        Eigen::VectorXd::Zero(grid.num_dofs());
    const LaplaceSpreadResult2D spread_result =
        spread.apply(zero_jumps, rhs_correction);

    const QuadraticField field;
    Eigen::VectorXd bulk(grid.num_dofs());
    for (int node = 0; node < grid.num_dofs(); ++node) {
        bulk[node] = field.value(structured_grid::point(grid, node));
    }
    const std::vector<std::pair<
        std::string,
        LaplaceP2JointPolynomialRestrictOptions2D>> routes = {
        {"USN-P2",
         make_laplace_p2_unified_spatial_normal_restrict_options_2d()},
        {"USN-P2-DOF-EXT",
         make_laplace_p2_dof_cauchy_exterior_restrict_options_2d()}};
    for (const auto& route : routes) {
        LaplaceP2CrossingOwnerJointPolynomialRestrict2D restrict(
            pair, route.second);
        const std::vector<LocalPoly2D> recovered =
            restrict.apply(bulk, spread_result);

        double max_value_error = 0.0;
        double max_normal_error = 0.0;
        for (int q = 0; q < iface.num_points(); ++q) {
            const Eigen::Vector2d point =
                iface.points().row(q).transpose();
            Eigen::Vector2d normal = iface.normals().row(q).transpose();
            normal.normalize();
            const LocalPoly2D& local =
                recovered[static_cast<std::size_t>(q)];
            max_value_error = std::max(
                max_value_error,
                std::abs(local.coeffs[0] - field.value(point)));
            const double recovered_normal =
                local.coeffs[1] * normal[0]
                + local.coeffs[2] * normal[1];
            max_normal_error = std::max(
                max_normal_error,
                std::abs(recovered_normal
                         - field.gradient(point).dot(normal)));
        }
        require(max_value_error <= 2.0e-11,
                route.first
                    + " did not reproduce a complete spatial quadratic value");
        require(max_normal_error <= 2.0e-10,
                route.first
                    + " did not reproduce the quadratic normal derivative");

        const auto& diagnostics = restrict.diagnostics();
        require(diagnostics.unified_p2_stencils
                    == 2 * iface.num_points(),
                route.first
                    + " did not build exactly one shared stencil per trace side");
        require(diagnostics.shared_side_spatial_polynomials
                    == 2 * iface.num_points(),
                route.first
                    + " did not share its spatial polynomial across normal layers");
        require(diagnostics.gap_fallback_owners == 0
                    && diagnostics.endpoint_fallback_owners == 0,
                route.first + " used a forbidden gap or endpoint fallback");
        require(std::isfinite(diagnostics.max_six_point_weight_l1)
                    && diagnostics.max_six_point_weight_l1 < 5.0,
                route.first + " interpolation weights are unstable");
    }
}

void test_dof_cauchy_exterior_constant_jump_reproduction()
{
    constexpr int n = 28;
    constexpr double box_min = -1.5;
    constexpr double box_length = 3.0;
    constexpr double value_jump = 0.37;
    const double h = box_length / static_cast<double>(n);
    const CartesianGrid2D grid(
        {box_min, box_min}, {h, h}, {n, n}, DofLayout2D::Node);
    auto curve = std::make_shared<UnitCircleCurve>();
    const Interface2D iface =
        CurveResampler2D::discretize_quadratic_lagrange(curve, h, 1.6);
    const GridPair2D pair(grid, iface);

    std::vector<LaplaceJumpData2D> jumps(
        static_cast<std::size_t>(iface.num_points()));
    for (LaplaceJumpData2D& jump : jumps) {
        jump.u_jump = value_jump;
        jump.un_jump = 0.0;
        jump.rhs_derivs = Eigen::VectorXd::Zero(1);
    }
    LaplaceQuadraticPanelCenterSpread2D spread(
        pair, 0.0, LaplaceCorrectionMethod2D::CrossingOwner);
    Eigen::VectorXd rhs_correction =
        Eigen::VectorXd::Zero(grid.num_dofs());
    const LaplaceSpreadResult2D spread_result =
        spread.apply(jumps, rhs_correction);

    const QuadraticField exterior_field;
    Eigen::VectorXd bulk(grid.num_dofs());
    for (int node = 0; node < grid.num_dofs(); ++node) {
        const Eigen::Vector2d point =
            structured_grid::point(grid, node);
        bulk[node] = exterior_field.value(point)
                   + (pair.domain_label(node) > 0 ? value_jump : 0.0);
    }

    LaplaceP2CrossingOwnerJointPolynomialRestrict2D restrict(
        pair,
        make_laplace_p2_dof_cauchy_exterior_restrict_options_2d());
    const std::vector<LocalPoly2D> average =
        restrict.apply(bulk, spread_result);
    const std::vector<LocalPoly2D> exterior =
        restrict.apply_exterior_virtual(bulk, spread_result);
    const std::vector<LocalPoly2D> interior =
        restrict.apply_interior_virtual(bulk, spread_result);

    double max_error = 0.0;
    for (int q = 0; q < iface.num_points(); ++q) {
        const Eigen::Vector2d point =
            iface.points().row(q).transpose();
        Eigen::Vector2d normal = iface.normals().row(q).transpose();
        normal.normalize();
        const double exterior_value = exterior_field.value(point);
        const double expected_normal =
            exterior_field.gradient(point).dot(normal);
        const auto check = [&](const LocalPoly2D& poly,
                               double expected_value) {
            max_error = std::max(
                max_error,
                std::abs(poly.coeffs[0] - expected_value));
            const double recovered_normal =
                poly.coeffs[1] * normal[0]
                + poly.coeffs[2] * normal[1];
            max_error = std::max(
                max_error,
                std::abs(recovered_normal - expected_normal));
        };
        check(exterior[static_cast<std::size_t>(q)], exterior_value);
        check(average[static_cast<std::size_t>(q)],
              exterior_value + 0.5 * value_jump);
        check(interior[static_cast<std::size_t>(q)],
              exterior_value + value_jump);
    }
    require(max_error <= 2.0e-10,
            "USN-P2-DOF-EXT failed exterior/average/interior trace reproduction for a constant Cauchy jump");
}

void test_rotated_straight_panel_reproduction()
{
    const Eigen::Vector2d origin(0.17, -0.23);
    Eigen::Vector2d tangent(std::cos(0.61), std::sin(0.61));
    tangent.normalize();
    const double half_length = 0.74;
    const double local_s = 0.37;
    const double alpha = 0.43;
    const QuadraticField field;

    const Interface2D iface =
        make_straight_panel(origin, tangent, half_length, false);
    Eigen::VectorXd value_jump;
    Eigen::VectorXd normal_jump;
    Eigen::VectorXd rhs_jump;
    fill_field_jumps(
        iface, field, alpha, value_jump, normal_jump, rhs_jump);

    P2CrossingOwner2D crossing;
    crossing.panel_index = 1;
    crossing.local_s = local_s;
    crossing.crossing_point =
        geometry2d::panel_point(iface, 1, local_s);
    crossing.crossing_normal =
        geometry2d::panel_normal(iface, 1, local_s);
    crossing.exact_intersection_count = 1;
    crossing.status = P2CrossingOwnerStatus2D::ExactIntersection;

    const LaplaceP2CrossingLocalPolynomial2D local =
        build_laplace_p2_crossing_local_polynomial_2d(
            iface,
            crossing,
            value_jump,
            normal_jump,
            rhs_jump,
            alpha);
    const LocalPoly2D cartesian = local.cartesian_taylor();
    const Eigen::Vector2d center = crossing.crossing_point;
    const Eigen::Vector2d expected_gradient = field.gradient(center);

    require_close(cartesian.center[0], center[0], 2.0e-13,
                  "crossing center x");
    require_close(cartesian.center[1], center[1], 2.0e-13,
                  "crossing center y");
    require_close(cartesian.coeffs[0], field.value(center), 2.0e-12,
                  "crossing value reproduction");
    require_close(cartesian.coeffs[1], expected_gradient[0], 2.0e-12,
                  "crossing x-gradient reproduction");
    require_close(cartesian.coeffs[2], expected_gradient[1], 2.0e-12,
                  "crossing y-gradient reproduction");
    require_close(cartesian.coeffs[3], field.hessian(0, 0), 2.0e-12,
                  "crossing xx-Hessian reproduction");
    require_close(cartesian.coeffs[4], field.hessian(0, 1), 2.0e-12,
                  "crossing xy-Hessian reproduction");
    require_close(cartesian.coeffs[5], field.hessian(1, 1), 2.0e-12,
                  "crossing yy-Hessian reproduction");

    const Eigen::Vector2d normal(tangent[1], -tangent[0]);
    const Eigen::Vector2d query = center + 0.29 * tangent - 0.18 * normal;
    require_close(local.evaluate(query), field.value(query), 3.0e-12,
                  "local-coordinate quadratic evaluation");
    require_close(local.evaluate(query),
                  evaluate_taylor_poly_2d(cartesian, query),
                  3.0e-13,
                  "local and Cartesian representations");
    require_close(local.evaluate(center + 0.21 * normal),
                  local.evaluate_on_normal(0.21),
                  3.0e-13,
                  "normal polynomial is the xi=0 restriction");

    const Interface2D reversed =
        make_straight_panel(origin, tangent, half_length, true);
    Eigen::VectorXd reversed_value;
    Eigen::VectorXd reversed_normal;
    Eigen::VectorXd reversed_rhs;
    fill_field_jumps(
        reversed,
        field,
        alpha,
        reversed_value,
        reversed_normal,
        reversed_rhs);
    crossing.local_s = -local_s;
    crossing.crossing_point =
        geometry2d::panel_point(reversed, 1, -local_s);
    crossing.crossing_normal =
        geometry2d::panel_normal(reversed, 1, -local_s);
    const LocalPoly2D reversed_cartesian =
        build_laplace_p2_crossing_local_polynomial_2d(
            reversed,
            crossing,
            reversed_value,
            reversed_normal,
            reversed_rhs,
            alpha)
            .cartesian_taylor();
    require(
        (reversed_cartesian.coeffs - cartesian.coeffs).cwiseAbs().maxCoeff()
            <= 3.0e-12,
        "Cartesian crossing polynomial changed when panel orientation reversed");

    crossing.exact_intersection_count = 2;
    require(!can_build_laplace_p2_crossing_local_polynomial_2d(
                reversed, crossing),
            "multiple crossing must not use one local polynomial");
}

void test_curved_panel_geometry_terms()
{
    constexpr double bend = 0.38;
    Eigen::MatrixX2d points(7, 2);
    Eigen::MatrixX2d normals(7, 2);
    Eigen::VectorXd value_jump(7);
    Eigen::VectorXd normal_jump = Eigen::VectorXd::Constant(7, -1.0);
    Eigen::VectorXd rhs_jump = Eigen::VectorXd::Zero(7);
    for (int point = 0; point < 7; ++point) {
        const double x = static_cast<double>(point - 3);
        const Eigen::Vector2d tangent(1.0, 2.0 * bend * x);
        const double speed = tangent.norm();
        const Eigen::Vector2d normal(
            tangent[1] / speed, -tangent[0] / speed);
        points.row(point) << x, bend * x * x;
        normals.row(point) = normal.transpose();

        // Prescribe A(l)=bend*l^2 in signed physical arclength.  Its P3
        // reconstruction has A_ll=2*bend exactly at the center, while the
        // P2 Neumann reconstruction is the constant B=-1.
        const double arclength =
            0.5
            * (x * std::sqrt(1.0 + 4.0 * bend * bend * x * x)
               + std::asinh(2.0 * bend * x) / (2.0 * bend));
        value_jump[point] = bend * arclength * arclength;
    }
    Eigen::MatrixXi connectivity(3, 3);
    connectivity << 0, 1, 2,
                    2, 3, 4,
                    4, 5, 6;
    const Interface2D iface(
        std::move(points),
        std::move(normals),
        Eigen::VectorXd::Ones(7),
        3,
        std::move(connectivity),
        Eigen::VectorXi::Zero(3),
        PanelNodeLayout2D::QuadraticLagrange);

    P2CrossingOwner2D crossing;
    crossing.panel_index = 1;
    crossing.local_s = 0.0;
    crossing.crossing_point = Eigen::Vector2d::Zero();
    crossing.crossing_normal = Eigen::Vector2d(0.0, -1.0);
    crossing.exact_intersection_count = 1;
    crossing.status = P2CrossingOwnerStatus2D::ExactIntersection;

    const LaplaceP2CrossingLocalPolynomial2D local =
        build_laplace_p2_crossing_local_polynomial_2d(
            iface,
            crossing,
            value_jump,
            normal_jump,
            rhs_jump,
            0.0);
    const LocalPoly2D cartesian = local.cartesian_taylor();
    require_close(cartesian.coeffs[0], 0.0, 2.0e-12,
                  "curved-panel value");
    require_close(cartesian.coeffs[1], 0.0, 2.0e-12,
                  "curved-panel x-gradient");
    require_close(cartesian.coeffs[2], 1.0, 2.0e-12,
                  "curved-panel y-gradient");
    require_close(cartesian.coeffs[3], 0.0, 2.0e-12,
                  "curvature cancellation in xx-Hessian");
    require_close(cartesian.coeffs[4], 0.0, 2.0e-12,
                  "curvature cancellation in xy-Hessian");
    require_close(cartesian.coeffs[5], 0.0, 2.0e-12,
                  "curvature cancellation in yy-Hessian");
    require_close(local.evaluate(Eigen::Vector2d(0.21, -0.17)),
                  -0.17,
                  2.0e-12,
                  "curved-panel local polynomial reproduction");
}

void test_kfbi_trace_degree_hierarchy()
{
    const Interface2D iface = make_straight_panel(
        Eigen::Vector2d::Zero(), Eigen::Vector2d::UnitX(), 1.0, false);
    Eigen::VectorXd value_jump =
        Eigen::VectorXd::Zero(iface.num_points());
    Eigen::VectorXd normal_jump(iface.num_points());
    Eigen::VectorXd rhs_jump =
        Eigen::VectorXd::Zero(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        const double x = iface.points()(q, 0);
        normal_jump[q] = x * x;
    }

    P2CrossingOwner2D crossing;
    crossing.panel_index = 1;
    crossing.local_s = 0.37;
    crossing.crossing_point = Eigen::Vector2d(0.37, 0.0);
    crossing.crossing_normal = Eigen::Vector2d(0.0, -1.0);
    crossing.exact_intersection_count = 1;
    crossing.status = P2CrossingOwnerStatus2D::ExactIntersection;

    const LaplaceP2CrossingLocalPolynomial2D local =
        build_laplace_p2_crossing_local_polynomial_2d(
            iface,
            crossing,
            value_jump,
            normal_jump,
            rhs_jump,
            0.0);

    // Psi=x^2 must be reproduced exactly by its P2 arclength fit.
    require_close(local.normal_derivative,
                  0.37 * 0.37,
                  2.0e-13,
                  "normal jump did not use the requested P2 trace fit");
    require_close(local.hessian_tn,
                  2.0 * 0.37,
                  2.0e-13,
                  "P2 normal-jump first derivative");

    for (int q = 0; q < iface.num_points(); ++q) {
        const double x = iface.points()(q, 0);
        value_jump[q] = x * x * x;
    }
    normal_jump.setZero();
    const LaplaceP2CrossingLocalPolynomial2D zero_normal =
        build_laplace_p2_crossing_local_polynomial_2d(
            iface,
            crossing,
            value_jump,
            normal_jump,
            rhs_jump,
            0.0);
    require_close(zero_normal.normal_derivative,
                  0.0,
                  1.0e-14,
                  "zero Neumann jump block");
    require_close(zero_normal.hessian_tn,
                  0.0,
                  1.0e-14,
                  "zero Neumann derivative block");
    require_close(zero_normal.hessian_tt,
                  6.0 * 0.37,
                  1.0e-13,
                  "P3 value-jump second derivative");
    require_close(zero_normal.hessian_nn,
                  -6.0 * 0.37,
                  1.0e-13,
                  "PDE closure with zero Neumann jump");
}

void test_bvp_selected_trace_stencils()
{
    require(laplace_crossing_trace_stencil_for_bvp_2d(
                LaplaceBvpType2D::InteriorDirichlet)
                == LaplaceCrossingTraceStencil2D::PhiP3
            && laplace_crossing_trace_stencil_for_bvp_2d(
                   LaplaceBvpType2D::ExteriorDirichlet)
                   == LaplaceCrossingTraceStencil2D::PhiP3,
            "Dirichlet BVP types must select phi-P3");
    require(laplace_crossing_trace_stencil_for_bvp_2d(
                LaplaceBvpType2D::InteriorNeumann)
                == LaplaceCrossingTraceStencil2D::PsiP2
            && laplace_crossing_trace_stencil_for_bvp_2d(
                   LaplaceBvpType2D::ExteriorNeumann)
                   == LaplaceCrossingTraceStencil2D::PsiP2,
            "Neumann BVP types must select psi-P2");

    const Interface2D psi_iface = make_single_straight_panel();
    P2CrossingOwner2D crossing;
    crossing.panel_index = 0;
    crossing.local_s = 0.37;
    crossing.crossing_point = Eigen::Vector2d(0.37, 0.0);
    crossing.crossing_normal = Eigen::Vector2d(0.0, -1.0);
    crossing.exact_intersection_count = 1;
    crossing.status = P2CrossingOwnerStatus2D::ExactIntersection;

    require(can_build_laplace_p2_crossing_local_polynomial_2d(
                psi_iface,
                crossing,
                LaplaceCrossingTraceStencil2D::PsiP2),
            "Neumann psi-P2 stencil should need only one three-DOF panel");
    require(!can_build_laplace_p2_crossing_local_polynomial_2d(
                psi_iface,
                crossing,
                LaplaceCrossingTraceStencil2D::PhiP3),
            "Dirichlet phi-P3 stencil must require a fourth smooth DOF");
    require(!can_build_laplace_p2_crossing_local_polynomial_2d(
                psi_iface,
                crossing,
                LaplaceCrossingTraceStencil2D::PhiP3PsiP2),
            "combined stencil must retain the phi-P3 four-DOF requirement");
    require(can_build_laplace_p2_crossing_local_polynomial_2d(
                psi_iface,
                crossing,
                LaplaceCrossingTraceStencil2D::RhsOnly),
            "RHS-only crossing must not require trace interpolation DOFs");

    Eigen::VectorXd phi(psi_iface.num_points());
    Eigen::VectorXd psi(psi_iface.num_points());
    const Eigen::VectorXd rhs =
        Eigen::VectorXd::Zero(psi_iface.num_points());
    for (int q = 0; q < psi_iface.num_points(); ++q) {
        const double x = psi_iface.points()(q, 0);
        phi[q] = 5.0 + x * x;
        psi[q] = 1.0 - 0.4 * x + 0.7 * x * x;
    }
    const LaplaceP2CrossingLocalPolynomial2D psi_only =
        build_laplace_p2_crossing_local_polynomial_2d(
            psi_iface,
            crossing,
            phi,
            psi,
            rhs,
            0.0,
            LaplaceCrossingTraceStencil2D::PsiP2);
    require_close(psi_only.value, 0.0, 1.0e-14,
                  "Neumann stencil must ignore a nonzero phi vector");
    require_close(psi_only.tangent_derivative, 0.0, 1.0e-14,
                  "Neumann stencil must omit the phi derivative block");
    require_close(psi_only.normal_derivative,
                  1.0 - 0.4 * 0.37 + 0.7 * 0.37 * 0.37,
                  2.0e-13,
                  "Neumann stencil psi-P2 value");
    require_close(psi_only.hessian_tn,
                  -0.4 + 1.4 * 0.37,
                  2.0e-13,
                  "Neumann stencil psi-P2 derivative");

    const Eigen::VectorXd constant_rhs =
        Eigen::VectorXd::Constant(psi_iface.num_points(), 2.5);
    const LaplaceP2CrossingLocalPolynomial2D rhs_only =
        build_laplace_p2_crossing_local_polynomial_2d(
            psi_iface,
            crossing,
            phi,
            psi,
            constant_rhs,
            0.0,
            LaplaceCrossingTraceStencil2D::RhsOnly);
    require_close(rhs_only.value, 0.0, 1.0e-14,
                  "RHS-only stencil value jump");
    require_close(rhs_only.normal_derivative, 0.0, 1.0e-14,
                  "RHS-only stencil normal jump");
    require_close(rhs_only.hessian_nn, -2.5, 1.0e-14,
                  "RHS-only PDE closure");

    const Interface2D phi_iface = make_straight_panel(
        Eigen::Vector2d::Zero(), Eigen::Vector2d::UnitX(), 1.0, false);
    crossing.panel_index = 1;
    crossing.crossing_point = Eigen::Vector2d(0.37, 0.0);
    phi.resize(phi_iface.num_points());
    psi.resize(phi_iface.num_points());
    const Eigen::VectorXd phi_rhs =
        Eigen::VectorXd::Zero(phi_iface.num_points());
    for (int q = 0; q < phi_iface.num_points(); ++q) {
        const double x = phi_iface.points()(q, 0);
        phi[q] = x * x * x;
        psi[q] = 9.0 - 2.0 * x;
    }
    const LaplaceP2CrossingLocalPolynomial2D phi_only =
        build_laplace_p2_crossing_local_polynomial_2d(
            phi_iface,
            crossing,
            phi,
            psi,
            phi_rhs,
            0.0,
            LaplaceCrossingTraceStencil2D::PhiP3);
    require_close(phi_only.normal_derivative, 0.0, 1.0e-14,
                  "Dirichlet stencil must ignore a nonzero psi vector");
    require_close(phi_only.hessian_tn, 0.0, 1.0e-14,
                  "Dirichlet stencil must omit the psi derivative block");
    require_close(phi_only.hessian_tt,
                  6.0 * 0.37,
                  2.0e-13,
                  "Dirichlet stencil phi-P3 second derivative");
}

Interface2D make_offset_box_interface()
{
    Eigen::MatrixX2d points(8, 2);
    points << -0.63, -0.57,
               0.04, -0.57,
               0.71, -0.57,
               0.71,  0.045,
               0.71,  0.66,
               0.04,  0.66,
              -0.63,  0.66,
              -0.63,  0.045;

    constexpr double inv_sqrt_two = 0.70710678118654752440;
    Eigen::MatrixX2d normals(8, 2);
    normals << -inv_sqrt_two, -inv_sqrt_two,
                0.0, -1.0,
                inv_sqrt_two, -inv_sqrt_two,
                1.0, 0.0,
                inv_sqrt_two, inv_sqrt_two,
                0.0, 1.0,
               -inv_sqrt_two, inv_sqrt_two,
               -1.0, 0.0;

    Eigen::MatrixXi connectivity(4, 3);
    connectivity << 0, 1, 2,
                    2, 3, 4,
                    4, 5, 6,
                    6, 7, 0;
    return Interface2D(
        std::move(points),
        std::move(normals),
        Eigen::VectorXd::Ones(8),
        3,
        std::move(connectivity),
        Eigen::VectorXi::Zero(4),
        PanelNodeLayout2D::QuadraticLagrange);
}

void test_spread_uses_crossing_local_polynomial()
{
    const Interface2D iface = make_offset_box_interface();
    const CartesianGrid2D grid(
        {-1.25, -1.25}, {0.25, 0.25}, {10, 10}, DofLayout2D::Node);
    const GridPair2D pair(grid, iface);
    const double alpha = 0.31;
    const QuadraticField field;

    Eigen::VectorXd value_jump;
    Eigen::VectorXd normal_jump;
    Eigen::VectorXd rhs_jump;
    fill_field_jumps(
        iface, field, alpha, value_jump, normal_jump, rhs_jump);
    std::vector<LaplaceJumpData2D> jumps(
        static_cast<std::size_t>(iface.num_points()));
    for (int q = 0; q < iface.num_points(); ++q) {
        jumps[static_cast<std::size_t>(q)].u_jump = value_jump[q];
        jumps[static_cast<std::size_t>(q)].un_jump = normal_jump[q];
        jumps[static_cast<std::size_t>(q)].rhs_derivs.resize(1);
        jumps[static_cast<std::size_t>(q)].rhs_derivs[0] = rhs_jump[q];
    }

    LaplaceQuadraticPanelCenterSpread2D spread(
        pair,
        alpha,
        LaplaceCorrectionMethod2D::CrossingOwner);
    Eigen::VectorXd actual = Eigen::VectorXd::Zero(grid.num_dofs());
    const LaplaceSpreadResult2D result = spread.apply(jumps, actual);
    const LaplaceCorrectionSupport2D support =
        build_laplace_correction_support_2d(
            pair, "laplace_crossing_local_polynomial_2d_test");

    Eigen::VectorXd expected = Eigen::VectorXd::Zero(grid.num_dofs());
    Eigen::VectorXd center_owner = Eigen::VectorXd::Zero(grid.num_dofs());
    int local_crossings = 0;
    for (const LaplaceCrossingCorrectionOp& op : support.crossing_ops) {
        if (op.kind != LaplaceCrossingKind::InterfaceJump)
            continue;
        const P2CrossingOwner2D owner =
            pair.p2_crossing_owner_between(
                op.rhs_node, op.correction_node);
        const Eigen::Vector2d point = structured_grid::point(
            grid, op.correction_node);
        double correction = 0.0;
        if (can_build_laplace_p2_crossing_local_polynomial_2d(
                iface, owner)) {
            correction =
                build_laplace_p2_crossing_local_polynomial_2d(
                    iface,
                    owner,
                    value_jump,
                    normal_jump,
                    rhs_jump,
                    alpha)
                    .evaluate(point);
            ++local_crossings;
        } else {
            correction = evaluate_taylor_poly_2d(
                result.correction_polys[
                    static_cast<std::size_t>(owner.center_index)],
                point);
        }
        const double scale =
            static_cast<double>(op.side_delta) * op.stencil_weight;
        expected[op.rhs_node] += scale * correction;
        center_owner[op.rhs_node] += scale
            * evaluate_taylor_poly_2d(
                result.correction_polys[
                    static_cast<std::size_t>(owner.center_index)],
                point);
    }

    require(local_crossings > 0,
            "spread test did not find an exact P2 crossing");
    require((actual - expected).cwiseAbs().maxCoeff() <= 2.0e-11,
            "crossing-owner spread did not evaluate crossing-local polynomials");
    require((actual - center_owner).cwiseAbs().maxCoeff() > 1.0e-8,
            "crossing-local spread unexpectedly collapsed to fixed-center evaluation");
}

} // namespace

int main()
{
    try {
        test_als_cj_periodic_continuity_and_accuracy();
        test_als_cj_open_branch_polynomial_reproduction();
        test_nsp_cj_same_parameter_and_gap_crossing_reproduction();
        test_nurbs_boundary_analytic_curvature();
        test_nsp_cj_periodic_nurbs_density_space();
        test_exterior_trace_fixed_density_is_already_a_jump();
        test_unified_spatial_normal_p2_quadratic_reproduction();
        test_dof_cauchy_exterior_constant_jump_reproduction();
        test_rotated_straight_panel_reproduction();
        test_curved_panel_geometry_terms();
        test_kfbi_trace_degree_hierarchy();
        test_bvp_selected_trace_stencils();
        test_spread_uses_crossing_local_polynomial();
        std::cout << "2D crossing-local polynomial tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "2D crossing-local polynomial test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
