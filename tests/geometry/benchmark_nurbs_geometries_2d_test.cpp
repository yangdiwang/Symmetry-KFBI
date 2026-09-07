#include "src/support/geometry/benchmark_nurbs_geometries_2d.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using kfbim::Interface2D;
using kfbim::app2d::BenchmarkGeometryContinuity2D;
using kfbim::app2d::BenchmarkNurbsGeometry2D;
using kfbim::app2d::BenchmarkNurbsGeometryKind2D;
using kfbim::app2d::benchmark_nurbs_provider_2d;
using kfbim::app2d::make_benchmark_nurbs_geometry_2d;
using kfbim::app2d::make_benchmark_nurbs_geometry_for_panel_length_2d;

constexpr double kPi = 3.141592653589793238462643383279502884;

const std::array<BenchmarkNurbsGeometryKind2D, 5> kKinds{{
    BenchmarkNurbsGeometryKind2D::Circle,
    BenchmarkNurbsGeometryKind2D::Ellipse,
    BenchmarkNurbsGeometryKind2D::Flower,
    BenchmarkNurbsGeometryKind2D::Heart,
    BenchmarkNurbsGeometryKind2D::LShape
}};

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void require_near(double actual,
                  double expected,
                  double tolerance,
                  const std::string& message)
{
    require(std::abs(actual - expected) <= tolerance, message);
}

void require_near(Eigen::Vector2d actual,
                  Eigen::Vector2d expected,
                  double tolerance,
                  const std::string& message)
{
    require((actual - expected).norm() <= tolerance, message);
}

Eigen::Vector2d rotate(Eigen::Vector2d point, double angle)
{
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    return {cosine * point[0] - sine * point[1],
            sine * point[0] + cosine * point[1]};
}

Eigen::Vector2d one_sided_unit_tangent(
    const kfbim::geometry2d::NurbsCurve2D& curve,
    double parameter,
    double toward)
{
    Eigen::Vector2d tangent = curve.derivative(
        std::nextafter(parameter, toward));
    return tangent / tangent.norm();
}

double one_sided_curvature(
    const kfbim::geometry2d::NurbsCurve2D& curve,
    double parameter,
    double toward)
{
    parameter = std::nextafter(parameter, toward);
    const Eigen::Vector2d derivative = curve.derivative(parameter);
    const Eigen::Vector2d second = curve.second_derivative(parameter);
    const double speed = derivative.norm();
    const Eigen::Vector2d tangent = derivative / speed;
    const Eigen::Vector2d outward(tangent[1], -tangent[0]);
    return -second.dot(outward) / (speed * speed);
}

template <class Function>
void require_invalid_argument(Function&& function,
                              const std::string& message)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_names_and_parse()
{
    const std::array<std::string, 5> names{{
        "circle", "ellipse", "flower", "heart", "lshape"
    }};
    for (std::size_t i = 0; i < kKinds.size(); ++i) {
        require(
            kfbim::app2d::benchmark_nurbs_geometry_name_2d(kKinds[i])
                == names[i],
            "benchmark geometry name is unstable");
        require(
            kfbim::app2d::parse_benchmark_nurbs_geometry_2d(names[i])
                == kKinds[i],
            "benchmark geometry parser does not invert its name");
    }
    require(
        kfbim::app2d::parse_benchmark_nurbs_geometry_2d("l_shape")
            == BenchmarkNurbsGeometryKind2D::LShape,
        "L-shape parser alias is missing");
    require_invalid_argument(
        [] {
            (void)kfbim::app2d::parse_benchmark_nurbs_geometry_2d("unknown");
        },
        "unknown benchmark geometry was accepted");
    require_invalid_argument(
        [] {
            (void)make_benchmark_nurbs_geometry_2d(
                BenchmarkNurbsGeometryKind2D::Circle, 0.0);
        },
        "zero grid spacing was accepted");
    require_invalid_argument(
        [] {
            (void)make_benchmark_nurbs_geometry_for_panel_length_2d(
                BenchmarkNurbsGeometryKind2D::Circle, -1.0);
        },
        "negative panel length was accepted");
}

void test_provider_and_compatibility_panels()
{
    for (BenchmarkNurbsGeometryKind2D kind : kKinds) {
        const BenchmarkNurbsGeometry2D geometry =
            make_benchmark_nurbs_geometry_for_panel_length_2d(kind, 0.16);
        const Interface2D& interface = geometry.interface;
        const auto& provider = benchmark_nurbs_provider_2d(geometry);

        require(interface.has_panel_geometry(),
                "benchmark interface lost its exact provider");
        require(provider.closed(), "benchmark boundary is not closed");
        require(interface.points_per_panel() == 3,
                "benchmark compatibility cells are not P2");
        require(interface.num_panels()
                    == static_cast<int>(geometry.active_trace_points.size()),
                "there is not exactly one trace candidate per panel");
        require(geometry.maximum_panel_length
                    <= 1.000000001 * geometry.target_panel_length,
                "compatibility panel exceeds requested physical length");

        double active_weight_sum = 0.0;
        for (int panel = 0; panel < interface.num_panels(); ++panel) {
            const int candidate =
                geometry.active_trace_points[static_cast<std::size_t>(panel)];
            require(interface.point_index(panel, 1) == candidate,
                    "trace candidate is not the panel interior point");
            require(!interface.is_corner_point(candidate),
                    "feature point leaked into trace candidates");
            require(interface.weights()[candidate] > 0.0,
                    "trace candidate has nonpositive arc-length weight");
            active_weight_sum += interface.weights()[candidate];

            for (int local = 0; local < 3; ++local) {
                const double local_s = static_cast<double>(local - 1);
                const int point = interface.point_index(panel, local);
                require_near(
                    provider.point(panel, local_s),
                    interface.points().row(point).transpose(),
                    2.0e-13,
                    "compatibility point is not on authoritative NURBS");
                require_near(
                    provider.normal(panel, local_s).norm(),
                    1.0,
                    2.0e-12,
                    "provider returned a nonunit normal");
            }
            for (double local_s : {-0.71, -0.17, 0.36, 0.83}) {
                const Eigen::Vector2d point = provider.point(panel, local_s);
                const Eigen::Vector2d normal = provider.normal(panel, local_s);
                require(point.allFinite() && normal.allFinite(),
                        "interior exact-NURBS panel query is nonfinite");
                require_near(normal.norm(), 1.0, 2.0e-12,
                             "interior provider normal is not unit");
            }
        }
        require(active_weight_sum > 0.0,
                "benchmark trace quadrature has zero total length");

        const auto& curve = provider.curve();
        double twice_signed_area = 0.0;
        Eigen::Vector2d previous = curve.evaluate(curve.domain_start());
        constexpr int area_samples = 800;
        for (int sample = 1; sample <= area_samples; ++sample) {
            const double parameter = curve.domain_start()
                + (curve.domain_end() - curve.domain_start())
                    * static_cast<double>(sample)
                    / static_cast<double>(area_samples);
            const Eigen::Vector2d current = curve.evaluate(parameter);
            twice_signed_area += previous[0] * current[1]
                               - previous[1] * current[0];
            previous = current;
        }
        require(twice_signed_area > 0.0,
                "benchmark NURBS is not counter-clockwise");

        int next_panel = 0;
        for (int span = 0; span < provider.num_spans(); ++span) {
            const auto span_interval = provider.span_interval(span);
            require(next_panel < interface.num_panels(),
                    "NURBS span has no compatibility panels");
            require(provider.panel_span(next_panel) == span,
                    "compatibility panels are not ordered by NURBS span");
            require_near(
                provider.panel_parameter(next_panel, -1.0),
                span_interval.first,
                2.0e-13,
                "first compatibility panel does not start at span boundary");
            double previous_end = span_interval.first;
            while (next_panel < interface.num_panels()
                   && provider.panel_span(next_panel) == span) {
                const double start =
                    provider.panel_parameter(next_panel, -1.0);
                const double end =
                    provider.panel_parameter(next_panel, 1.0);
                require_near(start, previous_end, 2.0e-13,
                             "compatibility panel coverage has a gap");
                require(end > start,
                        "compatibility panel parameter interval is empty");
                previous_end = end;
                ++next_panel;
            }
            require_near(previous_end, span_interval.second, 2.0e-13,
                         "last compatibility panel misses span endpoint");
        }
        require(next_panel == interface.num_panels(),
                "compatibility panel has no authoritative NURBS span");
    }
}

void test_feature_metadata()
{
    const std::array<int, 5> expected_features{{0, 0, 0, 4, 6}};
    const std::array<int, 5> expected_g1{{0, 0, 0, 2, 0}};
    const std::array<int, 5> expected_g0{{0, 0, 0, 2, 6}};
    for (std::size_t i = 0; i < kKinds.size(); ++i) {
        const BenchmarkNurbsGeometry2D geometry =
            make_benchmark_nurbs_geometry_for_panel_length_2d(
                kKinds[i], 0.18);
        const Interface2D& interface = geometry.interface;
        require(static_cast<int>(geometry.features.size())
                    == expected_features[i],
                "benchmark feature count is wrong");
        require(static_cast<int>(interface.corners().size())
                    == expected_features[i],
                "Interface2D feature metadata count is wrong");

        int g1 = 0;
        int g0 = 0;
        for (int feature = 0;
             feature < static_cast<int>(geometry.features.size());
             ++feature) {
            const auto& record =
                geometry.features[static_cast<std::size_t>(feature)];
            require(record.point >= 0
                        && record.point < interface.num_points(),
                    "feature point is out of range");
            require(interface.is_corner_point(record.point),
                    "feature is not marked in Interface2D");
            require(interface.corner_index(record.point) == feature,
                    "feature index disagrees with Interface2D");
            require(interface.weights()[record.point] == 0.0,
                    "feature point unexpectedly carries trace weight");
            require(std::find(geometry.active_trace_points.begin(),
                              geometry.active_trace_points.end(),
                              record.point)
                        == geometry.active_trace_points.end(),
                    "feature is an active trace candidate");
            const auto& corner = interface.corner(feature);
            require(corner.prev_panel >= 0 && corner.next_panel >= 0,
                    "feature has no one-sided compatibility panels");
            require_near(corner.tangent_minus.norm(), 1.0, 2.0e-12,
                         "minus feature tangent is not unit");
            require_near(corner.tangent_plus.norm(), 1.0, 2.0e-12,
                         "plus feature tangent is not unit");
            if (record.continuity == BenchmarkGeometryContinuity2D::G1) {
                ++g1;
                require(corner.tangent_minus.dot(corner.tangent_plus)
                            > 1.0 - 1.0e-12,
                        "declared G1 feature has a tangent turn");
            } else if (record.continuity
                       == BenchmarkGeometryContinuity2D::G0) {
                ++g0;
                require(corner.tangent_minus.dot(corner.tangent_plus)
                            < 1.0 - 1.0e-4,
                        "declared G0 feature has matching tangents");
            }
        }
        require(g1 == expected_g1[i], "G1 feature count is wrong");
        require(g0 == expected_g0[i], "G0 feature count is wrong");
    }
}

void test_exact_circle_and_affine_ellipse()
{
    const BenchmarkNurbsGeometry2D circle =
        make_benchmark_nurbs_geometry_for_panel_length_2d(
            BenchmarkNurbsGeometryKind2D::Circle, 0.17);
    const auto& circle_curve = benchmark_nurbs_provider_2d(circle).curve();
    const Eigen::Vector2d circle_center(0.06, -0.04);
    for (int sample = 0; sample <= 160; ++sample) {
        const double u = circle_curve.domain_start()
            + (circle_curve.domain_end() - circle_curve.domain_start())
                * static_cast<double>(sample) / 160.0;
        require_near((circle_curve.evaluate(u) - circle_center).norm(),
                     0.72,
                     4.0e-13,
                     "rational quadratic circle is not exact");
    }
    for (double knot : {0.25, 0.50, 0.75}) {
        require_near(
            one_sided_unit_tangent(circle_curve, knot, 0.0),
            one_sided_unit_tangent(circle_curve, knot, 1.0),
            2.0e-12,
            "circle quarter-arc join is not geometrically G1");
        require_near(
            one_sided_curvature(circle_curve, knot, 0.0),
            one_sided_curvature(circle_curve, knot, 1.0),
            2.0e-10,
            "circle quarter-arc join is not geometrically G2");
    }

    const BenchmarkNurbsGeometry2D ellipse =
        make_benchmark_nurbs_geometry_for_panel_length_2d(
            BenchmarkNurbsGeometryKind2D::Ellipse, 0.17);
    const auto& ellipse_curve = benchmark_nurbs_provider_2d(ellipse).curve();
    const Eigen::Vector2d ellipse_center(0.08, -0.06);
    for (int sample = 0; sample <= 160; ++sample) {
        const double u = ellipse_curve.domain_start()
            + (ellipse_curve.domain_end() - ellipse_curve.domain_start())
                * static_cast<double>(sample) / 160.0;
        Eigen::Vector2d local =
            rotate(ellipse_curve.evaluate(u) - ellipse_center, -0.37);
        require_near(
            local[0] * local[0] / (0.92 * 0.92)
                + local[1] * local[1] / (0.61 * 0.61),
            1.0,
            8.0e-13,
            "affine rational quadratic ellipse is not exact");
    }
    for (double knot : {0.25, 0.50, 0.75}) {
        require_near(
            one_sided_unit_tangent(ellipse_curve, knot, 0.0),
            one_sided_unit_tangent(ellipse_curve, knot, 1.0),
            3.0e-12,
            "ellipse quarter-arc join is not geometrically G1");
        require_near(
            one_sided_curvature(ellipse_curve, knot, 0.0),
            one_sided_curvature(ellipse_curve, knot, 1.0),
            5.0e-10,
            "ellipse quarter-arc join is not geometrically G2");
    }
}

void test_periodic_flower_and_piecewise_features()
{
    const BenchmarkNurbsGeometry2D flower =
        make_benchmark_nurbs_geometry_for_panel_length_2d(
            BenchmarkNurbsGeometryKind2D::Flower, 0.15);
    const auto& flower_curve = benchmark_nurbs_provider_2d(flower).curve();
    const double start = flower_curve.domain_start();
    const double end = flower_curve.domain_end();
    require_near(flower_curve.evaluate(start), flower_curve.evaluate(end),
                 2.0e-13, "periodic flower does not close");
    require_near(flower_curve.derivative(start), flower_curve.derivative(end),
                 2.0e-12, "periodic flower is not C1 at its seam");
    require_near(flower_curve.second_derivative(start),
                 flower_curve.second_derivative(end),
                 2.0e-11,
                 "periodic flower is not C2 at its seam");

    const Eigen::Vector2d center(0.03, -0.02);
    for (double offset : {0.17, 0.83, 1.61, 2.74}) {
        const Eigen::Vector2d base =
            flower_curve.evaluate(start + offset) - center;
        const Eigen::Vector2d next =
            flower_curve.evaluate(start + offset + 4.0) - center;
        require_near(next, rotate(base, 2.0 * kPi / 5.0), 3.0e-13,
                     "periodic NURBS flower lost five-fold symmetry");
    }

    const BenchmarkNurbsGeometry2D heart =
        make_benchmark_nurbs_geometry_for_panel_length_2d(
            BenchmarkNurbsGeometryKind2D::Heart, 0.15);
    const auto& heart_provider = benchmark_nurbs_provider_2d(heart);
    require_near(heart_provider.curve().evaluate(0.0), {0.0, 0.40},
                 2.0e-14, "heart top feature moved");
    require_near(heart_provider.curve().evaluate(2.0), {0.0, -1.12},
                 2.0e-14, "heart bottom feature moved");
    for (double parameter : {1.0, 3.0}) {
        const double left_curvature = one_sided_curvature(
            heart_provider.curve(), parameter, parameter - 1.0);
        const double right_curvature = one_sided_curvature(
            heart_provider.curve(), parameter, parameter + 1.0);
        require(std::abs(left_curvature - right_curvature) > 1.0e-3,
                "heart G1 feature accidentally has matching curvature");
    }

    const BenchmarkNurbsGeometry2D lshape =
        make_benchmark_nurbs_geometry_for_panel_length_2d(
            BenchmarkNurbsGeometryKind2D::LShape, 0.15);
    const auto& lshape_provider = benchmark_nurbs_provider_2d(lshape);
    const std::array<Eigen::Vector2d, 7> expected{{
        {-0.93, -1.04}, {1.07, -1.04}, {1.07, -0.04},
        {0.07, -0.04}, {0.07, 0.96}, {-0.93, 0.96},
        {-0.93, -1.04}
    }};
    const std::array<double, 7> parameters{{0, 2, 3, 4, 5, 6, 8}};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require_near(lshape_provider.curve().evaluate(parameters[i]),
                     expected[i], 2.0e-14,
                     "fixed L-shape NURBS vertex moved");
    }
}

void test_authoritative_geometry_is_grid_independent()
{
    for (BenchmarkNurbsGeometryKind2D kind : kKinds) {
        const BenchmarkNurbsGeometry2D coarse =
            make_benchmark_nurbs_geometry_2d(kind, 0.12, 1.5);
        const BenchmarkNurbsGeometry2D fine =
            make_benchmark_nurbs_geometry_2d(kind, 0.06, 1.5);
        const auto& coarse_curve = benchmark_nurbs_provider_2d(coarse).curve();
        const auto& fine_curve = benchmark_nurbs_provider_2d(fine).curve();

        require(fine.interface.num_panels() > coarse.interface.num_panels(),
                "refining h did not refine compatibility panels");
        require(coarse_curve.basis().degree()
                    == fine_curve.basis().degree(),
                "NURBS degree changed with grid level");
        require(coarse_curve.basis().knots()
                    == fine_curve.basis().knots(),
                "NURBS knots changed with grid level");
        require(coarse_curve.weights() == fine_curve.weights(),
                "NURBS weights changed with grid level");
        require(coarse_curve.control_points().size()
                    == fine_curve.control_points().size(),
                "NURBS control count changed with grid level");
        for (std::size_t control = 0;
             control < coarse_curve.control_points().size();
             ++control) {
            require_near(
                coarse_curve.control_points()[control],
                fine_curve.control_points()[control],
                0.0,
                "NURBS control point changed with grid level");
        }
        for (int sample = 0; sample <= 40; ++sample) {
            const double fraction = static_cast<double>(sample) / 40.0;
            const double coarse_u = coarse_curve.domain_start()
                + fraction
                    * (coarse_curve.domain_end()
                       - coarse_curve.domain_start());
            const double fine_u = fine_curve.domain_start()
                + fraction
                    * (fine_curve.domain_end() - fine_curve.domain_start());
            require_near(coarse_curve.evaluate(coarse_u),
                         fine_curve.evaluate(fine_u),
                         0.0,
                         "authoritative geometry depends on h");
        }
    }
}

} // namespace

int main()
{
    try {
        test_names_and_parse();
        test_provider_and_compatibility_panels();
        test_feature_metadata();
        test_exact_circle_and_affine_ellipse();
        test_periodic_flower_and_piecewise_features();
        test_authoritative_geometry_is_grid_independent();
        std::cout << "2D benchmark NURBS geometry tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "2D benchmark NURBS geometry test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
