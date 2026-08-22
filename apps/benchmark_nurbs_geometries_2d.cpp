#include "benchmark_nurbs_geometries_2d.hpp"

#include "src/geometry/nurbs_basis.hpp"
#include "src/geometry/nurbs_curve_2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace kfbim::app2d {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

using geometry2d::NurbsCurve2D;

struct FeatureSpec {
    double parameter = 0.0;
    BenchmarkGeometryContinuity2D continuity =
        BenchmarkGeometryContinuity2D::G0;
};

struct CurveSpec {
    NurbsCurve2D curve;
    std::vector<double> span_breaks;
    std::vector<FeatureSpec> features;
};

Eigen::Vector2d rotate(Eigen::Vector2d point, double angle)
{
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    return {cosine * point[0] - sine * point[1],
            sine * point[0] + cosine * point[1]};
}

CurveSpec make_exact_circle_or_ellipse(bool ellipse)
{
    constexpr double middle_weight = 0.707106781186547524400844362104849039;
    const std::array<Eigen::Vector2d, 9> unit_control{{
        { 1.0,  0.0},
        { 1.0,  1.0},
        { 0.0,  1.0},
        {-1.0,  1.0},
        {-1.0,  0.0},
        {-1.0, -1.0},
        { 0.0, -1.0},
        { 1.0, -1.0},
        { 1.0,  0.0}
    }};

    const Eigen::Vector2d center = ellipse
        ? Eigen::Vector2d(0.08, -0.06)
        : Eigen::Vector2d(0.06, -0.04);
    const Eigen::Vector2d axes = ellipse
        ? Eigen::Vector2d(0.92, 0.61)
        : Eigen::Vector2d(0.72, 0.72);
    const double angle = ellipse ? 0.37 : 0.0;

    NurbsCurve2D::ControlPointVector control_points;
    control_points.reserve(unit_control.size());
    for (const Eigen::Vector2d& unit : unit_control) {
        const Eigen::Vector2d scaled(
            axes[0] * unit[0], axes[1] * unit[1]);
        control_points.push_back(center + rotate(scaled, angle));
    }

    return {
        NurbsCurve2D(
            geometry::NurbsBasis1D(
                2,
                {0.0, 0.0, 0.0,
                 0.25, 0.25,
                 0.50, 0.50,
                 0.75, 0.75,
                 1.0, 1.0, 1.0}),
            std::move(control_points),
            {1.0, middle_weight, 1.0,
             middle_weight, 1.0, middle_weight,
             1.0, middle_weight, 1.0}),
        {0.0, 1.0},
        {}};
}

CurveSpec make_flower()
{
    // A fixed periodic cubic NURBS is the authoritative flower.  It is not a
    // sampled surrogate used by the solver: all subsequent geometry queries
    // evaluate this same NURBS object.  Twenty base controls give four
    // controls per petal; the first three are repeated to close a cardinal
    // cubic basis with matching value, first derivative, and second derivative.
    constexpr int base_count = 20;
    constexpr int degree = 3;
    constexpr double base_radius = 0.72;
    constexpr double modulation = 0.24;
    constexpr double phase = 0.11;
    const Eigen::Vector2d center(0.03, -0.02);

    std::array<Eigen::Vector2d, base_count> base_controls;
    for (int i = 0; i < base_count; ++i) {
        const double theta = phase
            + 2.0 * kPi * static_cast<double>(i)
                / static_cast<double>(base_count);
        const double radius = base_radius
            * (1.0 + modulation
                * std::cos(5.0 * (theta - phase)));
        base_controls[static_cast<std::size_t>(i)] =
            center + radius * Eigen::Vector2d(
                std::cos(theta), std::sin(theta));
    }

    NurbsCurve2D::ControlPointVector control_points;
    control_points.reserve(base_count + degree);
    for (const Eigen::Vector2d& point : base_controls)
        control_points.push_back(point);
    for (int i = 0; i < degree; ++i)
        control_points.push_back(base_controls[static_cast<std::size_t>(i)]);

    std::vector<double> knots;
    knots.reserve(control_points.size() + degree + 1);
    for (int i = 0;
         i < static_cast<int>(control_points.size()) + degree + 1;
         ++i) {
        knots.push_back(static_cast<double>(i));
    }
    const double domain_start = static_cast<double>(degree);
    const double domain_end = domain_start + static_cast<double>(base_count);

    return {
        NurbsCurve2D(
            geometry::NurbsBasis1D(degree, std::move(knots)),
            std::move(control_points),
            std::vector<double>(base_count + degree, 1.0)),
        {domain_start, domain_end},
        {}};
}

CurveSpec make_heart()
{
    // Four exact polynomial cubic Bezier pieces represented as one NURBS.
    // The top notch and bottom tip are G0.  The left/right joins have aligned
    // one-sided tangents but different curvature and are therefore G1.
    NurbsCurve2D::ControlPointVector control_points{
        { 0.00,  0.40},
        {-0.38,  0.93},
        {-1.00,  0.75},
        {-1.00,  0.18},

        {-1.00, -0.38},
        {-0.38, -0.84},
        { 0.00, -1.12},

        { 0.38, -0.84},
        { 1.00, -0.38},
        { 1.00,  0.18},

        { 1.00,  0.75},
        { 0.38,  0.93},
        { 0.00,  0.40}
    };
    return {
        NurbsCurve2D(
            geometry::NurbsBasis1D(
                3,
                {0.0, 0.0, 0.0, 0.0,
                 1.0, 1.0, 1.0,
                 2.0, 2.0, 2.0,
                 3.0, 3.0, 3.0,
                 4.0, 4.0, 4.0, 4.0}),
            std::move(control_points),
            std::vector<double>(13, 1.0)),
        {0.0, 1.0, 2.0, 3.0, 4.0},
        {{0.0, BenchmarkGeometryContinuity2D::G0},
         {1.0, BenchmarkGeometryContinuity2D::G1},
         {2.0, BenchmarkGeometryContinuity2D::G0},
         {3.0, BenchmarkGeometryContinuity2D::G1}}};
}

CurveSpec make_l_shape()
{
    NurbsCurve2D::ControlPointVector control_points{
        {-0.93, -1.04},
        { 1.07, -1.04},
        { 1.07, -0.04},
        { 0.07, -0.04},
        { 0.07,  0.96},
        {-0.93,  0.96},
        {-0.93, -1.04}
    };
    return {
        NurbsCurve2D(
            geometry::NurbsBasis1D(
                1,
                {0.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 8.0}),
            std::move(control_points),
            std::vector<double>(7, 1.0)),
        {0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0},
        {{0.0, BenchmarkGeometryContinuity2D::G0},
         {2.0, BenchmarkGeometryContinuity2D::G0},
         {3.0, BenchmarkGeometryContinuity2D::G0},
         {4.0, BenchmarkGeometryContinuity2D::G0},
         {5.0, BenchmarkGeometryContinuity2D::G0},
         {6.0, BenchmarkGeometryContinuity2D::G0}}};
}

CurveSpec make_curve_spec(BenchmarkNurbsGeometryKind2D kind)
{
    switch (kind) {
    case BenchmarkNurbsGeometryKind2D::Circle:
        return make_exact_circle_or_ellipse(false);
    case BenchmarkNurbsGeometryKind2D::Ellipse:
        return make_exact_circle_or_ellipse(true);
    case BenchmarkNurbsGeometryKind2D::Flower:
        return make_flower();
    case BenchmarkNurbsGeometryKind2D::Heart:
        return make_heart();
    case BenchmarkNurbsGeometryKind2D::LShape:
        return make_l_shape();
    }
    throw std::invalid_argument("unsupported benchmark NURBS geometry");
}

double gauss_speed_integral(const NurbsCurve2D& curve,
                            double lower,
                            double upper)
{
    // Eight-point Gauss-Legendre rule.  Integration is split at every NURBS
    // knot by arc_length(), so this rule never crosses a derivative break.
    constexpr std::array<double, 4> nodes{{
        0.18343464249564980494,
        0.52553240991632898582,
        0.79666647741362673959,
        0.96028985649753623168
    }};
    constexpr std::array<double, 4> weights{{
        0.36268378337836198297,
        0.31370664587788728734,
        0.22238103445337447054,
        0.10122853629037625915
    }};
    if (!(upper > lower))
        return 0.0;
    const double midpoint = 0.5 * (lower + upper);
    const double half = 0.5 * (upper - lower);
    double result = 0.0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const double offset = half * nodes[i];
        result += weights[i]
            * (curve.derivative(midpoint - offset).norm()
               + curve.derivative(midpoint + offset).norm());
    }
    return half * result;
}

double arc_length(const NurbsCurve2D& curve, double lower, double upper)
{
    if (!(upper > lower))
        return 0.0;
    std::vector<double> subdivisions;
    subdivisions.push_back(lower);
    for (double knot : curve.basis().knots()) {
        if (knot > lower && knot < upper
            && (subdivisions.empty()
                || knot > subdivisions.back())) {
            subdivisions.push_back(knot);
        }
    }
    subdivisions.push_back(upper);
    double result = 0.0;
    for (std::size_t i = 1; i < subdivisions.size(); ++i) {
        result += gauss_speed_integral(
            curve, subdivisions[i - 1], subdivisions[i]);
    }
    return result;
}

double parameter_at_arc_distance(const NurbsCurve2D& curve,
                                 double lower,
                                 double upper,
                                 double target_distance)
{
    const double full_length = arc_length(curve, lower, upper);
    if (!(target_distance > 0.0))
        return lower;
    if (target_distance >= full_length)
        return upper;
    double left = lower;
    double right = upper;
    for (int iteration = 0; iteration < 60; ++iteration) {
        const double middle = 0.5 * (left + right);
        if (arc_length(curve, lower, middle) < target_distance)
            left = middle;
        else
            right = middle;
    }
    return 0.5 * (left + right);
}

std::vector<double> equal_arc_panel_parameters(
    const NurbsCurve2D& curve,
    double lower,
    double upper,
    int panel_count)
{
    std::vector<double> result(
        static_cast<std::size_t>(panel_count + 1), lower);
    result.back() = upper;
    double current = lower;
    double remaining_length = arc_length(curve, lower, upper);
    for (int panel = 0; panel + 1 < panel_count; ++panel) {
        const int remaining_panels = panel_count - panel;
        const double step = remaining_length
            / static_cast<double>(remaining_panels);
        const double next = parameter_at_arc_distance(
            curve, current, upper, step);
        result[static_cast<std::size_t>(panel + 1)] = next;
        current = next;
        remaining_length = arc_length(curve, current, upper);
    }
    return result;
}

double canonical_parameter(const CurveSpec& spec, double parameter)
{
    const double start = spec.span_breaks.front();
    const double end = spec.span_breaks.back();
    const double tolerance = 1.0e-12 * std::max(1.0, end - start);
    return std::abs(parameter - end) <= tolerance ? start : parameter;
}

int feature_index(const CurveSpec& spec, double parameter)
{
    parameter = canonical_parameter(spec, parameter);
    const double scale = std::max(
        1.0, spec.span_breaks.back() - spec.span_breaks.front());
    for (int feature = 0;
         feature < static_cast<int>(spec.features.size());
         ++feature) {
        if (std::abs(parameter
                     - spec.features[static_cast<std::size_t>(feature)]
                           .parameter)
            <= 1.0e-12 * scale) {
            return feature;
        }
    }
    return -1;
}

std::pair<int, int> feature_spans(const CurveSpec& spec,
                                  double parameter)
{
    parameter = canonical_parameter(spec, parameter);
    const int span_count =
        static_cast<int>(spec.span_breaks.size()) - 1;
    const double scale = std::max(
        1.0, spec.span_breaks.back() - spec.span_breaks.front());
    if (std::abs(parameter - spec.span_breaks.front())
        <= 1.0e-12 * scale) {
        return {span_count - 1, 0};
    }
    for (int span = 1; span < span_count; ++span) {
        if (std::abs(parameter
                     - spec.span_breaks[static_cast<std::size_t>(span)])
            <= 1.0e-12 * scale) {
            return {span - 1, span};
        }
    }
    throw std::logic_error(
        "benchmark feature is not a smooth-branch boundary");
}

double one_sided_parameter(const CurveSpec& spec,
                           int span,
                           double parameter)
{
    const double lower = spec.span_breaks[static_cast<std::size_t>(span)];
    const double upper = spec.span_breaks[static_cast<std::size_t>(span + 1)];
    parameter = std::clamp(parameter, lower, upper);
    if (parameter <= lower)
        return std::nextafter(lower, upper);
    if (parameter >= upper)
        return std::nextafter(upper, lower);
    return parameter;
}

Eigen::Vector2d unit_tangent(const CurveSpec& spec,
                             int span,
                             double parameter)
{
    Eigen::Vector2d tangent = spec.curve.derivative(
        one_sided_parameter(spec, span, parameter));
    const double length = tangent.norm();
    if (!(length > 1.0e-14) || !tangent.allFinite())
        throw std::runtime_error(
            "benchmark NURBS has degenerate one-sided tangent");
    return tangent / length;
}

Eigen::Vector2d outward_normal(Eigen::Vector2d tangent)
{
    return {tangent[1], -tangent[0]};
}

Eigen::Vector2d normalized_or(Eigen::Vector2d value,
                              Eigen::Vector2d fallback)
{
    const double length = value.norm();
    return length > 1.0e-14 ? value / length : fallback;
}

BenchmarkNurbsGeometry2D build_geometry(
    BenchmarkNurbsGeometryKind2D kind,
    CurveSpec spec,
    double target_panel_length)
{
    if (!(target_panel_length > 0.0)
        || !std::isfinite(target_panel_length)) {
        throw std::invalid_argument(
            "benchmark target panel length must be finite and positive");
    }
    const int span_count =
        static_cast<int>(spec.span_breaks.size()) - 1;
    if (span_count <= 0)
        throw std::logic_error("benchmark curve has no smooth spans");

    std::vector<std::vector<double>> panel_breaks(
        static_cast<std::size_t>(span_count));
    for (int span = 0; span < span_count; ++span) {
        const double lower =
            spec.span_breaks[static_cast<std::size_t>(span)];
        const double upper =
            spec.span_breaks[static_cast<std::size_t>(span + 1)];
        const double length = arc_length(spec.curve, lower, upper);
        const int minimum_panels = spec.features.empty() ? 3 : 2;
        const int count = std::max(
            minimum_panels,
            static_cast<int>(std::ceil(length / target_panel_length)));
        panel_breaks[static_cast<std::size_t>(span)] =
            equal_arc_panel_parameters(
                spec.curve, lower, upper, count);
    }

    std::vector<Eigen::Vector2d> points;
    std::vector<Eigen::Vector2d> normals;
    std::vector<double> weights;
    std::vector<int> point_spans;
    std::vector<double> point_parameters;
    std::vector<int> active_points;
    std::vector<int> feature_points(spec.features.size(), -1);
    std::vector<std::array<int, 3>> panel_rows;
    std::vector<int> panel_spans;
    std::vector<std::array<double, 2>> panel_parameters;
    std::vector<int> first_panel_by_span(
        static_cast<std::size_t>(span_count), -1);
    std::vector<int> last_panel_by_span(
        static_cast<std::size_t>(span_count), -1);
    double maximum_panel_length = 0.0;

    const auto append_point = [&](double parameter,
                                  int span,
                                  double weight) {
        const int point = static_cast<int>(points.size());
        points.push_back(spec.curve.evaluate(parameter));
        normals.push_back(outward_normal(
            unit_tangent(spec, span, parameter)));
        weights.push_back(weight);
        point_spans.push_back(span);
        point_parameters.push_back(parameter);
        const int feature = feature_index(spec, parameter);
        if (feature >= 0)
            feature_points[static_cast<std::size_t>(feature)] = point;
        return point;
    };

    int first_endpoint = append_point(
        spec.span_breaks.front(), 0, 0.0);
    int current_endpoint = first_endpoint;
    for (int span = 0; span < span_count; ++span) {
        const std::vector<double>& breaks =
            panel_breaks[static_cast<std::size_t>(span)];
        for (int local_panel = 0;
             local_panel + 1 < static_cast<int>(breaks.size());
             ++local_panel) {
            const double lower =
                breaks[static_cast<std::size_t>(local_panel)];
            const double upper =
                breaks[static_cast<std::size_t>(local_panel + 1)];
            const double middle = 0.5 * (lower + upper);
            const double panel_length = arc_length(spec.curve, lower, upper);
            maximum_panel_length =
                std::max(maximum_panel_length, panel_length);

            const int middle_point = append_point(
                middle, span, panel_length);
            active_points.push_back(middle_point);

            const bool final_panel = span + 1 == span_count
                                  && local_panel + 2
                                         == static_cast<int>(breaks.size());
            const int end_point = final_panel
                ? first_endpoint
                : append_point(upper, span, 0.0);
            const int panel = static_cast<int>(panel_rows.size());
            panel_rows.push_back(
                {current_endpoint, middle_point, end_point});
            panel_spans.push_back(span);
            panel_parameters.push_back({lower, upper});
            if (first_panel_by_span[static_cast<std::size_t>(span)] < 0)
                first_panel_by_span[static_cast<std::size_t>(span)] = panel;
            last_panel_by_span[static_cast<std::size_t>(span)] = panel;
            current_endpoint = end_point;
        }
    }

    for (int feature = 0;
         feature < static_cast<int>(spec.features.size());
         ++feature) {
        const int point = feature_points[static_cast<std::size_t>(feature)];
        if (point < 0)
            throw std::logic_error("benchmark feature has no interface point");
        const auto sides = feature_spans(
            spec,
            spec.features[static_cast<std::size_t>(feature)].parameter);
        point_spans[static_cast<std::size_t>(point)] = sides.second;
        point_parameters[static_cast<std::size_t>(point)] =
            spec.features[static_cast<std::size_t>(feature)].parameter;
        const Eigen::Vector2d tangent_minus = unit_tangent(
            spec, sides.first,
            spec.features[static_cast<std::size_t>(feature)].parameter);
        const Eigen::Vector2d tangent_plus = unit_tangent(
            spec, sides.second,
            spec.features[static_cast<std::size_t>(feature)].parameter);
        const Eigen::Vector2d normal_plus = outward_normal(tangent_plus);
        normals[static_cast<std::size_t>(point)] = normalized_or(
            outward_normal(tangent_minus) + normal_plus,
            normal_plus);
        weights[static_cast<std::size_t>(point)] = 0.0;
    }

    Eigen::MatrixX2d point_matrix(points.size(), 2);
    Eigen::MatrixX2d normal_matrix(normals.size(), 2);
    Eigen::VectorXd weight_vector(weights.size());
    for (int point = 0; point < static_cast<int>(points.size()); ++point) {
        point_matrix.row(point) =
            points[static_cast<std::size_t>(point)].transpose();
        normal_matrix.row(point) =
            normals[static_cast<std::size_t>(point)].transpose();
        weight_vector[point] = weights[static_cast<std::size_t>(point)];
    }

    Eigen::MatrixXi connectivity(panel_rows.size(), 3);
    Eigen::VectorXi components =
        Eigen::VectorXi::Zero(static_cast<int>(panel_rows.size()));
    PanelSideGeometry2D panel_sides;
    panel_sides.point_normals.resize(
        3 * static_cast<int>(panel_rows.size()), 2);
    panel_sides.point_tangents.resize(
        3 * static_cast<int>(panel_rows.size()), 2);
    for (int panel = 0;
         panel < static_cast<int>(panel_rows.size());
         ++panel) {
        const auto row = panel_rows[static_cast<std::size_t>(panel)];
        const auto parameters =
            panel_parameters[static_cast<std::size_t>(panel)];
        const int span = panel_spans[static_cast<std::size_t>(panel)];
        const std::array<double, 3> local_parameters{{
            parameters[0],
            0.5 * (parameters[0] + parameters[1]),
            parameters[1]
        }};
        for (int local = 0; local < 3; ++local) {
            connectivity(panel, local) =
                row[static_cast<std::size_t>(local)];
            const Eigen::Vector2d tangent = unit_tangent(
                spec, span,
                local_parameters[static_cast<std::size_t>(local)]);
            panel_sides.point_tangents.row(3 * panel + local) =
                tangent.transpose();
            panel_sides.point_normals.row(3 * panel + local) =
                outward_normal(tangent).transpose();
        }
    }

    std::vector<InterfacePointKind2D> point_kind(
        points.size(), InterfacePointKind2D::Smooth);
    std::vector<int> corner_index_by_point(points.size(), -1);
    std::vector<CornerData2D> corners;
    std::vector<BenchmarkNurbsFeature2D> features;
    corners.reserve(spec.features.size());
    features.reserve(spec.features.size());
    for (int feature = 0;
         feature < static_cast<int>(spec.features.size());
         ++feature) {
        const FeatureSpec& source =
            spec.features[static_cast<std::size_t>(feature)];
        const auto sides = feature_spans(spec, source.parameter);
        const int point = feature_points[static_cast<std::size_t>(feature)];
        point_kind[static_cast<std::size_t>(point)] =
            InterfacePointKind2D::Corner;
        corner_index_by_point[static_cast<std::size_t>(point)] = feature;

        CornerData2D corner;
        corner.point = point;
        corner.prev_panel =
            last_panel_by_span[static_cast<std::size_t>(sides.first)];
        corner.next_panel =
            first_panel_by_span[static_cast<std::size_t>(sides.second)];
        corner.tangent_minus = unit_tangent(
            spec, sides.first, source.parameter);
        corner.tangent_plus = unit_tangent(
            spec, sides.second, source.parameter);
        corner.normal_minus = outward_normal(corner.tangent_minus);
        corner.normal_plus = outward_normal(corner.tangent_plus);
        corner.turn_angle = std::atan2(
            corner.tangent_minus[0] * corner.tangent_plus[1]
                - corner.tangent_minus[1] * corner.tangent_plus[0],
            corner.tangent_minus.dot(corner.tangent_plus));
        corners.push_back(corner);
        features.push_back(
            {point,
             sides.first,
             sides.second,
             source.parameter,
             source.continuity});
    }

    std::shared_ptr<const IPanelGeometry2D> provider =
        std::make_shared<geometry2d::NurbsBoundaryPanelGeometry2D>(
            std::move(spec.curve),
            std::move(spec.span_breaks),
            std::move(panel_spans),
            std::move(panel_parameters),
            std::move(point_spans),
            std::move(point_parameters),
            true,
            true);

    Interface2D interface(
        std::move(point_matrix),
        std::move(normal_matrix),
        std::move(weight_vector),
        3,
        std::move(connectivity),
        std::move(components),
        std::move(panel_sides),
        std::move(point_kind),
        std::move(corner_index_by_point),
        std::move(corners),
        PanelNodeLayout2D::QuadraticLagrange,
        {},
        std::move(provider));

    return {kind,
            benchmark_nurbs_geometry_name_2d(kind),
            std::move(interface),
            std::move(active_points),
            std::move(features),
            target_panel_length,
            maximum_panel_length};
}

} // namespace

const char* benchmark_nurbs_geometry_name_2d(
    BenchmarkNurbsGeometryKind2D kind) noexcept
{
    switch (kind) {
    case BenchmarkNurbsGeometryKind2D::Circle:
        return "circle";
    case BenchmarkNurbsGeometryKind2D::Ellipse:
        return "ellipse";
    case BenchmarkNurbsGeometryKind2D::Flower:
        return "flower";
    case BenchmarkNurbsGeometryKind2D::Heart:
        return "heart";
    case BenchmarkNurbsGeometryKind2D::LShape:
        return "lshape";
    }
    return "unknown";
}

BenchmarkNurbsGeometryKind2D parse_benchmark_nurbs_geometry_2d(
    const std::string& name)
{
    if (name == "circle")
        return BenchmarkNurbsGeometryKind2D::Circle;
    if (name == "ellipse")
        return BenchmarkNurbsGeometryKind2D::Ellipse;
    if (name == "flower")
        return BenchmarkNurbsGeometryKind2D::Flower;
    if (name == "heart")
        return BenchmarkNurbsGeometryKind2D::Heart;
    if (name == "lshape" || name == "l_shape")
        return BenchmarkNurbsGeometryKind2D::LShape;
    throw std::invalid_argument(
        "geometry must be circle, ellipse, flower, heart, or lshape");
}

BenchmarkNurbsGeometry2D make_benchmark_nurbs_geometry_2d(
    BenchmarkNurbsGeometryKind2D kind,
    double grid_spacing,
    double panel_length_over_h)
{
    if (!(grid_spacing > 0.0) || !std::isfinite(grid_spacing))
        throw std::invalid_argument(
            "benchmark grid spacing must be finite and positive");
    if (!(panel_length_over_h > 0.0)
        || !std::isfinite(panel_length_over_h)) {
        throw std::invalid_argument(
            "benchmark panel_length_over_h must be finite and positive");
    }
    return make_benchmark_nurbs_geometry_for_panel_length_2d(
        kind, grid_spacing * panel_length_over_h);
}

BenchmarkNurbsGeometry2D make_benchmark_nurbs_geometry_for_panel_length_2d(
    BenchmarkNurbsGeometryKind2D kind,
    double target_panel_length)
{
    return build_geometry(
        kind, make_curve_spec(kind), target_panel_length);
}

const geometry2d::NurbsBoundaryPanelGeometry2D&
benchmark_nurbs_provider_2d(const BenchmarkNurbsGeometry2D& geometry)
{
    if (!geometry.interface.has_panel_geometry())
        throw std::logic_error("benchmark interface has no exact provider");
    const auto* provider = dynamic_cast<
        const geometry2d::NurbsBoundaryPanelGeometry2D*>(
            &geometry.interface.panel_geometry());
    if (provider == nullptr)
        throw std::logic_error("benchmark provider is not a NURBS boundary");
    return *provider;
}

} // namespace kfbim::app2d
