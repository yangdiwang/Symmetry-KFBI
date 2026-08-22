#include "nurbs_boundary_2d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kfbim::geometry2d {
namespace {

void require_index(int index, int size, const char* what)
{
    if (index < 0 || index >= size)
        throw std::out_of_range(what);
}

} // namespace

NurbsBoundaryPanelGeometry2D::NurbsBoundaryPanelGeometry2D(
    NurbsCurve2D curve,
    std::vector<double> span_breaks,
    std::vector<int> panel_spans,
    std::vector<std::array<double, 2>> panel_parameters,
    std::vector<int> point_spans,
    std::vector<double> point_parameters,
    bool outward_normal_is_right,
    bool closed)
    : curve_(std::move(curve))
    , span_breaks_(std::move(span_breaks))
    , panel_spans_(std::move(panel_spans))
    , panel_parameters_(std::move(panel_parameters))
    , point_spans_(std::move(point_spans))
    , point_parameters_(std::move(point_parameters))
    , outward_normal_is_right_(outward_normal_is_right)
    , closed_(closed)
{
    validate();
}

void NurbsBoundaryPanelGeometry2D::validate() const
{
    if (span_breaks_.size() < 2)
        throw std::invalid_argument("NURBS boundary needs at least one span");
    if (panel_spans_.size() != panel_parameters_.size())
        throw std::invalid_argument(
            "NURBS boundary panel span/parameter sizes differ");
    if (point_spans_.size() != point_parameters_.size())
        throw std::invalid_argument(
            "NURBS boundary point span/parameter sizes differ");
    const double scale = std::max(
        1.0, std::abs(span_breaks_.back() - span_breaks_.front()));
    const double tolerance = 1.0e-12 * scale;
    if (std::abs(span_breaks_.front() - curve_.domain_start()) > tolerance
        || std::abs(span_breaks_.back() - curve_.domain_end()) > tolerance) {
        throw std::invalid_argument(
            "NURBS boundary span breaks do not cover the curve domain");
    }
    for (std::size_t i = 1; i < span_breaks_.size(); ++i) {
        if (!std::isfinite(span_breaks_[i - 1])
            || !(span_breaks_[i] > span_breaks_[i - 1])) {
            throw std::invalid_argument(
                "NURBS boundary span breaks must be finite and increasing");
        }
    }
    for (std::size_t panel = 0; panel < panel_spans_.size(); ++panel) {
        const int span = panel_spans_[panel];
        require_index(span, num_spans(),
                      "NURBS boundary panel span is out of range");
        const auto interval = span_interval(span);
        const auto parameters = panel_parameters_[panel];
        if (!std::isfinite(parameters[0]) || !std::isfinite(parameters[1])
            || parameters[0] < interval.first - tolerance
            || parameters[1] > interval.second + tolerance
            || !(parameters[1] > parameters[0])) {
            throw std::invalid_argument(
                "NURBS boundary panel parameter interval is invalid");
        }
    }
    for (std::size_t point = 0; point < point_spans_.size(); ++point) {
        const int span = point_spans_[point];
        require_index(span, num_spans(),
                      "NURBS boundary point span is out of range");
        const auto interval = span_interval(span);
        const double parameter = point_parameters_[point];
        if (!std::isfinite(parameter)
            || parameter < interval.first - tolerance
            || parameter > interval.second + tolerance) {
            throw std::invalid_argument(
                "NURBS boundary point parameter is outside its span");
        }
    }
}

int NurbsBoundaryPanelGeometry2D::num_panels() const
{
    return static_cast<int>(panel_spans_.size());
}

std::pair<double, double>
NurbsBoundaryPanelGeometry2D::span_interval(int span) const
{
    require_index(span, num_spans(),
                  "NURBS boundary span is out of range");
    return {span_breaks_[static_cast<std::size_t>(span)],
            span_breaks_[static_cast<std::size_t>(span + 1)]};
}

int NurbsBoundaryPanelGeometry2D::panel_span(int panel) const
{
    require_index(panel, num_panels(),
                  "NURBS boundary panel is out of range");
    return panel_spans_[static_cast<std::size_t>(panel)];
}

double NurbsBoundaryPanelGeometry2D::panel_parameter(
    int panel,
    double local_s) const
{
    require_index(panel, num_panels(),
                  "NURBS boundary panel is out of range");
    if (!std::isfinite(local_s))
        throw std::invalid_argument(
            "NURBS boundary panel parameter must be finite");
    local_s = std::clamp(local_s, -1.0, 1.0);
    const auto interval = panel_parameters_[static_cast<std::size_t>(panel)];
    return 0.5 * ((1.0 - local_s) * interval[0]
                + (1.0 + local_s) * interval[1]);
}

int NurbsBoundaryPanelGeometry2D::point_span(int point) const
{
    require_index(point, num_parameterized_points(),
                  "NURBS boundary point is out of range");
    return point_spans_[static_cast<std::size_t>(point)];
}

double NurbsBoundaryPanelGeometry2D::point_parameter(int point) const
{
    require_index(point, num_parameterized_points(),
                  "NURBS boundary point is out of range");
    return point_parameters_[static_cast<std::size_t>(point)];
}

double NurbsBoundaryPanelGeometry2D::one_sided_parameter(
    int span,
    double parameter) const
{
    const auto interval = span_interval(span);
    parameter = std::clamp(parameter, interval.first, interval.second);
    if (parameter <= interval.first)
        return std::nextafter(interval.first, interval.second);
    if (parameter >= interval.second)
        return std::nextafter(interval.second, interval.first);
    return parameter;
}

NurbsBoundaryGeometry2D NurbsBoundaryPanelGeometry2D::evaluate_on_span(
    int span,
    double parameter) const
{
    const auto interval = span_interval(span);
    if (!std::isfinite(parameter))
        throw std::invalid_argument(
            "NURBS boundary evaluation parameter must be finite");
    parameter = std::clamp(parameter, interval.first, interval.second);
    const double differential_parameter =
        one_sided_parameter(span, parameter);

    NurbsBoundaryGeometry2D result;
    result.span = span;
    result.parameter = parameter;
    result.point = curve_.evaluate(parameter);
    result.parameter_tangent = curve_.derivative(differential_parameter);
    result.parameter_second =
        curve_.second_derivative(differential_parameter);
    result.speed = result.parameter_tangent.norm();
    if (!(result.speed > 1.0e-14) || !result.point.allFinite()
        || !result.parameter_tangent.allFinite()
        || !result.parameter_second.allFinite()) {
        throw std::runtime_error(
            "NURBS boundary has degenerate differential geometry");
    }
    result.tangent = result.parameter_tangent / result.speed;
    result.normal = outward_normal_is_right_
        ? Eigen::Vector2d(result.tangent[1], -result.tangent[0])
        : Eigen::Vector2d(-result.tangent[1], result.tangent[0]);
    result.curvature =
        -result.parameter_second.dot(result.normal)
        / (result.speed * result.speed);
    return result;
}

Eigen::Vector2d NurbsBoundaryPanelGeometry2D::point(
    int panel,
    double local_s) const
{
    return curve_.evaluate(panel_parameter(panel, local_s));
}

Eigen::Vector2d NurbsBoundaryPanelGeometry2D::tangent(
    int panel,
    double local_s) const
{
    require_index(panel, num_panels(),
                  "NURBS boundary panel is out of range");
    const auto parameters = panel_parameters_[static_cast<std::size_t>(panel)];
    const double dt_ds = 0.5 * (parameters[1] - parameters[0]);
    return dt_ds
         * evaluate_on_span(panel_span(panel),
                            panel_parameter(panel, local_s))
               .parameter_tangent;
}

Eigen::Vector2d NurbsBoundaryPanelGeometry2D::second_derivative(
    int panel,
    double local_s) const
{
    require_index(panel, num_panels(),
                  "NURBS boundary panel is out of range");
    const auto parameters = panel_parameters_[static_cast<std::size_t>(panel)];
    const double dt_ds = 0.5 * (parameters[1] - parameters[0]);
    return dt_ds * dt_ds
         * evaluate_on_span(panel_span(panel),
                            panel_parameter(panel, local_s))
               .parameter_second;
}

Eigen::Vector2d NurbsBoundaryPanelGeometry2D::normal(
    int panel,
    double local_s) const
{
    return evaluate_on_span(panel_span(panel),
                            panel_parameter(panel, local_s))
        .normal;
}

NurbsSpanProjection2D NurbsBoundaryPanelGeometry2D::project_to_span(
    int span,
    const Eigen::Vector2d& query) const
{
    if (!query.allFinite())
        throw std::invalid_argument(
            "NURBS boundary projection query must be finite");
    auto interval = span_interval(span);
    if (curve_.basis().degree() == 1) {
        const Eigen::Vector2d start = curve_.evaluate(interval.first);
        const Eigen::Vector2d end = curve_.evaluate(interval.second);
        const Eigen::Vector2d edge = end - start;
        const double denominator = edge.squaredNorm();
        if (!(denominator > 1.0e-28))
            throw std::runtime_error("linear NURBS span is degenerate");
        const double fraction = std::clamp(
            (query - start).dot(edge) / denominator, 0.0, 1.0);
        NurbsSpanProjection2D result;
        result.span = span;
        result.parameter = interval.first
                         + fraction * (interval.second - interval.first);
        result.point = curve_.evaluate(result.parameter);
        result.distance = (result.point - query).norm();
        result.converged = true;
        return result;
    }
    constexpr double inverse_golden_ratio = 0.6180339887498948482;
    double a = interval.first;
    double b = interval.second;
    double c = b - inverse_golden_ratio * (b - a);
    double d = a + inverse_golden_ratio * (b - a);
    auto distance_squared = [&](double parameter) {
        return (curve_.evaluate(parameter) - query).squaredNorm();
    };
    double fc = distance_squared(c);
    double fd = distance_squared(d);
    const double tolerance =
        64.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(b - a));
    int iteration = 0;
    for (; iteration < 128 && b - a > tolerance; ++iteration) {
        if (fc <= fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - inverse_golden_ratio * (b - a);
            fc = distance_squared(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + inverse_golden_ratio * (b - a);
            fd = distance_squared(d);
        }
    }
    double parameter = 0.5 * (a + b);
    // Explicit endpoint comparison prevents a corner-gap hit at a true knot
    // from being displaced into the span by the scalar minimizer.
    const double interior_distance = distance_squared(parameter);
    const double left_distance = distance_squared(interval.first);
    const double right_distance = distance_squared(interval.second);
    if (left_distance <= interior_distance && left_distance <= right_distance)
        parameter = interval.first;
    else if (right_distance <= interior_distance)
        parameter = interval.second;

    NurbsSpanProjection2D result;
    result.span = span;
    result.parameter = parameter;
    result.point = curve_.evaluate(parameter);
    result.distance = (result.point - query).norm();
    result.converged = iteration < 128;
    return result;
}

} // namespace kfbim::geometry2d
