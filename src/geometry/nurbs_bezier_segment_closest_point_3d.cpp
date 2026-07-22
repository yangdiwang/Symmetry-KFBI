#include "nurbs_bezier_segment_closest_point_3d.hpp"

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace kfbim::geometry3d {

namespace {

struct Evaluation {
    Eigen::Vector2d parameter = Eigen::Vector2d::Zero();
    Eigen::Vector3d surface_point = Eigen::Vector3d::Zero();
    Eigen::Vector3d segment_point = Eigen::Vector3d::Zero();
    Eigen::Vector3d residual = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, 2> jacobian =
        Eigen::Matrix<double, 3, 2>::Zero();
    Eigen::Vector2d gradient = Eigen::Vector2d::Zero();
    Eigen::Vector2d projected_gradient = Eigen::Vector2d::Zero();
    double t = 0.0;
    double objective = std::numeric_limits<double>::infinity();
};

double clamp_unit(double value)
{
    return std::max(0.0, std::min(1.0, value));
}

Eigen::Vector2d clamp_unit(const Eigen::Vector2d& value)
{
    return {clamp_unit(value.x()), clamp_unit(value.y())};
}

Eigen::Vector2d native_parameter(const RationalBezierElement3D& element,
                                 const Eigen::Vector2d& unit_parameter)
{
    return {
        element.u0() + unit_parameter.x() * (element.u1() - element.u0()),
        element.v0() + unit_parameter.y() * (element.v1() - element.v0())};
}

Eigen::Vector2d unit_parameter(const RationalBezierElement3D& element,
                               const Eigen::Vector2d& native)
{
    return clamp_unit({
        (native.x() - element.u0()) / (element.u1() - element.u0()),
        (native.y() - element.v0()) / (element.v1() - element.v0())});
}

bool active_bound(double gradient,
                  double parameter,
                  double tolerance)
{
    return (parameter <= tolerance && gradient >= 0.0)
        || (parameter >= 1.0 - tolerance && gradient <= 0.0);
}

Eigen::Vector2d projected_gradient(const Eigen::Vector2d& gradient,
                                   const Eigen::Vector2d& parameter,
                                   double tolerance)
{
    Eigen::Vector2d result = gradient;
    for (int coordinate = 0; coordinate < 2; ++coordinate) {
        if (active_bound(
                gradient[coordinate], parameter[coordinate], tolerance)) {
            result[coordinate] = 0.0;
        }
    }
    return result;
}

Evaluation evaluate(const RationalBezierElement3D& element,
                    const NurbsSurfacePatch3D& patch,
                    const Eigen::Vector3d& segment_start,
                    const Eigen::Vector3d& segment_delta,
                    double segment_length_squared,
                    const Eigen::Vector2d& unit,
                    double parameter_tolerance)
{
    Evaluation result;
    result.parameter = clamp_unit(unit);
    const Eigen::Vector2d native = native_parameter(element, result.parameter);
    const NurbsSurfaceDerivatives3D derivatives =
        patch.evaluate_with_derivatives(native.x(), native.y());
    if (!derivatives.point.allFinite() || !derivatives.du.allFinite()
        || !derivatives.dv.allFinite()) {
        return result;
    }

    const double raw_t =
        (derivatives.point - segment_start).dot(segment_delta)
        / segment_length_squared;
    result.t = clamp_unit(raw_t);
    result.surface_point = derivatives.point;
    result.segment_point = segment_start + result.t * segment_delta;
    result.residual = result.surface_point - result.segment_point;
    result.objective = 0.5 * result.residual.squaredNorm();

    const double span_u = element.u1() - element.u0();
    const double span_v = element.v1() - element.v0();
    Eigen::Vector3d tangent_u = span_u * derivatives.du;
    Eigen::Vector3d tangent_v = span_v * derivatives.dv;
    const double t_tolerance = 32.0 * std::numeric_limits<double>::epsilon();
    if (raw_t > t_tolerance && raw_t < 1.0 - t_tolerance) {
        const Eigen::Vector3d direction =
            segment_delta / std::sqrt(segment_length_squared);
        tangent_u -= direction * direction.dot(tangent_u);
        tangent_v -= direction * direction.dot(tangent_v);
    }
    result.jacobian.col(0) = tangent_u;
    result.jacobian.col(1) = tangent_v;
    result.gradient = result.jacobian.transpose() * result.residual;
    result.projected_gradient = projected_gradient(
        result.gradient, result.parameter, parameter_tolerance);
    return result;
}

double gradient_tolerance(const Evaluation& evaluation,
                          double distance_tolerance)
{
    const double derivative_scale = std::max(
        evaluation.jacobian.col(0).norm(),
        evaluation.jacobian.col(1).norm());
    const double absolute_scale = std::max(1.0, derivative_scale);
    const double residual_scale = evaluation.residual.norm();
    const double evaluation_roundoff =
        128.0 * std::numeric_limits<double>::epsilon()
        * absolute_scale * std::max(1.0, residual_scale);
    // Once the predicted first-order decrease is below objective-value
    // roundoff, a strict line search cannot distinguish a better parameter.
    const double resolvable_descent =
        8.0 * std::sqrt(std::numeric_limits<double>::epsilon())
        * derivative_scale * std::max(distance_tolerance, residual_scale);
    return std::max(
        {distance_tolerance
             * std::max(derivative_scale, distance_tolerance),
         evaluation_roundoff,
         resolvable_descent});
}

bool finite(const Evaluation& evaluation)
{
    return evaluation.parameter.allFinite()
        && evaluation.surface_point.allFinite()
        && evaluation.segment_point.allFinite()
        && evaluation.residual.allFinite()
        && evaluation.jacobian.allFinite()
        && evaluation.gradient.allFinite()
        && evaluation.projected_gradient.allFinite()
        && std::isfinite(evaluation.t)
        && std::isfinite(evaluation.objective);
}

NurbsElementSegmentClosestPointResult3D make_result(
    const RationalBezierElement3D& element,
    const Evaluation& evaluation,
    bool converged,
    int iterations)
{
    NurbsElementSegmentClosestPointResult3D result;
    const Eigen::Vector2d native = native_parameter(
        element, evaluation.parameter);
    result.converged = converged;
    result.u = native.x();
    result.v = native.y();
    result.t = evaluation.t;
    result.surface_point = evaluation.surface_point;
    result.segment_point = evaluation.segment_point;
    result.distance = std::sqrt(std::max(0.0, 2.0 * evaluation.objective));
    result.projected_gradient_norm =
        evaluation.projected_gradient.norm();
    result.iterations = iterations;
    return result;
}

NurbsElementSegmentClosestPointResult3D solve_from_seed(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_delta,
    double segment_length_squared,
    const NurbsElementSegmentClosestPointOptions3D& options,
    Eigen::Vector2d parameter)
{
    Evaluation current = evaluate(
        element, patch, segment_start, segment_delta,
        segment_length_squared, parameter, options.parameter_tolerance);
    if (!finite(current))
        return make_result(element, current, false, 0);

    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
        const double tolerance = gradient_tolerance(
            current, options.distance_tolerance);

        const Eigen::Matrix2d normal =
            current.jacobian.transpose() * current.jacobian;
        const double damping = std::max(
            options.distance_tolerance * options.distance_tolerance,
            64.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, normal.trace()));
        Eigen::Matrix2d regularized = normal;
        regularized.diagonal().array() += damping;
        for (int coordinate = 0; coordinate < 2; ++coordinate) {
            if (!active_bound(
                    current.gradient[coordinate],
                    current.parameter[coordinate],
                    options.parameter_tolerance)) {
                continue;
            }
            regularized.row(coordinate).setZero();
            regularized.col(coordinate).setZero();
            regularized(coordinate, coordinate) = 1.0;
        }
        Eigen::Vector2d direction =
            -regularized.ldlt().solve(current.projected_gradient);
        if (!direction.allFinite()
            || direction.dot(current.projected_gradient) >= 0.0) {
            direction = -current.projected_gradient;
        }

        Evaluation candidate;
        const auto accept_direction =
            [&](const Eigen::Vector2d& candidate_direction) {
            double step_scale = 1.0;
            for (int line_search = 0; line_search < 24; ++line_search) {
                const Eigen::Vector2d trial = clamp_unit(
                    current.parameter
                    + step_scale * candidate_direction);
                if ((trial - current.parameter).norm()
                    <= options.parameter_tolerance) {
                    step_scale *= 0.5;
                    continue;
                }
                candidate = evaluate(
                    element, patch, segment_start, segment_delta,
                    segment_length_squared, trial,
                    options.parameter_tolerance);
                if (finite(candidate)
                    && candidate.objective < current.objective) {
                    return true;
                }
                step_scale *= 0.5;
            }
            return false;
        };
        bool accepted = accept_direction(direction);
        Eigen::Vector2d coordinate_direction;
        for (int coordinate = 0; coordinate < 2; ++coordinate) {
            coordinate_direction[coordinate] =
                -current.projected_gradient[coordinate]
                / std::max(normal(coordinate, coordinate), damping);
        }
        if (!accepted
            && (direction - coordinate_direction).norm()
                   > options.parameter_tolerance) {
            accepted = accept_direction(coordinate_direction);
        }

        if (!accepted) {
            const bool stationary =
                current.projected_gradient.norm() <= 8.0 * tolerance;
            return make_result(
                element, current, stationary, iteration + 1);
        }

        const double step_norm =
            (candidate.parameter - current.parameter).norm();
        current = candidate;
        if (step_norm <= options.parameter_tolerance) {
            const double updated_tolerance = gradient_tolerance(
                current, options.distance_tolerance);
            if (current.projected_gradient.norm()
                <= 8.0 * updated_tolerance) {
                return make_result(element, current, true, iteration + 1);
            }
        }
    }

    const double tolerance = gradient_tolerance(
        current, options.distance_tolerance);
    return make_result(
        element, current,
        current.projected_gradient.norm() <= 8.0 * tolerance,
        options.max_iterations);
}

void validate_inputs(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end,
    const NurbsElementSegmentClosestPointOptions3D& options)
{
    if (!segment_start.allFinite() || !segment_end.allFinite()) {
        throw std::invalid_argument(
            "NURBS closest-point segment must be finite");
    }
    if (!std::isfinite(options.distance_tolerance)
        || options.distance_tolerance <= 0.0) {
        throw std::invalid_argument(
            "NURBS closest-point distance tolerance must be positive and finite");
    }
    if (!std::isfinite(options.parameter_tolerance)
        || options.parameter_tolerance <= 0.0) {
        throw std::invalid_argument(
            "NURBS closest-point parameter tolerance must be positive and finite");
    }
    if (options.max_iterations <= 0
        || options.max_iterations
               > std::numeric_limits<int>::max() / 2) {
        throw std::invalid_argument(
            "NURBS closest-point iteration limit must be positive "
            "and permit two-seed accounting");
    }
    const double segment_length = (segment_end - segment_start).norm();
    if (!std::isfinite(segment_length)
        || segment_length <= options.distance_tolerance) {
        throw std::invalid_argument(
            "NURBS closest-point segment endpoints must be distinct");
    }
    if (!std::isfinite(element.u0()) || !std::isfinite(element.u1())
        || !std::isfinite(element.v0()) || !std::isfinite(element.v1())
        || element.u0() >= element.u1() || element.v0() >= element.v1()) {
        throw std::invalid_argument("Bezier element has a zero parameter span");
    }
    if (element.u0() < patch.domain_start_u() - options.parameter_tolerance
        || element.u1() > patch.domain_end_u() + options.parameter_tolerance
        || element.v0() < patch.domain_start_v() - options.parameter_tolerance
        || element.v1() > patch.domain_end_v() + options.parameter_tolerance) {
        throw std::invalid_argument(
            "Bezier element parameter span lies outside the source patch");
    }
    (void)element.bounds();
}

} // namespace

NurbsElementSegmentClosestPointResult3D
closest_point_nurbs_bezier_element_to_segment_3d(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end,
    const NurbsElementSegmentClosestPointOptions3D& options,
    std::optional<Eigen::Vector2d> preferred_seed)
{
    validate_inputs(
        element, patch, segment_start, segment_end, options);
    if (preferred_seed && !preferred_seed->allFinite()) {
        throw std::invalid_argument(
            "NURBS closest-point preferred seed must be finite");
    }

    std::vector<Eigen::Vector2d> seeds;
    seeds.reserve(2);
    if (preferred_seed)
        seeds.push_back(unit_parameter(element, *preferred_seed));
    const Eigen::Vector2d center(0.5, 0.5);
    if (seeds.empty()
        || (seeds.front() - center).norm()
               > options.parameter_tolerance) {
        seeds.push_back(center);
    }

    const Eigen::Vector3d segment_delta = segment_end - segment_start;
    const double segment_length_squared = segment_delta.squaredNorm();
    NurbsElementSegmentClosestPointResult3D best;
    int total_iterations = 0;
    for (const Eigen::Vector2d& seed : seeds) {
        const auto candidate = solve_from_seed(
            element, patch, segment_start, segment_delta,
            segment_length_squared, options, seed);
        total_iterations += candidate.iterations;
        const bool closer = candidate.distance < best.distance;
        const bool equally_close =
            candidate.distance == best.distance;
        if (closer
            || (equally_close
                && candidate.converged && !best.converged)) {
            best = candidate;
        }
    }
    best.iterations = total_iterations;
    return best;
}

} // namespace kfbim::geometry3d
