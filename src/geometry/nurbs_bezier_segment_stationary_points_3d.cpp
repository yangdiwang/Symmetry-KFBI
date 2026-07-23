#include "nurbs_bezier_segment_stationary_points_3d.hpp"

#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace kfbim::geometry3d {
namespace {

struct StationaryState {
    Eigen::Vector2d equations = Eigen::Vector2d::Zero();
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d residual = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double t = 0.0;
    double distance = 0.0;
    double tangent_measure = 0.0;
};

bool finite_box(const NurbsParameterBox2D& box)
{
    return std::isfinite(box.u0) && std::isfinite(box.u1)
        && std::isfinite(box.v0) && std::isfinite(box.v1)
        && box.u0 < box.u1 && box.v0 < box.v1;
}

} // namespace

NurbsLineSurfaceStationaryPoint3D
solve_nurbs_bezier_segment_stationary_point_3d(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end,
    const NurbsParameterBox2D& parameter_box,
    const Eigen::Vector2d& initial_parameter,
    const NurbsLineSurfaceStationaryPointOptions3D& options)
{
    if (!segment_start.allFinite() || !segment_end.allFinite()
        || !initial_parameter.allFinite()) {
        throw std::invalid_argument(
            "NURBS stationary-point inputs must be finite");
    }
    if (!std::isfinite(options.geometry_tolerance)
        || options.geometry_tolerance <= 0.0
        || !std::isfinite(options.parameter_tolerance)
        || options.parameter_tolerance <= 0.0
        || options.max_iterations <= 0) {
        throw std::invalid_argument(
            "NURBS stationary-point options are invalid");
    }
    if (!finite_box(parameter_box)
        || parameter_box.u0 < element.u0() - options.parameter_tolerance
        || parameter_box.u1 > element.u1() + options.parameter_tolerance
        || parameter_box.v0 < element.v0() - options.parameter_tolerance
        || parameter_box.v1 > element.v1() + options.parameter_tolerance) {
        throw std::invalid_argument(
            "NURBS stationary-point box lies outside the Bezier element");
    }

    const Eigen::Vector3d delta = segment_end - segment_start;
    const double squared_length = delta.squaredNorm();
    const double length = std::sqrt(squared_length);
    if (!std::isfinite(length) || length <= options.geometry_tolerance) {
        throw std::invalid_argument(
            "NURBS stationary-point segment endpoints must be distinct");
    }
    const Eigen::Vector3d direction = delta / length;

    const auto evaluate = [&](const Eigen::Vector2d& parameter) {
        const NurbsSurfaceDerivatives3D derivatives =
            patch.evaluate_with_derivatives(parameter.x(), parameter.y());
        StationaryState state;
        state.point = derivatives.point;
        state.t = std::clamp(
            (state.point - segment_start).dot(delta) / squared_length,
            0.0, 1.0);
        state.residual =
            state.point - (segment_start + state.t * delta);
        state.equations = {
            state.residual.dot(derivatives.du),
            state.residual.dot(derivatives.dv)};
        state.distance = state.residual.norm();
        const Eigen::Vector3d cross =
            derivatives.du.cross(derivatives.dv);
        const double normal_norm = cross.norm();
        if (!state.point.allFinite() || !state.residual.allFinite()
            || !state.equations.allFinite()
            || !std::isfinite(state.distance)
            || !std::isfinite(normal_norm) || normal_norm <= 0.0) {
            throw std::runtime_error(
                "NURBS stationary-point evaluation is singular");
        }
        state.normal = cross / normal_norm;
        state.tangent_measure = std::abs(state.normal.dot(direction));
        return state;
    };
    const auto clamp_parameter = [&](Eigen::Vector2d parameter) {
        parameter.x() = std::clamp(
            parameter.x(), parameter_box.u0, parameter_box.u1);
        parameter.y() = std::clamp(
            parameter.y(), parameter_box.v0, parameter_box.v1);
        return parameter;
    };

    Eigen::Vector2d parameter = clamp_parameter(initial_parameter);
    NurbsLineSurfaceStationaryPoint3D result;
    StationaryState state = evaluate(parameter);
    const double equation_scale = std::max(
        {1.0, element.bounds().max_extent(), length});
    const double equation_tolerance = std::max(
        64.0 * std::numeric_limits<double>::epsilon()
            * equation_scale * equation_scale,
        options.geometry_tolerance * options.geometry_tolerance);

    for (int iteration = 0; iteration <= options.max_iterations;
         ++iteration) {
        result.u = parameter.x();
        result.v = parameter.y();
        result.t = state.t;
        result.surface_point = state.point;
        result.distance = state.distance;
        result.tangent_measure = state.tangent_measure;
        result.iterations = iteration;
        if (state.equations.norm() <= equation_tolerance) {
            result.converged = true;
            return result;
        }
        if (iteration == options.max_iterations)
            break;

        Eigen::Matrix2d jacobian;
        for (int axis = 0; axis < 2; ++axis) {
            const double begin =
                axis == 0 ? parameter_box.u0 : parameter_box.v0;
            const double end =
                axis == 0 ? parameter_box.u1 : parameter_box.v1;
            const double span = end - begin;
            const double nominal_step = std::max(
                1.0e-6 * span,
                64.0 * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, std::abs(parameter[axis])));
            Eigen::Vector2d lower = parameter;
            Eigen::Vector2d upper = parameter;
            lower[axis] = std::max(begin, parameter[axis] - nominal_step);
            upper[axis] = std::min(end, parameter[axis] + nominal_step);
            const double denominator = upper[axis] - lower[axis];
            if (!(denominator > 0.0))
                return result;
            jacobian.col(axis) =
                (evaluate(upper).equations - evaluate(lower).equations)
                / denominator;
        }
        Eigen::FullPivLU<Eigen::Matrix2d> factorization(jacobian);
        if (factorization.rank() < 2)
            return result;
        const Eigen::Vector2d step =
            factorization.solve(-state.equations);
        if (!step.allFinite())
            return result;

        bool accepted = false;
        double scale = 1.0;
        for (int damping = 0; damping < 24; ++damping) {
            const Eigen::Vector2d candidate =
                clamp_parameter(parameter + scale * step);
            if ((candidate - parameter).norm()
                <= options.parameter_tolerance) {
                scale *= 0.5;
                continue;
            }
            const StationaryState candidate_state = evaluate(candidate);
            if (candidate_state.equations.squaredNorm()
                < state.equations.squaredNorm()) {
                parameter = candidate;
                state = candidate_state;
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        if (!accepted)
            return result;
    }
    return result;
}

std::vector<NurbsLineSurfaceStationaryPoint3D>
find_nurbs_bezier_segment_stationary_points_3d(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end,
    const NurbsParameterBox2D& parameter_box,
    const std::vector<Eigen::Vector2d>& initial_parameters,
    const NurbsLineSurfaceStationaryPointOptions3D& options)
{
    const double segment_length = (segment_end - segment_start).norm();
    if (!std::isfinite(segment_length)
        || segment_length <= options.geometry_tolerance) {
        throw std::invalid_argument(
            "NURBS stationary-point segment endpoints must be distinct");
    }
    const double coordinate_scale = std::max({
        1.0,
        segment_start.cwiseAbs().maxCoeff(),
        segment_end.cwiseAbs().maxCoeff(),
        element.bounds().lower.cwiseAbs().maxCoeff(),
        element.bounds().upper.cwiseAbs().maxCoeff()});
    const double stationary_numerical_tolerance =
        16.0 * std::sqrt(std::numeric_limits<double>::epsilon());
    const double parameter_deduplication_tolerance = std::max(
        8.0 * options.parameter_tolerance,
        stationary_numerical_tolerance);
    const double physical_deduplication_tolerance = std::max(
        16.0 * options.geometry_tolerance,
        stationary_numerical_tolerance * coordinate_scale);
    const double t_tolerance = std::max(
        16.0 * std::numeric_limits<double>::epsilon(),
        physical_deduplication_tolerance / segment_length);
    std::vector<NurbsLineSurfaceStationaryPoint3D> points;
    points.reserve(initial_parameters.size());
    for (const Eigen::Vector2d& seed : initial_parameters) {
        const NurbsLineSurfaceStationaryPoint3D candidate =
            solve_nurbs_bezier_segment_stationary_point_3d(
                element, patch, segment_start, segment_end,
                parameter_box, seed, options);
        if (!candidate.converged)
            continue;
        const auto duplicate = std::find_if(
            points.begin(), points.end(), [&](const auto& existing) {
                return std::abs(existing.u - candidate.u)
                           <= parameter_deduplication_tolerance
                    && std::abs(existing.v - candidate.v)
                           <= parameter_deduplication_tolerance
                    && std::abs(existing.t - candidate.t) <= t_tolerance
                    && (existing.surface_point - candidate.surface_point)
                           .norm()
                           <= physical_deduplication_tolerance;
            });
        if (duplicate == points.end())
            points.push_back(candidate);
    }
    std::sort(points.begin(), points.end(), [](const auto& first,
                                               const auto& second) {
        if (first.t != second.t)
            return first.t < second.t;
        if (first.u != second.u)
            return first.u < second.u;
        return first.v < second.v;
    });
    return points;
}
} // namespace kfbim::geometry3d
