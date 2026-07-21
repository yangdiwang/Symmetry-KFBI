#include "nurbs_bezier_intersection_3d.hpp"

#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kfbim::geometry3d {

namespace {

struct SegmentFrame {
    Eigen::Vector3d start = Eigen::Vector3d::Zero();
    Eigen::Vector3d delta = Eigen::Vector3d::Zero();
    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
    Eigen::Vector3d transverse_first = Eigen::Vector3d::Zero();
    Eigen::Vector3d transverse_second = Eigen::Vector3d::Zero();
    double length = 0.0;
};

struct ProjectedRanges {
    double transverse_first_min = std::numeric_limits<double>::infinity();
    double transverse_first_max = -std::numeric_limits<double>::infinity();
    double transverse_second_min = std::numeric_limits<double>::infinity();
    double transverse_second_max = -std::numeric_limits<double>::infinity();
    double longitudinal_min = std::numeric_limits<double>::infinity();
    double longitudinal_max = -std::numeric_limits<double>::infinity();
};

struct TriangleSeed {
    double u = 0.0;
    double v = 0.0;
    double t = 0.0;
};

struct PendingBox {
    RationalBezierElement3D element;
    int depth = 0;
};

double midpoint(double begin, double end)
{
    return 0.5 * begin + 0.5 * end;
}

bool has_representable_midpoint(double begin, double end)
{
    const double value = midpoint(begin, end);
    return std::isfinite(value) && value > begin && value < end;
}

Eigen::Vector3d projected_control(const Eigen::Vector4d& control)
{
    if (!control.allFinite() || control.w() <= 0.0) {
        throw std::invalid_argument(
            "Bezier element has a nonpositive homogeneous weight");
    }
    const Eigen::Vector3d point = control.head<3>() / control.w();
    if (!point.allFinite())
        throw std::invalid_argument("Bezier element control point must be finite");
    return point;
}

void validate_inputs(const RationalBezierElement3D& element,
                     const NurbsSurfacePatch3D& patch,
                     const Eigen::Vector3d& segment_start,
                     const Eigen::Vector3d& segment_end,
                     const NurbsElementIntersectionOptions3D& options)
{
    if (!segment_start.allFinite() || !segment_end.allFinite())
        throw std::invalid_argument("NURBS intersection segment must be finite");
    if (!std::isfinite(options.geometry_tolerance)
        || options.geometry_tolerance <= 0.0) {
        throw std::invalid_argument(
            "NURBS intersection geometry tolerance must be positive and finite");
    }
    if (!std::isfinite(options.parameter_tolerance)
        || options.parameter_tolerance <= 0.0) {
        throw std::invalid_argument(
            "NURBS intersection parameter tolerance must be positive and finite");
    }
    if (options.max_subdivision_depth < 0)
        throw std::invalid_argument("NURBS intersection depth must be nonnegative");
    if (options.max_newton_iterations <= 0) {
        throw std::invalid_argument(
            "NURBS intersection Newton iteration limit must be positive");
    }
    const double segment_length = (segment_end - segment_start).norm();
    if (!std::isfinite(segment_length)
        || segment_length <= options.geometry_tolerance) {
        throw std::invalid_argument(
            "NURBS intersection segment endpoints must be distinct");
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
    for (int i = 0; i <= element.degree_u; ++i) {
        for (int j = 0; j <= element.degree_v; ++j)
            (void)projected_control(element.control(i, j));
    }
}

SegmentFrame make_segment_frame(const Eigen::Vector3d& segment_start,
                                const Eigen::Vector3d& segment_end)
{
    SegmentFrame frame;
    frame.start = segment_start;
    frame.delta = segment_end - segment_start;
    frame.length = frame.delta.norm();
    frame.direction = frame.delta / frame.length;

    Eigen::Index least_aligned = 0;
    frame.direction.cwiseAbs().minCoeff(&least_aligned);
    Eigen::Vector3d axis = Eigen::Vector3d::Zero();
    axis[least_aligned] = 1.0;
    frame.transverse_first = frame.direction.cross(axis).normalized();
    frame.transverse_second =
        frame.direction.cross(frame.transverse_first).normalized();
    return frame;
}

ProjectedRanges projected_ranges(const RationalBezierElement3D& element,
                                 const SegmentFrame& frame)
{
    ProjectedRanges ranges;
    for (int i = 0; i <= element.degree_u; ++i) {
        for (int j = 0; j <= element.degree_v; ++j) {
            const Eigen::Vector3d relative =
                projected_control(element.control(i, j)) - frame.start;
            const double first = relative.dot(frame.transverse_first);
            const double second = relative.dot(frame.transverse_second);
            const double longitudinal = relative.dot(frame.direction);
            ranges.transverse_first_min =
                std::min(ranges.transverse_first_min, first);
            ranges.transverse_first_max =
                std::max(ranges.transverse_first_max, first);
            ranges.transverse_second_min =
                std::min(ranges.transverse_second_min, second);
            ranges.transverse_second_max =
                std::max(ranges.transverse_second_max, second);
            ranges.longitudinal_min =
                std::min(ranges.longitudinal_min, longitudinal);
            ranges.longitudinal_max =
                std::max(ranges.longitudinal_max, longitudinal);
        }
    }
    return ranges;
}

bool is_conservative_candidate(const ProjectedRanges& ranges,
                               const SegmentFrame& frame,
                               double tolerance)
{
    return ranges.transverse_first_min <= tolerance
        && ranges.transverse_first_max >= -tolerance
        && ranges.transverse_second_min <= tolerance
        && ranges.transverse_second_max >= -tolerance
        && ranges.longitudinal_min <= frame.length + tolerance
        && ranges.longitudinal_max >= -tolerance;
}

double projected_control_variation(const RationalBezierElement3D& element,
                                   bool along_u)
{
    double maximum = 0.0;
    if (along_u) {
        for (int j = 0; j <= element.degree_v; ++j) {
            double variation = 0.0;
            for (int i = 1; i <= element.degree_u; ++i) {
                variation +=
                    (projected_control(element.control(i, j))
                     - projected_control(element.control(i - 1, j))).norm();
            }
            maximum = std::max(maximum, variation);
        }
    } else {
        for (int i = 0; i <= element.degree_u; ++i) {
            double variation = 0.0;
            for (int j = 1; j <= element.degree_v; ++j) {
                variation +=
                    (projected_control(element.control(i, j))
                     - projected_control(element.control(i, j - 1))).norm();
            }
            maximum = std::max(maximum, variation);
        }
    }
    return maximum;
}

std::optional<TriangleSeed> intersect_triangle(
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2,
    const Eigen::Vector2d& uv0,
    const Eigen::Vector2d& uv1,
    const Eigen::Vector2d& uv2,
    const SegmentFrame& frame,
    double parameter_tolerance)
{
    Eigen::Matrix3d matrix;
    matrix.col(0) = p1 - p0;
    matrix.col(1) = p2 - p0;
    matrix.col(2) = -frame.delta;
    Eigen::FullPivLU<Eigen::Matrix3d> factorization(matrix);
    if (factorization.rank() < 3)
        return std::nullopt;
    const Eigen::Vector3d solution =
        factorization.solve(frame.start - p0);
    if (!solution.allFinite())
        return std::nullopt;
    const double first = solution.x();
    const double second = solution.y();
    const double t = solution.z();
    if (first < -parameter_tolerance || second < -parameter_tolerance
        || first + second > 1.0 + parameter_tolerance
        || t < -parameter_tolerance || t > 1.0 + parameter_tolerance) {
        return std::nullopt;
    }
    const double clamped_first = std::clamp(first, 0.0, 1.0);
    const double clamped_second = std::clamp(second, 0.0, 1.0);
    const Eigen::Vector2d uv = uv0
        + clamped_first * (uv1 - uv0)
        + clamped_second * (uv2 - uv0);
    return TriangleSeed{uv.x(), uv.y(), std::clamp(t, 0.0, 1.0)};
}

std::vector<TriangleSeed> corner_triangle_seeds(
    const RationalBezierElement3D& element,
    const SegmentFrame& frame,
    double parameter_tolerance,
    int& triangle_seed_hits)
{
    const double u0 = element.u0();
    const double u1 = element.u1();
    const double v0 = element.v0();
    const double v1 = element.v1();
    const Eigen::Vector3d p00 = element.evaluate(u0, v0);
    const Eigen::Vector3d p10 = element.evaluate(u1, v0);
    const Eigen::Vector3d p01 = element.evaluate(u0, v1);
    const Eigen::Vector3d p11 = element.evaluate(u1, v1);

    std::vector<TriangleSeed> seeds;
    const auto append = [&](const std::optional<TriangleSeed>& seed) {
        if (!seed)
            return;
        ++triangle_seed_hits;
        const bool duplicate = std::any_of(
            seeds.begin(), seeds.end(), [&](const TriangleSeed& existing) {
                return std::abs(existing.u - seed->u) <= parameter_tolerance
                    && std::abs(existing.v - seed->v) <= parameter_tolerance
                    && std::abs(existing.t - seed->t) <= parameter_tolerance;
            });
        if (!duplicate)
            seeds.push_back(*seed);
    };
    append(intersect_triangle(
        p00, p10, p11, {u0, v0}, {u1, v0}, {u1, v1},
        frame, parameter_tolerance));
    append(intersect_triangle(
        p00, p11, p01, {u0, v0}, {u1, v1}, {u0, v1},
        frame, parameter_tolerance));
    return seeds;
}

std::optional<NurbsElementRoot3D> native_newton(
    const RationalBezierElement3D& box,
    const NurbsSurfacePatch3D& patch,
    const SegmentFrame& frame,
    const TriangleSeed& seed,
    const NurbsElementIntersectionOptions3D& options,
    NurbsElementIntersectionDiagnostics3D& diagnostics)
{
    ++diagnostics.newton_attempts;
    Eigen::Vector3d state(seed.u, seed.v, seed.t);

    const auto in_box = [&](const Eigen::Vector3d& candidate) {
        return candidate.x() >= box.u0() - options.parameter_tolerance
            && candidate.x() <= box.u1() + options.parameter_tolerance
            && candidate.y() >= box.v0() - options.parameter_tolerance
            && candidate.y() <= box.v1() + options.parameter_tolerance
            && candidate.z() >= -options.parameter_tolerance
            && candidate.z() <= 1.0 + options.parameter_tolerance;
    };
    const auto clamp_to_box = [&](Eigen::Vector3d candidate) {
        candidate.x() = std::clamp(candidate.x(), box.u0(), box.u1());
        candidate.y() = std::clamp(candidate.y(), box.v0(), box.v1());
        candidate.z() = std::clamp(candidate.z(), 0.0, 1.0);
        return candidate;
    };
    if (!state.allFinite() || !in_box(state))
        return std::nullopt;
    state = clamp_to_box(state);

    for (int iteration = 0; iteration <= options.max_newton_iterations;
         ++iteration) {
        const NurbsSurfaceDerivatives3D derivatives =
            patch.evaluate_with_derivatives(state.x(), state.y());
        const Eigen::Vector3d segment_point =
            frame.start + state.z() * frame.delta;
        const Eigen::Vector3d residual_vector =
            derivatives.point - segment_point;
        const double residual = residual_vector.norm();
        if (!std::isfinite(residual))
            return std::nullopt;
        if (residual <= options.geometry_tolerance
            && state.z() >= -options.parameter_tolerance
            && state.z() <= 1.0 + options.parameter_tolerance) {
            const Eigen::Vector3d cross = derivatives.du.cross(derivatives.dv);
            const double normal_norm = cross.norm();
            if (!std::isfinite(normal_norm) || normal_norm <= 0.0)
                return std::nullopt;
            NurbsElementRoot3D root;
            root.patch_index = box.patch_index;
            root.component = box.component;
            root.u = state.x();
            root.v = state.y();
            root.t = state.z();
            root.point = derivatives.point;
            root.normal = cross / normal_norm;
            root.residual = residual;
            root.transversality =
                std::abs(root.normal.dot(frame.direction));
            return root;
        }
        if (iteration == options.max_newton_iterations)
            break;

        Eigen::Matrix3d jacobian;
        jacobian.col(0) = derivatives.du;
        jacobian.col(1) = derivatives.dv;
        jacobian.col(2) = -frame.delta;
        Eigen::FullPivLU<Eigen::Matrix3d> factorization(jacobian);
        if (factorization.rank() < 3)
            return std::nullopt;
        const Eigen::Vector3d step = factorization.solve(-residual_vector);
        if (!step.allFinite())
            return std::nullopt;
        ++diagnostics.newton_iterations;

        bool accepted = false;
        double step_scale = 1.0;
        for (int line_search = 0; line_search < 64; ++line_search) {
            Eigen::Vector3d candidate = state + step_scale * step;
            if (in_box(candidate)) {
                candidate = clamp_to_box(candidate);
                const NurbsSurfaceDerivatives3D candidate_derivatives =
                    patch.evaluate_with_derivatives(
                        candidate.x(), candidate.y());
                const double candidate_residual =
                    (candidate_derivatives.point
                     - (frame.start + candidate.z() * frame.delta)).norm();
                if (std::isfinite(candidate_residual)
                    && (candidate_residual < residual
                        || candidate_residual <= options.geometry_tolerance)) {
                    state = candidate;
                    accepted = true;
                    break;
                }
            }
            step_scale *= 0.5;
        }
        if (!accepted)
            return std::nullopt;
    }
    return std::nullopt;
}

bool roots_agree(const NurbsElementRoot3D& first,
                 const NurbsElementRoot3D& second,
                 const NurbsElementIntersectionOptions3D& options)
{
    return first.patch_index == second.patch_index
        && (first.point - second.point).norm()
               <= 8.0 * options.geometry_tolerance
        && std::abs(first.t - second.t)
               <= 8.0 * options.parameter_tolerance
        && std::abs(first.u - second.u)
               <= 8.0 * options.parameter_tolerance
        && std::abs(first.v - second.v)
               <= 8.0 * options.parameter_tolerance;
}

bool append_root(std::vector<NurbsElementRoot3D>& roots,
                 NurbsElementRoot3D root,
                 const NurbsElementIntersectionOptions3D& options)
{
    for (NurbsElementRoot3D& existing : roots) {
        if (!roots_agree(existing, root, options))
            continue;
        if (root.residual < existing.residual)
            existing = std::move(root);
        return false;
    }
    roots.push_back(std::move(root));
    return true;
}

bool root_lies_in_box(const NurbsElementRoot3D& root,
                      const RationalBezierElement3D& box,
                      double parameter_tolerance)
{
    return root.u >= box.u0() - parameter_tolerance
        && root.u <= box.u1() + parameter_tolerance
        && root.v >= box.v0() - parameter_tolerance
        && root.v <= box.v1() + parameter_tolerance;
}

std::vector<double> overlap_candidate_parameters(
    const RationalBezierElement3D& element,
    const SegmentFrame& frame)
{
    std::vector<double> parameters{0.25, 0.5, 0.75};
    const Eigen::Vector3d center = element.evaluate(
        midpoint(element.u0(), element.u1()),
        midpoint(element.v0(), element.v1()));
    parameters.push_back(
        (center - frame.start).dot(frame.delta) / frame.delta.squaredNorm());
    return parameters;
}

bool is_strict_interior(double value,
                        double begin,
                        double end,
                        double tolerance)
{
    return value > begin + tolerance && value < end - tolerance;
}

bool certifies_planar_overlap(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const SegmentFrame& frame,
    const NurbsElementIntersectionOptions3D& options)
{
    std::vector<Eigen::Vector3d> controls;
    controls.reserve(static_cast<std::size_t>(
        (element.degree_u + 1) * (element.degree_v + 1)));
    for (int i = 0; i <= element.degree_u; ++i) {
        for (int j = 0; j <= element.degree_v; ++j)
            controls.push_back(projected_control(element.control(i, j)));
    }
    if (controls.size() < 3)
        return false;

    const Eigen::Vector3d origin = controls.front();
    std::size_t first_index = controls.size();
    for (std::size_t i = 1; i < controls.size(); ++i) {
        if ((controls[i] - origin).norm() > options.geometry_tolerance) {
            first_index = i;
            break;
        }
    }
    if (first_index == controls.size())
        return false;
    const Eigen::Vector3d first_direction = controls[first_index] - origin;
    Eigen::Vector3d plane_normal = Eigen::Vector3d::Zero();
    for (std::size_t i = first_index + 1; i < controls.size(); ++i) {
        const Eigen::Vector3d cross =
            first_direction.cross(controls[i] - origin);
        if (cross.norm() > options.geometry_tolerance) {
            plane_normal = cross.normalized();
            break;
        }
    }
    if (plane_normal.isZero(0.0))
        return false;
    for (const Eigen::Vector3d& control : controls) {
        if (std::abs((control - origin).dot(plane_normal))
            > options.geometry_tolerance) {
            return false;
        }
    }
    if (std::abs((frame.start - origin).dot(plane_normal))
            > options.geometry_tolerance
        || std::abs((frame.start + frame.delta - origin).dot(plane_normal))
            > options.geometry_tolerance) {
        return false;
    }

    for (const double t : overlap_candidate_parameters(element, frame)) {
        if (!is_strict_interior(t, 0.0, 1.0, options.parameter_tolerance))
            continue;
        try {
            const auto projection = patch.project(frame.start + t * frame.delta);
            if (projection.distance > options.geometry_tolerance
                || !is_strict_interior(
                    projection.u, element.u0(), element.u1(),
                    options.parameter_tolerance)
                || !is_strict_interior(
                    projection.v, element.v0(), element.v1(),
                    options.parameter_tolerance)) {
                continue;
            }
            const NurbsSurfaceDerivatives3D derivatives =
                patch.evaluate_with_derivatives(projection.u, projection.v);
            if (derivatives.du.cross(derivatives.dv).norm()
                > options.geometry_tolerance) {
                return true;
            }
        } catch (const std::exception&) {
        }
    }
    return false;
}

bool controls_form_parallel_rulings(
    const RationalBezierElement3D& element,
    bool ruling_along_u,
    const SegmentFrame& frame,
    double tolerance)
{
    const int ruling_degree =
        ruling_along_u ? element.degree_u : element.degree_v;
    const int transverse_degree =
        ruling_along_u ? element.degree_v : element.degree_u;
    if (ruling_degree < 1)
        return false;

    const double reference_weight = element.control(0, 0).w();
    for (int transverse = 0; transverse <= transverse_degree; ++transverse) {
        for (int ruling = 0; ruling <= ruling_degree; ++ruling) {
            const double weight = ruling_along_u
                ? element.control(ruling, transverse).w()
                : element.control(transverse, ruling).w();
            const double ruling_reference = ruling_along_u
                ? element.control(ruling, 0).w()
                : element.control(0, ruling).w();
            const double transverse_reference = ruling_along_u
                ? element.control(0, transverse).w()
                : element.control(transverse, 0).w();
            const double lhs = weight * reference_weight;
            const double rhs = ruling_reference * transverse_reference;
            const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
            if (std::abs(lhs - rhs) > tolerance * scale)
                return false;
        }
    }

    bool has_nonzero_ruling = false;
    for (int transverse = 0; transverse <= transverse_degree; ++transverse) {
        const Eigen::Vector3d origin = projected_control(
            ruling_along_u ? element.control(0, transverse)
                           : element.control(transverse, 0));
        for (int ruling = 1; ruling <= ruling_degree; ++ruling) {
            const Eigen::Vector3d point = projected_control(
                ruling_along_u ? element.control(ruling, transverse)
                               : element.control(transverse, ruling));
            const Eigen::Vector3d difference = point - origin;
            if (difference.cross(frame.direction).norm() > tolerance)
                return false;
            has_nonzero_ruling = has_nonzero_ruling
                || std::abs(difference.dot(frame.direction)) > tolerance;
        }
    }
    return has_nonzero_ruling;
}

bool certifies_ruled_overlap(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const SegmentFrame& frame,
    const NurbsElementIntersectionOptions3D& options,
    bool ruling_along_u)
{
    if (!controls_form_parallel_rulings(
            element, ruling_along_u, frame, options.geometry_tolerance)) {
        return false;
    }

    for (const double t : overlap_candidate_parameters(element, frame)) {
        if (!is_strict_interior(t, 0.0, 1.0, options.parameter_tolerance))
            continue;
        try {
            const auto projection = patch.project(frame.start + t * frame.delta);
            if (projection.distance > options.geometry_tolerance)
                continue;
            const double fixed_parameter =
                ruling_along_u ? projection.v : projection.u;
            const double fixed_begin =
                ruling_along_u ? element.v0() : element.u0();
            const double fixed_end =
                ruling_along_u ? element.v1() : element.u1();
            if (!is_strict_interior(
                    fixed_parameter, fixed_begin, fixed_end,
                    options.parameter_tolerance)) {
                continue;
            }

            const Eigen::Vector3d ruling_start = ruling_along_u
                ? element.evaluate(element.u0(), fixed_parameter)
                : element.evaluate(fixed_parameter, element.v0());
            const Eigen::Vector3d ruling_end = ruling_along_u
                ? element.evaluate(element.u1(), fixed_parameter)
                : element.evaluate(fixed_parameter, element.v1());
            if ((ruling_start - frame.start).cross(frame.direction).norm()
                    > options.geometry_tolerance
                || (ruling_end - frame.start).cross(frame.direction).norm()
                    > options.geometry_tolerance
                || (ruling_end - ruling_start).cross(frame.direction).norm()
                    > options.geometry_tolerance) {
                continue;
            }

            const double first_longitudinal =
                (ruling_start - frame.start).dot(frame.direction);
            const double second_longitudinal =
                (ruling_end - frame.start).dot(frame.direction);
            const double overlap_begin = std::max(
                0.0, std::min(first_longitudinal, second_longitudinal));
            const double overlap_end = std::min(
                frame.length,
                std::max(first_longitudinal, second_longitudinal));
            if (overlap_end - overlap_begin > options.geometry_tolerance) {
                const NurbsSurfaceDerivatives3D derivatives =
                    patch.evaluate_with_derivatives(projection.u, projection.v);
                if (derivatives.du.cross(derivatives.dv).norm()
                    > options.geometry_tolerance) {
                    return true;
                }
            }
        } catch (const std::exception&) {
        }
    }
    return false;
}

bool certifies_supported_overlap(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const SegmentFrame& frame,
    const NurbsElementIntersectionOptions3D& options)
{
    return certifies_planar_overlap(element, patch, frame, options)
        || certifies_ruled_overlap(
            element, patch, frame, options, true)
        || certifies_ruled_overlap(
            element, patch, frame, options, false);
}

TriangleSeed midpoint_seed(const RationalBezierElement3D& box,
                           const SegmentFrame& frame)
{
    const double u = midpoint(box.u0(), box.u1());
    const double v = midpoint(box.v0(), box.v1());
    const Eigen::Vector3d point = box.evaluate(u, v);
    const double t = std::clamp(
        (point - frame.start).dot(frame.delta) / frame.delta.squaredNorm(),
        0.0, 1.0);
    return {u, v, t};
}

} // namespace

NurbsElementIntersectionResult3D intersect_nurbs_bezier_element_3d(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end,
    const NurbsElementIntersectionOptions3D& options)
{
    validate_inputs(element, patch, segment_start, segment_end, options);
    const SegmentFrame frame = make_segment_frame(segment_start, segment_end);

    NurbsElementIntersectionResult3D result;
    const ProjectedRanges initial_ranges = projected_ranges(element, frame);
    ++result.diagnostics.subdivision_boxes;
    if (!is_conservative_candidate(
            initial_ranges, frame, options.geometry_tolerance)) {
        ++result.diagnostics.conservative_rejections;
        return result;
    }

    if (certifies_supported_overlap(element, patch, frame, options)) {
        result.overlap_detected = true;
        return result;
    }

    if (options.use_triangle_seed) {
        const std::vector<TriangleSeed> seeds = corner_triangle_seeds(
            element, frame, options.parameter_tolerance,
            result.diagnostics.triangle_seed_hits);
        for (const TriangleSeed& seed : seeds) {
            const auto root = native_newton(
                element, patch, frame, seed, options, result.diagnostics);
            if (root)
                (void)append_root(result.roots, *root, options);
        }
    }

    std::vector<PendingBox> pending{{element, 0}};
    bool skipped_initial_count = true;
    while (!pending.empty()) {
        PendingBox current = std::move(pending.back());
        pending.pop_back();
        if (skipped_initial_count)
            skipped_initial_count = false;
        else
            ++result.diagnostics.subdivision_boxes;

        const ProjectedRanges ranges = projected_ranges(current.element, frame);
        if (!is_conservative_candidate(
                ranges, frame, options.geometry_tolerance)) {
            ++result.diagnostics.conservative_rejections;
            continue;
        }

        if (current.depth >= options.max_subdivision_depth) {
            const bool already_validated = std::any_of(
                result.roots.begin(), result.roots.end(),
                [&](const NurbsElementRoot3D& root) {
                    return root_lies_in_box(
                        root, current.element, options.parameter_tolerance);
                });
            if (already_validated)
                continue;
            const auto root = native_newton(
                current.element, patch, frame,
                midpoint_seed(current.element, frame),
                options, result.diagnostics);
            if (!root) {
                ++result.diagnostics.unresolved_boxes;
                continue;
            }
            if (append_root(result.roots, *root, options))
                ++result.diagnostics.roots_recovered_without_triangle_seed;
            continue;
        }

        bool split_along_u =
            projected_control_variation(current.element, true)
            >= projected_control_variation(current.element, false);
        if (split_along_u
            && !has_representable_midpoint(
                current.element.u0(), current.element.u1())) {
            split_along_u = false;
        } else if (!split_along_u
                   && !has_representable_midpoint(
                       current.element.v0(), current.element.v1())) {
            split_along_u = true;
        }
        if ((split_along_u
                && !has_representable_midpoint(
                    current.element.u0(), current.element.u1()))
            || (!split_along_u
                && !has_representable_midpoint(
                    current.element.v0(), current.element.v1()))) {
            ++result.diagnostics.unresolved_boxes;
            continue;
        }

        const auto children = split_along_u
            ? current.element.split_u()
            : current.element.split_v();
        pending.push_back({children[1], current.depth + 1});
        pending.push_back({children[0], current.depth + 1});
    }

    if (result.diagnostics.unresolved_boxes > 0) {
        throw std::runtime_error(
            "unresolved conservative NURBS intersection candidate");
    }
    std::sort(
        result.roots.begin(), result.roots.end(),
        [](const NurbsElementRoot3D& first,
           const NurbsElementRoot3D& second) {
            if (first.t != second.t)
                return first.t < second.t;
            if (first.u != second.u)
                return first.u < second.u;
            return first.v < second.v;
        });
    return result;
}

} // namespace kfbim::geometry3d
