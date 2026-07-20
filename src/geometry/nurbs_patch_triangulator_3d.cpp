#include "nurbs_patch_triangulator_3d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace kfbim::geometry3d {

namespace {

constexpr double kTiny = 1.0e-14;

enum class PatchEdge {
    UMin,
    UMax,
    VMin,
    VMax
};

struct RectCell {
    double u0 = 0.0;
    double u1 = 0.0;
    double v0 = 0.0;
    double v1 = 0.0;
    int depth = 0;
};

struct RectQuality {
    double max_edge = 0.0;
    double metric_ratio = 1.0;
    int split_direction = 0;
};

struct ParamCornerCut {
    bool enabled = false;
    Eigen::Vector2d corner = Eigen::Vector2d::Zero();
    double du = 0.0;
    double dv = 0.0;
    int u_sign = 1;
    int v_sign = 1;
};

using PatchCornerCuts = std::array<ParamCornerCut, 4>;

struct NodeStore {
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> normals;
    std::vector<double> weights;
    std::map<std::tuple<int, long long, long long, long long>, int> node_by_key;
};

struct PatchEdgeDescriptor {
    int patch = -1;
    PatchEdge edge = PatchEdge::UMin;
    Eigen::Vector3d start = Eigen::Vector3d::Zero();
    Eigen::Vector3d mid = Eigen::Vector3d::Zero();
    Eigen::Vector3d end = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
};

class PatchDisjointSet {
public:
    explicit PatchDisjointSet(int count) : parent_(static_cast<std::size_t>(count))
    {
        for (int i = 0; i < count; ++i)
            parent_[static_cast<std::size_t>(i)] = i;
    }

    int find(int value)
    {
        int& parent = parent_[static_cast<std::size_t>(value)];
        if (parent != value)
            parent = find(parent);
        return parent;
    }

    void unite(int a, int b)
    {
        a = find(a);
        b = find(b);
        if (a != b)
            parent_[static_cast<std::size_t>(b)] = a;
    }

private:
    std::vector<int> parent_;
};

void validate_options(double h, const NurbsPatchTriangulatorOptions3D& options)
{
    if (!std::isfinite(h) || h <= 0.0)
        throw std::invalid_argument("NURBS patch triangulation requires positive h");
    if (!std::isfinite(options.edge_buffer_factor)
        || options.edge_buffer_factor <= 0.0) {
        throw std::invalid_argument("edge_buffer_factor must be positive");
    }
    if (!std::isfinite(options.edge_sample_step_factor)
        || options.edge_sample_step_factor <= 0.0) {
        throw std::invalid_argument("edge_sample_step_factor must be positive");
    }
    if (!std::isfinite(options.max_param_offset_fraction)
        || options.max_param_offset_fraction <= 0.0
        || options.max_param_offset_fraction >= 0.5) {
        throw std::invalid_argument(
            "max_param_offset_fraction must be in the interval (0, 0.5)");
    }
    if (!std::isfinite(options.H_factor) || options.H_factor <= 0.0)
        throw std::invalid_argument("H_factor must be positive");
    if (!std::isfinite(options.max_edge_over_H)
        || options.max_edge_over_H <= 0.0) {
        throw std::invalid_argument("max_edge_over_H must be positive");
    }
    if (!std::isfinite(options.metric_ratio_tolerance)
        || options.metric_ratio_tolerance < 1.0) {
        throw std::invalid_argument("metric_ratio_tolerance must be at least one");
    }
    if (options.min_edge_samples < 2)
        throw std::invalid_argument("min_edge_samples must be at least two");
    if (options.edge_length_quadrature_samples < 2)
        throw std::invalid_argument(
            "edge_length_quadrature_samples must be at least two");
    if (options.max_depth < 0)
        throw std::invalid_argument("max_depth must be nonnegative");
    if (!std::isfinite(options.edge_match_tolerance)
        || options.edge_match_tolerance <= 0.0) {
        throw std::invalid_argument("edge_match_tolerance must be positive");
    }
    if (!std::isfinite(options.smooth_normal_cosine)
        || options.smooth_normal_cosine < -1.0
        || options.smooth_normal_cosine > 1.0) {
        throw std::invalid_argument(
            "smooth_normal_cosine must lie in [-1, 1]");
    }
}

double lerp(double a, double b, double t)
{
    return (1.0 - t) * a + t * b;
}

Eigen::Matrix2d first_fundamental_form(const NurbsSurfaceDerivatives3D& d)
{
    Eigen::Matrix2d metric;
    metric(0, 0) = d.du.dot(d.du);
    metric(0, 1) = d.du.dot(d.dv);
    metric(1, 0) = metric(0, 1);
    metric(1, 1) = d.dv.dot(d.dv);
    return metric;
}

double tangent_speed(const NurbsSurfaceDerivatives3D& d, PatchEdge edge)
{
    switch (edge) {
    case PatchEdge::UMin:
    case PatchEdge::UMax:
        return d.dv.norm();
    case PatchEdge::VMin:
    case PatchEdge::VMax:
        return d.du.norm();
    }
    return 0.0;
}

double cross_edge_speed(const NurbsSurfaceDerivatives3D& d, PatchEdge edge)
{
    const Eigen::Matrix2d g = first_fundamental_form(d);
    const double E = std::max(0.0, g(0, 0));
    const double F = g(0, 1);
    const double G = std::max(0.0, g(1, 1));

    if (edge == PatchEdge::UMin || edge == PatchEdge::UMax) {
        if (G <= kTiny)
            return std::sqrt(E);
        return std::sqrt(std::max(0.0, E - F * F / G));
    }

    if (E <= kTiny)
        return std::sqrt(G);
    return std::sqrt(std::max(0.0, G - F * F / E));
}

std::pair<double, double> edge_parameter_range(const NurbsSurfacePatch3D& patch,
                                               PatchEdge edge)
{
    if (edge == PatchEdge::UMin || edge == PatchEdge::UMax)
        return {patch.domain_start_v(), patch.domain_end_v()};
    return {patch.domain_start_u(), patch.domain_end_u()};
}

std::pair<double, double> edge_uv(const NurbsSurfacePatch3D& patch,
                                   PatchEdge edge,
                                   double edge_parameter)
{
    switch (edge) {
    case PatchEdge::UMin:
        return {patch.domain_start_u(), edge_parameter};
    case PatchEdge::UMax:
        return {patch.domain_end_u(), edge_parameter};
    case PatchEdge::VMin:
        return {edge_parameter, patch.domain_start_v()};
    case PatchEdge::VMax:
        return {edge_parameter, patch.domain_end_v()};
    }
    return {0.0, 0.0};
}

NurbsPatchEdgeOffset3D& edge_offset(NurbsPatchEdgeOffsets3D& offsets,
                                    PatchEdge edge)
{
    switch (edge) {
    case PatchEdge::UMin: return offsets.u_min;
    case PatchEdge::UMax: return offsets.u_max;
    case PatchEdge::VMin: return offsets.v_min;
    case PatchEdge::VMax: return offsets.v_max;
    }
    return offsets.u_min;
}

PatchEdgeDescriptor describe_patch_edge(const NurbsSurfacePatch3D& patch,
                                        int patch_index,
                                        PatchEdge edge)
{
    const auto range = edge_parameter_range(patch, edge);
    const double middle = 0.5 * (range.first + range.second);
    const auto uv0 = edge_uv(patch, edge, range.first);
    const auto uvm = edge_uv(patch, edge, middle);
    const auto uv1 = edge_uv(patch, edge, range.second);
    const NurbsSurfaceDerivatives3D dm =
        patch.evaluate_with_derivatives(uvm.first, uvm.second);
    Eigen::Vector3d normal = dm.du.cross(dm.dv);
    const double normal_length = normal.norm();
    if (!std::isfinite(normal_length) || normal_length <= kTiny)
        throw std::runtime_error("NURBS patch edge has degenerate midpoint normal");
    normal /= normal_length;
    return {patch_index,
            edge,
            patch.evaluate(uv0.first, uv0.second),
            patch.evaluate(uvm.first, uvm.second),
            patch.evaluate(uv1.first, uv1.second),
            normal};
}

bool edge_descriptors_match(const PatchEdgeDescriptor& a,
                            const PatchEdgeDescriptor& b,
                            double tolerance)
{
    const bool direct = (a.start - b.start).norm() <= tolerance
                     && (a.mid - b.mid).norm() <= tolerance
                     && (a.end - b.end).norm() <= tolerance;
    const bool reverse = (a.start - b.end).norm() <= tolerance
                      && (a.mid - b.mid).norm() <= tolerance
                      && (a.end - b.start).norm() <= tolerance;
    return direct || reverse;
}

std::vector<Eigen::Vector3d> unique_feature_vertices(
    const std::vector<PatchEdgeDescriptor>& descriptors,
    double tolerance,
    double smooth_normal_cosine)
{
    std::vector<Eigen::Vector3d> candidates;
    for (const PatchEdgeDescriptor& edge : descriptors) {
        candidates.push_back(edge.start);
        candidates.push_back(edge.end);
    }

    std::vector<Eigen::Vector3d> vertices;
    for (const Eigen::Vector3d& point : candidates) {
        if (std::any_of(vertices.begin(), vertices.end(),
                        [&](const Eigen::Vector3d& existing) {
                            return (existing - point).norm() <= tolerance;
                        })) {
            continue;
        }

        std::vector<Eigen::Vector3d> distinct_normals;
        for (const PatchEdgeDescriptor& edge : descriptors) {
            if ((edge.start - point).norm() > tolerance
                && (edge.end - point).norm() > tolerance) {
                continue;
            }
            const bool duplicate = std::any_of(
                distinct_normals.begin(), distinct_normals.end(),
                [&](const Eigen::Vector3d& normal) {
                    return std::abs(normal.dot(edge.normal))
                        >= smooth_normal_cosine;
                });
            if (!duplicate)
                distinct_normals.push_back(edge.normal);
        }
        if (distinct_normals.size() >= 3)
            vertices.push_back(point);
    }
    return vertices;
}

double edge_length(const NurbsSurfacePatch3D& patch,
                   PatchEdge edge,
                   int num_samples)
{
    const auto range = edge_parameter_range(patch, edge);
    const double a = range.first;
    const double b = range.second;
    const double span = b - a;
    if (span <= 0.0)
        throw std::invalid_argument("NURBS patch has invalid edge parameter span");

    double integral = 0.0;
    for (int i = 0; i < num_samples; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(num_samples - 1);
        const double parameter = lerp(a, b, t);
        const auto uv = edge_uv(patch, edge, parameter);
        const NurbsSurfaceDerivatives3D d =
            patch.evaluate_with_derivatives(uv.first, uv.second);
        const double weight = (i == 0 || i == num_samples - 1) ? 0.5 : 1.0;
        integral += weight * tangent_speed(d, edge);
    }
    return integral * span / static_cast<double>(num_samples - 1);
}

NurbsPatchEdgeOffset3D compute_one_edge_offset(
    const NurbsSurfacePatch3D& patch,
    PatchEdge edge,
    double h,
    const NurbsPatchTriangulatorOptions3D& options)
{
    const int length_samples = std::max(2, options.edge_length_quadrature_samples);
    const double length = edge_length(patch, edge, length_samples);
    const double sample_step = options.edge_sample_step_factor * h;
    const int sample_count = std::max(
        options.min_edge_samples,
        static_cast<int>(std::ceil(length / sample_step)) + 1);

    const auto range = edge_parameter_range(patch, edge);
    const double physical_buffer = options.edge_buffer_factor * h;
    double offset = std::numeric_limits<double>::infinity();
    double min_cross_speed = std::numeric_limits<double>::infinity();
    double max_cross_speed = 0.0;

    for (int i = 0; i < sample_count; ++i) {
        const double t = sample_count == 1
            ? 0.0
            : static_cast<double>(i) / static_cast<double>(sample_count - 1);
        const double parameter = lerp(range.first, range.second, t);
        const auto uv = edge_uv(patch, edge, parameter);
        const NurbsSurfaceDerivatives3D d =
            patch.evaluate_with_derivatives(uv.first, uv.second);
        const double speed = cross_edge_speed(d, edge);
        if (!std::isfinite(speed) || speed <= kTiny)
            throw std::runtime_error(
                "NURBS patch edge has degenerate cross-edge surface speed");

        min_cross_speed = std::min(min_cross_speed, speed);
        max_cross_speed = std::max(max_cross_speed, speed);
        offset = std::min(offset, physical_buffer / speed);
    }

    const double span = (edge == PatchEdge::UMin || edge == PatchEdge::UMax)
        ? patch.domain_end_u() - patch.domain_start_u()
        : patch.domain_end_v() - patch.domain_start_v();
    const double capped_offset =
        std::min(offset, options.max_param_offset_fraction * span);

    return {capped_offset, length, sample_count, min_cross_speed, max_cross_speed};
}

Eigen::Vector3d point_on_patch(const NurbsSurfacePatch3D& patch,
                               const Eigen::Vector2d& uv)
{
    return patch.evaluate(uv.x(), uv.y());
}

double triangle_area(const Eigen::Vector3d& a,
                     const Eigen::Vector3d& b,
                     const Eigen::Vector3d& c)
{
    return 0.5 * (b - a).cross(c - a).norm();
}

double physical_distance(const NurbsSurfacePatch3D& patch,
                         const Eigen::Vector2d& a,
                         const Eigen::Vector2d& b)
{
    return (point_on_patch(patch, a) - point_on_patch(patch, b)).norm();
}

RectQuality rect_quality(const NurbsSurfacePatch3D& patch, const RectCell& cell)
{
    const Eigen::Vector2d p00(cell.u0, cell.v0);
    const Eigen::Vector2d p10(cell.u1, cell.v0);
    const Eigen::Vector2d p01(cell.u0, cell.v1);
    const Eigen::Vector2d p11(cell.u1, cell.v1);

    const double u_len = std::max(
        physical_distance(patch, p00, p10),
        physical_distance(patch, p01, p11));
    const double v_len = std::max(
        physical_distance(patch, p00, p01),
        physical_distance(patch, p10, p11));

    const double min_len = std::max(kTiny, std::min(u_len, v_len));
    const double cell_aspect_ratio = std::max(u_len, v_len) / min_len;

    return {std::max(u_len, v_len), cell_aspect_ratio, u_len >= v_len ? 0 : 1};
}

double max_trimmed_u_length(const NurbsSurfacePatch3D& patch,
                            double u0,
                            double u1,
                            double v0,
                            double v1)
{
    double length = 0.0;
    for (const double alpha : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        const double v = lerp(v0, v1, alpha);
        length = std::max(
            length,
            physical_distance(
                patch,
                Eigen::Vector2d(u0, v),
                Eigen::Vector2d(u1, v)));
    }
    return length;
}

double max_trimmed_v_length(const NurbsSurfacePatch3D& patch,
                            double u0,
                            double u1,
                            double v0,
                            double v1)
{
    double length = 0.0;
    for (const double alpha : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        const double u = lerp(u0, u1, alpha);
        length = std::max(
            length,
            physical_distance(
                patch,
                Eigen::Vector2d(u, v0),
                Eigen::Vector2d(u, v1)));
    }
    return length;
}

int subdivision_count_for_length(double length, double target_length)
{
    if (!std::isfinite(length) || length <= 0.0)
        return 1;
    return std::max(1, static_cast<int>(std::ceil(length / target_length)));
}

std::vector<RectCell> make_metric_seed_cells(const NurbsSurfacePatch3D& patch,
                                             double u0,
                                             double u1,
                                             double v0,
                                             double v1,
                                             double target_H)
{
    const int num_u = subdivision_count_for_length(
        max_trimmed_u_length(patch, u0, u1, v0, v1),
        target_H);
    const int num_v = subdivision_count_for_length(
        max_trimmed_v_length(patch, u0, u1, v0, v1),
        target_H);

    std::vector<RectCell> cells;
    cells.reserve(static_cast<std::size_t>(num_u * num_v));
    for (int i = 0; i < num_u; ++i) {
        const double ua = lerp(u0, u1, static_cast<double>(i) / num_u);
        const double ub = lerp(u0, u1, static_cast<double>(i + 1) / num_u);
        for (int j = 0; j < num_v; ++j) {
            const double va = lerp(v0, v1, static_cast<double>(j) / num_v);
            const double vb = lerp(v0, v1, static_cast<double>(j + 1) / num_v);
            cells.push_back({ua, ub, va, vb, 0});
        }
    }
    return cells;
}

std::vector<RectCell> split_cells_for_patch(
    const NurbsSurfacePatch3D& patch,
    const NurbsPatchEdgeOffsets3D& offsets,
    double h,
    const NurbsPatchTriangulatorOptions3D& options,
    NurbsPatchTriangulationSummary3D& summary)
{
    const double u0 = patch.domain_start_u() + offsets.u_min.knot_offset;
    const double u1 = patch.domain_end_u() - offsets.u_max.knot_offset;
    const double v0 = patch.domain_start_v() + offsets.v_min.knot_offset;
    const double v1 = patch.domain_end_v() - offsets.v_max.knot_offset;
    if (!(u0 < u1 && v0 < v1))
        throw std::runtime_error(
            "NURBS patch edge offsets leave no interior parameter domain");

    const double target_H = options.H_factor * h;
    std::vector<RectCell> stack =
        make_metric_seed_cells(patch, u0, u1, v0, v1, target_H);
    std::vector<RectCell> accepted;
    while (!stack.empty()) {
        const RectCell cell = stack.back();
        stack.pop_back();

        const RectQuality quality = rect_quality(patch, cell);
        summary.max_metric_ratio =
            std::max(summary.max_metric_ratio, quality.metric_ratio);
        summary.max_physical_edge_over_H = std::max(
            summary.max_physical_edge_over_H,
            quality.max_edge / target_H);

        const bool acceptable =
            quality.max_edge <= options.max_edge_over_H * target_H
            && quality.metric_ratio <= options.metric_ratio_tolerance;
        if (acceptable || cell.depth >= options.max_depth) {
            accepted.push_back(cell);
            continue;
        }

        ++summary.num_splits;
        if (quality.split_direction == 0) {
            const double um = 0.5 * (cell.u0 + cell.u1);
            stack.push_back({um, cell.u1, cell.v0, cell.v1, cell.depth + 1});
            stack.push_back({cell.u0, um, cell.v0, cell.v1, cell.depth + 1});
        } else {
            const double vm = 0.5 * (cell.v0 + cell.v1);
            stack.push_back({cell.u0, cell.u1, vm, cell.v1, cell.depth + 1});
            stack.push_back({cell.u0, cell.u1, cell.v0, vm, cell.depth + 1});
        }
    }
    summary.num_rectangles += static_cast<int>(accepted.size());
    return accepted;
}

void append_rect_triangles(int patch_index,
                           const RectCell& cell,
                           const PatchCornerCuts& cuts,
                           std::vector<NurbsParamTriangle3D>& triangles)
{
    const Eigen::Vector2d p00(cell.u0, cell.v0);
    const Eigen::Vector2d p10(cell.u1, cell.v0);
    const Eigen::Vector2d p01(cell.u0, cell.v1);
    const Eigen::Vector2d p11(cell.u1, cell.v1);

    std::vector<Eigen::Vector2d> polygon = {p00, p10, p11, p01};
    for (const ParamCornerCut& cut : cuts) {
        if (!cut.enabled || polygon.empty())
            continue;
        auto signed_value = [&](const Eigen::Vector2d& point) {
            const double xi = cut.u_sign * (point.x() - cut.corner.x()) / cut.du;
            const double eta = cut.v_sign * (point.y() - cut.corner.y()) / cut.dv;
            return xi + eta - 1.0;
        };
        std::vector<Eigen::Vector2d> clipped;
        for (std::size_t i = 0; i < polygon.size(); ++i) {
            const Eigen::Vector2d a = polygon[i];
            const Eigen::Vector2d b = polygon[(i + 1) % polygon.size()];
            const double fa = signed_value(a);
            const double fb = signed_value(b);
            const bool keep_a = fa >= -1.0e-14;
            const bool keep_b = fb >= -1.0e-14;
            if (keep_a)
                clipped.push_back(a);
            if (keep_a != keep_b) {
                const double t = fa / (fa - fb);
                clipped.push_back((1.0 - t) * a + t * b);
            }
        }
        polygon = std::move(clipped);
    }

    for (std::size_t i = 1; i + 1 < polygon.size(); ++i)
        triangles.push_back({patch_index,
                             {polygon[0], polygon[i], polygon[i + 1]}});
}

std::tuple<int, long long, long long, long long> node_key(
    int smooth_group,
    const Eigen::Vector3d& point)
{
    constexpr double scale = 1.0e12;
    return {smooth_group,
            static_cast<long long>(std::llround(point.x() * scale)),
            static_cast<long long>(std::llround(point.y() * scale)),
            static_cast<long long>(std::llround(point.z() * scale))};
}

int add_node(NodeStore& store,
              const std::vector<NurbsSurfacePatch3D>& patches,
              int patch_index,
              int smooth_group,
              const Eigen::Vector2d& uv)
{
    const NurbsSurfacePatch3D& patch = patches[static_cast<std::size_t>(patch_index)];
    const NurbsSurfaceDerivatives3D d =
        patch.evaluate_with_derivatives(uv.x(), uv.y());
    const auto key = node_key(smooth_group, d.point);
    const auto it = store.node_by_key.find(key);
    if (it != store.node_by_key.end())
        return it->second;

    Eigen::Vector3d normal = d.du.cross(d.dv);
    const double normal_norm = normal.norm();
    if (!std::isfinite(normal_norm) || normal_norm <= kTiny)
        throw std::runtime_error("NURBS patch has degenerate normal at a P2 node");
    normal /= normal_norm;

    const int index = static_cast<int>(store.points.size());
    store.node_by_key.emplace(key, index);
    store.points.push_back(d.point);
    store.normals.push_back(normal);
    store.weights.push_back(0.0);
    return index;
}

kfbim::Interface3D build_interface_from_triangles(
    const std::vector<NurbsSurfacePatch3D>& patches,
    const std::vector<NurbsParamTriangle3D>& triangles,
    const std::vector<int>& patch_smooth_group)
{
    NodeStore store;
    std::vector<std::array<int, 3>> panels;
    std::vector<std::array<int, 6>> panel_points;
    std::vector<int> panel_smooth_groups;
    panels.reserve(triangles.size());
    panel_points.reserve(triangles.size());
    panel_smooth_groups.reserve(triangles.size());

    for (const NurbsParamTriangle3D& tri : triangles) {
        const int smooth_group =
            patch_smooth_group[static_cast<std::size_t>(tri.patch_index)];
        const Eigen::Vector2d uv0 = tri.uv[0];
        const Eigen::Vector2d uv1 = tri.uv[1];
        const Eigen::Vector2d uv2 = tri.uv[2];
        const Eigen::Vector2d uv01 = 0.5 * (uv0 + uv1);
        const Eigen::Vector2d uv12 = 0.5 * (uv1 + uv2);
        const Eigen::Vector2d uv20 = 0.5 * (uv2 + uv0);

        const int p0 = add_node(store, patches, tri.patch_index, smooth_group, uv0);
        const int p1 = add_node(store, patches, tri.patch_index, smooth_group, uv1);
        const int p2 = add_node(store, patches, tri.patch_index, smooth_group, uv2);
        const int p01 = add_node(store, patches, tri.patch_index, smooth_group, uv01);
        const int p12 = add_node(store, patches, tri.patch_index, smooth_group, uv12);
        const int p20 = add_node(store, patches, tri.patch_index, smooth_group, uv20);

        panels.push_back({p0, p1, p2});
        panel_points.push_back({p0, p1, p2, p01, p12, p20});
        panel_smooth_groups.push_back(smooth_group);

        const double area = triangle_area(
            store.points[static_cast<std::size_t>(p0)],
            store.points[static_cast<std::size_t>(p1)],
            store.points[static_cast<std::size_t>(p2)]);
        const double weight = area / 6.0;
        for (const int idx : {p0, p1, p2, p01, p12, p20})
            store.weights[static_cast<std::size_t>(idx)] += weight;
    }

    const int n_points = static_cast<int>(store.points.size());
    const int n_panels = static_cast<int>(panels.size());

    Eigen::MatrixX3d points(n_points, 3);
    Eigen::MatrixX3d normals(n_points, 3);
    Eigen::VectorXd weights(n_points);
    for (int i = 0; i < n_points; ++i) {
        points.row(i) = store.points[static_cast<std::size_t>(i)].transpose();
        normals.row(i) = store.normals[static_cast<std::size_t>(i)].transpose();
        weights[i] = store.weights[static_cast<std::size_t>(i)];
    }

    Eigen::MatrixX3d vertices = points;
    Eigen::MatrixX3i panel_matrix(n_panels, 3);
    Eigen::MatrixXi panel_point_indices(n_panels, 6);
    Eigen::VectorXi smooth_group_vector(n_panels);
    for (int p = 0; p < n_panels; ++p) {
        for (int q = 0; q < 3; ++q)
            panel_matrix(p, q) = panels[static_cast<std::size_t>(p)]
                                       [static_cast<std::size_t>(q)];
        for (int q = 0; q < 6; ++q)
            panel_point_indices(p, q) =
                panel_points[static_cast<std::size_t>(p)]
                            [static_cast<std::size_t>(q)];
        smooth_group_vector[p] =
            panel_smooth_groups[static_cast<std::size_t>(p)];
    }

    return kfbim::Interface3D(vertices,
                              panel_matrix,
                              points,
                              normals,
                              weights,
                              6,
                              panel_point_indices,
                              Eigen::VectorXi::Zero(n_panels),
                              smooth_group_vector,
                              kfbim::PanelNodeLayout3D::QuadraticLagrange);
}

} // namespace

NurbsPatchEdgeOffsets3D compute_nurbs_patch_edge_offsets_3d(
    const NurbsSurfacePatch3D& patch,
    double h,
    const NurbsPatchTriangulatorOptions3D& options)
{
    validate_options(h, options);
    return {
        compute_one_edge_offset(patch, PatchEdge::UMin, h, options),
        compute_one_edge_offset(patch, PatchEdge::UMax, h, options),
        compute_one_edge_offset(patch, PatchEdge::VMin, h, options),
        compute_one_edge_offset(patch, PatchEdge::VMax, h, options)
    };
}

NurbsPatchTriangulation3D triangulate_nurbs_surface_patches_3d(
    const std::vector<NurbsSurfacePatch3D>& patches,
    double h,
    const NurbsPatchTriangulatorOptions3D& options)
{
    validate_options(h, options);
    if (patches.empty())
        throw std::invalid_argument("at least one NURBS surface patch is required");

    NurbsPatchTriangulationSummary3D summary;
    summary.num_patches = static_cast<int>(patches.size());
    summary.target_H = options.H_factor * h;

    std::vector<NurbsPatchEdgeOffsets3D> offsets;
    offsets.reserve(patches.size());
    for (const NurbsSurfacePatch3D& patch : patches)
        offsets.push_back(compute_nurbs_patch_edge_offsets_3d(patch, h, options));
    const std::vector<NurbsPatchEdgeOffsets3D> fully_buffered_offsets = offsets;

    const std::array<PatchEdge, 4> patch_edges = {
        PatchEdge::UMin, PatchEdge::UMax, PatchEdge::VMin, PatchEdge::VMax};
    std::vector<PatchEdgeDescriptor> descriptors;
    descriptors.reserve(4 * patches.size());
    for (int patch_index = 0; patch_index < static_cast<int>(patches.size());
         ++patch_index) {
        for (const PatchEdge edge : patch_edges) {
            descriptors.push_back(describe_patch_edge(
                patches[static_cast<std::size_t>(patch_index)],
                patch_index,
                edge));
        }
    }

    PatchDisjointSet smooth_groups(static_cast<int>(patches.size()));
    std::vector<bool> matched(descriptors.size(), false);
    std::vector<NurbsPatchGeometricEdge3D> feature_edges;
    for (std::size_t i = 0; i < descriptors.size(); ++i) {
        if (matched[i])
            continue;
        const PatchEdgeDescriptor& a = descriptors[i];
        int match = -1;
        for (std::size_t j = i + 1; j < descriptors.size(); ++j) {
            if (matched[j] || descriptors[j].patch == a.patch)
                continue;
            if (edge_descriptors_match(a,
                                       descriptors[j],
                                       options.edge_match_tolerance)) {
                match = static_cast<int>(j);
                break;
            }
        }

        matched[i] = true;
        if (match < 0) {
            feature_edges.push_back(
                {a.start,
                 a.end,
                 {{a.patch, -1}},
                 {{edge_offset(offsets[static_cast<std::size_t>(a.patch)],
                                a.edge).sample_count,
                   0}}});
            continue;
        }

        matched[static_cast<std::size_t>(match)] = true;
        const PatchEdgeDescriptor& b =
            descriptors[static_cast<std::size_t>(match)];
        const bool smooth = options.detect_smooth_seams
                         && a.normal.dot(b.normal)
                                >= options.smooth_normal_cosine;
        if (smooth) {
            edge_offset(offsets[static_cast<std::size_t>(a.patch)], a.edge)
                .knot_offset = 0.0;
            edge_offset(offsets[static_cast<std::size_t>(b.patch)], b.edge)
                .knot_offset = 0.0;
            smooth_groups.unite(a.patch, b.patch);
            ++summary.num_smooth_seams;
        } else {
            feature_edges.push_back(
                {a.start,
                 a.end,
                 {{a.patch, b.patch}},
                 {{edge_offset(offsets[static_cast<std::size_t>(a.patch)],
                                a.edge).sample_count,
                   edge_offset(offsets[static_cast<std::size_t>(b.patch)],
                                b.edge).sample_count}}});
        }
    }

    std::vector<int> patch_smooth_group(patches.size(), 0);
    for (int patch_index = 0; patch_index < static_cast<int>(patches.size());
         ++patch_index) {
        patch_smooth_group[static_cast<std::size_t>(patch_index)] =
            smooth_groups.find(patch_index);
    }
    std::vector<Eigen::Vector3d> feature_vertices =
        unique_feature_vertices(descriptors,
                                options.edge_match_tolerance,
                                options.smooth_normal_cosine);
    summary.num_feature_edges = static_cast<int>(feature_edges.size());
    summary.num_feature_vertices = static_cast<int>(feature_vertices.size());

    std::vector<EdgePatchSource3D> edge_patch_sources;
    edge_patch_sources.reserve(feature_edges.size());
    for (const NurbsPatchGeometricEdge3D& edge : feature_edges) {
        edge_patch_sources.push_back(
            {edge.start, edge.end, edge.incident_patches});
    }
    std::vector<EdgePatch3D> edge_patches =
        build_edge_patches_3d(edge_patch_sources, options.edge_patch);

    std::vector<PatchCornerCuts> corner_cuts(patches.size());
    for (int patch_index = 0; patch_index < static_cast<int>(patches.size());
         ++patch_index) {
        const NurbsSurfacePatch3D& patch =
            patches[static_cast<std::size_t>(patch_index)];
        const double u0 = patch.domain_start_u();
        const double u1 = patch.domain_end_u();
        const double v0 = patch.domain_start_v();
        const double v1 = patch.domain_end_v();
        const std::array<Eigen::Vector2d, 4> uv_corners = {
            Eigen::Vector2d(u0, v0), Eigen::Vector2d(u1, v0),
            Eigen::Vector2d(u1, v1), Eigen::Vector2d(u0, v1)};
        const std::array<bool, 4> adjacent_offsets_are_zero = {
            offsets[static_cast<std::size_t>(patch_index)].u_min.knot_offset == 0.0
                && offsets[static_cast<std::size_t>(patch_index)].v_min.knot_offset == 0.0,
            offsets[static_cast<std::size_t>(patch_index)].u_max.knot_offset == 0.0
                && offsets[static_cast<std::size_t>(patch_index)].v_min.knot_offset == 0.0,
            offsets[static_cast<std::size_t>(patch_index)].u_max.knot_offset == 0.0
                && offsets[static_cast<std::size_t>(patch_index)].v_max.knot_offset == 0.0,
            offsets[static_cast<std::size_t>(patch_index)].u_min.knot_offset == 0.0
                && offsets[static_cast<std::size_t>(patch_index)].v_max.knot_offset == 0.0};
        const std::array<double, 4> du = {
            fully_buffered_offsets[static_cast<std::size_t>(patch_index)].u_min.knot_offset,
            fully_buffered_offsets[static_cast<std::size_t>(patch_index)].u_max.knot_offset,
            fully_buffered_offsets[static_cast<std::size_t>(patch_index)].u_max.knot_offset,
            fully_buffered_offsets[static_cast<std::size_t>(patch_index)].u_min.knot_offset};
        const std::array<double, 4> dv = {
            fully_buffered_offsets[static_cast<std::size_t>(patch_index)].v_min.knot_offset,
            fully_buffered_offsets[static_cast<std::size_t>(patch_index)].v_min.knot_offset,
            fully_buffered_offsets[static_cast<std::size_t>(patch_index)].v_max.knot_offset,
            fully_buffered_offsets[static_cast<std::size_t>(patch_index)].v_max.knot_offset};
        const std::array<int, 4> u_sign = {{1, -1, -1, 1}};
        const std::array<int, 4> v_sign = {{1, 1, -1, -1}};
        for (int corner = 0; corner < 4; ++corner) {
            if (!adjacent_offsets_are_zero[static_cast<std::size_t>(corner)])
                continue;
            const Eigen::Vector3d physical = patch.evaluate(
                uv_corners[static_cast<std::size_t>(corner)].x(),
                uv_corners[static_cast<std::size_t>(corner)].y());
            const bool is_feature_vertex = std::any_of(
                feature_vertices.begin(), feature_vertices.end(),
                [&](const Eigen::Vector3d& feature) {
                    return (feature - physical).norm()
                        <= options.edge_match_tolerance;
                });
            if (is_feature_vertex) {
                corner_cuts[static_cast<std::size_t>(patch_index)]
                           [static_cast<std::size_t>(corner)] =
                    {true,
                     uv_corners[static_cast<std::size_t>(corner)],
                     du[static_cast<std::size_t>(corner)],
                     dv[static_cast<std::size_t>(corner)],
                     u_sign[static_cast<std::size_t>(corner)],
                     v_sign[static_cast<std::size_t>(corner)]};
            }
        }
    }

    std::vector<NurbsParamTriangle3D> triangles;
    for (int patch_index = 0; patch_index < static_cast<int>(patches.size());
         ++patch_index) {
        const std::vector<RectCell> cells = split_cells_for_patch(
            patches[static_cast<std::size_t>(patch_index)],
            offsets[static_cast<std::size_t>(patch_index)],
            h,
            options,
            summary);
        for (const RectCell& cell : cells)
            append_rect_triangles(
                patch_index,
                cell,
                corner_cuts[static_cast<std::size_t>(patch_index)],
                triangles);
    }

    summary.num_triangles = static_cast<int>(triangles.size());
    kfbim::Interface3D interface =
        build_interface_from_triangles(patches, triangles, patch_smooth_group);
    summary.num_points = interface.num_points();

    // Build a second P2 tessellation on the complete NURBS patch domains.
    // This mesh is geometry-only: it closes feature-edge gaps for labeling
    // and segment/surface intersections, while the interface above retains
    // the requested edge buffer for its iteration DOFs and Cauchy patches.
    std::vector<NurbsPatchEdgeOffsets3D> geometry_offsets = offsets;
    for (NurbsPatchEdgeOffsets3D& patch_offsets : geometry_offsets) {
        patch_offsets.u_min.knot_offset = 0.0;
        patch_offsets.u_max.knot_offset = 0.0;
        patch_offsets.v_min.knot_offset = 0.0;
        patch_offsets.v_max.knot_offset = 0.0;
    }
    NurbsPatchTriangulationSummary3D geometry_summary;
    geometry_summary.target_H = options.H_factor * h;
    std::vector<NurbsParamTriangle3D> geometry_triangles;
    const PatchCornerCuts no_corner_cuts{};
    for (int patch_index = 0; patch_index < static_cast<int>(patches.size());
         ++patch_index) {
        const std::vector<RectCell> cells = split_cells_for_patch(
            patches[static_cast<std::size_t>(patch_index)],
            geometry_offsets[static_cast<std::size_t>(patch_index)],
            h,
            options,
            geometry_summary);
        for (const RectCell& cell : cells) {
            append_rect_triangles(
                patch_index, cell, no_corner_cuts, geometry_triangles);
        }
    }
    kfbim::Interface3D geometry_interface = build_interface_from_triangles(
        patches, geometry_triangles, patch_smooth_group);

    return {std::move(interface),
            std::move(triangles),
            std::move(geometry_interface),
            std::move(geometry_triangles),
            std::move(offsets),
            std::move(feature_edges),
            std::move(edge_patches),
            std::move(feature_vertices),
            summary};
}

} // namespace kfbim::geometry3d
