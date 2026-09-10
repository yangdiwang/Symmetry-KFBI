#pragma once

#include "src/geometry/industrial_nurbs_models_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace industrial_example {
using namespace kfbim::geometry3d;

struct Diagnostics {
    int patch_count = 0, connection_count = 0, smooth_connection_count = 0;
    int euler_characteristic = 0, connected_components = 0;
    double min_jacobian = std::numeric_limits<double>::infinity();
    double max_seam_gap = 0, volume = 0, area = 0, normal_flux = 0;
    double volume_quadrature_difference = 0, relative_volume_error = 0;
};

inline Eigen::Vector3d edge_point(const NurbsSurfacePatch3D& p, NurbsPatchEdge3D e, double t)
{
    switch (e) {
    case NurbsPatchEdge3D::UMin: return p.evaluate(0, t);
    case NurbsPatchEdge3D::UMax: return p.evaluate(1, t);
    case NurbsPatchEdge3D::VMin: return p.evaluate(t, 0);
    case NurbsPatchEdge3D::VMax: return p.evaluate(t, 1);
    }
    throw std::logic_error("invalid edge");
}

inline std::vector<std::pair<double, double>> gauss_rule(int n)
{
    std::vector<std::pair<double, double>> rule;
    for (int i = 0; i < n; ++i) {
        double x = std::cos(std::acos(-1.0) * (i + 0.75) / (n + 0.5)), dp = 0;
        for (int iteration = 0; iteration < 30; ++iteration) {
            double p0 = 1, p1 = x;
            for (int k = 2; k <= n; ++k) {
                double next = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k;
                p0 = p1; p1 = next;
            }
            dp = n * (x * p1 - p0) / (x * x - 1);
            const double step = p1 / dp;
            x -= step;
            if (std::abs(step) < 2e-15) break;
        }
        rule.emplace_back((x + 1) / 2, 1 / ((1 - x * x) * dp * dp));
    }
    return rule;
}

inline std::array<double, 3> integrate(const IndustrialNurbsModel3D& model, int order)
{
    const auto rule = gauss_rule(order);
    double area = 0, volume = 0;
    Eigen::Vector3d flux = Eigen::Vector3d::Zero();
    for (const auto& p : model.patches) for (const auto& u : rule) for (const auto& v : rule) {
        const auto d = p.evaluate_with_derivatives(u.first, v.first);
        const Eigen::Vector3d n = d.du.cross(d.dv);
        const double w = u.second * v.second;
        area += n.norm() * w;
        volume += d.point.dot(n) * w / 3;
        flux += w * n;
    }
    return {{area, volume, flux.norm()}};
}

struct DisjointSet {
    std::vector<int> parent;
    explicit DisjointSet(int n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
    int find(int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); }
    void join(int x, int y) { parent[find(x)] = find(y); }
    int count() { int n = 0; for (int i = 0; i < static_cast<int>(parent.size()); ++i) n += find(i) == i; return n; }
};

inline Diagnostics inspect(const IndustrialNurbsModel3D& model)
{
    model.geometry_model().validate_closed();
    Diagnostics result;
    result.patch_count = static_cast<int>(model.patches.size());
    result.connection_count = static_cast<int>(model.connections.size());
    DisjointSet vertices(4 * result.patch_count), components(result.patch_count);
    const std::array<std::array<int, 2>, 4> corners{{{{0,1}}, {{2,3}}, {{0,2}}, {{1,3}}}};
    for (const auto& c : model.connections) {
        const auto a = corners[static_cast<int>(c.first.edge)];
        const auto b = corners[static_cast<int>(c.second.edge)];
        for (int end = 0; end < 2; ++end)
            vertices.join(4 * c.first.patch + a[end], 4 * c.second.patch + b[c.reversed ? 1-end : end]);
        components.join(c.first.patch, c.second.patch);
        result.smooth_connection_count += c.g1;
        for (int i = 0; i <= 64; ++i) {
            const double t = i / 64.0;
            const auto p = edge_point(model.patches[c.first.patch], c.first.edge, t);
            const auto q = edge_point(model.patches[c.second.patch], c.second.edge, c.reversed ? 1-t : t);
            result.max_seam_gap = std::max(result.max_seam_gap, (p-q).norm());
        }
    }
    result.connected_components = components.count();
    result.euler_characteristic = vertices.count() - result.connection_count + result.patch_count;
    for (const auto& p : model.patches) {
        const auto center = p.evaluate_with_derivatives(0.5, 0.5);
        const Eigen::Vector3d center_normal = center.du.cross(center.dv);
        bool planar = center_normal.norm() > 1e-12;
        for (const auto& row : p.control_net()) for (const auto& point : row)
            planar = planar && std::abs((point - center.point).dot(center_normal))
                < 1e-12 * center_normal.norm();
        for (int i = 0; i <= 40; ++i) for (int j = 0; j <= 40; ++j) {
            const auto d = p.evaluate_with_derivatives(i / 40.0, j / 40.0);
            const Eigen::Vector3d normal = d.du.cross(d.dv);
            if (!d.point.allFinite() || !normal.allFinite()) throw std::runtime_error("nonfinite surface sample");
            result.min_jacobian = std::min(result.min_jacobian, normal.norm());
            if (planar && normal.dot(center_normal) <= 0)
                throw std::runtime_error("folded planar cap in " + model.name);
        }
    }
    const auto fine = integrate(model, 24), coarse = integrate(model, 16);
    result.area = fine[0]; result.volume = fine[1]; result.normal_flux = fine[2];
    result.volume_quadrature_difference = std::abs(fine[1] - coarse[1]);
    if (model.expected_volume > 0)
        result.relative_volume_error = std::abs(fine[1] - model.expected_volume) / model.expected_volume;
    if (result.connected_components != 1 || result.euler_characteristic != 2 - 2 * model.expected_genus)
        throw std::runtime_error("incorrect topology in " + model.name);
    if (!(result.min_jacobian > 1e-8) || !(result.volume > 0) || result.max_seam_gap > 1e-9
        || result.normal_flux > 1e-9 * result.area || result.volume_quadrature_difference > 1e-9 * result.volume
        || result.relative_volume_error > 1e-9)
        throw std::runtime_error("geometric diagnostics failed in " + model.name);
    return result;
}
} // namespace industrial_example
