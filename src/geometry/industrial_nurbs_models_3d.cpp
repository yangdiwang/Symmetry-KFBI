#include "industrial_nurbs_models_3d.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace kfbim::geometry3d {

NurbsSurfaceModel3D IndustrialNurbsModel3D::geometry_model() const
{
    return {patches, std::vector<int>(patches.size(), 0), connections};
}

namespace {
using Edge = NurbsPatchEdge3D;
constexpr std::array<Edge, 4> edges{{Edge::UMin, Edge::UMax, Edge::VMin, Edge::VMax}};

NurbsSurfaceDerivatives3D on_edge(const NurbsSurfacePatch3D& p, Edge edge, double t)
{
    switch (edge) {
    case Edge::UMin: return p.evaluate_with_derivatives(0, t);
    case Edge::UMax: return p.evaluate_with_derivatives(1, t);
    case Edge::VMin: return p.evaluate_with_derivatives(t, 0);
    case Edge::VMax: return p.evaluate_with_derivatives(t, 1);
    }
    throw std::logic_error("invalid edge");
}
}

IndustrialNurbsModel3D finalize_industrial_model(
    std::string name, std::string description, std::vector<NurbsSurfacePatch3D> patches,
    std::vector<std::string> names, double volume, int genus)
{
    if (patches.empty() || patches.size() != names.size())
        throw std::invalid_argument("industrial model needs named patches");
    const int count = static_cast<int>(patches.size()) * 4;
    std::vector<std::array<Eigen::Vector3d, 17>> samples(count);
    std::vector<bool> used(count, false);
    double scale = 1;
    for (int i = 0; i < count; ++i) for (int k = 0; k <= 16; ++k) {
        samples[i][k] = on_edge(patches[i / 4], edges[i % 4], k / 16.0).point;
        scale = std::max(scale, samples[i][k].norm());
    }
    const double tolerance = 1e-10 * scale;
    std::vector<NurbsPatchEdgeConnection3D> connections;
    for (int i = 0; i < count; ++i) {
        if (used[i]) continue;
        int mate = -1;
        bool reverse = false;
        for (int j = i + 1; j < count; ++j) {
            if (used[j] || i / 4 == j / 4) continue;
            bool same = true, rev = true;
            for (int k = 0; k <= 16 && (same || rev); ++k) {
                same = same && (samples[i][k] - samples[j][k]).norm() <= tolerance;
                rev = rev && (samples[i][k] - samples[j][16 - k]).norm() <= tolerance;
            }
            if (same || rev) {
                if (mate >= 0) throw std::runtime_error(name + ": ambiguous edge mate");
                mate = j;
                reverse = !same;
            }
        }
        if (mate < 0) throw std::runtime_error(name + ": unpaired edge on " + names[i / 4]
            + " edge " + std::to_string(i % 4));
        bool smooth = true;
        for (int k = 0; k <= 16; ++k) {
            const double t = k / 16.0;
            const auto a = on_edge(patches[i / 4], edges[i % 4], t);
            const auto b = on_edge(patches[mate / 4], edges[mate % 4], reverse ? 1 - t : t);
            const auto na = a.du.cross(a.dv), nb = b.du.cross(b.dv);
            smooth = smooth && na.norm() > 1e-12 && nb.norm() > 1e-12
                && na.normalized().dot(nb.normalized()) > 1 - 1e-9;
        }
        connections.push_back({{i / 4, edges[i % 4], 0, 1},
                               {mate / 4, edges[mate % 4], 0, 1}, reverse, smooth});
        used[i] = used[mate] = true;
    }
    IndustrialNurbsModel3D result{std::move(name), std::move(description), std::move(patches),
        std::move(names), std::move(connections), volume, genus};
    result.geometry_model().validate_closed();
    return result;
}

} // namespace kfbim::geometry3d
