#include "src/support/geometry/native_nurbs_surface_3d.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace kfbim::app3d {

namespace {

using geometry3d::NurbsPatchEdgeConnection3D;
using geometry3d::NurbsPatchEdgeInterval3D;
using geometry3d::NurbsSurfacePatch3D;

int edge_index(PatchEdge3D edge)
{
    return static_cast<int>(edge);
}

bool is_u_edge(PatchEdge3D edge)
{
    return edge == PatchEdge3D::UMin || edge == PatchEdge3D::UMax;
}

std::pair<double, double> edge_parameter_domain(
    const NativeNurbsSurface3D& surface,
    int patch,
    PatchEdge3D edge)
{
    if (patch < 0 || patch >= static_cast<int>(surface.patches.size()))
        throw std::out_of_range("smooth edge has an invalid patch");
    const NurbsSurfacePatch3D& geometry =
        surface.patches[static_cast<std::size_t>(patch)];
    return is_u_edge(edge)
        ? std::pair<double, double>{
              geometry.domain_start_v(), geometry.domain_end_v()}
        : std::pair<double, double>{
              geometry.domain_start_u(), geometry.domain_end_u()};
}

bool interval_is_full_edge(const NativeNurbsSurface3D& surface,
                           const NurbsPatchEdgeInterval3D& interval)
{
    const auto domain = edge_parameter_domain(
        surface, interval.patch, interval.edge);
    const double scale = std::max(1.0, domain.second - domain.first);
    const double tolerance = 2.0e-12 * scale;
    return std::abs(interval.begin - domain.first) <= tolerance
        && std::abs(interval.end - domain.second) <= tolerance;
}

struct SmoothEdgeTransition3D {
    int patch = -1;
    PatchEdge3D edge = PatchEdge3D::UMin;
    double source_begin = 0.0;
    double source_end = 1.0;
    double destination_begin = 0.0;
    double destination_end = 1.0;
    bool reversed = false;
};

std::vector<SmoothEdgeTransition3D> smooth_edge_transitions(
    const NativeNurbsSurface3D& surface,
    int source_patch,
    PatchEdge3D source_edge)
{
    if (source_patch < 0
        || source_patch >= static_cast<int>(surface.patches.size())) {
        throw std::out_of_range("smooth edge transition has an invalid patch");
    }
    if (!surface.smooth_neighbors.empty()
        && surface.smooth_neighbors.size() != surface.patches.size()) {
        throw std::invalid_argument(
            "legacy smooth-neighbor metadata has an invalid size");
    }

    std::vector<SmoothEdgeTransition3D> result;
    for (const NurbsPatchEdgeConnection3D& connection :
         surface.geometric_connections) {
        if (!connection.g1)
            continue;
        const bool source_is_first =
            connection.first.patch == source_patch
            && connection.first.edge == source_edge;
        const bool source_is_second =
            connection.second.patch == source_patch
            && connection.second.edge == source_edge;
        if (!source_is_first && !source_is_second)
            continue;
        const NurbsPatchEdgeInterval3D& source = source_is_first
            ? connection.first : connection.second;
        const NurbsPatchEdgeInterval3D& destination = source_is_first
            ? connection.second : connection.first;
        if (!(source.end > source.begin)
            || !(destination.end > destination.begin)) {
            throw std::invalid_argument(
                "smooth geometric connection has a degenerate interval");
        }
        result.push_back({destination.patch,
                          destination.edge,
                          source.begin,
                          source.end,
                          destination.begin,
                          destination.end,
                          connection.reversed});
    }

    // Old unit fixtures sometimes provide only the fixed-size neighbor
    // table.  Use it only when no interval-aware connection exists on this
    // edge, so a long edge may own several geometric children without being
    // shadowed by a legacy slot.
    if (result.empty() && !surface.smooth_neighbors.empty()) {
        const auto& legacy = surface.smooth_neighbors[
            static_cast<std::size_t>(source_patch)]
            [static_cast<std::size_t>(edge_index(source_edge))];
        if (legacy) {
            const auto source_domain = edge_parameter_domain(
                surface, source_patch, source_edge);
            const auto destination_domain = edge_parameter_domain(
                surface, legacy->patch, legacy->edge);
            result.push_back({legacy->patch,
                              legacy->edge,
                              source_domain.first,
                              source_domain.second,
                              destination_domain.first,
                              destination_domain.second,
                              legacy->reversed});
        }
    }
    std::sort(result.begin(), result.end(),
              [](const SmoothEdgeTransition3D& a,
                 const SmoothEdgeTransition3D& b) {
                  return std::tie(a.source_begin, a.source_end, a.patch,
                                  a.edge, a.destination_begin,
                                  a.destination_end, a.reversed)
                       < std::tie(b.source_begin, b.source_end, b.patch,
                                  b.edge, b.destination_begin,
                                  b.destination_end, b.reversed);
              });
    return result;
}

} // namespace

namespace {
double sampled_isoline_length(const NurbsSurfacePatch3D& patch,
                              bool varying_u,
                              double fixed_fraction)
{
    constexpr int segments = 16;
    const double u0 = patch.domain_start_u();
    const double u1 = patch.domain_end_u();
    const double v0 = patch.domain_start_v();
    const double v1 = patch.domain_end_v();
    auto point = [&](double fraction) {
        const double u_fraction = varying_u ? fraction : fixed_fraction;
        const double v_fraction = varying_u ? fixed_fraction : fraction;
        return patch.evaluate(
            u0 + u_fraction * (u1 - u0),
            v0 + v_fraction * (v1 - v0));
    };
    Eigen::Vector3d previous = point(0.0);
    double length = 0.0;
    for (int segment = 1; segment <= segments; ++segment) {
        const Eigen::Vector3d current =
            point(static_cast<double>(segment) / segments);
        length += (current - previous).norm();
        previous = current;
    }
    return length;
}

double max_sampled_direction_length(const NurbsSurfacePatch3D& patch,
                                    bool varying_u)
{
    double result = 0.0;
    for (int transverse = 0; transverse < 5; ++transverse) {
        result = std::max(
            result,
            sampled_isoline_length(
                patch, varying_u, 0.25 * static_cast<double>(transverse)));
    }
    return result;
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> tangent_frame(
    const Eigen::Vector3d& du,
    const Eigen::Vector3d& normal)
{
    Eigen::Vector3d tangent1 = du - du.dot(normal) * normal;
    if (tangent1.norm() <= 1.0e-14) {
        const Eigen::Vector3d axis = std::abs(normal.x()) < 0.8
            ? Eigen::Vector3d::UnitX()
            : Eigen::Vector3d::UnitY();
        tangent1 = axis - axis.dot(normal) * normal;
    }
    tangent1.normalize();
    Eigen::Vector3d tangent2 = normal.cross(tangent1).normalized();
    return {tangent1, tangent2};
}

} // namespace

std::vector<int> smooth_patch_neighbors_3d(
    const NativeNurbsSurface3D& surface,
    int patch)
{
    if (patch < 0 || patch >= static_cast<int>(surface.patches.size()))
        throw std::out_of_range("invalid patch for smooth-neighbor query");
    std::vector<int> result;
    for (int edge_value = 0; edge_value < 4; ++edge_value) {
        const auto transitions = smooth_edge_transitions(
            surface, patch, static_cast<PatchEdge3D>(edge_value));
        for (const SmoothEdgeTransition3D& transition : transitions) {
            if (transition.patch < 0
                || transition.patch
                       >= static_cast<int>(surface.patches.size())) {
                throw std::runtime_error(
                    "smooth topology contains an invalid neighboring patch");
            }
            result.push_back(transition.patch);
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

SurfaceDofCloud3D make_native_surface_dofs_3d(
    const NativeNurbsSurface3D& surface,
    double h)
{
    if (!std::isfinite(h) || h <= 0.0)
        throw std::invalid_argument("native surface DOFs require positive h");
    if (surface.patches.empty()
        || surface.patch_names.size() != surface.patches.size()
        || (!surface.smooth_neighbors.empty()
            && surface.smooth_neighbors.size() != surface.patches.size())
        || surface.topological_patch_neighbors.size()
               != surface.patches.size()) {
        throw std::invalid_argument("native surface metadata is inconsistent");
    }

    SurfaceDofCloud3D cloud;
    cloud.expected_area = surface.expected_area;
    cloud.patches.reserve(surface.patches.size());
    std::vector<double> direction_length_u(surface.patches.size(), 0.0);
    std::vector<double> direction_length_v(surface.patches.size(), 0.0);
    for (int patch_id = 0;
         patch_id < static_cast<int>(surface.patches.size());
         ++patch_id) {
        const NurbsSurfacePatch3D& patch =
            surface.patches[static_cast<std::size_t>(patch_id)];
        SurfaceDofPatch3D tensor_patch;
        tensor_patch.name = surface.patch_names[static_cast<std::size_t>(patch_id)];
        direction_length_u[static_cast<std::size_t>(patch_id)] =
            max_sampled_direction_length(patch, true);
        direction_length_v[static_cast<std::size_t>(patch_id)] =
            max_sampled_direction_length(patch, false);
        tensor_patch.nu = std::max(
            2,
            static_cast<int>(std::ceil(
                direction_length_u[static_cast<std::size_t>(patch_id)] / h)));
        tensor_patch.nv = std::max(
            2,
            static_cast<int>(std::ceil(
                direction_length_v[static_cast<std::size_t>(patch_id)] / h)));
        tensor_patch.smooth_patch_ids =
            smooth_patch_component(surface, patch_id);
        cloud.patches.push_back(std::move(tensor_patch));
    }

    // A glued 2x2 tensor neighborhood must preserve distinct along-edge rows.
    // Propagate the larger count across every smooth edge until all coupled
    // patch directions agree. The parameter locations remain uniformly spaced
    // inside every native patch.
    auto harmonize_smooth_edge_counts = [&]() {
        bool changed = true;
        while (changed) {
            changed = false;
            const auto harmonize = [&](int patch_a,
                                       PatchEdge3D edge_a,
                                       int patch_b,
                                       PatchEdge3D edge_b) {
                SurfaceDofPatch3D& a =
                    cloud.patches[static_cast<std::size_t>(patch_a)];
                SurfaceDofPatch3D& b =
                    cloud.patches[static_cast<std::size_t>(patch_b)];
                int& count_a = is_u_edge(edge_a) ? a.nv : a.nu;
                int& count_b = is_u_edge(edge_b) ? b.nv : b.nu;
                const int shared = std::max(count_a, count_b);
                if (count_a != shared || count_b != shared) {
                    count_a = shared;
                    count_b = shared;
                    changed = true;
                }
            };

            // Only full/full joins require identical tensor row counts.
            // At a smooth T, a long-edge interval and a short full edge have
            // different total counts; interval-aware crossing below maps the
            // physical sub-interval instead of inflating the short patch.
            for (const NurbsPatchEdgeConnection3D& connection :
                 surface.geometric_connections) {
                if (connection.g1
                    && interval_is_full_edge(surface, connection.first)
                    && interval_is_full_edge(surface, connection.second)) {
                    harmonize(connection.first.patch, connection.first.edge,
                              connection.second.patch,
                              connection.second.edge);
                }
            }

            // Preserve old synthetic fixtures that carry only the legacy
            // fixed-size table and no geometric connection list.
            if (surface.geometric_connections.empty()
                && !surface.smooth_neighbors.empty()) {
                for (int patch_id = 0;
                     patch_id < static_cast<int>(cloud.patches.size());
                     ++patch_id) {
                    for (int edge_value = 0; edge_value < 4; ++edge_value) {
                        const auto& connection = surface.smooth_neighbors[
                            static_cast<std::size_t>(patch_id)]
                            [static_cast<std::size_t>(edge_value)];
                        if (connection) {
                            harmonize(
                                patch_id,
                                static_cast<PatchEdge3D>(edge_value),
                                connection->patch, connection->edge);
                        }
                    }
                }
            }
        }
    };
    harmonize_smooth_edge_counts();

    for (int patch_id = 0;
         patch_id < static_cast<int>(surface.patches.size());
         ++patch_id) {
        const NurbsSurfacePatch3D& patch =
            surface.patches[static_cast<std::size_t>(patch_id)];
        SurfaceDofPatch3D& tensor_patch =
            cloud.patches[static_cast<std::size_t>(patch_id)];
        tensor_patch.first_dof = static_cast<int>(cloud.dofs.size());
        const double u0 = patch.domain_start_u();
        const double u1 = patch.domain_end_u();
        const double v0 = patch.domain_start_v();
        const double v1 = patch.domain_end_v();
        const double du_parameter =
            (u1 - u0) / static_cast<double>(tensor_patch.nu);
        const double dv_parameter =
            (v1 - v0) / static_cast<double>(tensor_patch.nv);
        for (int i = 0; i < tensor_patch.nu; ++i) {
            const double u = u0 + (static_cast<double>(i) + 0.5) * du_parameter;
            for (int j = 0; j < tensor_patch.nv; ++j) {
                const double v =
                    v0 + (static_cast<double>(j) + 0.5) * dv_parameter;
                const geometry3d::NurbsSurfaceDerivatives3D d =
                    patch.evaluate_with_derivatives(u, v);
                Eigen::Vector3d normal = d.du.cross(d.dv);
                const double jacobian = normal.norm();
                if (!std::isfinite(jacobian) || jacobian <= 1.0e-14)
                    throw std::runtime_error("native NURBS DOF has degenerate Jacobian");
                normal /= jacobian;
                const auto tangents = tangent_frame(d.du, normal);
                cloud.dofs.push_back({d.point,
                                      normal,
                                      tangents.first,
                                      tangents.second,
                                      jacobian * du_parameter * dv_parameter,
                                      u,
                                      v,
                                      patch_id,
                                      i,
                                      j});
            }
        }
    }
    return cloud;
}

namespace {

struct DofLatticeCoordinate3D {
    int patch = -1;
    int i = -1;
    int j = -1;
};

bool lattice_coordinate_is_inside(const SurfaceDofCloud3D& cloud,
                                  const DofLatticeCoordinate3D& coordinate)
{
    if (coordinate.patch < 0
        || coordinate.patch >= static_cast<int>(cloud.patches.size())) {
        return false;
    }
    const SurfaceDofPatch3D& patch =
        cloud.patches[static_cast<std::size_t>(coordinate.patch)];
    return coordinate.i >= 0 && coordinate.i < patch.nu
        && coordinate.j >= 0 && coordinate.j < patch.nv;
}

PatchEdge3D first_crossed_edge(const SurfaceDofPatch3D& patch,
                               const DofLatticeCoordinate3D& coordinate)
{
    if (coordinate.i < 0)
        return PatchEdge3D::UMin;
    if (coordinate.i >= patch.nu)
        return PatchEdge3D::UMax;
    if (coordinate.j < 0)
        return PatchEdge3D::VMin;
    return PatchEdge3D::VMax;
}

int along_index(PatchEdge3D edge,
                const DofLatticeCoordinate3D& coordinate)
{
    return edge == PatchEdge3D::UMin || edge == PatchEdge3D::UMax
        ? coordinate.j
        : coordinate.i;
}

int along_count(PatchEdge3D edge, const SurfaceDofPatch3D& patch)
{
    return edge == PatchEdge3D::UMin || edge == PatchEdge3D::UMax
        ? patch.nv
        : patch.nu;
}

void reflect_across_feature_edge(PatchEdge3D edge,
                                 const SurfaceDofPatch3D& patch,
                                 DofLatticeCoordinate3D& coordinate)
{
    switch (edge) {
    case PatchEdge3D::UMin:
        coordinate.i = -coordinate.i;
        break;
    case PatchEdge3D::UMax:
        coordinate.i = 2 * patch.nu - 2 - coordinate.i;
        break;
    case PatchEdge3D::VMin:
        coordinate.j = -coordinate.j;
        break;
    case PatchEdge3D::VMax:
        coordinate.j = 2 * patch.nv - 2 - coordinate.j;
        break;
    }
}

std::optional<SmoothEdgeTransition3D> smooth_transition_for_lattice(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    PatchEdge3D source_edge,
    const DofLatticeCoordinate3D& coordinate)
{
    const SurfaceDofPatch3D& source =
        cloud.patches[static_cast<std::size_t>(coordinate.patch)];
    const int source_along_index = along_index(source_edge, coordinate);
    const int source_along_count = along_count(source_edge, source);
    const auto source_domain = edge_parameter_domain(
        surface, coordinate.patch, source_edge);
    const double along =
        (static_cast<double>(source_along_index) + 0.5)
        / static_cast<double>(source_along_count);
    const double source_parameter =
        source_domain.first
        + along * (source_domain.second - source_domain.first);

    // A corner candidate can be half a lattice cell beyond the edge-domain
    // endpoint because the 2x2 stencil crosses two patch edges in sequence.
    // Clamp only the interval-selection probe; retain the extrapolated
    // parameter below so the second edge crossing is not lost.
    const double probe = std::clamp(
        source_parameter, source_domain.first, source_domain.second);
    const double scale = std::max(1.0, source_domain.second - source_domain.first);
    const double tolerance = 2.0e-12 * scale;
    const std::vector<SmoothEdgeTransition3D> transitions =
        smooth_edge_transitions(surface, coordinate.patch, source_edge);
    for (const SmoothEdgeTransition3D& transition : transitions) {
        if (probe >= transition.source_begin - tolerance
            && probe <= transition.source_end + tolerance) {
            return transition;
        }
    }
    return std::nullopt;
}

void cross_smooth_edge(const NativeNurbsSurface3D& surface,
                       const SurfaceDofCloud3D& cloud,
                       PatchEdge3D source_edge,
                       const SmoothEdgeTransition3D& connection,
                       DofLatticeCoordinate3D& coordinate)
{
    const SurfaceDofPatch3D& source =
        cloud.patches[static_cast<std::size_t>(coordinate.patch)];
    const int source_along_index = along_index(source_edge, coordinate);
    const int source_along_count = along_count(source_edge, source);
    const auto source_domain = edge_parameter_domain(
        surface, coordinate.patch, source_edge);
    const double source_parameter =
        source_domain.first
        + (static_cast<double>(source_along_index) + 0.5)
              / static_cast<double>(source_along_count)
              * (source_domain.second - source_domain.first);
    double interval_coordinate =
        (source_parameter - connection.source_begin)
        / (connection.source_end - connection.source_begin);
    if (connection.reversed)
        interval_coordinate = 1.0 - interval_coordinate;
    const double destination_parameter =
        connection.destination_begin
        + interval_coordinate
              * (connection.destination_end - connection.destination_begin);
    const auto destination_domain = edge_parameter_domain(
        surface, connection.patch, connection.edge);
    const double destination_along =
        (destination_parameter - destination_domain.first)
        / (destination_domain.second - destination_domain.first);

    coordinate.patch = connection.patch;
    const SurfaceDofPatch3D& destination =
        cloud.patches[static_cast<std::size_t>(coordinate.patch)];
    const int destination_along_count =
        along_count(connection.edge, destination);
    const int destination_along_index = static_cast<int>(
        std::floor(destination_along
                   * static_cast<double>(destination_along_count)));
    switch (connection.edge) {
    case PatchEdge3D::UMin:
        coordinate.i = 0;
        coordinate.j = destination_along_index;
        break;
    case PatchEdge3D::UMax:
        coordinate.i = destination.nu - 1;
        coordinate.j = destination_along_index;
        break;
    case PatchEdge3D::VMin:
        coordinate.i = destination_along_index;
        coordinate.j = 0;
        break;
    case PatchEdge3D::VMax:
        coordinate.i = destination_along_index;
        coordinate.j = destination.nv - 1;
        break;
    }
}

int resolve_lattice_candidate(const NativeNurbsSurface3D& surface,
                              const SurfaceDofCloud3D& cloud,
                              DofLatticeCoordinate3D coordinate)
{
    for (int crossing = 0; crossing < 4; ++crossing) {
        if (lattice_coordinate_is_inside(cloud, coordinate)) {
            const SurfaceDofPatch3D& patch =
                cloud.patches[static_cast<std::size_t>(coordinate.patch)];
            return patch.dof_index(coordinate.i, coordinate.j);
        }
        if (coordinate.patch < 0
            || coordinate.patch >= static_cast<int>(cloud.patches.size())) {
            throw std::runtime_error("2x2 candidate has invalid patch");
        }
        const SurfaceDofPatch3D& patch =
            cloud.patches[static_cast<std::size_t>(coordinate.patch)];
        const PatchEdge3D edge = first_crossed_edge(patch, coordinate);
        const std::optional<SmoothEdgeTransition3D> connection =
            smooth_transition_for_lattice(surface, cloud, edge, coordinate);
        if (connection)
            cross_smooth_edge(surface, cloud, edge, *connection, coordinate);
        else
            reflect_across_feature_edge(edge, patch, coordinate);
    }
    throw std::runtime_error("2x2 candidate crossed too many patch edges");
}

std::array<int, 2> bracketing_center_indices(double parameter,
                                             double start,
                                             double end,
                                             int count)
{
    const double delta = (end - start) / static_cast<double>(count);
    const int lower = static_cast<int>(
        std::floor((parameter - start) / delta - 0.5));
    return {{lower, lower + 1}};
}

using DistanceDofPair = std::pair<double, int>;

bool distance_dof_less(const DistanceDofPair& a,
                       const DistanceDofPair& b)
{
    if (a.first != b.first)
        return a.first < b.first;
    return a.second < b.second;
}

struct TopologicalCauchyCandidates {
    int center_patch = -1;
    std::vector<DistanceDofPair> sorted;
};

TopologicalCauchyCandidates build_topological_cauchy_candidates(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    int center_dof,
    int count)
{
    if (center_dof < 0
        || center_dof >= static_cast<int>(cloud.dofs.size())) {
        throw std::out_of_range("Cauchy center DOF is outside the surface cloud");
    }
    if (count <= 0)
        throw std::invalid_argument("Cauchy sample count must be positive");
    if (surface.patches.size() != cloud.patches.size()
        || surface.topological_patch_neighbors.size()
               != surface.patches.size()) {
        throw std::invalid_argument(
            "Cauchy selection requires complete patch topology");
    }

    TopologicalCauchyCandidates result;
    result.center_patch =
        cloud.dofs[static_cast<std::size_t>(center_dof)].patch_id;
    if (result.center_patch < 0
        || result.center_patch >= static_cast<int>(cloud.patches.size())) {
        throw std::runtime_error("Cauchy center DOF has invalid patch ownership");
    }

    std::vector<bool> visited(surface.patches.size(), false);
    visited[static_cast<std::size_t>(result.center_patch)] = true;
    std::vector<int> included{result.center_patch};
    std::vector<int> frontier{result.center_patch};
    int candidate_count = cloud.patches[
        static_cast<std::size_t>(result.center_patch)].dof_count();

    bool expanded_one_ring = false;
    while ((!expanded_one_ring || candidate_count < count)
           && !frontier.empty()) {
        std::vector<int> next;
        for (int patch : frontier) {
            for (int neighbor : surface.topological_patch_neighbors[
                     static_cast<std::size_t>(patch)]) {
                if (neighbor < 0
                    || neighbor >= static_cast<int>(surface.patches.size())) {
                    throw std::runtime_error(
                        "Cauchy topology contains an invalid patch");
                }
                if (!visited[static_cast<std::size_t>(neighbor)]) {
                    visited[static_cast<std::size_t>(neighbor)] = true;
                    next.push_back(neighbor);
                }
            }
        }
        std::sort(next.begin(), next.end());
        for (int patch : next) {
            included.push_back(patch);
            candidate_count += cloud.patches[
                static_cast<std::size_t>(patch)].dof_count();
        }
        frontier = std::move(next);
        expanded_one_ring = true;
    }
    if (candidate_count < count) {
        throw std::runtime_error(
            "topological Cauchy neighborhood has too few surface DOFs");
    }

    const Eigen::Vector3d& center_point =
        cloud.dofs[static_cast<std::size_t>(center_dof)].point;
    result.sorted.reserve(static_cast<std::size_t>(candidate_count));
    for (int patch : included) {
        const SurfaceDofPatch3D& tensor =
            cloud.patches[static_cast<std::size_t>(patch)];
        for (int local = 0; local < tensor.dof_count(); ++local) {
            const int q = tensor.first_dof + local;
            if (q < 0 || q >= static_cast<int>(cloud.dofs.size())) {
                throw std::runtime_error(
                    "Cauchy patch has invalid contiguous DOF range");
            }
            result.sorted.emplace_back(
                (cloud.dofs[static_cast<std::size_t>(q)].point - center_point)
                    .squaredNorm(),
                q);
        }
    }
    std::sort(result.sorted.begin(), result.sorted.end(), distance_dof_less);
    return result;
}

} // namespace

std::array<int, 4> parameter_dof_candidates_2x2(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    int patch_id,
    double u,
    double v)
{
    if (patch_id < 0
        || patch_id >= static_cast<int>(surface.patches.size())
        || surface.patches.size() != cloud.patches.size()
        || (!surface.smooth_neighbors.empty()
            && surface.smooth_neighbors.size() != surface.patches.size())) {
        throw std::invalid_argument("invalid native patch for 2x2 DOF lookup");
    }
    const geometry3d::NurbsSurfacePatch3D& patch =
        surface.patches[static_cast<std::size_t>(patch_id)];
    const SurfaceDofPatch3D& tensor =
        cloud.patches[static_cast<std::size_t>(patch_id)];
    const std::array<int, 2> indices_u = bracketing_center_indices(
        u, patch.domain_start_u(), patch.domain_end_u(), tensor.nu);
    const std::array<int, 2> indices_v = bracketing_center_indices(
        v, patch.domain_start_v(), patch.domain_end_v(), tensor.nv);

    std::array<int, 4> result{};
    int candidate = 0;
    for (int i : indices_u) {
        for (int j : indices_v) {
            result[static_cast<std::size_t>(candidate++)] =
                resolve_lattice_candidate(
                    surface, cloud, {patch_id, i, j});
        }
    }
    const std::set<int> unique(result.begin(), result.end());
    if (unique.size() != result.size()) {
        std::ostringstream message;
        message << "2x2 parameter lookup did not produce four DOFs: patch="
                << patch_id << " uv=(" << u << ',' << v << ") ids="
                << result[0] << ',' << result[1] << ','
                << result[2] << ',' << result[3];
        throw std::runtime_error(message.str());
    }
    return result;
}

Eigen::Vector2d interpolate_triangle_parameter(
    const geometry3d::NurbsParamTriangle3D& triangle,
    const Eigen::Vector3d& barycentric)
{
    if (!barycentric.allFinite())
        throw std::invalid_argument("triangle barycentric coordinates are not finite");
    return barycentric.x() * triangle.uv[0]
         + barycentric.y() * triangle.uv[1]
         + barycentric.z() * triangle.uv[2];
}

std::vector<int> nearest_topological_cauchy_dofs(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    int center_dof,
    int count)
{
    const TopologicalCauchyCandidates candidates =
        build_topological_cauchy_candidates(surface, cloud, center_dof, count);

    std::vector<int> result;
    result.reserve(static_cast<std::size_t>(count));
    for (int k = 0; k < count; ++k)
        result.push_back(candidates.sorted[static_cast<std::size_t>(k)].second);
    if (std::find(result.begin(), result.end(), center_dof) == result.end()) {
        throw std::runtime_error(
            "topological Cauchy selection lost its center DOF");
    }
    return result;
}

std::vector<int> nearest_g1_cauchy_dofs(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    int center_dof,
    int count)
{
    if (center_dof < 0
        || center_dof >= static_cast<int>(cloud.dofs.size())) {
        throw std::out_of_range("Cauchy center DOF is outside the surface cloud");
    }
    if (count <= 0)
        throw std::invalid_argument("Cauchy sample count must be positive");
    if (surface.patches.size() != cloud.patches.size()
        || (!surface.smooth_neighbors.empty()
            && surface.smooth_neighbors.size() != surface.patches.size())) {
        throw std::invalid_argument(
            "G1 Cauchy selection requires complete smooth topology");
    }

    const int center_patch =
        cloud.dofs[static_cast<std::size_t>(center_dof)].patch_id;
    if (center_patch < 0
        || center_patch >= static_cast<int>(cloud.patches.size())) {
        throw std::runtime_error("Cauchy center DOF has invalid patch ownership");
    }

    std::vector<bool> visited(surface.patches.size(), false);
    visited[static_cast<std::size_t>(center_patch)] = true;
    std::vector<int> included{center_patch};
    std::vector<int> frontier{center_patch};
    int candidate_count =
        cloud.patches[static_cast<std::size_t>(center_patch)].dof_count();
    bool expanded_one_ring = false;
    while ((!expanded_one_ring || candidate_count < count)
           && !frontier.empty()) {
        std::vector<int> next;
        for (int patch : frontier) {
            for (int neighbor : smooth_patch_neighbors_3d(surface, patch)) {
                if (neighbor < 0
                    || neighbor >= static_cast<int>(surface.patches.size())) {
                    throw std::runtime_error(
                        "G1 Cauchy topology contains an invalid patch");
                }
                if (!visited[static_cast<std::size_t>(neighbor)]) {
                    visited[static_cast<std::size_t>(neighbor)] = true;
                    next.push_back(neighbor);
                }
            }
        }
        std::sort(next.begin(), next.end());
        for (int patch : next) {
            included.push_back(patch);
            candidate_count +=
                cloud.patches[static_cast<std::size_t>(patch)].dof_count();
        }
        frontier = std::move(next);
        expanded_one_ring = true;
    }

    const Eigen::Vector3d& center_point =
        cloud.dofs[static_cast<std::size_t>(center_dof)].point;
    std::vector<DistanceDofPair> candidates;
    candidates.reserve(static_cast<std::size_t>(candidate_count));
    for (int patch : included) {
        const SurfaceDofPatch3D& tensor =
            cloud.patches[static_cast<std::size_t>(patch)];
        for (int local = 0; local < tensor.dof_count(); ++local) {
            const int q = tensor.first_dof + local;
            if (q < 0 || q >= static_cast<int>(cloud.dofs.size())) {
                throw std::runtime_error(
                    "G1 Cauchy patch has invalid contiguous DOF range");
            }
            candidates.emplace_back(
                (cloud.dofs[static_cast<std::size_t>(q)].point - center_point)
                    .squaredNorm(),
                q);
        }
    }
    std::sort(candidates.begin(), candidates.end(), distance_dof_less);
    const int selected = std::min(count, static_cast<int>(candidates.size()));
    std::vector<int> result;
    result.reserve(static_cast<std::size_t>(selected));
    for (int k = 0; k < selected; ++k)
        result.push_back(candidates[static_cast<std::size_t>(k)].second);
    if (std::find(result.begin(), result.end(), center_dof) == result.end())
        throw std::runtime_error("G1 Cauchy selection lost its center DOF");
    return result;
}

std::vector<int> nearest_same_patch_cauchy_dofs(
    const SurfaceDofCloud3D& cloud,
    int center_dof,
    int count)
{
    if (center_dof < 0
        || center_dof >= static_cast<int>(cloud.dofs.size())) {
        throw std::out_of_range("Cauchy center DOF is outside the surface cloud");
    }
    if (count <= 0)
        throw std::invalid_argument("Cauchy sample count must be positive");
    const int patch_id =
        cloud.dofs[static_cast<std::size_t>(center_dof)].patch_id;
    if (patch_id < 0
        || patch_id >= static_cast<int>(cloud.patches.size())) {
        throw std::runtime_error("Cauchy center DOF has invalid patch ownership");
    }
    const SurfaceDofPatch3D& patch =
        cloud.patches[static_cast<std::size_t>(patch_id)];
    if (patch.dof_count() <= 0)
        throw std::runtime_error("same-patch Cauchy neighborhood is empty");

    const Eigen::Vector3d& center_point =
        cloud.dofs[static_cast<std::size_t>(center_dof)].point;
    std::vector<DistanceDofPair> candidates;
    candidates.reserve(static_cast<std::size_t>(patch.dof_count()));
    for (int local = 0; local < patch.dof_count(); ++local) {
        const int q = patch.first_dof + local;
        if (q < 0 || q >= static_cast<int>(cloud.dofs.size())) {
            throw std::runtime_error(
                "same-patch Cauchy neighborhood has an invalid DOF range");
        }
        candidates.emplace_back(
            (cloud.dofs[static_cast<std::size_t>(q)].point - center_point)
                .squaredNorm(),
            q);
    }
    std::sort(candidates.begin(), candidates.end(), distance_dof_less);
    const int selected = std::min(count, static_cast<int>(candidates.size()));
    std::vector<int> result;
    result.reserve(static_cast<std::size_t>(selected));
    for (int k = 0; k < selected; ++k)
        result.push_back(candidates[static_cast<std::size_t>(k)].second);
    if (std::find(result.begin(), result.end(), center_dof) == result.end())
        throw std::runtime_error("same-patch Cauchy selection lost its center DOF");
    return result;
}

std::vector<int> balanced_topological_cauchy_dofs(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    int center_dof,
    int count)
{
    const TopologicalCauchyCandidates candidates =
        build_topological_cauchy_candidates(surface, cloud, center_dof, count);
    std::set<int> active_patch_set;
    for (int k = 0; k < count; ++k) {
        const int q = candidates.sorted[static_cast<std::size_t>(k)].second;
        active_patch_set.insert(
            cloud.dofs[static_cast<std::size_t>(q)].patch_id);
    }
    std::vector<int> active_patches(
        active_patch_set.begin(), active_patch_set.end());
    std::vector<std::vector<int>> groups(active_patches.size());
    for (const DistanceDofPair& candidate : candidates.sorted) {
        const int patch =
            cloud.dofs[static_cast<std::size_t>(candidate.second)].patch_id;
        const auto found = std::lower_bound(
            active_patches.begin(), active_patches.end(), patch);
        if (found != active_patches.end() && *found == patch) {
            groups[static_cast<std::size_t>(found - active_patches.begin())]
                .push_back(candidate.second);
        }
    }

    std::vector<std::size_t> cursors(groups.size(), 0);
    std::vector<int> result;
    result.reserve(static_cast<std::size_t>(count));
    while (static_cast<int>(result.size()) < count) {
        bool progressed = false;
        for (std::size_t group = 0;
             group < groups.size() && static_cast<int>(result.size()) < count;
             ++group) {
            if (cursors[group] >= groups[group].size())
                continue;
            result.push_back(groups[group][cursors[group]++]);
            progressed = true;
        }
        if (!progressed) {
            throw std::runtime_error(
                "balanced Cauchy neighborhood cannot fill the requested count");
        }
    }

    const Eigen::Vector3d& center_point =
        cloud.dofs[static_cast<std::size_t>(center_dof)].point;
    std::sort(result.begin(), result.end(), [&](int a, int b) {
        return distance_dof_less(
            {(cloud.dofs[static_cast<std::size_t>(a)].point - center_point)
                 .squaredNorm(), a},
            {(cloud.dofs[static_cast<std::size_t>(b)].point - center_point)
                 .squaredNorm(), b});
    });
    if (std::find(result.begin(), result.end(), center_dof) == result.end()) {
        throw std::runtime_error(
            "balanced Cauchy selection lost its center DOF");
    }
    return result;
}

std::vector<int> smooth_patch_component(const NativeNurbsSurface3D& surface,
                                        int patch)
{
    if (patch < 0 || patch >= static_cast<int>(surface.patches.size())
        || (!surface.smooth_neighbors.empty()
            && surface.smooth_neighbors.size() != surface.patches.size())) {
        throw std::invalid_argument("invalid patch for smooth component query");
    }
    std::vector<int> result;
    std::vector<bool> visited(surface.patches.size(), false);
    std::queue<int> pending;
    visited[static_cast<std::size_t>(patch)] = true;
    pending.push(patch);
    while (!pending.empty()) {
        const int current = pending.front();
        pending.pop();
        result.push_back(current);
        for (int neighbor : smooth_patch_neighbors_3d(surface, current)) {
            if (!visited[static_cast<std::size_t>(neighbor)]) {
                visited[static_cast<std::size_t>(neighbor)] = true;
                pending.push(neighbor);
            }
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace kfbim::app3d
