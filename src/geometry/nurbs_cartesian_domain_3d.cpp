#include "nurbs_cartesian_domain_3d.hpp"

#include "nurbs_bezier_extraction_3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace kfbim::geometry3d {
namespace {

constexpr std::uint64_t kNodeMask =
    (std::uint64_t{1} << 62) - std::uint64_t{1};

std::uint64_t edge_key(int axis, int start_node)
{
    return (std::uint64_t(axis) << 62)
        | std::uint64_t(start_node);
}

void accumulate_intersection_diagnostics(
    NurbsSurfaceIntersectionDiagnostics3D& total,
    const NurbsSurfaceIntersectionDiagnostics3D& increment)
{
    total.candidate_elements += increment.candidate_elements;
    total.triangle_seed_hits += increment.triangle_seed_hits;
    total.triangle_seed_misses_recovered +=
        increment.triangle_seed_misses_recovered;
    total.subdivision_boxes += increment.subdivision_boxes;
    total.newton_attempts += increment.newton_attempts;
    total.newton_iterations += increment.newton_iterations;
    total.same_patch_deduplications +=
        increment.same_patch_deduplications;
    total.seam_deduplications += increment.seam_deduplications;
    total.unresolved_candidates += increment.unresolved_candidates;
}

Eigen::Vector3d as_vector(const std::array<double, 3>& values)
{
    return {values[0], values[1], values[2]};
}

std::string strict_box_error(const NurbsAabb3D& surface,
                             const Eigen::Vector3d& box_lower,
                             const Eigen::Vector3d& box_upper)
{
    std::ostringstream message;
    message << std::setprecision(17)
            << "NURBS surface must lie strictly inside Cartesian box"
            << ": surface lower=(" << surface.lower.x() << ','
            << surface.lower.y() << ',' << surface.lower.z() << ')'
            << " upper=(" << surface.upper.x() << ','
            << surface.upper.y() << ',' << surface.upper.z() << ')'
            << ", Cartesian box lower=(" << box_lower.x() << ','
            << box_lower.y() << ',' << box_lower.z() << ')'
            << " upper=(" << box_upper.x() << ','
            << box_upper.y() << ',' << box_upper.z() << ')';
    return message.str();
}

} // namespace

struct NurbsCartesianDomain3D::Impl {
    std::array<int, 3> dims{{0, 0, 0}};
    std::array<std::vector<std::uint8_t>, 3> barriers;
    std::vector<NurbsSurfaceCrossing3D> crossings;
    std::unordered_map<std::uint64_t, int> crossing_by_edge;
    std::vector<int> node_labels;
    NurbsCartesianDomainDiagnostics3D diagnostics;
    NurbsAabb3D bounds;
    double tolerance = 0.0;

    Impl(const CartesianGrid3D& grid,
         NurbsSurfaceModel3D model,
         NurbsCartesianDomainOptions3D options)
    {
        if (grid.layout() != DofLayout3D::Node) {
            throw std::invalid_argument(
                "NurbsCartesianDomain3D requires a node-layout Cartesian grid");
        }
        const auto spacing = grid.spacing();
        for (double h : spacing) {
            if (!std::isfinite(h) || h <= 0.0) {
                throw std::invalid_argument(
                    "NurbsCartesianDomain3D requires positive finite Cartesian spacing");
            }
        }
        dims = grid.dof_dims();
        if (std::any_of(dims.begin(), dims.end(), [](int extent) {
                return extent < 2;
            })) {
            throw std::invalid_argument(
                "NurbsCartesianDomain3D requires at least one Cartesian cell per axis");
        }

        NurbsSurfaceIntersectorOptions3D intersector_options;
        intersector_options.use_triangle_seeds = options.use_triangle_seeds;
        NurbsSurfaceIntersector3D intersector(
            std::move(model), intersector_options);
        bounds = intersector.bounds();
        tolerance = intersector.geometry_tolerance();
        diagnostics.nurbs_patch_count = intersector.model().num_patches();
        diagnostics.bezier_element_count = static_cast<int>(
            extract_rational_bezier_elements_3d(
                intersector.model()).size());

        const Eigen::Vector3d box_lower = as_vector(grid.origin());
        Eigen::Vector3d box_upper = box_lower;
        for (int axis = 0; axis < 3; ++axis) {
            box_upper[axis] += spacing[static_cast<std::size_t>(axis)]
                * static_cast<double>(
                    dims[static_cast<std::size_t>(axis)] - 1);
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (!(bounds.lower[axis] > box_lower[axis]
                  && bounds.upper[axis] < box_upper[axis])) {
                throw std::runtime_error(
                    strict_box_error(bounds, box_lower, box_upper));
            }
        }

        const int nx = dims[0];
        const int ny = dims[1];
        const int nz = dims[2];
        barriers[0].assign(
            static_cast<std::size_t>(nx - 1)
                * static_cast<std::size_t>(ny)
                * static_cast<std::size_t>(nz),
            std::uint8_t{0});
        barriers[1].assign(
            static_cast<std::size_t>(nx)
                * static_cast<std::size_t>(ny - 1)
                * static_cast<std::size_t>(nz),
            std::uint8_t{0});
        barriers[2].assign(
            static_cast<std::size_t>(nx)
                * static_cast<std::size_t>(ny)
                * static_cast<std::size_t>(nz - 1),
            std::uint8_t{0});

        const double maximum_spacing =
            *std::max_element(spacing.begin(), spacing.end());
        const std::vector<RationalBezierElement3D> leaves =
            intersector.acceleration_leaves(2.0 * maximum_spacing);
        diagnostics.acceleration_leaf_count =
            static_cast<int>(leaves.size());

        const auto origin = grid.origin();
        std::vector<std::uint64_t> candidate_keys;
        for (const RationalBezierElement3D& leaf : leaves) {
            const NurbsAabb3D leaf_bounds = leaf.bounds();
            for (int axis = 0; axis < 3; ++axis) {
                const double h = spacing[static_cast<std::size_t>(axis)];
                int start_min = static_cast<int>(std::ceil(
                    (leaf_bounds.lower[axis]
                        - origin[static_cast<std::size_t>(axis)]
                        - tolerance) / h)) - 1;
                int start_max = static_cast<int>(std::floor(
                    (leaf_bounds.upper[axis]
                        - origin[static_cast<std::size_t>(axis)]
                        + tolerance) / h));
                start_min = std::clamp(
                    start_min, 0,
                    dims[static_cast<std::size_t>(axis)] - 2);
                start_max = std::clamp(
                    start_max, 0,
                    dims[static_cast<std::size_t>(axis)] - 2);
                if (start_min > start_max)
                    continue;

                std::array<int, 3> transverse_min{{0, 0, 0}};
                std::array<int, 3> transverse_max{{0, 0, 0}};
                bool empty = false;
                for (int transverse = 0; transverse < 3; ++transverse) {
                    if (transverse == axis)
                        continue;
                    const double transverse_h =
                        spacing[static_cast<std::size_t>(transverse)];
                    transverse_min[static_cast<std::size_t>(transverse)] =
                        static_cast<int>(std::ceil(
                            (leaf_bounds.lower[transverse]
                                - origin[static_cast<std::size_t>(transverse)]
                                - tolerance) / transverse_h));
                    transverse_max[static_cast<std::size_t>(transverse)] =
                        static_cast<int>(std::floor(
                            (leaf_bounds.upper[transverse]
                                - origin[static_cast<std::size_t>(transverse)]
                                + tolerance) / transverse_h));
                    transverse_min[static_cast<std::size_t>(transverse)] =
                        std::clamp(
                            transverse_min[static_cast<std::size_t>(transverse)],
                            0,
                            dims[static_cast<std::size_t>(transverse)] - 1);
                    transverse_max[static_cast<std::size_t>(transverse)] =
                        std::clamp(
                            transverse_max[static_cast<std::size_t>(transverse)],
                            0,
                            dims[static_cast<std::size_t>(transverse)] - 1);
                    empty = empty
                        || transverse_min[static_cast<std::size_t>(transverse)]
                            > transverse_max[static_cast<std::size_t>(transverse)];
                }
                if (empty)
                    continue;

                if (axis == 0) {
                    for (int k = transverse_min[2];
                         k <= transverse_max[2]; ++k) {
                        for (int j = transverse_min[1];
                             j <= transverse_max[1]; ++j) {
                            for (int i = start_min; i <= start_max; ++i) {
                                candidate_keys.push_back(
                                    edge_key(axis, grid.index(i, j, k)));
                            }
                        }
                    }
                } else if (axis == 1) {
                    for (int k = transverse_min[2];
                         k <= transverse_max[2]; ++k) {
                        for (int j = start_min; j <= start_max; ++j) {
                            for (int i = transverse_min[0];
                                 i <= transverse_max[0]; ++i) {
                                candidate_keys.push_back(
                                    edge_key(axis, grid.index(i, j, k)));
                            }
                        }
                    }
                } else {
                    for (int k = start_min; k <= start_max; ++k) {
                        for (int j = transverse_min[1];
                             j <= transverse_max[1]; ++j) {
                            for (int i = transverse_min[0];
                                 i <= transverse_max[0]; ++i) {
                                candidate_keys.push_back(
                                    edge_key(axis, grid.index(i, j, k)));
                            }
                        }
                    }
                }
            }
        }
        std::sort(candidate_keys.begin(), candidate_keys.end());
        candidate_keys.erase(
            std::unique(candidate_keys.begin(), candidate_keys.end()),
            candidate_keys.end());
        diagnostics.candidate_grid_edge_count = candidate_keys.size();
        crossings.reserve(candidate_keys.size());
        crossing_by_edge.reserve(candidate_keys.size());

        for (const std::uint64_t key : candidate_keys) {
            const int axis = static_cast<int>(key >> 62);
            const int start_node = static_cast<int>(key & kNodeMask);
            const auto ijk = node_coordinates(start_node);
            const int end_node = start_node
                + (axis == 0 ? 1 : (axis == 1 ? nx : nx * ny));
            const Eigen::Vector3d start = as_vector(grid.coord(start_node));
            const Eigen::Vector3d end = as_vector(grid.coord(end_node));
            NurbsSurfaceIntersectionDiagnostics3D local_diagnostics;
            const std::optional<NurbsSurfaceCrossing3D> crossing =
                intersector.intersect_cartesian_edge(
                    NurbsCartesianEdgeQuery3D{
                        axis, ijk[0], ijk[1], ijk[2], start, end},
                    &local_diagnostics);
            accumulate_intersection_diagnostics(
                diagnostics.intersections, local_diagnostics);
            if (!crossing)
                continue;

            barriers[static_cast<std::size_t>(axis)]
                    [barrier_index(axis, ijk[0], ijk[1], ijk[2])] =
                std::uint8_t{1};
            ++diagnostics.barrier_edge_counts[
                static_cast<std::size_t>(axis)];
            diagnostics.maximum_root_residual = std::max(
                diagnostics.maximum_root_residual, crossing->residual);
            const int crossing_index = static_cast<int>(crossings.size());
            crossings.push_back(*crossing);
            crossing_by_edge.emplace(key, crossing_index);
        }

        flood_and_label(grid, intersector);
        verify_barriers();
    }

    std::array<int, 3> node_coordinates(int node) const
    {
        const int nx = dims[0];
        const int ny = dims[1];
        const int plane = nx * ny;
        const int k = node / plane;
        const int remainder = node % plane;
        return {{remainder % nx, remainder / nx, k}};
    }

    std::size_t barrier_index(int axis, int i, int j, int k) const
    {
        const std::size_t nx = static_cast<std::size_t>(dims[0]);
        const std::size_t ny = static_cast<std::size_t>(dims[1]);
        if (axis == 0) {
            return (static_cast<std::size_t>(k) * ny
                    + static_cast<std::size_t>(j))
                    * (nx - 1)
                + static_cast<std::size_t>(i);
        }
        if (axis == 1) {
            return (static_cast<std::size_t>(k) * (ny - 1)
                    + static_cast<std::size_t>(j))
                    * nx
                + static_cast<std::size_t>(i);
        }
        return (static_cast<std::size_t>(k) * ny
                + static_cast<std::size_t>(j))
                * nx
            + static_cast<std::size_t>(i);
    }

    bool barrier_at(int axis, int i, int j, int k) const
    {
        return barriers[static_cast<std::size_t>(axis)]
                    [barrier_index(axis, i, j, k)] != 0;
    }

    std::pair<int, int> adjacent_edge(int node_a, int node_b) const
    {
        const int node_count = static_cast<int>(node_labels.size());
        if (node_a < 0 || node_a >= node_count
            || node_b < 0 || node_b >= node_count) {
            throw std::out_of_range(
                "NURBS Cartesian node index is outside the grid");
        }
        const auto first = node_coordinates(node_a);
        const auto second = node_coordinates(node_b);
        int axis = -1;
        int distance = 0;
        for (int candidate_axis = 0; candidate_axis < 3;
             ++candidate_axis) {
            const int difference = std::abs(
                first[static_cast<std::size_t>(candidate_axis)]
                - second[static_cast<std::size_t>(candidate_axis)]);
            distance += difference;
            if (difference != 0)
                axis = candidate_axis;
        }
        if (distance != 1) {
            throw std::invalid_argument(
                "NURBS Cartesian barrier query requires neighboring nodes");
        }
        return {axis, std::min(node_a, node_b)};
    }

    bool has_barrier(int node_a, int node_b) const
    {
        const auto edge = adjacent_edge(node_a, node_b);
        const auto ijk = node_coordinates(edge.second);
        return barrier_at(edge.first, ijk[0], ijk[1], ijk[2]);
    }

    void flood_and_label(const CartesianGrid3D& grid,
                         const NurbsSurfaceIntersector3D& intersector)
    {
        const int nx = dims[0];
        const int ny = dims[1];
        const int nz = dims[2];
        const int node_count = grid.num_dofs();
        std::vector<int> graph_component(
            static_cast<std::size_t>(node_count), -1);
        std::vector<int> representatives;
        std::vector<bool> touches_box_boundary;
        std::vector<int> queue;

        for (int seed = 0; seed < node_count; ++seed) {
            if (graph_component[static_cast<std::size_t>(seed)] >= 0)
                continue;
            const int component =
                static_cast<int>(representatives.size());
            representatives.push_back(seed);
            touches_box_boundary.push_back(false);
            diagnostics.component_sizes.push_back(0);
            queue.clear();
            queue.push_back(seed);
            graph_component[static_cast<std::size_t>(seed)] = component;

            for (std::size_t head = 0; head < queue.size(); ++head) {
                const int node = queue[head];
                ++diagnostics.component_sizes[
                    static_cast<std::size_t>(component)];
                const auto ijk = node_coordinates(node);
                const int i = ijk[0];
                const int j = ijk[1];
                const int k = ijk[2];
                if (i == 0 || i == nx - 1
                    || j == 0 || j == ny - 1
                    || k == 0 || k == nz - 1) {
                    touches_box_boundary[
                        static_cast<std::size_t>(component)] = true;
                }

                const std::array<int, 6> neighbors{{
                    i > 0 && !barrier_at(0, i - 1, j, k) ? node - 1 : -1,
                    i + 1 < nx && !barrier_at(0, i, j, k) ? node + 1 : -1,
                    j > 0 && !barrier_at(1, i, j - 1, k) ? node - nx : -1,
                    j + 1 < ny && !barrier_at(1, i, j, k) ? node + nx : -1,
                    k > 0 && !barrier_at(2, i, j, k - 1)
                        ? node - nx * ny : -1,
                    k + 1 < nz && !barrier_at(2, i, j, k)
                        ? node + nx * ny : -1}};
                for (const int neighbor : neighbors) {
                    if (neighbor >= 0
                        && graph_component[
                            static_cast<std::size_t>(neighbor)] < 0) {
                        graph_component[
                            static_cast<std::size_t>(neighbor)] = component;
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        diagnostics.grid_component_count =
            static_cast<int>(representatives.size());
        std::vector<int> component_labels(representatives.size(), 0);
        for (int component = 0;
             component < static_cast<int>(representatives.size());
             ++component) {
            if (touches_box_boundary[
                    static_cast<std::size_t>(component)]) {
                component_labels[static_cast<std::size_t>(component)] = 0;
                ++diagnostics.box_exterior_component_count;
            } else {
                ++diagnostics.representative_query_count;
                const Eigen::Vector3d representative_point = as_vector(
                    grid.coord(representatives[
                        static_cast<std::size_t>(component)]));
                const std::vector<int> containing =
                    intersector.containing_components(
                        representative_point);
                if (containing.empty()) {
                    component_labels[static_cast<std::size_t>(component)] = 0;
                } else if (containing.size() == 1) {
                    component_labels[static_cast<std::size_t>(component)] =
                        containing.front() + 1;
                } else {
                    throw std::runtime_error(
                        "overlapping NURBS solid components");
                }
            }
        }

        node_labels.resize(static_cast<std::size_t>(node_count));
        for (int node = 0; node < node_count; ++node) {
            node_labels[static_cast<std::size_t>(node)] =
                component_labels[static_cast<std::size_t>(
                    graph_component[static_cast<std::size_t>(node)])];
        }
    }

    void verify_edge(int axis, int i, int j, int k,
                     int first, int second) const
    {
        const bool barrier = barrier_at(axis, i, j, k);
        const bool binary_change =
            (node_labels[static_cast<std::size_t>(first)] > 0)
            != (node_labels[static_cast<std::size_t>(second)] > 0);
        const std::uint64_t key = edge_key(axis, first);
        if (binary_change && !barrier) {
            throw std::runtime_error(
                "NURBS binary label change lacks Cartesian barrier");
        }
        if (barrier && crossing_by_edge.find(key)
                == crossing_by_edge.end()) {
            throw std::runtime_error(
                "NURBS Cartesian barrier lacks crossing record");
        }
        if (barrier && !binary_change) {
            throw std::runtime_error(
                "NURBS barrier does not separate opposite sides");
        }
    }

    void verify_barriers() const
    {
        const int nx = dims[0];
        const int ny = dims[1];
        const int nz = dims[2];
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int node = (k * ny + j) * nx + i;
                    if (i + 1 < nx)
                        verify_edge(0, i, j, k, node, node + 1);
                    if (j + 1 < ny)
                        verify_edge(1, i, j, k, node, node + nx);
                    if (k + 1 < nz)
                        verify_edge(2, i, j, k, node, node + nx * ny);
                }
            }
        }
    }
};

NurbsCartesianDomain3D::NurbsCartesianDomain3D(
    const CartesianGrid3D& grid,
    NurbsSurfaceModel3D model,
    NurbsCartesianDomainOptions3D options)
    : impl_(std::make_shared<Impl>(
          grid, std::move(model), options))
{}

int NurbsCartesianDomain3D::label(int node) const
{
    if (node < 0
        || node >= static_cast<int>(impl_->node_labels.size())) {
        throw std::out_of_range(
            "NURBS Cartesian node index is outside the grid");
    }
    return impl_->node_labels[static_cast<std::size_t>(node)];
}

bool NurbsCartesianDomain3D::has_barrier_between(
    int node_a, int node_b) const
{
    return impl_->has_barrier(node_a, node_b);
}

const NurbsSurfaceCrossing3D& NurbsCartesianDomain3D::crossing_between(
    int node_a, int node_b) const
{
    const auto edge = impl_->adjacent_edge(node_a, node_b);
    const auto found = impl_->crossing_by_edge.find(
        edge_key(edge.first, edge.second));
    if (found == impl_->crossing_by_edge.end()) {
        throw std::runtime_error(
            "no NURBS crossing between Cartesian nodes");
    }
    return impl_->crossings[static_cast<std::size_t>(found->second)];
}

const std::vector<int>& NurbsCartesianDomain3D::labels() const
{
    return impl_->node_labels;
}

const NurbsCartesianDomainDiagnostics3D&
NurbsCartesianDomain3D::diagnostics() const
{
    return impl_->diagnostics;
}

const NurbsAabb3D& NurbsCartesianDomain3D::surface_bounds() const
{
    return impl_->bounds;
}

double NurbsCartesianDomain3D::geometry_tolerance() const
{
    return impl_->tolerance;
}

} // namespace kfbim::geometry3d
