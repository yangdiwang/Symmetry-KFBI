#include "nurbs_cartesian_domain_3d.hpp"

#include "nurbs_bezier_extraction_3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace kfbim::geometry3d {
namespace {

constexpr std::uint64_t kNodeMask =
    (std::uint64_t{1} << 62) - std::uint64_t{1};

struct ClosedIntInterval {
    int lower = 1;
    int upper = 0;

    bool empty() const
    {
        return lower > upper;
    }
};
struct EdgeCrossingRange3D {
    int begin = 0;
    int count = 0;
    int confirmed_transverse_count = 0;
    bool changes_inside_outside = false;
    bool changes_component_membership = false;
};

std::size_t checked_size_product(std::size_t first,
                                 std::size_t second,
                                 const char* message)
{
    if (first != 0
        && second > std::numeric_limits<std::size_t>::max() / first) {
        throw std::overflow_error(message);
    }
    return first * second;
}

std::size_t checked_size_add(std::size_t first,
                             std::size_t second,
                             const char* message)
{
    if (second > std::numeric_limits<std::size_t>::max() - first)
        throw std::overflow_error(message);
    return first + second;
}

int checked_size_to_int(std::size_t value, const char* message)
{
    if (value > static_cast<std::size_t>(
                    std::numeric_limits<int>::max())) {
        throw std::overflow_error(message);
    }
    return static_cast<int>(value);
}

int checked_add_int(int first, int second, const char* message)
{
    if ((second > 0
         && first > std::numeric_limits<int>::max() - second)
        || (second < 0
            && first < std::numeric_limits<int>::min() - second)) {
        throw std::overflow_error(message);
    }
    return first + second;
}

void checked_accumulate_int(int& total,
                            int increment,
                            const char* message)
{
    if (total < 0 || increment < 0)
        throw std::overflow_error(message);
    total = checked_add_int(total, increment, message);
}

ClosedIntInterval clamped_closed_integer_interval(
    double lower,
    double upper,
    int target_lower,
    int target_upper)
{
    if (!std::isfinite(lower) || !std::isfinite(upper)) {
        throw std::overflow_error(
            "non-finite Cartesian candidate index range");
    }
    if (target_lower > target_upper || lower > upper)
        return {};

    const double target_lower_double =
        static_cast<double>(target_lower);
    const double target_upper_double =
        static_cast<double>(target_upper);
    if (upper < target_lower_double || lower > target_upper_double)
        return {};

    const double clipped_lower = std::max(lower, target_lower_double);
    const double clipped_upper = std::min(upper, target_upper_double);
    if (!(clipped_lower >= target_lower_double
          && clipped_lower <= target_upper_double
          && clipped_upper >= target_lower_double
          && clipped_upper <= target_upper_double
          && clipped_lower <= clipped_upper
          && std::trunc(clipped_lower) == clipped_lower
          && std::trunc(clipped_upper) == clipped_upper)) {
        throw std::overflow_error(
            "Cartesian candidate index range cannot be represented");
    }
    return {static_cast<int>(clipped_lower),
            static_cast<int>(clipped_upper)};
}

ClosedIntInterval candidate_index_interval(double lower,
                                           double upper,
                                           double origin,
                                           double spacing,
                                           double tolerance,
                                           bool edge_starts,
                                           int target_upper)
{
    const double expanded_lower = lower - origin - tolerance;
    const double expanded_upper = upper - origin + tolerance;
    if (!std::isfinite(expanded_lower)
        || !std::isfinite(expanded_upper)) {
        throw std::overflow_error(
            "non-finite Cartesian candidate index range");
    }
    const double normalized_lower = expanded_lower / spacing;
    const double normalized_upper = expanded_upper / spacing;
    if (!std::isfinite(normalized_lower)
        || !std::isfinite(normalized_upper)) {
        throw std::overflow_error(
            "non-finite Cartesian candidate index range");
    }
    double integer_lower = std::ceil(normalized_lower);
    const double integer_upper = std::floor(normalized_upper);
    if (edge_starts)
        integer_lower -= 1.0;
    return clamped_closed_integer_interval(
        integer_lower, integer_upper, 0, target_upper);
}

std::uint64_t edge_key(int axis, int start_node)
{
    return (std::uint64_t(axis) << 62)
        | std::uint64_t(start_node);
}

void accumulate_intersection_diagnostics(
    NurbsSurfaceIntersectionDiagnostics3D& total,
    const NurbsSurfaceIntersectionDiagnostics3D& increment)
{
    checked_accumulate_int(
        total.candidate_elements, increment.candidate_elements,
        "NURBS intersection candidate diagnostic overflow");
    checked_accumulate_int(
        total.triangle_seed_hits, increment.triangle_seed_hits,
        "NURBS triangle-seed diagnostic overflow");
    checked_accumulate_int(
        total.triangle_seed_misses_recovered,
        increment.triangle_seed_misses_recovered,
        "NURBS triangle-seed recovery diagnostic overflow");
    checked_accumulate_int(
        total.subdivision_boxes, increment.subdivision_boxes,
        "NURBS subdivision diagnostic overflow");
    checked_accumulate_int(
        total.newton_attempts, increment.newton_attempts,
        "NURBS Newton-attempt diagnostic overflow");
    checked_accumulate_int(
        total.newton_iterations, increment.newton_iterations,
        "NURBS Newton-iteration diagnostic overflow");
    checked_accumulate_int(
        total.same_patch_deduplications,
        increment.same_patch_deduplications,
        "NURBS same-patch diagnostic overflow");
    checked_accumulate_int(
        total.seam_deduplications,
        increment.seam_deduplications,
        "NURBS seam diagnostic overflow");
    checked_accumulate_int(
        total.unresolved_candidates,
        increment.unresolved_candidates,
        "NURBS unresolved-candidate diagnostic overflow");
    total.maximum_subdivision_depth_reached = std::max(
        total.maximum_subdivision_depth_reached,
        increment.maximum_subdivision_depth_reached);
    checked_accumulate_int(
        total.terminal_certificate_boxes,
        increment.terminal_certificate_boxes,
        "NURBS terminal-certificate diagnostic overflow");
    total.maximum_terminal_certificate_depth_reached = std::max(
        total.maximum_terminal_certificate_depth_reached,
        increment.maximum_terminal_certificate_depth_reached);
    checked_accumulate_int(
        total.closest_point_attempts, increment.closest_point_attempts,
        "NURBS closest-point attempt diagnostic overflow");
    checked_accumulate_int(
        total.closest_point_iterations, increment.closest_point_iterations,
        "NURBS closest-point iteration diagnostic overflow");
    checked_accumulate_int(
        total.roots_recovered_by_closest_point,
        increment.roots_recovered_by_closest_point,
        "NURBS closest-point recovery diagnostic overflow");
    checked_accumulate_int(
        total.terminal_misses_by_closest_point,
        increment.terminal_misses_by_closest_point,
        "NURBS closest-point terminal-miss diagnostic overflow");
    checked_accumulate_int(
        total.closest_point_failures, increment.closest_point_failures,
        "NURBS closest-point failure diagnostic overflow");
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
    DofLayout3D grid_layout = DofLayout3D::Node;
    std::array<int, 3> grid_cells{{0, 0, 0}};
    std::array<double, 3> grid_origin{{0.0, 0.0, 0.0}};
    std::array<double, 3> grid_spacing{{0.0, 0.0, 0.0}};
    std::array<int, 3> dims{{0, 0, 0}};
    int plane_stride = 0;
    int node_count = 0;
    std::array<std::vector<std::uint8_t>, 3> barriers;
    std::array<std::vector<std::uint8_t>, 3> interface_edges;
    std::vector<NurbsSurfaceCrossing3D> crossings;
    std::unordered_map<std::uint64_t, EdgeCrossingRange3D>
        crossing_ranges_by_edge;
    std::vector<int> node_labels;
    NurbsCartesianDomainDiagnostics3D diagnostics;
    NurbsAabb3D bounds;
    double tolerance = 0.0;

    Impl(const CartesianGrid3D& grid,
         NurbsSurfaceModel3D model,
         NurbsCartesianDomainOptions3D options)
        : grid_layout(grid.layout())
        , grid_cells(grid.num_cells())
        , grid_origin(grid.origin())
        , grid_spacing(grid.spacing())
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
        const auto cells = grid.num_cells();
        for (int axis = 0; axis < 3; ++axis) {
            const int cell_count = cells[static_cast<std::size_t>(axis)];
            if (cell_count < 1) {
                throw std::invalid_argument(
                    "NurbsCartesianDomain3D requires at least one Cartesian cell per axis");
            }
            if (cell_count > std::numeric_limits<int>::max() - 1) {
                throw std::overflow_error(
                    "Cartesian cell-to-node dimension overflow");
            }
            dims[static_cast<std::size_t>(axis)] = cell_count + 1;
        }

        const std::size_t nx_size = static_cast<std::size_t>(dims[0]);
        const std::size_t ny_size = static_cast<std::size_t>(dims[1]);
        const std::size_t nz_size = static_cast<std::size_t>(dims[2]);
        const std::size_t plane_size = checked_size_product(
            nx_size, ny_size, "Cartesian node-plane storage overflow");
        if (plane_size > static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "Cartesian node count exceeds int range");
        }
        const std::size_t node_count_size = checked_size_product(
            plane_size, nz_size, "Cartesian node storage overflow");
        if (node_count_size > static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "Cartesian node count exceeds int range");
        }
        plane_stride = checked_size_to_int(
            plane_size, "Cartesian node stride exceeds int range");
        node_count = checked_size_to_int(
            node_count_size, "Cartesian node count exceeds int range");

        const Eigen::Vector3d box_lower = as_vector(grid.origin());
        Eigen::Vector3d box_upper = box_lower;
        for (int axis = 0; axis < 3; ++axis) {
            const double extent =
                spacing[static_cast<std::size_t>(axis)]
                * static_cast<double>(
                    cells[static_cast<std::size_t>(axis)]);
            box_upper[axis] += extent;
        }
        if (!box_lower.allFinite() || !box_upper.allFinite()) {
            throw std::overflow_error(
                "Cartesian box bounds must be finite");
        }

        const double maximum_spacing =
            *std::max_element(spacing.begin(), spacing.end());
        const double maximum_leaf_extent = 2.0 * maximum_spacing;
        if (!std::isfinite(maximum_leaf_extent)) {
            throw std::overflow_error(
                "NURBS acceleration leaf extent must be finite");
        }

        NurbsSurfaceIntersectorOptions3D intersector_options;
        intersector_options.use_triangle_seeds = options.use_triangle_seeds;
        intersector_options.maximum_element_extent = maximum_leaf_extent;
        intersector_options.local_max_subdivision_depth = 4;
        NurbsSurfaceIntersector3D intersector(
            std::move(model), intersector_options);
        bounds = intersector.bounds();
        tolerance = intersector.geometry_tolerance();
        if (!std::isfinite(tolerance) || tolerance < 0.0) {
            throw std::overflow_error(
                "NURBS geometry tolerance must be finite");
        }
        diagnostics.maximum_query_element_extent =
            intersector.maximum_query_element_extent();
        diagnostics.nurbs_patch_count = checked_size_to_int(
            intersector.model().patches().size(),
            "NURBS patch diagnostic count overflow");
        {
            const std::vector<RationalBezierElement3D> bezier_elements =
                extract_rational_bezier_elements_3d(intersector.model());
            diagnostics.bezier_element_count = checked_size_to_int(
                bezier_elements.size(),
                "NURBS Bezier-element diagnostic count overflow");
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
        const std::size_t x_barrier_size = checked_size_product(
            checked_size_product(
                static_cast<std::size_t>(nx - 1),
                static_cast<std::size_t>(ny),
                "Cartesian x-barrier row storage overflow"),
            static_cast<std::size_t>(nz),
            "Cartesian x-barrier storage overflow");
        const std::size_t y_barrier_size = checked_size_product(
            checked_size_product(
                static_cast<std::size_t>(nx),
                static_cast<std::size_t>(ny - 1),
                "Cartesian y-barrier row storage overflow"),
            static_cast<std::size_t>(nz),
            "Cartesian y-barrier storage overflow");
        const std::size_t z_barrier_size = checked_size_product(
            checked_size_product(
                static_cast<std::size_t>(nx),
                static_cast<std::size_t>(ny),
                "Cartesian z-barrier row storage overflow"),
            static_cast<std::size_t>(nz - 1),
            "Cartesian z-barrier storage overflow");
        barriers[0].assign(
            x_barrier_size, std::uint8_t{0});
        barriers[1].assign(
            y_barrier_size, std::uint8_t{0});
        barriers[2].assign(
            z_barrier_size, std::uint8_t{0});
        interface_edges[0].assign(
            x_barrier_size, std::uint8_t{0});
        interface_edges[1].assign(
            y_barrier_size, std::uint8_t{0});
        interface_edges[2].assign(
            z_barrier_size, std::uint8_t{0});

        const std::vector<RationalBezierElement3D> leaves =
            intersector.acceleration_leaves(maximum_leaf_extent);
        if (leaves.size() != intersector.query_element_count()) {
            throw std::logic_error(
                "NURBS acceleration leaves differ from BVH query elements");
        }
        diagnostics.acceleration_leaf_count =
            checked_size_to_int(
                leaves.size(),
                "NURBS acceleration-leaf diagnostic count overflow");

        const auto origin = grid.origin();
        std::vector<std::uint64_t> candidate_keys;
        for (const RationalBezierElement3D& leaf : leaves) {
            const NurbsAabb3D leaf_bounds = leaf.bounds();
            for (int axis = 0; axis < 3; ++axis) {
                const double h = spacing[static_cast<std::size_t>(axis)];
                const ClosedIntInterval start_range =
                    candidate_index_interval(
                        leaf_bounds.lower[axis],
                        leaf_bounds.upper[axis],
                        origin[static_cast<std::size_t>(axis)],
                        h, tolerance, true,
                        dims[static_cast<std::size_t>(axis)] - 2);
                if (start_range.empty())
                    continue;

                std::array<int, 3> transverse_min{{0, 0, 0}};
                std::array<int, 3> transverse_max{{0, 0, 0}};
                bool empty = false;
                for (int transverse = 0; transverse < 3; ++transverse) {
                    if (transverse == axis)
                        continue;
                    const ClosedIntInterval transverse_range =
                        candidate_index_interval(
                            leaf_bounds.lower[transverse],
                            leaf_bounds.upper[transverse],
                            origin[static_cast<std::size_t>(transverse)],
                            spacing[static_cast<std::size_t>(transverse)],
                            tolerance, false,
                            dims[static_cast<std::size_t>(transverse)] - 1);
                    transverse_min[static_cast<std::size_t>(transverse)] =
                        transverse_range.lower;
                    transverse_max[static_cast<std::size_t>(transverse)] =
                        transverse_range.upper;
                    empty = empty || transverse_range.empty();
                }
                if (empty)
                    continue;

                if (axis == 0) {
                    for (int k = transverse_min[2];
                         k <= transverse_max[2]; ++k) {
                        for (int j = transverse_min[1];
                             j <= transverse_max[1]; ++j) {
                            for (int i = start_range.lower;
                                 i <= start_range.upper; ++i) {
                                candidate_keys.push_back(
                                    edge_key(
                                        axis,
                                        structured_node_index(i, j, k)));
                            }
                        }
                    }
                } else if (axis == 1) {
                    for (int k = transverse_min[2];
                         k <= transverse_max[2]; ++k) {
                        for (int j = start_range.lower;
                             j <= start_range.upper; ++j) {
                            for (int i = transverse_min[0];
                                 i <= transverse_max[0]; ++i) {
                                candidate_keys.push_back(
                                    edge_key(
                                        axis,
                                        structured_node_index(i, j, k)));
                            }
                        }
                    }
                } else {
                    for (int k = start_range.lower;
                         k <= start_range.upper; ++k) {
                        for (int j = transverse_min[1];
                             j <= transverse_max[1]; ++j) {
                            for (int i = transverse_min[0];
                                 i <= transverse_max[0]; ++i) {
                                candidate_keys.push_back(
                                    edge_key(
                                        axis,
                                        structured_node_index(i, j, k)));
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

        std::unordered_map<int, std::vector<int>> endpoint_membership_cache;
        const auto endpoint_membership = [&](int node) {
            const auto found = endpoint_membership_cache.find(node);
            if (found != endpoint_membership_cache.end())
                return found->second;
            std::vector<int> containing = intersector.containing_components(
                as_vector(grid.coord(node)));
            std::sort(containing.begin(), containing.end());
            endpoint_membership_cache.emplace(node, containing);
            diagnostics.endpoint_classification_query_count = checked_size_add(
                diagnostics.endpoint_classification_query_count,
                std::size_t{1},
                "NURBS endpoint-classification diagnostic overflow");
            return containing;
        };

        for (const std::uint64_t key : candidate_keys) {
            const std::uint64_t axis_bits = key >> 62;
            if (axis_bits > 2)
                throw std::overflow_error("invalid Cartesian edge key axis");
            const int axis = static_cast<int>(axis_bits);
            const std::uint64_t start_node_bits = key & kNodeMask;
            if (start_node_bits > static_cast<std::uint64_t>(
                    std::numeric_limits<int>::max())) {
                throw std::overflow_error(
                    "Cartesian edge-key node exceeds int range");
            }
            const int start_node = static_cast<int>(start_node_bits);
            const auto ijk = node_coordinates(start_node);
            const int axis_stride =
                axis == 0 ? 1 : (axis == 1 ? nx : plane_stride);
            const int end_node = checked_add_int(
                start_node, axis_stride,
                "Cartesian edge endpoint index overflow");
            if (end_node >= node_count) {
                throw std::overflow_error(
                    "Cartesian edge endpoint is outside checked node storage");
            }
            const Eigen::Vector3d start = as_vector(grid.coord(start_node));
            const Eigen::Vector3d end = as_vector(grid.coord(end_node));
            const NurbsCartesianEdgeIntersections3D edge_result =
                intersector.intersect_cartesian_edge(
                    NurbsCartesianEdgeQuery3D{
                        axis, ijk[0], ijk[1], ijk[2], start, end});
            accumulate_intersection_diagnostics(
                diagnostics.intersections, edge_result.diagnostics);
            std::vector<int> toggled_components;
            bool changes_inside_outside = false;
            bool changes_component_membership = false;
            if (edge_result.parity_known_from_roots) {
                toggled_components = edge_result.toggled_components;
                changes_inside_outside =
                    edge_result.changes_inside_outside;
                changes_component_membership =
                    edge_result.changes_component_membership;
            } else {
                diagnostics.ambiguous_parity_edge_count = checked_size_add(
                    diagnostics.ambiguous_parity_edge_count,
                    std::size_t{1},
                    "NURBS ambiguous-parity diagnostic overflow");
                diagnostics.endpoint_parity_fallback_count = checked_size_add(
                    diagnostics.endpoint_parity_fallback_count,
                    std::size_t{1},
                    "NURBS endpoint-fallback diagnostic overflow");
                const std::vector<int> start_membership =
                    endpoint_membership(start_node);
                const std::vector<int> end_membership =
                    endpoint_membership(end_node);
                changes_inside_outside =
                    start_membership.empty() != end_membership.empty();
                changes_component_membership =
                    start_membership != end_membership;
                std::set_symmetric_difference(
                    start_membership.begin(), start_membership.end(),
                    end_membership.begin(), end_membership.end(),
                    std::back_inserter(toggled_components));
            }
            diagnostics.component_parity_toggle_count = checked_size_add(
                diagnostics.component_parity_toggle_count,
                toggled_components.size(),
                "NURBS component-parity diagnostic overflow");

            const std::size_t storage_index =
                barrier_index(axis, ijk[0], ijk[1], ijk[2]);
            if (!edge_result.crossings.empty()) {
                interface_edges[static_cast<std::size_t>(axis)]
                               [storage_index] = std::uint8_t{1};
                std::size_t& interface_count =
                    diagnostics.interface_edge_counts[
                        static_cast<std::size_t>(axis)];
                interface_count = checked_size_add(
                    interface_count, std::size_t{1},
                    "NURBS interface diagnostic count overflow");
                if (edge_result.crossings.size() > 1) {
                    diagnostics.multi_crossing_edge_count = checked_size_add(
                        diagnostics.multi_crossing_edge_count,
                        std::size_t{1},
                        "NURBS multi-crossing diagnostic overflow");
                }
                if (edge_result.parity_known_from_roots) {
                    std::size_t& parity_count =
                        edge_result.confirmed_transverse_count % 2 == 0
                        ? diagnostics.even_parity_interface_edge_count
                        : diagnostics.odd_parity_interface_edge_count;
                    parity_count = checked_size_add(
                        parity_count, std::size_t{1},
                        "NURBS interface-parity diagnostic overflow");
                }

                EdgeCrossingRange3D range;
                range.begin = checked_size_to_int(
                    crossings.size(),
                    "NURBS crossing range begin exceeds int range");
                range.count = checked_size_to_int(
                    edge_result.crossings.size(),
                    "NURBS crossing range count exceeds int range");
                range.confirmed_transverse_count =
                    edge_result.confirmed_transverse_count;
                range.changes_inside_outside = changes_inside_outside;
                range.changes_component_membership =
                    changes_component_membership;
                for (const NurbsSurfaceCrossing3D& crossing :
                     edge_result.crossings) {
                    diagnostics.maximum_root_residual = std::max(
                        diagnostics.maximum_root_residual,
                        crossing.residual);
                    crossings.push_back(crossing);
                }
                if (!crossing_ranges_by_edge.emplace(key, range).second) {
                    throw std::logic_error(
                        "duplicate NURBS Cartesian crossing range");
                }
            }

            if (changes_component_membership) {
                barriers[static_cast<std::size_t>(axis)][storage_index] =
                    std::uint8_t{1};
                std::size_t& barrier_count =
                    diagnostics.barrier_edge_counts[
                        static_cast<std::size_t>(axis)];
                barrier_count = checked_size_add(
                    barrier_count, std::size_t{1},
                    "NURBS barrier diagnostic count overflow");
            }
        }

        flood_and_label(grid, intersector);
        verify_barriers();
    }

    bool is_compatible_grid(const CartesianGrid3D& grid) const noexcept
    {
        return grid.layout() == grid_layout
            && grid.num_cells() == grid_cells
            && grid.origin() == grid_origin
            && grid.spacing() == grid_spacing;
    }

    std::array<int, 3> node_coordinates(int node) const
    {
        const int nx = dims[0];
        const int k = node / plane_stride;
        const int remainder = node % plane_stride;
        return {{remainder % nx, remainder / nx, k}};
    }

    int structured_node_index(int i, int j, int k) const
    {
        const std::size_t plane_offset = checked_size_product(
            static_cast<std::size_t>(k),
            static_cast<std::size_t>(plane_stride),
            "Cartesian structured node index overflow");
        const std::size_t row_offset = checked_size_product(
            static_cast<std::size_t>(j),
            static_cast<std::size_t>(dims[0]),
            "Cartesian structured node index overflow");
        const std::size_t index = checked_size_add(
            checked_size_add(
                plane_offset, row_offset,
                "Cartesian structured node index overflow"),
            static_cast<std::size_t>(i),
            "Cartesian structured node index overflow");
        if (index >= static_cast<std::size_t>(node_count)) {
            throw std::overflow_error(
                "Cartesian structured node index is outside checked storage");
        }
        return checked_size_to_int(
            index, "Cartesian structured node index exceeds int range");
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
    bool interface_at(int axis, int i, int j, int k) const
    {
        return interface_edges[static_cast<std::size_t>(axis)]
                              [barrier_index(axis, i, j, k)] != 0;
    }

    std::pair<int, int> adjacent_edge(int node_a, int node_b) const
    {
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
    bool has_interface(int node_a, int node_b) const
    {
        const auto edge = adjacent_edge(node_a, node_b);
        const auto ijk = node_coordinates(edge.second);
        return interface_at(edge.first, ijk[0], ijk[1], ijk[2]);
    }

    void flood_and_label(const CartesianGrid3D& grid,
                         const NurbsSurfaceIntersector3D& intersector)
    {
        const int nx = dims[0];
        const int ny = dims[1];
        const int nz = dims[2];
        std::vector<int> graph_component(
            static_cast<std::size_t>(node_count), -1);
        std::vector<int> representatives;
        std::vector<bool> touches_box_boundary;
        std::vector<int> queue;

        for (int seed = 0; seed < node_count; ++seed) {
            if (graph_component[static_cast<std::size_t>(seed)] >= 0)
                continue;
            const int component = checked_size_to_int(
                representatives.size(),
                "Cartesian grid-component index exceeds int range");
            representatives.push_back(seed);
            touches_box_boundary.push_back(false);
            diagnostics.component_sizes.push_back(0);
            queue.clear();
            queue.push_back(seed);
            graph_component[static_cast<std::size_t>(seed)] = component;

            for (std::size_t head = 0; head < queue.size(); ++head) {
                const int node = queue[head];
                std::size_t& component_size =
                    diagnostics.component_sizes[
                        static_cast<std::size_t>(component)];
                component_size = checked_size_add(
                    component_size, std::size_t{1},
                    "Cartesian component-size diagnostic overflow");
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
                        ? node - plane_stride : -1,
                    k + 1 < nz && !barrier_at(2, i, j, k)
                        ? node + plane_stride : -1}};
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

        diagnostics.grid_component_count = checked_size_to_int(
            representatives.size(),
            "Cartesian grid-component diagnostic count overflow");
        std::vector<int> component_labels(representatives.size(), 0);
        for (std::size_t component = 0;
             component < representatives.size(); ++component) {
            if (touches_box_boundary[component]) {
                component_labels[component] = 0;
                diagnostics.box_exterior_component_count = checked_add_int(
                    diagnostics.box_exterior_component_count, 1,
                    "Cartesian box-exterior diagnostic count overflow");
            } else {
                diagnostics.representative_query_count = checked_add_int(
                    diagnostics.representative_query_count, 1,
                    "Cartesian representative-query diagnostic count overflow");
                const Eigen::Vector3d representative_point = as_vector(
                    grid.coord(representatives[component]));
                const std::vector<int> containing =
                    intersector.containing_components(
                        representative_point);
                if (containing.empty()) {
                    component_labels[component] = 0;
                } else if (containing.size() == 1) {
                    component_labels[component] =
                        checked_add_int(
                            containing.front(), 1,
                            "NURBS component label exceeds int range");
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
        const bool interface_edge = interface_at(axis, i, j, k);
        const int first_label =
            node_labels[static_cast<std::size_t>(first)];
        const int second_label =
            node_labels[static_cast<std::size_t>(second)];
        const bool component_label_change = first_label != second_label;
        const bool binary_change = (first_label > 0) != (second_label > 0);
        if (component_label_change != barrier) {
            throw std::runtime_error(
                "NURBS component label change disagrees with Cartesian barrier");
        }

        const std::uint64_t key = edge_key(axis, first);
        const auto found = crossing_ranges_by_edge.find(key);
        const bool has_range = found != crossing_ranges_by_edge.end();
        if (interface_edge != has_range) {
            throw std::runtime_error(
                "NURBS interface edge disagrees with crossing range");
        }
        if (!has_range)
            return;

        const EdgeCrossingRange3D& range = found->second;
        if (range.begin < 0 || range.count <= 0
            || static_cast<std::size_t>(range.begin)
                   > crossings.size()
            || static_cast<std::size_t>(range.count)
                   > crossings.size()
                       - static_cast<std::size_t>(range.begin)) {
            throw std::runtime_error(
                "NURBS Cartesian crossing range is invalid");
        }
        if (range.changes_component_membership != barrier
            || range.changes_inside_outside != binary_change) {
            throw std::runtime_error(
                "NURBS crossing parity disagrees with endpoint labels");
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
                    const int node = structured_node_index(i, j, k);
                    if (i + 1 < nx)
                        verify_edge(0, i, j, k, node, node + 1);
                    if (j + 1 < ny)
                        verify_edge(1, i, j, k, node, node + nx);
                    if (k + 1 < nz)
                        verify_edge(
                            2, i, j, k, node, node + plane_stride);
                }
            }
        }
    }
};

NurbsSurfaceCrossingRange3D::NurbsSurfaceCrossingRange3D(
    const NurbsSurfaceCrossing3D* data, std::size_t size) noexcept
    : data_(data)
    , size_(size)
{}

const NurbsSurfaceCrossing3D*
NurbsSurfaceCrossingRange3D::begin() const noexcept
{
    return data_;
}

const NurbsSurfaceCrossing3D*
NurbsSurfaceCrossingRange3D::end() const noexcept
{
    return data_ == nullptr ? nullptr : data_ + size_;
}

std::size_t NurbsSurfaceCrossingRange3D::size() const noexcept
{
    return size_;
}

const NurbsSurfaceCrossing3D&
NurbsSurfaceCrossingRange3D::operator[](std::size_t index) const
{
    if (index >= size_)
        throw std::out_of_range("NURBS crossing range index is out of bounds");
    return data_[index];
}

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
        || node >= impl_->node_count) {
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

bool NurbsCartesianDomain3D::has_interface_between(
    int node_a, int node_b) const
{
    return impl_->has_interface(node_a, node_b);
}

NurbsSurfaceCrossingRange3D NurbsCartesianDomain3D::crossings_between(
    int node_a, int node_b) const
{
    const auto edge = impl_->adjacent_edge(node_a, node_b);
    const auto found = impl_->crossing_ranges_by_edge.find(
        edge_key(edge.first, edge.second));
    if (found == impl_->crossing_ranges_by_edge.end())
        return {};
    const EdgeCrossingRange3D& range = found->second;
    return NurbsSurfaceCrossingRange3D(
        impl_->crossings.data() + range.begin,
        static_cast<std::size_t>(range.count));
}

const NurbsSurfaceCrossing3D& NurbsCartesianDomain3D::crossing_between(
    int node_a, int node_b) const
{
    const NurbsSurfaceCrossingRange3D range =
        crossings_between(node_a, node_b);
    if (range.size() == 0) {
        throw std::runtime_error(
            "no NURBS crossing between Cartesian nodes");
    }
    if (range.size() != 1) {
        throw std::runtime_error(
            "NURBS crossing lookup requires exactly one crossing");
    }
    return range[0];
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

bool NurbsCartesianDomain3D::is_compatible_grid(
    const CartesianGrid3D& grid) const noexcept
{
    return impl_->is_compatible_grid(grid);
}

} // namespace kfbim::geometry3d
