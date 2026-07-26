#include <apps/harmonic_cauchy_fit_3d.hpp>
#include <apps/native_nurbs_surface_transform_3d.hpp>
#include <src/geometry/nurbs_patch_triangulator_3d.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace {

using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::NativeNurbsSurface3D;
using kfbim::app3d::RigidTransform3D;
using kfbim::app3d::SurfaceDofCloud3D;
using kfbim::app3d::build_surface_non_g1_edge_neighborhoods_3d;
using kfbim::app3d::make_native_nurbs_surface_3d;
using kfbim::app3d::make_native_surface_dofs_3d;
using kfbim::app3d::make_shared_edge_points_3d;
using kfbim::app3d::nearest_g1_cauchy_dofs;
using kfbim::app3d::select_direct_cross_face_value_dofs_3d;
using kfbim::app3d::select_surface_edge_points_3d;
using kfbim::app3d::transform_native_nurbs_surface_3d;
using kfbim::geometry3d::NurbsPatchEdge3D;
using kfbim::geometry3d::NurbsPatchEdgeConnection3D;
using kfbim::geometry3d::NurbsSurfacePatch3D;
using kfbim::geometry3d::closest_point_to_nurbs_patch_edge_interval_3d;
using kfbim::geometry3d::estimate_nurbs_patch_edge_interval_length_3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

NativeNurbsSurface3D reversed_unit_wedge()
{
    NativeNurbsSurface3D surface;
    surface.name = "reversed_unit_wedge";
    surface.description = "two planes sharing a reversed non-G1 edge";
    surface.patches = {
        NurbsSurfacePatch3D::make_bilinear_plane(
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}),
        NurbsSurfacePatch3D::make_bilinear_plane(
            {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
            {1.0, 0.0, -1.0}, {0.0, 0.0, -1.0})};
    surface.patch_names = {"horizontal", "vertical_reversed"};
    surface.smooth_neighbors.resize(2);
    surface.topological_patch_neighbors = {{1}, {0}};
    surface.geometric_connections = {
        {{0, NurbsPatchEdge3D::VMin, 0.0, 1.0},
         {1, NurbsPatchEdge3D::VMin, 0.0, 1.0},
         true,
         false}};
    surface.patch_components = {0, 0};
    surface.expected_area = 2.0;
    surface.exact_inside = [](const Eigen::Vector3d&) { return false; };
    return surface;
}

void test_interval_length_uses_requested_native_subinterval()
{
    // Catches ignoring begin/end or changing the exact unit-speed integral.
    const NativeNurbsSurface3D surface = reversed_unit_wedge();
    const auto& patch = surface.patches[0];
    require(std::abs(estimate_nurbs_patch_edge_interval_length_3d(
                         patch, NurbsPatchEdge3D::VMin, 0.0, 1.0)
                     - 1.0)
                < 1.0e-14,
            "full unit edge has literal length 1");
    require(std::abs(estimate_nurbs_patch_edge_interval_length_3d(
                         patch, NurbsPatchEdge3D::VMin, 0.25, 0.75)
                     - 0.5)
                < 1.0e-14,
            "middle half of unit edge has literal length 0.5");
}

void test_reversed_shared_edge_points_use_cell_midpoints_and_mapped_frames()
{
    // Catches endpoint quadrature, missing reversed parameter/sign mapping,
    // merged G1 sectors, and non-orthonormal or left-handed edge frames.
    const auto points = make_shared_edge_points_3d(reversed_unit_wedge(), 0.2);
    require(points.points.size() == 5, "unit edge at h=.2 has five cells");
    require(points.point_ids_by_connection.size() == 1,
            "one feature connection has one point-id group");
    const std::array<double, 5> expected_fraction{
        {0.1, 0.3, 0.5, 0.7, 0.9}};
    for (int i = 0; i < 5; ++i) {
        const auto& point = points.points[static_cast<std::size_t>(i)];
        const double s = expected_fraction[static_cast<std::size_t>(i)];
        require(point.id == i && point.connection_id == 0
                    && point.cell_id == i && point.cell_count == 5,
                "shared point IDs follow literal connection/cell order");
        require(std::abs(point.fraction - s) < 1.0e-14,
                "shared point fraction is the cell midpoint");
        require(std::abs(point.native_parameters[0] - s) < 1.0e-14,
                "first native parameter increases with cell fraction");
        require(std::abs(point.native_parameters[1] - (1.0 - s)) < 1.0e-14,
                "reversed second native parameter decreases");
        require((point.point - Eigen::Vector3d(s, 0.0, 0.0)).norm()
                    < 1.0e-12,
                "shared point lies at the hand-derived physical position");
        require(std::abs(point.quadrature_weight - 0.2) < 1.0e-14,
                "unit edge cell weight is the literal .2");
        require(point.sector_patch_ids[0] == std::vector<int>{0}
                    && point.sector_patch_ids[1] == std::vector<int>{1},
                "non-G1 incident patches remain distinct sectors");
        require(std::abs(point.frame.col(0).norm() - 1.0) < 1.0e-12
                    && std::abs(point.frame.col(1).norm() - 1.0) < 1.0e-12
                    && std::abs(point.frame.col(2).norm() - 1.0) < 1.0e-12,
                "edge frame columns are unit length");
        require(std::abs(point.frame.col(0).dot(point.frame.col(1))) < 1.0e-12
                    && std::abs(point.frame.col(0).dot(point.frame.col(2)))
                           < 1.0e-12
                    && std::abs(point.frame.col(1).dot(point.frame.col(2)))
                           < 1.0e-12,
                "edge frame columns are mutually orthogonal");
        require(point.frame.determinant() > 1.0 - 1.0e-12,
                "edge frame is right handed");
    }
    require(points.max_position_mismatch < 1.0e-12,
            "mapped physical points agree");
    require(points.min_mapped_tangent_dot > 1.0 - 1.0e-10,
            "mapped unit tangents agree after reversal");
}

const NurbsPatchEdgeConnection3D& lprism_patch6_vmin_split(
    const NativeNurbsSurface3D& surface,
    double begin)
{
    const auto found = std::find_if(
        surface.geometric_connections.begin(),
        surface.geometric_connections.end(),
        [begin](const NurbsPatchEdgeConnection3D& connection) {
            return !connection.g1
                && connection.first.patch == 6
                && connection.first.edge == NurbsPatchEdge3D::VMin
                && connection.first.begin == begin
                && connection.first.end == begin + 0.5;
        });
    require(found != surface.geometric_connections.end(),
            "L-prism fixture has requested patch-6 split");
    return *found;
}

void test_lprism_partial_interval_closest_point_clamps_and_ties()
{
    // Catches projecting to a full edge instead of each partial connection,
    // omitting endpoints, or choosing one split at a shared endpoint by noise.
    const NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const auto& patch = surface.patches[6];
    const auto& lower = lprism_patch6_vmin_split(surface, 0.0);
    const auto& upper = lprism_patch6_vmin_split(surface, 0.5);
    const auto at_quarter = patch.evaluate_with_derivatives(0.25, 0.0);
    const Eigen::Vector3d normal =
        at_quarter.du.cross(at_quarter.dv).normalized();
    const Eigen::Vector3d query = at_quarter.point + 0.1 * normal;
    const auto lower_result = closest_point_to_nurbs_patch_edge_interval_3d(
        patch, lower.first.edge, lower.first.begin, lower.first.end, query, 3.0);
    const auto upper_result = closest_point_to_nurbs_patch_edge_interval_3d(
        patch, upper.first.edge, upper.first.begin, upper.first.end, query, 3.0);
    require(lower_result.converged
                && std::abs(lower_result.parameter - 0.25) < 1.0e-12
                && std::abs(lower_result.distance - 0.1) < 1.0e-12,
            "quarter query projects to literal .25 at distance .1");
    require(upper_result.converged
                && std::abs(upper_result.parameter - 0.5) < 1.0e-12,
            "other partial interval clamps to literal endpoint .5");

    const auto at_split = patch.evaluate_with_derivatives(0.5, 0.0);
    const Eigen::Vector3d split_normal =
        at_split.du.cross(at_split.dv).normalized();
    const Eigen::Vector3d split_query = at_split.point + 0.1 * split_normal;
    const auto from_lower = closest_point_to_nurbs_patch_edge_interval_3d(
        patch, NurbsPatchEdge3D::VMin, 0.0, 0.5, split_query, 3.0);
    const auto from_upper = closest_point_to_nurbs_patch_edge_interval_3d(
        patch, NurbsPatchEdge3D::VMin, 0.5, 1.0, split_query, 3.0);
    require(std::abs(from_lower.parameter - 0.5) < 1.0e-12
                && std::abs(from_upper.parameter - 0.5) < 1.0e-12,
            "both partial intervals retain their shared endpoint");
    require(std::abs(from_lower.distance - from_upper.distance) <= 3.0e-12,
            "split connections tie within literal physical tolerance");
}

Eigen::Vector3d analytic_quarter_circle_point(double radius, double t)
{
    const double w = std::sqrt(0.5);
    const double a = 1.0 - t;
    const double denominator = a * a + 2.0 * w * a * t + t * t;
    return {radius * (a * a + 2.0 * w * a * t) / denominator,
            radius * (2.0 * w * a * t + t * t) / denominator,
            -0.63};
}

void test_circular_edge_closest_point_is_rigid_transform_invariant()
{
    // Catches chord projection on a rational edge and world-axis-dependent
    // optimization. Literal native parameter is .25 and radial offset .1.
    constexpr double radius = 0.55;
    constexpr double parameter = 0.25;
    const NurbsSurfacePatch3D cylinder =
        NurbsSurfacePatch3D::make_quarter_cylinder_patch(
            radius, -0.63, 0.67);
    const Eigen::Vector3d circle_point =
        analytic_quarter_circle_point(radius, parameter);
    const Eigen::Vector3d radial =
        Eigen::Vector3d(circle_point.x(), circle_point.y(), 0.0).normalized();
    const Eigen::Vector3d query = circle_point + 0.1 * radial;
    const auto original = closest_point_to_nurbs_patch_edge_interval_3d(
        cylinder, NurbsPatchEdge3D::VMin, 0.0, 1.0, query, 2.0);
    require(original.converged
                && std::abs(original.parameter - 0.25) < 2.0e-12
                && std::abs(original.distance - 0.1) < 2.0e-12,
            "circular edge returns analytic parameter and distance");

    NativeNurbsSurface3D one_patch;
    one_patch.name = "quarter_cylinder";
    one_patch.description = "rigid closest-point fixture";
    one_patch.patches = {cylinder};
    one_patch.patch_names = {"quarter"};
    one_patch.smooth_neighbors.resize(1);
    one_patch.topological_patch_neighbors.resize(1);
    one_patch.patch_components = {0};
    one_patch.exact_inside = [](const Eigen::Vector3d&) { return false; };
    const RigidTransform3D transform = RigidTransform3D::from_axis_angle(
        Eigen::Vector3d(1.0, 2.0, -1.0),
        0.37,
        Eigen::Vector3d(0.2, -0.1, 0.3),
        Eigen::Vector3d(-0.4, 0.25, 0.15));
    const NativeNurbsSurface3D moved =
        transform_native_nurbs_surface_3d(one_patch, transform);
    const auto transformed = closest_point_to_nurbs_patch_edge_interval_3d(
        moved.patches[0], NurbsPatchEdge3D::VMin, 0.0, 1.0,
        transform.forward_point(query), 2.0);
    require(transformed.converged
                && std::abs(transformed.parameter - 0.25) < 2.0e-12
                && std::abs(transformed.distance - 0.1) < 2.0e-12,
            "fixed rigid transform preserves analytic closest point");
}

int connection_id_matching(
    const NativeNurbsSurface3D& surface,
    int first_patch,
    NurbsPatchEdge3D first_edge,
    double first_begin,
    bool reversed)
{
    for (int id = 0;
         id < static_cast<int>(surface.geometric_connections.size());
         ++id) {
        const auto& connection =
            surface.geometric_connections[static_cast<std::size_t>(id)];
        if (!connection.g1 && connection.first.patch == first_patch
            && connection.first.edge == first_edge
            && connection.first.begin == first_begin
            && connection.reversed == reversed) {
            return id;
        }
    }
    throw std::runtime_error("literal native connection fixture is missing");
}

void test_lprism_split_edges_have_disjoint_midpoint_ids_and_reversed_parameters()
{
    // Catches assigning endpoints/duplicate IDs across partial connections,
    // using the whole edge length for each split, or ignoring reversed maps.
    constexpr double h = 3.0 / 32.0;
    const NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const auto points = make_shared_edge_points_3d(surface, h);
    const int lower_id = connection_id_matching(
        surface, 6, NurbsPatchEdge3D::VMin, 0.0, false);
    const int upper_id = connection_id_matching(
        surface, 6, NurbsPatchEdge3D::VMin, 0.5, false);
    const auto& lower =
        points.point_ids_by_connection[static_cast<std::size_t>(lower_id)];
    const auto& upper =
        points.point_ids_by_connection[static_cast<std::size_t>(upper_id)];
    require(lower.size() == 7 && upper.size() == 7,
            "each literal .6 L-prism split has ceil(.6/(3/32)) = 7 points");
    std::set<int> split_ids(lower.begin(), lower.end());
    split_ids.insert(upper.begin(), upper.end());
    require(split_ids.size() == 14,
            "two partial connections have fourteen disjoint IDs");
    for (int id : split_ids) {
        require(std::abs(points.points[static_cast<std::size_t>(id)]
                             .native_parameters[0]
                         - 0.5)
                    > 1.0e-14,
                "cell-midpoint quadrature never places a point at split .5");
    }

    const int reversed_id = connection_id_matching(
        surface, 11, NurbsPatchEdge3D::VMin, 0.5, true);
    const auto& reversed = points.point_ids_by_connection[
        static_cast<std::size_t>(reversed_id)];
    require(reversed.size() == 7,
            "reversed literal .6 split also has seven points");
    for (std::size_t q = 1; q < reversed.size(); ++q) {
        require(points.points[static_cast<std::size_t>(reversed[q])]
                        .native_parameters[1]
                    < points.points[static_cast<std::size_t>(reversed[q - 1])]
                          .native_parameters[1],
                "reversed connection second native parameters strictly decrease");
    }
}

bool unique_valid_ids(const std::vector<int>& ids,
                      const SurfaceDofCloud3D& cloud)
{
    const std::set<int> unique(ids.begin(), ids.end());
    return unique.size() == ids.size()
        && std::all_of(ids.begin(), ids.end(), [&](int id) {
               return id >= 0 && id < static_cast<int>(cloud.dofs.size());
           });
}

bool contains_patch(const std::vector<int>& ids,
                    const SurfaceDofCloud3D& cloud,
                    int patch)
{
    return std::any_of(ids.begin(), ids.end(), [&](int id) {
        return cloud.dofs[static_cast<std::size_t>(id)].patch_id == patch;
    });
}

std::map<int, int> counts_by_sector(
    const std::vector<int>& ids,
    const SurfaceDofCloud3D& cloud,
    const std::vector<std::vector<int>>& sectors)
{
    std::map<int, int> counts;
    for (int id : ids) {
        const int patch = cloud.dofs[static_cast<std::size_t>(id)].patch_id;
        int owner = -1;
        for (int sector = 0; sector < static_cast<int>(sectors.size()); ++sector) {
            if (std::find(sectors[static_cast<std::size_t>(sector)].begin(),
                          sectors[static_cast<std::size_t>(sector)].end(),
                          patch)
                != sectors[static_cast<std::size_t>(sector)].end()) {
                require(owner < 0, "deduplicated sectors have disjoint patches");
                owner = sector;
            }
        }
        require(owner >= 0, "every selected DOF belongs to an admitted sector");
        ++counts[owner];
    }
    return counts;
}

void test_direct_selector_balances_regular_and_three_sector_centers()
{
    // Catches patch-wise rather than G1-sector balancing, wrong remainder
    // order, lost center IDs, duplicate IDs, and missed split-edge ties.
    constexpr double h = 3.0 / 32.0;
    const NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const auto neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(surface, cloud, h);
    const auto& side = cloud.patches[6];

    const int regular_center = side.dof_index(3, 0);
    const auto regular = select_direct_cross_face_value_dofs_3d(
        surface, cloud, neighborhoods, regular_center, 48, h);
    require(regular.dof_ids.size() == 48
                && unique_valid_ids(regular.dof_ids, cloud),
            "regular cross-face center returns 48 unique valid IDs");
    require(regular.sector_patch_ids.size() == 2
                && regular.sector_sample_counts == std::vector<int>({24, 24}),
            "regular cross-face center splits literal 48 as 24/24");
    require(std::find(regular.dof_ids.begin(), regular.dof_ids.end(),
                      regular_center)
                != regular.dof_ids.end(),
            "regular direct selection explicitly retains its center");

    const int split_center = side.dof_index(side.nu / 2, 0);
    const auto& split_neighborhood = neighborhoods.centers[
        static_cast<std::size_t>(split_center)];
    require(split_neighborhood.relevant_connection_ids.size() == 2,
            "native .5 center ties both partial patch-6 VMin connections");

    const int feature_center = side.dof_index(0, 0);
    const auto feature = select_direct_cross_face_value_dofs_3d(
        surface, cloud, neighborhoods, feature_center, 50, h);
    require(feature.dof_ids.size() == 50
                && unique_valid_ids(feature.dof_ids, cloud)
                && feature.sector_patch_ids.size() == 3,
            "feature-vertex direct selection has 50 unique IDs in three sectors");
    const auto counts = counts_by_sector(
        feature.dof_ids, cloud, feature.sector_patch_ids);
    int minimum = 50;
    int maximum = 0;
    for (const auto& item : counts) {
        minimum = std::min(minimum, item.second);
        maximum = std::max(maximum, item.second);
    }
    require(counts.size() == 3 && maximum - minimum <= 1,
            "three-sector sample counts differ by at most one");
    int center_sector = -1;
    int lowest_other_sector = -1;
    int lowest_other_patch = std::numeric_limits<int>::max();
    for (int sector = 0;
         sector < static_cast<int>(feature.sector_patch_ids.size());
         ++sector) {
        const auto& patches = feature.sector_patch_ids[
            static_cast<std::size_t>(sector)];
        if (std::find(patches.begin(), patches.end(), 6) != patches.end()) {
            center_sector = sector;
        } else if (!patches.empty() && patches.front() < lowest_other_patch) {
            lowest_other_patch = patches.front();
            lowest_other_sector = sector;
        }
    }
    require(center_sector >= 0 && lowest_other_sector >= 0,
            "feature fixture identifies center and ascending remainder sectors");
    require(counts.at(center_sector) == 17
                && counts.at(lowest_other_sector) == 17,
            "50/3 remainders go to center sector then lowest patch-ID sector");
    require(std::find(feature.dof_ids.begin(), feature.dof_ids.end(),
                      feature_center)
                != feature.dof_ids.end(),
            "feature-vertex direct selection retains its center");
}

NativeNurbsSurface3D wedge_with_unrelated_coincident_patch()
{
    NativeNurbsSurface3D surface = reversed_unit_wedge();
    surface.patches.push_back(surface.patches[1]);
    surface.patch_names.push_back("unrelated_coincident");
    surface.smooth_neighbors.resize(3);
    surface.topological_patch_neighbors.resize(3);
    surface.patch_components.push_back(0);
    surface.expected_area = 3.0;
    return surface;
}

void test_direct_selector_uses_topology_not_physical_proximity_and_g1_outside_band()
{
    // Catches global Euclidean admission of an unrelated coincident patch and
    // any change to the baseline element-for-element G1 route outside 2h.
    const NativeNurbsSurface3D coincident =
        wedge_with_unrelated_coincident_patch();
    const SurfaceDofCloud3D coincident_cloud =
        make_native_surface_dofs_3d(coincident, 0.2);
    const auto coincident_neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(
            coincident, coincident_cloud, 0.2);
    const int edge_center = coincident_cloud.patches[0].dof_index(2, 0);
    const auto selection = select_direct_cross_face_value_dofs_3d(
        coincident, coincident_cloud, coincident_neighborhoods,
        edge_center, 48, 0.2);
    require(!contains_patch(selection.dof_ids, coincident_cloud, 2),
            "topology-unrelated coincident patch is never admitted");

    constexpr double h = 3.0 / 32.0;
    const NativeNurbsSurface3D lprism =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(lprism, h);
    const auto neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(lprism, cloud, h);
    const auto& side = cloud.patches[6];
    const int interior = side.dof_index(side.nu / 2, side.nv / 2);
    require(neighborhoods.centers[static_cast<std::size_t>(interior)]
                .nearest_distance_over_h > 2.0,
            "literal side interior lies outside the 2h edge band");
    const auto direct = select_direct_cross_face_value_dofs_3d(
        lprism, cloud, neighborhoods, interior, 48, h);
    const std::vector<int> baseline =
        nearest_g1_cauchy_dofs(lprism, cloud, interior, 48);
    require(direct.dof_ids == baseline,
            "outside 2h direct IDs equal nearest_g1_cauchy_dofs in order");
}

void test_edge_selector_contributes_each_connection_and_is_deterministic()
{
    // Catches repeating geometry queries in selectors, omitting a relevant
    // edge, accepting a fifth point, accepting distant extras, or unstable IDs.
    constexpr double h = 3.0 / 32.0;
    const NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const auto first_points = make_shared_edge_points_3d(surface, h);
    const auto second_points = make_shared_edge_points_3d(surface, h);
    const auto first_neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(surface, cloud, h);
    const auto second_neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(surface, cloud, h);
    const auto& side = cloud.patches[6];
    const int center = side.dof_index(0, 0);
    const auto first = select_surface_edge_points_3d(
        surface, cloud, first_points, first_neighborhoods, center, h);
    const auto second = select_surface_edge_points_3d(
        surface, cloud, second_points, second_neighborhoods, center, h);
    require(first_points.fingerprint == second_points.fingerprint
                && first_neighborhoods.fingerprint
                       == second_neighborhoods.fingerprint
                && first_neighborhoods.geometry_query_count
                       == second_neighborhoods.geometry_query_count
                && first.edge_point_ids == second.edge_point_ids
                && first.relevant_connection_ids
                       == second.relevant_connection_ids,
            "repeated builds and edge selections are bitwise deterministic");
    const Eigen::Vector3d center_point =
        cloud.dofs[static_cast<std::size_t>(center)].point;
    std::map<int, std::vector<int>> selected_by_connection;
    for (int id : first.edge_point_ids) {
        const auto& point = first_points.points[static_cast<std::size_t>(id)];
        selected_by_connection[point.connection_id].push_back(id);
    }
    require(first.relevant_connection_ids
                == first_neighborhoods.centers[static_cast<std::size_t>(center)]
                       .relevant_connection_ids,
            "edge selector consumes cached relevant connection IDs");
    for (int connection_id : first.relevant_connection_ids) {
        const auto& selected = selected_by_connection[connection_id];
        require(!selected.empty() && selected.size() <= 4,
                "each relevant connection contributes one through four points");
        const auto& all = first_points.point_ids_by_connection[
            static_cast<std::size_t>(connection_id)];
        const int nearest = *std::min_element(
            all.begin(), all.end(), [&](int a, int b) {
                const double da = (first_points.points[static_cast<std::size_t>(a)]
                                       .point - center_point).squaredNorm();
                const double db = (first_points.points[static_cast<std::size_t>(b)]
                                       .point - center_point).squaredNorm();
                return da != db ? da < db : a < b;
            });
        require(std::find(selected.begin(), selected.end(), nearest)
                    != selected.end(),
                "each relevant connection contributes its nearest point");
        for (int id : selected) {
            if (id == nearest)
                continue;
            require((first_points.points[static_cast<std::size_t>(id)].point
                        - center_point).norm() <= 2.0 * h + 3.0e-12,
                    "only the nearest point may be outside the literal 2h radius");
        }
    }
}

void test_hollow_cylinder_periodic_g1_sectors_are_admitted_without_other_sheets()
{
    // Catches using only immediate patch neighbors instead of transitive G1
    // sectors, and accidental jumps to the physically nearby inner/bottom sheets.
    constexpr double h = 3.0 / 32.0;
    const NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const auto neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(surface, cloud, h);
    const auto& outer = cloud.patches[0];
    const int center = outer.dof_index(0, outer.nv - 1);
    const auto selection = select_direct_cross_face_value_dofs_3d(
        surface, cloud, neighborhoods, center, 48, h);
    require(selection.sector_patch_ids.size() == 2,
            "outer/top sharp circle admits exactly two G1 sectors");
    require(std::find(selection.sector_patch_ids.begin(),
                      selection.sector_patch_ids.end(),
                      std::vector<int>({0, 1, 2, 3}))
                != selection.sector_patch_ids.end()
                && std::find(selection.sector_patch_ids.begin(),
                             selection.sector_patch_ids.end(),
                             std::vector<int>({8, 9, 10, 11}))
                       != selection.sector_patch_ids.end(),
            "admitted sectors are literal outer wall and top annulus quarters");
    require(contains_patch(selection.dof_ids, cloud, 3)
                && contains_patch(selection.dof_ids, cloud, 11),
            "selected values cross both outer and top periodic G1 seams");
    for (int id : selection.dof_ids) {
        const int patch = cloud.dofs[static_cast<std::size_t>(id)].patch_id;
        require((patch >= 0 && patch < 4) || (patch >= 8 && patch < 12),
                "selection excludes inner wall and bottom annulus sheets");
    }
}

} // namespace

int main()
{
    try {
        test_interval_length_uses_requested_native_subinterval();
        test_reversed_shared_edge_points_use_cell_midpoints_and_mapped_frames();
        test_lprism_partial_interval_closest_point_clamps_and_ties();
        test_circular_edge_closest_point_is_rigid_transform_invariant();
        test_lprism_split_edges_have_disjoint_midpoint_ids_and_reversed_parameters();
        test_direct_selector_balances_regular_and_three_sector_centers();
        test_direct_selector_uses_topology_not_physical_proximity_and_g1_outside_band();
        test_edge_selector_contributes_each_connection_and_is_deterministic();
        test_hollow_cylinder_periodic_g1_sectors_are_admitted_without_other_sheets();
        std::cout << "harmonic_cauchy_fit_3d_test passed" << std::endl;
        return 0;
    } catch (const kfbim::app3d::HarmonicCauchyError3D& error) {
        const auto& diagnostic = error.diagnostic();
        std::cerr << "harmonic_cauchy_fit_3d_test structured failure: stage="
                  << diagnostic.stage << " entity=" << diagnostic.entity_kind
                  << " id=" << diagnostic.entity_id
                  << " connection=" << diagnostic.connection_id
                  << " message=" << diagnostic.message << std::endl;
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "harmonic_cauchy_fit_3d_test failed: "
                  << error.what() << std::endl;
        return 1;
    }
}
