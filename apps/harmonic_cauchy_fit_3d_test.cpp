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
using kfbim::app3d::HarmonicCauchyFit3D;
using kfbim::app3d::HarmonicCauchyRoute3D;
using kfbim::app3d::LegacySurfaceCauchyPolicy3D;
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
using kfbim::geometry::NurbsBasis1D;
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

NativeNurbsSurface3D unit_y_edge_wedge()
{
    NativeNurbsSurface3D surface;
    surface.name = "unit_y_edge_wedge";
    surface.description = "two unit planes sharing (0,y,0)";
    surface.patches = {
        NurbsSurfacePatch3D::make_bilinear_plane(
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}),
        NurbsSurfacePatch3D::make_bilinear_plane(
            {0.0, 0.0, 0.0}, {0.0, 0.0, -1.0},
            {0.0, 1.0, 0.0}, {0.0, 1.0, -1.0})};
    surface.patch_names = {"horizontal", "vertical"};
    surface.smooth_neighbors.resize(2);
    surface.topological_patch_neighbors = {{1}, {0}};
    surface.geometric_connections = {
        {{0, NurbsPatchEdge3D::UMin, 0.0, 1.0},
         {1, NurbsPatchEdge3D::UMin, 0.0, 1.0},
         false,
         false}};
    surface.patch_components = {0, 0};
    surface.expected_area = 2.0;
    surface.exact_inside = [](const Eigen::Vector3d&) { return false; };
    return surface;
}

NativeNurbsSurface3D zero_tangent_edge_wedge()
{
    NativeNurbsSurface3D surface = unit_y_edge_wedge();
    surface.patches = {
        NurbsSurfacePatch3D::make_bilinear_plane(
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
            {0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}),
        NurbsSurfacePatch3D::make_bilinear_plane(
            {0.0, 0.0, 0.0}, {0.0, 0.0, -1.0},
            {0.0, 0.0, 0.0}, {0.0, 1.0, -1.0})};
    return surface;
}

NativeNurbsSurface3D nonfinite_tangent_edge_wedge()
{
    const NurbsBasis1D basis_u(
        1, std::vector<double>{0.0, 0.0, 1.0, 1.0});
    const NurbsBasis1D basis_v(
        1, std::vector<double>{0.0, 0.0, 1.0e-308, 1.0e-308},
        1.0e-320);
    const std::vector<std::vector<double>> weights{{1.0, 1.0}, {1.0, 1.0}};
    NativeNurbsSurface3D surface = unit_y_edge_wedge();
    surface.patches = {
        NurbsSurfacePatch3D(
            basis_u, basis_v,
            {{{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
             {{1.0e-308, 0.0, 0.0}, {1.0e-308, 1.0, 0.0}}},
            weights),
        NurbsSurfacePatch3D(
            basis_u, basis_v,
            {{{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
             {{0.0, 0.0, -1.0e-308}, {0.0, 1.0, -1.0e-308}}},
            weights)};
    surface.geometric_connections[0].first.end = 1.0e-308;
    surface.geometric_connections[0].second.end = 1.0e-308;
    return surface;
}

NativeNurbsSurface3D opposite_normal_bisector_wedge()
{
    NativeNurbsSurface3D surface = unit_y_edge_wedge();
    surface.patches[1] = NurbsSurfacePatch3D::make_bilinear_plane(
        {0.0, 0.0, 0.0}, {-1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}, {-1.0, 1.0, 0.0});
    return surface;
}

double cubic_wedge_polynomial(const Eigen::Vector3d& x)
{
    return 1.0 + x.y() - 0.5 * x.z() + 2.0 * x.x() * x.z()
         + x.x() * x.x() * x.x() - 3.0 * x.x() * x.y() * x.y();
}

Eigen::Vector3d cubic_wedge_gradient(const Eigen::Vector3d& x)
{
    return {2.0 * x.z() + 3.0 * x.x() * x.x() - 3.0 * x.y() * x.y(),
            1.0 - 6.0 * x.x() * x.y(),
            -0.5 + 2.0 * x.x()};
}

void test_first_level_edge_maps_reproduce_cubic()
{
    // Catches treating the rotated harmonic coefficient zero as p(0) and
    // multiplying normal data by h twice.
    constexpr double h = 1.0 / 8.0;
    const NativeNurbsSurface3D surface = unit_y_edge_wedge();
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const auto neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(surface, cloud, h);
    const HarmonicCauchyFit3D fit = HarmonicCauchyFit3D::build(
        surface, cloud, neighborhoods, h,
        HarmonicCauchyRoute3D::EdgeReconstructedValue);

    require(fit.space().dimension() == 16,
            "cubic harmonic space has literal dimension 16");
    require(fit.edge_maps().size() == 8,
            "unit shared edge at h=1/8 has eight first-level maps");
    for (const auto& map : fit.edge_maps()) {
        require(map.value_ids.size() == 48
                    && map.normal_ids.size() == 28,
                "each first-level map has literal 48 value and 28 normal IDs");
        require(map.value_sector_counts == std::array<int, 2>{{24, 24}}
                    && map.normal_sector_counts
                           == std::array<int, 2>{{14, 14}},
                "each first-level sector contributes exactly 24/14 IDs");
    }

    Eigen::VectorXd mu(static_cast<int>(cloud.dofs.size()));
    Eigen::VectorXd eta(static_cast<int>(cloud.dofs.size()));
    for (int q = 0; q < mu.size(); ++q) {
        const auto& dof = cloud.dofs[static_cast<std::size_t>(q)];
        mu[q] = cubic_wedge_polynomial(dof.point);
        eta[q] = cubic_wedge_gradient(dof.point).dot(dof.normal);
    }
    const auto manufactured = fit.apply(mu, eta);
    require(manufactured.edge_values.size()
                == static_cast<int>(fit.edge_maps().size()),
            "apply forms one global scalar per first-level edge map");
    for (int e = 0; e < manufactured.edge_values.size(); ++e) {
        require(std::abs(
                    manufactured.edge_values[e]
                    - cubic_wedge_polynomial(
                        fit.edge_maps()[static_cast<std::size_t>(e)].point.point))
                    < 2.0e-11,
                "first-level map reproduces the analytic cubic edge value");
    }
}

void test_edge_map_short_sector_reports_complete_structured_failure()
{
    // Catches borrowing across a short sector or throwing an unstructured
    // count-only error without point, sectors, radii, and stage.
    constexpr double h = 1.0 / 4.0;
    const NativeNurbsSurface3D surface = unit_y_edge_wedge();
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const auto neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(surface, cloud, h);
    bool caught = false;
    try {
        (void)HarmonicCauchyFit3D::build(
            surface, cloud, neighborhoods, h,
            HarmonicCauchyRoute3D::EdgeReconstructedValue);
    } catch (const kfbim::app3d::HarmonicCauchyError3D& error) {
        const auto& d = error.diagnostic();
        caught = true;
        require(d.stage == "edge_sector_selection"
                    && d.entity_kind == "edge_point" && d.entity_id == 0
                    && d.connection_id == 0,
                "short sector identifies its edge point, connection, and stage");
        require(d.incident_sectors
                        == std::vector<std::vector<int>>{{0}, {1}}
                    && d.required_value_count == 24
                    && d.required_normal_count == 14
                    && d.actual_value_counts == std::vector<int>({16, 16})
                    && d.actual_normal_counts == std::vector<int>({14, 14}),
                "short sector reports literal sectors and required/actual counts");
        require(d.value_radius_over_h > 0.0
                    && d.normal_radius_over_h > 0.0
                    && error.what() == d.message,
                "short sector carries radii and its structured message");
    }
    require(caught, "a 16-DOF sector cannot silently fill 24 value rows");
}

void test_non_g1_connection_requires_distinct_g1_sectors()
{
    // Catches accepting a declared non-G1 connection whose two sides resolve
    // to the same transitive G1 component.
    NativeNurbsSurface3D surface = unit_y_edge_wedge();
    surface.geometric_connections[0].second.patch = 0;
    surface.geometric_connections[0].second.edge = NurbsPatchEdge3D::UMin;
    bool caught = false;
    try {
        (void)make_shared_edge_points_3d(surface, 1.0 / 8.0);
    } catch (const kfbim::app3d::HarmonicCauchyError3D& error) {
        const auto& d = error.diagnostic();
        caught = true;
        require(d.stage == "shared_edge_geometry"
                    && d.entity_kind == "connection" && d.entity_id == 0
                    && d.connection_id == 0
                    && d.incident_sectors
                           == std::vector<std::vector<int>>{{0}, {0}},
                "distinct-sector failure reports the connection and both sectors");
        require(d.message.find("not distinct") != std::string::npos,
                "distinct-sector failure names the violated topology contract");
    }
    require(caught, "a non-G1 connection cannot join one G1 component");
}

void test_zero_and_nonfinite_edge_tangents_are_structured()
{
    // Catches ceil/int conversion of invalid edge lengths and any geometry
    // failure that loses its connection and incident-sector context.
    const std::array<NativeNurbsSurface3D, 2> surfaces{{
        zero_tangent_edge_wedge(), nonfinite_tangent_edge_wedge()}};
    for (int fixture = 0; fixture < 2; ++fixture) {
        bool caught = false;
        try {
            (void)make_shared_edge_points_3d(
                surfaces[static_cast<std::size_t>(fixture)], 1.0 / 8.0);
        } catch (const kfbim::app3d::HarmonicCauchyError3D& error) {
            const auto& d = error.diagnostic();
            caught = true;
            require(d.stage == "shared_edge_geometry"
                        && d.entity_kind == "connection"
                        && d.entity_id == 0 && d.connection_id == 0
                        && d.incident_sectors
                               == std::vector<std::vector<int>>{{0}, {1}},
                    "invalid tangent reports connection, sectors, and stage fixture "
                        + std::to_string(fixture));
            if (fixture == 0) {
                require(d.message.find("length") != std::string::npos,
                        "zero tangent is rejected as a non-positive edge length");
            } else {
                require(d.message.find("tangent") != std::string::npos,
                        "non-finite mapped tangent names its failure stage: "
                            + d.message);
            }
        }
        require(caught, "each invalid tangent fixture throws structured failure");
    }
}

void test_degenerate_normal_bisector_uses_deterministic_fallback_frame()
{
    // Catches normalizing a zero opposite-normal bisector instead of using
    // the deterministic lowest-sector fallback before frame construction.
    const auto points = make_shared_edge_points_3d(
        opposite_normal_bisector_wedge(), 1.0 / 8.0);
    require(points.points.size() == 8,
            "opposite-normal unit edge retains eight shared points");
    for (const auto& point : points.points) {
        require(point.sector_patch_ids[0] == std::vector<int>{0}
                    && point.sector_patch_ids[1] == std::vector<int>{1}
                    && (point.frame.col(2) - Eigen::Vector3d::UnitZ()).norm()
                           < 2.0e-12
                    && point.frame.determinant() > 1.0 - 2.0e-12,
                "degenerate bisector uses patch-0 normal as right-handed fallback");
    }
}

void test_both_map_levels_report_rank_failure_diagnostics()
{
    // Catches accepting a cutoff-rank-deficient design and dropping singular
    // extrema, counts, sectors, radii, or entity IDs from either level.
    constexpr double h = 1.0 / 8.0;
    const NativeNurbsSurface3D surface = unit_y_edge_wedge();
    const SurfaceDofCloud3D original =
        make_native_surface_dofs_3d(surface, h);
    const auto neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(surface, original, h);
    SurfaceDofCloud3D collapsed = original;
    for (auto& dof : collapsed.dofs)
        dof.point = Eigen::Vector3d(0.0, 0.5, 0.0);

    bool edge_caught = false;
    try {
        (void)HarmonicCauchyFit3D::build(
            surface, collapsed, neighborhoods, h,
            HarmonicCauchyRoute3D::EdgeReconstructedValue);
    } catch (const kfbim::app3d::HarmonicCauchyError3D& error) {
        const auto& d = error.diagnostic();
        edge_caught = true;
        require(d.stage == "edge_map_factorization"
                    && d.entity_kind == "edge_point" && d.entity_id >= 0
                    && d.connection_id == 0,
                "first-level rank failure identifies edge entity and stage");
        require(d.incident_sectors
                        == std::vector<std::vector<int>>{{0}, {1}}
                    && d.actual_value_counts == std::vector<int>({24, 24})
                    && d.actual_normal_counts == std::vector<int>({14, 14})
                    && d.required_value_count == 24
                    && d.required_normal_count == 14,
                "first-level rank failure carries sectors and exact counts");
        require(d.value_radius_over_h >= 0.0
                    && d.normal_radius_over_h >= 0.0
                    && d.sigma_max > 0.0 && d.sigma_min >= 0.0
                    && (d.sigma_min > 0.0 ? d.condition >= 1.0
                                          : std::isinf(d.condition)),
                "first-level rank failure carries radii and singular extrema");
    }
    require(edge_caught, "collapsed edge samples must fail first-level rank");

    bool surface_caught = false;
    try {
        (void)HarmonicCauchyFit3D::build(
            surface, collapsed, neighborhoods, h,
            HarmonicCauchyRoute3D::G1ValueG1Normal);
    } catch (const kfbim::app3d::HarmonicCauchyError3D& error) {
        const auto& d = error.diagnostic();
        surface_caught = true;
        require(d.stage == "surface_map_factorization"
                    && d.entity_kind == "surface_dof" && d.entity_id == 0,
                "second-level rank failure identifies surface entity and stage");
        require(d.actual_value_counts == std::vector<int>{48}
                    && d.actual_normal_counts == std::vector<int>{28}
                    && d.required_value_count == 48
                    && d.required_normal_count == 28
                    && d.actual_edge_count == 0,
                "second-level rank failure carries ordinary and edge counts");
        require(d.value_radius_over_h >= 0.0
                    && d.normal_radius_over_h >= 0.0
                    && d.edge_radius_over_h == 0.0
                    && d.sigma_max > 0.0 && d.sigma_min >= 0.0
                    && (d.sigma_min > 0.0 ? d.condition >= 1.0
                                          : std::isinf(d.condition)),
                "second-level rank failure carries all radii and singular extrema");
    }
    require(surface_caught,
            "collapsed surface samples must fail second-level rank");
}

void test_second_level_routes_reproduce_cubic_and_share_edge_values()
{
    // Catches route-specific neighborhood rebuilds, edge values formed per
    // surface row, a third zero product in the G1 path, and wrong normal h
    // scaling in the second-level maps.
    constexpr double h = 1.0 / 8.0;
    const NativeNurbsSurface3D surface = unit_y_edge_wedge();
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const auto neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(surface, cloud, h);
    const HarmonicCauchyFit3D g1 = HarmonicCauchyFit3D::build(
        surface, cloud, neighborhoods, h,
        HarmonicCauchyRoute3D::G1ValueG1Normal);
    const HarmonicCauchyFit3D direct = HarmonicCauchyFit3D::build(
        surface, cloud, neighborhoods, h,
        HarmonicCauchyRoute3D::DirectCrossFaceValue);
    const HarmonicCauchyFit3D edge = HarmonicCauchyFit3D::build(
        surface, cloud, neighborhoods, h,
        HarmonicCauchyRoute3D::EdgeReconstructedValue);
    require(g1.surface_maps().size() == cloud.dofs.size()
                && direct.surface_maps().size() == cloud.dofs.size()
                && edge.surface_maps().size() == cloud.dofs.size(),
            "all three routes own one second-level map per surface center");
    for (int q = 0; q < static_cast<int>(cloud.dofs.size()); ++q) {
        const auto& g1_map = g1.surface_maps()[static_cast<std::size_t>(q)];
        const auto& direct_map =
            direct.surface_maps()[static_cast<std::size_t>(q)];
        const auto& edge_map = edge.surface_maps()[static_cast<std::size_t>(q)];
        const std::vector<int> baseline_values =
            nearest_g1_cauchy_dofs(surface, cloud, q, 48);
        const std::vector<int> baseline_normals =
            nearest_g1_cauchy_dofs(surface, cloud, q, 28);
        const auto direct_values = select_direct_cross_face_value_dofs_3d(
            surface, cloud, neighborhoods, q, 48, h);
        require(g1_map.value_ids.size() == 48
                    && g1_map.normal_ids.size() == 28
                    && g1_map.edge_point_ids.empty()
                    && g1_map.M_edge.size() == 0,
                "G1 route has the literal two-term 48/28 map");
        require(direct_map.value_ids.size() == 48
                    && direct_map.normal_ids.size() == 28
                    && direct_map.edge_point_ids.empty(),
                "direct route changes values only and has no edge rows");
        require(edge_map.value_ids.size() == 48
                    && edge_map.normal_ids.size() == 28,
                "edge route retains ordinary G1 48/28 rows");
        require(g1_map.value_ids == baseline_values
                    && edge_map.value_ids == baseline_values,
                "G1 and edge routes preserve the ordered 48-ID G1 baseline at "
                    + std::to_string(q));
        require(g1_map.normal_ids == baseline_normals
                    && direct_map.normal_ids == baseline_normals
                    && edge_map.normal_ids == baseline_normals,
                "all routes preserve the ordered 28-ID G1 normal baseline at "
                    + std::to_string(q));
        require(direct_map.value_ids == direct_values.dof_ids,
                "only the direct route replaces ordinary value IDs at "
                    + std::to_string(q));
        if (direct_map.relevant_connection_ids.empty()
            || direct_map.nearest_edge_distance_over_h > 2.0) {
            require(direct_map.value_ids == baseline_values
                        && direct_map.edge_point_ids.empty()
                        && direct_map.M_edge.size() == 0,
                    "outside-band direct maps retain the exact G1 baseline and no edge rows at "
                        + std::to_string(q));
        }
        require(g1_map.nearest_edge_distance_over_h
                        == direct_map.nearest_edge_distance_over_h
                    && g1_map.nearest_edge_distance_over_h
                           == edge_map.nearest_edge_distance_over_h,
                "route-independent cached edge distance is copied verbatim");
        require(g1_map.neighborhood_fingerprint == neighborhoods.fingerprint
                    && direct_map.neighborhood_fingerprint
                           == neighborhoods.fingerprint
                    && edge_map.neighborhood_fingerprint
                           == neighborhoods.fingerprint,
                "every route copies the shared neighborhood fingerprint "
                "verbatim");
    }

    const int left = cloud.patches[0].dof_index(0, 3);
    const int right = cloud.patches[1].dof_index(0, 3);
    const auto& left_map = edge.surface_maps()[static_cast<std::size_t>(left)];
    const auto& right_map = edge.surface_maps()[static_cast<std::size_t>(right)];
    require(!left_map.edge_point_ids.empty()
                && !right_map.edge_point_ids.empty()
                && left_map.edge_point_ids.front()
                       == right_map.edge_point_ids.front(),
            "both incident sides gather the same nearest global edge point ID");

    Eigen::VectorXd mu(static_cast<int>(cloud.dofs.size()));
    Eigen::VectorXd eta(static_cast<int>(cloud.dofs.size()));
    for (int q = 0; q < mu.size(); ++q) {
        const auto& dof = cloud.dofs[static_cast<std::size_t>(q)];
        mu[q] = cubic_wedge_polynomial(dof.point);
        eta[q] = cubic_wedge_gradient(dof.point).dot(dof.normal);
    }
    const Eigen::VectorXd origin = edge.space().basis(0.0, 0.0, 0.0);
    for (const auto* fit : {&g1, &direct, &edge}) {
        const auto applied = fit->apply(mu, eta);
        require(applied.coefficients.rows() == mu.size()
                    && applied.coefficients.cols() == 16,
                "second-level apply returns one cubic coefficient row per center");
        for (int q = 0; q < mu.size(); ++q) {
            require(std::abs(
                        origin.dot(applied.coefficients.row(q).transpose())
                        - mu[q]) < 2.0e-11,
                    "each route reproduces the analytic cubic center value");
        }
    }
}

void test_sparse_apply_is_linear_and_preserves_immutable_audit()
{
    // Catches recomputing geometry/SVD in apply, mutating precomputed maps,
    // and forming an edge vector separately for each surface map.
    constexpr double h = 1.0 / 8.0;
    const NativeNurbsSurface3D surface = unit_y_edge_wedge();
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const auto neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(surface, cloud, h);
    const HarmonicCauchyFit3D fit = HarmonicCauchyFit3D::build(
        surface, cloud, neighborhoods, h,
        HarmonicCauchyRoute3D::EdgeReconstructedValue);
    const int size = static_cast<int>(cloud.dofs.size());
    Eigen::VectorXd x_mu(size), x_eta(size), y_mu(size), y_eta(size);
    for (int q = 0; q < size; ++q) {
        x_mu[q] = 0.25 + 0.031 * static_cast<double>(q);
        x_eta[q] = -0.4 + 0.017 * static_cast<double>(q);
        y_mu[q] = std::sin(0.13 * static_cast<double>(q + 1));
        y_eta[q] = std::cos(0.19 * static_cast<double>(q + 2));
    }
    constexpr double a = 0.37;
    constexpr double b = -1.2;
    const auto audit_before = fit.audit();
    std::vector<std::uint64_t> neighborhood_fingerprints_before;
    neighborhood_fingerprints_before.reserve(fit.surface_maps().size());
    for (const auto& map : fit.surface_maps())
        neighborhood_fingerprints_before.push_back(
            map.neighborhood_fingerprint);
    const auto fx = fit.apply(x_mu, x_eta);
    const auto fy = fit.apply(y_mu, y_eta);
    const auto fxy = fit.apply(a * x_mu + b * y_mu,
                               a * x_eta + b * y_eta);
    const Eigen::VectorXd edge_expected =
        a * fx.edge_values + b * fy.edge_values;
    const Eigen::MatrixXd coefficient_expected =
        a * fx.coefficients + b * fy.coefficients;
    require((fxy.edge_values - edge_expected).norm()
                <= 2.0e-13 * std::max(1.0, edge_expected.norm()),
            "first-level sparse apply is linear to relative 2e-13");
    require((fxy.coefficients - coefficient_expected).norm()
                <= 2.0e-13 * std::max(1.0, coefficient_expected.norm()),
            "second-level sparse apply is linear to relative 2e-13");
    require(fit.audit().geometry_query_count == audit_before.geometry_query_count
                && fit.audit().svd_factorization_count
                       == audit_before.svd_factorization_count
                && fit.audit().fingerprint == audit_before.fingerprint,
            "repeated apply leaves the preprocessing audit bitwise unchanged");
    for (std::size_t q = 0; q < fit.surface_maps().size(); ++q) {
        require(fit.surface_maps()[q].neighborhood_fingerprint
                    == neighborhood_fingerprints_before[q],
                "apply preserves every copied neighborhood fingerprint");
    }
    for (const auto* applied : {&fx, &fy, &fxy}) {
        require(applied->edge_values.size()
                    == static_cast<int>(fit.edge_maps().size()),
                "one apply owns one global edge vector");
        require(applied->runtime_geometry_query_count == 0
                    && applied->runtime_svd_factorization_count == 0
                    && applied->fingerprint_before == audit_before.fingerprint
                    && applied->fingerprint_after == audit_before.fingerprint,
                "apply is gather-plus-matrix-only with stable fingerprints");
    }
}

void test_legacy_policies_keep_two_term_maps_and_publish_summary()
{
    // Catches dropping an existing environment policy, accidentally building
    // edge maps for legacy fits, and summary values diverging from owned maps.
    constexpr double h = 1.0 / 8.0;
    const NativeNurbsSurface3D surface = unit_y_edge_wedge();
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const std::array<LegacySurfaceCauchyPolicy3D, 4> policies{{
        LegacySurfaceCauchyPolicy3D::G1Nearest,
        LegacySurfaceCauchyPolicy3D::TopologicalNearest,
        LegacySurfaceCauchyPolicy3D::SamePatch,
        LegacySurfaceCauchyPolicy3D::BalancedPatches}};
    for (const auto policy : policies) {
        const HarmonicCauchyFit3D fit = HarmonicCauchyFit3D::build_legacy(
            surface, cloud, h, policy, 3, 48, 28);
        require(fit.legacy_policy() == policy && fit.legacy_summary().has_value(),
                "legacy fit retains its exact policy and publishes a summary");
        require(fit.edge_maps().empty()
                    && fit.surface_maps().size() == cloud.dofs.size(),
                "legacy policy owns only one two-term map per surface center");
        const auto summary = *fit.legacy_summary();
        require(summary.policy == policy && summary.degree == 3
                    && summary.value_count == 48 && summary.normal_count == 28
                    && summary.value_count_min == 48
                    && summary.value_count_max == 48
                    && summary.normal_count_min == 28
                    && summary.normal_count_max == 28,
                "legacy summary reports literal degree and stencil counts");
        for (const auto& map : fit.surface_maps()) {
            require(map.edge_point_ids.empty() && map.M_edge.size() == 0,
                    "legacy apply preserves the original two-term expression");
        }
    }
}

void test_legacy_lprism_n16_keeps_short_actual_stencils()
{
    // Catches applying the strict native-route 48/28 selector contract to
    // legacy fits, whose historical N=16 behavior intentionally records
    // requested counts while retaining shorter per-center selected rows.
    constexpr double h = 3.0 / 16.0;
    const NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const HarmonicCauchyFit3D fit = HarmonicCauchyFit3D::build_legacy(
        surface, cloud, h, LegacySurfaceCauchyPolicy3D::G1Nearest,
        3, 48, 28);
    const auto summary = *fit.legacy_summary();
    require(summary.value_count == 48 && summary.normal_count == 28,
            "short legacy summary keeps literal requested 48/28 counts");
    require(summary.value_count_min < summary.value_count_max
                && summary.value_count_max == 48
                && summary.normal_count_min == 28
                && summary.normal_count_max == 28,
            "short legacy summary reports actual truncated value min/max");
    for (int center = 0; center < static_cast<int>(cloud.dofs.size()); ++center) {
        const auto& map = fit.surface_maps()[static_cast<std::size_t>(center)];
        require(map.value_ids
                        == nearest_g1_cauchy_dofs(surface, cloud, center, 48)
                    && map.normal_ids
                        == nearest_g1_cauchy_dofs(surface, cloud, center, 28),
                "short legacy fit preserves ordered selector IDs");
    }
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

void test_direct_selector_orders_exact_unequal_distances_before_ids()
{
    // Catches quantizing unequal squared distances into one tie bucket and then
    // allowing the smaller ID to precede the geometrically nearer DOF.
    constexpr double h = 1.0 / 8.0;
    const NativeNurbsSurface3D surface = unit_y_edge_wedge();
    SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const auto neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(surface, cloud, h);
    const int center = cloud.patches[0].dof_index(0, 3);
    const Eigen::Vector3d center_point =
        cloud.dofs[static_cast<std::size_t>(center)].point;
    const int farther_lower_id = 0;
    const int nearer_higher_id = 1;
    const double quantum = 1.0e-12 * h * h;
    const double key = 10000000000.0;
    cloud.dofs[static_cast<std::size_t>(farther_lower_id)].point =
        center_point
        + Eigen::Vector3d(std::sqrt((key + 0.3) * quantum), 0.0, 0.0);
    cloud.dofs[static_cast<std::size_t>(nearer_higher_id)].point =
        center_point
        + Eigen::Vector3d(std::sqrt((key + 0.1) * quantum), 0.0, 0.0);

    const auto selection = select_direct_cross_face_value_dofs_3d(
        surface, cloud, neighborhoods, center, 48, h);
    const auto farther = std::find(
        selection.dof_ids.begin(), selection.dof_ids.end(), farther_lower_id);
    const auto nearer = std::find(
        selection.dof_ids.begin(), selection.dof_ids.end(), nearer_higher_id);
    require(farther != selection.dof_ids.end()
                && nearer != selection.dof_ids.end() && nearer < farther,
            "direct selector orders unequal distances exactly before using ID ties");
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

void test_corrupted_unrelated_connection_cache_fails_structurally()
{
    // Catches trusting a corrupted cached relevant-connection ID that is not
    // incident to the center G1 sector and admitting unrelated patches.
    NativeNurbsSurface3D surface = wedge_with_unrelated_coincident_patch();
    surface.geometric_connections.push_back(
        {{1, NurbsPatchEdge3D::UMax, 0.0, 1.0},
         {2, NurbsPatchEdge3D::UMax, 0.0, 1.0}, false, false});
    constexpr double h = 1.0 / 8.0;
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const int center = cloud.patches[0].dof_index(0, 3);
    kfbim::app3d::SurfaceNonG1EdgeNeighborhoodSet3D corrupted;
    corrupted.centers.resize(cloud.dofs.size());
    for (int q = 0; q < static_cast<int>(cloud.dofs.size()); ++q)
        corrupted.centers[static_cast<std::size_t>(q)].center_dof = q;
    corrupted.centers[static_cast<std::size_t>(center)]
        .nearest_distance_over_h = 1.0;
    corrupted.centers[static_cast<std::size_t>(center)]
        .relevant_connection_ids = {1};
    bool caught = false;
    try {
        (void)select_direct_cross_face_value_dofs_3d(
            surface, cloud, corrupted, center, 48, h);
    } catch (const kfbim::app3d::HarmonicCauchyError3D& error) {
        const auto& d = error.diagnostic();
        caught = true;
        require(d.stage == "direct_selector_topology"
                    && d.entity_kind == "surface_dof"
                    && d.entity_id == center && d.connection_id == 1,
                "corrupted cache failure identifies center, connection, and stage");
        require(d.incident_sectors
                        == std::vector<std::vector<int>>{{0}, {1}, {2}},
                    "corrupted cache failure reports center and foreign sectors");
    }
    require(caught,
            "a cached connection unrelated to the center sector cannot be admitted");
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

void test_outside_band_short_g1_template_fails_with_value_count_diagnostic()
{
    // Catches silently returning fewer values than the requested direct-route
    // template when the outside-band baseline G1 component is too small.
    constexpr double h = 0.2;
    const NativeNurbsSurface3D surface = reversed_unit_wedge();
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const auto neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(surface, cloud, h);
    const auto& patch = cloud.patches[0];
    const int center = patch.dof_index(patch.nu / 2, patch.nv / 2);
    require(neighborhoods.centers[static_cast<std::size_t>(center)]
                .nearest_distance_over_h > 2.0,
            "short-template fixture is outside the literal 2h edge band");

    bool caught = false;
    try {
        (void)select_direct_cross_face_value_dofs_3d(
            surface, cloud, neighborhoods, center, 48, h);
    } catch (const kfbim::app3d::HarmonicCauchyError3D& error) {
        const auto& diagnostic = error.diagnostic();
        caught = true;
        require(diagnostic.stage == "direct_selector"
                    && diagnostic.entity_kind == "surface_dof"
                    && diagnostic.entity_id == center,
                "short outside-band template reports its selector center");
        require(diagnostic.required_value_count == 48
                    && diagnostic.actual_value_counts == std::vector<int>{25},
                "short outside-band template reports literal required 48 and actual 25");
        require(diagnostic.incident_sectors
                    == std::vector<std::vector<int>>{{0}},
                "short outside-band template reports the center G1 sector");
        require(diagnostic.value_radius_over_h > 0.0,
                "short outside-band template reports its available value radius");
    }
    require(caught,
            "outside-band short G1 template must fail instead of returning 25 IDs");
}

void test_in_band_short_direct_template_reports_available_value_radius()
{
    // Catches throwing after exhausting admitted in-band sectors without
    // measuring the value IDs that were actually selected.
    constexpr double h = 1.0 / 4.0;
    const NativeNurbsSurface3D surface = unit_y_edge_wedge();
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const auto neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(surface, cloud, h);
    const int center = cloud.patches[0].dof_index(0, 1);
    require(neighborhoods.centers[static_cast<std::size_t>(center)]
                    .nearest_distance_over_h <= 2.0,
            "short direct fixture lies inside the literal 2h edge band");

    bool caught = false;
    try {
        (void)select_direct_cross_face_value_dofs_3d(
            surface, cloud, neighborhoods, center, 48, h);
    } catch (const kfbim::app3d::HarmonicCauchyError3D& error) {
        const auto& diagnostic = error.diagnostic();
        caught = true;
        require(diagnostic.stage == "direct_selector"
                    && diagnostic.entity_kind == "surface_dof"
                    && diagnostic.entity_id == center
                    && diagnostic.required_value_count == 48,
                "short in-band template reports its selector center and target");
        require(diagnostic.incident_sectors
                        == std::vector<std::vector<int>>{{0}, {1}}
                    && diagnostic.actual_value_counts
                           == std::vector<int>({16, 16}),
                "short in-band template reports both exhausted G1 sectors");
        require(diagnostic.value_radius_over_h > 0.0,
                "short in-band template reports its available value radius");
    }
    require(caught,
            "two 16-DOF sectors cannot fill the 48-value direct template");
}

void test_short_surface_template_reports_available_value_and_normal_radii()
{
    // Catches throwing the second-level count failure before measuring the
    // available ordinary value and normal selections.
    constexpr double h = 0.2;
    const NativeNurbsSurface3D surface = reversed_unit_wedge();
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const auto neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(surface, cloud, h);

    bool caught = false;
    try {
        (void)HarmonicCauchyFit3D::build(
            surface, cloud, neighborhoods, h,
            HarmonicCauchyRoute3D::G1ValueG1Normal);
    } catch (const kfbim::app3d::HarmonicCauchyError3D& error) {
        const auto& diagnostic = error.diagnostic();
        caught = true;
        require(diagnostic.stage == "surface_sector_selection"
                    && diagnostic.actual_value_counts == std::vector<int>{25}
                    && diagnostic.actual_normal_counts == std::vector<int>{25},
                "short surface template reports its actual ordinary row counts");
        require(diagnostic.value_radius_over_h > 0.0
                    && diagnostic.normal_radius_over_h > 0.0,
                "short surface template reports both available ordinary radii");
    }
    require(caught, "a 25-DOF G1 sector cannot fill the 48/28 surface template");
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

void test_edge_selector_orders_exact_unequal_distances_before_ids()
{
    // Catches quantizing unequal squared distances into one tie bucket in the
    // shared-edge selector.
    constexpr double h = 0.2;
    const NativeNurbsSurface3D surface = reversed_unit_wedge();
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const int center = cloud.patches[0].dof_index(2, 2);
    const Eigen::Vector3d center_point =
        cloud.dofs[static_cast<std::size_t>(center)].point;

    kfbim::app3d::SurfaceNonG1EdgeNeighborhoodSet3D neighborhoods;
    neighborhoods.centers.resize(cloud.dofs.size());
    for (int q = 0; q < static_cast<int>(cloud.dofs.size()); ++q)
        neighborhoods.centers[static_cast<std::size_t>(q)].center_dof = q;
    neighborhoods.centers[static_cast<std::size_t>(center)]
        .nearest_distance_over_h = 1.0;
    neighborhoods.centers[static_cast<std::size_t>(center)]
        .relevant_connection_ids = {0};

    const double quantum = 1.0e-12 * h * h;
    const double key = 10000000000.0;
    kfbim::app3d::SharedEdgePoint3D farther;
    farther.id = 0;
    farther.connection_id = 0;
    farther.point = center_point
        + Eigen::Vector3d(std::sqrt((key + 0.3) * quantum), 0.0, 0.0);
    kfbim::app3d::SharedEdgePoint3D nearer;
    nearer.id = 1;
    nearer.connection_id = 0;
    nearer.point = center_point
        + Eigen::Vector3d(std::sqrt((key + 0.1) * quantum), 0.0, 0.0);
    kfbim::app3d::SharedEdgePointSet3D edge_points;
    edge_points.points = {farther, nearer};
    edge_points.point_ids_by_connection.resize(1);
    edge_points.point_ids_by_connection[0] = {0, 1};

    const auto selection = select_surface_edge_points_3d(
        surface, cloud, edge_points, neighborhoods, center, h);
    require(selection.edge_point_ids == std::vector<int>({1, 0}),
            "edge selector orders unequal distances exactly before using ID ties");
}

SurfaceDofCloud3D transform_test_cloud(
    SurfaceDofCloud3D cloud,
    const RigidTransform3D& transform)
{
    for (auto& dof : cloud.dofs) {
        dof.point = transform.forward_point(dof.point);
        dof.normal = transform.forward_vector(dof.normal);
        dof.tangent1 = transform.forward_vector(dof.tangent1);
        dof.tangent2 = transform.forward_vector(dof.tangent2);
    }
    return cloud;
}

void test_two_level_maps_follow_selector_contracts_after_rigid_transform()
{
    // Catches world-axis local coordinates while allowing exact floating-point
    // distances to choose each geometry's IDs through its public selectors.
    constexpr double h = 1.0 / 8.0;
    constexpr double pi = 3.1415926535897932384626433832795;
    const NativeNurbsSurface3D surface = unit_y_edge_wedge();
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const auto neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(surface, cloud, h);
    const HarmonicCauchyFit3D original = HarmonicCauchyFit3D::build(
        surface, cloud, neighborhoods, h,
        HarmonicCauchyRoute3D::EdgeReconstructedValue);
    const HarmonicCauchyFit3D original_direct = HarmonicCauchyFit3D::build(
        surface, cloud, neighborhoods, h,
        HarmonicCauchyRoute3D::DirectCrossFaceValue);
    const RigidTransform3D transform = RigidTransform3D::from_axis_angle(
        Eigen::Vector3d(1.0, 2.0, 3.0), 17.0 * pi / 180.0,
        Eigen::Vector3d::Zero(), Eigen::Vector3d(0.31, -0.27, 0.19));
    const NativeNurbsSurface3D moved_surface =
        transform_native_nurbs_surface_3d(surface, transform);
    const SurfaceDofCloud3D moved_cloud =
        transform_test_cloud(cloud, transform);
    const auto moved_neighborhoods =
        build_surface_non_g1_edge_neighborhoods_3d(
            moved_surface, moved_cloud, h);
    const HarmonicCauchyFit3D moved = HarmonicCauchyFit3D::build(
        moved_surface, moved_cloud, moved_neighborhoods, h,
        HarmonicCauchyRoute3D::EdgeReconstructedValue);
    const HarmonicCauchyFit3D moved_direct = HarmonicCauchyFit3D::build(
        moved_surface, moved_cloud, moved_neighborhoods, h,
        HarmonicCauchyRoute3D::DirectCrossFaceValue);
    const auto original_points = make_shared_edge_points_3d(surface, h);
    const auto moved_points = make_shared_edge_points_3d(moved_surface, h);
    require(original.edge_maps().size() == moved.edge_maps().size()
                && original.surface_maps().size() == moved.surface_maps().size(),
            "rigid transform preserves both map counts");
    for (int e = 0; e < static_cast<int>(original.edge_maps().size()); ++e) {
        const auto& a = original.edge_maps()[static_cast<std::size_t>(e)];
        const auto& b = moved.edge_maps()[static_cast<std::size_t>(e)];
        require(a.value_ids == b.value_ids && a.normal_ids == b.normal_ids
                    && a.value_sector_counts == b.value_sector_counts
                    && a.normal_sector_counts == b.normal_sector_counts
                    && a.point.fraction == b.point.fraction,
                "rigid transform preserves first-level IDs, fractions, and counts");
        require((b.point.point - transform.forward_point(a.point.point)).norm()
                        < 2.0e-12
                    && (b.point.frame
                        - transform.rotation() * a.point.frame).norm()
                           < 2.0e-12,
                "rigid transform moves edge points and frames covariantly");
        const double scale = std::max(1.0, std::abs(a.sigma_max));
        require(std::abs(a.sigma_max - b.sigma_max) <= 5.0e-12 * scale
                    && std::abs(a.sigma_min - b.sigma_min)
                           <= 5.0e-12 * std::max(1.0, std::abs(a.sigma_min))
                    && std::abs(a.condition - b.condition)
                           <= 5.0e-12 * std::max(1.0, std::abs(a.condition)),
                "rigid transform preserves first-level singular diagnostics");
    }
    for (int q = 0; q < static_cast<int>(original.surface_maps().size()); ++q) {
        const auto& a = original.surface_maps()[static_cast<std::size_t>(q)];
        const auto& b = moved.surface_maps()[static_cast<std::size_t>(q)];
        require(a.value_ids
                        == nearest_g1_cauchy_dofs(surface, cloud, q, 48)
                    && b.value_ids
                           == nearest_g1_cauchy_dofs(
                               moved_surface, moved_cloud, q, 48)
                    && a.normal_ids
                           == nearest_g1_cauchy_dofs(surface, cloud, q, 28)
                    && b.normal_ids
                           == nearest_g1_cauchy_dofs(
                               moved_surface, moved_cloud, q, 28)
                    && a.edge_point_ids
                           == select_surface_edge_points_3d(
                               surface, cloud, original_points,
                               neighborhoods, q, h).edge_point_ids
                    && b.edge_point_ids
                           == select_surface_edge_points_3d(
                               moved_surface, moved_cloud, moved_points,
                               moved_neighborhoods, q, h).edge_point_ids
                    && a.value_sector_counts == b.value_sector_counts
                    && a.normal_sector_counts == b.normal_sector_counts,
                "each rigid geometry consumes its G1 and edge selector IDs at "
                    + std::to_string(q));
        const auto& direct_a =
            original_direct.surface_maps()[static_cast<std::size_t>(q)];
        const auto& direct_b =
            moved_direct.surface_maps()[static_cast<std::size_t>(q)];
        require(direct_a.value_ids
                        == select_direct_cross_face_value_dofs_3d(
                            surface, cloud, neighborhoods, q, 48, h).dof_ids
                    && direct_b.value_ids
                           == select_direct_cross_face_value_dofs_3d(
                                moved_surface, moved_cloud,
                                moved_neighborhoods, q, 48, h).dof_ids
                    && direct_a.normal_ids
                           == nearest_g1_cauchy_dofs(surface, cloud, q, 28)
                    && direct_b.normal_ids
                           == nearest_g1_cauchy_dofs(
                               moved_surface, moved_cloud, q, 28),
                "direct rigid maps consume local direct values and G1 normals at "
                    + std::to_string(q));
    }

    Eigen::VectorXd moved_mu(static_cast<int>(moved_cloud.dofs.size()));
    Eigen::VectorXd moved_eta(static_cast<int>(moved_cloud.dofs.size()));
    for (int q = 0; q < moved_mu.size(); ++q) {
        const auto& dof = moved_cloud.dofs[static_cast<std::size_t>(q)];
        const Eigen::Vector3d original_point = transform.inverse_point(dof.point);
        moved_mu[q] = cubic_wedge_polynomial(original_point);
        moved_eta[q] = transform.forward_vector(
            cubic_wedge_gradient(original_point)).dot(dof.normal);
    }
    const auto applied = moved.apply(moved_mu, moved_eta);
    for (int e = 0; e < applied.edge_values.size(); ++e) {
        const Eigen::Vector3d original_point = transform.inverse_point(
            moved.edge_maps()[static_cast<std::size_t>(e)].point.point);
        require(std::abs(applied.edge_values[e]
                         - cubic_wedge_polynomial(original_point)) < 2.0e-11,
                "moved first-level map reproduces transformed cubic edge data");
    }
}

void test_edge_selector_snaps_exact_two_h_boundary_under_rigid_transform()
{
    // Catches comparing raw rounded point distance with 2h: a mathematically
    // exact boundary point must keep the same ID after a fixed rigid transform.
    constexpr double h = 0.2;
    const NativeNurbsSurface3D surface = reversed_unit_wedge();
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    const auto& patch = cloud.patches[0];
    const int center = patch.dof_index(patch.nu / 2, patch.nv / 2);
    const Eigen::Vector3d center_point =
        cloud.dofs[static_cast<std::size_t>(center)].point;

    kfbim::app3d::SurfaceNonG1EdgeNeighborhoodSet3D neighborhoods;
    neighborhoods.centers.resize(cloud.dofs.size());
    for (int q = 0; q < static_cast<int>(cloud.dofs.size()); ++q)
        neighborhoods.centers[static_cast<std::size_t>(q)].center_dof = q;
    neighborhoods.centers[static_cast<std::size_t>(center)]
        .nearest_distance_over_h = 1.0;
    neighborhoods.centers[static_cast<std::size_t>(center)]
        .relevant_connection_ids = {0};

    kfbim::app3d::SharedEdgePointSet3D edge_points;
    edge_points.point_ids_by_connection.resize(
        surface.geometric_connections.size());
    edge_points.point_ids_by_connection[0] = {0, 1};
    kfbim::app3d::SharedEdgePoint3D nearest;
    nearest.id = 0;
    nearest.connection_id = 0;
    nearest.point = center_point + Eigen::Vector3d(0.0, 0.5 * h, 0.0);
    kfbim::app3d::SharedEdgePoint3D boundary;
    boundary.id = 1;
    boundary.connection_id = 0;
    boundary.point = center_point + Eigen::Vector3d(2.0 * h, 0.0, 0.0);
    edge_points.points = {nearest, boundary};

    const auto original = select_surface_edge_points_3d(
        surface, cloud, edge_points, neighborhoods, center, h);
    require(original.edge_point_ids == std::vector<int>({0, 1}),
            "literal nearest and exact-2h shared points are both selected");

    const RigidTransform3D transform = RigidTransform3D::from_axis_angle(
        Eigen::Vector3d(1.0, 2.0, -1.0),
        0.37,
        Eigen::Vector3d(0.2, -0.1, 0.3),
        Eigen::Vector3d(-0.4, 0.25, 0.15));
    const NativeNurbsSurface3D moved_surface =
        transform_native_nurbs_surface_3d(surface, transform);
    const SurfaceDofCloud3D moved_cloud =
        transform_test_cloud(cloud, transform);
    auto moved_edge_points = edge_points;
    for (auto& point : moved_edge_points.points)
        point.point = transform.forward_point(point.point);
    const auto moved = select_surface_edge_points_3d(
        moved_surface, moved_cloud, moved_edge_points,
        neighborhoods, center, h);
    require(moved.edge_point_ids == original.edge_point_ids,
            "exact-2h selected IDs are invariant under the fixed rigid transform");
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
        test_direct_selector_orders_exact_unequal_distances_before_ids();
        test_direct_selector_uses_topology_not_physical_proximity_and_g1_outside_band();
        test_corrupted_unrelated_connection_cache_fails_structurally();
        test_outside_band_short_g1_template_fails_with_value_count_diagnostic();
        test_in_band_short_direct_template_reports_available_value_radius();
        test_short_surface_template_reports_available_value_and_normal_radii();
        test_edge_selector_contributes_each_connection_and_is_deterministic();
        test_edge_selector_orders_exact_unequal_distances_before_ids();
        test_edge_selector_snaps_exact_two_h_boundary_under_rigid_transform();
        test_hollow_cylinder_periodic_g1_sectors_are_admitted_without_other_sheets();
        test_first_level_edge_maps_reproduce_cubic();
        test_edge_map_short_sector_reports_complete_structured_failure();
        test_non_g1_connection_requires_distinct_g1_sectors();
        test_zero_and_nonfinite_edge_tangents_are_structured();
        test_degenerate_normal_bisector_uses_deterministic_fallback_frame();
        test_both_map_levels_report_rank_failure_diagnostics();
        test_second_level_routes_reproduce_cubic_and_share_edge_values();
        test_sparse_apply_is_linear_and_preserves_immutable_audit();
        test_legacy_policies_keep_two_term_maps_and_publish_summary();
        test_legacy_lprism_n16_keeps_short_actual_stencils();
        test_two_level_maps_follow_selector_contracts_after_rigid_transform();
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
