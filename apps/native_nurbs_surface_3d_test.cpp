#include "native_nurbs_surface_3d.hpp"

#include "src/geometry/grid_pair_3d.hpp"
#include "src/geometry/nurbs_bezier_extraction_3d.hpp"
#include "src/geometry/nurbs_bezier_intersection_3d.hpp"
#include "src/geometry/nurbs_bezier_segment_closest_point_3d.hpp"
#include "src/geometry/nurbs_bezier_segment_stationary_points_3d.hpp"
#include "src/geometry/nurbs_cartesian_domain_3d.hpp"
#include "src/geometry/rational_bezier_element_3d.hpp"
#include "src/geometry/rational_bezier_subdivision_3d.hpp"
#include "src/geometry/nurbs_surface_model_3d.hpp"
#include "src/geometry/nurbs_surface_intersector_3d.hpp"
#include "src/grid/cartesian_grid_3d.hpp"
#include "src/transfer/laplace_correction_support.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

namespace {

using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::NativeNurbsSurface3D;
using kfbim::app3d::PatchEdge3D;
using kfbim::app3d::SmoothPatchNeighbor3D;
using kfbim::app3d::SurfaceDofCloud3D;
using kfbim::app3d::balanced_topological_cauchy_dofs;
using kfbim::app3d::interpolate_triangle_parameter;
using kfbim::app3d::make_native_nurbs_surface_3d;
using kfbim::app3d::make_native_surface_dofs_3d;
using kfbim::app3d::nearest_same_patch_cauchy_dofs;
using kfbim::app3d::nearest_topological_cauchy_dofs;
using kfbim::app3d::parameter_dof_candidates_2x2;
using kfbim::app3d::smooth_patch_component;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <class Function>
void require_throws_contains(Function&& function,
                             const std::string& needle,
                             const std::string& message)
{
    try {
        function();
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(needle) != std::string::npos,
                message + ": " + error.what());
        return;
    }
    throw std::runtime_error(message + ": no exception");
}

template <class Exception, class Function>
void require_throws_type_contains(Function&& function,
                                  const std::string& needle,
                                  const std::string& message)
{
    try {
        function();
    } catch (const Exception& error) {
        require(std::string(error.what()).find(needle) != std::string::npos,
                message + ": " + error.what());
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(
            message + ": wrong exception type: " + error.what());
    }
    throw std::runtime_error(message + ": no exception");
}

template <class Function>
void require_throws_contains_all(
    Function&& function,
    std::initializer_list<const char*> needles,
    const std::string& message)
{
    try {
        function();
    } catch (const std::exception& error) {
        const std::string text = error.what();
        for (const char* needle : needles) {
            require(text.find(needle) != std::string::npos,
                    message + ": " + text);
        }
        return;
    }
    throw std::runtime_error(message + ": no exception");
}

template <class Function>
void require_throws_contains_all_with_count(
    Function&& function,
    std::initializer_list<const char*> needles,
    const std::string& repeated,
    std::size_t minimum_count,
    const std::string& message)
{
    try {
        function();
    } catch (const std::exception& error) {
        const std::string text = error.what();
        for (const char* needle : needles) {
            require(text.find(needle) != std::string::npos,
                    message + ": " + text);
        }
        std::size_t count = 0;
        for (std::size_t position = 0;
             (position = text.find(repeated, position)) != std::string::npos;
             position += repeated.size()) {
            ++count;
        }
        require(count >= minimum_count, message + ": " + text);
        return;
    }
    throw std::runtime_error(message + ": no exception");
}

void check_patch_regular(const NativeNurbsSurface3D& surface)
{
    require(surface.patch_names.size() == surface.patches.size(),
            surface.name + " patch-name count");
    require(surface.smooth_neighbors.size() == surface.patches.size(),
            surface.name + " smooth-neighbor count");
    require(surface.topological_patch_neighbors.size() == surface.patches.size(),
            surface.name + " topological-neighbor count");
    for (int patch = 0; patch < static_cast<int>(surface.patches.size()); ++patch) {
        for (int neighbor : surface.topological_patch_neighbors[
                 static_cast<std::size_t>(patch)]) {
            require(neighbor >= 0
                        && neighbor < static_cast<int>(surface.patches.size())
                        && neighbor != patch,
                    surface.name + " valid topological neighbor");
            const auto& reverse = surface.topological_patch_neighbors[
                static_cast<std::size_t>(neighbor)];
            require(std::find(reverse.begin(), reverse.end(), patch)
                        != reverse.end(),
                    surface.name + " symmetric topological adjacency");
        }
    }
    for (const auto& patch : surface.patches) {
        const double u = 0.5 * (patch.domain_start_u() + patch.domain_end_u());
        const double v = 0.5 * (patch.domain_start_v() + patch.domain_end_v());
        const auto d = patch.evaluate_with_derivatives(u, v);
        require(d.point.allFinite() && d.du.allFinite() && d.dv.allFinite(),
                surface.name + " finite patch evaluation");
        require(d.du.cross(d.dv).norm() > 1.0e-12,
                surface.name + " nondegenerate patch Jacobian");
    }
}

void test_native_models()
{
    constexpr double pi = 3.141592653589793238462643383279502884;
    const NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    const NativeNurbsSurface3D cylinder =
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    const auto& outer_neighbors = cylinder.topological_patch_neighbors[0];
    require(std::find(outer_neighbors.begin(), outer_neighbors.end(), 8)
                != outer_neighbors.end()
                && std::find(outer_neighbors.begin(), outer_neighbors.end(), 12)
                       != outer_neighbors.end(),
            "outer cylinder wall is topologically adjacent to both caps");
    require(std::find(outer_neighbors.begin(), outer_neighbors.end(), 4)
                == outer_neighbors.end(),
            "outer and inner cylinder walls are not direct neighbors");
    const NativeNurbsSurface3D lprism =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);

    require(torus.patches.size() == 16,
            "torus must have 4x4 native patches");
    require(cylinder.patches.size() == 16,
            "cylinder must have four quarters on each of four sheets");
    require(lprism.patches.size() == 12,
            "L-prism must retain its twelve native patches");
    check_patch_regular(torus);
    check_patch_regular(cylinder);
    check_patch_regular(lprism);

    for (const auto& patch : torus.patches) {
        const double u = 0.5 * (patch.domain_start_u() + patch.domain_end_u());
        const double v = 0.5 * (patch.domain_start_v() + patch.domain_end_v());
        const Eigen::Vector3d p = patch.evaluate(u, v);
        const double rho = std::hypot(p.x() - 0.07, p.y() + 0.04);
        const double residual = (rho - 0.55) * (rho - 0.55)
                              + (p.z() - 0.03) * (p.z() - 0.03)
                              - 0.20 * 0.20;
        require(std::abs(residual) < 1.0e-11,
                "torus NURBS patch must be exact");
    }
    require(std::abs(torus.expected_area - 4.0 * pi * pi * 0.55 * 0.20)
                < 1.0e-13,
            "torus exact area");

    for (std::size_t patch_id = 0; patch_id < cylinder.patches.size(); ++patch_id) {
        const auto& patch = cylinder.patches[patch_id];
        const Eigen::Vector3d p = patch.evaluate(
            0.5 * (patch.domain_start_u() + patch.domain_end_u()),
            0.5 * (patch.domain_start_v() + patch.domain_end_v()));
        const double radius = std::hypot(p.x() - 0.06, p.y() + 0.05);
        if (patch_id < 4)
            require(std::abs(radius - 0.55) < 1.0e-12, "outer cylinder wall");
        else if (patch_id < 8)
            require(std::abs(radius - 0.25) < 1.0e-12, "inner cylinder wall");
        else if (patch_id < 12)
            require(std::abs(p.z() - 0.67) < 1.0e-12, "top annulus");
        else
            require(std::abs(p.z() + 0.63) < 1.0e-12, "bottom annulus");
    }

    const double lprism_area = 2.0 * 3.0 * 0.60 * 0.60
                             + 8.0 * 0.60 * 1.30;
    require(std::abs(lprism.expected_area - lprism_area) < 1.0e-13,
            "L-prism exact area");
}

void test_native_surface_interval_topology()
{
    require_throws_type_contains<std::invalid_argument>(
        [] {
            (void)kfbim::geometry3d::NurbsSurfaceModel3D({}, {}, {});
        },
        "NURBS surface model requires at least one patch",
        "empty NURBS surface model is rejected");

    struct Expected {
        GeometryKind3D kind;
        int patches;
        int connections;
    };
    for (const Expected expected : {
             Expected{GeometryKind3D::Torus, 16, 32},
             Expected{GeometryKind3D::HollowCylinder, 16, 32},
             Expected{GeometryKind3D::LPrism, 12, 26}}) {
        const NativeNurbsSurface3D surface =
            make_native_nurbs_surface_3d(expected.kind);
        const kfbim::geometry3d::NurbsSurfaceModel3D model =
            surface.geometry_model();
        const auto diagnostic = model.validate_closed();
        require(model.num_patches() == expected.patches,
                surface.name + " geometry-model patch count");
        require(model.connections().size()
                    == static_cast<std::size_t>(expected.connections),
                surface.name + " interval-connection count");
        require(diagnostic.uncovered_interval_count == 0
                    && diagnostic.multiply_covered_interval_count == 0
                    && diagnostic.position_mismatch_count == 0
                    && diagnostic.orientation_mismatch_count == 0
                    && diagnostic.g1_normal_mismatch_count == 0,
                surface.name + " closed oriented interval topology");
    }

    const kfbim::geometry3d::NurbsSurfaceModel3D open_model(
        {kfbim::geometry3d::NurbsSurfacePatch3D::make_unit_square_xy()},
        {0}, {});
    require_throws_contains(
        [&] { (void)open_model.validate_closed(); },
        "uncovered patch-edge interval",
        "open NURBS patch is rejected by closed-topology validation");
}

std::vector<kfbim::geometry3d::NurbsPatchEdgeConnection3D>
unit_square_pair_connections()
{
    using Edge = kfbim::geometry3d::NurbsPatchEdge3D;
    std::vector<kfbim::geometry3d::NurbsPatchEdgeConnection3D> connections;
    for (const Edge edge : {Edge::UMin, Edge::UMax, Edge::VMin, Edge::VMax}) {
        connections.push_back(
            kfbim::geometry3d::NurbsPatchEdgeConnection3D{
                {0, edge, 0.0, 1.0}, {1, edge, 0.0, 1.0}, false, false});
    }
    return connections;
}

void test_interval_topology_rejects_out_of_domain_endpoint()
{
    using Edge = kfbim::geometry3d::NurbsPatchEdge3D;
    const auto patch = kfbim::geometry3d::NurbsSurfacePatch3D::make_unit_square_xy();
    require_throws_contains(
        [&] {
            (void)kfbim::geometry3d::NurbsSurfaceModel3D(
                {patch}, {0},
                {{{0, Edge::UMin, 0.0, std::nextafter(1.0, 2.0)},
                  {0, Edge::UMax, 0.0, 1.0}, false, false}});
        },
        "outside its patch-edge parameter domain",
        "interval endpoint just outside the domain is rejected");
}

void test_interval_topology_rejects_tiny_gap()
{
    using Edge = kfbim::geometry3d::NurbsPatchEdge3D;
    const auto patch = kfbim::geometry3d::NurbsSurfacePatch3D::make_unit_square_xy();
    auto connections = unit_square_pair_connections();
    connections[0] = kfbim::geometry3d::NurbsPatchEdgeConnection3D{
        {0, Edge::UMin, 0.0, 0.5}, {1, Edge::UMin, 0.0, 0.5}, false, false};
    connections.insert(connections.begin() + 1,
                       kfbim::geometry3d::NurbsPatchEdgeConnection3D{
                           {0, Edge::UMin, std::nextafter(0.5, 1.0), 1.0},
                           {1, Edge::UMin, std::nextafter(0.5, 1.0), 1.0},
                           false,
                           false});
    const kfbim::geometry3d::NurbsSurfaceModel3D model(
        {patch, patch}, {0, 0}, std::move(connections));
    require_throws_contains(
        [&] { (void)model.validate_closed(); },
        "uncovered patch-edge interval",
        "tiny positive patch-edge gap is rejected");
}

void test_interval_topology_rejects_tiny_overlap()
{
    using Edge = kfbim::geometry3d::NurbsPatchEdge3D;
    const auto patch = kfbim::geometry3d::NurbsSurfacePatch3D::make_unit_square_xy();
    auto connections = unit_square_pair_connections();
    connections[0] = kfbim::geometry3d::NurbsPatchEdgeConnection3D{
        {0, Edge::UMin, 0.0, 0.5}, {1, Edge::UMin, 0.0, 0.5}, false, false};
    connections.insert(connections.begin() + 1,
                       kfbim::geometry3d::NurbsPatchEdgeConnection3D{
                           {0, Edge::UMin, std::nextafter(0.5, 0.0), 1.0},
                           {1, Edge::UMin, std::nextafter(0.5, 0.0), 1.0},
                           false,
                           false});
    const kfbim::geometry3d::NurbsSurfaceModel3D model(
        {patch, patch}, {0, 0}, std::move(connections));
    require_throws_contains(
        [&] { (void)model.validate_closed(); },
        "multiply covered patch-edge interval",
        "tiny positive patch-edge overlap is rejected");
}

void check_bezier_element_samples(
    const kfbim::geometry3d::RationalBezierElement3D& element,
    const kfbim::geometry3d::NurbsSurfacePatch3D& patch,
    const std::string& context)
{
    for (const Eigen::Vector4d& control : element.homogeneous_controls) {
        require(control.w() > 0.0,
                context + " positive homogeneous control weight");
    }
    for (int iu = 0; iu <= 4; ++iu) {
        for (int iv = 0; iv <= 4; ++iv) {
            const double u = element.u0()
                + (element.u1() - element.u0()) * iu / 4.0;
            const double v = element.v0()
                + (element.v1() - element.v0()) * iv / 4.0;
            const Eigen::Vector3d exact = patch.evaluate(u, v);
            require((element.evaluate(u, v) - exact).norm() < 2.0e-12,
                    context + " Bezier extraction preserves NURBS evaluation at ("
                        + std::to_string(iu) + ", " + std::to_string(iv) + ")");
            require(element.bounds().contains(exact, 2.0e-12),
                    context + " Bezier control hull conservatively bounds surface");
        }
    }
}

void test_rational_bezier_extraction_and_subdivision()
{
    using kfbim::geometry::NurbsBasis1D;
    using kfbim::geometry3d::NurbsSurfacePatch3D;
    const NurbsSurfacePatch3D multispan_patch(
        NurbsBasis1D(2, {0, 0, 0, 0.5, 1, 1, 1}),
        NurbsBasis1D(1, {0, 0, 1, 1}),
        {{{0.00, 0.0, 0.0}, {0.00, 1.0, 0.0}},
         {{0.25, 0.0, 0.0}, {0.25, 1.0, 0.0}},
         {{0.75, 0.0, 0.0}, {0.75, 1.0, 0.0}},
         {{1.00, 0.0, 0.0}, {1.00, 1.0, 0.0}}},
        {{1.0, 1.2}, {0.8, 1.1}, {1.3, 0.9}, {1.0, 1.4}});
    const kfbim::geometry3d::NurbsSurfaceModel3D model(
        {multispan_patch}, {0}, {});
    const auto elements =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(model);
    require(elements.size() == 2, "two U knot spans produce two elements");
    require(elements[0].u0() == 0.0 && elements[0].u1() == 0.5
                && elements[1].u0() == 0.5 && elements[1].u1() == 1.0,
            "Bezier elements retain their native U knot spans");
    for (const auto& element : elements) {
        require(element.patch_index == 0 && element.component == 0,
                "Bezier element retains patch and component IDs");
        require(element.degree_u == 2 && element.degree_v == 1,
                "Bezier element retains tensor-product degrees");
        check_bezier_element_samples(element, model.patch(0), "original element");
        for (const auto& u_child : element.split_u()) {
            for (const auto& child : u_child.split_v()) {
                check_bezier_element_samples(
                    child, model.patch(0), "de Casteljau child");
            }
        }
    }

    const auto refined =
        kfbim::geometry3d::subdivide_rational_bezier_elements_to_extent_3d(
            elements, 0.2);
    require(refined.size() > elements.size(),
            "extent subdivision refines oversized Bezier elements");
    for (const auto& element : refined) {
        require(element.bounds().max_extent() <= 0.2,
                "extent subdivision returns only bounded leaves");
        check_bezier_element_samples(element, model.patch(0), "extent child");
    }

    kfbim::geometry3d::RationalBezierElement3D deep_element;
    deep_element.patch_index = 0;
    deep_element.component = 0;
    deep_element.degree_u = 1;
    deep_element.degree_v = 0;
    deep_element.parameter_u0 = 0.0;
    deep_element.parameter_u1 = 1.0;
    deep_element.parameter_v0 = 0.0;
    deep_element.parameter_v1 = 1.0;
    deep_element.homogeneous_controls = {
        Eigen::Vector4d(0.0, 0.0, 0.0, 1.0),
        Eigen::Vector4d(1.0, 0.0, 0.0, 1.0)};
    require_throws_contains(
        [&deep_element] {
            (void)kfbim::geometry3d::
                subdivide_rational_bezier_elements_to_extent_3d(
                    {deep_element}, std::ldexp(1.0, -49));
        },
        "Bezier acceleration subdivision exceeded depth 48",
        "extent subdivision reports its depth-48 failure");
}

void test_nurbs_basis_constructor_multiplicity_contract()
{
    using kfbim::geometry::NurbsBasis1D;

    const NurbsBasis1D basis(
        2, {0.0, 0.5, 1.0}, {3, 2, 3});
    require(basis.knots()
                == std::vector<double>({0.0, 0.0, 0.0, 0.5, 0.5,
                                        1.0, 1.0, 1.0}),
            "the maximum legal internal multiplicity p expands exactly");

    require_throws_contains(
        [] { (void)NurbsBasis1D(2, {0.0, 0.5, 1.0}, {3, 3, 3}); },
        "interior knot multiplicity",
        "unique-knot constructor rejects an internal p+1 break");
    require_throws_contains(
        [] {
            (void)NurbsBasis1D(
                2, {0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0});
        },
        "interior knot multiplicity",
        "expanded-knot constructor cannot bypass the internal p+1 rule");
    require_throws_contains(
        [] { (void)NurbsBasis1D(2, {0.0, 1.0}, {3}); },
        "same size",
        "unique knots and multiplicities must have matching sizes");
    require_throws_contains(
        [] { (void)NurbsBasis1D(2, {0.0, 0.5, 1.0}, {4, 1, 3}); },
        "endpoint knot multiplicity",
        "endpoint multiplicity cannot exceed p+1");
    require_throws_contains(
        [] { (void)NurbsBasis1D(2, {0.0, 0.5, 1.0}, {3, 0, 3}); },
        "positive",
        "knot multiplicities must be positive");
    require_throws_contains(
        [] { (void)NurbsBasis1D(2, {0.0, 0.75, 0.5}, {3, 1, 3}); },
        "strictly increasing",
        "unique knot values must be ordered strictly increasingly");
    require_throws_contains(
        [] { (void)NurbsBasis1D(0, {0.0, 0.5, 1.0}, {1, 1, 1}); },
        "split into separate patches",
        "degree-zero multispan geometry must be split into patches");
}

void test_benchmark_rational_bezier_elements_are_conservative()
{
    for (GeometryKind3D kind : {GeometryKind3D::Torus,
                                GeometryKind3D::HollowCylinder,
                                GeometryKind3D::LPrism}) {
        const NativeNurbsSurface3D surface = make_native_nurbs_surface_3d(kind);
        const kfbim::geometry3d::NurbsSurfaceModel3D model =
            surface.geometry_model();
        const auto elements =
            kfbim::geometry3d::extract_rational_bezier_elements_3d(model);
        require(!elements.empty(), surface.name + " has Bezier elements");
        for (const auto& element : elements) {
            check_bezier_element_samples(
                element,
                model.patch(element.patch_index),
                surface.name + " benchmark element");
        }
    }
}

void test_rational_bezier_element_intersection()
{
    const kfbim::geometry3d::NurbsSurfaceModel3D plane_model(
        {kfbim::geometry3d::NurbsSurfacePatch3D::make_unit_square_xy()},
        {0}, {});
    const auto plane_element =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(
            plane_model).front();
    kfbim::geometry3d::NurbsElementIntersectionOptions3D options;
    options.geometry_tolerance = 1e-12;
    const auto plane = kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
        plane_element, plane_model.patch(0),
        {0.25, 0.75, -1.0}, {0.25, 0.75, 1.0}, options);
    require(plane.roots.size() == 1
                && std::abs(plane.roots[0].u - 0.25) < 2e-12
                && std::abs(plane.roots[0].v - 0.75) < 2e-12
                && std::abs(plane.roots[0].t - 0.5) < 2e-12,
            "unit-plane native root");

    const kfbim::geometry3d::NurbsSurfaceModel3D cylinder_model(
        {kfbim::geometry3d::NurbsSurfacePatch3D::
             make_quarter_cylinder_patch(1.0, 0.0, 1.0)},
        {0}, {});
    const auto cylinder_element =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(
            cylinder_model).front();
    const auto cylinder = kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
        cylinder_element, cylinder_model.patch(0),
        {0.8, 0.5, 0.5}, {0.95, 0.5, 0.5}, options);
    require(cylinder.roots.size() == 1
                && std::abs(cylinder.roots[0].point.x() - std::sqrt(0.75))
                       < 2e-12
                && cylinder.roots[0].residual < 2e-12,
            "quarter-cylinder native root");
    require(cylinder.diagnostics.triangle_seed_hits == 0
                && cylinder.diagnostics.roots_recovered_without_triangle_seed == 1,
            "curved root survives triangle-seed miss");

    const auto miss = kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
        plane_element, plane_model.patch(0),
        {2.0, 2.0, -1.0}, {2.0, 2.0, 1.0}, options);
    require(miss.roots.empty() && miss.diagnostics.newton_attempts == 0
                && miss.diagnostics.conservative_rejections > 0,
            "control hull rejects an impossible line before Newton");
}

void test_nurbs_bezier_segment_closest_point()
{
    using kfbim::geometry3d::NurbsElementSegmentClosestPointOptions3D;
    using kfbim::geometry3d::NurbsSurfaceModel3D;
    using kfbim::geometry3d::NurbsSurfacePatch3D;
    using kfbim::geometry3d::closest_point_nurbs_bezier_element_to_segment_3d;

    NurbsElementSegmentClosestPointOptions3D options;
    options.distance_tolerance = 1.0e-12;
    options.parameter_tolerance = 1.0e-12;

    const NurbsSurfaceModel3D plane_model(
        {NurbsSurfacePatch3D::make_unit_square_xy()}, {0}, {});
    const auto plane_element =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(
            plane_model).front();
    const auto separated =
        closest_point_nurbs_bezier_element_to_segment_3d(
            plane_element, plane_model.patch(0),
            {0.3, 0.4, 0.25}, {0.3, 0.4, 0.75}, options);
    require(separated.converged
                && std::abs(separated.u - 0.3) < 2.0e-10
                && std::abs(separated.v - 0.4) < 2.0e-10
                && std::abs(separated.t) < 2.0e-10
                && std::abs(separated.distance - 0.25) < 2.0e-10,
            "bounded closest point finds a positive-distance endpoint minimum");

    const NurbsSurfaceModel3D cylinder_model(
        {NurbsSurfacePatch3D::make_quarter_cylinder_patch(1.0, 0.0, 1.0)},
        {0}, {});
    const auto cylinder_element =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(
            cylinder_model).front();
    const double coordinate = std::sqrt(0.5);
    const Eigen::Vector3d contact(coordinate, coordinate, 0.5);
    const Eigen::Vector3d tangent(-coordinate, coordinate, 0.0);
    const auto touching =
        closest_point_nurbs_bezier_element_to_segment_3d(
            cylinder_element, cylinder_model.patch(0),
            contact - 0.25 * tangent,
            contact + 0.25 * tangent,
            options,
            Eigen::Vector2d(0.45, 0.55));
    require(touching.converged
                && touching.distance < 2.0e-11
                && std::abs(touching.u - 0.5) < 2.0e-9
                && std::abs(touching.v - 0.5) < 2.0e-9
                && std::abs(touching.t - 0.5) < 2.0e-9,
            "bounded closest point recovers a curved tangential contact");
}

void test_closest_point_classifies_terminal_intersection_boxes()
{
    using kfbim::geometry::NurbsBasis1D;
    using kfbim::geometry3d::NurbsElementIntersectionOptions3D;
    using kfbim::geometry3d::NurbsSurfaceModel3D;
    using kfbim::geometry3d::NurbsSurfacePatch3D;

    const NurbsSurfacePatch3D separated_plane(
        NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0}),
        NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0}),
        {{{0.0, -1.0, 2.0}, {0.0, 2.0, -1.0}},
         {{1.0, -1.0, 2.0}, {1.0, 2.0, -1.0}}},
        {{1.0, 1.0}, {1.0, 1.0}});
    const NurbsSurfaceModel3D separated_model(
        {separated_plane}, {0}, {});
    const auto separated_element =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(
            separated_model).front();

    NurbsElementIntersectionOptions3D options;
    options.geometry_tolerance = 1.0e-12;
    options.parameter_tolerance = 1.0e-12;
    options.max_subdivision_depth = 0;
    options.use_triangle_seed = false;
    const auto miss =
        kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
            separated_element, separated_model.patch(0),
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, options);
    require(miss.roots.empty()
                && miss.diagnostics.closest_point_attempts == 1
                && miss.diagnostics.terminal_misses_by_closest_point == 1
                && miss.diagnostics.closest_point_failures == 0
                && miss.diagnostics.unresolved_boxes == 0
                && miss.diagnostics.maximum_subdivision_depth_reached == 0
                && miss.diagnostics.terminal_certificate_boxes == 1
                && miss.diagnostics
                       .maximum_terminal_certificate_depth_reached == 0,
            "control-hull separation certifies a terminal miss");

    std::vector<double> degree_five_knots(6, 0.0);
    degree_five_knots.insert(degree_five_knots.end(), 6, 1.0);
    std::vector<std::vector<Eigen::Vector3d>> high_degree_controls(
        6, std::vector<Eigen::Vector3d>(6, Eigen::Vector3d::Zero()));
    std::vector<std::vector<double>> high_degree_weights(
        6, std::vector<double>(6, 1.0));
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            const double u = static_cast<double>(i) / 5.0;
            const double v = static_cast<double>(j) / 5.0;
            high_degree_controls[static_cast<std::size_t>(i)]
                                [static_cast<std::size_t>(j)] =
                {u, -1.0 + 3.0 * v, 2.0 - 3.0 * v};
        }
    }
    const NurbsSurfaceModel3D high_degree_model(
        {NurbsSurfacePatch3D(
            NurbsBasis1D(5, degree_five_knots),
            NurbsBasis1D(5, degree_five_knots),
            std::move(high_degree_controls),
            std::move(high_degree_weights))},
        {0}, {});
    const auto high_degree_element =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(
            high_degree_model).front();
    const auto high_degree_miss =
        kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
            high_degree_element, high_degree_model.patch(0),
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, options);
    require(high_degree_miss.roots.empty()
                && high_degree_miss.diagnostics.unresolved_boxes == 0
                && high_degree_miss.diagnostics.terminal_certificate_boxes > 1
                && high_degree_miss.diagnostics
                       .maximum_terminal_certificate_depth_reached <= 6
                && high_degree_miss.diagnostics
                       .high_degree_control_hull_fallbacks > 0,
            "high-degree terminal certification has bounded recorded work");

    const NurbsSurfacePatch3D two_root_graph(
        NurbsBasis1D(2, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}),
        NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0}),
        {{{0.0, 0.0, 0.16}, {0.0, 1.0, 0.16}},
         {{0.5, 0.0, -0.34}, {0.5, 1.0, -0.34}},
         {{1.0, 0.0, 0.16}, {1.0, 1.0, 0.16}}},
        {{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}});
    const NurbsSurfaceModel3D two_root_model(
        {two_root_graph}, {0}, {});
    const auto two_root_element =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(
            two_root_model).front();

    kfbim::geometry3d::NurbsElementSegmentClosestPointOptions3D
        arbitration_options;
    arbitration_options.distance_tolerance = 1.0e-12;
    arbitration_options.parameter_tolerance = 1.0e-12;
    arbitration_options.max_iterations = 1;
    const auto closer_failed_seed =
        kfbim::geometry3d::closest_point_nurbs_bezier_element_to_segment_3d(
            two_root_element, two_root_model.patch(0),
            {0.0, 0.5, 0.0}, {1.0, 0.5, 0.0}, arbitration_options,
            Eigen::Vector2d(0.3, 0.5));
    require(!closer_failed_seed.converged
                && closer_failed_seed.distance < 0.02
                && std::abs(closer_failed_seed.u - 0.5) > 0.1,
            "closer failed seed is retained over a farther stationary seed");
    arbitration_options.max_iterations =
        std::numeric_limits<int>::max();
    require_throws_contains(
        [&] {
            (void)kfbim::geometry3d::
                closest_point_nurbs_bezier_element_to_segment_3d(
                    two_root_element, two_root_model.patch(0),
                    {0.0, 0.5, 0.0}, {1.0, 0.5, 0.0},
                    arbitration_options);
        },
        "permit two-seed accounting",
        "closest-point iteration count rejects signed overflow");

    require_throws_type_contains<
        kfbim::geometry3d::UnresolvedNurbsIntersectionCandidate3D>(
        [&] {
            (void)kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
                two_root_element, two_root_model.patch(0),
                {0.0, 0.5, 0.0}, {1.0, 0.5, 0.0}, options);
        },
        "unresolved conservative NURBS intersection candidate",
        "stationary positive distance cannot hide two terminal roots");

    const NurbsSurfaceModel3D cylinder_model(
        {NurbsSurfacePatch3D::make_quarter_cylinder_patch(1.0, 0.0, 1.0)},
        {0}, {});
    const auto cylinder_element =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(
            cylinder_model).front();
    const auto contact_data =
        cylinder_model.patch(0).evaluate_with_derivatives(0.4, 0.6);
    const Eigen::Vector3d contact = contact_data.point;
    const Eigen::Vector3d tangent = contact_data.du.normalized();
    options.max_newton_iterations = 1;
    bool tangent_failed_closed = false;
    try {
        (void)kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
            cylinder_element, cylinder_model.patch(0),
            contact - 0.25 * tangent,
            contact + 0.25 * tangent,
            options);
    } catch (
        const kfbim::geometry3d::UnresolvedNurbsIntersectionCandidate3D&
            error) {
        const auto& partial = error.partial_result();
        tangent_failed_closed =
            partial.roots.size() == 1
            && partial.roots.front().residual
                   <= 8.0 * options.geometry_tolerance
            && partial.roots.front().transversality < 1.0e-10
            && partial.diagnostics.closest_point_attempts == 1
            && partial.diagnostics.roots_recovered_by_closest_point == 1
            && partial.diagnostics.unresolved_boxes == 1;
    }
    require(tangent_failed_closed,
            "uncertified terminal tangent is retained in a fail-closed result");
}

void test_bezier_element_intersection_isolates_all_roots_and_fails_safe()
{
    using kfbim::geometry::NurbsBasis1D;
    using kfbim::geometry3d::NurbsSurfacePatch3D;

    const NurbsSurfacePatch3D quadratic_graph(
        NurbsBasis1D(2, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}),
        NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0}),
        {{{0.0, 0.0, 0.14}, {0.0, 1.0, 0.14}},
         {{0.5, 0.0, -0.31}, {0.5, 1.0, -0.31}},
         {{1.0, 0.0, 0.24}, {1.0, 1.0, 0.24}}},
        {{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}});
    const kfbim::geometry3d::NurbsSurfaceModel3D model(
        {quadratic_graph}, {0}, {});
    const auto element =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(model).front();

    kfbim::geometry3d::NurbsElementIntersectionOptions3D options;
    options.geometry_tolerance = 1.0e-12;
    const auto result = kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
        element, model.patch(0),
        {0.0, 0.5, 0.0}, {1.0, 0.5, 0.0}, options);
    require(result.roots.size() == 2
                && std::abs(result.roots[0].u - 0.2) < 2.0e-11
                && std::abs(result.roots[1].u - 0.7) < 2.0e-11,
            "quadratic ruled graph returns both isolated roots");

    options.max_subdivision_depth = 0;
    require_throws_contains(
        [&] {
            (void)kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
                element, model.patch(0),
                {0.0, 0.5, 0.0}, {1.0, 0.5, 0.0}, options);
        },
        "unresolved conservative NURBS intersection candidate",
        "depth-zero multi-root box reports the unresolved diagnostic");
}

void test_analytic_ruled_graph_root_cases()
{
    using kfbim::geometry::NurbsBasis1D;
    using kfbim::geometry3d::NurbsSurfaceModel3D;
    using kfbim::geometry3d::NurbsSurfacePatch3D;
    using kfbim::geometry3d::UnresolvedNurbsIntersectionCandidate3D;

    const auto solve_graph = [](
        int degree, const std::vector<double>& bernstein_coefficients) {
        std::vector<double> knots(
            static_cast<std::size_t>(2 * (degree + 1)), 1.0);
        std::fill(knots.begin(), knots.begin() + degree + 1, 0.0);
        std::vector<std::vector<Eigen::Vector3d>> controls(
            static_cast<std::size_t>(degree + 1),
            std::vector<Eigen::Vector3d>(2));
        std::vector<std::vector<double>> weights(
            static_cast<std::size_t>(degree + 1),
            std::vector<double>(2, 1.0));
        for (int i = 0; i <= degree; ++i) {
            for (int j = 0; j < 2; ++j) {
                controls[static_cast<std::size_t>(i)]
                        [static_cast<std::size_t>(j)] = {
                    static_cast<double>(i) / degree,
                    static_cast<double>(j),
                    bernstein_coefficients[static_cast<std::size_t>(i)]};
            }
        }
        const NurbsSurfaceModel3D model(
            {NurbsSurfacePatch3D(
                NurbsBasis1D(degree, knots),
                NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0}),
                std::move(controls), std::move(weights))},
            {0}, {});
        const auto element =
            kfbim::geometry3d::extract_rational_bezier_elements_3d(model)
                .front();
        kfbim::geometry3d::NurbsElementIntersectionOptions3D options;
        options.geometry_tolerance = 1.0e-12;
        try {
            return std::make_pair(
                kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
                    element, model.patch(0),
                    {0.0, 0.5, 0.0}, {1.0, 0.5, 0.0}, options),
                false);
        } catch (const UnresolvedNurbsIntersectionCandidate3D& error) {
            return std::make_pair(error.partial_result(), true);
        }
    };

    const auto three = solve_graph(3, {-0.08, 0.14, -0.14, 0.08});
    require(!three.second && three.first.roots.size() == 3
                && std::abs(three.first.roots[0].u - 0.2) < 2.0e-10
                && std::abs(three.first.roots[1].u - 0.5) < 2.0e-10
                && std::abs(three.first.roots[2].u - 0.8) < 2.0e-10,
            "cubic ruled graph returns all three transverse roots");

    const auto contact = solve_graph(2, {0.25, -0.25, 0.25});
    require(contact.first.roots.size() == 1
                && std::abs(contact.first.roots.front().u - 0.5) < 2.0e-10
                && contact.first.roots.front().transversality < 1.0e-10,
            "quadratic ruled graph preserves its tangent contact");

    const auto odd_flat = solve_graph(
        3, {-0.125, 0.125, -0.125, 0.125});
    require(odd_flat.second
                && odd_flat.first.diagnostics.unresolved_boxes > 0,
            "zero-derivative odd crossing remains an unresolved candidate");
}

kfbim::geometry3d::NurbsSurfaceModel3D
make_single_patch_periodic_torus();

void test_close_roots_have_stationary_witness()
{
    require(kfbim::geometry3d::NurbsElementIntersectionOptions3D{}
                    .max_subdivision_depth == 4,
            "element intersection default depth is bounded");
    require(kfbim::geometry3d::NurbsSurfaceIntersectorOptions3D{}
                    .local_max_subdivision_depth == 4,
            "surface intersection default depth is bounded");
    using kfbim::geometry::NurbsBasis1D;
    using kfbim::geometry3d::NurbsParameterBox2D;
    using kfbim::geometry3d::NurbsSurfaceModel3D;
    using kfbim::geometry3d::NurbsSurfacePatch3D;

    constexpr double separation = 1.0e-3;
    constexpr double endpoint_value = 0.25 - separation * separation;
    const NurbsSurfacePatch3D graph(
        NurbsBasis1D(2, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}),
        NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0}),
        {{{0.0, 0.0, endpoint_value}, {0.0, 1.0, endpoint_value}},
         {{0.5, 0.0, -0.25 - separation * separation},
          {0.5, 1.0, -0.25 - separation * separation}},
         {{1.0, 0.0, endpoint_value}, {1.0, 1.0, endpoint_value}}},
        {{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}});
    const NurbsSurfaceModel3D model({graph}, {0}, {});
    const auto element =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(model).front();
    const Eigen::Vector3d start(0.0, 0.5, 0.0);
    const Eigen::Vector3d end(1.0, 0.5, 0.0);

    kfbim::geometry3d::NurbsElementIntersectionOptions3D intersection_options;
    intersection_options.geometry_tolerance = 1.0e-12;
    intersection_options.max_subdivision_depth = 4;
    intersection_options.parameter_seeds = {
        {0.5 - separation, 0.5, 0.5 - separation},
        {0.5 + separation, 0.5, 0.5 + separation}};
    kfbim::geometry3d::NurbsElementIntersectionResult3D roots;
    try {
        roots = kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
            element, model.patch(0), start, end, intersection_options);
    } catch (const kfbim::geometry3d::
                 UnresolvedNurbsIntersectionCandidate3D& error) {
        roots = error.partial_result();
    }
    require(roots.roots.size() == 2,
            "close ruled-graph roots remain distinct");

    for (int N : {32, 64, 128}) {
        const double half_width = 2.0 / static_cast<double>(N);
        const auto stationary =
            kfbim::geometry3d::
                solve_nurbs_bezier_segment_stationary_point_3d(
                    element, model.patch(0), start, end,
                    NurbsParameterBox2D{
                        0.5 - half_width, 0.5 + half_width, 0.25, 0.75},
                    Eigen::Vector2d(0.5, 0.5));
        require(stationary.converged
                    && std::abs(stationary.u - 0.5) < 2.0e-10
                    && std::abs(stationary.t - 0.5) < 2.0e-10
                    && stationary.distance > 0.5e-6
                    && stationary.distance < 2.0e-6
                    && stationary.tangent_measure < 1.0e-10,
                "close roots retain a positive stationary witness at N="
                    + std::to_string(N));
        const auto stationary_points =
            kfbim::geometry3d::
                find_nurbs_bezier_segment_stationary_points_3d(
                    element, model.patch(0), start, end,
                    NurbsParameterBox2D{
                        0.5 - half_width, 0.5 + half_width, 0.25, 0.75},
                    {Eigen::Vector2d(0.5, 0.5),
                     Eigen::Vector2d(0.5 - 1.0e-5, 0.5),
                     Eigen::Vector2d(0.5 + 1.0e-5, 0.5)});
        require(stationary_points.size() == 1
                    && stationary_points.front().distance > 0.5e-6,
                "stationary seeds deduplicate to one witness at N="
                    + std::to_string(N));
    }

    constexpr double cross_section_offset = 0.10;
    kfbim::geometry3d::NurbsSurfaceIntersectorOptions3D surface_options;
    surface_options.local_max_subdivision_depth = 4;

    const kfbim::geometry3d::NurbsSurfaceIntersector3D periodic_intersector(
        make_single_patch_periodic_torus(), surface_options);
    const auto periodic_edge = periodic_intersector.intersect_cartesian_edge(
        kfbim::geometry3d::NurbsCartesianEdgeQuery3D{
            2, 80, 81, 82,
            {1.0 + cross_section_offset, 0.0, -20.0},
            {1.0 + cross_section_offset, 0.0, 20.0}});
    require(periodic_edge.crossings.size() == 2
                && periodic_edge.crossings[0].patch_index
                       == periodic_edge.crossings[1].patch_index
                && periodic_edge.ambiguous_clusters.empty()
                && periodic_edge.root_count_known
                && periodic_edge.diagnostics
                       .root_pairs_protected_by_stationary_witness >= 1,
            "close roots across same-patch elements retain a witness");

    const NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    constexpr double inverse_sqrt_two =
        0.707106781186547524400844362104849039;
    const Eigen::Vector3d radial(
        inverse_sqrt_two, inverse_sqrt_two, 0.0);

    const Eigen::Vector3d g1_start =
        Eigen::Vector3d(0.07, -0.04, 0.03)
        + cross_section_offset * Eigen::Vector3d::UnitZ()
        + 0.10 * radial;
    const Eigen::Vector3d g1_end =
        Eigen::Vector3d(0.07, -0.04, 0.03)
        + cross_section_offset * Eigen::Vector3d::UnitZ()
        + 40.0 * radial;
    const kfbim::geometry3d::NurbsSurfaceIntersector3D g1_intersector(
        torus.geometry_model(), surface_options);
    const auto g1_edge = g1_intersector.intersect_cartesian_edge(
        kfbim::geometry3d::NurbsCartesianEdgeQuery3D{
            0, 83, 84, 85, g1_start, g1_end});
    require(g1_edge.crossings.size() == 2
                && g1_edge.crossings[0].patch_index
                       != g1_edge.crossings[1].patch_index
                && g1_edge.ambiguous_clusters.empty()
                && g1_edge.root_count_known
                && g1_edge.diagnostics
                       .root_pairs_protected_by_stationary_witness >= 1,
            "close roots across a G1 patch edge retain a witness");

    auto non_g1_connections = torus.geometry_model().connections();
    bool changed_connection = false;
    for (auto& connection : non_g1_connections) {
        const int first = connection.first.patch;
        const int second = connection.second.patch;
        if (connection.g1
            && ((first == 0 && second == 1)
                || (first == 1 && second == 0))) {
            connection.g1 = false;
            changed_connection = true;
        }
    }
    require(changed_connection, "torus fixture contains the selected G1 edge");
    std::vector<int> torus_components;
    for (int patch = 0;
         patch < torus.geometry_model().num_patches(); ++patch) {
        torus_components.push_back(
            torus.geometry_model().patch_component(patch));
    }
    const kfbim::geometry3d::NurbsSurfaceIntersector3D non_g1_intersector(
        kfbim::geometry3d::NurbsSurfaceModel3D(
            torus.geometry_model().patches(), std::move(torus_components),
            std::move(non_g1_connections)),
        surface_options);
    const auto non_g1_edge = non_g1_intersector.intersect_cartesian_edge(
        kfbim::geometry3d::NurbsCartesianEdgeQuery3D{
            0, 86, 87, 88, g1_start, g1_end});
    require(non_g1_edge.crossings.size() == 2
                && non_g1_edge.diagnostics
                       .root_pairs_protected_by_stationary_witness == 0,
            "close-root witness does not cross a non-G1 patch edge");
}

void test_split_boundary_root_is_not_ambiguous()
{
    using kfbim::geometry3d::NurbsCartesianEdgeQuery3D;
    using kfbim::geometry3d::NurbsSurfaceIntersector3D;
    using kfbim::geometry3d::NurbsSurfaceIntersectorOptions3D;

    const NativeNurbsSurface3D lprism =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    NurbsSurfaceIntersectorOptions3D options;
    options.maximum_element_extent = 0.7;
    options.local_max_subdivision_depth = 4;
    const NurbsSurfaceIntersector3D intersector(
        lprism.geometry_model(), options);
    const auto edge = intersector.intersect_cartesian_edge(
        NurbsCartesianEdgeQuery3D{
            0, 70, 71, 72,
            {-0.60, -0.30, 0.02}, {-0.46, -0.30, 0.02}});
    require(edge.crossings.size() == 1
                && edge.ambiguous_clusters.empty()
                && edge.root_count_known
                && edge.diagnostics.same_patch_deduplications >= 1
                && edge.diagnostics.stationary_witnesses == 0,
            "a root on a query-leaf split is one unambiguous root");
}
void test_ruled_overlap_certificate_is_physical_and_fail_safe()
{
    using kfbim::geometry::NurbsBasis1D;
    using kfbim::geometry3d::NurbsSurfacePatch3D;

    const kfbim::geometry3d::NurbsSurfaceModel3D cylinder_model(
        {NurbsSurfacePatch3D::make_quarter_cylinder_patch(1.0, 0.0, 1.0)},
        {0}, {});
    const auto cylinder_element =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(
            cylinder_model).front();
    kfbim::geometry3d::NurbsElementIntersectionOptions3D options;
    options.geometry_tolerance = 1.0e-12;
    const auto ruled = kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
        cylinder_element, cylinder_model.patch(0),
        {std::sqrt(0.5), std::sqrt(0.5), 0.2},
        {std::sqrt(0.5), std::sqrt(0.5), 0.8}, options);
    require(ruled.overlap_detected && ruled.roots.empty(),
            "exact constant-weight cylinder ruling certifies overlap");

    constexpr double large_coordinate = 1.0e12;
    constexpr double weight_perturbation = 1.0e-14;
    const std::vector<double> ruling_x{0.0, 0.5, 0.5, 0.5, 1.0};
    std::vector<std::vector<Eigen::Vector3d>> controls(
        3, std::vector<Eigen::Vector3d>(5));
    for (int j = 0; j < 5; ++j) {
        controls[0][static_cast<std::size_t>(j)] =
            {ruling_x[static_cast<std::size_t>(j)], 0.0, 0.0};
        controls[1][static_cast<std::size_t>(j)] =
            {ruling_x[static_cast<std::size_t>(j)], large_coordinate, 0.0};
        controls[2][static_cast<std::size_t>(j)] =
            {ruling_x[static_cast<std::size_t>(j)], 0.0, large_coordinate};
    }
    std::vector<std::vector<double>> weights(
        3, std::vector<double>(5, 1.0));
    weights[1][1] += weight_perturbation;
    weights[1][3] -= weight_perturbation;
    const NurbsSurfacePatch3D near_ruled(
        NurbsBasis1D(2, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}),
        NurbsBasis1D(
            4, {0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0, 1.0}),
        std::move(controls), std::move(weights));
    const kfbim::geometry3d::NurbsSurfaceModel3D near_model(
        {near_ruled}, {0}, {});
    const auto near_element =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(
            near_model).front();

    auto near_options = options;
    near_options.max_subdivision_depth = 0;
    bool failed_safely = false;
    try {
        const auto near_result =
            kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
                near_element, near_model.patch(0),
                {0.0, 0.5 * large_coordinate,
                      0.25 * large_coordinate},
                {1.0, 0.5 * large_coordinate,
                      0.25 * large_coordinate},
                near_options);
        failed_safely = !near_result.overlap_detected;
    } catch (const std::exception& error) {
        failed_safely = std::string(error.what()).find(
            "unresolved conservative NURBS intersection candidate")
            != std::string::npos;
    }
    require(failed_safely,
            "near-separable large-coordinate control net cannot certify overlap");
}

void test_exact_predicate_triangle_seed_on_thin_element()
{
    using kfbim::geometry::NurbsBasis1D;
    using kfbim::geometry3d::NurbsSurfacePatch3D;

    constexpr double width = 1.0e-18;
    const NurbsSurfacePatch3D thin_patch(
        NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0}),
        NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0}),
        {{{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
         {{1.0, 0.0, 0.0}, {1.0, width, 0.0}}},
        {{1.0, 1.0}, {1.0, 1.0}});
    const kfbim::geometry3d::NurbsSurfaceModel3D model(
        {thin_patch}, {0}, {});
    const auto element =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(model).front();
    kfbim::geometry3d::NurbsElementIntersectionOptions3D options;
    options.geometry_tolerance = 1.0e-16;
    const auto result = [&] {
        try {
            return kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
                element, model.patch(0),
                {0.25, 0.1 * width, -1.0},
                {0.25, 0.1 * width, 1.0}, options);
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("thin exact-predicate seed: ") + error.what());
        }
    }();
    require(result.roots.size() == 1
                && result.diagnostics.triangle_seed_hits >= 1,
            "exact-predicate triangle seed survives a thin element");
}

void test_reweighted_ruled_overlap_certificate()
{
    using kfbim::geometry3d::NurbsSurfacePatch3D;

    const NurbsSurfacePatch3D base =
        NurbsSurfacePatch3D::make_quarter_cylinder_patch(1.0, 0.0, 1.0);
    auto weights = base.weights();
    for (auto& row : weights) {
        require(row.size() == 2,
                "quarter-cylinder ruling has two axial controls");
        row[1] *= 2.0;
    }
    const NurbsSurfacePatch3D reweighted(
        base.basis_u(), base.basis_v(), base.control_net(),
        std::move(weights));
    const kfbim::geometry3d::NurbsSurfaceModel3D model(
        {reweighted}, {0}, {});
    const auto element =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(model).front();
    kfbim::geometry3d::NurbsElementIntersectionOptions3D options;
    options.geometry_tolerance = 1.0e-12;
    const double coordinate = std::sqrt(0.5);

    const auto interior =
        kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
            element, model.patch(0),
            {coordinate, coordinate, 0.2},
            {coordinate, coordinate, 0.8}, options);
    require(interior.overlap_detected && interior.roots.empty(),
            "separable positive weights preserve a ruled overlap");

    const auto narrow =
        kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
            element, model.patch(0),
            {coordinate, coordinate, 0.999},
            {coordinate, coordinate, 100.0}, options);
    require(narrow.overlap_detected && narrow.roots.empty(),
            "transverse-curve solve finds a narrow ruling overlap");
    auto boundary_options = options;
    boundary_options.max_subdivision_depth = 8;
    const auto boundary_ruling =
        kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
            element, model.patch(0),
            {1.0, 0.0, 0.2}, {1.0, 0.0, 0.8}, boundary_options);
    require(boundary_ruling.overlap_detected,
            "positive-length boundary ruling is classified as overlap");
}

void test_nonclamped_rational_bezier_extraction()
{
    using kfbim::geometry::NurbsBasis1D;
    using kfbim::geometry3d::NurbsSurfacePatch3D;
    const NurbsSurfacePatch3D patch(
        NurbsBasis1D(2, {0, 1, 2, 3, 4, 5}),
        NurbsBasis1D(1, {0, 0, 1, 1}),
        {{{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
         {{2.0, 0.0, 0.0}, {2.0, 1.0, 0.0}},
         {{4.0, 0.0, 0.0}, {4.0, 1.0, 0.0}}},
        {{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}});
    const kfbim::geometry3d::NurbsSurfaceModel3D model({patch}, {0}, {});
    const auto elements =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(model);
    require(elements.size() == 1,
            "non-clamped active domain produces one Bezier element");
    check_bezier_element_samples(
        elements.front(), patch, "non-clamped element");

    const NurbsSurfacePatch3D cubic_patch(
        NurbsBasis1D(3, {0, 1, 2, 3, 4, 4, 5, 6}),
        NurbsBasis1D(1, {0, 0, 1, 1}),
        {{{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
         {{1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}},
         {{2.0, 0.0, 0.0}, {2.0, 1.0, 0.0}},
         {{3.0, 0.0, 0.0}, {3.0, 1.0, 0.0}}},
        {{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}});
    const kfbim::geometry3d::NurbsSurfaceModel3D cubic_model(
        {cubic_patch}, {0}, {});
    const auto cubic_elements =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(cubic_model);
    require(cubic_elements.size() == 1,
            "non-clamped cubic active domain produces one Bezier element");
    const std::array<double, 4> expected_cubic_x{
        13.0 / 12.0, 1.5, 2.0, 2.5};
    for (int i = 0; i < 4; ++i) {
        const Eigen::Vector4d& control = cubic_elements.front().control(i, 0);
        require(std::abs(control.x() / control.w()
                         - expected_cubic_x[static_cast<std::size_t>(i)])
                    < 2.0e-12,
                "non-clamped cubic extraction has exact control "
                    + std::to_string(i));
    }
    check_bezier_element_samples(
        cubic_elements.front(), cubic_patch, "non-clamped cubic element");
}

void test_bezier_subdivision_rejects_collapsed_midpoint()
{
    kfbim::geometry3d::RationalBezierElement3D element;
    element.degree_u = 1;
    element.degree_v = 0;
    element.parameter_u0 = 1.0;
    element.parameter_u1 = std::nextafter(1.0, 2.0);
    element.parameter_v0 = 0.0;
    element.parameter_v1 = 1.0;
    element.homogeneous_controls = {
        {0.0, 0.0, 0.0, 1.0},
        {1.0, 0.0, 0.0, 1.0}};
    require_throws_contains(
        [&] { (void)element.split_u(); },
        "representable midpoint",
        "Bezier split rejects a collapsed floating-point midpoint");
    require_throws_contains(
        [&] {
            (void)kfbim::geometry3d::
                subdivide_rational_bezier_elements_to_extent_3d(
                    {element}, 0.5);
        },
        "Bezier acceleration subdivision exceeded depth 48",
        "oversized unsplittable Bezier element uses the depth-limit failure");
}

void test_bezier_split_preserves_large_finite_controls()
{
    const double maximum = std::numeric_limits<double>::max();
    const double weight = 0.25 * maximum;
    kfbim::geometry3d::RationalBezierElement3D element;
    element.degree_u = 1;
    element.degree_v = 0;
    element.parameter_u0 = 0.0;
    element.parameter_u1 = 1.0;
    element.parameter_v0 = 0.0;
    element.parameter_v1 = 1.0;
    element.homogeneous_controls = {
        {maximum, 0.0, 0.0, weight},
        {0.75 * maximum, 0.0, 0.0, weight}};

    const auto children = element.split_u();
    for (const auto& child : children) {
        for (const Eigen::Vector4d& control : child.homogeneous_controls) {
            require(control.allFinite(),
                    "large finite Bezier split retains finite controls");
        }
        require(child.bounds().max_extent() <= 0.5,
                "large finite Bezier child has the exact half extent");
    }
    require(std::abs(children[0].evaluate(0.5, 0.5).x() - 3.5) < 1.0e-14
                && std::abs(children[1].evaluate(0.5, 0.5).x() - 3.5)
                       < 1.0e-14,
            "large finite Bezier children meet at their exact midpoint");

    const auto refined =
        kfbim::geometry3d::subdivide_rational_bezier_elements_to_extent_3d(
            {element}, 0.75);
    require(refined.size() == 2,
            "large finite Bezier extent subdivision produces two leaves");
    for (const auto& child : refined) {
        require(child.bounds().max_extent() <= 0.75,
                "large finite Bezier extent leaf satisfies the limit");
    }
}

void require_intersector_bounds_dense_samples(
    const NativeNurbsSurface3D& surface,
    const kfbim::geometry3d::NurbsSurfaceIntersector3D& intersector)
{
    for (const auto& patch : surface.patches) {
        for (int iu = 0; iu <= 16; ++iu) {
            const double u = patch.domain_start_u()
                + (patch.domain_end_u() - patch.domain_start_u())
                    * static_cast<double>(iu) / 16.0;
            for (int iv = 0; iv <= 16; ++iv) {
                const double v = patch.domain_start_v()
                    + (patch.domain_end_v() - patch.domain_start_v())
                        * static_cast<double>(iv) / 16.0;
                require(intersector.bounds().contains(
                            patch.evaluate(u, v),
                            intersector.geometry_tolerance()),
                        surface.name + " intersector bounds contain dense samples");
            }
        }
    }
}

kfbim::geometry3d::NurbsSurfaceModel3D make_single_patch_periodic_torus()
{
    using kfbim::geometry::NurbsBasis1D;
    using kfbim::geometry3d::NurbsPatchEdge3D;
    using kfbim::geometry3d::NurbsPatchEdgeConnection3D;
    using kfbim::geometry3d::NurbsSurfacePatch3D;

    constexpr double inverse_sqrt_two =
        0.707106781186547524400844362104849039;
    const std::vector<double> knots{
        0.0, 0.0, 0.0,
        0.25, 0.25,
        0.5, 0.5,
        0.75, 0.75,
        1.0, 1.0, 1.0};
    const std::array<Eigen::Vector2d, 9> circle_controls{{
        {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0},
        {-1.0, 1.0}, {-1.0, 0.0}, {-1.0, -1.0},
        {0.0, -1.0}, {1.0, -1.0}, {1.0, 0.0}}};
    const std::array<double, 9> circle_weights{{
        1.0, inverse_sqrt_two, 1.0,
        inverse_sqrt_two, 1.0, inverse_sqrt_two,
        1.0, inverse_sqrt_two, 1.0}};

    constexpr double major_radius = 1.0;
    constexpr double minor_radius = 0.2;
    std::vector<std::vector<Eigen::Vector3d>> controls(
        9, std::vector<Eigen::Vector3d>(9));
    std::vector<std::vector<double>> weights(
        9, std::vector<double>(9));
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            const double radius = major_radius
                + minor_radius
                    * circle_controls[static_cast<std::size_t>(j)].x();
            controls[static_cast<std::size_t>(i)]
                    [static_cast<std::size_t>(j)] = {
                radius * circle_controls[static_cast<std::size_t>(i)].x(),
                radius * circle_controls[static_cast<std::size_t>(i)].y(),
                minor_radius
                    * circle_controls[static_cast<std::size_t>(j)].y()};
            weights[static_cast<std::size_t>(i)]
                   [static_cast<std::size_t>(j)] =
                circle_weights[static_cast<std::size_t>(i)]
                * circle_weights[static_cast<std::size_t>(j)];
        }
    }

    NurbsSurfacePatch3D patch(
        NurbsBasis1D(2, knots), NurbsBasis1D(2, knots),
        std::move(controls), std::move(weights));
    std::vector<NurbsPatchEdgeConnection3D> connections{
        {{0, NurbsPatchEdge3D::UMin, 0.0, 1.0},
         {0, NurbsPatchEdge3D::UMax, 0.0, 1.0}, false, true},
        // Store the first half in the opposite endpoint order and split both
        // periodic sides, exercising orientation by interval membership.
        {{0, NurbsPatchEdge3D::VMax, 0.0, 0.5},
         {0, NurbsPatchEdge3D::VMin, 0.0, 0.5}, false, true},
        {{0, NurbsPatchEdge3D::VMin, 0.5, 1.0},
         {0, NurbsPatchEdge3D::VMax, 0.5, 1.0}, false, true}};
    return kfbim::geometry3d::NurbsSurfaceModel3D(
        {std::move(patch)}, {0}, std::move(connections));
}

void test_single_patch_periodic_seam_canonicalization()
{
    using kfbim::geometry3d::NurbsSurfaceIntersector3D;
    const NurbsSurfaceIntersector3D intersector(
        make_single_patch_periodic_torus());
    const auto hit = intersector.intersect_segment(
        {1.19, 0.0, 0.0}, {1.21, 0.0, 0.0});
    require(hit.crossings.size() == 1
                && hit.diagnostics.seam_deduplications >= 1,
            "single-patch periodic seam root is canonicalized once");
    require(intersector.containing_components({1.0, 0.0, 0.0})
                == std::vector<int>{0},
            "single-patch periodic seam preserves component parity");
}

kfbim::geometry3d::NurbsSurfaceModel3D
make_mixed_tangent_unresolved_model()
{
    using kfbim::geometry::NurbsBasis1D;
    using kfbim::geometry3d::NurbsPatchEdge3D;
    using kfbim::geometry3d::NurbsPatchEdgeConnection3D;
    using kfbim::geometry3d::NurbsSurfacePatch3D;

    // Bernstein coefficients of
    // (u - 0.5)^2 (u - 0.7)^2.  Both contacts are singular candidates;
    // finding a witness for either one cannot certify the other away.
    constexpr std::array<double, 5> graph_coefficients{{
        0.1225, -0.0875, 0.05916666666666667, -0.0375, 0.0225}};
    auto make_controls = [&](bool reverse_u) {
        std::vector<std::vector<Eigen::Vector3d>> controls(
            5, std::vector<Eigen::Vector3d>(2));
        for (int i = 0; i < 5; ++i) {
            const int source = reverse_u ? 4 - i : i;
            for (int j = 0; j < 2; ++j) {
                controls[static_cast<std::size_t>(i)]
                        [static_cast<std::size_t>(j)] = {
                    static_cast<double>(source) / 4.0,
                    static_cast<double>(j),
                    graph_coefficients[static_cast<std::size_t>(source)]};
            }
        }
        return controls;
    };
    const std::vector<std::vector<double>> weights(
        5, std::vector<double>(2, 1.0));
    constexpr double narrow_v_begin = 1.0;
    constexpr double narrow_v_end = 1.0 + 1.0e-10;
    const NurbsBasis1D quartic(
        4, {0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0});
    const NurbsBasis1D linear(
        1, {narrow_v_begin, narrow_v_begin,
            narrow_v_end, narrow_v_end});
    NurbsSurfacePatch3D forward(
        quartic, linear, make_controls(false), weights);
    NurbsSurfacePatch3D reverse(
        quartic, linear, make_controls(true), weights);
    std::vector<NurbsPatchEdgeConnection3D> connections{
        {{0, NurbsPatchEdge3D::UMin, narrow_v_begin, narrow_v_end},
         {1, NurbsPatchEdge3D::UMax, narrow_v_begin, narrow_v_end},
         false, false},
        {{0, NurbsPatchEdge3D::UMax, narrow_v_begin, narrow_v_end},
         {1, NurbsPatchEdge3D::UMin, narrow_v_begin, narrow_v_end},
         false, false},
        {{0, NurbsPatchEdge3D::VMin, 0.0, 1.0},
         {1, NurbsPatchEdge3D::VMin, 0.0, 1.0}, true, false},
        {{0, NurbsPatchEdge3D::VMax, 0.0, 1.0},
         {1, NurbsPatchEdge3D::VMax, 0.0, 1.0}, true, false}};
    return kfbim::geometry3d::NurbsSurfaceModel3D(
        {std::move(forward), std::move(reverse)}, {0, 0},
        std::move(connections));
}

void test_tangent_witness_does_not_swallow_unresolved_candidate()
{
    const kfbim::geometry3d::NurbsSurfaceIntersector3D intersector(
        make_mixed_tangent_unresolved_model());
    require_throws_contains(
        [&] {
            (void)intersector.intersect_segment(
                {0.0, 0.5, 0.0}, {1.0, 0.5, 0.0});
        },
        "unresolved conservative NURBS intersection candidate",
        "tangent witness preserves a coexisting unresolved candidate");
    const auto unresolved_edge = intersector.intersect_cartesian_edge(
        kfbim::geometry3d::NurbsCartesianEdgeQuery3D{
            0, 60, 61, 62,
            {0.0, 0.5, 0.0}, {1.0, 0.5, 0.0}});
    require(!unresolved_edge.root_count_known
                && !unresolved_edge.parity_known_from_roots
                && !unresolved_edge.crossings.empty(),
            "Cartesian unresolved query preserves roots and unknown parity");
}

void test_native_nurbs_surface_intersector()
{
    using kfbim::geometry3d::NurbsCartesianEdgeQuery3D;
    using kfbim::geometry3d::NurbsSurfaceIntersector3D;

    require_throws_contains(
        [] {
            (void)NurbsSurfaceIntersector3D(
                kfbim::geometry3d::NurbsSurfaceModel3D(
                    {kfbim::geometry3d::NurbsSurfacePatch3D::
                         make_unit_square_xy()},
                    {0}, {}));
        },
        "uncovered patch-edge interval",
        "surface intersector rejects an open model");

    const NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    const NurbsSurfaceIntersector3D torus_intersector(torus.geometry_model());
    require_intersector_bounds_dense_samples(torus, torus_intersector);
    const auto leaves = torus_intersector.acceleration_leaves(0.05);
    require(std::all_of(leaves.begin(), leaves.end(), [](const auto& leaf) {
                return leaf.bounds().max_extent() <= 0.05 + 1e-13;
            }),
            "all acceleration leaves satisfy requested extent");

    kfbim::geometry3d::NurbsSurfaceIntersectorOptions3D
        grid_scaled_options;
    grid_scaled_options.maximum_element_extent = 0.05;
    grid_scaled_options.local_max_subdivision_depth = 4;
    const NurbsSurfaceIntersector3D grid_scaled_intersector(
        torus.geometry_model(), grid_scaled_options);
    require(grid_scaled_intersector.query_element_count()
                == grid_scaled_intersector.acceleration_leaves(0.05).size()
                && grid_scaled_intersector.maximum_query_element_extent()
                       <= 0.05 + 1.0e-13
                && grid_scaled_intersector.query_element_count()
                       > torus_intersector.query_element_count(),
            "extent-limited Bezier leaves are the actual BVH query elements");
    for (std::size_t element = 0;
         element < grid_scaled_intersector.query_element_count(); ++element) {
        const auto& samples =
            grid_scaled_intersector.query_element_samples(element);
        require(samples.size() == 16
                    && std::all_of(
                        samples.begin(), samples.end(), [](const auto& sample) {
                            return std::isfinite(sample.u)
                                && std::isfinite(sample.v)
                                && sample.point.allFinite();
                        }),
                "each query leaf owns sixteen finite parameter samples");
    }
    const auto grid_scaled_hit =
        grid_scaled_intersector.intersect_cartesian_edge(
            NurbsCartesianEdgeQuery3D{
                0, 7, 8, 9,
                {0.80, -0.04, 0.03}, {0.84, -0.04, 0.03}});
    const auto& grid_scaled_diagnostics = grid_scaled_hit.diagnostics;
    require(grid_scaled_hit.crossings.size() == 1
                && grid_scaled_diagnostics.maximum_subdivision_depth_reached <= 4
                && grid_scaled_diagnostics.maximum_sample_seeds_per_element <= 4
                && grid_scaled_diagnostics.sample_seeds_accepted > 0
                && grid_scaled_diagnostics.unresolved_candidates == 0,
            "grid-scaled Cartesian query obeys the local depth-four budget");

    auto app_grid_options = grid_scaled_options;
    app_grid_options.maximum_element_extent = 0.375;
    const NurbsSurfaceIntersector3D app_grid_intersector(
        torus.geometry_model(), app_grid_options);
    const auto app_grid_miss =
        app_grid_intersector.intersect_cartesian_edge(
            NurbsCartesianEdgeQuery3D{
                0, 9, 5, 9,
                {0.1875, -0.5625, 0.1875},
                {0.375, -0.5625, 0.1875}});
    const auto& app_grid_diagnostics = app_grid_miss.diagnostics;
    require(app_grid_miss.crossings.empty()
                && app_grid_diagnostics.maximum_subdivision_depth_reached <= 4
                && app_grid_diagnostics.unresolved_candidates == 0
                && app_grid_diagnostics.closest_point_failures == 0
                && app_grid_diagnostics.terminal_misses_by_closest_point > 0
                && app_grid_diagnostics.terminal_certificate_boxes > 0
                && app_grid_diagnostics
                       .maximum_terminal_certificate_depth_reached <= 6,
            "N=16 torus endpoint minimum is a resolved terminal miss");

    auto fine_grid_options = grid_scaled_options;
    fine_grid_options.maximum_element_extent = 0.046875;
    const NurbsSurfaceIntersector3D fine_grid_intersector(
        torus.geometry_model(), fine_grid_options);
    const auto fine_grid_hit =
        fine_grid_intersector.intersect_cartesian_edge(
            NurbsCartesianEdgeQuery3D{
                0, 92, 60, 57,
                {0.65625, -0.09375, -0.1640625},
                {0.6796875, -0.09375, -0.1640625}});
    const auto& fine_grid_diagnostics = fine_grid_hit.diagnostics;
    require(fine_grid_hit.crossings.size() == 1
                && fine_grid_diagnostics.maximum_subdivision_depth_reached <= 4
                && fine_grid_diagnostics.unresolved_candidates == 0
                && fine_grid_diagnostics.terminal_certificate_boxes > 0
                && fine_grid_diagnostics
                       .maximum_terminal_certificate_depth_reached <= 6,
            "N=128 torus terminal certificate resolves one crossing");

    const auto seam_hit = torus_intersector.intersect_segment(
        {0.80, -0.04, 0.03}, {0.84, -0.04, 0.03});
    require(seam_hit.crossings.size() == 1
                && seam_hit.diagnostics.seam_deduplications >= 1,
            "periodic torus seam root is canonicalized once");
    const NurbsCartesianEdgeQuery3D seam_edge{
        0, 1, 2, 3,
        {0.80, -0.04, 0.03}, {0.84, -0.04, 0.03}};
    const auto seam_crossing =
        torus_intersector.intersect_cartesian_edge(seam_edge);
    const auto& seam_edge_diagnostics = seam_crossing.diagnostics;
    require(seam_crossing.crossings.size() == 1
                && seam_crossing.crossings.front().component == 0
                && seam_crossing.crossings.front().residual
                       <= torus_intersector.geometry_tolerance()
                && std::abs(
                    seam_crossing.crossings.front().normal.norm() - 1.0)
                       < 1.0e-12
                && seam_edge_diagnostics.seam_deduplications >= 1,
            "single Cartesian crossing returns canonical root data");
    require(torus_intersector.intersect_cartesian_edge(
                NurbsCartesianEdgeQuery3D{
                    0, 4, 5, 6, {2.0, 2.0, 2.0}, {2.1, 2.0, 2.0}})
                .crossings.empty(),
            "Cartesian edge with zero roots returns an empty result");

    kfbim::geometry3d::NurbsSurfaceIntersectorOptions3D no_seed_options;
    no_seed_options.use_triangle_seeds = false;
    const NurbsSurfaceIntersector3D torus_without_seeds(
        torus.geometry_model(), no_seed_options);
    const auto no_seed_hit = torus_without_seeds.intersect_segment(
        seam_edge.start, seam_edge.end);
    require(no_seed_hit.crossings.size() == 1
                && no_seed_hit.diagnostics.triangle_seed_hits == 0
                && (no_seed_hit.crossings.front().point
                    - seam_hit.crossings.front().point).norm()
                       <= 8.0 * torus_intersector.geometry_tolerance()
                && torus_without_seeds.containing_components(
                       {0.62, -0.04, 0.03}) == std::vector<int>{0},
            "triangle seeds do not change canonical roots or parity");

    const auto torus_model = torus.geometry_model();
    const int patch_offset = torus_model.num_patches();
    const int component_offset = torus_model.num_components();
    auto duplicate_patches = torus_model.patches();
    duplicate_patches.insert(
        duplicate_patches.end(),
        torus_model.patches().begin(), torus_model.patches().end());
    std::vector<int> duplicate_components;
    duplicate_components.reserve(
        static_cast<std::size_t>(2 * patch_offset));
    for (int patch = 0; patch < patch_offset; ++patch)
        duplicate_components.push_back(torus_model.patch_component(patch));
    for (int patch = 0; patch < patch_offset; ++patch) {
        duplicate_components.push_back(
            torus_model.patch_component(patch) + component_offset);
    }
    auto duplicate_connections = torus_model.connections();
    for (auto connection : torus_model.connections()) {
        connection.first.patch += patch_offset;
        connection.second.patch += patch_offset;
        duplicate_connections.push_back(connection);
    }
    const NurbsSurfaceIntersector3D coincident_tori(
        kfbim::geometry3d::NurbsSurfaceModel3D(
            std::move(duplicate_patches),
            std::move(duplicate_components),
            std::move(duplicate_connections)));
    require_throws_contains_all_with_count(
        [&] { (void)coincident_tori.intersect_cartesian_edge(seam_edge); },
        {"coincident roots on unrelated NURBS patches", "axis=0",
         "(1,2,3)"},
        "root=(patch=", 2,
        "unrelated coincident patches report full Cartesian context");
    require(torus_intersector.containing_components(
                {0.62, -0.04, 0.03}) == std::vector<int>{0}
                && torus_intersector.containing_components(
                    {0.07, -0.04, 0.03}).empty(),
            "torus component parity distinguishes tube from central void");
    const auto long_torus_edge = torus_intersector.intersect_cartesian_edge(
        NurbsCartesianEdgeQuery3D{
            0, 10, 11, 12,
            {-1.0, -0.04, 0.03}, {1.0, -0.04, 0.03}});
    require(long_torus_edge.crossings.size() == 4
                && long_torus_edge.toggled_components.empty()
                && long_torus_edge.parity_known_from_roots,
            "torus Cartesian edge retains all four transverse crossings");

    const NativeNurbsSurface3D cylinder =
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    const NurbsSurfaceIntersector3D cylinder_intersector(
        cylinder.geometry_model());
    require_intersector_bounds_dense_samples(cylinder, cylinder_intersector);

    auto cylinder_grid_options = grid_scaled_options;
    cylinder_grid_options.maximum_element_extent = 0.09375;
    const NurbsSurfaceIntersector3D cylinder_grid_intersector(
        cylinder.geometry_model(), cylinder_grid_options);
    const auto cylinder_grid_miss =
        cylinder_grid_intersector.intersect_cartesian_edge(
            NurbsCartesianEdgeQuery3D{
                0, 41, 22, 38,
                {0.421875, -0.46875, 0.28125},
                {0.46875, -0.46875, 0.28125}});
    const auto& cylinder_grid_diagnostics =
        cylinder_grid_miss.diagnostics;
    require(cylinder_grid_miss.crossings.empty()
                && cylinder_grid_diagnostics
                       .maximum_subdivision_depth_reached <= 4
                && cylinder_grid_diagnostics.unresolved_candidates == 0
                && cylinder_grid_diagnostics.closest_point_failures == 0
                && cylinder_grid_diagnostics
                       .terminal_misses_by_closest_point > 0,
            "N=64 cylinder endpoint minimum is a resolved terminal miss");
    const auto grid_tangent =
        cylinder_grid_intersector.intersect_cartesian_edge(
            NurbsCartesianEdgeQuery3D{
                1, 20, 21, 22,
                {0.61, -0.10, 0.0}, {0.61, 0.0, 0.0}});
    require(grid_tangent.crossings.size() == 1
                && grid_tangent.has_near_tangent_candidate
                && !grid_tangent.parity_known_from_roots,
            "extent-limited cylinder leaf preserves a tangent candidate");

    const auto smooth_hit = cylinder_intersector.intersect_segment(
        {0.06, 0.48, 0.0}, {0.06, 0.52, 0.0});
    require(smooth_hit.crossings.size() == 1
                && smooth_hit.diagnostics.seam_deduplications >= 1,
            "smooth cylinder-quarter seam root is canonicalized once");
    require(cylinder_intersector.containing_components(
                {0.46, -0.05, 0.0}) == std::vector<int>{0}
                && cylinder_intersector.containing_components(
                    {0.06, -0.05, 0.0}).empty(),
            "hollow-cylinder parity distinguishes shell from bore");
    const Eigen::Vector3d reference_ray_direction =
        Eigen::Vector3d(1.0, 0.3713906763541037,
                        0.6947465906068658).normalized();
    Eigen::Vector2d planar_direction(
        reference_ray_direction.x(), reference_ray_direction.y());
    planar_direction.normalize();
    const Eigen::Vector3d near_tangent_radius(
        -0.55 * planar_direction.y(),
         0.55 * planar_direction.x(), 0.0);
    const Eigen::Vector3d near_tangent_point =
        Eigen::Vector3d(0.06, -0.05, 0.0) + near_tangent_radius;
    const Eigen::Vector3d near_tangent_ray_start =
        near_tangent_point - 0.2 * reference_ray_direction;
    require_throws_contains(
        [&] {
            (void)cylinder_intersector.intersect_segment(
                near_tangent_ray_start,
                near_tangent_ray_start + reference_ray_direction);
        },
        "unresolved conservative NURBS intersection candidate",
        "non-Cartesian near-secant preserves unresolved status");
    const auto cylinder_tangent =
        cylinder_intersector.intersect_cartesian_edge(
            NurbsCartesianEdgeQuery3D{
                1, 20, 21, 22,
                {0.61, -0.10, 0.0}, {0.61, 0.0, 0.0}});
    require(cylinder_tangent.crossings.size() == 1
                && cylinder_tangent.has_near_tangent_candidate
                && !cylinder_tangent.parity_known_from_roots,
            "cylinder Cartesian edge reports rather than rejects tangency");
    const double outer_endpoint_y =
        -0.05 - std::sqrt(0.55 * 0.55 - 0.25 * 0.25);
    require_throws_contains_all_with_count(
        [&] {
            (void)cylinder_intersector.intersect_cartesian_edge(
                NurbsCartesianEdgeQuery3D{
                    1, 23, 24, 25,
                    {0.31, outer_endpoint_y, 0.0},
                    {0.31, 0.10, 0.0}});
        },
        {"surface intersects Cartesian node", "axis=1", "(23,24,25)"},
        "root=(patch=", 2,
        "endpoint root takes priority over an interior fallback tangent");

    require_throws_contains_all(
        [&] {
            (void)cylinder_intersector.intersect_cartesian_edge(
                NurbsCartesianEdgeQuery3D{
                    0, 30, 31, 32,
                    {0.59, -0.05, 0.67}, {0.63, -0.05, 0.67}});
        },
        {"surface overlaps Cartesian edge", "axis=0", "(30,31,32)",
         "root=(patch="},
        "cylinder top-rim query gives overlap priority over feature contact");
    const auto cylinder_feature_contact =
        cylinder_intersector.intersect_cartesian_edge(
            NurbsCartesianEdgeQuery3D{
                1, 33, 34, 35,
                {0.61, -0.07, 0.67}, {0.61, -0.03, 0.67}});
    require(cylinder_feature_contact.crossings.size() == 1
                && cylinder_feature_contact.crossings.front()
                       .feature_edge_contact
                && cylinder_feature_contact.ambiguous_clusters.size() == 1
                && cylinder_feature_contact.ambiguous_clusters.front()
                       .candidates.empty()
                && !cylinder_feature_contact.root_count_known
                && cylinder_feature_contact.has_near_tangent_candidate
                && cylinder_feature_contact.diagnostics
                       .unresolved_candidates > 0
                && cylinder_feature_contact.diagnostics
                       .non_g1_topology_merges >= 1,
            "bounded-depth cylinder rim contact remains explicitly unresolved");

    const auto transverse_feature_hit =
        cylinder_intersector.intersect_segment(
            {0.63, -0.05, 0.69}, {0.59, -0.05, 0.65});
    require(transverse_feature_hit.crossings.size() == 1
                && transverse_feature_hit.crossings.front()
                       .feature_edge_contact
                && transverse_feature_hit.diagnostics
                       .non_g1_topology_merges >= 1,
            "declared non-G1 topology merges a transverse rim root once");

    const NativeNurbsSurface3D lprism =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const NurbsSurfaceIntersector3D lprism_intersector(
        lprism.geometry_model());
    require_intersector_bounds_dense_samples(lprism, lprism_intersector);
    require(lprism_intersector.containing_components(
                {0.0, -0.3, 0.0}) == std::vector<int>{0},
            "L-prism component parity classifies its reentrant solid");
    require_throws_contains_all(
        [&] {
            (void)lprism_intersector.intersect_cartesian_edge(
                NurbsCartesianEdgeQuery3D{
                    2, 40, 41, 42,
                    {0.0, -0.3, 0.67}, {0.0, -0.3, 0.8}});
        },
        {"surface intersects Cartesian node", "axis=2", "(40,41,42)",
         "root=(patch="},
        "L-prism Cartesian edge rejects a node intersection");
    require_throws_contains_all(
        [&] {
            (void)lprism_intersector.intersect_cartesian_edge(
                NurbsCartesianEdgeQuery3D{
                    0, 50, 51, 52,
                    {-0.2, -0.3, 0.67}, {-0.1, -0.3, 0.67}});
        },
        {"surface overlaps Cartesian edge", "axis=0", "(50,51,52)"},
        "L-prism Cartesian edge rejects a surface overlap");
}

kfbim::geometry3d::NurbsSurfacePatch3D translated_patch(
    const kfbim::geometry3d::NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& offset)
{
    auto controls = patch.control_net();
    for (auto& row : controls) {
        for (Eigen::Vector3d& point : row)
            point += offset;
    }
    return kfbim::geometry3d::NurbsSurfacePatch3D(
        patch.basis_u(), patch.basis_v(), controls, patch.weights());
}

kfbim::geometry3d::NurbsSurfaceModel3D two_translated_components(
    const kfbim::geometry3d::NurbsSurfaceModel3D& source,
    const Eigen::Vector3d& offset)
{
    using kfbim::geometry3d::NurbsPatchEdgeConnection3D;
    using kfbim::geometry3d::NurbsSurfacePatch3D;

    std::vector<NurbsSurfacePatch3D> patches = source.patches();
    std::vector<int> components(
        static_cast<std::size_t>(source.num_patches()), 0);
    std::vector<NurbsPatchEdgeConnection3D> connections =
        source.connections();
    const int patch_offset = source.num_patches();
    for (int patch = 0; patch < source.num_patches(); ++patch) {
        patches.push_back(translated_patch(source.patch(patch), offset));
        components.push_back(1);
    }
    for (auto connection : source.connections()) {
        connection.first.patch += patch_offset;
        connection.second.patch += patch_offset;
        connections.push_back(connection);
    }
    return kfbim::geometry3d::NurbsSurfaceModel3D(
        std::move(patches), std::move(components),
        std::move(connections));
}

void test_direct_nurbs_point_classification()
{
    using kfbim::geometry3d::NurbsSurfaceIntersector3D;
    using kfbim::geometry3d::NurbsSurfaceIntersectorOptions3D;
    struct Case { Eigen::Vector3d point; std::vector<int> expected; const char* name; };
    const auto verify = [](const NativeNurbsSurface3D& surface,
                           const std::vector<Case>& cases) {
        NurbsSurfaceIntersectorOptions3D no_seeds;
        no_seeds.use_triangle_seeds = false;
        const NurbsSurfaceIntersector3D seeded(surface.geometry_model());
        const NurbsSurfaceIntersector3D unseeded(
            surface.geometry_model(), no_seeds);
        for (const auto& item : cases) {
            require(seeded.containing_components(item.point) == item.expected,
                    std::string(item.name) + " with triangle seeds");
            require(unseeded.containing_components(item.point) == item.expected,
                    std::string(item.name) + " without triangle seeds");
        }
    };

    const NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    verify(torus, {
        {{0.62, -0.04, 0.03}, {0}, "torus material"},
        {{0.07, -0.04, 0.03}, {}, "torus hole/two-crossing ray"},
        {{1.20, 0.80, 0.80}, {}, "torus far exterior"}});
    verify(make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder), {
        {{0.46, -0.05, 0.0}, {0}, "cylinder wall"},
        {{0.06, -0.05, 0.0}, {}, "cylinder bore"},
        {{0.70, -0.05, 0.0}, {}, "cylinder radial exterior"},
        {{0.46, -0.05, 0.80}, {}, "cylinder above cap"},
        {{0.46, -0.05, 0.669}, {0}, "cylinder inside cap"},
        {{0.46, -0.05, 0.671}, {}, "cylinder outside cap"}});
    verify(make_native_nurbs_surface_3d(GeometryKind3D::LPrism), {
        {{0.40, -0.30, 0.0}, {0}, "L-prism lower arm"},
        {{-0.20, 0.20, 0.0}, {0}, "L-prism upper arm"},
        {{0.30, 0.20, 0.0}, {}, "L-prism missing quadrant"},
        {{0.0, -0.30, 0.80}, {}, "L-prism outside height"},
        {{0.02, -0.12, 0.0}, {0}, "L-prism corner southwest"},
        {{0.12, -0.12, 0.0}, {0}, "L-prism corner southeast"},
        {{0.02, -0.02, 0.0}, {0}, "L-prism corner northwest"},
        {{0.12, -0.02, 0.0}, {}, "L-prism corner northeast"}});

    const auto translated_model = two_translated_components(
        torus.geometry_model(), Eigen::Vector3d(2.0, 0.0, 0.0));
    for (bool use_seeds : {true, false}) {
        NurbsSurfaceIntersectorOptions3D options;
        options.use_triangle_seeds = use_seeds;
        const NurbsSurfaceIntersector3D translated(translated_model, options);
        require(translated.containing_components({0.62, -0.04, 0.03})
                    == std::vector<int>{0}
                && translated.containing_components({2.62, -0.04, 0.03})
                    == std::vector<int>{1}
                && translated.containing_components({1.07, -0.04, 0.03})
                    .empty(),
                "translated components classify independently");
    }

    const NurbsSurfaceIntersector3D torus_intersector(torus.geometry_model());
    require_throws_contains(
        [&] { (void)torus_intersector.containing_components(
                  {0.82, -0.04, 0.03}); },
        "point lies on NURBS surface",
        "direct classification rejects an exact surface point");
}
void require_triangle_seed_independent_domain(
    const kfbim::CartesianGrid3D& grid,
    const kfbim::geometry3d::NurbsSurfaceModel3D& model,
    int exterior_node,
    int interior_node,
    const std::string& name)
{
    using kfbim::geometry3d::NurbsCartesianDomain3D;
    using kfbim::geometry3d::NurbsCartesianDomainOptions3D;

    const NurbsCartesianDomain3D seeded(grid, model);
    NurbsCartesianDomainOptions3D no_seed_options;
    no_seed_options.use_triangle_seeds = false;
    const NurbsCartesianDomain3D unseeded(
        grid, model, no_seed_options);

    require(seeded.label(exterior_node) == 0,
            name + " central void is exterior");
    require(seeded.label(interior_node) == 1,
            name + " material point is inside");
    require(seeded.labels() == unseeded.labels(),
            name + " labels do not depend on triangle seeds");
    require(seeded.diagnostics().barrier_edge_counts
                == unseeded.diagnostics().barrier_edge_counts,
            name + " barrier counts do not depend on triangle seeds");
    require(seeded.diagnostics().interface_edge_counts
                == unseeded.diagnostics().interface_edge_counts,
            name + " interface counts do not depend on triangle seeds");

    const auto dims = grid.dof_dims();
    const double crossing_tolerance = 8.0 * std::max(
        seeded.geometry_tolerance(), unseeded.geometry_tolerance());
    bool checked_reverse_crossings = false;
    bool checked_missing_crossing = false;
    for (int k = 0; k < dims[2]; ++k) {
        for (int j = 0; j < dims[1]; ++j) {
            for (int i = 0; i < dims[0]; ++i) {
                const int node = grid.index(i, j, k);
                const std::array<int, 3> positive_neighbors{{
                    i + 1 < dims[0] ? grid.index(i + 1, j, k) : -1,
                    j + 1 < dims[1] ? grid.index(i, j + 1, k) : -1,
                    k + 1 < dims[2] ? grid.index(i, j, k + 1) : -1}};
                for (const int neighbor : positive_neighbors) {
                    if (neighbor < 0)
                        continue;
                    require(seeded.has_barrier_between(node, neighbor)
                                == unseeded.has_barrier_between(
                                    node, neighbor),
                            name + " barrier locations do not depend on triangle seeds");
                    const bool seeded_interface =
                        seeded.has_interface_between(node, neighbor);
                    require(seeded_interface
                                == unseeded.has_interface_between(
                                    node, neighbor),
                            name + " interface locations do not depend on triangle seeds");
                    const auto seeded_crossings =
                        seeded.crossings_between(node, neighbor);
                    const auto unseeded_crossings =
                        unseeded.crossings_between(node, neighbor);
                    require(seeded_interface
                                == (seeded_crossings.size() > 0)
                                && seeded_crossings.size()
                                    == unseeded_crossings.size(),
                            name + " crossing ranges do not depend on triangle seeds");
                    for (std::size_t root = 0;
                         root < seeded_crossings.size(); ++root) {
                        require((seeded_crossings[root].point
                                    - unseeded_crossings[root].point).norm()
                                    <= crossing_tolerance,
                                name + " crossing points do not depend on triangle seeds");
                    }
                    if (seeded_crossings.size() == 0) {
                        if (!checked_missing_crossing) {
                            require_throws_contains(
                                [&] {
                                    (void)seeded.crossing_between(
                                        node, neighbor);
                                },
                                "no NURBS crossing between Cartesian nodes",
                                name + " empty range has no single crossing");
                            checked_missing_crossing = true;
                        }
                        continue;
                    }
                    const auto reverse =
                        seeded.crossings_between(neighbor, node);
                    require(reverse.begin() == seeded_crossings.begin()
                                && reverse.size() == seeded_crossings.size(),
                            name + " reverse endpoint lookup returns the same range");
                    checked_reverse_crossings = true;
                    if (seeded_crossings.size() == 1) {
                        require(&seeded.crossing_between(node, neighbor)
                                    == &seeded_crossings[0],
                                name + " legacy lookup returns the sole crossing");
                    } else {
                        require_throws_contains(
                            [&] {
                                (void)seeded.crossing_between(
                                    node, neighbor);
                            },
                            "requires exactly one crossing",
                            name + " legacy lookup rejects a crossing range");
                    }
                }
            }
        }
    }
    require(checked_reverse_crossings,
            name + " has an interface for reverse range lookup");
    require(checked_missing_crossing,
            name + " has an edge with an empty crossing range");
}
void test_nurbs_cartesian_l_prism_labels()
{
    constexpr double h = 3.0 / 128.0;
    const NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const kfbim::CartesianGrid3D grid(
        {-0.5625, -0.703125, -0.6328125},
        {h, h, h}, {54, 54, 58}, kfbim::DofLayout3D::Node);

    const auto domain = std::make_shared<const
        kfbim::geometry3d::NurbsCartesianDomain3D>(
            grid, surface.geometry_model());
    require(domain->label(grid.index(27, 27, 4)) == 1,
            "L-prism reentrant lower-arm node is inside");
    require(domain->label(grid.index(28, 28, 4)) == 0,
            "L-prism missing quadrant is outside");

    int mismatches = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        const auto x = grid.coord(node);
        const Eigen::Vector3d point(x[0], x[1], x[2]);
        const bool exact = surface.exact_inside(point);
        mismatches += (domain->label(node) > 0) != exact ? 1 : 0;
    }
    require(mismatches == 0, "L-prism cropped-grid labels are exact");
    require(domain->diagnostics().barrier_edge_counts[0]
                + domain->diagnostics().barrier_edge_counts[1]
                + domain->diagnostics().barrier_edge_counts[2] > 0,
            "native crossings create Cartesian barriers");
    require(domain->diagnostics().acceleration_leaf_count
                > domain->diagnostics().bezier_element_count
                && domain->diagnostics().maximum_query_element_extent
                       <= 2.0 * h + domain->geometry_tolerance(),
            "Cartesian domain builds its query BVH from physical 2h leaves");
    require(domain->diagnostics().intersections
                    .maximum_subdivision_depth_reached <= 4
                && domain->diagnostics().intersections
                       .unresolved_candidates == 0
                && domain->diagnostics().intersections
                       .closest_point_failures == 0
                && domain->diagnostics().intersections
                       .maximum_terminal_certificate_depth_reached <= 6,
            "Cartesian domain obeys depth four without unresolved closest solves");

    const auto triangulation =
        kfbim::geometry3d::triangulate_nurbs_surface_patches_3d(
            surface.patches, h);
    require(domain->is_compatible_grid(grid),
            "native domain recognizes its Cartesian grid");
    auto require_grid_mismatch = [&](const kfbim::CartesianGrid3D& other,
                                     const std::string& name) {
        require(other.num_dofs() == grid.num_dofs(),
                name + " preserves the grid DOF count");
        require(!domain->is_compatible_grid(other),
                name + " changes the native domain grid signature");
        require_throws_contains(
            [&] {
                (void)kfbim::GridPair3D(
                    other,
                    triangulation.interface,
                    triangulation.geometry_interface,
                    domain);
            },
            "native NURBS domain grid does not match",
            name + " is rejected before native labels are installed");
    };

    auto shifted_origin = grid.origin();
    shifted_origin[0] += 0.125 * h;
    require_grid_mismatch(
        kfbim::CartesianGrid3D(
            shifted_origin,
            grid.spacing(),
            grid.num_cells(),
            kfbim::DofLayout3D::Node),
        "same-size shifted grid");

    auto reshaped_cells = grid.num_cells();
    std::swap(reshaped_cells[1], reshaped_cells[2]);
    require_grid_mismatch(
        kfbim::CartesianGrid3D(
            grid.origin(),
            grid.spacing(),
            reshaped_cells,
            kfbim::DofLayout3D::Node),
        "same-DOF reshaped grid");

    require_grid_mismatch(
        kfbim::CartesianGrid3D(
            grid.origin(),
            grid.spacing(),
            grid.dof_dims(),
            kfbim::DofLayout3D::CellCenter),
        "same-DOF cell-center grid");

    kfbim::GridPair3D pair(grid,
                           triangulation.interface,
                           triangulation.geometry_interface,
                           domain);
    require(pair.has_nurbs_domain(),
            "GridPair reports its native NURBS domain");
    require(&pair.nurbs_domain_diagnostics() == &domain->diagnostics(),
            "GridPair exposes the shared native-domain diagnostics");
    require(pair.domain_label(grid.index(27, 27, 4)) == 1,
            "GridPair uses NURBS barrier labels");

    const auto dims = grid.dof_dims();
    int crossing_edges = 0;
    for (int k = 0; k < dims[2]; ++k) {
        for (int j = 0; j < dims[1]; ++j) {
            for (int i = 0; i < dims[0]; ++i) {
                const int node = grid.index(i, j, k);
                for (int axis = 0; axis < 3; ++axis) {
                    std::array<int, 3> neighbor_index{{i, j, k}};
                    if (++neighbor_index[static_cast<std::size_t>(axis)]
                        >= dims[static_cast<std::size_t>(axis)]) {
                        continue;
                    }
                    const int neighbor = grid.index(
                        neighbor_index[0],
                        neighbor_index[1],
                        neighbor_index[2]);
                    if ((pair.domain_label(node) > 0)
                        == (pair.domain_label(neighbor) > 0)) {
                        continue;
                    }

                    ++crossing_edges;
                    const auto classification =
                        domain->edge_classification_between(node, neighbor);
                    require(classification.queried
                                && classification.has_confirmed_interface
                                && classification.changes_component_membership
                                && classification.root_count_known
                                && classification.parity_known_from_roots
                                && classification.confirmed_crossing_count == 1
                                && classification.confirmed_transverse_count == 1
                                && classification.ambiguous_cluster_count == 0
                                && !classification.has_near_tangent_candidate
                                && classification.correction_safe,
                            "L-prism label-changing edge is correction-safe");
                    const auto& strict_crossing =
                        domain->correction_crossing_between(node, neighbor);
                    require(&strict_crossing
                                == &domain->correction_crossing_between(
                                    neighbor, node),
                            "strict crossing reverse lookup has identity");
                    const kfbim::P2CrossingOwner3D owner =
                        pair.p2_crossing_owner_between(node, neighbor);
                    require(owner.status
                                == kfbim::P2CrossingOwnerStatus3D::
                                       ExactIntersection,
                            "native label-changing edge has an exact owner");
                    require(owner.nurbs_patch_index >= 0
                                && owner.nurbs_patch_index
                                       < static_cast<int>(surface.patches.size()),
                            "native crossing owner has a valid NURBS patch");
                    require(owner.nurbs_parameter.allFinite(),
                            "native crossing owner has finite parameters");
                    require(owner.crossing_point.allFinite()
                                && owner.crossing_normal.allFinite(),
                            "native crossing owner has finite geometry");
                    require(owner.surface_component >= 0,
                            "native crossing owner has a surface component");
                    require(std::isfinite(owner.crossing_residual)
                                && owner.crossing_residual
                                       <= domain->geometry_tolerance(),
                            "native crossing owner residual is within tolerance");

                    const auto& native =
                        domain->correction_crossing_between(node, neighbor);
                    require(owner.nurbs_patch_index == native.patch_index
                                && owner.surface_component == native.component,
                            "native crossing owner preserves patch and component");
                    require((owner.nurbs_parameter
                                - Eigen::Vector2d(native.u, native.v)).norm()
                                <= 1.0e-15
                                && (owner.crossing_point - native.point).norm()
                                       <= 1.0e-15
                                && (owner.crossing_normal - native.normal).norm()
                                       <= 1.0e-15
                                && std::abs(owner.crossing_residual
                                            - native.residual) <= 1.0e-15,
                            "native crossing owner preserves authoritative fields");

                    const auto a_coord = grid.coord(node);
                    const auto b_coord = grid.coord(neighbor);
                    const Eigen::Vector3d a(
                        a_coord[0], a_coord[1], a_coord[2]);
                    const Eigen::Vector3d b(
                        b_coord[0], b_coord[1], b_coord[2]);
                    const Eigen::Vector3d edge = b - a;
                    require(owner.edge_parameter >= 0.0
                                && owner.edge_parameter <= 1.0,
                            "native crossing owner has a directed edge parameter");
                    require((owner.crossing_point
                                - (a + owner.edge_parameter * edge)).norm()
                                <= 8.0 * domain->geometry_tolerance(),
                            "native crossing point lies on the Cartesian edge");
                    const auto surface_point = surface.patches[
                        static_cast<std::size_t>(owner.nurbs_patch_index)]
                        .evaluate(owner.nurbs_parameter.x(),
                                  owner.nurbs_parameter.y());
                    require((owner.crossing_point - surface_point).norm()
                                <= 8.0 * domain->geometry_tolerance(),
                            "native crossing point lies on its NURBS patch");

                    const kfbim::P2CrossingOwner3D reverse =
                        pair.p2_crossing_owner_between(neighbor, node);
                    require((reverse.crossing_point - owner.crossing_point).norm()
                                <= 1.0e-15
                                && std::abs(reverse.edge_parameter
                                            + owner.edge_parameter - 1.0)
                                       <= 8.0 * domain->geometry_tolerance(),
                            "native crossing phase follows the requested direction");
                }
            }
        }
    }
    require(crossing_edges > 0,
            "L-prism native GridPair has label-changing edges");
}

void test_nurbs_cartesian_curved_targets_and_triangle_independence()
{
    const kfbim::CartesianGrid3D grid(
        {-0.7767, -0.8329, -0.7093}, {0.05, 0.05, 0.05},
        {34, 33, 31}, kfbim::DofLayout3D::Node);
    const NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    const NativeNurbsSurface3D cylinder =
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);

    require_triangle_seed_independent_domain(
        grid, torus.geometry_model(),
        grid.index(17, 16, 15), grid.index(28, 16, 15), "torus");
    require_triangle_seed_independent_domain(
        grid, cylinder.geometry_model(),
        grid.index(17, 16, 15), grid.index(25, 16, 15),
        "hollow cylinder");
}

void test_cartesian_edge_collects_multiple_crossings()
{
    using kfbim::geometry3d::NurbsCartesianEdgeQuery3D;
    using kfbim::geometry3d::NurbsSurfaceIntersector3D;
    using kfbim::geometry3d::NurbsSurfaceIntersectorOptions3D;

    const NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    NurbsSurfaceIntersectorOptions3D options;
    options.maximum_element_extent = 0.1;
    options.local_max_subdivision_depth = 4;
    const NurbsSurfaceIntersector3D intersector(
        torus.geometry_model(), options);

    const auto result = intersector.intersect_cartesian_edge(
        NurbsCartesianEdgeQuery3D{
            0, 6, 11, 8,
            {0.0, -0.04, 0.03},
            {1.0, -0.04, 0.03}});
    require(result.crossings.size() == 2,
            "Cartesian edge retains both torus crossings");
    require(result.toggled_components.empty(),
            "two torus crossings have even component parity");
    require(result.parity_known_from_roots,
            "two transverse torus roots determine parity");
}

void test_nurbs_cartesian_under_resolved_torus_uses_parity()
{
    const kfbim::CartesianGrid3D grid(
        {-0.781, -0.841, -0.721}, {0.05, 0.05, 0.05},
        {34, 33, 31}, kfbim::DofLayout3D::Node);
    const NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    const kfbim::geometry3d::NurbsCartesianDomain3D domain(
        grid, torus.geometry_model());

    int label_mismatches = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        const auto x = grid.coord(node);
        label_mismatches +=
            (domain.label(node) > 0)
                    != torus.exact_inside({x[0], x[1], x[2]})
                ? 1 : 0;
    }
    require(label_mismatches == 0,
            "under-resolved torus labels match the analytic solid");

    const auto dims = grid.dof_dims();
    int multiple_crossing_edges = 0;
    int same_component_even_edges = 0;
    for (int k = 0; k < dims[2]; ++k) {
        for (int j = 0; j < dims[1]; ++j) {
            for (int i = 0; i < dims[0]; ++i) {
                const int node = grid.index(i, j, k);
                const std::array<int, 3> neighbors{{
                    i + 1 < dims[0] ? grid.index(i + 1, j, k) : -1,
                    j + 1 < dims[1] ? grid.index(i, j + 1, k) : -1,
                    k + 1 < dims[2] ? grid.index(i, j, k + 1) : -1}};
                for (const int neighbor : neighbors) {
                    if (neighbor < 0)
                        continue;
                    const auto crossings =
                        domain.crossings_between(node, neighbor);
                    if (crossings.size() <= 1)
                        continue;
                    ++multiple_crossing_edges;
                    const int component = crossings[0].component;
                    const bool same_component = std::all_of(
                        crossings.begin(), crossings.end(),
                        [&](const auto& root) {
                            return root.component == component;
                        });
                    if (same_component && crossings.size() % 2 == 0
                        && !domain.has_barrier_between(node, neighbor)) {
                        ++same_component_even_edges;
                    }
                }
            }
        }
    }
    require(multiple_crossing_edges > 0
                && same_component_even_edges > 0
                && domain.diagnostics().multi_crossing_edge_count > 0
                && domain.diagnostics().even_parity_interface_edge_count > 0,
            "under-resolved torus keeps even multi-crossing interface edges");
}
void test_nurbs_cartesian_targeted_retry()
{
    constexpr int N = 128;
    constexpr double h = 3.0 / static_cast<double>(N);
    const kfbim::CartesianGrid3D grid(
        {-1.5, -1.5, -1.5}, {h, h, h}, {N, N, N},
        kfbim::DofLayout3D::Node);
    const NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    const kfbim::geometry3d::NurbsCartesianDomain3D domain(
        grid, torus.geometry_model());
    const auto& diagnostics = domain.diagnostics();
    require(diagnostics.targeted_retry_count > 0
                && diagnostics.ambiguous_label_changing_edge_count
                       == diagnostics.targeted_retry_count
                && diagnostics.targeted_retry_resolved_count
                       == diagnostics.targeted_retry_count
                && diagnostics.targeted_retry_unsafe_count == 0
                && diagnostics.unsafe_label_changing_edge_count == 0
                && diagnostics.intersections
                       .maximum_subdivision_depth_reached <= 4
                && diagnostics.targeted_retry_intersections
                       .maximum_subdivision_depth_reached <= 6,
            "N=128 torus resolves every ambiguous label-changing edge");

    const auto dims = grid.dof_dims();
    std::size_t correction_safe_barriers = 0;
    std::size_t retried_edges = 0;
    for (int k = 0; k < dims[2]; ++k) {
        for (int j = 0; j < dims[1]; ++j) {
            for (int i = 0; i < dims[0]; ++i) {
                const int node = grid.index(i, j, k);
                const std::array<int, 3> neighbors{{
                    i + 1 < dims[0] ? grid.index(i + 1, j, k) : -1,
                    j + 1 < dims[1] ? grid.index(i, j + 1, k) : -1,
                    k + 1 < dims[2] ? grid.index(i, j, k + 1) : -1}};
                for (const int neighbor : neighbors) {
                    if (neighbor < 0)
                        continue;
                    const auto info =
                        domain.edge_classification_between(node, neighbor);
                    if (info.used_targeted_retry) {
                        ++retried_edges;
                        require(info.root_count_known
                                    && info.parity_known_from_roots,
                                "targeted retry resolves root parity");
                    }
                    if (!domain.has_barrier_between(node, neighbor))
                        continue;
                    require(info.queried && info.correction_safe,
                            "every torus barrier is correction-safe");
                    ++correction_safe_barriers;
                }
            }
        }
    }
    require(retried_edges == diagnostics.targeted_retry_count
                && correction_safe_barriers
                       == diagnostics.correction_safe_edge_count,
            "retry and correction-safe diagnostics match edge records");
}
void test_nurbs_cartesian_input_contracts()
{
    const NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    const auto open_model = [] {
        return kfbim::geometry3d::NurbsSurfaceModel3D(
            {kfbim::geometry3d::NurbsSurfacePatch3D::
                 make_unit_square_xy()},
            {0}, {});
    };
    require_throws_contains(
        [&] {
            (void)kfbim::geometry3d::NurbsCartesianDomain3D(
                kfbim::CartesianGrid3D(
                    {-1.0, -1.0, -1.0}, {0.1, 0.1, 0.1},
                    {20, 20, 20}, kfbim::DofLayout3D::CellCenter),
                torus.geometry_model());
        },
        "node-layout Cartesian grid",
        "NURBS Cartesian domain rejects non-node layouts");
    require_throws_contains(
        [&] {
            (void)kfbim::geometry3d::NurbsCartesianDomain3D(
                kfbim::CartesianGrid3D(
                    {-1.0, -1.0, -1.0}, {0.0, 0.1, 0.1},
                    {20, 20, 20}, kfbim::DofLayout3D::Node),
                torus.geometry_model());
        },
        "positive finite Cartesian spacing",
        "NURBS Cartesian domain rejects nonpositive spacing");

    const auto model = torus.geometry_model();
    const auto bounds = model.control_bounds();
    require_throws_contains_all(
        [&] {
            (void)kfbim::geometry3d::NurbsCartesianDomain3D(
                kfbim::CartesianGrid3D(
                    {bounds.lower.x(), bounds.lower.y(), bounds.lower.z()},
                    {0.1, 0.1, 0.1}, {20, 20, 20},
                    kfbim::DofLayout3D::Node),
                model);
        },
        {"NURBS surface must lie strictly inside Cartesian box",
         "surface lower=(", "Cartesian box lower=("},
        "NURBS Cartesian domain reports non-strict box bounds");

    require_throws_type_contains<std::overflow_error>(
        [&] {
            (void)kfbim::geometry3d::NurbsCartesianDomain3D(
                kfbim::CartesianGrid3D(
                    {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                    {std::numeric_limits<int>::max(), 1, 1},
                    kfbim::DofLayout3D::Node),
                open_model());
        },
        "Cartesian cell-to-node dimension overflow",
        "NURBS Cartesian domain rejects cell-to-node integer overflow");
    require_throws_type_contains<std::overflow_error>(
        [&] {
            (void)kfbim::geometry3d::NurbsCartesianDomain3D(
                kfbim::CartesianGrid3D(
                    {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                    {50000, 50000, 1}, kfbim::DofLayout3D::Node),
                open_model());
        },
        "Cartesian node count exceeds int range",
        "NURBS Cartesian domain rejects node-product integer overflow before allocation");
    require_throws_type_contains<std::overflow_error>(
        [&] {
            const double largest = std::numeric_limits<double>::max();
            (void)kfbim::geometry3d::NurbsCartesianDomain3D(
                kfbim::CartesianGrid3D(
                    {largest, 0.0, 0.0}, {largest, 1.0, 1.0},
                    {1, 1, 1}, kfbim::DofLayout3D::Node),
                open_model());
        },
        "Cartesian box bounds must be finite",
        "NURBS Cartesian domain rejects non-finite box upper bounds before geometry queries");
}

void test_nurbs_cartesian_multiple_components()
{
    const NativeNurbsSurface3D source =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const kfbim::CartesianGrid3D grid(
        {-0.7, -0.8, -0.7}, {0.05, 0.05, 0.05},
        {64, 30, 30}, kfbim::DofLayout3D::Node);
    const kfbim::geometry3d::NurbsCartesianDomain3D domain(
        grid, two_translated_components(
                  source.geometry_model(), Eigen::Vector3d(1.6, 0.0, 0.0)));

    require(domain.label(grid.index(14, 10, 14)) == 1,
            "first L-prism component has label one");
    require(domain.label(grid.index(46, 10, 14)) == 2,
            "translated L-prism component has label two");
    require(domain.label(grid.index(30, 10, 14)) == 0,
            "space between L-prism components is exterior");

    const NativeNurbsSurface3D cylinder =
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    const kfbim::CartesianGrid3D coarse_grid(
        {-1.04, -1.05, -1.0}, {1.5, 1.0, 1.0},
        {3, 2, 2}, kfbim::DofLayout3D::Node);
    const auto coarse_model = two_translated_components(
        cylinder.geometry_model(), Eigen::Vector3d(1.5, 0.0, 0.0));
    const kfbim::geometry3d::NurbsCartesianDomain3D coarse_domain(
        coarse_grid, coarse_model);
    const int first_component_node = coarse_grid.index(1, 1, 1);
    const int second_component_node = coarse_grid.index(2, 1, 1);
    const auto component_change_crossings = coarse_domain.crossings_between(
        first_component_node, second_component_node);
    require(coarse_domain.label(first_component_node) == 1
                && coarse_domain.label(second_component_node) == 2
                && component_change_crossings.size() == 4
                && coarse_domain.has_interface_between(
                    first_component_node, second_component_node)
                && coarse_domain.has_barrier_between(
                    first_component_node, second_component_node)
                && coarse_domain.diagnostics()
                       .component_parity_toggle_count >= 2,
            "component membership change is a barrier despite even total parity");
    require_throws_contains(
        [&] {
            (void)coarse_domain.crossing_between(
                first_component_node, second_component_node);
        },
        "requires exactly one crossing",
        "legacy crossing lookup rejects a four-root component-change edge");
    const auto coarse_classification =
        coarse_domain.edge_classification_between(
            first_component_node, second_component_node);
    require(coarse_classification.queried
                && coarse_classification.has_confirmed_interface
                && coarse_classification.changes_component_membership
                && coarse_classification.root_count_known
                && coarse_classification.parity_known_from_roots
                && coarse_classification.confirmed_crossing_count == 4
                && coarse_classification.confirmed_transverse_count == 4
                && !coarse_classification.used_targeted_retry
                && !coarse_classification.correction_safe,
            "multi-crossing component change is not correction-safe");
    require_throws_contains(
        [&] {
            (void)coarse_domain.correction_crossing_between(
                first_component_node, second_component_node);
        },
        "under-resolved NURBS Cartesian edge",
        "multi-crossing correction lookup fails explicitly");
    const auto coarse_triangulation =
        kfbim::geometry3d::triangulate_nurbs_surface_patches_3d(
            coarse_model.patches(), 1.0);
    const auto coarse_domain_ptr = std::make_shared<const
        kfbim::geometry3d::NurbsCartesianDomain3D>(coarse_domain);
    require_throws_contains(
        [&] {
            (void)kfbim::GridPair3D(
                coarse_grid,
                coarse_triangulation.interface,
                coarse_triangulation.geometry_interface,
                coarse_domain_ptr);
        },
        "under-resolved NURBS Cartesian edge",
        "GridPair rejects an unsafe native correction edge");
}

void test_bezier_rejects_extreme_degree_control_count()
{
    kfbim::geometry3d::RationalBezierElement3D element;
    element.degree_u = std::numeric_limits<int>::max() / 2 + 1;
    element.degree_v = std::numeric_limits<int>::max() / 2 + 1;
    element.parameter_u0 = 0.0;
    element.parameter_u1 = 1.0;
    element.parameter_v0 = 0.0;
    element.parameter_v1 = 1.0;
    require_throws_contains(
        [&] { (void)element.bounds(); },
        "Bezier element control count is too large",
        "extreme Bezier degrees are rejected before count overflow");

    element.degree_u = std::numeric_limits<int>::max();
    element.degree_v = 0;
    require_throws_contains(
        [&] { (void)element.bounds(); },
        "Bezier element degree is too large for inclusive indexing",
        "maximum integer Bezier degree is rejected before adding one");
}

double dof_area(const SurfaceDofCloud3D& cloud)
{
    double result = 0.0;
    for (const auto& dof : cloud.dofs)
        result += dof.weight;
    return result;
}

double check_dof_cloud(const NativeNurbsSurface3D& surface, double h)
{
    const SurfaceDofCloud3D cloud = make_native_surface_dofs_3d(surface, h);
    require(cloud.patches.size() == surface.patches.size(),
            surface.name + " one DOF patch per NURBS patch");
    require(!cloud.dofs.empty(), surface.name + " nonempty DOF cloud");
    for (const auto& tensor_patch : cloud.patches) {
        require(tensor_patch.nu >= 2 && tensor_patch.nv >= 2,
                surface.name + " at least two cells per parameter direction");
        require(tensor_patch.dof_count() == tensor_patch.nu * tensor_patch.nv,
                surface.name + " dense tensor indexing");
    }
    for (int patch_id = 0;
         patch_id < static_cast<int>(surface.patches.size());
         ++patch_id) {
        for (int edge_index = 0; edge_index < 4; ++edge_index) {
            const auto& connection =
                surface.smooth_neighbors[static_cast<std::size_t>(patch_id)]
                                        [static_cast<std::size_t>(edge_index)];
            if (!connection)
                continue;
            const PatchEdge3D edge = static_cast<PatchEdge3D>(edge_index);
            const auto& source = cloud.patches[static_cast<std::size_t>(patch_id)];
            const auto& destination =
                cloud.patches[static_cast<std::size_t>(connection->patch)];
            const int source_along =
                edge == PatchEdge3D::UMin || edge == PatchEdge3D::UMax
                    ? source.nv : source.nu;
            const int destination_along =
                connection->edge == PatchEdge3D::UMin
                        || connection->edge == PatchEdge3D::UMax
                    ? destination.nv : destination.nu;
            require(source_along == destination_along,
                    surface.name + " G1 seam has matching tensor rows");
        }
        const auto& tensor =
            cloud.patches[static_cast<std::size_t>(patch_id)];
        const int center = tensor.dof_index(tensor.nu / 2, tensor.nv / 2);
        require(nearest_topological_cauchy_dofs(
                    surface, cloud, center, 48).size() == 48,
                surface.name + " topological neighborhood supports 48 values");
    }
    for (const auto& dof : cloud.dofs) {
        require(dof.patch_id >= 0
                    && dof.patch_id < static_cast<int>(surface.patches.size()),
                surface.name + " valid native patch ownership");
        const auto& patch = surface.patches[static_cast<std::size_t>(dof.patch_id)];
        require(dof.u > patch.domain_start_u()
                    && dof.u < patch.domain_end_u(),
                surface.name + " strict interior u center");
        require(dof.v > patch.domain_start_v()
                    && dof.v < patch.domain_end_v(),
                surface.name + " strict interior v center");
        require((dof.point - patch.evaluate(dof.u, dof.v)).norm() < 1.0e-13,
                surface.name + " native point evaluation");
        require(std::abs(dof.normal.norm() - 1.0) < 1.0e-13,
                surface.name + " unit normal");
        require(std::abs(dof.tangent1.norm() - 1.0) < 1.0e-13
                    && std::abs(dof.tangent2.norm() - 1.0) < 1.0e-13
                    && std::abs(dof.tangent1.dot(dof.tangent2)) < 1.0e-13
                    && std::abs(dof.normal.dot(dof.tangent1)) < 1.0e-13
                    && std::abs(dof.normal.dot(dof.tangent2)) < 1.0e-13,
                surface.name + " orthonormal local frame");
        require(dof.weight > 0.0 && std::isfinite(dof.weight),
                surface.name + " positive finite midpoint weight");
        const double offset = 0.1 * h;
        require(surface.exact_inside(dof.point - offset * dof.normal),
                surface.name + " inward normal enters domain");
        require(!surface.exact_inside(dof.point + offset * dof.normal),
                surface.name + " outward normal exits domain");
    }
    return std::abs(dof_area(cloud) - surface.expected_area)
         / surface.expected_area;
}

void test_uniform_native_dofs()
{
    for (GeometryKind3D kind : {GeometryKind3D::Torus,
                                GeometryKind3D::HollowCylinder,
                                GeometryKind3D::LPrism}) {
        const NativeNurbsSurface3D surface = make_native_nurbs_surface_3d(kind);
        const double coarse_error = check_dof_cloud(surface, 3.0 / 16.0);
        const double fine_error = check_dof_cloud(surface, 3.0 / 32.0);
        require(fine_error < 3.0e-2,
                surface.name + " midpoint area accuracy at N=32");
        require((coarse_error < 1.0e-12 && fine_error < 1.0e-12)
                    || fine_error < coarse_error,
                surface.name + " midpoint area improves under refinement");
    }
}

bool four_unique_valid_ids(const std::array<int, 4>& ids,
                           const SurfaceDofCloud3D& cloud)
{
    std::set<int> unique;
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(cloud.dofs.size()))
            return false;
        unique.insert(id);
    }
    return unique.size() == 4;
}

bool contains_patch(const std::array<int, 4>& ids,
                    const SurfaceDofCloud3D& cloud,
                    int patch_id)
{
    for (int id : ids) {
        if (cloud.dofs[static_cast<std::size_t>(id)].patch_id == patch_id)
            return true;
    }
    return false;
}

bool all_on_patch(const std::array<int, 4>& ids,
                  const SurfaceDofCloud3D& cloud,
                  int patch_id)
{
    for (int id : ids) {
        if (cloud.dofs[static_cast<std::size_t>(id)].patch_id != patch_id)
            return false;
    }
    return true;
}

bool contains_non_source_patch(const std::vector<int>& ids,
                               const SurfaceDofCloud3D& cloud,
                               int source_patch)
{
    for (int id : ids) {
        if (cloud.dofs[static_cast<std::size_t>(id)].patch_id != source_patch)
            return true;
    }
    return false;
}

bool no_patch_in_range(const std::vector<int>& ids,
                       const SurfaceDofCloud3D& cloud,
                       int first,
                       int end)
{
    for (int id : ids) {
        const int patch = cloud.dofs[static_cast<std::size_t>(id)].patch_id;
        if (patch >= first && patch < end)
            return false;
    }
    return true;
}

void test_parameter_candidates()
{
    const NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    const SurfaceDofCloud3D torus_cloud =
        make_native_surface_dofs_3d(torus, 3.0 / 32.0);
    const auto interior = parameter_dof_candidates_2x2(
        torus, torus_cloud, 0, 0.5, 0.5);
    require(four_unique_valid_ids(interior, torus_cloud),
            "interior query must have four valid DOFs");
    require(all_on_patch(interior, torus_cloud, 0),
            "interior query remains on its native patch");

    const auto& torus_tensor = torus_cloud.patches[0];
    const double torus_du = 1.0 / static_cast<double>(torus_tensor.nu);
    const auto periodic = parameter_dof_candidates_2x2(
        torus, torus_cloud, 0, 0.1 * torus_du, 0.5);
    require(four_unique_valid_ids(periodic, torus_cloud),
            "periodic query must have four valid DOFs");
    require(contains_patch(periodic, torus_cloud, 12),
            "torus closed seam reaches previous u quarter");
    const int torus_cauchy_center = torus_tensor.dof_index(
        0, torus_tensor.nv / 2);
    const std::vector<int> torus_cauchy =
        nearest_topological_cauchy_dofs(
            torus, torus_cloud, torus_cauchy_center, 48);
    require(contains_non_source_patch(torus_cauchy, torus_cloud, 0),
            "fine-grid Cauchy selector includes a topological seam neighbor");

    const NativeNurbsSurface3D lprism =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const SurfaceDofCloud3D lcloud =
        make_native_surface_dofs_3d(lprism, 3.0 / 32.0);
    const auto& top_left = lcloud.patches[3];
    const double ldu = 1.0 / static_cast<double>(top_left.nu);
    const auto sharp = parameter_dof_candidates_2x2(
        lprism, lcloud, 3, 0.1 * ldu, 0.5);
    require(four_unique_valid_ids(sharp, lcloud),
            "sharp-edge query must retain four candidates");
    require(all_on_patch(sharp, lcloud, 3),
            "non-G1 edge remains on the source patch");

    const auto smooth = parameter_dof_candidates_2x2(
        lprism, lcloud, 3, 1.0 - 0.1 * ldu, 0.5);
    require(four_unique_valid_ids(smooth, lcloud),
            "smooth L-cap seam must retain four candidates");
    require(contains_patch(smooth, lcloud, 4),
            "smooth L-cap seam reaches its neighboring patch");

    NativeNurbsSurface3D reversed;
    reversed.name = "reversed_test";
    reversed.description = "two coplanar patches with reversed edge parameter";
    reversed.patches.push_back(
        kfbim::geometry3d::NurbsSurfacePatch3D::make_bilinear_plane(
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}));
    reversed.patches.push_back(
        kfbim::geometry3d::NurbsSurfacePatch3D::make_bilinear_plane(
            {2.0, 1.0, 0.0}, {1.0, 1.0, 0.0},
            {2.0, 0.0, 0.0}, {1.0, 0.0, 0.0}));
    reversed.patch_names = {"left", "right_reversed"};
    reversed.smooth_neighbors.resize(2);
    reversed.smooth_neighbors[0][static_cast<int>(PatchEdge3D::UMax)] =
        SmoothPatchNeighbor3D{1, PatchEdge3D::UMax, true};
    reversed.smooth_neighbors[1][static_cast<int>(PatchEdge3D::UMax)] =
        SmoothPatchNeighbor3D{0, PatchEdge3D::UMax, true};
    reversed.topological_patch_neighbors = {{1}, {0}};
    reversed.expected_area = 2.0;
    reversed.exact_inside = [](const Eigen::Vector3d&) { return false; };
    const SurfaceDofCloud3D reversed_cloud =
        make_native_surface_dofs_3d(reversed, 0.2);
    const auto reversed_ids = parameter_dof_candidates_2x2(
        reversed, reversed_cloud, 0, 0.99, 0.2);
    require(contains_patch(reversed_ids, reversed_cloud, 1),
            "reversed smooth seam reaches its neighbor");
    bool found_high_v = false;
    for (int id : reversed_ids) {
        const auto& dof = reversed_cloud.dofs[static_cast<std::size_t>(id)];
        if (dof.patch_id == 1 && dof.v > 0.5)
            found_high_v = true;
    }
    require(found_high_v,
            "reversed smooth seam reverses the along-edge parameter");

    kfbim::geometry3d::NurbsParamTriangle3D triangle;
    triangle.patch_index = 7;
    triangle.uv = {{{0.0, 0.0}, {1.0, 0.0}, {0.0, 2.0}}};
    const Eigen::Vector2d uv = interpolate_triangle_parameter(
        triangle, Eigen::Vector3d(0.2, 0.3, 0.5));
    require((uv - Eigen::Vector2d(0.3, 1.0)).norm() < 1.0e-14,
            "triangle barycentric UV interpolation");

    require(smooth_patch_component(torus, 0).size() == 16,
            "torus smooth component contains every native patch");

    const NativeNurbsSurface3D cylinder =
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    for (int first : {0, 4, 8, 12}) {
        const std::vector<int> component =
            smooth_patch_component(cylinder, first);
        require(component.size() == 4,
                "each cylinder sheet has four smooth quarter patches");
        for (int patch : component)
            require(patch >= first && patch < first + 4,
                    "cylinder smooth component cannot cross a sharp circle");
    }
    require(smooth_patch_component(lprism, 0).size() == 3,
            "L-prism bottom contains three smooth native patches");
    require(smooth_patch_component(lprism, 3).size() == 3,
            "L-prism top contains three smooth native patches");
    for (int side = 6; side < 12; ++side) {
        require(smooth_patch_component(lprism, side).size() == 1,
                "each L-prism side is a separate non-G1 component");
    }
    const auto& long_side_neighbors = lprism.topological_patch_neighbors[6];
    for (int expected : {0, 1, 3, 4, 7, 11}) {
        require(std::find(long_side_neighbors.begin(), long_side_neighbors.end(),
                          expected) != long_side_neighbors.end(),
                "long L side retains one-to-many cap topology");
    }

    const SurfaceDofCloud3D coarse_lcloud =
        make_native_surface_dofs_3d(lprism, 3.0 / 16.0);
    const auto& lside = coarse_lcloud.patches[7];
    const int lcenter = lside.dof_index(lside.nu / 2, lside.nv / 2);
    const std::vector<int> l_cauchy = nearest_topological_cauchy_dofs(
        lprism, coarse_lcloud, lcenter, 48);
    require(l_cauchy.size() == 48,
            "L side Cauchy selector retains 48 values");
    require(contains_non_source_patch(l_cauchy, coarse_lcloud, 7),
            "L side Cauchy selector crosses a non-G1 topological edge");
    const std::vector<int> same_patch = nearest_same_patch_cauchy_dofs(
        coarse_lcloud, lcenter, 48);
    require(same_patch.size() == static_cast<std::size_t>(lside.dof_count()),
            "same-patch Cauchy selector uses every available coarse DOF");
    require(std::find(same_patch.begin(), same_patch.end(), lcenter)
                != same_patch.end(),
            "same-patch Cauchy selector retains its center");
    for (int q : same_patch) {
        require(coarse_lcloud.dofs[static_cast<std::size_t>(q)].patch_id == 7,
                "same-patch Cauchy selector cannot cross a patch edge");
    }

    const SurfaceDofCloud3D fine_lcloud =
        make_native_surface_dofs_3d(lprism, 3.0 / 32.0);
    const auto& fine_lside = fine_lcloud.patches[7];
    const int fine_lcenter = fine_lside.dof_index(0, fine_lside.nv / 2);
    const std::vector<int> fine_l_cauchy =
        nearest_topological_cauchy_dofs(
            lprism, fine_lcloud, fine_lcenter, 48);
    require(contains_non_source_patch(fine_l_cauchy, fine_lcloud, 7),
            "fine-grid L Cauchy selector crosses a non-G1 neighbor");
    const std::vector<int> balanced = balanced_topological_cauchy_dofs(
        lprism, fine_lcloud, fine_lcenter, 48);
    require(balanced.size() == 48,
            "balanced Cauchy selector retains the requested count");
    require(std::find(balanced.begin(), balanced.end(), fine_lcenter)
                != balanced.end(),
            "balanced Cauchy selector retains its center");
    std::set<int> control_patches;
    for (int q : fine_l_cauchy) {
        control_patches.insert(
            fine_lcloud.dofs[static_cast<std::size_t>(q)].patch_id);
    }
    require(control_patches.size() > 1,
            "fine-grid control stencil must activate multiple patches");
    std::map<int, int> balanced_counts;
    for (int q : balanced) {
        const int patch =
            fine_lcloud.dofs[static_cast<std::size_t>(q)].patch_id;
        require(control_patches.count(patch) == 1,
                "balanced Cauchy selector cannot activate a distant patch");
        ++balanced_counts[patch];
    }
    require(balanced_counts.size() == control_patches.size(),
            "balanced Cauchy selector retains every active patch");
    int balanced_min = 48;
    int balanced_max = 0;
    for (const auto& item : balanced_counts) {
        balanced_min = std::min(balanced_min, item.second);
        balanced_max = std::max(balanced_max, item.second);
    }
    require(balanced_max - balanced_min <= 1,
            "balanced Cauchy patch counts differ by at most one");

    const SurfaceDofCloud3D coarse_cylinder_cloud =
        make_native_surface_dofs_3d(cylinder, 3.0 / 16.0);
    const auto& outer_wall = coarse_cylinder_cloud.patches[0];
    const int cylinder_center = outer_wall.dof_index(
        outer_wall.nu / 2, outer_wall.nv / 2);
    const std::vector<int> cylinder_cauchy =
        nearest_topological_cauchy_dofs(
            cylinder, coarse_cylinder_cloud, cylinder_center, 48);
    require(cylinder_cauchy.size() == 48,
            "cylinder Cauchy selector retains 48 values");
    require(no_patch_in_range(
                cylinder_cauchy, coarse_cylinder_cloud, 4, 8),
            "outer-wall Cauchy selector does not jump to the inner wall");
}

void test_grid_edge_triangle_owners()
{
    constexpr int N = 16;
    constexpr double box_min = -1.5;
    const double h = 3.0 / static_cast<double>(N);
    const NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    kfbim::geometry3d::NurbsPatchTriangulatorOptions3D options;
    options.H_factor = 2.0;
    options.max_depth = 6;
    const auto triangulation =
        kfbim::geometry3d::triangulate_nurbs_surface_patches_3d(
            torus.patches, h, options);
    require(triangulation.geometry_triangles.size()
                == static_cast<std::size_t>(
                    triangulation.geometry_interface.num_panels()),
            "geometry triangle metadata must align with panels");
    kfbim::CartesianGrid3D grid(
        {box_min, box_min, box_min}, {h, h, h}, {N, N, N},
        kfbim::DofLayout3D::Node);
    kfbim::GridPair3D pair(
        grid, triangulation.interface, triangulation.geometry_interface);
    require(!pair.has_nurbs_domain(),
            "legacy GridPair has no native NURBS domain");
    require_throws_contains(
        [&] { (void)pair.nurbs_domain_diagnostics(); },
        "no native NURBS domain",
        "legacy GridPair rejects native diagnostics access");
    const kfbim::LaplaceCorrectionSupport3D support =
        kfbim::build_laplace_correction_support_3d(
            pair, "native NURBS crossing-owner test");
    std::set<std::pair<int, int>> crossing_edges;
    for (const auto& operation : support.crossing_ops) {
        crossing_edges.emplace(
            std::min(operation.rhs_node, operation.correction_node),
            std::max(operation.rhs_node, operation.correction_node));
    }
    require(!crossing_edges.empty(), "torus must cross Cartesian edges");
    for (const auto& edge : crossing_edges) {
        const kfbim::P2CrossingOwner3D owner =
            pair.p2_crossing_owner_between(edge.first, edge.second);
        require(owner.status
                    != kfbim::P2CrossingOwnerStatus3D::EndpointNearestCenter,
                "closed NURBS geometry must provide a triangle owner");
        require(owner.geometry_panel_index >= 0
                    && owner.geometry_panel_index
                           < triangulation.geometry_interface.num_panels(),
                "valid geometry panel index");
        require(std::abs(owner.geometry_barycentric.sum() - 1.0) < 1.0e-12,
                "triangle barycentric partition");
        require(owner.geometry_barycentric.minCoeff() >= -1.0e-10,
                "triangle barycentric lower bound");
        require(owner.edge_parameter >= 0.0 && owner.edge_parameter <= 1.0,
                "Cartesian-edge phase");
    }
}

} // namespace

int main()
{
    try {
        test_native_models();
        test_native_surface_interval_topology();
        test_interval_topology_rejects_out_of_domain_endpoint();
        test_interval_topology_rejects_tiny_gap();
        test_interval_topology_rejects_tiny_overlap();
        test_rational_bezier_extraction_and_subdivision();
        test_nurbs_basis_constructor_multiplicity_contract();
        test_benchmark_rational_bezier_elements_are_conservative();
        test_nurbs_bezier_segment_closest_point();
        test_closest_point_classifies_terminal_intersection_boxes();
        test_rational_bezier_element_intersection();
        test_bezier_element_intersection_isolates_all_roots_and_fails_safe();
        test_analytic_ruled_graph_root_cases();
        test_close_roots_have_stationary_witness();
        test_split_boundary_root_is_not_ambiguous();
        test_ruled_overlap_certificate_is_physical_and_fail_safe();
        test_exact_predicate_triangle_seed_on_thin_element();
        test_reweighted_ruled_overlap_certificate();
        test_nonclamped_rational_bezier_extraction();
        test_bezier_subdivision_rejects_collapsed_midpoint();
        test_bezier_split_preserves_large_finite_controls();
        test_bezier_rejects_extreme_degree_control_count();
        test_single_patch_periodic_seam_canonicalization();
        test_tangent_witness_does_not_swallow_unresolved_candidate();
        test_native_nurbs_surface_intersector();
        test_direct_nurbs_point_classification();
        test_nurbs_cartesian_l_prism_labels();
        test_nurbs_cartesian_under_resolved_torus_uses_parity();
        test_nurbs_cartesian_targeted_retry();
        test_nurbs_cartesian_input_contracts();
        test_nurbs_cartesian_multiple_components();
        test_nurbs_cartesian_curved_targets_and_triangle_independence();
        test_cartesian_edge_collects_multiple_crossings();
        test_uniform_native_dofs();
        test_parameter_candidates();
        test_grid_edge_triangle_owners();
        std::cout << "native NURBS model tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native NURBS model test failure: " << error.what() << '\n';
        return 1;
    }
}
