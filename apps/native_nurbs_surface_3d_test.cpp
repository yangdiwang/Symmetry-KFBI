#include "native_nurbs_surface_3d.hpp"

#include "src/geometry/grid_pair_3d.hpp"
#include "src/geometry/rational_bezier_surface_3d.hpp"
#include "src/geometry/nurbs_surface_model_3d.hpp"
#include "src/grid/cartesian_grid_3d.hpp"
#include "src/transfer/laplace_correction_support.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
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

void test_degree_zero_multispan_bezier_extraction()
{
    using kfbim::geometry::NurbsBasis1D;
    using kfbim::geometry3d::NurbsSurfacePatch3D;
    const NurbsSurfacePatch3D patch(
        NurbsBasis1D(0, {0.0, 0.5, 1.0}),
        NurbsBasis1D(0, {0.0, 1.0}),
        {{{0.0, 0.0, 0.0}},
         {{2.0, 0.0, 0.0}}},
        {{1.0}, {1.0}});
    const kfbim::geometry3d::NurbsSurfaceModel3D model({patch}, {0}, {});
    const auto elements =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(model);
    require(elements.size() == 2,
            "degree-zero U spans produce two Bezier elements");
    require((elements[0].evaluate(0.25, 0.5) - patch.evaluate(0.25, 0.5)).norm()
                < 2.0e-12,
            "first degree-zero Bezier span preserves NURBS evaluation");
    require((elements[1].evaluate(0.75, 0.5) - patch.evaluate(0.75, 0.5)).norm()
                < 2.0e-12,
            "second degree-zero Bezier span preserves NURBS evaluation");
}

void test_discontinuous_multispan_bezier_extraction()
{
    using kfbim::geometry::NurbsBasis1D;
    using kfbim::geometry3d::NurbsSurfacePatch3D;
    const NurbsSurfacePatch3D patch(
        NurbsBasis1D(1, {0.0, 0.0, 0.5, 0.5, 1.0, 1.0}),
        NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0}),
        {{{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
         {{1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}},
         {{3.0, 0.0, 0.0}, {3.0, 1.0, 0.0}},
         {{4.0, 0.0, 0.0}, {4.0, 1.0, 0.0}}},
        {{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}});
    const kfbim::geometry3d::NurbsSurfaceModel3D model({patch}, {0}, {});
    const auto elements =
        kfbim::geometry3d::extract_rational_bezier_elements_3d(model);
    require(elements.size() == 2,
            "fully discontinuous U break produces two Bezier elements");
    for (const auto& element : elements) {
        const double u = 0.5 * (element.u0() + element.u1());
        for (int iv = 0; iv <= 4; ++iv) {
            const double v = element.v0()
                + (element.v1() - element.v0()) * iv / 4.0;
            const Eigen::Vector3d exact = patch.evaluate(u, v);
            require((element.evaluate(u, v) - exact).norm() < 2.0e-12,
                    "discontinuous Bezier span preserves NURBS evaluation");
            require(element.bounds().contains(exact, 2.0e-12),
                    "discontinuous Bezier span has conservative bounds");
        }
    }
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

void test_bezier_rejects_extreme_degree_control_count()
{
    kfbim::geometry3d::RationalBezierElement3D element;
    element.degree_u = std::numeric_limits<int>::max();
    element.degree_v = std::numeric_limits<int>::max();
    element.parameter_u0 = 0.0;
    element.parameter_u1 = 1.0;
    element.parameter_v0 = 0.0;
    element.parameter_v1 = 1.0;
    require_throws_contains(
        [&] { (void)element.bounds(); },
        "Bezier element control count is too large",
        "extreme Bezier degrees are rejected before count overflow");
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
        test_benchmark_rational_bezier_elements_are_conservative();
        test_nonclamped_rational_bezier_extraction();
        test_degree_zero_multispan_bezier_extraction();
        test_discontinuous_multispan_bezier_extraction();
        test_bezier_subdivision_rejects_collapsed_midpoint();
        test_bezier_split_preserves_large_finite_controls();
        test_bezier_rejects_extreme_degree_control_count();
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
