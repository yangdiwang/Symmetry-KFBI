#include "native_nurbs_surface_3d.hpp"

#include "src/geometry/grid_pair_3d.hpp"
#include "src/grid/cartesian_grid_3d.hpp"
#include "src/transfer/laplace_correction_support.hpp"

#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace {

using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::NativeNurbsSurface3D;
using kfbim::app3d::PatchEdge3D;
using kfbim::app3d::SmoothPatchNeighbor3D;
using kfbim::app3d::SurfaceDofCloud3D;
using kfbim::app3d::interpolate_triangle_parameter;
using kfbim::app3d::make_native_nurbs_surface_3d;
using kfbim::app3d::make_native_surface_dofs_3d;
using kfbim::app3d::parameter_dof_candidates_2x2;
using kfbim::app3d::smooth_patch_component;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void check_patch_regular(const NativeNurbsSurface3D& surface)
{
    require(surface.patch_names.size() == surface.patches.size(),
            surface.name + " patch-name count");
    require(surface.smooth_neighbors.size() == surface.patches.size(),
            surface.name + " smooth-neighbor count");
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
        const std::vector<int> component =
            smooth_patch_component(surface, patch_id);
        int component_dofs = 0;
        for (int component_patch : component)
            component_dofs += cloud.patches[
                static_cast<std::size_t>(component_patch)].dof_count();
        require(component_dofs >= 48,
                surface.name + " G1 component supports 48 Cauchy values");
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
