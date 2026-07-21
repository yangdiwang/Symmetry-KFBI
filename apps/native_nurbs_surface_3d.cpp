#include "native_nurbs_surface_3d.hpp"

#include "src/geometry/nurbs_basis.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

using geometry::NurbsBasis1D;
using geometry3d::NurbsPatchEdgeConnection3D;
using geometry3d::NurbsSurfacePatch3D;
using geometry3d::full_patch_edge_interval;

struct RationalQuarterArc2D {
    std::array<Eigen::Vector2d, 3> controls;
    std::array<double, 3> weights;
};

int edge_index(PatchEdge3D edge)
{
    return static_cast<int>(edge);
}

RationalQuarterArc2D quarter_arc(double angle0)
{
    const double angle1 = angle0 + 0.5 * kPi;
    const double angle_mid = 0.5 * (angle0 + angle1);
    const double middle_weight = std::sqrt(0.5);
    return {{{{std::cos(angle0), std::sin(angle0)},
              {std::cos(angle_mid) / middle_weight,
               std::sin(angle_mid) / middle_weight},
              {std::cos(angle1), std::sin(angle1)}}},
            {{1.0, middle_weight, 1.0}}};
}

NurbsBasis1D quadratic_unit_basis()
{
    return NurbsBasis1D(2, {0.0, 0.0, 0.0, 1.0, 1.0, 1.0});
}

NurbsBasis1D linear_unit_basis()
{
    return NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0});
}

void append_patch(NativeNurbsSurface3D& surface,
                  std::string name,
                  NurbsSurfacePatch3D patch)
{
    surface.patch_names.push_back(std::move(name));
    surface.patches.push_back(std::move(patch));
    surface.smooth_neighbors.emplace_back();
    surface.topological_patch_neighbors.emplace_back();
    surface.patch_components.push_back(0);
}

void connect_topological_patches(NativeNurbsSurface3D& surface,
                                 int patch_a,
                                 int patch_b)
{
    if (patch_a < 0 || patch_a >= static_cast<int>(surface.patches.size())
        || patch_b < 0 || patch_b >= static_cast<int>(surface.patches.size())
        || patch_a == patch_b) {
        throw std::out_of_range(
            "topological NURBS connection has invalid patch");
    }
    auto insert_unique = [](std::vector<int>& neighbors, int patch) {
        if (std::find(neighbors.begin(), neighbors.end(), patch)
            == neighbors.end()) {
            neighbors.push_back(patch);
            std::sort(neighbors.begin(), neighbors.end());
        }
    };
    insert_unique(
        surface.topological_patch_neighbors[static_cast<std::size_t>(patch_a)],
        patch_b);
    insert_unique(
        surface.topological_patch_neighbors[static_cast<std::size_t>(patch_b)],
        patch_a);
}

void connect_smooth(NativeNurbsSurface3D& surface,
                    int patch_a,
                    PatchEdge3D edge_a,
                    int patch_b,
                    PatchEdge3D edge_b,
                    bool reversed)
{
    if (patch_a < 0 || patch_a >= static_cast<int>(surface.patches.size())
        || patch_b < 0 || patch_b >= static_cast<int>(surface.patches.size())) {
        throw std::out_of_range("smooth NURBS connection has invalid patch");
    }
    connect_topological_patches(surface, patch_a, patch_b);
    auto& slot_a = surface.smooth_neighbors[static_cast<std::size_t>(patch_a)]
                                           [static_cast<std::size_t>(edge_index(edge_a))];
    auto& slot_b = surface.smooth_neighbors[static_cast<std::size_t>(patch_b)]
                                           [static_cast<std::size_t>(edge_index(edge_b))];
    if (slot_a || slot_b)
        throw std::runtime_error("NURBS patch edge has multiple smooth neighbors");
    slot_a = SmoothPatchNeighbor3D{patch_b, edge_b, reversed};
    slot_b = SmoothPatchNeighbor3D{patch_a, edge_a, reversed};
    surface.geometric_connections.push_back({
        full_patch_edge_interval(
            surface.patches[static_cast<std::size_t>(patch_a)], patch_a, edge_a),
        full_patch_edge_interval(
            surface.patches[static_cast<std::size_t>(patch_b)], patch_b, edge_b),
        reversed,
        true});
}

void connect_feature(NativeNurbsSurface3D& surface,
                     int patch_a,
                     PatchEdge3D edge_a,
                     double begin_a,
                     double end_a,
                     int patch_b,
                     PatchEdge3D edge_b,
                     double begin_b,
                     double end_b,
                     bool reversed)
{
    connect_topological_patches(surface, patch_a, patch_b);
    surface.geometric_connections.push_back({
        {patch_a, edge_a, begin_a, end_a},
        {patch_b, edge_b, begin_b, end_b},
        reversed,
        false});
}

NurbsSurfacePatch3D make_torus_quarter_patch(int u_quarter,
                                             int v_quarter)
{
    constexpr double cx = 0.07;
    constexpr double cy = -0.04;
    constexpr double cz = 0.03;
    constexpr double major_radius = 0.55;
    constexpr double minor_radius = 0.20;
    const RationalQuarterArc2D u_arc = quarter_arc(0.5 * kPi * u_quarter);
    const RationalQuarterArc2D v_arc = quarter_arc(0.5 * kPi * v_quarter);

    std::vector<std::vector<Eigen::Vector3d>> controls(
        3, std::vector<Eigen::Vector3d>(3));
    std::vector<std::vector<double>> weights(3, std::vector<double>(3));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double rho = major_radius
                             + minor_radius * v_arc.controls[j].x();
            controls[i][j] = {
                cx + rho * u_arc.controls[i].x(),
                cy + rho * u_arc.controls[i].y(),
                cz + minor_radius * v_arc.controls[j].y()};
            weights[i][j] = u_arc.weights[i] * v_arc.weights[j];
        }
    }
    return NurbsSurfacePatch3D(
        quadratic_unit_basis(), quadratic_unit_basis(),
        std::move(controls), std::move(weights));
}

NativeNurbsSurface3D make_torus()
{
    NativeNurbsSurface3D surface;
    surface.name = "torus";
    surface.description = "exact rational-quadratic NURBS torus";
    for (int iu = 0; iu < 4; ++iu) {
        for (int iv = 0; iv < 4; ++iv) {
            append_patch(surface,
                         "torus_u" + std::to_string(iu)
                             + "_v" + std::to_string(iv),
                         make_torus_quarter_patch(iu, iv));
        }
    }
    auto patch_id = [](int iu, int iv) {
        return 4 * (iu % 4) + (iv % 4);
    };
    for (int iu = 0; iu < 4; ++iu) {
        for (int iv = 0; iv < 4; ++iv) {
            connect_smooth(surface,
                           patch_id(iu, iv), PatchEdge3D::UMax,
                           patch_id(iu + 1, iv), PatchEdge3D::UMin,
                           false);
            connect_smooth(surface,
                           patch_id(iu, iv), PatchEdge3D::VMax,
                           patch_id(iu, iv + 1), PatchEdge3D::VMin,
                           false);
        }
    }
    surface.expected_area = 4.0 * kPi * kPi * 0.55 * 0.20;
    surface.exact_inside = [](const Eigen::Vector3d& x) {
        const double rho = std::hypot(x.x() - 0.07, x.y() + 0.04);
        return (rho - 0.55) * (rho - 0.55)
             + (x.z() - 0.03) * (x.z() - 0.03) < 0.20 * 0.20;
    };
    return surface;
}

NurbsSurfacePatch3D make_cylinder_wall_quarter(double radius,
                                               int quarter,
                                               double z_first,
                                               double z_second)
{
    constexpr double cx = 0.06;
    constexpr double cy = -0.05;
    const RationalQuarterArc2D arc = quarter_arc(0.5 * kPi * quarter);
    std::vector<std::vector<Eigen::Vector3d>> controls(
        3, std::vector<Eigen::Vector3d>(2));
    std::vector<std::vector<double>> weights(3, std::vector<double>(2));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            controls[i][j] = {
                cx + radius * arc.controls[i].x(),
                cy + radius * arc.controls[i].y(),
                j == 0 ? z_first : z_second};
            weights[i][j] = arc.weights[i];
        }
    }
    return NurbsSurfacePatch3D(
        quadratic_unit_basis(), linear_unit_basis(),
        std::move(controls), std::move(weights));
}

NurbsSurfacePatch3D make_annulus_quarter(double z,
                                         int quarter,
                                         double radius_first,
                                         double radius_second)
{
    constexpr double cx = 0.06;
    constexpr double cy = -0.05;
    const RationalQuarterArc2D arc = quarter_arc(0.5 * kPi * quarter);
    const std::array<double, 2> radii{{radius_first, radius_second}};
    std::vector<std::vector<Eigen::Vector3d>> controls(
        3, std::vector<Eigen::Vector3d>(2));
    std::vector<std::vector<double>> weights(3, std::vector<double>(2));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            controls[i][j] = {
                cx + radii[j] * arc.controls[i].x(),
                cy + radii[j] * arc.controls[i].y(),
                z};
            weights[i][j] = arc.weights[i];
        }
    }
    return NurbsSurfacePatch3D(
        quadratic_unit_basis(), linear_unit_basis(),
        std::move(controls), std::move(weights));
}

NativeNurbsSurface3D make_hollow_cylinder()
{
    constexpr double outer_radius = 0.55;
    constexpr double inner_radius = 0.25;
    constexpr double z0 = -0.63;
    constexpr double z1 = 0.67;
    NativeNurbsSurface3D surface;
    surface.name = "cylinder";
    surface.description = "exact rational-quadratic NURBS hollow cylinder";
    for (int quarter = 0; quarter < 4; ++quarter) {
        append_patch(surface,
                     "outer_wall_q" + std::to_string(quarter),
                     make_cylinder_wall_quarter(
                         outer_radius, quarter, z0, z1));
    }
    for (int quarter = 0; quarter < 4; ++quarter) {
        append_patch(surface,
                     "inner_wall_q" + std::to_string(quarter),
                     make_cylinder_wall_quarter(
                         inner_radius, quarter, z1, z0));
    }
    for (int quarter = 0; quarter < 4; ++quarter) {
        append_patch(surface,
                     "top_annulus_q" + std::to_string(quarter),
                     make_annulus_quarter(
                         z1, quarter, outer_radius, inner_radius));
    }
    for (int quarter = 0; quarter < 4; ++quarter) {
        append_patch(surface,
                     "bottom_annulus_q" + std::to_string(quarter),
                     make_annulus_quarter(
                         z0, quarter, inner_radius, outer_radius));
    }
    for (int sheet = 0; sheet < 4; ++sheet) {
        const int first = 4 * sheet;
        for (int quarter = 0; quarter < 4; ++quarter) {
            connect_smooth(surface,
                           first + quarter, PatchEdge3D::UMax,
                           first + (quarter + 1) % 4, PatchEdge3D::UMin,
                           false);
        }
    }
    for (int quarter = 0; quarter < 4; ++quarter) {
        connect_feature(surface,
            quarter, PatchEdge3D::VMax, 0.0, 1.0,
            8 + quarter, PatchEdge3D::VMin, 0.0, 1.0, false);
        connect_feature(surface,
            quarter, PatchEdge3D::VMin, 0.0, 1.0,
            12 + quarter, PatchEdge3D::VMax, 0.0, 1.0, false);
        connect_feature(surface,
            4 + quarter, PatchEdge3D::VMin, 0.0, 1.0,
            8 + quarter, PatchEdge3D::VMax, 0.0, 1.0, false);
        connect_feature(surface,
            4 + quarter, PatchEdge3D::VMax, 0.0, 1.0,
            12 + quarter, PatchEdge3D::VMin, 0.0, 1.0, false);
    }
    const double height = z1 - z0;
    surface.expected_area = 2.0 * kPi * (outer_radius + inner_radius) * height
        + 2.0 * kPi * (outer_radius * outer_radius
                       - inner_radius * inner_radius);
    surface.exact_inside = [](const Eigen::Vector3d& x) {
        const double radius_sq = (x.x() - 0.06) * (x.x() - 0.06)
                               + (x.y() + 0.05) * (x.y() + 0.05);
        return radius_sq > 0.25 * 0.25
            && radius_sq < 0.55 * 0.55
            && x.z() > -0.63 && x.z() < 0.67;
    };
    return surface;
}

void append_l_prism_top(NativeNurbsSurface3D& surface,
                        const std::string& name,
                        double xmin,
                        double xmax,
                        double ymin,
                        double ymax,
                        double z)
{
    append_patch(surface, name, NurbsSurfacePatch3D::make_bilinear_plane(
        {xmin, ymin, z}, {xmax, ymin, z},
        {xmin, ymax, z}, {xmax, ymax, z}));
}

void append_l_prism_bottom(NativeNurbsSurface3D& surface,
                           const std::string& name,
                           double xmin,
                           double xmax,
                           double ymin,
                           double ymax,
                           double z)
{
    append_patch(surface, name, NurbsSurfacePatch3D::make_bilinear_plane(
        {xmin, ymin, z}, {xmin, ymax, z},
        {xmax, ymin, z}, {xmax, ymax, z}));
}

NativeNurbsSurface3D make_l_prism()
{
    constexpr double sx = 0.07;
    constexpr double sy = -0.07;
    constexpr double z0 = -0.63;
    constexpr double z1 = 0.67;
    constexpr double arm = 0.60;
    const std::array<std::array<double, 4>, 3> cells{{
        {{sx - arm, sx, sy - arm, sy}},
        {{sx, sx + arm, sy - arm, sy}},
        {{sx - arm, sx, sy, sy + arm}}}};
    NativeNurbsSurface3D surface;
    surface.name = "l_prism";
    surface.description = "twelve native bilinear NURBS L-prism patches";
    for (int cell = 0; cell < 3; ++cell) {
        const auto& bounds = cells[static_cast<std::size_t>(cell)];
        append_l_prism_bottom(surface,
                              "bottom_cell_" + std::to_string(cell),
                              bounds[0], bounds[1], bounds[2], bounds[3], z0);
    }
    for (int cell = 0; cell < 3; ++cell) {
        const auto& bounds = cells[static_cast<std::size_t>(cell)];
        append_l_prism_top(surface,
                           "top_cell_" + std::to_string(cell),
                           bounds[0], bounds[1], bounds[2], bounds[3], z1);
    }
    const std::array<Eigen::Vector2d, 6> boundary{{
        {sx - arm, sy - arm}, {sx + arm, sy - arm},
        {sx + arm, sy},       {sx, sy},
        {sx, sy + arm},       {sx - arm, sy + arm}}};
    for (int side = 0; side < 6; ++side) {
        const Eigen::Vector2d a = boundary[static_cast<std::size_t>(side)];
        const Eigen::Vector2d b = boundary[static_cast<std::size_t>((side + 1) % 6)];
        append_patch(surface,
                     "side_" + std::to_string(side),
                     NurbsSurfacePatch3D::make_bilinear_plane(
                         {a.x(), a.y(), z0}, {b.x(), b.y(), z0},
                         {a.x(), a.y(), z1}, {b.x(), b.y(), z1}));
    }

    connect_smooth(surface, 0, PatchEdge3D::VMax,
                   1, PatchEdge3D::VMin, false);
    connect_smooth(surface, 0, PatchEdge3D::UMax,
                   2, PatchEdge3D::UMin, false);
    connect_smooth(surface, 3, PatchEdge3D::UMax,
                   4, PatchEdge3D::UMin, false);
    connect_smooth(surface, 3, PatchEdge3D::VMax,
                   5, PatchEdge3D::VMin, false);

    for (int side = 0; side < 6; ++side) {
        const int side_patch = 6 + side;
        connect_feature(surface,
            side_patch, PatchEdge3D::UMax, 0.0, 1.0,
            6 + (side + 1) % 6, PatchEdge3D::UMin, 0.0, 1.0, false);
    }

    connect_feature(surface, 6, PatchEdge3D::VMin, 0.0, 0.5,
                    0, PatchEdge3D::UMin, 0.0, 1.0, false);
    connect_feature(surface, 6, PatchEdge3D::VMin, 0.5, 1.0,
                    1, PatchEdge3D::UMin, 0.0, 1.0, false);
    connect_feature(surface, 7, PatchEdge3D::VMin, 0.0, 1.0,
                    1, PatchEdge3D::VMax, 0.0, 1.0, false);
    connect_feature(surface, 8, PatchEdge3D::VMin, 0.0, 1.0,
                    1, PatchEdge3D::UMax, 0.0, 1.0, true);
    connect_feature(surface, 9, PatchEdge3D::VMin, 0.0, 1.0,
                    2, PatchEdge3D::VMax, 0.0, 1.0, false);
    connect_feature(surface, 10, PatchEdge3D::VMin, 0.0, 1.0,
                    2, PatchEdge3D::UMax, 0.0, 1.0, true);
    connect_feature(surface, 11, PatchEdge3D::VMin, 0.0, 0.5,
                    2, PatchEdge3D::VMin, 0.0, 1.0, true);
    connect_feature(surface, 11, PatchEdge3D::VMin, 0.5, 1.0,
                    0, PatchEdge3D::VMin, 0.0, 1.0, true);

    connect_feature(surface, 6, PatchEdge3D::VMax, 0.0, 0.5,
                    3, PatchEdge3D::VMin, 0.0, 1.0, false);
    connect_feature(surface, 6, PatchEdge3D::VMax, 0.5, 1.0,
                    4, PatchEdge3D::VMin, 0.0, 1.0, false);
    connect_feature(surface, 7, PatchEdge3D::VMax, 0.0, 1.0,
                    4, PatchEdge3D::UMax, 0.0, 1.0, false);
    connect_feature(surface, 8, PatchEdge3D::VMax, 0.0, 1.0,
                    4, PatchEdge3D::VMax, 0.0, 1.0, true);
    connect_feature(surface, 9, PatchEdge3D::VMax, 0.0, 1.0,
                    5, PatchEdge3D::UMax, 0.0, 1.0, false);
    connect_feature(surface, 10, PatchEdge3D::VMax, 0.0, 1.0,
                    5, PatchEdge3D::VMax, 0.0, 1.0, true);
    connect_feature(surface, 11, PatchEdge3D::VMax, 0.0, 0.5,
                    5, PatchEdge3D::UMin, 0.0, 1.0, true);
    connect_feature(surface, 11, PatchEdge3D::VMax, 0.5, 1.0,
                    3, PatchEdge3D::UMin, 0.0, 1.0, true);

    surface.expected_area = 2.0 * 3.0 * arm * arm
                          + 8.0 * arm * (z1 - z0);
    surface.exact_inside = [](const Eigen::Vector3d& x) {
        const double px = x.x() - 0.07;
        const double py = x.y() + 0.07;
        const bool in_plan =
            (px > -0.60 && px < 0.60 && py > -0.60 && py < 0.0)
         || (px > -0.60 && px < 0.0 && py > 0.0 && py < 0.60);
        return in_plan && x.z() > -0.63 && x.z() < 0.67;
    };
    return surface;
}

} // namespace

geometry3d::NurbsSurfaceModel3D NativeNurbsSurface3D::geometry_model() const
{
    return {patches, patch_components, geometric_connections};
}

NativeNurbsSurface3D make_native_nurbs_surface_3d(GeometryKind3D kind)
{
    switch (kind) {
    case GeometryKind3D::Torus:
        return make_torus();
    case GeometryKind3D::HollowCylinder:
        return make_hollow_cylinder();
    case GeometryKind3D::LPrism:
        return make_l_prism();
    }
    throw std::invalid_argument("unknown native 3D geometry kind");
}

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

SurfaceDofCloud3D make_native_surface_dofs_3d(
    const NativeNurbsSurface3D& surface,
    double h)
{
    if (!std::isfinite(h) || h <= 0.0)
        throw std::invalid_argument("native surface DOFs require positive h");
    if (surface.patches.empty()
        || surface.patch_names.size() != surface.patches.size()
        || surface.smooth_neighbors.size() != surface.patches.size()
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
        tensor_patch.smooth_patch_ids.push_back(patch_id);
        for (const auto& neighbor_slot :
             surface.smooth_neighbors[static_cast<std::size_t>(patch_id)]) {
            if (neighbor_slot
                && std::find(tensor_patch.smooth_patch_ids.begin(),
                             tensor_patch.smooth_patch_ids.end(),
                             neighbor_slot->patch)
                       == tensor_patch.smooth_patch_ids.end()) {
                tensor_patch.smooth_patch_ids.push_back(neighbor_slot->patch);
            }
        }
        std::sort(tensor_patch.smooth_patch_ids.begin(),
                  tensor_patch.smooth_patch_ids.end());
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
            for (int patch_id = 0;
                 patch_id < static_cast<int>(cloud.patches.size());
                 ++patch_id) {
                for (int edge_value = 0; edge_value < 4; ++edge_value) {
                    const auto& connection = surface.smooth_neighbors[
                        static_cast<std::size_t>(patch_id)]
                        [static_cast<std::size_t>(edge_value)];
                    if (!connection)
                        continue;
                    const PatchEdge3D edge =
                        static_cast<PatchEdge3D>(edge_value);
                    SurfaceDofPatch3D& source =
                        cloud.patches[static_cast<std::size_t>(patch_id)];
                    SurfaceDofPatch3D& destination = cloud.patches[
                        static_cast<std::size_t>(connection->patch)];
                    int& source_count =
                        edge == PatchEdge3D::UMin || edge == PatchEdge3D::UMax
                            ? source.nv : source.nu;
                    int& destination_count =
                        connection->edge == PatchEdge3D::UMin
                                || connection->edge == PatchEdge3D::UMax
                            ? destination.nv : destination.nu;
                    const int shared_count =
                        std::max(source_count, destination_count);
                    if (source_count != shared_count
                        || destination_count != shared_count) {
                        source_count = shared_count;
                        destination_count = shared_count;
                        changed = true;
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

void cross_smooth_edge(const SurfaceDofCloud3D& cloud,
                       PatchEdge3D source_edge,
                       const SmoothPatchNeighbor3D& connection,
                       DofLatticeCoordinate3D& coordinate)
{
    const SurfaceDofPatch3D& source =
        cloud.patches[static_cast<std::size_t>(coordinate.patch)];
    const int source_along_index = along_index(source_edge, coordinate);
    const int source_along_count = along_count(source_edge, source);
    double along = (static_cast<double>(source_along_index) + 0.5)
                 / static_cast<double>(source_along_count);
    if (connection.reversed)
        along = 1.0 - along;

    coordinate.patch = connection.patch;
    const SurfaceDofPatch3D& destination =
        cloud.patches[static_cast<std::size_t>(coordinate.patch)];
    const int destination_along_count =
        along_count(connection.edge, destination);
    const int destination_along_index = static_cast<int>(
        std::floor(along * static_cast<double>(destination_along_count)));
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
        const auto& connection =
            surface.smooth_neighbors[static_cast<std::size_t>(coordinate.patch)]
                                    [static_cast<std::size_t>(edge_index(edge))];
        if (connection)
            cross_smooth_edge(cloud, edge, *connection, coordinate);
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
        || surface.smooth_neighbors.size() != surface.patches.size()) {
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
        || surface.smooth_neighbors.size() != surface.patches.size()) {
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
        for (const auto& connection :
             surface.smooth_neighbors[static_cast<std::size_t>(current)]) {
            if (connection
                && !visited[static_cast<std::size_t>(connection->patch)]) {
                visited[static_cast<std::size_t>(connection->patch)] = true;
                pending.push(connection->patch);
            }
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace kfbim::app3d
