#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "src/bulk_solvers/laplace_zfft_bulk_solver_3d.hpp"
#include "src/geometry/grid_pair_3d.hpp"
#include "src/geometry/nurbs_patch_triangulator_3d.hpp"
#include "src/geometry/p2_surface_3d.hpp"
#include "src/gmres/gmres.hpp"
#include "src/grid/cartesian_grid_3d.hpp"
#include "src/potentials/laplace_potential.hpp"
#include "src/transfer/laplace_correction_support.hpp"
#include "src/transfer/laplace_restrict_3d.hpp"
#include "src/transfer/laplace_spread_3d.hpp"

using namespace kfbim;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kBoxMin = -1.5;
constexpr double kBoxSide = 3.0;
constexpr double kTargetP2NodeSpacingOverH = 1.2;
constexpr int kCauchyValueNeighborCount = 48;
constexpr int kCauchyDerivativeNeighborCount = 28;

enum class GeometryKind {
    Torus,
    HollowCylinder,
    LPrism
};

struct GeometryBundle {
    std::string name;
    std::string description;
    Interface3D correction_interface;
    Interface3D crossing_interface;
    int feature_edges = 0;
    int feature_vertices = 0;
    std::function<bool(const Eigen::Vector3d&)> exact_inside;
};

struct SurfacePatchInfo {
    std::string name;
    std::vector<int> adjacent_patch_ids;
};

struct SurfaceDof {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    Eigen::Vector3d tangent1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d tangent2 = Eigen::Vector3d::Zero();
    double weight = 0.0;
    double parameter_a = 0.0;
    double parameter_b = 0.0;
    int patch_id = -1;
};

struct SurfaceDofCloud {
    std::vector<SurfacePatchInfo> patches;
    std::vector<SurfaceDof> dofs;
    double expected_area = 0.0;
};

struct SurfaceCloudDiagnostics {
    double area = 0.0;
    double area_relative_error = 0.0;
    double normal_error = 0.0;
    double nearest_spacing_min_over_h = 0.0;
    double nearest_spacing_mean_over_h = 0.0;
    double nearest_spacing_max_over_h = 0.0;
};

struct CauchyStencil {
    std::vector<int> value_ids;
    std::vector<int> derivative_ids;
    double radius_over_h = 0.0;
    int incident_patch_count = 0;
};

struct CauchyStencilSet {
    int value_count = 0;
    int derivative_count = 0;
    std::vector<CauchyStencil> rows;
    double radius_max_over_h = 0.0;
    double radius_mean_over_h = 0.0;
    int incident_patch_count_min = 0;
    int incident_patch_count_max = 0;
};

struct ReadinessResult {
    std::string geometry;
    int N = 0;
    double h = 0.0;
    int correction_panels = 0;
    int correction_dofs = 0;
    int crossing_panels = 0;
    int feature_edges = 0;
    int feature_vertices = 0;
    int interior_nodes = 0;
    int exterior_nodes = 0;
    int label_mismatches = 0;
    int crossings = 0;
    int exact_crossings = 0;
    int gap_crossings = 0;
    int endpoint_crossings = 0;
    int correction_nodes = 0;
    double correction_area = 0.0;
    double crossing_area = 0.0;
    double normal_error = 0.0;
    double min_box_margin_over_h = 0.0;
    double constant_A1_linf = 0.0;
    double constant_exterior_trace_linf = 0.0;
    double constant_interior_trace_linf = 0.0;
    double constant_bulk_linf = 0.0;
    double harmonic_constant_exterior_trace_linf = 0.0;
    double harmonic_constant_interior_trace_linf = 0.0;
    double harmonic_constant_exterior_normal_linf = 0.0;
    double harmonic_constant_interior_normal_linf = 0.0;
    double harmonic_constant_bulk_linf = 0.0;
    int surface_patches = 0;
    int surface_dofs = 0;
    double surface_dof_area = 0.0;
    double surface_area_relative_error = 0.0;
    double surface_spacing_min_over_h = 0.0;
    double surface_spacing_mean_over_h = 0.0;
    double surface_spacing_max_over_h = 0.0;
    int cauchy_value_neighbors = 0;
    int cauchy_derivative_neighbors = 0;
    double cauchy_radius_max_over_h = 0.0;
    double cauchy_radius_mean_over_h = 0.0;
    int cauchy_incident_patches_min = 0;
    int cauchy_incident_patches_max = 0;
};

double triangle_area(const Eigen::Vector3d& a,
                     const Eigen::Vector3d& b,
                     const Eigen::Vector3d& c)
{
    return 0.5 * (b - a).cross(c - a).norm();
}

bool is_power_of_two(int value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

GeometryKind parse_geometry(const std::string& name)
{
    if (name == "torus" || name == "ring")
        return GeometryKind::Torus;
    if (name == "cylinder" || name == "hollow_cylinder"
        || name == "pipe" || name == "capped_cylinder")
        return GeometryKind::HollowCylinder;
    if (name == "l_prism" || name == "l-prism" || name == "lprism")
        return GeometryKind::LPrism;
    throw std::invalid_argument(
        "geometry must be torus, cylinder, l_prism, or all");
}

double manufactured_u_3d(const Eigen::Vector3d& point)
{
    const double exponential = std::exp(0.35 * point.x());
    return exponential * std::cos(0.21 * point.y())
        * std::cos(0.28 * point.z());
}

Eigen::Vector3d manufactured_gradient_3d(const Eigen::Vector3d& point)
{
    const double exponential = std::exp(0.35 * point.x());
    const double cos_y = std::cos(0.21 * point.y());
    const double sin_y = std::sin(0.21 * point.y());
    const double cos_z = std::cos(0.28 * point.z());
    const double sin_z = std::sin(0.28 * point.z());
    return {
        0.35 * exponential * cos_y * cos_z,
        -0.21 * exponential * sin_y * cos_z,
        -0.28 * exponential * cos_y * sin_z
    };
}

std::pair<Eigen::Vector3d, Eigen::Vector3d>
frame_from_normal(const Eigen::Vector3d& input_normal)
{
    Eigen::Vector3d normal = input_normal;
    const double norm = normal.norm();
    if (!(norm > 0.0) || !std::isfinite(norm))
        throw std::runtime_error("cannot construct a frame from a zero normal");
    normal /= norm;
    const Eigen::Vector3d seed = std::abs(normal.z()) < 0.85
        ? Eigen::Vector3d::UnitZ()
        : Eigen::Vector3d::UnitX();
    Eigen::Vector3d tangent1 = seed.cross(normal).normalized();
    Eigen::Vector3d tangent2 = normal.cross(tangent1).normalized();
    return {tangent1, tangent2};
}

void add_surface_dof(SurfaceDofCloud& cloud,
                     int patch_id,
                     const Eigen::Vector3d& point,
                     const Eigen::Vector3d& input_normal,
                     double weight,
                     double parameter_a,
                     double parameter_b)
{
    if (patch_id < 0 || patch_id >= static_cast<int>(cloud.patches.size()))
        throw std::runtime_error("surface DOF has an invalid patch id");
    if (!(weight > 0.0) || !std::isfinite(weight))
        throw std::runtime_error("surface DOF has a non-positive weight");
    Eigen::Vector3d normal = input_normal;
    if (!(normal.norm() > 0.0))
        throw std::runtime_error("surface DOF has a zero normal");
    normal.normalize();
    const auto frame = frame_from_normal(normal);
    SurfaceDof dof;
    dof.point = point;
    dof.normal = normal;
    dof.tangent1 = frame.first;
    dof.tangent2 = frame.second;
    dof.weight = weight;
    dof.parameter_a = parameter_a;
    dof.parameter_b = parameter_b;
    dof.patch_id = patch_id;
    cloud.dofs.push_back(dof);
}

SurfaceDofCloud make_torus_surface_dofs(double h)
{
    constexpr double cx = 0.07;
    constexpr double cy = -0.04;
    constexpr double cz = 0.03;
    constexpr double major_radius = 0.55;
    constexpr double minor_radius = 0.20;

    SurfaceDofCloud cloud;
    cloud.patches = {{"periodic_torus", {0}}};
    cloud.expected_area = 4.0 * kPi * kPi * major_radius * minor_radius;

    // A single periodic parameter patch.  Both periodic seams are panel
    // boundaries, while every unknown is placed at a parameter-cell center.
    const int nu = std::max(8, static_cast<int>(std::ceil(
        2.0 * kPi * major_radius / h)));
    const int nv = std::max(6, static_cast<int>(std::ceil(
        2.0 * kPi * minor_radius / h)));
    const double du = 2.0 * kPi / static_cast<double>(nu);
    const double dv = 2.0 * kPi / static_cast<double>(nv);
    cloud.dofs.reserve(static_cast<std::size_t>(nu * nv));
    for (int i = 0; i < nu; ++i) {
        const double u = (static_cast<double>(i) + 0.5) * du;
        const double cu = std::cos(u);
        const double su = std::sin(u);
        for (int j = 0; j < nv; ++j) {
            const double v = (static_cast<double>(j) + 0.5) * dv;
            const double cv = std::cos(v);
            const double sv = std::sin(v);
            const double rho = major_radius + minor_radius * cv;
            add_surface_dof(
                cloud,
                0,
                {cx + rho * cu, cy + rho * su, cz + minor_radius * sv},
                {cu * cv, su * cv, sv},
                rho * minor_radius * du * dv,
                u,
                v);
        }
    }
    return cloud;
}

SurfaceDofCloud make_cylinder_surface_dofs(double h)
{
    constexpr double cx = 0.06;
    constexpr double cy = -0.05;
    constexpr double cz = 0.02;
    constexpr double outer_radius = 0.55;
    constexpr double inner_radius = 0.25;
    constexpr double half_height = 0.65;

    SurfaceDofCloud cloud;
    cloud.patches = {
        {"outer_side", {0, 2, 3}},
        {"inner_side", {1, 2, 3}},
        {"top_annulus", {0, 1, 2}},
        {"bottom_annulus", {0, 1, 3}}};
    cloud.expected_area =
        4.0 * kPi * (outer_radius + inner_radius) * half_height
        + 2.0 * kPi * (outer_radius * outer_radius
                       - inner_radius * inner_radius);

    const int nz = std::max(2, static_cast<int>(std::ceil(
        2.0 * half_height / h)));
    const double dz = 2.0 * half_height / static_cast<double>(nz);
    for (int patch_id : {0, 1}) {
        const double radius = patch_id == 0 ? outer_radius : inner_radius;
        const int ntheta = std::max(8, static_cast<int>(std::ceil(
            2.0 * kPi * radius / h)));
        const double dtheta = 2.0 * kPi / static_cast<double>(ntheta);
        for (int i = 0; i < ntheta; ++i) {
            const double theta = (static_cast<double>(i) + 0.5) * dtheta;
            const double ct = std::cos(theta);
            const double st = std::sin(theta);
            const Eigen::Vector3d normal = patch_id == 0
                ? Eigen::Vector3d(ct, st, 0.0)
                : Eigen::Vector3d(-ct, -st, 0.0);
            for (int k = 0; k < nz; ++k) {
                const double z = cz - half_height
                               + (static_cast<double>(k) + 0.5) * dz;
                add_surface_dof(
                    cloud,
                    patch_id,
                    {cx + radius * ct, cy + radius * st, z},
                    normal,
                    radius * dtheta * dz,
                    theta,
                    z);
            }
        }
    }

    // Each cap is split into concentric sector panels.  Radial and angular
    // unknowns are strictly inside their parameter cells, and every weight is
    // the exact sector-annulus area 0.5*(r1^2-r0^2)*dtheta.
    const int nr = std::max(2, static_cast<int>(std::ceil(
        (outer_radius - inner_radius) / h)));
    const double dr = (outer_radius - inner_radius) / static_cast<double>(nr);
    for (int ir = 0; ir < nr; ++ir) {
        const double r0 = inner_radius + static_cast<double>(ir) * dr;
        const double r1 = r0 + dr;
        const double radius = 0.5 * (r0 + r1);
        const int ntheta = std::max(8, static_cast<int>(std::ceil(
            2.0 * kPi * radius / h)));
        const double dtheta = 2.0 * kPi / static_cast<double>(ntheta);
        const double weight = 0.5 * (r1 * r1 - r0 * r0) * dtheta;
        for (int i = 0; i < ntheta; ++i) {
            const double theta = (static_cast<double>(i) + 0.5) * dtheta;
            const double x = cx + radius * std::cos(theta);
            const double y = cy + radius * std::sin(theta);
            add_surface_dof(
                cloud, 2, {x, y, cz + half_height},
                Eigen::Vector3d::UnitZ(), weight, radius, theta);
            add_surface_dof(
                cloud, 3, {x, y, cz - half_height},
                -Eigen::Vector3d::UnitZ(), weight, radius, theta);
        }
    }
    return cloud;
}

SurfaceDofCloud make_l_prism_surface_dofs(double h)
{
    constexpr double sx = 0.07;
    constexpr double sy = -0.07;
    constexpr double z0 = -0.63;
    constexpr double z1 = 0.67;
    constexpr double arm = 0.60;

    SurfaceDofCloud cloud;
    cloud.patches.resize(8);
    const std::array<std::string, 6> side_names{{
        "y_min", "x_max_y_neg", "y_zero_x_pos",
        "x_zero_y_pos", "y_max_x_neg", "x_min"}};
    for (int side = 0; side < 6; ++side) {
        cloud.patches[static_cast<std::size_t>(side)].name = side_names[side];
        cloud.patches[static_cast<std::size_t>(side)].adjacent_patch_ids = {
            side, (side + 5) % 6, (side + 1) % 6, 6, 7};
    }
    cloud.patches[6] = {"top_l_face", {6, 0, 1, 2, 3, 4, 5}};
    cloud.patches[7] = {"bottom_l_face", {7, 0, 1, 2, 3, 4, 5}};
    cloud.expected_area = 2.0 * 3.0 * arm * arm
                        + (8.0 * arm) * (z1 - z0);

    const std::array<Eigen::Vector2d, 6> boundary{{
        {sx - arm, sy - arm}, {sx + arm, sy - arm},
        {sx + arm, sy},       {sx, sy},
        {sx, sy + arm},       {sx - arm, sy + arm}}};
    const Eigen::Vector3d vertical(0.0, 0.0, z1 - z0);
    for (int side = 0; side < 6; ++side) {
        const Eigen::Vector2d a = boundary[static_cast<std::size_t>(side)];
        const Eigen::Vector2d b = boundary[static_cast<std::size_t>((side + 1) % 6)];
        const Eigen::Vector3d edge(b.x() - a.x(), b.y() - a.y(), 0.0);
        const double edge_length = edge.norm();
        const int na = std::max(2, static_cast<int>(std::ceil(edge_length / h)));
        const int nb = std::max(2, static_cast<int>(std::ceil((z1 - z0) / h)));
        const Eigen::Vector3d normal = edge.cross(vertical).normalized();
        const double weight = edge_length * (z1 - z0)
                            / static_cast<double>(na * nb);
        for (int ia = 0; ia < na; ++ia) {
            const double alpha = (static_cast<double>(ia) + 0.5)
                               / static_cast<double>(na);
            for (int ib = 0; ib < nb; ++ib) {
                const double beta = (static_cast<double>(ib) + 0.5)
                              / static_cast<double>(nb);
                const Eigen::Vector3d point(
                    a.x() + alpha * (b.x() - a.x()),
                    a.y() + alpha * (b.y() - a.y()),
                    z0 + beta * (z1 - z0));
                add_surface_dof(
                    cloud, side, point, normal, weight, alpha, beta);
            }
        }
    }

    // Use an even number of cells so the two re-entrant half-lines are panel
    // boundaries, never panel centers.  The missing upper-right quadrant is
    // removed using exactly the L-face center test.
    int nplanar = std::max(2, static_cast<int>(std::ceil(2.0 * arm / h)));
    if (nplanar % 2 != 0)
        ++nplanar;
    const double ds = 2.0 * arm / static_cast<double>(nplanar);
    for (int i = 0; i < nplanar; ++i) {
        const double x = sx - arm + (static_cast<double>(i) + 0.5) * ds;
        for (int j = 0; j < nplanar; ++j) {
            const double y = sy - arm + (static_cast<double>(j) + 0.5) * ds;
            if (!(x < sx || y < sy))
                continue;
            add_surface_dof(
                cloud, 6, {x, y, z1}, Eigen::Vector3d::UnitZ(),
                ds * ds, x, y);
            add_surface_dof(
                cloud, 7, {x, y, z0}, -Eigen::Vector3d::UnitZ(),
                ds * ds, x, y);
        }
    }
    return cloud;
}

SurfaceDofCloud make_surface_dofs(GeometryKind kind, double h)
{
    if (kind == GeometryKind::Torus)
        return make_torus_surface_dofs(h);
    if (kind == GeometryKind::HollowCylinder)
        return make_cylinder_surface_dofs(h);
    return make_l_prism_surface_dofs(h);
}

SurfaceCloudDiagnostics validate_surface_dofs(const SurfaceDofCloud& cloud,
                                              double h)
{
    if (cloud.patches.empty() || cloud.dofs.size() < 2)
        throw std::runtime_error("surface panel-center cloud is empty");
    std::vector<int> patch_counts(cloud.patches.size(), 0);
    SurfaceCloudDiagnostics result;
    for (std::size_t i = 0; i < cloud.patches.size(); ++i) {
        const SurfacePatchInfo& patch = cloud.patches[i];
        if (patch.adjacent_patch_ids.empty()
            || std::find(patch.adjacent_patch_ids.begin(),
                         patch.adjacent_patch_ids.end(),
                         static_cast<int>(i)) == patch.adjacent_patch_ids.end()) {
            throw std::runtime_error("surface patch adjacency must include itself");
        }
        for (int adjacent : patch.adjacent_patch_ids) {
            if (adjacent < 0 || adjacent >= static_cast<int>(cloud.patches.size()))
                throw std::runtime_error("surface patch has invalid adjacency");
        }
    }
    for (const SurfaceDof& dof : cloud.dofs) {
        if (dof.patch_id < 0
            || dof.patch_id >= static_cast<int>(cloud.patches.size()))
            throw std::runtime_error("surface DOF has invalid patch ownership");
        if (!dof.point.allFinite() || !dof.normal.allFinite()
            || !std::isfinite(dof.weight) || !(dof.weight > 0.0))
            throw std::runtime_error("surface DOF contains NaN/Inf");
        ++patch_counts[static_cast<std::size_t>(dof.patch_id)];
        result.area += dof.weight;
        result.normal_error = std::max(
            result.normal_error, std::abs(dof.normal.norm() - 1.0));
    }
    if (std::find(patch_counts.begin(), patch_counts.end(), 0) != patch_counts.end())
        throw std::runtime_error("at least one surface patch has no center DOF");
    result.area_relative_error = std::abs(result.area - cloud.expected_area)
                               / cloud.expected_area;

    double spacing_sum = 0.0;
    double spacing_min = std::numeric_limits<double>::infinity();
    double spacing_max = 0.0;
    for (std::size_t i = 0; i < cloud.dofs.size(); ++i) {
        double nearest_sq = std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < cloud.dofs.size(); ++j) {
            if (i == j)
                continue;
            nearest_sq = std::min(
                nearest_sq,
                (cloud.dofs[i].point - cloud.dofs[j].point).squaredNorm());
        }
        const double spacing = std::sqrt(nearest_sq) / h;
        spacing_sum += spacing;
        spacing_min = std::min(spacing_min, spacing);
        spacing_max = std::max(spacing_max, spacing);
    }
    result.nearest_spacing_min_over_h = spacing_min;
    result.nearest_spacing_mean_over_h = spacing_sum
        / static_cast<double>(cloud.dofs.size());
    result.nearest_spacing_max_over_h = spacing_max;
    return result;
}

CauchyStencilSet build_cauchy_stencils(const SurfaceDofCloud& cloud,
                                       double h,
                                       int requested_value_count,
                                       int requested_derivative_count)
{
    if (requested_value_count <= 0 || requested_derivative_count < 0)
        throw std::invalid_argument("invalid Cauchy stencil sizes");
    std::vector<std::vector<int>> patch_dofs(cloud.patches.size());
    for (int q = 0; q < static_cast<int>(cloud.dofs.size()); ++q)
        patch_dofs[static_cast<std::size_t>(cloud.dofs[q].patch_id)].push_back(q);

    int minimum_candidate_count = std::numeric_limits<int>::max();
    for (const SurfacePatchInfo& patch : cloud.patches) {
        int count = 0;
        std::set<int> unique_patches;
        for (int adjacent : patch.adjacent_patch_ids)
            unique_patches.insert(adjacent);
        for (int adjacent : unique_patches)
            count += static_cast<int>(patch_dofs[static_cast<std::size_t>(adjacent)].size());
        minimum_candidate_count = std::min(minimum_candidate_count, count);
    }

    CauchyStencilSet result;
    result.value_count = std::min(requested_value_count, minimum_candidate_count);
    result.derivative_count = std::min(
        requested_derivative_count, result.value_count);
    if (result.value_count <= 0)
        throw std::runtime_error("no candidates available for a Cauchy stencil");
    result.rows.reserve(cloud.dofs.size());
    result.incident_patch_count_min = std::numeric_limits<int>::max();

    for (int center = 0; center < static_cast<int>(cloud.dofs.size()); ++center) {
        const SurfaceDof& target = cloud.dofs[static_cast<std::size_t>(center)];
        const SurfacePatchInfo& patch =
            cloud.patches[static_cast<std::size_t>(target.patch_id)];
        std::vector<std::pair<double, int>> candidates;
        std::set<int> unique_patches(
            patch.adjacent_patch_ids.begin(), patch.adjacent_patch_ids.end());
        for (int adjacent : unique_patches) {
            for (int q : patch_dofs[static_cast<std::size_t>(adjacent)]) {
                const double distance_sq =
                    (cloud.dofs[static_cast<std::size_t>(q)].point - target.point)
                    .squaredNorm();
                candidates.emplace_back(distance_sq, q);
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) {
                      if (a.first != b.first)
                          return a.first < b.first;
                      return a.second < b.second;
                  });
        if (static_cast<int>(candidates.size()) < result.value_count)
            throw std::runtime_error("Cauchy candidate count changed unexpectedly");

        CauchyStencil stencil;
        stencil.value_ids.reserve(static_cast<std::size_t>(result.value_count));
        for (int k = 0; k < result.value_count; ++k)
            stencil.value_ids.push_back(candidates[static_cast<std::size_t>(k)].second);
        if (std::find(stencil.value_ids.begin(), stencil.value_ids.end(), center)
            == stencil.value_ids.end()) {
            throw std::runtime_error("Cauchy value stencil does not contain its center");
        }
        stencil.derivative_ids.assign(
            stencil.value_ids.begin(),
            stencil.value_ids.begin() + result.derivative_count);
        stencil.radius_over_h = std::sqrt(
            candidates[static_cast<std::size_t>(result.value_count - 1)].first) / h;
        std::set<int> selected_patches;
        for (int q : stencil.value_ids)
            selected_patches.insert(cloud.dofs[static_cast<std::size_t>(q)].patch_id);
        stencil.incident_patch_count = static_cast<int>(selected_patches.size());

        result.radius_max_over_h = std::max(
            result.radius_max_over_h, stencil.radius_over_h);
        result.radius_mean_over_h += stencil.radius_over_h;
        result.incident_patch_count_min = std::min(
            result.incident_patch_count_min, stencil.incident_patch_count);
        result.incident_patch_count_max = std::max(
            result.incident_patch_count_max, stencil.incident_patch_count);
        result.rows.push_back(std::move(stencil));
    }
    result.radius_mean_over_h /= static_cast<double>(result.rows.size());
    return result;
}

struct PolynomialPower3D {
    int x = 0;
    int y = 0;
    int z = 0;
};

double integer_power(double value, int power)
{
    double result = 1.0;
    for (int i = 0; i < power; ++i)
        result *= value;
    return result;
}

class HarmonicPolynomialSpace3D {
public:
    explicit HarmonicPolynomialSpace3D(int degree = 4)
        : degree_(degree)
    {
        if (degree_ < 1)
            throw std::invalid_argument("harmonic polynomial degree must be positive");
        powers_ = powers_through_degree(degree_);
        const std::vector<PolynomialPower3D> lower =
            powers_through_degree(degree_ - 2);
        std::map<std::array<int, 3>, int> lower_index;
        for (int i = 0; i < static_cast<int>(lower.size()); ++i) {
            lower_index[{lower[static_cast<std::size_t>(i)].x,
                         lower[static_cast<std::size_t>(i)].y,
                         lower[static_cast<std::size_t>(i)].z}] = i;
        }

        Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(
            static_cast<int>(lower.size()), static_cast<int>(powers_.size()));
        for (int col = 0; col < static_cast<int>(powers_.size()); ++col) {
            const PolynomialPower3D p = powers_[static_cast<std::size_t>(col)];
            if (p.x >= 2) {
                laplacian(lower_index.at({p.x - 2, p.y, p.z}), col)
                    += static_cast<double>(p.x * (p.x - 1));
            }
            if (p.y >= 2) {
                laplacian(lower_index.at({p.x, p.y - 2, p.z}), col)
                    += static_cast<double>(p.y * (p.y - 1));
            }
            if (p.z >= 2) {
                laplacian(lower_index.at({p.x, p.y, p.z - 2}), col)
                    += static_cast<double>(p.z * (p.z - 1));
            }
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            laplacian, Eigen::ComputeFullV);
        const int expected_dimension = (degree_ + 1) * (degree_ + 1);
        if (static_cast<int>(powers_.size()) - laplacian.rows()
            != expected_dimension) {
            throw std::runtime_error("unexpected 3D harmonic-space dimension");
        }
        transform_ = svd.matrixV().rightCols(expected_dimension);
        if ((laplacian * transform_).norm() > 1.0e-10)
            throw std::runtime_error("failed to construct a harmonic polynomial basis");
    }

    int dimension() const
    {
        return static_cast<int>(transform_.cols());
    }

    Eigen::VectorXd basis(double x, double y, double z) const
    {
        Eigen::VectorXd monomials(static_cast<int>(powers_.size()));
        for (int i = 0; i < monomials.size(); ++i) {
            const PolynomialPower3D p = powers_[static_cast<std::size_t>(i)];
            monomials[i] = integer_power(x, p.x)
                         * integer_power(y, p.y)
                         * integer_power(z, p.z);
        }
        return transform_.transpose() * monomials;
    }

    Eigen::MatrixXd gradient(double x, double y, double z) const
    {
        Eigen::MatrixXd monomial_gradient = Eigen::MatrixXd::Zero(
            3, static_cast<int>(powers_.size()));
        for (int i = 0; i < static_cast<int>(powers_.size()); ++i) {
            const PolynomialPower3D p = powers_[static_cast<std::size_t>(i)];
            if (p.x > 0) {
                monomial_gradient(0, i) = static_cast<double>(p.x)
                    * integer_power(x, p.x - 1)
                    * integer_power(y, p.y)
                    * integer_power(z, p.z);
            }
            if (p.y > 0) {
                monomial_gradient(1, i) = static_cast<double>(p.y)
                    * integer_power(x, p.x)
                    * integer_power(y, p.y - 1)
                    * integer_power(z, p.z);
            }
            if (p.z > 0) {
                monomial_gradient(2, i) = static_cast<double>(p.z)
                    * integer_power(x, p.x)
                    * integer_power(y, p.y)
                    * integer_power(z, p.z - 1);
            }
        }
        return monomial_gradient * transform_;
    }

private:
    static std::vector<PolynomialPower3D> powers_through_degree(int degree)
    {
        std::vector<PolynomialPower3D> result;
        if (degree < 0)
            return result;
        for (int total = 0; total <= degree; ++total) {
            for (int px = 0; px <= total; ++px) {
                for (int py = 0; py <= total - px; ++py)
                    result.push_back({px, py, total - px - py});
            }
        }
        return result;
    }

    int degree_ = 4;
    std::vector<PolynomialPower3D> powers_;
    Eigen::MatrixXd transform_;
};

Eigen::MatrixXd svd_pseudoinverse(const Eigen::MatrixXd& matrix,
                                  double relative_cutoff)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd singular = svd.singularValues();
    if (singular.size() == 0 || !(singular[0] > 0.0))
        throw std::runtime_error("cannot invert an empty Cauchy design matrix");
    Eigen::VectorXd inverse = singular;
    const double cutoff = relative_cutoff * singular[0];
    for (int i = 0; i < inverse.size(); ++i)
        inverse[i] = singular[i] > cutoff ? 1.0 / singular[i] : 0.0;
    return svd.matrixV() * inverse.asDiagonal() * svd.matrixU().transpose();
}

struct CauchyFitMap3D {
    Eigen::MatrixXd value_map;
    Eigen::MatrixXd normal_map;
    double condition = 0.0;
};

class PanelCenterCauchyFit3D {
public:
    PanelCenterCauchyFit3D(const SurfaceDofCloud& cloud,
                           const CauchyStencilSet& stencils,
                           double h,
                           int degree = 4)
        : cloud_(cloud)
        , stencils_(stencils)
        , h_(h)
        , space_(degree)
    {
        if (stencils_.rows.size() != cloud_.dofs.size())
            throw std::invalid_argument("Cauchy stencil count does not match surface DOFs");
        maps_.reserve(cloud_.dofs.size());
        for (int center = 0; center < static_cast<int>(cloud_.dofs.size()); ++center)
            maps_.push_back(build_map(center));
    }

    int dimension() const { return space_.dimension(); }
    const HarmonicPolynomialSpace3D& space() const { return space_; }

    Eigen::MatrixXd coefficients(const Eigen::VectorXd& value_jump,
                                 const Eigen::VectorXd& normal_jump) const
    {
        const int size = static_cast<int>(cloud_.dofs.size());
        if (value_jump.size() != size || normal_jump.size() != size)
            throw std::invalid_argument("jump data size does not match surface DOFs");
        Eigen::MatrixXd result(size, dimension());
        for (int center = 0; center < size; ++center) {
            const CauchyStencil& stencil =
                stencils_.rows[static_cast<std::size_t>(center)];
            Eigen::VectorXd values(stencils_.value_count);
            Eigen::VectorXd normals(stencils_.derivative_count);
            for (int k = 0; k < stencils_.value_count; ++k)
                values[k] = value_jump[stencil.value_ids[static_cast<std::size_t>(k)]];
            for (int k = 0; k < stencils_.derivative_count; ++k)
                normals[k] = normal_jump[stencil.derivative_ids[static_cast<std::size_t>(k)]];
            const CauchyFitMap3D& map = maps_[static_cast<std::size_t>(center)];
            result.row(center) =
                (map.value_map * values + map.normal_map * normals).transpose();
        }
        return result;
    }

private:
    Eigen::Vector3d local_coordinate(const SurfaceDof& center,
                                     const Eigen::Vector3d& point) const
    {
        const Eigen::Vector3d displacement = (point - center.point) / h_;
        return {displacement.dot(center.tangent1),
                displacement.dot(center.tangent2),
                displacement.dot(center.normal)};
    }

    CauchyFitMap3D build_map(int center_id) const
    {
        const SurfaceDof& center = cloud_.dofs[static_cast<std::size_t>(center_id)];
        const CauchyStencil& stencil =
            stencils_.rows[static_cast<std::size_t>(center_id)];
        const int value_count = stencils_.value_count;
        const int normal_count = stencils_.derivative_count;
        const int rows = value_count + normal_count;
        Eigen::MatrixXd design(rows, dimension());
        Eigen::VectorXd sqrt_weights(rows);

        for (int k = 0; k < value_count; ++k) {
            const SurfaceDof& sample = cloud_.dofs[static_cast<std::size_t>(
                stencil.value_ids[static_cast<std::size_t>(k)])];
            const Eigen::Vector3d xi = local_coordinate(center, sample.point);
            design.row(k) = space_.basis(xi.x(), xi.y(), xi.z()).transpose();
            sqrt_weights[k] = std::sqrt(
                1.0 / std::pow(0.35 + xi.norm(), 2.0));
        }
        for (int k = 0; k < normal_count; ++k) {
            const SurfaceDof& sample = cloud_.dofs[static_cast<std::size_t>(
                stencil.derivative_ids[static_cast<std::size_t>(k)])];
            const Eigen::Vector3d xi = local_coordinate(center, sample.point);
            const Eigen::Vector3d normal_components(
                sample.normal.dot(center.tangent1),
                sample.normal.dot(center.tangent2),
                sample.normal.dot(center.normal));
            design.row(value_count + k) =
                normal_components.transpose()
                * space_.gradient(xi.x(), xi.y(), xi.z());
            sqrt_weights[value_count + k] = std::sqrt(
                0.85 / std::pow(0.35 + xi.norm(), 2.0));
        }

        const Eigen::MatrixXd weighted = sqrt_weights.asDiagonal() * design;
        Eigen::JacobiSVD<Eigen::MatrixXd> condition_svd(weighted);
        const Eigen::VectorXd singular = condition_svd.singularValues();
        CauchyFitMap3D result;
        result.condition = singular[0] / singular[singular.size() - 1];
        const Eigen::MatrixXd pinv = svd_pseudoinverse(weighted, 3.0e-12);
        result.value_map.resize(dimension(), value_count);
        result.normal_map.resize(dimension(), normal_count);
        for (int k = 0; k < value_count; ++k)
            result.value_map.col(k) = pinv.col(k) * sqrt_weights[k];
        for (int k = 0; k < normal_count; ++k) {
            result.normal_map.col(k) =
                pinv.col(value_count + k) * sqrt_weights[value_count + k] * h_;
        }
        return result;
    }

    const SurfaceDofCloud& cloud_;
    const CauchyStencilSet& stencils_;
    double h_ = 0.0;
    HarmonicPolynomialSpace3D space_;
    std::vector<CauchyFitMap3D> maps_;
};

struct TorusMeshData {
    std::map<std::pair<int, int>, int> node_by_key;
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> normals;
    std::vector<double> weights;
    std::vector<std::array<int, 3>> panels;
    std::vector<std::array<int, 6>> panel_points;
};

int positive_mod(int value, int period)
{
    const int result = value % period;
    return result < 0 ? result + period : result;
}

int add_torus_node(TorusMeshData& mesh, int iu, int iv, int nu, int nv)
{
    const int ku = positive_mod(iu, 2 * nu);
    const int kv = positive_mod(iv, 2 * nv);
    const std::pair<int, int> key{ku, kv};
    const auto found = mesh.node_by_key.find(key);
    if (found != mesh.node_by_key.end())
        return found->second;

    constexpr double cx = 0.07;
    constexpr double cy = -0.04;
    constexpr double cz = 0.03;
    constexpr double major_radius = 0.55;
    constexpr double minor_radius = 0.20;
    const double u = kPi * static_cast<double>(ku) / static_cast<double>(nu);
    const double v = kPi * static_cast<double>(kv) / static_cast<double>(nv);
    const double cu = std::cos(u);
    const double su = std::sin(u);
    const double cv = std::cos(v);
    const double sv = std::sin(v);
    const double rho = major_radius + minor_radius * cv;

    const int index = static_cast<int>(mesh.points.size());
    mesh.node_by_key.emplace(key, index);
    mesh.points.emplace_back(cx + rho * cu, cy + rho * su, cz + minor_radius * sv);
    mesh.normals.emplace_back(cu * cv, su * cv, sv);
    mesh.weights.push_back(0.0);
    return index;
}

void add_torus_panel(TorusMeshData& mesh,
                     int a, int b, int c, int eab, int ebc, int eca)
{
    mesh.panels.push_back({a, b, c});
    mesh.panel_points.push_back({a, b, c, eab, ebc, eca});
    const double weight = triangle_area(
        mesh.points[a], mesh.points[b], mesh.points[c]) / 6.0;
    for (int point : {a, b, c, eab, ebc, eca})
        mesh.weights[static_cast<std::size_t>(point)] += weight;
}

Interface3D make_torus_interface(double h)
{
    constexpr double major_radius = 0.55;
    constexpr double minor_radius = 0.20;
    const double target_H = 2.0 * kTargetP2NodeSpacingOverH * h;
    const int nu = std::max(8, static_cast<int>(std::lround(
        2.0 * kPi * (major_radius + minor_radius) / target_H)));
    const int nv = std::max(6, static_cast<int>(std::lround(
        2.0 * kPi * minor_radius / target_H)));

    TorusMeshData mesh;
    for (int i = 0; i < nu; ++i) {
        for (int j = 0; j < nv; ++j) {
            const int a = add_torus_node(mesh, 2 * i, 2 * j, nu, nv);
            const int b = add_torus_node(mesh, 2 * (i + 1), 2 * j, nu, nv);
            const int c = add_torus_node(mesh, 2 * i, 2 * (j + 1), nu, nv);
            const int d = add_torus_node(
                mesh, 2 * (i + 1), 2 * (j + 1), nu, nv);
            const int eab = add_torus_node(mesh, 2 * i + 1, 2 * j, nu, nv);
            const int ebc = add_torus_node(
                mesh, 2 * i + 1, 2 * j + 1, nu, nv);
            const int eca = add_torus_node(mesh, 2 * i, 2 * j + 1, nu, nv);
            add_torus_panel(mesh, a, b, c, eab, ebc, eca);

            const int ebd = add_torus_node(
                mesh, 2 * (i + 1), 2 * j + 1, nu, nv);
            const int edc = add_torus_node(
                mesh, 2 * i + 1, 2 * (j + 1), nu, nv);
            add_torus_panel(mesh, b, d, c, ebd, edc, ebc);
        }
    }

    const int nq = static_cast<int>(mesh.points.size());
    const int np = static_cast<int>(mesh.panels.size());
    Eigen::MatrixX3d points(nq, 3);
    Eigen::MatrixX3d normals(nq, 3);
    Eigen::VectorXd weights(nq);
    for (int q = 0; q < nq; ++q) {
        points.row(q) = mesh.points[static_cast<std::size_t>(q)].transpose();
        normals.row(q) = mesh.normals[static_cast<std::size_t>(q)].transpose();
        weights[q] = mesh.weights[static_cast<std::size_t>(q)];
    }
    Eigen::MatrixX3i panels(np, 3);
    Eigen::MatrixXi panel_points(np, 6);
    for (int p = 0; p < np; ++p) {
        for (int j = 0; j < 3; ++j)
            panels(p, j) = mesh.panels[static_cast<std::size_t>(p)][j];
        for (int j = 0; j < 6; ++j)
            panel_points(p, j) = mesh.panel_points[static_cast<std::size_t>(p)][j];
    }
    return Interface3D(points,
                       panels,
                       points,
                       normals,
                       weights,
                       6,
                       panel_points,
                       Eigen::VectorXi::Zero(np),
                       PanelNodeLayout3D::QuadraticLagrange);
}

struct CoarseTriangle {
    std::array<int, 3> vertices;
    int smooth_group = 0;
};

Interface3D make_hollow_cylinder_interface(double h)
{
    constexpr double cx = 0.06;
    constexpr double cy = -0.05;
    constexpr double cz = 0.02;
    constexpr double outer_radius = 0.55;
    constexpr double inner_radius = 0.25;
    constexpr double half_height = 0.65;
    const double target_H = 2.0 * kTargetP2NodeSpacingOverH * h;
    const int ntheta = std::max(12, static_cast<int>(std::lround(
        2.0 * kPi * outer_radius / target_H)));
    const int nz = std::max(2, static_cast<int>(std::ceil(
        2.0 * half_height / target_H)));
    const int nr = std::max(1, static_cast<int>(std::ceil(
        (outer_radius - inner_radius) / target_H)));

    std::vector<Eigen::Vector3d> vertices;
    auto add_vertex = [&](double radius, double theta, double z) {
        const int id = static_cast<int>(vertices.size());
        vertices.emplace_back(cx + radius * std::cos(theta),
                              cy + radius * std::sin(theta), z);
        return id;
    };

    std::vector<std::vector<int>> bottom_rings(
        static_cast<std::size_t>(nr + 1),
        std::vector<int>(static_cast<std::size_t>(ntheta)));
    std::vector<std::vector<int>> top_rings = bottom_rings;
    for (int ir = 0; ir <= nr; ++ir) {
        const double radius = inner_radius
            + (outer_radius - inner_radius) * static_cast<double>(ir)
              / static_cast<double>(nr);
        for (int i = 0; i < ntheta; ++i) {
            const double theta = 2.0 * kPi * static_cast<double>(i)
                               / static_cast<double>(ntheta);
            bottom_rings[static_cast<std::size_t>(ir)][static_cast<std::size_t>(i)] =
                add_vertex(radius, theta, cz - half_height);
            top_rings[static_cast<std::size_t>(ir)][static_cast<std::size_t>(i)] =
                add_vertex(radius, theta, cz + half_height);
        }
    }

    std::vector<std::vector<int>> outer_rings(
        static_cast<std::size_t>(nz + 1),
        std::vector<int>(static_cast<std::size_t>(ntheta)));
    std::vector<std::vector<int>> inner_rings = outer_rings;
    for (int i = 0; i < ntheta; ++i) {
        outer_rings[0][static_cast<std::size_t>(i)] =
            bottom_rings[static_cast<std::size_t>(nr)][static_cast<std::size_t>(i)];
        outer_rings[static_cast<std::size_t>(nz)][static_cast<std::size_t>(i)] =
            top_rings[static_cast<std::size_t>(nr)][static_cast<std::size_t>(i)];
        inner_rings[0][static_cast<std::size_t>(i)] =
            bottom_rings[0][static_cast<std::size_t>(i)];
        inner_rings[static_cast<std::size_t>(nz)][static_cast<std::size_t>(i)] =
            top_rings[0][static_cast<std::size_t>(i)];
    }
    for (int k = 1; k < nz; ++k) {
        const double z = cz - half_height
            + 2.0 * half_height * static_cast<double>(k)
              / static_cast<double>(nz);
        for (int i = 0; i < ntheta; ++i) {
            const double theta = 2.0 * kPi * static_cast<double>(i)
                               / static_cast<double>(ntheta);
            outer_rings[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)] =
                add_vertex(outer_radius, theta, z);
            inner_rings[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)] =
                add_vertex(inner_radius, theta, z);
        }
    }

    std::vector<CoarseTriangle> triangles;
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < ntheta; ++i) {
            const int ip = positive_mod(i + 1, ntheta);
            const int a = outer_rings[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)];
            const int b = outer_rings[static_cast<std::size_t>(k)][static_cast<std::size_t>(ip)];
            const int c = outer_rings[static_cast<std::size_t>(k + 1)][static_cast<std::size_t>(i)];
            const int d = outer_rings[static_cast<std::size_t>(k + 1)][static_cast<std::size_t>(ip)];
            triangles.push_back({{a, b, c}, 0});
            triangles.push_back({{b, d, c}, 0});

            const int ai = inner_rings[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)];
            const int bi = inner_rings[static_cast<std::size_t>(k)][static_cast<std::size_t>(ip)];
            const int ci = inner_rings[static_cast<std::size_t>(k + 1)][static_cast<std::size_t>(i)];
            const int di = inner_rings[static_cast<std::size_t>(k + 1)][static_cast<std::size_t>(ip)];
            triangles.push_back({{ai, ci, bi}, 1});
            triangles.push_back({{bi, ci, di}, 1});
        }
    }
    for (int ir = 0; ir < nr; ++ir) {
        for (int i = 0; i < ntheta; ++i) {
            const int ip = positive_mod(i + 1, ntheta);
            const int a = top_rings[static_cast<std::size_t>(ir)][static_cast<std::size_t>(i)];
            const int b = top_rings[static_cast<std::size_t>(ir + 1)][static_cast<std::size_t>(i)];
            const int c = top_rings[static_cast<std::size_t>(ir)][static_cast<std::size_t>(ip)];
            const int d = top_rings[static_cast<std::size_t>(ir + 1)][static_cast<std::size_t>(ip)];
            triangles.push_back({{a, b, c}, 2});
            triangles.push_back({{b, d, c}, 2});

            const int ab = bottom_rings[static_cast<std::size_t>(ir)][static_cast<std::size_t>(i)];
            const int bb = bottom_rings[static_cast<std::size_t>(ir + 1)][static_cast<std::size_t>(i)];
            const int cb = bottom_rings[static_cast<std::size_t>(ir)][static_cast<std::size_t>(ip)];
            const int db = bottom_rings[static_cast<std::size_t>(ir + 1)][static_cast<std::size_t>(ip)];
            triangles.push_back({{ab, cb, bb}, 3});
            triangles.push_back({{bb, cb, db}, 3});
        }
    }

    std::map<std::tuple<int, int, int>, int> point_by_key;
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> normals;
    std::vector<double> weights;
    auto add_point = [&](int group, int va, int vb) {
        const int lo = std::min(va, vb);
        const int hi = std::max(va, vb);
        const std::tuple<int, int, int> key{group, lo, hi};
        const auto found = point_by_key.find(key);
        if (found != point_by_key.end())
            return found->second;

        Eigen::Vector3d point = va == vb
            ? vertices[static_cast<std::size_t>(va)]
            : 0.5 * (vertices[static_cast<std::size_t>(va)]
                     + vertices[static_cast<std::size_t>(vb)]);
        Eigen::Vector3d normal = Eigen::Vector3d::Zero();
        if (group == 0 || group == 1) {
            Eigen::Vector2d radial(point.x() - cx, point.y() - cy);
            if (radial.norm() <= 1.0e-14)
                throw std::runtime_error("degenerate cylinder side point");
            radial.normalize();
            const double radius = group == 0 ? outer_radius : inner_radius;
            point.x() = cx + radius * radial.x();
            point.y() = cy + radius * radial.y();
            normal = group == 0
                ? Eigen::Vector3d(radial.x(), radial.y(), 0.0)
                : Eigen::Vector3d(-radial.x(), -radial.y(), 0.0);
        } else {
            if (group == 2)
                normal = Eigen::Vector3d::UnitZ();
            else
                normal = -Eigen::Vector3d::UnitZ();
        }
        const int index = static_cast<int>(points.size());
        point_by_key.emplace(key, index);
        points.push_back(point);
        normals.push_back(normal);
        weights.push_back(0.0);
        return index;
    };

    const int np = static_cast<int>(triangles.size());
    Eigen::MatrixX3i panels(np, 3);
    Eigen::MatrixXi panel_points(np, 6);
    Eigen::VectorXi smooth_groups(np);
    for (int p = 0; p < np; ++p) {
        const CoarseTriangle& triangle = triangles[static_cast<std::size_t>(p)];
        const int a = triangle.vertices[0];
        const int b = triangle.vertices[1];
        const int c = triangle.vertices[2];
        panels.row(p) << a, b, c;
        smooth_groups[p] = triangle.smooth_group;
        const std::array<int, 6> local{{
            add_point(triangle.smooth_group, a, a),
            add_point(triangle.smooth_group, b, b),
            add_point(triangle.smooth_group, c, c),
            add_point(triangle.smooth_group, a, b),
            add_point(triangle.smooth_group, b, c),
            add_point(triangle.smooth_group, c, a)}};
        const double weight = triangle_area(vertices[a], vertices[b], vertices[c]) / 6.0;
        for (int q = 0; q < 6; ++q) {
            panel_points(p, q) = local[q];
            weights[static_cast<std::size_t>(local[q])] += weight;
        }
    }

    const int nv = static_cast<int>(vertices.size());
    const int nq = static_cast<int>(points.size());
    Eigen::MatrixX3d vertex_matrix(nv, 3);
    Eigen::MatrixX3d point_matrix(nq, 3);
    Eigen::MatrixX3d normal_matrix(nq, 3);
    Eigen::VectorXd weight_vector(nq);
    for (int i = 0; i < nv; ++i)
        vertex_matrix.row(i) = vertices[static_cast<std::size_t>(i)].transpose();
    for (int i = 0; i < nq; ++i) {
        point_matrix.row(i) = points[static_cast<std::size_t>(i)].transpose();
        normal_matrix.row(i) = normals[static_cast<std::size_t>(i)].transpose();
        weight_vector[i] = weights[static_cast<std::size_t>(i)];
    }
    return Interface3D(vertex_matrix,
                       panels,
                       point_matrix,
                       normal_matrix,
                       weight_vector,
                       6,
                       panel_points,
                       Eigen::VectorXi::Zero(np),
                       smooth_groups,
                       PanelNodeLayout3D::QuadraticLagrange);
}

using geometry3d::NurbsSurfacePatch3D;

void add_l_prism_top(std::vector<NurbsSurfacePatch3D>& patches,
                     double xmin, double xmax,
                     double ymin, double ymax, double z)
{
    patches.push_back(NurbsSurfacePatch3D::make_bilinear_plane(
        {xmin, ymin, z}, {xmax, ymin, z},
        {xmin, ymax, z}, {xmax, ymax, z}));
}

void add_l_prism_bottom(std::vector<NurbsSurfacePatch3D>& patches,
                        double xmin, double xmax,
                        double ymin, double ymax, double z)
{
    patches.push_back(NurbsSurfacePatch3D::make_bilinear_plane(
        {xmin, ymin, z}, {xmin, ymax, z},
        {xmax, ymin, z}, {xmax, ymax, z}));
}

void add_l_prism_side(std::vector<NurbsSurfacePatch3D>& patches,
                      const Eigen::Vector2d& a,
                      const Eigen::Vector2d& b,
                      double z0, double z1)
{
    patches.push_back(NurbsSurfacePatch3D::make_bilinear_plane(
        {a.x(), a.y(), z0}, {b.x(), b.y(), z0},
        {a.x(), a.y(), z1}, {b.x(), b.y(), z1}));
}

std::vector<NurbsSurfacePatch3D> make_l_prism_patches()
{
    // Keep all feature planes away from the N=2^k Cartesian nodes used by
    // convergence studies; near-coincidence makes local-normal labels
    // ambiguous at a sharp edge.
    constexpr double sx = 0.07;
    constexpr double sy = -0.07;
    constexpr double z0 = -0.63;
    constexpr double z1 = 0.67;
    const std::array<std::array<double, 4>, 3> cells{{
        {{sx - 0.60, sx, sy - 0.60, sy}},
        {{sx, sx + 0.60, sy - 0.60, sy}},
        {{sx - 0.60, sx, sy, sy + 0.60}}}};

    std::vector<NurbsSurfacePatch3D> patches;
    patches.reserve(12);
    for (const auto& cell : cells)
        add_l_prism_bottom(
            patches, cell[0], cell[1], cell[2], cell[3], z0);
    for (const auto& cell : cells)
        add_l_prism_top(
            patches, cell[0], cell[1], cell[2], cell[3], z1);

    const std::array<Eigen::Vector2d, 6> boundary{{
        {sx - 0.60, sy - 0.60}, {sx + 0.60, sy - 0.60},
        {sx + 0.60, sy},        {sx, sy},
        {sx, sy + 0.60},        {sx - 0.60, sy + 0.60}}};
    for (std::size_t i = 0; i < boundary.size(); ++i)
        add_l_prism_side(
            patches, boundary[i], boundary[(i + 1) % boundary.size()], z0, z1);
    return patches;
}

GeometryBundle make_geometry(GeometryKind kind, double h)
{
    if (kind == GeometryKind::Torus) {
        Interface3D iface = make_torus_interface(h);
        Interface3D crossing = iface;
        return {"torus",
                "smooth P2 torus (circular ring)",
                std::move(iface),
                std::move(crossing),
                0,
                0,
                [](const Eigen::Vector3d& x) {
                    constexpr double cx = 0.07;
                    constexpr double cy = -0.04;
                    constexpr double cz = 0.03;
                    constexpr double R = 0.55;
                    constexpr double r = 0.20;
                    const double rho = std::hypot(x.x() - cx, x.y() - cy);
                    return (rho - R) * (rho - R) + (x.z() - cz) * (x.z() - cz)
                         < r * r;
                }};
    }
    if (kind == GeometryKind::HollowCylinder) {
        Interface3D iface = make_hollow_cylinder_interface(h);
        Interface3D crossing = iface;
        return {"cylinder",
                "finite hollow cylinder with annular sector end caps",
                std::move(iface),
                std::move(crossing),
                4,
                0,
                [](const Eigen::Vector3d& x) {
                    constexpr double cx = 0.06;
                    constexpr double cy = -0.05;
                    constexpr double cz = 0.02;
                    constexpr double outer_radius = 0.55;
                    constexpr double inner_radius = 0.25;
                    constexpr double half_height = 0.65;
                    const double radial_sq =
                        (x.x() - cx) * (x.x() - cx)
                      + (x.y() - cy) * (x.y() - cy);
                    return radial_sq > inner_radius * inner_radius
                        && radial_sq < outer_radius * outer_radius
                        && std::abs(x.z() - cz) < half_height;
                }};
    }

    geometry3d::NurbsPatchTriangulatorOptions3D options;
    options.edge_buffer_factor = 0.5;
    options.edge_sample_step_factor = 1.0;
    options.min_edge_samples = 4;
    options.edge_length_quadrature_samples = 16;
    options.H_factor = 2.0 * kTargetP2NodeSpacingOverH;
    options.max_edge_over_H = 1.10;
    options.metric_ratio_tolerance = 1.50;
    options.max_depth = 6;
    const auto triangulation =
        geometry3d::triangulate_nurbs_surface_patches_3d(
            make_l_prism_patches(), h, options);
    const int feature_edges = triangulation.summary.num_feature_edges;
    const int feature_vertices = triangulation.summary.num_feature_vertices;
    return {"l_prism",
            "edge-trimmed P2 correction surface plus seamless L-prism crossing surface",
            triangulation.interface,
            triangulation.geometry_interface,
            feature_edges,
            feature_vertices,
            [](const Eigen::Vector3d& x) {
                constexpr double sx = 0.07;
                constexpr double sy = -0.07;
                const double px = x.x() - sx;
                const double py = x.y() - sy;
                const bool in_plan =
                    (px > -0.60 && px < 0.60 && py > -0.60 && py < 0.0)
                 || (px > -0.60 && px < 0.0 && py > 0.0 && py < 0.60);
                return in_plan && x.z() > -0.63 && x.z() < 0.67;
            }};
}

Eigen::Vector3d grid_point(const CartesianGrid3D& grid, int node)
{
    const std::array<double, 3> x = grid.coord(node);
    return {x[0], x[1], x[2]};
}

std::array<double, 4> cubic_lagrange_weights(double fraction)
{
    constexpr std::array<double, 4> nodes{{-1.0, 0.0, 1.0, 2.0}};
    std::array<double, 4> weights{{1.0, 1.0, 1.0, 1.0}};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i != j) {
                weights[static_cast<std::size_t>(i)] *=
                    (fraction - nodes[static_cast<std::size_t>(j)])
                    / (nodes[static_cast<std::size_t>(i)]
                       - nodes[static_cast<std::size_t>(j)]);
            }
        }
    }
    return weights;
}

struct HarmonicJetField3D {
    Eigen::VectorXd potential;
    Eigen::MatrixXd coefficients;
};

struct HarmonicCrossingRow3D {
    int rhs_node = -1;
    int center_dof = -1;
    double scale = 0.0;
    Eigen::VectorXd evaluation;
};

struct HarmonicTraceSample3D {
    std::array<int, 64> grid_ids{};
    std::array<double, 64> weights{};
    Eigen::VectorXd correction_evaluation;
};

class PanelCenterHarmonicJetKFBI3D {
public:
    PanelCenterHarmonicJetKFBI3D(const CartesianGrid3D& grid,
                                 const GridPair3D& grid_pair,
                                 const SurfaceDofCloud& cloud,
                                 const CauchyStencilSet& stencils)
        : grid_(grid)
        , grid_pair_(grid_pair)
        , cloud_(cloud)
        , h_(grid.spacing()[0])
        , fit_(cloud, stencils, h_, 3)
        , bulk_(grid, ZfftBcType::Dirichlet, 0.0, 2)
        , correction_support_(build_laplace_correction_support_3d(
              grid_pair, "PanelCenterHarmonicJetKFBI3D"))
    {
        const auto spacing = grid_.spacing();
        if (std::abs(spacing[0] - spacing[1]) > 1.0e-13
            || std::abs(spacing[0] - spacing[2]) > 1.0e-13) {
            throw std::invalid_argument(
                "harmonic-jet KFBI3D requires an isotropic Cartesian grid");
        }
        build_crossing_rows();
        build_trace_templates();
        build_joint_trace_fit();
    }

    int surface_size() const
    {
        return static_cast<int>(cloud_.dofs.size());
    }

    double surface_area() const
    {
        double area = 0.0;
        for (const SurfaceDof& dof : cloud_.dofs)
            area += dof.weight;
        return area;
    }

    const SurfaceDofCloud& surface() const { return cloud_; }

    HarmonicJetField3D evaluate(const Eigen::VectorXd& value_jump,
                                const Eigen::VectorXd& normal_jump) const
    {
        HarmonicJetField3D result;
        result.coefficients = fit_.coefficients(value_jump, normal_jump);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(grid_.num_dofs());
        for (const HarmonicCrossingRow3D& row : crossing_rows_) {
            rhs[row.rhs_node] += row.scale
                * row.evaluation.dot(
                    result.coefficients.row(row.center_dof).transpose());
        }
        // The spread correction is assembled for Delta_h, while the project
        // bulk solver accepts the right-hand side of -Delta_h.
        bulk_.solve(-rhs, result.potential);
        return result;
    }

    Eigen::VectorXd exterior_trace(const HarmonicJetField3D& field,
                                   const Eigen::VectorXd& value_jump,
                                   const Eigen::VectorXd& normal_jump) const
    {
        return recover_trace(
            continued_samples(field, value_jump, normal_jump, false),
            c0_weights_, 1.0);
    }

    Eigen::VectorXd interior_trace(const HarmonicJetField3D& field,
                                   const Eigen::VectorXd& value_jump,
                                   const Eigen::VectorXd& normal_jump) const
    {
        return recover_trace(
            continued_samples(field, value_jump, normal_jump, true),
            c0_weights_, 1.0);
    }

    Eigen::VectorXd exterior_normal_trace(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        return recover_trace(
            continued_samples(field, value_jump, normal_jump, false),
            c1_weights_, 1.0 / h_);
    }

    Eigen::VectorXd interior_normal_trace(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        return recover_trace(
            continued_samples(field, value_jump, normal_jump, true),
            c1_weights_, 1.0 / h_);
    }

private:
    Eigen::MatrixXd continued_samples(const HarmonicJetField3D& field,
                                      const Eigen::VectorXd& value_jump,
                                      const Eigen::VectorXd& normal_jump,
                                      bool interior_continuation) const
    {
        const int size = surface_size();
        if (field.potential.size() != grid_.num_dofs()
            || field.coefficients.rows() != size
            || value_jump.size() != size
            || normal_jump.size() != size) {
            throw std::invalid_argument(
                "exterior trace received incompatible field or jump sizes");
        }

        Eigen::MatrixXd samples = Eigen::MatrixXd::Zero(size, 8);
        for (int center = 0; center < size; ++center) {
            for (int side = 0; side < 2; ++side) {
                for (int layer = 0; layer < 4; ++layer) {
                    const HarmonicTraceSample3D& sample = trace_samples_[
                        trace_sample_index(center, side, layer)];
                    double value = 0.0;
                    for (int q = 0; q < 64; ++q) {
                        value += sample.weights[static_cast<std::size_t>(q)]
                               * field.potential[
                                   sample.grid_ids[static_cast<std::size_t>(q)]];
                    }
                    value += sample.correction_evaluation.dot(
                        field.coefficients.row(center).transpose());
                    samples(center, 4 * side + layer) = value;
                }
            }

            if (interior_continuation) {
                // At rho=+tau*h, W^- = W^+ + [W] + rho[W_n].
                for (int layer = 0; layer < 4; ++layer) {
                    samples(center, 4 + layer) += value_jump[center];
                    samples(center, 4 + layer) +=
                        normal_layers_[static_cast<std::size_t>(layer)]
                        * h_ * normal_jump[center];
                }
            } else {
                // At rho=-tau*h, W^+ = W^- - [W] - rho[W_n].
                for (int layer = 0; layer < 4; ++layer) {
                    samples(center, layer) -= value_jump[center];
                    samples(center, layer) +=
                        normal_layers_[static_cast<std::size_t>(layer)]
                        * h_ * normal_jump[center];
                }
            }
        }
        return samples;
    }

    Eigen::VectorXd recover_trace(
        const Eigen::MatrixXd& samples,
        const std::array<double, 8>& weights,
        double scale) const
    {
        Eigen::VectorXd trace(samples.rows());
        for (int center = 0; center < samples.rows(); ++center) {
            double value = 0.0;
            for (int q = 0; q < 8; ++q)
                value += weights[static_cast<std::size_t>(q)]
                       * samples(center, q);
            trace[center] = scale * value;
        }
        return trace;
    }

    std::size_t trace_sample_index(int center, int side, int layer) const
    {
        return static_cast<std::size_t>((center * 2 + side) * 4 + layer);
    }

    Eigen::Vector3d local_coordinate(int center,
                                     const Eigen::Vector3d& point) const
    {
        const SurfaceDof& dof = cloud_.dofs[static_cast<std::size_t>(center)];
        const Eigen::Vector3d displacement = (point - dof.point) / h_;
        return {displacement.dot(dof.tangent1),
                displacement.dot(dof.tangent2),
                displacement.dot(dof.normal)};
    }

    int nearest_surface_dof(const Eigen::Vector3d& point,
                            const Eigen::Vector3d* reference_normal) const
    {
        int nearest = -1;
        double best = std::numeric_limits<double>::infinity();
        for (int pass = 0; pass < 2 && nearest < 0; ++pass) {
            for (int q = 0; q < surface_size(); ++q) {
                const SurfaceDof& dof = cloud_.dofs[static_cast<std::size_t>(q)];
                if (pass == 0 && reference_normal != nullptr
                    && dof.normal.dot(*reference_normal) < 0.50) {
                    continue;
                }
                const double distance = (dof.point - point).squaredNorm();
                if (distance < best) {
                    best = distance;
                    nearest = q;
                }
            }
        }
        if (nearest < 0)
            throw std::runtime_error("failed to associate a crossing with a surface DOF");
        return nearest;
    }

    void build_crossing_rows()
    {
        crossing_rows_.reserve(correction_support_.crossing_ops.size());
        for (const LaplaceCrossingCorrectionOp& op
             : correction_support_.crossing_ops) {
            const P2CrossingOwner3D owner =
                grid_pair_.p2_crossing_owner_between(
                    op.rhs_node, op.correction_node);
            const Eigen::Vector3d a = grid_point(grid_, op.rhs_node);
            const Eigen::Vector3d b = grid_point(grid_, op.correction_node);
            const double phase = std::max(0.0, std::min(1.0, owner.edge_parameter));
            const Eigen::Vector3d hit = a + phase * (b - a);

            Eigen::Vector3d reference_normal = Eigen::Vector3d::Zero();
            const Eigen::Vector3d* normal_ptr = nullptr;
            if (owner.geometry_panel_index >= 0) {
                reference_normal = geometry3d::panel_oriented_normal(
                    grid_pair_.crossing_geometry(),
                    owner.geometry_panel_index,
                    owner.geometry_barycentric);
                normal_ptr = &reference_normal;
            } else if (owner.panel_index >= 0) {
                reference_normal = geometry3d::panel_oriented_normal(
                    grid_pair_.interface(), owner.panel_index, owner.barycentric);
                normal_ptr = &reference_normal;
            }

            const int center = nearest_surface_dof(hit, normal_ptr);
            const Eigen::Vector3d xi =
                local_coordinate(center, grid_point(grid_, op.correction_node));
            HarmonicCrossingRow3D row;
            row.rhs_node = op.rhs_node;
            row.center_dof = center;
            row.scale = static_cast<double>(op.side_delta) * op.stencil_weight;
            row.evaluation = fit_.space().basis(xi.x(), xi.y(), xi.z());
            crossing_rows_.push_back(std::move(row));
        }
    }

    HarmonicTraceSample3D build_trace_sample(int center,
                                             bool desired_inside,
                                             double signed_layer) const
    {
        const SurfaceDof& dof = cloud_.dofs[static_cast<std::size_t>(center)];
        const Eigen::Vector3d query =
            dof.point + signed_layer * h_ * dof.normal;
        const std::array<double, 3> origin = grid_.origin();
        const std::array<int, 3> dims = grid_.dof_dims();
        std::array<int, 3> lo{};
        std::array<std::array<double, 4>, 3> axis_weights{};
        for (int axis = 0; axis < 3; ++axis) {
            const double coordinate =
                (query[axis] - origin[static_cast<std::size_t>(axis)]) / h_;
            if (coordinate < 0.0
                || coordinate
                   > static_cast<double>(dims[static_cast<std::size_t>(axis)] - 1)) {
                throw std::runtime_error(
                    "normal-layer sample exits the embedding box");
            }
            const int floor_index = static_cast<int>(std::floor(coordinate));
            const int start = std::max(
                0,
                std::min(floor_index - 1,
                         dims[static_cast<std::size_t>(axis)] - 4));
            lo[static_cast<std::size_t>(axis)] = start + 1;
            const double fraction = coordinate
                - static_cast<double>(lo[static_cast<std::size_t>(axis)]);
            axis_weights[static_cast<std::size_t>(axis)] =
                cubic_lagrange_weights(fraction);
        }

        // Four Cartesian slices are first interpolated by a 4x4 bicubic
        // polynomial; a final cubic interpolation in the third direction is
        // algebraically the standard 4x4x4 tensor-product tricubic formula.
        HarmonicTraceSample3D result;
        result.correction_evaluation = Eigen::VectorXd::Zero(fit_.dimension());
        int q = 0;
        for (int iz = 0; iz < 4; ++iz) {
            for (int iy = 0; iy < 4; ++iy) {
                for (int ix = 0; ix < 4; ++ix) {
                    const int i = lo[0] - 1 + ix;
                    const int j = lo[1] - 1 + iy;
                    const int k = lo[2] - 1 + iz;
                    const int node = grid_.index(i, j, k);
                    const double weight = axis_weights[0][static_cast<std::size_t>(ix)]
                        * axis_weights[1][static_cast<std::size_t>(iy)]
                        * axis_weights[2][static_cast<std::size_t>(iz)];
                    result.grid_ids[static_cast<std::size_t>(q)] = node;
                    result.weights[static_cast<std::size_t>(q)] = weight;
                    const bool node_inside = grid_pair_.domain_label(node) > 0;
                    if (node_inside != desired_inside) {
                        const Eigen::Vector3d xi =
                            local_coordinate(center, grid_point(grid_, node));
                        const double correction_sign = desired_inside ? 1.0 : -1.0;
                        result.correction_evaluation += correction_sign * weight
                            * fit_.space().basis(xi.x(), xi.y(), xi.z());
                    }
                    ++q;
                }
            }
        }
        return result;
    }

    void build_trace_templates()
    {
        trace_samples_.reserve(
            static_cast<std::size_t>(surface_size() * 2 * 4));
        for (int center = 0; center < surface_size(); ++center) {
            for (int side = 0; side < 2; ++side) {
                const bool desired_inside = side == 0;
                const double sign = desired_inside ? -1.0 : 1.0;
                for (double layer : normal_layers_)
                    trace_samples_.push_back(
                        build_trace_sample(center, desired_inside, sign * layer));
            }
        }
    }

    void build_joint_trace_fit()
    {
        Eigen::MatrixXd design = Eigen::MatrixXd::Zero(8, 6);
        for (int q = 0; q < 4; ++q) {
            const double xm = -normal_layers_[static_cast<std::size_t>(q)];
            const double xp = normal_layers_[static_cast<std::size_t>(q)];
            design(q, 0) = 1.0;
            design(q, 1) = xm;
            design(q, 2) = xm * xm;
            design(q, 3) = xm * xm * xm;
            design(4 + q, 0) = 1.0;
            design(4 + q, 1) = xp;
            design(4 + q, 4) = xp * xp;
            design(4 + q, 5) = xp * xp * xp;
        }
        const Eigen::MatrixXd pinv = svd_pseudoinverse(design, 1.0e-13);
        for (int q = 0; q < 8; ++q) {
            c0_weights_[static_cast<std::size_t>(q)] = pinv(0, q);
            c1_weights_[static_cast<std::size_t>(q)] = pinv(1, q);
        }
    }

    const CartesianGrid3D& grid_;
    const GridPair3D& grid_pair_;
    const SurfaceDofCloud& cloud_;
    double h_ = 0.0;
    PanelCenterCauchyFit3D fit_;
    LaplaceFftBulkSolverZfft3D bulk_;
    LaplaceCorrectionSupport3D correction_support_;
    std::vector<HarmonicCrossingRow3D> crossing_rows_;
    std::vector<HarmonicTraceSample3D> trace_samples_;
    const std::array<double, 4> normal_layers_{{0.5, 1.5, 2.5, 3.5}};
    std::array<double, 8> c0_weights_{};
    std::array<double, 8> c1_weights_{};
};

class ExteriorZeroTraceOperator3D final : public IKFBIOperator {
public:
    explicit ExteriorZeroTraceOperator3D(
        const PanelCenterHarmonicJetKFBI3D& pipeline)
        : pipeline_(pipeline)
    {}

    int problem_size() const override
    {
        return pipeline_.surface_size() + 1;
    }

    void apply(const Eigen::VectorXd& unknown,
               Eigen::VectorXd& result) const override
    {
        if (unknown.size() != problem_size())
            throw std::invalid_argument("exterior-zero operator input has wrong size");
        const int size = pipeline_.surface_size();
        const Eigen::VectorXd value_jump = unknown.head(size);
        const Eigen::VectorXd zero_normal = Eigen::VectorXd::Zero(size);
        const HarmonicJetField3D field =
            pipeline_.evaluate(value_jump, zero_normal);
        const Eigen::VectorXd trace =
            pipeline_.exterior_trace(field, value_jump, zero_normal);
        result.resize(size + 1);
        result.head(size) = trace.array() + unknown[size];
        double weighted_mean = 0.0;
        for (int q = 0; q < size; ++q) {
            weighted_mean += pipeline_.surface().dofs[static_cast<std::size_t>(q)].weight
                           * value_jump[q];
        }
        result[size] = weighted_mean / pipeline_.surface_area();
    }

    Eigen::VectorXd right_hand_side(
        const Eigen::VectorXd& prescribed_normal_jump) const
    {
        const int size = pipeline_.surface_size();
        if (prescribed_normal_jump.size() != size)
            throw std::invalid_argument("prescribed Neumann data has wrong size");
        const Eigen::VectorXd zero_value = Eigen::VectorXd::Zero(size);
        const HarmonicJetField3D field =
            pipeline_.evaluate(zero_value, prescribed_normal_jump);
        Eigen::VectorXd result = Eigen::VectorXd::Zero(size + 1);
        result.head(size) = -pipeline_.exterior_trace(
            field, zero_value, prescribed_normal_jump);
        return result;
    }

private:
    const PanelCenterHarmonicJetKFBI3D& pipeline_;
};

struct ExteriorZeroTraceSolution3D {
    Eigen::VectorXd value_jump;
    Eigen::VectorXd potential;
    Eigen::VectorXd augmented_residual;
    std::vector<double> gmres_residuals;
    double lagrange_multiplier = 0.0;
    int iterations = 0;
    bool converged = false;
};

ExteriorZeroTraceSolution3D solve_exterior_zero_trace_neumann_3d(
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const Eigen::VectorXd& prescribed_normal_jump,
    double tolerance,
    int restart,
    int max_iterations)
{
    ExteriorZeroTraceOperator3D op(pipeline);
    const Eigen::VectorXd rhs = op.right_hand_side(prescribed_normal_jump);
    Eigen::VectorXd augmented_unknown = Eigen::VectorXd::Zero(op.problem_size());
    GMRES gmres(max_iterations, tolerance, restart);
    ExteriorZeroTraceSolution3D result;
    result.iterations = gmres.solve(op, rhs, augmented_unknown);
    result.converged = gmres.converged();
    result.gmres_residuals = gmres.residuals();
    const int size = pipeline.surface_size();
    result.value_jump = augmented_unknown.head(size);
    result.lagrange_multiplier = augmented_unknown[size];
    const HarmonicJetField3D field =
        pipeline.evaluate(result.value_jump, prescribed_normal_jump);
    result.potential = field.potential;
    Eigen::VectorXd applied;
    op.apply(augmented_unknown, applied);
    result.augmented_residual = applied - rhs;
    return result;
}

double validate_interface(const Interface3D& iface)
{
    if (iface.panel_node_layout() != PanelNodeLayout3D::QuadraticLagrange
        || iface.points_per_panel() != 6
        || iface.num_panels() <= 0
        || iface.num_points() <= 0
        || iface.weights().sum() <= 0.0) {
        throw std::runtime_error("invalid P2 surface interface");
    }
    double normal_error = 0.0;
    for (int q = 0; q < iface.num_points(); ++q) {
        const double norm = iface.normals().row(q).norm();
        if (!std::isfinite(norm))
            throw std::runtime_error("surface contains a non-finite normal");
        normal_error = std::max(normal_error, std::abs(norm - 1.0));
    }
    if (normal_error > 1.0e-9)
        throw std::runtime_error("surface normals are not unit length");
    return normal_error;
}

std::ofstream open_output_file(const std::filesystem::path& path)
{
    std::ofstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot open output file: " + path.string());
    return stream;
}

void write_surface_files(const std::filesystem::path& output_dir,
                         const GeometryBundle& geometry,
                         int N)
{
    std::filesystem::create_directories(output_dir);
    const std::string stem = geometry.name + "_N" + std::to_string(N);
    const Interface3D& surface = geometry.crossing_interface;
    std::ofstream obj = open_output_file(
        output_dir / (stem + "_crossing_surface.obj"));
    obj << std::setprecision(17);
    obj << "# " << geometry.description << '\n';
    for (int v = 0; v < surface.num_vertices(); ++v)
        obj << "v " << surface.vertices()(v, 0) << ' '
            << surface.vertices()(v, 1) << ' '
            << surface.vertices()(v, 2) << '\n';
    for (int p = 0; p < surface.num_panels(); ++p)
        obj << "f " << surface.panels()(p, 0) + 1 << ' '
            << surface.panels()(p, 1) + 1 << ' '
            << surface.panels()(p, 2) + 1 << '\n';

    const Interface3D& iface = geometry.correction_interface;
    std::ofstream csv = open_output_file(
        output_dir / (stem + "_correction_dofs.csv"));
    csv << std::setprecision(17);
    csv << "q,x,y,z,nx,ny,nz,weight\n";
    for (int q = 0; q < iface.num_points(); ++q)
        csv << q << ','
            << iface.points()(q, 0) << ',' << iface.points()(q, 1) << ','
            << iface.points()(q, 2) << ',' << iface.normals()(q, 0) << ','
            << iface.normals()(q, 1) << ',' << iface.normals()(q, 2) << ','
            << iface.weights()[q] << '\n';
}

void write_panel_center_files(const std::filesystem::path& output_dir,
                              const std::string& geometry_name,
                              int N,
                              const SurfaceDofCloud& cloud,
                              const CauchyStencilSet& stencils)
{
    std::filesystem::create_directories(output_dir);
    const std::string stem = geometry_name + "_N" + std::to_string(N);

    std::ofstream patch_csv = open_output_file(
        output_dir / (stem + "_surface_patches.csv"));
    patch_csv << "patch_id,patch_name,adjacent_patch_ids\n";
    for (int patch_id = 0;
         patch_id < static_cast<int>(cloud.patches.size()); ++patch_id) {
        const SurfacePatchInfo& patch =
            cloud.patches[static_cast<std::size_t>(patch_id)];
        patch_csv << patch_id << ',' << patch.name << ',';
        for (std::size_t j = 0; j < patch.adjacent_patch_ids.size(); ++j) {
            if (j != 0)
                patch_csv << ';';
            patch_csv << patch.adjacent_patch_ids[j];
        }
        patch_csv << '\n';
    }

    std::ofstream dof_csv = open_output_file(
        output_dir / (stem + "_surface_panel_center_dofs.csv"));
    dof_csv << std::setprecision(17);
    dof_csv << "q,patch_id,patch_name,parameter_a,parameter_b,"
               "x,y,z,nx,ny,nz,t1x,t1y,t1z,t2x,t2y,t2z,weight\n";
    for (int q = 0; q < static_cast<int>(cloud.dofs.size()); ++q) {
        const SurfaceDof& dof = cloud.dofs[static_cast<std::size_t>(q)];
        dof_csv << q << ',' << dof.patch_id << ','
                << cloud.patches[static_cast<std::size_t>(dof.patch_id)].name
                << ',' << dof.parameter_a << ',' << dof.parameter_b << ','
                << dof.point.x() << ',' << dof.point.y() << ',' << dof.point.z() << ','
                << dof.normal.x() << ',' << dof.normal.y() << ',' << dof.normal.z() << ','
                << dof.tangent1.x() << ',' << dof.tangent1.y() << ','
                << dof.tangent1.z() << ',' << dof.tangent2.x() << ','
                << dof.tangent2.y() << ',' << dof.tangent2.z() << ','
                << dof.weight << '\n';
    }

    std::ofstream stencil_csv = open_output_file(
        output_dir / (stem + "_cauchy_stencils.csv"));
    stencil_csv << std::setprecision(17);
    stencil_csv << "q,patch_id,radius_over_h,incident_patch_count";
    for (int k = 0; k < stencils.value_count; ++k)
        stencil_csv << ",value_" << k;
    for (int k = 0; k < stencils.derivative_count; ++k)
        stencil_csv << ",normal_" << k;
    stencil_csv << '\n';
    for (int q = 0; q < static_cast<int>(stencils.rows.size()); ++q) {
        const CauchyStencil& stencil =
            stencils.rows[static_cast<std::size_t>(q)];
        stencil_csv << q << ','
                    << cloud.dofs[static_cast<std::size_t>(q)].patch_id << ','
                    << stencil.radius_over_h << ','
                    << stencil.incident_patch_count;
        for (int id : stencil.value_ids)
            stencil_csv << ',' << id;
        for (int id : stencil.derivative_ids)
            stencil_csv << ',' << id;
        stencil_csv << '\n';
    }
}

ReadinessResult run_readiness_case(GeometryKind kind,
                                   int N,
                                   const std::filesystem::path& output_dir)
{
    const double h = kBoxSide / static_cast<double>(N);
    CartesianGrid3D grid({kBoxMin, kBoxMin, kBoxMin},
                         {h, h, h},
                         {N, N, N},
                         DofLayout3D::Node);
    GeometryBundle geometry = make_geometry(kind, h);
    const SurfaceDofCloud surface_dofs = make_surface_dofs(kind, h);
    const SurfaceCloudDiagnostics surface_diagnostics =
        validate_surface_dofs(surface_dofs, h);
    const CauchyStencilSet cauchy_stencils = build_cauchy_stencils(
        surface_dofs,
        h,
        kCauchyValueNeighborCount,
        kCauchyDerivativeNeighborCount);
    write_surface_files(output_dir, geometry, N);
    write_panel_center_files(
        output_dir, geometry.name, N, surface_dofs, cauchy_stencils);

    ReadinessResult result;
    result.geometry = geometry.name;
    result.N = N;
    result.h = h;
    result.correction_panels = geometry.correction_interface.num_panels();
    result.correction_dofs = geometry.correction_interface.num_points();
    result.crossing_panels = geometry.crossing_interface.num_panels();
    result.feature_edges = geometry.feature_edges;
    result.feature_vertices = geometry.feature_vertices;
    result.correction_area = geometry.correction_interface.weights().sum();
    result.crossing_area = geometry.crossing_interface.weights().sum();
    result.normal_error = validate_interface(geometry.correction_interface);
    validate_interface(geometry.crossing_interface);
    result.surface_patches = static_cast<int>(surface_dofs.patches.size());
    result.surface_dofs = static_cast<int>(surface_dofs.dofs.size());
    result.surface_dof_area = surface_diagnostics.area;
    result.surface_area_relative_error = surface_diagnostics.area_relative_error;
    result.surface_spacing_min_over_h =
        surface_diagnostics.nearest_spacing_min_over_h;
    result.surface_spacing_mean_over_h =
        surface_diagnostics.nearest_spacing_mean_over_h;
    result.surface_spacing_max_over_h =
        surface_diagnostics.nearest_spacing_max_over_h;
    result.cauchy_value_neighbors = cauchy_stencils.value_count;
    result.cauchy_derivative_neighbors = cauchy_stencils.derivative_count;
    result.cauchy_radius_max_over_h = cauchy_stencils.radius_max_over_h;
    result.cauchy_radius_mean_over_h = cauchy_stencils.radius_mean_over_h;
    result.cauchy_incident_patches_min =
        cauchy_stencils.incident_patch_count_min;
    result.cauchy_incident_patches_max =
        cauchy_stencils.incident_patch_count_max;

    const auto dims = grid.dof_dims();
    Eigen::Vector3d grid_min(kBoxMin, kBoxMin, kBoxMin);
    Eigen::Vector3d grid_max(
        kBoxMin + static_cast<double>(dims[0] - 1) * h,
        kBoxMin + static_cast<double>(dims[1] - 1) * h,
        kBoxMin + static_cast<double>(dims[2] - 1) * h);
    double min_margin = std::numeric_limits<double>::infinity();
    for (int q = 0; q < geometry.crossing_interface.num_points(); ++q) {
        const Eigen::Vector3d point =
            geometry.crossing_interface.points().row(q).transpose();
        min_margin = std::min(min_margin, (point - grid_min).minCoeff());
        min_margin = std::min(min_margin, (grid_max - point).minCoeff());
    }
    result.min_box_margin_over_h = min_margin / h;
    if (result.min_box_margin_over_h < 2.0)
        throw std::runtime_error("surface is too close to the Cartesian box boundary");

    GridPair3D grid_pair(grid,
                         geometry.correction_interface,
                         geometry.crossing_interface);
    for (int n = 0; n < grid.num_dofs(); ++n) {
        const bool numerical_inside = grid_pair.domain_label(n) > 0;
        const auto coordinate = grid.coord(n);
        const Eigen::Vector3d point(
            coordinate[0], coordinate[1], coordinate[2]);
        const bool exact_inside = geometry.exact_inside(point);
        if (numerical_inside)
            ++result.interior_nodes;
        else
            ++result.exterior_nodes;
        if (numerical_inside != exact_inside) {
            ++result.label_mismatches;
            if (result.label_mismatches <= 5) {
                std::cout << "  [label mismatch] x=("
                          << point.x() << ',' << point.y() << ',' << point.z()
                          << ") numerical=" << numerical_inside
                          << " exact=" << exact_inside << '\n';
            }
        }
    }
    if (result.interior_nodes == 0 || result.exterior_nodes == 0)
        throw std::runtime_error("domain labeling did not produce both sides");
    if (result.label_mismatches > std::max(2, grid.num_dofs() / 100))
        throw std::runtime_error("more than one percent of grid labels are incorrect");

    const LaplaceCorrectionSupport3D support =
        build_laplace_correction_support_3d(
            grid_pair, "neumann exterior-zero-trace 3D geometry readiness");
    result.crossings = static_cast<int>(support.crossing_ops.size());
    result.correction_nodes = static_cast<int>(support.correction_nodes.size());
    if (result.crossings == 0
        || support.restrict_stencils.size()
           != static_cast<std::size_t>(geometry.correction_interface.num_points())) {
        throw std::runtime_error("failed to build fixed crossing/restrict routes");
    }

    std::set<std::pair<int, int>> crossing_edges;
    for (const LaplaceCrossingCorrectionOp& op : support.crossing_ops) {
        const std::pair<int, int> edge{
            std::min(op.rhs_node, op.correction_node),
            std::max(op.rhs_node, op.correction_node)};
        if (!crossing_edges.insert(edge).second)
            continue;
        const P2CrossingOwner3D owner =
            grid_pair.p2_crossing_owner_between(edge.first, edge.second);
        if (owner.status == P2CrossingOwnerStatus3D::ExactIntersection)
            ++result.exact_crossings;
        else if (owner.status == P2CrossingOwnerStatus3D::GapFallback)
            ++result.gap_crossings;
        else
            ++result.endpoint_crossings;
    }

    LaplaceQuadraticPatchCenterSpread3D spread(grid_pair, 0.0);
    LaplaceFftBulkSolverZfft3D bulk(
        grid, ZfftBcType::Dirichlet, 0.0, 2);
    LaplaceQuadraticPatchCenterRestrict3D restrict_op(grid_pair, 2);
    LaplacePotentialEval3D potentials(spread, bulk, restrict_op);

    std::vector<LaplaceJumpData3D> jumps(
        static_cast<std::size_t>(geometry.correction_interface.num_points()));
    for (LaplaceJumpData3D& jump : jumps) {
        jump.u_jump = 1.0;
        jump.un_jump = 0.0;
        jump.rhs_derivs = Eigen::VectorXd::Zero(1);
    }
    const LaplacePotentialEvalResult3D probe = potentials.evaluate(
        jumps, Eigen::VectorXd::Zero(grid.num_dofs()));
    for (int q = 0; q < probe.u_avg.size(); ++q) {
        const double interior_trace = probe.u_avg[q] + 0.5;
        const double exterior_trace = probe.u_avg[q] - 0.5;
        result.constant_A1_linf = std::max(
            result.constant_A1_linf, std::abs(1.0 - interior_trace));
        result.constant_exterior_trace_linf = std::max(
            result.constant_exterior_trace_linf, std::abs(exterior_trace));
        result.constant_interior_trace_linf = std::max(
            result.constant_interior_trace_linf, std::abs(interior_trace - 1.0));
    }
    for (int n = 0; n < grid.num_dofs(); ++n) {
        const double expected = grid_pair.domain_label(n) > 0 ? 1.0 : 0.0;
        result.constant_bulk_linf = std::max(
            result.constant_bulk_linf, std::abs(probe.u_bulk[n] - expected));
    }
    if (!std::isfinite(result.constant_A1_linf)
        || !std::isfinite(result.constant_bulk_linf)) {
        throw std::runtime_error("constant value-jump readiness probe produced NaN/Inf");
    }

    PanelCenterHarmonicJetKFBI3D harmonic_pipeline(
        grid, grid_pair, surface_dofs, cauchy_stencils);
    const Eigen::VectorXd constant_value =
        Eigen::VectorXd::Ones(harmonic_pipeline.surface_size());
    const Eigen::VectorXd zero_normal =
        Eigen::VectorXd::Zero(harmonic_pipeline.surface_size());
    const HarmonicJetField3D harmonic_probe =
        harmonic_pipeline.evaluate(constant_value, zero_normal);
    result.harmonic_constant_exterior_trace_linf =
        harmonic_pipeline.exterior_trace(
            harmonic_probe, constant_value, zero_normal).lpNorm<Eigen::Infinity>();
    result.harmonic_constant_interior_trace_linf =
        (harmonic_pipeline.interior_trace(
             harmonic_probe, constant_value, zero_normal).array() - 1.0)
        .matrix().lpNorm<Eigen::Infinity>();
    result.harmonic_constant_exterior_normal_linf =
        harmonic_pipeline.exterior_normal_trace(
            harmonic_probe, constant_value, zero_normal).lpNorm<Eigen::Infinity>();
    result.harmonic_constant_interior_normal_linf =
        harmonic_pipeline.interior_normal_trace(
            harmonic_probe, constant_value, zero_normal).lpNorm<Eigen::Infinity>();
    for (int n = 0; n < grid.num_dofs(); ++n) {
        const double expected = grid_pair.domain_label(n) > 0 ? 1.0 : 0.0;
        result.harmonic_constant_bulk_linf = std::max(
            result.harmonic_constant_bulk_linf,
            std::abs(harmonic_probe.potential[n] - expected));
    }
    if (!std::isfinite(result.harmonic_constant_exterior_trace_linf)
        || !std::isfinite(result.harmonic_constant_exterior_normal_linf)
        || !std::isfinite(result.harmonic_constant_bulk_linf)) {
        throw std::runtime_error(
            "harmonic-jet constant probe produced NaN/Inf");
    }

    std::cout << "[ready] " << geometry.name << " - " << geometry.description << '\n'
              << "  panel-center surface patches/dofs="
              << result.surface_patches << '/' << result.surface_dofs
              << " area=" << result.surface_dof_area
              << " area_rel_err=" << result.surface_area_relative_error << '\n'
              << "  nearest surface spacing/h min/mean/max="
              << result.surface_spacing_min_over_h << '/'
              << result.surface_spacing_mean_over_h << '/'
              << result.surface_spacing_max_over_h << '\n'
              << "  Cauchy values/normals="
              << result.cauchy_value_neighbors << '/'
              << result.cauchy_derivative_neighbors
              << " radius/h mean/max=" << result.cauchy_radius_mean_over_h
              << '/' << result.cauchy_radius_max_over_h
              << " selected patches min/max="
              << result.cauchy_incident_patches_min << '/'
              << result.cauchy_incident_patches_max << '\n'
              << "  P2 correction panels/dofs=" << result.correction_panels
              << '/' << result.correction_dofs
              << " crossing_panels=" << result.crossing_panels
              << " area(correction/full)=" << result.correction_area
              << '/' << result.crossing_area << '\n'
              << "  features edges/vertices=" << result.feature_edges
              << '/' << result.feature_vertices
              << " grid inside/outside=" << result.interior_nodes
              << '/' << result.exterior_nodes
              << " label_mismatches=" << result.label_mismatches
              << " box_margin/h=" << result.min_box_margin_over_h << '\n'
              << "  fixed routes crossing_ops/correction_nodes="
              << result.crossings << '/' << result.correction_nodes
              << " owners exact/gap/endpoint=" << result.exact_crossings
              << '/' << result.gap_crossings
              << '/' << result.endpoint_crossings << '\n'
              << "  constant [u]=1 probe: |A1|inf="
              << result.constant_A1_linf
              << " exterior_trace=" << result.constant_exterior_trace_linf
              << " interior_trace_err=" << result.constant_interior_trace_linf
              << " bulk_err=" << result.constant_bulk_linf << '\n'
              << "  cubic harmonic-jet [u]=1 probe: exterior/interior trace="
              << result.harmonic_constant_exterior_trace_linf << '/'
              << result.harmonic_constant_interior_trace_linf
              << " exterior/interior normal="
              << result.harmonic_constant_exterior_normal_linf << '/'
              << result.harmonic_constant_interior_normal_linf
              << " bulk_err=" << result.harmonic_constant_bulk_linf << '\n';
    return result;
}

void write_summary(const std::filesystem::path& output_dir,
                   const std::vector<ReadinessResult>& results)
{
    std::filesystem::create_directories(output_dir);
    if (results.empty())
        return;
    auto write_csv = [&](const std::filesystem::path& path,
                         const std::vector<ReadinessResult>& rows) {
        std::ofstream csv = open_output_file(path);
        csv << std::setprecision(17);
        csv << "geometry,N,h,correction_panels,correction_dofs,crossing_panels,"
               "feature_edges,feature_vertices,interior_nodes,exterior_nodes,"
               "label_mismatches,crossing_ops,exact_crossings,gap_crossings,"
               "endpoint_crossings,correction_nodes,correction_area,crossing_area,"
               "normal_error,box_margin_over_h,A1_linf,exterior_trace_linf,"
               "interior_trace_error,bulk_error,surface_patches,surface_dofs,"
               "harmonic_exterior_trace_linf,harmonic_interior_trace_linf,"
               "harmonic_exterior_normal_linf,harmonic_interior_normal_linf,"
               "harmonic_bulk_linf,"
               "surface_dof_area,surface_area_relative_error,"
               "surface_spacing_min_over_h,surface_spacing_mean_over_h,"
               "surface_spacing_max_over_h,cauchy_value_neighbors,"
               "cauchy_derivative_neighbors,cauchy_radius_max_over_h,"
               "cauchy_radius_mean_over_h,cauchy_incident_patches_min,"
               "cauchy_incident_patches_max\n";
        for (const ReadinessResult& row : rows)
            csv << row.geometry << ',' << row.N << ',' << row.h << ','
                << row.correction_panels << ',' << row.correction_dofs << ','
                << row.crossing_panels << ',' << row.feature_edges << ','
                << row.feature_vertices << ',' << row.interior_nodes << ','
                << row.exterior_nodes << ',' << row.label_mismatches << ','
                << row.crossings << ',' << row.exact_crossings << ','
                << row.gap_crossings << ',' << row.endpoint_crossings << ','
                << row.correction_nodes << ',' << row.correction_area << ','
                << row.crossing_area << ',' << row.normal_error << ','
                << row.min_box_margin_over_h << ',' << row.constant_A1_linf << ','
                << row.constant_exterior_trace_linf << ','
                << row.constant_interior_trace_linf << ','
                << row.constant_bulk_linf << ',' << row.surface_patches << ','
                << row.surface_dofs << ','
                << row.harmonic_constant_exterior_trace_linf << ','
                << row.harmonic_constant_interior_trace_linf << ','
                << row.harmonic_constant_exterior_normal_linf << ','
                << row.harmonic_constant_interior_normal_linf << ','
                << row.harmonic_constant_bulk_linf << ','
                << row.surface_dof_area << ','
                << row.surface_area_relative_error << ','
                << row.surface_spacing_min_over_h << ','
                << row.surface_spacing_mean_over_h << ','
                << row.surface_spacing_max_over_h << ','
                << row.cauchy_value_neighbors << ','
                << row.cauchy_derivative_neighbors << ','
                << row.cauchy_radius_max_over_h << ','
                << row.cauchy_radius_mean_over_h << ','
                << row.cauchy_incident_patches_min << ','
                << row.cauchy_incident_patches_max << '\n';
    };
    write_csv(output_dir / "geometry_readiness.csv", results);
    std::set<int> levels;
    for (const ReadinessResult& row : results)
        levels.insert(row.N);
    for (int level : levels) {
        std::vector<ReadinessResult> level_rows;
        for (const ReadinessResult& row : results) {
            if (row.N == level)
                level_rows.push_back(row);
        }
        write_csv(output_dir / (
            "geometry_readiness_N" + std::to_string(level) + ".csv"),
            level_rows);
    }
}

void print_usage(const char* executable)
{
    std::cout
        << "usage: " << executable
        << " [torus|cylinder|l_prism|all] [N ...]\n"
        << "  Each N must be a power of two and at least 16 (default: 16).\n"
        << "  This stage builds independent parameter-panel-center surface\n"
        << "  unknowns, topology-filtered 48/28 Cauchy stencils, validates\n"
        << "  fixed transfer routes, and runs a constant value-jump probe.\n"
        << "  The exterior-zero-trace harmonic-jet GMRES path is compiled,\n"
        << "  but this readiness command does not execute a numerical solve.\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        std::cout << std::scientific << std::setprecision(6) << std::unitbuf;
        std::string selection = "all";
        std::vector<int> levels = {16};
        if (argc >= 2)
            selection = argv[1];
        if (selection == "--help" || selection == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (argc >= 3) {
            levels.clear();
            for (int argument = 2; argument < argc; ++argument) {
                const int N = std::stoi(argv[argument]);
                if (N < 16 || !is_power_of_two(N)) {
                    throw std::invalid_argument(
                        "each N must be a power of two and at least 16");
                }
                levels.push_back(N);
            }
        }

        std::vector<GeometryKind> geometries;
        if (selection == "all") {
            geometries = {GeometryKind::Torus,
                          GeometryKind::HollowCylinder,
                          GeometryKind::LPrism};
        } else {
            geometries = {parse_geometry(selection)};
        }

#ifdef KFBIM_APP_OUTPUT_DIR
        const std::filesystem::path output_dir =
            std::filesystem::path(KFBIM_APP_OUTPUT_DIR)
            / "neumann_exterior_zero_trace_3d";
#else
        const std::filesystem::path output_dir =
            "output/neumann_exterior_zero_trace_3d";
#endif

        std::cout << "KFBI3D exterior-zero-trace Neumann readiness stage\n"
                  << "  implemented equation: A(f)=f-R_minus H_D(f), "
                     "b=R_minus H_N(g)\n"
                  << "  current stage: parameter-panel-center surface DOFs + "
                     "topological Cauchy stencils + fixed routes\n"
                  << "  levels=";
        for (std::size_t index = 0; index < levels.size(); ++index) {
            if (index != 0)
                std::cout << ',';
            std::cout << levels[index];
        }
        std::cout << '\n';

        std::vector<ReadinessResult> results;
        for (int N : levels) {
            for (GeometryKind geometry : geometries)
                results.push_back(run_readiness_case(geometry, N, output_dir));
        }
        write_summary(output_dir, results);

        std::cout << "Geometry and transfer-pipeline readiness passed.\n"
                  << "The compiled exterior-zero-trace GMRES algorithm was "
                     "not executed in readiness mode.\n"
                  << "Output: " << output_dir.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
