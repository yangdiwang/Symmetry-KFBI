#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#ifdef interface
#undef interface
#endif
#else
#include <sys/resource.h>
#endif

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "crossing_owner_restrict_3d.hpp"
#include "dirichlet_rigid_transform_study_3d.hpp"
#include "exterior_only_cubic_normal_restrict_3d.hpp"
#include "harmonic_cauchy_fit_3d.hpp"
#include "harmonic_trace_correction_3d.hpp"
#include "harmonic_polynomial_space_3d.hpp"
#include "kfbi_phase_profile_3d.hpp"
#include "native_nurbs_surface_3d.hpp"
#include "neumann_edge_cauchy_study_3d.hpp"
#include "neumann_edge_continuity_3d.hpp"
#include "neumann_rigid_transform_study_3d.hpp"
#include "restrict_owner_geometry_preprocessor_3d.hpp"
#include "src/bulk_solvers/laplace_zfft_bulk_solver_3d.hpp"
#include "src/geometry/grid_pair_3d.hpp"
#include "src/geometry/nurbs_bezier_intersection_3d.hpp"
#include "src/geometry/nurbs_cartesian_domain_3d.hpp"
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

constexpr double kBoxMin = -1.5;
constexpr double kBoxSide = 3.0;
constexpr double kTargetP2NodeSpacingOverH = 1.2;
constexpr int kCauchyValueNeighborCount = 48;
constexpr int kCauchyDerivativeNeighborCount = 28;
constexpr int kCauchyPolynomialDegree = 3;
constexpr int kRestrictGridDegree = 3;
constexpr int kRestrictNormalDegree = 3;

using GeometryKind = app3d::GeometryKind3D;
using NativeNurbsSurface3D = app3d::NativeNurbsSurface3D;
using SurfaceDof = app3d::SurfaceDof3D;
using SurfaceDofCloud = app3d::SurfaceDofCloud3D;
using SurfacePatchInfo = app3d::SurfaceDofPatch3D;
using PhaseProfileKind3D = app3d::PhaseProfileKind3D;

enum class SolveSelection3D {
    Both,
    DirichletNormalOnly
};

enum class ExteriorValueRestrictMode3D {
    JointTricubicCauchy,
    JointTricubicCrossingOwner
};

enum class ExteriorNormalRestrictMode3D {
    JointTricubicCauchy,
    JointTricubicCrossingOwner,
    ExteriorOnlyHarmonicCubic
};

using ProfileClock3D = std::chrono::steady_clock;

template <class Function>
auto profile_phase_3d(app3d::PhaseProfile3D* profile,
                      PhaseProfileKind3D kind,
                      std::uint64_t calls,
                      Function&& function)
    -> decltype(function())
{
    if (profile == nullptr)
        return function();

    profile->note_timer_reads(1);
    const ProfileClock3D::time_point start = ProfileClock3D::now();
    const auto finish = [&] {
        profile->note_timer_reads(1);
        profile->add(
            kind,
            std::chrono::duration<double>(
                ProfileClock3D::now() - start).count(),
            calls);
    };
    try {
        using Result = decltype(function());
        if constexpr (std::is_void<Result>::value) {
            function();
            finish();
        } else {
            Result result = function();
            finish();
            return result;
        }
    } catch (...) {
        finish();
        throw;
    }
}

ProfileClock3D::time_point profile_timer_start_3d(
    app3d::PhaseProfile3D* profile)
{
    if (profile == nullptr)
        return {};
    profile->note_timer_reads(1);
    return ProfileClock3D::now();
}

double profile_timer_elapsed_3d(
    app3d::PhaseProfile3D* profile,
    ProfileClock3D::time_point start)
{
    if (profile == nullptr)
        return 0.0;
    profile->note_timer_reads(1);
    return std::chrono::duration<double>(
        ProfileClock3D::now() - start).count();
}

void profile_add_elapsed_3d(
    app3d::PhaseProfile3D* profile,
    PhaseProfileKind3D kind,
    ProfileClock3D::time_point start,
    std::uint64_t calls = 1)
{
    if (profile != nullptr) {
        profile->add(
            kind, profile_timer_elapsed_3d(profile, start), calls);
    }
}

double nonnegative_profile_remainder_3d(
    double parent_seconds,
    double child_seconds,
    const char* context)
{
    const double remainder = parent_seconds - child_seconds;
    const double tolerance =
        std::max(1.0e-9, 1.0e-8 * parent_seconds);
    if (remainder < -tolerance)
        throw std::logic_error(std::string(context) + " child timers overlap");
    return std::max(0.0, remainder);
}

const char* harmonic_cauchy_route_name_3d(
    app3d::HarmonicCauchyRoute3D route)
{
    switch (route) {
    case app3d::HarmonicCauchyRoute3D::G1ValueG1Normal:
        return "g1_value_g1_normal";
    case app3d::HarmonicCauchyRoute3D::DirectCrossFaceValue:
        return "direct_cross_face_value";
    case app3d::HarmonicCauchyRoute3D::EdgeReconstructedValue:
        return "edge_reconstructed_value";
    }
    throw std::runtime_error("unknown 3D harmonic Cauchy route");
}

std::string cauchy_policy_name(app3d::LegacySurfaceCauchyPolicy3D policy)
{
    switch (policy) {
    case app3d::LegacySurfaceCauchyPolicy3D::G1Nearest:
        return "g1_nearest";
    case app3d::LegacySurfaceCauchyPolicy3D::TopologicalNearest:
        return "topological_nearest";
    case app3d::LegacySurfaceCauchyPolicy3D::SamePatch:
        return "same_patch";
    case app3d::LegacySurfaceCauchyPolicy3D::BalancedPatches:
        return "balanced_patches";
    }
    throw std::runtime_error("unknown 3D Cauchy stencil policy");
}

app3d::LegacySurfaceCauchyPolicy3D selected_cauchy_policy()
{
    const char* raw = std::getenv("KFBIM_3D_CAUCHY_POLICY");
    if (raw == nullptr || std::string(raw).empty()
        || std::string(raw) == "g1_nearest") {
        return app3d::LegacySurfaceCauchyPolicy3D::G1Nearest;
    }
    if (std::string(raw) == "same_patch")
        return app3d::LegacySurfaceCauchyPolicy3D::SamePatch;
    if (std::string(raw) == "topological_nearest")
        return app3d::LegacySurfaceCauchyPolicy3D::TopologicalNearest;
    if (std::string(raw) == "balanced_patches")
        return app3d::LegacySurfaceCauchyPolicy3D::BalancedPatches;
    throw std::invalid_argument(
        "KFBIM_3D_CAUCHY_POLICY must be g1_nearest, same_patch, "
        "topological_nearest, or balanced_patches");
}

app3d::RestrictOwnerPreprocessMode3D
selected_restrict_owner_preprocess_mode()
{
    const char* raw = std::getenv("KFBIM_3D_RESTRICT_OWNER_MODE");
    if (raw == nullptr) {
        return app3d::RestrictOwnerPreprocessMode3D::
            FullIntersectionReference;
    }
    return app3d::parse_restrict_owner_preprocess_mode_3d(raw);
}

bool is_power_of_two(int value);

int parse_grid_level_argument(const char* raw)
{
    const std::string text(raw);
    std::size_t consumed = 0;
    int value = 0;
    try {
        value = std::stoi(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(
            "each N must be a power of two and at least 16");
    }
    if (consumed != text.size()
        || value < 16
        || !is_power_of_two(value)) {
        throw std::invalid_argument(
            "each N must be a power of two and at least 16");
    }
    return value;
}

int positive_environment_integer(const char* name, int default_value)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr)
        return default_value;

    const std::string text(raw);
    std::size_t consumed = 0;
    int value = 0;
    try {
        value = std::stoi(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name)
                                    + " must be a positive integer");
    }
    if (consumed != text.size() || value <= 0) {
        throw std::invalid_argument(std::string(name)
                                    + " must be a positive integer");
    }
    return value;
}

struct GeometryBundle {
    std::string name;
    std::string description;
    NativeNurbsSurface3D native_surface;
    Interface3D correction_interface;
    Interface3D crossing_interface;
    std::vector<geometry3d::NurbsParamTriangle3D> correction_triangles;
    std::vector<geometry3d::NurbsParamTriangle3D> geometry_triangles;
    std::vector<geometry3d::NurbsPatchGeometricEdge3D>
        feature_edge_segments;
    int feature_edges = 0;
    int feature_vertices = 0;
    std::function<bool(const Eigen::Vector3d&)> exact_inside;
};

struct SurfaceCloudDiagnostics {
    double area = 0.0;
    double area_relative_error = 0.0;
    double normal_error = 0.0;
    double nearest_spacing_min_over_h = 0.0;
    double nearest_spacing_mean_over_h = 0.0;
    double nearest_spacing_max_over_h = 0.0;
};

struct SolveMetrics3D {
    std::string formulation;
    int iterations = 0;
    bool converged = false;
    double seconds = 0.0;
    double gmres_relative_residual = 0.0;
    double operator_residual_linf = 0.0;
    double exterior_condition_linf = 0.0;
    double boundary_residual_linf = 0.0;
    double route_mismatch_linf = 0.0;
    double density_linf = 0.0;
    double density_l2 = 0.0;
    double interior_linf = 0.0;
    double interior_l2 = 0.0;
    double exterior_bulk_linf = 0.0;
    double exterior_bulk_l2 = 0.0;
    double data_weighted_mean = 0.0;
    double density_weighted_mean = 0.0;
    double constant_shift = 0.0;
};

struct ReadinessResult {
    std::string geometry;
    std::string cauchy_policy;
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
    int nurbs_patches = 0;
    int bezier_elements = 0;
    int acceleration_leaves = 0;
    double maximum_query_element_extent = 0.0;
    std::size_t candidate_grid_edges = 0;
    int triangle_seed_hits = 0;
    int triangle_seed_misses_recovered = 0;
    int subdivision_boxes = 0;
    int newton_attempts = 0;
    int newton_iterations = 0;
    int maximum_subdivision_depth = 0;
    int terminal_certificate_boxes = 0;
    int maximum_terminal_certificate_depth = 0;
    int closest_point_attempts = 0;
    int closest_point_iterations = 0;
    int closest_point_roots_recovered = 0;
    int closest_point_terminal_misses = 0;
    int closest_point_failures = 0;
    int seam_deduplications = 0;
    int sample_seed_candidates = 0;
    int sample_seeds_accepted = 0;
    int sample_seed_roots_recovered = 0;
    int maximum_sample_seeds_per_element = 0;
    int stationary_solve_attempts = 0;
    int stationary_solve_converged = 0;
    int stationary_witnesses = 0;
    int stationary_protected_root_pairs = 0;
    int ambiguous_root_clusters = 0;
    int non_g1_topology_merges = 0;
    int high_degree_fallbacks = 0;
    std::size_t interface_x = 0;
    std::size_t interface_y = 0;
    std::size_t interface_z = 0;
    std::size_t multi_crossing_edges = 0;
    std::size_t even_parity_interface_edges = 0;
    std::size_t odd_parity_interface_edges = 0;
    std::size_t ambiguous_parity_edges = 0;
    std::size_t ambiguous_label_changing_edges = 0;
    std::size_t targeted_retries = 0;
    std::size_t targeted_retries_resolved = 0;
    std::size_t targeted_retries_unsafe = 0;
    std::size_t correction_safe_edges = 0;
    std::size_t unsafe_label_changing_edges = 0;
    int maximum_targeted_retry_subdivision_depth = 0;
    std::size_t endpoint_parity_fallbacks = 0;
    std::size_t endpoint_classification_queries = 0;
    std::size_t component_parity_toggles = 0;
    std::size_t barrier_x = 0;
    std::size_t barrier_y = 0;
    std::size_t barrier_z = 0;
    int grid_components = 0;
    int box_exterior_components = 0;
    int representative_queries = 0;
    double nurbs_geometry_tolerance = 0.0;
    double nurbs_root_residual_max = 0.0;
    int triangle_fallback_crossings = 0;
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
    int cauchy_value_neighbors_min = 0;
    int cauchy_value_neighbors_max = 0;
    int cauchy_derivative_neighbors_min = 0;
    int cauchy_derivative_neighbors_max = 0;
    double cauchy_radius_max_over_h = 0.0;
    double cauchy_radius_mean_over_h = 0.0;
    int cauchy_incident_patches_min = 0;
    int cauchy_incident_patches_max = 0;
    int cauchy_value_patch_imbalance_max = 0;
    int cauchy_derivative_patch_imbalance_max = 0;
    double cauchy_condition_median = 0.0;
    double cauchy_condition_p95 = 0.0;
    double cauchy_condition_max = 0.0;
    SolveMetrics3D neumann;
    SolveMetrics3D dirichlet_normal;
};

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

SurfaceCloudDiagnostics validate_surface_dofs(const SurfaceDofCloud& cloud,
                                              double h)
{
    if (cloud.patches.empty() || cloud.dofs.size() < 2)
        throw std::runtime_error("surface panel-center cloud is empty");
    std::vector<int> patch_counts(cloud.patches.size(), 0);
    SurfaceCloudDiagnostics result;
    for (std::size_t i = 0; i < cloud.patches.size(); ++i) {
        const SurfacePatchInfo& patch = cloud.patches[i];
        if (patch.smooth_patch_ids.empty()
            || std::find(patch.smooth_patch_ids.begin(),
                         patch.smooth_patch_ids.end(),
                         static_cast<int>(i)) == patch.smooth_patch_ids.end()) {
            throw std::runtime_error(
                "surface patch smooth adjacency must include itself");
        }
        for (int adjacent : patch.smooth_patch_ids) {
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

using app3d::HarmonicPolynomialSpace3D;
using app3d::svd_pseudoinverse_3d;

GeometryBundle make_geometry(
    GeometryKind kind,
    double h,
    const app3d::RigidTransform3D& transform)
{
    const NativeNurbsSurface3D original_surface =
        app3d::make_native_nurbs_surface_3d(kind);
    NativeNurbsSurface3D native_surface =
        app3d::transform_native_nurbs_surface_3d(original_surface, transform);
    for (const auto& patch : native_surface.patches) {
        for (const auto& row : patch.control_net()) {
            for (const Eigen::Vector3d& point : row) {
                if (!point.allFinite()
                    || !(point.array().minCoeff() > kBoxMin)
                    || !(point.array().maxCoeff() < kBoxMin + kBoxSide)) {
                    throw std::runtime_error(
                        "transformed NURBS control point is outside the fixed box");
                }
            }
        }
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
    geometry3d::NurbsPatchTriangulation3D triangulation =
        geometry3d::triangulate_nurbs_surface_patches_3d(
            native_surface.patches, h, options);
    std::string name = native_surface.name;
    std::string description = native_surface.description;
    std::function<bool(const Eigen::Vector3d&)> exact_inside =
        native_surface.exact_inside;
    const int feature_edges = triangulation.summary.num_feature_edges;
    const int feature_vertices = triangulation.summary.num_feature_vertices;
    return {std::move(name),
            std::move(description),
            std::move(native_surface),
            std::move(triangulation.interface),
            std::move(triangulation.geometry_interface),
            std::move(triangulation.triangles),
            std::move(triangulation.geometry_triangles),
            std::move(triangulation.feature_edges),
            feature_edges,
            feature_vertices,
            std::move(exact_inside)};
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

using HarmonicTraceCorrectionTerm3D =
    app3d::HarmonicTraceCorrectionTermInput3D;

struct RestrictOwnerTraceStencil3D {
    app3d::RestrictOwnerSampleInput3D owner_input;
    std::array<int, 64> grid_nodes{};
    std::array<double, 64> weights{};
};

struct RestrictOwnerOracleNode3D {
    std::uint8_t slot = 0;
    int owner_dof = -1;
    app3d::RestrictOwnerNormalizedClass3D owner_class =
        app3d::RestrictOwnerNormalizedClass3D::Target;
    int crossing_patch = -1;
    std::optional<app3d::RestrictOwnerDecisionKind3D>
        legacy_decision_kind;
};

struct RestrictOwnerWorkloadSample3D {
    int target_dof = -1;
    int side = -1;
    int layer = -1;
    Eigen::Vector3d query = Eigen::Vector3d::Zero();
    std::array<int, 3> lower_stencil_index{};
    std::uint64_t wrong_side_mask = 0;
    std::vector<RestrictOwnerOracleNode3D> oracle_nodes;
};

using RestrictOwnerWorkload3D =
    std::vector<RestrictOwnerWorkloadSample3D>;

RestrictOwnerTraceStencil3D build_restrict_owner_trace_stencil_3d(
    const CartesianGrid3D& grid,
    const GridPair3D& grid_pair,
    int target_dof,
    bool desired_inside,
    const Eigen::Vector3d& query,
    const std::array<int, 3>& lower_stencil_index)
{
    RestrictOwnerTraceStencil3D result;
    result.owner_input.target_dof = target_dof;
    result.owner_input.query = query;

    const double h = grid.spacing()[0];
    const std::array<double, 3> origin = grid.origin();
    std::array<std::array<double, 4>, 3> axis_weights{};
    for (int axis = 0; axis < 3; ++axis) {
        const double coordinate =
            (query[axis] - origin[static_cast<std::size_t>(axis)]) / h;
        const double fraction = coordinate
            - static_cast<double>(
                lower_stencil_index[static_cast<std::size_t>(axis)]);
        axis_weights[static_cast<std::size_t>(axis)] =
            cubic_lagrange_weights(fraction);
    }

    int q = 0;
    for (int iz = 0; iz < 4; ++iz) {
        for (int iy = 0; iy < 4; ++iy) {
            for (int ix = 0; ix < 4; ++ix) {
                const int i = lower_stencil_index[0] - 1 + ix;
                const int j = lower_stencil_index[1] - 1 + iy;
                const int k = lower_stencil_index[2] - 1 + iz;
                const int node = grid.index(i, j, k);
                const double weight =
                    axis_weights[0][static_cast<std::size_t>(ix)]
                    * axis_weights[1][static_cast<std::size_t>(iy)]
                    * axis_weights[2][static_cast<std::size_t>(iz)];
                const std::size_t slot = static_cast<std::size_t>(q);
                const Eigen::Vector3d support = grid_point(grid, node);
                result.grid_nodes[slot] = node;
                result.weights[slot] = weight;
                result.owner_input.support_points[slot] = support;
                result.owner_input.wrong_side[slot] =
                    (grid_pair.domain_label(node) > 0) != desired_inside;
                ++q;
            }
        }
    }
    return result;
}

app3d::RestrictOwnerSampleInput3D
reconstruct_restrict_owner_sample_input_3d(
    const CartesianGrid3D& grid,
    int target_dof,
    const Eigen::Vector3d& query,
    const std::array<int, 3>& lower_stencil_index,
    std::uint64_t wrong_side_mask)
{
    app3d::RestrictOwnerSampleInput3D input;
    input.target_dof = target_dof;
    input.query = query;
    int q = 0;
    for (int iz = 0; iz < 4; ++iz) {
        for (int iy = 0; iy < 4; ++iy) {
            for (int ix = 0; ix < 4; ++ix) {
                const int node = grid.index(
                    lower_stencil_index[0] - 1 + ix,
                    lower_stencil_index[1] - 1 + iy,
                    lower_stencil_index[2] - 1 + iz);
                const std::size_t slot = static_cast<std::size_t>(q);
                input.support_points[slot] = grid_point(grid, node);
                input.wrong_side[slot] =
                    (wrong_side_mask
                     & (UINT64_C(1) << static_cast<unsigned>(q))) != 0;
                ++q;
            }
        }
    }
    return input;
}

void restrict_owner_fingerprint_append(
    std::uint64_t& fingerprint,
    std::uint64_t value) noexcept
{
    constexpr std::uint64_t prime = UINT64_C(1099511628211);
    for (int byte = 0; byte < 8; ++byte) {
        fingerprint ^= static_cast<std::uint8_t>(
            value >> static_cast<unsigned>(8 * byte));
        fingerprint *= prime;
    }
}

void append_restrict_owner_trace_fingerprint(
    std::uint64_t& fingerprint,
    int target_dof,
    int side,
    int layer,
    const std::array<int, 3>& lower_stencil_index,
    const RestrictOwnerTraceStencil3D& trace) noexcept
{
    restrict_owner_fingerprint_append(
        fingerprint, static_cast<std::uint32_t>(target_dof));
    restrict_owner_fingerprint_append(
        fingerprint, static_cast<std::uint32_t>(side));
    restrict_owner_fingerprint_append(
        fingerprint, static_cast<std::uint32_t>(layer));
    for (int lower : lower_stencil_index) {
        restrict_owner_fingerprint_append(
            fingerprint, static_cast<std::uint32_t>(lower));
    }
    for (std::size_t q = 0; q < trace.grid_nodes.size(); ++q) {
        std::uint64_t weight_bits = 0;
        static_assert(sizeof(weight_bits) == sizeof(trace.weights[q]),
                      "fingerprint requires binary64 weights");
        std::memcpy(&weight_bits, &trace.weights[q], sizeof(weight_bits));
        restrict_owner_fingerprint_append(
            fingerprint,
            static_cast<std::uint32_t>(trace.grid_nodes[q]));
        restrict_owner_fingerprint_append(fingerprint, weight_bits);
        restrict_owner_fingerprint_append(
            fingerprint,
            trace.owner_input.wrong_side[q] ? UINT64_C(1) : UINT64_C(0));
    }
}

bool restrict_owner_preprocess_diagnostics_equal(
    const app3d::RestrictOwnerPreprocessDiagnostics3D& lhs,
    const app3d::RestrictOwnerPreprocessDiagnostics3D& rhs) noexcept
{
    return lhs.wrong_side_queries == rhs.wrong_side_queries
        && lhs.path_counts == rhs.path_counts
        && lhs.fallback_counts == rhs.fallback_counts
        && lhs.compatible_aabb_candidates == rhs.compatible_aabb_candidates
        && lhs.foreign_aabb_candidates == rhs.foreign_aabb_candidates
        && lhs.control_hull_rejections == rhs.control_hull_rejections
        && lhs.closest_point_attempts == rhs.closest_point_attempts
        && lhs.closest_point_converged == rhs.closest_point_converged
        && lhs.closest_point_iterations == rhs.closest_point_iterations
        && lhs.closest_certified_misses == rhs.closest_certified_misses
        && lhs.closest_certified_roots == rhs.closest_certified_roots
        && lhs.closest_unresolved == rhs.closest_unresolved
        && lhs.optimized_intersection_calls
               == rhs.optimized_intersection_calls
        && lhs.full_fallback_calls == rhs.full_fallback_calls
        && lhs.target_decisions == rhs.target_decisions
        && lhs.foreign_decisions == rhs.foreign_decisions
        && lhs.fail_closed_target_decisions
               == rhs.fail_closed_target_decisions
        && lhs.region_seconds == rhs.region_seconds
        && lhs.closest_point_seconds == rhs.closest_point_seconds
        && lhs.optimized_intersection_seconds
               == rhs.optimized_intersection_seconds
        && lhs.full_fallback_seconds == rhs.full_fallback_seconds;
}

template <std::size_t Count>
std::uint64_t restrict_owner_count_sum(
    const std::array<std::uint64_t, Count>& counts) noexcept
{
    std::uint64_t result = 0;
    for (std::uint64_t count : counts)
        result += count;
    return result;
}

void validate_restrict_owner_preprocess_diagnostics(
    const app3d::RestrictOwnerPreprocessDiagnostics3D& diagnostics,
    std::size_t compatible_query_count)
{
    const std::uint64_t query_count =
        diagnostics.wrong_side_queries;
    if (query_count == 0
        || query_count
               != static_cast<std::uint64_t>(
                   compatible_query_count)) {
        throw std::logic_error(
            "restrict-owner preprocessing query count is inconsistent");
    }
    if (restrict_owner_count_sum(diagnostics.path_counts)
            != query_count
        || restrict_owner_count_sum(diagnostics.fallback_counts)
               != query_count
        || diagnostics.target_decisions
                + diagnostics.foreign_decisions
                + diagnostics.fail_closed_target_decisions
               != query_count) {
        throw std::logic_error(
            "restrict-owner preprocessing diagnostics do not conserve "
            "classified queries");
    }
}

std::uint64_t current_working_set_bytes_3d()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    const BOOL memory_ok = GetProcessMemoryInfo(GetCurrentProcess(),
        reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(std::addressof(counters)),
        sizeof(counters));
    if (memory_ok == FALSE)
        throw std::runtime_error("GetProcessMemoryInfo failed");
    return static_cast<std::uint64_t>(counters.WorkingSetSize);
#else
    rusage usage{};
    const int resource_status = getrusage(RUSAGE_SELF, std::addressof(usage));
    if (resource_status != 0)
        throw std::runtime_error("getrusage(RUSAGE_SELF) failed");
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * UINT64_C(1024);
#endif
#endif
}

class WorkingSetPeakSampler3D {
public:
    WorkingSetPeakSampler3D()
    {
        const std::uint64_t initial = current_working_set_bytes_3d();
        peak_bytes_.store(initial, std::memory_order_relaxed);
#ifdef _WIN32
        worker_ = std::thread([this] {
            while (!stop_.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                if (!enabled_.load(std::memory_order_acquire))
                    continue;
                std::lock_guard<std::mutex> lock(sample_mutex_);
                if (!enabled_.load(std::memory_order_relaxed))
                    continue;
                try {
                    update_sample(current_working_set_bytes_3d());
                } catch (...) {
                    failed_.store(true, std::memory_order_relaxed);
                    enabled_.store(false, std::memory_order_release);
                }
            }
        });
#endif
    }

    WorkingSetPeakSampler3D(const WorkingSetPeakSampler3D&) = delete;
    WorkingSetPeakSampler3D& operator=(
        const WorkingSetPeakSampler3D&) = delete;

    ~WorkingSetPeakSampler3D()
    {
        stop_worker();
    }

    void resume()
    {
        std::lock_guard<std::mutex> lock(sample_mutex_);
        if (finished_)
            throw std::logic_error("working-set sampler is finished");
        if (enabled_.load(std::memory_order_relaxed))
            throw std::logic_error("working-set sampler is already active");
        if (failed_.load(std::memory_order_relaxed))
            throw std::runtime_error("working-set sampler failed");
        const std::uint64_t baseline = current_working_set_bytes_3d();
        active_window_baseline_bytes_ = baseline;
        update_sample(baseline);
        enabled_.store(true, std::memory_order_release);
    }

    void pause() noexcept
    {
        enabled_.store(false, std::memory_order_release);
        std::lock_guard<std::mutex> lock(sample_mutex_);
    }

    void checkpoint()
    {
        std::lock_guard<std::mutex> lock(sample_mutex_);
        if (!enabled_.load(std::memory_order_relaxed))
            throw std::logic_error(
                "working-set checkpoint requires an active window");
        update_sample(current_working_set_bytes_3d());
    }

    void finish()
    {
        if (enabled_.load(std::memory_order_relaxed))
            throw std::logic_error(
                "working-set sampler must be paused before finish");
        stop_worker();
        if (failed_.load(std::memory_order_relaxed))
            throw std::runtime_error("working-set sampler failed");
        finished_ = true;
    }

    std::uint64_t peak_bytes() const
    {
        if (!finished_)
            throw std::logic_error("working-set sampler is not finished");
        return peak_bytes_.load(std::memory_order_relaxed);
    }

    std::uint64_t max_active_increase_bytes() const
    {
        if (!finished_)
            throw std::logic_error("working-set sampler is not finished");
        return max_active_increase_bytes_.load(
            std::memory_order_relaxed);
    }

private:
    void update_sample(std::uint64_t sample) noexcept
    {
        update_atomic_max(peak_bytes_, sample);
        const std::uint64_t active_increase =
            sample > active_window_baseline_bytes_
                ? sample - active_window_baseline_bytes_ : 0;
        update_atomic_max(max_active_increase_bytes_, active_increase);
    }

    static void update_atomic_max(
        std::atomic<std::uint64_t>& destination,
        std::uint64_t sample) noexcept
    {
        std::uint64_t peak = destination.load(std::memory_order_relaxed);
        while (peak < sample
               && !destination.compare_exchange_weak(
                   peak, sample, std::memory_order_relaxed)) {
        }
    }

    void stop_worker() noexcept
    {
#ifdef _WIN32
        enabled_.store(false, std::memory_order_release);
        stop_.store(true, std::memory_order_relaxed);
        if (worker_.joinable())
            worker_.join();
#endif
    }

    std::uint64_t active_window_baseline_bytes_ = 0;
    std::atomic<std::uint64_t> peak_bytes_{0};
    std::atomic<std::uint64_t> max_active_increase_bytes_{0};
    std::atomic<bool> stop_{false};
    std::atomic<bool> enabled_{false};
    std::atomic<bool> failed_{false};
    std::mutex sample_mutex_;
    bool finished_ = false;
#ifdef _WIN32
    std::thread worker_;
#endif
};
struct RestrictOwnerPipelinePreprocessTiming3D {
    double construction_seconds = 0.0;
    double query_seconds = 0.0;
    std::uint64_t peak_working_set_bytes = 0;
    std::uint64_t max_active_increase_bytes = 0;
    std::uint64_t output_digest = UINT64_C(14695981039346656037);
    double assembly_seconds = 0.0;
    double workload_capture_seconds = 0.0;
};

double production_pipeline_setup_seconds_3d(
    double outer_constructor_seconds,
    const RestrictOwnerPipelinePreprocessTiming3D& timing);

constexpr std::size_t kRestrictOwnerDecisionKindCount =
    static_cast<std::size_t>(
        app3d::RestrictOwnerDecisionKind3D::AmbiguousEdgeFallback)
    + std::size_t{1};

struct HarmonicTraceOwnerAuditTerm3D {
    int grid_node = -1;
    double interpolation_weight = 0.0;
    int owner_dof = -1;
    int crossing_patch = -1;
    double crossing_u = 0.0;
    double crossing_v = 0.0;
    double segment_parameter = 0.0;
    double residual = 0.0;
    double transversality = 0.0;
};

struct HarmonicTraceSample3D {
    std::array<int, 64> grid_ids{};
    std::array<double, 64> weights{};
    Eigen::VectorXd legacy_correction_evaluation;
    std::vector<HarmonicTraceCorrectionTerm3D> owner_corrections;
    int wrong_side_node_count = 0;
    double wrong_side_sum_abs_weight = 0.0;
    std::array<int, kRestrictOwnerDecisionKindCount> owner_decision_counts{};
    std::array<double, kRestrictOwnerDecisionKindCount>
        owner_decision_sum_abs_weights{};
    int owner_unresolved_fallback_count = 0;
    double owner_unresolved_fallback_sum_abs_weight = 0.0;
    int owner_unrelated_coincidence_fallback_count = 0;
    double owner_unrelated_coincidence_fallback_sum_abs_weight = 0.0;
    std::size_t owner_geometry_query_count = 0;
    std::vector<HarmonicTraceOwnerAuditTerm3D> owner_reroute_terms;
};

struct RestrictOwnerStagedDecision3D {
    std::uint8_t slot = 0;
    int owner_dof = -1;
    app3d::RestrictOwnerNormalizedClass3D owner_class =
        app3d::RestrictOwnerNormalizedClass3D::Target;
    app3d::RestrictOwnerFallbackCause3D fallback_cause =
        app3d::RestrictOwnerFallbackCause3D::None;
    app3d::RestrictOwnerQueryPath3D query_path =
        app3d::RestrictOwnerQueryPath3D::FullIntersection;
    int legacy_decision_kind = -1;
    int foreign_crossing_index = -1;
};

struct RestrictOwnerTraceSeed3D {
    int target_dof = -1;
    int side = -1;
    int layer = -1;
    Eigen::Vector3d query = Eigen::Vector3d::Zero();
    std::array<int, 3> lower_stencil_index{};
    std::uint64_t wrong_side_mask = 0;
};

struct RestrictOwnerPreparedTraceSample3D : RestrictOwnerTraceSeed3D {
    std::uint64_t output_digest = UINT64_C(14695981039346656037);
    std::vector<RestrictOwnerStagedDecision3D> decisions;
    std::vector<geometry3d::NurbsSurfaceCrossing3D>
        foreign_crossings;
};

int restrict_owner_mask_count_3d(std::uint64_t mask) noexcept
{
    int count = 0;
    while (mask != 0) {
        mask &= mask - UINT64_C(1);
        ++count;
    }
    return count;
}

void prepare_restrict_owner_result_sink_3d(
    RestrictOwnerPreparedTraceSample3D& sink)
{
    const std::size_t capacity = static_cast<std::size_t>(
        restrict_owner_mask_count_3d(sink.wrong_side_mask));
    sink.decisions.clear();
    sink.foreign_crossings.clear();
    sink.decisions.reserve(capacity);
    sink.foreign_crossings.reserve(capacity);
}

void append_restrict_owner_output_digest_double_3d(
    std::uint64_t& digest, double value) noexcept
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, std::addressof(value), sizeof(bits));
    restrict_owner_fingerprint_append(digest, bits);
}

void append_restrict_owner_output_digest_crossing_3d(
    std::uint64_t& digest,
    const geometry3d::NurbsSurfaceCrossing3D& crossing) noexcept
{
    restrict_owner_fingerprint_append(
        digest, static_cast<std::uint32_t>(crossing.patch_index));
    restrict_owner_fingerprint_append(
        digest, static_cast<std::uint32_t>(crossing.component));
    append_restrict_owner_output_digest_double_3d(digest, crossing.u);
    append_restrict_owner_output_digest_double_3d(digest, crossing.v);
    append_restrict_owner_output_digest_double_3d(
        digest, crossing.edge_parameter);
    for (int axis = 0; axis < 3; ++axis) {
        append_restrict_owner_output_digest_double_3d(
            digest, crossing.point[axis]);
        append_restrict_owner_output_digest_double_3d(
            digest, crossing.normal[axis]);
    }
    append_restrict_owner_output_digest_double_3d(
        digest, crossing.residual);
    append_restrict_owner_output_digest_double_3d(
        digest, crossing.transversality);
    restrict_owner_fingerprint_append(
        digest, crossing.feature_edge_contact ? UINT64_C(1) : UINT64_C(0));
    append_restrict_owner_output_digest_double_3d(
        digest, crossing.reliable_transversality_tolerance);
}

void stage_restrict_owner_compact_result_3d(
    const app3d::RestrictOwnerSampleInput3D& input,
    app3d::RestrictOwnerSampleResult3D owner_result,
    RestrictOwnerPreparedTraceSample3D& sink)
{
    const std::size_t expected_count = static_cast<std::size_t>(
        restrict_owner_mask_count_3d(sink.wrong_side_mask));
    if (sink.decisions.size() != 0 || sink.foreign_crossings.size() != 0
        || sink.decisions.capacity() < expected_count
        || sink.foreign_crossings.capacity() < expected_count) {
        throw std::logic_error(
            "restrict-owner result sink was not preallocated");
    }
    sink.output_digest = UINT64_C(14695981039346656037);
    for (int q = 0; q < 64; ++q) {
        const std::size_t slot = static_cast<std::size_t>(q);
        if (!input.wrong_side[slot]) {
            if (owner_result.nodes[slot].has_value()) {
                throw std::logic_error(
                    "restrict-owner preprocessing classified a "
                    "same-side support node");
            }
            continue;
        }
        if (!owner_result.nodes[slot].has_value()) {
            throw std::logic_error(
                "restrict-owner preprocessing omitted a wrong-side "
                "support node");
        }
        app3d::RestrictOwnerPreprocessResult3D& source =
            owner_result.nodes[slot].value();
        RestrictOwnerStagedDecision3D decision;
        decision.slot = static_cast<std::uint8_t>(q);
        decision.owner_dof = source.owner_dof;
        decision.owner_class = source.owner_class;
        decision.fallback_cause = source.fallback_cause;
        decision.query_path = source.query_path;
        if (source.legacy_decision_kind.has_value()) {
            decision.legacy_decision_kind = static_cast<int>(
                *source.legacy_decision_kind);
        }
        if (source.owner_class
            == app3d::RestrictOwnerNormalizedClass3D::UniqueForeign) {
            if (!source.foreign_crossing.has_value()) {
                throw std::logic_error(
                    "foreign restrict-owner decision has no crossing");
            }
            decision.foreign_crossing_index = static_cast<int>(
                sink.foreign_crossings.size());
            sink.foreign_crossings.push_back(
                std::move(*source.foreign_crossing));
        } else if (source.foreign_crossing.has_value()) {
            throw std::logic_error(
                "target restrict-owner decision has a foreign crossing");
        }
        restrict_owner_fingerprint_append(
            sink.output_digest, static_cast<std::uint32_t>(decision.slot));
        restrict_owner_fingerprint_append(
            sink.output_digest,
            static_cast<std::uint32_t>(decision.owner_dof));
        restrict_owner_fingerprint_append(
            sink.output_digest,
            static_cast<std::uint32_t>(decision.owner_class));
        restrict_owner_fingerprint_append(
            sink.output_digest,
            static_cast<std::uint32_t>(decision.query_path));
        restrict_owner_fingerprint_append(
            sink.output_digest,
            static_cast<std::uint32_t>(decision.fallback_cause));
        restrict_owner_fingerprint_append(
            sink.output_digest,
            static_cast<std::uint32_t>(decision.legacy_decision_kind));
        if (decision.foreign_crossing_index >= 0) {
            append_restrict_owner_output_digest_crossing_3d(
                sink.output_digest,
                sink.foreign_crossings[static_cast<std::size_t>(
                    decision.foreign_crossing_index)]);
        }
        sink.decisions.push_back(std::move(decision));
    }
    if (sink.decisions.size() != expected_count)
        throw std::logic_error(
            "restrict-owner compact result count is inconsistent");
}

void run_restrict_owner_sample_core_3d(
    const CartesianGrid3D& grid,
    app3d::RestrictOwnerGeometryPreprocessor3D& preprocessor,
    RestrictOwnerPreparedTraceSample3D& sink)
{
    const app3d::RestrictOwnerSampleInput3D input =
        reconstruct_restrict_owner_sample_input_3d(
            grid, sink.target_dof, sink.query,
            sink.lower_stencil_index, sink.wrong_side_mask);
    stage_restrict_owner_compact_result_3d(
        input, preprocessor.preprocess_sample(input), sink);
}

double run_restrict_owner_timed_sample_3d(
    const CartesianGrid3D& grid,
    app3d::RestrictOwnerGeometryPreprocessor3D& preprocessor,
    RestrictOwnerPreparedTraceSample3D& sink,
    WorkingSetPeakSampler3D* memory_sampler = nullptr)
{
    if (memory_sampler != nullptr) {
        memory_sampler->resume();
        memory_sampler->checkpoint();
    }
    const auto query_start = std::chrono::steady_clock::now();
    try {
        run_restrict_owner_sample_core_3d(grid, preprocessor, sink);
        const auto query_end = std::chrono::steady_clock::now();
        if (memory_sampler != nullptr) {
            memory_sampler->checkpoint();
            memory_sampler->pause();
        }
        return std::chrono::duration<double>(
            query_end - query_start).count();
    } catch (...) {
        if (memory_sampler != nullptr)
            memory_sampler->pause();
        throw;
    }
}

struct RestrictOwnerSampleDiagnostics3D {
    int target_dof = -1;
    int target_patch = -1;
    int side = -1;
    int layer = -1;
    int wrong_side_count = 0;
    double wrong_side_sum_abs_weight = 0.0;
    std::array<int, kRestrictOwnerDecisionKindCount> decision_counts{};
    std::array<double, kRestrictOwnerDecisionKindCount>
        decision_sum_abs_weights{};
    int unresolved_fallback_count = 0;
    double unresolved_fallback_sum_abs_weight = 0.0;
    int unrelated_coincidence_fallback_count = 0;
    double unrelated_coincidence_fallback_sum_abs_weight = 0.0;
    std::size_t geometry_query_count = 0;
};

struct RestrictOwnerAuditRecord3D {
    int target_dof = -1;
    int target_patch = -1;
    int side = -1;
    int layer = -1;
    int grid_node = -1;
    double interpolation_weight = 0.0;
    int owner_dof = -1;
    int owner_patch = -1;
    int crossing_patch = -1;
    double crossing_u = 0.0;
    double crossing_v = 0.0;
    double segment_parameter = 0.0;
    double residual = 0.0;
    double transversality = 0.0;
};

class PanelCenterHarmonicJetKFBI3D {
public:
    PanelCenterHarmonicJetKFBI3D(
        const CartesianGrid3D& grid,
        const GridPair3D& grid_pair,
        const NativeNurbsSurface3D& native_surface,
        const std::vector<geometry3d::NurbsParamTriangle3D>&
            correction_triangles,
        const std::vector<geometry3d::NurbsParamTriangle3D>&
            geometry_triangles,
        const SurfaceDofCloud& cloud,
        app3d::HarmonicCauchyFit3D fit,
        bool build_exterior_only_restrict,
        bool build_crossing_owner)
        : PanelCenterHarmonicJetKFBI3D(
              grid, grid_pair, native_surface, correction_triangles,
              geometry_triangles, cloud, std::move(fit),
              build_exterior_only_restrict,
              build_crossing_owner
                  ? std::optional<app3d::RestrictOwnerPreprocessMode3D>(
                        app3d::RestrictOwnerPreprocessMode3D::
                            FullIntersectionReference)
                  : std::nullopt)
    {}

    PanelCenterHarmonicJetKFBI3D(const CartesianGrid3D& grid,
                                 const GridPair3D& grid_pair,
                                 const NativeNurbsSurface3D& native_surface,
                                 const std::vector<geometry3d::NurbsParamTriangle3D>&
                                     correction_triangles,
                                 const std::vector<geometry3d::NurbsParamTriangle3D>&
                                     geometry_triangles,
                                 const SurfaceDofCloud& cloud,
                                 app3d::HarmonicCauchyFit3D fit,
                                 bool build_exterior_only_restrict = false,
                                 std::optional<
                                     app3d::RestrictOwnerPreprocessMode3D>
                                     restrict_owner_mode = std::nullopt,
                                 app3d::PhaseProfile3D* phase_profile = nullptr,
                                 RestrictOwnerWorkload3D* workload_capture =
                                     nullptr,
                                 RestrictOwnerPipelinePreprocessTiming3D*
                                     preprocess_timing = nullptr)
        : grid_(grid)
        , grid_pair_(grid_pair)
        , native_surface_(native_surface)
        , correction_triangles_(correction_triangles)
        , geometry_triangles_(geometry_triangles)
        , cloud_(cloud)
        , phase_profile_(phase_profile)
        , workload_capture_(workload_capture)
        , h_(grid.spacing()[0])
        , fit_(std::move(fit))
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
        if (build_exterior_only_restrict) {
            exterior_only_restrict_ =
                std::make_unique<app3d::ExteriorOnlyCubicNormalRestrict3D>(
                    grid_, grid_pair_, cloud_);
        }
        profile_phase_3d(
            phase_profile_, PhaseProfileKind3D::CrossingRows, 1,
            [&] {
                build_crossing_rows();
            });
        if (restrict_owner_mode.has_value()) {
            RestrictOwnerPipelinePreprocessTiming3D phase_timing;
            RestrictOwnerPipelinePreprocessTiming3D* timing =
                preprocess_timing != nullptr
                    ? preprocess_timing
                    : (phase_profile_ != nullptr
                        ? std::addressof(phase_timing) : nullptr);
            if (preprocess_timing != nullptr)
                *preprocess_timing = {};
            std::chrono::steady_clock::time_point preparation_start;
            if (phase_profile_ != nullptr)
                preparation_start = std::chrono::steady_clock::now();
            std::vector<RestrictOwnerTraceSeed3D>
                prepared_samples =
                    prepare_restrict_owner_trace_templates();
            RestrictOwnerPreparedTraceSample3D scratch;
            scratch.decisions.reserve(64);
            scratch.foreign_crossings.reserve(64);
            double preparation_seconds = 0.0;
            if (phase_profile_ != nullptr) {
                preparation_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now()
                    - preparation_start).count();
            }
            if (workload_capture_ != nullptr) {
                std::chrono::steady_clock::time_point capture_start;
                if (timing != nullptr)
                    capture_start = std::chrono::steady_clock::now();
                workload_capture_->clear();
                workload_capture_->reserve(prepared_samples.size());
                if (timing != nullptr) {
                    timing->workload_capture_seconds +=
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now()
                            - capture_start).count();
                }
            }
            std::unique_ptr<WorkingSetPeakSampler3D> memory_sampler;
            const bool sample_working_set =
                preprocess_timing != nullptr
                && phase_profile_ == nullptr;
            if (sample_working_set) {
                memory_sampler = std::make_unique<
                    WorkingSetPeakSampler3D>();
            }
            if (timing != nullptr) {
                if (memory_sampler != nullptr) {
                    memory_sampler->resume();
                    memory_sampler->checkpoint();
                }
                try {
                    const auto construction_start =
                        std::chrono::steady_clock::now();
                    restrict_owner_preprocessor_ = std::make_unique<
                        app3d::RestrictOwnerGeometryPreprocessor3D>(
                            native_surface_, cloud_, h_,
                            *restrict_owner_mode);
                    const auto construction_end =
                        std::chrono::steady_clock::now();
                    if (memory_sampler != nullptr) {
                        memory_sampler->checkpoint();
                        memory_sampler->pause();
                    }
                    timing->construction_seconds =
                        std::chrono::duration<double>(
                            construction_end
                            - construction_start).count();
                } catch (...) {
                    if (memory_sampler != nullptr)
                        memory_sampler->pause();
                    throw;
                }
            } else {
                restrict_owner_preprocessor_ = std::make_unique<
                    app3d::RestrictOwnerGeometryPreprocessor3D>(
                        native_surface_, cloud_, h_,
                        *restrict_owner_mode);
            }
            stream_restrict_owner_trace_templates(
                prepared_samples, scratch, timing,
                memory_sampler.get());
            if (memory_sampler != nullptr) {
                memory_sampler->finish();
                preprocess_timing->peak_working_set_bytes =
                    memory_sampler->peak_bytes();
                preprocess_timing->max_active_increase_bytes =
                    memory_sampler->max_active_increase_bytes();
            }
            if (phase_profile_ != nullptr) {
                const std::uint64_t samples =
                    static_cast<std::uint64_t>(
                        prepared_samples.size());
                std::uint64_t internal_timer_reads =
                    UINT64_C(2)  // template preparation
                    + UINT64_C(2)  // preprocessor construction
                    + UINT64_C(2) * samples  // owner queries
                    + UINT64_C(2) * samples; // trace assembly
                if (workload_capture_ != nullptr) {
                    internal_timer_reads +=
                        UINT64_C(2)  // workload initialization
                        + UINT64_C(2) * samples;
                }
                phase_profile_->note_timer_reads(
                    internal_timer_reads);
                phase_profile_->add(
                    PhaseProfileKind3D::RestrictOwnerGeometryPreprocessing,
                    timing->construction_seconds + timing->query_seconds,
                    1);
                phase_profile_->add(
                    PhaseProfileKind3D::TraceOwnerTemplateAssembly,
                    preparation_seconds + timing->assembly_seconds,
                    1);
            }
            crossing_owner_templates_built_ = true;
        } else {
            if (workload_capture_ != nullptr) {
                throw std::invalid_argument(
                    "restrict-owner workload capture requires a mode");
            }
            build_trace_templates();
        }
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

    const app3d::HarmonicCauchyFit3D& cauchy_fit() const noexcept
    {
        return fit_;
    }

    const LaplaceCorrectionSupport3D& correction_support() const noexcept
    {
        return correction_support_;
    }

    std::vector<double> cauchy_condition_values() const
    {
        return fit_.condition_values();
    }

    std::vector<double> exterior_only_restrict_condition_values() const
    {
        if (!exterior_only_restrict_)
            throw std::runtime_error(
                "exterior-only normal restrict was not initialized");
        std::vector<double> result;
        result.reserve(exterior_only_restrict_->stencils().size());
        for (const auto& stencil : exterior_only_restrict_->stencils())
            result.push_back(stencil.condition);
        return result;
    }

    std::vector<int> joint_trace_wrong_side_node_counts() const
    {
        std::vector<int> result(static_cast<std::size_t>(surface_size()), 0);
        for (int center = 0; center < surface_size(); ++center) {
            for (int side = 0; side < 2; ++side) {
                for (int layer = 0; layer < 4; ++layer) {
                    result[static_cast<std::size_t>(center)] +=
                        trace_samples_[trace_sample_index(center, side, layer)]
                            .wrong_side_node_count;
                }
            }
        }
        return result;
    }

    std::vector<RestrictOwnerSampleDiagnostics3D>
    restrict_owner_sample_diagnostics() const
    {
        if (!crossing_owner_templates_built_) {
            throw std::runtime_error(
                "crossing-owner normal restrict was not initialized");
        }
        std::vector<RestrictOwnerSampleDiagnostics3D> result;
        result.reserve(trace_samples_.size());
        for (int center = 0; center < surface_size(); ++center) {
            const int target_patch =
                cloud_.dofs[static_cast<std::size_t>(center)].patch_id;
            for (int side = 0; side < 2; ++side) {
                for (int layer = 0; layer < 4; ++layer) {
                    const HarmonicTraceSample3D& sample = trace_samples_[
                        trace_sample_index(center, side, layer)];
                    RestrictOwnerSampleDiagnostics3D item;
                    item.target_dof = center;
                    item.target_patch = target_patch;
                    item.side = side;
                    item.layer = layer;
                    item.wrong_side_count = sample.wrong_side_node_count;
                    item.wrong_side_sum_abs_weight =
                        sample.wrong_side_sum_abs_weight;
                    item.decision_counts = sample.owner_decision_counts;
                    item.decision_sum_abs_weights =
                        sample.owner_decision_sum_abs_weights;
                    item.unresolved_fallback_count =
                        sample.owner_unresolved_fallback_count;
                    item.unresolved_fallback_sum_abs_weight =
                        sample.owner_unresolved_fallback_sum_abs_weight;
                    item.unrelated_coincidence_fallback_count =
                        sample.owner_unrelated_coincidence_fallback_count;
                    item.unrelated_coincidence_fallback_sum_abs_weight =
                        sample.
                            owner_unrelated_coincidence_fallback_sum_abs_weight;
                    item.geometry_query_count =
                        sample.owner_geometry_query_count;
                    result.push_back(std::move(item));
                }
            }
        }
        return result;
    }

    std::vector<RestrictOwnerAuditRecord3D>
    restrict_owner_audit_records() const
    {
        if (!crossing_owner_templates_built_) {
            throw std::runtime_error(
                "crossing-owner normal restrict was not initialized");
        }
        std::vector<RestrictOwnerAuditRecord3D> result;
        for (int center = 0; center < surface_size(); ++center) {
            const int target_patch =
                cloud_.dofs[static_cast<std::size_t>(center)].patch_id;
            for (int side = 0; side < 2; ++side) {
                for (int layer = 0; layer < 4; ++layer) {
                    const HarmonicTraceSample3D& sample = trace_samples_[
                        trace_sample_index(center, side, layer)];
                    for (const HarmonicTraceOwnerAuditTerm3D& term
                         : sample.owner_reroute_terms) {
                        if (term.owner_dof < 0
                            || term.owner_dof >= surface_size()) {
                            throw std::logic_error(
                                "crossing-owner audit has invalid owner DOF");
                        }
                        RestrictOwnerAuditRecord3D item;
                        item.target_dof = center;
                        item.target_patch = target_patch;
                        item.side = side;
                        item.layer = layer;
                        item.grid_node = term.grid_node;
                        item.interpolation_weight =
                            term.interpolation_weight;
                        item.owner_dof = term.owner_dof;
                        item.owner_patch = cloud_.dofs[
                            static_cast<std::size_t>(term.owner_dof)].patch_id;
                        item.crossing_patch = term.crossing_patch;
                        item.crossing_u = term.crossing_u;
                        item.crossing_v = term.crossing_v;
                        item.segment_parameter = term.segment_parameter;
                        item.residual = term.residual;
                        item.transversality = term.transversality;
                        result.push_back(std::move(item));
                    }
                }
            }
        }
        return result;
    }

    std::uint64_t restrict_owner_workload_fingerprint() const noexcept
    {
        return restrict_owner_workload_fingerprint_;
    }

    std::uint64_t restrict_owner_output_digest() const noexcept
    {
        return restrict_owner_output_digest_;
    }

    const app3d::RestrictOwnerPreprocessDiagnostics3D&
    restrict_owner_preprocess_diagnostics() const
    {
        if (!restrict_owner_preprocessor_) {
            throw std::runtime_error(
                "crossing-owner normal restrict was not initialized");
        }
        return restrict_owner_preprocessor_->diagnostics();
    }

    double restrict_owner_correction_linf() const
    {
        double result = 0.0;
        for (const HarmonicTraceSample3D& sample : trace_samples_) {
            for (const HarmonicTraceCorrectionTerm3D& term
                 : sample.owner_corrections) {
                result = std::max(
                    result, term.evaluation.lpNorm<Eigen::Infinity>());
            }
        }
        return result;
    }

    double restrict_owner_correction_linf_difference(
        const PanelCenterHarmonicJetKFBI3D& other) const
    {
        if (!crossing_owner_templates_built_
            || !other.crossing_owner_templates_built_) {
            throw std::invalid_argument(
                "crossing-owner correction comparison requires two "
                "initialized pipelines");
        }
        if (restrict_owner_workload_fingerprint_
                != other.restrict_owner_workload_fingerprint_
            || trace_samples_.size() != other.trace_samples_.size()
            || fit_.space().dimension()
                   != other.fit_.space().dimension()) {
            throw std::invalid_argument(
                "crossing-owner correction comparison received different "
                "trace workloads");
        }

        double result = 0.0;
        for (std::size_t sample = 0;
             sample < trace_samples_.size(); ++sample) {
            const auto& lhs =
                trace_samples_[sample].owner_corrections;
            const auto& rhs =
                other.trace_samples_[sample].owner_corrections;
            std::size_t i = 0;
            std::size_t j = 0;
            while (i < lhs.size() || j < rhs.size()) {
                if (j >= rhs.size()
                    || (i < lhs.size()
                        && lhs[i].owner_dof < rhs[j].owner_dof)) {
                    result = std::max(
                        result,
                        lhs[i].evaluation.lpNorm<Eigen::Infinity>());
                    ++i;
                } else if (i >= lhs.size()
                           || rhs[j].owner_dof < lhs[i].owner_dof) {
                    result = std::max(
                        result,
                        rhs[j].evaluation.lpNorm<Eigen::Infinity>());
                    ++j;
                } else {
                    result = std::max(
                        result,
                        (lhs[i].evaluation - rhs[j].evaluation)
                            .lpNorm<Eigen::Infinity>());
                    ++i;
                    ++j;
                }
            }
        }
        return result;
    }

    std::size_t restrict_owner_geometry_query_count() const
    {
        std::size_t result = 0;
        for (const HarmonicTraceSample3D& sample : trace_samples_)
            result += sample.owner_geometry_query_count;
        return result;
    }

    HarmonicJetField3D evaluate(
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        HarmonicJetField3D result;
        app3d::HarmonicCauchyApplyResult3D applied = profile_phase_3d(
            phase_profile_, PhaseProfileKind3D::CauchyCoefficients, 1,
            [&] {
                return fit_.apply(value_jump, normal_jump);
            });
        result.coefficients = std::move(applied.coefficients);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(grid_.num_dofs());
        profile_phase_3d(
            phase_profile_, PhaseProfileKind3D::SpreadRhsAssembly, 1,
            [&] {
                for (const HarmonicCrossingRow3D& row : crossing_rows_) {
                    rhs[row.rhs_node] += row.scale
                        * row.evaluation.dot(
                            result.coefficients.row(
                                row.center_dof).transpose());
                }
            });
        // The spread correction is assembled for Delta_h, while the project
        // bulk solver accepts the right-hand side of -Delta_h.
        profile_phase_3d(
            phase_profile_, PhaseProfileKind3D::FftBulkSolve, 1,
            [&] {
                bulk_.solve(-rhs, result.potential);
            });
        return result;
    }

    HarmonicJetField3D field_from_grid_and_jumps(
        const Eigen::VectorXd& potential,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        if (potential.size() != grid_.num_dofs()
            || value_jump.size() != surface_size()
            || normal_jump.size() != surface_size()) {
            throw std::invalid_argument(
                "exact-grid field received incompatible sizes");
        }
        app3d::HarmonicCauchyApplyResult3D applied = profile_phase_3d(
            phase_profile_, PhaseProfileKind3D::CauchyCoefficients, 1,
            [&] {
                return fit_.apply(value_jump, normal_jump);
            });
        return {potential, std::move(applied.coefficients)};
    }

    Eigen::VectorXd exterior_trace(const HarmonicJetField3D& field,
                                   const Eigen::VectorXd& value_jump,
                                   const Eigen::VectorXd& normal_jump) const
    {
        return exterior_trace(
            field, value_jump, normal_jump,
            ExteriorValueRestrictMode3D::JointTricubicCauchy);
    }

    Eigen::VectorXd exterior_trace(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        ExteriorValueRestrictMode3D mode) const
    {
        const auto correction_mode =
            mode == ExteriorValueRestrictMode3D::
                        JointTricubicCrossingOwner
                ? app3d::TraceCorrectionOwnerMode3D::CrossingOwner
                : app3d::TraceCorrectionOwnerMode3D::CenterDof;
        return recover_trace(
            continued_samples(
                field, value_jump, normal_jump, false, correction_mode),
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

    Eigen::VectorXd interior_trace(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        ExteriorValueRestrictMode3D mode) const
    {
        const auto correction_mode =
            mode == ExteriorValueRestrictMode3D::
                        JointTricubicCrossingOwner
                ? app3d::TraceCorrectionOwnerMode3D::CrossingOwner
                : app3d::TraceCorrectionOwnerMode3D::CenterDof;
        return recover_trace(
            continued_samples(
                field, value_jump, normal_jump, true, correction_mode),
            c0_weights_, 1.0);
    }

    Eigen::VectorXd exterior_normal_trace(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        return exterior_normal_trace(
            field, value_jump, normal_jump,
            ExteriorNormalRestrictMode3D::JointTricubicCauchy);
    }

    Eigen::VectorXd exterior_normal_trace(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        ExteriorNormalRestrictMode3D mode) const
    {
        if (mode == ExteriorNormalRestrictMode3D::ExteriorOnlyHarmonicCubic) {
            if (!exterior_only_restrict_) {
                throw std::runtime_error(
                    "exterior-only normal restrict was not initialized");
            }
            return exterior_only_restrict_->apply(field.potential);
        }
        const auto correction_mode =
            mode == ExteriorNormalRestrictMode3D::
                        JointTricubicCrossingOwner
                ? app3d::TraceCorrectionOwnerMode3D::CrossingOwner
                : app3d::TraceCorrectionOwnerMode3D::CenterDof;
        return recover_trace(
            continued_samples(
                field, value_jump, normal_jump, false, correction_mode),
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
                                      bool interior_continuation,
                                      app3d::TraceCorrectionOwnerMode3D mode =
                                          app3d::
                                              TraceCorrectionOwnerMode3D::
                                                  CenterDof) const
    {
        return profile_phase_3d(
            phase_profile_, PhaseProfileKind3D::RestrictContinuedSamples, 1,
            [&] {
        const bool use_crossing_owner =
            mode == app3d::TraceCorrectionOwnerMode3D::CrossingOwner;
        if (use_crossing_owner && !crossing_owner_templates_built_) {
            throw std::runtime_error(
                "crossing-owner trace restrict was not initialized");
        }
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
                    value += app3d::apply_harmonic_trace_correction_3d(
                        center,
                        field.coefficients,
                        sample.legacy_correction_evaluation,
                        sample.owner_corrections,
                        mode);
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
            });
    }

    Eigen::VectorXd recover_trace(
        const Eigen::MatrixXd& samples,
        const std::array<double, 8>& weights,
        double scale) const
    {
        return profile_phase_3d(
            phase_profile_, PhaseProfileKind3D::RestrictRecovery, 1,
            [&] {
        Eigen::VectorXd trace(samples.rows());
        for (int center = 0; center < samples.rows(); ++center) {
            double value = 0.0;
            for (int q = 0; q < 8; ++q)
                value += weights[static_cast<std::size_t>(q)]
                       * samples(center, q);
            trace[center] = scale * value;
        }
        return trace;
            });
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

    int surface_dof_for_crossing(const P2CrossingOwner3D& owner,
                                 const Eigen::Vector3d& point) const
    {
        int patch = owner.nurbs_patch_index;
        Eigen::Vector2d uv = owner.nurbs_parameter;
        if (patch >= 0) {
            if (patch >= static_cast<int>(native_surface_.patches.size())) {
                throw std::runtime_error(
                    "native crossing patch index is outside the readiness NURBS surface");
            }
            if (!uv.allFinite()) {
                throw std::runtime_error(
                    "native crossing parameters are not finite");
            }
        }
        if (patch < 0) {
            const geometry3d::NurbsParamTriangle3D* triangle = nullptr;
            Eigen::Vector3d barycentric = Eigen::Vector3d::Zero();
            if (owner.geometry_panel_index >= 0
                && owner.geometry_panel_index
                       < static_cast<int>(geometry_triangles_.size())) {
                triangle = &geometry_triangles_[
                    static_cast<std::size_t>(owner.geometry_panel_index)];
                barycentric = owner.geometry_barycentric;
            } else if (owner.panel_index >= 0
                       && owner.panel_index
                              < static_cast<int>(correction_triangles_.size())) {
                triangle = &correction_triangles_[
                    static_cast<std::size_t>(owner.panel_index)];
                barycentric = owner.barycentric;
            }
            if (triangle == nullptr) {
                throw std::runtime_error(
                    "crossing has no native NURBS triangle owner");
            }
            patch = triangle->patch_index;
            uv = app3d::interpolate_triangle_parameter(
                *triangle, barycentric);
        }
        const std::array<int, 4> candidates =
            app3d::parameter_dof_candidates_2x2(
                native_surface_, cloud_, patch, uv.x(), uv.y());
        const geometry3d::NurbsSurfaceDerivatives3D derivatives =
            native_surface_.patches[
                static_cast<std::size_t>(patch)]
                .evaluate_with_derivatives(uv.x(), uv.y());
        Eigen::Vector3d reference_normal =
            derivatives.du.cross(derivatives.dv).normalized();
        int nearest = -1;
        double best = std::numeric_limits<double>::infinity();
        for (int pass = 0; pass < 2 && nearest < 0; ++pass) {
            for (int q : candidates) {
                const SurfaceDof& dof = cloud_.dofs[static_cast<std::size_t>(q)];
                if (pass == 0 && dof.normal.dot(reference_normal) < 0.50) {
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
            throw std::runtime_error(
                "failed to associate a crossing with four parameter DOFs");
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
            const Eigen::Vector3d hit = owner.nurbs_patch_index >= 0
                ? owner.crossing_point
                : a + phase * (b - a);

            const int center = surface_dof_for_crossing(owner, hit);
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

    HarmonicTraceSample3D build_trace_sample(
        int center,
        int side,
        int layer,
        RestrictOwnerTraceSeed3D* prepared = nullptr)
    {
        if (side < 0 || side >= 2
            || layer < 0
            || layer >= static_cast<int>(normal_layers_.size())) {
            throw std::logic_error(
                "normal-layer trace sample index is invalid");
        }
        const SurfaceDof& dof = cloud_.dofs[static_cast<std::size_t>(center)];
        const bool desired_inside = side == 0;
        const double sign = desired_inside ? -1.0 : 1.0;
        const double signed_layer =
            sign * normal_layers_[static_cast<std::size_t>(layer)];
        const Eigen::Vector3d query =
            dof.point + signed_layer * h_ * dof.normal;
        const std::array<double, 3> origin = grid_.origin();
        const std::array<int, 3> dims = grid_.dof_dims();
        std::array<int, 3> lo{};
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
        }

        if (prepared != nullptr) {
            prepared->target_dof = center;
            prepared->side = side;
            prepared->layer = layer;
            prepared->query = query;
            prepared->lower_stencil_index = lo;
        }

        // Four Cartesian slices are first interpolated by a 4x4 bicubic
        // polynomial; a final cubic interpolation in the third direction is
        // algebraically the standard 4x4x4 tensor-product tricubic formula.
        const RestrictOwnerTraceStencil3D trace =
            build_restrict_owner_trace_stencil_3d(
                grid_, grid_pair_, center, desired_inside, query, lo);
        append_restrict_owner_trace_fingerprint(
            restrict_owner_workload_fingerprint_,
            center, side, layer, lo, trace);

        HarmonicTraceSample3D result;
        result.grid_ids = trace.grid_nodes;
        result.weights = trace.weights;
        result.legacy_correction_evaluation =
            Eigen::VectorXd::Zero(fit_.space().dimension());
        const double correction_sign = desired_inside ? 1.0 : -1.0;
        for (int q = 0; q < 64; ++q) {
            const std::size_t slot = static_cast<std::size_t>(q);
            if (!trace.owner_input.wrong_side[slot]) {
                continue;
            }
            if (prepared != nullptr) {
                prepared->wrong_side_mask |=
                    UINT64_C(1) << static_cast<unsigned>(q);
            }
            ++result.wrong_side_node_count;
            const double weight = trace.weights[slot];
            result.wrong_side_sum_abs_weight += std::abs(weight);
            const Eigen::Vector3d& node_point =
                trace.owner_input.support_points[slot];
            const Eigen::Vector3d target_xi =
                local_coordinate(center, node_point);
            result.legacy_correction_evaluation +=
                correction_sign * weight
                * fit_.space().basis(
                    target_xi.x(), target_xi.y(), target_xi.z());
        }

        if (prepared != nullptr) {
            int mask_count = 0;
            std::uint64_t remaining = prepared->wrong_side_mask;
            while (remaining != 0) {
                remaining &= remaining - UINT64_C(1);
                ++mask_count;
            }
            if (mask_count != result.wrong_side_node_count) {
                throw std::logic_error(
                    "restrict-owner wrong-side mask is inconsistent");
            }
        }

        return result;
    }

    std::vector<RestrictOwnerTraceSeed3D>
    prepare_restrict_owner_trace_templates()
    {
        const std::size_t sample_count = static_cast<std::size_t>(
            surface_size() * 2 * normal_layers_.size());
        std::vector<RestrictOwnerTraceSeed3D> prepared;
        prepared.reserve(sample_count);
        trace_samples_.reserve(sample_count);
        for (int center = 0; center < surface_size(); ++center) {
            for (int side = 0; side < 2; ++side) {
                for (int layer = 0;
                     layer < static_cast<int>(normal_layers_.size());
                     ++layer) {
                    RestrictOwnerTraceSeed3D item;
                    HarmonicTraceSample3D sample =
                        build_trace_sample(
                            center, side, layer, &item);
                    prepared.push_back(std::move(item));
                    trace_samples_.push_back(std::move(sample));
                }
            }
        }
        return prepared;
    }

    void stream_restrict_owner_trace_templates(
        const std::vector<RestrictOwnerTraceSeed3D>& prepared,
        RestrictOwnerPreparedTraceSample3D& scratch,
        RestrictOwnerPipelinePreprocessTiming3D* preprocess_timing,
        WorkingSetPeakSampler3D* memory_sampler)
    {
        if (!restrict_owner_preprocessor_
            || prepared.size() != trace_samples_.size()) {
            throw std::logic_error(
                "restrict-owner preprocessing batch is invalid");
        }
        for (std::size_t sample = 0;
             sample < prepared.size(); ++sample) {
            static_cast<RestrictOwnerTraceSeed3D&>(scratch) =
                prepared[sample];
            prepare_restrict_owner_result_sink_3d(scratch);
            if (preprocess_timing != nullptr) {
                preprocess_timing->query_seconds +=
                    run_restrict_owner_timed_sample_3d(
                        grid_, *restrict_owner_preprocessor_, scratch,
                        memory_sampler);
            } else {
                run_restrict_owner_sample_core_3d(
                    grid_, *restrict_owner_preprocessor_, scratch);
            }
            restrict_owner_fingerprint_append(
                restrict_owner_output_digest_, scratch.output_digest);
            if (preprocess_timing != nullptr) {
                restrict_owner_fingerprint_append(
                    preprocess_timing->output_digest,
                    scratch.output_digest);
                const auto assembly_start =
                    std::chrono::steady_clock::now();
                assemble_restrict_owner_trace_sample(sample, scratch);
                preprocess_timing->assembly_seconds +=
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now()
                        - assembly_start).count();
            } else {
                assemble_restrict_owner_trace_sample(sample, scratch);
            }
            if (workload_capture_ != nullptr) {
                if (preprocess_timing != nullptr) {
                    const auto capture_start =
                        std::chrono::steady_clock::now();
                    capture_restrict_owner_workload_sample(scratch);
                    preprocess_timing->workload_capture_seconds +=
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now()
                            - capture_start).count();
                } else {
                    capture_restrict_owner_workload_sample(scratch);
                }
            }
        }
    }

    void assemble_restrict_owner_trace_sample(
        std::size_t sample_index,
        const RestrictOwnerPreparedTraceSample3D& prepared)
    {
        HarmonicTraceSample3D& result =
            trace_samples_[sample_index];
        result.owner_geometry_query_count =
            prepared.decisions.size();
        const bool desired_inside = prepared.side == 0;
        const double correction_sign =
            desired_inside ? 1.0 : -1.0;
        const bool full_reference =
            restrict_owner_preprocessor_->mode()
            == app3d::RestrictOwnerPreprocessMode3D::
                FullIntersectionReference;
        std::map<int, Eigen::VectorXd> owner_evaluations;
        int normalized_foreign_count = 0;

        std::size_t decision_index = 0;
        for (int q = 0; q < 64; ++q) {
            if (decision_index >= prepared.decisions.size()
                || prepared.decisions[decision_index].slot
                       > static_cast<std::uint8_t>(q)) {
                continue;
            }
            const RestrictOwnerStagedDecision3D& decision =
                prepared.decisions[decision_index];
            if (decision.slot != static_cast<std::uint8_t>(q)) {
                throw std::logic_error(
                    "restrict-owner compact slots are not ordered");
            }
            ++decision_index;
            const std::size_t slot = static_cast<std::size_t>(q);
            const int owner = decision.owner_dof;
            if (owner < 0 || owner >= surface_size()) {
                throw std::logic_error(
                    "restrict-owner preprocessing returned an invalid "
                    "owner DOF");
            }
            const double weight = result.weights[slot];
            const double absolute_weight = std::abs(weight);

            if (full_reference) {
                if (decision.legacy_decision_kind >= 0) {
                    const std::size_t kind = static_cast<std::size_t>(
                        decision.legacy_decision_kind);
                    if (kind
                        >= result.owner_decision_counts.size()) {
                        throw std::logic_error(
                            "crossing-owner decision kind is invalid");
                    }
                    ++result.owner_decision_counts[kind];
                    result.owner_decision_sum_abs_weights[kind] +=
                        absolute_weight;
                } else if (decision.fallback_cause
                           == app3d::RestrictOwnerFallbackCause3D::
                               Unresolved) {
                    ++result.owner_unresolved_fallback_count;
                    result.owner_unresolved_fallback_sum_abs_weight +=
                        absolute_weight;
                } else if (decision.fallback_cause
                           == app3d::RestrictOwnerFallbackCause3D::
                               Coincidence) {
                    ++result.
                        owner_unrelated_coincidence_fallback_count;
                    result.
                        owner_unrelated_coincidence_fallback_sum_abs_weight +=
                            absolute_weight;
                } else {
                    throw std::logic_error(
                        "full-reference restrict-owner result has no "
                        "legacy diagnostic class");
                }
            }

            if (decision.owner_class
                == app3d::RestrictOwnerNormalizedClass3D::
                    UniqueForeign) {
                if (decision.foreign_crossing_index < 0
                    || decision.foreign_crossing_index
                           >= static_cast<int>(
                               prepared.foreign_crossings.size())) {
                    throw std::logic_error(
                        "foreign restrict-owner crossing index is invalid");
                }
                ++normalized_foreign_count;
                const auto& root = prepared.foreign_crossings[
                    static_cast<std::size_t>(
                        decision.foreign_crossing_index)];
                HarmonicTraceOwnerAuditTerm3D audit;
                audit.grid_node = result.grid_ids[slot];
                audit.interpolation_weight = weight;
                audit.owner_dof = owner;
                audit.crossing_patch = root.patch_index;
                audit.crossing_u = root.u;
                audit.crossing_v = root.v;
                audit.segment_parameter = root.edge_parameter;
                audit.residual = root.residual;
                audit.transversality = root.transversality;
                result.owner_reroute_terms.push_back(
                    std::move(audit));
            } else if (decision.foreign_crossing_index >= 0) {
                throw std::logic_error(
                    "target restrict-owner decision has a crossing index");
            }

            const Eigen::Vector3d node_point =
                grid_point(grid_, result.grid_ids[slot]);
            const Eigen::Vector3d owner_xi =
                local_coordinate(owner, node_point);
            Eigen::VectorXd& owner_evaluation =
                owner_evaluations[owner];
            if (owner_evaluation.size() == 0) {
                owner_evaluation =
                    Eigen::VectorXd::Zero(fit_.space().dimension());
            }
            owner_evaluation +=
                correction_sign
                * result.weights[slot]
                * fit_.space().basis(
                    owner_xi.x(), owner_xi.y(), owner_xi.z());
        }
        if (decision_index != prepared.decisions.size()) {
            throw std::logic_error(
                "restrict-owner compact decisions were not consumed");
        }
        result.owner_corrections.reserve(owner_evaluations.size());
        for (auto& owner_evaluation : owner_evaluations) {
            HarmonicTraceCorrectionTerm3D term;
            term.owner_dof = owner_evaluation.first;
            term.evaluation = std::move(owner_evaluation.second);
            result.owner_corrections.push_back(std::move(term));
        }
        if (full_reference) {
            int classified_count =
                result.owner_unresolved_fallback_count
                + result.owner_unrelated_coincidence_fallback_count;
            for (int count : result.owner_decision_counts)
                classified_count += count;
            if (classified_count != result.wrong_side_node_count) {
                throw std::logic_error(
                    "full-reference restrict-owner diagnostics did not "
                    "classify every wrong-side support node");
            }
        }
        if (result.owner_geometry_query_count
                != static_cast<std::size_t>(
                    result.wrong_side_node_count)
            || result.owner_reroute_terms.size()
                   != static_cast<std::size_t>(
                       normalized_foreign_count)) {
            throw std::logic_error(
                "crossing-owner preprocessing diagnostics are "
                "inconsistent");
        }
    }

    void capture_restrict_owner_workload_sample(
        const RestrictOwnerPreparedTraceSample3D& source)
    {
        if (workload_capture_ == nullptr)
            return;
        RestrictOwnerWorkloadSample3D workload;
        workload.target_dof = source.target_dof;
        workload.side = source.side;
        workload.layer = source.layer;
        workload.query = source.query;
        workload.lower_stencil_index = source.lower_stencil_index;
        workload.wrong_side_mask = source.wrong_side_mask;
        workload.oracle_nodes.reserve(source.decisions.size());
        for (const RestrictOwnerStagedDecision3D& decision
             : source.decisions) {
            if (decision.owner_dof < 0
                || decision.owner_dof >= surface_size()) {
                throw std::logic_error(
                    "restrict-owner workload owner DOF is invalid");
            }
            RestrictOwnerOracleNode3D oracle;
            oracle.slot = decision.slot;
            oracle.owner_dof = decision.owner_dof;
            oracle.owner_class = decision.owner_class;
            if (decision.foreign_crossing_index >= 0) {
                if (decision.foreign_crossing_index
                    >= static_cast<int>(
                        source.foreign_crossings.size())) {
                    throw std::logic_error(
                        "restrict-owner workload crossing is invalid");
                }
                oracle.crossing_patch = source.foreign_crossings[
                    static_cast<std::size_t>(
                        decision.foreign_crossing_index)].patch_index;
            }
            if (decision.legacy_decision_kind >= 0) {
                oracle.legacy_decision_kind =
                    static_cast<app3d::RestrictOwnerDecisionKind3D>(
                        decision.legacy_decision_kind);
            }
            workload.oracle_nodes.push_back(std::move(oracle));
        }
        if (workload.oracle_nodes.size()
            != static_cast<std::size_t>(
                restrict_owner_mask_count_3d(
                    workload.wrong_side_mask))) {
            throw std::logic_error(
                "restrict-owner workload count is inconsistent");
        }
        workload_capture_->push_back(std::move(workload));
    }

    void build_trace_templates()
    {
        trace_samples_.reserve(
            static_cast<std::size_t>(surface_size() * 2 * 4));
        for (int center = 0; center < surface_size(); ++center) {
            for (int side = 0; side < 2; ++side) {
                for (int layer = 0;
                     layer < static_cast<int>(normal_layers_.size());
                     ++layer) {
                    trace_samples_.push_back(
                        build_trace_sample(center, side, layer));
                }
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
        const Eigen::MatrixXd pinv = svd_pseudoinverse_3d(design, 1.0e-13);
        for (int q = 0; q < 8; ++q) {
            c0_weights_[static_cast<std::size_t>(q)] = pinv(0, q);
            c1_weights_[static_cast<std::size_t>(q)] = pinv(1, q);
        }
    }

    const CartesianGrid3D& grid_;
    const GridPair3D& grid_pair_;
    const NativeNurbsSurface3D& native_surface_;
    const std::vector<geometry3d::NurbsParamTriangle3D>& correction_triangles_;
    const std::vector<geometry3d::NurbsParamTriangle3D>& geometry_triangles_;
    const SurfaceDofCloud& cloud_;
    app3d::PhaseProfile3D* phase_profile_ = nullptr;
    RestrictOwnerWorkload3D* workload_capture_ = nullptr;
    std::unique_ptr<app3d::RestrictOwnerGeometryPreprocessor3D>
        restrict_owner_preprocessor_;
    bool crossing_owner_templates_built_ = false;
    std::uint64_t restrict_owner_workload_fingerprint_ =
        UINT64_C(14695981039346656037);
    std::uint64_t restrict_owner_output_digest_ =
        UINT64_C(14695981039346656037);
    double h_ = 0.0;
    app3d::HarmonicCauchyFit3D fit_;
    std::unique_ptr<app3d::ExteriorOnlyCubicNormalRestrict3D>
        exterior_only_restrict_;
    LaplaceFftBulkSolverZfft3D bulk_;
    LaplaceCorrectionSupport3D correction_support_;
    std::vector<HarmonicCrossingRow3D> crossing_rows_;
    std::vector<HarmonicTraceSample3D> trace_samples_;
    const std::array<double, kRestrictNormalDegree + 1> normal_layers_{
        {0.2, 0.6, 1.0, 1.4}};
    std::array<double, 8> c0_weights_{};
    std::array<double, 8> c1_weights_{};
};

class ExteriorZeroTraceOperator3D final : public IKFBIOperator {
public:
    explicit ExteriorZeroTraceOperator3D(
        const PanelCenterHarmonicJetKFBI3D& pipeline,
        ExteriorValueRestrictMode3D restrict_mode =
            ExteriorValueRestrictMode3D::JointTricubicCauchy)
        : pipeline_(pipeline)
        , restrict_mode_(restrict_mode)
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
            pipeline_.exterior_trace(
                field, value_jump, zero_normal, restrict_mode_);
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
            field, zero_value, prescribed_normal_jump, restrict_mode_);
        return result;
    }

private:
    const PanelCenterHarmonicJetKFBI3D& pipeline_;
    ExteriorValueRestrictMode3D restrict_mode_;
};

struct ExteriorZeroTraceSolution3D {
    Eigen::VectorXd value_jump;
    Eigen::VectorXd potential;
    Eigen::MatrixXd coefficients;
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
    int max_iterations,
    ExteriorValueRestrictMode3D restrict_mode =
        ExteriorValueRestrictMode3D::JointTricubicCauchy,
    const app3d::NeumannEdgeContinuityProjector3D*
        edge_projector = nullptr)
{
    if (edge_projector == nullptr) {
        ExteriorZeroTraceOperator3D op(pipeline, restrict_mode);
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
        const HarmonicJetField3D field = pipeline.evaluate(
            result.value_jump, prescribed_normal_jump);
        result.potential = field.potential;
        result.coefficients = field.coefficients;
        Eigen::VectorXd applied;
        op.apply(augmented_unknown, applied);
        result.augmented_residual = applied - rhs;
        return result;
    }

    ExteriorZeroTraceOperator3D op(pipeline, restrict_mode);
    const app3d::NeumannEdgeProjectedAugmentedOperator3D projected_op(
        op, *edge_projector);
    const Eigen::VectorXd rhs = projected_op.project_right_hand_side(
        op.right_hand_side(prescribed_normal_jump));
    Eigen::VectorXd augmented_unknown = Eigen::VectorXd::Zero(op.problem_size());
    GMRES gmres(max_iterations, tolerance, restart);
    ExteriorZeroTraceSolution3D result;
    result.iterations = gmres.solve(projected_op, rhs, augmented_unknown);
    result.converged = gmres.converged();
    result.gmres_residuals = gmres.residuals();
    const int size = pipeline.surface_size();
    result.value_jump = edge_projector->project(
        augmented_unknown.head(size));
    augmented_unknown.head(size) = result.value_jump;
    result.lagrange_multiplier = augmented_unknown[size];
    const HarmonicJetField3D field = pipeline.evaluate(
        result.value_jump, prescribed_normal_jump);
    result.potential = field.potential;
    result.coefficients = field.coefficients;
    Eigen::VectorXd applied;
    projected_op.apply(augmented_unknown, applied);
    result.augmented_residual = applied - rhs;
    return result;
}

class ExteriorNormalTraceOperator3D final : public IKFBIOperator {
public:
    ExteriorNormalTraceOperator3D(
        const PanelCenterHarmonicJetKFBI3D& pipeline,
        ExteriorNormalRestrictMode3D mode)
        : pipeline_(pipeline)
        , mode_(mode)
    {}

    int problem_size() const override
    {
        return pipeline_.surface_size();
    }

    void apply(const Eigen::VectorXd& normal_jump,
               Eigen::VectorXd& result) const override
    {
        if (normal_jump.size() != problem_size()) {
            throw std::invalid_argument(
                "exterior-normal operator input has wrong size");
        }
        const Eigen::VectorXd zero_value =
            Eigen::VectorXd::Zero(problem_size());
        const HarmonicJetField3D field =
            pipeline_.evaluate(zero_value, normal_jump);
        result = pipeline_.exterior_normal_trace(
            field, zero_value, normal_jump, mode_);
    }

    Eigen::VectorXd right_hand_side(
        const Eigen::VectorXd& prescribed_value_jump) const
    {
        if (prescribed_value_jump.size() != problem_size()) {
            throw std::invalid_argument(
                "prescribed Dirichlet data has wrong size");
        }
        const Eigen::VectorXd zero_normal =
            Eigen::VectorXd::Zero(problem_size());
        const HarmonicJetField3D field =
            pipeline_.evaluate(prescribed_value_jump, zero_normal);
        return -pipeline_.exterior_normal_trace(
            field, prescribed_value_jump, zero_normal, mode_);
    }

private:
    const PanelCenterHarmonicJetKFBI3D& pipeline_;
    ExteriorNormalRestrictMode3D mode_;
};

struct ExteriorNormalTraceSolution3D {
    Eigen::VectorXd normal_jump;
    Eigen::VectorXd potential;
    Eigen::MatrixXd coefficients;
    Eigen::VectorXd operator_residual;
    std::vector<double> gmres_residuals;
    int iterations = 0;
    bool converged = false;
};

ExteriorNormalTraceSolution3D solve_exterior_zero_normal_dirichlet_3d(
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const Eigen::VectorXd& prescribed_value_jump,
    ExteriorNormalRestrictMode3D mode,
    double tolerance,
    int restart,
    int max_iterations)
{
    ExteriorNormalTraceOperator3D op(pipeline, mode);
    const Eigen::VectorXd rhs = op.right_hand_side(prescribed_value_jump);
    Eigen::VectorXd normal_jump = Eigen::VectorXd::Zero(op.problem_size());
    GMRES gmres(max_iterations, tolerance, restart);
    ExteriorNormalTraceSolution3D result;
    result.iterations = gmres.solve(op, rhs, normal_jump);
    result.converged = gmres.converged();
    result.gmres_residuals = gmres.residuals();
    result.normal_jump = normal_jump;
    const HarmonicJetField3D field =
        pipeline.evaluate(prescribed_value_jump, normal_jump);
    result.potential = field.potential;
    result.coefficients = field.coefficients;
    Eigen::VectorXd applied;
    op.apply(normal_jump, applied);
    result.operator_residual = applied - rhs;
    return result;
}

double vector_linf(const Eigen::VectorXd& values)
{
    return values.size() == 0 ? 0.0 : values.lpNorm<Eigen::Infinity>();
}

double vector_rms(const Eigen::VectorXd& values)
{
    return values.size() == 0
        ? 0.0
        : std::sqrt(values.squaredNorm() / static_cast<double>(values.size()));
}

struct ConditionStatistics3D {
    double median = 0.0;
    double p95 = 0.0;
    double maximum = 0.0;
};

ConditionStatistics3D summarize_conditions(std::vector<double> values)
{
    if (values.empty())
        throw std::invalid_argument("Cauchy condition list is empty");
    if (!std::all_of(values.begin(), values.end(), [](double value) {
            return std::isfinite(value) && value > 0.0;
        })) {
        throw std::runtime_error("Cauchy condition list contains invalid values");
    }
    std::sort(values.begin(), values.end());
    ConditionStatistics3D result;
    result.median = values[(values.size() - 1) / 2];
    const std::size_t p95_index = std::min(
        values.size() - 1,
        static_cast<std::size_t>(
            std::ceil(0.95 * static_cast<double>(values.size()))) - 1);
    result.p95 = values[p95_index];
    result.maximum = values.back();
    return result;
}

double surface_weighted_mean(const SurfaceDofCloud& surface,
                             const Eigen::VectorXd& values)
{
    if (values.size() != static_cast<int>(surface.dofs.size()))
        throw std::invalid_argument("surface weighted mean received wrong size");
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    for (int q = 0; q < values.size(); ++q) {
        const double weight = surface.dofs[static_cast<std::size_t>(q)].weight;
        weighted_sum += weight * values[q];
        weight_sum += weight;
    }
    return weighted_sum / weight_sum;
}

struct NeumannManufacturedData3D {
    Eigen::VectorXd prescribed_normal_jump;
    Eigen::VectorXd exact_density;
    double density_mean_shift = 0.0;
};

NeumannManufacturedData3D make_neumann_manufactured_data_3d(
    const SurfaceDofCloud& surface,
    const app3d::RigidTransform3D& transform)
{
    const int size = static_cast<int>(surface.dofs.size());
    NeumannManufacturedData3D result;
    result.prescribed_normal_jump.resize(size);
    result.exact_density.resize(size);
    for (int q = 0; q < size; ++q) {
        const SurfaceDof& dof = surface.dofs[static_cast<std::size_t>(q)];
        result.exact_density[q] =
            app3d::transformed_manufactured_harmonic_value_3d(
                transform, dof.point);
        result.prescribed_normal_jump[q] =
            app3d::transformed_manufactured_harmonic_gradient_3d(
                transform, dof.point).dot(dof.normal);
    }
    result.density_mean_shift = surface_weighted_mean(
        surface, result.exact_density);
    result.exact_density.array() -= result.density_mean_shift;
    result.prescribed_normal_jump.array() -= surface_weighted_mean(
        surface, result.prescribed_normal_jump);
    if (!result.exact_density.allFinite()
        || !result.prescribed_normal_jump.allFinite()
        || !std::isfinite(result.density_mean_shift)) {
        throw std::runtime_error("Neumann manufactured data are non-finite");
    }
    return result;
}

Eigen::VectorXd make_common_neumann_augmented_rhs_3d(
    const NativeNurbsSurface3D& native_surface,
    const SurfaceDofCloud& surface)
{
    const int size = static_cast<int>(surface.dofs.size());
    Eigen::VectorXd result = Eigen::VectorXd::Zero(size + 1);
    const double pi = std::acos(-1.0);
    for (int q = 0; q < size; ++q) {
        const SurfaceDof& dof = surface.dofs[static_cast<std::size_t>(q)];
        if (dof.patch_id < 0
            || dof.patch_id >= static_cast<int>(native_surface.patches.size())) {
            throw std::invalid_argument(
                "common Neumann RHS has an invalid patch ID");
        }
        const auto& patch = native_surface.patches[
            static_cast<std::size_t>(dof.patch_id)];
        const double u_length = patch.domain_end_u() - patch.domain_start_u();
        const double v_length = patch.domain_end_v() - patch.domain_start_v();
        if (!(u_length > 0.0) || !(v_length > 0.0)) {
            throw std::invalid_argument(
                "common Neumann RHS has a degenerate native domain");
        }
        const double normalized_u =
            (dof.u - patch.domain_start_u()) / u_length;
        const double normalized_v =
            (dof.v - patch.domain_start_v()) / v_length;
        const double parameter_tolerance =
            64.0 * std::numeric_limits<double>::epsilon();
        if (normalized_u < -parameter_tolerance
            || normalized_u > 1.0 + parameter_tolerance
            || normalized_v < -parameter_tolerance
            || normalized_v > 1.0 + parameter_tolerance) {
            throw std::invalid_argument(
                "common Neumann RHS has an out-of-domain native parameter");
        }
        const double uhat = std::clamp(normalized_u, 0.0, 1.0);
        const double vhat = std::clamp(normalized_v, 0.0, 1.0);
        const double patch_phase = static_cast<double>(dof.patch_id + 1);
        result[q] = std::sin(2.0 * pi * uhat + 0.37 * patch_phase)
            + 0.5 * std::cos(2.0 * pi * vhat - 0.23 * patch_phase)
            + 0.25 * std::sin(2.0 * pi * (uhat + vhat));
    }
    result.head(size).array() -= surface_weighted_mean(
        surface, result.head(size));
    double weighted_square_sum = 0.0;
    double weight_sum = 0.0;
    for (int q = 0; q < size; ++q) {
        const double weight = surface.dofs[static_cast<std::size_t>(q)].weight;
        weighted_square_sum += weight * result[q] * result[q];
        weight_sum += weight;
    }
    const double weighted_rms = std::sqrt(weighted_square_sum / weight_sum);
    if (!(weighted_rms > 0.0) || !std::isfinite(weighted_rms))
        throw std::runtime_error("common Neumann RHS has invalid weighted RMS");
    result.head(size) /= weighted_rms;
    return result;
}

SolveMetrics3D run_neumann_case(
    const CartesianGrid3D& grid,
    const GridPair3D& grid_pair,
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const app3d::RigidTransform3D& transform,
    int gmres_max_iterations,
    ExteriorValueRestrictMode3D restrict_mode =
        ExteriorValueRestrictMode3D::JointTricubicCauchy,
    std::vector<double>* residual_history = nullptr,
    const app3d::NeumannEdgeContinuityProjector3D*
        edge_projector = nullptr,
    Eigen::VectorXd* solved_value_jump = nullptr,
    Eigen::MatrixXd* solved_coefficients = nullptr,
    Eigen::VectorXd* prescribed_normal_jump_output = nullptr,
    Eigen::VectorXd* prescribed_exact_trace_output = nullptr,
    const Eigen::VectorXd* shared_exact_trace_input = nullptr,
    const Eigen::VectorXd* shared_normal_jump_input = nullptr)
{
    const int size = pipeline.surface_size();
    if ((shared_exact_trace_input == nullptr)
        != (shared_normal_jump_input == nullptr)) {
        throw std::invalid_argument(
            "shared Neumann exact trace and normal jump must be provided together");
    }
    Eigen::VectorXd exact_trace_storage;
    Eigen::VectorXd normal_data_storage;
    if (shared_exact_trace_input == nullptr) {
        const NeumannManufacturedData3D manufactured =
            make_neumann_manufactured_data_3d(pipeline.surface(), transform);
        exact_trace_storage = manufactured.exact_density;
        normal_data_storage = manufactured.prescribed_normal_jump;
    } else if (shared_exact_trace_input->size() != size
               || shared_normal_jump_input->size() != size
               || !shared_exact_trace_input->allFinite()
               || !shared_normal_jump_input->allFinite()) {
        throw std::invalid_argument(
            "shared Neumann manufactured data has invalid size or values");
    }
    const Eigen::VectorXd& exact_trace = shared_exact_trace_input == nullptr
        ? exact_trace_storage : *shared_exact_trace_input;
    const Eigen::VectorXd& normal_data = shared_normal_jump_input == nullptr
        ? normal_data_storage : *shared_normal_jump_input;

    const auto solve_start = std::chrono::steady_clock::now();
    const ExteriorZeroTraceSolution3D solution =
        solve_exterior_zero_trace_neumann_3d(
            pipeline, normal_data, 2.0e-10, 80, gmres_max_iterations,
            restrict_mode, edge_projector);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();

    const HarmonicJetField3D field{
        solution.potential, solution.coefficients};
    const Eigen::VectorXd direct_exterior = pipeline.exterior_trace(
        field, solution.value_jump, normal_data, restrict_mode);
    const Eigen::VectorXd exterior_from_jump = pipeline.interior_trace(
        field, solution.value_jump, normal_data, restrict_mode)
        - solution.value_jump;

    SolveMetrics3D result;
    result.formulation = "neumann_exterior_zero_value_trace";
    result.iterations = solution.iterations;
    result.converged = solution.converged;
    result.seconds = seconds;
    result.gmres_relative_residual = solution.gmres_residuals.empty()
        ? std::numeric_limits<double>::quiet_NaN()
        : solution.gmres_residuals.back();
    result.operator_residual_linf = vector_linf(
        solution.augmented_residual.head(size));
    result.exterior_condition_linf = vector_linf(direct_exterior);
    result.route_mismatch_linf = vector_linf(
        direct_exterior - exterior_from_jump);
    result.data_weighted_mean = surface_weighted_mean(
        pipeline.surface(), normal_data);
    result.density_weighted_mean = surface_weighted_mean(
        pipeline.surface(), solution.value_jump);

    const Eigen::VectorXd density_error = solution.value_jump - exact_trace;
    result.density_linf = vector_linf(density_error);
    result.density_l2 = vector_rms(density_error);

    double shift_sum = 0.0;
    int interior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (grid_pair.domain_label(node) <= 0)
            continue;
        shift_sum += app3d::transformed_manufactured_harmonic_value_3d(
                         transform, grid_point(grid, node))
                   - solution.potential[node];
        ++interior_count;
    }
    result.constant_shift = shift_sum / static_cast<double>(interior_count);
    double interior_error_sq = 0.0;
    double exterior_error_sq = 0.0;
    int exterior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (grid_pair.domain_label(node) > 0) {
            const double error = solution.potential[node]
                               + result.constant_shift
                               - app3d::transformed_manufactured_harmonic_value_3d(
                                     transform, grid_point(grid, node));
            result.interior_linf = std::max(
                result.interior_linf, std::abs(error));
            interior_error_sq += error * error;
        } else {
            const double error = solution.potential[node];
            result.exterior_bulk_linf = std::max(
                result.exterior_bulk_linf, std::abs(error));
            exterior_error_sq += error * error;
            ++exterior_count;
        }
    }
    result.interior_l2 = std::sqrt(
        interior_error_sq / static_cast<double>(interior_count));
    result.exterior_bulk_l2 = std::sqrt(
        exterior_error_sq / static_cast<double>(exterior_count));
    if (residual_history != nullptr)
        *residual_history = solution.gmres_residuals;
    if (solved_value_jump != nullptr)
        *solved_value_jump = solution.value_jump;
    if (solved_coefficients != nullptr)
        *solved_coefficients = solution.coefficients;
    if (prescribed_normal_jump_output != nullptr)
        *prescribed_normal_jump_output = normal_data;
    if (prescribed_exact_trace_output != nullptr)
        *prescribed_exact_trace_output = exact_trace;
    return result;
}

SolveMetrics3D run_dirichlet_normal_case(
    const CartesianGrid3D& grid,
    const GridPair3D& grid_pair,
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const app3d::RigidTransform3D& transform,
    int gmres_max_iterations,
    ExteriorNormalRestrictMode3D mode =
        ExteriorNormalRestrictMode3D::JointTricubicCauchy,
    std::vector<double>* residual_history = nullptr)
{
    const int size = pipeline.surface_size();
    Eigen::VectorXd value_data(size);
    Eigen::VectorXd exact_normal(size);
    for (int q = 0; q < size; ++q) {
        const SurfaceDof& dof =
            pipeline.surface().dofs[static_cast<std::size_t>(q)];
        value_data[q] = app3d::transformed_manufactured_harmonic_value_3d(
            transform, dof.point);
        exact_normal[q] =
            app3d::transformed_manufactured_harmonic_gradient_3d(
                transform, dof.point).dot(dof.normal);
    }

    const auto solve_start = std::chrono::steady_clock::now();
    const ExteriorNormalTraceSolution3D solution =
        solve_exterior_zero_normal_dirichlet_3d(
            pipeline, value_data,
            mode, 2.0e-10, 0, gmres_max_iterations);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();

    const HarmonicJetField3D field{
        solution.potential, solution.coefficients};
    const Eigen::VectorXd direct_exterior_normal =
        pipeline.exterior_normal_trace(
            field, value_data, solution.normal_jump, mode);
    const Eigen::VectorXd exterior_normal_from_jump =
        pipeline.interior_normal_trace(
            field, value_data, solution.normal_jump) - solution.normal_jump;
    const Eigen::VectorXd boundary_residual = pipeline.interior_trace(
        field, value_data, solution.normal_jump) - value_data;

    SolveMetrics3D result;
    result.formulation = "dirichlet_exterior_zero_normal_trace";
    result.iterations = solution.iterations;
    result.converged = solution.converged;
    result.seconds = seconds;
    result.gmres_relative_residual = solution.gmres_residuals.empty()
        ? std::numeric_limits<double>::quiet_NaN()
        : solution.gmres_residuals.back();
    result.operator_residual_linf = vector_linf(solution.operator_residual);
    result.exterior_condition_linf = vector_linf(direct_exterior_normal);
    result.boundary_residual_linf = vector_linf(boundary_residual);
    result.route_mismatch_linf = vector_linf(
        direct_exterior_normal - exterior_normal_from_jump);
    result.data_weighted_mean = surface_weighted_mean(
        pipeline.surface(), value_data);
    result.density_weighted_mean = surface_weighted_mean(
        pipeline.surface(), solution.normal_jump);

    const Eigen::VectorXd density_error = solution.normal_jump - exact_normal;
    result.density_linf = vector_linf(density_error);
    result.density_l2 = vector_rms(density_error);

    double interior_error_sq = 0.0;
    double exterior_error_sq = 0.0;
    int interior_count = 0;
    int exterior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (grid_pair.domain_label(node) > 0) {
            const double error = solution.potential[node]
                               - app3d::transformed_manufactured_harmonic_value_3d(
                                     transform, grid_point(grid, node));
            result.interior_linf = std::max(
                result.interior_linf, std::abs(error));
            interior_error_sq += error * error;
            ++interior_count;
        } else {
            const double error = solution.potential[node];
            result.exterior_bulk_linf = std::max(
                result.exterior_bulk_linf, std::abs(error));
            exterior_error_sq += error * error;
            ++exterior_count;
        }
    }
    result.interior_l2 = std::sqrt(
        interior_error_sq / static_cast<double>(interior_count));
    result.exterior_bulk_l2 = std::sqrt(
        exterior_error_sq / static_cast<double>(exterior_count));
    if (residual_history != nullptr)
        *residual_history = solution.gmres_residuals;
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
                              const app3d::HarmonicCauchyFit3D& fit)
{
    std::filesystem::create_directories(output_dir);
    const std::string stem = geometry_name + "_N" + std::to_string(N);

    std::ofstream patch_csv = open_output_file(
        output_dir / (stem + "_surface_patches.csv"));
    patch_csv << "patch_id,patch_name,nu,nv,smooth_patch_ids\n";
    for (int patch_id = 0;
         patch_id < static_cast<int>(cloud.patches.size()); ++patch_id) {
        const SurfacePatchInfo& patch =
            cloud.patches[static_cast<std::size_t>(patch_id)];
        patch_csv << patch_id << ',' << patch.name << ','
                  << patch.nu << ',' << patch.nv << ',';
        for (std::size_t j = 0; j < patch.smooth_patch_ids.size(); ++j) {
            if (j != 0)
                patch_csv << ';';
            patch_csv << patch.smooth_patch_ids[j];
        }
        patch_csv << '\n';
    }

    std::ofstream dof_csv = open_output_file(
        output_dir / (stem + "_surface_panel_center_dofs.csv"));
    dof_csv << std::setprecision(17);
    dof_csv << "q,patch_id,patch_name,i,j,u,v,"
               "x,y,z,nx,ny,nz,t1x,t1y,t1z,t2x,t2y,t2z,weight\n";
    for (int q = 0; q < static_cast<int>(cloud.dofs.size()); ++q) {
        const SurfaceDof& dof = cloud.dofs[static_cast<std::size_t>(q)];
        dof_csv << q << ',' << dof.patch_id << ','
                << cloud.patches[static_cast<std::size_t>(dof.patch_id)].name
                << ',' << dof.i << ',' << dof.j << ','
                << dof.u << ',' << dof.v << ','
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
    stencil_csv << "q,patch_id,radius_over_h,incident_patch_count,"
                   "value_count,normal_count,value_patch_imbalance,"
                   "normal_patch_imbalance";
    for (int k = 0; k < fit.value_count(); ++k)
        stencil_csv << ",value_" << k;
    for (int k = 0; k < fit.normal_count(); ++k)
        stencil_csv << ",normal_" << k;
    stencil_csv << '\n';
    for (int q = 0; q < static_cast<int>(fit.surface_maps().size()); ++q) {
        const app3d::SurfaceCauchyMap3D& stencil =
            fit.surface_maps()[static_cast<std::size_t>(q)];
        stencil_csv << q << ','
                    << cloud.dofs[static_cast<std::size_t>(q)].patch_id << ','
                    << std::max(stencil.value_radius_over_h,
                                stencil.normal_radius_over_h) << ','
                    << stencil.incident_patch_count << ','
                    << stencil.value_ids.size() << ','
                    << stencil.normal_ids.size() << ','
                    << stencil.value_patch_imbalance << ','
                    << stencil.normal_patch_imbalance;
        for (int id : stencil.value_ids)
            stencil_csv << ',' << id;
        for (int k = static_cast<int>(stencil.value_ids.size());
             k < fit.value_count(); ++k) {
            stencil_csv << ',';
        }
        for (int id : stencil.normal_ids)
            stencil_csv << ',' << id;
        for (int k = static_cast<int>(stencil.normal_ids.size());
             k < fit.normal_count(); ++k) {
            stencil_csv << ',';
        }
        stencil_csv << '\n';
    }
}

ReadinessResult run_readiness_case(GeometryKind kind,
                                   int N,
                                   const std::filesystem::path& output_dir,
                                   app3d::LegacySurfaceCauchyPolicy3D cauchy_policy,
                                   int cauchy_value_count,
                                   int cauchy_normal_count,
                                   int gmres_max_iterations,
                                   const app3d::RigidTransform3D& transform,
                                   SolveSelection3D solve_selection)
{
    const double h = kBoxSide / static_cast<double>(N);
    CartesianGrid3D grid({kBoxMin, kBoxMin, kBoxMin},
                         {h, h, h},
                         {N, N, N},
                         DofLayout3D::Node);
    GeometryBundle geometry = make_geometry(kind, h, transform);
    const auto domain = std::make_shared<const
        geometry3d::NurbsCartesianDomain3D>(
            grid, geometry.native_surface.geometry_model());
    const SurfaceDofCloud surface_dofs =
        app3d::make_native_surface_dofs_3d(geometry.native_surface, h);
    const SurfaceCloudDiagnostics surface_diagnostics =
        validate_surface_dofs(surface_dofs, h);
    app3d::HarmonicCauchyFit3D cauchy_fit =
        app3d::HarmonicCauchyFit3D::build_legacy(
            geometry.native_surface, surface_dofs, h, cauchy_policy,
            kCauchyPolynomialDegree, cauchy_value_count,
            cauchy_normal_count);
    const app3d::LegacyCauchySummary3D cauchy_summary =
        *cauchy_fit.legacy_summary();
    if (solve_selection == SolveSelection3D::Both) {
        write_surface_files(output_dir, geometry, N);
        write_panel_center_files(
            output_dir, geometry.name, N, surface_dofs, cauchy_fit);
    }

    ReadinessResult result;
    result.geometry = geometry.name;
    result.cauchy_policy = cauchy_policy_name(cauchy_policy);
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
    result.cauchy_value_neighbors = cauchy_fit.value_count();
    result.cauchy_derivative_neighbors = cauchy_fit.normal_count();
    result.cauchy_value_neighbors_min = cauchy_summary.value_count_min;
    result.cauchy_value_neighbors_max = cauchy_summary.value_count_max;
    result.cauchy_derivative_neighbors_min =
        cauchy_summary.normal_count_min;
    result.cauchy_derivative_neighbors_max =
        cauchy_summary.normal_count_max;
    result.cauchy_radius_max_over_h = cauchy_summary.radius_max_over_h;
    result.cauchy_radius_mean_over_h = cauchy_summary.radius_mean_over_h;
    result.cauchy_incident_patches_min =
        cauchy_summary.incident_patch_count_min;
    result.cauchy_incident_patches_max =
        cauchy_summary.incident_patch_count_max;
    result.cauchy_value_patch_imbalance_max =
        cauchy_summary.value_patch_imbalance_max;
    result.cauchy_derivative_patch_imbalance_max =
        cauchy_summary.normal_patch_imbalance_max;

    const auto dims = grid.dof_dims();
    Eigen::Vector3d grid_min(kBoxMin, kBoxMin, kBoxMin);
    Eigen::Vector3d grid_max(
        kBoxMin + static_cast<double>(dims[0] - 1) * h,
        kBoxMin + static_cast<double>(dims[1] - 1) * h,
        kBoxMin + static_cast<double>(dims[2] - 1) * h);
    const geometry3d::NurbsAabb3D& surface_bounds = domain->surface_bounds();
    const double min_margin = std::min(
        (surface_bounds.lower - grid_min).minCoeff(),
        (grid_max - surface_bounds.upper).minCoeff());
    result.min_box_margin_over_h = min_margin / h;
    if (result.min_box_margin_over_h < 2.0)
        throw std::runtime_error("surface is too close to the Cartesian box boundary");

    GridPair3D grid_pair(grid,
                         geometry.correction_interface,
                         geometry.crossing_interface,
                         domain);
    const geometry3d::NurbsCartesianDomainDiagnostics3D&
        native_diagnostics = grid_pair.nurbs_domain_diagnostics();
    result.nurbs_patches = native_diagnostics.nurbs_patch_count;
    result.bezier_elements = native_diagnostics.bezier_element_count;
    result.acceleration_leaves = native_diagnostics.acceleration_leaf_count;
    result.maximum_query_element_extent =
        native_diagnostics.maximum_query_element_extent;
    result.candidate_grid_edges =
        native_diagnostics.candidate_grid_edge_count;
    result.triangle_seed_hits =
        native_diagnostics.intersections.triangle_seed_hits;
    result.triangle_seed_misses_recovered =
        native_diagnostics.intersections.triangle_seed_misses_recovered;
    result.subdivision_boxes =
        native_diagnostics.intersections.subdivision_boxes;
    result.newton_attempts =
        native_diagnostics.intersections.newton_attempts;
    result.newton_iterations =
        native_diagnostics.intersections.newton_iterations;
    result.maximum_subdivision_depth =
        native_diagnostics.intersections.maximum_subdivision_depth_reached;
    result.terminal_certificate_boxes =
        native_diagnostics.intersections.terminal_certificate_boxes;
    result.maximum_terminal_certificate_depth =
        native_diagnostics.intersections
            .maximum_terminal_certificate_depth_reached;
    result.closest_point_attempts =
        native_diagnostics.intersections.closest_point_attempts;
    result.closest_point_iterations =
        native_diagnostics.intersections.closest_point_iterations;
    result.closest_point_roots_recovered =
        native_diagnostics.intersections.roots_recovered_by_closest_point;
    result.closest_point_terminal_misses =
        native_diagnostics.intersections.terminal_misses_by_closest_point;
    result.closest_point_failures =
        native_diagnostics.intersections.closest_point_failures;
    result.seam_deduplications =
        native_diagnostics.intersections.seam_deduplications;
    result.sample_seed_candidates =
        native_diagnostics.intersections.sample_seed_candidates;
    result.sample_seeds_accepted =
        native_diagnostics.intersections.sample_seeds_accepted;
    result.sample_seed_roots_recovered =
        native_diagnostics.intersections.roots_recovered_by_sample_seed;
    result.maximum_sample_seeds_per_element =
        native_diagnostics.intersections.maximum_sample_seeds_per_element;
    result.stationary_solve_attempts =
        native_diagnostics.intersections.stationary_solve_attempts;
    result.stationary_solve_converged =
        native_diagnostics.intersections.stationary_solve_converged;
    result.stationary_witnesses =
        native_diagnostics.intersections.stationary_witnesses;
    result.stationary_protected_root_pairs =
        native_diagnostics.intersections
            .root_pairs_protected_by_stationary_witness;
    result.ambiguous_root_clusters =
        native_diagnostics.intersections.ambiguous_root_clusters;
    result.non_g1_topology_merges =
        native_diagnostics.intersections.non_g1_topology_merges;
    result.high_degree_fallbacks =
        native_diagnostics.intersections.high_degree_fallbacks;
    result.interface_x = native_diagnostics.interface_edge_counts[0];
    result.interface_y = native_diagnostics.interface_edge_counts[1];
    result.interface_z = native_diagnostics.interface_edge_counts[2];
    result.multi_crossing_edges =
        native_diagnostics.multi_crossing_edge_count;
    result.even_parity_interface_edges =
        native_diagnostics.even_parity_interface_edge_count;
    result.odd_parity_interface_edges =
        native_diagnostics.odd_parity_interface_edge_count;
    result.ambiguous_parity_edges =
        native_diagnostics.ambiguous_parity_edge_count;
    result.ambiguous_label_changing_edges =
        native_diagnostics.ambiguous_label_changing_edge_count;
    result.targeted_retries =
        native_diagnostics.targeted_retry_count;
    result.targeted_retries_resolved =
        native_diagnostics.targeted_retry_resolved_count;
    result.targeted_retries_unsafe =
        native_diagnostics.targeted_retry_unsafe_count;
    result.correction_safe_edges =
        native_diagnostics.correction_safe_edge_count;
    result.unsafe_label_changing_edges =
        native_diagnostics.unsafe_label_changing_edge_count;
    result.maximum_targeted_retry_subdivision_depth =
        native_diagnostics.targeted_retry_intersections
            .maximum_subdivision_depth_reached;
    result.endpoint_parity_fallbacks =
        native_diagnostics.endpoint_parity_fallback_count;
    result.endpoint_classification_queries =
        native_diagnostics.endpoint_classification_query_count;
    result.component_parity_toggles =
        native_diagnostics.component_parity_toggle_count;
    result.barrier_x = native_diagnostics.barrier_edge_counts[0];
    result.barrier_y = native_diagnostics.barrier_edge_counts[1];
    result.barrier_z = native_diagnostics.barrier_edge_counts[2];
    result.grid_components = native_diagnostics.grid_component_count;
    result.box_exterior_components =
        native_diagnostics.box_exterior_component_count;
    result.representative_queries =
        native_diagnostics.representative_query_count;
    result.nurbs_geometry_tolerance = domain->geometry_tolerance();
    result.nurbs_root_residual_max =
        native_diagnostics.maximum_root_residual;
    if (result.unsafe_label_changing_edges != 0)
        throw std::runtime_error("unsafe native label-changing edge remains");
    if (result.maximum_subdivision_depth > 4)
        throw std::runtime_error("primary NURBS query exceeded depth four");
    if (result.maximum_targeted_retry_subdivision_depth > 6)
        throw std::runtime_error("targeted NURBS retry exceeded depth six");
    if (result.nurbs_root_residual_max
        > result.nurbs_geometry_tolerance) {
        throw std::runtime_error(
            "native NURBS root residual exceeds geometry tolerance");
    }
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
    if (result.label_mismatches != 0)
        throw std::runtime_error("native NURBS grid labels are not exact");

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
    bool all_native_owners_exact = true;
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

        // surface_dof_for_crossing uses legacy triangle/barycentric ownership
        // exactly when the owner has no native NURBS patch.
        if (owner.nurbs_patch_index < 0)
            ++result.triangle_fallback_crossings;
        if (owner.status != P2CrossingOwnerStatus3D::ExactIntersection
            || owner.nurbs_patch_index < 0
            || !owner.nurbs_parameter.allFinite()
            || !owner.crossing_point.allFinite()) {
            all_native_owners_exact = false;
        }
    }
    if (result.gap_crossings != 0)
        throw std::runtime_error(
            "native readiness crossing owner used gap fallback");
    if (result.triangle_fallback_crossings != 0)
        throw std::runtime_error(
            "native readiness crossing owner used legacy triangle/barycentric fallback");
    if (!all_native_owners_exact)
        throw std::runtime_error(
            "native readiness crossing owner is not exact");

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
        grid,
        grid_pair,
        geometry.native_surface,
        geometry.correction_triangles,
        geometry.geometry_triangles,
        surface_dofs,
        std::move(cauchy_fit));
    const ConditionStatistics3D condition_statistics = summarize_conditions(
        harmonic_pipeline.cauchy_condition_values());
    result.cauchy_condition_median = condition_statistics.median;
    result.cauchy_condition_p95 = condition_statistics.p95;
    result.cauchy_condition_max = condition_statistics.maximum;
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
    if (solve_selection == SolveSelection3D::Both) {
        result.neumann = run_neumann_case(
            grid, grid_pair, harmonic_pipeline, transform,
            gmres_max_iterations,
            ExteriorValueRestrictMode3D::JointTricubicCauchy);
    }
    result.dirichlet_normal = run_dirichlet_normal_case(
        grid, grid_pair, harmonic_pipeline, transform,
        gmres_max_iterations);

    std::cout << "[ready] " << geometry.name << " - " << geometry.description << '\n'
              << "domain_label_mode=nurbs_barrier_components barriers="
              << result.barrier_x << '/' << result.barrier_y << '/'
              << result.barrier_z << " components=" << result.grid_components
              << '/' << result.box_exterior_components
              << " queries=" << result.representative_queries
              << " root_residual=" << result.nurbs_root_residual_max
              << " gap_crossings=" << result.gap_crossings
              << " triangle_fallback_crossings="
              << result.triangle_fallback_crossings
              << " correction_safe_edges=" << result.correction_safe_edges
              << " unsafe_label_changing_edges="
              << result.unsafe_label_changing_edges
              << " targeted_retries=" << result.targeted_retries << '/'
              << result.targeted_retries_resolved << '/'
              << result.targeted_retries_unsafe
              << " targeted_retry_depth="
              << result.maximum_targeted_retry_subdivision_depth << '\n'
              << "  interface/barrier="
              << result.interface_x << '/' << result.interface_y << '/'
              << result.interface_z << " / "
              << result.barrier_x << '/' << result.barrier_y << '/'
              << result.barrier_z
              << " multi=" << result.multi_crossing_edges
              << " even/odd=" << result.even_parity_interface_edges << '/'
              << result.odd_parity_interface_edges
              << " ambiguous/label-changing/fallback="
              << result.ambiguous_parity_edges << '/'
              << result.ambiguous_label_changing_edges << '/'
              << result.endpoint_parity_fallbacks
              << " component_toggles=" << result.component_parity_toggles
              << '\n'
              << "  NURBS query elements=" << result.acceleration_leaves
              << " max_extent=" << result.maximum_query_element_extent
              << " local_depth=" << result.maximum_subdivision_depth
              << " certificate_boxes/depth="
              << result.terminal_certificate_boxes << '/'
              << result.maximum_terminal_certificate_depth
              << " closest attempts/iterations/recovered/misses/failures="
              << result.closest_point_attempts << '/'
              << result.closest_point_iterations << '/'
              << result.closest_point_roots_recovered << '/'
              << result.closest_point_terminal_misses << '/'
              << result.closest_point_failures << '\n'
              << "  sample seeds candidates/accepted/recovered/max="
              << result.sample_seed_candidates << '/'
              << result.sample_seeds_accepted << '/'
              << result.sample_seed_roots_recovered << '/'
              << result.maximum_sample_seeds_per_element
              << " stationary attempts/converged/witnesses/protected="
              << result.stationary_solve_attempts << '/'
              << result.stationary_solve_converged << '/'
              << result.stationary_witnesses << '/'
              << result.stationary_protected_root_pairs
              << " clusters/nonG1/high_degree="
              << result.ambiguous_root_clusters << '/'
              << result.non_g1_topology_merges << '/'
              << result.high_degree_fallbacks << '\n'
              << "  panel-center surface patches/dofs="
              << result.surface_patches << '/' << result.surface_dofs
              << " area=" << result.surface_dof_area
              << " area_rel_err=" << result.surface_area_relative_error << '\n'
              << "  nearest surface spacing/h min/mean/max="
              << result.surface_spacing_min_over_h << '/'
              << result.surface_spacing_mean_over_h << '/'
              << result.surface_spacing_max_over_h << '\n'
              << "  Cauchy policy=" << result.cauchy_policy
              << " requested values/normals="
              << result.cauchy_value_neighbors << '/'
              << result.cauchy_derivative_neighbors
              << " actual values min/max="
              << result.cauchy_value_neighbors_min << '/'
              << result.cauchy_value_neighbors_max
              << " normals min/max="
              << result.cauchy_derivative_neighbors_min << '/'
              << result.cauchy_derivative_neighbors_max
              << " radius/h mean/max=" << result.cauchy_radius_mean_over_h
              << '/' << result.cauchy_radius_max_over_h
              << " selected patches min/max="
              << result.cauchy_incident_patches_min << '/'
              << result.cauchy_incident_patches_max
              << " imbalance value/normal max="
              << result.cauchy_value_patch_imbalance_max << '/'
              << result.cauchy_derivative_patch_imbalance_max << '\n'
              << "  Cauchy condition median/p95/max="
              << result.cauchy_condition_median << '/'
              << result.cauchy_condition_p95 << '/'
              << result.cauchy_condition_max << '\n'
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
    if (solve_selection == SolveSelection3D::Both) {
        std::cout << "  [neumann] converged/iterations="
              << result.neumann.converged << '/' << result.neumann.iterations
              << " gmres=" << result.neumann.gmres_relative_residual
              << " operator=" << result.neumann.operator_residual_linf
              << " exterior_trace=" << result.neumann.exterior_condition_linf
              << " route_mismatch=" << result.neumann.route_mismatch_linf
              << " density_linf/l2=" << result.neumann.density_linf << '/'
              << result.neumann.density_l2
              << " interior_linf/l2=" << result.neumann.interior_linf << '/'
              << result.neumann.interior_l2
              << " exterior_bulk_linf/l2="
              << result.neumann.exterior_bulk_linf << '/'
              << result.neumann.exterior_bulk_l2
              << " flux_mean=" << result.neumann.data_weighted_mean
              << " seconds=" << result.neumann.seconds << '\n';
    }
    std::cout << "  [dirichlet-normal] converged/iterations="
              << result.dirichlet_normal.converged << '/'
              << result.dirichlet_normal.iterations
              << " gmres="
              << result.dirichlet_normal.gmres_relative_residual
              << " operator="
              << result.dirichlet_normal.operator_residual_linf
              << " exterior_normal="
              << result.dirichlet_normal.exterior_condition_linf
              << " interior_value_res="
              << result.dirichlet_normal.boundary_residual_linf
              << " route_mismatch="
              << result.dirichlet_normal.route_mismatch_linf
              << " density_linf/l2="
              << result.dirichlet_normal.density_linf << '/'
              << result.dirichlet_normal.density_l2
              << " interior_linf/l2="
              << result.dirichlet_normal.interior_linf << '/'
              << result.dirichlet_normal.interior_l2
              << " exterior_bulk_linf/l2="
              << result.dirichlet_normal.exterior_bulk_linf << '/'
              << result.dirichlet_normal.exterior_bulk_l2
              << " seconds=" << result.dirichlet_normal.seconds << '\n';
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
        csv << "geometry,cauchy_policy,N,h,correction_panels,correction_dofs,crossing_panels,"
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
               "cauchy_derivative_neighbors,cauchy_value_neighbors_min,"
               "cauchy_value_neighbors_max,cauchy_derivative_neighbors_min,"
               "cauchy_derivative_neighbors_max,cauchy_radius_max_over_h,"
               "cauchy_radius_mean_over_h,cauchy_incident_patches_min,"
               "cauchy_incident_patches_max,cauchy_value_patch_imbalance_max,"
               "cauchy_derivative_patch_imbalance_max,cauchy_condition_median,"
               "cauchy_condition_p95,cauchy_condition_max,nurbs_patches,"
               "bezier_elements,acceleration_leaves,maximum_query_element_extent,"
               "candidate_grid_edges,"
               "triangle_seed_hits,triangle_seed_misses_recovered,subdivision_boxes,"
               "newton_attempts,newton_iterations,maximum_subdivision_depth,"
               "terminal_certificate_boxes,maximum_terminal_certificate_depth,"
               "closest_point_attempts,closest_point_iterations,"
               "closest_point_roots_recovered,closest_point_terminal_misses,"
               "closest_point_failures,seam_deduplications,"
               "sample_seed_candidates,sample_seeds_accepted,"
               "sample_seed_roots_recovered,maximum_sample_seeds_per_element,"
               "stationary_solve_attempts,stationary_solve_converged,"
               "stationary_witnesses,stationary_protected_root_pairs,"
               "ambiguous_root_clusters,non_g1_topology_merges,"
               "high_degree_fallbacks,interface_x,interface_y,interface_z,"
               "multi_crossing_edges,even_parity_interface_edges,"
               "odd_parity_interface_edges,ambiguous_parity_edges,"
               "ambiguous_label_changing_edges,targeted_retries,"
               "targeted_retries_resolved,targeted_retries_unsafe,"
               "correction_safe_edges,unsafe_label_changing_edges,"
               "maximum_targeted_retry_subdivision_depth,"
               "endpoint_parity_fallbacks,endpoint_classification_queries,"
               "component_parity_toggles,"
               "barrier_x,barrier_y,barrier_z,grid_components,"
               "box_exterior_components,representative_queries,"
               "nurbs_geometry_tolerance,nurbs_root_residual_max,"
               "triangle_fallback_crossings\n";
        for (const ReadinessResult& row : rows)
            csv << row.geometry << ',' << row.cauchy_policy << ','
                << row.N << ',' << row.h << ','
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
                << row.cauchy_value_neighbors_min << ','
                << row.cauchy_value_neighbors_max << ','
                << row.cauchy_derivative_neighbors_min << ','
                << row.cauchy_derivative_neighbors_max << ','
                << row.cauchy_radius_max_over_h << ','
                << row.cauchy_radius_mean_over_h << ','
                << row.cauchy_incident_patches_min << ','
                << row.cauchy_incident_patches_max << ','
                << row.cauchy_value_patch_imbalance_max << ','
                << row.cauchy_derivative_patch_imbalance_max << ','
                << row.cauchy_condition_median << ','
                << row.cauchy_condition_p95 << ','
                << row.cauchy_condition_max << ','
                << row.nurbs_patches << ',' << row.bezier_elements << ','
                << row.acceleration_leaves << ','
                << row.maximum_query_element_extent << ','
                << row.candidate_grid_edges << ','
                << row.triangle_seed_hits << ','
                << row.triangle_seed_misses_recovered << ','
                << row.subdivision_boxes << ',' << row.newton_attempts << ','
                << row.newton_iterations << ','
                << row.maximum_subdivision_depth << ','
                << row.terminal_certificate_boxes << ','
                << row.maximum_terminal_certificate_depth << ','
                << row.closest_point_attempts << ','
                << row.closest_point_iterations << ','
                << row.closest_point_roots_recovered << ','
                << row.closest_point_terminal_misses << ','
                << row.closest_point_failures << ','
                << row.seam_deduplications << ','
                << row.sample_seed_candidates << ','
                << row.sample_seeds_accepted << ','
                << row.sample_seed_roots_recovered << ','
                << row.maximum_sample_seeds_per_element << ','
                << row.stationary_solve_attempts << ','
                << row.stationary_solve_converged << ','
                << row.stationary_witnesses << ','
                << row.stationary_protected_root_pairs << ','
                << row.ambiguous_root_clusters << ','
                << row.non_g1_topology_merges << ','
                << row.high_degree_fallbacks << ','
                << row.interface_x << ',' << row.interface_y << ','
                << row.interface_z << ',' << row.multi_crossing_edges << ','
                << row.even_parity_interface_edges << ','
                << row.odd_parity_interface_edges << ','
                << row.ambiguous_parity_edges << ','
                << row.ambiguous_label_changing_edges << ','
                << row.targeted_retries << ','
                << row.targeted_retries_resolved << ','
                << row.targeted_retries_unsafe << ','
                << row.correction_safe_edges << ','
                << row.unsafe_label_changing_edges << ','
                << row.maximum_targeted_retry_subdivision_depth << ','
                << row.endpoint_parity_fallbacks << ','
                << row.endpoint_classification_queries << ','
                << row.component_parity_toggles << ','
                << row.barrier_x << ',' << row.barrier_y << ','
                << row.barrier_z << ',' << row.grid_components << ','
                << row.box_exterior_components << ','
                << row.representative_queries << ','
                << row.nurbs_geometry_tolerance << ','
                << row.nurbs_root_residual_max << ','
                << row.triangle_fallback_crossings << '\n';
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

double observed_order(double coarse_error,
                      double fine_error,
                      double coarse_h,
                      double fine_h)
{
    if (!(coarse_error > 0.0) || !(fine_error > 0.0)
        || !(coarse_h > fine_h)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::log(coarse_error / fine_error)
         / std::log(coarse_h / fine_h);
}

void write_solve_summaries(const std::filesystem::path& output_dir,
                           const std::vector<ReadinessResult>& results)
{
    using MetricSelector =
        std::function<const SolveMetrics3D&(const ReadinessResult&)>;
    auto write_csv = [&](const std::filesystem::path& path,
                         const MetricSelector& select_metric) {
        std::vector<const ReadinessResult*> rows;
        rows.reserve(results.size());
        for (const ReadinessResult& result : results)
            rows.push_back(&result);
        std::sort(rows.begin(), rows.end(),
                  [](const ReadinessResult* a, const ReadinessResult* b) {
                      return std::tie(a->geometry, a->N)
                           < std::tie(b->geometry, b->N);
                  });

        std::ofstream csv = open_output_file(path);
        csv << std::setprecision(17);
        csv << "geometry,cauchy_policy,N,h,dofs,formulation,iterations,converged,seconds,"
               "gmres_relative_residual,operator_residual_linf,"
               "exterior_condition_linf,boundary_residual_linf,"
               "route_mismatch_linf,data_weighted_mean,density_weighted_mean,"
               "density_linf,density_l2,density_order_linf,density_order_l2,"
               "interior_linf,interior_l2,interior_order_linf,"
               "interior_order_l2,exterior_bulk_linf,exterior_bulk_l2,"
               "exterior_bulk_order_linf,exterior_bulk_order_l2,"
               "constant_shift\n";
        std::map<std::string, const ReadinessResult*> previous;
        for (const ReadinessResult* row : rows) {
            const SolveMetrics3D& metric = select_metric(*row);
            double density_order_linf =
                std::numeric_limits<double>::quiet_NaN();
            double density_order_l2 =
                std::numeric_limits<double>::quiet_NaN();
            double interior_order_linf =
                std::numeric_limits<double>::quiet_NaN();
            double interior_order_l2 =
                std::numeric_limits<double>::quiet_NaN();
            double exterior_order_linf =
                std::numeric_limits<double>::quiet_NaN();
            double exterior_order_l2 =
                std::numeric_limits<double>::quiet_NaN();
            const auto found = previous.find(row->geometry);
            if (found != previous.end()) {
                const ReadinessResult& coarse = *found->second;
                const SolveMetrics3D& coarse_metric = select_metric(coarse);
                density_order_linf = observed_order(
                    coarse_metric.density_linf, metric.density_linf,
                    coarse.h, row->h);
                density_order_l2 = observed_order(
                    coarse_metric.density_l2, metric.density_l2,
                    coarse.h, row->h);
                interior_order_linf = observed_order(
                    coarse_metric.interior_linf, metric.interior_linf,
                    coarse.h, row->h);
                interior_order_l2 = observed_order(
                    coarse_metric.interior_l2, metric.interior_l2,
                    coarse.h, row->h);
                exterior_order_linf = observed_order(
                    coarse_metric.exterior_bulk_linf,
                    metric.exterior_bulk_linf, coarse.h, row->h);
                exterior_order_l2 = observed_order(
                    coarse_metric.exterior_bulk_l2,
                    metric.exterior_bulk_l2, coarse.h, row->h);
            }
            csv << row->geometry << ',' << row->cauchy_policy << ','
                << row->N << ',' << row->h << ','
                << row->surface_dofs << ',' << metric.formulation << ','
                << metric.iterations << ',' << metric.converged << ','
                << metric.seconds << ',' << metric.gmres_relative_residual << ','
                << metric.operator_residual_linf << ','
                << metric.exterior_condition_linf << ','
                << metric.boundary_residual_linf << ','
                << metric.route_mismatch_linf << ','
                << metric.data_weighted_mean << ','
                << metric.density_weighted_mean << ','
                << metric.density_linf << ',' << metric.density_l2 << ','
                << density_order_linf << ',' << density_order_l2 << ','
                << metric.interior_linf << ',' << metric.interior_l2 << ','
                << interior_order_linf << ',' << interior_order_l2 << ','
                << metric.exterior_bulk_linf << ','
                << metric.exterior_bulk_l2 << ','
                << exterior_order_linf << ',' << exterior_order_l2 << ','
                << metric.constant_shift << '\n';
            previous[row->geometry] = row;
        }
    };

    write_csv(output_dir / "neumann_results.csv",
              [](const ReadinessResult& row) -> const SolveMetrics3D& {
                  return row.neumann;
              });
    write_csv(output_dir / "dirichlet_normal_results.csv",
              [](const ReadinessResult& row) -> const SolveMetrics3D& {
                  return row.dirichlet_normal;
              });
}

using CriterionStatus3D = app3d::RigidStudyCriterionStatus3D;

const char* criterion_status_name(CriterionStatus3D status)
{
    switch (status) {
    case CriterionStatus3D::Pass:
        return "pass";
    case CriterionStatus3D::Fail:
        return "fail";
    case CriterionStatus3D::NotEvaluated:
        return "not_evaluated";
    }
    throw std::runtime_error("unknown rigid-study criterion status");
}

struct RigidStudyRow3D {
    app3d::DirichletRigidStudyCase3D study_case;
    ReadinessResult readiness;
    double total_seconds = 0.0;
    double observed_order = std::numeric_limits<double>::quiet_NaN();
    double baseline_error_ratio = std::numeric_limits<double>::quiet_NaN();
    double baseline_iteration_ratio = std::numeric_limits<double>::quiet_NaN();
    CriterionStatus3D gmres_pass = CriterionStatus3D::NotEvaluated;
    CriterionStatus3D monotone_error_pass = CriterionStatus3D::NotEvaluated;
    CriterionStatus3D order_64_128_pass = CriterionStatus3D::NotEvaluated;
    CriterionStatus3D baseline_ratio_pass = CriterionStatus3D::NotEvaluated;
    CriterionStatus3D geometry_diagnostics_pass =
        CriterionStatus3D::NotEvaluated;
    CriterionStatus3D overall_pass = CriterionStatus3D::NotEvaluated;
};

struct RigidStudyAcceptance3D {
    std::string case_id;
    CriterionStatus3D gmres_pass = CriterionStatus3D::NotEvaluated;
    CriterionStatus3D monotone_error_pass = CriterionStatus3D::NotEvaluated;
    CriterionStatus3D order_64_128_pass = CriterionStatus3D::NotEvaluated;
    CriterionStatus3D baseline_ratio_pass = CriterionStatus3D::NotEvaluated;
    CriterionStatus3D geometry_diagnostics_pass =
        CriterionStatus3D::NotEvaluated;
    CriterionStatus3D overall_pass = CriterionStatus3D::NotEvaluated;
};

bool rigid_geometry_diagnostics_pass(const ReadinessResult& result)
{
    return result.label_mismatches == 0
        && result.unsafe_label_changing_edges == 0
        && result.gap_crossings == 0
        && result.endpoint_crossings == 0
        && result.triangle_fallback_crossings == 0;
}

bool is_rigid_acceptance_level(int N)
{
    return N == 32 || N == 64 || N == 128;
}

const RigidStudyRow3D* find_rigid_study_row(
    const std::vector<RigidStudyRow3D>& rows,
    const std::string& case_id,
    int N)
{
    const auto found = std::find_if(
        rows.begin(), rows.end(), [&](const RigidStudyRow3D& row) {
            return row.study_case.id == case_id && row.readiness.N == N;
        });
    return found == rows.end() ? nullptr : &*found;
}

double rigid_observed_order(const RigidStudyRow3D& previous,
                            const RigidStudyRow3D& current)
{
    const double previous_error =
        previous.readiness.dirichlet_normal.interior_linf;
    const double current_error =
        current.readiness.dirichlet_normal.interior_linf;
    if (!(previous_error > 0.0) || !(current_error > 0.0)
        || current.readiness.N <= previous.readiness.N) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::log(previous_error / current_error)
         / std::log(static_cast<double>(current.readiness.N)
                    / previous.readiness.N);
}

std::vector<RigidStudyAcceptance3D> update_rigid_study_criteria(
    std::vector<RigidStudyRow3D>& rows,
    const std::vector<app3d::DirichletRigidStudyCase3D>& cases,
    bool require_complete_acceptance)
{
    for (RigidStudyRow3D& row : rows) {
        const RigidStudyRow3D* previous = nullptr;
        for (const RigidStudyRow3D& candidate : rows) {
            if (candidate.study_case.id == row.study_case.id
                && candidate.readiness.N < row.readiness.N
                && (previous == nullptr
                    || candidate.readiness.N > previous->readiness.N)) {
                previous = &candidate;
            }
        }
        row.observed_order = previous == nullptr
            ? std::numeric_limits<double>::quiet_NaN()
            : rigid_observed_order(*previous, row);

        const RigidStudyRow3D* baseline = find_rigid_study_row(
            rows, "baseline", row.readiness.N);
        if (baseline != nullptr
            && baseline->readiness.dirichlet_normal.interior_linf > 0.0) {
            row.baseline_error_ratio =
                row.readiness.dirichlet_normal.interior_linf
                / baseline->readiness.dirichlet_normal.interior_linf;
        }
        if (baseline != nullptr
            && baseline->readiness.dirichlet_normal.iterations > 0) {
            row.baseline_iteration_ratio =
                static_cast<double>(row.readiness.dirichlet_normal.iterations)
                / baseline->readiness.dirichlet_normal.iterations;
        }
        row.gmres_pass = row.readiness.dirichlet_normal.converged
                      && row.readiness.dirichlet_normal.iterations <= 80
            ? CriterionStatus3D::Pass : CriterionStatus3D::Fail;
        if (is_rigid_acceptance_level(row.readiness.N)) {
            row.baseline_ratio_pass = std::isfinite(row.baseline_error_ratio)
                                   && row.baseline_error_ratio <= 3.0
                ? CriterionStatus3D::Pass : CriterionStatus3D::Fail;
        } else {
            row.baseline_ratio_pass = CriterionStatus3D::NotEvaluated;
        }
        row.geometry_diagnostics_pass =
            rigid_geometry_diagnostics_pass(row.readiness)
            ? CriterionStatus3D::Pass : CriterionStatus3D::Fail;
    }

    std::vector<RigidStudyAcceptance3D> acceptance;
    acceptance.reserve(cases.size());
    for (const auto& study_case : cases) {
        std::vector<RigidStudyRow3D*> case_rows;
        for (RigidStudyRow3D& row : rows) {
            if (row.study_case.id == study_case.id)
                case_rows.push_back(&row);
        }

        RigidStudyAcceptance3D item;
        item.case_id = study_case.id;
        if (!case_rows.empty()) {
            const bool gmres_pass = std::all_of(
                case_rows.begin(), case_rows.end(), [](const auto* row) {
                    return row->gmres_pass == CriterionStatus3D::Pass;
                });
            item.gmres_pass = gmres_pass ? CriterionStatus3D::Pass
                                        : CriterionStatus3D::Fail;
            std::vector<RigidStudyRow3D*> acceptance_level_rows;
            std::copy_if(
                case_rows.begin(), case_rows.end(),
                std::back_inserter(acceptance_level_rows), [](const auto* row) {
                    return is_rigid_acceptance_level(row->readiness.N);
                });
            if (!acceptance_level_rows.empty()) {
                const bool baseline_pass = std::all_of(
                    acceptance_level_rows.begin(), acceptance_level_rows.end(),
                    [](const auto* row) {
                        return row->baseline_ratio_pass
                            == CriterionStatus3D::Pass;
                    });
                item.baseline_ratio_pass = baseline_pass
                    ? CriterionStatus3D::Pass : CriterionStatus3D::Fail;
            }
            const bool geometry_pass = std::all_of(
                case_rows.begin(), case_rows.end(), [](const auto* row) {
                    return row->geometry_diagnostics_pass
                        == CriterionStatus3D::Pass;
                });
            item.geometry_diagnostics_pass = geometry_pass
                ? CriterionStatus3D::Pass : CriterionStatus3D::Fail;
        }

        const RigidStudyRow3D* row32 = find_rigid_study_row(
            rows, study_case.id, 32);
        const RigidStudyRow3D* row64 = find_rigid_study_row(
            rows, study_case.id, 64);
        const RigidStudyRow3D* row128 = find_rigid_study_row(
            rows, study_case.id, 128);
        if (row32 != nullptr && row64 != nullptr && row128 != nullptr) {
            const double error32 = row32->readiness.dirichlet_normal.interior_linf;
            const double error64 = row64->readiness.dirichlet_normal.interior_linf;
            const double error128 = row128->readiness.dirichlet_normal.interior_linf;
            item.monotone_error_pass = error32 > error64 && error64 > error128
                ? CriterionStatus3D::Pass : CriterionStatus3D::Fail;
        }
        if (row64 != nullptr && row128 != nullptr) {
            const double order = rigid_observed_order(*row64, *row128);
            item.order_64_128_pass = std::isfinite(order) && order >= 1.8
                ? CriterionStatus3D::Pass : CriterionStatus3D::Fail;
        }
        item.overall_pass = app3d::combine_rigid_study_criteria_3d(
            {item.gmres_pass,
             item.monotone_error_pass,
             item.order_64_128_pass,
             item.baseline_ratio_pass,
             item.geometry_diagnostics_pass},
            !case_rows.empty(),
            require_complete_acceptance);

        for (RigidStudyRow3D* row : case_rows) {
            row->monotone_error_pass = item.monotone_error_pass;
            row->order_64_128_pass = item.order_64_128_pass;
            row->overall_pass = item.overall_pass;
        }
        acceptance.push_back(std::move(item));
    }
    return acceptance;
}

void write_rigid_study_results(
    const std::filesystem::path& output_dir,
    const std::vector<RigidStudyRow3D>& rows)
{
    std::filesystem::create_directories(output_dir);
    std::ofstream csv = open_output_file(
        output_dir / "rigid_transform_results.csv");
    csv << std::setprecision(17) << std::boolalpha;
    csv << "case_id,geometry,cauchy_policy,N,h,"
           "rotation_axis_x,rotation_axis_y,rotation_axis_z,"
           "rotation_angle_degrees,rotation_center_x,rotation_center_y,"
           "rotation_center_z,translation_x,translation_y,translation_z,"
           "rotation_00,rotation_01,rotation_02,rotation_10,rotation_11,"
           "rotation_12,rotation_20,rotation_21,rotation_22,surface_dofs,"
           "interior_linf,interior_l2,observed_order,gmres_converged,"
           "gmres_iterations,gmres_relative_residual,solve_seconds,"
           "total_seconds,operator_residual_linf,exterior_condition_linf,"
           "boundary_residual_linf,route_mismatch_linf,density_linf,"
           "density_l2,exterior_bulk_linf,exterior_bulk_l2,"
           "cauchy_value_neighbors,cauchy_derivative_neighbors,"
           "cauchy_condition_median,cauchy_condition_p95,"
           "cauchy_condition_max,label_mismatches,"
           "unsafe_label_changing_edges,gap_crossings,endpoint_crossings,"
           "triangle_fallback_crossings,targeted_retries_unsafe,"
           "ambiguous_parity_edges,ambiguous_label_changing_edges,"
           "endpoint_parity_fallbacks,baseline_error_ratio,"
           "baseline_iteration_ratio,gmres_pass,monotone_error_pass,"
           "order_64_128_pass,baseline_ratio_pass,"
           "geometry_diagnostics_pass,overall_pass\n";
    for (const RigidStudyRow3D& row : rows) {
        const auto& transform = row.study_case.transform;
        const auto& result = row.readiness;
        const auto& metric = result.dirichlet_normal;
        const Eigen::Matrix3d& rotation = transform.rotation();
        csv << row.study_case.id << ',' << result.geometry << ','
            << result.cauchy_policy << ',' << result.N << ',' << result.h
            << ',' << row.study_case.rotation_axis.x()
            << ',' << row.study_case.rotation_axis.y()
            << ',' << row.study_case.rotation_axis.z()
            << ',' << row.study_case.rotation_angle_degrees
            << ',' << transform.center().x()
            << ',' << transform.center().y()
            << ',' << transform.center().z()
            << ',' << transform.translation().x()
            << ',' << transform.translation().y()
            << ',' << transform.translation().z();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j)
                csv << ',' << rotation(i, j);
        }
        csv << ',' << result.surface_dofs
            << ',' << metric.interior_linf
            << ',' << metric.interior_l2
            << ',' << row.observed_order
            << ',' << metric.converged
            << ',' << metric.iterations
            << ',' << metric.gmres_relative_residual
            << ',' << metric.seconds
            << ',' << row.total_seconds
            << ',' << metric.operator_residual_linf
            << ',' << metric.exterior_condition_linf
            << ',' << metric.boundary_residual_linf
            << ',' << metric.route_mismatch_linf
            << ',' << metric.density_linf
            << ',' << metric.density_l2
            << ',' << metric.exterior_bulk_linf
            << ',' << metric.exterior_bulk_l2
            << ',' << result.cauchy_value_neighbors
            << ',' << result.cauchy_derivative_neighbors
            << ',' << result.cauchy_condition_median
            << ',' << result.cauchy_condition_p95
            << ',' << result.cauchy_condition_max
            << ',' << result.label_mismatches
            << ',' << result.unsafe_label_changing_edges
            << ',' << result.gap_crossings
            << ',' << result.endpoint_crossings
            << ',' << result.triangle_fallback_crossings
            << ',' << result.targeted_retries_unsafe
            << ',' << result.ambiguous_parity_edges
            << ',' << result.ambiguous_label_changing_edges
            << ',' << result.endpoint_parity_fallbacks
            << ',' << row.baseline_error_ratio
            << ',' << row.baseline_iteration_ratio
            << ',' << criterion_status_name(row.gmres_pass)
            << ',' << criterion_status_name(row.monotone_error_pass)
            << ',' << criterion_status_name(row.order_64_128_pass)
            << ',' << criterion_status_name(row.baseline_ratio_pass)
            << ',' << criterion_status_name(row.geometry_diagnostics_pass)
            << ',' << criterion_status_name(row.overall_pass) << '\n';
    }
}

void write_rigid_study_acceptance(
    const std::filesystem::path& output_dir,
    const std::vector<RigidStudyAcceptance3D>& acceptance)
{
    std::filesystem::create_directories(output_dir);
    std::ofstream csv = open_output_file(
        output_dir / "rigid_transform_acceptance.csv");
    csv << "case_id,gmres_pass,monotone_error_pass,order_64_128_pass,"
           "baseline_ratio_pass,geometry_diagnostics_pass,overall_pass\n";
    for (const RigidStudyAcceptance3D& item : acceptance) {
        csv << item.case_id
            << ',' << criterion_status_name(item.gmres_pass)
            << ',' << criterion_status_name(item.monotone_error_pass)
            << ',' << criterion_status_name(item.order_64_128_pass)
            << ',' << criterion_status_name(item.baseline_ratio_pass)
            << ',' << criterion_status_name(item.geometry_diagnostics_pass)
            << ',' << criterion_status_name(item.overall_pass) << '\n';
    }
}

int run_dirichlet_rigid_study(
    std::vector<int> levels,
    app3d::LegacySurfaceCauchyPolicy3D cauchy_policy,
    int cauchy_value_count,
    int cauchy_normal_count,
    int gmres_max_iterations)
{
    if (cauchy_policy != app3d::LegacySurfaceCauchyPolicy3D::G1Nearest
        || cauchy_value_count != kCauchyValueNeighborCount
        || cauchy_normal_count != kCauchyDerivativeNeighborCount) {
        throw std::invalid_argument(
            "--rigid-study requires KFBIM_3D_CAUCHY_POLICY=g1_nearest, "
            "KFBIM_3D_CAUCHY_VALUE_COUNT=48, and "
            "KFBIM_3D_CAUCHY_NORMAL_COUNT=28");
    }
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());

    const bool require_complete_acceptance =
        std::binary_search(levels.begin(), levels.end(), 32)
        && std::binary_search(levels.begin(), levels.end(), 64)
        && std::binary_search(levels.begin(), levels.end(), 128);

#ifdef KFBIM_APP_OUTPUT_DIR
    const std::filesystem::path output_dir =
        std::filesystem::path(KFBIM_APP_OUTPUT_DIR)
        / "dirichlet_rigid_transform_stability_3d";
#else
    const std::filesystem::path output_dir =
        "output/dirichlet_rigid_transform_stability_3d";
#endif
    const std::vector<app3d::DirichletRigidStudyCase3D> cases =
        app3d::make_l_prism_dirichlet_rigid_study_cases_3d();
    std::vector<RigidStudyRow3D> rows;
    rows.reserve(levels.size() * cases.size());

    std::cout << "KFBI3D Dirichlet rigid-transform stability study\n"
              << "  geometry=l_prism formulation=dirichlet_normal_only\n"
              << "  cauchy_policy=" << cauchy_policy_name(cauchy_policy)
              << " cauchy_counts=" << cauchy_value_count << '/'
              << cauchy_normal_count << '\n'
              << "  gmres_max_iterations=" << gmres_max_iterations << '\n'
              << "  levels=";
    for (std::size_t index = 0; index < levels.size(); ++index) {
        if (index != 0)
            std::cout << ',';
        std::cout << levels[index];
    }
    std::cout << " cases=" << cases.size() << '\n';

    for (int N : levels) {
        for (const auto& study_case : cases) {
            const auto case_start = std::chrono::steady_clock::now();
            ReadinessResult result = run_readiness_case(
                GeometryKind::LPrism,
                N,
                output_dir,
                cauchy_policy,
                cauchy_value_count,
                cauchy_normal_count,
                gmres_max_iterations,
                study_case.transform,
                SolveSelection3D::DirichletNormalOnly);
            const double total_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - case_start).count();
            rows.push_back({study_case, std::move(result), total_seconds});
            const std::vector<RigidStudyAcceptance3D> acceptance =
                update_rigid_study_criteria(
                    rows, cases, require_complete_acceptance);
            write_rigid_study_results(output_dir, rows);
            write_rigid_study_acceptance(output_dir, acceptance);

            const RigidStudyRow3D& row = rows.back();
            const SolveMetrics3D& metric = row.readiness.dirichlet_normal;
            std::cout << "[rigid-study] case=" << row.study_case.id
                      << " N=" << row.readiness.N
                      << " dirichlet_converged/iterations="
                      << metric.converged << '/' << metric.iterations
                      << " interior_linf=" << metric.interior_linf
                      << " order=" << row.observed_order
                      << " baseline_error_ratio="
                      << row.baseline_error_ratio
                      << " solve_seconds=" << metric.seconds
                      << " total_seconds=" << row.total_seconds << '\n';
            if (!metric.converged) {
                std::cerr << "error: 3D rigid-study GMRES did not converge: "
                          << "case=" << row.study_case.id
                          << " N=" << row.readiness.N
                          << " iterations=" << metric.iterations
                          << " final_relative_residual="
                          << std::scientific << std::setprecision(17)
                          << metric.gmres_relative_residual << '\n';
                return 1;
            }
        }
    }

    const std::vector<RigidStudyAcceptance3D> acceptance =
        update_rigid_study_criteria(
            rows, cases, require_complete_acceptance);
    write_rigid_study_results(output_dir, rows);
    write_rigid_study_acceptance(output_dir, acceptance);
    bool all_pass = true;
    for (const RigidStudyAcceptance3D& item : acceptance) {
        std::cout << "[rigid-acceptance] case=" << item.case_id
                  << " gmres=" << criterion_status_name(item.gmres_pass)
                  << " monotone="
                  << criterion_status_name(item.monotone_error_pass)
                  << " order_64_128="
                  << criterion_status_name(item.order_64_128_pass)
                  << " baseline_ratio="
                  << criterion_status_name(item.baseline_ratio_pass)
                  << " geometry="
                  << criterion_status_name(item.geometry_diagnostics_pass)
                  << " overall=" << criterion_status_name(item.overall_pass)
                  << '\n';
        all_pass = all_pass && item.overall_pass == CriterionStatus3D::Pass;
    }
    std::cout << "Rigid-transform study output: " << output_dir.string()
              << '\n';
    if (!all_pass) {
        std::cerr << "error: one or more rigid-study acceptance criteria failed\n";
        return 1;
    }
    return 0;
}

struct NormalRestrictNorms3D {
    double linf = 0.0;
    double weighted_rms = 0.0;
};

struct CommonRhsGmresProbe3D {
    bool converged = false;
    int iterations = 0;
    double final_residual = 0.0;
    std::vector<double> residuals;
    Eigen::VectorXd right_hand_side;
    double rhs_weighted_mean = 0.0;
    double rhs_weighted_rms = 0.0;
};

struct EdgeBinMetrics3D {
    std::string bin;
    int count = 0;
    double weight_sum = 0.0;
    double density_linf = 0.0;
    double density_weighted_rms = 0.0;
    double defect_linf = 0.0;
    double defect_weighted_rms = 0.0;
    bool empty = true;
};

struct NeumannRouteProbe3D {
    SolveMetrics3D physical;
    std::vector<double> physical_residuals;
    CommonRhsGmresProbe3D common;
    Eigen::VectorXd density_error;
    Eigen::VectorXd exact_equation_defect;
    Eigen::VectorXd exact_input_edge_values;
    Eigen::VectorXd exact_edge_value_error;
    Eigen::VectorXd exact_edge_quadrature_weights;
    std::optional<double> edge_value_linf;
    std::optional<double> edge_value_weighted_rms;
    double exact_mean_row_defect = 0.0;
    std::array<EdgeBinMetrics3D, 3> bins;
};

double surface_weighted_rms_3d(const SurfaceDofCloud& surface,
                               const Eigen::VectorXd& values)
{
    if (values.size() != static_cast<int>(surface.dofs.size())) {
        throw std::invalid_argument(
            "surface weighted RMS received wrong size");
    }
    double weighted_square_sum = 0.0;
    double weight_sum = 0.0;
    for (int q = 0; q < values.size(); ++q) {
        const double weight = surface.dofs[static_cast<std::size_t>(q)].weight;
        weighted_square_sum += weight * values[q] * values[q];
        weight_sum += weight;
    }
    return std::sqrt(weighted_square_sum / weight_sum);
}

CommonRhsGmresProbe3D run_common_neumann_gmres_3d(
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const Eigen::VectorXd& augmented_rhs,
    ExteriorValueRestrictMode3D restrict_mode)
{
    ExteriorZeroTraceOperator3D op(pipeline, restrict_mode);
    const int size = pipeline.surface_size();
    if (augmented_rhs.size() != op.problem_size()
        || augmented_rhs[size] != 0.0
        || !augmented_rhs.allFinite()) {
        throw std::invalid_argument(
            "common Neumann augmented RHS is invalid");
    }
    CommonRhsGmresProbe3D result;
    result.right_hand_side = augmented_rhs;
    result.rhs_weighted_mean = surface_weighted_mean(
        pipeline.surface(), augmented_rhs.head(size));
    result.rhs_weighted_rms = surface_weighted_rms_3d(
        pipeline.surface(), augmented_rhs.head(size));
    if (std::abs(result.rhs_weighted_mean) > 5.0e-13
        || std::abs(result.rhs_weighted_rms - 1.0) > 5.0e-13) {
        throw std::runtime_error(
            "common Neumann RHS failed weighted mean/RMS audit");
    }
    Eigen::VectorXd unknown = Eigen::VectorXd::Zero(op.problem_size());
    GMRES gmres(80, 2.0e-10, 80);
    result.iterations = gmres.solve(op, augmented_rhs, unknown);
    result.converged = gmres.converged();
    result.residuals = gmres.residuals();
    result.final_residual = result.residuals.empty()
        ? std::numeric_limits<double>::quiet_NaN()
        : result.residuals.back();
    if (result.residuals.size()
        != static_cast<std::size_t>(result.iterations + 1)) {
        throw std::runtime_error(
            "common Neumann GMRES residual history has wrong length");
    }
    return result;
}

NeumannRouteProbe3D run_neumann_route_probe_3d(
    const CartesianGrid3D& grid,
    const GridPair3D& grid_pair,
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const NativeNurbsSurface3D& native_surface,
    const app3d::RigidTransform3D& transform,
    const app3d::SurfaceNonG1EdgeNeighborhoodSet3D& neighborhoods,
    double h,
    app3d::HarmonicCauchyRoute3D route)
{
    const int size = pipeline.surface_size();
    if (!std::isfinite(h) || !(h > 0.0))
        throw std::invalid_argument("Neumann route probe requires positive h");
    const auto& maps = pipeline.cauchy_fit().surface_maps();
    if (maps.size() != neighborhoods.centers.size()
        || maps.size() != static_cast<std::size_t>(size)) {
        throw std::invalid_argument(
            "Neumann route probe requires complete shared neighborhoods");
    }
    for (int q = 0; q < size; ++q) {
        const auto& map = maps[static_cast<std::size_t>(q)];
        const auto& neighborhood =
            neighborhoods.centers[static_cast<std::size_t>(q)];
        if (map.neighborhood_fingerprint != neighborhoods.fingerprint
            || neighborhood.center_dof != q
            || map.nearest_edge_distance_over_h
                != neighborhood.nearest_distance_over_h
            || map.relevant_connection_ids
                != neighborhood.relevant_connection_ids) {
            throw std::logic_error(
                "Neumann route map changed the shared edge neighborhood");
        }
    }

    const NeumannManufacturedData3D data =
        make_neumann_manufactured_data_3d(pipeline.surface(), transform);
    NeumannRouteProbe3D result;
    Eigen::VectorXd solved_density;
    result.physical = run_neumann_case(
        grid, grid_pair, pipeline, transform, 80,
        ExteriorValueRestrictMode3D::JointTricubicCrossingOwner,
        &result.physical_residuals, nullptr, &solved_density,
        nullptr, nullptr, nullptr,
        &data.exact_density, &data.prescribed_normal_jump);
    if (result.physical_residuals.size()
        != static_cast<std::size_t>(result.physical.iterations + 1)) {
        throw std::runtime_error(
            "physical Neumann GMRES residual history has wrong length");
    }
    result.density_error = solved_density - data.exact_density;

    ExteriorZeroTraceOperator3D op(
        pipeline, ExteriorValueRestrictMode3D::JointTricubicCrossingOwner);
    Eigen::VectorXd exact_augmented = Eigen::VectorXd::Zero(op.problem_size());
    exact_augmented.head(size) = data.exact_density;
    Eigen::VectorXd applied;
    op.apply(exact_augmented, applied);
    const Eigen::VectorXd residual =
        applied - op.right_hand_side(data.prescribed_normal_jump);
    result.exact_equation_defect = residual.head(size);
    result.exact_mean_row_defect = residual[size];

    const Eigen::VectorXd common_rhs =
        make_common_neumann_augmented_rhs_3d(
            native_surface, pipeline.surface());
    result.common = run_common_neumann_gmres_3d(
        pipeline, common_rhs,
        ExteriorValueRestrictMode3D::JointTricubicCrossingOwner);

    result.bins[0].bin = "lt_h";
    result.bins[1].bin = "h_to_2h";
    result.bins[2].bin = "gt_2h";
    std::array<double, 3> density_squares{{0.0, 0.0, 0.0}};
    std::array<double, 3> defect_squares{{0.0, 0.0, 0.0}};
    for (int q = 0; q < size; ++q) {
        const double distance = neighborhoods.centers[
            static_cast<std::size_t>(q)].nearest_distance_over_h;
        const int bin = distance < 1.0 ? 0 : distance <= 2.0 ? 1 : 2;
        EdgeBinMetrics3D& metrics =
            result.bins[static_cast<std::size_t>(bin)];
        const double weight =
            pipeline.surface().dofs[static_cast<std::size_t>(q)].weight;
        const double density = result.density_error[q];
        const double defect = result.exact_equation_defect[q];
        metrics.empty = false;
        ++metrics.count;
        metrics.weight_sum += weight;
        metrics.density_linf = std::max(
            metrics.density_linf, std::abs(density));
        metrics.defect_linf = std::max(
            metrics.defect_linf, std::abs(defect));
        density_squares[static_cast<std::size_t>(bin)] +=
            weight * density * density;
        defect_squares[static_cast<std::size_t>(bin)] +=
            weight * defect * defect;
    }
    for (int bin = 0; bin < 3; ++bin) {
        EdgeBinMetrics3D& metrics =
            result.bins[static_cast<std::size_t>(bin)];
        if (metrics.empty)
            continue;
        metrics.density_weighted_rms = std::sqrt(
            density_squares[static_cast<std::size_t>(bin)]
            / metrics.weight_sum);
        metrics.defect_weighted_rms = std::sqrt(
            defect_squares[static_cast<std::size_t>(bin)]
            / metrics.weight_sum);
    }

    if (route == app3d::HarmonicCauchyRoute3D::EdgeReconstructedValue) {
        const auto exact_fit = pipeline.cauchy_fit().apply(
            data.exact_density, data.prescribed_normal_jump);
        const auto& edge_maps = pipeline.cauchy_fit().edge_maps();
        const int edge_count = static_cast<int>(edge_maps.size());
        result.exact_input_edge_values.resize(edge_count);
        result.exact_edge_value_error.resize(edge_count);
        result.exact_edge_quadrature_weights.resize(edge_count);
        double weighted_square_sum = 0.0;
        double weight_sum = 0.0;
        double linf = 0.0;
        for (int q = 0; q < edge_count; ++q) {
            const auto& point = edge_maps[static_cast<std::size_t>(q)].point;
            const double exact =
                app3d::transformed_manufactured_harmonic_value_3d(
                    transform, point.point) - data.density_mean_shift;
            const double error = exact_fit.edge_values[q] - exact;
            result.exact_input_edge_values[q] = exact;
            result.exact_edge_value_error[q] = error;
            result.exact_edge_quadrature_weights[q] = point.quadrature_weight;
            linf = std::max(linf, std::abs(error));
            weighted_square_sum += point.quadrature_weight * error * error;
            weight_sum += point.quadrature_weight;
        }
        if (!(weight_sum > 0.0)) {
            throw std::runtime_error(
                "shared-edge Neumann probe has no edge quadrature weight");
        }
        result.edge_value_linf = linf;
        result.edge_value_weighted_rms =
            std::sqrt(weighted_square_sum / weight_sum);
    } else {
        result.exact_input_edge_values.resize(0);
        result.exact_edge_value_error.resize(0);
        result.exact_edge_quadrature_weights.resize(0);
        result.edge_value_linf.reset();
        result.edge_value_weighted_rms.reset();
    }
    return result;
}

double residual_contraction(
    const std::vector<double>& residuals,
    std::size_t begin,
    std::size_t end)
{
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    if (begin >= end || end >= residuals.size())
        return invalid;
    const double first = residuals[begin];
    const double last = residuals[end];
    if (!std::isfinite(first) || !std::isfinite(last)
        || !(first > 0.0) || last < 0.0) {
        return invalid;
    }
    if (last == 0.0)
        return 0.0;
    return std::exp(
        (std::log(last) - std::log(first))
        / static_cast<double>(end - begin));
}

double worst_five_step_contraction(const std::vector<double>& residuals)
{
    double worst = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t begin = 0; begin + 5 < residuals.size(); ++begin) {
        const double candidate =
            residual_contraction(residuals, begin, begin + 5);
        if (std::isfinite(candidate)
            && (!std::isfinite(worst) || candidate > worst)) {
            worst = candidate;
        }
    }
    return worst;
}

struct NormalRestrictRouteProbe3D {
    bool complete = false;
    std::string route;
    ConditionStatistics3D conditions;
    Eigen::VectorXd exact_grid_error;
    Eigen::VectorXd smooth_grid_error;
    Eigen::VectorXd exact_equation_residual;
    NormalRestrictNorms3D exact_grid_norms;
    NormalRestrictNorms3D smooth_grid_norms;
    NormalRestrictNorms3D exact_equation_norms;
    SolveMetrics3D physical;
    std::vector<double> physical_residuals;
    CommonRhsGmresProbe3D common;
    double physical_rho_10_30 =
        std::numeric_limits<double>::quiet_NaN();
    double physical_worst_rho_5 =
        std::numeric_limits<double>::quiet_NaN();
    double common_rho_10_30 = std::numeric_limits<double>::quiet_NaN();
    double common_worst_rho_5 = std::numeric_limits<double>::quiet_NaN();
    double seconds = 0.0;
};

struct NormalRestrictCaseProbe3D {
    std::string case_id;
    int N = 0;
    double h = 0.0;
    std::vector<int> patch_ids;
    std::vector<Eigen::Vector3d> points;
    std::vector<double> weights;
    std::vector<double> feature_edge_distance_over_h;
    std::vector<int> wrong_side_support_count;
    std::vector<RestrictOwnerSampleDiagnostics3D> owner_diagnostics;
    std::vector<RestrictOwnerAuditRecord3D> owner_audit_records;
    std::size_t owner_geometry_query_count = 0;
    std::uint64_t owner_workload_fingerprint = 0;
    app3d::RestrictOwnerPreprocessDiagnostics3D
        owner_preprocess_diagnostics;
    double setup_seconds = 0.0;
    double pipeline_setup_seconds = 0.0;
    std::array<NormalRestrictRouteProbe3D, 3> routes;
};

const char* normal_restrict_route_name(ExteriorNormalRestrictMode3D mode)
{
    switch (mode) {
    case ExteriorNormalRestrictMode3D::JointTricubicCauchy:
        return "joint_tricubic_cauchy";
    case ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner:
        return "joint_tricubic_crossing_owner";
    case ExteriorNormalRestrictMode3D::ExteriorOnlyHarmonicCubic:
        return "exterior_only_harmonic_cubic";
    }
    throw std::runtime_error("unknown exterior-normal restrict mode");
}

int normal_restrict_route_index(ExteriorNormalRestrictMode3D mode)
{
    switch (mode) {
    case ExteriorNormalRestrictMode3D::JointTricubicCauchy:
        return 0;
    case ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner:
        return 1;
    case ExteriorNormalRestrictMode3D::ExteriorOnlyHarmonicCubic:
        return 2;
    }
    throw std::runtime_error("unknown exterior-normal restrict mode");
}

NormalRestrictNorms3D normal_restrict_weighted_norms(
    const Eigen::VectorXd& values,
    const std::vector<double>& weights,
    const std::vector<int>* selected = nullptr)
{
    if (values.size() != static_cast<int>(weights.size()))
        throw std::invalid_argument("normal-restrict norm size mismatch");
    NormalRestrictNorms3D result;
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    const auto accumulate = [&](int q) {
        const double value = values[q];
        const double weight = weights[static_cast<std::size_t>(q)];
        if (!std::isfinite(value) || !std::isfinite(weight) || weight <= 0.0)
            throw std::runtime_error("normal-restrict norm received invalid data");
        result.linf = std::max(result.linf, std::abs(value));
        weighted_sum += weight * value * value;
        weight_sum += weight;
    };
    if (selected == nullptr) {
        for (int q = 0; q < values.size(); ++q)
            accumulate(q);
    } else {
        for (int q : *selected)
            accumulate(q);
    }
    if (weight_sum > 0.0)
        result.weighted_rms = std::sqrt(weighted_sum / weight_sum);
    return result;
}

double point_segment_distance_3d(
    const Eigen::Vector3d& point,
    const geometry3d::NurbsPatchGeometricEdge3D& edge)
{
    const Eigen::Vector3d direction = edge.end - edge.start;
    const double length_squared = direction.squaredNorm();
    if (!(length_squared > 0.0))
        return (point - edge.start).norm();
    const double phase = std::max(
        0.0, std::min(1.0, (point - edge.start).dot(direction) / length_squared));
    return (point - (edge.start + phase * direction)).norm();
}

Eigen::VectorXd make_common_normal_restrict_rhs(int size)
{
    Eigen::VectorXd rhs(size);
    for (int q = 0; q < size; ++q) {
        const double index = static_cast<double>(q + 1);
        rhs[q] = std::sin(0.73 * index) + 0.25 * std::cos(0.19 * index);
    }
    const double norm = rhs.norm();
    if (!(norm > 0.0) || !std::isfinite(norm))
        throw std::runtime_error("common normal-restrict RHS is invalid");
    rhs /= norm;
    return rhs;
}

CommonRhsGmresProbe3D run_common_normal_restrict_gmres(
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    ExteriorNormalRestrictMode3D mode,
    const Eigen::VectorXd& rhs)
{
    ExteriorNormalTraceOperator3D op(pipeline, mode);
    Eigen::VectorXd unknown = Eigen::VectorXd::Zero(op.problem_size());
    GMRES gmres(160, 2.0e-10, 0);
    CommonRhsGmresProbe3D result;
    result.iterations = gmres.solve(op, rhs, unknown);
    result.converged = gmres.converged();
    result.residuals = gmres.residuals();
    result.final_residual = result.residuals.empty()
        ? 0.0 : result.residuals.back();
    return result;
}

void write_normal_restrict_probe_outputs(
    const std::filesystem::path& output_dir,
    const std::vector<NormalRestrictCaseProbe3D>& cases)
{
    std::filesystem::create_directories(output_dir);
    std::ofstream summary = open_output_file(output_dir / "summary.csv");
    summary << std::setprecision(17)
            << "case_id,N,route,restrict_condition_median,"
               "restrict_condition_p95,restrict_condition_max,"
               "exact_grid_linf,exact_grid_wrms,smooth_grid_linf,"
               "smooth_grid_wrms,exact_equation_linf,exact_equation_wrms,"
               "physical_converged,physical_iterations,"
               "physical_final_residual,physical_interior_linf,"
               "physical_rho_10_30,physical_worst_rho_5,"
               "common_converged,common_iterations,common_final_residual,"
               "common_rho_10_30,common_worst_rho_5,"
               "setup_seconds,pipeline_setup_seconds,"
               "route_seconds\n";
    std::ofstream residuals = open_output_file(
        output_dir / "gmres_residuals.csv");
    residuals << std::setprecision(17)
              << "case_id,N,route,solve_kind,iteration,relative_residual\n";
    for (const NormalRestrictCaseProbe3D& probe_case : cases) {
        for (const NormalRestrictRouteProbe3D& route : probe_case.routes) {
            if (!route.complete)
                continue;
            const double physical_final = route.physical_residuals.empty()
                ? 0.0 : route.physical_residuals.back();
            summary << probe_case.case_id << ',' << probe_case.N << ','
                    << route.route << ',' << route.conditions.median << ','
                    << route.conditions.p95 << ',' << route.conditions.maximum
                    << ',' << route.exact_grid_norms.linf << ','
                    << route.exact_grid_norms.weighted_rms << ','
                    << route.smooth_grid_norms.linf << ','
                    << route.smooth_grid_norms.weighted_rms << ','
                    << route.exact_equation_norms.linf << ','
                    << route.exact_equation_norms.weighted_rms << ','
                    << route.physical.converged << ','
                    << route.physical.iterations << ',' << physical_final << ','
                    << route.physical.interior_linf << ','
                    << route.physical_rho_10_30 << ','
                    << route.physical_worst_rho_5 << ','
                    << route.common.converged << ',' << route.common.iterations
                    << ',' << route.common.final_residual << ','
                    << route.common_rho_10_30 << ','
                    << route.common_worst_rho_5 << ','
                    << probe_case.setup_seconds << ','
                    << probe_case.pipeline_setup_seconds << ','
                    << route.seconds << '\n';
            for (std::size_t k = 0; k < route.physical_residuals.size(); ++k)
                residuals << probe_case.case_id << ',' << probe_case.N << ','
                          << route.route << ",physical," << k << ','
                          << route.physical_residuals[k] << '\n';
            for (std::size_t k = 0; k < route.common.residuals.size(); ++k)
                residuals << probe_case.case_id << ',' << probe_case.N << ','
                          << route.route << ",common," << k << ','
                          << route.common.residuals[k] << '\n';
        }
    }
}

void write_normal_restrict_probe_localization(
    const std::filesystem::path& output_dir,
    const std::vector<NormalRestrictCaseProbe3D>& cases)
{
    std::ofstream dofs = open_output_file(output_dir / "dof_diagnostics.csv");
    dofs << std::setprecision(17)
         << "case_id,N,route,dof,patch_id,x,y,z,weight,"
            "feature_edge_distance_over_h,wrong_side_support_count,"
            "exact_grid_error,smooth_grid_error,"
            "exact_equation_residual\n";
    std::ofstream bins = open_output_file(output_dir / "edge_distance_bins.csv");
    bins << std::setprecision(17)
         << "case_id,N,route,distance_bin,count,"
            "exact_grid_linf,exact_grid_wrms,"
            "exact_equation_linf,exact_equation_wrms\n";
    const std::array<double, 5> upper{{
        1.0, 2.0, 4.0, 8.0,
        std::numeric_limits<double>::infinity()}};
    const std::array<const char*, 5> labels{{
        "0_1", "1_2", "2_4", "4_8", "8_inf"}};
    for (const NormalRestrictCaseProbe3D& probe_case : cases) {
        const int size = static_cast<int>(probe_case.weights.size());
        std::array<std::vector<int>, 5> members;
        for (int q = 0; q < size; ++q) {
            const double distance = probe_case.feature_edge_distance_over_h[
                static_cast<std::size_t>(q)];
            int bin = 0;
            while (bin < 4 && !(distance < upper[static_cast<std::size_t>(bin)]))
                ++bin;
            members[static_cast<std::size_t>(bin)].push_back(q);
        }
        for (const NormalRestrictRouteProbe3D& route : probe_case.routes) {
            if (!route.complete)
                continue;
            for (int q = 0; q < size; ++q) {
                const Eigen::Vector3d& point =
                    probe_case.points[static_cast<std::size_t>(q)];
                dofs << probe_case.case_id << ',' << probe_case.N << ','
                     << route.route << ',' << q << ','
                     << probe_case.patch_ids[static_cast<std::size_t>(q)] << ','
                     << point.x() << ',' << point.y() << ',' << point.z() << ','
                     << probe_case.weights[static_cast<std::size_t>(q)] << ','
                     << probe_case.feature_edge_distance_over_h[
                            static_cast<std::size_t>(q)] << ','
                     << probe_case.wrong_side_support_count[
                            static_cast<std::size_t>(q)] << ','
                     << route.exact_grid_error[q] << ','
                     << route.smooth_grid_error[q] << ','
                     << route.exact_equation_residual[q] << '\n';
            }
            for (int bin = 0; bin < 5; ++bin) {
                const std::vector<int>& selected =
                    members[static_cast<std::size_t>(bin)];
                const NormalRestrictNorms3D grid_norms =
                    normal_restrict_weighted_norms(
                        route.exact_grid_error, probe_case.weights, &selected);
                const NormalRestrictNorms3D equation_norms =
                    normal_restrict_weighted_norms(
                        route.exact_equation_residual,
                        probe_case.weights, &selected);
                bins << probe_case.case_id << ',' << probe_case.N << ','
                     << route.route << ','
                     << labels[static_cast<std::size_t>(bin)] << ','
                     << selected.size() << ','
                     << grid_norms.linf << ',' << grid_norms.weighted_rms << ','
                     << equation_norms.linf << ','
                     << equation_norms.weighted_rms << '\n';
            }
        }
    }
}

const char* restrict_owner_decision_kind_name(std::size_t kind)
{
    switch (static_cast<app3d::RestrictOwnerDecisionKind3D>(kind)) {
    case app3d::RestrictOwnerDecisionKind3D::TargetSideNode:
        return "target_side_node";
    case app3d::RestrictOwnerDecisionKind3D::TargetOrG1SingleCrossing:
        return "target_or_g1_single_crossing";
    case app3d::RestrictOwnerDecisionKind3D::ForeignNonG1SingleCrossing:
        return "foreign_non_g1_single_crossing";
    case app3d::RestrictOwnerDecisionKind3D::NoCrossingFallback:
        return "no_crossing_fallback";
    case app3d::RestrictOwnerDecisionKind3D::MultipleCrossingFallback:
        return "multiple_crossing_fallback";
    case app3d::RestrictOwnerDecisionKind3D::DegenerateCrossingFallback:
        return "degenerate_crossing_fallback";
    case app3d::RestrictOwnerDecisionKind3D::AmbiguousEdgeFallback:
        return "ambiguous_edge_fallback";
    }
    throw std::logic_error("unknown crossing-owner decision kind");
}

void write_restrict_owner_probe_outputs(
    const std::filesystem::path& output_dir,
    const std::vector<NormalRestrictCaseProbe3D>& cases)
{
    std::ofstream summary =
        open_output_file(output_dir / "restrict_owner_summary.csv");
    summary << std::setprecision(17)
            << "case_id,N,target_dof,target_patch,side,layer,decision_kind,"
               "count,sum_abs_weight,wrong_side_count,"
               "wrong_side_sum_abs_weight,geometry_query_count\n";
    std::ofstream terms =
        open_output_file(output_dir / "restrict_owner_terms.csv");
    terms << std::setprecision(17)
          << "case_id,N,target_dof,target_patch,side,layer,grid_node,weight,"
             "owner_dof,owner_patch,crossing_patch,crossing_u,crossing_v,"
             "segment_parameter,residual,transversality\n";
    for (const NormalRestrictCaseProbe3D& probe_case : cases) {
        for (const RestrictOwnerSampleDiagnostics3D& item
             : probe_case.owner_diagnostics) {
            for (std::size_t kind = 0;
                 kind < kRestrictOwnerDecisionKindCount; ++kind) {
                summary << probe_case.case_id << ',' << probe_case.N << ','
                        << item.target_dof << ',' << item.target_patch << ','
                        << item.side << ',' << item.layer << ','
                        << restrict_owner_decision_kind_name(kind) << ','
                        << item.decision_counts[kind] << ','
                        << item.decision_sum_abs_weights[kind] << ','
                        << item.wrong_side_count << ','
                        << item.wrong_side_sum_abs_weight << ','
                        << item.geometry_query_count << '\n';
            }
            summary << probe_case.case_id << ',' << probe_case.N << ','
                    << item.target_dof << ',' << item.target_patch << ','
                    << item.side << ',' << item.layer
                    << ",unresolved_exception_fallback,"
                    << item.unresolved_fallback_count << ','
                    << item.unresolved_fallback_sum_abs_weight << ','
                    << item.wrong_side_count << ','
                    << item.wrong_side_sum_abs_weight << ','
                    << item.geometry_query_count << '\n';
            summary << probe_case.case_id << ',' << probe_case.N << ','
                    << item.target_dof << ',' << item.target_patch << ','
                    << item.side << ',' << item.layer
                    << ",unrelated_coincidence_fallback,"
                    << item.unrelated_coincidence_fallback_count << ','
                    << item.unrelated_coincidence_fallback_sum_abs_weight << ','
                    << item.wrong_side_count << ','
                    << item.wrong_side_sum_abs_weight << ','
                    << item.geometry_query_count << '\n';
        }
        for (const RestrictOwnerAuditRecord3D& item
             : probe_case.owner_audit_records) {
            terms << probe_case.case_id << ',' << probe_case.N << ','
                  << item.target_dof << ',' << item.target_patch << ','
                  << item.side << ',' << item.layer << ',' << item.grid_node
                  << ',' << item.interpolation_weight << ',' << item.owner_dof
                  << ',' << item.owner_patch << ',' << item.crossing_patch
                  << ',' << item.crossing_u << ',' << item.crossing_v << ','
                  << item.segment_parameter << ',' << item.residual << ','
                  << item.transversality << '\n';
        }
    }
}

void write_all_normal_restrict_probe_outputs(
    const std::filesystem::path& output_dir,
    const std::vector<NormalRestrictCaseProbe3D>& cases)
{
    write_normal_restrict_probe_outputs(output_dir, cases);
    write_normal_restrict_probe_localization(output_dir, cases);
}

const NormalRestrictCaseProbe3D* find_normal_restrict_probe_case(
    const std::vector<NormalRestrictCaseProbe3D>& cases,
    const std::string& case_id,
    int N)
{
    for (const NormalRestrictCaseProbe3D& probe_case : cases) {
        if (probe_case.case_id == case_id && probe_case.N == N)
            return &probe_case;
    }
    return nullptr;
}

bool normal_restrict_hypothesis_supported(
    const std::vector<NormalRestrictCaseProbe3D>& cases,
    const std::vector<int>& levels)
{
    const std::array<std::string, 2> rotated_ids{{
        "rot_axis123_17deg", "rot_axis123_17deg_t_xyz_1"}};
    bool supported = true;
    for (int N : levels) {
        const NormalRestrictCaseProbe3D* baseline =
            find_normal_restrict_probe_case(cases, "baseline", N);
        if (baseline == nullptr
            || !baseline->routes[0].complete
            || !baseline->routes[1].complete) {
            supported = false;
            continue;
        }
        const NormalRestrictRouteProbe3D& baseline_legacy =
            baseline->routes[0];
        const NormalRestrictRouteProbe3D& baseline_owner =
            baseline->routes[1];
        const int baseline_physical_margin = std::max(
            5, static_cast<int>(std::ceil(
                   0.10 * static_cast<double>(
                       std::max(1, baseline_legacy.physical.iterations)))));
        const int baseline_common_margin = std::max(
            5, static_cast<int>(std::ceil(
                   0.10 * static_cast<double>(
                       std::max(1, baseline_legacy.common.iterations)))));
        const bool baseline_behavior_stable =
            std::abs(baseline_owner.physical.iterations
                     - baseline_legacy.physical.iterations)
                <= baseline_physical_margin
            && std::abs(baseline_owner.common.iterations
                        - baseline_legacy.common.iterations)
                <= baseline_common_margin;
        const bool baseline_valid =
            baseline_legacy.physical.converged
            && baseline_legacy.common.converged
            && baseline_owner.physical.converged
            && baseline_owner.common.converged
            && baseline_behavior_stable
            && baseline_owner.physical.interior_linf
               <= 1.25 * std::max(
                    baseline_legacy.physical.interior_linf, 1.0e-14);
        supported = supported && baseline_valid;
        for (const std::string& rotated_id : rotated_ids) {
            const NormalRestrictCaseProbe3D* rotated =
                find_normal_restrict_probe_case(cases, rotated_id, N);
            if (rotated == nullptr
                || !rotated->routes[0].complete
                || !rotated->routes[1].complete) {
                supported = false;
                continue;
            }
            const NormalRestrictRouteProbe3D& legacy = rotated->routes[0];
            const NormalRestrictRouteProbe3D& owner = rotated->routes[1];
            const bool routes_valid =
                legacy.physical.converged && legacy.common.converged
                && owner.physical.converged && owner.common.converged;
            const bool accuracy_stable =
                owner.physical.interior_linf
                <= 1.50 * std::max(
                    legacy.physical.interior_linf, 1.0e-14);
            const bool physical_plateau_shortened =
                owner.physical.iterations + 3
                    <= legacy.physical.iterations
                || (std::isfinite(owner.physical_rho_10_30)
                    && std::isfinite(legacy.physical_rho_10_30)
                    && owner.physical_rho_10_30
                       <= 0.98 * legacy.physical_rho_10_30);
            const bool common_plateau_shortened =
                owner.common.iterations + 3 <= legacy.common.iterations
                || (std::isfinite(owner.common_rho_10_30)
                    && std::isfinite(legacy.common_rho_10_30)
                    && owner.common_rho_10_30
                       <= 0.98 * legacy.common_rho_10_30);
            const bool pair_supported = baseline_valid && routes_valid
                && accuracy_stable
                && (physical_plateau_shortened
                    || common_plateau_shortened);
            supported = supported && pair_supported;
            std::cout << "[restrict-decision] case=" << rotated_id
                      << " N=" << N
                      << " physical_iterations(legacy/owner)="
                      << legacy.physical.iterations << '/'
                      << owner.physical.iterations
                      << " common_iterations(legacy/owner)="
                      << legacy.common.iterations << '/'
                      << owner.common.iterations
                      << " physical_plateau_shortened="
                      << physical_plateau_shortened
                      << " common_plateau_shortened="
                      << common_plateau_shortened
                      << " baseline_behavior_stable="
                      << baseline_behavior_stable
                      << " accuracy_stable=" << accuracy_stable << '\n';
        }
    }
    return supported;
}

int run_normal_restrict_causal_probe(std::vector<int> levels,
                                     bool owner_only,
                                     bool phase_profile = false)
{
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    for (int N : levels) {
        if (N < 16 || !is_power_of_two(N))
            throw std::invalid_argument(
                "restrict-probe N must be a power of two and at least 16");
    }
    if (phase_profile && !owner_only)
        throw std::invalid_argument(
            "phase profiling requires the crossing-owner-only route");
    if (phase_profile && levels.size() != 1)
        throw std::invalid_argument(
            "phase profiling accepts exactly one grid level");

    const app3d::RestrictOwnerPreprocessMode3D owner_preprocess_mode =
        selected_restrict_owner_preprocess_mode();

    const std::string output_leaf = phase_profile
        ? "dirichlet_normal_restrict_crossing_owner_profile_3d"
        : owner_only
            ? "dirichlet_normal_restrict_crossing_owner_3d"
            : "dirichlet_normal_restrict_causal_probe_3d";
#ifdef KFBIM_APP_OUTPUT_DIR
    const std::filesystem::path output_dir =
        std::filesystem::path(KFBIM_APP_OUTPUT_DIR)
        / output_leaf;
#else
    const std::filesystem::path output_dir =
        std::filesystem::path("output") / output_leaf;
#endif
    const std::vector<std::string> selected_ids = phase_profile
        ? std::vector<std::string>{"rot_axis123_17deg"}
        : std::vector<std::string>{
            "baseline", "rot_axis123_17deg",
            "rot_axis123_17deg_t_xyz_1"};
    const std::vector<app3d::DirichletRigidStudyCase3D> all_cases =
        app3d::make_l_prism_dirichlet_rigid_study_cases_3d();
    std::vector<app3d::DirichletRigidStudyCase3D> study_cases;
    for (const std::string& id : selected_ids) {
        const auto found = std::find_if(
            all_cases.begin(), all_cases.end(),
            [&](const app3d::DirichletRigidStudyCase3D& item) {
                return item.id == id;
            });
        if (found == all_cases.end())
            throw std::runtime_error("missing restrict-probe rigid case: " + id);
        study_cases.push_back(*found);
    }

    std::vector<NormalRestrictCaseProbe3D> results;
    if (!phase_profile) {
        write_all_normal_restrict_probe_outputs(output_dir, results);
        write_restrict_owner_probe_outputs(output_dir, results);
    }
    const double seconds_per_clock_read = phase_profile
        ? app3d::calibrate_steady_clock_read_seconds_3d() : 0.0;
    std::cout << "KFBI3D exterior-normal restrict causal probe\n"
              << "  geometry=l_prism cauchy=g1_nearest/degree3/48/28\n"
              << "  gmres_tolerance=2e-10 restart=0 cap=160\n"
              << "  restrict_owner_preprocess_mode="
              << app3d::restrict_owner_preprocess_mode_name_3d(
                     owner_preprocess_mode) << '\n';
    for (int N : levels) {
        const double h = kBoxSide / static_cast<double>(N);
        for (const auto& study_case : study_cases) {
            app3d::PhaseProfile3D profile_storage;
            app3d::PhaseProfile3D* profile =
                phase_profile ? &profile_storage : nullptr;
            const ProfileClock3D::time_point profile_wall_start =
                profile_timer_start_3d(profile);
            if (profile != nullptr) {
                profile_phase_3d(
                    profile, PhaseProfileKind3D::DiagnosticOutput, 1,
                    [&] {
                        std::filesystem::remove(
                            output_dir / "phase_profile.csv");
                    });
                profile_phase_3d(
                    profile, PhaseProfileKind3D::DiagnosticOutput, 1,
                    [&] {
                        write_all_normal_restrict_probe_outputs(
                            output_dir, results);
                    });
                profile_phase_3d(
                    profile, PhaseProfileKind3D::DiagnosticOutput, 1,
                    [&] {
                        write_restrict_owner_probe_outputs(
                            output_dir, results);
                    });
            }

            const auto setup_start = std::chrono::steady_clock::now();
            const ProfileClock3D::time_point geometry_start =
                profile_timer_start_3d(profile);
            CartesianGrid3D grid({kBoxMin, kBoxMin, kBoxMin},
                                 {h, h, h}, {N, N, N}, DofLayout3D::Node);
            GeometryBundle geometry = make_geometry(
                GeometryKind::LPrism, h, study_case.transform);
            const auto domain = std::make_shared<const
                geometry3d::NurbsCartesianDomain3D>(
                    grid, geometry.native_surface.geometry_model());
            profile_add_elapsed_3d(
                profile, PhaseProfileKind3D::GeometryAndDomain,
                geometry_start);

            const ProfileClock3D::time_point surface_start =
                profile_timer_start_3d(profile);
            const SurfaceDofCloud surface_dofs =
                app3d::make_native_surface_dofs_3d(
                    geometry.native_surface, h);
            validate_surface_dofs(surface_dofs, h);
            app3d::HarmonicCauchyFit3D cauchy_fit =
                app3d::HarmonicCauchyFit3D::build_legacy(
                    geometry.native_surface, surface_dofs, h,
                    app3d::LegacySurfaceCauchyPolicy3D::G1Nearest,
                    kCauchyPolynomialDegree, kCauchyValueNeighborCount,
                    kCauchyDerivativeNeighborCount);
            profile_add_elapsed_3d(
                profile, PhaseProfileKind3D::SurfaceDofsAndStencils,
                surface_start);

            const ProfileClock3D::time_point grid_pair_start =
                profile_timer_start_3d(profile);
            GridPair3D grid_pair(grid,
                                 geometry.correction_interface,
                                 geometry.crossing_interface,
                                 domain);
            for (int node = 0; node < grid.num_dofs(); ++node) {
                const bool numerical_inside = grid_pair.domain_label(node) > 0;
                if (numerical_inside
                    != geometry.exact_inside(grid_point(grid, node))) {
                    throw std::runtime_error(
                        "restrict-probe native NURBS label mismatch");
                }
            }
            profile_add_elapsed_3d(
                profile, PhaseProfileKind3D::GridPairAndLabelValidation,
                grid_pair_start);

            const double crossing_rows_before = profile != nullptr
                ? profile->record(
                    PhaseProfileKind3D::CrossingRows).seconds : 0.0;
            const double intersections_before = profile != nullptr
                ? profile->record(
                    PhaseProfileKind3D::NurbsSegmentIntersections).seconds
                : 0.0;
            const double owner_preprocess_before = profile != nullptr
                ? profile->record(
                    PhaseProfileKind3D::
                        RestrictOwnerGeometryPreprocessing).seconds : 0.0;
            const double trace_assembly_before = profile != nullptr
                ? profile->record(
                    PhaseProfileKind3D::TraceOwnerTemplateAssembly).seconds
                : 0.0;
            if (profile != nullptr)
                profile->note_timer_reads(1);
            const auto pipeline_start = std::chrono::steady_clock::now();
            RestrictOwnerWorkload3D workload_capture;
            RestrictOwnerPipelinePreprocessTiming3D
                phase_pipeline_timing;
            PanelCenterHarmonicJetKFBI3D pipeline(
                grid, grid_pair, geometry.native_surface,
                geometry.correction_triangles,
                geometry.geometry_triangles,
                surface_dofs, std::move(cauchy_fit), !owner_only,
                owner_preprocess_mode,
                profile, &workload_capture,
                profile != nullptr
                    ? std::addressof(phase_pipeline_timing) : nullptr);
            if (profile != nullptr)
                profile->note_timer_reads(1);
            const auto pipeline_end = std::chrono::steady_clock::now();
            const double pipeline_wall_seconds =
                std::chrono::duration<double>(
                    pipeline_end - pipeline_start).count();
            const double pipeline_setup_seconds =
                profile != nullptr
                    ? production_pipeline_setup_seconds_3d(
                        pipeline_wall_seconds, phase_pipeline_timing)
                    : pipeline_wall_seconds;
            if (profile != nullptr) {
                const double child_seconds =
                    profile->record(
                        PhaseProfileKind3D::CrossingRows).seconds
                    - crossing_rows_before
                    + profile->record(
                        PhaseProfileKind3D::NurbsSegmentIntersections).seconds
                    - intersections_before
                    + profile->record(
                        PhaseProfileKind3D::
                            RestrictOwnerGeometryPreprocessing).seconds
                    - owner_preprocess_before
                    + profile->record(
                        PhaseProfileKind3D::TraceOwnerTemplateAssembly).seconds
                    - trace_assembly_before;
                profile->add(
                    PhaseProfileKind3D::PipelineFixedInitialization,
                    nonnegative_profile_remainder_3d(
                        pipeline_setup_seconds, child_seconds,
                        "pipeline setup"),
                    1);
            }

            const ProfileClock3D::time_point other_setup_start =
                profile_timer_start_3d(profile);
            const double setup_coefficients_before = profile != nullptr
                ? profile->record(
                    PhaseProfileKind3D::CauchyCoefficients).seconds : 0.0;
            NormalRestrictCaseProbe3D probe_case;
            probe_case.case_id = study_case.id;
            probe_case.N = N;
            probe_case.h = h;
            probe_case.wrong_side_support_count =
                pipeline.joint_trace_wrong_side_node_counts();
            probe_case.owner_diagnostics =
                pipeline.restrict_owner_sample_diagnostics();
            probe_case.owner_audit_records =
                pipeline.restrict_owner_audit_records();
            probe_case.owner_geometry_query_count =
                pipeline.restrict_owner_geometry_query_count();
            probe_case.owner_workload_fingerprint =
                pipeline.restrict_owner_workload_fingerprint();
            probe_case.owner_preprocess_diagnostics =
                pipeline.restrict_owner_preprocess_diagnostics();
            probe_case.pipeline_setup_seconds = pipeline_setup_seconds;
            if (probe_case.owner_diagnostics.empty()
                || workload_capture.empty()
                || workload_capture.size()
                       != probe_case.owner_diagnostics.size()) {
                throw std::runtime_error(
                    "restrict-probe crossing-owner diagnostics are empty");
            }
            std::size_t classified_queries = 0;
            for (std::size_t sample = 0;
                 sample < probe_case.owner_diagnostics.size(); ++sample) {
                const RestrictOwnerSampleDiagnostics3D& diagnostic =
                    probe_case.owner_diagnostics[sample];
                const RestrictOwnerWorkloadSample3D& workload =
                    workload_capture[sample];
                classified_queries +=
                    static_cast<std::size_t>(diagnostic.wrong_side_count);
                if (workload.target_dof != diagnostic.target_dof
                    || workload.side != diagnostic.side
                    || workload.layer != diagnostic.layer
                    || workload.oracle_nodes.size()
                           != static_cast<std::size_t>(
                               diagnostic.wrong_side_count)) {
                    throw std::logic_error(
                        "restrict-owner workload capture is inconsistent");
                }
            }
            if (classified_queries != probe_case.owner_geometry_query_count)
                throw std::logic_error(
                    "crossing-owner probe query count is inconsistent");
            validate_restrict_owner_preprocess_diagnostics(
                probe_case.owner_preprocess_diagnostics,
                probe_case.owner_geometry_query_count);
            std::cout << "[restrict-owner-preprocess] case="
                      << probe_case.case_id << " N=" << N
                      << " mode="
                      << app3d::restrict_owner_preprocess_mode_name_3d(
                             owner_preprocess_mode)
                      << " workload_fingerprint="
                      << probe_case.owner_workload_fingerprint
                      << " wrong_side_queries="
                      << probe_case.owner_preprocess_diagnostics.
                             wrong_side_queries << '\n';
            const int size = pipeline.surface_size();
            probe_case.patch_ids.reserve(static_cast<std::size_t>(size));
            probe_case.points.reserve(static_cast<std::size_t>(size));
            probe_case.weights.reserve(static_cast<std::size_t>(size));
            probe_case.feature_edge_distance_over_h.reserve(
                static_cast<std::size_t>(size));
            if (geometry.feature_edge_segments.empty())
                throw std::runtime_error("L-prism has no feature-edge segments");
            for (const SurfaceDof& dof : pipeline.surface().dofs) {
                probe_case.patch_ids.push_back(dof.patch_id);
                probe_case.points.push_back(dof.point);
                probe_case.weights.push_back(dof.weight);
                double distance = std::numeric_limits<double>::infinity();
                for (const auto& edge : geometry.feature_edge_segments) {
                    distance = std::min(
                        distance, point_segment_distance_3d(dof.point, edge));
                }
                probe_case.feature_edge_distance_over_h.push_back(distance / h);
            }

            Eigen::VectorXd value_data(size);
            Eigen::VectorXd exact_normal(size);
            const Eigen::VectorXd zero_jump = Eigen::VectorXd::Zero(size);
            for (int q = 0; q < size; ++q) {
                const SurfaceDof& dof =
                    pipeline.surface().dofs[static_cast<std::size_t>(q)];
                value_data[q] =
                    app3d::transformed_manufactured_harmonic_value_3d(
                        study_case.transform, dof.point);
                exact_normal[q] =
                    app3d::transformed_manufactured_harmonic_gradient_3d(
                        study_case.transform, dof.point).dot(dof.normal);
            }
            Eigen::VectorXd piecewise_exact(grid.num_dofs());
            Eigen::VectorXd smooth_exact(grid.num_dofs());
            for (int node = 0; node < grid.num_dofs(); ++node) {
                const double value =
                    app3d::transformed_manufactured_harmonic_value_3d(
                        study_case.transform, grid_point(grid, node));
                smooth_exact[node] = value;
                piecewise_exact[node] = grid_pair.domain_label(node) > 0
                    ? value : 0.0;
            }
            const HarmonicJetField3D piecewise_field =
                pipeline.field_from_grid_and_jumps(
                    piecewise_exact, value_data, exact_normal);
            const HarmonicJetField3D smooth_field =
                pipeline.field_from_grid_and_jumps(
                    smooth_exact, zero_jump, zero_jump);
            const Eigen::VectorXd common_rhs =
                make_common_normal_restrict_rhs(size);
            if (profile != nullptr) {
                const double other_setup_seconds =
                    profile_timer_elapsed_3d(profile, other_setup_start);
                const double coefficient_seconds =
                    profile->record(
                        PhaseProfileKind3D::CauchyCoefficients).seconds
                    - setup_coefficients_before;
                profile->add(
                    PhaseProfileKind3D::ExactFieldsAndOtherSetup,
                    nonnegative_profile_remainder_3d(
                        other_setup_seconds, coefficient_seconds,
                        "remaining setup"),
                    1);
            }
            probe_case.setup_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - setup_start).count();
            results.push_back(std::move(probe_case));
            profile_phase_3d(
                profile, PhaseProfileKind3D::DiagnosticOutput, 1,
                [&] {
                    write_restrict_owner_probe_outputs(output_dir, results);
                });
            NormalRestrictCaseProbe3D& stored = results.back();

            const std::vector<ExteriorNormalRestrictMode3D> modes = owner_only
                ? std::vector<ExteriorNormalRestrictMode3D>{
                    ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner}
                : std::vector<ExteriorNormalRestrictMode3D>{
                    ExteriorNormalRestrictMode3D::JointTricubicCauchy,
                    ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner,
                    ExteriorNormalRestrictMode3D::ExteriorOnlyHarmonicCubic};
            for (ExteriorNormalRestrictMode3D mode : modes) {
                const std::array<PhaseProfileKind3D, 5> route_kinds{{
                    PhaseProfileKind3D::CauchyCoefficients,
                    PhaseProfileKind3D::SpreadRhsAssembly,
                    PhaseProfileKind3D::FftBulkSolve,
                    PhaseProfileKind3D::RestrictContinuedSamples,
                    PhaseProfileKind3D::RestrictRecovery,
                }};
                std::array<double, 5> route_children_before{};
                if (profile != nullptr) {
                    for (std::size_t q = 0; q < route_kinds.size(); ++q) {
                        route_children_before[q] =
                            profile->record(route_kinds[q]).seconds;
                    }
                    profile->note_timer_reads(1);
                }
                const auto route_start = std::chrono::steady_clock::now();
                NormalRestrictRouteProbe3D& route = stored.routes[
                    static_cast<std::size_t>(normal_restrict_route_index(mode))];
                route.route = normal_restrict_route_name(mode);
                route.conditions = summarize_conditions(
                    mode == ExteriorNormalRestrictMode3D::ExteriorOnlyHarmonicCubic
                        ? pipeline.exterior_only_restrict_condition_values()
                        : pipeline.cauchy_condition_values());
                route.exact_grid_error = pipeline.exterior_normal_trace(
                    piecewise_field, value_data, exact_normal, mode);
                route.smooth_grid_error = pipeline.exterior_normal_trace(
                    smooth_field, zero_jump, zero_jump, mode) - exact_normal;
                ExteriorNormalTraceOperator3D op(pipeline, mode);
                Eigen::VectorXd applied;
                op.apply(exact_normal, applied);
                route.exact_equation_residual =
                    applied - op.right_hand_side(value_data);
                route.exact_grid_norms = normal_restrict_weighted_norms(
                    route.exact_grid_error, stored.weights);
                route.smooth_grid_norms = normal_restrict_weighted_norms(
                    route.smooth_grid_error, stored.weights);
                route.exact_equation_norms = normal_restrict_weighted_norms(
                    route.exact_equation_residual, stored.weights);
                const app3d::RestrictOwnerPreprocessDiagnostics3D
                    diagnostics_before_gmres =
                        pipeline.restrict_owner_preprocess_diagnostics();
                const std::size_t queries_before_gmres =
                    pipeline.restrict_owner_geometry_query_count();
                route.physical = run_dirichlet_normal_case(
                    grid, grid_pair, pipeline, study_case.transform, 160,
                    mode, &route.physical_residuals);
                route.common = run_common_normal_restrict_gmres(
                    pipeline, mode, common_rhs);
                const app3d::RestrictOwnerPreprocessDiagnostics3D
                    diagnostics_after_gmres =
                        pipeline.restrict_owner_preprocess_diagnostics();
                const std::size_t queries_after_gmres =
                    pipeline.restrict_owner_geometry_query_count();
                if (!restrict_owner_preprocess_diagnostics_equal(
                        diagnostics_before_gmres,
                        diagnostics_after_gmres)) {
                    throw std::runtime_error(
                        "GMRES changed restrict-owner preprocessing "
                        "diagnostics");
                }
                route.physical_rho_10_30 =
                    residual_contraction(route.physical_residuals, 10, 30);
                route.physical_worst_rho_5 =
                    worst_five_step_contraction(route.physical_residuals);
                route.common_rho_10_30 =
                    residual_contraction(route.common.residuals, 10, 30);
                route.common_worst_rho_5 =
                    worst_five_step_contraction(route.common.residuals);
                if (queries_before_gmres != queries_after_gmres
                    || queries_after_gmres
                    != stored.owner_geometry_query_count) {
                    throw std::runtime_error(
                        "GMRES apply performed a crossing-owner geometry query");
                }
                if (profile != nullptr)
                    profile->note_timer_reads(1);
                const auto route_end = std::chrono::steady_clock::now();
                route.seconds = std::chrono::duration<double>(
                    route_end - route_start).count();
                if (profile != nullptr) {
                    double child_seconds = 0.0;
                    for (std::size_t q = 0; q < route_kinds.size(); ++q) {
                        child_seconds +=
                            profile->record(route_kinds[q]).seconds
                            - route_children_before[q];
                    }
                    profile->add(
                        PhaseProfileKind3D::GmresAndOtherRoute,
                        nonnegative_profile_remainder_3d(
                            route.seconds, child_seconds,
                            "normal-restrict route"),
                        1);
                }
                route.complete = true;
                profile_phase_3d(
                    profile, PhaseProfileKind3D::DiagnosticOutput, 1,
                    [&] {
                        write_all_normal_restrict_probe_outputs(
                            output_dir, results);
                    });
                std::cout << "[restrict-probe] case=" << stored.case_id
                          << " N=" << N << " route=" << route.route
                          << " exact_grid_linf="
                          << route.exact_grid_norms.linf
                          << " smooth_grid_linf="
                          << route.smooth_grid_norms.linf
                          << " equation_linf="
                          << route.exact_equation_norms.linf
                          << " physical_iter/error="
                          << route.physical.iterations << '/'
                          << route.physical.interior_linf
                          << " common_iter=" << route.common.iterations
                          << " preprocess_queries(pre/post)="
                          << diagnostics_before_gmres.wrong_side_queries
                          << '/' << diagnostics_after_gmres.wrong_side_queries
                          << " diagnostics_unchanged=1"
                          << " seconds=" << route.seconds << '\n';
            }
            const double total_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - setup_start).count();
            std::cout << "[restrict-probe-case] case=" << stored.case_id
                      << " N=" << N << " dofs=" << size
                      << " setup_seconds=" << stored.setup_seconds
                      << " pipeline_setup_seconds="
                      << stored.pipeline_setup_seconds
                      << " total_seconds=" << total_seconds << '\n';

            if (profile != nullptr) {
                profile->set_seconds_per_clock_read(
                    seconds_per_clock_read);
                const double profile_wall_seconds =
                    profile_timer_elapsed_3d(profile, profile_wall_start);
                profile->finalize(profile_wall_seconds);
                double recorded_wall_seconds = 0.0;
                for (std::size_t q = 0;
                     q < app3d::phase_profile_kind_count_3d(); ++q) {
                    recorded_wall_seconds += profile->record(
                        static_cast<PhaseProfileKind3D>(q)).seconds;
                }
                const double exhaustive_tolerance =
                    std::max(1.0e-6, 1.0e-8 * profile_wall_seconds);
                if (std::abs(
                        recorded_wall_seconds - profile_wall_seconds)
                    > exhaustive_tolerance) {
                    throw std::logic_error(
                        "phase-profile leaves are not exhaustive");
                }
                app3d::write_phase_profile_csv_3d(
                    output_dir / "phase_profile.csv",
                    stored.case_id, N, *profile);
                const double overhead_percent =
                    profile_wall_seconds > 0.0
                    ? 100.0
                        * profile->estimated_timer_overhead_seconds()
                        / profile_wall_seconds
                    : 0.0;
                std::cout << "[phase-profile] algorithm_seconds="
                          << profile->algorithm_seconds()
                          << " wall_seconds=" << profile_wall_seconds
                          << " estimated_timer_overhead_percent="
                          << overhead_percent << '\n';
            }
        }
    }
    if (!owner_only) {
        const bool supported =
            normal_restrict_hypothesis_supported(results, levels);
        std::cout << "[restrict-hypothesis] "
                  << (supported ? "supported" : "not_proven") << '\n';
    }
    std::cout << "Restrict probe output: " << output_dir.string() << '\n';
    return 0;
}

using OwnerMode3D = app3d::RestrictOwnerPreprocessMode3D;
using OwnerClass3D = app3d::RestrictOwnerNormalizedClass3D;
using OwnerPath3D = app3d::RestrictOwnerQueryPath3D;
using OwnerFallback3D = app3d::RestrictOwnerFallbackCause3D;

struct OwnerPreprocessStudyCase3D {
    std::string case_id;
    std::string geometry;
    std::string pose;
    GeometryKind kind = GeometryKind::Torus;
    app3d::RigidTransform3D transform;
};

struct PreprocessTimingRepeatRow3D {
    std::string case_id, geometry, pose, mode;
    int N = 0, repetition = 0, run_order = 0;
    double h = 0.0;
    std::uint64_t wrong_side_queries = 0;
    double construction_seconds = 0.0, query_seconds = 0.0;
    double total_preprocess_seconds = 0.0, queries_per_second = 0.0;
    std::uint64_t peak_working_set_bytes = 0;
    std::uint64_t working_set_increase_bytes = 0;
};

struct PreprocessSummaryRow3D {
    std::string case_id, geometry, pose, mode;
    int N = 0, repetitions = 0;
    double h = 0.0;
    std::uint64_t wrong_side_queries = 0;
    double construction_seconds_median = 0.0;
    double query_seconds_median = 0.0, query_seconds_p95 = 0.0;
    double total_seconds_median = 0.0, total_seconds_p95 = 0.0;
    double queries_per_second_median = 0.0, speedup_vs_full = 0.0;
    std::uint64_t peak_working_set_bytes_max = 0;
    bool accepted = false;
};

struct PreprocessAccuracyRow3D {
    std::string case_id, geometry, pose, mode;
    int N = 0;
    double h = 0.0;
    std::uint64_t query_count = 0, oracle_target = 0, oracle_foreign = 0;
    std::uint64_t owner_mismatch = 0, foreign_true_positive = 0;
    std::uint64_t foreign_false_positive = 0, foreign_false_negative = 0;
    std::uint64_t foreign_wrong_owner = 0;
    double foreign_precision = 0.0, foreign_recall = 0.0;
    double max_mismatched_abs_weight = 0.0;
    double max_owner_correction_abs_difference = 0.0;
    std::uint64_t legacy_reason_mismatch = 0;
    bool pass = false;
};

struct PreprocessPathCountRow3D {
    std::string case_id, geometry, pose, mode, path;
    int N = 0;
    double h = 0.0;
    std::uint64_t count = 0;
    double fraction = 0.0;
    app3d::RestrictOwnerPreprocessDiagnostics3D diagnostics;
};

struct PreprocessMismatchRow3D {
    std::string case_id, geometry, pose, mode;
    int N = 0, target_dof = -1, target_patch = -1, side = -1, layer = -1;
    double h = 0.0;
    int grid_node = -1;
    double weight = 0.0;
    Eigen::Vector3d query = Eigen::Vector3d::Zero();
    Eigen::Vector3d support = Eigen::Vector3d::Zero();
    int oracle_owner_dof = -1, oracle_owner_patch = -1;
    int candidate_owner_dof = -1, candidate_owner_patch = -1;
    std::string oracle_class, candidate_class, candidate_path;
    int oracle_crossing_patch = -1, candidate_crossing_patch = -1;
};

struct KfbiNumericalResultRow3D {
    std::string case_id, geometry, pose, mode;
    int N = 0, dofs = 0;
    double h = 0.0, preprocess_seconds = 0.0;
    double pipeline_setup_seconds = 0.0, pipeline_speedup_vs_full = 0.0;
    SolveMetrics3D solve;
    double interior_order = std::numeric_limits<double>::quiet_NaN();
    double exact_grid_linf = 0.0, exact_equation_linf = 0.0;
    double interior_relative_difference_vs_full = 0.0;
    double exact_grid_relative_difference_vs_full = 0.0;
    double exact_equation_relative_difference_vs_full = 0.0;
    std::uint64_t geometry_queries_before_gmres = 0;
    std::uint64_t geometry_queries_after_gmres = 0;
    bool pass = false;
};

struct OwnerPreprocessStudyRows3D {
    std::vector<PreprocessTimingRepeatRow3D> timing;
    std::vector<PreprocessSummaryRow3D> summary;
    std::vector<PreprocessAccuracyRow3D> accuracy;
    std::vector<PreprocessPathCountRow3D> paths;
    std::vector<PreprocessMismatchRow3D> mismatches;
    std::vector<KfbiNumericalResultRow3D> numerical;
};

struct RestrictOwnerAccuracyComparison3D {
    PreprocessAccuracyRow3D accuracy;
    std::vector<PreprocessMismatchRow3D> mismatches;
};

struct RestrictOwnerReplayResult3D {
    std::optional<RestrictOwnerAccuracyComparison3D> comparison;
    app3d::RestrictOwnerPreprocessDiagnostics3D diagnostics;
    double construction_seconds = 0.0, query_seconds = 0.0;
    std::uint64_t peak_working_set_bytes = 0;
    std::uint64_t max_active_increase_bytes = 0;
    std::uint64_t output_digest = UINT64_C(14695981039346656037);
};

std::string owner_class_name_3d(OwnerClass3D value)
{
    if (value == OwnerClass3D::Target)
        return "target";
    if (value == OwnerClass3D::UniqueForeign)
        return "unique_foreign";
    return "fail_closed_target";
}

std::string owner_path_name_3d(OwnerPath3D value)
{
    if (value == OwnerPath3D::FullIntersection)
        return "full_intersection";
    if (value == OwnerPath3D::SweepTargetOnly)
        return "sweep_target_only";
    if (value == OwnerPath3D::SegmentTargetOnly)
        return "segment_target_only";
    if (value == OwnerPath3D::ClosestCertifiedMiss)
        return "closest_certified_miss";
    if (value == OwnerPath3D::ClosestCertifiedRoot)
        return "closest_certified_root";
    if (value == OwnerPath3D::OptimizedIntersection)
        return "optimized_intersection";
    return "full_intersection_fallback";
}

std::string owner_fallback_name_3d(OwnerFallback3D value)
{
    if (value == OwnerFallback3D::MultipleCrossings)
        return "multiple_crossings";
    if (value == OwnerFallback3D::Unresolved)
        return "unresolved";
    if (value == OwnerFallback3D::Overlap)
        return "overlap";
    if (value == OwnerFallback3D::Endpoint)
        return "endpoint";
    if (value == OwnerFallback3D::NearTangent)
        return "near_tangent";
    if (value == OwnerFallback3D::FeatureContact)
        return "feature_contact";
    if (value == OwnerFallback3D::ParameterBoundary)
        return "parameter_boundary";
    if (value == OwnerFallback3D::Seam)
        return "seam";
    if (value == OwnerFallback3D::Coincidence)
        return "coincidence";
    return "none";
}

app3d::DirichletRigidStudyCase3D rigid_case_by_id_3d(
    std::vector<app3d::DirichletRigidStudyCase3D> cases,
    std::string id)
{
    for (std::size_t index = 0; index not_eq cases.size(); index += 1) {
        if (cases[index].id == id)
            return cases[index];
    }
    throw std::logic_error("missing rigid study case " + id);
}

std::vector<OwnerPreprocessStudyCase3D> owner_study_cases_3d()
{
    const auto rigid = app3d::make_l_prism_dirichlet_rigid_study_cases_3d();
    const auto identity = rigid_case_by_id_3d(rigid, "baseline");
    const auto rotation = rigid_case_by_id_3d(
        rigid, "rot_axis123_17deg");
    const auto translated = rigid_case_by_id_3d(
        rigid, "rot_axis123_17deg_t_xyz_1");
    std::vector<OwnerPreprocessStudyCase3D> result;
    result.reserve(7);
    result.push_back({"torus_identity", "torus", "identity",
        GeometryKind::Torus, identity.transform});
    result.push_back({"hollow_cylinder_identity", "hollow_cylinder",
        "identity", GeometryKind::HollowCylinder, identity.transform});
    result.push_back({"l_prism_identity", "l_prism", "identity",
        GeometryKind::LPrism, identity.transform});
    result.push_back({"torus_rot_axis123_17deg", "torus",
        "rot_axis123_17deg", GeometryKind::Torus, rotation.transform});
    result.push_back({"hollow_cylinder_rot_axis123_17deg",
        "hollow_cylinder", "rot_axis123_17deg",
        GeometryKind::HollowCylinder, rotation.transform});
    result.push_back({"l_prism_rot_axis123_17deg", "l_prism",
        "rot_axis123_17deg", GeometryKind::LPrism, rotation.transform});
    result.push_back({"l_prism_rot_axis123_17deg_t_xyz_1", "l_prism",
        "rot_axis123_17deg_t_xyz_1", GeometryKind::LPrism,
        translated.transform});
    return result;
}

bool owner_case_selected_for_level_3d(
    OwnerPreprocessStudyCase3D study_case, int N)
{
    if (N == 16)
        return study_case.pose == "identity";
    if (N == 32 or N == 64)
        return true;
    return N == 128
        and study_case.case_id == "l_prism_rot_axis123_17deg";
}

int owner_candidate_repetitions_3d(int N)
{
    if (N <= 32) return 5;
    if (N == 64) return 3;
    return 1;
}

double sorted_median_3d(std::vector<double> values)
{
    if (values.empty())
        return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0)
        return 0.5 * (values[middle - 1] + values[middle]);
    return values[middle];
}

double sorted_nearest_rank_p95_3d(std::vector<double> values)
{
    if (values.empty())
        return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const std::size_t rank = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(values.size())));
    return values[std::max<std::size_t>(1, rank) - 1];
}

double symmetric_relative_difference_3d(double a, double b)
{
    return std::abs(a - b)
         / std::max({1.0e-300, std::abs(a), std::abs(b)});
}


double production_pipeline_setup_seconds_3d(
    double outer_constructor_seconds,
    const RestrictOwnerPipelinePreprocessTiming3D& timing)
{
    if (!std::isfinite(outer_constructor_seconds)
        || !std::isfinite(timing.workload_capture_seconds)
        || outer_constructor_seconds <= 0.0
        || timing.workload_capture_seconds < 0.0
        || timing.workload_capture_seconds > outer_constructor_seconds) {
        throw std::logic_error(
            "invalid production-equivalent pipeline setup timing");
    }
    const double result =
        outer_constructor_seconds - timing.workload_capture_seconds;
    if (result <= 0.0)
        throw std::logic_error(
            "nonpositive production-equivalent pipeline setup timing");
    return result;
}

RestrictOwnerAccuracyComparison3D make_owner_accuracy_comparison_3d(
    const CartesianGrid3D& grid,
    OwnerPreprocessStudyCase3D study_case,
    int N,
    OwnerMode3D mode)
{
    RestrictOwnerAccuracyComparison3D result;
    result.accuracy.case_id = study_case.case_id;
    result.accuracy.geometry = study_case.geometry;
    result.accuracy.pose = study_case.pose;
    result.accuracy.N = N;
    result.accuracy.h = grid.spacing()[0];
    result.accuracy.mode =
        app3d::restrict_owner_preprocess_mode_name_3d(mode);
    return result;
}

void accumulate_owner_accuracy_sample_3d(
    RestrictOwnerAccuracyComparison3D& result,
    const SurfaceDofCloud& cloud,
    const RestrictOwnerWorkloadSample3D& sample,
    const RestrictOwnerTraceStencil3D& trace,
    const RestrictOwnerPreparedTraceSample3D& candidate_sink)
{
    if (candidate_sink.decisions.size() != sample.oracle_nodes.size())
        throw std::logic_error("candidate replay is incomplete");
    for (std::size_t index = 0; index < sample.oracle_nodes.size(); ++index) {
        const RestrictOwnerOracleNode3D& oracle = sample.oracle_nodes[index];
        const RestrictOwnerStagedDecision3D& candidate =
            candidate_sink.decisions[index];
        const std::size_t slot = oracle.slot;
        if (candidate.slot != oracle.slot)
            throw std::logic_error("candidate replay slot mismatch");
        result.accuracy.query_count += 1;
        const bool oracle_foreign = oracle.owner_class
            == OwnerClass3D::UniqueForeign;
        const bool candidate_foreign = candidate.owner_class
            == OwnerClass3D::UniqueForeign;
        if (oracle_foreign)
            result.accuracy.oracle_foreign += 1;
        else
            result.accuracy.oracle_target += 1;
        if (oracle_foreign and candidate_foreign
            and oracle.owner_dof == candidate.owner_dof) {
            result.accuracy.foreign_true_positive += 1;
        } else if (oracle_foreign and candidate_foreign) {
            result.accuracy.foreign_wrong_owner += 1;
        } else if (oracle_foreign) {
            result.accuracy.foreign_false_negative += 1;
        } else if (candidate_foreign) {
            result.accuracy.foreign_false_positive += 1;
        }
        if (oracle.legacy_decision_kind.has_value()
            and candidate.legacy_decision_kind >= 0
            and oracle.legacy_decision_kind.value()
                not_eq static_cast<app3d::RestrictOwnerDecisionKind3D>(
                    candidate.legacy_decision_kind)) {
            result.accuracy.legacy_reason_mismatch += 1;
        }
        if (oracle.owner_dof not_eq candidate.owner_dof) {
            result.accuracy.owner_mismatch += 1;
            result.accuracy.max_mismatched_abs_weight = std::max(
                result.accuracy.max_mismatched_abs_weight,
                std::abs(trace.weights[slot]));
            PreprocessMismatchRow3D mismatch;
            mismatch.case_id = result.accuracy.case_id;
            mismatch.geometry = result.accuracy.geometry;
            mismatch.pose = result.accuracy.pose;
            mismatch.mode = result.accuracy.mode;
            mismatch.N = result.accuracy.N;
            mismatch.h = result.accuracy.h;
            mismatch.target_dof = sample.target_dof;
            mismatch.target_patch = cloud.dofs[
                static_cast<std::size_t>(sample.target_dof)].patch_id;
            mismatch.side = sample.side;
            mismatch.layer = sample.layer;
            mismatch.grid_node = trace.grid_nodes[slot];
            mismatch.weight = trace.weights[slot];
            mismatch.query = sample.query;
            mismatch.support = trace.owner_input.support_points[slot];
            mismatch.oracle_owner_dof = oracle.owner_dof;
            mismatch.candidate_owner_dof = candidate.owner_dof;
            mismatch.oracle_owner_patch = cloud.dofs[
                static_cast<std::size_t>(oracle.owner_dof)].patch_id;
            mismatch.candidate_owner_patch = cloud.dofs[
                static_cast<std::size_t>(candidate.owner_dof)].patch_id;
            mismatch.oracle_class = owner_class_name_3d(
                oracle.owner_class);
            mismatch.candidate_class = owner_class_name_3d(
                candidate.owner_class);
            mismatch.candidate_path = owner_path_name_3d(
                candidate.query_path);
            mismatch.oracle_crossing_patch = oracle.crossing_patch;
            if (candidate.foreign_crossing_index >= 0) {
                mismatch.candidate_crossing_patch =
                    candidate_sink.foreign_crossings[
                        static_cast<std::size_t>(
                            candidate.foreign_crossing_index)].patch_index;
            }
            result.mismatches.push_back(std::move(mismatch));
        }
    }
}

void finalize_owner_accuracy_comparison_3d(
    RestrictOwnerAccuracyComparison3D& result)
{
    const std::uint64_t precision_denominator =
        result.accuracy.foreign_true_positive
        + result.accuracy.foreign_false_positive
        + result.accuracy.foreign_wrong_owner;
    const std::uint64_t recall_denominator =
        result.accuracy.foreign_true_positive
        + result.accuracy.foreign_false_negative
        + result.accuracy.foreign_wrong_owner;
    result.accuracy.foreign_precision = precision_denominator == 0
        ? 1.0
        : static_cast<double>(result.accuracy.foreign_true_positive)
          / static_cast<double>(precision_denominator);
    result.accuracy.foreign_recall = recall_denominator == 0
        ? 1.0
        : static_cast<double>(result.accuracy.foreign_true_positive)
          / static_cast<double>(recall_denominator);
    result.accuracy.pass = result.accuracy.owner_mismatch == 0
        and result.accuracy.foreign_false_positive == 0
        and result.accuracy.foreign_false_negative == 0
        and result.accuracy.foreign_wrong_owner == 0;
}

std::uint64_t restrict_owner_oracle_query_count_3d(
    const RestrictOwnerWorkload3D& workload)
{
    return std::accumulate(
        workload.begin(), workload.end(), std::uint64_t{0},
        [](std::uint64_t count, const RestrictOwnerWorkloadSample3D& sample) {
            return count
                + static_cast<std::uint64_t>(sample.oracle_nodes.size());
        });
}

RestrictOwnerReplayResult3D replay_restrict_owner_workload_3d(
    const CartesianGrid3D& grid,
    const GridPair3D& grid_pair,
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud& cloud,
    const RestrictOwnerWorkload3D& workload,
    OwnerMode3D mode,
    app3d::RestrictOwnerPreprocessOptions3D options,
    bool measure_benchmark,
    const OwnerPreprocessStudyCase3D* study_case = nullptr,
    int N = 0)
{
    RestrictOwnerReplayResult3D result;
    if (study_case != nullptr) {
        result.comparison = make_owner_accuracy_comparison_3d(
            grid, *study_case, N, mode);
    }
    const std::uint64_t expected_queries =
        restrict_owner_oracle_query_count_3d(workload);
    RestrictOwnerPreparedTraceSample3D candidate_sink;
    candidate_sink.decisions.reserve(64);
    candidate_sink.foreign_crossings.reserve(64);
    std::unique_ptr<WorkingSetPeakSampler3D> memory_sampler;
    if (measure_benchmark) {
        memory_sampler = std::make_unique<WorkingSetPeakSampler3D>();
        memory_sampler->resume();
        memory_sampler->checkpoint();
    }
    std::unique_ptr<app3d::RestrictOwnerGeometryPreprocessor3D>
        preprocessor;
    try {
        const auto construction_start =
            std::chrono::steady_clock::now();
        preprocessor = std::make_unique<
            app3d::RestrictOwnerGeometryPreprocessor3D>(
                surface, cloud, grid.spacing()[0], mode, options);
        const auto construction_end =
            std::chrono::steady_clock::now();
        if (measure_benchmark) {
            memory_sampler->checkpoint();
            memory_sampler->pause();
            result.construction_seconds =
                std::chrono::duration<double>(
                    construction_end - construction_start).count();
        }
    } catch (...) {
        if (memory_sampler != nullptr)
            memory_sampler->pause();
        throw;
    }
    for (const RestrictOwnerWorkloadSample3D& sample : workload) {
        candidate_sink.target_dof = sample.target_dof;
        candidate_sink.side = sample.side;
        candidate_sink.layer = sample.layer;
        candidate_sink.query = sample.query;
        candidate_sink.lower_stencil_index = sample.lower_stencil_index;
        candidate_sink.wrong_side_mask = sample.wrong_side_mask;
        prepare_restrict_owner_result_sink_3d(candidate_sink);
        if (measure_benchmark) {
            result.query_seconds += run_restrict_owner_timed_sample_3d(
                grid, *preprocessor, candidate_sink,
                memory_sampler.get());
        } else {
            run_restrict_owner_sample_core_3d(
                grid, *preprocessor, candidate_sink);
        }
        restrict_owner_fingerprint_append(
            result.output_digest, candidate_sink.output_digest);
        if (result.comparison.has_value()) {
            const RestrictOwnerTraceStencil3D trace =
                build_restrict_owner_trace_stencil_3d(
                    grid, grid_pair, sample.target_dof,
                    sample.side == 0, sample.query,
                    sample.lower_stencil_index);
            std::uint64_t trace_mask = 0;
            for (std::size_t slot = 0; slot < 64; ++slot) {
                if (trace.owner_input.wrong_side[slot]) {
                    trace_mask |= UINT64_C(1)
                        << static_cast<unsigned>(slot);
                }
            }
            if (trace_mask != sample.wrong_side_mask)
                throw std::logic_error(
                    "candidate replay seed mask mismatch");
            accumulate_owner_accuracy_sample_3d(
                result.comparison.value(), cloud, sample, trace,
                candidate_sink);
        }
    }
    if (measure_benchmark) {
        memory_sampler->finish();
        result.peak_working_set_bytes = memory_sampler->peak_bytes();
        result.max_active_increase_bytes =
            memory_sampler->max_active_increase_bytes();
    }
    result.diagnostics = preprocessor->diagnostics();
    validate_restrict_owner_preprocess_diagnostics(
        result.diagnostics, expected_queries);
    if (result.comparison.has_value()) {
        if (result.comparison->accuracy.query_count != expected_queries)
            throw std::logic_error("candidate comparison query count mismatch");
        finalize_owner_accuracy_comparison_3d(
            result.comparison.value());
    }
    return result;
}

void write_owner_study_checkpoints_3d(
    const std::filesystem::path& output_dir,
    const OwnerPreprocessStudyRows3D& rows)
{
    std::filesystem::create_directories(output_dir);
    std::ofstream timing = open_output_file(
        output_dir / "preprocess_timing_repeats.csv");
    timing << std::setprecision(17)
        << "case_id,geometry,pose,N,h,mode,repetition,run_order,"
           "wrong_side_queries,construction_seconds,query_seconds,"
           "total_preprocess_seconds,queries_per_second,"
           "peak_working_set_bytes,working_set_increase_bytes"
        << std::endl;
    for (const auto& row : rows.timing) {
        timing << row.case_id << ',' << row.geometry << ',' << row.pose << ','
               << row.N << ',' << row.h << ',' << row.mode << ','
               << row.repetition << ',' << row.run_order << ','
               << row.wrong_side_queries << ',' << row.construction_seconds
               << ',' << row.query_seconds << ','
               << row.total_preprocess_seconds << ','
               << row.queries_per_second << ','
               << row.peak_working_set_bytes << ','
               << row.working_set_increase_bytes << '\n';
    }
    std::ofstream summary = open_output_file(
        output_dir / "preprocess_summary.csv");
    summary << std::setprecision(17)
        << "case_id,geometry,pose,N,h,mode,repetitions,wrong_side_queries,"
           "construction_seconds_median,query_seconds_median,"
           "query_seconds_p95,total_seconds_median,total_seconds_p95,"
           "queries_per_second_median,speedup_vs_full,"
           "peak_working_set_bytes_max,accepted" << std::endl;
    for (const auto& row : rows.summary) {
        summary << row.case_id << ',' << row.geometry << ',' << row.pose << ','
                << row.N << ',' << row.h << ',' << row.mode << ','
                << row.repetitions << ',' << row.wrong_side_queries << ','
                << row.construction_seconds_median << ','
                << row.query_seconds_median << ',' << row.query_seconds_p95
                << ',' << row.total_seconds_median << ','
                << row.total_seconds_p95 << ','
                << row.queries_per_second_median << ',' << row.speedup_vs_full
                << ',' << row.peak_working_set_bytes_max << ','
                << row.accepted << '\n';
    }
    std::ofstream accuracy = open_output_file(
        output_dir / "preprocess_accuracy.csv");
    accuracy << std::setprecision(17)
        << "case_id,geometry,pose,N,h,mode,query_count,oracle_target,"
           "oracle_foreign,owner_mismatch,foreign_true_positive,"
           "foreign_false_positive,foreign_false_negative,"
           "foreign_wrong_owner,foreign_precision,foreign_recall,"
           "max_mismatched_abs_weight,max_owner_correction_abs_difference,"
           "legacy_reason_mismatch,pass" << std::endl;
    for (const auto& row : rows.accuracy) {
        accuracy << row.case_id << ',' << row.geometry << ',' << row.pose << ','
                 << row.N << ',' << row.h << ',' << row.mode << ','
                 << row.query_count << ',' << row.oracle_target << ','
                 << row.oracle_foreign << ',' << row.owner_mismatch << ','
                 << row.foreign_true_positive << ','
                 << row.foreign_false_positive << ','
                 << row.foreign_false_negative << ','
                 << row.foreign_wrong_owner << ',' << row.foreign_precision
                 << ',' << row.foreign_recall << ','
                 << row.max_mismatched_abs_weight << ','
                 << row.max_owner_correction_abs_difference << ','
                 << row.legacy_reason_mismatch << ',' << row.pass << '\n';
    }
    std::ofstream paths = open_output_file(
        output_dir / "preprocess_path_counts.csv");
    paths << std::setprecision(17)
        << "case_id,geometry,pose,N,h,mode,path,count,fraction,"
           "compatible_aabb_candidates,foreign_aabb_candidates,"
           "control_hull_rejections,closest_attempts,closest_converged,"
           "closest_iterations,closest_certified_misses,"
           "closest_certified_roots,closest_unresolved,optimized_calls,"
           "full_fallback_calls,region_seconds,closest_seconds,"
           "optimized_seconds,full_fallback_seconds" << std::endl;
    for (const auto& row : rows.paths) {
        const auto& d = row.diagnostics;
        paths << row.case_id << ',' << row.geometry << ',' << row.pose << ','
              << row.N << ',' << row.h << ',' << row.mode << ',' << row.path
              << ',' << row.count << ',' << row.fraction << ','
              << d.compatible_aabb_candidates << ','
              << d.foreign_aabb_candidates << ','
              << d.control_hull_rejections << ','
              << d.closest_point_attempts << ',' << d.closest_point_converged
              << ',' << d.closest_point_iterations << ','
              << d.closest_certified_misses << ','
              << d.closest_certified_roots << ',' << d.closest_unresolved
              << ',' << d.optimized_intersection_calls << ','
              << d.full_fallback_calls << ',' << d.region_seconds << ','
              << d.closest_point_seconds << ','
              << d.optimized_intersection_seconds << ','
              << d.full_fallback_seconds << '\n';
    }
    std::ofstream mismatches = open_output_file(
        output_dir / "preprocess_mismatches.csv");
    mismatches << std::setprecision(17)
        << "case_id,geometry,pose,N,h,mode,target_dof,target_patch,side,layer,"
           "grid_node,weight,query_x,query_y,query_z,support_x,support_y,"
           "support_z,oracle_owner_dof,oracle_owner_patch,"
           "candidate_owner_dof,candidate_owner_patch,oracle_class,"
           "candidate_class,candidate_path,oracle_crossing_patch,"
           "candidate_crossing_patch" << std::endl;
    for (const auto& row : rows.mismatches) {
        mismatches << row.case_id << ',' << row.geometry << ',' << row.pose
                   << ',' << row.N << ',' << row.h << ',' << row.mode << ','
                   << row.target_dof << ',' << row.target_patch << ','
                   << row.side << ',' << row.layer << ',' << row.grid_node
                   << ',' << row.weight << ',' << row.query.x() << ','
                   << row.query.y() << ',' << row.query.z() << ','
                   << row.support.x() << ',' << row.support.y() << ','
                   << row.support.z() << ',' << row.oracle_owner_dof << ','
                   << row.oracle_owner_patch << ','
                   << row.candidate_owner_dof << ','
                   << row.candidate_owner_patch << ',' << row.oracle_class
                   << ',' << row.candidate_class << ',' << row.candidate_path
                   << ',' << row.oracle_crossing_patch << ','
                   << row.candidate_crossing_patch << '\n';
    }
    std::ofstream numerical = open_output_file(
        output_dir / "kfbi_numerical_results.csv");
    numerical << std::setprecision(17)
        << "case_id,geometry,pose,N,h,mode,dofs,preprocess_seconds,"
           "pipeline_setup_seconds,pipeline_speedup_vs_full,solve_seconds,"
           "converged,iterations,final_residual,operator_residual_linf,"
           "exterior_condition_linf,boundary_residual_linf,"
           "route_mismatch_linf,interior_linf,interior_order,"
           "exact_grid_linf,exact_equation_linf,"
           "interior_relative_difference_vs_full,"
           "exact_grid_relative_difference_vs_full,"
           "exact_equation_relative_difference_vs_full,"
           "geometry_queries_before_gmres,geometry_queries_after_gmres,pass"
        << std::endl;
    for (const auto& row : rows.numerical) {
        numerical << row.case_id << ',' << row.geometry << ',' << row.pose
                  << ',' << row.N << ',' << row.h << ',' << row.mode << ','
                  << row.dofs << ',' << row.preprocess_seconds << ','
                  << row.pipeline_setup_seconds << ','
                  << row.pipeline_speedup_vs_full << ',' << row.solve.seconds
                  << ',' << row.solve.converged << ',' << row.solve.iterations
                  << ',' << row.solve.gmres_relative_residual << ','
                  << row.solve.operator_residual_linf << ','
                  << row.solve.exterior_condition_linf << ','
                  << row.solve.boundary_residual_linf << ','
                  << row.solve.route_mismatch_linf << ','
                  << row.solve.interior_linf << ',' << row.interior_order << ','
                  << row.exact_grid_linf << ',' << row.exact_equation_linf
                  << ',' << row.interior_relative_difference_vs_full << ','
                  << row.exact_grid_relative_difference_vs_full << ','
                  << row.exact_equation_relative_difference_vs_full << ','
                  << row.geometry_queries_before_gmres << ','
                  << row.geometry_queries_after_gmres << ',' << row.pass
                  << '\n';
    }
}

struct OwnerNumericalEvaluation3D {
    SolveMetrics3D solve;
    double exact_grid_linf = 0.0;
    double exact_equation_linf = 0.0;
    std::uint64_t geometry_queries_before_gmres = 0;
    std::uint64_t geometry_queries_after_gmres = 0;
};

OwnerNumericalEvaluation3D evaluate_owner_pipeline_numerics_3d(
    const CartesianGrid3D& grid,
    const GridPair3D& grid_pair,
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const app3d::RigidTransform3D& transform,
    int gmres_max_iterations)
{
    const int size = pipeline.surface_size();
    Eigen::VectorXd value_data(size), exact_normal(size);
    for (int q = 0; q < size; q += 1) {
        const SurfaceDof& dof = pipeline.surface().dofs[
            static_cast<std::size_t>(q)];
        value_data[q] = app3d::transformed_manufactured_harmonic_value_3d(
            transform, dof.point);
        exact_normal[q] =
            app3d::transformed_manufactured_harmonic_gradient_3d(
                transform, dof.point).dot(dof.normal);
    }
    Eigen::VectorXd piecewise_exact(grid.num_dofs());
    for (int node = 0; node < grid.num_dofs(); node += 1) {
        const double value =
            app3d::transformed_manufactured_harmonic_value_3d(
                transform, grid_point(grid, node));
        piecewise_exact[node] = grid_pair.domain_label(node) > 0
            ? value : 0.0;
    }
    const HarmonicJetField3D field = pipeline.field_from_grid_and_jumps(
        piecewise_exact, value_data, exact_normal);
    const auto restrict_mode =
        ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner;
    const Eigen::VectorXd exact_grid = pipeline.exterior_normal_trace(
        field, value_data, exact_normal, restrict_mode);
    ExteriorNormalTraceOperator3D op(pipeline, restrict_mode);
    Eigen::VectorXd applied;
    op.apply(exact_normal, applied);
    const Eigen::VectorXd exact_equation =
        applied - op.right_hand_side(value_data);

    OwnerNumericalEvaluation3D result;
    result.exact_grid_linf = vector_linf(exact_grid);
    result.exact_equation_linf = vector_linf(exact_equation);
    const auto before = pipeline.restrict_owner_preprocess_diagnostics();
    result.geometry_queries_before_gmres =
        pipeline.restrict_owner_geometry_query_count();
    result.solve = run_dirichlet_normal_case(
        grid, grid_pair, pipeline, transform, gmres_max_iterations,
        restrict_mode);
    const auto after = pipeline.restrict_owner_preprocess_diagnostics();
    result.geometry_queries_after_gmres =
        pipeline.restrict_owner_geometry_query_count();
    if (restrict_owner_preprocess_diagnostics_equal(before, after) == false)
        throw std::logic_error("GMRES changed owner preprocessing diagnostics");
    return result;
}

void append_owner_path_rows_3d(
    OwnerPreprocessStudyRows3D& rows,
    OwnerPreprocessStudyCase3D study_case,
    int N,
    double h,
    OwnerMode3D mode,
    app3d::RestrictOwnerPreprocessDiagnostics3D diagnostics)
{
    const std::string mode_name =
        app3d::restrict_owner_preprocess_mode_name_3d(mode);
    for (std::size_t index = 0;
         index not_eq diagnostics.path_counts.size(); index += 1) {
        PreprocessPathCountRow3D row;
        row.case_id = study_case.case_id;
        row.geometry = study_case.geometry;
        row.pose = study_case.pose;
        row.mode = mode_name;
        row.N = N;
        row.h = h;
        row.path = owner_path_name_3d(static_cast<OwnerPath3D>(index));
        row.count = diagnostics.path_counts[index];
        row.fraction = diagnostics.wrong_side_queries == 0 ? 0.0
            : static_cast<double>(row.count)
              / static_cast<double>(diagnostics.wrong_side_queries);
        row.diagnostics = diagnostics;
        rows.paths.push_back(std::move(row));
    }
    for (std::size_t index = 1;
         index not_eq diagnostics.fallback_counts.size(); index += 1) {
        PreprocessPathCountRow3D row;
        row.case_id = study_case.case_id;
        row.geometry = study_case.geometry;
        row.pose = study_case.pose;
        row.mode = mode_name;
        row.N = N;
        row.h = h;
        row.path = "full_intersection_fallback:"
            + owner_fallback_name_3d(static_cast<OwnerFallback3D>(index));
        row.count = diagnostics.fallback_counts[index];
        row.fraction = diagnostics.wrong_side_queries == 0 ? 0.0
            : static_cast<double>(row.count)
              / static_cast<double>(diagnostics.wrong_side_queries);
        row.diagnostics = diagnostics;
        rows.paths.push_back(std::move(row));
    }
}

void append_owner_summary_row_3d(
    OwnerPreprocessStudyRows3D& rows,
    OwnerPreprocessStudyCase3D study_case,
    int N,
    double h,
    OwnerMode3D mode,
    bool accepted,
    double full_total_median)
{
    const std::string mode_name =
        app3d::restrict_owner_preprocess_mode_name_3d(mode);
    std::vector<double> construction, query, total, throughput;
    std::uint64_t queries = 0, peak = 0;
    for (const auto& timing : rows.timing) {
        if (timing.case_id == study_case.case_id
            and timing.N == N and timing.mode == mode_name) {
            construction.push_back(timing.construction_seconds);
            query.push_back(timing.query_seconds);
            total.push_back(timing.total_preprocess_seconds);
            throughput.push_back(timing.queries_per_second);
            queries = timing.wrong_side_queries;
            peak = std::max(peak, timing.peak_working_set_bytes);
        }
    }
    if (total.empty())
        throw std::logic_error("missing owner timing rows");
    PreprocessSummaryRow3D row;
    row.case_id = study_case.case_id;
    row.geometry = study_case.geometry;
    row.pose = study_case.pose;
    row.mode = mode_name;
    row.N = N;
    row.h = h;
    row.repetitions = static_cast<int>(total.size());
    row.wrong_side_queries = queries;
    row.construction_seconds_median = sorted_median_3d(construction);
    row.query_seconds_median = sorted_median_3d(query);
    row.query_seconds_p95 = sorted_nearest_rank_p95_3d(query);
    row.total_seconds_median = sorted_median_3d(total);
    row.total_seconds_p95 = sorted_nearest_rank_p95_3d(total);
    row.queries_per_second_median = sorted_median_3d(throughput);
    row.speedup_vs_full = full_total_median / row.total_seconds_median;
    row.peak_working_set_bytes_max = peak;
    row.accepted = accepted;
    rows.summary.push_back(std::move(row));
}

void append_owner_timing_row_3d(
    OwnerPreprocessStudyRows3D& rows,
    OwnerPreprocessStudyCase3D study_case,
    int N,
    OwnerMode3D mode,
    int repetition,
    int run_order,
    std::uint64_t queries,
    double construction_seconds,
    double query_seconds,
    std::uint64_t peak_bytes,
    std::uint64_t max_active_increase_bytes)
{
    PreprocessTimingRepeatRow3D row;
    row.case_id = study_case.case_id;
    row.geometry = study_case.geometry;
    row.pose = study_case.pose;
    row.mode = app3d::restrict_owner_preprocess_mode_name_3d(mode);
    row.N = N;
    row.h = kBoxSide / static_cast<double>(N);
    row.repetition = repetition;
    row.run_order = run_order;
    row.wrong_side_queries = queries;
    row.construction_seconds = construction_seconds;
    row.query_seconds = query_seconds;
    row.total_preprocess_seconds = construction_seconds + query_seconds;
    row.queries_per_second = query_seconds > 0.0
        ? static_cast<double>(queries) / query_seconds : 0.0;
    row.peak_working_set_bytes = peak_bytes;
    row.working_set_increase_bytes =
        max_active_increase_bytes;
    rows.timing.push_back(std::move(row));
}

KfbiNumericalResultRow3D make_owner_numerical_row_3d(
    OwnerPreprocessStudyCase3D study_case,
    int N,
    OwnerMode3D mode,
    const CartesianGrid3D& grid,
    const GridPair3D& grid_pair,
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    double preprocess_seconds,
    double pipeline_setup_seconds,
    double full_pipeline_setup_seconds,
    int gmres_max_iterations)
{
    const OwnerNumericalEvaluation3D evaluated =
        evaluate_owner_pipeline_numerics_3d(
            grid, grid_pair, pipeline, study_case.transform,
            gmres_max_iterations);
    KfbiNumericalResultRow3D row;
    row.case_id = study_case.case_id;
    row.geometry = study_case.geometry;
    row.pose = study_case.pose;
    row.mode = app3d::restrict_owner_preprocess_mode_name_3d(mode);
    row.N = N;
    row.h = grid.spacing()[0];
    row.dofs = pipeline.surface_size();
    row.preprocess_seconds = preprocess_seconds;
    row.pipeline_setup_seconds = pipeline_setup_seconds;
    row.pipeline_speedup_vs_full =
        full_pipeline_setup_seconds / pipeline_setup_seconds;
    row.solve = evaluated.solve;
    row.exact_grid_linf = evaluated.exact_grid_linf;
    row.exact_equation_linf = evaluated.exact_equation_linf;
    row.geometry_queries_before_gmres =
        evaluated.geometry_queries_before_gmres;
    row.geometry_queries_after_gmres =
        evaluated.geometry_queries_after_gmres;
    row.pass = row.solve.converged
        and row.solve.gmres_relative_residual <= 2.0e-10
        and row.geometry_queries_before_gmres
            == row.geometry_queries_after_gmres;
    return row;
}

void assign_owner_interior_order_3d(
    KfbiNumericalResultRow3D& row,
    const std::vector<KfbiNumericalResultRow3D>& previous_rows)
{
    const KfbiNumericalResultRow3D* previous = nullptr;
    for (const auto& candidate : previous_rows) {
        if (candidate.case_id == row.case_id and candidate.mode == row.mode
            and candidate.N < row.N
            and (previous == nullptr or candidate.N > previous->N)) {
            previous = std::addressof(candidate);
        }
    }
    if (previous != nullptr) {
        row.interior_order = observed_order(
            previous->solve.interior_linf, row.solve.interior_linf,
            previous->h, row.h);
    }
}

bool run_owner_preprocess_case_3d(
    OwnerPreprocessStudyCase3D study_case,
    int N,
    int gmres_max_iterations,
    OwnerPreprocessStudyRows3D& rows)
{
    const double h = kBoxSide / static_cast<double>(N);
    CartesianGrid3D grid({kBoxMin, kBoxMin, kBoxMin},
                         {h, h, h}, {N, N, N}, DofLayout3D::Node);
    GeometryBundle geometry = make_geometry(
        study_case.kind, h, study_case.transform);
    const auto domain = std::make_shared<const
        geometry3d::NurbsCartesianDomain3D>(
            grid, geometry.native_surface.geometry_model());
    const SurfaceDofCloud surface_dofs =
        app3d::make_native_surface_dofs_3d(geometry.native_surface, h);
    validate_surface_dofs(surface_dofs, h);
    const auto build_cauchy_fit = [&] {
        return app3d::HarmonicCauchyFit3D::build_legacy(
            geometry.native_surface, surface_dofs, h,
            app3d::LegacySurfaceCauchyPolicy3D::G1Nearest,
            kCauchyPolynomialDegree, kCauchyValueNeighborCount,
            kCauchyDerivativeNeighborCount);
    };
    GridPair3D grid_pair(grid, geometry.correction_interface,
                         geometry.crossing_interface, domain);
    for (int node = 0; node < grid.num_dofs(); node += 1) {
        const bool numerical_inside = grid_pair.domain_label(node) > 0;
        if (numerical_inside
            not_eq geometry.exact_inside(grid_point(grid, node))) {
            throw std::runtime_error("owner study native NURBS label mismatch");
        }
    }

    RestrictOwnerWorkload3D workload;
    RestrictOwnerPipelinePreprocessTiming3D full_preprocess_timing;
    const auto full_pipeline_start = std::chrono::steady_clock::now();
    PanelCenterHarmonicJetKFBI3D full_pipeline(
        grid, grid_pair, geometry.native_surface,
        geometry.correction_triangles, geometry.geometry_triangles,
        surface_dofs, build_cauchy_fit(), false,
        OwnerMode3D::FullIntersectionReference, nullptr, &workload,
        &full_preprocess_timing);
    const double full_pipeline_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - full_pipeline_start).count();
    const double full_pipeline_seconds =
        production_pipeline_setup_seconds_3d(
            full_pipeline_wall_seconds, full_preprocess_timing);
    const auto full_diagnostics =
        full_pipeline.restrict_owner_preprocess_diagnostics();
    if (workload.empty())
        throw std::logic_error("owner study captured an empty workload");
    const std::uint64_t expected_queries =
        restrict_owner_oracle_query_count_3d(workload);
    validate_restrict_owner_preprocess_diagnostics(
        full_diagnostics, expected_queries);
    if (full_pipeline.restrict_owner_geometry_query_count()
        != expected_queries) {
        throw std::logic_error(
            "full pipeline geometry query count mismatch");
    }
    append_owner_timing_row_3d(
        rows, study_case, N, OwnerMode3D::FullIntersectionReference,
        0, 0, full_diagnostics.wrong_side_queries,
        full_preprocess_timing.construction_seconds,
        full_preprocess_timing.query_seconds,
        full_preprocess_timing.peak_working_set_bytes,
        full_preprocess_timing.max_active_increase_bytes);
    const std::array<OwnerMode3D, 2> candidates{{
        OwnerMode3D::OptimizedIntersection,
        OwnerMode3D::RegionClosestHybrid}};
    std::array<std::uint64_t, 2> candidate_output_digests{};
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const RestrictOwnerReplayResult3D warmup =
            replay_restrict_owner_workload_3d(
            grid, grid_pair, geometry.native_surface, surface_dofs,
            workload, candidates[index], {}, false);
        candidate_output_digests[index] = warmup.output_digest;
    }
    const int repetitions = owner_candidate_repetitions_3d(N);
    for (int repetition = 0; repetition < repetitions; repetition += 1) {
        std::array<OwnerMode3D, 2> order = candidates;
        if (repetition % 2 == 1)
            std::reverse(order.begin(), order.end());
        for (int position = 0; position < 2; position += 1) {
            const OwnerMode3D mode = order[static_cast<std::size_t>(position)];
            const RestrictOwnerReplayResult3D replay =
                replay_restrict_owner_workload_3d(
                    grid, grid_pair, geometry.native_surface, surface_dofs,
                    workload, mode, {}, true);
            const std::size_t mode_index = mode == candidates[0] ? 0 : 1;
            if (replay.output_digest
                != candidate_output_digests[mode_index]) {
                throw std::logic_error(
                    "candidate timed replay output is nondeterministic");
            }
            append_owner_timing_row_3d(
                rows, study_case, N, mode, repetition,
                repetition * 2 + position + 1,
                replay.diagnostics.wrong_side_queries,
                replay.construction_seconds, replay.query_seconds,
                replay.peak_working_set_bytes,
                replay.max_active_increase_bytes);
        }
    }
    std::array<bool, 2> candidate_pass{{false, false}};
    std::array<std::size_t, 2> accuracy_row_indices{};
    for (std::size_t index = 0; index not_eq candidates.size(); index += 1) {
        app3d::RestrictOwnerPreprocessOptions3D instrumented_options;
        instrumented_options.collect_stage_timings = true;
        RestrictOwnerReplayResult3D replay =
            replay_restrict_owner_workload_3d(
                grid, grid_pair, geometry.native_surface, surface_dofs,
                workload, candidates[index], instrumented_options, false,
                std::addressof(study_case), N);
        if (!replay.comparison.has_value())
            throw std::logic_error("instrumented replay omitted accuracy");
        if (replay.output_digest != candidate_output_digests[index])
            throw std::logic_error(
                "candidate instrumented replay output is nondeterministic");
        RestrictOwnerAccuracyComparison3D comparison =
            std::move(replay.comparison.value());
        candidate_pass[index] = comparison.accuracy.pass;
        accuracy_row_indices[index] = rows.accuracy.size();
        rows.accuracy.push_back(std::move(comparison.accuracy));
        rows.mismatches.insert(
            rows.mismatches.end(),
            std::make_move_iterator(comparison.mismatches.begin()),
            std::make_move_iterator(comparison.mismatches.end()));
        append_owner_path_rows_3d(
            rows, study_case, N, h, candidates[index], replay.diagnostics);
    }
    append_owner_path_rows_3d(
        rows, study_case, N, h, OwnerMode3D::FullIntersectionReference,
        full_diagnostics);
    const double full_total = full_preprocess_timing.construction_seconds
                            + full_preprocess_timing.query_seconds;
    const bool numerical_gate_required = N >= 32;
    const std::size_t full_summary_row_index = rows.summary.size();
    append_owner_summary_row_3d(
        rows, study_case, N, h, OwnerMode3D::FullIntersectionReference,
        !numerical_gate_required, full_total);
    std::array<std::size_t, 2> candidate_summary_row_indices{};
    for (std::size_t index = 0; index not_eq candidates.size(); index += 1) {
        candidate_summary_row_indices[index] = rows.summary.size();
        append_owner_summary_row_3d(
            rows, study_case, N, h, candidates[index],
            !numerical_gate_required && candidate_pass[index], full_total);
    }
    const bool all_candidates_pass = candidate_pass[0] and candidate_pass[1];
    if (N < 32 or all_candidates_pass == false)
        return all_candidates_pass;
    RestrictOwnerWorkload3D{}.swap(workload);
    KfbiNumericalResultRow3D full_numerical = make_owner_numerical_row_3d(
        study_case, N, OwnerMode3D::FullIntersectionReference,
        grid, grid_pair, full_pipeline, full_total, full_pipeline_seconds,
        full_pipeline_seconds, gmres_max_iterations);
    assign_owner_interior_order_3d(full_numerical, rows.numerical);
    const KfbiNumericalResultRow3D full_reference_numerical = full_numerical;
    rows.numerical.push_back(std::move(full_numerical));
    rows.summary[full_summary_row_index].accepted =
        full_reference_numerical.pass;
    bool numerical_pass = full_reference_numerical.pass;
    const double correction_tolerance =
        64.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, full_pipeline.restrict_owner_correction_linf());
    for (std::size_t index = 0; index not_eq candidates.size(); index += 1) {
        RestrictOwnerPipelinePreprocessTiming3D candidate_preprocess_timing;
        const auto pipeline_start = std::chrono::steady_clock::now();
        PanelCenterHarmonicJetKFBI3D candidate_pipeline(
            grid, grid_pair, geometry.native_surface,
            geometry.correction_triangles, geometry.geometry_triangles,
            surface_dofs, build_cauchy_fit(), false, candidates[index], nullptr,
            nullptr, &candidate_preprocess_timing);
        const double pipeline_wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pipeline_start).count();
        const double pipeline_seconds =
            production_pipeline_setup_seconds_3d(
                pipeline_wall_seconds, candidate_preprocess_timing);
        if (candidate_preprocess_timing.output_digest
            != candidate_output_digests[index]) {
            throw std::logic_error(
                "candidate pipeline output is nondeterministic");
        }
        if (candidate_pipeline.restrict_owner_workload_fingerprint()
            not_eq full_pipeline.restrict_owner_workload_fingerprint()) {
            throw std::logic_error("owner pipeline workload fingerprint mismatch");
        }
        const double correction_difference =
            full_pipeline.restrict_owner_correction_linf_difference(
                candidate_pipeline);
        PreprocessAccuracyRow3D& accuracy = rows.accuracy[
            accuracy_row_indices[index]];
        accuracy.max_owner_correction_abs_difference = correction_difference;
        accuracy.pass = accuracy.pass
            and correction_difference <= correction_tolerance;
        const double preprocess_seconds =
            candidate_preprocess_timing.construction_seconds
            + candidate_preprocess_timing.query_seconds;
        KfbiNumericalResultRow3D numerical = make_owner_numerical_row_3d(
            study_case, N, candidates[index], grid, grid_pair,
            candidate_pipeline, preprocess_seconds, pipeline_seconds,
            full_pipeline_seconds, gmres_max_iterations);
        assign_owner_interior_order_3d(numerical, rows.numerical);
        numerical.interior_relative_difference_vs_full =
            symmetric_relative_difference_3d(
                numerical.solve.interior_linf,
                full_reference_numerical.solve.interior_linf);
        numerical.exact_grid_relative_difference_vs_full =
            symmetric_relative_difference_3d(
                numerical.exact_grid_linf,
                full_reference_numerical.exact_grid_linf);
        numerical.exact_equation_relative_difference_vs_full =
            symmetric_relative_difference_3d(
                numerical.exact_equation_linf,
                full_reference_numerical.exact_equation_linf);
        numerical.pass = numerical.pass and accuracy.pass
            and numerical.solve.converged
                == full_reference_numerical.solve.converged
            and numerical.solve.iterations
                == full_reference_numerical.solve.iterations
            and numerical.interior_relative_difference_vs_full <= 1.0e-12
            and numerical.exact_grid_relative_difference_vs_full <= 1.0e-12
            and numerical.exact_equation_relative_difference_vs_full
                <= 1.0e-12;
        if (std::isfinite(numerical.interior_order)
            and std::isfinite(full_reference_numerical.interior_order)) {
            numerical.pass = numerical.pass
                and numerical.interior_order + 1.0e-12
                    >= full_reference_numerical.interior_order;
        }
        numerical_pass = numerical_pass and numerical.pass;
        rows.summary[candidate_summary_row_indices[index]].accepted =
            numerical.pass;
        rows.numerical.push_back(std::move(numerical));
    }
    return numerical_pass;
}

std::string owner_preprocess_failure_mode_3d(
    const OwnerPreprocessStudyRows3D& rows,
    const OwnerPreprocessStudyCase3D& study_case,
    int N)
{
    for (auto row = rows.numerical.rbegin(); row != rows.numerical.rend();
         ++row) {
        if (row->case_id == study_case.case_id && row->N == N
            && !row->pass) {
            return row->mode;
        }
    }
    for (auto row = rows.accuracy.rbegin(); row != rows.accuracy.rend();
         ++row) {
        if (row->case_id == study_case.case_id && row->N == N
            && !row->pass) {
            return row->mode;
        }
    }
    return "unknown";
}

int run_owner_preprocess_study_3d(std::vector<int> levels)
{
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    for (int N : levels) {
        if (N < 16 or is_power_of_two(N) == false)
            throw std::invalid_argument(
                "owner-preprocess-study N must be a power of two and at least 16");
        if (N > 128)
            throw std::invalid_argument(
                "owner-preprocess-study supports levels through 128");
    }
    const bool requests_128 = std::binary_search(
        levels.begin(), levels.end(), 128);
    if (requests_128) {
        const bool has_32 = std::binary_search(levels.begin(), levels.end(), 32);
        const bool has_64 = std::binary_search(levels.begin(), levels.end(), 64);
        if (has_32 == false or has_64 == false)
            throw std::invalid_argument(
                "N=128 requires selected N=32 and N=64 accuracy gates");
    }
#ifdef KFBIM_APP_OUTPUT_DIR
    std::filesystem::path output_dir =
        std::filesystem::path(KFBIM_APP_OUTPUT_DIR)
        / "restrict_owner_preprocess_study";
#else
    std::filesystem::path output_dir =
        "output/restrict_owner_preprocess_study";
#endif
    const char* output_override = std::getenv(
        "KFBIM_3D_RESTRICT_OWNER_STUDY_OUTPUT_DIR");
    if (output_override != nullptr)
        output_dir = output_override;
    OwnerPreprocessStudyRows3D rows;
    write_owner_study_checkpoints_3d(output_dir, rows);
    const auto cases = owner_study_cases_3d();
    const int gmres_max_iterations = positive_environment_integer(
        "KFBIM_3D_GMRES_MAX_ITERATIONS", 160);
    bool smaller_accuracy_pass = true;
    bool completed_smaller_level = false;
    for (int N : levels) {
        if (N == 128
            and (completed_smaller_level == false
                 or smaller_accuracy_pass == false)) {
            std::cerr << "error: N=128 blocked by smaller accuracy gate" << '\n';
            return 1;
        }
        for (const auto& study_case : cases) {
            if (owner_case_selected_for_level_3d(study_case, N) == false)
                continue;
            std::cout << "[owner-preprocess-study] case="
                      << study_case.case_id << " N=" << N << '\n';
            const bool pass = run_owner_preprocess_case_3d(
                study_case, N, gmres_max_iterations, rows);
            write_owner_study_checkpoints_3d(output_dir, rows);
            if (N < 128) {
                completed_smaller_level = true;
                smaller_accuracy_pass = smaller_accuracy_pass and pass;
            }
            if (pass == false) {
                std::cerr << "error: owner preprocessing gate failed: case="
                          << study_case.case_id << " mode="
                          << owner_preprocess_failure_mode_3d(
                                 rows, study_case, N)
                          << " N=" << N << '\n';
                return 1;
            }
        }
    }
    std::cout << "Owner preprocessing study output: "
              << output_dir.string() << '\n';
    return 0;
}

const char* exterior_value_restrict_route_name_3d(
    ExteriorValueRestrictMode3D mode)
{
    switch (mode) {
    case ExteriorValueRestrictMode3D::JointTricubicCauchy:
        return "joint_tricubic_cauchy";
    case ExteriorValueRestrictMode3D::JointTricubicCrossingOwner:
        return "joint_tricubic_crossing_owner";
    }
    throw std::logic_error("unknown exterior value restrict mode");
}

bool bitwise_equal_vector_3d(
    const Eigen::VectorXd& lhs,
    const Eigen::VectorXd& rhs) noexcept
{
    return lhs.size() == rhs.size()
        && (lhs.size() == 0
            || std::memcmp(
                   lhs.data(), rhs.data(),
                   static_cast<std::size_t>(lhs.size())
                       * sizeof(double)) == 0);
}

bool finite_neumann_owner_metrics_3d(const SolveMetrics3D& metrics)
{
    const std::array<double, 15> values{{
        metrics.seconds,
        metrics.gmres_relative_residual,
        metrics.operator_residual_linf,
        metrics.exterior_condition_linf,
        metrics.boundary_residual_linf,
        metrics.route_mismatch_linf,
        metrics.density_linf,
        metrics.density_l2,
        metrics.interior_linf,
        metrics.interior_l2,
        metrics.exterior_bulk_linf,
        metrics.exterior_bulk_l2,
        metrics.data_weighted_mean,
        metrics.density_weighted_mean,
        metrics.constant_shift}};
    return std::all_of(
        values.begin(), values.end(),
        [](double value) { return std::isfinite(value); });
}

struct NeumannOwnerStudyRow3D {
    app3d::LPrismRigidStudyCase3D study_case;
    int N = 0;
    double h = 0.0;
    std::string mode;
    int dofs = 0;
    double setup_seconds = 0.0;
    double pipeline_setup_seconds = 0.0;
    double total_seconds = 0.0;
    int label_mismatches = 0;
    int unsafe_label_changing_edges = 0;
    int gap_crossings = 0;
    int endpoint_crossings = 0;
    int triangle_fallback_crossings = 0;
    SolveMetrics3D solve;
    double interior_order = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> residual_history;
    std::string preprocess_mode;
    std::uint64_t workload_fingerprint = 0;
    std::uint64_t output_digest = 0;
    std::uint64_t wrong_side_queries = 0;
    std::uint64_t geometry_queries_before_gmres = 0;
    std::uint64_t geometry_queries_after_gmres = 0;
    bool diagnostics_unchanged = false;
    bool default_route_bitwise_equal = false;
    bool probe_invariants_pass = false;
    bool pass = false;
};

void assign_neumann_owner_order_3d(
    NeumannOwnerStudyRow3D& row,
    const std::vector<NeumannOwnerStudyRow3D>& previous_rows)
{
    const NeumannOwnerStudyRow3D* previous = nullptr;
    for (const NeumannOwnerStudyRow3D& candidate : previous_rows) {
        if (candidate.study_case.id == row.study_case.id
            && candidate.mode == row.mode
            && candidate.N < row.N
            && (previous == nullptr || candidate.N > previous->N)) {
            previous = std::addressof(candidate);
        }
    }
    if (previous != nullptr) {
        row.interior_order = observed_order(
            previous->solve.interior_linf,
            row.solve.interior_linf,
            previous->h,
            row.h);
    }
}

void write_neumann_owner_study_checkpoints_3d(
    const std::filesystem::path& output_dir,
    const std::vector<NeumannOwnerStudyRow3D>& rows)
{
    std::filesystem::create_directories(output_dir);
    std::ofstream summary = open_output_file(output_dir / "summary.csv");
    summary << std::setprecision(17)
        << "N,h,mode,dofs,pipeline_setup_seconds,solve_seconds,"
           "converged,iterations,final_residual,operator_residual_linf,"
           "exterior_condition_linf,route_mismatch_linf,density_linf,"
           "density_l2,interior_linf,interior_l2,interior_order,"
           "geometry_queries_before_gmres,"
           "geometry_queries_after_gmres,pass\n";
    for (const NeumannOwnerStudyRow3D& row : rows) {
        summary << row.N << ',' << row.h << ',' << row.mode << ','
                << row.dofs << ',' << row.pipeline_setup_seconds << ','
                << row.solve.seconds << ',' << row.solve.converged << ','
                << row.solve.iterations << ','
                << row.solve.gmres_relative_residual << ','
                << row.solve.operator_residual_linf << ','
                << row.solve.exterior_condition_linf << ','
                << row.solve.route_mismatch_linf << ','
                << row.solve.density_linf << ',' << row.solve.density_l2
                << ',' << row.solve.interior_linf << ','
                << row.solve.interior_l2 << ',' << row.interior_order << ','
                << row.geometry_queries_before_gmres << ','
                << row.geometry_queries_after_gmres << ',' << row.pass
                << '\n';
    }

    std::ofstream residuals = open_output_file(
        output_dir / "gmres_residuals.csv");
    residuals << std::setprecision(17)
              << "N,mode,iteration,residual\n";
    for (const NeumannOwnerStudyRow3D& row : rows) {
        for (std::size_t iteration = 0;
             iteration < row.residual_history.size(); ++iteration) {
            residuals << row.N << ',' << row.mode << ',' << iteration
                      << ',' << row.residual_history[iteration] << '\n';
        }
    }

    std::ofstream diagnostics = open_output_file(
        output_dir / "owner_diagnostics.csv");
    diagnostics << std::setprecision(17)
        << "N,mode,preprocess_mode,workload_fingerprint,output_digest,"
           "wrong_side_queries,geometry_queries_before_gmres,"
           "geometry_queries_after_gmres,diagnostics_unchanged,"
           "default_route_bitwise_equal\n";
    for (const NeumannOwnerStudyRow3D& row : rows) {
        diagnostics << row.N << ',' << row.mode << ','
                    << row.preprocess_mode << ','
                    << row.workload_fingerprint << ','
                    << row.output_digest << ','
                    << row.wrong_side_queries << ','
                    << row.geometry_queries_before_gmres << ','
                    << row.geometry_queries_after_gmres << ','
                    << row.diagnostics_unchanged << ','
                    << row.default_route_bitwise_equal << '\n';
    }
}

int run_neumann_owner_study_3d(std::vector<int> levels)
{
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    for (int N : levels) {
        if (N < 32 || N > 128 || !is_power_of_two(N)) {
            throw std::invalid_argument(
                "neumann-owner-study N must be 32, 64, or 128");
        }
    }
    if (std::binary_search(levels.begin(), levels.end(), 128)
        && (!std::binary_search(levels.begin(), levels.end(), 32)
            || !std::binary_search(levels.begin(), levels.end(), 64))) {
        throw std::invalid_argument(
            "neumann-owner-study N=128 requires selected N=32 and N=64");
    }

#ifdef KFBIM_APP_OUTPUT_DIR
    std::filesystem::path output_dir =
        std::filesystem::path(KFBIM_APP_OUTPUT_DIR)
        / "neumann_value_trace_crossing_owner_3d";
#else
    std::filesystem::path output_dir =
        "output/neumann_value_trace_crossing_owner_3d";
#endif
    const char* output_override = std::getenv(
        "KFBIM_3D_NEUMANN_OWNER_STUDY_OUTPUT_DIR");
    if (output_override != nullptr)
        output_dir = output_override;

    std::optional<OwnerPreprocessStudyCase3D> selected_case;
    for (const OwnerPreprocessStudyCase3D& candidate
         : owner_study_cases_3d()) {
        if (candidate.case_id == "l_prism_rot_axis123_17deg") {
            selected_case = candidate;
            break;
        }
    }
    if (!selected_case.has_value())
        throw std::logic_error("missing rotated L-prism study case");

    const app3d::LPrismRigidStudyCase3D rigid_case =
        rigid_case_by_id_3d(
            app3d::make_l_prism_rigid_study_cases_3d(),
            selected_case->pose);
    constexpr int gmres_max_iterations = 80;
    constexpr double gmres_tolerance = 2.0e-10;
    const std::array<ExteriorValueRestrictMode3D, 2> modes{{
        ExteriorValueRestrictMode3D::JointTricubicCauchy,
        ExteriorValueRestrictMode3D::JointTricubicCrossingOwner}};
    std::vector<NeumannOwnerStudyRow3D> rows;
    write_neumann_owner_study_checkpoints_3d(output_dir, rows);

    for (int N : levels) {
        const double h = kBoxSide / static_cast<double>(N);
        std::cout << "[neumann-owner-study] N=" << N
                  << " h=" << h << " setup\n";
        CartesianGrid3D grid(
            {kBoxMin, kBoxMin, kBoxMin},
            {h, h, h},
            {N, N, N},
            DofLayout3D::Node);
        GeometryBundle geometry = make_geometry(
            GeometryKind::LPrism, h, selected_case->transform);
        const auto domain = std::make_shared<const
            geometry3d::NurbsCartesianDomain3D>(
                grid, geometry.native_surface.geometry_model());
        const SurfaceDofCloud surface_dofs =
            app3d::make_native_surface_dofs_3d(
                geometry.native_surface, h);
        validate_surface_dofs(surface_dofs, h);
        app3d::HarmonicCauchyFit3D cauchy_fit =
            app3d::HarmonicCauchyFit3D::build_legacy(
                geometry.native_surface, surface_dofs, h,
                app3d::LegacySurfaceCauchyPolicy3D::G1Nearest,
                kCauchyPolynomialDegree, kCauchyValueNeighborCount,
                kCauchyDerivativeNeighborCount);
        GridPair3D grid_pair(
            grid,
            geometry.correction_interface,
            geometry.crossing_interface,
            domain);
        for (int node = 0; node < grid.num_dofs(); ++node) {
            const bool numerical_inside =
                grid_pair.domain_label(node) > 0;
            if (numerical_inside
                != geometry.exact_inside(grid_point(grid, node))) {
                throw std::runtime_error(
                    "neumann owner study native NURBS label mismatch");
            }
        }

        RestrictOwnerPipelinePreprocessTiming3D preprocess_timing;
        const auto setup_start = std::chrono::steady_clock::now();
        PanelCenterHarmonicJetKFBI3D pipeline(
            grid,
            grid_pair,
            geometry.native_surface,
            geometry.correction_triangles,
            geometry.geometry_triangles,
            surface_dofs,
            std::move(cauchy_fit),
            false,
            OwnerMode3D::RegionClosestHybrid,
            nullptr,
            nullptr,
            &preprocess_timing);
        const double setup_wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - setup_start).count();
        const double pipeline_setup_seconds =
            production_pipeline_setup_seconds_3d(
                setup_wall_seconds, preprocess_timing);
        const auto setup_diagnostics =
            pipeline.restrict_owner_preprocess_diagnostics();
        const std::uint64_t setup_queries = static_cast<std::uint64_t>(
            pipeline.restrict_owner_geometry_query_count());
        if (setup_queries != setup_diagnostics.wrong_side_queries) {
            throw std::logic_error(
                "Neumann owner pipeline query-count mismatch");
        }

        const int surface_size = pipeline.surface_size();
        Eigen::VectorXd normal_data(surface_size);
        for (int q = 0; q < surface_size; ++q) {
            const SurfaceDof& dof =
                pipeline.surface().dofs[static_cast<std::size_t>(q)];
            normal_data[q] =
                app3d::transformed_manufactured_harmonic_gradient_3d(
                    selected_case->transform, dof.point).dot(dof.normal);
        }
        normal_data.array() -= surface_weighted_mean(
            pipeline.surface(), normal_data);
        const Eigen::VectorXd zero_value =
            Eigen::VectorXd::Zero(surface_size);
        const HarmonicJetField3D probe_field =
            pipeline.field_from_grid_and_jumps(
                Eigen::VectorXd::Zero(grid.num_dofs()),
                zero_value,
                normal_data);
        const auto probe_before =
            pipeline.restrict_owner_preprocess_diagnostics();
        const Eigen::VectorXd default_probe = pipeline.exterior_trace(
            probe_field, zero_value, normal_data);
        const Eigen::VectorXd explicit_legacy_probe =
            pipeline.exterior_trace(
                probe_field,
                zero_value,
                normal_data,
                ExteriorValueRestrictMode3D::JointTricubicCauchy);
        const Eigen::VectorXd crossing_probe = pipeline.exterior_trace(
            probe_field,
            zero_value,
            normal_data,
            ExteriorValueRestrictMode3D::
                JointTricubicCrossingOwner);
        const auto probe_after =
            pipeline.restrict_owner_preprocess_diagnostics();
        const bool default_route_bitwise_equal =
            bitwise_equal_vector_3d(
                default_probe, explicit_legacy_probe);
        const bool probe_invariants_pass =
            default_route_bitwise_equal
            && crossing_probe.allFinite()
            && restrict_owner_preprocess_diagnostics_equal(
                   probe_before, probe_after)
            && static_cast<std::uint64_t>(
                   pipeline.restrict_owner_geometry_query_count())
                   == setup_queries;
        if (!probe_invariants_pass) {
            throw std::logic_error(
                "Neumann value-trace route invariant failed");
        }

        bool level_pass = true;
        for (ExteriorValueRestrictMode3D mode : modes) {
            NeumannOwnerStudyRow3D row;
            row.study_case = rigid_case;
            row.N = N;
            row.h = h;
            row.mode = exterior_value_restrict_route_name_3d(mode);
            row.dofs = surface_size;
            row.pipeline_setup_seconds = pipeline_setup_seconds;
            row.preprocess_mode =
                app3d::restrict_owner_preprocess_mode_name_3d(
                    OwnerMode3D::RegionClosestHybrid);
            row.workload_fingerprint =
                pipeline.restrict_owner_workload_fingerprint();
            row.output_digest = preprocess_timing.output_digest;
            row.wrong_side_queries =
                setup_diagnostics.wrong_side_queries;
            row.default_route_bitwise_equal =
                default_route_bitwise_equal;
            row.probe_invariants_pass = probe_invariants_pass;

            const auto before =
                pipeline.restrict_owner_preprocess_diagnostics();
            row.geometry_queries_before_gmres =
                static_cast<std::uint64_t>(
                    pipeline.restrict_owner_geometry_query_count());
            row.solve = run_neumann_case(
                grid,
                grid_pair,
                pipeline,
                selected_case->transform,
                gmres_max_iterations,
                mode,
                &row.residual_history);
            const auto after =
                pipeline.restrict_owner_preprocess_diagnostics();
            row.geometry_queries_after_gmres =
                static_cast<std::uint64_t>(
                    pipeline.restrict_owner_geometry_query_count());
            row.diagnostics_unchanged =
                restrict_owner_preprocess_diagnostics_equal(before, after);
            assign_neumann_owner_order_3d(row, rows);
            row.pass = probe_invariants_pass
                && row.solve.converged
                && row.solve.iterations <= gmres_max_iterations
                && row.solve.gmres_relative_residual <= gmres_tolerance
                && !row.residual_history.empty()
                && std::all_of(
                       row.residual_history.begin(),
                       row.residual_history.end(),
                       [](double value) {
                           return std::isfinite(value) && value >= 0.0;
                       })
                && finite_neumann_owner_metrics_3d(row.solve)
                && row.diagnostics_unchanged
                && row.geometry_queries_before_gmres
                    == row.geometry_queries_after_gmres;
            level_pass = level_pass && row.pass;
            std::cout
                << "[neumann-owner-study] N=" << row.N
                << " mode=" << row.mode
                << " dofs=" << row.dofs
                << " iter=" << row.solve.iterations
                << " residual=" << row.solve.gmres_relative_residual
                << " interior_linf=" << row.solve.interior_linf
                << " density_linf=" << row.solve.density_linf
                << " solve_s=" << row.solve.seconds
                << " pass=" << row.pass << '\n';
            rows.push_back(std::move(row));
        }
        write_neumann_owner_study_checkpoints_3d(output_dir, rows);
        if (!level_pass) {
            std::cerr
                << "error: Neumann owner study gate failed at N="
                << N << '\n';
            return 1;
        }
    }

    std::cout << "Neumann owner study output: "
              << output_dir.string() << '\n';
    return 0;
}

struct NeumannRigidGeometryDiagnostics3D {
    int unsafe_label_changing_edges = 0;
    int gap_crossings = 0;
    int endpoint_crossings = 0;
    int triangle_fallback_crossings = 0;
};

NeumannRigidGeometryDiagnostics3D
neumann_rigid_geometry_diagnostics_3d(
    const GridPair3D& grid_pair,
    const LaplaceCorrectionSupport3D& support)
{
    NeumannRigidGeometryDiagnostics3D result;
    result.unsafe_label_changing_edges =
        grid_pair.nurbs_domain_diagnostics()
            .unsafe_label_changing_edge_count;
    std::set<std::pair<int, int>> crossing_edges;
    for (const LaplaceCrossingCorrectionOp& op : support.crossing_ops) {
        const std::pair<int, int> edge{
            std::min(op.rhs_node, op.correction_node),
            std::max(op.rhs_node, op.correction_node)};
        if (!crossing_edges.insert(edge).second)
            continue;
        const P2CrossingOwner3D owner =
            grid_pair.p2_crossing_owner_between(edge.first, edge.second);
        if (owner.status == P2CrossingOwnerStatus3D::GapFallback)
            ++result.gap_crossings;
        else if (owner.status !=
                 P2CrossingOwnerStatus3D::ExactIntersection)
            ++result.endpoint_crossings;
        if (owner.nurbs_patch_index < 0)
            ++result.triangle_fallback_crossings;
    }
    return result;
}

bool neumann_rigid_geometry_pass_3d(
    const NeumannOwnerStudyRow3D& row)
{
    return row.label_mismatches == 0
        && row.unsafe_label_changing_edges == 0
        && row.gap_crossings == 0
        && row.endpoint_crossings == 0
        && row.triangle_fallback_crossings == 0;
}

bool neumann_rigid_owner_invariants_pass_3d(
    const NeumannOwnerStudyRow3D& row)
{
    return row.probe_invariants_pass
        && row.default_route_bitwise_equal
        && row.diagnostics_unchanged
        && row.geometry_queries_before_gmres
            == row.geometry_queries_after_gmres;
}

NeumannOwnerStudyRow3D run_neumann_rigid_pose_3d(
    int N,
    const app3d::LPrismRigidStudyCase3D& study_case)
{
    constexpr int gmres_max_iterations = 80;
    constexpr double gmres_tolerance = 2.0e-10;
    constexpr ExteriorValueRestrictMode3D mode =
        ExteriorValueRestrictMode3D::JointTricubicCrossingOwner;
    const auto case_start = std::chrono::steady_clock::now();
    const double h = kBoxSide / static_cast<double>(N);
    CartesianGrid3D grid(
        {kBoxMin, kBoxMin, kBoxMin},
        {h, h, h},
        {N, N, N},
        DofLayout3D::Node);
    GeometryBundle geometry = make_geometry(
        GeometryKind::LPrism, h, study_case.transform);
    const auto domain = std::make_shared<const
        geometry3d::NurbsCartesianDomain3D>(
            grid, geometry.native_surface.geometry_model());
    const SurfaceDofCloud surface_dofs =
        app3d::make_native_surface_dofs_3d(
            geometry.native_surface, h);
    validate_surface_dofs(surface_dofs, h);
    app3d::HarmonicCauchyFit3D cauchy_fit =
        app3d::HarmonicCauchyFit3D::build_legacy(
            geometry.native_surface, surface_dofs, h,
            app3d::LegacySurfaceCauchyPolicy3D::G1Nearest,
            kCauchyPolynomialDegree, kCauchyValueNeighborCount,
            kCauchyDerivativeNeighborCount);
    GridPair3D grid_pair(
        grid,
        geometry.correction_interface,
        geometry.crossing_interface,
        domain);
    int label_mismatches = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        const bool numerical_inside =
            grid_pair.domain_label(node) > 0;
        if (numerical_inside
            != geometry.exact_inside(grid_point(grid, node))) {
            ++label_mismatches;
        }
    }
    if (label_mismatches != 0) {
        throw std::runtime_error(
            "Neumann rigid study native NURBS label mismatch: case="
            + study_case.id + " N=" + std::to_string(N));
    }

    RestrictOwnerPipelinePreprocessTiming3D preprocess_timing;
    const auto pipeline_start = std::chrono::steady_clock::now();
    PanelCenterHarmonicJetKFBI3D pipeline(
        grid,
        grid_pair,
        geometry.native_surface,
        geometry.correction_triangles,
        geometry.geometry_triangles,
        surface_dofs,
        std::move(cauchy_fit),
        false,
        OwnerMode3D::RegionClosestHybrid,
        nullptr,
        nullptr,
        &preprocess_timing);
    const double setup_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - pipeline_start).count();
    const double pipeline_setup_seconds =
        production_pipeline_setup_seconds_3d(
            setup_wall_seconds, preprocess_timing);
    const auto setup_diagnostics =
        pipeline.restrict_owner_preprocess_diagnostics();
    const std::uint64_t setup_queries = static_cast<std::uint64_t>(
        pipeline.restrict_owner_geometry_query_count());
    if (setup_queries != setup_diagnostics.wrong_side_queries) {
        throw std::logic_error(
            "Neumann rigid pipeline query-count mismatch");
    }
    const NeumannRigidGeometryDiagnostics3D geometry_diagnostics =
        neumann_rigid_geometry_diagnostics_3d(
            grid_pair, pipeline.correction_support());

    const int surface_size = pipeline.surface_size();
    Eigen::VectorXd normal_data(surface_size);
    for (int q = 0; q < surface_size; ++q) {
        const SurfaceDof& dof =
            pipeline.surface().dofs[static_cast<std::size_t>(q)];
        normal_data[q] =
            app3d::transformed_manufactured_harmonic_gradient_3d(
                study_case.transform, dof.point).dot(dof.normal);
    }
    normal_data.array() -= surface_weighted_mean(
        pipeline.surface(), normal_data);
    const Eigen::VectorXd zero_value =
        Eigen::VectorXd::Zero(surface_size);
    const HarmonicJetField3D probe_field =
        pipeline.field_from_grid_and_jumps(
            Eigen::VectorXd::Zero(grid.num_dofs()),
            zero_value,
            normal_data);
    const auto probe_before =
        pipeline.restrict_owner_preprocess_diagnostics();
    const Eigen::VectorXd default_probe = pipeline.exterior_trace(
        probe_field, zero_value, normal_data);
    const Eigen::VectorXd explicit_legacy_probe =
        pipeline.exterior_trace(
            probe_field,
            zero_value,
            normal_data,
            ExteriorValueRestrictMode3D::JointTricubicCauchy);
    const Eigen::VectorXd crossing_probe = pipeline.exterior_trace(
        probe_field, zero_value, normal_data, mode);
    const auto probe_after =
        pipeline.restrict_owner_preprocess_diagnostics();
    const bool default_route_bitwise_equal =
        bitwise_equal_vector_3d(default_probe, explicit_legacy_probe);
    const bool probe_invariants_pass =
        default_route_bitwise_equal
        && crossing_probe.allFinite()
        && restrict_owner_preprocess_diagnostics_equal(
               probe_before, probe_after)
        && static_cast<std::uint64_t>(
               pipeline.restrict_owner_geometry_query_count())
               == setup_queries;

    NeumannOwnerStudyRow3D row;
    row.study_case = study_case;
    row.N = N;
    row.h = h;
    row.mode = exterior_value_restrict_route_name_3d(mode);
    row.dofs = surface_size;
    row.pipeline_setup_seconds = pipeline_setup_seconds;
    row.preprocess_mode =
        app3d::restrict_owner_preprocess_mode_name_3d(
            OwnerMode3D::RegionClosestHybrid);
    row.workload_fingerprint =
        pipeline.restrict_owner_workload_fingerprint();
    row.output_digest = preprocess_timing.output_digest;
    row.wrong_side_queries =
        setup_diagnostics.wrong_side_queries;
    row.label_mismatches = label_mismatches;
    row.unsafe_label_changing_edges =
        geometry_diagnostics.unsafe_label_changing_edges;
    row.gap_crossings = geometry_diagnostics.gap_crossings;
    row.endpoint_crossings =
        geometry_diagnostics.endpoint_crossings;
    row.triangle_fallback_crossings =
        geometry_diagnostics.triangle_fallback_crossings;
    row.default_route_bitwise_equal =
        default_route_bitwise_equal;
    row.probe_invariants_pass = probe_invariants_pass;
    row.setup_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - case_start).count();

    const auto before =
        pipeline.restrict_owner_preprocess_diagnostics();
    row.geometry_queries_before_gmres =
        static_cast<std::uint64_t>(
            pipeline.restrict_owner_geometry_query_count());
    row.solve = run_neumann_case(
        grid,
        grid_pair,
        pipeline,
        study_case.transform,
        gmres_max_iterations,
        mode,
        &row.residual_history);
    const auto after =
        pipeline.restrict_owner_preprocess_diagnostics();
    row.geometry_queries_after_gmres =
        static_cast<std::uint64_t>(
            pipeline.restrict_owner_geometry_query_count());
    row.diagnostics_unchanged =
        restrict_owner_preprocess_diagnostics_equal(before, after);
    row.total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - case_start).count();
    row.pass = row.probe_invariants_pass
        && neumann_rigid_geometry_pass_3d(row)
        && row.solve.converged
        && row.solve.iterations <= gmres_max_iterations
        && row.solve.gmres_relative_residual <= gmres_tolerance
        && !row.residual_history.empty()
        && std::all_of(
               row.residual_history.begin(),
               row.residual_history.end(),
               [](double value) {
                   return std::isfinite(value) && value >= 0.0;
               })
        && finite_neumann_owner_metrics_3d(row.solve)
        && neumann_rigid_owner_invariants_pass_3d(row)
        && std::isfinite(row.setup_seconds)
        && std::isfinite(row.pipeline_setup_seconds)
        && std::isfinite(row.total_seconds);
    return row;
}

std::vector<app3d::NeumannRigidStudyMeasurement3D>
neumann_rigid_measurements_3d(
    const std::vector<NeumannOwnerStudyRow3D>& rows)
{
    std::vector<app3d::NeumannRigidStudyMeasurement3D> result;
    result.reserve(rows.size());
    for (const NeumannOwnerStudyRow3D& row : rows) {
        app3d::NeumannRigidStudyMeasurement3D measurement;
        measurement.case_id = row.study_case.id;
        measurement.N = row.N;
        measurement.h = row.h;
        measurement.finite_metrics =
            finite_neumann_owner_metrics_3d(row.solve)
            && std::isfinite(row.setup_seconds)
            && std::isfinite(row.pipeline_setup_seconds)
            && std::isfinite(row.total_seconds);
        measurement.gmres_converged = row.solve.converged;
        measurement.gmres_iterations = row.solve.iterations;
        measurement.gmres_relative_residual =
            row.solve.gmres_relative_residual;
        measurement.interior_linf = row.solve.interior_linf;
        measurement.geometry_diagnostics_pass =
            neumann_rigid_geometry_pass_3d(row);
        measurement.owner_invariants_pass =
            neumann_rigid_owner_invariants_pass_3d(row);
        result.push_back(std::move(measurement));
    }
    return result;
}

const app3d::NeumannRigidStudyDerivedRow3D*
find_neumann_rigid_derived_row_3d(
    const app3d::NeumannRigidStudyEvaluation3D& evaluation,
    const NeumannOwnerStudyRow3D& raw)
{
    const auto found = std::find_if(
        evaluation.rows.begin(),
        evaluation.rows.end(),
        [&](const app3d::NeumannRigidStudyDerivedRow3D& row) {
            return row.measurement.case_id == raw.study_case.id
                && row.measurement.N == raw.N;
        });
    return found == evaluation.rows.end()
        ? nullptr : std::addressof(*found);
}

void write_neumann_rigid_study_checkpoints_3d(
    const std::filesystem::path& output_dir,
    const std::vector<NeumannOwnerStudyRow3D>& rows,
    const app3d::NeumannRigidStudyEvaluation3D& evaluation)
{
    std::filesystem::create_directories(output_dir);
    std::ofstream results = open_output_file(
        output_dir / "rigid_transform_results.csv");
    results << std::setprecision(17) << std::boolalpha
        << "case_id,N,h,mode,preprocess_mode,dofs,setup_seconds,"
           "pipeline_setup_seconds,solve_seconds,total_seconds,"
           "rotation_axis_x,rotation_axis_y,rotation_axis_z,"
           "rotation_angle_degrees,rotation_center_x,rotation_center_y,"
           "rotation_center_z,translation_x,translation_y,translation_z,"
           "rotation_00,rotation_01,rotation_02,rotation_10,rotation_11,"
           "rotation_12,rotation_20,rotation_21,rotation_22,"
           "converged,iterations,final_residual,operator_residual_linf,"
           "exterior_condition_linf,route_mismatch_linf,density_linf,"
           "density_l2,interior_linf,interior_l2,interior_order,"
           "baseline_error_ratio,baseline_iteration_ratio,"
           "label_mismatches,unsafe_label_changing_edges,gap_crossings,"
           "endpoint_crossings,triangle_fallback_crossings,"
           "geometry_queries_before_gmres,geometry_queries_after_gmres,"
           "diagnostics_unchanged,default_route_bitwise_equal,"
           "probe_invariants_pass,row_pass\n";
    for (const NeumannOwnerStudyRow3D& row : rows) {
        const auto* derived =
            find_neumann_rigid_derived_row_3d(evaluation, row);
        if (derived == nullptr)
            throw std::logic_error("missing Neumann rigid derived row");
        const auto& transform = row.study_case.transform;
        const Eigen::Matrix3d& rotation = transform.rotation();
        results << row.study_case.id << ',' << row.N << ',' << row.h
                << ',' << row.mode << ',' << row.preprocess_mode
                << ',' << row.dofs << ',' << row.setup_seconds
                << ',' << row.pipeline_setup_seconds
                << ',' << row.solve.seconds << ',' << row.total_seconds
                << ',' << row.study_case.rotation_axis.x()
                << ',' << row.study_case.rotation_axis.y()
                << ',' << row.study_case.rotation_axis.z()
                << ',' << row.study_case.rotation_angle_degrees
                << ',' << transform.center().x()
                << ',' << transform.center().y()
                << ',' << transform.center().z()
                << ',' << transform.translation().x()
                << ',' << transform.translation().y()
                << ',' << transform.translation().z();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j)
                results << ',' << rotation(i, j);
        }
        results << ',' << row.solve.converged
                << ',' << row.solve.iterations
                << ',' << row.solve.gmres_relative_residual
                << ',' << row.solve.operator_residual_linf
                << ',' << row.solve.exterior_condition_linf
                << ',' << row.solve.route_mismatch_linf
                << ',' << row.solve.density_linf
                << ',' << row.solve.density_l2
                << ',' << row.solve.interior_linf
                << ',' << row.solve.interior_l2
                << ',' << derived->interior_order
                << ',' << derived->baseline_error_ratio
                << ',' << derived->baseline_iteration_ratio
                << ',' << row.label_mismatches
                << ',' << row.unsafe_label_changing_edges
                << ',' << row.gap_crossings
                << ',' << row.endpoint_crossings
                << ',' << row.triangle_fallback_crossings
                << ',' << row.geometry_queries_before_gmres
                << ',' << row.geometry_queries_after_gmres
                << ',' << row.diagnostics_unchanged
                << ',' << row.default_route_bitwise_equal
                << ',' << row.probe_invariants_pass
                << ',' << criterion_status_name(derived->row_pass)
                << '\n';
    }

    std::ofstream residuals = open_output_file(
        output_dir / "gmres_residuals.csv");
    residuals << std::setprecision(17)
              << "case_id,N,mode,iteration,residual\n";
    for (const NeumannOwnerStudyRow3D& row : rows) {
        for (std::size_t iteration = 0;
             iteration < row.residual_history.size(); ++iteration) {
            residuals << row.study_case.id << ',' << row.N << ','
                      << row.mode << ',' << iteration << ','
                      << row.residual_history[iteration] << '\n';
        }
    }

    std::ofstream owners = open_output_file(
        output_dir / "owner_diagnostics.csv");
    owners << std::setprecision(17) << std::boolalpha
        << "case_id,N,mode,preprocess_mode,workload_fingerprint,"
           "output_digest,wrong_side_queries,"
           "geometry_queries_before_gmres,"
           "geometry_queries_after_gmres,diagnostics_unchanged,"
           "default_route_bitwise_equal,probe_invariants_pass\n";
    for (const NeumannOwnerStudyRow3D& row : rows) {
        owners << row.study_case.id << ',' << row.N << ',' << row.mode
               << ',' << row.preprocess_mode
               << ',' << row.workload_fingerprint
               << ',' << row.output_digest
               << ',' << row.wrong_side_queries
               << ',' << row.geometry_queries_before_gmres
               << ',' << row.geometry_queries_after_gmres
               << ',' << row.diagnostics_unchanged
               << ',' << row.default_route_bitwise_equal
               << ',' << row.probe_invariants_pass << '\n';
    }

    std::ofstream acceptance = open_output_file(
        output_dir / "rigid_transform_acceptance.csv");
    acceptance
        << "case_id,completeness_pass,row_pass,gmres_pass,"
           "monotone_error_pass,order_64_128_pass,baseline_ratio_pass,"
           "geometry_diagnostics_pass,owner_invariants_pass,overall_pass\n";
    for (const app3d::NeumannRigidStudyAcceptance3D& item
         : evaluation.cases) {
        acceptance << item.case_id
                   << ',' << criterion_status_name(item.completeness_pass)
                   << ',' << criterion_status_name(item.row_pass)
                   << ',' << criterion_status_name(item.gmres_pass)
                   << ',' << criterion_status_name(
                          item.monotone_error_pass)
                   << ',' << criterion_status_name(
                          item.order_64_128_pass)
                   << ',' << criterion_status_name(
                          item.baseline_ratio_pass)
                   << ',' << criterion_status_name(
                          item.geometry_diagnostics_pass)
                   << ',' << criterion_status_name(
                          item.owner_invariants_pass)
                   << ',' << criterion_status_name(item.overall_pass)
                   << '\n';
    }
}

int run_neumann_rigid_study_3d(std::vector<int> levels)
{
    levels = app3d::normalize_neumann_rigid_levels_3d(
        std::move(levels));
    const bool require_complete_acceptance =
        levels == std::vector<int>({32, 64, 128});
    const std::vector<app3d::LPrismRigidStudyCase3D> cases =
        app3d::make_l_prism_rigid_study_cases_3d();
    std::vector<std::string> case_ids;
    case_ids.reserve(cases.size());
    for (const auto& study_case : cases)
        case_ids.push_back(study_case.id);

#ifdef KFBIM_APP_OUTPUT_DIR
    std::filesystem::path output_dir =
        std::filesystem::path(KFBIM_APP_OUTPUT_DIR)
        / "neumann_rigid_transform_stability_3d";
#else
    std::filesystem::path output_dir =
        "output/neumann_rigid_transform_stability_3d";
#endif
    const char* output_override = std::getenv(
        "KFBIM_3D_NEUMANN_RIGID_STUDY_OUTPUT_DIR");
    if (output_override != nullptr)
        output_dir = output_override;

    std::vector<NeumannOwnerStudyRow3D> rows;
    rows.reserve(levels.size() * cases.size());
    auto evaluation = app3d::evaluate_neumann_rigid_study_3d(
        neumann_rigid_measurements_3d(rows),
        case_ids,
        require_complete_acceptance);
    write_neumann_rigid_study_checkpoints_3d(
        output_dir, rows, evaluation);

    std::cout
        << "KFBI3D Neumann L-prism rigid-transform stability study\n"
        << "  route=joint_tricubic_crossing_owner"
           " owner=region_closest_hybrid"
           " cauchy=g1_nearest/degree3/48/28\n"
        << "  gmres_tolerance=2e-10 restart=80 cap=80 levels=";
    for (std::size_t index = 0; index < levels.size(); ++index) {
        if (index != 0)
            std::cout << ',';
        std::cout << levels[index];
    }
    std::cout << " cases=" << cases.size() << '\n';

    for (int N : levels) {
        for (const auto& study_case : cases) {
            std::cout << "[neumann-rigid-study] case="
                      << study_case.id << " N=" << N << " setup\n";
            NeumannOwnerStudyRow3D row =
                run_neumann_rigid_pose_3d(N, study_case);
            assign_neumann_owner_order_3d(row, rows);
            rows.push_back(std::move(row));
            evaluation = app3d::evaluate_neumann_rigid_study_3d(
                neumann_rigid_measurements_3d(rows),
                case_ids,
                require_complete_acceptance);
            write_neumann_rigid_study_checkpoints_3d(
                output_dir, rows, evaluation);

            const NeumannOwnerStudyRow3D& stored = rows.back();
            const auto* derived =
                find_neumann_rigid_derived_row_3d(
                    evaluation, stored);
            std::cout << "[neumann-rigid-study] case="
                      << stored.study_case.id
                      << " N=" << stored.N
                      << " iter=" << stored.solve.iterations
                      << " residual="
                      << stored.solve.gmres_relative_residual
                      << " interior_linf="
                      << stored.solve.interior_linf
                      << " order="
                      << (derived == nullptr
                          ? std::numeric_limits<double>::quiet_NaN()
                          : derived->interior_order)
                      << " setup_s=" << stored.setup_seconds
                      << " solve_s=" << stored.solve.seconds
                      << " total_s=" << stored.total_seconds
                      << " pass=" << stored.pass << '\n';
            if (!stored.pass) {
                std::cerr
                    << "error: Neumann rigid execution gate failed: case="
                    << stored.study_case.id
                    << " N=" << stored.N << '\n';
                return 1;
            }
        }
    }

    evaluation = app3d::evaluate_neumann_rigid_study_3d(
        neumann_rigid_measurements_3d(rows),
        case_ids,
        require_complete_acceptance);
    write_neumann_rigid_study_checkpoints_3d(
        output_dir, rows, evaluation);
    for (const auto& item : evaluation.cases) {
        std::cout << "[neumann-rigid-acceptance] case="
                  << item.case_id
                  << " complete="
                  << criterion_status_name(item.completeness_pass)
                  << " gmres="
                  << criterion_status_name(item.gmres_pass)
                  << " monotone="
                  << criterion_status_name(item.monotone_error_pass)
                  << " order_64_128="
                  << criterion_status_name(item.order_64_128_pass)
                  << " baseline_ratio="
                  << criterion_status_name(item.baseline_ratio_pass)
                  << " geometry="
                  << criterion_status_name(
                         item.geometry_diagnostics_pass)
                  << " owner="
                  << criterion_status_name(item.owner_invariants_pass)
                  << " overall="
                  << criterion_status_name(item.overall_pass)
                  << '\n';
    }
    std::cout << "Neumann rigid-transform study output: "
              << output_dir.string() << '\n';
    const bool exit_pass =
        app3d::neumann_rigid_study_exit_pass_3d(
            evaluation, require_complete_acceptance);
    if (!exit_pass) {
        std::cerr
            << "error: Neumann rigid-study "
            << (require_complete_acceptance
                ? "complete numerical acceptance failed\n"
                : "execution integrity failed\n");
        return 1;
    }
    if (!require_complete_acceptance) {
        std::cout
            << "Neumann rigid-study prefix completed; full numerical "
               "acceptance is not enforced without N=32,64,128\n";
    }
    return 0;
}

struct NeumannEdgeSampleDiagnostics3D {
    app3d::NeumannEdgeConstraintSample3D sample;
    double exact_signed_mismatch = 0.0;
    double unconstrained_signed_mismatch = 0.0;
    double projected_signed_mismatch = 0.0;
};

struct NeumannEdgeStudyRow3D {
    app3d::LPrismRigidStudyCase3D study_case;
    int N = 0;
    double h = 0.0;
    app3d::NeumannDensitySpace3D density_space = app3d::NeumannDensitySpace3D::PatchIndependent;
    int dofs = 0;
    double geometry_setup_seconds = 0.0;
    double pipeline_setup_seconds = 0.0;
    double projector_setup_seconds = 0.0;
    double total_seconds = 0.0;
    SolveMetrics3D solve;
    std::vector<double> residual_history;
    double edge_mismatch_linf = 0.0;
    double edge_mismatch_weighted_rms = 0.0;
    double exact_edge_mismatch_linf = 0.0;
    int expected_non_g1_connections = 0;
    int covered_non_g1_connections = 0;
    int duplicate_connection_intervals = 0;
    int g1_constraint_rows = 0;
    int unrelated_constraint_rows = 0;
    int constraint_rows = 0;
    int constraint_rank = 0;
    int reduced_order_rows = 0;
    double constant_constraint_defect = 0.0;
    double projected_constraint_defect = 0.0;
    double projection_idempotence_defect = 0.0;
    double constant_projection_defect = 0.0;
    int label_mismatches = 0;
    int unsafe_label_changing_edges = 0;
    int gap_crossings = 0;
    int endpoint_crossings = 0;
    int triangle_fallback_crossings = 0;
    std::string preprocess_mode;
    std::uint64_t workload_fingerprint = 0;
    std::uint64_t output_digest = 0;
    std::uint64_t wrong_side_queries = 0;
    std::uint64_t workload_fingerprint_before_gmres = 0;
    std::uint64_t workload_fingerprint_after_gmres = 0;
    std::uint64_t output_digest_before_gmres = 0;
    std::uint64_t output_digest_after_gmres = 0;
    std::uint64_t wrong_side_queries_before_gmres = 0;
    std::uint64_t wrong_side_queries_after_gmres = 0;
    std::uint64_t geometry_queries_before_gmres = 0;
    std::uint64_t geometry_queries_after_gmres = 0;
    bool diagnostics_unchanged = false;
    bool default_route_bitwise_equal = false;
    bool probe_invariants_pass = false;
    bool geometry_diagnostics_pass = false;
    bool owner_invariants_pass = false;
    bool shared_preprocess_pass = false;
    std::vector<NeumannEdgeSampleDiagnostics3D> edge_samples;
};

bool finite_neumann_edge_row_3d(const NeumannEdgeStudyRow3D& row)
{
    const std::array<double, 12> values{{row.geometry_setup_seconds,
        row.pipeline_setup_seconds, row.projector_setup_seconds,
        row.total_seconds, row.edge_mismatch_linf,
        row.edge_mismatch_weighted_rms, row.exact_edge_mismatch_linf,
        row.constant_constraint_defect, row.projected_constraint_defect,
        row.projection_idempotence_defect, row.constant_projection_defect,
        row.h}};
    return finite_neumann_owner_metrics_3d(row.solve)
        && std::all_of(values.begin(), values.end(), [](double value) {
               return std::isfinite(value);
           });
}

struct NeumannEdgePreprocessSnapshot3D {
    std::uint64_t workload_fingerprint = 0;
    std::uint64_t output_digest = 0;
    std::uint64_t wrong_side_queries = 0;
    std::uint64_t geometry_queries = 0;
    app3d::RestrictOwnerPreprocessDiagnostics3D diagnostics;
};

NeumannEdgePreprocessSnapshot3D capture_neumann_edge_preprocess_snapshot_3d(
    const PanelCenterHarmonicJetKFBI3D& pipeline)
{
    NeumannEdgePreprocessSnapshot3D result;
    result.workload_fingerprint =
        pipeline.restrict_owner_workload_fingerprint();
    result.output_digest = pipeline.restrict_owner_output_digest();
    result.diagnostics = pipeline.restrict_owner_preprocess_diagnostics();
    result.wrong_side_queries = result.diagnostics.wrong_side_queries;
    result.geometry_queries = static_cast<std::uint64_t>(
        pipeline.restrict_owner_geometry_query_count());
    return result;
}

bool neumann_edge_preprocess_snapshot_equal_3d(
    const NeumannEdgePreprocessSnapshot3D& lhs,
    const NeumannEdgePreprocessSnapshot3D& rhs)
{
    return lhs.workload_fingerprint == rhs.workload_fingerprint
        && lhs.output_digest == rhs.output_digest
        && lhs.wrong_side_queries == rhs.wrong_side_queries
        && lhs.geometry_queries == rhs.geometry_queries
        && restrict_owner_preprocess_diagnostics_equal(
               lhs.diagnostics, rhs.diagnostics);
}
app3d::NeumannEdgePreprocessInvariantSnapshot3D
neumann_edge_preprocess_invariant_snapshot_3d(
    const NeumannEdgePreprocessSnapshot3D& snapshot,
    const NeumannEdgePreprocessSnapshot3D& reference)
{
    app3d::NeumannEdgePreprocessInvariantSnapshot3D result;
    result.workload_fingerprint = snapshot.workload_fingerprint;
    result.output_digest = snapshot.output_digest;
    result.wrong_side_queries = snapshot.wrong_side_queries;
    result.geometry_queries = snapshot.geometry_queries;
    result.diagnostics_match_reference =
        restrict_owner_preprocess_diagnostics_equal(
            snapshot.diagnostics, reference.diagnostics);
    return result;
}

std::array<NeumannEdgeStudyRow3D, 2>
run_neumann_edge_continuity_pair_3d(
    int N, const app3d::LPrismRigidStudyCase3D& study_case)
{
    constexpr int gmres_max_iterations = 80;
    constexpr ExteriorValueRestrictMode3D mode =
        ExteriorValueRestrictMode3D::JointTricubicCrossingOwner;
    const double h = kBoxSide / static_cast<double>(N);
    const auto geometry_start = std::chrono::steady_clock::now();
    CartesianGrid3D grid({kBoxMin, kBoxMin, kBoxMin}, {h, h, h},
                         {N, N, N}, DofLayout3D::Node);
    GeometryBundle geometry = make_geometry(
        GeometryKind::LPrism, h, study_case.transform);
    const auto domain = std::make_shared<const geometry3d::NurbsCartesianDomain3D>(
        grid, geometry.native_surface.geometry_model());
    const SurfaceDofCloud surface_dofs = app3d::make_native_surface_dofs_3d(
        geometry.native_surface, h);
    validate_surface_dofs(surface_dofs, h);
    app3d::HarmonicCauchyFit3D cauchy_fit =
        app3d::HarmonicCauchyFit3D::build_legacy(
            geometry.native_surface, surface_dofs, h,
            app3d::LegacySurfaceCauchyPolicy3D::G1Nearest,
            kCauchyPolynomialDegree, kCauchyValueNeighborCount,
            kCauchyDerivativeNeighborCount);
    GridPair3D grid_pair(grid, geometry.correction_interface,
                         geometry.crossing_interface, domain);
    int label_mismatches = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        const bool numerical_inside = grid_pair.domain_label(node) > 0;
        if (numerical_inside != geometry.exact_inside(grid_point(grid, node)))
            ++label_mismatches;
    }
    const double geometry_setup_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - geometry_start).count();

    RestrictOwnerPipelinePreprocessTiming3D preprocess_timing;
    const auto pipeline_start = std::chrono::steady_clock::now();
    PanelCenterHarmonicJetKFBI3D pipeline(grid, grid_pair,
        geometry.native_surface, geometry.correction_triangles,
        geometry.geometry_triangles, surface_dofs, std::move(cauchy_fit), false,
        OwnerMode3D::RegionClosestHybrid, nullptr, nullptr, &preprocess_timing);
    const double pipeline_setup_seconds = production_pipeline_setup_seconds_3d(
        std::chrono::duration<double>(std::chrono::steady_clock::now()
            - pipeline_start).count(), preprocess_timing);

    const auto projector_start = std::chrono::steady_clock::now();
    const app3d::NeumannEdgeConstraintSet3D constraints =
        app3d::build_neumann_edge_constraints_3d(
            geometry.native_surface, surface_dofs, h);
    const app3d::NeumannEdgeContinuityProjector3D projector(
        constraints, surface_dofs, 1.0e-12);
    const double projector_setup_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - projector_start).count();
    const int surface_size = pipeline.surface_size();
    Eigen::VectorXd exact_density(surface_size);
    Eigen::VectorXd normal_data(surface_size);
    for (int q = 0; q < surface_size; ++q) {
        const SurfaceDof& dof = pipeline.surface().dofs[static_cast<std::size_t>(q)];
        exact_density[q] = app3d::transformed_manufactured_harmonic_value_3d(
            study_case.transform, dof.point);
        normal_data[q] = app3d::transformed_manufactured_harmonic_gradient_3d(
            study_case.transform, dof.point).dot(dof.normal);
    }
    exact_density.array() -= surface_weighted_mean(pipeline.surface(), exact_density);
    normal_data.array() -= surface_weighted_mean(pipeline.surface(), normal_data);

    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(surface_size);
    Eigen::VectorXd projector_probe(surface_size);
    for (int q = 0; q < surface_size; ++q) {
        projector_probe[q] = std::sin(0.37 * (q + 1))
                           + 0.2 * std::cos(0.11 * (q + 1));
    }
    const Eigen::VectorXd projected_probe = projector.project(projector_probe);
    const double probe_scale = std::max(1.0, vector_linf(projector_probe));
    const double projected_constraint_defect = vector_linf(
        app3d::apply_neumann_edge_constraints_3d(constraints, projected_probe))
        / probe_scale;
    const double projection_idempotence_defect = vector_linf(
        projector.project(projected_probe) - projected_probe) / probe_scale;
    const double constant_projection_defect = vector_linf(projector.project(ones) - ones);
    const double constant_constraint_defect = vector_linf(
        app3d::apply_neumann_edge_constraints_3d(constraints, ones));

    const app3d::NeumannEdgeConstraintAudit3D constraint_audit =
        app3d::audit_neumann_edge_constraints_3d(
            geometry.native_surface, surface_dofs, h, constraints);

    const auto setup_diagnostics = pipeline.restrict_owner_preprocess_diagnostics();
    const std::uint64_t setup_queries = static_cast<std::uint64_t>(
        pipeline.restrict_owner_geometry_query_count());
    if (setup_queries != setup_diagnostics.wrong_side_queries)
        throw std::logic_error("Neumann edge-continuity pipeline query-count mismatch");
    const NeumannRigidGeometryDiagnostics3D geometry_diagnostics =
        neumann_rigid_geometry_diagnostics_3d(
            grid_pair, pipeline.correction_support());

    const Eigen::VectorXd zero_value = Eigen::VectorXd::Zero(surface_size);
    const HarmonicJetField3D probe_field = pipeline.field_from_grid_and_jumps(
        Eigen::VectorXd::Zero(grid.num_dofs()), zero_value, normal_data);
    const auto probe_before = pipeline.restrict_owner_preprocess_diagnostics();
    const Eigen::VectorXd default_probe = pipeline.exterior_trace(
        probe_field, zero_value, normal_data);
    const Eigen::VectorXd explicit_legacy_probe = pipeline.exterior_trace(
        probe_field, zero_value, normal_data,
        ExteriorValueRestrictMode3D::JointTricubicCauchy);
    const Eigen::VectorXd crossing_probe = pipeline.exterior_trace(
        probe_field, zero_value, normal_data, mode);
    const auto probe_after = pipeline.restrict_owner_preprocess_diagnostics();
    const bool default_route_bitwise_equal =
        bitwise_equal_vector_3d(default_probe, explicit_legacy_probe);
    const bool probe_invariants_pass = default_route_bitwise_equal
        && crossing_probe.allFinite()
        && restrict_owner_preprocess_diagnostics_equal(probe_before, probe_after)
        && static_cast<std::uint64_t>(pipeline.restrict_owner_geometry_query_count())
            == setup_queries;

    std::array<NeumannEdgeStudyRow3D, 2> rows;
    rows[0].density_space = app3d::NeumannDensitySpace3D::PatchIndependent;
    rows[1].density_space = app3d::NeumannDensitySpace3D::NonG1EdgeProjected;
    std::array<Eigen::VectorXd, 2> solved_densities;
    const NeumannEdgePreprocessSnapshot3D stable_snapshot =
        capture_neumann_edge_preprocess_snapshot_3d(pipeline);
    if (stable_snapshot.output_digest != preprocess_timing.output_digest
        || stable_snapshot.wrong_side_queries
            != setup_diagnostics.wrong_side_queries
        || stable_snapshot.geometry_queries != setup_queries) {
        throw std::logic_error(
            "Neumann edge-continuity setup snapshot mismatch");
    }
    std::array<NeumannEdgePreprocessSnapshot3D, 2> before_snapshots;
    std::array<NeumannEdgePreprocessSnapshot3D, 2> after_snapshots;
    for (std::size_t index = 0; index < rows.size(); ++index) {
        NeumannEdgeStudyRow3D& row = rows[index];
        row.study_case = study_case;
        row.N = N;
        row.h = h;
        row.dofs = surface_size;
        row.geometry_setup_seconds = geometry_setup_seconds;
        row.pipeline_setup_seconds = pipeline_setup_seconds;
        row.projector_setup_seconds = projector_setup_seconds;
        row.expected_non_g1_connections =
            constraint_audit.expected_non_g1_connections;
        row.covered_non_g1_connections =
            constraint_audit.covered_non_g1_connections;
        row.duplicate_connection_intervals =
            constraint_audit.duplicate_connection_intervals;
        row.g1_constraint_rows = constraint_audit.g1_constraint_rows;
        row.unrelated_constraint_rows =
            constraint_audit.unrelated_constraint_rows;
        row.constraint_rows = constraint_audit.constraint_rows;
        row.constraint_rank = projector.retained_rank();
        row.reduced_order_rows = constraint_audit.reduced_order_rows;
        row.constant_constraint_defect = constant_constraint_defect;
        row.projected_constraint_defect = projected_constraint_defect;
        row.projection_idempotence_defect = projection_idempotence_defect;
        row.constant_projection_defect = constant_projection_defect;
        row.label_mismatches = label_mismatches;
        row.unsafe_label_changing_edges = geometry_diagnostics.unsafe_label_changing_edges;
        row.gap_crossings = geometry_diagnostics.gap_crossings;
        row.endpoint_crossings = geometry_diagnostics.endpoint_crossings;
        row.triangle_fallback_crossings = geometry_diagnostics.triangle_fallback_crossings;
        row.preprocess_mode = app3d::restrict_owner_preprocess_mode_name_3d(
            OwnerMode3D::RegionClosestHybrid);
        row.default_route_bitwise_equal = default_route_bitwise_equal;
        row.probe_invariants_pass = probe_invariants_pass;
        before_snapshots[index] =
            capture_neumann_edge_preprocess_snapshot_3d(pipeline);
        const NeumannEdgePreprocessSnapshot3D& before =
            before_snapshots[index];
        row.workload_fingerprint_before_gmres =
            before.workload_fingerprint;
        row.output_digest_before_gmres = before.output_digest;
        row.wrong_side_queries_before_gmres =
            before.wrong_side_queries;
        row.geometry_queries_before_gmres = before.geometry_queries;
        const app3d::NeumannEdgeContinuityProjector3D* solve_projector =
            row.density_space == app3d::NeumannDensitySpace3D::NonG1EdgeProjected
                ? std::addressof(projector) : nullptr;
        row.solve = run_neumann_case(grid, grid_pair, pipeline,
            study_case.transform, gmres_max_iterations, mode,
            &row.residual_history, solve_projector, &solved_densities[index]);
        after_snapshots[index] =
            capture_neumann_edge_preprocess_snapshot_3d(pipeline);
        const NeumannEdgePreprocessSnapshot3D& after =
            after_snapshots[index];
        row.workload_fingerprint_after_gmres =
            after.workload_fingerprint;
        row.output_digest_after_gmres = after.output_digest;
        row.wrong_side_queries_after_gmres =
            after.wrong_side_queries;
        row.geometry_queries_after_gmres = after.geometry_queries;
        row.workload_fingerprint = after.workload_fingerprint;
        row.output_digest = after.output_digest;
        row.wrong_side_queries = after.wrong_side_queries;
        row.diagnostics_unchanged =
            restrict_owner_preprocess_diagnostics_equal(
                before.diagnostics, after.diagnostics)
            && restrict_owner_preprocess_diagnostics_equal(
                stable_snapshot.diagnostics, after.diagnostics);
        row.geometry_diagnostics_pass = row.label_mismatches == 0
            && row.unsafe_label_changing_edges == 0 && row.gap_crossings == 0
            && row.endpoint_crossings == 0 && row.triangle_fallback_crossings == 0;
        row.owner_invariants_pass = row.probe_invariants_pass
            && row.default_route_bitwise_equal
            && row.diagnostics_unchanged
            && neumann_edge_preprocess_snapshot_equal_3d(before, after)
            && neumann_edge_preprocess_snapshot_equal_3d(
                   stable_snapshot, after);
        row.edge_mismatch_linf = app3d::neumann_edge_mismatch_linf_3d(
            constraints, solved_densities[index]);
        row.edge_mismatch_weighted_rms = app3d::neumann_edge_mismatch_weighted_rms_3d(
            constraints, solved_densities[index]);
        row.exact_edge_mismatch_linf = app3d::neumann_edge_mismatch_linf_3d(
            constraints, exact_density);
        row.total_seconds = row.geometry_setup_seconds + row.pipeline_setup_seconds
            + row.projector_setup_seconds + row.solve.seconds;
    }

    const NeumannEdgePreprocessSnapshot3D final_snapshot =
        capture_neumann_edge_preprocess_snapshot_3d(pipeline);
    const std::vector<app3d::NeumannEdgePreprocessInvariantSnapshot3D>
        invariant_snapshots{
            neumann_edge_preprocess_invariant_snapshot_3d(
                stable_snapshot, stable_snapshot),
            neumann_edge_preprocess_invariant_snapshot_3d(
                before_snapshots[0], stable_snapshot),
            neumann_edge_preprocess_invariant_snapshot_3d(
                after_snapshots[0], stable_snapshot),
            neumann_edge_preprocess_invariant_snapshot_3d(
                before_snapshots[1], stable_snapshot),
            neumann_edge_preprocess_invariant_snapshot_3d(
                after_snapshots[1], stable_snapshot),
            neumann_edge_preprocess_invariant_snapshot_3d(
                final_snapshot, stable_snapshot)};
    const bool shared_preprocess_pass =
        app3d::neumann_edge_shared_preprocess_pass_3d(
            invariant_snapshots);
    rows[0].shared_preprocess_pass = shared_preprocess_pass;
    rows[1].shared_preprocess_pass = shared_preprocess_pass;

    const Eigen::VectorXd exact_mismatch = app3d::apply_neumann_edge_constraints_3d(
        constraints, exact_density);
    const Eigen::VectorXd unconstrained_mismatch = app3d::apply_neumann_edge_constraints_3d(
        constraints, solved_densities[0]);
    const Eigen::VectorXd projected_mismatch = app3d::apply_neumann_edge_constraints_3d(
        constraints, solved_densities[1]);
    for (std::size_t q = 0; q < constraints.samples.size(); ++q) {
        const NeumannEdgeSampleDiagnostics3D item{constraints.samples[q],
            exact_mismatch[static_cast<Eigen::Index>(q)],
            unconstrained_mismatch[static_cast<Eigen::Index>(q)],
            projected_mismatch[static_cast<Eigen::Index>(q)]};
        rows[0].edge_samples.push_back(item);
        rows[1].edge_samples.push_back(item);
    }
    return rows;
}

std::vector<app3d::NeumannEdgeContinuityMeasurement3D>
neumann_edge_measurements_3d(const std::vector<NeumannEdgeStudyRow3D>& rows)
{
    std::vector<app3d::NeumannEdgeContinuityMeasurement3D> result;
    result.reserve(rows.size());
    for (const NeumannEdgeStudyRow3D& row : rows) {
        app3d::NeumannEdgeContinuityMeasurement3D m;
        m.case_id = row.study_case.id;
        m.N = row.N;
        m.h = row.h;
        m.density_space = row.density_space;
        m.finite_metrics = finite_neumann_edge_row_3d(row);
        m.gmres_converged = row.solve.converged;
        m.gmres_iterations = row.solve.iterations;
        m.gmres_relative_residual = row.solve.gmres_relative_residual;
        m.density_linf = row.solve.density_linf;
        m.density_l2 = row.solve.density_l2;
        m.interior_linf = row.solve.interior_linf;
        m.interior_l2 = row.solve.interior_l2;
        m.edge_mismatch_linf = row.edge_mismatch_linf;
        m.edge_mismatch_weighted_rms = row.edge_mismatch_weighted_rms;
        m.exact_edge_mismatch_linf = row.exact_edge_mismatch_linf;
        m.expected_non_g1_connections = row.expected_non_g1_connections;
        m.covered_non_g1_connections = row.covered_non_g1_connections;
        m.duplicate_connection_intervals = row.duplicate_connection_intervals;
        m.g1_constraint_rows = row.g1_constraint_rows;
        m.unrelated_constraint_rows = row.unrelated_constraint_rows;
        m.constraint_rows = row.constraint_rows;
        m.constraint_rank = row.constraint_rank;
        m.reduced_order_rows = row.reduced_order_rows;
        m.constant_constraint_defect = row.constant_constraint_defect;
        m.projected_constraint_defect = row.projected_constraint_defect;
        m.projection_idempotence_defect = row.projection_idempotence_defect;
        m.constant_projection_defect = row.constant_projection_defect;
        m.geometry_diagnostics_pass = row.geometry_diagnostics_pass;
        m.owner_invariants_pass = row.owner_invariants_pass;
        m.shared_preprocess_pass = row.shared_preprocess_pass;
        result.push_back(std::move(m));
    }
    return result;
}

const app3d::NeumannEdgeContinuityDerivedRow3D* find_neumann_edge_derived_row_3d(
    const app3d::NeumannEdgeContinuityEvaluation3D& evaluation,
    const NeumannEdgeStudyRow3D& raw)
{
    const auto found = std::find_if(evaluation.rows.begin(), evaluation.rows.end(),
        [&](const app3d::NeumannEdgeContinuityDerivedRow3D& row) {
            return row.measurement.case_id == raw.study_case.id
                && row.measurement.N == raw.N
                && row.measurement.density_space == raw.density_space;
        });
    return found == evaluation.rows.end() ? nullptr : std::addressof(*found);
}

const char* patch_edge_name_3d(app3d::PatchEdge3D edge)
{
    switch (edge) {
    case app3d::PatchEdge3D::UMin: return "u_min";
    case app3d::PatchEdge3D::UMax: return "u_max";
    case app3d::PatchEdge3D::VMin: return "v_min";
    case app3d::PatchEdge3D::VMax: return "v_max";
    }
    return "unknown";
}

void write_neumann_edge_continuity_checkpoints_3d(
    const std::filesystem::path& output_dir,
    const std::vector<NeumannEdgeStudyRow3D>& rows,
    const app3d::NeumannEdgeContinuityEvaluation3D& evaluation)
{
    std::filesystem::create_directories(output_dir);
    std::ofstream summary = open_output_file(output_dir / "summary.csv");
    summary << std::setprecision(17) << std::boolalpha
        << "case_id,N,h,density_space,dofs,geometry_setup_seconds,"
           "pipeline_setup_seconds,projector_setup_seconds,solve_seconds,"
           "total_seconds,converged,iterations,final_residual,"
           "operator_residual_linf,density_linf,density_l2,interior_linf,"
           "interior_l2,density_linf_order,density_l2_order,interior_linf_order,"
           "interior_l2_order,edge_mismatch_linf,edge_mismatch_weighted_rms,"
           "exact_edge_mismatch_linf,exact_edge_mismatch_order,edge_reduction_ratio,"
           "constraint_rows,constraint_rank,expected_non_g1_connections,"
           "covered_non_g1_connections,duplicate_connection_intervals,"
           "g1_constraint_rows,unrelated_constraint_rows,reduced_order_rows,"
           "constant_constraint_defect,projected_constraint_defect,"
           "projection_idempotence_defect,constant_projection_defect,"
           "geometry_diagnostics_pass,owner_invariants_pass,shared_preprocess_pass,row_pass\n";
    for (const NeumannEdgeStudyRow3D& row : rows) {
        const auto* d = find_neumann_edge_derived_row_3d(evaluation, row);
        if (d == nullptr) throw std::logic_error("missing Neumann edge derived row");
        summary << row.study_case.id << ',' << row.N << ',' << row.h << ','
            << app3d::neumann_density_space_name_3d(row.density_space) << ','
            << row.dofs << ',' << row.geometry_setup_seconds << ','
            << row.pipeline_setup_seconds << ',' << row.projector_setup_seconds << ','
            << row.solve.seconds << ',' << row.total_seconds << ','
            << row.solve.converged << ',' << row.solve.iterations << ','
            << row.solve.gmres_relative_residual << ','
            << row.solve.operator_residual_linf << ',' << row.solve.density_linf << ','
            << row.solve.density_l2 << ',' << row.solve.interior_linf << ','
            << row.solve.interior_l2 << ',' << d->density_linf_order << ','
            << d->density_l2_order << ',' << d->interior_linf_order << ','
            << d->interior_l2_order << ',' << row.edge_mismatch_linf << ','
            << row.edge_mismatch_weighted_rms << ',' << row.exact_edge_mismatch_linf << ','
            << d->exact_edge_mismatch_order << ',' << d->edge_reduction_ratio << ','
            << row.constraint_rows << ',' << row.constraint_rank << ','
            << row.expected_non_g1_connections << ',' << row.covered_non_g1_connections << ','
            << row.duplicate_connection_intervals << ',' << row.g1_constraint_rows << ','
            << row.unrelated_constraint_rows << ',' << row.reduced_order_rows << ','
            << row.constant_constraint_defect << ',' << row.projected_constraint_defect << ','
            << row.projection_idempotence_defect << ',' << row.constant_projection_defect << ','
            << row.geometry_diagnostics_pass << ',' << row.owner_invariants_pass << ','
            << row.shared_preprocess_pass << ',' << criterion_status_name(d->row_pass) << '\n';
    }
    std::ofstream residuals = open_output_file(output_dir / "gmres_residuals.csv");
    residuals << std::setprecision(17)
              << "case_id,N,density_space,iteration,residual\n";
    for (const NeumannEdgeStudyRow3D& row : rows) {
        for (std::size_t iteration = 0; iteration < row.residual_history.size(); ++iteration) {
            residuals << row.study_case.id << ',' << row.N << ','
                << app3d::neumann_density_space_name_3d(row.density_space) << ','
                << iteration << ',' << row.residual_history[iteration] << '\n';
        }
    }

    std::ofstream edges = open_output_file(output_dir / "edge_diagnostics.csv");
    edges << std::setprecision(17) << std::boolalpha
        << "case_id,N,connection_index,sample_index,sample_count,first_patch,first_edge,"
           "second_patch,second_edge,normalized_parameter,first_parameter,second_parameter,"
           "physical_weight,mapped_point_gap,first_reduced_order,second_reduced_order,"
           "exact_signed_mismatch,unconstrained_signed_mismatch,projected_signed_mismatch\n";
    for (const NeumannEdgeStudyRow3D& row : rows) {
        if (row.density_space != app3d::NeumannDensitySpace3D::PatchIndependent) continue;
        for (const auto& item : row.edge_samples) {
            const auto& x = item.sample;
            edges << row.study_case.id << ',' << row.N << ',' << x.connection_index << ','
                << x.sample_index << ',' << x.sample_count << ',' << x.first_patch << ','
                << patch_edge_name_3d(x.first_edge) << ',' << x.second_patch << ','
                << patch_edge_name_3d(x.second_edge) << ',' << x.normalized_parameter << ','
                << x.first_parameter << ',' << x.second_parameter << ','
                << x.quadrature_weight << ',' << x.mapped_point_gap << ','
                << x.first_reduced_order << ',' << x.second_reduced_order << ','
                << item.exact_signed_mismatch << ',' << item.unconstrained_signed_mismatch << ','
                << item.projected_signed_mismatch << '\n';
        }
    }

    std::ofstream owners = open_output_file(output_dir / "owner_diagnostics.csv");
    owners << std::setprecision(17) << std::boolalpha
        << "case_id,N,density_space,preprocess_mode,workload_fingerprint,output_digest,"
           "wrong_side_queries,workload_fingerprint_before_gmres,"
           "workload_fingerprint_after_gmres,output_digest_before_gmres,"
           "output_digest_after_gmres,wrong_side_queries_before_gmres,"
           "wrong_side_queries_after_gmres,label_mismatches,"
           "unsafe_label_changing_edges,gap_crossings,"
           "endpoint_crossings,triangle_fallback_crossings,geometry_queries_before_gmres,"
           "geometry_queries_after_gmres,diagnostics_unchanged,default_route_bitwise_equal,"
           "probe_invariants_pass,geometry_diagnostics_pass,owner_invariants_pass,"
           "shared_preprocess_pass\n";
    for (const NeumannEdgeStudyRow3D& row : rows) {
        owners << row.study_case.id << ',' << row.N << ','
            << app3d::neumann_density_space_name_3d(row.density_space) << ','
            << row.preprocess_mode << ',' << row.workload_fingerprint << ','
            << row.output_digest << ',' << row.wrong_side_queries << ','
            << row.workload_fingerprint_before_gmres << ','
            << row.workload_fingerprint_after_gmres << ','
            << row.output_digest_before_gmres << ','
            << row.output_digest_after_gmres << ','
            << row.wrong_side_queries_before_gmres << ','
            << row.wrong_side_queries_after_gmres << ','
            << row.label_mismatches << ',' << row.unsafe_label_changing_edges << ','
            << row.gap_crossings << ',' << row.endpoint_crossings << ','
            << row.triangle_fallback_crossings << ',' << row.geometry_queries_before_gmres << ','
            << row.geometry_queries_after_gmres << ',' << row.diagnostics_unchanged << ','
            << row.default_route_bitwise_equal << ',' << row.probe_invariants_pass << ','
            << row.geometry_diagnostics_pass << ',' << row.owner_invariants_pass << ','
            << row.shared_preprocess_pass << '\n';
    }

    std::ofstream acceptance = open_output_file(output_dir / "acceptance.csv");
    acceptance << "completeness_pass,topology_pass,projector_pass,exact_trace_order_pass,"
                  "gmres_pass,error_guard_pass,edge_reduction_pass,trend_pass,"
                  "geometry_owner_pass,extended_evidence_pass,overall_pass\n"
        << criterion_status_name(evaluation.acceptance.completeness_pass) << ','
        << criterion_status_name(evaluation.acceptance.topology_pass) << ','
        << criterion_status_name(evaluation.acceptance.projector_pass) << ','
        << criterion_status_name(evaluation.acceptance.exact_trace_order_pass) << ','
        << criterion_status_name(evaluation.acceptance.gmres_pass) << ','
        << criterion_status_name(evaluation.acceptance.error_guard_pass) << ','
        << criterion_status_name(evaluation.acceptance.edge_reduction_pass) << ','
        << criterion_status_name(evaluation.acceptance.trend_pass) << ','
        << criterion_status_name(evaluation.acceptance.geometry_owner_pass) << ','
        << criterion_status_name(evaluation.acceptance.extended_evidence_pass) << ','
        << criterion_status_name(evaluation.acceptance.overall_pass) << '\n';
}

using PhaseRecordArray3D = std::array<
    app3d::PhaseProfileRecord3D,
    app3d::phase_profile_kind_count_3d()>;

struct NeumannEdgeCauchyEdgeValueRow3D {
    int connection_index = -1;
    int sample_index = -1;
    int sample_count = 0;
    int first_owner_dof = -1;
    int second_owner_dof = -1;
    double first_value = 0.0;
    double second_value = 0.0;
    double shared_auxiliary_value = 0.0;
    double first_second_difference = 0.0;
    double first_shared_difference = 0.0;
    double second_shared_difference = 0.0;
};

struct NeumannEdgeCauchyPairRun3D {
    std::array<app3d::NeumannEdgeCauchyMeasurement3D, 2> measurements;
    std::array<std::vector<double>, 2> residual_histories;
    std::array<std::vector<NeumannEdgeCauchyEdgeValueRow3D>, 2>
        edge_value_rows;
    PhaseRecordArray3D shared_setup_phases{};
    std::array<PhaseRecordArray3D, 2> runtime_phase_deltas{};
    std::vector<app3d::NeumannEdgePreprocessInvariantSnapshot3D>
        owner_snapshots;
    int setup_factorization_count = 0;
    std::array<int, 2> factorization_counts_before{};
    std::array<int, 2> factorization_counts_after{};
    std::array<bool, 2> shared_exact_trace_bitwise{};
    std::array<bool, 2> shared_normal_jump_bitwise{};
};

PhaseRecordArray3D capture_phase_records_3d(
    const app3d::PhaseProfile3D& profile)
{
    PhaseRecordArray3D result{};
    for (std::size_t q = 0; q < result.size(); ++q) {
        result[q] = profile.record(static_cast<PhaseProfileKind3D>(q));
    }
    return result;
}

PhaseRecordArray3D subtract_phase_records_3d(
    const PhaseRecordArray3D& after,
    const PhaseRecordArray3D& before,
    const char* context)
{
    PhaseRecordArray3D result{};
    for (std::size_t q = 0; q < result.size(); ++q) {
        if (after[q].calls < before[q].calls
            || after[q].seconds < before[q].seconds) {
            throw std::logic_error(
                std::string(context) + " phase counters decreased");
        }
        result[q].seconds = after[q].seconds - before[q].seconds;
        result[q].calls = after[q].calls - before[q].calls;
        if (!std::isfinite(result[q].seconds)
            || result[q].seconds < 0.0) {
            throw std::logic_error(
                std::string(context) + " has invalid phase delta");
        }
    }
    return result;
}

double phase_record_sum_3d(const PhaseRecordArray3D& records)
{
    double result = 0.0;
    for (const auto& record : records) result += record.seconds;
    return result;
}

void require_phase_wall_match_3d(double phase_seconds,
                                 double wall_seconds,
                                 const char* context)
{
    const double tolerance = std::max(1.0e-9, 1.0e-8 * wall_seconds);
    if (!std::isfinite(phase_seconds) || !std::isfinite(wall_seconds)
        || wall_seconds < 0.0
        || std::abs(phase_seconds - wall_seconds) > tolerance) {
        throw std::logic_error(
            std::string(context) + " phase sum does not match wall time");
    }
}

bool bitwise_equal_double_3d(double lhs, double rhs) noexcept
{
    return std::memcmp(&lhs, &rhs, sizeof(double)) == 0;
}

bool far_cauchy_rows_bitwise_equal_3d(
    const Eigen::MatrixXd& legacy,
    const Eigen::MatrixXd& augmented,
    const std::vector<bool>& affected)
{
    if (legacy.rows() != augmented.rows()
        || legacy.cols() != augmented.cols()
        || legacy.rows() != static_cast<Eigen::Index>(affected.size())) {
        return false;
    }
    for (Eigen::Index row = 0; row < legacy.rows(); ++row) {
        if (affected[static_cast<std::size_t>(row)]) continue;
        for (Eigen::Index column = 0; column < legacy.cols(); ++column) {
            if (!bitwise_equal_double_3d(
                    legacy(row, column), augmented(row, column))) {
                return false;
            }
        }
    }
    return true;
}

double evaluate_edge_owner_value_3d(
    const SurfaceDofCloud& surface,
    int owner_dof,
    const Eigen::Vector3d& point,
    double h,
    const Eigen::MatrixXd& coefficients)
{
    if (owner_dof < 0
        || owner_dof >= static_cast<int>(surface.dofs.size())
        || owner_dof >= coefficients.rows()) {
        throw std::logic_error("invalid edge owner DOF");
    }
    const SurfaceDof& owner =
        surface.dofs[static_cast<std::size_t>(owner_dof)];
    const Eigen::Vector3d d = (point - owner.point) / h;
    const Eigen::Vector3d xi(
        d.dot(owner.tangent1), d.dot(owner.tangent2),
        d.dot(owner.normal));
    const app3d::HarmonicPolynomialSpace3D polynomial_space(3);
    return polynomial_space.basis(xi.x(), xi.y(), xi.z()).dot(
        coefficients.row(owner_dof).transpose());
}

struct PublicEdgeCauchyDiagnostics3D {
    int expected_non_g1_connections = 0;
    int covered_non_g1_connections = 0;
    int edge_sample_count = 0;
    int affected_center_count = 0;
    int corner_center_count = 0;
    int unrelated_sample_or_attachment_count = 0;
    int rank_deficient_fit_count = 0;
    int legacy_factorization_count = 0;
    double harmonic_cubic_reproduction_defect = 0.0;
    double edge_condition_max = 0.0;
    double local_condition_max = 0.0;
};

PublicEdgeCauchyDiagnostics3D public_edge_cauchy_diagnostics_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud& cloud,
    double h,
    const app3d::HarmonicCauchyFit3D& fit)
{
    PublicEdgeCauchyDiagnostics3D result;
    std::set<int> covered_connections;
    for (const auto& connection : surface.geometric_connections) {
        if (!connection.g1)
            ++result.expected_non_g1_connections;
    }
    for (const auto& map : fit.edge_maps()) {
        covered_connections.insert(map.point.connection_id);
        result.edge_condition_max = std::max(
            result.edge_condition_max, map.condition);
    }
    result.covered_non_g1_connections =
        static_cast<int>(covered_connections.size());
    result.edge_sample_count = static_cast<int>(fit.edge_maps().size());
    for (const auto& map : fit.surface_maps()) {
        if (map.edge_point_ids.empty())
            continue;
        ++result.affected_center_count;
        if (map.relevant_connection_ids.size() >= 2)
            ++result.corner_center_count;
        result.local_condition_max = std::max(
            result.local_condition_max, map.condition);
        for (int point_id : map.edge_point_ids) {
            if (point_id < 0
                || point_id >= static_cast<int>(fit.edge_maps().size())) {
                throw std::logic_error(
                    "public edge-Cauchy map has an invalid edge point ID");
            }
            const int connection_id = fit.edge_maps()[
                static_cast<std::size_t>(point_id)].point.connection_id;
            if (std::find(map.relevant_connection_ids.begin(),
                          map.relevant_connection_ids.end(), connection_id)
                == map.relevant_connection_ids.end()) {
                ++result.unrelated_sample_or_attachment_count;
            }
        }
    }
    result.legacy_factorization_count = 2 * (
        result.edge_sample_count + result.affected_center_count);

    const auto& space = fit.space();
    for (const auto& map : fit.edge_maps()) {
        for (int column = 0; column < space.dimension(); ++column) {
            Eigen::VectorXd values(static_cast<int>(map.value_ids.size()));
            Eigen::VectorXd normals(static_cast<int>(map.normal_ids.size()));
            for (int q = 0; q < values.size(); ++q) {
                const auto& sample = cloud.dofs[static_cast<std::size_t>(
                    map.value_ids[static_cast<std::size_t>(q)])];
                const Eigen::Vector3d xi = map.point.frame.transpose()
                    * (sample.point - map.point.point) / h;
                values[q] = space.basis(xi.x(), xi.y(), xi.z())[column];
            }
            for (int q = 0; q < normals.size(); ++q) {
                const auto& sample = cloud.dofs[static_cast<std::size_t>(
                    map.normal_ids[static_cast<std::size_t>(q)])];
                const Eigen::Vector3d xi = map.point.frame.transpose()
                    * (sample.point - map.point.point) / h;
                const Eigen::Vector3d normal =
                    map.point.frame.transpose() * sample.normal;
                normals[q] = normal.dot(
                    space.gradient(xi.x(), xi.y(), xi.z()).col(column)) / h;
            }
            const double predicted =
                map.E_value.dot(values) + map.E_normal.dot(normals);
            const double expected = space.basis(0.0, 0.0, 0.0)[column];
            result.harmonic_cubic_reproduction_defect = std::max(
                result.harmonic_cubic_reproduction_defect,
                std::abs(predicted - expected));
        }
    }

    for (int center = 0;
         center < static_cast<int>(fit.surface_maps().size()); ++center) {
        const auto& map = fit.surface_maps()[static_cast<std::size_t>(center)];
        if (map.edge_point_ids.empty())
            continue;
        const SurfaceDof& target = cloud.dofs[static_cast<std::size_t>(center)];
        Eigen::Matrix3d frame;
        frame.col(0) = target.tangent1;
        frame.col(1) = target.tangent2;
        frame.col(2) = target.normal;
        for (int column = 0; column < space.dimension(); ++column) {
            Eigen::VectorXd values(static_cast<int>(map.value_ids.size()));
            Eigen::VectorXd normals(static_cast<int>(map.normal_ids.size()));
            Eigen::VectorXd edges(static_cast<int>(map.edge_point_ids.size()));
            for (int q = 0; q < values.size(); ++q) {
                const auto& sample = cloud.dofs[static_cast<std::size_t>(
                    map.value_ids[static_cast<std::size_t>(q)])];
                const Eigen::Vector3d xi = frame.transpose()
                    * (sample.point - target.point) / h;
                values[q] = space.basis(xi.x(), xi.y(), xi.z())[column];
            }
            for (int q = 0; q < normals.size(); ++q) {
                const auto& sample = cloud.dofs[static_cast<std::size_t>(
                    map.normal_ids[static_cast<std::size_t>(q)])];
                const Eigen::Vector3d xi = frame.transpose()
                    * (sample.point - target.point) / h;
                const Eigen::Vector3d normal = frame.transpose() * sample.normal;
                normals[q] = normal.dot(
                    space.gradient(xi.x(), xi.y(), xi.z()).col(column)) / h;
            }
            for (int q = 0; q < edges.size(); ++q) {
                const auto& point = fit.edge_maps()[static_cast<std::size_t>(
                    map.edge_point_ids[static_cast<std::size_t>(q)])].point;
                const Eigen::Vector3d xi = frame.transpose()
                    * (point.point - target.point) / h;
                edges[q] = space.basis(xi.x(), xi.y(), xi.z())[column];
            }
            const Eigen::VectorXd predicted = map.M_value * values
                + map.M_normal * normals + map.M_edge * edges;
            for (int row = 0; row < predicted.size(); ++row) {
                const double expected = row == column ? 1.0 : 0.0;
                result.harmonic_cubic_reproduction_defect = std::max(
                    result.harmonic_cubic_reproduction_defect,
                    std::abs(predicted[row] - expected));
            }
        }
    }

    return result;
}

NeumannEdgeCauchyPairRun3D run_neumann_edge_cauchy_pair_3d(
    int N, const app3d::LPrismRigidStudyCase3D& study_case)
{
    constexpr int gmres_max_iterations = 80;
    constexpr ExteriorValueRestrictMode3D restrict_mode =
        ExteriorValueRestrictMode3D::JointTricubicCrossingOwner;
    const std::array<app3d::NeumannEdgeCauchyMode3D, 2> modes{{
        app3d::NeumannEdgeCauchyMode3D::None,
        app3d::NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues}};
    const double h = kBoxSide / static_cast<double>(N);
    app3d::PhaseProfile3D profile;
    const auto setup_start = std::chrono::steady_clock::now();

    auto phase_start = std::chrono::steady_clock::now();
    CartesianGrid3D grid({kBoxMin, kBoxMin, kBoxMin}, {h, h, h},
                         {N, N, N}, DofLayout3D::Node);
    GeometryBundle geometry = make_geometry(
        GeometryKind::LPrism, h, study_case.transform);
    const auto domain = std::make_shared<const
        geometry3d::NurbsCartesianDomain3D>(
            grid, geometry.native_surface.geometry_model());
    profile.add(PhaseProfileKind3D::GeometryAndDomain,
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - phase_start).count(), 1);

    phase_start = std::chrono::steady_clock::now();
    const SurfaceDofCloud surface_dofs =
        app3d::make_native_surface_dofs_3d(
            geometry.native_surface, h);
    validate_surface_dofs(surface_dofs, h);
    const app3d::SurfaceNonG1EdgeNeighborhoodSet3D neighborhoods =
        app3d::build_surface_non_g1_edge_neighborhoods_3d(
            geometry.native_surface, surface_dofs, h);
    std::array<app3d::HarmonicCauchyFit3D, 2> cauchy_fits{{
        app3d::HarmonicCauchyFit3D::build_legacy(
            geometry.native_surface, surface_dofs, h,
            app3d::LegacySurfaceCauchyPolicy3D::G1Nearest,
            kCauchyPolynomialDegree, kCauchyValueNeighborCount,
            kCauchyDerivativeNeighborCount),
        app3d::HarmonicCauchyFit3D::build(
            geometry.native_surface, surface_dofs, neighborhoods, h,
            app3d::HarmonicCauchyRoute3D::EdgeReconstructedValue)}};
    profile.add(PhaseProfileKind3D::SurfaceDofsAndStencils,
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - phase_start).count(), 1);

    phase_start = std::chrono::steady_clock::now();
    GridPair3D grid_pair(grid, geometry.correction_interface,
                         geometry.crossing_interface, domain);
    int label_mismatches = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        const bool numerical_inside = grid_pair.domain_label(node) > 0;
        if (numerical_inside
            != geometry.exact_inside(grid_point(grid, node))) {
            ++label_mismatches;
        }
    }
    profile.add(PhaseProfileKind3D::GridPairAndLabelValidation,
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - phase_start).count(), 1);

    const std::array<PhaseProfileKind3D, 4> pipeline_children{{
        PhaseProfileKind3D::CrossingRows,
        PhaseProfileKind3D::NurbsSegmentIntersections,
        PhaseProfileKind3D::RestrictOwnerGeometryPreprocessing,
        PhaseProfileKind3D::TraceOwnerTemplateAssembly}};
    std::array<double, 4> pipeline_child_before{};
    for (std::size_t q = 0; q < pipeline_children.size(); ++q) {
        pipeline_child_before[q] =
            profile.record(pipeline_children[q]).seconds;
    }
    std::array<RestrictOwnerPipelinePreprocessTiming3D, 2>
        preprocess_timings;
    const auto pipeline_start = std::chrono::steady_clock::now();
    std::array<std::unique_ptr<PanelCenterHarmonicJetKFBI3D>, 2>
        pipelines;
    for (std::size_t index = 0; index < pipelines.size(); ++index) {
        pipelines[index] = std::make_unique<PanelCenterHarmonicJetKFBI3D>(
            grid, grid_pair, geometry.native_surface,
            geometry.correction_triangles, geometry.geometry_triangles,
            surface_dofs, std::move(cauchy_fits[index]), false,
            OwnerMode3D::RegionClosestHybrid, &profile, nullptr,
            &preprocess_timings[index]);
    }
    const double pipeline_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - pipeline_start).count();
    double pipeline_child_seconds = 0.0;
    for (std::size_t q = 0; q < pipeline_children.size(); ++q) {
        pipeline_child_seconds += profile.record(pipeline_children[q]).seconds
            - pipeline_child_before[q];
    }
    profile.add(PhaseProfileKind3D::PipelineFixedInitialization,
        nonnegative_profile_remainder_3d(
            pipeline_wall_seconds, pipeline_child_seconds,
            "Neumann edge-Cauchy pipeline setup"), 1);

    const int surface_size = pipelines[0]->surface_size();
    Eigen::VectorXd shared_exact_trace(surface_size);
    Eigen::VectorXd prescribed_normal_jump(surface_size);
    for (int q = 0; q < surface_size; ++q) {
        const SurfaceDof& dof =
            pipelines[0]->surface().dofs[static_cast<std::size_t>(q)];
        shared_exact_trace[q] =
            app3d::transformed_manufactured_harmonic_value_3d(
                study_case.transform, dof.point);
        prescribed_normal_jump[q] =
            app3d::transformed_manufactured_harmonic_gradient_3d(
                study_case.transform, dof.point).dot(dof.normal);
    }
    shared_exact_trace.array() -= surface_weighted_mean(
        pipelines[0]->surface(), shared_exact_trace);
    prescribed_normal_jump.array() -= surface_weighted_mean(
        pipelines[0]->surface(), prescribed_normal_jump);
    if (!shared_exact_trace.allFinite()
        || !prescribed_normal_jump.allFinite()) {
        throw std::runtime_error(
            "Neumann edge-Cauchy manufactured data are non-finite");
    }

    const double setup_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - setup_start).count();
    const double setup_recorded_before_remainder = phase_record_sum_3d(
        capture_phase_records_3d(profile));
    profile.add(PhaseProfileKind3D::ExactFieldsAndOtherSetup,
        nonnegative_profile_remainder_3d(
            setup_wall_seconds, setup_recorded_before_remainder,
            "Neumann edge-Cauchy shared setup"), 1);

    NeumannEdgeCauchyPairRun3D result;
    result.shared_setup_phases = capture_phase_records_3d(profile);
    require_phase_wall_match_3d(
        phase_record_sum_3d(result.shared_setup_phases),
        setup_wall_seconds, "Neumann edge-Cauchy shared setup");

    const app3d::HarmonicCauchyFit3D& edge_fit =
        pipelines[1]->cauchy_fit();
    const PublicEdgeCauchyDiagnostics3D edge_diagnostics =
        public_edge_cauchy_diagnostics_3d(
            geometry.native_surface, surface_dofs, h, edge_fit);
    result.setup_factorization_count =
        edge_diagnostics.legacy_factorization_count;
    const NeumannRigidGeometryDiagnostics3D geometry_diagnostics =
        neumann_rigid_geometry_diagnostics_3d(
            grid_pair, pipelines[0]->correction_support());
    for (const auto& pipeline : pipelines) {
        const auto setup_diagnostics =
            pipeline->restrict_owner_preprocess_diagnostics();
        const std::uint64_t setup_queries = static_cast<std::uint64_t>(
            pipeline->restrict_owner_geometry_query_count());
        if (setup_queries != setup_diagnostics.wrong_side_queries) {
            throw std::logic_error(
                "Neumann edge-Cauchy setup query-count mismatch");
        }
    }

    Eigen::VectorXd value_probe(surface_size);
    for (int q = 0; q < surface_size; ++q) {
        value_probe[q] = std::sin(0.37 * (q + 1))
                       + 0.2 * std::cos(0.11 * (q + 1));
    }
    const Eigen::VectorXd zero_grid =
        Eigen::VectorXd::Zero(grid.num_dofs());
    const HarmonicJetField3D legacy_probe =
        pipelines[0]->field_from_grid_and_jumps(
            zero_grid, value_probe, prescribed_normal_jump);
    const HarmonicJetField3D augmented_probe =
        pipelines[1]->field_from_grid_and_jumps(
            zero_grid, value_probe, prescribed_normal_jump);
    std::vector<bool> affected(
        static_cast<std::size_t>(surface_size), false);
    for (int center = 0;
         center < static_cast<int>(edge_fit.surface_maps().size()); ++center) {
        affected[static_cast<std::size_t>(center)] =
            !edge_fit.surface_maps()[static_cast<std::size_t>(center)]
                 .edge_point_ids.empty();
    }
    const bool far_centers_bitwise_legacy =
        far_cauchy_rows_bitwise_equal_3d(
            legacy_probe.coefficients, augmented_probe.coefficients,
            affected);
    if (!far_centers_bitwise_legacy) {
        throw std::logic_error(
            "Neumann edge-Cauchy structural probe changed a far center");
    }

    std::array<NeumannEdgePreprocessSnapshot3D, 2> stable_snapshots;
    for (std::size_t index = 0; index < pipelines.size(); ++index) {
        stable_snapshots[index] =
            capture_neumann_edge_preprocess_snapshot_3d(*pipelines[index]);
        const auto& setup_diagnostics =
            pipelines[index]->restrict_owner_preprocess_diagnostics();
        if (stable_snapshots[index].output_digest
                    != preprocess_timings[index].output_digest
            || stable_snapshots[index].wrong_side_queries
                    != setup_diagnostics.wrong_side_queries
            || stable_snapshots[index].geometry_queries
                    != static_cast<std::uint64_t>(
                        pipelines[index]->restrict_owner_geometry_query_count())) {
            throw std::logic_error(
                "Neumann edge-Cauchy stable setup snapshot mismatch");
        }
    }
    if (stable_snapshots[0].workload_fingerprint
            != stable_snapshots[1].workload_fingerprint
        || stable_snapshots[0].output_digest
            != stable_snapshots[1].output_digest) {
        throw std::logic_error(
            "Neumann edge-Cauchy route pipelines changed owner preprocessing");
    }

    std::array<NeumannEdgePreprocessSnapshot3D, 2> before_snapshots;
    std::array<NeumannEdgePreprocessSnapshot3D, 2> after_snapshots;
    std::array<Eigen::VectorXd, 2> solved_value_jumps;
    std::array<Eigen::MatrixXd, 2> solved_coefficients;
    std::array<Eigen::VectorXd, 2> solved_normal_jumps;
    std::array<Eigen::VectorXd, 2> used_exact_traces;
    for (std::size_t index = 0; index < modes.size(); ++index) {
        PanelCenterHarmonicJetKFBI3D& pipeline = *pipelines[index];
        auto& measurement = result.measurements[index];
        measurement.case_id = study_case.id;
        measurement.N = N;
        measurement.h = h;
        measurement.mode = modes[index];
        measurement.expected_non_g1_connections =
            edge_diagnostics.expected_non_g1_connections;
        measurement.covered_non_g1_connections =
            edge_diagnostics.covered_non_g1_connections;
        measurement.edge_sample_count =
            edge_diagnostics.edge_sample_count;
        measurement.affected_center_count =
            edge_diagnostics.affected_center_count;
        measurement.corner_center_count =
            edge_diagnostics.corner_center_count;
        measurement.unrelated_sample_or_attachment_count =
            edge_diagnostics.unrelated_sample_or_attachment_count;
        measurement.rank_deficient_fit_count =
            edge_diagnostics.rank_deficient_fit_count;
        measurement.harmonic_cubic_reproduction_defect =
            edge_diagnostics.harmonic_cubic_reproduction_defect;
        measurement.edge_condition_max =
            edge_diagnostics.edge_condition_max;
        measurement.local_condition_max =
            edge_diagnostics.local_condition_max;
        measurement.shared_setup_seconds = setup_wall_seconds;
        measurement.far_centers_bitwise_legacy =
            far_centers_bitwise_legacy;
        measurement.geometry_diagnostics_pass = label_mismatches == 0
            && geometry_diagnostics.unsafe_label_changing_edges == 0
            && geometry_diagnostics.gap_crossings == 0
            && geometry_diagnostics.endpoint_crossings == 0
            && geometry_diagnostics.triangle_fallback_crossings == 0;

        before_snapshots[index] =
            capture_neumann_edge_preprocess_snapshot_3d(pipeline);
        result.factorization_counts_before[index] =
            edge_diagnostics.legacy_factorization_count;
        const app3d::HarmonicCauchyPreprocessAudit3D cauchy_audit_before =
            pipeline.cauchy_fit().audit();
        const PhaseRecordArray3D phase_before =
            capture_phase_records_3d(profile);
        const auto route_start = std::chrono::steady_clock::now();
        const SolveMetrics3D solve = run_neumann_case(
            grid, grid_pair, pipeline, study_case.transform,
            gmres_max_iterations, restrict_mode,
            &result.residual_histories[index], nullptr,
            &solved_value_jumps[index], &solved_coefficients[index],
            &solved_normal_jumps[index], &used_exact_traces[index],
            &shared_exact_trace, &prescribed_normal_jump);
        const auto route_end = std::chrono::steady_clock::now();
        const PhaseRecordArray3D phase_after_children =
            capture_phase_records_3d(profile);
        const double mode_wall_seconds = std::chrono::duration<double>(
            route_end - route_start).count();

        result.shared_exact_trace_bitwise[index] =
            bitwise_equal_vector_3d(
                used_exact_traces[index], shared_exact_trace);
        result.shared_normal_jump_bitwise[index] =
            bitwise_equal_vector_3d(
                solved_normal_jumps[index], prescribed_normal_jump);
        if (index != 0) {
            result.shared_exact_trace_bitwise[index] =
                result.shared_exact_trace_bitwise[index]
                && bitwise_equal_vector_3d(
                    used_exact_traces[0], used_exact_traces[index]);
            result.shared_normal_jump_bitwise[index] =
                result.shared_normal_jump_bitwise[index]
                && bitwise_equal_vector_3d(
                    solved_normal_jumps[0], solved_normal_jumps[index]);
        }
        if (!result.shared_exact_trace_bitwise[index]
            || !result.shared_normal_jump_bitwise[index]) {
            throw std::logic_error(
                "Neumann edge-Cauchy modes did not share manufactured data bitwise");
        }
        after_snapshots[index] =
            capture_neumann_edge_preprocess_snapshot_3d(pipeline);
        result.factorization_counts_after[index] =
            edge_diagnostics.legacy_factorization_count;
        const auto& cauchy_audit_after = pipeline.cauchy_fit().audit();
        if (result.factorization_counts_after[index]
                != result.factorization_counts_before[index]
            || result.factorization_counts_before[index]
                != result.setup_factorization_count
            || cauchy_audit_before.geometry_query_count
                != cauchy_audit_after.geometry_query_count
            || cauchy_audit_before.svd_factorization_count
                != cauchy_audit_after.svd_factorization_count
            || cauchy_audit_before.fingerprint
                != cauchy_audit_after.fingerprint) {
            throw std::logic_error(
                "Neumann edge-Cauchy GMRES changed factorization count");
        }

        const PhaseRecordArray3D child_deltas =
            subtract_phase_records_3d(
                phase_after_children, phase_before,
                "Neumann edge-Cauchy route children");
        const std::array<PhaseProfileKind3D, 6> route_children{{
            PhaseProfileKind3D::EdgeAuxiliaryValues,
            PhaseProfileKind3D::CauchyCoefficients,
            PhaseProfileKind3D::SpreadRhsAssembly,
            PhaseProfileKind3D::FftBulkSolve,
            PhaseProfileKind3D::RestrictContinuedSamples,
            PhaseProfileKind3D::RestrictRecovery}};
        double child_seconds = 0.0;
        for (PhaseProfileKind3D kind : route_children) {
            child_seconds += child_deltas[static_cast<std::size_t>(kind)].seconds;
        }
        profile.add(PhaseProfileKind3D::GmresAndOtherRoute,
            nonnegative_profile_remainder_3d(
                mode_wall_seconds, child_seconds,
                "Neumann edge-Cauchy mode runtime"), 1);
        const PhaseRecordArray3D phase_after =
            capture_phase_records_3d(profile);
        result.runtime_phase_deltas[index] =
            subtract_phase_records_3d(
                phase_after, phase_before,
                "Neumann edge-Cauchy mode runtime");
        require_phase_wall_match_3d(
            phase_record_sum_3d(result.runtime_phase_deltas[index]),
            mode_wall_seconds,
            "Neumann edge-Cauchy mode runtime");

        measurement.gmres_converged = solve.converged;
        measurement.gmres_iterations = solve.iterations;
        measurement.gmres_relative_residual =
            solve.gmres_relative_residual;
        measurement.residual_history_valid =
            app3d::neumann_edge_cauchy_residual_history_valid_3d(
                result.residual_histories[index], solve.iterations,
                solve.gmres_relative_residual);
        if (!measurement.residual_history_valid) {
            throw std::runtime_error(
                "Neumann edge-Cauchy GMRES residual history is invalid");
        }
        measurement.density_linf = solve.density_linf;
        measurement.density_l2 = solve.density_l2;
        measurement.interior_linf = solve.interior_linf;
        measurement.interior_l2 = solve.interior_l2;
        measurement.mode_runtime_seconds = mode_wall_seconds;
        measurement.total_seconds =
            measurement.shared_setup_seconds
            + measurement.mode_runtime_seconds;

        const Eigen::VectorXd shared_edge_values = edge_fit.apply(
            solved_value_jumps[index], solved_normal_jumps[index]).edge_values;
        const auto& samples = edge_fit.edge_maps();
        if (shared_edge_values.size()
            != static_cast<Eigen::Index>(samples.size())) {
            throw std::logic_error(
                "Neumann edge-Cauchy edge-value output has wrong size");
        }
        if (!shared_edge_values.allFinite()) {
            throw std::runtime_error(
                "Neumann edge-Cauchy auxiliary values are non-finite");
        }
        double discrepancy = 0.0;
        for (std::size_t q = 0; q < samples.size(); ++q) {
            const auto& sample = samples[q];
            if (sample.value_sector_counts[0] <= 0
                || sample.value_sector_counts[1] <= 0) {
                throw std::logic_error(
                    "public edge-Cauchy diagnostic has an empty owner sector");
            }
            const int first_owner_dof = sample.value_ids.front();
            const int second_owner_dof = sample.value_ids[
                static_cast<std::size_t>(sample.value_sector_counts[0])];
            NeumannEdgeCauchyEdgeValueRow3D edge_row;
            edge_row.connection_index = sample.point.connection_id;
            edge_row.sample_index = sample.point.cell_id;
            edge_row.sample_count = sample.point.cell_count;
            edge_row.first_owner_dof = first_owner_dof;
            edge_row.second_owner_dof = second_owner_dof;
            edge_row.first_value = evaluate_edge_owner_value_3d(
                pipeline.surface(), first_owner_dof,
                sample.point.point, h, solved_coefficients[index]);
            edge_row.second_value = evaluate_edge_owner_value_3d(
                pipeline.surface(), second_owner_dof,
                sample.point.point, h, solved_coefficients[index]);
            edge_row.shared_auxiliary_value =
                shared_edge_values[static_cast<Eigen::Index>(q)];
            edge_row.first_second_difference = std::abs(
                edge_row.first_value - edge_row.second_value);
            edge_row.first_shared_difference = std::abs(
                edge_row.first_value - edge_row.shared_auxiliary_value);
            edge_row.second_shared_difference = std::abs(
                edge_row.second_value - edge_row.shared_auxiliary_value);
            const std::array<double, 6> edge_values{{
                edge_row.first_value, edge_row.second_value,
                edge_row.shared_auxiliary_value,
                edge_row.first_second_difference,
                edge_row.first_shared_difference,
                edge_row.second_shared_difference}};
            if (!app3d::neumann_edge_cauchy_edge_value_row_finite_3d(
                    edge_values)) {
                throw std::runtime_error(
                    "Neumann edge-Cauchy edge comparison is non-finite");
            }
            discrepancy = std::max(
                discrepancy, edge_row.first_second_difference);
            result.edge_value_rows[index].push_back(edge_row);
        }
        measurement.incident_edge_discrepancy_linf = discrepancy;
        measurement.finite_metrics = measurement.residual_history_valid
            && finite_neumann_owner_metrics_3d(solve)
            && std::isfinite(measurement.shared_setup_seconds)
            && std::isfinite(measurement.mode_runtime_seconds)
            && std::isfinite(measurement.total_seconds)
            && std::isfinite(measurement.incident_edge_discrepancy_linf)
            && std::isfinite(measurement.edge_condition_max)
            && std::isfinite(measurement.local_condition_max);
        measurement.pair_completed = true;
        measurement.owner_invariants_pass =
            neumann_edge_preprocess_snapshot_equal_3d(
                stable_snapshots[index], before_snapshots[index])
            && neumann_edge_preprocess_snapshot_equal_3d(
                stable_snapshots[index], after_snapshots[index]);
    }

    std::array<NeumannEdgePreprocessSnapshot3D, 2> final_snapshots;
    for (std::size_t index = 0; index < pipelines.size(); ++index) {
        final_snapshots[index] =
            capture_neumann_edge_preprocess_snapshot_3d(*pipelines[index]);
    }
    result.owner_snapshots = {
        neumann_edge_preprocess_invariant_snapshot_3d(
            stable_snapshots[0], stable_snapshots[0]),
        neumann_edge_preprocess_invariant_snapshot_3d(
            before_snapshots[0], stable_snapshots[0]),
        neumann_edge_preprocess_invariant_snapshot_3d(
            after_snapshots[0], stable_snapshots[0]),
        neumann_edge_preprocess_invariant_snapshot_3d(
            before_snapshots[1], stable_snapshots[1]),
        neumann_edge_preprocess_invariant_snapshot_3d(
            after_snapshots[1], stable_snapshots[1]),
        neumann_edge_preprocess_invariant_snapshot_3d(
            final_snapshots[1], stable_snapshots[1])};
    const bool shared_preprocess_pass =
        app3d::neumann_edge_shared_preprocess_pass_3d(
            result.owner_snapshots);
    for (auto& measurement : result.measurements) {
        measurement.owner_invariants_pass =
            measurement.owner_invariants_pass && shared_preprocess_pass;
        measurement.shared_preprocess_pass = shared_preprocess_pass;
    }
    return result;
}

NeumannEdgeCauchyPairRun3D failed_neumann_edge_cauchy_pair_3d(
    int N, const app3d::LPrismRigidStudyCase3D& study_case)
{
    const std::array<app3d::NeumannEdgeCauchyMode3D, 2> modes{{
        app3d::NeumannEdgeCauchyMode3D::None,
        app3d::NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues}};
    NeumannEdgeCauchyPairRun3D result;
    result.owner_snapshots.resize(6);
    for (std::size_t index = 0; index < modes.size(); ++index) {
        auto& measurement = result.measurements[index];
        measurement.case_id = study_case.id;
        measurement.N = N;
        measurement.h = kBoxSide / static_cast<double>(N);
        measurement.mode = modes[index];
        measurement.pair_completed = false;
        measurement.residual_history_valid = false;
        measurement.finite_metrics = false;
    }
    return result;
}

std::vector<app3d::LPrismRigidStudyCase3D> neumann_edge_pilot_cases_3d()
{
    const std::array<std::string, 3> wanted{{
        "baseline", "ty_m0083", "rot_axis123_17deg"}};
    const auto available = app3d::make_l_prism_rigid_study_cases_3d();
    std::vector<app3d::LPrismRigidStudyCase3D> result;
    for (const std::string& id : wanted) {
        const auto found = std::find_if(available.begin(), available.end(),
            [&](const app3d::LPrismRigidStudyCase3D& item) { return item.id == id; });
        if (found == available.end())
            throw std::logic_error("missing Neumann edge pilot case: " + id);
        result.push_back(*found);
    }
    return result;
}

std::vector<app3d::NeumannEdgeCauchyMeasurement3D>
neumann_edge_cauchy_measurements_3d(
    const std::vector<NeumannEdgeCauchyPairRun3D>& pairs)
{
    std::vector<app3d::NeumannEdgeCauchyMeasurement3D> result;
    result.reserve(2 * pairs.size());
    for (const auto& pair : pairs) {
        result.push_back(pair.measurements[0]);
        result.push_back(pair.measurements[1]);
    }
    return result;
}

const app3d::NeumannEdgeCauchyDerivedRow3D*
find_neumann_edge_cauchy_derived_row_3d(
    const app3d::NeumannEdgeCauchyEvaluation3D& evaluation,
    const app3d::NeumannEdgeCauchyMeasurement3D& measurement)
{
    const auto found = std::find_if(
        evaluation.rows.begin(), evaluation.rows.end(),
        [&](const app3d::NeumannEdgeCauchyDerivedRow3D& row) {
            return row.measurement.case_id == measurement.case_id
                && row.measurement.N == measurement.N
                && row.measurement.mode == measurement.mode;
        });
    return found == evaluation.rows.end()
        ? nullptr : std::addressof(*found);
}

bool neumann_edge_cauchy_pair_execution_pass_3d(
    const NeumannEdgeCauchyPairRun3D& pair,
    const app3d::NeumannEdgeCauchyEvaluation3D& evaluation)
{
    return std::all_of(
        pair.measurements.begin(), pair.measurements.end(),
        [&](const auto& measurement) {
            const auto* derived =
                find_neumann_edge_cauchy_derived_row_3d(
                    evaluation, measurement);
            return derived != nullptr
                && derived->row_pass
                    == app3d::RigidStudyCriterionStatus3D::Pass;
        });
}

void write_neumann_edge_cauchy_checkpoints_3d(
    const std::filesystem::path& output_dir,
    const std::vector<NeumannEdgeCauchyPairRun3D>& pairs,
    const app3d::NeumannEdgeCauchyEvaluation3D& evaluation)
{
    std::filesystem::create_directories(output_dir);
    std::ofstream summary = open_output_file(output_dir / "summary.csv");
    summary << std::setprecision(17) << std::boolalpha
        << "case_id,N,h,mode,pair_completed,residual_history_valid,"
           "shared_exact_trace_bitwise,shared_normal_jump_bitwise,"
           "finite_metrics,gmres_converged,"
           "gmres_iterations,"
           "gmres_relative_residual,density_linf,density_l2,interior_linf,"
           "interior_l2,incident_edge_discrepancy_linf,"
           "expected_non_g1_connections,covered_non_g1_connections,"
           "edge_sample_count,affected_center_count,corner_center_count,"
           "unrelated_sample_or_attachment_count,rank_deficient_fit_count,"
           "harmonic_cubic_reproduction_defect,edge_condition_max,"
           "local_condition_max,far_centers_bitwise_legacy,"
           "geometry_diagnostics_pass,owner_invariants_pass,"
           "shared_preprocess_pass,density_linf_order,density_l2_order,"
           "interior_linf_order,interior_l2_order,"
           "density_linf_ratio_to_legacy,density_l2_ratio_to_legacy,"
           "interior_linf_ratio_to_legacy,interior_l2_ratio_to_legacy,"
           "edge_discrepancy_ratio_to_legacy,row_pass,"
           "shared_setup_seconds,edge_auxiliary_values_seconds,"
           "cauchy_coefficients_seconds,spread_rhs_assembly_seconds,"
           "fft_bulk_solve_seconds,restrict_continued_samples_seconds,"
           "restrict_recovery_seconds,gmres_and_other_route_seconds,"
           "mode_runtime_seconds,total_seconds\n";
    for (const auto& pair : pairs) {
        for (std::size_t mode_index = 0; mode_index < 2; ++mode_index) {
            const auto& measurement = pair.measurements[mode_index];
            const auto* derived =
                find_neumann_edge_cauchy_derived_row_3d(
                    evaluation, measurement);
            if (derived == nullptr) {
                throw std::logic_error(
                    "missing Neumann edge-Cauchy derived row");
            }
            const auto& phases = pair.runtime_phase_deltas[mode_index];
            const auto seconds = [&](PhaseProfileKind3D kind) {
                return phases[static_cast<std::size_t>(kind)].seconds;
            };
            summary << measurement.case_id << ',' << measurement.N
                << ',' << measurement.h << ','
                << app3d::neumann_edge_cauchy_mode_name_3d(
                       measurement.mode)
                << ',' << measurement.pair_completed
                << ',' << measurement.residual_history_valid
                << ',' << pair.shared_exact_trace_bitwise[mode_index]
                << ',' << pair.shared_normal_jump_bitwise[mode_index]
                << ',' << measurement.finite_metrics
                << ',' << measurement.gmres_converged
                << ',' << measurement.gmres_iterations
                << ',' << measurement.gmres_relative_residual
                << ',' << measurement.density_linf
                << ',' << measurement.density_l2
                << ',' << measurement.interior_linf
                << ',' << measurement.interior_l2
                << ',' << measurement.incident_edge_discrepancy_linf
                << ',' << measurement.expected_non_g1_connections
                << ',' << measurement.covered_non_g1_connections
                << ',' << measurement.edge_sample_count
                << ',' << measurement.affected_center_count
                << ',' << measurement.corner_center_count
                << ',' << measurement.unrelated_sample_or_attachment_count
                << ',' << measurement.rank_deficient_fit_count
                << ',' << measurement.harmonic_cubic_reproduction_defect
                << ',' << measurement.edge_condition_max
                << ',' << measurement.local_condition_max
                << ',' << measurement.far_centers_bitwise_legacy
                << ',' << measurement.geometry_diagnostics_pass
                << ',' << measurement.owner_invariants_pass
                << ',' << measurement.shared_preprocess_pass
                << ',' << derived->density_linf_order
                << ',' << derived->density_l2_order
                << ',' << derived->interior_linf_order
                << ',' << derived->interior_l2_order
                << ',' << derived->density_linf_ratio_to_legacy
                << ',' << derived->density_l2_ratio_to_legacy
                << ',' << derived->interior_linf_ratio_to_legacy
                << ',' << derived->interior_l2_ratio_to_legacy
                << ',' << derived->edge_discrepancy_ratio_to_legacy
                << ',' << criterion_status_name(derived->row_pass)
                << ',' << measurement.shared_setup_seconds
                << ',' << seconds(PhaseProfileKind3D::EdgeAuxiliaryValues)
                << ',' << seconds(PhaseProfileKind3D::CauchyCoefficients)
                << ',' << seconds(PhaseProfileKind3D::SpreadRhsAssembly)
                << ',' << seconds(PhaseProfileKind3D::FftBulkSolve)
                << ',' << seconds(PhaseProfileKind3D::RestrictContinuedSamples)
                << ',' << seconds(PhaseProfileKind3D::RestrictRecovery)
                << ',' << seconds(PhaseProfileKind3D::GmresAndOtherRoute)
                << ',' << measurement.mode_runtime_seconds
                << ',' << measurement.total_seconds << '\n';
        }
    }

    std::ofstream edges = open_output_file(
        output_dir / "edge_values.csv");
    edges << std::setprecision(17)
        << "case_id,N,mode,connection_index,sample_index,sample_count,"
           "first_owner_dof,second_owner_dof,first_value,second_value,"
           "shared_auxiliary_value,first_second_difference,"
           "first_shared_difference,second_shared_difference\n";
    for (const auto& pair : pairs) {
        for (std::size_t mode_index = 0; mode_index < 2; ++mode_index) {
            const auto& measurement = pair.measurements[mode_index];
            for (const auto& row : pair.edge_value_rows[mode_index]) {
                edges << measurement.case_id << ',' << measurement.N
                    << ',' << app3d::neumann_edge_cauchy_mode_name_3d(
                           measurement.mode)
                    << ',' << row.connection_index
                    << ',' << row.sample_index
                    << ',' << row.sample_count
                    << ',' << row.first_owner_dof
                    << ',' << row.second_owner_dof
                    << ',' << row.first_value
                    << ',' << row.second_value
                    << ',' << row.shared_auxiliary_value
                    << ',' << row.first_second_difference
                    << ',' << row.first_shared_difference
                    << ',' << row.second_shared_difference << '\n';
            }
        }
    }

    std::ofstream residuals = open_output_file(
        output_dir / "gmres_residuals.csv");
    residuals << std::setprecision(17)
              << "case_id,N,mode,iteration,residual\n";
    for (const auto& pair : pairs) {
        for (std::size_t mode_index = 0; mode_index < 2; ++mode_index) {
            const auto& measurement = pair.measurements[mode_index];
            for (std::size_t iteration = 0;
                 iteration < pair.residual_histories[mode_index].size();
                 ++iteration) {
                residuals << measurement.case_id << ',' << measurement.N
                    << ',' << app3d::neumann_edge_cauchy_mode_name_3d(
                           measurement.mode)
                    << ',' << iteration << ','
                    << pair.residual_histories[mode_index][iteration]
                    << '\n';
            }
        }
    }

    std::ofstream owners = open_output_file(
        output_dir / "owner_diagnostics.csv");
    owners << std::boolalpha
        << "case_id,N,mode,pair_completed,shared_exact_trace_bitwise,"
           "shared_normal_jump_bitwise,stable_workload_fingerprint,"
           "stable_output_digest,stable_wrong_side_queries,"
           "stable_geometry_queries,before_workload_fingerprint,"
           "after_workload_fingerprint,before_output_digest,"
           "after_output_digest,before_wrong_side_queries,"
           "after_wrong_side_queries,before_geometry_queries,"
           "after_geometry_queries,final_workload_fingerprint,"
           "final_output_digest,final_wrong_side_queries,"
           "final_geometry_queries,diagnostics_match_reference,"
           "setup_factorization_count,factorization_count_before_gmres,"
           "factorization_count_after_gmres,geometry_diagnostics_pass,"
           "owner_invariants_pass,shared_preprocess_pass\n";
    for (const auto& pair : pairs) {
        if (pair.owner_snapshots.size() != 6) {
            throw std::logic_error(
                "Neumann edge-Cauchy owner snapshot chain is incomplete");
        }
        const auto& stable = pair.owner_snapshots[0];
        const auto& final = pair.owner_snapshots[5];
        for (std::size_t mode_index = 0; mode_index < 2; ++mode_index) {
            const auto& measurement = pair.measurements[mode_index];
            const auto& before = pair.owner_snapshots[1 + 2 * mode_index];
            const auto& after = pair.owner_snapshots[2 + 2 * mode_index];
            owners << measurement.case_id << ',' << measurement.N
                << ',' << app3d::neumann_edge_cauchy_mode_name_3d(
                       measurement.mode)
                << ',' << measurement.pair_completed
                << ',' << pair.shared_exact_trace_bitwise[mode_index]
                << ',' << pair.shared_normal_jump_bitwise[mode_index]
                << ',' << stable.workload_fingerprint
                << ',' << stable.output_digest
                << ',' << stable.wrong_side_queries
                << ',' << stable.geometry_queries
                << ',' << before.workload_fingerprint
                << ',' << after.workload_fingerprint
                << ',' << before.output_digest
                << ',' << after.output_digest
                << ',' << before.wrong_side_queries
                << ',' << after.wrong_side_queries
                << ',' << before.geometry_queries
                << ',' << after.geometry_queries
                << ',' << final.workload_fingerprint
                << ',' << final.output_digest
                << ',' << final.wrong_side_queries
                << ',' << final.geometry_queries
                << ',' << (stable.diagnostics_match_reference
                    && before.diagnostics_match_reference
                    && after.diagnostics_match_reference
                    && final.diagnostics_match_reference)
                << ',' << pair.setup_factorization_count
                << ',' << pair.factorization_counts_before[mode_index]
                << ',' << pair.factorization_counts_after[mode_index]
                << ',' << measurement.geometry_diagnostics_pass
                << ',' << measurement.owner_invariants_pass
                << ',' << measurement.shared_preprocess_pass << '\n';
        }
    }

    std::ofstream phases = open_output_file(
        output_dir / "phase_profile.csv");
    phases << std::setprecision(17)
        << "case_id,N,mode,scope,phase,seconds,calls\n";
    for (const auto& pair : pairs) {
        const auto& first = pair.measurements[0];
        for (std::size_t q = 0;
             q < app3d::phase_profile_kind_count_3d(); ++q) {
            const auto kind = static_cast<PhaseProfileKind3D>(q);
            phases << first.case_id << ',' << first.N
                << ",shared,shared_setup,"
                << app3d::phase_profile_name_3d(kind) << ','
                << pair.shared_setup_phases[q].seconds << ','
                << pair.shared_setup_phases[q].calls << '\n';
        }
        for (std::size_t mode_index = 0; mode_index < 2; ++mode_index) {
            const auto& measurement = pair.measurements[mode_index];
            for (std::size_t q = 0;
                 q < app3d::phase_profile_kind_count_3d(); ++q) {
                const auto kind = static_cast<PhaseProfileKind3D>(q);
                phases << measurement.case_id << ',' << measurement.N
                    << ',' << app3d::neumann_edge_cauchy_mode_name_3d(
                           measurement.mode)
                    << ",mode_runtime,"
                    << app3d::phase_profile_name_3d(kind) << ','
                    << pair.runtime_phase_deltas[mode_index][q].seconds
                    << ',' << pair.runtime_phase_deltas[mode_index][q].calls
                    << '\n';
            }
        }
    }

    std::ofstream acceptance = open_output_file(
        output_dir / "acceptance.csv");
    acceptance
        << "completeness_pass,structure_pass,reproduction_pass,gmres_pass,"
           "error_guard_pass,order_pass,rigid_spread_pass,"
           "edge_discrepancy_pass,geometry_owner_pass,"
           "extended_evidence_pass,overall_pass\n"
        << criterion_status_name(
               evaluation.acceptance.completeness_pass) << ','
        << criterion_status_name(
               evaluation.acceptance.structure_pass) << ','
        << criterion_status_name(
               evaluation.acceptance.reproduction_pass) << ','
        << criterion_status_name(
               evaluation.acceptance.gmres_pass) << ','
        << criterion_status_name(
               evaluation.acceptance.error_guard_pass) << ','
        << criterion_status_name(
               evaluation.acceptance.order_pass) << ','
        << criterion_status_name(
               evaluation.acceptance.rigid_spread_pass) << ','
        << criterion_status_name(
               evaluation.acceptance.edge_discrepancy_pass) << ','
        << criterion_status_name(
               evaluation.acceptance.geometry_owner_pass) << ','
        << criterion_status_name(
               evaluation.acceptance.extended_evidence_pass) << ','
        << criterion_status_name(
               evaluation.acceptance.overall_pass) << '\n';
}

int run_neumann_edge_cauchy_study_3d(
    std::vector<int> levels,
    bool force_extended)
{
    try {
        levels = app3d::normalize_neumann_edge_cauchy_levels_3d(
            std::move(levels));
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(std::string(error.what())
            + "; accepted prefixes: 32; 32 64; 32 64 128");
    }
    const auto cases = neumann_edge_pilot_cases_3d();
    std::vector<std::string> case_ids;
    for (const auto& study_case : cases)
        case_ids.push_back(study_case.id);
#ifdef KFBIM_APP_OUTPUT_DIR
    std::filesystem::path output_dir =
        std::filesystem::path(KFBIM_APP_OUTPUT_DIR)
        / "neumann_edge_cauchy_3d";
#else
    std::filesystem::path output_dir =
        "output/neumann_edge_cauchy_3d";
#endif
    const char* output_override = std::getenv(
        "KFBIM_3D_NEUMANN_EDGE_CAUCHY_OUTPUT_DIR");
    if (output_override != nullptr) output_dir = output_override;

    const bool require_complete_pilot =
        std::find(levels.begin(), levels.end(), 64) != levels.end();
    std::vector<NeumannEdgeCauchyPairRun3D> pairs;
    app3d::NeumannEdgeCauchyEvaluation3D evaluation =
        app3d::evaluate_neumann_edge_cauchy_study_3d(
            {}, case_ids, require_complete_pilot);
    std::cout << "KFBI3D Neumann non-G1 edge-Cauchy A/B study\n"
              << "  route=joint_tricubic_crossing_owner"
                 " owner=region_closest_hybrid"
                 " cauchy=g1_nearest/degree3/48/28\n"
              << "  modes=none,non_g1_auxiliary_values"
                 " gmres_tolerance=2e-10 restart=80 cap=80 levels=";
    for (std::size_t index = 0; index < levels.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << levels[index];
    }
    std::cout << " cases=" << cases.size() << '\n';

    for (int N : levels) {
        if (N == 128) {
            evaluation = app3d::evaluate_neumann_edge_cauchy_study_3d(
                neumann_edge_cauchy_measurements_3d(pairs),
                case_ids, true);
            write_neumann_edge_cauchy_checkpoints_3d(
                output_dir, pairs, evaluation);
            const bool enter_extended =
                app3d::neumann_edge_cauchy_should_enter_n128_3d(
                    evaluation.all_pass, force_extended);
            if (!enter_extended) {
                std::cerr
                    << "error: N=128 gated off because the completed N=32/64 "
                       "Neumann edge-Cauchy pilot did not pass\n";
                return 1;
            }
            if (!evaluation.all_pass) {
                std::cerr
                    << "warning: forcing N=128 extended evidence after failed "
                       "N=32/64 acceptance; acceptance thresholds are unchanged\n";
            }
        }
        for (const auto& study_case : cases) {
            std::cout << "[neumann-edge-cauchy-study] case="
                << study_case.id << " N=" << N << " setup\n";
            bool pair_execution_pass = false;
            const auto process =
                app3d::process_neumann_edge_cauchy_pair_3d(
                    N,
                    [&] {
                        pairs.push_back(run_neumann_edge_cauchy_pair_3d(
                            N, study_case));
                    },
                    [&] {
                        pairs.push_back(failed_neumann_edge_cauchy_pair_3d(
                            N, study_case));
                    },
                    [&] {
                        evaluation =
                            app3d::evaluate_neumann_edge_cauchy_study_3d(
                                neumann_edge_cauchy_measurements_3d(pairs),
                                case_ids, require_complete_pilot);
                        pair_execution_pass =
                            neumann_edge_cauchy_pair_execution_pass_3d(
                                pairs.back(), evaluation);
                        write_neumann_edge_cauchy_checkpoints_3d(
                            output_dir, pairs, evaluation);
                        return pair_execution_pass;
                    });
            if (!process.evidence_completed) {
                std::cerr << (N == 128 ? "warning: " : "error: ")
                    << "N=" << N
                    << " Neumann edge-Cauchy evidence failed: case="
                    << study_case.id
                    << " reason=" << process.failure_message
                    << "; checkpoint records pair_completed=false\n";
            }

            const auto& pair = pairs.back();
            for (const auto& measurement : pair.measurements) {
                const auto* derived =
                    find_neumann_edge_cauchy_derived_row_3d(
                        evaluation, measurement);
                const bool row_pass = derived != nullptr
                    && derived->row_pass
                        == app3d::RigidStudyCriterionStatus3D::Pass;
                std::cout << "[neumann-edge-cauchy-study] case="
                    << measurement.case_id << " N=" << measurement.N
                    << " mode="
                    << app3d::neumann_edge_cauchy_mode_name_3d(
                           measurement.mode)
                    << " iter=" << measurement.gmres_iterations
                    << " residual="
                    << measurement.gmres_relative_residual
                    << " edge_linf="
                    << measurement.incident_edge_discrepancy_linf
                    << " setup_s=" << measurement.shared_setup_seconds
                    << " runtime_s=" << measurement.mode_runtime_seconds
                    << " pass=" << row_pass << '\n';
            }
            if (!pair_execution_pass) {
                std::cerr << (N == 128 ? "warning: " : "error: ")
                    << "Neumann edge-Cauchy structural or GMRES row gate "
                       "failed: case=" << study_case.id
                    << " N=" << N << '\n';
            }
            if (!process.continue_study) return 1;
        }
    }

    evaluation = app3d::evaluate_neumann_edge_cauchy_study_3d(
        neumann_edge_cauchy_measurements_3d(pairs),
        case_ids, require_complete_pilot);
    write_neumann_edge_cauchy_checkpoints_3d(
        output_dir, pairs, evaluation);
    std::cout << "[neumann-edge-cauchy-acceptance] completeness="
        << criterion_status_name(evaluation.acceptance.completeness_pass)
        << " structure="
        << criterion_status_name(evaluation.acceptance.structure_pass)
        << " reproduction="
        << criterion_status_name(evaluation.acceptance.reproduction_pass)
        << " gmres="
        << criterion_status_name(evaluation.acceptance.gmres_pass)
        << " error_guard="
        << criterion_status_name(evaluation.acceptance.error_guard_pass)
        << " order="
        << criterion_status_name(evaluation.acceptance.order_pass)
        << " rigid_spread="
        << criterion_status_name(evaluation.acceptance.rigid_spread_pass)
        << " edge_discrepancy="
        << criterion_status_name(
               evaluation.acceptance.edge_discrepancy_pass)
        << " geometry_owner="
        << criterion_status_name(evaluation.acceptance.geometry_owner_pass)
        << " extended_evidence="
        << criterion_status_name(
               evaluation.acceptance.extended_evidence_pass)
        << " overall="
        << criterion_status_name(evaluation.acceptance.overall_pass)
        << '\n';
    std::cout << "Neumann edge-Cauchy study output: "
              << output_dir.string() << '\n';

    if (evaluation.acceptance.extended_evidence_pass
        == app3d::RigidStudyCriterionStatus3D::Fail) {
        std::cerr << "warning: N=128 Neumann edge-Cauchy extended evidence "
                     "failed; coarse process acceptance is unchanged\n";
    }
    const bool exit_pass =
        app3d::neumann_edge_cauchy_study_exit_pass_3d(
            evaluation, require_complete_pilot);
    if (!exit_pass) {
        std::cerr << "error: Neumann edge-Cauchy study acceptance failed\n";
        return 1;
    }
    if (!require_complete_pilot) {
        std::cout << "Neumann edge-Cauchy N=32 prefix completed; two-level "
                     "comparisons and order remain non-gating\n";
    }
    return 0;
}

int run_neumann_edge_continuity_study_3d(std::vector<int> levels){
    try {
        levels = app3d::normalize_neumann_edge_continuity_levels_3d(std::move(levels));
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(std::string(error.what())
            + "; accepted prefixes: 32; 32 64; 32 64 128");
    }
    const std::vector<app3d::LPrismRigidStudyCase3D> cases = neumann_edge_pilot_cases_3d();
    std::vector<std::string> case_ids;
    for (const auto& study_case : cases) case_ids.push_back(study_case.id);
#ifdef KFBIM_APP_OUTPUT_DIR
    std::filesystem::path output_dir = std::filesystem::path(KFBIM_APP_OUTPUT_DIR)
        / "neumann_edge_continuity_3d";
#else
    std::filesystem::path output_dir = "output/neumann_edge_continuity_3d";
#endif
    const char* output_override = std::getenv(
        "KFBIM_3D_NEUMANN_EDGE_CONTINUITY_OUTPUT_DIR");
    if (output_override != nullptr) output_dir = output_override;

    const bool require_complete_pilot =
        std::find(levels.begin(), levels.end(), 64) != levels.end();
    std::vector<NeumannEdgeStudyRow3D> rows;
    app3d::NeumannEdgeContinuityEvaluation3D evaluation =
        app3d::evaluate_neumann_edge_continuity_study_3d(
            neumann_edge_measurements_3d(rows), case_ids, require_complete_pilot);
    std::cout << "KFBI3D Neumann non-G1 edge-continuity A/B study\n"
              << "  route=joint_tricubic_crossing_owner owner=region_closest_hybrid"
                 " cauchy=g1_nearest/degree3/48/28\n"
              << "  gmres_tolerance=2e-10 restart=80 cap=80 levels=";
    for (std::size_t index = 0; index < levels.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << levels[index];
    }
    std::cout << " cases=" << cases.size() << '\n';

    for (int N : levels) {
        if (N == 128) {
            evaluation = app3d::evaluate_neumann_edge_continuity_study_3d(
                neumann_edge_measurements_3d(rows), case_ids, true);
            write_neumann_edge_continuity_checkpoints_3d(output_dir, rows, evaluation);
            if (!evaluation.all_pass) {
                std::cerr << "error: N=128 gated off because the completed N=32/64 "
                             "Neumann edge pilot did not pass\n";
                return 1;
            }
        }
        for (const auto& study_case : cases) {
            std::cout << "[neumann-edge-continuity-study] case=" << study_case.id
                      << " N=" << N << " setup\n";
            std::array<NeumannEdgeStudyRow3D, 2> pair =
                run_neumann_edge_continuity_pair_3d(N, study_case);
            for (NeumannEdgeStudyRow3D& row : pair) rows.push_back(std::move(row));
            evaluation = app3d::evaluate_neumann_edge_continuity_study_3d(
                neumann_edge_measurements_3d(rows), case_ids, require_complete_pilot);
            write_neumann_edge_continuity_checkpoints_3d(output_dir, rows, evaluation);
            bool pair_pass = true;
            for (std::size_t q = rows.size() - 2; q < rows.size(); ++q) {
                const NeumannEdgeStudyRow3D& row = rows[q];
                const auto* derived = find_neumann_edge_derived_row_3d(evaluation, row);
                const bool row_pass = derived != nullptr
                    && derived->row_pass == app3d::RigidStudyCriterionStatus3D::Pass;
                pair_pass = pair_pass && row_pass;
                std::cout << "[neumann-edge-continuity-study] case=" << row.study_case.id
                    << " N=" << row.N << " density_space="
                    << app3d::neumann_density_space_name_3d(row.density_space)
                    << " iter=" << row.solve.iterations
                    << " residual=" << row.solve.gmres_relative_residual
                    << " edge_linf=" << row.edge_mismatch_linf
                    << " geometry=" << row.geometry_diagnostics_pass
                    << " owner=" << row.owner_invariants_pass
                    << " shared=" << row.shared_preprocess_pass
                    << " pass=" << row_pass << '\n';
            }
            if (!pair_pass) {
                std::cerr << (N == 128 ? "warning: " : "error: ")
                    << "Neumann edge-continuity structural or GMRES row gate "
                       "failed: case=" << study_case.id << " N=" << N << '\n';
                if (N != 128) return 1;
            }
        }
    }

    evaluation = app3d::evaluate_neumann_edge_continuity_study_3d(
        neumann_edge_measurements_3d(rows), case_ids, require_complete_pilot);
    write_neumann_edge_continuity_checkpoints_3d(output_dir, rows, evaluation);
    std::cout << "[neumann-edge-continuity-acceptance] completeness="
        << criterion_status_name(evaluation.acceptance.completeness_pass)
        << " topology=" << criterion_status_name(evaluation.acceptance.topology_pass)
        << " projector=" << criterion_status_name(evaluation.acceptance.projector_pass)
        << " exact_order=" << criterion_status_name(evaluation.acceptance.exact_trace_order_pass)
        << " gmres=" << criterion_status_name(evaluation.acceptance.gmres_pass)
        << " error_guard=" << criterion_status_name(evaluation.acceptance.error_guard_pass)
        << " edge_reduction=" << criterion_status_name(evaluation.acceptance.edge_reduction_pass)
        << " trend=" << criterion_status_name(evaluation.acceptance.trend_pass)
        << " geometry_owner=" << criterion_status_name(evaluation.acceptance.geometry_owner_pass)
        << " extended_evidence="
        << criterion_status_name(evaluation.acceptance.extended_evidence_pass)
        << " overall=" << criterion_status_name(evaluation.acceptance.overall_pass) << '\n';
    std::cout << "Neumann edge-continuity study output: " << output_dir.string() << '\n';
    const bool exit_pass =
        app3d::neumann_edge_continuity_study_exit_pass_3d(
            evaluation, require_complete_pilot);
    if (!exit_pass) {
        std::cerr << "error: Neumann edge-continuity study acceptance failed\n";
        return 1;
    }
    if (!require_complete_pilot && !evaluation.all_pass) {
        std::cout << "Neumann edge-continuity N=32 prefix completed; comparative "
                     "A/B hypotheses remain recorded but are not an execution gate\n";
    }
    return 0;
}
void print_usage(const char* executable)
{
    std::cout
        << "usage: " << executable
        << " [torus|cylinder|l_prism|all] [N ...]\n"
        << "       " << executable << " --rigid-study [N ...]\n"
        << "       " << executable << " --restrict-probe [N ...]\n"
        << "       " << executable << " --restrict-probe-owner [N ...]\n"
        << "       " << executable << " --restrict-profile-owner [N]\n"
        << "       " << executable << " --owner-preprocess-study [N ...]\n"
        << "       " << executable << " --neumann-owner-study [N ...]\n"
        << "       " << executable << " --neumann-rigid-study [N ...]\n"
        << "       " << executable << " --neumann-edge-continuity-study [N ...]\n"
        << "       " << executable
        << " --neumann-edge-cauchy-study [--force-extended] [N ...]\n"
        << "  Each N must be a power of two and at least 16 (default: 32).\n"
        << "  Rigid-study default levels: 32, 64, 128.\n"
        << "  Restrict-probe default levels: 32, 64.\n"
        << "  Restrict-profile default level: 128 (one level only).\n"
        << "  Owner-preprocess-study default levels: 16, 32, 64.\n"
        << "  Neumann-owner-study default levels: 32, 64, 128.\n"
        << "  Neumann-rigid-study levels are the refinement prefixes "
           "32; 32,64; or 32,64,128 (default: 32,64,128).\n"
        << "  Neumann-edge-continuity-study levels are the refinement prefixes "
           "32; 32,64; or 32,64,128 (default: 32,64).\n"
        << "  Neumann-edge-cauchy-study levels are the refinement prefixes "
           "32; 32,64; or 32,64,128 (default: 32,64).\n"
        << "  --force-extended records N=128 evidence after a failed coarse "
           "gate without changing acceptance.\n"
        << "  This stage builds native NURBS parameter-cell-center surface\n"
        << "  unknowns, topology-filtered 48/28 Cauchy stencils, validates\n"
        << "  fixed transfer routes, and executes the Neumann value-jump and\n"
        << "  Dirichlet normal-jump harmonic-jet GMRES formulations.\n"
        << "  KFBIM_3D_CAUCHY_POLICY selects g1_nearest (default),\n"
        << "  same_patch, topological_nearest, or balanced_patches.\n"
        << "  KFBIM_3D_CAUCHY_VALUE_COUNT and\n"
        << "  KFBIM_3D_CAUCHY_NORMAL_COUNT select positive stencil counts\n"
        << "  (defaults: 48 and 28; normal count may not exceed value count).\n"
        << "  KFBIM_3D_GMRES_MAX_ITERATIONS selects a positive GMRES cap\n"
        << "  for both formulations (default: 80).\n"
        << "  KFBIM_3D_RESTRICT_OWNER_MODE selects "
           "full_intersection_reference\n"
        << "  (default), optimized_intersection, or "
           "region_closest_hybrid.\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        std::cout << std::scientific << std::setprecision(6) << std::unitbuf;
        const bool rigid_study = argc >= 2
                              && std::string(argv[1]) == "--rigid-study";
        const bool restrict_probe = argc >= 2
                                  && std::string(argv[1]) == "--restrict-probe";
        const bool restrict_probe_owner = argc >= 2
            && std::string(argv[1]) == "--restrict-probe-owner";
        const bool restrict_profile_owner = argc >= 2
            && std::string(argv[1]) == "--restrict-profile-owner";
        const bool owner_preprocess_study = argc >= 2
            && std::string(argv[1]) == "--owner-preprocess-study";
        const bool neumann_owner_study = argc >= 2
            && std::string(argv[1]) == "--neumann-owner-study";
        const bool neumann_rigid_study = argc >= 2
            && std::string(argv[1]) == "--neumann-rigid-study";
        const bool neumann_edge_continuity_study = argc >= 2
            && std::string(argv[1]) == "--neumann-edge-continuity-study";
        const bool neumann_edge_cauchy_study = argc >= 2
            && std::string(argv[1]) == "--neumann-edge-cauchy-study";
        const bool force_neumann_edge_cauchy_extended =
            neumann_edge_cauchy_study && argc >= 3
            && std::string(argv[2]) == "--force-extended";
        const int first_level_argument =
            force_neumann_edge_cauchy_extended ? 3 : 2;
        std::string selection = "all";
        std::vector<int> levels = rigid_study
            ? std::vector<int>{32, 64, 128}
            : neumann_owner_study
                ? std::vector<int>{32, 64, 128}
            : neumann_rigid_study
                ? std::vector<int>{32, 64, 128}
            : (neumann_edge_continuity_study || neumann_edge_cauchy_study)
                ? std::vector<int>{32, 64}
            : owner_preprocess_study
                ? std::vector<int>{16, 32, 64}
            : restrict_profile_owner
                ? std::vector<int>{128}
                : (restrict_probe || restrict_probe_owner)
                ? std::vector<int>{32, 64}
                : std::vector<int>{32};
        if (argc >= 2 && !rigid_study && !restrict_probe
            && !restrict_probe_owner && !restrict_profile_owner
            && !owner_preprocess_study && !neumann_owner_study
            && !neumann_rigid_study && !neumann_edge_continuity_study
            && !neumann_edge_cauchy_study)
            selection = argv[1];
        if (selection == "--help" || selection == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (argc > first_level_argument) {
            levels.clear();
            for (int argument = first_level_argument; argument < argc; ++argument)
                levels.push_back(parse_grid_level_argument(argv[argument]));
        }
        if (restrict_probe || restrict_probe_owner
            || restrict_profile_owner) {
            return run_normal_restrict_causal_probe(levels,
                restrict_probe_owner || restrict_profile_owner,
                restrict_profile_owner);
        }
        if (owner_preprocess_study)
            return run_owner_preprocess_study_3d(levels);
        if (neumann_owner_study)
            return run_neumann_owner_study_3d(levels);
        if (neumann_rigid_study)
            return run_neumann_rigid_study_3d(levels);
        if (neumann_edge_continuity_study)
            return run_neumann_edge_continuity_study_3d(levels);
        if (neumann_edge_cauchy_study)
            return run_neumann_edge_cauchy_study_3d(
                levels, force_neumann_edge_cauchy_extended);
        const app3d::LegacySurfaceCauchyPolicy3D cauchy_policy = selected_cauchy_policy();
        const int cauchy_value_count = positive_environment_integer(
            "KFBIM_3D_CAUCHY_VALUE_COUNT", kCauchyValueNeighborCount);
        const int cauchy_normal_count = positive_environment_integer(
            "KFBIM_3D_CAUCHY_NORMAL_COUNT", kCauchyDerivativeNeighborCount);
        const int gmres_max_iterations = positive_environment_integer(
            "KFBIM_3D_GMRES_MAX_ITERATIONS", 80);
        if (cauchy_normal_count > cauchy_value_count) {
            throw std::invalid_argument(
                "KFBIM_3D_CAUCHY_NORMAL_COUNT may not exceed "
                "KFBIM_3D_CAUCHY_VALUE_COUNT");
        }
        if (rigid_study) {
            return run_dirichlet_rigid_study(
                levels,
                cauchy_policy,
                cauchy_value_count,
                cauchy_normal_count,
                gmres_max_iterations);
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
        std::filesystem::path output_dir =
            std::filesystem::path(KFBIM_APP_OUTPUT_DIR)
            / "neumann_exterior_zero_trace_3d";
#else
        std::filesystem::path output_dir =
            "output/neumann_exterior_zero_trace_3d";
#endif
        if (cauchy_policy != app3d::LegacySurfaceCauchyPolicy3D::TopologicalNearest)
            output_dir /= cauchy_policy_name(cauchy_policy);
        if (cauchy_value_count != kCauchyValueNeighborCount
            || cauchy_normal_count != kCauchyDerivativeNeighborCount) {
            output_dir /= "v" + std::to_string(cauchy_value_count) + "_n"
                          + std::to_string(cauchy_normal_count);
        }
        output_dir.make_preferred();

        std::cout << "KFBI3D harmonic-jet convergence study\n"
                  << "  Neumann target: exterior value trace = 0\n"
                  << "  Dirichlet target: exterior normal trace = 0\n"
                  << "  current stage: native NURBS surface DOFs + "
                     "selected Cauchy neighborhoods + G1 parameter-owned routes\n"
                  << "  cauchy_policy=" << cauchy_policy_name(cauchy_policy)
                  << '\n'
                  << "  cauchy_degree=" << kCauchyPolynomialDegree
                  << " restrict_grid_degree=" << kRestrictGridDegree
                  << " restrict_normal_degree=" << kRestrictNormalDegree
                  << '\n'
                  << "  cauchy_counts=" << cauchy_value_count << '/'
                  << cauchy_normal_count << '\n'
                  << "  gmres_max_iterations=" << gmres_max_iterations << '\n'
                  << "  levels=";
        for (std::size_t index = 0; index < levels.size(); ++index) {
            if (index != 0)
                std::cout << ',';
            std::cout << levels[index];
        }
        std::cout << '\n';

        std::vector<ReadinessResult> results;
        const app3d::RigidTransform3D identity_transform;
        for (int N : levels) {
            for (GeometryKind geometry : geometries) {
                results.push_back(run_readiness_case(
                    geometry,
                    N,
                    output_dir,
                    cauchy_policy,
                    cauchy_value_count,
                    cauchy_normal_count,
                    gmres_max_iterations,
                    identity_transform,
                    SolveSelection3D::Both));
                write_summary(output_dir, results);
                write_solve_summaries(output_dir, results);

                const ReadinessResult& result = results.back();
                const SolveMetrics3D* failed_solve = nullptr;
                if (!result.neumann.converged)
                    failed_solve = &result.neumann;
                else if (!result.dirichlet_normal.converged)
                    failed_solve = &result.dirichlet_normal;
                if (failed_solve != nullptr) {
                    std::ostringstream message;
                    message
                        << "3D GMRES did not converge: geometry="
                        << result.geometry
                        << " N=" << result.N
                        << " formulation=" << failed_solve->formulation
                        << " iterations=" << failed_solve->iterations
                        << " final_relative_residual="
                        << std::scientific << std::setprecision(17)
                        << failed_solve->gmres_relative_residual;
                    throw std::runtime_error(message.str());
                }
            }
        }

        std::cout << "Geometry checks and both GMRES formulations completed.\n"
                  << "Output: " << output_dir.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
