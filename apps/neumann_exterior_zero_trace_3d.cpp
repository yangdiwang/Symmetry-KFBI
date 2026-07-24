#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "crossing_owner_restrict_3d.hpp"
#include "dirichlet_rigid_transform_study_3d.hpp"
#include "exterior_only_cubic_normal_restrict_3d.hpp"
#include "harmonic_polynomial_space_3d.hpp"
#include "native_nurbs_surface_3d.hpp"
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

enum class CauchyStencilPolicy3D {
    G1Nearest,
    TopologicalNearest,
    SamePatch,
    BalancedPatches
};

enum class SolveSelection3D {
    Both,
    DirichletNormalOnly
};

enum class ExteriorNormalRestrictMode3D {
    JointTricubicCauchy,
    JointTricubicCrossingOwner,
    ExteriorOnlyHarmonicCubic
};

std::string cauchy_policy_name(CauchyStencilPolicy3D policy)
{
    switch (policy) {
    case CauchyStencilPolicy3D::G1Nearest:
        return "g1_nearest";
    case CauchyStencilPolicy3D::TopologicalNearest:
        return "topological_nearest";
    case CauchyStencilPolicy3D::SamePatch:
        return "same_patch";
    case CauchyStencilPolicy3D::BalancedPatches:
        return "balanced_patches";
    }
    throw std::runtime_error("unknown 3D Cauchy stencil policy");
}

CauchyStencilPolicy3D selected_cauchy_policy()
{
    const char* raw = std::getenv("KFBIM_3D_CAUCHY_POLICY");
    if (raw == nullptr || std::string(raw).empty()
        || std::string(raw) == "g1_nearest") {
        return CauchyStencilPolicy3D::G1Nearest;
    }
    if (std::string(raw) == "same_patch")
        return CauchyStencilPolicy3D::SamePatch;
    if (std::string(raw) == "topological_nearest")
        return CauchyStencilPolicy3D::TopologicalNearest;
    if (std::string(raw) == "balanced_patches")
        return CauchyStencilPolicy3D::BalancedPatches;
    throw std::invalid_argument(
        "KFBIM_3D_CAUCHY_POLICY must be g1_nearest, same_patch, "
        "topological_nearest, or balanced_patches");
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

struct CauchyStencil {
    std::vector<int> value_ids;
    std::vector<int> derivative_ids;
    double radius_over_h = 0.0;
    int incident_patch_count = 0;
    int value_patch_imbalance = 0;
    int derivative_patch_imbalance = 0;
};

struct CauchyStencilSet {
    CauchyStencilPolicy3D policy = CauchyStencilPolicy3D::G1Nearest;
    int value_count = 0;
    int derivative_count = 0;
    int value_count_min = 0;
    int value_count_max = 0;
    int derivative_count_min = 0;
    int derivative_count_max = 0;
    std::vector<CauchyStencil> rows;
    double radius_max_over_h = 0.0;
    double radius_mean_over_h = 0.0;
    int incident_patch_count_min = 0;
    int incident_patch_count_max = 0;
    int value_patch_imbalance_max = 0;
    int derivative_patch_imbalance_max = 0;
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

std::vector<int> select_cauchy_dofs(CauchyStencilPolicy3D policy,
                                    const NativeNurbsSurface3D& surface,
                                    const SurfaceDofCloud& cloud,
                                    int center,
                                    int count)
{
    switch (policy) {
    case CauchyStencilPolicy3D::G1Nearest:
        return app3d::nearest_g1_cauchy_dofs(
            surface, cloud, center, count);
    case CauchyStencilPolicy3D::TopologicalNearest:
        return app3d::nearest_topological_cauchy_dofs(
            surface, cloud, center, count);
    case CauchyStencilPolicy3D::SamePatch:
        return app3d::nearest_same_patch_cauchy_dofs(cloud, center, count);
    case CauchyStencilPolicy3D::BalancedPatches:
        return app3d::balanced_topological_cauchy_dofs(
            surface, cloud, center, count);
    }
    throw std::runtime_error("unknown Cauchy stencil selection policy");
}

int selected_patch_imbalance(const SurfaceDofCloud& cloud,
                             const std::vector<int>& ids)
{
    std::map<int, int> counts;
    for (int q : ids)
        ++counts[cloud.dofs[static_cast<std::size_t>(q)].patch_id];
    if (counts.empty())
        return 0;
    int minimum = std::numeric_limits<int>::max();
    int maximum = 0;
    for (const auto& item : counts) {
        minimum = std::min(minimum, item.second);
        maximum = std::max(maximum, item.second);
    }
    return maximum - minimum;
}

CauchyStencilSet build_cauchy_stencils(const NativeNurbsSurface3D& surface,
                                       const SurfaceDofCloud& cloud,
                                       double h,
                                       int requested_value_count,
                                       int requested_derivative_count,
                                       CauchyStencilPolicy3D policy)
{
    if (requested_value_count <= 0 || requested_derivative_count < 0
        || requested_derivative_count > requested_value_count) {
        throw std::invalid_argument("invalid Cauchy stencil sizes");
    }

    CauchyStencilSet result;
    result.policy = policy;
    result.value_count = requested_value_count;
    result.derivative_count = requested_derivative_count;
    result.value_count_min = std::numeric_limits<int>::max();
    result.derivative_count_min = std::numeric_limits<int>::max();
    result.rows.reserve(cloud.dofs.size());
    result.incident_patch_count_min = std::numeric_limits<int>::max();

    for (int center = 0; center < static_cast<int>(cloud.dofs.size()); ++center) {
        const SurfaceDof& target = cloud.dofs[static_cast<std::size_t>(center)];
        CauchyStencil stencil;
        stencil.value_ids = select_cauchy_dofs(
            policy, surface, cloud, center, result.value_count);
        stencil.derivative_ids = select_cauchy_dofs(
            policy, surface, cloud, center, result.derivative_count);
        if (std::find(stencil.value_ids.begin(), stencil.value_ids.end(), center)
            == stencil.value_ids.end()) {
            throw std::runtime_error("Cauchy value stencil does not contain its center");
        }
        if (std::find(stencil.derivative_ids.begin(),
                      stencil.derivative_ids.end(), center)
            == stencil.derivative_ids.end()) {
            throw std::runtime_error(
                "Cauchy normal stencil does not contain its center");
        }
        double radius_sq = 0.0;
        for (int q : stencil.value_ids) {
            radius_sq = std::max(
                radius_sq,
                (cloud.dofs[static_cast<std::size_t>(q)].point - target.point)
                    .squaredNorm());
        }
        for (int q : stencil.derivative_ids) {
            radius_sq = std::max(
                radius_sq,
                (cloud.dofs[static_cast<std::size_t>(q)].point - target.point)
                    .squaredNorm());
        }
        stencil.radius_over_h = std::sqrt(radius_sq) / h;
        std::set<int> selected_patches;
        for (int q : stencil.value_ids)
            selected_patches.insert(cloud.dofs[static_cast<std::size_t>(q)].patch_id);
        for (int q : stencil.derivative_ids)
            selected_patches.insert(cloud.dofs[static_cast<std::size_t>(q)].patch_id);
        stencil.incident_patch_count = static_cast<int>(selected_patches.size());
        stencil.value_patch_imbalance =
            selected_patch_imbalance(cloud, stencil.value_ids);
        stencil.derivative_patch_imbalance =
            selected_patch_imbalance(cloud, stencil.derivative_ids);

        result.radius_max_over_h = std::max(
            result.radius_max_over_h, stencil.radius_over_h);
        result.radius_mean_over_h += stencil.radius_over_h;
        result.incident_patch_count_min = std::min(
            result.incident_patch_count_min, stencil.incident_patch_count);
        result.incident_patch_count_max = std::max(
            result.incident_patch_count_max, stencil.incident_patch_count);
        result.value_count_min = std::min(
            result.value_count_min, static_cast<int>(stencil.value_ids.size()));
        result.value_count_max = std::max(
            result.value_count_max, static_cast<int>(stencil.value_ids.size()));
        result.derivative_count_min = std::min(
            result.derivative_count_min,
            static_cast<int>(stencil.derivative_ids.size()));
        result.derivative_count_max = std::max(
            result.derivative_count_max,
            static_cast<int>(stencil.derivative_ids.size()));
        result.value_patch_imbalance_max = std::max(
            result.value_patch_imbalance_max, stencil.value_patch_imbalance);
        result.derivative_patch_imbalance_max = std::max(
            result.derivative_patch_imbalance_max,
            stencil.derivative_patch_imbalance);
        result.rows.push_back(std::move(stencil));
    }
    result.radius_mean_over_h /= static_cast<double>(result.rows.size());
    return result;
}

using app3d::HarmonicPolynomialSpace3D;
using app3d::svd_pseudoinverse_3d;

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
    std::vector<double> condition_values() const
    {
        std::vector<double> result;
        result.reserve(maps_.size());
        for (const CauchyFitMap3D& map : maps_)
            result.push_back(map.condition);
        return result;
    }
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
            const int value_count = static_cast<int>(stencil.value_ids.size());
            const int normal_count =
                static_cast<int>(stencil.derivative_ids.size());
            Eigen::VectorXd values(value_count);
            Eigen::VectorXd normals(normal_count);
            for (int k = 0; k < value_count; ++k)
                values[k] = value_jump[stencil.value_ids[static_cast<std::size_t>(k)]];
            for (int k = 0; k < normal_count; ++k)
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
        const int value_count = static_cast<int>(stencil.value_ids.size());
        const int normal_count = static_cast<int>(stencil.derivative_ids.size());
        const int rows = value_count + normal_count;
        if (rows < dimension()) {
            throw std::runtime_error(
                "Cauchy fit has fewer conditions than cubic coefficients");
        }
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
        if (singular.size() != dimension() || singular.size() == 0
            || !singular.allFinite() || !(singular[0] > 0.0)
            || !(singular[singular.size() - 1] > 3.0e-12 * singular[0])) {
            throw std::runtime_error(
                "Cauchy fit is rank deficient at surface DOF "
                + std::to_string(center_id));
        }
        CauchyFitMap3D result;
        result.condition = singular[0] / singular[singular.size() - 1];
        const Eigen::MatrixXd pinv = svd_pseudoinverse_3d(weighted, 3.0e-12);
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

struct HarmonicTraceCorrectionTerm3D {
    int owner_dof = -1;
    Eigen::VectorXd evaluation;
};

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
    PanelCenterHarmonicJetKFBI3D(const CartesianGrid3D& grid,
                                 const GridPair3D& grid_pair,
                                 const NativeNurbsSurface3D& native_surface,
                                 const std::vector<geometry3d::NurbsParamTriangle3D>&
                                     correction_triangles,
                                 const std::vector<geometry3d::NurbsParamTriangle3D>&
                                     geometry_triangles,
                                 const SurfaceDofCloud& cloud,
                                 const CauchyStencilSet& stencils,
                                 bool build_exterior_only_restrict = false,
                                 bool build_crossing_owner_restrict = false)
        : grid_(grid)
        , grid_pair_(grid_pair)
        , native_surface_(native_surface)
        , correction_triangles_(correction_triangles)
        , geometry_triangles_(geometry_triangles)
        , cloud_(cloud)
        , h_(grid.spacing()[0])
        , fit_(cloud, stencils, h_, kCauchyPolynomialDegree)
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
        build_crossing_rows();
        if (build_crossing_owner_restrict) {
            geometry3d::NurbsSurfaceIntersectorOptions3D options;
            options.maximum_element_extent = 2.0 * h_;
            options.local_max_subdivision_depth = 4;
            const geometry3d::NurbsSurfaceIntersector3D intersector(
                native_surface_.geometry_model(), options);
            build_trace_templates(&intersector);
            crossing_owner_templates_built_ = true;
        } else {
            build_trace_templates(nullptr);
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

    std::size_t restrict_owner_geometry_query_count() const
    {
        std::size_t result = 0;
        for (const HarmonicTraceSample3D& sample : trace_samples_)
            result += sample.owner_geometry_query_count;
        return result;
    }

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
        return {potential, fit_.coefficients(value_jump, normal_jump)};
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
        return recover_trace(
            continued_samples(field, value_jump, normal_jump, false, mode),
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
                                      ExteriorNormalRestrictMode3D mode =
                                          ExteriorNormalRestrictMode3D::
                                              JointTricubicCauchy) const
    {
        const bool use_crossing_owner =
            mode == ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner;
        if (use_crossing_owner && !crossing_owner_templates_built_) {
            throw std::runtime_error(
                "crossing-owner normal restrict was not initialized");
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
                    if (use_crossing_owner) {
                        for (const HarmonicTraceCorrectionTerm3D& term
                             : sample.owner_corrections) {
                            value += term.evaluation.dot(
                                field.coefficients.row(
                                    term.owner_dof).transpose());
                        }
                    } else {
                        value += sample.legacy_correction_evaluation.dot(
                            field.coefficients.row(center).transpose());
                    }
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

    HarmonicTraceSample3D build_trace_sample(int center,
                                             bool desired_inside,
                                             double signed_layer,
                                             const geometry3d::
                                                 NurbsSurfaceIntersector3D*
                                                 restrict_intersector) const
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
        result.legacy_correction_evaluation =
            Eigen::VectorXd::Zero(fit_.dimension());
        std::map<int, Eigen::VectorXd> owner_evaluations;
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
                        ++result.wrong_side_node_count;
                        const double absolute_weight = std::abs(weight);
                        result.wrong_side_sum_abs_weight += absolute_weight;
                        const Eigen::Vector3d node_point =
                            grid_point(grid_, node);
                        const Eigen::Vector3d target_xi =
                            local_coordinate(center, node_point);
                        const double correction_sign = desired_inside ? 1.0 : -1.0;
                        result.legacy_correction_evaluation +=
                            correction_sign * weight
                            * fit_.space().basis(
                                target_xi.x(), target_xi.y(), target_xi.z());

                        if (restrict_intersector != nullptr) {
                            ++result.owner_geometry_query_count;
                            int owner = center;
                            try {
                                const geometry3d::
                                    NurbsSurfaceIntersectionResult3D intersection =
                                        restrict_intersector->intersect_segment(
                                            query, node_point);
                                const app3d::RestrictOwnerDecision3D decision =
                                    app3d::select_restrict_correction_owner_3d(
                                        center, query, node_point,
                                        native_surface_, cloud_, intersection);
                                owner = decision.owner_dof;
                                const std::size_t kind =
                                    static_cast<std::size_t>(decision.kind);
                                if (kind
                                    >= result.owner_decision_counts.size()) {
                                    throw std::logic_error(
                                        "crossing-owner decision kind is invalid");
                                }
                                ++result.owner_decision_counts[kind];
                                result.owner_decision_sum_abs_weights[kind] +=
                                    absolute_weight;
                                if (decision.kind == app3d::
                                        RestrictOwnerDecisionKind3D::
                                            ForeignNonG1SingleCrossing) {
                                    if (intersection.crossings.size() != 1) {
                                        throw std::logic_error(
                                            "foreign crossing-owner decision "
                                            "does not have one crossing");
                                    }
                                    const geometry3d::NurbsSurfaceCrossing3D&
                                        root = intersection.crossings.front();
                                    HarmonicTraceOwnerAuditTerm3D audit;
                                    audit.grid_node = node;
                                    audit.interpolation_weight = weight;
                                    audit.owner_dof = owner;
                                    audit.crossing_patch = root.patch_index;
                                    audit.crossing_u = root.u;
                                    audit.crossing_v = root.v;
                                    audit.segment_parameter =
                                        root.edge_parameter;
                                    audit.residual = root.residual;
                                    audit.transversality =
                                        root.transversality;
                                    result.owner_reroute_terms.push_back(
                                        std::move(audit));
                                }
                            } catch (const geometry3d::
                                         UnresolvedNurbsIntersectionCandidate3D&) {
                                ++result.owner_unresolved_fallback_count;
                                result.
                                    owner_unresolved_fallback_sum_abs_weight +=
                                        absolute_weight;
                            } catch (const std::runtime_error& error) {
                                if (std::string(error.what())
                                    != "coincident roots on unrelated NURBS patches") {
                                    throw;
                                }
                                ++result.
                                    owner_unrelated_coincidence_fallback_count;
                                result.
                                    owner_unrelated_coincidence_fallback_sum_abs_weight +=
                                        absolute_weight;
                            }

                            const Eigen::Vector3d owner_xi =
                                local_coordinate(owner, node_point);
                            Eigen::VectorXd& owner_evaluation =
                                owner_evaluations[owner];
                            if (owner_evaluation.size() == 0) {
                                owner_evaluation =
                                    Eigen::VectorXd::Zero(fit_.dimension());
                            }
                            owner_evaluation += correction_sign * weight
                                * fit_.space().basis(
                                    owner_xi.x(), owner_xi.y(), owner_xi.z());
                        }
                    }
                    ++q;
                }
            }
        }
        result.owner_corrections.reserve(owner_evaluations.size());
        for (auto& owner_evaluation : owner_evaluations) {
            HarmonicTraceCorrectionTerm3D term;
            term.owner_dof = owner_evaluation.first;
            term.evaluation = std::move(owner_evaluation.second);
            result.owner_corrections.push_back(std::move(term));
        }
        if (restrict_intersector != nullptr) {
            int classified_count =
                result.owner_unresolved_fallback_count
                + result.owner_unrelated_coincidence_fallback_count;
            for (int count : result.owner_decision_counts)
                classified_count += count;
            if (classified_count != result.wrong_side_node_count
                || result.owner_geometry_query_count
                       != static_cast<std::size_t>(
                           result.wrong_side_node_count)) {
                throw std::logic_error(
                    "crossing-owner restrict did not classify every "
                    "wrong-side support node");
            }
            const std::size_t foreign_kind = static_cast<std::size_t>(
                app3d::RestrictOwnerDecisionKind3D::
                    ForeignNonG1SingleCrossing);
            if (result.owner_reroute_terms.size()
                != static_cast<std::size_t>(
                    result.owner_decision_counts[foreign_kind])) {
                throw std::logic_error(
                    "crossing-owner reroute audit count is inconsistent");
            }
        }
        return result;
    }

    void build_trace_templates(
        const geometry3d::NurbsSurfaceIntersector3D* restrict_intersector)
    {
        trace_samples_.reserve(
            static_cast<std::size_t>(surface_size() * 2 * 4));
        for (int center = 0; center < surface_size(); ++center) {
            for (int side = 0; side < 2; ++side) {
                const bool desired_inside = side == 0;
                const double sign = desired_inside ? -1.0 : 1.0;
                for (double layer : normal_layers_)
                    trace_samples_.push_back(
                        build_trace_sample(
                            center, desired_inside, sign * layer,
                            restrict_intersector));
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
    bool crossing_owner_templates_built_ = false;
    double h_ = 0.0;
    PanelCenterCauchyFit3D fit_;
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
    result.coefficients = field.coefficients;
    Eigen::VectorXd applied;
    op.apply(augmented_unknown, applied);
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

SolveMetrics3D run_neumann_case(
    const CartesianGrid3D& grid,
    const GridPair3D& grid_pair,
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const app3d::RigidTransform3D& transform,
    int gmres_max_iterations)
{
    const int size = pipeline.surface_size();
    Eigen::VectorXd exact_trace(size);
    Eigen::VectorXd normal_data(size);
    for (int q = 0; q < size; ++q) {
        const SurfaceDof& dof =
            pipeline.surface().dofs[static_cast<std::size_t>(q)];
        exact_trace[q] = app3d::transformed_manufactured_harmonic_value_3d(
            transform, dof.point);
        normal_data[q] =
            app3d::transformed_manufactured_harmonic_gradient_3d(
                transform, dof.point).dot(dof.normal);
    }
    // Enforce the discrete compatibility condition.  The exact flux has zero
    // continuous mean; the tiny quadrature defect is otherwise amplified by
    // the first-kind exterior-trace equation.
    normal_data.array() -= surface_weighted_mean(
        pipeline.surface(), normal_data);

    const auto solve_start = std::chrono::steady_clock::now();
    const ExteriorZeroTraceSolution3D solution =
        solve_exterior_zero_trace_neumann_3d(
            pipeline, normal_data, 2.0e-10, 80, gmres_max_iterations);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();

    const HarmonicJetField3D field{
        solution.potential, solution.coefficients};
    const Eigen::VectorXd direct_exterior = pipeline.exterior_trace(
        field, solution.value_jump, normal_data);
    const Eigen::VectorXd exterior_from_jump = pipeline.interior_trace(
        field, solution.value_jump, normal_data) - solution.value_jump;

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

    exact_trace.array() -= surface_weighted_mean(
        pipeline.surface(), exact_trace);
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
                              const CauchyStencilSet& stencils)
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
                    << stencil.incident_patch_count << ','
                    << stencil.value_ids.size() << ','
                    << stencil.derivative_ids.size() << ','
                    << stencil.value_patch_imbalance << ','
                    << stencil.derivative_patch_imbalance;
        for (int id : stencil.value_ids)
            stencil_csv << ',' << id;
        for (int k = static_cast<int>(stencil.value_ids.size());
             k < stencils.value_count; ++k) {
            stencil_csv << ',';
        }
        for (int id : stencil.derivative_ids)
            stencil_csv << ',' << id;
        for (int k = static_cast<int>(stencil.derivative_ids.size());
             k < stencils.derivative_count; ++k) {
            stencil_csv << ',';
        }
        stencil_csv << '\n';
    }
}

ReadinessResult run_readiness_case(GeometryKind kind,
                                   int N,
                                   const std::filesystem::path& output_dir,
                                   CauchyStencilPolicy3D cauchy_policy,
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
    const CauchyStencilSet cauchy_stencils = build_cauchy_stencils(
        geometry.native_surface,
        surface_dofs,
        h,
        cauchy_value_count,
        cauchy_normal_count,
        cauchy_policy);
    if (solve_selection == SolveSelection3D::Both) {
        write_surface_files(output_dir, geometry, N);
        write_panel_center_files(
            output_dir, geometry.name, N, surface_dofs, cauchy_stencils);
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
    result.cauchy_value_neighbors = cauchy_stencils.value_count;
    result.cauchy_derivative_neighbors = cauchy_stencils.derivative_count;
    result.cauchy_value_neighbors_min = cauchy_stencils.value_count_min;
    result.cauchy_value_neighbors_max = cauchy_stencils.value_count_max;
    result.cauchy_derivative_neighbors_min =
        cauchy_stencils.derivative_count_min;
    result.cauchy_derivative_neighbors_max =
        cauchy_stencils.derivative_count_max;
    result.cauchy_radius_max_over_h = cauchy_stencils.radius_max_over_h;
    result.cauchy_radius_mean_over_h = cauchy_stencils.radius_mean_over_h;
    result.cauchy_incident_patches_min =
        cauchy_stencils.incident_patch_count_min;
    result.cauchy_incident_patches_max =
        cauchy_stencils.incident_patch_count_max;
    result.cauchy_value_patch_imbalance_max =
        cauchy_stencils.value_patch_imbalance_max;
    result.cauchy_derivative_patch_imbalance_max =
        cauchy_stencils.derivative_patch_imbalance_max;

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
        cauchy_stencils);
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
            gmres_max_iterations);
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
    CauchyStencilPolicy3D cauchy_policy,
    int cauchy_value_count,
    int cauchy_normal_count,
    int gmres_max_iterations)
{
    if (cauchy_policy != CauchyStencilPolicy3D::G1Nearest
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
};

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

int run_normal_restrict_causal_probe(std::vector<int> levels)
{
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    for (int N : levels) {
        if (N < 16 || !is_power_of_two(N))
            throw std::invalid_argument(
                "restrict-probe N must be a power of two and at least 16");
    }
#ifdef KFBIM_APP_OUTPUT_DIR
    const std::filesystem::path output_dir =
        std::filesystem::path(KFBIM_APP_OUTPUT_DIR)
        / "dirichlet_normal_restrict_causal_probe_3d";
#else
    const std::filesystem::path output_dir =
        "output/dirichlet_normal_restrict_causal_probe_3d";
#endif
    const std::array<std::string, 3> selected_ids{{
        "baseline", "rot_axis123_17deg",
        "rot_axis123_17deg_t_xyz_1"}};
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
    write_all_normal_restrict_probe_outputs(output_dir, results);
    write_restrict_owner_probe_outputs(output_dir, results);
    std::cout << "KFBI3D exterior-normal restrict causal probe\n"
              << "  geometry=l_prism cauchy=g1_nearest/degree3/48/28\n"
              << "  gmres_tolerance=2e-10 restart=0 cap=160\n";
    for (int N : levels) {
        const double h = kBoxSide / static_cast<double>(N);
        for (const auto& study_case : study_cases) {
            const auto setup_start = std::chrono::steady_clock::now();
            CartesianGrid3D grid({kBoxMin, kBoxMin, kBoxMin},
                                 {h, h, h}, {N, N, N}, DofLayout3D::Node);
            GeometryBundle geometry = make_geometry(
                GeometryKind::LPrism, h, study_case.transform);
            const auto domain = std::make_shared<const
                geometry3d::NurbsCartesianDomain3D>(
                    grid, geometry.native_surface.geometry_model());
            const SurfaceDofCloud surface_dofs =
                app3d::make_native_surface_dofs_3d(
                    geometry.native_surface, h);
            validate_surface_dofs(surface_dofs, h);
            const CauchyStencilSet cauchy_stencils = build_cauchy_stencils(
                geometry.native_surface, surface_dofs, h,
                kCauchyValueNeighborCount,
                kCauchyDerivativeNeighborCount,
                CauchyStencilPolicy3D::G1Nearest);
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
            const auto pipeline_start = std::chrono::steady_clock::now();
            PanelCenterHarmonicJetKFBI3D pipeline(
                grid, grid_pair, geometry.native_surface,
                geometry.correction_triangles,
                geometry.geometry_triangles,
                surface_dofs, cauchy_stencils, true, true);
            const double pipeline_setup_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - pipeline_start).count();

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
            probe_case.pipeline_setup_seconds = pipeline_setup_seconds;
            if (probe_case.owner_diagnostics.empty()
                || probe_case.owner_geometry_query_count == 0) {
                throw std::runtime_error(
                    "restrict-probe crossing-owner diagnostics are empty");
            }
            std::size_t classified_queries = 0;
            for (const RestrictOwnerSampleDiagnostics3D& diagnostic
                 : probe_case.owner_diagnostics) {
                classified_queries +=
                    static_cast<std::size_t>(diagnostic.wrong_side_count);
            }
            if (classified_queries != probe_case.owner_geometry_query_count)
                throw std::logic_error(
                    "crossing-owner probe query count is inconsistent");
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
            probe_case.setup_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - setup_start).count();
            results.push_back(std::move(probe_case));
            write_restrict_owner_probe_outputs(output_dir, results);
            NormalRestrictCaseProbe3D& stored = results.back();

            const std::array<ExteriorNormalRestrictMode3D, 3> modes{{
                ExteriorNormalRestrictMode3D::JointTricubicCauchy,
                ExteriorNormalRestrictMode3D::JointTricubicCrossingOwner,
                ExteriorNormalRestrictMode3D::ExteriorOnlyHarmonicCubic}};
            for (ExteriorNormalRestrictMode3D mode : modes) {
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
                route.physical = run_dirichlet_normal_case(
                    grid, grid_pair, pipeline, study_case.transform, 160,
                    mode, &route.physical_residuals);
                route.common = run_common_normal_restrict_gmres(
                    pipeline, mode, common_rhs);
                route.physical_rho_10_30 =
                    residual_contraction(route.physical_residuals, 10, 30);
                route.physical_worst_rho_5 =
                    worst_five_step_contraction(route.physical_residuals);
                route.common_rho_10_30 =
                    residual_contraction(route.common.residuals, 10, 30);
                route.common_worst_rho_5 =
                    worst_five_step_contraction(route.common.residuals);
                if (pipeline.restrict_owner_geometry_query_count()
                    != stored.owner_geometry_query_count) {
                    throw std::runtime_error(
                        "GMRES apply performed a crossing-owner geometry query");
                }
                route.seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - route_start).count();
                route.complete = true;
                write_all_normal_restrict_probe_outputs(output_dir, results);
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
        }
    }
    const bool supported = normal_restrict_hypothesis_supported(results, levels);
    std::cout << "[restrict-hypothesis] "
              << (supported ? "supported" : "not_proven") << '\n'
              << "Restrict probe output: " << output_dir.string() << '\n';
    return 0;
}

void print_usage(const char* executable)
{
    std::cout
        << "usage: " << executable
        << " [torus|cylinder|l_prism|all] [N ...]\n"
        << "       " << executable << " --rigid-study [N ...]\n"
        << "       " << executable << " --restrict-probe [N ...]\n"
        << "  Each N must be a power of two and at least 16 (default: 32).\n"
        << "  Rigid-study default levels: 32, 64, 128.\n"
        << "  Restrict-probe default levels: 32, 64.\n"
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
        << "  for both formulations (default: 80).\n";
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
        std::string selection = "all";
        std::vector<int> levels = rigid_study
            ? std::vector<int>{32, 64, 128}
            : restrict_probe ? std::vector<int>{32, 64}
                             : std::vector<int>{32};
        if (argc >= 2 && !rigid_study && !restrict_probe)
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
        if (restrict_probe)
            return run_normal_restrict_causal_probe(levels);
        const CauchyStencilPolicy3D cauchy_policy = selected_cauchy_policy();
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
        if (cauchy_policy != CauchyStencilPolicy3D::TopologicalNearest)
            output_dir /= cauchy_policy_name(cauchy_policy);
        if (cauchy_value_count != kCauchyValueNeighborCount
            || cauchy_normal_count != kCauchyDerivativeNeighborCount) {
            output_dir /= "v" + std::to_string(cauchy_value_count) + "_n"
                          + std::to_string(cauchy_normal_count);
        }

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
