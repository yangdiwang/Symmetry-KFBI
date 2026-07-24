#include "nurbs_geometry_preprocess_benchmark_3d.hpp"

#include "src/geometry/grid_pair_3d.hpp"
#include "src/geometry/nurbs_patch_triangulator_3d.hpp"
#include "src/grid/cartesian_grid_3d.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace kfbim::app3d::benchmark3d {
namespace {

constexpr double kBoxMin = -1.5;
constexpr double kBoxSide = 3.0;
constexpr double kTargetP2NodeSpacingOverH = 1.2;
constexpr std::size_t kMaximumMismatchRows = 1000;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

using Clock = std::chrono::steady_clock;
using geometry3d::NurbsCartesianDomain3D;
using geometry3d::NurbsCartesianDomainOptions3D;
using geometry3d::NurbsCartesianEdgeClassification3D;
using geometry3d::NurbsCartesianPreprocessStrategy3D;
using geometry3d::NurbsPatchTriangulation3D;
using geometry3d::NurbsSurfaceCrossing3D;

std::uint64_t checked_node_dimension(int N);

std::string format_double(double value)
{
    if (!std::isfinite(value))
        return {};
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(17) << value;
    return stream.str();
}

std::string format_vector(const Eigen::Vector3d& value)
{
    return "[" + format_double(value.x()) + ";"
        + format_double(value.y()) + ";"
        + format_double(value.z()) + "]";
}

std::string format_vector(const Eigen::Vector2d& value)
{
    return "[" + format_double(value.x()) + ";"
        + format_double(value.y()) + "]";
}

class FnvChecksum {
public:
    void add_unsigned(std::uint64_t value)
    {
        for (int byte = 0; byte < 8; ++byte) {
            state_ ^= value & 0xffULL;
            state_ *= kFnvPrime;
            value >>= 8;
        }
    }

    void add_signed(std::int64_t value)
    {
        add_unsigned(static_cast<std::uint64_t>(value));
    }

    void add_bool(bool value)
    {
        add_unsigned(value ? 1ULL : 0ULL);
    }

    void add_quantized(double value, double quantum)
    {
        if (std::isnan(value)) {
            add_signed(std::numeric_limits<std::int64_t>::min());
            return;
        }
        if (value == std::numeric_limits<double>::infinity()) {
            add_signed(std::numeric_limits<std::int64_t>::max());
            return;
        }
        if (value == -std::numeric_limits<double>::infinity()) {
            add_signed(std::numeric_limits<std::int64_t>::min() + 1);
            return;
        }
        const long double scaled =
            static_cast<long double>(value) / static_cast<long double>(quantum);
        const long double lower = static_cast<long double>(
            std::numeric_limits<std::int64_t>::min() + 2);
        const long double upper = static_cast<long double>(
            std::numeric_limits<std::int64_t>::max() - 1);
        if (scaled <= lower) {
            add_signed(std::numeric_limits<std::int64_t>::min() + 2);
        } else if (scaled >= upper) {
            add_signed(std::numeric_limits<std::int64_t>::max() - 1);
        } else {
            add_signed(static_cast<std::int64_t>(std::llround(scaled)));
        }
    }

    void add_vector(const Eigen::Vector3d& value, double quantum)
    {
        for (int coordinate = 0; coordinate < 3; ++coordinate)
            add_quantized(value[coordinate], quantum);
    }

    std::uint64_t value() const noexcept
    {
        return state_;
    }

private:
    std::uint64_t state_ = kFnvOffset;
};

NurbsPatchTriangulation3D make_production_triangulation(
    const NativeNurbsSurface3D& surface,
    double h)
{
    geometry3d::NurbsPatchTriangulatorOptions3D options;
    options.edge_buffer_factor = 0.5;
    options.edge_sample_step_factor = 1.0;
    options.min_edge_samples = 4;
    options.edge_length_quadrature_samples = 16;
    options.H_factor = 2.0 * kTargetP2NodeSpacingOverH;
    options.max_edge_over_H = 1.10;
    options.metric_ratio_tolerance = 1.50;
    options.max_depth = 6;
    return geometry3d::triangulate_nurbs_surface_patches_3d(
        surface.patches, h, options);
}

double production_grid_spacing(int N)
{
    (void)checked_node_dimension(N);
    return kBoxSide / static_cast<double>(N);
}

struct PreparedCase {
    GeometryKind3D geometry;
    int N;
    double h;
    NativeNurbsSurface3D surface;
    geometry3d::NurbsSurfaceModel3D model;
    CartesianGrid3D grid;
    NurbsPatchTriangulation3D triangulation;

    PreparedCase(GeometryKind3D selected_geometry, int selected_N)
        : geometry(selected_geometry),
          N(selected_N),
          h(production_grid_spacing(selected_N)),
          surface(make_native_nurbs_surface_3d(selected_geometry)),
          model(surface.geometry_model()),
          grid({kBoxMin, kBoxMin, kBoxMin},
               {h, h, h},
               {selected_N, selected_N, selected_N},
               DofLayout3D::Node),
          triangulation(make_production_triangulation(surface, h))
    {
        for (const auto& patch : surface.patches) {
            for (const auto& row : patch.control_net()) {
                for (const Eigen::Vector3d& point : row) {
                    if (!point.allFinite()
                        || !(point.array().minCoeff() > kBoxMin)
                        || !(point.array().maxCoeff() < kBoxMin + kBoxSide)) {
                        throw std::runtime_error(
                            "native NURBS control point is outside the fixed box");
                    }
                }
            }
        }
        if (!surface.exact_inside)
            throw std::runtime_error(
                "native NURBS benchmark geometry has no exact inside oracle");
    }
};

NurbsCartesianPreprocessStrategy3D strategy(Backend3D backend)
{
    switch (backend) {
    case Backend3D::Baseline:
        return NurbsCartesianPreprocessStrategy3D::CertifiedBaseline;
    case Backend3D::Pure:
        return NurbsCartesianPreprocessStrategy3D::OptimizedIntersection;
    case Backend3D::Hybrid:
        return NurbsCartesianPreprocessStrategy3D::Hybrid;
    }
    throw std::invalid_argument("invalid geometry preprocess backend");
}

struct BuiltBackend {
    std::shared_ptr<const NurbsCartesianDomain3D> domain;
    std::unique_ptr<GridPair3D> grid_pair;
    double domain_seconds = 0.0;
    double grid_pair_seconds = 0.0;
};

BuiltBackend build_backend(const PreparedCase& prepared, Backend3D backend)
{
    NurbsCartesianDomainOptions3D options;
    options.strategy = strategy(backend);
    const auto domain_start = Clock::now();
    auto domain = std::make_shared<const NurbsCartesianDomain3D>(
        prepared.grid, prepared.model, options);
    const double domain_seconds = std::chrono::duration<double>(
        Clock::now() - domain_start).count();

    const auto pair_start = Clock::now();
    auto grid_pair = std::make_unique<GridPair3D>(
        prepared.grid,
        prepared.triangulation.interface,
        prepared.triangulation.geometry_interface,
        domain);
    const double grid_pair_seconds = std::chrono::duration<double>(
        Clock::now() - pair_start).count();
    return {std::move(domain), std::move(grid_pair),
            domain_seconds, grid_pair_seconds};
}

void validate_production_grid(
    const PreparedCase& prepared,
    const NurbsCartesianDomain3D& domain)
{
    const auto dims = prepared.grid.dof_dims();
    const Eigen::Vector3d grid_min(kBoxMin, kBoxMin, kBoxMin);
    const Eigen::Vector3d grid_max(
        kBoxMin + static_cast<double>(dims[0] - 1) * prepared.h,
        kBoxMin + static_cast<double>(dims[1] - 1) * prepared.h,
        kBoxMin + static_cast<double>(dims[2] - 1) * prepared.h);
    const auto& bounds = domain.surface_bounds();
    const double minimum_margin = std::min(
        (bounds.lower - grid_min).minCoeff(),
        (grid_max - bounds.upper).minCoeff());
    if (minimum_margin / prepared.h < 2.0) {
        throw std::invalid_argument(
            "selected grid is rejected by the production geometry margin check");
    }
}

void record_mismatch(std::vector<std::string>& rows,
                     const PreparedCase& prepared,
                     Backend3D backend,
                     const std::string& category,
                     int axis,
                     int i,
                     int j,
                     int k,
                     const std::string& detail)
{
    if (rows.size() >= kMaximumMismatchRows)
        return;
    rows.push_back(encode_csv_row_3d({
        geometry_name_3d(prepared.geometry),
        std::to_string(prepared.N),
        backend_name_3d(backend),
        category,
        std::to_string(axis),
        std::to_string(i),
        std::to_string(j),
        std::to_string(k),
        detail}));
}

bool same_classification(
    const NurbsCartesianEdgeClassification3D& first,
    const NurbsCartesianEdgeClassification3D& second)
{
    return first.queried == second.queried
        && first.has_confirmed_interface == second.has_confirmed_interface
        && first.changes_component_membership
            == second.changes_component_membership
        && first.root_count_known == second.root_count_known
        && first.parity_known_from_roots == second.parity_known_from_roots
        && first.has_near_tangent_candidate
            == second.has_near_tangent_candidate
        && first.used_targeted_retry == second.used_targeted_retry
        && first.correction_safe == second.correction_safe
        && first.confirmed_crossing_count == second.confirmed_crossing_count
        && first.ambiguous_cluster_count == second.ambiguous_cluster_count
        && first.confirmed_transverse_count
            == second.confirmed_transverse_count;
}

void hash_classification(
    FnvChecksum& checksum,
    const NurbsCartesianEdgeClassification3D& value)
{
    checksum.add_bool(value.queried);
    checksum.add_bool(value.has_confirmed_interface);
    checksum.add_bool(value.changes_component_membership);
    checksum.add_bool(value.root_count_known);
    checksum.add_bool(value.parity_known_from_roots);
    checksum.add_bool(value.has_near_tangent_candidate);
    checksum.add_bool(value.used_targeted_retry);
    checksum.add_bool(value.correction_safe);
    checksum.add_unsigned(value.confirmed_crossing_count);
    checksum.add_unsigned(value.ambiguous_cluster_count);
    checksum.add_signed(value.confirmed_transverse_count);
}

struct CrossingTolerance {
    double physical = 0.0;
    double dimensionless = 0.0;
};

CrossingTolerance crossing_tolerance(
    const NurbsCartesianDomain3D& baseline,
    const NurbsCartesianDomain3D& candidate)
{
    const double physical = 8.0 * std::max(
        baseline.geometry_tolerance(), candidate.geometry_tolerance());
    const double dimensionless = std::max(
        64.0 * std::numeric_limits<double>::epsilon(),
        physical / std::max(
            baseline.surface_bounds().diameter(), physical));
    return {physical, dimensionless};
}

std::string tolerance_payload(const CrossingTolerance& value)
{
    return "{physical=" + format_double(value.physical)
        + ";dimensionless=" + format_double(value.dimensionless) + '}';
}

std::string tolerance_detail(const CrossingTolerance& value)
{
    return "tolerance=" + tolerance_payload(value);
}

std::string classification_error_detail(
    const NurbsCartesianEdgeClassification3D& baseline,
    const NurbsCartesianEdgeClassification3D& candidate)
{
    std::ostringstream stream;
    stream << "{queried=" << (baseline.queried != candidate.queried)
           << ";has_confirmed_interface="
           << (baseline.has_confirmed_interface
               != candidate.has_confirmed_interface)
           << ";changes_component_membership="
           << (baseline.changes_component_membership
               != candidate.changes_component_membership)
           << ";root_count_known="
           << (baseline.root_count_known != candidate.root_count_known)
           << ";parity_known_from_roots="
           << (baseline.parity_known_from_roots
               != candidate.parity_known_from_roots)
           << ";has_near_tangent_candidate="
           << (baseline.has_near_tangent_candidate
               != candidate.has_near_tangent_candidate)
           << ";used_targeted_retry="
           << (baseline.used_targeted_retry != candidate.used_targeted_retry)
           << ";correction_safe="
           << (baseline.correction_safe != candidate.correction_safe)
           << ";confirmed_crossing_count="
           << (baseline.confirmed_crossing_count
               != candidate.confirmed_crossing_count)
           << ";ambiguous_cluster_count="
           << (baseline.ambiguous_cluster_count
               != candidate.ambiguous_cluster_count)
           << ";confirmed_transverse_count="
           << (baseline.confirmed_transverse_count
               != candidate.confirmed_transverse_count) << '}';
    return stream.str();
}

std::string classification_comparison_detail(
    const NurbsCartesianEdgeClassification3D& baseline,
    const NurbsCartesianEdgeClassification3D& candidate,
    const CrossingTolerance& tolerance)
{
    return compose_mismatch_detail_3d(
        serialize_edge_classification_3d(baseline),
        serialize_edge_classification_3d(candidate),
        classification_error_detail(baseline, candidate),
        tolerance_payload(tolerance));
}

std::string comparison_number(double value)
{
    return std::isfinite(value) ? format_double(value) : "nonfinite";
}

std::string crossing_comparison_detail(
    const PreparedCase& prepared,
    const NurbsSurfaceCrossing3D& baseline,
    const NurbsSurfaceCrossing3D& candidate,
    double edge_length,
    const CrossingTolerance& tolerance)
{
    double uv_error = std::numeric_limits<double>::infinity();
    double baseline_normal_error = std::numeric_limits<double>::infinity();
    double candidate_normal_error = std::numeric_limits<double>::infinity();
    const int patch_count = static_cast<int>(prepared.surface.patches.size());
    if (baseline.patch_index >= 0 && baseline.patch_index < patch_count
        && candidate.patch_index >= 0 && candidate.patch_index < patch_count) {
        const auto& baseline_patch = prepared.surface.patches[
            static_cast<std::size_t>(baseline.patch_index)];
        const auto& candidate_patch = prepared.surface.patches[
            static_cast<std::size_t>(candidate.patch_index)];
        uv_error = (baseline_patch.evaluate(baseline.u, baseline.v)
            - candidate_patch.evaluate(candidate.u, candidate.v)).norm();
        baseline_normal_error = (baseline.normal
            - baseline_patch.normal(baseline.u, baseline.v)).norm();
        candidate_normal_error = (candidate.normal
            - candidate_patch.normal(candidate.u, candidate.v)).norm();
    }
    std::ostringstream errors;
    errors << "{patch_index="
           << (baseline.patch_index != candidate.patch_index)
           << ";component=" << (baseline.component != candidate.component)
           << ";uv_point=" << comparison_number(uv_error)
           << ";edge_parameter="
           << format_double(std::abs(
                baseline.edge_parameter - candidate.edge_parameter)
                * edge_length)
           << ";point=" << format_double((baseline.point - candidate.point).norm())
           << ";normal_delta="
           << format_double((baseline.normal - candidate.normal).norm())
           << ";baseline_normal_to_patch="
           << comparison_number(baseline_normal_error)
           << ";candidate_normal_to_patch="
           << comparison_number(candidate_normal_error)
           << ";residual_delta="
           << format_double(std::abs(baseline.residual - candidate.residual))
           << ";baseline_abs_residual=" << format_double(std::abs(baseline.residual))
           << ";candidate_abs_residual=" << format_double(std::abs(candidate.residual))
           << ";transversality="
           << format_double(std::abs(
                baseline.transversality - candidate.transversality))
           << ";feature_edge_contact="
           << (baseline.feature_edge_contact != candidate.feature_edge_contact)
           << ";reliability_tolerance=" << format_double(std::abs(
                baseline.reliable_transversality_tolerance
                - candidate.reliable_transversality_tolerance)) << '}';
    const std::string thresholds =
        "{physical=" + format_double(tolerance.physical)
        + ";dimensionless=" + format_double(tolerance.dimensionless)
        + ";uv_point=physical;edge_parameter=physical;point=physical"
        + ";normal_to_patch=dimensionless;residual_delta=physical"
        + ";absolute_residual=" + format_double(tolerance.physical / 8.0)
        + ";transversality=normal_delta+dimensionless"
        + ";reliability_tolerance=dimensionless}";
    return compose_mismatch_detail_3d(
        serialize_surface_crossing_3d(baseline),
        serialize_surface_crossing_3d(candidate), errors.str(), thresholds);
}

bool equivalent_crossing(
    const PreparedCase& prepared,
    const NurbsSurfaceCrossing3D& baseline,
    const NurbsSurfaceCrossing3D& candidate,
    double edge_length,
    const CrossingTolerance& tolerance,
    BackendMaterialization3D& result)
{
    bool equivalent = baseline.patch_index == candidate.patch_index
        && baseline.component == candidate.component
        && baseline.feature_edge_contact == candidate.feature_edge_contact;
    const int patch_count = static_cast<int>(prepared.surface.patches.size());
    if (baseline.patch_index < 0 || baseline.patch_index >= patch_count
        || candidate.patch_index < 0 || candidate.patch_index >= patch_count)
        return false;
    const auto& baseline_patch = prepared.surface.patches[
        static_cast<std::size_t>(baseline.patch_index)];
    const auto& candidate_patch = prepared.surface.patches[
        static_cast<std::size_t>(candidate.patch_index)];
    const Eigen::Vector3d baseline_uv_point =
        baseline_patch.evaluate(baseline.u, baseline.v);
    const Eigen::Vector3d candidate_uv_point =
        candidate_patch.evaluate(candidate.u, candidate.v);
    const double uv_error =
        (baseline_uv_point - candidate_uv_point).norm();
    const double parameter_error =
        std::abs(baseline.edge_parameter - candidate.edge_parameter)
        * edge_length;
    const double point_error = (baseline.point - candidate.point).norm();
    const double normal_error = (baseline.normal - candidate.normal).norm();
    const double residual_error =
        std::abs(baseline.residual - candidate.residual);
    const double transversality_error =
        std::abs(baseline.transversality - candidate.transversality);
    const double reliability_error = std::abs(
        baseline.reliable_transversality_tolerance
        - candidate.reliable_transversality_tolerance);
    result.maximum_uv_point_error =
        std::max(result.maximum_uv_point_error, uv_error);
    result.maximum_edge_parameter_error =
        std::max(result.maximum_edge_parameter_error, parameter_error);
    result.maximum_point_error =
        std::max(result.maximum_point_error, point_error);
    result.maximum_normal_error =
        std::max(result.maximum_normal_error, normal_error);
    result.maximum_residual_error =
        std::max(result.maximum_residual_error, residual_error);
    result.maximum_transversality_error =
        std::max(result.maximum_transversality_error, transversality_error);
    result.maximum_reliability_tolerance_error = std::max(
        result.maximum_reliability_tolerance_error, reliability_error);
    const Eigen::Vector3d baseline_patch_normal =
        baseline_patch.normal(baseline.u, baseline.v);
    const Eigen::Vector3d candidate_patch_normal =
        candidate_patch.normal(candidate.u, candidate.v);
    equivalent = equivalent
        && uv_error <= tolerance.physical
        && parameter_error <= tolerance.physical
        && point_error <= tolerance.physical
        && (baseline.normal - baseline_patch_normal).norm()
            <= tolerance.dimensionless
        && (candidate.normal - candidate_patch_normal).norm()
            <= tolerance.dimensionless
        && residual_error <= tolerance.physical
        && std::abs(baseline.residual) <= tolerance.physical / 8.0
        && std::abs(candidate.residual) <= tolerance.physical / 8.0
        && transversality_error
            <= normal_error + tolerance.dimensionless
        && reliability_error <= tolerance.dimensionless;
    return equivalent;
}

void hash_crossing(
    const PreparedCase& prepared,
    FnvChecksum& topology,
    FnvChecksum& materialization,
    const NurbsSurfaceCrossing3D& crossing,
    double edge_length,
    const CrossingTolerance& tolerance)
{
    topology.add_signed(crossing.patch_index);
    topology.add_signed(crossing.component);
    topology.add_bool(crossing.feature_edge_contact);
    materialization.add_signed(crossing.patch_index);
    materialization.add_signed(crossing.component);
    materialization.add_bool(crossing.feature_edge_contact);
    const int patch_count = static_cast<int>(prepared.surface.patches.size());
    if (crossing.patch_index >= 0 && crossing.patch_index < patch_count) {
        const Eigen::Vector3d uv_point = prepared.surface.patches[
            static_cast<std::size_t>(crossing.patch_index)]
                .evaluate(crossing.u, crossing.v);
        materialization.add_vector(uv_point, tolerance.physical);
    } else {
        materialization.add_quantized(
            crossing.u, tolerance.dimensionless);
        materialization.add_quantized(
            crossing.v, tolerance.dimensionless);
    }
    materialization.add_quantized(
        crossing.edge_parameter,
        tolerance.physical / std::max(edge_length, tolerance.physical));
    materialization.add_vector(crossing.point, tolerance.physical);
    materialization.add_vector(crossing.normal, tolerance.dimensionless);
    materialization.add_quantized(crossing.residual, tolerance.physical);
    materialization.add_quantized(
        crossing.transversality, tolerance.dimensionless);
    materialization.add_quantized(
        crossing.reliable_transversality_tolerance,
        tolerance.dimensionless);
}

void hash_owner(
    const PreparedCase& prepared,
    FnvChecksum& topology,
    FnvChecksum& materialization,
    const P2CrossingOwner3D& owner,
    double edge_length,
    const CrossingTolerance& tolerance)
{
    topology.add_signed(owner.center_index);
    topology.add_signed(owner.panel_index);
    topology.add_signed(owner.geometry_panel_index);
    topology.add_signed(owner.nurbs_patch_index);
    topology.add_signed(owner.surface_component);
    topology.add_signed(static_cast<int>(owner.status));
    materialization.add_signed(owner.center_index);
    materialization.add_signed(owner.panel_index);
    materialization.add_signed(owner.geometry_panel_index);
    materialization.add_signed(owner.nurbs_patch_index);
    materialization.add_signed(owner.surface_component);
    materialization.add_signed(static_cast<int>(owner.status));
    materialization.add_quantized(
        owner.edge_parameter,
        tolerance.physical / std::max(edge_length, tolerance.physical));
    materialization.add_vector(owner.barycentric, tolerance.dimensionless);
    materialization.add_vector(
        owner.geometry_barycentric, tolerance.dimensionless);
    const int patch_count = static_cast<int>(prepared.surface.patches.size());
    if (owner.nurbs_patch_index >= 0
        && owner.nurbs_patch_index < patch_count) {
        const Eigen::Vector3d uv_point = prepared.surface.patches[
            static_cast<std::size_t>(owner.nurbs_patch_index)]
                .evaluate(owner.nurbs_parameter.x(), owner.nurbs_parameter.y());
        materialization.add_vector(uv_point, tolerance.physical);
    } else {
        materialization.add_quantized(
            owner.nurbs_parameter.x(), tolerance.dimensionless);
        materialization.add_quantized(
            owner.nurbs_parameter.y(), tolerance.dimensionless);
    }
    materialization.add_vector(owner.crossing_point, tolerance.physical);
    materialization.add_vector(owner.crossing_normal, tolerance.dimensionless);
    materialization.add_quantized(
        owner.crossing_residual, tolerance.physical);
}

bool owner_matches_strict_crossing(
    const PreparedCase& prepared,
    const P2CrossingOwner3D& owner,
    const NurbsSurfaceCrossing3D& crossing,
    double edge_length,
    const CrossingTolerance& tolerance,
    BackendMaterialization3D& result)
{
    bool equivalent =
        owner.status == P2CrossingOwnerStatus3D::ExactIntersection
        && owner.nurbs_patch_index == crossing.patch_index
        && owner.surface_component == crossing.component;
    const int patch_count = static_cast<int>(prepared.surface.patches.size());
    if (owner.nurbs_patch_index < 0
        || owner.nurbs_patch_index >= patch_count
        || crossing.patch_index < 0 || crossing.patch_index >= patch_count)
        return false;
    const Eigen::Vector3d owner_uv_point = prepared.surface.patches[
        static_cast<std::size_t>(owner.nurbs_patch_index)]
            .evaluate(owner.nurbs_parameter.x(), owner.nurbs_parameter.y());
    const Eigen::Vector3d crossing_uv_point = prepared.surface.patches[
        static_cast<std::size_t>(crossing.patch_index)]
            .evaluate(crossing.u, crossing.v);
    const double uv_error = (owner_uv_point - crossing_uv_point).norm();
    const double parameter_error =
        std::abs(owner.edge_parameter - crossing.edge_parameter) * edge_length;
    const double point_error =
        (owner.crossing_point - crossing.point).norm();
    const double normal_error =
        (owner.crossing_normal - crossing.normal).norm();
    const double residual_error =
        std::abs(owner.crossing_residual - crossing.residual);
    result.maximum_owner_error = std::max(
        result.maximum_owner_error,
        std::max({uv_error, parameter_error, point_error,
                  normal_error, residual_error}));
    return equivalent
        && owner.nurbs_parameter.allFinite()
        && owner.crossing_point.allFinite()
        && owner.crossing_normal.allFinite()
        && uv_error <= tolerance.physical
        && parameter_error <= tolerance.physical
        && point_error <= tolerance.physical
        && normal_error <= tolerance.dimensionless
        && residual_error <= tolerance.physical;
}

bool equivalent_owner(
    const PreparedCase& prepared,
    const P2CrossingOwner3D& baseline,
    const P2CrossingOwner3D& candidate,
    double edge_length,
    const CrossingTolerance& tolerance,
    BackendMaterialization3D& result)
{
    bool equivalent = baseline.center_index == candidate.center_index
        && baseline.panel_index == candidate.panel_index
        && baseline.geometry_panel_index == candidate.geometry_panel_index
        && baseline.nurbs_patch_index == candidate.nurbs_patch_index
        && baseline.surface_component == candidate.surface_component
        && baseline.status == candidate.status;
    const double parameter_error =
        std::abs(baseline.edge_parameter - candidate.edge_parameter)
        * edge_length;
    const double barycentric_error =
        (baseline.barycentric - candidate.barycentric).norm();
    const double geometry_barycentric_error =
        (baseline.geometry_barycentric - candidate.geometry_barycentric).norm();
    const double point_error =
        (baseline.crossing_point - candidate.crossing_point).norm();
    const double normal_error =
        (baseline.crossing_normal - candidate.crossing_normal).norm();
    const double residual_error = std::abs(
        baseline.crossing_residual - candidate.crossing_residual);
    double uv_error = std::numeric_limits<double>::infinity();
    const int patch_count = static_cast<int>(prepared.surface.patches.size());
    if (baseline.nurbs_patch_index >= 0
        && baseline.nurbs_patch_index < patch_count
        && candidate.nurbs_patch_index >= 0
        && candidate.nurbs_patch_index < patch_count) {
        const Eigen::Vector3d baseline_uv = prepared.surface.patches[
            static_cast<std::size_t>(baseline.nurbs_patch_index)]
                .evaluate(baseline.nurbs_parameter.x(),
                          baseline.nurbs_parameter.y());
        const Eigen::Vector3d candidate_uv = prepared.surface.patches[
            static_cast<std::size_t>(candidate.nurbs_patch_index)]
                .evaluate(candidate.nurbs_parameter.x(),
                          candidate.nurbs_parameter.y());
        uv_error = (baseline_uv - candidate_uv).norm();
        const Eigen::Vector3d baseline_patch_normal = prepared.surface.patches[
            static_cast<std::size_t>(baseline.nurbs_patch_index)]
                .normal(baseline.nurbs_parameter.x(),
                        baseline.nurbs_parameter.y());
        const Eigen::Vector3d candidate_patch_normal = prepared.surface.patches[
            static_cast<std::size_t>(candidate.nurbs_patch_index)]
                .normal(candidate.nurbs_parameter.x(),
                        candidate.nurbs_parameter.y());
        equivalent = equivalent
            && (baseline.crossing_normal - baseline_patch_normal).norm()
                <= tolerance.dimensionless
            && (candidate.crossing_normal - candidate_patch_normal).norm()
                <= tolerance.dimensionless;
    }
    result.maximum_owner_error = std::max(
        result.maximum_owner_error,
        std::max({parameter_error, barycentric_error,
                  geometry_barycentric_error, point_error, normal_error,
                  residual_error, uv_error}));
    return equivalent
        && parameter_error <= tolerance.physical
        && barycentric_error <= tolerance.dimensionless
        && geometry_barycentric_error <= tolerance.dimensionless
        && uv_error <= tolerance.physical
        && point_error <= tolerance.physical
        && residual_error <= tolerance.physical;
}

std::string edge_state_detail(
    int label_a,
    int label_b,
    const NurbsCartesianEdgeClassification3D& classification)
{
    return "{label_a=" + std::to_string(label_a)
        + ";label_b=" + std::to_string(label_b)
        + ";classification="
        + serialize_edge_classification_3d(classification) + '}';
}

struct OwnerComparisonErrors {
    bool center_index = false;
    bool panel_index = false;
    bool geometry_panel_index = false;
    bool nurbs_patch_index = false;
    bool surface_component = false;
    bool status = false;
    double edge_parameter = 0.0;
    double barycentric = 0.0;
    double geometry_barycentric = 0.0;
    double uv_point = std::numeric_limits<double>::infinity();
    double crossing_point = 0.0;
    double baseline_normal_to_patch = std::numeric_limits<double>::infinity();
    double candidate_normal_to_patch = std::numeric_limits<double>::infinity();
    double crossing_residual = 0.0;
    bool strict_status = false;
    bool strict_patch_index = false;
    bool strict_component = false;
    bool strict_owner_nonfinite = false;
    double strict_uv_point = std::numeric_limits<double>::infinity();
    double strict_edge_parameter = 0.0;
    double strict_point = 0.0;
    double strict_normal_delta = 0.0;
    double strict_residual = 0.0;
};

OwnerComparisonErrors measure_owner_comparison(
    const PreparedCase& prepared,
    const P2CrossingOwner3D& baseline,
    const P2CrossingOwner3D& candidate,
    const NurbsSurfaceCrossing3D& strict_crossing,
    double edge_length)
{
    OwnerComparisonErrors value;
    value.center_index = baseline.center_index != candidate.center_index;
    value.panel_index = baseline.panel_index != candidate.panel_index;
    value.geometry_panel_index =
        baseline.geometry_panel_index != candidate.geometry_panel_index;
    value.nurbs_patch_index =
        baseline.nurbs_patch_index != candidate.nurbs_patch_index;
    value.surface_component =
        baseline.surface_component != candidate.surface_component;
    value.status = baseline.status != candidate.status;
    value.edge_parameter = std::abs(
        baseline.edge_parameter - candidate.edge_parameter) * edge_length;
    value.barycentric =
        (baseline.barycentric - candidate.barycentric).norm();
    value.geometry_barycentric =
        (baseline.geometry_barycentric
         - candidate.geometry_barycentric).norm();
    value.crossing_point =
        (baseline.crossing_point - candidate.crossing_point).norm();
    value.crossing_residual = std::abs(
        baseline.crossing_residual - candidate.crossing_residual);
    const int patch_count = static_cast<int>(prepared.surface.patches.size());
    if (baseline.nurbs_patch_index >= 0
        && baseline.nurbs_patch_index < patch_count
        && candidate.nurbs_patch_index >= 0
        && candidate.nurbs_patch_index < patch_count) {
        const auto& baseline_patch = prepared.surface.patches[
            static_cast<std::size_t>(baseline.nurbs_patch_index)];
        const auto& candidate_patch = prepared.surface.patches[
            static_cast<std::size_t>(candidate.nurbs_patch_index)];
        const Eigen::Vector3d baseline_uv = baseline_patch.evaluate(
            baseline.nurbs_parameter.x(), baseline.nurbs_parameter.y());
        const Eigen::Vector3d candidate_uv = candidate_patch.evaluate(
            candidate.nurbs_parameter.x(), candidate.nurbs_parameter.y());
        value.uv_point = (baseline_uv - candidate_uv).norm();
        value.baseline_normal_to_patch = (baseline.crossing_normal
            - baseline_patch.normal(baseline.nurbs_parameter.x(),
                                    baseline.nurbs_parameter.y())).norm();
        value.candidate_normal_to_patch = (candidate.crossing_normal
            - candidate_patch.normal(candidate.nurbs_parameter.x(),
                                     candidate.nurbs_parameter.y())).norm();
    }
    value.strict_status =
        candidate.status != P2CrossingOwnerStatus3D::ExactIntersection;
    value.strict_patch_index =
        candidate.nurbs_patch_index != strict_crossing.patch_index;
    value.strict_component =
        candidate.surface_component != strict_crossing.component;
    value.strict_owner_nonfinite = !(candidate.nurbs_parameter.allFinite()
        && candidate.crossing_point.allFinite()
        && candidate.crossing_normal.allFinite());
    value.strict_edge_parameter = std::abs(
        candidate.edge_parameter - strict_crossing.edge_parameter) * edge_length;
    value.strict_point =
        (candidate.crossing_point - strict_crossing.point).norm();
    value.strict_normal_delta =
        (candidate.crossing_normal - strict_crossing.normal).norm();
    value.strict_residual = std::abs(
        candidate.crossing_residual - strict_crossing.residual);
    if (candidate.nurbs_patch_index >= 0
        && candidate.nurbs_patch_index < patch_count
        && strict_crossing.patch_index >= 0
        && strict_crossing.patch_index < patch_count) {
        value.strict_uv_point = (prepared.surface.patches[
            static_cast<std::size_t>(candidate.nurbs_patch_index)].evaluate(
                candidate.nurbs_parameter.x(), candidate.nurbs_parameter.y())
            - prepared.surface.patches[
                static_cast<std::size_t>(strict_crossing.patch_index)].evaluate(
                    strict_crossing.u, strict_crossing.v)).norm();
    }
    return value;
}

std::string serialize_owner_comparison_errors(
    const OwnerComparisonErrors& value)
{
    std::ostringstream stream;
    stream << "{center_index=" << value.center_index
           << ";panel_index=" << value.panel_index
           << ";geometry_panel_index=" << value.geometry_panel_index
           << ";nurbs_patch_index=" << value.nurbs_patch_index
           << ";surface_component=" << value.surface_component
           << ";status=" << value.status
           << ";edge_parameter=" << format_double(value.edge_parameter)
           << ";barycentric=" << format_double(value.barycentric)
           << ";geometry_barycentric="
           << format_double(value.geometry_barycentric)
           << ";uv_point=" << comparison_number(value.uv_point)
           << ";crossing_point=" << format_double(value.crossing_point)
           << ";baseline_normal_to_patch="
           << comparison_number(value.baseline_normal_to_patch)
           << ";candidate_normal_to_patch="
           << comparison_number(value.candidate_normal_to_patch)
           << ";crossing_residual="
           << format_double(value.crossing_residual)
           << ";strict_status=" << value.strict_status
           << ";strict_patch_index=" << value.strict_patch_index
           << ";strict_component=" << value.strict_component
           << ";strict_owner_nonfinite=" << value.strict_owner_nonfinite
           << ";strict_uv_point="
           << comparison_number(value.strict_uv_point)
           << ";strict_edge_parameter="
           << format_double(value.strict_edge_parameter)
           << ";strict_point=" << format_double(value.strict_point)
           << ";strict_normal_delta="
           << format_double(value.strict_normal_delta)
           << ";strict_residual=" << format_double(value.strict_residual)
           << '}';
    return stream.str();
}

std::string owner_comparison_detail(
    const PreparedCase& prepared,
    const P2CrossingOwner3D& baseline,
    const P2CrossingOwner3D& candidate,
    const NurbsSurfaceCrossing3D& strict_crossing,
    double edge_length,
    const CrossingTolerance& tolerance)
{
    const OwnerComparisonErrors errors = measure_owner_comparison(
        prepared, baseline, candidate, strict_crossing, edge_length);
    const std::string thresholds =
        "{physical=" + format_double(tolerance.physical)
        + ";dimensionless=" + format_double(tolerance.dimensionless)
        + ";discrete=exact;edge_parameter=physical"
        + ";barycentric=dimensionless"
        + ";geometry_barycentric=dimensionless;uv_point=physical"
        + ";normal_to_patch=dimensionless;crossing_point=physical"
        + ";crossing_residual=physical;strict_normal_delta=dimensionless}";
    return compose_mismatch_detail_3d(
        serialize_crossing_owner_3d(baseline),
        serialize_crossing_owner_3d(candidate),
        serialize_owner_comparison_errors(errors), thresholds)
        + ";strict_crossing="
        + serialize_surface_crossing_3d(strict_crossing);
}

struct ScanContext {
    const PreparedCase& prepared;
    Backend3D backend;
    const NurbsCartesianDomain3D& candidate;
    const GridPair3D& candidate_pair;
    const NurbsCartesianDomain3D& baseline;
    const GridPair3D& baseline_pair;
    CrossingTolerance tolerance;
    BackendMaterialization3D result;
    FnvChecksum topology;
    FnvChecksum materialization;
    std::vector<std::string>& mismatches;

    ScanContext(const PreparedCase& prepared_case,
                Backend3D selected_backend,
                const NurbsCartesianDomain3D& candidate_domain,
                const GridPair3D& candidate_grid_pair,
                const NurbsCartesianDomain3D& baseline_domain,
                const GridPair3D& baseline_grid_pair,
                std::vector<std::string>& mismatch_rows)
        : prepared(prepared_case),
          backend(selected_backend),
          candidate(candidate_domain),
          candidate_pair(candidate_grid_pair),
          baseline(baseline_domain),
          baseline_pair(baseline_grid_pair),
          tolerance(crossing_tolerance(baseline_domain, candidate_domain)),
          mismatches(mismatch_rows)
    {
        result.backend = selected_backend;
        topology.add_signed(static_cast<int>(prepared.geometry));
        topology.add_signed(prepared.N);
        materialization.add_signed(static_cast<int>(prepared.geometry));
        materialization.add_signed(prepared.N);
    }
};

void scan_nodes(ScanContext& context)
{
    const auto& grid = context.prepared.grid;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        const int candidate_label = context.candidate.label(node);
        const int pair_label = context.candidate_pair.domain_label(node);
        const int baseline_label = context.baseline.label(node);
        const auto coordinate = grid.coord(node);
        const Eigen::Vector3d point(
            coordinate[0], coordinate[1], coordinate[2]);
        const bool exact = context.prepared.surface.exact_inside(point);
        ++context.result.node_count;
        context.topology.add_signed(node);
        context.topology.add_signed(candidate_label);
        context.topology.add_signed(pair_label);
        context.materialization.add_signed(node);
        context.materialization.add_signed(candidate_label);
        context.materialization.add_signed(pair_label);
        context.materialization.add_bool(exact);
        if ((candidate_label > 0) != exact) {
            ++context.result.analytic_label_mismatch_count;
            record_mismatch(
                context.mismatches, context.prepared, context.backend,
                "analytic_label", -1, node, -1, -1,
                compose_mismatch_detail_3d(
                    "{analytic_inside=" + std::to_string(exact ? 1 : 0) + '}',
                    "{domain_label=" + std::to_string(candidate_label) + '}',
                    "{membership_mismatch=1}", "{label=exact}"));
        }
        if (candidate_label != baseline_label) {
            ++context.result.baseline_label_mismatch_count;
            record_mismatch(
                context.mismatches, context.prepared, context.backend,
                "baseline_label", -1, node, -1, -1,
                compose_mismatch_detail_3d(
                    "{domain_label=" + std::to_string(baseline_label) + '}',
                    "{domain_label=" + std::to_string(candidate_label) + '}',
                    "{label_mismatch=1}", "{label=exact}"));
        }
        if (pair_label != candidate_label) {
            ++context.result.baseline_label_mismatch_count;
            record_mismatch(
                context.mismatches, context.prepared, context.backend,
                "grid_pair_label", -1, node, -1, -1,
                compose_mismatch_detail_3d(
                    "{domain_label=" + std::to_string(candidate_label) + '}',
                    "{grid_pair_label=" + std::to_string(pair_label) + '}',
                    "{label_mismatch=1}", "{label=exact}"));
        }
    }
}

void scan_crossings(ScanContext& context,
                    int axis,
                    int i,
                    int j,
                    int k,
                    int node,
                    int neighbor,
                    double edge_length)
{
    const auto candidate_crossings =
        context.candidate.crossings_between(node, neighbor);
    const auto baseline_crossings =
        context.baseline.crossings_between(node, neighbor);
    context.result.crossing_count += candidate_crossings.size();
    context.topology.add_unsigned(candidate_crossings.size());
    context.materialization.add_unsigned(candidate_crossings.size());
    for (const NurbsSurfaceCrossing3D& crossing : candidate_crossings) {
        hash_crossing(
            context.prepared, context.topology, context.materialization,
            crossing, edge_length, context.tolerance);
    }
    if (candidate_crossings.size() != baseline_crossings.size()) {
        ++context.result.crossing_count_mismatch_count;
        record_mismatch(
            context.mismatches, context.prepared, context.backend,
            "crossing_count", axis, i, j, k,
            "baseline={count=" + std::to_string(baseline_crossings.size())
                + "};candidate={count="
                + std::to_string(candidate_crossings.size())
                + "};errors={count_mismatch=1};"
                + tolerance_detail(context.tolerance) + ";count_tolerance=exact");
    }
    const std::size_t common =
        std::min(candidate_crossings.size(), baseline_crossings.size());
    for (std::size_t root = 0; root < common; ++root) {
        if (!equivalent_crossing(
                context.prepared,
                baseline_crossings[root], candidate_crossings[root],
                edge_length, context.tolerance, context.result)) {
            ++context.result.crossing_field_mismatch_count;
            record_mismatch(
                context.mismatches, context.prepared, context.backend,
                "crossing_fields", axis, i, j, k,
                "root=" + std::to_string(root) + ';'
                    + crossing_comparison_detail(
                        context.prepared, baseline_crossings[root],
                        candidate_crossings[root], edge_length,
                        context.tolerance));
        }
    }
    if (candidate_crossings.size() == 1) {
        const NurbsSurfaceCrossing3D& single =
            context.candidate.crossing_between(node, neighbor);
        if (!equivalent_crossing(
                context.prepared, candidate_crossings[0], single,
                edge_length, context.tolerance, context.result)) {
            ++context.result.crossing_field_mismatch_count;
            record_mismatch(
                context.mismatches, context.prepared, context.backend,
                "single_crossing_lookup", axis, i, j, k,
                "baseline_source=crossings_between;candidate_source="
                    "crossing_between;"
                    + crossing_comparison_detail(
                        context.prepared, candidate_crossings[0], single,
                        edge_length, context.tolerance));
        }
    }
}

void scan_owner(ScanContext& context,
                int axis,
                int i,
                int j,
                int k,
                int node,
                int neighbor,
                double edge_length,
                const NurbsSurfaceCrossing3D& candidate_crossing)
{
    ++context.result.correction_owner_count;
    const int candidate_label_a = context.candidate.label(node);
    const int candidate_label_b = context.candidate.label(neighbor);
    const int baseline_label_a = context.baseline.label(node);
    const int baseline_label_b = context.baseline.label(neighbor);
    const auto candidate_info =
        context.candidate.edge_classification_between(node, neighbor);
    const auto baseline_info =
        context.baseline.edge_classification_between(node, neighbor);
    const std::string candidate_state = edge_state_detail(
        candidate_label_a, candidate_label_b, candidate_info);
    const std::string baseline_state = edge_state_detail(
        baseline_label_a, baseline_label_b, baseline_info);
    P2CrossingOwner3D candidate_owner;
    P2CrossingOwner3D baseline_owner;
    bool candidate_owner_available = false;
    bool baseline_owner_available = false;
    try {
        candidate_owner =
            context.candidate_pair.p2_crossing_owner_between(node, neighbor);
        candidate_owner_available = true;
        hash_owner(
            context.prepared, context.topology, context.materialization,
            candidate_owner, edge_length, context.tolerance);
        bool owner_matches = owner_matches_strict_crossing(
            context.prepared, candidate_owner, candidate_crossing,
            edge_length, context.tolerance, context.result);
        if ((baseline_label_a > 0) == (baseline_label_b > 0)
            || !baseline_info.correction_safe) {
            owner_matches = false;
        } else {
            baseline_owner =
                context.baseline_pair.p2_crossing_owner_between(node, neighbor);
            baseline_owner_available = true;
            owner_matches = owner_matches
                && equivalent_owner(
                    context.prepared, baseline_owner, candidate_owner,
                    edge_length, context.tolerance, context.result);
        }
        if (!owner_matches) {
            ++context.result.correction_owner_mismatch_count;
            record_mismatch(
                context.mismatches, context.prepared, context.backend,
                "correction_owner", axis, i, j, k,
                "baseline_edge=" + baseline_state
                    + ";candidate_edge=" + candidate_state + ';'
                    + (baseline_owner_available
                        ? owner_comparison_detail(
                            context.prepared, baseline_owner, candidate_owner,
                            candidate_crossing, edge_length, context.tolerance)
                        : "baseline={owner=unavailable};candidate="
                            + serialize_crossing_owner_3d(candidate_owner)
                            + ";strict_crossing="
                            + serialize_surface_crossing_3d(candidate_crossing)
                            + ";errors={baseline_owner_unavailable=1};"
                            + tolerance_detail(context.tolerance)));
        }
    } catch (const std::exception& error) {
        ++context.result.correction_owner_mismatch_count;
        record_mismatch(
            context.mismatches, context.prepared, context.backend,
            "correction_owner_exception", axis, i, j, k,
            "baseline={edge=" + baseline_state + ";owner="
                + (baseline_owner_available
                    ? serialize_crossing_owner_3d(baseline_owner)
                    : "unavailable")
                + "};candidate={edge=" + candidate_state + ";owner="
                + (candidate_owner_available
                    ? serialize_crossing_owner_3d(candidate_owner)
                    : "unavailable")
                + "};errors={exception=" + error.what() + "};"
                + tolerance_detail(context.tolerance));
    }
}

void scan_label_changing_edge(
    ScanContext& context,
    const NurbsCartesianEdgeClassification3D& candidate_info,
    int axis,
    int i,
    int j,
    int k,
    int node,
    int neighbor,
    double edge_length)
{
    const int candidate_label_a = context.candidate.label(node);
    const int candidate_label_b = context.candidate.label(neighbor);
    if ((candidate_label_a > 0) == (candidate_label_b > 0))
        return;
    const int baseline_label_a = context.baseline.label(node);
    const int baseline_label_b = context.baseline.label(neighbor);
    const auto baseline_info =
        context.baseline.edge_classification_between(node, neighbor);
    const std::string baseline_state = edge_state_detail(
        baseline_label_a, baseline_label_b, baseline_info);
    const std::string candidate_state = edge_state_detail(
        candidate_label_a, candidate_label_b, candidate_info);
    ++context.result.label_changing_edge_count;
    if (!candidate_info.correction_safe) {
        ++context.result.unsafe_label_changing_edge_count;
        record_mismatch(
            context.mismatches, context.prepared, context.backend,
            "unsafe_label_change", axis, i, j, k,
            "baseline=" + baseline_state + ";candidate=" + candidate_state
                + ";errors={candidate_correction_unsafe=1};"
                + tolerance_detail(context.tolerance)
                + ";correction_safe_tolerance=exact_true");
        return;
    }
    try {
        const NurbsSurfaceCrossing3D& candidate_crossing =
            context.candidate.correction_crossing_between(node, neighbor);
        hash_crossing(
            context.prepared, context.topology, context.materialization,
            candidate_crossing, edge_length, context.tolerance);
        bool crossing_matches = false;
        const NurbsSurfaceCrossing3D* baseline_crossing_ptr = nullptr;
        if ((baseline_label_a > 0) != (baseline_label_b > 0)
            && baseline_info.correction_safe) {
            const NurbsSurfaceCrossing3D& baseline_crossing =
                context.baseline.correction_crossing_between(node, neighbor);
            baseline_crossing_ptr = &baseline_crossing;
            crossing_matches = equivalent_crossing(
                context.prepared, baseline_crossing, candidate_crossing,
                edge_length, context.tolerance, context.result);
        }
        if (!crossing_matches) {
            ++context.result.correction_crossing_mismatch_count;
            record_mismatch(
                context.mismatches, context.prepared, context.backend,
                "correction_crossing", axis, i, j, k,
                "baseline_edge=" + baseline_state
                    + ";candidate_edge=" + candidate_state + ';'
                    + (baseline_crossing_ptr != nullptr
                        ? crossing_comparison_detail(
                            context.prepared, *baseline_crossing_ptr,
                            candidate_crossing, edge_length,
                            context.tolerance)
                        : "baseline={crossing=unavailable};candidate="
                            + serialize_surface_crossing_3d(candidate_crossing)
                            + ";errors={baseline_correction_unavailable=1};"
                            + tolerance_detail(context.tolerance)));
        }
        scan_owner(
            context, axis, i, j, k, node, neighbor,
            edge_length, candidate_crossing);
    } catch (const std::exception& error) {
        ++context.result.correction_crossing_mismatch_count;
        ++context.result.correction_owner_mismatch_count;
        record_mismatch(
            context.mismatches, context.prepared, context.backend,
            "correction_crossing_exception", axis, i, j, k,
            "baseline=" + baseline_state + ";candidate=" + candidate_state
                + ";errors={exception=" + error.what() + "};"
                + tolerance_detail(context.tolerance));
    }
}

void scan_edge(ScanContext& context,
               int axis,
               int i,
               int j,
               int k,
               int node,
               int neighbor)
{
    const auto a_coord = context.prepared.grid.coord(node);
    const auto b_coord = context.prepared.grid.coord(neighbor);
    const Eigen::Vector3d edge(
        b_coord[0] - a_coord[0],
        b_coord[1] - a_coord[1],
        b_coord[2] - a_coord[2]);
    const double edge_length = edge.norm();
    const bool candidate_barrier =
        context.candidate.has_barrier_between(node, neighbor);
    const bool candidate_interface =
        context.candidate.has_interface_between(node, neighbor);
    const bool baseline_barrier =
        context.baseline.has_barrier_between(node, neighbor);
    const bool baseline_interface =
        context.baseline.has_interface_between(node, neighbor);
    const auto candidate_info =
        context.candidate.edge_classification_between(node, neighbor);
    const auto baseline_info =
        context.baseline.edge_classification_between(node, neighbor);
    ++context.result.edge_count;
    context.result.barrier_edge_count += candidate_barrier ? 1ULL : 0ULL;
    context.result.interface_edge_count += candidate_interface ? 1ULL : 0ULL;
    context.topology.add_signed(axis);
    context.topology.add_signed(i);
    context.topology.add_signed(j);
    context.topology.add_signed(k);
    context.topology.add_bool(candidate_barrier);
    context.topology.add_bool(candidate_interface);
    hash_classification(context.topology, candidate_info);
    context.materialization.add_signed(axis);
    context.materialization.add_signed(i);
    context.materialization.add_signed(j);
    context.materialization.add_signed(k);
    context.materialization.add_bool(candidate_barrier);
    context.materialization.add_bool(candidate_interface);
    hash_classification(context.materialization, candidate_info);
    if (candidate_barrier != baseline_barrier
        || candidate_interface != baseline_interface) {
        ++context.result.edge_flag_mismatch_count;
        record_mismatch(
            context.mismatches, context.prepared, context.backend,
            "edge_flags", axis, i, j, k,
            "baseline={barrier=" + std::to_string(baseline_barrier ? 1 : 0)
                + ";interface="
                + std::to_string(baseline_interface ? 1 : 0)
                + ";classification="
                + serialize_edge_classification_3d(baseline_info)
                + "};candidate={barrier="
                + std::to_string(candidate_barrier ? 1 : 0)
                + ";interface="
                + std::to_string(candidate_interface ? 1 : 0)
                + ";classification="
                + serialize_edge_classification_3d(candidate_info)
                + "};errors={barrier="
                + std::to_string(candidate_barrier != baseline_barrier)
                + ";interface="
                + std::to_string(candidate_interface != baseline_interface)
                + "};" + tolerance_detail(context.tolerance)
                + ";flag_tolerance=exact");
    }
    if (!same_classification(baseline_info, candidate_info)) {
        ++context.result.edge_classification_mismatch_count;
        record_mismatch(
            context.mismatches, context.prepared, context.backend,
            "edge_classification", axis, i, j, k,
            classification_comparison_detail(
                baseline_info, candidate_info, context.tolerance)
                + ";classification_tolerance=exact");
    }
    scan_crossings(
        context, axis, i, j, k, node, neighbor, edge_length);
    scan_label_changing_edge(
        context, candidate_info, axis, i, j, k,
        node, neighbor, edge_length);
}

BackendMaterialization3D materialize_backend(
    const PreparedCase& prepared,
    Backend3D backend,
    const BuiltBackend& candidate,
    const BuiltBackend& baseline,
    std::vector<std::string>& mismatches)
{
    if (!candidate.grid_pair->has_nurbs_domain()
        || !baseline.grid_pair->has_nurbs_domain()) {
        throw std::runtime_error(
            "benchmark GridPair3D does not expose its native NURBS domain");
    }
    ScanContext context(
        prepared, backend, *candidate.domain, *candidate.grid_pair,
        *baseline.domain, *baseline.grid_pair, mismatches);
    scan_nodes(context);
    const auto dims = prepared.grid.dof_dims();
    for (int k = 0; k < dims[2]; ++k) {
        for (int j = 0; j < dims[1]; ++j) {
            for (int i = 0; i < dims[0]; ++i) {
                const int node = prepared.grid.index(i, j, k);
                for (int axis = 0; axis < 3; ++axis) {
                    std::array<int, 3> next{{i, j, k}};
                    if (++next[static_cast<std::size_t>(axis)]
                        >= dims[static_cast<std::size_t>(axis)]) {
                        continue;
                    }
                    const int neighbor = prepared.grid.index(
                        next[0], next[1], next[2]);
                    scan_edge(context, axis, i, j, k, node, neighbor);
                }
            }
        }
    }
    const std::uint64_t N = static_cast<std::uint64_t>(prepared.N);
    const std::uint64_t dimension = N + 1ULL;
    const std::uint64_t expected_edges = 3ULL * N * dimension * dimension;
    if (context.result.edge_count != expected_edges)
        throw std::logic_error("benchmark did not scan every structured edge");
    context.result.topology_checksum = context.topology.value();
    context.result.materialization_checksum = context.materialization.value();
    return context.result;
}

using IntersectionDiagnostics =
    geometry3d::NurbsSurfaceIntersectionDiagnostics3D;
using IntersectionMember = int IntersectionDiagnostics::*;

const std::vector<std::pair<const char*, IntersectionMember>>&
intersection_fields()
{
    static const std::vector<std::pair<const char*, IntersectionMember>> fields{
        {"candidate_elements", &IntersectionDiagnostics::candidate_elements},
        {"bvh_candidate_elements", &IntersectionDiagnostics::bvh_candidate_elements},
        {"mapped_candidate_elements", &IntersectionDiagnostics::mapped_candidate_elements},
        {"maximum_candidate_elements_per_edge", &IntersectionDiagnostics::maximum_candidate_elements_per_edge},
        {"triangle_seed_hits", &IntersectionDiagnostics::triangle_seed_hits},
        {"triangle_seed_misses_recovered", &IntersectionDiagnostics::triangle_seed_misses_recovered},
        {"subdivision_boxes", &IntersectionDiagnostics::subdivision_boxes},
        {"newton_attempts", &IntersectionDiagnostics::newton_attempts},
        {"newton_iterations", &IntersectionDiagnostics::newton_iterations},
        {"early_unique_certificate_attempts", &IntersectionDiagnostics::early_unique_certificate_attempts},
        {"early_unique_certificate_successes", &IntersectionDiagnostics::early_unique_certificate_successes},
        {"planar_analytic_hits", &IntersectionDiagnostics::planar_analytic_hits},
        {"planar_analytic_misses", &IntersectionDiagnostics::planar_analytic_misses},
        {"planar_analytic_fallbacks", &IntersectionDiagnostics::planar_analytic_fallbacks},
        {"closest_point_prefilter_attempts", &IntersectionDiagnostics::closest_point_prefilter_attempts},
        {"closest_point_prefilter_certified_hits", &IntersectionDiagnostics::closest_point_prefilter_certified_hits},
        {"closest_point_prefilter_certified_misses", &IntersectionDiagnostics::closest_point_prefilter_certified_misses},
        {"closest_point_prefilter_fallbacks", &IntersectionDiagnostics::closest_point_prefilter_fallbacks},
        {"certified_fallback_elements", &IntersectionDiagnostics::certified_fallback_elements},
        {"same_patch_deduplications", &IntersectionDiagnostics::same_patch_deduplications},
        {"seam_deduplications", &IntersectionDiagnostics::seam_deduplications},
        {"unresolved_candidates", &IntersectionDiagnostics::unresolved_candidates},
        {"maximum_subdivision_depth_reached", &IntersectionDiagnostics::maximum_subdivision_depth_reached},
        {"terminal_certificate_boxes", &IntersectionDiagnostics::terminal_certificate_boxes},
        {"maximum_terminal_certificate_depth_reached", &IntersectionDiagnostics::maximum_terminal_certificate_depth_reached},
        {"closest_point_attempts", &IntersectionDiagnostics::closest_point_attempts},
        {"closest_point_iterations", &IntersectionDiagnostics::closest_point_iterations},
        {"roots_recovered_by_closest_point", &IntersectionDiagnostics::roots_recovered_by_closest_point},
        {"terminal_misses_by_closest_point", &IntersectionDiagnostics::terminal_misses_by_closest_point},
        {"closest_point_failures", &IntersectionDiagnostics::closest_point_failures},
        {"sample_seed_candidates", &IntersectionDiagnostics::sample_seed_candidates},
        {"sample_seeds_accepted", &IntersectionDiagnostics::sample_seeds_accepted},
        {"roots_recovered_by_sample_seed", &IntersectionDiagnostics::roots_recovered_by_sample_seed},
        {"maximum_sample_seeds_per_element", &IntersectionDiagnostics::maximum_sample_seeds_per_element},
        {"stationary_solve_attempts", &IntersectionDiagnostics::stationary_solve_attempts},
        {"stationary_solve_converged", &IntersectionDiagnostics::stationary_solve_converged},
        {"stationary_witnesses", &IntersectionDiagnostics::stationary_witnesses},
        {"root_pairs_protected_by_stationary_witness", &IntersectionDiagnostics::root_pairs_protected_by_stationary_witness},
        {"ambiguous_root_clusters", &IntersectionDiagnostics::ambiguous_root_clusters},
        {"non_g1_topology_merges", &IntersectionDiagnostics::non_g1_topology_merges},
        {"high_degree_fallbacks", &IntersectionDiagnostics::high_degree_fallbacks}};
    return fields;
}

void append_column(std::vector<std::string>& columns,
                   bool header,
                   const std::string& name,
                   const std::string& value)
{
    columns.push_back(header ? name : value);
}

void append_intersection_columns(
    std::vector<std::string>& columns,
    bool header,
    const std::string& prefix,
    const IntersectionDiagnostics& diagnostics)
{
    for (const auto& field : intersection_fields()) {
        append_column(
            columns, header, prefix + field.first,
            std::to_string(diagnostics.*(field.second)));
    }
}

void append_materialization_columns(
    std::vector<std::string>& columns,
    bool header,
    const BackendMaterialization3D& value)
{
    using U64Member = std::uint64_t BackendMaterialization3D::*;
    static const std::vector<std::pair<const char*, U64Member>> integers{
        {"node_count", &BackendMaterialization3D::node_count},
        {"edge_count", &BackendMaterialization3D::edge_count},
        {"barrier_edge_count", &BackendMaterialization3D::barrier_edge_count},
        {"interface_edge_count", &BackendMaterialization3D::interface_edge_count},
        {"crossing_count", &BackendMaterialization3D::crossing_count},
        {"label_changing_edge_count", &BackendMaterialization3D::label_changing_edge_count},
        {"correction_owner_count", &BackendMaterialization3D::correction_owner_count},
        {"analytic_label_mismatch_count", &BackendMaterialization3D::analytic_label_mismatch_count},
        {"baseline_label_mismatch_count", &BackendMaterialization3D::baseline_label_mismatch_count},
        {"edge_flag_mismatch_count", &BackendMaterialization3D::edge_flag_mismatch_count},
        {"edge_classification_mismatch_count", &BackendMaterialization3D::edge_classification_mismatch_count},
        {"crossing_count_mismatch_count", &BackendMaterialization3D::crossing_count_mismatch_count},
        {"crossing_field_mismatch_count", &BackendMaterialization3D::crossing_field_mismatch_count},
        {"correction_crossing_mismatch_count", &BackendMaterialization3D::correction_crossing_mismatch_count},
        {"correction_owner_mismatch_count", &BackendMaterialization3D::correction_owner_mismatch_count},
        {"unsafe_label_changing_edge_count", &BackendMaterialization3D::unsafe_label_changing_edge_count},
        {"topology_checksum", &BackendMaterialization3D::topology_checksum},
        {"materialization_checksum", &BackendMaterialization3D::materialization_checksum}};
    for (const auto& field : integers) {
        append_column(columns, header, field.first,
                      std::to_string(value.*(field.second)));
    }
    using DoubleMember = double BackendMaterialization3D::*;
    static const std::vector<std::pair<const char*, DoubleMember>> errors{
        {"maximum_point_error", &BackendMaterialization3D::maximum_point_error},
        {"maximum_normal_error", &BackendMaterialization3D::maximum_normal_error},
        {"maximum_uv_point_error", &BackendMaterialization3D::maximum_uv_point_error},
        {"maximum_edge_parameter_error", &BackendMaterialization3D::maximum_edge_parameter_error},
        {"maximum_residual_error", &BackendMaterialization3D::maximum_residual_error},
        {"maximum_transversality_error", &BackendMaterialization3D::maximum_transversality_error},
        {"maximum_reliability_tolerance_error", &BackendMaterialization3D::maximum_reliability_tolerance_error},
        {"maximum_owner_error", &BackendMaterialization3D::maximum_owner_error}};
    for (const auto& field : errors) {
        append_column(columns, header, field.first,
                      format_double(value.*(field.second)));
    }
}

std::string component_sizes_text(const std::vector<std::size_t>& sizes)
{
    std::ostringstream stream;
    for (std::size_t index = 0; index < sizes.size(); ++index) {
        if (index != 0)
            stream << ';';
        stream << sizes[index];
    }
    return stream.str();
}

void append_domain_counter_columns(
    std::vector<std::string>& columns,
    bool header,
    const geometry3d::NurbsCartesianDomainDiagnostics3D& value)
{
    using Diagnostics = geometry3d::NurbsCartesianDomainDiagnostics3D;
    using IntMember = int Diagnostics::*;
    static const std::vector<std::pair<const char*, IntMember>> integers{
        {"nurbs_patch_count", &Diagnostics::nurbs_patch_count},
        {"bezier_element_count", &Diagnostics::bezier_element_count},
        {"acceleration_leaf_count", &Diagnostics::acceleration_leaf_count},
        {"grid_component_count", &Diagnostics::grid_component_count},
        {"box_exterior_component_count", &Diagnostics::box_exterior_component_count},
        {"representative_query_count", &Diagnostics::representative_query_count}};
    for (const auto& field : integers) {
        append_column(columns, header, field.first,
                      std::to_string(value.*(field.second)));
    }
    using SizeMember = std::size_t Diagnostics::*;
    static const std::vector<std::pair<const char*, SizeMember>> sizes{
        {"candidate_grid_edge_count", &Diagnostics::candidate_grid_edge_count},
        {"candidate_element_incidence_count", &Diagnostics::candidate_element_incidence_count},
        {"multi_crossing_edge_count", &Diagnostics::multi_crossing_edge_count},
        {"even_parity_interface_edge_count", &Diagnostics::even_parity_interface_edge_count},
        {"odd_parity_interface_edge_count", &Diagnostics::odd_parity_interface_edge_count},
        {"ambiguous_parity_edge_count", &Diagnostics::ambiguous_parity_edge_count},
        {"ambiguous_label_changing_edge_count", &Diagnostics::ambiguous_label_changing_edge_count},
        {"targeted_retry_count", &Diagnostics::targeted_retry_count},
        {"targeted_retry_resolved_count", &Diagnostics::targeted_retry_resolved_count},
        {"targeted_retry_unsafe_count", &Diagnostics::targeted_retry_unsafe_count},
        {"correction_safe_edge_count", &Diagnostics::correction_safe_edge_count},
        {"unsafe_label_changing_edge_count_diagnostic", &Diagnostics::unsafe_label_changing_edge_count},
        {"endpoint_parity_fallback_count", &Diagnostics::endpoint_parity_fallback_count},
        {"endpoint_classification_query_count", &Diagnostics::endpoint_classification_query_count},
        {"component_parity_toggle_count", &Diagnostics::component_parity_toggle_count}};
    for (const auto& field : sizes) {
        append_column(columns, header, field.first,
                      std::to_string(value.*(field.second)));
    }
    for (int axis = 0; axis < 3; ++axis) {
        static const char* names[] = {"x", "y", "z"};
        append_column(columns, header,
                      std::string("barrier_edge_count_") + names[axis],
                      std::to_string(value.barrier_edge_counts[axis]));
        append_column(columns, header,
                      std::string("interface_edge_count_") + names[axis],
                      std::to_string(value.interface_edge_counts[axis]));
    }
    append_column(columns, header, "component_sizes",
                  component_sizes_text(value.component_sizes));
    append_column(columns, header, "maximum_query_element_extent",
                  format_double(value.maximum_query_element_extent));
    append_column(columns, header, "maximum_root_residual",
                  format_double(value.maximum_root_residual));
}

std::vector<std::string> raw_columns(
    const RawBenchmarkRecord3D& record,
    bool header)
{
    std::vector<std::string> columns;
    append_column(columns, header, "geometry", record.geometry);
    append_column(columns, header, "N", std::to_string(record.N));
    append_column(columns, header, "backend", backend_name_3d(record.backend));
    append_column(columns, header, "repetition", std::to_string(record.repetition));
    append_column(columns, header, "execution_order", std::to_string(record.execution_order));
    append_column(columns, header, "domain_seconds", format_double(record.domain_seconds));
    append_column(columns, header, "grid_pair_seconds", format_double(record.grid_pair_seconds));
    append_column(columns, header, "combined_seconds", format_double(record.combined_seconds));
    append_materialization_columns(columns, header, record.materialization);
    append_domain_counter_columns(columns, header, record.diagnostics);
    append_column(columns, header, "intersector_build_seconds",
                  format_double(record.diagnostics.intersector_build_seconds));
    append_column(columns, header, "candidate_enumeration_seconds",
                  format_double(record.diagnostics.candidate_enumeration_seconds));
    append_column(columns, header, "edge_intersection_seconds",
                  format_double(record.diagnostics.edge_intersection_seconds));
    append_column(columns, header, "edge_materialization_seconds",
                  format_double(record.diagnostics.edge_materialization_seconds));
    append_column(columns, header, "flood_labeling_seconds",
                  format_double(record.diagnostics.flood_labeling_seconds));
    append_column(columns, header, "representative_classification_seconds",
                  format_double(record.diagnostics.representative_classification_seconds));
    append_column(columns, header, "invariant_verification_seconds",
                  format_double(record.diagnostics.invariant_verification_seconds));
    append_column(columns, header, "total_construction_seconds",
                  format_double(record.diagnostics.total_construction_seconds));
    append_intersection_columns(
        columns, header, "intersections_", record.diagnostics.intersections);
    append_intersection_columns(
        columns, header, "targeted_retry_intersections_",
        record.diagnostics.targeted_retry_intersections);
    return columns;
}

void append_timing_summary_columns(
    std::vector<std::string>& columns,
    bool header,
    const std::string& prefix,
    const TimingSummary3D& value)
{
    append_column(columns, header, prefix + "_sample_count",
                  std::to_string(value.sample_count));
    append_column(columns, header, prefix + "_minimum",
                  format_double(value.minimum));
    append_column(columns, header, prefix + "_median",
                  format_double(value.median));
    append_column(columns, header, prefix + "_maximum",
                  format_double(value.maximum));
    append_column(columns, header, prefix + "_coefficient_of_variation",
                  format_double(value.coefficient_of_variation));
    append_column(columns, header, prefix + "_speedup_vs_baseline",
                  format_double(value.speedup));
}

std::vector<std::string> summary_columns(
    const SummaryBenchmarkRecord3D& record,
    bool header)
{
    std::vector<std::string> columns;
    append_column(columns, header, "geometry", record.geometry);
    append_column(columns, header, "N", std::to_string(record.N));
    append_column(columns, header, "backend", backend_name_3d(record.backend));
    append_timing_summary_columns(columns, header, "domain", record.domain);
    append_timing_summary_columns(columns, header, "grid_pair", record.grid_pair);
    append_timing_summary_columns(columns, header, "combined", record.combined);
    return columns;
}

Backend3D parse_backend_name(const std::string& name)
{
    if (name == "baseline")
        return Backend3D::Baseline;
    if (name == "pure")
        return Backend3D::Pure;
    if (name == "hybrid")
        return Backend3D::Hybrid;
    throw std::invalid_argument("invalid backend name: " + name);
}

GeometryKind3D parse_geometry_name(const std::string& name)
{
    if (name == "torus")
        return GeometryKind3D::Torus;
    if (name == "cylinder")
        return GeometryKind3D::HollowCylinder;
    if (name == "l_prism")
        return GeometryKind3D::LPrism;
    throw std::invalid_argument("invalid geometry name: " + name);
}

int parse_integer(const std::string& text,
                  const std::string& option,
                  bool allow_zero)
{
    try {
        std::size_t consumed = 0;
        const long long parsed = std::stoll(text, &consumed);
        if (consumed != text.size()
            || parsed > std::numeric_limits<int>::max()
            || parsed < std::numeric_limits<int>::min()
            || (allow_zero ? parsed < 0 : parsed <= 0)) {
            throw std::invalid_argument("range");
        }
        return static_cast<int>(parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument(
            option + " requires "
            + (allow_zero ? "a nonnegative integer" : "a positive integer"));
    }
}

std::uint64_t checked_node_dimension(int N)
{
    if (N <= 0)
        throw std::invalid_argument("benchmark grid N must be positive");
    const std::uint64_t dimension =
        static_cast<std::uint64_t>(N) + 1ULL;
    const std::uint64_t maximum_dofs =
        static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    if (dimension > maximum_dofs / dimension)
        throw std::invalid_argument(
            "benchmark Node grid DOF count exceeds int capacity");
    const std::uint64_t plane_dofs = dimension * dimension;
    if (dimension > maximum_dofs / plane_dofs)
        throw std::invalid_argument(
            "benchmark Node grid DOF count exceeds int capacity");
    return dimension;
}

bool is_option(const std::string& value)
{
    return value.size() >= 2 && value[0] == '-' && value[1] == '-';
}

void write_csv_file(
    const std::filesystem::path& path,
    const std::vector<std::string>& header,
    const std::vector<std::vector<std::string>>& rows)
{
    std::ofstream stream(path);
    if (!stream)
        throw std::runtime_error("unable to open benchmark CSV: " + path.string());
    stream << encode_csv_row_3d(header) << '\n';
    for (const auto& row : rows) {
        if (row.size() != header.size())
            throw std::logic_error("benchmark CSV header/row width mismatch");
        stream << encode_csv_row_3d(row) << '\n';
    }
    if (!stream)
        throw std::runtime_error("failed while writing benchmark CSV: " + path.string());
}

void write_mismatch_file(
    const std::filesystem::path& path,
    const std::vector<std::string>& rows)
{
    std::ofstream stream(path);
    if (!stream)
        throw std::runtime_error(
            "unable to open benchmark mismatch CSV: " + path.string());
    stream << "geometry,N,backend,category,axis,i,j,k,detail\n";
    for (const std::string& row : rows)
        stream << row << '\n';
    if (!stream)
        throw std::runtime_error(
            "failed while writing benchmark mismatch CSV: " + path.string());
}

std::vector<SummaryBenchmarkRecord3D> build_summary_records(
    const std::vector<RawBenchmarkRecord3D>& raw)
{
    using Key = std::tuple<std::string, int, Backend3D>;
    struct Samples {
        std::vector<double> domain;
        std::vector<double> grid_pair;
        std::vector<double> combined;
    };
    std::map<Key, Samples> grouped;
    for (const RawBenchmarkRecord3D& record : raw) {
        Samples& samples = grouped[{record.geometry, record.N, record.backend}];
        samples.domain.push_back(record.domain_seconds);
        samples.grid_pair.push_back(record.grid_pair_seconds);
        samples.combined.push_back(record.combined_seconds);
    }
    std::map<std::pair<std::string, int>, std::array<double, 3>> baselines;
    const double missing = std::numeric_limits<double>::quiet_NaN();
    for (const auto& group : grouped) {
        if (std::get<2>(group.first) != Backend3D::Baseline)
            continue;
        const Samples& samples = group.second;
        baselines[{std::get<0>(group.first), std::get<1>(group.first)}] = {{
            summarize_timings_3d(samples.domain, missing).median,
            summarize_timings_3d(samples.grid_pair, missing).median,
            summarize_timings_3d(samples.combined, missing).median}};
    }
    std::vector<SummaryBenchmarkRecord3D> summary;
    for (const auto& group : grouped) {
        const auto baseline = baselines.find(
            {std::get<0>(group.first), std::get<1>(group.first)});
        const std::array<double, 3> reference = baseline == baselines.end()
            ? std::array<double, 3>{{missing, missing, missing}}
            : baseline->second;
        const Samples& samples = group.second;
        SummaryBenchmarkRecord3D record;
        record.geometry = std::get<0>(group.first);
        record.N = std::get<1>(group.first);
        record.backend = std::get<2>(group.first);
        record.domain = summarize_timings_3d(samples.domain, reference[0]);
        record.grid_pair =
            summarize_timings_3d(samples.grid_pair, reference[1]);
        record.combined =
            summarize_timings_3d(samples.combined, reference[2]);
        summary.push_back(std::move(record));
    }
    return summary;
}

void write_benchmark_outputs(
    const BenchmarkOptions3D& options,
    const BenchmarkRunResult3D& result)
{
    std::filesystem::create_directories(options.output_directory);
    std::vector<std::vector<std::string>> raw_rows;
    raw_rows.reserve(result.raw.size());
    for (const RawBenchmarkRecord3D& record : result.raw)
        raw_rows.push_back(raw_csv_row_3d(record));
    write_csv_file(
        options.output_directory / "nurbs_geometry_preprocess_raw.csv",
        raw_csv_header_3d(), raw_rows);
    std::vector<std::vector<std::string>> summary_rows;
    summary_rows.reserve(result.summary.size());
    for (const SummaryBenchmarkRecord3D& record : result.summary)
        summary_rows.push_back(summary_csv_row_3d(record));
    write_csv_file(
        options.output_directory / "nurbs_geometry_preprocess_summary.csv",
        summary_csv_header_3d(), summary_rows);
    const std::filesystem::path mismatch_path =
        options.output_directory / "nurbs_geometry_preprocess_mismatches.csv";
    if (!result.mismatches.empty()) {
        write_mismatch_file(
            mismatch_path, result.mismatches);
    } else {
        std::filesystem::remove(mismatch_path);
    }
}

} // namespace

const char* backend_name_3d(Backend3D backend) noexcept
{
    switch (backend) {
    case Backend3D::Baseline: return "baseline";
    case Backend3D::Pure: return "pure";
    case Backend3D::Hybrid: return "hybrid";
    }
    return "invalid";
}

const char* geometry_name_3d(GeometryKind3D geometry) noexcept
{
    switch (geometry) {
    case GeometryKind3D::Torus: return "torus";
    case GeometryKind3D::HollowCylinder: return "cylinder";
    case GeometryKind3D::LPrism: return "l_prism";
    }
    return "invalid";
}

TimingSummary3D summarize_timings_3d(
    const std::vector<double>& samples,
    double matching_baseline_median)
{
    if (samples.empty())
        throw std::invalid_argument("cannot summarize an empty timing sample");
    for (double sample : samples) {
        if (!std::isfinite(sample) || sample < 0.0)
            throw std::invalid_argument("timing samples must be finite and nonnegative");
    }
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    TimingSummary3D result;
    result.sample_count = sorted.size();
    result.minimum = sorted.front();
    result.maximum = sorted.back();
    const std::size_t middle = sorted.size() / 2;
    result.median = sorted.size() % 2 == 0
        ? 0.5 * (sorted[middle - 1] + sorted[middle])
        : sorted[middle];
    const double mean = std::accumulate(
        sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());
    double variance = 0.0;
    for (double sample : sorted) {
        const double delta = sample - mean;
        variance += delta * delta;
    }
    variance /= static_cast<double>(sorted.size());
    result.coefficient_of_variation = mean == 0.0
        ? 0.0 : std::sqrt(variance) / mean;
    result.speedup = std::isfinite(matching_baseline_median)
            && matching_baseline_median >= 0.0 && result.median > 0.0
        ? matching_baseline_median / result.median
        : std::numeric_limits<double>::quiet_NaN();
    return result;
}

std::vector<std::string> raw_csv_header_3d()
{
    return raw_columns(RawBenchmarkRecord3D{}, true);
}

std::vector<std::string> raw_csv_row_3d(
    const RawBenchmarkRecord3D& record)
{
    return raw_columns(record, false);
}

std::vector<std::string> summary_csv_header_3d()
{
    return summary_columns(SummaryBenchmarkRecord3D{}, true);
}

std::vector<std::string> summary_csv_row_3d(
    const SummaryBenchmarkRecord3D& record)
{
    return summary_columns(record, false);
}

std::string encode_csv_row_3d(const std::vector<std::string>& fields)
{
    constexpr char quote_char = static_cast<char>(34);
    std::ostringstream stream;
    for (std::size_t index = 0; index < fields.size(); ++index) {
        if (index != 0)
            stream << ',';
        const std::string& field = fields[index];
        const bool quote = field.find(',') != std::string::npos
            || field.find(quote_char) != std::string::npos
            || field.find('\r') != std::string::npos
            || field.find('\n') != std::string::npos;
        if (!quote) {
            stream << field;
            continue;
        }
        stream << quote_char;
        for (char character : field) {
            if (character == quote_char)
                stream << quote_char;
            stream << character;
        }
        stream << quote_char;
    }
    return stream.str();
}

std::string serialize_edge_classification_3d(
    const geometry3d::NurbsCartesianEdgeClassification3D& value)
{
    std::ostringstream stream;
    stream << "{queried=" << value.queried
           << ";has_confirmed_interface=" << value.has_confirmed_interface
           << ";changes_component_membership="
           << value.changes_component_membership
           << ";root_count_known=" << value.root_count_known
           << ";parity_known_from_roots=" << value.parity_known_from_roots
           << ";has_near_tangent_candidate="
           << value.has_near_tangent_candidate
           << ";used_targeted_retry=" << value.used_targeted_retry
           << ";correction_safe=" << value.correction_safe
           << ";confirmed_crossing_count="
           << value.confirmed_crossing_count
           << ";ambiguous_cluster_count=" << value.ambiguous_cluster_count
           << ";confirmed_transverse_count="
           << value.confirmed_transverse_count << '}';
    return stream.str();
}

std::string serialize_surface_crossing_3d(
    const geometry3d::NurbsSurfaceCrossing3D& value)
{
    std::ostringstream stream;
    stream << "{patch_index=" << value.patch_index
           << ";component=" << value.component
           << ";u=" << format_double(value.u)
           << ";v=" << format_double(value.v)
           << ";edge_parameter=" << format_double(value.edge_parameter)
           << ";point=" << format_vector(value.point)
           << ";normal=" << format_vector(value.normal)
           << ";residual=" << format_double(value.residual)
           << ";transversality=" << format_double(value.transversality)
           << ";feature_edge_contact=" << value.feature_edge_contact
           << ";reliable_transversality_tolerance="
           << format_double(value.reliable_transversality_tolerance) << '}';
    return stream.str();
}

std::string serialize_crossing_owner_3d(const P2CrossingOwner3D& value)
{
    std::ostringstream stream;
    stream << "{center_index=" << value.center_index
           << ";panel_index=" << value.panel_index
           << ";edge_parameter=" << format_double(value.edge_parameter)
           << ";barycentric=" << format_vector(value.barycentric)
           << ";geometry_panel_index=" << value.geometry_panel_index
           << ";geometry_barycentric="
           << format_vector(value.geometry_barycentric)
           << ";nurbs_patch_index=" << value.nurbs_patch_index
           << ";nurbs_parameter=" << format_vector(value.nurbs_parameter)
           << ";crossing_point=" << format_vector(value.crossing_point)
           << ";crossing_normal=" << format_vector(value.crossing_normal)
           << ";surface_component=" << value.surface_component
           << ";crossing_residual="
           << format_double(value.crossing_residual)
           << ";status=" << static_cast<int>(value.status) << '}';
    return stream.str();
}

std::string compose_mismatch_detail_3d(
    const std::string& baseline,
    const std::string& candidate,
    const std::string& errors,
    const std::string& tolerance)
{
    return "baseline=" + baseline + ";candidate=" + candidate
        + ";errors=" + errors + ";tolerance=" + tolerance;
}

BenchmarkOptions3D parse_benchmark_cli_3d(
    const std::vector<std::string>& arguments)
{
    BenchmarkOptions3D options;
    bool saw_backend = false;
    bool saw_geometry = false;
    bool saw_levels = false;
    bool saw_warmup = false;
    bool saw_repetitions = false;
    bool saw_output = false;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const std::string& option = arguments[index];
        const auto require_value = [&]() -> const std::string& {
            if (index + 1 >= arguments.size()
                || is_option(arguments[index + 1])) {
                throw std::invalid_argument(option + " requires a value");
            }
            return arguments[++index];
        };
        if (option == "--backend") {
            if (saw_backend)
                throw std::invalid_argument("duplicate --backend option");
            saw_backend = true;
            const std::string& value = require_value();
            options.backends = value == "all"
                ? std::vector<Backend3D>{
                    Backend3D::Baseline, Backend3D::Pure, Backend3D::Hybrid}
                : std::vector<Backend3D>{parse_backend_name(value)};
            continue;
        }
        if (option == "--geometry") {
            if (saw_geometry)
                throw std::invalid_argument("duplicate --geometry option");
            saw_geometry = true;
            const std::string& value = require_value();
            options.geometries = value == "all"
                ? std::vector<GeometryKind3D>{
                    GeometryKind3D::Torus,
                    GeometryKind3D::HollowCylinder,
                    GeometryKind3D::LPrism}
                : std::vector<GeometryKind3D>{parse_geometry_name(value)};
            continue;
        }
        if (option == "--N") {
            if (saw_levels)
                throw std::invalid_argument("duplicate --N option");
            saw_levels = true;
            options.levels.clear();
            while (index + 1 < arguments.size()
                   && !is_option(arguments[index + 1])) {
                const int level = parse_integer(arguments[++index], "--N", false);
                (void)checked_node_dimension(level);
                if (std::find(options.levels.begin(), options.levels.end(), level)
                    != options.levels.end()) {
                    throw std::invalid_argument("duplicate --N value");
                }
                options.levels.push_back(level);
            }
            if (options.levels.empty())
                throw std::invalid_argument("--N requires one or more values");
            continue;
        }
        if (option == "--warmup") {
            if (saw_warmup)
                throw std::invalid_argument("duplicate --warmup option");
            saw_warmup = true;
            options.warmup = parse_integer(require_value(), "--warmup", true);
            continue;
        }
        if (option == "--reps") {
            if (saw_repetitions)
                throw std::invalid_argument("duplicate --reps option");
            saw_repetitions = true;
            options.repetitions =
                parse_integer(require_value(), "--reps", false);
            continue;
        }
        if (option == "--out") {
            if (saw_output)
                throw std::invalid_argument("duplicate --out option");
            saw_output = true;
            const std::string& value = require_value();
            if (value.empty())
                throw std::invalid_argument("--out requires a nonempty path");
            options.output_directory = value;
            continue;
        }
        throw std::invalid_argument("unknown benchmark option: " + option);
    }
    return options;
}

void validate_benchmark_options_3d(const BenchmarkOptions3D& options)
{
    if (options.backends.empty() || options.geometries.empty()
        || options.levels.empty()) {
        throw std::invalid_argument("benchmark selections must be nonempty");
    }
    if (options.warmup < 0 || options.repetitions <= 0)
        throw std::invalid_argument("invalid warmup or repetition count");
    if (options.output_directory.empty())
        throw std::invalid_argument("benchmark output directory is empty");
    for (int level : options.levels)
        (void)checked_node_dimension(level);
}

BenchmarkSupportCase3D run_benchmark_support_case_3d(
    GeometryKind3D geometry,
    int N)
{
    if (N <= 0)
        throw std::invalid_argument("benchmark support grid N must be positive");
    const PreparedCase prepared(geometry, N);
    const BuiltBackend baseline =
        build_backend(prepared, Backend3D::Baseline);
    validate_production_grid(prepared, *baseline.domain);
    BenchmarkSupportCase3D result;
    result.runs.push_back(materialize_backend(
        prepared, Backend3D::Baseline, baseline, baseline,
        result.mismatches));
    for (Backend3D backend : {Backend3D::Pure, Backend3D::Hybrid}) {
        const BuiltBackend candidate = build_backend(prepared, backend);
        result.runs.push_back(materialize_backend(
            prepared, backend, candidate, baseline, result.mismatches));
    }
    return result;
}

BenchmarkRunResult3D run_nurbs_geometry_preprocess_benchmark_3d(
    const BenchmarkOptions3D& options)
{
    validate_benchmark_options_3d(options);
    BenchmarkRunResult3D result;
    for (GeometryKind3D geometry : options.geometries) {
        for (int level : options.levels) {
            const PreparedCase prepared(geometry, level);
            const BuiltBackend baseline_reference =
                build_backend(prepared, Backend3D::Baseline);
            validate_production_grid(prepared, *baseline_reference.domain);
            for (int warmup = 0; warmup < options.warmup; ++warmup) {
                std::vector<Backend3D> order = options.backends;
                std::rotate(
                    order.begin(),
                    order.begin() + (warmup % order.size()),
                    order.end());
                for (Backend3D backend : order) {
                    const BuiltBackend ignored =
                        build_backend(prepared, backend);
                    (void)ignored;
                }
            }
            for (int repetition = 0;
                 repetition < options.repetitions; ++repetition) {
                std::vector<Backend3D> order = options.backends;
                std::rotate(
                    order.begin(),
                    order.begin() + (repetition % order.size()),
                    order.end());
                for (std::size_t execution_order = 0;
                     execution_order < order.size(); ++execution_order) {
                    const Backend3D backend = order[execution_order];
                    const BuiltBackend measured =
                        build_backend(prepared, backend);
                    BackendMaterialization3D materialization =
                        materialize_backend(
                            prepared, backend, measured, baseline_reference,
                            result.mismatches);
                    RawBenchmarkRecord3D record;
                    record.geometry = geometry_name_3d(geometry);
                    record.N = level;
                    record.backend = backend;
                    record.repetition = repetition;
                    record.execution_order =
                        static_cast<int>(execution_order);
                    record.domain_seconds = measured.domain_seconds;
                    record.grid_pair_seconds = measured.grid_pair_seconds;
                    record.combined_seconds =
                        measured.domain_seconds + measured.grid_pair_seconds;
                    record.diagnostics = measured.domain->diagnostics();
                    record.materialization = std::move(materialization);
                    result.raw.push_back(std::move(record));
                }
            }
        }
    }
    result.summary = build_summary_records(result.raw);
    write_benchmark_outputs(options, result);
    return result;
}

} // namespace kfbim::app3d::benchmark3d

#ifdef KFBIM_NURBS_GEOMETRY_PREPROCESS_BENCHMARK_3D_MAIN
int main(int argc, char** argv)
{
    try {
        std::vector<std::string> arguments;
        for (int index = 1; index < argc; ++index)
            arguments.emplace_back(argv[index]);
        const auto options =
            kfbim::app3d::benchmark3d::parse_benchmark_cli_3d(arguments);
        const auto result = kfbim::app3d::benchmark3d::
            run_nurbs_geometry_preprocess_benchmark_3d(options);
        std::cout << "wrote " << result.raw.size()
                  << " raw rows and " << result.summary.size()
                  << " summary rows\n";
        if (!result.mismatches.empty()) {
            std::cerr << "geometry preprocess mismatches: "
                      << result.mismatches.size() << '\n';
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "nurbs geometry preprocess benchmark failure: "
                  << error.what() << '\n';
        return 1;
    }
}
#endif
