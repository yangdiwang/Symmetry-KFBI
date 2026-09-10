#pragma once

#include "src/support/cauchy/direct_coefficient_cubic_cauchy_3d.hpp"
#include "src/support/trace/resource_restrict_assembly_3d.hpp"
#include "src/transfer/laplace_correction_support.hpp"

#include <functional>
#include <string>

namespace kfbim::app3d {

using ResourceKnownAmbientCallback3D = std::function<KnownAmbientThird3D(const Eigen::Vector3d&)>;

// One immutable model/grid/policy context. Native parameters remain unrounded;
// full Cartesian IDs are used throughout (the current C++ bulk field includes
// boundary nodes). This is an anchor strategy, NOT a path certificate.
struct ResourceGeometryPlan3D {
    std::vector<RestrictResourceAnchor3D> trace_anchors;
    std::vector<RestrictResourceAnchor3D> spread_anchors;
    std::vector<int> crossing_for_op;
    std::vector<SharedSideCoverRestrictStencil3D> stencils;
    std::vector<RestrictSupportVisit3D> visits;
    RestrictResourcePlan3D restrict_plan;
    double planning_seconds = 0.0;
};

[[nodiscard]] ResourceGeometryPlan3D build_resource_geometry_plan_3d(
    const CartesianGrid3D& grid, const GridPair3D& pair,
    const NativeNurbsSurface3D& surface, const LaplaceCorrectionSupport3D& support,
    std::vector<RestrictResourceAnchor3D> native_trace_anchors,
    TensorProductCoverKind3D kind, RestrictResourceMode3D engine);

// Common geometric setup without a correction-center selector. The old
// resource planner and the opt-in trace-first planner then make independent
// decisions, without paying for two candidate searches.
[[nodiscard]] ResourceGeometryPlan3D build_resource_geometry_skeleton_3d(
    const CartesianGrid3D& grid, const GridPair3D& pair,
    const NativeNurbsSurface3D& surface, const LaplaceCorrectionSupport3D& support,
    std::vector<RestrictResourceAnchor3D> native_trace_anchors,
    TensorProductCoverKind3D kind, bool python_mean_centered_cover = false,
    double cover_endpoint_snap = 0.0);

[[nodiscard]] Eigen::VectorXd resource_constant_trace_3d(
    const ResourceGeometryPlan3D& geometry, const GridPair3D& pair,
    const Eigen::VectorXd& potential, bool interior, bool normal);

struct ResourceBuildStatistics3D {
    RestrictResourceCounts3D planning;
    std::size_t spread_centers = 0;
    std::size_t retained_crossing_centers = 0;
    std::size_t p2_centers = 0;
    std::size_t p2_from_p3 = 0;
    std::size_t endpoint_row_hits = 0;
    std::size_t p2_row_evaluations = 0;
    std::size_t temporary_centers_after_build = 0;
    double planning_seconds = 0.0;
    double spread_seconds = 0.0;
    double restrict_seconds = 0.0;
    // Subsets of spread_seconds: Restrict work performed during Spread.
    // Add these to planning_seconds + restrict_seconds for an equivalent
    // Restrict cost; do not add them again to total setup time.
    double retained_p2_seconds = 0.0;
    double endpoint_p2_seconds = 0.0;
};

struct ResourceBvpOperators3D {
    // Spread follows the C++ Delta_h correction convention. The existing bulk
    // solver is called with -(S*c+known_spread), not +(S*c+known_spread).
    ResourceRestrictSparseMatrix3D S;
    Eigen::VectorXd known_spread;
    ResourceRestrictAssembly3D restrict;
    ResourceRestrictSparseMatrix3D trace_basis;
    ResourceBuildStatistics3D statistics;
};

// All callbacks execute at setup only. Every matvec thereafter uses fixed
// sparse matrices. A changed geometry, allocation, BVP or known function
// requires a fresh build. No shared/global cache and no sampled density fit.
[[nodiscard]] ResourceBvpOperators3D build_resource_bvp_operators_3d(
    const ResourceGeometryPlan3D& geometry, const CartesianGrid3D& grid,
    const GridPair3D& pair, const LaplaceCorrectionSupport3D& support,
    const NativeNurbsDensitySpace3D& density, NativeDensityField3D unknown,
    const ResourceKnownAmbientCallback3D& known, double neumann_mean_removed = 0.0);

[[nodiscard]] double resource_operator_max_difference_3d(
    const ResourceBvpOperators3D& a, const ResourceBvpOperators3D& b);

// Matrix Market files are an explicit opt-in diagnostic, outside setup timing.
void dump_resource_operators_3d(const ResourceBvpOperators3D& operators,
                              const std::string& directory);

} // namespace kfbim::app3d
