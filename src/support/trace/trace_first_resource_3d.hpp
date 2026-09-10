#pragma once

#include "src/support/trace/spread_restrict_resource_3d.hpp"

#include <memory>

namespace kfbim::app3d {

enum class TraceFirstCenterPolicy3D {
    EventCentered,
    TracePolynomialFirst,
    TraceSpreadEventRestrict,
    EventSpreadTraceRestrict,
    // 93-case policy: the certified event chooses the owner. A trace may only
    // replace its polynomial center, on that same analysis/native patch ID.
    OwnerValidatedTraceFirst,
};
enum class TraceFirstEventMode3D { LocalAllEvent, PythonEndpointEvent, Python93LastEvent };
enum class TraceFirstChart3D { PhysicalGraph, ExtendedTube };

struct OwnerValidatedTraceOptions3D {
    double event_radius_h = 1.75;
    double target_radius_h = 3.25;
    // Python93 Grid.cover uses +/-1e-13 in GRID coordinates before
    // floor/ceil. Opt in explicitly; the older Torus path keeps zero.
    double cover_endpoint_snap = 0.0;
};

// Public, deterministic guard for a candidate already selected on the event's
// owner patch. For Spread pass both Cartesian endpoints as targets; Restrict
// passes only its support point. No sheet/neighbor substitution is allowed.
[[nodiscard]] bool owner_validated_trace_candidate_3d(
    const RestrictResourceAnchor3D& trace, const RestrictResourceAnchor3D& event,
    const Eigen::Vector3d& target, double h,
    const OwnerValidatedTraceOptions3D& options = {},
    const Eigen::Vector3d* second_target = nullptr);

struct TraceFirstAnchorRef3D {
    RestrictAnchorSource3D source = RestrictAnchorSource3D::Trace;
    int index = -1;
};
struct TraceFirstSignedTerm3D {
    TraceFirstAnchorRef3D anchor;
    double sign = 0.0;
};

struct TraceFirstGeometryPlan3D {
    ResourceGeometryPlan3D resource;
    TraceFirstCenterPolicy3D center_policy = TraceFirstCenterPolicy3D::TracePolynomialFirst;
    TraceFirstEventMode3D event_mode = TraceFirstEventMode3D::LocalAllEvent;
    // Same indices/order as the input crossing_ops. A false mask deliberately
    // reproduces Python's endpoint-event comparison, never the local baseline.
    std::vector<bool> spread_active;
    std::vector<TraceFirstAnchorRef3D> spread_centers;
    // Aligned with resource.restrict_plan.requests; explicit continuation
    // recipes preserve every certified event in local event-centered mode.
    std::vector<std::vector<TraceFirstSignedTerm3D>> restrict_recipes;
    std::size_t spread_trace_uses = 0;
    std::size_t spread_event_fallbacks = 0;
    std::size_t current_trace_requests = 0;
    std::size_t nearest_trace_requests = 0;
    std::size_t far_trace_requests = 0;
    std::size_t owner_validated_requests = 0;
    std::size_t restrict_event_fallbacks = 0;
    std::size_t certified_support_paths = 0;
    std::size_t support_path_cache_hits = 0;
    std::size_t multi_event_support_paths = 0;
    std::size_t python93_discarded_spread_operations = 0;
    double maximum_trace_event_distance_h = 0.0;
    double maximum_trace_target_distance_h = 0.0;
    bool support_paths_certified = false;
};

struct TracePolynomialCatalogStatistics3D {
    std::size_t p3_centers = 0;
    std::size_t p2_centers = 0;
    std::size_t p2_from_p3 = 0;
    std::size_t p2_row_evaluations = 0;
    std::size_t p3_row_evaluations = 0;
    std::size_t row_cache_hits = 0;
    std::size_t projection_not_converged = 0;
    double maximum_projection_residual = 0.0;
};

// Owned by one BVP setup. Stable input-vector indices identify trace anchors.
// A target-grid row is cached by (degree, source, anchor, full grid node).
// This is not a cache of an iteration's evaluated density.
class TracePolynomialCatalog3D {
public:
    TracePolynomialCatalog3D(const NativeNurbsDensitySpace3D& density,
        const std::vector<RestrictResourceAnchor3D>& trace_anchors,
        const std::vector<RestrictResourceAnchor3D>& event_anchors,
        NativeDensityField3D unknown, ResourceKnownAmbientCallback3D known,
        double neumann_mean_removed, TraceFirstChart3D chart, bool reuse_rows = true);
    ~TracePolynomialCatalog3D();
    TracePolynomialCatalog3D(TracePolynomialCatalog3D&&) noexcept;
    TracePolynomialCatalog3D& operator=(TracePolynomialCatalog3D&&) noexcept;
    TracePolynomialCatalog3D(const TracePolynomialCatalog3D&) = delete;
    TracePolynomialCatalog3D& operator=(const TracePolynomialCatalog3D&) = delete;

    void prepare(int degree, TraceFirstAnchorRef3D anchor);
    [[nodiscard]] ResourceAffineRow3D evaluate(int degree, TraceFirstAnchorRef3D anchor,
        const Eigen::Vector3d& target, int full_grid_id = -1);
    [[nodiscard]] NativeDensityC0Stencil3D trace_value_basis(int trace_id) const;
    [[nodiscard]] const TracePolynomialCatalogStatistics3D& statistics() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Trace-first mapping is visit-dependent: choose q when distance <=2.25h,
// otherwise choose the nearest trace in q's patch and its immediate declared
// patch neighbours. Only AFTER this choice are anchor/node requests merged.
// The rule does not certify a path or implement DDA cut-cell topology.
// OwnerValidatedTraceFirst is independent: it queries certified paths, then
// uses only the event-owner patch and the configured event/target radii.
[[nodiscard]] TraceFirstGeometryPlan3D build_trace_first_geometry_plan_3d(
    const CartesianGrid3D& grid, const GridPair3D& pair,
    const NativeNurbsSurface3D& surface, const LaplaceCorrectionSupport3D& support,
    std::vector<RestrictResourceAnchor3D> native_trace_anchors,
    TensorProductCoverKind3D kind, RestrictResourceMode3D engine,
    TraceFirstCenterPolicy3D policy = TraceFirstCenterPolicy3D::TracePolynomialFirst,
    TraceFirstEventMode3D event_mode = TraceFirstEventMode3D::LocalAllEvent,
    OwnerValidatedTraceOptions3D owner_options = {});

struct TraceFirstBuildStatistics3D {
    TracePolynomialCatalogStatistics3D catalog;
    std::size_t active_spread_operations = 0;
};

[[nodiscard]] ResourceBvpOperators3D build_trace_first_bvp_operators_3d(
    const TraceFirstGeometryPlan3D& geometry, const CartesianGrid3D& grid,
    const GridPair3D& pair, const LaplaceCorrectionSupport3D& support,
    const NativeNurbsDensitySpace3D& density, NativeDensityField3D unknown,
    const ResourceKnownAmbientCallback3D& known, double neumann_mean_removed = 0.0,
    TraceFirstChart3D chart = TraceFirstChart3D::PhysicalGraph,
    TraceFirstBuildStatistics3D* trace_statistics = nullptr);

} // namespace kfbim::app3d
