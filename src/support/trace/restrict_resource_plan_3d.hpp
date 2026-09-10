#pragma once

#include "src/support/trace/tensor_product_cover_restrict_3d.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace kfbim::app3d {

enum class RestrictResourceMode3D {
    ReferencePerVisit,
    ReuseBySheetNode,
    GuardedReuseBySheetNode,
};

enum class RestrictAnchorSource3D { Trace, SpreadCrossing };

// sheet_id is a physical smooth sheet, NOT a closed-component identifier.
// Indices in the immutable input vectors are stable anchor identities.  Native
// parameters are retained verbatim, never rounded/merged as a cache key.
struct RestrictResourceAnchor3D {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    int patch_id = -1;
    double u = 0.0;
    double v = 0.0;
    int sheet_id = -1;
    int crossing_id = -1; // Required only for SpreadCrossing anchors.
};

struct RestrictSupportVisit3D {
    int trace_center_id = -1; // Index in trace_anchors.
    bool desired_inside = false;
    int grid_full_id = -1; // CartesianGrid3D::index, NOT a bulk interior ID.
    Eigen::Vector3d grid_point = Eigen::Vector3d::Zero();
    bool actual_inside = false;
};

struct RestrictResourceRequest3D {
    int sheet_id = -1;
    int grid_full_id = -1;
    Eigen::Vector3d grid_point = Eigen::Vector3d::Zero();
    RestrictAnchorSource3D anchor_source = RestrictAnchorSource3D::Trace;
    int anchor_index = -1;
    double distance = 0.0;
    double nearest_trace_distance = 0.0;
    bool too_far = false; // distance > 2.25 h; diagnostic, not rejection.
    bool competing_sheet = false; // other trace distance < distance - .15 h.
    // This is only a request to the caller's independently validated fallback.
    // Neither a near anchor nor an unflagged request certifies a support path.
    bool fallback_required = false;
    // Explicit all-event continuation, including same-label multi-event paths.
    // The row already includes transition signs; do not multiply by label delta.
    bool signed_extension = false;
};

struct RestrictResourceCounts3D {
    std::size_t support_visits = 0;
    std::size_t wrong_side_visits = 0;
    std::size_t requests = 0;
    std::size_t trace_requests = 0;
    std::size_t spread_requests = 0;
    std::size_t unique_trace_centers = 0;
    std::size_t unique_spread_centers = 0;
    std::size_t too_far_requests = 0;
    std::size_t competing_sheet_requests = 0;
    std::size_t fallback_requests = 0;
};

struct RestrictResourcePlan3D {
    RestrictResourceMode3D mode = RestrictResourceMode3D::ReuseBySheetNode;
    std::vector<int> visit_to_request; // -1 means no wrong-side correction.
    std::vector<RestrictResourceRequest3D> requests;
    std::vector<int> required_crossing_ids; // Stable, unique, first-use order.
    RestrictResourceCounts3D counts;
};

// Operator-setup-owned data only. A KD tree per sheet avoids scanning all
// anchors per visit. Reference and reuse use exactly the same selection rule:
// nearest same-sheet trace; if >1.5h, use a strictly nearer same-sheet crossing.
// Equal distances use the first input anchor, without a floating-point tolerance.
// This planner does not inspect path events: callers must retain their all-event
// correction/fallback when required, including paths with multiple crossings.
[[nodiscard]] RestrictResourcePlan3D build_restrict_resource_plan_3d(
    const std::vector<RestrictResourceAnchor3D>& trace_anchors,
    const std::vector<RestrictResourceAnchor3D>& spread_anchors,
    const std::vector<RestrictSupportVisit3D>& visits,
    double h,
    RestrictResourceMode3D mode = RestrictResourceMode3D::ReuseBySheetNode);

struct SharedSideCoverRestrictSide3D {
    bool desired_inside = false;
    std::array<double, 3> signed_rho{};
    std::array<Eigen::Vector3d, 3> sample_points{};
    std::vector<int> grid_ids; // Full Cartesian IDs shared by all three samples.
    Eigen::Matrix<double, 3, Eigen::Dynamic> sampling_weights;
    // The appropriate three entries of the global six-point cubic recovery.
    Eigen::RowVector3d value_recovery = Eigen::RowVector3d::Zero();
    Eigen::RowVector3d normal_recovery = Eigen::RowVector3d::Zero(); // includes /h
    // Convenience contractions, valid after correcting to one common field.
    Eigen::RowVectorXd value_weights;
    Eigen::RowVectorXd normal_weights;
};

struct SharedSideCoverRestrictStencil3D {
    TensorProductCoverKind3D kind = TensorProductCoverKind3D::Q27Cover3;
    double h = 0.0;
    std::array<SharedSideCoverRestrictSide3D, 2> sides; // Interior, exterior.
    Eigen::Matrix<double, 4, 6> cubic_pseudoinverse;
};

// rho=(-1.5,-.75,-.5,+.5,+.75,+1.5). Each side shares one Cartesian
// Q27/Q64 tensor cover. Per axis, prefer a cover enclosing all three samples,
// centred about their coordinate-range midpoint; clamp to the node box.
// Samples themselves must be inside the box, as in the existing tensor API.
// A six-by-four Vandermonde pseudoinverse recovers value and outward derivative.
[[nodiscard]] SharedSideCoverRestrictStencil3D
build_shared_side_cover_restrict_stencil_3d(
    const CartesianGrid3D& grid,
    const Eigen::Vector3d& trace_point,
    const Eigen::Vector3d& outward_normal,
    TensorProductCoverKind3D kind,
    bool python_mean_centered_cover = false,
    double cover_endpoint_snap = 0.0);

} // namespace kfbim::app3d
