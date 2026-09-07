#pragma once

#include <Eigen/Dense>

#include <functional>
#include <cstdint>
#include <vector>

namespace kfbim::app3d {

namespace detail {
// Intentionally empty; keeps this header independent of the native-surface
// triangulation stack.
}

} // namespace kfbim::app3d

namespace kfbim::geometry3d {
struct NurbsSurfaceCrossing3D;
struct NurbsSurfaceIntersectionResult3D;
}

namespace kfbim::app3d {

// Selects the physical intersection nearest to the Cartesian support node.
// A multiple-root segment is valid when its nearest root is unique.  A tie is
// deliberately left unresolved because coincident roots can represent the two
// sides of a C0 patch seam.
enum class SegmentCrossingSelectionKind3D {
    Selected,
    NoCrossing,
    OverlapFallback,
    AmbiguousNearestFallback
};

struct SegmentCrossingSelection3D {
    int crossing_index = -1;
    SegmentCrossingSelectionKind3D kind =
        SegmentCrossingSelectionKind3D::NoCrossing;
    int crossing_count = 0;
    int nearest_tie_count = 0;
    double distance_to_grid_point = 0.0;
    bool multiple_crossings = false;
    bool overlap_detected = false;
    int unresolved_candidates = 0;
    int ambiguous_root_clusters = 0;
    bool selected_feature_edge_contact = false;
    bool selected_transversality_reliable = false;
};

[[nodiscard]] SegmentCrossingSelection3D
select_segment_crossing_nearest_grid_point_3d(
    const geometry3d::NurbsSurfaceIntersectionResult3D& intersection,
    const Eigen::Vector3d& grid_point,
    double distance_tie_tolerance);

// A support-to-trace continuation is a path operation, not a nearest-root
// operation.  Every physical transverse event on the path contributes once.
// crossing_indices identifies the already-canonical physical crossing.  Its
// complete parametric owner set remains in NurbsSurfaceCrossing3D::owners;
// separate crossings are never merged by a distance tolerance.
// continuation_sign is
//
//     inside_after - inside_before,
//
// for traversal from segment_start to segment_end.  With outward-oriented
// surface normals this is -sign(normal dot segment_direction).
struct SegmentPhysicalCrossingEvent3D {
    std::vector<int> crossing_indices;
    double edge_parameter = 0.0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    int component = -1;
    int continuation_sign = 0;
    bool feature_edge_contact = false;
    // Path-local certificate identities; zero denotes a legacy event.
    std::uint64_t native_event_id = 0;
    std::uint64_t native_proof_id = 0;
};

enum class SegmentPhysicalEventSequenceKind3D {
    Certified,
    NoCrossing,
    Overlap,
    IncompleteRootSet,
    AmbiguousPhysicalCluster,
    NonTransverseContact,
    InconsistentContinuationSign
};

struct SegmentPhysicalEventSequence3D {
    SegmentPhysicalEventSequenceKind3D kind =
        SegmentPhysicalEventSequenceKind3D::NoCrossing;
    std::vector<SegmentPhysicalCrossingEvent3D> events;
    int input_crossing_count = 0;
    int physical_event_count = 0;
};

// Partitions a complete closed-segment intersection into the known trace
// endpoint and every physical event strictly before it.  Endpoint identity is
// established by both position on the segment and the native patch/parameter
// owner; a merely nearby root is retained as an open event.  The routine
// fails closed unless the intersector certified the entire closed segment and
// returned exactly one canonical owner of the known endpoint.
enum class SegmentEndpointPartitionKind3D {
    Certified,
    Overlap,
    IncompleteRootSet,
    MissingExactEndpoint,
    AmbiguousExactEndpoint,
    AmbiguousRootGeometry
};

struct SegmentEndpointPartition3D {
    SegmentEndpointPartitionKind3D kind =
        SegmentEndpointPartitionKind3D::IncompleteRootSet;
    std::vector<int> open_crossing_indices;
    std::vector<int> post_endpoint_crossing_indices;
    int endpoint_crossing_index = -1;
    int input_crossing_count = 0;
};

[[nodiscard]] SegmentEndpointPartition3D
partition_certified_segment_endpoint_3d(
    const geometry3d::NurbsSurfaceIntersectionResult3D& intersection,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& exact_endpoint,
    const Eigen::Vector3d& certified_segment_end,
    int endpoint_patch,
    double endpoint_u,
    double endpoint_v,
    double distance_tie_tolerance,
    double parameter_tolerance);

// Result of advancing an inside/outside label through the certified events on
// a half-open support-to-trace segment.  The trace endpoint itself is omitted
// from the input sequence.  ExactEndpointRequired is therefore an instruction
// to add that known endpoint event; it is never a fallback for an uncertified
// or inconsistent open-segment root set.
enum class OpenSegmentContinuationKind3D {
    Complete,
    ExactEndpointRequired,
    UncertifiedSequence,
    InconsistentSequenceMetadata,
    InvalidEventOrder,
    InvalidContinuation
};

struct OpenSegmentContinuation3D {
    OpenSegmentContinuationKind3D kind =
        OpenSegmentContinuationKind3D::UncertifiedSequence;
    int state_after_open_events = -1;
    int required_endpoint_sign = 0;
    int processed_event_count = 0;

    [[nodiscard]] bool exact_endpoint_required() const noexcept
    {
        return kind
            == OpenSegmentContinuationKind3D::ExactEndpointRequired;
    }
};

// Applies every event sign to start_inside and compares the resulting branch
// with desired_inside.  Certified events must be strictly ordered in the
// half-open interval [0,1), carry sign -1/0/+1, and preserve a valid binary
// state.  Zero is an exactly classified feature contact and does not change
// the branch.
// NoCrossing is also accepted, but only with an empty event list.  All other
// sequence kinds (including tangent/contact and incomplete root sets) fail
// closed and can never request endpoint completion.
[[nodiscard]] OpenSegmentContinuation3D
evaluate_open_segment_continuation_3d(
    const SegmentPhysicalEventSequence3D& open_sequence,
    bool start_inside,
    bool desired_inside);

// Builds a strictly ordered, all-event continuation sequence.  The routine
// fails closed when the intersector did not certify a complete discrete root
// set.  Smooth single-owner roots are always certified from their oriented
// normal.  A feature/multi-owner event whose normal signs are unreliable or
// inconsistent may instead be certified by exact_inside: the callback is
// sampled strictly before and after the event, without crossing either a
// neighboring event or a segment endpoint.  Equal side labels produce a
// certified contact event with continuation_sign=0: it remains in the ordered
// physical-event audit but contributes no jump correction.  With an empty
// callback the historical normal-only, fail-closed behavior is preserved.
// The routine never falls back to the nearest root.
[[nodiscard]] SegmentPhysicalEventSequence3D
select_certified_segment_physical_events_3d(
    const geometry3d::NurbsSurfaceIntersectionResult3D& intersection,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end,
    double distance_tie_tolerance,
    const std::function<bool(const Eigen::Vector3d&)>& exact_inside = {});

// Lightweight view of a Cartesian grid-line/surface crossing.  payload_index
// is owned by the caller (for example, an index into P2CrossingOwner3D data).
struct CartesianGridlineCrossingRecord3D {
    int payload_index = -1;
    int patch_index = -1;
    int surface_component = -1;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
};

// Minimal topology needed by the selector.  g1_neighbors stores immediate G1
// connections; the selector computes their transitive component.  Keeping
// this input lightweight lets both native NURBS and other surface backends use
// exactly the same C0-safe selection rule.
struct G1PatchTopology3D {
    std::vector<int> surface_component_by_patch;
    std::vector<std::vector<int>> g1_neighbors;
};

enum class GridlineCrossingSelectionKind3D {
    Selected,
    NoSameSurfaceComponentFallback,
    NoSameG1SheetFallback
};

struct GridlineCrossingSelection3D {
    int record_index = -1;
    int payload_index = -1;
    GridlineCrossingSelectionKind3D kind =
        GridlineCrossingSelectionKind3D::NoSameSurfaceComponentFallback;
    int input_record_count = 0;
    int same_surface_component_count = 0;
    int same_g1_sheet_count = 0;
    int nearest_tie_count = 0;
    double distance_to_segment_root = 0.0;
};

// Searches strictly inside the transitive G1-connected patch component of
// segment_root.  C0-adjacent patches are never admitted, even if their record
// is geometrically closer.  A missing candidate is reported to the caller so
// that the caller can choose an explicit fallback policy.
[[nodiscard]] GridlineCrossingSelection3D
select_nearest_gridline_crossing_on_g1_sheet_3d(
    const geometry3d::NurbsSurfaceCrossing3D& segment_root,
    const G1PatchTopology3D& topology,
    const std::vector<CartesianGridlineCrossingRecord3D>& records,
    double distance_tie_tolerance);

} // namespace kfbim::app3d
