#include "src/support/trace/restrict_crossing_selector_3d.hpp"
#include "src/geometry/nurbs_surface_intersector_3d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kfbim::app3d {
namespace {

void validate_tolerance(double tolerance)
{
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw std::invalid_argument(
            "restrict crossing selection requires a finite nonnegative "
            "distance tolerance");
    }
}

bool distance_tied(double candidate,
                   double best,
                   double tolerance)
{
    return std::abs(candidate - best) <= tolerance;
}

void validate_segment_root(
    const geometry3d::NurbsSurfaceCrossing3D& root)
{
    if (root.patch_index < 0 || !root.point.allFinite()) {
        throw std::invalid_argument(
            "segment crossing selection received an invalid root");
    }
}

void validate_topology(const G1PatchTopology3D& topology)
{
    if (topology.surface_component_by_patch.empty()
        || topology.g1_neighbors.size()
               != topology.surface_component_by_patch.size()) {
        throw std::invalid_argument(
            "G1 gridline crossing selection has inconsistent surface "
            "metadata");
    }
    for (int patch = 0;
         patch < static_cast<int>(topology.g1_neighbors.size()); ++patch) {
        if (topology.surface_component_by_patch[
                static_cast<std::size_t>(patch)] < 0) {
            throw std::invalid_argument(
                "G1 topology contains an invalid surface component");
        }
        for (int neighbor :
             topology.g1_neighbors[static_cast<std::size_t>(patch)]) {
            if (neighbor < 0
                || neighbor
                       >= static_cast<int>(
                           topology.g1_neighbors.size())) {
                throw std::invalid_argument(
                    "G1 topology contains an invalid neighbor");
            }
        }
    }
}

void validate_gridline_record(
    const CartesianGridlineCrossingRecord3D& record,
    const G1PatchTopology3D& topology)
{
    if (record.payload_index < 0 || record.patch_index < 0
        || record.patch_index
               >= static_cast<int>(
                   topology.surface_component_by_patch.size())
        || record.surface_component < 0 || !record.point.allFinite()) {
        throw std::invalid_argument(
            "G1 gridline crossing selection received an invalid record");
    }
    if (topology.surface_component_by_patch[
            static_cast<std::size_t>(record.patch_index)]
        != record.surface_component) {
        throw std::invalid_argument(
            "gridline crossing record component disagrees with its patch");
    }
}

std::vector<int> g1_patch_component(const G1PatchTopology3D& topology,
                                    int seed)
{
    std::vector<bool> visited(topology.g1_neighbors.size(), false);
    std::vector<int> pending{seed};
    std::vector<int> result;
    visited[static_cast<std::size_t>(seed)] = true;
    for (std::size_t cursor = 0; cursor < pending.size(); ++cursor) {
        const int patch = pending[cursor];
        result.push_back(patch);
        for (int neighbor :
             topology.g1_neighbors[static_cast<std::size_t>(patch)]) {
            if (!visited[static_cast<std::size_t>(neighbor)]) {
                visited[static_cast<std::size_t>(neighbor)] = true;
                pending.push_back(neighbor);
            }
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace

SegmentCrossingSelection3D
select_segment_crossing_nearest_grid_point_3d(
    const geometry3d::NurbsSurfaceIntersectionResult3D& intersection,
    const Eigen::Vector3d& grid_point,
    double distance_tie_tolerance)
{
    validate_tolerance(distance_tie_tolerance);
    if (!grid_point.allFinite()) {
        throw std::invalid_argument(
            "segment crossing selection grid point must be finite");
    }

    SegmentCrossingSelection3D result;
    result.crossing_count =
        static_cast<int>(intersection.crossings.size());
    result.multiple_crossings = intersection.crossings.size() > 1;
    result.overlap_detected = intersection.overlap_detected;
    result.unresolved_candidates =
        intersection.diagnostics.unresolved_candidates;
    result.ambiguous_root_clusters =
        intersection.diagnostics.ambiguous_root_clusters;
    for (const auto& root : intersection.crossings)
        validate_segment_root(root);

    if (intersection.overlap_detected) {
        result.kind = SegmentCrossingSelectionKind3D::OverlapFallback;
        return result;
    }
    if (intersection.crossings.empty()) {
        result.kind = SegmentCrossingSelectionKind3D::NoCrossing;
        return result;
    }

    // Determine the strict minimum first, then collect its tolerance cluster.
    // A one-pass tolerant minimum is order-dependent when distances form a
    // non-transitive tolerance chain.
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < result.crossing_count; ++i) {
        const double distance =
            (intersection.crossings[static_cast<std::size_t>(i)].point
             - grid_point)
                .norm();
        if (distance < best)
            best = distance;
    }
    int best_index = -1;
    int tie_count = 0;
    for (int i = 0; i < result.crossing_count; ++i) {
        const double distance =
            (intersection.crossings[static_cast<std::size_t>(i)].point
             - grid_point)
                .norm();
        if (distance_tied(distance, best, distance_tie_tolerance)) {
            if (best_index < 0)
                best_index = i;
            ++tie_count;
        }
    }
    result.distance_to_grid_point = best;
    result.nearest_tie_count = tie_count;
    if (tie_count != 1) {
        result.kind =
            SegmentCrossingSelectionKind3D::AmbiguousNearestFallback;
        return result;
    }

    result.crossing_index = best_index;
    result.kind = SegmentCrossingSelectionKind3D::Selected;
    const auto& selected =
        intersection.crossings[static_cast<std::size_t>(best_index)];
    result.selected_feature_edge_contact = selected.feature_edge_contact;
    result.selected_transversality_reliable =
        std::isfinite(selected.transversality)
        && std::isfinite(selected.reliable_transversality_tolerance)
        && selected.reliable_transversality_tolerance > 0.0
        && selected.transversality
               > selected.reliable_transversality_tolerance;
    return result;
}

SegmentPhysicalEventSequence3D
select_certified_segment_physical_events_3d(
    const geometry3d::NurbsSurfaceIntersectionResult3D& intersection,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end,
    double distance_tie_tolerance,
    const std::function<bool(const Eigen::Vector3d&)>& exact_inside)
{
    validate_tolerance(distance_tie_tolerance);
    if (!segment_start.allFinite() || !segment_end.allFinite()) {
        throw std::invalid_argument(
            "segment event selection endpoints must be finite");
    }
    const Eigen::Vector3d displacement = segment_end - segment_start;
    const double length = displacement.norm();
    if (!std::isfinite(length) || !(length > distance_tie_tolerance)) {
        throw std::invalid_argument(
            "segment event selection requires a nondegenerate segment");
    }
    const Eigen::Vector3d direction = displacement / length;

    SegmentPhysicalEventSequence3D result;
    result.input_crossing_count =
        static_cast<int>(intersection.crossings.size());
    if (intersection.overlap_detected) {
        result.kind = SegmentPhysicalEventSequenceKind3D::Overlap;
        return result;
    }
    if (intersection.diagnostics.unresolved_candidates != 0
        || intersection.diagnostics.ambiguous_root_clusters != 0
        || !intersection.diagnostics
                .unresolved_longitudinal_intervals.empty()) {
        result.kind =
            SegmentPhysicalEventSequenceKind3D::IncompleteRootSet;
        return result;
    }
    if (intersection.crossings.empty()) {
        result.kind = SegmentPhysicalEventSequenceKind3D::NoCrossing;
        return result;
    }

    struct OrderedRoot {
        int index = -1;
        double distance = 0.0;
    };
    std::vector<OrderedRoot> ordered;
    ordered.reserve(intersection.crossings.size());
    for (int index = 0;
         index < static_cast<int>(intersection.crossings.size()); ++index) {
        const auto& crossing =
            intersection.crossings[static_cast<std::size_t>(index)];
        validate_segment_root(crossing);
        const Eigen::Vector3d offset = crossing.point - segment_start;
        const double distance = offset.dot(direction);
        const double off_line = (offset - distance * direction).norm();
        if (!std::isfinite(distance) || !std::isfinite(off_line)
            || distance < -distance_tie_tolerance
            || distance > length + distance_tie_tolerance
            || off_line > 4.0 * distance_tie_tolerance) {
            result.kind =
                SegmentPhysicalEventSequenceKind3D::AmbiguousPhysicalCluster;
            return result;
        }
        ordered.push_back({index, std::clamp(distance, 0.0, length)});
    }
    std::sort(
        ordered.begin(), ordered.end(),
        [](const OrderedRoot& first, const OrderedRoot& second) {
            if (first.distance != second.distance)
                return first.distance < second.distance;
            return first.index < second.index;
        });

    // intersect_segment has already canonicalized declared seam owners into
    // crossing.owners.  Therefore every crossing here is one distinct
    // physical event.  Never distance-merge separate crossings: a torus (or
    // another folded surface) can have two certified roots much closer than
    // a geometry tolerance.
    for (std::size_t ordered_index = 0;
         ordered_index < ordered.size(); ++ordered_index) {
        const OrderedRoot& seed = ordered[ordered_index];
        const auto& crossing = intersection.crossings[
            static_cast<std::size_t>(seed.index)];
        SegmentPhysicalCrossingEvent3D event;
        event.edge_parameter = seed.distance / length;
        event.point = crossing.point;
        event.component = crossing.component;
        event.crossing_indices.push_back(seed.index);
        event.feature_edge_contact = crossing.feature_edge_contact;
        int event_sign = 0;
        bool unreliable_owner = false;
        bool inconsistent_owner_sign = false;
        const auto accept_owner = [&](const Eigen::Vector3d& normal,
                                      double transversality,
                                      double reliable_tolerance) {
            if (!normal.allFinite()
                || !(normal.norm() > 0.5)
                || !std::isfinite(transversality)
                || !std::isfinite(reliable_tolerance)
                || !(reliable_tolerance > 0.0)
                || !(transversality > reliable_tolerance)) {
                return 0;
            }
            const double signed_transversality =
                normal.normalized().dot(direction);
            if (!std::isfinite(signed_transversality)
                || !(std::abs(signed_transversality)
                     > reliable_tolerance)) {
                return 0;
            }
            return signed_transversality < 0.0 ? 1 : -1;
        };

        if (crossing.owners.empty()) {
            const int owner_sign = accept_owner(
                crossing.normal, crossing.transversality,
                crossing.reliable_transversality_tolerance);
            if (owner_sign == 0) {
                unreliable_owner = true;
            } else {
                event_sign = owner_sign;
            }
        } else {
            for (const auto& owner : crossing.owners) {
                const int owner_sign = accept_owner(
                    owner.normal, owner.transversality,
                    owner.reliable_transversality_tolerance);
                if (owner_sign == 0) {
                    unreliable_owner = true;
                    continue;
                }
                if (event_sign != 0 && owner_sign != event_sign) {
                    inconsistent_owner_sign = true;
                    continue;
                }
                event_sign = owner_sign;
            }
        }

        const bool feature_or_multi_owner =
            crossing.feature_edge_contact || crossing.owners.size() > 1;
        const bool needs_exact_side_classification =
            feature_or_multi_owner
            && (unreliable_owner || inconsistent_owner_sign);
        if (needs_exact_side_classification && exact_inside) {
            const double left_boundary = ordered_index == 0
                ? 0.0
                : ordered[ordered_index - 1].distance;
            const double right_boundary =
                ordered_index + 1 == ordered.size()
                ? length
                : ordered[ordered_index + 1].distance;
            const double before_distance =
                left_boundary + 0.5 * (seed.distance - left_boundary);
            const double after_distance =
                seed.distance + 0.5 * (right_boundary - seed.distance);

            // Strict inequalities are intentional.  In particular, two
            // roots that are closer than distance_tie_tolerance are still
            // distinct physical events, and neither probe may step across
            // the neighboring root.  If floating-point spacing cannot
            // represent both one-sided probes, classification fails closed.
            if (!(left_boundary < before_distance
                  && before_distance < seed.distance
                  && seed.distance < after_distance
                  && after_distance < right_boundary)) {
                result.kind = SegmentPhysicalEventSequenceKind3D::
                    InconsistentContinuationSign;
                result.events.clear();
                return result;
            }

            bool inside_before = false;
            bool inside_after = false;
            try {
                inside_before = exact_inside(
                    segment_start + before_distance * direction);
                inside_after = exact_inside(
                    segment_start + after_distance * direction);
            } catch (...) {
                result.kind = SegmentPhysicalEventSequenceKind3D::
                    InconsistentContinuationSign;
                result.events.clear();
                return result;
            }
            if (inside_before == inside_after) {
                // A certified feature contact does not change the physical
                // branch.  Retain it in the ordered event sequence with zero
                // continuation so callers can audit it without inventing an
                // entering/leaving jump.
                event_sign = 0;
            } else {
                event_sign = static_cast<int>(inside_after)
                           - static_cast<int>(inside_before);
            }
        } else if (unreliable_owner) {
            result.kind = SegmentPhysicalEventSequenceKind3D::
                NonTransverseContact;
            result.events.clear();
            return result;
        } else if (inconsistent_owner_sign) {
            result.kind = SegmentPhysicalEventSequenceKind3D::
                InconsistentContinuationSign;
            result.events.clear();
            return result;
        }
        event.continuation_sign = event_sign;
        result.events.push_back(std::move(event));
    }

    result.physical_event_count = static_cast<int>(result.events.size());
    result.kind = SegmentPhysicalEventSequenceKind3D::Certified;
    return result;
}

SegmentEndpointPartition3D
partition_certified_segment_endpoint_3d(
    const geometry3d::NurbsSurfaceIntersectionResult3D& intersection,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& exact_endpoint,
    const Eigen::Vector3d& certified_segment_end,
    int endpoint_patch,
    double endpoint_u,
    double endpoint_v,
    double distance_tie_tolerance,
    double parameter_tolerance)
{
    validate_tolerance(distance_tie_tolerance);
    validate_tolerance(parameter_tolerance);
    if (!segment_start.allFinite() || !exact_endpoint.allFinite()
        || !certified_segment_end.allFinite()
        || endpoint_patch < 0 || !std::isfinite(endpoint_u)
        || !std::isfinite(endpoint_v)) {
        throw std::invalid_argument(
            "endpoint partition requires finite native endpoint metadata");
    }
    const Eigen::Vector3d displacement =
        certified_segment_end - segment_start;
    const double length = displacement.norm();
    if (!std::isfinite(length) || !(length > distance_tie_tolerance)) {
        throw std::invalid_argument(
            "endpoint partition requires a nondegenerate segment");
    }
    const Eigen::Vector3d direction = displacement / length;
    const Eigen::Vector3d endpoint_relative = exact_endpoint - segment_start;
    const double endpoint_distance = endpoint_relative.dot(direction);
    const double endpoint_off_line =
        (endpoint_relative - endpoint_distance * direction).norm();
    if (!std::isfinite(endpoint_distance)
        || !std::isfinite(endpoint_off_line)
        || endpoint_off_line > 4.0 * distance_tie_tolerance
        || !(endpoint_distance > distance_tie_tolerance)
        || !(endpoint_distance
             < length - distance_tie_tolerance)) {
        throw std::invalid_argument(
            "endpoint partition requires the exact endpoint strictly inside the certified segment");
    }

    SegmentEndpointPartition3D result;
    result.input_crossing_count =
        static_cast<int>(intersection.crossings.size());
    if (intersection.overlap_detected) {
        result.kind = SegmentEndpointPartitionKind3D::Overlap;
        return result;
    }
    if (intersection.diagnostics.unresolved_candidates != 0
        || intersection.diagnostics.ambiguous_root_clusters != 0
        || !intersection.diagnostics
                .unresolved_longitudinal_intervals.empty()) {
        result.kind = SegmentEndpointPartitionKind3D::IncompleteRootSet;
        return result;
    }

    const auto owner_matches_endpoint =
        [&](const geometry3d::NurbsSurfaceCrossing3D& crossing) {
            const auto matches = [&](int patch, double u, double v) {
                return patch == endpoint_patch && std::isfinite(u)
                    && std::isfinite(v)
                    && std::abs(u - endpoint_u) <= parameter_tolerance
                    && std::abs(v - endpoint_v) <= parameter_tolerance;
            };
            if (matches(crossing.patch_index, crossing.u, crossing.v))
                return true;
            return std::any_of(
                crossing.owners.begin(), crossing.owners.end(),
                [&](const geometry3d::NurbsSurfaceRootOwner3D& owner) {
                    return matches(owner.patch_index, owner.u, owner.v);
                });
        };

    for (int index = 0;
         index < static_cast<int>(intersection.crossings.size()); ++index) {
        const auto& crossing =
            intersection.crossings[static_cast<std::size_t>(index)];
        validate_segment_root(crossing);
        const Eigen::Vector3d relative = crossing.point - segment_start;
        const double distance = relative.dot(direction);
        const double off_line = (relative - distance * direction).norm();
        if (!std::isfinite(distance) || !std::isfinite(off_line)
            || distance < -distance_tie_tolerance
            || distance > length + distance_tie_tolerance
            || off_line > 4.0 * distance_tie_tolerance) {
            result.kind =
                SegmentEndpointPartitionKind3D::AmbiguousRootGeometry;
            result.open_crossing_indices.clear();
            result.post_endpoint_crossing_indices.clear();
            result.endpoint_crossing_index = -1;
            return result;
        }

        const bool endpoint_owner = owner_matches_endpoint(crossing);
        const bool endpoint_geometry =
            (crossing.point - exact_endpoint).norm()
                <= distance_tie_tolerance
            && std::abs(distance - endpoint_distance)
                   <= distance_tie_tolerance;
        if (endpoint_owner != endpoint_geometry) {
            // A native endpoint owner away from the exact endpoint, or an
            // endpoint-position root without that owner, is not safe to
            // discard as the known trace event.
            if (endpoint_owner) {
                result.kind =
                    SegmentEndpointPartitionKind3D::AmbiguousRootGeometry;
                result.open_crossing_indices.clear();
                result.post_endpoint_crossing_indices.clear();
                result.endpoint_crossing_index = -1;
                return result;
            }
            if (distance < endpoint_distance) {
                result.open_crossing_indices.push_back(index);
            } else if (distance > endpoint_distance) {
                result.post_endpoint_crossing_indices.push_back(index);
            } else {
                result.kind =
                    SegmentEndpointPartitionKind3D::AmbiguousRootGeometry;
                result.open_crossing_indices.clear();
                result.post_endpoint_crossing_indices.clear();
                return result;
            }
            continue;
        }
        if (!endpoint_owner) {
            if (distance < endpoint_distance) {
                result.open_crossing_indices.push_back(index);
            } else if (distance > endpoint_distance) {
                result.post_endpoint_crossing_indices.push_back(index);
            } else {
                result.kind =
                    SegmentEndpointPartitionKind3D::AmbiguousRootGeometry;
                result.open_crossing_indices.clear();
                result.post_endpoint_crossing_indices.clear();
                return result;
            }
            continue;
        }
        if (result.endpoint_crossing_index >= 0) {
            result.kind =
                SegmentEndpointPartitionKind3D::AmbiguousExactEndpoint;
            result.open_crossing_indices.clear();
            result.post_endpoint_crossing_indices.clear();
            result.endpoint_crossing_index = -1;
            return result;
        }
        result.endpoint_crossing_index = index;
    }

    if (result.endpoint_crossing_index < 0) {
        result.kind = SegmentEndpointPartitionKind3D::MissingExactEndpoint;
        result.open_crossing_indices.clear();
        return result;
    }
    result.kind = SegmentEndpointPartitionKind3D::Certified;
    return result;
}

GridlineCrossingSelection3D
select_nearest_gridline_crossing_on_g1_sheet_3d(
    const geometry3d::NurbsSurfaceCrossing3D& segment_root,
    const G1PatchTopology3D& topology,
    const std::vector<CartesianGridlineCrossingRecord3D>& records,
    double distance_tie_tolerance)
{
    validate_tolerance(distance_tie_tolerance);
    validate_topology(topology);
    validate_segment_root(segment_root);
    if (segment_root.patch_index
            >= static_cast<int>(topology.g1_neighbors.size())
        || segment_root.component < 0) {
        throw std::invalid_argument(
            "G1 gridline crossing selection received invalid root "
            "topology");
    }
    if (topology.surface_component_by_patch[
            static_cast<std::size_t>(segment_root.patch_index)]
        != segment_root.component) {
        throw std::invalid_argument(
            "segment root component disagrees with its patch");
    }
    for (const auto& record : records)
        validate_gridline_record(record, topology);

    GridlineCrossingSelection3D result;
    result.input_record_count = static_cast<int>(records.size());
    const std::vector<int> sheet =
        g1_patch_component(topology, segment_root.patch_index);

    double best = std::numeric_limits<double>::infinity();
    int best_index = -1;
    int tie_count = 0;
    for (int i = 0; i < static_cast<int>(records.size()); ++i) {
        const auto& record = records[static_cast<std::size_t>(i)];
        if (record.surface_component != segment_root.component)
            continue;
        ++result.same_surface_component_count;
        if (!std::binary_search(
                sheet.begin(), sheet.end(), record.patch_index)) {
            continue;
        }
        ++result.same_g1_sheet_count;
        const double distance = (record.point - segment_root.point).norm();
        if (distance < best - distance_tie_tolerance) {
            best = distance;
            best_index = i;
            tie_count = 1;
        } else if (distance_tied(
                       distance, best, distance_tie_tolerance)) {
            // Ties are harmless here: all candidates have already been
            // certified to lie on the same G1 sheet.  Keep the first record
            // deterministically and expose the tie count as a diagnostic.
            ++tie_count;
        }
    }

    if (result.same_surface_component_count == 0) {
        result.kind = GridlineCrossingSelectionKind3D::
            NoSameSurfaceComponentFallback;
        return result;
    }
    if (result.same_g1_sheet_count == 0) {
        result.kind =
            GridlineCrossingSelectionKind3D::NoSameG1SheetFallback;
        return result;
    }

    result.record_index = best_index;
    result.payload_index =
        records[static_cast<std::size_t>(best_index)].payload_index;
    result.kind = GridlineCrossingSelectionKind3D::Selected;
    result.nearest_tie_count = tie_count;
    result.distance_to_segment_root = best;
    return result;
}

} // namespace kfbim::app3d
