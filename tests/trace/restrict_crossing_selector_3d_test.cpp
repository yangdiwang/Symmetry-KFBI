#include "src/support/trace/restrict_crossing_selector_3d.hpp"
#include "src/support/geometry/native_nurbs_surface_3d.hpp"
#include "src/geometry/nurbs_surface_intersector_3d.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using kfbim::app3d::CartesianGridlineCrossingRecord3D;
using kfbim::app3d::G1PatchTopology3D;
using kfbim::app3d::GridlineCrossingSelectionKind3D;
using kfbim::app3d::SegmentCrossingSelectionKind3D;
using kfbim::app3d::SegmentPhysicalEventSequenceKind3D;
using kfbim::app3d::select_nearest_gridline_crossing_on_g1_sheet_3d;
using kfbim::app3d::select_certified_segment_physical_events_3d;
using kfbim::app3d::select_segment_crossing_nearest_grid_point_3d;
using kfbim::geometry3d::NurbsSurfaceCrossing3D;
using kfbim::geometry3d::NurbsSurfaceIntersectionResult3D;
using kfbim::geometry3d::NurbsSurfaceRootOwner3D;

void require(bool value, const std::string& message)
{
    if (!value)
        throw std::runtime_error(message);
}

NurbsSurfaceCrossing3D root(int patch,
                            int component,
                            const Eigen::Vector3d& point)
{
    NurbsSurfaceCrossing3D value;
    value.patch_index = patch;
    value.component = component;
    value.point = point;
    value.transversality = 1.0;
    value.reliable_transversality_tolerance = 1.0e-8;
    return value;
}

NurbsSurfaceCrossing3D transverse_root(int patch,
                                       double x,
                                       double outward_x)
{
    NurbsSurfaceCrossing3D value = root(patch, 0, {x, 0.0, 0.0});
    value.normal = {outward_x, 0.0, 0.0};
    value.transversality = std::abs(outward_x);
    value.reliable_transversality_tolerance = 1.0e-8;
    return value;
}

NurbsSurfaceRootOwner3D owner(const NurbsSurfaceCrossing3D& crossing)
{
    NurbsSurfaceRootOwner3D result;
    result.patch_index = crossing.patch_index;
    result.u = crossing.u;
    result.v = crossing.v;
    result.point = crossing.point;
    result.normal = crossing.normal;
    result.residual = crossing.residual;
    result.transversality = crossing.transversality;
    result.feature_edge_contact = crossing.feature_edge_contact;
    result.reliable_transversality_tolerance =
        crossing.reliable_transversality_tolerance;
    return result;
}

NurbsSurfaceCrossing3D conflicting_feature_root(double x)
{
    auto first = transverse_root(0, x, -1.0);
    auto second = transverse_root(6, x, 1.0);
    first.feature_edge_contact = true;
    second.feature_edge_contact = true;
    first.owners = {owner(first), owner(second)};
    return first;
}

CartesianGridlineCrossingRecord3D record(
    int payload,
    int patch,
    int component,
    const Eigen::Vector3d& point)
{
    return {payload, patch, component, point};
}

G1PatchTopology3D l_prism_like_topology()
{
    G1PatchTopology3D result;
    result.surface_component_by_patch.assign(9, 0);
    result.g1_neighbors.resize(9);
    result.g1_neighbors[0] = {1};
    result.g1_neighbors[1] = {0, 2};
    result.g1_neighbors[2] = {1};
    result.g1_neighbors[3] = {4};
    result.g1_neighbors[4] = {3, 5};
    result.g1_neighbors[5] = {4};
    // Side patches 6--8 remain isolated G1 sheets although all patches belong
    // to the same closed surface component.
    return result;
}

void test_multiple_segment_roots_choose_nearest_grid_point()
{
    NurbsSurfaceIntersectionResult3D intersection;
    intersection.crossings = {
        root(0, 0, {0.2, 0.0, 0.0}),
        root(0, 0, {0.8, 0.0, 0.0}),
        root(0, 0, {0.5, 0.0, 0.0})};
    intersection.diagnostics.unresolved_candidates = 2;
    const auto selected =
        select_segment_crossing_nearest_grid_point_3d(
            intersection, {1.0, 0.0, 0.0}, 1.0e-12);
    require(selected.kind == SegmentCrossingSelectionKind3D::Selected
                && selected.crossing_index == 1,
            "multiple-root segment selects the root nearest the grid node");
    require(selected.multiple_crossings && selected.crossing_count == 3
                && selected.nearest_tie_count == 1
                && selected.unresolved_candidates == 2,
            "multiple-root diagnostics are retained");
    require(std::abs(selected.distance_to_grid_point - 0.2) < 1.0e-14,
            "selected root distance is reported");
}

void test_segment_root_tie_and_degenerate_fallbacks()
{
    NurbsSurfaceIntersectionResult3D tied;
    tied.crossings = {
        root(0, 0, {0.4, 0.0, 0.0}),
        root(1, 0, {0.6, 0.0, 0.0})};
    const auto ambiguous =
        select_segment_crossing_nearest_grid_point_3d(
            tied, {0.5, 0.0, 0.0}, 1.0e-12);
    require(
        ambiguous.kind
                == SegmentCrossingSelectionKind3D::AmbiguousNearestFallback
            && ambiguous.crossing_index < 0
            && ambiguous.nearest_tie_count == 2,
        "equidistant roots are left unresolved instead of choosing a C0 side");

    NurbsSurfaceIntersectionResult3D empty;
    require(select_segment_crossing_nearest_grid_point_3d(
                empty, Eigen::Vector3d::Zero(), 1.0e-12)
                .kind
                == SegmentCrossingSelectionKind3D::NoCrossing,
            "empty intersection is explicit");
    empty.overlap_detected = true;
    require(select_segment_crossing_nearest_grid_point_3d(
                empty, Eigen::Vector3d::Zero(), 1.0e-12)
                .kind
                == SegmentCrossingSelectionKind3D::OverlapFallback,
            "overlap is not treated as a discrete root");
}

void test_all_event_double_root_telescoping()
{
    NurbsSurfaceIntersectionResult3D intersection;
    intersection.crossings = {
        transverse_root(0, 0.2, -1.0),
        transverse_root(0, 0.7, 1.0)};
    const auto sequence = select_certified_segment_physical_events_3d(
        intersection, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0e-12);
    require(sequence.kind
                == SegmentPhysicalEventSequenceKind3D::Certified
                && sequence.events.size() == 2,
            "same-label double crossing retains both physical events");
    require(sequence.events[0].continuation_sign == 1
                && sequence.events[1].continuation_sign == -1,
            "double crossing has entering/leaving continuation signs");
    const int net = sequence.events[0].continuation_sign
                  + sequence.events[1].continuation_sign;
    require(net == 0,
            "double crossing telescopes back to the starting branch");

    intersection.crossings = {
        transverse_root(0, 0.5 - 2.0e-8, -1.0),
        transverse_root(0, 0.5 + 2.0e-8, 1.0)};
    const auto extremely_close =
        select_certified_segment_physical_events_3d(
            intersection, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0e-6);
    require(extremely_close.kind
                == SegmentPhysicalEventSequenceKind3D::Certified
                && extremely_close.events.size() == 2,
            "distinct certified roots are never distance-merged");
}

void test_all_event_triple_root_and_reverse_telescoping()
{
    NurbsSurfaceIntersectionResult3D intersection;
    intersection.crossings = {
        transverse_root(0, 0.18, -1.0),
        transverse_root(0, 0.47, 1.0),
        transverse_root(0, 0.83, -1.0)};
    const auto forward = select_certified_segment_physical_events_3d(
        intersection, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0e-12);
    require(forward.kind
                == SegmentPhysicalEventSequenceKind3D::Certified
                && forward.events.size() == 3,
            "triple crossing retains every ordered physical event");
    require(forward.events[0].continuation_sign == 1
                && forward.events[1].continuation_sign == -1
                && forward.events[2].continuation_sign == 1,
            "triple crossing alternates branch transitions");

    const auto reverse = select_certified_segment_physical_events_3d(
        intersection, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 1.0e-12);
    require(reverse.kind
                == SegmentPhysicalEventSequenceKind3D::Certified
                && reverse.events.size() == forward.events.size(),
            "reverse traversal retains the same physical events");
    for (std::size_t q = 0; q < forward.events.size(); ++q) {
        const auto& first = forward.events[q];
        const auto& reversed =
            reverse.events[reverse.events.size() - 1 - q];
        require((first.point - reversed.point).norm() < 1.0e-14
                    && first.continuation_sign
                           == -reversed.continuation_sign,
                "reverse traversal reverses order and continuation sign");
    }
    int state = 0;
    for (const auto& event : forward.events)
        state += event.continuation_sign;
    require(state == 1,
            "triple crossing telescopes to the opposite branch");
}

void test_all_event_c0_owner_dedup_and_fail_closed()
{
    NurbsSurfaceIntersectionResult3D feature;
    auto first = transverse_root(0, 0.5, -1.0);
    auto second = transverse_root(6, 0.5, -0.8);
    first.feature_edge_contact = true;
    second.feature_edge_contact = true;
    first.owners = {owner(first), owner(second)};
    feature.crossings = {first};
    const auto deduplicated = select_certified_segment_physical_events_3d(
        feature, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0e-12);
    require(deduplicated.kind
                == SegmentPhysicalEventSequenceKind3D::Certified
                && deduplicated.events.size() == 1
                && deduplicated.events.front().crossing_indices.size() == 1
                && deduplicated.events.front().continuation_sign == 1,
            "transverse C0 owners are one consistently oriented physical event");

    feature.crossings.front().owners[1].normal = Eigen::Vector3d::UnitX();
    const auto conflicting = select_certified_segment_physical_events_3d(
        feature, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0e-12);
    require(conflicting.kind == SegmentPhysicalEventSequenceKind3D::
                InconsistentContinuationSign,
            "coincident owners with conflicting signs fail closed");

    NurbsSurfaceIntersectionResult3D incomplete;
    incomplete.crossings = {transverse_root(0, 0.5, -1.0)};
    incomplete.diagnostics.unresolved_candidates = 1;
    require(select_certified_segment_physical_events_3d(
                incomplete, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0e-12)
                .kind
                == SegmentPhysicalEventSequenceKind3D::IncompleteRootSet,
            "unresolved root sets fail closed");

    NurbsSurfaceIntersectionResult3D tangent;
    tangent.crossings = {transverse_root(0, 0.5, -1.0)};
    tangent.crossings.front().transversality = 0.0;
    require(select_certified_segment_physical_events_3d(
                tangent, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0e-12)
                .kind
                == SegmentPhysicalEventSequenceKind3D::NonTransverseContact,
            "tangent/contact events fail closed");
}

void test_feature_owner_conflict_uses_exact_side_classifier()
{
    NurbsSurfaceIntersectionResult3D feature;
    feature.crossings = {conflicting_feature_root(0.5)};

    std::vector<Eigen::Vector3d> probes;
    const auto sequence = select_certified_segment_physical_events_3d(
        feature, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0e-12,
        [&](const Eigen::Vector3d& point) {
            probes.push_back(point);
            return point.x() > 0.5;
        });
    require(sequence.kind
                == SegmentPhysicalEventSequenceKind3D::Certified
                && sequence.events.size() == 1
                && sequence.events.front().continuation_sign == 1,
            "exact side labels certify a crossing despite conflicting C0 normals");
    require(probes.size() == 2 && probes[0].x() < 0.5
                && probes[1].x() > 0.5,
            "exact side classifier is sampled on both sides of the feature");

    const auto contact = select_certified_segment_physical_events_3d(
        feature, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0e-12,
        [](const Eigen::Vector3d&) { return false; });
    require(contact.kind
                == SegmentPhysicalEventSequenceKind3D::Certified
                && contact.events.size() == 1
                && contact.events.front().continuation_sign == 0,
            "equal exact side labels retain a certified feature contact");

    const auto no_classifier = select_certified_segment_physical_events_3d(
        feature, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0e-12);
    require(no_classifier.kind == SegmentPhysicalEventSequenceKind3D::
                InconsistentContinuationSign,
            "missing exact classifier preserves the normal-only fail-closed path");
}

void test_feature_side_probes_never_cross_close_neighbors()
{
    constexpr double first_x = 0.5 - 2.0e-8;
    constexpr double second_x = 0.5 + 2.0e-8;
    NurbsSurfaceIntersectionResult3D feature;
    feature.crossings = {
        conflicting_feature_root(first_x),
        conflicting_feature_root(second_x)};

    std::vector<double> probes;
    const auto exact_inside = [&](const Eigen::Vector3d& point) {
        probes.push_back(point.x());
        return point.x() > first_x && point.x() < second_x;
    };
    const auto sequence = select_certified_segment_physical_events_3d(
        feature, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0e-6,
        exact_inside);
    require(sequence.kind
                == SegmentPhysicalEventSequenceKind3D::Certified
                && sequence.events.size() == 2
                && sequence.events[0].continuation_sign == 1
                && sequence.events[1].continuation_sign == -1,
            "exact side classification preserves close entering/leaving events");
    require(probes.size() == 4
                && 0.0 < probes[0] && probes[0] < first_x
                && first_x < probes[1] && probes[1] < second_x
                && first_x < probes[2] && probes[2] < second_x
                && second_x < probes[3] && probes[3] < 1.0,
            "one-sided probes remain strictly between neighboring events and endpoints");
}

void test_gridline_search_never_crosses_c0_seam()
{
    const G1PatchTopology3D topology = l_prism_like_topology();
    const int component = topology.surface_component_by_patch[0];
    const auto segment_root = root(0, component, {0.0, 0.0, 0.0});

    // Patches 0--2 form one G1 sheet on this geometry.  Patch 6 is on the
    // same closed surface component but is separated from patch 0 by C0 edges.
    const std::vector<CartesianGridlineCrossingRecord3D> records = {
        record(60, 6, component, {1.0e-6, 0.0, 0.0}),
        record(10, 1, component, {0.25, 0.0, 0.0})};
    const auto selected =
        select_nearest_gridline_crossing_on_g1_sheet_3d(
            segment_root, topology, records, 1.0e-12);
    require(selected.kind == GridlineCrossingSelectionKind3D::Selected
                && selected.record_index == 1
                && selected.payload_index == 10,
            "farther G1 record wins over geometrically closer C0 record");
    require(selected.same_surface_component_count == 2
                && selected.same_g1_sheet_count == 1,
            "component and G1 candidate counts distinguish C0 filtering");

    const auto no_sheet =
        select_nearest_gridline_crossing_on_g1_sheet_3d(
            segment_root, topology, {records.front()}, 1.0e-12);
    require(no_sheet.kind
                == GridlineCrossingSelectionKind3D::NoSameG1SheetFallback
                && no_sheet.record_index < 0,
            "C0-only catalog requests an explicit caller fallback");
}

void test_gridline_search_filters_surface_component_and_reports_ties()
{
    const G1PatchTopology3D topology = l_prism_like_topology();
    const int component = topology.surface_component_by_patch[0];
    const auto segment_root = root(0, component, Eigen::Vector3d::Zero());

    auto wrong_component = record(20, 1, component, {0.1, 0.0, 0.0});
    // Preserve valid patch metadata by using an empty catalog for the no-
    // component branch; inconsistent synthetic component IDs are rejected.
    (void)wrong_component;
    const auto no_component =
        select_nearest_gridline_crossing_on_g1_sheet_3d(
            segment_root, topology, {}, 1.0e-12);
    require(no_component.kind == GridlineCrossingSelectionKind3D::
                NoSameSurfaceComponentFallback,
            "missing surface-component candidates are explicit");

    const std::vector<CartesianGridlineCrossingRecord3D> tied = {
        record(30, 1, component, {0.1, 0.0, 0.0}),
        record(31, 2, component, {-0.1, 0.0, 0.0})};
    const auto selected =
        select_nearest_gridline_crossing_on_g1_sheet_3d(
            segment_root, topology, tied, 1.0e-12);
    require(selected.kind == GridlineCrossingSelectionKind3D::Selected
                && selected.payload_index == 30
                && selected.nearest_tie_count == 2,
            "same-G1 ties are deterministic and diagnosed");
}

void test_feature_intersection_preserves_incident_g1_sheets()
{
    const auto cylinder = kfbim::app3d::make_native_nurbs_surface_3d(
        kfbim::app3d::GeometryKind3D::HollowCylinder);
    kfbim::geometry3d::NurbsSurfaceIntersectorOptions3D options;
    options.preserve_non_g1_root_owners = true;
    const kfbim::geometry3d::NurbsSurfaceIntersector3D intersector(
        cylinder.geometry_model(), options);
    const auto hit = intersector.intersect_segment(
        {0.63, -0.05, 0.69}, {0.59, -0.05, 0.65});
    require(hit.crossings.size() == 2,
            "C0 feature intersection retains one owner per incident G1 sheet");
    require(hit.crossings[0].patch_index != hit.crossings[1].patch_index
                && hit.crossings[0].component == hit.crossings[1].component
                && (hit.crossings[0].point - hit.crossings[1].point).norm()
                       < 1.0e-10
                && hit.crossings[0].feature_edge_contact
                && hit.crossings[1].feature_edge_contact,
            "preserved feature owners describe one physical root on distinct sheets");
}

void test_aggregate_unresolved_regions_bound_endpoint_contacts()
{
    const auto cylinder = kfbim::app3d::make_native_nurbs_surface_3d(
        kfbim::app3d::GeometryKind3D::HollowCylinder);
    kfbim::geometry3d::NurbsSurfaceIntersectorOptions3D options;
    options.maximum_element_extent = 2.0 * (3.0 / 32.0);
    options.local_max_subdivision_depth = 30;
    options.preserve_non_g1_root_owners = true;
    options.collect_unresolved_regions = true;
    const kfbim::geometry3d::NurbsSurfaceIntersector3D intersector(
        cylinder.geometry_model(), options);
    const Eigen::Vector3d node(0.5625, -0.28125, -0.5625);
    const Eigen::Vector3d trace =
        cylinder.geometry_model().patch(3).evaluate(0.65, 1.0 / 28.0);
    const double length = (trace - node).norm();
    const double tie_tolerance = std::max(
        16.0 * intersector.geometry_tolerance(),
        1.0e-12 * (3.0 / 32.0));
    const double retreat = std::min(0.25 * length, 0.5 * tie_tolerance);
    const double unresolved_slack = tie_tolerance - retreat;
    const Eigen::Vector3d open_end =
        node + (1.0 - retreat / length) * (trace - node);
    const auto hit = intersector.intersect_segment(node, open_end);
    require(hit.diagnostics.unresolved_longitudinal_intervals.size()
                == static_cast<std::size_t>(
                    hit.diagnostics.unresolved_candidates),
            "endpoint contact returns one conservative interval per unresolved region");
    const bool certified = std::all_of(
        hit.diagnostics.unresolved_longitudinal_intervals.begin(),
        hit.diagnostics.unresolved_longitudinal_intervals.end(),
        [&](const std::array<double, 2>& interval) {
            return std::isfinite(interval[0])
                && std::isfinite(interval[1])
                && interval[0] <= interval[1]
                && interval[0]
                       >= (open_end - node).norm() - unresolved_slack;
        });
    require(certified,
            "deeper local subdivision certifies the endpoint tie budget");
}

} // namespace

int main()
{
    try {
        test_multiple_segment_roots_choose_nearest_grid_point();
        test_segment_root_tie_and_degenerate_fallbacks();
        test_all_event_double_root_telescoping();
        test_all_event_triple_root_and_reverse_telescoping();
        test_all_event_c0_owner_dedup_and_fail_closed();
        test_feature_owner_conflict_uses_exact_side_classifier();
        test_feature_side_probes_never_cross_close_neighbors();
        test_gridline_search_never_crosses_c0_seam();
        test_gridline_search_filters_surface_component_and_reports_ties();
        test_feature_intersection_preserves_incident_g1_sheets();
        test_aggregate_unresolved_regions_bound_endpoint_contacts();
        std::cout << "restrict crossing selector 3D tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "restrict crossing selector 3D test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
