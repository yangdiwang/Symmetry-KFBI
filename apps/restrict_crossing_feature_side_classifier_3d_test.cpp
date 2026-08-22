#include "restrict_crossing_selector_3d.hpp"
#include "src/geometry/nurbs_surface_intersector_3d.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using kfbim::app3d::SegmentPhysicalEventSequenceKind3D;
using kfbim::app3d::SegmentEndpointPartitionKind3D;
using kfbim::app3d::partition_certified_segment_endpoint_3d;
using kfbim::app3d::select_certified_segment_physical_events_3d;
using kfbim::geometry3d::NurbsSurfaceCrossing3D;
using kfbim::geometry3d::NurbsSurfaceIntersectionResult3D;
using kfbim::geometry3d::NurbsSurfaceRootOwner3D;

void require(bool value, const std::string& message)
{
    if (!value)
        throw std::runtime_error(message);
}

NurbsSurfaceRootOwner3D owner(int patch,
                              double x,
                              double outward_x)
{
    NurbsSurfaceRootOwner3D result;
    result.patch_index = patch;
    result.point = {x, 0.0, 0.0};
    result.normal = {outward_x, 0.0, 0.0};
    result.transversality = 1.0;
    result.reliable_transversality_tolerance = 1.0e-8;
    result.feature_edge_contact = true;
    return result;
}

NurbsSurfaceCrossing3D conflicting_feature_root(double x)
{
    NurbsSurfaceCrossing3D result;
    result.patch_index = 0;
    result.component = 0;
    result.point = {x, 0.0, 0.0};
    result.normal = {-1.0, 0.0, 0.0};
    result.transversality = 1.0;
    result.reliable_transversality_tolerance = 1.0e-8;
    result.feature_edge_contact = true;
    result.owners = {owner(0, x, -1.0), owner(1, x, 1.0)};
    return result;
}

NurbsSurfaceCrossing3D native_root(int patch,
                                   double x,
                                   double u,
                                   double v)
{
    NurbsSurfaceCrossing3D result;
    result.patch_index = patch;
    result.component = 0;
    result.u = u;
    result.v = v;
    result.point = {x, 0.0, 0.0};
    result.normal = {-1.0, 0.0, 0.0};
    result.transversality = 1.0;
    result.reliable_transversality_tolerance = 1.0e-8;
    return result;
}

void test_conflicting_normals_with_and_without_classifier()
{
    NurbsSurfaceIntersectionResult3D intersection;
    intersection.crossings = {conflicting_feature_root(0.5)};

    const auto without_classifier =
        select_certified_segment_physical_events_3d(
            intersection, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
            1.0e-12);
    require(without_classifier.kind
                == SegmentPhysicalEventSequenceKind3D::
                    InconsistentContinuationSign,
            "conflicting feature normals fail closed without a classifier");

    std::vector<double> probes;
    const auto crossing = select_certified_segment_physical_events_3d(
        intersection, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0e-12,
        [&](const Eigen::Vector3d& point) {
            probes.push_back(point.x());
            return point.x() > 0.5;
        });
    require(crossing.kind
                == SegmentPhysicalEventSequenceKind3D::Certified
                && crossing.events.size() == 1
                && crossing.events.front().continuation_sign == 1,
            "exact side flip certifies conflicting feature normals");
    require(probes.size() == 2 && probes[0] < 0.5 && probes[1] > 0.5,
            "classifier receives one strictly one-sided probe per side");

    const auto contact = select_certified_segment_physical_events_3d(
        intersection, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0e-12,
        [](const Eigen::Vector3d&) { return false; });
    require(contact.kind
                == SegmentPhysicalEventSequenceKind3D::Certified
                && contact.events.size() == 1
                && contact.events.front().continuation_sign == 0,
            "equal exact labels retain a certified zero-continuation contact");
}

void test_close_neighbor_probe_isolation()
{
    constexpr double first_x = 0.5 - 2.0e-8;
    constexpr double second_x = 0.5 + 2.0e-8;
    NurbsSurfaceIntersectionResult3D intersection;
    intersection.crossings = {
        conflicting_feature_root(first_x),
        conflicting_feature_root(second_x)};

    std::vector<double> probes;
    const auto sequence = select_certified_segment_physical_events_3d(
        intersection, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0e-6,
        [&](const Eigen::Vector3d& point) {
            probes.push_back(point.x());
            return point.x() > first_x && point.x() < second_x;
        });
    require(sequence.kind
                == SegmentPhysicalEventSequenceKind3D::Certified
                && sequence.events.size() == 2
                && sequence.events[0].continuation_sign == 1
                && sequence.events[1].continuation_sign == -1,
            "close feature roots retain entering and leaving transitions");
    require(probes.size() == 4
                && 0.0 < probes[0] && probes[0] < first_x
                && first_x < probes[1] && probes[1] < second_x
                && first_x < probes[2] && probes[2] < second_x
                && second_x < probes[3] && probes[3] < 1.0,
            "close-root probes never cross an adjacent event or endpoint");
}

void test_smooth_single_owner_never_calls_exact_classifier()
{
    NurbsSurfaceCrossing3D smooth;
    smooth.patch_index = 0;
    smooth.component = 0;
    smooth.point = {0.5, 0.0, 0.0};
    smooth.normal = {-1.0, 0.0, 0.0};
    smooth.transversality = 1.0;
    smooth.reliable_transversality_tolerance = 1.0e-8;
    NurbsSurfaceIntersectionResult3D intersection;
    intersection.crossings = {smooth};

    bool callback_called = false;
    const auto sequence = select_certified_segment_physical_events_3d(
        intersection, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0e-12,
        [&](const Eigen::Vector3d&) -> bool {
            callback_called = true;
            throw std::runtime_error("smooth event used exact classifier");
        });
    require(sequence.kind
                == SegmentPhysicalEventSequenceKind3D::Certified
                && sequence.events.size() == 1
                && sequence.events.front().continuation_sign == 1
                && !callback_called,
            "reliable smooth single-owner events stay on normal certification");
}

void test_closed_segment_endpoint_partition_preserves_near_roots()
{
    constexpr int endpoint_patch = 7;
    constexpr double endpoint_u = 0.3;
    constexpr double endpoint_v = 0.4;
    NurbsSurfaceIntersectionResult3D intersection;
    intersection.crossings = {
        native_root(2, 0.2, 0.1, 0.2),
        native_root(endpoint_patch, 1.0 - 2.0e-10, 0.2, 0.4),
        native_root(9, 1.0 - 5.0e-13, 0.8, 0.1),
        native_root(endpoint_patch, 1.0, endpoint_u, endpoint_v),
        native_root(4, 1.05, 0.2, 0.7)};

    const auto partition = partition_certified_segment_endpoint_3d(
        intersection, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        {1.1, 0.0, 0.0},
        endpoint_patch, endpoint_u, endpoint_v, 1.0e-12, 1.0e-10);
    require(partition.kind == SegmentEndpointPartitionKind3D::Certified
                && partition.endpoint_crossing_index == 3
                && partition.open_crossing_indices
                       == std::vector<int>({0, 1, 2})
                && partition.post_endpoint_crossing_indices
                       == std::vector<int>({4}),
            "closed-segment partition removes only the native exact endpoint");
}

void test_closed_segment_endpoint_partition_fails_closed()
{
    constexpr int endpoint_patch = 7;
    constexpr double endpoint_u = 0.3;
    constexpr double endpoint_v = 0.4;
    NurbsSurfaceIntersectionResult3D unresolved;
    unresolved.crossings = {
        native_root(endpoint_patch, 1.0, endpoint_u, endpoint_v)};
    unresolved.diagnostics.unresolved_candidates = 1;
    unresolved.diagnostics.unresolved_longitudinal_intervals.push_back(
        {0.9, 1.0});
    require(partition_certified_segment_endpoint_3d(
                unresolved, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                {1.1, 0.0, 0.0},
                endpoint_patch, endpoint_u, endpoint_v, 1.0e-12, 1.0e-10)
                .kind
                == SegmentEndpointPartitionKind3D::IncompleteRootSet,
            "any unresolved endpoint neighborhood fails closed");

    NurbsSurfaceIntersectionResult3D missing;
    missing.crossings = {native_root(2, 0.5, 0.1, 0.2)};
    require(partition_certified_segment_endpoint_3d(
                missing, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                {1.1, 0.0, 0.0},
                endpoint_patch, endpoint_u, endpoint_v, 1.0e-12, 1.0e-10)
                .kind
                == SegmentEndpointPartitionKind3D::MissingExactEndpoint,
            "a closed query that omits the known endpoint fails closed");

    NurbsSurfaceIntersectionResult3D duplicate;
    duplicate.crossings = {
        native_root(endpoint_patch, 1.0, endpoint_u, endpoint_v),
        native_root(endpoint_patch, 1.0, endpoint_u, endpoint_v)};
    require(partition_certified_segment_endpoint_3d(
                duplicate, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                {1.1, 0.0, 0.0},
                endpoint_patch, endpoint_u, endpoint_v, 1.0e-12, 1.0e-10)
                .kind
                == SegmentEndpointPartitionKind3D::AmbiguousExactEndpoint,
            "duplicate exact endpoint roots fail closed instead of merging");
}

} // namespace

int main()
{
    try {
        test_conflicting_normals_with_and_without_classifier();
        test_close_neighbor_probe_isolation();
        test_smooth_single_owner_never_calls_exact_classifier();
        test_closed_segment_endpoint_partition_preserves_near_roots();
        test_closed_segment_endpoint_partition_fails_closed();
        std::cout << "restrict crossing feature side classifier 3D tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "restrict crossing feature side classifier 3D test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
