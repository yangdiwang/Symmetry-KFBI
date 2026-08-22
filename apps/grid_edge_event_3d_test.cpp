#include "src/geometry/grid_pair_3d.hpp"
#include "src/geometry/nurbs_cartesian_domain_3d.hpp"
#include "src/transfer/laplace_grid_edge_event_correction_3d.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using kfbim::geometry3d::GridEdgeEventCertification3D;
using kfbim::geometry3d::GridEdgeEvent3D;
using kfbim::geometry3d::NurbsCartesianEdgeClassification3D;
using kfbim::geometry3d::NurbsCartesianEdgeIntersections3D;
using kfbim::geometry3d::NurbsSurfaceCrossing3D;
using kfbim::geometry3d::NurbsSurfaceRootOwner3D;
using kfbim::geometry3d::build_canonical_grid_edge_events_3d;
using kfbim::geometry3d::grid_edge_root_set_requires_targeted_retry_3d;
using kfbim::geometry3d::grid_edge_event_views_3d;
using kfbim::CartesianGrid3D;
using kfbim::DofLayout3D;
using kfbim::LaplaceCrossingCorrectionOp;
using kfbim::P2NativeSheetCenterCandidate3D;
using kfbim::native_grid_edge_event_correction_ops_3d;
using kfbim::select_p2_native_event_owner_center_3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void require_close(double actual,
                   double expected,
                   const std::string& message)
{
    if (std::abs(actual - expected) > 1.0e-14)
        throw std::runtime_error(message);
}

NurbsSurfaceCrossing3D crossing(double t,
                               int normal_sign,
                               int patch = 0,
                               int component = 0)
{
    NurbsSurfaceCrossing3D result;
    result.patch_index = patch;
    result.component = component;
    result.u = t;
    result.v = 0.25;
    result.edge_parameter = t;
    result.point = {t, 0.0, 0.0};
    result.normal = {static_cast<double>(normal_sign), 0.0, 0.0};
    result.residual = 1.0e-14;
    result.transversality = 1.0;
    result.reliable_transversality_tolerance = 1.0e-8;
    return result;
}

NurbsSurfaceCrossing3D conflicting_feature_crossing(double t,
                                                    int component = 0)
{
    NurbsSurfaceCrossing3D result = crossing(t, 1, 0, component);
    result.feature_edge_contact = true;
    NurbsSurfaceRootOwner3D positive;
    positive.patch_index = 0;
    positive.u = t;
    positive.v = 0.25;
    positive.point = result.point;
    positive.normal = {1.0, 0.0, 0.0};
    positive.residual = 1.0e-14;
    positive.transversality = 1.0;
    positive.feature_edge_contact = true;
    positive.reliable_transversality_tolerance = 1.0e-8;
    NurbsSurfaceRootOwner3D negative = positive;
    negative.patch_index = 1;
    negative.normal = {-1.0, 0.0, 0.0};
    result.owners = {positive, negative};
    return result;
}

NurbsCartesianEdgeClassification3D complete_classification(
    std::size_t root_count,
    bool changes_component_membership)
{
    NurbsCartesianEdgeClassification3D result;
    result.queried = true;
    result.has_confirmed_interface = root_count != 0;
    result.changes_component_membership = changes_component_membership;
    result.root_count_known = true;
    result.parity_known_from_roots = true;
    result.confirmed_crossing_count = root_count;
    result.confirmed_transverse_count = static_cast<int>(root_count);
    return result;
}

std::vector<GridEdgeEvent3D> build(
    const std::vector<NurbsSurfaceCrossing3D>& roots,
    bool changes_component_membership)
{
    return build_canonical_grid_edge_events_3d(
        2, 9, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, roots,
        complete_classification(
            roots.size(), changes_component_membership));
}

void test_same_label_double_root_is_not_filtered()
{
    const std::vector<GridEdgeEvent3D> events = build(
        {crossing(0.2, 1), crossing(0.8, -1)}, false);
    require(events.size() == 2,
            "same-label edge retains both physical crossings");
    require(events[0].id.first_node == 2
                && events[0].id.second_node == 9
                && events[0].id.ordinal == 0
                && events[1].id.ordinal == 1,
            "same-label events receive stable per-edge ids");
    require(events[0].certified_transverse()
                && events[1].certified_transverse()
                && events[0].canonical_sign == 1
                && events[1].canonical_sign == -1,
            "same-label crossings retain certified oriented signs");
    require(events[0].owners.size() == 1
                && events[1].owners.size() == 1,
            "canonical events expose their parametric owners");
}

void test_three_events_have_ordered_ids_and_signs()
{
    const std::vector<GridEdgeEvent3D> events = build(
        {crossing(0.73, 1, 2), crossing(0.11, 1, 0),
         crossing(0.42, -1, 1)},
        true);
    require(events.size() == 3,
            "three-root edge retains every certified event");
    const std::array<double, 3> expected_t{{0.11, 0.42, 0.73}};
    const std::array<int, 3> expected_sign{{1, -1, 1}};
    for (int event = 0; event < 3; ++event) {
        require(events[static_cast<std::size_t>(event)].id.ordinal == event,
                "three-root ids follow canonical geometric order");
        require_close(
            events[static_cast<std::size_t>(event)].canonical_parameter,
            expected_t[static_cast<std::size_t>(event)],
            "three-root canonical parameter is ordered");
        require(events[static_cast<std::size_t>(event)].canonical_sign
                    == expected_sign[static_cast<std::size_t>(event)],
                "three-root canonical sign follows the outward normal");
    }
}

void test_reverse_view_reverses_order_parameter_and_sign_only()
{
    const std::vector<GridEdgeEvent3D> events = build(
        {crossing(0.15, 1), crossing(0.45, -1), crossing(0.9, 1)},
        true);
    const auto reverse = grid_edge_event_views_3d(events, 9, 2);
    require(reverse.size() == 3,
            "reverse view retains all edge events");
    for (int query_ordinal = 0; query_ordinal < 3; ++query_ordinal) {
        const int canonical_ordinal = 2 - query_ordinal;
        const auto& canonical =
            events[static_cast<std::size_t>(canonical_ordinal)];
        const auto& view = reverse[static_cast<std::size_t>(query_ordinal)];
        require(view.id() == canonical.id,
                "event id is invariant under edge reversal");
        require_close(view.edge_parameter(),
                      1.0 - canonical.canonical_parameter,
                      "reverse parameter is measured from the query endpoint");
        require(view.sign() == -canonical.canonical_sign,
                "reverse query flips the oriented event sign");
        if (query_ordinal > 0) {
            require(reverse[static_cast<std::size_t>(query_ordinal - 1)]
                            .edge_parameter()
                        < view.edge_parameter(),
                    "reverse views remain ordered from their query endpoint");
        }
    }

    std::vector<NurbsSurfaceCrossing3D> reverse_input;
    const std::vector<NurbsSurfaceCrossing3D> forward_roots{
        crossing(0.15, 1), crossing(0.45, -1), crossing(0.9, 1)};
    for (auto root = forward_roots.rbegin(); root != forward_roots.rend();
         ++root) {
        NurbsSurfaceCrossing3D reversed_root = *root;
        reversed_root.edge_parameter = 1.0 - root->edge_parameter;
        reverse_input.push_back(std::move(reversed_root));
    }
    const std::vector<GridEdgeEvent3D> rebuilt =
        build_canonical_grid_edge_events_3d(
            9, 2, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, reverse_input,
            complete_classification(reverse_input.size(), true));
    require(rebuilt.size() == events.size(),
            "reverse-input canonicalization retains event count");
    for (std::size_t event = 0; event < events.size(); ++event) {
        require(rebuilt[event].id == events[event].id,
                "reverse-input canonicalization preserves event ids");
        require_close(rebuilt[event].canonical_parameter,
                      events[event].canonical_parameter,
                      "reverse-input canonicalization preserves parameters");
        require(rebuilt[event].canonical_sign
                    == events[event].canonical_sign,
                "reverse-input canonicalization preserves canonical signs");
    }
}

void test_uncertified_contacts_and_tangencies_fail_closed()
{
    NurbsSurfaceCrossing3D feature = crossing(0.3, 1);
    feature.feature_edge_contact = true;
    auto events = build({feature}, true);
    require(events.size() == 1
                && events[0].certified_transverse()
                && events[0].canonical_sign == 1,
            "declared feature with one reliable orientation is certified");

    NurbsSurfaceRootOwner3D positive_owner;
    positive_owner.patch_index = 0;
    positive_owner.u = 0.3;
    positive_owner.v = 0.25;
    positive_owner.point = feature.point;
    positive_owner.normal = {1.0, 0.0, 0.0};
    positive_owner.residual = 1.0e-14;
    positive_owner.transversality = 1.0;
    positive_owner.reliable_transversality_tolerance = 1.0e-8;
    positive_owner.feature_edge_contact = true;
    NurbsSurfaceRootOwner3D negative_owner = positive_owner;
    negative_owner.patch_index = 1;
    negative_owner.normal = {-1.0, 0.0, 0.0};
    feature.owners = {positive_owner, negative_owner};
    events = build({feature}, true);
    require(events.size() == 1
                && events[0].certification
                       == GridEdgeEventCertification3D::FeatureContact
                && events[0].canonical_sign == 0,
            "feature owners with conflicting orientation fail closed");

    NurbsCartesianEdgeClassification3D tangent_classification =
        complete_classification(1, false);
    tangent_classification.has_near_tangent_candidate = true;
    events = build_canonical_grid_edge_events_3d(
        2, 9, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        {crossing(0.4, 1)}, tangent_classification);
    require(events.size() == 1
                && events[0].certification
                       == GridEdgeEventCertification3D::NearTangentEdge
                && events[0].canonical_sign == 0,
            "near-tangent edge is not exposed as a signed jump");

    NurbsCartesianEdgeClassification3D incomplete =
        complete_classification(1, true);
    incomplete.root_count_known = false;
    incomplete.parity_known_from_roots = false;
    events = build_canonical_grid_edge_events_3d(
        2, 9, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        {crossing(0.5, -1)}, incomplete);
    require(events.size() == 1
                && events[0].certification
                       == GridEdgeEventCertification3D::IncompleteRootSet
                && events[0].canonical_sign == 0,
            "incomplete root set is retained for diagnostics but unsigned");
}

void test_every_uncertified_root_set_requires_targeted_retry()
{
    NurbsCartesianEdgeIntersections3D result;
    require(!grid_edge_root_set_requires_targeted_retry_3d(result),
            "a certified empty root set needs no retry");

    // Endpoint labels are intentionally absent from this decision: an
    // unknown zero-root result on a same-label candidate edge must still be
    // retried.
    result.root_count_known = false;
    result.parity_known_from_roots = false;
    require(grid_edge_root_set_requires_targeted_retry_3d(result),
            "unknown same-label zero-root result requires retry");

    result = {};
    result.crossings = {crossing(0.3, 1), crossing(0.7, -1)};
    result.ambiguous_clusters.push_back({});
    require(grid_edge_root_set_requires_targeted_retry_3d(result),
            "ambiguous same-label double root requires retry");

    result = {};
    result.has_near_tangent_candidate = true;
    require(grid_edge_root_set_requires_targeted_retry_3d(result),
            "near-tangent candidate requires retry");

    result = {};
    result.diagnostics.unresolved_candidates = 1;
    require(grid_edge_root_set_requires_targeted_retry_3d(result),
            "unresolved candidate requires retry");
}

void test_feature_orientation_uses_component_containment()
{
    const NurbsSurfaceCrossing3D feature =
        conflicting_feature_crossing(0.5);
    const auto classification = complete_classification(1, true);
    std::vector<double> probes;
    const auto events = build_canonical_grid_edge_events_3d(
        2, 9, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {feature},
        classification,
        [&](const Eigen::Vector3d& point, int component) {
            require(component == 0,
                    "feature containment receives the physical component");
            probes.push_back(point.x());
            return point.x() < 0.5;
        });
    require(events.size() == 1 && events[0].certified_transverse()
                && events[0].canonical_sign == 1,
            "containment flip certifies conflicting feature owners");
    require(probes.size() == 2 && probes[0] > 0.0 && probes[0] < 0.5
                && probes[1] > 0.5 && probes[1] < 1.0,
            "feature orientation uses exactly one strict probe per side");

    bool smooth_probe_called = false;
    const auto smooth = build_canonical_grid_edge_events_3d(
        2, 9, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        {crossing(0.4, 1)}, classification,
        [&](const Eigen::Vector3d&, int) {
            smooth_probe_called = true;
            return false;
        });
    require(smooth.size() == 1 && smooth[0].certified_transverse()
                && smooth[0].canonical_sign == 1
                && !smooth_probe_called,
            "reliable smooth events do not use containment fallback");

    const auto contact = build_canonical_grid_edge_events_3d(
        2, 9, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {feature},
        classification,
        [](const Eigen::Vector3d&, int) { return false; });
    require(contact.size() == 1
                && contact[0].certification
                       == GridEdgeEventCertification3D::FeatureContact
                && contact[0].canonical_sign == 0,
            "same-side feature probes remain a fail-closed contact");

    bool classification_failed = false;
    try {
        (void)build_canonical_grid_edge_events_3d(
            2, 9, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {feature},
            classification,
            [](const Eigen::Vector3d&, int) -> bool {
                throw std::runtime_error("synthetic containment failure");
            });
    } catch (const std::runtime_error& error) {
        classification_failed = std::string(error.what()).find(
                                    "could not certify")
            != std::string::npos;
    }
    require(classification_failed,
            "failed feature containment propagates as setup failure");
}

void test_close_feature_probes_stay_between_neighboring_events()
{
    constexpr double first_root = 0.5 - 1.0e-9;
    constexpr double second_root = 0.5 + 1.0e-9;
    const std::vector<NurbsSurfaceCrossing3D> roots{
        conflicting_feature_crossing(first_root),
        conflicting_feature_crossing(second_root)};
    std::vector<double> probes;
    const auto events = build_canonical_grid_edge_events_3d(
        2, 9, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, roots,
        complete_classification(roots.size(), false),
        [&](const Eigen::Vector3d& point, int) {
            probes.push_back(point.x());
            return point.x() > first_root && point.x() < second_root;
        });
    require(events.size() == 2 && events[0].certified_transverse()
                && events[1].certified_transverse()
                && events[0].canonical_sign == -1
                && events[1].canonical_sign == 1,
            "close distinct feature roots retain two containment signs");
    require(probes.size() == 4
                && probes[0] > 0.0 && probes[0] < first_root
                && probes[1] > first_root && probes[1] < second_root
                && probes[2] > first_root && probes[2] < second_root
                && probes[3] > second_root && probes[3] < 1.0,
            "feature probes never cross a neighboring physical event");
}

void test_multi_event_corrections_preserve_ids_and_reverse_signs()
{
    const CartesianGrid3D grid(
        {0.0, 0.0, 0.0}, {0.25, 0.25, 0.25}, {4, 4, 4},
        DofLayout3D::Node);
    const int first = grid.index(2, 2, 2);
    const int second = grid.index(3, 2, 2);

    std::vector<GridEdgeEvent3D> events = build(
        {crossing(0.15, 1), crossing(0.45, -1), crossing(0.9, 1)},
        true);
    std::vector<LaplaceCrossingCorrectionOp> ops;
    for (std::size_t ordinal = 0; ordinal < events.size(); ++ordinal) {
        GridEdgeEvent3D& event = events[ordinal];
        event.id.first_node = first;
        event.id.second_node = second;
        event.id.ordinal = static_cast<int>(ordinal);
        event.point = {
            0.5 + 0.25 * event.canonical_parameter, 0.5, 0.5};
        const auto directed =
            native_grid_edge_event_correction_ops_3d(grid, event);
        require(directed.size() == 2,
                "each physical event produces two directed PDE corrections");
        ops.insert(ops.end(), directed.begin(), directed.end());
    }
    require(ops.size() == 6,
            "three physical roots produce six directed corrections");
    int forward_sum = 0;
    int reverse_sum = 0;
    for (std::size_t ordinal = 0; ordinal < events.size(); ++ordinal) {
        const auto& forward = ops[2 * ordinal];
        const auto& reverse = ops[2 * ordinal + 1];
        require(forward.grid_edge_event == events[ordinal].id
                    && reverse.grid_edge_event == events[ordinal].id,
                "directed corrections share the stable physical event id");
        require(forward.rhs_node == first
                    && forward.correction_node == second
                    && reverse.rhs_node == second
                    && reverse.correction_node == first,
                "directed event corrections reverse their endpoints");
        require(forward.side_delta == events[ordinal].canonical_sign
                    && reverse.side_delta == -forward.side_delta,
                "directed event corrections reverse the oriented sign");
        require_close(forward.stencil_weight, reverse.stencil_weight,
                      "reversed Cartesian stencil weights agree");
        forward_sum += forward.side_delta;
        reverse_sum += reverse.side_delta;
    }
    require(forward_sum == 1 && reverse_sum == -1,
            "three-root correction signs aggregate by physical event");

    const auto double_events = build(
        {crossing(0.2, 1), crossing(0.8, -1)}, false);
    int double_sum = 0;
    int double_op_count = 0;
    for (std::size_t ordinal = 0; ordinal < double_events.size(); ++ordinal) {
        GridEdgeEvent3D event = double_events[ordinal];
        event.id = {first, second, static_cast<int>(ordinal)};
        event.point = {
            0.5 + 0.25 * event.canonical_parameter, 0.5, 0.5};
        const auto directed =
            native_grid_edge_event_correction_ops_3d(grid, event);
        double_op_count += static_cast<int>(directed.size());
        double_sum += directed.front().side_delta;
    }
    require(double_op_count == 4 && double_sum == 0,
            "same-label double roots are retained and telescope by sign");
}

void test_uncertified_event_correction_throws()
{
    const CartesianGrid3D grid(
        {0.0, 0.0, 0.0}, {0.25, 0.25, 0.25}, {4, 4, 4},
        DofLayout3D::Node);
    GridEdgeEvent3D event = build({crossing(0.5, 1)}, true).front();
    event.id = {grid.index(2, 2, 2), grid.index(3, 2, 2), 0};
    event.point = {0.625, 0.5, 0.5};
    event.certification = GridEdgeEventCertification3D::NearTangentEdge;
    event.canonical_sign = 0;
    bool threw = false;
    try {
        (void)native_grid_edge_event_correction_ops_3d(grid, event);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw,
            "uncertified physical events fail closed during PDE setup");
}

void test_unmerged_coincident_feature_owners_fail_closed()
{
    NurbsSurfaceCrossing3D first = crossing(0.4, 1, 0);
    NurbsSurfaceCrossing3D second = crossing(0.4, 1, 1);
    first.feature_edge_contact = true;
    second.feature_edge_contact = true;
    bool threw = false;
    try {
        (void)build({first, second}, false);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw,
            "unmerged coincident feature owners cannot become two PDE events");
}

void test_native_owner_center_selection_respects_g1_sheets()
{
    GridEdgeEvent3D event = build({crossing(0.5, 1, 0)}, true).front();
    require(event.owners.size() == 1,
            "synthetic transverse event has one canonical owner");

    // Patches 0/1 and 2/3 form two distinct transitive G1 sheets.  A center
    // on the wrong sheet is deliberately much closer to the physical event.
    const std::vector<int> patch_g1_components = {0, 0, 2, 2};
    const std::vector<P2NativeSheetCenterCandidate3D> centers = {
        {10, 100, 1, {0.70, 0.0, 0.0}},
        {20, 200, 3, {0.5001, 0.0, 0.0}}};
    const auto single = select_p2_native_event_owner_center_3d(
        event, patch_g1_components, centers, 1.0e-14);
    require(single.owner_index == 0 && single.center_candidate_index == 0,
            "nearest center on another C0 sheet is never paired with an owner");

    // At a feature event both incident sheets are legal.  Equal-distance
    // pairs choose the owner with the stronger certified transversality.
    event.owners.front().transversality = 0.25;
    NurbsSurfaceRootOwner3D second_owner = event.owners.front();
    second_owner.patch_index = 2;
    second_owner.u = 0.75;
    second_owner.normal = {0.0, 1.0, 0.0};
    second_owner.transversality = 0.75;
    event.owners.push_back(second_owner);
    const std::vector<P2NativeSheetCenterCandidate3D> tied_centers = {
        {30, 300, 1, {0.60, 0.0, 0.0}},
        {40, 400, 3, {0.40, 0.0, 0.0}}};
    const auto feature = select_p2_native_event_owner_center_3d(
        event, patch_g1_components, tied_centers, 1.0e-12);
    require(feature.owner_index == 1
                && feature.center_candidate_index == 1,
            "feature tie selects the most transverse same-sheet owner/center pair");

    event.owners.resize(1);
    bool threw = false;
    try {
        (void)select_p2_native_event_owner_center_3d(
            event, patch_g1_components,
            {{50, 500, 3, {0.5001, 0.0, 0.0}}}, 1.0e-14);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw,
            "native event ownership fails closed without a same-G1 center");
}

} // namespace

int main()
{
    try {
        test_same_label_double_root_is_not_filtered();
        test_three_events_have_ordered_ids_and_signs();
        test_reverse_view_reverses_order_parameter_and_sign_only();
        test_uncertified_contacts_and_tangencies_fail_closed();
        test_every_uncertified_root_set_requires_targeted_retry();
        test_feature_orientation_uses_component_containment();
        test_close_feature_probes_stay_between_neighboring_events();
        test_multi_event_corrections_preserve_ids_and_reverse_signs();
        test_uncertified_event_correction_throws();
        test_unmerged_coincident_feature_owners_fail_closed();
        test_native_owner_center_selection_respects_g1_sheets();
        std::cout << "grid-edge event 3D tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "grid-edge event 3D test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
