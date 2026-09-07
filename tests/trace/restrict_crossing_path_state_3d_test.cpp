#include "src/support/trace/restrict_crossing_selector_3d.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using kfbim::app3d::OpenSegmentContinuationKind3D;
using kfbim::app3d::SegmentPhysicalCrossingEvent3D;
using kfbim::app3d::SegmentPhysicalEventSequence3D;
using kfbim::app3d::SegmentPhysicalEventSequenceKind3D;
using kfbim::app3d::evaluate_open_segment_continuation_3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

SegmentPhysicalEventSequence3D certified_sequence(
    const std::vector<std::pair<double, int>>& parameter_signs)
{
    SegmentPhysicalEventSequence3D result;
    result.kind = SegmentPhysicalEventSequenceKind3D::Certified;
    result.input_crossing_count =
        static_cast<int>(parameter_signs.size());
    result.physical_event_count =
        static_cast<int>(parameter_signs.size());
    for (const auto& parameter_sign : parameter_signs) {
        SegmentPhysicalCrossingEvent3D event;
        event.edge_parameter = parameter_sign.first;
        event.continuation_sign = parameter_sign.second;
        result.events.push_back(std::move(event));
    }
    return result;
}

void test_equal_endpoint_labels_keep_both_open_events()
{
    // outside -> inside -> outside.  The support node and requested trace side
    // have the same label, but both physical crossings remain necessary.
    const auto sequence =
        certified_sequence({{0.2, 1}, {0.7, -1}});
    const auto decision = evaluate_open_segment_continuation_3d(
        sequence, false, false);
    require(decision.kind == OpenSegmentContinuationKind3D::Complete,
            "same-label double crossing completes without an endpoint toggle");
    require(!decision.exact_endpoint_required()
                && decision.state_after_open_events == 0
                && decision.processed_event_count == 2,
            "same-label path consumes both certified events");
}

void test_endpoint_is_added_only_for_a_state_mismatch()
{
    SegmentPhysicalEventSequence3D empty;
    empty.kind = SegmentPhysicalEventSequenceKind3D::NoCrossing;
    const auto mismatch = evaluate_open_segment_continuation_3d(
        empty, false, true);
    require(mismatch.kind
                == OpenSegmentContinuationKind3D::ExactEndpointRequired
                && mismatch.exact_endpoint_required()
                && mismatch.state_after_open_events == 0
                && mismatch.required_endpoint_sign == 1,
            "an unmatched interior trace side requires the exact endpoint");

    const auto entering = certified_sequence({{0.4, 1}});
    const auto matched = evaluate_open_segment_continuation_3d(
        entering, false, true);
    require(matched.kind == OpenSegmentContinuationKind3D::Complete
                && !matched.exact_endpoint_required()
                && matched.state_after_open_events == 1,
            "an open event that reaches the requested side suppresses endpoint completion");

    const auto double_crossing =
        certified_sequence({{0.2, 1}, {0.7, -1}});
    const auto opposite_target = evaluate_open_segment_continuation_3d(
        double_crossing, false, true);
    require(opposite_target.kind
                == OpenSegmentContinuationKind3D::ExactEndpointRequired
                && opposite_target.required_endpoint_sign == 1,
            "endpoint completion follows the post-event state, not the start label");

}

void test_uncertified_and_contact_sequences_fail_closed()
{
    const std::vector<SegmentPhysicalEventSequenceKind3D> rejected = {
        SegmentPhysicalEventSequenceKind3D::Overlap,
        SegmentPhysicalEventSequenceKind3D::IncompleteRootSet,
        SegmentPhysicalEventSequenceKind3D::AmbiguousPhysicalCluster,
        SegmentPhysicalEventSequenceKind3D::NonTransverseContact,
        SegmentPhysicalEventSequenceKind3D::InconsistentContinuationSign};
    for (const auto kind : rejected) {
        auto sequence = certified_sequence({{0.5, 1}});
        sequence.kind = kind;
        const auto decision = evaluate_open_segment_continuation_3d(
            sequence, false, true);
        require(decision.kind
                    == OpenSegmentContinuationKind3D::UncertifiedSequence
                    && !decision.exact_endpoint_required(),
                "uncertified/contact sequences cannot request an endpoint fallback");
    }
}

void test_certified_feature_contact_preserves_state()
{
    const auto contact = certified_sequence({{0.4, 0}});
    const auto matched = evaluate_open_segment_continuation_3d(
        contact, true, true);
    require(matched.kind == OpenSegmentContinuationKind3D::Complete
                && matched.state_after_open_events == 1
                && matched.processed_event_count == 1,
            "a certified feature contact is audited without changing side");

    const auto endpoint_needed = evaluate_open_segment_continuation_3d(
        contact, false, true);
    require(endpoint_needed.kind
                == OpenSegmentContinuationKind3D::ExactEndpointRequired
                && endpoint_needed.required_endpoint_sign == 1,
            "a zero-sign contact cannot hide a required trace endpoint");
}

void test_invalid_order_sign_and_transition_fail_closed()
{
    const auto unordered =
        certified_sequence({{0.7, 1}, {0.2, -1}});
    require(evaluate_open_segment_continuation_3d(
                unordered, false, false)
                .kind == OpenSegmentContinuationKind3D::InvalidEventOrder,
            "unordered events fail closed");

    const auto endpoint_in_open_sequence =
        certified_sequence({{1.0, 1}});
    require(evaluate_open_segment_continuation_3d(
                endpoint_in_open_sequence, false, true)
                .kind == OpenSegmentContinuationKind3D::InvalidEventOrder,
            "the open sequence cannot silently contain the trace endpoint");

    const auto invalid_sign = certified_sequence({{0.4, 2}});
    require(evaluate_open_segment_continuation_3d(
                invalid_sign, false, true)
                .kind
                == OpenSegmentContinuationKind3D::InvalidContinuation,
            "an event sign outside -1/0/+1 fails closed");

    const auto impossible_transition = certified_sequence({{0.4, 1}});
    require(evaluate_open_segment_continuation_3d(
                impossible_transition, true, false)
                .kind == OpenSegmentContinuationKind3D::InvalidContinuation,
            "events cannot leave the binary inside/outside state space");

    auto inconsistent_empty = certified_sequence({});
    require(evaluate_open_segment_continuation_3d(
                inconsistent_empty, false, false)
                .kind
                == OpenSegmentContinuationKind3D::InconsistentSequenceMetadata,
            "Certified requires at least one event");
}

} // namespace

int main()
{
    try {
        test_equal_endpoint_labels_keep_both_open_events();
        test_endpoint_is_added_only_for_a_state_mismatch();
        test_uncertified_and_contact_sequences_fail_closed();
        test_certified_feature_contact_preserves_state();
        test_invalid_order_sign_and_transition_fail_closed();
        std::cout << "restrict crossing path-state 3D tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "restrict crossing path-state 3D test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
