#include "src/support/trace/restrict_crossing_selector_3d.hpp"

#include <cmath>
#include <limits>

namespace kfbim::app3d {

OpenSegmentContinuation3D evaluate_open_segment_continuation_3d(
    const SegmentPhysicalEventSequence3D& open_sequence,
    bool start_inside,
    bool desired_inside)
{
    OpenSegmentContinuation3D result;
    const bool certified =
        open_sequence.kind == SegmentPhysicalEventSequenceKind3D::Certified;
    const bool no_crossing = open_sequence.kind
        == SegmentPhysicalEventSequenceKind3D::NoCrossing;
    if (!certified && !no_crossing) {
        result.kind = OpenSegmentContinuationKind3D::UncertifiedSequence;
        return result;
    }

    const int event_count = static_cast<int>(open_sequence.events.size());
    if (open_sequence.physical_event_count != event_count
        || (certified && event_count == 0)
        || (no_crossing && event_count != 0)) {
        result.kind =
            OpenSegmentContinuationKind3D::InconsistentSequenceMetadata;
        return result;
    }

    int state = start_inside ? 1 : 0;
    double previous_parameter = -std::numeric_limits<double>::infinity();
    for (const SegmentPhysicalCrossingEvent3D& event
         : open_sequence.events) {
        if (!std::isfinite(event.edge_parameter)
            || event.edge_parameter < 0.0
            || !(event.edge_parameter < 1.0)
            || !(event.edge_parameter > previous_parameter)) {
            result.kind = OpenSegmentContinuationKind3D::InvalidEventOrder;
            return result;
        }
        previous_parameter = event.edge_parameter;
        if (event.continuation_sign < -1
            || event.continuation_sign > 1) {
            result.kind = OpenSegmentContinuationKind3D::InvalidContinuation;
            return result;
        }
        const int next_state = state + event.continuation_sign;
        if (next_state < 0 || next_state > 1) {
            result.kind = OpenSegmentContinuationKind3D::InvalidContinuation;
            return result;
        }
        state = next_state;
        ++result.processed_event_count;
    }

    result.state_after_open_events = state;
    const int desired_state = desired_inside ? 1 : 0;
    if (state == desired_state) {
        result.kind = OpenSegmentContinuationKind3D::Complete;
        return result;
    }
    result.kind = OpenSegmentContinuationKind3D::ExactEndpointRequired;
    result.required_endpoint_sign = desired_state - state;
    return result;
}

} // namespace kfbim::app3d
