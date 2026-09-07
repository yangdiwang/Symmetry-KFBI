#include "native_endpoint_path_adapter_3d.hpp"

#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace kfbim::app3d {
namespace {

void require_native(bool value, const char* detail)
{
    if (!value)
        throw std::invalid_argument(
            std::string("native endpoint event adapter: ") + detail);
}

bool finite_interval(const geometry3d::RootInterval3D& interval)
{
    return std::isfinite(interval.lower) && std::isfinite(interval.upper)
        && interval.lower <= interval.upper;
}

void validate_root(const geometry3d::CertifiedNativePathRoot3D& root,
                   bool endpoint)
{
    const auto& certificate = root.certificate;
    const auto& transition = root.transition;
    const auto& point = root.representative;
    require_native(certificate.kind == (endpoint
        ? geometry3d::RootProofKind3D::KnownEndpointUnique
        : geometry3d::RootProofKind3D::ExistsUnique), "wrong root proof kind");
    require_native(certificate.existence_proved
                       && certificate.uniqueness_proved,
                   "missing existence or uniqueness proof");
    require_native(certificate.proof_id != 0 && certificate.event_id != 0,
                   "missing proof/event identity");
    require_native(std::isfinite(certificate.contraction_bound)
                       && certificate.contraction_bound >= 0.0
                       && certificate.contraction_bound < 1.0,
                   "invalid uniqueness bound");
    require_native(finite_interval(certificate.t_interval)
                       && finite_interval(certificate.root_enclosure.u)
                       && finite_interval(certificate.root_enclosure.v)
                       && finite_interval(certificate.uniqueness_box.u)
                       && finite_interval(certificate.uniqueness_box.v),
                   "invalid root enclosure");
    require_native(certificate.patch_index == point.patch_index
                       && point.patch_index >= 0 && point.component >= 0
                       && point.point.allFinite() && point.normal.allFinite()
                       && std::isfinite(point.u) && std::isfinite(point.v)
                       && std::isfinite(point.edge_parameter),
                   "invalid representative/owner");
    require_native(certificate.uniqueness_box.u.lower
                           <= certificate.root_enclosure.u.lower
                       && certificate.root_enclosure.u.upper
                           <= certificate.uniqueness_box.u.upper
                       && certificate.uniqueness_box.v.lower
                           <= certificate.root_enclosure.v.lower
                       && certificate.root_enclosure.v.upper
                           <= certificate.uniqueness_box.v.upper
                       && certificate.root_enclosure.u.lower <= point.u
                       && point.u <= certificate.root_enclosure.u.upper
                       && certificate.root_enclosure.v.lower <= point.v
                       && point.v <= certificate.root_enclosure.v.upper
                       && certificate.t_interval.lower <= point.edge_parameter
                       && point.edge_parameter <= certificate.t_interval.upper,
                   "representative/enclosure lies outside its certificate");
    require_native(root.accuracy.certified
                       && std::isfinite(root.accuracy.point_error_bound)
                       && root.accuracy.point_error_bound >= 0.0
                       && std::isfinite(root.accuracy.parameter_error_bound)
                       && root.accuracy.parameter_error_bound >= 0.0
                       && std::isfinite(root.accuracy.normal_error_bound)
                       && root.accuracy.normal_error_bound >= 0.0,
                   "uncertified representative accuracy");
    require_native(transition.certified
                       && std::isfinite(transition.oriented_dot_lower)
                       && std::isfinite(transition.oriented_dot_upper)
                       && transition.oriented_dot_lower
                            <= transition.oriented_dot_upper,
                   "uncertified event direction");
    const bool entering = transition.oriented_dot_upper < 0.0;
    const bool leaving = transition.oriented_dot_lower > 0.0;
    require_native(entering || leaving, "nontransverse event is unsupported");
    require_native(transition.sign == (entering ? 1 : -1)
                       && transition.incoming_inside == (entering ? 0 : 1)
                       && transition.outgoing_inside == (entering ? 1 : 0),
                   "event direction disagrees with its geometric proof");
    if (endpoint) {
        require_native(certificate.t_interval.lower == 1.0
                           && certificate.t_interval.upper == 1.0
                           && point.edge_parameter == 1.0,
                       "native endpoint is not t=1");
    } else {
        require_native(certificate.t_interval.lower > 0.0
                           && certificate.t_interval.upper < 1.0
                           && point.edge_parameter > 0.0
                           && point.edge_parameter < 1.0,
                       "root is not strictly inside the open segment");
    }
}

} // namespace

NativeEndpointEventSequence3D select_native_endpoint_path_events_3d(
    const geometry3d::NativeEndpointPathResult3D& path,
    bool start_inside, bool desired_inside)
{
    require_native(path.status == geometry3d::PathStatus3D::Certified
                       && path.coverage.complete
                       && path.coverage.order_certified
                       && path.coverage.transitions_certified
                       && path.coverage.representatives_certified
                       && path.coverage.unresolved_regions == 0,
                   "path did not pass all four certification gates");
    require_native(path.endpoint.has_value(), "missing native endpoint proof");
    require_native(path.post_endpoint_roots.empty(),
                   "first-phase adapter only accepts a closed native query");
    require_native(path.coverage.certified_start_inside
                       == (start_inside ? 1 : 0),
                   "grid label disagrees with certified start state");
    validate_root(*path.endpoint, true);

    NativeEndpointEventSequence3D result;
    std::set<std::uint64_t> event_ids;
    event_ids.insert(path.endpoint->certificate.event_id);
    double previous_upper = 0.0;
    int state = start_inside ? 1 : 0;
    const auto append = [&](const geometry3d::CertifiedNativePathRoot3D& root) {
        require_native(result.intersection.crossings.size()
                           < static_cast<std::size_t>(
                               std::numeric_limits<int>::max()),
                       "event count exceeds index range");
        const int index = static_cast<int>(result.intersection.crossings.size());
        result.intersection.crossings.push_back(root.representative);
        SegmentPhysicalCrossingEvent3D event;
        event.crossing_indices.push_back(index);
        event.edge_parameter = root.representative.edge_parameter;
        event.point = root.representative.point;
        event.component = root.representative.component;
        event.continuation_sign = root.transition.sign;
        event.feature_edge_contact = root.representative.feature_edge_contact;
        event.native_event_id = root.certificate.event_id;
        event.native_proof_id = root.certificate.proof_id;
        result.sequence.events.push_back(std::move(event));
        result.sequence.input_crossing_count = index + 1;
        result.sequence.physical_event_count = index + 1;
        result.sequence.kind = SegmentPhysicalEventSequenceKind3D::Certified;
    };

    for (const auto& root : path.open_roots) {
        validate_root(root, false);
        require_native(event_ids.insert(root.certificate.event_id).second,
                       "duplicate physical event identity");
        require_native(previous_upper < root.certificate.t_interval.lower,
                       "root certificates do not establish strict order");
        require_native(root.transition.incoming_inside == state,
                       "open event disagrees with incoming state");
        state = root.transition.outgoing_inside;
        previous_upper = root.certificate.t_interval.upper;
        append(root);
    }
    require_native(state == path.endpoint->transition.incoming_inside,
                   "open events disagree with certified endpoint approach side");
    const auto continuation = evaluate_open_segment_continuation_3d(
        result.sequence, start_inside, desired_inside);
    require_native(continuation.kind == OpenSegmentContinuationKind3D::Complete
                       || continuation.exact_endpoint_required(),
                   "open-segment continuation failed");
    if (continuation.exact_endpoint_required()) {
        require_native(continuation.required_endpoint_sign
                           == path.endpoint->transition.sign,
                       "requested trace side disagrees with endpoint direction");
        append(*path.endpoint);
        state = path.endpoint->transition.outgoing_inside;
    }
    require_native(state == (desired_inside ? 1 : 0),
                   "conditional endpoint did not reach requested trace side");
    return result;
}

} // namespace kfbim::app3d
