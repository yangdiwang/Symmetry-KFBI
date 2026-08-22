#include "nurbs_cartesian_domain_3d.hpp"

#include "nurbs_bezier_extraction_3d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace kfbim::geometry3d {
namespace {

using PreprocessClock3D = std::chrono::steady_clock;

double elapsed_seconds(PreprocessClock3D::time_point begin,
                       PreprocessClock3D::time_point end)
{
    return std::chrono::duration<double>(end - begin).count();
}

constexpr std::uint64_t kNodeMask =
    (std::uint64_t{1} << 62) - std::uint64_t{1};

struct ClosedIntInterval {
    int lower = 1;
    int upper = 0;

    bool empty() const
    {
        return lower > upper;
    }
};
struct EdgeCrossingRange3D {
    int begin = 0;
    int count = 0;
    int confirmed_transverse_count = 0;
    bool changes_inside_outside = false;
    bool changes_component_membership = false;
};
struct EdgeIntersectionRecord3D {
    EdgeCrossingRange3D range;
    int event_begin = 0;
    int event_count = 0;
    NurbsCartesianEdgeClassification3D classification;
};

std::size_t checked_size_product(std::size_t first,
                                 std::size_t second,
                                 const char* message)
{
    if (first != 0
        && second > std::numeric_limits<std::size_t>::max() / first) {
        throw std::overflow_error(message);
    }
    return first * second;
}

std::size_t checked_size_add(std::size_t first,
                             std::size_t second,
                             const char* message)
{
    if (second > std::numeric_limits<std::size_t>::max() - first)
        throw std::overflow_error(message);
    return first + second;
}

int checked_size_to_int(std::size_t value, const char* message)
{
    if (value > static_cast<std::size_t>(
                    std::numeric_limits<int>::max())) {
        throw std::overflow_error(message);
    }
    return static_cast<int>(value);
}

std::vector<int> build_patch_g1_components(
    const NurbsSurfaceModel3D& model)
{
    const int patch_count = model.num_patches();
    std::vector<int> parent(static_cast<std::size_t>(patch_count));
    for (int patch = 0; patch < patch_count; ++patch)
        parent[static_cast<std::size_t>(patch)] = patch;
    const auto find = [&](int patch, const auto& self) -> int {
        int& root = parent[static_cast<std::size_t>(patch)];
        if (root != patch)
            root = self(root, self);
        return root;
    };
    for (const NurbsPatchEdgeConnection3D& connection : model.connections()) {
        if (!connection.g1)
            continue;
        int first = find(connection.first.patch, find);
        int second = find(connection.second.patch, find);
        if (first == second)
            continue;
        const int root = std::min(first, second);
        parent[static_cast<std::size_t>(first)] = root;
        parent[static_cast<std::size_t>(second)] = root;
    }
    for (int patch = 0; patch < patch_count; ++patch)
        parent[static_cast<std::size_t>(patch)] = find(patch, find);
    return parent;
}

int checked_add_int(int first, int second, const char* message)
{
    if ((second > 0
         && first > std::numeric_limits<int>::max() - second)
        || (second < 0
            && first < std::numeric_limits<int>::min() - second)) {
        throw std::overflow_error(message);
    }
    return first + second;
}

void checked_accumulate_int(int& total,
                            int increment,
                            const char* message)
{
    if (total < 0 || increment < 0)
        throw std::overflow_error(message);
    total = checked_add_int(total, increment, message);
}

ClosedIntInterval clamped_closed_integer_interval(
    double lower,
    double upper,
    int target_lower,
    int target_upper)
{
    if (!std::isfinite(lower) || !std::isfinite(upper)) {
        throw std::overflow_error(
            "non-finite Cartesian candidate index range");
    }
    if (target_lower > target_upper || lower > upper)
        return {};

    const double target_lower_double =
        static_cast<double>(target_lower);
    const double target_upper_double =
        static_cast<double>(target_upper);
    if (upper < target_lower_double || lower > target_upper_double)
        return {};

    const double clipped_lower = std::max(lower, target_lower_double);
    const double clipped_upper = std::min(upper, target_upper_double);
    if (!(clipped_lower >= target_lower_double
          && clipped_lower <= target_upper_double
          && clipped_upper >= target_lower_double
          && clipped_upper <= target_upper_double
          && clipped_lower <= clipped_upper
          && std::trunc(clipped_lower) == clipped_lower
          && std::trunc(clipped_upper) == clipped_upper)) {
        throw std::overflow_error(
            "Cartesian candidate index range cannot be represented");
    }
    return {static_cast<int>(clipped_lower),
            static_cast<int>(clipped_upper)};
}

ClosedIntInterval candidate_index_interval(double lower,
                                           double upper,
                                           double origin,
                                           double spacing,
                                           double tolerance,
                                           bool edge_starts,
                                           int target_upper)
{
    const double expanded_lower = lower - origin - tolerance;
    const double expanded_upper = upper - origin + tolerance;
    if (!std::isfinite(expanded_lower)
        || !std::isfinite(expanded_upper)) {
        throw std::overflow_error(
            "non-finite Cartesian candidate index range");
    }
    const double normalized_lower = expanded_lower / spacing;
    const double normalized_upper = expanded_upper / spacing;
    if (!std::isfinite(normalized_lower)
        || !std::isfinite(normalized_upper)) {
        throw std::overflow_error(
            "non-finite Cartesian candidate index range");
    }
    double integer_lower = std::ceil(normalized_lower);
    const double integer_upper = std::floor(normalized_upper);
    if (edge_starts)
        integer_lower -= 1.0;
    return clamped_closed_integer_interval(
        integer_lower, integer_upper, 0, target_upper);
}

std::uint64_t edge_key(int axis, int start_node)
{
    return (std::uint64_t(axis) << 62)
        | std::uint64_t(start_node);
}

void accumulate_intersection_diagnostics(
    NurbsSurfaceIntersectionDiagnostics3D& total,
    const NurbsSurfaceIntersectionDiagnostics3D& increment)
{
    checked_accumulate_int(
        total.candidate_elements, increment.candidate_elements,
        "NURBS intersection candidate diagnostic overflow");
    checked_accumulate_int(
        total.bvh_candidate_elements, increment.bvh_candidate_elements,
        "NURBS BVH-candidate diagnostic overflow");
    checked_accumulate_int(
        total.mapped_candidate_elements, increment.mapped_candidate_elements,
        "NURBS mapped-candidate diagnostic overflow");
    total.maximum_candidate_elements_per_edge = std::max(
        total.maximum_candidate_elements_per_edge,
        increment.maximum_candidate_elements_per_edge);
    checked_accumulate_int(
        total.triangle_seed_hits, increment.triangle_seed_hits,
        "NURBS triangle-seed diagnostic overflow");
    checked_accumulate_int(
        total.triangle_seed_misses_recovered,
        increment.triangle_seed_misses_recovered,
        "NURBS triangle-seed recovery diagnostic overflow");
    checked_accumulate_int(
        total.subdivision_boxes, increment.subdivision_boxes,
        "NURBS subdivision diagnostic overflow");
    checked_accumulate_int(
        total.newton_attempts, increment.newton_attempts,
        "NURBS Newton-attempt diagnostic overflow");
    checked_accumulate_int(
        total.newton_iterations, increment.newton_iterations,
        "NURBS Newton-iteration diagnostic overflow");
    checked_accumulate_int(
        total.early_unique_certificate_attempts,
        increment.early_unique_certificate_attempts,
        "NURBS early unique-root certificate diagnostic overflow");
    checked_accumulate_int(
        total.early_unique_certificate_successes,
        increment.early_unique_certificate_successes,
        "NURBS early unique-root certificate success diagnostic overflow");
    checked_accumulate_int(
        total.planar_analytic_hits, increment.planar_analytic_hits,
        "NURBS planar analytic-hit diagnostic overflow");
    checked_accumulate_int(
        total.planar_analytic_misses, increment.planar_analytic_misses,
        "NURBS planar analytic-miss diagnostic overflow");
    checked_accumulate_int(
        total.planar_analytic_fallbacks,
        increment.planar_analytic_fallbacks,
        "NURBS planar analytic-fallback diagnostic overflow");
    checked_accumulate_int(
        total.closest_point_prefilter_attempts,
        increment.closest_point_prefilter_attempts,
        "NURBS closest-point prefilter-attempt diagnostic overflow");
    checked_accumulate_int(
        total.closest_point_prefilter_certified_hits,
        increment.closest_point_prefilter_certified_hits,
        "NURBS closest-point prefilter-hit diagnostic overflow");
    checked_accumulate_int(
        total.closest_point_prefilter_certified_misses,
        increment.closest_point_prefilter_certified_misses,
        "NURBS closest-point prefilter-miss diagnostic overflow");
    checked_accumulate_int(
        total.closest_point_prefilter_fallbacks,
        increment.closest_point_prefilter_fallbacks,
        "NURBS closest-point prefilter-fallback diagnostic overflow");
    checked_accumulate_int(
        total.certified_fallback_elements,
        increment.certified_fallback_elements,
        "NURBS certified-fallback diagnostic overflow");
    checked_accumulate_int(
        total.same_patch_deduplications,
        increment.same_patch_deduplications,
        "NURBS same-patch diagnostic overflow");
    checked_accumulate_int(
        total.seam_deduplications,
        increment.seam_deduplications,
        "NURBS seam diagnostic overflow");
    checked_accumulate_int(
        total.unresolved_candidates,
        increment.unresolved_candidates,
        "NURBS unresolved-candidate diagnostic overflow");
    total.unresolved_longitudinal_intervals.insert(
        total.unresolved_longitudinal_intervals.end(),
        increment.unresolved_longitudinal_intervals.begin(),
        increment.unresolved_longitudinal_intervals.end());
    total.maximum_subdivision_depth_reached = std::max(
        total.maximum_subdivision_depth_reached,
        increment.maximum_subdivision_depth_reached);
    checked_accumulate_int(
        total.terminal_certificate_boxes,
        increment.terminal_certificate_boxes,
        "NURBS terminal-certificate diagnostic overflow");
    total.maximum_terminal_certificate_depth_reached = std::max(
        total.maximum_terminal_certificate_depth_reached,
        increment.maximum_terminal_certificate_depth_reached);
    checked_accumulate_int(
        total.closest_point_attempts, increment.closest_point_attempts,
        "NURBS closest-point attempt diagnostic overflow");
    checked_accumulate_int(
        total.closest_point_iterations, increment.closest_point_iterations,
        "NURBS closest-point iteration diagnostic overflow");
    checked_accumulate_int(
        total.roots_recovered_by_closest_point,
        increment.roots_recovered_by_closest_point,
        "NURBS closest-point recovery diagnostic overflow");
    checked_accumulate_int(
        total.terminal_misses_by_closest_point,
        increment.terminal_misses_by_closest_point,
        "NURBS closest-point terminal-miss diagnostic overflow");
    checked_accumulate_int(
        total.closest_point_failures, increment.closest_point_failures,
        "NURBS closest-point failure diagnostic overflow");
    checked_accumulate_int(
        total.sample_seed_candidates, increment.sample_seed_candidates,
        "NURBS sample-candidate diagnostic overflow");
    checked_accumulate_int(
        total.sample_seeds_accepted, increment.sample_seeds_accepted,
        "NURBS sample-seed diagnostic overflow");
    checked_accumulate_int(
        total.roots_recovered_by_sample_seed,
        increment.roots_recovered_by_sample_seed,
        "NURBS sample-seed root diagnostic overflow");
    total.maximum_sample_seeds_per_element = std::max(
        total.maximum_sample_seeds_per_element,
        increment.maximum_sample_seeds_per_element);
    checked_accumulate_int(
        total.stationary_solve_attempts,
        increment.stationary_solve_attempts,
        "NURBS stationary-solve diagnostic overflow");
    checked_accumulate_int(
        total.stationary_solve_converged,
        increment.stationary_solve_converged,
        "NURBS stationary-convergence diagnostic overflow");
    checked_accumulate_int(
        total.stationary_witnesses, increment.stationary_witnesses,
        "NURBS stationary-witness diagnostic overflow");
    checked_accumulate_int(
        total.root_pairs_protected_by_stationary_witness,
        increment.root_pairs_protected_by_stationary_witness,
        "NURBS protected-root diagnostic overflow");
    checked_accumulate_int(
        total.ambiguous_root_clusters,
        increment.ambiguous_root_clusters,
        "NURBS ambiguous-cluster diagnostic overflow");
    checked_accumulate_int(
        total.non_g1_topology_merges,
        increment.non_g1_topology_merges,
        "NURBS non-G1 topology-merge diagnostic overflow");
    checked_accumulate_int(
        total.high_degree_fallbacks,
        increment.high_degree_fallbacks,
        "NURBS high-degree fallback diagnostic overflow");
}

Eigen::Vector3d as_vector(const std::array<double, 3>& values)
{
    return {values[0], values[1], values[2]};
}

std::string strict_box_error(const NurbsAabb3D& surface,
                             const Eigen::Vector3d& box_lower,
                             const Eigen::Vector3d& box_upper)
{
    std::ostringstream message;
    message << std::setprecision(17)
            << "NURBS surface must lie strictly inside Cartesian box"
            << ": surface lower=(" << surface.lower.x() << ','
            << surface.lower.y() << ',' << surface.lower.z() << ')'
            << " upper=(" << surface.upper.x() << ','
            << surface.upper.y() << ',' << surface.upper.z() << ')'
            << ", Cartesian box lower=(" << box_lower.x() << ','
            << box_lower.y() << ',' << box_lower.z() << ')'
            << " upper=(" << box_upper.x() << ','
            << box_upper.y() << ',' << box_upper.z() << ')';
    return message.str();
}

} // namespace

namespace {

NurbsSurfaceRootOwner3D event_owner_from_crossing(
    const NurbsSurfaceCrossing3D& crossing)
{
    NurbsSurfaceRootOwner3D owner;
    owner.patch_index = crossing.patch_index;
    owner.u = crossing.u;
    owner.v = crossing.v;
    owner.point = crossing.point;
    owner.normal = crossing.normal;
    owner.residual = crossing.residual;
    owner.transversality = crossing.transversality;
    owner.feature_edge_contact = crossing.feature_edge_contact;
    owner.reliable_transversality_tolerance =
        crossing.reliable_transversality_tolerance;
    return owner;
}

struct EventNormalCertification3D {
    GridEdgeEventCertification3D certification =
        GridEdgeEventCertification3D::IncompleteRootSet;
    int canonical_sign = 0;
    bool containment_eligible = false;
};

EventNormalCertification3D event_normal_certification(
    const NurbsSurfaceCrossing3D& crossing,
    const NurbsCartesianEdgeClassification3D& classification,
    bool complete_edge_record,
    const Eigen::Vector3d& canonical_direction)
{
    if (!complete_edge_record || !classification.root_count_known)
        return {GridEdgeEventCertification3D::IncompleteRootSet, 0, false};
    if (!classification.parity_known_from_roots)
        return {GridEdgeEventCertification3D::UnknownParity, 0, false};
    if (classification.has_near_tangent_candidate)
        return {GridEdgeEventCertification3D::NearTangentEdge, 0, false};
    const double oriented_normal_dot =
        crossing.normal.dot(canonical_direction);
    std::vector<NurbsSurfaceRootOwner3D> owners = crossing.owners;
    if (owners.empty())
        owners.push_back(event_owner_from_crossing(crossing));
    const bool needs_incident_owner_agreement =
        crossing.feature_edge_contact || owners.size() > 1;
    if (needs_incident_owner_agreement) {
        int common_sign = 0;
        for (const NurbsSurfaceRootOwner3D& owner : owners) {
            const double dot = owner.normal.dot(canonical_direction);
            if (!owner.normal.allFinite()
                || !std::isfinite(owner.transversality)
                || !std::isfinite(
                       owner.reliable_transversality_tolerance)
                || owner.reliable_transversality_tolerance <= 0.0
                || owner.transversality
                       <= owner.reliable_transversality_tolerance
                || !std::isfinite(dot)
                || std::abs(dot)
                       <= owner.reliable_transversality_tolerance) {
                return {GridEdgeEventCertification3D::FeatureContact,
                        0, true};
            }
            const int sign = dot > 0.0 ? 1 : -1;
            if (common_sign != 0 && common_sign != sign)
                return {GridEdgeEventCertification3D::FeatureContact,
                        0, true};
            common_sign = sign;
        }
        // A declared topological feature is still one physical event when all
        // coincident owners certify the same transverse orientation.
        return common_sign == 0
            ? EventNormalCertification3D{
                  GridEdgeEventCertification3D::FeatureContact, 0, true}
            : EventNormalCertification3D{
                  GridEdgeEventCertification3D::CertifiedTransverse,
                  common_sign, false};
    }
    if (!std::isfinite(crossing.transversality)
        || !std::isfinite(crossing.reliable_transversality_tolerance)
        || crossing.reliable_transversality_tolerance <= 0.0
        || crossing.transversality
               <= crossing.reliable_transversality_tolerance
        || !std::isfinite(oriented_normal_dot)
        || std::abs(oriented_normal_dot)
               <= crossing.reliable_transversality_tolerance) {
        return {GridEdgeEventCertification3D::UnreliableTransversality,
                0, false};
    }
    return {GridEdgeEventCertification3D::CertifiedTransverse,
            oriented_normal_dot > 0.0 ? 1 : -1, false};
}

} // namespace

bool grid_edge_root_set_requires_targeted_retry_3d(
    const NurbsCartesianEdgeIntersections3D& intersections) noexcept
{
    return !intersections.root_count_known
        || !intersections.parity_known_from_roots
        || !intersections.ambiguous_clusters.empty()
        || intersections.has_near_tangent_candidate
        || intersections.diagnostics.unresolved_candidates != 0;
}

std::vector<GridEdgeEvent3D> build_canonical_grid_edge_events_3d(
    int node_a,
    int node_b,
    const Eigen::Vector3d& point_a,
    const Eigen::Vector3d& point_b,
    const std::vector<NurbsSurfaceCrossing3D>& crossings,
    const NurbsCartesianEdgeClassification3D& classification,
    const GridEdgeComponentContainment3D& component_contains)
{
    if (node_a < 0 || node_b < 0 || node_a == node_b) {
        throw std::invalid_argument(
            "grid-edge event construction requires two distinct nonnegative nodes");
    }
    if (!point_a.allFinite() || !point_b.allFinite()) {
        throw std::invalid_argument(
            "grid-edge event construction requires finite endpoints");
    }
    const bool input_reversed = node_a > node_b;
    const int first_node = std::min(node_a, node_b);
    const int second_node = std::max(node_a, node_b);
    const Eigen::Vector3d first_point =
        input_reversed ? point_b : point_a;
    const Eigen::Vector3d second_point =
        input_reversed ? point_a : point_b;
    const Eigen::Vector3d edge = second_point - first_point;
    const double edge_length = edge.norm();
    if (!std::isfinite(edge_length) || edge_length <= 0.0) {
        throw std::invalid_argument(
            "grid-edge event construction requires a nondegenerate edge");
    }
    const Eigen::Vector3d direction = edge / edge_length;
    const bool complete_edge_record = classification.queried
        && classification.confirmed_crossing_count == crossings.size()
        && classification.ambiguous_cluster_count == 0;

    struct OrderedCrossing {
        const NurbsSurfaceCrossing3D* crossing = nullptr;
        double canonical_parameter = 0.0;
    };
    std::vector<OrderedCrossing> ordered;
    ordered.reserve(crossings.size());
    for (const NurbsSurfaceCrossing3D& crossing : crossings) {
        if (!std::isfinite(crossing.edge_parameter)
            || crossing.edge_parameter < 0.0
            || crossing.edge_parameter > 1.0
            || crossing.component < 0 || !crossing.point.allFinite()
            || !crossing.normal.allFinite()) {
            throw std::invalid_argument(
                "grid-edge event construction received an invalid crossing");
        }
        ordered.push_back({
            &crossing,
            input_reversed
                ? 1.0 - crossing.edge_parameter
                : crossing.edge_parameter});
    }
    std::sort(
        ordered.begin(), ordered.end(),
        [](const OrderedCrossing& first, const OrderedCrossing& second) {
            if (first.canonical_parameter != second.canonical_parameter) {
                return first.canonical_parameter
                    < second.canonical_parameter;
            }
            if (first.crossing->component != second.crossing->component) {
                return first.crossing->component
                    < second.crossing->component;
            }
            if (first.crossing->patch_index
                != second.crossing->patch_index) {
                return first.crossing->patch_index
                    < second.crossing->patch_index;
            }
            if (first.crossing->u != second.crossing->u)
                return first.crossing->u < second.crossing->u;
            return first.crossing->v < second.crossing->v;
        });

    // A Cartesian-domain query is configured to merge every declared
    // topological owner of one feature event upstream.  If a caller supplies
    // two numerically indistinguishable feature roots anyway, do not assign
    // two ordinals (and hence double the PDE jump).  The uncertainty test uses
    // residual/transversality bounds; it is intentionally not a bare |dt|
    // tolerance and does not merge resolvable close roots.
    for (std::size_t index = 1; index < ordered.size(); ++index) {
        const NurbsSurfaceCrossing3D& first =
            *ordered[index - 1].crossing;
        const NurbsSurfaceCrossing3D& second =
            *ordered[index].crossing;
        if (!first.feature_edge_contact || !second.feature_edge_contact
            || first.component != second.component) {
            continue;
        }
        const auto parameter_uncertainty = [&](const auto& root) {
            const double normal_dot =
                std::abs(root.normal.dot(direction));
            const double guarded_dot = std::max(
                normal_dot,
                std::max(root.reliable_transversality_tolerance,
                         64.0 * std::numeric_limits<double>::epsilon()));
            return std::max(0.0, root.residual)
                     / (edge_length * guarded_dot)
                + 64.0 * std::numeric_limits<double>::epsilon();
        };
        const double t_uncertainty =
            parameter_uncertainty(first)
            + parameter_uncertainty(second);
        const double point_uncertainty =
            std::max(0.0, first.residual)
            + std::max(0.0, second.residual)
            + edge_length * t_uncertainty;
        if (std::abs(ordered[index].canonical_parameter
                     - ordered[index - 1].canonical_parameter)
                    <= t_uncertainty
            && (first.point - second.point).norm()
                   <= point_uncertainty) {
            throw std::runtime_error(
                "grid-edge event catalog received unmerged coincident feature owners");
        }
    }

    std::vector<GridEdgeEvent3D> result;
    result.reserve(ordered.size());
    for (std::size_t ordinal = 0; ordinal < ordered.size(); ++ordinal) {
        const NurbsSurfaceCrossing3D& crossing =
            *ordered[ordinal].crossing;
        EventNormalCertification3D orientation =
            event_normal_certification(
                crossing, classification, complete_edge_record, direction);
        if (orientation.containment_eligible && component_contains) {
            const double root_parameter =
                ordered[ordinal].canonical_parameter;
            const double left_parameter = ordinal == 0
                ? 0.0
                : ordered[ordinal - 1].canonical_parameter;
            const double right_parameter = ordinal + 1 == ordered.size()
                ? 1.0
                : ordered[ordinal + 1].canonical_parameter;
            const double before_parameter =
                left_parameter + 0.5 * (root_parameter - left_parameter);
            const double after_parameter =
                root_parameter + 0.5 * (right_parameter - root_parameter);

            // Midpoints maximize clearance from this root and its neighbors.
            // If floating point cannot represent both strict open intervals,
            // containment is not allowed to guess an orientation.
            if (left_parameter < before_parameter
                && before_parameter < root_parameter
                && root_parameter < after_parameter
                && after_parameter < right_parameter) {
                const Eigen::Vector3d before_point =
                    first_point + before_parameter * edge;
                const Eigen::Vector3d after_point =
                    first_point + after_parameter * edge;
                bool before_contains = false;
                bool after_contains = false;
                try {
                    before_contains = component_contains(
                        before_point, crossing.component);
                    after_contains = component_contains(
                        after_point, crossing.component);
                } catch (const std::exception& error) {
                    std::ostringstream message;
                    message
                        << "grid-edge feature containment could not certify "
                           "event "
                        << ordinal << " on edge (" << first_node << ','
                        << second_node << "): " << error.what();
                    throw std::runtime_error(message.str());
                }
                if (before_contains != after_contains) {
                    orientation.certification =
                        GridEdgeEventCertification3D::CertifiedTransverse;
                    // canonical_sign is inside_before - inside_after, equal
                    // to sign(n dot canonical_edge) on a smooth crossing.
                    orientation.canonical_sign =
                        static_cast<int>(before_contains)
                        - static_cast<int>(after_contains);
                }
            }
        }
        GridEdgeEvent3D event;
        event.id = {
            first_node,
            second_node,
            checked_size_to_int(
                ordinal, "grid-edge event ordinal exceeds int range")};
        event.canonical_parameter = ordered[ordinal].canonical_parameter;
        event.component = crossing.component;
        event.point = crossing.point;
        event.normal = crossing.normal;
        event.residual = crossing.residual;
        event.transversality = crossing.transversality;
        event.feature_edge_contact = crossing.feature_edge_contact;
        event.certification = orientation.certification;
        // Fail closed: a contact, tangent, or otherwise uncertified root is
        // not exposed as a signed jump event.
        if (event.certified_transverse())
            event.canonical_sign = orientation.canonical_sign;
        event.owners = crossing.owners;
        if (event.owners.empty())
            event.owners.push_back(event_owner_from_crossing(crossing));
        std::sort(
            event.owners.begin(), event.owners.end(),
            [](const NurbsSurfaceRootOwner3D& first,
               const NurbsSurfaceRootOwner3D& second) {
                if (first.patch_index != second.patch_index)
                    return first.patch_index < second.patch_index;
                if (first.u != second.u)
                    return first.u < second.u;
                return first.v < second.v;
            });
        result.push_back(std::move(event));
    }
    return result;
}

GridEdgeEventView3D::GridEdgeEventView3D(
    const GridEdgeEvent3D* event,
    int first_node,
    int second_node,
    bool reversed) noexcept
    : event_(event)
    , first_node_(first_node)
    , second_node_(second_node)
    , reversed_(reversed)
{}

const GridEdgeEvent3D& GridEdgeEventView3D::canonical_event() const
{
    if (event_ == nullptr)
        throw std::logic_error("empty grid-edge event view");
    return *event_;
}

const GridEdgeEventId3D& GridEdgeEventView3D::id() const
{
    return canonical_event().id;
}

int GridEdgeEventView3D::first_node() const noexcept
{
    return first_node_;
}

int GridEdgeEventView3D::second_node() const noexcept
{
    return second_node_;
}

double GridEdgeEventView3D::edge_parameter() const noexcept
{
    return event_ == nullptr
        ? 0.0
        : (reversed_ ? 1.0 - event_->canonical_parameter
                     : event_->canonical_parameter);
}

int GridEdgeEventView3D::sign() const noexcept
{
    return event_ == nullptr
        ? 0
        : (reversed_ ? -event_->canonical_sign
                     : event_->canonical_sign);
}

int GridEdgeEventView3D::component() const noexcept
{
    return event_ == nullptr ? -1 : event_->component;
}

GridEdgeEventCertification3D
GridEdgeEventView3D::certification() const noexcept
{
    return event_ == nullptr
        ? GridEdgeEventCertification3D::IncompleteRootSet
        : event_->certification;
}

bool GridEdgeEventView3D::certified_transverse() const noexcept
{
    return event_ != nullptr && event_->certified_transverse();
}

const std::vector<NurbsSurfaceRootOwner3D>&
GridEdgeEventView3D::owners() const
{
    return canonical_event().owners;
}

std::vector<GridEdgeEventView3D> grid_edge_event_views_3d(
    const std::vector<GridEdgeEvent3D>& canonical_events,
    int node_a,
    int node_b)
{
    if (node_a < 0 || node_b < 0 || node_a == node_b) {
        throw std::invalid_argument(
            "grid-edge event view requires two distinct nonnegative nodes");
    }
    const int first_node = std::min(node_a, node_b);
    const int second_node = std::max(node_a, node_b);
    for (const GridEdgeEvent3D& event : canonical_events) {
        if (event.id.first_node != first_node
            || event.id.second_node != second_node) {
            throw std::invalid_argument(
                "grid-edge event view received events from different edges");
        }
    }
    const bool reversed = node_a > node_b;
    std::vector<GridEdgeEventView3D> result;
    result.reserve(canonical_events.size());
    if (!reversed) {
        for (const GridEdgeEvent3D& event : canonical_events) {
            result.push_back(GridEdgeEventView3D(
                &event, node_a, node_b, false));
        }
    } else {
        for (auto event = canonical_events.rbegin();
             event != canonical_events.rend(); ++event) {
            result.push_back(GridEdgeEventView3D(
                &*event, node_a, node_b, true));
        }
    }
    return result;
}

struct NurbsCartesianDomain3D::Impl {
    DofLayout3D grid_layout = DofLayout3D::Node;
    std::array<int, 3> grid_cells{{0, 0, 0}};
    std::array<double, 3> grid_origin{{0.0, 0.0, 0.0}};
    std::array<double, 3> grid_spacing{{0.0, 0.0, 0.0}};
    std::array<int, 3> dims{{0, 0, 0}};
    int plane_stride = 0;
    int node_count = 0;
    std::array<std::vector<std::uint8_t>, 3> barriers;
    std::array<std::vector<std::uint8_t>, 3> interface_edges;
    std::vector<NurbsSurfaceCrossing3D> crossings;
    std::vector<GridEdgeEvent3D> grid_edge_events;
    std::unordered_map<std::uint64_t, EdgeIntersectionRecord3D>
        edge_records_by_edge;
    std::vector<int> node_labels;
    NurbsCartesianDomainDiagnostics3D diagnostics;
    NurbsAabb3D bounds;
    double tolerance = 0.0;
    std::vector<int> patch_g1_components;

    Impl(const CartesianGrid3D& grid,
         NurbsSurfaceModel3D model,
         NurbsCartesianDomainOptions3D options)
        : grid_layout(grid.layout())
        , grid_cells(grid.num_cells())
        , grid_origin(grid.origin())
        , grid_spacing(grid.spacing())
    {
        const PreprocessClock3D::time_point construction_begin =
            PreprocessClock3D::now();
        patch_g1_components = build_patch_g1_components(model);
        if (grid.layout() != DofLayout3D::Node) {
            throw std::invalid_argument(
                "NurbsCartesianDomain3D requires a node-layout Cartesian grid");
        }
        const auto spacing = grid.spacing();
        for (double h : spacing) {
            if (!std::isfinite(h) || h <= 0.0) {
                throw std::invalid_argument(
                    "NurbsCartesianDomain3D requires positive finite Cartesian spacing");
            }
        }
        if (std::isnan(options.maximum_element_extent_cap)
            || options.maximum_element_extent_cap <= 0.0) {
            throw std::invalid_argument(
                "NURBS maximum element extent cap must be positive");
        }
        const auto cells = grid.num_cells();
        for (int axis = 0; axis < 3; ++axis) {
            const int cell_count = cells[static_cast<std::size_t>(axis)];
            if (cell_count < 1) {
                throw std::invalid_argument(
                    "NurbsCartesianDomain3D requires at least one Cartesian cell per axis");
            }
            if (cell_count > std::numeric_limits<int>::max() - 1) {
                throw std::overflow_error(
                    "Cartesian cell-to-node dimension overflow");
            }
            dims[static_cast<std::size_t>(axis)] = cell_count + 1;
        }

        const std::size_t nx_size = static_cast<std::size_t>(dims[0]);
        const std::size_t ny_size = static_cast<std::size_t>(dims[1]);
        const std::size_t nz_size = static_cast<std::size_t>(dims[2]);
        const std::size_t plane_size = checked_size_product(
            nx_size, ny_size, "Cartesian node-plane storage overflow");
        if (plane_size > static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "Cartesian node count exceeds int range");
        }
        const std::size_t node_count_size = checked_size_product(
            plane_size, nz_size, "Cartesian node storage overflow");
        if (node_count_size > static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "Cartesian node count exceeds int range");
        }
        plane_stride = checked_size_to_int(
            plane_size, "Cartesian node stride exceeds int range");
        node_count = checked_size_to_int(
            node_count_size, "Cartesian node count exceeds int range");

        const Eigen::Vector3d box_lower = as_vector(grid.origin());
        Eigen::Vector3d box_upper = box_lower;
        for (int axis = 0; axis < 3; ++axis) {
            const double extent =
                spacing[static_cast<std::size_t>(axis)]
                * static_cast<double>(
                    cells[static_cast<std::size_t>(axis)]);
            box_upper[axis] += extent;
        }
        if (!box_lower.allFinite() || !box_upper.allFinite()) {
            throw std::overflow_error(
                "Cartesian box bounds must be finite");
        }

        const double maximum_spacing =
            *std::max_element(spacing.begin(), spacing.end());
        const double maximum_leaf_extent = std::min(
            2.0 * maximum_spacing,
            options.maximum_element_extent_cap);
        if (!std::isfinite(maximum_leaf_extent)) {
            throw std::overflow_error(
                "NURBS acceleration leaf extent must be finite");
        }

        NurbsSurfaceIntersectorOptions3D intersector_options;
        intersector_options.use_triangle_seeds = options.use_triangle_seeds;
        intersector_options.use_early_unique_root_certificate =
            options.strategy
            != NurbsCartesianPreprocessStrategy3D::CertifiedBaseline;
        intersector_options.use_affine_planar_fast_path =
            options.strategy
            == NurbsCartesianPreprocessStrategy3D::Hybrid;
        intersector_options.use_closest_point_prefilter =
            options.strategy
            == NurbsCartesianPreprocessStrategy3D::Hybrid;
        intersector_options.maximum_element_extent = maximum_leaf_extent;
        intersector_options.local_max_subdivision_depth = 4;
        // The Cartesian event catalog stores physical events, not per-patch
        // roots.  Declared non-G1 owners therefore have to be merged upstream
        // into one crossing with an owners list.
        intersector_options.preserve_non_g1_root_owners = false;
        const PreprocessClock3D::time_point intersector_begin =
            PreprocessClock3D::now();
        NurbsSurfaceIntersector3D intersector(
            std::move(model), intersector_options);
        diagnostics.intersector_build_seconds = elapsed_seconds(
            intersector_begin, PreprocessClock3D::now());
        bounds = intersector.bounds();
        tolerance = intersector.geometry_tolerance();
        if (!std::isfinite(tolerance) || tolerance < 0.0) {
            throw std::overflow_error(
                "NURBS geometry tolerance must be finite");
        }
        diagnostics.maximum_query_element_extent =
            intersector.maximum_query_element_extent();
        diagnostics.nurbs_patch_count = checked_size_to_int(
            intersector.model().patches().size(),
            "NURBS patch diagnostic count overflow");
        {
            const std::vector<RationalBezierElement3D> bezier_elements =
                extract_rational_bezier_elements_3d(intersector.model());
            diagnostics.bezier_element_count = checked_size_to_int(
                bezier_elements.size(),
                "NURBS Bezier-element diagnostic count overflow");
        }

        for (int axis = 0; axis < 3; ++axis) {
            if (!(bounds.lower[axis] > box_lower[axis]
                  && bounds.upper[axis] < box_upper[axis])) {
                throw std::runtime_error(
                    strict_box_error(bounds, box_lower, box_upper));
            }
        }

        const int nx = dims[0];
        const int ny = dims[1];
        const int nz = dims[2];
        const std::size_t x_barrier_size = checked_size_product(
            checked_size_product(
                static_cast<std::size_t>(nx - 1),
                static_cast<std::size_t>(ny),
                "Cartesian x-barrier row storage overflow"),
            static_cast<std::size_t>(nz),
            "Cartesian x-barrier storage overflow");
        const std::size_t y_barrier_size = checked_size_product(
            checked_size_product(
                static_cast<std::size_t>(nx),
                static_cast<std::size_t>(ny - 1),
                "Cartesian y-barrier row storage overflow"),
            static_cast<std::size_t>(nz),
            "Cartesian y-barrier storage overflow");
        const std::size_t z_barrier_size = checked_size_product(
            checked_size_product(
                static_cast<std::size_t>(nx),
                static_cast<std::size_t>(ny),
                "Cartesian z-barrier row storage overflow"),
            static_cast<std::size_t>(nz - 1),
            "Cartesian z-barrier storage overflow");
        barriers[0].assign(
            x_barrier_size, std::uint8_t{0});
        barriers[1].assign(
            y_barrier_size, std::uint8_t{0});
        barriers[2].assign(
            z_barrier_size, std::uint8_t{0});
        interface_edges[0].assign(
            x_barrier_size, std::uint8_t{0});
        interface_edges[1].assign(
            y_barrier_size, std::uint8_t{0});
        interface_edges[2].assign(
            z_barrier_size, std::uint8_t{0});

        const PreprocessClock3D::time_point candidate_begin =
            PreprocessClock3D::now();
        const auto& leaves = intersector.query_elements();
        if (leaves.size() != intersector.query_element_count()) {
            throw std::logic_error(
                "NURBS acceleration leaves differ from BVH query elements");
        }
        diagnostics.acceleration_leaf_count =
            checked_size_to_int(
                leaves.size(),
                "NURBS acceleration-leaf diagnostic count overflow");

        const auto origin = grid.origin();
        std::vector<std::uint64_t> candidate_keys;
        std::vector<std::pair<std::uint64_t, std::size_t>>
            candidate_incidences;
        const bool use_mapped_candidates =
            options.strategy
            != NurbsCartesianPreprocessStrategy3D::CertifiedBaseline;
        const auto record_candidate =
            [&](std::uint64_t key, std::size_t element_id) {
                if (use_mapped_candidates)
                    candidate_incidences.emplace_back(key, element_id);
                else
                    candidate_keys.push_back(key);
            };
        for (const NurbsQueryElementDescriptor3D& leaf : leaves) {
            const NurbsAabb3D& leaf_bounds = leaf.bounds;
            for (int axis = 0; axis < 3; ++axis) {
                const double h = spacing[static_cast<std::size_t>(axis)];
                const ClosedIntInterval start_range =
                    candidate_index_interval(
                        leaf_bounds.lower[axis],
                        leaf_bounds.upper[axis],
                        origin[static_cast<std::size_t>(axis)],
                        h, tolerance, true,
                        dims[static_cast<std::size_t>(axis)] - 2);
                if (start_range.empty())
                    continue;

                std::array<int, 3> transverse_min{{0, 0, 0}};
                std::array<int, 3> transverse_max{{0, 0, 0}};
                bool empty = false;
                for (int transverse = 0; transverse < 3; ++transverse) {
                    if (transverse == axis)
                        continue;
                    const ClosedIntInterval transverse_range =
                        candidate_index_interval(
                            leaf_bounds.lower[transverse],
                            leaf_bounds.upper[transverse],
                            origin[static_cast<std::size_t>(transverse)],
                            spacing[static_cast<std::size_t>(transverse)],
                            tolerance, false,
                            dims[static_cast<std::size_t>(transverse)] - 1);
                    transverse_min[static_cast<std::size_t>(transverse)] =
                        transverse_range.lower;
                    transverse_max[static_cast<std::size_t>(transverse)] =
                        transverse_range.upper;
                    empty = empty || transverse_range.empty();
                }
                if (empty)
                    continue;

                if (axis == 0) {
                    for (int k = transverse_min[2];
                         k <= transverse_max[2]; ++k) {
                        for (int j = transverse_min[1];
                             j <= transverse_max[1]; ++j) {
                            for (int i = start_range.lower;
                                 i <= start_range.upper; ++i) {
                                record_candidate(
                                    edge_key(
                                        axis,
                                        structured_node_index(i, j, k)),
                                    leaf.id);
                            }
                        }
                    }
                } else if (axis == 1) {
                    for (int k = transverse_min[2];
                         k <= transverse_max[2]; ++k) {
                        for (int j = start_range.lower;
                             j <= start_range.upper; ++j) {
                            for (int i = transverse_min[0];
                                 i <= transverse_max[0]; ++i) {
                                record_candidate(
                                    edge_key(
                                        axis,
                                        structured_node_index(i, j, k)),
                                    leaf.id);
                            }
                        }
                    }
                } else {
                    for (int k = start_range.lower;
                         k <= start_range.upper; ++k) {
                        for (int j = transverse_min[1];
                             j <= transverse_max[1]; ++j) {
                            for (int i = transverse_min[0];
                                 i <= transverse_max[0]; ++i) {
                                record_candidate(
                                    edge_key(
                                        axis,
                                        structured_node_index(i, j, k)),
                                    leaf.id);
                            }
                        }
                    }
                }
            }
        }
        if (use_mapped_candidates) {
            std::sort(
                candidate_incidences.begin(), candidate_incidences.end());
            candidate_incidences.erase(
                std::unique(
                    candidate_incidences.begin(), candidate_incidences.end()),
                candidate_incidences.end());
            diagnostics.candidate_element_incidence_count =
                candidate_incidences.size();
            candidate_keys.reserve(candidate_incidences.size());
            for (const auto& incidence : candidate_incidences) {
                if (candidate_keys.empty()
                    || candidate_keys.back() != incidence.first) {
                    candidate_keys.push_back(incidence.first);
                }
            }
        } else {
            std::sort(candidate_keys.begin(), candidate_keys.end());
            candidate_keys.erase(
                std::unique(candidate_keys.begin(), candidate_keys.end()),
                candidate_keys.end());
        }
        diagnostics.candidate_grid_edge_count = candidate_keys.size();
        diagnostics.candidate_enumeration_seconds = elapsed_seconds(
            candidate_begin, PreprocessClock3D::now());

        std::unique_ptr<NurbsSurfaceIntersector3D> retry_intersector;
        std::unique_ptr<NurbsSurfaceIntersector3D> final_retry_intersector;
        const auto deep_intersector = [&]()
            -> NurbsSurfaceIntersector3D& {
            if (!retry_intersector) {
                NurbsSurfaceIntersectorOptions3D retry_options =
                    intersector_options;
                retry_options.local_max_subdivision_depth = 6;
                retry_intersector =
                    std::make_unique<NurbsSurfaceIntersector3D>(
                        intersector.model(), retry_options);
            }
            return *retry_intersector;
        };
        const auto final_intersector = [&]()
            -> NurbsSurfaceIntersector3D& {
            if (!final_retry_intersector) {
                NurbsSurfaceIntersectorOptions3D retry_options =
                    intersector_options;
                retry_options.local_max_subdivision_depth = 10;
                final_retry_intersector =
                    std::make_unique<NurbsSurfaceIntersector3D>(
                        intersector.model(), retry_options);
            }
            return *final_retry_intersector;
        };
        const auto containing_components = [&](const Eigen::Vector3d& point) {
            try {
                return intersector.containing_components(point);
            } catch (const std::runtime_error& error) {
                if (std::string(error.what()).find(
                        "unable to classify point with deterministic NURBS rays")
                    == std::string::npos) {
                    throw;
                }
                try {
                    return deep_intersector().containing_components(point);
                } catch (const std::runtime_error& retry_error) {
                    if (std::string(retry_error.what()).find(
                            "unable to classify point with deterministic NURBS rays")
                        == std::string::npos) {
                        throw;
                    }
                    return final_intersector().containing_components(point);
                }
            }
        };
        const GridEdgeComponentContainment3D component_contains =
            [&](const Eigen::Vector3d& point, int component) {
                const std::vector<int> containing =
                    containing_components(point);
                return std::find(
                           containing.begin(), containing.end(), component)
                    != containing.end();
            };

        std::unordered_map<int, std::vector<int>> endpoint_membership_cache;
        const auto endpoint_membership = [&](int node) {
            const auto found = endpoint_membership_cache.find(node);
            if (found != endpoint_membership_cache.end())
                return found->second;
            const Eigen::Vector3d point = as_vector(grid.coord(node));
            std::vector<int> containing = containing_components(point);
            std::sort(containing.begin(), containing.end());
            endpoint_membership_cache.emplace(node, containing);
            diagnostics.endpoint_classification_query_count = checked_size_add(
                diagnostics.endpoint_classification_query_count,
                std::size_t{1},
                "NURBS endpoint-classification diagnostic overflow");
            return containing;
        };

        const auto targeted_retry = [&]
            (const NurbsCartesianEdgeQuery3D& query,
             const std::vector<std::size_t>& candidate_ids) {
            NurbsCartesianEdgeIntersections3D result;
            if (use_mapped_candidates) {
                result = intersector.intersect_cartesian_edge(
                    query, candidate_ids,
                    NurbsCartesianEdgeQueryOptions3D{
                        6,
                        NurbsCartesianEdgeQueryRoute3D::
                            OptimizedCertified});
            } else {
                result = deep_intersector().intersect_cartesian_edge(query);
            }
            // The mapped/optimized retry is only a first certified attempt.
            // If any part of the complete root set remains ambiguous, fall
            // back to the deeper full-BVH route rather than accepting a
            // unique/first-root approximation.
            if (grid_edge_root_set_requires_targeted_retry_3d(result)) {
                result = final_intersector().intersect_cartesian_edge(query);
            }
            return result;
        };

        std::size_t incidence_cursor = 0;
        for (const std::uint64_t key : candidate_keys) {
            std::vector<std::size_t> candidate_ids;
            if (use_mapped_candidates) {
                if (incidence_cursor >= candidate_incidences.size()
                    || candidate_incidences[incidence_cursor].first != key) {
                    throw std::logic_error(
                        "NURBS candidate incidence grouping is inconsistent");
                }
                while (incidence_cursor < candidate_incidences.size()
                       && candidate_incidences[incidence_cursor].first
                              == key) {
                    candidate_ids.push_back(
                        candidate_incidences[incidence_cursor].second);
                    ++incidence_cursor;
                }
            }
            const std::uint64_t axis_bits = key >> 62;
            if (axis_bits > 2)
                throw std::overflow_error("invalid Cartesian edge key axis");
            const int axis = static_cast<int>(axis_bits);
            const std::uint64_t start_node_bits = key & kNodeMask;
            if (start_node_bits > static_cast<std::uint64_t>(
                    std::numeric_limits<int>::max())) {
                throw std::overflow_error(
                    "Cartesian edge-key node exceeds int range");
            }
            const int start_node = static_cast<int>(start_node_bits);
            const auto ijk = node_coordinates(start_node);
            const int axis_stride =
                axis == 0 ? 1 : (axis == 1 ? nx : plane_stride);
            const int end_node = checked_add_int(
                start_node, axis_stride,
                "Cartesian edge endpoint index overflow");
            if (end_node >= node_count) {
                throw std::overflow_error(
                    "Cartesian edge endpoint is outside checked node storage");
            }
            const Eigen::Vector3d start = as_vector(grid.coord(start_node));
            const Eigen::Vector3d end = as_vector(grid.coord(end_node));
            const NurbsCartesianEdgeQuery3D query{
                axis, ijk[0], ijk[1], ijk[2], start, end};
            const PreprocessClock3D::time_point intersection_begin =
                PreprocessClock3D::now();
            NurbsCartesianEdgeIntersections3D edge_result =
                use_mapped_candidates
                ? intersector.intersect_cartesian_edge(query, candidate_ids)
                : intersector.intersect_cartesian_edge(query);
            diagnostics.edge_intersection_seconds += elapsed_seconds(
                intersection_begin, PreprocessClock3D::now());
            PreprocessClock3D::time_point materialization_begin =
                PreprocessClock3D::now();
            accumulate_intersection_diagnostics(
                diagnostics.intersections, edge_result.diagnostics);
            std::vector<int> toggled_components;
            bool changes_inside_outside = false;
            bool changes_component_membership = false;
            bool used_targeted_retry = false;
            std::vector<int> start_membership;
            std::vector<int> end_membership;
            bool endpoint_membership_loaded = false;
            const auto load_endpoint_membership = [&] {
                if (endpoint_membership_loaded)
                    return;
                start_membership = endpoint_membership(start_node);
                end_membership = endpoint_membership(end_node);
                endpoint_membership_loaded = true;
            };

            if (!edge_result.parity_known_from_roots) {
                diagnostics.ambiguous_parity_edge_count = checked_size_add(
                    diagnostics.ambiguous_parity_edge_count,
                    std::size_t{1},
                    "NURBS ambiguous-parity diagnostic overflow");
                load_endpoint_membership();
                if (start_membership != end_membership) {
                    diagnostics.ambiguous_label_changing_edge_count =
                        checked_size_add(
                            diagnostics.ambiguous_label_changing_edge_count,
                            std::size_t{1},
                            "NURBS ambiguous label-changing diagnostic overflow");
                }
            }
            // Endpoint parity is diagnostic information only.  Every
            // candidate edge with an incomplete/ambiguous root set is retried,
            // including same-label double crossings and unknown zero-root
            // results.
            if (grid_edge_root_set_requires_targeted_retry_3d(edge_result)) {
                diagnostics.targeted_retry_count = checked_size_add(
                    diagnostics.targeted_retry_count,
                    std::size_t{1},
                    "NURBS targeted-retry diagnostic overflow");
                diagnostics.edge_materialization_seconds += elapsed_seconds(
                    materialization_begin, PreprocessClock3D::now());
                const PreprocessClock3D::time_point retry_begin =
                    PreprocessClock3D::now();
                edge_result = targeted_retry(query, candidate_ids);
                diagnostics.edge_intersection_seconds += elapsed_seconds(
                    retry_begin, PreprocessClock3D::now());
                materialization_begin = PreprocessClock3D::now();
                used_targeted_retry = true;
                accumulate_intersection_diagnostics(
                    diagnostics.targeted_retry_intersections,
                    edge_result.diagnostics);
            }

            if (edge_result.parity_known_from_roots) {
                toggled_components = edge_result.toggled_components;
                changes_inside_outside =
                    edge_result.changes_inside_outside;
                changes_component_membership =
                    edge_result.changes_component_membership;
            } else {
                diagnostics.endpoint_parity_fallback_count = checked_size_add(
                    diagnostics.endpoint_parity_fallback_count,
                    std::size_t{1},
                    "NURBS endpoint-fallback diagnostic overflow");
                load_endpoint_membership();
                changes_inside_outside =
                    start_membership.empty() != end_membership.empty();
                changes_component_membership =
                    start_membership != end_membership;
                std::set_symmetric_difference(
                    start_membership.begin(), start_membership.end(),
                    end_membership.begin(), end_membership.end(),
                    std::back_inserter(toggled_components));
            }
            diagnostics.component_parity_toggle_count = checked_size_add(
                diagnostics.component_parity_toggle_count,
                toggled_components.size(),
                "NURBS component-parity diagnostic overflow");

            const std::size_t storage_index =
                barrier_index(axis, ijk[0], ijk[1], ijk[2]);
            EdgeIntersectionRecord3D record;
            record.range.begin = checked_size_to_int(
                crossings.size(),
                "NURBS crossing range begin exceeds int range");
            record.range.count = checked_size_to_int(
                edge_result.crossings.size(),
                "NURBS crossing range count exceeds int range");
            record.range.confirmed_transverse_count =
                edge_result.confirmed_transverse_count;
            record.range.changes_inside_outside = changes_inside_outside;
            record.range.changes_component_membership =
                changes_component_membership;
            record.classification.queried = true;
            record.classification.has_confirmed_interface =
                !edge_result.crossings.empty();
            record.classification.changes_component_membership =
                changes_component_membership;
            record.classification.root_count_known =
                edge_result.root_count_known;
            record.classification.parity_known_from_roots =
                edge_result.parity_known_from_roots;
            record.classification.has_near_tangent_candidate =
                edge_result.has_near_tangent_candidate;
            record.classification.used_targeted_retry =
                used_targeted_retry;
            record.classification.confirmed_crossing_count =
                edge_result.crossings.size();
            record.classification.ambiguous_cluster_count =
                edge_result.ambiguous_clusters.size();
            record.classification.confirmed_transverse_count =
                edge_result.confirmed_transverse_count;
            const std::vector<GridEdgeEvent3D> edge_events =
                build_canonical_grid_edge_events_3d(
                    start_node, end_node, start, end,
                    edge_result.crossings, record.classification,
                    component_contains);
            const bool root_set_certified =
                !grid_edge_root_set_requires_targeted_retry_3d(edge_result);
            const bool every_event_certified = std::all_of(
                edge_events.begin(), edge_events.end(),
                [](const GridEdgeEvent3D& event) {
                    return event.certified_transverse();
                });
            record.classification.physical_events_certified =
                root_set_certified
                && edge_events.size() == edge_result.crossings.size()
                && every_event_certified;
            if (!record.classification.physical_events_certified) {
                diagnostics.uncertified_grid_edge_count = checked_size_add(
                    diagnostics.uncertified_grid_edge_count,
                    std::size_t{1},
                    "NURBS uncertified grid-edge diagnostic overflow");
            }
            record.classification.correction_safe =
                changes_component_membership
                && record.classification.physical_events_certified
                && edge_result.crossings.size() == 1
                && edge_result.confirmed_transverse_count == 1;
            record.event_begin = checked_size_to_int(
                grid_edge_events.size(),
                "NURBS grid-edge event range begin exceeds int range");
            record.event_count = checked_size_to_int(
                edge_events.size(),
                "NURBS grid-edge event range count exceeds int range");
            for (const GridEdgeEvent3D& event : edge_events) {
                diagnostics.canonical_grid_edge_event_count =
                    checked_size_add(
                        diagnostics.canonical_grid_edge_event_count,
                        std::size_t{1},
                        "NURBS grid-edge event diagnostic overflow");
                if (event.certified_transverse()) {
                    diagnostics.certified_transverse_event_count =
                        checked_size_add(
                            diagnostics.certified_transverse_event_count,
                            std::size_t{1},
                            "NURBS certified-event diagnostic overflow");
                }
                grid_edge_events.push_back(event);
            }
            if (record.classification.correction_safe) {
                diagnostics.correction_safe_edge_count = checked_size_add(
                    diagnostics.correction_safe_edge_count,
                    std::size_t{1},
                    "NURBS correction-safe diagnostic overflow");
            } else if (changes_component_membership) {
                diagnostics.unsafe_label_changing_edge_count =
                    checked_size_add(
                        diagnostics.unsafe_label_changing_edge_count,
                        std::size_t{1},
                        "NURBS unsafe label-changing diagnostic overflow");
            }
            if (used_targeted_retry) {
                std::size_t& retry_result_count =
                    record.classification.physical_events_certified
                    ? diagnostics.targeted_retry_resolved_count
                    : diagnostics.targeted_retry_unsafe_count;
                retry_result_count = checked_size_add(
                    retry_result_count, std::size_t{1},
                    "NURBS targeted-retry result diagnostic overflow");
            }

            if (!edge_result.crossings.empty()) {
                interface_edges[static_cast<std::size_t>(axis)]
                               [storage_index] = std::uint8_t{1};
                std::size_t& interface_count =
                    diagnostics.interface_edge_counts[
                        static_cast<std::size_t>(axis)];
                interface_count = checked_size_add(
                    interface_count, std::size_t{1},
                    "NURBS interface diagnostic count overflow");
                if (edge_result.crossings.size() > 1) {
                    diagnostics.multi_crossing_edge_count = checked_size_add(
                        diagnostics.multi_crossing_edge_count,
                        std::size_t{1},
                        "NURBS multi-crossing diagnostic overflow");
                }
                if (edge_result.parity_known_from_roots) {
                    std::size_t& parity_count =
                        edge_result.confirmed_transverse_count % 2 == 0
                        ? diagnostics.even_parity_interface_edge_count
                        : diagnostics.odd_parity_interface_edge_count;
                    parity_count = checked_size_add(
                        parity_count, std::size_t{1},
                        "NURBS interface-parity diagnostic overflow");
                }
                for (const NurbsSurfaceCrossing3D& crossing :
                     edge_result.crossings) {
                    diagnostics.maximum_root_residual = std::max(
                        diagnostics.maximum_root_residual,
                        crossing.residual);
                    crossings.push_back(crossing);
                }
            }
            if (!edge_records_by_edge.emplace(key, record).second) {
                throw std::logic_error(
                    "duplicate NURBS Cartesian edge record");
            }
            if (changes_component_membership) {
                barriers[static_cast<std::size_t>(axis)][storage_index] =
                    std::uint8_t{1};
                std::size_t& barrier_count =
                    diagnostics.barrier_edge_counts[
                        static_cast<std::size_t>(axis)];
                barrier_count = checked_size_add(
                    barrier_count, std::size_t{1},
                    "NURBS barrier diagnostic count overflow");
            }
            diagnostics.edge_materialization_seconds += elapsed_seconds(
                materialization_begin, PreprocessClock3D::now());
        }

        if (use_mapped_candidates
            && incidence_cursor != candidate_incidences.size()) {
            throw std::logic_error(
                "NURBS candidate incidences were not fully consumed");
        }
        const PreprocessClock3D::time_point flood_begin =
            PreprocessClock3D::now();
        flood_and_label(grid, intersector);
        diagnostics.flood_labeling_seconds = std::max(
            0.0,
            elapsed_seconds(flood_begin, PreprocessClock3D::now())
                - diagnostics.representative_classification_seconds);
        const PreprocessClock3D::time_point verification_begin =
            PreprocessClock3D::now();
        verify_barriers();
        diagnostics.invariant_verification_seconds = elapsed_seconds(
            verification_begin, PreprocessClock3D::now());
        diagnostics.total_construction_seconds = elapsed_seconds(
            construction_begin, PreprocessClock3D::now());
    }

    bool is_compatible_grid(const CartesianGrid3D& grid) const noexcept
    {
        return grid.layout() == grid_layout
            && grid.num_cells() == grid_cells
            && grid.origin() == grid_origin
            && grid.spacing() == grid_spacing;
    }

    std::array<int, 3> node_coordinates(int node) const
    {
        const int nx = dims[0];
        const int k = node / plane_stride;
        const int remainder = node % plane_stride;
        return {{remainder % nx, remainder / nx, k}};
    }

    int structured_node_index(int i, int j, int k) const
    {
        const std::size_t plane_offset = checked_size_product(
            static_cast<std::size_t>(k),
            static_cast<std::size_t>(plane_stride),
            "Cartesian structured node index overflow");
        const std::size_t row_offset = checked_size_product(
            static_cast<std::size_t>(j),
            static_cast<std::size_t>(dims[0]),
            "Cartesian structured node index overflow");
        const std::size_t index = checked_size_add(
            checked_size_add(
                plane_offset, row_offset,
                "Cartesian structured node index overflow"),
            static_cast<std::size_t>(i),
            "Cartesian structured node index overflow");
        if (index >= static_cast<std::size_t>(node_count)) {
            throw std::overflow_error(
                "Cartesian structured node index is outside checked storage");
        }
        return checked_size_to_int(
            index, "Cartesian structured node index exceeds int range");
    }

    std::size_t barrier_index(int axis, int i, int j, int k) const
    {
        const std::size_t nx = static_cast<std::size_t>(dims[0]);
        const std::size_t ny = static_cast<std::size_t>(dims[1]);
        if (axis == 0) {
            return (static_cast<std::size_t>(k) * ny
                    + static_cast<std::size_t>(j))
                    * (nx - 1)
                + static_cast<std::size_t>(i);
        }
        if (axis == 1) {
            return (static_cast<std::size_t>(k) * (ny - 1)
                    + static_cast<std::size_t>(j))
                    * nx
                + static_cast<std::size_t>(i);
        }
        return (static_cast<std::size_t>(k) * ny
                + static_cast<std::size_t>(j))
                * nx
            + static_cast<std::size_t>(i);
    }

    bool barrier_at(int axis, int i, int j, int k) const
    {
        return barriers[static_cast<std::size_t>(axis)]
                    [barrier_index(axis, i, j, k)] != 0;
    }
    bool interface_at(int axis, int i, int j, int k) const
    {
        return interface_edges[static_cast<std::size_t>(axis)]
                              [barrier_index(axis, i, j, k)] != 0;
    }

    std::pair<int, int> adjacent_edge(int node_a, int node_b) const
    {
        if (node_a < 0 || node_a >= node_count
            || node_b < 0 || node_b >= node_count) {
            throw std::out_of_range(
                "NURBS Cartesian node index is outside the grid");
        }
        const auto first = node_coordinates(node_a);
        const auto second = node_coordinates(node_b);
        int axis = -1;
        int distance = 0;
        for (int candidate_axis = 0; candidate_axis < 3;
             ++candidate_axis) {
            const int difference = std::abs(
                first[static_cast<std::size_t>(candidate_axis)]
                - second[static_cast<std::size_t>(candidate_axis)]);
            distance += difference;
            if (difference != 0)
                axis = candidate_axis;
        }
        if (distance != 1) {
            throw std::invalid_argument(
                "NURBS Cartesian barrier query requires neighboring nodes");
        }
        return {axis, std::min(node_a, node_b)};
    }

    bool has_barrier(int node_a, int node_b) const
    {
        const auto edge = adjacent_edge(node_a, node_b);
        const auto ijk = node_coordinates(edge.second);
        return barrier_at(edge.first, ijk[0], ijk[1], ijk[2]);
    }
    bool has_interface(int node_a, int node_b) const
    {
        const auto edge = adjacent_edge(node_a, node_b);
        const auto ijk = node_coordinates(edge.second);
        return interface_at(edge.first, ijk[0], ijk[1], ijk[2]);
    }

    void flood_and_label(const CartesianGrid3D& grid,
                         const NurbsSurfaceIntersector3D& intersector)
    {
        const int nx = dims[0];
        const int ny = dims[1];
        const int nz = dims[2];
        std::vector<int> graph_component(
            static_cast<std::size_t>(node_count), -1);
        std::vector<int> representatives;
        std::vector<bool> touches_box_boundary;
        std::vector<int> queue;

        for (int seed = 0; seed < node_count; ++seed) {
            if (graph_component[static_cast<std::size_t>(seed)] >= 0)
                continue;
            const int component = checked_size_to_int(
                representatives.size(),
                "Cartesian grid-component index exceeds int range");
            representatives.push_back(seed);
            touches_box_boundary.push_back(false);
            diagnostics.component_sizes.push_back(0);
            queue.clear();
            queue.push_back(seed);
            graph_component[static_cast<std::size_t>(seed)] = component;

            for (std::size_t head = 0; head < queue.size(); ++head) {
                const int node = queue[head];
                std::size_t& component_size =
                    diagnostics.component_sizes[
                        static_cast<std::size_t>(component)];
                component_size = checked_size_add(
                    component_size, std::size_t{1},
                    "Cartesian component-size diagnostic overflow");
                const auto ijk = node_coordinates(node);
                const int i = ijk[0];
                const int j = ijk[1];
                const int k = ijk[2];
                if (i == 0 || i == nx - 1
                    || j == 0 || j == ny - 1
                    || k == 0 || k == nz - 1) {
                    touches_box_boundary[
                        static_cast<std::size_t>(component)] = true;
                }

                const std::array<int, 6> neighbors{{
                    i > 0 && !barrier_at(0, i - 1, j, k) ? node - 1 : -1,
                    i + 1 < nx && !barrier_at(0, i, j, k) ? node + 1 : -1,
                    j > 0 && !barrier_at(1, i, j - 1, k) ? node - nx : -1,
                    j + 1 < ny && !barrier_at(1, i, j, k) ? node + nx : -1,
                    k > 0 && !barrier_at(2, i, j, k - 1)
                        ? node - plane_stride : -1,
                    k + 1 < nz && !barrier_at(2, i, j, k)
                        ? node + plane_stride : -1}};
                for (const int neighbor : neighbors) {
                    if (neighbor >= 0
                        && graph_component[
                            static_cast<std::size_t>(neighbor)] < 0) {
                        graph_component[
                            static_cast<std::size_t>(neighbor)] = component;
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        diagnostics.grid_component_count = checked_size_to_int(
            representatives.size(),
            "Cartesian grid-component diagnostic count overflow");
        std::vector<int> component_labels(representatives.size(), 0);
        for (std::size_t component = 0;
             component < representatives.size(); ++component) {
            if (touches_box_boundary[component]) {
                component_labels[component] = 0;
                diagnostics.box_exterior_component_count = checked_add_int(
                    diagnostics.box_exterior_component_count, 1,
                    "Cartesian box-exterior diagnostic count overflow");
            } else {
                diagnostics.representative_query_count = checked_add_int(
                    diagnostics.representative_query_count, 1,
                    "Cartesian representative-query diagnostic count overflow");
                const Eigen::Vector3d representative_point = as_vector(
                    grid.coord(representatives[component]));
                const PreprocessClock3D::time_point classification_begin =
                    PreprocessClock3D::now();
                const std::vector<int> containing =
                    intersector.containing_components(
                        representative_point);
                diagnostics.representative_classification_seconds +=
                    elapsed_seconds(
                        classification_begin, PreprocessClock3D::now());
                if (containing.empty()) {
                    component_labels[component] = 0;
                } else if (containing.size() == 1) {
                    component_labels[component] =
                        checked_add_int(
                            containing.front(), 1,
                            "NURBS component label exceeds int range");
                } else {
                    throw std::runtime_error(
                        "overlapping NURBS solid components");
                }
            }
        }

        node_labels.resize(static_cast<std::size_t>(node_count));
        for (int node = 0; node < node_count; ++node) {
            node_labels[static_cast<std::size_t>(node)] =
                component_labels[static_cast<std::size_t>(
                    graph_component[static_cast<std::size_t>(node)])];
        }
    }

    void verify_edge(int axis, int i, int j, int k,
                     int first, int second) const
    {
        const bool barrier = barrier_at(axis, i, j, k);
        const bool interface_edge = interface_at(axis, i, j, k);
        const int first_label =
            node_labels[static_cast<std::size_t>(first)];
        const int second_label =
            node_labels[static_cast<std::size_t>(second)];
        const bool component_label_change = first_label != second_label;
        const bool binary_change = (first_label > 0) != (second_label > 0);
        if (component_label_change != barrier) {
            throw std::runtime_error(
                "NURBS component label change disagrees with Cartesian barrier");
        }

        const std::uint64_t key = edge_key(axis, first);
        const auto found = edge_records_by_edge.find(key);
        const bool has_record = found != edge_records_by_edge.end();
        const bool has_range =
            has_record && found->second.range.count > 0;
        if (interface_edge != has_range) {
            throw std::runtime_error(
                "NURBS interface edge disagrees with crossing range");
        }
        if (!has_record)
            return;

        const EdgeIntersectionRecord3D& record = found->second;
        const EdgeCrossingRange3D& range = record.range;
        if (!record.classification.queried
            || record.classification.changes_component_membership != barrier
            || range.changes_component_membership != barrier
            || range.changes_inside_outside != binary_change) {
            throw std::runtime_error(
                "NURBS crossing parity disagrees with endpoint labels");
        }
        if (range.begin < 0 || range.count < 0
            || static_cast<std::size_t>(range.begin)
                   > crossings.size()
            || static_cast<std::size_t>(range.count)
                   > crossings.size()
                       - static_cast<std::size_t>(range.begin)) {
            throw std::runtime_error(
                "NURBS Cartesian crossing range is invalid");
        }
    }
    void verify_barriers() const
    {
        const int nx = dims[0];
        const int ny = dims[1];
        const int nz = dims[2];
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int node = structured_node_index(i, j, k);
                    if (i + 1 < nx)
                        verify_edge(0, i, j, k, node, node + 1);
                    if (j + 1 < ny)
                        verify_edge(1, i, j, k, node, node + nx);
                    if (k + 1 < nz)
                        verify_edge(
                            2, i, j, k, node, node + plane_stride);
                }
            }
        }
    }
};

NurbsSurfaceCrossingRange3D::NurbsSurfaceCrossingRange3D(
    const NurbsSurfaceCrossing3D* data, std::size_t size) noexcept
    : data_(data)
    , size_(size)
{}

const NurbsSurfaceCrossing3D*
NurbsSurfaceCrossingRange3D::begin() const noexcept
{
    return data_;
}

const NurbsSurfaceCrossing3D*
NurbsSurfaceCrossingRange3D::end() const noexcept
{
    return data_ == nullptr ? nullptr : data_ + size_;
}

std::size_t NurbsSurfaceCrossingRange3D::size() const noexcept
{
    return size_;
}

const NurbsSurfaceCrossing3D&
NurbsSurfaceCrossingRange3D::operator[](std::size_t index) const
{
    if (index >= size_)
        throw std::out_of_range("NURBS crossing range index is out of bounds");
    return data_[index];
}

NurbsCartesianDomain3D::NurbsCartesianDomain3D(
    const CartesianGrid3D& grid,
    NurbsSurfaceModel3D model,
    NurbsCartesianDomainOptions3D options)
    : impl_(std::make_shared<Impl>(
          grid, std::move(model), options))
{}

int NurbsCartesianDomain3D::label(int node) const
{
    if (node < 0
        || node >= impl_->node_count) {
        throw std::out_of_range(
            "NURBS Cartesian node index is outside the grid");
    }
    return impl_->node_labels[static_cast<std::size_t>(node)];
}

bool NurbsCartesianDomain3D::has_barrier_between(
    int node_a, int node_b) const
{
    return impl_->has_barrier(node_a, node_b);
}

bool NurbsCartesianDomain3D::has_interface_between(
    int node_a, int node_b) const
{
    return impl_->has_interface(node_a, node_b);
}

NurbsSurfaceCrossingRange3D NurbsCartesianDomain3D::crossings_between(
    int node_a, int node_b) const
{
    const auto edge = impl_->adjacent_edge(node_a, node_b);
    const auto found = impl_->edge_records_by_edge.find(
        edge_key(edge.first, edge.second));
    if (found == impl_->edge_records_by_edge.end()
        || found->second.range.count == 0) {
        return {};
    }
    const EdgeCrossingRange3D& range = found->second.range;
    return NurbsSurfaceCrossingRange3D(
        impl_->crossings.data() + range.begin,
        static_cast<std::size_t>(range.count));
}

std::vector<GridEdgeEventView3D>
NurbsCartesianDomain3D::grid_edge_events_between(
    int node_a, int node_b) const
{
    const auto edge = impl_->adjacent_edge(node_a, node_b);
    const auto found = impl_->edge_records_by_edge.find(
        edge_key(edge.first, edge.second));
    if (found == impl_->edge_records_by_edge.end()
        || found->second.event_count == 0) {
        return {};
    }
    const EdgeIntersectionRecord3D& record = found->second;
    const bool reversed = node_a > node_b;
    std::vector<GridEdgeEventView3D> result;
    result.reserve(static_cast<std::size_t>(record.event_count));
    for (int query_ordinal = 0;
         query_ordinal < record.event_count; ++query_ordinal) {
        const int canonical_ordinal = reversed
            ? record.event_count - 1 - query_ordinal
            : query_ordinal;
        const std::size_t catalog_index = static_cast<std::size_t>(
            record.event_begin + canonical_ordinal);
        if (catalog_index >= impl_->grid_edge_events.size()) {
            throw std::logic_error(
                "NURBS grid-edge event record is outside the catalog");
        }
        result.push_back(GridEdgeEventView3D(
            &impl_->grid_edge_events[catalog_index],
            node_a, node_b, reversed));
    }
    return result;
}

const std::vector<GridEdgeEvent3D>&
NurbsCartesianDomain3D::grid_edge_event_catalog() const
{
    return impl_->grid_edge_events;
}

const NurbsSurfaceCrossing3D& NurbsCartesianDomain3D::crossing_between(
    int node_a, int node_b) const
{
    const NurbsSurfaceCrossingRange3D range =
        crossings_between(node_a, node_b);
    if (range.size() == 0) {
        throw std::runtime_error(
            "no NURBS crossing between Cartesian nodes");
    }
    if (range.size() != 1) {
        throw std::runtime_error(
            "NURBS crossing lookup requires exactly one crossing");
    }
    return range[0];
}
NurbsCartesianEdgeClassification3D
NurbsCartesianDomain3D::edge_classification_between(
    int node_a, int node_b) const
{
    const auto edge = impl_->adjacent_edge(node_a, node_b);
    const auto found = impl_->edge_records_by_edge.find(
        edge_key(edge.first, edge.second));
    if (found == impl_->edge_records_by_edge.end())
        return {};
    return found->second.classification;
}

const NurbsSurfaceCrossing3D&
NurbsCartesianDomain3D::correction_crossing_between(
    int node_a, int node_b) const
{
    const auto edge = impl_->adjacent_edge(node_a, node_b);
    const auto found = impl_->edge_records_by_edge.find(
        edge_key(edge.first, edge.second));
    if (found != impl_->edge_records_by_edge.end()
        && found->second.classification.correction_safe) {
        const EdgeCrossingRange3D& range = found->second.range;
        if (range.count != 1 || range.begin < 0
            || static_cast<std::size_t>(range.begin)
                   >= impl_->crossings.size()) {
            throw std::logic_error(
                "correction-safe NURBS edge has an invalid crossing range");
        }
        return impl_->crossings[static_cast<std::size_t>(range.begin)];
    }

    const auto start = impl_->node_coordinates(edge.second);
    auto end = start;
    ++end[static_cast<std::size_t>(edge.first)];
    std::ostringstream message;
    message << "under-resolved NURBS Cartesian edge"
            << ": axis=" << edge.first
            << ", start_node=" << edge.second
            << ", start=(" << start[0] << ',' << start[1] << ','
            << start[2] << ")"
            << ", end=(" << end[0] << ',' << end[1] << ','
            << end[2] << ")";
    if (found == impl_->edge_records_by_edge.end()) {
        message << ", queried=0";
    } else {
        const EdgeIntersectionRecord3D& record = found->second;
        const auto& info = record.classification;
        message << ", queried=" << info.queried
                << ", changes_component_membership="
                << info.changes_component_membership
                << ", root_count_known=" << info.root_count_known
                << ", parity_known_from_roots="
                << info.parity_known_from_roots
                << ", near_tangent="
                << info.has_near_tangent_candidate
                << ", confirmed_crossings="
                << info.confirmed_crossing_count
                << ", ambiguous_clusters="
                << info.ambiguous_cluster_count
                << ", confirmed_transverse="
                << info.confirmed_transverse_count;
        const EdgeCrossingRange3D& range = record.range;
        for (int root_index = 0; root_index < range.count; ++root_index) {
            const NurbsSurfaceCrossing3D& root =
                impl_->crossings[static_cast<std::size_t>(
                    range.begin + root_index)];
            message << " root=(patch=" << root.patch_index
                    << ",component=" << root.component
                    << ",u=" << root.u << ",v=" << root.v
                    << ",t=" << root.edge_parameter
                    << ",residual=" << root.residual
                    << ",transversality=" << root.transversality << ')';
        }
    }
    throw std::runtime_error(message.str());
}
const std::vector<int>& NurbsCartesianDomain3D::labels() const
{
    return impl_->node_labels;
}

const NurbsCartesianDomainDiagnostics3D&
NurbsCartesianDomain3D::diagnostics() const
{
    return impl_->diagnostics;
}

const NurbsAabb3D& NurbsCartesianDomain3D::surface_bounds() const
{
    return impl_->bounds;
}

double NurbsCartesianDomain3D::geometry_tolerance() const
{
    return impl_->tolerance;
}

const std::vector<int>&
NurbsCartesianDomain3D::patch_g1_components() const
{
    return impl_->patch_g1_components;
}

bool NurbsCartesianDomain3D::is_compatible_grid(
    const CartesianGrid3D& grid) const noexcept
{
    return impl_->is_compatible_grid(grid);
}

} // namespace kfbim::geometry3d
