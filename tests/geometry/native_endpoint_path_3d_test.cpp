#include "src/support/geometry/native_nurbs_surface_3d.hpp"
#include "src/support/geometry/native_nurbs_surface_transform_3d.hpp"
#include "src/support/trace/native_endpoint_path_adapter_3d.hpp"
#include "src/support/trace/restrict_crossing_selector_3d.hpp"

#include "src/geometry/native_nurbs_exact_geometry_3d.hpp"
#include "src/geometry/native_endpoint_path_3d.hpp"
#include "src/geometry/nurbs_surface_intersector_3d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::NativeNurbsSurface3D;
using kfbim::app3d::RigidTransform3D;
using kfbim::app3d::make_native_nurbs_surface_3d;
using kfbim::app3d::transform_native_nurbs_surface_3d;
using kfbim::geometry3d::NurbsSurfaceModel3D;
using kfbim::geometry3d::NurbsSurfacePatch3D;
using kfbim::geometry3d::NativeEndpointPathResult3D;
using kfbim::geometry3d::NativeEndpointQueryOptions3D;
using kfbim::geometry3d::NativeSurfaceEndpoint3D;
using kfbim::geometry3d::PathCertificationBudget3D;
using kfbim::geometry3d::PathStatus3D;
using kfbim::geometry3d::RootProofKind3D;
using kfbim::geometry3d::certify_native_endpoint_path_3d;
using kfbim::geometry3d::extract_exact_native_bezier_elements_3d;
using kfbim::app3d::select_native_endpoint_path_events_3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template<class Action>
void require_rejected(Action&& action, const std::string& message)
{
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected, message);
}

double from_bits(std::uint64_t bits)
{
    static_assert(sizeof(double) == sizeof(bits), "binary64 fixture required");
    static_assert(std::numeric_limits<double>::is_iec559,
                  "IEEE-754 fixtures require IEC 559 double");
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

struct CylinderPathFixture {
    const char* name;
    int n;
    int center;
    int grid_node;
    int patch;
    std::uint64_t u_bits;
    std::uint64_t v_bits;
    Eigen::Vector3d start;
    Eigen::Vector3d translation;
    Eigen::Vector3d approximate_endpoint;
    int expected_open_roots;
    int expected_endpoint_sign;
    double approximate_open_t;

    double u() const { return from_bits(u_bits); }
    double v() const { return from_bits(v_bits); }
};

// These are the original surface-sampling parameter bits, not rounded ideal
// fractions or a line through the printed endpoint coordinate.  In particular,
// the D32 u differs by one ULP from the ordinary C++ literal 0.15.
// The approximate coordinates/t below are sanity checks only.  Production
// queries must define q by the current native patch and these parameters.
const std::vector<CylinderPathFixture>& cylinder_fixtures()
{
    static const std::vector<CylinderPathFixture> fixtures = {
        {"cylinder_n32", 32, 507, 13519, 3,
         UINT64_C(0x3FE4CCCCCCCCCCCD), UINT64_C(0x3FD0000000000000),
         {0.5625, -0.28125, -0.375}, {0.0, 0.0, 0.0},
         {0.5322656175648182, -0.3318957013970257, -0.305},
         0, +1, 0.0},
        {"cylinder_d32_tx", 32, 17, 12531, 0,
         UINT64_C(0x3FC3333333333334), UINT64_C(0x3FD0000000000000),
         {0.75, 0.0, -0.46875}, {0.137, 0.0, 0.0},
         {0.7336261359036995, 0.0705503640187959, -0.305},
         0, +1, 0.0},
        {"cylinder_d64_tx", 64, 29, 95013, 0,
         UINT64_C(0x3FB435E50D79435E), UINT64_C(0x3FAB6DB6DB6DB6DB),
         {0.75, -0.046875, -0.46875}, {0.137, 0.0, 0.0},
         {0.7434195102165537, 0.0126555572531406, -0.5603571428571429},
         1, -1, 0.925171384642}
    };
    return fixtures;
}

NativeNurbsSurface3D fixture_surface(const CylinderPathFixture& fixture)
{
    auto surface = make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    if (fixture.translation.squaredNorm() == 0.0)
        return surface;
    return transform_native_nurbs_surface_3d(
        surface, RigidTransform3D(Eigen::Matrix3d::Identity(),
                                  Eigen::Vector3d::Zero(),
                                  fixture.translation));
}

NurbsSurfacePatch3D make_plane(double z, bool reverse_normal = false)
{
    using kfbim::geometry::NurbsBasis1D;
    const double y0 = reverse_normal ? 1.0 : 0.0;
    const double y1 = reverse_normal ? 0.0 : 1.0;
    return NurbsSurfacePatch3D(
        NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0}),
        NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0}),
        {{{0.0, y0, z}, {0.0, y1, z}},
         {{1.0, y0, z}, {1.0, y1, z}}},
        {{1.0, 1.0}, {1.0, 1.0}});
}

void check_fixture(const CylinderPathFixture& fixture)
{
    const auto surface = fixture_surface(fixture);
    require(surface.patches.size() == 16,
            "fixture must use the current closed 16-patch hollow cylinder");
    require(surface.patch_names.at(fixture.patch)
                == "outer_wall_q" + std::to_string(fixture.patch),
            "fixture endpoint must remain on the intended outer wall patch");
    const auto& patch = surface.patches.at(fixture.patch);
    const auto q = patch.evaluate(fixture.u(), fixture.v());
    const auto n = patch.normal(fixture.u(), fixture.v());
    require((q - fixture.approximate_endpoint).norm() < 2.0e-14,
            "native fixture/model no longer matches the saved failure path");
    require(fixture.u() > patch.domain_start_u()
                && fixture.u() < patch.domain_end_u()
                && fixture.v() > patch.domain_start_v()
                && fixture.v() < patch.domain_end_v(),
            "failure fixture is a smooth patch-interior endpoint");
    require(!surface.exact_inside(fixture.start),
            "all three saved support nodes must be outside");
    const double b = n.dot(q - fixture.start);
    require(fixture.expected_endpoint_sign * b < 0.0,
            "independent outward-normal sign sanity check");

    const int width = fixture.n + 1;
    const int i = fixture.grid_node % width;
    const int j = (fixture.grid_node / width) % width;
    const int k = fixture.grid_node / (width * width);
    const double h = 3.0 / fixture.n;
    const Eigen::Vector3d grid_point(-1.5 + i * h,
                                   -1.5 + j * h,
                                   -1.5 + k * h);
    require(grid_point == fixture.start, "saved Cartesian node indexing");
    std::cout << " fixture=" << fixture.name
              << " patch=" << fixture.patch
              << " u=" << std::setprecision(17) << fixture.u()
              << " v=" << fixture.v()
              << " endpoint_dot=" << b << '\n';
}

void test_fixture_metadata()
{
    for (const auto& fixture : cylinder_fixtures())
        check_fixture(fixture);
    require(cylinder_fixtures().at(1).u() != 0.15,
            "do not replace the D32 original parameter with decimal 0.15");
}

PathCertificationBudget3D test_budget()
{
    PathCertificationBudget3D budget;
    budget.max_candidate_regions = 1024;
    budget.max_subdivision_nodes = 8192;
    budget.max_newton_steps = 8192;
    budget.max_relation_refinements = 1024;
    budget.max_closest_point_evaluations = 0;
    return budget;
}

struct QueryContext {
    NurbsSurfaceModel3D model;
    std::vector<kfbim::geometry3d::ExactNativeBezierElement3D> exact;

    explicit QueryContext(NurbsSurfaceModel3D source)
        : model(std::move(source))
        , exact(extract_exact_native_bezier_elements_3d(model))
    {}
};

NativeEndpointPathResult3D query(
    const QueryContext& context, const Eigen::Vector3d& start,
    NativeSurfaceEndpoint3D endpoint, bool start_inside,
    const PathCertificationBudget3D& budget = test_budget())
{
    NativeEndpointQueryOptions3D options;
    options.expected_start_inside = start_inside;
    auto result = certify_native_endpoint_path_3d(
        context.model, context.exact, start, endpoint, options, budget);
    std::cout << " status="
              << kfbim::geometry3d::native_endpoint_path_status_name_3d(result.status)
              << " open=" << result.open_roots.size()
              << " endpoint=" << result.endpoint.has_value()
              << " visited=" << result.coverage.visited_regions
              << " unresolved=" << result.coverage.unresolved_regions
              << " newton=" << result.diagnostics.newton_steps
              << " precision_bits=" << result.diagnostics.highest_precision_bits
              << " reason=" << result.diagnostics.reason << '\n';
    if (result.status != PathStatus3D::Certified)
        std::cout << " message=" << result.message
                  << "\n replay=" << result.dump << '\n';
    return result;
}

void require_root_proof(
    const kfbim::geometry3d::CertifiedNativePathRoot3D& root,
    bool endpoint)
{
    const auto& proof = root.certificate;
    require(proof.kind == (endpoint ? RootProofKind3D::KnownEndpointUnique
                                   : RootProofKind3D::ExistsUnique),
            "application roots must not be numerical candidates");
    require(proof.existence_proved && proof.uniqueness_proved,
            "existence and uniqueness must both be proved");
    require(proof.proof_id != 0 && proof.event_id != 0,
            "application certificates and physical events have explicit identities");
    require(std::isfinite(proof.contraction_bound)
                && proof.contraction_bound >= 0.0
                && proof.contraction_bound < 1.0,
            "root must carry a valid contraction bound");
    require(std::isfinite(proof.t_interval.lower)
                && std::isfinite(proof.t_interval.upper)
                && proof.t_interval.lower <= proof.t_interval.upper,
            "finite ordered root parameter enclosure");
    if (endpoint) {
        require(proof.t_interval.lower == 1.0
                    && proof.t_interval.upper == 1.0,
                "known native endpoint is exactly t=1");
    } else {
        require(0.0 < proof.t_interval.lower
                    && proof.t_interval.upper < 1.0,
                "ordinary root enclosure must be strictly inside (0,1)");
    }
    require(root.transition.certified && root.accuracy.certified,
            "geometry transition and representative accuracy are separate gates");
    require(root.transition.incoming_inside >= 0
                && root.transition.incoming_inside <= 1
                && root.transition.outgoing_inside >= 0
                && root.transition.outgoing_inside <= 1
                && root.transition.outgoing_inside
                       - root.transition.incoming_inside == root.transition.sign,
            "certified event sign must agree with its physical states");
    if (root.transition.sign == +1)
        require(root.transition.oriented_dot_upper < 0.0,
                "entering event requires a strictly negative dot enclosure");
    else if (root.transition.sign == -1)
        require(root.transition.oriented_dot_lower > 0.0,
                "leaving event requires a strictly positive dot enclosure");
    else
        require(false, "these fixtures contain only transverse events");
    const NativeEndpointQueryOptions3D accuracy;
    require(std::isfinite(root.accuracy.point_error_bound)
                && root.accuracy.point_error_bound >= 0.0
                && root.accuracy.point_error_bound <= accuracy.point_tolerance
                && std::isfinite(root.accuracy.parameter_error_bound)
                && root.accuracy.parameter_error_bound >= 0.0
                && root.accuracy.parameter_error_bound <= accuracy.parameter_tolerance
                && std::isfinite(root.accuracy.normal_error_bound)
                && root.accuracy.normal_error_bound >= 0.0
                && root.accuracy.normal_error_bound <= accuracy.normal_tolerance,
            "representative error bounds must meet the requested tolerances");
}

void require_certified(const NativeEndpointPathResult3D& result,
                       std::size_t expected_open, int endpoint_sign,
                       bool start_inside)
{
    require(result.status == PathStatus3D::Certified,
            "native path certification failed: " + result.message);
    require(result.coverage.complete && result.coverage.order_certified
                && result.coverage.transitions_certified
                && result.coverage.representatives_certified
                && result.coverage.unresolved_regions == 0,
            "Certified must require all four application gates");
    require(result.coverage.certified_start_inside == int(start_inside),
            "certified path start side");
    require(result.open_roots.size() == expected_open,
            "complete path must retain exactly the expected open roots");
    require(result.endpoint.has_value(), "complete native endpoint proof");
    require(result.post_endpoint_roots.empty(), "closed query has no post roots");
    double previous_upper = 0.0;
    int state = int(start_inside);
    for (const auto& root : result.open_roots) {
        require_root_proof(root, false);
        require(previous_upper < root.certificate.t_interval.lower,
                "independent physical events must have disjoint ordered t boxes");
        previous_upper = root.certificate.t_interval.upper;
        require(root.transition.incoming_inside == state,
                "open event incoming state follows the previous event");
        state += root.transition.sign;
    }
    require_root_proof(*result.endpoint, true);
    require(result.endpoint->transition.sign == endpoint_sign
                && result.endpoint->transition.incoming_inside == state,
            "endpoint physical direction agrees with open-event continuation");

    for (const bool desired : {false, true}) {
        const auto adapted = select_native_endpoint_path_events_3d(
            result, start_inside, desired);
        const bool add_endpoint = int(desired) != state;
        const std::size_t expected = expected_open + std::size_t(add_endpoint);
        require(adapted.sequence.events.size() == expected,
                "production adapter adds endpoint only for the opposite trace side");
        require(adapted.sequence.physical_event_count == int(expected)
                    && adapted.sequence.input_crossing_count
                           == int(adapted.intersection.crossings.size()),
                "production adapter event metadata must remain consistent");
        int adapted_state = int(start_inside);
        for (std::size_t index = 0; index < adapted.sequence.events.size(); ++index) {
            const auto& event = adapted.sequence.events[index];
            const auto& source = index < result.open_roots.size()
                ? result.open_roots[index] : *result.endpoint;
            require(event.native_event_id == source.certificate.event_id
                        && event.native_proof_id == source.certificate.proof_id
                        && event.native_event_id != 0 && event.native_proof_id != 0,
                    "production adapter must preserve each selected root's proof/event identity");
            adapted_state += event.continuation_sign;
        }
        require(adapted_state == int(desired), "both trace branches reach their target");
        if (add_endpoint) {
            require(adapted.sequence.events.back().edge_parameter == 1.0
                        && adapted.sequence.events.back().continuation_sign
                               == endpoint_sign,
                    "conditional endpoint must use its geometric sign");
        }
    }
}

void test_cylinder_path(const CylinderPathFixture& fixture)
{
    check_fixture(fixture);
    const auto surface = fixture_surface(fixture);
    const QueryContext context(surface.geometry_model());
    const auto result = query(context, fixture.start,
                              {fixture.patch, fixture.u(), fixture.v()}, false);
    require_certified(result, std::size_t(fixture.expected_open_roots),
                      fixture.expected_endpoint_sign, false);
    if (fixture.expected_open_roots != 0) {
        require(result.open_roots.front().transition.sign == +1,
                "D64 open event enters the domain before the endpoint leaves");
        // Analytic ideal-circle reference checks location only, not proof.
        require(std::abs(result.open_roots.front().representative.edge_parameter
                             - fixture.approximate_open_t) < 2.0e-8,
                "D64 must retain the genuine open root near t=0.92517");
    }
}

QueryContext plane_context()
{
    return QueryContext(NurbsSurfaceModel3D({make_plane(1.0)}, {0}, {}));
}

QueryContext three_plane_context()
{
    // Along +z: enter at 1/4, leave at 3/4, enter at the known endpoint.
    return QueryContext(NurbsSurfaceModel3D(
        {make_plane(0.25, true), make_plane(0.75), make_plane(1.0, true)},
        {0, 0, 0}, {}));
}

void test_plane_endpoint()
{
    const auto context = plane_context();
    require_certified(query(context, {0.5, 0.5, 0.0}, {0, 0.5, 0.5}, true),
                      0, -1, true);
}

void test_two_ordinary_roots()
{
    const auto context = three_plane_context();
    const auto result = query(context, {0.5, 0.5, 0.0}, {2, 0.5, 0.5}, false);
    require_certified(result, 2, +1, false);
    require(std::abs(result.open_roots[0].representative.edge_parameter - 0.25)
                < 1.0e-12
                && std::abs(result.open_roots[1].representative.edge_parameter - 0.75)
                       < 1.0e-12,
            "both independent ordinary plane roots are preserved");
}

void test_split_boundary_endpoint()
{
    using kfbim::geometry::NurbsBasis1D;
    const NurbsSurfacePatch3D split_plane(
        NurbsBasis1D(1, {0.0, 0.0, 0.5, 1.0, 1.0}),
        NurbsBasis1D(1, {0.0, 0.0, 1.0, 1.0}),
        {{{0.0, 0.0, 1.0}, {0.0, 1.0, 1.0}},
         {{0.5, 0.0, 1.0}, {0.5, 1.0, 1.0}},
         {{1.0, 0.0, 1.0}, {1.0, 1.0, 1.0}}},
        {{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}});
    const QueryContext context(NurbsSurfaceModel3D({split_plane}, {0}, {}));
    require(context.exact.size() == 2, "split-boundary fixture has two exact leaves");
    require_certified(query(context, {0.5, 0.5, 0.0}, {0, 0.5, 0.5}, true),
                      0, -1, true);
}

void test_outside_small_residual()
{
    const double tiny = std::ldexp(1.0, -44);
    const QueryContext context(NurbsSurfaceModel3D(
        {make_plane(-tiny, true), make_plane(1.0)}, {0, 0}, {}));
    require_certified(query(context, {0.5, 0.5, 0.0}, {1, 0.5, 0.5}, true),
                      0, -1, true);
}

void test_close_independent_root_before_endpoint()
{
    const double before = 1.0 - std::ldexp(1.0, -40);
    const QueryContext context(NurbsSurfaceModel3D(
        {make_plane(before, true), make_plane(1.0)}, {0, 0}, {}));
    const auto result = query(context, {0.5, 0.5, 0.0}, {1, 0.5, 0.5}, false);
    require_certified(result, 1, -1, false);
    require(result.open_roots.front().certificate.t_interval.upper < 1.0
                && std::abs(result.open_roots.front().representative.edge_parameter
                               - before) < 1.0e-13,
            "root below the old tie scale remains distinct from native endpoint");
}

void test_nonfinite_interval_cannot_be_excluded()
{
    const auto extreme_plane = [](double z, bool reversed) {
        const auto plane = make_plane(z, reversed);
        return NurbsSurfacePatch3D(
            plane.basis_u(), plane.basis_v(), plane.control_net(),
            {{1.0e-300, 1.0e300}, {1.0e300, 1.0e-300}});
    };
    // All source coordinates and weights are finite, and all weights positive.
    // The present binary64-endpoint interval backend cannot represent the
    // approximately 1e600 ratios in these first-weight-normalized pieces.
    // Two genuine open roots cancel in parity. If invalid intervals were
    // mistaken for CertifiedMiss, the healthy final plane alone could produce
    // a false Certified result with the same expected start side. Therefore a
    // pathological endpoint alone would NOT be an adequate regression fixture.
    const QueryContext context(NurbsSurfaceModel3D(
        {extreme_plane(0.25, true), extreme_plane(0.75, false),
         make_plane(1.0, true)}, {0, 0, 0}, {}));
    auto budget = test_budget();
    budget.max_subdivision_nodes = 64;
    budget.max_newton_steps = 64;
    budget.max_precision_bits = 128;
    const auto result = query(context, {0.5, 0.5, 0.0}, {2, 0.5, 0.5}, false, budget);
    require(result.status == PathStatus3D::Unresolved,
            "nonfinite interval arithmetic must return an explicit unresolved path");
    require(result.diagnostics.reason.find("NonFinite") != std::string::npos,
            "range failure must be diagnosed, not confused with geometric exclusion");
    require_rejected([&] { select_native_endpoint_path_events_3d(result, false, true); },
                     "invalid intervals cannot yield an assemblable partial root set");
}

void test_adapter_rejects_incomplete_or_wrong_state()
{
    const auto context = plane_context();
    const auto good = query(context, {0.5, 0.5, 0.0}, {0, 0.5, 0.5}, true);
    require_certified(good, 0, -1, true);
    for (int gate = 0; gate < 4; ++gate) {
        auto bad = good;
        if (gate == 0) bad.coverage.complete = false;
        if (gate == 1) bad.coverage.order_certified = false;
        if (gate == 2) bad.coverage.transitions_certified = false;
        if (gate == 3) bad.coverage.representatives_certified = false;
        require_rejected([&] { select_native_endpoint_path_events_3d(bad, true, false); },
                         "adapter must reject each independently failed gate");
    }
    require_rejected([&] { select_native_endpoint_path_events_3d(good, false, false); },
                     "wrong support label cannot be repaired by desired side");
    auto candidate = good;
    candidate.endpoint->certificate.kind = RootProofKind3D::Candidate;
    candidate.endpoint->certificate.existence_proved = false;
    require_rejected([&] { select_native_endpoint_path_events_3d(candidate, true, false); },
                     "numerical candidate cannot be adapted as a certified endpoint");
}

void test_low_budget()
{
    const auto context = three_plane_context();
    auto budget = test_budget();
    budget.max_candidate_regions = 1;
    budget.max_subdivision_nodes = 1;
    const auto result = query(context, {0.5, 0.5, 0.0}, {2, 0.5, 0.5}, false, budget);
    require(result.status == PathStatus3D::BudgetExceeded,
            "small deterministic budget must fail explicitly");
    require(!result.coverage.complete, "budget exhaustion cannot certify partial coverage");
    require_rejected([&] { select_native_endpoint_path_events_3d(result, false, true); },
                     "partial roots cannot enter correction assembly");
}

void test_invalid_and_unsupported()
{
    const auto context = plane_context();
    const auto invalid = query(context, {0.5, 0.5, 0.0},
                               {0, std::numeric_limits<double>::quiet_NaN(), 0.5}, true);
    require(invalid.status == PathStatus3D::InvalidInput, "nonfinite native parameter");
    const auto boundary = query(context, {0.0, 0.5, 0.0}, {0, 0.0, 0.5}, true);
    require(boundary.status == PathStatus3D::UnsupportedEndpoint,
            "true patch-boundary endpoint is explicitly unsupported");
    const auto degenerate = query(context, {0.5, 0.5, 1.0}, {0, 0.5, 0.5}, true);
    require(degenerate.status == PathStatus3D::DegenerateSegment,
            "exactly zero native segment is not a certified path");
    const auto tangent = query(context, {0.0, 0.5, 1.0}, {0, 0.5, 0.5}, true);
    require(tangent.status != PathStatus3D::Certified,
            "overlapping/tangent path cannot receive a transverse certificate");
    const auto wrong_side = query(context, {0.5, 0.5, 0.0}, {0, 0.5, 0.5}, false);
    require(wrong_side.status != PathStatus3D::Certified,
            "expected start label cannot override the geometric side certificate");
}

void test_repeated_query_and_rigid_transform()
{
    const auto context = plane_context();
    const Eigen::Vector3d start(0.5, 0.5, 0.0);
    const auto first = query(context, start, {0, 0.5, 0.5}, true);
    const auto second = query(context, start, {0, 0.5, 0.5}, true);
    require_certified(first, 0, -1, true);
    require_certified(second, 0, -1, true);
    require(first.endpoint->certificate.t_interval.lower
                == second.endpoint->certificate.t_interval.lower
                && first.endpoint->certificate.t_interval.upper
                       == second.endpoint->certificate.t_interval.upper
                && first.endpoint->transition.sign == second.endpoint->transition.sign
                && first.coverage.visited_regions == second.coverage.visited_regions,
            "repeated identical query has deterministic geometry and work counts");
    NativeNurbsSurface3D source;
    source.patches = context.model.patches();
    source.patch_components = {0};
    source.exact_inside = [](const Eigen::Vector3d& point) { return point.z() < 1.0; };
    Eigen::Matrix3d rotation;
    rotation << 0.0, 0.0, 1.0,
                0.0, 1.0, 0.0,
               -1.0, 0.0, 0.0;
    const RigidTransform3D transform(rotation, Eigen::Vector3d::Zero(),
                                     {0.125, -0.25, 0.5});
    const auto transformed = transform_native_nurbs_surface_3d(source, transform);
    const QueryContext moved(transformed.geometry_model());
    const auto result = query(moved, transform.forward_point(start),
                              {0, 0.5, 0.5}, true);
    require_certified(result, 0, -1, true);
    require((result.endpoint->representative.point
                 - transform.forward_point(first.endpoint->representative.point)).norm()
                < 1.0e-12,
            "rigid transform preserves native endpoint and physical direction");
}

void test_intersector_cached_native_query()
{
    const auto& fixture = cylinder_fixtures().at(1); // D32 Tx: fast closed model.
    const auto surface = fixture_surface(fixture);
    const kfbim::geometry3d::NurbsSurfaceIntersector3D intersector(
        surface.geometry_model());
    NativeEndpointQueryOptions3D options;
    options.expected_start_inside = false;
    const NativeSurfaceEndpoint3D endpoint{fixture.patch, fixture.u(), fixture.v()};
    const auto run = [&](const kfbim::geometry3d::NurbsSurfaceIntersector3D& source,
                         const char* phase) {
        auto result = source.intersect_segment_to_native_endpoint(
            fixture.start, endpoint, options, test_budget());
        std::cout << " cache_phase=" << phase << " status="
                  << kfbim::geometry3d::native_endpoint_path_status_name_3d(result.status)
                  << " open=" << result.open_roots.size()
                  << " visited=" << result.coverage.visited_regions
                  << " reason=" << result.diagnostics.reason << '\n';
        if (result.status != PathStatus3D::Certified)
            std::cout << " message=" << result.message << "\n replay=" << result.dump << '\n';
        require_certified(result, 0, +1, false);
        return result;
    };
    const auto first = run(intersector, "initial_call_once");
    const auto second = run(intersector, "reused_geometry_and_bounds");
    const auto copied_intersector = intersector;
    const auto copied = run(copied_intersector, "copied_immutable_intersector");

    const auto same_box = [](const kfbim::geometry3d::RootParameterBox3D& a,
                             const kfbim::geometry3d::RootParameterBox3D& b) {
        return a.u.lower == b.u.lower && a.u.upper == b.u.upper
            && a.v.lower == b.v.lower && a.v.upper == b.v.upper;
    };
    for (const auto* result : {&second, &copied}) {
        const auto& a = *first.endpoint;
        const auto& b = *result->endpoint;
        require(a.certificate.kind == b.certificate.kind
                    && a.certificate.proof_id == b.certificate.proof_id
                    && a.certificate.event_id == b.certificate.event_id
                    && a.certificate.patch_index == b.certificate.patch_index
                    && a.certificate.source_element == b.certificate.source_element
                    && a.certificate.precision_bits == b.certificate.precision_bits
                    && a.certificate.contraction_bound == b.certificate.contraction_bound
                    && same_box(a.certificate.uniqueness_box, b.certificate.uniqueness_box)
                    && same_box(a.certificate.root_enclosure, b.certificate.root_enclosure)
                    && a.certificate.t_interval.lower == b.certificate.t_interval.lower
                    && a.certificate.t_interval.upper == b.certificate.t_interval.upper,
                "member cache/copy must preserve native endpoint certificate and event identity");
        require(a.representative.u == b.representative.u
                    && a.representative.v == b.representative.v
                    && a.representative.point == b.representative.point
                    && a.representative.normal == b.representative.normal
                    && a.transition.sign == b.transition.sign
                    && a.transition.oriented_dot_lower == b.transition.oriented_dot_lower
                    && a.transition.oriented_dot_upper == b.transition.oriented_dot_upper
                    && a.accuracy.point_error_bound == b.accuracy.point_error_bound
                    && a.accuracy.parameter_error_bound == b.accuracy.parameter_error_bound
                    && a.accuracy.normal_error_bound == b.accuracy.normal_error_bound,
                "member cache/copy must preserve representative, orientation and accuracy");
        require(first.coverage.candidate_regions == result->coverage.candidate_regions
                    && first.coverage.visited_regions == result->coverage.visited_regions
                    && first.diagnostics.newton_steps == result->diagnostics.newton_steps,
                "cached conservative bounds must preserve deterministic query work");
    }
}

struct TestCase {
    std::string name;
    std::function<void()> run;
};

std::vector<TestCase> make_tests()
{
    std::vector<TestCase> tests = {
        {"fixture_metadata", test_fixture_metadata},
        {"plane_endpoint", test_plane_endpoint},
        {"two_ordinary_roots", test_two_ordinary_roots},
        {"split_boundary_endpoint", test_split_boundary_endpoint},
        {"outside_small_residual", test_outside_small_residual},
        {"close_independent_root_before_endpoint", test_close_independent_root_before_endpoint},
        {"nonfinite_interval_cannot_be_excluded", test_nonfinite_interval_cannot_be_excluded},
        {"adapter_rejects_incomplete_or_wrong_state", test_adapter_rejects_incomplete_or_wrong_state},
        {"low_budget", test_low_budget},
        {"invalid_and_unsupported", test_invalid_and_unsupported},
        {"repeated_query_and_rigid_transform", test_repeated_query_and_rigid_transform},
        {"intersector_cached_native_query", test_intersector_cached_native_query}
    };
    for (const auto& fixture : cylinder_fixtures())
        tests.push_back({fixture.name, [fixture] { test_cylinder_path(fixture); }});
    return tests;
}

} // namespace

int main(int argc, char** argv)
{
    std::string filter;
    bool list = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--case" && i + 1 < argc) {
            filter = argv[++i];
        } else if (argument == "--list") {
            list = true;
        } else if (argument == "--help") {
            std::cout << "native_endpoint_path_3d_test [--case SUBSTRING] [--list]\n"
                         "Runs bounded geometric queries only; never runs a PDE.\n";
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: " << argument << '\n';
            return 2;
        }
    }
    const auto tests = make_tests();
    int selected = 0;
    int failures = 0;
    for (const auto& test : tests) {
        if (!filter.empty() && test.name.find(filter) == std::string::npos)
            continue;
        ++selected;
        if (list) {
            std::cout << test.name << '\n';
            continue;
        }
        std::cout << "BEGIN " << test.name << std::endl;
        const auto begin = std::chrono::steady_clock::now();
        try {
            test.run();
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - begin).count();
            std::cout << "PASS " << test.name << " wall_seconds="
                      << std::setprecision(9) << seconds << std::endl;
        } catch (const std::exception& error) {
            ++failures;
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - begin).count();
            std::cerr << "FAIL " << test.name << " wall_seconds="
                      << std::setprecision(9) << seconds
                      << " reason=" << error.what() << std::endl;
        }
    }
    if (selected == 0) {
        std::cerr << "No test matches --case " << filter << '\n';
        return 2;
    }
    if (!list)
        std::cout << "SUMMARY selected=" << selected
                  << " failures=" << failures << std::endl;
    return failures == 0 ? 0 : 1;
}
