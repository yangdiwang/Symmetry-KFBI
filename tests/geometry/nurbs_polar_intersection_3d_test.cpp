#include "src/geometry/nurbs_surface_intersector_3d.hpp"
#include "src/support/geometry/native_nurbs_surface_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace kfbim::geometry3d;
using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::make_native_nurbs_surface_3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void near(double a, double b, double tolerance, const std::string& message)
{
    require(std::isfinite(a) && std::isfinite(b)
                && std::abs(a - b) <= tolerance, message);
}

void near(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
          double tolerance, const std::string& message)
{
    require(a.allFinite() && b.allFinite() && (a - b).norm() <= tolerance,
            message);
}

// A tangent root is ill-conditioned in its longitudinal coordinate. Its
// residual is checked independently below, against the original NURBS patch.
template<class Root>
void compare_owner(const Root& a, const Root& b, const std::string& label,
                   double location_tolerance)
{
    require(a.patch_index == b.patch_index, label + " patch owner");
    near(a.u, b.u, location_tolerance, label + " u");
    near(a.v, b.v, location_tolerance, label + " v");
    near(a.point, b.point, location_tolerance, label + " point");
    near(a.normal, b.normal, 4.0 * location_tolerance, label + " normal");
    near(a.residual, b.residual, 2e-11, label + " residual");
    near(a.transversality, b.transversality, 4.0 * location_tolerance,
         label + " transversality");
    near(a.reliable_transversality_tolerance,
         b.reliable_transversality_tolerance, 1e-12,
         label + " transversality threshold");
    require(a.feature_edge_contact == b.feature_edge_contact,
            label + " feature contact");
}

void compare_crossing(const NurbsSurfaceCrossing3D& a,
                      const NurbsSurfaceCrossing3D& b,
                      const std::string& label, double tolerance)
{
    compare_owner(a, b, label, tolerance);
    require(a.component == b.component, label + " component");
    near(a.edge_parameter, b.edge_parameter, 10.0 * tolerance,
         label + " longitudinal parameter");
    require(a.owners.size() == b.owners.size(), label + " owner count");
    for (std::size_t i = 0; i < a.owners.size(); ++i)
        compare_owner(a.owners[i], b.owners[i],
                      label + " owner " + std::to_string(i), tolerance);
}

void compare_roots(const std::vector<NurbsSurfaceCrossing3D>& a,
                   const std::vector<NurbsSurfaceCrossing3D>& b,
                   const std::string& label, double tolerance)
{
    require(a.size() == b.size(), label + " root count");
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (i != 0) {
            require(a[i - 1].edge_parameter <= a[i].edge_parameter,
                    label + " reference root ordering");
            require(b[i - 1].edge_parameter <= b[i].edge_parameter,
                    label + " polar root ordering");
        }
        compare_crossing(a[i], b[i], label + " root " + std::to_string(i),
                         tolerance);
    }
}

std::vector<std::array<double, 2>> interval_union(
    std::vector<std::array<double, 2>> intervals)
{
    std::sort(intervals.begin(), intervals.end());
    std::vector<std::array<double, 2>> merged;
    for (const auto& interval : intervals) {
        require(std::isfinite(interval[0]) && std::isfinite(interval[1])
                    && interval[0] <= interval[1], "valid unresolved interval");
        if (!merged.empty() && interval[0] <= merged.back()[1] + 1e-12)
            merged.back()[1] = std::max(merged.back()[1], interval[1]);
        else
            merged.push_back(interval);
    }
    return merged;
}

template<class Diagnostics>
void compare_unresolved(const Diagnostics& a, const Diagnostics& b,
                        bool a_unresolved, bool b_unresolved,
                        const std::string& label)
{
    require(a_unresolved == b_unresolved, label + " unresolved status");
    const auto first = interval_union(a.unresolved_longitudinal_intervals);
    const auto second = interval_union(b.unresolved_longitudinal_intervals);
    require(first.size() == second.size(), label + " unresolved interval union");
    for (std::size_t i = 0; i < first.size(); ++i) {
        near(first[i][0], second[i][0], 1e-8, label + " unresolved lower bound");
        near(first[i][1], second[i][1], 1e-8, label + " unresolved upper bound");
    }
}

void validate_roots(const NurbsSurfaceIntersector3D& intersector,
                    const std::vector<NurbsSurfaceCrossing3D>& roots,
                    const Eigen::Vector3d& start, const Eigen::Vector3d& end,
                    const std::string& label)
{
    const double tolerance = 8.0 * intersector.geometry_tolerance();
    for (const auto& root : roots) {
        require(root.residual <= tolerance && root.residual >= 0.0,
                label + " accepted residual bound");
        const auto original_point =
            intersector.model().patch(root.patch_index).evaluate(root.u, root.v);
        near(original_point, root.point, tolerance, label + " original patch");
        near(original_point, start + root.edge_parameter * (end - start),
             tolerance, label + " original patch/segment residual");
        for (const auto& owner : root.owners) {
            const auto owner_point = intersector.model().patch(owner.patch_index)
                                         .evaluate(owner.u, owner.v);
            near(owner_point, owner.point, tolerance, label + " owner patch");
        }
    }
}

void compare_segment(const NurbsSurfaceIntersectionResult3D& a,
                     const NurbsSurfaceIntersectionResult3D& b,
                     const std::string& label, double tolerance = 2e-8)
{
    compare_roots(a.crossings, b.crossings, label, tolerance);
    require(a.overlap_detected == b.overlap_detected, label + " overlap");
    compare_unresolved(a.diagnostics, b.diagnostics,
                       a.diagnostics.unresolved_candidates != 0,
                       b.diagnostics.unresolved_candidates != 0, label);
}

void compare_edge(const NurbsCartesianEdgeIntersections3D& a,
                  const NurbsCartesianEdgeIntersections3D& b,
                  const std::string& label, double tolerance)
{
    compare_roots(a.crossings, b.crossings, label, tolerance);
    require(a.toggled_components == b.toggled_components, label + " toggles");
    require(a.confirmed_transverse_count == b.confirmed_transverse_count,
            label + " transverse count");
    require(a.root_count_known == b.root_count_known, label + " root certainty");
    require(a.parity_known_from_roots == b.parity_known_from_roots,
            label + " parity certainty");
    require(a.has_near_tangent_candidate == b.has_near_tangent_candidate,
            label + " tangent status");
    require(a.changes_inside_outside == b.changes_inside_outside,
            label + " inside/outside change");
    require(a.changes_component_membership == b.changes_component_membership,
            label + " membership change");
    compare_unresolved(a.diagnostics, b.diagnostics,
                       a.diagnostics.unresolved_candidates != 0,
                       b.diagnostics.unresolved_candidates != 0, label);
    require(a.ambiguous_clusters.size() == b.ambiguous_clusters.size(),
            label + " ambiguous cluster count");
    for (std::size_t i = 0; i < a.ambiguous_clusters.size(); ++i) {
        const auto& first = a.ambiguous_clusters[i];
        const auto& second = b.ambiguous_clusters[i];
        require(first.component == second.component, label + " cluster component");
        near(first.edge_parameter_begin, second.edge_parameter_begin, 1e-8,
             label + " cluster begin");
        near(first.edge_parameter_end, second.edge_parameter_end, 1e-8,
             label + " cluster end");
        compare_roots(first.candidates, second.candidates,
                      label + " cluster candidates", tolerance);
    }
}

std::vector<std::size_t> candidate_ids(const NurbsSurfaceIntersector3D& intersector,
                                     const NurbsCartesianEdgeQuery3D& edge)
{
    const NurbsAabb3D bounds{edge.start.cwiseMin(edge.end),
                             edge.start.cwiseMax(edge.end)};
    std::vector<std::size_t> ids;
    for (const auto& descriptor : intersector.query_elements())
        if (descriptor.bounds.overlaps(bounds, intersector.geometry_tolerance()))
            ids.push_back(descriptor.id);
    return ids;
}

void test_hollow_cylinder()
{
    const auto native = make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    NurbsSurfaceIntersectorOptions3D options;
    options.collect_unresolved_regions = true;
    options.preserve_non_g1_root_owners = true;
    options.maximum_element_extent = 0.09375;
    options.use_polar_surface_evaluation = false;
    const NurbsSurfaceIntersector3D reference(native.geometry_model(), options);
    options.use_polar_surface_evaluation = true;
    const NurbsSurfaceIntersector3D polar(native.geometry_model(), options);
    require(reference.query_element_count() == polar.query_element_count()
                && polar.query_element_count() > 16,
            "finite extent creates matching subdivided cylinder leaves");
    require(polar.maximum_query_element_extent() <= options.maximum_element_extent,
            "query leaves obey requested extent");

    struct Fixture {
        const char* name;
        NurbsCartesianEdgeQuery3D edge;
        int expected_roots; // -1: conservative contact classification is allowed.
        double tolerance = 2e-8;
    };
    const std::vector<Fixture> fixtures{
        {"outer wall", {0, 1, 2, 3, {0.58, -0.01, 0.03}, {0.64, -0.01, 0.03}}, 1},
        {"inner wall", {0, 2, 2, 3, {0.28, -0.01, 0.03}, {0.34, -0.01, 0.03}}, 1},
        {"four ordered wall crossings", {0, 8, 2, 3,
            {-0.6, -0.01, 0.03}, {0.7, -0.01, 0.03}}, 4},
        {"top cap", {2, 3, 2, 3, {0.43, 0.01, 0.64}, {0.43, 0.01, 0.70}}, 1},
        {"smooth seam", {1, 4, 2, 3, {0.06, 0.48, 0.0}, {0.06, 0.52, 0.0}}, 1},
        {"cap rim contact", {1, 5, 2, 3, {0.61, -0.07, 0.67}, {0.61, -0.03, 0.67}}, -1, 5e-7},
        {"tangent", {1, 20, 21, 22, {0.61, -0.10, 0.0}, {0.61, 0.0, 0.0}}, 1, 5e-7},
        {"near tangent miss", {1, 6, 2, 3, {0.6100001, -0.10, 0.0}, {0.6100001, 0.0, 0.0}}, 0},
        {"grid endpoint minimum miss", {0, 41, 22, 38,
            {0.421875, -0.46875, 0.28125}, {0.46875, -0.46875, 0.28125}}, 0},
        {"far miss", {0, 7, 2, 3, {0.8, 0.8, 0.0}, {0.9, 0.8, 0.0}}, 0}};

    for (const auto& fixture : fixtures) {
        const std::string label(fixture.name);
        const auto a = reference.intersect_cartesian_edge(fixture.edge);
        const auto b = polar.intersect_cartesian_edge(fixture.edge);
        compare_edge(a, b, label, fixture.tolerance);
        if (fixture.expected_roots >= 0)
            require(a.crossings.size() == static_cast<std::size_t>(fixture.expected_roots),
                    label + " known reference root count");
        validate_roots(reference, a.crossings, fixture.edge.start, fixture.edge.end, label);
        validate_roots(polar, b.crossings, fixture.edge.start, fixture.edge.end, label);
        if (label == "tangent")
            require(a.has_near_tangent_candidate && !a.parity_known_from_roots,
                    "tangent remains a conservative event");

        NurbsCartesianEdgeQueryOptions3D query_options;
        query_options.route = NurbsCartesianEdgeQueryRoute3D::OptimizedCertified;
        const auto optimized_a = reference.intersect_cartesian_edge(
            fixture.edge, candidate_ids(reference, fixture.edge), query_options);
        const auto optimized_b = polar.intersect_cartesian_edge(
            fixture.edge, candidate_ids(polar, fixture.edge), query_options);
        compare_edge(optimized_a, optimized_b, label + " optimized route", fixture.tolerance);
        validate_roots(polar, optimized_b.crossings, fixture.edge.start,
                       fixture.edge.end, label + " optimized route");
        if (label == "outer wall" || label == "inner wall")
            require(optimized_a.crossings.size() == 1,
                    label + " optimized route finds the known crossing");
    }

    const Eigen::Vector3d oblique_start(0.56, 0.08, -0.12);
    const Eigen::Vector3d oblique_end(0.66, 0.10, -0.08);
    const auto oblique_a = reference.intersect_segment(oblique_start, oblique_end);
    const auto oblique_b = polar.intersect_segment(oblique_start, oblique_end);
    compare_segment(oblique_a, oblique_b, "oblique wall crossing");
    require(oblique_a.crossings.size() == 1, "oblique fixture crosses outer wall");
    validate_roots(polar, oblique_b.crossings, oblique_start, oblique_end, "oblique");

    // Return an actual copy after its source dies; then move it and query after
    // the moved-from copy also dies. This detects evaluator/model pointer aliasing.
    const auto copied = [&]() {
        const NurbsSurfaceIntersector3D original(native.geometry_model(), options);
        return NurbsSurfaceIntersector3D(original);
    }();
    compare_segment(oblique_b, copied.intersect_segment(oblique_start, oblique_end),
                    "copy survives source destruction");
    const auto moved = [&]() {
        NurbsSurfaceIntersector3D temporary(copied);
        return NurbsSurfaceIntersector3D(std::move(temporary));
    }();
    compare_segment(oblique_b, moved.intersect_segment(oblique_start, oblique_end),
                    "move survives source destruction");
    for (const auto& point : std::array<Eigen::Vector3d, 2>{
             Eigen::Vector3d(0.46, -0.05, 0.0), Eigen::Vector3d(0.06, -0.05, 0.0)})
        require(reference.containing_components(point) == polar.containing_components(point),
                "containment parity agrees");
    require(polar.containing_components({0.46, -0.05, 0.0}) == std::vector<int>{0}
                && polar.containing_components({0.06, -0.05, 0.0}).empty(),
            "known shell and bore containment");
}

void test_candidate_certificates()
{
    // This existing native torus fixture has a certifiable transverse crossing.
    const auto native = make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    NurbsSurfaceIntersectorOptions3D options;
    options.collect_unresolved_regions = true;
    options.use_polar_surface_evaluation = false;
    const NurbsSurfaceIntersector3D reference(native.geometry_model(), options);
    options.use_polar_surface_evaluation = true;
    const NurbsSurfaceIntersector3D polar(native.geometry_model(), options);
    const Eigen::Vector3d start(0.80, -0.04, 0.03), end(0.84, -0.04, 0.03);
    const auto first = reference.conservative_segment_candidates(start, end);
    const auto second = polar.conservative_segment_candidates(start, end);
    require(!first.empty() && first.size() == second.size(), "certificate candidates");
    bool found_unique = false;
    for (std::size_t i = 0; i < first.size(); ++i) {
        require(first[i].patch_index() == second[i].patch_index()
                    && first[i].component() == second[i].component(),
                "candidate metadata order");
        near(first[i].bounds().lower, second[i].bounds().lower, 0.0, "candidate lower");
        near(first[i].bounds().upper, second[i].bounds().upper, 0.0, "candidate upper");
        const auto a = reference.certify_candidate_segment(first[i], start, end);
        const auto b = polar.certify_candidate_segment(second[i], start, end);
        require(a.kind == b.kind && a.crossing.has_value() == b.crossing.has_value(),
                "candidate certificate kind/root availability");
        require(a.overlap_detected == b.overlap_detected
                    && a.segment_parameter_strictly_interior == b.segment_parameter_strictly_interior
                    && a.element_parameter_strictly_interior == b.element_parameter_strictly_interior
                    && a.patch_parameter_strictly_interior == b.patch_parameter_strictly_interior,
                "candidate certificate interior/overlap status");
        compare_unresolved(a.diagnostics, b.diagnostics,
                           a.diagnostics.unresolved_boxes != 0,
                           b.diagnostics.unresolved_boxes != 0, "candidate certificate");
        if (a.crossing) {
            compare_crossing(*a.crossing, *b.crossing, "certificate crossing", 2e-8);
            validate_roots(polar, {*b.crossing}, start, end, "certificate crossing");
        }
        found_unique = found_unique
            || (a.kind == NurbsElementSegmentCertificateKind3D::CertifiedUniqueTransverseRoot
                && a.crossing.has_value());
    }
    require(found_unique, "at least one nonempty unique-root certificate");
    const auto a = reference.intersect_segment_candidates(start, end, first);
    const auto b = polar.intersect_segment_candidates(start, end, second);
    compare_segment(a.intersection, b.intersection, "filtered candidate intersection");
    require(a.all_candidates_processed == b.all_candidates_processed
                && a.independent_crossing_limit_reached == b.independent_crossing_limit_reached,
            "filtered candidate completion status");
    require(a.intersection.crossings.size() == 1, "filtered candidate known crossing");
}

} // namespace

int main()
{
    try {
        test_hollow_cylinder();
        test_candidate_certificates();
        std::cout << "polar intersection integration tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "polar intersection integration test failed: " << error.what() << '\n';
        return 1;
    }
}
