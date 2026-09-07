#include "src/support/trace/restrict_owner_geometry_preprocessor_3d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>

#include <utility>
#include <vector>
namespace {

using Mode = kfbim::app3d::RestrictOwnerPreprocessMode3D;
using NormalizedClass =
    kfbim::app3d::RestrictOwnerNormalizedClass3D;
using QueryPath = kfbim::app3d::RestrictOwnerQueryPath3D;
using FallbackCause =
    kfbim::app3d::RestrictOwnerFallbackCause3D;
using CertificateKind = kfbim::geometry3d::NurbsElementSegmentCertificateKind3D;
using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::NativeNurbsSurface3D;
using kfbim::app3d::RestrictOwnerGeometryPreprocessor3D;
using kfbim::app3d::RestrictOwnerPreprocessOptions3D;
using kfbim::app3d::RestrictOwnerSampleInput3D;
using kfbim::app3d::SurfaceDofCloud3D;
using kfbim::app3d::make_native_nurbs_surface_3d;
using kfbim::app3d::make_native_surface_dofs_3d;
using kfbim::app3d::parameter_dof_candidates_2x2;


void require(bool value, const std::string& message)
{
    if (!value)
        throw std::runtime_error(message);
}

template <class Function>
void require_throws_contains(
    Function&& function,
    const std::string& expected,
    const std::string& message)
{
    try {
        function();
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(expected)
                    != std::string::npos,
                message + ": unexpected message: " + error.what());
        return;
    }
    throw std::runtime_error(message + ": exception not thrown");
}

void test_mode_parsing()
{
    using kfbim::app3d::parse_restrict_owner_preprocess_mode_3d;
    require(parse_restrict_owner_preprocess_mode_3d(
                "full_intersection_reference")
                == Mode::FullIntersectionReference,
            "parse full reference mode");
    require(parse_restrict_owner_preprocess_mode_3d(
                "optimized_intersection")
                == Mode::OptimizedIntersection,
            "parse optimized mode");
    require(parse_restrict_owner_preprocess_mode_3d(
                "region_closest_hybrid")
                == Mode::RegionClosestHybrid,
            "parse hybrid mode");
    for (const std::string& invalid :
         {"", "full", "full_intersection_reference trailing"}) {
        require_throws_contains(
            [&] {
                (void)parse_restrict_owner_preprocess_mode_3d(invalid);
            },
            "invalid restrict-owner preprocessing mode",
            "mode parser rejects non-exact text");
    }
    require(NormalizedClass::Target != NormalizedClass::UniqueForeign
                && QueryPath::FullIntersection != QueryPath::Count
                && FallbackCause::None != FallbackCause::Count
                && CertificateKind::CertifiedMiss
                       != CertificateKind::Unresolved,
            "stable app and core enum interfaces are available");
}


int center_dof(const SurfaceDofCloud3D& cloud, int patch)
{
    const auto& p = cloud.patches.at(static_cast<std::size_t>(patch));
    return p.dof_index(p.nu / 2, p.nv / 2);
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> transverse_segment(
    const NativeNurbsSurface3D& surface,
    int patch,
    double half_length = 0.04)
{
    const auto& p = surface.patches.at(static_cast<std::size_t>(patch));
    const double u = 0.5 * (p.domain_start_u() + p.domain_end_u());
    const double v = 0.5 * (p.domain_start_v() + p.domain_end_v());
    const auto data = p.evaluate_with_derivatives(u, v);
    const Eigen::Vector3d normal =
        data.du.cross(data.dv).normalized();
    return {data.point - half_length * normal,
            data.point + half_length * normal};
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> transverse_segment_at(
    const NativeNurbsSurface3D& surface,
    int patch,
    double u_fraction,
    double v_fraction,
    double half_length = 0.04)
{
    const auto& p = surface.patches.at(static_cast<std::size_t>(patch));
    const double u = p.domain_start_u()
        + u_fraction * (p.domain_end_u() - p.domain_start_u());
    const double v = p.domain_start_v()
        + v_fraction * (p.domain_end_v() - p.domain_start_v());
    const auto data = p.evaluate_with_derivatives(u, v);
    const Eigen::Vector3d normal =
        data.du.cross(data.dv).normalized();
    return {data.point - half_length * normal,
            data.point + half_length * normal};
}

RestrictOwnerSampleInput3D one_wrong_side_sample(
    int target,
    const Eigen::Vector3d& query,
    const Eigen::Vector3d& support)
{
    RestrictOwnerSampleInput3D input;
    input.target_dof = target;
    input.query = query;
    input.support_points.fill(
        query + Eigen::Vector3d(0.01, 0.0, 0.0));
    input.support_points[7] = support;
    input.wrong_side[7] = true;
    return input;
}

kfbim::app3d::RestrictOwnerPreprocessResult3D one_result(
    RestrictOwnerGeometryPreprocessor3D& preprocessor,
    const RestrictOwnerSampleInput3D& input)
{
    const auto result = preprocessor.preprocess_sample(input);
    for (std::size_t q = 0; q < result.nodes.size(); ++q) {
        require(result.nodes[q].has_value() == (q == 7),
                "only the wrong-side slot is classified");
    }
    return *result.nodes[7];
}

void require_full_result(
    const kfbim::app3d::RestrictOwnerPreprocessResult3D& result,
    int owner,
    NormalizedClass owner_class,
    const std::string& message)
{
    require(result.owner_dof == owner
                && result.owner_class == owner_class
                && result.query_path == QueryPath::FullIntersection,
            message);
}

struct Fixture {
    NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    SurfaceDofCloud3D cloud =
        make_native_surface_dofs_3d(surface, 3.0 / 16.0);
};

void test_full_reference_and_local_topology()
{
    Fixture fixture;
    RestrictOwnerGeometryPreprocessor3D preprocessor(
        fixture.surface, fixture.cloud, 3.0 / 16.0,
        Mode::FullIntersectionReference);
    const int target = center_dof(fixture.cloud, 0);

    require_full_result(
        one_result(
            preprocessor,
            one_wrong_side_sample(
                target, {2.0, 2.0, 2.0}, {2.1, 2.0, 2.0})),
        target, NormalizedClass::Target,
        "no crossing retains target");

    const auto same = transverse_segment(fixture.surface, 0);
    require_full_result(
        one_result(
            preprocessor,
            one_wrong_side_sample(target, same.first, same.second)),
        target, NormalizedClass::Target,
        "same patch crossing retains target");

    const auto& smooth = fixture.cloud.patches[0].smooth_patch_ids;
    const auto g1 = std::find_if(
        smooth.begin(), smooth.end(),
        [](int patch) { return patch != 0; });
    require(g1 != smooth.end(), "fixture has an explicit G1 neighbor");
    const auto g1_segment = transverse_segment(fixture.surface, *g1);
    require_full_result(
        one_result(
            preprocessor,
            one_wrong_side_sample(
                target, g1_segment.first, g1_segment.second)),
        target, NormalizedClass::Target,
        "explicit G1 crossing retains target");

    const int foreign_target = center_dof(fixture.cloud, 6);
    const int foreign_patch = 7;
    const auto foreign_segment =
        transverse_segment(fixture.surface, foreign_patch);
    const kfbim::geometry3d::NurbsSurfaceIntersector3D oracle(
        fixture.surface.geometry_model());
    const auto intersection = oracle.intersect_segment(
        foreign_segment.first, foreign_segment.second);
    require(intersection.crossings.size() == 1,
            "foreign fixture has one real crossing");
    const auto candidates = parameter_dof_candidates_2x2(
        fixture.surface, fixture.cloud, foreign_patch,
        intersection.crossings.front().u,
        intersection.crossings.front().v);
    const int expected_owner = *std::min_element(
        candidates.begin(), candidates.end(),
        [&](int a, int b) {
            const auto distance = [&](int dof) {
                return (fixture.cloud.dofs[
                            static_cast<std::size_t>(dof)].point
                        - intersection.crossings.front().point)
                    .squaredNorm();
            };
            return distance(a) < distance(b);
        });
    const auto foreign = one_result(
        preprocessor,
        one_wrong_side_sample(
            foreign_target,
            foreign_segment.first,
            foreign_segment.second));
    require_full_result(
        foreign, expected_owner, NormalizedClass::UniqueForeign,
        "foreign crossing selects the nearest 2-by-2 DOF");
    require(foreign.foreign_crossing.has_value()
                && foreign.foreign_crossing->patch_index == foreign_patch,
            "foreign result preserves the crossing");

    fixture.cloud.patches[0].smooth_patch_ids = {1};
    fixture.cloud.patches[1].smooth_patch_ids = {0, 2};
    fixture.cloud.patches[2].smooth_patch_ids = {1};
    RestrictOwnerGeometryPreprocessor3D chain_preprocessor(
        fixture.surface, fixture.cloud, 3.0 / 16.0,
        Mode::FullIntersectionReference);
    const auto patch_two = transverse_segment(fixture.surface, 2);
    const auto chain = one_result(
        chain_preprocessor,
        one_wrong_side_sample(
            target, patch_two.first, patch_two.second));
    require(chain.owner_class == NormalizedClass::UniqueForeign
                && chain.owner_dof != target
                && fixture.cloud.dofs[
                       static_cast<std::size_t>(chain.owner_dof)]
                       .patch_id == 2,
            "smooth compatibility is local and non-transitive");
}

void test_optimized_no_foreign_candidate()
{
    Fixture fixture;
    RestrictOwnerGeometryPreprocessor3D preprocessor(
        fixture.surface, fixture.cloud, 3.0 / 16.0,
        Mode::OptimizedIntersection);
    const int target = center_dof(fixture.cloud, 0);
    const auto result = one_result(
        preprocessor,
        one_wrong_side_sample(
            target, {2.0, 2.0, 2.0}, {2.1, 2.0, 2.0}));
    require(result.owner_dof == target
                && result.owner_class == NormalizedClass::Target
                && result.query_path == QueryPath::SegmentTargetOnly,
            "no foreign AABB candidate certifies the target");
    require(preprocessor.diagnostics().optimized_intersection_calls == 0,
            "target-only AABB decision does not run a root solver");
}

void require_matches_reference_owner(
    const kfbim::app3d::RestrictOwnerPreprocessResult3D& optimized,
    const kfbim::app3d::RestrictOwnerPreprocessResult3D& reference,
    const std::string& message)
{
    require(optimized.owner_dof == reference.owner_dof
                && optimized.owner_class == reference.owner_class,
            message
                + ": optimized owner="
                + std::to_string(optimized.owner_dof)
                + " class=" + std::to_string(
                    static_cast<int>(optimized.owner_class))
                + ", reference owner=" + std::to_string(reference.owner_dof)
                + " class=" + std::to_string(
                    static_cast<int>(reference.owner_class)));
}

void require_matches_reference_target_side(
    const kfbim::app3d::RestrictOwnerPreprocessResult3D& actual,
    const kfbim::app3d::RestrictOwnerPreprocessResult3D& reference,
    const std::string& message)
{
    const bool actual_is_foreign =
        actual.owner_class == NormalizedClass::UniqueForeign;
    const bool reference_is_foreign =
        reference.owner_class == NormalizedClass::UniqueForeign;
    require(actual.owner_dof == reference.owner_dof
                && actual_is_foreign == reference_is_foreign,
            message
                + ": actual owner="
                + std::to_string(actual.owner_dof)
                + " is_foreign=" + std::to_string(actual_is_foreign)
                + ", reference owner="
                + std::to_string(reference.owner_dof)
                + " is_foreign="
                + std::to_string(reference_is_foreign));
}

void test_optimized_local_and_unique_foreign_paths()
{
    Fixture fixture;
    RestrictOwnerGeometryPreprocessor3D optimized(
        fixture.surface, fixture.cloud, 3.0 / 16.0,
        Mode::OptimizedIntersection);
    RestrictOwnerGeometryPreprocessor3D reference(
        fixture.surface, fixture.cloud, 3.0 / 16.0,
        Mode::FullIntersectionReference);
    const int target = center_dof(fixture.cloud, 0);

    const auto same = transverse_segment(fixture.surface, 0);
    const auto same_result = one_result(
        optimized,
        one_wrong_side_sample(target, same.first, same.second));
    require(same_result.owner_dof == target
                && same_result.query_path == QueryPath::SegmentTargetOnly,
            "target-patch AABB candidates remain target-owned");

    const auto& smooth = fixture.cloud.patches[0].smooth_patch_ids;
    const auto g1 = std::find_if(
        smooth.begin(), smooth.end(),
        [](int patch) { return patch != 0; });
    require(g1 != smooth.end(), "optimized fixture has a local G1 patch");
    const auto g1_segment = transverse_segment(fixture.surface, *g1);
    const auto g1_result = one_result(
        optimized,
        one_wrong_side_sample(
            target, g1_segment.first, g1_segment.second));
    require(g1_result.owner_dof == target
                && g1_result.query_path == QueryPath::SegmentTargetOnly,
            "explicit local G1 AABB candidates remain target-owned");

    const int foreign_target = center_dof(fixture.cloud, 6);
    const auto foreign_segment = transverse_segment(fixture.surface, 7);
    const auto foreign_sample = one_wrong_side_sample(
        foreign_target, foreign_segment.first, foreign_segment.second);
    const auto oracle = one_result(reference, foreign_sample);
    const auto foreign = one_result(optimized, foreign_sample);
    require_matches_reference_owner(
        foreign, oracle, "filtered unique foreign root matches the oracle");
    require(foreign.query_path == QueryPath::OptimizedIntersection
                && foreign.owner_class == NormalizedClass::UniqueForeign
                && foreign.foreign_crossing.has_value(),
            "unique foreign non-G1 root uses optimized intersection");

    fixture.cloud.patches[0].smooth_patch_ids = {0, 1};
    fixture.cloud.patches[1].smooth_patch_ids = {0, 1, 2};
    fixture.cloud.patches[2].smooth_patch_ids = {1, 2};
    RestrictOwnerGeometryPreprocessor3D chain_optimized(
        fixture.surface, fixture.cloud, 3.0 / 16.0,
        Mode::OptimizedIntersection);
    const auto patch_two = transverse_segment(fixture.surface, 2);
    const auto chain = one_result(
        chain_optimized,
        one_wrong_side_sample(
            target, patch_two.first, patch_two.second));
    require(chain.owner_class == NormalizedClass::UniqueForeign
                && fixture.cloud.dofs[
                       static_cast<std::size_t>(chain.owner_dof)]
                       .patch_id == 2,
            "optimized compatibility does not take a transitive G1 closure");
    const auto& diagnostics = optimized.diagnostics();
    std::uint64_t path_sum = 0;
    for (std::uint64_t count : diagnostics.path_counts)
        path_sum += count;
    std::uint64_t fallback_sum = 0;
    for (std::uint64_t count : diagnostics.fallback_counts)
        fallback_sum += count;
    require(path_sum == diagnostics.wrong_side_queries
                && fallback_sum == diagnostics.wrong_side_queries
                && diagnostics.wrong_side_queries == 3
                && diagnostics.compatible_aabb_candidates > 0
                && diagnostics.foreign_aabb_candidates > 0,
            "optimized aggregate diagnostics classify each query once");
}

void test_optimized_same_leaf_roots_and_unresolved_fallback()
{
    const NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    const SurfaceDofCloud3D cloud =
        make_native_surface_dofs_3d(surface, 1.0);
    const Eigen::Vector3d center(0.06, -0.05, 0.0);
    const Eigen::Vector3d start =
        center + Eigen::Vector3d(0.0, 0.70, 0.10);
    const Eigen::Vector3d end =
        center + Eigen::Vector3d(0.70, 0.0, 0.10);
    const kfbim::geometry3d::NurbsSurfaceIntersector3D intersector(
        surface.geometry_model());
    const auto intersection = intersector.intersect_segment(start, end);
    require(intersection.crossings.size() == 2
                && intersection.crossings[0].patch_index
                       == intersection.crossings[1].patch_index,
            "hollow-cylinder chord has two roots on one Bezier leaf");
    const int crossing_patch = intersection.crossings.front().patch_index;
    int target_patch = -1;
    for (int patch = 0; patch < static_cast<int>(cloud.patches.size());
         ++patch) {
        const auto& local = cloud.patches[
            static_cast<std::size_t>(patch)].smooth_patch_ids;
        if (patch != crossing_patch
            && std::find(local.begin(), local.end(), crossing_patch)
                   == local.end()) {
            target_patch = patch;
            break;
        }
    }
    require(target_patch >= 0,
            "same-leaf fixture has a foreign target patch");
    const int target = center_dof(cloud, target_patch);
    const auto sample = one_wrong_side_sample(
        target, start, end);
    RestrictOwnerGeometryPreprocessor3D reference(
        surface, cloud, 1.0, Mode::FullIntersectionReference);
    RestrictOwnerGeometryPreprocessor3D optimized(
        surface, cloud, 1.0, Mode::OptimizedIntersection);
    const auto oracle = one_result(reference, sample);
    const auto result = one_result(optimized, sample);
    require_matches_reference_owner(
        result, oracle, "same-leaf multiple roots match the oracle");
    require(result.owner_dof == target
                && result.owner_class == NormalizedClass::FailClosedTarget
                && result.query_path == QueryPath::OptimizedIntersection
                && result.fallback_cause
                       == FallbackCause::MultipleCrossings,
            "two independent roots on one Bezier leaf stop early");

    const SurfaceDofCloud3D fine_cloud =
        make_native_surface_dofs_3d(surface, 3.0 / 16.0);
    const Eigen::Vector3d direction =
        Eigen::Vector3d(
            1.0, 0.3713906763541037, 0.6947465906068658)
            .normalized();
    Eigen::Vector2d planar(direction.x(), direction.y());
    planar.normalize();
    const Eigen::Vector3d radius(
        -0.55 * planar.y(), 0.55 * planar.x(), 0.0);
    const Eigen::Vector3d unresolved_start =
        center + radius - 0.2 * direction;
    const auto unresolved_candidates =
        intersector.conservative_segment_candidates(
            unresolved_start, unresolved_start + direction);
    require(!unresolved_candidates.empty(),
            "unresolved fixture has conservative AABB candidates");
    int unresolved_target_patch = -1;
    for (int patch = 0;
         patch < static_cast<int>(fine_cloud.patches.size()); ++patch) {
        const auto& local = fine_cloud.patches[
            static_cast<std::size_t>(patch)].smooth_patch_ids;
        if (std::any_of(
                unresolved_candidates.begin(), unresolved_candidates.end(),
                [&](const auto& candidate) {
                    return candidate.patch_index() != patch
                        && std::find(
                               local.begin(), local.end(),
                               candidate.patch_index()) == local.end();
                })) {
            unresolved_target_patch = patch;
            break;
        }
    }
    require(unresolved_target_patch >= 0,
            "unresolved fixture has a foreign target patch");
    const int unresolved_target =
        center_dof(fine_cloud, unresolved_target_patch);
    const auto unresolved_sample = one_wrong_side_sample(
        unresolved_target, unresolved_start,
        unresolved_start + direction);
    RestrictOwnerGeometryPreprocessor3D unresolved_reference(
        surface, fine_cloud, 3.0 / 16.0,
        Mode::FullIntersectionReference);
    RestrictOwnerGeometryPreprocessor3D unresolved_optimized(
        surface, fine_cloud, 3.0 / 16.0,
        Mode::OptimizedIntersection);
    const auto unresolved_oracle =
        one_result(unresolved_reference, unresolved_sample);
    const auto unresolved =
        one_result(unresolved_optimized, unresolved_sample);
    require_matches_reference_owner(
        unresolved, unresolved_oracle,
        "unresolved filtered work matches the full oracle");
    require(unresolved.query_path == QueryPath::FullIntersectionFallback
                && unresolved.fallback_cause == FallbackCause::Unresolved,
            "unresolved filtered work falls back to the full oracle");
}

int foreign_target_dof_for_segment(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& end)
{
    const kfbim::geometry3d::NurbsSurfaceIntersector3D intersector(
        surface.geometry_model());
    const auto candidates =
        intersector.conservative_segment_candidates(start, end);
    for (int patch = 0; patch < static_cast<int>(cloud.patches.size());
         ++patch) {
        const auto& local = cloud.patches[
            static_cast<std::size_t>(patch)].smooth_patch_ids;
        if (std::any_of(
                candidates.begin(), candidates.end(),
                [&](const auto& candidate) {
                    return candidate.patch_index() != patch
                        && std::find(
                               local.begin(), local.end(),
                               candidate.patch_index()) == local.end();
                })) {
            return center_dof(cloud, patch);
        }
    }
    throw std::runtime_error(
        "pathology fixture has no foreign target patch");
}

void test_optimized_uncertain_geometry_uses_full_fallback()
{
    const NativeNurbsSurface3D cylinder =
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    const SurfaceDofCloud3D cloud =
        make_native_surface_dofs_3d(cylinder, 3.0 / 16.0);
    const Eigen::Vector3d center(0.06, -0.05, 0.0);
    const double c = 0.55 * std::sqrt(0.5);
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> segments{
        {center + Eigen::Vector3d(c, c, -0.2),
         center + Eigen::Vector3d(c, c, 0.2)},
        {{0.61, -0.10, 0.0}, {0.61, 0.0, 0.0}},
        {{0.63, -0.05, 0.69}, {0.59, -0.05, 0.65}}};

    const int endpoint_patch = 0;
    const auto& patch =
        cylinder.patches[static_cast<std::size_t>(endpoint_patch)];
    const auto endpoint_data = patch.evaluate_with_derivatives(
        0.5 * (patch.domain_start_u() + patch.domain_end_u()),
        0.5 * (patch.domain_start_v() + patch.domain_end_v()));
    const Eigen::Vector3d endpoint_normal =
        endpoint_data.du.cross(endpoint_data.dv).normalized();
    segments.push_back({
        endpoint_data.point,
        endpoint_data.point + 0.04 * endpoint_normal});

    for (std::size_t index = 0; index < segments.size(); ++index) {
        const auto& segment = segments[index];
        const int target = foreign_target_dof_for_segment(
            cylinder, cloud, segment.first, segment.second);
        const auto sample = one_wrong_side_sample(
            target, segment.first, segment.second);
        RestrictOwnerGeometryPreprocessor3D reference(
            cylinder, cloud, 3.0 / 16.0,
            Mode::FullIntersectionReference);
        RestrictOwnerGeometryPreprocessor3D optimized(
            cylinder, cloud, 3.0 / 16.0,
            Mode::OptimizedIntersection);
        const auto oracle = one_result(reference, sample);
        const auto result = one_result(optimized, sample);
        require_matches_reference_owner(
            result, oracle,
            "uncertain optimized geometry matches full fallback");
        require(result.query_path == QueryPath::FullIntersectionFallback
                    && result.fallback_cause != FallbackCause::None,
                "uncertain optimized geometry records a fallback cause");
        const auto& diagnostics = optimized.diagnostics();
        require(diagnostics.wrong_side_queries == 1
                    && diagnostics.optimized_intersection_calls == 1
                    && diagnostics.full_fallback_calls == 1
                    && diagnostics.path_counts[
                           static_cast<std::size_t>(
                               QueryPath::FullIntersectionFallback)] == 1
                    && diagnostics.fallback_counts[
                           static_cast<std::size_t>(
                               result.fallback_cause)] == 1,
                "full fallback and its single cause are counted once");
        if (index + 1 == segments.size()) {
            require(result.fallback_cause == FallbackCause::Endpoint,
                    "segment endpoint contact has its exact cause");
        }
    }
}

void test_full_reference_pathologies_fail_closed()
{
    const NativeNurbsSurface3D cylinder =
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    const SurfaceDofCloud3D cloud =
        make_native_surface_dofs_3d(cylinder, 3.0 / 16.0);
    RestrictOwnerGeometryPreprocessor3D preprocessor(
        cylinder, cloud, 3.0 / 16.0,
        Mode::FullIntersectionReference);
    const int target = center_dof(cloud, 0);
    const Eigen::Vector3d center(0.06, -0.05, 0.0);
    const double c = 0.55 * std::sqrt(0.5);

    for (const auto& segment :
         std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>{
             {center + Eigen::Vector3d(c, c, -0.2),
              center + Eigen::Vector3d(c, c, 0.2)},
             {{0.61, -0.10, 0.0}, {0.61, 0.0, 0.0}},
             {{0.63, -0.05, 0.69}, {0.59, -0.05, 0.65}}}) {
        require_full_result(
            one_result(
                preprocessor,
                one_wrong_side_sample(
                    target, segment.first, segment.second)),
            target, NormalizedClass::FailClosedTarget,
            "overlap, tangent, and feature contact fail closed");
    }

    const Eigen::Vector3d direction =
        Eigen::Vector3d(
            1.0, 0.3713906763541037, 0.6947465906068658)
            .normalized();
    Eigen::Vector2d planar(direction.x(), direction.y());
    planar.normalize();
    const Eigen::Vector3d radius(
        -0.55 * planar.y(), 0.55 * planar.x(), 0.0);
    const Eigen::Vector3d start =
        center + radius - 0.2 * direction;
    require_full_result(
        one_result(
            preprocessor,
            one_wrong_side_sample(target, start, start + direction)),
        target, NormalizedClass::FailClosedTarget,
        "unresolved geometry fails closed");

    const NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    const SurfaceDofCloud3D torus_cloud =
        make_native_surface_dofs_3d(torus, 3.0 / 16.0);
    RestrictOwnerGeometryPreprocessor3D torus_preprocessor(
        torus, torus_cloud, 3.0 / 16.0,
        Mode::FullIntersectionReference);
    const int torus_target = center_dof(torus_cloud, 0);
    require_full_result(
        one_result(
            torus_preprocessor,
            one_wrong_side_sample(
                torus_target,
                {-1.0, -0.04, 0.03}, {1.0, -0.04, 0.03})),
        torus_target, NormalizedClass::FailClosedTarget,
        "multiple crossings fail closed");
}

NativeNurbsSurface3D duplicate_surface(
    const NativeNurbsSurface3D& source)
{
    NativeNurbsSurface3D result = source;
    const int patch_offset = static_cast<int>(source.patches.size());
    const int component_offset =
        1 + *std::max_element(
                source.patch_components.begin(),
                source.patch_components.end());
    result.patches.insert(
        result.patches.end(), source.patches.begin(), source.patches.end());
    for (const std::string& name : source.patch_names)
        result.patch_names.push_back(name + "_duplicate");
    for (const auto& neighbors : source.smooth_neighbors) {
        auto shifted = neighbors;
        for (auto& neighbor : shifted) {
            if (neighbor)
                neighbor->patch += patch_offset;
        }
        result.smooth_neighbors.push_back(std::move(shifted));
    }
    for (const auto& neighbors : source.topological_patch_neighbors) {
        auto shifted = neighbors;
        for (int& patch : shifted)
            patch += patch_offset;
        result.topological_patch_neighbors.push_back(std::move(shifted));
    }
    for (auto connection : source.geometric_connections) {
        connection.first.patch += patch_offset;
        connection.second.patch += patch_offset;
        result.geometric_connections.push_back(std::move(connection));
    }
    for (int component : source.patch_components)
        result.patch_components.push_back(component + component_offset);
    return result;
}

void test_unrelated_coincidence_fails_closed()
{
    const NativeNurbsSurface3D surface = duplicate_surface(
        make_native_nurbs_surface_3d(GeometryKind3D::Torus));
    const SurfaceDofCloud3D cloud =
        make_native_surface_dofs_3d(surface, 3.0 / 16.0);
    RestrictOwnerGeometryPreprocessor3D preprocessor(
        surface, cloud, 3.0 / 16.0,
        Mode::FullIntersectionReference);
    const int target = center_dof(cloud, 0);
    const auto result = one_result(
        preprocessor,
        one_wrong_side_sample(
            target,
            {0.80, -0.04, 0.03}, {0.84, -0.04, 0.03}));
    require_full_result(
        result, target, NormalizedClass::FailClosedTarget,
        "unrelated coincidence fails closed");
    require(result.fallback_cause == FallbackCause::Coincidence,
            "coincidence has its exact fallback cause");
}

void test_optimized_multi_root_seam_and_coincidence_paths()
{
    const NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    const SurfaceDofCloud3D cloud =
        make_native_surface_dofs_3d(torus, 3.0 / 16.0);
    const int target = center_dof(cloud, 0);
    const auto two_leaf_sample = one_wrong_side_sample(
        target, {-1.0, -0.04, 0.03}, {1.0, -0.04, 0.03});
    RestrictOwnerGeometryPreprocessor3D two_leaf_reference(
        torus, cloud, 3.0 / 16.0, Mode::FullIntersectionReference);
    RestrictOwnerGeometryPreprocessor3D two_leaf_optimized(
        torus, cloud, 3.0 / 16.0, Mode::OptimizedIntersection);
    const auto two_leaf_oracle =
        one_result(two_leaf_reference, two_leaf_sample);
    const auto two_leaf =
        one_result(two_leaf_optimized, two_leaf_sample);
    require_matches_reference_owner(
        two_leaf, two_leaf_oracle,
        "different-leaf multiple roots match the oracle");
    require(two_leaf.owner_dof == target
                && two_leaf.owner_class
                       == NormalizedClass::FailClosedTarget
                && two_leaf.query_path == QueryPath::OptimizedIntersection
                && two_leaf.fallback_cause
                       == FallbackCause::MultipleCrossings,
            "two independent roots on different leaves stop early");
    require(two_leaf_optimized.diagnostics()
                .optimized_intersection_calls == 1
                && two_leaf_optimized.diagnostics()
                       .full_fallback_calls == 0,
            "independent-root cap does not run the full fallback");

    const Eigen::Vector3d seam_start(0.80, -0.04, 0.03);
    const Eigen::Vector3d seam_end(0.84, -0.04, 0.03);
    const kfbim::geometry3d::NurbsSurfaceIntersector3D seam_oracle(
        torus.geometry_model());
    const auto seam_intersection =
        seam_oracle.intersect_segment(seam_start, seam_end);
    require(seam_intersection.crossings.size() == 1
                && seam_intersection.diagnostics.seam_deduplications > 0,
            "periodic seam fixture has one canonical root");
    const int seam_patch = seam_intersection.crossings.front().patch_index;
    int seam_target_patch = -1;
    for (int patch = 0; patch < static_cast<int>(cloud.patches.size());
         ++patch) {
        const auto& local = cloud.patches[
            static_cast<std::size_t>(patch)].smooth_patch_ids;
        if (patch != seam_patch
            && std::find(local.begin(), local.end(), seam_patch)
                   == local.end()) {
            seam_target_patch = patch;
            break;
        }
    }
    require(seam_target_patch >= 0,
            "periodic seam fixture has a foreign target patch");
    const auto seam_sample = one_wrong_side_sample(
        center_dof(cloud, seam_target_patch), seam_start, seam_end);
    RestrictOwnerGeometryPreprocessor3D seam_reference(
        torus, cloud, 3.0 / 16.0, Mode::FullIntersectionReference);
    RestrictOwnerGeometryPreprocessor3D seam_optimized(
        torus, cloud, 3.0 / 16.0, Mode::OptimizedIntersection);
    const auto seam_reference_result =
        one_result(seam_reference, seam_sample);
    const auto seam_result = one_result(seam_optimized, seam_sample);
    require_matches_reference_owner(
        seam_result, seam_reference_result,
        "periodic seam fallback matches the oracle");
    require(seam_result.query_path == QueryPath::FullIntersectionFallback
                && seam_result.fallback_cause == FallbackCause::Seam,
            "periodic seam invokes full fallback");

    const NativeNurbsSurface3D coincident = duplicate_surface(torus);
    const SurfaceDofCloud3D coincident_cloud =
        make_native_surface_dofs_3d(coincident, 3.0 / 16.0);
    require(coincident.geometry_model().num_components() > 1,
            "coincidence fixture contains multiple components");
    const auto coincidence_sample = one_wrong_side_sample(
        center_dof(coincident_cloud, 0), seam_start, seam_end);
    RestrictOwnerGeometryPreprocessor3D coincidence_optimized(
        coincident, coincident_cloud, 3.0 / 16.0,
        Mode::OptimizedIntersection);
    const auto coincidence =
        one_result(coincidence_optimized, coincidence_sample);
    require(coincidence.owner_class
                    == NormalizedClass::FailClosedTarget
                && coincidence.query_path
                       == QueryPath::FullIntersectionFallback
                && coincidence.fallback_cause
                       == FallbackCause::Coincidence,
            "unrelated coincident components use full fallback");
}

void test_hybrid_sweep_target_path()
{
    Fixture fixture;
    const int target = center_dof(fixture.cloud, 0);
    const auto local = transverse_segment(fixture.surface, 0);
    RestrictOwnerSampleInput3D input;
    input.target_dof = target;
    input.query = local.first;
    input.support_points.fill(local.first);
    input.support_points[7] = local.second;
    input.support_points[11] = local.second;
    input.wrong_side[7] = true;
    input.wrong_side[11] = true;

    RestrictOwnerGeometryPreprocessor3D hybrid(
        fixture.surface, fixture.cloud, 3.0 / 16.0,
        Mode::RegionClosestHybrid);
    const auto result = hybrid.preprocess_sample(input);
    require(result.nodes[7].has_value()
                && result.nodes[11].has_value()
                && result.nodes[7]->owner_dof == target
                && result.nodes[11]->owner_dof == target
                && result.nodes[7]->query_path
                       == QueryPath::SweepTargetOnly
                && result.nodes[11]->query_path
                       == QueryPath::SweepTargetOnly,
            "compatible sample sweep classifies every wrong-side slot");
    require(hybrid.diagnostics().path_counts[
                static_cast<std::size_t>(
                    QueryPath::SweepTargetOnly)] == 2,
            "sample sweep records one path per wrong-side slot");
}

struct ThreeModeOwnerResults {
    kfbim::app3d::RestrictOwnerPreprocessResult3D reference;
    kfbim::app3d::RestrictOwnerPreprocessResult3D optimized;
    kfbim::app3d::RestrictOwnerPreprocessResult3D hybrid;
};

ThreeModeOwnerResults evaluate_three_modes(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double spacing,
    const RestrictOwnerSampleInput3D& sample)
{
    RestrictOwnerGeometryPreprocessor3D reference(
        surface, cloud, spacing, Mode::FullIntersectionReference);
    RestrictOwnerGeometryPreprocessor3D optimized(
        surface, cloud, spacing, Mode::OptimizedIntersection);
    RestrictOwnerGeometryPreprocessor3D hybrid(
        surface, cloud, spacing, Mode::RegionClosestHybrid);
    ThreeModeOwnerResults result{
        one_result(reference, sample),
        one_result(optimized, sample),
        one_result(hybrid, sample)};
    require_matches_reference_target_side(
        result.optimized, result.reference,
        "optimized policy agrees with the reference owner");
    require_matches_reference_target_side(
        result.hybrid, result.reference,
        "hybrid policy agrees with the reference owner");
    return result;
}

struct ClosestMissFixture {
    int target_dof = -1;
    Eigen::Vector3d start = Eigen::Vector3d::Zero();
    Eigen::Vector3d end = Eigen::Vector3d::Zero();
};

ClosestMissFixture find_closest_miss_fixture(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double spacing)
{
    kfbim::geometry3d::NurbsSurfaceIntersectorOptions3D options;
    options.maximum_element_extent = 2.0 * spacing;
    options.local_max_subdivision_depth = 4;
    const kfbim::geometry3d::NurbsSurfaceIntersector3D intersector(
        surface.geometry_model(), options);
    const Eigen::Vector3d center(0.06, -0.05, 0.0);
    constexpr double pi = 3.14159265358979323846;
    for (double radius : {0.56, 0.57, 0.58, 0.59}) {
        for (double degrees : {15.0, 30.0, 45.0, 60.0, 75.0}) {
            const double angle = degrees * pi / 180.0;
            const Eigen::Vector3d point(
                center.x() + radius * std::cos(angle),
                center.y() + radius * std::sin(angle), 0.0);
            const Eigen::Vector3d start =
                point + Eigen::Vector3d(0.0, 0.0, -0.08);
            const Eigen::Vector3d end =
                point + Eigen::Vector3d(0.0, 0.0, 0.08);
            const auto candidates =
                intersector.conservative_segment_candidates(start, end);
            if (candidates.empty())
                continue;
            bool all_miss = true;
            int closest_attempts = 0;
            for (const auto& candidate : candidates) {
                const auto certificate =
                    intersector.certify_candidate_segment(
                        candidate, start, end);
                all_miss = all_miss
                    && certificate.kind == CertificateKind::CertifiedMiss;
                closest_attempts +=
                    certificate.diagnostics.closest_point_attempts;
            }
            if (!all_miss || closest_attempts == 0)
                continue;
            for (int patch = 0;
                 patch < static_cast<int>(cloud.patches.size()); ++patch) {
                const auto& smooth = cloud.patches[
                    static_cast<std::size_t>(patch)].smooth_patch_ids;
                const bool all_foreign = std::all_of(
                    candidates.begin(), candidates.end(),
                    [&](const auto& candidate) {
                        return candidate.patch_index() != patch
                            && std::find(
                                   smooth.begin(), smooth.end(),
                                   candidate.patch_index()) == smooth.end();
                    });
                if (all_foreign) {
                    return {
                        center_dof(cloud, patch), start, end};
                }
            }
        }
    }
    throw std::runtime_error(
        "failed to find a deterministic closest-certified miss fixture");
}

void test_hybrid_segment_closest_miss_and_root_paths()
{
    Fixture fixture;
    const int local_target = center_dof(fixture.cloud, 0);
    const auto local = transverse_segment(fixture.surface, 0);
    auto segment_input =
        one_wrong_side_sample(local_target, local.first, local.second);
    segment_input.support_points[0] =
        transverse_segment(fixture.surface, 7).second;
    const auto segment_results = evaluate_three_modes(
        fixture.surface, fixture.cloud, 3.0 / 16.0, segment_input);
    require(segment_results.hybrid.query_path
                == QueryPath::SegmentTargetOnly,
            "failed sweep with no foreign segment leaf uses segment target");

    const NativeNurbsSurface3D cylinder =
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    const double spacing = 3.0 / 16.0;
    const SurfaceDofCloud3D cloud =
        make_native_surface_dofs_3d(cylinder, spacing);
    const ClosestMissFixture miss =
        find_closest_miss_fixture(cylinder, cloud, spacing);
    const auto miss_results = evaluate_three_modes(
        cylinder, cloud, spacing,
        one_wrong_side_sample(
            miss.target_dof, miss.start, miss.end));
    require(miss_results.hybrid.query_path
                    == QueryPath::ClosestCertifiedMiss
                && miss_results.hybrid.owner_class
                       == NormalizedClass::Target
                && miss_results.hybrid.owner_dof == miss.target_dof,
            "closest terminal separation certifies an AABB false positive");

    const auto root_segment =
        transverse_segment_at(cylinder, 7, 0.37, 0.43);
    const int target = foreign_target_dof_for_segment(
        cylinder, cloud, root_segment.first, root_segment.second);
    const auto root_results = evaluate_three_modes(
        cylinder, cloud, spacing,
        one_wrong_side_sample(
            target, root_segment.first, root_segment.second));
    require(root_results.hybrid.query_path
                    == QueryPath::ClosestCertifiedRoot
                && root_results.hybrid.owner_class
                       == NormalizedClass::UniqueForeign
                && root_results.hybrid.foreign_crossing.has_value(),
            "one safe interior foreign root uses the closest certificate: "
                + std::to_string(static_cast<int>(
                    root_results.hybrid.query_path))
                + ", class=" + std::to_string(static_cast<int>(
                    root_results.hybrid.owner_class)));
}

void test_hybrid_hollow_grid_stencil_matches_reference()
{
    const NativeNurbsSurface3D cylinder =
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    const double spacing = 3.0 / 16.0;
    const SurfaceDofCloud3D cloud =
        make_native_surface_dofs_3d(cylinder, spacing);
    const int target = 19;
    require(target < static_cast<int>(cloud.dofs.size()),
            "hollow-grid regression target DOF exists");

    const Eigen::Vector3d query(
        0.26329319959113251,
        0.15329319959113238,
        0.39142857142857140);
    const Eigen::Vector3d support(0.5625, -0.1875, 0.75);

    const RestrictOwnerSampleInput3D sample =
        one_wrong_side_sample(target, query, support);
    RestrictOwnerGeometryPreprocessor3D reference(
        cylinder, cloud, spacing, Mode::FullIntersectionReference);
    RestrictOwnerGeometryPreprocessor3D optimized(
        cylinder, cloud, spacing, Mode::OptimizedIntersection);
    RestrictOwnerGeometryPreprocessor3D hybrid(
        cylinder, cloud, spacing, Mode::RegionClosestHybrid);
    const auto oracle = one_result(reference, sample);
    const auto optimized_result = one_result(optimized, sample);
    const auto hybrid_result = one_result(hybrid, sample);

    require(oracle.owner_dof == target
                && oracle.owner_class == NormalizedClass::FailClosedTarget,
            "hollow-grid oracle conservatively keeps the target owner");
    require_matches_reference_owner(
        optimized_result, oracle,
        "hollow-grid optimized path matches the oracle");
    require_matches_reference_owner(
        hybrid_result, oracle,
        "hollow-grid hybrid certificate matches the oracle");
    require(hybrid_result.query_path == optimized_result.query_path
                && hybrid_result.fallback_cause
                       == optimized_result.fallback_cause
                && hybrid_result.foreign_crossing.has_value()
                       == optimized_result.foreign_crossing.has_value(),
            "hollow-grid hybrid preserves the optimized confirmation "
            "classification");
    if (hybrid_result.foreign_crossing.has_value()) {
        require(hybrid_result.foreign_crossing->patch_index
                    == optimized_result.foreign_crossing->patch_index,
                "hollow-grid hybrid preserves the optimized crossing "
                "patch");
    }

    const auto& hybrid_diagnostics = hybrid.diagnostics();
    require(hybrid_diagnostics.optimized_intersection_calls == 1,
            "hollow-grid closest root is confirmed exactly once by the "
            "optimized policy");
    require(hybrid_diagnostics.path_counts[
                static_cast<std::size_t>(
                    hybrid_result.query_path)] == 1,
            "hollow-grid hybrid records its final normalized path");
    const bool used_full_fallback =
        hybrid_result.query_path
        == QueryPath::FullIntersectionFallback;
    require(hybrid_diagnostics.full_fallback_calls
                == (used_full_fallback ? 1u : 0u),
            "hollow-grid hybrid fallback count matches its final path");
}

bool certificate_has_safe_root(
    const kfbim::geometry3d::NurbsSurfaceCandidateCertificate3D& certificate)
{
    return certificate.kind
               == CertificateKind::CertifiedUniqueTransverseRoot
        && certificate.crossing.has_value()
        && certificate.segment_parameter_strictly_interior
        && certificate.element_parameter_strictly_interior
        && certificate.patch_parameter_strictly_interior
        && !certificate.crossing->feature_edge_contact
        && certificate.crossing->transversality
               > certificate.crossing
                     ->reliable_transversality_tolerance;
}

struct DangerousCompatibleRootFixture {
    int target_patch = -1;
    Eigen::Vector3d start = Eigen::Vector3d::Zero();
    Eigen::Vector3d end = Eigen::Vector3d::Zero();
};

DangerousCompatibleRootFixture find_dangerous_compatible_root_fixture(
    const NativeNurbsSurface3D& surface,
    SurfaceDofCloud3D& cloud,
    double spacing)
{
    kfbim::geometry3d::NurbsSurfaceIntersectorOptions3D options;
    options.maximum_element_extent = 2.0 * spacing;
    options.local_max_subdivision_depth = 4;
    const kfbim::geometry3d::NurbsSurfaceIntersector3D intersector(
        surface.geometry_model(), options);
    const std::array<std::pair<double, double>, 5> parameters{{
        {0.0, 0.37}, {1.0, 0.43}, {0.37, 0.0},
        {0.43, 1.0}, {0.5, 0.5}}};

    for (int patch_index = 0;
         patch_index < static_cast<int>(surface.patches.size());
         ++patch_index) {
        const auto& patch = surface.patches[
            static_cast<std::size_t>(patch_index)];
        for (const auto& parameter : parameters) {
            const double u = patch.domain_start_u()
                + parameter.first
                    * (patch.domain_end_u() - patch.domain_start_u());
            const double v = patch.domain_start_v()
                + parameter.second
                    * (patch.domain_end_v() - patch.domain_start_v());
            const auto data = patch.evaluate_with_derivatives(u, v);
            const Eigen::Vector3d normal =
                data.du.cross(data.dv).normalized();
            const std::array<
                std::pair<Eigen::Vector3d, Eigen::Vector3d>, 2>
                segments{{
                    {data.point, data.point + 0.40 * normal},
                    {data.point - 0.20 * normal,
                     data.point + 0.40 * normal}}};
            for (const auto& segment : segments) {
                const auto candidates =
                    intersector.conservative_segment_candidates(
                        segment.first, segment.second);
                std::vector<int> root_patches;
                bool unsafe_root = false;
                bool unresolved = false;
                for (const auto& candidate : candidates) {
                    const auto certificate =
                        intersector.certify_candidate_segment(
                            candidate, segment.first, segment.second);
                    if (certificate.kind == CertificateKind::Unresolved) {
                        unresolved = true;
                        break;
                    }
                    if (certificate.kind
                        == CertificateKind::CertifiedUniqueTransverseRoot) {
                        root_patches.push_back(candidate.patch_index());
                        unsafe_root = unsafe_root
                            || !certificate_has_safe_root(certificate);
                    }
                }
                if (unresolved || root_patches.empty() || !unsafe_root)
                    continue;
                std::sort(root_patches.begin(), root_patches.end());
                root_patches.erase(
                    std::unique(root_patches.begin(), root_patches.end()),
                    root_patches.end());
                const bool has_foreign_miss = std::any_of(
                    candidates.begin(), candidates.end(),
                    [&](const auto& candidate) {
                        return std::find(
                                   root_patches.begin(), root_patches.end(),
                                   candidate.patch_index())
                            == root_patches.end();
                    });
                if (!has_foreign_miss)
                    continue;
                const int target_patch = root_patches.front();
                cloud.patches[static_cast<std::size_t>(target_patch)]
                    .smooth_patch_ids = root_patches;
                return {
                    target_patch, segment.first, segment.second};
            }
        }
    }
    throw std::runtime_error(
        "failed to find compatible unsafe-root plus foreign-miss fixture");
}

void test_hybrid_compatible_unsafe_root_uses_optimized_fallback()
{
    const NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    const double spacing = 3.0 / 16.0;
    SurfaceDofCloud3D cloud =
        make_native_surface_dofs_3d(surface, spacing);
    const DangerousCompatibleRootFixture fixture =
        find_dangerous_compatible_root_fixture(surface, cloud, spacing);

    kfbim::geometry3d::NurbsSurfaceIntersectorOptions3D options;
    options.maximum_element_extent = 2.0 * spacing;
    options.local_max_subdivision_depth = 4;
    const kfbim::geometry3d::NurbsSurfaceIntersector3D intersector(
        surface.geometry_model(), options);
    const auto candidates = intersector.conservative_segment_candidates(
        fixture.start, fixture.end);
    bool saw_compatible_unsafe_root = false;
    bool saw_foreign_miss = false;
    const auto& compatible = cloud.patches[
        static_cast<std::size_t>(fixture.target_patch)].smooth_patch_ids;
    for (const auto& candidate : candidates) {
        const auto certificate = intersector.certify_candidate_segment(
            candidate, fixture.start, fixture.end);
        const bool local = candidate.patch_index() == fixture.target_patch
            || std::find(
                   compatible.begin(), compatible.end(),
                   candidate.patch_index()) != compatible.end();
        if (local) {
            saw_compatible_unsafe_root = saw_compatible_unsafe_root
                || (certificate.kind
                        == CertificateKind::CertifiedUniqueTransverseRoot
                    && !certificate_has_safe_root(certificate));
        } else {
            require(certificate.kind == CertificateKind::CertifiedMiss,
                    "dangerous-root fixture has only certified foreign misses");
            saw_foreign_miss = true;
        }
    }
    require(saw_compatible_unsafe_root && saw_foreign_miss,
            "dangerous-root fixture validates both certificate partitions");

    const auto results = evaluate_three_modes(
        surface, cloud, spacing,
        one_wrong_side_sample(
            center_dof(cloud, fixture.target_patch),
            fixture.start, fixture.end));
    require(results.hybrid.query_path == results.optimized.query_path
                && results.hybrid.query_path
                       != QueryPath::ClosestCertifiedMiss
                && results.hybrid.query_path
                       != QueryPath::SegmentTargetOnly,
            "compatible unsafe root delegates to optimized intersection");
}

void test_hybrid_unsafe_geometry_uses_optimized_fallback()
{
    const NativeNurbsSurface3D cylinder =
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    const SurfaceDofCloud3D cloud =
        make_native_surface_dofs_3d(cylinder, 3.0 / 16.0);
    const Eigen::Vector3d center(0.06, -0.05, 0.0);
    const double c = 0.55 * std::sqrt(0.5);
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> segments{
        {center + Eigen::Vector3d(c, c, -0.2),
         center + Eigen::Vector3d(c, c, 0.2)},
        {{0.61, -0.10, 0.0}, {0.61, 0.0, 0.0}},
        {{0.63, -0.05, 0.69}, {0.59, -0.05, 0.65}}};
    const auto& patch = cylinder.patches[0];
    const auto endpoint = patch.evaluate_with_derivatives(
        0.5 * (patch.domain_start_u() + patch.domain_end_u()),
        0.5 * (patch.domain_start_v() + patch.domain_end_v()));
    const Eigen::Vector3d normal =
        endpoint.du.cross(endpoint.dv).normalized();
    segments.push_back(
        {endpoint.point, endpoint.point + 0.04 * normal});
    for (const auto& segment : segments) {
        const int target = foreign_target_dof_for_segment(
            cylinder, cloud, segment.first, segment.second);
        const auto results = evaluate_three_modes(
            cylinder, cloud, 3.0 / 16.0,
            one_wrong_side_sample(
                target, segment.first, segment.second));
        require(results.hybrid.query_path
                    == results.optimized.query_path
                    && results.hybrid.query_path
                           != QueryPath::ClosestCertifiedMiss
                    && results.hybrid.query_path
                           != QueryPath::ClosestCertifiedRoot,
                "unsafe closest state delegates to optimized intersection");
    }
}

void test_hybrid_unresolved_boundary_seam_and_multi_fallbacks()
{
    const NativeNurbsSurface3D cylinder =
        make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
    const double spacing = 3.0 / 16.0;
    const SurfaceDofCloud3D cylinder_cloud =
        make_native_surface_dofs_3d(cylinder, spacing);
    const Eigen::Vector3d center(0.06, -0.05, 0.0);
    const Eigen::Vector3d direction =
        Eigen::Vector3d(
            1.0, 0.3713906763541037, 0.6947465906068658)
            .normalized();
    Eigen::Vector2d planar(direction.x(), direction.y());
    planar.normalize();
    const Eigen::Vector3d radius(
        -0.55 * planar.y(), 0.55 * planar.x(), 0.0);
    const Eigen::Vector3d unresolved_start =
        center + radius - 0.2 * direction;
    const Eigen::Vector3d unresolved_end =
        unresolved_start + direction;
    const NativeNurbsSurface3D unresolved_surface =
        duplicate_surface(cylinder);
    const SurfaceDofCloud3D unresolved_cloud =
        make_native_surface_dofs_3d(unresolved_surface, spacing);

    kfbim::geometry3d::NurbsSurfaceIntersectorOptions3D unresolved_options;
    unresolved_options.maximum_element_extent = 2.0 * spacing;
    unresolved_options.local_max_subdivision_depth = 4;
    const kfbim::geometry3d::NurbsSurfaceIntersector3D
        unresolved_intersector(
            unresolved_surface.geometry_model(), unresolved_options);
    const auto unresolved_candidates =
        unresolved_intersector.conservative_segment_candidates(
            unresolved_start, unresolved_end);
    std::vector<std::pair<int, CertificateKind>>
        unresolved_certificates;
    unresolved_certificates.reserve(unresolved_candidates.size());
    for (const auto& candidate : unresolved_candidates) {
        const auto certificate =
            unresolved_intersector.certify_candidate_segment(
                candidate, unresolved_start, unresolved_end);
        unresolved_certificates.push_back(
            {candidate.patch_index(), certificate.kind});
    }
    int unresolved_target_patch = -1;
    int foreign_unresolved_count = -1;
    for (int patch = 0;
         patch < static_cast<int>(unresolved_cloud.patches.size()); ++patch) {
        const auto& compatible = unresolved_cloud.patches[
            static_cast<std::size_t>(patch)].smooth_patch_ids;
        const int count = static_cast<int>(std::count_if(
            unresolved_certificates.begin(),
            unresolved_certificates.end(),
            [&](const auto& certificate) {
                const bool local = certificate.first == patch
                    || std::find(
                           compatible.begin(), compatible.end(),
                           certificate.first) != compatible.end();
                return !local
                    && certificate.second == CertificateKind::Unresolved;
            }));
        if (count > foreign_unresolved_count) {
            unresolved_target_patch = patch;
            foreign_unresolved_count = count;
        }
    }
    require(unresolved_target_patch >= 0 && foreign_unresolved_count >= 2,
            "unresolved fixture has at least two foreign unresolved leaves: "
                + std::to_string(foreign_unresolved_count)
                + " foreign, " + std::to_string(static_cast<int>(
                    std::count_if(
                        unresolved_certificates.begin(),
                        unresolved_certificates.end(), [](const auto& c) {
                            return c.second == CertificateKind::Unresolved;
                        }))) + " total");
    const int unresolved_target =
        center_dof(unresolved_cloud, unresolved_target_patch);

    const auto unresolved = evaluate_three_modes(
        unresolved_surface, unresolved_cloud, spacing,
        one_wrong_side_sample(
            unresolved_target, unresolved_start, unresolved_end));
    require(unresolved.hybrid.query_path
                == unresolved.optimized.query_path,
            "unresolved closest work delegates to optimized intersection");

    const auto element_boundary = transverse_segment(cylinder, 7);
    const int boundary_target = foreign_target_dof_for_segment(
        cylinder, cylinder_cloud,
        element_boundary.first, element_boundary.second);
    const auto boundary = evaluate_three_modes(
        cylinder, cylinder_cloud, spacing,
        one_wrong_side_sample(
            boundary_target,
            element_boundary.first, element_boundary.second));
    require(boundary.hybrid.query_path
                == boundary.optimized.query_path
                && boundary.hybrid.query_path
                       != QueryPath::ClosestCertifiedRoot,
            "element-boundary root delegates to optimized intersection");

    const NativeNurbsSurface3D torus =
        make_native_nurbs_surface_3d(GeometryKind3D::Torus);
    const SurfaceDofCloud3D torus_cloud =
        make_native_surface_dofs_3d(torus, spacing);
    const auto multiple = evaluate_three_modes(
        torus, torus_cloud, spacing,
        one_wrong_side_sample(
            center_dof(torus_cloud, 0),
            {-1.0, -0.04, 0.03}, {1.0, -0.04, 0.03}));
    require(multiple.hybrid.query_path
                == multiple.optimized.query_path
                && multiple.hybrid.owner_class
                       == NormalizedClass::FailClosedTarget,
            "multi-root closest state delegates to optimized intersection");

    const Eigen::Vector3d seam_start(0.80, -0.04, 0.03);
    const Eigen::Vector3d seam_end(0.84, -0.04, 0.03);
    const int seam_target = foreign_target_dof_for_segment(
        torus, torus_cloud, seam_start, seam_end);
    const auto seam = evaluate_three_modes(
        torus, torus_cloud, spacing,
        one_wrong_side_sample(
            seam_target, seam_start, seam_end));
    require(seam.hybrid.query_path
                == seam.optimized.query_path
                && seam.hybrid.query_path
                       == QueryPath::FullIntersectionFallback,
            "periodic seam delegates to optimized intersection");

    const NativeNurbsSurface3D coincident = duplicate_surface(torus);
    const SurfaceDofCloud3D coincident_cloud =
        make_native_surface_dofs_3d(coincident, spacing);
    const auto coincidence = evaluate_three_modes(
        coincident, coincident_cloud, spacing,
        one_wrong_side_sample(
            center_dof(coincident_cloud, 0),
            seam_start, seam_end));
    require(coincidence.hybrid.query_path
                == coincidence.optimized.query_path
                && coincidence.hybrid.fallback_cause
                       == FallbackCause::Coincidence,
            "unrelated coincidence delegates to optimized intersection");

    Fixture chain_fixture;
    chain_fixture.cloud.patches[0].smooth_patch_ids = {0, 1};
    chain_fixture.cloud.patches[1].smooth_patch_ids = {0, 1, 2};
    chain_fixture.cloud.patches[2].smooth_patch_ids = {1, 2};
    const auto chain_segment =
        transverse_segment_at(chain_fixture.surface, 2, 0.37, 0.43);
    const auto chain = evaluate_three_modes(
        chain_fixture.surface, chain_fixture.cloud, spacing,
        one_wrong_side_sample(
            center_dof(chain_fixture.cloud, 0),
            chain_segment.first, chain_segment.second));
    require(chain.hybrid.owner_class == NormalizedClass::UniqueForeign
                && chain_fixture.cloud.dofs[
                       static_cast<std::size_t>(chain.hybrid.owner_dof)]
                       .patch_id == 2,
            "hybrid compatibility does not take a transitive G1 closure");
}

void test_hybrid_partial_connection_and_stage_timing()
{
    Fixture fixture;
    const int partial_target = center_dof(fixture.cloud, 6);
    const auto partial_sample = one_wrong_side_sample(
        partial_target, {0.07, -0.67, -0.61},
        {0.07, -0.67, -0.65});
    const auto partial_results = evaluate_three_modes(
        fixture.surface, fixture.cloud, 3.0 / 16.0, partial_sample);
    require(partial_results.hybrid.query_path
                != QueryPath::SweepTargetOnly,
            "partial one-to-many connection cannot certify a local sweep");

    const auto root_segment =
        transverse_segment_at(fixture.surface, 7, 0.37, 0.43);
    const auto root_sample = one_wrong_side_sample(
        center_dof(fixture.cloud, 6),
        root_segment.first, root_segment.second);
    RestrictOwnerGeometryPreprocessor3D ordinary(
        fixture.surface, fixture.cloud, 3.0 / 16.0,
        Mode::RegionClosestHybrid);
    (void)one_result(ordinary, partial_sample);
    (void)one_result(ordinary, root_sample);
    const auto& ordinary_diagnostics = ordinary.diagnostics();
    require(ordinary_diagnostics.region_seconds == 0.0
                && ordinary_diagnostics.closest_point_seconds == 0.0
                && ordinary_diagnostics.optimized_intersection_seconds
                       == 0.0
                && ordinary_diagnostics.full_fallback_seconds == 0.0,
            "ordinary policy execution performs no per-query clock reads");

    RestrictOwnerPreprocessOptions3D timing;
    timing.collect_stage_timings = true;
    RestrictOwnerGeometryPreprocessor3D timed(
        fixture.surface, fixture.cloud, 3.0 / 16.0,
        Mode::RegionClosestHybrid, timing);
    (void)one_result(timed, partial_sample);
    (void)one_result(timed, root_sample);
    const auto& timed_diagnostics = timed.diagnostics();
    require(timed_diagnostics.region_seconds > 0.0
                && timed_diagnostics.closest_point_seconds > 0.0
                && timed_diagnostics.optimized_intersection_seconds > 0.0
                && timed_diagnostics.full_fallback_seconds > 0.0,
            "instrumentation replay records all hybrid geometry stages");
}

void test_validation_slots_and_diagnostics()
{
    const Fixture fixture;
    require_throws_contains(
        [&] {
            (void)RestrictOwnerGeometryPreprocessor3D(
                fixture.surface, fixture.cloud, 0.0,
                Mode::FullIntersectionReference);
        },
        "positive finite grid spacing",
        "nonpositive spacing rejected");
    require_throws_contains(
        [&] {
            (void)RestrictOwnerGeometryPreprocessor3D(
                fixture.surface, fixture.cloud,
                std::numeric_limits<double>::infinity(),
                Mode::FullIntersectionReference);
        },
        "positive finite grid spacing",
        "nonfinite spacing rejected");
    RestrictOwnerPreprocessOptions3D options;
    options.maximum_independent_crossings = 1;
    require_throws_contains(
        [&] {
            (void)RestrictOwnerGeometryPreprocessor3D(
                fixture.surface, fixture.cloud, 3.0 / 16.0,
                Mode::FullIntersectionReference, options);
        },
        "exactly two",
        "invalid crossing cap rejected");

    RestrictOwnerGeometryPreprocessor3D preprocessor(
        fixture.surface, fixture.cloud, 3.0 / 16.0,
        Mode::FullIntersectionReference);
    auto input = one_wrong_side_sample(
        center_dof(fixture.cloud, 0),
        {2.0, 2.0, 2.0}, {2.1, 2.0, 2.0});
    input.wrong_side[11] = true;
    input.support_points[11] = {2.2, 2.0, 2.0};
    const auto result = preprocessor.preprocess_sample(input);
    require(result.nodes[7].has_value()
                && result.nodes[11].has_value()
                && std::count_if(
                       result.nodes.begin(), result.nodes.end(),
                       [](const auto& node) {
                           return node.has_value();
                       }) == 2,
            "64-slot result classifies only wrong-side nodes");
    const auto& diagnostics = preprocessor.diagnostics();
    require(diagnostics.wrong_side_queries == 2
                && diagnostics.path_counts[
                       static_cast<std::size_t>(
                           QueryPath::FullIntersection)] == 2
                && diagnostics.target_decisions == 2,
            "diagnostics count every classification exactly once");
    preprocessor.reset_diagnostics();
    require(preprocessor.diagnostics().wrong_side_queries == 0
                && std::all_of(
                    preprocessor.diagnostics().path_counts.begin(),
                    preprocessor.diagnostics().path_counts.end(),
                    [](std::uint64_t count) { return count == 0; }),
            "diagnostics reset zeros the complete aggregate");

    auto bad_target = input;
    bad_target.target_dof = static_cast<int>(fixture.cloud.dofs.size());
    require_throws_contains(
        [&] { (void)preprocessor.preprocess_sample(bad_target); },
        "target DOF", "invalid target rejected");
    auto nonfinite = input;
    nonfinite.support_points[0].x() =
        std::numeric_limits<double>::quiet_NaN();
    require_throws_contains(
        [&] { (void)preprocessor.preprocess_sample(nonfinite); },
        "finite", "nonfinite sample point rejected");
    auto zero = input;
    zero.support_points[7] = zero.query;
    require_throws_contains(
        [&] { (void)preprocessor.preprocess_sample(zero); },
        "nonzero length", "zero-length wrong-side segment rejected");
}
} // namespace

int main()
{
    try {
        test_mode_parsing();
        test_full_reference_and_local_topology();
        test_optimized_no_foreign_candidate();
        test_optimized_local_and_unique_foreign_paths();
        test_optimized_same_leaf_roots_and_unresolved_fallback();
        test_optimized_uncertain_geometry_uses_full_fallback();
        test_full_reference_pathologies_fail_closed();
        test_unrelated_coincidence_fails_closed();
        test_optimized_multi_root_seam_and_coincidence_paths();
        test_hybrid_sweep_target_path();
        test_hybrid_segment_closest_miss_and_root_paths();
        test_hybrid_hollow_grid_stencil_matches_reference();
        test_hybrid_unsafe_geometry_uses_optimized_fallback();
        test_hybrid_unresolved_boundary_seam_and_multi_fallbacks();
        test_hybrid_compatible_unsafe_root_uses_optimized_fallback();
        test_hybrid_partial_connection_and_stage_timing();
        test_validation_slots_and_diagnostics();
        std::cout
            << "restrict-owner geometry preprocessor tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "restrict-owner geometry preprocessor test failure: "
            << error.what() << '\n';
        return 1;
    }
}
