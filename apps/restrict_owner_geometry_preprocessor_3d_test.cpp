#include "restrict_owner_geometry_preprocessor_3d.hpp"

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
        test_full_reference_pathologies_fail_closed();
        test_unrelated_coincidence_fails_closed();
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
