#include "crossing_owner_restrict_3d.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::NativeNurbsSurface3D;
using kfbim::app3d::RestrictOwnerDecisionKind3D;
using kfbim::app3d::SurfaceDofCloud3D;
using kfbim::app3d::make_native_nurbs_surface_3d;
using kfbim::app3d::make_native_surface_dofs_3d;
using kfbim::app3d::select_restrict_correction_owner_3d;
using kfbim::geometry3d::NurbsSurfaceCrossing3D;
using kfbim::geometry3d::NurbsSurfaceIntersectionResult3D;

void require(bool value, const std::string& message)
{
    if (!value)
        throw std::runtime_error(message);
}

template <class Function>
void require_invalid(Function&& function, const std::string& message)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

struct Fixture {
    NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    SurfaceDofCloud3D cloud =
        make_native_surface_dofs_3d(surface, 3.0 / 16.0);

    int target(int patch) const
    {
        const auto& p = cloud.patches.at(static_cast<std::size_t>(patch));
        return p.dof_index(p.nu / 2, p.nv / 2);
    }

    NurbsSurfaceCrossing3D root(int patch, double t = 0.5) const
    {
        const auto& p = surface.patches.at(static_cast<std::size_t>(patch));
        const double u = 0.5 * (p.domain_start_u() + p.domain_end_u());
        const double v = 0.5 * (p.domain_start_v() + p.domain_end_v());
        const auto d = p.evaluate_with_derivatives(u, v);
        return {patch, -1, u, v, t, d.point,
                d.du.cross(d.dv).normalized(), 1.0e-14, 1.0, false};
    }

    NurbsSurfaceIntersectionResult3D result(
        const NurbsSurfaceCrossing3D& root) const
    {
        NurbsSurfaceIntersectionResult3D value;
        value.crossings.push_back(root);
        return value;
    }

    static Eigen::Vector3d query(const NurbsSurfaceCrossing3D& root)
    {
        return root.point - 0.25 * root.normal;
    }

    static Eigen::Vector3d support(const NurbsSurfaceCrossing3D& root)
    {
        return root.point + 0.25 * root.normal;
    }
};

void test_same_and_g1_retain_target()
{
    const Fixture fixture;
    const int target = fixture.target(0);
    const auto same = fixture.root(0);
    const auto same_owner = select_restrict_correction_owner_3d(
        target, Fixture::query(same), Fixture::support(same),
        fixture.surface, fixture.cloud, fixture.result(same));
    require(
        same_owner.owner_dof == target
            && same_owner.kind
                == RestrictOwnerDecisionKind3D::TargetOrG1SingleCrossing,
        "same patch retains target");

    const auto& ids = fixture.cloud.patches[0].smooth_patch_ids;
    const auto found = std::find_if(
        ids.begin(), ids.end(), [](int patch) { return patch != 0; });
    require(found != ids.end(), "reference patch has a G1 neighbor");
    const auto g1 = fixture.root(*found);
    const auto g1_owner = select_restrict_correction_owner_3d(
        target, Fixture::query(g1), Fixture::support(g1),
        fixture.surface, fixture.cloud, fixture.result(g1));
    require(
        g1_owner.owner_dof == target
            && g1_owner.kind
                == RestrictOwnerDecisionKind3D::TargetOrG1SingleCrossing,
        "G1 patch retains target");
}

void test_foreign_non_g1_selects_crossing_patch()
{
    const Fixture fixture;
    const int target = fixture.target(6);
    const auto root = fixture.root(7);
    const auto decision = select_restrict_correction_owner_3d(
        target, Fixture::query(root), Fixture::support(root),
        fixture.surface, fixture.cloud, fixture.result(root));
    require(
        decision.kind
            == RestrictOwnerDecisionKind3D::ForeignNonG1SingleCrossing,
        "foreign non-G1 changes owner");
    require(
        decision.owner_dof >= 0
            && fixture.cloud.dofs.at(
                   static_cast<std::size_t>(decision.owner_dof)).patch_id == 7,
        "owner lies on foreign patch");
    require(
        decision.crossing_patch == 7
            && std::abs(decision.segment_parameter - root.edge_parameter)
                < 1.0e-14
            && std::abs(decision.residual - root.residual) < 1.0e-14
            && std::abs(decision.transversality - root.transversality)
                < 1.0e-14,
        "diagnostics copied");
}

void test_fallbacks_retain_target()
{
    const Fixture fixture;
    const int target = fixture.target(6);
    const auto root = fixture.root(7);
    const auto query = Fixture::query(root);
    const auto support = Fixture::support(root);

    const NurbsSurfaceIntersectionResult3D empty;
    const auto no_root = select_restrict_correction_owner_3d(
        target, query, support, fixture.surface, fixture.cloud, empty);
    require(
        no_root.owner_dof == target
            && no_root.kind
                == RestrictOwnerDecisionKind3D::NoCrossingFallback,
        "empty roots retain target");

    auto multiple = fixture.result(root);
    multiple.crossings.push_back(fixture.root(8, 0.75));
    const auto multi = select_restrict_correction_owner_3d(
        target, query, support, fixture.surface, fixture.cloud, multiple);
    require(
        multi.owner_dof == target
            && multi.kind
                == RestrictOwnerDecisionKind3D::MultipleCrossingFallback,
        "multiple roots retain target");

    auto overlap = fixture.result(root);
    overlap.overlap_detected = true;
    const auto overlap_owner = select_restrict_correction_owner_3d(
        target, query, support, fixture.surface, fixture.cloud, overlap);
    require(
        overlap_owner.owner_dof == target
            && overlap_owner.kind
                == RestrictOwnerDecisionKind3D::DegenerateCrossingFallback,
        "overlap retains target");

    auto unresolved = fixture.result(root);
    unresolved.diagnostics.unresolved_candidates = 1;
    const auto unresolved_owner = select_restrict_correction_owner_3d(
        target, query, support, fixture.surface, fixture.cloud, unresolved);
    require(
        unresolved_owner.owner_dof == target
            && unresolved_owner.kind
                == RestrictOwnerDecisionKind3D::AmbiguousEdgeFallback,
        "unresolved retains target");

    auto tangent = fixture.result(root);
    tangent.crossings.front().transversality = 0.0;
    const auto tangent_owner = select_restrict_correction_owner_3d(
        target, query, support, fixture.surface, fixture.cloud, tangent);
    require(
        tangent_owner.owner_dof == target
            && tangent_owner.kind
                == RestrictOwnerDecisionKind3D::DegenerateCrossingFallback,
        "tangent retains target");

    auto feature = fixture.result(root);
    feature.crossings.front().feature_edge_contact = true;
    const auto feature_owner = select_restrict_correction_owner_3d(
        target, query, support, fixture.surface, fixture.cloud, feature);
    require(
        feature_owner.owner_dof == target
            && feature_owner.kind
                == RestrictOwnerDecisionKind3D::AmbiguousEdgeFallback,
        "feature contact retains target");
}

void test_invalid_inputs_are_rejected()
{
    const Fixture fixture;
    const int target = fixture.target(6);
    const auto root = fixture.root(7);
    const auto result = fixture.result(root);
    const auto query = Fixture::query(root);
    const auto support = Fixture::support(root);

    require_invalid(
        [&] {
            (void)select_restrict_correction_owner_3d(
                -1, query, support, fixture.surface, fixture.cloud, result);
        },
        "invalid target rejected");

    auto bad_patch = result;
    bad_patch.crossings.front().patch_index = 999;
    require_invalid(
        [&] {
            (void)select_restrict_correction_owner_3d(
                target, query, support,
                fixture.surface, fixture.cloud, bad_patch);
        },
        "invalid patch rejected");

    auto nonfinite = result;
    nonfinite.crossings.front().u =
        std::numeric_limits<double>::quiet_NaN();
    require_invalid(
        [&] {
            (void)select_restrict_correction_owner_3d(
                target, query, support,
                fixture.surface, fixture.cloud, nonfinite);
        },
        "nonfinite root rejected");

    require_invalid(
        [&] {
            (void)select_restrict_correction_owner_3d(
                target, query, query,
                fixture.surface, fixture.cloud, result);
        },
        "zero segment rejected");
}
} // namespace

int main()
{
    try {
        test_same_and_g1_retain_target();
        test_foreign_non_g1_selects_crossing_patch();
        test_fallbacks_retain_target();
        test_invalid_inputs_are_rejected();
        std::cout << "crossing owner restrict 3D tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "crossing owner restrict 3D test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
