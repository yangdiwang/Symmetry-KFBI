#include "src/support/trace/closest_point_trace_extension_3d.hpp"
#include "src/support/trace/tensor_product_cover_restrict_3d.hpp"
#include "src/support/cauchy/direct_coefficient_cauchy_3d.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace kfbim::app3d;
using namespace kfbim::geometry3d;

void require(bool value, const std::string& message)
{
    if (!value) throw std::runtime_error(message);
}
void near(double actual, double expected, double tolerance, const std::string& message)
{
    require(std::isfinite(actual) && std::isfinite(expected)
                && std::abs(actual - expected) <= tolerance, message);
}

void diagnose_anchor(const NurbsSurfaceClosestPointResult3D& projection,
                     const ClosestPointTraceAnchor3D& anchor)
{
    if (anchor.accepted()) return;
    std::cerr << "anchor refusal=" << closest_point_trace_anchor_status_name_3d(anchor.status)
              << " projection_status=" << static_cast<int>(projection.status)
              << " localized=" << projection.localized
              << " radius=" << projection.localization_radius_upper
              << " ties=" << projection.separated_near_ties
              << " candidates=" << projection.candidates.size()
              << " boxes=" << projection.stats.boxes_visited << '\n';
    for (const auto& candidate : projection.candidates)
        std::cerr << " candidate patch=" << candidate.patch_index
                  << " uv=" << candidate.u << ',' << candidate.v
                  << " point=" << candidate.point.transpose()
                  << " distance=" << candidate.distance_upper << '\n';
}

struct HarmonicQuadratic {
    double constant;
    Eigen::Vector3d linear;
    Eigen::Matrix3d hessian;
    double value(const Eigen::Vector3d& x) const
    {
        return constant + linear.dot(x) + 0.5 * x.dot(hessian * x);
    }
    Eigen::Vector3d gradient(const Eigen::Vector3d& x) const
    {
        return linear + hessian * x;
    }
};

// The expected fields are independent ambient polynomials, never evaluated
// through the trace-extension helper or through its Cauchy reconstruction.
HarmonicQuadratic jump_field()
{
    Eigen::Matrix3d h;
    h << 2.0, 0.7, -0.4, 0.7, -3.0, 0.3, -0.4, 0.3, 1.0;
    return {0.83, {0.6, -0.9, 0.4}, h};
}
HarmonicQuadratic exterior_field()
{
    Eigen::Matrix3d h;
    h << -0.4, 0.2, 0.5, 0.2, 0.6, -0.1, 0.5, -0.1, -0.2;
    return {-0.19, {0.3, 0.8, -0.6}, h};
}

double extend_jump(const NativeNurbsDensitySpace3D& density,
                   const ClosestPointTraceAnchor3D& anchor,
                   const Eigen::Vector3d& node)
{
    const auto& owner = anchor.owner;
    const auto& patch = density.surface().patches[owner.nurbs_patch_index];
    const double u = (owner.nurbs_parameter.x() - patch.domain_start_u())
                   / (patch.domain_end_u() - patch.domain_start_u());
    const double v = (owner.nurbs_parameter.y() - patch.domain_start_v())
                   / (patch.domain_end_v() - patch.domain_start_v());
    const auto geometry = native_surface_parameter_jet_3d(density, owner.nurbs_patch_index, u, v);
    const auto frame = make_local_orthonormal_frame_3d(geometry.normal, geometry.x_u);
    const auto curvature = tangent_graph_hessian_from_parameter_jet_3d(geometry, frame);
    const auto field = jump_field();
    const auto gradient = field.gradient(geometry.point);
    const auto j0 = known_dirichlet_jet_from_ambient_derivatives_3d(
        {field.value(geometry.point), gradient, field.hessian}, geometry, frame);
    // Derivatives of J1 = n.grad(J) include the changing surface normal.
    NormalJet3D j1;
    j1 << gradient.dot(frame.normal),
        frame.normal.dot(field.hessian * frame.tangent1)
            - curvature.h11 * gradient.dot(frame.tangent1)
            - curvature.h12 * gradient.dot(frame.tangent2),
        frame.normal.dot(field.hessian * frame.tangent2)
            - curvature.h12 * gradient.dot(frame.tangent1)
            - curvature.h22 * gradient.dot(frame.tangent2);
    const auto weights = cauchy_polynomial_weights_3d(
        geometry.point, frame, curvature, node);
    return weights.apply_value_jet(j0) + weights.apply_normal_jet(j1);
}

void test_failed_support_query(const NativeNurbsSurface3D& surface,
                               const NurbsSurfaceClosestPointIndex3D& index)
{
    const Eigen::Vector3d a(0.5625, -0.28125, -0.375);
    const Eigen::Vector3d p(0.53226561756481816, -0.33189570139702573, -0.305);
    const auto n = surface.patches[3].normal(0.65, 0.25);
    NurbsSurfaceClosestPointWorkspace3D workspace;
    NurbsSurfaceClosestPointOptions3D options;
    const auto projection = index.query(a, options, workspace);
    const auto anchor = select_closest_point_trace_anchor_3d(
        surface, projection, a, p, 3, n, 3.0 * (3.0 / 32.0));
    diagnose_anchor(projection, anchor);
    require(anchor.accepted(), "former support-query fixture has a usable closest anchor");
    require(anchor.owner.nurbs_patch_index == 3, "closest anchor stays on outer wall q3");
    const double radius = std::hypot(a.x() - 0.06, a.y() + 0.05);
    const Eigen::Vector3d exact(0.06 + 0.55 * (a.x() - 0.06) / radius,
                               -0.05 + 0.55 * (a.y() + 0.05) / radius, a.z());
    require((anchor.owner.crossing_point - exact).norm() <= 2e-6,
            "closest anchor agrees with analytic cylinder projection");
    require((anchor.owner.crossing_point - p).norm() > 0.01,
            "closest anchor is not the old trace endpoint");
    near(anchor.distance, radius - 0.55, 2e-8, "closest distance to outer cylinder");

    auto invalid = projection;
    invalid.localized = false;
    require(!select_closest_point_trace_anchor_3d(surface, invalid, a, p, 3, n, 0.3).accepted(),
            "unlocalized projection is refused");
    invalid = projection;
    invalid.separated_near_ties = true;
    require(!select_closest_point_trace_anchor_3d(surface, invalid, a, p, 3, n, 0.3).accepted(),
            "separated near-minima are refused");
    require(!select_closest_point_trace_anchor_3d(surface, projection, a, p, 3, n, 1e-4).accepted(),
            "distance budget is enforced");
    const auto inner_p = surface.patches[7].evaluate(0.65, 0.25);
    const auto inner_n = surface.patches[7].normal(0.65, 0.25);
    require(!select_closest_point_trace_anchor_3d(surface, projection, a, inner_p, 7, inner_n, 0.5).accepted(),
            "same solid component cannot substitute outer wall for inner wall");
    const Eigen::Vector3d slanted_a = a + 0.03 * Eigen::Vector3d::UnitZ();
    require(!select_closest_point_trace_anchor_3d(surface, projection, slanted_a, p, 3, n, 0.3).accepted(),
            "a projection from another query cannot authorize a non-normal extension");
}

void test_manufactured_cover(const NativeNurbsSurface3D& surface,
                             const NurbsSurfaceClosestPointIndex3D& index,
                             const NativeNurbsDensitySpace3D& density,
                             int target_patch, double u, double v)
{
    constexpr int cells = 64;
    constexpr double h = 3.0 / cells;
    const kfbim::CartesianGrid3D grid({-1.5, -1.5, -1.5}, {h, h, h},
                                     {cells, cells, cells}, kfbim::DofLayout3D::Node);
    const auto p = surface.patches[target_patch].evaluate(u, v);
    const auto normal = surface.patches[target_patch].normal(u, v);
    const auto stencil = build_tensor_product_cover_restrict_stencil_3d(
        grid, p, normal, TensorProductCoverKind3D::Q27Cover3);
    NurbsSurfaceClosestPointWorkspace3D workspace;
    NurbsSurfaceClosestPointOptions3D options;
    Eigen::VectorXd interior(27), exterior(27);
    int inside_nodes = 0, outside_nodes = 0;
    const auto base = exterior_field();
    const auto jump = jump_field();
    for (int slot = 0; slot < 27; ++slot) {
        const auto xyz = grid.coord(stencil.grid_ids[slot]);
        const Eigen::Vector3d a(xyz[0], xyz[1], xyz[2]);
        const bool node_inside = surface.exact_inside(a);
        node_inside ? ++inside_nodes : ++outside_nodes;
        const auto projection = index.query(a, options, workspace);
        const auto anchor = select_closest_point_trace_anchor_3d(
            surface, projection, a, p, target_patch, normal, 3.0 * h);
        diagnose_anchor(projection, anchor);
        require(anchor.accepted(), "smooth Q27 support node has a compatible closest anchor");
        const double extension = extend_jump(density, anchor, a);
        near(extension, jump.value(a), 2e-10, "curved Cauchy extension reproduces harmonic P2 at A");
        const double physical = base.value(a) + (node_inside ? jump.value(a) : 0.0);
        for (bool desired_inside : {false, true}) {
            const int sign = static_cast<int>(desired_inside) - static_cast<int>(node_inside);
            const double continued = physical + sign * extension;
            near(continued, base.value(a) + (desired_inside ? jump.value(a) : 0.0),
                 2e-10, "positive/negative/zero branch conversion reproduces target field");
            (desired_inside ? interior : exterior)[slot] = continued;
        }
    }
    require(inside_nodes > 0 && outside_nodes > 0, "manufactured cover really straddles the interface");
    for (bool desired_inside : {false, true}) {
        const auto& values = desired_inside ? interior : exterior;
        near(stencil.value_weights.dot(values), base.value(p) + (desired_inside ? jump.value(p) : 0.0),
             1e-9, "Q27 target-side value reproduction");
        const Eigen::Vector3d gradient = base.gradient(p)
            + (desired_inside ? jump.gradient(p) : Eigen::Vector3d::Zero());
        near(stencil.normal_weights.dot(values), normal.dot(gradient), 2e-8,
             "Q27 target-side outward-normal derivative reproduction");
    }
}

void test_feature_refusal(const NativeNurbsSurface3D& surface,
                          const NurbsSurfaceClosestPointIndex3D& index)
{
    NurbsSurfaceClosestPointWorkspace3D workspace;
    NurbsSurfaceClosestPointOptions3D options;
    const Eigen::Vector3d a(0.64, -0.05, 0.70);
    const auto p = surface.patches[0].evaluate(0.1, 0.95);
    const auto n = surface.patches[0].normal(0.1, 0.95);
    const auto projection = index.query(a, options, workspace);
    require(!select_closest_point_trace_anchor_3d(surface, projection, a, p, 0, n, 0.3).accepted(),
            "C0 outer rim projection does not invent a unique smooth extension");
}

void test_internal_c0_knot_refusal()
{
    // A single degree-two patch represents z=max(0,x-0.5). The repeated
    // interior u knot is a real crease, absent from patch-connection metadata.
    using Basis = kfbim::geometry::NurbsBasis1D;
    std::vector<std::vector<Eigen::Vector3d>> controls;
    for (int i = 0; i < 5; ++i) {
        const double x = 0.25 * i;
        const double z = std::max(0.0, x - 0.5);
        controls.push_back({{x, 0.0, z}, {x, 1.0, z}});
    }
    NativeNurbsSurface3D surface;
    surface.patches.emplace_back(Basis(2, {0, 0, 0, 0.5, 0.5, 1, 1, 1}),
                                 Basis(1, {0, 0, 1, 1}), controls,
                                 std::vector<std::vector<double>>(5, std::vector<double>(2, 1.0)));
    surface.patch_components = {0};
    const auto& patch = surface.patches.front();
    require(patch.normal(0.49, 0.5).dot(patch.normal(0.51, 0.5)) < 0.8,
            "repeated-knot fixture must have a real normal discontinuity");
    const NurbsSurfaceClosestPointIndex3D index(surface.geometry_model());
    NurbsSurfaceClosestPointWorkspace3D workspace;
    NurbsSurfaceClosestPointOptions3D options;
    const auto p = patch.evaluate(0.4, 0.5);
    const auto normal = patch.normal(0.4, 0.5);
    for (const Eigen::Vector3d a : {Eigen::Vector3d(0.47, 0.5, 0.01),
                                    Eigen::Vector3d(0.5, 0.5, -0.01)}) {
        const auto projection = index.query(a, options, workspace);
        require(projection.status == NurbsSurfaceClosestPointStatus3D::Bounded
                    && projection.localized && projection.normal_valid,
                "real folded-patch projection must finish before testing feature refusal");
        const Eigen::Vector3d expected(a.x(), a.y(), 0.0);
        require((projection.point - expected).norm() < 1e-7,
                "folded patch query locates the known near-crease or crease minimum");
        const auto anchor = select_closest_point_trace_anchor_3d(
            surface, projection, a, p, 0, normal, 0.3);
        require(anchor.status == ClosestPointTraceAnchorStatus3D::FeatureContact,
                "an internal C0 knot conservatively refuses the entire owner/target patch");
    }
}
} // namespace

int main()
{
    try {
        test_internal_c0_knot_refusal();
        const auto surface = make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
        const NurbsSurfaceClosestPointIndex3D index(surface.geometry_model());
        NativeNurbsDensityOptions3D density_options;
        density_options.field = NativeDensityField3D::ValueTrace;
        density_options.reduction_backend = NativeDensityReductionBackend3D::BaseOnly;
        density_options.coefficients_per_direction = 5;
        const NativeNurbsDensitySpace3D density(surface, density_options);
        test_failed_support_query(surface, index);
        test_manufactured_cover(surface, index, density, 3, 0.65, 0.25);
        test_manufactured_cover(surface, index, density, 8, 0.35, 0.5);
        test_manufactured_cover(surface, index, density, 0, 0.0, 0.5);
        test_feature_refusal(surface, index);
        std::cout << "closest-point trace extension tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "closest-point trace extension test failure: " << error.what() << '\n';
        return 1;
    }
}
