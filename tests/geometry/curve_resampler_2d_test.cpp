#include "src/geometry/curve_2d.hpp"
#include "src/geometry/curve_resampler_2d.hpp"
#include "src/geometry/grid_pair_2d.hpp"
#include "src/geometry/p2_curve_2d.hpp"
#include "src/grid/cartesian_grid_2d.hpp"
#include "src/local_cauchy/laplace_panel_solver_2d.hpp"

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using kfbim::CurveResampler2D;
using kfbim::CurvaturePanelMonitor2D;
using kfbim::ICurve2D;
using kfbim::Interface2D;
using kfbim::PanelNodeLayout2D;

constexpr double kPi = 3.141592653589793238462643383279502884;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void require_near(double actual,
                  double expected,
                  double tolerance,
                  const std::string& message)
{
    require(std::abs(actual - expected) <= tolerance, message);
}

void require_near(const Eigen::Vector2d& actual,
                  const Eigen::Vector2d& expected,
                  double tolerance,
                  const std::string& message)
{
    require((actual - expected).norm() <= tolerance, message);
}

template <class Function>
void require_invalid_argument(Function&& function,
                              const std::string& message)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

class SmoothFlowerCurve final : public ICurve2D {
public:
    Eigen::Vector2d eval(double t) const override
    {
        const double r = radius(t);
        return {r * std::cos(t), r * std::sin(t)};
    }

    Eigen::Vector2d deriv(double t) const override
    {
        const double r = radius(t);
        const double rp = radius_derivative(t);
        return {
            rp * std::cos(t) - r * std::sin(t),
            rp * std::sin(t) + r * std::cos(t)
        };
    }

    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }

private:
    static double radius(double theta)
    {
        return 0.88 * (1.0 + 0.18 * std::cos(5.0 * theta));
    }

    static double radius_derivative(double theta)
    {
        return -0.88 * 0.18 * 5.0 * std::sin(5.0 * theta);
    }
};

class RigidlyTransformedCurve final : public ICurve2D {
public:
    RigidlyTransformedCurve(const ICurve2D& curve,
                            Eigen::Matrix2d rotation,
                            Eigen::Vector2d translation)
        : curve_(curve)
        , rotation_(std::move(rotation))
        , translation_(std::move(translation))
    {
    }

    Eigen::Vector2d eval(double t) const override
    {
        return translation_ + rotation_ * curve_.eval(t);
    }

    Eigen::Vector2d deriv(double t) const override
    {
        return rotation_ * curve_.deriv(t);
    }

    double t_min() const override { return curve_.t_min(); }
    double t_max() const override { return curve_.t_max(); }

private:
    const ICurve2D& curve_;
    Eigen::Matrix2d rotation_;
    Eigen::Vector2d translation_;
};

class CrossingRegressionFlower7Curve final : public ICurve2D {
public:
    Eigen::Vector2d eval(double t) const override
    {
        const double r = radius(t);
        return center() + rotate({r * std::cos(t), r * std::sin(t)});
    }

    Eigen::Vector2d deriv(double t) const override
    {
        const double r = radius(t);
        const double dr = -0.68 * 0.10 * 7.0 * std::sin(7.0 * t);
        return rotate({dr * std::cos(t) - r * std::sin(t),
                       dr * std::sin(t) + r * std::cos(t)});
    }

    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }

private:
    static double radius(double t)
    {
        return 0.68 * (1.0 + 0.10 * std::cos(7.0 * t));
    }

    static Eigen::Vector2d center()
    {
        return {0.02, 0.04};
    }

    static Eigen::Vector2d rotate(const Eigen::Vector2d& point)
    {
        constexpr double angle = 0.31;
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        return {c * point[0] - s * point[1],
                s * point[0] + c * point[1]};
    }
};

Interface2D adaptive_flower(const ICurve2D& curve)
{
    return CurveResampler2D::
        discretize_quadratic_lagrange_curvature_adaptive(
            curve,
            5.0 / 32.0,
            1.6,
            0.8,
            CurvaturePanelMonitor2D::Additive);
}

Interface2D adaptive_flower(std::shared_ptr<const ICurve2D> curve)
{
    return CurveResampler2D::
        discretize_quadratic_lagrange_curvature_adaptive(
            std::move(curve),
            5.0 / 32.0,
            1.6,
            0.8,
            CurvaturePanelMonitor2D::Additive);
}

Eigen::Vector2d positional_p2_point(const Interface2D& iface,
                                    int panel,
                                    double local_s)
{
    double shape[3];
    kfbim::geometry2d::p2_shape(local_s, shape);
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    for (int local = 0; local < 3; ++local) {
        point += shape[local]
               * iface.points().row(
                   iface.point_index(panel, local)).transpose();
    }
    return point;
}

void test_retained_exact_geometry_and_const_reference_fallback()
{
    const SmoothFlowerCurve borrowed_curve;
    const Interface2D positional = adaptive_flower(borrowed_curve);
    require(!positional.has_panel_geometry(),
            "const-reference resampling unexpectedly retained geometry");
    for (int panel = 0; panel < positional.num_panels(); ++panel) {
        for (const double local_s : {-0.73, -0.19, 0.41, 0.88}) {
            require_near(
                kfbim::geometry2d::panel_point(
                    positional, panel, local_s),
                positional_p2_point(positional, panel, local_s),
                2.0e-14,
                "const-reference interface no longer uses positional P2 geometry");
        }
    }

    std::weak_ptr<SmoothFlowerCurve> retained_curve;
    {
        auto curve = std::make_shared<SmoothFlowerCurve>();
        retained_curve = curve;
        const Interface2D exact = adaptive_flower(std::move(curve));

        require(exact.has_panel_geometry(),
                "shared_ptr resampling did not retain exact panel geometry");
        require(exact.panel_geometry().num_panels() == exact.num_panels(),
                "retained geometry panel count does not match interface");
        require(!retained_curve.expired(),
                "interface did not retain the source curve lifetime");

        const Eigen::Vector2d probe =
            kfbim::geometry2d::panel_point(exact, 0, 0.37);
        require(probe.allFinite(),
                "retained geometry cannot be evaluated after owner move");
    }
    require(retained_curve.expired(),
            "retained source curve outlived its interface");
}

void test_exact_geometry_is_continuous_and_on_source_curve()
{
    const auto curve = std::make_shared<SmoothFlowerCurve>();
    const Interface2D iface = adaptive_flower(curve);
    require(iface.has_panel_geometry(),
            "exact-geometry continuity test has no geometry provider");

    double maximum_exact_to_p2_gap = 0.0;
    for (int panel = 0; panel < iface.num_panels(); ++panel) {
        const int next = (panel + 1) % iface.num_panels();
        const Eigen::Vector2d left_point =
            kfbim::geometry2d::panel_point(iface, panel, 1.0);
        const Eigen::Vector2d right_point =
            kfbim::geometry2d::panel_point(iface, next, -1.0);
        const Eigen::Vector2d left_tangent =
            kfbim::geometry2d::panel_tangent(iface, panel, 1.0).normalized();
        const Eigen::Vector2d right_tangent =
            kfbim::geometry2d::panel_tangent(iface, next, -1.0).normalized();
        const Eigen::Vector2d left_normal =
            kfbim::geometry2d::panel_normal(iface, panel, 1.0);
        const Eigen::Vector2d right_normal =
            kfbim::geometry2d::panel_normal(iface, next, -1.0);

        require_near(left_point, right_point, 2.0e-12,
                     "retained panel points are discontinuous at a shared endpoint");
        require_near(left_tangent, right_tangent, 2.0e-12,
                     "retained unit tangents are discontinuous at a shared endpoint");
        require_near(left_normal, right_normal, 2.0e-12,
                     "retained normals are discontinuous at a shared endpoint");

        for (const double local_s : {-0.83, -0.31, 0.17, 0.69}) {
            const Eigen::Vector2d point =
                kfbim::geometry2d::panel_point(iface, panel, local_s);
            const double theta = std::atan2(point.y(), point.x());
            const double expected_radius =
                0.88 * (1.0 + 0.18 * std::cos(5.0 * theta));
            require_near(point.norm(), expected_radius, 2.0e-12,
                         "retained panel point is not on the source flower");
            maximum_exact_to_p2_gap = std::max(
                maximum_exact_to_p2_gap,
                (point - positional_p2_point(
                             iface, panel, local_s)).norm());
        }

        for (int local = 0; local < 3; ++local) {
            require_near(
                kfbim::geometry2d::panel_point(
                    iface, panel, kfbim::geometry2d::kP2NodeS[local]),
                iface.points().row(
                    iface.point_index(panel, local)).transpose(),
                2.0e-12,
                "retained geometry does not interpolate its P2 DOF point");
        }
    }
    require(maximum_exact_to_p2_gap > 1.0e-10,
            "exact-geometry test did not distinguish provider geometry from positional P2");
}

void test_cauchy_geometry_cache_uses_exact_provider()
{
    const auto curve = std::make_shared<SmoothFlowerCurve>();
    const Interface2D iface = adaptive_flower(curve);
    const kfbim::PanelCenterCauchyGeometryCache2D cache =
        kfbim::build_panel_center_cauchy_geometry_cache_2d(iface);

    require(cache.num_panels == iface.num_panels(),
            "Cauchy cache panel count is inconsistent");
    require(cache.num_centers
                == kfbim::detail::kQuadraticPanelExpansionCount
                   * iface.num_panels(),
            "Cauchy cache center count is inconsistent");
    require(static_cast<int>(cache.points.size()) == cache.num_centers,
            "Cauchy cache point metadata count is inconsistent");

    double maximum_center_to_p2_gap = 0.0;
    for (int center = 0; center < cache.num_centers; ++center) {
        const int panel = cache.panel_index[center];
        const double local_s = cache.local_s[center];
        const auto& cached =
            cache.points[static_cast<std::size_t>(center)];

        require(cached.panel == panel,
                "Cauchy cache point has the wrong owner panel");
        require_near(cached.local_s, local_s, 0.0,
                     "Cauchy cache point has the wrong local coordinate");
        require_near(
            cache.centers.row(center).transpose(),
            kfbim::geometry2d::panel_point(iface, panel, local_s),
            2.0e-14,
            "Cauchy center bypassed the exact geometry provider");
        require_near(
            cache.normals.row(center).transpose(),
            kfbim::geometry2d::panel_normal(iface, panel, local_s),
            2.0e-14,
            "Cauchy center normal bypassed the exact geometry provider");
        maximum_center_to_p2_gap = std::max(
            maximum_center_to_p2_gap,
            (cache.centers.row(center).transpose()
             - positional_p2_point(iface, panel, local_s)).norm());

        for (int collocation = 0; collocation < 3; ++collocation) {
            const double collocation_s =
                local_s
                + (collocation - 1) * kfbim::detail::kCollocationDelta;
            require_near(
                cached.collocation_points[
                    static_cast<std::size_t>(collocation)],
                kfbim::geometry2d::panel_point(
                    iface, panel, collocation_s),
                2.0e-14,
                "Cauchy collocation point bypassed the exact geometry provider");
            require_near(
                cached.collocation_normals[
                    static_cast<std::size_t>(collocation)],
                kfbim::geometry2d::panel_normal(
                    iface, panel, collocation_s),
                2.0e-14,
                "Cauchy collocation normal bypassed the exact geometry provider");
        }
    }
    require(maximum_center_to_p2_gap > 1.0e-10,
            "Cauchy cache test did not distinguish exact geometry from positional P2");
}

void test_closed_p2_topology_and_positive_weights()
{
    const SmoothFlowerCurve curve;
    const Interface2D iface = adaptive_flower(curve);
    const int panels = iface.num_panels();

    require(panels > 2, "adaptive flower has too few panels");
    require(iface.num_points() == 2 * panels,
            "P2 interface must have two global points per closed panel");
    require(iface.points_per_panel() == 3,
            "P2 interface must have three local points per panel");
    require(iface.panel_node_layout()
                == PanelNodeLayout2D::QuadraticLagrange,
            "adaptive interface must use quadratic Lagrange layout");

    std::vector<int> incidence(
        static_cast<std::size_t>(iface.num_points()), 0);
    for (int panel = 0; panel < panels; ++panel) {
        require(iface.point_index(panel, 0) == panel,
                "P2 left endpoint numbering changed");
        require(iface.point_index(panel, 1) == panels + panel,
                "P2 midpoint numbering changed");
        require(iface.point_index(panel, 2)
                    == (panel + 1) % panels,
                "P2 right endpoint does not close cyclically");
        for (int local = 0; local < 3; ++local) {
            ++incidence[static_cast<std::size_t>(
                iface.point_index(panel, local))];
        }
    }
    require(iface.point_index(panels - 1, 2) == 0,
            "last P2 panel must share the first endpoint");
    for (int point = 0; point < panels; ++point) {
        require(incidence[static_cast<std::size_t>(point)] == 2,
                "closed P2 endpoint must belong to exactly two panels");
    }
    for (int point = panels; point < 2 * panels; ++point) {
        require(incidence[static_cast<std::size_t>(point)] == 1,
                "P2 midpoint must belong to exactly one panel");
    }

    for (int point = 0; point < iface.num_points(); ++point) {
        require(std::isfinite(iface.weights()[point])
                    && iface.weights()[point] > 0.0,
                "adaptive P2 quadrature weights must be finite and positive");
    }

    for (int panel = 0; panel < panels; ++panel) {
        const int previous = (panel + panels - 1) % panels;
        const double panel_length =
            1.5 * iface.weights()[panels + panel];
        const double previous_length =
            1.5 * iface.weights()[panels + previous];
        require_near(
            iface.weights()[panel],
            (previous_length + panel_length) / 6.0,
            2.0e-13,
            "shared endpoint weight is inconsistent with adjacent panels");
    }
}

void test_curvature_monitor_adds_panels()
{
    const SmoothFlowerCurve curve;
    constexpr double h = 5.0 / 32.0;
    const Interface2D uniform =
        CurveResampler2D::discretize_quadratic_lagrange(
            curve, h, 1.6);
    const Interface2D maximum =
        CurveResampler2D::
            discretize_quadratic_lagrange_curvature_adaptive(
                curve, h, 1.6, 0.8);
    const Interface2D additive =
        CurveResampler2D::
            discretize_quadratic_lagrange_curvature_adaptive(
                curve,
                h,
                1.6,
                0.8,
                CurvaturePanelMonitor2D::Additive);

    require(maximum.num_panels() > uniform.num_panels(),
            "maximum curvature monitor did not refine the coarse flower");
    require(additive.num_panels() >= maximum.num_panels(),
            "additive monitor must not use fewer panels than maximum monitor");
    require(additive.num_points() == 2 * additive.num_panels(),
            "adaptive refinement broke the P2 DOF count");
    require_near(additive.weights().sum(),
                 uniform.weights().sum(),
                 2.0e-11,
                 "adaptive and uniform interfaces cover different arc length");
}

void test_rigid_transform_invariance()
{
    const SmoothFlowerCurve curve;
    const double angle = 0.37;
    Eigen::Matrix2d rotation;
    rotation << std::cos(angle), -std::sin(angle),
                std::sin(angle),  std::cos(angle);
    const Eigen::Vector2d translation(0.137, -0.083);
    const RigidlyTransformedCurve transformed(
        curve, rotation, translation);

    const Interface2D original = adaptive_flower(curve);
    const Interface2D moved = adaptive_flower(transformed);

    require(moved.num_panels() == original.num_panels(),
            "rigid transform changed adaptive panel count");
    require(moved.num_points() == original.num_points(),
            "rigid transform changed adaptive point count");
    require((moved.panel_point_indices().array()
             == original.panel_point_indices().array()).all(),
            "rigid transform changed P2 connectivity");

    for (int point = 0; point < original.num_points(); ++point) {
        const Eigen::Vector2d source_point =
            original.points().row(point).transpose();
        const Eigen::Vector2d source_normal =
            original.normals().row(point).transpose();
        require_near(
            moved.points().row(point).transpose(),
            translation + rotation * source_point,
            2.0e-10,
            "adaptive point is not rigid-transform invariant");
        require_near(
            moved.normals().row(point).transpose(),
            rotation * source_normal,
            2.0e-10,
            "adaptive normal is not rigid-transform invariant");
        require_near(
            moved.weights()[point],
            original.weights()[point],
            2.0e-11,
            "adaptive weight is not rigid-transform invariant");
    }
}

void test_invalid_turn_limit()
{
    const SmoothFlowerCurve curve;
    constexpr double h = 5.0 / 32.0;
    require_invalid_argument(
        [&] {
            (void)CurveResampler2D::
                discretize_quadratic_lagrange_curvature_adaptive(
                    curve, h, 1.6, 0.0);
        },
        "zero panel-turn limit was not rejected");
    require_invalid_argument(
        [&] {
            (void)CurveResampler2D::
                discretize_quadratic_lagrange_curvature_adaptive(
                    curve, h, 1.6, -0.8);
        },
        "negative panel-turn limit was not rejected");
    require_invalid_argument(
        [&] {
            (void)CurveResampler2D::
                discretize_quadratic_lagrange_curvature_adaptive(
                    curve,
                    h,
                    1.6,
                    std::numeric_limits<double>::quiet_NaN());
        },
        "non-finite panel-turn limit was not rejected");
}

void require_label_crossing_invariant(const kfbim::GridPair2D& pair)
{
    const auto& grid = pair.grid();
    int side_transition_edges = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        const auto neighbors = grid.neighbors(node);
        for (const int slot : {1, 3}) {
            const int neighbor = neighbors[static_cast<std::size_t>(slot)];
            if (neighbor < 0
                || pair.domain_label(node)
                       == pair.domain_label(neighbor)) {
                continue;
            }
            ++side_transition_edges;
            const kfbim::P2CrossingOwner2D crossing =
                pair.p2_crossing_owner_between(node, neighbor);
            require(
                crossing.flips_domain_side(),
                "P2 domain labels differ without an odd transverse crossing on edge "
                    + std::to_string(node) + " -> "
                    + std::to_string(neighbor));
        }
    }
    require(side_transition_edges > 0,
            "P2 crossing invariant test found no interface edges");
}

void require_regression_edge(const kfbim::GridPair2D& pair,
                             int node,
                             int neighbor,
                             int expected_exact_crossings,
                             const std::string& name)
{
    const kfbim::P2CrossingOwner2D crossing =
        pair.p2_crossing_owner_between(node, neighbor);
    require(crossing.exact_intersection_count
                == expected_exact_crossings,
            name + " exact intersection count changed");
    require(crossing.transverse_intersection_count
                == expected_exact_crossings,
            name + " transverse intersection count changed");
    require(pair.domain_label(node) == pair.domain_label(neighbor),
            name + " endpoints must have the same domain label");
}

void test_p2_labels_follow_exact_crossing_parity()
{
    for (const int n : {32, 128}) {
        constexpr double box_min = -1.25;
        constexpr double box_length = 2.50;
        const double h = box_length / static_cast<double>(n);
        const kfbim::CartesianGrid2D grid(
            {box_min, box_min}, {h, h}, {n, n},
            kfbim::DofLayout2D::Node);
        auto curve = std::make_shared<CrossingRegressionFlower7Curve>();
        const Interface2D iface =
            CurveResampler2D::discretize_quadratic_lagrange(
                curve, h, 2.20);
        const kfbim::GridPair2D pair(grid, iface);

        require_label_crossing_invariant(pair);
        if (n == 32) {
            const int center_node = grid.index(24, 15);
            require_regression_edge(
                pair, grid.index(23, 15), center_node, 0,
                "flower7 N=32 zero-hit horizontal edge");
            require_regression_edge(
                pair, grid.index(24, 14), center_node, 0,
                "flower7 N=32 zero-hit vertical edge");
            require_regression_edge(
                pair, center_node, grid.index(24, 16), 2,
                "flower7 N=32 double-hit vertical edge");
        } else {
            const int center_node = grid.index(35, 57);
            require_regression_edge(
                pair, center_node, grid.index(36, 57), 0,
                "flower7 N=128 zero-hit horizontal edge");
            require_regression_edge(
                pair, center_node, grid.index(35, 58), 0,
                "flower7 N=128 zero-hit vertical edge");
        }
    }
}

} // namespace

int main()
{
    try {
        test_retained_exact_geometry_and_const_reference_fallback();
        test_exact_geometry_is_continuous_and_on_source_curve();
        test_cauchy_geometry_cache_uses_exact_provider();
        test_closed_p2_topology_and_positive_weights();
        test_curvature_monitor_adds_panels();
        test_rigid_transform_invariance();
        test_invalid_turn_limit();
        test_p2_labels_follow_exact_crossing_parity();
        std::cout << "2D curve resampler tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "2D curve resampler test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
