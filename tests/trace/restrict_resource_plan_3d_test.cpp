#include "src/support/trace/restrict_resource_plan_3d.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace {
using namespace kfbim::app3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template<class F> void require_throws(F function, const std::string& message)
{
    bool threw = false;
    try { function(); } catch (const std::exception&) { threw = true; }
    require(threw, message);
}

RestrictResourceAnchor3D anchor(double x, int sheet, int crossing = -1)
{
    return {Eigen::Vector3d(x, 0.0, 0.0), sheet, 0.123, 0.456, sheet, crossing};
}

RestrictSupportVisit3D visit(int trace, int id, double x,
                            bool desired = true, bool actual = false)
{
    return {trace, desired, id, Eigen::Vector3d(x, 0.0, 0.0), actual};
}

void test_selection_reuse_and_thresholds()
{
    const std::vector<RestrictResourceAnchor3D> trace{
        anchor(0.0, 7), anchor(4.0, 7), anchor(2.05, 9)};
    const std::vector<RestrictResourceAnchor3D> spread{
        anchor(1.4, 7, 100), anchor(2.4, 7, 200), anchor(2.0, 9, 300)};
    const std::vector<RestrictSupportVisit3D> visits{
        visit(0, 10, 1.5), // Exactly 1.5h: crossing lookup must not replace trace.
        visit(0, 11, 2.0), // Trace tie picks first; crossing strictly improves.
        visit(1, 11, 2.0), // Same sheet and node is the same request.
        visit(2, 11, 2.0), // Different sheet must remain a distinct request.
        visit(0, 12, 1.0, false, false), // Correct side: no correction.
        visit(0, 13, 8.0), // Too far is diagnostic, never hard rejection.
    };
    const auto reused = build_restrict_resource_plan_3d(trace, spread, visits, 1.0);
    const auto reference = build_restrict_resource_plan_3d(trace, spread, visits, 1.0,
        RestrictResourceMode3D::ReferencePerVisit);
    const auto guarded = build_restrict_resource_plan_3d(trace, spread, visits, 1.0,
        RestrictResourceMode3D::GuardedReuseBySheetNode);
    require(reused.counts.support_visits == 6 && reused.counts.wrong_side_visits == 5
                && reused.requests.size() == 4 && reference.requests.size() == 5,
            "only wrong-side visits become (sheet,node) resources");
    require(reused.visit_to_request[1] == reused.visit_to_request[2]
                && reused.visit_to_request[1] != reused.visit_to_request[3]
                && reused.visit_to_request[4] == -1,
            "same physical sheet, not component or centre, controls sharing");
    require(reused.requests[0].anchor_source == RestrictAnchorSource3D::Trace
                && reused.requests[0].anchor_index == 0,
            "exactly 1.5h retains trace anchor even if spread is nearer");
    require(reused.requests[1].anchor_source == RestrictAnchorSource3D::SpreadCrossing
                && reused.requests[1].anchor_index == 1,
            "strictly nearer same-sheet spread selected beyond trigger");
    require(reused.required_crossing_ids == std::vector<int>{200},
            "only required spread crossing IDs are exported once");
    require(reused.requests.back().too_far && !reused.requests.back().fallback_required,
            "2.25h is only a mixed-core diagnostic");
    require(guarded.requests[1].competing_sheet && guarded.requests[1].fallback_required
                && guarded.requests.back().fallback_required,
            "guarded mode requests explicit competing-sheet/long-path fallback");
    for (std::size_t i = 0; i < visits.size(); ++i) {
        const int r = reused.visit_to_request[i];
        const int b = reference.visit_to_request[i];
        require((r < 0) == (b < 0), "reference and reused access order agree");
        if (r < 0) continue;
        const auto& x = reused.requests[static_cast<std::size_t>(r)];
        const auto& y = reference.requests[static_cast<std::size_t>(b)];
        require(x.anchor_source == y.anchor_source && x.anchor_index == y.anchor_index
                    && x.distance == y.distance && x.too_far == y.too_far
                    && x.competing_sheet == y.competing_sheet,
                "reference and reuse select exactly identical resources per visit");
    }

    auto ties = build_restrict_resource_plan_3d({anchor(-2.0, 0), anchor(2.0, 0)},
        {anchor(2.0, 0, 1)}, {visit(0, 0, 0.0)}, 1.0);
    require(ties.requests[0].anchor_source == RestrictAnchorSource3D::Trace
                && ties.requests[0].anchor_index == 0,
            "exact nearest ties retain first trace and do not switch to spread");
    auto close_native = std::vector<RestrictResourceAnchor3D>{anchor(0, 0), anchor(0, 0)};
    close_native[1].u = std::nextafter(close_native[0].u, 1.0);
    const auto native_plan = build_restrict_resource_plan_3d(close_native, {},
        {visit(1, 42, 0.5)}, 1.0);
    require(native_plan.requests[0].anchor_index == 0
                && close_native[0].u != close_native[1].u,
            "native parameters are never quantized or merged");
    const auto exact_far = build_restrict_resource_plan_3d({anchor(0, 0)}, {},
        {visit(0, 0, 2.25)}, 1.0);
    require(!exact_far.requests[0].too_far, "exact 2.25h is not too_far");
}

void test_kdtree_against_exhaustive()
{
    std::mt19937 engine(91731);
    std::uniform_real_distribution<double> random(-3.0, 3.0);
    std::vector<RestrictResourceAnchor3D> trace, spread;
    for (int i = 0; i < 150; ++i) {
        auto a = anchor(0, i % 3);
        a.point = Eigen::Vector3d(random(engine), random(engine), random(engine));
        trace.push_back(a);
        a.point = Eigen::Vector3d(random(engine), random(engine), random(engine));
        a.crossing_id = 1000 + i;
        spread.push_back(a);
    }
    std::vector<RestrictSupportVisit3D> visits;
    for (int i = 0; i < 200; ++i) {
        auto v = visit(i % 3, i, 0);
        v.grid_point = Eigen::Vector3d(random(engine), random(engine), random(engine));
        visits.push_back(v);
    }
    const double h = 0.35;
    const auto plan = build_restrict_resource_plan_3d(trace, spread, visits, h);
    for (std::size_t i = 0; i < visits.size(); ++i) {
        const auto& v = visits[i];
        const int sheet = trace[static_cast<std::size_t>(v.trace_center_id)].sheet_id;
        const auto exhaustive = [&](const auto& anchors) {
            int best = -1;
            double distance = std::numeric_limits<double>::infinity();
            for (int j = 0; j < static_cast<int>(anchors.size()); ++j) {
                const auto& a = anchors[static_cast<std::size_t>(j)];
                const double d = (a.point - v.grid_point).squaredNorm();
                if (a.sheet_id == sheet && d < distance) { best = j; distance = d; }
            }
            return std::make_pair(best, distance);
        };
        auto selected = exhaustive(trace);
        auto source = RestrictAnchorSource3D::Trace;
        if (std::sqrt(selected.second) > 1.5 * h) {
            const auto cross = exhaustive(spread);
            if (cross.second < selected.second) {
                selected = cross;
                source = RestrictAnchorSource3D::SpreadCrossing;
            }
        }
        const auto& actual = plan.requests[static_cast<std::size_t>(plan.visit_to_request[i])];
        require(actual.anchor_index == selected.first && actual.anchor_source == source
                    && actual.distance == std::sqrt(selected.second),
                "KD indexed selection equals exhaustive same-sheet reference");
    }
}

double monomial(const Eigen::Vector3d& p, int a, int b, int c)
{
    return std::pow(p.x(), a) * std::pow(p.y(), b) * std::pow(p.z(), c);
}

void test_shared_cover_reproduction()
{
    const kfbim::CartesianGrid3D grid({-2.0, -2.0, -2.0}, {0.125, 0.125, 0.125},
                                     {32, 32, 32}, kfbim::DofLayout3D::Node);
    const Eigen::Vector3d q(0.071, -0.043, 0.083);
    const Eigen::Vector3d n = Eigen::Vector3d(1.0, -2.0, 3.0).normalized();
    for (auto kind : {TensorProductCoverKind3D::Q27Cover3, TensorProductCoverKind3D::Q64Cover4}) {
        const auto stencil = build_shared_side_cover_restrict_stencil_3d(grid, q, n, kind);
        Eigen::Matrix<double, 6, 4> vandermonde;
        for (int side = 0; side < 2; ++side) {
            const auto& s = stencil.sides[static_cast<std::size_t>(side)];
            require(s.desired_inside == (side == 0), "negative side is interior");
            require(s.grid_ids.size() == static_cast<std::size_t>(tensor_product_cover_node_count_3d(kind)),
                    "three samples share exactly one Q27 or Q64 cover");
            for (int k = 0; k < 3; ++k) {
                const double r = s.signed_rho[static_cast<std::size_t>(k)];
                for (int d = 0; d < 4; ++d)
                    vandermonde(3 * side + k, d) = std::pow(r, d);
            }
        }
        require((stencil.cubic_pseudoinverse * vandermonde
                 - Eigen::Matrix4d::Identity()).cwiseAbs().maxCoeff() < 2.0e-14,
                "six-point cubic pseudoinverse reproduces all four coefficients");
        const int degree = tensor_product_cover_width_3d(kind) - 1;
        for (int a = 0; a <= degree; ++a) {
            for (int b = 0; b <= degree; ++b) {
                for (int c = 0; c <= degree; ++c) {
                    double value = 0.0, normal = 0.0;
                    for (const auto& side : stencil.sides) {
                        Eigen::VectorXd data(static_cast<int>(side.grid_ids.size()));
                        for (int i = 0; i < data.size(); ++i) {
                            const auto p = grid.coord(side.grid_ids[static_cast<std::size_t>(i)]);
                            data[i] = monomial(Eigen::Vector3d(p[0], p[1], p[2]), a, b, c);
                        }
                        const Eigen::Vector3d samples = side.sampling_weights * data;
                        for (int i = 0; i < 3; ++i)
                            require(std::abs(samples[i] - monomial(side.sample_points[static_cast<std::size_t>(i)], a, b, c))
                                        < 2.0e-13,
                                    "tensor cover reproduces every per-axis polynomial degree");
                        value += (side.value_weights * data)[0];
                        normal += (side.normal_weights * data)[0];
                    }
                    if (a + b + c <= degree) {
                        double derivative = 0.0;
                        if (a) derivative += n.x() * a * monomial(q, a - 1, b, c);
                        if (b) derivative += n.y() * b * monomial(q, a, b - 1, c);
                        if (c) derivative += n.z() * c * monomial(q, a, b, c - 1);
                        require(std::abs(value - monomial(q, a, b, c)) < 2.0e-13
                                    && std::abs(normal - derivative) < 2.0e-12,
                                "3+3 cover recovers polynomial value and physical normal derivative");
                    }
                }
            }
        }
        const auto repeated = build_shared_side_cover_restrict_stencil_3d(grid, q, n, kind);
        require(repeated.sides[0].grid_ids == stencil.sides[0].grid_ids
                    && (repeated.sides[1].sampling_weights - stencil.sides[1].sampling_weights).norm() == 0,
                "shared cover selection and weights are deterministic");
        const auto near_box = build_shared_side_cover_restrict_stencil_3d(grid,
            Eigen::Vector3d(1.79, 1.79, 1.79), n, kind);
        for (const auto& side : near_box.sides)
            for (int id : side.grid_ids)
                require(id >= 0 && id < grid.num_dofs(), "near-box cover uses valid full grid IDs");
    }
    require_throws([&] { (void)build_shared_side_cover_restrict_stencil_3d(grid,
        Eigen::Vector3d(1.99, 0, 0), Eigen::Vector3d::UnitX(), TensorProductCoverKind3D::Q27Cover3); },
        "out-of-box normal sample is rejected instead of silently clamped");
}

void test_invalid_inputs()
{
    const std::vector<RestrictResourceAnchor3D> trace{anchor(0, 0)};
    require_throws([&] { (void)build_restrict_resource_plan_3d(trace, {}, {}, 0.0); },
                   "nonpositive h rejected");
    require_throws([&] { (void)build_restrict_resource_plan_3d(trace, {}, {visit(1, 0, 0)}, 1.0); },
                   "unknown trace centre rejected");
    require_throws([&] { (void)build_restrict_resource_plan_3d(trace, {},
        {visit(0, 0, 0), visit(0, 0, 0.1)}, 1.0); }, "same full ID cannot map to different coordinates");
    require_throws([&] { (void)build_restrict_resource_plan_3d(trace, {anchor(0, 0)}, {}, 1.0); },
                   "spread anchor must retain its crossing identity");
    const auto empty = build_restrict_resource_plan_3d({}, {}, {}, 1.0);
    require(empty.requests.empty(), "empty operator setup is supported");
}
} // namespace

int main()
{
    try {
        test_selection_reuse_and_thresholds();
        test_kdtree_against_exhaustive();
        test_shared_cover_reproduction();
        test_invalid_inputs();
        std::cout << "3D restrict resource plan and shared-side cover tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "3D restrict resource plan test failure: " << error.what() << '\n';
        return 1;
    }
}
