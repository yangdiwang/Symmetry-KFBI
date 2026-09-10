#include "src/support/trace/resource_restrict_assembly_3d.hpp"

#include <Eigen/QR>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace kfbim::app3d;

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

template<class F> void rejects(F operation, const std::string& message)
{
    bool threw = false;
    try { operation(); } catch (const std::invalid_argument&) { threw = true; }
    require(threw, message);
}

struct Fixture {
    static constexpr int full_grid_size = 11;
    static constexpr int coefficient_count = 3;
    std::vector<SharedSideCoverRestrictStencil3D> stencils;
    std::vector<RestrictSupportVisit3D> visits;
    std::vector<RestrictResourceAnchor3D> anchors;
    std::vector<ResourceTraceSampleJumpRows3D> trace_rows;
    RestrictResourcePlan3D plan;
    std::vector<ResourceAffineRow3D> rows;
};

ResourceAffineRow3D jump_row(double z, bool known)
{
    ResourceAffineRow3D row;
    row.columns = {0, 1, 2};
    row.values = {1.0, z, z * z};
    if (known) row.known = 0.8 - 0.4 * z + 0.6 * z * z;
    return row;
}

Fixture fixture(RestrictResourceMode3D mode, bool same_side = false, bool known = true)
{
    Fixture f;
    SharedSideCoverRestrictStencil3D stencil;
    stencil.h = 0.2;
    Eigen::Matrix<double, 6, 4> vandermonde;
    const std::array<double, 6> rho{{-1.5, -0.75, -0.5, 0.5, 0.75, 1.5}};
    for (int i = 0; i < 6; ++i)
        vandermonde.row(i) << 1.0, rho[i], rho[i] * rho[i], rho[i] * rho[i] * rho[i];
    stencil.cubic_pseudoinverse = vandermonde.colPivHouseholderQr().solve(
        Eigen::Matrix<double, 6, 6>::Identity());
    const std::array<std::array<int, 3>, 2> ids{{{{1, 4, 9}}, {{2, 7, 10}}}};
    for (int side_id = 0; side_id < 2; ++side_id) {
        auto& side = stencil.sides[side_id];
        side.desired_inside = side_id == 0;
        side.grid_ids.assign(ids[side_id].begin(), ids[side_id].end());
        side.sampling_weights = Eigen::Matrix3d::Identity();
        side.value_recovery = stencil.cubic_pseudoinverse.block<1, 3>(0, 3 * side_id);
        side.normal_recovery = stencil.cubic_pseudoinverse.block<1, 3>(1, 3 * side_id) / stencil.h;
        side.value_weights = side.value_recovery;
        side.normal_weights = side.normal_recovery;
        for (int layer = 0; layer < 3; ++layer) {
            side.signed_rho[layer] = rho[3 * side_id + layer];
            side.sample_points[layer] = Eigen::Vector3d(0, 0, stencil.h * side.signed_rho[layer]);
        }
    }
    for (int center = 0; center < 2; ++center) {
        f.stencils.push_back(stencil);
        RestrictResourceAnchor3D anchor;
        anchor.patch_id = 0;
        anchor.u = anchor.v = 0.5;
        anchor.sheet_id = 2;
        f.anchors.push_back(anchor);
        ResourceTraceSampleJumpRows3D qrows;
        for (int side = 0; side < 2; ++side) {
            for (int layer = 0; layer < 3; ++layer) {
                const auto& s = stencil.sides[side];
                RestrictSupportVisit3D visit;
                visit.trace_center_id = center;
                visit.desired_inside = s.desired_inside;
                visit.grid_full_id = s.grid_ids[layer];
                visit.grid_point = s.sample_points[layer];
                // Algebraic fixture deliberately includes wrong-side and
                // same-side nodes on both sample covers.
                visit.actual_inside = same_side ? visit.desired_inside
                    : (layer == 1 ? visit.desired_inside : !visit.desired_inside);
                f.visits.push_back(visit);
                qrows[3 * side + layer] = jump_row(visit.grid_point.z(), known);
            }
        }
        f.trace_rows.push_back(qrows);
    }
    f.plan = build_restrict_resource_plan_3d(f.anchors, {}, f.visits, stencil.h, mode);
    for (const auto& request : f.plan.requests)
        f.rows.push_back(jump_row(request.grid_point.z(), known));
    return f;
}

ResourceRestrictAssembly3D assemble(const Fixture& f)
{
    return assemble_resource_restrict_3d(f.plan, f.stencils, f.visits,
        f.rows, f.trace_rows, Fixture::full_grid_size, Fixture::coefficient_count);
}

double value(const ResourceAffineRow3D& row, const Eigen::VectorXd& c)
{
    double result = row.known;
    for (std::size_t k = 0; k < row.columns.size(); ++k)
        result += row.values[k] * c[row.columns[k]];
    return result;
}

Eigen::VectorXd physical_grid(const Fixture& f, const Eigen::VectorXd& c, bool known)
{
    Eigen::VectorXd grid = Eigen::VectorXd::Zero(Fixture::full_grid_size);
    for (const auto& visit : f.visits) {
        const double z = visit.grid_point.z();
        const double exterior = 2.0 - 0.7 * z + 0.4 * z * z - 0.3 * z * z * z;
        grid[visit.grid_full_id] = exterior
            + (visit.actual_inside ? value(jump_row(z, known), c) : 0.0);
    }
    return grid;
}

void polynomial_sign_tests(bool same_side, bool known)
{
    const Fixture f = fixture(RestrictResourceMode3D::ReuseBySheetNode, same_side, known);
    const auto a = assemble(f);
    require(a.Rg_value.cols() == Fixture::full_grid_size, "grid map must use full Cartesian dimensions");
    require(a.exterior.Rc_normal.cols() == 3 && a.exterior.Rc_normal.rows() == 2,
        "coefficient and trace dimensions");
    for (int degree = 0; degree < 3; ++degree) {
        Eigen::Vector3d c = Eigen::Vector3d::Zero();
        c[degree] = 1.3;
        const auto grid = physical_grid(f, c, known);
        const Eigen::VectorXd exterior_value = a.Rg_value * grid + a.exterior.Rc_value * c + a.exterior.known_value;
        const Eigen::VectorXd exterior_normal = a.Rg_normal * grid + a.exterior.Rc_normal * c + a.exterior.known_normal;
        const Eigen::VectorXd interior_value = a.Rg_value * grid + a.interior.Rc_value * c + a.interior.known_value;
        const Eigen::VectorXd interior_normal = a.Rg_normal * grid + a.interior.Rc_normal * c + a.interior.known_normal;
        require((exterior_value.array() - 2.0).abs().maxCoeff() < 2e-13, "exterior constant/linear/quadratic jump sign");
        require((exterior_normal.array() + 0.7).abs().maxCoeff() < 2e-12, "exterior normal physical 1/h sign");
        require((interior_value.array() - (2.0 + c[0] + (known ? 0.8 : 0.0))).abs().maxCoeff() < 2e-13,
            "interior positive samples add q-centered jump");
        require((interior_normal.array() - (-0.7 + c[1] - (known ? 0.4 : 0.0))).abs().maxCoeff() < 2e-12,
            "interior derivative retains affine jump");
    }
    if (same_side) require(f.plan.requests.empty() && a.stats.wrong_side_visits == 0,
        "same-side visits need no resource requests");
}

void reference_reuse_and_affine()
{
    const Fixture ref = fixture(RestrictResourceMode3D::ReferencePerVisit);
    const Fixture reuse = fixture(RestrictResourceMode3D::ReuseBySheetNode);
    require(ref.plan.requests.size() > reuse.plan.requests.size(), "resource fixture must actually reuse requests");
    const auto a = assemble(ref), b = assemble(reuse);
    for (const auto pair : {
        std::make_pair(&a.Rg_value, &b.Rg_value), std::make_pair(&a.Rg_normal, &b.Rg_normal),
        std::make_pair(&a.interior.Rc_value, &b.interior.Rc_value),
        std::make_pair(&a.interior.Rc_normal, &b.interior.Rc_normal),
        std::make_pair(&a.exterior.Rc_value, &b.exterior.Rc_value),
        std::make_pair(&a.exterior.Rc_normal, &b.exterior.Rc_normal)})
        require(resource_sparse_max_difference_3d(*pair.first, *pair.second) == 0.0,
            "reference/reuse sparse matrices agree exactly");
    require((a.exterior.known_value - b.exterior.known_value).norm() == 0
        && (a.exterior.known_normal - b.exterior.known_normal).norm() == 0
        && (a.interior.known_value - b.interior.known_value).norm() == 0
        && (a.interior.known_normal - b.interior.known_normal).norm() == 0,
        "reference/reuse affine known terms agree");
    Eigen::Vector3d p(0.2, -0.4, 0.7);
    Eigen::Matrix<double, 3, 2> G;
    G << 1, 0.2, -0.3, 0.9, 0.4, -0.2;
    Eigen::Vector2d z(0.6, -0.8);
    const Eigen::VectorXd u0 = Eigen::VectorXd::LinSpaced(11, -0.4, 1.2);
    Eigen::Matrix<double, 11, 2> H;
    H.col(0) = Eigen::VectorXd::LinSpaced(11, -0.2, 0.6);
    H.col(1) = Eigen::VectorXd::LinSpaced(11, 0.3, -0.1);
    const Eigen::VectorXd full = a.Rg_normal * (u0 + H * z)
        + a.exterior.Rc_normal * (p + G * z) + a.exterior.known_normal;
    const Eigen::VectorXd affine = a.Rg_normal * u0 + a.exterior.Rc_normal * p + a.exterior.known_normal;
    const Eigen::VectorXd linear = (a.Rg_normal * H + a.exterior.Rc_normal * G) * z;
    require((full - affine - linear).lpNorm<Eigen::Infinity>() < 2e-13,
        "nonzero particular lifting splits exactly into affine and homogeneous parts");
    require((a.exterior.Rc_normal * G * z).norm() > 1e-3, "RcG term is nonzero and cannot be omitted");
    require(a.stats.grid_value_nnz == static_cast<std::size_t>(a.Rg_value.nonZeros()), "nnz statistics agree");
}

void invalid_input_and_small_values()
{
    const Fixture good = fixture(RestrictResourceMode3D::ReuseBySheetNode);
    rejects([&] { (void)assemble_resource_restrict_3d(good.plan, good.stencils, good.visits,
        good.rows, good.trace_rows, 10, 3); }, "interior-grid dimensions cannot replace full-grid dimensions");
    auto bad = good;
    std::swap(bad.visits[0], bad.visits[1]);
    rejects([&] { (void)assemble(bad); }, "visit ordering is validated");
    bad = good;
    bad.rows[0].columns = {0, 0, 2};
    rejects([&] { (void)assemble(bad); }, "duplicate density columns are rejected");
    bad = good;
    bad.rows[0].columns = {0, 2, 1};
    rejects([&] { (void)assemble(bad); }, "unsorted density columns are rejected");
    bad = good;
    bad.plan.requests[0].fallback_required = true;
    rejects([&] { (void)assemble(bad); }, "guarded fallback must not silently use an unresolved row");
    bad.rows[0].fallback_resolved = true;
    (void)assemble(bad);
    auto tiny = good;
    for (auto& row : tiny.rows) { for (double& x : row.values) x *= 1e-80; row.known = 0; }
    for (auto& rows : tiny.trace_rows)
        for (auto& row : rows) { for (double& x : row.values) x *= 1e-80; row.known = 0; }
    const auto a = assemble(good), t = assemble(tiny);
    require(t.exterior.Rc_normal.nonZeros() == a.exterior.Rc_normal.nonZeros(), "small coefficients are not pruned");
    require(Eigen::MatrixXd(t.exterior.Rc_normal).norm() > 0, "very small nonzero coefficients are preserved");
    ResourceRestrictSparseMatrix3D wrong(1, 1);
    rejects([&] { (void)resource_sparse_max_difference_3d(a.Rg_value, wrong); }, "sparse comparison checks dimensions");
    ResourceRestrictSparseMatrix3D left(1, 3), right(1, 3);
    left.insert(0, 0) = 0.0;
    left.insert(0, 1) = -2.0;
    right.insert(0, 2) = 3.0;
    left.makeCompressed();
    right.makeCompressed();
    require(resource_sparse_max_difference_3d(left, right) == 3.0,
        "sparse comparison includes unmatched columns and explicit stored zeros");
}
} // namespace

int main()
{
    try {
        polynomial_sign_tests(false, false);
        polynomial_sign_tests(false, true);
        polynomial_sign_tests(true, true);
        reference_reuse_and_affine();
        invalid_input_and_small_values();
        std::cout << "resource restrict sparse assembly tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "resource restrict sparse assembly test failure: " << error.what() << '\n';
        return 1;
    }
}
