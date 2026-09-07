#include "src/support/topology/topology_reachable_target_projection_3d.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace kfbim::app3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

ConstraintBlock3D block(ConstraintBlockKind3D kind,
                        std::string key,
                        std::initializer_list<Eigen::Index> rows)
{
    ConstraintBlock3D result;
    result.kind = kind;
    result.key = std::move(key);
    result.row_ids.assign(rows.begin(), rows.end());
    return result;
}

AffineReduction3D fixed_topology()
{
    SparseConstraintBuilder3D builder(3);
    builder.append_row({0, 1}, {1.0, -1.0}, 0.0);
    AffineReductionOptions3D options;
    options.schedule =
        AffineEliminationSchedule3D::StagedVertexThenEdge;
    return affine_eliminate_local_svd_3d(
        builder.finalize(),
        {block(ConstraintBlockKind3D::Vertex, "fixed-star", {0})},
        options);
}

ConstraintSystem3D target_system(double first, double second)
{
    SparseConstraintBuilder3D builder(3);
    builder.append_row({0}, {1.0}, first);
    builder.append_row({1}, {1.0}, second);
    builder.append_row({2}, {1.0}, 4.0);
    return builder.finalize();
}

std::vector<ConstraintBlock3D> target_blocks()
{
    return {block(ConstraintBlockKind3D::Vertex, "target-a", {0}),
            block(ConstraintBlockKind3D::Vertex, "target-b", {1}),
            block(ConstraintBlockKind3D::Edge, "target-free", {2})};
}

void test_incompatible_target_is_projected_component_locally()
{
    const AffineReduction3D topology = fixed_topology();
    const auto projection = project_constraint_target_to_reachable_space_3d(
        topology, target_system(1.0, 3.0), target_blocks());
    require(!projection.constraints_exact,
            "incompatible requested target is explicitly marked inexact");
    require((projection.projected_rhs
             - (Eigen::Vector3d() << 2.0, 2.0, 4.0).finished())
                    .norm()
                < 2.0e-13,
            "shared vertex target is projected to its reachable common jet");
    require(std::abs(projection.projection_linf - 1.0) < 2.0e-13
                && std::abs(projection.projection_l2 - std::sqrt(2.0))
                       < 2.0e-13,
            "requested/projected target diagnostics are exact");
    require(projection.records.size() == 2
                && projection.records[0].row_ids
                    == std::vector<Eigen::Index>({0, 1})
                && projection.records[1].row_ids
                    == std::vector<Eigen::Index>({2}),
            "only current-CE connected target blocks share a local SVD");
    require(projection.reachable_certification_linf < 3.0e-13,
            "projected target has a certified fixed-topology preimage");

    // The requested system is genuinely incompatible with x0=x1.
    SparseConstraintBuilder3D requested(3);
    requested.append_row({0, 1}, {1.0, -1.0}, 0.0);
    requested.append_row({0}, {1.0}, 1.0);
    requested.append_row({1}, {1.0}, 3.0);
    requested.append_row({2}, {1.0}, 4.0);
    AffineReductionOptions3D options;
    options.schedule =
        AffineEliminationSchedule3D::StagedVertexThenEdge;
    bool incompatible = false;
    try {
        (void)affine_eliminate_local_svd_3d(
            requested.finalize(),
            {block(ConstraintBlockKind3D::Vertex, "all-star", {0, 1, 2}),
             block(ConstraintBlockKind3D::Edge, "free", {3})}, options);
    } catch (const ConstraintIncompatibility3D&) {
        incompatible = true;
    }
    require(incompatible,
            "unprojected incompatible target remains fail-fast");

    SparseConstraintBuilder3D projected(3);
    projected.append_row({0, 1}, {1.0, -1.0}, 0.0);
    projected.append_row({0}, {1.0}, projection.projected_rhs[0]);
    projected.append_row({1}, {1.0}, projection.projected_rhs[1]);
    projected.append_row({2}, {1.0}, projection.projected_rhs[2]);
    const AffineReduction3D combined = affine_eliminate_local_svd_3d(
        projected.finalize(),
        {block(ConstraintBlockKind3D::Vertex, "all-star", {0, 1, 2}),
         block(ConstraintBlockKind3D::Edge, "free", {3})}, options);
    require(combined.particular_residual_linf() < 3.0e-13
                && combined.homogeneous_residual_linf() < 3.0e-13,
            "projected target supports exact combined affine elimination");
}

void test_already_reachable_target_is_unchanged()
{
    const auto projection = project_constraint_target_to_reachable_space_3d(
        fixed_topology(), target_system(2.0, 2.0), target_blocks());
    require(projection.constraints_exact
                && projection.projection_linf < 3.0e-13
                && (projection.requested_rhs - projection.projected_rhs)
                       .norm()
                    < 3.0e-13,
            "reachable target passes through unchanged and remains exact");
}

} // namespace

int main()
{
    try {
        test_incompatible_target_is_projected_component_locally();
        test_already_reachable_target_is_unchanged();
        std::cout << "topology_reachable_target_projection_3d_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "topology_reachable_target_projection_3d_test: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
