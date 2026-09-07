#include "src/support/topology/topology_affine_reduction_3d.hpp"

#include <Eigen/SVD>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using kfbim::app3d::AffineReduction3D;
using kfbim::app3d::AffineReductionOptions3D;
using kfbim::app3d::AffineEliminationSchedule3D;
using kfbim::app3d::ConstraintBlock3D;
using kfbim::app3d::ConstraintBlockKind3D;
using kfbim::app3d::ConstraintIncompatibility3D;
using kfbim::app3d::ConstraintMeta3D;
using kfbim::app3d::ConstraintSystem3D;
using kfbim::app3d::PatchDofLayout3D;
using kfbim::app3d::SparseConstraintBuilder3D;
using kfbim::app3d::SparseMatrixCSR3D;
using kfbim::app3d::affine_eliminate_local_svd_3d;
using kfbim::app3d::merge_constraint_blocks_by_support_3d;
using kfbim::app3d::order_constraint_blocks_vertex_before_edge_3d;

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

ConstraintMeta3D meta(const std::string& kind,
                      const std::string& segment,
                      int connection = -1)
{
    ConstraintMeta3D result;
    result.kind = kind;
    result.segment = segment;
    result.connection = connection;
    return result;
}

ConstraintBlock3D block(ConstraintBlockKind3D kind,
                        const std::string& key,
                        std::initializer_list<Eigen::Index> rows)
{
    ConstraintBlock3D result;
    result.kind = kind;
    result.key = key;
    result.row_ids.assign(rows.begin(), rows.end());
    return result;
}

void test_patch_layout_round_trip()
{
    const PatchDofLayout3D layout{4, 3, 5, 11};
    require(layout.size() == 15, "patch layout coefficient count");
    for (int i = 0; i < layout.n_u; ++i) {
        for (int j = 0; j < layout.n_v; ++j) {
            const Eigen::Index global = layout.global_index(i, j);
            require(global == 11 + i * 5 + j,
                    "patch layout uses documented u-major indexing");
            require(layout.decode(global) == std::make_pair(i, j),
                    "patch layout encode/decode round trip");
            require(layout.contains(global),
                    "patch layout recognizes its coefficient");
        }
    }
    require(!layout.contains(10) && !layout.contains(26),
            "patch layout rejects neighboring patch ranges");

    bool threw = false;
    try {
        (void)layout.global_index(3, 0);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    require(threw, "patch layout rejects an invalid local index");
}

ConstraintSystem3D make_affine_system()
{
    SparseConstraintBuilder3D builder(6);
    builder.append_row({0, 1}, {1.0, 1.0}, 2.0,
                       meta("c0", "star-a", 7));
    builder.append_row({1, 2}, {1.0, -1.0}, -1.0,
                       meta("smooth_c1", "star-a", 7));
    builder.append_row({3, 4, 5}, {1.0, 2.0, -1.0}, 0.5,
                       meta("feature_jet", "edge-b", 8));
    return builder.finalize();
}

AffineReduction3D make_affine_reduction()
{
    return affine_eliminate_local_svd_3d(
        make_affine_system(),
        {block(ConstraintBlockKind3D::Edge, "edge-b", {2}),
         block(ConstraintBlockKind3D::Vertex, "star-a", {0, 1})});
}

void test_affine_invariants_lifts_and_support()
{
    const AffineReduction3D reduction = make_affine_reduction();
    require(reduction.base_size() == 6, "affine base dimension");
    require(reduction.reduced_size() == 3, "affine reduced dimension");
    require(reduction.particular_residual_linf() < 2.0e-13,
            "particular solution satisfies Cp=d");
    require(reduction.homogeneous_residual_linf() < 2.0e-13,
            "homogeneous map satisfies CE=0");
    require(reduction.homogeneous_orthogonality_residual() < 2.0e-13,
            "sequential local maps retain independent orthonormal columns");
    require(reduction.records().size() == 2
                && reduction.records()[0].kind
                    == ConstraintBlockKind3D::Vertex
                && reduction.records()[1].kind
                    == ConstraintBlockKind3D::Edge,
            "vertex blocks are eliminated before edge blocks");

    Eigen::Vector3d reduced;
    reduced << 0.37, -0.91, 1.13;
    const Eigen::VectorXd base = reduction.lift_base(reduced);
    require((reduction.constraints().C * base
             - reduction.constraints().d)
                    .lpNorm<Eigen::Infinity>()
                < 3.0e-13,
            "every affine lift satisfies all constraints");
    require((reduction.lift_homogeneous_base(reduced)
             - reduction.homogeneous_base() * reduced)
                    .norm()
                < 1.0e-15,
            "homogeneous lift excludes the affine particular");

    SparseMatrixCSR3D sparse_expansion(7, 6);
    std::vector<Eigen::Triplet<double>> expansion_entries;
    for (Eigen::Index index = 0; index < 6; ++index)
        expansion_entries.emplace_back(index, index, 1.0 + 0.1 * index);
    expansion_entries.emplace_back(6, 0, 0.25);
    expansion_entries.emplace_back(6, 5, -0.5);
    sparse_expansion.setFromTriplets(
        expansion_entries.begin(), expansion_entries.end());
    sparse_expansion.makeCompressed();
    const Eigen::MatrixXd dense_expansion(sparse_expansion);
    require((reduction.particular_full(sparse_expansion)
             - reduction.particular_full(dense_expansion))
                    .norm()
                < 2.0e-14,
            "dense and sparse particular-full APIs agree");
    require((reduction.lift_full(reduced, sparse_expansion)
             - dense_expansion * base)
                    .norm()
                < 2.0e-14,
            "full lift applies A0 after the affine base lift");
    require((Eigen::MatrixXd(reduction.homogeneous_full(sparse_expansion))
             - reduction.homogeneous_full(dense_expansion))
                    .norm()
                < 2.0e-14,
            "dense and sparse full homogeneous maps agree");

    const auto support = reduction.base_basis_support(0);
    require(!support.empty(), "reduced density basis support is observable");
    const Eigen::Index inspected_row = support.front().first;
    const auto formula = reduction.base_coefficient_formula(inspected_row);
    double reconstructed = formula.particular;
    for (const auto& term : formula.reduced_terms)
        reconstructed += term.second * reduced[term.first];
    require_near(reconstructed, base[inspected_row], 2.0e-14,
                 "coefficient formula reconstructs a lifted coefficient");
}

void test_global_svd_oracle_subspace()
{
    SparseConstraintBuilder3D builder(7);
    builder.append_row({0, 1}, {1.0, -1.0}, 0.0,
                       meta("c0", "v"));
    builder.append_row({1, 2}, {1.0, -1.0}, 0.0,
                       meta("c0", "v"));
    builder.append_row({2, 3, 4}, {1.0, 1.0, -1.0}, 0.0,
                       meta("smooth_c1", "e0"));
    builder.append_row({4, 5, 6}, {1.0, 1.0, -2.0}, 0.0,
                       meta("smooth_c1", "e1"));
    const ConstraintSystem3D system = builder.finalize();
    AffineReductionOptions3D sequential_options;
    sequential_options.merge_shared_support_blocks = false;
    const AffineReduction3D reduction = affine_eliminate_local_svd_3d(
        system,
        {block(ConstraintBlockKind3D::Edge, "e1", {3}),
         block(ConstraintBlockKind3D::Vertex, "v", {0, 1}),
         block(ConstraintBlockKind3D::Edge, "e0", {2})},
        sequential_options);

    // This global dense SVD is deliberately test-only: it is an oracle for
    // the sequential sparse/local construction, never a production path.
    const Eigen::MatrixXd dense_constraints(system.C);
    Eigen::JacobiSVD<Eigen::MatrixXd> oracle(
        dense_constraints, Eigen::ComputeFullV);
    const Eigen::VectorXd singular = oracle.singularValues();
    const double cutoff = singular[0] * 1.0e-11;
    Eigen::Index rank = 0;
    while (rank < singular.size() && singular[rank] > cutoff)
        ++rank;
    const Eigen::MatrixXd oracle_nullspace =
        oracle.matrixV().rightCols(7 - rank);
    const Eigen::MatrixXd local_nullspace(reduction.homogeneous_base());
    require(local_nullspace.cols() == oracle_nullspace.cols(),
            "local reduction and global oracle have equal nullity");
    const Eigen::MatrixXd projector_defect =
        local_nullspace * local_nullspace.transpose()
        - oracle_nullspace * oracle_nullspace.transpose();
    require(projector_defect.cwiseAbs().maxCoeff() < 3.0e-12,
            "sequential local SVD spans the global-SVD oracle nullspace");
}

void test_zero_row_and_incompatibility_are_explicit()
{
    SparseConstraintBuilder3D builder(2);
    builder.append_row({0, 0}, {1.0, -1.0}, 0.25,
                       meta("c0", "cancelled"));
    bool zero_row_threw = false;
    try {
        (void)builder.finalize();
    } catch (const ConstraintIncompatibility3D& error) {
        zero_row_threw = error.row_id() == 0 && error.residual() > 0.0;
    }
    require(zero_row_threw,
            "zero sparse row with nonzero RHS raises incompatibility");

    SparseConstraintBuilder3D inconsistent(1);
    inconsistent.append_row({0}, {1.0}, 1.0, meta("c0", "bad"));
    inconsistent.append_row({0}, {2.0}, 3.0, meta("c0", "bad"));
    bool svd_threw = false;
    try {
        (void)affine_eliminate_local_svd_3d(
            inconsistent.finalize(),
            {block(ConstraintBlockKind3D::Vertex, "bad", {0, 1})});
    } catch (const ConstraintIncompatibility3D& error) {
        svd_threw = error.block_key() == "bad" && error.residual() > 0.0;
    }
    require(svd_threw,
            "inconsistent redundant equations are not hidden by least squares");
}

void test_support_union_find_and_block_order()
{
    SparseConstraintBuilder3D builder(6);
    builder.append_row({0}, {1.0}, 0.0, meta("c0", "vertex-a"));
    builder.append_row({1}, {1.0}, 0.0, meta("c0", "edge-a"));
    builder.append_row({1, 2}, {1.0, -1.0}, 0.0,
                       meta("smooth_c1", "edge-b"));
    builder.append_row({5}, {1.0}, 0.0, meta("c0", "vertex-b"));
    const ConstraintSystem3D system = builder.finalize();

    const auto ordered = order_constraint_blocks_vertex_before_edge_3d(
        {block(ConstraintBlockKind3D::Edge, "edge-a", {1}),
         block(ConstraintBlockKind3D::Vertex, "vertex-a", {0}),
         block(ConstraintBlockKind3D::Edge, "edge-b", {2}),
         block(ConstraintBlockKind3D::Vertex, "vertex-b", {3})});
    require(ordered[0].kind == ConstraintBlockKind3D::Vertex
                && ordered[1].kind == ConstraintBlockKind3D::Vertex,
            "stable ordering places every vertex block first");

    const auto merged = merge_constraint_blocks_by_support_3d(
        system.C, ordered);
    require(merged.size() == 3,
            "union-find merges exactly the blocks sharing an active column");
    require(merged[0].kind == ConstraintBlockKind3D::Vertex
                && merged[1].kind == ConstraintBlockKind3D::Vertex
                && merged[2].kind == ConstraintBlockKind3D::Edge,
            "support merging preserves vertex-before-edge ordering");
    require(merged[2].row_ids == std::vector<Eigen::Index>({1, 2})
                && merged[2].source_keys.size() == 2,
            "merged support component retains rows and physical source keys");
}

void test_staged_vertex_then_current_edge_support_closure()
{
    SparseConstraintBuilder3D builder(4);
    // The two edge blocks have disjoint support in raw C.  Eliminating the
    // vertex constraint first identifies x0/x1, so their rows share a column
    // in the updated C*E and must form one edge-stage component.
    builder.append_row({0, 1}, {1.0, -1.0}, 1.0,
                       meta("c0", "vertex-star"));
    builder.append_row({0, 2}, {1.0, -1.0}, 2.0,
                       meta("feature_jet", "edge-a"));
    builder.append_row({1, 3}, {1.0, -1.0}, -1.0,
                       meta("feature_jet", "edge-b"));
    const ConstraintSystem3D system = builder.finalize();
    const std::vector<ConstraintBlock3D> physical_blocks{
        block(ConstraintBlockKind3D::Edge, "edge-a", {1}),
        block(ConstraintBlockKind3D::Vertex, "vertex-star", {0}),
        block(ConstraintBlockKind3D::Edge, "edge-b", {2})};

    AffineReductionOptions3D staged_options;
    staged_options.schedule =
        AffineEliminationSchedule3D::StagedVertexThenEdge;
    const AffineReduction3D staged = affine_eliminate_local_svd_3d(
        system, physical_blocks, staged_options);
    require(staged.records().size() == 2,
            "staged reduction retains a strict vertex/edge boundary");
    require(staged.records()[0].kind == ConstraintBlockKind3D::Vertex
                && staged.records()[0].row_ids
                    == std::vector<Eigen::Index>({0}),
            "vertex/T-star component is eliminated by itself first");
    require(staged.records()[1].kind == ConstraintBlockKind3D::Edge
                && staged.records()[1].row_ids
                    == std::vector<Eigen::Index>({1, 2})
                && staged.records()[1].source_keys.size() == 2,
            "edge closure is formed from support in the updated C*E");
    require(staged.particular_residual_linf() < 3.0e-13,
            "staged affine particular satisfies Cp=d");
    require(staged.homogeneous_residual_linf() < 3.0e-13,
            "staged affine basis satisfies CE=0");

    // A global dense SVD is test-only and supplies the subspace oracle.
    const Eigen::MatrixXd dense_constraints(system.C);
    Eigen::JacobiSVD<Eigen::MatrixXd> oracle(
        dense_constraints, Eigen::ComputeFullV);
    const Eigen::VectorXd singular = oracle.singularValues();
    const double cutoff = singular[0] * 1.0e-11;
    Eigen::Index rank = 0;
    while (rank < singular.size() && singular[rank] > cutoff)
        ++rank;
    const Eigen::MatrixXd oracle_nullspace =
        oracle.matrixV().rightCols(4 - rank);
    const Eigen::MatrixXd staged_nullspace(staged.homogeneous_base());
    require(staged_nullspace.cols() == oracle_nullspace.cols(),
            "staged and global SVD nullities agree");
    require((staged_nullspace * staged_nullspace.transpose()
             - oracle_nullspace * oracle_nullspace.transpose())
                    .cwiseAbs().maxCoeff()
                < 3.0e-12,
            "staged local SVD spans the global-SVD oracle nullspace");

    // The compatibility schedule deliberately keeps the historical raw-C
    // premerge behavior available: this same graph collapses to one block.
    const AffineReduction3D legacy = affine_eliminate_local_svd_3d(
        system, physical_blocks);
    require(legacy.records().size() == 1,
            "legacy schedule remains backward compatible");
}

void test_staged_incompatibility_expands_current_component_and_retries()
{
    SparseConstraintBuilder3D builder(3);
    constexpr double weak_direction = 5.0e-3;
    constexpr double hidden_coupling = 9.0e-15;
    constexpr double neighboring_scale = 1.1e-13;
    // Solved together, these four rows have the exact solution (0,1,0).
    // The first block alone loses its weak direction at the deliberately
    // coarse rank cutoff.  The second block is connected only through an
    // entry below the initial support threshold; a strict current-C*E closure
    // on incompatibility exposes that connection and makes the retry exact.
    builder.append_row({0}, {1.0}, 0.0, meta("feature_jet", "weak"));
    builder.append_row({0, 1}, {1.0, weak_direction}, weak_direction,
                       meta("feature_jet", "weak"));
    builder.append_row({1, 2}, {hidden_coupling, neighboring_scale},
                       hidden_coupling,
                       meta("feature_jet", "neighbor"));
    builder.append_row({2}, {neighboring_scale}, 0.0,
                       meta("feature_jet", "neighbor"));

    AffineReductionOptions3D options;
    options.schedule =
        AffineEliminationSchedule3D::StagedVertexThenEdge;
    options.zero_row_tolerance = 1.0e-13;
    options.sparse_prune_tolerance = 1.0e-14;
    options.relative_rank_tolerance = 2.0e-2;
    const AffineReduction3D reduction = affine_eliminate_local_svd_3d(
        builder.finalize(),
        {block(ConstraintBlockKind3D::Edge, "weak", {0, 1}),
         block(ConstraintBlockKind3D::Edge, "neighbor", {2, 3})},
        options);
    require(reduction.records().size() == 1
                && reduction.records()[0].source_keys.size() == 2
                && reduction.records()[0].merge_retry_count == 1,
            "incompatible local block expands by current C*E and retries");
    require(reduction.reduced_size() == 0
                && (reduction.particular_base()
                    - (Eigen::Vector3d() << 0.0, 1.0, 0.0).finished())
                           .norm()
                    < 5.0e-12,
            "expanded component retry recovers the exact affine solution");
    require(reduction.particular_residual_linf() < 1.0e-14
                && reduction.homogeneous_residual_linf() == 0.0,
            "component retry preserves Cp and CE invariants");
}

void test_local_freedom_exhaustion_and_later_zero_rows()
{
    SparseConstraintBuilder3D builder(2);
    builder.append_row({0}, {1.0}, 1.0, meta("c0", "pin"));
    builder.append_row({1}, {1.0}, -2.0, meta("c0", "pin"));
    builder.append_row({0, 1}, {1.0, 1.0}, -1.0,
                       meta("c0", "already-satisfied"));
    AffineReductionOptions3D options;
    options.merge_shared_support_blocks = false;
    const AffineReduction3D reduction = affine_eliminate_local_svd_3d(
        builder.finalize(),
        {block(ConstraintBlockKind3D::Vertex, "pin", {0, 1}),
         block(ConstraintBlockKind3D::Edge, "already-satisfied", {2})},
        options);
    require(reduction.reduced_size() == 0,
            "a compatible full-rank local block may consume all freedom");
    require(reduction.records()[0].local_nullity == 0
                && reduction.records()[0].remaining == 0
                && reduction.records()[1].active_rows == 0,
            "exhaustion and later satisfied zero rows are diagnosed");
    require((reduction.particular_base()
             - (Eigen::Vector2d() << 1.0, -2.0).finished())
                    .norm()
                < 2.0e-14,
            "exhausted local solve retains its unique affine particular");

    SparseConstraintBuilder3D impossible(1);
    impossible.append_row({0}, {1.0}, 0.0, meta("c0", "first"));
    impossible.append_row({0}, {1.0}, 1.0, meta("c0", "second"));
    bool threw = false;
    try {
        (void)affine_eliminate_local_svd_3d(
            impossible.finalize(),
            {block(ConstraintBlockKind3D::Vertex, "first", {0}),
             block(ConstraintBlockKind3D::Edge, "second", {1})},
            options);
    } catch (const ConstraintIncompatibility3D& error) {
        threw = error.row_id() == 1;
    }
    require(threw,
            "a later nonzero RHS after freedom exhaustion is explicit");
}

} // namespace

int main()
{
    try {
        test_patch_layout_round_trip();
        test_affine_invariants_lifts_and_support();
        test_global_svd_oracle_subspace();
        test_zero_row_and_incompatibility_are_explicit();
        test_support_union_find_and_block_order();
        test_staged_vertex_then_current_edge_support_closure();
        test_staged_incompatibility_expands_current_component_and_retries();
        test_local_freedom_exhaustion_and_later_zero_rows();
        std::cout << "topology_affine_reduction_3d_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "topology_affine_reduction_3d_test: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
