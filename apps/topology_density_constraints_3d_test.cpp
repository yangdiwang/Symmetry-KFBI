#include "topology_density_constraints_3d.hpp"
#include "csv_rfc4180.hpp"
#include "native_nurbs_surface_transform_3d.hpp"

#include <Eigen/Core>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using kfbim::app3d::ConstraintBlockKind3D;
using kfbim::app3d::ConstraintBlock3D;
using kfbim::app3d::ConstraintSystem3D;
using kfbim::app3d::AffineEliminationSchedule3D;
using kfbim::app3d::AffineReduction3D;
using kfbim::app3d::AffineReductionOptions3D;
using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::KnownDirichletValueGradientHessian3D;
using kfbim::app3d::KnownDirichletValueGradientHessianCallback3D;
using kfbim::app3d::NativeDensityField3D;
using kfbim::app3d::NativeDensityReductionBackend3D;
using kfbim::app3d::NativeDensitySeamCoupling3D;
using kfbim::app3d::NativeNurbsDensityOptions3D;
using kfbim::app3d::NativeNurbsDensitySpace3D;
using kfbim::app3d::NativeNurbsSurface3D;
using kfbim::app3d::RigidTransform3D;
using kfbim::app3d::TopologyDensityConstraintPlan3D;
using kfbim::app3d::TopologyDirichletFeatureConstraintOptions3D;
using kfbim::app3d::TopologyDirichletFeatureConstraintPlan3D;
using kfbim::app3d::TopologyFeatureJumpJetOperators3D;
using kfbim::app3d::TopologyNeumannFeatureC1Form3D;
using kfbim::app3d::SparseMatrixCSR3D;
using kfbim::app3d::affine_eliminate_local_svd_3d;
using kfbim::app3d::make_native_nurbs_surface_3d;
using kfbim::app3d::make_topology_dirichlet_feature_constraint_plan_3d;
using kfbim::app3d::make_topology_feature_jump_jet_operators_3d;
using kfbim::app3d::make_topology_density_constraint_plan_3d;
using kfbim::app3d::physical_smooth_sheet_ids_3d;
using kfbim::app3d::rfc4180_csv_field;
using kfbim::app3d::select_topology_dirichlet_unisolvent_constraints_3d;
using kfbim::app3d::transform_native_nurbs_surface_3d;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

std::pair<ConstraintSystem3D, std::vector<ConstraintBlock3D>>
combine_constraint_plans(
    const ConstraintSystem3D& first,
    const std::vector<ConstraintBlock3D>& first_blocks,
    const ConstraintSystem3D& second,
    const std::vector<ConstraintBlock3D>& second_blocks)
{
    require(first.C.cols() == second.C.cols(),
            "combined constraint plans use the same coefficient space");
    const Eigen::Index first_rows = first.C.rows();
    ConstraintSystem3D combined;
    combined.C.resize(first.C.rows() + second.C.rows(), first.C.cols());
    combined.d.resize(first.d.size() + second.d.size());
    combined.d.head(first.d.size()) = first.d;
    combined.d.tail(second.d.size()) = second.d;
    combined.meta = first.meta;
    combined.meta.insert(
        combined.meta.end(), second.meta.begin(), second.meta.end());
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(
        first.C.nonZeros() + second.C.nonZeros()));
    for (Eigen::Index row = 0; row < first.C.rows(); ++row) {
        for (SparseMatrixCSR3D::InnerIterator entry(first.C, row);
             entry; ++entry) {
            triplets.emplace_back(row, entry.col(), entry.value());
        }
    }
    for (Eigen::Index row = 0; row < second.C.rows(); ++row) {
        for (SparseMatrixCSR3D::InnerIterator entry(second.C, row);
             entry; ++entry) {
            triplets.emplace_back(
                first_rows + row, entry.col(), entry.value());
        }
    }
    combined.C.setFromTriplets(triplets.begin(), triplets.end());
    combined.C.makeCompressed();
    combined.validate();

    std::vector<ConstraintBlock3D> blocks = first_blocks;
    for (ConstraintBlock3D block : second_blocks) {
        for (Eigen::Index& row : block.row_ids)
            row += first_rows;
        blocks.push_back(std::move(block));
    }
    return {std::move(combined), std::move(blocks)};
}

NativeNurbsDensitySpace3D make_base(GeometryKind3D geometry,
                                    NativeDensityField3D field,
                                    int coefficients = 6)
{
    NativeNurbsDensityOptions3D options;
    options.field = field;
    options.coefficients_per_direction = coefficients;
    options.reduction_backend =
        NativeDensityReductionBackend3D::TopologyBase;
    return NativeNurbsDensitySpace3D(
        make_native_nurbs_surface_3d(geometry), options);
}

NativeNurbsDensitySpace3D make_base(NativeNurbsSurface3D surface,
                                    NativeDensityField3D field,
                                    int coefficients)
{
    NativeNurbsDensityOptions3D options;
    options.field = field;
    options.coefficients_per_direction = coefficients;
    options.reduction_backend =
        NativeDensityReductionBackend3D::TopologyBase;
    return NativeNurbsDensitySpace3D(std::move(surface), options);
}

void check_base_only(const NativeNurbsDensitySpace3D& space,
                     const std::string& context)
{
    require(space.options().reduction_backend
                != NativeDensityReductionBackend3D::Legacy,
            context + " selects the topology base backend");
    require(space.reduced_coefficient_count()
                == space.c0_coefficient_count()
                && space.continuous_coefficient_count()
                       == space.c0_coefficient_count(),
            context + " has an identity base reduction");
    require(space.partial_c0_constraint_count() == 0
                && space.partial_c0_constraint_rank() == 0
                && space.weak_c1_constraint_count() == 0
                && space.weak_c1_constraint_rank() == 0,
            context + " skips legacy dense nullspaces");
    require(space.uses_identity_reduction(),
            context + " stores the base identity implicitly");
    const Eigen::VectorXd identity_probe = Eigen::VectorXd::LinSpaced(
        space.c0_coefficient_count(), -0.75, 0.5);
    require((space.expand_reduced(identity_probe) - identity_probe)
                    .lpNorm<Eigen::Infinity>() < 1.0e-15,
            context + " implicit reduction acts exactly as identity");
    require(space.trace_c0_design().cols() == space.c0_coefficient_count()
                && space.trace_c0_design().rows()
                       == static_cast<Eigen::Index>(space.trace_points().size())
                && space.surface_area() > 0.0,
            context + " retains base trace quadrature");

    const Eigen::Vector3d direction =
        space.geometry(0, 0.37, 0.41).tangents.col(0).normalized();
    const Eigen::RowVectorXd dense =
        space.c0_physical_directional_derivative_row(
            0, 0.37, 0.41, direction);
    const auto sparse =
        space.c0_physical_directional_derivative_stencil(
            0, 0.37, 0.41, direction);
    Eigen::RowVectorXd expanded = Eigen::RowVectorXd::Zero(dense.size());
    for (int q = 0; q < sparse.count; ++q)
        expanded[sparse.indices[static_cast<std::size_t>(q)]] +=
            sparse.weights[static_cast<std::size_t>(q)];
    require((dense - expanded).cwiseAbs().maxCoeff() < 2.0e-13,
            context + " sparse physical derivative matches dense reference");
}

void check_constraint_algebra(const NativeNurbsDensitySpace3D& space,
                              const TopologyDensityConstraintPlan3D& plan,
                              const std::string& context)
{
    plan.system.validate();
    require(plan.system.C.cols() == space.c0_coefficient_count()
                && plan.system.C.rows() > 0
                && plan.system.C.isCompressed(),
            context + " has a compressed sparse constraint matrix");
    const Eigen::VectorXd residual =
        plan.system.C * Eigen::VectorXd::Ones(space.c0_coefficient_count());
    require(residual.size() > 0
                && residual.lpNorm<Eigen::Infinity>() < 2.0e-11,
            context + " annihilates the constant density");

    Eigen::Index maximum_support = 0;
    for (Eigen::Index row = 0; row < plan.system.C.rows(); ++row) {
        Eigen::Index support = 0;
        for (kfbim::app3d::SparseMatrixCSR3D::InnerIterator entry(
                 plan.system.C, row);
             entry; ++entry) {
            require(std::isfinite(entry.value())
                        && std::abs(entry.value()) > 0.0,
                    context + " stores only finite nonzeros");
            ++support;
        }
        require(support > 0, context + " has no retained zero row");
        maximum_support = std::max(maximum_support, support);
    }
    require(maximum_support <= 32,
            context + " rows retain local cubic support");

    bool seen_edge = false;
    std::vector<int> ownership(static_cast<std::size_t>(plan.system.C.rows()), 0);
    for (const auto& block : plan.blocks) {
        if (block.kind == ConstraintBlockKind3D::Edge)
            seen_edge = true;
        else
            require(!seen_edge, context + " orders every vertex block first");
        require(!block.row_ids.empty() && !block.key.empty(),
                context + " block has rows and a stable key");
        for (Eigen::Index row : block.row_ids) {
            require(row >= 0 && row < plan.system.C.rows(),
                    context + " block row is in range");
            ++ownership[static_cast<std::size_t>(row)];
        }
    }
    require(std::all_of(ownership.begin(), ownership.end(),
                        [](int count) { return count == 1; }),
            context + " assigns every row to exactly one block");
    require(plan.vertex_block_count > 0 && plan.edge_block_count > 0,
            context + " creates both star and edge-interior blocks");
}

void check_feature_policy(const NativeNurbsDensitySpace3D& space,
                          const TopologyDensityConstraintPlan3D& plan,
                          const std::string& context)
{
    const auto& connections = space.surface().geometric_connections;
    for (const auto& meta : plan.system.meta) {
        require(meta.connection >= 0
                    && meta.connection < static_cast<int>(connections.size()),
                context + " metadata records its geometric connection");
        const bool feature = !connections[
            static_cast<std::size_t>(meta.connection)].g1;
        if (feature) {
            require(space.options().field == NativeDensityField3D::ValueTrace
                        && meta.kind == "c0",
                    context + " feature rows are ValueTrace C0 only");
        }
        if (space.options().field == NativeDensityField3D::ValueTrace) {
            require(meta.sheet_id < 0,
                    context + " retains one joint Neumann physical star");
        }
    }
    if (space.options().field == NativeDensityField3D::NormalTrace) {
        for (const auto& seam : space.seams()) {
            if (seam.coupling != NativeDensitySeamCoupling3D::Broken)
                continue;
            require(std::none_of(
                        plan.system.meta.begin(), plan.system.meta.end(),
                        [&](const auto& meta) {
                            return meta.connection == seam.connection;
                        }),
                    context + " never constrains a feature normal trace");
        }
    }
}

void check_physical_sheet_vertex_blocks(
    GeometryKind3D geometry,
    int expected_sheet_count,
    bool expect_multi_sheet_vertex,
    const std::string& context)
{
    auto surface = make_native_nurbs_surface_3d(geometry);
    const std::vector<int> sheet_ids =
        physical_smooth_sheet_ids_3d(surface);
    require(sheet_ids.size() == surface.patches.size(),
            context + " returns one physical sheet id per patch");
    const std::set<int> unique_sheets(sheet_ids.begin(), sheet_ids.end());
    require(static_cast<int>(unique_sheets.size()) == expected_sheet_count
                && *unique_sheets.begin() == 0
                && *unique_sheets.rbegin() == expected_sheet_count - 1,
            context + " assigns compact deterministic sheet ids");
    for (const auto& connection : surface.geometric_connections) {
        const int first = sheet_ids[static_cast<std::size_t>(
            connection.first.patch)];
        const int second = sheet_ids[static_cast<std::size_t>(
            connection.second.patch)];
        require(connection.g1 ? first == second : first != second,
                context + " sheet graph follows G1 rather than C0 adjacency");
    }

    // Connection traversal order must not affect the topology-only id.
    std::reverse(surface.geometric_connections.begin(),
                 surface.geometric_connections.end());
    require(physical_smooth_sheet_ids_3d(surface) == sheet_ids,
            context + " sheet ids are invariant to connection ordering");

    const NativeNurbsDensitySpace3D normal = make_base(
        geometry, NativeDensityField3D::NormalTrace);
    const TopologyDensityConstraintPlan3D plan =
        make_topology_density_constraint_plan_3d(normal);
    std::map<std::array<double, 3>, std::set<int>> sheets_at_vertex;
    for (const auto& meta : plan.system.meta) {
        require(meta.sheet_id >= 0,
                context + " every Dirichlet NormalTrace row owns a sheet");
        if (meta.star)
            sheets_at_vertex[*meta.star].insert(meta.sheet_id);
    }
    bool found_multi_sheet_vertex = false;
    for (const auto& [point, sheets] : sheets_at_vertex) {
        (void)point;
        found_multi_sheet_vertex = found_multi_sheet_vertex
            || sheets.size() > 1;
    }
    require(found_multi_sheet_vertex == expect_multi_sheet_vertex,
            context + " detects the expected physical feature vertices");

    for (const auto& block : plan.blocks) {
        if (block.kind != ConstraintBlockKind3D::Vertex)
            continue;
        std::set<int> block_sheets;
        for (Eigen::Index row : block.row_ids) {
            block_sheets.insert(plan.system.meta[
                static_cast<std::size_t>(row)].sheet_id);
        }
        require(block_sheets.size() == 1,
                context + " never combines different G1 sheets in a star");
        const std::string suffix = "|sheet_id="
            + std::to_string(*block_sheets.begin());
        require(block.key.find(suffix) != std::string::npos,
                context + " writes the physical sheet id into the star key");
    }
}

void test_default_legacy_is_unchanged()
{
    NativeNurbsDensityOptions3D options;
    options.field = NativeDensityField3D::ValueTrace;
    options.coefficients_per_direction = 6;
    require(options.reduction_backend
                == NativeDensityReductionBackend3D::Legacy,
            "legacy reduction remains the default");
    NativeNurbsDensitySpace3D legacy(
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism), options);
    require(legacy.partial_c0_constraint_count() > 0
                && legacy.weak_c1_constraint_count() > 0
                && legacy.reduced_coefficient_count()
                       < legacy.c0_coefficient_count(),
            "default path still builds the legacy dense reductions");
}

void test_cylinder()
{
    check_physical_sheet_vertex_blocks(
        GeometryKind3D::HollowCylinder, 4, true, "cylinder");
    for (NativeDensityField3D field : {NativeDensityField3D::ValueTrace,
                                       NativeDensityField3D::NormalTrace}) {
        NativeNurbsDensitySpace3D space =
            make_base(GeometryKind3D::HollowCylinder, field);
        const std::string context = field == NativeDensityField3D::ValueTrace
            ? "cylinder ValueTrace"
            : "cylinder NormalTrace";
        check_base_only(space, context);
        const TopologyDensityConstraintPlan3D plan =
            make_topology_density_constraint_plan_3d(space);
        check_constraint_algebra(space, plan, context);
        check_feature_policy(space, plan, context);
        require(plan.c0_row_count == 0,
                context + " has no redundant full-edge C0 moment rows");
        require(plan.smooth_c1_row_count > 0,
                context + " retains smooth common-conormal moments");
    }
}

void test_l_prism()
{
    check_physical_sheet_vertex_blocks(
        GeometryKind3D::LPrism, 8, false, "L-prism");
    NativeNurbsDensitySpace3D value =
        make_base(GeometryKind3D::LPrism, NativeDensityField3D::ValueTrace);
    check_base_only(value, "L-prism ValueTrace");
    const TopologyDensityConstraintPlan3D value_plan =
        make_topology_density_constraint_plan_3d(value);
    check_constraint_algebra(value, value_plan, "L-prism ValueTrace");
    check_feature_policy(value, value_plan, "L-prism ValueTrace");
    require(value_plan.c0_row_count > 0
                && value_plan.smooth_c1_row_count > 0,
            "L-prism ValueTrace has partial C0 and smooth C1 moments");

    int partial_connection = -1;
    for (const auto& seam : value.seams()) {
        if (!seam.full_edge_pair) {
            partial_connection = seam.connection;
            break;
        }
    }
    require(partial_connection >= 0,
            "L-prism exposes a long-edge/short-edge connection");
    std::set<std::array<double, 2>> partial_cells;
    int partial_rows = 0;
    for (const auto& meta : value_plan.system.meta) {
        if (meta.connection != partial_connection)
            continue;
        require(meta.kind == "c0",
                "L-prism feature long-short connection has C0 only");
        partial_cells.insert(meta.cell);
        ++partial_rows;
    }
    require(partial_cells.size() >= 3
                && partial_rows
                       == static_cast<int>(4 * partial_cells.size()),
            "L-prism long-short overlay uses every knot cell and P0..P3");

    NativeNurbsDensitySpace3D normal =
        make_base(GeometryKind3D::LPrism, NativeDensityField3D::NormalTrace);
    check_base_only(normal, "L-prism NormalTrace");
    const TopologyDensityConstraintPlan3D normal_plan =
        make_topology_density_constraint_plan_3d(normal);
    check_constraint_algebra(normal, normal_plan, "L-prism NormalTrace");
    check_feature_policy(normal, normal_plan, "L-prism NormalTrace");
    require(normal_plan.c0_row_count == 0
                && normal_plan.smooth_c1_row_count > 0,
            "L-prism NormalTrace keeps features fully broken");
}

Eigen::VectorXd fit_affine_c0_trace(
    const NativeNurbsDensitySpace3D& space,
    const Eigen::Vector3d& gradient,
    double offset,
    const std::string& context)
{
    const int samples = space.coefficients_per_direction() + 1;
    const int rows = space.patch_count() * samples * samples;
    Eigen::MatrixXd design(rows, space.c0_coefficient_count());
    Eigen::VectorXd values(rows);
    int row = 0;
    for (int patch = 0; patch < space.patch_count(); ++patch) {
        for (int i = 0; i < samples; ++i) {
            const double u = (static_cast<double>(i) + 0.37) / samples;
            for (int j = 0; j < samples; ++j) {
                const double v = (static_cast<double>(j) + 0.61) / samples;
                const auto point = space.geometry(patch, u, v);
                design.row(row) = space.c0_basis_row(patch, u, v);
                values[row] = space.options().field
                        == NativeDensityField3D::ValueTrace
                    ? offset + gradient.dot(point.point)
                    : gradient.dot(point.normal);
                ++row;
            }
        }
    }
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(design);
    const Eigen::VectorXd coefficients = qr.solve(values);
    const double fit_residual =
        (design * coefficients - values).lpNorm<Eigen::Infinity>();
    require(coefficients.allFinite() && fit_residual < 5.0e-8,
            context + " represents the ambient affine trace; residual="
                + std::to_string(fit_residual));
    return coefficients;
}

void check_cellwise_feature_jump_jet(
    GeometryKind3D geometry,
    int expected_feature_edges,
    const std::string& context,
    TopologyNeumannFeatureC1Form3D formulation =
        TopologyNeumannFeatureC1Form3D::ConormalSolved)
{
    NativeNurbsDensitySpace3D value = make_base(
        geometry, NativeDensityField3D::ValueTrace, 6);
    NativeNurbsDensitySpace3D normal = make_base(
        geometry, NativeDensityField3D::NormalTrace, 6);
    kfbim::app3d::NativeFeatureEdgeJumpJetOptions3D feature_options;
    feature_options.mortar_gauss_order = 5;
    const TopologyFeatureJumpJetOperators3D feature =
        make_topology_feature_jump_jet_operators_3d(
            value, normal, feature_options, {}, formulation);

    require(feature.formulation == formulation
                && feature.feature_edge_count() == expected_feature_edges
                && feature.constraint_count() > 0
                && feature.value_matrix.isCompressed()
                && feature.normal_target_matrix.isCompressed(),
            context + " builds compressed feature operators");
    require(feature.value_matrix.cols() == value.c0_coefficient_count()
                && feature.normal_target_matrix.cols()
                       == normal.c0_coefficient_count()
                && feature.meta.size()
                       == static_cast<std::size_t>(
                           feature.constraint_count()),
            context + " feature operator shapes are consistent");

    int next_row = 0;
    for (const auto& edge : feature.edges) {
        require(edge.first_row == next_row && edge.row_count > 0,
                context + " feature connections own contiguous rows");
        std::set<std::array<double, 2>> cells;
        std::set<std::tuple<std::array<double, 2>, int, std::string>> tests;
        for (int local = 0; local < edge.row_count; ++local) {
            const auto& meta = feature.meta[static_cast<std::size_t>(
                edge.first_row + local)];
            const bool metadata_matches = formulation
                    == TopologyNeumannFeatureC1Form3D::ConormalSolved
                ? (meta.kind == "feature_jet"
                   && (meta.side == "first" || meta.side == "second"))
                : (meta.kind == "feature_ambient_gradient"
                   && (meta.side == "transverse_0"
                       || meta.side == "transverse_1"));
            require(metadata_matches
                        && meta.connection == edge.connection
                        && meta.mode >= 0 && meta.mode <= 3
                        && meta.sheet_id < 0,
                    context + " feature row metadata is complete");
            cells.insert(meta.cell);
            tests.emplace(meta.cell, meta.mode, meta.side);
        }
        require(cells.size() >= 3
                    && edge.test_function_count
                           == static_cast<int>(4 * cells.size())
                    && edge.row_count
                           == static_cast<int>(8 * cells.size())
                    && tests.size()
                           == static_cast<std::size_t>(edge.row_count),
                context + " uses P0..P3 separately on every overlay cell");

        const double first_begin = cells.begin()->at(0);
        const double last_end = cells.rbegin()->at(1);
        for (int local = 0; local < edge.row_count; ++local) {
            const auto& meta = feature.meta[static_cast<std::size_t>(
                edge.first_row + local)];
            const bool endpoint_cell =
                std::abs(meta.cell[0] - first_begin) < 1.0e-14
                || std::abs(meta.cell[1] - last_end) < 1.0e-14;
            require(static_cast<bool>(meta.star) == endpoint_cell,
                    context + " assigns endpoint-cell feature rows to stars");
        }
        next_row += edge.row_count;
    }
    require(next_row == feature.constraint_count(),
            context + " accounts for every cellwise feature row");

    bool seen_vertex = false;
    bool seen_edge = false;
    std::vector<int> ownership(
        static_cast<std::size_t>(feature.constraint_count()), 0);
    for (const auto& block : feature.blocks) {
        seen_vertex = seen_vertex
            || block.kind == ConstraintBlockKind3D::Vertex;
        seen_edge = seen_edge || block.kind == ConstraintBlockKind3D::Edge;
        for (Eigen::Index row : block.row_ids)
            ++ownership[static_cast<std::size_t>(row)];
    }
    require(seen_vertex && seen_edge
                && std::all_of(ownership.begin(), ownership.end(),
                               [](int count) { return count == 1; }),
            context + " partitions feature rows into vertex then edge blocks");

    Eigen::Index maximum_value_support = 0;
    Eigen::Index maximum_normal_support = 0;
    for (Eigen::Index row = 0; row < feature.value_matrix.rows(); ++row) {
        Eigen::Index value_support = 0;
        Eigen::Index normal_support = 0;
        double squared_norm = 0.0;
        for (kfbim::app3d::SparseMatrixCSR3D::InnerIterator entry(
                 feature.value_matrix, row); entry; ++entry) {
            ++value_support;
            squared_norm += entry.value() * entry.value();
        }
        for (kfbim::app3d::SparseMatrixCSR3D::InnerIterator entry(
                 feature.normal_target_matrix, row); entry; ++entry) {
            ++normal_support;
            squared_norm += entry.value() * entry.value();
        }
        maximum_value_support = std::max(
            maximum_value_support, value_support);
        maximum_normal_support = std::max(
            maximum_normal_support, normal_support);
        require(std::abs(std::sqrt(squared_norm) - 1.0) < 3.0e-12,
                context + " preserves unit combined row normalization");
    }
    const Eigen::Index expected_maximum_value_support = formulation
            == TopologyNeumannFeatureC1Form3D::ConormalSolved
        ? 16 : 32;
    require(maximum_value_support <= expected_maximum_value_support
                && maximum_normal_support <= 8,
            context + " retains knot-cell-local cubic support");
    require(feature.constant_value_residual < 2.0e-11
                && feature.maximum_point_mismatch < 2.0e-11
                && feature.minimum_abs_dihedral_sine > 1.0e-3,
            context + " feature geometry and constant diagnostics pass");

    const Eigen::Vector3d gradient =
        geometry == GeometryKind3D::HollowCylinder
        ? Eigen::Vector3d(0.0, 0.0, 0.31)
        : Eigen::Vector3d(0.31, -0.22, 0.17);
    const Eigen::VectorXd value_c0 = fit_affine_c0_trace(
        value, gradient, 0.37, context + " value");
    const Eigen::VectorXd normal_c0 = fit_affine_c0_trace(
        normal, gradient, 0.0, context + " normal");
    require(feature.residual(value_c0, normal_c0)
                    .lpNorm<Eigen::Infinity>() < 8.0e-8,
            context + " ambient affine gradient satisfies every cell mode");
    const auto bound = feature.bind_normal_target(normal_c0);
    bound.validate();
    require((bound.d - feature.normal_target_matrix * normal_c0)
                    .lpNorm<Eigen::Infinity>() < 2.0e-14,
            context + " binds the known Neumann target exactly");
}

void test_cellwise_feature_jump_jets()
{
    check_cellwise_feature_jump_jet(
        GeometryKind3D::HollowCylinder, 16, "cylinder");
    check_cellwise_feature_jump_jet(
        GeometryKind3D::LPrism, 22, "L-prism");
    check_cellwise_feature_jump_jet(
        GeometryKind3D::UPrism, 32, "U-prism ambient gradient",
        TopologyNeumannFeatureC1Form3D::AmbientGradient);
}

Eigen::Index numerical_row_rank(const Eigen::MatrixXd& matrix,
                                double relative_tolerance = 1.0e-11)
{
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(matrix.transpose());
    qr.setThreshold(relative_tolerance);
    return qr.rank();
}

void test_u_prism_neumann_ambient_gradient_row_space()
{
    NativeNurbsDensitySpace3D value = make_base(
        GeometryKind3D::UPrism, NativeDensityField3D::ValueTrace, 4);
    NativeNurbsDensitySpace3D normal = make_base(
        GeometryKind3D::UPrism, NativeDensityField3D::NormalTrace, 4);
    const TopologyDensityConstraintPlan3D topology =
        make_topology_density_constraint_plan_3d(value);
    AffineReductionOptions3D reduction_options;
    reduction_options.schedule =
        AffineEliminationSchedule3D::StagedVertexThenEdge;
    const AffineReduction3D topology_reduction =
        affine_eliminate_local_svd_3d(
            topology.system, topology.blocks, reduction_options);

    kfbim::app3d::NativeFeatureEdgeJumpJetOptions3D feature_options;
    feature_options.mortar_gauss_order = 5;
    const TopologyFeatureJumpJetOperators3D solved =
        make_topology_feature_jump_jet_operators_3d(
            value, normal, feature_options, {},
            TopologyNeumannFeatureC1Form3D::ConormalSolved);
    const TopologyFeatureJumpJetOperators3D ambient =
        make_topology_feature_jump_jet_operators_3d(
            value, normal, feature_options, {},
            TopologyNeumannFeatureC1Form3D::AmbientGradient);

    const SparseMatrixCSR3D solved_restricted =
        solved.value_matrix * topology_reduction.homogeneous_base();
    const SparseMatrixCSR3D ambient_restricted =
        ambient.value_matrix * topology_reduction.homogeneous_base();
    const Eigen::MatrixXd solved_dense(solved_restricted);
    const Eigen::MatrixXd ambient_dense(ambient_restricted);
    Eigen::MatrixXd stacked(
        solved_dense.rows() + ambient_dense.rows(), solved_dense.cols());
    stacked.topRows(solved_dense.rows()) = solved_dense;
    stacked.bottomRows(ambient_dense.rows()) = ambient_dense;
    const Eigen::Index solved_rank = numerical_row_rank(solved_dense);
    const Eigen::Index ambient_rank = numerical_row_rank(ambient_dense);
    const Eigen::Index stacked_rank = numerical_row_rank(stacked);
    require(solved_rank > 0 && solved_rank == ambient_rank
                && stacked_rank == solved_rank,
            "U-prism direct ambient gradients and solved conormals have "
            "the same C0-quotient row space; solved/ambient/union ranks="
                + std::to_string(solved_rank) + "/"
                + std::to_string(ambient_rank) + "/"
                + std::to_string(stacked_rank));

    const Eigen::Vector3d gradient(0.31, -0.22, 0.17);
    const Eigen::VectorXd value_c0 = fit_affine_c0_trace(
        value, gradient, 0.37, "U-prism ambient row-space value");
    const Eigen::VectorXd normal_c0 = fit_affine_c0_trace(
        normal, gradient, 0.0, "U-prism ambient row-space normal");
    require(ambient.residual(value_c0, normal_c0)
                    .lpNorm<Eigen::Infinity>() < 8.0e-8,
            "U-prism direct ambient C1 reproduces one common affine "
            "world gradient");
    require(ambient.residual(value_c0, -normal_c0)
                    .lpNorm<Eigen::Infinity>() > 1.0e-3,
            "U-prism direct ambient C1 detects a reversed Neumann sign");
}

void test_u_prism_neumann_ambient_gradient_rigid_covariance()
{
    const NativeNurbsSurface3D source =
        make_native_nurbs_surface_3d(GeometryKind3D::UPrism);
    constexpr double pi = 3.141592653589793238462643383279502884;
    const RigidTransform3D transform = RigidTransform3D::from_axis_angle(
        {1.0, 2.0, 3.0}, 17.0 * pi / 180.0,
        {0.07, -0.07, 0.02}, {0.137, -0.083, 0.061});
    NativeNurbsDensitySpace3D value_source = make_base(
        source, NativeDensityField3D::ValueTrace, 4);
    NativeNurbsDensitySpace3D normal_source = make_base(
        source, NativeDensityField3D::NormalTrace, 4);
    NativeNurbsDensitySpace3D value_moved = make_base(
        transform_native_nurbs_surface_3d(source, transform),
        NativeDensityField3D::ValueTrace, 4);
    NativeNurbsDensitySpace3D normal_moved = make_base(
        transform_native_nurbs_surface_3d(source, transform),
        NativeDensityField3D::NormalTrace, 4);

    kfbim::app3d::NativeFeatureEdgeJumpJetOptions3D feature_options;
    feature_options.mortar_gauss_order = 5;
    const TopologyFeatureJumpJetOperators3D original =
        make_topology_feature_jump_jet_operators_3d(
            value_source, normal_source, feature_options, {},
            TopologyNeumannFeatureC1Form3D::AmbientGradient);
    const TopologyFeatureJumpJetOperators3D moved =
        make_topology_feature_jump_jet_operators_3d(
            value_moved, normal_moved, feature_options, {},
            TopologyNeumannFeatureC1Form3D::AmbientGradient);
    require(original.feature_edge_count() == moved.feature_edge_count()
                && original.constraint_count() == moved.constraint_count(),
            "rigid U-prism preserves ambient-gradient feature catalog");
    require((Eigen::MatrixXd(original.value_matrix)
             - Eigen::MatrixXd(moved.value_matrix))
                    .cwiseAbs().maxCoeff() < 2.0e-10
                && (Eigen::MatrixXd(original.normal_target_matrix)
                    - Eigen::MatrixXd(moved.normal_target_matrix))
                           .cwiseAbs().maxCoeff() < 2.0e-10,
            "ambient-gradient coefficient operators are rigid-motion "
            "covariant");

    const Eigen::Vector3d gradient(0.31, -0.22, 0.17);
    const Eigen::Vector3d moved_gradient =
        transform.forward_vector(gradient);
    const Eigen::VectorXd moved_value_c0 = fit_affine_c0_trace(
        value_moved, moved_gradient, 0.37,
        "rigid U-prism ambient value");
    const Eigen::VectorXd moved_normal_c0 = fit_affine_c0_trace(
        normal_moved, moved_gradient, 0.0,
        "rigid U-prism ambient normal");
    require(moved.residual(moved_value_c0, moved_normal_c0)
                    .lpNorm<Eigen::Infinity>() < 8.0e-8,
            "rigid U-prism reproduces the rotated common ambient gradient");
}

void test_dirichlet_affine_feature_constraints_on_right_angle_planes()
{
    NativeNurbsDensitySpace3D normal = make_base(
        GeometryKind3D::LPrism, NativeDensityField3D::NormalTrace, 6);

    // Merely constructing the feature plan is harmless unless the caller
    // opts in.  In particular no callback is needed and NormalTrace feature
    // seams retain their broken base-space policy.
    const TopologyDirichletFeatureConstraintPlan3D disabled =
        make_topology_dirichlet_feature_constraint_plan_3d(
            normal, KnownDirichletValueGradientHessianCallback3D{});
    require(disabled.constraint_count() == 0
                && disabled.feature_edge_count() == 0
                && disabled.system.C.cols()
                       == normal.c0_coefficient_count(),
            "Dirichlet feature constraints are opt-in");
    for (const auto& seam : normal.seams()) {
        if (!normal.surface().geometric_connections[
                 static_cast<std::size_t>(seam.connection)].g1) {
            require(seam.coupling == NativeDensitySeamCoupling3D::Broken,
                    "Dirichlet feature opt-in leaves NormalTrace broken");
        }
    }

    const Eigen::Vector3d gradient(0.31, -0.22, 0.17);
    constexpr double offset = 0.37;
    const KnownDirichletValueGradientHessianCallback3D known_dirichlet =
        [gradient](int, double, double, const Eigen::Vector3d& point,
                   const Eigen::Vector3d&) {
            KnownDirichletValueGradientHessian3D data;
            data.value = offset + gradient.dot(point);
            data.ambient_gradient = gradient;
            data.ambient_hessian.setZero();
            return data;
        };
    TopologyDirichletFeatureConstraintOptions3D options;
    options.enabled = true;
    options.mortar_gauss_order = 5;
    const TopologyDirichletFeatureConstraintPlan3D feature =
        make_topology_dirichlet_feature_constraint_plan_3d(
            normal, known_dirichlet, options);
    feature.system.validate();
    require(feature.feature_edge_count() == 22
                && feature.constraint_count() > 0
                && feature.system.C.cols()
                       == normal.c0_coefficient_count()
                && feature.system.C.isCompressed(),
            "right-angle planar L-prism builds sparse affine J1 rows");
    require(feature.minimum_abs_dihedral_sine > 1.0 - 2.0e-12
                && feature.maximum_point_mismatch < 2.0e-11
                && feature.maximum_tangential_derivative_mismatch < 2.0e-12,
            "analytic planar feature geometry is exactly ninety degrees");
    require(feature.row_scalings.size() == feature.system.C.rows(),
            "Dirichlet feature rows retain their normalization diagnostics");

    bool seen_vertex = false;
    bool seen_edge = false;
    std::vector<int> ownership(
        static_cast<std::size_t>(feature.constraint_count()), 0);
    for (const auto& block : feature.blocks) {
        seen_vertex = seen_vertex
            || block.kind == ConstraintBlockKind3D::Vertex;
        seen_edge = seen_edge || block.kind == ConstraintBlockKind3D::Edge;
        for (Eigen::Index row : block.row_ids)
            ++ownership[static_cast<std::size_t>(row)];
    }
    require(seen_vertex && seen_edge
                && std::all_of(ownership.begin(), ownership.end(),
                               [](int count) { return count == 1; }),
            "Dirichlet feature rows form vertex-before-edge blocks");
    for (Eigen::Index row = 0; row < feature.system.C.rows(); ++row) {
        double squared_norm = 0.0;
        for (kfbim::app3d::SparseMatrixCSR3D::InnerIterator entry(
                 feature.system.C, row); entry; ++entry) {
            squared_norm += entry.value() * entry.value();
        }
        require(std::abs(std::sqrt(squared_norm) - 1.0) < 3.0e-12,
                "Dirichlet affine feature row has unit normal-space norm");
        require(feature.system.meta[static_cast<std::size_t>(row)].kind
                    == "dirichlet_feature_jet",
                "Dirichlet affine feature metadata is explicit");
    }

    // For u(x)=offset+gradient.x, the exact two-sided normal density is
    // J1=gradient.n.  The mortar RHS comes only from the analytic g_D
    // callback, and the independently represented exact J1 satisfies it.
    const Eigen::VectorXd normal_c0 = fit_affine_c0_trace(
        normal, gradient, 0.0, "right-angle Dirichlet normal trace");
    const double residual =
        (feature.system.C * normal_c0 - feature.system.d)
            .lpNorm<Eigen::Infinity>();
    require(residual < 8.0e-8,
            "ambient affine gradient satisfies affine J1 feature moments; "
            "residual=" + std::to_string(residual));
}

void test_u_prism_dirichlet_feature_vertex_blocks()
{
    NativeNurbsDensitySpace3D normal = make_base(
        GeometryKind3D::UPrism, NativeDensityField3D::NormalTrace, 6);
    const Eigen::Vector3d gradient(0.31, -0.22, 0.17);
    constexpr double offset = 0.37;
    const KnownDirichletValueGradientHessianCallback3D known_dirichlet =
        [gradient](int, double, double, const Eigen::Vector3d& point,
                   const Eigen::Vector3d&) {
            KnownDirichletValueGradientHessian3D data;
            data.value = offset + gradient.dot(point);
            data.ambient_gradient = gradient;
            data.ambient_hessian.setZero();
            return data;
        };
    TopologyDirichletFeatureConstraintOptions3D options;
    options.enabled = true;
    const TopologyDirichletFeatureConstraintPlan3D feature =
        make_topology_dirichlet_feature_constraint_plan_3d(
            normal, known_dirichlet, options);
    feature.system.validate();
    require(feature.feature_edge_count() == 32
                && feature.constraint_count() > 0
                && feature.maximum_value_mismatch < 2.0e-12,
            "U-prism covers every C0 feature interval with compatible data");

    bool seen_vertex = false;
    bool seen_edge = false;
    bool seen_multi_connection_star = false;
    for (const auto& block : feature.blocks) {
        seen_vertex = seen_vertex
            || block.kind == ConstraintBlockKind3D::Vertex;
        seen_edge = seen_edge || block.kind == ConstraintBlockKind3D::Edge;
        if (block.kind != ConstraintBlockKind3D::Vertex)
            continue;
        std::set<int> connections;
        for (Eigen::Index row : block.row_ids) {
            connections.insert(feature.system.meta[
                static_cast<std::size_t>(row)].connection);
        }
        seen_multi_connection_star = seen_multi_connection_star
            || connections.size() >= 3;
    }
    require(seen_vertex && seen_edge && seen_multi_connection_star,
            "U-prism feature moments form edge-interior and multi-edge "
            "Vertex/T-star blocks");

    const Eigen::VectorXd normal_c0 = fit_affine_c0_trace(
        normal, gradient, 0.0, "U-prism Dirichlet normal trace");
    require((feature.system.C * normal_c0 - feature.system.d)
                    .lpNorm<Eigen::Infinity>() < 8.0e-8,
            "U-prism ambient affine J1 satisfies edge and vertex-star rows");

    const KnownDirichletValueGradientHessianCallback3D discontinuous_value =
        [gradient](int patch, double, double, const Eigen::Vector3d& point,
                   const Eigen::Vector3d&) {
            KnownDirichletValueGradientHessian3D data;
            data.value = offset + gradient.dot(point)
                       + 0.01 * static_cast<double>(patch);
            data.ambient_gradient = gradient;
            data.ambient_hessian.setZero();
            return data;
        };
    bool rejected_discontinuous_value = false;
    try {
        (void)make_topology_dirichlet_feature_constraint_plan_3d(
            normal, discontinuous_value, options);
    } catch (const std::runtime_error& error) {
        rejected_discontinuous_value =
            std::string(error.what()).find("incompatible feature-edge values")
            != std::string::npos;
    }
    require(rejected_discontinuous_value,
            "Dirichlet feature setup rejects patch-dependent g_D values");
}

void test_dirichlet_unisolvent_selection_is_vertex_first_and_rhs_blind()
{
    TopologyDirichletFeatureConstraintPlan3D candidates;
    candidates.system.C.resize(3, 2);
    const std::vector<Eigen::Triplet<double>> triplets{
        {0, 0, 1.0},   // Vertex row retained first.
        {1, 0, 10.0},  // Larger but dependent Edge row must be discarded.
        {2, 1, 1.0}};  // Independent Edge increment.
    candidates.system.C.setFromTriplets(triplets.begin(), triplets.end());
    candidates.system.C.makeCompressed();
    candidates.system.d = Eigen::Vector3d(0.25, -7.0, 0.5);
    candidates.system.meta.resize(3);
    candidates.system.meta[0].kind = "dirichlet_feature_jet";
    candidates.system.meta[0].connection = 0;
    candidates.system.meta[0].mode = 0;
    candidates.system.meta[0].star = std::array<double, 3>{{0.0, 0.0, 0.0}};
    candidates.system.meta[1].kind = "dirichlet_feature_jet";
    candidates.system.meta[1].connection = 1;
    candidates.system.meta[1].mode = 0;
    candidates.system.meta[2].kind = "dirichlet_feature_jet";
    candidates.system.meta[2].connection = 2;
    candidates.system.meta[2].mode = 0;
    candidates.system.validate();

    ConstraintBlock3D vertex;
    vertex.kind = ConstraintBlockKind3D::Vertex;
    vertex.key = "vertex";
    vertex.row_ids = {0};
    ConstraintBlock3D edge;
    edge.kind = ConstraintBlockKind3D::Edge;
    edge.key = "edge";
    edge.row_ids = {1, 2};
    candidates.blocks = {vertex, edge};

    SparseMatrixCSR3D topology_homogeneous(2, 2);
    topology_homogeneous.setIdentity();
    topology_homogeneous.makeCompressed();
    const auto selected =
        select_topology_dirichlet_unisolvent_constraints_3d(
            candidates, topology_homogeneous);
    const std::vector<Eigen::Index> expected_rows{0, 2};
    require(selected.retained_candidate_rows == expected_rows
                && selected.candidate_constraint_count == 3
                && selected.constraint_count() == 2
                && selected.discarded_constraint_count() == 1
                && selected.retained_vertex_rows == 1
                && selected.retained_edge_rows == 1,
            "unisolvent selection retains Vertex rows before dependent Edge "
            "rows");

    TopologyDirichletFeatureConstraintPlan3D changed_rhs = candidates;
    changed_rhs.system.d = Eigen::Vector3d(900.0, -0.125, 41.0);
    const auto selected_changed_rhs =
        select_topology_dirichlet_unisolvent_constraints_3d(
            changed_rhs, topology_homogeneous);
    require(selected_changed_rhs.retained_candidate_rows
                    == selected.retained_candidate_rows,
            "unisolvent row identities depend only on C*G, never on d");
    for (Eigen::Index row = 0;
         row < selected_changed_rhs.system.C.rows(); ++row) {
        const Eigen::Index source =
            selected_changed_rhs.retained_candidate_rows[
                static_cast<std::size_t>(row)];
        require(selected_changed_rhs.system.d[row]
                        == changed_rhs.system.d[source],
                "changed analytic RHS entries are copied exactly after "
                "RHS-blind selection");
    }
}

void test_u_prism_dirichlet_unisolvent_affine_reduction()
{
    const KnownDirichletValueGradientHessianCallback3D harmonic =
        [](int, double, double, const Eigen::Vector3d& point,
           const Eigen::Vector3d&) {
            constexpr double ax = 0.35;
            constexpr double by = 0.21;
            constexpr double cz = 0.28;
            const double exponential = std::exp(ax * point.x());
            const double cos_y = std::cos(by * point.y());
            const double sin_y = std::sin(by * point.y());
            const double cos_z = std::cos(cz * point.z());
            const double sin_z = std::sin(cz * point.z());
            KnownDirichletValueGradientHessian3D data;
            data.value = exponential * cos_y * cos_z;
            data.ambient_gradient = {
                ax * data.value,
                -by * exponential * sin_y * cos_z,
                -cz * exponential * cos_y * sin_z};
            data.ambient_hessian.setZero();
            return data;
        };

    struct Expected {
        int ncoef;
        int topology_size;
        int candidate_rows;
        int candidate_vertex_rows;
        int candidate_edge_rows;
        int feature_rank;
        int retained_vertex_rows;
        int retained_edge_rows;
        int final_size;
    };
    const std::array<Expected, 2> expected{{
        {4, 224, 256, 256, 0, 160, 160, 0, 64},
        {6, 552, 768, 512, 256, 272, 272, 0, 280}}};
    for (const Expected& item : expected) {
        NativeNurbsDensitySpace3D normal = make_base(
            GeometryKind3D::UPrism, NativeDensityField3D::NormalTrace,
            item.ncoef);
        const TopologyDensityConstraintPlan3D topology =
            make_topology_density_constraint_plan_3d(normal);
        AffineReductionOptions3D reduction_options;
        reduction_options.schedule =
            AffineEliminationSchedule3D::StagedVertexThenEdge;
        const AffineReduction3D topology_reduction =
            affine_eliminate_local_svd_3d(
                topology.system, topology.blocks, reduction_options);
        require(topology_reduction.reduced_size() == item.topology_size,
                "U-prism topology-only dimension matches the baseline");

        TopologyDirichletFeatureConstraintOptions3D feature_options;
        feature_options.enabled = true;
        const TopologyDirichletFeatureConstraintPlan3D candidates =
            make_topology_dirichlet_feature_constraint_plan_3d(
                normal, harmonic, feature_options);
        int candidate_vertex_rows = 0;
        int candidate_edge_rows = 0;
        for (const ConstraintBlock3D& block : candidates.blocks) {
            if (block.kind == ConstraintBlockKind3D::Vertex)
                candidate_vertex_rows += static_cast<int>(block.row_ids.size());
            else
                candidate_edge_rows += static_cast<int>(block.row_ids.size());
        }
        require(candidate_vertex_rows == item.candidate_vertex_rows
                    && candidate_edge_rows == item.candidate_edge_rows,
                "U-prism candidate catalog has the expected explicit "
                "Vertex/Edge split: ncoef=" + std::to_string(item.ncoef));
        const auto selected =
            select_topology_dirichlet_unisolvent_constraints_3d(
                candidates, topology_reduction.homogeneous_base(),
                reduction_options.relative_rank_tolerance);
        require(candidates.constraint_count() == item.candidate_rows
                    && selected.candidate_constraint_count
                           == item.candidate_rows
                    && selected.constraint_count() == item.feature_rank
                    && selected.discarded_constraint_count()
                           == item.candidate_rows - item.feature_rank,
                "U-prism retains the expected independent analytic feature "
                "rows: ncoef=" + std::to_string(item.ncoef)
                + ", candidates="
                + std::to_string(candidates.constraint_count())
                + ", retained="
                + std::to_string(selected.constraint_count()));
        require(selected.retained_vertex_rows
                        + selected.retained_edge_rows == item.feature_rank
                    && selected.retained_vertex_rows
                           == item.retained_vertex_rows
                    && selected.retained_edge_rows
                           == item.retained_edge_rows,
                "U-prism retains the expected Neumann-style Vertex/T-star "
                "then edge increment: "
                "candidate_vertex="
                + std::to_string(candidate_vertex_rows)
                + ", candidate_edge="
                + std::to_string(candidate_edge_rows)
                + ", retained_vertex="
                + std::to_string(selected.retained_vertex_rows)
                + ", retained_edge="
                + std::to_string(selected.retained_edge_rows)
                + ", ncoef=" + std::to_string(item.ncoef));
        for (Eigen::Index row = 0; row < selected.system.C.rows(); ++row) {
            const Eigen::Index source = selected.retained_candidate_rows[
                static_cast<std::size_t>(row)];
            require(selected.system.d[row] == candidates.system.d[source],
                    "unisolvent selection copies analytic RHS values exactly");
        }

        auto combined = combine_constraint_plans(
            topology.system, topology.blocks,
            selected.system, selected.blocks);
        const AffineReduction3D reduction =
            affine_eliminate_local_svd_3d(
                std::move(combined.first), std::move(combined.second),
                reduction_options);
        require(reduction.reduced_size() == item.final_size
                    && topology_reduction.reduced_size()
                           - reduction.reduced_size()
                           == item.feature_rank,
                "U-prism analytic feature rows reduce the iteration space "
                "by their restricted rank");
        require(reduction.particular_residual_linf() < 2.0e-9
                    && reduction.homogeneous_residual_linf() < 2.0e-9,
                "U-prism retained analytic affine constraints close to "
                "roundoff");

        const Eigen::VectorXd full_candidate_residual =
            candidates.system.C * reduction.particular_base()
            - candidates.system.d;
        require(full_candidate_residual.lpNorm<Eigen::Infinity>() > 1.0e-9,
                "non-polynomial analytic data expose discarded dependent "
                "mortar defects instead of fitting them");
    }
}

void test_rfc4180_csv_field()
{
    require(rfc4180_csv_field("plain") == "plain",
            "plain CSV fields remain unquoted");
    require(rfc4180_csv_field("vertex:1,2,3")
                == "\"vertex:1,2,3\"",
            "comma-containing topology keys are quoted");
    require(rfc4180_csv_field("a\"b") == "\"a\"\"b\"",
            "embedded CSV quotes are doubled");
    require(rfc4180_csv_field("a\r\nb") == "\"a\r\nb\"",
            "CSV CRLF content is quoted without alteration");
}

} // namespace

int main()
{
    try {
        test_default_legacy_is_unchanged();
        test_cylinder();
        test_l_prism();
        test_cellwise_feature_jump_jets();
        test_u_prism_neumann_ambient_gradient_row_space();
        test_u_prism_neumann_ambient_gradient_rigid_covariance();
        test_dirichlet_affine_feature_constraints_on_right_angle_planes();
        test_u_prism_dirichlet_feature_vertex_blocks();
        test_dirichlet_unisolvent_selection_is_vertex_first_and_rhs_blind();
        test_u_prism_dirichlet_unisolvent_affine_reduction();
        test_rfc4180_csv_field();
        std::cout << "topology density constraint tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "topology density constraint test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
