#include "topology_density_constraints_3d.hpp"
#include "csv_rfc4180.hpp"

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
using kfbim::app3d::GeometryKind3D;
using kfbim::app3d::KnownDirichletValueGradientHessian3D;
using kfbim::app3d::KnownDirichletValueGradientHessianCallback3D;
using kfbim::app3d::NativeDensityField3D;
using kfbim::app3d::NativeDensityReductionBackend3D;
using kfbim::app3d::NativeDensitySeamCoupling3D;
using kfbim::app3d::NativeNurbsDensityOptions3D;
using kfbim::app3d::NativeNurbsDensitySpace3D;
using kfbim::app3d::TopologyDensityConstraintPlan3D;
using kfbim::app3d::TopologyDirichletFeatureConstraintOptions3D;
using kfbim::app3d::TopologyDirichletFeatureConstraintPlan3D;
using kfbim::app3d::TopologyFeatureJumpJetOperators3D;
using kfbim::app3d::make_native_nurbs_surface_3d;
using kfbim::app3d::make_topology_dirichlet_feature_constraint_plan_3d;
using kfbim::app3d::make_topology_feature_jump_jet_operators_3d;
using kfbim::app3d::make_topology_density_constraint_plan_3d;
using kfbim::app3d::physical_smooth_sheet_ids_3d;
using kfbim::app3d::rfc4180_csv_field;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
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
    const std::string& context)
{
    NativeNurbsDensitySpace3D value = make_base(
        geometry, NativeDensityField3D::ValueTrace, 6);
    NativeNurbsDensitySpace3D normal = make_base(
        geometry, NativeDensityField3D::NormalTrace, 6);
    kfbim::app3d::NativeFeatureEdgeJumpJetOptions3D feature_options;
    feature_options.mortar_gauss_order = 5;
    const TopologyFeatureJumpJetOperators3D feature =
        make_topology_feature_jump_jet_operators_3d(
            value, normal, feature_options);

    require(feature.feature_edge_count() == expected_feature_edges
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
            require(meta.kind == "feature_jet"
                        && meta.connection == edge.connection
                        && meta.mode >= 0 && meta.mode <= 3
                        && (meta.side == "first" || meta.side == "second")
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
    require(maximum_value_support <= 16 && maximum_normal_support <= 8,
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
        test_dirichlet_affine_feature_constraints_on_right_angle_planes();
        test_rfc4180_csv_field();
        std::cout << "topology density constraint tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "topology density constraint test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
