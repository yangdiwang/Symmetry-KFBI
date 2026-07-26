#include <apps/harmonic_cauchy_fit_3d.hpp>

#define main neumann_exterior_zero_trace_3d_application_main
#include "neumann_exterior_zero_trace_3d.cpp"
#undef main

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

Eigen::VectorXd gather_route_fixture(
    const Eigen::VectorXd& values,
    const std::vector<int>& ids)
{
    Eigen::VectorXd result(static_cast<int>(ids.size()));
    for (int q = 0; q < result.size(); ++q)
        result[q] = values[ids[static_cast<std::size_t>(q)]];
    return result;
}

void require_matrix_exact(const Eigen::MatrixXd& actual,
                          const Eigen::MatrixXd& expected,
                          const std::string& message)
{
    require(actual.rows() == expected.rows()
                && actual.cols() == expected.cols()
                && (actual.array() == expected.array()).all(),
            message);
}

struct LPrismFitFixture3D {
    static constexpr int N = 32;
    double h = kBoxSide / static_cast<double>(N);
    GeometryBundle geometry = make_geometry(
        GeometryKind::LPrism, h, app3d::RigidTransform3D());
    SurfaceDofCloud cloud = app3d::make_native_surface_dofs_3d(
        geometry.native_surface, h);
    app3d::SurfaceNonG1EdgeNeighborhoodSet3D neighborhoods =
        app3d::build_surface_non_g1_edge_neighborhoods_3d(
            geometry.native_surface, cloud, h);
};

struct DetailedNeumannRouteFixture3D {
    static constexpr int N = 32;
    double h = kBoxSide / static_cast<double>(N);
    CartesianGrid3D grid{{kBoxMin, kBoxMin, kBoxMin}, {h, h, h},
                         {N, N, N}, DofLayout3D::Node};
    GeometryBundle geometry = make_geometry(
        GeometryKind::LPrism, h, app3d::RigidTransform3D());
    std::shared_ptr<const geometry3d::NurbsCartesianDomain3D> domain =
        std::make_shared<const geometry3d::NurbsCartesianDomain3D>(
            grid, geometry.native_surface.geometry_model());
    GridPair3D grid_pair{grid, geometry.correction_interface,
                         geometry.crossing_interface, domain};
    SurfaceDofCloud cloud = app3d::make_native_surface_dofs_3d(
        geometry.native_surface, h);
    app3d::SurfaceNonG1EdgeNeighborhoodSet3D neighborhoods =
        app3d::build_surface_non_g1_edge_neighborhoods_3d(
            geometry.native_surface, cloud, h);

    std::unique_ptr<PanelCenterHarmonicJetKFBI3D> make_pipeline(
        app3d::HarmonicCauchyRoute3D route)
    {
        auto fit = app3d::HarmonicCauchyFit3D::build(
            geometry.native_surface, cloud, neighborhoods, h, route);
        return std::make_unique<PanelCenterHarmonicJetKFBI3D>(
            grid, grid_pair, geometry.native_surface,
            geometry.correction_triangles, geometry.geometry_triangles,
            cloud, std::move(fit), false, true);
    }
};

void test_harmonic_cauchy_route_names_are_exact()
{
    require(std::string(harmonic_cauchy_route_name_3d(
                app3d::HarmonicCauchyRoute3D::G1ValueG1Normal))
                == "g1_value_g1_normal",
            "G1 route name is stable");
    require(std::string(harmonic_cauchy_route_name_3d(
                app3d::HarmonicCauchyRoute3D::DirectCrossFaceValue))
                == "direct_cross_face_value",
            "direct route name is stable");
    require(std::string(harmonic_cauchy_route_name_3d(
                app3d::HarmonicCauchyRoute3D::EdgeReconstructedValue))
                == "edge_reconstructed_value",
            "shared-edge route name is stable");
}

void test_lprism_routes_have_exact_second_level_structure()
{
    LPrismFitFixture3D fixture;
    const auto g1 = app3d::HarmonicCauchyFit3D::build(
        fixture.geometry.native_surface, fixture.cloud,
        fixture.neighborhoods, fixture.h,
        app3d::HarmonicCauchyRoute3D::G1ValueG1Normal);
    const auto direct = app3d::HarmonicCauchyFit3D::build(
        fixture.geometry.native_surface, fixture.cloud,
        fixture.neighborhoods, fixture.h,
        app3d::HarmonicCauchyRoute3D::DirectCrossFaceValue);
    const auto edge = app3d::HarmonicCauchyFit3D::build(
        fixture.geometry.native_surface, fixture.cloud,
        fixture.neighborhoods, fixture.h,
        app3d::HarmonicCauchyRoute3D::EdgeReconstructedValue);

    require(g1.edge_maps().empty(), "G1 route has no global edge maps");
    require(!edge.edge_maps().empty(),
            "shared-edge route owns global first-level edge maps");
    require(g1.surface_maps().size() == fixture.cloud.dofs.size()
                && direct.surface_maps().size() == fixture.cloud.dofs.size()
                && edge.surface_maps().size() == fixture.cloud.dofs.size(),
            "all routes own one surface map per L-prism center");

    bool saw_balanced_cross_face = false;
    bool saw_outside_two_h = false;
    for (int center = 0;
         center < static_cast<int>(fixture.cloud.dofs.size()); ++center) {
        const auto& gm = g1.surface_maps()[static_cast<std::size_t>(center)];
        const auto& dm = direct.surface_maps()[static_cast<std::size_t>(center)];
        const auto& em = edge.surface_maps()[static_cast<std::size_t>(center)];
        require(gm.edge_point_ids.empty() && gm.M_edge.cols() == 0,
                "G1 surface maps have no edge IDs or edge columns");
        require(dm.normal_ids == gm.normal_ids
                    && dm.edge_point_ids.empty() && dm.M_edge.cols() == 0,
                "direct route preserves G1 normals and has no edge rows");
        require(em.value_ids == gm.value_ids && em.normal_ids == gm.normal_ids,
                "shared-edge route preserves ordered G1 value/normal IDs");
        if (dm.nearest_edge_distance_over_h <= 2.0
            && dm.value_sector_counts.size() >= 2) {
            const auto bounds = std::minmax_element(
                dm.value_sector_counts.begin(), dm.value_sector_counts.end());
            require(*bounds.second - *bounds.first <= 1,
                    "direct cross-face value sectors are balanced");
            saw_balanced_cross_face = true;
        }
        if (em.nearest_edge_distance_over_h > 2.0) {
            require(em.edge_point_ids.empty() && em.M_edge.cols() == 0,
                    "outside 2h shared-edge map uses the literal two-term path");
            require(em.value_ids == gm.value_ids
                        && em.normal_ids == gm.normal_ids,
                    "outside 2h shared-edge map keeps exact G1 IDs");
            require_matrix_exact(em.M_value, gm.M_value,
                                 "outside 2h value matrix is exactly G1");
            require_matrix_exact(em.M_normal, gm.M_normal,
                                 "outside 2h normal matrix is exactly G1");
            saw_outside_two_h = true;
        }
    }
    require(saw_balanced_cross_face,
            "L-prism direct route exercises a balanced cross-face center");
    require(saw_outside_two_h,
            "L-prism shared-edge route exercises an outside-2h G1 center");

    bool reused_from_both_faces = false;
    for (const auto& edge_map : edge.edge_maps()) {
        const int point_id = edge_map.point.id;
        bool first_face = false;
        bool second_face = false;
        for (int center = 0;
             center < static_cast<int>(fixture.cloud.dofs.size()); ++center) {
            const auto& map = edge.surface_maps()[static_cast<std::size_t>(center)];
            if (std::find(map.edge_point_ids.begin(), map.edge_point_ids.end(),
                          point_id) == map.edge_point_ids.end())
                continue;
            const int patch = fixture.cloud.dofs[
                static_cast<std::size_t>(center)].patch_id;
            first_face = first_face
                || std::find(edge_map.point.sector_patch_ids[0].begin(),
                             edge_map.point.sector_patch_ids[0].end(), patch)
                       != edge_map.point.sector_patch_ids[0].end();
            second_face = second_face
                || std::find(edge_map.point.sector_patch_ids[1].begin(),
                             edge_map.point.sector_patch_ids[1].end(), patch)
                       != edge_map.point.sector_patch_ids[1].end();
        }
        reused_from_both_faces = reused_from_both_faces
            || (first_face && second_face);
    }
    require(reused_from_both_faces,
            "one global edge-point ID is reused by both incident faces");

    Eigen::VectorXd mu(static_cast<int>(fixture.cloud.dofs.size()));
    Eigen::VectorXd eta(mu.size());
    for (int q = 0; q < mu.size(); ++q) {
        mu[q] = std::sin(0.017 * static_cast<double>(q + 1));
        eta[q] = std::cos(0.023 * static_cast<double>(q + 2));
    }
    const auto edge_applied = edge.apply(mu, eta);
    for (int center = 0; center < edge_applied.coefficients.rows(); ++center) {
        const auto& map = edge.surface_maps()[static_cast<std::size_t>(center)];
        if (map.nearest_edge_distance_over_h <= 2.0)
            continue;
        const Eigen::VectorXd expected =
            map.M_value * gather_route_fixture(mu, map.value_ids)
            + map.M_normal * gather_route_fixture(eta, map.normal_ids);
        require((edge_applied.coefficients.row(center).transpose() - expected)
                        .lpNorm<Eigen::Infinity>() == 0.0,
                "outside 2h apply executes the exact two-term G1 product");
        break;
    }
}

void test_legacy_g1_fixture_is_preserved_by_public_fit()
{
    LPrismFitFixture3D fixture;
    const auto fit = app3d::HarmonicCauchyFit3D::build_legacy(
        fixture.geometry.native_surface, fixture.cloud, fixture.h,
        app3d::LegacySurfaceCauchyPolicy3D::G1Nearest,
        kCauchyPolynomialDegree, kCauchyValueNeighborCount,
        kCauchyDerivativeNeighborCount);
    require(fit.legacy_policy()
                == app3d::LegacySurfaceCauchyPolicy3D::G1Nearest,
            "public legacy fit records the requested G1 policy");
    for (int center = 0;
         center < static_cast<int>(fixture.cloud.dofs.size()); ++center) {
        const auto& map = fit.surface_maps()[static_cast<std::size_t>(center)];
        require(map.value_ids == app3d::nearest_g1_cauchy_dofs(
                    fixture.geometry.native_surface, fixture.cloud, center,
                    kCauchyValueNeighborCount)
                    && map.normal_ids == app3d::nearest_g1_cauchy_dofs(
                        fixture.geometry.native_surface, fixture.cloud, center,
                        kCauchyDerivativeNeighborCount),
                "public G1 fit preserves captured ordered fixture IDs at "
                    + std::to_string(center));
    }
    const auto summary = fit.legacy_summary();
    require(summary.has_value()
                && summary->value_count == kCauchyValueNeighborCount
                && summary->normal_count == kCauchyDerivativeNeighborCount
                && summary->value_count_min == kCauchyValueNeighborCount
                && summary->value_count_max == kCauchyValueNeighborCount
                && summary->normal_count_min == kCauchyDerivativeNeighborCount
                && summary->normal_count_max == kCauchyDerivativeNeighborCount,
            "public G1 fit preserves the legacy aggregate count diagnostics");
}

void test_legacy_n16_short_sectors_preserve_requested_and_actual_counts()
{
    constexpr int N = 16;
    const double h = kBoxSide / static_cast<double>(N);
    const GeometryBundle geometry = make_geometry(
        GeometryKind::LPrism, h, app3d::RigidTransform3D());
    const SurfaceDofCloud cloud = app3d::make_native_surface_dofs_3d(
        geometry.native_surface, h);
    const auto fit = app3d::HarmonicCauchyFit3D::build_legacy(
        geometry.native_surface, cloud, h,
        app3d::LegacySurfaceCauchyPolicy3D::G1Nearest,
        kCauchyPolynomialDegree, kCauchyValueNeighborCount,
        kCauchyDerivativeNeighborCount);
    const auto summary = fit.legacy_summary();
    require(summary.has_value()
                && summary->value_count == kCauchyValueNeighborCount
                && summary->normal_count == kCauchyDerivativeNeighborCount,
            "N=16 legacy summary preserves the requested 48/28 counts");
    require(summary->value_count_min < summary->value_count_max
                && summary->value_count_max == kCauchyValueNeighborCount
                && summary->normal_count_min == kCauchyDerivativeNeighborCount
                && summary->normal_count_max == kCauchyDerivativeNeighborCount,
            "N=16 legacy summary preserves truncated actual min/max counts");
    for (int center = 0; center < static_cast<int>(cloud.dofs.size()); ++center) {
        const auto& map = fit.surface_maps()[static_cast<std::size_t>(center)];
        require(map.value_ids == app3d::nearest_g1_cauchy_dofs(
                    geometry.native_surface, cloud, center,
                    kCauchyValueNeighborCount)
                    && map.normal_ids == app3d::nearest_g1_cauchy_dofs(
                        geometry.native_surface, cloud, center,
                        kCauchyDerivativeNeighborCount),
                "N=16 legacy fit preserves truncated ordered IDs");
    }
}

void test_owner_pipeline_and_bordered_operator_split_mu_eta()
{
    LPrismFitFixture3D fixture;
    CartesianGrid3D grid({kBoxMin, kBoxMin, kBoxMin},
                         {fixture.h, fixture.h, fixture.h},
                         {LPrismFitFixture3D::N, LPrismFitFixture3D::N,
                          LPrismFitFixture3D::N},
                         DofLayout3D::Node);
    const auto domain = std::make_shared<const
        geometry3d::NurbsCartesianDomain3D>(
            grid, fixture.geometry.native_surface.geometry_model());
    GridPair3D grid_pair(grid, fixture.geometry.correction_interface,
                         fixture.geometry.crossing_interface, domain);
    auto fit = app3d::HarmonicCauchyFit3D::build(
        fixture.geometry.native_surface, fixture.cloud,
        fixture.neighborhoods, fixture.h,
        app3d::HarmonicCauchyRoute3D::EdgeReconstructedValue);
    PanelCenterHarmonicJetKFBI3D pipeline(
        grid, grid_pair, fixture.geometry.native_surface,
        fixture.geometry.correction_triangles,
        fixture.geometry.geometry_triangles, fixture.cloud,
        std::move(fit), false, true);

    const int size = pipeline.surface_size();
    Eigen::VectorXd mu(size);
    Eigen::VectorXd eta(size);
    for (int q = 0; q < size; ++q) {
        mu[q] = std::sin(0.031 * static_cast<double>(q + 1));
        eta[q] = std::cos(0.047 * static_cast<double>(q + 3));
    }
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(size);
    const auto trace = [&](const Eigen::VectorXd& value_jump,
                           const Eigen::VectorXd& normal_jump) {
        const HarmonicJetField3D field =
            pipeline.evaluate(value_jump, normal_jump);
        return pipeline.exterior_trace(
            field, value_jump, normal_jump,
            ExteriorValueRestrictMode3D::JointTricubicCrossingOwner);
    };

    const auto cauchy_before = pipeline.cauchy_fit().audit();
    const std::size_t owner_queries_before =
        pipeline.restrict_owner_geometry_query_count();
    const std::uint64_t owner_fingerprint_before =
        pipeline.restrict_owner_workload_fingerprint();
    const std::uint64_t owner_digest_before =
        pipeline.restrict_owner_output_digest();

    const Eigen::VectorXd combined = trace(mu, eta);
    const Eigen::VectorXd split = trace(mu, zero) + trace(zero, eta);
    require((combined - split).lpNorm<Eigen::Infinity>() < 5.0e-11,
            "owner-enabled T(mu,eta) splits into value and normal routes");

    ExteriorZeroTraceOperator3D op(
        pipeline, ExteriorValueRestrictMode3D::JointTricubicCrossingOwner);
    Eigen::VectorXd unknown(size + 1);
    unknown.head(size) = mu;
    unknown[size] = 0.375;
    Eigen::VectorXd applied;
    op.apply(unknown, applied);
    const Eigen::VectorXd residual = applied - op.right_hand_side(eta);
    Eigen::VectorXd expected(size + 1);
    expected.head(size) = combined.array() + unknown[size];
    expected[size] = surface_weighted_mean(pipeline.surface(), mu);
    require((residual - expected).lpNorm<Eigen::Infinity>() < 5.0e-11,
            "bordered operator has the exact sign and weighted-mean row");

    const auto& cauchy_after = pipeline.cauchy_fit().audit();
    require(cauchy_after.geometry_query_count
                    == cauchy_before.geometry_query_count
                && cauchy_after.svd_factorization_count
                    == cauchy_before.svd_factorization_count
                && cauchy_after.fingerprint == cauchy_before.fingerprint,
            "Cauchy query/SVD counters and fingerprint are immutable at runtime");
    require(pipeline.restrict_owner_geometry_query_count()
                    == owner_queries_before
                && pipeline.restrict_owner_workload_fingerprint()
                    == owner_fingerprint_before
                && pipeline.restrict_owner_output_digest()
                    == owner_digest_before,
            "owner query count and fingerprints are immutable at runtime");
}

void test_common_neumann_rhs_uses_native_parameters_and_surface_weights()
{
    LPrismFitFixture3D fixture;
    const Eigen::VectorXd first = make_common_neumann_augmented_rhs_3d(
        fixture.geometry.native_surface, fixture.cloud);
    const Eigen::VectorXd second = make_common_neumann_augmented_rhs_3d(
        fixture.geometry.native_surface, fixture.cloud);
    const Eigen::VectorXd third = make_common_neumann_augmented_rhs_3d(
        fixture.geometry.native_surface, fixture.cloud);
    const int size = static_cast<int>(fixture.cloud.dofs.size());
    require(first.size() == size + 1 && first[size] == 0.0,
            "common Neumann RHS has one exactly-zero augmented tail");
    require((first.array() == second.array()).all()
                && (first.array() == third.array()).all(),
            "all three routes receive the identical deterministic RHS");
    require(std::abs(surface_weighted_mean(
                fixture.cloud, first.head(size))) <= 5.0e-13,
            "common Neumann RHS is surface-weighted demeaned");
    double weighted_square_sum = 0.0;
    double weight_sum = 0.0;
    for (int q = 0; q < size; ++q) {
        const double weight =
            fixture.cloud.dofs[static_cast<std::size_t>(q)].weight;
        weighted_square_sum += weight * first[q] * first[q];
        weight_sum += weight;
    }
    require(std::abs(std::sqrt(weighted_square_sum / weight_sum) - 1.0)
                <= 5.0e-13,
            "common Neumann RHS has unit surface-weighted RMS");

    Eigen::VectorXd expected(size);
    const double pi = std::acos(-1.0);
    for (int q = 0; q < size; ++q) {
        const SurfaceDof& dof =
            fixture.cloud.dofs[static_cast<std::size_t>(q)];
        const auto& patch = fixture.geometry.native_surface.patches[
            static_cast<std::size_t>(dof.patch_id)];
        const double uhat = (dof.u - patch.domain_start_u())
            / (patch.domain_end_u() - patch.domain_start_u());
        const double vhat = (dof.v - patch.domain_start_v())
            / (patch.domain_end_v() - patch.domain_start_v());
        expected[q] = std::sin(2.0 * pi * uhat
                               + 0.37 * static_cast<double>(dof.patch_id + 1))
            + 0.5 * std::cos(2.0 * pi * vhat
                            - 0.23 * static_cast<double>(dof.patch_id + 1))
            + 0.25 * std::sin(2.0 * pi * (uhat + vhat));
    }
    expected.array() -= surface_weighted_mean(fixture.cloud, expected);
    weighted_square_sum = 0.0;
    for (int q = 0; q < size; ++q) {
        weighted_square_sum +=
            fixture.cloud.dofs[static_cast<std::size_t>(q)].weight
            * expected[q] * expected[q];
    }
    expected /= std::sqrt(weighted_square_sum / weight_sum);
    require((first.head(size) - expected).lpNorm<Eigen::Infinity>()
                <= 5.0e-15,
            "common Neumann RHS follows the specified native-parameter formula");
}

void test_detailed_neumann_probe_uses_literal_defect_and_exact_edge_fit()
{
    DetailedNeumannRouteFixture3D fixture;
    auto pipeline = fixture.make_pipeline(
        app3d::HarmonicCauchyRoute3D::EdgeReconstructedValue);
    const app3d::RigidTransform3D transform;
    const NeumannRouteProbe3D probe = run_neumann_route_probe_3d(
        fixture.grid, fixture.grid_pair, *pipeline,
        fixture.geometry.native_surface, transform,
        fixture.neighborhoods, fixture.h,
        app3d::HarmonicCauchyRoute3D::EdgeReconstructedValue);
    const int size = pipeline->surface_size();
    require(probe.physical_residuals.size()
                    == static_cast<std::size_t>(probe.physical.iterations + 1)
                && probe.common.residuals.size()
                    == static_cast<std::size_t>(probe.common.iterations + 1),
            "physical and common probes retain iterations+1 residuals");
    require(probe.density_error.size() == size
                && probe.exact_equation_defect.size() == size
                && std::isfinite(probe.exact_mean_row_defect),
            "detailed probe stores surface-sized density/defect vectors and "
            "a finite mean-row defect");

    const NeumannManufacturedData3D data =
        make_neumann_manufactured_data_3d(pipeline->surface(), transform);
    ExteriorZeroTraceOperator3D op(
        *pipeline, ExteriorValueRestrictMode3D::JointTricubicCrossingOwner);
    Eigen::VectorXd exact_augmented = Eigen::VectorXd::Zero(op.problem_size());
    exact_augmented.head(size) = data.exact_density;
    Eigen::VectorXd applied;
    op.apply(exact_augmented, applied);
    const Eigen::VectorXd literal =
        applied - op.right_hand_side(data.prescribed_normal_jump);
    require((probe.exact_equation_defect - literal.head(size))
                    .lpNorm<Eigen::Infinity>() == 0.0
                && probe.exact_mean_row_defect == literal[size],
            "exact-density defect is the literal bordered operator residual");

    const auto exact_fit = pipeline->cauchy_fit().apply(
        data.exact_density, data.prescribed_normal_jump);
    const auto& edge_maps = pipeline->cauchy_fit().edge_maps();
    require(probe.exact_input_edge_values.size()
                    == static_cast<int>(edge_maps.size())
                && probe.exact_edge_value_error.size()
                    == static_cast<int>(edge_maps.size())
                && probe.exact_edge_quadrature_weights.size()
                    == static_cast<int>(edge_maps.size())
                && probe.edge_value_linf.has_value()
                && probe.edge_value_weighted_rms.has_value(),
            "shared-edge probe stores exact inputs, errors, native weights, "
            "and finite metrics");
    for (int q = 0; q < static_cast<int>(edge_maps.size()); ++q) {
        const auto& point = edge_maps[static_cast<std::size_t>(q)].point;
        const double exact =
            app3d::transformed_manufactured_harmonic_value_3d(
                transform, point.point) - data.density_mean_shift;
        require(probe.exact_input_edge_values[q] == exact
                    && probe.exact_edge_quadrature_weights[q]
                           == point.quadrature_weight
                    && probe.exact_edge_value_error[q]
                           == exact_fit.edge_values[q] - exact,
                "shared-edge exact comparison uses corrected data and native "
                "interval_length/cell_count weights");
    }

    int total_count = 0;
    for (int bin = 0; bin < 3; ++bin) {
        const EdgeBinMetrics3D& metrics =
            probe.bins[static_cast<std::size_t>(bin)];
        total_count += metrics.count;
        require(metrics.empty == (metrics.count == 0),
                "edge bin exposes its empty state explicitly");
    }
    require(probe.bins[0].bin == "lt_h"
                && probe.bins[1].bin == "h_to_2h"
                && probe.bins[2].bin == "gt_2h"
                && total_count == size,
            "edge-distance bins are exact, named, disjoint, and exhaustive");
    for (const auto& map : pipeline->cauchy_fit().surface_maps()) {
        require(map.neighborhood_fingerprint == fixture.neighborhoods.fingerprint,
                "probe route retains the shared neighborhood fingerprint");
    }

    const Eigen::VectorXd edge_common_rhs = probe.common.right_hand_side;
    pipeline.reset();
    auto g1_pipeline = fixture.make_pipeline(
        app3d::HarmonicCauchyRoute3D::G1ValueG1Normal);
    const NeumannRouteProbe3D g1_probe = run_neumann_route_probe_3d(
        fixture.grid, fixture.grid_pair, *g1_pipeline,
        fixture.geometry.native_surface, transform,
        fixture.neighborhoods, fixture.h,
        app3d::HarmonicCauchyRoute3D::G1ValueG1Normal);
    require(g1_probe.exact_input_edge_values.size() == 0
                && g1_probe.exact_edge_value_error.size() == 0
                && g1_probe.exact_edge_quadrature_weights.size() == 0
                && !g1_probe.edge_value_linf.has_value()
                && !g1_probe.edge_value_weighted_rms.has_value(),
            "G1 control stores empty edge vectors and explicit N/A metrics");
    require((g1_probe.common.right_hand_side.array()
                == edge_common_rhs.array()).all(),
            "G1 and shared-edge routes solve the identical common RHS");
    g1_pipeline.reset();

    auto direct_pipeline = fixture.make_pipeline(
        app3d::HarmonicCauchyRoute3D::DirectCrossFaceValue);
    const NeumannRouteProbe3D direct_probe = run_neumann_route_probe_3d(
        fixture.grid, fixture.grid_pair, *direct_pipeline,
        fixture.geometry.native_surface, transform,
        fixture.neighborhoods, fixture.h,
        app3d::HarmonicCauchyRoute3D::DirectCrossFaceValue);
    require(direct_probe.exact_input_edge_values.size() == 0
                && direct_probe.exact_edge_value_error.size() == 0
                && direct_probe.exact_edge_quadrature_weights.size() == 0
                && !direct_probe.edge_value_linf.has_value()
                && !direct_probe.edge_value_weighted_rms.has_value(),
            "direct control stores empty edge vectors and explicit N/A metrics");
    require((direct_probe.common.right_hand_side.array()
                == edge_common_rhs.array()).all(),
            "direct and shared-edge routes solve the identical common RHS");
}

std::string read_text_file(const std::filesystem::path& path)
{
    std::ifstream input(path);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

std::vector<std::vector<std::string>> parse_csv_records_3d(
    const std::string& text)
{
    std::vector<std::vector<std::string>> records;
    std::vector<std::string> record;
    std::string field;
    bool quoted = false;
    for (std::size_t index = 0; index < text.size(); ++index) {
        const char character = text[index];
        if (character == static_cast<char>(34)) {
            if (quoted && index + 1 < text.size()
                && text[index + 1] == static_cast<char>(34)) {
                field.push_back(character);
                ++index;
            } else {
                quoted = !quoted;
            }
        } else if (character == ',' && !quoted) {
            record.push_back(field);
            field.clear();
        } else if (character == static_cast<char>(10) && !quoted) {
            if (!field.empty() || !record.empty()) {
                record.push_back(field);
                records.push_back(record);
            }
            field.clear();
            record.clear();
        } else if (character != static_cast<char>(13) || quoted) {
            field.push_back(character);
        }
    }
    require(!quoted, "test CSV parser found an unterminated quote");
    return records;
}

std::string csv_value_3d(
    const std::vector<std::vector<std::string>>& records,
    const std::string& caseId,
    const std::string& column)
{
    require(!records.empty(), "CSV lookup received no records");
    const auto columnIt = std::find(records.front().begin(), records.front().end(),
                                    column);
    require(columnIt != records.front().end(), "CSV lookup missing column " + column);
    const std::size_t columnIndex = static_cast<std::size_t>(
        std::distance(records.front().begin(), columnIt));
    for (std::size_t row = 1; row < records.size(); ++row) {
        if (!records[row].empty() && records[row][0] == caseId) {
            require(columnIndex < records[row].size(),
                    "CSV lookup found a short record for " + caseId);
            return records[row][columnIndex];
        }
    }
    throw std::runtime_error("CSV lookup missing case " + caseId);
}

void write_csv_records_3d(
    const std::filesystem::path& path,
    const std::vector<std::vector<std::string>>& records)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    for (const auto& record : records) {
        for (std::size_t column = 0; column < record.size(); ++column) {
            if (column != 0) output << ',';
            output << csv_field_3d(record[column]);
        }
        output << '\n';
    }
}

void mutate_csv_field_3d(
    const std::filesystem::path& path, std::size_t dataRow,
    const std::string& column, const std::string& value)
{
    auto records = parse_csv_records_3d(read_text_file(path));
    require(dataRow > 0 && dataRow < records.size(),
            "CSV field mutation has no requested data row");
    const auto found = std::find(
        records.front().begin(), records.front().end(), column);
    require(found != records.front().end(),
            "CSV field mutation missing column " + column);
    const std::size_t index = static_cast<std::size_t>(
        std::distance(records.front().begin(), found));
    records[dataRow][index] = value;
    write_csv_records_3d(path, records);
}

void mutate_csv_column_all_data_3d(
    const std::filesystem::path& path,
    const std::string& column, const std::string& value)
{
    auto records = parse_csv_records_3d(read_text_file(path));
    const auto found = std::find(
        records.front().begin(), records.front().end(), column);
    require(found != records.front().end(),
            "CSV column mutation missing column " + column);
    const std::size_t index = static_cast<std::size_t>(
        std::distance(records.front().begin(), found));
    for (std::size_t row = 1; row < records.size(); ++row)
        records[row][index] = value;
    write_csv_records_3d(path, records);
}

TwoLevelNeumannStudyRouteRow3D make_shared_writer_fixture_3d()
{
    TwoLevelNeumannStudyRouteRow3D row;
    row.case_id = "synthetic";
    row.N = 32;
    row.route = app3d::HarmonicCauchyRoute3D::EdgeReconstructedValue;
    row.status = "ok";
    row.h = 1.0 / 32.0;
    row.patch_count = 2;
    row.physical_iterations = 2;
    row.common_iterations = 1;
    row.physical_converged = true;
    row.common_converged = true;
    row.physical_final_residual = 0.01;
    row.common_final_residual = 0.02;
    row.physical_residuals = {1.0, 0.1, 0.01};
    row.common_residuals = {1.0, 0.02};
    row.common_rhs_rms = 1.0;
    row.defect_linf = 0.5;
    row.defect_rms = std::sqrt(0.59 / 3.0);
    row.edge_value_linf = 0.05;
    row.edge_value_rms = 0.05;
    row.edge_bins = {{
        {"lt_h", 1, 1.0, 0.2, 0.2, 0.3, 0.3, false},
        {"h_to_2h", 1, 2.0, 0.4, 0.4, 0.5, 0.5, false},
        {"gt_2h", 0, 0.0, 0.0, 0.0, 0.0, 0.0, true}}};
    row.owner.available = true;
    row.owner.owner_query_count = 17;
    row.owner.owner_fingerprint_before = 101;
    row.owner.owner_fingerprint_after = 101;
    row.owner.cauchy_fingerprint_before = 202;
    row.owner.cauchy_fingerprint_after = 202;
    row.owner.reference_equal = true;
    row.owner.common_rhs_hash_equal = true;
    TwoLevelEdgePointDiagnostic3D point;
    point.connection_id = 5;
    point.cell_id = 6;
    point.fraction = 0.25;
    point.native_parameters = {{0.25, 0.75}};
    point.native_uv[0] = Eigen::Vector2d(0.25, 0.0);
    point.native_uv[1] = Eigen::Vector2d(0.75, 0.0);
    point.point = Eigen::Vector3d(0.25, 0.0, 0.0);
    point.tangent = Eigen::Vector3d::UnitX();
    point.sectors = "{{0};{1}}";
    point.exact_value = 1.25;
    point.reconstructed_value = 1.20;
    point.error = -0.05;
    row.edge_points.push_back(point);
    TwoLevelEdgeFitDiagnostic3D fit;
    fit.connection_id = 5;
    fit.cell_id = 6;
    fit.value_sector_counts = {{24, 24}};
    fit.normal_sector_counts = {{14, 14}};
    fit.value_radius_over_h = 1.1;
    fit.normal_radius_over_h = 1.2;
    fit.sigma_max = 2.0;
    fit.sigma_min = 0.5;
    fit.condition = 4.0;
    row.edge_fits.push_back(fit);
    TwoLevelEdgePointDiagnostic3D secondPoint = point;
    secondPoint.cell_id = 7;
    secondPoint.fraction = 0.75;
    secondPoint.native_parameters = {{0.75, 0.25}};
    row.edge_points.push_back(secondPoint);
    TwoLevelEdgeFitDiagnostic3D secondFit = fit;
    secondFit.cell_id = 7;
    row.edge_fits.push_back(secondFit);
    row.surface_dof_count = 2;
    row.shared_edge_point_count = 2;
    row.surface_map_count = 2;
    row.value_map_count = 2;
    row.normal_map_count = 2;
    row.edge_map_count = 2;
    row.value_radius_max_over_h = 1.1;
    row.normal_radius_max_over_h = 1.2;
    row.condition_max = 4.0;
    TwoLevelSurfaceFitDiagnostic3D surfaceFit;
    surfaceFit.center_dof = 0;
    surfaceFit.ordinary_value_count = 48;
    surfaceFit.normal_count = 28;
    surfaceFit.edge_count = 1;
    surfaceFit.value_sector_patch_ids = "{{0};{1}}";
    surfaceFit.value_sector_counts = "{24;24}";
    surfaceFit.normal_sector_patch_ids = "{{0};{1}}";
    surfaceFit.normal_sector_counts = "{14;14}";
    surfaceFit.value_radius_over_h = 1.1;
    surfaceFit.normal_radius_over_h = 1.2;
    surfaceFit.sigma_min = 0.5;
    surfaceFit.sigma_max = 2.0;
    surfaceFit.condition = 4.0;
    row.surface_fits.push_back(surfaceFit);
    TwoLevelSurfaceFitDiagnostic3D secondSurfaceFit = surfaceFit;
    secondSurfaceFit.center_dof = 1;
    row.surface_fits.push_back(secondSurfaceFit);
    TwoLevelDofDiagnostic3D dof;
    dof.dof_id = 0;
    dof.patch_id = 0;
    dof.point = Eigen::Vector3d(0.25, 0.0, 0.0);
    dof.weight = 1.0;
    dof.edge_distance_over_h = 0.5;
    dof.density_error = 0.2;
    dof.equation_defect = 0.3;
    row.dofs.push_back(dof);
    TwoLevelDofDiagnostic3D secondDof = dof;
    secondDof.dof_id = 1;
    secondDof.point = Eigen::Vector3d(0.75, 0.0, 0.0);
    secondDof.weight = 2.0;
    secondDof.edge_distance_over_h = 1.5;
    secondDof.density_error = 0.4;
    secondDof.equation_defect = 0.5;
    row.dofs.push_back(secondDof);
    return row;
}

void duplicate_first_csv_data_record_over_second_3d(
    const std::filesystem::path& path)
{
    std::string text = read_study_test_file_3d(path);
    const std::size_t headerEnd = text.find(static_cast<char>(10));
    const std::size_t firstEnd = text.find(static_cast<char>(10), headerEnd + 1);
    const std::size_t secondEnd = text.find(static_cast<char>(10), firstEnd + 1);
    require(headerEnd != std::string::npos && firstEnd != std::string::npos
                && secondEnd != std::string::npos,
            "duplicate-record mutation requires two data records");
    text.replace(firstEnd + 1, secondEnd - firstEnd,
                 text.substr(headerEnd + 1, firstEnd - headerEnd));
    write_study_test_file_3d(path, text);
}

void append_csv_record_for_case_3d(
    const std::filesystem::path& path, const std::string& sourceCase,
    const std::string& destinationCase)
{
    auto records = parse_csv_records_3d(read_text_file(path));
    auto found = records.end();
    for (auto row = records.begin() + 1; row != records.end(); ++row) {
        if (!row->empty() && (*row)[0] == sourceCase) {
            found = row;
            break;
        }
    }
    require(found != records.end(),
            "append-record mutation missing case " + sourceCase);
    auto appended = *found;
    appended[0] = destinationCase;
    records.push_back(std::move(appended));
    write_csv_records_3d(path, records);
}

void mutate_csv_field_for_case_3d(
    const std::filesystem::path& path, const std::string& caseId,
    const std::string& column, const std::string& value)
{
    auto records = parse_csv_records_3d(read_text_file(path));
    const auto columnIt = std::find(
        records.front().begin(), records.front().end(), column);
    require(columnIt != records.front().end(),
            "case mutation missing column " + column);
    const std::size_t columnIndex = static_cast<std::size_t>(
        std::distance(records.front().begin(), columnIt));
    auto row = records.end();
    for (auto candidate = records.begin() + 1;
         candidate != records.end(); ++candidate) {
        if (candidate->size() > columnIndex && (*candidate)[0] == caseId) {
            row = candidate;
            break;
        }
    }
    require(row != records.end(), "case mutation missing case " + caseId);
    (*row)[columnIndex] = value;
    write_csv_records_3d(path, records);
}

void mutate_csv_field_for_case_and_kind_3d(
    const std::filesystem::path& path, const std::string& caseId,
    const std::string& kind, const std::string& column,
    const std::string& value)
{
    auto records = parse_csv_records_3d(read_text_file(path));
    const auto columnIt = std::find(
        records.front().begin(), records.front().end(), column);
    const auto kindIt = std::find(
        records.front().begin(), records.front().end(), "rhs_kind");
    require(columnIt != records.front().end()
                && kindIt != records.front().end(),
            "case/kind mutation missing a requested column");
    const std::size_t columnIndex = static_cast<std::size_t>(
        std::distance(records.front().begin(), columnIt));
    const std::size_t kindIndex = static_cast<std::size_t>(
        std::distance(records.front().begin(), kindIt));
    auto row = records.end();
    for (auto candidate = records.begin() + 1;
         candidate != records.end(); ++candidate) {
        if (candidate->size() > std::max(columnIndex, kindIndex)
            && (*candidate)[0] == caseId
            && (*candidate)[kindIndex] == kind) {
            row = candidate;
            break;
        }
    }
    require(row != records.end(),
            "case/kind mutation missing case " + caseId);
    (*row)[columnIndex] = value;
    write_csv_records_3d(path, records);
}

void mutate_csv_field_matching_3d(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, std::string>>& matches,
    const std::string& column, const std::string& value)
{
    auto records = parse_csv_records_3d(read_text_file(path));
    const auto targetIt = std::find(
        records.front().begin(), records.front().end(), column);
    require(targetIt != records.front().end(),
            "matching mutation missing target column " + column);
    const std::size_t targetIndex = static_cast<std::size_t>(
        std::distance(records.front().begin(), targetIt));
    std::vector<std::pair<std::size_t, std::string>> indexedMatches;
    for (const auto& match : matches) {
        const auto found = std::find(
            records.front().begin(), records.front().end(), match.first);
        require(found != records.front().end(),
                "matching mutation missing selector column " + match.first);
        indexedMatches.emplace_back(
            static_cast<std::size_t>(
                std::distance(records.front().begin(), found)),
            match.second);
    }
    auto row = records.end();
    for (auto candidate = records.begin() + 1;
         candidate != records.end(); ++candidate) {
        bool equal = candidate->size() > targetIndex;
        for (const auto& match : indexedMatches)
            equal = equal && candidate->size() > match.first
                && (*candidate)[match.first] == match.second;
        if (equal) {
            row = candidate;
            break;
        }
    }
    require(row != records.end(), "matching mutation found no requested row");
    (*row)[targetIndex] = value;
    write_csv_records_3d(path, records);
}

void erase_csv_record_matching_3d(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, std::string>>& matches)
{
    auto records = parse_csv_records_3d(read_text_file(path));
    std::vector<std::pair<std::size_t, std::string>> indexedMatches;
    for (const auto& match : matches) {
        const auto found = std::find(
            records.front().begin(), records.front().end(), match.first);
        require(found != records.front().end(),
                "record erasure missing selector column " + match.first);
        indexedMatches.emplace_back(
            static_cast<std::size_t>(
                std::distance(records.front().begin(), found)),
            match.second);
    }
    const auto row = std::find_if(records.begin() + 1, records.end(),
        [&indexedMatches](const auto& candidate) {
            return std::all_of(indexedMatches.begin(), indexedMatches.end(),
                [&candidate](const auto& match) {
                    return candidate.size() > match.first
                        && candidate[match.first] == match.second;
                });
        });
    require(row != records.end(), "record erasure found no requested row");
    records.erase(row);
    write_csv_records_3d(path, records);
}

void require_review_schema_negative_mutations_3d(
    const std::filesystem::path& script,
    const std::filesystem::path& source)
{
    const auto exercise = [&](const std::string& name,
                              const std::function<void(
                                  const std::filesystem::path&)>& mutate) {
        const std::filesystem::path destination =
            source.parent_path() / (source.filename().string() + '_' + name);
        std::error_code error;
        std::filesystem::remove_all(destination, error);
        std::filesystem::copy(source, destination,
            std::filesystem::copy_options::recursive);
        mutate(destination);
        require(run_two_level_neumann_schema_audit_3d(script, destination) != 0,
                "schema audit accepted review mutation: " + name);
        std::filesystem::remove_all(destination, error);
    };
    exercise("reordered_header", [](const std::filesystem::path& directory) {
        const auto path = directory / "summary.csv";
        std::string text = read_study_test_file_3d(path);
        const std::string ordered = ",setup_seconds,fit_seconds,";
        const std::size_t offset = text.find(ordered);
        require(offset != std::string::npos,
                "reordered-header mutation found no target");
        text.replace(offset, ordered.size(), ",fit_seconds,setup_seconds,");
        write_study_test_file_3d(path, text);
    });
    exercise("bad_record_width", [](const std::filesystem::path& directory) {
        const auto path = directory / "summary.csv";
        std::string text = read_study_test_file_3d(path);
        const std::size_t headerEnd = text.find(static_cast<char>(10));
        const std::size_t firstEnd = text.find(static_cast<char>(10), headerEnd + 1);
        require(firstEnd != std::string::npos,
                "bad-width mutation found no data record");
        text.insert(firstEnd, ",unexpected_field");
        write_study_test_file_3d(path, text);
    });
    for (const std::string& nonfinite : {"NaN", "Infinity", "-Infinity"}) {
        exercise("nonfinite_residual_" + nonfinite,
            [nonfinite](const std::filesystem::path& directory) {
                mutate_csv_field_3d(directory / "gmres_residuals.csv", 1,
                                    "relative_residual", nonfinite);
            });
    }
    exercise("negative_intermediate_residual", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "gmres_residuals.csv",
            {{"case_id", "synthetic"}, {"rhs_kind", "physical"},
             {"iteration", "1"}},
            "relative_residual", "-1.0e-1");
    });
    exercise("negative_edge_radius", [](const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "edge_fit_diagnostics.csv",
            {{"case_id", "synthetic"}, {"cell_id", "6"}},
            "value_radius_over_h", "-1.0");
    });
    exercise("negative_frame_error", [](const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "edge_point_diagnostics.csv",
            {{"case_id", "synthetic"}, {"cell_id", "6"},
             {"fraction", "2.50000000000000000e-01"}},
            "frame_orthogonality_error", "-1.0");
    });
    exercise("successful_owner_unavailable", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "owner_diagnostics.csv",
            {{"case_id", "synthetic"}}, "available", "0");
    });
    exercise("negative_dof_weight", [](const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "dof_diagnostics.csv",
            {{"case_id", "synthetic"}, {"dof_id", "0"}},
            "weight", "-1.0");
    });
    exercise("negative_dof_edge_distance", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "dof_diagnostics.csv",
            {{"case_id", "synthetic"}, {"dof_id", "0"}},
            "edge_distance_over_h", "-1.0");
    });
    exercise("summary_zero_h", [](const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "summary.csv", {{"case_id", "synthetic"}},
            "h", "0");
    });
    exercise("common_rhs_mean_out_of_tolerance", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "summary.csv", {{"case_id", "synthetic"}},
            "common_rhs_mean", "5.1e-13");
    });
    exercise("common_rhs_rms_out_of_tolerance", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "summary.csv", {{"case_id", "synthetic"}},
            "common_rhs_rms", "9.999999999994e-1");
    });
    exercise("summary_nonfinite_defect", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "summary.csv", {{"case_id", "synthetic"}},
            "defect_linf", "NaN");
    });
    exercise("summary_defect_linf_inconsistent", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "summary.csv", {{"case_id", "synthetic"}},
            "defect_linf", "4.0e-1");
    });
    exercise("summary_defect_rms_inconsistent", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "summary.csv", {{"case_id", "synthetic"}},
            "defect_rms", "4.0e-1");
    });
    exercise("owner_nonnumeric_output_digest", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "owner_diagnostics.csv",
            {{"case_id", "synthetic"}},
            "owner_output_digest_before", "not-a-uint64");
    });
    exercise("surface_malformed_sector_syntax", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "surface_fit_diagnostics.csv",
            {{"case_id", "synthetic"}, {"center_dof", "0"}},
            "value_sector_patch_ids", "{{0};oops}");
    });
    exercise("surface_sector_count_total_mismatch", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "surface_fit_diagnostics.csv",
            {{"case_id", "synthetic"}, {"center_dof", "0"}},
            "value_sector_counts", "{23;24}");
    });
    exercise("surface_patch_id_out_of_range", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "surface_fit_diagnostics.csv",
            {{"case_id", "synthetic"}, {"center_dof", "0"}},
            "value_sector_patch_ids", "{{0};{2}}");
    });
    exercise("shared_surface_edge_count_above_six", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "surface_fit_diagnostics.csv",
            {{"case_id", "synthetic"}, {"center_dof", "0"}},
            "edge_count", "7");
    });
    exercise("edge_point_malformed_sectors", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "edge_point_diagnostics.csv",
            {{"case_id", "synthetic"}, {"cell_id", "6"}},
            "sectors", "{0;1}");
    });
    exercise("edge_point_sector_trailing_newline", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "edge_point_diagnostics.csv",
            {{"case_id", "synthetic"}, {"cell_id", "6"}},
            "sectors", std::string("{{0};{1}}") + static_cast<char>(10));
    });
    exercise("successful_summary_failure_metadata", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "summary.csv", {{"case_id", "synthetic"}},
            "failure_stage", "solve");
    });
    exercise("successful_summary_blank_failure_metadata", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "summary.csv", {{"case_id", "synthetic"}},
            "failure_stage", "");
    });
    exercise("adjusted_missing_surface_row", [](
        const std::filesystem::path& directory) {
        erase_csv_record_matching_3d(
            directory / "surface_fit_diagnostics.csv",
            {{"case_id", "synthetic"}, {"center_dof", "1"}});
        for (const std::string& field : {
                 "surface_map_count", "value_map_count", "normal_map_count"})
            mutate_csv_field_matching_3d(
                directory / "summary.csv", {{"case_id", "synthetic"}},
                field, "1");
    });
    exercise("surface_invalid_svd_ordering", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "surface_fit_diagnostics.csv",
            {{"case_id", "synthetic"}, {"center_dof", "0"}},
            "sigma_min", "3.0");
    });
    exercise("surface_condition_ratio_mismatch", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "surface_fit_diagnostics.csv",
            {{"case_id", "synthetic"}, {"center_dof", "0"}},
            "condition", "5.0");
    });
    exercise("edge_invalid_svd_ordering", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "edge_fit_diagnostics.csv",
            {{"case_id", "synthetic"}, {"cell_id", "6"}},
            "sigma_min", "3.0");
    });
    exercise("edge_condition_ratio_mismatch", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "edge_fit_diagnostics.csv",
            {{"case_id", "synthetic"}, {"cell_id", "6"}},
            "condition", "5.0");
    });
    exercise("summary_fit_maximum_mismatch", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "summary.csv", {{"case_id", "synthetic"}},
            "condition_max", "5.0");
    });
    exercise("raw_dof_square_overflow", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "dof_diagnostics.csv",
            {{"case_id", "synthetic"}, {"dof_id", "0"}},
            "density_error", "1.0e308");
    });
    exercise("raw_dof_rms_overflow_after_finite_square", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "dof_diagnostics.csv",
            {{"case_id", "synthetic"}, {"dof_id", "0"}},
            "weight", "1.0e-308");
        mutate_csv_field_matching_3d(
            directory / "dof_diagnostics.csv",
            {{"case_id", "synthetic"}, {"dof_id", "0"}},
            "density_error", "1.4e154");
    });
    exercise("empty_bin_nonzero_norm", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "edge_distance_bins.csv",
            {{"case_id", "synthetic"}, {"distance_bin", "gt_2h"}},
            "defect_linf", "1.0e-1");
    });
    exercise("raw_edge_nonfinite_value", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "edge_point_diagnostics.csv",
            {{"case_id", "synthetic"}, {"cell_id", "6"}},
            "exact_value", "NaN");
    });
    exercise("raw_edge_error_identity_mismatch", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "edge_point_diagnostics.csv",
            {{"case_id", "synthetic"}, {"cell_id", "6"}},
            "error", "-1.0e-1");
    });
    exercise("summary_edge_linf_raw_mismatch", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "summary.csv", {{"case_id", "synthetic"}},
            "edge_value_linf", "1.0e-1");
    });
    exercise("edge_rms_without_raw_weights_nonfinite", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "summary.csv", {{"case_id", "synthetic"}},
            "edge_value_rms", "NaN");
    });
    exercise("bare_quote_in_unquoted_field", [](
        const std::filesystem::path& directory) {
        const auto path = directory / "summary.csv";
        std::string text = read_study_test_file_3d(path);
        const std::string original = ",first_level,";
        const std::size_t offset = text.find(original);
        require(offset != std::string::npos,
                "bare-quote mutation found no failure stage");
        const std::string malformed = std::string(",first")
            + static_cast<char>(34) + "level,";
        text.replace(offset, original.size(), malformed);
        write_study_test_file_3d(path, text);
    });
    const std::array<std::string, 4> duplicateFiles{{
        "surface_fit_diagnostics.csv", "dof_diagnostics.csv",
        "edge_point_diagnostics.csv", "edge_fit_diagnostics.csv"}};
    for (const std::string& filename : duplicateFiles) {
        exercise("duplicate_" + filename,
            [filename](const std::filesystem::path& directory) {
                duplicate_first_csv_data_record_over_second_3d(
                    directory / filename);
            });
    }
    exercise("orphan_surface_fit", [](const std::filesystem::path& directory) {
        const auto path = directory / "surface_fit_diagnostics.csv";
        std::string text = read_study_test_file_3d(path);
        const std::size_t headerEnd = text.find(static_cast<char>(10));
        const std::size_t caseEnd = text.find(',', headerEnd + 1);
        require(caseEnd != std::string::npos,
                "orphan mutation found no surface-fit record");
        text.replace(headerEnd + 1, caseEnd - headerEnd - 1, "orphan_case");
        write_study_test_file_3d(path, text);
    });
    exercise("orphan_owner", [](const std::filesystem::path& directory) {
        append_csv_record_for_case_3d(
            directory / "owner_diagnostics.csv", "synthetic", "orphan_case");
    });
    exercise("orphan_bin", [](const std::filesystem::path& directory) {
        append_csv_record_for_case_3d(
            directory / "edge_distance_bins.csv", "synthetic", "orphan_case");
    });
    exercise("failed_duplicate_residual", [](
        const std::filesystem::path& directory) {
        append_csv_record_for_case_3d(
            directory / "gmres_residuals.csv", "post_solve_available",
            "post_solve_available");
    });
    exercise("failed_residual_gap", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_for_case_and_kind_3d(
            directory / "gmres_residuals.csv", "post_solve_available",
            "physical", "iteration", "3");
    });
    exercise("failed_surface_id_gap", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_for_case_3d(
            directory / "surface_fit_diagnostics.csv", "post_solve_available",
            "center_dof", "2");
    });
    exercise("failed_dof_id_gap", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_for_case_3d(
            directory / "dof_diagnostics.csv", "post_solve_available",
            "dof_id", "2");
    });
    exercise("failed_summary_residual_mismatch", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_for_case_3d(
            directory / "summary.csv", "post_solve_available",
            "physical_final_residual", "5.0e-1");
    });
    exercise("failed_na_summary_with_residual_history", [](
        const std::filesystem::path& directory) {
        append_csv_record_for_case_3d(
            directory / "gmres_residuals.csv", "synthetic",
            "solve_available");
    });
    exercise("failed_dof_partial_postsolve_values", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "dof_diagnostics.csv",
            {{"case_id", "solve_available"}, {"dof_id", "0"}},
            "density_error", "1.0e-1");
    });
    exercise("mismatched_edge_key_set", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_3d(directory / "edge_point_diagnostics.csv", 2,
                            "cell_id", "8");
    });
    exercise("surface_id_gap", [](const std::filesystem::path& directory) {
        mutate_csv_field_3d(directory / "surface_fit_diagnostics.csv", 2,
                            "center_dof", "2");
    });
    exercise("dof_id_gap", [](const std::filesystem::path& directory) {
        mutate_csv_field_3d(directory / "dof_diagnostics.csv", 2,
                            "dof_id", "2");
    });
    exercise("summary_residual_mismatch", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_3d(directory / "summary.csv", 1,
                            "physical_final_residual", "5.0e-1");
    });
}

int run_two_level_neumann_audit_command_3d(
    const std::filesystem::path& script,
    const std::filesystem::path& outputDirectory,
    const std::string& expectedLevels,
    bool schemaOnly,
    const std::filesystem::path& decision = {})
{
    std::ostringstream command;
    command << "powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "
            << static_cast<char>(34) << "& { try { & '" << script.string()
            << "' -OutputDirectory '" << outputDirectory.string()
            << "' -ExpectedLevels @(" << expectedLevels
            << ") -AllowNumericalFailure";
    if (schemaOnly) command << " -SchemaOnly";
    if (!decision.empty())
        command << " -DecisionJson '" << decision.string() << "'";
    command << " } catch { Write-Error $_; exit 1 } }"
            << static_cast<char>(34);
    return std::system(command.str().c_str());
}

TwoLevelNeumannStudyRouteRow3D make_failed_writer_fixture_3d(
    std::size_t index, const std::string& stage);

void test_failed_coarse_requires_na_adjacent_order()
{
    const std::filesystem::path root =
        std::filesystem::current_path() / "task5_failed_pair_test_output";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    TwoLevelNeumannStudyWriter3D writer(root);
    auto coarse = make_failed_writer_fixture_3d(20, "solve");
    coarse.case_id = "failed_pair";
    coarse.route = app3d::HarmonicCauchyRoute3D::EdgeReconstructedValue;
    coarse.N = 32;
    writer.append(coarse);
    auto fine = make_shared_writer_fixture_3d();
    fine.case_id = coarse.case_id;
    fine.N = 64;
    writer.append(fine);
    writer.close();
    const auto script = std::filesystem::current_path()
        / "apps/audit_neumann_two_level_edge_cauchy_3d.ps1";
    require(run_two_level_neumann_audit_command_3d(
                script, root, "32,64", true) == 0,
            "failed coarse row must make fine adjacent orders explicitly NA");
    std::filesystem::remove_all(root, error);
}

void test_nonpositive_adjacent_order_error_is_rejected()
{
    const std::filesystem::path root =
        std::filesystem::current_path() / "task5_bad_order_test_output";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    TwoLevelNeumannStudyWriter3D writer(root);
    auto coarse = make_shared_writer_fixture_3d();
    coarse.case_id = "bad_order_pair";
    coarse.N = 32;
    coarse.density_linf = 0.0;
    coarse.density_l2 = 0.0;
    coarse.interior_linf = 0.0;
    coarse.interior_l2 = 0.0;
    writer.append(coarse);
    auto fine = coarse;
    fine.N = 64;
    for (auto& order : fine.orders_32_64) order = 1.0;
    writer.append(fine);
    writer.close();
    const auto script = std::filesystem::current_path()
        / "apps/audit_neumann_two_level_edge_cauchy_3d.ps1";
    require(run_two_level_neumann_audit_command_3d(
                script, root, "32,64", true) != 0,
            "audit accepted nonpositive adjacent-order error denominator");
    std::filesystem::remove_all(root, error);
}

void test_two_level_help_separates_unforced_and_legacy_modes()
{
    std::ostringstream output;
    std::streambuf* previous = std::cout.rdbuf(output.rdbuf());
    print_usage("neumann_app");
    std::cout.rdbuf(previous);
    const std::string text = output.str();
    require(text.find(
                "neumann_app --neumann-edge-cauchy-study [N ...]")
                    != std::string::npos
                && text.find(
                "neumann_app --neumann-edge-cauchy-study --force-extended [N ...]")
                    != std::string::npos,
            "help does not expose distinct unforced and legacy forced commands");
    require(text.find(
                "Unforced Neumann-edge-cauchy-study default levels: 32, 64, 128.")
                    != std::string::npos
                && text.find(
                "Legacy --force-extended levels are the gated prefixes 32; "
                "32,64; or 32,64,128 (default: 32,64).")
                    != std::string::npos,
            "help does not distinguish new and legacy defaults/gates");
}

TwoLevelNeumannStudyRouteRow3D make_decision_writer_fixture_3d(
    const std::string& caseId, int N,
    app3d::HarmonicCauchyRoute3D route, int poseIndex)
{
    auto row = make_shared_writer_fixture_3d();
    row.case_id = caseId;
    row.N = N;
    row.route = route;
    row.setup_available = true;
    row.fit_available = true;
    row.owner_before_available = true;
    row.solve_available = true;
    row.physical_iterations = 1;
    row.common_iterations = 1;
    row.physical_residuals = {1.0, 1.0e-12};
    row.common_residuals = {1.0, 1.0e-12};
    row.physical_final_residual = 1.0e-12;
    row.common_final_residual = 1.0e-12;
    row.physical_converged = true;
    row.common_converged = true;
    row.physical_contraction = 1.0e-12;
    row.common_contraction = 1.0e-12;
    row.common_rhs_rms = 1.0;
    row.common_rhs_hash = static_cast<std::uint64_t>(10000 + 10 * poseIndex + N);
    const double error = 32.0 / static_cast<double>(N);
    row.density_linf = error;
    row.density_l2 = error;
    row.interior_linf = error;
    row.interior_l2 = error;
    if (N == 64)
        for (auto& order : row.orders_32_64) order = 1.0;
    if (N == 128)
        for (auto& order : row.orders_64_128) order = 1.0;
    row.defect_linf = 0.5;
    row.defect_rms = std::sqrt(0.59 / 3.0);
    row.edge_value_linf = 1.0 / static_cast<double>(N);
    row.edge_value_rms = 0.5 / static_cast<double>(N);
    for (auto& point : row.edge_points) {
        point.error = -row.edge_value_linf.value();
        point.reconstructed_value =
            point.exact_value.value() + point.error.value();
    }
    row.label_inside_count = 20 + poseIndex;
    row.label_outside_count = 80 - poseIndex;
    row.label_fingerprint = static_cast<std::uint64_t>(300 + poseIndex);
    row.neighborhood_fingerprint = static_cast<std::uint64_t>(400 + poseIndex);
    row.owner.available = true;
    row.owner.owner_query_count = 17;
    row.owner.owner_fingerprint_before =
        static_cast<std::uint64_t>(500 + poseIndex);
    row.owner.owner_fingerprint_after = row.owner.owner_fingerprint_before;
    row.owner.owner_output_digest_before =
        static_cast<std::uint64_t>(600 + poseIndex);
    row.owner.owner_output_digest_after = row.owner.owner_output_digest_before;
    row.owner.cauchy_geometry_queries = 9;
    row.owner.cauchy_svd_factorizations = 3;
    row.owner.cauchy_fingerprint_before = static_cast<std::uint64_t>(
        700 + static_cast<int>(route));
    row.owner.cauchy_fingerprint_after = row.owner.cauchy_fingerprint_before;
    row.owner.reference_equal = true;
    row.owner.label_equal = true;
    row.owner.neighborhood_equal = true;
    row.owner.common_rhs_hash_equal = true;
    if (route != app3d::HarmonicCauchyRoute3D::EdgeReconstructedValue) {
        row.shared_edge_point_count = 0;
        row.edge_map_count = 0;
        row.edge_points.clear();
        row.edge_fits.clear();
        row.edge_value_linf.reset();
        row.edge_value_rms.reset();
        for (auto& fit : row.surface_fits) fit.edge_count = 0;
    }
    return row;
}

void write_decision_fixture_3d(
    const std::filesystem::path& root,
    const std::vector<int>& levels,
    bool failBaselineG1Coarse)
{
    const std::array<std::string, 3> cases{{
        "baseline", "rot_axis123_17deg", "rot_axis123_17deg_t_xyz_1"}};
    const std::array<app3d::HarmonicCauchyRoute3D, 3> routes{{
        app3d::HarmonicCauchyRoute3D::G1ValueG1Normal,
        app3d::HarmonicCauchyRoute3D::DirectCrossFaceValue,
        app3d::HarmonicCauchyRoute3D::EdgeReconstructedValue}};
    TwoLevelNeumannStudyWriter3D writer(root);
    for (std::size_t pose = 0; pose < cases.size(); ++pose) {
        for (int N : levels) {
            for (const auto route : routes) {
                if (failBaselineG1Coarse && pose == 0 && N == 32
                    && route == app3d::HarmonicCauchyRoute3D::G1ValueG1Normal) {
                    auto failed = make_failed_writer_fixture_3d(30, "solve");
                    failed.case_id = cases[pose];
                    failed.N = N;
                    failed.route = route;
                    writer.append(failed);
                    continue;
                }
                auto row = make_decision_writer_fixture_3d(
                    cases[pose], N, route, static_cast<int>(pose));
                if (failBaselineG1Coarse && pose == 0 && N == 64
                    && route == app3d::HarmonicCauchyRoute3D::G1ValueG1Normal) {
                    for (auto& order : row.orders_32_64) order.reset();
                }
                writer.append(row);
            }
        }
    }
    writer.close();
}

void require_json_state_3d(
    const std::filesystem::path& decision,
    const std::string& selectedRoute,
    const std::string& formalEvidence,
    const std::string& mandatoryPass)
{
    const std::string json = read_text_file(decision);
    const std::string quote(1, static_cast<char>(34));
    require(json.find(static_cast<char>(34) + selectedRoute
                + static_cast<char>(34)) != std::string::npos,
            "decision JSON has wrong selected route");
    require(json.find(quote + "formal_evidence_complete" + quote + ":  "
                + formalEvidence)
                != std::string::npos
                && json.find(quote + "mandatory_numerical_pass" + quote + ":  "
                    + mandatoryPass)
                    != std::string::npos,
            "decision JSON has wrong formal/mandatory state");
}

void test_decision_state_distinguishes_partial_performance_and_failure()
{
    const auto script = std::filesystem::current_path()
        / "apps/audit_neumann_two_level_edge_cauchy_3d.ps1";
    std::error_code error;
    const auto partial = std::filesystem::current_path()
        / "task5_partial_decision_test_output";
    std::filesystem::remove_all(partial, error);
    write_decision_fixture_3d(partial, {32}, false);
    const auto partialDecision = partial / "decision.json";
    require(run_two_level_neumann_audit_command_3d(
                script, partial, "32", false, partialDecision) == 0,
            "partial N32 decision audit failed");
    require_json_state_3d(
        partialDecision, "insufficient_evidence", "false", "true");
    const auto zeroWeight = std::filesystem::current_path()
        / "task5_zero_weight_test_output";
    std::filesystem::remove_all(zeroWeight, error);
    std::filesystem::copy(partial, zeroWeight,
        std::filesystem::copy_options::recursive);
    mutate_csv_column_all_data_3d(
        zeroWeight / "edge_distance_bins.csv", "weight_sum", "0");
    require(run_two_level_neumann_audit_command_3d(
                script, zeroWeight, "32", false,
                zeroWeight / "decision.json") != 0,
            "audit accepted zero combined near-edge weight");

    const auto performance = std::filesystem::current_path()
        / "task5_formal_performance_test_output";
    std::filesystem::remove_all(performance, error);
    write_decision_fixture_3d(performance, {32, 64, 128}, false);
    const auto performanceDecision = performance / "decision.json";
    require(run_two_level_neumann_audit_command_3d(
                script, performance, "32,64,128", false,
                performanceDecision) == 0,
            "formal performance decision audit failed");
    require_json_state_3d(performanceDecision,
        "sector_polynomials_with_shared_edge_constraints", "true", "true");

    const auto exerciseFormalRejection = [&performance, &script, &error](
        const std::string& name,
        const std::function<void(const std::filesystem::path&)>& mutate) {
        const auto destination = performance.parent_path()
            / (performance.filename().string() + "_" + name);
        std::filesystem::remove_all(destination, error);
        std::filesystem::copy(performance, destination,
            std::filesystem::copy_options::recursive);
        mutate(destination);
        require(run_two_level_neumann_audit_command_3d(
                    script, destination, "32,64,128", false,
                    destination / "decision.json") != 0,
                "audit accepted invalid formal evidence: " + name);
        std::filesystem::remove_all(destination, error);
    };
    exerciseFormalRejection("control_surface_nonzero_edge_count", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "surface_fit_diagnostics.csv",
            {{"case_id", "baseline"}, {"N", "32"},
             {"route", "g1_value_g1_normal"}, {"center_dof", "0"}},
            "edge_count", "1");
    });
    exerciseFormalRejection("derived_order_overflow", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "summary.csv",
            {{"case_id", "baseline"}, {"N", "32"},
             {"route", "g1_value_g1_normal"}},
            "density_linf", "1.0e308");
        mutate_csv_field_matching_3d(
            directory / "summary.csv",
            {{"case_id", "baseline"}, {"N", "64"},
             {"route", "g1_value_g1_normal"}},
            "density_linf", "1.0e-308");
    });
    exerciseFormalRejection("combined_weight_overflow", [](
        const std::filesystem::path& directory) {
        for (const std::string& dof : {"0", "1"}) {
            mutate_csv_field_matching_3d(
                directory / "dof_diagnostics.csv",
                {{"case_id", "baseline"}, {"N", "32"},
                 {"route", "g1_value_g1_normal"}, {"dof_id", dof}},
                "weight", "1.0e308");
        }
        for (const std::string& bin : {"lt_h", "h_to_2h"}) {
            mutate_csv_field_matching_3d(
                directory / "edge_distance_bins.csv",
                {{"case_id", "baseline"}, {"N", "32"},
                 {"route", "g1_value_g1_normal"}, {"distance_bin", bin}},
                "weight_sum", "1.0e308");
        }
    });
    exerciseFormalRejection("combined_square_overflow", [](
        const std::filesystem::path& directory) {
        for (const std::string& dof : {"0", "1"}) {
            mutate_csv_field_matching_3d(
                directory / "dof_diagnostics.csv",
                {{"case_id", "baseline"}, {"N", "32"},
                 {"route", "g1_value_g1_normal"}, {"dof_id", dof}},
                "weight", "9.0e-1");
            mutate_csv_field_matching_3d(
                directory / "dof_diagnostics.csv",
                {{"case_id", "baseline"}, {"N", "32"},
                 {"route", "g1_value_g1_normal"}, {"dof_id", dof}},
                "equation_defect", "1.0e154");
        }
        for (const std::string& bin : {"lt_h", "h_to_2h"}) {
            mutate_csv_field_matching_3d(
                directory / "edge_distance_bins.csv",
                {{"case_id", "baseline"}, {"N", "32"},
                 {"route", "g1_value_g1_normal"}, {"distance_bin", bin}},
                "weight_sum", "9.0e-1");
            mutate_csv_field_matching_3d(
                directory / "edge_distance_bins.csv",
                {{"case_id", "baseline"}, {"N", "32"},
                 {"route", "g1_value_g1_normal"}, {"distance_bin", bin}},
                "defect_linf", "1.0e154");
            mutate_csv_field_matching_3d(
                directory / "edge_distance_bins.csv",
                {{"case_id", "baseline"}, {"N", "32"},
                 {"route", "g1_value_g1_normal"}, {"distance_bin", bin}},
                "defect_weighted_rms", "1.0e154");
        }
        mutate_csv_field_matching_3d(
            directory / "summary.csv",
            {{"case_id", "baseline"}, {"N", "32"},
             {"route", "g1_value_g1_normal"}},
            "defect_linf", "1.0e154");
        mutate_csv_field_matching_3d(
            directory / "summary.csv",
            {{"case_id", "baseline"}, {"N", "32"},
             {"route", "g1_value_g1_normal"}},
            "defect_rms", "1.0e154");
    });

    const auto controlEdgeValue = performance.parent_path()
        / (performance.filename().string() + "_control_edge_value");
    std::filesystem::remove_all(controlEdgeValue, error);
    std::filesystem::copy(performance, controlEdgeValue,
        std::filesystem::copy_options::recursive);
    mutate_csv_field_matching_3d(
        controlEdgeValue / "summary.csv",
        {{"case_id", "baseline"}, {"N", "32"},
         {"route", "g1_value_g1_normal"}},
        "edge_value_linf", "0.0");
    require(run_two_level_neumann_audit_command_3d(
                script, controlEdgeValue, "32,64,128", true) != 0,
            "schema audit accepted a fabricated control edge error");
    mutate_csv_field_matching_3d(
        controlEdgeValue / "summary.csv",
        {{"case_id", "baseline"}, {"N", "32"},
         {"route", "g1_value_g1_normal"}},
        "edge_value_linf", "");
    require(run_two_level_neumann_audit_command_3d(
                script, controlEdgeValue, "32,64,128", true) != 0,
            "schema audit accepted a blank control edge error");
    std::filesystem::remove_all(controlEdgeValue, error);

    const auto exerciseFormalMandatoryFailure = [&performance, &script, &error](
        const std::string& name,
        const std::function<void(const std::filesystem::path&)>& mutate) {
        const auto destination = performance.parent_path()
            / (performance.filename().string() + "_" + name);
        std::filesystem::remove_all(destination, error);
        std::filesystem::copy(performance, destination,
            std::filesystem::copy_options::recursive);
        mutate(destination);
        const auto decision = destination / "decision.json";
        require(run_two_level_neumann_audit_command_3d(
                    script, destination, "32,64,128", false, decision) == 0,
                "formal mandatory mutation did not write decision: " + name);
        require_json_state_3d(
            decision, "rerun_after_numerical_failure", "true", "false");
        std::filesystem::remove_all(destination, error);
    };
    exerciseFormalMandatoryFailure("edge_near_rank_cutoff", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "edge_fit_diagnostics.csv",
            {{"case_id", "baseline"}, {"N", "32"},
             {"route", "edge_reconstructed_value"}, {"cell_id", "6"}},
            "sigma_min", "5.0e-12");
        mutate_csv_field_matching_3d(
            directory / "edge_fit_diagnostics.csv",
            {{"case_id", "baseline"}, {"N", "32"},
             {"route", "edge_reconstructed_value"}, {"cell_id", "6"}},
            "condition", "4.0e11");
        mutate_csv_field_matching_3d(
            directory / "summary.csv",
            {{"case_id", "baseline"}, {"N", "32"},
             {"route", "edge_reconstructed_value"}},
            "condition_max", "4.0e11");
    });
    exerciseFormalMandatoryFailure("surface_near_rank_cutoff", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "surface_fit_diagnostics.csv",
            {{"case_id", "baseline"}, {"N", "32"},
             {"route", "g1_value_g1_normal"}, {"center_dof", "0"}},
            "sigma_min", "5.0e-12");
        mutate_csv_field_matching_3d(
            directory / "surface_fit_diagnostics.csv",
            {{"case_id", "baseline"}, {"N", "32"},
             {"route", "g1_value_g1_normal"}, {"center_dof", "0"}},
            "condition", "4.0e11");
        mutate_csv_field_matching_3d(
            directory / "summary.csv",
            {{"case_id", "baseline"}, {"N", "32"},
             {"route", "g1_value_g1_normal"}},
            "condition_max", "4.0e11");
    });
    exerciseFormalMandatoryFailure("raw_bin_count_mismatch", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "edge_distance_bins.csv",
            {{"case_id", "baseline"}, {"N", "32"},
             {"route", "g1_value_g1_normal"}, {"distance_bin", "lt_h"}},
            "count", "2");
    });
    exerciseFormalMandatoryFailure("raw_bin_norm_mismatch", [](
        const std::filesystem::path& directory) {
        mutate_csv_field_matching_3d(
            directory / "edge_distance_bins.csv",
            {{"case_id", "baseline"}, {"N", "32"},
             {"route", "g1_value_g1_normal"}, {"distance_bin", "lt_h"}},
            "density_linf", "2.5e-1");
    });

    const auto failed = std::filesystem::current_path()
        / "task5_formal_failure_test_output";
    std::filesystem::remove_all(failed, error);
    write_decision_fixture_3d(failed, {32, 64, 128}, true);
    const auto failedDecision = failed / "decision.json";
    require(run_two_level_neumann_audit_command_3d(
                script, failed, "32,64,128", false, failedDecision) == 0,
            "formal failed-pair decision audit did not produce JSON");
    require_json_state_3d(failedDecision,
        "rerun_after_numerical_failure", "true", "false");

    std::filesystem::remove_all(partial, error);
    std::filesystem::remove_all(zeroWeight, error);
    std::filesystem::remove_all(performance, error);
    std::filesystem::remove_all(failed, error);
}

TwoLevelNeumannStudyRouteRow3D make_failed_writer_fixture_3d(
    std::size_t index, const std::string& stage)
{
    TwoLevelNeumannStudyRouteRow3D row;
    row.case_id = "failed_" + std::to_string(index);
    row.N = 32;
    row.route = index % 2 == 0
        ? app3d::HarmonicCauchyRoute3D::G1ValueG1Normal
        : app3d::HarmonicCauchyRoute3D::DirectCrossFaceValue;
    row.status = "failed";
    row.failure.stage = stage;
    row.failure.entity_kind = "surface,center";
    row.failure.entity_id = static_cast<int>(100 + index);
    row.failure.connection_id = static_cast<int>(200 + index);
    row.failure.incident_sectors = {{1, 2}, {3}};
    row.failure.actual_value_counts = {7, 8};
    row.failure.actual_normal_counts = {5, 6};
    row.failure.required_value_count = 48;
    row.failure.required_normal_count = 28;
    row.failure.actual_edge_count = 2;
    row.failure.value_radius_over_h = 1.5;
    row.failure.normal_radius_over_h = 1.6;
    row.failure.edge_radius_over_h = 1.7;
    row.failure.sigma_max = 9.0;
    row.failure.sigma_min = 0.25;
    row.failure.condition = 36.0;
    row.failure.message = std::string("failure, with ")
        + static_cast<char>(34) + "quoted" + static_cast<char>(34)
        + " detail";
    row.edge_bins = {{{"lt_h"}, {"h_to_2h"}, {"gt_2h"}}};
    return row;
}

void test_structured_common_setup_failure_reaches_all_writer_rows()
{
    class SetupHookError3D : public app3d::HarmonicCauchyError3D {
    public:
        explicit SetupHookError3D(
            const app3d::HarmonicCauchyFailure3D& diagnostic)
            : app3d::HarmonicCauchyError3D(diagnostic)
        {}

        const char* what() const noexcept override
        {
            return "structured setup hook failure";
        }
    };

    app3d::HarmonicCauchyFailure3D diagnostic;
    diagnostic.entity_kind = "surface";
    diagnostic.entity_id = 17;
    diagnostic.connection_id = 23;
    diagnostic.incident_sectors = {{2, 5}, {7, 11}};
    diagnostic.actual_value_counts = {19, 29};
    diagnostic.actual_normal_counts = {13, 17};
    diagnostic.required_value_count = 48;
    diagnostic.required_normal_count = 28;
    diagnostic.actual_edge_count = 3;
    diagnostic.value_radius_over_h = 1.25;
    diagnostic.normal_radius_over_h = 1.5;
    diagnostic.edge_radius_over_h = 1.75;
    diagnostic.sigma_max = 12.0;
    diagnostic.sigma_min = 0.125;
    diagnostic.condition = 96.0;
    require(diagnostic.stage.empty() && diagnostic.message.empty(),
            "setup hook fixture must exercise empty stage/message fallback");

    const std::filesystem::path root = std::filesystem::current_path()
        / "structured_setup_catch_test_output";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    int setupCalls = 0;
    const int status = run_neumann_two_level_edge_cauchy_study_impl_3d(
        {16}, root, [&] {
            ++setupCalls;
            throw SetupHookError3D(diagnostic);
        });
    require(status == 2 && setupCalls == 3,
            "structured setup hook did not fail each study case");

    const auto value = [](const std::vector<std::vector<std::string>>& records,
                          const std::vector<std::string>& row,
                          const std::string& column) -> const std::string& {
        const auto found = std::find(
            records.front().begin(), records.front().end(), column);
        require(found != records.front().end(),
                "structured setup CSV missing column " + column);
        const std::size_t index = static_cast<std::size_t>(
            std::distance(records.front().begin(), found));
        require(index < row.size(),
                "structured setup CSV has short row for " + column);
        return row[index];
    };
    const auto requireFailure = [&](const auto& records, const auto& row) {
        require(value(records, row, "failure_stage") == "setup"
                    && value(records, row, "failure_message")
                        == "structured setup hook failure"
                    && value(records, row, "failure_entity_kind") == "surface"
                    && value(records, row, "failure_entity_id") == "17"
                    && value(records, row, "failure_connection_id") == "23"
                    && value(records, row, "failure_incident_sectors")
                        == "{{2;5};{7;11}}"
                    && value(records, row, "failure_actual_value_counts")
                        == "{19;29}"
                    && value(records, row, "failure_actual_normal_counts")
                        == "{13;17}"
                    && value(records, row, "failure_required_value_count")
                        == "48"
                    && value(records, row, "failure_required_normal_count")
                        == "28"
                    && value(records, row, "failure_actual_edge_count") == "3"
                    && value(records, row, "failure_value_radius_over_h")
                        == "1.25000000000000000e+00"
                    && value(records, row, "failure_normal_radius_over_h")
                        == "1.50000000000000000e+00"
                    && value(records, row, "failure_edge_radius_over_h")
                        == "1.75000000000000000e+00"
                    && value(records, row, "failure_sigma_max")
                        == "1.20000000000000000e+01"
                    && value(records, row, "failure_sigma_min")
                        == "1.25000000000000000e-01"
                    && value(records, row, "failure_condition")
                        == "9.60000000000000000e+01",
                "structured setup diagnostic did not round-trip completely");
    };
    const std::array<std::string, 3> expectedCases{{
        "baseline", "rot_axis123_17deg", "rot_axis123_17deg_t_xyz_1"}};
    const std::array<std::string, 3> expectedRoutes{{
        "g1_value_g1_normal", "direct_cross_face_value",
        "edge_reconstructed_value"}};
    const std::array<std::string, 3> expectedBins{{
        "lt_h", "h_to_2h", "gt_2h"}};
    const auto contains = [](const auto& expected, const std::string& item) {
        return std::find(expected.begin(), expected.end(), item)
            != expected.end();
    };

    const auto summary = parse_csv_records_3d(
        read_text_file(root / "summary.csv"));
    require(summary.size() == 10,
            "structured setup catch did not write exactly nine summary rows");
    std::set<std::pair<std::string, std::string>> summaryKeys;
    for (auto row = summary.begin() + 1; row != summary.end(); ++row) {
        const std::string& caseId = value(summary, *row, "case_id");
        const std::string& route = value(summary, *row, "route");
        require(contains(expectedCases, caseId)
                    && contains(expectedRoutes, route)
                    && summaryKeys.emplace(caseId, route).second,
                "structured setup summary has a missing/duplicate route key");
        require(value(summary, *row, "N") == "16"
                    && value(summary, *row, "status") == "failed"
                    && value(summary, *row, "patch_count") == "NA"
                    && value(summary, *row, "surface_dof_count") == "NA"
                    && value(summary, *row, "shared_edge_point_count") == "NA"
                    && value(summary, *row, "setup_seconds") == "NA"
                    && value(summary, *row, "neighborhood_geometry_queries")
                        == "NA"
                    && value(summary, *row, "label_inside_count") == "NA"
                    && value(summary, *row, "label_outside_count") == "NA"
                    && value(summary, *row, "label_fingerprint") == "NA"
                    && value(summary, *row, "neighborhood_fingerprint")
                        == "NA",
                "structured setup failure invented unavailable setup fields");
        requireFailure(summary, *row);
    }
    require(summaryKeys.size() == 9,
            "structured setup summary did not cover three cases by three routes");

    const auto owners = parse_csv_records_3d(
        read_text_file(root / "owner_diagnostics.csv"));
    require(owners.size() == 10,
            "structured setup catch did not write nine failed owner rows");
    std::set<std::pair<std::string, std::string>> ownerKeys;
    for (auto row = owners.begin() + 1; row != owners.end(); ++row) {
        const auto key = std::make_pair(
            value(owners, *row, "case_id"), value(owners, *row, "route"));
        require(ownerKeys.insert(key).second
                    && value(owners, *row, "status") == "failed"
                    && value(owners, *row, "available") == "NA"
                    && value(owners, *row, "owner_query_count") == "NA",
                "structured setup owner row is duplicated or partially available");
        requireFailure(owners, *row);
    }

    const auto bins = parse_csv_records_3d(
        read_text_file(root / "edge_distance_bins.csv"));
    require(bins.size() == 28,
            "structured setup catch did not write 27 failed bin rows");
    std::set<std::tuple<std::string, std::string, std::string>> binKeys;
    for (auto row = bins.begin() + 1; row != bins.end(); ++row) {
        const std::string& bin = value(bins, *row, "distance_bin");
        const auto key = std::make_tuple(
            value(bins, *row, "case_id"), value(bins, *row, "route"), bin);
        require(contains(expectedBins, bin) && binKeys.insert(key).second
                    && value(bins, *row, "status") == "failed"
                    && value(bins, *row, "count") == "NA"
                    && value(bins, *row, "weight_sum") == "NA",
                "structured setup bin row is duplicated or partially available");
        requireFailure(bins, *row);
    }
    require(binKeys.size() == 27,
            "structured setup bins did not cover all case/route/bin keys");

    for (const std::string& filename : {
             "gmres_residuals.csv", "edge_point_diagnostics.csv",
             "edge_fit_diagnostics.csv", "surface_fit_diagnostics.csv",
             "dof_diagnostics.csv"}) {
        require(parse_csv_records_3d(read_text_file(root / filename)).size() == 1,
                "structured setup failure invented detail rows in " + filename);
    }
    std::filesystem::remove_all(root, error);
}

void test_two_level_study_writer_and_schema_audit()
{
    const std::filesystem::path root =
        std::filesystem::current_path() / "task5_writer_test_output";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    TwoLevelNeumannStudyWriter3D writer(root);
    writer.append(make_shared_writer_fixture_3d());
    const std::array<std::string, 5> stages{{
        "first_level", "second_level", "frame", "rank",
        "unrelated_selection"}};
    for (std::size_t index = 0; index < stages.size(); ++index)
        writer.append(make_failed_writer_fixture_3d(index, stages[index]));
    auto solveFailure = make_shared_writer_fixture_3d();
    solveFailure.case_id = "solve_available";
    solveFailure.status = "failed";
    solveFailure.failure.stage = "solve";
    solveFailure.failure.message = "synthetic solve failure";
    solveFailure.fit_available = true;
    solveFailure.owner_before_available = true;
    solveFailure.solve_available = false;
    solveFailure.owner.cauchy_geometry_queries = 9;
    solveFailure.owner.cauchy_svd_factorizations = 3;
    solveFailure.physical_residuals.clear();
    solveFailure.common_residuals.clear();
    for (auto& point : solveFailure.edge_points) {
        point.exact_value.reset();
        point.reconstructed_value.reset();
        point.error.reset();
    }
    for (auto& dof : solveFailure.dofs) {
        dof.density_error.reset();
        dof.equation_defect.reset();
    }
    writer.append(solveFailure);
    auto postSolveFailure = make_shared_writer_fixture_3d();
    postSolveFailure.case_id = "post_solve_available";
    postSolveFailure.status = "failed";
    postSolveFailure.failure.stage = "post_solve_diagnostics";
    postSolveFailure.failure.message = "synthetic post-solve failure";
    postSolveFailure.fit_available = true;
    postSolveFailure.owner_before_available = true;
    postSolveFailure.solve_available = true;
    writer.append(postSolveFailure);
    writer.close();

    for (const std::string& filename : two_level_neumann_study_csv_files_3d()) {
        const std::string text = read_text_file(root / filename);
        require(text.rfind("case_id,N,route", 0) == 0,
                filename + " does not begin with the common key");
    }
    const std::string summary = read_text_file(root / "summary.csv");
    const std::string quoted = std::string(1, static_cast<char>(34))
        + "failure, with " + std::string(2, static_cast<char>(34))
        + "quoted" + std::string(2, static_cast<char>(34)) + " detail"
        + std::string(1, static_cast<char>(34));
    require(summary.find(quoted) != std::string::npos,
            "CSV quoting did not preserve commas and quotes");
    for (const std::string& stage : stages)
        require(summary.find(stage) != std::string::npos,
                "structured failure stage did not round-trip: " + stage);
    for (std::size_t index = 0; index < stages.size(); ++index) {
        const std::string entity = std::string(1, static_cast<char>(34))
            + "surface,center" + std::string(1, static_cast<char>(34))
            + ',' + std::to_string(100 + index) + ','
            + std::to_string(200 + index);
        require(summary.find(entity) != std::string::npos,
                "failure entity/connection IDs did not round-trip");
    }
    require(summary.find("{{1;2};{3}}") != std::string::npos
                && summary.find("{7;8},{5;6},48,28,2")
                    != std::string::npos
                && summary.find("1.50000000000000000e+00,")
                    != std::string::npos
                && summary.find("1.60000000000000009e+00,")
                    != std::string::npos
                && summary.find("1.69999999999999996e+00,")
                    != std::string::npos
                && summary.find("9.00000000000000000e+00,")
                    != std::string::npos
                && summary.find("2.50000000000000000e-01,")
                    != std::string::npos
                && summary.find("3.60000000000000000e+01")
                    != std::string::npos,
            "structured sectors/counts/radii/singular extrema/condition "
            "did not round-trip");
    const std::string residuals = read_text_file(root / "gmres_residuals.csv");
    require(residuals.find(",physical,") != std::string::npos
                && residuals.find(",common,") != std::string::npos,
            "residual rows do not distinguish physical/common histories");
    const std::string edgePoints =
        read_text_file(root / "edge_point_diagnostics.csv");
    const std::string edgeFits =
        read_text_file(root / "edge_fit_diagnostics.csv");
    require(edgePoints.find("synthetic,32,edge_reconstructed_value")
                    != std::string::npos
                && edgeFits.find("synthetic,32,edge_reconstructed_value")
                    != std::string::npos,
            "successful shared-edge diagnostics are incomplete");
    require(edgePoints.find("g1_value_g1_normal") == std::string::npos
                && edgePoints.find("direct_cross_face_value")
                    == std::string::npos
                && edgeFits.find("g1_value_g1_normal")
                    == std::string::npos
                && edgeFits.find("direct_cross_face_value")
                    == std::string::npos,
            "control routes invented edge-point or edge-fit rows");
    const auto summaryRecords = parse_csv_records_3d(summary);
    require(csv_value_3d(summaryRecords, "solve_available", "physical_iterations")
                == "NA"
                && csv_value_3d(summaryRecords, "solve_available",
                                "surface_map_count") == "2"
                && csv_value_3d(summaryRecords, "solve_available",
                                "edge_map_count") == "2"
                && csv_value_3d(summaryRecords, "solve_available",
                                "cauchy_geometry_queries") == "9",
            "solve failure did not preserve only its available summary blocks");
    const auto ownerRecords = parse_csv_records_3d(
        read_text_file(root / "owner_diagnostics.csv"));
    require(csv_value_3d(ownerRecords, "solve_available", "available") == "1"
                && csv_value_3d(ownerRecords, "solve_available",
                                "owner_query_count") == "17"
                && csv_value_3d(ownerRecords, "solve_available",
                                "owner_fingerprint_before") == "101"
                && csv_value_3d(ownerRecords, "solve_available",
                                "owner_fingerprint_after") == "NA"
                && csv_value_3d(ownerRecords, "solve_available",
                                "cauchy_fingerprint_before") == "202"
                && csv_value_3d(ownerRecords, "solve_available",
                                "cauchy_fingerprint_after") == "NA",
            "solve failure did not preserve owner/Cauchy before-only fields");
    const auto pointRecords = parse_csv_records_3d(edgePoints);
    const auto dofRecords = parse_csv_records_3d(
        read_text_file(root / "dof_diagnostics.csv"));
    require(csv_value_3d(pointRecords, "solve_available", "exact_value") == "NA"
                && csv_value_3d(dofRecords, "solve_available", "density_error")
                    == "NA",
            "solve failure invented unavailable post-solve diagnostic values");
    const std::filesystem::path script = std::filesystem::current_path()
        / "apps/audit_neumann_two_level_edge_cauchy_3d.ps1";
    require(run_two_level_neumann_schema_audit_3d(script, root) == 0,
            "valid synthetic study directory failed schema audit");
    require_schema_negative_mutations_3d(script, root);
    require_review_schema_negative_mutations_3d(script, root);
    std::filesystem::remove_all(root, error);
}

} // namespace

int main()
{
    try {
        test_harmonic_cauchy_route_names_are_exact();
        test_lprism_routes_have_exact_second_level_structure();
        test_legacy_g1_fixture_is_preserved_by_public_fit();
        test_legacy_n16_short_sectors_preserve_requested_and_actual_counts();
        test_owner_pipeline_and_bordered_operator_split_mu_eta();
        test_common_neumann_rhs_uses_native_parameters_and_surface_weights();
        test_detailed_neumann_probe_uses_literal_defect_and_exact_edge_fit();
        test_structured_common_setup_failure_reaches_all_writer_rows();
        test_two_level_study_writer_and_schema_audit();
        test_failed_coarse_requires_na_adjacent_order();
        test_nonpositive_adjacent_order_error_is_rejected();
        test_two_level_help_separates_unforced_and_legacy_modes();
        test_decision_state_distinguishes_partial_performance_and_failure();
        std::cout << "Neumann exterior value route integration test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Neumann exterior value route integration test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
