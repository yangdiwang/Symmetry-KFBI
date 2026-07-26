#include <apps/harmonic_cauchy_fit_3d.hpp>

#define main neumann_exterior_zero_trace_3d_application_main
#include "neumann_exterior_zero_trace_3d.cpp"
#undef main

#include <algorithm>
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
        std::cout << "Neumann exterior value route integration test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Neumann exterior value route integration test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
