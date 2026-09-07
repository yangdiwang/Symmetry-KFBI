#include "src/support/geometry/benchmark_nurbs_geometries_2d.hpp"

#include "src/grid/cartesian_grid_2d.hpp"
#include "src/operators/laplace_nsc_et_bvp_2d.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/SVD>

namespace {

using namespace kfbim;
using namespace kfbim::app2d;

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void run_case(LaplaceNscEtBvpType2D type,
              NurbsDensityCoordinate2D coordinate,
              NurbsDensityDerivativeScheme2D derivative_scheme)
{
    constexpr int n = 16;
    constexpr double lower = -1.5;
    constexpr double h = 3.0 / static_cast<double>(n);
    CartesianGrid2D grid({lower, lower}, {h, h}, {n, n},
                         DofLayout2D::Node);
    BenchmarkNurbsGeometry2D geometry =
        make_benchmark_nurbs_geometry_2d(
            BenchmarkNurbsGeometryKind2D::Circle, h, 1.5);
    const int branches = benchmark_nurbs_provider_2d(geometry).num_spans();

    LaplaceNscEtOptions2D options;
    options.density_spacing_over_h = 1.5;
    options.density_coordinate = coordinate;
    options.density_derivative_scheme = derivative_scheme;
    options.phi_connections.assign(
        static_cast<std::size_t>(branches),
        NurbsDensityContinuity2D::C2);
    options.psi_connections.assign(
        static_cast<std::size_t>(branches),
        NurbsDensityContinuity2D::C1);
    options.restrict_options =
        make_laplace_p2_dof_cauchy_exterior_restrict_options_2d();

    LaplaceNscEtBvp2D solver(grid, geometry.interface, type, options);
    const auto& provider = benchmark_nurbs_provider_2d(geometry);
    int density_spans = 0;
    double density_length = 0.0;
    for (int branch = 0; branch < solver.unknown_space().branch_count();
         ++branch) {
        density_spans +=
            solver.unknown_space().branch_info(branch).spline_span_count;
        density_length +=
            solver.unknown_space().branch_info(branch).arclength;
    }
    const auto& references = solver.trace_reference_points();
    require(static_cast<int>(references.size()) >= 4 * density_spans,
            "NSC-ET-KFBI undersampled a density span with Gauss points");
    std::vector<int> references_per_span(
        static_cast<std::size_t>(density_spans), 0);
    for (const auto& reference : references) {
        require(reference.density_span >= 0
                    && reference.density_span < density_spans,
                "NSC-ET-KFBI Gauss reference has an invalid density span");
        ++references_per_span[static_cast<std::size_t>(
            reference.density_span)];
        require(reference.physical_weight > 0.0
                    && std::isfinite(reference.physical_weight),
                "NSC-ET-KFBI Gauss reference has an invalid weight");
        const auto interval = provider.span_interval(reference.branch);
        require(reference.parameter > interval.first
                    && reference.parameter < interval.second,
                "NSC-ET-KFBI Gauss reference is not an interior point");
    }
    for (int count : references_per_span) {
        require(count >= 4 && count % 4 == 0,
                "NSC-ET-KFBI density span does not contain complete Gauss cells");
    }
    const double length_relative_error = std::abs(
        solver.trace_quadrature_weights().sum() - density_length)
        / density_length;
    if (!(length_relative_error < 1.0e-6)) {
        throw std::runtime_error(
            "NSC-ET-KFBI Gauss boundary-length relative error is "
            + std::to_string(length_relative_error));
    }
    const Eigen::MatrixXd& design =
        solver.trace_design_matrix();
    const Eigen::MatrixXd& projection =
        solver.trace_projection_matrix();
    const Eigen::MatrixXd cached_projection = projection;
    require(design.rows() == static_cast<int>(references.size())
                && design.cols() == solver.density_unknown_count(),
            "NSC-ET-KFBI Gauss design has invalid dimensions");
    require(projection.rows() == design.cols()
                && projection.cols() == design.rows(),
            "NSC-ET-KFBI trace projection has invalid dimensions");
    require(solver.trace_projection_build_count() == 1,
            "NSC-ET-KFBI trace projection was not cached exactly once");
    const Eigen::MatrixXd identity_error = projection * design
        - Eigen::MatrixXd::Identity(
            design.cols(), design.cols());
    require(identity_error.cwiseAbs().maxCoeff() < 5.0e-9,
            "NSC-ET-KFBI Gauss projection does not reproduce coefficients");
    Eigen::VectorXd deterministic(design.cols());
    for (int i = 0; i < deterministic.size(); ++i)
        deterministic[i] = std::sin(0.37 * static_cast<double>(i + 1));
    require((projection * (design * deterministic) - deterministic).norm()
                < 2.0e-9 * deterministic.norm(),
            "NSC-ET-KFBI cached trace conversion failed reconstruction");
    require(solver.trace_projection_condition() < 1.0e3,
            "NSC-ET-KFBI weighted Gauss trace design is ill-conditioned");
    if (type == LaplaceNscEtBvpType2D::InteriorNeumann) {
        require(std::abs(solver.neumann_border_row_norm() - 1.0)
                        < 5.0e-13
                    && std::abs(solver.neumann_border_column_norm() - 1.0)
                        < 5.0e-13,
                "NSC-ET-KFBI Neumann border is not norm-balanced");
    }

    NurbsDensitySpace2D::Function boundary;
    if (type == LaplaceNscEtBvpType2D::InteriorDirichlet) {
        boundary = [](int, double, const Eigen::Vector2d& point) {
            return point[0];
        };
    } else {
        boundary = [&provider](int branch,
                               double parameter,
                               const Eigen::Vector2d&) {
            return provider.evaluate_on_span(branch, parameter).normal[0];
        };
    }

    const LaplaceNscEtSolveResult2D result =
        solver.solve_harmonic(boundary, 80, 1.0e-8, 40);
    require(solver.trace_projection_build_count() == 1
                && (solver.trace_projection_matrix()
                    - cached_projection).norm() == 0.0,
            "NSC-ET-KFBI rebuilt or changed its cached trace conversion");
    require(result.gmres_converged,
            "NSC-ET-KFBI smoke GMRES did not converge");
    require(result.bordered_residual_relative < 2.0e-7,
            "NSC-ET-KFBI smoke bordered residual is too large");
    require(result.exterior_trace_relative < 1.5e-1
                && result.exterior_trace.allFinite(),
            "NSC-ET-KFBI smoke raw Gauss exterior trace is too large");
    if (type == LaplaceNscEtBvpType2D::InteriorNeumann) {
        require(std::abs(result.density_mean) < 2.0e-7,
                "NSC-ET-KFBI rank-one Neumann gauge did not fix the mean");
    }
    require(result.u_bulk.allFinite(),
            "NSC-ET-KFBI smoke bulk solution is nonfinite");

    double interior_error = 0.0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (solver.grid_pair().domain_label(node) <= 0)
            continue;
        const auto point = grid.coord(node);
        interior_error = std::max(
            interior_error, std::abs(result.u_bulk[node] - point[0]));
    }
    require(interior_error < 8.0e-2,
            "NSC-ET-KFBI smoke interior error is too large");

    const auto& diagnostics = solver.restrict_diagnostics();
    require(diagnostics.reference_trace_points
                    == static_cast<int>(references.size())
                && diagnostics.reference_trace_samples
                    == 6 * static_cast<int>(references.size())
                && diagnostics.reference_interpolation_nodes
                    == 36 * static_cast<int>(references.size()),
            "NSC-ET-KFBI reference restrict diagnostics are incomplete");
    require(diagnostics.reference_unresolved_gap_fallback_owners == 0,
            "NSC-ET-KFBI Gauss references used a gap fallback");
    require(diagnostics.reference_endpoint_fallback_owners == 0,
            "NSC-ET-KFBI Gauss references used an endpoint fallback");
}

void test_rejects_c2_psi_continuity()
{
    constexpr int n = 16;
    constexpr double lower = -1.5;
    constexpr double h = 3.0 / static_cast<double>(n);
    CartesianGrid2D grid({lower, lower}, {h, h}, {n, n},
                         DofLayout2D::Node);
    BenchmarkNurbsGeometry2D geometry =
        make_benchmark_nurbs_geometry_2d(
            BenchmarkNurbsGeometryKind2D::Circle, h, 1.5);
    const int branches =
        benchmark_nurbs_provider_2d(geometry).num_spans();

    LaplaceNscEtOptions2D options;
    options.phi_connections.assign(
        static_cast<std::size_t>(branches),
        NurbsDensityContinuity2D::C2);
    options.psi_connections.assign(
        static_cast<std::size_t>(branches),
        NurbsDensityContinuity2D::C2);
    bool rejected = false;
    try {
        LaplaceNscEtBvp2D solver(
            grid,
            geometry.interface,
            LaplaceNscEtBvpType2D::InteriorDirichlet,
            options);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected,
            "NSC-ET-KFBI accepted illegal C2 continuity for P2 psi");
}

} // namespace

int main()
{
    try {
        const std::vector<std::pair<
            NurbsDensityCoordinate2D,
            NurbsDensityDerivativeScheme2D>> methods{
            {NurbsDensityCoordinate2D::PhysicalArclength,
             NurbsDensityDerivativeScheme2D::AnalyticBasis},
            {NurbsDensityCoordinate2D::PhysicalArclength,
             NurbsDensityDerivativeScheme2D::SampledFiniteDifference},
            {NurbsDensityCoordinate2D::LegacyNurbsParameter,
             NurbsDensityDerivativeScheme2D::CovariantParameterFiniteDifference}};
        for (const auto& method : methods) {
            run_case(LaplaceNscEtBvpType2D::InteriorDirichlet,
                     method.first,
                     method.second);
            run_case(LaplaceNscEtBvpType2D::InteriorNeumann,
                     method.first,
                     method.second);
        }
        test_rejects_c2_psi_continuity();
        std::cout << "NSC-ET-KFBI 2D BVP smoke tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "laplace_nsc_et_bvp_2d_test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
