#include "benchmark_nurbs_geometries_2d.hpp"

#include "src/grid/cartesian_grid_2d.hpp"
#include "src/operators/laplace_nsc_et_bvp_2d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/SVD>

namespace {

using namespace kfbim;
using namespace kfbim::app2d;

constexpr double kBoxLower = -1.5;
constexpr double kBoxLength = 3.0;

enum class RequestedBvp { Dirichlet, Neumann };

struct DensityMethod {
    NurbsDensityCoordinate2D coordinate =
        NurbsDensityCoordinate2D::PhysicalArclength;
    NurbsDensityDerivativeScheme2D derivative =
        NurbsDensityDerivativeScheme2D::SampledFiniteDifference;
};

bool is_compatible_density_method(const DensityMethod& method)
{
    switch (method.derivative) {
    case NurbsDensityDerivativeScheme2D::AnalyticBasis:
        return true;
    case NurbsDensityDerivativeScheme2D::SampledFiniteDifference:
        return method.coordinate
            == NurbsDensityCoordinate2D::PhysicalArclength;
    case NurbsDensityDerivativeScheme2D::CovariantParameterFiniteDifference:
        return method.coordinate
            == NurbsDensityCoordinate2D::LegacyNurbsParameter;
    }
    return false;
}

std::vector<DensityMethod> three_way_density_methods()
{
    return {
        {NurbsDensityCoordinate2D::PhysicalArclength,
         NurbsDensityDerivativeScheme2D::AnalyticBasis},
        {NurbsDensityCoordinate2D::PhysicalArclength,
         NurbsDensityDerivativeScheme2D::SampledFiniteDifference},
        {NurbsDensityCoordinate2D::LegacyNurbsParameter,
         NurbsDensityDerivativeScheme2D::CovariantParameterFiniteDifference}};
}

const char* derivative_short_name(NurbsDensityDerivativeScheme2D scheme)
{
    switch (scheme) {
    case NurbsDensityDerivativeScheme2D::AnalyticBasis:
        return "an";
    case NurbsDensityDerivativeScheme2D::SampledFiniteDifference:
        return "sfd";
    case NurbsDensityDerivativeScheme2D::CovariantParameterFiniteDifference:
        return "cfd";
    }
    return "invalid";
}

struct Metrics {
    std::string scheme;
    std::string geometry;
    std::string bvp;
    int n = 0;
    double h = 0.0;
    double mean_density_spacing_over_h = 0.0;
    double difference_step_over_span = 0.0;
    int density_spans = 0;
    int raw_coefficients = 0;
    int reduced_coefficients = 0;
    int solver_unknowns = 0;
    int trace_points = 0;
    double density_collocation_condition = 0.0;
    int iterations = 0;
    bool converged = false;
    double exterior_relative = 0.0;
    double exterior_inf = 0.0;
    double projected_exterior_relative = 0.0;
    double bordered_residual_relative = 0.0;
    double exterior_weighted_mean = 0.0;
    double density_mean = 0.0;
    double neumann_gauge_multiplier = 0.0;
    double projection_identity_residual = 0.0;
    int projection_build_count = 0;
    double boundary_inf = 0.0;
    double density_inf = 0.0;
    double bulk_inf = 0.0;
    double bulk_rms = 0.0;
    double exterior_bulk_inf = 0.0;
    double bulk_order = std::numeric_limits<double>::quiet_NaN();
    double boundary_order = std::numeric_limits<double>::quiet_NaN();
    double density_order = std::numeric_limits<double>::quiet_NaN();
    int exact_crossings = 0;
    int unresolved_gap = 0;
    int endpoint_fallback = 0;
    int relocated_stencils = 0;
    int rejected_candidates = 0;
    double seconds = 0.0;
};

double exact_u(double x, double y)
{
    return x * x * x - 3.0 * x * y * y
         + 0.35 * (x * x - y * y)
         + 0.20 * x - 0.15 * y + 0.10;
}

Eigen::Vector2d exact_gradient(double x, double y)
{
    return {3.0 * x * x - 3.0 * y * y + 0.70 * x + 0.20,
            -6.0 * x * y - 0.70 * y - 0.15};
}

double order(double previous_error,
             double current_error,
             double previous_h,
             double current_h)
{
    if (!(previous_error > 0.0) || !(current_error > 0.0)
        || !(previous_h > current_h) || !(current_h > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::log(previous_error / current_error)
         / std::log(previous_h / current_h);
}

std::pair<std::vector<NurbsDensityContinuity2D>,
          std::vector<NurbsDensityContinuity2D>>
continuity_for(const BenchmarkNurbsGeometry2D& geometry)
{
    const int branches = benchmark_nurbs_provider_2d(geometry).num_spans();
    std::vector<NurbsDensityContinuity2D> phi(
        static_cast<std::size_t>(branches),
        NurbsDensityContinuity2D::C2);
    std::vector<NurbsDensityContinuity2D> psi(
        static_cast<std::size_t>(branches),
        NurbsDensityContinuity2D::C1);
    for (const BenchmarkNurbsFeature2D& feature : geometry.features) {
        if (feature.span_minus < 0 || feature.span_minus >= branches)
            throw std::logic_error("benchmark feature has an invalid branch");
        if (feature.continuity == BenchmarkGeometryContinuity2D::G1) {
            phi[static_cast<std::size_t>(feature.span_minus)] =
                NurbsDensityContinuity2D::C1;
            psi[static_cast<std::size_t>(feature.span_minus)] =
                NurbsDensityContinuity2D::C0;
        } else if (feature.continuity
                   == BenchmarkGeometryContinuity2D::G0) {
            phi[static_cast<std::size_t>(feature.span_minus)] =
                NurbsDensityContinuity2D::C0;
            psi[static_cast<std::size_t>(feature.span_minus)] =
                NurbsDensityContinuity2D::Discontinuous;
        }
    }
    return {std::move(phi), std::move(psi)};
}

double boundary_mean_exact_u(const LaplaceNscEtBvp2D& solver)
{
    double integral = 0.0;
    double length = 0.0;
    const auto& references = solver.trace_reference_points();
    const Eigen::VectorXd& weights = solver.trace_quadrature_weights();
    for (int q = 0; q < static_cast<int>(references.size()); ++q) {
        const double weight = weights[q];
        const Eigen::Vector2d& point =
            references[static_cast<std::size_t>(q)].point;
        integral += weight * exact_u(point[0], point[1]);
        length += weight;
    }
    if (!(length > 0.0))
        throw std::runtime_error("benchmark boundary has zero quadrature length");
    return integral / length;
}

Metrics run_level(BenchmarkNurbsGeometryKind2D kind,
                  RequestedBvp requested_bvp,
                  int n,
                  int max_iter,
                  double tolerance,
                  int restart,
                  double density_spacing_over_h,
                  NurbsDensityCoordinate2D density_coordinate,
                  NurbsDensityDerivativeScheme2D derivative_scheme,
                  double difference_step_over_span,
                  bool use_orthogonal_neumann_gauge,
                  bool use_full_neumann_bordered_system,
                  int trace_gauss_order)
{
    if (n < 16)
        throw std::invalid_argument("NSC-ET benchmark N must be at least 16");
    const double h = kBoxLength / static_cast<double>(n);
    CartesianGrid2D grid({kBoxLower, kBoxLower},
                         {h, h},
                         {n, n},
                         DofLayout2D::Node);
    BenchmarkNurbsGeometry2D geometry =
        make_benchmark_nurbs_geometry_2d(
            kind, h, density_spacing_over_h);
    const auto connections = continuity_for(geometry);

    LaplaceNscEtOptions2D options;
    options.density_spacing_over_h = density_spacing_over_h;
    options.density_coordinate = density_coordinate;
    options.density_derivative_scheme = derivative_scheme;
    options.density_difference_step_over_span =
        difference_step_over_span;
    options.covariant_parameter_difference_step_over_span =
        difference_step_over_span;
    options.use_orthogonal_neumann_gauge =
        use_orthogonal_neumann_gauge;
    options.use_full_neumann_bordered_system =
        use_full_neumann_bordered_system;
    options.trace_gauss_order = trace_gauss_order;
    options.phi_connections = connections.first;
    options.psi_connections = connections.second;
    // Empty intentionally lets the operator augment midpoint candidates with
    // smooth compatibility endpoints before rank-revealing selection.
    options.trace_candidates.clear();
    options.restrict_options =
        make_laplace_p2_dof_cauchy_exterior_restrict_options_2d();

    const LaplaceNscEtBvpType2D type =
        requested_bvp == RequestedBvp::Dirichlet
        ? LaplaceNscEtBvpType2D::InteriorDirichlet
        : LaplaceNscEtBvpType2D::InteriorNeumann;
    LaplaceNscEtBvp2D solver(grid, geometry.interface, type, options);
    const auto& provider = benchmark_nurbs_provider_2d(geometry);

    NurbsDensitySpace2D::Function boundary_function;
    if (requested_bvp == RequestedBvp::Dirichlet) {
        boundary_function = [](int,
                               double,
                               const Eigen::Vector2d& point) {
            return exact_u(point[0], point[1]);
        };
    } else {
        boundary_function = [&provider](int branch,
                                        double parameter,
                                        const Eigen::Vector2d& point) {
            const Eigen::Vector2d normal =
                provider.evaluate_on_span(branch, parameter).normal;
            return exact_gradient(point[0], point[1]).dot(normal);
        };
    }

    const auto start = std::chrono::steady_clock::now();
    const LaplaceNscEtSolveResult2D result = solver.solve_harmonic(
        boundary_function, max_iter, tolerance, restart);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();

    const double gauge = requested_bvp == RequestedBvp::Neumann
        ? boundary_mean_exact_u(solver) : 0.0;
    double boundary_inf = 0.0;
    double density_inf = 0.0;
    const auto& trace_points = solver.trace_reference_points();
    for (int row = 0; row < static_cast<int>(trace_points.size()); ++row) {
        const LaplaceNurbsTraceReferencePoint2D& reference =
            trace_points[static_cast<std::size_t>(row)];
        const Eigen::Vector2d point = reference.point;
        const Eigen::Vector2d normal = reference.normal;
        const double value = exact_u(point[0], point[1]) - gauge;
        const double flux = exact_gradient(point[0], point[1]).dot(normal);
        boundary_inf = std::max(
            boundary_inf,
            std::abs(requested_bvp == RequestedBvp::Dirichlet
                ? result.physical_trace[row] - value
                : result.physical_normal_trace[row] - flux));
        density_inf = std::max(
            density_inf,
            std::abs(result.unknown_at_trace_points[row]
                - (requested_bvp == RequestedBvp::Dirichlet
                    ? flux : value)));
    }

    double bulk_inf = 0.0;
    double bulk_sum_sq = 0.0;
    double exterior_bulk_inf = 0.0;
    int interior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (solver.grid_pair().domain_label(node) > 0) {
            const auto coordinate = grid.coord(node);
            const double error = std::abs(
                result.u_bulk[node]
                - (exact_u(coordinate[0], coordinate[1]) - gauge));
            bulk_inf = std::max(bulk_inf, error);
            bulk_sum_sq += error * error;
            ++interior_count;
        } else {
            exterior_bulk_inf = std::max(
                exterior_bulk_inf, std::abs(result.u_bulk[node]));
        }
    }
    if (interior_count == 0)
        throw std::runtime_error("benchmark geometry contains no grid nodes");

    const NurbsDensitySpace2D& density_space = solver.unknown_space();
    double total_length = 0.0;
    int density_spans = 0;
    for (int branch = 0; branch < density_space.branch_count(); ++branch) {
        total_length += density_space.branch_info(branch).arclength;
        density_spans +=
            density_space.branch_info(branch).spline_span_count;
    }
    const auto& diagnostics = solver.restrict_diagnostics();
    Metrics metrics;
    metrics.scheme = density_coordinate
            == NurbsDensityCoordinate2D::PhysicalArclength
        ? "nsc_et_gauss_coeff_periodic_arclength"
        : "nsc_et_gauss_coeff_parameter_charts";
    switch (derivative_scheme) {
    case NurbsDensityDerivativeScheme2D::AnalyticBasis:
        metrics.scheme += "_analytic_derivative";
        break;
    case NurbsDensityDerivativeScheme2D::SampledFiniteDifference:
        metrics.scheme += "_sampled_fd";
        break;
    case NurbsDensityDerivativeScheme2D::CovariantParameterFiniteDifference:
        metrics.scheme += "_covariant_fd";
        break;
    }
    metrics.scheme += "_q" + std::to_string(trace_gauss_order);
    if (requested_bvp == RequestedBvp::Neumann) {
        metrics.scheme += use_full_neumann_bordered_system
            ? "_bordered_gauge"
            : (use_orthogonal_neumann_gauge
                   ? "_orthogonal_gauge" : "_pivot_gauge");
    } else {
        metrics.scheme += "_no_gauge";
    }
    metrics.geometry = geometry.name;
    metrics.bvp = requested_bvp == RequestedBvp::Dirichlet
        ? "dirichlet" : "neumann";
    metrics.n = n;
    metrics.h = h;
    metrics.mean_density_spacing_over_h =
        total_length / static_cast<double>(density_spans) / h;
    metrics.difference_step_over_span = difference_step_over_span;
    metrics.density_spans = density_spans;
    metrics.raw_coefficients = solver.raw_density_unknown_count();
    metrics.reduced_coefficients = solver.density_unknown_count();
    metrics.solver_unknowns = solver.problem_size();
    metrics.trace_points = static_cast<int>(trace_points.size());
    metrics.density_collocation_condition =
        solver.trace_projection_condition();
    metrics.iterations = result.iterations;
    metrics.converged = result.gmres_converged;
    metrics.exterior_relative = result.exterior_trace_relative;
    metrics.exterior_inf = result.exterior_trace_inf;
    metrics.projected_exterior_relative =
        result.projected_exterior_trace_relative;
    metrics.bordered_residual_relative =
        result.bordered_residual_relative;
    metrics.exterior_weighted_mean =
        result.exterior_trace_weighted_mean;
    metrics.density_mean = result.density_mean;
    metrics.neumann_gauge_multiplier =
        result.neumann_gauge_multiplier;
    metrics.projection_identity_residual =
        solver.trace_projection_identity_residual();
    metrics.projection_build_count =
        solver.trace_projection_build_count();
    metrics.boundary_inf = boundary_inf;
    metrics.density_inf = density_inf;
    metrics.bulk_inf = bulk_inf;
    metrics.bulk_rms = std::sqrt(
        bulk_sum_sq / static_cast<double>(interior_count));
    metrics.exterior_bulk_inf = exterior_bulk_inf;
    metrics.exact_crossings =
        diagnostics.reference_exact_crossing_owners;
    metrics.unresolved_gap =
        diagnostics.reference_unresolved_gap_fallback_owners;
    metrics.endpoint_fallback =
        diagnostics.reference_endpoint_fallback_owners;
    metrics.relocated_stencils =
        diagnostics.reference_unified_p2_relocated_stencils;
    metrics.rejected_candidates =
        diagnostics.reference_unified_p2_candidate_rejections;
    metrics.seconds = seconds;
    return metrics;
}

void write_header(std::ostream& out)
{
    out << "scheme,geometry,bvp,N,h,mean_H_over_h,difference_step_over_span,density_spans,"
           "raw_coefficients,reduced_coefficients,solver_unknowns,trace_points,"
           "weighted_trace_design_condition,"
           "gmres_iterations,gmres_converged,raw_gauss_trace_relative,"
           "raw_gauss_trace_linf,projected_trace_relative,"
           "bordered_residual_relative,"
           "raw_gauss_weighted_mean,density_mean,neumann_gauge_multiplier,"
           "projection_identity_residual,"
           "projection_build_count,boundary_linf,boundary_order,density_linf,"
           "density_order,bulk_linf,bulk_order,bulk_rms,exterior_bulk_linf,"
           "exact_crossings,unresolved_gap_fallback,endpoint_fallback,"
           "relocated_stencils,rejected_candidates,seconds\n";
}

void write_row(std::ostream& out, const Metrics& m)
{
    out << m.scheme << ',' << m.geometry << ',' << m.bvp << ','
        << m.n << ',' << m.h << ',' << m.mean_density_spacing_over_h << ','
        << m.difference_step_over_span << ',' << m.density_spans << ','
        << m.raw_coefficients << ','
        << m.reduced_coefficients << ',' << m.solver_unknowns << ','
        << m.trace_points << ',' << m.density_collocation_condition << ','
        << m.iterations << ','
        << (m.converged ? 1 : 0) << ',' << m.exterior_relative << ','
        << m.exterior_inf << ',' << m.projected_exterior_relative << ','
        << m.bordered_residual_relative << ','
        << m.exterior_weighted_mean << ',' << m.density_mean << ','
        << m.neumann_gauge_multiplier << ','
        << m.projection_identity_residual << ','
        << m.projection_build_count << ',' << m.boundary_inf << ','
        << m.boundary_order << ',' << m.density_inf << ','
        << m.density_order << ',' << m.bulk_inf << ',' << m.bulk_order << ','
        << m.bulk_rms << ',' << m.exterior_bulk_inf << ','
        << m.exact_crossings << ',' << m.unresolved_gap << ','
        << m.endpoint_fallback << ',' << m.relocated_stencils << ','
        << m.rejected_candidates << ',' << m.seconds << '\n';
}

std::vector<BenchmarkNurbsGeometryKind2D> all_geometries()
{
    return {BenchmarkNurbsGeometryKind2D::Circle,
            BenchmarkNurbsGeometryKind2D::Ellipse,
            BenchmarkNurbsGeometryKind2D::Flower,
            BenchmarkNurbsGeometryKind2D::Heart,
            BenchmarkNurbsGeometryKind2D::LShape};
}

} // namespace

int main(int argc, char** argv)
{
    try {
        std::vector<BenchmarkNurbsGeometryKind2D> geometries =
            all_geometries();
        std::vector<RequestedBvp> bvps{
            RequestedBvp::Dirichlet, RequestedBvp::Neumann};
        std::vector<NurbsDensityCoordinate2D> density_coordinates{
            NurbsDensityCoordinate2D::PhysicalArclength};
        std::vector<NurbsDensityDerivativeScheme2D> derivative_schemes{
            NurbsDensityDerivativeScheme2D::SampledFiniteDifference};
        std::vector<DensityMethod> density_methods;
        bool density_method_flag = false;
        bool density_axis_flag = false;
        double difference_step_over_span = 0.5;
        double covariant_difference_step_over_span = 0.125;
        double density_spacing_over_h = 1.5;
        bool use_orthogonal_neumann_gauge = true;
        bool use_full_neumann_bordered_system = true;
        int trace_gauss_order = 4;
        std::vector<int> levels;
        std::filesystem::path requested_csv;
        for (int arg = 1; arg < argc; ++arg) {
            const std::string token(argv[arg]);
            if (token == "--geometry" && arg + 1 < argc) {
                const std::string name(argv[++arg]);
                geometries = name == "all"
                    ? all_geometries()
                    : std::vector<BenchmarkNurbsGeometryKind2D>{
                          parse_benchmark_nurbs_geometry_2d(name)};
            } else if (token == "--bvp" && arg + 1 < argc) {
                const std::string name(argv[++arg]);
                if (name == "both") {
                    bvps = {RequestedBvp::Dirichlet,
                            RequestedBvp::Neumann};
                } else if (name == "dirichlet") {
                    bvps = {RequestedBvp::Dirichlet};
                } else if (name == "neumann") {
                    bvps = {RequestedBvp::Neumann};
                } else {
                    throw std::invalid_argument(
                        "--bvp must be both, dirichlet, or neumann");
                }
            } else if (token == "--output-csv" && arg + 1 < argc) {
                requested_csv = argv[++arg];
            } else if (token == "--density-method" && arg + 1 < argc) {
                density_method_flag = true;
                const std::string name(argv[++arg]);
                if (name == "analytic") {
                    density_methods = {{
                        NurbsDensityCoordinate2D::PhysicalArclength,
                        NurbsDensityDerivativeScheme2D::AnalyticBasis}};
                } else if (name == "arclength-fd"
                           || name == "sampled") {
                    density_methods = {{
                        NurbsDensityCoordinate2D::PhysicalArclength,
                        NurbsDensityDerivativeScheme2D::SampledFiniteDifference}};
                } else if (name == "covariant-parameter-fd"
                           || name == "covariant") {
                    density_methods = {{
                        NurbsDensityCoordinate2D::LegacyNurbsParameter,
                        NurbsDensityDerivativeScheme2D::CovariantParameterFiniteDifference}};
                } else if (name == "all" || name == "three-way") {
                    density_methods = three_way_density_methods();
                } else {
                    throw std::invalid_argument(
                        "--density-method must be analytic, arclength-fd, covariant-parameter-fd, or all");
                }
            } else if (token == "--density-coordinate"
                       && arg + 1 < argc) {
                density_axis_flag = true;
                const std::string name(argv[++arg]);
                if (name == "arclength" || name == "periodic") {
                    density_coordinates = {
                        NurbsDensityCoordinate2D::PhysicalArclength};
                } else if (name == "legacy" || name == "xi"
                           || name == "parameter" || name == "nurbs") {
                    density_coordinates = {
                        NurbsDensityCoordinate2D::LegacyNurbsParameter};
                } else if (name == "both") {
                    density_coordinates = {
                        NurbsDensityCoordinate2D::LegacyNurbsParameter,
                        NurbsDensityCoordinate2D::PhysicalArclength};
                } else {
                    throw std::invalid_argument(
                        "--density-coordinate must be arclength, parameter, or both");
                }
            } else if (token == "--density-spacing-over-h"
                       && arg + 1 < argc) {
                density_spacing_over_h = std::stod(argv[++arg]);
                if (!(density_spacing_over_h > 0.0)
                    || !std::isfinite(density_spacing_over_h)) {
                    throw std::invalid_argument(
                        "--density-spacing-over-h must be finite and positive");
                }
            } else if (token == "--density-derivative"
                       && arg + 1 < argc) {
                density_axis_flag = true;
                const std::string name(argv[++arg]);
                if (name == "sampled" || name == "fd") {
                    derivative_schemes = {
                        NurbsDensityDerivativeScheme2D::SampledFiniteDifference};
                } else if (name == "analytic" || name == "basis") {
                    derivative_schemes = {
                        NurbsDensityDerivativeScheme2D::AnalyticBasis};
                } else if (name == "covariant"
                           || name == "parameter-fd") {
                    derivative_schemes = {
                        NurbsDensityDerivativeScheme2D::CovariantParameterFiniteDifference};
                } else if (name == "both") {
                    derivative_schemes = {
                        NurbsDensityDerivativeScheme2D::AnalyticBasis,
                        NurbsDensityDerivativeScheme2D::SampledFiniteDifference};
                } else if (name == "all") {
                    derivative_schemes = {
                        NurbsDensityDerivativeScheme2D::AnalyticBasis,
                        NurbsDensityDerivativeScheme2D::SampledFiniteDifference,
                        NurbsDensityDerivativeScheme2D::CovariantParameterFiniteDifference};
                } else {
                    throw std::invalid_argument(
                        "--density-derivative must be sampled, analytic, covariant, both, or all");
                }
            } else if (token == "--density-difference-step-over-span"
                       && arg + 1 < argc) {
                difference_step_over_span = std::stod(argv[++arg]);
                if (!(difference_step_over_span > 0.0)
                    || !std::isfinite(difference_step_over_span)) {
                    throw std::invalid_argument(
                        "--density-difference-step-over-span must be finite and positive");
                }
                covariant_difference_step_over_span =
                    difference_step_over_span;
            } else if (token
                           == "--covariant-difference-step-over-span"
                       && arg + 1 < argc) {
                covariant_difference_step_over_span =
                    std::stod(argv[++arg]);
                if (!(covariant_difference_step_over_span > 0.0)
                    || !std::isfinite(
                        covariant_difference_step_over_span)) {
                    throw std::invalid_argument(
                        "--covariant-difference-step-over-span must be finite and positive");
                }
            } else if (token == "--neumann-gauge" && arg + 1 < argc) {
                const std::string name(argv[++arg]);
                if (name == "bordered") {
                    use_full_neumann_bordered_system = true;
                } else if (name == "orthogonal") {
                    use_full_neumann_bordered_system = false;
                    use_orthogonal_neumann_gauge = true;
                } else if (name == "pivot" || name == "legacy") {
                    use_full_neumann_bordered_system = false;
                    use_orthogonal_neumann_gauge = false;
                } else {
                    throw std::invalid_argument(
                        "--neumann-gauge must be bordered, orthogonal, or pivot");
                }
            } else if (token == "--trace-gauss-order"
                       && arg + 1 < argc) {
                trace_gauss_order = std::stoi(argv[++arg]);
                if (trace_gauss_order != 2 && trace_gauss_order != 3
                    && trace_gauss_order != 4
                    && trace_gauss_order != 8) {
                    throw std::invalid_argument(
                        "--trace-gauss-order must be 2, 3, 4, or 8");
                }
            } else {
                levels.push_back(std::stoi(token));
            }
        }
        if (levels.empty())
            levels = {32, 64, 128};
        std::sort(levels.begin(), levels.end());
        if (density_method_flag && density_axis_flag) {
            throw std::invalid_argument(
                "--density-method cannot be combined with density coordinate/derivative axes");
        }
        if (!density_method_flag) {
            for (NurbsDensityCoordinate2D coordinate : density_coordinates) {
                for (NurbsDensityDerivativeScheme2D derivative
                     : derivative_schemes) {
                    const DensityMethod method{coordinate, derivative};
                    if (is_compatible_density_method(method))
                        density_methods.push_back(method);
                }
            }
        }
        if (density_methods.empty()) {
            throw std::invalid_argument(
                "the selected density coordinate/derivative axes contain no compatible method");
        }

        constexpr int max_iter = 160;
        constexpr int restart = 80;
        constexpr double tolerance = 1.0e-9;

#ifdef KFBIM_APP_OUTPUT_DIR
        const std::filesystem::path output_dir(KFBIM_APP_OUTPUT_DIR);
#else
        const std::filesystem::path output_dir("output");
#endif
        const std::filesystem::path csv_path = requested_csv.empty()
            ? output_dir / "laplace_nsc_et_benchmark_2d.csv"
            : requested_csv;
        if (!csv_path.parent_path().empty())
            std::filesystem::create_directories(csv_path.parent_path());
        std::ofstream csv(csv_path);
        if (!csv)
            throw std::runtime_error("cannot open NSC-ET benchmark CSV");
        csv << std::setprecision(16);
        write_header(csv);

        std::map<std::tuple<std::string, std::string, std::string>, Metrics>
            previous;
        std::cout << std::scientific << std::setprecision(4);
        std::cout << "NSC-ET-KFBI exact-NURBS coefficient benchmark\n";
        for (const DensityMethod& density_method : density_methods) {
                const NurbsDensityCoordinate2D density_coordinate =
                    density_method.coordinate;
                const NurbsDensityDerivativeScheme2D derivative_scheme =
                    density_method.derivative;
                for (BenchmarkNurbsGeometryKind2D geometry : geometries) {
                    for (RequestedBvp bvp : bvps) {
                        for (int n : levels) {
                        Metrics metrics = run_level(
                            geometry,
                            bvp,
                            n,
                            max_iter,
                            tolerance,
                            restart,
                            density_spacing_over_h,
                            density_coordinate,
                            derivative_scheme,
                            derivative_scheme
                                    == NurbsDensityDerivativeScheme2D::CovariantParameterFiniteDifference
                                ? covariant_difference_step_over_span
                                : difference_step_over_span,
                            use_orthogonal_neumann_gauge,
                            use_full_neumann_bordered_system,
                            trace_gauss_order);
                    const auto key = std::make_tuple(
                        metrics.scheme, metrics.geometry, metrics.bvp);
                    const auto found = previous.find(key);
                    if (found != previous.end()) {
                        metrics.bulk_order = order(
                            found->second.bulk_inf, metrics.bulk_inf,
                            found->second.h, metrics.h);
                        metrics.boundary_order = order(
                            found->second.boundary_inf, metrics.boundary_inf,
                            found->second.h, metrics.h);
                        metrics.density_order = order(
                            found->second.density_inf, metrics.density_inf,
                            found->second.h, metrics.h);
                    }
                    previous[key] = metrics;
                    write_row(csv, metrics);
                    csv.flush();
                    std::cout << std::setw(8) << metrics.geometry << ' '
                              << std::setw(9) << metrics.bvp
                              << ' ' << (density_coordinate
                                      == NurbsDensityCoordinate2D::PhysicalArclength
                                  ? "arc" : "xi ")
                              << ' ' << derivative_short_name(derivative_scheme)
                              << " N=" << std::setw(4) << metrics.n
                              << " coeff=" << std::setw(4)
                              << metrics.solver_unknowns
                              << " it=" << std::setw(3)
                              << metrics.iterations
                              << " ext=" << metrics.exterior_inf
                              << " bulk=" << metrics.bulk_inf
                              << " p=" << metrics.bulk_order
                              << " fallback="
                              << metrics.unresolved_gap + metrics.endpoint_fallback
                              << '\n';
                        }
                    }
                }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "laplace_nsc_et_benchmark_2d failure: "
                  << error.what() << '\n';
        return 1;
    }
}
