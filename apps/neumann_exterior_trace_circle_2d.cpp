#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "src/geometry/curve_2d.hpp"
#include "src/geometry/curve_resampler_2d.hpp"
#include "src/grid/cartesian_grid_2d.hpp"
#include "src/operators/laplace_bvp_2d.hpp"
#include "src/operators/laplace_neumann_exterior_trace_2d.hpp"

using namespace kfbim;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kCx = 0.07;
constexpr double kCy = -0.04;
constexpr double kRadius = 0.50;
constexpr double kGaugeConstant = 0.37;

class OffsetCircle2D final : public ICurve2D {
public:
    Eigen::Vector2d eval(double t) const override
    {
        return {kCx + kRadius * std::cos(t),
                kCy + kRadius * std::sin(t)};
    }

    Eigen::Vector2d deriv(double t) const override
    {
        return {-kRadius * std::sin(t),
                 kRadius * std::cos(t)};
    }

    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }
};

struct BulkErrors {
    double interior_inf = 0.0;
    double interior_rms = 0.0;
    double exterior_inf = 0.0;
};

struct AlignedInteriorErrors {
    // Additive constant applied to the numerical field before comparison.
    double shift = 0.0;
    double inf = 0.0;
    double rms = 0.0;
};

struct CircleMetrics {
    int n = 0;
    int panels = 0;
    int interface_points = 0;
    int iterations = 0;
    bool augmented_converged = false;
    bool physical_converged = false;
    double h = 0.0;
    double rhs_build_sec = 0.0;
    double solve_sec = 0.0;
    double raw_g_mean = 0.0;
    double compatible_g_mean = 0.0;
    double exact_f_mean = 0.0;
    double rhs_inf = 0.0;
    double rhs_route_mismatch = 0.0;
    double a_one_inside_inf = 0.0;
    double a_one_direct_inf = 0.0;
    double a_one_route_mismatch = 0.0;
    double manufactured_residual_inside_inf = 0.0;
    double manufactured_residual_direct_inf = 0.0;
    double manufactured_operator_route_mismatch = 0.0;
    double exact_trace_interior_error_inf = 0.0;
    double exact_exterior_from_jump_inf = 0.0;
    double exact_exterior_virtual_inf = 0.0;
    double exact_exterior_route_mismatch = 0.0;
    double exact_normal_error_inf = 0.0;
    BulkErrors exact_cauchy_bulk;
    double lambda = 0.0;
    double weighted_f_mean = 0.0;
    double augmented_relative_residual = 0.0;
    double physical_relative_residual = 0.0;
    double solved_exterior_from_jump_inf = 0.0;
    double solved_exterior_virtual_inf = 0.0;
    double independent_closure_inf = 0.0;
    double trace_error_inf = 0.0;
    double trace_error_wrms = 0.0;
    double normal_error_inf = 0.0;
    BulkErrors solved_bulk;
    AlignedInteriorErrors new_aligned_bulk;

    int legacy_iterations = 0;
    bool legacy_converged = false;
    double legacy_solve_sec = 0.0;
    double legacy_final_relative_residual = 0.0;
    double legacy_density_mean = 0.0;
    double legacy_raw_normal_residual_inf = 0.0;
    double legacy_projected_normal_residual_inf = 0.0;
    double legacy_raw_normal_residual_mean = 0.0;
    double legacy_weighted_normal_residual_mean = 0.0;
    AlignedInteriorErrors legacy_aligned_bulk;
};

double inf_norm(const Eigen::VectorXd& values)
{
    return values.size() == 0 ? 0.0 : values.cwiseAbs().maxCoeff();
}

double exact_raw_u(double x, double y)
{
    const double X = x - kCx;
    const double Y = y - kCy;
    const double r4 = kRadius * kRadius * kRadius * kRadius;
    return kGaugeConstant
         + (X * X * X * X
            - 6.0 * X * X * Y * Y
            + Y * Y * Y * Y) / r4;
}

Eigen::Vector2d exact_gradient(double x, double y)
{
    const double X = x - kCx;
    const double Y = y - kCy;
    const double r4 = kRadius * kRadius * kRadius * kRadius;
    return {(4.0 * X * X * X - 12.0 * X * Y * Y) / r4,
            (4.0 * Y * Y * Y - 12.0 * X * X * Y) / r4};
}

int environment_int(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : std::stoi(value);
}

double environment_double(const char* name, double fallback)
{
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : std::stod(value);
}

BulkErrors measure_bulk_errors(
    const CartesianGrid2D& grid,
    const GridPair2D& grid_pair,
    const Eigen::VectorXd& field,
    double exact_trace_mean)
{
    BulkErrors errors;
    double sum_sq = 0.0;
    int interior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        const std::array<double, 2> point = grid.coord(node);
        if (grid_pair.domain_label(node) > 0) {
            const double error = std::abs(
                field[node]
                - (exact_raw_u(point[0], point[1]) - exact_trace_mean));
            errors.interior_inf = std::max(errors.interior_inf, error);
            sum_sq += error * error;
            ++interior_count;
        } else {
            errors.exterior_inf =
                std::max(errors.exterior_inf, std::abs(field[node]));
        }
    }
    if (interior_count > 0) {
        errors.interior_rms =
            std::sqrt(sum_sq / static_cast<double>(interior_count));
    }
    return errors;
}

AlignedInteriorErrors measure_aligned_interior_errors(
    const CartesianGrid2D& grid,
    const GridPair2D& grid_pair,
    const Eigen::VectorXd& field,
    double exact_trace_mean)
{
    if (field.size() != grid.num_dofs())
        throw std::invalid_argument(
            "aligned field size must equal grid DOF count");

    AlignedInteriorErrors errors;
    double shift_sum = 0.0;
    int interior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (grid_pair.domain_label(node) <= 0)
            continue;
        const std::array<double, 2> point = grid.coord(node);
        const double exact =
            exact_raw_u(point[0], point[1]) - exact_trace_mean;
        shift_sum += exact - field[node];
        ++interior_count;
    }
    if (interior_count == 0)
        throw std::runtime_error("circle contains no interior grid nodes");
    errors.shift = shift_sum / static_cast<double>(interior_count);

    double sum_sq = 0.0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (grid_pair.domain_label(node) <= 0)
            continue;
        const std::array<double, 2> point = grid.coord(node);
        const double exact =
            exact_raw_u(point[0], point[1]) - exact_trace_mean;
        const double error = field[node] + errors.shift - exact;
        errors.inf = std::max(errors.inf, std::abs(error));
        sum_sq += error * error;
    }
    errors.rms =
        std::sqrt(sum_sq / static_cast<double>(interior_count));
    return errors;
}

CircleMetrics run_level(int n, int max_iter, double tolerance, int restart)
{
    if (n < 16)
        throw std::invalid_argument("circle grid size must be at least 16");

    const double h = 2.0 / static_cast<double>(n);
    CartesianGrid2D grid({-1.0, -1.0},
                         {h, h},
                         {n, n},
                         DofLayout2D::Node);
    const OffsetCircle2D circle;
    const Interface2D iface =
        CurveResampler2D::discretize_quadratic_lagrange(
            circle, h, 2.4);
    LaplaceNeumannExteriorTrace2D solver(grid, iface);
    LaplaceBvpOptions2D legacy_options;
    legacy_options.panel_method =
        LaplaceBvpPanelMethod2D::QuadraticPanelCenter;
    legacy_options.eta = 0.0;
    legacy_options.restrict_stencil_radius = 2;
    legacy_options.correction_method =
        LaplaceCorrectionMethod2D::NearestExpansionCenter;
    LaplaceBvp2D legacy_solver(
        grid,
        iface,
        LaplaceBvpType2D::InteriorNeumann,
        legacy_options);

    const int nq = solver.problem_size();
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(nq);
    Eigen::VectorXd f_raw(nq);
    Eigen::VectorXd g_raw(nq);
    for (int q = 0; q < nq; ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        f_raw[q] = exact_raw_u(x, y);
        g_raw[q] = exact_gradient(x, y).dot(
            iface.normals().row(q).transpose());
    }
    const double exact_f_mean = solver.normalized_weights().dot(f_raw);
    const Eigen::VectorXd f_exact = f_raw - exact_f_mean * ones;

    // Stage 1: construct and expose the fixed RHS before any Krylov solve.
    const auto rhs_build_start = std::chrono::steady_clock::now();
    const LaplaceNeumannExteriorTraceRhs2D rhs_data =
        solver.build_rhs(g_raw);
    const double rhs_build_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - rhs_build_start).count();

    // Stage 2: non-iterative sign, nullspace, route, and manufactured checks.
    const LaplaceExteriorTraceField2D one_field = solver.evaluate_cauchy(
        ones, Eigen::VectorXd::Zero(nq));
    const Eigen::VectorXd a_one_inside =
        one_field.trace_interior - ones;
    const Eigen::VectorXd a_one_direct =
        one_field.trace_exterior_virtual;

    const LaplaceExteriorTraceField2D value_field = solver.evaluate_cauchy(
        f_exact, Eigen::VectorXd::Zero(nq));
    const Eigen::VectorXd af_inside =
        value_field.trace_interior - f_exact;
    const Eigen::VectorXd af_direct =
        value_field.trace_exterior_virtual;

    const LaplaceExteriorTraceField2D exact_cauchy =
        solver.evaluate_cauchy(
            f_exact, rhs_data.compatible_neumann_data);
    const Eigen::VectorXd manufactured_inside =
        exact_cauchy.trace_exterior_from_jump;
    const Eigen::VectorXd manufactured_direct =
        exact_cauchy.trace_exterior_virtual;

    std::cout << "  pre-GMRES N=" << n
              << ": |mean(g_c)|="
              << std::abs(rhs_data.compatibility_residual)
              << " |A_int 1|inf=" << inf_norm(a_one_inside)
              << " |A_ext 1|inf=" << inf_norm(a_one_direct) << '\n';
    std::cout << "    manufactured residual int/ext="
              << inf_norm(manufactured_inside) << '/'
              << inf_norm(manufactured_direct)
              << " exact exterior jump/direct="
              << inf_norm(exact_cauchy.trace_exterior_from_jump) << '/'
              << inf_norm(exact_cauchy.trace_exterior_virtual)
              << " exact bulk exterior="
              << measure_bulk_errors(grid,
                                     solver.grid_pair(),
                                     exact_cauchy.u_bulk,
                                     exact_f_mean).exterior_inf
              << '\n';
    std::cout << "    starting bordered GMRES...\n";

    // Stage 3: only now solve the bordered system, reusing the same RHS.
    const auto solve_start = std::chrono::steady_clock::now();
    const LaplaceNeumannExteriorTraceSolveResult2D result =
        solver.solve(rhs_data, max_iter, tolerance, restart);
    const double solve_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();

    // The legacy formulation solves directly for a normal-jump density.
    // It uses the same grid, interface, Neumann data, transfer options, and
    // Krylov tolerances.  Both PDE volume data and their boundary derivatives
    // vanish for this harmonic manufactured solution.
    std::cout << "    starting legacy projected Neumann GMRES...\n";
    const Eigen::VectorXd zero_bulk =
        Eigen::VectorXd::Zero(grid.num_dofs());
    const std::vector<Eigen::VectorXd> zero_rhs_derivs(
        static_cast<std::size_t>(nq), Eigen::VectorXd::Zero(1));
    const auto legacy_start = std::chrono::steady_clock::now();
    const LaplaceBvpSolveResult2D legacy_result = legacy_solver.solve(
        g_raw,
        zero_bulk,
        zero_rhs_derivs,
        max_iter,
        tolerance,
        restart);
    const double legacy_solve_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - legacy_start).count();

    Eigen::VectorXd legacy_applied;
    legacy_solver.apply(legacy_result.density, legacy_applied);
    const Eigen::VectorXd legacy_raw_normal_residual =
        g_raw - legacy_applied;
    const double legacy_raw_normal_residual_mean =
        legacy_raw_normal_residual.mean();
    const Eigen::VectorXd legacy_projected_normal_residual =
        legacy_raw_normal_residual
        - legacy_raw_normal_residual_mean * ones;

    const Eigen::VectorXd trace_error =
        result.dirichlet_trace - f_exact;
    const double trace_error_wrms = std::sqrt(
        (solver.active_weights().array()
         * trace_error.array().square()).sum()
        / solver.active_weights().sum());

    CircleMetrics metrics;
    metrics.n = n;
    metrics.panels = iface.num_panels();
    metrics.interface_points = nq;
    metrics.iterations = result.iterations;
    metrics.augmented_converged = result.augmented_converged;
    metrics.physical_converged = result.physical_converged;
    metrics.h = h;
    metrics.rhs_build_sec = rhs_build_sec;
    metrics.solve_sec = solve_sec;
    metrics.raw_g_mean = rhs_data.compatibility_mean_removed;
    metrics.compatible_g_mean = rhs_data.compatibility_residual;
    metrics.exact_f_mean = exact_f_mean;
    metrics.rhs_inf = inf_norm(rhs_data.trace_rhs);
    metrics.rhs_route_mismatch = rhs_data.trace_route_mismatch_inf;
    metrics.a_one_inside_inf = inf_norm(a_one_inside);
    metrics.a_one_direct_inf = inf_norm(a_one_direct);
    metrics.a_one_route_mismatch = inf_norm(
        a_one_inside - a_one_direct);
    metrics.manufactured_residual_inside_inf =
        inf_norm(manufactured_inside);
    metrics.manufactured_residual_direct_inf =
        inf_norm(manufactured_direct);
    metrics.manufactured_operator_route_mismatch =
        inf_norm(af_inside - af_direct);
    metrics.exact_trace_interior_error_inf = inf_norm(
        exact_cauchy.trace_interior - f_exact);
    metrics.exact_exterior_from_jump_inf =
        inf_norm(exact_cauchy.trace_exterior_from_jump);
    metrics.exact_exterior_virtual_inf =
        inf_norm(exact_cauchy.trace_exterior_virtual);
    metrics.exact_exterior_route_mismatch =
        exact_cauchy.trace_route_mismatch_inf;
    metrics.exact_normal_error_inf = inf_norm(
        exact_cauchy.normal_trace_interior
        - rhs_data.compatible_neumann_data);
    metrics.exact_cauchy_bulk = measure_bulk_errors(
        grid, solver.grid_pair(), exact_cauchy.u_bulk, exact_f_mean);
    metrics.lambda = result.bordered_multiplier;
    metrics.weighted_f_mean = result.weighted_trace_mean;
    metrics.augmented_relative_residual =
        result.augmented_relative_residual;
    metrics.physical_relative_residual =
        result.physical_relative_residual;
    metrics.solved_exterior_from_jump_inf =
        inf_norm(result.trace_exterior);
    metrics.solved_exterior_virtual_inf =
        inf_norm(result.trace_exterior_virtual);
    metrics.independent_closure_inf =
        result.independent_exterior_closure_inf;
    metrics.trace_error_inf = inf_norm(trace_error);
    metrics.trace_error_wrms = trace_error_wrms;
    metrics.normal_error_inf = inf_norm(
        result.normal_trace_interior
        - result.compatible_neumann_data);
    metrics.solved_bulk = measure_bulk_errors(
        grid, solver.grid_pair(), result.u_bulk, exact_f_mean);
    metrics.new_aligned_bulk = measure_aligned_interior_errors(
        grid, solver.grid_pair(), result.u_bulk, exact_f_mean);
    metrics.legacy_iterations = legacy_result.iterations;
    metrics.legacy_converged = legacy_result.converged;
    metrics.legacy_solve_sec = legacy_solve_sec;
    metrics.legacy_final_relative_residual =
        legacy_result.residuals.empty()
        ? std::numeric_limits<double>::quiet_NaN()
        : legacy_result.residuals.back();
    metrics.legacy_density_mean = legacy_result.density.mean();
    metrics.legacy_raw_normal_residual_inf =
        inf_norm(legacy_raw_normal_residual);
    metrics.legacy_projected_normal_residual_inf =
        inf_norm(legacy_projected_normal_residual);
    metrics.legacy_raw_normal_residual_mean =
        legacy_raw_normal_residual_mean;
    metrics.legacy_weighted_normal_residual_mean =
        solver.normalized_weights().dot(legacy_raw_normal_residual);
    metrics.legacy_aligned_bulk = measure_aligned_interior_errors(
        grid,
        legacy_solver.grid_pair(),
        legacy_result.u_physical,
        exact_f_mean);
    return metrics;
}

void print_metrics(const CircleMetrics& m)
{
    std::cout << "N=" << m.n << " h=" << m.h
              << " panels=" << m.panels
              << " Nq=" << m.interface_points << '\n';
    std::cout << "  RHS: removed_mean=" << m.raw_g_mean
              << " compatible_mean=" << m.compatible_g_mean
              << " |b|inf=" << m.rhs_inf
              << " route_mismatch=" << m.rhs_route_mismatch << '\n';
    std::cout << "  constant: |A_int 1|inf=" << m.a_one_inside_inf
              << " |A_ext 1|inf=" << m.a_one_direct_inf
              << " operator_mismatch=" << m.a_one_route_mismatch << '\n';
    std::cout << "  manufactured: |A_int f-b_int|inf="
              << m.manufactured_residual_inside_inf
              << " |A_ext f-b_ext|inf="
              << m.manufactured_residual_direct_inf
              << " operator_mismatch="
              << m.manufactured_operator_route_mismatch << '\n';
    std::cout << "  exact Cauchy: trace_in_err="
              << m.exact_trace_interior_error_inf
              << " trace_out(jump/direct)="
              << m.exact_exterior_from_jump_inf << '/'
              << m.exact_exterior_virtual_inf
              << " route_mismatch=" << m.exact_exterior_route_mismatch
              << " bulk_in/out=" << m.exact_cauchy_bulk.interior_inf
              << '/' << m.exact_cauchy_bulk.exterior_inf << '\n';
    std::cout << "  solve: iterations=" << m.iterations
              << " augmented_ok=" << (m.augmented_converged ? "yes" : "no")
              << " physical_ok=" << (m.physical_converged ? "yes" : "no")
              << " aug_rel=" << m.augmented_relative_residual
              << " phys_rel=" << m.physical_relative_residual
              << " lambda=" << m.lambda
              << " mean(f)=" << m.weighted_f_mean << '\n';
    std::cout << "  solved errors: trace=" << m.trace_error_inf
              << " normal=" << m.normal_error_inf
              << " trace_out(jump/direct)="
              << m.solved_exterior_from_jump_inf << '/'
              << m.solved_exterior_virtual_inf
              << " closure=" << m.independent_closure_inf
              << " bulk_in/out=" << m.solved_bulk.interior_inf
              << '/' << m.solved_bulk.exterior_inf
              << " time(rhs/solve/total)=" << m.rhs_build_sec << '/'
              << m.solve_sec << '/'
              << (m.rhs_build_sec + m.solve_sec) << "s\n";
    const double aligned_ratio = m.legacy_aligned_bulk.inf > 0.0
        ? m.new_aligned_bulk.inf / m.legacy_aligned_bulk.inf
        : std::numeric_limits<double>::infinity();
    std::cout << "  same-problem bulk (least-squares constant alignment):\n"
              << "    new    Linf/RMS=" << m.new_aligned_bulk.inf << '/'
              << m.new_aligned_bulk.rms
              << " shift=" << m.new_aligned_bulk.shift << '\n'
              << "    legacy Linf/RMS=" << m.legacy_aligned_bulk.inf << '/'
              << m.legacy_aligned_bulk.rms
              << " shift=" << m.legacy_aligned_bulk.shift
              << " new/legacy Linf=" << aligned_ratio << '\n';
    std::cout << "  legacy solve: iterations=" << m.legacy_iterations
              << " converged=" << (m.legacy_converged ? "yes" : "no")
              << " gmres_rel=" << m.legacy_final_relative_residual
              << " density_mean=" << m.legacy_density_mean
              << " normal_residual raw/projected="
              << m.legacy_raw_normal_residual_inf << '/'
              << m.legacy_projected_normal_residual_inf
              << " residual mean arithmetic/weighted="
              << m.legacy_raw_normal_residual_mean << '/'
              << m.legacy_weighted_normal_residual_mean
              << " time=" << m.legacy_solve_sec << "s\n";
}

double observed_order(double coarse, double fine)
{
    if (!(coarse > 0.0) || !(fine > 0.0))
        return std::numeric_limits<double>::quiet_NaN();
    return std::log2(coarse / fine);
}

void print_comparison_summary(const std::vector<CircleMetrics>& rows)
{
    std::cout << "\nSame circle, same g, same grid: aligned interior bulk comparison\n";
    std::cout << "  " << std::setw(5) << "N"
              << std::setw(16) << "new_Linf"
              << std::setw(13) << "new_p"
              << std::setw(16) << "old_Linf"
              << std::setw(13) << "old_p"
              << std::setw(14) << "ratio"
              << std::setw(9) << "new_it"
              << std::setw(9) << "old_it" << '\n';
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const CircleMetrics& m = rows[i];
        const double ratio = m.legacy_aligned_bulk.inf > 0.0
            ? m.new_aligned_bulk.inf / m.legacy_aligned_bulk.inf
            : std::numeric_limits<double>::infinity();
        std::cout << "  " << std::setw(5) << m.n
                  << std::setw(16) << m.new_aligned_bulk.inf;
        if (i == 0) {
            std::cout << std::setw(13) << "-";
        } else {
            std::cout << std::setw(13)
                      << observed_order(rows[i - 1].new_aligned_bulk.inf,
                                        m.new_aligned_bulk.inf);
        }
        std::cout << std::setw(16) << m.legacy_aligned_bulk.inf;
        if (i == 0) {
            std::cout << std::setw(13) << "-";
        } else {
            std::cout << std::setw(13)
                      << observed_order(rows[i - 1].legacy_aligned_bulk.inf,
                                        m.legacy_aligned_bulk.inf);
        }
        std::cout << std::setw(14) << ratio
                  << std::setw(9) << m.iterations
                  << std::setw(9) << m.legacy_iterations << '\n';
    }
}

void write_csv(const std::filesystem::path& path,
               const std::vector<CircleMetrics>& rows)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot open result CSV: " + path.string());
    out << "N,h,panels,Nq,iterations,augmented_converged,physical_converged,"
           "rhs_build_sec,solve_sec,new_total_sec,raw_g_mean,"
           "compatible_g_mean,exact_f_mean,rhs_inf,"
           "rhs_route_mismatch,a_one_inside_inf,a_one_direct_inf,"
           "a_one_route_mismatch,manufactured_residual_inside_inf,"
           "manufactured_residual_direct_inf,manufactured_operator_route_mismatch,"
           "exact_trace_interior_error_inf,exact_exterior_from_jump_inf,"
           "exact_exterior_virtual_inf,exact_exterior_route_mismatch,"
           "exact_normal_error_inf,exact_bulk_in_inf,exact_bulk_in_rms,"
           "exact_bulk_out_inf,lambda,weighted_f_mean,augmented_rel,physical_rel,"
           "solved_exterior_from_jump_inf,solved_exterior_virtual_inf,"
           "independent_closure_inf,trace_error_inf,trace_error_wrms,"
           "normal_error_inf,solved_bulk_in_inf,solved_bulk_in_rms,"
           "solved_bulk_out_inf,new_aligned_shift,new_aligned_inf,"
           "new_aligned_rms,legacy_iterations,legacy_converged,"
           "legacy_solve_sec,legacy_final_rel,legacy_density_mean,"
           "legacy_raw_normal_residual_inf,"
           "legacy_projected_normal_residual_inf,"
           "legacy_raw_normal_residual_mean,"
           "legacy_weighted_normal_residual_mean,legacy_aligned_shift,"
           "legacy_aligned_inf,legacy_aligned_rms\n";
    out << std::setprecision(17);
    for (const CircleMetrics& m : rows) {
        out << m.n << ',' << m.h << ',' << m.panels << ','
            << m.interface_points << ',' << m.iterations << ','
            << (m.augmented_converged ? 1 : 0) << ','
            << (m.physical_converged ? 1 : 0) << ','
            << m.rhs_build_sec << ',' << m.solve_sec << ','
            << (m.rhs_build_sec + m.solve_sec) << ','
            << m.raw_g_mean << ',' << m.compatible_g_mean << ','
            << m.exact_f_mean << ',' << m.rhs_inf << ','
            << m.rhs_route_mismatch << ',' << m.a_one_inside_inf << ','
            << m.a_one_direct_inf << ',' << m.a_one_route_mismatch << ','
            << m.manufactured_residual_inside_inf << ','
            << m.manufactured_residual_direct_inf << ','
            << m.manufactured_operator_route_mismatch << ','
            << m.exact_trace_interior_error_inf << ','
            << m.exact_exterior_from_jump_inf << ','
            << m.exact_exterior_virtual_inf << ','
            << m.exact_exterior_route_mismatch << ','
            << m.exact_normal_error_inf << ','
            << m.exact_cauchy_bulk.interior_inf << ','
            << m.exact_cauchy_bulk.interior_rms << ','
            << m.exact_cauchy_bulk.exterior_inf << ',' << m.lambda << ','
            << m.weighted_f_mean << ',' << m.augmented_relative_residual
            << ',' << m.physical_relative_residual << ','
            << m.solved_exterior_from_jump_inf << ','
            << m.solved_exterior_virtual_inf << ','
            << m.independent_closure_inf << ',' << m.trace_error_inf << ','
            << m.trace_error_wrms << ',' << m.normal_error_inf << ','
            << m.solved_bulk.interior_inf << ','
            << m.solved_bulk.interior_rms << ','
            << m.solved_bulk.exterior_inf << ','
            << m.new_aligned_bulk.shift << ','
            << m.new_aligned_bulk.inf << ','
            << m.new_aligned_bulk.rms << ','
            << m.legacy_iterations << ','
            << (m.legacy_converged ? 1 : 0) << ','
            << m.legacy_solve_sec << ','
            << m.legacy_final_relative_residual << ','
            << m.legacy_density_mean << ','
            << m.legacy_raw_normal_residual_inf << ','
            << m.legacy_projected_normal_residual_inf << ','
            << m.legacy_raw_normal_residual_mean << ','
            << m.legacy_weighted_normal_residual_mean << ','
            << m.legacy_aligned_bulk.shift << ','
            << m.legacy_aligned_bulk.inf << ','
            << m.legacy_aligned_bulk.rms << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        std::cout << std::unitbuf << std::scientific << std::setprecision(6);
        std::vector<int> levels;
        for (int i = 1; i < argc; ++i)
            levels.push_back(std::stoi(argv[i]));
        if (levels.empty())
            levels = {32, 64, 128};

        const int max_iter = environment_int(
            "KFBIM_CIRCLE_EXT_TRACE_MAX_ITER", 40);
        const int restart = environment_int(
            "KFBIM_CIRCLE_EXT_TRACE_RESTART", 40);
        const double tolerance = environment_double(
            "KFBIM_CIRCLE_EXT_TRACE_TOL", 1.0e-9);
        std::cout << "KFBI2D smooth-circle Neumann new-vs-legacy comparison\n"
                  << "tol=" << tolerance << " max_iter=" << max_iter
                  << " restart=" << restart << '\n';

        std::vector<CircleMetrics> results;
        for (int n : levels) {
            CircleMetrics metrics =
                run_level(n, max_iter, tolerance, restart);
            print_metrics(metrics);
            results.push_back(metrics);
        }
        print_comparison_summary(results);

#ifdef KFBIM_APP_OUTPUT_DIR
        const std::filesystem::path output_dir = KFBIM_APP_OUTPUT_DIR;
#else
        const std::filesystem::path output_dir = "output";
#endif
        const std::filesystem::path csv_path =
            output_dir / "neumann_circle_new_vs_legacy_2d.csv";
        write_csv(csv_path, results);
        std::cout << "CSV: " << csv_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
