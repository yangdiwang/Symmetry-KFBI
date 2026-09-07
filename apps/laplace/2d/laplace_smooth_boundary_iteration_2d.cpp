#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "src/geometry/curve_2d.hpp"
#include "src/geometry/curve_resampler_2d.hpp"
#include "src/grid/cartesian_grid_2d.hpp"
#include "src/operators/laplace_bvp_2d.hpp"

using namespace kfbim;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kBoxMin = -2.5;
constexpr double kBoxSide = 5.0;
constexpr const char* kSpreadLabel = "p2_quadratic_cauchy";
constexpr const char* kRestrictLabel = "native_six_point_quadratic";
constexpr const char* kCorrectionLabel = "nearest_expansion_center";

enum class ProblemKind {
    Dirichlet,
    Neumann
};

Eigen::Matrix2d rotation(double angle_degrees)
{
    const double angle = angle_degrees * kPi / 180.0;
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    Eigen::Matrix2d result;
    result << c, -s,
              s,  c;
    return result;
}

class RotatedEllipseCurve2D final : public ICurve2D {
public:
    RotatedEllipseCurve2D()
        : rotation_(rotation(27.0))
    {}

    Eigen::Vector2d eval(double t) const override
    {
        const Eigen::Vector2d local(
            1.10 * std::cos(t), 0.76 * std::sin(t));
        return Eigen::Vector2d(0.13, -0.08) + rotation_ * local;
    }

    Eigen::Vector2d deriv(double t) const override
    {
        const Eigen::Vector2d local(
            -1.10 * std::sin(t), 0.76 * std::cos(t));
        return rotation_ * local;
    }

    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }

private:
    Eigen::Matrix2d rotation_;
};

class SmoothFlowerCurve2D final : public ICurve2D {
public:
    SmoothFlowerCurve2D()
        : rotation_(rotation(14.0))
    {}

    Eigen::Vector2d eval(double t) const override
    {
        const double r = radius(t);
        const Eigen::Vector2d local(r * std::cos(t), r * std::sin(t));
        return Eigen::Vector2d(0.09, -0.06) + rotation_ * local;
    }

    Eigen::Vector2d deriv(double t) const override
    {
        const double r = radius(t);
        const double rp = radius_derivative(t);
        const Eigen::Vector2d local(
            rp * std::cos(t) - r * std::sin(t),
            rp * std::sin(t) + r * std::cos(t));
        return rotation_ * local;
    }

    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }

private:
    static double radius(double theta)
    {
        return 0.88 * (1.0 + 0.18 * std::cos(5.0 * theta));
    }

    static double radius_derivative(double theta)
    {
        return -0.88 * 0.18 * 5.0 * std::sin(5.0 * theta);
    }

    Eigen::Matrix2d rotation_;
};

std::unique_ptr<ICurve2D> make_curve(const std::string& geometry)
{
    if (geometry == "ellipse")
        return std::make_unique<RotatedEllipseCurve2D>();
    if (geometry == "flower")
        return std::make_unique<SmoothFlowerCurve2D>();
    throw std::invalid_argument("geometry must be both, ellipse, or flower");
}

double exact_solution(double x, double y)
{
    constexpr double a = 0.42;
    return std::exp(a * x) * std::cos(a * y)
         + 0.08 * (x * x - y * y) + 0.11 * x - 0.07 * y;
}

Eigen::Vector2d exact_gradient(double x, double y)
{
    constexpr double a = 0.42;
    const double exponential = std::exp(a * x);
    return {
        a * exponential * std::cos(a * y) + 0.16 * x + 0.11,
       -a * exponential * std::sin(a * y) - 0.16 * y - 0.07
    };
}

bool analytic_inside(const std::string& geometry, double x, double y)
{
    Eigen::Vector2d translated;
    double angle_degrees = 0.0;
    if (geometry == "ellipse") {
        translated = Eigen::Vector2d(x - 0.13, y + 0.08);
        angle_degrees = 27.0;
    } else if (geometry == "flower") {
        translated = Eigen::Vector2d(x - 0.09, y + 0.06);
        angle_degrees = 14.0;
    } else {
        throw std::invalid_argument("geometry must be ellipse or flower");
    }

    const Eigen::Vector2d local =
        rotation(angle_degrees).transpose() * translated;
    if (geometry == "ellipse") {
        const double scaled_x = local[0] / 1.10;
        const double scaled_y = local[1] / 0.76;
        return scaled_x * scaled_x + scaled_y * scaled_y <= 1.0;
    }

    const double theta = std::atan2(local[1], local[0]);
    const double radius = 0.88 * (1.0 + 0.18 * std::cos(5.0 * theta));
    return local.norm() <= radius;
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

const char* problem_name(ProblemKind problem)
{
    return problem == ProblemKind::Dirichlet ? "dirichlet" : "neumann";
}

const char* formulation_name(ProblemKind problem)
{
    return problem == ProblemKind::Dirichlet
        ? "value_jump_interior_dirichlet"
        : "normal_jump_projected_interior_neumann";
}

const char* gmres_target_name(ProblemKind problem)
{
    return problem == ProblemKind::Dirichlet
        ? "raw_dirichlet_boundary_trace"
        : "arithmetic_mean_projected_neumann_trace";
}

LaplaceBvpType2D bvp_type(ProblemKind problem)
{
    return problem == ProblemKind::Dirichlet
        ? LaplaceBvpType2D::InteriorDirichlet
        : LaplaceBvpType2D::InteriorNeumann;
}

double inf_norm(const Eigen::VectorXd& values)
{
    return values.size() == 0 ? 0.0 : values.cwiseAbs().maxCoeff();
}

struct BulkError {
    double linf = 0.0;
    double rms = 0.0;
    int count = 0;
};

bool selected_bulk_node(const std::string& geometry,
                        const CartesianGrid2D& grid,
                        const GridPair2D& grid_pair,
                        int node,
                        bool use_analytic_mask)
{
    if (!use_analytic_mask)
        return grid_pair.domain_label(node) > 0;
    const auto point = grid.coord(node);
    return analytic_inside(geometry, point[0], point[1]);
}

double neumann_alignment_shift(const std::string& geometry,
                               const CartesianGrid2D& grid,
                               const GridPair2D& grid_pair,
                               const Eigen::VectorXd& field)
{
    double sum = 0.0;
    int count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (!selected_bulk_node(
                geometry, grid, grid_pair, node, true)) {
            continue;
        }
        const auto point = grid.coord(node);
        sum += exact_solution(point[0], point[1]) - field[node];
        ++count;
    }
    if (count == 0)
        throw std::runtime_error("geometry contains no analytic interior nodes");
    return sum / static_cast<double>(count);
}

BulkError measure_bulk_error(const std::string& geometry,
                             const CartesianGrid2D& grid,
                             const GridPair2D& grid_pair,
                             bool use_analytic_mask,
                             const Eigen::VectorXd& field,
                             double alignment_shift)
{
    BulkError result;
    double sum_sq = 0.0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (!selected_bulk_node(
                geometry, grid, grid_pair, node, use_analytic_mask)) {
            continue;
        }
        const auto point = grid.coord(node);
        const double error =
            field[node] + alignment_shift
            - exact_solution(point[0], point[1]);
        result.linf = std::max(result.linf, std::abs(error));
        sum_sq += error * error;
        ++result.count;
    }
    if (result.count == 0)
        throw std::runtime_error("geometry contains no selected interior nodes");
    result.rms = std::sqrt(sum_sq / static_cast<double>(result.count));
    return result;
}

struct StudyResult {
    ProblemKind problem = ProblemKind::Dirichlet;
    std::string geometry;
    int n = 0;
    double h = 0.0;
    double panel_length_over_h = 0.0;
    int panels = 0;
    int dofs = 0;
    int gmres_iterations = 0;
    bool converged = false;
    double gmres_final_relative_residual =
        std::numeric_limits<double>::quiet_NaN();
    double solve_seconds = 0.0;
    double boundary_residual_raw_mean = 0.0;
    double boundary_residual_weighted_mean = 0.0;
    double boundary_residual_raw_linf = 0.0;
    double boundary_residual_mean_projected_linf = 0.0;
    double boundary_data_arithmetic_mean = 0.0;
    double boundary_data_weighted_mean = 0.0;
    double constant_alignment_shift = 0.0;
    BulkError analytic_bulk;
    BulkError native_bulk;
    double analytic_linf_order = std::numeric_limits<double>::quiet_NaN();
    double analytic_rms_order = std::numeric_limits<double>::quiet_NaN();
    double native_linf_order = std::numeric_limits<double>::quiet_NaN();
    double native_rms_order = std::numeric_limits<double>::quiet_NaN();
};

StudyResult run_one(ProblemKind problem,
                    const std::string& geometry,
                    int n,
                    double panel_length_over_h,
                    int max_iter,
                    double tolerance,
                    int restart)
{
    if (n < 24)
        throw std::invalid_argument("N must be at least 24");

    const double h = kBoxSide / static_cast<double>(n);
    CartesianGrid2D grid(
        {kBoxMin, kBoxMin}, {h, h}, {n, n}, DofLayout2D::Node);
    const std::unique_ptr<ICurve2D> curve = make_curve(geometry);
    const Interface2D iface =
        CurveResampler2D::discretize_quadratic_lagrange(
            *curve, h, panel_length_over_h);

    // LaplaceBvp2D currently fixes this panel method to its default
    // QuadraticCauchy spread and native six-point quadratic restrict.
    LaplaceBvpOptions2D options;
    options.panel_method =
        LaplaceBvpPanelMethod2D::QuadraticPanelCenter;
    options.eta = 0.0;
    options.restrict_stencil_radius = 2;
    options.correction_method =
        LaplaceCorrectionMethod2D::NearestExpansionCenter;
    LaplaceBvp2D solver(grid, iface, bvp_type(problem), options);

    const Interface2D& effective_iface = solver.grid_pair().interface();
    const int nq = solver.problem_size();
    Eigen::VectorXd boundary_data(nq);
    for (int active = 0; active < nq; ++active) {
        const int q = solver.active_interface_point(active);
        const double x = effective_iface.points()(q, 0);
        const double y = effective_iface.points()(q, 1);
        if (problem == ProblemKind::Dirichlet) {
            boundary_data[active] = exact_solution(x, y);
        } else {
            boundary_data[active] =
                exact_gradient(x, y).dot(
                    effective_iface.normals().row(q).transpose());
        }
    }

    const Eigen::VectorXd f_bulk =
        Eigen::VectorXd::Zero(grid.num_dofs());
    const std::vector<Eigen::VectorXd> rhs_derivs(
        effective_iface.num_points(), Eigen::VectorXd::Zero(1));

    const auto solve_start = std::chrono::steady_clock::now();
    const LaplaceBvpSolveResult2D solution =
        solver.solve(boundary_data,
                     f_bulk,
                     rhs_derivs,
                     max_iter,
                     tolerance,
                     restart);
    const double solve_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();

    Eigen::VectorXd computed_boundary;
    solver.apply(solution.density, computed_boundary);
    const Eigen::VectorXd boundary_residual =
        computed_boundary - boundary_data;
    Eigen::VectorXd projected_boundary_residual = boundary_residual;
    projected_boundary_residual.array() -= boundary_residual.mean();

    double boundary_weight_sum = 0.0;
    double boundary_data_weighted_sum = 0.0;
    double boundary_residual_weighted_sum = 0.0;
    for (int active = 0; active < nq; ++active) {
        const int q = solver.active_interface_point(active);
        const double weight = effective_iface.weights()[q];
        boundary_weight_sum += weight;
        boundary_data_weighted_sum += weight * boundary_data[active];
        boundary_residual_weighted_sum +=
            weight * boundary_residual[active];
    }
    if (!(boundary_weight_sum > 0.0))
        throw std::runtime_error("boundary quadrature weight sum must be positive");

    const double alignment_shift =
        problem == ProblemKind::Neumann
        ? neumann_alignment_shift(
              geometry,
              grid,
              solver.grid_pair(),
              solution.u_physical)
        : 0.0;
    const BulkError analytic_bulk =
        measure_bulk_error(geometry,
                           grid,
                           solver.grid_pair(),
                           true,
                           solution.u_physical,
                           alignment_shift);
    const BulkError native_bulk =
        measure_bulk_error(geometry,
                           grid,
                           solver.grid_pair(),
                           false,
                           solution.u_physical,
                           alignment_shift);

    StudyResult result;
    result.problem = problem;
    result.geometry = geometry;
    result.n = n;
    result.h = h;
    result.panel_length_over_h = panel_length_over_h;
    result.panels = iface.num_panels();
    result.dofs = nq;
    result.gmres_iterations = solution.iterations;
    result.converged = solution.converged;
    if (!solution.residuals.empty()) {
        result.gmres_final_relative_residual =
            solution.residuals.back();
    }
    result.solve_seconds = solve_seconds;
    result.boundary_residual_raw_mean = boundary_residual.mean();
    result.boundary_residual_weighted_mean =
        boundary_residual_weighted_sum / boundary_weight_sum;
    result.boundary_residual_raw_linf = inf_norm(boundary_residual);
    result.boundary_residual_mean_projected_linf =
        inf_norm(projected_boundary_residual);
    result.boundary_data_arithmetic_mean = boundary_data.mean();
    result.boundary_data_weighted_mean =
        boundary_data_weighted_sum / boundary_weight_sum;
    result.constant_alignment_shift = alignment_shift;
    result.analytic_bulk = analytic_bulk;
    result.native_bulk = native_bulk;
    return result;
}

double observed_order(double coarse_error,
                      double fine_error,
                      double coarse_h,
                      double fine_h)
{
    if (!(coarse_error > 0.0) || !(fine_error > 0.0)
        || !(coarse_h > 0.0) || !(fine_h > 0.0)
        || coarse_h == fine_h) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::log(coarse_error / fine_error)
         / std::log(coarse_h / fine_h);
}

void add_orders(const StudyResult& coarse, StudyResult& fine)
{
    fine.analytic_linf_order =
        observed_order(coarse.analytic_bulk.linf,
                       fine.analytic_bulk.linf,
                       coarse.h,
                       fine.h);
    fine.analytic_rms_order =
        observed_order(coarse.analytic_bulk.rms,
                       fine.analytic_bulk.rms,
                       coarse.h,
                       fine.h);
    fine.native_linf_order =
        observed_order(coarse.native_bulk.linf,
                       fine.native_bulk.linf,
                       coarse.h,
                       fine.h);
    fine.native_rms_order =
        observed_order(coarse.native_bulk.rms,
                       fine.native_bulk.rms,
                       coarse.h,
                       fine.h);
}

void print_order(double order)
{
    if (std::isfinite(order))
        std::cout << order;
    else
        std::cout << '-';
}

void print_result(const StudyResult& result)
{
    std::cout << "  result"
              << " panels=" << result.panels
              << " dofs=" << result.dofs
              << " GMRES=" << result.gmres_iterations
              << " gmres_target=" << gmres_target_name(result.problem)
              << " gmres_converged="
              << (result.converged ? "yes" : "no")
              << " final_relres="
              << result.gmres_final_relative_residual
              << " bc_raw_mean=" << result.boundary_residual_raw_mean
              << " bc_weighted_mean="
              << result.boundary_residual_weighted_mean
              << " bc_raw_Linf=" << result.boundary_residual_raw_linf
              << " bc_mean_projected_Linf="
              << result.boundary_residual_mean_projected_linf;
    if (result.problem == ProblemKind::Neumann) {
        std::cout
                  << " g_arithmetic_mean="
                  << result.boundary_data_arithmetic_mean
                  << " g_weighted_mean="
                  << result.boundary_data_weighted_mean
                  << " shift=" << result.constant_alignment_shift;
    }
    std::cout << " analytic_bulk(Linf/RMS)="
              << result.analytic_bulk.linf << '/'
              << result.analytic_bulk.rms
              << " order(";
    print_order(result.analytic_linf_order);
    std::cout << '/';
    print_order(result.analytic_rms_order);
    std::cout << ") native_bulk(Linf/RMS)="
              << result.native_bulk.linf << '/'
              << result.native_bulk.rms
              << " order(";
    print_order(result.native_linf_order);
    std::cout << '/';
    print_order(result.native_rms_order);
    std::cout << ") time=" << result.solve_seconds << "s\n";
}

void write_csv(const std::filesystem::path& path,
               const std::vector<StudyResult>& results)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot open CSV: " + path.string());

    out << "problem,formulation,gmres_target,geometry,N,h,"
           "panel_length_over_h,panels,dofs,"
           "spread,restrict,correction,gmres_iterations,"
           "gmres_target_converged,"
           "gmres_final_relative_residual,solve_seconds,"
           "boundary_residual_raw_mean,boundary_residual_weighted_mean,"
           "boundary_residual_raw_linf,"
           "boundary_residual_mean_projected_linf,"
           "boundary_data_arithmetic_mean,boundary_data_weighted_mean,"
           "constant_alignment_shift,"
           "analytic_bulk_linf,analytic_bulk_rms,analytic_bulk_count,"
           "native_bulk_linf,native_bulk_rms,native_bulk_count,"
           "analytic_linf_order,analytic_rms_order,"
           "native_linf_order,native_rms_order\n";
    out << std::setprecision(17);
    for (const StudyResult& result : results) {
        out << problem_name(result.problem) << ','
            << formulation_name(result.problem) << ','
            << gmres_target_name(result.problem) << ','
            << result.geometry << ','
            << result.n << ','
            << result.h << ','
            << result.panel_length_over_h << ','
            << result.panels << ','
            << result.dofs << ','
            << kSpreadLabel << ','
            << kRestrictLabel << ','
            << kCorrectionLabel << ','
            << result.gmres_iterations << ','
            << (result.converged ? 1 : 0) << ','
            << result.gmres_final_relative_residual << ','
            << result.solve_seconds << ','
            << result.boundary_residual_raw_mean << ','
            << result.boundary_residual_weighted_mean << ','
            << result.boundary_residual_raw_linf << ','
            << result.boundary_residual_mean_projected_linf << ','
            << result.boundary_data_arithmetic_mean << ','
            << result.boundary_data_weighted_mean << ','
            << result.constant_alignment_shift << ','
            << result.analytic_bulk.linf << ','
            << result.analytic_bulk.rms << ','
            << result.analytic_bulk.count << ','
            << result.native_bulk.linf << ','
            << result.native_bulk.rms << ','
            << result.native_bulk.count << ','
            << result.analytic_linf_order << ','
            << result.analytic_rms_order << ','
            << result.native_linf_order << ','
            << result.native_rms_order << '\n';
    }
}

std::vector<ProblemKind> parse_problems(const std::string& selection)
{
    if (selection == "both")
        return {ProblemKind::Dirichlet, ProblemKind::Neumann};
    if (selection == "dirichlet")
        return {ProblemKind::Dirichlet};
    if (selection == "neumann")
        return {ProblemKind::Neumann};
    throw std::invalid_argument(
        "problem must be both, dirichlet, or neumann");
}

std::vector<std::string> parse_geometries(const std::string& selection)
{
    if (selection == "both")
        return {"ellipse", "flower"};
    if (selection == "ellipse" || selection == "flower")
        return {selection};
    throw std::invalid_argument(
        "geometry must be both, ellipse, or flower");
}

void print_usage(const char* program)
{
    std::cerr
        << "usage: " << program
        << " [both|dirichlet|neumann]"
           " [both|ellipse|flower] [N ...]\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc > 1 && (std::string(argv[1]) == "-h"
                         || std::string(argv[1]) == "--help")) {
            print_usage(argv[0]);
            return 0;
        }

        const std::vector<ProblemKind> problems =
            parse_problems(argc >= 2 ? argv[1] : "both");
        const std::vector<std::string> geometries =
            parse_geometries(argc >= 3 ? argv[2] : "both");
        std::vector<int> levels = {48, 72, 108, 162};
        if (argc >= 4) {
            levels.clear();
            for (int argument = 3; argument < argc; ++argument)
                levels.push_back(std::stoi(argv[argument]));
        }

        const double panel_length_over_h = environment_double(
            "KFBIM_SMOOTH_BVP_PANEL_LENGTH_OVER_H", 1.6);
        const int max_iter = environment_int(
            "KFBIM_SMOOTH_BVP_MAX_ITER", 100);
        const int restart = environment_int(
            "KFBIM_SMOOTH_BVP_RESTART", 80);
        const double tolerance = environment_double(
            "KFBIM_SMOOTH_BVP_TOL", 2.0e-10);

        std::cout << std::unitbuf
                  << std::scientific << std::setprecision(6);
        std::cout
            << "KFBI2D smooth-boundary core LaplaceBvp2D iteration study\n"
            << "box=[" << kBoxMin << ',' << kBoxMin + kBoxSide
            << "]^2 panel_length/h=" << panel_length_over_h
            << " tol=" << tolerance
            << " max_iter=" << max_iter
            << " restart=" << restart
            << " spread=" << kSpreadLabel
            << " restrict=" << kRestrictLabel
            << " correction=" << kCorrectionLabel << '\n';

        std::vector<StudyResult> results;
        for (const ProblemKind problem : problems) {
            for (const std::string& geometry : geometries) {
                bool have_previous = false;
                StudyResult previous;
                for (const int n : levels) {
                    std::cout << "[run] problem=" << problem_name(problem)
                              << " geometry=" << geometry
                              << " N=" << n << '\n';
                    StudyResult result =
                        run_one(problem,
                                geometry,
                                n,
                                panel_length_over_h,
                                max_iter,
                                tolerance,
                                restart);
                    if (have_previous)
                        add_orders(previous, result);
                    print_result(result);
                    previous = result;
                    have_previous = true;
                    results.push_back(std::move(result));
                }
            }
        }

#ifdef KFBIM_APP_OUTPUT_DIR
        const std::filesystem::path output_dir = KFBIM_APP_OUTPUT_DIR;
#else
        const std::filesystem::path output_dir = "output";
#endif
        const std::filesystem::path csv_path =
            output_dir / "laplace_smooth_boundary_iteration_2d.csv";
        write_csv(csv_path, results);
        std::cout << "CSV: " << csv_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        print_usage(argv[0]);
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
