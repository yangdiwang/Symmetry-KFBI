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
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "src/geometry/curve_2d.hpp"
#include "src/geometry/curve_resampler_2d.hpp"
#include "src/grid/cartesian_grid_2d.hpp"
#include "src/operators/laplace_neumann_exterior_trace_2d.hpp"

using namespace kfbim;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kBoxMin = -2.5;
constexpr double kBoxSide = 5.0;

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
    throw std::invalid_argument("geometry must be ellipse or flower");
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

double inf_norm(const Eigen::VectorXd& values)
{
    return values.size() == 0 ? 0.0 : values.cwiseAbs().maxCoeff();
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

struct BulkError {
    double shift = 0.0;
    double linf = 0.0;
    double l2 = 0.0;
    int count = 0;
};

BulkError measure_interior_bulk(
    const std::string& geometry,
    const CartesianGrid2D& grid,
    const GridPair2D& grid_pair,
    bool use_analytic_mask,
    const Eigen::VectorXd& field)
{
    const auto dims = grid.dof_dims();
    BulkError result;
    double shift_sum = 0.0;
    for (int j = 1; j + 1 < dims[1]; ++j) {
        for (int i = 1; i + 1 < dims[0]; ++i) {
            const int node = grid.index(i, j);
            const auto point = grid.coord(i, j);
            const bool is_inside = use_analytic_mask
                ? analytic_inside(geometry, point[0], point[1])
                : grid_pair.domain_label(node) > 0;
            if (!is_inside)
                continue;
            shift_sum += exact_solution(point[0], point[1]) - field[node];
            ++result.count;
        }
    }
    if (result.count == 0)
        throw std::runtime_error("geometry contains no interior grid nodes");
    result.shift = shift_sum / static_cast<double>(result.count);

    double sum_sq = 0.0;
    for (int j = 1; j + 1 < dims[1]; ++j) {
        for (int i = 1; i + 1 < dims[0]; ++i) {
            const int node = grid.index(i, j);
            const auto point = grid.coord(i, j);
            const bool is_inside = use_analytic_mask
                ? analytic_inside(geometry, point[0], point[1])
                : grid_pair.domain_label(node) > 0;
            if (!is_inside)
                continue;
            const double error = field[node] + result.shift
                               - exact_solution(point[0], point[1]);
            result.linf = std::max(result.linf, std::abs(error));
            sum_sq += error * error;
        }
    }
    result.l2 = std::sqrt(sum_sq / static_cast<double>(result.count));
    return result;
}

struct StudyResult {
    std::string geometry;
    int n = 0;
    double h = 0.0;
    double panel_length_over_h = 0.0;
    int dofs = 0;
    int panels = 0;
    int iterations = 0;
    bool augmented_converged = false;
    double seconds = 0.0;
    double flux_mean_removed = 0.0;
    double lambda = 0.0;
    double trace_mean = 0.0;
    double augmented_boundary_res_linf = 0.0;
    double raw_exterior_trace_linf = 0.0;
    double raw_exterior_trace_l2 = 0.0;
    double exterior_from_jump_linf = 0.0;
    double exterior_route_mismatch_linf = 0.0;
    double physical_relative_residual = 0.0;
    double global_linf = 0.0;
    double global_l2 = 0.0;
    double constant_shift = 0.0;
    int n_global = 0;
    double native_global_linf = 0.0;
    double native_global_l2 = 0.0;
    double native_constant_shift = 0.0;
    int n_native_global = 0;
    double trace_linf = 0.0;
    double trace_l2 = 0.0;
    double global_linf_order = std::numeric_limits<double>::quiet_NaN();
    double global_l2_order = std::numeric_limits<double>::quiet_NaN();
};

StudyResult run_one(
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
    const auto interface_start = std::chrono::steady_clock::now();
    const Interface2D iface =
        CurveResampler2D::discretize_quadratic_lagrange(
            *curve, h, panel_length_over_h);
    const double interface_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - interface_start).count();
    std::cout << "  setup interface: panels=" << iface.num_panels()
              << " dofs=" << iface.num_points()
              << " time=" << interface_sec << "s\n";

    const auto solver_setup_start = std::chrono::steady_clock::now();
    LaplaceNeumannExteriorTrace2D solver(grid, iface);
    const double solver_setup_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solver_setup_start).count();
    std::cout << "  setup operator: time=" << solver_setup_sec << "s\n";

    const int nq = solver.problem_size();
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(nq);
    Eigen::VectorXd exact_trace_raw(nq);
    Eigen::VectorXd normal_data(nq);
    for (int q = 0; q < nq; ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        exact_trace_raw[q] = exact_solution(x, y);
        normal_data[q] = exact_gradient(x, y).dot(
            iface.normals().row(q).transpose());
    }
    const double exact_trace_mean =
        solver.normalized_weights().dot(exact_trace_raw);
    const Eigen::VectorXd exact_trace =
        exact_trace_raw - exact_trace_mean * ones;

    const auto start = std::chrono::steady_clock::now();
    const LaplaceNeumannExteriorTraceRhs2D rhs_data =
        solver.build_rhs(normal_data);
    const double rhs_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "  fixed RHS: time=" << rhs_sec << "s\n";
    const LaplaceNeumannExteriorTraceSolveResult2D result =
        solver.solve(rhs_data, max_iter, tolerance, restart);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();

    const BulkError bulk = measure_interior_bulk(
        geometry, grid, solver.grid_pair(), true, result.u_bulk);
    const BulkError native_bulk = measure_interior_bulk(
        geometry, grid, solver.grid_pair(), false, result.u_bulk);
    const Eigen::VectorXd trace_error =
        result.dirichlet_trace - exact_trace;

    StudyResult metrics;
    metrics.geometry = geometry;
    metrics.n = n;
    metrics.h = h;
    metrics.panel_length_over_h = panel_length_over_h;
    metrics.dofs = nq;
    metrics.panels = iface.num_panels();
    metrics.iterations = result.iterations;
    metrics.augmented_converged = result.augmented_converged;
    metrics.seconds = seconds;
    metrics.flux_mean_removed = rhs_data.compatibility_mean_removed;
    metrics.lambda = result.bordered_multiplier;
    metrics.trace_mean = result.weighted_trace_mean;
    metrics.augmented_boundary_res_linf =
        inf_norm(result.augmented_residual.head(nq));
    // The Python tutorial's raw exterior trace is independently recovered
    // from exterior virtual samples.  Keep the jump-derived trace as a
    // separate diagnostic because the production C++ equation deliberately
    // obtains R_ext from the stable interior route and the analytic jump.
    metrics.raw_exterior_trace_linf =
        inf_norm(result.trace_exterior_virtual);
    metrics.raw_exterior_trace_l2 = std::sqrt(
        result.trace_exterior_virtual.squaredNorm()
        / static_cast<double>(nq));
    metrics.exterior_from_jump_linf = inf_norm(result.trace_exterior);
    metrics.exterior_route_mismatch_linf = inf_norm(
        result.trace_exterior_virtual - result.trace_exterior);
    metrics.physical_relative_residual = result.physical_relative_residual;
    metrics.global_linf = bulk.linf;
    metrics.global_l2 = bulk.l2;
    metrics.constant_shift = bulk.shift;
    metrics.n_global = bulk.count;
    metrics.native_global_linf = native_bulk.linf;
    metrics.native_global_l2 = native_bulk.l2;
    metrics.native_constant_shift = native_bulk.shift;
    metrics.n_native_global = native_bulk.count;
    metrics.trace_linf = inf_norm(trace_error);
    metrics.trace_l2 =
        std::sqrt(trace_error.squaredNorm() / static_cast<double>(nq));
    return metrics;
}

double observed_order(double coarse_error,
                      double fine_error,
                      double coarse_h,
                      double fine_h)
{
    return std::log(coarse_error / fine_error)
         / std::log(coarse_h / fine_h);
}

void add_orders(std::vector<StudyResult>& results)
{
    for (const std::string geometry : {"ellipse", "flower"}) {
        StudyResult* previous = nullptr;
        for (StudyResult& result : results) {
            if (result.geometry != geometry)
                continue;
            if (previous != nullptr) {
                result.global_linf_order = observed_order(
                    previous->global_linf,
                    result.global_linf,
                    previous->h,
                    result.h);
                result.global_l2_order = observed_order(
                    previous->global_l2,
                    result.global_l2,
                    previous->h,
                    result.h);
            }
            previous = &result;
        }
    }
}

void print_result(const StudyResult& result)
{
    std::cout << "  dofs=" << result.dofs
              << " panels=" << result.panels
              << " gmres=" << result.iterations
              << " aug_ok=" << (result.augmented_converged ? "yes" : "no")
              << " aug_res=" << result.augmented_boundary_res_linf
              << " raw_ext=" << result.raw_exterior_trace_linf
              << " Linf=" << result.global_linf
              << " L2=" << result.global_l2
              << " native_Linf=" << result.native_global_linf
              << " trace=" << result.trace_linf
              << " lambda=" << result.lambda
              << " time=" << result.seconds << "s\n";
}

void print_summary(const std::vector<StudyResult>& results)
{
    for (const std::string geometry : {"ellipse", "flower"}) {
        std::cout << "\nC++ P2 exterior-trace summary: " << geometry << '\n';
        std::cout << "  " << std::setw(5) << "N"
                  << std::setw(7) << "dofs"
                  << std::setw(14) << "Linf"
                  << std::setw(11) << "p_inf"
                  << std::setw(14) << "L2"
                  << std::setw(11) << "p_L2"
                  << std::setw(9) << "GMRES" << '\n';
        for (const StudyResult& result : results) {
            if (result.geometry != geometry)
                continue;
            std::cout << "  " << std::setw(5) << result.n
                      << std::setw(7) << result.dofs
                      << std::setw(14) << result.global_linf;
            if (std::isfinite(result.global_linf_order))
                std::cout << std::setw(11) << result.global_linf_order;
            else
                std::cout << std::setw(11) << "-";
            std::cout << std::setw(14) << result.global_l2;
            if (std::isfinite(result.global_l2_order))
                std::cout << std::setw(11) << result.global_l2_order;
            else
                std::cout << std::setw(11) << "-";
            std::cout << std::setw(9) << result.iterations << '\n';
        }
    }
}

void write_csv(const std::filesystem::path& path,
               const std::vector<StudyResult>& results)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot open CSV: " + path.string());
    out << "geometry,N,h,panel_length_over_h,dofs,panels,gmres_iterations,augmented_converged,"
           "seconds,augmented_boundary_res_linf,raw_exterior_trace_linf,"
           "raw_exterior_trace_l2,exterior_from_jump_linf,"
           "exterior_route_mismatch_linf,"
           "flux_mean_removed,lambda,trace_mean,physical_relative_residual,"
           "global_linf,global_l2,constant_shift,n_global,"
           "native_global_linf,native_global_l2,native_constant_shift,"
           "n_native_global,trace_linf,trace_l2,"
           "global_linf_order,global_l2_order\n";
    out << std::setprecision(17);
    for (const StudyResult& result : results) {
        out << result.geometry << ',' << result.n << ',' << result.h << ','
            << result.panel_length_over_h << ','
            << result.dofs << ',' << result.panels << ','
            << result.iterations << ','
            << (result.augmented_converged ? 1 : 0) << ','
            << result.seconds << ','
            << result.augmented_boundary_res_linf << ','
            << result.raw_exterior_trace_linf << ','
            << result.raw_exterior_trace_l2 << ','
            << result.exterior_from_jump_linf << ','
            << result.exterior_route_mismatch_linf << ','
            << result.flux_mean_removed << ',' << result.lambda << ','
            << result.trace_mean << ','
            << result.physical_relative_residual << ','
            << result.global_linf << ',' << result.global_l2 << ','
            << result.constant_shift << ',' << result.n_global << ','
            << result.native_global_linf << ','
            << result.native_global_l2 << ','
            << result.native_constant_shift << ','
            << result.n_native_global << ','
            << result.trace_linf << ',' << result.trace_l2 << ','
            << result.global_linf_order << ','
            << result.global_l2_order << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        std::cout << std::unitbuf << std::scientific << std::setprecision(6);
        std::vector<std::string> geometries = {"ellipse", "flower"};
        std::vector<int> levels = {48, 72, 108, 162};
        if (argc >= 2) {
            const std::string selection = argv[1];
            if (selection != "both")
                geometries = {selection};
        }
        if (argc >= 3) {
            levels.clear();
            for (int i = 2; i < argc; ++i)
                levels.push_back(std::stoi(argv[i]));
        }

        const double panel_length_over_h = environment_double(
            "KFBIM_HJET_PANEL_LENGTH_OVER_H", 1.6);
        const int max_iter = environment_int(
            "KFBIM_HJET_MAX_ITER", 100);
        const int restart = environment_int(
            "KFBIM_HJET_RESTART", 80);
        const double tolerance = environment_double(
            "KFBIM_HJET_TOL", 2.0e-10);

        std::cout << "KFBI2D P2 exterior-trace on harmonic-jet tutorial cases\n"
                  << "tol=" << tolerance << " max_iter=" << max_iter
                  << " restart=" << restart
                  << " panel_length/h=" << panel_length_over_h << '\n';

        std::vector<StudyResult> results;
        for (const std::string& geometry : geometries) {
            for (const int n : levels) {
                std::cout << "[run] geometry=" << geometry
                          << " N=" << n << '\n';
                StudyResult result = run_one(
                    geometry,
                    n,
                    panel_length_over_h,
                    max_iter,
                    tolerance,
                    restart);
                print_result(result);
                results.push_back(std::move(result));
            }
        }
        add_orders(results);
        print_summary(results);

#ifdef KFBIM_APP_OUTPUT_DIR
        const std::filesystem::path output_dir = KFBIM_APP_OUTPUT_DIR;
#else
        const std::filesystem::path output_dir = "output";
#endif
        const std::filesystem::path csv_path =
            output_dir / "harmonic_jet_case_cpp_p2.csv";
        write_csv(csv_path, results);
        std::cout << "CSV: " << csv_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
