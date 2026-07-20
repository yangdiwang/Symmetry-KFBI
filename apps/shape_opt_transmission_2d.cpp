#include "kfbim/kfbim.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
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

using namespace kfbim;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kCenterX = 0.07;
constexpr double kCenterY = -0.04;
constexpr double kBaseRadius = 0.50;
constexpr double kBoxSideLength = 2.40;
constexpr double kTargetNodeSpacingOverH = 1.20;
constexpr double kTargetPanelLengthOverH = 2.0 * kTargetNodeSpacingOverH;
constexpr double kBetaInt = 4.0;
constexpr double kBetaExt = 1.0;
constexpr double kLambdaSq = 1.1;
constexpr int kNumExcitations = 4;
constexpr int kBoundarySamples = 512;
constexpr int kShapeStatsSamples = 4096;
constexpr double kObservationInnerRadius = 0.82;
constexpr double kObservationOuterRadius = 1.08;
constexpr double kMinAllowedRadius = 0.25;
constexpr double kMaxAllowedRadius = 0.74;

#ifndef KFBIM_APP_OUTPUT_DIR
#define KFBIM_APP_OUTPUT_DIR "output"
#endif

enum class GradientMethod {
    Adjoint,
    FiniteDifference
};

struct Options {
    int modes = 3;
    int N = 64;
    int iterations = 12;
    int save_every = 1;
    double fd_eps = 1.0e-3;
    double alpha = 1.0e-5;
    double area_rho = 1.0e-2;
    std::filesystem::path out_dir =
        std::filesystem::path(KFBIM_APP_OUTPUT_DIR) / "shape_opt_transmission_2d";
    GradientMethod gradient_method = GradientMethod::Adjoint;
    bool gradient_check = false;
    bool show_help = false;
};

struct ShapeStats {
    double area = 0.0;
    double min_radius = 0.0;
    double max_radius = 0.0;
};

struct InterfaceFieldTrace {
    Eigen::VectorXd u;
    Eigen::VectorXd un_int;
    Eigen::VectorXd un_ext;
    Eigen::VectorXd utau;
};

struct ExcitationState {
    Eigen::VectorXd       u_bulk;
    Eigen::VectorXd       psi_density;
    InterfaceFieldTrace   trace;
};

struct ShapeSolve {
    Interface2D iface;
    std::vector<ExcitationState> excitations;
    int total_gmres_iterations = 0;
};

struct Evaluation {
    double objective = 0.0;
    double data_misfit = 0.0;
    double regularization = 0.0;
    double area_penalty = 0.0;
    double area = 0.0;
    int total_gmres_iterations = 0;
    bool valid = true;
    std::vector<double> measurements;
    std::shared_ptr<ShapeSolve> solve;
};

void print_usage(const char* argv0)
{
    std::cout
        << "usage: " << argv0 << " [options]\n"
        << "\n"
        << "Options:\n"
        << "  --N VALUE          Cartesian grid cells per side (default 64)\n"
        << "  --iters VALUE      Optimization iterations (default 12)\n"
        << "  --out PATH         Output directory (default output/shape_opt_transmission_2d)\n"
        << "  --fd-eps VALUE     Forward finite-difference step (default 1e-3)\n"
        << "  --alpha VALUE      Fourier regularization weight (default 1e-5)\n"
        << "  --area-rho VALUE   Area penalty weight (default 1e-2)\n"
        << "  --save-every VALUE Save boundary every VALUE iterations (default 1)\n"
        << "  --gradient METHOD  Gradient method: adjoint or fd (default adjoint)\n"
        << "  --gradient-check   Compare adjoint gradient to central finite differences\n"
        << "  --help             Show this message\n";
}

int parse_int_arg(const std::string& name, const std::string& value)
{
    char* end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0') {
        throw std::invalid_argument(name + " expects an integer value");
    }
    return static_cast<int>(parsed);
}

double parse_double_arg(const std::string& name, const std::string& value)
{
    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || *end != '\0') {
        throw std::invalid_argument(name + " expects a floating-point value");
    }
    return parsed;
}

GradientMethod parse_gradient_method(const std::string& value)
{
    if (value == "adjoint")
        return GradientMethod::Adjoint;
    if (value == "fd")
        return GradientMethod::FiniteDifference;
    throw std::invalid_argument("--gradient expects 'adjoint' or 'fd'");
}

const char* gradient_method_name(GradientMethod method)
{
    switch (method) {
    case GradientMethod::Adjoint:
        return "adjoint";
    case GradientMethod::FiniteDifference:
        return "fd";
    }
    return "unknown";
}

Options parse_options(int argc, char** argv)
{
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            opts.show_help = true;
            continue;
        }

        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(name + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--N") {
            opts.N = parse_int_arg(arg, require_value(arg));
        } else if (arg == "--iters") {
            opts.iterations = parse_int_arg(arg, require_value(arg));
        } else if (arg == "--out") {
            opts.out_dir = require_value(arg);
        } else if (arg == "--fd-eps") {
            opts.fd_eps = parse_double_arg(arg, require_value(arg));
        } else if (arg == "--alpha") {
            opts.alpha = parse_double_arg(arg, require_value(arg));
        } else if (arg == "--area-rho") {
            opts.area_rho = parse_double_arg(arg, require_value(arg));
        } else if (arg == "--save-every") {
            opts.save_every = parse_int_arg(arg, require_value(arg));
        } else if (arg == "--gradient") {
            opts.gradient_method = parse_gradient_method(require_value(arg));
        } else if (arg == "--gradient-check") {
            opts.gradient_check = true;
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }

    if (opts.N < 16)
        throw std::invalid_argument("--N must be at least 16");
    if (opts.iterations < 0)
        throw std::invalid_argument("--iters must be nonnegative");
    if (opts.save_every <= 0)
        throw std::invalid_argument("--save-every must be positive");
    if (opts.fd_eps <= 0.0)
        throw std::invalid_argument("--fd-eps must be positive");
    if (opts.alpha < 0.0)
        throw std::invalid_argument("--alpha must be nonnegative");
    if (opts.area_rho < 0.0)
        throw std::invalid_argument("--area-rho must be nonnegative");

    return opts;
}

class FourierRadialCurve final : public ICurve2D {
public:
    explicit FourierRadialCurve(Eigen::VectorXd params)
        : params_(std::move(params))
    {}

    Eigen::Vector2d eval(double t) const override
    {
        const double r = radius(t);
        return {kCenterX + r * std::cos(t), kCenterY + r * std::sin(t)};
    }

    Eigen::Vector2d deriv(double t) const override
    {
        const double r = radius(t);
        const double drdt = r * series_deriv(t);
        return {drdt * std::cos(t) - r * std::sin(t),
                drdt * std::sin(t) + r * std::cos(t)};
    }

    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }

    double radius(double t) const
    {
        return kBaseRadius * std::exp(series(t));
    }

private:
    double series(double t) const
    {
        double value = 0.0;
        const int modes = static_cast<int>(params_.size()) / 2;
        for (int k = 1; k <= modes; ++k) {
            const double a = params_[2 * (k - 1)];
            const double b = params_[2 * (k - 1) + 1];
            value += a * std::cos(k * t) + b * std::sin(k * t);
        }
        return value;
    }

    double series_deriv(double t) const
    {
        double value = 0.0;
        const int modes = static_cast<int>(params_.size()) / 2;
        for (int k = 1; k <= modes; ++k) {
            const double a = params_[2 * (k - 1)];
            const double b = params_[2 * (k - 1) + 1];
            value += -static_cast<double>(k) * a * std::sin(k * t)
                     + static_cast<double>(k) * b * std::cos(k * t);
        }
        return value;
    }

    Eigen::VectorXd params_;
};

Eigen::VectorXd initial_params(const Options& opts)
{
    return Eigen::VectorXd::Zero(2 * opts.modes);
}

Eigen::VectorXd target_params(const Options& opts)
{
    Eigen::VectorXd params = Eigen::VectorXd::Zero(2 * opts.modes);
    params[0] = 0.04;
    params[1] = 0.00;
    params[2] = 0.00;
    params[3] = -0.08;
    params[4] = 0.16;
    params[5] = 0.00;
    return params;
}

ShapeStats compute_shape_stats(const Eigen::VectorXd& params)
{
    FourierRadialCurve curve(params);
    ShapeStats stats;
    stats.min_radius = std::numeric_limits<double>::infinity();
    stats.max_radius = 0.0;

    const double dtheta = 2.0 * kPi / static_cast<double>(kShapeStatsSamples);
    double area_sum = 0.0;
    for (int i = 0; i < kShapeStatsSamples; ++i) {
        const double theta = dtheta * static_cast<double>(i);
        const double r = curve.radius(theta);
        stats.min_radius = std::min(stats.min_radius, r);
        stats.max_radius = std::max(stats.max_radius, r);
        area_sum += r * r;
    }
    stats.area = 0.5 * dtheta * area_sum;
    return stats;
}

double regularization_value(const Eigen::VectorXd& params)
{
    double value = 0.0;
    const int modes = static_cast<int>(params.size()) / 2;
    for (int k = 1; k <= modes; ++k) {
        const double a = params[2 * (k - 1)];
        const double b = params[2 * (k - 1) + 1];
        const double k2 = static_cast<double>(k * k);
        value += k2 * k2 * (a * a + b * b);
    }
    return value;
}

Eigen::VectorXd regularization_gradient(const Eigen::VectorXd& params,
                                        double alpha)
{
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(params.size());
    const int modes = static_cast<int>(params.size()) / 2;
    for (int k = 1; k <= modes; ++k) {
        const double k2 = static_cast<double>(k * k);
        const double weight = 2.0 * alpha * k2 * k2;
        grad[2 * (k - 1)] = weight * params[2 * (k - 1)];
        grad[2 * (k - 1) + 1] = weight * params[2 * (k - 1) + 1];
    }
    return grad;
}

Eigen::VectorXd area_gradient(const Eigen::VectorXd& params)
{
    FourierRadialCurve curve(params);
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(params.size());
    const int modes = static_cast<int>(params.size()) / 2;
    const double dtheta = 2.0 * kPi / static_cast<double>(kShapeStatsSamples);

    for (int i = 0; i < kShapeStatsSamples; ++i) {
        const double theta = dtheta * static_cast<double>(i);
        const double r = curve.radius(theta);
        const double r_sq_dtheta = r * r * dtheta;
        for (int k = 1; k <= modes; ++k) {
            grad[2 * (k - 1)] += r_sq_dtheta * std::cos(k * theta);
            grad[2 * (k - 1) + 1] += r_sq_dtheta * std::sin(k * theta);
        }
    }

    return grad;
}

bool shape_is_allowed(const ShapeStats& stats)
{
    return std::isfinite(stats.area)
           && stats.min_radius >= kMinAllowedRadius
           && stats.max_radius <= kMaxAllowedRadius;
}

CartesianGrid2D make_grid(const Options& opts)
{
    const double h = kBoxSideLength / static_cast<double>(opts.N);
    return CartesianGrid2D({-0.5 * kBoxSideLength, -0.5 * kBoxSideLength},
                           {h, h},
                           {opts.N, opts.N},
                           DofLayout2D::Node);
}

bool is_outer_boundary_node(int idx, int nx, int ny)
{
    const int i = idx % nx;
    const int j = idx / nx;
    return i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
}

double excitation_value(int excitation, double x, double y)
{
    switch (excitation) {
    case 0:
        return 1.0;
    case 1:
        return x;
    case 2:
        return y;
    case 3:
        return x * x - y * y;
    default:
        throw std::invalid_argument("unsupported excitation index");
    }
}

Eigen::VectorXd outer_dirichlet_values(const CartesianGrid2D& grid, int excitation)
{
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    Eigen::VectorXd values = Eigen::VectorXd::Zero(grid.num_dofs());

    for (int n = 0; n < grid.num_dofs(); ++n) {
        if (!is_outer_boundary_node(n, nx, ny))
            continue;
        const auto c = grid.coord(n);
        values[n] = excitation_value(excitation, c[0], c[1]);
    }
    return values;
}

std::vector<int> make_observation_nodes(const CartesianGrid2D& grid)
{
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    std::vector<int> nodes;

    for (int j = 2; j < ny - 2; ++j) {
        for (int i = 2; i < nx - 2; ++i) {
            const int n = grid.index(i, j);
            const auto c = grid.coord(i, j);
            const double dx = c[0] - kCenterX;
            const double dy = c[1] - kCenterY;
            const double r = std::sqrt(dx * dx + dy * dy);
            if (r >= kObservationInnerRadius && r <= kObservationOuterRadius)
                nodes.push_back(n);
        }
    }

    if (nodes.empty())
        throw std::runtime_error("no observation nodes selected; increase --N");
    return nodes;
}

LaplaceTransmissionRhsData2D zero_rhs_data(int n_dof, int n_iface)
{
    LaplaceTransmissionRhsData2D rhs;
    rhs.reduced_rhs_bulk = Eigen::VectorXd::Zero(n_dof);
    rhs.reduced_rhs_int_derivs =
        std::vector<Eigen::VectorXd>(n_iface, Eigen::VectorXd::Zero(1));
    rhs.reduced_rhs_ext_derivs =
        std::vector<Eigen::VectorXd>(n_iface, Eigen::VectorXd::Zero(1));
    return rhs;
}

std::vector<LaplaceJumpData2D> jumps_from_psi(const Eigen::VectorXd& psi)
{
    std::vector<LaplaceJumpData2D> jumps(psi.size());
    for (int q = 0; q < psi.size(); ++q) {
        jumps[q].u_jump = 0.0;
        jumps[q].un_jump = psi[q];
        jumps[q].rhs_derivs = Eigen::VectorXd::Zero(1);
    }
    return jumps;
}

InterfaceFieldTrace reconstruct_interface_trace(
    const GridPair2D&    grid_pair,
    const Eigen::VectorXd& u_bulk,
    const Eigen::VectorXd& psi_density)
{
    const Interface2D& iface = grid_pair.interface();
    const int n_iface = iface.num_points();
    if (psi_density.size() != n_iface) {
        throw std::invalid_argument(
            "reconstruct_interface_trace: psi size does not match interface");
    }

    LaplaceQuadraticPanelCenterSpread2D spread(
        grid_pair,
        kLambdaSq,
        LaplaceCorrectionMethod2D::NearestExpansionCenter,
        2);
    LaplaceQuadraticPanelCenterRestrict2D restrict_op(grid_pair, 2);

    Eigen::VectorXd rhs_correction =
        Eigen::VectorXd::Zero(grid_pair.grid().num_dofs());
    auto spread_result = spread.apply(jumps_from_psi(psi_density), rhs_correction);
    auto polys = restrict_op.apply(u_bulk, spread_result);

    InterfaceFieldTrace trace;
    trace.u = Eigen::VectorXd::Zero(n_iface);
    trace.un_int = Eigen::VectorXd::Zero(n_iface);
    trace.un_ext = Eigen::VectorXd::Zero(n_iface);
    trace.utau = Eigen::VectorXd::Zero(n_iface);

    const auto& normals = iface.normals();
    for (int q = 0; q < n_iface; ++q) {
        const double nx = normals(q, 0);
        const double ny = normals(q, 1);
        const Eigen::Vector2d grad_avg(polys[q].coeffs[1], polys[q].coeffs[2]);
        const Eigen::Vector2d normal(nx, ny);
        const Eigen::Vector2d tangent(-ny, nx);
        const double un_avg = grad_avg.dot(normal);

        trace.u[q] = polys[q].coeffs[0];
        trace.un_int[q] = un_avg + 0.5 * psi_density[q];
        trace.un_ext[q] = un_avg - 0.5 * psi_density[q];
        trace.utau[q] = grad_avg.dot(tangent);
    }

    return trace;
}

LaplaceTransmissionCoefficients2D transmission_coefficients()
{
    LaplaceTransmissionCoefficients2D coefficients;
    coefficients.beta_int = kBetaInt;
    coefficients.beta_ext = kBetaExt;
    coefficients.kappa_sq_int = kBetaInt * kLambdaSq;
    coefficients.kappa_sq_ext = kBetaExt * kLambdaSq;
    return coefficients;
}

LaplaceTransmissionOptions2D transmission_options()
{
    LaplaceTransmissionOptions2D options;
    options.correction_method = LaplaceCorrectionMethod2D::NearestExpansionCenter;
    return options;
}

ShapeSolve solve_shape(const Eigen::VectorXd& params,
                       const CartesianGrid2D& grid,
                       const Options& opts)
{
    const double h = kBoxSideLength / static_cast<double>(opts.N);
    FourierRadialCurve curve(params);
    Interface2D iface =
        CurveResampler2D::discretize(curve, h, kTargetPanelLengthOverH);

    const int n_iface = iface.num_points();
    const int n_dof = grid.num_dofs();
    const Eigen::VectorXd u_jump = Eigen::VectorXd::Zero(n_iface);
    const Eigen::VectorXd beta_flux_jump = Eigen::VectorXd::Zero(n_iface);
    const LaplaceTransmissionRhsData2D rhs = zero_rhs_data(n_dof, n_iface);

    LaplaceTransmission2D problem(grid,
                                  iface,
                                  LaplaceTransmissionMode2D::CommonRatio,
                                  transmission_coefficients(),
                                  transmission_options());

    ShapeSolve solve{iface, {}, 0};
    solve.excitations.reserve(kNumExcitations);

    for (int excitation = 0; excitation < kNumExcitations; ++excitation) {
        const Eigen::VectorXd outer_bc =
            outer_dirichlet_values(grid, excitation);
        auto result = problem.solve(u_jump,
                                    beta_flux_jump,
                                    rhs,
                                    outer_bc,
                                    200,
                                    1.0e-8,
                                    80);
        if (!result.converged) {
            throw std::runtime_error(
                "LaplaceTransmission2D did not converge for excitation "
                + std::to_string(excitation));
        }
        solve.total_gmres_iterations += result.iterations;
        ExcitationState state;
        state.u_bulk = std::move(result.u_bulk);
        state.psi_density = std::move(result.psi_density);
        state.trace = reconstruct_interface_trace(problem.grid_pair(),
                                                  state.u_bulk,
                                                  state.psi_density);
        solve.excitations.push_back(std::move(state));
    }

    return solve;
}

std::vector<double> sample_measurements(const ShapeSolve& solve,
                                        const std::vector<int>& obs_nodes)
{
    std::vector<double> measurements;
    measurements.reserve(solve.excitations.size() * obs_nodes.size());
    for (const auto& excitation : solve.excitations) {
        for (int node : obs_nodes)
            measurements.push_back(excitation.u_bulk[node]);
    }
    return measurements;
}

Evaluation invalid_evaluation(const Eigen::VectorXd& params,
                              double target_area,
                              const Options& opts)
{
    const ShapeStats stats = compute_shape_stats(params);
    const double reg = regularization_value(params);
    const double area_diff = stats.area - target_area;

    Evaluation eval;
    eval.objective = 1.0e30;
    eval.data_misfit = 1.0e30;
    eval.regularization = opts.alpha * reg;
    eval.area_penalty = opts.area_rho * area_diff * area_diff;
    eval.area = stats.area;
    eval.valid = false;
    return eval;
}

Evaluation evaluate_shape(const Eigen::VectorXd& params,
                          const CartesianGrid2D& grid,
                          const std::vector<int>& obs_nodes,
                          const std::vector<double>& target_measurements,
                          double target_area,
                          const Options& opts,
                          bool allow_invalid)
{
    const ShapeStats stats = compute_shape_stats(params);
    if (!shape_is_allowed(stats)) {
        if (allow_invalid)
            return invalid_evaluation(params, target_area, opts);
        throw std::runtime_error("shape radius left the allowed range");
    }

    try {
        ShapeSolve solve = solve_shape(params, grid, opts);
        std::vector<double> measurements = sample_measurements(solve, obs_nodes);
        if (measurements.size() != target_measurements.size()) {
            throw std::runtime_error("measurement vector size mismatch");
        }

        double sq = 0.0;
        for (std::size_t i = 0; i < measurements.size(); ++i) {
            const double diff = measurements[i] - target_measurements[i];
            sq += diff * diff;
        }

        const double data_misfit =
            0.5 * sq / static_cast<double>(measurements.size());
        const double reg_term = opts.alpha * regularization_value(params);
        const double area_diff = stats.area - target_area;
        const double area_term = opts.area_rho * area_diff * area_diff;

        Evaluation eval;
        eval.objective = data_misfit + reg_term + area_term;
        eval.data_misfit = data_misfit;
        eval.regularization = reg_term;
        eval.area_penalty = area_term;
        eval.area = stats.area;
        eval.total_gmres_iterations = solve.total_gmres_iterations;
        eval.valid = true;
        eval.measurements = std::move(measurements);
        eval.solve = std::make_shared<ShapeSolve>(std::move(solve));
        return eval;
    } catch (const std::exception&) {
        if (allow_invalid)
            return invalid_evaluation(params, target_area, opts);
        throw;
    }
}

Eigen::VectorXd finite_difference_gradient(
    const Eigen::VectorXd& params,
    const Evaluation& current,
    const CartesianGrid2D& grid,
    const std::vector<int>& obs_nodes,
    const std::vector<double>& target_measurements,
    double target_area,
    const Options& opts)
{
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(params.size());
    for (int p = 0; p < params.size(); ++p) {
        Eigen::VectorXd plus = params;
        plus[p] += opts.fd_eps;
        const Evaluation shifted = evaluate_shape(plus,
                                                  grid,
                                                  obs_nodes,
                                                  target_measurements,
                                                  target_area,
                                                  opts,
                                                  true);
        grad[p] = (shifted.objective - current.objective) / opts.fd_eps;
        if (!std::isfinite(grad[p]))
            grad[p] = 0.0;
    }
    return grad;
}

Eigen::VectorXd central_difference_gradient(
    const Eigen::VectorXd& params,
    const CartesianGrid2D& grid,
    const std::vector<int>& obs_nodes,
    const std::vector<double>& target_measurements,
    double target_area,
    const Options& opts)
{
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(params.size());
    const double eps = std::min(opts.fd_eps, 1.0e-4);
    for (int p = 0; p < params.size(); ++p) {
        Eigen::VectorXd plus = params;
        Eigen::VectorXd minus = params;
        plus[p] += eps;
        minus[p] -= eps;
        const Evaluation eval_plus = evaluate_shape(plus,
                                                    grid,
                                                    obs_nodes,
                                                    target_measurements,
                                                    target_area,
                                                    opts,
                                                    true);
        const Evaluation eval_minus = evaluate_shape(minus,
                                                     grid,
                                                     obs_nodes,
                                                     target_measurements,
                                                     target_area,
                                                     opts,
                                                     true);
        grad[p] = (eval_plus.objective - eval_minus.objective)
                  / (2.0 * eps);
        if (!std::isfinite(grad[p]))
            grad[p] = 0.0;
    }
    return grad;
}

LaplaceTransmissionRhsData2D adjoint_rhs_data(
    const GridPair2D& grid_pair,
    const std::vector<int>& obs_nodes,
    const std::vector<double>& target_measurements,
    const ExcitationState& forward,
    int excitation)
{
    const CartesianGrid2D& grid = grid_pair.grid();
    const int n_dof = grid.num_dofs();
    const int n_iface = grid_pair.interface().num_points();
    LaplaceTransmissionRhsData2D rhs = zero_rhs_data(n_dof, n_iface);

    const double h = grid.spacing()[0];
    const double total_measurements =
        static_cast<double>(target_measurements.size());
    const std::size_t offset =
        static_cast<std::size_t>(excitation) * obs_nodes.size();

    for (std::size_t m = 0; m < obs_nodes.size(); ++m) {
        const int node = obs_nodes[m];
        if (grid_pair.domain_label(node) != 0) {
            throw std::runtime_error(
                "adjoint_rhs_data: observation node is not exterior");
        }
        const double residual =
            forward.u_bulk[node] - target_measurements[offset + m];
        rhs.reduced_rhs_bulk[node] +=
            residual / (total_measurements * kBetaExt * h * h);
    }

    return rhs;
}

InterfaceFieldTrace solve_adjoint_trace(
    const LaplaceTransmission2D& problem,
    const LaplaceTransmissionRhsData2D& rhs,
    int& total_gmres_iterations)
{
    const int n_iface = problem.grid_pair().interface().num_points();
    const int n_dof = problem.grid_pair().grid().num_dofs();
    const Eigen::VectorXd zeros_iface = Eigen::VectorXd::Zero(n_iface);
    const Eigen::VectorXd zero_outer = Eigen::VectorXd::Zero(n_dof);

    auto result = problem.solve(zeros_iface,
                                zeros_iface,
                                rhs,
                                zero_outer,
                                200,
                                1.0e-8,
                                80);
    if (!result.converged)
        throw std::runtime_error("adjoint KFBIM solve did not converge");

    total_gmres_iterations += result.iterations;
    return reconstruct_interface_trace(problem.grid_pair(),
                                       result.u_bulk,
                                       result.psi_density);
}

double interface_theta(const Interface2D& iface, int q)
{
    double theta = std::atan2(iface.points()(q, 1) - kCenterY,
                              iface.points()(q, 0) - kCenterX);
    if (theta < 0.0)
        theta += 2.0 * kPi;
    return theta;
}

Eigen::VectorXd adjoint_gradient(
    const Eigen::VectorXd& params,
    const Evaluation& current,
    const CartesianGrid2D& grid,
    const std::vector<int>& obs_nodes,
    const std::vector<double>& target_measurements,
    double target_area,
    const Options& opts,
    int* adjoint_gmres_iterations = nullptr)
{
    if (!current.solve) {
        throw std::runtime_error(
            "adjoint_gradient requires current forward solve data");
    }

    const ShapeStats stats = compute_shape_stats(params);
    const ShapeSolve& forward = *current.solve;
    const Interface2D& iface = forward.iface;
    const int n_iface = iface.num_points();
    const int modes = static_cast<int>(params.size()) / 2;

    LaplaceTransmission2D adjoint_problem(grid,
                                          forward.iface,
                                          LaplaceTransmissionMode2D::CommonRatio,
                                          transmission_coefficients(),
                                          transmission_options());

    Eigen::VectorXd grad = Eigen::VectorXd::Zero(params.size());
    int total_adjoint_iters = 0;

    for (int excitation = 0; excitation < kNumExcitations; ++excitation) {
        const LaplaceTransmissionRhsData2D rhs =
            adjoint_rhs_data(adjoint_problem.grid_pair(),
                             obs_nodes,
                             target_measurements,
                             forward.excitations[excitation],
                             excitation);
        const InterfaceFieldTrace adjoint =
            solve_adjoint_trace(adjoint_problem, rhs, total_adjoint_iters);

        const InterfaceFieldTrace& utrace =
            forward.excitations[excitation].trace;
        for (int q = 0; q < n_iface; ++q) {
            const double normal_jump =
                kBetaInt * utrace.un_int[q] * adjoint.un_int[q]
                - kBetaExt * utrace.un_ext[q] * adjoint.un_ext[q];
            const double tangential_jump =
                (kBetaInt - kBetaExt)
                * utrace.utau[q] * adjoint.utau[q];
            const double reaction_jump =
                (kBetaInt - kBetaExt)
                * kLambdaSq * utrace.u[q] * adjoint.u[q];
            const double shape_density =
                normal_jump - tangential_jump - reaction_jump;

            const double theta = interface_theta(iface, q);
            const double r = std::hypot(iface.points()(q, 0) - kCenterX,
                                        iface.points()(q, 1) - kCenterY);
            const Eigen::Vector2d er(std::cos(theta), std::sin(theta));
            const Eigen::Vector2d normal(iface.normals()(q, 0),
                                         iface.normals()(q, 1));
            const double radial_normal = er.dot(normal);
            const double weight = iface.weights()[q];

            for (int k = 1; k <= modes; ++k) {
                const double common =
                    shape_density * r * radial_normal * weight;
                grad[2 * (k - 1)] += common * std::cos(k * theta);
                grad[2 * (k - 1) + 1] += common * std::sin(k * theta);
            }
        }
    }

    grad += regularization_gradient(params, opts.alpha);
    grad += 2.0 * opts.area_rho * (stats.area - target_area)
            * area_gradient(params);

    if (adjoint_gmres_iterations)
        *adjoint_gmres_iterations = total_adjoint_iters;

    return grad;
}

bool run_gradient_check(const Eigen::VectorXd& params,
                        const Evaluation& current,
                        const CartesianGrid2D& grid,
                        const std::vector<int>& obs_nodes,
                        const std::vector<double>& target_measurements,
                        double target_area,
                        const Options& opts)
{
    int adjoint_iters = 0;
    const Eigen::VectorXd grad_adj =
        adjoint_gradient(params,
                         current,
                         grid,
                         obs_nodes,
                         target_measurements,
                         target_area,
                         opts,
                         &adjoint_iters);
    const Eigen::VectorXd grad_fd =
        central_difference_gradient(params,
                                    grid,
                                    obs_nodes,
                                    target_measurements,
                                    target_area,
                                    opts);

    const double adj_norm = grad_adj.norm();
    const double fd_norm = grad_fd.norm();
    const double denom = std::max(1.0e-30, adj_norm * fd_norm);
    const double cosine = grad_adj.dot(grad_fd) / denom;

    std::printf("\n  Gradient check\n");
    std::printf("  central_fd_eps=%.3e  adjoint_gmres=%d  |adjoint|=%.6e  |fd|=%.6e  cosine=%.6f\n",
                std::min(opts.fd_eps, 1.0e-4),
                adjoint_iters,
                adj_norm,
                fd_norm,
                cosine);
    std::printf("  %6s  %14s  %14s\n", "param", "adjoint", "central_fd");
    for (int p = 0; p < params.size(); ++p) {
        std::printf("  %6d  %14.6e  %14.6e\n",
                    p,
                    grad_adj[p],
                    grad_fd[p]);
    }

    return std::isfinite(cosine) && cosine >= 0.90;
}

std::filesystem::path boundary_path(const std::filesystem::path& out_dir, int iter)
{
    char name[64];
    std::snprintf(name, sizeof(name), "boundary_iter_%04d.csv", iter);
    return out_dir / name;
}

void write_boundary_csv(const std::filesystem::path& path,
                        const Eigen::VectorXd& params)
{
    FourierRadialCurve curve(params);
    std::ofstream out(path);
    out << std::setprecision(16);
    out << "theta,x,y,r\n";
    for (int i = 0; i <= kBoundarySamples; ++i) {
        const double theta =
            2.0 * kPi * static_cast<double>(i) / static_cast<double>(kBoundarySamples);
        const double r = curve.radius(theta);
        const Eigen::Vector2d p = curve.eval(theta);
        out << theta << "," << p[0] << "," << p[1] << "," << r << "\n";
    }
}

void write_summary_header(std::ofstream& out, int modes)
{
    out << "iter,objective,data_misfit,regularization,area_penalty,area,"
           "grad_norm,step,accepted,total_gmres";
    for (int k = 1; k <= modes; ++k)
        out << ",a" << k << ",b" << k;
    out << "\n";
}

void write_summary_row(std::ofstream& out,
                       int iter,
                       const Evaluation& eval,
                       const Eigen::VectorXd& params,
                       double grad_norm,
                       double step,
                       bool accepted)
{
    out << std::setprecision(16)
        << iter << ","
        << eval.objective << ","
        << eval.data_misfit << ","
        << eval.regularization << ","
        << eval.area_penalty << ","
        << eval.area << ","
        << grad_norm << ","
        << step << ","
        << (accepted ? 1 : 0) << ","
        << eval.total_gmres_iterations;
    for (int p = 0; p < params.size(); ++p)
        out << "," << params[p];
    out << "\n";
}

void write_observations_csv(const std::filesystem::path& path,
                            const CartesianGrid2D& grid,
                            const std::vector<int>& obs_nodes,
                            const std::vector<double>& target_measurements,
                            const std::vector<double>& initial_measurements,
                            const std::vector<double>& final_measurements)
{
    std::ofstream out(path);
    out << std::setprecision(16);
    out << "obs,node,x,y,excitation,target,initial,final\n";

    int row = 0;
    for (int excitation = 0; excitation < kNumExcitations; ++excitation) {
        for (std::size_t q = 0; q < obs_nodes.size(); ++q) {
            const int node = obs_nodes[q];
            const auto c = grid.coord(node);
            out << row << ","
                << node << ","
                << c[0] << ","
                << c[1] << ","
                << excitation << ","
                << target_measurements[row] << ","
                << initial_measurements[row] << ","
                << final_measurements[row] << "\n";
            ++row;
        }
    }
}

void print_iteration(int iter,
                     const Evaluation& eval,
                     double grad_norm,
                     double step,
                     bool accepted)
{
    std::printf("  iter %02d  J=%.6e  data=%.6e  area=%.6e  |grad|=%.3e  step=%.3e  %s\n",
                iter,
                eval.objective,
                eval.data_misfit,
                eval.area,
                grad_norm,
                step,
                accepted ? "accepted" : "rejected");
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Options opts = parse_options(argc, argv);
        if (opts.show_help) {
            print_usage(argv[0]);
            return 0;
        }

        std::filesystem::create_directories(opts.out_dir);

        const CartesianGrid2D grid = make_grid(opts);
        const std::vector<int> obs_nodes = make_observation_nodes(grid);
        const Eigen::VectorXd target = target_params(opts);
        Eigen::VectorXd params = initial_params(opts);
        const double target_area = compute_shape_stats(target).area;

        std::printf("2D transmission shape optimization\n");
        std::printf("  N=%d  iterations=%d  observations=%zu  gradient=%s  output=%s\n",
                    opts.N,
                    opts.iterations,
                    obs_nodes.size(),
                    gradient_method_name(opts.gradient_method),
                    opts.out_dir.string().c_str());
        std::printf("  beta_int=%.3f beta_ext=%.3f lambda^2=%.3f\n",
                    kBetaInt,
                    kBetaExt,
                    kLambdaSq);

        std::printf("  generating target data...\n");
        const ShapeSolve target_solve = solve_shape(target, grid, opts);
        const std::vector<double> target_measurements =
            sample_measurements(target_solve, obs_nodes);
        write_boundary_csv(opts.out_dir / "target_boundary.csv", target);

        Evaluation current = evaluate_shape(params,
                                            grid,
                                            obs_nodes,
                                            target_measurements,
                                            target_area,
                                            opts,
                                            false);
        const std::vector<double> initial_measurements = current.measurements;

        std::ofstream summary(opts.out_dir / "summary.csv");
        write_summary_header(summary, opts.modes);
        write_summary_row(summary, 0, current, params, 0.0, 0.0, true);
        write_boundary_csv(boundary_path(opts.out_dir, 0), params);
        print_iteration(0, current, 0.0, 0.0, true);

        if (opts.gradient_check) {
            const bool gradient_ok = run_gradient_check(params,
                                                        current,
                                                        grid,
                                                        obs_nodes,
                                                        target_measurements,
                                                        target_area,
                                                        opts);
            if (!gradient_ok) {
                throw std::runtime_error(
                    "gradient check failed: cosine similarity below 0.90");
            }
        }

        int last_written_boundary_iter = 0;
        int last_iter = 0;

        for (int iter = 1; iter <= opts.iterations; ++iter) {
            int adjoint_gmres_iterations = 0;
            const Eigen::VectorXd grad =
                (opts.gradient_method == GradientMethod::Adjoint)
                    ? adjoint_gradient(params,
                                       current,
                                       grid,
                                       obs_nodes,
                                       target_measurements,
                                       target_area,
                                       opts,
                                       &adjoint_gmres_iterations)
                    : finite_difference_gradient(params,
                                                 current,
                                                 grid,
                                                 obs_nodes,
                                                 target_measurements,
                                                 target_area,
                                                 opts);
            const double grad_norm = grad.norm();
            if (!std::isfinite(grad_norm) || grad_norm <= 1.0e-14) {
                std::printf("  stopping: gradient is too small or non-finite\n");
                break;
            }
            if (opts.gradient_method == GradientMethod::Adjoint) {
                std::printf("  adjoint GMRES iterations this gradient: %d\n",
                            adjoint_gmres_iterations);
            }

            const Eigen::VectorXd direction = -grad / grad_norm;
            double step = 8.0e-2;
            bool accepted = false;
            Evaluation accepted_eval;
            Eigen::VectorXd accepted_params = params;

            for (int ls = 0; ls < 12; ++ls) {
                Eigen::VectorXd trial = params + step * direction;
                Evaluation trial_eval = evaluate_shape(trial,
                                                       grid,
                                                       obs_nodes,
                                                       target_measurements,
                                                       target_area,
                                                       opts,
                                                       true);
                if (trial_eval.valid && trial_eval.objective < current.objective) {
                    accepted = true;
                    accepted_eval = std::move(trial_eval);
                    accepted_params = std::move(trial);
                    break;
                }
                step *= 0.5;
            }

            if (accepted) {
                params = std::move(accepted_params);
                current = std::move(accepted_eval);
            } else {
                step = 0.0;
            }

            write_summary_row(summary, iter, current, params, grad_norm, step, accepted);
            print_iteration(iter, current, grad_norm, step, accepted);

            if (iter % opts.save_every == 0 || iter == opts.iterations || !accepted) {
                write_boundary_csv(boundary_path(opts.out_dir, iter), params);
                last_written_boundary_iter = iter;
            }
            last_iter = iter;

            if (!accepted)
                break;
        }

        if (last_written_boundary_iter != last_iter)
            write_boundary_csv(boundary_path(opts.out_dir, last_iter), params);

        write_observations_csv(opts.out_dir / "observations.csv",
                               grid,
                               obs_nodes,
                               target_measurements,
                               initial_measurements,
                               current.measurements);

        std::printf("  final objective %.6e\n", current.objective);
        std::printf("  wrote %s\n", opts.out_dir.string().c_str());
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "shape_opt_transmission_2d: " << ex.what() << "\n";
        return 1;
    }
}
