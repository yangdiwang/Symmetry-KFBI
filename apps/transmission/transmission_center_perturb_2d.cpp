#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kfbim/kfbim.hpp"

using namespace kfbim;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTargetNodeSpacingOverH = 1.2;
constexpr double kTargetPanelLengthOverH = 2.0 * kTargetNodeSpacingOverH;

constexpr double kBaseCx = 0.07;
constexpr double kBaseCy = -0.04;
constexpr double kStarRadius = 0.75;
constexpr double kStarAmplitude = 0.25;
constexpr int    kStarFolds = 3;
constexpr double kBoxMargin = 0.30;

#ifndef KFBIM_APP_OUTPUT_DIR
#define KFBIM_APP_OUTPUT_DIR "output"
#endif

class Star3Curve2D final : public ICurve2D {
public:
    Star3Curve2D(double cx, double cy) : cx_(cx), cy_(cy) {}

    Eigen::Vector2d eval(double t) const override
    {
        const double r = radius(t);
        return {cx_ + r * std::cos(t), cy_ + r * std::sin(t)};
    }

    Eigen::Vector2d deriv(double t) const override
    {
        const double r = radius(t);
        const double drdt = -kStarRadius * kStarAmplitude * kStarFolds
                            * std::sin(kStarFolds * t);
        return {drdt * std::cos(t) - r * std::sin(t),
                drdt * std::sin(t) + r * std::cos(t)};
    }

    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }

private:
    static double radius(double t)
    {
        return kStarRadius * (1.0 + kStarAmplitude * std::cos(kStarFolds * t));
    }

    double cx_;
    double cy_;
};

struct Box2D {
    std::array<double, 2> lower;
    double                side_length;
};

struct TransmissionCaseConfig {
    std::string                       label;
    LaplaceTransmissionMode2D         mode;
    LaplaceTransmissionCoefficients2D coefficients;
    int                               max_iter;
};

struct Options {
    int                   samples = 50;
    unsigned int          seed = 20260509u;
    double                jitter_radius = 0.04;
    std::vector<int>      levels = {32, 64, 128};
    std::filesystem::path out_dir =
        std::filesystem::path(KFBIM_APP_OUTPUT_DIR)
        / "transmission_center_perturb_2d";
    std::string mode = "common";
};

struct SolveResult {
    double bulk_err = std::numeric_limits<double>::quiet_NaN();
    double wall_s = 0.0;
    int    iterations = 0;
    int    interface_points = 0;
    bool   converged = false;
};

struct Record {
    int    sample = 0;
    double cx = 0.0;
    double cy = 0.0;
    int    N = 0;
    double max_err = std::numeric_limits<double>::quiet_NaN();
    double order = std::numeric_limits<double>::quiet_NaN();
    double wall_s = 0.0;
    int    gmres = 0;
    int    interface_points = 0;
    bool   converged = false;
};

std::vector<int> parse_levels(const std::string& text)
{
    std::vector<int> levels;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty())
            continue;
        levels.push_back(std::stoi(item));
    }
    if (levels.empty())
        throw std::runtime_error("--levels must contain at least one grid size");
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    return levels;
}

Options parse_options(int argc, char** argv)
{
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };

        if (arg == "--samples") {
            opts.samples = std::stoi(require_value("--samples"));
        } else if (arg == "--seed") {
            opts.seed = static_cast<unsigned int>(
                std::stoul(require_value("--seed")));
        } else if (arg == "--jitter") {
            opts.jitter_radius = std::stod(require_value("--jitter"));
        } else if (arg == "--levels") {
            opts.levels = parse_levels(require_value("--levels"));
        } else if (arg == "--out") {
            opts.out_dir = require_value("--out");
        } else if (arg == "--mode") {
            opts.mode = require_value("--mode");
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: transmission_center_perturb_2d [options]\n"
                << "  --samples N       random center perturbations (default 50)\n"
                << "  --seed N          RNG seed (default 20260509)\n"
                << "  --jitter R        max center perturbation radius (default 0.04)\n"
                << "  --levels CSV      grid levels, e.g. 32,64,128 (default)\n"
                << "  --mode common|different  transmission test case (default common)\n"
                << "  --out DIR         output directory\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (opts.samples <= 0)
        throw std::runtime_error("--samples must be positive");
    if (opts.jitter_radius < 0.0)
        throw std::runtime_error("--jitter must be nonnegative");
    if (opts.mode != "common" && opts.mode != "different")
        throw std::runtime_error("--mode must be common or different");
    return opts;
}

TransmissionCaseConfig make_config(const std::string& mode)
{
    if (mode == "common") {
        constexpr double beta_int = 2.0;
        constexpr double beta_ext = 1.0;
        constexpr double lambda_sq = 1.1;
        return {"Common-ratio transmission",
                LaplaceTransmissionMode2D::CommonRatio,
                {beta_int, beta_ext, beta_int * lambda_sq, beta_ext * lambda_sq},
                250};
    }

    return {"Different-ratio transmission",
            LaplaceTransmissionMode2D::DifferentRatios,
            {10.0, 1.0, 11.0, 0.7},
            400};
}

Box2D make_outer_box(const ICurve2D& curve)
{
    constexpr int kSamples = 8192;
    double xmin = std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double ymin = std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < kSamples; ++i) {
        const double t = curve.t_min()
                         + (curve.t_max() - curve.t_min())
                         * static_cast<double>(i) / static_cast<double>(kSamples);
        const Eigen::Vector2d p = curve.eval(t);
        xmin = std::min(xmin, p[0]);
        xmax = std::max(xmax, p[0]);
        ymin = std::min(ymin, p[1]);
        ymax = std::max(ymax, p[1]);
    }

    const double cx = 0.5 * (xmin + xmax);
    const double cy = 0.5 * (ymin + ymax);
    const double span = std::max(xmax - xmin, ymax - ymin);
    const double side = span + 2.0 * kBoxMargin;
    return {{cx - 0.5 * side, cy - 0.5 * side}, side};
}

bool is_outer_boundary_node(int idx, int nx, int ny)
{
    const int i = idx % nx;
    const int j = idx / nx;
    return i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
}

double lambda_sq_int(const TransmissionCaseConfig& config)
{
    return config.coefficients.kappa_sq_int / config.coefficients.beta_int;
}

double lambda_sq_ext(const TransmissionCaseConfig& config)
{
    return config.coefficients.kappa_sq_ext / config.coefficients.beta_ext;
}

double u_int(double x, double y)
{
    return std::exp(0.35 * x + 0.15 * y)
           + 0.2 * std::cos(1.7 * x - 0.3 * y);
}

Eigen::Vector2d grad_u_int(double x, double y)
{
    const double e = std::exp(0.35 * x + 0.15 * y);
    const double p = 1.7 * x - 0.3 * y;
    return {0.35 * e - 0.34 * std::sin(p),
            0.15 * e + 0.06 * std::sin(p)};
}

double q_int(double x, double y, double lambda_sq)
{
    const double e = std::exp(0.35 * x + 0.15 * y);
    const double p = 1.7 * x - 0.3 * y;
    return (lambda_sq - 0.145) * e
           + 0.2 * (lambda_sq + 2.98) * std::cos(p);
}

double u_ext(double x, double y)
{
    return 0.6 * std::sin(0.8 * x + 0.4 * y)
           + 0.25 * x - 0.15 * y + 0.1 * x * y + 0.5;
}

Eigen::Vector2d grad_u_ext(double x, double y)
{
    const double c = std::cos(0.8 * x + 0.4 * y);
    return {0.48 * c + 0.25 + 0.1 * y,
            0.24 * c - 0.15 + 0.1 * x};
}

double q_ext(double x, double y, double lambda_sq)
{
    return 0.48 * std::sin(0.8 * x + 0.4 * y)
           + lambda_sq * u_ext(x, y);
}

SolveResult solve_center_case(int N,
                              double cx,
                              double cy,
                              const Box2D& fixed_box,
                              const TransmissionCaseConfig& config)
{
    const auto wall_start = std::chrono::steady_clock::now();

    const Star3Curve2D star(cx, cy);
    const double h = fixed_box.side_length / static_cast<double>(N);
    CartesianGrid2D grid(fixed_box.lower, {h, h}, {N, N}, DofLayout2D::Node);
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int n_dof = nx * ny;

    Interface2D iface =
        CurveResampler2D::discretize(star, h, kTargetPanelLengthOverH);
    const int n_iface = iface.num_points();

    const double lambda_int = lambda_sq_int(config);
    const double lambda_ext = lambda_sq_ext(config);

    Eigen::VectorXd mu(n_iface);
    Eigen::VectorXd sigma(n_iface);
    LaplaceTransmissionRhsData2D rhs_data;
    rhs_data.reduced_rhs_int_derivs.resize(n_iface);
    rhs_data.reduced_rhs_ext_derivs.resize(n_iface);

    const auto& points = iface.points();
    const auto& normals = iface.normals();
    for (int q = 0; q < n_iface; ++q) {
        const double x = points(q, 0);
        const double y = points(q, 1);
        const Eigen::Vector2d normal(normals(q, 0), normals(q, 1));
        mu[q] = u_int(x, y) - u_ext(x, y);
        sigma[q] = config.coefficients.beta_int * grad_u_int(x, y).dot(normal)
                   - config.coefficients.beta_ext * grad_u_ext(x, y).dot(normal);
        rhs_data.reduced_rhs_int_derivs[q] =
            Eigen::VectorXd::Constant(1, q_int(x, y, lambda_int));
        rhs_data.reduced_rhs_ext_derivs[q] =
            Eigen::VectorXd::Constant(1, q_ext(x, y, lambda_ext));
    }

    GridPair2D gp(grid, iface);
    rhs_data.reduced_rhs_bulk = Eigen::VectorXd::Zero(n_dof);
    Eigen::VectorXd u_exact = Eigen::VectorXd::Zero(n_dof);
    Eigen::VectorXd outer_bc = Eigen::VectorXd::Zero(n_dof);

    for (int n = 0; n < n_dof; ++n) {
        const auto c = grid.coord(n);
        const double x = c[0];
        const double y = c[1];

        if (is_outer_boundary_node(n, nx, ny)) {
            outer_bc[n] = u_ext(x, y);
            u_exact[n] = outer_bc[n];
            continue;
        }

        if (gp.domain_label(n) > 0) {
            rhs_data.reduced_rhs_bulk[n] = q_int(x, y, lambda_int);
            u_exact[n] = u_int(x, y);
        } else {
            rhs_data.reduced_rhs_bulk[n] = q_ext(x, y, lambda_ext);
            u_exact[n] = u_ext(x, y);
        }
    }

    LaplaceTransmissionOptions2D options;
    options.correction_method = LaplaceCorrectionMethod2D::NearestExpansionCenter;
    LaplaceTransmission2D problem(grid,
                                  iface,
                                  config.mode,
                                  config.coefficients,
                                  options);

    auto result = problem.solve(mu,
                                sigma,
                                rhs_data,
                                outer_bc,
                                config.max_iter,
                                1.0e-8,
                                100);

    double bulk_err = 0.0;
    for (int n = 0; n < n_dof; ++n)
        bulk_err = std::max(bulk_err, std::abs(result.u_bulk[n] - u_exact[n]));

    const double wall_s =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start).count();

    return {bulk_err, wall_s, result.iterations, n_iface, result.converged};
}

double quantile(std::vector<double> values, double q)
{
    if (values.empty())
        return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return (1.0 - frac) * values[lo] + frac * values[hi];
}

void write_metadata(const Options& opts,
                    const TransmissionCaseConfig& config,
                    const Box2D& fixed_box)
{
    std::ofstream out(opts.out_dir / "metadata.txt");
    out << std::setprecision(16);
    out << "mode=" << opts.mode << "\n";
    out << "case=" << config.label << "\n";
    out << "samples=" << opts.samples << "\n";
    out << "seed=" << opts.seed << "\n";
    out << "base_center=" << kBaseCx << "," << kBaseCy << "\n";
    out << "jitter_radius=" << opts.jitter_radius << "\n";
    out << "fixed_box_lower=" << fixed_box.lower[0] << "," << fixed_box.lower[1] << "\n";
    out << "fixed_box_side=" << fixed_box.side_length << "\n";
    out << "target_node_spacing_over_h=" << kTargetNodeSpacingOverH << "\n";
    out << "levels=";
    for (std::size_t i = 0; i < opts.levels.size(); ++i) {
        if (i)
            out << ",";
        out << opts.levels[i];
    }
    out << "\n";
}

void write_summary_csv(const std::filesystem::path& path,
                       const std::vector<Record>& records,
                       const std::vector<int>& levels)
{
    std::ofstream out(path);
    out << std::setprecision(16);
    out << "N,count,err_min,err_q25,err_median,err_q75,err_max,err_mean,err_std,order_median\n";

    for (std::size_t li = 0; li < levels.size(); ++li) {
        std::vector<double> errs;
        std::vector<double> orders;
        for (const Record& r : records) {
            if (r.N != levels[li] || !r.converged || !std::isfinite(r.max_err))
                continue;
            errs.push_back(r.max_err);
            if (std::isfinite(r.order))
                orders.push_back(r.order);
        }

        double mean = std::numeric_limits<double>::quiet_NaN();
        double stdev = std::numeric_limits<double>::quiet_NaN();
        if (!errs.empty()) {
            double sum = 0.0;
            for (double v : errs)
                sum += v;
            mean = sum / static_cast<double>(errs.size());

            double var = 0.0;
            for (double v : errs)
                var += (v - mean) * (v - mean);
            stdev = std::sqrt(var / static_cast<double>(errs.size()));
        }

        out << levels[li] << ","
            << errs.size() << ","
            << quantile(errs, 0.0) << ","
            << quantile(errs, 0.25) << ","
            << quantile(errs, 0.5) << ","
            << quantile(errs, 0.75) << ","
            << quantile(errs, 1.0) << ","
            << mean << ","
            << stdev << ",";
        if (li == 0 || orders.empty())
            out << "";
        else
            out << quantile(orders, 0.5);
        out << "\n";
    }
}

void write_samples_csv(const std::filesystem::path& path,
                       const std::vector<Record>& records)
{
    std::ofstream out(path);
    out << std::setprecision(16);
    out << "sample,cx,cy,N,max_err,order,wall_s,GMRES,interface_points,converged\n";
    for (const Record& r : records) {
        out << r.sample << ","
            << r.cx << ","
            << r.cy << ","
            << r.N << ","
            << r.max_err << ",";
        if (std::isfinite(r.order))
            out << r.order;
        out << ","
            << r.wall_s << ","
            << r.gmres << ","
            << r.interface_points << ","
            << (r.converged ? 1 : 0) << "\n";
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Options opts = parse_options(argc, argv);
        const TransmissionCaseConfig config = make_config(opts.mode);
        std::filesystem::create_directories(opts.out_dir);

        const Star3Curve2D base_star(kBaseCx, kBaseCy);
        const Box2D fixed_box = make_outer_box(base_star);

        write_metadata(opts, config, fixed_box);

        std::mt19937 rng(opts.seed);
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        std::uniform_real_distribution<double> angle(0.0, 2.0 * kPi);

        std::vector<Record> records;
        records.reserve(static_cast<std::size_t>(opts.samples)
                        * opts.levels.size());

        std::cout << "2D transmission center perturbation experiment\n"
                  << "  case=" << config.label << "\n"
                  << "  samples=" << opts.samples
                  << " seed=" << opts.seed
                  << " jitter=" << opts.jitter_radius << "\n"
                  << "  output=" << opts.out_dir << "\n";

        for (int s = 0; s < opts.samples; ++s) {
            const double rho = opts.jitter_radius * std::sqrt(unit(rng));
            const double theta = angle(rng);
            const double cx = kBaseCx + rho * std::cos(theta);
            const double cy = kBaseCy + rho * std::sin(theta);

            std::cout << "sample " << (s + 1) << "/" << opts.samples
                      << " center=(" << cx << "," << cy << ")\n";

            double previous_err = std::numeric_limits<double>::quiet_NaN();
            for (std::size_t li = 0; li < opts.levels.size(); ++li) {
                const int N = opts.levels[li];
                const SolveResult result =
                    solve_center_case(N, cx, cy, fixed_box, config);

                Record rec;
                rec.sample = s;
                rec.cx = cx;
                rec.cy = cy;
                rec.N = N;
                rec.max_err = result.bulk_err;
                rec.wall_s = result.wall_s;
                rec.gmres = result.iterations;
                rec.interface_points = result.interface_points;
                rec.converged = result.converged;
                if (li > 0
                    && std::isfinite(previous_err)
                    && previous_err > 0.0
                    && std::isfinite(result.bulk_err)
                    && result.bulk_err > 0.0) {
                    rec.order = std::log2(previous_err / result.bulk_err);
                }
                records.push_back(rec);
                previous_err = result.bulk_err;

                std::cout << "  N=" << N
                          << " err=" << std::scientific << result.bulk_err
                          << std::defaultfloat
                          << " order=";
                if (std::isfinite(rec.order))
                    std::cout << rec.order;
                else
                    std::cout << "-";
                std::cout << " wall_s=" << result.wall_s
                          << " GMRES=" << result.iterations
                          << " converged=" << (result.converged ? "yes" : "no")
                          << "\n";
            }
        }

        write_samples_csv(opts.out_dir / "samples.csv", records);
        write_summary_csv(opts.out_dir / "summary.csv", records, opts.levels);

        std::cout << "wrote " << (opts.out_dir / "samples.csv") << "\n";
        std::cout << "wrote " << (opts.out_dir / "summary.csv") << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
