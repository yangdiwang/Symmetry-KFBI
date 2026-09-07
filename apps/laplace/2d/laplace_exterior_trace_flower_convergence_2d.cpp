#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kfbim/kfbim.hpp"

namespace {

using namespace kfbim;

constexpr double kPi = 3.1415926535897932384626433832795;
constexpr double kBoxMin = -1.25;
constexpr double kBoxLength = 2.50;
constexpr double kPanelLengthOverH = 2.20;
constexpr double kQuadraticCoefficient = 0.12;
constexpr double kVolumeForcing = -4.0 * kQuadraticCoefficient;

struct FlowerSpec {
    std::string name;
    int petals = 3;
    double radius = 0.72;
    double amplitude = 0.16;
    double rotation = 0.0;
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
};

class FlowerCurve2D final : public ICurve2D {
public:
    explicit FlowerCurve2D(FlowerSpec spec)
        : spec_(std::move(spec))
    {}

    Eigen::Vector2d eval(double t) const override
    {
        const double r = radial(t);
        const Eigen::Vector2d local(r * std::cos(t), r * std::sin(t));
        return spec_.center + rotate(local);
    }

    Eigen::Vector2d deriv(double t) const override
    {
        const double r = radial(t);
        const double dr = -spec_.radius * spec_.amplitude
                        * static_cast<double>(spec_.petals)
                        * std::sin(static_cast<double>(spec_.petals) * t);
        const Eigen::Vector2d local(
            dr * std::cos(t) - r * std::sin(t),
            dr * std::sin(t) + r * std::cos(t));
        return rotate(local);
    }

    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }

private:
    double radial(double t) const
    {
        return spec_.radius
             * (1.0 + spec_.amplitude
                          * std::cos(static_cast<double>(spec_.petals) * t));
    }

    Eigen::Vector2d rotate(const Eigen::Vector2d& point) const
    {
        const double c = std::cos(spec_.rotation);
        const double s = std::sin(spec_.rotation);
        return {c * point[0] - s * point[1],
                s * point[0] + c * point[1]};
    }

    FlowerSpec spec_;
};

double exact_value(const Eigen::Vector2d& point)
{
    return std::exp(point[0]) * std::cos(point[1])
         + kQuadraticCoefficient * point.squaredNorm();
}

Eigen::Vector2d exact_gradient(const Eigen::Vector2d& point)
{
    const double exponential = std::exp(point[0]);
    return {exponential * std::cos(point[1])
                + 2.0 * kQuadraticCoefficient * point[0],
            -exponential * std::sin(point[1])
                + 2.0 * kQuadraticCoefficient * point[1]};
}

double inf_norm(const Eigen::VectorXd& values)
{
    return values.size() == 0 ? 0.0 : values.cwiseAbs().maxCoeff();
}

std::string bvp_name(LaplaceBvpType2D type)
{
    return type == LaplaceBvpType2D::InteriorDirichlet
        ? "dirichlet" : "neumann";
}

struct RestrictSpec {
    std::string name;
    LaplaceBvpRestrictMethod2D method =
        LaplaceBvpRestrictMethod2D::SixPointQuadraticCrossingOwner;
    LaplaceP2JointPolynomialRestrictOptions2D joint_options;
};

struct Record {
    std::string flower;
    std::string bvp;
    std::string restrict;
    int n = 0;
    int panels = 0;
    int dofs = 0;
    double h = 0.0;
    double bulk_linf = 0.0;
    double bulk_rms = 0.0;
    double trace_linf = 0.0;
    double ghost_linf = 0.0;
    double density_linf = 0.0;
    double final_residual = std::numeric_limits<double>::quiet_NaN();
    double bulk_linf_order = std::numeric_limits<double>::quiet_NaN();
    double bulk_rms_order = std::numeric_limits<double>::quiet_NaN();
    int iterations = 0;
    bool converged = false;
    int shared_side_stencils = 0;
    int relocated_stencils = 0;
    int rejected_candidates = 0;
    int gap_fallbacks = 0;
    int endpoint_fallbacks = 0;
    double max_weight_l1 = std::numeric_limits<double>::quiet_NaN();
    double max_anchor_distance_over_h =
        std::numeric_limits<double>::quiet_NaN();
};

Record run_case(const FlowerSpec& spec,
                LaplaceBvpType2D type,
                const RestrictSpec& restrict,
                int n)
{
    if (n < 16)
        throw std::invalid_argument("flower grid size must be at least 16");
    const double h = kBoxLength / static_cast<double>(n);
    const CartesianGrid2D grid(
        {kBoxMin, kBoxMin}, {h, h}, {n, n}, DofLayout2D::Node);
    auto curve = std::make_shared<FlowerCurve2D>(spec);
    const Interface2D iface =
        CurveResampler2D::discretize_quadratic_lagrange(
            curve, h, kPanelLengthOverH);

    LaplaceBvpOptions2D options;
    options.formulation = LaplaceBvpFormulation2D::ExteriorTraceCauchy;
    options.correction_method = LaplaceCorrectionMethod2D::CrossingOwner;
    options.restrict_method = restrict.method;
    options.joint_restrict = restrict.joint_options;
    options.crossing_jet_scheme =
        LaplaceCrossingJetScheme2D::ArcLengthBSplineCrossingJet;
    options.restrict_stencil_radius = 2;
    LaplaceBvp2D solver(grid, iface, type, options);
    const auto* diagnostics =
        restrict.name.rfind("usn_p2", 0) == 0
            ? solver.joint_polynomial_restrict_diagnostics()
            : nullptr;
    if (diagnostics != nullptr) {
        std::cout << '[' << restrict.name << "] side-stencils="
                  << diagnostics->unified_p2_stencils
                  << " relocated="
                  << diagnostics->unified_p2_relocated_stencils
                  << " rejected-candidates="
                  << diagnostics->unified_p2_candidate_rejections
                  << " exact-owners="
                  << diagnostics->exact_crossing_owners
                  << " gap/endpoint="
                  << diagnostics->gap_fallback_owners << '/'
                  << diagnostics->endpoint_fallback_owners
                  << " max|w|_1="
                  << diagnostics->max_six_point_weight_l1
                  << " max-anchor/h="
                  << diagnostics->max_same_side_center_distance_over_h
                  << '\n';
    }

    const int nq = iface.num_points();
    Eigen::VectorXd dirichlet(nq);
    Eigen::VectorXd neumann(nq);
    for (int q = 0; q < nq; ++q) {
        const Eigen::Vector2d point = iface.points().row(q).transpose();
        Eigen::Vector2d normal = iface.normals().row(q).transpose();
        normal.normalize();
        dirichlet[q] = exact_value(point);
        neumann[q] = exact_gradient(point).dot(normal);
    }
    const Eigen::VectorXd& boundary =
        type == LaplaceBvpType2D::InteriorDirichlet
            ? dirichlet : neumann;
    Eigen::VectorXd volume_rhs = Eigen::VectorXd::Zero(grid.num_dofs());
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (solver.grid_pair().domain_label(node) > 0)
            volume_rhs[node] = kVolumeForcing;
    }
    std::vector<Eigen::VectorXd> rhs_derivs(
        static_cast<std::size_t>(nq), Eigen::VectorXd::Zero(1));
    for (Eigen::VectorXd& derivatives : rhs_derivs)
        derivatives[0] = kVolumeForcing;
    const LaplaceBvpSolveResult2D solution = solver.solve(
        boundary, volume_rhs, rhs_derivs, 100, 2.0e-10, 60);

    double shift = 0.0;
    int interior_count = 0;
    if (type == LaplaceBvpType2D::InteriorNeumann) {
        for (int node = 0; node < grid.num_dofs(); ++node) {
            if (solver.grid_pair().domain_label(node) <= 0)
                continue;
            const auto coordinate = grid.coord(node);
            const Eigen::Vector2d point(coordinate[0], coordinate[1]);
            shift += exact_value(point) - solution.u_physical[node];
            ++interior_count;
        }
        if (interior_count == 0)
            throw std::runtime_error("flower contains no interior grid nodes");
        shift /= static_cast<double>(interior_count);
    }

    double bulk_linf = 0.0;
    double bulk_sum_sq = 0.0;
    interior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (solver.grid_pair().domain_label(node) <= 0)
            continue;
        const auto coordinate = grid.coord(node);
        const Eigen::Vector2d point(coordinate[0], coordinate[1]);
        const double error =
            solution.u_physical[node] + shift - exact_value(point);
        bulk_linf = std::max(bulk_linf, std::abs(error));
        bulk_sum_sq += error * error;
        ++interior_count;
    }
    if (interior_count == 0)
        throw std::runtime_error("flower contains no interior grid nodes");

    const Eigen::VectorXd trace_error =
        type == LaplaceBvpType2D::InteriorDirichlet
            ? solution.trace - dirichlet
            : solution.normal_trace - neumann;
    const Eigen::VectorXd ghost_residual =
        type == LaplaceBvpType2D::InteriorDirichlet
            ? solution.ghost_normal_trace
            : solution.ghost_trace;

    Eigen::VectorXd density_error;
    if (type == LaplaceBvpType2D::InteriorDirichlet) {
        density_error = solution.density - neumann;
    } else {
        double boundary_shift = 0.0;
        double total_weight = 0.0;
        for (int q = 0; q < nq; ++q) {
            boundary_shift += iface.weights()[q]
                            * (dirichlet[q] - solution.density[q]);
            total_weight += iface.weights()[q];
        }
        boundary_shift /= total_weight;
        density_error = solution.density
                      + Eigen::VectorXd::Constant(nq, boundary_shift)
                      - dirichlet;
    }

    Record record{spec.name,
                  bvp_name(type),
                  restrict.name,
                  n,
                  iface.num_panels(),
                  nq,
                  h,
                  bulk_linf,
                  std::sqrt(
                      bulk_sum_sq / static_cast<double>(interior_count)),
                  inf_norm(trace_error),
                  inf_norm(ghost_residual),
                  inf_norm(density_error),
                  solution.residuals.empty()
                      ? std::numeric_limits<double>::quiet_NaN()
                      : solution.residuals.back(),
                  std::numeric_limits<double>::quiet_NaN(),
                  std::numeric_limits<double>::quiet_NaN(),
                  solution.iterations,
                  solution.converged};
    if (diagnostics != nullptr) {
        record.shared_side_stencils = diagnostics->unified_p2_stencils;
        record.relocated_stencils =
            diagnostics->unified_p2_relocated_stencils;
        record.rejected_candidates =
            diagnostics->unified_p2_candidate_rejections;
        record.gap_fallbacks = diagnostics->gap_fallback_owners;
        record.endpoint_fallbacks =
            diagnostics->endpoint_fallback_owners;
        record.max_weight_l1 = diagnostics->max_six_point_weight_l1;
        record.max_anchor_distance_over_h =
            diagnostics->max_same_side_center_distance_over_h;
    }
    return record;
}

void add_orders(std::vector<Record>& records)
{
    for (std::size_t i = 1; i < records.size(); ++i) {
        Record& current = records[i];
        const Record& previous = records[i - 1];
        if (current.flower != previous.flower
            || current.bvp != previous.bvp
            || current.restrict != previous.restrict)
            continue;
        const double denominator = std::log(previous.h / current.h);
        current.bulk_linf_order =
            std::log(previous.bulk_linf / current.bulk_linf) / denominator;
        current.bulk_rms_order =
            std::log(previous.bulk_rms / current.bulk_rms) / denominator;
    }
}

void print_records(const std::vector<Record>& records)
{
    std::cout << "\nflower      bvp         restrict          N  panels  dofs  gmres  conv"
                 "      bulk_Linf   p_inf       bulk_RMS   p_rms"
                 "      trace_inf      ghost_inf    density_inf      residual\n";
    for (const Record& record : records) {
        std::cout << std::left << std::setw(11) << record.flower
                  << std::setw(12) << record.bvp << std::right
                  << std::left << std::setw(17) << record.restrict
                  << std::right
                  << std::setw(5) << record.n
                  << std::setw(8) << record.panels
                  << std::setw(6) << record.dofs
                  << std::setw(7) << record.iterations
                  << std::setw(6) << (record.converged ? "yes" : "no")
                  << std::scientific << std::setprecision(4)
                  << std::setw(15) << record.bulk_linf;
        if (std::isfinite(record.bulk_linf_order))
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(8) << record.bulk_linf_order;
        else
            std::cout << std::setw(8) << "-";
        std::cout << std::scientific << std::setprecision(4)
                  << std::setw(15) << record.bulk_rms;
        if (std::isfinite(record.bulk_rms_order))
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(8) << record.bulk_rms_order;
        else
            std::cout << std::setw(8) << "-";
        std::cout << std::scientific << std::setprecision(4)
                  << std::setw(15) << record.trace_linf
                  << std::setw(15) << record.ghost_linf
                  << std::setw(15) << record.density_linf
                  << std::setw(14) << record.final_residual << '\n';
    }
}

double fitted_order(const std::vector<const Record*>& group,
                    bool use_rms)
{
    if (group.size() < 2)
        return std::numeric_limits<double>::quiet_NaN();
    double mean_log_h = 0.0;
    double mean_log_error = 0.0;
    for (const Record* record : group) {
        mean_log_h += std::log(record->h);
        mean_log_error += std::log(
            use_rms ? record->bulk_rms : record->bulk_linf);
    }
    mean_log_h /= static_cast<double>(group.size());
    mean_log_error /= static_cast<double>(group.size());
    double numerator = 0.0;
    double denominator = 0.0;
    for (const Record* record : group) {
        const double x = std::log(record->h) - mean_log_h;
        const double y = std::log(
            use_rms ? record->bulk_rms : record->bulk_linf)
                       - mean_log_error;
        numerator += x * y;
        denominator += x * x;
    }
    return numerator / denominator;
}

void print_fitted_summary(const std::vector<Record>& records)
{
    std::cout << "\nleast-squares order over all listed levels\n"
                 "flower      bvp         restrict       p_Linf   p_RMS   gmres_min/max\n";
    for (std::size_t begin = 0; begin < records.size();) {
        std::size_t end = begin + 1;
        while (end < records.size()
               && records[end].flower == records[begin].flower
               && records[end].bvp == records[begin].bvp
               && records[end].restrict == records[begin].restrict) {
            ++end;
        }
        std::vector<const Record*> group;
        int min_iterations = records[begin].iterations;
        int max_iterations = records[begin].iterations;
        for (std::size_t i = begin; i < end; ++i) {
            group.push_back(&records[i]);
            min_iterations = std::min(min_iterations, records[i].iterations);
            max_iterations = std::max(max_iterations, records[i].iterations);
        }
        const double linf_order = fitted_order(group, false);
        const double rms_order = fitted_order(group, true);
        std::cout << std::left << std::setw(11) << records[begin].flower
                  << std::setw(12) << records[begin].bvp
                  << std::setw(15) << records[begin].restrict
                  << std::right;
        if (std::isfinite(linf_order)) {
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(9) << linf_order
                      << std::setw(8) << rms_order;
        } else {
            std::cout << std::setw(9) << '-' << std::setw(8) << '-';
        }
        std::cout << std::setw(8) << min_iterations << '/'
                  << max_iterations << '\n';
        begin = end;
    }
}

void write_csv(const std::vector<Record>& records)
{
#ifdef KFBIM_APP_OUTPUT_DIR
    const std::filesystem::path output_dir = KFBIM_APP_OUTPUT_DIR;
#else
    const std::filesystem::path output_dir = "output";
#endif
    std::filesystem::create_directories(output_dir);
    const std::filesystem::path path =
        output_dir / "laplace_exterior_trace_flower_convergence_2d.csv";
    std::ofstream output(path);
    output << "flower,bvp,restrict,N,h,panels,dofs,gmres,converged,bulk_linf,"
              "bulk_linf_order,bulk_rms,bulk_rms_order,trace_linf,"
              "ghost_linf,density_linf,final_residual,shared_side_stencils,"
              "relocated_stencils,rejected_candidates,gap_fallbacks,"
              "endpoint_fallbacks,max_weight_l1,"
              "max_anchor_distance_over_h\n";
    output << std::setprecision(17);
    for (const Record& record : records) {
        output << record.flower << ',' << record.bvp << ','
               << record.restrict << ',' << record.n << ','
               << record.h << ',' << record.panels << ',' << record.dofs << ','
               << record.iterations << ',' << (record.converged ? 1 : 0) << ','
               << record.bulk_linf << ',' << record.bulk_linf_order << ','
               << record.bulk_rms << ',' << record.bulk_rms_order << ','
               << record.trace_linf << ',' << record.ghost_linf << ','
               << record.density_linf << ',' << record.final_residual << ','
               << record.shared_side_stencils << ','
               << record.relocated_stencils << ','
               << record.rejected_candidates << ','
               << record.gap_fallbacks << ','
               << record.endpoint_fallbacks << ','
               << record.max_weight_l1 << ','
               << record.max_anchor_distance_over_h << '\n';
    }
    std::cout << "CSV: " << path.string() << '\n';
}

std::vector<int> parse_levels(int argc, char** argv)
{
    std::vector<int> levels;
    for (int i = 1; i < argc; ++i)
        levels.push_back(std::stoi(argv[i]));
    if (levels.empty())
        levels = {32, 64, 128};
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    return levels;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        std::cout << std::unitbuf;
        const std::vector<int> levels = parse_levels(argc, argv);
        const std::vector<FlowerSpec> flowers = {
            {"flower3", 3, 0.72, 0.18, 0.17, {0.05, -0.03}},
            {"flower5", 5, 0.70, 0.16, -0.23, {-0.04, 0.03}},
            {"flower7", 7, 0.68, 0.10, 0.31, {0.02, 0.04}}};
        const std::vector<LaplaceBvpType2D> bvp_types = {
            LaplaceBvpType2D::InteriorDirichlet,
            LaplaceBvpType2D::InteriorNeumann};
        const std::vector<RestrictSpec> restricts = {
            {"six_point",
             LaplaceBvpRestrictMethod2D::
                 SixPointQuadraticCrossingOwner,
             {}},
            {"joint_cubic",
             LaplaceBvpRestrictMethod2D::
                 JointPolynomialCrossingOwner,
             {}},
            {"usn_p2",
             LaplaceBvpRestrictMethod2D::
                 UnifiedSpatialNormalP2CrossingOwner,
             {}},
            {"usn_p2_dof_ext",
             LaplaceBvpRestrictMethod2D::
                 UnifiedSpatialNormalP2DofCauchyExterior,
             {}}};

        std::cout << "KFBI2D exterior-trace Cauchy flower convergence\n"
                  << "exact u=exp(x)cos(y)+0.12(x^2+y^2), -Delta u=-0.48, "
                     "correction=crossing_owner, "
                     "scheme=ALS-CJ, phi=periodic-BSpline-P3, "
                     "psi=periodic-BSpline-P2\n"
                  << "USN-P2 = Unified Spatial-Normal P2 Restriction: "
                     "strict complete spatial P2 + trace-point Cauchy jump "
                     "+ unified normal P2\n"
                  << "USN-P2-DOF-EXT = one DOF-local Cauchy polynomial, "
                     "interior samples mapped directly to the exterior "
                     "virtual branch, one normal P2 fit\n";
        std::vector<Record> records;
        for (const FlowerSpec& flower : flowers) {
            for (LaplaceBvpType2D type : bvp_types) {
                for (const RestrictSpec& restrict : restricts) {
                    for (int n : levels) {
                        std::cout << "[run] flower=" << flower.name
                                  << " bvp=" << bvp_name(type)
                                  << " restrict=" << restrict.name
                                  << " N=" << n << '\n';
                        records.push_back(
                            run_case(flower, type, restrict, n));
                    }
                }
            }
        }
        add_orders(records);
        print_records(records);
        print_fitted_summary(records);
        write_csv(records);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "flower convergence error: " << error.what() << '\n';
        return 1;
    }
}
