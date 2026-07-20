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

#include "src/grid/cartesian_grid_2d.hpp"
#include "src/interface/interface_2d.hpp"
#include "src/operators/laplace_neumann_exterior_trace_2d.hpp"

using namespace kfbim;

namespace {

struct LShapeInterfaceData {
    Interface2D iface;
    std::vector<int> active_points;
};

struct LevelMetrics {
    int n = 0;
    int active_points = 0;
    double h = 0.0;
    int iterations = 0;
    bool augmented_converged = false;
    bool physical_converged = false;
    double elapsed_sec = 0.0;
    double compatibility_shift = 0.0;
    double constant_mode_inf = 0.0;
    double lambda = 0.0;
    double gauge = 0.0;
    double augmented_rel = 0.0;
    double physical_rel = 0.0;
    double exterior_trace_rel = 0.0;
    double physical_inf = 0.0;
    double exterior_trace_inf = 0.0;
    double trace_closure_inf = 0.0;
    double superposition_bulk_inf = 0.0;
    double trace_error_inf = 0.0;
    double trace_error_wrms = 0.0;
    double normal_error_inf = 0.0;
    double interior_bulk_inf = 0.0;
    double interior_bulk_rms = 0.0;
    double interior_core_inf = 0.0;
    double exterior_bulk_inf = 0.0;
    double max_error_x = 0.0;
    double max_error_y = 0.0;
};

Eigen::Vector2d normalized_or_fallback(Eigen::Vector2d value)
{
    const double length = value.norm();
    if (length <= 1.0e-14)
        return {1.0, 0.0};
    return value / length;
}

std::vector<Eigen::Vector2d> make_half_grid_l_shape_vertices(double h)
{
    constexpr double box_lower = -1.5;
    const int intervals =
        std::max(1, static_cast<int>(std::llround(3.0 / h)));
    const int margin_cells =
        static_cast<int>(std::floor(intervals / 6.0));
    const int arm_cells =
        std::max(2, static_cast<int>(std::floor(intervals / 3.0)));
    const double a =
        box_lower + (static_cast<double>(margin_cells) + 0.5) * h;
    const double b = a + static_cast<double>(arm_cells) * h;
    const double c = b + static_cast<double>(arm_cells) * h;
    return {{a, a}, {c, a}, {c, b}, {b, b}, {b, c}, {a, c}};
}

LShapeInterfaceData make_l_shape_interface(double h)
{
    const std::vector<Eigen::Vector2d> vertices =
        make_half_grid_l_shape_vertices(h);
    const int n_edges = static_cast<int>(vertices.size());

    std::vector<Eigen::Vector2d> edge_tangents(n_edges);
    std::vector<Eigen::Vector2d> edge_normals(n_edges);
    for (int edge = 0; edge < n_edges; ++edge) {
        edge_tangents[static_cast<std::size_t>(edge)] =
            normalized_or_fallback(
                vertices[static_cast<std::size_t>((edge + 1) % n_edges)]
                - vertices[static_cast<std::size_t>(edge)]);
        const Eigen::Vector2d tangent =
            edge_tangents[static_cast<std::size_t>(edge)];
        edge_normals[static_cast<std::size_t>(edge)] =
            {tangent[1], -tangent[0]};
    }

    std::vector<Eigen::Vector2d> points;
    std::vector<Eigen::Vector2d> normals;
    std::vector<double> weights;
    std::vector<int> active_points;

    // The six vertices are geometry-only metadata.  Each has zero weight and
    // is deliberately absent from every P2 panel.
    for (int edge = 0; edge < n_edges; ++edge) {
        const Eigen::Vector2d normal = normalized_or_fallback(
            edge_normals[static_cast<std::size_t>(
                (edge + n_edges - 1) % n_edges)]
            + edge_normals[static_cast<std::size_t>(edge)]);
        points.push_back(vertices[static_cast<std::size_t>(edge)]);
        normals.push_back(normal);
        weights.push_back(0.0);
    }

    std::vector<std::vector<int>> edge_point_ids(
        static_cast<std::size_t>(n_edges));
    for (int edge = 0; edge < n_edges; ++edge) {
        const Eigen::Vector2d start =
            vertices[static_cast<std::size_t>(edge)];
        const Eigen::Vector2d end =
            vertices[static_cast<std::size_t>((edge + 1) % n_edges)];
        const Eigen::Vector2d tangent =
            edge_tangents[static_cast<std::size_t>(edge)];
        const Eigen::Vector2d normal =
            edge_normals[static_cast<std::size_t>(edge)];
        const double length = (end - start).norm();
        const int count =
            std::max(1, static_cast<int>(std::llround(length / h)));
        for (int k = 0; k < count; ++k) {
            const double s = (static_cast<double>(k) + 0.5) * h;
            if (!(s > 0.0) || !(s < length))
                continue;
            const int q = static_cast<int>(points.size());
            points.push_back(start + s * tangent);
            normals.push_back(normal);
            weights.push_back(h);
            active_points.push_back(q);
            edge_point_ids[static_cast<std::size_t>(edge)].push_back(q);
        }
    }

    std::vector<std::array<int, 3>> panel_rows;
    std::vector<int> panel_edge_ids;
    for (int edge = 0; edge < n_edges; ++edge) {
        const std::vector<int>& ids =
            edge_point_ids[static_cast<std::size_t>(edge)];
        if (ids.size() < 3) {
            throw std::invalid_argument(
                "L-shape edge has too few points for a P2 panel");
        }
        int last_covered = -1;
        for (int start = 0;
             start + 2 < static_cast<int>(ids.size());
             start += 2) {
            panel_rows.push_back(
                {ids[static_cast<std::size_t>(start)],
                 ids[static_cast<std::size_t>(start + 1)],
                 ids[static_cast<std::size_t>(start + 2)]});
            panel_edge_ids.push_back(edge);
            last_covered = start + 2;
        }
        if (last_covered < static_cast<int>(ids.size()) - 1) {
            const int m = static_cast<int>(ids.size());
            panel_rows.push_back(
                {ids[static_cast<std::size_t>(m - 3)],
                 ids[static_cast<std::size_t>(m - 2)],
                 ids[static_cast<std::size_t>(m - 1)]});
            panel_edge_ids.push_back(edge);
        }
    }

    Eigen::MatrixX2d point_matrix(points.size(), 2);
    Eigen::MatrixX2d normal_matrix(normals.size(), 2);
    Eigen::VectorXd weight_vector(weights.size());
    for (int q = 0; q < static_cast<int>(points.size()); ++q) {
        point_matrix.row(q) =
            points[static_cast<std::size_t>(q)].transpose();
        normal_matrix.row(q) =
            normals[static_cast<std::size_t>(q)].transpose();
        weight_vector[q] = weights[static_cast<std::size_t>(q)];
    }

    Eigen::MatrixXi panel_point_indices(panel_rows.size(), 3);
    Eigen::VectorXi components =
        Eigen::VectorXi::Zero(static_cast<int>(panel_rows.size()));
    PanelSideGeometry2D panel_sides;
    panel_sides.point_normals.resize(
        3 * static_cast<int>(panel_rows.size()), 2);
    panel_sides.point_tangents.resize(
        3 * static_cast<int>(panel_rows.size()), 2);
    for (int panel = 0;
         panel < static_cast<int>(panel_rows.size());
         ++panel) {
        const std::array<int, 3>& row =
            panel_rows[static_cast<std::size_t>(panel)];
        const int edge = panel_edge_ids[static_cast<std::size_t>(panel)];
        for (int local = 0; local < 3; ++local) {
            panel_point_indices(panel, local) =
                row[static_cast<std::size_t>(local)];
            const int side_row = 3 * panel + local;
            panel_sides.point_normals.row(side_row) =
                edge_normals[static_cast<std::size_t>(edge)].transpose();
            panel_sides.point_tangents.row(side_row) =
                edge_tangents[static_cast<std::size_t>(edge)].transpose();
        }
    }

    std::vector<InterfacePointKind2D> point_kind(
        points.size(), InterfacePointKind2D::Smooth);
    std::vector<int> corner_index_by_point(points.size(), -1);
    std::vector<int> first_panel_by_edge(
        static_cast<std::size_t>(n_edges), -1);
    std::vector<int> last_panel_by_edge(
        static_cast<std::size_t>(n_edges), -1);
    for (int panel = 0;
         panel < static_cast<int>(panel_edge_ids.size());
         ++panel) {
        const int edge = panel_edge_ids[static_cast<std::size_t>(panel)];
        if (first_panel_by_edge[static_cast<std::size_t>(edge)] < 0)
            first_panel_by_edge[static_cast<std::size_t>(edge)] = panel;
        last_panel_by_edge[static_cast<std::size_t>(edge)] = panel;
    }

    std::vector<CornerData2D> corners;
    corners.reserve(static_cast<std::size_t>(n_edges));
    for (int edge = 0; edge < n_edges; ++edge) {
        point_kind[static_cast<std::size_t>(edge)] =
            InterfacePointKind2D::Corner;
        corner_index_by_point[static_cast<std::size_t>(edge)] = edge;
        const int previous = (edge + n_edges - 1) % n_edges;
        CornerData2D corner;
        corner.point = edge;
        corner.prev_panel =
            last_panel_by_edge[static_cast<std::size_t>(previous)];
        corner.next_panel =
            first_panel_by_edge[static_cast<std::size_t>(edge)];
        corner.tangent_minus =
            edge_tangents[static_cast<std::size_t>(previous)];
        corner.tangent_plus =
            edge_tangents[static_cast<std::size_t>(edge)];
        corner.normal_minus =
            edge_normals[static_cast<std::size_t>(previous)];
        corner.normal_plus =
            edge_normals[static_cast<std::size_t>(edge)];
        corner.turn_angle = std::atan2(
            corner.tangent_minus[0] * corner.tangent_plus[1]
                - corner.tangent_minus[1] * corner.tangent_plus[0],
            corner.tangent_minus.dot(corner.tangent_plus));
        corners.push_back(corner);
    }

    Interface2D iface(std::move(point_matrix),
                      std::move(normal_matrix),
                      std::move(weight_vector),
                      3,
                      std::move(panel_point_indices),
                      std::move(components),
                      std::move(panel_sides),
                      std::move(point_kind),
                      std::move(corner_index_by_point),
                      std::move(corners),
                      PanelNodeLayout2D::QuadraticLagrange);
    return {std::move(iface), std::move(active_points)};
}

double exact_u(double x, double y)
{
    return x * x * x + 3.0 * x * x * y
         - 3.0 * x * y * y - y * y * y;
}

Eigen::Vector2d exact_gradient(double x, double y)
{
    return {3.0 * x * x + 6.0 * x * y - 3.0 * y * y,
            3.0 * x * x - 6.0 * x * y - 3.0 * y * y};
}

double distance_to_l_shape_boundary(Eigen::Vector2d point, double h)
{
    const std::vector<Eigen::Vector2d> vertices =
        make_half_grid_l_shape_vertices(h);
    double distance = std::numeric_limits<double>::infinity();
    for (int edge = 0;
         edge < static_cast<int>(vertices.size());
         ++edge) {
        const Eigen::Vector2d a =
            vertices[static_cast<std::size_t>(edge)];
        const Eigen::Vector2d b =
            vertices[static_cast<std::size_t>(
                (edge + 1) % static_cast<int>(vertices.size()))];
        const Eigen::Vector2d delta = b - a;
        const double t = std::clamp(
            (point - a).dot(delta) / delta.squaredNorm(), 0.0, 1.0);
        distance = std::min(distance, (point - (a + t * delta)).norm());
    }
    return distance;
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

double convergence_order(double previous_error,
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

LevelMetrics run_level(int n, int max_iter, double tolerance, int restart)
{
    if (n < 12)
        throw std::invalid_argument("L-shape grid size must be at least 12");
    const double h = 3.0 / static_cast<double>(n);
    CartesianGrid2D grid({-1.5, -1.5},
                         {h, h},
                         {n, n},
                         DofLayout2D::Node);
    LShapeInterfaceData geometry = make_l_shape_interface(h);

    LaplaceNeumannExteriorTraceOptions2D options;
    options.active_interface_points = geometry.active_points;
    LaplaceNeumannExteriorTrace2D solver(grid, geometry.iface, options);

    const int n_active = solver.problem_size();
    Eigen::VectorXd g(n_active);
    Eigen::VectorXd exact_trace_raw(n_active);
    for (int a = 0; a < n_active; ++a) {
        const int q = geometry.active_points[static_cast<std::size_t>(a)];
        const double x = geometry.iface.points()(q, 0);
        const double y = geometry.iface.points()(q, 1);
        const Eigen::Vector2d normal =
            geometry.iface.normals().row(q).transpose();
        exact_trace_raw[a] = exact_u(x, y);
        g[a] = exact_gradient(x, y).dot(normal);
    }
    const double exact_trace_mean =
        solver.normalized_weights().dot(exact_trace_raw);
    const Eigen::VectorXd exact_trace =
        exact_trace_raw - exact_trace_mean
            * Eigen::VectorXd::Ones(n_active);

    const auto start = std::chrono::steady_clock::now();
    const LaplaceNeumannExteriorTraceSolveResult2D result =
        solver.solve(g, max_iter, tolerance, restart);
    const double elapsed =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();

    const Eigen::VectorXd trace_error =
        result.dirichlet_trace - exact_trace;
    const double trace_error_inf =
        trace_error.cwiseAbs().maxCoeff();
    const double trace_error_wrms = std::sqrt(
        (solver.active_weights().array()
         * trace_error.array().square()).sum()
        / solver.active_weights().sum());
    const double normal_error_inf =
        (result.normal_trace_interior
         - result.compatible_neumann_data)
            .cwiseAbs().maxCoeff();

    double interior_inf = 0.0;
    double interior_sum_sq = 0.0;
    double interior_core_inf = 0.0;
    double exterior_inf = 0.0;
    double max_x = 0.0;
    double max_y = 0.0;
    int interior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        const std::array<double, 2> coordinate = grid.coord(node);
        if (solver.grid_pair().domain_label(node) > 0) {
            const double error = std::abs(
                result.u_bulk[node]
                - (exact_u(coordinate[0], coordinate[1])
                   - exact_trace_mean));
            interior_sum_sq += error * error;
            ++interior_count;
            if (error > interior_inf) {
                interior_inf = error;
                max_x = coordinate[0];
                max_y = coordinate[1];
            }
            if (distance_to_l_shape_boundary(
                    {coordinate[0], coordinate[1]}, h) >= 4.0 * h) {
                interior_core_inf = std::max(interior_core_inf, error);
            }
        } else {
            exterior_inf =
                std::max(exterior_inf, std::abs(result.u_bulk[node]));
        }
    }

    LevelMetrics metrics;
    metrics.n = n;
    metrics.active_points = n_active;
    metrics.h = h;
    metrics.iterations = result.iterations;
    metrics.augmented_converged = result.augmented_converged;
    metrics.physical_converged = result.physical_converged;
    metrics.elapsed_sec = elapsed;
    metrics.compatibility_shift = result.compatibility_mean_removed;
    metrics.constant_mode_inf = result.constant_mode_residual_inf;
    metrics.lambda = result.bordered_multiplier;
    metrics.gauge = result.weighted_trace_mean;
    metrics.augmented_rel = result.augmented_relative_residual;
    metrics.physical_rel = result.physical_relative_residual;
    metrics.exterior_trace_rel = result.exterior_trace_relative;
    metrics.physical_inf =
        result.physical_trace_residual.cwiseAbs().maxCoeff();
    metrics.exterior_trace_inf =
        result.trace_exterior.cwiseAbs().maxCoeff();
    metrics.trace_closure_inf = result.trace_residual_closure_inf;
    metrics.superposition_bulk_inf = result.superposition_bulk_inf;
    metrics.trace_error_inf = trace_error_inf;
    metrics.trace_error_wrms = trace_error_wrms;
    metrics.normal_error_inf = normal_error_inf;
    metrics.interior_bulk_inf = interior_inf;
    metrics.interior_bulk_rms = interior_count > 0
        ? std::sqrt(interior_sum_sq / static_cast<double>(interior_count))
        : 0.0;
    metrics.interior_core_inf = interior_core_inf;
    metrics.exterior_bulk_inf = exterior_inf;
    metrics.max_error_x = max_x;
    metrics.max_error_y = max_y;
    return metrics;
}

void write_csv(const std::filesystem::path& path,
               const std::vector<LevelMetrics>& levels)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error("cannot open result CSV: " + path.string());
    output << "N,h,active,iterations,augmented_converged,"
              "physical_converged,elapsed_sec,"
              "compatibility_shift,constant_mode_inf,lambda,gauge,"
              "augmented_rel,physical_rel,exterior_trace_rel,physical_inf,"
              "exterior_trace_inf,"
              "trace_closure_inf,superposition_bulk_inf,trace_error_inf,"
              "trace_error_wrms,"
              "normal_error_inf,interior_bulk_inf,interior_bulk_rms,"
              "interior_core_inf,exterior_bulk_inf,max_error_x,max_error_y\n";
    output << std::setprecision(17);
    for (const LevelMetrics& level : levels) {
        output << level.n << ',' << level.h << ',' << level.active_points
               << ',' << level.iterations << ','
               << (level.augmented_converged ? 1 : 0) << ','
               << (level.physical_converged ? 1 : 0) << ','
               << level.elapsed_sec << ','
               << level.compatibility_shift << ','
               << level.constant_mode_inf << ',' << level.lambda << ','
               << level.gauge << ',' << level.augmented_rel << ','
               << level.physical_rel << ',' << level.exterior_trace_rel
               << ',' << level.physical_inf << ','
               << level.exterior_trace_inf << ','
               << level.trace_closure_inf << ','
               << level.superposition_bulk_inf << ','
               << level.trace_error_inf << ',' << level.trace_error_wrms << ','
               << level.normal_error_inf << ',' << level.interior_bulk_inf
               << ',' << level.interior_bulk_rms << ','
               << level.interior_core_inf << ',' << level.exterior_bulk_inf
               << ',' << level.max_error_x << ',' << level.max_error_y
               << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        std::vector<int> levels;
        for (int i = 1; i < argc; ++i)
            levels.push_back(std::stoi(argv[i]));
        if (levels.empty())
            levels = {30, 60, 120};

        const int max_iter = environment_int(
            "KFBIM_EXT_TRACE_MAX_ITER", 200);
        const int restart = environment_int(
            "KFBIM_EXT_TRACE_RESTART", 50);
        const double tolerance = environment_double(
            "KFBIM_EXT_TRACE_TOL", 1.0e-8);

        std::cout << "KFBI2D L-shape Neumann exterior-trace formulation\n"
                  << "smooth antisymmetric harmonic cubic; no singular correction\n"
                  << "tol=" << tolerance << " max_iter=" << max_iter
                  << " restart=" << restart << "\n\n";
        std::cout << std::setw(5) << "N"
                  << std::setw(7) << "Nq"
                  << std::setw(7) << "iter"
                  << std::setw(5) << "ok"
                  << std::setw(13) << "A1_inf"
                  << std::setw(13) << "ext_trace"
                  << std::setw(13) << "bulk_in"
                  << std::setw(8) << "order"
                  << std::setw(13) << "bulk_out"
                  << std::setw(13) << "trace_err"
                  << std::setw(13) << "lambda"
                  << std::setw(10) << "sec" << '\n';

        std::vector<LevelMetrics> results;
        results.reserve(levels.size());
        double previous_bulk = 0.0;
        double previous_h = 0.0;
        for (int n : levels) {
            LevelMetrics metrics =
                run_level(n, max_iter, tolerance, restart);
            const double order = convergence_order(
                previous_bulk,
                metrics.interior_bulk_inf,
                previous_h,
                metrics.h);
            const bool ok = metrics.augmented_converged
                         && metrics.physical_converged;
            std::cout << std::scientific << std::setprecision(3)
                      << std::setw(5) << metrics.n
                      << std::setw(7) << metrics.active_points
                      << std::setw(7) << metrics.iterations
                      << std::setw(5) << (ok ? "yes" : "no")
                      << std::setw(13) << metrics.constant_mode_inf
                      << std::setw(13) << metrics.exterior_trace_inf
                      << std::setw(13) << metrics.interior_bulk_inf
                      << std::setw(8)
                      << (std::isfinite(order) ? order : 0.0)
                      << std::setw(13) << metrics.exterior_bulk_inf
                      << std::setw(13) << metrics.trace_error_inf
                      << std::setw(13) << metrics.lambda
                      << std::setw(10) << metrics.elapsed_sec << '\n';
            previous_bulk = metrics.interior_bulk_inf;
            previous_h = metrics.h;
            results.push_back(metrics);
        }

#ifdef KFBIM_APP_OUTPUT_DIR
        const std::filesystem::path output_dir = KFBIM_APP_OUTPUT_DIR;
#else
        const std::filesystem::path output_dir = "output";
#endif
        const std::filesystem::path csv_path =
            output_dir / "neumann_exterior_trace_lshape_2d.csv";
        write_csv(csv_path, results);
        std::cout << "\nCSV: " << csv_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
