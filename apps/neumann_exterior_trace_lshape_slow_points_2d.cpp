#define main kfbim_embedded_neumann_exterior_trace_lshape_main
#include "neumann_exterior_trace_lshape_2d.cpp"
#undef main

#include <map>
#include <numeric>
#include <sstream>

namespace {

struct BorderedOperatorView final : public IKFBIOperator {
    explicit BorderedOperatorView(
        const LaplaceNeumannExteriorTrace2D& physical)
        : physical_(physical)
    {}

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override
    {
        const int n = physical_.problem_size();
        if (x.size() != n + 1)
            throw std::invalid_argument("bordered diagnostic input size mismatch");
        Eigen::VectorXd af;
        physical_.apply(x.head(n), af);
        y.resize(n + 1);
        y.head(n) = af + x[n] * physical_.border_column();
        y[n] = physical_.normalized_weights().dot(x.head(n));
    }

    int problem_size() const override
    {
        return physical_.problem_size() + 1;
    }

private:
    const LaplaceNeumannExteriorTrace2D& physical_;
};

struct ResidualSnapshot {
    int iteration = 0;
    double relative_norm = 0.0;
    Eigen::VectorXd residual;
};

struct GmresTrajectory {
    std::vector<ResidualSnapshot> snapshots;
    int iterations = 0;
    double final_true_relative_norm = 0.0;
    double arnoldi_true_residual_difference = 0.0;
};

GmresTrajectory trace_unrestarted_gmres(
    const IKFBIOperator& op,
    const Eigen::VectorXd& rhs,
    int max_iterations,
    double tolerance)
{
    const int n = op.problem_size();
    if (rhs.size() != n)
        throw std::invalid_argument("GMRES diagnostic RHS size mismatch");

    GmresTrajectory result;
    const double beta = rhs.norm();
    if (!(beta > 0.0))
        throw std::invalid_argument("GMRES diagnostic requires a nonzero RHS");
    result.snapshots.push_back({0, 1.0, rhs});

    Eigen::MatrixXd q = Eigen::MatrixXd::Zero(n, max_iterations + 1);
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(
        max_iterations + 1, max_iterations);
    q.col(0) = rhs / beta;

    Eigen::VectorXd last_x = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd last_arnoldi_residual = rhs;
    int completed = 0;
    for (int j = 0; j < max_iterations; ++j) {
        Eigen::VectorXd work;
        op.apply(q.col(j), work);
        for (int i = 0; i <= j; ++i) {
            h(i, j) = q.col(i).dot(work);
            work -= h(i, j) * q.col(i);
        }
        for (int i = 0; i <= j; ++i) {
            const double correction = q.col(i).dot(work);
            h(i, j) += correction;
            work -= correction * q.col(i);
        }
        h(j + 1, j) = work.norm();
        const bool happy_breakdown = h(j + 1, j) < 1.0e-14;
        if (!happy_breakdown)
            q.col(j + 1) = work / h(j + 1, j);

        const int columns = j + 1;
        const Eigen::MatrixXd hbar =
            h.topLeftCorner(columns + 1, columns);
        Eigen::VectorXd small_rhs = Eigen::VectorXd::Zero(columns + 1);
        small_rhs[0] = beta;
        const Eigen::VectorXd y =
            hbar.colPivHouseholderQr().solve(small_rhs);
        const Eigen::VectorXd small_residual = small_rhs - hbar * y;
        last_arnoldi_residual =
            q.leftCols(columns + 1) * small_residual;
        last_x = q.leftCols(columns) * y;
        const double relative = last_arnoldi_residual.norm() / beta;
        result.snapshots.push_back(
            {columns, relative, last_arnoldi_residual});
        completed = columns;
        if (relative < tolerance || happy_breakdown)
            break;
    }

    Eigen::VectorXd image;
    op.apply(last_x, image);
    const Eigen::VectorXd true_residual = rhs - image;
    result.iterations = completed;
    result.final_true_relative_norm = true_residual.norm() / beta;
    result.arnoldi_true_residual_difference =
        (true_residual - last_arnoldi_residual).norm() / beta;
    result.snapshots.back().residual = true_residual;
    result.snapshots.back().relative_norm = result.final_true_relative_norm;
    return result;
}

double unit_fraction(double value)
{
    double fraction = value - std::floor(value);
    if (fraction < 1.0e-13 || 1.0 - fraction < 1.0e-13)
        fraction = 0.0;
    return fraction;
}

double line_phase_distance(double fraction)
{
    return std::min(fraction, 1.0 - fraction);
}

struct PointGeometryDiagnostic {
    int active_index = -1;
    int interface_q = -1;
    int edge = -1;
    double edge_parameter = 0.0;
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    Eigen::Vector2d normal = Eigen::Vector2d::Zero();
    int nearest_corner = -1;
    double nearest_corner_distance_over_h = 0.0;
    double reentrant_distance_over_h = 0.0;
    double box_distance_over_h = 0.0;
    double phase_x = 0.0;
    double phase_y = 0.0;
    double nearest_grid_line_phase = 0.0;
    double nearest_normal_sample_grid_line_phase = 1.0;
    int normal_samples_near_grid_line_005 = 0;
    int restrict_wrong_side_nodes = 0;
    int restrict_exact_owners = 0;
    int restrict_gap_owners = 0;
    int restrict_identified_gap_owners = 0;
    int restrict_unresolved_gap_owners = 0;
    int restrict_endpoint_owners = 0;
    int boundary_clamped_samples = 0;
    int outside_box_queries = 0;
    double restrict_correction_weight_l1 = 0.0;
    double max_sample_correction_weight_l1 = 0.0;
    double max_sample_weight_l1 = 0.0;
};

std::array<double, 4> cubic_lagrange_weights(double coordinate)
{
    constexpr std::array<double, 4> nodes{{-1.0, 0.0, 1.0, 2.0}};
    std::array<double, 4> weights{{1.0, 1.0, 1.0, 1.0}};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i == j)
                continue;
            weights[static_cast<std::size_t>(i)] *=
                (coordinate - nodes[static_cast<std::size_t>(j)])
                / (nodes[static_cast<std::size_t>(i)]
                   - nodes[static_cast<std::size_t>(j)]);
        }
    }
    return weights;
}

void attach_restrict_geometry_diagnostic(
    PointGeometryDiagnostic& result,
    const GridPair2D& grid_pair,
    double h)
{
    constexpr std::array<double, 4> layers{{0.2, 0.6, 1.0, 1.4}};
    const CartesianGrid2D& grid = grid_pair.grid();
    const auto dimensions = grid.dof_dims();
    const std::array<double, 2> first_coordinate = grid.coord(0);
    const Eigen::Vector2d grid_first(
        first_coordinate[0], first_coordinate[1]);
    const Eigen::Vector2d normal = result.normal.normalized();

    for (int side = 0; side < 2; ++side) {
        const bool desired_inside = side == 0;
        const double side_sign = desired_inside ? -1.0 : 1.0;
        for (double layer : layers) {
            const Eigen::Vector2d query =
                result.point + side_sign * layer * h * normal;
            const double grid_x = (query[0] - grid_first[0]) / h;
            const double grid_y = (query[1] - grid_first[1]) / h;
            const double line_distance = std::min(
                line_phase_distance(unit_fraction(grid_x)),
                line_phase_distance(unit_fraction(grid_y)));
            result.nearest_normal_sample_grid_line_phase = std::min(
                result.nearest_normal_sample_grid_line_phase,
                line_distance);
            if (line_distance <= 0.05)
                ++result.normal_samples_near_grid_line_005;
            if (grid_x < 0.0 || grid_x > dimensions[0] - 1
                || grid_y < 0.0 || grid_y > dimensions[1] - 1) {
                ++result.outside_box_queries;
            }

            const int anchor_x = static_cast<int>(std::floor(grid_x));
            const int anchor_y = static_cast<int>(std::floor(grid_y));
            const int raw_start_x = anchor_x - 1;
            const int raw_start_y = anchor_y - 1;
            const int start_x = std::max(
                0, std::min(dimensions[0] - 4, raw_start_x));
            const int start_y = std::max(
                0, std::min(dimensions[1] - 4, raw_start_y));
            if (start_x != raw_start_x || start_y != raw_start_y)
                ++result.boundary_clamped_samples;

            const std::array<double, 4> weights_x =
                cubic_lagrange_weights(
                    grid_x - static_cast<double>(start_x + 1));
            const std::array<double, 4> weights_y =
                cubic_lagrange_weights(
                    grid_y - static_cast<double>(start_y + 1));
            double sample_weight_l1 = 0.0;
            double sample_correction_weight_l1 = 0.0;
            for (int local_y = 0; local_y < 4; ++local_y) {
                for (int local_x = 0; local_x < 4; ++local_x) {
                    const int node = grid.index(
                        start_x + local_x, start_y + local_y);
                    const double weight =
                        weights_x[static_cast<std::size_t>(local_x)]
                        * weights_y[static_cast<std::size_t>(local_y)];
                    sample_weight_l1 += std::abs(weight);
                    const bool node_inside =
                        grid_pair.domain_label(node) > 0;
                    if (node_inside == desired_inside)
                        continue;
                    ++result.restrict_wrong_side_nodes;
                    sample_correction_weight_l1 += std::abs(weight);
                    const std::array<double, 2> coordinate = grid.coord(node);
                    const Eigen::Vector2d node_point(
                        coordinate[0], coordinate[1]);
                    P2CrossingOwner2D owner;
                    if ((node_point - query).norm() <= 1.0e-14) {
                        owner.center_index =
                            grid_pair.nearest_p2_expansion_center(node);
                        owner.status = P2CrossingOwnerStatus2D::
                            EndpointNearestCenter;
                    } else {
                        owner = grid_pair.p2_crossing_owner_between(
                            query, node_point);
                    }
                    switch (owner.status) {
                    case P2CrossingOwnerStatus2D::ExactIntersection:
                        ++result.restrict_exact_owners;
                        break;
                    case P2CrossingOwnerStatus2D::GapFallback:
                        ++result.restrict_gap_owners;
                        if (owner.explicit_gap_intersection)
                            ++result.restrict_identified_gap_owners;
                        else
                            ++result.restrict_unresolved_gap_owners;
                        break;
                    case P2CrossingOwnerStatus2D::EndpointNearestCenter:
                        ++result.restrict_endpoint_owners;
                        break;
                    }
                }
            }
            result.restrict_correction_weight_l1 +=
                sample_correction_weight_l1;
            result.max_sample_correction_weight_l1 = std::max(
                result.max_sample_correction_weight_l1,
                sample_correction_weight_l1);
            result.max_sample_weight_l1 = std::max(
                result.max_sample_weight_l1, sample_weight_l1);
        }
    }
}

PointGeometryDiagnostic point_geometry_diagnostic(
    int active_index,
    int q,
    const Interface2D& iface,
    const std::vector<Eigen::Vector2d>& vertices,
    double h)
{
    PointGeometryDiagnostic result;
    result.active_index = active_index;
    result.interface_q = q;
    result.point = iface.points().row(q).transpose();
    result.normal = iface.normals().row(q).transpose().normalized();

    double best_edge_distance2 = std::numeric_limits<double>::infinity();
    for (int edge = 0; edge < static_cast<int>(vertices.size()); ++edge) {
        const Eigen::Vector2d a = vertices[static_cast<std::size_t>(edge)];
        const Eigen::Vector2d b = vertices[
            static_cast<std::size_t>((edge + 1) % vertices.size())];
        const Eigen::Vector2d direction = b - a;
        const double length2 = direction.squaredNorm();
        const double parameter = std::max(
            0.0,
            std::min(1.0, (result.point - a).dot(direction) / length2));
        const double distance2 =
            (result.point - (a + parameter * direction)).squaredNorm();
        if (distance2 < best_edge_distance2) {
            best_edge_distance2 = distance2;
            result.edge = edge;
            result.edge_parameter = parameter;
        }
    }

    double nearest_corner_distance =
        std::numeric_limits<double>::infinity();
    for (int corner = 0; corner < static_cast<int>(vertices.size()); ++corner) {
        const double distance =
            (result.point - vertices[static_cast<std::size_t>(corner)]).norm();
        if (distance < nearest_corner_distance) {
            nearest_corner_distance = distance;
            result.nearest_corner = corner;
        }
    }
    result.nearest_corner_distance_over_h = nearest_corner_distance / h;
    result.reentrant_distance_over_h =
        (result.point - vertices[3]).norm() / h;
    result.box_distance_over_h = std::min(
        {result.point[0] + 1.5,
         1.5 - result.point[0],
         result.point[1] + 1.5,
         1.5 - result.point[1]}) / h;
    result.phase_x = unit_fraction((result.point[0] + 1.5) / h);
    result.phase_y = unit_fraction((result.point[1] + 1.5) / h);
    result.nearest_grid_line_phase = std::min(
        line_phase_distance(result.phase_x),
        line_phase_distance(result.phase_y));
    return result;
}

struct PointSlowDiagnostic {
    PointGeometryDiagnostic geometry;
    double initial_share = 0.0;
    double early_share = 0.0;
    double late_share = 0.0;
    double late_gain_log10 = 0.0;
    double late_peak_share = 0.0;
    double penultimate_scaled_abs = 0.0;
    double baseline_late_share = 0.0;
    double late_share_excess = 0.0;
    double late_share_ratio = 0.0;
    int late_rank = 0;
};

struct CaseSlowDiagnostic {
    std::string case_id;
    int n_grid = 0;
    double h = 0.0;
    int iterations = 0;
    double final_relative = 0.0;
    double arnoldi_true_difference = 0.0;
    int late_first_snapshot = 0;
    int late_last_snapshot = 0;
    double late_border_energy_share = 0.0;
    std::array<double, 6> edge_late_shares{{0, 0, 0, 0, 0, 0}};
    double reentrant_2h_share = 0.0;
    double reentrant_4h_share = 0.0;
    double reentrant_8h_share = 0.0;
    double any_corner_2h_share = 0.0;
    double any_corner_4h_share = 0.0;
    double box_4h_share = 0.0;
    double box_8h_share = 0.0;
    double grid_line_phase_005_share = 0.0;
    double grid_line_phase_010_share = 0.0;
    double top_5_share = 0.0;
    double top_10_share = 0.0;
    std::vector<PointSlowDiagnostic> points;
    GmresTrajectory trajectory;
};

std::vector<int> snapshot_range(int first, int last)
{
    std::vector<int> indices;
    for (int index = first; index <= last; ++index)
        indices.push_back(index);
    return indices;
}

std::vector<double> energy_shares(
    const GmresTrajectory& trajectory,
    const std::vector<int>& snapshot_indices,
    int interface_size)
{
    Eigen::VectorXd energy = Eigen::VectorXd::Zero(interface_size);
    double total = 0.0;
    for (int index : snapshot_indices) {
        const Eigen::VectorXd head =
            trajectory.snapshots[static_cast<std::size_t>(index)]
                .residual.head(interface_size);
        energy.array() += head.array().square();
        total += head.squaredNorm();
    }
    if (total > 0.0)
        energy /= total;
    return std::vector<double>(energy.data(), energy.data() + energy.size());
}

CaseSlowDiagnostic run_case_slow_diagnostic(
    int n_grid,
    const RigidStudyCase2D& study_case)
{
    CaseSlowDiagnostic output;
    output.case_id = study_case.id;
    output.n_grid = n_grid;
    output.h = 3.0 / static_cast<double>(n_grid);

    CartesianGrid2D grid(
        {-1.5, -1.5}, {output.h, output.h}, {n_grid, n_grid},
        DofLayout2D::Node);
    const RigidTransform2D transform =
        make_rigid_transform(study_case, output.h);
    const std::vector<Eigen::Vector2d> vertices =
        transformed_l_shape_vertices(output.h, transform);
    LShapeInterfaceData geometry =
        make_l_shape_interface(output.h, transform);

    LaplaceNeumannExteriorTraceOptions2D options;
    options.active_interface_points = geometry.active_points;
    options.restrict_method =
        LaplaceNeumannExteriorRestrictMethod2D::
            JointBicubicCubicCrossingOwner;
    options.crossing_jet_scheme =
        LaplaceCrossingJetScheme2D::NurbsSameParameterCrossingJet;
    options.correction_method = LaplaceCorrectionMethod2D::CrossingOwner;
    LaplaceNeumannExteriorTrace2D solver(grid, geometry.iface, options);

    const int active_size = solver.problem_size();
    Eigen::VectorXd neumann(active_size);
    for (int active = 0; active < active_size; ++active) {
        const int q = geometry.active_points[static_cast<std::size_t>(active)];
        const Eigen::Vector2d point =
            geometry.iface.points().row(q).transpose();
        const Eigen::Vector2d normal =
            geometry.iface.normals().row(q).transpose();
        neumann[active] =
            transformed_exact_gradient(point, transform).dot(normal);
    }
    const LaplaceNeumannExteriorTraceRhs2D rhs_data =
        solver.build_rhs(neumann);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(active_size + 1);
    rhs.head(active_size) = rhs_data.trace_rhs;
    const BorderedOperatorView bordered(solver);
    output.trajectory = trace_unrestarted_gmres(
        bordered, rhs, 50, 1.0e-8);
    output.iterations = output.trajectory.iterations;
    output.final_relative = output.trajectory.final_true_relative_norm;
    output.arnoldi_true_difference =
        output.trajectory.arnoldi_true_residual_difference;

    const int final_snapshot =
        static_cast<int>(output.trajectory.snapshots.size()) - 1;
    output.late_last_snapshot = std::max(0, final_snapshot - 1);
    output.late_first_snapshot = std::max(
        1, output.late_last_snapshot - 5);
    const int early_last = std::min(4, output.late_first_snapshot - 1);
    const std::vector<int> early_indices =
        snapshot_range(0, std::max(0, early_last));
    const std::vector<int> late_indices = snapshot_range(
        output.late_first_snapshot, output.late_last_snapshot);
    const std::vector<double> early = energy_shares(
        output.trajectory, early_indices, active_size);
    const std::vector<double> late = energy_shares(
        output.trajectory, late_indices, active_size);
    const std::vector<double> initial = energy_shares(
        output.trajectory, {0}, active_size);

    double augmented_late_energy = 0.0;
    double border_late_energy = 0.0;
    for (int index : late_indices) {
        const Eigen::VectorXd& residual =
            output.trajectory.snapshots[static_cast<std::size_t>(index)]
                .residual;
        augmented_late_energy += residual.squaredNorm();
        border_late_energy += residual[active_size] * residual[active_size];
    }
    output.late_border_energy_share =
        augmented_late_energy > 0.0
            ? border_late_energy / augmented_late_energy
            : 0.0;

    output.points.resize(static_cast<std::size_t>(active_size));
    const double beta = output.trajectory.snapshots.front().residual.norm();
    for (int active = 0; active < active_size; ++active) {
        PointSlowDiagnostic& point =
            output.points[static_cast<std::size_t>(active)];
        const int q = geometry.active_points[static_cast<std::size_t>(active)];
        point.geometry = point_geometry_diagnostic(
            active, q, geometry.iface, vertices, output.h);
        attach_restrict_geometry_diagnostic(
            point.geometry, solver.grid_pair(), output.h);
        point.initial_share = initial[static_cast<std::size_t>(active)];
        point.early_share = early[static_cast<std::size_t>(active)];
        point.late_share = late[static_cast<std::size_t>(active)];
        point.late_gain_log10 = std::log10(
            (point.late_share + 1.0e-18)
            / (point.early_share + 1.0e-18));
        for (int index : late_indices) {
            const Eigen::VectorXd head =
                output.trajectory.snapshots[static_cast<std::size_t>(index)]
                    .residual.head(active_size);
            if (head.squaredNorm() > 0.0) {
                point.late_peak_share = std::max(
                    point.late_peak_share,
                    head[active] * head[active] / head.squaredNorm());
            }
        }
        if (output.late_last_snapshot >= 0) {
            point.penultimate_scaled_abs = std::abs(
                output.trajectory.snapshots[
                    static_cast<std::size_t>(output.late_last_snapshot)]
                    .residual[active]) / beta;
        }
    }

    std::vector<int> ordering(static_cast<std::size_t>(active_size));
    std::iota(ordering.begin(), ordering.end(), 0);
    std::sort(ordering.begin(), ordering.end(), [&](int lhs, int rhs_index) {
        return output.points[static_cast<std::size_t>(lhs)].late_share
             > output.points[static_cast<std::size_t>(rhs_index)].late_share;
    });
    for (int rank = 0; rank < active_size; ++rank) {
        output.points[static_cast<std::size_t>(
            ordering[static_cast<std::size_t>(rank)])].late_rank = rank + 1;
    }

    for (const PointSlowDiagnostic& point : output.points) {
        const double share = point.late_share;
        output.edge_late_shares[static_cast<std::size_t>(
            point.geometry.edge)] += share;
        if (point.geometry.reentrant_distance_over_h <= 2.0)
            output.reentrant_2h_share += share;
        if (point.geometry.reentrant_distance_over_h <= 4.0)
            output.reentrant_4h_share += share;
        if (point.geometry.reentrant_distance_over_h <= 8.0)
            output.reentrant_8h_share += share;
        if (point.geometry.nearest_corner_distance_over_h <= 2.0)
            output.any_corner_2h_share += share;
        if (point.geometry.nearest_corner_distance_over_h <= 4.0)
            output.any_corner_4h_share += share;
        if (point.geometry.box_distance_over_h <= 4.0)
            output.box_4h_share += share;
        if (point.geometry.box_distance_over_h <= 8.0)
            output.box_8h_share += share;
        if (point.geometry.nearest_grid_line_phase <= 0.05)
            output.grid_line_phase_005_share += share;
        if (point.geometry.nearest_grid_line_phase <= 0.10)
            output.grid_line_phase_010_share += share;
    }
    for (int rank = 0; rank < std::min(10, active_size); ++rank) {
        const double share = output.points[static_cast<std::size_t>(
            ordering[static_cast<std::size_t>(rank)])].late_share;
        if (rank < 5)
            output.top_5_share += share;
        output.top_10_share += share;
    }
    return output;
}

void attach_baseline_comparison(std::vector<CaseSlowDiagnostic>& cases)
{
    if (cases.empty())
        return;
    const std::vector<PointSlowDiagnostic>& baseline = cases.front().points;
    for (CaseSlowDiagnostic& diagnostic : cases) {
        if (diagnostic.points.size() != baseline.size())
            throw std::runtime_error("slow-point cases have incompatible DOFs");
        for (std::size_t point = 0; point < diagnostic.points.size(); ++point) {
            PointSlowDiagnostic& value = diagnostic.points[point];
            value.baseline_late_share = baseline[point].late_share;
            value.late_share_excess =
                value.late_share - value.baseline_late_share;
            value.late_share_ratio =
                value.late_share / (value.baseline_late_share + 1.0e-18);
        }
    }
}

std::filesystem::path output_directory()
{
#ifdef KFBIM_APP_OUTPUT_DIR
    return KFBIM_APP_OUTPUT_DIR;
#else
    return "output";
#endif
}

void write_point_csv(
    const std::filesystem::path& path,
    const std::vector<CaseSlowDiagnostic>& cases)
{
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("failed to open slow-point CSV");
    out << std::setprecision(17)
        << "case_id,N,h,iterations,active_index,interface_q,edge,edge_parameter,"
           "x,y,nx,ny,nearest_corner,nearest_corner_distance_over_h,"
           "reentrant_distance_over_h,box_distance_over_h,phase_x,phase_y,"
           "nearest_grid_line_phase,nearest_normal_sample_grid_line_phase,"
           "normal_samples_near_grid_line_005,restrict_wrong_side_nodes,"
           "restrict_exact_owners,restrict_gap_owners,"
           "restrict_identified_gap_owners,restrict_unresolved_gap_owners,"
           "restrict_endpoint_owners,boundary_clamped_samples,"
           "outside_box_queries,restrict_correction_weight_l1,"
           "max_sample_correction_weight_l1,max_sample_weight_l1,"
           "initial_share,early_share,late_share,"
           "late_gain_log10,late_peak_share,penultimate_scaled_abs,"
           "baseline_late_share,late_share_excess,late_share_ratio,late_rank\n";
    for (const CaseSlowDiagnostic& diagnostic : cases) {
        for (const PointSlowDiagnostic& point : diagnostic.points) {
            const PointGeometryDiagnostic& geometry = point.geometry;
            out << diagnostic.case_id << ',' << diagnostic.n_grid << ','
                << diagnostic.h << ',' << diagnostic.iterations << ','
                << geometry.active_index << ',' << geometry.interface_q << ','
                << geometry.edge << ',' << geometry.edge_parameter << ','
                << geometry.point[0] << ',' << geometry.point[1] << ','
                << geometry.normal[0] << ',' << geometry.normal[1] << ','
                << geometry.nearest_corner << ','
                << geometry.nearest_corner_distance_over_h << ','
                << geometry.reentrant_distance_over_h << ','
                << geometry.box_distance_over_h << ','
                << geometry.phase_x << ',' << geometry.phase_y << ','
                << geometry.nearest_grid_line_phase << ','
                << geometry.nearest_normal_sample_grid_line_phase << ','
                << geometry.normal_samples_near_grid_line_005 << ','
                << geometry.restrict_wrong_side_nodes << ','
                << geometry.restrict_exact_owners << ','
                << geometry.restrict_gap_owners << ','
                << geometry.restrict_identified_gap_owners << ','
                << geometry.restrict_unresolved_gap_owners << ','
                << geometry.restrict_endpoint_owners << ','
                << geometry.boundary_clamped_samples << ','
                << geometry.outside_box_queries << ','
                << geometry.restrict_correction_weight_l1 << ','
                << geometry.max_sample_correction_weight_l1 << ','
                << geometry.max_sample_weight_l1 << ','
                << point.initial_share << ',' << point.early_share << ','
                << point.late_share << ',' << point.late_gain_log10 << ','
                << point.late_peak_share << ','
                << point.penultimate_scaled_abs << ','
                << point.baseline_late_share << ','
                << point.late_share_excess << ','
                << point.late_share_ratio << ',' << point.late_rank << '\n';
        }
    }
}

void write_case_csv(
    const std::filesystem::path& path,
    const std::vector<CaseSlowDiagnostic>& cases)
{
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("failed to open slow-point case CSV");
    out << std::setprecision(17)
        << "case_id,N,h,iterations,final_relative,arnoldi_true_difference,"
           "late_first_iteration,late_last_iteration,late_border_energy_share,"
           "edge0_share,edge1_share,edge2_share,edge3_share,edge4_share,"
           "edge5_share,reentrant_2h_share,reentrant_4h_share,"
           "reentrant_8h_share,any_corner_2h_share,any_corner_4h_share,"
           "box_4h_share,box_8h_share,grid_line_phase_005_share,"
           "grid_line_phase_010_share,top_5_share,top_10_share\n";
    for (const CaseSlowDiagnostic& diagnostic : cases) {
        out << diagnostic.case_id << ',' << diagnostic.n_grid << ','
            << diagnostic.h << ',' << diagnostic.iterations << ','
            << diagnostic.final_relative << ','
            << diagnostic.arnoldi_true_difference << ','
            << diagnostic.late_first_snapshot << ','
            << diagnostic.late_last_snapshot << ','
            << diagnostic.late_border_energy_share;
        for (double share : diagnostic.edge_late_shares)
            out << ',' << share;
        out << ',' << diagnostic.reentrant_2h_share
            << ',' << diagnostic.reentrant_4h_share
            << ',' << diagnostic.reentrant_8h_share
            << ',' << diagnostic.any_corner_2h_share
            << ',' << diagnostic.any_corner_4h_share
            << ',' << diagnostic.box_4h_share
            << ',' << diagnostic.box_8h_share
            << ',' << diagnostic.grid_line_phase_005_share
            << ',' << diagnostic.grid_line_phase_010_share
            << ',' << diagnostic.top_5_share
            << ',' << diagnostic.top_10_share << '\n';
    }
}

void write_trajectory_csv(
    const std::filesystem::path& path,
    const std::vector<CaseSlowDiagnostic>& cases)
{
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("failed to open slow-point trajectory CSV");
    out << std::setprecision(17)
        << "case_id,iteration,relative_norm,active_index,interface_q,edge,"
           "edge_parameter,x,y,residual,abs_residual,interface_energy_share,"
           "border_residual\n";
    for (const CaseSlowDiagnostic& diagnostic : cases) {
        const int active_size = static_cast<int>(diagnostic.points.size());
        for (const ResidualSnapshot& snapshot : diagnostic.trajectory.snapshots) {
            const double interface_energy =
                snapshot.residual.head(active_size).squaredNorm();
            for (const PointSlowDiagnostic& point : diagnostic.points) {
                const int active = point.geometry.active_index;
                const double residual = snapshot.residual[active];
                out << diagnostic.case_id << ',' << snapshot.iteration << ','
                    << snapshot.relative_norm << ',' << active << ','
                    << point.geometry.interface_q << ',' << point.geometry.edge
                    << ',' << point.geometry.edge_parameter << ','
                    << point.geometry.point[0] << ',' << point.geometry.point[1]
                    << ',' << residual << ',' << std::abs(residual) << ','
                    << (interface_energy > 0.0
                            ? residual * residual / interface_energy
                            : 0.0)
                    << ',' << snapshot.residual[active_size] << '\n';
            }
        }
    }
}

void print_top_points(const CaseSlowDiagnostic& diagnostic)
{
    std::vector<const PointSlowDiagnostic*> points;
    points.reserve(diagnostic.points.size());
    for (const PointSlowDiagnostic& point : diagnostic.points)
        points.push_back(&point);
    std::sort(points.begin(), points.end(), [](const auto* lhs, const auto* rhs) {
        return lhs->late_share > rhs->late_share;
    });
    std::cout << diagnostic.case_id << ": iterations="
              << diagnostic.iterations << " late iter "
              << diagnostic.late_first_snapshot << '-'
              << diagnostic.late_last_snapshot
              << " border share=" << diagnostic.late_border_energy_share
              << " top points\n";
    for (int rank = 0; rank < std::min<int>(8, points.size()); ++rank) {
        const PointSlowDiagnostic& point = *points[static_cast<std::size_t>(rank)];
        std::cout << "  " << rank + 1 << ": a="
                  << point.geometry.active_index << " edge="
                  << point.geometry.edge << " tau="
                  << point.geometry.edge_parameter << " x=("
                  << point.geometry.point[0] << ',' << point.geometry.point[1]
                  << ") share=" << point.late_share << " gain="
                  << point.late_gain_log10 << " d_re/h="
                  << point.geometry.reentrant_distance_over_h << " d_box/h="
                  << point.geometry.box_distance_over_h << " phase=("
                  << point.geometry.phase_x << ',' << point.geometry.phase_y
                  << ")\n";
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const int n_grid = argc > 1 ? std::stoi(argv[1]) : 60;
        std::vector<CaseSlowDiagnostic> diagnostics;
        for (const RigidStudyCase2D& study_case : rigid_study_cases()) {
            std::cout << "running " << study_case.id << " N=" << n_grid
                      << std::endl;
            diagnostics.push_back(
                run_case_slow_diagnostic(n_grid, study_case));
        }
        attach_baseline_comparison(diagnostics);
        for (const CaseSlowDiagnostic& diagnostic : diagnostics)
            print_top_points(diagnostic);

        const std::filesystem::path directory = output_directory();
        std::filesystem::create_directories(directory);
        const std::string suffix = "_n" + std::to_string(n_grid) + ".csv";
        const std::filesystem::path point_path =
            directory / ("lshape_gmres_slow_points" + suffix);
        const std::filesystem::path case_path =
            directory / ("lshape_gmres_slow_point_cases" + suffix);
        const std::filesystem::path trajectory_path =
            directory / ("lshape_gmres_slow_point_trajectory" + suffix);
        write_point_csv(point_path, diagnostics);
        write_case_csv(case_path, diagnostics);
        write_trajectory_csv(trajectory_path, diagnostics);
        std::cout << "points: " << point_path.string() << '\n'
                  << "cases: " << case_path.string() << '\n'
                  << "trajectory: " << trajectory_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
