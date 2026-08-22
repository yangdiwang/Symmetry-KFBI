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

#include "src/grid/cartesian_grid_2d.hpp"
#include "src/geometry/nurbs_boundary_2d.hpp"
#include "src/interface/interface_2d.hpp"
#include "src/operators/laplace_neumann_exterior_trace_2d.hpp"
#include "src/transfer/laplace_arc_length_bspline_crossing_jet_2d.hpp"

using namespace kfbim;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct LShapeInterfaceData {
    Interface2D iface;
    std::vector<int> active_points;
};

struct RigidStudyCase2D {
    std::string id;
    double angle_degrees = 0.0;
    Eigen::Vector2d translation = Eigen::Vector2d::Zero();
    bool translation_in_grid_steps = false;
};

struct RigidTransform2D {
    Eigen::Matrix2d rotation = Eigen::Matrix2d::Identity();
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    Eigen::Vector2d translation = Eigen::Vector2d::Zero();

    Eigen::Vector2d forward_point(Eigen::Vector2d point) const
    {
        return center + rotation * (point - center) + translation;
    }

    Eigen::Vector2d inverse_point(Eigen::Vector2d point) const
    {
        return center
             + rotation.transpose() * (point - center - translation);
    }

    Eigen::Vector2d forward_vector(Eigen::Vector2d vector) const
    {
        return rotation * vector;
    }
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
    int restrict_wrong_side_nodes = 0;
    int restrict_corrected_nodes = 0;
    int exact_crossing_owners = 0;
    int gap_fallback_owners = 0;
    int identified_gap_crossing_owners = 0;
    int unresolved_gap_fallback_owners = 0;
    int endpoint_fallback_owners = 0;
    int virtual_side_flips = 0;
    int interior_sample_flips = 0;
    int exterior_sample_flips = 0;
    int shifted_stencils = 0;
    int center_cauchy_jump_samples = 0;
    int trace_points_without_incident_center = 0;
    int six_point_cross_stencils = 0;
    int diagonal_zero_crossing_rejections = 0;
    int diagonal_multiple_crossing_rejections = 0;
    int shared_side_spatial_polynomials = 0;
    int shared_side_spatial_polynomial_samples = 0;
    int unified_p2_stencils = 0;
    int unified_p2_relocated_stencils = 0;
    int unified_p2_candidate_rejections = 0;
    int expanded_same_side_center_stencils = 0;
    double max_same_side_center_distance_over_h = 0.0;
    double max_six_point_weight_l1 = 0.0;
    int domain_label_mismatches = 0;
    int als_periodic_components = 0;
    int als_open_branches = 0;
};

struct RigidStudyRow2D {
    std::string case_id;
    double angle_degrees = 0.0;
    Eigen::Vector2d rotation_center = Eigen::Vector2d::Zero();
    Eigen::Vector2d translation = Eigen::Vector2d::Zero();
    LevelMetrics metrics;
    double trace_order = std::numeric_limits<double>::quiet_NaN();
    double normal_order = std::numeric_limits<double>::quiet_NaN();
    double bulk_order = std::numeric_limits<double>::quiet_NaN();
    double baseline_trace_ratio = std::numeric_limits<double>::quiet_NaN();
    double baseline_normal_ratio = std::numeric_limits<double>::quiet_NaN();
    double baseline_bulk_ratio = std::numeric_limits<double>::quiet_NaN();
    double baseline_iteration_ratio = std::numeric_limits<double>::quiet_NaN();
};

Eigen::Vector2d normalized_or_fallback(Eigen::Vector2d value)
{
    const double length = value.norm();
    if (length <= 1.0e-14)
        return {1.0, 0.0};
    return value / length;
}

int p2_midpoint_count(double length, double target_spacing)
{
    if (!(length > 0.0) || !(target_spacing > 0.0))
        throw std::invalid_argument(
            "P2 midpoint count requires positive length and spacing");

    int count = std::max(
        3, static_cast<int>(std::llround(length / target_spacing)));
    if (count % 2 == 0) {
        const int lower = std::max(3, count - 1);
        const int upper = count + 1;
        const double lower_error =
            std::abs(length / static_cast<double>(lower) - target_spacing);
        const double upper_error =
            std::abs(length / static_cast<double>(upper) - target_spacing);
        // Prefer the denser discretization when the spacing errors tie.
        count = upper_error <= lower_error ? upper : lower;
    }
    return count;
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

RigidTransform2D make_rigid_transform(const RigidStudyCase2D& study_case,
                                      double h)
{
    const double angle = study_case.angle_degrees * kPi / 180.0;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    RigidTransform2D transform;
    transform.rotation << cosine, -sine,
                          sine, cosine;
    const std::vector<Eigen::Vector2d> vertices =
        make_half_grid_l_shape_vertices(h);
    transform.center = vertices[3];
    transform.translation =
        study_case.translation_in_grid_steps
            ? h * study_case.translation
            : study_case.translation;
    return transform;
}

std::vector<Eigen::Vector2d> transformed_l_shape_vertices(
    double h,
    const RigidTransform2D& transform)
{
    std::vector<Eigen::Vector2d> vertices =
        make_half_grid_l_shape_vertices(h);
    for (Eigen::Vector2d& vertex : vertices)
        vertex = transform.forward_point(vertex);
    return vertices;
}

LShapeInterfaceData make_l_shape_interface(
    double h,
    const RigidTransform2D& transform)
{
    const std::vector<Eigen::Vector2d> vertices =
        transformed_l_shape_vertices(h, transform);
    const int n_edges = static_cast<int>(vertices.size());

    std::vector<double> edge_lengths(static_cast<std::size_t>(n_edges));
    double perimeter = 0.0;
    for (int edge = 0; edge < n_edges; ++edge) {
        edge_lengths[static_cast<std::size_t>(edge)] =
            (vertices[static_cast<std::size_t>((edge + 1) % n_edges)]
             - vertices[static_cast<std::size_t>(edge)])
                .norm();
        perimeter += edge_lengths[static_cast<std::size_t>(edge)];
    }
    std::vector<double> span_breaks(
        static_cast<std::size_t>(n_edges + 1), 0.0);
    for (int edge = 0; edge < n_edges; ++edge) {
        span_breaks[static_cast<std::size_t>(edge + 1)] =
            span_breaks[static_cast<std::size_t>(edge)]
            + edge_lengths[static_cast<std::size_t>(edge)] / perimeter;
    }
    span_breaks.back() = 1.0;

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
    std::vector<int> point_spans;
    std::vector<double> point_parameters;

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
        point_spans.push_back(edge);
        point_parameters.push_back(
            span_breaks[static_cast<std::size_t>(edge)]);
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
        const int count = p2_midpoint_count(length, h);
        const double spacing = length / static_cast<double>(count);
        for (int k = 0; k < count; ++k) {
            const double s =
                (static_cast<double>(k) + 0.5) * spacing;
            const int q = static_cast<int>(points.size());
            points.push_back(start + s * tangent);
            normals.push_back(normal);
            weights.push_back(spacing);
            point_spans.push_back(edge);
            point_parameters.push_back(
                span_breaks[static_cast<std::size_t>(edge)]
                + (s / length)
                    * (span_breaks[static_cast<std::size_t>(edge + 1)]
                       - span_breaks[static_cast<std::size_t>(edge)]));
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
        if ((ids.size() - 1) % 2 != 0) {
            throw std::logic_error(
                "L-shape edge midpoint count must be odd for P2 panels");
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
        if (last_covered != static_cast<int>(ids.size()) - 1) {
            throw std::logic_error(
                "L-shape P2 panels do not cover every edge midpoint");
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

    // One closed degree-one NURBS curve is the authoritative L boundary.
    // Its six nonzero knot spans are the six exact line segments; the P2 rows
    // above are only compatibility cells for GridPair2D and never define the
    // geometry queried by crossing/spread/restrict.
    geometry2d::NurbsCurve2D::ControlPointVector control_points;
    control_points.reserve(static_cast<std::size_t>(n_edges + 1));
    for (const Eigen::Vector2d& vertex : vertices)
        control_points.push_back(vertex);
    control_points.push_back(vertices.front());
    std::vector<double> knots;
    knots.reserve(static_cast<std::size_t>(n_edges + 3));
    knots.push_back(0.0);
    knots.push_back(0.0);
    for (int edge = 1; edge < n_edges; ++edge)
        knots.push_back(span_breaks[static_cast<std::size_t>(edge)]);
    knots.push_back(1.0);
    knots.push_back(1.0);
    geometry2d::NurbsCurve2D nurbs_curve(
        geometry::NurbsBasis1D(1, std::move(knots)),
        std::move(control_points),
        std::vector<double>(static_cast<std::size_t>(n_edges + 1), 1.0));
    std::vector<std::array<double, 2>> panel_parameters;
    panel_parameters.reserve(panel_rows.size());
    for (const std::array<int, 3>& row : panel_rows) {
        panel_parameters.push_back(
            {point_parameters[static_cast<std::size_t>(row[0])],
             point_parameters[static_cast<std::size_t>(row[2])]});
    }
    std::shared_ptr<const IPanelGeometry2D> nurbs_geometry =
        std::make_shared<geometry2d::NurbsBoundaryPanelGeometry2D>(
            std::move(nurbs_curve),
            std::move(span_breaks),
            panel_edge_ids,
            std::move(panel_parameters),
            std::move(point_spans),
            std::move(point_parameters),
            true,
            true);

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
                      PanelNodeLayout2D::QuadraticLagrange,
                      {},
                      std::move(nurbs_geometry));
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

double transformed_exact_u(Eigen::Vector2d point,
                           const RigidTransform2D& transform)
{
    const Eigen::Vector2d local = transform.inverse_point(point);
    return exact_u(local[0], local[1]);
}

Eigen::Vector2d transformed_exact_gradient(
    Eigen::Vector2d point,
    const RigidTransform2D& transform)
{
    const Eigen::Vector2d local = transform.inverse_point(point);
    return transform.forward_vector(exact_gradient(local[0], local[1]));
}

double distance_to_l_shape_boundary(
    Eigen::Vector2d point,
    const std::vector<Eigen::Vector2d>& vertices)
{
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

bool reference_l_shape_contains(Eigen::Vector2d point, double h)
{
    const std::vector<Eigen::Vector2d> vertices =
        make_half_grid_l_shape_vertices(h);
    const double a = vertices[0][0];
    const double b = vertices[3][0];
    const double c = vertices[1][0];
    return point[0] >= a && point[0] <= c
        && point[1] >= a && point[1] <= c
        && (point[0] <= b || point[1] <= b);
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

LaplaceCrossingJetScheme2D selected_crossing_jet_scheme()
{
    const char* raw = std::getenv("KFBIM_LSHAPE_CROSSING_JET");
    if (raw == nullptr || std::string(raw).empty()
        || std::string(raw) == "nurbs_same_parameter_crossing_jet"
        || std::string(raw) == "nsp_cj"
        || std::string(raw) == "nurbs") {
        return LaplaceCrossingJetScheme2D::
            NurbsSameParameterCrossingJet;
    }
    if (std::string(raw) == "arc_length_bspline_crossing_jet"
        || std::string(raw) == "als_cj") {
        return LaplaceCrossingJetScheme2D::
            ArcLengthBSplineCrossingJet;
    }
    if (std::string(raw) == "local_arclength_lagrange"
        || std::string(raw) == "local") {
        return LaplaceCrossingJetScheme2D::LocalArclengthLagrange;
    }
    throw std::invalid_argument(
        "KFBIM_LSHAPE_CROSSING_JET must be nsp_cj, als_cj, or local");
}

LaplaceNeumannExteriorRestrictMethod2D selected_restrict_method()
{
    const char* raw = std::getenv("KFBIM_LSHAPE_RESTRICT");
    if (raw == nullptr || std::string(raw).empty()
        || std::string(raw) == "joint_bicubic_cubic_crossing_owner"
        || std::string(raw) == "crossing_owner") {
        return LaplaceNeumannExteriorRestrictMethod2D::
            JointBicubicCubicCrossingOwner;
    }
    if (std::string(raw)
            == "joint_biquadratic_quadratic_crossing_owner"
        || std::string(raw) == "crossing_owner_quadratic"
        || std::string(raw) == "joint_quadratic") {
        return LaplaceNeumannExteriorRestrictMethod2D::
            JointBiquadraticQuadraticCrossingOwner;
    }
    if (std::string(raw)
            == "joint_biquadratic_quadratic_center_cauchy_jump"
        || std::string(raw) == "center_cauchy_jump"
        || std::string(raw) == "panel_center_cauchy_jump") {
        return LaplaceNeumannExteriorRestrictMethod2D::
            JointBiquadraticQuadraticCenterCauchyJump;
    }
    if (std::string(raw)
            == "joint_six_point_quadratic_center_cauchy_jump"
        || std::string(raw) == "six_point_cross_stencil") {
        return LaplaceNeumannExteriorRestrictMethod2D::
            JointSixPointQuadraticCenterCauchyJump;
    }
    if (std::string(raw)
            == "joint_biquadratic_quadratic_virtual_side_flip"
        || std::string(raw) == "quadratic_virtual_side_flip"
        || std::string(raw) == "virtual_side_flip") {
        return LaplaceNeumannExteriorRestrictMethod2D::
            JointBiquadraticQuadraticVirtualSideFlip;
    }
    if (std::string(raw)
            == "unified_spatial_normal_p2_dof_cauchy_exterior"
        || std::string(raw) == "usn_p2_dof_ext") {
        return LaplaceNeumannExteriorRestrictMethod2D::
            UnifiedSpatialNormalP2DofCauchyExterior;
    }
    if (std::string(raw)
            == "unified_spatial_normal_p2_crossing_owner"
        || std::string(raw) == "usn_p2") {
        return LaplaceNeumannExteriorRestrictMethod2D::
            UnifiedSpatialNormalP2CrossingOwner;
    }
    if (std::string(raw) == "six_point_quadratic"
        || std::string(raw) == "legacy") {
        return LaplaceNeumannExteriorRestrictMethod2D::SixPointQuadratic;
    }
    throw std::invalid_argument(
        "KFBIM_LSHAPE_RESTRICT must be "
        "joint_bicubic_cubic_crossing_owner (or crossing_owner), "
        "joint_biquadratic_quadratic_crossing_owner "
        "(or crossing_owner_quadratic), "
        "joint_biquadratic_quadratic_center_cauchy_jump "
        "(or center_cauchy_jump), "
        "joint_six_point_quadratic_center_cauchy_jump "
        "(or six_point_cross_stencil), "
        "joint_biquadratic_quadratic_virtual_side_flip "
        "(or virtual_side_flip), or "
        "unified_spatial_normal_p2_dof_cauchy_exterior "
        "(or usn_p2_dof_ext), or "
        "unified_spatial_normal_p2_crossing_owner "
        "(or usn_p2), or "
        "six_point_quadratic (or legacy)");
}

const char* restrict_method_name(
    LaplaceNeumannExteriorRestrictMethod2D method)
{
    switch (method) {
    case LaplaceNeumannExteriorRestrictMethod2D::SixPointQuadratic:
        return "six_point_quadratic";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointBiquadraticQuadraticCrossingOwner:
        return "joint_biquadratic_quadratic_crossing_owner";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointBiquadraticQuadraticCenterCauchyJump:
        return "joint_biquadratic_quadratic_center_cauchy_jump";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointSixPointQuadraticCenterCauchyJump:
        return "joint_six_point_quadratic_center_cauchy_jump";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointBiquadraticQuadraticVirtualSideFlip:
        return "joint_biquadratic_quadratic_virtual_side_flip";
    case LaplaceNeumannExteriorRestrictMethod2D::
             UnifiedSpatialNormalP2DofCauchyExterior:
        return "unified_spatial_normal_p2_dof_cauchy_exterior";
    case LaplaceNeumannExteriorRestrictMethod2D::
             UnifiedSpatialNormalP2CrossingOwner:
        return "unified_spatial_normal_p2_crossing_owner";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointBicubicCubicCrossingOwner:
        return "joint_bicubic_cubic_crossing_owner";
    }
    throw std::runtime_error("unknown L-shape restrict method");
}

const char* restrict_output_tag(
    LaplaceNeumannExteriorRestrictMethod2D method)
{
    switch (method) {
    case LaplaceNeumannExteriorRestrictMethod2D::SixPointQuadratic:
        return "six_point_quadratic";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointBiquadraticQuadraticCrossingOwner:
        return "p2_crossing_owner_joint_quadratic";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointBiquadraticQuadraticCenterCauchyJump:
        return "p2_joint_quadratic_center_cauchy_jump";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointSixPointQuadraticCenterCauchyJump:
        return "p2_joint_six_point_quadratic_center_cauchy_jump";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointBiquadraticQuadraticVirtualSideFlip:
        return "p2_joint_quadratic_virtual_side_flip";
    case LaplaceNeumannExteriorRestrictMethod2D::
             UnifiedSpatialNormalP2DofCauchyExterior:
        return "usn_p2_dof_ext";
    case LaplaceNeumannExteriorRestrictMethod2D::
             UnifiedSpatialNormalP2CrossingOwner:
        return "usn_p2";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointBicubicCubicCrossingOwner:
        return "p2_crossing_owner_joint_cubic";
    }
    throw std::runtime_error("unknown L-shape restrict output tag");
}

const char* crossing_jet_output_tag(LaplaceCrossingJetScheme2D scheme)
{
    switch (scheme) {
    case LaplaceCrossingJetScheme2D::LocalArclengthLagrange:
        return "local_lagrange";
    case LaplaceCrossingJetScheme2D::ArcLengthBSplineCrossingJet:
        return "als_cj";
    case LaplaceCrossingJetScheme2D::NurbsSameParameterCrossingJet:
        return "nsp_cj";
    }
    throw std::runtime_error("unknown L-shape crossing-jet output tag");
}

const RigidStudyCase2D& baseline_rigid_case()
{
    static const RigidStudyCase2D baseline{
        "baseline", 0.0, Eigen::Vector2d::Zero(), false};
    return baseline;
}

std::vector<RigidStudyCase2D> rigid_study_cases()
{
    return {
        baseline_rigid_case(),
        {"translate_2h_m1h",
         0.0,
         Eigen::Vector2d(2.0, -1.0),
         true},
        {"translate_xy",
         0.0,
         Eigen::Vector2d(0.137, -0.083),
         false},
        {"rotate_90deg",
         90.0,
         Eigen::Vector2d::Zero(),
         false},
        {"rotate_17deg",
         17.0,
         Eigen::Vector2d::Zero(),
         false},
        {"rotate_17deg_translate_xy",
         17.0,
         Eigen::Vector2d(0.137, -0.083),
         false}
    };
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

LevelMetrics run_level(
    int n,
    int max_iter,
    double tolerance,
    int restart,
    LaplaceNeumannExteriorRestrictMethod2D restrict_method,
    const RigidStudyCase2D& rigid_case)
{
    if (n < 12)
        throw std::invalid_argument("L-shape grid size must be at least 12");
    const double h = 3.0 / static_cast<double>(n);
    CartesianGrid2D grid({-1.5, -1.5},
                         {h, h},
                         {n, n},
                         DofLayout2D::Node);
    const RigidTransform2D transform =
        make_rigid_transform(rigid_case, h);
    const std::vector<Eigen::Vector2d> transformed_vertices =
        transformed_l_shape_vertices(h, transform);
    LShapeInterfaceData geometry =
        make_l_shape_interface(h, transform);
    const LaplaceCrossingJetScheme2D crossing_jet_scheme =
        selected_crossing_jet_scheme();
    const LaplaceArcLengthBSplineCrossingJetPlan2D spline_plan(
        geometry.iface, crossing_jet_scheme);
    if (crossing_jet_scheme
            != LaplaceCrossingJetScheme2D::LocalArclengthLagrange
        && (spline_plan.periodic_component_count() != 0
            || spline_plan.open_branch_count() != 6)) {
        throw std::runtime_error(
            "L-shape crossing-jet preprocessing did not produce six independent one-sided spans");
    }
    if (crossing_jet_scheme
            == LaplaceCrossingJetScheme2D::NurbsSameParameterCrossingJet
        && (!spline_plan.uses_nurbs_same_parameter()
            || spline_plan.nurbs_span_count() != 6)) {
        throw std::runtime_error(
            "L-shape NSP-CJ preprocessing did not bind all six NURBS spans");
    }

    LaplaceNeumannExteriorTraceOptions2D options;
    options.active_interface_points = geometry.active_points;
    options.restrict_method = restrict_method;
    options.crossing_jet_scheme = crossing_jet_scheme;
    if (restrict_method
        != LaplaceNeumannExteriorRestrictMethod2D::
               SixPointQuadratic) {
        options.correction_method =
            LaplaceCorrectionMethod2D::CrossingOwner;
    }
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
        const Eigen::Vector2d point(x, y);
        exact_trace_raw[a] =
            transformed_exact_u(point, transform);
        g[a] =
            transformed_exact_gradient(point, transform).dot(normal);
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
    int domain_label_mismatches = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        const std::array<double, 2> coordinate = grid.coord(node);
        const Eigen::Vector2d point(coordinate[0], coordinate[1]);
        const bool expected_inside =
            reference_l_shape_contains(
                transform.inverse_point(point), h);
        const bool labeled_inside =
            solver.grid_pair().domain_label(node) > 0;
        if (expected_inside != labeled_inside
            && distance_to_l_shape_boundary(
                   point, transformed_vertices) > 1.0e-12) {
            ++domain_label_mismatches;
        }
        if (solver.grid_pair().domain_label(node) > 0) {
            const double error = std::abs(
                result.u_bulk[node]
                - (transformed_exact_u(
                       {coordinate[0], coordinate[1]}, transform)
                   - exact_trace_mean));
            interior_sum_sq += error * error;
            ++interior_count;
            if (error > interior_inf) {
                interior_inf = error;
                max_x = coordinate[0];
                max_y = coordinate[1];
            }
            if (distance_to_l_shape_boundary(
                    {coordinate[0], coordinate[1]},
                    transformed_vertices) >= 4.0 * h) {
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
    metrics.domain_label_mismatches = domain_label_mismatches;
    metrics.als_periodic_components =
        spline_plan.periodic_component_count();
    metrics.als_open_branches = spline_plan.open_branch_count();
    if (const auto* diagnostics =
            solver.joint_polynomial_restrict_diagnostics()) {
        metrics.restrict_wrong_side_nodes =
            diagnostics->wrong_side_nodes;
        metrics.restrict_corrected_nodes =
            diagnostics->corrected_nodes;
        metrics.exact_crossing_owners =
            diagnostics->exact_crossing_owners;
        metrics.gap_fallback_owners =
            diagnostics->gap_fallback_owners;
        metrics.identified_gap_crossing_owners =
            diagnostics->identified_gap_crossing_owners;
        metrics.unresolved_gap_fallback_owners =
            diagnostics->unresolved_gap_fallback_owners;
        metrics.endpoint_fallback_owners =
            diagnostics->endpoint_fallback_owners;
        metrics.virtual_side_flips =
            diagnostics->virtual_side_flips;
        metrics.interior_sample_flips =
            diagnostics->interior_sample_flips;
        metrics.exterior_sample_flips =
            diagnostics->exterior_sample_flips;
        metrics.shifted_stencils =
            diagnostics->shifted_stencils;
        metrics.center_cauchy_jump_samples =
            diagnostics->center_cauchy_jump_samples;
        metrics.trace_points_without_incident_center =
            diagnostics->trace_points_without_incident_center;
        metrics.six_point_cross_stencils =
            diagnostics->six_point_cross_stencils;
        metrics.diagonal_zero_crossing_rejections =
            diagnostics->diagonal_zero_crossing_rejections;
        metrics.diagonal_multiple_crossing_rejections =
            diagnostics->diagonal_multiple_crossing_rejections;
        metrics.shared_side_spatial_polynomials =
            diagnostics->shared_side_spatial_polynomials;
        metrics.shared_side_spatial_polynomial_samples =
            diagnostics->shared_side_spatial_polynomial_samples;
        metrics.unified_p2_stencils =
            diagnostics->unified_p2_stencils;
        metrics.unified_p2_relocated_stencils =
            diagnostics->unified_p2_relocated_stencils;
        metrics.unified_p2_candidate_rejections =
            diagnostics->unified_p2_candidate_rejections;
        metrics.expanded_same_side_center_stencils =
            diagnostics->expanded_same_side_center_stencils;
        metrics.max_same_side_center_distance_over_h =
            diagnostics->max_same_side_center_distance_over_h;
        metrics.max_six_point_weight_l1 =
            diagnostics->max_six_point_weight_l1;
        const bool uses_center_cauchy_jump =
            restrict_method
                == LaplaceNeumannExteriorRestrictMethod2D::
                       JointBiquadraticQuadraticCenterCauchyJump
            || restrict_method
                == LaplaceNeumannExteriorRestrictMethod2D::
                       JointSixPointQuadraticCenterCauchyJump;
        if (uses_center_cauchy_jump) {
            const int expected_samples = 6 * n_active;
            const int expected_metadata_points =
                geometry.iface.num_points() - n_active;
            if (metrics.center_cauchy_jump_samples
                    != expected_samples
                || metrics.trace_points_without_incident_center
                       != expected_metadata_points) {
                throw std::runtime_error(
                    "center-Cauchy jump restrict did not bind every active "
                    "P2 trace point to an incident panel center");
            }
            if (restrict_method
                    == LaplaceNeumannExteriorRestrictMethod2D::
                           JointSixPointQuadraticCenterCauchyJump
                && metrics.six_point_cross_stencils
                       != expected_samples) {
                throw std::runtime_error(
                    "six-point quadratic restrict did not build one cross "
                    "stencil for every active P2 trace sample");
            }
        }
        const bool uses_unified_p2 =
            restrict_method
                == LaplaceNeumannExteriorRestrictMethod2D::
                       UnifiedSpatialNormalP2CrossingOwner
            || restrict_method
                == LaplaceNeumannExteriorRestrictMethod2D::
                       UnifiedSpatialNormalP2DofCauchyExterior;
        if (uses_unified_p2) {
            if (metrics.unified_p2_stencils != 2 * n_active
                || metrics.shared_side_spatial_polynomials
                       != 2 * n_active
                || metrics.shared_side_spatial_polynomial_samples
                       != 6 * n_active
                || metrics.six_point_cross_stencils != 6 * n_active) {
                throw std::runtime_error(
                    "USN-P2 restrict did not build one shared spatial P2 per active trace side");
            }
            if (metrics.gap_fallback_owners != 0
                || metrics.endpoint_fallback_owners != 0) {
                throw std::runtime_error(
                    "USN-P2 restrict used a forbidden crossing fallback");
            }
        }
    }
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
               "interior_core_inf,exterior_bulk_inf,max_error_x,max_error_y,"
               "restrict_wrong_side_nodes,restrict_corrected_nodes,"
               "exact_crossing_owners,"
               "gap_fallback_owners,identified_gap_crossing_owners,"
               "unresolved_gap_fallback_owners,endpoint_fallback_owners,"
               "virtual_side_flips,interior_sample_flips,"
               "exterior_sample_flips,shifted_stencils,"
               "center_cauchy_jump_samples,"
               "trace_points_without_incident_center,"
               "six_point_cross_stencils,"
               "diagonal_zero_crossing_rejections,"
               "diagonal_multiple_crossing_rejections,"
               "shared_side_spatial_polynomials,"
               "shared_side_spatial_polynomial_samples,"
               "unified_p2_stencils,unified_p2_relocated_stencils,"
               "unified_p2_candidate_rejections,"
               "expanded_same_side_center_stencils,"
               "max_same_side_center_distance_over_h,"
               "max_six_point_weight_l1,"
               "domain_label_mismatches,als_periodic_components,"
               "als_open_branches\n";
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
               << ',' << level.restrict_wrong_side_nodes
               << ',' << level.restrict_corrected_nodes
               << ',' << level.exact_crossing_owners
               << ',' << level.gap_fallback_owners
               << ',' << level.identified_gap_crossing_owners
               << ',' << level.unresolved_gap_fallback_owners
               << ',' << level.endpoint_fallback_owners
               << ',' << level.virtual_side_flips
               << ',' << level.interior_sample_flips
               << ',' << level.exterior_sample_flips
               << ',' << level.shifted_stencils
               << ',' << level.center_cauchy_jump_samples
               << ',' << level.trace_points_without_incident_center
               << ',' << level.six_point_cross_stencils
               << ',' << level.diagonal_zero_crossing_rejections
               << ',' << level.diagonal_multiple_crossing_rejections
               << ',' << level.shared_side_spatial_polynomials
               << ',' << level.shared_side_spatial_polynomial_samples
               << ',' << level.unified_p2_stencils
               << ',' << level.unified_p2_relocated_stencils
               << ',' << level.unified_p2_candidate_rejections
               << ',' << level.expanded_same_side_center_stencils
               << ',' << level.max_same_side_center_distance_over_h
               << ',' << level.max_six_point_weight_l1
               << ',' << level.domain_label_mismatches
               << ',' << level.als_periodic_components
               << ',' << level.als_open_branches
               << '\n';
    }
}

double error_ratio(double value, double baseline)
{
    if (!(baseline > 0.0))
        return std::numeric_limits<double>::quiet_NaN();
    return value / baseline;
}

void write_rigid_study_csv(
    const std::filesystem::path& path,
    LaplaceNeumannExteriorRestrictMethod2D restrict_method,
    const std::vector<RigidStudyRow2D>& rows)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error(
            "cannot open rigid-study CSV: " + path.string());

    output
        << "case_id,restrict_method,N,h,angle_degrees,"
           "rotation_center_x,rotation_center_y,translation_x,translation_y,"
           "rotation_00,rotation_01,rotation_10,rotation_11,"
           "active,iterations,augmented_converged,physical_converged,"
           "elapsed_sec,trace_error_inf,trace_order,baseline_trace_ratio,"
           "normal_error_inf,normal_order,baseline_normal_ratio,"
           "interior_bulk_inf,bulk_order,baseline_bulk_ratio,"
           "baseline_iteration_ratio,interior_bulk_rms,interior_core_inf,"
           "exterior_bulk_inf,compatibility_shift,constant_mode_inf,"
           "lambda,gauge,augmented_rel,physical_rel,exterior_trace_rel,"
           "physical_inf,exterior_trace_inf,trace_closure_inf,"
           "superposition_bulk_inf,max_error_x,max_error_y,"
           "restrict_wrong_side_nodes,restrict_corrected_nodes,"
           "exact_crossing_owners,"
           "gap_fallback_owners,identified_gap_crossing_owners,"
           "unresolved_gap_fallback_owners,endpoint_fallback_owners,"
           "virtual_side_flips,interior_sample_flips,"
           "exterior_sample_flips,shifted_stencils,"
           "center_cauchy_jump_samples,"
           "trace_points_without_incident_center,"
           "six_point_cross_stencils,"
           "diagonal_zero_crossing_rejections,"
           "diagonal_multiple_crossing_rejections,"
           "shared_side_spatial_polynomials,"
           "shared_side_spatial_polynomial_samples,"
           "unified_p2_stencils,unified_p2_relocated_stencils,"
           "unified_p2_candidate_rejections,"
           "expanded_same_side_center_stencils,"
           "max_same_side_center_distance_over_h,"
           "max_six_point_weight_l1,"
           "domain_label_mismatches,als_periodic_components,"
           "als_open_branches\n";
    output << std::setprecision(17);
    for (const RigidStudyRow2D& row : rows) {
        const double angle = row.angle_degrees * kPi / 180.0;
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        const LevelMetrics& level = row.metrics;
        output
            << row.case_id << ',' << restrict_method_name(restrict_method)
            << ',' << level.n << ',' << level.h << ','
            << row.angle_degrees << ','
            << row.rotation_center[0] << ','
            << row.rotation_center[1] << ','
            << row.translation[0] << ',' << row.translation[1] << ','
            << cosine << ',' << -sine << ',' << sine << ',' << cosine
            << ',' << level.active_points << ',' << level.iterations << ','
            << (level.augmented_converged ? 1 : 0) << ','
            << (level.physical_converged ? 1 : 0) << ','
            << level.elapsed_sec << ','
            << level.trace_error_inf << ',' << row.trace_order << ','
            << row.baseline_trace_ratio << ','
            << level.normal_error_inf << ',' << row.normal_order << ','
            << row.baseline_normal_ratio << ','
            << level.interior_bulk_inf << ',' << row.bulk_order << ','
            << row.baseline_bulk_ratio << ','
            << row.baseline_iteration_ratio << ','
            << level.interior_bulk_rms << ','
            << level.interior_core_inf << ','
            << level.exterior_bulk_inf << ','
            << level.compatibility_shift << ','
            << level.constant_mode_inf << ',' << level.lambda << ','
            << level.gauge << ',' << level.augmented_rel << ','
            << level.physical_rel << ',' << level.exterior_trace_rel << ','
            << level.physical_inf << ',' << level.exterior_trace_inf << ','
            << level.trace_closure_inf << ','
            << level.superposition_bulk_inf << ','
            << level.max_error_x << ',' << level.max_error_y << ','
            << level.restrict_wrong_side_nodes << ','
            << level.restrict_corrected_nodes << ','
            << level.exact_crossing_owners << ','
            << level.gap_fallback_owners << ','
            << level.identified_gap_crossing_owners << ','
            << level.unresolved_gap_fallback_owners << ','
            << level.endpoint_fallback_owners << ','
            << level.virtual_side_flips << ','
            << level.interior_sample_flips << ','
            << level.exterior_sample_flips << ','
            << level.shifted_stencils << ','
            << level.center_cauchy_jump_samples << ','
            << level.trace_points_without_incident_center << ','
            << level.six_point_cross_stencils << ','
            << level.diagonal_zero_crossing_rejections << ','
            << level.diagonal_multiple_crossing_rejections << ','
            << level.shared_side_spatial_polynomials << ','
            << level.shared_side_spatial_polynomial_samples << ','
            << level.unified_p2_stencils << ','
            << level.unified_p2_relocated_stencils << ','
            << level.unified_p2_candidate_rejections << ','
            << level.expanded_same_side_center_stencils << ','
            << level.max_same_side_center_distance_over_h << ','
            << level.max_six_point_weight_l1 << ','
            << level.domain_label_mismatches << ','
            << level.als_periodic_components << ','
            << level.als_open_branches << '\n';
    }
}

int run_rigid_study(
    const std::vector<int>& levels,
    int max_iter,
    double tolerance,
    int restart,
    LaplaceNeumannExteriorRestrictMethod2D restrict_method,
    const std::filesystem::path& output_dir)
{
    const std::vector<RigidStudyCase2D> cases = rigid_study_cases();
    std::vector<LevelMetrics> baseline_levels;
    std::vector<RigidStudyRow2D> rows;
    rows.reserve(cases.size() * levels.size());

    std::cout
        << "KFBI2D L-shape rigid-transform study\n"
        << "P2 restrict=" << restrict_method_name(restrict_method) << '\n'
        << "crossing jet="
        << laplace_crossing_jet_scheme_name_2d(
               selected_crossing_jet_scheme())
        << '\n'
        << "rotation center=level-dependent reentrant corner; "
           "controls: translation=(2h,-h), rotation=90deg\n"
        << "tol=" << tolerance << " max_iter=" << max_iter
        << " restart=" << restart << "\n\n"
        << std::left << std::setw(28) << "case"
        << std::right << std::setw(5) << "N"
        << std::setw(7) << "iter"
        << std::setw(5) << "ok"
        << std::setw(13) << "trace"
        << std::setw(8) << "p_t"
        << std::setw(13) << "normal"
        << std::setw(8) << "p_n"
        << std::setw(13) << "bulk"
        << std::setw(8) << "p_b"
        << std::setw(9) << "bulk/x"
        << std::setw(8) << "exact"
        << std::setw(7) << "gap"
        << std::setw(8) << "expand"
        << std::setw(8) << "d/h"
        << std::setw(8) << "L1"
        << std::setw(8) << "labels" << '\n';

    for (int case_index = 0;
         case_index < static_cast<int>(cases.size());
         ++case_index) {
        const RigidStudyCase2D& study_case =
            cases[static_cast<std::size_t>(case_index)];
        double previous_h = 0.0;
        double previous_trace = 0.0;
        double previous_normal = 0.0;
        double previous_bulk = 0.0;

        for (int level_index = 0;
             level_index < static_cast<int>(levels.size());
             ++level_index) {
            const int n = levels[static_cast<std::size_t>(level_index)];
            const LevelMetrics metrics =
                run_level(n,
                          max_iter,
                          tolerance,
                          restart,
                          restrict_method,
                          study_case);
            const RigidTransform2D transform =
                make_rigid_transform(study_case, metrics.h);

            RigidStudyRow2D row;
            row.case_id = study_case.id;
            row.angle_degrees = study_case.angle_degrees;
            row.rotation_center = transform.center;
            row.translation = transform.translation;
            row.metrics = metrics;
            row.trace_order = convergence_order(
                previous_trace,
                metrics.trace_error_inf,
                previous_h,
                metrics.h);
            row.normal_order = convergence_order(
                previous_normal,
                metrics.normal_error_inf,
                previous_h,
                metrics.h);
            row.bulk_order = convergence_order(
                previous_bulk,
                metrics.interior_bulk_inf,
                previous_h,
                metrics.h);

            if (case_index == 0) {
                baseline_levels.push_back(metrics);
                row.baseline_trace_ratio = 1.0;
                row.baseline_normal_ratio = 1.0;
                row.baseline_bulk_ratio = 1.0;
                row.baseline_iteration_ratio = 1.0;
            } else {
                const LevelMetrics& baseline =
                    baseline_levels[static_cast<std::size_t>(level_index)];
                row.baseline_trace_ratio = error_ratio(
                    metrics.trace_error_inf, baseline.trace_error_inf);
                row.baseline_normal_ratio = error_ratio(
                    metrics.normal_error_inf, baseline.normal_error_inf);
                row.baseline_bulk_ratio = error_ratio(
                    metrics.interior_bulk_inf,
                    baseline.interior_bulk_inf);
                row.baseline_iteration_ratio =
                    baseline.iterations > 0
                        ? static_cast<double>(metrics.iterations)
                            / static_cast<double>(baseline.iterations)
                        : std::numeric_limits<double>::quiet_NaN();
            }

            const bool converged =
                metrics.augmented_converged && metrics.physical_converged;
            const auto printable_order = [](double order) {
                return std::isfinite(order) ? order : 0.0;
            };
            std::cout
                << std::left << std::setw(28) << row.case_id
                << std::right << std::setw(5) << metrics.n
                << std::setw(7) << metrics.iterations
                << std::setw(5) << (converged ? "yes" : "no")
                << std::scientific << std::setprecision(3)
                << std::setw(13) << metrics.trace_error_inf
                << std::fixed << std::setprecision(2)
                << std::setw(8) << printable_order(row.trace_order)
                << std::scientific << std::setprecision(3)
                << std::setw(13) << metrics.normal_error_inf
                << std::fixed << std::setprecision(2)
                << std::setw(8) << printable_order(row.normal_order)
                << std::scientific << std::setprecision(3)
                << std::setw(13) << metrics.interior_bulk_inf
                << std::fixed << std::setprecision(2)
                << std::setw(8) << printable_order(row.bulk_order)
                << std::setw(9) << row.baseline_bulk_ratio
                << std::setw(8) << metrics.exact_crossing_owners
                << std::setw(7) << metrics.gap_fallback_owners
                << std::setw(8)
                << metrics.expanded_same_side_center_stencils
                << std::setw(8)
                << metrics.max_same_side_center_distance_over_h
                << std::setw(8)
                << metrics.max_six_point_weight_l1
                << std::setw(8) << metrics.domain_label_mismatches
                << '\n';

            previous_h = metrics.h;
            previous_trace = metrics.trace_error_inf;
            previous_normal = metrics.normal_error_inf;
            previous_bulk = metrics.interior_bulk_inf;
            rows.push_back(row);
        }
    }

    const std::filesystem::path csv_path =
        output_dir
        / ("neumann_exterior_trace_lshape_2d_rigid_transform_"
           + std::string(restrict_output_tag(restrict_method)) + "_"
           + crossing_jet_output_tag(selected_crossing_jet_scheme())
           + ".csv");
    write_rigid_study_csv(csv_path, restrict_method, rows);
    std::cout << "\nRigid-study CSV: " << csv_path.string() << '\n';
    return 0;
}

void print_usage(const char* executable)
{
    std::cout
        << "Usage:\n"
        << "  " << executable << " [N ...]\n"
        << "  " << executable << " --rigid-study [N ...]\n\n"
        << "Default levels: 30 60 120.\n"
        << "KFBIM_LSHAPE_RESTRICT selects "
           "joint_bicubic_cubic_crossing_owner (default), "
           "joint_biquadratic_quadratic_crossing_owner, or "
           "joint_biquadratic_quadratic_center_cauchy_jump, or "
           "joint_six_point_quadratic_center_cauchy_jump, or "
           "joint_biquadratic_quadratic_virtual_side_flip, or "
           "unified_spatial_normal_p2_dof_cauchy_exterior, or "
           "unified_spatial_normal_p2_crossing_owner, or "
           "six_point_quadratic.\n"
        << "KFBIM_LSHAPE_CROSSING_JET selects nsp_cj (default), "
           "als_cj, or local.\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc >= 2
            && (std::string(argv[1]) == "--help"
                || std::string(argv[1]) == "-h")) {
            print_usage(argv[0]);
            return 0;
        }
        const bool rigid_study =
            argc >= 2 && std::string(argv[1]) == "--rigid-study";
        std::vector<int> levels;
        for (int i = rigid_study ? 2 : 1; i < argc; ++i)
            levels.push_back(std::stoi(argv[i]));
        if (levels.empty())
            levels = {30, 60, 120};

        const int max_iter = environment_int(
            "KFBIM_EXT_TRACE_MAX_ITER", 200);
        const int restart = environment_int(
            "KFBIM_EXT_TRACE_RESTART", 50);
        const double tolerance = environment_double(
            "KFBIM_EXT_TRACE_TOL", 1.0e-8);
        const LaplaceNeumannExteriorRestrictMethod2D restrict_method =
            selected_restrict_method();

#ifdef KFBIM_APP_OUTPUT_DIR
        const std::filesystem::path output_dir = KFBIM_APP_OUTPUT_DIR;
#else
        const std::filesystem::path output_dir = "output";
#endif
        if (rigid_study) {
            return run_rigid_study(levels,
                                   max_iter,
                                   tolerance,
                                   restart,
                                   restrict_method,
                                   output_dir);
        }

        std::cout << "KFBI2D L-shape Neumann exterior-trace formulation\n"
                  << "smooth antisymmetric harmonic cubic; no singular correction\n"
                  << "P2 restrict=" << restrict_method_name(restrict_method)
                  << '\n'
                  << "crossing jet="
                  << laplace_crossing_jet_scheme_name_2d(
                         selected_crossing_jet_scheme())
                  << '\n'
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
                run_level(
                    n,
                    max_iter,
                    tolerance,
                    restart,
                    restrict_method,
                    baseline_rigid_case());
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

        const std::filesystem::path csv_path = output_dir
            / ("neumann_exterior_trace_lshape_2d_"
               + std::string(restrict_output_tag(restrict_method)) + "_"
               + crossing_jet_output_tag(
                     selected_crossing_jet_scheme())
               + ".csv");
        write_csv(csv_path, results);
        std::cout << "\nCSV: " << csv_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
