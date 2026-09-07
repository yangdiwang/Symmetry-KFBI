#include <algorithm>
#include <array>
#include <cctype>
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
#include "src/geometry/p2_curve_2d.hpp"
#include "src/geometry/curve_resampler_2d.hpp"
#include "src/grid/cartesian_grid_2d.hpp"
#include "src/local_cauchy/laplace_panel_solver_2d.hpp"
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

struct RigidTransform2D {
    double angle_degrees = 0.0;
    Eigen::Matrix2d rotation = Eigen::Matrix2d::Identity();
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    Eigen::Vector2d translation = Eigen::Vector2d::Zero();

    Eigen::Vector2d forward_point(
        const Eigen::Vector2d& point) const
    {
        return center
             + rotation * (point - center)
             + translation;
    }

    Eigen::Vector2d inverse_point(
        const Eigen::Vector2d& point) const
    {
        return center
             + rotation.transpose()
                 * (point - center - translation);
    }

    Eigen::Vector2d forward_vector(
        const Eigen::Vector2d& vector) const
    {
        return rotation * vector;
    }

    bool is_identity() const
    {
        return std::abs(angle_degrees) <= 1.0e-14
            && translation.norm() <= 1.0e-14;
    }
};

RigidTransform2D make_rigid_transform(
    double angle_degrees,
    const Eigen::Vector2d& center,
    const Eigen::Vector2d& translation)
{
    if (!std::isfinite(angle_degrees)
        || !center.allFinite()
        || !translation.allFinite()) {
        throw std::invalid_argument(
            "harmonic-jet rigid transform must be finite");
    }
    RigidTransform2D transform;
    transform.angle_degrees = angle_degrees;
    transform.rotation = rotation(angle_degrees);
    transform.center = center;
    transform.translation = translation;
    return transform;
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

    Eigen::Vector2d second_deriv(double t) const override
    {
        const Eigen::Vector2d local(
            -1.10 * std::cos(t), -0.76 * std::sin(t));
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

    Eigen::Vector2d second_deriv(double t) const override
    {
        const double r = radius(t);
        const double rp = radius_derivative(t);
        const double rpp = radius_second_derivative(t);
        const Eigen::Vector2d local(
            rpp * std::cos(t) - 2.0 * rp * std::sin(t)
                - r * std::cos(t),
            rpp * std::sin(t) + 2.0 * rp * std::cos(t)
                - r * std::sin(t));
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

    static double radius_second_derivative(double theta)
    {
        return -0.88 * 0.18 * 25.0 * std::cos(5.0 * theta);
    }

    Eigen::Matrix2d rotation_;
};

class RigidlyTransformedCurve2D final : public ICurve2D {
public:
    RigidlyTransformedCurve2D(
        std::shared_ptr<const ICurve2D> source,
        RigidTransform2D transform)
        : source_(std::move(source))
        , transform_(std::move(transform))
    {
        if (!source_)
            throw std::invalid_argument(
                "rigidly transformed curve requires a source curve");
    }

    Eigen::Vector2d eval(double t) const override
    {
        return transform_.forward_point(source_->eval(t));
    }

    Eigen::Vector2d deriv(double t) const override
    {
        return transform_.forward_vector(source_->deriv(t));
    }

    Eigen::Vector2d second_deriv(double t) const override
    {
        return transform_.forward_vector(source_->second_deriv(t));
    }

    double t_min() const override { return source_->t_min(); }
    double t_max() const override { return source_->t_max(); }

private:
    std::shared_ptr<const ICurve2D> source_;
    RigidTransform2D transform_;
};

std::shared_ptr<const ICurve2D> make_reference_curve(
    const std::string& geometry)
{
    if (geometry == "ellipse")
        return std::make_shared<RotatedEllipseCurve2D>();
    if (geometry == "flower")
        return std::make_shared<SmoothFlowerCurve2D>();
    throw std::invalid_argument("geometry must be ellipse or flower");
}

std::shared_ptr<const ICurve2D> make_curve(
    const std::string& geometry,
    const RigidTransform2D& transform)
{
    return std::make_shared<RigidlyTransformedCurve2D>(
        make_reference_curve(geometry), transform);
}

double reference_exact_solution(double x, double y)
{
    constexpr double a = 0.42;
    return std::exp(a * x) * std::cos(a * y)
         + 0.08 * (x * x - y * y) + 0.11 * x - 0.07 * y;
}

Eigen::Vector2d reference_exact_gradient(double x, double y)
{
    constexpr double a = 0.42;
    const double exponential = std::exp(a * x);
    return {
        a * exponential * std::cos(a * y) + 0.16 * x + 0.11,
       -a * exponential * std::sin(a * y) - 0.16 * y - 0.07
    };
}

Eigen::Matrix2d reference_exact_hessian(double x, double y)
{
    constexpr double a = 0.42;
    const double exponential = std::exp(a * x);
    const double cosine = std::cos(a * y);
    const double sine = std::sin(a * y);
    Eigen::Matrix2d hessian;
    hessian
        << a * a * exponential * cosine + 0.16,
           -a * a * exponential * sine,
           -a * a * exponential * sine,
           -a * a * exponential * cosine - 0.16;
    return hessian;
}

double transformed_exact_solution(
    const RigidTransform2D& transform,
    const Eigen::Vector2d& point)
{
    const Eigen::Vector2d source =
        transform.inverse_point(point);
    return reference_exact_solution(source.x(), source.y());
}

Eigen::Vector2d transformed_exact_gradient(
    const RigidTransform2D& transform,
    const Eigen::Vector2d& point)
{
    const Eigen::Vector2d source =
        transform.inverse_point(point);
    return transform.forward_vector(
        reference_exact_gradient(source.x(), source.y()));
}

Eigen::Matrix2d transformed_exact_hessian(
    const RigidTransform2D& transform,
    const Eigen::Vector2d& point)
{
    const Eigen::Vector2d source =
        transform.inverse_point(point);
    return transform.rotation
         * reference_exact_hessian(source.x(), source.y())
         * transform.rotation.transpose();
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

std::string rigid_case_tag()
{
    const char* raw = std::getenv("KFBIM_HJET_RIGID_CASE");
    if (raw == nullptr || std::string(raw).empty())
        return {};
    const std::string tag(raw);
    for (const unsigned char character : tag) {
        if (!std::isalnum(character)
            && character != '_'
            && character != '-') {
            throw std::invalid_argument(
                "KFBIM_HJET_RIGID_CASE must contain only letters, "
                "digits, underscores, or hyphens");
        }
    }
    return tag;
}

CurvaturePanelMonitor2D selected_panel_curvature_monitor()
{
    const char* raw =
        std::getenv("KFBIM_HJET_PANEL_CURVATURE_MONITOR");
    if (raw == nullptr || std::string(raw).empty()
        || std::string(raw) == "maximum"
        || std::string(raw) == "max"
        || std::string(raw) == "minimal") {
        return CurvaturePanelMonitor2D::Maximum;
    }
    if (std::string(raw) == "additive"
        || std::string(raw) == "sum") {
        return CurvaturePanelMonitor2D::Additive;
    }
    throw std::invalid_argument(
        "KFBIM_HJET_PANEL_CURVATURE_MONITOR must be "
        "maximum (or max/minimal) or additive (or sum)");
}

const char* panel_curvature_monitor_name(
    CurvaturePanelMonitor2D monitor)
{
    switch (monitor) {
    case CurvaturePanelMonitor2D::Maximum:
        return "maximum";
    case CurvaturePanelMonitor2D::Additive:
        return "additive";
    }
    throw std::runtime_error("unknown curvature panel monitor");
}

LaplaceNeumannExteriorRestrictMethod2D selected_restrict_method()
{
    const char* raw = std::getenv("KFBIM_HJET_RESTRICT");
    if (raw == nullptr || std::string(raw).empty()
        || std::string(raw) == "six_point_quadratic"
        || std::string(raw) == "legacy"
        || std::string(raw) == "default") {
        return LaplaceNeumannExteriorRestrictMethod2D::SixPointQuadratic;
    }
    if (std::string(raw) == "joint_bicubic_cubic_crossing_owner"
        || std::string(raw) == "crossing_owner") {
        return LaplaceNeumannExteriorRestrictMethod2D::
            JointBicubicCubicCrossingOwner;
    }
    if (std::string(raw)
            == "joint_biquadratic_quadratic_center_cauchy_jump"
        || std::string(raw) == "center_cauchy_jump") {
        return LaplaceNeumannExteriorRestrictMethod2D::
            JointBiquadraticQuadraticCenterCauchyJump;
    }
    if (std::string(raw)
            == "joint_six_point_quadratic_interface_jump"
        || std::string(raw) == "six_point_cross_interface_jump"
        || std::string(raw) == "strong_decoupled") {
        return LaplaceNeumannExteriorRestrictMethod2D::
            JointSixPointQuadraticInterfaceJump;
    }
    if (std::string(raw)
            == "joint_six_point_quadratic_grid_edge_interface_jump"
        || std::string(raw) == "six_point_grid_edge_interface_jump"
        || std::string(raw) == "strong_decoupled_grid_edge") {
        return LaplaceNeumannExteriorRestrictMethod2D::
            JointSixPointQuadraticGridEdgeInterfaceJump;
    }
    if (std::string(raw)
            == "joint_six_point_quadratic_grid_edge_shared_side_polynomial_interface_jump"
        || std::string(raw)
               == "six_point_grid_edge_shared_side_polynomial"
        || std::string(raw) == "shared_side_polynomial") {
        return LaplaceNeumannExteriorRestrictMethod2D::
            JointSixPointQuadraticGridEdgeSharedSidePolynomialInterfaceJump;
    }
    if (std::string(raw)
            == "joint_six_point_quadratic_center_cauchy_jump"
        || std::string(raw) == "six_point_cross_stencil") {
        return LaplaceNeumannExteriorRestrictMethod2D::
            JointSixPointQuadraticCenterCauchyJump;
    }
    throw std::invalid_argument(
        "KFBIM_HJET_RESTRICT must be six_point_quadratic (or legacy), "
        "joint_bicubic_cubic_crossing_owner (or crossing_owner), "
        "joint_biquadratic_quadratic_center_cauchy_jump "
        "(or center_cauchy_jump), or "
        "joint_six_point_quadratic_interface_jump "
        "(or six_point_cross_interface_jump/strong_decoupled), or "
        "joint_six_point_quadratic_grid_edge_interface_jump "
        "(or six_point_grid_edge_interface_jump/"
        "strong_decoupled_grid_edge), or "
        "joint_six_point_quadratic_grid_edge_shared_side_polynomial_interface_jump "
        "(or six_point_grid_edge_shared_side_polynomial/"
        "shared_side_polynomial), or "
        "joint_six_point_quadratic_center_cauchy_jump "
        "(or six_point_cross_stencil)");
}

const char* restrict_method_name(
    LaplaceNeumannExteriorRestrictMethod2D method)
{
    switch (method) {
    case LaplaceNeumannExteriorRestrictMethod2D::SixPointQuadratic:
        return "six_point_quadratic";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointBicubicCubicCrossingOwner:
        return "joint_bicubic_cubic_crossing_owner";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointBiquadraticQuadraticCenterCauchyJump:
        return "joint_biquadratic_quadratic_center_cauchy_jump";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointSixPointQuadraticInterfaceJump:
        return "joint_six_point_quadratic_interface_jump";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointSixPointQuadraticGridEdgeInterfaceJump:
        return "joint_six_point_quadratic_grid_edge_interface_jump";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointSixPointQuadraticGridEdgeSharedSidePolynomialInterfaceJump:
        return "joint_six_point_quadratic_grid_edge_shared_side_polynomial_interface_jump";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointSixPointQuadraticCenterCauchyJump:
        return "joint_six_point_quadratic_center_cauchy_jump";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointBiquadraticQuadraticCrossingOwner:
        return "joint_biquadratic_quadratic_crossing_owner";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointBiquadraticQuadraticVirtualSideFlip:
        return "joint_biquadratic_quadratic_virtual_side_flip";
    }
    throw std::runtime_error("unknown harmonic-jet restrict method");
}

const char* restrict_output_tag(
    LaplaceNeumannExteriorRestrictMethod2D method)
{
    switch (method) {
    case LaplaceNeumannExteriorRestrictMethod2D::SixPointQuadratic:
        return "six_point_quadratic";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointBicubicCubicCrossingOwner:
        return "p2_crossing_owner_joint_cubic";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointBiquadraticQuadraticCenterCauchyJump:
        return "p2_joint_quadratic_center_cauchy_jump";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointSixPointQuadraticInterfaceJump:
        return "p2_joint_six_point_quadratic_interface_jump";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointSixPointQuadraticGridEdgeInterfaceJump:
        return "p2_joint_six_point_quadratic_grid_edge_interface_jump";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointSixPointQuadraticGridEdgeSharedSidePolynomialInterfaceJump:
        return "p2_joint_six_point_quadratic_grid_edge_shared_side_polynomial_interface_jump";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointSixPointQuadraticCenterCauchyJump:
        return "p2_joint_six_point_quadratic_center_cauchy_jump";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointBiquadraticQuadraticCrossingOwner:
        return "p2_crossing_owner_joint_quadratic";
    case LaplaceNeumannExteriorRestrictMethod2D::
             JointBiquadraticQuadraticVirtualSideFlip:
        return "p2_joint_quadratic_virtual_side_flip";
    }
    throw std::runtime_error("unknown harmonic-jet restrict output tag");
}

LaplaceP2PanelCenterSpreadMode2D selected_spread_mode()
{
    const char* raw = std::getenv("KFBIM_HJET_SPREAD");
    if (raw == nullptr || std::string(raw).empty()
        || std::string(raw) == "quadratic_cauchy"
        || std::string(raw) == "p2_quadratic"
        || std::string(raw) == "default") {
        return LaplaceP2PanelCenterSpreadMode2D::QuadraticCauchy;
    }
    if (std::string(raw) == "cubic_harmonic"
        || std::string(raw) == "cubic_harmonic_4p3") {
        return LaplaceP2PanelCenterSpreadMode2D::CubicHarmonic;
    }
    throw std::invalid_argument(
        "KFBIM_HJET_SPREAD must be quadratic_cauchy (or p2_quadratic) "
        "or cubic_harmonic (or cubic_harmonic_4p3)");
}

const char* spread_mode_name(LaplaceP2PanelCenterSpreadMode2D mode)
{
    switch (mode) {
    case LaplaceP2PanelCenterSpreadMode2D::QuadraticCauchy:
        return "quadratic_cauchy";
    case LaplaceP2PanelCenterSpreadMode2D::CubicHarmonic:
        return "cubic_harmonic_4p3";
    }
    throw std::runtime_error("unknown harmonic-jet spread mode");
}

double inf_norm(const Eigen::VectorXd& values)
{
    return values.size() == 0 ? 0.0 : values.cwiseAbs().maxCoeff();
}

bool reference_analytic_inside(
    const std::string& geometry,
    const Eigen::Vector2d& point)
{
    Eigen::Vector2d translated;
    double angle_degrees = 0.0;
    if (geometry == "ellipse") {
        translated =
            point - Eigen::Vector2d(0.13, -0.08);
        angle_degrees = 27.0;
    } else if (geometry == "flower") {
        translated =
            point - Eigen::Vector2d(0.09, -0.06);
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

bool transformed_analytic_inside(
    const std::string& geometry,
    const RigidTransform2D& transform,
    const Eigen::Vector2d& point)
{
    return reference_analytic_inside(
        geometry, transform.inverse_point(point));
}

struct BulkError {
    double shift = 0.0;
    double linf = 0.0;
    double l2 = 0.0;
    int count = 0;
    int linf_node = -1;
    int linf_i = -1;
    int linf_j = -1;
    double linf_x = std::numeric_limits<double>::quiet_NaN();
    double linf_y = std::numeric_limits<double>::quiet_NaN();
    double linf_signed_error = 0.0;
};

BulkError measure_interior_bulk(
    const std::string& geometry,
    const RigidTransform2D& transform,
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
            const auto grid_point = grid.coord(i, j);
            const Eigen::Vector2d point(
                grid_point[0], grid_point[1]);
            const bool is_inside = use_analytic_mask
                ? transformed_analytic_inside(
                      geometry, transform, point)
                : grid_pair.domain_label(node) > 0;
            if (!is_inside)
                continue;
            shift_sum +=
                transformed_exact_solution(transform, point)
                - field[node];
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
            const auto grid_point = grid.coord(i, j);
            const Eigen::Vector2d point(
                grid_point[0], grid_point[1]);
            const bool is_inside = use_analytic_mask
                ? transformed_analytic_inside(
                      geometry, transform, point)
                : grid_pair.domain_label(node) > 0;
            if (!is_inside)
                continue;
            const double error = field[node] + result.shift
                               - transformed_exact_solution(
                                     transform, point);
            const double abs_error = std::abs(error);
            if (result.linf_node < 0 || abs_error > result.linf) {
                result.linf = abs_error;
                result.linf_node = node;
                result.linf_i = i;
                result.linf_j = j;
                result.linf_x = point[0];
                result.linf_y = point[1];
                result.linf_signed_error = error;
            }
            sum_sq += error * error;
        }
    }
    result.l2 = std::sqrt(sum_sq / static_cast<double>(result.count));
    return result;
}

struct SelectedGridEdgeSampleDiagnostic {
    LaplaceP2JointPolynomialRestrictDiagnostics2D::GridEdgeSample sample;
    std::array<double, 2> solved_diagonal_corrections{{
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()}};
    std::array<double, 2> exact_diagonal_corrections{{
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()}};
    double analytic_target =
        std::numeric_limits<double>::quiet_NaN();
    double analytic_corrected_sample =
        std::numeric_limits<double>::quiet_NaN();
    double operator_corrected_sample =
        std::numeric_limits<double>::quiet_NaN();
    double smooth_interpolation_error =
        std::numeric_limits<double>::quiet_NaN();
    double cauchy_correction_error =
        std::numeric_limits<double>::quiet_NaN();
    double bulk_grid_error =
        std::numeric_limits<double>::quiet_NaN();
    double irregular_spread_bulk_error =
        std::numeric_limits<double>::quiet_NaN();
    double irregular_cauchy_fit_bulk_error =
        std::numeric_limits<double>::quiet_NaN();
    double value_jump_input_bulk_error =
        std::numeric_limits<double>::quiet_NaN();
    double normal_jump_input_bulk_error =
        std::numeric_limits<double>::quiet_NaN();
    double normal_node_frame_bulk_error =
        std::numeric_limits<double>::quiet_NaN();
    double normal_scalar_interpolation_bulk_error =
        std::numeric_limits<double>::quiet_NaN();
    double normal_input_split_closure_error =
        std::numeric_limits<double>::quiet_NaN();
    double cauchy_geometry_reconstruction_bulk_error =
        std::numeric_limits<double>::quiet_NaN();
    double cauchy_input_split_closure_error =
        std::numeric_limits<double>::quiet_NaN();
    double quadratic_taylor_remainder_bulk_error =
        std::numeric_limits<double>::quiet_NaN();
    double irregular_split_closure_error =
        std::numeric_limits<double>::quiet_NaN();
    double regular_five_point_bulk_error =
        std::numeric_limits<double>::quiet_NaN();
    double bulk_split_closure_error =
        std::numeric_limits<double>::quiet_NaN();
    double operator_sample_error =
        std::numeric_limits<double>::quiet_NaN();
};

struct SelectedBulkDefectSplitDiagnostic {
    int irregular_nodes = 0;
    int regular_nodes = 0;
    double spread_rhs_linf = 0.0;
    double irregular_defect_linf = 0.0;
    double regular_defect_linf = 0.0;
    double cauchy_fit_defect_linf = 0.0;
    double value_jump_input_defect_linf = 0.0;
    double normal_jump_input_defect_linf = 0.0;
    double normal_node_frame_defect_linf = 0.0;
    double normal_scalar_interpolation_defect_linf = 0.0;
    double normal_input_source_closure_linf = 0.0;
    double cauchy_geometry_reconstruction_defect_linf = 0.0;
    double cauchy_input_source_closure_linf = 0.0;
    double taylor_remainder_defect_linf = 0.0;
    double irregular_source_closure_linf = 0.0;
    double irregular_solution_linf = 0.0;
    double cauchy_fit_solution_linf = 0.0;
    double value_jump_input_solution_linf = 0.0;
    double normal_jump_input_solution_linf = 0.0;
    double normal_node_frame_solution_linf = 0.0;
    double normal_scalar_interpolation_solution_linf = 0.0;
    double normal_input_solution_closure_linf = 0.0;
    double cauchy_geometry_reconstruction_solution_linf = 0.0;
    double cauchy_input_solution_closure_linf = 0.0;
    double taylor_remainder_solution_linf = 0.0;
    double irregular_solution_closure_linf = 0.0;
    double regular_solution_linf = 0.0;
    double closure_linf = 0.0;
    double value_collocation_interpolation_linf = 0.0;
    double normal_collocation_interpolation_linf = 0.0;
    // Exact linear counterfactuals obtained by removing one diagnosed
    // Cauchy-input defect before applying the same bulk inverse.
    double exact_value_counterfactual_solution_linf = 0.0;
    double exact_normal_counterfactual_solution_linf = 0.0;
    double exact_value_normal_counterfactual_solution_linf = 0.0;
    // Crossing-owner continuity counterfactual.  These fields are populated
    // only by the opt-in KFBIM_HJET_STENCIL_Q diagnostic.  The production
    // spread continues to use its original hard nearest-center owner.
    int crossing_owner_edges = 0;
    int crossing_owner_exact_edges = 0;
    int crossing_owner_multiple_intersection_edges = 0;
    int crossing_owner_switch_distance_lt_001 = 0;
    int crossing_owner_switch_distance_lt_005 = 0;
    double crossing_owner_switch_distance_min =
        std::numeric_limits<double>::quiet_NaN();
    double crossing_owner_switch_distance_p10 =
        std::numeric_limits<double>::quiet_NaN();
    double crossing_owner_switch_distance_median =
        std::numeric_limits<double>::quiet_NaN();
    double crossing_owner_adjacent_gap_p95 = 0.0;
    double crossing_owner_adjacent_gap_linf = 0.0;
    int hotspot_panel_crossing_owner_edges = 0;
    double hotspot_panel_switch_distance_min =
        std::numeric_limits<double>::quiet_NaN();
    double hotspot_panel_adjacent_gap_linf = 0.0;
    double crossing_owner_hard_reconstruction_linf = 0.0;
    double crossing_owner_blend_rhs_delta_linf = 0.0;
    double crossing_owner_blended_defect_linf = 0.0;
    double crossing_owner_hard_total_solution_linf = 0.0;
    double crossing_owner_blended_solution_linf = 0.0;
    double crossing_owner_blended_total_solution_linf = 0.0;
    double crossing_owner_blend_solution_delta_linf = 0.0;
    // Boundary-value reconstruction counterfactuals.  The normal jump,
    // crossing-owner routing, quadratic Cauchy solve, spread stencil, and
    // bulk inverse are held fixed; only [u] at Cauchy collocation points is
    // reconstructed differently.  Populated only by KFBIM_HJET_STENCIL_Q.
    double c1_hermite_collocation_interpolation_linf = 0.0;
    double p3_curve_collocation_interpolation_linf = 0.0;
    double c1_hermite_spread_rhs_delta_linf = 0.0;
    double p3_curve_spread_rhs_delta_linf = 0.0;
    double c1_hermite_irregular_defect_linf = 0.0;
    double p3_curve_irregular_defect_linf = 0.0;
    double c1_hermite_irregular_solution_linf = 0.0;
    double p3_curve_irregular_solution_linf = 0.0;
    double c1_hermite_total_solution_linf = 0.0;
    double p3_curve_total_solution_linf = 0.0;
    int arc_length_map_samples = 0;
    double arc_length_map_mean_knot_spacing = 0.0;
    double cauchy_collocation_arc_step_min =
        std::numeric_limits<double>::infinity();
    double cauchy_collocation_arc_step_max = 0.0;
    double cauchy_step_over_mean_knot_min =
        std::numeric_limits<double>::infinity();
    double cauchy_step_over_mean_knot_max = 0.0;
    double panel_point_tangent_fd_relative_linf = 0.0;
    double panel_point_normal_fd_angle_linf = 0.0;
};

struct SelectedJointFitDiagnostic {
    int q = -1;
    double exterior_smooth_error = 0.0;
    double exterior_cauchy_error = 0.0;
    double exterior_bulk_error = 0.0;
    double exterior_irregular_spread_error = 0.0;
    double exterior_irregular_cauchy_fit_error = 0.0;
    double exterior_value_jump_input_error = 0.0;
    double exterior_normal_jump_input_error = 0.0;
    double exterior_normal_node_frame_error = 0.0;
    double exterior_normal_scalar_interpolation_error = 0.0;
    double exterior_normal_input_split_closure_error = 0.0;
    double exterior_cauchy_geometry_reconstruction_error = 0.0;
    double exterior_cauchy_input_split_closure_error = 0.0;
    double exterior_quadratic_taylor_remainder_error = 0.0;
    double exterior_irregular_split_closure_error = 0.0;
    double exterior_regular_five_point_error = 0.0;
    double exterior_bulk_split_closure_error = 0.0;
    double exterior_jump_error = 0.0;
    double exterior_total_error = 0.0;
    double exterior_solver_value = 0.0;
    double interior_smooth_error = 0.0;
    double interior_cauchy_error = 0.0;
    double interior_bulk_error = 0.0;
    double interior_irregular_spread_error = 0.0;
    double interior_irregular_cauchy_fit_error = 0.0;
    double interior_value_jump_input_error = 0.0;
    double interior_normal_jump_input_error = 0.0;
    double interior_normal_node_frame_error = 0.0;
    double interior_normal_scalar_interpolation_error = 0.0;
    double interior_normal_input_split_closure_error = 0.0;
    double interior_cauchy_geometry_reconstruction_error = 0.0;
    double interior_cauchy_input_split_closure_error = 0.0;
    double interior_quadratic_taylor_remainder_error = 0.0;
    double interior_irregular_split_closure_error = 0.0;
    double interior_regular_five_point_error = 0.0;
    double interior_bulk_split_closure_error = 0.0;
    double interior_jump_error = 0.0;
    double interior_total_error = 0.0;
    double interior_solver_error = 0.0;
};

struct GridEdgeStencilShapeSummary {
    int groups = 0;
    std::array<int, 3> diagonal_types{{0, 0, 0}};
    std::array<int, 4> diagonal_orientations{{0, 0, 0, 0}};
    std::array<int, 5> cardinal_correction_counts{{0, 0, 0, 0, 0}};
    std::array<int, 6> total_correction_counts{{0, 0, 0, 0, 0, 0}};
    std::array<int, 6> distinct_owner_panel_counts{{0, 0, 0, 0, 0, 0}};
    std::array<int, 16> cardinal_masks{{
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0}};
    int diagonal_toward_selection_query = 0;
    int mixed_owner_panel_groups = 0;
    int endpoint_groups = 0;
    int endpoint_mixed_owner_panel_groups = 0;
    int midpoint_groups = 0;
    int midpoint_mixed_owner_panel_groups = 0;
    std::array<double, 3> weight_l1_mean{{0.0, 0.0, 0.0}};
    std::array<double, 3> weight_l1_p95{{0.0, 0.0, 0.0}};
    std::array<double, 3> weight_l1_p99{{0.0, 0.0, 0.0}};
    std::array<double, 3> weight_l1_max{{0.0, 0.0, 0.0}};
};

struct PanelSamplingSummary {
    double min_polyline_length_over_h =
        std::numeric_limits<double>::infinity();
    double max_polyline_length_over_h = 0.0;
    double mean_polyline_length_over_h = 0.0;
    double max_absolute_turn = 0.0;
    double mean_absolute_turn = 0.0;
    double max_curvature_length_product = 0.0;
    double max_endpoint_normal_gap_angle = 0.0;
    double max_endpoint_normal_gap_norm = 0.0;
    double max_endpoint_normal_to_interface_angle = 0.0;
    int max_endpoint_normal_gap_q = -1;
    int max_endpoint_normal_gap_panel0 = -1;
    int max_endpoint_normal_gap_panel1 = -1;
};

double absolute_turn(
    const Eigen::Vector2d& first,
    const Eigen::Vector2d& second)
{
    return std::abs(std::atan2(
        first.x() * second.y() - first.y() * second.x(),
        first.dot(second)));
}

PanelSamplingSummary summarize_panel_sampling(
    const Interface2D& iface,
    double h)
{
    PanelSamplingSummary summary;
    for (int panel = 0; panel < iface.num_panels(); ++panel) {
        std::array<Eigen::Vector2d, 3> points;
        std::array<Eigen::Vector2d, 3> tangents;
        for (int local = 0; local < 3; ++local) {
            const int q = iface.point_index(panel, local);
            points[static_cast<std::size_t>(local)] =
                iface.points().row(q).transpose();
            const Eigen::Vector2d normal =
                iface.normals().row(q).transpose();
            tangents[static_cast<std::size_t>(local)] =
                Eigen::Vector2d(-normal.y(), normal.x());
        }
        const double length_over_h =
            ((points[1] - points[0]).norm()
             + (points[2] - points[1]).norm()) / h;
        const double turn =
            absolute_turn(tangents[0], tangents[1])
            + absolute_turn(tangents[1], tangents[2]);
        double max_panel_curvature = 0.0;
        for (const double s :
             std::array<double, 5>{{-1.0, -0.5, 0.0, 0.5, 1.0}}) {
            const Eigen::Vector2d tangent =
                geometry2d::panel_tangent(iface, panel, s);
            const Eigen::Vector2d second =
                geometry2d::panel_second_derivative(
                    iface, panel, s);
            const double speed = tangent.norm();
            if (!(speed > 1.0e-14))
                continue;
            const double curvature =
                std::abs(
                    tangent.x() * second.y()
                    - tangent.y() * second.x())
                / (speed * speed * speed);
            max_panel_curvature =
                std::max(max_panel_curvature, curvature);
        }
        const double panel_length =
            length_over_h * h;
        summary.min_polyline_length_over_h = std::min(
            summary.min_polyline_length_over_h, length_over_h);
        summary.max_polyline_length_over_h = std::max(
            summary.max_polyline_length_over_h, length_over_h);
        summary.mean_polyline_length_over_h += length_over_h;
        summary.max_absolute_turn =
            std::max(summary.max_absolute_turn, turn);
        summary.mean_absolute_turn += turn;
        summary.max_curvature_length_product = std::max(
            summary.max_curvature_length_product,
            max_panel_curvature * panel_length);
    }
    const double panel_count =
        static_cast<double>(iface.num_panels());
    summary.mean_polyline_length_over_h /= panel_count;
    summary.mean_absolute_turn /= panel_count;

    std::vector<std::vector<std::pair<int, int>>> incidents(
        static_cast<std::size_t>(iface.num_points()));
    for (int panel = 0; panel < iface.num_panels(); ++panel) {
        for (int local = 0; local < iface.points_per_panel(); ++local) {
            const int q = iface.point_index(panel, local);
            incidents[static_cast<std::size_t>(q)].push_back(
                {panel, local});
        }
    }
    const auto normal_angle =
        [](const Eigen::Vector2d& first,
           const Eigen::Vector2d& second) {
            return std::acos(std::max(
                -1.0, std::min(1.0, first.dot(second))));
        };
    for (int q = 0; q < iface.num_points(); ++q) {
        const auto& point_incidents =
            incidents[static_cast<std::size_t>(q)];
        if (point_incidents.size() != 2)
            continue;
        const auto [panel0, local0] = point_incidents[0];
        const auto [panel1, local1] = point_incidents[1];
        const Eigen::Vector2d normal0 =
            geometry2d::panel_normal(
                iface,
                panel0,
                geometry2d::kP2NodeS[
                    static_cast<std::size_t>(local0)]);
        const Eigen::Vector2d normal1 =
            geometry2d::panel_normal(
                iface,
                panel1,
                geometry2d::kP2NodeS[
                    static_cast<std::size_t>(local1)]);
        const Eigen::Vector2d interface_normal =
            iface.normals().row(q).transpose();
        const double gap_angle = normal_angle(normal0, normal1);
        const double interface_angle = std::max(
            normal_angle(normal0, interface_normal),
            normal_angle(normal1, interface_normal));
        summary.max_endpoint_normal_to_interface_angle =
            std::max(
                summary.max_endpoint_normal_to_interface_angle,
                interface_angle);
        if (summary.max_endpoint_normal_gap_q < 0
            || gap_angle
                   > summary.max_endpoint_normal_gap_angle) {
            summary.max_endpoint_normal_gap_angle = gap_angle;
            summary.max_endpoint_normal_gap_norm =
                (normal0 - normal1).norm();
            summary.max_endpoint_normal_gap_q = q;
            summary.max_endpoint_normal_gap_panel0 = panel0;
            summary.max_endpoint_normal_gap_panel1 = panel1;
        }
    }
    return summary;
}

struct StudyResult {
    std::string geometry;
    std::string spread_mode;
    std::string restrict_method;
    std::string panel_curvature_monitor;
    std::string rigid_case;
    double rigid_angle_degrees = 0.0;
    double rigid_center_x = 0.0;
    double rigid_center_y = 0.0;
    double rigid_translation_x = 0.0;
    double rigid_translation_y = 0.0;
    int n = 0;
    double h = 0.0;
    double panel_length_over_h = 0.0;
    double panel_curvature_turn_limit = 0.0;
    double panel_min_polyline_length_over_h = 0.0;
    double panel_max_polyline_length_over_h = 0.0;
    double panel_mean_polyline_length_over_h = 0.0;
    double panel_max_absolute_turn = 0.0;
    double panel_mean_absolute_turn = 0.0;
    double panel_max_curvature_length_product = 0.0;
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
    double exact_trace_consistency_linf = 0.0;
    double exact_trace_consistency_signed_residual = 0.0;
    double exact_trace_consistency_lambda = 0.0;
    int exact_trace_consistency_active_index = -1;
    int exact_trace_consistency_q = -1;
    double exact_trace_consistency_x =
        std::numeric_limits<double>::quiet_NaN();
    double exact_trace_consistency_y =
        std::numeric_limits<double>::quiet_NaN();
    int exact_trace_consistency_incident_count = 0;
    double exact_normal_consistency_linf = 0.0;
    double exact_normal_consistency_signed_residual = 0.0;
    int exact_normal_consistency_active_index = -1;
    int exact_normal_consistency_q = -1;
    double exact_normal_consistency_x =
        std::numeric_limits<double>::quiet_NaN();
    double exact_normal_consistency_y =
        std::numeric_limits<double>::quiet_NaN();
    int exact_normal_consistency_incident_count = 0;
    double global_linf = 0.0;
    double global_l2 = 0.0;
    double constant_shift = 0.0;
    int n_global = 0;
    double global_linf_signed_error = 0.0;
    int global_linf_node = -1;
    int global_linf_i = -1;
    int global_linf_j = -1;
    double global_linf_x = std::numeric_limits<double>::quiet_NaN();
    double global_linf_y = std::numeric_limits<double>::quiet_NaN();
    int global_linf_nearest_trace_q = -1;
    double global_linf_nearest_trace_x =
        std::numeric_limits<double>::quiet_NaN();
    double global_linf_nearest_trace_y =
        std::numeric_limits<double>::quiet_NaN();
    double global_linf_distance_to_trace_over_h =
        std::numeric_limits<double>::quiet_NaN();
    double native_global_linf = 0.0;
    double native_global_l2 = 0.0;
    double native_constant_shift = 0.0;
    int n_native_global = 0;
    double trace_linf = 0.0;
    double trace_l2 = 0.0;
    double trace_linf_signed_error = 0.0;
    int trace_linf_active_index = -1;
    int trace_linf_q = -1;
    double trace_linf_x = std::numeric_limits<double>::quiet_NaN();
    double trace_linf_y = std::numeric_limits<double>::quiet_NaN();
    int trace_linf_incident_count = 0;
    int trace_linf_panel0 = -1;
    int trace_linf_local0 = -1;
    int trace_linf_panel1 = -1;
    int trace_linf_local1 = -1;
    double normal_linf = 0.0;
    double normal_l2 = 0.0;
    double normal_linf_signed_error = 0.0;
    int normal_linf_active_index = -1;
    int normal_linf_q = -1;
    double normal_linf_x = std::numeric_limits<double>::quiet_NaN();
    double normal_linf_y = std::numeric_limits<double>::quiet_NaN();
    int restrict_wrong_side_nodes = 0;
    int restrict_corrected_nodes = 0;
    int exact_crossing_owners = 0;
    int gap_fallback_owners = 0;
    int endpoint_fallback_owners = 0;
    int center_cauchy_jump_samples = 0;
    int trace_points_without_incident_center = 0;
    int six_point_cross_stencils = 0;
    int diagonal_zero_crossing_rejections = 0;
    int diagonal_multiple_crossing_rejections = 0;
    int grid_edge_cross_stencils = 0;
    int grid_edge_cardinal_corrections = 0;
    int grid_edge_cardinal_ambiguous_rejections = 0;
    int grid_edge_diagonal_zero_owner_rejections = 0;
    int grid_edge_diagonal_ambiguous_edge_rejections = 0;
    int grid_edge_diagonal_same_side_stencils = 0;
    int grid_edge_diagonal_single_owner_stencils = 0;
    int grid_edge_diagonal_blended_owner_stencils = 0;
    int grid_edge_blended_correction_nodes = 0;
    int grid_edge_owner_terms = 0;
    int shared_side_spatial_polynomials = 0;
    int shared_side_spatial_polynomial_samples = 0;
    int expanded_same_side_center_stencils = 0;
    double max_same_side_center_distance_over_h = 0.0;
    double max_six_point_weight_l1 = 0.0;
    int max_six_point_weight_q = -1;
    int max_six_point_weight_side = -1;
    int max_six_point_weight_layer = -1;
    double cubic_harmonic_spread_max_condition = 0.0;
    GridEdgeStencilShapeSummary stencil_shapes;
    std::vector<SelectedGridEdgeSampleDiagnostic>
        selected_grid_edge_samples;
    SelectedBulkDefectSplitDiagnostic selected_bulk_defect_split;
    SelectedJointFitDiagnostic selected_joint_fit;
    double global_linf_order = std::numeric_limits<double>::quiet_NaN();
    double global_l2_order = std::numeric_limits<double>::quiet_NaN();
};

StudyResult run_one(
    const std::string& geometry,
    int n,
    double panel_length_over_h,
    double panel_curvature_turn_limit,
    CurvaturePanelMonitor2D panel_curvature_monitor,
    const RigidTransform2D& rigid_transform,
    int max_iter,
    double tolerance,
    int restart,
    LaplaceNeumannExteriorRestrictMethod2D restrict_method,
    LaplaceP2PanelCenterSpreadMode2D spread_mode)
{
    if (n < 24)
        throw std::invalid_argument("N must be at least 24");
    const double h = kBoxSide / static_cast<double>(n);
    CartesianGrid2D grid(
        {kBoxMin, kBoxMin}, {h, h}, {n, n}, DofLayout2D::Node);
    const std::shared_ptr<const ICurve2D> curve =
        make_curve(geometry, rigid_transform);
    const auto interface_start = std::chrono::steady_clock::now();
    const Interface2D iface =
        panel_curvature_turn_limit > 0.0
            ? CurveResampler2D::
                  discretize_quadratic_lagrange_curvature_adaptive(
                      curve,
                      h,
                      panel_length_over_h,
                      panel_curvature_turn_limit,
                      panel_curvature_monitor)
            : CurveResampler2D::discretize_quadratic_lagrange(
                  curve, h, panel_length_over_h);
    const PanelSamplingSummary panel_sampling =
        summarize_panel_sampling(iface, h);
    const double interface_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - interface_start).count();
    std::cout << "  setup interface: panels=" << iface.num_panels()
              << " dofs=" << iface.num_points()
              << " panel_L/h(min/mean/max)="
              << panel_sampling.min_polyline_length_over_h << '/'
              << panel_sampling.mean_polyline_length_over_h << '/'
              << panel_sampling.max_polyline_length_over_h
              << " panel_turn(mean/max)="
              << panel_sampling.mean_absolute_turn << '/'
              << panel_sampling.max_absolute_turn
              << " max_kappa_L="
              << panel_sampling.max_curvature_length_product
              << " endpoint_normal_gap(rad/deg/norm/q/panels)="
              << panel_sampling.max_endpoint_normal_gap_angle << '/'
              << panel_sampling.max_endpoint_normal_gap_angle
                    * 180.0 / kPi
              << '/'
              << panel_sampling.max_endpoint_normal_gap_norm << '/'
              << panel_sampling.max_endpoint_normal_gap_q << '/'
              << panel_sampling.max_endpoint_normal_gap_panel0 << '/'
              << panel_sampling.max_endpoint_normal_gap_panel1
              << " endpoint_normal_to_interface_max(rad/deg)="
              << panel_sampling
                    .max_endpoint_normal_to_interface_angle
              << '/'
              << panel_sampling
                    .max_endpoint_normal_to_interface_angle
                    * 180.0 / kPi
              << " rigid(angle/center/translation)="
              << rigid_transform.angle_degrees << '/'
              << rigid_transform.center.x() << '/'
              << rigid_transform.center.y() << '/'
              << rigid_transform.translation.x() << '/'
              << rigid_transform.translation.y()
              << " time=" << interface_sec << "s\n";
    for (int q = 0; q < iface.num_points(); ++q) {
        const Eigen::Vector2d point =
            iface.points().row(q).transpose();
        if (!(point.x() > kBoxMin && point.x() < kBoxMin + kBoxSide
              && point.y() > kBoxMin
              && point.y() < kBoxMin + kBoxSide)) {
            throw std::invalid_argument(
                "rigidly transformed interface leaves the fixed box");
        }
    }

    const auto solver_setup_start = std::chrono::steady_clock::now();
    LaplaceNeumannExteriorTraceOptions2D options;
    options.restrict_method = restrict_method;
    options.spread_mode = spread_mode;
    if (restrict_method
        != LaplaceNeumannExteriorRestrictMethod2D::SixPointQuadratic) {
        options.correction_method =
            LaplaceCorrectionMethod2D::CrossingOwner;
    }
    LaplaceNeumannExteriorTrace2D solver(grid, iface, options);
    const double solver_setup_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solver_setup_start).count();
    std::cout << "  setup operator: time=" << solver_setup_sec << "s\n";

    const int nq = solver.problem_size();
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(nq);
    Eigen::VectorXd exact_trace_raw(nq);
    Eigen::VectorXd normal_data(nq);
    for (int a = 0; a < nq; ++a) {
        const int q =
            solver.active_interface_points()[static_cast<std::size_t>(a)];
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        const Eigen::Vector2d point(x, y);
        exact_trace_raw[a] =
            transformed_exact_solution(rigid_transform, point);
        normal_data[a] =
            transformed_exact_gradient(rigid_transform, point).dot(
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

    // Insert the manufactured trace into the discrete bordered equation and
    // remove its optimal constant-mode multiplier.  Unlike the final GMRES
    // residual, this exposes the local spread/restrict consistency defect.
    Eigen::VectorXd exact_operator_image;
    solver.apply(exact_trace, exact_operator_image);
    const Eigen::VectorXd exact_trace_unbordered_residual =
        rhs_data.trace_rhs - exact_operator_image;
    const double border_denominator =
        solver.normalized_weights().dot(solver.border_column());
    if (!(std::abs(border_denominator) > 1.0e-14)) {
        throw std::runtime_error(
            "harmonic-jet exact consistency diagnostic has a singular border column");
    }
    const double exact_trace_consistency_lambda =
        solver.normalized_weights().dot(exact_trace_unbordered_residual)
        / border_denominator;
    const Eigen::VectorXd exact_trace_consistency_residual =
        exact_trace_unbordered_residual
        - exact_trace_consistency_lambda * solver.border_column();

    const LaplaceExteriorTraceField2D exact_cauchy_field =
        solver.evaluate_cauchy(
            exact_trace, rhs_data.compatible_neumann_data);
    const Eigen::VectorXd exact_normal_consistency_residual =
        exact_cauchy_field.normal_trace_interior
        - rhs_data.compatible_neumann_data;

    const BulkError bulk = measure_interior_bulk(
        geometry,
        rigid_transform,
        grid,
        solver.grid_pair(),
        true,
        result.u_bulk);
    const BulkError native_bulk = measure_interior_bulk(
        geometry,
        rigid_transform,
        grid,
        solver.grid_pair(),
        false,
        result.u_bulk);
    const Eigen::VectorXd trace_error =
        result.dirichlet_trace - exact_trace;

    StudyResult metrics;
    metrics.geometry = geometry;
    metrics.spread_mode = spread_mode_name(spread_mode);
    metrics.restrict_method = restrict_method_name(restrict_method);
    metrics.panel_curvature_monitor =
        panel_curvature_monitor_name(panel_curvature_monitor);
    metrics.rigid_angle_degrees =
        rigid_transform.angle_degrees;
    metrics.rigid_center_x = rigid_transform.center.x();
    metrics.rigid_center_y = rigid_transform.center.y();
    metrics.rigid_translation_x =
        rigid_transform.translation.x();
    metrics.rigid_translation_y =
        rigid_transform.translation.y();
    metrics.n = n;
    metrics.h = h;
    metrics.panel_length_over_h = panel_length_over_h;
    metrics.panel_curvature_turn_limit =
        panel_curvature_turn_limit;
    metrics.panel_min_polyline_length_over_h =
        panel_sampling.min_polyline_length_over_h;
    metrics.panel_max_polyline_length_over_h =
        panel_sampling.max_polyline_length_over_h;
    metrics.panel_mean_polyline_length_over_h =
        panel_sampling.mean_polyline_length_over_h;
    metrics.panel_max_absolute_turn =
        panel_sampling.max_absolute_turn;
    metrics.panel_mean_absolute_turn =
        panel_sampling.mean_absolute_turn;
    metrics.panel_max_curvature_length_product =
        panel_sampling.max_curvature_length_product;
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
    metrics.exact_trace_consistency_lambda =
        exact_trace_consistency_lambda;
    for (int a = 0; a < nq; ++a) {
        const double trace_residual =
            exact_trace_consistency_residual[a];
        if (metrics.exact_trace_consistency_active_index < 0
            || std::abs(trace_residual)
                   > metrics.exact_trace_consistency_linf) {
            metrics.exact_trace_consistency_linf =
                std::abs(trace_residual);
            metrics.exact_trace_consistency_signed_residual =
                trace_residual;
            metrics.exact_trace_consistency_active_index = a;
        }
        const double normal_residual =
            exact_normal_consistency_residual[a];
        if (metrics.exact_normal_consistency_active_index < 0
            || std::abs(normal_residual)
                   > metrics.exact_normal_consistency_linf) {
            metrics.exact_normal_consistency_linf =
                std::abs(normal_residual);
            metrics.exact_normal_consistency_signed_residual =
                normal_residual;
            metrics.exact_normal_consistency_active_index = a;
        }
    }
    const auto record_consistency_point =
        [&](int active_index,
            int& trace_q,
            double& x,
            double& y,
            int& incident_count) {
            if (active_index < 0)
                return;
            trace_q =
                solver.active_interface_points()[static_cast<std::size_t>(
                    active_index)];
            x = iface.points()(trace_q, 0);
            y = iface.points()(trace_q, 1);
            for (int panel = 0; panel < iface.num_panels(); ++panel) {
                for (int local = 0;
                     local < iface.points_per_panel();
                     ++local) {
                    if (iface.point_index(panel, local) == trace_q)
                        ++incident_count;
                }
            }
        };
    record_consistency_point(
        metrics.exact_trace_consistency_active_index,
        metrics.exact_trace_consistency_q,
        metrics.exact_trace_consistency_x,
        metrics.exact_trace_consistency_y,
        metrics.exact_trace_consistency_incident_count);
    record_consistency_point(
        metrics.exact_normal_consistency_active_index,
        metrics.exact_normal_consistency_q,
        metrics.exact_normal_consistency_x,
        metrics.exact_normal_consistency_y,
        metrics.exact_normal_consistency_incident_count);
    metrics.global_linf = bulk.linf;
    metrics.global_l2 = bulk.l2;
    metrics.constant_shift = bulk.shift;
    metrics.n_global = bulk.count;
    metrics.global_linf_signed_error = bulk.linf_signed_error;
    metrics.global_linf_node = bulk.linf_node;
    metrics.global_linf_i = bulk.linf_i;
    metrics.global_linf_j = bulk.linf_j;
    metrics.global_linf_x = bulk.linf_x;
    metrics.global_linf_y = bulk.linf_y;
    if (bulk.linf_node >= 0) {
        const int nearest_q =
            solver.grid_pair().closest_interface_point(bulk.linf_node);
        metrics.global_linf_nearest_trace_q = nearest_q;
        metrics.global_linf_nearest_trace_x = iface.points()(nearest_q, 0);
        metrics.global_linf_nearest_trace_y = iface.points()(nearest_q, 1);
        metrics.global_linf_distance_to_trace_over_h =
            (Eigen::Vector2d(bulk.linf_x, bulk.linf_y)
             - iface.points().row(nearest_q).transpose()).norm() / h;
    }
    metrics.native_global_linf = native_bulk.linf;
    metrics.native_global_l2 = native_bulk.l2;
    metrics.native_constant_shift = native_bulk.shift;
    metrics.n_native_global = native_bulk.count;
    for (int a = 0; a < trace_error.size(); ++a) {
        const double abs_error = std::abs(trace_error[a]);
        if (metrics.trace_linf_active_index < 0
            || abs_error > metrics.trace_linf) {
            metrics.trace_linf = abs_error;
            metrics.trace_linf_signed_error = trace_error[a];
            metrics.trace_linf_active_index = a;
        }
    }
    metrics.trace_l2 =
        std::sqrt(trace_error.squaredNorm() / static_cast<double>(nq));
    if (metrics.trace_linf_active_index >= 0) {
        const int trace_q =
            solver.active_interface_points()[static_cast<std::size_t>(
                metrics.trace_linf_active_index)];
        metrics.trace_linf_q = trace_q;
        metrics.trace_linf_x = iface.points()(trace_q, 0);
        metrics.trace_linf_y = iface.points()(trace_q, 1);
        for (int panel = 0; panel < iface.num_panels(); ++panel) {
            for (int local = 0; local < iface.points_per_panel(); ++local) {
                if (iface.point_index(panel, local) != trace_q)
                    continue;
                if (metrics.trace_linf_incident_count == 0) {
                    metrics.trace_linf_panel0 = panel;
                    metrics.trace_linf_local0 = local;
                } else if (metrics.trace_linf_incident_count == 1) {
                    metrics.trace_linf_panel1 = panel;
                    metrics.trace_linf_local1 = local;
                }
                ++metrics.trace_linf_incident_count;
            }
        }
    }
    const Eigen::VectorXd normal_error =
        result.normal_trace_interior
        - result.compatible_neumann_data;
    for (int a = 0; a < normal_error.size(); ++a) {
        const double abs_error = std::abs(normal_error[a]);
        if (metrics.normal_linf_active_index < 0
            || abs_error > metrics.normal_linf) {
            metrics.normal_linf = abs_error;
            metrics.normal_linf_signed_error = normal_error[a];
            metrics.normal_linf_active_index = a;
        }
    }
    metrics.normal_l2 =
        std::sqrt(
            normal_error.squaredNorm() / static_cast<double>(nq));
    if (metrics.normal_linf_active_index >= 0) {
        const int normal_q =
            solver.active_interface_points()[static_cast<std::size_t>(
                metrics.normal_linf_active_index)];
        metrics.normal_linf_q = normal_q;
        metrics.normal_linf_x = iface.points()(normal_q, 0);
        metrics.normal_linf_y = iface.points()(normal_q, 1);
    }
    metrics.cubic_harmonic_spread_max_condition =
        solver.cubic_harmonic_spread_max_condition();
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
        metrics.endpoint_fallback_owners =
            diagnostics->endpoint_fallback_owners;
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
        metrics.grid_edge_cross_stencils =
            diagnostics->grid_edge_cross_stencils;
        metrics.grid_edge_cardinal_corrections =
            diagnostics->grid_edge_cardinal_corrections;
        metrics.grid_edge_cardinal_ambiguous_rejections =
            diagnostics->grid_edge_cardinal_ambiguous_rejections;
        metrics.grid_edge_diagonal_zero_owner_rejections =
            diagnostics->grid_edge_diagonal_zero_owner_rejections;
        metrics.grid_edge_diagonal_ambiguous_edge_rejections =
            diagnostics->grid_edge_diagonal_ambiguous_edge_rejections;
        metrics.grid_edge_diagonal_same_side_stencils =
            diagnostics->grid_edge_diagonal_same_side_stencils;
        metrics.grid_edge_diagonal_single_owner_stencils =
            diagnostics->grid_edge_diagonal_single_owner_stencils;
        metrics.grid_edge_diagonal_blended_owner_stencils =
            diagnostics->grid_edge_diagonal_blended_owner_stencils;
        metrics.grid_edge_blended_correction_nodes =
            diagnostics->grid_edge_blended_correction_nodes;
        metrics.grid_edge_owner_terms =
            diagnostics->grid_edge_owner_terms;
        metrics.shared_side_spatial_polynomials =
            diagnostics->shared_side_spatial_polynomials;
        metrics.shared_side_spatial_polynomial_samples =
            diagnostics->shared_side_spatial_polynomial_samples;
        metrics.expanded_same_side_center_stencils =
            diagnostics->expanded_same_side_center_stencils;
        metrics.max_same_side_center_distance_over_h =
            diagnostics->max_same_side_center_distance_over_h;
        metrics.max_six_point_weight_l1 =
            diagnostics->max_six_point_weight_l1;
        double located_max_weight_l1 = -1.0;
        for (const auto& sample : diagnostics->grid_edge_samples) {
            double weight_l1 = 0.0;
            for (double weight : sample.interpolation_weights)
                weight_l1 += std::abs(weight);
            if (weight_l1 > located_max_weight_l1) {
                located_max_weight_l1 = weight_l1;
                metrics.max_six_point_weight_q = sample.q;
                metrics.max_six_point_weight_side = sample.side;
                metrics.max_six_point_weight_layer = sample.layer;
            }
        }
        std::vector<int> incident_panel_count(
            static_cast<std::size_t>(iface.num_points()), 0);
        for (int panel = 0; panel < iface.num_panels(); ++panel) {
            for (int local = 0;
                 local < iface.points_per_panel();
                 ++local) {
                ++incident_panel_count[static_cast<std::size_t>(
                    iface.point_index(panel, local))];
            }
        }
        std::array<std::vector<double>, 3> layer_weight_l1;
        for (const auto& sample : diagnostics->grid_edge_samples) {
            double weight_l1 = 0.0;
            for (double weight : sample.interpolation_weights)
                weight_l1 += std::abs(weight);
            if (sample.layer >= 0 && sample.layer < 3) {
                layer_weight_l1[static_cast<std::size_t>(
                    sample.layer)].push_back(weight_l1);
            }
            if (sample.layer != 0)
                continue;

            GridEdgeStencilShapeSummary& shapes =
                metrics.stencil_shapes;
            ++shapes.groups;
            int diagonal_type = 0;
            if (sample.diagonal_corrected) {
                diagonal_type =
                    sample.diagonal_owner_count == 1 ? 1 : 2;
            }
            ++shapes.diagonal_types[
                static_cast<std::size_t>(diagonal_type)];

            const Eigen::Vector2d diagonal_vector =
                sample.grid_points[5] - sample.grid_points[0];
            const int orientation =
                (diagonal_vector[0] > 0.0 ? 1 : 0)
                + (diagonal_vector[1] > 0.0 ? 2 : 0);
            ++shapes.diagonal_orientations[
                static_cast<std::size_t>(orientation)];
            if (diagonal_vector.dot(
                    sample.selection_query
                    - sample.grid_points[0])
                >= 0.0) {
                ++shapes.diagonal_toward_selection_query;
            }

            int cardinal_mask = 0;
            int cardinal_corrections = 0;
            std::array<int, 6> owner_panels{{
                -1, -1, -1, -1, -1, -1}};
            int distinct_owner_panels = 0;
            const auto add_owner_panel =
                [&](int panel_index) {
                    if (panel_index < 0)
                        return;
                    for (int i = 0;
                         i < distinct_owner_panels;
                         ++i) {
                        if (owner_panels[static_cast<std::size_t>(i)]
                            == panel_index) {
                            return;
                        }
                    }
                    owner_panels[static_cast<std::size_t>(
                        distinct_owner_panels++)] = panel_index;
                };
            for (int cardinal = 0; cardinal < 4; ++cardinal) {
                if (!sample.cardinal_corrected[
                        static_cast<std::size_t>(cardinal)]) {
                    continue;
                }
                cardinal_mask |= 1 << cardinal;
                ++cardinal_corrections;
                add_owner_panel(
                    sample.cardinal_owners[
                        static_cast<std::size_t>(cardinal)]
                        .panel_index);
            }
            if (sample.diagonal_corrected) {
                for (int owner = 0;
                     owner < sample.diagonal_owner_count;
                     ++owner) {
                    add_owner_panel(
                        sample.diagonal_owners[
                            static_cast<std::size_t>(owner)]
                            .panel_index);
                }
            }
            ++shapes.cardinal_masks[
                static_cast<std::size_t>(cardinal_mask)];
            ++shapes.cardinal_correction_counts[
                static_cast<std::size_t>(cardinal_corrections)];
            const int total_corrections =
                cardinal_corrections
                + (sample.diagonal_corrected ? 1 : 0);
            ++shapes.total_correction_counts[
                static_cast<std::size_t>(total_corrections)];
            ++shapes.distinct_owner_panel_counts[
                static_cast<std::size_t>(distinct_owner_panels)];
            const bool mixed_owner_panels =
                distinct_owner_panels > 1;
            if (mixed_owner_panels)
                ++shapes.mixed_owner_panel_groups;
            const bool endpoint =
                sample.q >= 0
                && sample.q < iface.num_points()
                && incident_panel_count[
                       static_cast<std::size_t>(sample.q)]
                       > 1;
            if (endpoint) {
                ++shapes.endpoint_groups;
                if (mixed_owner_panels) {
                    ++shapes
                          .endpoint_mixed_owner_panel_groups;
                }
            } else {
                ++shapes.midpoint_groups;
                if (mixed_owner_panels) {
                    ++shapes
                          .midpoint_mixed_owner_panel_groups;
                }
            }
        }
        for (int layer = 0; layer < 3; ++layer) {
            std::vector<double>& values =
                layer_weight_l1[static_cast<std::size_t>(layer)];
            if (values.empty())
                continue;
            std::sort(values.begin(), values.end());
            double sum = 0.0;
            for (double value : values)
                sum += value;
            metrics.stencil_shapes.weight_l1_mean[
                static_cast<std::size_t>(layer)] =
                    sum / static_cast<double>(values.size());
            const auto percentile =
                [&](double fraction) {
                    const double rank =
                        fraction
                        * static_cast<double>(values.size() - 1);
                    return values[static_cast<std::size_t>(
                        std::ceil(rank))];
                };
            metrics.stencil_shapes.weight_l1_p95[
                static_cast<std::size_t>(layer)] =
                    percentile(0.95);
            metrics.stencil_shapes.weight_l1_p99[
                static_cast<std::size_t>(layer)] =
                    percentile(0.99);
            metrics.stencil_shapes.weight_l1_max[
                static_cast<std::size_t>(layer)] =
                    values.back();
        }
        const int selected_stencil_q =
            environment_int("KFBIM_HJET_STENCIL_Q", -1);
        if (selected_stencil_q >= 0) {
            // Optional, read-only localization diagnostic. It is deliberately
            // nested under KFBIM_HJET_STENCIL_Q so neither the production
            // operator nor an ordinary study run pays for the additional
            // inverse/Green-function probes below.
            const int bulk_dump_node =
                environment_int("KFBIM_HJET_BULK_DUMP_NODE", -1);
            if (bulk_dump_node >= grid.num_dofs()) {
                throw std::invalid_argument(
                    "KFBIM_HJET_BULK_DUMP_NODE is outside the bulk grid");
            }
            const Eigen::VectorXd zero_rhs_jump =
                Eigen::VectorXd::Zero(nq);
            Eigen::VectorXd exact_trace_by_q =
                Eigen::VectorXd::Zero(iface.num_points());
            Eigen::VectorXd exact_normal_by_q =
                Eigen::VectorXd::Zero(iface.num_points());
            for (int a = 0; a < nq; ++a) {
                const int q = solver.active_interface_points()[
                    static_cast<std::size_t>(a)];
                exact_trace_by_q[q] = exact_trace[a];
                exact_normal_by_q[q] =
                    rhs_data.compatible_neumann_data[a];
            }

            // Targeted read-only audit for the R17/N=64 flower anomaly.  It
            // exposes both the panel-11/12 bulk hotspot neighborhood and the
            // panel-61/62 trace hotspot without changing either operator.
            if (geometry == "flower" && n == 64
                && iface.num_panels() == 63
                && iface.num_points() == 126) {
                const std::array<int, 10> local_qs{{
                    11, 74, 12, 75, 13,
                    61, 124, 62, 125, 0}};
                for (const int q : local_qs) {
                    int active = -1;
                    for (int a = 0; a < nq; ++a) {
                        if (solver.active_interface_points()[
                                static_cast<std::size_t>(a)] == q) {
                            active = a;
                            break;
                        }
                    }
                    if (active < 0)
                        continue;
                    int incident0 = -1;
                    int incident1 = -1;
                    int local0 = -1;
                    int local1 = -1;
                    for (int panel = 0;
                         panel < iface.num_panels();
                         ++panel) {
                        for (int local = 0; local < 3; ++local) {
                            if (iface.point_index(panel, local) != q)
                                continue;
                            if (incident0 < 0) {
                                incident0 = panel;
                                local0 = local;
                            } else if (incident1 < 0
                                       && panel != incident0) {
                                incident1 = panel;
                                local1 = local;
                            }
                        }
                    }
                    std::cout
                        << "    n64_local_dof q=" << q
                        << " kind="
                        << (incident1 >= 0 ? "endpoint" : "midpoint")
                        << " panels/local=" << incident0 << '/' << local0
                        << ',' << incident1 << '/' << local1
                        << " xy=" << iface.points()(q, 0) << '/'
                        << iface.points()(q, 1)
                        << " trace_error=" << trace_error[active]
                        << " exact_trace_residual="
                        << exact_trace_consistency_residual[active]
                        << " exact_normal_residual="
                        << exact_normal_consistency_residual[active]
                        << " solved_trace="
                        << result.dirichlet_trace[active]
                        << " exact_trace=" << exact_trace[active]
                        << '\n';
                }

                const std::array<int, 4> local_panels{{11, 12, 61, 62}};
                for (const int panel : local_panels) {
                    std::array<Eigen::Vector2d, 3> tangents;
                    std::array<double, 3> curvatures{};
                    double polyline_length = 0.0;
                    for (int local = 0; local < 3; ++local) {
                        const double local_s =
                            geometry2d::kP2NodeS[
                                static_cast<std::size_t>(local)];
                        const Eigen::Vector2d tangent =
                            geometry2d::panel_tangent(
                                iface, panel, local_s);
                        const Eigen::Vector2d second =
                            geometry2d::panel_second_derivative(
                                iface, panel, local_s);
                        tangents[static_cast<std::size_t>(local)] =
                            tangent / tangent.norm();
                        curvatures[static_cast<std::size_t>(local)] =
                            std::abs(
                                tangent.x() * second.y()
                                - tangent.y() * second.x())
                            / std::pow(tangent.norm(), 3.0);
                        if (local > 0) {
                            const int q0 =
                                iface.point_index(panel, local - 1);
                            const int q1 = iface.point_index(panel, local);
                            polyline_length +=
                                (iface.points().row(q1)
                                 - iface.points().row(q0)).norm();
                        }
                    }
                    const double panel_length =
                        2.0
                        * geometry2d::panel_tangent(
                              iface, panel, 0.0).norm();
                    const double panel_turn =
                        absolute_turn(tangents[0], tangents[1])
                        + absolute_turn(tangents[1], tangents[2]);
                    double max_curvature = 0.0;
                    for (const double local_s :
                         std::array<double, 5>{{
                             -1.0, -0.5, 0.0, 0.5, 1.0}}) {
                        const Eigen::Vector2d tangent =
                            geometry2d::panel_tangent(
                                iface, panel, local_s);
                        const Eigen::Vector2d second =
                            geometry2d::panel_second_derivative(
                                iface, panel, local_s);
                        max_curvature = std::max(
                            max_curvature,
                            std::abs(
                                tangent.x() * second.y()
                                - tangent.y() * second.x())
                                / std::pow(tangent.norm(), 3.0));
                    }
                    std::cout
                        << "    n64_local_panel panel=" << panel
                        << " q=" << iface.point_index(panel, 0) << '/'
                        << iface.point_index(panel, 1) << '/'
                        << iface.point_index(panel, 2)
                        << " length=" << panel_length
                        << " length_over_h=" << panel_length / h
                        << " polyline_over_h=" << polyline_length / h
                        << " turn=" << panel_turn
                        << " curvature(nodes/max)="
                        << curvatures[0] << '/' << curvatures[1] << '/'
                        << curvatures[2] << '/' << max_curvature
                        << " max_kappa_L="
                        << max_curvature * panel_length
                        << '\n';
                    for (int local = 0; local < 3; ++local) {
                        const int q = iface.point_index(panel, local);
                        const double local_s =
                            geometry2d::kP2NodeS[
                                static_cast<std::size_t>(local)];
                        const Eigen::Vector2d tangent =
                            geometry2d::panel_tangent(
                                iface, panel, local_s).normalized();
                        const Eigen::Vector2d normal =
                            geometry2d::panel_normal(
                                iface, panel, local_s);
                        std::cout
                            << "      panel_node local/q=" << local << '/'
                            << q << " tangent=" << tangent.x() << '/'
                            << tangent.y() << " normal=" << normal.x()
                            << '/' << normal.y() << '\n';
                    }
                    for (const double center_s :
                         geometry2d::kP2CenterS) {
                        for (const double offset :
                             std::array<double, 3>{{
                                 -detail::kCollocationDelta,
                                 0.0,
                                 detail::kCollocationDelta}}) {
                            const double local_s = center_s + offset;
                            const Eigen::Vector2d point =
                                geometry2d::panel_point(
                                    iface, panel, local_s);
                            const double exact_value =
                                transformed_exact_solution(
                                    rigid_transform, point)
                                - exact_trace_mean;
                            const double p2_value =
                                geometry2d::panel_scalar(
                                    iface,
                                    panel,
                                    exact_trace_by_q,
                                    local_s);
                            std::cout
                                << "      p2_value_collocation center_s="
                                << center_s << " local_s=" << local_s
                                << " signed_error="
                                << p2_value - exact_value << '\n';
                        }
                    }
                }
            }

            // Two opt-in counterfactual reconstructions of the boundary
            // value.  ArcLengthCurvePanelGeometry2D parametrizes each panel
            // linearly in true arclength, so |dX/ds| is the true panel half
            // length.  Keeping the reconstruction here (inside the selected
            // stencil diagnostic) guarantees that the production P2 path is
            // unchanged.
            const int diagnostic_panel_count = iface.num_panels();
            std::vector<double> diagnostic_panel_lengths(
                static_cast<std::size_t>(diagnostic_panel_count), 0.0);
            std::vector<double> diagnostic_panel_starts(
                static_cast<std::size_t>(diagnostic_panel_count + 1), 0.0);
            for (int panel = 0;
                 panel < diagnostic_panel_count;
                 ++panel) {
                const double length =
                    2.0
                    * geometry2d::panel_tangent(iface, panel, 0.0).norm();
                if (!(length > 1.0e-14)) {
                    throw std::runtime_error(
                        "selected boundary reconstruction found a "
                        "degenerate panel");
                }
                diagnostic_panel_lengths[static_cast<std::size_t>(panel)] =
                    length;
                diagnostic_panel_starts[
                    static_cast<std::size_t>(panel + 1)] =
                        diagnostic_panel_starts[
                            static_cast<std::size_t>(panel)]
                        + length;
            }
            const double diagnostic_curve_length =
                diagnostic_panel_starts.back();
            SelectedBulkDefectSplitDiagnostic& reconstruction_split =
                metrics.selected_bulk_defect_split;
            reconstruction_split.arc_length_map_samples =
                environment_int(
                    "KFBIM_CURVE_ARC_LENGTH_MAP_SAMPLES", 20000);
            if (reconstruction_split.arc_length_map_samples < 2) {
                throw std::invalid_argument(
                    "KFBIM_CURVE_ARC_LENGTH_MAP_SAMPLES must be at least 2");
            }
            reconstruction_split.arc_length_map_mean_knot_spacing =
                diagnostic_curve_length
                / static_cast<double>(
                    reconstruction_split.arc_length_map_samples - 1);
            for (const double length : diagnostic_panel_lengths) {
                const double collocation_arc_step =
                    detail::kCollocationDelta * 0.5 * length;
                reconstruction_split.cauchy_collocation_arc_step_min =
                    std::min(
                        reconstruction_split.cauchy_collocation_arc_step_min,
                        collocation_arc_step);
                reconstruction_split.cauchy_collocation_arc_step_max =
                    std::max(
                        reconstruction_split.cauchy_collocation_arc_step_max,
                        collocation_arc_step);
                const double ratio =
                    collocation_arc_step
                    / reconstruction_split
                          .arc_length_map_mean_knot_spacing;
                reconstruction_split.cauchy_step_over_mean_knot_min =
                    std::min(
                        reconstruction_split
                            .cauchy_step_over_mean_knot_min,
                        ratio);
                reconstruction_split.cauchy_step_over_mean_knot_max =
                    std::max(
                        reconstruction_split
                            .cauchy_step_over_mean_knot_max,
                        ratio);
            }

            // The slope at common endpoint p is the arithmetic mean of the
            // adjacent panels' one-sided P2 endpoint derivatives after each
            // has been normalized by that panel's true arclength scale.
            std::vector<double> c1_endpoint_arc_slopes(
                static_cast<std::size_t>(diagnostic_panel_count), 0.0);
            for (int panel = 0;
                 panel < diagnostic_panel_count;
                 ++panel) {
                const int previous =
                    (panel + diagnostic_panel_count - 1)
                    % diagnostic_panel_count;
                const double previous_half_length =
                    0.5
                    * diagnostic_panel_lengths[
                        static_cast<std::size_t>(previous)];
                const double current_half_length =
                    0.5
                    * diagnostic_panel_lengths[
                        static_cast<std::size_t>(panel)];
                const double previous_derivative =
                    geometry2d::panel_scalar_deriv(
                        iface,
                        previous,
                        exact_trace_by_q,
                        1.0)
                    / previous_half_length;
                const double current_derivative =
                    geometry2d::panel_scalar_deriv(
                        iface,
                        panel,
                        exact_trace_by_q,
                        -1.0)
                    / current_half_length;
                c1_endpoint_arc_slopes[
                    static_cast<std::size_t>(panel)] =
                        0.5
                        * (previous_derivative + current_derivative);
            }
            const auto c1_hermite_value =
                [&](int panel, double local_s) {
                    const int next =
                        (panel + 1) % diagnostic_panel_count;
                    const int q_left = iface.point_index(panel, 0);
                    const int q_right = iface.point_index(panel, 2);
                    const double t = 0.5 * (local_s + 1.0);
                    const double t2 = t * t;
                    const double t3 = t2 * t;
                    const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                    const double h10 = t3 - 2.0 * t2 + t;
                    const double h01 = -2.0 * t3 + 3.0 * t2;
                    const double h11 = t3 - t2;
                    const double length =
                        diagnostic_panel_lengths[
                            static_cast<std::size_t>(panel)];
                    return
                        h00 * exact_trace_by_q[q_left]
                        + h10 * length
                            * c1_endpoint_arc_slopes[
                                static_cast<std::size_t>(panel)]
                        + h01 * exact_trace_by_q[q_right]
                        + h11 * length
                            * c1_endpoint_arc_slopes[
                                static_cast<std::size_t>(next)];
                };

            // Four consecutive P2 curve nodes form a nonuniform cubic
            // Lagrange stencil in true arclength.  The selected window spans
            // the half-panel containing local_s and therefore crosses smooth
            // common panel endpoints.  A physical corner in the window
            // disables this counterfactual locally and falls back to P2.
            const int diagnostic_curve_node_count =
                2 * diagnostic_panel_count;
            const auto wrap_curve_node =
                [&](int node) {
                    int wrapped = node % diagnostic_curve_node_count;
                    if (wrapped < 0)
                        wrapped += diagnostic_curve_node_count;
                    return wrapped;
                };
            const auto curve_node_q =
                [&](int wrapped_node) {
                    const int panel = wrapped_node / 2;
                    return iface.point_index(
                        panel, (wrapped_node % 2 == 0) ? 0 : 1);
                };
            const auto curve_node_arc =
                [&](int wrapped_node) {
                    const int panel = wrapped_node / 2;
                    double arc = diagnostic_panel_starts[
                        static_cast<std::size_t>(panel)];
                    if (wrapped_node % 2 != 0) {
                        arc += 0.5
                            * diagnostic_panel_lengths[
                                static_cast<std::size_t>(panel)];
                    }
                    return arc;
                };
            const auto p3_curve_value =
                [&](int panel, double local_s) {
                    const double query_arc =
                        diagnostic_panel_starts[
                            static_cast<std::size_t>(panel)]
                        + 0.5 * (local_s + 1.0)
                            * diagnostic_panel_lengths[
                                static_cast<std::size_t>(panel)];
                    const int first_unwrapped =
                        (local_s < 0.0) ? 2 * panel - 1 : 2 * panel;
                    std::array<double, 4> node_arcs{};
                    std::array<double, 4> node_values{};
                    for (int local = 0; local < 4; ++local) {
                        const int wrapped = wrap_curve_node(
                            first_unwrapped + local);
                        const int q = curve_node_q(wrapped);
                        if (iface.is_corner_point(q)) {
                            return geometry2d::panel_scalar(
                                iface,
                                panel,
                                exact_trace_by_q,
                                local_s);
                        }
                        double arc = curve_node_arc(wrapped);
                        while (arc - query_arc
                               > 0.5 * diagnostic_curve_length) {
                            arc -= diagnostic_curve_length;
                        }
                        while (arc - query_arc
                               < -0.5 * diagnostic_curve_length) {
                            arc += diagnostic_curve_length;
                        }
                        node_arcs[static_cast<std::size_t>(local)] = arc;
                        node_values[static_cast<std::size_t>(local)] =
                            exact_trace_by_q[q];
                    }
                    double value = 0.0;
                    for (int basis = 0; basis < 4; ++basis) {
                        double weight = 1.0;
                        for (int other = 0; other < 4; ++other) {
                            if (other == basis)
                                continue;
                            weight *=
                                (query_arc
                                 - node_arcs[
                                     static_cast<std::size_t>(other)])
                                / (node_arcs[
                                       static_cast<std::size_t>(basis)]
                                   - node_arcs[
                                       static_cast<std::size_t>(other)]);
                        }
                        value += weight
                            * node_values[
                                static_cast<std::size_t>(basis)];
                    }
                    return value;
                };

            for (int panel = 0; panel < iface.num_panels(); ++panel) {
                for (const double center_s : geometry2d::kP2CenterS) {
                    for (const double offset :
                         std::array<double, 3>{{
                             -detail::kCollocationDelta,
                             0.0,
                             detail::kCollocationDelta}}) {
                        const double local_s = center_s + offset;
                        const Eigen::Vector2d point =
                            geometry2d::panel_point(
                                iface, panel, local_s);
                        const Eigen::Vector2d normal =
                            geometry2d::panel_normal(
                                iface, panel, local_s);
                        constexpr double tangent_fd_step = 1.0e-6;
                        const Eigen::Vector2d point_minus =
                            geometry2d::panel_point(
                                iface,
                                panel,
                                local_s - tangent_fd_step);
                        const Eigen::Vector2d point_plus =
                            geometry2d::panel_point(
                                iface,
                                panel,
                                local_s + tangent_fd_step);
                        const Eigen::Vector2d point_tangent_fd =
                            (point_plus - point_minus)
                            / (2.0 * tangent_fd_step);
                        const Eigen::Vector2d retained_tangent =
                            geometry2d::panel_tangent(
                                iface, panel, local_s);
                        const double retained_tangent_norm =
                            retained_tangent.norm();
                        if (retained_tangent_norm > 1.0e-14
                            && point_tangent_fd.norm() > 1.0e-14) {
                            reconstruction_split
                                .panel_point_tangent_fd_relative_linf =
                                    std::max(
                                        reconstruction_split
                                            .panel_point_tangent_fd_relative_linf,
                                        (point_tangent_fd
                                         - retained_tangent).norm()
                                            / retained_tangent_norm);
                            const Eigen::Vector2d point_normal_fd(
                                point_tangent_fd.y()
                                    / point_tangent_fd.norm(),
                                -point_tangent_fd.x()
                                    / point_tangent_fd.norm());
                            const double cosine = std::max(
                                -1.0,
                                std::min(1.0, normal.dot(point_normal_fd)));
                            reconstruction_split
                                .panel_point_normal_fd_angle_linf =
                                    std::max(
                                        reconstruction_split
                                            .panel_point_normal_fd_angle_linf,
                                        std::acos(cosine));
                        }
                        const double exact_value =
                            transformed_exact_solution(
                                rigid_transform, point)
                            - exact_trace_mean;
                        const double interpolated_value =
                            geometry2d::panel_scalar(
                                iface,
                                panel,
                                exact_trace_by_q,
                                local_s);
                        metrics.selected_bulk_defect_split
                            .value_collocation_interpolation_linf =
                                std::max(
                                    metrics.selected_bulk_defect_split
                                        .value_collocation_interpolation_linf,
                                     std::abs(
                                         interpolated_value - exact_value));
                        metrics.selected_bulk_defect_split
                            .c1_hermite_collocation_interpolation_linf =
                                std::max(
                                    metrics.selected_bulk_defect_split
                                        .c1_hermite_collocation_interpolation_linf,
                                    std::abs(
                                        c1_hermite_value(panel, local_s)
                                        - exact_value));
                        metrics.selected_bulk_defect_split
                            .p3_curve_collocation_interpolation_linf =
                                std::max(
                                    metrics.selected_bulk_defect_split
                                        .p3_curve_collocation_interpolation_linf,
                                    std::abs(
                                        p3_curve_value(panel, local_s)
                                        - exact_value));

                        const double exact_normal =
                            transformed_exact_gradient(
                                rigid_transform, point).dot(normal)
                            - rhs_data.compatibility_mean_removed;
                        const double interpolated_normal =
                            geometry2d::panel_scalar(
                                iface,
                                panel,
                                exact_normal_by_q,
                                local_s);
                        metrics.selected_bulk_defect_split
                            .normal_collocation_interpolation_linf =
                                std::max(
                                    metrics.selected_bulk_defect_split
                                        .normal_collocation_interpolation_linf,
                                    std::abs(
                                        interpolated_normal - exact_normal));
                    }
                }
            }
            const PanelCenterCauchyResult2D solved_cauchy =
                laplace_panel_quadratic_center_cauchy_2d(
                    iface,
                    result.dirichlet_trace,
                    result.compatible_neumann_data,
                    zero_rhs_jump);
            const PanelCenterCauchyResult2D exact_cauchy =
                laplace_panel_quadratic_center_cauchy_2d(
                    iface,
                    exact_trace_by_q,
                    exact_normal_by_q,
                    zero_rhs_jump);
            const auto make_callback_cauchy =
                [&](bool exact_value_jump,
                    bool exact_normal_jump) {
                    const PanelCenterJumpEvaluator2D evaluator =
                        [&, exact_value_jump, exact_normal_jump](
                            int panel,
                            double local_s,
                            Eigen::Vector2d point,
                            Eigen::Vector2d normal,
                            double& value_jump,
                            double& normal_jump,
                            double& rhs_jump) {
                            value_jump =
                                exact_value_jump
                                    ? transformed_exact_solution(
                                          rigid_transform, point)
                                          - exact_trace_mean
                                    : geometry2d::panel_scalar(
                                          iface,
                                          panel,
                                          exact_trace_by_q,
                                          local_s);
                            normal_jump =
                                exact_normal_jump
                                    ? transformed_exact_gradient(
                                          rigid_transform, point)
                                              .dot(normal)
                                          - rhs_data
                                                .compatibility_mean_removed
                                    : geometry2d::panel_scalar(
                                          iface,
                                          panel,
                                          exact_normal_by_q,
                                          local_s);
                            rhs_jump = 0.0;
                            return true;
                        };
                    return laplace_panel_quadratic_center_cauchy_2d(
                        iface,
                        exact_trace,
                        rhs_data.compatible_neumann_data,
                        zero_rhs_jump,
                        0.0,
                        evaluator);
                };
            const PanelCenterCauchyResult2D
                exact_value_callback_cauchy =
                    make_callback_cauchy(true, false);
            const PanelCenterCauchyResult2D
                exact_normal_callback_cauchy =
                    make_callback_cauchy(false, true);
            const PanelCenterCauchyResult2D
                exact_value_normal_callback_cauchy =
                    make_callback_cauchy(true, true);
            const auto make_reconstructed_value_cauchy =
                [&](bool use_c1_hermite) {
                    const PanelCenterJumpEvaluator2D evaluator =
                        [&, use_c1_hermite](
                            int panel,
                            double local_s,
                            Eigen::Vector2d,
                            Eigen::Vector2d,
                            double& value_jump,
                            double& normal_jump,
                            double& rhs_jump) {
                            value_jump = use_c1_hermite
                                ? c1_hermite_value(panel, local_s)
                                : p3_curve_value(panel, local_s);
                            normal_jump = geometry2d::panel_scalar(
                                iface,
                                panel,
                                exact_normal_by_q,
                                local_s);
                            rhs_jump = 0.0;
                            return true;
                        };
                    return laplace_panel_quadratic_center_cauchy_2d(
                        iface,
                        exact_trace_by_q,
                        exact_normal_by_q,
                        zero_rhs_jump,
                        0.0,
                        evaluator);
                };
            const PanelCenterCauchyResult2D
                c1_hermite_value_cauchy =
                    make_reconstructed_value_cauchy(true);
            const PanelCenterCauchyResult2D
                p3_curve_value_cauchy =
                    make_reconstructed_value_cauchy(false);
            // Causal counterfactuals: replace only the value samples used by
            // the Cauchy fit on selected panels. All other inputs and routing
            // are kept identical to the P2 diagnostic spread.
            const auto make_panel_exact_value_cauchy =
                [&](std::initializer_list<int> exact_panels) {
                    const std::vector<int> selected_panels(
                        exact_panels.begin(), exact_panels.end());
                    const PanelCenterJumpEvaluator2D evaluator =
                        [&, selected_panels](
                            int panel,
                            double local_s,
                            Eigen::Vector2d point,
                            Eigen::Vector2d,
                            double& value_jump,
                            double& normal_jump,
                            double& rhs_jump) {
                            const bool use_exact =
                                std::find(
                                    selected_panels.begin(),
                                    selected_panels.end(),
                                    panel)
                                != selected_panels.end();
                            value_jump = use_exact
                                ? transformed_exact_solution(
                                      rigid_transform, point)
                                      - exact_trace_mean
                                : geometry2d::panel_scalar(
                                      iface,
                                      panel,
                                      exact_trace_by_q,
                                      local_s);
                            normal_jump = geometry2d::panel_scalar(
                                iface,
                                panel,
                                exact_normal_by_q,
                                local_s);
                            rhs_jump = 0.0;
                            return true;
                        };
                    return laplace_panel_quadratic_center_cauchy_2d(
                        iface,
                        exact_trace_by_q,
                        exact_normal_by_q,
                        zero_rhs_jump,
                        0.0,
                        evaluator);
                };
            const PanelCenterCauchyResult2D panel11_exact_value_cauchy =
                make_panel_exact_value_cauchy({11});
            const PanelCenterCauchyResult2D panel12_exact_value_cauchy =
                make_panel_exact_value_cauchy({12});
            const PanelCenterCauchyResult2D panel11_12_exact_value_cauchy =
                make_panel_exact_value_cauchy({11, 12});
            const PanelCenterCauchyResult2D panel61_62_exact_value_cauchy =
                make_panel_exact_value_cauchy({61, 62});
            const PanelCenterJumpEvaluator2D
                panel_node_normal_evaluator =
                    [&](int panel,
                        double local_s,
                        Eigen::Vector2d,
                        Eigen::Vector2d,
                        double& value_jump,
                        double& normal_jump,
                        double& rhs_jump) {
                        value_jump =
                            geometry2d::panel_scalar(
                                iface,
                                panel,
                                exact_trace_by_q,
                                local_s);
                        double shape[3];
                        geometry2d::p2_shape(local_s, shape);
                        normal_jump = 0.0;
                        for (int local = 0; local < 3; ++local) {
                            const int q =
                                iface.point_index(panel, local);
                            const Eigen::Vector2d point =
                                iface.points().row(q).transpose();
                            const Eigen::Vector2d panel_normal =
                                geometry2d::panel_normal(
                                    iface,
                                    panel,
                                    geometry2d::kP2NodeS[
                                        static_cast<std::size_t>(
                                            local)]);
                            normal_jump +=
                                shape[local]
                                * (transformed_exact_gradient(
                                       rigid_transform, point)
                                           .dot(panel_normal)
                                   - rhs_data
                                         .compatibility_mean_removed);
                        }
                        rhs_jump = 0.0;
                        return true;
                    };
            const PanelCenterCauchyResult2D
                panel_node_normal_callback_cauchy =
                    laplace_panel_quadratic_center_cauchy_2d(
                        iface,
                        exact_trace_by_q,
                        exact_normal_by_q,
                        zero_rhs_jump,
                        0.0,
                        panel_node_normal_evaluator);
            const auto evaluate_cauchy_result =
                [](const PanelCenterCauchyResult2D& cauchy,
                   int center,
                   const Eigen::Vector2d& point) {
                    const Eigen::Vector2d displacement =
                        point
                        - cauchy.centers.row(center).transpose();
                    return cauchy.C[center]
                        + cauchy.Cx[center] * displacement.x()
                        + cauchy.Cy[center] * displacement.y()
                        + 0.5
                            * cauchy.Cxx[center]
                            * displacement.x() * displacement.x()
                        + cauchy.Cxy[center]
                            * displacement.x() * displacement.y()
                        + 0.5
                            * cauchy.Cyy[center]
                            * displacement.y() * displacement.y();
                };

            if (geometry == "flower" && n == 64
                && iface.num_panels() == 63) {
                for (const std::array<int, 2> pair :
                     std::array<std::array<int, 2>, 2>{{
                         {{11, 12}}, {{61, 62}}}}) {
                    const int left_panel = pair[0];
                    const int right_panel = pair[1];
                    const int q = iface.point_index(left_panel, 2);
                    const Eigen::Vector2d point =
                        iface.points().row(q).transpose();
                    const int left_center =
                        4 * left_panel
                        + static_cast<int>(
                            geometry2d::kP2CenterS.size()) - 1;
                    const int right_center = 4 * right_panel;
                    const double exact_left = evaluate_cauchy_result(
                        exact_cauchy, left_center, point);
                    const double exact_right = evaluate_cauchy_result(
                        exact_cauchy, right_center, point);
                    const double solved_left = evaluate_cauchy_result(
                        solved_cauchy, left_center, point);
                    const double solved_right = evaluate_cauchy_result(
                        solved_cauchy, right_center, point);
                    std::cout
                        << "    n64_adjacent_panel_cauchy_gap panels="
                        << left_panel << '/' << right_panel
                        << " q=" << q << " centers="
                        << left_center << '/' << right_center
                        << " exact(left/right/gap)="
                        << exact_left << '/' << exact_right << '/'
                        << exact_right - exact_left
                        << " solved(left/right/gap)="
                        << solved_left << '/' << solved_right << '/'
                        << solved_right - solved_left << '\n';
                }
            }

            // Split the already-observed bulk-grid error by the support of
            // the discrete defect.  Let U be the exact piecewise grid field
            // (manufactured harmonic solution inside and zero outside), S_h
            // the P2 spread RHS, and L_h the ordinary five-point Laplacian.
            // Since the production bulk solve is
            //
            //     L_h u_h = -S_h,
            //
            // its error satisfies
            //
            //     L_h (u_h - U) = -(S_h + L_h U).
            //
            // At every node, first subtract the ordinary five-point
            // truncation of the smooth branch continued through that node's
            // whole stencil (the manufactured branch inside, zero outside).
            // The remaining defect is supported only on stencils crossing
            // the interface and measures spread/interface closure.  Solving
            // the two defect fields with the same FFT inverse gives an exact
            // linear decomposition of the bulk-grid error.
            std::vector<LaplaceJumpData2D> diagnostic_jumps(
                static_cast<std::size_t>(iface.num_points()));
            for (LaplaceJumpData2D& jump : diagnostic_jumps) {
                jump.u_jump = 0.0;
                jump.un_jump = 0.0;
                jump.rhs_derivs = Eigen::VectorXd::Zero(1);
            }
            for (int a = 0; a < nq; ++a) {
                const int q =
                    solver.active_interface_points()[
                        static_cast<std::size_t>(a)];
                LaplaceJumpData2D& jump =
                    diagnostic_jumps[static_cast<std::size_t>(q)];
                jump.u_jump = exact_trace[a];
                jump.un_jump =
                    rhs_data.compatible_neumann_data[a];
            }
            LaplaceQuadraticPanelCenterSpread2D diagnostic_spread(
                solver.grid_pair(),
                0.0,
                options.correction_method,
                options.restrict_stencil_radius,
                {},
                {},
                spread_mode,
                options.cubic_harmonic_spread);
            Eigen::VectorXd diagnostic_spread_rhs =
                Eigen::VectorXd::Zero(grid.num_dofs());
            const LaplaceSpreadResult2D diagnostic_spread_result =
                diagnostic_spread.apply(
                    diagnostic_jumps, diagnostic_spread_rhs);

            Eigen::VectorXd analytic_piecewise_grid =
                Eigen::VectorXd::Zero(grid.num_dofs());
            for (int node = 0; node < grid.num_dofs(); ++node) {
                if (solver.grid_pair().domain_label(node) <= 0)
                    continue;
                const std::array<double, 2> coordinate =
                    grid.coord(node);
                analytic_piecewise_grid[node] =
                    transformed_exact_solution(
                        rigid_transform,
                        Eigen::Vector2d(
                            coordinate[0], coordinate[1]))
                    - exact_trace_mean;
            }

            Eigen::VectorXd exact_nodal_spread_rhs =
                Eigen::VectorXd::Zero(grid.num_dofs());
            Eigen::VectorXd analytic_quadratic_spread_rhs =
                Eigen::VectorXd::Zero(grid.num_dofs());
            Eigen::VectorXd exact_value_callback_spread_rhs =
                Eigen::VectorXd::Zero(grid.num_dofs());
            Eigen::VectorXd exact_normal_callback_spread_rhs =
                Eigen::VectorXd::Zero(grid.num_dofs());
            Eigen::VectorXd panel_node_normal_callback_spread_rhs =
                Eigen::VectorXd::Zero(grid.num_dofs());
            Eigen::VectorXd exact_value_normal_callback_spread_rhs =
                Eigen::VectorXd::Zero(grid.num_dofs());
            Eigen::VectorXd hard_owner_reconstructed_spread_rhs =
                Eigen::VectorXd::Zero(grid.num_dofs());
            Eigen::VectorXd blended_owner_spread_rhs =
                Eigen::VectorXd::Zero(grid.num_dofs());
            Eigen::VectorXd c1_hermite_value_spread_rhs =
                Eigen::VectorXd::Zero(grid.num_dofs());
            Eigen::VectorXd p3_curve_value_spread_rhs =
                Eigen::VectorXd::Zero(grid.num_dofs());
            Eigen::VectorXd panel11_exact_value_spread_rhs =
                Eigen::VectorXd::Zero(grid.num_dofs());
            Eigen::VectorXd panel12_exact_value_spread_rhs =
                Eigen::VectorXd::Zero(grid.num_dofs());
            Eigen::VectorXd panel11_12_exact_value_spread_rhs =
                Eigen::VectorXd::Zero(grid.num_dofs());
            Eigen::VectorXd panel61_62_exact_value_spread_rhs =
                Eigen::VectorXd::Zero(grid.num_dofs());
            std::vector<double> crossing_owner_switch_distances;
            std::vector<double> crossing_owner_adjacent_gaps;
            const std::array<int, 2> grid_dims = grid.dof_dims();
            const double inv_h2 = 1.0 / (h * h);
            for (int j = 1; j < grid_dims[1] - 1; ++j) {
                for (int i = 1; i < grid_dims[0] - 1; ++i) {
                    const int node = grid.index(i, j);
                    const int phase =
                        solver.grid_pair().domain_label(node) > 0
                            ? 1 : 0;
                    const std::array<int, 4> neighbors =
                        grid.neighbors(node);
                    for (const int neighbor : neighbors) {
                        const int neighbor_phase =
                            solver.grid_pair().domain_label(neighbor) > 0
                                ? 1 : 0;
                        if (neighbor_phase == phase)
                            continue;
                        const P2CrossingOwner2D owner =
                            solver.grid_pair().p2_crossing_owner_between(
                                node, neighbor);
                        if (owner.center_index < 0
                            || owner.center_index
                                   >= static_cast<int>(
                                       diagnostic_spread_result
                                           .correction_polys.size())) {
                            throw std::runtime_error(
                                "selected bulk split found an invalid "
                                "spread crossing owner");
                        }
                        const std::array<double, 2> coordinate =
                            grid.coord(neighbor);
                        const Eigen::Vector2d point(
                            coordinate[0], coordinate[1]);
                        const int centers_per_panel =
                            static_cast<int>(
                                geometry2d::kP2CenterS.size());
                        const int first_panel_center =
                            centers_per_panel * owner.panel_index;
                        if (owner.panel_index < 0
                            || first_panel_center < 0
                            || first_panel_center + centers_per_panel
                                   > static_cast<int>(
                                       diagnostic_spread_result
                                           .correction_polys.size())) {
                            throw std::runtime_error(
                                "selected bulk split found an invalid "
                                "spread crossing panel");
                        }
                        const double hard_owner_value =
                            evaluate_taylor_poly_2d(
                                diagnostic_spread_result
                                    .correction_polys[
                                        static_cast<std::size_t>(
                                            owner.center_index)],
                                point);
                        int blend_left = 0;
                        int blend_right = 0;
                        double blend_weight = 0.0;
                        if (owner.local_s
                            <= geometry2d::kP2CenterS.front()) {
                            blend_left = blend_right = 0;
                        } else if (owner.local_s
                                   >= geometry2d::kP2CenterS.back()) {
                            blend_left = blend_right =
                                centers_per_panel - 1;
                        } else {
                            for (int local = 0;
                                 local + 1 < centers_per_panel;
                                 ++local) {
                                const double left_s =
                                    geometry2d::kP2CenterS[
                                        static_cast<std::size_t>(local)];
                                const double right_s =
                                    geometry2d::kP2CenterS[
                                        static_cast<std::size_t>(local + 1)];
                                if (owner.local_s <= right_s) {
                                    blend_left = local;
                                    blend_right = local + 1;
                                    blend_weight =
                                        (owner.local_s - left_s)
                                        / (right_s - left_s);
                                    break;
                                }
                            }
                        }
                        const double blend_left_value =
                            evaluate_taylor_poly_2d(
                                diagnostic_spread_result
                                    .correction_polys[
                                        static_cast<std::size_t>(
                                            first_panel_center
                                            + blend_left)],
                                point);
                        const double blend_right_value =
                            evaluate_taylor_poly_2d(
                                diagnostic_spread_result
                                    .correction_polys[
                                        static_cast<std::size_t>(
                                            first_panel_center
                                            + blend_right)],
                                point);
                        const double blended_owner_value =
                            (1.0 - blend_weight) * blend_left_value
                            + blend_weight * blend_right_value;
                        const double exact_jump =
                            transformed_exact_solution(
                                rigid_transform, point)
                            - exact_trace_mean;
                        const Eigen::Vector2d center =
                            diagnostic_spread_result
                                .correction_polys[
                                    static_cast<std::size_t>(
                                        owner.center_index)]
                                .center;
                        const Eigen::Vector2d displacement =
                            point - center;
                        const double center_value =
                            transformed_exact_solution(
                                rigid_transform, center)
                            - exact_trace_mean;
                        const Eigen::Vector2d center_gradient =
                            transformed_exact_gradient(
                                rigid_transform, center);
                        const Eigen::Matrix2d center_hessian =
                            transformed_exact_hessian(
                                rigid_transform, center);
                        const double analytic_quadratic =
                            center_value
                            + center_gradient.dot(displacement)
                            + 0.5
                                * displacement.dot(
                                    center_hessian * displacement);
                        const double side_scale =
                            static_cast<double>(
                                phase - neighbor_phase);
                        hard_owner_reconstructed_spread_rhs[node] +=
                            side_scale * hard_owner_value * inv_h2;
                        blended_owner_spread_rhs[node] +=
                            side_scale * blended_owner_value * inv_h2;

                        ++metrics.selected_bulk_defect_split
                              .crossing_owner_edges;
                        if (owner.status
                            == P2CrossingOwnerStatus2D::
                                   ExactIntersection) {
                            SelectedBulkDefectSplitDiagnostic& split =
                                metrics.selected_bulk_defect_split;
                            ++split.crossing_owner_exact_edges;
                            if (owner.exact_intersection_count > 1) {
                                ++split
                                      .crossing_owner_multiple_intersection_edges;
                            }
                            const std::array<double, 3> thresholds = {
                                -0.5, 0.0, 0.5};
                            int nearest_threshold = 0;
                            double switch_distance =
                                std::abs(owner.local_s - thresholds[0]);
                            for (int threshold = 1;
                                 threshold < 3;
                                 ++threshold) {
                                const double candidate_distance =
                                    std::abs(
                                        owner.local_s
                                        - thresholds[
                                            static_cast<std::size_t>(
                                                threshold)]);
                                if (candidate_distance < switch_distance) {
                                    switch_distance = candidate_distance;
                                    nearest_threshold = threshold;
                                }
                            }
                            crossing_owner_switch_distances.push_back(
                                switch_distance);
                            if (switch_distance < 0.01)
                                ++split.crossing_owner_switch_distance_lt_001;
                            if (switch_distance < 0.05)
                                ++split.crossing_owner_switch_distance_lt_005;
                            const double threshold_left_value =
                                evaluate_taylor_poly_2d(
                                    diagnostic_spread_result
                                        .correction_polys[
                                            static_cast<std::size_t>(
                                                first_panel_center
                                                + nearest_threshold)],
                                    point);
                            const double threshold_right_value =
                                evaluate_taylor_poly_2d(
                                    diagnostic_spread_result
                                        .correction_polys[
                                            static_cast<std::size_t>(
                                                first_panel_center
                                                + nearest_threshold + 1)],
                                    point);
                            const double adjacent_gap = std::abs(
                                threshold_right_value
                                - threshold_left_value);
                            crossing_owner_adjacent_gaps.push_back(
                                adjacent_gap);

                            bool hotspot_panel = false;
                            for (int local = 0; local < 3; ++local) {
                                hotspot_panel =
                                    hotspot_panel
                                    || iface.point_index(
                                           owner.panel_index, local)
                                           == metrics
                                                  .global_linf_nearest_trace_q;
                            }
                            if (hotspot_panel) {
                                ++split.hotspot_panel_crossing_owner_edges;
                                if (!std::isfinite(
                                        split
                                            .hotspot_panel_switch_distance_min)
                                    || switch_distance
                                           < split
                                                 .hotspot_panel_switch_distance_min) {
                                    split
                                        .hotspot_panel_switch_distance_min =
                                        switch_distance;
                                }
                                split.hotspot_panel_adjacent_gap_linf =
                                    std::max(
                                        split
                                            .hotspot_panel_adjacent_gap_linf,
                                        adjacent_gap);
                            }
                        }
                        exact_nodal_spread_rhs[node] +=
                            side_scale * exact_jump * inv_h2;
                        analytic_quadratic_spread_rhs[node] +=
                            side_scale
                            * analytic_quadratic * inv_h2;
                        exact_value_callback_spread_rhs[node] +=
                            side_scale
                            * evaluate_cauchy_result(
                                  exact_value_callback_cauchy,
                                  owner.center_index,
                                  point)
                            * inv_h2;
                        exact_normal_callback_spread_rhs[node] +=
                            side_scale
                            * evaluate_cauchy_result(
                                  exact_normal_callback_cauchy,
                                  owner.center_index,
                                  point)
                            * inv_h2;
                        panel_node_normal_callback_spread_rhs[node] +=
                            side_scale
                            * evaluate_cauchy_result(
                                  panel_node_normal_callback_cauchy,
                                  owner.center_index,
                                  point)
                            * inv_h2;
                        exact_value_normal_callback_spread_rhs[node] +=
                            side_scale
                            * evaluate_cauchy_result(
                                  exact_value_normal_callback_cauchy,
                                  owner.center_index,
                                  point)
                            * inv_h2;
                        c1_hermite_value_spread_rhs[node] +=
                            side_scale
                            * evaluate_cauchy_result(
                                  c1_hermite_value_cauchy,
                                  owner.center_index,
                                  point)
                            * inv_h2;
                        p3_curve_value_spread_rhs[node] +=
                            side_scale
                            * evaluate_cauchy_result(
                                  p3_curve_value_cauchy,
                                  owner.center_index,
                                  point)
                            * inv_h2;
                        panel11_exact_value_spread_rhs[node] +=
                            side_scale
                            * evaluate_cauchy_result(
                                  panel11_exact_value_cauchy,
                                  owner.center_index,
                                  point)
                            * inv_h2;
                        panel12_exact_value_spread_rhs[node] +=
                            side_scale
                            * evaluate_cauchy_result(
                                  panel12_exact_value_cauchy,
                                  owner.center_index,
                                  point)
                            * inv_h2;
                        panel11_12_exact_value_spread_rhs[node] +=
                            side_scale
                            * evaluate_cauchy_result(
                                  panel11_12_exact_value_cauchy,
                                  owner.center_index,
                                  point)
                            * inv_h2;
                        panel61_62_exact_value_spread_rhs[node] +=
                            side_scale
                            * evaluate_cauchy_result(
                                  panel61_62_exact_value_cauchy,
                                  owner.center_index,
                                  point)
                            * inv_h2;
                    }
                }
            }

            const auto sorted_quantile = [](
                std::vector<double>& values,
                double probability) {
                if (values.empty())
                    return std::numeric_limits<double>::quiet_NaN();
                std::sort(values.begin(), values.end());
                const std::size_t index = static_cast<std::size_t>(
                    std::floor(
                        probability
                        * static_cast<double>(values.size() - 1)));
                return values[index];
            };
            SelectedBulkDefectSplitDiagnostic& crossing_split =
                metrics.selected_bulk_defect_split;
            if (!crossing_owner_switch_distances.empty()) {
                crossing_split.crossing_owner_switch_distance_min =
                    sorted_quantile(
                        crossing_owner_switch_distances, 0.0);
                crossing_split.crossing_owner_switch_distance_p10 =
                    sorted_quantile(
                        crossing_owner_switch_distances, 0.10);
                crossing_split.crossing_owner_switch_distance_median =
                    sorted_quantile(
                        crossing_owner_switch_distances, 0.50);
            }
            if (!crossing_owner_adjacent_gaps.empty()) {
                crossing_split.crossing_owner_adjacent_gap_p95 =
                    sorted_quantile(
                        crossing_owner_adjacent_gaps, 0.95);
                crossing_split.crossing_owner_adjacent_gap_linf =
                    sorted_quantile(
                        crossing_owner_adjacent_gaps, 1.0);
            }

            Eigen::VectorXd discrete_laplacian =
                Eigen::VectorXd::Zero(grid.num_dofs());
            Eigen::VectorXd irregular_defect =
                Eigen::VectorXd::Zero(grid.num_dofs());
            Eigen::VectorXd regular_defect =
                Eigen::VectorXd::Zero(grid.num_dofs());
            for (int j = 1; j < grid_dims[1] - 1; ++j) {
                for (int i = 1; i < grid_dims[0] - 1; ++i) {
                    const int node = grid.index(i, j);
                    const std::array<int, 4> neighbors =
                        grid.neighbors(node);
                    discrete_laplacian[node] =
                        (analytic_piecewise_grid[neighbors[0]]
                         + analytic_piecewise_grid[neighbors[1]]
                         + analytic_piecewise_grid[neighbors[2]]
                         + analytic_piecewise_grid[neighbors[3]]
                         - 4.0 * analytic_piecewise_grid[node])
                        * inv_h2;
                    const int phase =
                        solver.grid_pair().domain_label(node) > 0
                            ? 1 : 0;
                    double smooth_branch_laplacian = 0.0;
                    if (phase == 1) {
                        double neighbor_sum = 0.0;
                        for (const int neighbor : neighbors) {
                            const std::array<double, 2> coordinate =
                                grid.coord(neighbor);
                            neighbor_sum +=
                                transformed_exact_solution(
                                    rigid_transform,
                                    Eigen::Vector2d(
                                        coordinate[0], coordinate[1]))
                                - exact_trace_mean;
                        }
                        const std::array<double, 2> coordinate =
                            grid.coord(node);
                        const double center_value =
                            transformed_exact_solution(
                                rigid_transform,
                                Eigen::Vector2d(
                                    coordinate[0], coordinate[1]))
                            - exact_trace_mean;
                        smooth_branch_laplacian =
                            (neighbor_sum - 4.0 * center_value) * inv_h2;
                    }
                    bool irregular = false;
                    for (const int neighbor : neighbors) {
                        const int neighbor_phase =
                            solver.grid_pair().domain_label(neighbor) > 0
                                ? 1 : 0;
                        irregular =
                            irregular || neighbor_phase != phase;
                    }
                    const double defect =
                        diagnostic_spread_rhs[node]
                        + discrete_laplacian[node];
                    regular_defect[node] =
                        smooth_branch_laplacian;
                    if (irregular) {
                        irregular_defect[node] =
                            defect - smooth_branch_laplacian;
                        ++metrics.selected_bulk_defect_split
                              .irregular_nodes;
                    } else {
                        ++metrics.selected_bulk_defect_split.regular_nodes;
                    }
                }
            }

            const Eigen::VectorXd blended_owner_irregular_defect =
                blended_owner_spread_rhs
                + discrete_laplacian
                - regular_defect;
            const Eigen::VectorXd blended_minus_hard_defect =
                blended_owner_irregular_defect
                - irregular_defect;
            const Eigen::VectorXd c1_hermite_irregular_defect =
                c1_hermite_value_spread_rhs
                + discrete_laplacian
                - regular_defect;
            const Eigen::VectorXd p3_curve_irregular_defect =
                p3_curve_value_spread_rhs
                + discrete_laplacian
                - regular_defect;
            const Eigen::VectorXd panel11_exact_value_irregular_defect =
                panel11_exact_value_spread_rhs
                + discrete_laplacian
                - regular_defect;
            const Eigen::VectorXd panel12_exact_value_irregular_defect =
                panel12_exact_value_spread_rhs
                + discrete_laplacian
                - regular_defect;
            const Eigen::VectorXd panel11_12_exact_value_irregular_defect =
                panel11_12_exact_value_spread_rhs
                + discrete_laplacian
                - regular_defect;
            const Eigen::VectorXd panel61_62_exact_value_irregular_defect =
                panel61_62_exact_value_spread_rhs
                + discrete_laplacian
                - regular_defect;

            const Eigen::VectorXd cauchy_fit_defect =
                diagnostic_spread_rhs
                - analytic_quadratic_spread_rhs;
            const Eigen::VectorXd value_jump_input_defect =
                diagnostic_spread_rhs
                - exact_value_callback_spread_rhs;
            const Eigen::VectorXd normal_jump_input_defect =
                diagnostic_spread_rhs
                - exact_normal_callback_spread_rhs;
            const Eigen::VectorXd normal_node_frame_defect =
                diagnostic_spread_rhs
                - panel_node_normal_callback_spread_rhs;
            const Eigen::VectorXd
                normal_scalar_interpolation_defect =
                    panel_node_normal_callback_spread_rhs
                    - exact_normal_callback_spread_rhs;
            const Eigen::VectorXd normal_input_source_closure =
                normal_jump_input_defect
                - normal_node_frame_defect
                - normal_scalar_interpolation_defect;
            const Eigen::VectorXd
                cauchy_geometry_reconstruction_defect =
                    exact_value_normal_callback_spread_rhs
                    - analytic_quadratic_spread_rhs;
            const Eigen::VectorXd cauchy_input_source_closure =
                cauchy_fit_defect
                - value_jump_input_defect
                - normal_jump_input_defect
                - cauchy_geometry_reconstruction_defect;
            const Eigen::VectorXd taylor_remainder_defect =
                analytic_quadratic_spread_rhs
                - exact_nodal_spread_rhs;
            const Eigen::VectorXd irregular_source_split_closure =
                irregular_defect
                - cauchy_fit_defect
                - taylor_remainder_defect;

            LaplaceFftBulkSolverZfft2D diagnostic_bulk_solver(
                grid, ZfftBcType::Dirichlet, 0.0, 2);
            Eigen::VectorXd irregular_spread_bulk_error;
            Eigen::VectorXd irregular_cauchy_fit_bulk_error;
            Eigen::VectorXd value_jump_input_bulk_error;
            Eigen::VectorXd normal_jump_input_bulk_error;
            Eigen::VectorXd normal_node_frame_bulk_error;
            Eigen::VectorXd normal_scalar_interpolation_bulk_error;
            Eigen::VectorXd
                cauchy_geometry_reconstruction_bulk_error;
            Eigen::VectorXd quadratic_taylor_remainder_bulk_error;
            Eigen::VectorXd regular_five_point_bulk_error;
            Eigen::VectorXd blended_owner_irregular_bulk_error;
            Eigen::VectorXd blended_minus_hard_bulk_error;
            Eigen::VectorXd c1_hermite_irregular_bulk_error;
            Eigen::VectorXd p3_curve_irregular_bulk_error;
            Eigen::VectorXd panel11_exact_value_irregular_bulk_error;
            Eigen::VectorXd panel12_exact_value_irregular_bulk_error;
            Eigen::VectorXd panel11_12_exact_value_irregular_bulk_error;
            Eigen::VectorXd panel61_62_exact_value_irregular_bulk_error;
            diagnostic_bulk_solver.solve(
                -irregular_defect,
                irregular_spread_bulk_error);
            diagnostic_bulk_solver.solve(
                -cauchy_fit_defect,
                irregular_cauchy_fit_bulk_error);
            diagnostic_bulk_solver.solve(
                -value_jump_input_defect,
                value_jump_input_bulk_error);
            diagnostic_bulk_solver.solve(
                -normal_jump_input_defect,
                normal_jump_input_bulk_error);
            diagnostic_bulk_solver.solve(
                -normal_node_frame_defect,
                normal_node_frame_bulk_error);
            diagnostic_bulk_solver.solve(
                -normal_scalar_interpolation_defect,
                normal_scalar_interpolation_bulk_error);
            diagnostic_bulk_solver.solve(
                -cauchy_geometry_reconstruction_defect,
                cauchy_geometry_reconstruction_bulk_error);
            diagnostic_bulk_solver.solve(
                -taylor_remainder_defect,
                quadratic_taylor_remainder_bulk_error);
            diagnostic_bulk_solver.solve(
                -regular_defect,
                regular_five_point_bulk_error);
            diagnostic_bulk_solver.solve(
                -blended_owner_irregular_defect,
                blended_owner_irregular_bulk_error);
            diagnostic_bulk_solver.solve(
                -blended_minus_hard_defect,
                blended_minus_hard_bulk_error);
            diagnostic_bulk_solver.solve(
                -c1_hermite_irregular_defect,
                c1_hermite_irregular_bulk_error);
            diagnostic_bulk_solver.solve(
                -p3_curve_irregular_defect,
                p3_curve_irregular_bulk_error);
            diagnostic_bulk_solver.solve(
                -panel11_exact_value_irregular_defect,
                panel11_exact_value_irregular_bulk_error);
            diagnostic_bulk_solver.solve(
                -panel12_exact_value_irregular_defect,
                panel12_exact_value_irregular_bulk_error);
            diagnostic_bulk_solver.solve(
                -panel11_12_exact_value_irregular_defect,
                panel11_12_exact_value_irregular_bulk_error);
            diagnostic_bulk_solver.solve(
                -panel61_62_exact_value_irregular_defect,
                panel61_62_exact_value_irregular_bulk_error);
            const Eigen::VectorXd irregular_solution_split_closure =
                irregular_spread_bulk_error
                - irregular_cauchy_fit_bulk_error
                - quadratic_taylor_remainder_bulk_error;
            const Eigen::VectorXd cauchy_input_solution_closure =
                irregular_cauchy_fit_bulk_error
                - value_jump_input_bulk_error
                - normal_jump_input_bulk_error
                - cauchy_geometry_reconstruction_bulk_error;
            const Eigen::VectorXd normal_input_solution_closure =
                normal_jump_input_bulk_error
                - normal_node_frame_bulk_error
                - normal_scalar_interpolation_bulk_error;
            const Eigen::VectorXd actual_bulk_grid_error =
                exact_cauchy_field.u_bulk - analytic_piecewise_grid;
            const Eigen::VectorXd bulk_split_closure =
                actual_bulk_grid_error
                - irregular_spread_bulk_error
                - regular_five_point_bulk_error;
            const Eigen::VectorXd hard_owner_total_bulk_error =
                irregular_spread_bulk_error
                + regular_five_point_bulk_error;
            const Eigen::VectorXd blended_owner_total_bulk_error =
                blended_owner_irregular_bulk_error
                + regular_five_point_bulk_error;
            const Eigen::VectorXd c1_hermite_total_bulk_error =
                c1_hermite_irregular_bulk_error
                + regular_five_point_bulk_error;
            const Eigen::VectorXd p3_curve_total_bulk_error =
                p3_curve_irregular_bulk_error
                + regular_five_point_bulk_error;
            const Eigen::VectorXd panel11_exact_value_total_bulk_error =
                panel11_exact_value_irregular_bulk_error
                + regular_five_point_bulk_error;
            const Eigen::VectorXd panel12_exact_value_total_bulk_error =
                panel12_exact_value_irregular_bulk_error
                + regular_five_point_bulk_error;
            const Eigen::VectorXd panel11_12_exact_value_total_bulk_error =
                panel11_12_exact_value_irregular_bulk_error
                + regular_five_point_bulk_error;
            const Eigen::VectorXd panel61_62_exact_value_total_bulk_error =
                panel61_62_exact_value_irregular_bulk_error
                + regular_five_point_bulk_error;
            metrics.selected_bulk_defect_split.spread_rhs_linf =
                inf_norm(diagnostic_spread_rhs);
            metrics.selected_bulk_defect_split
                .crossing_owner_hard_reconstruction_linf =
                    inf_norm(
                        hard_owner_reconstructed_spread_rhs
                        - diagnostic_spread_rhs);
            metrics.selected_bulk_defect_split
                .crossing_owner_blend_rhs_delta_linf =
                    inf_norm(
                        blended_owner_spread_rhs
                        - diagnostic_spread_rhs);
            metrics.selected_bulk_defect_split
                .crossing_owner_blended_defect_linf =
                    inf_norm(blended_owner_irregular_defect);
            metrics.selected_bulk_defect_split
                .crossing_owner_hard_total_solution_linf =
                    inf_norm(hard_owner_total_bulk_error);
            metrics.selected_bulk_defect_split
                .crossing_owner_blended_solution_linf =
                    inf_norm(blended_owner_irregular_bulk_error);
            metrics.selected_bulk_defect_split
                .crossing_owner_blended_total_solution_linf =
                    inf_norm(blended_owner_total_bulk_error);
            metrics.selected_bulk_defect_split
                .crossing_owner_blend_solution_delta_linf =
                    inf_norm(blended_minus_hard_bulk_error);
            metrics.selected_bulk_defect_split
                .c1_hermite_spread_rhs_delta_linf =
                    inf_norm(
                        c1_hermite_value_spread_rhs
                        - diagnostic_spread_rhs);
            metrics.selected_bulk_defect_split
                .p3_curve_spread_rhs_delta_linf =
                    inf_norm(
                        p3_curve_value_spread_rhs
                        - diagnostic_spread_rhs);
            metrics.selected_bulk_defect_split
                .c1_hermite_irregular_defect_linf =
                    inf_norm(c1_hermite_irregular_defect);
            metrics.selected_bulk_defect_split
                .p3_curve_irregular_defect_linf =
                    inf_norm(p3_curve_irregular_defect);
            metrics.selected_bulk_defect_split
                .c1_hermite_irregular_solution_linf =
                    inf_norm(c1_hermite_irregular_bulk_error);
            metrics.selected_bulk_defect_split
                .p3_curve_irregular_solution_linf =
                    inf_norm(p3_curve_irregular_bulk_error);
            metrics.selected_bulk_defect_split
                .c1_hermite_total_solution_linf =
                    inf_norm(c1_hermite_total_bulk_error);
            metrics.selected_bulk_defect_split
                .p3_curve_total_solution_linf =
                    inf_norm(p3_curve_total_bulk_error);
            metrics.selected_bulk_defect_split.irregular_defect_linf =
                inf_norm(irregular_defect);
            metrics.selected_bulk_defect_split.regular_defect_linf =
                inf_norm(regular_defect);
            metrics.selected_bulk_defect_split.cauchy_fit_defect_linf =
                inf_norm(cauchy_fit_defect);
            metrics.selected_bulk_defect_split
                .value_jump_input_defect_linf =
                    inf_norm(value_jump_input_defect);
            metrics.selected_bulk_defect_split
                .normal_jump_input_defect_linf =
                    inf_norm(normal_jump_input_defect);
            metrics.selected_bulk_defect_split
                .normal_node_frame_defect_linf =
                    inf_norm(normal_node_frame_defect);
            metrics.selected_bulk_defect_split
                .normal_scalar_interpolation_defect_linf =
                    inf_norm(normal_scalar_interpolation_defect);
            metrics.selected_bulk_defect_split
                .normal_input_source_closure_linf =
                    inf_norm(normal_input_source_closure);
            metrics.selected_bulk_defect_split
                .cauchy_geometry_reconstruction_defect_linf =
                    inf_norm(
                        cauchy_geometry_reconstruction_defect);
            metrics.selected_bulk_defect_split
                .cauchy_input_source_closure_linf =
                    inf_norm(cauchy_input_source_closure);
            metrics.selected_bulk_defect_split
                .taylor_remainder_defect_linf =
                    inf_norm(taylor_remainder_defect);
            metrics.selected_bulk_defect_split
                .irregular_source_closure_linf =
                    inf_norm(irregular_source_split_closure);
            metrics.selected_bulk_defect_split.irregular_solution_linf =
                inf_norm(irregular_spread_bulk_error);
            metrics.selected_bulk_defect_split
                .cauchy_fit_solution_linf =
                    inf_norm(irregular_cauchy_fit_bulk_error);
            metrics.selected_bulk_defect_split
                .value_jump_input_solution_linf =
                    inf_norm(value_jump_input_bulk_error);
            metrics.selected_bulk_defect_split
                .normal_jump_input_solution_linf =
                    inf_norm(normal_jump_input_bulk_error);
            metrics.selected_bulk_defect_split
                .normal_node_frame_solution_linf =
                    inf_norm(normal_node_frame_bulk_error);
            metrics.selected_bulk_defect_split
                .normal_scalar_interpolation_solution_linf =
                    inf_norm(
                        normal_scalar_interpolation_bulk_error);
            metrics.selected_bulk_defect_split
                .normal_input_solution_closure_linf =
                    inf_norm(normal_input_solution_closure);
            metrics.selected_bulk_defect_split
                .cauchy_geometry_reconstruction_solution_linf =
                    inf_norm(
                        cauchy_geometry_reconstruction_bulk_error);
            metrics.selected_bulk_defect_split
                .cauchy_input_solution_closure_linf =
                    inf_norm(cauchy_input_solution_closure);
            metrics.selected_bulk_defect_split
                .taylor_remainder_solution_linf =
                    inf_norm(
                        quadratic_taylor_remainder_bulk_error);
            metrics.selected_bulk_defect_split
                .irregular_solution_closure_linf =
                    inf_norm(irregular_solution_split_closure);
            metrics.selected_bulk_defect_split.regular_solution_linf =
                inf_norm(regular_five_point_bulk_error);
            metrics.selected_bulk_defect_split.closure_linf =
                inf_norm(bulk_split_closure);
            metrics.selected_bulk_defect_split
                .exact_value_counterfactual_solution_linf =
                    inf_norm(
                        irregular_spread_bulk_error
                        - value_jump_input_bulk_error);
            metrics.selected_bulk_defect_split
                .exact_normal_counterfactual_solution_linf =
                    inf_norm(
                        irregular_spread_bulk_error
                        - normal_jump_input_bulk_error);
            metrics.selected_bulk_defect_split
                .exact_value_normal_counterfactual_solution_linf =
                    inf_norm(
                        irregular_spread_bulk_error
                        - value_jump_input_bulk_error
                        - normal_jump_input_bulk_error);

            if (bulk_dump_node >= 0) {
                const auto report_panel_counterfactual =
                    [&](const char* name,
                        const Eigen::VectorXd& field) {
                        int max_node = -1;
                        double max_abs = 0.0;
                        int interior_max_node = -1;
                        double interior_max_abs = 0.0;
                        for (int node = 0;
                             node < grid.num_dofs();
                             ++node) {
                            const double abs_value = std::abs(field[node]);
                            if (max_node < 0 || abs_value > max_abs) {
                                max_node = node;
                                max_abs = abs_value;
                            }
                            const std::array<double, 2> coordinate =
                                grid.coord(node);
                            const bool analytic_inside =
                                transformed_analytic_inside(
                                    geometry,
                                    rigid_transform,
                                    Eigen::Vector2d(
                                        coordinate[0], coordinate[1]));
                            if (analytic_inside
                                && (interior_max_node < 0
                                    || abs_value > interior_max_abs)) {
                                interior_max_node = node;
                                interior_max_abs = abs_value;
                            }
                        }
                        std::cout
                            << "    bulk_panel_value_counterfactual name="
                            << name
                            << " target_node=" << bulk_dump_node
                            << " target_error=" << field[bulk_dump_node]
                            << " delta_from_p2="
                            << field[bulk_dump_node]
                                   - hard_owner_total_bulk_error[
                                         bulk_dump_node]
                            << " linf_all=" << max_abs
                            << " max_node=" << max_node
                            << " linf_interior=" << interior_max_abs
                            << " interior_max_node=" << interior_max_node
                            << '\n';
                    };
                report_panel_counterfactual(
                    "p2", hard_owner_total_bulk_error);
                report_panel_counterfactual(
                    "panel11_exact_value",
                    panel11_exact_value_total_bulk_error);
                report_panel_counterfactual(
                    "panel12_exact_value",
                    panel12_exact_value_total_bulk_error);
                report_panel_counterfactual(
                    "panel11_12_exact_value",
                    panel11_12_exact_value_total_bulk_error);
                report_panel_counterfactual(
                    "panel61_62_exact_value",
                    panel61_62_exact_value_total_bulk_error);

                const int target_i = bulk_dump_node % grid_dims[0];
                const int target_j = bulk_dump_node / grid_dims[0];
                const std::array<double, 2> target_coordinate =
                    grid.coord(bulk_dump_node);
                std::cout
                    << "    bulk_neighborhood_header "
                    << "node,i,j,x,y,analytic_inside,discrete_inside,"
                    << "irregular,production_reported_error,"
                    << "production_piecewise_error,trace_response,"
                    << "exact_bulk_error,hard_total,bulk_closure,"
                    << "spread_rhs,discrete_laplacian,irregular_defect,"
                    << "regular_defect,cauchy_defect,value_defect,"
                    << "normal_defect,geometry_defect,taylor_defect,"
                    << "irregular_solution,cauchy_solution,value_solution,"
                    << "normal_solution,geometry_solution,taylor_solution,"
                    << "regular_solution\n";
                int neighborhood_large_50 = 0;
                int neighborhood_large_75 = 0;
                int neighborhood_large_90 = 0;
                int neighborhood_inside_count = 0;
                const double target_reported_error =
                    result.u_bulk[bulk_dump_node]
                    + bulk.shift
                    - transformed_exact_solution(
                        rigid_transform,
                        Eigen::Vector2d(
                            target_coordinate[0],
                            target_coordinate[1]));
                for (int j = std::max(1, target_j - 3);
                     j <= std::min(grid_dims[1] - 2, target_j + 3);
                     ++j) {
                    for (int i = std::max(1, target_i - 3);
                         i <= std::min(grid_dims[0] - 2, target_i + 3);
                         ++i) {
                        const int node = grid.index(i, j);
                        const std::array<double, 2> coordinate =
                            grid.coord(node);
                        const Eigen::Vector2d point(
                            coordinate[0], coordinate[1]);
                        const bool analytic_inside =
                            transformed_analytic_inside(
                                geometry, rigid_transform, point);
                        const bool discrete_inside =
                            solver.grid_pair().domain_label(node) > 0;
                        bool irregular = false;
                        for (const int neighbor : grid.neighbors(node)) {
                            irregular = irregular
                                || (solver.grid_pair().domain_label(neighbor)
                                        > 0)
                                   != discrete_inside;
                        }
                        const double production_piecewise_error =
                            result.u_bulk[node]
                            - analytic_piecewise_grid[node];
                        const double production_reported_error =
                            analytic_inside
                                ? result.u_bulk[node]
                                      + bulk.shift
                                      - transformed_exact_solution(
                                            rigid_transform, point)
                                : std::numeric_limits<double>::quiet_NaN();
                        if (analytic_inside) {
                            ++neighborhood_inside_count;
                            const double relative = std::abs(
                                production_reported_error
                                / target_reported_error);
                            neighborhood_large_50 += relative >= 0.50;
                            neighborhood_large_75 += relative >= 0.75;
                            neighborhood_large_90 += relative >= 0.90;
                        }
                        std::cout
                            << "    bulk_neighborhood_row "
                            << node << ',' << i << ',' << j << ','
                            << coordinate[0] << ',' << coordinate[1] << ','
                            << (analytic_inside ? 1 : 0) << ','
                            << (discrete_inside ? 1 : 0) << ','
                            << (irregular ? 1 : 0) << ','
                            << production_reported_error << ','
                            << production_piecewise_error << ','
                            << result.u_bulk[node]
                                   - exact_cauchy_field.u_bulk[node]
                            << ',' << actual_bulk_grid_error[node] << ','
                            << hard_owner_total_bulk_error[node] << ','
                            << bulk_split_closure[node] << ','
                            << diagnostic_spread_rhs[node] << ','
                            << discrete_laplacian[node] << ','
                            << irregular_defect[node] << ','
                            << regular_defect[node] << ','
                            << cauchy_fit_defect[node] << ','
                            << value_jump_input_defect[node] << ','
                            << normal_jump_input_defect[node] << ','
                            << cauchy_geometry_reconstruction_defect[node]
                            << ',' << taylor_remainder_defect[node] << ','
                            << irregular_spread_bulk_error[node] << ','
                            << irregular_cauchy_fit_bulk_error[node] << ','
                            << value_jump_input_bulk_error[node] << ','
                            << normal_jump_input_bulk_error[node] << ','
                            << cauchy_geometry_reconstruction_bulk_error[node]
                            << ','
                            << quadratic_taylor_remainder_bulk_error[node]
                            << ',' << regular_five_point_bulk_error[node]
                            << '\n';
                    }
                }
                std::cout
                    << "    bulk_neighborhood_shape inside="
                    << neighborhood_inside_count
                    << " count_abs_ge_50_75_90pct="
                    << neighborhood_large_50 << '/'
                    << neighborhood_large_75 << '/'
                    << neighborhood_large_90
                    << " target_production_error="
                    << target_reported_error
                    << " target_exact_bulk_error="
                    << actual_bulk_grid_error[bulk_dump_node]
                    << " target_trace_response_raw="
                    << result.u_bulk[bulk_dump_node]
                           - exact_cauchy_field.u_bulk[bulk_dump_node]
                    << '\n';

                // One FFT solve supplies the target row of the symmetric
                // discrete Green operator. Multiplying it by each source
                // defect gives that source node's signed contribution at the
                // selected target without hundreds of separate solves.
                Eigen::VectorXd target_impulse =
                    Eigen::VectorXd::Zero(grid.num_dofs());
                target_impulse[bulk_dump_node] = 1.0;
                Eigen::VectorXd target_green;
                diagnostic_bulk_solver.solve(target_impulse, target_green);
                struct SourceContribution {
                    int node = -1;
                    double total = 0.0;
                    double cauchy = 0.0;
                    double value = 0.0;
                    double normal = 0.0;
                    double geometry = 0.0;
                    double taylor = 0.0;
                    double regular = 0.0;
                };
                std::vector<SourceContribution> source_contributions;
                std::array<double, 5> radial_signed{{0, 0, 0, 0, 0}};
                std::array<double, 5> radial_absolute{{0, 0, 0, 0, 0}};
                for (int node = 0; node < grid.num_dofs(); ++node) {
                    if (std::abs(irregular_defect[node]) <= 1.0e-15
                        && std::abs(regular_defect[node]) <= 1.0e-15) {
                        continue;
                    }
                    SourceContribution contribution;
                    contribution.node = node;
                    contribution.total =
                        -irregular_defect[node] * target_green[node];
                    contribution.cauchy =
                        -cauchy_fit_defect[node] * target_green[node];
                    contribution.value =
                        -value_jump_input_defect[node] * target_green[node];
                    contribution.normal =
                        -normal_jump_input_defect[node] * target_green[node];
                    contribution.geometry =
                        -cauchy_geometry_reconstruction_defect[node]
                        * target_green[node];
                    contribution.taylor =
                        -taylor_remainder_defect[node] * target_green[node];
                    contribution.regular =
                        -regular_defect[node] * target_green[node];
                    source_contributions.push_back(contribution);
                    const std::array<double, 2> coordinate = grid.coord(node);
                    const double distance_over_h =
                        (Eigen::Vector2d(coordinate[0], coordinate[1])
                         - Eigen::Vector2d(
                             target_coordinate[0], target_coordinate[1]))
                            .norm()
                        / h;
                    int radial_bin = 4;
                    if (distance_over_h <= 2.0)
                        radial_bin = 0;
                    else if (distance_over_h <= 4.0)
                        radial_bin = 1;
                    else if (distance_over_h <= 8.0)
                        radial_bin = 2;
                    else if (distance_over_h <= 16.0)
                        radial_bin = 3;
                    radial_signed[static_cast<std::size_t>(radial_bin)] +=
                        contribution.total + contribution.regular;
                    radial_absolute[static_cast<std::size_t>(radial_bin)] +=
                        std::abs(contribution.total)
                        + std::abs(contribution.regular);
                }
                std::sort(
                    source_contributions.begin(),
                    source_contributions.end(),
                    [](const SourceContribution& left,
                       const SourceContribution& right) {
                        return std::abs(left.total + left.regular)
                             > std::abs(right.total + right.regular);
                    });
                std::cout
                    << "    bulk_source_radial bins_le_2_4_8_16_gt16h "
                    << "signed=";
                for (double value : radial_signed)
                    std::cout << value << '/';
                std::cout << " absolute=";
                for (double value : radial_absolute)
                    std::cout << value << '/';
                std::cout << " reconstructed_target="
                          << irregular_spread_bulk_error[bulk_dump_node]
                                 + regular_five_point_bulk_error[
                                       bulk_dump_node]
                          << '\n';
                const int source_limit = std::min(
                    30,
                    static_cast<int>(source_contributions.size()));
                for (int rank = 0; rank < source_limit; ++rank) {
                    const SourceContribution& contribution =
                        source_contributions[static_cast<std::size_t>(rank)];
                    const int node = contribution.node;
                    const std::array<double, 2> coordinate = grid.coord(node);
                    const int nearest_q =
                        solver.grid_pair().closest_interface_point(node);
                    int incident_panel0 = -1;
                    int incident_panel1 = -1;
                    for (int panel = 0;
                         panel < iface.num_panels();
                         ++panel) {
                        for (int local = 0; local < 3; ++local) {
                            if (iface.point_index(panel, local) != nearest_q)
                                continue;
                            if (incident_panel0 < 0)
                                incident_panel0 = panel;
                            else if (incident_panel1 < 0
                                     && panel != incident_panel0)
                                incident_panel1 = panel;
                        }
                    }
                    const double distance_over_h =
                        (Eigen::Vector2d(coordinate[0], coordinate[1])
                         - Eigen::Vector2d(
                             target_coordinate[0], target_coordinate[1]))
                            .norm()
                        / h;
                    std::cout
                        << "    bulk_source_contribution rank=" << rank + 1
                        << " node=" << node
                        << " ij=" << node % grid_dims[0] << '/'
                        << node / grid_dims[0]
                        << " xy=" << coordinate[0] << '/' << coordinate[1]
                        << " distance_over_h=" << distance_over_h
                        << " nearest_q=" << nearest_q
                        << " panels=" << incident_panel0 << '/'
                        << incident_panel1
                        << " source(irregular/regular)="
                        << irregular_defect[node] << '/'
                        << regular_defect[node]
                        << " target_contribution(total/cauchy/value/normal/geometry/taylor/regular)="
                        << contribution.total << '/'
                        << contribution.cauchy << '/'
                        << contribution.value << '/'
                        << contribution.normal << '/'
                        << contribution.geometry << '/'
                        << contribution.taylor << '/'
                        << contribution.regular
                        << '\n';
                }
                const auto report_panel_delta_sources =
                    [&](const char* name,
                        const Eigen::VectorXd& counterfactual_defect,
                        const Eigen::VectorXd& counterfactual_solution) {
                        std::vector<std::pair<int, double>> contributions;
                        double signed_sum = 0.0;
                        double absolute_sum = 0.0;
                        for (int node = 0;
                             node < grid.num_dofs();
                             ++node) {
                            const double delta_defect =
                                counterfactual_defect[node]
                                - irregular_defect[node];
                            if (std::abs(delta_defect) <= 1.0e-13)
                                continue;
                            const double target_contribution =
                                -delta_defect * target_green[node];
                            contributions.push_back(
                                {node, target_contribution});
                            signed_sum += target_contribution;
                            absolute_sum += std::abs(target_contribution);
                        }
                        std::sort(
                            contributions.begin(),
                            contributions.end(),
                            [](const auto& left, const auto& right) {
                                return std::abs(left.second)
                                     > std::abs(right.second);
                            });
                        std::cout
                            << "    bulk_panel_delta_source_summary name="
                            << name
                            << " nonzero_nodes=" << contributions.size()
                            << " signed_sum=" << signed_sum
                            << " absolute_sum=" << absolute_sum
                            << " direct_solution_delta="
                            << counterfactual_solution[bulk_dump_node]
                                   - irregular_spread_bulk_error[
                                         bulk_dump_node]
                            << '\n';
                        const int limit = std::min(
                            20, static_cast<int>(contributions.size()));
                        for (int rank = 0; rank < limit; ++rank) {
                            const int node = contributions[
                                static_cast<std::size_t>(rank)].first;
                            const std::array<double, 2> coordinate =
                                grid.coord(node);
                            const double delta_defect =
                                counterfactual_defect[node]
                                - irregular_defect[node];
                            const double distance_over_h =
                                (Eigen::Vector2d(
                                     coordinate[0], coordinate[1])
                                 - Eigen::Vector2d(
                                     target_coordinate[0],
                                     target_coordinate[1]))
                                    .norm()
                                / h;
                            std::cout
                                << "    bulk_panel_delta_source name="
                                << name
                                << " rank=" << rank + 1
                                << " node=" << node
                                << " ij=" << node % grid_dims[0] << '/'
                                << node / grid_dims[0]
                                << " xy=" << coordinate[0] << '/'
                                << coordinate[1]
                                << " distance_over_h=" << distance_over_h
                                << " delta_defect=" << delta_defect
                                << " target_green=" << target_green[node]
                                << " target_contribution="
                                << contributions[
                                       static_cast<std::size_t>(rank)].second
                                << '\n';
                        }
                    };
                report_panel_delta_sources(
                    "panel11_exact_value",
                    panel11_exact_value_irregular_defect,
                    panel11_exact_value_irregular_bulk_error);
                report_panel_delta_sources(
                    "panel12_exact_value",
                    panel12_exact_value_irregular_defect,
                    panel12_exact_value_irregular_bulk_error);
            }

            const auto evaluate_center_correction =
                [](const PanelCenterCauchyResult2D& cauchy,
                   int center,
                   const Eigen::Vector2d& point) {
                    if (center < 0 || center >= cauchy.centers.rows()) {
                        return std::numeric_limits<double>::quiet_NaN();
                    }
                    const Eigen::Vector2d d =
                        point - cauchy.centers.row(center).transpose();
                    return cauchy.C[center]
                        + cauchy.Cx[center] * d[0]
                        + cauchy.Cy[center] * d[1]
                        + 0.5 * cauchy.Cxx[center] * d[0] * d[0]
                        + cauchy.Cxy[center] * d[0] * d[1]
                        + 0.5 * cauchy.Cyy[center] * d[1] * d[1];
                };
            int selected_active_index = -1;
            for (int a = 0; a < nq; ++a) {
                if (solver.active_interface_points()[
                        static_cast<std::size_t>(a)]
                    == selected_stencil_q) {
                    selected_active_index = a;
                    break;
                }
            }
            if (selected_active_index < 0) {
                throw std::invalid_argument(
                    "KFBIM_HJET_STENCIL_Q is not an active interface point");
            }
            for (const auto& sample : diagnostics->grid_edge_samples) {
                if (sample.q != selected_stencil_q)
                    continue;
                SelectedGridEdgeSampleDiagnostic selected;
                selected.sample = sample;
                const Eigen::Vector2d& diagonal_point =
                    sample.grid_points[5];
                for (int owner = 0;
                     owner < sample.diagonal_owner_count;
                     ++owner) {
                    const int center =
                        sample.diagonal_owners[
                            static_cast<std::size_t>(owner)]
                            .center_index;
                    selected.solved_diagonal_corrections[
                        static_cast<std::size_t>(owner)] =
                            evaluate_center_correction(
                                solved_cauchy, center, diagonal_point);
                    selected.exact_diagonal_corrections[
                        static_cast<std::size_t>(owner)] =
                            evaluate_center_correction(
                                exact_cauchy, center, diagonal_point);
                }
                const double phase_scale =
                    sample.side == 0 ? 1.0 : -1.0;
                double ideal_interpolated = 0.0;
                double analytic_corrected = 0.0;
                double operator_corrected = 0.0;
                double irregular_spread_interpolated = 0.0;
                double irregular_cauchy_fit_interpolated = 0.0;
                double value_jump_input_interpolated = 0.0;
                double normal_jump_input_interpolated = 0.0;
                double normal_node_frame_interpolated = 0.0;
                double normal_scalar_interpolation_interpolated = 0.0;
                double normal_input_split_closure_interpolated = 0.0;
                double cauchy_geometry_reconstruction_interpolated =
                    0.0;
                double cauchy_input_split_closure_interpolated =
                    0.0;
                double quadratic_taylor_remainder_interpolated = 0.0;
                double irregular_split_closure_interpolated = 0.0;
                double regular_five_point_interpolated = 0.0;
                double bulk_split_closure_interpolated = 0.0;
                for (int row = 0; row < 6; ++row) {
                    const std::size_t r =
                        static_cast<std::size_t>(row);
                    const Eigen::Vector2d& point =
                        sample.grid_points[r];
                    const double weight =
                        sample.interpolation_weights[r];
                    const double true_jump =
                        transformed_exact_solution(
                            rigid_transform, point)
                        - exact_trace_mean;
                    const bool node_inside =
                        solver.grid_pair().domain_label(
                            sample.grid_nodes[r]) > 0;
                    const double analytic_raw =
                        node_inside ? true_jump : 0.0;
                    double correction = 0.0;
                    if (row >= 1 && row <= 4
                        && sample.cardinal_corrected[
                            static_cast<std::size_t>(row - 1)]) {
                        correction = evaluate_center_correction(
                            exact_cauchy,
                            sample.cardinal_owners[
                                static_cast<std::size_t>(row - 1)]
                                .center_index,
                            point);
                    } else if (row == 5
                               && sample.diagonal_corrected) {
                        for (int owner = 0;
                             owner < sample.diagonal_owner_count;
                             ++owner) {
                            correction +=
                                evaluate_center_correction(
                                    exact_cauchy,
                                    sample.diagonal_owners[
                                        static_cast<std::size_t>(
                                            owner)]
                                        .center_index,
                                    point);
                        }
                        correction /=
                            static_cast<double>(
                                sample.diagonal_owner_count);
                    }
                    const double applied_correction =
                        correction == 0.0
                            ? 0.0
                            : phase_scale * correction;
                    ideal_interpolated +=
                        weight
                        * (sample.side == 0 ? true_jump : 0.0);
                    analytic_corrected +=
                        weight * (analytic_raw + applied_correction);
                    operator_corrected +=
                        weight
                        * (exact_cauchy_field.u_bulk[
                               sample.grid_nodes[r]]
                           + applied_correction);
                    irregular_spread_interpolated +=
                        weight
                        * irregular_spread_bulk_error[
                              sample.grid_nodes[r]];
                    irregular_cauchy_fit_interpolated +=
                        weight
                        * irregular_cauchy_fit_bulk_error[
                              sample.grid_nodes[r]];
                    value_jump_input_interpolated +=
                        weight
                        * value_jump_input_bulk_error[
                              sample.grid_nodes[r]];
                    normal_jump_input_interpolated +=
                        weight
                        * normal_jump_input_bulk_error[
                              sample.grid_nodes[r]];
                    normal_node_frame_interpolated +=
                        weight
                        * normal_node_frame_bulk_error[
                              sample.grid_nodes[r]];
                    normal_scalar_interpolation_interpolated +=
                        weight
                        * normal_scalar_interpolation_bulk_error[
                              sample.grid_nodes[r]];
                    normal_input_split_closure_interpolated +=
                        weight
                        * normal_input_solution_closure[
                              sample.grid_nodes[r]];
                    cauchy_geometry_reconstruction_interpolated +=
                        weight
                        * cauchy_geometry_reconstruction_bulk_error[
                              sample.grid_nodes[r]];
                    cauchy_input_split_closure_interpolated +=
                        weight
                        * cauchy_input_solution_closure[
                              sample.grid_nodes[r]];
                    quadratic_taylor_remainder_interpolated +=
                        weight
                        * quadratic_taylor_remainder_bulk_error[
                              sample.grid_nodes[r]];
                    irregular_split_closure_interpolated +=
                        weight
                        * irregular_solution_split_closure[
                              sample.grid_nodes[r]];
                    regular_five_point_interpolated +=
                        weight
                        * regular_five_point_bulk_error[
                              sample.grid_nodes[r]];
                    bulk_split_closure_interpolated +=
                        weight
                        * bulk_split_closure[sample.grid_nodes[r]];
                }
                selected.analytic_target =
                    sample.side == 0
                        ? transformed_exact_solution(
                              rigid_transform, sample.query)
                              - exact_trace_mean
                        : 0.0;
                selected.analytic_corrected_sample =
                    analytic_corrected;
                selected.operator_corrected_sample =
                    operator_corrected;
                selected.smooth_interpolation_error =
                    ideal_interpolated - selected.analytic_target;
                selected.cauchy_correction_error =
                    analytic_corrected - ideal_interpolated;
                selected.bulk_grid_error =
                    operator_corrected - analytic_corrected;
                selected.irregular_spread_bulk_error =
                    irregular_spread_interpolated;
                selected.irregular_cauchy_fit_bulk_error =
                    irregular_cauchy_fit_interpolated;
                selected.value_jump_input_bulk_error =
                    value_jump_input_interpolated;
                selected.normal_jump_input_bulk_error =
                    normal_jump_input_interpolated;
                selected.normal_node_frame_bulk_error =
                    normal_node_frame_interpolated;
                selected.normal_scalar_interpolation_bulk_error =
                    normal_scalar_interpolation_interpolated;
                selected.normal_input_split_closure_error =
                    normal_input_split_closure_interpolated;
                selected.cauchy_geometry_reconstruction_bulk_error =
                    cauchy_geometry_reconstruction_interpolated;
                selected.cauchy_input_split_closure_error =
                    cauchy_input_split_closure_interpolated;
                selected.quadratic_taylor_remainder_bulk_error =
                    quadratic_taylor_remainder_interpolated;
                selected.irregular_split_closure_error =
                    irregular_split_closure_interpolated;
                selected.regular_five_point_bulk_error =
                    regular_five_point_interpolated;
                selected.bulk_split_closure_error =
                    bulk_split_closure_interpolated;
                selected.operator_sample_error =
                    operator_corrected - selected.analytic_target;
                metrics.selected_grid_edge_samples.push_back(
                    std::move(selected));
            }
            if (metrics.selected_grid_edge_samples.size() != 6) {
                throw std::runtime_error(
                    "selected shared-side quadratic diagnostic did not "
                    "find six normal samples");
            }
            std::array<const SelectedGridEdgeSampleDiagnostic*, 6>
                selected_by_side_layer{{
                    nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr}};
            for (const auto& selected :
                 metrics.selected_grid_edge_samples) {
                const int index =
                    selected.sample.side * 3
                    + selected.sample.layer;
                if (index < 0 || index >= 6) {
                    throw std::runtime_error(
                        "selected normal sample has an invalid side/layer");
                }
                selected_by_side_layer[
                    static_cast<std::size_t>(index)] = &selected;
            }
            Eigen::Matrix<double, 6, 4> design =
                Eigen::Matrix<double, 6, 4>::Zero();
            constexpr std::array<double, 3> normal_layers{{
                0.2, 0.6, 1.0}};
            for (int layer = 0; layer < 3; ++layer) {
                const double tau =
                    normal_layers[static_cast<std::size_t>(layer)];
                design(layer, 0) = 1.0;
                design(layer, 1) = -tau;
                design(layer, 2) = tau * tau;
                design(3 + layer, 0) = 1.0;
                design(3 + layer, 1) = tau;
                design(3 + layer, 3) = tau * tau;
            }
            const Eigen::Matrix<double, 4, 6> pseudo_inverse =
                design.colPivHouseholderQr().solve(
                    Eigen::Matrix<double, 6, 6>::Identity());
            const Eigen::Matrix<double, 1, 6> c0_weights =
                pseudo_inverse.row(0);
            SelectedJointFitDiagnostic joint;
            joint.q = selected_stencil_q;
            for (int side = 0; side < 2; ++side) {
                for (int layer = 0; layer < 3; ++layer) {
                    const int index = side * 3 + layer;
                    const auto* selected =
                        selected_by_side_layer[
                            static_cast<std::size_t>(index)];
                    if (selected == nullptr) {
                        throw std::runtime_error(
                            "selected normal sample is missing");
                    }
                    const double weight = c0_weights[index];
                    joint.exterior_smooth_error +=
                        weight
                        * selected->smooth_interpolation_error;
                    joint.exterior_cauchy_error +=
                        weight
                        * selected->cauchy_correction_error;
                    joint.exterior_bulk_error +=
                        weight * selected->bulk_grid_error;
                    joint.exterior_irregular_spread_error +=
                        weight
                        * selected->irregular_spread_bulk_error;
                    joint.exterior_irregular_cauchy_fit_error +=
                        weight
                        * selected->irregular_cauchy_fit_bulk_error;
                    joint.exterior_value_jump_input_error +=
                        weight
                        * selected->value_jump_input_bulk_error;
                    joint.exterior_normal_jump_input_error +=
                        weight
                        * selected->normal_jump_input_bulk_error;
                    joint.exterior_normal_node_frame_error +=
                        weight
                        * selected->normal_node_frame_bulk_error;
                    joint
                        .exterior_normal_scalar_interpolation_error +=
                            weight
                            * selected
                                  ->normal_scalar_interpolation_bulk_error;
                    joint
                        .exterior_normal_input_split_closure_error +=
                            weight
                            * selected
                                  ->normal_input_split_closure_error;
                    joint
                        .exterior_cauchy_geometry_reconstruction_error +=
                            weight
                            * selected
                                  ->cauchy_geometry_reconstruction_bulk_error;
                    joint
                        .exterior_cauchy_input_split_closure_error +=
                            weight
                            * selected
                                  ->cauchy_input_split_closure_error;
                    joint.exterior_quadratic_taylor_remainder_error +=
                        weight
                        * selected
                              ->quadratic_taylor_remainder_bulk_error;
                    joint.exterior_irregular_split_closure_error +=
                        weight
                        * selected->irregular_split_closure_error;
                    joint.exterior_regular_five_point_error +=
                        weight
                        * selected->regular_five_point_bulk_error;
                    joint.exterior_bulk_split_closure_error +=
                        weight
                        * selected->bulk_split_closure_error;
                    joint.interior_smooth_error +=
                        weight
                        * selected->smooth_interpolation_error;
                    joint.interior_cauchy_error +=
                        weight
                        * selected->cauchy_correction_error;
                    joint.interior_bulk_error +=
                        weight * selected->bulk_grid_error;
                    joint.interior_irregular_spread_error +=
                        weight
                        * selected->irregular_spread_bulk_error;
                    joint.interior_irregular_cauchy_fit_error +=
                        weight
                        * selected->irregular_cauchy_fit_bulk_error;
                    joint.interior_value_jump_input_error +=
                        weight
                        * selected->value_jump_input_bulk_error;
                    joint.interior_normal_jump_input_error +=
                        weight
                        * selected->normal_jump_input_bulk_error;
                    joint.interior_normal_node_frame_error +=
                        weight
                        * selected->normal_node_frame_bulk_error;
                    joint
                        .interior_normal_scalar_interpolation_error +=
                            weight
                            * selected
                                  ->normal_scalar_interpolation_bulk_error;
                    joint
                        .interior_normal_input_split_closure_error +=
                            weight
                            * selected
                                  ->normal_input_split_closure_error;
                    joint
                        .interior_cauchy_geometry_reconstruction_error +=
                            weight
                            * selected
                                  ->cauchy_geometry_reconstruction_bulk_error;
                    joint
                        .interior_cauchy_input_split_closure_error +=
                            weight
                            * selected
                                  ->cauchy_input_split_closure_error;
                    joint.interior_quadratic_taylor_remainder_error +=
                        weight
                        * selected
                              ->quadratic_taylor_remainder_bulk_error;
                    joint.interior_irregular_split_closure_error +=
                        weight
                        * selected->irregular_split_closure_error;
                    joint.interior_regular_five_point_error +=
                        weight
                        * selected->regular_five_point_bulk_error;
                    joint.interior_bulk_split_closure_error +=
                        weight
                        * selected->bulk_split_closure_error;

                    const double tau =
                        normal_layers[
                            static_cast<std::size_t>(layer)];
                    const double true_jump =
                        transformed_exact_solution(
                            rigid_transform,
                            selected->sample.query)
                        - exact_trace_mean;
                    if (side == 0) {
                        const double linear_inside_jump =
                            exact_trace[selected_active_index]
                            - tau * h
                                * rhs_data.compatible_neumann_data[
                                    selected_active_index];
                        joint.exterior_jump_error +=
                            weight
                            * (true_jump - linear_inside_jump);
                    } else {
                        const double linear_outside_jump =
                            exact_trace[selected_active_index]
                            + tau * h
                                * rhs_data.compatible_neumann_data[
                                    selected_active_index];
                        joint.interior_jump_error +=
                            weight
                            * (linear_outside_jump - true_jump);
                    }
                }
            }
            joint.exterior_total_error =
                joint.exterior_smooth_error
                + joint.exterior_cauchy_error
                + joint.exterior_bulk_error
                + joint.exterior_jump_error;
            joint.interior_total_error =
                joint.interior_smooth_error
                + joint.interior_cauchy_error
                + joint.interior_bulk_error
                + joint.interior_jump_error;
            joint.exterior_solver_value =
                exact_cauchy_field.trace_exterior_virtual[
                    selected_active_index];
            joint.interior_solver_error =
                exact_cauchy_field.trace_interior[
                    selected_active_index]
                - exact_trace[selected_active_index];
            metrics.selected_joint_fit = joint;
        }

        const bool uses_center_cauchy_jump =
            restrict_method
                == LaplaceNeumannExteriorRestrictMethod2D::
                       JointBiquadraticQuadraticCenterCauchyJump
            || restrict_method
                == LaplaceNeumannExteriorRestrictMethod2D::
                       JointSixPointQuadraticCenterCauchyJump;
        const bool uses_six_point_cross_stencil =
            restrict_method
                == LaplaceNeumannExteriorRestrictMethod2D::
                       JointSixPointQuadraticInterfaceJump
            || restrict_method
                == LaplaceNeumannExteriorRestrictMethod2D::
                       JointSixPointQuadraticGridEdgeInterfaceJump
            || restrict_method
                == LaplaceNeumannExteriorRestrictMethod2D::
                       JointSixPointQuadraticGridEdgeSharedSidePolynomialInterfaceJump
            || restrict_method
                == LaplaceNeumannExteriorRestrictMethod2D::
                       JointSixPointQuadraticCenterCauchyJump;
        const bool uses_grid_edge_cross_stencil =
            restrict_method
                == LaplaceNeumannExteriorRestrictMethod2D::
                       JointSixPointQuadraticGridEdgeInterfaceJump
            || restrict_method
                == LaplaceNeumannExteriorRestrictMethod2D::
                       JointSixPointQuadraticGridEdgeSharedSidePolynomialInterfaceJump;
        const bool uses_shared_side_spatial_polynomial =
            restrict_method
            == LaplaceNeumannExteriorRestrictMethod2D::
                   JointSixPointQuadraticGridEdgeSharedSidePolynomialInterfaceJump;
        const int expected_samples = 6 * nq;
        const bool uses_interface_jump =
            restrict_method
                == LaplaceNeumannExteriorRestrictMethod2D::
                       JointSixPointQuadraticInterfaceJump
            || uses_grid_edge_cross_stencil;
        if (uses_interface_jump
            && metrics.center_cauchy_jump_samples != 0) {
            throw std::runtime_error(
                "interface-jump restrict unexpectedly used panel-center "
                "Cauchy jump samples");
        }
        if (uses_center_cauchy_jump) {
            if (metrics.center_cauchy_jump_samples
                    != expected_samples
                || metrics.trace_points_without_incident_center != 0) {
                throw std::runtime_error(
                    "center-Cauchy jump restrict did not bind every smooth "
                    "harmonic-jet trace point");
            }
        }
        if (uses_six_point_cross_stencil
            && metrics.six_point_cross_stencils != expected_samples) {
            throw std::runtime_error(
                "six-point quadratic restrict did not build every "
                "smooth harmonic-jet trace stencil");
        }
        if (uses_grid_edge_cross_stencil) {
            const int corrected_diagonal_stencils =
                metrics.grid_edge_diagonal_single_owner_stencils
                + metrics.grid_edge_diagonal_blended_owner_stencils;
            const int diagonal_stencils =
                metrics.grid_edge_diagonal_same_side_stencils
                + corrected_diagonal_stencils;
            const int expected_corrected_nodes =
                metrics.grid_edge_cardinal_corrections
                + corrected_diagonal_stencils;
            const int expected_owner_terms =
                metrics.grid_edge_cardinal_corrections
                + metrics.grid_edge_diagonal_single_owner_stencils
                + 2
                    * metrics
                          .grid_edge_diagonal_blended_owner_stencils;
            const int recorded_owner_terms =
                metrics.exact_crossing_owners
                + metrics.gap_fallback_owners
                + metrics.endpoint_fallback_owners;
            if (metrics.grid_edge_cross_stencils
                    != expected_samples
                || diagonal_stencils != expected_samples
                || metrics.restrict_corrected_nodes
                       != expected_corrected_nodes
                || metrics.restrict_wrong_side_nodes
                       != metrics.restrict_corrected_nodes
                || metrics.grid_edge_blended_correction_nodes
                       != metrics
                              .grid_edge_diagonal_blended_owner_stencils
                || metrics.grid_edge_owner_terms
                       != expected_owner_terms
                || recorded_owner_terms != expected_owner_terms
                || metrics.endpoint_fallback_owners != 0) {
                throw std::runtime_error(
                    "grid-edge crossing restrict diagnostics violate "
                    "owner/node conservation");
            }
        }
        if (uses_shared_side_spatial_polynomial) {
            if (metrics.shared_side_spatial_polynomials != 2 * nq
                || metrics.shared_side_spatial_polynomial_samples
                       != expected_samples) {
                throw std::runtime_error(
                    "shared-side quadratic restrict did not reuse exactly "
                    "one spatial polynomial per trace point and side");
            }
        } else if (metrics.shared_side_spatial_polynomials != 0
                   || metrics
                          .shared_side_spatial_polynomial_samples
                          != 0) {
            throw std::runtime_error(
                "non-shared restrict unexpectedly recorded shared-side "
                "spatial polynomials");
        }
    }
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
    std::cout << "  spread=" << result.spread_mode
              << " restrict=" << result.restrict_method
              << " panel_monitor="
              << result.panel_curvature_monitor
              << " rigid_case="
              << (result.rigid_case.empty()
                      ? std::string("unspecified")
                      : result.rigid_case)
              << " rigid(angle/center/translation)="
              << result.rigid_angle_degrees << '/'
              << result.rigid_center_x << '/'
              << result.rigid_center_y << '/'
              << result.rigid_translation_x << '/'
              << result.rigid_translation_y
              << " dofs=" << result.dofs
              << " panels=" << result.panels
              << " panel_L/h(min/mean/max)="
              << result.panel_min_polyline_length_over_h << '/'
              << result.panel_mean_polyline_length_over_h << '/'
              << result.panel_max_polyline_length_over_h
              << " panel_turn(mean/max)="
              << result.panel_mean_absolute_turn << '/'
              << result.panel_max_absolute_turn
              << " max_kappa_L="
              << result.panel_max_curvature_length_product
              << " gmres=" << result.iterations
              << " aug_ok=" << (result.augmented_converged ? "yes" : "no")
              << " aug_res=" << result.augmented_boundary_res_linf
              << " raw_ext=" << result.raw_exterior_trace_linf
              << " Linf=" << result.global_linf
              << " L2=" << result.global_l2
              << " native_Linf=" << result.native_global_linf
              << " trace=" << result.trace_linf
              << " normal=" << result.normal_linf
              << " spread_cond="
              << result.cubic_harmonic_spread_max_condition
              << " lambda=" << result.lambda
              << " time=" << result.seconds << "s";
    if (result.exact_crossing_owners != 0
        || result.gap_fallback_owners != 0
        || result.six_point_cross_stencils != 0) {
        std::cout
            << " exact_owner=" << result.exact_crossing_owners
            << " gap_owner=" << result.gap_fallback_owners
            << " center_jump/no_center="
            << result.center_cauchy_jump_samples << '/'
            << result.trace_points_without_incident_center
            << " six_stencil=" << result.six_point_cross_stencils
            << " diag_reject(0/multi)="
            << result.diagonal_zero_crossing_rejections << '/'
            << result.diagonal_multiple_crossing_rejections
            << " expanded=" << result.expanded_same_side_center_stencils
            << " max_center_d/h="
            << result.max_same_side_center_distance_over_h
            << " max_weight_L1=" << result.max_six_point_weight_l1;
        if (result.max_six_point_weight_q >= 0) {
            std::cout
                << "@q/side/layer="
                << result.max_six_point_weight_q << '/'
                << result.max_six_point_weight_side << '/'
                << result.max_six_point_weight_layer;
        }
        if (result.grid_edge_cross_stencils != 0) {
            std::cout
                << " grid_edge=" << result.grid_edge_cross_stencils
                << " cardinal="
                << result.grid_edge_cardinal_corrections
                << " diag(same/1/2)="
                << result.grid_edge_diagonal_same_side_stencils
                << '/'
                << result.grid_edge_diagonal_single_owner_stencils
                << '/'
                << result.grid_edge_diagonal_blended_owner_stencils
                << " edge_reject(card/zero/amb)="
                << result.grid_edge_cardinal_ambiguous_rejections
                << '/'
                << result.grid_edge_diagonal_zero_owner_rejections
                << '/'
                << result.grid_edge_diagonal_ambiguous_edge_rejections
                << " owner_terms=" << result.grid_edge_owner_terms
                << " shared_poly/samples="
                << result.shared_side_spatial_polynomials << '/'
                << result.shared_side_spatial_polynomial_samples;
        }
    }
    std::cout << '\n';
    if (result.stencil_shapes.groups != 0) {
        const GridEdgeStencilShapeSummary& shapes =
            result.stencil_shapes;
        const auto percentage =
            [&](int count) {
                return 100.0 * static_cast<double>(count)
                    / static_cast<double>(shapes.groups);
            };
        std::cout
            << "    stencil_shapes groups=" << shapes.groups
            << " diag%(same/1/2)="
            << percentage(shapes.diagonal_types[0]) << '/'
            << percentage(shapes.diagonal_types[1]) << '/'
            << percentage(shapes.diagonal_types[2])
            << " orientation%(SW/SE/NW/NE)="
            << percentage(shapes.diagonal_orientations[0]) << '/'
            << percentage(shapes.diagonal_orientations[1]) << '/'
            << percentage(shapes.diagonal_orientations[2]) << '/'
            << percentage(shapes.diagonal_orientations[3])
            << " toward%="
            << percentage(
                   shapes.diagonal_toward_selection_query)
            << '\n'
            << "      cardinal_count%(0/1/2/3/4)=";
        for (int count = 0; count < 5; ++count) {
            if (count != 0)
                std::cout << '/';
            std::cout << percentage(
                shapes.cardinal_correction_counts[
                    static_cast<std::size_t>(count)]);
        }
        std::cout << " total_correction_count%(0/1/2/3/4/5)=";
        for (int count = 0; count < 6; ++count) {
            if (count != 0)
                std::cout << '/';
            std::cout << percentage(
                shapes.total_correction_counts[
                    static_cast<std::size_t>(count)]);
        }
        std::cout << '\n'
                  << "      owner_panel_count%(0/1/2/3/4/5)=";
        for (int count = 0; count < 6; ++count) {
            if (count != 0)
                std::cout << '/';
            std::cout << percentage(
                shapes.distinct_owner_panel_counts[
                    static_cast<std::size_t>(count)]);
        }
        std::cout
            << " mixed%=" << percentage(
                   shapes.mixed_owner_panel_groups)
            << " endpoint_mixed="
            << shapes.endpoint_mixed_owner_panel_groups << '/'
            << shapes.endpoint_groups
            << " midpoint_mixed="
            << shapes.midpoint_mixed_owner_panel_groups << '/'
            << shapes.midpoint_groups
            << '\n'
            << "      weight_L1 layer(mean/p95/p99/max)";
        for (int layer = 0; layer < 3; ++layer) {
            const std::size_t l = static_cast<std::size_t>(layer);
            std::cout
                << " [" << layer << ':'
                << shapes.weight_l1_mean[l] << '/'
                << shapes.weight_l1_p95[l] << '/'
                << shapes.weight_l1_p99[l] << '/'
                << shapes.weight_l1_max[l] << ']';
        }
        std::cout << '\n'
                  << "      cardinal_masks(L=1,R=2,D=4,U=8)";
        for (int mask = 0; mask < 16; ++mask) {
            const int count =
                shapes.cardinal_masks[
                    static_cast<std::size_t>(mask)];
            if (count == 0)
                continue;
            std::cout
                << " [" << mask << ':'
                << percentage(count) << "%]";
        }
        std::cout << '\n';
    }
    std::cout << "    bulk_max node=" << result.global_linf_node
              << " ij=" << result.global_linf_i << '/'
              << result.global_linf_j
              << " xy=" << result.global_linf_x << '/'
              << result.global_linf_y
              << " signed=" << result.global_linf_signed_error
              << " nearest_q=" << result.global_linf_nearest_trace_q
              << " distance/h="
              << result.global_linf_distance_to_trace_over_h
              << '\n'
              << "    trace_max a=" << result.trace_linf_active_index
              << " q=" << result.trace_linf_q
              << " xy=" << result.trace_linf_x << '/'
              << result.trace_linf_y
              << " signed=" << result.trace_linf_signed_error
              << " incident=" << result.trace_linf_incident_count
              << " panels/local=" << result.trace_linf_panel0 << '/'
              << result.trace_linf_local0 << ','
              << result.trace_linf_panel1 << '/'
              << result.trace_linf_local1
              << '\n'
              << "    normal_max a=" << result.normal_linf_active_index
              << " q=" << result.normal_linf_q
              << " xy=" << result.normal_linf_x << '/'
              << result.normal_linf_y
              << " signed=" << result.normal_linf_signed_error
              << '\n'
              << "    exact_trace_consistency_max a="
              << result.exact_trace_consistency_active_index
              << " q=" << result.exact_trace_consistency_q
              << " xy=" << result.exact_trace_consistency_x << '/'
              << result.exact_trace_consistency_y
              << " signed="
              << result.exact_trace_consistency_signed_residual
              << " incident="
              << result.exact_trace_consistency_incident_count
              << " lambda=" << result.exact_trace_consistency_lambda
              << '\n'
              << "    exact_normal_consistency_max a="
              << result.exact_normal_consistency_active_index
              << " q=" << result.exact_normal_consistency_q
              << " xy=" << result.exact_normal_consistency_x << '/'
              << result.exact_normal_consistency_y
              << " signed="
              << result.exact_normal_consistency_signed_residual
              << " incident="
              << result.exact_normal_consistency_incident_count
              << '\n';
    for (const auto& selected : result.selected_grid_edge_samples) {
        const auto& sample = selected.sample;
        double weight_l1 = 0.0;
        for (double weight : sample.interpolation_weights)
            weight_l1 += std::abs(weight);
        std::cout
            << "    grid_edge_sample q=" << sample.q
            << " side=" << sample.side
            << " layer=" << sample.layer
            << " query=" << sample.query[0] << '/' << sample.query[1]
            << " selection_query=" << sample.selection_query[0] << '/'
            << sample.selection_query[1]
            << " center_node=" << sample.center_node
            << " diag_node=" << sample.diagonal_node
            << " center_d/h(nearest/selected)="
            << sample.nearest_center_distance_over_h << '/'
            << sample.selected_center_distance_over_h
            << " eval_center_d/h="
            << sample.evaluation_center_distance_over_h
            << " expanded=" << (sample.expanded_center ? 1 : 0)
            << " diag_corrected="
            << (sample.diagonal_corrected ? 1 : 0)
            << " diag_owners=" << sample.diagonal_owner_count
            << " weight_L1=" << weight_l1
            << '\n'
            << "      nodes";
        for (int row = 0; row < 6; ++row) {
            const std::size_t r = static_cast<std::size_t>(row);
            std::cout
                << " [" << row << ':' << sample.grid_nodes[r]
                << '@' << sample.grid_points[r][0] << '/'
                << sample.grid_points[r][1]
                << " w=" << sample.interpolation_weights[r] << ']';
        }
        std::cout << '\n';
        for (int cardinal = 0; cardinal < 4; ++cardinal) {
            const std::size_t c = static_cast<std::size_t>(cardinal);
            if (!sample.cardinal_corrected[c])
                continue;
            const P2CrossingOwner2D& owner =
                sample.cardinal_owners[c];
            std::cout
                << "      cardinal_owner edge=" << cardinal
                << " center=" << owner.center_index
                << " panel=" << owner.panel_index
                << " s=" << owner.local_s
                << " edge_t=" << owner.edge_parameter
                << " cross=" << owner.crossing_point[0] << '/'
                << owner.crossing_point[1]
                << '\n';
        }
        for (int owner_index = 0;
             owner_index < sample.diagonal_owner_count;
             ++owner_index) {
            const P2CrossingOwner2D& owner =
                sample.diagonal_owners[
                    static_cast<std::size_t>(owner_index)];
            std::cout
                << "      diagonal_owner k=" << owner_index
                << " center=" << owner.center_index
                << " panel=" << owner.panel_index
                << " s=" << owner.local_s
                << " edge_t=" << owner.edge_parameter
                << " cross=" << owner.crossing_point[0] << '/'
                << owner.crossing_point[1]
                << '\n';
        }
        if (!sample.diagonal_corrected) {
            std::cout << "      diagonal_correction none (same side)";
        } else {
            std::cout
                << "      diagonal_correction solved="
                << selected.solved_diagonal_corrections[0] << '/'
                << selected.solved_diagonal_corrections[1]
                << " exact="
                << selected.exact_diagonal_corrections[0] << '/'
                << selected.exact_diagonal_corrections[1];
        }
        if (sample.diagonal_corrected
            && sample.diagonal_owner_count == 2) {
            const double diagonal_weight =
                sample.interpolation_weights[5];
            std::cout
                << " weighted_half_mismatch(solved/exact)="
                << 0.5 * std::abs(diagonal_weight)
                    * std::abs(
                        selected.solved_diagonal_corrections[0]
                        - selected.solved_diagonal_corrections[1])
                << '/'
                << 0.5 * std::abs(diagonal_weight)
                    * std::abs(
                        selected.exact_diagonal_corrections[0]
                        - selected.exact_diagonal_corrections[1]);
        }
        std::cout
            << '\n'
            << "      exact_sample target/analytic/operator="
            << selected.analytic_target << '/'
            << selected.analytic_corrected_sample << '/'
            << selected.operator_corrected_sample
            << " error(smooth/cauchy/bulk/total)="
            << selected.smooth_interpolation_error << '/'
            << selected.cauchy_correction_error << '/'
            << selected.bulk_grid_error << '/'
            << selected.operator_sample_error
            << " bulk_split(irregular_spread/regular_5pt/closure)="
            << selected.irregular_spread_bulk_error << '/'
            << selected.regular_five_point_bulk_error << '/'
            << selected.bulk_split_closure_error
            << " irregular_split(cauchy_fit/taylor_remainder/closure)="
            << selected.irregular_cauchy_fit_bulk_error << '/'
            << selected.quadratic_taylor_remainder_bulk_error << '/'
            << selected.irregular_split_closure_error
            << " cauchy_fit_split(value_input/normal_input/geometry_reconstruction/closure)="
            << selected.value_jump_input_bulk_error << '/'
            << selected.normal_jump_input_bulk_error << '/'
            << selected.cauchy_geometry_reconstruction_bulk_error << '/'
            << selected.cauchy_input_split_closure_error
            << " normal_input_split(node_frame/scalar_interp/closure)="
            << selected.normal_node_frame_bulk_error << '/'
            << selected.normal_scalar_interpolation_bulk_error << '/'
            << selected.normal_input_split_closure_error
            << '\n';
    }
    if (result.selected_joint_fit.q >= 0) {
        const SelectedBulkDefectSplitDiagnostic& split =
            result.selected_bulk_defect_split;
        std::cout
            << "    selected_bulk_defect_split nodes(irregular/regular)="
            << split.irregular_nodes << '/' << split.regular_nodes
            << " linf(spread_rhs/irregular_defect/regular_defect)="
            << split.spread_rhs_linf << '/'
            << split.irregular_defect_linf << '/'
            << split.regular_defect_linf
            << " irregular_source_linf(cauchy_fit/taylor/closure)="
            << split.cauchy_fit_defect_linf << '/'
            << split.taylor_remainder_defect_linf << '/'
            << split.irregular_source_closure_linf
            << " cauchy_source_linf(value_input/normal_input/geometry_reconstruction/closure)="
            << split.value_jump_input_defect_linf << '/'
            << split.normal_jump_input_defect_linf << '/'
            << split.cauchy_geometry_reconstruction_defect_linf << '/'
            << split.cauchy_input_source_closure_linf
            << " normal_source_linf(node_frame/scalar_interp/closure)="
            << split.normal_node_frame_defect_linf << '/'
            << split.normal_scalar_interpolation_defect_linf << '/'
            << split.normal_input_source_closure_linf
            << " solution_linf(irregular_spread/regular_5pt/closure)="
            << split.irregular_solution_linf << '/'
            << split.regular_solution_linf << '/'
            << split.closure_linf
            << " irregular_solution_linf(cauchy_fit/taylor/closure)="
            << split.cauchy_fit_solution_linf << '/'
            << split.taylor_remainder_solution_linf << '/'
            << split.irregular_solution_closure_linf
            << " cauchy_solution_linf(value_input/normal_input/geometry_reconstruction/closure)="
            << split.value_jump_input_solution_linf << '/'
            << split.normal_jump_input_solution_linf << '/'
            << split.cauchy_geometry_reconstruction_solution_linf
            << '/'
            << split.cauchy_input_solution_closure_linf
            << " normal_solution_linf(node_frame/scalar_interp/closure)="
            << split.normal_node_frame_solution_linf << '/'
            << split.normal_scalar_interpolation_solution_linf << '/'
            << split.normal_input_solution_closure_linf
            << " raw_collocation_interp_linf(value/normal)="
            << split.value_collocation_interpolation_linf << '/'
            << split.normal_collocation_interpolation_linf
            << " counterfactual_solution_linf(exact_value/exact_normal/exact_both)="
            << split.exact_value_counterfactual_solution_linf << '/'
            << split.exact_normal_counterfactual_solution_linf << '/'
            << split.exact_value_normal_counterfactual_solution_linf
            << '\n';
        std::cout
            << "    crossing_owner_blend directed_edges(total/exact/multiple)="
            << split.crossing_owner_edges << '/'
            << split.crossing_owner_exact_edges << '/'
            << split.crossing_owner_multiple_intersection_edges
            << " switch_distance(min/p10/median;lt.01/lt.05)="
            << split.crossing_owner_switch_distance_min << '/'
            << split.crossing_owner_switch_distance_p10 << '/'
            << split.crossing_owner_switch_distance_median << ';'
            << split.crossing_owner_switch_distance_lt_001 << '/'
            << split.crossing_owner_switch_distance_lt_005
            << " adjacent_gap(p95/max)="
            << split.crossing_owner_adjacent_gap_p95 << '/'
            << split.crossing_owner_adjacent_gap_linf
            << " hotspot_panel(edges/min_switch/max_gap)="
            << split.hotspot_panel_crossing_owner_edges << '/'
            << split.hotspot_panel_switch_distance_min << '/'
            << split.hotspot_panel_adjacent_gap_linf
            << " rhs_linf(hard_reconstruction/blend_delta)="
            << split.crossing_owner_hard_reconstruction_linf << '/'
            << split.crossing_owner_blend_rhs_delta_linf
            << " defect_linf(hard/blend)="
            << split.irregular_defect_linf << '/'
            << split.crossing_owner_blended_defect_linf
            << " solution_linf(irregular_hard/irregular_blend/total_hard/total_blend/blend_delta)="
            << split.irregular_solution_linf << '/'
            << split.crossing_owner_blended_solution_linf << '/'
            << split.crossing_owner_hard_total_solution_linf << '/'
            << split.crossing_owner_blended_total_solution_linf << '/'
            << split.crossing_owner_blend_solution_delta_linf
            << '\n';
        std::cout
            << "    boundary_value_reconstruction "
            << "collocation_linf(P2/C1/P3)="
            << split.value_collocation_interpolation_linf << '/'
            << split.c1_hermite_collocation_interpolation_linf << '/'
            << split.p3_curve_collocation_interpolation_linf
            << " rhs_delta_linf(C1/P3)="
            << split.c1_hermite_spread_rhs_delta_linf << '/'
            << split.p3_curve_spread_rhs_delta_linf
            << " irregular_defect_linf(P2/C1/P3)="
            << split.irregular_defect_linf << '/'
            << split.c1_hermite_irregular_defect_linf << '/'
            << split.p3_curve_irregular_defect_linf
            << " solution_linf_irregular(P2/C1/P3)="
            << split.irregular_solution_linf << '/'
            << split.c1_hermite_irregular_solution_linf << '/'
            << split.p3_curve_irregular_solution_linf
            << " solution_linf_total(P2/C1/P3)="
            << split.crossing_owner_hard_total_solution_linf << '/'
            << split.c1_hermite_total_solution_linf << '/'
            << split.p3_curve_total_solution_linf
            << '\n';
        std::cout
            << "    arc_map_resolution samples/mean_knot_spacing="
            << split.arc_length_map_samples << '/'
            << split.arc_length_map_mean_knot_spacing
            << " cauchy_arc_step(min/max;over_mean_knot_min/max)="
            << split.cauchy_collocation_arc_step_min << '/'
            << split.cauchy_collocation_arc_step_max << ';'
            << split.cauchy_step_over_mean_knot_min << '/'
            << split.cauchy_step_over_mean_knot_max
            << " retained_geometry_fd_linf(tangent_relative/normal_angle)="
            << split.panel_point_tangent_fd_relative_linf << '/'
            << split.panel_point_normal_fd_angle_linf
            << '\n';
        const SelectedJointFitDiagnostic& joint =
            result.selected_joint_fit;
        std::cout
            << "    selected_joint_fit q=" << joint.q
            << " exterior(smooth/cauchy/bulk/jump/total/solver)="
            << joint.exterior_smooth_error << '/'
            << joint.exterior_cauchy_error << '/'
            << joint.exterior_bulk_error << '/'
            << joint.exterior_jump_error << '/'
            << joint.exterior_total_error << '/'
            << joint.exterior_solver_value
            << " interior(smooth/cauchy/bulk/jump/total/solver_error)="
            << joint.interior_smooth_error << '/'
            << joint.interior_cauchy_error << '/'
            << joint.interior_bulk_error << '/'
            << joint.interior_jump_error << '/'
            << joint.interior_total_error << '/'
            << joint.interior_solver_error
            << " bulk_split_ext(irregular_spread/regular_5pt/closure)="
            << joint.exterior_irregular_spread_error << '/'
            << joint.exterior_regular_five_point_error << '/'
            << joint.exterior_bulk_split_closure_error
            << " irregular_split_ext(cauchy_fit/taylor/closure)="
            << joint.exterior_irregular_cauchy_fit_error << '/'
            << joint.exterior_quadratic_taylor_remainder_error << '/'
            << joint.exterior_irregular_split_closure_error
            << " cauchy_fit_split_ext(value_input/normal_input/geometry_reconstruction/closure)="
            << joint.exterior_value_jump_input_error << '/'
            << joint.exterior_normal_jump_input_error << '/'
            << joint.exterior_cauchy_geometry_reconstruction_error
            << '/'
            << joint.exterior_cauchy_input_split_closure_error
            << " normal_input_split_ext(node_frame/scalar_interp/closure)="
            << joint.exterior_normal_node_frame_error << '/'
            << joint.exterior_normal_scalar_interpolation_error << '/'
            << joint.exterior_normal_input_split_closure_error
            << " bulk_split_int(irregular_spread/regular_5pt/closure)="
            << joint.interior_irregular_spread_error << '/'
            << joint.interior_regular_five_point_error << '/'
            << joint.interior_bulk_split_closure_error
            << " irregular_split_int(cauchy_fit/taylor/closure)="
            << joint.interior_irregular_cauchy_fit_error << '/'
            << joint.interior_quadratic_taylor_remainder_error << '/'
            << joint.interior_irregular_split_closure_error
            << " cauchy_fit_split_int(value_input/normal_input/geometry_reconstruction/closure)="
            << joint.interior_value_jump_input_error << '/'
            << joint.interior_normal_jump_input_error << '/'
            << joint.interior_cauchy_geometry_reconstruction_error
            << '/'
            << joint.interior_cauchy_input_split_closure_error
            << " normal_input_split_int(node_frame/scalar_interp/closure)="
            << joint.interior_normal_node_frame_error << '/'
            << joint.interior_normal_scalar_interpolation_error << '/'
            << joint.interior_normal_input_split_closure_error
            << '\n';
    }
}

void print_summary(const std::vector<StudyResult>& results)
{
    for (const std::string geometry : {"ellipse", "flower"}) {
        std::cout << "\nC++ P2 exterior-trace summary: " << geometry
                  << " spread="
                  << (results.empty() ? "unknown"
                                      : results.front().spread_mode)
                  << " restrict="
                  << (results.empty() ? "unknown"
                                      : results.front().restrict_method)
                  << '\n';
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
    out << "geometry,spread_mode,restrict_method,"
           "panel_curvature_monitor,rigid_case,"
           "rigid_angle_degrees,rigid_center_x,rigid_center_y,"
           "rigid_translation_x,rigid_translation_y,"
           "N,h,panel_length_over_h,"
           "panel_curvature_turn_limit,"
           "panel_min_polyline_length_over_h,"
           "panel_mean_polyline_length_over_h,"
           "panel_max_polyline_length_over_h,"
           "panel_mean_absolute_turn,panel_max_absolute_turn,"
           "panel_max_curvature_length_product,"
           "dofs,panels,gmres_iterations,augmented_converged,"
           "seconds,augmented_boundary_res_linf,raw_exterior_trace_linf,"
           "raw_exterior_trace_l2,exterior_from_jump_linf,"
           "exterior_route_mismatch_linf,"
           "flux_mean_removed,lambda,trace_mean,physical_relative_residual,"
           "exact_trace_consistency_linf,"
           "exact_trace_consistency_signed_residual,"
           "exact_trace_consistency_lambda,"
           "exact_trace_consistency_active_index,"
           "exact_trace_consistency_q,"
           "exact_trace_consistency_x,exact_trace_consistency_y,"
           "exact_trace_consistency_incident_count,"
           "exact_normal_consistency_linf,"
           "exact_normal_consistency_signed_residual,"
           "exact_normal_consistency_active_index,"
           "exact_normal_consistency_q,"
           "exact_normal_consistency_x,exact_normal_consistency_y,"
           "exact_normal_consistency_incident_count,"
           "global_linf,global_l2,constant_shift,n_global,"
           "global_linf_signed_error,global_linf_node,"
           "global_linf_i,global_linf_j,global_linf_x,global_linf_y,"
           "global_linf_nearest_trace_q,global_linf_nearest_trace_x,"
           "global_linf_nearest_trace_y,"
           "global_linf_distance_to_trace_over_h,"
           "native_global_linf,native_global_l2,native_constant_shift,"
           "n_native_global,trace_linf,trace_l2,"
           "trace_linf_signed_error,trace_linf_active_index,trace_linf_q,"
           "trace_linf_x,trace_linf_y,trace_linf_incident_count,"
           "trace_linf_panel0,trace_linf_local0,"
           "trace_linf_panel1,trace_linf_local1,"
           "normal_linf,normal_l2,normal_linf_signed_error,"
           "normal_linf_active_index,normal_linf_q,"
           "normal_linf_x,normal_linf_y,"
           "restrict_wrong_side_nodes,restrict_corrected_nodes,"
           "exact_crossing_owners,gap_fallback_owners,"
           "endpoint_fallback_owners,"
           "center_cauchy_jump_samples,"
           "trace_points_without_incident_center,"
           "six_point_cross_stencils,"
           "diagonal_zero_crossing_rejections,"
           "diagonal_multiple_crossing_rejections,"
           "grid_edge_cross_stencils,"
           "grid_edge_cardinal_corrections,"
           "grid_edge_cardinal_ambiguous_rejections,"
           "grid_edge_diagonal_zero_owner_rejections,"
           "grid_edge_diagonal_ambiguous_edge_rejections,"
           "grid_edge_diagonal_same_side_stencils,"
           "grid_edge_diagonal_single_owner_stencils,"
           "grid_edge_diagonal_blended_owner_stencils,"
           "grid_edge_blended_correction_nodes,"
           "grid_edge_owner_terms,"
           "shared_side_spatial_polynomials,"
           "shared_side_spatial_polynomial_samples,"
           "expanded_same_side_center_stencils,"
           "max_same_side_center_distance_over_h,"
           "max_six_point_weight_l1,"
           "max_six_point_weight_q,"
           "max_six_point_weight_side,"
           "max_six_point_weight_layer,"
           "cubic_harmonic_spread_max_condition,"
           "global_linf_order,global_l2_order\n";
    out << std::setprecision(17);
    for (const StudyResult& result : results) {
        out << result.geometry << ',' << result.spread_mode << ','
            << result.restrict_method << ','
            << result.panel_curvature_monitor << ','
            << result.rigid_case << ','
            << result.rigid_angle_degrees << ','
            << result.rigid_center_x << ','
            << result.rigid_center_y << ','
            << result.rigid_translation_x << ','
            << result.rigid_translation_y << ','
            << result.n << ',' << result.h << ','
            << result.panel_length_over_h << ','
            << result.panel_curvature_turn_limit << ','
            << result.panel_min_polyline_length_over_h << ','
            << result.panel_mean_polyline_length_over_h << ','
            << result.panel_max_polyline_length_over_h << ','
            << result.panel_mean_absolute_turn << ','
            << result.panel_max_absolute_turn << ','
            << result.panel_max_curvature_length_product << ','
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
            << result.exact_trace_consistency_linf << ','
            << result.exact_trace_consistency_signed_residual << ','
            << result.exact_trace_consistency_lambda << ','
            << result.exact_trace_consistency_active_index << ','
            << result.exact_trace_consistency_q << ','
            << result.exact_trace_consistency_x << ','
            << result.exact_trace_consistency_y << ','
            << result.exact_trace_consistency_incident_count << ','
            << result.exact_normal_consistency_linf << ','
            << result.exact_normal_consistency_signed_residual << ','
            << result.exact_normal_consistency_active_index << ','
            << result.exact_normal_consistency_q << ','
            << result.exact_normal_consistency_x << ','
            << result.exact_normal_consistency_y << ','
            << result.exact_normal_consistency_incident_count << ','
            << result.global_linf << ',' << result.global_l2 << ','
            << result.constant_shift << ',' << result.n_global << ','
            << result.global_linf_signed_error << ','
            << result.global_linf_node << ','
            << result.global_linf_i << ',' << result.global_linf_j << ','
            << result.global_linf_x << ',' << result.global_linf_y << ','
            << result.global_linf_nearest_trace_q << ','
            << result.global_linf_nearest_trace_x << ','
            << result.global_linf_nearest_trace_y << ','
            << result.global_linf_distance_to_trace_over_h << ','
            << result.native_global_linf << ','
            << result.native_global_l2 << ','
            << result.native_constant_shift << ','
            << result.n_native_global << ','
            << result.trace_linf << ',' << result.trace_l2 << ','
            << result.trace_linf_signed_error << ','
            << result.trace_linf_active_index << ','
            << result.trace_linf_q << ','
            << result.trace_linf_x << ',' << result.trace_linf_y << ','
            << result.trace_linf_incident_count << ','
            << result.trace_linf_panel0 << ','
            << result.trace_linf_local0 << ','
            << result.trace_linf_panel1 << ','
            << result.trace_linf_local1 << ','
            << result.normal_linf << ',' << result.normal_l2 << ','
            << result.normal_linf_signed_error << ','
            << result.normal_linf_active_index << ','
            << result.normal_linf_q << ','
            << result.normal_linf_x << ',' << result.normal_linf_y << ','
            << result.restrict_wrong_side_nodes << ','
            << result.restrict_corrected_nodes << ','
            << result.exact_crossing_owners << ','
            << result.gap_fallback_owners << ','
            << result.endpoint_fallback_owners << ','
            << result.center_cauchy_jump_samples << ','
            << result.trace_points_without_incident_center << ','
            << result.six_point_cross_stencils << ','
            << result.diagonal_zero_crossing_rejections << ','
            << result.diagonal_multiple_crossing_rejections << ','
            << result.grid_edge_cross_stencils << ','
            << result.grid_edge_cardinal_corrections << ','
            << result.grid_edge_cardinal_ambiguous_rejections << ','
            << result.grid_edge_diagonal_zero_owner_rejections << ','
            << result.grid_edge_diagonal_ambiguous_edge_rejections << ','
            << result.grid_edge_diagonal_same_side_stencils << ','
            << result.grid_edge_diagonal_single_owner_stencils << ','
            << result.grid_edge_diagonal_blended_owner_stencils << ','
            << result.grid_edge_blended_correction_nodes << ','
            << result.grid_edge_owner_terms << ','
            << result.shared_side_spatial_polynomials << ','
            << result.shared_side_spatial_polynomial_samples << ','
            << result.expanded_same_side_center_stencils << ','
            << result.max_same_side_center_distance_over_h << ','
            << result.max_six_point_weight_l1 << ','
            << result.max_six_point_weight_q << ','
            << result.max_six_point_weight_side << ','
            << result.max_six_point_weight_layer << ','
            << result.cubic_harmonic_spread_max_condition << ','
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
        std::vector<int> levels = {32, 64, 128, 256};
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
        const double panel_curvature_turn_limit = environment_double(
            "KFBIM_HJET_PANEL_CURVATURE_TURN", 0.0);
        if (panel_curvature_turn_limit < 0.0
            || !std::isfinite(panel_curvature_turn_limit)) {
            throw std::invalid_argument(
                "KFBIM_HJET_PANEL_CURVATURE_TURN must be nonnegative");
        }
        const CurvaturePanelMonitor2D panel_curvature_monitor =
            selected_panel_curvature_monitor();
        const std::string rigid_case = rigid_case_tag();
        const RigidTransform2D rigid_transform =
            make_rigid_transform(
                environment_double(
                    "KFBIM_HJET_RIGID_ANGLE_DEG", 0.0),
                Eigen::Vector2d(
                    environment_double(
                        "KFBIM_HJET_RIGID_CENTER_X", 0.0),
                    environment_double(
                        "KFBIM_HJET_RIGID_CENTER_Y", 0.0)),
                Eigen::Vector2d(
                    environment_double(
                        "KFBIM_HJET_RIGID_TRANSLATION_X", 0.0),
                    environment_double(
                        "KFBIM_HJET_RIGID_TRANSLATION_Y", 0.0)));
        const int max_iter = environment_int(
            "KFBIM_HJET_MAX_ITER", 100);
        const int restart = environment_int(
            "KFBIM_HJET_RESTART", 80);
        const double tolerance = environment_double(
            "KFBIM_HJET_TOL", 2.0e-10);
        const LaplaceNeumannExteriorRestrictMethod2D restrict_method =
            selected_restrict_method();
        const LaplaceP2PanelCenterSpreadMode2D spread_mode =
            selected_spread_mode();

        std::cout << "KFBI2D P2 exterior-trace on harmonic-jet tutorial cases\n"
                  << "tol=" << tolerance << " max_iter=" << max_iter
                  << " restart=" << restart
                  << " panel_length/h=" << panel_length_over_h
                  << " panel_curvature_turn="
                  << panel_curvature_turn_limit
                  << " panel_curvature_monitor="
                  << panel_curvature_monitor_name(
                         panel_curvature_monitor)
                  << " spread=" << spread_mode_name(spread_mode)
                  << " restrict=" << restrict_method_name(restrict_method)
                  << " rigid_case="
                  << (rigid_case.empty()
                          ? std::string("unspecified")
                          : rigid_case)
                  << " rigid(angle/center/translation)="
                  << rigid_transform.angle_degrees << '/'
                  << rigid_transform.center.x() << '/'
                  << rigid_transform.center.y() << '/'
                  << rigid_transform.translation.x() << '/'
                  << rigid_transform.translation.y()
                  << '\n';

        std::vector<StudyResult> results;
        for (const std::string& geometry : geometries) {
            for (const int n : levels) {
                std::cout << "[run] geometry=" << geometry
                          << " N=" << n << '\n';
                StudyResult result = run_one(
                    geometry,
                    n,
                    panel_length_over_h,
                    panel_curvature_turn_limit,
                    panel_curvature_monitor,
                    rigid_transform,
                    max_iter,
                    tolerance,
                    restart,
                    restrict_method,
                    spread_mode);
                result.rigid_case = rigid_case;
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
        const std::string restrict_tag =
            restrict_output_tag(restrict_method);
        const std::filesystem::path csv_path =
            spread_mode
                    == LaplaceP2PanelCenterSpreadMode2D::QuadraticCauchy
                ? output_dir
                    / (restrict_method
                               == LaplaceNeumannExteriorRestrictMethod2D::
                                      SixPointQuadratic
                           ? "harmonic_jet_case_cpp_p2.csv"
                           : std::string("harmonic_jet_case_cpp_p2_")
                                 + restrict_tag + ".csv")
                : output_dir
                    / (std::string("harmonic_jet_case_cpp_p2_")
                       + spread_mode_name(spread_mode)
                       + "_spread_" + restrict_tag + "_restrict.csv");
        std::filesystem::path selected_csv_path = csv_path;
        if (panel_curvature_turn_limit > 0.0) {
            selected_csv_path =
                csv_path.parent_path()
                / (csv_path.stem().string()
                   + "_curvature_adaptive"
                   + std::string("_")
                   + panel_curvature_monitor_name(
                         panel_curvature_monitor)
                   + csv_path.extension().string());
        }
        if (!rigid_case.empty()) {
            selected_csv_path =
                selected_csv_path.parent_path()
                / (selected_csv_path.stem().string()
                   + "_rigid_" + rigid_case
                   + selected_csv_path.extension().string());
        } else if (!rigid_transform.is_identity()) {
            selected_csv_path =
                selected_csv_path.parent_path()
                / (selected_csv_path.stem().string()
                   + "_rigid_transform"
                   + selected_csv_path.extension().string());
        }
        write_csv(selected_csv_path, results);
        std::cout << "CSV: " << selected_csv_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
