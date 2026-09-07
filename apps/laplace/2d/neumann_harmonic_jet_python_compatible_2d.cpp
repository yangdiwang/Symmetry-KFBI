#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "src/bulk_solvers/laplace_zfft_bulk_solver_2d.hpp"
#include "src/gmres/gmres.hpp"
#include "src/grid/cartesian_grid_2d.hpp"
#include "src/operators/i_kfbi_operator.hpp"

using namespace kfbim;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kDefaultBoxMin = -2.5;
constexpr double kDefaultBoxMax = 2.5;
constexpr double kCircleBoxMin = -1.0;
constexpr double kCircleBoxMax = 1.0;
constexpr double kCircleCx = 0.07;
constexpr double kCircleCy = -0.04;
constexpr double kCircleRadius = 0.50;
constexpr double kCircleGaugeConstant = 0.37;
constexpr double kLShapeBoxMin = -1.5;
constexpr double kLShapeBoxMax = 1.5;
constexpr std::array<double, 4> kNormalLayers{{0.5, 1.5, 2.5, 3.5}};

enum class GeometryKind {
    Ellipse,
    Flower,
    Circle,
    LShape
};

enum class BvpKind {
    Neumann,
    Dirichlet
};

enum class DirichletFormulation {
    NormalJumpFirstKind,
    ValueJumpSecondKind,
    NormalJumpSecondKind
};

std::string dirichlet_formulation_name(DirichletFormulation formulation)
{
    switch (formulation) {
    case DirichletFormulation::NormalJumpFirstKind:
        return "normal_jump_first_kind";
    case DirichletFormulation::ValueJumpSecondKind:
        return "value_jump_second_kind";
    case DirichletFormulation::NormalJumpSecondKind:
        return "normal_jump_second_kind";
    }
    throw std::invalid_argument("unknown Dirichlet formulation");
}

std::vector<DirichletFormulation> parse_dirichlet_formulations(
    const std::string& name)
{
    if (name == "normal_jump_first_kind")
        return {DirichletFormulation::NormalJumpFirstKind};
    if (name == "value_jump_second_kind")
        return {DirichletFormulation::ValueJumpSecondKind};
    if (name == "normal_jump_second_kind")
        return {DirichletFormulation::NormalJumpSecondKind};
    if (name == "compare") {
        return {DirichletFormulation::NormalJumpFirstKind,
                DirichletFormulation::ValueJumpSecondKind,
                DirichletFormulation::NormalJumpSecondKind};
    }
    throw std::invalid_argument(
        "KFBIM_PYJET_DIRICHLET_FORMULATION must be "
        "normal_jump_first_kind, value_jump_second_kind, "
        "normal_jump_second_kind, or compare");
}

std::string bvp_name(BvpKind kind)
{
    return kind == BvpKind::Dirichlet ? "dirichlet" : "neumann";
}

BvpKind default_bvp_kind()
{
#ifdef KFBIM_PYJET_DEFAULT_BVP_DIRICHLET
    return BvpKind::Dirichlet;
#else
    return BvpKind::Neumann;
#endif
}

enum class SpreadCorrectionMode {
    HarmonicJetAtOppositeNode,
    CrossingDensity,
    QuadraticHarmonicAtOppositeNode,
    CubicHarmonicAtOppositeNode
};

enum class InterfaceDofMode {
    Crossing,
    UniformArcMidpoint
};

enum class RestrictMode {
    BicubicCubicNormal,
    BicubicQuadraticNormal,
    BiquadraticCubicNormal,
    BiquadraticQuadraticNormal,
    BiquadraticQuadraticTwoLayer,
    SixPointQuadraticExterior
};

std::string restrict_mode_name(RestrictMode mode)
{
    if (mode == RestrictMode::BicubicCubicNormal)
        return "bicubic_cubic";
    if (mode == RestrictMode::BicubicQuadraticNormal)
        return "bicubic_quadratic";
    if (mode == RestrictMode::BiquadraticCubicNormal)
        return "biquadratic_cubic";
    if (mode == RestrictMode::BiquadraticQuadraticNormal)
        return "biquadratic_quadratic";
    if (mode == RestrictMode::BiquadraticQuadraticTwoLayer)
        return "biquadratic_quadratic_two_layer";
    return "six_point_quadratic_exterior";
}

std::vector<RestrictMode> parse_restrict_modes(const std::string& name)
{
    if (name == "bicubic_cubic")
        return {RestrictMode::BicubicCubicNormal};
    if (name == "bicubic_quadratic")
        return {RestrictMode::BicubicQuadraticNormal};
    if (name == "biquadratic_cubic")
        return {RestrictMode::BiquadraticCubicNormal};
    if (name == "biquadratic_quadratic")
        return {RestrictMode::BiquadraticQuadraticNormal};
    if (name == "biquadratic_quadratic_two_layer")
        return {RestrictMode::BiquadraticQuadraticTwoLayer};
    if (name == "six_point_quadratic_exterior")
        return {RestrictMode::SixPointQuadraticExterior};
    if (name == "normal_compare") {
        return {RestrictMode::BicubicCubicNormal,
                RestrictMode::BiquadraticQuadraticNormal,
                RestrictMode::BiquadraticQuadraticTwoLayer};
    }
    if (name == "degree_compare") {
        return {RestrictMode::BiquadraticQuadraticNormal,
                RestrictMode::BiquadraticCubicNormal,
                RestrictMode::BicubicQuadraticNormal,
                RestrictMode::BicubicCubicNormal};
    }
    if (name == "compare") {
        return {RestrictMode::BicubicCubicNormal,
                RestrictMode::BiquadraticQuadraticNormal,
                RestrictMode::BiquadraticQuadraticTwoLayer,
                RestrictMode::SixPointQuadraticExterior};
    }
    throw std::invalid_argument(
        "KFBIM_PYJET_RESTRICT_MODE must be bicubic_cubic, "
        "bicubic_quadratic, biquadratic_cubic, biquadratic_quadratic, "
        "biquadratic_quadratic_two_layer, six_point_quadratic_exterior, "
        "normal_compare, degree_compare, or compare");
}

std::string dof_mode_name(InterfaceDofMode mode)
{
    return mode == InterfaceDofMode::Crossing
        ? "crossing" : "uniform_midpoint";
}

std::vector<InterfaceDofMode> parse_dof_modes(const std::string& name)
{
    if (name == "crossing")
        return {InterfaceDofMode::Crossing};
    if (name == "uniform_midpoint")
        return {InterfaceDofMode::UniformArcMidpoint};
    if (name == "compare") {
        return {InterfaceDofMode::Crossing,
                InterfaceDofMode::UniformArcMidpoint};
    }
    throw std::invalid_argument(
        "KFBIM_PYJET_DOF_MODE must be crossing, uniform_midpoint, or compare");
}

std::string spread_mode_name(SpreadCorrectionMode mode)
{
    if (mode == SpreadCorrectionMode::CrossingDensity)
        return "crossing_density";
    if (mode == SpreadCorrectionMode::QuadraticHarmonicAtOppositeNode)
        return "quadratic_harmonic";
    if (mode == SpreadCorrectionMode::CubicHarmonicAtOppositeNode)
        return "cubic_harmonic";
    return "harmonic_jet";
}

SpreadCorrectionMode parse_spread_mode(const std::string& name)
{
    if (name == "harmonic_jet")
        return SpreadCorrectionMode::HarmonicJetAtOppositeNode;
    if (name == "crossing_density")
        return SpreadCorrectionMode::CrossingDensity;
    if (name == "quadratic_harmonic")
        return SpreadCorrectionMode::QuadraticHarmonicAtOppositeNode;
    if (name == "cubic_harmonic")
        return SpreadCorrectionMode::CubicHarmonicAtOppositeNode;
    throw std::invalid_argument(
        "KFBIM_PYJET_SPREAD_MODE must be harmonic_jet, crossing_density, "
        "quadratic_harmonic, or cubic_harmonic");
}

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

const std::array<Eigen::Vector2d, 6>& lshape_vertices()
{
    static const std::array<Eigen::Vector2d, 6> vertices{{
        Eigen::Vector2d(-0.93, -1.04),
        Eigen::Vector2d( 1.07, -1.04),
        Eigen::Vector2d( 1.07, -0.04),
        Eigen::Vector2d( 0.07, -0.04),
        Eigen::Vector2d( 0.07,  0.96),
        Eigen::Vector2d(-0.93,  0.96)
    }};
    return vertices;
}

double lshape_perimeter()
{
    return 8.0;
}

int lshape_boundary_segment(const Eigen::Vector2d& point)
{
    const auto& vertices = lshape_vertices();
    int best_edge = -1;
    double best_distance_sq = std::numeric_limits<double>::infinity();
    for (int edge = 0; edge < static_cast<int>(vertices.size()); ++edge) {
        const Eigen::Vector2d& start =
            vertices[static_cast<std::size_t>(edge)];
        const Eigen::Vector2d& end = vertices[static_cast<std::size_t>(
            (edge + 1) % static_cast<int>(vertices.size()))];
        const Eigen::Vector2d delta = end - start;
        const double fraction = std::clamp(
            (point - start).dot(delta) / delta.squaredNorm(), 0.0, 1.0);
        const double distance_sq =
            (point - (start + fraction * delta)).squaredNorm();
        if (distance_sq < best_distance_sq) {
            best_distance_sq = distance_sq;
            best_edge = edge;
        }
    }
    if (best_edge < 0)
        throw std::runtime_error("L-shape point has no boundary segment");
    return best_edge;
}

GeometryKind parse_geometry(const std::string& name)
{
    if (name == "ellipse")
        return GeometryKind::Ellipse;
    if (name == "flower")
        return GeometryKind::Flower;
    if (name == "circle")
        return GeometryKind::Circle;
    if (name == "lshape")
        return GeometryKind::LShape;
    throw std::invalid_argument(
        "geometry must be ellipse, flower, circle, or lshape");
}

std::string geometry_name(GeometryKind kind)
{
    switch (kind) {
    case GeometryKind::Ellipse:
        return "ellipse";
    case GeometryKind::Flower:
        return "flower";
    case GeometryKind::Circle:
        return "circle";
    case GeometryKind::LShape:
        return "lshape";
    }
    throw std::invalid_argument("unsupported geometry");
}

Eigen::Vector2d geometry_translation(GeometryKind kind)
{
    switch (kind) {
    case GeometryKind::Ellipse:
        return Eigen::Vector2d(0.13, -0.08);
    case GeometryKind::Flower:
        return Eigen::Vector2d(0.09, -0.06);
    case GeometryKind::Circle:
        return Eigen::Vector2d(kCircleCx, kCircleCy);
    case GeometryKind::LShape:
        return Eigen::Vector2d(0.07, -0.04);
    }
    throw std::invalid_argument("unsupported geometry");
}

double geometry_angle(GeometryKind kind)
{
    switch (kind) {
    case GeometryKind::Ellipse:
        return 27.0;
    case GeometryKind::Flower:
        return 14.0;
    case GeometryKind::Circle:
        return 0.0;
    case GeometryKind::LShape:
        return 0.0;
    }
    throw std::invalid_argument("unsupported geometry");
}

double geometry_box_min(GeometryKind kind)
{
    if (kind == GeometryKind::Circle)
        return kCircleBoxMin;
    if (kind == GeometryKind::LShape)
        return kLShapeBoxMin;
    return kDefaultBoxMin;
}

double geometry_box_max(GeometryKind kind)
{
    if (kind == GeometryKind::Circle)
        return kCircleBoxMax;
    if (kind == GeometryKind::LShape)
        return kLShapeBoxMax;
    return kDefaultBoxMax;
}

Eigen::Vector2d geometry_semi_axes(GeometryKind kind)
{
    if (kind == GeometryKind::Circle)
        return Eigen::Vector2d(kCircleRadius, kCircleRadius);
    if (kind == GeometryKind::Ellipse)
        return Eigen::Vector2d(1.10, 0.76);
    throw std::invalid_argument(
        "semi-axes are only defined for ellipse and circle");
}

Eigen::Vector2d to_local(GeometryKind kind, const Eigen::Vector2d& point)
{
    return rotation(geometry_angle(kind)).transpose()
         * (point - geometry_translation(kind));
}

double flower_radius(double theta)
{
    return 0.88 * (1.0 + 0.18 * std::cos(5.0 * theta));
}

double flower_radius_derivative(double theta)
{
    return -0.88 * 0.18 * 5.0 * std::sin(5.0 * theta);
}

Eigen::Vector2d geometry_point(GeometryKind kind, double theta)
{
    if (kind == GeometryKind::LShape) {
        double distance = std::fmod(
            theta * lshape_perimeter() / (2.0 * kPi),
            lshape_perimeter());
        if (distance < 0.0)
            distance += lshape_perimeter();
        const auto& vertices = lshape_vertices();
        for (int edge = 0; edge < static_cast<int>(vertices.size()); ++edge) {
            const Eigen::Vector2d& start =
                vertices[static_cast<std::size_t>(edge)];
            const Eigen::Vector2d& end = vertices[static_cast<std::size_t>(
                (edge + 1) % static_cast<int>(vertices.size()))];
            const double length = (end - start).norm();
            if (distance <= length || edge + 1 == static_cast<int>(vertices.size()))
                return start + (distance / length) * (end - start);
            distance -= length;
        }
        throw std::runtime_error("L-shape boundary parameterization failed");
    }
    Eigen::Vector2d local;
    if (kind == GeometryKind::Flower) {
        const double radius = flower_radius(theta);
        local = radius * Eigen::Vector2d(std::cos(theta), std::sin(theta));
    } else {
        const Eigen::Vector2d axes = geometry_semi_axes(kind);
        local = Eigen::Vector2d(axes[0] * std::cos(theta),
                                axes[1] * std::sin(theta));
    }
    return geometry_translation(kind)
         + rotation(geometry_angle(kind)) * local;
}

double flower_level_set(const Eigen::Vector2d& point)
{
    const Eigen::Vector2d q = to_local(GeometryKind::Flower, point);
    const double theta = std::atan2(q[1], q[0]);
    return q.norm() - flower_radius(theta);
}

bool geometry_inside(GeometryKind kind, const Eigen::Vector2d& point)
{
    if (kind == GeometryKind::LShape) {
        constexpr double ax = -0.93;
        constexpr double bx = 0.07;
        constexpr double cx = 1.07;
        constexpr double ay = -1.04;
        constexpr double by = -0.04;
        constexpr double cy = 0.96;
        return point[0] > ax && point[0] < cx
            && point[1] > ay && point[1] < cy
            && (point[0] < bx || point[1] < by);
    }
    const Eigen::Vector2d q = to_local(kind, point);
    if (kind != GeometryKind::Flower) {
        const Eigen::Vector2d axes = geometry_semi_axes(kind);
        const double x = q[0] / axes[0];
        const double y = q[1] / axes[1];
        return x * x + y * y < 1.0;
    }
    return flower_level_set(point) < 0.0;
}

Eigen::Vector2d geometry_normal(GeometryKind kind,
                                const Eigen::Vector2d& point)
{
    if (kind == GeometryKind::LShape) {
        const auto& vertices = lshape_vertices();
        const int edge = lshape_boundary_segment(point);
        const Eigen::Vector2d tangent =
            (vertices[static_cast<std::size_t>((edge + 1) % 6)]
             - vertices[static_cast<std::size_t>(edge)]).normalized();
        return Eigen::Vector2d(tangent[1], -tangent[0]);
    }
    const Eigen::Vector2d q = to_local(kind, point);
    Eigen::Vector2d local;
    if (kind == GeometryKind::Flower) {
        const double radius = std::max(q.norm(), 1.0e-14);
        const double theta = std::atan2(q[1], q[0]);
        const double rp = flower_radius_derivative(theta);
        const double c = q[0] / radius;
        const double s = q[1] / radius;
        local = Eigen::Vector2d(c + rp * s / radius,
                                s - rp * c / radius);
    } else {
        const Eigen::Vector2d axes = geometry_semi_axes(kind);
        local = Eigen::Vector2d(q[0] / (axes[0] * axes[0]),
                                q[1] / (axes[1] * axes[1]));
    }
    Eigen::Vector2d world = rotation(geometry_angle(kind)) * local;
    return world / world.norm();
}

Eigen::Vector2d segment_intersection(GeometryKind kind,
                                     const Eigen::Vector2d& p,
                                     const Eigen::Vector2d& q,
                                     double& phase)
{
    if (kind == GeometryKind::LShape) {
        const bool p_inside = geometry_inside(kind, p);
        if (p_inside == geometry_inside(kind, q))
            throw std::runtime_error(
                "L-shape segment does not bracket the interface");
        double left = 0.0;
        double right = 1.0;
        for (int iteration = 0; iteration < 64; ++iteration) {
            const double middle = 0.5 * (left + right);
            const bool middle_inside =
                geometry_inside(kind, p + middle * (q - p));
            if (middle_inside == p_inside)
                left = middle;
            else
                right = middle;
        }
        phase = 0.5 * (left + right);
        return p + phase * (q - p);
    }
    if (kind != GeometryKind::Flower) {
        const Eigen::Vector2d a = to_local(kind, p);
        const Eigen::Vector2d b = to_local(kind, q);
        const Eigen::Vector2d d = b - a;
        const Eigen::Vector2d axes = geometry_semi_axes(kind);
        const double ax2 = axes[0] * axes[0];
        const double ay2 = axes[1] * axes[1];
        const double aa = d[0] * d[0] / ax2 + d[1] * d[1] / ay2;
        const double bb = 2.0 * (a[0] * d[0] / ax2
                               + a[1] * d[1] / ay2);
        const double cc = a[0] * a[0] / ax2
                        + a[1] * a[1] / ay2 - 1.0;
        const double discriminant = bb * bb - 4.0 * aa * cc;
        if (!(discriminant >= 0.0) || aa == 0.0)
            throw std::runtime_error(
                "quadratic geometry segment does not cross interface");
        const double root = std::sqrt(discriminant);
        const std::array<double, 2> roots{{(-bb - root) / (2.0 * aa),
                                           (-bb + root) / (2.0 * aa)}};
        int count = 0;
        for (double candidate : roots) {
            if (candidate > 1.0e-12 && candidate < 1.0 - 1.0e-12) {
                phase = candidate;
                ++count;
            }
        }
        if (count != 1)
            throw std::runtime_error(
                "quadratic geometry crossing root is ambiguous");
        return p + phase * (q - p);
    }

    double left = 0.0;
    double right = 1.0;
    double f_left = flower_level_set(p);
    double f_right = flower_level_set(q);
    if (f_left * f_right >= 0.0)
        throw std::runtime_error("flower segment does not bracket interface");
    for (int iteration = 0; iteration < 64; ++iteration) {
        const double middle = 0.5 * (left + right);
        const double f_middle = flower_level_set(p + middle * (q - p));
        if (f_left * f_middle <= 0.0) {
            right = middle;
            f_right = f_middle;
        } else {
            left = middle;
            f_left = f_middle;
        }
    }
    (void)f_right;
    phase = 0.5 * (left + right);
    return p + phase * (q - p);
}

double exact_solution(GeometryKind kind, double x, double y)
{
    if (kind == GeometryKind::LShape) {
        constexpr double a = 0.42;
        return std::exp(a * x) * std::cos(a * y);
    }
    if (kind == GeometryKind::Circle) {
        const double X = x - kCircleCx;
        const double Y = y - kCircleCy;
        const double r4 = kCircleRadius * kCircleRadius
                        * kCircleRadius * kCircleRadius;
        return kCircleGaugeConstant
             + (X * X * X * X - 6.0 * X * X * Y * Y
                + Y * Y * Y * Y) / r4;
    }
    constexpr double a = 0.42;
    return std::exp(a * x) * std::cos(a * y)
         + 0.08 * (x * x - y * y) + 0.11 * x - 0.07 * y;
}

Eigen::Vector2d exact_gradient(GeometryKind kind, double x, double y)
{
    if (kind == GeometryKind::LShape) {
        constexpr double a = 0.42;
        const double exponential = std::exp(a * x);
        return Eigen::Vector2d(
            a * exponential * std::cos(a * y),
           -a * exponential * std::sin(a * y));
    }
    if (kind == GeometryKind::Circle) {
        const double X = x - kCircleCx;
        const double Y = y - kCircleCy;
        const double r4 = kCircleRadius * kCircleRadius
                        * kCircleRadius * kCircleRadius;
        return Eigen::Vector2d(
            (4.0 * X * X * X - 12.0 * X * Y * Y) / r4,
            (4.0 * Y * Y * Y - 12.0 * X * X * Y) / r4);
    }
    constexpr double a = 0.42;
    const double exponential = std::exp(a * x);
    return Eigen::Vector2d(
        a * exponential * std::cos(a * y) + 0.16 * x + 0.11,
       -a * exponential * std::sin(a * y) - 0.16 * y - 0.07);
}

std::string exact_solution_name(GeometryKind kind)
{
    if (kind == GeometryKind::LShape)
        return "exp_cos_0p42";
    if (kind == GeometryKind::Circle)
        return "circle_quartic_gauge";
    return "mixed_exp_cos_harmonic";
}

Eigen::Vector2d tangent_from_normal(const Eigen::Vector2d& normal)
{
    return Eigen::Vector2d(-normal[1], normal[0]);
}

Eigen::VectorXd harmonic_basis(double s, double r, int degree)
{
    Eigen::VectorXd values(2 * degree + 1);
    values.setZero();
    values[0] = 1.0;
    std::complex<double> z(s, r);
    std::complex<double> power(1.0, 0.0);
    for (int k = 1; k <= degree; ++k) {
        power *= z;
        values[2 * k - 1] = power.real();
        values[2 * k] = power.imag();
    }
    return values;
}

Eigen::Matrix<double, 6, 1> total_quadratic_basis(double x, double y)
{
    Eigen::Matrix<double, 6, 1> values;
    values << 1.0, x, y, x * x, x * y, y * y;
    return values;
}

Eigen::MatrixXd harmonic_gradient(double s, double r, int degree)
{
    Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(2, 2 * degree + 1);
    std::complex<double> z(s, r);
    std::complex<double> power(1.0, 0.0);
    for (int k = 1; k <= degree; ++k) {
        if (k > 1)
            power *= z;
        const int re = 2 * k - 1;
        const int im = 2 * k;
        gradient(0, re) = k * power.real();
        gradient(1, re) = -k * power.imag();
        gradient(0, im) = k * power.imag();
        gradient(1, im) = k * power.real();
    }
    return gradient;
}

Eigen::MatrixXd pseudo_inverse(const Eigen::MatrixXd& matrix,
                               double rcond,
                               double* condition = nullptr)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd singular = svd.singularValues();
    if (singular.size() == 0 || singular[0] == 0.0)
        throw std::runtime_error("pseudoinverse received a zero matrix");
    const double cutoff = rcond * singular[0];
    Eigen::VectorXd inverse = singular;
    for (int i = 0; i < inverse.size(); ++i)
        inverse[i] = singular[i] > cutoff ? 1.0 / singular[i] : 0.0;
    if (condition != nullptr) {
        *condition = singular[singular.size() - 1] > 0.0
            ? singular[0] / singular[singular.size() - 1]
            : std::numeric_limits<double>::infinity();
    }
    return svd.matrixV() * inverse.asDiagonal() * svd.matrixU().transpose();
}

std::array<double, 4> lagrange4_weights(double fraction)
{
    constexpr std::array<double, 4> nodes{{-1.0, 0.0, 1.0, 2.0}};
    std::array<double, 4> weights{{1.0, 1.0, 1.0, 1.0}};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (i != j)
                weights[i] *= (fraction - nodes[j]) / (nodes[i] - nodes[j]);
    return weights;
}

std::array<double, 3> lagrange3_weights(double fraction)
{
    constexpr std::array<double, 3> nodes{{-1.0, 0.0, 1.0}};
    std::array<double, 3> weights{{1.0, 1.0, 1.0}};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (i != j)
                weights[i] *= (fraction - nodes[j]) / (nodes[i] - nodes[j]);
    return weights;
}

struct GridEdgeCrossing {
    int axis = 0;
    std::array<int, 2> a{{0, 0}};
    std::array<int, 2> b{{0, 0}};
    bool inside_a = false;
    Eigen::Vector2d gamma = Eigen::Vector2d::Zero();
    double phase = 0.0;
    Eigen::Vector2d normal = Eigen::Vector2d::Zero();
    Eigen::Vector2d tangent = Eigen::Vector2d::Zero();
    int boundary_segment = -1;
    double weight = 0.0;
    int nearest_dof = -1;
    Eigen::VectorXd eval_a;
    Eigen::VectorXd eval_b;
    Eigen::VectorXd quadratic_eval_a;
    Eigen::VectorXd quadratic_eval_b;
    Eigen::VectorXd cubic_eval_a;
    Eigen::VectorXd cubic_eval_b;
    Eigen::VectorXd corner_fit_eval_a;
    Eigen::VectorXd corner_fit_eval_b;
};

struct InterfaceDof {
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    Eigen::Vector2d normal = Eigen::Vector2d::Zero();
    Eigen::Vector2d tangent = Eigen::Vector2d::Zero();
    int boundary_segment = -1;
    double weight = 0.0;
    std::vector<int> neighbor_ids;
    Eigen::MatrixXd value_map;
    Eigen::MatrixXd normal_map;
    double condition = 0.0;
    std::vector<int> quadratic_neighbor_ids;
    Eigen::MatrixXd quadratic_value_map;
    Eigen::MatrixXd quadratic_normal_map;
    double quadratic_condition = 0.0;
    std::vector<int> cubic_neighbor_ids;
    Eigen::MatrixXd cubic_value_map;
    Eigen::MatrixXd cubic_normal_map;
    double cubic_condition = 0.0;
    bool cubic_use_full_fit = false;
    std::vector<int> corner_fit_neighbor_ids;
    Eigen::MatrixXd corner_fit_value_map;
    Eigen::MatrixXd corner_fit_normal_map;
    double corner_fit_condition = 0.0;
};

struct TraceTemplate {
    std::vector<int> ids;
    std::vector<double> weights;
    Eigen::MatrixXd correction;
};

class PythonCompatibleHarmonicJet2D {
public:
    PythonCompatibleHarmonicJet2D(int n_intervals,
                                  GeometryKind geometry,
                                  InterfaceDofMode dof_mode,
                                  RestrictMode restrict_mode,
                                  double uniform_dof_ratio,
                                  int degree,
                                  int neighbor_count,
                                  int derivative_count,
                                  SpreadCorrectionMode spread_mode,
                                  int quadratic_spread_neighbor_count,
                                  int quadratic_spread_derivative_count,
                                  int cubic_spread_neighbor_count,
                                  int cubic_spread_derivative_count,
                                  int corner_fit_degree,
                                  int corner_fit_neighbor_count,
                                  int corner_fit_derivative_count)
        : n_(n_intervals)
        , h_((geometry_box_max(geometry) - geometry_box_min(geometry))
             / static_cast<double>(n_intervals))
        , geometry_(geometry)
        , dof_mode_(dof_mode)
        , restrict_mode_(restrict_mode)
        , uniform_dof_ratio_(uniform_dof_ratio)
        , degree_(degree)
        , dim_(2 * degree + 1)
        , requested_neighbors_(neighbor_count)
        , requested_derivative_neighbors_(derivative_count)
        , spread_mode_(spread_mode)
        , quadratic_spread_neighbor_count_(quadratic_spread_neighbor_count)
        , quadratic_spread_derivative_count_(quadratic_spread_derivative_count)
        , cubic_spread_neighbor_count_(cubic_spread_neighbor_count)
        , cubic_spread_derivative_count_(cubic_spread_derivative_count)
        , corner_fit_degree_(corner_fit_degree)
        , corner_fit_dim_(2 * corner_fit_degree + 1)
        , corner_fit_neighbor_count_(corner_fit_neighbor_count)
        , corner_fit_derivative_count_(corner_fit_derivative_count)
        , grid_({geometry_box_min(geometry), geometry_box_min(geometry)},
                {h_, h_}, {n_, n_}, DofLayout2D::Node)
        , bulk_solver_(grid_, ZfftBcType::Dirichlet, 0.0, 2)
    {
        if (n_ < 24)
            throw std::invalid_argument("N must be at least 24");
        if (degree_ < 2 || degree_ > 5)
            throw std::invalid_argument("harmonic jet degree must be in [2,5]");
        if (corner_fit_degree_ < 2 || corner_fit_degree_ > 5)
            throw std::invalid_argument(
                "corner Cauchy fit degree must be in [2,5]");
        if (!(uniform_dof_ratio_ > 0.0)
            || !std::isfinite(uniform_dof_ratio_)) {
            throw std::invalid_argument(
                "uniform DOF ratio must be finite and positive");
        }
        if (requested_neighbors_ <= 0 || requested_derivative_neighbors_ <= 0
            || requested_derivative_neighbors_ > requested_neighbors_) {
            throw std::invalid_argument("invalid harmonic-jet neighbor counts");
        }
        if (quadratic_spread_neighbor_count_ <= 0
            || quadratic_spread_derivative_count_ <= 0
            || quadratic_spread_derivative_count_
                   > quadratic_spread_neighbor_count_) {
            throw std::invalid_argument(
                "invalid quadratic-spread neighbor counts");
        }
        if (cubic_spread_neighbor_count_ <= 0
            || cubic_spread_derivative_count_ <= 0
            || cubic_spread_derivative_count_
                   > cubic_spread_neighbor_count_) {
            throw std::invalid_argument(
                "invalid cubic-spread neighbor counts");
        }
        if (corner_fit_neighbor_count_ <= 0
            || corner_fit_derivative_count_ <= 0
            || corner_fit_derivative_count_ > corner_fit_neighbor_count_) {
            throw std::invalid_argument(
                "invalid corner-fit neighbor counts");
        }
        build_inside_mask();
        build_crossings();
        build_interface_dofs();
        neighbor_count_ = std::min(requested_neighbors_, size());
        derivative_count_ = std::min(requested_derivative_neighbors_,
                                     neighbor_count_);
        build_cauchy_maps();
        if (spread_mode_
                == SpreadCorrectionMode::QuadraticHarmonicAtOppositeNode
            || !restrict_uses_cubic_grid()) {
            quadratic_spread_neighbor_count_ = std::min(
                quadratic_spread_neighbor_count_, size());
            quadratic_spread_derivative_count_ = std::min(
                quadratic_spread_derivative_count_,
                quadratic_spread_neighbor_count_);
            build_quadratic_spread_maps();
        }
        if (spread_mode_
            == SpreadCorrectionMode::CubicHarmonicAtOppositeNode) {
            cubic_spread_neighbor_count_ = std::min(
                cubic_spread_neighbor_count_, size());
            cubic_spread_derivative_count_ = std::min(
                cubic_spread_derivative_count_,
                cubic_spread_neighbor_count_);
            build_cubic_spread_maps();
            if (geometry_ == GeometryKind::LShape
                && hybrid_full_fit_dofs() > 0) {
                corner_fit_neighbor_count_ = std::min(
                    corner_fit_neighbor_count_, size());
                corner_fit_derivative_count_ = std::min(
                    corner_fit_derivative_count_,
                    corner_fit_neighbor_count_);
                if (corner_fit_neighbor_count_
                        + corner_fit_derivative_count_ < corner_fit_dim_) {
                    throw std::invalid_argument(
                        "corner Cauchy fit has too few samples");
                }
                build_corner_cauchy_maps();
            }
        }
        build_crossing_rows();
        build_trace_templates();
        if (restrict_mode_ != RestrictMode::SixPointQuadraticExterior)
            build_joint_value_fit();
    }

    int size() const { return static_cast<int>(dofs_.size()); }
    int num_crossings() const { return static_cast<int>(crossings_.size()); }
    int hybrid_full_fit_dofs() const
    {
        return static_cast<int>(std::count_if(
            dofs_.begin(), dofs_.end(), [](const InterfaceDof& dof) {
                return dof.cubic_use_full_fit;
            }));
    }
    int n_intervals() const { return n_; }
    double h() const { return h_; }
    double boundary_length() const { return boundary_length_; }
    const CartesianGrid2D& grid() const { return grid_; }
    const std::vector<GridEdgeCrossing>& crossings() const { return crossings_; }
    const std::vector<InterfaceDof>& dofs() const { return dofs_; }

    double spread_condition(const InterfaceDof& dof) const
    {
        if (spread_mode_
            == SpreadCorrectionMode::QuadraticHarmonicAtOppositeNode) {
            return dof.quadratic_condition;
        }
        if (spread_mode_
            == SpreadCorrectionMode::CubicHarmonicAtOppositeNode) {
            return dof.cubic_use_full_fit
                ? dof.corner_fit_condition : dof.cubic_condition;
        }
        return dof.condition;
    }

    double restrict_condition(const InterfaceDof& dof) const
    {
        if (restrict_uses_shared_cubic_spread()) {
            return dof.cubic_use_full_fit
                ? dof.corner_fit_condition : dof.cubic_condition;
        }
        return restrict_uses_cubic_grid()
            ? dof.condition : dof.quadratic_condition;
    }

    std::string restrict_cauchy_source() const
    {
        if (restrict_mode_ == RestrictMode::SixPointQuadraticExterior)
            return "six_point_quadratic";
        return restrict_uses_shared_cubic_spread()
            ? "shared_cubic_spread" : "restrict_specific";
    }

    Eigen::VectorXd boundary_weights() const
    {
        Eigen::VectorXd result(size());
        for (int i = 0; i < size(); ++i)
            result[i] = dofs_[static_cast<std::size_t>(i)].weight;
        return result;
    }

    Eigen::VectorXd exact_trace() const
    {
        Eigen::VectorXd result(size());
        for (int i = 0; i < size(); ++i) {
            const Eigen::Vector2d& p = dofs_[static_cast<std::size_t>(i)].point;
            result[i] = exact_solution(geometry_, p[0], p[1]);
        }
        return result;
    }

    Eigen::VectorXd neumann_data() const
    {
        Eigen::VectorXd result(size());
        for (int i = 0; i < size(); ++i) {
            const InterfaceDof& dof = dofs_[static_cast<std::size_t>(i)];
            result[i] = exact_gradient(
                            geometry_, dof.point[0], dof.point[1])
                            .dot(dof.normal);
        }
        return result;
    }

    Eigen::MatrixXd correction_coefficients(
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        if (value_jump.size() != size() || normal_jump.size() != size())
            throw std::invalid_argument("jump vector size mismatch");
        Eigen::MatrixXd coefficients(size(), dim_);
        for (int i = 0; i < size(); ++i) {
            const InterfaceDof& dof = dofs_[static_cast<std::size_t>(i)];
            Eigen::VectorXd local_value(neighbor_count_);
            Eigen::VectorXd local_normal(neighbor_count_);
            for (int k = 0; k < neighbor_count_; ++k) {
                const int neighbor = dof.neighbor_ids[static_cast<std::size_t>(k)];
                local_value[k] = value_jump[neighbor];
                local_normal[k] = normal_jump[neighbor];
            }
            coefficients.row(i) =
                (dof.value_map * local_value
                 + dof.normal_map * local_normal).transpose();
        }
        return coefficients;
    }

    Eigen::MatrixXd quadratic_correction_coefficients(
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        constexpr int quadratic_dim = 5;
        if (value_jump.size() != size() || normal_jump.size() != size())
            throw std::invalid_argument("jump vector size mismatch");
        Eigen::MatrixXd coefficients(size(), quadratic_dim);
        for (int i = 0; i < size(); ++i) {
            const InterfaceDof& dof = dofs_[static_cast<std::size_t>(i)];
            Eigen::VectorXd local_value(quadratic_spread_neighbor_count_);
            Eigen::VectorXd local_normal(quadratic_spread_neighbor_count_);
            for (int k = 0; k < quadratic_spread_neighbor_count_; ++k) {
                const int neighbor = dof.quadratic_neighbor_ids[
                    static_cast<std::size_t>(k)];
                local_value[k] = value_jump[neighbor];
                local_normal[k] = normal_jump[neighbor];
            }
            coefficients.row(i) =
                (dof.quadratic_value_map * local_value
                 + dof.quadratic_normal_map * local_normal).transpose();
        }
        return coefficients;
    }

    Eigen::MatrixXd cubic_correction_coefficients(
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        constexpr int cubic_dim = 7;
        if (value_jump.size() != size() || normal_jump.size() != size())
            throw std::invalid_argument("jump vector size mismatch");
        Eigen::MatrixXd coefficients(size(), cubic_dim);
        for (int i = 0; i < size(); ++i) {
            const InterfaceDof& dof = dofs_[static_cast<std::size_t>(i)];
            Eigen::VectorXd local_value(cubic_spread_neighbor_count_);
            Eigen::VectorXd local_normal(cubic_spread_neighbor_count_);
            for (int k = 0; k < cubic_spread_neighbor_count_; ++k) {
                const int neighbor = dof.cubic_neighbor_ids[
                    static_cast<std::size_t>(k)];
                local_value[k] = value_jump[neighbor];
                local_normal[k] = normal_jump[neighbor];
            }
            coefficients.row(i) =
                (dof.cubic_value_map * local_value
                 + dof.cubic_normal_map * local_normal).transpose();
        }
        return coefficients;
    }

    Eigen::MatrixXd corner_fit_correction_coefficients(
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        if (value_jump.size() != size() || normal_jump.size() != size())
            throw std::invalid_argument("jump vector size mismatch");
        Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(
            size(), corner_fit_dim_);
        for (int i = 0; i < size(); ++i) {
            const InterfaceDof& dof = dofs_[static_cast<std::size_t>(i)];
            if (!dof.cubic_use_full_fit)
                continue;
            Eigen::VectorXd local_value(corner_fit_neighbor_count_);
            Eigen::VectorXd local_normal(corner_fit_neighbor_count_);
            for (int k = 0; k < corner_fit_neighbor_count_; ++k) {
                const int neighbor = dof.corner_fit_neighbor_ids[
                    static_cast<std::size_t>(k)];
                local_value[k] = value_jump[neighbor];
                local_normal[k] = normal_jump[neighbor];
            }
            coefficients.row(i) =
                (dof.corner_fit_value_map * local_value
                 + dof.corner_fit_normal_map * local_normal).transpose();
        }
        return coefficients;
    }

    std::vector<Eigen::VectorXd> cubic_spread_correction_coefficients(
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        const Eigen::MatrixXd cubic_coefficients =
            cubic_correction_coefficients(value_jump, normal_jump);
        const bool has_corner_fit = hybrid_full_fit_dofs() > 0;
        Eigen::MatrixXd corner_coefficients;
        if (has_corner_fit) {
            corner_coefficients = corner_fit_correction_coefficients(
                value_jump, normal_jump);
        }

        std::vector<Eigen::VectorXd> coefficients(
            static_cast<std::size_t>(size()));
        for (int i = 0; i < size(); ++i) {
            const InterfaceDof& dof = dofs_[static_cast<std::size_t>(i)];
            if (dof.cubic_use_full_fit) {
                coefficients[static_cast<std::size_t>(i)] =
                    corner_coefficients.row(i).transpose();
            } else {
                coefficients[static_cast<std::size_t>(i)] =
                    cubic_coefficients.row(i).transpose();
            }
        }
        return coefficients;
    }

    std::vector<Eigen::VectorXd> restrict_correction_coefficients(
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        if (restrict_uses_shared_cubic_spread()) {
            return cubic_spread_correction_coefficients(
                value_jump, normal_jump);
        }

        const Eigen::MatrixXd matrix = restrict_uses_cubic_grid()
            ? correction_coefficients(value_jump, normal_jump)
            : quadratic_correction_coefficients(value_jump, normal_jump);
        std::vector<Eigen::VectorXd> coefficients(
            static_cast<std::size_t>(size()));
        for (int i = 0; i < size(); ++i) {
            coefficients[static_cast<std::size_t>(i)] =
                matrix.row(i).transpose();
        }
        return coefficients;
    }

    Eigen::VectorXd rhs_from_jumps(const Eigen::VectorXd& value_jump,
                                   const Eigen::VectorXd& normal_jump) const
    {
        if (value_jump.size() != size() || normal_jump.size() != size())
            throw std::invalid_argument("jump vector size mismatch");
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(grid_.num_dofs());
        const double inv_h2 = 1.0 / (h_ * h_);
        if (spread_mode_ == SpreadCorrectionMode::CrossingDensity) {
            for (const GridEdgeCrossing& crossing : crossings_) {
                const double side_sign = crossing.inside_a ? 1.0 : -1.0;
                const double density = value_jump[crossing.nearest_dof];
                rhs[grid_.index(crossing.a[0], crossing.a[1])] +=
                    side_sign * density * inv_h2;
                rhs[grid_.index(crossing.b[0], crossing.b[1])] -=
                    side_sign * density * inv_h2;
            }
            return rhs;
        }

        if (spread_mode_
            == SpreadCorrectionMode::QuadraticHarmonicAtOppositeNode) {
            const Eigen::MatrixXd coefficients =
                quadratic_correction_coefficients(value_jump, normal_jump);
            for (const GridEdgeCrossing& crossing : crossings_) {
                const Eigen::VectorXd coeff = coefficients.row(
                    crossing.nearest_dof).transpose();
                rhs[grid_.index(crossing.a[0], crossing.a[1])] +=
                    crossing.quadratic_eval_a.dot(coeff) * inv_h2;
                rhs[grid_.index(crossing.b[0], crossing.b[1])] +=
                    crossing.quadratic_eval_b.dot(coeff) * inv_h2;
            }
            return rhs;
        }

        if (spread_mode_
            == SpreadCorrectionMode::CubicHarmonicAtOppositeNode) {
            const std::vector<Eigen::VectorXd> coefficients =
                cubic_spread_correction_coefficients(
                    value_jump, normal_jump);
            for (const GridEdgeCrossing& crossing : crossings_) {
                const InterfaceDof& dof = dofs_[static_cast<std::size_t>(
                    crossing.nearest_dof)];
                const Eigen::VectorXd& coeff = coefficients[
                    static_cast<std::size_t>(crossing.nearest_dof)];
                if (dof.cubic_use_full_fit) {
                    rhs[grid_.index(crossing.a[0], crossing.a[1])] +=
                        crossing.corner_fit_eval_a.dot(coeff) * inv_h2;
                    rhs[grid_.index(crossing.b[0], crossing.b[1])] +=
                        crossing.corner_fit_eval_b.dot(coeff) * inv_h2;
                } else {
                    rhs[grid_.index(crossing.a[0], crossing.a[1])] +=
                        crossing.cubic_eval_a.dot(coeff) * inv_h2;
                    rhs[grid_.index(crossing.b[0], crossing.b[1])] +=
                        crossing.cubic_eval_b.dot(coeff) * inv_h2;
                }
            }
            return rhs;
        }

        const Eigen::MatrixXd coefficients =
            correction_coefficients(value_jump, normal_jump);
        for (const GridEdgeCrossing& crossing : crossings_) {
            const Eigen::VectorXd coeff =
                coefficients.row(crossing.nearest_dof).transpose();
            rhs[grid_.index(crossing.a[0], crossing.a[1])] +=
                crossing.eval_a.dot(coeff) * inv_h2;
            rhs[grid_.index(crossing.b[0], crossing.b[1])] +=
                crossing.eval_b.dot(coeff) * inv_h2;
        }
        return rhs;
    }

    Eigen::VectorXd potential(const Eigen::VectorXd& value_jump,
                              const Eigen::VectorXd& normal_jump) const
    {
        const Eigen::VectorXd rhs = rhs_from_jumps(value_jump, normal_jump);
        Eigen::VectorXd result;
        // The project bulk-solver API takes the discrete Laplacian RHS,
        // whereas the Python tutorial forms the right-hand side of
        // -Delta_h u = rhs.  Negate here to preserve the Python convention.
        bulk_solver_.solve(-rhs, result);
        return result;
    }

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> one_sided_samples(
        const Eigen::VectorXd& potential_values,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        const std::vector<Eigen::VectorXd> coefficients =
            restrict_correction_coefficients(value_jump, normal_jump);
        const int layer_count = restrict_layer_count();
        Eigen::MatrixXd inside(size(), layer_count);
        Eigen::MatrixXd outside(size(), layer_count);
        for (int i = 0; i < size(); ++i) {
            const Eigen::VectorXd& coeff =
                coefficients[static_cast<std::size_t>(i)];
            for (int side = 0; side < 2; ++side) {
                for (int layer = 0; layer < layer_count; ++layer) {
                    const TraceTemplate& entry = trace_template(i, side, layer);
                    double value = 0.0;
                    for (int v = 0; v < static_cast<int>(entry.ids.size()); ++v) {
                        value += entry.weights[static_cast<std::size_t>(v)]
                               * (potential_values[entry.ids[static_cast<std::size_t>(v)]]
                                  + entry.correction.row(v).dot(coeff));
                    }
                    if (side == 0)
                        inside(i, layer) = value;
                    else
                        outside(i, layer) = value;
                }
            }
        }
        return {inside, outside};
    }

    Eigen::VectorXd exterior_trace(const Eigen::VectorXd& potential_values,
                                   const Eigen::VectorXd& value_jump,
                                   const Eigen::VectorXd& normal_jump,
                                   bool diagnostics = false) const
    {
        if (restrict_mode_ == RestrictMode::SixPointQuadraticExterior) {
            const Eigen::MatrixXd coefficients =
                quadratic_correction_coefficients(value_jump, normal_jump);
            Eigen::VectorXd trace(size());
            for (int i = 0; i < size(); ++i) {
                const TraceTemplate& entry =
                    trace_templates_[static_cast<std::size_t>(i)];
                const Eigen::VectorXd coeff = coefficients.row(i).transpose();
                double value = 0.0;
                for (int v = 0; v < static_cast<int>(entry.ids.size()); ++v) {
                    value += entry.weights[static_cast<std::size_t>(v)]
                           * (potential_values[
                                  entry.ids[static_cast<std::size_t>(v)]]
                              + entry.correction.row(v).dot(coeff));
                }
                trace[i] = value;
            }
            if (diagnostics)
                last_trace_fit_rel_l2_ = 0.0;
            return trace;
        }

        auto samples_pair = one_sided_samples(
            potential_values, value_jump, normal_jump);
        const Eigen::MatrixXd& inside = samples_pair.first;
        const Eigen::MatrixXd& outside = samples_pair.second;
        Eigen::VectorXd trace(size());
        double residual_sq = 0.0;
        double samples_sq = 0.0;
        const int layer_count = restrict_layer_count();
        for (int i = 0; i < size(); ++i) {
            Eigen::VectorXd samples(2 * layer_count);
            for (int layer = 0; layer < layer_count; ++layer) {
                const double rho_inside =
                    -kNormalLayers[static_cast<std::size_t>(layer)] * h_;
                samples[layer] = inside(i, layer)
                    - value_jump[i] - rho_inside * normal_jump[i];
                samples[layer_count + layer] = outside(i, layer);
            }
            trace[i] = c0_weights_.dot(samples);
            if (diagnostics) {
                const Eigen::VectorXd fitted = trace_projection_ * samples;
                residual_sq += (samples - fitted).squaredNorm();
                samples_sq += samples.squaredNorm();
            }
        }
        if (diagnostics) {
            last_trace_fit_rel_l2_ = std::sqrt(residual_sq)
                / std::max(std::sqrt(samples_sq), 1.0e-30);
        }
        return trace;
    }

    Eigen::VectorXd exterior_normal_trace(
        const Eigen::VectorXd& potential_values,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        bool diagnostics = false) const
    {
        if (restrict_mode_ == RestrictMode::SixPointQuadraticExterior) {
            throw std::invalid_argument(
                "six_point_quadratic_exterior does not recover a normal trace");
        }

        auto samples_pair = one_sided_samples(
            potential_values, value_jump, normal_jump);
        const Eigen::MatrixXd& inside = samples_pair.first;
        const Eigen::MatrixXd& outside = samples_pair.second;
        Eigen::VectorXd normal_trace(size());
        double residual_sq = 0.0;
        double samples_sq = 0.0;
        const int layer_count = restrict_layer_count();
        for (int i = 0; i < size(); ++i) {
            Eigen::VectorXd samples(2 * layer_count);
            for (int layer = 0; layer < layer_count; ++layer) {
                const double rho_inside =
                    -kNormalLayers[static_cast<std::size_t>(layer)] * h_;
                samples[layer] = inside(i, layer)
                    - value_jump[i] - rho_inside * normal_jump[i];
                samples[layer_count + layer] = outside(i, layer);
            }
            normal_trace[i] = c1_weights_.dot(samples) / h_;
            if (diagnostics) {
                const Eigen::VectorXd fitted = trace_projection_ * samples;
                residual_sq += (samples - fitted).squaredNorm();
                samples_sq += samples.squaredNorm();
            }
        }
        if (diagnostics) {
            last_trace_fit_rel_l2_ = std::sqrt(residual_sq)
                / std::max(std::sqrt(samples_sq), 1.0e-30);
        }
        return normal_trace;
    }

    Eigen::VectorXd interior_trace(const Eigen::VectorXd& potential_values,
                                   const Eigen::VectorXd& value_jump,
                                   const Eigen::VectorXd& normal_jump,
                                   bool diagnostics = false) const
    {
        if (restrict_mode_ == RestrictMode::SixPointQuadraticExterior) {
            Eigen::VectorXd trace = exterior_trace(
                potential_values, value_jump, normal_jump, diagnostics);
            trace += value_jump;
            return trace;
        }

        auto samples_pair = one_sided_samples(
            potential_values, value_jump, normal_jump);
        const Eigen::MatrixXd& inside = samples_pair.first;
        const Eigen::MatrixXd& outside = samples_pair.second;
        Eigen::VectorXd trace(size());
        double residual_sq = 0.0;
        double samples_sq = 0.0;
        const int layer_count = restrict_layer_count();
        for (int i = 0; i < size(); ++i) {
            Eigen::VectorXd samples(2 * layer_count);
            for (int layer = 0; layer < layer_count; ++layer) {
                const double rho_outside =
                    kNormalLayers[static_cast<std::size_t>(layer)] * h_;
                samples[layer] = inside(i, layer);
                samples[layer_count + layer] = outside(i, layer)
                    + value_jump[i] + rho_outside * normal_jump[i];
            }
            trace[i] = c0_weights_.dot(samples);
            if (diagnostics) {
                const Eigen::VectorXd fitted = trace_projection_ * samples;
                residual_sq += (samples - fitted).squaredNorm();
                samples_sq += samples.squaredNorm();
            }
        }
        if (diagnostics) {
            last_trace_fit_rel_l2_ = std::sqrt(residual_sq)
                / std::max(std::sqrt(samples_sq), 1.0e-30);
        }
        return trace;
    }

    Eigen::VectorXd interior_normal_trace(
        const Eigen::VectorXd& potential_values,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        bool diagnostics = false) const
    {
        if (restrict_mode_ == RestrictMode::SixPointQuadraticExterior) {
            throw std::invalid_argument(
                "six_point_quadratic_exterior does not recover a normal trace");
        }

        auto samples_pair = one_sided_samples(
            potential_values, value_jump, normal_jump);
        const Eigen::MatrixXd& inside = samples_pair.first;
        const Eigen::MatrixXd& outside = samples_pair.second;
        Eigen::VectorXd normal_trace(size());
        double residual_sq = 0.0;
        double samples_sq = 0.0;
        const int layer_count = restrict_layer_count();
        for (int i = 0; i < size(); ++i) {
            Eigen::VectorXd samples(2 * layer_count);
            for (int layer = 0; layer < layer_count; ++layer) {
                const double rho_outside =
                    kNormalLayers[static_cast<std::size_t>(layer)] * h_;
                samples[layer] = inside(i, layer);
                samples[layer_count + layer] = outside(i, layer)
                    + value_jump[i] + rho_outside * normal_jump[i];
            }
            normal_trace[i] = c1_weights_.dot(samples) / h_;
            if (diagnostics) {
                const Eigen::VectorXd fitted = trace_projection_ * samples;
                residual_sq += (samples - fitted).squaredNorm();
                samples_sq += samples.squaredNorm();
            }
        }
        if (diagnostics) {
            last_trace_fit_rel_l2_ = std::sqrt(residual_sq)
                / std::max(std::sqrt(samples_sq), 1.0e-30);
        }
        return normal_trace;
    }

    double last_trace_fit_rel_l2() const { return last_trace_fit_rel_l2_; }

    bool inside_node(int node) const
    {
        return inside_mask_[static_cast<std::size_t>(node)] != 0;
    }

    double nearest_crossing_distance(const Eigen::Vector2d& point) const
    {
        double best = std::numeric_limits<double>::infinity();
        for (const GridEdgeCrossing& crossing : crossings_)
            best = std::min(best, (point - crossing.gamma).norm());
        return best;
    }

private:
    bool restrict_uses_shared_cubic_spread() const
    {
        return spread_mode_
                == SpreadCorrectionMode::CubicHarmonicAtOppositeNode
            && restrict_mode_ != RestrictMode::SixPointQuadraticExterior;
    }

    bool restrict_uses_cubic_grid() const
    {
        return restrict_mode_ == RestrictMode::BicubicCubicNormal
            || restrict_mode_ == RestrictMode::BicubicQuadraticNormal;
    }

    bool restrict_uses_cubic_normal_fit() const
    {
        return restrict_mode_ == RestrictMode::BicubicCubicNormal
            || restrict_mode_ == RestrictMode::BiquadraticCubicNormal;
    }

    int restrict_layer_count() const
    {
        if (restrict_mode_ == RestrictMode::SixPointQuadraticExterior)
            return 0;
        if (restrict_mode_ == RestrictMode::BiquadraticQuadraticTwoLayer)
            return 2;
        if (restrict_uses_cubic_normal_fit())
            return 4;
        return 3;
    }

    int restrict_grid_side_count() const
    {
        if (restrict_uses_cubic_grid())
            return 4;
        if (restrict_mode_ == RestrictMode::SixPointQuadraticExterior)
            return 0;
        return 3;
    }

    int restrict_normal_degree() const
    {
        return restrict_uses_cubic_normal_fit() ? 3 : 2;
    }

    int restrict_cauchy_degree(const InterfaceDof& dof) const
    {
        if (restrict_uses_shared_cubic_spread()) {
            return dof.cubic_use_full_fit ? corner_fit_degree_ : 3;
        }
        return restrict_uses_cubic_grid() ? degree_ : 2;
    }

    void build_inside_mask()
    {
        inside_mask_.resize(static_cast<std::size_t>(grid_.num_dofs()), 0);
        for (int j = 0; j <= n_; ++j) {
            for (int i = 0; i <= n_; ++i) {
                const auto coord = grid_.coord(i, j);
                inside_mask_[static_cast<std::size_t>(grid_.index(i, j))] =
                    geometry_inside(geometry_, Eigen::Vector2d(coord[0], coord[1]))
                        ? 1 : 0;
            }
        }
    }

    void build_crossings()
    {
        crossings_.clear();
        for (int axis = 0; axis < 2; ++axis) {
            const int imax = axis == 0 ? n_ - 1 : n_;
            const int jmax = axis == 0 ? n_ : n_ - 1;
            for (int i = 0; i <= imax; ++i) {
                for (int j = 0; j <= jmax; ++j) {
                    std::array<int, 2> a{{i, j}};
                    std::array<int, 2> b = a;
                    ++b[static_cast<std::size_t>(axis)];
                    const bool inside_a = inside_node(grid_.index(a[0], a[1]));
                    const bool inside_b = inside_node(grid_.index(b[0], b[1]));
                    if (inside_a == inside_b)
                        continue;
                    if (!(a[0] > 0 && a[0] < n_ && a[1] > 0 && a[1] < n_
                          && b[0] > 0 && b[0] < n_ && b[1] > 0 && b[1] < n_)) {
                        continue;
                    }
                    const auto ca = grid_.coord(a[0], a[1]);
                    const auto cb = grid_.coord(b[0], b[1]);
                    GridEdgeCrossing crossing;
                    crossing.axis = axis;
                    crossing.a = a;
                    crossing.b = b;
                    crossing.inside_a = inside_a;
                    crossing.gamma = segment_intersection(
                        geometry_,
                        Eigen::Vector2d(ca[0], ca[1]),
                        Eigen::Vector2d(cb[0], cb[1]),
                        crossing.phase);
                    crossing.normal = geometry_normal(geometry_, crossing.gamma);
                    crossing.tangent = tangent_from_normal(crossing.normal);
                    if (geometry_ == GeometryKind::LShape) {
                        crossing.boundary_segment =
                            lshape_boundary_segment(crossing.gamma);
                    }
                    crossing.weight = h_ /
                        std::max(std::abs(crossing.normal[0])
                               + std::abs(crossing.normal[1]), 1.0e-14);
                    crossings_.push_back(std::move(crossing));
                }
            }
        }
        if (crossings_.empty())
            throw std::runtime_error("no interface crossings were found");
        boundary_length_ = 0.0;
        for (const GridEdgeCrossing& crossing : crossings_)
            boundary_length_ += crossing.weight;
    }

    void build_interface_dofs()
    {
        dofs_.clear();
        dofs_.reserve(crossings_.size());
        if (dof_mode_ == InterfaceDofMode::Crossing) {
            for (const GridEdgeCrossing& crossing : crossings_) {
                InterfaceDof dof;
                dof.point = crossing.gamma;
                dof.normal = crossing.normal;
                dof.tangent = crossing.tangent;
                dof.boundary_segment = crossing.boundary_segment;
                dof.weight = crossing.weight;
                dofs_.push_back(std::move(dof));
            }
        } else if (geometry_ == GeometryKind::LShape) {
            const int requested_count = std::max(
                24, static_cast<int>(std::lround(
                    uniform_dof_ratio_
                    * static_cast<double>(crossings_.size()))));
            const auto& vertices = lshape_vertices();
            boundary_length_ = lshape_perimeter();
            for (int edge = 0; edge < static_cast<int>(vertices.size()); ++edge) {
                const Eigen::Vector2d& start =
                    vertices[static_cast<std::size_t>(edge)];
                const Eigen::Vector2d& end = vertices[static_cast<std::size_t>(
                    (edge + 1) % static_cast<int>(vertices.size()))];
                const Eigen::Vector2d delta = end - start;
                const double length = delta.norm();
                const int edge_count = std::max(
                    cubic_spread_neighbor_count_,
                    static_cast<int>(std::lround(
                        static_cast<double>(requested_count)
                        * length / boundary_length_)));
                const double edge_interval =
                    length / static_cast<double>(edge_count);
                const Eigen::Vector2d normal(
                    delta[1] / length, -delta[0] / length);
                for (int k = 0; k < edge_count; ++k) {
                    InterfaceDof dof;
                    dof.point = start
                        + ((static_cast<double>(k) + 0.5)
                           / static_cast<double>(edge_count)) * delta;
                    dof.normal = normal;
                    dof.tangent = tangent_from_normal(normal);
                    dof.boundary_segment = edge;
                    dof.weight = edge_interval;
                    dofs_.push_back(std::move(dof));
                }
            }
        } else {
            // A dense parameter-space polyline supplies an inexpensive
            // approximate arc-length coordinate for the supported curves.
            const int count = std::max(
                5, static_cast<int>(std::lround(
                    uniform_dof_ratio_
                    * static_cast<double>(crossings_.size()))));
            const int fine_count = std::max(4096, 64 * count);
            std::vector<double> cumulative(
                static_cast<std::size_t>(fine_count + 1), 0.0);
            Eigen::Vector2d previous = geometry_point(geometry_, 0.0);
            for (int k = 1; k <= fine_count; ++k) {
                const double theta = 2.0 * kPi * static_cast<double>(k)
                                   / static_cast<double>(fine_count);
                const Eigen::Vector2d current = geometry_point(geometry_, theta);
                cumulative[static_cast<std::size_t>(k)] =
                    cumulative[static_cast<std::size_t>(k - 1)]
                    + (current - previous).norm();
                previous = current;
            }
            boundary_length_ = cumulative.back();
            const double interval_length = boundary_length_
                                         / static_cast<double>(count);
            for (int i = 0; i < count; ++i) {
                const double target = (static_cast<double>(i) + 0.5)
                                    * interval_length;
                const auto upper = std::lower_bound(
                    cumulative.begin(), cumulative.end(), target);
                const int hi = static_cast<int>(upper - cumulative.begin());
                const int lo = std::max(0, hi - 1);
                const double segment_length = cumulative[static_cast<std::size_t>(hi)]
                                            - cumulative[static_cast<std::size_t>(lo)];
                const double fraction = segment_length > 0.0
                    ? (target - cumulative[static_cast<std::size_t>(lo)])
                        / segment_length
                    : 0.0;
                const double theta = 2.0 * kPi
                    * (static_cast<double>(lo) + fraction)
                    / static_cast<double>(fine_count);
                InterfaceDof dof;
                dof.point = geometry_point(geometry_, theta);
                dof.normal = geometry_normal(geometry_, dof.point);
                dof.tangent = tangent_from_normal(dof.normal);
                if (geometry_ == GeometryKind::LShape) {
                    dof.boundary_segment =
                        lshape_boundary_segment(dof.point);
                }
                dof.weight = interval_length;
                dofs_.push_back(std::move(dof));
            }
        }

        if (dofs_.size() < 5)
            throw std::runtime_error("at least five interface DOFs are required");
        for (GridEdgeCrossing& crossing : crossings_) {
            double best_distance = std::numeric_limits<double>::infinity();
            for (int i = 0; i < size(); ++i) {
                if (geometry_ == GeometryKind::LShape
                    && dofs_[static_cast<std::size_t>(i)].boundary_segment
                           != crossing.boundary_segment) {
                    continue;
                }
                const double distance =
                    (crossing.gamma - dofs_[static_cast<std::size_t>(i)].point)
                        .squaredNorm();
                if (distance < best_distance) {
                    best_distance = distance;
                    crossing.nearest_dof = i;
                }
            }
            if (crossing.nearest_dof < 0)
                throw std::runtime_error("crossing has no nearest interface DOF");
        }
    }

    void build_cauchy_maps()
    {
        for (int i = 0; i < size(); ++i) {
            InterfaceDof& center = dofs_[static_cast<std::size_t>(i)];
            std::vector<int> order(static_cast<std::size_t>(size()));
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
                const double dl = (dofs_[static_cast<std::size_t>(lhs)].point
                                  - center.point).squaredNorm();
                const double dr = (dofs_[static_cast<std::size_t>(rhs)].point
                                  - center.point).squaredNorm();
                if (dl != dr)
                    return dl < dr;
                return lhs < rhs;
            });
            center.neighbor_ids.assign(order.begin(),
                                       order.begin() + neighbor_count_);

            const int rows = neighbor_count_ + derivative_count_;
            Eigen::MatrixXd design(rows, dim_);
            Eigen::VectorXd sqrt_weights(rows);
            for (int pos = 0; pos < neighbor_count_; ++pos) {
                const int neighbor = center.neighbor_ids[static_cast<std::size_t>(pos)];
                const Eigen::Vector2d d =
                    (dofs_[static_cast<std::size_t>(neighbor)].point
                     - center.point) / h_;
                const double s = d.dot(center.tangent);
                const double r = d.dot(center.normal);
                design.row(pos) = harmonic_basis(s, r, degree_).transpose();
                sqrt_weights[pos] = std::sqrt(
                    1.0 / std::pow(0.35 + d.norm(), 2));
            }
            for (int pos = 0; pos < derivative_count_; ++pos) {
                const int neighbor = center.neighbor_ids[static_cast<std::size_t>(pos)];
                const InterfaceDof& sample =
                    dofs_[static_cast<std::size_t>(neighbor)];
                const Eigen::Vector2d d = (sample.point - center.point) / h_;
                const double s = d.dot(center.tangent);
                const double r = d.dot(center.normal);
                const Eigen::MatrixXd gradient =
                    harmonic_gradient(s, r, degree_);
                const Eigen::Vector2d components(
                    sample.normal.dot(center.tangent),
                    sample.normal.dot(center.normal));
                design.row(neighbor_count_ + pos) =
                    components.transpose() * gradient;
                sqrt_weights[neighbor_count_ + pos] = std::sqrt(
                    0.85 / std::pow(0.35 + d.norm(), 2));
            }

            Eigen::MatrixXd weighted_design = design;
            for (int row = 0; row < rows; ++row)
                weighted_design.row(row) *= sqrt_weights[row];
            const Eigen::MatrixXd pinv = pseudo_inverse(
                weighted_design, 2.0e-12, &center.condition);
            center.value_map = Eigen::MatrixXd::Zero(dim_, neighbor_count_);
            center.normal_map = Eigen::MatrixXd::Zero(dim_, neighbor_count_);
            for (int pos = 0; pos < neighbor_count_; ++pos)
                center.value_map.col(pos) = pinv.col(pos) * sqrt_weights[pos];
            for (int pos = 0; pos < derivative_count_; ++pos) {
                center.normal_map.col(pos) =
                    pinv.col(neighbor_count_ + pos)
                    * sqrt_weights[neighbor_count_ + pos] * h_;
            }
        }
    }

    void build_corner_cauchy_maps()
    {
        for (int i = 0; i < size(); ++i) {
            InterfaceDof& center = dofs_[static_cast<std::size_t>(i)];
            if (!center.cubic_use_full_fit)
                continue;
            std::vector<int> order(static_cast<std::size_t>(size()));
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
                const double dl = (dofs_[static_cast<std::size_t>(lhs)].point
                                  - center.point).squaredNorm();
                const double dr = (dofs_[static_cast<std::size_t>(rhs)].point
                                  - center.point).squaredNorm();
                if (dl != dr)
                    return dl < dr;
                return lhs < rhs;
            });
            if (static_cast<int>(order.size()) < corner_fit_neighbor_count_) {
                throw std::runtime_error(
                    "too few L-shape DOFs for corner Cauchy fit");
            }
            center.corner_fit_neighbor_ids.assign(
                order.begin(), order.begin() + corner_fit_neighbor_count_);

            const int rows = corner_fit_neighbor_count_
                           + corner_fit_derivative_count_;
            Eigen::MatrixXd design(rows, corner_fit_dim_);
            Eigen::VectorXd sqrt_weights(rows);
            for (int pos = 0; pos < corner_fit_neighbor_count_; ++pos) {
                const int neighbor = center.corner_fit_neighbor_ids[
                    static_cast<std::size_t>(pos)];
                const Eigen::Vector2d d =
                    (dofs_[static_cast<std::size_t>(neighbor)].point
                     - center.point) / h_;
                design.row(pos) = harmonic_basis(
                    d.dot(center.tangent), d.dot(center.normal),
                    corner_fit_degree_).transpose();
                sqrt_weights[pos] = std::sqrt(
                    1.0 / std::pow(0.35 + d.norm(), 2));
            }
            for (int pos = 0; pos < corner_fit_derivative_count_; ++pos) {
                const int neighbor = center.corner_fit_neighbor_ids[
                    static_cast<std::size_t>(pos)];
                const InterfaceDof& sample =
                    dofs_[static_cast<std::size_t>(neighbor)];
                const Eigen::Vector2d d = (sample.point - center.point) / h_;
                const Eigen::MatrixXd gradient = harmonic_gradient(
                    d.dot(center.tangent), d.dot(center.normal),
                    corner_fit_degree_);
                const Eigen::Vector2d components(
                    sample.normal.dot(center.tangent),
                    sample.normal.dot(center.normal));
                design.row(corner_fit_neighbor_count_ + pos) =
                    components.transpose() * gradient;
                sqrt_weights[corner_fit_neighbor_count_ + pos] =
                    std::sqrt(0.85 / std::pow(0.35 + d.norm(), 2));
            }

            Eigen::MatrixXd weighted_design = design;
            for (int row = 0; row < rows; ++row)
                weighted_design.row(row) *= sqrt_weights[row];
            const Eigen::MatrixXd pinv = pseudo_inverse(
                weighted_design, 2.0e-12, &center.corner_fit_condition);
            center.corner_fit_value_map = Eigen::MatrixXd::Zero(
                corner_fit_dim_, corner_fit_neighbor_count_);
            center.corner_fit_normal_map = Eigen::MatrixXd::Zero(
                corner_fit_dim_, corner_fit_neighbor_count_);
            for (int pos = 0; pos < corner_fit_neighbor_count_; ++pos) {
                center.corner_fit_value_map.col(pos) =
                    pinv.col(pos) * sqrt_weights[pos];
            }
            for (int pos = 0; pos < corner_fit_derivative_count_; ++pos) {
                center.corner_fit_normal_map.col(pos) =
                    pinv.col(corner_fit_neighbor_count_ + pos)
                    * sqrt_weights[corner_fit_neighbor_count_ + pos] * h_;
            }
        }
    }
    void build_quadratic_spread_maps()
    {
        constexpr int quadratic_degree = 2;
        constexpr int quadratic_dim = 2 * quadratic_degree + 1;
        for (int i = 0; i < size(); ++i) {
            InterfaceDof& center = dofs_[static_cast<std::size_t>(i)];
            std::vector<int> order(static_cast<std::size_t>(size()));
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
                const double dl = (dofs_[static_cast<std::size_t>(lhs)].point
                                  - center.point).squaredNorm();
                const double dr = (dofs_[static_cast<std::size_t>(rhs)].point
                                  - center.point).squaredNorm();
                if (dl != dr)
                    return dl < dr;
                return lhs < rhs;
            });
            if (geometry_ == GeometryKind::LShape) {
                order.erase(std::remove_if(
                    order.begin(), order.end(), [&](int neighbor) {
                        return dofs_[static_cast<std::size_t>(neighbor)]
                                   .boundary_segment
                            != center.boundary_segment;
                    }), order.end());
            }
            if (static_cast<int>(order.size())
                < quadratic_spread_neighbor_count_) {
                throw std::runtime_error(
                    "too few same-edge DOFs for quadratic L-shape stencil");
            }
            center.quadratic_neighbor_ids.assign(
                order.begin(),
                order.begin() + quadratic_spread_neighbor_count_);

            const int rows = quadratic_spread_neighbor_count_
                           + quadratic_spread_derivative_count_;
            const int derivative_neighbor_offset =
                quadratic_spread_derivative_count_
                    == quadratic_spread_neighbor_count_ ? 0 : 1;
            Eigen::MatrixXd design(rows, quadratic_dim);
            Eigen::VectorXd sqrt_weights(rows);
            for (int pos = 0; pos < quadratic_spread_neighbor_count_; ++pos) {
                const int neighbor = center.quadratic_neighbor_ids[
                    static_cast<std::size_t>(pos)];
                const Eigen::Vector2d d =
                    (dofs_[static_cast<std::size_t>(neighbor)].point
                     - center.point) / h_;
                design.row(pos) = harmonic_basis(
                    d.dot(center.tangent), d.dot(center.normal),
                    quadratic_degree).transpose();
                sqrt_weights[pos] = std::sqrt(
                    1.0 / std::pow(0.35 + d.norm(), 2));
            }
            for (int pos = 0; pos < quadratic_spread_derivative_count_; ++pos) {
                const int neighbor = center.quadratic_neighbor_ids[
                    static_cast<std::size_t>(
                        pos + derivative_neighbor_offset)];
                const InterfaceDof& sample =
                    dofs_[static_cast<std::size_t>(neighbor)];
                const Eigen::Vector2d d = (sample.point - center.point) / h_;
                const Eigen::MatrixXd gradient = harmonic_gradient(
                    d.dot(center.tangent), d.dot(center.normal),
                    quadratic_degree);
                const Eigen::Vector2d components(
                    sample.normal.dot(center.tangent),
                    sample.normal.dot(center.normal));
                design.row(quadratic_spread_neighbor_count_ + pos) =
                    components.transpose() * gradient;
                sqrt_weights[quadratic_spread_neighbor_count_ + pos] =
                    std::sqrt(0.85 / std::pow(0.35 + d.norm(), 2));
            }

            Eigen::MatrixXd weighted_design = design;
            for (int row = 0; row < rows; ++row)
                weighted_design.row(row) *= sqrt_weights[row];
            const Eigen::MatrixXd pinv = pseudo_inverse(
                weighted_design, 2.0e-12, &center.quadratic_condition);
            center.quadratic_value_map = Eigen::MatrixXd::Zero(
                quadratic_dim, quadratic_spread_neighbor_count_);
            center.quadratic_normal_map = Eigen::MatrixXd::Zero(
                quadratic_dim, quadratic_spread_neighbor_count_);
            for (int pos = 0; pos < quadratic_spread_neighbor_count_; ++pos) {
                center.quadratic_value_map.col(pos) =
                    pinv.col(pos) * sqrt_weights[pos];
            }
            for (int pos = 0; pos < quadratic_spread_derivative_count_; ++pos) {
                center.quadratic_normal_map.col(
                    pos + derivative_neighbor_offset) =
                    pinv.col(quadratic_spread_neighbor_count_ + pos)
                    * sqrt_weights[quadratic_spread_neighbor_count_ + pos] * h_;
            }
        }
    }

    void build_cubic_spread_maps()
    {
        constexpr int cubic_degree = 3;
        constexpr int cubic_dim = 2 * cubic_degree + 1;
        for (int i = 0; i < size(); ++i) {
            InterfaceDof& center = dofs_[static_cast<std::size_t>(i)];
            std::vector<int> order(static_cast<std::size_t>(size()));
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
                const double dl = (dofs_[static_cast<std::size_t>(lhs)].point
                                  - center.point).squaredNorm();
                const double dr = (dofs_[static_cast<std::size_t>(rhs)].point
                                  - center.point).squaredNorm();
                if (dl != dr)
                    return dl < dr;
                return lhs < rhs;
            });
            if (geometry_ == GeometryKind::LShape) {
                if (static_cast<int>(order.size())
                    < cubic_spread_neighbor_count_) {
                    throw std::runtime_error(
                        "too few L-shape DOFs for cubic stencil");
                }
                center.cubic_use_full_fit = std::any_of(
                    order.begin(),
                    order.begin() + cubic_spread_neighbor_count_,
                    [&](int neighbor) {
                        return dofs_[static_cast<std::size_t>(neighbor)]
                                   .boundary_segment
                            != center.boundary_segment;
                    });
                order.erase(std::remove_if(
                    order.begin(), order.end(), [&](int neighbor) {
                        return dofs_[static_cast<std::size_t>(neighbor)]
                                   .boundary_segment
                            != center.boundary_segment;
                    }), order.end());
            }
            if (static_cast<int>(order.size())
                < cubic_spread_neighbor_count_) {
                throw std::runtime_error(
                    "too few same-edge DOFs for cubic L-shape stencil");
            }
            center.cubic_neighbor_ids.assign(
                order.begin(), order.begin() + cubic_spread_neighbor_count_);

            const int rows = cubic_spread_neighbor_count_
                           + cubic_spread_derivative_count_;
            Eigen::MatrixXd design(rows, cubic_dim);
            Eigen::VectorXd sqrt_weights(rows);
            for (int pos = 0; pos < cubic_spread_neighbor_count_; ++pos) {
                const int neighbor = center.cubic_neighbor_ids[
                    static_cast<std::size_t>(pos)];
                const Eigen::Vector2d d =
                    (dofs_[static_cast<std::size_t>(neighbor)].point
                     - center.point) / h_;
                design.row(pos) = harmonic_basis(
                    d.dot(center.tangent), d.dot(center.normal),
                    cubic_degree).transpose();
                sqrt_weights[pos] = std::sqrt(
                    1.0 / std::pow(0.35 + d.norm(), 2));
            }
            for (int pos = 0; pos < cubic_spread_derivative_count_; ++pos) {
                const int neighbor = center.cubic_neighbor_ids[
                    static_cast<std::size_t>(pos)];
                const InterfaceDof& sample =
                    dofs_[static_cast<std::size_t>(neighbor)];
                const Eigen::Vector2d d = (sample.point - center.point) / h_;
                const Eigen::MatrixXd gradient = harmonic_gradient(
                    d.dot(center.tangent), d.dot(center.normal), cubic_degree);
                const Eigen::Vector2d components(
                    sample.normal.dot(center.tangent),
                    sample.normal.dot(center.normal));
                design.row(cubic_spread_neighbor_count_ + pos) =
                    components.transpose() * gradient;
                sqrt_weights[cubic_spread_neighbor_count_ + pos] =
                    std::sqrt(0.85 / std::pow(0.35 + d.norm(), 2));
            }

            Eigen::MatrixXd weighted_design = design;
            for (int row = 0; row < rows; ++row)
                weighted_design.row(row) *= sqrt_weights[row];
            const Eigen::MatrixXd pinv = pseudo_inverse(
                weighted_design, 2.0e-12, &center.cubic_condition);
            center.cubic_value_map = Eigen::MatrixXd::Zero(
                cubic_dim, cubic_spread_neighbor_count_);
            center.cubic_normal_map = Eigen::MatrixXd::Zero(
                cubic_dim, cubic_spread_neighbor_count_);
            for (int pos = 0; pos < cubic_spread_neighbor_count_; ++pos) {
                center.cubic_value_map.col(pos) =
                    pinv.col(pos) * sqrt_weights[pos];
            }
            for (int pos = 0; pos < cubic_spread_derivative_count_; ++pos) {
                center.cubic_normal_map.col(pos) =
                    pinv.col(cubic_spread_neighbor_count_ + pos)
                    * sqrt_weights[cubic_spread_neighbor_count_ + pos] * h_;
            }
        }
    }

    void build_crossing_rows()
    {
        for (GridEdgeCrossing& crossing : crossings_) {
            const InterfaceDof& dof = dofs_[static_cast<std::size_t>(
                crossing.nearest_dof)];
            const auto ca = grid_.coord(crossing.a[0], crossing.a[1]);
            const auto cb = grid_.coord(crossing.b[0], crossing.b[1]);
            const Eigen::Vector2d pa(ca[0], ca[1]);
            const Eigen::Vector2d pb(cb[0], cb[1]);
            const Eigen::Vector2d da = (pb - dof.point) / h_;
            const Eigen::Vector2d db = (pa - dof.point) / h_;
            crossing.eval_a = harmonic_basis(
                da.dot(dof.tangent), da.dot(dof.normal), degree_);
            crossing.eval_b = harmonic_basis(
                db.dot(dof.tangent), db.dot(dof.normal), degree_);
            if (!crossing.inside_a)
                crossing.eval_a *= -1.0;
            if (crossing.inside_a)
                crossing.eval_b *= -1.0;
            if (spread_mode_
                    == SpreadCorrectionMode::CubicHarmonicAtOppositeNode
                && dof.cubic_use_full_fit) {
                crossing.corner_fit_eval_a = harmonic_basis(
                    da.dot(dof.tangent), da.dot(dof.normal),
                    corner_fit_degree_);
                crossing.corner_fit_eval_b = harmonic_basis(
                    db.dot(dof.tangent), db.dot(dof.normal),
                    corner_fit_degree_);
                if (!crossing.inside_a)
                    crossing.corner_fit_eval_a *= -1.0;
                if (crossing.inside_a)
                    crossing.corner_fit_eval_b *= -1.0;
            }
            if (spread_mode_
                == SpreadCorrectionMode::QuadraticHarmonicAtOppositeNode) {
                crossing.quadratic_eval_a = harmonic_basis(
                    da.dot(dof.tangent), da.dot(dof.normal), 2);
                crossing.quadratic_eval_b = harmonic_basis(
                    db.dot(dof.tangent), db.dot(dof.normal), 2);
                if (!crossing.inside_a)
                    crossing.quadratic_eval_a *= -1.0;
                if (crossing.inside_a)
                    crossing.quadratic_eval_b *= -1.0;
            }
            if (spread_mode_
                == SpreadCorrectionMode::CubicHarmonicAtOppositeNode) {
                crossing.cubic_eval_a = harmonic_basis(
                    da.dot(dof.tangent), da.dot(dof.normal), 3);
                crossing.cubic_eval_b = harmonic_basis(
                    db.dot(dof.tangent), db.dot(dof.normal), 3);
                if (!crossing.inside_a)
                    crossing.cubic_eval_a *= -1.0;
                if (crossing.inside_a)
                    crossing.cubic_eval_b *= -1.0;
            }
        }
    }

    int trace_template_index(int crossing, int side, int layer) const
    {
        return (crossing * 2 + side) * restrict_layer_count() + layer;
    }

    const TraceTemplate& trace_template(int crossing,
                                        int side,
                                        int layer) const
    {
        return trace_templates_[static_cast<std::size_t>(
            trace_template_index(crossing, side, layer))];
    }

    void build_six_point_trace_templates()
    {
        constexpr int node_count = 6;
        constexpr int cauchy_degree = 2;
        constexpr int cauchy_dim = 5;
        trace_templates_.resize(static_cast<std::size_t>(size()));
        for (int i = 0; i < size(); ++i) {
            const InterfaceDof& dof = dofs_[static_cast<std::size_t>(i)];
            const Eigen::Vector2d rr =
                (dof.point
                 - Eigen::Vector2d(geometry_box_min(geometry_),
                                   geometry_box_min(geometry_))) / h_;
            const int center_i = static_cast<int>(std::floor(rr[0] + 0.5));
            const int center_j = static_cast<int>(std::floor(rr[1] + 0.5));
            const auto center_coord_array = grid_.coord(center_i, center_j);
            const Eigen::Vector2d center_coord(
                center_coord_array[0], center_coord_array[1]);
            const int dx = dof.point[0] > center_coord[0] ? 1 : -1;
            const int dy = dof.point[1] > center_coord[1] ? 1 : -1;
            const std::array<std::array<int, 2>, node_count> ij{{
                {{center_i, center_j}},
                {{center_i + dx, center_j}},
                {{center_i, center_j + dy}},
                {{center_i + dx, center_j + dy}},
                {{center_i - dx, center_j}},
                {{center_i, center_j - dy}}
            }};

            Eigen::Matrix<double, node_count, node_count> design;
            TraceTemplate& entry = trace_templates_[static_cast<std::size_t>(i)];
            entry.ids.resize(node_count);
            entry.weights.resize(node_count);
            entry.correction = Eigen::MatrixXd::Zero(node_count, cauchy_dim);
            for (int v = 0; v < node_count; ++v) {
                const int I = ij[static_cast<std::size_t>(v)][0];
                const int J = ij[static_cast<std::size_t>(v)][1];
                if (I < 0 || I > n_ || J < 0 || J > n_)
                    throw std::runtime_error(
                        "six-point restrict stencil exits embedding box");
                const auto coord_array = grid_.coord(I, J);
                const Eigen::Vector2d coord(coord_array[0], coord_array[1]);
                const Eigen::Vector2d local = (coord - center_coord) / h_;
                design.row(v) =
                    total_quadratic_basis(local[0], local[1]).transpose();
                const int flat = grid_.index(I, J);
                entry.ids[static_cast<std::size_t>(v)] = flat;
                if (inside_node(flat)) {
                    const Eigen::Vector2d d = (coord - dof.point) / h_;
                    entry.correction.row(v) = -harmonic_basis(
                        d.dot(dof.tangent), d.dot(dof.normal),
                        cauchy_degree).transpose();
                }
            }
            const Eigen::Vector2d target = (dof.point - center_coord) / h_;
            const Eigen::MatrixXd pinv = pseudo_inverse(design, 1.0e-13);
            const Eigen::Matrix<double, node_count, 1> weights =
                pinv.transpose()
                * total_quadratic_basis(target[0], target[1]);
            for (int v = 0; v < node_count; ++v)
                entry.weights[static_cast<std::size_t>(v)] = weights[v];
        }
    }

    void build_trace_templates()
    {
        if (restrict_mode_ == RestrictMode::SixPointQuadraticExterior) {
            build_six_point_trace_templates();
            return;
        }
        const int layer_count = restrict_layer_count();
        const int grid_side_count = restrict_grid_side_count();
        const int grid_node_count = grid_side_count * grid_side_count;
        trace_templates_.resize(
            static_cast<std::size_t>(size() * 2 * layer_count));
        for (int i = 0; i < size(); ++i) {
            const InterfaceDof& dof = dofs_[static_cast<std::size_t>(i)];
            const int cauchy_degree = restrict_cauchy_degree(dof);
            const int cauchy_dim = 2 * cauchy_degree + 1;
            for (int side = 0; side < 2; ++side) {
                const double sign = side == 0 ? -1.0 : 1.0;
                const bool desired_inside = side == 0;
                for (int layer = 0; layer < layer_count; ++layer) {
                    TraceTemplate& entry = trace_templates_[static_cast<std::size_t>(
                        trace_template_index(i, side, layer))];
                    entry.ids.assign(static_cast<std::size_t>(grid_node_count), 0);
                    entry.weights.assign(
                        static_cast<std::size_t>(grid_node_count), 0.0);
                    entry.correction = Eigen::MatrixXd::Zero(
                        grid_node_count, cauchy_dim);
                    const Eigen::Vector2d query = dof.point
                        + sign * kNormalLayers[static_cast<std::size_t>(layer)]
                            * h_ * dof.normal;
                    const Eigen::Vector2d rr =
                        (query
                         - Eigen::Vector2d(geometry_box_min(geometry_),
                                           geometry_box_min(geometry_))) / h_;
                    std::vector<int> ix(static_cast<std::size_t>(grid_side_count));
                    std::vector<int> iy(static_cast<std::size_t>(grid_side_count));
                    std::vector<double> wx(
                        static_cast<std::size_t>(grid_side_count));
                    std::vector<double> wy(
                        static_cast<std::size_t>(grid_side_count));
                    if (grid_side_count == 4) {
                        const int lo_x = static_cast<int>(std::floor(rr[0]));
                        const int lo_y = static_cast<int>(std::floor(rr[1]));
                        const std::array<double, 4> wx_array =
                            lagrange4_weights(rr[0] - lo_x);
                        const std::array<double, 4> wy_array =
                            lagrange4_weights(rr[1] - lo_y);
                        for (int a = 0; a < 4; ++a) {
                            ix[static_cast<std::size_t>(a)] = lo_x - 1 + a;
                            iy[static_cast<std::size_t>(a)] = lo_y - 1 + a;
                            wx[static_cast<std::size_t>(a)] =
                                wx_array[static_cast<std::size_t>(a)];
                            wy[static_cast<std::size_t>(a)] =
                                wy_array[static_cast<std::size_t>(a)];
                        }
                    } else {
                        const int center_x = static_cast<int>(
                            std::floor(rr[0] + 0.5));
                        const int center_y = static_cast<int>(
                            std::floor(rr[1] + 0.5));
                        const std::array<double, 3> wx_array =
                            lagrange3_weights(rr[0] - center_x);
                        const std::array<double, 3> wy_array =
                            lagrange3_weights(rr[1] - center_y);
                        for (int a = 0; a < 3; ++a) {
                            ix[static_cast<std::size_t>(a)] = center_x - 1 + a;
                            iy[static_cast<std::size_t>(a)] = center_y - 1 + a;
                            wx[static_cast<std::size_t>(a)] =
                                wx_array[static_cast<std::size_t>(a)];
                            wy[static_cast<std::size_t>(a)] =
                                wy_array[static_cast<std::size_t>(a)];
                        }
                    }
                    int v = 0;
                    for (int ai = 0; ai < grid_side_count; ++ai) {
                        const int I = ix[static_cast<std::size_t>(ai)];
                        for (int aj = 0; aj < grid_side_count; ++aj) {
                            const int J = iy[static_cast<std::size_t>(aj)];
                            if (I < 0 || I > n_ || J < 0 || J > n_)
                                throw std::runtime_error(
                                    "restrict interpolation stencil exits embedding box");
                            const int flat = grid_.index(I, J);
                            entry.ids[static_cast<std::size_t>(v)] = flat;
                            entry.weights[static_cast<std::size_t>(v)] =
                                wx[static_cast<std::size_t>(ai)]
                              * wy[static_cast<std::size_t>(aj)];
                            if (inside_node(flat) != desired_inside) {
                                const auto coord = grid_.coord(I, J);
                                const Eigen::Vector2d d =
                                    (Eigen::Vector2d(coord[0], coord[1])
                                     - dof.point) / h_;
                                entry.correction.row(v) = harmonic_basis(
                                    d.dot(dof.tangent),
                                    d.dot(dof.normal),
                                    cauchy_degree).transpose();
                                if (!desired_inside)
                                    entry.correction.row(v) *= -1.0;
                            }
                            ++v;
                        }
                    }
                }
            }
        }
    }

    void build_joint_value_fit()
    {
        const int layer_count = restrict_layer_count();
        const int degree = restrict_normal_degree();
        Eigen::MatrixXd design = Eigen::MatrixXd::Zero(
            2 * layer_count, 2 * degree);
        for (int layer = 0; layer < layer_count; ++layer) {
            const double xm = -kNormalLayers[static_cast<std::size_t>(layer)];
            const double xp = kNormalLayers[static_cast<std::size_t>(layer)];
            design(layer, 0) = 1.0;
            design(layer, 1) = xm;
            design(layer_count + layer, 0) = 1.0;
            design(layer_count + layer, 1) = xp;
            double xm_power = xm;
            double xp_power = xp;
            for (int power = 2; power <= degree; ++power) {
                xm_power *= xm;
                xp_power *= xp;
                design(layer, power) = xm_power;
                design(layer_count + layer, degree + power - 1) = xp_power;
            }
        }
        const Eigen::MatrixXd pinv = pseudo_inverse(design, 1.0e-13);
        c0_weights_ = pinv.row(0).transpose();
        c1_weights_ = pinv.row(1).transpose();
        trace_projection_ = design * pinv;
    }

    int n_;
    double h_;
    GeometryKind geometry_;
    InterfaceDofMode dof_mode_;
    RestrictMode restrict_mode_;
    double uniform_dof_ratio_;
    int degree_;
    int dim_;
    int requested_neighbors_;
    int requested_derivative_neighbors_;
    SpreadCorrectionMode spread_mode_;
    int quadratic_spread_neighbor_count_;
    int quadratic_spread_derivative_count_;
    int cubic_spread_neighbor_count_;
    int cubic_spread_derivative_count_;
    int corner_fit_degree_;
    int corner_fit_dim_;
    int corner_fit_neighbor_count_;
    int corner_fit_derivative_count_;
    int neighbor_count_ = 0;
    int derivative_count_ = 0;
    CartesianGrid2D grid_;
    LaplaceFftBulkSolverZfft2D bulk_solver_;
    std::vector<char> inside_mask_;
    std::vector<GridEdgeCrossing> crossings_;
    std::vector<InterfaceDof> dofs_;
    std::vector<TraceTemplate> trace_templates_;
    double boundary_length_ = 0.0;
    Eigen::VectorXd c0_weights_;
    Eigen::VectorXd c1_weights_;
    Eigen::MatrixXd trace_projection_;
    mutable double last_trace_fit_rel_l2_ =
        std::numeric_limits<double>::quiet_NaN();
};

class AugmentedExteriorTraceOperator final : public IKFBIOperator {
public:
    explicit AugmentedExteriorTraceOperator(
        const PythonCompatibleHarmonicJet2D& solver)
        : solver_(solver)
        , weights_(solver.boundary_weights())
        , zero_(Eigen::VectorXd::Zero(solver.size()))
    {}

    int problem_size() const override { return solver_.size() + 1; }

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override
    {
        if (x.size() != problem_size())
            throw std::invalid_argument("augmented trace input size mismatch");
        const Eigen::VectorXd trace = x.head(solver_.size());
        const Eigen::VectorXd potential = solver_.potential(trace, zero_);
        y.resize(problem_size());
        y.head(solver_.size()) = solver_.exterior_trace(
            potential, trace, zero_);
        y.head(solver_.size()).array() += x[solver_.size()];
        y[solver_.size()] = weights_.dot(trace) / solver_.boundary_length();
    }

private:
    const PythonCompatibleHarmonicJet2D& solver_;
    Eigen::VectorXd weights_;
    Eigen::VectorXd zero_;
};

class InteriorDirichletNormalJumpOperator final : public IKFBIOperator {
public:
    explicit InteriorDirichletNormalJumpOperator(
        const PythonCompatibleHarmonicJet2D& solver)
        : solver_(solver)
        , zero_(Eigen::VectorXd::Zero(solver.size()))
    {}

    int problem_size() const override { return solver_.size(); }

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override
    {
        if (x.size() != problem_size())
            throw std::invalid_argument("Dirichlet trace input size mismatch");
        const Eigen::VectorXd potential = solver_.potential(zero_, x);
        y = solver_.interior_trace(potential, zero_, x);
    }

private:
    const PythonCompatibleHarmonicJet2D& solver_;
    Eigen::VectorXd zero_;
};

class ExteriorDirichletNormalJumpSecondKindOperator final
    : public IKFBIOperator {
public:
    explicit ExteriorDirichletNormalJumpSecondKindOperator(
        const PythonCompatibleHarmonicJet2D& solver)
        : solver_(solver)
        , zero_(Eigen::VectorXd::Zero(solver.size()))
    {}

    int problem_size() const override { return solver_.size(); }

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override
    {
        if (x.size() != problem_size()) {
            throw std::invalid_argument(
                "Dirichlet normal-jump second-kind input size mismatch");
        }
        const Eigen::VectorXd potential = solver_.potential(zero_, x);
        y = solver_.exterior_normal_trace(potential, zero_, x);
    }

private:
    const PythonCompatibleHarmonicJet2D& solver_;
    Eigen::VectorXd zero_;
};

class InteriorDirichletValueJumpOperator final : public IKFBIOperator {
public:
    explicit InteriorDirichletValueJumpOperator(
        const PythonCompatibleHarmonicJet2D& solver)
        : solver_(solver)
        , zero_(Eigen::VectorXd::Zero(solver.size()))
    {}

    int problem_size() const override { return solver_.size(); }

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override
    {
        if (x.size() != problem_size()) {
            throw std::invalid_argument(
                "Dirichlet value-jump input size mismatch");
        }
        const Eigen::VectorXd potential = solver_.potential(x, zero_);
        y = solver_.interior_trace(potential, x, zero_);
    }

private:
    const PythonCompatibleHarmonicJet2D& solver_;
    Eigen::VectorXd zero_;
};

struct StudyResult {
    std::string bvp;
    std::string dirichlet_formulation;
    std::string geometry;
    std::string exact_solution;
    std::string dof_mode;
    std::string restrict_mode;
    std::string restrict_cauchy_source;
    std::string spread_mode;
    int n = 0;
    double h = 0.0;
    int dofs = 0;
    int crossings = 0;
    double dof_nearest_spacing_min = 0.0;
    double dof_nearest_spacing_max = 0.0;
    double dof_nearest_spacing_mean = 0.0;
    int iterations = 0;
    bool converged = false;
    double setup_seconds = 0.0;
    double seconds = 0.0;
    double total_seconds = 0.0;
    double phase_min = 0.0;
    double phase_max = 0.0;
    double phase_std = 0.0;
    double condition_max = 0.0;
    double condition_mean = 0.0;
    double spread_condition_max = 0.0;
    double spread_condition_mean = 0.0;
    double augmented_residual_linf = 0.0;
    double raw_exterior_trace_linf = 0.0;
    double raw_exterior_trace_l2 = 0.0;
    double exterior_normal_linf = std::numeric_limits<double>::quiet_NaN();
    double exterior_normal_l2 = std::numeric_limits<double>::quiet_NaN();
    double normal_route_mismatch_linf =
        std::numeric_limits<double>::quiet_NaN();
    double boundary_trace_res_linf = 0.0;
    double boundary_trace_res_l2 = 0.0;
    double density_mean = 0.0;
    double density_linf = 0.0;
    double density_l2 = 0.0;
    double flux_mean = 0.0;
    double lagrange = 0.0;
    double trace_mean = 0.0;
    double global_linf = 0.0;
    double global_l2 = 0.0;
    double trace_linf = 0.0;
    double trace_l2 = 0.0;
    double exterior_linf_far = 0.0;
    double exterior_l2_far = 0.0;
    double constant_shift = 0.0;
    double trace_fit_rel_l2 = 0.0;
    int n_global = 0;
    int n_exterior_far = 0;
    double global_linf_order = std::numeric_limits<double>::quiet_NaN();
    double global_l2_order = std::numeric_limits<double>::quiet_NaN();
};

double inf_norm(const Eigen::VectorXd& values)
{
    return values.size() == 0 ? 0.0 : values.cwiseAbs().maxCoeff();
}

StudyResult run_one(BvpKind bvp,
                    DirichletFormulation formulation,
                    GeometryKind geometry,
                    int n,
                    InterfaceDofMode dof_mode,
                    RestrictMode restrict_mode,
                    double uniform_dof_ratio,
                    int degree,
                    int neighbors,
                    int derivative_neighbors,
                    SpreadCorrectionMode spread_mode,
                    int quadratic_spread_neighbors,
                    int quadratic_spread_derivative_neighbors,
                    int cubic_spread_neighbors,
                    int cubic_spread_derivative_neighbors,
                    int corner_fit_degree,
                    int corner_fit_neighbors,
                    int corner_fit_derivative_neighbors,
                    int max_iter,
                    double tolerance,
                    int restart)
{
    const auto setup_start = std::chrono::steady_clock::now();
    PythonCompatibleHarmonicJet2D solver(
        n, geometry, dof_mode, restrict_mode, uniform_dof_ratio,
        degree, neighbors,
        derivative_neighbors, spread_mode,
        quadratic_spread_neighbors,
        quadratic_spread_derivative_neighbors,
        cubic_spread_neighbors,
        cubic_spread_derivative_neighbors,
        corner_fit_degree,
        corner_fit_neighbors,
        corner_fit_derivative_neighbors);
    const double setup_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - setup_start).count();
    std::cout << "  setup crossings=" << solver.num_crossings()
              << " dofs=" << solver.size()
              << " hybrid_full_fit_dofs=" << solver.hybrid_full_fit_dofs()
              << " time=" << setup_seconds << "s\n";

    const int ns = solver.size();
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(ns);
    const Eigen::VectorXd normal_data = solver.neumann_data();
    const Eigen::VectorXd dirichlet_data = solver.exact_trace();
    const Eigen::VectorXd weights = solver.boundary_weights();

    const auto solve_start = std::chrono::steady_clock::now();
    Eigen::VectorXd density;
    Eigen::VectorXd potential;
    Eigen::VectorXd boundary_residual;
    Eigen::VectorXd raw_exterior;
    Eigen::VectorXd exterior_normal;
    double normal_route_mismatch_linf =
        std::numeric_limits<double>::quiet_NaN();
    double operator_residual_linf = 0.0;
    double lagrange = 0.0;
    int iterations = 0;
    bool converged = false;
    if (bvp == BvpKind::Neumann) {
        const Eigen::VectorXd fixed_potential =
            solver.potential(zero, normal_data);
        const Eigen::VectorXd fixed_exterior = solver.exterior_trace(
            fixed_potential, zero, normal_data);
        AugmentedExteriorTraceOperator op(solver);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(ns + 1);
        rhs.head(ns) = -fixed_exterior;
        Eigen::VectorXd augmented = Eigen::VectorXd::Zero(ns + 1);
        GMRES gmres(max_iter, tolerance, std::min(restart, ns + 1));
        iterations = gmres.solve(op, rhs, augmented);
        converged = gmres.converged();
        density = augmented.head(ns);
        lagrange = augmented[ns];
        potential = solver.potential(density, normal_data);
        boundary_residual = solver.exterior_trace(
            potential, density, normal_data, true);
        raw_exterior = boundary_residual;
        Eigen::VectorXd applied;
        op.apply(augmented, applied);
        operator_residual_linf = inf_norm((applied - rhs).head(ns));
    } else if (formulation
               == DirichletFormulation::NormalJumpFirstKind) {
        InteriorDirichletNormalJumpOperator op(solver);
        const Eigen::VectorXd fixed_potential =
            solver.potential(dirichlet_data, zero);
        const Eigen::VectorXd fixed_interior = solver.interior_trace(
            fixed_potential, dirichlet_data, zero);
        const Eigen::VectorXd rhs = dirichlet_data - fixed_interior;
        density = Eigen::VectorXd::Zero(ns);
        GMRES gmres(max_iter, tolerance, std::min(restart, ns));
        iterations = gmres.solve(op, rhs, density);
        converged = gmres.converged();
        potential = solver.potential(dirichlet_data, density);
        boundary_residual = solver.interior_trace(
            potential, dirichlet_data, density, true) - dirichlet_data;
        raw_exterior = solver.exterior_trace(
            potential, dirichlet_data, density);
        Eigen::VectorXd applied;
        op.apply(density, applied);
        operator_residual_linf = inf_norm(applied - rhs);
    } else if (formulation
               == DirichletFormulation::NormalJumpSecondKind) {
        ExteriorDirichletNormalJumpSecondKindOperator op(solver);
        const Eigen::VectorXd fixed_potential =
            solver.potential(dirichlet_data, zero);
        const Eigen::VectorXd rhs = -solver.exterior_normal_trace(
            fixed_potential, dirichlet_data, zero);
        density = Eigen::VectorXd::Zero(ns);
        GMRES gmres(max_iter, tolerance, std::min(restart, ns));
        iterations = gmres.solve(op, rhs, density);
        converged = gmres.converged();
        potential = solver.potential(dirichlet_data, density);
        boundary_residual = solver.interior_trace(
            potential, dirichlet_data, density) - dirichlet_data;
        raw_exterior = solver.exterior_trace(
            potential, dirichlet_data, density);
        exterior_normal = solver.exterior_normal_trace(
            potential, dirichlet_data, density, true);
        const Eigen::VectorXd exterior_normal_from_jump =
            solver.interior_normal_trace(
                potential, dirichlet_data, density) - density;
        normal_route_mismatch_linf = inf_norm(
            exterior_normal - exterior_normal_from_jump);
        Eigen::VectorXd applied;
        op.apply(density, applied);
        operator_residual_linf = inf_norm(applied - rhs);
    } else {
        InteriorDirichletValueJumpOperator op(solver);
        const Eigen::VectorXd rhs = dirichlet_data;
        density = Eigen::VectorXd::Zero(ns);
        GMRES gmres(max_iter, tolerance, std::min(restart, ns));
        iterations = gmres.solve(op, rhs, density);
        converged = gmres.converged();
        potential = solver.potential(density, zero);
        boundary_residual = solver.interior_trace(
            potential, density, zero, true) - dirichlet_data;
        raw_exterior = solver.exterior_trace(
            potential, density, zero);
        Eigen::VectorXd applied;
        op.apply(density, applied);
        operator_residual_linf = inf_norm(applied - rhs);
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();

    StudyResult result;
    result.bvp = bvp_name(bvp);
    result.dirichlet_formulation = bvp == BvpKind::Dirichlet
        ? dirichlet_formulation_name(formulation) : "not_applicable";
    result.geometry = geometry_name(geometry);
    result.exact_solution = exact_solution_name(geometry);
    result.dof_mode = dof_mode_name(dof_mode);
    result.restrict_mode = restrict_mode_name(restrict_mode);
    result.restrict_cauchy_source = solver.restrict_cauchy_source();
    result.spread_mode = spread_mode_name(spread_mode);
    result.n = n;
    result.h = solver.h();
    result.dofs = ns;
    result.crossings = solver.num_crossings();
    result.dof_nearest_spacing_min = std::numeric_limits<double>::infinity();
    for (int i = 0; i < ns; ++i) {
        double nearest = std::numeric_limits<double>::infinity();
        for (int j = 0; j < ns; ++j) {
            if (i == j)
                continue;
            nearest = std::min(
                nearest,
                (solver.dofs()[static_cast<std::size_t>(i)].point
                 - solver.dofs()[static_cast<std::size_t>(j)].point).norm());
        }
        result.dof_nearest_spacing_min = std::min(
            result.dof_nearest_spacing_min, nearest);
        result.dof_nearest_spacing_max = std::max(
            result.dof_nearest_spacing_max, nearest);
        result.dof_nearest_spacing_mean += nearest;
    }
    result.dof_nearest_spacing_mean /= static_cast<double>(ns);
    result.iterations = iterations;
    result.converged = converged;
    result.setup_seconds = setup_seconds;
    result.seconds = seconds;
    result.total_seconds = setup_seconds + seconds;
    result.lagrange = lagrange;
    result.augmented_residual_linf = operator_residual_linf;
    result.raw_exterior_trace_linf = inf_norm(raw_exterior);
    result.raw_exterior_trace_l2 = std::sqrt(
        raw_exterior.squaredNorm() / static_cast<double>(ns));
    if (exterior_normal.size() != 0) {
        result.exterior_normal_linf = inf_norm(exterior_normal);
        result.exterior_normal_l2 = std::sqrt(
            exterior_normal.squaredNorm() / static_cast<double>(ns));
        result.normal_route_mismatch_linf = normal_route_mismatch_linf;
    }
    result.boundary_trace_res_linf = inf_norm(boundary_residual);
    result.boundary_trace_res_l2 = std::sqrt(
        boundary_residual.squaredNorm() / static_cast<double>(ns));
    result.density_mean = weights.dot(density) / solver.boundary_length();
    if (bvp == BvpKind::Neumann) {
        result.flux_mean = weights.dot(normal_data)
                         / solver.boundary_length();
    } else if (formulation
               != DirichletFormulation::ValueJumpSecondKind) {
        result.flux_mean = weights.dot(density)
                         / solver.boundary_length();
    } else {
        result.flux_mean = 0.0;
    }
    result.trace_mean = weights.dot(
        bvp == BvpKind::Neumann ? density : dirichlet_data)
        / solver.boundary_length();
    result.trace_fit_rel_l2 = solver.last_trace_fit_rel_l2();

    result.phase_min = std::numeric_limits<double>::infinity();
    result.phase_max = -std::numeric_limits<double>::infinity();
    result.condition_max = 0.0;
    double phase_sum = 0.0;
    double condition_sum = 0.0;
    double spread_condition_sum = 0.0;
    for (const GridEdgeCrossing& crossing : solver.crossings()) {
        result.phase_min = std::min(result.phase_min, crossing.phase);
        result.phase_max = std::max(result.phase_max, crossing.phase);
        phase_sum += crossing.phase;
    }
    for (const InterfaceDof& dof : solver.dofs()) {
        result.condition_max = std::max(
            result.condition_max, solver.restrict_condition(dof));
        condition_sum += solver.restrict_condition(dof);
        result.spread_condition_max = std::max(
            result.spread_condition_max, solver.spread_condition(dof));
        spread_condition_sum += solver.spread_condition(dof);
    }
    const double phase_mean = phase_sum
        / static_cast<double>(solver.num_crossings());
    double phase_var = 0.0;
    for (const GridEdgeCrossing& crossing : solver.crossings())
        phase_var += (crossing.phase - phase_mean)
                   * (crossing.phase - phase_mean);
    result.phase_std = std::sqrt(
        phase_var / static_cast<double>(solver.num_crossings()));
    result.condition_mean = condition_sum / static_cast<double>(ns);
    result.spread_condition_mean =
        spread_condition_sum / static_cast<double>(ns);

    const auto dims = solver.grid().dof_dims();
    double shift_sum = 0.0;
    for (int j = 1; j + 1 < dims[1]; ++j) {
        for (int i = 1; i + 1 < dims[0]; ++i) {
            const int node = solver.grid().index(i, j);
            if (!solver.inside_node(node))
                continue;
            if (bvp == BvpKind::Neumann) {
                const auto point = solver.grid().coord(i, j);
                shift_sum += exact_solution(
                    geometry, point[0], point[1]) - potential[node];
            }
            ++result.n_global;
        }
    }
    result.constant_shift = bvp == BvpKind::Neumann
        ? shift_sum / static_cast<double>(result.n_global) : 0.0;
    double error_sq = 0.0;
    double outside_sq = 0.0;
    for (int j = 1; j + 1 < dims[1]; ++j) {
        for (int i = 1; i + 1 < dims[0]; ++i) {
            const int node = solver.grid().index(i, j);
            const auto point_array = solver.grid().coord(i, j);
            const Eigen::Vector2d point(point_array[0], point_array[1]);
            if (solver.inside_node(node)) {
                const double error = potential[node] + result.constant_shift
                                   - exact_solution(
                                       geometry, point[0], point[1]);
                result.global_linf = std::max(result.global_linf,
                                              std::abs(error));
                error_sq += error * error;
            } else if (solver.nearest_crossing_distance(point)
                       >= 3.0 * solver.h()) {
                result.exterior_linf_far = std::max(
                    result.exterior_linf_far, std::abs(potential[node]));
                outside_sq += potential[node] * potential[node];
                ++result.n_exterior_far;
            }
        }
    }
    result.global_l2 = std::sqrt(error_sq / static_cast<double>(result.n_global));
    result.exterior_l2_far = result.n_exterior_far > 0
        ? std::sqrt(outside_sq / static_cast<double>(result.n_exterior_far))
        : std::numeric_limits<double>::quiet_NaN();

    Eigen::VectorXd density_error;
    if (bvp == BvpKind::Neumann) {
        Eigen::VectorXd exact_trace = solver.exact_trace();
        exact_trace.array() -= weights.dot(exact_trace)
                             / solver.boundary_length();
        density_error = density - exact_trace;
        result.trace_linf = inf_norm(density_error);
        result.trace_l2 = std::sqrt(
            density_error.squaredNorm() / static_cast<double>(ns));
    } else if (formulation
               != DirichletFormulation::ValueJumpSecondKind) {
        density_error = density - normal_data;
        result.trace_linf = result.boundary_trace_res_linf;
        result.trace_l2 = result.boundary_trace_res_l2;
    } else {
        result.trace_linf = result.boundary_trace_res_linf;
        result.trace_l2 = result.boundary_trace_res_l2;
        result.density_linf = std::numeric_limits<double>::quiet_NaN();
        result.density_l2 = std::numeric_limits<double>::quiet_NaN();
        return result;
    }
    result.density_linf = inf_norm(density_error);
    result.density_l2 = std::sqrt(
        density_error.squaredNorm() / static_cast<double>(ns));
    return result;
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
    for (std::size_t i = 0; i < results.size(); ++i) {
        StudyResult& result = results[i];
        const StudyResult* previous = nullptr;
        for (std::size_t j = 0; j < i; ++j) {
            const StudyResult& candidate = results[j];
            if (candidate.geometry != result.geometry
                || candidate.dof_mode != result.dof_mode
                || candidate.restrict_mode != result.restrict_mode
                || candidate.dirichlet_formulation
                    != result.dirichlet_formulation
                || candidate.n >= result.n) {
                continue;
            }
            if (previous == nullptr || candidate.n > previous->n)
                previous = &candidate;
        }
        if (previous != nullptr) {
            result.global_linf_order = observed_order(
                previous->global_linf, result.global_linf,
                previous->h, result.h);
            result.global_l2_order = observed_order(
                previous->global_l2, result.global_l2,
                previous->h, result.h);
        }
    }
}

void print_result(const StudyResult& result)
{
    std::cout << "  dof_mode=" << result.dof_mode
              << " dirichlet_formulation="
              << result.dirichlet_formulation
              << " restrict_mode=" << result.restrict_mode
              << " restrict_cauchy_source="
              << result.restrict_cauchy_source
              << " exact_solution=" << result.exact_solution
              << " crossings=" << result.crossings
              << " dofs=" << result.dofs
              << " spacing[min/mean/max]="
              << result.dof_nearest_spacing_min << '/'
              << result.dof_nearest_spacing_mean << '/'
              << result.dof_nearest_spacing_max
              << " gmres=" << result.iterations
              << " converged=" << (result.converged ? "yes" : "no")
              << " aug_res=" << result.augmented_residual_linf
              << " boundary_res=" << result.boundary_trace_res_linf
              << " exterior_normal=" << result.exterior_normal_linf
              << " normal_route_mismatch="
              << result.normal_route_mismatch_linf
              << " Linf=" << result.global_linf
              << " L2=" << result.global_l2
              << " trace=" << result.trace_linf
              << " density_err=" << result.density_linf
              << " cond_max=" << result.condition_max
              << " spread_cond_max=" << result.spread_condition_max
              << " setup_time=" << result.setup_seconds << "s"
              << " solve_time=" << result.seconds << "s"
              << " total_time=" << result.total_seconds << "s\n";
}

void print_summary(const std::vector<StudyResult>& results)
{
    for (const std::string formulation : {
             std::string("not_applicable"),
             std::string("normal_jump_first_kind"),
             std::string("value_jump_second_kind"),
             std::string("normal_jump_second_kind")}) {
        for (const std::string name : {std::string("ellipse"),
                                       std::string("flower"),
                                       std::string("circle"),
                                       std::string("lshape")}) {
            for (const std::string mode : {std::string("crossing"),
                                           std::string("uniform_midpoint")}) {
                for (const std::string restrict_mode : {
                         std::string("bicubic_cubic"),
                         std::string("bicubic_quadratic"),
                         std::string("biquadratic_cubic"),
                         std::string("biquadratic_quadratic"),
                         std::string("biquadratic_quadratic_two_layer"),
                         std::string("six_point_quadratic_exterior")}) {
                    const bool present = std::any_of(
                        results.begin(), results.end(),
                        [&](const StudyResult& result) {
                            return result.geometry == name
                                && result.dof_mode == mode
                                && result.restrict_mode == restrict_mode
                                && result.dirichlet_formulation == formulation;
                        });
                    if (!present)
                        continue;
                    std::cout << "\nC++ harmonic-jet summary: " << name
                              << " / " << formulation
                              << " / " << mode
                              << " / " << restrict_mode << '\n';
                    std::cout << "  " << std::setw(5) << "N"
                              << std::setw(7) << "dofs"
                              << std::setw(14) << "Linf"
                              << std::setw(11) << "p_inf"
                              << std::setw(14) << "L2"
                              << std::setw(11) << "p_L2"
                              << std::setw(9) << "GMRES"
                              << std::setw(14) << "total(s)" << '\n';
                    for (const StudyResult& result : results) {
                        if (result.geometry != name
                            || result.dof_mode != mode
                            || result.restrict_mode != restrict_mode
                            || result.dirichlet_formulation != formulation) {
                            continue;
                        }
                        std::cout << "  " << std::setw(5) << result.n
                                  << std::setw(7) << result.dofs
                                  << std::setw(14) << result.global_linf;
                        if (std::isfinite(result.global_linf_order)) {
                            std::cout << std::setw(11)
                                      << result.global_linf_order;
                        } else {
                            std::cout << std::setw(11) << "-";
                        }
                        std::cout << std::setw(14) << result.global_l2;
                        if (std::isfinite(result.global_l2_order)) {
                            std::cout << std::setw(11)
                                      << result.global_l2_order;
                        } else {
                            std::cout << std::setw(11) << "-";
                        }
                        std::cout << std::setw(9) << result.iterations
                                  << std::setw(14)
                                  << result.total_seconds << '\n';
                    }
                }
            }
        }
    }
}

void print_dof_comparison(const std::vector<StudyResult>& results)
{
    bool printed_header = false;
    for (const StudyResult& baseline : results) {
        if (baseline.dof_mode != "crossing")
            continue;
        const auto match = std::find_if(
            results.begin(), results.end(), [&](const StudyResult& candidate) {
                return candidate.geometry == baseline.geometry
                    && candidate.n == baseline.n
                    && candidate.restrict_mode == baseline.restrict_mode
                    && candidate.dirichlet_formulation
                        == baseline.dirichlet_formulation
                    && candidate.dof_mode == "uniform_midpoint";
            });
        if (match == results.end())
            continue;
        if (!printed_header) {
            std::cout << "\nUniform-midpoint / crossing-DOF error comparison\n"
                      << "  " << std::setw(9) << "geometry"
                      << std::setw(28) << "formulation"
                      << std::setw(6) << "N"
                      << std::setw(27) << "restrict"
                      << std::setw(13) << "Linf ratio"
                      << std::setw(13) << "L2 ratio"
                      << std::setw(13) << "trace ratio" << '\n';
            printed_header = true;
        }
        std::cout << "  " << std::setw(9) << baseline.geometry
                  << std::setw(28) << baseline.dirichlet_formulation
                  << std::setw(6) << baseline.n
                  << std::setw(27) << baseline.restrict_mode
                  << std::setw(13) << match->global_linf / baseline.global_linf
                  << std::setw(13) << match->global_l2 / baseline.global_l2
                  << std::setw(13) << match->trace_linf / baseline.trace_linf
                  << '\n';
    }
}

void print_restrict_comparison(const std::vector<StudyResult>& results)
{
    bool printed_header = false;
    for (const StudyResult& baseline : results) {
        if (baseline.restrict_mode != "bicubic_cubic")
            continue;
        for (const std::string target_mode : {
                 std::string("bicubic_quadratic"),
                 std::string("biquadratic_cubic"),
                 std::string("biquadratic_quadratic"),
                 std::string("biquadratic_quadratic_two_layer"),
                 std::string("six_point_quadratic_exterior")}) {
            const auto match = std::find_if(
                results.begin(), results.end(), [&](const StudyResult& candidate) {
                    return candidate.geometry == baseline.geometry
                        && candidate.n == baseline.n
                        && candidate.dof_mode == baseline.dof_mode
                        && candidate.dirichlet_formulation
                            == baseline.dirichlet_formulation
                        && candidate.restrict_mode == target_mode;
                });
            if (match == results.end())
                continue;
            if (!printed_header) {
                std::cout << "\nRestrict mode / bicubic-cubic error comparison\n"
                          << "  " << std::setw(9) << "geometry"
                          << std::setw(28) << "formulation"
                          << std::setw(6) << "N"
                          << std::setw(19) << "dof mode"
                          << std::setw(31) << "restrict mode"
                          << std::setw(13) << "Linf ratio"
                          << std::setw(13) << "L2 ratio"
                          << std::setw(13) << "trace ratio" << '\n';
                printed_header = true;
            }
            std::cout << "  " << std::setw(9) << baseline.geometry
                      << std::setw(28) << baseline.dirichlet_formulation
                      << std::setw(6) << baseline.n
                      << std::setw(19) << baseline.dof_mode
                      << std::setw(31) << target_mode
                      << std::setw(13) << match->global_linf / baseline.global_linf
                      << std::setw(13) << match->global_l2 / baseline.global_l2
                      << std::setw(13) << match->trace_linf / baseline.trace_linf
                      << '\n';
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
    out << "geometry,dof_mode,restrict_mode,spread_mode,N,h,dofs,crossings,"
           "dof_nearest_spacing_min,dof_nearest_spacing_mean,"
           "dof_nearest_spacing_max,"
           "gmres_iterations,converged,seconds,"
           "phase_min,phase_max,phase_std,cauchy_condition_max,"
           "cauchy_condition_mean,spread_condition_max,"
           "spread_condition_mean,augmented_boundary_res_linf,"
           "raw_exterior_trace_linf,raw_exterior_trace_l2,flux_weighted_mean,"
           "exterior_normal_linf,exterior_normal_l2,"
           "normal_route_mismatch_linf,"
           "lagrange,trace_mean,global_linf,global_l2,trace_linf,trace_l2,"
           "exterior_linf_far,exterior_l2_far,constant_shift,n_global,"
           "n_exterior_far,trace_fit_rel_l2,global_linf_order,global_l2_order,"
           "bvp,setup_seconds,total_seconds,boundary_trace_res_linf,"
           "boundary_trace_res_l2,density_mean,density_linf,density_l2,"
           "dirichlet_formulation,exact_solution,restrict_cauchy_source\n";
    out << std::setprecision(17);
    for (const StudyResult& result : results) {
        out << result.geometry << ',' << result.dof_mode << ','
            << result.restrict_mode << ','
            << result.spread_mode << ','
            << result.n << ',' << result.h << ','
            << result.dofs << ',' << result.crossings << ','
            << result.dof_nearest_spacing_min << ','
            << result.dof_nearest_spacing_mean << ','
            << result.dof_nearest_spacing_max << ','
            << result.iterations << ','
            << (result.converged ? 1 : 0) << ',' << result.seconds << ','
            << result.phase_min << ',' << result.phase_max << ','
            << result.phase_std << ',' << result.condition_max << ','
            << result.condition_mean << ',' << result.spread_condition_max << ','
            << result.spread_condition_mean << ','
            << result.augmented_residual_linf << ','
            << result.raw_exterior_trace_linf << ','
            << result.raw_exterior_trace_l2 << ',' << result.flux_mean << ','
            << result.exterior_normal_linf << ','
            << result.exterior_normal_l2 << ','
            << result.normal_route_mismatch_linf << ','
            << result.lagrange << ',' << result.trace_mean << ','
            << result.global_linf << ',' << result.global_l2 << ','
            << result.trace_linf << ',' << result.trace_l2 << ','
            << result.exterior_linf_far << ',' << result.exterior_l2_far << ','
            << result.constant_shift << ',' << result.n_global << ','
            << result.n_exterior_far << ',' << result.trace_fit_rel_l2 << ','
            << result.global_linf_order << ',' << result.global_l2_order << ','
            << result.bvp << ',' << result.setup_seconds << ','
            << result.total_seconds << ',' << result.boundary_trace_res_linf << ','
            << result.boundary_trace_res_l2 << ',' << result.density_mean << ','
            << result.density_linf << ',' << result.density_l2 << ','
            << result.dirichlet_formulation << ','
            << result.exact_solution << ','
            << result.restrict_cauchy_source << '\n';
    }
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

std::string environment_string(const char* name, const std::string& fallback)
{
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : std::string(value);
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
            if (selection == "all")
                geometries = {"ellipse", "flower", "circle", "lshape"};
            else if (selection != "both")
                geometries = {selection};
        }
        if (argc >= 3) {
            levels.clear();
            for (int i = 2; i < argc; ++i)
                levels.push_back(std::stoi(argv[i]));
        }
        const BvpKind bvp = default_bvp_kind();
        const std::string formulation_selection =
            bvp == BvpKind::Dirichlet
            ? environment_string(
                  "KFBIM_PYJET_DIRICHLET_FORMULATION",
                  "normal_jump_first_kind")
            : "not_applicable";
        const std::vector<DirichletFormulation> formulations =
            bvp == BvpKind::Dirichlet
            ? parse_dirichlet_formulations(formulation_selection)
            : std::vector<DirichletFormulation>{
                  DirichletFormulation::NormalJumpFirstKind};

        const int degree = environment_int("KFBIM_PYJET_DEGREE", 4);
        const int neighbors = environment_int("KFBIM_PYJET_NEIGHBORS", 20);
        const int derivative_neighbors = environment_int(
            "KFBIM_PYJET_DERIVATIVE_NEIGHBORS", 12);
        const int max_iter = environment_int("KFBIM_PYJET_MAX_ITER", 100);
        const int restart = environment_int("KFBIM_PYJET_RESTART", 80);
        const double tolerance = environment_double(
            "KFBIM_PYJET_TOL", 2.0e-10);
        const SpreadCorrectionMode spread_mode = parse_spread_mode(
            environment_string(
                "KFBIM_PYJET_SPREAD_MODE", "cubic_harmonic"));
        const std::string dof_mode_selection = environment_string(
            "KFBIM_PYJET_DOF_MODE", "uniform_midpoint");
        const std::vector<InterfaceDofMode> dof_modes =
            parse_dof_modes(dof_mode_selection);
        const std::string restrict_mode_selection = environment_string(
            "KFBIM_PYJET_RESTRICT_MODE",
            "bicubic_cubic");
        const std::vector<RestrictMode> restrict_modes =
            parse_restrict_modes(restrict_mode_selection);
        if (bvp == BvpKind::Dirichlet
            && formulations.size() == 1
            && formulations.front()
                == DirichletFormulation::NormalJumpSecondKind
            && restrict_modes.size() == 1
            && restrict_modes.front()
                == RestrictMode::SixPointQuadraticExterior) {
            throw std::invalid_argument(
                "normal_jump_second_kind requires a joint normal restrict mode");
        }
        const int quadratic_spread_neighbors = environment_int(
            "KFBIM_PYJET_QUADRATIC_SPREAD_NEIGHBORS", 3);
        const int quadratic_spread_derivative_neighbors = environment_int(
            "KFBIM_PYJET_QUADRATIC_SPREAD_DERIVATIVE_NEIGHBORS", 3);
        const int cubic_spread_neighbors = environment_int(
            "KFBIM_PYJET_CUBIC_SPREAD_NEIGHBORS", 4);
        const int cubic_spread_derivative_neighbors = environment_int(
            "KFBIM_PYJET_CUBIC_SPREAD_DERIVATIVE_NEIGHBORS", 3);
        const int corner_fit_degree = environment_int(
            "KFBIM_PYJET_CORNER_FIT_DEGREE", 3);
        const int corner_fit_neighbors = environment_int(
            "KFBIM_PYJET_CORNER_FIT_NEIGHBORS", 5);
        const int corner_fit_derivative_neighbors = environment_int(
            "KFBIM_PYJET_CORNER_FIT_DERIVATIVE_NEIGHBORS", 4);
        const double uniform_dof_ratio = environment_double(
            "KFBIM_PYJET_UNIFORM_DOF_RATIO", 0.75);
        const std::string trace_target = bvp != BvpKind::Dirichlet
            ? "exterior_zero"
            : formulation_selection == "normal_jump_second_kind"
                ? "exterior_normal_zero"
                : formulation_selection == "compare"
                    ? "formulation_specific"
                    : "interior_residual_zero";

        std::cout << "KFBI2D harmonic-jet " << bvp_name(bvp) << " solver\n"
                  << "bvp=" << bvp_name(bvp)
                  << " trace_target=" << trace_target << '\n'
                  << "degree=" << degree
                  << " neighbors=" << neighbors
                  << " derivative_neighbors=" << derivative_neighbors
                  << " spread_mode=" << spread_mode_name(spread_mode)
                  << " dof_mode=" << dof_mode_selection
                  << " restrict_mode=" << restrict_mode_selection
                  << " uniform_dof_ratio=" << uniform_dof_ratio
                  << " quadratic_spread_neighbors="
                  << quadratic_spread_neighbors
                  << " quadratic_spread_derivative_neighbors="
                  << quadratic_spread_derivative_neighbors
                  << " quadratic_normal_sampling="
                  << (quadratic_spread_neighbors == 3
                          && quadratic_spread_derivative_neighbors == 3
                      ? "all_three"
                      : (quadratic_spread_neighbors == 3
                             && quadratic_spread_derivative_neighbors == 2
                         ? "adjacent_symmetric" : "nearest_noncenter"))
                  << " cubic_spread_neighbors="
                  << cubic_spread_neighbors
                  << " cubic_spread_derivative_neighbors="
                  << cubic_spread_derivative_neighbors
                  << " corner_fit_degree="
                  << corner_fit_degree
                  << " corner_fit_neighbors="
                  << corner_fit_neighbors
                  << " corner_fit_derivative_neighbors="
                  << corner_fit_derivative_neighbors
                  << " restrict_jump_extension=linear_taylor"
                  << " restrict_jump_degree=1"
                  << " lshape_neighbor_policy=hybrid_4plus3_corner_cauchy_fit"
                  << " lshape_corner_policy=edge_midpoints"
                  << " tol=" << tolerance
                  << " max_iter=" << max_iter
                  << " restart=" << restart << '\n';
        if (bvp == BvpKind::Dirichlet) {
            std::cout << "dirichlet_formulation="
                      << formulation_selection << '\n';
        }

        std::vector<StudyResult> results;
        for (const std::string& name : geometries) {
            const GeometryKind geometry = parse_geometry(name);
            for (int n : levels) {
                for (InterfaceDofMode dof_mode : dof_modes) {
                    for (RestrictMode restrict_mode : restrict_modes) {
                        for (DirichletFormulation formulation : formulations) {
                            if (formulation
                                    == DirichletFormulation::NormalJumpSecondKind
                                && restrict_mode
                                    == RestrictMode::SixPointQuadraticExterior) {
                                std::cout
                                    << "[skip] normal_jump_second_kind / "
                                       "six_point_quadratic_exterior: normal "
                                       "trace is unavailable\n";
                                continue;
                            }
                            std::cout << "[run] geometry=" << name
                                      << " N=" << n
                                      << " dof_mode="
                                      << dof_mode_name(dof_mode)
                                      << " restrict_mode="
                                      << restrict_mode_name(restrict_mode);
                            if (bvp == BvpKind::Dirichlet) {
                                std::cout << " dirichlet_formulation="
                                          << dirichlet_formulation_name(
                                                 formulation);
                            }
                            std::cout << '\n';
                            StudyResult result = run_one(
                                bvp, formulation, geometry, n,
                                dof_mode, restrict_mode,
                                uniform_dof_ratio, degree,
                                neighbors, derivative_neighbors, spread_mode,
                                quadratic_spread_neighbors,
                                quadratic_spread_derivative_neighbors,
                                cubic_spread_neighbors,
                                cubic_spread_derivative_neighbors,
                                corner_fit_degree,
                                corner_fit_neighbors,
                                corner_fit_derivative_neighbors,
                                max_iter, tolerance, restart);
                            print_result(result);
                            results.push_back(std::move(result));
                        }
                    }
                }
            }
        }
        add_orders(results);
        print_summary(results);
        print_dof_comparison(results);
        print_restrict_comparison(results);

#ifdef KFBIM_APP_OUTPUT_DIR
        const std::filesystem::path output_dir = KFBIM_APP_OUTPUT_DIR;
#else
        const std::filesystem::path output_dir = "output";
#endif
        std::string csv_name = "harmonic_jet_case_cpp_python_compatible.csv";
        if (spread_mode == SpreadCorrectionMode::CrossingDensity) {
            csv_name = "harmonic_jet_case_cpp_crossing_density_spread.csv";
        } else if (spread_mode
                   == SpreadCorrectionMode::QuadraticHarmonicAtOppositeNode) {
            csv_name = "harmonic_jet_case_cpp_quadratic_harmonic_spread_n"
                     + std::to_string(quadratic_spread_neighbors) + "_d"
                     + std::to_string(quadratic_spread_derivative_neighbors)
                     + ".csv";
        } else if (spread_mode
                   == SpreadCorrectionMode::CubicHarmonicAtOppositeNode) {
            csv_name = "harmonic_jet_case_cpp_cubic_harmonic_spread_n"
                     + std::to_string(cubic_spread_neighbors) + "_d"
                     + std::to_string(cubic_spread_derivative_neighbors)
                     + ".csv";
        }
        if (dof_mode_selection != "crossing") {
            const std::size_t extension = csv_name.rfind(".csv");
            csv_name.insert(extension, "_" + dof_mode_selection + "_dofs");
        }
        if (restrict_mode_selection != "bicubic_cubic") {
            const std::size_t extension = csv_name.rfind(".csv");
            csv_name.insert(
                extension, "_" + restrict_mode_selection + "_restrict");
        }
        if (bvp == BvpKind::Dirichlet) {
            csv_name.insert(0, "dirichlet_");
            const std::size_t extension = csv_name.rfind(".csv");
            csv_name.insert(
                extension, "_" + formulation_selection + "_formulation");
        }
        const std::filesystem::path csv_path = output_dir / csv_name;
        write_csv(csv_path, results);
        std::cout << "CSV: " << csv_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
