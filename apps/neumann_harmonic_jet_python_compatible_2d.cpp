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
constexpr double kBoxMin = -2.5;
constexpr double kBoxMax = 2.5;
constexpr std::array<double, 4> kNormalLayers{{0.5, 1.5, 2.5, 3.5}};

enum class GeometryKind {
    Ellipse,
    Flower
};

enum class SpreadCorrectionMode {
    HarmonicJetAtOppositeNode,
    CrossingDensity,
    QuadraticHarmonicAtOppositeNode
};

enum class InterfaceDofMode {
    Crossing,
    UniformArcMidpoint
};

enum class RestrictMode {
    BicubicCubicNormal,
    BiquadraticQuadraticNormal,
    BiquadraticQuadraticTwoLayer,
    SixPointQuadraticExterior
};

std::string restrict_mode_name(RestrictMode mode)
{
    if (mode == RestrictMode::BicubicCubicNormal)
        return "bicubic_cubic";
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
    if (name == "compare") {
        return {RestrictMode::BicubicCubicNormal,
                RestrictMode::BiquadraticQuadraticNormal,
                RestrictMode::BiquadraticQuadraticTwoLayer,
                RestrictMode::SixPointQuadraticExterior};
    }
    throw std::invalid_argument(
        "KFBIM_PYJET_RESTRICT_MODE must be bicubic_cubic, "
        "biquadratic_quadratic, biquadratic_quadratic_two_layer, "
        "six_point_quadratic_exterior, normal_compare, or compare");
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
    throw std::invalid_argument(
        "KFBIM_PYJET_SPREAD_MODE must be harmonic_jet, crossing_density, "
        "or quadratic_harmonic");
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

GeometryKind parse_geometry(const std::string& name)
{
    if (name == "ellipse")
        return GeometryKind::Ellipse;
    if (name == "flower")
        return GeometryKind::Flower;
    throw std::invalid_argument("geometry must be ellipse or flower");
}

std::string geometry_name(GeometryKind kind)
{
    return kind == GeometryKind::Ellipse ? "ellipse" : "flower";
}

Eigen::Vector2d geometry_translation(GeometryKind kind)
{
    return kind == GeometryKind::Ellipse ? Eigen::Vector2d(0.13, -0.08)
                                         : Eigen::Vector2d(0.09, -0.06);
}

double geometry_angle(GeometryKind kind)
{
    return kind == GeometryKind::Ellipse ? 27.0 : 14.0;
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
    Eigen::Vector2d local;
    if (kind == GeometryKind::Ellipse) {
        local = Eigen::Vector2d(1.10 * std::cos(theta),
                                0.76 * std::sin(theta));
    } else {
        const double radius = flower_radius(theta);
        local = radius * Eigen::Vector2d(std::cos(theta), std::sin(theta));
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
    const Eigen::Vector2d q = to_local(kind, point);
    if (kind == GeometryKind::Ellipse) {
        const double x = q[0] / 1.10;
        const double y = q[1] / 0.76;
        return x * x + y * y < 1.0;
    }
    return flower_level_set(point) < 0.0;
}

Eigen::Vector2d geometry_normal(GeometryKind kind,
                                const Eigen::Vector2d& point)
{
    const Eigen::Vector2d q = to_local(kind, point);
    Eigen::Vector2d local;
    if (kind == GeometryKind::Ellipse) {
        local = Eigen::Vector2d(q[0] / (1.10 * 1.10),
                                q[1] / (0.76 * 0.76));
    } else {
        const double radius = std::max(q.norm(), 1.0e-14);
        const double theta = std::atan2(q[1], q[0]);
        const double rp = flower_radius_derivative(theta);
        const double c = q[0] / radius;
        const double s = q[1] / radius;
        local = Eigen::Vector2d(c + rp * s / radius,
                                s - rp * c / radius);
    }
    Eigen::Vector2d world = rotation(geometry_angle(kind)) * local;
    return world / world.norm();
}

Eigen::Vector2d segment_intersection(GeometryKind kind,
                                     const Eigen::Vector2d& p,
                                     const Eigen::Vector2d& q,
                                     double& phase)
{
    if (kind == GeometryKind::Ellipse) {
        const Eigen::Vector2d a = to_local(kind, p);
        const Eigen::Vector2d b = to_local(kind, q);
        const Eigen::Vector2d d = b - a;
        const double aa = d[0] * d[0] / (1.10 * 1.10)
                        + d[1] * d[1] / (0.76 * 0.76);
        const double bb = 2.0 * (a[0] * d[0] / (1.10 * 1.10)
                               + a[1] * d[1] / (0.76 * 0.76));
        const double cc = a[0] * a[0] / (1.10 * 1.10)
                        + a[1] * a[1] / (0.76 * 0.76) - 1.0;
        const double discriminant = bb * bb - 4.0 * aa * cc;
        if (!(discriminant >= 0.0) || aa == 0.0)
            throw std::runtime_error("ellipse segment does not cross interface");
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
            throw std::runtime_error("ellipse crossing root is ambiguous");
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
    return Eigen::Vector2d(
        a * exponential * std::cos(a * y) + 0.16 * x + 0.11,
       -a * exponential * std::sin(a * y) - 0.16 * y - 0.07);
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
    double weight = 0.0;
    int nearest_dof = -1;
    Eigen::VectorXd eval_a;
    Eigen::VectorXd eval_b;
    Eigen::VectorXd quadratic_eval_a;
    Eigen::VectorXd quadratic_eval_b;
};

struct InterfaceDof {
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    Eigen::Vector2d normal = Eigen::Vector2d::Zero();
    Eigen::Vector2d tangent = Eigen::Vector2d::Zero();
    double weight = 0.0;
    std::vector<int> neighbor_ids;
    Eigen::MatrixXd value_map;
    Eigen::MatrixXd normal_map;
    double condition = 0.0;
    std::vector<int> quadratic_neighbor_ids;
    Eigen::MatrixXd quadratic_value_map;
    Eigen::MatrixXd quadratic_normal_map;
    double quadratic_condition = 0.0;
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
                                  int degree,
                                  int neighbor_count,
                                  int derivative_count,
                                  SpreadCorrectionMode spread_mode,
                                  int quadratic_spread_neighbor_count,
                                  int quadratic_spread_derivative_count)
        : n_(n_intervals)
        , h_((kBoxMax - kBoxMin) / static_cast<double>(n_intervals))
        , geometry_(geometry)
        , dof_mode_(dof_mode)
        , restrict_mode_(restrict_mode)
        , degree_(degree)
        , dim_(2 * degree + 1)
        , requested_neighbors_(neighbor_count)
        , requested_derivative_neighbors_(derivative_count)
        , spread_mode_(spread_mode)
        , quadratic_spread_neighbor_count_(quadratic_spread_neighbor_count)
        , quadratic_spread_derivative_count_(quadratic_spread_derivative_count)
        , grid_({kBoxMin, kBoxMin}, {h_, h_}, {n_, n_}, DofLayout2D::Node)
        , bulk_solver_(grid_, ZfftBcType::Dirichlet, 0.0, 2)
    {
        if (n_ < 24)
            throw std::invalid_argument("N must be at least 24");
        if (degree_ < 2 || degree_ > 5)
            throw std::invalid_argument("harmonic jet degree must be in [2,5]");
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
        build_inside_mask();
        build_crossings();
        build_interface_dofs();
        neighbor_count_ = std::min(requested_neighbors_, size());
        derivative_count_ = std::min(requested_derivative_neighbors_,
                                     neighbor_count_);
        build_cauchy_maps();
        if (spread_mode_
                == SpreadCorrectionMode::QuadraticHarmonicAtOppositeNode
            || restrict_mode_
                != RestrictMode::BicubicCubicNormal) {
            quadratic_spread_neighbor_count_ = std::min(
                quadratic_spread_neighbor_count_, size());
            quadratic_spread_derivative_count_ = std::min(
                quadratic_spread_derivative_count_,
                quadratic_spread_neighbor_count_);
            build_quadratic_spread_maps();
        }
        build_crossing_rows();
        build_trace_templates();
        if (restrict_mode_ != RestrictMode::SixPointQuadraticExterior)
            build_joint_value_fit();
    }

    int size() const { return static_cast<int>(dofs_.size()); }
    int num_crossings() const { return static_cast<int>(crossings_.size()); }
    int n_intervals() const { return n_; }
    double h() const { return h_; }
    double boundary_length() const { return boundary_length_; }
    const CartesianGrid2D& grid() const { return grid_; }
    const std::vector<GridEdgeCrossing>& crossings() const { return crossings_; }
    const std::vector<InterfaceDof>& dofs() const { return dofs_; }

    double spread_condition(const InterfaceDof& dof) const
    {
        return spread_mode_
                   == SpreadCorrectionMode::QuadraticHarmonicAtOppositeNode
            ? dof.quadratic_condition : dof.condition;
    }

    double restrict_condition(const InterfaceDof& dof) const
    {
        return restrict_mode_ == RestrictMode::BicubicCubicNormal
            ? dof.condition : dof.quadratic_condition;
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
            result[i] = exact_solution(p[0], p[1]);
        }
        return result;
    }

    Eigen::VectorXd neumann_data() const
    {
        Eigen::VectorXd result(size());
        for (int i = 0; i < size(); ++i) {
            const InterfaceDof& dof = dofs_[static_cast<std::size_t>(i)];
            result[i] = exact_gradient(dof.point[0], dof.point[1])
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
        const Eigen::MatrixXd coefficients = restrict_mode_
                == RestrictMode::BicubicCubicNormal
            ? correction_coefficients(value_jump, normal_jump)
            : quadratic_correction_coefficients(value_jump, normal_jump);
        const int layer_count = restrict_layer_count();
        Eigen::MatrixXd inside(size(), layer_count);
        Eigen::MatrixXd outside(size(), layer_count);
        for (int i = 0; i < size(); ++i) {
            const Eigen::VectorXd coeff = coefficients.row(i).transpose();
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
    int restrict_layer_count() const
    {
        if (restrict_mode_ == RestrictMode::BicubicCubicNormal)
            return 4;
        if (restrict_mode_ == RestrictMode::BiquadraticQuadraticNormal)
            return 3;
        if (restrict_mode_ == RestrictMode::BiquadraticQuadraticTwoLayer)
            return 2;
        return 0;
    }

    int restrict_grid_side_count() const
    {
        if (restrict_mode_ == RestrictMode::BicubicCubicNormal)
            return 4;
        if (restrict_mode_ == RestrictMode::SixPointQuadraticExterior)
            return 0;
        return 3;
    }

    int restrict_normal_degree() const
    {
        return restrict_mode_ == RestrictMode::BicubicCubicNormal ? 3 : 2;
    }

    int restrict_cauchy_degree() const
    {
        return restrict_mode_ == RestrictMode::BicubicCubicNormal ? degree_ : 2;
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
                dof.weight = crossing.weight;
                dofs_.push_back(std::move(dof));
            }
        } else {
            // Keep the same unknown count as the crossing scheme so the
            // numerical comparison changes placement, not system dimension.
            // A dense parameter-space polyline supplies an inexpensive
            // approximate arc-length coordinate for both supported curves.
            const int count = static_cast<int>(crossings_.size());
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
                dof.weight = interval_length;
                dofs_.push_back(std::move(dof));
            }
        }

        if (dofs_.size() < 5)
            throw std::runtime_error("at least five interface DOFs are required");
        for (GridEdgeCrossing& crossing : crossings_) {
            double best_distance = std::numeric_limits<double>::infinity();
            for (int i = 0; i < size(); ++i) {
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
            center.quadratic_neighbor_ids.assign(
                order.begin(),
                order.begin() + quadratic_spread_neighbor_count_);

            const int rows = quadratic_spread_neighbor_count_
                           + quadratic_spread_derivative_count_;
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
                    static_cast<std::size_t>(pos)];
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
                center.quadratic_normal_map.col(pos) =
                    pinv.col(quadratic_spread_neighbor_count_ + pos)
                    * sqrt_weights[quadratic_spread_neighbor_count_ + pos] * h_;
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
                (dof.point - Eigen::Vector2d(kBoxMin, kBoxMin)) / h_;
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
        const int cauchy_degree = restrict_cauchy_degree();
        const int cauchy_dim = 2 * cauchy_degree + 1;
        trace_templates_.resize(
            static_cast<std::size_t>(size() * 2 * layer_count));
        for (int i = 0; i < size(); ++i) {
            const InterfaceDof& dof = dofs_[static_cast<std::size_t>(i)];
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
                        (query - Eigen::Vector2d(kBoxMin, kBoxMin)) / h_;
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
        trace_projection_ = design * pinv;
    }

    int n_;
    double h_;
    GeometryKind geometry_;
    InterfaceDofMode dof_mode_;
    RestrictMode restrict_mode_;
    int degree_;
    int dim_;
    int requested_neighbors_;
    int requested_derivative_neighbors_;
    SpreadCorrectionMode spread_mode_;
    int quadratic_spread_neighbor_count_;
    int quadratic_spread_derivative_count_;
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

struct StudyResult {
    std::string geometry;
    std::string dof_mode;
    std::string restrict_mode;
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
    double seconds = 0.0;
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

StudyResult run_one(GeometryKind geometry,
                    int n,
                    InterfaceDofMode dof_mode,
                    RestrictMode restrict_mode,
                    int degree,
                    int neighbors,
                    int derivative_neighbors,
                    SpreadCorrectionMode spread_mode,
                    int quadratic_spread_neighbors,
                    int quadratic_spread_derivative_neighbors,
                    int max_iter,
                    double tolerance,
                    int restart)
{
    const auto setup_start = std::chrono::steady_clock::now();
    PythonCompatibleHarmonicJet2D solver(
        n, geometry, dof_mode, restrict_mode, degree, neighbors,
        derivative_neighbors, spread_mode,
        quadratic_spread_neighbors,
        quadratic_spread_derivative_neighbors);
    const double setup_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - setup_start).count();
    std::cout << "  setup crossings=" << solver.num_crossings()
              << " dofs=" << solver.size()
              << " time=" << setup_seconds << "s\n";

    const int ns = solver.size();
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(ns);
    const Eigen::VectorXd normal_data = solver.neumann_data();
    const Eigen::VectorXd weights = solver.boundary_weights();

    const auto solve_start = std::chrono::steady_clock::now();
    const Eigen::VectorXd fixed_potential = solver.potential(zero, normal_data);
    const Eigen::VectorXd fixed_exterior = solver.exterior_trace(
        fixed_potential, zero, normal_data);

    AugmentedExteriorTraceOperator op(solver);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(ns + 1);
    rhs.head(ns) = -fixed_exterior;
    Eigen::VectorXd augmented = Eigen::VectorXd::Zero(ns + 1);
    GMRES gmres(max_iter, tolerance, std::min(restart, ns + 1));
    const int iterations = gmres.solve(op, rhs, augmented);

    const Eigen::VectorXd trace = augmented.head(ns);
    const Eigen::VectorXd potential = solver.potential(trace, normal_data);
    const Eigen::VectorXd raw_exterior = solver.exterior_trace(
        potential, trace, normal_data, true);
    Eigen::VectorXd applied;
    op.apply(augmented, applied);
    const Eigen::VectorXd augmented_residual = applied - rhs;
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();

    StudyResult result;
    result.geometry = geometry_name(geometry);
    result.dof_mode = dof_mode_name(dof_mode);
    result.restrict_mode = restrict_mode_name(restrict_mode);
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
    result.converged = gmres.converged();
    result.seconds = seconds;
    result.lagrange = augmented[ns];
    result.augmented_residual_linf = inf_norm(augmented_residual.head(ns));
    result.raw_exterior_trace_linf = inf_norm(raw_exterior);
    result.raw_exterior_trace_l2 = std::sqrt(
        raw_exterior.squaredNorm() / static_cast<double>(ns));
    result.flux_mean = weights.dot(normal_data) / solver.boundary_length();
    result.trace_mean = weights.dot(trace) / solver.boundary_length();
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
            const auto point = solver.grid().coord(i, j);
            shift_sum += exact_solution(point[0], point[1]) - potential[node];
            ++result.n_global;
        }
    }
    result.constant_shift = shift_sum / static_cast<double>(result.n_global);
    double error_sq = 0.0;
    double outside_sq = 0.0;
    for (int j = 1; j + 1 < dims[1]; ++j) {
        for (int i = 1; i + 1 < dims[0]; ++i) {
            const int node = solver.grid().index(i, j);
            const auto point_array = solver.grid().coord(i, j);
            const Eigen::Vector2d point(point_array[0], point_array[1]);
            if (solver.inside_node(node)) {
                const double error = potential[node] + result.constant_shift
                                   - exact_solution(point[0], point[1]);
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

    Eigen::VectorXd exact_trace = solver.exact_trace();
    exact_trace.array() -= weights.dot(exact_trace) / solver.boundary_length();
    const Eigen::VectorXd trace_error = trace - exact_trace;
    result.trace_linf = inf_norm(trace_error);
    result.trace_l2 = std::sqrt(
        trace_error.squaredNorm() / static_cast<double>(ns));
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
    for (const std::string name : {std::string("ellipse"),
                                   std::string("flower")}) {
        for (const std::string mode : {std::string("crossing"),
                                       std::string("uniform_midpoint")}) {
            for (const std::string restrict_mode : {
                     std::string("bicubic_cubic"),
                     std::string("biquadratic_quadratic"),
                     std::string("biquadratic_quadratic_two_layer"),
                     std::string("six_point_quadratic_exterior")}) {
                StudyResult* previous = nullptr;
                for (StudyResult& result : results) {
                    if (result.geometry != name || result.dof_mode != mode
                        || result.restrict_mode != restrict_mode) {
                        continue;
                    }
                    if (previous != nullptr) {
                        result.global_linf_order = observed_order(
                            previous->global_linf, result.global_linf,
                            previous->h, result.h);
                        result.global_l2_order = observed_order(
                            previous->global_l2, result.global_l2,
                            previous->h, result.h);
                    }
                    previous = &result;
                }
            }
        }
    }
}

void print_result(const StudyResult& result)
{
    std::cout << "  dof_mode=" << result.dof_mode
              << " restrict_mode=" << result.restrict_mode
              << " crossings=" << result.crossings
              << " dofs=" << result.dofs
              << " spacing[min/mean/max]="
              << result.dof_nearest_spacing_min << '/'
              << result.dof_nearest_spacing_mean << '/'
              << result.dof_nearest_spacing_max
              << " gmres=" << result.iterations
              << " converged=" << (result.converged ? "yes" : "no")
              << " aug_res=" << result.augmented_residual_linf
              << " raw_ext=" << result.raw_exterior_trace_linf
              << " Linf=" << result.global_linf
              << " L2=" << result.global_l2
              << " trace=" << result.trace_linf
              << " cond_max=" << result.condition_max
              << " spread_cond_max=" << result.spread_condition_max
              << " time=" << result.seconds << "s\n";
}

void print_summary(const std::vector<StudyResult>& results)
{
    for (const std::string name : {std::string("ellipse"),
                                   std::string("flower")}) {
        for (const std::string mode : {std::string("crossing"),
                                       std::string("uniform_midpoint")}) {
            for (const std::string restrict_mode : {
                     std::string("bicubic_cubic"),
                     std::string("biquadratic_quadratic"),
                     std::string("biquadratic_quadratic_two_layer"),
                     std::string("six_point_quadratic_exterior")}) {
                const bool present = std::any_of(
                    results.begin(), results.end(), [&](const StudyResult& result) {
                        return result.geometry == name && result.dof_mode == mode
                            && result.restrict_mode == restrict_mode;
                    });
                if (!present)
                    continue;
                std::cout << "\nC++ harmonic-jet summary: " << name
                          << " / " << mode << " / " << restrict_mode << '\n';
                std::cout << "  " << std::setw(5) << "N"
                          << std::setw(7) << "dofs"
                          << std::setw(14) << "Linf"
                          << std::setw(11) << "p_inf"
                          << std::setw(14) << "L2"
                          << std::setw(11) << "p_L2"
                          << std::setw(9) << "GMRES" << '\n';
                for (const StudyResult& result : results) {
                    if (result.geometry != name || result.dof_mode != mode
                        || result.restrict_mode != restrict_mode) {
                        continue;
                    }
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
                    && candidate.dof_mode == "uniform_midpoint";
            });
        if (match == results.end())
            continue;
        if (!printed_header) {
            std::cout << "\nUniform-midpoint / crossing-DOF error comparison\n"
                      << "  " << std::setw(9) << "geometry"
                      << std::setw(6) << "N"
                      << std::setw(27) << "restrict"
                      << std::setw(13) << "Linf ratio"
                      << std::setw(13) << "L2 ratio"
                      << std::setw(13) << "trace ratio" << '\n';
            printed_header = true;
        }
        std::cout << "  " << std::setw(9) << baseline.geometry
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
                 std::string("biquadratic_quadratic"),
                 std::string("biquadratic_quadratic_two_layer"),
                 std::string("six_point_quadratic_exterior")}) {
            const auto match = std::find_if(
                results.begin(), results.end(), [&](const StudyResult& candidate) {
                    return candidate.geometry == baseline.geometry
                        && candidate.n == baseline.n
                        && candidate.dof_mode == baseline.dof_mode
                        && candidate.restrict_mode == target_mode;
                });
            if (match == results.end())
                continue;
            if (!printed_header) {
                std::cout << "\nRestrict mode / bicubic-cubic error comparison\n"
                          << "  " << std::setw(9) << "geometry"
                          << std::setw(6) << "N"
                          << std::setw(19) << "dof mode"
                          << std::setw(31) << "restrict mode"
                          << std::setw(13) << "Linf ratio"
                          << std::setw(13) << "L2 ratio"
                          << std::setw(13) << "trace ratio" << '\n';
                printed_header = true;
            }
            std::cout << "  " << std::setw(9) << baseline.geometry
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
           "lagrange,trace_mean,global_linf,global_l2,trace_linf,trace_l2,"
           "exterior_linf_far,exterior_l2_far,constant_shift,n_global,"
           "n_exterior_far,trace_fit_rel_l2,global_linf_order,global_l2_order\n";
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
            << result.lagrange << ',' << result.trace_mean << ','
            << result.global_linf << ',' << result.global_l2 << ','
            << result.trace_linf << ',' << result.trace_l2 << ','
            << result.exterior_linf_far << ',' << result.exterior_l2_far << ','
            << result.constant_shift << ',' << result.n_global << ','
            << result.n_exterior_far << ',' << result.trace_fit_rel_l2 << ','
            << result.global_linf_order << ',' << result.global_l2_order << '\n';
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
            if (selection != "both")
                geometries = {selection};
        }
        if (argc >= 3) {
            levels.clear();
            for (int i = 2; i < argc; ++i)
                levels.push_back(std::stoi(argv[i]));
        }

        const int degree = environment_int("KFBIM_PYJET_DEGREE", 4);
        const int neighbors = environment_int("KFBIM_PYJET_NEIGHBORS", 20);
        const int derivative_neighbors = environment_int(
            "KFBIM_PYJET_DERIVATIVE_NEIGHBORS", 12);
        const int max_iter = environment_int("KFBIM_PYJET_MAX_ITER", 100);
        const int restart = environment_int("KFBIM_PYJET_RESTART", 80);
        const double tolerance = environment_double(
            "KFBIM_PYJET_TOL", 2.0e-10);
        const SpreadCorrectionMode spread_mode = parse_spread_mode(
            environment_string("KFBIM_PYJET_SPREAD_MODE", "quadratic_harmonic"));
        const std::string dof_mode_selection = environment_string(
            "KFBIM_PYJET_DOF_MODE", "compare");
        const std::vector<InterfaceDofMode> dof_modes =
            parse_dof_modes(dof_mode_selection);
        const std::string restrict_mode_selection = environment_string(
            "KFBIM_PYJET_RESTRICT_MODE", "bicubic_cubic");
        const std::vector<RestrictMode> restrict_modes =
            parse_restrict_modes(restrict_mode_selection);
        const int quadratic_spread_neighbors = environment_int(
            "KFBIM_PYJET_QUADRATIC_SPREAD_NEIGHBORS", 3);
        const int quadratic_spread_derivative_neighbors = environment_int(
            "KFBIM_PYJET_QUADRATIC_SPREAD_DERIVATIVE_NEIGHBORS", 2);

        std::cout << "KFBI2D harmonic-jet Neumann solver\n"
                  << "degree=" << degree
                  << " neighbors=" << neighbors
                  << " derivative_neighbors=" << derivative_neighbors
                  << " spread_mode=" << spread_mode_name(spread_mode)
                  << " dof_mode=" << dof_mode_selection
                  << " restrict_mode=" << restrict_mode_selection
                  << " quadratic_spread_neighbors="
                  << quadratic_spread_neighbors
                  << " quadratic_spread_derivative_neighbors="
                  << quadratic_spread_derivative_neighbors
                  << " tol=" << tolerance
                  << " max_iter=" << max_iter
                  << " restart=" << restart << '\n';

        std::vector<StudyResult> results;
        for (const std::string& name : geometries) {
            const GeometryKind geometry = parse_geometry(name);
            for (int n : levels) {
                for (InterfaceDofMode dof_mode : dof_modes) {
                    for (RestrictMode restrict_mode : restrict_modes) {
                        std::cout << "[run] geometry=" << name << " N=" << n
                                  << " dof_mode=" << dof_mode_name(dof_mode)
                                  << " restrict_mode="
                                  << restrict_mode_name(restrict_mode) << '\n';
                        StudyResult result = run_one(
                            geometry, n, dof_mode, restrict_mode, degree,
                            neighbors, derivative_neighbors, spread_mode,
                            quadratic_spread_neighbors,
                            quadratic_spread_derivative_neighbors,
                            max_iter, tolerance, restart);
                        print_result(result);
                        results.push_back(std::move(result));
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
        const std::filesystem::path csv_path = output_dir / csv_name;
        write_csv(csv_path, results);
        std::cout << "CSV: " << csv_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
