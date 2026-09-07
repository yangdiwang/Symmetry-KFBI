#include "src/support/density/cap_atlas_density_space_3d.hpp"
#include "src/support/cauchy/crossing_cauchy_plan_3d.hpp"
#include "src/support/trace/mean_free_trace_mass_coordinates_3d.hpp"
#include "src/support/trace/reduced_trace_projection_3d.hpp"

#include "src/bulk_solvers/laplace_zfft_bulk_solver_3d.hpp"
#include "src/bulk_solvers/zfft_bc_type.hpp"
#include "src/gmres/gmres.hpp"
#include "src/grid/cartesian_grid_3d.hpp"
#include "src/grid/dof_layout.hpp"
#include "src/operators/i_kfbi_operator.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SparseCore>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef KFBIM_APP_OUTPUT_DIR
#define KFBIM_APP_OUTPUT_DIR "output"
#endif

namespace {

using kfbim::CartesianGrid3D;
using kfbim::DofLayout3D;
using kfbim::GMRES;
using kfbim::IKFBIOperator;
using kfbim::LaplaceFftBulkSolverZfft3D;
using kfbim::ZfftBcType;
using kfbim::app3d::CapAtlasDensityOptions3D;
using kfbim::app3d::CapAtlasDensitySpace3D;
using kfbim::app3d::CapC0BasisStencil3D;
using kfbim::app3d::CapShape3D;
using kfbim::app3d::CauchyPolynomialWeights3D;
using kfbim::app3d::JetRecoveryMatrix3D;
using kfbim::app3d::JetRecoveryOptions3D;
using kfbim::app3d::LocalOrthonormalFrame3D;
using kfbim::app3d::MeanFreeTraceMassCoordinates3D;
using kfbim::app3d::ReducedTraceProjection3D;
using kfbim::app3d::SurfacePointMatrix3D;
using kfbim::app3d::TangentGraphHessian3D;
using kfbim::app3d::build_normal_jet_recovery_3d;
using kfbim::app3d::build_value_jet_recovery_3d;
using kfbim::app3d::cauchy_polynomial_weights_3d;
using kfbim::app3d::make_local_orthonormal_frame_3d;

// Match docs/kfbi_general_cap_exterior_trace.py exactly.  With its reference
// levels N >= 32, the 1.4 h trace layers and four-node interpolation support
// remain inside this box.
constexpr double box_min = -1.5;
constexpr double box_side = 3.0;
constexpr double aa = 0.35;
constexpr double bb = 0.21;
constexpr double cc = 0.28;

struct RigidPose3D {
    std::string id;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();

    Eigen::Vector3d inverse_point(const Eigen::Vector3d& point) const
    {
        return center
               + rotation.transpose()
                     * (point - center - translation);
    }

    Eigen::Vector3d forward_vector(const Eigen::Vector3d& vector) const
    {
        return rotation * vector;
    }
};

std::vector<RigidPose3D> rigid_pose_catalog()
{
    constexpr double degrees_to_radians =
        3.141592653589793238462643383279502884 / 180.0;
    const Eigen::Vector3d center(0.07, -0.07, 0.02);
    const Eigen::Vector3d axis = Eigen::Vector3d(1.0, 2.0, 3.0).normalized();
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(17.0 * degrees_to_radians, axis)
            .toRotationMatrix();
    const Eigen::Vector3d xyz1(0.137, -0.083, 0.061);
    return {
        {"baseline", Eigen::Matrix3d::Identity(), center,
         Eigen::Vector3d::Zero()},
        {"tx_p0137", Eigen::Matrix3d::Identity(), center,
         Eigen::Vector3d(0.137, 0.0, 0.0)},
        {"ty_m0083", Eigen::Matrix3d::Identity(), center,
         Eigen::Vector3d(0.0, -0.083, 0.0)},
        {"tz_p0061", Eigen::Matrix3d::Identity(), center,
         Eigen::Vector3d(0.0, 0.0, 0.061)},
        {"t_xyz_1", Eigen::Matrix3d::Identity(), center, xyz1},
        {"t_xyz_2", Eigen::Matrix3d::Identity(), center,
         Eigen::Vector3d(-0.109, 0.151, -0.047)},
        {"rot_axis123_17deg", rotation, center,
         Eigen::Vector3d::Zero()},
        {"rot_axis123_17deg_t_xyz_1", rotation, center, xyz1},
    };
}

RigidPose3D selected_rigid_pose()
{
    const char* raw = std::getenv("KFBIM_3D_RIGID_CASE");
    const std::string requested =
        raw == nullptr || std::string(raw).empty() ? "baseline" : raw;
    const std::vector<RigidPose3D> poses = rigid_pose_catalog();
    const auto found = std::find_if(
        poses.begin(), poses.end(), [&](const RigidPose3D& pose) {
            return pose.id == requested;
        });
    if (found != poses.end())
        return *found;
    std::ostringstream message;
    message << "KFBIM_3D_RIGID_CASE must name one of";
    for (const RigidPose3D& pose : poses)
        message << ' ' << pose.id;
    throw std::invalid_argument(message.str());
}

enum class SolveSelection3D {
    Both,
    NeumannOnly,
    DirichletNormalOnly
};

SolveSelection3D selected_solve_selection()
{
    const char* raw = std::getenv("KFBIM_3D_SOLVE_SELECTION");
    if (raw == nullptr || std::string(raw).empty()
        || std::string(raw) == "both")
        return SolveSelection3D::Both;
    if (std::string(raw) == "neumann_only")
        return SolveSelection3D::NeumannOnly;
    if (std::string(raw) == "dirichlet_normal_only")
        return SolveSelection3D::DirichletNormalOnly;
    throw std::invalid_argument(
        "KFBIM_3D_SOLVE_SELECTION must be both, neumann_only, or "
        "dirichlet_normal_only");
}

const char* solve_selection_name(SolveSelection3D selection)
{
    switch (selection) {
    case SolveSelection3D::Both:
        return "both";
    case SolveSelection3D::NeumannOnly:
        return "neumann_only";
    case SolveSelection3D::DirichletNormalOnly:
        return "dirichlet_normal_only";
    }
    throw std::runtime_error("unknown solve selection");
}

int positive_integer_environment(const char* name, int default_value)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || std::string(raw).empty())
        return default_value;
    const std::string text(raw);
    std::size_t consumed = 0;
    int value = 0;
    try {
        value = std::stoi(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name)
                                    + " must be a positive integer");
    }
    if (consumed != text.size() || value <= 0)
        throw std::invalid_argument(std::string(name)
                                    + " must be a positive integer");
    return value;
}

double positive_double_environment(const char* name, double default_value)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || std::string(raw).empty())
        return default_value;
    const std::string text(raw);
    std::size_t consumed = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name)
                                    + " must be a positive finite number");
    }
    if (consumed != text.size() || !(value > 0.0) || !std::isfinite(value))
        throw std::invalid_argument(std::string(name)
                                    + " must be a positive finite number");
    return value;
}

struct ShapeSpecification3D {
    std::string name;
    CapShape3D kind = CapShape3D::Ellipsoid;
    Eigen::Vector3d ellipsoid_axes = Eigen::Vector3d(1.20, 0.90, 0.72);
};

ShapeSpecification3D shape_specification(const std::string& name)
{
    if (name == "sphere")
        return {"sphere", CapShape3D::Ellipsoid, Eigen::Vector3d::Ones()};
    if (name == "ellipsoid")
        return {"ellipsoid", CapShape3D::Ellipsoid,
                Eigen::Vector3d(1.20, 0.90, 0.72)};
    if (name == "flower")
        return {"flower", CapShape3D::Flower,
                Eigen::Vector3d(1.20, 0.90, 0.72)};
    throw std::invalid_argument(
        "shape must be sphere, ellipsoid, flower, or all");
}

CapAtlasDensityOptions3D make_cap_options(
    const ShapeSpecification3D& shape,
    const RigidPose3D& pose,
    int coefficients_per_direction)
{
    CapAtlasDensityOptions3D options;
    options.shape = shape.kind;
    options.ellipsoid_axes = shape.ellipsoid_axes;
    options.coefficients_per_direction = coefficients_per_direction;
    options.mortar_gauss_order = 3;
    options.trace_gauss_order = 3;
    options.rigid_rotation = pose.rotation;
    options.rigid_center = pose.center;
    options.rigid_translation = pose.translation;
    return options;
}

double exact_u(const Eigen::Vector3d& p)
{
    return std::exp(aa * p.x()) * std::cos(bb * p.y())
           * std::cos(cc * p.z());
}

Eigen::Vector3d exact_gradient(const Eigen::Vector3d& p)
{
    const double e = std::exp(aa * p.x());
    return {aa * e * std::cos(bb * p.y()) * std::cos(cc * p.z()),
            -bb * e * std::sin(bb * p.y()) * std::cos(cc * p.z()),
            -cc * e * std::cos(bb * p.y()) * std::sin(cc * p.z())};
}

struct ShapeModel3D {
    CapAtlasDensityOptions3D options;

    Eigen::Vector3d inverse_point(const Eigen::Vector3d& p) const
    {
        return options.rigid_center
               + options.rigid_rotation.transpose()
                     * (p - options.rigid_center
                        - options.rigid_translation);
    }

    double exact_value(const Eigen::Vector3d& p) const
    {
        return exact_u(inverse_point(p));
    }

    Eigen::Vector3d exact_physical_gradient(const Eigen::Vector3d& p) const
    {
        return options.rigid_rotation * exact_gradient(inverse_point(p));
    }

    double flower_radius(const Eigen::Vector3d& d) const
    {
        const double x = d.x();
        const double y = d.y();
        const double z = d.z();
        const double h4 = x * x * x * x - 6.0 * x * x * y * y
                          + y * y * y * y;
        return 1.0 + options.flower_epsilon * h4
               + options.flower_eta * (3.0 * z * z - 1.0);
    }

    Eigen::Vector3d flower_radius_gradient(const Eigen::Vector3d& d) const
    {
        const double x = d.x();
        const double y = d.y();
        const double z = d.z();
        return {options.flower_epsilon
                    * (4.0 * x * x * x - 12.0 * x * y * y),
                options.flower_epsilon
                    * (-12.0 * x * x * y + 4.0 * y * y * y),
                6.0 * options.flower_eta * z};
    }

    double implicit(const Eigen::Vector3d& p) const
    {
        const Eigen::Vector3d local = inverse_point(p);
        if (options.shape == CapShape3D::Ellipsoid) {
            return local.cwiseQuotient(options.ellipsoid_axes).squaredNorm()
                   - 1.0;
        }
        const double r = local.norm();
        if (r <= 1.0e-14)
            return -1.0;
        return r - flower_radius(local / r);
    }

    Eigen::Vector3d gradient(const Eigen::Vector3d& p) const
    {
        const Eigen::Vector3d local = inverse_point(p);
        Eigen::Vector3d local_gradient;
        if (options.shape == CapShape3D::Ellipsoid) {
            local_gradient = 2.0
                             * local.cwiseQuotient(
                                 options.ellipsoid_axes.cwiseProduct(
                                     options.ellipsoid_axes));
        } else {
            const double r = local.norm();
            if (r <= 1.0e-14)
                local_gradient = Eigen::Vector3d::UnitX();
            else {
                const Eigen::Vector3d d = local / r;
                const Eigen::Vector3d gd = flower_radius_gradient(d);
                const Eigen::Vector3d tangential = gd - gd.dot(d) * d;
                local_gradient = d - tangential / r;
            }
        }
        return options.rigid_rotation * local_gradient;
    }

    bool inside(const Eigen::Vector3d& p) const { return implicit(p) < 0.0; }

    Eigen::Vector3d normal(const Eigen::Vector3d& p) const
    {
        return gradient(p).normalized();
    }

    TangentGraphHessian3D graph_hessian(
        const Eigen::Vector3d& q,
        const LocalOrthonormalFrame3D& frame) const
    {
        const double eps = 2.0e-5;
        const Eigen::Vector3d g0 = gradient(q);
        const double gn = g0.norm();
        const Eigen::Vector3d h_t1 =
            (gradient(q + eps * frame.tangent1)
             - gradient(q - eps * frame.tangent1))
            / (2.0 * eps);
        const Eigen::Vector3d h_t2 =
            (gradient(q + eps * frame.tangent2)
             - gradient(q - eps * frame.tangent2))
            / (2.0 * eps);
        TangentGraphHessian3D h;
        h.h11 = -frame.tangent1.dot(h_t1) / gn;
        h.h22 = -frame.tangent2.dot(h_t2) / gn;
        h.h12 = -0.5
                * (frame.tangent1.dot(h_t2)
                   + frame.tangent2.dot(h_t1))
                / gn;
        return h;
    }

    Eigen::Vector3d project_tangent_graph(
        const Eigen::Vector3d& q,
        const LocalOrthonormalFrame3D& frame,
        const TangentGraphHessian3D& h,
        double s,
        double t) const
    {
        double r = 0.5
                   * (h.h11 * s * s + 2.0 * h.h12 * s * t
                      + h.h22 * t * t);
        for (int iteration = 0; iteration < 8; ++iteration) {
            const Eigen::Vector3d p = q + s * frame.tangent1
                                      + t * frame.tangent2
                                      + r * frame.normal;
            const double denominator = gradient(p).dot(frame.normal);
            if (std::abs(denominator) < 1.0e-13)
                throw std::runtime_error(
                    "Normal graph projection encountered a tangent Newton line");
            r -= implicit(p) / denominator;
        }
        const Eigen::Vector3d p = q + s * frame.tangent1
                                  + t * frame.tangent2 + r * frame.normal;
        if (std::abs(implicit(p)) > 2.0e-10)
            throw std::runtime_error("Normal graph projection did not converge");
        return p;
    }

    Eigen::Vector3d crossing(const Eigen::Vector3d& a,
                             const Eigen::Vector3d& b) const
    {
        double lo = 0.0;
        double hi = 1.0;
        const double fa = implicit(a);
        for (int iteration = 0; iteration < 45; ++iteration) {
            const double mid = 0.5 * (lo + hi);
            const double fm = implicit(a + mid * (b - a));
            if (fm * fa > 0.0)
                lo = mid;
            else
                hi = mid;
        }
        return a + 0.5 * (lo + hi) * (b - a);
    }
};

constexpr int local_density_sample_count = 9;

struct LocalDensityJetPlan3D {
    LocalOrthonormalFrame3D frame;
    TangentGraphHessian3D graph_hessian;
    std::array<CapC0BasisStencil3D, local_density_sample_count> samples;
    Eigen::Matrix<double, 6, local_density_sample_count> value_recovery =
        Eigen::Matrix<double, 6, local_density_sample_count>::Zero();
    Eigen::Matrix<double, 3, local_density_sample_count> normal_recovery =
        Eigen::Matrix<double, 3, local_density_sample_count>::Zero();
    Eigen::Matrix<double, 6, 1> exact_value_jet;
    Eigen::Matrix<double, 3, 1> exact_normal_jet;
    double value_condition = 0.0;
    double normal_condition = 0.0;

    Eigen::Matrix<double, local_density_sample_count, 1> sample_values(
        const Eigen::VectorXd& c0_coefficients) const
    {
        Eigen::Matrix<double, local_density_sample_count, 1> values;
        for (int sample = 0; sample < local_density_sample_count; ++sample) {
            values[sample] =
                samples[static_cast<std::size_t>(sample)].dot(c0_coefficients);
        }
        return values;
    }

    Eigen::Matrix<double, 6, 1> recover_value_jet(
        const Eigen::VectorXd& c0_coefficients) const
    {
        return value_recovery * sample_values(c0_coefficients);
    }

    Eigen::Matrix<double, 3, 1> recover_normal_jet(
        const Eigen::VectorXd& c0_coefficients) const
    {
        return normal_recovery * sample_values(c0_coefficients);
    }
};

LocalDensityJetPlan3D build_local_density_jet_plan(
    const Eigen::Vector3d& q,
    double delta,
    const ShapeModel3D& shape,
    const CapAtlasDensitySpace3D& density)
{
    LocalDensityJetPlan3D result;
    result.frame = make_local_orthonormal_frame_3d(shape.normal(q));
    result.graph_hessian = shape.graph_hessian(q, result.frame);

    SurfacePointMatrix3D sample_points(local_density_sample_count, 3);
    Eigen::VectorXd exact_values(local_density_sample_count);
    Eigen::VectorXd exact_normals(local_density_sample_count);
    int row = 0;
    for (int js = -1; js <= 1; ++js) {
        for (int is = -1; is <= 1; ++is) {
            const Eigen::Vector3d p = shape.project_tangent_graph(
                q,
                result.frame,
                result.graph_hessian,
                delta * static_cast<double>(is),
                delta * static_cast<double>(js));
            sample_points.row(row) = p.transpose();
            const auto location = density.locate(p);
            if (!location.valid)
                throw std::runtime_error(
                    "Cap atlas could not locate a local Cauchy sample");
            result.samples[static_cast<std::size_t>(row)] =
                density.c0_basis_stencil(
                    location.patch, location.u, location.v);
            exact_values[row] = shape.exact_value(p);
            exact_normals[row] =
                shape.exact_physical_gradient(p).dot(shape.normal(p));
            ++row;
        }
    }

    JetRecoveryOptions3D recovery_options;
    recovery_options.coordinate_scale = delta;
    const JetRecoveryMatrix3D value_recovery =
        build_value_jet_recovery_3d(
            q, result.frame, sample_points, recovery_options);
    const JetRecoveryMatrix3D normal_recovery =
        build_normal_jet_recovery_3d(
            q, result.frame, sample_points, recovery_options);
    result.value_recovery = value_recovery.matrix;
    result.normal_recovery = normal_recovery.matrix;
    result.exact_value_jet = value_recovery.recover(exact_values);
    result.exact_normal_jet = normal_recovery.recover(exact_normals);
    result.value_condition = value_recovery.diagnostics.condition_number;
    result.normal_condition = normal_recovery.diagnostics.condition_number;
    return result;
}

struct DirectedCrossingTarget3D {
    int grid_index = -1;
    Eigen::Matrix<double, 1, 6> value_jet_weight =
        Eigen::Matrix<double, 1, 6>::Zero();
    Eigen::Matrix<double, 1, 3> normal_jet_weight =
        Eigen::Matrix<double, 1, 3>::Zero();
    double scale = 0.0;
    double exact_value_correction = 0.0;
    double exact_normal_correction = 0.0;
};

struct CrossingPlan3D {
    LocalDensityJetPlan3D density;
    std::array<DirectedCrossingTarget3D, 2> targets;
};

struct GridTraceWeight3D {
    int index = -1;
    double value_weight = 0.0;
    double normal_weight = 0.0;
};

struct ExteriorTracePlan3D {
    LocalDensityJetPlan3D density;
    std::vector<GridTraceWeight3D> grid_weights;
    Eigen::Matrix<double, 1, 6> value_from_value_jet =
        Eigen::Matrix<double, 1, 6>::Zero();
    Eigen::Matrix<double, 1, 3> value_from_normal_jet =
        Eigen::Matrix<double, 1, 3>::Zero();
    Eigen::Matrix<double, 1, 6> normal_from_value_jet =
        Eigen::Matrix<double, 1, 6>::Zero();
    Eigen::Matrix<double, 1, 3> normal_from_normal_jet =
        Eigen::Matrix<double, 1, 3>::Zero();
    double value_from_exact_value_jump = 0.0;
    double value_from_exact_normal_jump = 0.0;
    double normal_from_exact_value_jump = 0.0;
    double normal_from_exact_normal_jump = 0.0;
};

struct TracePair3D {
    Eigen::VectorXd value;
    Eigen::VectorXd normal;
    Eigen::VectorXd raw_value;
    Eigen::VectorXd raw_normal;
};

Eigen::Matrix<double, 4, 1> cubic_grid_weights(double f)
{
    Eigen::Matrix<double, 4, 1> w;
    w[0] = -f * (f - 1.0) * (f - 2.0) / 6.0;
    w[1] = (f + 1.0) * (f - 1.0) * (f - 2.0) / 2.0;
    w[2] = -(f + 1.0) * f * (f - 2.0) / 2.0;
    w[3] = (f + 1.0) * f * (f - 1.0) / 6.0;
    return w;
}

struct CubicGridStencil1D {
    int start = 0;
    Eigen::Matrix<double, 4, 1> weights =
        Eigen::Matrix<double, 4, 1>::Zero();
};

CubicGridStencil1D cubic_grid_stencil(double coordinate, int intervals)
{
    if (intervals < 3 || !std::isfinite(coordinate)
        || coordinate < 0.0 || coordinate > intervals) {
        throw std::out_of_range(
            "Cubic interpolation coordinate lies outside the grid");
    }
    const int lower = static_cast<int>(std::floor(coordinate));
    const int centered_start = lower - 1;
    CubicGridStencil1D stencil;
    stencil.start = std::clamp(centered_start, 0, intervals - 3);
    if (stencil.start == centered_start) {
        // Preserve the historical centred path, including its floating-point
        // operation order, whenever all four nodes lie inside the box.
        stencil.weights = cubic_grid_weights(coordinate - lower);
        return stencil;
    }

    // Near a box face use the four nearest in-box nodes.  This is the same
    // degree-three Lagrange interpolant, only expressed on nodes 0,1,2,3
    // relative to the clamped stencil start.
    const double x = coordinate - stencil.start;
    for (int j = 0; j < 4; ++j) {
        double weight = 1.0;
        for (int m = 0; m < 4; ++m) {
            if (m == j)
                continue;
            weight *= (x - static_cast<double>(m))
                      / static_cast<double>(j - m);
        }
        stencil.weights[j] = weight;
    }
    return stencil;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> normal_trace_fit_weights()
{
    const std::array<double, 8> rho = {
        -1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4};
    Eigen::MatrixXd v(8, 4);
    for (int i = 0; i < 8; ++i) {
        v(i, 0) = 1.0;
        v(i, 1) = rho[static_cast<std::size_t>(i)];
        v(i, 2) = v(i, 1) * v(i, 1);
        v(i, 3) = v(i, 2) * v(i, 1);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        v, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd inverse = svd.singularValues();
    for (Eigen::Index i = 0; i < inverse.size(); ++i)
        inverse[i] = 1.0 / inverse[i];
    const Eigen::MatrixXd pinverse =
        svd.matrixV() * inverse.asDiagonal() * svd.matrixU().transpose();
    return {pinverse.row(0).transpose(), pinverse.row(1).transpose()};
}

constexpr double target_crossings_per_density_cell = 14.0;

struct DensityResolutionChoice3D {
    int patch_count = 0;
    double orientation_integral = 0.0;
    double predicted_crossings = 0.0;
    double continuous_elements_per_direction = 0.0;
    int elements_per_direction = 1;
    int coefficients_per_direction = 4;
};

// Geometry-only density initialization.  For a Cartesian grid the expected
// number of coordinate-edge crossings is
//
//   h^{-2} integral_Gamma (|n_x|+|n_y|+|n_z|) dS.
//
// Dividing it among P*e^2 tensor knot cells and prescribing a target number
// of crossings per cell gives e directly, without trying candidate density
// spaces or using an already-counted crossing set.
DensityResolutionChoice3D choose_density_resolution(
    const ShapeSpecification3D& shape,
    const RigidPose3D& pose,
    int intervals)
{
    if (intervals < 8)
        throw std::invalid_argument("At least eight grid intervals are required");
    CapAtlasDensityOptions3D probe_options =
        make_cap_options(shape, pose, 4);
    CapAtlasDensitySpace3D geometry_probe(probe_options);

    DensityResolutionChoice3D choice;
    choice.patch_count = geometry_probe.patch_count();
    constexpr int geometry_subdivisions = 8;
    constexpr std::array<double, 5> gauss_nodes = {
        -0.90617984593866399280,
        -0.53846931010568309104,
        0.0,
        0.53846931010568309104,
        0.90617984593866399280};
    constexpr std::array<double, 5> gauss_weights = {
        0.23692688505618908751,
        0.47862867049936646804,
        0.56888888888888888889,
        0.47862867049936646804,
        0.23692688505618908751};
    for (int patch = 0; patch < geometry_probe.patch_count(); ++patch) {
        for (int eu = 0; eu < geometry_subdivisions; ++eu) {
            const double ua = static_cast<double>(eu) / geometry_subdivisions;
            const double ub = static_cast<double>(eu + 1)
                              / geometry_subdivisions;
            for (int ev = 0; ev < geometry_subdivisions; ++ev) {
                const double va =
                    static_cast<double>(ev) / geometry_subdivisions;
                const double vb = static_cast<double>(ev + 1)
                                  / geometry_subdivisions;
                for (std::size_t gu = 0; gu < gauss_nodes.size(); ++gu) {
                    const double u = 0.5
                                     * ((ub - ua) * gauss_nodes[gu]
                                        + ub + ua);
                    const double wu =
                        0.5 * (ub - ua) * gauss_weights[gu];
                    for (std::size_t gv = 0; gv < gauss_nodes.size(); ++gv) {
                        const double v = 0.5
                                         * ((vb - va) * gauss_nodes[gv]
                                            + vb + va);
                        const double wv =
                            0.5 * (vb - va) * gauss_weights[gv];
                        const auto geometry =
                            geometry_probe.geometry(patch, u, v);
                        choice.orientation_integral +=
                            wu * wv * geometry.area_element
                            * geometry.normal.cwiseAbs().sum();
                    }
                }
            }
        }
    }
    const double h = box_side / static_cast<double>(intervals);
    choice.predicted_crossings =
        choice.orientation_integral / (h * h);
    choice.continuous_elements_per_direction = std::sqrt(
        choice.orientation_integral
        / (static_cast<double>(choice.patch_count)
           * target_crossings_per_density_cell * h * h));
    // floor selects the most resolved space whose predicted cell coverage is
    // still at least the prescribed target.  Moving by one representable
    // value only avoids losing an analytically integral value to roundoff.
    const double threshold_safe_elements = std::nextafter(
        choice.continuous_elements_per_direction,
        std::numeric_limits<double>::infinity());
    choice.elements_per_direction = std::max(
        1,
        static_cast<int>(std::floor(threshold_safe_elements)));
    choice.coefficients_per_direction = choice.elements_per_direction + 3;
    return choice;
}

class CoefficientKfbiPipeline3D {
public:
    CoefficientKfbiPipeline3D(const ShapeSpecification3D& shape,
                              const RigidPose3D& pose,
                              int intervals,
                              int coefficients_per_direction)
        : density_options_(make_cap_options(
              shape, pose, coefficients_per_direction)),
          shape_{density_options_},
          density_(density_options_),
          intervals_(intervals),
          h_(box_side / static_cast<double>(intervals_)),
          grid_({box_min, box_min, box_min},
                {h_, h_, h_},
                {intervals_, intervals_, intervals_},
                DofLayout3D::Node),
          bulk_(grid_, ZfftBcType::Dirichlet, 0.0, 2)
    {
        if (intervals_ < 8)
            throw std::invalid_argument("At least eight grid intervals are required");
        const auto begin = std::chrono::steady_clock::now();
        classify_grid();
        build_crossing_rows();
        build_trace_plans();
        setup_seconds_ = elapsed_seconds(begin);
    }

    int coefficient_count() const
    {
        return density_.reduced_coefficient_count();
    }
    int trace_count() const { return static_cast<int>(trace_plans_.size()); }
    int unique_crossing_count() const { return unique_crossings_; }
    int directed_crossing_count() const
    {
        return 2 * static_cast<int>(crossing_plans_.size());
    }
    int grid_dof_count() const { return grid_.num_dofs(); }
    double spacing() const { return h_; }
    double setup_seconds() const { return setup_seconds_; }
    double maximum_value_recovery_condition() const
    {
        return max_value_condition_;
    }
    double maximum_normal_recovery_condition() const
    {
        return max_normal_condition_;
    }
    const CapAtlasDensitySpace3D& density() const { return density_; }
    const ReducedTraceProjection3D& projection() const
    {
        return *projection_;
    }
    const Eigen::VectorXd& trace_weights() const { return trace_weights_; }
    const Eigen::VectorXd& mass_row() const { return mass_row_; }
    const Eigen::VectorXd& constant_coefficients() const
    {
        return constant_coefficients_;
    }
    double area() const { return trace_weights_.sum(); }

    Eigen::VectorXd project_trace(const Eigen::VectorXd& values) const
    {
        return projection_->apply(values);
    }

    Eigen::VectorXd evaluate_trace(
        const Eigen::Ref<const Eigen::VectorXd>& coefficients) const
    {
        check_coefficients(coefficients, "trace density");
        return trace_c0_design_ * density_.expand_reduced(coefficients);
    }

    Eigen::VectorXd exact_value_trace() const
    {
        Eigen::VectorXd values(trace_count());
        const auto& points = density_.trace_points();
        for (int i = 0; i < trace_count(); ++i)
            values[i] = shape_.exact_value(
                points[static_cast<std::size_t>(i)].point);
        return values;
    }

    Eigen::VectorXd exact_normal_trace() const
    {
        Eigen::VectorXd values(trace_count());
        const auto& points = density_.trace_points();
        for (int i = 0; i < trace_count(); ++i) {
            const auto& point = points[static_cast<std::size_t>(i)];
            values[i] = shape_.exact_physical_gradient(point.point).dot(
                point.normal);
        }
        return values;
    }

    Eigen::VectorXd solve_jump_field(
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        bool add_exact_value_jump = false,
        bool add_exact_normal_jump = false) const
    {
        check_coefficients(value_jump, "value jump");
        check_coefficients(normal_jump, "normal jump");
        const bool has_value_jump = !value_jump.isZero(0.0);
        const bool has_normal_jump = !normal_jump.isZero(0.0);
        const Eigen::VectorXd value_c0 = has_value_jump
            ? density_.expand_reduced(value_jump)
            : Eigen::VectorXd();
        const Eigen::VectorXd normal_c0 = has_normal_jump
            ? density_.expand_reduced(normal_jump)
            : Eigen::VectorXd();
        Eigen::VectorXd spread = Eigen::VectorXd::Zero(grid_.num_dofs());
        for (const CrossingPlan3D& plan : crossing_plans_) {
            Eigen::Matrix<double, 6, 1> value_jet =
                Eigen::Matrix<double, 6, 1>::Zero();
            Eigen::Matrix<double, 3, 1> normal_jet =
                Eigen::Matrix<double, 3, 1>::Zero();
            if (has_value_jump)
                value_jet = plan.density.recover_value_jet(value_c0);
            if (has_normal_jump)
                normal_jet = plan.density.recover_normal_jet(normal_c0);
            for (const DirectedCrossingTarget3D& target : plan.targets) {
                spread[target.grid_index] +=
                    target.scale
                    * (target.value_jet_weight.dot(value_jet)
                       + target.normal_jet_weight.dot(normal_jet)
                       + (add_exact_value_jump
                              ? target.exact_value_correction
                              : 0.0)
                       + (add_exact_normal_jump
                              ? target.exact_normal_correction
                              : 0.0));
            }
        }
        Eigen::VectorXd solution;
        // The directed correction rows discretize -Delta.  The bulk API
        // accepts the physical right-hand side of Delta u = rhs.
        bulk_.solve(-spread, solution);
        return solution;
    }

    Eigen::VectorXd solve_exact_cauchy_oracle() const
    {
        Eigen::VectorXd spread = Eigen::VectorXd::Zero(grid_.num_dofs());
        for (const CrossingPlan3D& plan : crossing_plans_)
            for (const DirectedCrossingTarget3D& target : plan.targets)
                spread[target.grid_index] +=
                    target.scale
                    * (target.exact_value_correction
                       + target.exact_normal_correction);
        Eigen::VectorXd solution;
        bulk_.solve(-spread, solution);
        return solution;
    }

    TracePair3D restrict_exterior(
        const Eigen::VectorXd& grid_solution,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump,
        bool add_exact_value_jump = false,
        bool add_exact_normal_jump = false) const
    {
        if (grid_solution.size() != grid_.num_dofs())
            throw std::invalid_argument("Grid solution has the wrong size");
        check_coefficients(value_jump, "value jump");
        check_coefficients(normal_jump, "normal jump");
        const bool has_value_jump = !value_jump.isZero(0.0);
        const bool has_normal_jump = !normal_jump.isZero(0.0);
        const Eigen::VectorXd value_c0 = has_value_jump
            ? density_.expand_reduced(value_jump)
            : Eigen::VectorXd();
        const Eigen::VectorXd normal_c0 = has_normal_jump
            ? density_.expand_reduced(normal_jump)
            : Eigen::VectorXd();
        TracePair3D result;
        result.value.resize(trace_count());
        result.normal.resize(trace_count());
        result.raw_value.resize(trace_count());
        result.raw_normal.resize(trace_count());
        for (int q = 0; q < trace_count(); ++q) {
            const ExteriorTracePlan3D& plan =
                trace_plans_[static_cast<std::size_t>(q)];
            double raw_value = 0.0;
            double raw_normal = 0.0;
            for (const GridTraceWeight3D& entry : plan.grid_weights) {
                raw_value += entry.value_weight * grid_solution[entry.index];
                raw_normal += entry.normal_weight * grid_solution[entry.index];
            }
            result.raw_value[q] = raw_value;
            result.raw_normal[q] = raw_normal;
            Eigen::Matrix<double, 6, 1> value_jet =
                Eigen::Matrix<double, 6, 1>::Zero();
            Eigen::Matrix<double, 3, 1> normal_jet =
                Eigen::Matrix<double, 3, 1>::Zero();
            if (has_value_jump)
                value_jet = plan.density.recover_value_jet(value_c0);
            if (has_normal_jump)
                normal_jet = plan.density.recover_normal_jet(normal_c0);
            result.value[q] =
                raw_value + plan.value_from_value_jet.dot(value_jet)
                + plan.value_from_normal_jet.dot(normal_jet)
                + (add_exact_value_jump
                       ? plan.value_from_exact_value_jump
                       : 0.0)
                + (add_exact_normal_jump
                       ? plan.value_from_exact_normal_jump
                       : 0.0);
            result.normal[q] =
                raw_normal + plan.normal_from_value_jet.dot(value_jet)
                + plan.normal_from_normal_jet.dot(normal_jet)
                + (add_exact_value_jump
                       ? plan.normal_from_exact_value_jump
                       : 0.0)
                + (add_exact_normal_jump
                       ? plan.normal_from_exact_normal_jump
                       : 0.0);
        }
        return result;
    }

    double interior_linf_error(const Eigen::VectorXd& field,
                               double shift = 0.0) const
    {
        double maximum = 0.0;
        for (int index = 0; index < grid_.num_dofs(); ++index) {
            if (!inside_[static_cast<std::size_t>(index)])
                continue;
            const auto coordinate = grid_.coord(index);
            const Eigen::Vector3d p(
                coordinate[0], coordinate[1], coordinate[2]);
            maximum = std::max(
                maximum,
                std::abs(field[index] + shift - shape_.exact_value(p)));
        }
        return maximum;
    }

    double exterior_linf(const Eigen::VectorXd& field) const
    {
        double maximum = 0.0;
        for (int index = 0; index < grid_.num_dofs(); ++index)
            if (!inside_[static_cast<std::size_t>(index)])
                maximum = std::max(maximum, std::abs(field[index]));
        return maximum;
    }

    double optimal_interior_shift(const Eigen::VectorXd& field) const
    {
        double sum = 0.0;
        int count = 0;
        for (int index = 0; index < grid_.num_dofs(); ++index) {
            if (!inside_[static_cast<std::size_t>(index)])
                continue;
            const auto coordinate = grid_.coord(index);
            const Eigen::Vector3d p(
                coordinate[0], coordinate[1], coordinate[2]);
            sum += shape_.exact_value(p) - field[index];
            ++count;
        }
        return sum / static_cast<double>(count);
    }

    double constant_oracle_error(const Eigen::VectorXd& field) const
    {
        double maximum = 0.0;
        for (int index = 0; index < grid_.num_dofs(); ++index) {
            const double target =
                inside_[static_cast<std::size_t>(index)] ? 1.0 : 0.0;
            maximum = std::max(maximum, std::abs(field[index] - target));
        }
        return maximum;
    }

private:
    static double elapsed_seconds(
        const std::chrono::steady_clock::time_point& begin)
    {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - begin)
            .count();
    }

    void check_coefficients(const Eigen::VectorXd& values,
                            const char* label) const
    {
        if (values.size() != coefficient_count())
            throw std::invalid_argument(std::string(label)
                                        + " coefficient vector has the wrong size");
    }

    void classify_grid()
    {
        inside_.resize(static_cast<std::size_t>(grid_.num_dofs()));
        for (int index = 0; index < grid_.num_dofs(); ++index) {
            const auto coordinate = grid_.coord(index);
            inside_[static_cast<std::size_t>(index)] = shape_.inside(
                Eigen::Vector3d(
                    coordinate[0], coordinate[1], coordinate[2]));
        }
    }

    void build_crossing_rows()
    {
        const double inverse_h2 = 1.0 / (h_ * h_);
        const double delta = 0.35 * h_;
        const std::array<Eigen::Vector3i, 3> directions = {
            Eigen::Vector3i(1, 0, 0),
            Eigen::Vector3i(0, 1, 0),
            Eigen::Vector3i(0, 0, 1)};

        for (int iz = 1; iz < intervals_; ++iz) {
            for (int iy = 1; iy < intervals_; ++iy) {
                for (int ix = 1; ix < intervals_; ++ix) {
                    for (const Eigen::Vector3i& direction : directions) {
                        const Eigen::Vector3i a_index(ix, iy, iz);
                        const Eigen::Vector3i b_index = a_index + direction;
                        if (b_index.x() >= intervals_
                            || b_index.y() >= intervals_
                            || b_index.z() >= intervals_)
                            continue;
                        const int ia = grid_.index(
                            a_index.x(), a_index.y(), a_index.z());
                        const int ib = grid_.index(
                            b_index.x(), b_index.y(), b_index.z());
                        const bool inside_a = inside_[static_cast<std::size_t>(ia)];
                        const bool inside_b = inside_[static_cast<std::size_t>(ib)];
                        if (inside_a == inside_b)
                            continue;
                        const auto ca = grid_.coord(ia);
                        const auto cb = grid_.coord(ib);
                        const Eigen::Vector3d pa(ca[0], ca[1], ca[2]);
                        const Eigen::Vector3d pb(cb[0], cb[1], cb[2]);
                        const Eigen::Vector3d q = shape_.crossing(pa, pb);
                        CrossingPlan3D plan;
                        plan.density = build_local_density_jet_plan(
                                q, delta, shape_, density_);
                        const LocalDensityJetPlan3D& jets = plan.density;
                        max_value_condition_ = std::max(
                            max_value_condition_, jets.value_condition);
                        max_normal_condition_ = std::max(
                            max_normal_condition_, jets.normal_condition);
                        // A correction is inserted in the finite-difference
                        // equation at the target endpoint, but its Cauchy
                        // continuation is evaluated at the *opposite* node of
                        // the crossing edge.  This is the directed P -> Q
                        // convention used by the reference KFBI stencil.
                        struct DirectedEndpoint {
                            int target_index;
                            Eigen::Vector3d opposite_point;
                        };
                        const std::array<DirectedEndpoint, 2> endpoints = {
                            DirectedEndpoint{ia, pb},
                            DirectedEndpoint{ib, pa}};
                        for (std::size_t endpoint_index = 0;
                             endpoint_index < endpoints.size();
                             ++endpoint_index) {
                            const DirectedEndpoint& endpoint =
                                endpoints[endpoint_index];
                            const CauchyPolynomialWeights3D weights =
                                cauchy_polynomial_weights_3d(
                                    q,
                                    jets.frame,
                                    jets.graph_hessian,
                                    endpoint.opposite_point);
                            DirectedCrossingTarget3D& target =
                                plan.targets[endpoint_index];
                            target.grid_index = endpoint.target_index;
                            target.scale =
                                (inside_[static_cast<std::size_t>(
                                     endpoint.target_index)]
                                     ? 1.0
                                     : -1.0)
                                * inverse_h2;
                            target.value_jet_weight = weights.w0;
                            target.normal_jet_weight = weights.w1;
                            target.exact_value_correction =
                                weights.apply_value_jet(jets.exact_value_jet);
                            target.exact_normal_correction =
                                weights.apply_normal_jet(
                                    jets.exact_normal_jet);
                        }
                        crossing_plans_.push_back(std::move(plan));
                        ++unique_crossings_;
                    }
                }
            }
        }
        if (crossing_plans_.empty())
            throw std::runtime_error("No Cartesian/surface crossings were found");
    }

    void build_trace_plans()
    {
        const auto& points = density_.trace_points();
        const int nt = static_cast<int>(points.size());
        trace_weights_.resize(nt);
        trace_plans_.reserve(points.size());
        std::vector<Eigen::Triplet<double>> trace_entries;
        trace_entries.reserve(
            static_cast<std::size_t>(nt)
            * CapC0BasisStencil3D{}.indices.size());
        for (int q = 0; q < nt; ++q) {
            const auto& point = points[static_cast<std::size_t>(q)];
            const CapC0BasisStencil3D stencil =
                density_.c0_basis_stencil(point.patch, point.u, point.v);
            for (int entry = 0; entry < stencil.count; ++entry) {
                trace_entries.emplace_back(
                    q,
                    stencil.indices[static_cast<std::size_t>(entry)],
                    stencil.weights[static_cast<std::size_t>(entry)]);
            }
            trace_weights_[q] = point.surface_weight;
            trace_plans_.push_back(build_one_trace_plan(point.point));
        }
        const int final_dirichlet_dofs =
            density_.reduced_coefficient_count();
        if (nt <= final_dirichlet_dofs) {
            std::ostringstream message;
            message
                << "general-cap trace projection must be strictly "
                   "oversampled: trace_points="
                << nt << " final_dirichlet_dofs="
                << final_dirichlet_dofs;
            throw std::runtime_error(message.str());
        }
        trace_c0_design_.resize(nt, density_.c0_coefficient_count());
        trace_c0_design_.setFromTriplets(
            trace_entries.begin(), trace_entries.end());
        trace_c0_design_.makeCompressed();
        projection_ = std::make_unique<ReducedTraceProjection3D>(
            trace_c0_design_,
            density_.reduction_matrix(),
            trace_weights_);
        const Eigen::VectorXd c0_mass =
            trace_c0_design_.transpose() * trace_weights_;
        mass_row_ = density_.reduction_matrix().transpose() * c0_mass;
        constant_coefficients_ =
            projection_->apply(Eigen::VectorXd::Ones(nt));
        const double constant_error =
            (evaluate_trace(constant_coefficients_)
             - Eigen::VectorXd::Ones(nt))
                .lpNorm<Eigen::Infinity>();
        if (constant_error > 2.0e-9)
            throw std::runtime_error(
                "Reduced cap space does not reproduce a constant density");
    }

    ExteriorTracePlan3D build_one_trace_plan(const Eigen::Vector3d& q)
    {
        static const std::array<double, 8> rho = {
            -1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4};
        const auto fit = normal_trace_fit_weights();
        const Eigen::VectorXd& trace_w0 = fit.first;
        const Eigen::VectorXd& trace_w1 = fit.second;
        ExteriorTracePlan3D plan;
        plan.density = build_local_density_jet_plan(
            q, 0.35 * h_, shape_, density_);
        const LocalDensityJetPlan3D& jets = plan.density;
        max_value_condition_ =
            std::max(max_value_condition_, jets.value_condition);
        max_normal_condition_ =
            std::max(max_normal_condition_, jets.normal_condition);

        Eigen::Matrix<double, 1, 6> m0v =
            Eigen::Matrix<double, 1, 6>::Zero();
        Eigen::Matrix<double, 1, 3> m1v =
            Eigen::Matrix<double, 1, 3>::Zero();
        Eigen::Matrix<double, 1, 6> m0n =
            Eigen::Matrix<double, 1, 6>::Zero();
        Eigen::Matrix<double, 1, 3> m1n =
            Eigen::Matrix<double, 1, 3>::Zero();

        plan.grid_weights.reserve(8 * 64);
        for (int layer = 0; layer < 8; ++layer) {
            const Eigen::Vector3d query =
                q + rho[static_cast<std::size_t>(layer)] * h_
                        * jets.frame.normal;
            const Eigen::Vector3d grid_coordinate =
                (query - Eigen::Vector3d::Constant(box_min)) / h_;
            if ((grid_coordinate.array() < 0.0).any()
                || (grid_coordinate.array() > intervals_).any()) {
                std::ostringstream message;
                message << "Exterior trace query left the box: q="
                        << q.transpose() << " query=" << query.transpose()
                        << " grid-coordinate=" << grid_coordinate.transpose()
                        << " N=" << intervals_;
                throw std::runtime_error(message.str());
            }
            const CubicGridStencil1D sx =
                cubic_grid_stencil(grid_coordinate.x(), intervals_);
            const CubicGridStencil1D sy =
                cubic_grid_stencil(grid_coordinate.y(), intervals_);
            const CubicGridStencil1D sz =
                cubic_grid_stencil(grid_coordinate.z(), intervals_);
            for (int oz = 0; oz < 4; ++oz) {
                for (int oy = 0; oy < 4; ++oy) {
                    for (int ox = 0; ox < 4; ++ox) {
                        const int ix = sx.start + ox;
                        const int iy = sy.start + oy;
                        const int iz = sz.start + oz;
                        const double interpolation = sx.weights[ox]
                                                   * sy.weights[oy]
                                                   * sz.weights[oz];
                        const double value_weight =
                            trace_w0[layer] * interpolation;
                        const double normal_weight =
                            trace_w1[layer] * interpolation / h_;
                        const int index = grid_.index(ix, iy, iz);
                        plan.grid_weights.push_back(
                            {index, value_weight, normal_weight});
                        if (!inside_[static_cast<std::size_t>(index)])
                            continue;
                        const auto coordinate = grid_.coord(index);
                        const Eigen::Vector3d node(
                            coordinate[0], coordinate[1], coordinate[2]);
                        const CauchyPolynomialWeights3D weights =
                            cauchy_polynomial_weights_3d(
                                q,
                                jets.frame,
                                jets.graph_hessian,
                                node);
                        m0v.noalias() -= value_weight * weights.w0;
                        m1v.noalias() -= value_weight * weights.w1;
                        m0n.noalias() -= normal_weight * weights.w0;
                        m1n.noalias() -= normal_weight * weights.w1;
                    }
                }
            }
        }
        plan.value_from_value_jet = m0v;
        plan.value_from_normal_jet = m1v;
        plan.normal_from_value_jet = m0n;
        plan.normal_from_normal_jet = m1n;
        plan.value_from_exact_value_jump =
            (m0v * jets.exact_value_jet)(0, 0);
        plan.value_from_exact_normal_jump =
            (m1v * jets.exact_normal_jet)(0, 0);
        plan.normal_from_exact_value_jump =
            (m0n * jets.exact_value_jet)(0, 0);
        plan.normal_from_exact_normal_jump =
            (m1n * jets.exact_normal_jet)(0, 0);
        return plan;
    }

    CapAtlasDensityOptions3D density_options_;
    ShapeModel3D shape_;
    CapAtlasDensitySpace3D density_;
    int intervals_ = 0;
    double h_ = 0.0;
    CartesianGrid3D grid_;
    LaplaceFftBulkSolverZfft3D bulk_;
    std::vector<unsigned char> inside_;
    std::vector<CrossingPlan3D> crossing_plans_;
    int unique_crossings_ = 0;
    std::vector<ExteriorTracePlan3D> trace_plans_;
    Eigen::SparseMatrix<double> trace_c0_design_;
    Eigen::VectorXd trace_weights_;
    std::unique_ptr<ReducedTraceProjection3D> projection_;
    Eigen::VectorXd mass_row_;
    Eigen::VectorXd constant_coefficients_;
    double max_value_condition_ = 0.0;
    double max_normal_condition_ = 0.0;
    double setup_seconds_ = 0.0;
};

class FunctionOperator final : public IKFBIOperator {
public:
    using Function =
        std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;

    FunctionOperator(int size, Function function)
        : size_(size), function_(std::move(function))
    {
    }

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override
    {
        function_(x, y);
    }

    int problem_size() const override { return size_; }

private:
    int size_;
    Function function_;
};

// Historical coefficient-metric diagnostic for the weak-C1 elimination
// basis c=R*a.  The production Dirichlet and Neumann solvers below now use
// the physical trace-mass factor of B^T W B; this class is retained only to
// report the conditioning of the former C0-coefficient coordinate map.
class OrthonormalReducedCoordinates3D {
public:
    explicit OrthonormalReducedCoordinates3D(const Eigen::MatrixXd& reduction)
    {
        if (reduction.rows() < reduction.cols() || reduction.cols() == 0)
            throw std::invalid_argument("Invalid reduced C1 basis");
        size_ = static_cast<int>(reduction.cols());
        if (reduction.cols() > 1000) {
            build_low_rank_transform(reduction);
            return;
        }
        const Eigen::MatrixXd gram = reduction.transpose() * reduction;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(gram);
        if (eigensolver.info() != Eigen::Success)
            throw std::runtime_error(
                "Could not diagonalize the reduced-coordinate Gram matrix");
        eigenvectors_ = eigensolver.eigenvectors();
        const Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
        const double largest = eigenvalues.maxCoeff();
        const double smallest = eigenvalues.minCoeff();
        if (!(smallest > 128.0 * std::numeric_limits<double>::epsilon()
                             * largest))
            throw std::runtime_error(
                "Reduced C1 elimination basis is numerically rank deficient");
        square_roots_ = eigenvalues.array().sqrt();
        inverse_square_roots_ = square_roots_.array().inverse();
        condition_ = std::sqrt(largest / smallest);
    }

    Eigen::VectorXd to_native(const Eigen::VectorXd& orthonormal) const
    {
        if (orthonormal.size() != size_)
            throw std::invalid_argument(
                "Orthonormal reduced vector has the wrong size");
        if (low_rank_) {
            const Eigen::VectorXd projected =
                low_rank_vectors_.transpose() * orthonormal;
            return orthonormal
                   + low_rank_vectors_
                         * (inverse_square_root_updates_.array()
                            * projected.array())
                               .matrix();
        }
        return eigenvectors_
               * (inverse_square_roots_.array() * orthonormal.array()).matrix();
    }

    Eigen::VectorXd from_native(const Eigen::VectorXd& native) const
    {
        if (native.size() != size_)
            throw std::invalid_argument("Native reduced vector has the wrong size");
        if (low_rank_) {
            const Eigen::VectorXd projected =
                low_rank_vectors_.transpose() * native;
            return native
                   + low_rank_vectors_
                         * (square_root_updates_.array()
                            * projected.array())
                               .matrix();
        }
        return (square_roots_.array()
                * (eigenvectors_.transpose() * native).array())
            .matrix();
    }

    double condition() const { return condition_; }

private:
    void build_low_rank_transform(const Eigen::MatrixXd& reduction)
    {
        const double scale = std::max(1.0, reduction.cwiseAbs().maxCoeff());
        const double tolerance =
            128.0 * std::numeric_limits<double>::epsilon() * scale;
        std::vector<unsigned char> identity_columns(
            static_cast<std::size_t>(size_), 0);
        std::vector<int> remainder_rows;
        remainder_rows.reserve(
            static_cast<std::size_t>(reduction.rows() - reduction.cols()));
        for (Eigen::Index row = 0; row < reduction.rows(); ++row) {
            int active_count = 0;
            int active_column = -1;
            double active_value = 0.0;
            for (Eigen::Index column = 0; column < reduction.cols(); ++column) {
                const double value = reduction(row, column);
                if (std::abs(value) <= tolerance)
                    continue;
                ++active_count;
                active_column = static_cast<int>(column);
                active_value = value;
                if (active_count > 1)
                    break;
            }
            const bool unused_identity =
                active_count == 1
                && std::abs(active_value - 1.0) <= tolerance
                && !identity_columns[static_cast<std::size_t>(active_column)];
            if (unused_identity) {
                identity_columns[static_cast<std::size_t>(active_column)] = 1;
            } else {
                remainder_rows.push_back(static_cast<int>(row));
            }
        }
        if (std::find(identity_columns.begin(), identity_columns.end(), 0)
                != identity_columns.end()
            || remainder_rows.size()
                   != static_cast<std::size_t>(
                       reduction.rows() - reduction.cols())) {
            throw std::runtime_error(
                "Large reduced basis does not expose its identity free rows");
        }

        Eigen::MatrixXd eliminated(
            static_cast<Eigen::Index>(remainder_rows.size()),
            reduction.cols());
        for (Eigen::Index row = 0; row < eliminated.rows(); ++row)
            eliminated.row(row) = reduction.row(
                remainder_rows[static_cast<std::size_t>(row)]);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            eliminated, Eigen::ComputeThinV);
        const Eigen::VectorXd singular_values = svd.singularValues();
        if (singular_values.size() == 0 || !singular_values.allFinite())
            throw std::runtime_error(
                "Could not factor the low-rank coordinate Gram update");
        const double largest = singular_values[0] * singular_values[0];
        const double singular_tolerance =
            std::sqrt(256.0 * std::numeric_limits<double>::epsilon()
                      * std::max(1.0, largest));
        int positive_rank = 0;
        for (Eigen::Index i = 0; i < singular_values.size(); ++i)
            if (singular_values[i] > singular_tolerance)
                ++positive_rank;
        low_rank_vectors_ = svd.matrixV().leftCols(positive_rank);
        square_root_updates_.resize(positive_rank);
        inverse_square_root_updates_.resize(positive_rank);
        for (int column = 0; column < positive_rank; ++column) {
            const double sigma = singular_values[column];
            const double mu = sigma * sigma;
            const double root = std::sqrt(1.0 + mu);
            square_root_updates_[column] = root - 1.0;
            inverse_square_root_updates_[column] = 1.0 / root - 1.0;
        }
        const double orthogonality_error =
            positive_rank == 0
                ? 0.0
                : (low_rank_vectors_.transpose() * low_rank_vectors_
                   - Eigen::MatrixXd::Identity(positive_rank, positive_rank))
                      .cwiseAbs()
                      .maxCoeff();
        if (orthogonality_error > 2.0e-8)
            throw std::runtime_error(
                "Low-rank coordinate eigenvectors lost orthogonality");
        condition_ = std::sqrt(1.0 + largest);
        low_rank_ = true;
    }

    Eigen::MatrixXd eigenvectors_;
    Eigen::VectorXd square_roots_;
    Eigen::VectorXd inverse_square_roots_;
    Eigen::MatrixXd low_rank_vectors_;
    Eigen::VectorXd square_root_updates_;
    Eigen::VectorXd inverse_square_root_updates_;
    int size_ = 0;
    bool low_rank_ = false;
    double condition_ = 0.0;
};

struct SolveMetrics3D {
    std::string shape;
    std::string pose;
    int gmres_max_iterations = 80;
    double gmres_tolerance = 2.0e-10;
    int intervals = 0;
    int coefficients_per_direction = 0;
    bool automatic_density = false;
    int automatic_coefficients_per_direction = 0;
    double orientation_integral = 0.0;
    double continuous_density_elements = 0.0;
    double predicted_crossings = 0.0;
    double predicted_crossings_per_density_cell = 0.0;
    int reduced_dofs = 0;
    int trace_points = 0;
    int crossings = 0;
    double h = 0.0;
    double setup_seconds = 0.0;
    double solve_seconds = 0.0;
    double projection_pb_error = 0.0;
    double projection_condition = 0.0;
    double elimination_basis_condition = 0.0;
    double value_projection_error = 0.0;
    double normal_projection_error = 0.0;
    double value_c1_jump = 0.0;
    double normal_c1_jump = 0.0;
    double c1_reduction_residual = 0.0;
    double value_jet_condition = 0.0;
    double normal_jet_condition = 0.0;
    double constant_oracle_error = 0.0;
    double exact_cauchy_oracle_error = 0.0;
    double projected_oracle_error = 0.0;
    int neumann_iterations = 0;
    bool neumann_converged = false;
    double neumann_residual = 0.0;
    double neumann_interior_error = 0.0;
    double neumann_exterior_trace = 0.0;
    double neumann_projected_trace = 0.0;
    double neumann_raw_trace = 0.0;
    double neumann_density_error = 0.0;
    double neumann_iteration_density_error = 0.0;
    double neumann_shift = 0.0;
    double neumann_flux_correction = 0.0;
    double neumann_compatibility_residual = 0.0;
    double neumann_closure_residual = 0.0;
    double neumann_operator_residual = 0.0;
    double neumann_full_projected_trace = 0.0;
    int neumann_pre_mean_dofs = 0;
    int neumann_final_dofs = 0;
    int neumann_mean_pivot = -1;
    double neumann_mean_pivot_moment = 0.0;
    double neumann_mean_relative_observability = 0.0;
    double neumann_mean_maximum_multiplier = 0.0;
    double neumann_mean_coordinate_condition = 0.0;
    double neumann_mean_orthonormality_residual = 0.0;
    double neumann_mean_orthogonality_residual = 0.0;
    double neumann_constant_projection_linf = 0.0;
    double neumann_density_mean = 0.0;
    double neumann_mean_direction_agreement = 0.0;
    int dirichlet_iterations = 0;
    bool dirichlet_converged = false;
    double dirichlet_residual = 0.0;
    double dirichlet_interior_error = 0.0;
    double dirichlet_exterior_trace = 0.0;
    double dirichlet_projected_trace = 0.0;
    double dirichlet_raw_trace = 0.0;
    double dirichlet_density_error = 0.0;
    double dirichlet_iteration_density_error = 0.0;
};

double infinity_norm(const Eigen::VectorXd& values)
{
    return values.size() == 0 ? 0.0 : values.lpNorm<Eigen::Infinity>();
}

SolveMetrics3D run_case(const ShapeSpecification3D& shape,
                        const RigidPose3D& pose,
                        int intervals,
                        int coefficients_per_direction,
                        const DensityResolutionChoice3D& density_choice,
                        bool automatic_density,
                        SolveSelection3D solve_selection,
                        int gmres_max_iterations,
                        double gmres_tolerance)
{
    const std::string& shape_name = shape.name;
    std::cout << "\n[build] shape=" << shape_name << " N=" << intervals
              << " pose=" << pose.id
              << " ncoef=" << coefficients_per_direction
              << (automatic_density ? " (automatic)" : " (override)")
              << '\n'
              << std::scientific << std::setprecision(6)
              << "[density-init] integral|n|_1="
              << density_choice.orientation_integral
              << " predicted-crossings="
              << density_choice.predicted_crossings
              << " continuous-elements="
              << density_choice.continuous_elements_per_direction
              << " selected-elements="
              << density_choice.elements_per_direction
              << " target-crossings/cell="
              << target_crossings_per_density_cell << std::endl;
    CoefficientKfbiPipeline3D pipeline(
        shape, pose, intervals, coefficients_per_direction);
    const int k = pipeline.coefficient_count();
    const bool solve_neumann =
        solve_selection != SolveSelection3D::DirichletNormalOnly;
    const bool solve_dirichlet =
        solve_selection != SolveSelection3D::NeumannOnly;
    std::unique_ptr<OrthonormalReducedCoordinates3D>
        coefficient_metric_diagnostic;
    if (solve_dirichlet) {
        coefficient_metric_diagnostic =
            std::make_unique<OrthonormalReducedCoordinates3D>(
                pipeline.density().reduction_matrix());
    }
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(k);
    const Eigen::VectorXd exact_value_samples = pipeline.exact_value_trace();
    const Eigen::VectorXd exact_normal_samples = pipeline.exact_normal_trace();
    Eigen::VectorXd exact_value_coefficients =
        pipeline.project_trace(exact_value_samples);
    Eigen::VectorXd exact_normal_coefficients =
        pipeline.project_trace(exact_normal_samples);
    const Eigen::VectorXd& mass = pipeline.mass_row();
    const Eigen::VectorXd& constant = pipeline.constant_coefficients();
    const double area = pipeline.area();

    const double flux_before = mass.dot(exact_normal_coefficients);
    exact_normal_coefficients -=
        constant * (flux_before / mass.dot(constant));
    Eigen::VectorXd mean_zero_value_coefficients = exact_value_coefficients;
    mean_zero_value_coefficients -=
        constant * (mass.dot(mean_zero_value_coefficients) / area);

    SolveMetrics3D metrics;
    metrics.shape = shape_name;
    metrics.pose = pose.id;
    metrics.gmres_max_iterations = gmres_max_iterations;
    metrics.gmres_tolerance = gmres_tolerance;
    metrics.intervals = intervals;
    metrics.coefficients_per_direction = coefficients_per_direction;
    metrics.automatic_density = automatic_density;
    metrics.automatic_coefficients_per_direction =
        density_choice.coefficients_per_direction;
    metrics.orientation_integral = density_choice.orientation_integral;
    metrics.continuous_density_elements =
        density_choice.continuous_elements_per_direction;
    metrics.predicted_crossings = density_choice.predicted_crossings;
    const int actual_elements = coefficients_per_direction - 3;
    metrics.predicted_crossings_per_density_cell =
        density_choice.predicted_crossings
        / (static_cast<double>(density_choice.patch_count)
           * actual_elements * actual_elements);
    metrics.reduced_dofs = k;
    metrics.trace_points = pipeline.trace_count();
    metrics.crossings = pipeline.unique_crossing_count();
    metrics.h = pipeline.spacing();
    metrics.setup_seconds = pipeline.setup_seconds();
    metrics.projection_pb_error = pipeline.projection().pb_max_error();
    metrics.projection_condition = pipeline.projection().condition();
    metrics.elimination_basis_condition = coefficient_metric_diagnostic
        ? coefficient_metric_diagnostic->condition()
        : std::numeric_limits<double>::quiet_NaN();
    metrics.value_projection_error = infinity_norm(
        pipeline.evaluate_trace(exact_value_coefficients)
        - exact_value_samples);
    metrics.normal_projection_error = infinity_norm(
        pipeline.evaluate_trace(exact_normal_coefficients)
        - exact_normal_samples);
    metrics.value_c1_jump = pipeline.density().max_physical_conormal_jump(
        exact_value_coefficients);
    metrics.normal_c1_jump = pipeline.density().max_physical_conormal_jump(
        exact_normal_coefficients);
    metrics.c1_reduction_residual =
        pipeline.density().reduction_constraint_residual();
    metrics.value_jet_condition =
        pipeline.maximum_value_recovery_condition();
    metrics.normal_jet_condition =
        pipeline.maximum_normal_recovery_condition();
    metrics.neumann_flux_correction = std::abs(flux_before) / area;

    const Eigen::VectorXd constant_field =
        pipeline.solve_jump_field(constant, zero);
    metrics.constant_oracle_error =
        pipeline.constant_oracle_error(constant_field);
    const Eigen::VectorXd exact_oracle = pipeline.solve_exact_cauchy_oracle();
    metrics.exact_cauchy_oracle_error =
        pipeline.interior_linf_error(exact_oracle);
    const Eigen::VectorXd projected_oracle = pipeline.solve_jump_field(
        exact_value_coefficients, exact_normal_coefficients);
    metrics.projected_oracle_error =
        pipeline.interior_linf_error(projected_oracle);

    const auto solve_begin = std::chrono::steady_clock::now();
    const ReducedTraceProjection3D& projection = pipeline.projection();
    const auto trace_mass_from_native =
        [&](const Eigen::VectorXd& native) {
            return projection.native_to_trace_mass(native);
        };
    const auto native_from_trace_mass =
        [&](const Eigen::VectorXd& trace_mass) {
            return projection.trace_mass_to_native(trace_mass);
        };
    const auto project_to_trace_mass =
        [&](const Eigen::VectorXd& trace) {
            return trace_mass_from_native(pipeline.project_trace(trace));
        };
    if (solve_neumann) {
        // In full trace-mass coordinates z=U*a, the normalized surface mean
        // is d^T z with d=U^{-T}m/area.  Eliminate one maximum-moment
        // coordinate and orthonormalize that pivot map implicitly.  The
        // resulting Q has K-1 columns, Q^T Q=I, and d^T Q=0.
        const Eigen::VectorXd mean_dual =
            projection.native_dual_to_trace_mass(mass / area);
        const MeanFreeTraceMassCoordinates3D mean_free(mean_dual);
        const int neumann_size = static_cast<int>(
            mean_free.reduced_coordinate_count());
        const auto project_to_mean_free =
            [&](const Eigen::VectorXd& trace) {
                return mean_free.restrict(project_to_trace_mass(trace));
            };
        const auto apply_trace_block =
            [&](const Eigen::VectorXd& mean_free_value_jump,
                Eigen::VectorXd& mean_free_image) {
                const Eigen::VectorXd trace_mass_value_jump =
                    mean_free.lift(mean_free_value_jump);
                const Eigen::VectorXd value_jump =
                    native_from_trace_mass(trace_mass_value_jump);
                const Eigen::VectorXd field =
                    pipeline.solve_jump_field(value_jump, zero);
                const TracePair3D trace =
                    pipeline.restrict_exterior(field, value_jump, zero);
                mean_free_image = project_to_mean_free(trace.value);
            };

        const Eigen::VectorXd known_neumann_field =
            pipeline.solve_jump_field(zero, zero, false, true);
        const TracePair3D known_neumann_trace = pipeline.restrict_exterior(
            known_neumann_field, zero, zero, false, true);
        const Eigen::VectorXd rhs_neumann =
            -project_to_mean_free(known_neumann_trace.value);
        FunctionOperator mean_free_neumann_operator(
            neumann_size,
            [&](const Eigen::VectorXd& y, Eigen::VectorXd& image) {
                apply_trace_block(y, image);
            });
        Eigen::VectorXd y = Eigen::VectorXd::Zero(neumann_size);
        GMRES neumann_gmres(gmres_max_iterations, gmres_tolerance, 0);
        metrics.neumann_iterations = neumann_gmres.solve(
            mean_free_neumann_operator, rhs_neumann, y);
        metrics.neumann_converged = neumann_gmres.converged();
        metrics.neumann_residual = neumann_gmres.residuals().empty()
            ? std::numeric_limits<double>::quiet_NaN()
            : neumann_gmres.residuals().back();
        const Eigen::VectorXd neumann_trace_mass_coefficients =
            mean_free.lift(y);
        const Eigen::VectorXd neumann_value_coefficients =
            native_from_trace_mass(neumann_trace_mass_coefficients);
        const Eigen::VectorXd neumann_field = pipeline.solve_jump_field(
            neumann_value_coefficients, zero, false, true);
        const TracePair3D neumann_trace = pipeline.restrict_exterior(
            neumann_field,
            neumann_value_coefficients,
            zero,
            false,
            true);
        const Eigen::VectorXd final_projected_residual =
            project_to_mean_free(neumann_trace.value);
        const Eigen::VectorXd mean_free_projected_native =
            native_from_trace_mass(
                mean_free.lift(final_projected_residual));
        metrics.neumann_operator_residual =
            infinity_norm(final_projected_residual);
        metrics.neumann_density_mean =
            mass.dot(neumann_value_coefficients) / area;
        metrics.neumann_compatibility_residual =
            std::abs(metrics.neumann_density_mean);
        metrics.neumann_closure_residual = std::max(
            metrics.neumann_operator_residual,
            metrics.neumann_compatibility_residual);
        metrics.neumann_pre_mean_dofs = k;
        metrics.neumann_final_dofs = neumann_size;
        const auto& mean_diagnostics = mean_free.diagnostics();
        metrics.neumann_mean_pivot = static_cast<int>(
            mean_diagnostics.pivot_coordinate);
        metrics.neumann_mean_pivot_moment =
            mean_diagnostics.pivot_moment;
        metrics.neumann_mean_relative_observability =
            mean_diagnostics.relative_observability;
        metrics.neumann_mean_maximum_multiplier =
            mean_diagnostics.maximum_elimination_multiplier;
        metrics.neumann_mean_coordinate_condition =
            mean_diagnostics.coordinate_map_condition;
        metrics.neumann_mean_orthonormality_residual =
            mean_diagnostics.orthonormality_residual;
        metrics.neumann_mean_orthogonality_residual =
            mean_diagnostics.mean_orthogonality_residual;
        const Eigen::VectorXd constant_trace_mass =
            project_to_trace_mass(
                Eigen::VectorXd::Ones(pipeline.trace_count()));
        metrics.neumann_constant_projection_linf = infinity_norm(
            mean_free.restrict(constant_trace_mass));
        const Eigen::VectorXd constant_mean_direction =
            trace_mass_from_native(constant) / area;
        metrics.neumann_mean_direction_agreement = infinity_norm(
            mean_dual - constant_mean_direction);
        metrics.neumann_shift = pipeline.optimal_interior_shift(neumann_field);
        metrics.neumann_interior_error = pipeline.interior_linf_error(
            neumann_field, metrics.neumann_shift);
        metrics.neumann_exterior_trace = infinity_norm(neumann_trace.value);
        metrics.neumann_projected_trace = infinity_norm(
            pipeline.evaluate_trace(mean_free_projected_native));
        metrics.neumann_full_projected_trace = infinity_norm(
            pipeline.evaluate_trace(
                pipeline.project_trace(neumann_trace.value)));
        metrics.neumann_raw_trace = infinity_norm(neumann_trace.raw_value);
        metrics.neumann_iteration_density_error = infinity_norm(
            pipeline.evaluate_trace(
                neumann_value_coefficients - mean_zero_value_coefficients));
        const double exact_surface_mean =
            pipeline.trace_weights().dot(exact_value_samples) / area;
        metrics.neumann_density_error = infinity_norm(
            pipeline.evaluate_trace(neumann_value_coefficients)
            - (exact_value_samples
               - Eigen::VectorXd::Constant(
                   exact_value_samples.size(), exact_surface_mean)));
    }

    if (solve_dirichlet) {
        const Eigen::VectorXd known_dirichlet_field =
            pipeline.solve_jump_field(zero, zero, true, false);
        const TracePair3D known_dirichlet_trace = pipeline.restrict_exterior(
            known_dirichlet_field, zero, zero, true, false);
        const Eigen::VectorXd rhs_dirichlet =
            -project_to_trace_mass(known_dirichlet_trace.normal);
        FunctionOperator dirichlet_operator(
            k,
            [&](const Eigen::VectorXd& trace_mass_normal_jump,
                Eigen::VectorXd& y) {
                const Eigen::VectorXd normal_jump =
                    native_from_trace_mass(trace_mass_normal_jump);
                const Eigen::VectorXd field =
                    pipeline.solve_jump_field(zero, normal_jump);
                const TracePair3D trace =
                    pipeline.restrict_exterior(field, zero, normal_jump);
                y = project_to_trace_mass(trace.normal);
            });
        GMRES dirichlet_gmres(gmres_max_iterations, gmres_tolerance, 0);
        Eigen::VectorXd dirichlet_normal_trace_mass =
            Eigen::VectorXd::Zero(k);
        metrics.dirichlet_iterations = dirichlet_gmres.solve(
            dirichlet_operator,
            rhs_dirichlet,
            dirichlet_normal_trace_mass);
        metrics.dirichlet_converged = dirichlet_gmres.converged();
        metrics.dirichlet_residual = dirichlet_gmres.residuals().empty()
            ? std::numeric_limits<double>::quiet_NaN()
            : dirichlet_gmres.residuals().back();
        const Eigen::VectorXd dirichlet_normal_coefficients =
            native_from_trace_mass(dirichlet_normal_trace_mass);
        const Eigen::VectorXd dirichlet_field = pipeline.solve_jump_field(
            zero, dirichlet_normal_coefficients, true, false);
        const TracePair3D dirichlet_trace = pipeline.restrict_exterior(
            dirichlet_field,
            zero,
            dirichlet_normal_coefficients,
            true,
            false);
        metrics.dirichlet_interior_error =
            pipeline.interior_linf_error(dirichlet_field);
        metrics.dirichlet_exterior_trace = infinity_norm(
            dirichlet_trace.normal);
        metrics.dirichlet_projected_trace = infinity_norm(
            pipeline.evaluate_trace(
                pipeline.project_trace(dirichlet_trace.normal)));
        metrics.dirichlet_raw_trace = infinity_norm(
            dirichlet_trace.raw_normal);
        metrics.dirichlet_iteration_density_error = infinity_norm(
            pipeline.evaluate_trace(
                dirichlet_normal_coefficients - exact_normal_coefficients));
        metrics.dirichlet_density_error = infinity_norm(
            pipeline.evaluate_trace(dirichlet_normal_coefficients)
            - exact_normal_samples);
    }
    metrics.solve_seconds = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - solve_begin)
                                .count();

    std::cout << std::scientific << std::setprecision(4)
              << "[algebra] K=" << k
              << " trace=" << pipeline.trace_count()
              << " crossings=" << pipeline.unique_crossing_count()
              << " PB=" << metrics.projection_pb_error
              << " C*R=" << metrics.c1_reduction_residual
              << " cond(B)=" << metrics.projection_condition
              << " cond(R)=" << metrics.elimination_basis_condition << '\n'
              << "[density-coverage] predicted-crossings/cell="
              << metrics.predicted_crossings_per_density_cell
              << " actual-crossings/cell="
              << static_cast<double>(metrics.crossings)
                     / (static_cast<double>(density_choice.patch_count)
                        * (metrics.coefficients_per_direction - 3)
                        * (metrics.coefficients_per_direction - 3))
              << '\n'
              << "[density-fit] value=" << metrics.value_projection_error
              << " normal=" << metrics.normal_projection_error
              << " C1(value)=" << metrics.value_c1_jump
              << " C1(normal)=" << metrics.normal_c1_jump
              << " C1(value)/h=" << metrics.value_c1_jump / metrics.h
              << " C1(normal)/h=" << metrics.normal_c1_jump / metrics.h
              << '\n'
              << "[oracle] constant=" << metrics.constant_oracle_error
              << " exact-Cauchy=" << metrics.exact_cauchy_oracle_error
              << " projected=" << metrics.projected_oracle_error << '\n';
    if (solve_neumann) {
        std::cout << "[Neumann] coordinates=trace_mass"
                  << " mean_solver=mean_free_pivot_elimination"
                  << " dofs=" << metrics.neumann_pre_mean_dofs
                  << "->" << metrics.neumann_final_dofs
                  << " pivot=" << metrics.neumann_mean_pivot
                  << " pivot_moment="
                  << metrics.neumann_mean_pivot_moment
                  << " relative_observability="
                  << metrics.neumann_mean_relative_observability
                  << " max_multiplier="
                  << metrics.neumann_mean_maximum_multiplier
                  << " coordinate_condition="
                  << metrics.neumann_mean_coordinate_condition
                  << " it=" << metrics.neumann_iterations
                  << " conv=" << metrics.neumann_converged
                  << " residual=" << metrics.neumann_residual
                  << " operator=" << metrics.neumann_operator_residual
                  << " closure=" << metrics.neumann_closure_residual
                  << " compatibility="
                  << metrics.neumann_compatibility_residual
                  << " P1="
                  << metrics.neumann_constant_projection_linf
                  << " density-mean=" << metrics.neumann_density_mean
                  << " interior=" << metrics.neumann_interior_error
                  << " exterior-trace=" << metrics.neumann_exterior_trace
                  << " projected-trace=" << metrics.neumann_projected_trace
                  << " density-total=" << metrics.neumann_density_error
                  << " density-iteration="
                  << metrics.neumann_iteration_density_error << '\n';
    }
    if (solve_dirichlet) {
        std::cout << "[Dirichlet] coordinates=trace_mass"
                  << " it=" << metrics.dirichlet_iterations
                  << " conv=" << metrics.dirichlet_converged
                  << " residual=" << metrics.dirichlet_residual
                  << " interior=" << metrics.dirichlet_interior_error
                  << " exterior-normal=" << metrics.dirichlet_exterior_trace
                  << " projected-trace=" << metrics.dirichlet_projected_trace
                  << " density-total=" << metrics.dirichlet_density_error
                  << " density-iteration="
                  << metrics.dirichlet_iteration_density_error << '\n';
    }
    return metrics;
}

double observed_order(double coarse_error,
                      double fine_error,
                      double coarse_h,
                      double fine_h)
{
    if (!(coarse_error > 0.0) || !(fine_error > 0.0))
        return std::numeric_limits<double>::quiet_NaN();
    if (std::abs(coarse_h - fine_h)
        <= 32.0 * std::numeric_limits<double>::epsilon()
               * std::max(std::abs(coarse_h), std::abs(fine_h)))
        return std::numeric_limits<double>::quiet_NaN();
    return std::log(coarse_error / fine_error)
           / std::log(coarse_h / fine_h);
}

void write_results(const std::vector<SolveMetrics3D>& results,
                   const RigidPose3D& pose,
                   SolveSelection3D solve_selection)
{
    const std::filesystem::path directory =
        std::filesystem::path(KFBIM_APP_OUTPUT_DIR)
        / "kfbi_general_cap_exterior_trace_3d"
        / ("pose_" + pose.id)
        / solve_selection_name(solve_selection);
    std::filesystem::create_directories(directory);
    const std::filesystem::path file = directory / "refinement.csv";
    std::ofstream out(file);
    if (!out)
        throw std::runtime_error("Could not open numerical result CSV");
    out << "shape,pose,solve_selection,N,h,ncoef,automatic_density,"
           "automatic_ncoef,gmres_max_iterations,gmres_tolerance,"
           "orientation_integral,continuous_density_elements,"
           "predicted_crossings,predicted_crossings_per_density_cell,"
           "reduced_dofs,trace_points,crossings,setup_seconds,"
           "solve_seconds,PB_error,projection_condition,elimination_basis_condition,"
           "value_projection_error,"
           "normal_projection_error,value_C1_jump,normal_C1_jump,"
           "value_C1_jump_over_h,normal_C1_jump_over_h,"
           "C1_reduction_residual,value_jet_condition,"
           "normal_jet_condition,constant_oracle,exact_cauchy_oracle,"
           "exact_cauchy_order,projected_oracle,neumann_iterations,"
           "neumann_converged,neumann_residual,neumann_interior,neumann_order,"
           "neumann_density_coordinates,neumann_mean_solver,"
           "neumann_pre_mean_dofs,neumann_final_dofs,neumann_mean_pivot,"
           "neumann_mean_pivot_moment,neumann_mean_relative_observability,"
           "neumann_mean_maximum_multiplier,"
           "neumann_mean_coordinate_condition,"
           "neumann_mean_orthonormality_residual,"
           "neumann_mean_orthogonality_residual,neumann_P1_linf,"
           "neumann_density_mean,neumann_mean_direction_agreement,"
           "neumann_compatibility_residual,neumann_operator_residual,"
           "neumann_closure_residual,neumann_exterior_trace,"
           "neumann_projected_trace,neumann_full_projected_trace,"
           "neumann_raw_trace,"
           "neumann_density,neumann_iteration_density,neumann_shift,"
           "neumann_flux_correction,"
           "dirichlet_density_coordinates,dirichlet_iterations,"
           "dirichlet_converged,dirichlet_residual,"
           "dirichlet_interior,dirichlet_order,dirichlet_exterior_normal,"
           "dirichlet_projected_trace,dirichlet_raw_trace,dirichlet_density,"
           "dirichlet_iteration_density\n";
    out << std::scientific << std::setprecision(16);
    for (std::size_t i = 0; i < results.size(); ++i) {
        const SolveMetrics3D& r = results[i];
        double oracle_order = std::numeric_limits<double>::quiet_NaN();
        double neumann_order = std::numeric_limits<double>::quiet_NaN();
        double dirichlet_order = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t j = i; j-- > 0;) {
            if (results[j].shape != r.shape || results[j].pose != r.pose)
                continue;
            oracle_order = observed_order(
                results[j].exact_cauchy_oracle_error,
                r.exact_cauchy_oracle_error,
                results[j].h,
                r.h);
            neumann_order = observed_order(
                results[j].neumann_interior_error,
                r.neumann_interior_error,
                results[j].h,
                r.h);
            dirichlet_order = observed_order(
                results[j].dirichlet_interior_error,
                r.dirichlet_interior_error,
                results[j].h,
                r.h);
            break;
        }
        out << r.shape << ',' << r.pose << ','
            << solve_selection_name(solve_selection) << ','
            << r.intervals << ',' << r.h << ','
            << r.coefficients_per_direction << ','
            << (r.automatic_density ? 1 : 0) << ','
            << r.automatic_coefficients_per_direction << ','
            << r.gmres_max_iterations << ',' << r.gmres_tolerance << ','
            << r.orientation_integral << ','
            << r.continuous_density_elements << ','
            << r.predicted_crossings << ','
            << r.predicted_crossings_per_density_cell << ','
            << r.reduced_dofs << ','
            << r.trace_points << ',' << r.crossings << ',' << r.setup_seconds
            << ',' << r.solve_seconds << ',' << r.projection_pb_error << ','
            << r.projection_condition << ','
            << r.elimination_basis_condition << ','
            << r.value_projection_error << ','
            << r.normal_projection_error << ',' << r.value_c1_jump << ','
            << r.normal_c1_jump << ',' << r.value_c1_jump / r.h << ','
            << r.normal_c1_jump / r.h << ','
            << r.c1_reduction_residual << ','
            << r.value_jet_condition << ','
            << r.normal_jet_condition << ',' << r.constant_oracle_error << ','
            << r.exact_cauchy_oracle_error << ',' << oracle_order << ','
            << r.projected_oracle_error << ',' << r.neumann_iterations << ','
            << (r.neumann_converged ? 1 : 0) << ',' << r.neumann_residual << ','
            << r.neumann_interior_error << ',' << neumann_order << ','
            << "trace_mass,mean_free_pivot_elimination,"
            << r.neumann_pre_mean_dofs << ','
            << r.neumann_final_dofs << ','
            << r.neumann_mean_pivot << ','
            << r.neumann_mean_pivot_moment << ','
            << r.neumann_mean_relative_observability << ','
            << r.neumann_mean_maximum_multiplier << ','
            << r.neumann_mean_coordinate_condition << ','
            << r.neumann_mean_orthonormality_residual << ','
            << r.neumann_mean_orthogonality_residual << ','
            << r.neumann_constant_projection_linf << ','
            << r.neumann_density_mean << ','
            << r.neumann_mean_direction_agreement << ','
            << r.neumann_compatibility_residual << ','
            << r.neumann_operator_residual << ','
            << r.neumann_closure_residual << ','
            << r.neumann_exterior_trace << ',' << r.neumann_projected_trace
            << ',' << r.neumann_full_projected_trace
            << ',' << r.neumann_raw_trace << ',' << r.neumann_density_error
            << ',' << r.neumann_iteration_density_error << ','
            << r.neumann_shift << ',' << r.neumann_flux_correction << ','
            << "trace_mass," << r.dirichlet_iterations << ','
            << (r.dirichlet_converged ? 1 : 0) << ','
            << r.dirichlet_residual << ',' << r.dirichlet_interior_error << ','
            << dirichlet_order << ',' << r.dirichlet_exterior_trace << ','
            << r.dirichlet_projected_trace << ',' << r.dirichlet_raw_trace
            << ',' << r.dirichlet_density_error << ','
            << r.dirichlet_iteration_density_error << '\n';
    }
    std::cout << "\n[result] " << file.string() << std::endl;
}

struct LevelSpecification3D {
    int intervals = 0;
    // Zero means geometry/grid-driven automatic initialization.  A positive
    // value is an explicit N:ncoef diagnostic override.
    int coefficients_per_direction = 0;
};

std::vector<LevelSpecification3D> parse_levels(
    int argc,
    char** argv,
    int first_level_argument)
{
    std::vector<LevelSpecification3D> levels;
    for (int i = first_level_argument; i < argc; ++i) {
        const std::string token = argv[i];
        const std::size_t separator = token.find(':');
        const int intervals = std::stoi(token.substr(0, separator));
        const int ncoef = separator == std::string::npos
                              ? 0
                              : std::stoi(token.substr(separator + 1));
        if (intervals < 8)
            throw std::invalid_argument("Every grid level must be at least 8");
        if (ncoef != 0 && ncoef < 4)
            throw std::invalid_argument(
                "Every cubic density direction needs at least 4 coefficients");
        levels.push_back({intervals, ncoef});
    }
    if (levels.empty())
        // N=128 is accepted explicitly, but is not a safe dense
        // default until the trace/crossing caches are made sparse/low-rank.
        levels = {{32, 0}, {64, 0}};
    return levels;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        std::string selector = "all";
        int first_level_argument = 1;
        if (argc > 1) {
            const std::string candidate = argv[1];
            if (candidate == "sphere" || candidate == "ellipsoid"
                || candidate == "flower" || candidate == "all") {
                selector = candidate;
                first_level_argument = 2;
            }
        }
        const std::vector<LevelSpecification3D> levels =
            parse_levels(argc, argv, first_level_argument);
        const RigidPose3D pose = selected_rigid_pose();
        const SolveSelection3D solve_selection = selected_solve_selection();
        const int gmres_max_iterations = positive_integer_environment(
            "KFBIM_3D_GMRES_MAX_ITERATIONS", 80);
        const double gmres_tolerance = positive_double_environment(
            "KFBIM_3D_GMRES_TOLERANCE", 2.0e-10);
        std::vector<ShapeSpecification3D> shapes;
        if (selector == "all" || selector == "sphere")
            shapes.push_back(shape_specification("sphere"));
        if (selector == "all" || selector == "ellipsoid")
            shapes.push_back(shape_specification("ellipsoid"));
        if (selector == "all" || selector == "flower")
            shapes.push_back(shape_specification("flower"));
        std::cout << "KFBI3D general-cap coefficient study\n"
                  << "  pose=" << pose.id << '\n'
                  << "  solve_selection="
                  << solve_selection_name(solve_selection) << '\n'
                  << "  neumann_density_coordinates=trace_mass\n"
                  << "  neumann_mean_solver="
                     "mean_free_pivot_elimination\n"
                  << "  dirichlet_density_coordinates=trace_mass\n"
                  << "  gmres_tolerance=" << std::scientific
                  << gmres_tolerance
                  << " max_iterations=" << gmres_max_iterations << '\n';
        std::vector<SolveMetrics3D> results;
        for (const ShapeSpecification3D& shape : shapes) {
            for (const LevelSpecification3D& level : levels) {
                const DensityResolutionChoice3D density_choice =
                    choose_density_resolution(shape, pose, level.intervals);
                const bool automatic_density =
                    level.coefficients_per_direction == 0;
                const int coefficients_per_direction = automatic_density
                    ? density_choice.coefficients_per_direction
                    : level.coefficients_per_direction;
                results.push_back(run_case(
                    shape,
                    pose,
                    level.intervals,
                    coefficients_per_direction,
                    density_choice,
                    automatic_density,
                    solve_selection,
                    gmres_max_iterations,
                    gmres_tolerance));
            }
        }
        write_results(results, pose, solve_selection);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "kfbi_general_cap_exterior_trace_3d: " << error.what()
                  << std::endl;
        return 1;
    }
}
