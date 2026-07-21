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
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "native_nurbs_surface_3d.hpp"
#include "src/bulk_solvers/laplace_zfft_bulk_solver_3d.hpp"
#include "src/geometry/grid_pair_3d.hpp"
#include "src/geometry/nurbs_patch_triangulator_3d.hpp"
#include "src/geometry/p2_surface_3d.hpp"
#include "src/gmres/gmres.hpp"
#include "src/grid/cartesian_grid_3d.hpp"
#include "src/potentials/laplace_potential.hpp"
#include "src/transfer/laplace_correction_support.hpp"
#include "src/transfer/laplace_restrict_3d.hpp"
#include "src/transfer/laplace_spread_3d.hpp"

using namespace kfbim;

namespace {

constexpr double kBoxMin = -1.5;
constexpr double kBoxSide = 3.0;
constexpr double kTargetP2NodeSpacingOverH = 1.2;
constexpr int kCauchyValueNeighborCount = 48;
constexpr int kCauchyDerivativeNeighborCount = 28;
constexpr int kCauchyPolynomialDegree = 3;
constexpr int kRestrictGridDegree = 3;
constexpr int kRestrictNormalDegree = 3;

using GeometryKind = app3d::GeometryKind3D;
using NativeNurbsSurface3D = app3d::NativeNurbsSurface3D;
using SurfaceDof = app3d::SurfaceDof3D;
using SurfaceDofCloud = app3d::SurfaceDofCloud3D;
using SurfacePatchInfo = app3d::SurfaceDofPatch3D;

enum class CauchyStencilPolicy3D {
    TopologicalNearest,
    SamePatch,
    BalancedPatches
};

std::string cauchy_policy_name(CauchyStencilPolicy3D policy)
{
    switch (policy) {
    case CauchyStencilPolicy3D::TopologicalNearest:
        return "topological_nearest";
    case CauchyStencilPolicy3D::SamePatch:
        return "same_patch";
    case CauchyStencilPolicy3D::BalancedPatches:
        return "balanced_patches";
    }
    throw std::runtime_error("unknown 3D Cauchy stencil policy");
}

CauchyStencilPolicy3D selected_cauchy_policy()
{
    const char* raw = std::getenv("KFBIM_3D_CAUCHY_POLICY");
    if (raw == nullptr || std::string(raw).empty()
        || std::string(raw) == "topological_nearest") {
        return CauchyStencilPolicy3D::TopologicalNearest;
    }
    if (std::string(raw) == "same_patch")
        return CauchyStencilPolicy3D::SamePatch;
    if (std::string(raw) == "balanced_patches")
        return CauchyStencilPolicy3D::BalancedPatches;
    throw std::invalid_argument(
        "KFBIM_3D_CAUCHY_POLICY must be topological_nearest, same_patch, or balanced_patches");
}

int positive_environment_integer(const char* name, int default_value)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr)
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
    if (consumed != text.size() || value <= 0) {
        throw std::invalid_argument(std::string(name)
                                    + " must be a positive integer");
    }
    return value;
}

struct GeometryBundle {
    std::string name;
    std::string description;
    NativeNurbsSurface3D native_surface;
    Interface3D correction_interface;
    Interface3D crossing_interface;
    std::vector<geometry3d::NurbsParamTriangle3D> correction_triangles;
    std::vector<geometry3d::NurbsParamTriangle3D> geometry_triangles;
    int feature_edges = 0;
    int feature_vertices = 0;
    std::function<bool(const Eigen::Vector3d&)> exact_inside;
};

struct SurfaceCloudDiagnostics {
    double area = 0.0;
    double area_relative_error = 0.0;
    double normal_error = 0.0;
    double nearest_spacing_min_over_h = 0.0;
    double nearest_spacing_mean_over_h = 0.0;
    double nearest_spacing_max_over_h = 0.0;
};

struct CauchyStencil {
    std::vector<int> value_ids;
    std::vector<int> derivative_ids;
    double radius_over_h = 0.0;
    int incident_patch_count = 0;
    int value_patch_imbalance = 0;
    int derivative_patch_imbalance = 0;
};

struct CauchyStencilSet {
    CauchyStencilPolicy3D policy = CauchyStencilPolicy3D::TopologicalNearest;
    int value_count = 0;
    int derivative_count = 0;
    int value_count_min = 0;
    int value_count_max = 0;
    int derivative_count_min = 0;
    int derivative_count_max = 0;
    std::vector<CauchyStencil> rows;
    double radius_max_over_h = 0.0;
    double radius_mean_over_h = 0.0;
    int incident_patch_count_min = 0;
    int incident_patch_count_max = 0;
    int value_patch_imbalance_max = 0;
    int derivative_patch_imbalance_max = 0;
};

struct SolveMetrics3D {
    std::string formulation;
    int iterations = 0;
    bool converged = false;
    double seconds = 0.0;
    double gmres_relative_residual = 0.0;
    double operator_residual_linf = 0.0;
    double exterior_condition_linf = 0.0;
    double boundary_residual_linf = 0.0;
    double route_mismatch_linf = 0.0;
    double density_linf = 0.0;
    double density_l2 = 0.0;
    double interior_linf = 0.0;
    double interior_l2 = 0.0;
    double exterior_bulk_linf = 0.0;
    double exterior_bulk_l2 = 0.0;
    double data_weighted_mean = 0.0;
    double density_weighted_mean = 0.0;
    double constant_shift = 0.0;
};

struct ReadinessResult {
    std::string geometry;
    std::string cauchy_policy;
    int N = 0;
    double h = 0.0;
    int correction_panels = 0;
    int correction_dofs = 0;
    int crossing_panels = 0;
    int feature_edges = 0;
    int feature_vertices = 0;
    int interior_nodes = 0;
    int exterior_nodes = 0;
    int label_mismatches = 0;
    int crossings = 0;
    int exact_crossings = 0;
    int gap_crossings = 0;
    int endpoint_crossings = 0;
    int correction_nodes = 0;
    double correction_area = 0.0;
    double crossing_area = 0.0;
    double normal_error = 0.0;
    double min_box_margin_over_h = 0.0;
    double constant_A1_linf = 0.0;
    double constant_exterior_trace_linf = 0.0;
    double constant_interior_trace_linf = 0.0;
    double constant_bulk_linf = 0.0;
    double harmonic_constant_exterior_trace_linf = 0.0;
    double harmonic_constant_interior_trace_linf = 0.0;
    double harmonic_constant_exterior_normal_linf = 0.0;
    double harmonic_constant_interior_normal_linf = 0.0;
    double harmonic_constant_bulk_linf = 0.0;
    int surface_patches = 0;
    int surface_dofs = 0;
    double surface_dof_area = 0.0;
    double surface_area_relative_error = 0.0;
    double surface_spacing_min_over_h = 0.0;
    double surface_spacing_mean_over_h = 0.0;
    double surface_spacing_max_over_h = 0.0;
    int cauchy_value_neighbors = 0;
    int cauchy_derivative_neighbors = 0;
    int cauchy_value_neighbors_min = 0;
    int cauchy_value_neighbors_max = 0;
    int cauchy_derivative_neighbors_min = 0;
    int cauchy_derivative_neighbors_max = 0;
    double cauchy_radius_max_over_h = 0.0;
    double cauchy_radius_mean_over_h = 0.0;
    int cauchy_incident_patches_min = 0;
    int cauchy_incident_patches_max = 0;
    int cauchy_value_patch_imbalance_max = 0;
    int cauchy_derivative_patch_imbalance_max = 0;
    double cauchy_condition_median = 0.0;
    double cauchy_condition_p95 = 0.0;
    double cauchy_condition_max = 0.0;
    SolveMetrics3D neumann;
    SolveMetrics3D dirichlet_normal;
};

bool is_power_of_two(int value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

GeometryKind parse_geometry(const std::string& name)
{
    if (name == "torus" || name == "ring")
        return GeometryKind::Torus;
    if (name == "cylinder" || name == "hollow_cylinder"
        || name == "pipe" || name == "capped_cylinder")
        return GeometryKind::HollowCylinder;
    if (name == "l_prism" || name == "l-prism" || name == "lprism")
        return GeometryKind::LPrism;
    throw std::invalid_argument(
        "geometry must be torus, cylinder, l_prism, or all");
}

double manufactured_u_3d(const Eigen::Vector3d& point)
{
    const double exponential = std::exp(0.35 * point.x());
    return exponential * std::cos(0.21 * point.y())
        * std::cos(0.28 * point.z());
}

Eigen::Vector3d manufactured_gradient_3d(const Eigen::Vector3d& point)
{
    const double exponential = std::exp(0.35 * point.x());
    const double cos_y = std::cos(0.21 * point.y());
    const double sin_y = std::sin(0.21 * point.y());
    const double cos_z = std::cos(0.28 * point.z());
    const double sin_z = std::sin(0.28 * point.z());
    return {
        0.35 * exponential * cos_y * cos_z,
        -0.21 * exponential * sin_y * cos_z,
        -0.28 * exponential * cos_y * sin_z
    };
}

SurfaceCloudDiagnostics validate_surface_dofs(const SurfaceDofCloud& cloud,
                                              double h)
{
    if (cloud.patches.empty() || cloud.dofs.size() < 2)
        throw std::runtime_error("surface panel-center cloud is empty");
    std::vector<int> patch_counts(cloud.patches.size(), 0);
    SurfaceCloudDiagnostics result;
    for (std::size_t i = 0; i < cloud.patches.size(); ++i) {
        const SurfacePatchInfo& patch = cloud.patches[i];
        if (patch.smooth_patch_ids.empty()
            || std::find(patch.smooth_patch_ids.begin(),
                         patch.smooth_patch_ids.end(),
                         static_cast<int>(i)) == patch.smooth_patch_ids.end()) {
            throw std::runtime_error(
                "surface patch smooth adjacency must include itself");
        }
        for (int adjacent : patch.smooth_patch_ids) {
            if (adjacent < 0 || adjacent >= static_cast<int>(cloud.patches.size()))
                throw std::runtime_error("surface patch has invalid adjacency");
        }
    }
    for (const SurfaceDof& dof : cloud.dofs) {
        if (dof.patch_id < 0
            || dof.patch_id >= static_cast<int>(cloud.patches.size()))
            throw std::runtime_error("surface DOF has invalid patch ownership");
        if (!dof.point.allFinite() || !dof.normal.allFinite()
            || !std::isfinite(dof.weight) || !(dof.weight > 0.0))
            throw std::runtime_error("surface DOF contains NaN/Inf");
        ++patch_counts[static_cast<std::size_t>(dof.patch_id)];
        result.area += dof.weight;
        result.normal_error = std::max(
            result.normal_error, std::abs(dof.normal.norm() - 1.0));
    }
    if (std::find(patch_counts.begin(), patch_counts.end(), 0) != patch_counts.end())
        throw std::runtime_error("at least one surface patch has no center DOF");
    result.area_relative_error = std::abs(result.area - cloud.expected_area)
                               / cloud.expected_area;

    double spacing_sum = 0.0;
    double spacing_min = std::numeric_limits<double>::infinity();
    double spacing_max = 0.0;
    for (std::size_t i = 0; i < cloud.dofs.size(); ++i) {
        double nearest_sq = std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < cloud.dofs.size(); ++j) {
            if (i == j)
                continue;
            nearest_sq = std::min(
                nearest_sq,
                (cloud.dofs[i].point - cloud.dofs[j].point).squaredNorm());
        }
        const double spacing = std::sqrt(nearest_sq) / h;
        spacing_sum += spacing;
        spacing_min = std::min(spacing_min, spacing);
        spacing_max = std::max(spacing_max, spacing);
    }
    result.nearest_spacing_min_over_h = spacing_min;
    result.nearest_spacing_mean_over_h = spacing_sum
        / static_cast<double>(cloud.dofs.size());
    result.nearest_spacing_max_over_h = spacing_max;
    return result;
}

std::vector<int> select_cauchy_dofs(CauchyStencilPolicy3D policy,
                                    const NativeNurbsSurface3D& surface,
                                    const SurfaceDofCloud& cloud,
                                    int center,
                                    int count)
{
    switch (policy) {
    case CauchyStencilPolicy3D::TopologicalNearest:
        return app3d::nearest_topological_cauchy_dofs(
            surface, cloud, center, count);
    case CauchyStencilPolicy3D::SamePatch:
        return app3d::nearest_same_patch_cauchy_dofs(cloud, center, count);
    case CauchyStencilPolicy3D::BalancedPatches:
        return app3d::balanced_topological_cauchy_dofs(
            surface, cloud, center, count);
    }
    throw std::runtime_error("unknown Cauchy stencil selection policy");
}

int selected_patch_imbalance(const SurfaceDofCloud& cloud,
                             const std::vector<int>& ids)
{
    std::map<int, int> counts;
    for (int q : ids)
        ++counts[cloud.dofs[static_cast<std::size_t>(q)].patch_id];
    if (counts.empty())
        return 0;
    int minimum = std::numeric_limits<int>::max();
    int maximum = 0;
    for (const auto& item : counts) {
        minimum = std::min(minimum, item.second);
        maximum = std::max(maximum, item.second);
    }
    return maximum - minimum;
}

CauchyStencilSet build_cauchy_stencils(const NativeNurbsSurface3D& surface,
                                       const SurfaceDofCloud& cloud,
                                       double h,
                                       int requested_value_count,
                                       int requested_derivative_count,
                                       CauchyStencilPolicy3D policy)
{
    if (requested_value_count <= 0 || requested_derivative_count < 0
        || requested_derivative_count > requested_value_count) {
        throw std::invalid_argument("invalid Cauchy stencil sizes");
    }

    CauchyStencilSet result;
    result.policy = policy;
    result.value_count = requested_value_count;
    result.derivative_count = requested_derivative_count;
    result.value_count_min = std::numeric_limits<int>::max();
    result.derivative_count_min = std::numeric_limits<int>::max();
    result.rows.reserve(cloud.dofs.size());
    result.incident_patch_count_min = std::numeric_limits<int>::max();

    for (int center = 0; center < static_cast<int>(cloud.dofs.size()); ++center) {
        const SurfaceDof& target = cloud.dofs[static_cast<std::size_t>(center)];
        CauchyStencil stencil;
        stencil.value_ids = select_cauchy_dofs(
            policy, surface, cloud, center, result.value_count);
        stencil.derivative_ids = select_cauchy_dofs(
            policy, surface, cloud, center, result.derivative_count);
        if (std::find(stencil.value_ids.begin(), stencil.value_ids.end(), center)
            == stencil.value_ids.end()) {
            throw std::runtime_error("Cauchy value stencil does not contain its center");
        }
        if (std::find(stencil.derivative_ids.begin(),
                      stencil.derivative_ids.end(), center)
            == stencil.derivative_ids.end()) {
            throw std::runtime_error(
                "Cauchy normal stencil does not contain its center");
        }
        double radius_sq = 0.0;
        for (int q : stencil.value_ids) {
            radius_sq = std::max(
                radius_sq,
                (cloud.dofs[static_cast<std::size_t>(q)].point - target.point)
                    .squaredNorm());
        }
        for (int q : stencil.derivative_ids) {
            radius_sq = std::max(
                radius_sq,
                (cloud.dofs[static_cast<std::size_t>(q)].point - target.point)
                    .squaredNorm());
        }
        stencil.radius_over_h = std::sqrt(radius_sq) / h;
        std::set<int> selected_patches;
        for (int q : stencil.value_ids)
            selected_patches.insert(cloud.dofs[static_cast<std::size_t>(q)].patch_id);
        for (int q : stencil.derivative_ids)
            selected_patches.insert(cloud.dofs[static_cast<std::size_t>(q)].patch_id);
        stencil.incident_patch_count = static_cast<int>(selected_patches.size());
        stencil.value_patch_imbalance =
            selected_patch_imbalance(cloud, stencil.value_ids);
        stencil.derivative_patch_imbalance =
            selected_patch_imbalance(cloud, stencil.derivative_ids);

        result.radius_max_over_h = std::max(
            result.radius_max_over_h, stencil.radius_over_h);
        result.radius_mean_over_h += stencil.radius_over_h;
        result.incident_patch_count_min = std::min(
            result.incident_patch_count_min, stencil.incident_patch_count);
        result.incident_patch_count_max = std::max(
            result.incident_patch_count_max, stencil.incident_patch_count);
        result.value_count_min = std::min(
            result.value_count_min, static_cast<int>(stencil.value_ids.size()));
        result.value_count_max = std::max(
            result.value_count_max, static_cast<int>(stencil.value_ids.size()));
        result.derivative_count_min = std::min(
            result.derivative_count_min,
            static_cast<int>(stencil.derivative_ids.size()));
        result.derivative_count_max = std::max(
            result.derivative_count_max,
            static_cast<int>(stencil.derivative_ids.size()));
        result.value_patch_imbalance_max = std::max(
            result.value_patch_imbalance_max, stencil.value_patch_imbalance);
        result.derivative_patch_imbalance_max = std::max(
            result.derivative_patch_imbalance_max,
            stencil.derivative_patch_imbalance);
        result.rows.push_back(std::move(stencil));
    }
    result.radius_mean_over_h /= static_cast<double>(result.rows.size());
    return result;
}

struct PolynomialPower3D {
    int x = 0;
    int y = 0;
    int z = 0;
};

double integer_power(double value, int power)
{
    double result = 1.0;
    for (int i = 0; i < power; ++i)
        result *= value;
    return result;
}

class HarmonicPolynomialSpace3D {
public:
    explicit HarmonicPolynomialSpace3D(int degree = 4)
        : degree_(degree)
    {
        if (degree_ < 1)
            throw std::invalid_argument("harmonic polynomial degree must be positive");
        powers_ = powers_through_degree(degree_);
        const std::vector<PolynomialPower3D> lower =
            powers_through_degree(degree_ - 2);
        std::map<std::array<int, 3>, int> lower_index;
        for (int i = 0; i < static_cast<int>(lower.size()); ++i) {
            lower_index[{lower[static_cast<std::size_t>(i)].x,
                         lower[static_cast<std::size_t>(i)].y,
                         lower[static_cast<std::size_t>(i)].z}] = i;
        }

        Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(
            static_cast<int>(lower.size()), static_cast<int>(powers_.size()));
        for (int col = 0; col < static_cast<int>(powers_.size()); ++col) {
            const PolynomialPower3D p = powers_[static_cast<std::size_t>(col)];
            if (p.x >= 2) {
                laplacian(lower_index.at({p.x - 2, p.y, p.z}), col)
                    += static_cast<double>(p.x * (p.x - 1));
            }
            if (p.y >= 2) {
                laplacian(lower_index.at({p.x, p.y - 2, p.z}), col)
                    += static_cast<double>(p.y * (p.y - 1));
            }
            if (p.z >= 2) {
                laplacian(lower_index.at({p.x, p.y, p.z - 2}), col)
                    += static_cast<double>(p.z * (p.z - 1));
            }
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            laplacian, Eigen::ComputeFullV);
        const int expected_dimension = (degree_ + 1) * (degree_ + 1);
        if (static_cast<int>(powers_.size()) - laplacian.rows()
            != expected_dimension) {
            throw std::runtime_error("unexpected 3D harmonic-space dimension");
        }
        transform_ = svd.matrixV().rightCols(expected_dimension);
        if ((laplacian * transform_).norm() > 1.0e-10)
            throw std::runtime_error("failed to construct a harmonic polynomial basis");
    }

    int dimension() const
    {
        return static_cast<int>(transform_.cols());
    }

    Eigen::VectorXd basis(double x, double y, double z) const
    {
        Eigen::VectorXd monomials(static_cast<int>(powers_.size()));
        for (int i = 0; i < monomials.size(); ++i) {
            const PolynomialPower3D p = powers_[static_cast<std::size_t>(i)];
            monomials[i] = integer_power(x, p.x)
                         * integer_power(y, p.y)
                         * integer_power(z, p.z);
        }
        return transform_.transpose() * monomials;
    }

    Eigen::MatrixXd gradient(double x, double y, double z) const
    {
        Eigen::MatrixXd monomial_gradient = Eigen::MatrixXd::Zero(
            3, static_cast<int>(powers_.size()));
        for (int i = 0; i < static_cast<int>(powers_.size()); ++i) {
            const PolynomialPower3D p = powers_[static_cast<std::size_t>(i)];
            if (p.x > 0) {
                monomial_gradient(0, i) = static_cast<double>(p.x)
                    * integer_power(x, p.x - 1)
                    * integer_power(y, p.y)
                    * integer_power(z, p.z);
            }
            if (p.y > 0) {
                monomial_gradient(1, i) = static_cast<double>(p.y)
                    * integer_power(x, p.x)
                    * integer_power(y, p.y - 1)
                    * integer_power(z, p.z);
            }
            if (p.z > 0) {
                monomial_gradient(2, i) = static_cast<double>(p.z)
                    * integer_power(x, p.x)
                    * integer_power(y, p.y)
                    * integer_power(z, p.z - 1);
            }
        }
        return monomial_gradient * transform_;
    }

private:
    static std::vector<PolynomialPower3D> powers_through_degree(int degree)
    {
        std::vector<PolynomialPower3D> result;
        if (degree < 0)
            return result;
        for (int total = 0; total <= degree; ++total) {
            for (int px = 0; px <= total; ++px) {
                for (int py = 0; py <= total - px; ++py)
                    result.push_back({px, py, total - px - py});
            }
        }
        return result;
    }

    int degree_ = 4;
    std::vector<PolynomialPower3D> powers_;
    Eigen::MatrixXd transform_;
};

Eigen::MatrixXd svd_pseudoinverse(const Eigen::MatrixXd& matrix,
                                  double relative_cutoff)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd singular = svd.singularValues();
    if (singular.size() == 0 || !(singular[0] > 0.0))
        throw std::runtime_error("cannot invert an empty Cauchy design matrix");
    Eigen::VectorXd inverse = singular;
    const double cutoff = relative_cutoff * singular[0];
    for (int i = 0; i < inverse.size(); ++i)
        inverse[i] = singular[i] > cutoff ? 1.0 / singular[i] : 0.0;
    return svd.matrixV() * inverse.asDiagonal() * svd.matrixU().transpose();
}

struct CauchyFitMap3D {
    Eigen::MatrixXd value_map;
    Eigen::MatrixXd normal_map;
    double condition = 0.0;
};

class PanelCenterCauchyFit3D {
public:
    PanelCenterCauchyFit3D(const SurfaceDofCloud& cloud,
                           const CauchyStencilSet& stencils,
                           double h,
                           int degree = 4)
        : cloud_(cloud)
        , stencils_(stencils)
        , h_(h)
        , space_(degree)
    {
        if (stencils_.rows.size() != cloud_.dofs.size())
            throw std::invalid_argument("Cauchy stencil count does not match surface DOFs");
        maps_.reserve(cloud_.dofs.size());
        for (int center = 0; center < static_cast<int>(cloud_.dofs.size()); ++center)
            maps_.push_back(build_map(center));
    }

    int dimension() const { return space_.dimension(); }
    std::vector<double> condition_values() const
    {
        std::vector<double> result;
        result.reserve(maps_.size());
        for (const CauchyFitMap3D& map : maps_)
            result.push_back(map.condition);
        return result;
    }
    const HarmonicPolynomialSpace3D& space() const { return space_; }

    Eigen::MatrixXd coefficients(const Eigen::VectorXd& value_jump,
                                 const Eigen::VectorXd& normal_jump) const
    {
        const int size = static_cast<int>(cloud_.dofs.size());
        if (value_jump.size() != size || normal_jump.size() != size)
            throw std::invalid_argument("jump data size does not match surface DOFs");
        Eigen::MatrixXd result(size, dimension());
        for (int center = 0; center < size; ++center) {
            const CauchyStencil& stencil =
                stencils_.rows[static_cast<std::size_t>(center)];
            const int value_count = static_cast<int>(stencil.value_ids.size());
            const int normal_count =
                static_cast<int>(stencil.derivative_ids.size());
            Eigen::VectorXd values(value_count);
            Eigen::VectorXd normals(normal_count);
            for (int k = 0; k < value_count; ++k)
                values[k] = value_jump[stencil.value_ids[static_cast<std::size_t>(k)]];
            for (int k = 0; k < normal_count; ++k)
                normals[k] = normal_jump[stencil.derivative_ids[static_cast<std::size_t>(k)]];
            const CauchyFitMap3D& map = maps_[static_cast<std::size_t>(center)];
            result.row(center) =
                (map.value_map * values + map.normal_map * normals).transpose();
        }
        return result;
    }

private:
    Eigen::Vector3d local_coordinate(const SurfaceDof& center,
                                     const Eigen::Vector3d& point) const
    {
        const Eigen::Vector3d displacement = (point - center.point) / h_;
        return {displacement.dot(center.tangent1),
                displacement.dot(center.tangent2),
                displacement.dot(center.normal)};
    }

    CauchyFitMap3D build_map(int center_id) const
    {
        const SurfaceDof& center = cloud_.dofs[static_cast<std::size_t>(center_id)];
        const CauchyStencil& stencil =
            stencils_.rows[static_cast<std::size_t>(center_id)];
        const int value_count = static_cast<int>(stencil.value_ids.size());
        const int normal_count = static_cast<int>(stencil.derivative_ids.size());
        const int rows = value_count + normal_count;
        if (rows < dimension()) {
            throw std::runtime_error(
                "Cauchy fit has fewer conditions than cubic coefficients");
        }
        Eigen::MatrixXd design(rows, dimension());
        Eigen::VectorXd sqrt_weights(rows);

        for (int k = 0; k < value_count; ++k) {
            const SurfaceDof& sample = cloud_.dofs[static_cast<std::size_t>(
                stencil.value_ids[static_cast<std::size_t>(k)])];
            const Eigen::Vector3d xi = local_coordinate(center, sample.point);
            design.row(k) = space_.basis(xi.x(), xi.y(), xi.z()).transpose();
            sqrt_weights[k] = std::sqrt(
                1.0 / std::pow(0.35 + xi.norm(), 2.0));
        }
        for (int k = 0; k < normal_count; ++k) {
            const SurfaceDof& sample = cloud_.dofs[static_cast<std::size_t>(
                stencil.derivative_ids[static_cast<std::size_t>(k)])];
            const Eigen::Vector3d xi = local_coordinate(center, sample.point);
            const Eigen::Vector3d normal_components(
                sample.normal.dot(center.tangent1),
                sample.normal.dot(center.tangent2),
                sample.normal.dot(center.normal));
            design.row(value_count + k) =
                normal_components.transpose()
                * space_.gradient(xi.x(), xi.y(), xi.z());
            sqrt_weights[value_count + k] = std::sqrt(
                0.85 / std::pow(0.35 + xi.norm(), 2.0));
        }

        const Eigen::MatrixXd weighted = sqrt_weights.asDiagonal() * design;
        Eigen::JacobiSVD<Eigen::MatrixXd> condition_svd(weighted);
        const Eigen::VectorXd singular = condition_svd.singularValues();
        if (singular.size() != dimension() || singular.size() == 0
            || !singular.allFinite() || !(singular[0] > 0.0)
            || !(singular[singular.size() - 1] > 3.0e-12 * singular[0])) {
            throw std::runtime_error(
                "Cauchy fit is rank deficient at surface DOF "
                + std::to_string(center_id));
        }
        CauchyFitMap3D result;
        result.condition = singular[0] / singular[singular.size() - 1];
        const Eigen::MatrixXd pinv = svd_pseudoinverse(weighted, 3.0e-12);
        result.value_map.resize(dimension(), value_count);
        result.normal_map.resize(dimension(), normal_count);
        for (int k = 0; k < value_count; ++k)
            result.value_map.col(k) = pinv.col(k) * sqrt_weights[k];
        for (int k = 0; k < normal_count; ++k) {
            result.normal_map.col(k) =
                pinv.col(value_count + k) * sqrt_weights[value_count + k] * h_;
        }
        return result;
    }

    const SurfaceDofCloud& cloud_;
    const CauchyStencilSet& stencils_;
    double h_ = 0.0;
    HarmonicPolynomialSpace3D space_;
    std::vector<CauchyFitMap3D> maps_;
};

GeometryBundle make_geometry(GeometryKind kind, double h)
{
    NativeNurbsSurface3D native_surface =
        app3d::make_native_nurbs_surface_3d(kind);
    geometry3d::NurbsPatchTriangulatorOptions3D options;
    options.edge_buffer_factor = 0.5;
    options.edge_sample_step_factor = 1.0;
    options.min_edge_samples = 4;
    options.edge_length_quadrature_samples = 16;
    options.H_factor = 2.0 * kTargetP2NodeSpacingOverH;
    options.max_edge_over_H = 1.10;
    options.metric_ratio_tolerance = 1.50;
    options.max_depth = 6;
    geometry3d::NurbsPatchTriangulation3D triangulation =
        geometry3d::triangulate_nurbs_surface_patches_3d(
            native_surface.patches, h, options);
    std::string name = native_surface.name;
    std::string description = native_surface.description;
    std::function<bool(const Eigen::Vector3d&)> exact_inside =
        native_surface.exact_inside;
    const int feature_edges = triangulation.summary.num_feature_edges;
    const int feature_vertices = triangulation.summary.num_feature_vertices;
    return {std::move(name),
            std::move(description),
            std::move(native_surface),
            std::move(triangulation.interface),
            std::move(triangulation.geometry_interface),
            std::move(triangulation.triangles),
            std::move(triangulation.geometry_triangles),
            feature_edges,
            feature_vertices,
            std::move(exact_inside)};
}

Eigen::Vector3d grid_point(const CartesianGrid3D& grid, int node)
{
    const std::array<double, 3> x = grid.coord(node);
    return {x[0], x[1], x[2]};
}

std::array<double, 4> cubic_lagrange_weights(double fraction)
{
    constexpr std::array<double, 4> nodes{{-1.0, 0.0, 1.0, 2.0}};
    std::array<double, 4> weights{{1.0, 1.0, 1.0, 1.0}};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i != j) {
                weights[static_cast<std::size_t>(i)] *=
                    (fraction - nodes[static_cast<std::size_t>(j)])
                    / (nodes[static_cast<std::size_t>(i)]
                       - nodes[static_cast<std::size_t>(j)]);
            }
        }
    }
    return weights;
}

struct HarmonicJetField3D {
    Eigen::VectorXd potential;
    Eigen::MatrixXd coefficients;
};

struct HarmonicCrossingRow3D {
    int rhs_node = -1;
    int center_dof = -1;
    double scale = 0.0;
    Eigen::VectorXd evaluation;
};

struct HarmonicTraceSample3D {
    std::array<int, 64> grid_ids{};
    std::array<double, 64> weights{};
    Eigen::VectorXd correction_evaluation;
};

class PanelCenterHarmonicJetKFBI3D {
public:
    PanelCenterHarmonicJetKFBI3D(const CartesianGrid3D& grid,
                                 const GridPair3D& grid_pair,
                                 const NativeNurbsSurface3D& native_surface,
                                 const std::vector<geometry3d::NurbsParamTriangle3D>&
                                     correction_triangles,
                                 const std::vector<geometry3d::NurbsParamTriangle3D>&
                                     geometry_triangles,
                                 const SurfaceDofCloud& cloud,
                                 const CauchyStencilSet& stencils)
        : grid_(grid)
        , grid_pair_(grid_pair)
        , native_surface_(native_surface)
        , correction_triangles_(correction_triangles)
        , geometry_triangles_(geometry_triangles)
        , cloud_(cloud)
        , h_(grid.spacing()[0])
        , fit_(cloud, stencils, h_, kCauchyPolynomialDegree)
        , bulk_(grid, ZfftBcType::Dirichlet, 0.0, 2)
        , correction_support_(build_laplace_correction_support_3d(
              grid_pair, "PanelCenterHarmonicJetKFBI3D"))
    {
        const auto spacing = grid_.spacing();
        if (std::abs(spacing[0] - spacing[1]) > 1.0e-13
            || std::abs(spacing[0] - spacing[2]) > 1.0e-13) {
            throw std::invalid_argument(
                "harmonic-jet KFBI3D requires an isotropic Cartesian grid");
        }
        build_crossing_rows();
        build_trace_templates();
        build_joint_trace_fit();
    }

    int surface_size() const
    {
        return static_cast<int>(cloud_.dofs.size());
    }

    double surface_area() const
    {
        double area = 0.0;
        for (const SurfaceDof& dof : cloud_.dofs)
            area += dof.weight;
        return area;
    }

    const SurfaceDofCloud& surface() const { return cloud_; }

    std::vector<double> cauchy_condition_values() const
    {
        return fit_.condition_values();
    }

    HarmonicJetField3D evaluate(const Eigen::VectorXd& value_jump,
                                const Eigen::VectorXd& normal_jump) const
    {
        HarmonicJetField3D result;
        result.coefficients = fit_.coefficients(value_jump, normal_jump);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(grid_.num_dofs());
        for (const HarmonicCrossingRow3D& row : crossing_rows_) {
            rhs[row.rhs_node] += row.scale
                * row.evaluation.dot(
                    result.coefficients.row(row.center_dof).transpose());
        }
        // The spread correction is assembled for Delta_h, while the project
        // bulk solver accepts the right-hand side of -Delta_h.
        bulk_.solve(-rhs, result.potential);
        return result;
    }

    Eigen::VectorXd exterior_trace(const HarmonicJetField3D& field,
                                   const Eigen::VectorXd& value_jump,
                                   const Eigen::VectorXd& normal_jump) const
    {
        return recover_trace(
            continued_samples(field, value_jump, normal_jump, false),
            c0_weights_, 1.0);
    }

    Eigen::VectorXd interior_trace(const HarmonicJetField3D& field,
                                   const Eigen::VectorXd& value_jump,
                                   const Eigen::VectorXd& normal_jump) const
    {
        return recover_trace(
            continued_samples(field, value_jump, normal_jump, true),
            c0_weights_, 1.0);
    }

    Eigen::VectorXd exterior_normal_trace(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        return recover_trace(
            continued_samples(field, value_jump, normal_jump, false),
            c1_weights_, 1.0 / h_);
    }

    Eigen::VectorXd interior_normal_trace(
        const HarmonicJetField3D& field,
        const Eigen::VectorXd& value_jump,
        const Eigen::VectorXd& normal_jump) const
    {
        return recover_trace(
            continued_samples(field, value_jump, normal_jump, true),
            c1_weights_, 1.0 / h_);
    }

private:
    Eigen::MatrixXd continued_samples(const HarmonicJetField3D& field,
                                      const Eigen::VectorXd& value_jump,
                                      const Eigen::VectorXd& normal_jump,
                                      bool interior_continuation) const
    {
        const int size = surface_size();
        if (field.potential.size() != grid_.num_dofs()
            || field.coefficients.rows() != size
            || value_jump.size() != size
            || normal_jump.size() != size) {
            throw std::invalid_argument(
                "exterior trace received incompatible field or jump sizes");
        }

        Eigen::MatrixXd samples = Eigen::MatrixXd::Zero(size, 8);
        for (int center = 0; center < size; ++center) {
            for (int side = 0; side < 2; ++side) {
                for (int layer = 0; layer < 4; ++layer) {
                    const HarmonicTraceSample3D& sample = trace_samples_[
                        trace_sample_index(center, side, layer)];
                    double value = 0.0;
                    for (int q = 0; q < 64; ++q) {
                        value += sample.weights[static_cast<std::size_t>(q)]
                               * field.potential[
                                   sample.grid_ids[static_cast<std::size_t>(q)]];
                    }
                    value += sample.correction_evaluation.dot(
                        field.coefficients.row(center).transpose());
                    samples(center, 4 * side + layer) = value;
                }
            }

            if (interior_continuation) {
                // At rho=+tau*h, W^- = W^+ + [W] + rho[W_n].
                for (int layer = 0; layer < 4; ++layer) {
                    samples(center, 4 + layer) += value_jump[center];
                    samples(center, 4 + layer) +=
                        normal_layers_[static_cast<std::size_t>(layer)]
                        * h_ * normal_jump[center];
                }
            } else {
                // At rho=-tau*h, W^+ = W^- - [W] - rho[W_n].
                for (int layer = 0; layer < 4; ++layer) {
                    samples(center, layer) -= value_jump[center];
                    samples(center, layer) +=
                        normal_layers_[static_cast<std::size_t>(layer)]
                        * h_ * normal_jump[center];
                }
            }
        }
        return samples;
    }

    Eigen::VectorXd recover_trace(
        const Eigen::MatrixXd& samples,
        const std::array<double, 8>& weights,
        double scale) const
    {
        Eigen::VectorXd trace(samples.rows());
        for (int center = 0; center < samples.rows(); ++center) {
            double value = 0.0;
            for (int q = 0; q < 8; ++q)
                value += weights[static_cast<std::size_t>(q)]
                       * samples(center, q);
            trace[center] = scale * value;
        }
        return trace;
    }

    std::size_t trace_sample_index(int center, int side, int layer) const
    {
        return static_cast<std::size_t>((center * 2 + side) * 4 + layer);
    }

    Eigen::Vector3d local_coordinate(int center,
                                     const Eigen::Vector3d& point) const
    {
        const SurfaceDof& dof = cloud_.dofs[static_cast<std::size_t>(center)];
        const Eigen::Vector3d displacement = (point - dof.point) / h_;
        return {displacement.dot(dof.tangent1),
                displacement.dot(dof.tangent2),
                displacement.dot(dof.normal)};
    }

    int surface_dof_for_crossing(const P2CrossingOwner3D& owner,
                                 const Eigen::Vector3d& point) const
    {
        const geometry3d::NurbsParamTriangle3D* triangle = nullptr;
        Eigen::Vector3d barycentric = Eigen::Vector3d::Zero();
        if (owner.geometry_panel_index >= 0
            && owner.geometry_panel_index
                   < static_cast<int>(geometry_triangles_.size())) {
            triangle = &geometry_triangles_[
                static_cast<std::size_t>(owner.geometry_panel_index)];
            barycentric = owner.geometry_barycentric;
        } else if (owner.panel_index >= 0
                   && owner.panel_index
                          < static_cast<int>(correction_triangles_.size())) {
            triangle = &correction_triangles_[
                static_cast<std::size_t>(owner.panel_index)];
            barycentric = owner.barycentric;
        }
        if (triangle == nullptr)
            throw std::runtime_error("crossing has no native NURBS triangle owner");
        const Eigen::Vector2d uv = app3d::interpolate_triangle_parameter(
            *triangle, barycentric);
        const std::array<int, 4> candidates =
            app3d::parameter_dof_candidates_2x2(
                native_surface_, cloud_, triangle->patch_index,
                uv.x(), uv.y());
        const geometry3d::NurbsSurfaceDerivatives3D derivatives =
            native_surface_.patches[
                static_cast<std::size_t>(triangle->patch_index)]
                .evaluate_with_derivatives(uv.x(), uv.y());
        Eigen::Vector3d reference_normal =
            derivatives.du.cross(derivatives.dv).normalized();
        int nearest = -1;
        double best = std::numeric_limits<double>::infinity();
        for (int pass = 0; pass < 2 && nearest < 0; ++pass) {
            for (int q : candidates) {
                const SurfaceDof& dof = cloud_.dofs[static_cast<std::size_t>(q)];
                if (pass == 0 && dof.normal.dot(reference_normal) < 0.50) {
                    continue;
                }
                const double distance = (dof.point - point).squaredNorm();
                if (distance < best) {
                    best = distance;
                    nearest = q;
                }
            }
        }
        if (nearest < 0)
            throw std::runtime_error(
                "failed to associate a crossing with four parameter DOFs");
        return nearest;
    }

    void build_crossing_rows()
    {
        crossing_rows_.reserve(correction_support_.crossing_ops.size());
        for (const LaplaceCrossingCorrectionOp& op
             : correction_support_.crossing_ops) {
            const P2CrossingOwner3D owner =
                grid_pair_.p2_crossing_owner_between(
                    op.rhs_node, op.correction_node);
            const Eigen::Vector3d a = grid_point(grid_, op.rhs_node);
            const Eigen::Vector3d b = grid_point(grid_, op.correction_node);
            const double phase = std::max(0.0, std::min(1.0, owner.edge_parameter));
            const Eigen::Vector3d hit = a + phase * (b - a);

            const int center = surface_dof_for_crossing(owner, hit);
            const Eigen::Vector3d xi =
                local_coordinate(center, grid_point(grid_, op.correction_node));
            HarmonicCrossingRow3D row;
            row.rhs_node = op.rhs_node;
            row.center_dof = center;
            row.scale = static_cast<double>(op.side_delta) * op.stencil_weight;
            row.evaluation = fit_.space().basis(xi.x(), xi.y(), xi.z());
            crossing_rows_.push_back(std::move(row));
        }
    }

    HarmonicTraceSample3D build_trace_sample(int center,
                                             bool desired_inside,
                                             double signed_layer) const
    {
        const SurfaceDof& dof = cloud_.dofs[static_cast<std::size_t>(center)];
        const Eigen::Vector3d query =
            dof.point + signed_layer * h_ * dof.normal;
        const std::array<double, 3> origin = grid_.origin();
        const std::array<int, 3> dims = grid_.dof_dims();
        std::array<int, 3> lo{};
        std::array<std::array<double, 4>, 3> axis_weights{};
        for (int axis = 0; axis < 3; ++axis) {
            const double coordinate =
                (query[axis] - origin[static_cast<std::size_t>(axis)]) / h_;
            if (coordinate < 0.0
                || coordinate
                   > static_cast<double>(dims[static_cast<std::size_t>(axis)] - 1)) {
                throw std::runtime_error(
                    "normal-layer sample exits the embedding box");
            }
            const int floor_index = static_cast<int>(std::floor(coordinate));
            const int start = std::max(
                0,
                std::min(floor_index - 1,
                         dims[static_cast<std::size_t>(axis)] - 4));
            lo[static_cast<std::size_t>(axis)] = start + 1;
            const double fraction = coordinate
                - static_cast<double>(lo[static_cast<std::size_t>(axis)]);
            axis_weights[static_cast<std::size_t>(axis)] =
                cubic_lagrange_weights(fraction);
        }

        // Four Cartesian slices are first interpolated by a 4x4 bicubic
        // polynomial; a final cubic interpolation in the third direction is
        // algebraically the standard 4x4x4 tensor-product tricubic formula.
        HarmonicTraceSample3D result;
        result.correction_evaluation = Eigen::VectorXd::Zero(fit_.dimension());
        int q = 0;
        for (int iz = 0; iz < 4; ++iz) {
            for (int iy = 0; iy < 4; ++iy) {
                for (int ix = 0; ix < 4; ++ix) {
                    const int i = lo[0] - 1 + ix;
                    const int j = lo[1] - 1 + iy;
                    const int k = lo[2] - 1 + iz;
                    const int node = grid_.index(i, j, k);
                    const double weight = axis_weights[0][static_cast<std::size_t>(ix)]
                        * axis_weights[1][static_cast<std::size_t>(iy)]
                        * axis_weights[2][static_cast<std::size_t>(iz)];
                    result.grid_ids[static_cast<std::size_t>(q)] = node;
                    result.weights[static_cast<std::size_t>(q)] = weight;
                    const bool node_inside = grid_pair_.domain_label(node) > 0;
                    if (node_inside != desired_inside) {
                        const Eigen::Vector3d xi =
                            local_coordinate(center, grid_point(grid_, node));
                        const double correction_sign = desired_inside ? 1.0 : -1.0;
                        result.correction_evaluation += correction_sign * weight
                            * fit_.space().basis(xi.x(), xi.y(), xi.z());
                    }
                    ++q;
                }
            }
        }
        return result;
    }

    void build_trace_templates()
    {
        trace_samples_.reserve(
            static_cast<std::size_t>(surface_size() * 2 * 4));
        for (int center = 0; center < surface_size(); ++center) {
            for (int side = 0; side < 2; ++side) {
                const bool desired_inside = side == 0;
                const double sign = desired_inside ? -1.0 : 1.0;
                for (double layer : normal_layers_)
                    trace_samples_.push_back(
                        build_trace_sample(center, desired_inside, sign * layer));
            }
        }
    }

    void build_joint_trace_fit()
    {
        Eigen::MatrixXd design = Eigen::MatrixXd::Zero(8, 6);
        for (int q = 0; q < 4; ++q) {
            const double xm = -normal_layers_[static_cast<std::size_t>(q)];
            const double xp = normal_layers_[static_cast<std::size_t>(q)];
            design(q, 0) = 1.0;
            design(q, 1) = xm;
            design(q, 2) = xm * xm;
            design(q, 3) = xm * xm * xm;
            design(4 + q, 0) = 1.0;
            design(4 + q, 1) = xp;
            design(4 + q, 4) = xp * xp;
            design(4 + q, 5) = xp * xp * xp;
        }
        const Eigen::MatrixXd pinv = svd_pseudoinverse(design, 1.0e-13);
        for (int q = 0; q < 8; ++q) {
            c0_weights_[static_cast<std::size_t>(q)] = pinv(0, q);
            c1_weights_[static_cast<std::size_t>(q)] = pinv(1, q);
        }
    }

    const CartesianGrid3D& grid_;
    const GridPair3D& grid_pair_;
    const NativeNurbsSurface3D& native_surface_;
    const std::vector<geometry3d::NurbsParamTriangle3D>& correction_triangles_;
    const std::vector<geometry3d::NurbsParamTriangle3D>& geometry_triangles_;
    const SurfaceDofCloud& cloud_;
    double h_ = 0.0;
    PanelCenterCauchyFit3D fit_;
    LaplaceFftBulkSolverZfft3D bulk_;
    LaplaceCorrectionSupport3D correction_support_;
    std::vector<HarmonicCrossingRow3D> crossing_rows_;
    std::vector<HarmonicTraceSample3D> trace_samples_;
    const std::array<double, kRestrictNormalDegree + 1> normal_layers_{
        {0.2, 0.6, 1.0, 1.4}};
    std::array<double, 8> c0_weights_{};
    std::array<double, 8> c1_weights_{};
};

class ExteriorZeroTraceOperator3D final : public IKFBIOperator {
public:
    explicit ExteriorZeroTraceOperator3D(
        const PanelCenterHarmonicJetKFBI3D& pipeline)
        : pipeline_(pipeline)
    {}

    int problem_size() const override
    {
        return pipeline_.surface_size() + 1;
    }

    void apply(const Eigen::VectorXd& unknown,
               Eigen::VectorXd& result) const override
    {
        if (unknown.size() != problem_size())
            throw std::invalid_argument("exterior-zero operator input has wrong size");
        const int size = pipeline_.surface_size();
        const Eigen::VectorXd value_jump = unknown.head(size);
        const Eigen::VectorXd zero_normal = Eigen::VectorXd::Zero(size);
        const HarmonicJetField3D field =
            pipeline_.evaluate(value_jump, zero_normal);
        const Eigen::VectorXd trace =
            pipeline_.exterior_trace(field, value_jump, zero_normal);
        result.resize(size + 1);
        result.head(size) = trace.array() + unknown[size];
        double weighted_mean = 0.0;
        for (int q = 0; q < size; ++q) {
            weighted_mean += pipeline_.surface().dofs[static_cast<std::size_t>(q)].weight
                           * value_jump[q];
        }
        result[size] = weighted_mean / pipeline_.surface_area();
    }

    Eigen::VectorXd right_hand_side(
        const Eigen::VectorXd& prescribed_normal_jump) const
    {
        const int size = pipeline_.surface_size();
        if (prescribed_normal_jump.size() != size)
            throw std::invalid_argument("prescribed Neumann data has wrong size");
        const Eigen::VectorXd zero_value = Eigen::VectorXd::Zero(size);
        const HarmonicJetField3D field =
            pipeline_.evaluate(zero_value, prescribed_normal_jump);
        Eigen::VectorXd result = Eigen::VectorXd::Zero(size + 1);
        result.head(size) = -pipeline_.exterior_trace(
            field, zero_value, prescribed_normal_jump);
        return result;
    }

private:
    const PanelCenterHarmonicJetKFBI3D& pipeline_;
};

struct ExteriorZeroTraceSolution3D {
    Eigen::VectorXd value_jump;
    Eigen::VectorXd potential;
    Eigen::MatrixXd coefficients;
    Eigen::VectorXd augmented_residual;
    std::vector<double> gmres_residuals;
    double lagrange_multiplier = 0.0;
    int iterations = 0;
    bool converged = false;
};

ExteriorZeroTraceSolution3D solve_exterior_zero_trace_neumann_3d(
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const Eigen::VectorXd& prescribed_normal_jump,
    double tolerance,
    int restart,
    int max_iterations)
{
    ExteriorZeroTraceOperator3D op(pipeline);
    const Eigen::VectorXd rhs = op.right_hand_side(prescribed_normal_jump);
    Eigen::VectorXd augmented_unknown = Eigen::VectorXd::Zero(op.problem_size());
    GMRES gmres(max_iterations, tolerance, restart);
    ExteriorZeroTraceSolution3D result;
    result.iterations = gmres.solve(op, rhs, augmented_unknown);
    result.converged = gmres.converged();
    result.gmres_residuals = gmres.residuals();
    const int size = pipeline.surface_size();
    result.value_jump = augmented_unknown.head(size);
    result.lagrange_multiplier = augmented_unknown[size];
    const HarmonicJetField3D field =
        pipeline.evaluate(result.value_jump, prescribed_normal_jump);
    result.potential = field.potential;
    result.coefficients = field.coefficients;
    Eigen::VectorXd applied;
    op.apply(augmented_unknown, applied);
    result.augmented_residual = applied - rhs;
    return result;
}

class ExteriorNormalTraceOperator3D final : public IKFBIOperator {
public:
    explicit ExteriorNormalTraceOperator3D(
        const PanelCenterHarmonicJetKFBI3D& pipeline)
        : pipeline_(pipeline)
    {}

    int problem_size() const override
    {
        return pipeline_.surface_size();
    }

    void apply(const Eigen::VectorXd& normal_jump,
               Eigen::VectorXd& result) const override
    {
        if (normal_jump.size() != problem_size()) {
            throw std::invalid_argument(
                "exterior-normal operator input has wrong size");
        }
        const Eigen::VectorXd zero_value =
            Eigen::VectorXd::Zero(problem_size());
        const HarmonicJetField3D field =
            pipeline_.evaluate(zero_value, normal_jump);
        result = pipeline_.exterior_normal_trace(
            field, zero_value, normal_jump);
    }

    Eigen::VectorXd right_hand_side(
        const Eigen::VectorXd& prescribed_value_jump) const
    {
        if (prescribed_value_jump.size() != problem_size()) {
            throw std::invalid_argument(
                "prescribed Dirichlet data has wrong size");
        }
        const Eigen::VectorXd zero_normal =
            Eigen::VectorXd::Zero(problem_size());
        const HarmonicJetField3D field =
            pipeline_.evaluate(prescribed_value_jump, zero_normal);
        return -pipeline_.exterior_normal_trace(
            field, prescribed_value_jump, zero_normal);
    }

private:
    const PanelCenterHarmonicJetKFBI3D& pipeline_;
};

struct ExteriorNormalTraceSolution3D {
    Eigen::VectorXd normal_jump;
    Eigen::VectorXd potential;
    Eigen::MatrixXd coefficients;
    Eigen::VectorXd operator_residual;
    std::vector<double> gmres_residuals;
    int iterations = 0;
    bool converged = false;
};

ExteriorNormalTraceSolution3D solve_exterior_zero_normal_dirichlet_3d(
    const PanelCenterHarmonicJetKFBI3D& pipeline,
    const Eigen::VectorXd& prescribed_value_jump,
    double tolerance,
    int restart,
    int max_iterations)
{
    ExteriorNormalTraceOperator3D op(pipeline);
    const Eigen::VectorXd rhs = op.right_hand_side(prescribed_value_jump);
    Eigen::VectorXd normal_jump = Eigen::VectorXd::Zero(op.problem_size());
    GMRES gmres(max_iterations, tolerance, restart);
    ExteriorNormalTraceSolution3D result;
    result.iterations = gmres.solve(op, rhs, normal_jump);
    result.converged = gmres.converged();
    result.gmres_residuals = gmres.residuals();
    result.normal_jump = normal_jump;
    const HarmonicJetField3D field =
        pipeline.evaluate(prescribed_value_jump, normal_jump);
    result.potential = field.potential;
    result.coefficients = field.coefficients;
    Eigen::VectorXd applied;
    op.apply(normal_jump, applied);
    result.operator_residual = applied - rhs;
    return result;
}

double vector_linf(const Eigen::VectorXd& values)
{
    return values.size() == 0 ? 0.0 : values.lpNorm<Eigen::Infinity>();
}

double vector_rms(const Eigen::VectorXd& values)
{
    return values.size() == 0
        ? 0.0
        : std::sqrt(values.squaredNorm() / static_cast<double>(values.size()));
}

struct ConditionStatistics3D {
    double median = 0.0;
    double p95 = 0.0;
    double maximum = 0.0;
};

ConditionStatistics3D summarize_conditions(std::vector<double> values)
{
    if (values.empty())
        throw std::invalid_argument("Cauchy condition list is empty");
    if (!std::all_of(values.begin(), values.end(), [](double value) {
            return std::isfinite(value) && value > 0.0;
        })) {
        throw std::runtime_error("Cauchy condition list contains invalid values");
    }
    std::sort(values.begin(), values.end());
    ConditionStatistics3D result;
    result.median = values[(values.size() - 1) / 2];
    const std::size_t p95_index = std::min(
        values.size() - 1,
        static_cast<std::size_t>(
            std::ceil(0.95 * static_cast<double>(values.size()))) - 1);
    result.p95 = values[p95_index];
    result.maximum = values.back();
    return result;
}

double surface_weighted_mean(const SurfaceDofCloud& surface,
                             const Eigen::VectorXd& values)
{
    if (values.size() != static_cast<int>(surface.dofs.size()))
        throw std::invalid_argument("surface weighted mean received wrong size");
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    for (int q = 0; q < values.size(); ++q) {
        const double weight = surface.dofs[static_cast<std::size_t>(q)].weight;
        weighted_sum += weight * values[q];
        weight_sum += weight;
    }
    return weighted_sum / weight_sum;
}

SolveMetrics3D run_neumann_case(
    const CartesianGrid3D& grid,
    const GridPair3D& grid_pair,
    const PanelCenterHarmonicJetKFBI3D& pipeline)
{
    const int size = pipeline.surface_size();
    Eigen::VectorXd exact_trace(size);
    Eigen::VectorXd normal_data(size);
    for (int q = 0; q < size; ++q) {
        const SurfaceDof& dof =
            pipeline.surface().dofs[static_cast<std::size_t>(q)];
        exact_trace[q] = manufactured_u_3d(dof.point);
        normal_data[q] = manufactured_gradient_3d(dof.point).dot(dof.normal);
    }
    // Enforce the discrete compatibility condition.  The exact flux has zero
    // continuous mean; the tiny quadrature defect is otherwise amplified by
    // the first-kind exterior-trace equation.
    normal_data.array() -= surface_weighted_mean(
        pipeline.surface(), normal_data);

    const auto solve_start = std::chrono::steady_clock::now();
    const ExteriorZeroTraceSolution3D solution =
        solve_exterior_zero_trace_neumann_3d(
            pipeline, normal_data, 2.0e-10, 80, 200);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();

    const HarmonicJetField3D field{
        solution.potential, solution.coefficients};
    const Eigen::VectorXd direct_exterior = pipeline.exterior_trace(
        field, solution.value_jump, normal_data);
    const Eigen::VectorXd exterior_from_jump = pipeline.interior_trace(
        field, solution.value_jump, normal_data) - solution.value_jump;

    SolveMetrics3D result;
    result.formulation = "neumann_exterior_zero_value_trace";
    result.iterations = solution.iterations;
    result.converged = solution.converged;
    result.seconds = seconds;
    result.gmres_relative_residual = solution.gmres_residuals.empty()
        ? std::numeric_limits<double>::quiet_NaN()
        : solution.gmres_residuals.back();
    result.operator_residual_linf = vector_linf(
        solution.augmented_residual.head(size));
    result.exterior_condition_linf = vector_linf(direct_exterior);
    result.route_mismatch_linf = vector_linf(
        direct_exterior - exterior_from_jump);
    result.data_weighted_mean = surface_weighted_mean(
        pipeline.surface(), normal_data);
    result.density_weighted_mean = surface_weighted_mean(
        pipeline.surface(), solution.value_jump);

    exact_trace.array() -= surface_weighted_mean(
        pipeline.surface(), exact_trace);
    const Eigen::VectorXd density_error = solution.value_jump - exact_trace;
    result.density_linf = vector_linf(density_error);
    result.density_l2 = vector_rms(density_error);

    double shift_sum = 0.0;
    int interior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (grid_pair.domain_label(node) <= 0)
            continue;
        shift_sum += manufactured_u_3d(grid_point(grid, node))
                   - solution.potential[node];
        ++interior_count;
    }
    result.constant_shift = shift_sum / static_cast<double>(interior_count);
    double interior_error_sq = 0.0;
    double exterior_error_sq = 0.0;
    int exterior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (grid_pair.domain_label(node) > 0) {
            const double error = solution.potential[node]
                               + result.constant_shift
                               - manufactured_u_3d(grid_point(grid, node));
            result.interior_linf = std::max(
                result.interior_linf, std::abs(error));
            interior_error_sq += error * error;
        } else {
            const double error = solution.potential[node];
            result.exterior_bulk_linf = std::max(
                result.exterior_bulk_linf, std::abs(error));
            exterior_error_sq += error * error;
            ++exterior_count;
        }
    }
    result.interior_l2 = std::sqrt(
        interior_error_sq / static_cast<double>(interior_count));
    result.exterior_bulk_l2 = std::sqrt(
        exterior_error_sq / static_cast<double>(exterior_count));
    return result;
}

SolveMetrics3D run_dirichlet_normal_case(
    const CartesianGrid3D& grid,
    const GridPair3D& grid_pair,
    const PanelCenterHarmonicJetKFBI3D& pipeline)
{
    const int size = pipeline.surface_size();
    Eigen::VectorXd value_data(size);
    Eigen::VectorXd exact_normal(size);
    for (int q = 0; q < size; ++q) {
        const SurfaceDof& dof =
            pipeline.surface().dofs[static_cast<std::size_t>(q)];
        value_data[q] = manufactured_u_3d(dof.point);
        exact_normal[q] = manufactured_gradient_3d(dof.point).dot(dof.normal);
    }

    const auto solve_start = std::chrono::steady_clock::now();
    const ExteriorNormalTraceSolution3D solution =
        solve_exterior_zero_normal_dirichlet_3d(
            pipeline, value_data, 2.0e-10, 0, 400);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();

    const HarmonicJetField3D field{
        solution.potential, solution.coefficients};
    const Eigen::VectorXd direct_exterior_normal =
        pipeline.exterior_normal_trace(
            field, value_data, solution.normal_jump);
    const Eigen::VectorXd exterior_normal_from_jump =
        pipeline.interior_normal_trace(
            field, value_data, solution.normal_jump) - solution.normal_jump;
    const Eigen::VectorXd boundary_residual = pipeline.interior_trace(
        field, value_data, solution.normal_jump) - value_data;

    SolveMetrics3D result;
    result.formulation = "dirichlet_exterior_zero_normal_trace";
    result.iterations = solution.iterations;
    result.converged = solution.converged;
    result.seconds = seconds;
    result.gmres_relative_residual = solution.gmres_residuals.empty()
        ? std::numeric_limits<double>::quiet_NaN()
        : solution.gmres_residuals.back();
    result.operator_residual_linf = vector_linf(solution.operator_residual);
    result.exterior_condition_linf = vector_linf(direct_exterior_normal);
    result.boundary_residual_linf = vector_linf(boundary_residual);
    result.route_mismatch_linf = vector_linf(
        direct_exterior_normal - exterior_normal_from_jump);
    result.data_weighted_mean = surface_weighted_mean(
        pipeline.surface(), value_data);
    result.density_weighted_mean = surface_weighted_mean(
        pipeline.surface(), solution.normal_jump);

    const Eigen::VectorXd density_error = solution.normal_jump - exact_normal;
    result.density_linf = vector_linf(density_error);
    result.density_l2 = vector_rms(density_error);

    double interior_error_sq = 0.0;
    double exterior_error_sq = 0.0;
    int interior_count = 0;
    int exterior_count = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        if (grid_pair.domain_label(node) > 0) {
            const double error = solution.potential[node]
                               - manufactured_u_3d(grid_point(grid, node));
            result.interior_linf = std::max(
                result.interior_linf, std::abs(error));
            interior_error_sq += error * error;
            ++interior_count;
        } else {
            const double error = solution.potential[node];
            result.exterior_bulk_linf = std::max(
                result.exterior_bulk_linf, std::abs(error));
            exterior_error_sq += error * error;
            ++exterior_count;
        }
    }
    result.interior_l2 = std::sqrt(
        interior_error_sq / static_cast<double>(interior_count));
    result.exterior_bulk_l2 = std::sqrt(
        exterior_error_sq / static_cast<double>(exterior_count));
    return result;
}

double validate_interface(const Interface3D& iface)
{
    if (iface.panel_node_layout() != PanelNodeLayout3D::QuadraticLagrange
        || iface.points_per_panel() != 6
        || iface.num_panels() <= 0
        || iface.num_points() <= 0
        || iface.weights().sum() <= 0.0) {
        throw std::runtime_error("invalid P2 surface interface");
    }
    double normal_error = 0.0;
    for (int q = 0; q < iface.num_points(); ++q) {
        const double norm = iface.normals().row(q).norm();
        if (!std::isfinite(norm))
            throw std::runtime_error("surface contains a non-finite normal");
        normal_error = std::max(normal_error, std::abs(norm - 1.0));
    }
    if (normal_error > 1.0e-9)
        throw std::runtime_error("surface normals are not unit length");
    return normal_error;
}

std::ofstream open_output_file(const std::filesystem::path& path)
{
    std::ofstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot open output file: " + path.string());
    return stream;
}

void write_surface_files(const std::filesystem::path& output_dir,
                         const GeometryBundle& geometry,
                         int N)
{
    std::filesystem::create_directories(output_dir);
    const std::string stem = geometry.name + "_N" + std::to_string(N);
    const Interface3D& surface = geometry.crossing_interface;
    std::ofstream obj = open_output_file(
        output_dir / (stem + "_crossing_surface.obj"));
    obj << std::setprecision(17);
    obj << "# " << geometry.description << '\n';
    for (int v = 0; v < surface.num_vertices(); ++v)
        obj << "v " << surface.vertices()(v, 0) << ' '
            << surface.vertices()(v, 1) << ' '
            << surface.vertices()(v, 2) << '\n';
    for (int p = 0; p < surface.num_panels(); ++p)
        obj << "f " << surface.panels()(p, 0) + 1 << ' '
            << surface.panels()(p, 1) + 1 << ' '
            << surface.panels()(p, 2) + 1 << '\n';

    const Interface3D& iface = geometry.correction_interface;
    std::ofstream csv = open_output_file(
        output_dir / (stem + "_correction_dofs.csv"));
    csv << std::setprecision(17);
    csv << "q,x,y,z,nx,ny,nz,weight\n";
    for (int q = 0; q < iface.num_points(); ++q)
        csv << q << ','
            << iface.points()(q, 0) << ',' << iface.points()(q, 1) << ','
            << iface.points()(q, 2) << ',' << iface.normals()(q, 0) << ','
            << iface.normals()(q, 1) << ',' << iface.normals()(q, 2) << ','
            << iface.weights()[q] << '\n';
}

void write_panel_center_files(const std::filesystem::path& output_dir,
                              const std::string& geometry_name,
                              int N,
                              const SurfaceDofCloud& cloud,
                              const CauchyStencilSet& stencils)
{
    std::filesystem::create_directories(output_dir);
    const std::string stem = geometry_name + "_N" + std::to_string(N);

    std::ofstream patch_csv = open_output_file(
        output_dir / (stem + "_surface_patches.csv"));
    patch_csv << "patch_id,patch_name,nu,nv,smooth_patch_ids\n";
    for (int patch_id = 0;
         patch_id < static_cast<int>(cloud.patches.size()); ++patch_id) {
        const SurfacePatchInfo& patch =
            cloud.patches[static_cast<std::size_t>(patch_id)];
        patch_csv << patch_id << ',' << patch.name << ','
                  << patch.nu << ',' << patch.nv << ',';
        for (std::size_t j = 0; j < patch.smooth_patch_ids.size(); ++j) {
            if (j != 0)
                patch_csv << ';';
            patch_csv << patch.smooth_patch_ids[j];
        }
        patch_csv << '\n';
    }

    std::ofstream dof_csv = open_output_file(
        output_dir / (stem + "_surface_panel_center_dofs.csv"));
    dof_csv << std::setprecision(17);
    dof_csv << "q,patch_id,patch_name,i,j,u,v,"
               "x,y,z,nx,ny,nz,t1x,t1y,t1z,t2x,t2y,t2z,weight\n";
    for (int q = 0; q < static_cast<int>(cloud.dofs.size()); ++q) {
        const SurfaceDof& dof = cloud.dofs[static_cast<std::size_t>(q)];
        dof_csv << q << ',' << dof.patch_id << ','
                << cloud.patches[static_cast<std::size_t>(dof.patch_id)].name
                << ',' << dof.i << ',' << dof.j << ','
                << dof.u << ',' << dof.v << ','
                << dof.point.x() << ',' << dof.point.y() << ',' << dof.point.z() << ','
                << dof.normal.x() << ',' << dof.normal.y() << ',' << dof.normal.z() << ','
                << dof.tangent1.x() << ',' << dof.tangent1.y() << ','
                << dof.tangent1.z() << ',' << dof.tangent2.x() << ','
                << dof.tangent2.y() << ',' << dof.tangent2.z() << ','
                << dof.weight << '\n';
    }

    std::ofstream stencil_csv = open_output_file(
        output_dir / (stem + "_cauchy_stencils.csv"));
    stencil_csv << std::setprecision(17);
    stencil_csv << "q,patch_id,radius_over_h,incident_patch_count,"
                   "value_count,normal_count,value_patch_imbalance,"
                   "normal_patch_imbalance";
    for (int k = 0; k < stencils.value_count; ++k)
        stencil_csv << ",value_" << k;
    for (int k = 0; k < stencils.derivative_count; ++k)
        stencil_csv << ",normal_" << k;
    stencil_csv << '\n';
    for (int q = 0; q < static_cast<int>(stencils.rows.size()); ++q) {
        const CauchyStencil& stencil =
            stencils.rows[static_cast<std::size_t>(q)];
        stencil_csv << q << ','
                    << cloud.dofs[static_cast<std::size_t>(q)].patch_id << ','
                    << stencil.radius_over_h << ','
                    << stencil.incident_patch_count << ','
                    << stencil.value_ids.size() << ','
                    << stencil.derivative_ids.size() << ','
                    << stencil.value_patch_imbalance << ','
                    << stencil.derivative_patch_imbalance;
        for (int id : stencil.value_ids)
            stencil_csv << ',' << id;
        for (int k = static_cast<int>(stencil.value_ids.size());
             k < stencils.value_count; ++k) {
            stencil_csv << ',';
        }
        for (int id : stencil.derivative_ids)
            stencil_csv << ',' << id;
        for (int k = static_cast<int>(stencil.derivative_ids.size());
             k < stencils.derivative_count; ++k) {
            stencil_csv << ',';
        }
        stencil_csv << '\n';
    }
}

ReadinessResult run_readiness_case(GeometryKind kind,
                                   int N,
                                   const std::filesystem::path& output_dir,
                                   CauchyStencilPolicy3D cauchy_policy,
                                   int cauchy_value_count,
                                   int cauchy_normal_count)
{
    const double h = kBoxSide / static_cast<double>(N);
    CartesianGrid3D grid({kBoxMin, kBoxMin, kBoxMin},
                         {h, h, h},
                         {N, N, N},
                         DofLayout3D::Node);
    GeometryBundle geometry = make_geometry(kind, h);
    const SurfaceDofCloud surface_dofs =
        app3d::make_native_surface_dofs_3d(geometry.native_surface, h);
    const SurfaceCloudDiagnostics surface_diagnostics =
        validate_surface_dofs(surface_dofs, h);
    const CauchyStencilSet cauchy_stencils = build_cauchy_stencils(
        geometry.native_surface,
        surface_dofs,
        h,
        cauchy_value_count,
        cauchy_normal_count,
        cauchy_policy);
    write_surface_files(output_dir, geometry, N);
    write_panel_center_files(
        output_dir, geometry.name, N, surface_dofs, cauchy_stencils);

    ReadinessResult result;
    result.geometry = geometry.name;
    result.cauchy_policy = cauchy_policy_name(cauchy_policy);
    result.N = N;
    result.h = h;
    result.correction_panels = geometry.correction_interface.num_panels();
    result.correction_dofs = geometry.correction_interface.num_points();
    result.crossing_panels = geometry.crossing_interface.num_panels();
    result.feature_edges = geometry.feature_edges;
    result.feature_vertices = geometry.feature_vertices;
    result.correction_area = geometry.correction_interface.weights().sum();
    result.crossing_area = geometry.crossing_interface.weights().sum();
    result.normal_error = validate_interface(geometry.correction_interface);
    validate_interface(geometry.crossing_interface);
    result.surface_patches = static_cast<int>(surface_dofs.patches.size());
    result.surface_dofs = static_cast<int>(surface_dofs.dofs.size());
    result.surface_dof_area = surface_diagnostics.area;
    result.surface_area_relative_error = surface_diagnostics.area_relative_error;
    result.surface_spacing_min_over_h =
        surface_diagnostics.nearest_spacing_min_over_h;
    result.surface_spacing_mean_over_h =
        surface_diagnostics.nearest_spacing_mean_over_h;
    result.surface_spacing_max_over_h =
        surface_diagnostics.nearest_spacing_max_over_h;
    result.cauchy_value_neighbors = cauchy_stencils.value_count;
    result.cauchy_derivative_neighbors = cauchy_stencils.derivative_count;
    result.cauchy_value_neighbors_min = cauchy_stencils.value_count_min;
    result.cauchy_value_neighbors_max = cauchy_stencils.value_count_max;
    result.cauchy_derivative_neighbors_min =
        cauchy_stencils.derivative_count_min;
    result.cauchy_derivative_neighbors_max =
        cauchy_stencils.derivative_count_max;
    result.cauchy_radius_max_over_h = cauchy_stencils.radius_max_over_h;
    result.cauchy_radius_mean_over_h = cauchy_stencils.radius_mean_over_h;
    result.cauchy_incident_patches_min =
        cauchy_stencils.incident_patch_count_min;
    result.cauchy_incident_patches_max =
        cauchy_stencils.incident_patch_count_max;
    result.cauchy_value_patch_imbalance_max =
        cauchy_stencils.value_patch_imbalance_max;
    result.cauchy_derivative_patch_imbalance_max =
        cauchy_stencils.derivative_patch_imbalance_max;

    const auto dims = grid.dof_dims();
    Eigen::Vector3d grid_min(kBoxMin, kBoxMin, kBoxMin);
    Eigen::Vector3d grid_max(
        kBoxMin + static_cast<double>(dims[0] - 1) * h,
        kBoxMin + static_cast<double>(dims[1] - 1) * h,
        kBoxMin + static_cast<double>(dims[2] - 1) * h);
    double min_margin = std::numeric_limits<double>::infinity();
    for (int q = 0; q < geometry.crossing_interface.num_points(); ++q) {
        const Eigen::Vector3d point =
            geometry.crossing_interface.points().row(q).transpose();
        min_margin = std::min(min_margin, (point - grid_min).minCoeff());
        min_margin = std::min(min_margin, (grid_max - point).minCoeff());
    }
    result.min_box_margin_over_h = min_margin / h;
    if (result.min_box_margin_over_h < 2.0)
        throw std::runtime_error("surface is too close to the Cartesian box boundary");

    GridPair3D grid_pair(grid,
                         geometry.correction_interface,
                         geometry.crossing_interface);
    for (int n = 0; n < grid.num_dofs(); ++n) {
        const bool numerical_inside = grid_pair.domain_label(n) > 0;
        const auto coordinate = grid.coord(n);
        const Eigen::Vector3d point(
            coordinate[0], coordinate[1], coordinate[2]);
        const bool exact_inside = geometry.exact_inside(point);
        if (numerical_inside)
            ++result.interior_nodes;
        else
            ++result.exterior_nodes;
        if (numerical_inside != exact_inside) {
            ++result.label_mismatches;
            if (result.label_mismatches <= 5) {
                std::cout << "  [label mismatch] x=("
                          << point.x() << ',' << point.y() << ',' << point.z()
                          << ") numerical=" << numerical_inside
                          << " exact=" << exact_inside << '\n';
            }
        }
    }
    if (result.interior_nodes == 0 || result.exterior_nodes == 0)
        throw std::runtime_error("domain labeling did not produce both sides");
    if (result.label_mismatches > std::max(2, grid.num_dofs() / 100))
        throw std::runtime_error("more than one percent of grid labels are incorrect");

    const LaplaceCorrectionSupport3D support =
        build_laplace_correction_support_3d(
            grid_pair, "neumann exterior-zero-trace 3D geometry readiness");
    result.crossings = static_cast<int>(support.crossing_ops.size());
    result.correction_nodes = static_cast<int>(support.correction_nodes.size());
    if (result.crossings == 0
        || support.restrict_stencils.size()
           != static_cast<std::size_t>(geometry.correction_interface.num_points())) {
        throw std::runtime_error("failed to build fixed crossing/restrict routes");
    }

    std::set<std::pair<int, int>> crossing_edges;
    for (const LaplaceCrossingCorrectionOp& op : support.crossing_ops) {
        const std::pair<int, int> edge{
            std::min(op.rhs_node, op.correction_node),
            std::max(op.rhs_node, op.correction_node)};
        if (!crossing_edges.insert(edge).second)
            continue;
        const P2CrossingOwner3D owner =
            grid_pair.p2_crossing_owner_between(edge.first, edge.second);
        if (owner.status == P2CrossingOwnerStatus3D::ExactIntersection)
            ++result.exact_crossings;
        else if (owner.status == P2CrossingOwnerStatus3D::GapFallback)
            ++result.gap_crossings;
        else
            ++result.endpoint_crossings;
    }

    LaplaceQuadraticPatchCenterSpread3D spread(grid_pair, 0.0);
    LaplaceFftBulkSolverZfft3D bulk(
        grid, ZfftBcType::Dirichlet, 0.0, 2);
    LaplaceQuadraticPatchCenterRestrict3D restrict_op(grid_pair, 2);
    LaplacePotentialEval3D potentials(spread, bulk, restrict_op);

    std::vector<LaplaceJumpData3D> jumps(
        static_cast<std::size_t>(geometry.correction_interface.num_points()));
    for (LaplaceJumpData3D& jump : jumps) {
        jump.u_jump = 1.0;
        jump.un_jump = 0.0;
        jump.rhs_derivs = Eigen::VectorXd::Zero(1);
    }
    const LaplacePotentialEvalResult3D probe = potentials.evaluate(
        jumps, Eigen::VectorXd::Zero(grid.num_dofs()));
    for (int q = 0; q < probe.u_avg.size(); ++q) {
        const double interior_trace = probe.u_avg[q] + 0.5;
        const double exterior_trace = probe.u_avg[q] - 0.5;
        result.constant_A1_linf = std::max(
            result.constant_A1_linf, std::abs(1.0 - interior_trace));
        result.constant_exterior_trace_linf = std::max(
            result.constant_exterior_trace_linf, std::abs(exterior_trace));
        result.constant_interior_trace_linf = std::max(
            result.constant_interior_trace_linf, std::abs(interior_trace - 1.0));
    }
    for (int n = 0; n < grid.num_dofs(); ++n) {
        const double expected = grid_pair.domain_label(n) > 0 ? 1.0 : 0.0;
        result.constant_bulk_linf = std::max(
            result.constant_bulk_linf, std::abs(probe.u_bulk[n] - expected));
    }
    if (!std::isfinite(result.constant_A1_linf)
        || !std::isfinite(result.constant_bulk_linf)) {
        throw std::runtime_error("constant value-jump readiness probe produced NaN/Inf");
    }

    PanelCenterHarmonicJetKFBI3D harmonic_pipeline(
        grid,
        grid_pair,
        geometry.native_surface,
        geometry.correction_triangles,
        geometry.geometry_triangles,
        surface_dofs,
        cauchy_stencils);
    const ConditionStatistics3D condition_statistics = summarize_conditions(
        harmonic_pipeline.cauchy_condition_values());
    result.cauchy_condition_median = condition_statistics.median;
    result.cauchy_condition_p95 = condition_statistics.p95;
    result.cauchy_condition_max = condition_statistics.maximum;
    const Eigen::VectorXd constant_value =
        Eigen::VectorXd::Ones(harmonic_pipeline.surface_size());
    const Eigen::VectorXd zero_normal =
        Eigen::VectorXd::Zero(harmonic_pipeline.surface_size());
    const HarmonicJetField3D harmonic_probe =
        harmonic_pipeline.evaluate(constant_value, zero_normal);
    result.harmonic_constant_exterior_trace_linf =
        harmonic_pipeline.exterior_trace(
            harmonic_probe, constant_value, zero_normal).lpNorm<Eigen::Infinity>();
    result.harmonic_constant_interior_trace_linf =
        (harmonic_pipeline.interior_trace(
             harmonic_probe, constant_value, zero_normal).array() - 1.0)
        .matrix().lpNorm<Eigen::Infinity>();
    result.harmonic_constant_exterior_normal_linf =
        harmonic_pipeline.exterior_normal_trace(
            harmonic_probe, constant_value, zero_normal).lpNorm<Eigen::Infinity>();
    result.harmonic_constant_interior_normal_linf =
        harmonic_pipeline.interior_normal_trace(
            harmonic_probe, constant_value, zero_normal).lpNorm<Eigen::Infinity>();
    for (int n = 0; n < grid.num_dofs(); ++n) {
        const double expected = grid_pair.domain_label(n) > 0 ? 1.0 : 0.0;
        result.harmonic_constant_bulk_linf = std::max(
            result.harmonic_constant_bulk_linf,
            std::abs(harmonic_probe.potential[n] - expected));
    }
    if (!std::isfinite(result.harmonic_constant_exterior_trace_linf)
        || !std::isfinite(result.harmonic_constant_exterior_normal_linf)
        || !std::isfinite(result.harmonic_constant_bulk_linf)) {
        throw std::runtime_error(
            "harmonic-jet constant probe produced NaN/Inf");
    }
    result.neumann = run_neumann_case(grid, grid_pair, harmonic_pipeline);
    result.dirichlet_normal = run_dirichlet_normal_case(
        grid, grid_pair, harmonic_pipeline);

    std::cout << "[ready] " << geometry.name << " - " << geometry.description << '\n'
              << "  panel-center surface patches/dofs="
              << result.surface_patches << '/' << result.surface_dofs
              << " area=" << result.surface_dof_area
              << " area_rel_err=" << result.surface_area_relative_error << '\n'
              << "  nearest surface spacing/h min/mean/max="
              << result.surface_spacing_min_over_h << '/'
              << result.surface_spacing_mean_over_h << '/'
              << result.surface_spacing_max_over_h << '\n'
              << "  Cauchy policy=" << result.cauchy_policy
              << " requested values/normals="
              << result.cauchy_value_neighbors << '/'
              << result.cauchy_derivative_neighbors
              << " actual values min/max="
              << result.cauchy_value_neighbors_min << '/'
              << result.cauchy_value_neighbors_max
              << " normals min/max="
              << result.cauchy_derivative_neighbors_min << '/'
              << result.cauchy_derivative_neighbors_max
              << " radius/h mean/max=" << result.cauchy_radius_mean_over_h
              << '/' << result.cauchy_radius_max_over_h
              << " selected patches min/max="
              << result.cauchy_incident_patches_min << '/'
              << result.cauchy_incident_patches_max
              << " imbalance value/normal max="
              << result.cauchy_value_patch_imbalance_max << '/'
              << result.cauchy_derivative_patch_imbalance_max << '\n'
              << "  Cauchy condition median/p95/max="
              << result.cauchy_condition_median << '/'
              << result.cauchy_condition_p95 << '/'
              << result.cauchy_condition_max << '\n'
              << "  P2 correction panels/dofs=" << result.correction_panels
              << '/' << result.correction_dofs
              << " crossing_panels=" << result.crossing_panels
              << " area(correction/full)=" << result.correction_area
              << '/' << result.crossing_area << '\n'
              << "  features edges/vertices=" << result.feature_edges
              << '/' << result.feature_vertices
              << " grid inside/outside=" << result.interior_nodes
              << '/' << result.exterior_nodes
              << " label_mismatches=" << result.label_mismatches
              << " box_margin/h=" << result.min_box_margin_over_h << '\n'
              << "  fixed routes crossing_ops/correction_nodes="
              << result.crossings << '/' << result.correction_nodes
              << " owners exact/gap/endpoint=" << result.exact_crossings
              << '/' << result.gap_crossings
              << '/' << result.endpoint_crossings << '\n'
              << "  constant [u]=1 probe: |A1|inf="
              << result.constant_A1_linf
              << " exterior_trace=" << result.constant_exterior_trace_linf
              << " interior_trace_err=" << result.constant_interior_trace_linf
              << " bulk_err=" << result.constant_bulk_linf << '\n'
              << "  cubic harmonic-jet [u]=1 probe: exterior/interior trace="
              << result.harmonic_constant_exterior_trace_linf << '/'
              << result.harmonic_constant_interior_trace_linf
              << " exterior/interior normal="
              << result.harmonic_constant_exterior_normal_linf << '/'
              << result.harmonic_constant_interior_normal_linf
              << " bulk_err=" << result.harmonic_constant_bulk_linf << '\n'
              << "  [neumann] converged/iterations="
              << result.neumann.converged << '/' << result.neumann.iterations
              << " gmres=" << result.neumann.gmres_relative_residual
              << " operator=" << result.neumann.operator_residual_linf
              << " exterior_trace=" << result.neumann.exterior_condition_linf
              << " route_mismatch=" << result.neumann.route_mismatch_linf
              << " density_linf/l2=" << result.neumann.density_linf << '/'
              << result.neumann.density_l2
              << " interior_linf/l2=" << result.neumann.interior_linf << '/'
              << result.neumann.interior_l2
              << " exterior_bulk_linf/l2="
              << result.neumann.exterior_bulk_linf << '/'
              << result.neumann.exterior_bulk_l2
              << " flux_mean=" << result.neumann.data_weighted_mean
              << " seconds=" << result.neumann.seconds << '\n'
              << "  [dirichlet-normal] converged/iterations="
              << result.dirichlet_normal.converged << '/'
              << result.dirichlet_normal.iterations
              << " gmres="
              << result.dirichlet_normal.gmres_relative_residual
              << " operator="
              << result.dirichlet_normal.operator_residual_linf
              << " exterior_normal="
              << result.dirichlet_normal.exterior_condition_linf
              << " interior_value_res="
              << result.dirichlet_normal.boundary_residual_linf
              << " route_mismatch="
              << result.dirichlet_normal.route_mismatch_linf
              << " density_linf/l2="
              << result.dirichlet_normal.density_linf << '/'
              << result.dirichlet_normal.density_l2
              << " interior_linf/l2="
              << result.dirichlet_normal.interior_linf << '/'
              << result.dirichlet_normal.interior_l2
              << " exterior_bulk_linf/l2="
              << result.dirichlet_normal.exterior_bulk_linf << '/'
              << result.dirichlet_normal.exterior_bulk_l2
              << " seconds=" << result.dirichlet_normal.seconds << '\n';
    return result;
}

void write_summary(const std::filesystem::path& output_dir,
                   const std::vector<ReadinessResult>& results)
{
    std::filesystem::create_directories(output_dir);
    if (results.empty())
        return;
    auto write_csv = [&](const std::filesystem::path& path,
                         const std::vector<ReadinessResult>& rows) {
        std::ofstream csv = open_output_file(path);
        csv << std::setprecision(17);
        csv << "geometry,cauchy_policy,N,h,correction_panels,correction_dofs,crossing_panels,"
               "feature_edges,feature_vertices,interior_nodes,exterior_nodes,"
               "label_mismatches,crossing_ops,exact_crossings,gap_crossings,"
               "endpoint_crossings,correction_nodes,correction_area,crossing_area,"
               "normal_error,box_margin_over_h,A1_linf,exterior_trace_linf,"
               "interior_trace_error,bulk_error,surface_patches,surface_dofs,"
               "harmonic_exterior_trace_linf,harmonic_interior_trace_linf,"
               "harmonic_exterior_normal_linf,harmonic_interior_normal_linf,"
               "harmonic_bulk_linf,"
               "surface_dof_area,surface_area_relative_error,"
               "surface_spacing_min_over_h,surface_spacing_mean_over_h,"
               "surface_spacing_max_over_h,cauchy_value_neighbors,"
               "cauchy_derivative_neighbors,cauchy_value_neighbors_min,"
               "cauchy_value_neighbors_max,cauchy_derivative_neighbors_min,"
               "cauchy_derivative_neighbors_max,cauchy_radius_max_over_h,"
               "cauchy_radius_mean_over_h,cauchy_incident_patches_min,"
               "cauchy_incident_patches_max,cauchy_value_patch_imbalance_max,"
               "cauchy_derivative_patch_imbalance_max,cauchy_condition_median,"
               "cauchy_condition_p95,cauchy_condition_max\n";
        for (const ReadinessResult& row : rows)
            csv << row.geometry << ',' << row.cauchy_policy << ','
                << row.N << ',' << row.h << ','
                << row.correction_panels << ',' << row.correction_dofs << ','
                << row.crossing_panels << ',' << row.feature_edges << ','
                << row.feature_vertices << ',' << row.interior_nodes << ','
                << row.exterior_nodes << ',' << row.label_mismatches << ','
                << row.crossings << ',' << row.exact_crossings << ','
                << row.gap_crossings << ',' << row.endpoint_crossings << ','
                << row.correction_nodes << ',' << row.correction_area << ','
                << row.crossing_area << ',' << row.normal_error << ','
                << row.min_box_margin_over_h << ',' << row.constant_A1_linf << ','
                << row.constant_exterior_trace_linf << ','
                << row.constant_interior_trace_linf << ','
                << row.constant_bulk_linf << ',' << row.surface_patches << ','
                << row.surface_dofs << ','
                << row.harmonic_constant_exterior_trace_linf << ','
                << row.harmonic_constant_interior_trace_linf << ','
                << row.harmonic_constant_exterior_normal_linf << ','
                << row.harmonic_constant_interior_normal_linf << ','
                << row.harmonic_constant_bulk_linf << ','
                << row.surface_dof_area << ','
                << row.surface_area_relative_error << ','
                << row.surface_spacing_min_over_h << ','
                << row.surface_spacing_mean_over_h << ','
                << row.surface_spacing_max_over_h << ','
                << row.cauchy_value_neighbors << ','
                << row.cauchy_derivative_neighbors << ','
                << row.cauchy_value_neighbors_min << ','
                << row.cauchy_value_neighbors_max << ','
                << row.cauchy_derivative_neighbors_min << ','
                << row.cauchy_derivative_neighbors_max << ','
                << row.cauchy_radius_max_over_h << ','
                << row.cauchy_radius_mean_over_h << ','
                << row.cauchy_incident_patches_min << ','
                << row.cauchy_incident_patches_max << ','
                << row.cauchy_value_patch_imbalance_max << ','
                << row.cauchy_derivative_patch_imbalance_max << ','
                << row.cauchy_condition_median << ','
                << row.cauchy_condition_p95 << ','
                << row.cauchy_condition_max << '\n';
    };
    write_csv(output_dir / "geometry_readiness.csv", results);
    std::set<int> levels;
    for (const ReadinessResult& row : results)
        levels.insert(row.N);
    for (int level : levels) {
        std::vector<ReadinessResult> level_rows;
        for (const ReadinessResult& row : results) {
            if (row.N == level)
                level_rows.push_back(row);
        }
        write_csv(output_dir / (
            "geometry_readiness_N" + std::to_string(level) + ".csv"),
            level_rows);
    }
}

double observed_order(double coarse_error,
                      double fine_error,
                      double coarse_h,
                      double fine_h)
{
    if (!(coarse_error > 0.0) || !(fine_error > 0.0)
        || !(coarse_h > fine_h)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::log(coarse_error / fine_error)
         / std::log(coarse_h / fine_h);
}

void write_solve_summaries(const std::filesystem::path& output_dir,
                           const std::vector<ReadinessResult>& results)
{
    using MetricSelector =
        std::function<const SolveMetrics3D&(const ReadinessResult&)>;
    auto write_csv = [&](const std::filesystem::path& path,
                         const MetricSelector& select_metric) {
        std::vector<const ReadinessResult*> rows;
        rows.reserve(results.size());
        for (const ReadinessResult& result : results)
            rows.push_back(&result);
        std::sort(rows.begin(), rows.end(),
                  [](const ReadinessResult* a, const ReadinessResult* b) {
                      return std::tie(a->geometry, a->N)
                           < std::tie(b->geometry, b->N);
                  });

        std::ofstream csv = open_output_file(path);
        csv << std::setprecision(17);
        csv << "geometry,cauchy_policy,N,h,dofs,formulation,iterations,converged,seconds,"
               "gmres_relative_residual,operator_residual_linf,"
               "exterior_condition_linf,boundary_residual_linf,"
               "route_mismatch_linf,data_weighted_mean,density_weighted_mean,"
               "density_linf,density_l2,density_order_linf,density_order_l2,"
               "interior_linf,interior_l2,interior_order_linf,"
               "interior_order_l2,exterior_bulk_linf,exterior_bulk_l2,"
               "exterior_bulk_order_linf,exterior_bulk_order_l2,"
               "constant_shift\n";
        std::map<std::string, const ReadinessResult*> previous;
        for (const ReadinessResult* row : rows) {
            const SolveMetrics3D& metric = select_metric(*row);
            double density_order_linf =
                std::numeric_limits<double>::quiet_NaN();
            double density_order_l2 =
                std::numeric_limits<double>::quiet_NaN();
            double interior_order_linf =
                std::numeric_limits<double>::quiet_NaN();
            double interior_order_l2 =
                std::numeric_limits<double>::quiet_NaN();
            double exterior_order_linf =
                std::numeric_limits<double>::quiet_NaN();
            double exterior_order_l2 =
                std::numeric_limits<double>::quiet_NaN();
            const auto found = previous.find(row->geometry);
            if (found != previous.end()) {
                const ReadinessResult& coarse = *found->second;
                const SolveMetrics3D& coarse_metric = select_metric(coarse);
                density_order_linf = observed_order(
                    coarse_metric.density_linf, metric.density_linf,
                    coarse.h, row->h);
                density_order_l2 = observed_order(
                    coarse_metric.density_l2, metric.density_l2,
                    coarse.h, row->h);
                interior_order_linf = observed_order(
                    coarse_metric.interior_linf, metric.interior_linf,
                    coarse.h, row->h);
                interior_order_l2 = observed_order(
                    coarse_metric.interior_l2, metric.interior_l2,
                    coarse.h, row->h);
                exterior_order_linf = observed_order(
                    coarse_metric.exterior_bulk_linf,
                    metric.exterior_bulk_linf, coarse.h, row->h);
                exterior_order_l2 = observed_order(
                    coarse_metric.exterior_bulk_l2,
                    metric.exterior_bulk_l2, coarse.h, row->h);
            }
            csv << row->geometry << ',' << row->cauchy_policy << ','
                << row->N << ',' << row->h << ','
                << row->surface_dofs << ',' << metric.formulation << ','
                << metric.iterations << ',' << metric.converged << ','
                << metric.seconds << ',' << metric.gmres_relative_residual << ','
                << metric.operator_residual_linf << ','
                << metric.exterior_condition_linf << ','
                << metric.boundary_residual_linf << ','
                << metric.route_mismatch_linf << ','
                << metric.data_weighted_mean << ','
                << metric.density_weighted_mean << ','
                << metric.density_linf << ',' << metric.density_l2 << ','
                << density_order_linf << ',' << density_order_l2 << ','
                << metric.interior_linf << ',' << metric.interior_l2 << ','
                << interior_order_linf << ',' << interior_order_l2 << ','
                << metric.exterior_bulk_linf << ','
                << metric.exterior_bulk_l2 << ','
                << exterior_order_linf << ',' << exterior_order_l2 << ','
                << metric.constant_shift << '\n';
            previous[row->geometry] = row;
        }
    };

    write_csv(output_dir / "neumann_results.csv",
              [](const ReadinessResult& row) -> const SolveMetrics3D& {
                  return row.neumann;
              });
    write_csv(output_dir / "dirichlet_normal_results.csv",
              [](const ReadinessResult& row) -> const SolveMetrics3D& {
                  return row.dirichlet_normal;
              });
}

void print_usage(const char* executable)
{
    std::cout
        << "usage: " << executable
        << " [torus|cylinder|l_prism|all] [N ...]\n"
        << "  Each N must be a power of two and at least 16 (default: 16).\n"
        << "  This stage builds native NURBS parameter-cell-center surface\n"
        << "  unknowns, topology-filtered 48/28 Cauchy stencils, validates\n"
        << "  fixed transfer routes, and executes the Neumann value-jump and\n"
        << "  Dirichlet normal-jump harmonic-jet GMRES formulations.\n"
        << "  KFBIM_3D_CAUCHY_POLICY selects topological_nearest (default),\n"
        << "  same_patch, or balanced_patches.\n"
        << "  KFBIM_3D_CAUCHY_VALUE_COUNT and\n"
        << "  KFBIM_3D_CAUCHY_NORMAL_COUNT select positive stencil counts\n"
        << "  (defaults: 48 and 28; normal count may not exceed value count).\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        std::cout << std::scientific << std::setprecision(6) << std::unitbuf;
        std::string selection = "all";
        std::vector<int> levels = {16};
        if (argc >= 2)
            selection = argv[1];
        if (selection == "--help" || selection == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (argc >= 3) {
            levels.clear();
            for (int argument = 2; argument < argc; ++argument) {
                const int N = std::stoi(argv[argument]);
                if (N < 16 || !is_power_of_two(N)) {
                    throw std::invalid_argument(
                        "each N must be a power of two and at least 16");
                }
                levels.push_back(N);
            }
        }
        const CauchyStencilPolicy3D cauchy_policy = selected_cauchy_policy();
        const int cauchy_value_count = positive_environment_integer(
            "KFBIM_3D_CAUCHY_VALUE_COUNT", kCauchyValueNeighborCount);
        const int cauchy_normal_count = positive_environment_integer(
            "KFBIM_3D_CAUCHY_NORMAL_COUNT", kCauchyDerivativeNeighborCount);
        if (cauchy_normal_count > cauchy_value_count) {
            throw std::invalid_argument(
                "KFBIM_3D_CAUCHY_NORMAL_COUNT may not exceed "
                "KFBIM_3D_CAUCHY_VALUE_COUNT");
        }

        std::vector<GeometryKind> geometries;
        if (selection == "all") {
            geometries = {GeometryKind::Torus,
                          GeometryKind::HollowCylinder,
                          GeometryKind::LPrism};
        } else {
            geometries = {parse_geometry(selection)};
        }

#ifdef KFBIM_APP_OUTPUT_DIR
        std::filesystem::path output_dir =
            std::filesystem::path(KFBIM_APP_OUTPUT_DIR)
            / "neumann_exterior_zero_trace_3d";
#else
        std::filesystem::path output_dir =
            "output/neumann_exterior_zero_trace_3d";
#endif
        if (cauchy_policy != CauchyStencilPolicy3D::TopologicalNearest)
            output_dir /= cauchy_policy_name(cauchy_policy);
        if (cauchy_value_count != kCauchyValueNeighborCount
            || cauchy_normal_count != kCauchyDerivativeNeighborCount) {
            output_dir /= "v" + std::to_string(cauchy_value_count) + "_n"
                          + std::to_string(cauchy_normal_count);
        }

        std::cout << "KFBI3D harmonic-jet convergence study\n"
                  << "  Neumann target: exterior value trace = 0\n"
                  << "  Dirichlet target: exterior normal trace = 0\n"
                  << "  current stage: native NURBS surface DOFs + "
                     "topological Cauchy neighborhoods + G1 parameter-owned routes\n"
                  << "  cauchy_policy=" << cauchy_policy_name(cauchy_policy)
                  << '\n'
                  << "  cauchy_degree=" << kCauchyPolynomialDegree
                  << " restrict_grid_degree=" << kRestrictGridDegree
                  << " restrict_normal_degree=" << kRestrictNormalDegree
                  << '\n'
                  << "  cauchy_counts=" << cauchy_value_count << '/'
                  << cauchy_normal_count << '\n'
                  << "  levels=";
        for (std::size_t index = 0; index < levels.size(); ++index) {
            if (index != 0)
                std::cout << ',';
            std::cout << levels[index];
        }
        std::cout << '\n';

        std::vector<ReadinessResult> results;
        for (int N : levels) {
            for (GeometryKind geometry : geometries)
                results.push_back(run_readiness_case(
                    geometry,
                    N,
                    output_dir,
                    cauchy_policy,
                    cauchy_value_count,
                    cauchy_normal_count));
        }
        write_summary(output_dir, results);
        write_solve_summaries(output_dir, results);
        for (const ReadinessResult& result : results) {
            if (!result.neumann.converged
                || !result.dirichlet_normal.converged) {
                throw std::runtime_error(
                    "at least one requested 3D GMRES solve did not converge");
            }
        }

        std::cout << "Geometry checks and both GMRES formulations completed.\n"
                  << "Output: " << output_dir.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
