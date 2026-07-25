#include "neumann_edge_continuity_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d {
namespace {

using geometry3d::NurbsPatchEdgeInterval3D;

void require_finite(double value, const char* message)
{
    if (!std::isfinite(value)) throw std::runtime_error(message);
}

double edge_parameter(const NurbsPatchEdgeInterval3D& interval, double s)
{
    return interval.begin + s * (interval.end - interval.begin);
}

std::pair<double, double> edge_uv(const geometry3d::NurbsSurfacePatch3D& patch,
                                  PatchEdge3D edge,
                                  double parameter)
{
    switch (edge) {
    case PatchEdge3D::UMin: return {patch.domain_start_u(), parameter};
    case PatchEdge3D::UMax: return {patch.domain_end_u(), parameter};
    case PatchEdge3D::VMin: return {parameter, patch.domain_start_v()};
    case PatchEdge3D::VMax: return {parameter, patch.domain_end_v()};
    }
    throw std::runtime_error("unknown NURBS patch edge");
}

Eigen::Vector3d edge_point(const NativeNurbsSurface3D& surface,
                           const NurbsPatchEdgeInterval3D& interval,
                           double parameter)
{
    const auto& patch = surface.patches.at(static_cast<std::size_t>(interval.patch));
    const auto uv = edge_uv(patch, interval.edge, parameter);
    return patch.evaluate(uv.first, uv.second);
}

double edge_speed(const NativeNurbsSurface3D& surface,
                  const NurbsPatchEdgeInterval3D& interval,
                  double parameter)
{
    const auto& patch = surface.patches.at(static_cast<std::size_t>(interval.patch));
    const auto uv = edge_uv(patch, interval.edge, parameter);
    const auto derivatives = patch.evaluate_with_derivatives(uv.first, uv.second);
    const double speed = (interval.edge == PatchEdge3D::UMin || interval.edge == PatchEdge3D::UMax)
        ? derivatives.dv.norm() : derivatives.du.norm();
    if (!std::isfinite(speed) || speed <= 0.0)
        throw std::runtime_error("NURBS edge has invalid physical speed");
    return speed;
}

double edge_length(const NativeNurbsSurface3D& surface,
                   const NurbsPatchEdgeInterval3D& interval)
{
    constexpr std::array<double, 8> x = {{-0.9602898564975363, -0.7966664774136267,
        -0.5255324099163290, -0.1834346424956498, 0.1834346424956498,
         0.5255324099163290, 0.7966664774136267, 0.9602898564975363}};
    constexpr std::array<double, 8> w = {{0.1012285362903763, 0.2223810344533745,
        0.3137066458778873, 0.3626837833783620, 0.3626837833783620,
        0.3137066458778873, 0.2223810344533745, 0.1012285362903763}};
    const double half = 0.5 * (interval.end - interval.begin);
    const double middle = 0.5 * (interval.begin + interval.end);
    double result = 0.0;
    for (std::size_t q = 0; q < x.size(); ++q)
        result += w[q] * edge_speed(surface, interval, middle + half * x[q]);
    return half * result;
}

std::vector<int> contiguous_indices(int count, int wanted, double coordinate,
                                    double first_center, double spacing)
{
    const int use = std::min(count, wanted);
    int first = static_cast<int>(std::floor((coordinate - first_center) / spacing))
        - (use - 1) / 2;
    first = std::max(0, std::min(first, count - use));
    std::vector<int> indices;
    for (int q = 0; q < use; ++q) indices.push_back(first + q);
    return indices;
}

std::vector<double> lagrange_weights(double value, const std::vector<double>& nodes)
{
    std::vector<double> weights(nodes.size(), 1.0);
    for (std::size_t a = 0; a < nodes.size(); ++a) {
        for (std::size_t b = 0; b < nodes.size(); ++b) {
            if (a != b) weights[a] *= (value - nodes[b]) / (nodes[a] - nodes[b]);
        }
        require_finite(weights[a], "non-finite edge interpolation weight");
    }
    return weights;
}

std::vector<std::pair<int, double>> edge_trace_coefficients(
    const NativeNurbsSurface3D& surface, const SurfaceDofCloud3D& cloud,
    const NurbsPatchEdgeInterval3D& interval, double parameter, bool& reduced_order)
{
    const auto& patch = surface.patches.at(static_cast<std::size_t>(interval.patch));
    const auto& tensor = cloud.patches.at(static_cast<std::size_t>(interval.patch));
    if (tensor.nu <= 0 || tensor.nv <= 0) throw std::invalid_argument("invalid native tensor grid");
    const bool u_edge = interval.edge == PatchEdge3D::UMin || interval.edge == PatchEdge3D::UMax;
    const int along_count = u_edge ? tensor.nv : tensor.nu;
    const int inward_count = u_edge ? tensor.nu : tensor.nv;
    reduced_order = along_count < 4 || inward_count < 3;
    const double along0 = u_edge ? patch.domain_start_v() : patch.domain_start_u();
    const double along1 = u_edge ? patch.domain_end_v() : patch.domain_end_u();
    const double along_h = (along1 - along0) / static_cast<double>(along_count);
    const auto along = contiguous_indices(along_count, 4, parameter, along0 + 0.5 * along_h, along_h);
    std::vector<double> along_nodes;
    for (int index : along) along_nodes.push_back(along0 + (index + 0.5) * along_h);
    const auto along_weights = lagrange_weights(parameter, along_nodes);
    std::vector<int> inward;
    const int inward_use = std::min(inward_count, 3);
    for (int q = 0; q < inward_use; ++q) {
        const bool minimum = interval.edge == PatchEdge3D::UMin || interval.edge == PatchEdge3D::VMin;
        inward.push_back(minimum ? q : inward_count - 1 - q);
    }
    const double fixed0 = u_edge ? patch.domain_start_u() : patch.domain_start_v();
    const double fixed1 = u_edge ? patch.domain_end_u() : patch.domain_end_v();
    const double fixed_h = (fixed1 - fixed0) / static_cast<double>(inward_count);
    std::vector<double> inward_nodes;
    for (int index : inward) inward_nodes.push_back(fixed0 + (index + 0.5) * fixed_h);
    const double boundary = (interval.edge == PatchEdge3D::UMin || interval.edge == PatchEdge3D::VMin) ? fixed0 : fixed1;
    const auto inward_weights = lagrange_weights(boundary, inward_nodes);
    std::vector<std::pair<int, double>> result;
    for (std::size_t a = 0; a < along.size(); ++a) for (std::size_t b = 0; b < inward.size(); ++b) {
        const int i = u_edge ? inward[b] : along[a];
        const int j = u_edge ? along[a] : inward[b];
        result.emplace_back(tensor.dof_index(i, j), along_weights[a] * inward_weights[b]);
    }
    return result;
}

} // namespace

const char* neumann_density_space_name_3d(NeumannDensitySpace3D mode)
{
    switch (mode) {
    case NeumannDensitySpace3D::PatchIndependent: return "patch_independent";
    case NeumannDensitySpace3D::NonG1EdgeProjected: return "non_g1_edge_projected";
    }
    return "unknown";
}

NeumannEdgeConstraintSet3D build_neumann_edge_constraints_3d(
    const NativeNurbsSurface3D& surface, const SurfaceDofCloud3D& cloud, double h)
{
    if (!std::isfinite(h) || h <= 0.0) throw std::invalid_argument("edge constraints require positive h");
    if (cloud.patches.size() != surface.patches.size()) throw std::invalid_argument("native tensor grids do not match surface patches");
    NeumannEdgeConstraintSet3D result;
    result.density_size = static_cast<int>(cloud.dofs.size());
    std::vector<Eigen::Triplet<double>> entries;
    std::vector<double> weights;
    for (int c = 0; c < static_cast<int>(surface.geometric_connections.size()); ++c) {
        const auto& connection = surface.geometric_connections[static_cast<std::size_t>(c)];
        if (connection.g1) continue;
        ++result.non_g1_connection_count;
        const int count = std::max(2, static_cast<int>(std::ceil(edge_length(surface, connection.first) / h)));
        const double parameter_width = (connection.first.end - connection.first.begin) / count;
        for (int q = 0; q < count; ++q) {
            const double s = (q + 0.5) / static_cast<double>(count);
            const double first_parameter = edge_parameter(connection.first, s);
            const double second_parameter = edge_parameter(connection.second, connection.reversed ? 1.0 - s : s);
            bool first_reduced = false, second_reduced = false;
            const auto first = edge_trace_coefficients(surface, cloud, connection.first, first_parameter, first_reduced);
            const auto second = edge_trace_coefficients(surface, cloud, connection.second, second_parameter, second_reduced);
            std::map<int, double> row;
            for (const auto& item : first) row[item.first] += item.second;
            for (const auto& item : second) row[item.first] -= item.second;
            double sum = 0.0;
            for (const auto& item : row) { require_finite(item.second, "non-finite edge row coefficient"); sum += item.second; }
            if (std::abs(sum) > 5.0e-13) throw std::runtime_error("edge trace row does not preserve constants");
            const int row_index = static_cast<int>(result.samples.size());
            for (const auto& item : row) if (item.second != 0.0) entries.emplace_back(row_index, item.first, item.second);
            const double weight = edge_speed(surface, connection.first, first_parameter) * parameter_width;
            if (!std::isfinite(weight) || weight <= 0.0) throw std::runtime_error("invalid edge quadrature weight");
            const double gap = (edge_point(surface, connection.first, first_parameter)
                              - edge_point(surface, connection.second, second_parameter)).norm();
            result.samples.push_back({c, q, count, connection.first.patch, connection.second.patch,
                connection.first.edge, connection.second.edge, s, first_parameter, second_parameter,
                weight, gap, first_reduced, second_reduced});
            weights.push_back(weight);
            if (first_reduced || second_reduced) ++result.reduced_order_row_count;
        }
    }
    result.matrix.resize(static_cast<Eigen::Index>(result.samples.size()), result.density_size);
    result.matrix.setFromTriplets(entries.begin(), entries.end());
    result.quadrature_weights = Eigen::Map<const Eigen::VectorXd>(weights.data(), static_cast<Eigen::Index>(weights.size()));
    return result;
}

Eigen::VectorXd apply_neumann_edge_constraints_3d(const NeumannEdgeConstraintSet3D& constraints,
                                                   const Eigen::VectorXd& density)
{
    if (density.size() != constraints.density_size || !density.allFinite())
        throw std::invalid_argument("edge constraint density has invalid size or values");
    return constraints.matrix * density;
}

double neumann_edge_mismatch_linf_3d(const NeumannEdgeConstraintSet3D& constraints,
                                      const Eigen::VectorXd& density)
{
    const Eigen::VectorXd residual = apply_neumann_edge_constraints_3d(constraints, density);
    return residual.size() == 0 ? 0.0 : residual.lpNorm<Eigen::Infinity>();
}

double neumann_edge_mismatch_weighted_rms_3d(const NeumannEdgeConstraintSet3D& constraints,
                                              const Eigen::VectorXd& density)
{
    const Eigen::VectorXd residual = apply_neumann_edge_constraints_3d(constraints, density);
    if (residual.size() != constraints.quadrature_weights.size()) throw std::invalid_argument("edge quadrature weights have invalid size");
    const double denominator = constraints.quadrature_weights.sum();
    if (residual.size() == 0) return 0.0;
    if (!std::isfinite(denominator) || denominator <= 0.0 || !constraints.quadrature_weights.allFinite()) throw std::invalid_argument("edge quadrature weights are invalid");
    const double value = std::sqrt((constraints.quadrature_weights.array() * residual.array().square()).sum() / denominator);
    require_finite(value, "edge mismatch is non-finite");
    return value;
}

} // namespace kfbim::app3d