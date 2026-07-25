#include "neumann_edge_continuity_3d.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

#include <Eigen/SparseQR>

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

namespace {

void validate_projector_density(const Eigen::VectorXd& density, int expected_size)
{
    if (density.size() != expected_size || !density.allFinite())
        throw std::invalid_argument("edge projector density has invalid size or values");
}

} // namespace

NeumannEdgeContinuityProjector3D::NeumannEdgeContinuityProjector3D(
    const NeumannEdgeConstraintSet3D& constraints,
    const SurfaceDofCloud3D& cloud,
    double rank_tolerance)
    : density_size_(constraints.density_size),
      physical_constraints_(constraints.matrix),
      edge_quadrature_weights_(constraints.quadrature_weights)
{
    if (!std::isfinite(rank_tolerance) || rank_tolerance <= 0.0)
        throw std::invalid_argument("edge projector rank tolerance must be positive and finite");
    if (density_size_ != static_cast<int>(cloud.dofs.size()) || density_size_ < 0
        || physical_constraints_.cols() != density_size_
        || physical_constraints_.rows() != edge_quadrature_weights_.size())
        throw std::invalid_argument("edge projector constraint dimensions are invalid");
    diagnostics_.input_constraint_count = static_cast<int>(physical_constraints_.rows());
    diagnostics_.rank_tolerance = rank_tolerance;
    inverse_surface_mass_.resize(density_size_);
    for (int q = 0; q < density_size_; ++q) {
        const double mass = cloud.dofs[static_cast<std::size_t>(q)].weight;
        if (!std::isfinite(mass) || mass <= 0.0)
            throw std::invalid_argument("edge projector surface mass is invalid");
        inverse_surface_mass_[q] = 1.0 / mass;
    }
    for (Eigen::Index row = 0; row < edge_quadrature_weights_.size(); ++row)
        if (!std::isfinite(edge_quadrature_weights_[row]) || edge_quadrature_weights_[row] <= 0.0)
            throw std::invalid_argument("edge projector quadrature weight is invalid");
    for (int row = 0; row < physical_constraints_.outerSize(); ++row)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(physical_constraints_, row); it; ++it)
            if (!std::isfinite(it.value()))
                throw std::invalid_argument("edge projector constraint coefficient is invalid");

    const Eigen::VectorXd constant_mismatch = physical_constraints_ * Eigen::VectorXd::Ones(density_size_);
    diagnostics_.constant_constraint_defect = constant_mismatch.size() == 0 ? 0.0 : constant_mismatch.lpNorm<Eigen::Infinity>();
    if (!std::isfinite(diagnostics_.constant_constraint_defect))
        throw std::invalid_argument("edge projector constant constraint defect is invalid");
    if (physical_constraints_.rows() == 0)
        return;

    const auto setup_start = std::chrono::steady_clock::now();
    std::vector<Eigen::Triplet<double>> scaled_entries;
    scaled_entries.reserve(static_cast<std::size_t>(physical_constraints_.nonZeros()));
    for (int row = 0; row < physical_constraints_.outerSize(); ++row) {
        const double row_scale = std::sqrt(edge_quadrature_weights_[row]);
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(physical_constraints_, row); it; ++it)
            scaled_entries.emplace_back(row, it.col(), row_scale * it.value() * std::sqrt(inverse_surface_mass_[it.col()]));
    }
    Eigen::SparseMatrix<double> scaled_constraint(physical_constraints_.rows(), physical_constraints_.cols());
    scaled_constraint.setFromTriplets(scaled_entries.begin(), scaled_entries.end());
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr;
    qr.setPivotThreshold(rank_tolerance);
    qr.compute(scaled_constraint.transpose());
    if (qr.info() != Eigen::Success)
        throw std::runtime_error("edge projector sparse QR factorization failed");
    const int rank = static_cast<int>(qr.rank());
    if (rank < 0 || rank > physical_constraints_.rows())
        throw std::runtime_error("edge projector sparse QR rank is invalid");
    diagnostics_.retained_constraint_rank = rank;
    retained_original_rows_.reserve(static_cast<std::size_t>(rank));
    for (int q = 0; q < rank; ++q) {
        const int row = qr.colsPermutation().indices()[q];
        if (row < 0 || row >= physical_constraints_.rows())
            throw std::runtime_error("edge projector sparse QR selected an invalid row");
        retained_original_rows_.push_back(row);
    }
    std::sort(retained_original_rows_.begin(), retained_original_rows_.end());
    if (std::adjacent_find(retained_original_rows_.begin(), retained_original_rows_.end()) != retained_original_rows_.end())
        throw std::runtime_error("edge projector sparse QR selected duplicate rows");
    if (rank == 0) {
        diagnostics_.factorization_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - setup_start).count();
        return;
    }

    std::vector<Eigen::Triplet<double>> retained_entries;
    std::vector<std::vector<std::pair<int, double>>> retained_rows;
    retained_rows.reserve(static_cast<std::size_t>(rank));
    for (int retained = 0; retained < rank; ++retained) {
        const int source = retained_original_rows_[static_cast<std::size_t>(retained)];
        const double row_scale = std::sqrt(edge_quadrature_weights_[source]);
        std::vector<std::pair<int, double>> row_entries;
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(physical_constraints_, source); it; ++it) {
            const double value = row_scale * it.value();
            retained_entries.emplace_back(retained, it.col(), value);
            row_entries.emplace_back(it.col(), value);
        }
        retained_rows.push_back(std::move(row_entries));
    }
    retained_scaled_constraints_.resize(rank, density_size_);
    retained_scaled_constraints_.setFromTriplets(retained_entries.begin(), retained_entries.end());
    Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(rank, rank);
    for (int first = 0; first < rank; ++first) for (int second = first; second < rank; ++second) {
        const auto& a = retained_rows[static_cast<std::size_t>(first)];
        const auto& b = retained_rows[static_cast<std::size_t>(second)];
        std::size_t ia = 0, ib = 0;
        double value = 0.0;
        while (ia < a.size() && ib < b.size()) {
            if (a[ia].first < b[ib].first) ++ia;
            else if (b[ib].first < a[ia].first) ++ib;
            else { value += a[ia].second * b[ib].second * inverse_surface_mass_[a[ia].first]; ++ia; ++ib; }
        }
        gram(first, second) = value;
        gram(second, first) = value;
    }
    if (!gram.allFinite())
        throw std::runtime_error("edge projector Gram matrix is non-finite");
    gram_factor_.compute(gram);
    if (gram_factor_.info() != Eigen::Success || !gram_factor_.vectorD().allFinite()
        || (gram_factor_.vectorD().array() <= 0.0).any())
        throw std::runtime_error("edge projector Gram factorization is not positive definite");
    diagnostics_.factorization_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - setup_start).count();
}

int NeumannEdgeContinuityProjector3D::density_size() const { return density_size_; }
int NeumannEdgeContinuityProjector3D::constraint_count() const { return diagnostics_.input_constraint_count; }
int NeumannEdgeContinuityProjector3D::retained_rank() const { return diagnostics_.retained_constraint_rank; }

Eigen::VectorXd NeumannEdgeContinuityProjector3D::project(const Eigen::VectorXd& density) const
{
    validate_projector_density(density, density_size_);
    if (retained_rank() == 0)
        return density;
    const Eigen::VectorXd residual = retained_scaled_constraints_ * density;
    const Eigen::VectorXd coefficients = gram_factor_.solve(residual);
    if (gram_factor_.info() != Eigen::Success || !coefficients.allFinite())
        throw std::runtime_error("edge projector Gram solve failed");
    return density - inverse_surface_mass_.asDiagonal() * (retained_scaled_constraints_.transpose() * coefficients);
}

Eigen::VectorXd NeumannEdgeContinuityProjector3D::complement(const Eigen::VectorXd& density) const
{
    validate_projector_density(density, density_size_);
    return density - project(density);
}

Eigen::VectorXd NeumannEdgeContinuityProjector3D::constraint_mismatch(const Eigen::VectorXd& density) const
{
    validate_projector_density(density, density_size_);
    return physical_constraints_ * density;
}

double NeumannEdgeContinuityProjector3D::mismatch_linf(const Eigen::VectorXd& density) const
{
    const Eigen::VectorXd mismatch = constraint_mismatch(density);
    return mismatch.size() == 0 ? 0.0 : mismatch.lpNorm<Eigen::Infinity>();
}

double NeumannEdgeContinuityProjector3D::mismatch_weighted_rms(const Eigen::VectorXd& density) const
{
    const Eigen::VectorXd mismatch = constraint_mismatch(density);
    if (mismatch.size() == 0)
        return 0.0;
    return std::sqrt((edge_quadrature_weights_.array() * mismatch.array().square()).sum() / edge_quadrature_weights_.sum());
}

const NeumannEdgeProjectionDiagnostics3D& NeumannEdgeContinuityProjector3D::diagnostics() const { return diagnostics_; }

NeumannEdgeProjectedAugmentedOperator3D::NeumannEdgeProjectedAugmentedOperator3D(
    const kfbim::IKFBIOperator& base, const NeumannEdgeContinuityProjector3D& projector)
    : base_(base), projector_(projector)
{
    if (base_.problem_size() != projector_.density_size() + 1)
        throw std::invalid_argument("projected augmented operator base size is invalid");
}

int NeumannEdgeProjectedAugmentedOperator3D::problem_size() const { return base_.problem_size(); }

void NeumannEdgeProjectedAugmentedOperator3D::apply(const Eigen::VectorXd& unknown, Eigen::VectorXd& result) const
{
    if (unknown.size() != problem_size())
        throw std::invalid_argument("projected augmented operator input has invalid size");
    const int n = projector_.density_size();
    Eigen::VectorXd projected_unknown = unknown;
    projected_unknown.head(n) = projector_.project(unknown.head(n));
    Eigen::VectorXd base_result;
    base_.apply(projected_unknown, base_result);
    if (base_result.size() != problem_size())
        throw std::invalid_argument("projected augmented operator base result has invalid size");
    result = base_result;
    result.head(n) = projector_.project(base_result.head(n)) + unknown.head(n) - projected_unknown.head(n);
}

Eigen::VectorXd NeumannEdgeProjectedAugmentedOperator3D::project_right_hand_side(const Eigen::VectorXd& base_rhs) const
{
    if (base_rhs.size() != problem_size())
        throw std::invalid_argument("projected augmented operator RHS has invalid size");
    Eigen::VectorXd projected_rhs = base_rhs;
    projected_rhs.head(projector_.density_size()) = projector_.project(base_rhs.head(projector_.density_size()));
    return projected_rhs;
}

} // namespace kfbim::app3d