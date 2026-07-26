#include "neumann_edge_continuity_3d.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
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

bool neumann_edge_shared_preprocess_pass_3d(
    const std::vector<NeumannEdgePreprocessInvariantSnapshot3D>& snapshots)
{
    if (snapshots.size() != 6)
        return false;
    const NeumannEdgePreprocessInvariantSnapshot3D& reference =
        snapshots.front();
    return std::all_of(
        snapshots.begin(), snapshots.end(),
        [&](const NeumannEdgePreprocessInvariantSnapshot3D& item) {
            return item.diagnostics_match_reference
                && item.workload_fingerprint
                    == reference.workload_fingerprint
                && item.output_digest == reference.output_digest
                && item.wrong_side_queries
                    == reference.wrong_side_queries
                && item.geometry_queries == reference.geometry_queries;
        });
}

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

NeumannEdgeConstraintAudit3D audit_neumann_edge_constraints_3d(
    const NativeNurbsSurface3D& surface, const SurfaceDofCloud3D& cloud,
    double h, const NeumannEdgeConstraintSet3D& constraints)
{
    if (!std::isfinite(h) || h <= 0.0)
        throw std::invalid_argument("edge constraint audit requires positive h");
    if (cloud.patches.size() != surface.patches.size())
        throw std::invalid_argument(
            "edge constraint audit patch grids do not match the surface");

    NeumannEdgeConstraintAudit3D result;
    result.constraint_rows = static_cast<int>(constraints.matrix.rows());
    bool dimensions_ok = constraints.density_size
            == static_cast<int>(cloud.dofs.size())
        && constraints.matrix.cols() == constraints.density_size
        && constraints.matrix.rows()
            == static_cast<Eigen::Index>(constraints.samples.size())
        && constraints.quadrature_weights.size()
            == static_cast<Eigen::Index>(constraints.samples.size());
    if (constraints.matrix.rows()
        > static_cast<Eigen::Index>(constraints.samples.size())) {
        result.unrelated_constraint_rows += static_cast<int>(
            constraints.matrix.rows() - constraints.samples.size());
    }
    if (constraints.quadrature_weights.size()
        > static_cast<Eigen::Index>(constraints.samples.size())) {
        result.unrelated_constraint_rows += static_cast<int>(
            constraints.quadrature_weights.size()
            - constraints.samples.size());
    }
    if (constraints.density_size != static_cast<int>(cloud.dofs.size())
        || constraints.matrix.cols() != constraints.density_size) {
        result.unrelated_constraint_rows +=
            std::max(1, result.constraint_rows);
    }

    const auto nearly_equal = [](double actual, double expected) {
        const double scale = std::max(
            {1.0, std::abs(actual), std::abs(expected)});
        return std::isfinite(actual) && std::isfinite(expected)
            && std::abs(actual - expected)
                <= 128.0 * std::numeric_limits<double>::epsilon() * scale;
    };
    std::vector<int> expected_counts(
        surface.geometric_connections.size(), -1);
    std::vector<std::vector<int>> occurrences(
        surface.geometric_connections.size());
    std::vector<bool> connection_valid(
        surface.geometric_connections.size(), true);
    std::vector<bool> connection_has_duplicate(
        surface.geometric_connections.size(), false);
    using EdgeIntervalKey = std::tuple<int, int, double, double>;
    using ConnectionIntervalKey =
        std::pair<EdgeIntervalKey, EdgeIntervalKey>;
    std::set<ConnectionIntervalKey> declared_intervals;
    for (std::size_t c = 0; c < surface.geometric_connections.size(); ++c) {
        const auto& connection = surface.geometric_connections[c];
        if (connection.g1) continue;
        ++result.expected_non_g1_connections;
        EdgeIntervalKey first{connection.first.patch,
            static_cast<int>(connection.first.edge),
            connection.first.begin, connection.first.end};
        EdgeIntervalKey second{connection.second.patch,
            static_cast<int>(connection.second.edge),
            connection.second.begin, connection.second.end};
        if (second < first) std::swap(first, second);
        if (!declared_intervals.insert({first, second}).second)
            ++result.duplicate_connection_intervals;
        const int expected = std::max(2, static_cast<int>(std::ceil(
            edge_length(surface, connection.first) / h)));
        expected_counts[c] = expected;
        occurrences[c].assign(static_cast<std::size_t>(expected), 0);
    }

    for (std::size_t row_index = 0;
         row_index < constraints.samples.size(); ++row_index) {
        const auto& sample = constraints.samples[row_index];
        if (sample.first_reduced_order || sample.second_reduced_order)
            ++result.reduced_order_rows;
        if (sample.connection_index < 0
            || sample.connection_index >= static_cast<int>(
                surface.geometric_connections.size())) {
            ++result.unrelated_constraint_rows;
            continue;
        }
        const std::size_t connection_index =
            static_cast<std::size_t>(sample.connection_index);
        const auto& connection =
            surface.geometric_connections[connection_index];
        if (connection.g1) {
            ++result.g1_constraint_rows;
            continue;
        }

        bool row_valid = true;
        const int expected_count = expected_counts[connection_index];
        if (sample.sample_index < 0
            || sample.sample_index >= expected_count) {
            row_valid = false;
        } else {
            int& count = occurrences[connection_index][
                static_cast<std::size_t>(sample.sample_index)];
            ++count;
            if (count > 1)
                connection_has_duplicate[connection_index] = true;
        }
        row_valid = row_valid && sample.sample_count == expected_count
            && sample.first_patch == connection.first.patch
            && sample.second_patch == connection.second.patch
            && sample.first_edge == connection.first.edge
            && sample.second_edge == connection.second.edge;
        if (sample.sample_index >= 0
            && sample.sample_index < expected_count) {
            const double s = (sample.sample_index + 0.5)
                / static_cast<double>(expected_count);
            const double expected_first = edge_parameter(connection.first, s);
            const double expected_second = edge_parameter(
                connection.second, connection.reversed ? 1.0 - s : s);
            row_valid = row_valid
                && nearly_equal(sample.normalized_parameter, s)
                && nearly_equal(sample.first_parameter, expected_first)
                && nearly_equal(sample.second_parameter, expected_second);
        }

        if (row_index >= static_cast<std::size_t>(
                constraints.quadrature_weights.size())
            || !nearly_equal(sample.quadrature_weight,
                constraints.quadrature_weights[
                    static_cast<Eigen::Index>(row_index)])
            || sample.quadrature_weight <= 0.0) {
            row_valid = false;
        }

        if (row_index >= static_cast<std::size_t>(
                constraints.matrix.rows())
            || connection.first.patch < 0
            || connection.second.patch < 0
            || connection.first.patch >= static_cast<int>(cloud.patches.size())
            || connection.second.patch >= static_cast<int>(cloud.patches.size())) {
            row_valid = false;
        } else {
            const auto& first_patch = cloud.patches[
                static_cast<std::size_t>(connection.first.patch)];
            const auto& second_patch = cloud.patches[
                static_cast<std::size_t>(connection.second.patch)];
            const auto in_patch = [](int column, const SurfaceDofPatch3D& patch) {
                return column >= patch.first_dof
                    && column < patch.first_dof + patch.dof_count();
            };
            bool has_first_support = false;
            bool has_second_support = false;
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(
                     constraints.matrix, static_cast<int>(row_index));
                 it; ++it) {
                if (!std::isfinite(it.value())) {
                    row_valid = false;
                    continue;
                }
                if (it.value() == 0.0) continue;
                const bool first_support = in_patch(it.col(), first_patch);
                const bool second_support = in_patch(it.col(), second_patch);
                if (!first_support && !second_support) row_valid = false;
                has_first_support = has_first_support || first_support;
                has_second_support = has_second_support || second_support;
            }
            row_valid = row_valid && has_first_support && has_second_support;
        }
        if (!row_valid) {
            connection_valid[connection_index] = false;
            ++result.unrelated_constraint_rows;
        }
    }

    for (std::size_t c = 0; c < surface.geometric_connections.size(); ++c) {
        if (expected_counts[c] < 0) continue;
        for (int count : occurrences[c])
            if (count != 1) connection_valid[c] = false;
        if (connection_has_duplicate[c])
            ++result.duplicate_connection_intervals;
        if (connection_valid[c])
            ++result.covered_non_g1_connections;
    }
    result.pass = dimensions_ok
        && result.expected_non_g1_connections > 0
        && result.covered_non_g1_connections
            == result.expected_non_g1_connections
        && result.duplicate_connection_intervals == 0
        && result.g1_constraint_rows == 0
        && result.unrelated_constraint_rows == 0
        && result.constraint_rows > 0;
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

namespace {

using EdgeStatus = RigidStudyCriterionStatus3D;
using EdgeKey = std::tuple<std::string, int, NeumannDensitySpace3D>;

EdgeStatus edge_status(bool pass)
{
    return pass ? EdgeStatus::Pass : EdgeStatus::Fail;
}

double edge_order(double coarse_error, double fine_error, double coarse_h, double fine_h)
{
    if (!std::isfinite(coarse_error) || !std::isfinite(fine_error)
        || !std::isfinite(coarse_h) || !std::isfinite(fine_h)
        || coarse_error <= 0.0 || fine_error <= 0.0 || coarse_h <= fine_h || fine_h <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();
    return std::log(coarse_error / fine_error) / std::log(coarse_h / fine_h);
}

bool edge_measurement_finite(const NeumannEdgeContinuityMeasurement3D& row)
{
    return row.finite_metrics && row.N > 0 && std::isfinite(row.h) && row.h > 0.0
        && std::isfinite(row.gmres_relative_residual) && std::isfinite(row.density_linf)
        && std::isfinite(row.density_l2) && std::isfinite(row.interior_linf)
        && std::isfinite(row.interior_l2) && std::isfinite(row.edge_mismatch_linf)
        && std::isfinite(row.edge_mismatch_weighted_rms)
        && std::isfinite(row.exact_edge_mismatch_linf);
}

bool topology_ok(const NeumannEdgeContinuityMeasurement3D& row)
{
    return row.expected_non_g1_connections > 0
        && row.covered_non_g1_connections == row.expected_non_g1_connections
        && row.duplicate_connection_intervals == 0 && row.g1_constraint_rows == 0
        && row.unrelated_constraint_rows == 0 && row.constraint_rows > 0
        && row.constraint_rank > 0 && row.constraint_rank <= row.constraint_rows
        && row.reduced_order_rows >= 0;
}

bool projector_ok(const NeumannEdgeContinuityMeasurement3D& row)
{
    constexpr double tolerance = 1.0e-11;
    return std::isfinite(row.constant_constraint_defect)
        && std::isfinite(row.projected_constraint_defect)
        && std::isfinite(row.projection_idempotence_defect)
        && std::isfinite(row.constant_projection_defect)
        && std::abs(row.constant_constraint_defect) <= tolerance
        && std::abs(row.projected_constraint_defect) <= tolerance
        && std::abs(row.projection_idempotence_defect) <= tolerance
        && std::abs(row.constant_projection_defect) <= tolerance;
}

bool gmres_row_ok(const NeumannEdgeContinuityMeasurement3D& row)
{
    return row.gmres_converged && row.gmres_iterations >= 0
        && row.gmres_iterations <= 80 && std::isfinite(row.gmres_relative_residual)
        && row.gmres_relative_residual <= 2.0e-10;
}

bool geometry_owner_ok(const NeumannEdgeContinuityMeasurement3D& row)
{
    return row.geometry_diagnostics_pass && row.owner_invariants_pass
        && row.shared_preprocess_pass;
}

const NeumannEdgeContinuityMeasurement3D* edge_measurement(
    const std::map<EdgeKey, const NeumannEdgeContinuityMeasurement3D*>& rows,
    const std::string& case_id, int N, NeumannDensitySpace3D density_space)
{
    const auto found = rows.find({case_id, N, density_space});
    return found == rows.end() ? nullptr : found->second;
}

bool all_pair_rows_available(
    const std::map<EdgeKey, const NeumannEdgeContinuityMeasurement3D*>& rows,
    const std::vector<std::string>& case_ids)
{
    for (const std::string& case_id : case_ids) {
        for (int N : {32, 64}) {
            const bool any = edge_measurement(rows, case_id, N, NeumannDensitySpace3D::PatchIndependent)
                || edge_measurement(rows, case_id, N, NeumannDensitySpace3D::NonG1EdgeProjected);
            if (any && (!edge_measurement(rows, case_id, N, NeumannDensitySpace3D::PatchIndependent)
                        || !edge_measurement(rows, case_id, N, NeumannDensitySpace3D::NonG1EdgeProjected)))
                return false;
        }
    }
    return true;
}

} // namespace

std::vector<int> normalize_neumann_edge_continuity_levels_3d(std::vector<int> levels)
{
    if (levels.empty()) levels = {32, 64};
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    for (int N : levels)
        if (N != 32 && N != 64 && N != 128)
            throw std::invalid_argument("Neumann edge-continuity study N must be 32, 64, or 128");
    const bool has32 = std::binary_search(levels.begin(), levels.end(), 32);
    const bool has64 = std::binary_search(levels.begin(), levels.end(), 64);
    const bool has128 = std::binary_search(levels.begin(), levels.end(), 128);
    if (has64 && !has32)
        throw std::invalid_argument("Neumann edge-continuity study N=64 requires N=32");
    if (has128 && (!has32 || !has64))
        throw std::invalid_argument("Neumann edge-continuity study N=128 requires N=32 and N=64");
    return levels;
}

NeumannEdgeContinuityEvaluation3D evaluate_neumann_edge_continuity_study_3d(
    const std::vector<NeumannEdgeContinuityMeasurement3D>& measurements,
    const std::vector<std::string>& case_ids, bool require_complete_pilot)
{
    std::set<std::string> known_cases;
    for (const std::string& case_id : case_ids)
        if (case_id.empty() || !known_cases.insert(case_id).second)
            throw std::invalid_argument("Neumann edge-continuity study case IDs must be nonempty and unique");

    std::map<EdgeKey, const NeumannEdgeContinuityMeasurement3D*> keyed;
    for (const auto& row : measurements) {
        if (known_cases.count(row.case_id) == 0)
            throw std::invalid_argument("Neumann edge-continuity study measurement has an unknown case ID");
        if (row.N != 32 && row.N != 64 && row.N != 128)
            throw std::invalid_argument("Neumann edge-continuity study measurement has an invalid level");
        if (row.density_space != NeumannDensitySpace3D::PatchIndependent
            && row.density_space != NeumannDensitySpace3D::NonG1EdgeProjected)
            throw std::invalid_argument("Neumann edge-continuity study measurement has an invalid density space");
        if (!keyed.emplace(EdgeKey{row.case_id, row.N, row.density_space}, std::addressof(row)).second)
            throw std::invalid_argument("Neumann edge-continuity study measurement key is duplicated");
    }

    std::vector<const NeumannEdgeContinuityMeasurement3D*> coarse_measurements;
    std::vector<const NeumannEdgeContinuityMeasurement3D*> extended_measurements;
    for (const auto& row : measurements) {
        (row.N == 128 ? extended_measurements : coarse_measurements)
            .push_back(std::addressof(row));
    }

    const auto all_rows_satisfy = [](const auto& rows, const auto& predicate) {
        return std::all_of(rows.begin(), rows.end(),
            [&](const auto* row) { return predicate(*row); });
    };

    NeumannEdgeContinuityEvaluation3D result;
    result.rows.reserve(measurements.size());
    for (const auto& measurement : measurements) {
        NeumannEdgeContinuityDerivedRow3D derived;
        derived.measurement = measurement;
        const NeumannEdgeContinuityMeasurement3D* previous = nullptr;
        for (const auto& candidate : measurements) {
            if (candidate.case_id == measurement.case_id
                && candidate.density_space == measurement.density_space
                && candidate.N < measurement.N
                && (previous == nullptr || candidate.N > previous->N))
                previous = std::addressof(candidate);
        }
        if (previous != nullptr) {
            derived.density_linf_order = edge_order(previous->density_linf, measurement.density_linf, previous->h, measurement.h);
            derived.density_l2_order = edge_order(previous->density_l2, measurement.density_l2, previous->h, measurement.h);
            derived.interior_linf_order = edge_order(previous->interior_linf, measurement.interior_linf, previous->h, measurement.h);
            derived.interior_l2_order = edge_order(previous->interior_l2, measurement.interior_l2, previous->h, measurement.h);
            derived.exact_edge_mismatch_order = edge_order(previous->exact_edge_mismatch_linf, measurement.exact_edge_mismatch_linf, previous->h, measurement.h);
        }
        const auto* unconstrained = edge_measurement(keyed, measurement.case_id, measurement.N,
            NeumannDensitySpace3D::PatchIndependent);
        if (measurement.density_space == NeumannDensitySpace3D::NonG1EdgeProjected
            && unconstrained != nullptr) {
            const auto ratio = [](double numerator, double denominator) {
                return std::isfinite(numerator) && std::isfinite(denominator) && denominator > 0.0
                    ? numerator / denominator : std::numeric_limits<double>::quiet_NaN();
            };
            derived.density_linf_ratio_to_unconstrained = ratio(measurement.density_linf, unconstrained->density_linf);
            derived.density_l2_ratio_to_unconstrained = ratio(measurement.density_l2, unconstrained->density_l2);
            derived.interior_linf_ratio_to_unconstrained = ratio(measurement.interior_linf, unconstrained->interior_linf);
            derived.interior_l2_ratio_to_unconstrained = ratio(measurement.interior_l2, unconstrained->interior_l2);
            derived.edge_reduction_ratio = ratio(unconstrained->edge_mismatch_linf, measurement.edge_mismatch_linf);
        }
        derived.row_pass = edge_status(edge_measurement_finite(measurement)
            && gmres_row_ok(measurement) && topology_ok(measurement)
            && projector_ok(measurement) && geometry_owner_ok(measurement));
        result.rows.push_back(std::move(derived));
    }

    const bool complete = std::all_of(case_ids.begin(), case_ids.end(), [&](const std::string& case_id) {
        for (int N : {32, 64})
            for (NeumannDensitySpace3D mode : {NeumannDensitySpace3D::PatchIndependent,
                                               NeumannDensitySpace3D::NonG1EdgeProjected})
                if (edge_measurement(keyed, case_id, N, mode) == nullptr) return false;
        return true;
    });
    const bool pairs_available = all_pair_rows_available(keyed, case_ids);
    const bool rows_finite = all_rows_satisfy(coarse_measurements, edge_measurement_finite);
    const bool topology = all_rows_satisfy(coarse_measurements, topology_ok);
    const bool projector = all_rows_satisfy(coarse_measurements, projector_ok);
    const bool geometry_owner = all_rows_satisfy(coarse_measurements, geometry_owner_ok);
    result.acceptance.completeness_pass = complete ? EdgeStatus::Pass
        : require_complete_pilot ? EdgeStatus::Fail : EdgeStatus::NotEvaluated;
    result.acceptance.topology_pass = coarse_measurements.empty() ? EdgeStatus::NotEvaluated : edge_status(topology);
    result.acceptance.projector_pass = coarse_measurements.empty() ? EdgeStatus::NotEvaluated : edge_status(projector);
    result.acceptance.geometry_owner_pass = coarse_measurements.empty() ? EdgeStatus::NotEvaluated : edge_status(geometry_owner);

    bool gmres_ok = rows_finite && all_rows_satisfy(coarse_measurements, gmres_row_ok);
    if (!gmres_ok) result.acceptance.gmres_pass = EdgeStatus::Fail;
    else if (!pairs_available || coarse_measurements.empty()) result.acceptance.gmres_pass = EdgeStatus::NotEvaluated;
    else {
        int projected_max = 0, unconstrained_max = 0, projected_ty_max = 0, unconstrained_ty_max = 0;
        for (const auto* row_pointer : coarse_measurements) {
            const auto& row = *row_pointer;
            int* maximum = row.density_space == NeumannDensitySpace3D::NonG1EdgeProjected
                ? &projected_max : &unconstrained_max;
            *maximum = std::max(*maximum, row.gmres_iterations);
            if (row.case_id == "ty_m0083") {
                maximum = row.density_space == NeumannDensitySpace3D::NonG1EdgeProjected
                    ? &projected_ty_max : &unconstrained_ty_max;
                *maximum = std::max(*maximum, row.gmres_iterations);
            }
        }
        result.acceptance.gmres_pass = edge_status(projected_max <= unconstrained_max
            && projected_ty_max < unconstrained_ty_max);
    }

    const auto ratio32 = [&](const std::string& case_id, NeumannDensitySpace3D mode) {
        const auto* coarse = edge_measurement(keyed, case_id, 32, mode);
        const auto* fine = edge_measurement(keyed, case_id, 64, mode);
        if (coarse == nullptr || fine == nullptr) return std::numeric_limits<double>::quiet_NaN();
        return coarse->exact_edge_mismatch_linf / fine->exact_edge_mismatch_linf;
    };
    bool exact_ready = complete;
    bool exact_ok = exact_ready;
    for (const std::string& case_id : case_ids) {
        const double value = ratio32(case_id, NeumannDensitySpace3D::NonG1EdgeProjected);
        exact_ok = exact_ok && std::isfinite(value) && value >= 6.0;
    }
    result.acceptance.exact_trace_order_pass = exact_ready ? edge_status(exact_ok) : EdgeStatus::NotEvaluated;

    if (!pairs_available || coarse_measurements.empty()) {
        result.acceptance.error_guard_pass = EdgeStatus::NotEvaluated;
        result.acceptance.edge_reduction_pass = EdgeStatus::NotEvaluated;
    } else {
        bool error_ok = true, edge_ok = true;
        for (const auto* row_pointer : coarse_measurements) {
            const auto& row = *row_pointer;
            if (row.density_space != NeumannDensitySpace3D::NonG1EdgeProjected) continue;
            const auto* base = edge_measurement(keyed, row.case_id, row.N, NeumannDensitySpace3D::PatchIndependent);
            error_ok = error_ok && base != nullptr && row.density_linf <= 1.10 * base->density_linf
                && row.density_l2 <= 1.10 * base->density_l2 && row.interior_linf <= 1.10 * base->interior_linf
                && row.interior_l2 <= 1.10 * base->interior_l2;
            edge_ok = edge_ok && base != nullptr && base->edge_mismatch_linf / row.edge_mismatch_linf >= 1.0e4;
        }
        result.acceptance.error_guard_pass = edge_status(error_ok);
        result.acceptance.edge_reduction_pass = edge_status(edge_ok);
    }

    if (!complete) result.acceptance.trend_pass = EdgeStatus::NotEvaluated;
    else {
        constexpr double roundoff = 64.0 * std::numeric_limits<double>::epsilon();
        bool trend_ok = true;
        for (const std::string& case_id : case_ids) {
            const auto* u32 = edge_measurement(keyed, case_id, 32, NeumannDensitySpace3D::PatchIndependent);
            const auto* u64 = edge_measurement(keyed, case_id, 64, NeumannDensitySpace3D::PatchIndependent);
            const auto* p32 = edge_measurement(keyed, case_id, 32, NeumannDensitySpace3D::NonG1EdgeProjected);
            const auto* p64 = edge_measurement(keyed, case_id, 64, NeumannDensitySpace3D::NonG1EdgeProjected);
            const auto no_worse = [roundoff](double p_fine, double p_coarse, double u_fine, double u_coarse) {
                return p_coarse > 0.0 && u_coarse > 0.0 && p_fine / p_coarse <= (u_fine / u_coarse) * (1.0 + roundoff);
            };
            trend_ok = trend_ok && no_worse(p64->density_linf, p32->density_linf, u64->density_linf, u32->density_linf)
                && no_worse(p64->density_l2, p32->density_l2, u64->density_l2, u32->density_l2)
                && no_worse(p64->interior_linf, p32->interior_linf, u64->interior_linf, u32->interior_linf)
                && no_worse(p64->interior_l2, p32->interior_l2, u64->interior_l2, u32->interior_l2);
        }
        result.acceptance.trend_pass = edge_status(trend_ok);
    }

    if (extended_measurements.empty()) {
        result.acceptance.extended_evidence_pass = EdgeStatus::NotEvaluated;
    } else {
        const bool extended_complete = std::all_of(
            case_ids.begin(), case_ids.end(), [&](const std::string& case_id) {
                return edge_measurement(keyed, case_id, 128,
                           NeumannDensitySpace3D::PatchIndependent)
                    && edge_measurement(keyed, case_id, 128,
                           NeumannDensitySpace3D::NonG1EdgeProjected);
            });
        bool extended_ok = extended_complete
            && all_rows_satisfy(extended_measurements, edge_measurement_finite)
            && all_rows_satisfy(extended_measurements, gmres_row_ok)
            && all_rows_satisfy(extended_measurements, topology_ok)
            && all_rows_satisfy(extended_measurements, projector_ok)
            && all_rows_satisfy(extended_measurements, geometry_owner_ok);
        int projected_max = 0;
        int unconstrained_max = 0;
        int projected_ty_max = 0;
        int unconstrained_ty_max = 0;
        if (extended_complete) {
            for (const std::string& case_id : case_ids) {
                const auto* base = edge_measurement(keyed, case_id, 128,
                    NeumannDensitySpace3D::PatchIndependent);
                const auto* projected = edge_measurement(keyed, case_id, 128,
                    NeumannDensitySpace3D::NonG1EdgeProjected);
                unconstrained_max = std::max(
                    unconstrained_max, base->gmres_iterations);
                projected_max = std::max(
                    projected_max, projected->gmres_iterations);
                if (case_id == "ty_m0083") {
                    unconstrained_ty_max = base->gmres_iterations;
                    projected_ty_max = projected->gmres_iterations;
                }
                extended_ok = extended_ok
                    && projected->density_linf <= 1.10 * base->density_linf
                    && projected->density_l2 <= 1.10 * base->density_l2
                    && projected->interior_linf <= 1.10 * base->interior_linf
                    && projected->interior_l2 <= 1.10 * base->interior_l2
                    && base->edge_mismatch_linf
                           / projected->edge_mismatch_linf >= 1.0e4;
            }
            extended_ok = extended_ok
                && projected_max <= unconstrained_max
                && projected_ty_max < unconstrained_ty_max;
        }
        result.acceptance.extended_evidence_pass = edge_status(extended_ok);
    }

    result.acceptance.overall_pass = combine_rigid_study_criteria_3d(
        {result.acceptance.completeness_pass, result.acceptance.topology_pass,
         result.acceptance.projector_pass, result.acceptance.exact_trace_order_pass,
         result.acceptance.gmres_pass, result.acceptance.error_guard_pass,
         result.acceptance.edge_reduction_pass, result.acceptance.trend_pass,
         result.acceptance.geometry_owner_pass}, !coarse_measurements.empty(), require_complete_pilot);
    result.all_pass = result.acceptance.overall_pass == EdgeStatus::Pass;
    return result;
}

bool neumann_edge_continuity_study_exit_pass_3d(
    const NeumannEdgeContinuityEvaluation3D& evaluation,
    bool require_complete_pilot)
{
    if (require_complete_pilot) return evaluation.all_pass;
    return std::all_of(evaluation.rows.begin(), evaluation.rows.end(),
               [](const NeumannEdgeContinuityDerivedRow3D& row) {
                   return row.row_pass == EdgeStatus::Pass;
               })
        && evaluation.acceptance.topology_pass == EdgeStatus::Pass
        && evaluation.acceptance.projector_pass == EdgeStatus::Pass
        && evaluation.acceptance.edge_reduction_pass == EdgeStatus::Pass
        && evaluation.acceptance.geometry_owner_pass == EdgeStatus::Pass;
}
} // namespace kfbim::app3d
