#include "src/support/cauchy/neumann_edge_augmented_cauchy_3d.hpp"
#include "src/support/cauchy/neumann_edge_augmented_cauchy_3d_detail.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace kfbim::app3d {
namespace detail {

NeumannEdgeSampleCountPlan3D plan_neumann_edge_sample_count_3d(
    double requested_sample_count,
    int minimum_edge_samples,
    int cumulative_sample_count,
    int reserved_factorization_units,
    int connection_index)
{
    const std::string context = " at connection "
        + std::to_string(connection_index);
    if (minimum_edge_samples <= 0 || cumulative_sample_count < 0
        || reserved_factorization_units < 0) {
        throw std::invalid_argument(
            "Neumann auxiliary edge sample count planner has invalid input"
            + context);
    }
    if (!std::isfinite(requested_sample_count)
        || requested_sample_count < 0.0
        || requested_sample_count
            >= static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "Neumann auxiliary edge sample count exceeds int range"
            + context);
    }

    const int connection_sample_count = std::max(
        minimum_edge_samples,
        static_cast<int>(requested_sample_count));

    // The value map emits at most 48 raw triplets per sample (the normal
    // map emits 28). Eigen's default SparseMatrix StorageIndex is int;
    // duplicate collapse can only reduce this raw count. At this sample
    // cap, the row-major outer-index array's samples+1 length is also safe.
    constexpr int maximum_sparse_samples =
        std::numeric_limits<int>::max() / 48;
    if (cumulative_sample_count > maximum_sparse_samples
        || connection_sample_count
            > maximum_sparse_samples - cumulative_sample_count) {
        throw std::overflow_error(
            "Neumann auxiliary cumulative sample count exceeds int-backed "
            "sparse storage range" + context);
    }

    constexpr int maximum_factorization_units =
        std::numeric_limits<int>::max() / 2;
    if (reserved_factorization_units > maximum_factorization_units
        || cumulative_sample_count
            > maximum_factorization_units - reserved_factorization_units
        || connection_sample_count
            > maximum_factorization_units - reserved_factorization_units
                - cumulative_sample_count) {
        throw std::overflow_error(
            "Neumann auxiliary cumulative sample count exceeds int-backed "
            "factorization range" + context);
    }
    return {
        connection_sample_count,
        cumulative_sample_count + connection_sample_count};
}

std::vector<int> select_neumann_edge_distance_candidates_3d(
    std::vector<NeumannEdgeDistanceCandidate3D> candidates,
    double radius_squared,
    int count)
{
    if (!std::isfinite(radius_squared) || radius_squared < 0.0)
        throw std::invalid_argument("Neumann edge attachment radius must be finite and nonnegative");
    if (count <= 0)
        throw std::invalid_argument("Neumann edge attachment count must be positive");
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return std::tie(a.squared_distance, a.sample_index, a.edge_sample_index)
            < std::tie(b.squared_distance, b.sample_index, b.edge_sample_index);
    });
    if (candidates.empty() || candidates.front().squared_distance > radius_squared)
        return {};

    // A certified pair is mathematically equidistant even if floating-point
    // distance evaluation orders its members differently after a rigid transform.
    // No uncertified distance difference is ever treated as a tie.
    for (std::size_t first = 0; first < candidates.size(); ++first) {
        for (std::size_t second = first + 1; second < candidates.size(); ++second) {
            if (candidates[first].certified_symmetric_partner
                    == candidates[second].sample_index
                && candidates[second].certified_symmetric_partner
                    == candidates[first].sample_index) {
                if (candidates[second].sample_index < candidates[first].sample_index)
                    std::swap(candidates[first], candidates[second]);
                break;
            }
        }
    }
    if (candidates.size() < static_cast<std::size_t>(count))
        throw std::runtime_error("Neumann edge attachment has fewer samples than requested");
    std::vector<int> selected;
    selected.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index)
        selected.push_back(candidates[static_cast<std::size_t>(index)].edge_sample_index);
    return selected;
}

void validate_neumann_edge_local_map_3d(
    const NeumannEdgeLocalMap3D& local,
    int surface_size,
    int edge_sample_count)
{
    const int local_edge_count =
        static_cast<int>(local.edge_sample_indices.size());
    if (local.center_dof < 0 || local.center_dof >= surface_size
        || local.value_dofs.size() != 48 || local.normal_dofs.size() != 28
        || local.value_map.rows() != 16 || local.value_map.cols() != 48
        || local.normal_map.rows() != 16 || local.normal_map.cols() != 28
        || local.edge_map.rows() != 16
        || local.edge_map.cols() != local_edge_count) {
        throw std::invalid_argument(
            "Neumann augmented Cauchy local map dimensions are invalid");
    }
    for (const int id : local.value_dofs) {
        if (id < 0 || id >= surface_size)
            throw std::invalid_argument(
                "Neumann augmented Cauchy local map has invalid value DOF");
    }
    for (const int id : local.normal_dofs) {
        if (id < 0 || id >= surface_size)
            throw std::invalid_argument(
                "Neumann augmented Cauchy local map has invalid normal DOF");
    }
    for (int first = 0; first < local_edge_count; ++first) {
        const int id = local.edge_sample_indices[static_cast<std::size_t>(first)];
        if (id < 0 || id >= edge_sample_count)
            throw std::invalid_argument(
                "Neumann augmented Cauchy local map has invalid edge sample");
        for (int second = first + 1; second < local_edge_count; ++second) {
            if (id == local.edge_sample_indices[static_cast<std::size_t>(second)])
                throw std::invalid_argument(
                    "Neumann augmented Cauchy local map has duplicate edge sample");
        }
    }
}

} // namespace detail
namespace {

using geometry3d::NurbsPatchEdgeInterval3D;

std::string sample_context(int connection, int sample, const char* detail)
{
    return "Neumann auxiliary edge connection " + std::to_string(connection)
        + " sample " + std::to_string(sample) + ": " + detail;
}

void validate_options(const NeumannEdgeAuxiliaryOptions3D& options)
{
    if (options.degree != 3)
        throw std::invalid_argument("Neumann auxiliary edge degree must be 3");
    if (options.value_samples_per_side != 24)
        throw std::invalid_argument("Neumann auxiliary edge value samples per side must be 24");
    if (options.normal_samples_per_side != 14)
        throw std::invalid_argument("Neumann auxiliary edge normal samples per side must be 14");
    if (options.minimum_edge_samples != 4)
        throw std::invalid_argument("Neumann auxiliary edge minimum sample count must be 4");
    if (options.rank_relative_cutoff != 3.0e-12)
        throw std::invalid_argument("Neumann auxiliary edge rank cutoff must be 3e-12");
}

void validate_interval(const NativeNurbsSurface3D& surface,
                       const NurbsPatchEdgeInterval3D& interval,
                       int connection,
                       const char* side)
{
    if (interval.patch < 0
        || interval.patch >= static_cast<int>(surface.patches.size())) {
        throw std::invalid_argument(
            "Neumann auxiliary edge connection " + std::to_string(connection)
            + " has invalid " + side + " patch");
    }
    if (!std::isfinite(interval.begin) || !std::isfinite(interval.end)
        || interval.begin == interval.end) {
        throw std::invalid_argument(
            "Neumann auxiliary edge connection " + std::to_string(connection)
            + " has invalid " + side + " interval");
    }
    const auto& patch = surface.patches[static_cast<std::size_t>(interval.patch)];
    const bool u_edge = interval.edge == PatchEdge3D::UMin
        || interval.edge == PatchEdge3D::UMax;
    const bool v_edge = interval.edge == PatchEdge3D::VMin
        || interval.edge == PatchEdge3D::VMax;
    if (!u_edge && !v_edge)
        throw std::invalid_argument("Neumann auxiliary edge has invalid edge enum");
    const double low = u_edge ? patch.domain_start_v() : patch.domain_start_u();
    const double high = u_edge ? patch.domain_end_v() : patch.domain_end_u();
    const double tolerance = 1.0e-12 * std::max(1.0, std::abs(high - low));
    if (interval.begin < low - tolerance || interval.begin > high + tolerance
        || interval.end < low - tolerance || interval.end > high + tolerance) {
        throw std::invalid_argument(
            "Neumann auxiliary edge connection " + std::to_string(connection)
            + " has out-of-domain " + side + " interval");
    }
}

bool has_topological_neighbor(const NativeNurbsSurface3D& surface,
                              int patch,
                              int neighbor)
{
    const auto& neighbors = surface.topological_patch_neighbors.at(
        static_cast<std::size_t>(patch));
    return std::find(neighbors.begin(), neighbors.end(), neighbor)
        != neighbors.end();
}

void validate_surface_topology(const NativeNurbsSurface3D& surface)
{
    const int patch_count = static_cast<int>(surface.patches.size());
    if (patch_count == 0
        || surface.patch_names.size() != surface.patches.size()
        || surface.smooth_neighbors.size() != surface.patches.size()
        || surface.topological_patch_neighbors.size() != surface.patches.size()
        || surface.patch_components.size() != surface.patches.size()) {
        throw std::invalid_argument(
            "Neumann auxiliary surface topology metadata is inconsistent");
    }

    for (int patch = 0; patch < patch_count; ++patch) {
        std::set<int> unique_neighbors;
        for (const int neighbor : surface.topological_patch_neighbors[
                 static_cast<std::size_t>(patch)]) {
            if (neighbor < 0 || neighbor >= patch_count || neighbor == patch
                || !unique_neighbors.insert(neighbor).second) {
                throw std::invalid_argument(
                    "Neumann auxiliary topological patch "
                    + std::to_string(patch) + " has invalid neighbor patch "
                    + std::to_string(neighbor));
            }
            if (!has_topological_neighbor(surface, neighbor, patch)) {
                throw std::invalid_argument(
                    "Neumann auxiliary topological patch "
                    + std::to_string(patch) + " neighbor patch "
                    + std::to_string(neighbor) + " is not reciprocal");
            }
        }
    }

    for (int patch = 0; patch < patch_count; ++patch) {
        for (int edge_index = 0; edge_index < 4; ++edge_index) {
            const auto& slot = surface.smooth_neighbors[
                static_cast<std::size_t>(patch)]
                [static_cast<std::size_t>(edge_index)];
            if (!slot)
                continue;
            const int neighbor_edge = static_cast<int>(slot->edge);
            if (slot->patch < 0 || slot->patch >= patch_count
                || slot->patch == patch
                || neighbor_edge < 0 || neighbor_edge >= 4) {
                throw std::invalid_argument(
                    "Neumann auxiliary smooth topology patch "
                    + std::to_string(patch) + " edge "
                    + std::to_string(edge_index) + " is invalid");
            }
            const auto& reciprocal = surface.smooth_neighbors[
                static_cast<std::size_t>(slot->patch)]
                [static_cast<std::size_t>(neighbor_edge)];
            if (!reciprocal || reciprocal->patch != patch
                || static_cast<int>(reciprocal->edge) != edge_index
                || reciprocal->reversed != slot->reversed) {
                throw std::invalid_argument(
                    "Neumann auxiliary smooth topology patch "
                    + std::to_string(patch) + " edge "
                    + std::to_string(edge_index) + " and patch "
                    + std::to_string(slot->patch) + " edge "
                    + std::to_string(neighbor_edge)
                    + " are not reciprocal");
            }
            if (!has_topological_neighbor(surface, patch, slot->patch)
                || !has_topological_neighbor(surface, slot->patch, patch)) {
                throw std::invalid_argument(
                    "Neumann auxiliary smooth topology patch "
                    + std::to_string(patch) + " and patch "
                    + std::to_string(slot->patch)
                    + " are not mutual topological neighbors");
            }
        }
    }

    for (int connection_index = 0;
         connection_index < static_cast<int>(surface.geometric_connections.size());
         ++connection_index) {
        const auto& connection = surface.geometric_connections[
            static_cast<std::size_t>(connection_index)];
        validate_interval(surface, connection.first, connection_index, "first");
        validate_interval(surface, connection.second, connection_index, "second");
        if (connection.first.patch == connection.second.patch
            || !has_topological_neighbor(
                surface, connection.first.patch, connection.second.patch)
            || !has_topological_neighbor(
                surface, connection.second.patch, connection.first.patch)) {
            throw std::invalid_argument(
                "Neumann auxiliary edge connection "
                + std::to_string(connection_index) + " patches "
                + std::to_string(connection.first.patch) + " and "
                + std::to_string(connection.second.patch)
                + " are not mutual topological neighbors");
        }
        const auto& first_slot = surface.smooth_neighbors[
            static_cast<std::size_t>(connection.first.patch)]
            [static_cast<std::size_t>(connection.first.edge)];
        const auto& second_slot = surface.smooth_neighbors[
            static_cast<std::size_t>(connection.second.patch)]
            [static_cast<std::size_t>(connection.second.edge)];
        if (connection.g1) {
            if (!first_slot || !second_slot
                || first_slot->patch != connection.second.patch
                || first_slot->edge != connection.second.edge
                || second_slot->patch != connection.first.patch
                || second_slot->edge != connection.first.edge
                || first_slot->reversed != connection.reversed
                || second_slot->reversed != connection.reversed) {
                throw std::invalid_argument(
                    "Neumann auxiliary G1 edge connection "
                    + std::to_string(connection_index)
                    + " does not match reciprocal smooth topology");
            }
        } else if (first_slot || second_slot) {
            throw std::invalid_argument(
                "Neumann auxiliary non-G1 edge connection "
                + std::to_string(connection_index)
                + " must be absent from smooth topology");
        }
    }
}
std::pair<double, double> edge_uv(
    const geometry3d::NurbsSurfacePatch3D& patch,
    PatchEdge3D edge,
    double parameter)
{
    switch (edge) {
    case PatchEdge3D::UMin: return {patch.domain_start_u(), parameter};
    case PatchEdge3D::UMax: return {patch.domain_end_u(), parameter};
    case PatchEdge3D::VMin: return {parameter, patch.domain_start_v()};
    case PatchEdge3D::VMax: return {parameter, patch.domain_end_v()};
    }
    throw std::invalid_argument("Neumann auxiliary edge has invalid edge enum");
}

double edge_parameter(const NurbsPatchEdgeInterval3D& interval, double s)
{
    return interval.begin + s * (interval.end - interval.begin);
}

Eigen::Vector3d edge_point(const NativeNurbsSurface3D& surface,
                           const NurbsPatchEdgeInterval3D& interval,
                           double parameter)
{
    const auto& patch = surface.patches.at(static_cast<std::size_t>(interval.patch));
    const auto uv = edge_uv(patch, interval.edge, parameter);
    return patch.evaluate(uv.first, uv.second);
}

Eigen::Vector3d edge_normal(const NativeNurbsSurface3D& surface,
                            const NurbsPatchEdgeInterval3D& interval,
                            double parameter)
{
    const auto& patch = surface.patches.at(static_cast<std::size_t>(interval.patch));
    const auto uv = edge_uv(patch, interval.edge, parameter);
    return patch.normal(uv.first, uv.second);
}

Eigen::Vector3d oriented_edge_tangent(
    const NativeNurbsSurface3D& surface,
    const NurbsPatchEdgeInterval3D& interval,
    double parameter)
{
    const auto& patch = surface.patches.at(static_cast<std::size_t>(interval.patch));
    const auto uv = edge_uv(patch, interval.edge, parameter);
    const auto derivatives = patch.evaluate_with_derivatives(uv.first, uv.second);
    Eigen::Vector3d tangent =
        interval.edge == PatchEdge3D::UMin || interval.edge == PatchEdge3D::UMax
            ? derivatives.dv : derivatives.du;
    tangent *= interval.end - interval.begin;
    return tangent;
}

double connection_length_8_point_gauss(
    const NativeNurbsSurface3D& surface,
    const NurbsPatchEdgeInterval3D& interval)
{
    constexpr std::array<double, 8> nodes{{
        -0.9602898564975363, -0.7966664774136267,
        -0.5255324099163290, -0.1834346424956498,
         0.1834346424956498,  0.5255324099163290,
         0.7966664774136267,  0.9602898564975363}};
    constexpr std::array<double, 8> weights{{
        0.1012285362903763, 0.2223810344533745,
        0.3137066458778873, 0.3626837833783620,
        0.3626837833783620, 0.3137066458778873,
        0.2223810344533745, 0.1012285362903763}};
    const double half = 0.5 * (interval.end - interval.begin);
    const double middle = 0.5 * (interval.begin + interval.end);
    double result = 0.0;
    for (std::size_t q = 0; q < nodes.size(); ++q) {
        const double parameter = middle + half * nodes[q];
        const double speed = oriented_edge_tangent(surface, interval, parameter).norm()
            / std::abs(interval.end - interval.begin);
        if (!std::isfinite(speed) || speed <= 1.0e-12)
            throw std::runtime_error("Neumann auxiliary NURBS edge has invalid speed");
        result += weights[q] * speed;
    }
    result *= std::abs(half);
    if (!std::isfinite(result) || result <= 1.0e-12)
        throw std::runtime_error("Neumann auxiliary NURBS edge has invalid length");
    return result;
}

std::vector<int> nearest_g1_side_dofs(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    int incident_patch,
    int forbidden_patch,
    const Eigen::Vector3d& point,
    int count)
{
    if (incident_patch < 0
        || incident_patch >= static_cast<int>(surface.patches.size())
        || forbidden_patch < 0
        || forbidden_patch >= static_cast<int>(surface.patches.size())
        || incident_patch == forbidden_patch
        || surface.smooth_neighbors.size() != surface.patches.size()) {
        throw std::invalid_argument("invalid patch for Neumann auxiliary G1-side search");
    }
    std::vector<bool> selected(surface.patches.size(), false);
    selected[static_cast<std::size_t>(incident_patch)] = true;
    std::vector<int> frontier{incident_patch};
    std::vector<int> candidates;
    auto append_patch = [&](int patch) {
        for (int dof = 0; dof < static_cast<int>(cloud.dofs.size()); ++dof) {
            if (cloud.dofs[static_cast<std::size_t>(dof)].patch_id == patch)
                candidates.push_back(dof);
        }
    };
    append_patch(incident_patch);
    int completed_rings = 0;
    while (completed_rings < 1 || static_cast<int>(candidates.size()) < count) {
        std::vector<int> next;
        for (const int patch : frontier) {
            for (const auto& slot :
                 surface.smooth_neighbors[static_cast<std::size_t>(patch)]) {
                if (!slot || slot->patch == forbidden_patch)
                    continue;
                if (slot->patch < 0
                    || slot->patch >= static_cast<int>(surface.patches.size())) {
                    throw std::invalid_argument("invalid smooth neighbor in Neumann auxiliary surface");
                }
                if (!selected[static_cast<std::size_t>(slot->patch)]) {
                    selected[static_cast<std::size_t>(slot->patch)] = true;
                    next.push_back(slot->patch);
                }
            }
        }
        ++completed_rings;
        if (next.empty())
            break;
        std::sort(next.begin(), next.end());
        next.erase(std::unique(next.begin(), next.end()), next.end());
        for (const int patch : next)
            append_patch(patch);
        frontier = std::move(next);
    }
    std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        const double da = (cloud.dofs[static_cast<std::size_t>(a)].point - point).squaredNorm();
        const double db = (cloud.dofs[static_cast<std::size_t>(b)].point - point).squaredNorm();
        return da < db || (da == db && a < b);
    });
    if (static_cast<int>(candidates.size()) < count)
        throw std::runtime_error("insufficient samples on Neumann auxiliary G1 side");
    candidates.resize(static_cast<std::size_t>(count));
    return candidates;
}

bool matches_tensor_center_parameter(
    double stored,
    double domain_start,
    double domain_end,
    int index,
    int count)
{
    if (!std::isfinite(stored) || count <= 0 || index < 0 || index >= count)
        return false;
    const double width = domain_end - domain_start;
    const double step = width / static_cast<double>(count);
    const double offset = static_cast<double>(index) + 0.5;
    const double expected = domain_start + offset * step;
    const double operation_product = 8.0
        * std::numeric_limits<double>::epsilon();
    const double gamma = operation_product / (1.0 - operation_product);
    const double scale = std::abs(domain_start) + std::abs(domain_end)
        + std::abs(width) + std::abs(offset * step) + std::abs(expected);
    return std::abs(stored - expected) <= gamma * scale;
}

void validate_cloud(const NativeNurbsSurface3D& surface,
                    const SurfaceDofCloud3D& cloud,
                    double geometry_diameter)
{
    if (cloud.patches.size() != surface.patches.size()
        || cloud.dofs.empty()) {
        throw std::invalid_argument(
            "Neumann auxiliary surface/cloud metadata is inconsistent");
    }
    const double point_tolerance = 1.0e-11 * geometry_diameter;
    const double normal_tolerance = 1.0e-11;
    int expected_first_dof = 0;
    for (int patch_id = 0;
         patch_id < static_cast<int>(surface.patches.size());
         ++patch_id) {
        const auto& tensor = cloud.patches[static_cast<std::size_t>(patch_id)];
        const auto& patch = surface.patches[static_cast<std::size_t>(patch_id)];
        const long long dof_count = static_cast<long long>(tensor.nu)
            * static_cast<long long>(tensor.nv);
        if (tensor.name != surface.patch_names[static_cast<std::size_t>(patch_id)]
            || tensor.nu <= 0 || tensor.nv <= 0
            || tensor.first_dof != expected_first_dof
            || dof_count <= 0
            || dof_count > static_cast<long long>(cloud.dofs.size())
            || expected_first_dof + dof_count
                > static_cast<long long>(cloud.dofs.size())) {
            throw std::invalid_argument(
                "Neumann auxiliary cloud tensor layout is invalid at patch "
                + std::to_string(patch_id));
        }

        for (int local = 0; local < static_cast<int>(dof_count); ++local) {
            const int dof_id = expected_first_dof + local;
            const auto& dof = cloud.dofs[static_cast<std::size_t>(dof_id)];
            const int expected_i = local / tensor.nv;
            const int expected_j = local % tensor.nv;
            const std::string context = "Neumann auxiliary cloud DOF "
                + std::to_string(dof_id) + " patch "
                + std::to_string(patch_id);
            if (dof.patch_id != patch_id || dof.i != expected_i
                || dof.j != expected_j) {
                throw std::invalid_argument(context
                    + " is inconsistent with tensor layout");
            }
            if (!matches_tensor_center_parameter(
                    dof.u, patch.domain_start_u(), patch.domain_end_u(),
                    expected_i, tensor.nu)
                || !matches_tensor_center_parameter(
                    dof.v, patch.domain_start_v(), patch.domain_end_v(),
                    expected_j, tensor.nv)) {
                throw std::invalid_argument(context
                    + " has inconsistent tensor-center parameters");
            }
            if (!dof.point.allFinite() || !dof.normal.allFinite())
                throw std::invalid_argument(context + " is non-finite");
            const Eigen::Vector3d expected_point = patch.evaluate(dof.u, dof.v);
            const Eigen::Vector3d expected_normal = patch.normal(dof.u, dof.v);
            if ((dof.point - expected_point).norm() > point_tolerance)
                throw std::invalid_argument(context
                    + " does not lie on the supplied NURBS patch");
            if (!expected_normal.allFinite()
                || expected_normal.norm() <= 1.0e-12
                || (dof.normal - expected_normal).norm() > normal_tolerance) {
                throw std::invalid_argument(context
                    + " has the wrong physical normal");
            }
        }
        expected_first_dof += static_cast<int>(dof_count);
    }
    if (expected_first_dof != static_cast<int>(cloud.dofs.size()))
        throw std::invalid_argument(
            "Neumann auxiliary cloud tensor layout does not cover all DOFs");
}

void append_side_rows(Eigen::MatrixXd& design,
                      Eigen::VectorXd& sqrt_weights,
                      int first_row,
                      const std::vector<int>& ids,
                      bool normal_rows,
                      const NeumannEdgeAuxiliarySample3D& sample,
                      const SurfaceDofCloud3D& cloud,
                      double h,
                      const HarmonicPolynomialSpace3D& space)
{
    for (int k = 0; k < static_cast<int>(ids.size()); ++k) {
        const auto& dof = cloud.dofs.at(static_cast<std::size_t>(ids[static_cast<std::size_t>(k)]));
        const Eigen::Vector3d xi = sample.frame.transpose()
            * (dof.point - sample.point) / h;
        const double denominator = std::pow(0.35 + xi.norm(), 2.0);
        if (normal_rows) {
            const Eigen::Vector3d normal_components = sample.frame.transpose() * dof.normal;
            design.row(first_row + k) = normal_components.transpose()
                * space.gradient(xi.x(), xi.y(), xi.z());
            sqrt_weights[first_row + k] = std::sqrt(0.85 / denominator);
        } else {
            design.row(first_row + k) =
                space.basis(xi.x(), xi.y(), xi.z()).transpose();
            sqrt_weights[first_row + k] = std::sqrt(1.0 / denominator);
        }
    }
}

void validate_local_options(
    const NeumannEdgeAugmentedCauchyOptions3D& options)
{
    if (options.degree != 3)
        throw std::invalid_argument("Neumann augmented Cauchy degree must be 3");
    if (options.edge_samples_per_connection != 4) {
        throw std::invalid_argument(
            "Neumann augmented Cauchy edge samples per connection must be 4");
    }
    if (options.edge_weight_scale != 1.0) {
        throw std::invalid_argument(
            "Neumann augmented Cauchy edge weight scale must be 1");
    }
    if (options.rank_relative_cutoff != 3.0e-12) {
        throw std::invalid_argument(
            "Neumann augmented Cauchy rank cutoff must be 3e-12");
    }
}

void validate_face_stencils(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const std::vector<NeumannEdgeFaceStencil3D>& stencils)
{
    const int surface_size = static_cast<int>(cloud.dofs.size());
    if (stencils.size() != static_cast<std::size_t>(surface_size)) {
        throw std::invalid_argument(
            "Neumann augmented Cauchy face stencil count does not match surface DOFs");
    }
    std::vector<std::vector<bool>> component_masks(surface.patches.size());
    auto component_mask = [&](int patch) -> const std::vector<bool>& {
        auto& mask = component_masks.at(static_cast<std::size_t>(patch));
        if (mask.empty()) {
            mask.assign(surface.patches.size(), false);
            for (const int member : smooth_patch_component(surface, patch))
                mask.at(static_cast<std::size_t>(member)) = true;
        }
        return mask;
    };
    for (int center = 0; center < surface_size; ++center) {
        const auto& center_dof = cloud.dofs[static_cast<std::size_t>(center)];
        if (center_dof.patch_id < 0
            || center_dof.patch_id >= static_cast<int>(surface.patches.size())) {
            throw std::invalid_argument(
                "Neumann augmented Cauchy center " + std::to_string(center)
                + " has invalid patch " + std::to_string(center_dof.patch_id));
        }
        const auto& allowed = component_mask(center_dof.patch_id);
        const auto& stencil = stencils[static_cast<std::size_t>(center)];
        if (stencil.value_dofs.size() != 48) {
            throw std::invalid_argument(
                "Neumann augmented Cauchy center " + std::to_string(center)
                + " must have exactly 48 value DOFs");
        }
        if (stencil.normal_dofs.size() != 28) {
            throw std::invalid_argument(
                "Neumann augmented Cauchy center " + std::to_string(center)
                + " must have exactly 28 normal DOFs");
        }
        auto validate_ids = [&](const std::vector<int>& ids, const char* kind) {
            for (const int id : ids) {
                if (id < 0 || id >= surface_size) {
                    throw std::invalid_argument(
                        "Neumann augmented Cauchy center "
                        + std::to_string(center) + " has invalid " + kind
                        + " DOF " + std::to_string(id));
                }
                const int patch = cloud.dofs[static_cast<std::size_t>(id)].patch_id;
                if (patch < 0 || patch >= static_cast<int>(allowed.size())
                    || !allowed[static_cast<std::size_t>(patch)]) {
                    throw std::invalid_argument(
                        "Neumann augmented Cauchy center "
                        + std::to_string(center) + " patch "
                        + std::to_string(center_dof.patch_id) + " " + kind
                        + " DOF " + std::to_string(id) + " patch "
                        + std::to_string(patch)
                        + " lies outside its G1 component");
                }
            }
        };
        validate_ids(stencil.value_dofs, "value");
        validate_ids(stencil.normal_dofs, "normal");
    }
}

Eigen::Vector3d center_coordinate(
    const SurfaceDof3D& center,
    const Eigen::Vector3d& point,
    double h)
{
    const Eigen::Vector3d displacement = (point - center.point) / h;
    return {displacement.dot(center.tangent1),
            displacement.dot(center.tangent2),
            displacement.dot(center.normal)};
}

double forward_error_gamma(int operation_count)
{
    const double product = operation_count
        * std::numeric_limits<double>::epsilon();
    return product / (1.0 - product);
}

bool native_l_prism_affine_patch(const NativeNurbsSurface3D& surface, int patch_id)
{
    if (surface.name != "l_prism"
        || surface.description != "twelve native bilinear NURBS L-prism patches"
        || surface.patches.size() != 12
        || patch_id < 0
        || patch_id >= static_cast<int>(surface.patches.size())) {
        return false;
    }
    const auto& patch = surface.patches[static_cast<std::size_t>(patch_id)];
    if (patch.basis_u().degree() != 1 || patch.basis_v().degree() != 1
        || patch.basis_u().num_basis_functions() != 2
        || patch.basis_v().num_basis_functions() != 2
        || patch.domain_start_u() != 0.0 || patch.domain_end_u() != 1.0
        || patch.domain_start_v() != 0.0 || patch.domain_end_v() != 1.0) {
        return false;
    }
    for (const auto& row : patch.weights())
        for (const double weight : row)
            if (weight != 1.0) return false;

    const auto& net = patch.control_net();
    const Eigen::Vector3d& p00 = net[0][0];
    const Eigen::Vector3d& p01 = net[0][1];
    const Eigen::Vector3d& p10 = net[1][0];
    const Eigen::Vector3d& p11 = net[1][1];
    // One transformed control coordinate uses fewer than 16 elementary
    // operations in R*(p-center)+center+translation. Gamma(64) also covers
    // the native corner construction and leaves a conservative forward-error
    // envelope without introducing a geometry-scale-independent tolerance.
    const double transform_gamma = forward_error_gamma(64);
    const double closure_arithmetic_gamma = forward_error_gamma(3);
    const double geometry_scale = p00.norm() + p01.norm()
        + p10.norm() + p11.norm()
        + (p10 - p00).norm() + (p01 - p00).norm();
    for (int axis = 0; axis < 3; ++axis) {
        const double closure = (p00[axis] - p01[axis])
            - p10[axis] + p11[axis];
        const double closure_bound =
            (transform_gamma + closure_arithmetic_gamma
                + transform_gamma * closure_arithmetic_gamma)
            * geometry_scale;
        if (std::abs(closure) > closure_bound)
            return false;
    }

    // A degree-one, unit-weight patch is affine after closure. Reflection in
    // its edge parameter preserves Euclidean distance from an interior point
    // exactly only when its physical U and V directions are orthogonal.
    const Eigen::Vector3d u = p10 - p00;
    const Eigen::Vector3d v = p01 - p00;
    Eigen::Vector3d u_error;
    Eigen::Vector3d v_error;
    const double subtraction_gamma = forward_error_gamma(1);
    const double direction_gamma = transform_gamma + subtraction_gamma
        + transform_gamma * subtraction_gamma;
    for (int axis = 0; axis < 3; ++axis) {
        u_error[axis] = direction_gamma
            * (std::abs(p10[axis]) + std::abs(p00[axis])
                + geometry_scale);
        v_error[axis] = direction_gamma
            * (std::abs(p01[axis]) + std::abs(p00[axis])
                + geometry_scale);
    }
    const double u_norm = u.norm();
    const double v_norm = v.norm();
    if (!std::isfinite(u_norm) || !std::isfinite(v_norm)
        || u_norm <= u_error.norm() || v_norm <= v_error.norm()) {
        return false;
    }
    double input_error_bound = 0.0;
    double dot_product_scale = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        input_error_bound += std::abs(u[axis]) * v_error[axis]
            + std::abs(v[axis]) * u_error[axis]
            + u_error[axis] * v_error[axis];
        dot_product_scale += std::abs(u[axis] * v[axis]);
    }
    const double dot_arithmetic_bound = forward_error_gamma(5)
        * dot_product_scale;
    return std::abs(u.dot(v))
        <= input_error_bound + dot_arithmetic_bound;
}

bool twice_dyadic_endpoint(double parameter, long long& twice)
{
    if (!std::isfinite(parameter)) return false;
    const double doubled = 2.0 * parameter;
    const long long integer = static_cast<long long>(std::llround(doubled));
    if (doubled != static_cast<double>(integer)) return false;
    twice = integer;
    return true;
}

int certified_l_prism_symmetric_partner(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const SurfaceDof3D& center,
    const geometry3d::NurbsPatchEdgeConnection3D& connection,
    const NeumannEdgeAuxiliarySample3D& sample)
{
    const NurbsPatchEdgeInterval3D* interval = nullptr;
    bool reverse_to_first = false;
    if (center.patch_id == connection.first.patch) {
        interval = &connection.first;
    } else if (center.patch_id == connection.second.patch) {
        interval = &connection.second;
        reverse_to_first = connection.reversed;
    } else {
        return -1;
    }
    if (!native_l_prism_affine_patch(surface, center.patch_id)
        || center.patch_id < 0
        || center.patch_id >= static_cast<int>(cloud.patches.size())
        || sample.sample_count <= 0
        || sample.sample_index < 0
        || sample.sample_index >= sample.sample_count) {
        return -1;
    }

    const auto& tensor = cloud.patches[static_cast<std::size_t>(center.patch_id)];
    int lattice_index = -1;
    int lattice_count = 0;
    if (interval->edge == PatchEdge3D::UMin
        || interval->edge == PatchEdge3D::UMax) {
        lattice_index = center.j;
        lattice_count = tensor.nv;
    } else {
        lattice_index = center.i;
        lattice_count = tensor.nu;
    }
    if (lattice_count <= 0 || lattice_index < 0 || lattice_index >= lattice_count)
        return -1;

    long long twice_begin = 0;
    long long twice_end = 0;
    if (!twice_dyadic_endpoint(interval->begin, twice_begin)
        || !twice_dyadic_endpoint(interval->end, twice_end)
        || twice_begin == twice_end) {
        return -1;
    }
    long long numerator = 2LL * lattice_index + 1LL
        - twice_begin * lattice_count;
    long long denominator = static_cast<long long>(lattice_count)
        * (twice_end - twice_begin);
    if (denominator < 0) {
        numerator = -numerator;
        denominator = -denominator;
    }
    if (reverse_to_first)
        numerator = denominator - numerator;

    // Edge samples have s_q=(2q+1)/(2M). Two samples q and q'
    // are symmetric about the exact center parameter c=num/den iff
    // (q+q'+1)*den=2*num*M.
    long long scaled_numerator = 2LL * numerator;
    long long reduced_denominator = denominator;
    const long long first_gcd = std::gcd(
        std::abs(scaled_numerator), reduced_denominator);
    scaled_numerator /= first_gcd;
    reduced_denominator /= first_gcd;
    long long sample_factor = sample.sample_count;
    const long long second_gcd = std::gcd(
        sample_factor, reduced_denominator);
    sample_factor /= second_gcd;
    reduced_denominator /= second_gcd;
    if (reduced_denominator != 1)
        return -1;
    if ((scaled_numerator > 0
            && sample_factor
                > std::numeric_limits<long long>::max() / scaled_numerator)
        || (scaled_numerator < 0
            && scaled_numerator
                < std::numeric_limits<long long>::min() / sample_factor)) {
        return -1;
    }
    const long long reflected_sum = scaled_numerator * sample_factor;
    if (reflected_sum < std::numeric_limits<long long>::min()
            + sample.sample_index + 1LL) {
        return -1;
    }
    const long long partner = reflected_sum
        - sample.sample_index - 1LL;
    if (partner < 0 || partner >= sample.sample_count
        || partner == sample.sample_index) {
        return -1;
    }
    return static_cast<int>(partner);
}

std::vector<int> attached_edge_samples(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const NeumannEdgeFaceStencil3D& stencil,
    const NeumannEdgeAuxiliaryValueMap3D& edge_map,
    const std::map<int, std::vector<int>>& samples_by_connection,
    int center,
    double h,
    int samples_per_connection,
    int& connection_group_count)
{
    const auto& center_dof = cloud.dofs.at(static_cast<std::size_t>(center));
    double radius_squared = 0.0;
    for (const int id : stencil.value_dofs) {
        radius_squared = std::max(
            radius_squared,
            (cloud.dofs.at(static_cast<std::size_t>(id)).point
                - center_dof.point).squaredNorm());
    }
    const std::vector<int> component =
        smooth_patch_component(surface, center_dof.patch_id);
    std::vector<bool> in_component(surface.patches.size(), false);
    for (const int patch : component) {
        if (patch < 0 || patch >= static_cast<int>(surface.patches.size())) {
            throw std::runtime_error(
                "Neumann augmented Cauchy center " + std::to_string(center)
                + " has invalid G1 component patch");
        }
        in_component[static_cast<std::size_t>(patch)] = true;
    }

    std::vector<int> result;
    connection_group_count = 0;
    for (const auto& item : samples_by_connection) {
        const int connection_index = item.first;
        if (connection_index < 0
            || connection_index
                >= static_cast<int>(surface.geometric_connections.size())) {
            throw std::runtime_error(
                "Neumann augmented Cauchy edge map has invalid connection ID");
        }
        const auto& connection = surface.geometric_connections[
            static_cast<std::size_t>(connection_index)];
        if (connection.g1) {
            throw std::runtime_error(
                "Neumann augmented Cauchy edge map contains a G1 connection");
        }
        const bool first = in_component.at(
            static_cast<std::size_t>(connection.first.patch));
        const bool second = in_component.at(
            static_cast<std::size_t>(connection.second.patch));
        if (first == second)
            continue;

        std::vector<detail::NeumannEdgeDistanceCandidate3D> nearest;
        nearest.reserve(item.second.size());
        for (const int sample_index : item.second) {
            if (sample_index < 0
                || sample_index >= static_cast<int>(edge_map.samples.size())) {
                throw std::runtime_error(
                    "Neumann augmented Cauchy edge map has invalid sample ID");
            }
            const auto& sample = edge_map.samples[
                static_cast<std::size_t>(sample_index)];
            if (sample.connection_index != connection_index) {
                throw std::runtime_error(
                    "Neumann augmented Cauchy edge sample group is inconsistent");
            }
            detail::NeumannEdgeDistanceCandidate3D candidate;
            candidate.squared_distance =
                (sample.point - center_dof.point).squaredNorm();
            candidate.sample_index = sample.sample_index;
            candidate.edge_sample_index = sample_index;
            candidate.certified_symmetric_partner =
                detail::certified_l_prism_symmetric_partner_3d(
                    surface, cloud, center_dof, connection, sample);
            nearest.push_back(candidate);
        }
        if (nearest.empty())
            continue;
        const auto exact_minimum = std::min_element(
            nearest.begin(), nearest.end(), [](const auto& a, const auto& b) {
                return a.squared_distance < b.squared_distance;
            });
        if (exact_minimum->squared_distance > radius_squared)
            continue;
        if (nearest.size() < static_cast<std::size_t>(samples_per_connection)) {
            throw std::runtime_error(
                "Neumann augmented Cauchy connection "
                + std::to_string(connection_index)
                + " has fewer than four edge samples");
        }
        std::vector<int> selected =
            detail::select_neumann_edge_distance_candidates_3d(
                std::move(nearest), radius_squared, samples_per_connection);
        std::sort(selected.begin(), selected.end(), [&](int a, int b) {
            const auto& first_sample = edge_map.samples[static_cast<std::size_t>(a)];
            const auto& second_sample = edge_map.samples[static_cast<std::size_t>(b)];
            return std::tie(first_sample.connection_index, first_sample.sample_index)
                < std::tie(second_sample.connection_index, second_sample.sample_index);
        });
        result.insert(result.end(), selected.begin(), selected.end());
        ++connection_group_count;
    }
    return result;
}

void append_local_value_rows(
    Eigen::MatrixXd& design,
    Eigen::VectorXd& sqrt_weights,
    int first_row,
    const std::vector<int>& ids,
    const SurfaceDof3D& center,
    const SurfaceDofCloud3D& cloud,
    double h,
    const HarmonicPolynomialSpace3D& space)
{
    for (int k = 0; k < static_cast<int>(ids.size()); ++k) {
        const auto& sample = cloud.dofs.at(
            static_cast<std::size_t>(ids[static_cast<std::size_t>(k)]));
        const Eigen::Vector3d xi = center_coordinate(center, sample.point, h);
        design.row(first_row + k) =
            space.basis(xi.x(), xi.y(), xi.z()).transpose();
        sqrt_weights[first_row + k] = std::sqrt(
            1.0 / std::pow(0.35 + xi.norm(), 2.0));
    }
}

void append_local_normal_rows(
    Eigen::MatrixXd& design,
    Eigen::VectorXd& sqrt_weights,
    int first_row,
    const std::vector<int>& ids,
    const SurfaceDof3D& center,
    const SurfaceDofCloud3D& cloud,
    double h,
    const HarmonicPolynomialSpace3D& space)
{
    for (int k = 0; k < static_cast<int>(ids.size()); ++k) {
        const auto& sample = cloud.dofs.at(
            static_cast<std::size_t>(ids[static_cast<std::size_t>(k)]));
        const Eigen::Vector3d xi = center_coordinate(center, sample.point, h);
        const Eigen::Vector3d normal_components(
            sample.normal.dot(center.tangent1),
            sample.normal.dot(center.tangent2),
            sample.normal.dot(center.normal));
        design.row(first_row + k) = normal_components.transpose()
            * space.gradient(xi.x(), xi.y(), xi.z());
        sqrt_weights[first_row + k] = std::sqrt(
            0.85 / std::pow(0.35 + xi.norm(), 2.0));
    }
}
} // namespace

namespace detail {

int certified_l_prism_symmetric_partner_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    const SurfaceDof3D& center,
    const geometry3d::NurbsPatchEdgeConnection3D& connection,
    const NeumannEdgeAuxiliarySample3D& sample)
{
    return certified_l_prism_symmetric_partner(
        surface, cloud, center, connection, sample);
}

} // namespace detail

const char* neumann_edge_cauchy_mode_name_3d(NeumannEdgeCauchyMode3D mode)
{
    switch (mode) {
    case NeumannEdgeCauchyMode3D::None: return "none";
    case NeumannEdgeCauchyMode3D::NonG1AuxiliaryValues:
        return "non_g1_auxiliary_values";
    }
    return "unknown";
}

NeumannEdgeAuxiliaryValueMap3D
build_neumann_edge_auxiliary_value_map_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double h,
    const NeumannEdgeAuxiliaryOptions3D& options)
{
    validate_options(options);
    if (!std::isfinite(h) || h <= 0.0)
        throw std::invalid_argument("Neumann auxiliary edge map requires positive h");
    validate_surface_topology(surface);
    const double geometry_diameter = surface.geometry_model().control_bounds().diameter();
    if (!std::isfinite(geometry_diameter) || geometry_diameter <= 0.0)
        throw std::invalid_argument("Neumann auxiliary surface has invalid diameter");
    validate_cloud(surface, cloud, geometry_diameter);
    const double gap_limit = 1.0e-11 * geometry_diameter;
    HarmonicPolynomialSpace3D space(options.degree);
    const Eigen::VectorXd origin_basis = space.basis(0.0, 0.0, 0.0);

    NeumannEdgeAuxiliaryValueMap3D result;
    result.surface_size = static_cast<int>(cloud.dofs.size());
    std::vector<int> connection_sample_counts(
        surface.geometric_connections.size(), 0);
    int cumulative_sample_count = 0;
    // Preflight every count before constructing any auxiliary sample.
    for (int c = 0; c < static_cast<int>(surface.geometric_connections.size()); ++c) {
        const auto& connection = surface.geometric_connections[static_cast<std::size_t>(c)];
        if (connection.g1)
            continue;
        ++result.diagnostics.expected_non_g1_connections;
        const double first_length = connection_length_8_point_gauss(surface, connection.first);
        const double second_length = connection_length_8_point_gauss(surface, connection.second);
        const double length_scale = std::max(first_length, second_length);
        if (std::abs(first_length - second_length) > 1.0e-11 * length_scale)
            throw std::runtime_error("Neumann auxiliary mapped edge lengths disagree at connection " + std::to_string(c));
        const double length = 0.5 * (first_length + second_length);
        const auto plan = detail::plan_neumann_edge_sample_count_3d(
            std::ceil(length / h),
            options.minimum_edge_samples,
            cumulative_sample_count,
            result.surface_size,
            c);
        connection_sample_counts[static_cast<std::size_t>(c)] =
            plan.connection_sample_count;
        cumulative_sample_count = plan.cumulative_sample_count;
    }

    std::vector<Eigen::Triplet<double>> value_entries;
    std::vector<Eigen::Triplet<double>> normal_entries;
    for (int c = 0; c < static_cast<int>(surface.geometric_connections.size()); ++c) {
        const int sample_count = connection_sample_counts[
            static_cast<std::size_t>(c)];
        if (sample_count == 0)
            continue;
        const auto& connection = surface.geometric_connections[
            static_cast<std::size_t>(c)];
        for (int q = 0; q < sample_count; ++q) {
            const double s = (static_cast<double>(q) + 0.5)
                / static_cast<double>(sample_count);
            NeumannEdgeAuxiliarySample3D sample;
            sample.connection_index = c;
            sample.sample_index = q;
            sample.sample_count = sample_count;
            sample.first_patch = connection.first.patch;
            sample.second_patch = connection.second.patch;
            sample.first_edge = connection.first.edge;
            sample.second_edge = connection.second.edge;
            sample.normalized_parameter = s;
            sample.first_parameter = edge_parameter(connection.first, s);
            sample.second_parameter = edge_parameter(
                connection.second, connection.reversed ? 1.0 - s : s);

            const Eigen::Vector3d first_point = edge_point(
                surface, connection.first, sample.first_parameter);
            const Eigen::Vector3d second_point = edge_point(
                surface, connection.second, sample.second_parameter);
            sample.mapped_point_gap = (first_point - second_point).norm();
            if (!std::isfinite(sample.mapped_point_gap)
                || sample.mapped_point_gap > gap_limit) {
                throw std::runtime_error(sample_context(c, q, "mapped NURBS points disagree"));
            }
            sample.point = 0.5 * (first_point + second_point);

            Eigen::Vector3d first_tangent = oriented_edge_tangent(
                surface, connection.first, sample.first_parameter);
            Eigen::Vector3d second_tangent = oriented_edge_tangent(
                surface, connection.second, sample.second_parameter);
            if (connection.reversed)
                second_tangent = -second_tangent;
            if (!first_tangent.allFinite() || !second_tangent.allFinite()
                || first_tangent.norm() <= 1.0e-12
                || second_tangent.norm() <= 1.0e-12) {
                throw std::runtime_error(sample_context(c, q, "degenerate tangent"));
            }
            const Eigen::Vector3d first_unit = first_tangent.normalized();
            const Eigen::Vector3d second_unit = second_tangent.normalized();
            if (first_unit.dot(second_unit) < 1.0 - 1.0e-11)
                throw std::runtime_error(sample_context(c, q, "mapped tangents disagree"));
            sample.oriented_tangent = (first_unit + second_unit).normalized();
            sample.first_normal = edge_normal(surface, connection.first, sample.first_parameter);
            sample.second_normal = edge_normal(surface, connection.second, sample.second_parameter);
            if (!sample.first_normal.allFinite() || !sample.second_normal.allFinite()
                || sample.first_normal.norm() <= 1.0e-12
                || sample.second_normal.norm() <= 1.0e-12) {
                throw std::runtime_error(sample_context(c, q, "degenerate normal"));
            }
            const Eigen::Vector3d bisector = sample.first_normal + sample.second_normal;
            Eigen::Vector3d e1 = bisector
                - bisector.dot(sample.oriented_tangent) * sample.oriented_tangent;
            if (!e1.allFinite() || e1.norm() <= 1.0e-12)
                throw std::runtime_error(sample_context(c, q, "degenerate normal bisector"));
            e1.normalize();
            const Eigen::Vector3d e2 = sample.oriented_tangent.cross(e1);
            sample.frame.col(0) = sample.oriented_tangent;
            sample.frame.col(1) = e1;
            sample.frame.col(2) = e2;
            if (!sample.frame.allFinite() || sample.frame.determinant() <= 1.0e-12)
                throw std::runtime_error(sample_context(c, q, "degenerate edge frame"));

            auto select_side = [&](int incident_patch,
                                   int forbidden_patch,
                                   int count,
                                   const char* side) {
                try {
                    return nearest_g1_side_dofs(
                        surface, cloud, incident_patch, forbidden_patch,
                        sample.point, count);
                } catch (const std::exception& error) {
                    const std::string detail = std::string(side)
                        + " side: " + error.what();
                    throw std::runtime_error(
                        sample_context(c, q, detail.c_str()));
                }
            };
            sample.first_value_dofs = select_side(
                sample.first_patch, sample.second_patch,
                options.value_samples_per_side, "first-value");
            sample.first_normal_dofs = select_side(
                sample.first_patch, sample.second_patch,
                options.normal_samples_per_side, "first-normal");
            sample.second_value_dofs = select_side(
                sample.second_patch, sample.first_patch,
                options.value_samples_per_side, "second-value");
            sample.second_normal_dofs = select_side(
                sample.second_patch, sample.first_patch,
                options.normal_samples_per_side, "second-normal");
            sample.first_owner_dof = sample.first_value_dofs.front();
            sample.second_owner_dof = sample.second_value_dofs.front();

            constexpr int rows = 2 * 24 + 2 * 14;
            Eigen::MatrixXd design(rows, space.dimension());
            Eigen::VectorXd sqrt_weights(rows);
            append_side_rows(design, sqrt_weights, 0,
                sample.first_value_dofs, false, sample, cloud, h, space);
            append_side_rows(design, sqrt_weights, 24,
                sample.second_value_dofs, false, sample, cloud, h, space);
            append_side_rows(design, sqrt_weights, 48,
                sample.first_normal_dofs, true, sample, cloud, h, space);
            append_side_rows(design, sqrt_weights, 62,
                sample.second_normal_dofs, true, sample, cloud, h, space);
            const Eigen::MatrixXd weighted = sqrt_weights.asDiagonal() * design;
            Eigen::JacobiSVD<Eigen::MatrixXd> condition_svd(weighted);
            const Eigen::VectorXd singular = condition_svd.singularValues();
            if (singular.size() != space.dimension() || !singular.allFinite()
                || !(singular[0] > 0.0)
                || !(singular[singular.size() - 1]
                     > options.rank_relative_cutoff * singular[0])) {
                ++result.diagnostics.rank_deficient_fit_count;
                throw std::runtime_error(sample_context(c, q, "rank-deficient weighted fit"));
            }
            sample.condition = singular[0] / singular[singular.size() - 1];
            const Eigen::MatrixXd pinv =
                svd_pseudoinverse_3d(weighted, options.rank_relative_cutoff);
            const int row = static_cast<int>(result.samples.size());
            auto append_coefficients = [&](const std::vector<int>& ids,
                                           int offset,
                                           bool normals) {
                for (int k = 0; k < static_cast<int>(ids.size()); ++k) {
                    double coefficient = origin_basis.dot(pinv.col(offset + k))
                        * sqrt_weights[offset + k];
                    if (normals)
                        coefficient *= h;
                    if (!std::isfinite(coefficient))
                        throw std::runtime_error(sample_context(c, q, "non-finite sparse coefficient"));
                    if (coefficient != 0.0) {
                        (normals ? normal_entries : value_entries).emplace_back(
                            row, ids[static_cast<std::size_t>(k)], coefficient);
                    }
                }
            };
            append_coefficients(sample.first_value_dofs, 0, false);
            append_coefficients(sample.second_value_dofs, 24, false);
            append_coefficients(sample.first_normal_dofs, 48, true);
            append_coefficients(sample.second_normal_dofs, 62, true);

            result.diagnostics.mapped_point_gap_max = std::max(
                result.diagnostics.mapped_point_gap_max, sample.mapped_point_gap);
            result.diagnostics.frame_orthogonality_defect_max = std::max(
                result.diagnostics.frame_orthogonality_defect_max,
                (sample.frame.transpose() * sample.frame
                    - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff());
            result.diagnostics.condition_max = std::max(
                result.diagnostics.condition_max, sample.condition);
            result.samples.push_back(std::move(sample));
        }
        ++result.diagnostics.covered_non_g1_connections;
    }

    result.value_map.resize(static_cast<Eigen::Index>(result.samples.size()), result.surface_size);
    result.normal_map.resize(static_cast<Eigen::Index>(result.samples.size()), result.surface_size);
    result.value_map.setFromTriplets(value_entries.begin(), value_entries.end());
    result.normal_map.setFromTriplets(normal_entries.begin(), normal_entries.end());
    result.value_map.makeCompressed();
    result.normal_map.makeCompressed();
    result.diagnostics.edge_sample_count = static_cast<int>(result.samples.size());

    for (int row = 0; row < static_cast<int>(result.samples.size()); ++row) {
        const auto& sample = result.samples[static_cast<std::size_t>(row)];
        for (int basis_column = 0; basis_column < space.dimension(); ++basis_column) {
            double predicted = 0.0;
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(result.value_map, row); it; ++it) {
                const auto& dof = cloud.dofs[static_cast<std::size_t>(it.col())];
                const Eigen::Vector3d xi = sample.frame.transpose()
                    * (dof.point - sample.point) / h;
                predicted += it.value()
                    * space.basis(xi.x(), xi.y(), xi.z())[basis_column];
            }
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(result.normal_map, row); it; ++it) {
                const auto& dof = cloud.dofs[static_cast<std::size_t>(it.col())];
                const Eigen::Vector3d xi = sample.frame.transpose()
                    * (dof.point - sample.point) / h;
                const Eigen::Vector3d normal_components = sample.frame.transpose() * dof.normal;
                predicted += it.value() * normal_components.dot(
                    space.gradient(xi.x(), xi.y(), xi.z()).col(basis_column)) / h;
            }
            result.diagnostics.harmonic_cubic_reproduction_defect_max = std::max(
                result.diagnostics.harmonic_cubic_reproduction_defect_max,
                std::abs(predicted - origin_basis[basis_column]));
        }
    }
    result.diagnostics.pass =
        result.diagnostics.expected_non_g1_connections > 0
        && result.diagnostics.covered_non_g1_connections
            == result.diagnostics.expected_non_g1_connections
        && result.diagnostics.edge_sample_count
            == static_cast<int>(result.samples.size())
        && result.diagnostics.unrelated_sample_count == 0
        && result.diagnostics.asymmetric_sample_count == 0
        && result.diagnostics.rank_deficient_fit_count == 0
        && result.diagnostics.mapped_point_gap_max <= gap_limit
        && result.diagnostics.frame_orthogonality_defect_max <= 1.0e-12
        && result.diagnostics.harmonic_cubic_reproduction_defect_max <= 1.0e-11
        && std::isfinite(result.diagnostics.condition_max);
    return result;
}

Eigen::VectorXd evaluate_neumann_edge_values_3d(
    const NeumannEdgeAuxiliaryValueMap3D& map,
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump)
{
    if (map.surface_size < 0
        || map.value_map.cols() != map.surface_size
        || map.normal_map.cols() != map.surface_size
        || map.value_map.rows() != map.normal_map.rows()
        || map.value_map.rows() != static_cast<Eigen::Index>(map.samples.size())) {
        throw std::invalid_argument("Neumann auxiliary edge map dimensions are invalid");
    }
    if (value_jump.size() != map.surface_size
        || normal_jump.size() != map.surface_size
        || !value_jump.allFinite() || !normal_jump.allFinite()) {
        throw std::invalid_argument("Neumann auxiliary edge jump data is invalid");
    }
    Eigen::VectorXd result = map.value_map * value_jump
        + map.normal_map * normal_jump;
    if (!result.allFinite())
        throw std::runtime_error("Neumann auxiliary edge values are non-finite");
    return result;
}

int NeumannEdgeAugmentedCauchy3D::surface_size() const
{
    return surface_size_;
}

int NeumannEdgeAugmentedCauchy3D::edge_sample_count() const
{
    return static_cast<int>(edge_value_map_.samples.size());
}

const NeumannEdgeAuxiliaryValueMap3D&
NeumannEdgeAugmentedCauchy3D::edge_value_map() const
{
    return edge_value_map_;
}

const std::vector<NeumannEdgeLocalMap3D>&
NeumannEdgeAugmentedCauchy3D::local_maps() const
{
    return local_maps_;
}

const NeumannEdgeAugmentedCauchyDiagnostics3D&
NeumannEdgeAugmentedCauchy3D::diagnostics() const
{
    return diagnostics_;
}

Eigen::VectorXd NeumannEdgeAugmentedCauchy3D::edge_values(
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump) const
{
    if (value_jump.size() != surface_size_ || !value_jump.allFinite())
        throw std::invalid_argument("Neumann augmented Cauchy value data is invalid");
    if (normal_jump.size() != surface_size_ || !normal_jump.allFinite())
        throw std::invalid_argument("Neumann augmented Cauchy normal data is invalid");
    return evaluate_neumann_edge_values_3d(
        edge_value_map_, value_jump, normal_jump);
}

void NeumannEdgeAugmentedCauchy3D::overwrite_affected_coefficients(
    const Eigen::VectorXd& value_jump,
    const Eigen::VectorXd& normal_jump,
    const Eigen::VectorXd& edge_values_data,
    Eigen::MatrixXd& coefficients) const
{
    if (value_jump.size() != surface_size_ || !value_jump.allFinite())
        throw std::invalid_argument("Neumann augmented Cauchy value data is invalid");
    if (normal_jump.size() != surface_size_ || !normal_jump.allFinite())
        throw std::invalid_argument("Neumann augmented Cauchy normal data is invalid");
    if (edge_values_data.size() != edge_sample_count()
        || !edge_values_data.allFinite()) {
        throw std::invalid_argument("Neumann augmented Cauchy edge values are invalid");
    }
    if (coefficients.rows() != surface_size_ || coefficients.cols() != 16
        || !coefficients.allFinite()) {
        throw std::invalid_argument(
            "Neumann augmented Cauchy coefficient matrix is invalid");
    }

    for (const auto& local : local_maps_) {
        Eigen::Matrix<double, 16, 1> row;
        for (int coefficient = 0; coefficient < 16; ++coefficient) {
            double value_sum = 0.0;
            for (int k = 0; k < 48; ++k) {
                value_sum += local.value_map(coefficient, k)
                    * value_jump[local.value_dofs[static_cast<std::size_t>(k)]];
            }
            double normal_sum = 0.0;
            for (int k = 0; k < 28; ++k) {
                normal_sum += local.normal_map(coefficient, k)
                    * normal_jump[local.normal_dofs[static_cast<std::size_t>(k)]];
            }
            double edge_sum = 0.0;
            for (int k = 0;
                 k < static_cast<int>(local.edge_sample_indices.size());
                 ++k) {
                edge_sum += local.edge_map(coefficient, k)
                    * edge_values_data[local.edge_sample_indices[
                        static_cast<std::size_t>(k)]];
            }
            row[coefficient] = value_sum + normal_sum + edge_sum;
        }
        if (!row.allFinite()) {
            throw std::runtime_error(
                "Neumann augmented Cauchy coefficient row is non-finite");
        }
        coefficients.row(local.center_dof) = row.transpose();
    }
}

NeumannEdgeAugmentedCauchy3D
build_neumann_edge_augmented_cauchy_3d(
    const NativeNurbsSurface3D& surface,
    const SurfaceDofCloud3D& cloud,
    double h,
    const std::vector<NeumannEdgeFaceStencil3D>& face_stencils,
    const NeumannEdgeAuxiliaryOptions3D& edge_options,
    const NeumannEdgeAugmentedCauchyOptions3D& local_options)
{
    validate_local_options(local_options);
    NeumannEdgeAugmentedCauchy3D result;
    result.surface_size_ = static_cast<int>(cloud.dofs.size());
    result.edge_value_map_ = build_neumann_edge_auxiliary_value_map_3d(
        surface, cloud, h, edge_options);
    validate_face_stencils(surface, cloud, face_stencils);
    result.diagnostics_.edge = result.edge_value_map_.diagnostics;
    result.diagnostics_.factorization_count = 2 * static_cast<int>(
        result.edge_value_map_.samples.size());
    result.diagnostics_.harmonic_cubic_reproduction_defect_max =
        result.diagnostics_.edge.harmonic_cubic_reproduction_defect_max;

    std::map<int, std::vector<int>> samples_by_connection;
    for (int index = 0;
         index < static_cast<int>(result.edge_value_map_.samples.size());
         ++index) {
        const auto& sample = result.edge_value_map_.samples[
            static_cast<std::size_t>(index)];
        samples_by_connection[sample.connection_index].push_back(index);
    }

    const HarmonicPolynomialSpace3D space(local_options.degree);
    for (int center_id = 0; center_id < result.surface_size_; ++center_id) {
        int connection_group_count = 0;
        std::vector<int> attached = attached_edge_samples(
            surface, cloud, face_stencils[static_cast<std::size_t>(center_id)],
            result.edge_value_map_, samples_by_connection, center_id, h,
            local_options.edge_samples_per_connection, connection_group_count);
        if (attached.empty())
            continue;
        if (attached.size()
            != static_cast<std::size_t>(connection_group_count
                * local_options.edge_samples_per_connection)) {
            throw std::runtime_error(
                "Neumann augmented Cauchy center " + std::to_string(center_id)
                + " has a partial edge sample group");
        }
        if (connection_group_count >= 2)
            ++result.diagnostics_.corner_center_count;

        NeumannEdgeLocalMap3D local;
        local.center_dof = center_id;
        local.value_dofs = face_stencils[static_cast<std::size_t>(center_id)].value_dofs;
        local.normal_dofs = face_stencils[static_cast<std::size_t>(center_id)].normal_dofs;
        local.edge_sample_indices = std::move(attached);
        const int edge_count = static_cast<int>(local.edge_sample_indices.size());
        const int rows = 48 + 28 + edge_count;
        Eigen::MatrixXd design(rows, space.dimension());
        Eigen::VectorXd sqrt_weights(rows);
        const auto& center = cloud.dofs[static_cast<std::size_t>(center_id)];
        append_local_value_rows(
            design, sqrt_weights, 0, local.value_dofs,
            center, cloud, h, space);
        append_local_normal_rows(
            design, sqrt_weights, 48, local.normal_dofs,
            center, cloud, h, space);
        for (int k = 0; k < edge_count; ++k) {
            const auto& sample = result.edge_value_map_.samples[
                static_cast<std::size_t>(local.edge_sample_indices[
                    static_cast<std::size_t>(k)])];
            const Eigen::Vector3d xi = center_coordinate(center, sample.point, h);
            design.row(76 + k) =
                space.basis(xi.x(), xi.y(), xi.z()).transpose();
            sqrt_weights[76 + k] = std::sqrt(
                local_options.edge_weight_scale
                / std::pow(0.35 + xi.norm(), 2.0));
        }

        const Eigen::MatrixXd weighted = sqrt_weights.asDiagonal() * design;
        Eigen::JacobiSVD<Eigen::MatrixXd> condition_svd(weighted);
        const Eigen::VectorXd singular = condition_svd.singularValues();
        result.diagnostics_.factorization_count += 2;
        if (singular.size() != space.dimension() || singular.size() == 0
            || !singular.allFinite() || !(singular[0] > 0.0)
            || !(singular[singular.size() - 1]
                > local_options.rank_relative_cutoff * singular[0])) {
            ++result.diagnostics_.rank_deficient_local_fit_count;
            throw std::runtime_error(
                "Neumann augmented Cauchy rank-deficient local fit at center "
                + std::to_string(center_id));
        }
        local.condition = singular[0] / singular[singular.size() - 1];
        const Eigen::MatrixXd pinv = svd_pseudoinverse_3d(
            weighted, local_options.rank_relative_cutoff);
        local.value_map.resize(space.dimension(), 48);
        local.normal_map.resize(space.dimension(), 28);
        local.edge_map.resize(space.dimension(), edge_count);
        for (int k = 0; k < 48; ++k)
            local.value_map.col(k) = pinv.col(k) * sqrt_weights[k];
        for (int k = 0; k < 28; ++k) {
            local.normal_map.col(k) =
                pinv.col(48 + k) * sqrt_weights[48 + k] * h;
        }
        for (int k = 0; k < edge_count; ++k)
            local.edge_map.col(k) = pinv.col(76 + k) * sqrt_weights[76 + k];
        if (!local.value_map.allFinite() || !local.normal_map.allFinite()
            || !local.edge_map.allFinite() || !std::isfinite(local.condition)) {
            throw std::runtime_error(
                "Neumann augmented Cauchy local map is non-finite at center "
                + std::to_string(center_id));
        }
        detail::validate_neumann_edge_local_map_3d(
            local, result.surface_size_, result.edge_sample_count());
        result.diagnostics_.local_condition_max = std::max(
            result.diagnostics_.local_condition_max, local.condition);
        result.local_maps_.push_back(std::move(local));
    }
    result.diagnostics_.affected_center_count =
        static_cast<int>(result.local_maps_.size());

    std::vector<int> center_to_local(
        static_cast<std::size_t>(result.surface_size_), -1);
    for (int index = 0; index < static_cast<int>(result.local_maps_.size()); ++index) {
        const int center = result.local_maps_[static_cast<std::size_t>(index)].center_dof;
        if (center_to_local[static_cast<std::size_t>(center)] != -1)
            throw std::runtime_error("Neumann augmented Cauchy center occurs more than once");
        center_to_local[static_cast<std::size_t>(center)] = index;
    }
    for (int sample_index = 0;
         sample_index < static_cast<int>(result.edge_value_map_.samples.size());
         ++sample_index) {
        const auto& sample = result.edge_value_map_.samples[
            static_cast<std::size_t>(sample_index)];
        for (const int owner : {sample.first_owner_dof, sample.second_owner_dof}) {
            if (owner < 0 || owner >= result.surface_size_
                || center_to_local[static_cast<std::size_t>(owner)] < 0) {
                throw std::runtime_error(
                    "Neumann augmented Cauchy edge sample owner lacks a local map");
            }
            const auto& local = result.local_maps_[static_cast<std::size_t>(
                center_to_local[static_cast<std::size_t>(owner)])];
            if (std::find(local.edge_sample_indices.begin(),
                          local.edge_sample_indices.end(), sample_index)
                == local.edge_sample_indices.end()) {
                throw std::runtime_error(
                    "Neumann augmented Cauchy edge sample owner lacks its sample");
            }
        }
    }

    for (const auto& local : result.local_maps_) {
        const auto& center = cloud.dofs[static_cast<std::size_t>(local.center_dof)];
        for (int basis_column = 0; basis_column < space.dimension(); ++basis_column) {
            Eigen::VectorXd predicted = Eigen::VectorXd::Zero(space.dimension());
            for (int k = 0; k < 48; ++k) {
                const auto& sample = cloud.dofs[static_cast<std::size_t>(
                    local.value_dofs[static_cast<std::size_t>(k)])];
                const Eigen::Vector3d xi = center_coordinate(center, sample.point, h);
                predicted += local.value_map.col(k)
                    * space.basis(xi.x(), xi.y(), xi.z())[basis_column];
            }
            for (int k = 0; k < 28; ++k) {
                const auto& sample = cloud.dofs[static_cast<std::size_t>(
                    local.normal_dofs[static_cast<std::size_t>(k)])];
                const Eigen::Vector3d xi = center_coordinate(center, sample.point, h);
                const Eigen::Vector3d normal_components(
                    sample.normal.dot(center.tangent1),
                    sample.normal.dot(center.tangent2),
                    sample.normal.dot(center.normal));
                const double physical_normal = normal_components.dot(
                    space.gradient(xi.x(), xi.y(), xi.z()).col(basis_column)) / h;
                predicted += local.normal_map.col(k) * physical_normal;
            }
            for (int k = 0;
                 k < static_cast<int>(local.edge_sample_indices.size());
                 ++k) {
                const auto& sample = result.edge_value_map_.samples[
                    static_cast<std::size_t>(local.edge_sample_indices[
                        static_cast<std::size_t>(k)])];
                const Eigen::Vector3d xi = center_coordinate(center, sample.point, h);
                predicted += local.edge_map.col(k)
                    * space.basis(xi.x(), xi.y(), xi.z())[basis_column];
            }
            Eigen::VectorXd exact = Eigen::VectorXd::Zero(space.dimension());
            exact[basis_column] = 1.0;
            result.diagnostics_.harmonic_cubic_reproduction_defect_max = std::max(
                result.diagnostics_.harmonic_cubic_reproduction_defect_max,
                (predicted - exact).lpNorm<Eigen::Infinity>());
            const Eigen::VectorXd center_basis = space.basis(0.0, 0.0, 0.0);
            result.diagnostics_.harmonic_cubic_reproduction_defect_max = std::max(
                result.diagnostics_.harmonic_cubic_reproduction_defect_max,
                std::abs(center_basis.dot(predicted)
                    - center_basis[basis_column]));
            for (const int edge_index : local.edge_sample_indices) {
                const auto& edge_sample = result.edge_value_map_.samples[
                    static_cast<std::size_t>(edge_index)];
                const Eigen::Vector3d edge_xi = center_coordinate(
                    center, edge_sample.point, h);
                const Eigen::VectorXd edge_basis = space.basis(
                    edge_xi.x(), edge_xi.y(), edge_xi.z());
                result.diagnostics_.harmonic_cubic_reproduction_defect_max =
                    std::max(
                        result.diagnostics_.harmonic_cubic_reproduction_defect_max,
                        std::abs(edge_basis.dot(predicted)
                            - edge_basis[basis_column]));
            }
        }
    }

    result.diagnostics_.pass = result.diagnostics_.edge.pass
        && result.diagnostics_.affected_center_count > 0
        && result.diagnostics_.corner_center_count > 0
        && result.diagnostics_.unrelated_attachment_count == 0
        && result.diagnostics_.rank_deficient_local_fit_count == 0
        && result.diagnostics_.factorization_count
            == 2 * (static_cast<int>(result.edge_value_map_.samples.size())
                + result.diagnostics_.affected_center_count)
        && result.diagnostics_.harmonic_cubic_reproduction_defect_max <= 1.0e-11
        && std::isfinite(result.diagnostics_.local_condition_max);
    return result;
}
} // namespace kfbim::app3d
