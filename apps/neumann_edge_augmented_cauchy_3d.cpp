#include "neumann_edge_augmented_cauchy_3d.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace kfbim::app3d {
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

void validate_cloud(const NativeNurbsSurface3D& surface,
                    const SurfaceDofCloud3D& cloud)
{
    if (surface.patches.empty()
        || surface.patch_names.size() != surface.patches.size()
        || surface.smooth_neighbors.size() != surface.patches.size()
        || cloud.patches.size() != surface.patches.size()
        || cloud.dofs.empty()) {
        throw std::invalid_argument("Neumann auxiliary surface/cloud metadata is inconsistent");
    }
    for (const auto& dof : cloud.dofs) {
        if (dof.patch_id < 0
            || dof.patch_id >= static_cast<int>(surface.patches.size())
            || !dof.point.allFinite() || !dof.normal.allFinite()) {
            throw std::invalid_argument("Neumann auxiliary cloud contains an invalid DOF");
        }
    }
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

} // namespace

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
    validate_cloud(surface, cloud);
    for (int c = 0; c < static_cast<int>(surface.geometric_connections.size()); ++c) {
        validate_interval(surface, surface.geometric_connections[static_cast<std::size_t>(c)].first, c, "first");
        validate_interval(surface, surface.geometric_connections[static_cast<std::size_t>(c)].second, c, "second");
    }

    const double geometry_diameter = surface.geometry_model().control_bounds().diameter();
    if (!std::isfinite(geometry_diameter) || geometry_diameter <= 0.0)
        throw std::invalid_argument("Neumann auxiliary surface has invalid diameter");
    const double gap_limit = 1.0e-11 * geometry_diameter;
    HarmonicPolynomialSpace3D space(options.degree);
    const Eigen::VectorXd origin_basis = space.basis(0.0, 0.0, 0.0);

    NeumannEdgeAuxiliaryValueMap3D result;
    result.surface_size = static_cast<int>(cloud.dofs.size());
    std::vector<Eigen::Triplet<double>> value_entries;
    std::vector<Eigen::Triplet<double>> normal_entries;

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
        const int sample_count = std::max(options.minimum_edge_samples,
            static_cast<int>(std::ceil(length / h)));

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

            try {
                sample.first_value_dofs = nearest_g1_side_dofs(
                    surface, cloud, sample.first_patch, sample.second_patch,
                    sample.point, options.value_samples_per_side);
                sample.first_normal_dofs = nearest_g1_side_dofs(
                    surface, cloud, sample.first_patch, sample.second_patch,
                    sample.point, options.normal_samples_per_side);
                sample.second_value_dofs = nearest_g1_side_dofs(
                    surface, cloud, sample.second_patch, sample.first_patch,
                    sample.point, options.value_samples_per_side);
                sample.second_normal_dofs = nearest_g1_side_dofs(
                    surface, cloud, sample.second_patch, sample.first_patch,
                    sample.point, options.normal_samples_per_side);
            } catch (const std::exception& error) {
                throw std::runtime_error(sample_context(c, q, error.what()));
            }
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

} // namespace kfbim::app3d
