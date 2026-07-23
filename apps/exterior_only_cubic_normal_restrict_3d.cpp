#include "exterior_only_cubic_normal_restrict_3d.hpp"

#include "harmonic_polynomial_space_3d.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace kfbim::app3d {
namespace {

constexpr int kCubicHarmonicDimension = 16;

void validate_frame(const Eigen::Vector3d& tangent1,
                    const Eigen::Vector3d& tangent2,
                    const Eigen::Vector3d& normal)
{
    if (!tangent1.allFinite() || !tangent2.allFinite()
        || !normal.allFinite()) {
        throw std::invalid_argument(
            "exterior-only restrict frame must be finite");
    }
    const double norm_error = std::max({
        std::abs(tangent1.norm() - 1.0),
        std::abs(tangent2.norm() - 1.0),
        std::abs(normal.norm() - 1.0)});
    const double orthogonality_error = std::max({
        std::abs(tangent1.dot(tangent2)),
        std::abs(tangent1.dot(normal)),
        std::abs(tangent2.dot(normal))});
    const double orientation = tangent1.cross(tangent2).dot(normal);
    if (norm_error > 1.0e-10 || orthogonality_error > 1.0e-10
        || orientation < 1.0 - 1.0e-10) {
        throw std::invalid_argument(
            "exterior-only restrict requires a right-handed orthonormal frame");
    }
}

Eigen::Vector3d grid_point(const CartesianGrid3D& grid, int node)
{
    const std::array<double, 3> coordinate = grid.coord(node);
    return {coordinate[0], coordinate[1], coordinate[2]};
}

} // namespace

ExteriorOnlyCubicNormalStencil3D
build_exterior_only_cubic_normal_stencil_3d(
    const Eigen::Vector3d& center,
    const Eigen::Vector3d& tangent1,
    const Eigen::Vector3d& tangent2,
    const Eigen::Vector3d& normal,
    double h,
    const std::vector<std::pair<int, Eigen::Vector3d>>& exterior_candidates,
    int sample_count)
{
    if (!center.allFinite() || !std::isfinite(h) || !(h > 0.0))
        throw std::invalid_argument(
            "exterior-only restrict requires a finite center and positive h");
    validate_frame(tangent1, tangent2, normal);
    if (sample_count < kCubicHarmonicDimension
        || sample_count > static_cast<int>(exterior_candidates.size())) {
        throw std::invalid_argument(
            "exterior-only restrict has an invalid sample count");
    }
    if (!std::all_of(
            exterior_candidates.begin(), exterior_candidates.end(),
            [](const auto& candidate) {
                return candidate.first >= 0 && candidate.second.allFinite();
            })) {
        throw std::invalid_argument(
            "exterior-only restrict candidate is invalid");
    }

    std::vector<std::pair<int, Eigen::Vector3d>> candidates =
        exterior_candidates;
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [&](const auto& first, const auto& second) {
            const double first_distance =
                (first.second - center).squaredNorm();
            const double second_distance =
                (second.second - center).squaredNorm();
            if (first_distance != second_distance)
                return first_distance < second_distance;
            return first.first < second.first;
        });
    candidates.resize(static_cast<std::size_t>(sample_count));

    const HarmonicPolynomialSpace3D space(3);
    Eigen::MatrixXd design(sample_count, space.dimension());
    Eigen::VectorXd sqrt_weights(sample_count);
    for (int row = 0; row < sample_count; ++row) {
        const Eigen::Vector3d displacement =
            (candidates[static_cast<std::size_t>(row)].second - center) / h;
        const Eigen::Vector3d local(
            displacement.dot(tangent1),
            displacement.dot(tangent2),
            displacement.dot(normal));
        design.row(row) =
            space.basis(local.x(), local.y(), local.z()).transpose();
        sqrt_weights[row] =
            1.0 / (0.35 + local.norm());
    }
    const Eigen::MatrixXd weighted =
        sqrt_weights.asDiagonal() * design;
    Eigen::JacobiSVD<Eigen::MatrixXd> condition_svd(weighted);
    const Eigen::VectorXd singular = condition_svd.singularValues();
    if (singular.size() != space.dimension() || !singular.allFinite()
        || !(singular[0] > 0.0)
        || !(singular[singular.size() - 1] > 3.0e-12 * singular[0])) {
        throw std::runtime_error(
            "exterior-only cubic normal restrict is rank deficient");
    }

    ExteriorOnlyCubicNormalStencil3D result;
    result.grid_ids.reserve(static_cast<std::size_t>(sample_count));
    for (const auto& candidate : candidates)
        result.grid_ids.push_back(candidate.first);
    result.condition = singular[0] / singular[singular.size() - 1];
    const Eigen::RowVectorXd derivative =
        space.gradient(0.0, 0.0, 0.0).row(2) / h;
    result.weights =
        (derivative
         * svd_pseudoinverse_3d(weighted, 3.0e-12)
         * sqrt_weights.asDiagonal())
            .transpose();
    if (!result.weights.allFinite()) {
        throw std::runtime_error(
            "exterior-only cubic normal restrict produced invalid weights");
    }
    return result;
}

ExteriorOnlyCubicNormalRestrict3D::ExteriorOnlyCubicNormalRestrict3D(
    const CartesianGrid3D& grid,
    const GridPair3D& grid_pair,
    const SurfaceDofCloud3D& cloud,
    int sample_count,
    int maximum_index_radius)
    : grid_dof_count_(grid.num_dofs())
{
    const std::array<double, 3> spacing = grid.spacing();
    const double h = spacing[0];
    if (std::abs(spacing[1] - h) > 1.0e-13
        || std::abs(spacing[2] - h) > 1.0e-13) {
        throw std::invalid_argument(
            "exterior-only restrict requires an isotropic grid");
    }
    if (sample_count < kCubicHarmonicDimension
        || maximum_index_radius < 1) {
        throw std::invalid_argument(
            "exterior-only restrict configuration is invalid");
    }
    const std::array<double, 3> origin = grid.origin();
    const std::array<int, 3> dims = grid.dof_dims();
    stencils_.reserve(cloud.dofs.size());
    for (int center_id = 0;
         center_id < static_cast<int>(cloud.dofs.size());
         ++center_id) {
        const SurfaceDof3D& dof =
            cloud.dofs[static_cast<std::size_t>(center_id)];
        std::array<int, 3> nearest{};
        for (int axis = 0; axis < 3; ++axis) {
            const double coordinate =
                (dof.point[axis] - origin[static_cast<std::size_t>(axis)]) / h;
            nearest[static_cast<std::size_t>(axis)] =
                static_cast<int>(std::llround(coordinate));
        }

        std::vector<std::pair<int, Eigen::Vector3d>> candidates;
        for (int k = std::max(0, nearest[2] - maximum_index_radius);
             k <= std::min(
                 dims[2] - 1, nearest[2] + maximum_index_radius);
             ++k) {
            for (int j = std::max(0, nearest[1] - maximum_index_radius);
                 j <= std::min(
                     dims[1] - 1, nearest[1] + maximum_index_radius);
                 ++j) {
                for (int i = std::max(0, nearest[0] - maximum_index_radius);
                     i <= std::min(
                         dims[0] - 1, nearest[0] + maximum_index_radius);
                     ++i) {
                    const int node = grid.index(i, j, k);
                    if (grid_pair.domain_label(node) == 0)
                        candidates.emplace_back(node, grid_point(grid, node));
                }
            }
        }
        if (static_cast<int>(candidates.size()) < sample_count) {
            throw std::runtime_error(
                "exterior-only restrict has too few exterior nodes at DOF "
                + std::to_string(center_id));
        }
        stencils_.push_back(
            build_exterior_only_cubic_normal_stencil_3d(
                dof.point,
                dof.tangent1,
                dof.tangent2,
                dof.normal,
                h,
                candidates,
                sample_count));
    }
}

Eigen::VectorXd ExteriorOnlyCubicNormalRestrict3D::apply(
    const Eigen::VectorXd& potential) const
{
    if (potential.size() != grid_dof_count_) {
        throw std::invalid_argument(
            "exterior-only restrict potential has the wrong size");
    }
    Eigen::VectorXd result(static_cast<int>(stencils_.size()));
    for (int center = 0; center < static_cast<int>(stencils_.size()); ++center) {
        const ExteriorOnlyCubicNormalStencil3D& stencil =
            stencils_[static_cast<std::size_t>(center)];
        double value = 0.0;
        for (int k = 0; k < static_cast<int>(stencil.grid_ids.size()); ++k) {
            value += stencil.weights[k]
                   * potential[
                       stencil.grid_ids[static_cast<std::size_t>(k)]];
        }
        result[center] = value;
    }
    return result;
}

const std::vector<ExteriorOnlyCubicNormalStencil3D>&
ExteriorOnlyCubicNormalRestrict3D::stencils() const noexcept
{
    return stencils_;
}

} // namespace kfbim::app3d
