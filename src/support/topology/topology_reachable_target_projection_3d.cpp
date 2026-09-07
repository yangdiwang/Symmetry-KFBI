#include "src/support/topology/topology_reachable_target_projection_3d.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace kfbim::app3d {
namespace {

double vector_linf(const Eigen::VectorXd& values)
{
    return values.size() == 0 ? 0.0
                              : values.lpNorm<Eigen::Infinity>();
}

void validate_options(const ReachableTargetProjectionOptions3D& options)
{
    if (!(options.relative_rank_tolerance > 0.0)
        || !(options.relative_rank_tolerance < 1.0)
        || !std::isfinite(options.relative_rank_tolerance)) {
        throw std::invalid_argument(
            "reachable-target relative rank tolerance must be in (0,1)");
    }
    for (const auto& value_and_name : {
             std::pair<double, const char*>{
                 options.exact_tolerance, "exact tolerance"},
             {options.certification_tolerance,
              "certification tolerance"}}) {
        if (!(value_and_name.first >= 0.0)
            || !std::isfinite(value_and_name.first)) {
            throw std::invalid_argument(
                std::string("reachable-target ") + value_and_name.second
                + " must be finite and nonnegative");
        }
    }
}

void require_partition(Eigen::Index rows,
                       const std::vector<ConstraintBlock3D>& blocks)
{
    std::vector<int> count(static_cast<std::size_t>(rows), 0);
    for (const ConstraintBlock3D& block : blocks) {
        if (block.row_ids.empty())
            throw std::invalid_argument("reachable-target block is empty");
        for (const Eigen::Index row : block.row_ids) {
            if (row < 0 || row >= rows)
                throw std::out_of_range(
                    "reachable-target block row is out of range");
            ++count[static_cast<std::size_t>(row)];
        }
    }
    for (Eigen::Index row = 0; row < rows; ++row) {
        if (count[static_cast<std::size_t>(row)] != 1) {
            throw std::invalid_argument(
                "reachable-target blocks must partition every target row");
        }
    }
}

} // namespace

ReachableConstraintTargetProjection3D
project_constraint_target_to_reachable_space_3d(
    const AffineReduction3D& fixed_topology,
    ConstraintSystem3D target_system,
    std::vector<ConstraintBlock3D> target_blocks,
    ReachableTargetProjectionOptions3D options)
{
    validate_options(options);
    target_system.validate();
    target_system.C.makeCompressed();
    if (target_system.C.cols() != fixed_topology.base_size()) {
        throw std::invalid_argument(
            "reachable-target and fixed-topology base sizes differ");
    }
    if (target_blocks.empty() && target_system.C.rows() != 0) {
        ConstraintBlock3D all;
        all.kind = ConstraintBlockKind3D::Edge;
        all.key = "all_target_rows";
        all.row_ids.resize(
            static_cast<std::size_t>(target_system.C.rows()));
        std::iota(all.row_ids.begin(), all.row_ids.end(), Eigen::Index{0});
        target_blocks.push_back(std::move(all));
    }
    require_partition(target_system.C.rows(), target_blocks);

    SparseMatrixCSR3D reachable_map =
        target_system.C * fixed_topology.homogeneous_base();
    reachable_map.makeCompressed();
    target_blocks = merge_constraint_blocks_by_support_3d(
        reachable_map, std::move(target_blocks), 0.0);

    ReachableConstraintTargetProjection3D result;
    result.projected_system = target_system;
    result.requested_rhs = target_system.d;
    result.projected_rhs = target_system.C
        * fixed_topology.particular_base();
    result.reachable_coordinates = Eigen::VectorXd::Zero(
        fixed_topology.reduced_size());
    result.records.reserve(target_blocks.size());

    const Eigen::VectorXd requested_reduced_rhs =
        result.requested_rhs - result.projected_rhs;
    for (const ConstraintBlock3D& block : target_blocks) {
        std::vector<Eigen::Index> columns;
        for (const Eigen::Index row : block.row_ids) {
            for (SparseMatrixCSR3D::InnerIterator entry(
                     reachable_map, row); entry; ++entry) {
                if (entry.value() != 0.0)
                    columns.push_back(entry.col());
            }
        }
        std::sort(columns.begin(), columns.end());
        columns.erase(std::unique(columns.begin(), columns.end()),
                      columns.end());

        const Eigen::Index local_rows =
            static_cast<Eigen::Index>(block.row_ids.size());
        Eigen::VectorXd local_requested(local_rows);
        for (Eigen::Index row = 0; row < local_rows; ++row) {
            local_requested[row] = requested_reduced_rhs[
                block.row_ids[static_cast<std::size_t>(row)]];
        }

        Eigen::VectorXd local_projected = Eigen::VectorXd::Zero(local_rows);
        Eigen::Index rank = 0;
        if (!columns.empty()) {
            std::unordered_map<Eigen::Index, Eigen::Index> local_column;
            local_column.reserve(columns.size());
            for (Eigen::Index column = 0;
                 column < static_cast<Eigen::Index>(columns.size());
                 ++column) {
                local_column.emplace(
                    columns[static_cast<std::size_t>(column)], column);
            }
            Eigen::MatrixXd local_matrix = Eigen::MatrixXd::Zero(
                local_rows, static_cast<Eigen::Index>(columns.size()));
            for (Eigen::Index row = 0; row < local_rows; ++row) {
                const Eigen::Index source =
                    block.row_ids[static_cast<std::size_t>(row)];
                for (SparseMatrixCSR3D::InnerIterator entry(
                         reachable_map, source); entry; ++entry) {
                    local_matrix(row, local_column.at(entry.col())) +=
                        entry.value();
                }
            }
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                local_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
            const Eigen::VectorXd singular = svd.singularValues();
            if (singular.size() > 0) {
                const double cutoff = options.relative_rank_tolerance
                    * singular[0];
                while (rank < singular.size() && singular[rank] > cutoff)
                    ++rank;
            }
            Eigen::VectorXd local_coordinates = Eigen::VectorXd::Zero(
                static_cast<Eigen::Index>(columns.size()));
            if (rank > 0) {
                local_coordinates = svd.matrixV().leftCols(rank)
                    * (singular.head(rank).cwiseInverse().asDiagonal()
                       * (svd.matrixU().leftCols(rank).transpose()
                          * local_requested));
                local_projected = local_matrix * local_coordinates;
            }
            for (Eigen::Index column = 0;
                 column < static_cast<Eigen::Index>(columns.size());
                 ++column) {
                result.reachable_coordinates[
                    columns[static_cast<std::size_t>(column)]] =
                    local_coordinates[column];
            }
        }
        for (Eigen::Index row = 0; row < local_rows; ++row) {
            result.projected_rhs[
                block.row_ids[static_cast<std::size_t>(row)]] +=
                local_projected[row];
        }

        const Eigen::VectorXd local_residual =
            local_requested - local_projected;
        ReachableTargetProjectionRecord3D record;
        record.kind = block.kind;
        record.key = block.key;
        record.source_keys = block.source_keys;
        record.row_ids = block.row_ids;
        record.rows = local_rows;
        record.support_cols =
            static_cast<Eigen::Index>(columns.size());
        record.rank = rank;
        record.requested_l2 = local_requested.norm();
        record.projection_l2 = local_residual.norm();
        record.projection_linf = vector_linf(local_residual);
        result.records.push_back(std::move(record));
    }

    result.projected_system.d = result.projected_rhs;
    result.projected_system.validate();
    result.requested_linf = vector_linf(result.requested_rhs);
    result.requested_l2 = result.requested_rhs.norm();
    result.projected_linf = vector_linf(result.projected_rhs);
    result.projected_l2 = result.projected_rhs.norm();
    const Eigen::VectorXd residual =
        result.requested_rhs - result.projected_rhs;
    result.projection_linf = vector_linf(residual);
    result.projection_l2 = residual.norm();
    result.relative_projection_l2 = result.projection_l2
        / (result.requested_rhs.norm()
           + std::numeric_limits<double>::epsilon());
    const double exact_scale = std::max(1.0, vector_linf(
        result.requested_rhs));
    result.constraints_exact = result.projection_linf
        <= options.exact_tolerance * exact_scale;

    const Eigen::VectorXd reachable_base =
        fixed_topology.particular_base()
        + fixed_topology.homogeneous_base()
              * result.reachable_coordinates;
    result.reachable_certification_linf = vector_linf(
        target_system.C * reachable_base - result.projected_rhs);
    const double certification_scale = std::max(
        1.0, vector_linf(result.projected_rhs));
    if (!std::isfinite(result.reachable_certification_linf)
        || result.reachable_certification_linf
               > options.certification_tolerance * certification_scale) {
        throw std::runtime_error(
            "component-local reachable-target projection failed its "
            "reachability certification");
    }
    return result;
}

} // namespace kfbim::app3d
