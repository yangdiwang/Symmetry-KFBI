#include "topology_affine_reduction_3d.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace kfbim::app3d {
namespace {

double vector_linf(const Eigen::VectorXd& values)
{
    return values.size() == 0
        ? 0.0
        : values.lpNorm<Eigen::Infinity>();
}

double dense_max_abs(const Eigen::MatrixXd& values)
{
    return values.size() == 0 ? 0.0 : values.cwiseAbs().maxCoeff();
}

double sparse_max_abs(const SparseMatrixCSR3D& matrix)
{
    double result = 0.0;
    for (Eigen::Index row = 0; row < matrix.outerSize(); ++row) {
        for (SparseMatrixCSR3D::InnerIterator entry(matrix, row); entry;
             ++entry) {
            result = std::max(result, std::abs(entry.value()));
        }
    }
    return result;
}

bool sparse_all_finite(const SparseMatrixCSR3D& matrix)
{
    for (Eigen::Index row = 0; row < matrix.outerSize(); ++row) {
        for (SparseMatrixCSR3D::InnerIterator entry(matrix, row); entry;
             ++entry) {
            if (!std::isfinite(entry.value()))
                return false;
        }
    }
    return true;
}

void validate_nonnegative_tolerance(double value, const char* name)
{
    if (!(value >= 0.0) || !std::isfinite(value)) {
        throw std::invalid_argument(
            std::string(name) + " must be finite and nonnegative");
    }
}

void validate_positive_tolerance(double value, const char* name)
{
    if (!(value > 0.0) || !std::isfinite(value)) {
        throw std::invalid_argument(
            std::string(name) + " must be finite and positive");
    }
}

void validate_options(const AffineReductionOptions3D& options)
{
    validate_nonnegative_tolerance(
        options.zero_row_tolerance, "zero-row tolerance");
    validate_nonnegative_tolerance(
        options.rhs_tolerance, "RHS tolerance");
    if (!(options.relative_rank_tolerance > 0.0)
        || !(options.relative_rank_tolerance < 1.0)
        || !std::isfinite(options.relative_rank_tolerance)) {
        throw std::invalid_argument(
            "relative rank tolerance must be finite and in (0,1)");
    }
    validate_positive_tolerance(
        options.consistency_tolerance, "consistency tolerance");
    validate_nonnegative_tolerance(
        options.sparse_prune_tolerance, "sparse prune tolerance");
    validate_positive_tolerance(
        options.invariant_tolerance, "invariant tolerance");
    if (options.sparse_prune_tolerance > options.zero_row_tolerance
        && options.zero_row_tolerance > 0.0) {
        throw std::invalid_argument(
            "sparse prune tolerance may not exceed zero-row tolerance");
    }
}

SparseMatrixCSR3D extract_rows(
    const SparseMatrixCSR3D& matrix,
    const std::vector<Eigen::Index>& row_ids)
{
    std::vector<Eigen::Triplet<double>> triplets;
    Eigen::Index reservation = 0;
    for (const Eigen::Index row : row_ids)
        reservation += matrix.innerVector(row).nonZeros();
    triplets.reserve(static_cast<std::size_t>(reservation));
    for (Eigen::Index local_row = 0;
         local_row < static_cast<Eigen::Index>(row_ids.size());
         ++local_row) {
        const Eigen::Index source_row =
            row_ids[static_cast<std::size_t>(local_row)];
        for (SparseMatrixCSR3D::InnerIterator entry(matrix, source_row);
             entry; ++entry) {
            triplets.emplace_back(local_row, entry.col(), entry.value());
        }
    }
    SparseMatrixCSR3D result(
        static_cast<Eigen::Index>(row_ids.size()), matrix.cols());
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();
    return result;
}

SparseMatrixCSR3D prune_sparse(const SparseMatrixCSR3D& matrix,
                               double tolerance)
{
    if (tolerance == 0.0) {
        SparseMatrixCSR3D result = matrix;
        result.makeCompressed();
        return result;
    }
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(matrix.nonZeros()));
    for (Eigen::Index row = 0; row < matrix.outerSize(); ++row) {
        for (SparseMatrixCSR3D::InnerIterator entry(matrix, row); entry;
             ++entry) {
            if (std::abs(entry.value()) > tolerance) {
                triplets.emplace_back(
                    entry.row(), entry.col(), entry.value());
            }
        }
    }
    SparseMatrixCSR3D result(matrix.rows(), matrix.cols());
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();
    return result;
}

std::pair<double, double> processed_invariants(
    const ConstraintSystem3D& constraints,
    const Eigen::VectorXd& particular,
    const SparseMatrixCSR3D& homogeneous,
    const std::vector<Eigen::Index>& processed_rows)
{
    if (processed_rows.empty())
        return {0.0, 0.0};
    const SparseMatrixCSR3D selected =
        extract_rows(constraints.C, processed_rows);
    Eigen::VectorXd selected_rhs(
        static_cast<Eigen::Index>(processed_rows.size()));
    for (Eigen::Index local = 0;
         local < static_cast<Eigen::Index>(processed_rows.size()); ++local) {
        selected_rhs[local] =
            constraints.d[processed_rows[static_cast<std::size_t>(local)]];
    }
    const double particular_residual = vector_linf(
        selected * particular - selected_rhs);
    const SparseMatrixCSR3D homogeneous_residual = selected * homogeneous;
    return {particular_residual, sparse_max_abs(homogeneous_residual)};
}

double compute_homogeneous_orthogonality_residual(
    const SparseMatrixCSR3D& homogeneous)
{
    const Eigen::Index columns = homogeneous.cols();
    if (columns == 0)
        return 0.0;
    SparseMatrixCSR3D gram = homogeneous.transpose() * homogeneous;
    gram.makeCompressed();
    std::vector<double> diagonal(
        static_cast<std::size_t>(columns), 0.0);
    double residual = 0.0;
    for (Eigen::Index row = 0; row < gram.outerSize(); ++row) {
        for (SparseMatrixCSR3D::InnerIterator entry(gram, row); entry;
             ++entry) {
            if (entry.row() == entry.col()) {
                diagonal[static_cast<std::size_t>(entry.row())] +=
                    entry.value();
            } else {
                residual = std::max(residual, std::abs(entry.value()));
            }
        }
    }
    for (const double value : diagonal)
        residual = std::max(residual, std::abs(value - 1.0));
    return residual;
}

std::string block_name(const ConstraintBlock3D& block)
{
    return block.key.empty() ? std::string("unnamed") : block.key;
}

class DisjointSet {
public:
    explicit DisjointSet(std::size_t size)
        : parent_(size), rank_(size, 0)
    {
        std::iota(parent_.begin(), parent_.end(), std::size_t{0});
    }

    std::size_t find(std::size_t value)
    {
        if (parent_[value] != value)
            parent_[value] = find(parent_[value]);
        return parent_[value];
    }

    void unite(std::size_t first, std::size_t second)
    {
        first = find(first);
        second = find(second);
        if (first == second)
            return;
        if (rank_[first] < rank_[second])
            std::swap(first, second);
        parent_[second] = first;
        if (rank_[first] == rank_[second])
            ++rank_[first];
    }

private:
    std::vector<std::size_t> parent_;
    std::vector<unsigned char> rank_;
};

void validate_block_rows(const SparseMatrixCSR3D& matrix,
                         const std::vector<ConstraintBlock3D>& blocks)
{
    for (const ConstraintBlock3D& block : blocks) {
        for (const Eigen::Index row : block.row_ids) {
            if (row < 0 || row >= matrix.rows()) {
                throw std::out_of_range(
                    "constraint block '" + block_name(block)
                    + "' contains an out-of-range row id");
            }
        }
    }
}

void require_complete_partition(
    Eigen::Index row_count,
    const std::vector<ConstraintBlock3D>& blocks)
{
    std::vector<int> occurrences(
        static_cast<std::size_t>(row_count), 0);
    for (const ConstraintBlock3D& block : blocks) {
        for (const Eigen::Index row : block.row_ids)
            ++occurrences[static_cast<std::size_t>(row)];
    }
    for (Eigen::Index row = 0; row < row_count; ++row) {
        const int count = occurrences[static_cast<std::size_t>(row)];
        if (count != 1) {
            std::ostringstream message;
            message << "constraint row " << row << " occurs " << count
                    << " times in the elimination block partition";
            throw std::invalid_argument(message.str());
        }
    }
}

} // namespace

Eigen::Index PatchDofLayout3D::size() const
{
    validate();
    return static_cast<Eigen::Index>(n_u)
        * static_cast<Eigen::Index>(n_v);
}

Eigen::Index PatchDofLayout3D::global_index(int i, int j) const
{
    validate();
    if (i < 0 || i >= n_u || j < 0 || j >= n_v)
        throw std::out_of_range("patch coefficient index is out of range");
    // u-major, v-contiguous indexing; decode() is its exact inverse.
    return offset + static_cast<Eigen::Index>(i) * n_v + j;
}

std::pair<int, int> PatchDofLayout3D::decode(
    Eigen::Index global_index_value) const
{
    validate();
    if (!contains(global_index_value))
        throw std::out_of_range("global coefficient is outside this patch");
    const Eigen::Index local = global_index_value - offset;
    return {static_cast<int>(local / n_v),
            static_cast<int>(local % n_v)};
}

bool PatchDofLayout3D::contains(Eigen::Index global_index_value) const
{
    validate();
    const Eigen::Index count = static_cast<Eigen::Index>(n_u) * n_v;
    return global_index_value >= offset
        && global_index_value - offset < count;
}

void PatchDofLayout3D::validate() const
{
    if (patch_id < 0 || n_u <= 0 || n_v <= 0 || offset < 0) {
        throw std::invalid_argument(
            "patch layout requires nonnegative id/offset and positive sizes");
    }
    const Eigen::Index maximum =
        std::numeric_limits<Eigen::Index>::max();
    if (static_cast<Eigen::Index>(n_u) > maximum / n_v) {
        throw std::overflow_error("patch coefficient count overflows Index");
    }
    const Eigen::Index count = static_cast<Eigen::Index>(n_u) * n_v;
    if (offset > maximum - count)
        throw std::overflow_error("patch global index range overflows Index");
}

void ConstraintSystem3D::validate() const
{
    if (C.rows() < 0 || C.cols() < 0 || d.size() != C.rows()
        || static_cast<Eigen::Index>(meta.size()) != C.rows()) {
        throw std::invalid_argument(
            "constraint matrix, RHS, and metadata shapes are inconsistent");
    }
    if (!d.allFinite() || !sparse_all_finite(C)) {
        throw std::invalid_argument(
            "constraint matrix and RHS must contain only finite values");
    }
}

SparseConstraintBuilder3D::SparseConstraintBuilder3D(
    Eigen::Index coordinate_count)
    : coordinate_count_(coordinate_count)
{
    if (coordinate_count_ < 0)
        throw std::invalid_argument("constraint coordinate count is negative");
}

void SparseConstraintBuilder3D::append_row(
    const std::vector<Eigen::Index>& column_ids,
    const std::vector<double>& values,
    double rhs,
    ConstraintMeta3D meta)
{
    if (column_ids.size() != values.size()) {
        throw std::invalid_argument(
            "constraint row column/value sizes are inconsistent");
    }
    std::vector<std::pair<Eigen::Index, double>> entries;
    entries.reserve(column_ids.size());
    for (std::size_t index = 0; index < column_ids.size(); ++index)
        entries.emplace_back(column_ids[index], values[index]);
    append_row(entries, rhs, std::move(meta));
}

void SparseConstraintBuilder3D::append_row(
    const std::vector<std::pair<Eigen::Index, double>>& entries,
    double rhs,
    ConstraintMeta3D meta)
{
    if (!std::isfinite(rhs))
        throw std::invalid_argument("constraint RHS must be finite");
    for (const auto& entry : entries) {
        if (entry.first < 0 || entry.first >= coordinate_count_) {
            throw std::out_of_range(
                "constraint row contains an out-of-range column");
        }
        if (!std::isfinite(entry.second)) {
            throw std::invalid_argument(
                "constraint coefficient must be finite");
        }
    }
    rows_.push_back({entries, rhs, std::move(meta)});
}

Eigen::Index SparseConstraintBuilder3D::coordinate_count() const noexcept
{
    return coordinate_count_;
}

Eigen::Index SparseConstraintBuilder3D::pending_row_count() const noexcept
{
    return static_cast<Eigen::Index>(rows_.size());
}

ConstraintSystem3D SparseConstraintBuilder3D::finalize(
    double zero_row_tolerance,
    double rhs_tolerance) const
{
    validate_nonnegative_tolerance(zero_row_tolerance,
                                   "zero-row tolerance");
    validate_nonnegative_tolerance(rhs_tolerance, "RHS tolerance");

    std::vector<Eigen::Triplet<double>> triplets;
    std::vector<double> retained_rhs;
    std::vector<ConstraintMeta3D> retained_meta;
    retained_rhs.reserve(rows_.size());
    retained_meta.reserve(rows_.size());

    for (std::size_t input_row = 0; input_row < rows_.size(); ++input_row) {
        const PendingRow& row = rows_[input_row];
        std::map<Eigen::Index, double> combined;
        for (const auto& entry : row.entries)
            combined[entry.first] += entry.second;

        double norm = 0.0;
        for (const auto& entry : combined)
            norm = std::hypot(norm, entry.second);
        if (norm <= zero_row_tolerance) {
            if (std::abs(row.rhs) > rhs_tolerance) {
                std::ostringstream message;
                message << "zero constraint row " << input_row
                        << " has nonzero RHS " << row.rhs;
                throw ConstraintIncompatibility3D(
                    message.str(), row.meta.segment,
                    static_cast<Eigen::Index>(input_row),
                    std::abs(row.rhs));
            }
            continue;
        }

        const Eigen::Index output_row =
            static_cast<Eigen::Index>(retained_rhs.size());
        for (const auto& entry : combined) {
            if (entry.second != 0.0)
                triplets.emplace_back(output_row, entry.first, entry.second);
        }
        retained_rhs.push_back(row.rhs);
        retained_meta.push_back(row.meta);
    }

    ConstraintSystem3D result;
    result.C.resize(
        static_cast<Eigen::Index>(retained_rhs.size()), coordinate_count_);
    result.C.setFromTriplets(triplets.begin(), triplets.end());
    result.C.makeCompressed();
    result.d.resize(static_cast<Eigen::Index>(retained_rhs.size()));
    for (Eigen::Index row = 0; row < result.d.size(); ++row)
        result.d[row] = retained_rhs[static_cast<std::size_t>(row)];
    result.meta = std::move(retained_meta);
    result.validate();
    return result;
}

std::vector<ConstraintBlock3D>
order_constraint_blocks_vertex_before_edge_3d(
    std::vector<ConstraintBlock3D> blocks)
{
    std::stable_partition(
        blocks.begin(), blocks.end(), [](const ConstraintBlock3D& block) {
            return block.kind == ConstraintBlockKind3D::Vertex;
        });
    return blocks;
}

std::vector<ConstraintBlock3D> merge_constraint_blocks_by_support_3d(
    const SparseMatrixCSR3D& support_matrix,
    std::vector<ConstraintBlock3D> blocks,
    double support_tolerance)
{
    validate_nonnegative_tolerance(
        support_tolerance, "block support tolerance");
    if (!sparse_all_finite(support_matrix))
        throw std::invalid_argument("block support matrix is not finite");
    blocks = order_constraint_blocks_vertex_before_edge_3d(
        std::move(blocks));
    validate_block_rows(support_matrix, blocks);
    if (blocks.empty())
        return blocks;

    DisjointSet components(blocks.size());
    std::unordered_map<Eigen::Index, std::size_t> column_owner;
    std::unordered_map<Eigen::Index, std::size_t> row_owner;
    for (std::size_t block_index = 0; block_index < blocks.size();
         ++block_index) {
        for (const Eigen::Index row : blocks[block_index].row_ids) {
            const auto row_inserted = row_owner.emplace(row, block_index);
            if (!row_inserted.second)
                components.unite(block_index, row_inserted.first->second);
            for (SparseMatrixCSR3D::InnerIterator entry(
                     support_matrix, row);
                 entry; ++entry) {
                if (std::abs(entry.value()) <= support_tolerance)
                    continue;
                const auto inserted =
                    column_owner.emplace(entry.col(), block_index);
                if (!inserted.second)
                    components.unite(block_index, inserted.first->second);
            }
        }
    }

    std::unordered_map<std::size_t, std::size_t> output_by_root;
    std::vector<ConstraintBlock3D> merged;
    merged.reserve(blocks.size());
    for (std::size_t block_index = 0; block_index < blocks.size();
         ++block_index) {
        const std::size_t root = components.find(block_index);
        auto inserted = output_by_root.emplace(root, merged.size());
        if (inserted.second) {
            ConstraintBlock3D output;
            output.kind = blocks[block_index].kind;
            output.key = blocks[block_index].key;
            merged.push_back(std::move(output));
        }
        ConstraintBlock3D& output = merged[inserted.first->second];
        if (blocks[block_index].kind == ConstraintBlockKind3D::Vertex)
            output.kind = ConstraintBlockKind3D::Vertex;
        output.row_ids.insert(output.row_ids.end(),
                              blocks[block_index].row_ids.begin(),
                              blocks[block_index].row_ids.end());
        if (blocks[block_index].source_keys.empty()) {
            output.source_keys.push_back(blocks[block_index].key);
        } else {
            output.source_keys.insert(output.source_keys.end(),
                                      blocks[block_index].source_keys.begin(),
                                      blocks[block_index].source_keys.end());
        }
    }

    for (ConstraintBlock3D& block : merged) {
        std::sort(block.row_ids.begin(), block.row_ids.end());
        block.row_ids.erase(
            std::unique(block.row_ids.begin(), block.row_ids.end()),
            block.row_ids.end());
        std::sort(block.source_keys.begin(), block.source_keys.end());
        block.source_keys.erase(
            std::unique(block.source_keys.begin(), block.source_keys.end()),
            block.source_keys.end());
        if (block.source_keys.size() > 1) {
            std::ostringstream key;
            key << "support{";
            for (std::size_t index = 0; index < block.source_keys.size();
                 ++index) {
                if (index != 0)
                    key << ',';
                key << block.source_keys[index];
            }
            key << '}';
            block.key = key.str();
        }
    }
    return order_constraint_blocks_vertex_before_edge_3d(std::move(merged));
}

ConstraintIncompatibility3D::ConstraintIncompatibility3D(
    std::string message,
    std::string block_key,
    Eigen::Index row_id,
    double residual)
    : std::runtime_error(std::move(message)),
      block_key_(std::move(block_key)),
      row_id_(row_id),
      residual_(residual)
{
}

const std::string& ConstraintIncompatibility3D::block_key() const noexcept
{
    return block_key_;
}

Eigen::Index ConstraintIncompatibility3D::row_id() const noexcept
{
    return row_id_;
}

double ConstraintIncompatibility3D::residual() const noexcept
{
    return residual_;
}

AffineReduction3D::AffineReduction3D(
    Eigen::VectorXd particular_base,
    SparseMatrixCSR3D homogeneous_base,
    ConstraintSystem3D constraints,
    std::vector<AffineEliminationRecord3D> records,
    double particular_residual_linf,
    double homogeneous_residual_linf,
    double homogeneous_orthogonality_residual)
    : particular_base_(std::move(particular_base)),
      homogeneous_base_(std::move(homogeneous_base)),
      constraints_(std::move(constraints)),
      records_(std::move(records)),
      particular_residual_linf_(particular_residual_linf),
      homogeneous_residual_linf_(homogeneous_residual_linf),
      homogeneous_orthogonality_residual_(
          homogeneous_orthogonality_residual)
{
    homogeneous_base_.makeCompressed();
    constraints_.C.makeCompressed();
    validate();
}

Eigen::Index AffineReduction3D::base_size() const noexcept
{
    return particular_base_.size();
}

Eigen::Index AffineReduction3D::reduced_size() const noexcept
{
    return homogeneous_base_.cols();
}

const Eigen::VectorXd& AffineReduction3D::particular_base() const noexcept
{
    return particular_base_;
}

const SparseMatrixCSR3D& AffineReduction3D::homogeneous_base() const noexcept
{
    return homogeneous_base_;
}

const ConstraintSystem3D& AffineReduction3D::constraints() const noexcept
{
    return constraints_;
}

const std::vector<AffineEliminationRecord3D>&
AffineReduction3D::records() const noexcept
{
    return records_;
}

double AffineReduction3D::particular_residual_linf() const noexcept
{
    return particular_residual_linf_;
}

double AffineReduction3D::homogeneous_residual_linf() const noexcept
{
    return homogeneous_residual_linf_;
}

double AffineReduction3D::homogeneous_orthogonality_residual() const noexcept
{
    return homogeneous_orthogonality_residual_;
}

Eigen::VectorXd AffineReduction3D::lift_homogeneous_base(
    const Eigen::Ref<const Eigen::VectorXd>& reduced) const
{
    require_reduced_size(reduced.size());
    return homogeneous_base_ * reduced;
}

Eigen::VectorXd AffineReduction3D::lift_base(
    const Eigen::Ref<const Eigen::VectorXd>& reduced) const
{
    return particular_base_ + lift_homogeneous_base(reduced);
}

Eigen::VectorXd AffineReduction3D::particular_full(
    const SparseMatrixCSR3D& base_expansion) const
{
    require_base_expansion_columns(base_expansion.cols());
    if (!sparse_all_finite(base_expansion))
        throw std::invalid_argument("base expansion is not finite");
    return base_expansion * particular_base_;
}

Eigen::VectorXd AffineReduction3D::particular_full(
    const Eigen::Ref<const Eigen::MatrixXd>& base_expansion) const
{
    require_base_expansion_columns(base_expansion.cols());
    if (!base_expansion.allFinite())
        throw std::invalid_argument("base expansion is not finite");
    return base_expansion * particular_base_;
}

Eigen::VectorXd AffineReduction3D::lift_full(
    const Eigen::Ref<const Eigen::VectorXd>& reduced,
    const SparseMatrixCSR3D& base_expansion) const
{
    require_base_expansion_columns(base_expansion.cols());
    return base_expansion * lift_base(reduced);
}

Eigen::VectorXd AffineReduction3D::lift_full(
    const Eigen::Ref<const Eigen::VectorXd>& reduced,
    const Eigen::Ref<const Eigen::MatrixXd>& base_expansion) const
{
    require_base_expansion_columns(base_expansion.cols());
    return base_expansion * lift_base(reduced);
}

SparseMatrixCSR3D AffineReduction3D::homogeneous_full(
    const SparseMatrixCSR3D& base_expansion) const
{
    require_base_expansion_columns(base_expansion.cols());
    if (!sparse_all_finite(base_expansion))
        throw std::invalid_argument("base expansion is not finite");
    SparseMatrixCSR3D result = base_expansion * homogeneous_base_;
    result.makeCompressed();
    return result;
}

Eigen::MatrixXd AffineReduction3D::homogeneous_full(
    const Eigen::Ref<const Eigen::MatrixXd>& base_expansion) const
{
    require_base_expansion_columns(base_expansion.cols());
    if (!base_expansion.allFinite())
        throw std::invalid_argument("base expansion is not finite");
    Eigen::MatrixXd result = base_expansion * homogeneous_base_;
    return result;
}

std::vector<std::pair<Eigen::Index, double>>
AffineReduction3D::base_basis_support(Eigen::Index reduced_column,
                                      double tolerance) const
{
    return reduced_basis_support_3d(
        homogeneous_base_, reduced_column, tolerance);
}

FullCoefficientFormula3D AffineReduction3D::base_coefficient_formula(
    Eigen::Index base_row,
    double tolerance) const
{
    return full_coefficient_formula_3d(
        particular_base_, homogeneous_base_, base_row, tolerance);
}

void AffineReduction3D::require_reduced_size(Eigen::Index size) const
{
    if (size != reduced_size())
        throw std::invalid_argument("reduced coordinate size is inconsistent");
}

void AffineReduction3D::require_base_expansion_columns(
    Eigen::Index columns) const
{
    if (columns != base_size())
        throw std::invalid_argument("base expansion column count is invalid");
}

void AffineReduction3D::validate() const
{
    constraints_.validate();
    if (particular_base_.size() != constraints_.C.cols()
        || homogeneous_base_.rows() != constraints_.C.cols()) {
        throw std::invalid_argument(
            "affine reduction base dimensions are inconsistent");
    }
    if (!particular_base_.allFinite()
        || !sparse_all_finite(homogeneous_base_)
        || !std::isfinite(particular_residual_linf_)
        || !std::isfinite(homogeneous_residual_linf_)
        || !std::isfinite(homogeneous_orthogonality_residual_)
        || particular_residual_linf_ < 0.0
        || homogeneous_residual_linf_ < 0.0
        || homogeneous_orthogonality_residual_ < 0.0) {
        throw std::invalid_argument(
            "affine reduction contains invalid numerical data");
    }
    // Sequential local SVD starts from I and composes column-orthonormal
    // embeddings.  Checking the sparse Gram matrix therefore verifies both
    // the advertised representation and full column rank without ever
    // materializing the global nullspace as a dense matrix.
    const double measured_orthogonality =
        compute_homogeneous_orthogonality_residual(homogeneous_base_);
    if (measured_orthogonality > 1.0e-8) {
        throw std::invalid_argument(
            "affine homogeneous map is rank deficient or not column-orthonormal");
    }
}

AffineReduction3D affine_eliminate_local_svd_3d(
    ConstraintSystem3D constraints,
    std::vector<ConstraintBlock3D> blocks,
    AffineReductionOptions3D options)
{
    validate_options(options);
    constraints.validate();
    constraints.C.makeCompressed();
    const Eigen::Index base_size = constraints.C.cols();

    if (blocks.empty() && constraints.C.rows() != 0) {
        ConstraintBlock3D all;
        all.kind = ConstraintBlockKind3D::Edge;
        all.key = "all_constraints";
        all.row_ids.resize(
            static_cast<std::size_t>(constraints.C.rows()));
        std::iota(all.row_ids.begin(), all.row_ids.end(), Eigen::Index{0});
        blocks.push_back(std::move(all));
    }
    validate_block_rows(constraints.C, blocks);
    blocks = order_constraint_blocks_vertex_before_edge_3d(
        std::move(blocks));
    require_complete_partition(constraints.C.rows(), blocks);
    if (options.schedule
            == AffineEliminationSchedule3D::LegacyOriginalSupport
        && options.merge_shared_support_blocks) {
        blocks = merge_constraint_blocks_by_support_3d(
            constraints.C, std::move(blocks),
            options.sparse_prune_tolerance);
    }

    Eigen::VectorXd particular = Eigen::VectorXd::Zero(base_size);
    SparseMatrixCSR3D homogeneous(base_size, base_size);
    homogeneous.setIdentity();
    homogeneous.makeCompressed();
    std::vector<AffineEliminationRecord3D> records;
    records.reserve(blocks.size());
    std::vector<Eigen::Index> processed_rows;
    processed_rows.reserve(
        static_cast<std::size_t>(constraints.C.rows()));

    const auto eliminate_block = [&] (
        const ConstraintBlock3D& block, int merge_retry_count) {
        const SparseMatrixCSR3D C_block =
            extract_rows(constraints.C, block.row_ids);
        SparseMatrixCSR3D active_map = C_block * homogeneous;
        active_map.makeCompressed();

        Eigen::VectorXd rhs(
            static_cast<Eigen::Index>(block.row_ids.size()));
        for (Eigen::Index local = 0;
             local < static_cast<Eigen::Index>(block.row_ids.size()); ++local) {
            rhs[local] = constraints.d[
                block.row_ids[static_cast<std::size_t>(local)]];
        }
        rhs -= C_block * particular;

        std::vector<Eigen::Index> active_rows;
        active_rows.reserve(block.row_ids.size());
        for (Eigen::Index row = 0; row < active_map.rows(); ++row) {
            double norm = 0.0;
            for (SparseMatrixCSR3D::InnerIterator entry(active_map, row);
                 entry; ++entry) {
                norm = std::hypot(norm, entry.value());
            }
            if (norm <= options.zero_row_tolerance) {
                if (std::abs(rhs[row]) > options.rhs_tolerance) {
                    std::ostringstream message;
                    message << "block '" << block_name(block)
                            << "' has a zero active row with nonzero RHS "
                            << rhs[row];
                    throw ConstraintIncompatibility3D(
                        message.str(), block.key,
                        block.row_ids[static_cast<std::size_t>(row)],
                        std::abs(rhs[row]));
                }
            } else {
                active_rows.push_back(row);
            }
        }

        AffineEliminationRecord3D record;
        record.kind = block.kind;
        record.key = block.key;
        record.source_keys = block.source_keys;
        record.row_ids = block.row_ids;
        record.rows = static_cast<Eigen::Index>(block.row_ids.size());
        record.active_rows = static_cast<Eigen::Index>(active_rows.size());
        record.merge_retry_count = merge_retry_count;

        if (!active_rows.empty()) {
            std::vector<Eigen::Index> candidates;
            for (const Eigen::Index row : active_rows) {
                for (SparseMatrixCSR3D::InnerIterator entry(active_map, row);
                     entry; ++entry) {
                    if (entry.value() != 0.0)
                        candidates.push_back(entry.col());
                }
            }
            std::sort(candidates.begin(), candidates.end());
            candidates.erase(
                std::unique(candidates.begin(), candidates.end()),
                candidates.end());
            if (candidates.empty()) {
                throw ConstraintIncompatibility3D(
                    "active constraint block has no remaining coordinates",
                    block.key, -1, vector_linf(rhs));
            }

            std::unordered_map<Eigen::Index, Eigen::Index> local_column;
            local_column.reserve(candidates.size());
            for (Eigen::Index column = 0;
                 column < static_cast<Eigen::Index>(candidates.size());
                 ++column) {
                local_column.emplace(
                    candidates[static_cast<std::size_t>(column)], column);
            }

            Eigen::MatrixXd local_matrix = Eigen::MatrixXd::Zero(
                static_cast<Eigen::Index>(active_rows.size()),
                static_cast<Eigen::Index>(candidates.size()));
            Eigen::VectorXd local_rhs(
                static_cast<Eigen::Index>(active_rows.size()));
            for (Eigen::Index local_row = 0;
                 local_row < static_cast<Eigen::Index>(active_rows.size());
                 ++local_row) {
                const Eigen::Index source_row =
                    active_rows[static_cast<std::size_t>(local_row)];
                local_rhs[local_row] = rhs[source_row];
                for (SparseMatrixCSR3D::InnerIterator entry(
                         active_map, source_row);
                     entry; ++entry) {
                    if (entry.value() != 0.0) {
                        local_matrix(
                            local_row,
                            local_column.at(entry.col())) += entry.value();
                    }
                }
            }

            Eigen::MatrixXd scaled_matrix = local_matrix;
            Eigen::VectorXd scaled_rhs = local_rhs;
            for (Eigen::Index row = 0; row < scaled_matrix.rows(); ++row) {
                const double norm = scaled_matrix.row(row).norm();
                if (!(norm > options.zero_row_tolerance)) {
                    throw std::logic_error(
                        "active row vanished while building local SVD");
                }
                scaled_matrix.row(row) /= norm;
                scaled_rhs[row] /= norm;
            }

            Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                scaled_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
            const Eigen::VectorXd singular = svd.singularValues();
            if (singular.size() == 0 || !(singular[0] > 0.0)
                || !singular.allFinite()) {
                throw std::runtime_error(
                    "local scaled SVD did not produce a finite leading value");
            }
            const double cutoff =
                options.relative_rank_tolerance * singular[0];
            Eigen::Index rank = 0;
            while (rank < singular.size() && singular[rank] > cutoff)
                ++rank;

            Eigen::VectorXd local_particular = Eigen::VectorXd::Zero(
                static_cast<Eigen::Index>(candidates.size()));
            if (rank != 0) {
                local_particular = svd.matrixV().leftCols(rank)
                    * (singular.head(rank).cwiseInverse().asDiagonal()
                       * (svd.matrixU().leftCols(rank).transpose()
                          * scaled_rhs));
            }
            const Eigen::VectorXd local_defect =
                local_matrix * local_particular - local_rhs;
            const double residual = local_defect.norm();
            const double denominator = local_rhs.norm()
                + std::numeric_limits<double>::epsilon();
            const double consistency = residual / denominator;
            if (!local_particular.allFinite()
                || !std::isfinite(consistency)
                || consistency > options.consistency_tolerance) {
                std::ostringstream message;
                message << "block '" << block_name(block)
                        << "' is incompatible after local scaled SVD (eta="
                        << consistency << ')';
                throw ConstraintIncompatibility3D(
                    message.str(), block.key, -1, consistency);
            }

            const Eigen::Index local_nullity =
                static_cast<Eigen::Index>(candidates.size()) - rank;
            const Eigen::MatrixXd local_nullspace =
                svd.matrixV().rightCols(local_nullity);
            const double local_homogeneous_residual = dense_max_abs(
                local_matrix * local_nullspace);

            const Eigen::Index old_coordinates = homogeneous.cols();
            std::vector<unsigned char> is_candidate(
                static_cast<std::size_t>(old_coordinates), 0);
            for (const Eigen::Index column : candidates)
                is_candidate[static_cast<std::size_t>(column)] = 1;
            const Eigen::Index untouched_count = old_coordinates
                - static_cast<Eigen::Index>(candidates.size());
            const Eigen::Index new_coordinates =
                untouched_count + local_nullity;

            Eigen::VectorXd local_shift =
                Eigen::VectorXd::Zero(old_coordinates);
            for (Eigen::Index local = 0;
                 local < static_cast<Eigen::Index>(candidates.size()); ++local) {
                local_shift[candidates[static_cast<std::size_t>(local)]] =
                    local_particular[local];
            }

            std::vector<Eigen::Triplet<double>> map_triplets;
            map_triplets.reserve(
                static_cast<std::size_t>(untouched_count)
                + static_cast<std::size_t>(local_nullspace.size()));
            Eigen::Index new_column = 0;
            for (Eigen::Index old = 0; old < old_coordinates; ++old) {
                if (!is_candidate[static_cast<std::size_t>(old)])
                    map_triplets.emplace_back(old, new_column++, 1.0);
            }
            for (Eigen::Index local_row = 0;
                 local_row < static_cast<Eigen::Index>(candidates.size());
                 ++local_row) {
                for (Eigen::Index null_column = 0;
                     null_column < local_nullity; ++null_column) {
                    const double value =
                        local_nullspace(local_row, null_column);
                    if (std::abs(value) > options.sparse_prune_tolerance) {
                        map_triplets.emplace_back(
                            candidates[static_cast<std::size_t>(local_row)],
                            untouched_count + null_column, value);
                    }
                }
            }
            SparseMatrixCSR3D coordinate_map(
                old_coordinates, new_coordinates);
            coordinate_map.setFromTriplets(
                map_triplets.begin(), map_triplets.end());
            coordinate_map.makeCompressed();

            particular += homogeneous * local_shift;
            SparseMatrixCSR3D composed = homogeneous * coordinate_map;
            homogeneous = prune_sparse(
                composed, options.sparse_prune_tolerance);

            record.support_cols =
                static_cast<Eigen::Index>(candidates.size());
            record.rank = rank;
            record.local_nullity = local_nullity;
            record.residual = residual;
            record.normalized_consistency = consistency;
            record.homogeneous_residual =
                local_homogeneous_residual;
        }

        processed_rows.insert(processed_rows.end(),
                              block.row_ids.begin(), block.row_ids.end());
        std::sort(processed_rows.begin(), processed_rows.end());
        processed_rows.erase(
            std::unique(processed_rows.begin(), processed_rows.end()),
            processed_rows.end());
        const auto invariants = processed_invariants(
            constraints, particular, homogeneous, processed_rows);
        record.processed_particular_residual_linf = invariants.first;
        record.processed_homogeneous_residual_linf = invariants.second;
        record.remaining = homogeneous.cols();
        records.push_back(std::move(record));
    };

    if (options.schedule
        == AffineEliminationSchedule3D::LegacyOriginalSupport) {
        for (const ConstraintBlock3D& block : blocks)
            eliminate_block(block, 0);
    } else {
        // Each stage owns a disjoint set of physical constraint rows.  The
        // edge closure is deliberately delayed until every vertex/T-star
        // component has updated E, so a raw C support bridge can never merge
        // a vertex block with an edge-interior block.
        const auto eliminate_stage = [&] (ConstraintBlockKind3D kind) {
            std::vector<ConstraintBlock3D> pending;
            for (const ConstraintBlock3D& block : blocks) {
                if (block.kind == kind)
                    pending.push_back(block);
            }

            while (!pending.empty()) {
                SparseMatrixCSR3D current_support =
                    constraints.C * homogeneous;
                current_support.makeCompressed();
                if (options.merge_shared_support_blocks) {
                    pending = merge_constraint_blocks_by_support_3d(
                        current_support, std::move(pending),
                        options.sparse_prune_tolerance);
                }

                ConstraintBlock3D current = std::move(pending.front());
                pending.erase(pending.begin());
                int retry_count = 0;
                for (;;) {
                    try {
                        eliminate_block(current, retry_count);
                        break;
                    } catch (const ConstraintIncompatibility3D& error) {
                        const auto throw_exhausted = [&] (
                            const std::string& reason) -> void {
                            std::ostringstream message;
                            message << error.what()
                                    << "; staged incompatibility fallback "
                                       "exhausted: "
                                    << reason
                                    << ". Vertex/star support enlargement, "
                                       "common g_N jet projection, trace knot "
                                       "insertion, and prioritized constraint "
                                       "relaxation are upstream operations and "
                                       "were not attempted; weighted least "
                                       "squares is disabled";
                            throw ConstraintIncompatibility3D(
                                message.str(), error.block_key(),
                                error.row_id(), error.residual());
                        };
                        if (!options.merge_shared_support_blocks
                            || pending.empty()) {
                            throw_exhausted(
                                !options.merge_shared_support_blocks
                                    ? "same-stage support merging is disabled"
                                    : "no unprocessed same-stage component "
                                      "remains to merge");
                        }

                        // A first closure may intentionally ignore entries
                        // below sparse_prune_tolerance.  On incompatibility,
                        // retry after a deterministic zero-tolerance closure
                        // of the *current* C*E and all still-pending blocks in
                        // this stage.  No LS result is accepted silently: if
                        // the component cannot grow, the original exception
                        // is propagated.
                        current_support = constraints.C * homogeneous;
                        current_support.makeCompressed();
                        std::vector<ConstraintBlock3D> retry_blocks;
                        retry_blocks.reserve(pending.size() + 1);
                        retry_blocks.push_back(current);
                        retry_blocks.insert(retry_blocks.end(),
                                            pending.begin(), pending.end());
                        std::vector<ConstraintBlock3D> strict_components =
                            merge_constraint_blocks_by_support_3d(
                                current_support, std::move(retry_blocks), 0.0);
                        const Eigen::Index anchor_row = current.row_ids.front();
                        auto expanded = std::find_if(
                            strict_components.begin(), strict_components.end(),
                            [anchor_row] (const ConstraintBlock3D& candidate) {
                                return std::binary_search(
                                    candidate.row_ids.begin(),
                                    candidate.row_ids.end(), anchor_row);
                            });
                        if (expanded == strict_components.end()
                            || expanded->row_ids.size()
                                <= current.row_ids.size()) {
                            throw_exhausted(
                                "zero-tolerance current-C*E support closure "
                                "found no adjacent component");
                        }
                        current = std::move(*expanded);
                        strict_components.erase(expanded);
                        pending = std::move(strict_components);
                        ++retry_count;
                    }
                }
            }
        };

        eliminate_stage(ConstraintBlockKind3D::Vertex);
        eliminate_stage(ConstraintBlockKind3D::Edge);
    }

    const Eigen::VectorXd particular_defect =
        constraints.C * particular - constraints.d;
    const double particular_residual = vector_linf(particular_defect);
    const SparseMatrixCSR3D homogeneous_defect =
        constraints.C * homogeneous;
    const double homogeneous_residual =
        sparse_max_abs(homogeneous_defect);
    const double orthogonality_residual =
        compute_homogeneous_orthogonality_residual(homogeneous);
    const double scale = std::max(
        {1.0, vector_linf(constraints.d), sparse_max_abs(constraints.C)});
    if (particular_residual > options.invariant_tolerance * scale
        || homogeneous_residual > options.invariant_tolerance * scale
        || orthogonality_residual > options.invariant_tolerance) {
        std::ostringstream message;
        message << "affine reduction invariant failed: ||Cp-d||_inf="
                << particular_residual << ", ||CE||_max="
                << homogeneous_residual << ", ||E^T E-I||_max="
                << orthogonality_residual;
        throw std::runtime_error(message.str());
    }

    return AffineReduction3D(
        std::move(particular), std::move(homogeneous),
        std::move(constraints), std::move(records), particular_residual,
        homogeneous_residual, orthogonality_residual);
}

std::vector<std::pair<Eigen::Index, double>> reduced_basis_support_3d(
    const SparseMatrixCSR3D& basis,
    Eigen::Index reduced_column,
    double tolerance)
{
    validate_nonnegative_tolerance(tolerance, "basis support tolerance");
    if (reduced_column < 0 || reduced_column >= basis.cols())
        throw std::out_of_range("reduced basis column is out of range");
    if (!sparse_all_finite(basis))
        throw std::invalid_argument("reduced basis is not finite");

    std::vector<std::pair<Eigen::Index, double>> result;
    for (Eigen::Index row = 0; row < basis.outerSize(); ++row) {
        for (SparseMatrixCSR3D::InnerIterator entry(basis, row); entry;
             ++entry) {
            if (entry.col() == reduced_column
                && std::abs(entry.value()) > tolerance) {
                result.emplace_back(row, entry.value());
            }
        }
    }
    return result;
}

FullCoefficientFormula3D full_coefficient_formula_3d(
    const Eigen::Ref<const Eigen::VectorXd>& particular,
    const SparseMatrixCSR3D& basis,
    Eigen::Index coefficient_row,
    double tolerance)
{
    validate_nonnegative_tolerance(tolerance, "coefficient formula tolerance");
    if (basis.rows() != particular.size()) {
        throw std::invalid_argument(
            "coefficient formula dimensions are inconsistent");
    }
    if (coefficient_row < 0 || coefficient_row >= basis.rows())
        throw std::out_of_range("coefficient formula row is out of range");
    if (!particular.allFinite() || !sparse_all_finite(basis))
        throw std::invalid_argument("coefficient formula is not finite");

    FullCoefficientFormula3D result;
    result.particular = particular[coefficient_row];
    for (SparseMatrixCSR3D::InnerIterator entry(basis, coefficient_row);
         entry; ++entry) {
        if (std::abs(entry.value()) > tolerance)
            result.reduced_terms.emplace_back(entry.col(), entry.value());
    }
    return result;
}

} // namespace kfbim::app3d
