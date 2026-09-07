#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kfbim::app3d {

// Eigen's RowMajor sparse matrix is a compressed-row matrix after
// makeCompressed().  Constraint rows are the dominant access pattern in the
// topology reduction, so this is the canonical sparse type for this module.
using SparseMatrixCSR3D = Eigen::SparseMatrix<double, Eigen::RowMajor>;

struct PatchDofLayout3D {
    int patch_id = -1;
    int n_u = 0;
    int n_v = 0;
    Eigen::Index offset = 0;

    [[nodiscard]] Eigen::Index size() const;
    [[nodiscard]] Eigen::Index global_index(int i, int j) const;
    [[nodiscard]] std::pair<int, int> decode(Eigen::Index global_index) const;
    [[nodiscard]] bool contains(Eigen::Index global_index) const;
    void validate() const;
};

struct ConstraintMeta3D {
    std::string kind;       // c0, smooth_c1, feature_jet, gauge, ...
    std::string macro;
    std::string segment;
    int connection = -1;
    // Stable topology-only identifier of the physical G1 sheet.  A negative
    // value means that the row either spans multiple sheets (feature_jet) or
    // does not need sheet-local vertex grouping.  In particular, Dirichlet
    // NormalTrace smooth rows always carry a nonnegative sheet_id.
    int sheet_id = -1;
    std::array<double, 2> cell{{0.0, 0.0}};
    int mode = -1;
    std::optional<std::array<double, 3>> star;
    std::string side = "both";
};

struct ConstraintSystem3D {
    SparseMatrixCSR3D C;
    Eigen::VectorXd d;
    std::vector<ConstraintMeta3D> meta;

    void validate() const;
    [[nodiscard]] Eigen::Index row_count() const noexcept { return C.rows(); }
    [[nodiscard]] Eigen::Index coordinate_count() const noexcept
    {
        return C.cols();
    }
};

// Sparse, row-at-a-time assembly.  Duplicate columns in one row are summed
// without ever materializing a dense global row.  Satisfied zero rows are
// omitted; a zero row with nonzero RHS is an explicit incompatibility.
class SparseConstraintBuilder3D {
public:
    explicit SparseConstraintBuilder3D(Eigen::Index coordinate_count);

    void append_row(const std::vector<Eigen::Index>& column_ids,
                    const std::vector<double>& values,
                    double rhs,
                    ConstraintMeta3D meta = {});
    void append_row(
        const std::vector<std::pair<Eigen::Index, double>>& entries,
        double rhs,
        ConstraintMeta3D meta = {});

    [[nodiscard]] Eigen::Index coordinate_count() const noexcept;
    [[nodiscard]] Eigen::Index pending_row_count() const noexcept;
    [[nodiscard]] ConstraintSystem3D finalize(
        double zero_row_tolerance = 1.0e-14,
        double rhs_tolerance = 1.0e-12) const;

private:
    struct PendingRow {
        std::vector<std::pair<Eigen::Index, double>> entries;
        double rhs = 0.0;
        ConstraintMeta3D meta;
    };

    Eigen::Index coordinate_count_ = 0;
    std::vector<PendingRow> rows_;
};

enum class ConstraintBlockKind3D {
    Vertex,
    Edge
};

struct ConstraintBlock3D {
    ConstraintBlockKind3D kind = ConstraintBlockKind3D::Edge;
    std::string key;
    std::vector<Eigen::Index> row_ids;
    // A merged support component retains the original physical block keys.
    std::vector<std::string> source_keys;
};

// Stable vertex/T-star-before-edge ordering from the implementation spec.
[[nodiscard]] std::vector<ConstraintBlock3D>
order_constraint_blocks_vertex_before_edge_3d(
    std::vector<ConstraintBlock3D> blocks);

// Union-find closure of blocks that touch at least one common sparse support
// column.  support_matrix may be C itself or the current C*E map; its rows use
// the same row ids as the blocks.
[[nodiscard]] std::vector<ConstraintBlock3D>
merge_constraint_blocks_by_support_3d(
    const SparseMatrixCSR3D& support_matrix,
    std::vector<ConstraintBlock3D> blocks,
    double support_tolerance = 0.0);

class ConstraintIncompatibility3D final : public std::runtime_error {
public:
    ConstraintIncompatibility3D(std::string message,
                                std::string block_key = {},
                                Eigen::Index row_id = -1,
                                double residual = 0.0);

    [[nodiscard]] const std::string& block_key() const noexcept;
    [[nodiscard]] Eigen::Index row_id() const noexcept;
    [[nodiscard]] double residual() const noexcept;

private:
    std::string block_key_;
    Eigen::Index row_id_ = -1;
    double residual_ = 0.0;
};

struct AffineEliminationRecord3D {
    ConstraintBlockKind3D kind = ConstraintBlockKind3D::Edge;
    std::string key;
    std::vector<std::string> source_keys;
    std::vector<Eigen::Index> row_ids;
    Eigen::Index rows = 0;
    Eigen::Index active_rows = 0;
    Eigen::Index support_cols = 0;
    Eigen::Index rank = 0;
    Eigen::Index local_nullity = 0;
    Eigen::Index remaining = 0;
    double residual = 0.0;
    double normalized_consistency = 0.0;
    double homogeneous_residual = 0.0;
    double processed_particular_residual_linf = 0.0;
    double processed_homogeneous_residual_linf = 0.0;
    // Number of deterministic same-stage support-component expansions made
    // after a local incompatibility.  A nonzero value means that the failed
    // block was retried together with blocks coupled through the current C*E.
    int merge_retry_count = 0;
};

enum class AffineEliminationSchedule3D {
    // Backward-compatible behavior: order vertex blocks first, then form one
    // support closure from the original C before any elimination.
    LegacyOriginalSupport,
    // Topology route: eliminate vertex/T-star components first, recompute
    // C*E, then form and eliminate edge-interior components.  Components are
    // never merged across the stage boundary.
    StagedVertexThenEdge
};

struct AffineReductionOptions3D {
    double zero_row_tolerance = 1.0e-13;
    double rhs_tolerance = 1.0e-11;
    double relative_rank_tolerance = 1.0e-11;
    double consistency_tolerance = 2.0e-10;
    double sparse_prune_tolerance = 1.0e-14;
    double invariant_tolerance = 2.0e-9;
    bool merge_shared_support_blocks = true;
    AffineEliminationSchedule3D schedule =
        AffineEliminationSchedule3D::LegacyOriginalSupport;
};

struct FullCoefficientFormula3D {
    double particular = 0.0;
    std::vector<std::pair<Eigen::Index, double>> reduced_terms;
};

class AffineReduction3D {
public:
    AffineReduction3D() = default;
    AffineReduction3D(Eigen::VectorXd particular_base,
                      SparseMatrixCSR3D homogeneous_base,
                      ConstraintSystem3D constraints,
                      std::vector<AffineEliminationRecord3D> records,
                      double particular_residual_linf,
                      double homogeneous_residual_linf,
                      double homogeneous_orthogonality_residual);

    [[nodiscard]] Eigen::Index base_size() const noexcept;
    [[nodiscard]] Eigen::Index reduced_size() const noexcept;
    [[nodiscard]] const Eigen::VectorXd& particular_base() const noexcept;
    [[nodiscard]] const SparseMatrixCSR3D& homogeneous_base() const noexcept;
    [[nodiscard]] const ConstraintSystem3D& constraints() const noexcept;
    [[nodiscard]] const std::vector<AffineEliminationRecord3D>& records()
        const noexcept;

    [[nodiscard]] double particular_residual_linf() const noexcept;
    [[nodiscard]] double homogeneous_residual_linf() const noexcept;
    [[nodiscard]] double homogeneous_orthogonality_residual() const noexcept;

    [[nodiscard]] Eigen::VectorXd lift_homogeneous_base(
        const Eigen::Ref<const Eigen::VectorXd>& reduced) const;
    [[nodiscard]] Eigen::VectorXd lift_base(
        const Eigen::Ref<const Eigen::VectorXd>& reduced) const;

    [[nodiscard]] Eigen::VectorXd particular_full(
        const SparseMatrixCSR3D& base_expansion) const;
    [[nodiscard]] Eigen::VectorXd particular_full(
        const Eigen::Ref<const Eigen::MatrixXd>& base_expansion) const;
    [[nodiscard]] Eigen::VectorXd lift_full(
        const Eigen::Ref<const Eigen::VectorXd>& reduced,
        const SparseMatrixCSR3D& base_expansion) const;
    [[nodiscard]] Eigen::VectorXd lift_full(
        const Eigen::Ref<const Eigen::VectorXd>& reduced,
        const Eigen::Ref<const Eigen::MatrixXd>& base_expansion) const;

    [[nodiscard]] SparseMatrixCSR3D homogeneous_full(
        const SparseMatrixCSR3D& base_expansion) const;
    [[nodiscard]] Eigen::MatrixXd homogeneous_full(
        const Eigen::Ref<const Eigen::MatrixXd>& base_expansion) const;

    [[nodiscard]] std::vector<std::pair<Eigen::Index, double>>
    base_basis_support(Eigen::Index reduced_column,
                       double tolerance = 1.0e-13) const;
    [[nodiscard]] FullCoefficientFormula3D base_coefficient_formula(
        Eigen::Index base_row,
        double tolerance = 1.0e-13) const;

private:
    void require_reduced_size(Eigen::Index size) const;
    void require_base_expansion_columns(Eigen::Index columns) const;
    void validate() const;

    Eigen::VectorXd particular_base_;
    SparseMatrixCSR3D homogeneous_base_;
    ConstraintSystem3D constraints_;
    std::vector<AffineEliminationRecord3D> records_;
    double particular_residual_linf_ = 0.0;
    double homogeneous_residual_linf_ = 0.0;
    double homogeneous_orthogonality_residual_ = 0.0;
};

[[nodiscard]] AffineReduction3D affine_eliminate_local_svd_3d(
    ConstraintSystem3D constraints,
    std::vector<ConstraintBlock3D> blocks,
    AffineReductionOptions3D options = {});

[[nodiscard]] std::vector<std::pair<Eigen::Index, double>>
reduced_basis_support_3d(const SparseMatrixCSR3D& basis,
                         Eigen::Index reduced_column,
                         double tolerance = 1.0e-13);

[[nodiscard]] FullCoefficientFormula3D full_coefficient_formula_3d(
    const Eigen::Ref<const Eigen::VectorXd>& particular,
    const SparseMatrixCSR3D& basis,
    Eigen::Index coefficient_row,
    double tolerance = 1.0e-13);

} // namespace kfbim::app3d
