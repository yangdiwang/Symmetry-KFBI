#include "src/support/trace/resource_restrict_assembly_3d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace kfbim::app3d {
namespace {

using Row = std::map<int, double>;

void validate_affine_row(const ResourceAffineRow3D& row, int columns)
{
    if (row.columns.size() != row.values.size() || !std::isfinite(row.known))
        throw std::invalid_argument("resource affine row has invalid dimensions or known term");
    int previous = -1;
    for (std::size_t k = 0; k < row.columns.size(); ++k) {
        if (row.columns[k] <= previous || row.columns[k] >= columns
            || !std::isfinite(row.values[k]))
            throw std::invalid_argument("resource affine columns must be unique, sorted and in range");
        previous = row.columns[k];
    }
}

void accumulate(Row& target, double& known, const ResourceAffineRow3D& source,
                double weight)
{
    // Preserve even exact zero sums. In particular no near-zero pruning policy
    // can make reference and resource-reuse operators depend on request counts.
    for (std::size_t k = 0; k < source.columns.size(); ++k)
        target[source.columns[k]] += weight * source.values[k];
    known += weight * source.known;
}

void initialize(ResourceRestrictSparseMatrix3D& matrix, int rows, int columns)
{
    matrix.resize(rows, columns);
    matrix.reserve(static_cast<Eigen::Index>(rows) * std::min(columns, 32));
}

void append_row(ResourceRestrictSparseMatrix3D& matrix, int row, const Row& values)
{
    matrix.startVec(row);
    for (const auto& entry : values) {
        if (!std::isfinite(entry.second))
            throw std::runtime_error("resource restrict assembly produced non-finite coefficients");
        matrix.insertBack(row, entry.first) = entry.second;
    }
}

void finish(ResourceRestrictSparseMatrix3D& matrix)
{
    matrix.finalize();
    matrix.makeCompressed();
}

} // namespace

ResourceRestrictAssembly3D assemble_resource_restrict_3d(
    const RestrictResourcePlan3D& plan,
    const std::vector<SharedSideCoverRestrictStencil3D>& stencils,
    const std::vector<RestrictSupportVisit3D>& visits,
    const std::vector<ResourceAffineRow3D>& request_rows,
    const std::vector<ResourceTraceSampleJumpRows3D>& trace_sample_jump_rows,
    int grid_full_dof_count,
    int coefficient_count)
{
    if (grid_full_dof_count <= 0 || coefficient_count < 0
        || stencils.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || stencils.size() != trace_sample_jump_rows.size()
        || visits.size() != plan.visit_to_request.size()
        || request_rows.size() != plan.requests.size())
        throw std::invalid_argument("resource restrict assembly input dimensions do not agree");
    for (std::size_t k = 0; k < request_rows.size(); ++k) {
        validate_affine_row(request_rows[k], coefficient_count);
        const auto& request = plan.requests[k];
        if (request.grid_full_id < 0 || request.grid_full_id >= grid_full_dof_count)
            throw std::invalid_argument("resource request full Cartesian ID is out of range");
        if (request.fallback_required && !request_rows[k].fallback_resolved)
            throw std::invalid_argument("guarded resource request has no independently resolved fallback row");
    }
    for (const auto& rows : trace_sample_jump_rows)
        for (const auto& row : rows) validate_affine_row(row, coefficient_count);

    const int trace_count = static_cast<int>(stencils.size());
    ResourceRestrictAssembly3D result;
    initialize(result.Rg_value, trace_count, grid_full_dof_count);
    initialize(result.Rg_normal, trace_count, grid_full_dof_count);
    for (auto* branch : {&result.interior, &result.exterior}) {
        initialize(branch->Rc_value, trace_count, coefficient_count);
        initialize(branch->Rc_normal, trace_count, coefficient_count);
        branch->known_value = Eigen::VectorXd::Zero(trace_count);
        branch->known_normal = Eigen::VectorXd::Zero(trace_count);
    }
    std::size_t visit_id = 0;
    for (int center = 0; center < trace_count; ++center) {
        const auto& stencil = stencils[static_cast<std::size_t>(center)];
        if (!(stencil.h > 0.0) || !std::isfinite(stencil.h)
            || !stencil.cubic_pseudoinverse.allFinite())
            throw std::invalid_argument("resource restrict stencil has invalid scaling or recovery");
        Row grid_value, grid_normal;
        std::array<Row, 2> coefficient_value, coefficient_normal;
        std::array<double, 2> known_value{{0.0, 0.0}}, known_normal{{0.0, 0.0}};
        for (int side_id = 0; side_id < 2; ++side_id) {
            const auto& side = stencil.sides[static_cast<std::size_t>(side_id)];
            if (side.desired_inside != (side_id == 0) || side.grid_ids.empty()
                || side.sampling_weights.cols() != static_cast<Eigen::Index>(side.grid_ids.size())
                || !side.sampling_weights.allFinite() || !side.value_recovery.allFinite()
                || !side.normal_recovery.allFinite())
                throw std::invalid_argument("resource restrict side has invalid branch, shape or weights");
            for (std::size_t slot = 0; slot < side.grid_ids.size(); ++slot, ++visit_id) {
                const int node = side.grid_ids[slot];
                if (visit_id >= visits.size() || node < 0 || node >= grid_full_dof_count)
                    throw std::invalid_argument("resource restrict support count or full Cartesian ID is invalid");
                const auto& visit = visits[visit_id];
                if (visit.trace_center_id != center || visit.desired_inside != side.desired_inside
                    || visit.grid_full_id != node)
                    throw std::invalid_argument("resource restrict visits must follow center/side/grid-slot order");
                const double value_weight = side.value_recovery.dot(
                    side.sampling_weights.col(static_cast<Eigen::Index>(slot)));
                const double normal_weight = side.normal_recovery.dot(
                    side.sampling_weights.col(static_cast<Eigen::Index>(slot)));
                grid_value[node] += value_weight;
                grid_normal[node] += normal_weight;
                const int sign = static_cast<int>(side.desired_inside)
                    - static_cast<int>(visit.actual_inside);
                const int request = plan.visit_to_request[visit_id];
                if (sign == 0 && request == -1) continue;
                if (request < 0 || request >= static_cast<int>(request_rows.size())
                    || plan.requests[static_cast<std::size_t>(request)].grid_full_id != node)
                    throw std::invalid_argument("wrong-side resource visit has no matching full-ID request");
                const bool signed_extension = plan.requests[static_cast<std::size_t>(request)].signed_extension;
                if (sign == 0 && !signed_extension)
                    throw std::invalid_argument("same-side resource visit requires an explicit signed extension");
                if (sign != 0) ++result.stats.wrong_side_visits;
                const double correction_sign = signed_extension ? 1.0 : static_cast<double>(sign);
                for (int branch = 0; branch < 2; ++branch) {
                    accumulate(coefficient_value[branch], known_value[branch],
                        request_rows[static_cast<std::size_t>(request)], correction_sign * value_weight);
                    accumulate(coefficient_normal[branch], known_normal[branch],
                        request_rows[static_cast<std::size_t>(request)], correction_sign * normal_weight);
                }
            }
            // Interior target: extend the positive-side samples by +J.
            // Exterior target: extend the negative-side samples by -J.
            const int branch = side_id == 0 ? 1 : 0;
            const double sign = side_id == 0 ? -1.0 : 1.0;
            for (int layer = 0; layer < 3; ++layer) {
                const auto& jump = trace_sample_jump_rows[static_cast<std::size_t>(center)]
                    [static_cast<std::size_t>(3 * side_id + layer)];
                accumulate(coefficient_value[branch], known_value[branch], jump,
                    sign * side.value_recovery[layer]);
                accumulate(coefficient_normal[branch], known_normal[branch], jump,
                    sign * side.normal_recovery[layer]);
            }
        }
        append_row(result.Rg_value, center, grid_value);
        append_row(result.Rg_normal, center, grid_normal);
        std::array<ResourceRestrictBranch3D*, 2> branches{{&result.interior, &result.exterior}};
        for (int branch = 0; branch < 2; ++branch) {
            append_row(branches[branch]->Rc_value, center, coefficient_value[branch]);
            append_row(branches[branch]->Rc_normal, center, coefficient_normal[branch]);
            if (!std::isfinite(known_value[branch]) || !std::isfinite(known_normal[branch]))
                throw std::runtime_error("resource restrict assembly produced non-finite known terms");
            branches[branch]->known_value[center] = known_value[branch];
            branches[branch]->known_normal[center] = known_normal[branch];
        }
    }
    if (visit_id != visits.size())
        throw std::invalid_argument("resource restrict has unconsumed support visits");
    finish(result.Rg_value);
    finish(result.Rg_normal);
    finish(result.interior.Rc_value);
    finish(result.interior.Rc_normal);
    finish(result.exterior.Rc_value);
    finish(result.exterior.Rc_normal);
    result.stats.trace_count = stencils.size();
    result.stats.support_visits = visits.size();
    result.stats.request_count = request_rows.size();
    result.stats.grid_value_nnz = static_cast<std::size_t>(result.Rg_value.nonZeros());
    result.stats.grid_normal_nnz = static_cast<std::size_t>(result.Rg_normal.nonZeros());
    result.stats.interior_value_nnz = static_cast<std::size_t>(result.interior.Rc_value.nonZeros());
    result.stats.interior_normal_nnz = static_cast<std::size_t>(result.interior.Rc_normal.nonZeros());
    result.stats.exterior_value_nnz = static_cast<std::size_t>(result.exterior.Rc_value.nonZeros());
    result.stats.exterior_normal_nnz = static_cast<std::size_t>(result.exterior.Rc_normal.nonZeros());
    return result;
}

double resource_sparse_max_difference_3d(
    const ResourceRestrictSparseMatrix3D& a,
    const ResourceRestrictSparseMatrix3D& b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw std::invalid_argument("resource sparse comparison requires identical dimensions");
    double result = 0.0;
    for (int row = 0; row < a.rows(); ++row) {
        ResourceRestrictSparseMatrix3D::InnerIterator x(a, row), y(b, row);
        while (x || y) {
            double difference = 0.0;
            if (x && (!y || x.col() < y.col())) { difference = x.value(); ++x; }
            else if (y && (!x || y.col() < x.col())) { difference = -y.value(); ++y; }
            else { difference = x.value() - y.value(); ++x; ++y; }
            if (!std::isfinite(difference))
                throw std::invalid_argument("resource sparse comparison received non-finite coefficients");
            result = std::max(result, std::abs(difference));
        }
    }
    return result;
}

} // namespace kfbim::app3d
