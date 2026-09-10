#pragma once

#include "src/support/trace/restrict_resource_plan_3d.hpp"

#include <Eigen/SparseCore>

#include <array>
#include <cstddef>
#include <vector>

namespace kfbim::app3d {

using ResourceRestrictSparseMatrix3D = Eigen::SparseMatrix<double, Eigen::RowMajor>;

// Fixed affine evaluation J(c)=sum(values[k]*c[columns[k]])+known.
// Columns must be unique and strictly sorted, in the supplied coefficient
// coordinate system. No density fit or data callback is performed here.
struct ResourceAffineRow3D {
    std::vector<int> columns;
    std::vector<double> values;
    double known = 0.0;
    // Required only when the associated guarded plan request asks for fallback.
    // The caller must independently construct and validate that fallback row.
    bool fallback_resolved = false;
};

using ResourceTraceSampleJumpRows3D = std::array<ResourceAffineRow3D, 6>;

struct ResourceRestrictBranch3D {
    ResourceRestrictSparseMatrix3D Rc_value;
    ResourceRestrictSparseMatrix3D Rc_normal;
    Eigen::VectorXd known_value;
    Eigen::VectorXd known_normal;
};

struct ResourceRestrictAssemblyStats3D {
    std::size_t trace_count = 0;
    std::size_t support_visits = 0;
    std::size_t wrong_side_visits = 0;
    std::size_t request_count = 0;
    std::size_t grid_value_nnz = 0;
    std::size_t grid_normal_nnz = 0;
    std::size_t interior_value_nnz = 0;
    std::size_t interior_normal_nnz = 0;
    std::size_t exterior_value_nnz = 0;
    std::size_t exterior_normal_nnz = 0;
};

// t_branch = Rg * U_full + Rc_branch * c + known_branch.
// In particular Rc*G*z must be retained when c=p+G*z is used by GMRES.
// Normal matrices already contain the physical 1/h recovery scaling.
struct ResourceRestrictAssembly3D {
    ResourceRestrictSparseMatrix3D Rg_value;
    ResourceRestrictSparseMatrix3D Rg_normal;
    ResourceRestrictBranch3D interior;
    ResourceRestrictBranch3D exterior;
    ResourceRestrictAssemblyStats3D stats;
};

// Visits must follow trace-center, side (interior then exterior), grid-slot
// order, exactly as the supplied stencils. U is indexed by full Cartesian ID.
// request_rows is indexed by plan.requests; trace_sample_jump_rows evaluates
// the q-centred P2 jump at (-1.5,-.75,-.5,+.5,+.75,+1.5)*h*n.
//
// First each side's support values are continued using
//     desired_inside - actual_inside.
// Then exterior recovery subtracts q-centred J at negative samples, whereas
// interior recovery adds it at positive samples. Both use the same six-point
// cubic pseudoinverse supplied through each side's recovery rows.
[[nodiscard]] ResourceRestrictAssembly3D assemble_resource_restrict_3d(
    const RestrictResourcePlan3D& plan,
    const std::vector<SharedSideCoverRestrictStencil3D>& stencils,
    const std::vector<RestrictSupportVisit3D>& visits,
    const std::vector<ResourceAffineRow3D>& request_rows,
    const std::vector<ResourceTraceSampleJumpRows3D>& trace_sample_jump_rows,
    int grid_full_dof_count,
    int coefficient_count);

// Compares numerical coefficients, including entries stored by only one side.
// Different matrix dimensions throw; explicit stored zeros do not matter.
[[nodiscard]] double resource_sparse_max_difference_3d(
    const ResourceRestrictSparseMatrix3D& a,
    const ResourceRestrictSparseMatrix3D& b);

} // namespace kfbim::app3d
