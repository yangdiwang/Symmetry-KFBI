#pragma once

#include <algorithm>
#include <vector>
#include <Eigen/Dense>
#include "../local_cauchy/jump_data.hpp"
#include "../local_cauchy/local_poly.hpp"
#include "../local_cauchy/laplace_corner_patch_solver_2d.hpp"
#include "../geometry/edge_patch_3d.hpp"
#include "../geometry/grid_pair_2d.hpp"
#include "../geometry/grid_pair_3d.hpp"
#include "laplace_correction_support.hpp"
#include "laplace_sd_corner_lifting_2d.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Spread (Layer 1): interface jump data → bulk RHS correction
//
// For each interface quadrature point, apply() builds the correction data
// needed to accumulate bulk RHS corrections near the interface.
//
// Spreads return a result object because projection-point correction needs
// surface jump data during restrict.
//
// rhs_correction is *accumulated into*, not zeroed — the caller zeros it first
// when starting a fresh iteration, or accumulates across multiple interfaces.
// ---------------------------------------------------------------------------

enum class LaplaceCorrectionMethod {
    NearestExpansionCenter,
    ProjectionPoint
};

using LaplaceCorrectionMethod2D = LaplaceCorrectionMethod;
using LaplaceCorrectionMethod3D = LaplaceCorrectionMethod;

struct IndexedLocalPolys2D {
    std::vector<int> center_indices;
    std::vector<LocalPoly2D> polys;
};

struct IndexedLocalPolys3D {
    std::vector<int> center_indices;
    std::vector<LocalPoly3D> polys;
};

// Snapshot consumed by 3D restrict after an edge-singular spread.  Keeping
// values in the context avoids retaining pointers to a temporary fitted edge
// field after apply() returns.
struct EdgePatchRestrictData3D {
    geometry3d::EdgePatch3D patch;
    int active_domain_label = 1;
    // Truncated physical field on every bulk node, used after restrict to
    // recover u from the solved regular variable.
    Eigen::VectorXd physical_grid_values;
    // Analytic continuation on grid nodes used by restrict branch conversion.
    Eigen::VectorXd continuation_grid_values;
    // Physical singular trace and gradient at interface iteration points.
    Eigen::VectorXd interface_values;
    Eigen::MatrixX3d interface_gradients;
};

inline const LocalPoly2D* find_indexed_local_poly_2d(
    const IndexedLocalPolys2D& indexed,
    int center_idx)
{
    const auto it = std::lower_bound(indexed.center_indices.begin(),
                                     indexed.center_indices.end(),
                                     center_idx);
    if (it == indexed.center_indices.end() || *it != center_idx)
        return nullptr;
    const std::size_t offset =
        static_cast<std::size_t>(it - indexed.center_indices.begin());
    if (offset >= indexed.polys.size())
        return nullptr;
    const LocalPoly2D& poly = indexed.polys[offset];
    return poly.coeffs.size() > 0 ? &poly : nullptr;
}

inline const LocalPoly3D* find_indexed_local_poly_3d(
    const IndexedLocalPolys3D& indexed,
    int center_idx)
{
    const auto it = std::lower_bound(indexed.center_indices.begin(),
                                     indexed.center_indices.end(),
                                     center_idx);
    if (it == indexed.center_indices.end() || *it != center_idx)
        return nullptr;
    const std::size_t offset =
        static_cast<std::size_t>(it - indexed.center_indices.begin());
    if (offset >= indexed.polys.size())
        return nullptr;
    const LocalPoly3D& poly = indexed.polys[offset];
    return poly.coeffs.size() > 0 ? &poly : nullptr;
}

struct LaplaceCorrectionContext2D {
    LaplaceCorrectionMethod correction_method =
        LaplaceCorrectionMethod::NearestExpansionCenter;

    // Existing center-polynomial correction data. In projection-point mode this
    // may be empty because C(x) is evaluated directly from projected P2 data.
    std::vector<LocalPoly2D> correction_polys;

    // Surface data needed by the projection-point evaluator. Vectors are sized
    // to Interface2D::num_points() when correction_method == ProjectionPoint.
    Eigen::VectorXd u_jump;
    Eigen::VectorXd un_jump;
    Eigen::VectorXd rhs_jump;

    // Screened coefficient in -Delta C + alpha*C = [f].
    double alpha = 0.0;

    // Projection-point cache for grid nodes where C(x) is actually needed.
    NarrowBandProjection2D projection_cache;

    // Corner-patch branch correction data. Empty when no active corner patches
    // are present.
    std::vector<CornerPatchCorrectionData2D> corner_patch_corrections;

    // Optional displacement-jump corner lifting based on local
    // S_D = Im(z log z) bases. When present, grid nodes inside active corner
    // patches store u - W, and the physical solution is recovered by adding W
    // back inside those patches.
    SdCornerLiftingOptions2D sd_corner_lifting;

    // Nearest-center correction polynomials with the corner singular branch
    // removed from the interface jump, indexed by patch id. Each patch stores
    // only the selected global expansion-center indices used by spread/restrict.
    std::vector<IndexedLocalPolys2D> patch_regularized_correction_polys;

    PatchInterfaceJumpMode2D restrict_jump_mode =
        PatchInterfaceJumpMode2D::SingularRemovedJump;
};

using LaplaceSpreadResult2D = LaplaceCorrectionContext2D;

class ILaplaceSpread2D {
public:
    virtual ~ILaplaceSpread2D() = default;

    // jumps[i]         = jump data at interface quadrature point i
    // rhs_correction   = correction accumulated into, length grid_pair().grid().num_dofs()
    // returns          = correction context for the paired restrict operator
    virtual LaplaceSpreadResult2D apply(
        const std::vector<LaplaceJumpData2D>& jumps,
        Eigen::VectorXd&                      rhs_correction) const = 0;

    virtual const GridPair2D& grid_pair() const = 0;
};

struct LaplaceCorrectionContext3D {
    LaplaceCorrectionMethod correction_method =
        LaplaceCorrectionMethod::NearestExpansionCenter;

    // Existing center-polynomial correction data. In projection-point mode this
    // may be empty because C(x) is evaluated directly from projected P2 data.
    std::vector<LocalPoly3D> correction_polys;

    // Surface data needed by the projection-point evaluator. Vectors are sized
    // to Interface3D::num_points() when correction_method == ProjectionPoint.
    Eigen::VectorXd u_jump;
    Eigen::VectorXd un_jump;
    Eigen::VectorXd rhs_jump;

    // Screened coefficient in -Delta C + alpha*C = [f].
    double alpha = 0.0;

    // Projection-point cache for grid nodes where C(x) is actually needed.
    NarrowBandProjection3D projection_cache;

    // Selected sub-triangle-center polynomials formed after subtracting the
    // edge singular value, normal trace, and operator trace.  Spread uses
    // these only for physical crossings whose actual segment hit is inside an
    // artificial edge cylinder.  Restrict uses the same polynomials for
    // interface points inside one of edge_regularization_patches.
    IndexedLocalPolys3D edge_regularized_correction_polys;
    std::vector<geometry3d::EdgePatch3D> edge_regularization_patches;
    std::vector<EdgePatchRestrictData3D> edge_restrict_data;
};

// Temporary compatibility name for the 3D spread -> restrict contract.
using LaplaceSpreadResult3D = LaplaceCorrectionContext3D;

class ILaplaceSpread3D {
public:
    virtual ~ILaplaceSpread3D() = default;

    // The 3D result carries either nearest-center correction polynomials or
    // projection-point surface data for the paired restrict operator.
    virtual LaplaceSpreadResult3D apply(
        const std::vector<LaplaceJumpData3D>& jumps,
        Eigen::VectorXd&                      rhs_correction) const = 0;

    virtual const GridPair3D& grid_pair() const = 0;
};

// ---------------------------------------------------------------------------
// Stokes Spread: correction accumulated separately into each velocity and
// pressure sub-grid RHS (ux, uy, p in 2D; ux, uy, uz, p in 3D).
// Each rhs_* is sized to the DOF count of the corresponding MACGrid sub-grid.
// ---------------------------------------------------------------------------

class IStokesSpread2D {
public:
    virtual ~IStokesSpread2D() = default;

    virtual std::vector<StokesLocalPoly2D> apply(
        const std::vector<StokesJumpData2D>& jumps,
        Eigen::VectorXd&                     rhs_ux,
        Eigen::VectorXd&                     rhs_uy,
        Eigen::VectorXd&                     rhs_p) const = 0;

    virtual const GridPair2D& grid_pair() const = 0;
};

class IStokesSpread3D {
public:
    virtual ~IStokesSpread3D() = default;

    virtual std::vector<StokesLocalPoly3D> apply(
        const std::vector<StokesJumpData3D>& jumps,
        Eigen::VectorXd&                     rhs_ux,
        Eigen::VectorXd&                     rhs_uy,
        Eigen::VectorXd&                     rhs_uz,
        Eigen::VectorXd&                     rhs_p) const = 0;

    virtual const GridPair3D& grid_pair() const = 0;
};

} // namespace kfbim
