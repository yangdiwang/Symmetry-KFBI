#pragma once

#include "i_spread.hpp"
#include "laplace_correction_support.hpp"

namespace kfbim {

class EdgePatchSingularLaplacian3D;

// Explicit edge-singular decomposition for one spread/RHS construction.
// It is passed per apply (instead of stored in the BVP operator) so a fixed
// fitted singular field cannot accidentally turn a linear GMRES apply into an
// affine operation.
struct EdgePatchSpreadCorrection3D {
    std::vector<const EdgePatchSingularLaplacian3D*> singular_fields;
    bool accumulate_volume_rhs = true;
};

// Preferred 3D Laplace spread for shared P2 triangular patch geometry.
//
// Each parent triangle is subdivided twice; the 16 small-triangle barycenters
// are used as generated correction expansion centers.
class LaplaceQuadraticPatchCenterSpread3D final : public ILaplaceSpread3D {
public:
    explicit LaplaceQuadraticPatchCenterSpread3D(
        const GridPair3D&         grid_pair,
        double                    kappa = 0.0,
        LaplaceCorrectionMethod3D correction_method =
            LaplaceCorrectionMethod3D::NearestExpansionCenter,
        int                       projection_restrict_stencil_radius = 2);

    LaplaceSpreadResult3D apply(
        const std::vector<LaplaceJumpData3D>& jumps,
        Eigen::VectorXd&                      rhs_correction) const override;

    // For physical crossings whose segment/interface hit lies inside an
    // artificial edge cylinder, rebuild the owning sub-triangle-center Cauchy
    // polynomial from singular-removed jump data.  The same call optionally
    // accumulates the cylinder volume/artificial-boundary RHS correction.
    LaplaceSpreadResult3D apply(
        const std::vector<LaplaceJumpData3D>& jumps,
        Eigen::VectorXd&                      rhs_correction,
        const EdgePatchSpreadCorrection3D&   edge_correction) const;

    const GridPair3D& grid_pair() const override { return grid_pair_; }

private:
    const GridPair3D&         grid_pair_;
    double                    kappa_;
    LaplaceCorrectionMethod3D correction_method_;
    int                       projection_restrict_stencil_radius_;
    LaplaceCorrectionSupport3D support_;
    NarrowBandProjection3D projection_cache_;
    std::vector<P2CrossingOwner3D> crossing_owners_;
};

} // namespace kfbim
