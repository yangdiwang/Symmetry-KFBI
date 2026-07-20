#pragma once

#include "i_restrict.hpp"
#include "laplace_correction_support.hpp"

namespace kfbim {

class LaplaceRestrictCorrectionEvaluator3D;

// Restrict consumes the regular bulk variable. Call this only after restrict
// when the physical bulk solution u = w + sum_e A_e is required.
void restore_edge_patch_physical_solution_3d(
    Eigen::VectorXd& u_bulk,
    const LaplaceSpreadResult3D& spread_result);

// Restrict companion for LaplaceQuadraticPatchCenterSpread3D. Each interface
// point anchors the fixed ten-point quadratic stencil to the closest bulk
// node's expansion-center type, so all samples in that fit use one jump source.
// stencil_radius is retained for API compatibility and nearest-center cache
// construction; it no longer changes the interpolation stencil size.
class LaplaceQuadraticPatchCenterRestrict3D final : public ILaplaceRestrict3D {
public:
    explicit LaplaceQuadraticPatchCenterRestrict3D(const GridPair3D& grid_pair,
                                                   int               stencil_radius = 2);

    std::vector<LocalPoly3D> apply(
        const Eigen::VectorXd&       bulk_solution,
        const LaplaceSpreadResult3D& spread_result) const override;

    const GridPair3D& grid_pair() const override { return grid_pair_; }

    int owner_center(int interface_point) const;
    int owner_panel(int interface_point) const;
    double dimensionless_fit_rcond(int interface_point) const;

private:
    LocalPoly3D fit_at_interface_point(
        const Eigen::VectorXd&                       bulk_solution,
        int                                          q,
        const LaplaceRestrictCorrectionEvaluator3D&  correction_evaluator) const;

    const GridPair3D& grid_pair_;
    int               stencil_radius_;
    LaplaceCorrectionSupport3D support_;
    std::vector<int> owner_center_by_point_;
    std::vector<Eigen::Matrix<double, 10, 10>>
        fit_inverse_dimensionless_by_point_;
    std::vector<double> fit_rcond_by_point_;
};

} // namespace kfbim
