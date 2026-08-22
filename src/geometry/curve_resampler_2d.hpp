#pragma once

#include "curve_2d.hpp"
#include "../interface/interface_2d.hpp"
#include <memory>
#include <vector>

namespace kfbim {

enum class CurvaturePanelMonitor2D {
    // Add only the density required by the more restrictive local scale.
    Maximum,

    // Add arc-length and tangent-turn densities.  This is more conservative
    // and continues to add curvature resolution even when the h scale alone
    // already satisfies the requested turn limit.
    Additive
};

// ---------------------------------------------------------------------------
// CurveResampler2D
//
// Resamples a parametric curve into a set of quasi-uniform panels based on arc
// length. The default discretization is the quadratic Lagrange P2 panel layout
// used by the current 2D KFBIM correction path; the Gauss-point layout remains
// available as an explicit legacy option.
// ---------------------------------------------------------------------------
class CurveResampler2D {
public:
    // Discretize the curve into quadratic Lagrange P2 Interface2D panels.
    // target_L_h_ratio determines the target panel arc length L relative to h.
    static Interface2D discretize(const ICurve2D& curve, double h, double target_L_h_ratio = 2.0);
    static Interface2D discretize(std::shared_ptr<const ICurve2D> curve,
                                  double h,
                                  double target_L_h_ratio = 2.0);
    static Interface2D discretize_quadratic_lagrange(const ICurve2D& curve,
                                                     double h,
                                                     double target_L_h_ratio = 2.0);
    // Owning overload: retains the source curve as the smooth geometry
    // provider on the returned Interface2D.  The P2 nodes still carry the
    // discrete jump degrees of freedom.
    static Interface2D discretize_quadratic_lagrange(
        std::shared_ptr<const ICurve2D> curve,
        double h,
        double target_L_h_ratio = 2.0);

    // Curvature-aware P2 discretization.  Panels are equidistributed in a
    // dimensionless monitor built from
    //
    //   d mu_h     = ds / (target_L_h_ratio * h),
    //   d mu_curve = |d tangent_angle| / max_panel_turn.
    //
    // Maximum uses max(d mu_h, d mu_curve), the minimal density associated
    // with the two independent constraints.  Additive uses their sum, so a
    // locally smooth panel has the more conservative approximate length
    //
    //   L = 1 / (1 / (target_L_h_ratio * h)
    //            + |curvature| / max_panel_turn),
    //
    // which remains O(h) while adding resolution in high-curvature regions.
    static Interface2D discretize_quadratic_lagrange_curvature_adaptive(
        const ICurve2D& curve,
        double h,
        double target_L_h_ratio,
        double max_panel_turn,
        CurvaturePanelMonitor2D monitor =
            CurvaturePanelMonitor2D::Maximum);
    static Interface2D discretize_quadratic_lagrange_curvature_adaptive(
        std::shared_ptr<const ICurve2D> curve,
        double h,
        double target_L_h_ratio,
        double max_panel_turn,
        CurvaturePanelMonitor2D monitor =
            CurvaturePanelMonitor2D::Maximum);

    // Backward-compatible wrapper for the former active 2D name.
    static Interface2D discretize_chebyshev_lobatto(const ICurve2D& curve,
                                                    double h,
                                                    double target_L_h_ratio = 2.0);

    // Legacy 3-point Gauss-Legendre panel discretization.
    static Interface2D discretize_legacy_gauss(const ICurve2D& curve,
                                               double h,
                                               double target_L_h_ratio = 4.0);

private:
    struct ArcLengthMap {
        std::vector<double> t_vals;
        std::vector<double> s_vals;
        std::vector<double> absolute_turn_vals;
        double total_length;
        double total_absolute_turn;
        
        double get_t(double s) const;
    };

    static ArcLengthMap build_arc_length_map(const ICurve2D& curve, int num_samples = 10000);
    static Interface2D discretize_quadratic_lagrange_impl(
        const ICurve2D& curve,
        std::shared_ptr<const ICurve2D> curve_owner,
        double h,
        double target_L_h_ratio);
    static Interface2D
    discretize_quadratic_lagrange_curvature_adaptive_impl(
        const ICurve2D& curve,
        std::shared_ptr<const ICurve2D> curve_owner,
        double h,
        double target_L_h_ratio,
        double max_panel_turn,
        CurvaturePanelMonitor2D monitor);
    static Interface2D build_quadratic_lagrange_interface(
        const ICurve2D& curve,
        const ArcLengthMap& map,
        const std::vector<double>& panel_edges,
        std::shared_ptr<const ICurve2D> curve_owner);
};

} // namespace kfbim
