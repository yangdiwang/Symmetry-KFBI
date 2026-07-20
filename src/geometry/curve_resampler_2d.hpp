#pragma once

#include "curve_2d.hpp"
#include "../interface/interface_2d.hpp"
#include <vector>

namespace kfbim {

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
    static Interface2D discretize_quadratic_lagrange(const ICurve2D& curve,
                                                     double h,
                                                     double target_L_h_ratio = 2.0);

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
        double total_length;
        
        double get_t(double s) const;
    };

    static ArcLengthMap build_arc_length_map(const ICurve2D& curve, int num_samples = 10000);
};

} // namespace kfbim
