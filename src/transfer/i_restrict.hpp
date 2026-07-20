#pragma once

#include <vector>
#include <Eigen/Dense>
#include "../local_cauchy/local_poly.hpp"
#include "../geometry/grid_pair_2d.hpp"
#include "../geometry/grid_pair_3d.hpp"
#include "i_spread.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Restrict (Layer 1): bulk solution → local polynomial at each interface point
//
// The active 2D apply() path does two things at each quadrature point x_i on Γ:
//   1. Interpolates a local polynomial from a fixed square stencil around the
//      nearest bulk grid node. In 2D the 6-node stencil recovers the degree-2
//      expansion: coeffs = [u, u_x, u_y, u_xx, u_xy, u_yy] at x_i.
//      In 3D the analogous fixed 10-node stencil recovers degree-2 data.
//   2. Uses the correction polynomial from the preceding Spread::apply() to
//      map side-specific grid samples onto the interface average branch before
//      interpolation. Interior samples subtract C/2; exterior samples add C/2.
//
// The returned LocalPoly2D for point i is centered at x_i and stores the
// averaged solution expansion in the standard monomial basis (same ordering
// as jump_data.hpp):
//
//   poly.coeffs[0]                         = (u+ + u-)/2
//   poly.coeffs[1]*n_x + coeffs[2]*n_y     = (∂ₙu+ + ∂ₙu-)/2
//
// where n = Interface2D::normals().row(i).  This unified return type covers
// both Dirichlet (trace comparison) and Neumann (flux comparison) BVPs
// without separate restrict interfaces.
// ---------------------------------------------------------------------------

class ILaplaceRestrict2D {
public:
    virtual ~ILaplaceRestrict2D() = default;

    // spread_result:    context returned by the preceding Spread::apply()
    // returns:          averaged solution polynomial at each interface point;
    //                   length = Interface2D::num_points()
    virtual std::vector<LocalPoly2D> apply(
        const Eigen::VectorXd&       bulk_solution,
        const LaplaceSpreadResult2D& spread_result) const = 0;

    virtual const GridPair2D& grid_pair() const = 0;
};

class ILaplaceRestrict3D {
public:
    virtual ~ILaplaceRestrict3D() = default;

    virtual std::vector<LocalPoly3D> apply(
        const Eigen::VectorXd&       bulk_solution,
        const LaplaceSpreadResult3D& spread_result) const = 0;

    virtual const GridPair3D& grid_pair() const = 0;
};

// ---------------------------------------------------------------------------
// Stokes Restrict: returns the corrected MAC solution as a StokesLocalPoly at
// each interface point.  Callers extract velocity trace and traction from
// the polynomial coefficients:
//
//   velocity trace:  (vel_coeffs(0,0), vel_coeffs(0,1))   (= (ux, uy) at x_i)
//   traction t = σ(u,p)·n:
//     t_x = μ(2·vel_coeffs(1,0)·n_x + (vel_coeffs(2,0)+vel_coeffs(1,1))·n_y)
//           − pres_coeffs(0)·n_x
//     t_y = μ((vel_coeffs(2,0)+vel_coeffs(1,1))·n_x + 2·vel_coeffs(2,1)·n_y)
//           − pres_coeffs(0)·n_y
//
// Monomial index → derivative correspondence (same as LocalPoly / JumpData):
//   vel_coeffs row 0: (ux,  uy)  at x_i
//   vel_coeffs row 1: (∂ux/∂x, ∂uy/∂x)
//   vel_coeffs row 2: (∂ux/∂y, ∂uy/∂y)
//   pres_coeffs row 0: p at x_i
// ---------------------------------------------------------------------------

class IStokesRestrict2D {
public:
    virtual ~IStokesRestrict2D() = default;

    virtual std::vector<StokesLocalPoly2D> apply(
        const Eigen::VectorXd&                sol_ux,
        const Eigen::VectorXd&                sol_uy,
        const Eigen::VectorXd&                sol_p,
        const std::vector<StokesLocalPoly2D>& correction_polys) const = 0;

    virtual const GridPair2D& grid_pair() const = 0;
};

class IStokesRestrict3D {
public:
    virtual ~IStokesRestrict3D() = default;

    virtual std::vector<StokesLocalPoly3D> apply(
        const Eigen::VectorXd&                sol_ux,
        const Eigen::VectorXd&                sol_uy,
        const Eigen::VectorXd&                sol_uz,
        const Eigen::VectorXd&                sol_p,
        const std::vector<StokesLocalPoly3D>& correction_polys) const = 0;

    virtual const GridPair3D& grid_pair() const = 0;
};

} // namespace kfbim
