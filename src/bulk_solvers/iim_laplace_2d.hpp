#pragma once

#include "../grid/cartesian_grid_2d.hpp"
#include "../geometry/grid_pair_2d.hpp"
#include "../interface/interface_2d.hpp"
#include <Eigen/Dense>
#include <vector>

namespace kfbim {

// ---------------------------------------------------------------------------
// Immersed Interface Method (IIM) correction for the 2D Laplace equation.
//
// Problem: −Δu = f on a box domain with homogeneous Dirichlet BC,
//          interface Γ inside the domain with jump conditions [u]=a, [∂_n u]=b.
//
// Define u_int and u_ext as the smooth extensions of u into each half-domain,
// and the correction function C(x) = u_int(x) − u_ext(x), which is smooth
// everywhere.
//
// Domain labels (from GridPair2D):
//   label = 0  →  Ω_ext (exterior region)
//   label = 1  →  Ω_int (interior region)
//
// At an interior irregular node n with cross-interface interior neighbor nb,
// the 5-point stencil applied to the piecewise solution u picks up the wrong
// branch at nb.  The defect is (label[n] − label[nb]) · C[nb] / h², so the
// corrected RHS is:
//
//   F[n] = f[n]  +  Σ_{nb: cross-interface} (label[n] − label[nb]) · C[nb] / h²
//
// Sign derivation:
//   n ∈ Ω_ext (label=0), nb ∈ Ω_int (label=1):
//     u[nb] = u_int[nb] = u_ext[nb] + C[nb]  → stencil picks up an extra +C[nb]/h²
//     → must subtract it from RHS: correction = (0−1)·C[nb]/h² = −C[nb]/h² ✓
//   n ∈ Ω_int (label=1), nb ∈ Ω_ext (label=0):
//     u[nb] = u_ext[nb] = u_int[nb] − C[nb]  → stencil is short by C[nb]/h²
//     → must add it to RHS: correction = (1−0)·C[nb]/h² = +C[nb]/h² ✓
// ---------------------------------------------------------------------------

// Return flat indices of all interior irregular nodes: interior nodes that
// have at least one face-adjacent interior neighbor with a different label.
// Box boundary nodes (i=0, i=nx−1, j=0, j=ny−1) are never returned.
inline std::vector<int> iim_irregular_nodes(
    const CartesianGrid2D&  grid,
    const std::vector<int>& domain_labels)
{
    auto d  = grid.dof_dims();
    int  nx = d[0], ny = d[1];
    std::vector<int> result;

    for (int n = 0; n < nx * ny; ++n) {
        int i = n % nx, j = n / nx;
        if (i == 0 || i == nx-1 || j == 0 || j == ny-1) continue;
        int ln = domain_labels[n];
        for (int nb : grid.neighbors(n)) {
            if (nb < 0) continue;
            int inb = nb % nx, jnb = nb / nx;
            if (inb == 0 || inb == nx-1 || jnb == 0 || jnb == ny-1) continue;
            if (domain_labels[nb] != ln) { result.push_back(n); break; }
        }
    }
    return result;
}

// Compute the IIM-corrected RHS for −Δu = f.
//
// Inputs:
//   grid          — Node-layout grid (dof_dims includes boundary nodes).
//   f             — Piecewise RHS evaluated at every grid node (0 at boundary).
//   C             — Correction function C(x) = u_int(x) − u_ext(x) at every node.
//   domain_labels — Per-node label (0 = Ω_ext, 1 = Ω_int), e.g. from GridPair2D.
//
// Returns F, the corrected physical RHS for −Δ_h u = F.
// Solver convention: LaplaceFftBulkSolverZfft2D solves Δ_h u = rhs, so pass
// −F (not F) to the solver:  solver.solve(−F, u).
inline Eigen::VectorXd iim_correct_rhs(
    const CartesianGrid2D&  grid,
    const Eigen::VectorXd&  f,
    const Eigen::VectorXd&  C,
    const std::vector<int>& domain_labels)
{
    auto d  = grid.dof_dims();
    int  nx = d[0], ny = d[1];
    double h2 = grid.spacing()[0] * grid.spacing()[0];

    Eigen::VectorXd F = f;

    for (int n = 0; n < nx * ny; ++n) {
        int i = n % nx, j = n / nx;
        if (i == 0 || i == nx-1 || j == 0 || j == ny-1) continue;
        int ln = domain_labels[n];
        for (int nb : grid.neighbors(n)) {
            if (nb < 0) continue;
            int inb = nb % nx, jnb = nb / nx;
            // Skip box-boundary neighbors: their value is 0 by Dirichlet BC,
            // which the solver enforces directly — no interface crossing there.
            if (inb == 0 || inb == nx-1 || jnb == 0 || jnb == ny-1) continue;
            int lnb = domain_labels[nb];
            if (lnb == ln) continue;
            F[n] += static_cast<double>(ln - lnb) * C[nb] / h2;
        }
    }
    return F;
}

// ---------------------------------------------------------------------------
// IIM correction using a degree-2 Taylor expansion of C from the nearest
// interface quadrature point.
//
// Instead of the global correction function C(x) = u_int(x) − u_ext(x), the
// caller supplies C and its first/second partial derivatives evaluated at
// each interface quadrature point.  For each cross-interface interior
// neighbor nb, we locate the nearest quadrature point q and approximate:
//
//   C(x_nb) ≈ C_q + Cx_q·dx + Cy_q·dy
//              + ½ Cxx_q·dx² + Cxy_q·dx·dy + ½ Cyy_q·dy²
//
// where (dx, dy) = x_nb − x_q.
//
// Taylor error: O(h³)  (nearest q is O(h) away when interface pts ~ 8N)
// Correction error: O(h³)/h² = O(h)
// → Global solution error is O(h)  (first-order convergence)
//
// Inputs:
//   gp            — GridPair2D providing closest_interface_point(nb)
//   iface         — Interface2D providing quadrature point coordinates
//   C_q, Cx_q, Cy_q, Cxx_q, Cxy_q, Cyy_q — per-quadrature-point values
//   All other arguments: same as iim_correct_rhs.
// ---------------------------------------------------------------------------
inline Eigen::VectorXd iim_correct_rhs_taylor(
    const CartesianGrid2D&  grid,
    const GridPair2D&       gp,
    const Interface2D&      iface,
    const Eigen::VectorXd&  f,
    const Eigen::VectorXd&  C_q,
    const Eigen::VectorXd&  Cx_q,
    const Eigen::VectorXd&  Cy_q,
    const Eigen::VectorXd&  Cxx_q,
    const Eigen::VectorXd&  Cxy_q,
    const Eigen::VectorXd&  Cyy_q,
    const std::vector<int>& domain_labels)
{
    auto d  = grid.dof_dims();
    int  nx = d[0], ny = d[1];
    double h2 = grid.spacing()[0] * grid.spacing()[0];

    Eigen::VectorXd F = f;

    for (int n = 0; n < nx * ny; ++n) {
        int i = n % nx, j = n / nx;
        if (i == 0 || i == nx-1 || j == 0 || j == ny-1) continue;
        int ln = domain_labels[n];
        for (int nb : grid.neighbors(n)) {
            if (nb < 0) continue;
            int inb = nb % nx, jnb = nb / nx;
            if (inb == 0 || inb == nx-1 || jnb == 0 || jnb == ny-1) continue;
            int lnb = domain_labels[nb];
            if (lnb == ln) continue;
            // Degree-2 Taylor expansion of C at nb from nearest interface point.
            int q = gp.closest_interface_point(nb);
            auto cnb = grid.coord(nb);
            double dx = cnb[0] - iface.points()(q, 0);
            double dy = cnb[1] - iface.points()(q, 1);
            double C_approx = C_q[q] + Cx_q[q]*dx + Cy_q[q]*dy
                              + 0.5*Cxx_q[q]*dx*dx + Cxy_q[q]*dx*dy
                              + 0.5*Cyy_q[q]*dy*dy;
            F[n] += static_cast<double>(ln - lnb) * C_approx / h2;
        }
    }
    return F;
}

} // namespace kfbim
