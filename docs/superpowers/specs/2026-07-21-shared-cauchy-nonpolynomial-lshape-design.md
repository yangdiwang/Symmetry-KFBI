# Shared Cauchy Polynomial and Non-polynomial L-shape Design

## Goal

Make the Dirichlet `normal_jump_second_kind` operator use the same local
Cauchy jump polynomial for both RHS spreading and exterior-normal restriction,
then measure the four grid-interpolation/normal-fit degree combinations on a
non-polynomial harmonic solution.

## Scope

- Keep `KFBIM_PYJET_SPREAD_MODE=cubic_harmonic` as the comparison route.
- Keep all existing restrict selector names and legacy selector membership.
- Keep grid interpolation degree and the final one-dimensional normal fit
  degree independently selectable by `degree_compare`.
- Run the L-shape geometry at `N=64,128,256,512,1024` with
  `uniform_midpoint`, `normal_jump_second_kind`, and `tol=2e-10`.

## Canonical Local Cauchy Polynomial

For cubic-harmonic spreading, each interface DOF has one canonical local
Cauchy polynomial reconstructed from the jump value and jump normal derivative:

- Away from L-shape corners, use the existing same-edge cubic map with four
  value samples and three normal-derivative samples.
- For DOFs marked by the existing hybrid corner policy, use the existing
  cross-corner two-dimensional cubic map with five value samples and four
  normal-derivative samples.

One coefficient-building function selects the correct map per DOF. Both
`rhs_from_jumps()` and `one_sided_samples()` consume those coefficients. They
must not independently select quadratic, full harmonic-jet, or another
neighbor set when `cubic_harmonic` is active.

The restrict interpolation stencil remains biquadratic or bicubic according to
the restrict mode. Only the interpolation weights change between those modes;
the Cauchy correction applied to opposite-side stencil nodes comes from the
same canonical cubic polynomial. The subsequent joint normal fit remains
quadratic or cubic according to the second degree choice.

Other spread modes retain their existing behavior; this change does not invent
a Cauchy polynomial for `crossing_density`.

## Manufactured Solution

Remove the polynomial-only L-shape special case and use

\[
u(x,y)=\exp(0.42x)\cos(0.42y),
\]

with analytic gradient

\[
\nabla u = \left(0.42\exp(0.42x)\cos(0.42y),
-0.42\exp(0.42x)\sin(0.42y)\right).
\]

This function is smooth and harmonic on the embedding box and cannot be
reproduced exactly by a finite-degree polynomial fit.

## Diagnostics and Compatibility

- Report that restrict uses the shared cubic-spread Cauchy source when this
  route is active.
- Report the condition number of the canonical per-DOF Cauchy map, including
  the cross-corner map for marked corner DOFs.
- Preserve `normal_compare`, `compare`, and `degree_compare` membership.
- Add an L-shape smoke assertion that all four `degree_compare` cases use the
  shared Cauchy source and converge.

## Numerical Acceptance

- The Release target builds successfully.
- The L-shape smoke produces exactly four converged runs.
- The full study produces 20 finite, converged rows: four modes at five levels.
- Report `Linf`, `L2`, observed orders, inner Dirichlet residual, density error,
  exterior-normal residual, GMRES iterations, condition number, and runtime.
- Interpret the results without treating the non-polynomial case as exact
  polynomial reproduction.
