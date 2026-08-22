# NURBS Same-Parameter Cauchy Jet (NSP-CJ) in 2D

## Definition

NSP-CJ makes the NURBS parameter the common coordinate for geometry,
interface-density reconstruction, crossing localization, and the local Cauchy
polynomial. The geometric boundary is

\[
  X(t)=\frac{\sum_i N_{i,p}(t)w_iP_i}
              {\sum_i N_{i,p}(t)w_i}.
\]

The L boundary is one closed degree-one NURBS curve. Its seven control points
are the six vertices followed by the first vertex, all weights are one, and
the six knot intervals are proportional to physical edge length. Consequently
the L geometry and every rigid transform of it are represented exactly.

The old P2 rows remain only as compatibility cells for `GridPair2D` and the
existing KFBI operator API. Their point, tangent, second derivative, normal,
curvature, and crossing queries are all redirected to the authoritative NURBS
provider; they are not a second geometric approximation.

## Density spaces

Each active density DOF stores a NURBS parameter \(t_i\). On a smooth closed
branch, NSP-CJ constructs a periodic cubic B-spline for
\(\phi=[u]\) and a periodic quadratic B-spline for
\(\psi=[u_n]\). On the L shape, each physical edge is an independent one-sided
smooth branch:

- \(\phi\): P3 extrapolation from the first/last four interior DOFs;
- \(\psi\): P2 extrapolation from the first/last three interior DOFs;
- the two one-sided \(\phi\) predictions share their averaged scalar corner
  value;
- \(\psi\) stays one-sided because the corner has two different normals.

The resulting open clamped B-splines interpolate the interior DOFs and the two
derived true-span endpoints. Thus \(\phi\in C^2\) and \(\psi\in C^1\) while a
crossing moves inside one smooth branch. No derivative continuity is imposed
across a physical corner.

## Crossing-local polynomial

For a represented-panel crossing, its NURBS parameter comes directly from the
panel-to-parameter map. For an explicitly certified corner-gap hit, the true
`crossing_point` is projected onto the selected one-sided NURBS branch. A
degree-one span uses an analytic line projection; general NURBS branches use a
bounded curve projection.

Let

\[
  J=\lVert X_t\rVert,\qquad
  J_t=\frac{X_t\cdot X_{tt}}{J}.
\]

The spline derivatives are converted to physical arclength derivatives by

\[
  \phi_s=\frac{\phi_t}{J},\qquad
  \phi_{ss}=\frac{\phi_{tt}}{J^2}
            -\frac{\phi_tJ_t}{J^3},\qquad
  \psi_s=\frac{\psi_t}{J}.
\]

Exact NURBS differential geometry supplies \(T\), \(N\), and

\[
  \kappa=-\frac{X_{tt}\cdot N}{J^2}.
\]

The complete frozen-frame quadratic polynomial at the actual crossing uses

\[
\begin{aligned}
 P_{tt}&=\phi_{ss}+\kappa\psi,\\
 P_{tn}&=\psi_s-\kappa\phi_s,\\
 P_{nn}&=\alpha\phi-[f]-\phi_{ss}-\kappa\psi.
\end{aligned}
\]

Spread and restrict retain the same immutable NURBS/spline plan and the same
per-application fitted density state. They therefore evaluate the identical
crossing-bound polynomial.

## L-shape convergence

Configuration: harmonic cubic manufactured solution, joint bicubic/cubic
crossing-owner restrict, GMRES tolerance \(10^{-8}\), and NSP-CJ. Errors are
maximum norms.

| N | h | active DOFs | trace error | order | normal error | order | interior bulk error | order | GMRES |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.100 | 86  | 4.110e-3 | -     | 3.596e-3 | -     | 4.158e-3 | -     | 12 |
| 60 | 0.050 | 166 | 1.135e-3 | 1.856 | 9.005e-4 | 1.998 | 1.140e-3 | 1.867 | 13 |
|120 | 0.025 | 326 | 2.978e-4 | 1.931 | 2.250e-4 | 2.001 | 2.983e-4 | 1.934 | 14 |

The legacy status counter reports 334/360/360 corner-gap owners. The refined
diagnostics classify all of them as certified intersections consumed by
NSP-CJ: unresolved gap fallbacks are 0, endpoint fallbacks are 0, and domain
label mismatches are 0 on every level.

The complete data are in
`output/neumann_exterior_trace_lshape_2d_p2_crossing_owner_joint_cubic_nsp_cj.csv`.

## ALS-CJ comparison and rigid transforms

With the same exact NURBS geometry and restrict method, ALS-CJ gives nearly the
same axis-aligned asymptotic error. NSP-CJ extends its density space to the
true branch endpoints and can use certified gap crossings directly. At
\(N=60\), representative rigid-transform results are:

| transform | NSP-CJ bulk error | ALS-CJ bulk error | NSP-CJ GMRES | ALS-CJ GMRES |
|:--|--:|--:|--:|--:|
| baseline | 1.140e-3 | 1.139e-3 | 13 | 12 |
| translate \((2h,-h)\) | 1.142e-3 | 1.141e-3 | 18 | 16 |
| translate \((0.137,-0.083)\) | 2.518e-3 | 3.958e-3 | 30 | 31 |
| rotate \(90^\circ\) | 1.141e-3 | 1.140e-3 | 17 | 16 |
| rotate \(17^\circ\) | 1.863e-3 | 2.013e-3 | 19 | 18 |
| rotate \(17^\circ\) + translate | 2.115e-3 | 1.649e-3 | 21 | 20 |

The rigid study shows that NSP-CJ removes geometry approximation and
unresolved crossing fallback, but it does not by itself make the bordered
exterior-trace equation rigid-motion invariant. The augmented GMRES residual
converges in every transformed case, while a nonzero bordered multiplier can
leave the separately checked physical residual above \(10^{-8}\). That
remaining issue belongs to the constant-mode/bordered formulation rather than
the NURBS density reconstruction.

Rigid-study data:

- `output/neumann_exterior_trace_lshape_2d_rigid_transform_p2_crossing_owner_joint_cubic_nsp_cj.csv`
- `output/neumann_exterior_trace_lshape_2d_rigid_transform_p2_crossing_owner_joint_cubic_als_cj.csv`

## Selection

The L-shape application selects NSP-CJ by default. It can be changed without
recompilation:

```text
KFBIM_LSHAPE_CROSSING_JET=nsp_cj   # default
KFBIM_LSHAPE_CROSSING_JET=als_cj   # comparison
KFBIM_LSHAPE_CROSSING_JET=local    # compatibility path
```
