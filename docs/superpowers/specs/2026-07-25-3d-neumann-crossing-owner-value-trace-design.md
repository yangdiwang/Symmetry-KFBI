# 3D Neumann Crossing-Owner Value-Trace Study Design

## Goal

Apply the accepted 3D Dirichlet trace-restriction combination to the
Neumann exterior-zero-value-trace formulation and measure its numerical
effect without changing the Neumann boundary equation.

The transferred combination is:

- degree-three local harmonic Cauchy reconstruction;
- G1-nearest surface stencils with 48 value and 28 normal conditions;
- 4x4x4 tricubic Cartesian interpolation;
- crossing-aware correction ownership at foreign non-G1 crossings;
- precomputed owner templates, with no geometry query inside GMRES.

The existing center-owned value trace remains the reference route.

## Boundary Equation

Given the compatible Neumann data

\[
g_N=[u_n],
\]

solve for the value jump \(f=[u]\) and the scalar gauge multiplier:

\[
R_u^+P_h(f,0)+\lambda=-R_u^+P_h(0,g_N),\qquad
\int_\Gamma f\,dS=0.
\]

The unknown, compatibility correction, mean constraint, FFT box problem,
spread operator and GMRES implementation remain unchanged.  Because the
value-jump trace contains the jump term, this is a bordered second-kind
equation with a constant nullspace; changing the restrict route does not
change its operator class.

## Trace Routes

The Neumann operator receives one explicit trace mode:

1. `JointTricubicCauchy`: preserve the current behavior, assigning every
   wrong-side interpolation correction to the target surface DOF.
2. `JointTricubicCrossingOwner`: use the precomputed owner selected for each
   wrong-side node.  Foreign non-G1 crossings use the NURBS crossing patch
   and parameter coordinates; target and G1-compatible crossings retain the
   target DOF.

Both modes recover the exterior value using the existing `c0_weights_` and
do not divide by `h`.  Dirichlet normal traces continue to use
`c1_weights_/h`.  Operator application, right-hand-side construction,
post-solve exterior-trace validation and route-mismatch diagnostics must all
receive exactly the same mode.

## Preprocessing

Owner decisions are built once during pipeline setup.  The numerical study
uses the available crossing-owner template route on `main`; if the
region-closest Hybrid owner preprocessor can be integrated without changing
the accepted owner map, it is selected for the timed setup.  Otherwise the
full certified owner builder remains the correctness reference and the
Hybrid timing is reported separately rather than inferred.

No owner geometry query is permitted after GMRES begins.  Query counts and
an owner-template fingerprint are recorded before and after each solve.

## Study Driver

Add a dedicated application route:

```text
neumann_exterior_zero_trace_3d.exe --neumann-owner-study 32 64 128
```

The study uses the L-prism with three poses:

- identity;
- rotation by 17 degrees about normalized `(1,2,3)`;
- the same rotation followed by the existing `(x,y,z)` translation case.

For every pose and level it executes the legacy and crossing-owner Neumann
routes on one common pipeline.  The two routes therefore share geometry,
surface DOFs, Cauchy matrices, prescribed data and bulk solver.  GMRES uses
relative tolerance `2e-10`, maximum 80 iterations and the existing Neumann
restart/augmentation policy.

Generated files are written below
`output/neumann_value_trace_crossing_owner_3d`:

- `summary.csv`: errors, orders, iterations, residuals and timing;
- `gmres_residuals.csv`: complete residual histories;
- `owner_diagnostics.csv`: owner counts, setup mode, queries and fingerprints.

## Tests

Test-driven implementation must first demonstrate these missing behaviors:

- requesting crossing-owner exterior value trace fails before the new API
  exists;
- the legacy overload gives the same result as explicit
  `JointTricubicCauchy`;
- a synthetic foreign non-G1 correction changes the exterior value trace
  through the selected owner coefficient while preserving deterministic
  accumulation;
- the Neumann operator uses one mode consistently in `apply`, RHS and
  post-solve residual evaluation;
- owner query count and fingerprint do not change during GMRES;
- the study CLI emits complete finite rows for a small smoke level.

Existing crossing-owner, geometry, Dirichlet restrict and native-NURBS tests
must remain green.

## Acceptance

- Both routes converge at all requested levels and poses, or the precise
  failed case and residual are retained as a valid negative result.
- Every converged result has final relative residual at most `2e-10`.
- No geometry query occurs inside GMRES.
- Legacy results remain unchanged within the existing floating-point
  tolerances.
- Full and Hybrid owner preprocessing, when both are available, produce the
  same owner fingerprint and numerical solution.
- The final report compares interior/density errors, observed orders, GMRES
  counts and setup/solve time without requiring the new route to win.
