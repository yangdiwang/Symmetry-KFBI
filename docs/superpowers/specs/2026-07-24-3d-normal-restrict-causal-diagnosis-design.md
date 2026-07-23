# 3D Normal-Restrict Causal Diagnosis Design

## Goal

Determine whether the rotation-induced GMRES deterioration in the main 3D
Dirichlet exterior-zero-normal-trace formulation is caused primarily by the
current normal-trace restriction.

The experiment must distinguish three claims:

1. the current restrict has a rotation-dependent consistency defect;
2. that defect is localized near geometric features rather than distributed
   through the bulk;
3. changing only the restrict changes the GMRES behavior.

The experiment is diagnostic only. It does not replace the production default
or claim that the reference restrict is the final algorithm.

## Cases and Cost Limit

Use the existing L-prism rigid cases:

- `baseline`;
- `rot_axis123_17deg`;
- `rot_axis123_17deg_t_xyz_1`.

Run `N=32,64` first. Use GMRES tolerance `2e-10` and a diagnostic maximum of
160 iterations. Run `N=128` only if the first two levels do not distinguish
the hypothesis.

Keep the current main configuration fixed:

- `g1_nearest` Cauchy stencils;
- degree-three harmonic Cauchy fitting;
- 48 value samples and 28 normal samples;
- the same native NURBS geometry, surface DOFs, spread correction, FFT bulk
  solver, and manufactured Dirichlet data.

## Two Restrict Routes

Add an explicit diagnostic restrict mode:

```cpp
enum class ExteriorNormalRestrictMode3D {
    JointTricubicCauchy,
    ExteriorOnlyHarmonicCubic
};
```

`JointTricubicCauchy` is the current route without numerical or behavioral
changes: four normal layers on each side, `4x4x4` tensor-product tricubic
interpolation, Cauchy continuation for support nodes on the wrong side, and
the existing joint cubic normal fit.

`ExteriorOnlyHarmonicCubic` is an independent reference route. At each surface
DOF it:

1. gathers Cartesian nodes from the exterior label only;
2. selects the 48 nearest candidates from a bounded local cube;
3. expresses their coordinates in the DOF's `(t1,t2,n)` frame and scales them
   by `h`;
4. constructs a weighted least-squares fit in the existing degree-three
   harmonic polynomial space;
5. precomputes the linear weights that return the normal derivative at the
   surface point, including the physical `1/h` scale.

The reference route reads only exterior bulk values. It does not use jump
data or cross-interface Cauchy continuation. Every local fit must have full
rank, finite weights, and a reported condition number. An invalid fit is a
hard diagnostic failure rather than a fallback to the current route.

## Exact-Grid Restrict Isolation

For every case and level, construct a piecewise exact grid field without
calling spread or the FFT solver:

```text
U*(x_i) = u_T(x_i),  x_i inside
        = 0,         x_i outside
```

At the surface DOFs use:

```text
[U]*   = u_T
[U_n]* = grad(u_T) dot n
```

Build the current local Cauchy coefficients from these exact jump data, then
apply both restrict routes to `U*`. The exact exterior normal trace is zero.
This produces:

- current restrict `Linf` and weighted RMS error;
- reference restrict `Linf` and weighted RMS error;
- the error at every surface DOF;
- the number of wrong-side nodes used by the current trace templates.

Because this probe bypasses spread, the FFT solve, and GMRES, a defect found
here belongs to the boundary recovery path.

Also evaluate the exact density in the complete discrete equation:

```text
r_exact = A_h q_exact - b_h
```

for both restrict modes. Comparing this result with the exact-grid probe
separates the restrict contribution from the remaining spread/bulk
discretization contribution.

## Causal GMRES A/B

Construct both discrete operators from the same initialized pipeline. The
only switched component is the exterior normal restrict used by
`ExteriorNormalTraceOperator3D::apply`.

Two GMRES comparisons are required:

1. the physical manufactured Dirichlet equation, with each route used
   consistently in both the operator and its right-hand side;
2. a normalized deterministic common right-hand side, identical for both
   matrices, to prevent changes in the manufactured right-hand side from
   explaining an iteration difference.

Both solves start from zero and use the same tolerance, maximum iterations,
and restart policy. Store the full relative-residual history.

The current route must reproduce the already observed rotation-related
iteration increase before any conclusion is drawn.

## Localization

Retain the physical non-G1 feature-edge segments from the NURBS
triangulation. For every surface DOF report:

- case and `N`;
- DOF and patch identifiers;
- position and normal;
- distance divided by `h` to the nearest non-G1 feature edge;
- current trace-template wrong-side-node count;
- exact-grid restrict error for both routes;
- exact-density equation residual for both routes.

Summaries use distance bins:

```text
[0,1)h, [1,2)h, [2,4)h, [4,8)h, [8,infinity)h
```

Report `Linf`, weighted RMS, and sample count in every bin. This determines
whether a max-norm defect is concentrated near non-G1 edges.

## Decision Rule

The statement “the current restrict is the primary cause of rotation-induced
GMRES deterioration” is supported only if all of the following hold:

1. the reference restrict passes polynomial-reproduction and conditioning
   tests;
2. the current route reproduces the rotation-related iteration increase;
3. the exact-grid current-restrict error is amplified by rotation and the
   reference route does not show comparable amplification;
4. changing only the restrict removes at least half of the excess rotated
   GMRES iterations relative to the corresponding baseline:

   ```text
   I_ref(rot) - I_ref(base)
       <= 0.5 * (I_current(rot) - I_current(base))
   ```

5. the same qualitative result holds for both the physical equation and the
   common-right-hand-side experiment.

If any condition fails, report the restrict hypothesis as not proven. Do not
reinterpret a failed test as confirmation.

## Code and Output

Create:

- `apps/exterior_only_cubic_normal_restrict_3d.hpp`;
- `apps/exterior_only_cubic_normal_restrict_3d.cpp`;
- `apps/exterior_only_cubic_normal_restrict_3d_test.cpp`.

Modify:

- `apps/neumann_exterior_zero_trace_3d.cpp`;
- `apps/CMakeLists.txt`.

Add:

```text
neumann_exterior_zero_trace_3d --restrict-probe [N ...]
```

The default probe levels are `32 64`. Legacy app modes and the existing rigid
study remain unchanged.

Write only beneath:

```text
output/dirichlet_normal_restrict_causal_probe_3d
```

with:

- `summary.csv`: route, errors, equation residuals, GMRES counts, timings, and
  causal ratios;
- `dof_diagnostics.csv`: per-DOF localization data;
- `gmres_residuals.csv`: complete residual histories;
- `edge_distance_bins.csv`: binned localization summaries.

Write accumulated output after every completed case/level/route.

## Tests

The focused reference-restrict test must verify:

- exact normal derivatives for constant, linear, and degree-three harmonic
  polynomials on several rotated local frames;
- invariance of the polynomial results under rigid rotation and translation;
- rejection of rank-deficient or insufficient same-side samples;
- finite local weights and condition numbers;
- candidate selection uses exterior labels only.

App verification must cover:

- the new help text and `--restrict-probe 16` smoke path;
- both operator and right-hand-side construction use the selected route;
- the current default route produces unchanged legacy results;
- all existing native-NURBS and rigid-transform tests continue to pass.

The numerical evidence run is the three approved cases at `N=32,64`, followed
by `N=128` only if required by the decision rule.
