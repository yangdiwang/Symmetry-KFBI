# 3D Harmonic-Jet Full-Solve Design

## Goal

Complete the existing three-dimensional app so that it performs numerical
KFBI solves rather than geometry readiness alone.  Keep all new implementation
inside `apps/neumann_exterior_zero_trace_3d.cpp` for this stage.

The completed app must:

- retain the existing readiness checks;
- execute the existing exterior-zero-trace Neumann formulation;
- add the Dirichlet normal-jump second-kind formulation;
- use the same cubic local Cauchy polynomial for spreading and restriction;
- run torus, hollow-cylinder, and L-prism studies at `N=16,32,64`.

## Canonical 3D Cauchy polynomial

Use the degree-three three-dimensional harmonic polynomial space.  Its
dimension is `(3+1)^2 = 16`.  Retain the existing topology-filtered 48 value
samples and 28 normal-derivative samples initially; they provide an
overdetermined weighted least-squares fit and avoid introducing a second
stencil change in the same work.

Each surface DOF owns one canonical coefficient vector reconstructed from
`[u]` and `[u_n]`:

- a smooth-face DOF selects nearby samples on its connected patch;
- an edge or vertex DOF may use samples from its incident patches;
- samples from nonincident patches or disconnected surface sheets are
  forbidden;
- the same coefficient vector is consumed by both spread and restrict.

For the hollow cylinder, this prevents mixing the inner and outer cylindrical
sheets.  For the L-prism, incident faces at an edge or vertex participate in
one three-dimensional harmonic Cauchy fit rather than independent face fits.

## Spread, bulk solve, and restrict

Spread evaluates the canonical cubic polynomial at the existing fixed
crossing-correction nodes and adds the signed discrete-Laplacian correction to
the Cartesian right-hand side.

The bulk problem remains the existing three-dimensional ZFFT Laplace solve
with homogeneous Dirichlet data on the embedding box.

Restrict samples both sides of every surface DOF at

`rho/h = +/-0.5, +/-1.5, +/-2.5, +/-3.5`.

Each sample uses a `4x4x4` tensor-product tricubic interpolation.  Stencil
nodes lying on the wrong side of the interface are corrected with the same
canonical cubic Cauchy polynomial used by spread.  Interior samples are then
continued to the exterior branch with the prescribed jumps.

Fit the eight corrected samples with the existing joint piecewise-cubic normal
model.  The two sides share their value and first normal derivative at the
interface and have independent quadratic and cubic coefficients.  Row zero of
the pseudoinverse gives the exterior value trace.  Row one gives the exterior
normal derivative with respect to `rho/h`, so the physical derivative is
`c1_weights.dot(samples) / h`.

## Boundary integral formulations

Let `P_h(a,b)` denote the discrete KFBI potential with value jump `a` and
normal jump `b`.

### Neumann exterior-zero-trace formulation

Given `g_N=[u_n]`, solve for `f=[u]` and a bordered multiplier:

`R_u^+ P_h(f,0) + lambda = -R_u^+ P_h(0,g_N)`,

`integral_Gamma f dS = 0`.

This reuses the existing augmented GMRES operator, but the app must now invoke
it and report the physical residual and gauge condition.  Remove the discrete
weighted mean from sampled Neumann data and report the removed compatibility
shift.  Compare the recovered value jump and bulk solution after the same
weighted constant alignment.

### Dirichlet normal-jump second-kind formulation

Given `g_D=[u]`, solve for `q=[u_n]`:

`R_n^+ P_h(0,q) = -R_n^+ P_h(g_D,0)`.

The operator residual is the directly restricted exterior normal derivative.
No bordered constraint is added.  Independently reconstruct the same exterior
normal derivative from the opposite trace and the jump, and report the route
mismatch.

After convergence, evaluate `P_h(g_D,q)` and report the interior Dirichlet
trace, recovered density, exterior normal trace, and bulk errors.

## Manufactured solution

Use the same non-polynomial, fully three-dimensional harmonic function for all
geometries:

`u(x,y,z) = exp(0.35 x) cos(0.21 y) cos(0.28 z)`.

It is harmonic because `0.35^2 = 0.21^2 + 0.28^2`.  Evaluate its analytic
gradient and define

- `g_D = u` on the interface;
- `g_N = grad(u) dot n` on the interface.

The intended piecewise field is this function in the physical domain and zero
in the exterior box region.

## App interface and output

Preserve the existing executable and geometry selector.  Extend the grid
arguments so the old command remains valid and multiple levels are accepted:

```text
neumann_exterior_zero_trace_3d.exe all 16 32 64
```

For each geometry and level, execute readiness first, followed by the Neumann
and Dirichlet normal-second-kind solves.  Keep the existing readiness CSV and
write separate Neumann and Dirichlet convergence CSV files.

Each numerical row records at least:

- geometry, `N`, `h`, surface DOFs, and Cauchy conditioning;
- GMRES iterations, convergence flag, and relative residual;
- target exterior-trace or exterior-normal residual;
- direct/opposite-route mismatch;
- interface value and normal-density errors;
- interior bulk `Linf` and `L2` errors;
- exterior bulk `Linf` error;
- compatibility or gauge values and runtime;
- observed refinement orders.

## Error handling and acceptance

- Reject non-power-of-two levels below 16.
- Reject nonfinite Cauchy maps, traces, residuals, or error metrics with a
  geometry-and-level-specific message.
- Preserve the existing geometry-label mismatch guard.
- A Release build of the app must succeed with the current x64 build.
- `all 16` must complete readiness and both GMRES solves for all geometries.
- `all 16 32 64` must produce exactly 18 finite, converged numerical rows:
  three geometries times three levels times two formulations.
- For every geometry and formulation, the `N=64` interior bulk `L2` error must
  be smaller than its `N=16` value.  Observed orders are reported rather than
  forced to an assumed asymptotic rate.
- The algebraic GMRES tolerance defaults to the existing two-dimensional value
  `2e-10`.  It remains a relative algebraic tolerance independent of `h`;
  physical trace residuals are reported separately so any `1/h` amplification
  in normal restriction remains visible.

## Non-goals

- Do not move the pipeline into `src/` in this change.
- Do not template-unify the two- and three-dimensional implementations.
- Do not add alternative tricubic/triquadratic comparison modes yet.
- Do not change the outer homogeneous Dirichlet box problem.
