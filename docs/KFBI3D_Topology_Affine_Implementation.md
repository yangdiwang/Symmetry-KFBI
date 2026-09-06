# KFBI3D topology-affine implementation status

This document records the C++ implementation derived from
`KFBI3D_Codex_Implementation_Spec.md`. It describes what is present in the
repository, the boundary between the new route and shared legacy code, and
the implementation state through 2026-09-04. Numerical claims are limited to
the verification records explicitly identified below. The topology-affine
route is an additional selectable scheme; it does not replace the legacy
executable or its defaults.

## Entry point, selectors, and defaults

- New executable: `kfbi_topology_affine_exterior_trace_3d`
- Existing executable retained: `kfbi_native_c0_exterior_trace_3d`
- Validation variant: `topology_affine` in
  `apps/run_native_c0_validation.ps1`
- Formal-default output root when both formulations run:
  `output/kfbi_topology_affine_3d/q27_c3/d_q64_c4/nm_pc_tm_mf/topology_affine`
- Neumann-only validation-runner output root:
  `output/kfbi_topology_affine_3d/q27_c3/nm_pc_tm_mf/topology_affine`

Both executables compile `apps/neumann_exterior_zero_trace_3d.cpp` and link
the same `kfbim_3d_app_geometry` and `kfbim_core` libraries. The new target
adds both compile definitions
`KFBIM_3D_NATIVE_COEFFICIENT_DEFAULT=1` and
`KFBIM_3D_TOPOLOGY_AFFINE_DEFAULT=1`; the legacy target adds only the first.
Thus this is a separately selected algorithmic branch and output namespace,
not an isolated copy of the full KFBI solver. Geometry, Cartesian-grid PDE
solves, crossing construction, GMRES, and several restrict utilities remain
shared. Changes in those common modules are visible to both targets, while
the legacy target keeps its prior run-time defaults.

The following are the no-environment-override defaults of the new target.
The last column notes the settings imposed by the validation script used for
the numerical runs below.

| Selector | New-target default | Validation setting |
|---|---:|---:|
| `KFBIM_3D_DENSITY_MODE` | `reduced_coefficients` | same |
| `KFBIM_3D_NEUMANN_EDGE_JUMP_JET` | `topology_affine_local_svd` | same |
| `KFBIM_3D_NEUMANN_DENSITY_COORDINATES` | `trace_mass` | same |
| `KFBIM_3D_NEUMANN_COMPATIBILITY` | `trace_border_legacy` | same |
| `KFBIM_3D_NEUMANN_BORDER_SOLVER` | `mean_free_pivot_elimination` | same |
| `KFBIM_3D_NEUMANN_TRACE_RESTRICT` | `q27_cover3_all_event_cauchy` | same |
| `KFBIM_3D_DIRICHLET_NORMAL_RESTRICT` | `q64_cover4_all_event_cauchy` | not exercised |
| `KFBIM_3D_NEUMANN_TRACE_SAMPLING` | `panel_centers` | same |
| `KFBIM_3D_FEATURE_TRACE_FIT` | `geometric_feature` | same |
| `KFBIM_3D_SOLVE_SELECTION` | `both` | `neumann_only` |

The Neumann feature selector also accepts the opt-in A/B backend
`topology_ambient_gradient_local_svd`.  The default remains
`topology_affine_local_svd`, so this feature-constraint default is unchanged.

These remain run-time selectors: setting an environment variable can override
the corresponding compile-time fallback. In particular, the new executable
does not imply `neumann_only`; that restriction comes from the validation
script. The validation script selects Q27 explicitly so the formal Neumann
baseline is auditable. Q10 must be chosen explicitly for a comparison run and
is not an implicit fallback from either direct-cover route.

## Source map

| Source | Responsibility |
|---|---|
| `apps/CMakeLists.txt` | New target, shared library membership, and test executables |
| `apps/neumann_exterior_zero_trace_3d.cpp` | Selectors, four-stage driver integration, final mean-free Neumann operator, Dirichlet branch, restrict route, diagnostics, and CSV output |
| `apps/native_nurbs_density_space_3d.{hpp,cpp}` | `TopologyBase`/`BaseOnly` coefficient layout, implicit identity reduction, sparse local value/derivative stencils |
| `apps/topology_density_constraints_3d.{hpp,cpp}` | Sparse topology moments, smooth-sheet IDs, vertex/T-star and edge blocks, Neumann feature jump-jet operators, and analytic nonhomogeneous Dirichlet feature constraints |
| `apps/topology_affine_reduction_3d.{hpp,cpp}` | Sparse affine state `p+Ez`, staged current-`C E` support closure, row-scaled component SVD, and invariants |
| `apps/topology_mean_free_reduction_3d.{hpp,cpp}` | Final global rank-one mean constraint, deterministic maximum-moment pivot, and sparse `K -> K-1` affine map |
| `apps/topology_reachable_target_projection_3d.{hpp,cpp}` | Projection of a requested feature target onto the value-topology reachable range |
| `apps/topology_trace_projector_3d.{hpp,cpp}` | Unified reduced trace projector and trace-mass coordinate transforms |
| `apps/reduced_trace_projection_3d.{hpp,cpp}` | Opt-in weighted QR and large sparse Gram/Cholesky projector backends; legacy dense-LLT default retained |
| `apps/direct_coefficient_cauchy_3d.{hpp,cpp}` | Analytic coefficient-to-Cauchy rows, rational geometry derivatives, inverse chain rule, and conversion of known Dirichlet value/ambient-gradient/ambient-Hessian data into crossing jets |
| `apps/native_nurbs_surface_3d.{hpp,cpp}` | Multi-neighbor G1 topology and partial/reversed edge maps used by crossing ownership and sheet traversal |
| `apps/tensor_product_cover_restrict_3d.{hpp,cpp}` and the main driver | Formal-default Q27/Q64 Cartesian tensor-product covers, direct interface value weights, and direct outward-normal derivative weights |
| `apps/shared_quadratic_restrict_3d.{hpp,cpp}` and the main driver | Comparison-only shared ten-node Cartesian quadratic stencil machinery and the topology cubic six-sample recovery profile |
| `apps/restrict_crossing_selector_3d.{hpp,cpp}` | Nearest same-sheet grid-line crossing selection for wrong-side corrections |
| `apps/csv_rfc4180.hpp` | Quoting of diagnostic string fields such as merged block keys |
| `apps/run_native_c0_validation.ps1` | Legacy/topology variant selection and selectable build directory |
| `apps/run_kfbi_3d_full_validation.ps1` | Seven-geometry and rigid-pose matrix that pins Q27 for topology Neumann and Q64 for topology Dirichlet and normalizes the result schemas |

## Preprocessing and affine coefficient space

### 1. `TopologyBase`

`NativeDensityReductionBackend3D::TopologyBase` is an alias of the explicit
`BaseOnly` backend. It performs the exact full-edge Union-Find merge and then
stops. Partial-edge C0 and smooth-C1 equations are intentionally left to the
new sparse topology layer. Its base-to-C0 map is an implicit identity: the
implementation neither stores a dense identity matrix nor runs the legacy
global dense nullspace construction.

The value and normal fields use different topology semantics. Value traces
are C0 across geometric features and acquire physical common-co-normal C1
constraints only across G1 sheets. Normal traces stay broken where the
geometric normal jumps; smooth T-junction constraints are restricted to their
physical sheet.

### 2. Constraint assembly and actual quadrature orders

`topology_density_constraints_3d` writes only the touched spline coefficients
to CSR rows. Long/short edge pairs use a common knot overlay and cellwise
Legendre modes P0--P3.

There are two distinct quadrature settings and they must not be conflated:

- C0 and smooth common-co-normal C1 topology moments use
  `TopologyDensityConstraintOptions3D::gauss_order=5`; the implementation
  currently requires exactly the five-point Gauss-Legendre rule.
- Feature jump-jet rows use
  `NativeFeatureEdgeJumpJetOptions3D::mortar_gauss_order`; the supported range
  is 4--5 and the production driver currently uses its default value **4**.

The reported N=32 runs use panel centers as the independent trace-projector
test set. `trace_sampling=panel_centers` is unrelated to either edge-moment
Gauss rule.

The final topology projector enforces strict oversampling before it forms a
QR or Cholesky factorization.  Because every retained trace weight must be
finite and strictly positive, the effective observation count is the number
of panel-center rows, and construction requires

```text
positive-weight trace samples > final homogeneous GMRES coordinates.
```

The comparison is made after all local topology constraints and, for
Neumann, after the global mean coordinate has been eliminated.  It is not a
comparison with raw patch coefficients or an intermediate C0 space.  Full
column rank and the configured condition bound remain separate, stronger
checks.  Console and CSV diagnostics report the sample count, final DOFs,
integer margin, and ratio.

### 3. Strict staged affine elimination

The reducer maintains

```text
c_base = p + E z.
```

The topology route explicitly chooses
`AffineEliminationSchedule3D::StagedVertexThenEdge`. It eliminates every
vertex/T-star component first, recomputes `C E`, and only then forms and
eliminates edge-interior components. A component is therefore determined by
shared active columns in the **current** `C E`, not solely by the original
row support. Components never merge across the vertex/edge stage boundary.

Before each elimination, pending same-stage blocks are closed by current
support. If a local SVD reports incompatibility, the reducer recomputes
`C E` and retries once the anchor block has been enlarged by a deterministic,
zero-drop-tolerance closure with all still-pending blocks in that stage. If
that closure cannot grow the component, the operation fails with an explicit
`ConstraintIncompatibility3D`; no approximate local solution is accepted.

For each resulting component, only its active columns are materialized in a
row-scaled dense SVD. This is graph-local, but locality is not a mathematical
size bound: if the active-column graph is connected, one component can contain
the complete stage (or complete feature target) and its dense SVD is then
effectively global. Both N=32 runs below report one final elimination record
and one reachable-target component. The implementation avoids an
unconditional whole-problem dense matrix/global LSMR, but it does not promise
that every connected component stays small.

After each block and at completion, the reducer audits the processed-row and
full-system values of `||C p-d||_inf` and `||C E||_max`.

### 4. Known Neumann data and reachable feature targets

The prescribed normal samples `prescribed_normal_jump` are first projected
into an independently represented, sheet-aware broken `NormalTrace`
coefficient space. That fitted `normal_c0` is used only to construct the
discrete feature constraint target

```text
d_feature = C_normal normal_c0.
```

After the homogeneous value-topology space `p_t+E_t z` is fixed, the code
forms `F=C_feature E_t`. Each zero-tolerance active-column component uses an
SVD to project the requested feature right-hand side onto `range(F)`, and a
preimage check certifies that the projected target is reachable. As with the
reducer, a connected component may in the worst case be the whole target
system.

This projection does **not** replace the physical Neumann boundary data. The
raw `prescribed_normal_jump` and the analytic known-normal jet continue to be
used by the KFBI spread/base-field construction, crossing Cauchy data, and
trace restriction. Only the algebraic feature-constraint right-hand side is
changed. Consequently `edge_normal_fit_linf` measures the auxiliary normal
coefficient fit, while `feature_target_projection_*` measures a separate
reachable-target modification.

The final staged reduction solves the topology equations and the **projected**
feature equations exactly. Therefore
`feature_target_constraints_exact=0` means that the projected feature target
differs from the originally requested one by more than its tolerance; it does
not mean that the final `C p=d_projected` or `C E=0` invariants were relaxed.

### 5. Direct ambient-gradient Neumann feature backend

For a non-G1 feature shared by patches `a` and `b`, the established backend
first solves the common-gradient law for the two face conormal derivatives.
With `m_i=n_i x t`, `c=n_a.n_b`, and `s=n_b.m_a`, it assembles

```text
d_ma J0_a = (J1_b-c J1_a)/s,
d_mb J0_b = (c J1_b-J1_a)/s.
```

The optional `topology_ambient_gradient_local_svd` backend instead forms the
two candidate world gradients directly,

```text
G_a = grad_Gamma(J0_a) + J1_a n_a,
G_b = grad_Gamma(J0_b) + J1_b n_b,
```

and imposes two moment equations `r_i.(G_a-G_b)=0`, where `r_0,r_1` are a
rigid-motion-covariant orthonormal basis of the plane perpendicular to the
common edge tangent.  The frame starts from the two-normal bisector and uses
its tangent cross product for the second direction.  Existing C0 trace
continuity already supplies `t.(G_a-G_b)=0`, so adding a third Cartesian row
would be redundant.  In coefficient form each row is

```text
[r_i.grad_Gamma(B_a) - r_i.grad_Gamma(B_b)] c_value
    = [r_i.n_b B_b - r_i.n_a B_a] c_normal.
```

This form contains no `1/s` division.  It is pointwise equivalent to the
solved-conormal equations whenever the C0 trace is exact and `s` is nonzero.
For straight planar features (including the L- and U-prism edges), the two
cellwise P0--P3 discretizations have the same row space after restriction to
the C0 topology space.  On a curved feature the moving frames remain
pointwise equivalent, but finite moment test spaces need not be identical;
that case must be judged by reproduction, rank stability, and convergence,
not by entrywise matrix equality.

Both backends deliberately retain the same Neumann data path described in
Section 4: sampled `g_N` is represented in the broken NormalTrace space and
the requested feature target is projected to the reachable value-topology
range.  Thus the new backend isolates the geometric C1 formulation for A/B
testing; it is not yet an analytic-`g_N`, fail-on-unreachable strict mode.

## Trace projector and GMRES matvec

`TopologyTraceProjector3D` constructs

```text
B_z = B_test A0 E
```

from the independent geometric test design, base expansion, and homogeneous
affine map. Topology calls explicitly opt into weighted column-pivoted QR for
fewer than 512 reduced coordinates. At 512 or more coordinates, the sparse
route retains the reduced Gram matrix and uses AMD-ordered
`Eigen::SimplicialLLT`. Ordinary legacy constructor calls still default to
the previous dense-LLT backend.

The factorization is cached in preprocessing, but the projector is part of
every GMRES operator application. A topology matvec performs, schematically,

```text
trace-mass z -> triangular lift -> C0 coefficients -> coefficient Cauchy
             -> Cartesian KFBI bulk solve -> full exterior trace
             -> weighted QR solve (small/medium) or B^T W + sparse-LLT solve
             -> trace-mass residual
```

Thus no QR/Cholesky factorization is repeated in Krylov iteration, and the
KFBI response remains matrix-free, but it is inaccurate to describe the
matvec as only dot products: it includes projector solves. The optional
`projection_matrix()` is a lazy diagnostic materialization and is not used by
the normal operator path.

For Neumann, topology constraints first give the value-jump density

```text
c = c_p + G z,                 dim(z)=K.
```

The final discrete area-mean covector uses the same panel-center design and
weights as the trace projector,

```text
m = B_test^T W 1 / (1^T W 1).
```

With `a=G^T m`, the largest `|a_p|` is the deterministic pivot. The particular
coordinate is corrected in coordinate `p`, and each non-pivot column is
`e_j-(a_j/a_p)e_p`. This gives

```text
c = c_p_mf + G_mf y,           dim(y)=K-1,
m^T c_p_mf=0,                  m^T G_mf=0.
```

Every column of the coordinate map has at most two nonzeros. The trace-mass
projector is then rebuilt from `B_test G_mf`, so its input and output both have
dimension `K-1` and `P_mf 1` is a directly audited roundoff-level quantity.
GMRES solves only

```text
P_mf R_u^+(G_mf y,0) = -P_mf R_u^+(c_p_mf,g_N).
```

There is no appended mean row, constant border column, or lambda unknown in
this branch. The raw known normal jump enters only the fixed right-hand side.

### Dirichlet: analytic known jump and affine unknown normal jump

The topology-affine Dirichlet formulation now uses the boundary condition as
the known value jump itself,

```text
J0 = g_D,
```

and represents only the unknown normal jump in an affine coefficient space,

```text
J1 = c_p + G z.
```

By default, `NormalTrace` stays broken across physical C0 features. The
ordinary within-sheet topology rows are homogeneous, so `c_p=0` and `G` spans
their exact nullspace. The complete jump pair is nevertheless affine:

```text
(J0,J1) = (g_D,0) + (0,G z).
```

Thus the known boundary condition is the particular jump and the Krylov
unknown is only the legal homogeneous normal-jump space. There is no Neumann
mean constraint or additional scalar in the Dirichlet system. An optional
stronger feature model can add nonhomogeneous rows, in which case the same
staged eliminator produces a nonzero `c_p`; that model is described below and
is not the production default.

Let `P_D` denote the normal-trace projector, with its cached trace-mass
coordinate lift included in the meaning of `G z`. The matrix-free Krylov
system is

```text
A_D z = P_D R_n^+(0,G z),
b_D   = -P_D R_n^+(g_D,c_p),
A_D z = b_D.
```

Consequently every GMRES matvec contains only the homogeneous pair
`(J0,J1)=(0,G z)`. The analytic boundary data and the affine particular enter
exactly once through the fixed right-hand side. After GMRES, the physical
normal jump is reconstructed as `c_p+G z`. In the default broken-sheet route
`c_p=0`; in the opt-in feature route it carries the nonhomogeneous feature
data.

The known `g_D` path accepts its value, ambient gradient, and ambient Hessian
from an analytic callback. At each actual crossing, rational surface
derivatives and the local orthonormal frame convert those ambient derivatives
to the value, tangential gradient, and tangential Hessian required by the
Cauchy correction. Curvature and the second-order parameter-to-Cauchy chain
are retained. The topology-affine path therefore does **not** fit `g_D` to a
panel density, and it does not project `g_D` to a reachable feature target.
The sampled `prescribed_value_jump` used by the bulk/restrict interfaces is
the evaluation of the same boundary condition, not an independently fitted
unknown. Residual projection by `P_D` after the PDE solve is still required
to obtain a square reduced GMRES operator; it must not be confused with a fit
of the known jump.

At a sharp feature, `NormalTrace` begins as a broken two-sided space. The new
opt-in affine feature layer assumes that the solution has one common ambient
gradient at the feature. For oriented edge tangent `t`, side normals `n_a`,
`n_b`, co-normals `m_i=n_i x t`, `c=n_a.n_b`, and
`s=n_b.m_a`, this gives

```text
(J1_b-c J1_a)/s = grad_Gamma(g_D)_a . m_a,
(c J1_b-J1_a)/s = grad_Gamma(g_D)_b . m_b.
```

The code assembles these equations directly from the analytic gradient as
cellwise Legendre P0--P3 mortar moments. They form `C_feature c=d_feature` and
are appended to the topology system before affine elimination; there is no
least-squares solve and no reachable-target projection. Endpoint-cell rows
are audited in vertex/T-star blocks and the remaining rows in edge blocks.
Inconsistent edge-tangent derivatives or a nearly degenerate dihedral sine
cause setup to fail instead of silently weakening the equations.

The common-ambient-gradient condition is a regularity assumption, not a
universal corner law.  It therefore remains opt-in through
`KFBIM_3D_DIRICHLET_FEATURE_COUPLING=ambient_gradient_affine_mortar`.
Endpoint-cell moments from all incident feature edges are eliminated together
in Vertex/T-star blocks; these are edge-cell mortar moment rows assigned to a
vertex star, not extra point-value or point-gradient equations at a geometric
vertex.  The remaining moments are eliminated in edge-interior blocks,
matching the Neumann staged schedule.  Before affine elimination, a
global rank-revealing QR certifies `rank(C G_topology)`.  A second QR first
selects a maximal independent subset from all Vertex/T-star candidates; Edge
candidates are then projected off the complete vertex row space and a final
QR selects exactly the remaining rank increment.  Selection uses only the
homogeneous left operator: the analytic right-hand side is not inspected until
the retained row identities are final, after which those entries are copied
unchanged, with no fit or reachable-target projection.  The default
`broken_sheets` mode leaves the one-sided normal jumps independent across
physical features for general nonsmooth-domain solutions.  The legacy
Dirichlet branch retains its previous sample-fit implementation only for A/B
comparison; none of that fitting is used by either analytic topology-affine
feature mode.

For the current U-prism regression, N32 has 256 Vertex/T-star candidates and
no Edge candidates; 160 Vertex rows are retained.  N64 has 512 Vertex/T-star
and 256 Edge candidates, but the Vertex stage already spans the complete
restricted rank 272, so 272 Vertex rows and zero Edge rows are retained.  The
N64 Edge catalog is therefore exercised and reported, but it contributes no
new rank in the topology quotient and must not be retained artificially.

## Direct coefficient Cauchy and restriction

For topology-affine runs, unknown crossing jets are evaluated analytically
from cubic tensor-product spline coefficients. The implementation includes
first and second parameter derivatives, rational NURBS geometry and normal
derivatives, the full second-order inverse chain rule, and compact rows with
at most 16 coefficient entries. GMRES therefore does not refit an unknown
crossing jet from surface samples. In the Dirichlet operator this statement
applies to the unknown `J1` jet; the known `J0=g_D` jet comes directly from
the analytic value/gradient/Hessian callback. A homogeneous Dirichlet matvec
sets `J0` identically to zero and therefore performs neither a callback
evaluation nor a panel fit for that component.

The formal topology-affine restriction baseline uses direct tensor-product
Cartesian covers. For a trace point `x_Gamma`, let `U_ijk^beta` denote the
grid value at one support node after its complete support-to-trace event
sequence has continued it to the requested branch `beta`. The Neumann value
trace uses the 27 nodes of one 3-by-3-by-3 Q2 cover and evaluates

```text
R_Q27^beta U(x_Gamma)
  = sum_{i,j,k=0}^2 L_i^x(x_Gamma) L_j^y(x_Gamma) L_k^z(x_Gamma) U_ijk^beta.
```

The Dirichlet normal trace uses the 64 nodes of one 4-by-4-by-4 Q3 cover and
evaluates its physical outward-normal derivative directly,

```text
R_n,Q64^beta U(x_Gamma)
  = sum_{i,j,k=0}^3
      n_Gamma dot grad(L_i^x L_j^y L_k^z)(x_Gamma) U_ijk^beta.
```

Thus `cover3` and `cover4` mean one Cartesian tensor-product interpolation
cover at each trace point. They do not mean one cube shared by several
off-interface normal queries. The 27 or 64 Cartesian support nodes are not
additional topology trace samples and do not alter the strict
`trace_sample_count > final_reduced_dofs` requirement. The formal selectors are
`KFBIM_3D_NEUMANN_TRACE_RESTRICT=q27_cover3_all_event_cauchy` and
`KFBIM_3D_DIRICHLET_NORMAL_RESTRICT=q64_cover4_all_event_cauchy`. In
particular, the Dirichlet route neither samples normal layers nor recovers a
coefficient `a1/h`; the `1/h` scaling is already present analytically in the
Cartesian Lagrange derivative weights.

Every cover-node-to-trace path is queried, including paths whose endpoint
labels agree. All certified transverse events are applied in strict path
order; therefore an equal-label double crossing contributes both corrections
instead of being skipped. A C0 feature contact whose exact before/after labels
agree is retained as a certified zero-continuation event and contributes no
jump correction. The exact trace endpoint is appended only if the open-path
state has not reached the requested one-sided branch. Unresolved, overlapping,
unclassified smooth tangent, inconsistently oriented, or unordered events
stop setup rather than selecting one root. The nearest Cartesian grid-line
Cauchy anchor is retained only for a unique smooth event on the same G1 sheet;
multi-event and C0-owner cases use their exact event centers.

The reduced-coefficient direct-cover pipeline does not construct the legacy
panel-centred Cauchy maps during setup. Those maps remain mandatory for the
legacy/standard trace routes, and any accidental call to them from a
direct-cover-only run fails explicitly. Independent crossing-centred direct
row rank certification is still performed. Intersection searches, crossing
ownership, coefficient rows, cover indices, and interpolation/derivative
weights are prepared outside GMRES; coefficient row products, the bulk solve,
and the trace-projector solve remain inside a matvec.

`shared_q10_cubic_gridline_cauchy` remains available only as an explicitly
selected A/B/legacy comparison. It takes three samples on each side at
`signed_distance/h = +/-{0.5, 0.75, 1.5}`, uses one shared ten-node
complete-P2 Cartesian stencil for each three-query side, and fits all six
continued values to one cubic normal profile. Its value and normal outputs are
the fitted `a0` and `a1/h`. It is not the topology-affine default and is never
an implicit fallback when Q27/Q64 event certification fails.

## CSV semantics

`neumann_results.csv` and `dirichlet_normal_results.csv` contain both algebraic
and physical diagnostics. Important interpretations are:

- `dofs` is the final density-coordinate count. For mean-free Neumann it is
  exactly the GMRES operator dimension `K-1`; no hidden scalar is appended.
- `topology_full_coordinates` is the broken patch coefficient count,
  `topology_base_coordinates` is the exact full-edge-C0 base count,
  `topology_pre_mean_coordinates` is the dimension after topology/feature
  constraints, and `topology_reduced_coordinates` is one smaller after the
  final mean pivot. `topology_constraint_rank` continues to describe only the
  topology/feature constraints; the separately reported mean reduction has
  rank one.
- `topology_block_count` counts final sequential elimination records after
  support-component merging, not the number of original physical blocks.
- `trace_restrict_mode` records the route actually used. The formal defaults
  are `q27_cover3_all_event_cauchy` in `neumann_results.csv` and
  `q64_cover4_all_event_cauchy` in `dirichlet_normal_results.csv`; a Q10 value
  identifies an explicit comparison run rather than the production baseline.
- `converged` is the GMRES algebraic flag. It does not certify the boundary
  conditions or gauge.
- `mean_free_*` records the selected pivot, its observability/coordinate
  condition, the particular and homogeneous mean certificates, and
  `||P_mf 1||_inf`. `lagrange_multiplier` and `border_column_linf` are `nan`
  because those objects do not exist in this branch.
- `density_weighted_mean` is the panel-area mean actually eliminated by the
  discrete system. `mean_free_intrinsic_mean` recomputes the final density
  with the independent native-Gauss mass row; it is a quadrature-consistency
  diagnostic and is not a second imposed constraint.
- Neumann `physical_converged` additionally requires the operator residual,
  exterior condition, projected-trace closure, mean-constraint residual,
  weighted density mean, and active edge-constraint residual all to be no
  larger than `10*gmres_tolerance`.
- `feature_target_constraints_exact` compares requested and projected feature
  targets as explained above. Reachability of the projected target is
  recorded separately by `feature_target_reachable_certification_linf`.
- For topology-affine Dirichlet, `known_value_jump=analytic_boundary_jet`
  certifies that crossing `J0` jets came from the value/gradient/Hessian
  callback. `edge_jump_jet=ambient_gradient_affine_mortar` identifies the
  nonhomogeneous feature law. `edge_constraint_rows` is the overcomplete
  candidate count, `edge_constraint_rank` is the retained unisolvent count,
  and their difference is the discarded count. The separate
  `feature_vertex_residual_linf` and `feature_edge_residual_linf` values audit
  the final `c_p+G z` against retained endpoint/T-star and edge-interior rows.
  `feature_discarded_residual_linf` audits the original analytic equations that
  were not retained; it is reported rather than fitted or projected away.
- Dirichlet GMRES controls only the projected normal trace. To expose a
  possible unrepresented component, let `t_n` be the complete sampled
  exterior normal trace, `P_D t_n` its reduced trace-mass coordinates,
  `L_h` the homogeneous C0 lift, and `B_D` the full normal-trace design. The
  reported leakage is

  ```text
  trace_projection_leakage_relative_l2
    = ||W^(1/2) (t_n-B_D L_h P_D t_n)||_2
      / max(||W^(1/2) t_n||_2, machine_epsilon).
  ```

  Thus a small projected operator residual with large leakage is not accepted
  as evidence that the complete exterior normal trace is controlled.
- `density_order_*`, `interior_order_*`, and `exterior_bulk_order_*` are `nan`
  for a single-level run and are not convergence estimates.

`topology_block_records.csv` gives one row per final elimination record:
kind, merged source keys, row/support counts, local rank/nullity, remaining
coordinates, residual audits, and retry count. Keys and merged source-key
fields use RFC-4180 quoting; `source_keys` are pipe-joined within that single
quoted field.

## Previously recorded Q10 comparison baseline

The results in this section were recorded with the explicit
`shared_q10_cubic_gridline_cauchy` route, before Q27/Q64 became the formal
defaults. The older rows also predate the analytic Dirichlet `J0=g_D`,
`J1=c_p+G z` change described above. They remain useful as historical Neumann
and Q10 A/B evidence, but they do not validate either direct-cover default or
the new Dirichlet affine operator. Formal-default verification is listed
separately under "Items not yet verified."

A clean `build-topology-final` configuration built both
`kfbi_topology_affine_exterior_trace_3d` and
`kfbi_native_c0_exterior_trace_3d`. The following thirteen executables were run
manually and all exited successfully:

1. `smooth_t_topology_3d_test`
2. `native_nurbs_density_space_3d_test`
3. `restrict_crossing_selector_3d_test`
4. `shared_quadratic_restrict_3d_test`
5. `topology_affine_reduction_3d_test`
6. `topology_density_constraints_3d_test`
7. `topology_reachable_target_projection_3d_test`
8. `topology_trace_projector_3d_test`
9. `direct_coefficient_cauchy_3d_test`
10. `topology_mean_free_reduction_3d_test`
11. `grid_edge_event_3d_test`
12. `restrict_crossing_path_state_3d_test`
13. `restrict_crossing_feature_side_classifier_3d_test`

These are CMake executable targets; the current CMake files do not register
them with `add_test`, so “all thirteen tests passed” here means direct executable
runs rather than one `ctest` invocation.

The newer `tensor_product_cover_restrict_3d_test` target checks Q27/Q64 tensor
sizes and exact value/normal-derivative reproduction for every tensor monomial
through Q2/Q3, respectively. It is not part of the historical thirteen-run
claim above; one formal-default end-to-end smoke result is recorded below.

### Historical Q10 N=32 Neumann smoke results

The following manufactured-solution comparison runs used N=32
(`h=0.09375`), `neumann_only`, the explicit Q10 restrict,
`g1_nearest`, automatically selected coefficient density, panel-center trace
tests, and `gmres_tolerance=2e-10`.

| Geometry | ncoef/dir | raw / base -> pre-mean -> final | constraint rows/rank | GMRES dimension | iterations | GMRES rel. residual | operator residual inf | exterior condition inf | interior error inf | `P_mf 1` inf | physical converged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cylinder | 4 | 256 / 144 -> 32 -> 31 | 192 / 112 | 31 | 16 | 1.704e-10 | 4.663e-12 | 1.282e-3 | 1.568e-3 | 5.690e-16 | 0 |
| L-prism | 5 | 300 / 216 -> 50 -> 49 | 448 / 166 | 49 | 19 | 6.377e-11 | 1.934e-12 | 4.069e-6 | 2.434e-5 | 1.360e-15 | 0 |

Additional diagnostics relevant to interpretation are:

| Geometry | normal fit inf | feature projection relative L2 | reachable certificate inf | `C p-d` inf | `C E` max | mean particular / homogeneous | panel density mean | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| cylinder | 4.900e-3 | 2.757e-15 | 2.429e-17 | 9.159e-16 | 3.608e-15 | 8.674e-19 / 1.041e-17 | 7.059e-18 | 15.03 |
| L-prism | 2.429e-7 | 3.606e-6 | 1.546e-17 | 6.245e-16 | 2.274e-15 | 0 / 7.030e-18 | -7.016e-17 | 23.86 |

Both Krylov solves set `converged=1`, but both set
`physical_converged=0`. With this tolerance the physical threshold is
`2e-9`; the exterior zero-trace condition is `1.282e-3` for the cylinder and
`4.069e-6` for the L-prism. The final panel-area density means and projected
operator residuals are at roundoff/solver tolerance. These runs therefore
verify execution, exact mean elimination, algebraic closure, and
GMRES convergence, but they are **not** numerical acceptance results for the
full boundary-value problem and must not be reported as such.

The source CSVs are:

- `output/kfbi_topology_affine_3d/tac_q10/nm_pc_tm_mf/topology_affine/abmfc321dcb/neumann_results.csv`
- `output/kfbi_topology_affine_3d/tac_q10/nm_pc_tm_mf/topology_affine/abmfl321dlb/neumann_results.csv`

### Historical Dirichlet and rigid-transform smoke results

The analytic, broken-sheet Dirichlet route has completed a fresh explicit-Q10
L-prism N=32 smoke run. It used `ncoef=5`, 1050 positive-weight panel-center
trace samples, and 260 final GMRES coordinates (margin 790, ratio 4.038). The
known jump was reported as `analytic_boundary_jet`, feature coupling as
`broken_sheets`, and no feature target rows or fit were present. GMRES
converged in 18 iterations with relative residual `1.021e-10` and projected
operator residual `6.070e-12`; the projector closure was `1.486e-15`. The
interior infinity error was `4.671e-4`, while the complete unprojected
exterior-normal trace remained `1.568e-4`. Its relative projection leakage
was approximately one, so this is an algebraic/route smoke result, not a
physical accuracy acceptance result.

Before choosing the broken-sheet default, the opt-in common-ambient-gradient
feature system was also exercised at N=16 and N=32. It failed exact affine
elimination with scaled incompatibilities `1.133e-5` and `2.266e-6`,
respectively. No least-squares relaxation was used. This demonstrates that
the extra corner regularity assumption cannot be made a mandatory production
constraint merely because the prescribed manufactured boundary data are
analytic.

Before the analytic affine Dirichlet change, the L-prism N=32
Dirichlet-only branch was executed. It used 260 reduced
normal-density coordinates, converged in 18 iterations with relative residual
`1.022e-10`, and had full operator residual `6.068e-12`. The topology system
had 32 rows, rank 20, `||C E||=1.027e-15`, projector `PB-I` error
`3.309e-15`, and full-trace/projector closure `1.245e-15`. Its exterior-normal
condition was `1.575e-4`. Besides being only an algebraic/route smoke result,
this run used the superseded Dirichlet construction and must not be quoted as
a result for the analytic `g_D`/affine-`J1` implementation.

Under a rigid transform consisting of a 17-degree rotation about axis
`(1,2,3)` followed by the catalogued translation, the mean-free L-prism case
retained 50 pre-mean and 49 final coordinates and again used 19 Neumann GMRES
iterations. The full operator residual was `1.957e-12`, `P_mf 1` was
`1.957e-15`, and the panel-area density mean was `3.330e-17`. The exterior
condition changed from `4.069e-6` to `3.586e-6` and the interior infinity
error from `2.434e-5` to `2.037e-5`, consistent
with Cartesian-grid phase sensitivity but no Krylov/topology instability.

## Implemented and unimplemented specification fallbacks

For the ordered strategy in specification section 9.3:

| Level | Status |
|---:|---|
| 1. Merge neighboring blocks sharing active columns | Implemented with current-`C E` same-stage closure and zero-tolerance incompatibility retry |
| 2. Enlarge vertex support/star layers | Not automated; assembly currently uses the configured one-layer star |
| 3. Project multiple known-normal jets into a common reachable space | Implemented for the Neumann feature target as the component-wise `C_feature E_t` range projection described above |
| 4. Knot insertion/additional density modes | Not implemented as an incompatibility fallback |
| 5. Priority relaxation of transverse feature constraints | Not implemented |
| 6. Explicit opt-in weighted least squares with `constraints_exact=false` | Not implemented in this C++ production route; it fails instead of returning an approximate affine reduction |

There is no silent global LSMR or weighted least-squares fallback. After the
implemented support merge and feature-target projection, unresolved
incompatibility is fail-fast and reports the unattempted upstream operations.

## Formal Q27/Q64 U-prism smoke result

A fresh untransformed U-prism run at `N=32` (`h=0.09375`) exercised both
formal defaults in one process. The output is under
`output/kfbi_topology_affine_3d/q27_c3/d_q64_c4/nm_pc_tm_mf/topology_affine/qsm2`.
No segment or G1-sheet fallback was used, and both coefficient systems
converged algebraically.

| Problem | Restrict | GMRES it | GMRES relative residual | operator residual Linf | condition-trace Linf | density Linf | interior Linf | trace samples / final DOFs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Neumann | `q27_cover3_all_event_cauchy` | 15 | 6.489e-16 | 2.429e-17 | 3.077e-5 | 6.515e-5 | 5.986e-5 | 832 / 15 |
| Dirichlet | `q64_cover4_all_event_cauchy` | 23 | 7.836e-11 | 6.334e-12 | 8.536e-6 | 1.538e-4 | 1.726e-5 | 832 / 224 |

The oversampling margins are therefore 817 and 608, respectively. Both rows
have `physical_converged=0`: their finite-grid condition-trace errors exceed
the deliberately strict approximately `2e-9` physical acceptance threshold,
even though GMRES converged. This one level is an integration smoke test, not
a convergence study.

## Items not yet verified

- No N=64/128 formal-default runs or 32--128 convergence orders are recorded
  for this code state.
- Only one L-prism rigid transform at N=32 has been run; this is not the full
  multi-pose or multi-level stability suite required by the specification.
- The analytic topology-affine Dirichlet route has focused coverage and one
  fresh explicit-Q10 L-prism N=32 end-to-end smoke run, including analytic
  known-J0 jets, broken-sheet affine reduction, all-event path continuation,
  and strict trace oversampling. The formal Q64 normal-trace route additionally
  has the U-prism N=32 smoke result above, but no recorded 32--128 convergence
  or rigid-transform suite.
- Sphere, ellipsoid, torus, flower, and other catalog geometries have not been
  numerically validated with the formal direct-cover baseline.
- The sparse `K>=512` projector backend has targeted unit coverage, but no
  end-to-end topology-affine PDE case at that size is recorded here.
- Component locality and memory scaling are input dependent; the two N=32
  cases already collapse to one support component, so no small-component
  scalability claim follows from them.
- Because the physical acceptance flag is false in both current smoke runs,
  restrict accuracy, gauge behavior, and the raw-known-data versus fitted
  feature-target split still require numerical refinement.

## Reference scope

The specification names Python modules that are not present in this
repository or its Git object database. In addition,
`docs/kfbi_general_cap_exterior_trace.py` imports external absolute
`/mnt/data` paths. Consequently no unavailable Python implementation is a
runtime dependency, and no final Python/C++ equality result is claimed. The
legacy C++ executable and small dense unit-test oracles are the currently
available comparison references.
