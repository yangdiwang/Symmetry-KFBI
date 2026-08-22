# KFBI 3D Baseline Integration Policy

## Baseline

The immutable reference for this integration is commit
`f68683eab514a6b2d2b9ad6fc9dea13204dc9d52`, tagged as
`baseline/topology-affine-v1`.

The integration branch must preserve the baseline behavior unless a route is
selected explicitly.  In particular, the production topology target keeps:

- the canonical all-event Cartesian-edge catalog and shared event identifiers;
- fail-closed complete-root-set certification, including same-label multiple
  crossings and feature-owner classification;
- direct-coefficient Cauchy jets and shared-Q10 restriction;
- topology-affine density coordinates;
- mean-free pivot elimination for Neumann problems;
- analytic `J0 = g_D` and affine `J1 = c_p + G z` for Dirichlet problems;
- broken physical sheets at C0 Dirichlet features by default; and
- strictly more positive-weight trace samples than final reduced coordinates.

## Imported remote work

The work merged from `origin/main` at `9ec7a24d0488df690f0a2db3e1bc7436a4ceb03b`
is supplementary.  Geometry certificates, mapped-candidate preprocessing,
phase profiling, benchmarks, diagnostics, studies, and audit scripts may be
used directly when they do not change the baseline operator.

The following algorithms remain legacy or experimental until they pass the
baseline validation gates:

- unique crossing-owner restriction;
- non-G1 mass-projected panel-value density;
- edge-augmented and two-level fitted Cauchy maps;
- augmented Neumann solves with a Lagrange multiplier; and
- sample-fitted Dirichlet jump data.

No experimental route may be used as an implicit fallback from an uncertified
baseline event.  It must be selected explicitly and reported in result data.

Supplementary app-level algorithms are built only when
`KFBIM_BUILD_EXPERIMENTAL_3D=ON`; the default is `OFF`.  The historical remote
solver route test additionally requires
`KFBIM_BUILD_REMOTE_SOLVER_ROUTE_TEST=ON`.  That second switch remains off
until the test is adapted to the baseline solver API; its source is retained
as reference evidence, not as a production validation target.

## Merge invariants

Remote candidate acceleration may implement the geometry query, but its output
must reproduce the complete baseline physical-event catalog.  A filtered
first-hit or unique-root query cannot feed all-event spread or restriction.

The merged implementation must preserve:

1. all certified events, including two or three roots on one grid edge;
2. event identifiers under reversed queries, with reversed order and sign;
3. all incident physical-root owners required for G1-sheet selection;
4. targeted retry for every uncertified root set, not only label-changing
   edges;
5. a hard setup failure for unresolved, ambiguous, or unclassified events;
6. coefficient-space topology constraints before the Neumann mean reduction;
7. independent full-trace diagnostics in addition to projected residuals; and
8. `trace_sample_count > final_reduced_dofs`.

## Promotion gates

An experimental route may become a production candidate only after it:

- passes the focused geometry, event, topology, projector, and Cauchy tests;
- reproduces baseline geometry catalogs under rigid transformations;
- converges algebraically for the seven-geometry N32 matrix;
- completes the required N32/N64/N128 baseline and rigid sequences;
- reports full, unprojected boundary traces and projection leakage; and
- does not regress density, interior, exterior, or near-feature errors under
  the documented acceptance tolerances.

The integration merge itself does not promote an experimental route.
