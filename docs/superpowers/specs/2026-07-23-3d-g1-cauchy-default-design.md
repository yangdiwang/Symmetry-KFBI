# 3D G1-Nearest Cauchy Default Design

## Goal

Allow Cauchy samples to cross native NURBS G1 seams while prohibiting every
crossing of a non-G1 feature edge.

## Selection rule

For each center DOF, start from its owning patch and add at least the direct
G1-neighbor ring from `NativeNurbsSurface3D::smooth_neighbors`. Continue a
breadth-first expansion along G1 edges only when the requested sample count
is not yet available. Sort all admitted DOFs by three-dimensional Euclidean
distance, using DOF ID as the deterministic tie breaker, and return the
nearest available `min(requested, admitted)` samples.

The selector never traverses `topological_patch_neighbors`, so it cannot
cross a non-G1 edge. Explicit periodic seams already registered in
`smooth_neighbors` are ordinary G1 edges and are crossed naturally.

## Policies

- `g1_nearest`: new default and production experiment.
- `same_patch`: retained as the stricter diagnostic used in the previous run.
- `topological_nearest`: retained as the diagnostic that may cross non-G1
  edges, specifically for the requested Neumann comparison.
- `balanced_patches`: retained unchanged.

Cauchy degree, requested `48/28` samples, tricubic restrict, cubic normal fit,
manufactured data, and GMRES tolerances do not change.

## Verification

Tests must prove that `g1_nearest` crosses the torus periodic G1 seam and an
L-prism cap G1 seam, while remaining on one side patch at an L-prism non-G1
edge and remaining on one smooth cylinder sheet. The default application
smoke must report `g1_nearest` and `N=32`.

Numerical runs use `N=32,64,128,256`. The production run uses `g1_nearest`;
the comparison run uses `topological_nearest`. Report interior maximum error,
adjacent order, and GMRES iterations, with special attention to Neumann.
