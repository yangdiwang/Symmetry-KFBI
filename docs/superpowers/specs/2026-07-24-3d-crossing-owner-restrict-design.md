# 3D Crossing-Owner Normal Restrict Design

## Goal

Add a retained, selectable 3D exterior-normal restrict route that tests
whether rotation-induced GMRES stagnation is caused by assigning a
wrong-side Cartesian interpolation node to the target surface DOF's Cauchy
polynomial when the segment from the normal-layer query to that node actually
crosses a different non-G1 surface branch.

The experiment is a strict single-variable comparison. It changes only the
Cauchy owner used for a subset of wrong-side `4x4x4` support nodes.

## Fixed Parts of the Algorithm

The new route must keep all of the following identical to
`JointTricubicCauchy`:

- the surface DOFs and their normals;
- the four normal layers on each side;
- all `4x4x4` Cartesian support node IDs and tensor-product weights;
- the degree-three local harmonic Cauchy fits and their neighborhoods;
- spread correction, FFT bulk solve, GMRES parameters, and right-hand side;
- current trace recovery and the final physical `1/h` normal scaling.

The legacy route remains the production default. The new route is retained as
an explicit algorithm option and is exercised by the restrict causal probe.

## Routes

Extend the mode enum to:

```cpp
enum class ExteriorNormalRestrictMode3D {
    JointTricubicCauchy,
    JointTricubicCrossingOwner,
    ExteriorOnlyHarmonicCubic
};
```

`JointTricubicCauchy` remains bit-for-bit behaviorally unchanged.

`JointTricubicCrossingOwner` uses the same grid interpolation and changes only
the correction

```text
sign * weight * P_target(grid_node)
```

to

```text
sign * weight * P_owner(grid_node)
```

when one reliable NURBS crossing proves that the support node lies across a
non-G1 branch owned by another surface DOF.

`ExteriorOnlyHarmonicCubic` remains the existing independent diagnostic
reference.

## Crossing-Owner Rule

For every normal-layer query and every Cartesian support node:

1. If the support node has the desired domain label, add no Cauchy
   correction and perform no owner rerouting.
2. If the support node has the wrong domain label, intersect the segment from
   the query point to the support node with the native NURBS surface.
3. If there is exactly one transverse, resolved crossing:
   - map its exact `(patch,u,v)` to an owner DOF using the existing
     `parameter_dof_candidates_2x2()` rule and Euclidean-distance tie break;
   - if the crossing patch belongs to the target patch's G1-smooth component,
     retain the target DOF;
   - otherwise use the crossing owner DOF.
4. If the result is empty, multiple, tangential, overlapping, unresolved, or
   lies on a non-G1 edge with non-unique ownership, retain the legacy target
   DOF and record the fallback reason.

This first experiment intentionally does not define a new multiple-crossing
transport rule. The previously measured dominant event is a reliable single
crossing on a foreign non-G1 patch; ambiguous cases must not be guessed.

## Precomputation and Application

All arbitrary NURBS segment intersections occur once while trace templates
are built. No geometry query may occur inside a GMRES operator application.

Each trace sample stores the unchanged grid IDs and interpolation weights plus
two precomputed correction representations:

- the legacy correction evaluated with the target center;
- correction terms grouped by the selected owner DOF.

A grouped correction term contains an owner DOF ID and the accumulated
weighted harmonic-basis evaluation in that owner's local frame. Applying the
new route remains linear in the jump data:

```cpp
value += term.evaluation.dot(
    field.coefficients.row(term.owner_dof).transpose());
```

## Diagnostics

Record per target DOF and in aggregate:

- wrong-side support count and absolute interpolation weight;
- same-target/G1 single-crossing count and absolute weight;
- rerouted non-G1 single-crossing count and absolute weight;
- no-root, multiple-root, tangent/overlap, unresolved, and ambiguous-edge
  fallback counts and absolute weights.

For every rerouted support term, retain enough data for audit:

- target DOF and patch;
- side and normal layer;
- grid node and interpolation weight;
- crossing patch, `(u,v)`, segment parameter, residual, and transversality;
- selected owner DOF.

Write these beneath the existing
`output/dirichlet_normal_restrict_causal_probe_3d` directory without changing
legacy application output.

## Numerical Comparison

Run the L-prism cases:

- `baseline`;
- `rot_axis123_17deg`;
- `rot_axis123_17deg_t_xyz_1`;

at `N=32,64`. Compare the legacy and crossing-owner routes with identical
physical and deterministic common right-hand sides. Retain the existing
exterior-only route as context, not as part of the strict A/B.

Report:

- full GMRES residual histories and iteration counts;
- geometric contraction over iterations 10--30;
- worst five-step contraction;
- exact-grid restrict `Linf` and weighted RMS errors;
- exact-density equation residuals;
- physical interior `Linf` error and the `32 -> 64` observed order;
- setup and solve times.

The hypothesis is supported if the crossing-owner route materially shortens
the rotated residual plateaus while leaving baseline behavior and physical
accuracy essentially unchanged. A lack of improvement rejects reliable
single-crossing foreign-owner assignment as the primary cause, but does not
rule out multiple-crossing or general interpolation-support effects.

## Tests

Focused tests must verify:

- same/G1 crossings retain the target owner;
- a reliable single crossing on a non-G1 patch selects that patch's `2x2`
  parameter-owned DOF;
- ambiguous and non-single-root results retain the target owner;
- grouped owner correction equals the legacy correction when all terms retain
  the target owner;
- operator application performs no geometry queries;
- the legacy route and default application output remain unchanged.

All existing 3D NURBS intersection, transformation, restrict, and application
tests must continue to pass.
