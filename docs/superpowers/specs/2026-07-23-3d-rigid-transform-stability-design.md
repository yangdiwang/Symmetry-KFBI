# 3D Dirichlet Rigid-Transform Stability Design

## Goal

Measure whether the current main 3D Dirichlet normal-jump formulation depends
on the placement or orientation of the L-prism relative to the Cartesian grid.
The continuous geometry and manufactured harmonic problem are transformed
together, so differences between cases measure numerical grid sensitivity
rather than a change of boundary data.

The study uses only:

- the L-prism geometry;
- the current `g1_nearest` Cauchy policy;
- degree-three local Cauchy fitting;
- tricubic grid restriction with cubic normal fitting;
- the Dirichlet exterior-zero-normal-trace formulation;
- `N=32,64,128`.

The existing default 3D app behavior remains available and unchanged.

## Rigid Transform

Use the proper rigid transform

```text
x' = c + Q (x - c) + t
x  = c + Q^T (x' - c - t)
```

where `Q` is a rotation matrix, `c` is the rotation center, and `t` is the
translation applied after rotation.

Add a small reusable `RigidTransform3D` value type with forward-point,
inverse-point, and forward-vector operations. Reject non-finite data and
rotation matrices that are not proper orthogonal matrices within numerical
tolerance.

Transform a native NURBS surface by rebuilding every patch from:

- the original `u` and `v` bases;
- the original weights;
- forward-transformed control points.

Keep patch order, names, UV coordinates, smooth-neighbor topology,
topological neighbors, geometric connections, component IDs, and expected
area unchanged. Wrap `exact_inside` so it evaluates the original predicate at
the inverse-transformed query point.

## Manufactured Harmonic Solution

Let `u0` and `grad(u0)` be the existing non-polynomial manufactured harmonic
solution. For each rigid-transform case, evaluate

```text
u_T(x')       = u0(inverse(x'))
grad u_T(x')  = Q * grad u0(inverse(x'))
```

The boundary value is `u_T` at each transformed surface DOF. The exact normal
jump is `grad u_T` dotted with the normal computed from the transformed native
NURBS patch. Interior bulk error uses the same transformed exact solution.

## Study Cases

Run these eight cases:

| ID | Rotation | Translation |
| --- | --- | --- |
| `baseline` | identity | `(0, 0, 0)` |
| `tx_p0137` | identity | `(0.137, 0, 0)` |
| `ty_m0083` | identity | `(0, -0.083, 0)` |
| `tz_p0061` | identity | `(0, 0, 0.061)` |
| `t_xyz_1` | identity | `(0.137, -0.083, 0.061)` |
| `t_xyz_2` | identity | `(-0.109, 0.151, -0.047)` |
| `rot_axis123_17deg` | axis `(1,2,3)/sqrt(14)`, `17` degrees | zero |
| `rot_axis123_17deg_t_xyz_1` | same rotation | `(0.137, -0.083, 0.061)` |

The rotation center is the midpoint of the L-prism reentrant edge:
`(0.07, -0.07, 0.02)`.

Before solving, verify that every transformed NURBS control point is finite
and remains strictly inside the fixed box `[-1.5,1.5]^3`.

## App Integration

Extend `neumann_exterior_zero_trace_3d` with an explicit rigid-study mode.
Legacy positional invocations continue to run their existing geometry and
formulation selections.

Rigid-study mode:

- fixes the geometry to the L-prism;
- fixes the levels to the supplied `N` values, with the acceptance run using
  `32 64 128`;
- executes all eight cases;
- builds geometry, triangulation, Cartesian classification, surface DOFs,
  Cauchy stencils, and the harmonic-jet pipeline independently for each case;
- runs only the Dirichlet normal-jump solve;
- uses GMRES relative tolerance `2e-10` and the existing configurable maximum,
  defaulting to 80;
- writes accumulated output after every completed solve and stops immediately
  if a solve does not converge.

This mode must not silently use an alternate stencil policy or stencil count.
It validates that the active configuration is the current main scheme:
`g1_nearest`, 48 value samples, and 28 normal samples.

## Results

Write study data under:

```text
output/dirichlet_rigid_transform_stability_3d
```

The primary CSV contains one row per case and grid level, including:

- case ID, `N`, and `h`;
- rotation axis, angle, center, and translation;
- interior `Linf` error;
- observed order relative to the preceding level of the same case;
- GMRES convergence flag, iteration count, and final relative residual;
- solve time and total case time;
- Cauchy condition statistics;
- label mismatch and crossing-safety diagnostics;
- error and iteration ratios relative to the baseline at the same `N`.

Write a second acceptance summary with one row per transform case.

## Acceptance

A case passes when:

- every solve converges within 80 GMRES iterations;
- the interior `Linf` error decreases from `N=32` to `64` to `128`;
- the `N=64` to `128` observed order is at least `1.8`;
- its error at every level is no more than three times the baseline error at
  that level;
- there are no exact-label mismatches, unsafe label-changing crossings, gap
  ownership fallbacks, endpoint ownership fallbacks, or triangle ownership
  fallbacks.

The program reports all individual criteria. A failed criterion makes the
overall study fail, but completed rows remain on disk for diagnosis.

## Verification

Add focused tests for:

- forward/inverse point round trips;
- vector and gradient rotation;
- transformed NURBS evaluation matching the forward transform of the original
  patch at identical UV coordinates;
- preservation of weights, bases, topology, area, and patch metadata;
- transformed `exact_inside` equivalence;
- rejection of an invalid rotation matrix;
- construction and metadata of all eight study cases;
- backward compatibility of the existing command-line mode;
- a small rigid-study smoke run that executes only Dirichlet.

The final numerical acceptance run is the full eight-case
`N=32,64,128` study.
