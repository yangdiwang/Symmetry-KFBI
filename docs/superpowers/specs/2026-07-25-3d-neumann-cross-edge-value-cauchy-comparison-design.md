# 3D Neumann Two-Level Shared-Edge Cauchy Reconstruction Design

## Goal

Determine whether the unstable L-prism Neumann value-trace solve can be
stabilized by explicitly reconstructing one shared value-jump field on every
native non-G1 edge and then using those reconstructed edge values as virtual
samples in the ordinary surface-centered Cauchy fits.

The comparison changes only how value data near a non-G1 edge enter the local
Cauchy reconstruction. It keeps the crossing-owner value restrict, cubic
harmonic space, tricubic Cartesian interpolation, cubic normal-line fit,
compatibility correction, bordered Neumann equation, GMRES tolerance, and
manufactured solution unchanged.

## Hypothesis

The Neumann unknown is the value jump. For a smooth ambient harmonic solution,
its trace has one shared scalar value on a geometric edge even though the two
incident faces have different outward normals and therefore different normal
derivatives.

Directly enlarging a surface-centered stencil across the edge supplies
two-face information but does not explicitly construct that shared edge value.
The proposed two-level method first reconstructs the edge value from both
faces' Cauchy data, then inserts the same reconstructed scalar into every
nearby surface-centered fit. This weakly enforces cross-edge value continuity
without imposing equality of the two face-normal derivatives.

## Compared Routes

Every route uses `JointTricubicCrossingOwner`. The second-level normal sample
pool remains the existing G1-nearest pool in every route so that the comparison
isolates value-density treatment.

| Route | Second-level value conditions | Second-level normal conditions |
|---|---|---|
| `g1_value_g1_normal` | Existing 48 G1-nearest values | Existing 28 G1-nearest normals |
| `direct_cross_face_value` | 48 topology-controlled, two-face-balanced values | Existing 28 G1-nearest normals |
| `edge_reconstructed_value` | Existing 48 G1-nearest values plus reconstructed shared-edge values | Existing 28 G1-nearest normals |

The direct-cross-face route is a diagnostic control, not the primary
candidate. The shared-edge route is the primary candidate.

For the direct control, a center strictly farther than `2h` from every
incident non-G1 edge returns the existing G1 value IDs in exactly the existing
order. Inside the closed `2h` band, collect every non-G1 connection incident
to the center's transitive G1 component whose curve distance is at most `2h`,
then deduplicate the G1 sectors on both sides of those connections. A regular
edge has two sectors and therefore receives a `24/24` split. For \(S\)
sectors near a feature vertex, allocate \(\lfloor48/S\rfloor\) samples to each;
give the remainder first to the center's sector and then by smallest sector
patch ID. Deduplicate DOF IDs before selection. If a sector exhausts its
candidates, fill from the globally nearest remaining candidate in the
admitted sectors. Always include the center DOF. Its 28 normal IDs always
remain the existing G1 IDs.

## Native Shared-Edge Points

The authoritative edges are the non-G1 entries in
`NativeNurbsSurface3D::geometric_connections`. Patch-edge intervals, including
partial one-to-many L-prism connections, must be preserved. Triangulator
feature-edge chords are not used to define the reconstruction points.

For each native non-G1 edge interval:

1. Estimate its physical NURBS-curve length \(L_e\) with the existing
   derivative-based composite-trapezoid rule using 64 parameter samples on
   the interval.
2. Set the number of edge cells to
   \(m_e=\max(1,\lceil L_e/h\rceil)\).
3. Place one point at each cell center by uniform subdivision of the native
   edge parameter interval: for zero-based cell \(r\), use fraction
   \((r+1/2)/m_e\).
4. Do not place points at interval endpoints. This avoids duplicate virtual
   values at feature vertices and at adjacent interval boundaries.
5. Store the physical point, native parameter, edge tangent, the two incident
   G1 face sectors, and a deterministic edge/point ID.

The first connection interval is canonical. For fraction \(s\), use

\[
t_1(s)=a_1+s(b_1-a_1), \qquad
t_2(s)=a_2+\begin{cases}
s(b_2-a_2), & \text{if not reversed},\\
(1-s)(b_2-a_2), & \text{if reversed}.
\end{cases}
\]

Evaluate both incident NURBS patches at these edge parameters. Require the two
physical points to agree within
`max(1e-12*model_diameter,1e-14)`. Differentiate with respect to the common
fraction \(s\), including the reversed sign, require finite nonzero tangents,
and require their mapped unit-vector dot product to be at least `1-1e-10`.
Use the first interval's point and mapped tangent after this audit, while
retaining both native parameters for diagnostics.

The points are global and shared: an edge value is reconstructed once per
operator evaluation and reused by every nearby surface-centered fit on either
face.

## First-Level Edge Cauchy Fit

For a virtual edge point \(x_e\), construct a deterministic orthonormal frame.
Let \(t_e\) be the normalized tangent oriented by the first native edge
interval. Project the incident-normal sum off that tangent and normalize it:

\[
b_e=\operatorname{normalize}\left(
 n_1+n_2-((n_1+n_2)\cdot t_e)t_e\right).
\]

The frame axes are:

- first axis \(t_e\);
- second axis \(b_e\times t_e\);
- third axis \(b_e\).

If the normal bisector is numerically degenerate, use the lower-ID incident
sector's outward normal, projected off \(t_e\), as the deterministic
third-axis seed. Fail the route if the two incident G1 sectors are not
distinct or the fallback is also degenerate. The full cubic harmonic space is
rotationally invariant; the frame is chosen for stable, repeatable coordinates
rather than to add a physical constraint.

Use local coordinates

\[
\xi=(x-x_e)/h
\]

and one degree-three harmonic polynomial

\[
q_e(\xi)=\sum_{k=1}^{16}c_{e,k}H_k(\xi), \qquad \Delta q_e=0.
\]

Select conditions symmetrically from the two incident G1 sectors:

- 24 value-jump samples per sector, 48 total;
- 14 normal-jump samples per sector, 28 total.

Rank candidates inside each sector by Euclidean distance to \(x_e\), then by
deterministic DOF ID. Never admit a physically close but topologically
unrelated patch.

The weighted least-squares rows are

\[
q_e(\xi_j)\simeq \mu_j
\]

for value data and

\[
n_j^{(e)}\cdot\nabla_\xi q_e(\xi_j)\simeq h\,\eta_j
\]

for normal data. Here \(n_j^{(e)}\) is each sample's actual face normal
expressed in the edge frame. The two face normals are never averaged in the
equations.

Use the current value/normal distance weights and SVD cutoff unchanged. The
reconstructed virtual value is

\[
\mu_e=q_e(0)=E_e^v\mu+E_e^n\eta.
\]

The SVD and the linear weights \(E_e^v,E_e^n\) are preprocessing products.

## Second-Level Surface Cauchy Fit

For a surface DOF farther than `2h` from every incident non-G1 edge, preserve
the existing G1 `48/28` Cauchy fit exactly.

For a surface DOF within the closed `2h` edge band, relevant edges are exactly
the non-G1 connections incident to its transitive G1 patch component whose
physical distance from the center is at most `2h`. Then:

1. Keep its existing 48 G1-nearest value conditions.
2. Keep its existing 28 G1-nearest normal conditions.
3. For each relevant incident edge, always add its nearest shared-edge point.
   Then add up to three more points from that edge whose Euclidean distance
   from the center is at most `2h`. Thus a continuously relevant edge cannot
   contribute zero rows merely because its cell-center point has an along-edge
   offset.
4. Insert every selected edge value as an additional value row

   \[
   p_i((x_e-x_i)/h)=\mu_e.
   \]

5. Give the edge row the same distance-weight formula as an ordinary value
   row. Do not add a separate tuning parameter in the first experiment.

At a feature vertex, a surface center may use shared points from every relevant
incident edge, but never from a topology-unrelated edge. No virtual edge-normal
equation is added in this design.

The edge values augment rather than replace the 48 original face values. Thus
the primary route changes only the presence of explicit shared-edge value
conditions.

## Linear GMRES Data Flow

No least-squares solve, geometry query, or topology search occurs during
GMRES.

For each operator or right-hand-side evaluation:

1. Apply the precomputed first-level maps:

   \[
   \mu_E=E^v\mu+E^n\eta.
   \]

2. Apply the precomputed second-level maps:

   \[
   c_i=M_i^v\mu+M_i^n\eta+M_i^E\mu_E.
   \]

3. Use the resulting coefficients in the unchanged KFBI correction and
   crossing-owner exterior-value restrict.

The bordered Neumann operator calls this map with \(\eta=0\). Its right-hand
side calls it with \(\mu=0\) and the prescribed compatibility-corrected
normal jump. By linearity, their sum is exactly the two-level reconstruction
using both kinds of Cauchy data.

The edge values must be evaluated once as a sparse gathered vector per
pipeline evaluation and reused by all surface centers. Fully expanding every
edge map into every surface map is unnecessary and would increase memory.

## Common Algebraic RHS

The common RHS must be identical across the three routes for a fixed
`(pose,N)`. For a surface DOF, normalize its native patch parameters to
\(\hat u,\hat v\in[0,1]\), let \(p\) be its zero-based patch ID, and define

\[
r_q=\sin(2\pi\hat u+0.37(p+1))
 +0.5\cos(2\pi\hat v-0.23(p+1))
 +0.25\sin(2\pi(\hat u+\hat v)).
\]

Subtract the surface-quadrature weighted mean and divide by the weighted RMS.
Require the resulting weighted mean magnitude to be at most `5e-13` and its
weighted RMS to differ from one by at most `5e-13`. Use this vector as the head
of the augmented RHS and set the bordered component to exactly zero. Start
GMRES from zero with tolerance `2e-10`, restart `80`, and cap `80`, exactly as
for the physical RHS.

## Failure Handling

A first- or second-level fit must fail its route explicitly when:

- an admitted sector cannot supply its required sample quota;
- a fitted matrix is rank deficient at the unchanged SVD cutoff;
- an edge frame is non-finite or degenerate after deterministic fallback;
- a virtual edge point or surface center selects an unrelated patch;
- any geometry query or reconstruction fingerprint changes during GMRES.

The failure record must include pose, `N`, route, edge/center ID, incident
sectors, actual counts, radii, singular extrema, condition number, and failure
stage. There is no silent fallback to a broader stencil.

## Numerical Comparison

Use the L-prism with:

- `N = 32,64,128`;
- identity;
- 17-degree rotation about `(1,2,3)`;
- the same rotation plus the existing `t_xyz_1` translation.

This gives 27 route configurations. Each configuration records:

- physical-RHS and common weighted-mean-zero RHS GMRES histories;
- density and interior `Linf/L2` errors and adjacent orders;
- exact-density equation defect;
- density error and equation defect in the disjoint edge-distance bins
  `<h` (`d/h<1`), `h--2h` (`1<=d/h<=2`), and `>2h` (`d/h>2`);
- first-level virtual-edge value error against the manufactured exact density
  \(u_{\mathrm{exact}}(x_e)-\overline{u}_{h}\), using the same discrete
  surface-weighted mean shift as the density error;
- first- and second-level stencil radii, sector balance, singular values, and
  condition numbers;
- crossing-owner query count/fingerprint and two-level reconstruction-map
  fingerprint before and after GMRES;
- preprocessing and solve time.

The direct-cross-face route determines whether merely seeing both faces is
sufficient. The shared-edge route determines whether an explicitly common
edge value supplies additional stability.

## Decision Rules

The completeness gate comes first. The formal output must contain exactly 27
unique `(pose,N,route)` rows. Every row must complete both RHS solves, converge
below `2e-10`, contain `iterations+1` residual-history entries, and preserve
label, crossing-owner, and two-level-map invariants. Any missing,
nonconvergent, rank-failed, or geometry-failed row prevents promotion, while
remaining a reported numerical result.

For route \(r\), RHS kind \(k\), pose \(p\), and level \(N\), let
\(I_{r,k,p,N}\) be its GMRES count. Define

\[
W_{r,k}=\max_{p,N} I_{r,k,p,N}, \qquad
S_{r,k,N}=\max_p I_{r,k,p,N}-\min_p I_{r,k,p,N}.
\]

The shared-edge route must not exceed either control in any \(W_{r,k}\) or
\(S_{r,k,N}\), must be strictly smaller in at least one worst-count
comparison, and strictly smaller in at least one pose-spread comparison.

At `N=128`, combine the `<h` and `h--2h` rows into the closed `d/h<=2` set for
each pose. Compute exact-equation-defect `Linf` and surface-weighted RMS on that
union. The shared-edge route must be strictly smaller than each control for
both norms and every pose.

For each error metric

\[
E\in\{\text{density Linf},\text{density L2},
       \text{interior Linf},\text{interior L2}\},
\]

compute the baseline refinement order
\(\log_2(E_{64}/E_{128})\). All four shared-edge orders must be nonnegative.
For every pose, every listed metric, and each control, the shared-edge
`N=128` error ratio must be at most `1.10`.

The shared-edge route must also reduce both virtual-edge `Linf` and
surface-weighted-RMS error from `N=64` to `N=128` for every pose, and repeated
operator applications must perform zero geometry queries and zero SVD
factorizations.

If the direct-cross-face route satisfies the same completeness, iteration,
defect, order, and error-ratio gates relative to G1, and the shared-edge route
does not strictly improve both an iteration metric and an edge-defect metric
relative to it, keep the simpler direct method. If neither route passes, the
next design is sector-wise polynomials with explicit shared-edge
value/tangential constraints.

No route becomes the production default from an `N=32` smoke result.

## Verification

Focused tests must prove:

- exact reproduction of a degree-three harmonic polynomial at shared edge
  points on a synthetic two-plane wedge;
- linearity of the edge map in both value and normal data;
- end-to-end linear splitting
  \(T(\mu,\eta)=T(\mu,0)+T(0,\eta)\) through edge reconstruction,
  second-level coefficients, KFBI correction, and crossing-owner restrict;
- rigid-transform invariance of virtual values, selected IDs, and conditions;
- balanced two-sector `48/28` first-level stencils;
- no selection of a nearby topology-unrelated sheet;
- deterministic handling of partial edge intervals and periodic G1 sectors;
- exact preservation of the old G1 surface fit outside `2h`;
- reuse of one shared edge value by both incident faces;
- zero geometry-query/SVD count during repeated operator applications;
- exact common-RHS weighted-mean/RMS normalization and identical common RHS
  across all three routes for a fixed `(pose,N)`;
- complete incremental diagnostics for successful and failed routes.

An `N=32` smoke attempts all nine pose/route configurations and audits output
completeness. The formal run attempts all 27 configurations and retains
partial output after every route.
