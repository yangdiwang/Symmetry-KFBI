# Grid-Scaled NURBS Intersection with Closest-Point Certification

## Goal

Make native NURBS Cartesian-edge queries both bounded and reliable. The
Cartesian-grid route uses rational Bezier leaves whose physical AABB extent is
at most `2h`, performs at most four further local subdivisions, and uses a
bounded segment-to-surface closest-point solve only when Newton is
ill-conditioned or the local subdivision budget is exhausted.

## Scope

- Rational Bezier geometry remains the conservative source of truth.
- Newton/root solving remains the primary intersection method.
- Triangles remain optional seeds only.
- The closest-point solve assists difficult/tangent cases and classifies
  terminal boxes; it does not replace conservative broad-phase rejection.
- Generic segment queries keep their existing configurable defaults. The
  Cartesian-domain path explicitly selects `2h` query elements and local depth
  `4`.

## Query hierarchy

The surface intersector first extracts knot-span rational Bezier elements. If
`maximum_element_extent` is configured, those elements are subdivided until
each leaf AABB has maximum extent no greater than that value. The BVH is built
over these leaves, so the same leaves are used for candidate enumeration and
the actual local intersection query.

For a Cartesian grid with spacing `(hx,hy,hz)`, the domain constructor uses

```text
maximum_element_extent = 2 * max(hx,hy,hz)
local_max_subdivision_depth = 4
```

This avoids the previous mismatch in which `2h` leaves screened grid edges
but the query returned to large original knot-span elements and a depth-36
local search.

## Local intersection

For each candidate leaf:

1. Reject parameter boxes conservatively using rational control bounds.
2. Try safeguarded Newton from triangle seeds and/or the box center.
3. Preserve the best failed Newton state and mark rank-deficient or stalled
   attempts as ill-conditioned.
4. Continue subdivision unless the root is certified unique/transverse or the
   box is otherwise resolved.
5. Invoke closest-point assistance only for an ill-conditioned Newton state or
   at local depth `4`.

The closest-point objective is

```text
min ||S(u,v) - (a + t(b-a))||,
u0 <= u <= u1, v0 <= v <= v1, 0 <= t <= 1.
```

For fixed `(u,v)`, `t` is the clamped orthogonal projection onto the segment.
The remaining bounded two-variable problem is solved by projected,
safeguarded Gauss-Newton. Active parameter bounds are removed from the local
normal equations so a boundary minimum is solved only in its free directions.
The stationarity test also accounts for objective-value roundoff. At most two
seeds are used: the best failed Newton state and the parameter-box center.

## Terminal decision

Let `tol_x` be the intersector geometry tolerance and
`tol_contact = 8 * tol_x`.

The local closest-point solve supplies a witness, not a global certificate. A
stationary point can be a local minimum or even hide two crossings elsewhere
in the same parameter box. Therefore every terminal decision is checked by an
adaptive rational-Bezier branch-and-bound certificate:

- Converged closest distance `> tol_contact`: the entire terminal box is
  classified as a miss only if projected rational bounds or a separating
  plane for the Bezier-control-hull/segment Minkowski difference excludes an
  intersection.
- Converged closest distance `<= tol_contact`: the recovered contact is
  accepted only if the root certificate proves one transverse root in the
  entire terminal box and every other certificate child is separated.
- Non-finite/non-converged closest solve, failed separation, a tangent contact,
  or an uncertified neighboring child remains unresolved and reports the
  existing typed failure.

The main root-isolation recursion remains limited to depth `4`. Terminal
certification may perform up to six additional Bezier subdivisions, used only
to prove separation or single-root uniqueness; it never searches for and
selects an arbitrary root. Exhaustive control-hull support directions are
bounded to elements with at most 16 controls (covering bicubic elements).
Higher-degree inputs use the cheaper projected certificates and fail closed if
those cannot resolve every child. If all children cannot be certified, the
Cartesian query fails closed.

## Diagnostics

Record and aggregate:

- maximum local root-isolation depth reached;
- terminal-certificate boxes and maximum certificate-only depth;
- closest-point attempts and iterations;
- contacts recovered by closest-point assistance;
- terminal boxes rejected by positive closest distance;
- closest-point failures that remain unresolved.

These fields distinguish insufficient root solving from genuine terminal
separation and make the depth-4 grid policy auditable.

## Compatibility and acceptance

- The generic intersector defaults remain unscaled with depth `36` and retain
  fail-closed terminal-contact handling.
- The Cartesian-grid route accepts a terminal contact only after adaptive
  whole-box single-root certification; otherwise it remains unresolved.
- Existing overlap, endpoint, feature-edge, tangency, seam-deduplication, and
  multiple-crossing rules remain unchanged. Extent-limited cylinder leaves
  retain explicit tangency diagnostics by combining a whole single-span source
  patch certificate with a witness restricted to the current leaf. In
  particular, a Cartesian edge
  with more than one transverse crossing is reported as under-resolved rather
  than selecting one root.
- Native NURBS regression tests must pass with and without triangle seeds.
- Cartesian-domain label tests must retain exact benchmark labels and report
  no unresolved candidates.
- The full Release build and regression executable must pass.
