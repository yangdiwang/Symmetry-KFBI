# 3D NURBS-Native Domain Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace nearest-normal and triangle-truth 3D labels/crossings with native NURBS intersections, Cartesian barriers, and component flood fill for the torus, hollow cylinder, and L-prism apps.

**Architecture:** A reusable core NURBS surface model carries explicit interval topology. Rational Bézier extraction supplies conservative bounds and a native segment solver; a grid-domain object converts its crossings into barrier arrays and flood-filled labels. `GridPair3D` consumes immutable NURBS labels and crossing records while retaining the existing P2 correction interface and all current KFBI numerical formulas.

**Tech Stack:** C++17, Eigen, CGAL exact-predicate triangle seeds only, CMake/MSBuild x64, existing NURBS basis/surface classes, `CartesianGrid3D`, `GridPair3D`, and native-NURBS 3D tests.

## Global Constraints

- Work directly on `main`; preserve unrelated user changes.
- Native NURBS patches are the sole geometric truth for production labels, crossings, positions, normals, patch IDs, and `(u,v)` parameters.
- A triangle may prioritize a candidate or seed Newton, but a triangle miss may never reject a conservative NURBS/Bézier candidate.
- Keep parameter-uniform surface DOFs, P2 correction triangles, feature-edge buffers, cubic Cauchy fitting, tricubic/cubic restrict, spread, bulk finite differences, and GMRES equations/tolerances unchanged.
- Require positive NURBS weights, declared outward patch orientation, closed interval topology, and a box strictly containing every surface component.
- Subdivide conservative rational Bézier bounds for Cartesian candidate enumeration until each leaf AABB has maximum physical extent at most `2 * max(hx,hy,hz)`.
- Use `tol_x = max(1e-12 * geometry_diameter, 1e-14)` for accepted physical roots; never turn an unresolved conservative candidate into “no intersection.”
- Allow zero or one transverse crossing per Cartesian grid edge. Two or more crossings stop with `multiple crossings on Cartesian edge`; no crossing is discarded.
- Stop explicitly for `surface intersects Cartesian node`, `surface overlaps Cartesian edge`, non-G1 feature-edge alignment, or tangential grid-edge contact.
- Keep domain labels `0 = exterior`, `component + 1 = interior`.
- Keep the old raw-triangle `GridPair3D` path for legacy callers, but the 3D native-NURBS app must use the new path with no triangle fallback.
- Do not run `N=256` until the L-prism `N=128` label, crossing, convergence, and order gates pass.

The model normal convention is fixed: every stored patch parameterization is
outward-oriented, so the authoritative normal is
`normalize(S_u.cross(S_v))`. There is no per-query normal flip. Closure
validation checks that this convention is mutually consistent across every
declared connection.

---

### Task 1: Add explicit native NURBS surface topology

**Files:**
- Create: `src/geometry/nurbs_surface_model_3d.hpp`
- Create: `src/geometry/nurbs_surface_model_3d.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `apps/native_nurbs_surface_3d.hpp`
- Modify: `apps/native_nurbs_surface_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Consumes: existing `geometry3d::NurbsSurfacePatch3D` values.
- Produces: `NurbsSurfaceModel3D`, explicit interval connections, topology validation, and `NativeNurbsSurface3D::geometry_model()` for every benchmark.

- [ ] **Step 1: Write failing topology tests**

Include the new header and add these assertions to
`apps/native_nurbs_surface_3d_test.cpp`:

```cpp
template <class Function>
void require_throws_contains(Function&& function,
                             const std::string& needle,
                             const std::string& message)
{
    try {
        function();
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(needle) != std::string::npos,
                message + ": " + error.what());
        return;
    }
    throw std::runtime_error(message + ": no exception");
}

void test_native_surface_interval_topology()
{
    struct Expected {
        GeometryKind3D kind;
        int patches;
        int connections;
    };
    for (const Expected expected : {
             Expected{GeometryKind3D::Torus, 16, 32},
             Expected{GeometryKind3D::HollowCylinder, 16, 32},
             Expected{GeometryKind3D::LPrism, 12, 26}}) {
        const NativeNurbsSurface3D surface =
            make_native_nurbs_surface_3d(expected.kind);
        const kfbim::geometry3d::NurbsSurfaceModel3D model =
            surface.geometry_model();
        const auto diagnostic = model.validate_closed();
        require(model.num_patches() == expected.patches,
                surface.name + " geometry-model patch count");
        require(model.connections().size()
                    == static_cast<std::size_t>(expected.connections),
                surface.name + " interval-connection count");
        require(diagnostic.uncovered_interval_count == 0
                    && diagnostic.multiply_covered_interval_count == 0
                    && diagnostic.position_mismatch_count == 0
                    && diagnostic.orientation_mismatch_count == 0
                    && diagnostic.g1_normal_mismatch_count == 0,
                surface.name + " closed oriented interval topology");
    }
}
```

Append this negative case inside `test_native_surface_interval_topology()`
after the benchmark loop, then call that test from `main()` before
`test_uniform_native_dofs()`:

```cpp
const kfbim::geometry3d::NurbsSurfaceModel3D open_model(
    {kfbim::geometry3d::NurbsSurfacePatch3D::make_unit_square_xy()},
    {0}, {});
require_throws_contains(
    [&] { (void)open_model.validate_closed(); },
    "uncovered patch-edge interval",
    "open NURBS patch is rejected by closed-topology validation");
```

- [ ] **Step 2: Run RED**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test --parallel 2
```

Expected: compilation fails because `nurbs_surface_model_3d.hpp` and
`NativeNurbsSurface3D::geometry_model()` do not exist.

- [ ] **Step 3: Add the core topology types**

Create `nurbs_surface_model_3d.hpp` with these public declarations:

```cpp
#pragma once

#include "nurbs_surface_3d.hpp"
#include <Eigen/Dense>
#include <cstddef>
#include <vector>

namespace kfbim::geometry3d {

enum class NurbsPatchEdge3D { UMin, UMax, VMin, VMax };

struct NurbsPatchEdgeInterval3D {
    int patch = -1;
    NurbsPatchEdge3D edge = NurbsPatchEdge3D::UMin;
    double begin = 0.0;
    double end = 0.0;
};

struct NurbsPatchEdgeConnection3D {
    NurbsPatchEdgeInterval3D first;
    NurbsPatchEdgeInterval3D second;
    bool reversed = false;
    bool g1 = false;
};

struct NurbsSurfaceTopologyDiagnostics3D {
    int patch_count = 0;
    int component_count = 0;
    int connection_count = 0;
    int uncovered_interval_count = 0;
    int multiply_covered_interval_count = 0;
    int position_mismatch_count = 0;
    int orientation_mismatch_count = 0;
    int g1_normal_mismatch_count = 0;
};

struct NurbsAabb3D {
    Eigen::Vector3d lower = Eigen::Vector3d::Zero();
    Eigen::Vector3d upper = Eigen::Vector3d::Zero();
    bool overlaps(const NurbsAabb3D& other, double tolerance = 0.0) const;
    bool contains(const Eigen::Vector3d& point,
                  double tolerance = 0.0) const;
    double max_extent() const;
    double diameter() const;
};

class NurbsSurfaceModel3D {
public:
    NurbsSurfaceModel3D(
        std::vector<NurbsSurfacePatch3D> patches,
        std::vector<int> patch_components,
        std::vector<NurbsPatchEdgeConnection3D> connections);

    int num_patches() const;
    int num_components() const;
    const NurbsSurfacePatch3D& patch(int patch) const;
    int patch_component(int patch) const;
    const std::vector<NurbsSurfacePatch3D>& patches() const;
    const std::vector<NurbsPatchEdgeConnection3D>& connections() const;
    NurbsSurfaceTopologyDiagnostics3D validate_closed(
        double relative_tolerance = 1e-10) const;
    NurbsAabb3D control_bounds() const;

private:
    std::vector<NurbsSurfacePatch3D> patches_;
    std::vector<int> patch_components_;
    std::vector<NurbsPatchEdgeConnection3D> connections_;
};

NurbsPatchEdgeInterval3D full_patch_edge_interval(
    const NurbsSurfacePatch3D& patch,
    int patch_index,
    NurbsPatchEdge3D edge);

} // namespace kfbim::geometry3d
```

- [ ] **Step 4: Implement interval mapping and closure validation**

In `nurbs_surface_model_3d.cpp`, implement one edge evaluator used by both
position and orientation checks:

```cpp
struct EdgeSample3D {
    Eigen::Vector3d point;
    Eigen::Vector3d positive_parameter_tangent;
    Eigen::Vector3d oriented_boundary_tangent;
};

EdgeSample3D edge_sample(const NurbsSurfacePatch3D& patch,
                         NurbsPatchEdge3D edge,
                         double parameter)
{
    double u = patch.domain_start_u();
    double v = patch.domain_start_v();
    switch (edge) {
    case NurbsPatchEdge3D::UMin:
        v = parameter;
        break;
    case NurbsPatchEdge3D::UMax:
        u = patch.domain_end_u();
        v = parameter;
        break;
    case NurbsPatchEdge3D::VMin:
        u = parameter;
        break;
    case NurbsPatchEdge3D::VMax:
        u = parameter;
        v = patch.domain_end_v();
        break;
    }
    const auto d = patch.evaluate_with_derivatives(u, v);
    const Eigen::Vector3d positive =
        edge == NurbsPatchEdge3D::UMin
             || edge == NurbsPatchEdge3D::UMax ? d.dv : d.du;
    Eigen::Vector3d boundary = positive;
    if (edge == NurbsPatchEdge3D::UMin
        || edge == NurbsPatchEdge3D::VMax) {
        boundary = -boundary;
    }
    return {d.point, positive, boundary};
}
```

For every patch edge, collect all incident connection interval endpoints,
sort them, and sweep the atomic intervals. Require coverage count exactly one.
For each connection sample nine mapped parameters including both endpoints;
require physical mismatch no larger than
`max(1e-14, relative_tolerance * model_diameter)` and normalized oriented
boundary tangents to have dot product below `-1 + 1e-8`.
For a `g1=true` connection, also require the two outward
`normalize(S_u.cross(S_v))` normals to have dot product above `1 - 1e-8` at
the same samples. A `g1=false` connection has no normal-continuity
requirement.

Reject invalid patch/component indices, non-increasing intervals, intervals
outside their varying parameter domain, connections between different
components, and non-contiguous component IDs. Use these exact error fragments:

```text
uncovered patch-edge interval
multiply covered patch-edge interval
connected NURBS edges disagree in position
connected NURBS edges have inconsistent orientation
declared G1 NURBS edges disagree in normal
```

Add `geometry/nurbs_surface_model_3d.cpp` to the 3D sources in
`src/CMakeLists.txt`.

- [ ] **Step 5: Populate explicit benchmark topology**

Add to `NativeNurbsSurface3D`:

```cpp
std::vector<geometry3d::NurbsPatchEdgeConnection3D>
    geometric_connections;
std::vector<int> patch_components;

[[nodiscard]] geometry3d::NurbsSurfaceModel3D geometry_model() const;
```

Keep the existing app `PatchEdge3D` as an alias of the core enum. Make
`connect_smooth()` also append one full-edge `g1=true` connection. Replace
feature-only patch-neighbor calls with full or partial `g1=false` interval
connections.

Represent a periodic wrap explicitly by the full-edge connection between the
last and first parameter patches. Its interval map supplies the periodic
identification and `g1=true` supplies the canonicalization rule; no coordinate
welding or implicit modulo search is used.

Add this app helper; it also calls the existing
`connect_topological_patches()` so Cauchy topology is preserved:

```cpp
void connect_feature(
    NativeNurbsSurface3D& surface,
    int patch_a, PatchEdge3D edge_a, double begin_a, double end_a,
    int patch_b, PatchEdge3D edge_b, double begin_b, double end_b,
    bool reversed);
```

Use these cylinder rim pairs for every quarter `q`:

```cpp
for (int q = 0; q < 4; ++q) {
    connect_feature(surface,
        q, PatchEdge3D::VMax, 0.0, 1.0,
        8 + q, PatchEdge3D::VMin, 0.0, 1.0, false);
    connect_feature(surface,
        q, PatchEdge3D::VMin, 0.0, 1.0,
        12 + q, PatchEdge3D::VMax, 0.0, 1.0, false);
    connect_feature(surface,
        4 + q, PatchEdge3D::VMin, 0.0, 1.0,
        8 + q, PatchEdge3D::VMax, 0.0, 1.0, false);
    connect_feature(surface,
        4 + q, PatchEdge3D::VMax, 0.0, 1.0,
        12 + q, PatchEdge3D::VMin, 0.0, 1.0, false);
}
```

For the L-prism keep the four existing smooth top/bottom connections and add
six full `(6+s, UMax) -> (6+(s+1)%6, UMin)` non-G1 connections. Add these
exact bottom maps (`side patch/edge/interval -> cell patch/edge/full interval`):

```text
6/VMin/[0,.5] -> 0/UMin/[0,1], same
6/VMin/[.5,1] -> 1/UMin/[0,1], same
7/VMin/[0,1]  -> 1/VMax/[0,1], same
8/VMin/[0,1]  -> 1/UMax/[0,1], reversed
9/VMin/[0,1]  -> 2/VMax/[0,1], same
10/VMin/[0,1] -> 2/UMax/[0,1], reversed
11/VMin/[0,.5] -> 2/VMin/[0,1], reversed
11/VMin/[.5,1] -> 0/VMin/[0,1], reversed
```

Add the matching top maps:

```text
6/VMax/[0,.5] -> 3/VMin/[0,1], same
6/VMax/[.5,1] -> 4/VMin/[0,1], same
7/VMax/[0,1]  -> 4/UMax/[0,1], same
8/VMax/[0,1]  -> 4/VMax/[0,1], reversed
9/VMax/[0,1]  -> 5/UMax/[0,1], same
10/VMax/[0,1] -> 5/VMax/[0,1], reversed
11/VMax/[0,.5] -> 5/UMin/[0,1], reversed
11/VMax/[.5,1] -> 3/UMin/[0,1], reversed
```

Encode the tables as `NurbsPatchEdgeConnection3D` values rather than
inferring nearest patches. Set all benchmark patch components to zero.

- [ ] **Step 6: Run GREEN and commit**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test --parallel 2
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
git diff --check
git add src/geometry/nurbs_surface_model_3d.hpp src/geometry/nurbs_surface_model_3d.cpp src/CMakeLists.txt apps/native_nurbs_surface_3d.hpp apps/native_nurbs_surface_3d.cpp apps/native_nurbs_surface_3d_test.cpp
git commit -m "feat: model native NURBS surface topology"
```

Expected: `native NURBS model tests passed`.

### Task 2: Extract and subdivide conservative rational Bézier elements

**Files:**
- Create: `src/geometry/rational_bezier_surface_3d.hpp`
- Create: `src/geometry/rational_bezier_surface_3d.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Consumes: `NurbsSurfaceModel3D` from Task 1.
- Produces: exact span-wise homogeneous Bézier controls, conservative AABBs, and de Casteljau subdivision used by Tasks 3–5.

- [ ] **Step 1: Add failing extraction and subdivision tests**

Construct this quadratic two-span planar test model:

```cpp
using kfbim::geometry::NurbsBasis1D;
using kfbim::geometry3d::NurbsSurfacePatch3D;
const NurbsSurfacePatch3D multispan_patch(
    NurbsBasis1D(2, {0,0,0,0.5,1,1,1}),
    NurbsBasis1D(1, {0,0,1,1}),
    {{{0.00,0.0,0.0}, {0.00,1.0,0.0}},
     {{0.25,0.0,0.0}, {0.25,1.0,0.0}},
     {{0.75,0.0,0.0}, {0.75,1.0,0.0}},
     {{1.00,0.0,0.0}, {1.00,1.0,0.0}}},
    {{1.0,1.2}, {0.8,1.1}, {1.3,0.9}, {1.0,1.4}});
const kfbim::geometry3d::NurbsSurfaceModel3D model(
    {multispan_patch}, {0}, {});
const auto elements =
    kfbim::geometry3d::extract_rational_bezier_elements_3d(model);
require(elements.size() == 2, "two U knot spans produce two elements");
for (const auto& element : elements) {
    for (int iu = 0; iu <= 4; ++iu) {
        for (int iv = 0; iv <= 4; ++iv) {
            const double u = element.u0()
                + (element.u1() - element.u0()) * iu / 4.0;
            const double v = element.v0()
                + (element.v1() - element.v0()) * iv / 4.0;
            const Eigen::Vector3d exact = model.patch(0).evaluate(u, v);
            require((element.evaluate(u, v) - exact).norm() < 2e-12,
                    "Bezier extraction preserves NURBS evaluation");
            require(element.bounds().contains(exact, 2e-12),
                    "Bezier control hull conservatively bounds surface");
        }
    }
}
```

For each original element, call `split_u()`, then call `split_v()` on both U
children and repeat the evaluation/bounds assertions for the resulting four
children. Also extract every benchmark model and require every homogeneous
control has `w > 0` and all sampled NURBS points lie inside their element
AABBs.

- [ ] **Step 2: Run RED**

Expected: compilation fails because `rational_bezier_surface_3d.hpp` is absent.

- [ ] **Step 3: Add the exact public element API**

```cpp
struct RationalBezierElement3D {
    int patch_index = -1;
    int component = -1;
    int degree_u = 0;
    int degree_v = 0;
    double parameter_u0 = 0.0;
    double parameter_u1 = 0.0;
    double parameter_v0 = 0.0;
    double parameter_v1 = 0.0;
    std::vector<Eigen::Vector4d> homogeneous_controls;

    double u0() const;
    double u1() const;
    double v0() const;
    double v1() const;
    const Eigen::Vector4d& control(int i, int j) const;
    Eigen::Vector3d evaluate(double u, double v) const;
    NurbsAabb3D bounds() const;
    std::array<RationalBezierElement3D, 2> split_u() const;
    std::array<RationalBezierElement3D, 2> split_v() const;
};

std::vector<RationalBezierElement3D>
extract_rational_bezier_elements_3d(const NurbsSurfaceModel3D& model);

std::vector<RationalBezierElement3D>
subdivide_rational_bezier_elements_to_extent_3d(
    const std::vector<RationalBezierElement3D>& elements,
    double maximum_extent);
```

- [ ] **Step 4: Implement tensor-product knot insertion**

Convert every control point to `(w*x,w*y,w*z,w)`. Repeatedly insert each
interior U knot until multiplicity equals `degree_u`, applying the standard
single-knot curve rule independently to every V column:

```cpp
Q[i] = alpha * P[i] + (1.0 - alpha) * P[i - 1];
alpha = (knot - U[i]) / (U[i + degree] - U[i]);
```

Then do the same in V for every refined U row. For every nonzero refined knot
span `k`, its active homogeneous controls are the inclusive index range from
`k-degree` through `k`.
Form the tensor block for each U/V span pair and retain its original parameter
limits, patch index, and component.

Reject a nonpositive homogeneous weight, zero parameter span, or inconsistent
control count. Do not fit controls from samples.

- [ ] **Step 5: Implement rational bounds and de Casteljau subdivision**

Project each positive homogeneous control as `xyz / w`; their componentwise
minimum/maximum defines the conservative AABB. Split homogeneous control rows
or columns at parameter fraction `0.5` using de Casteljau, update the native
parameter interval, and preserve patch/component IDs.

`subdivide_rational_bezier_elements_to_extent_3d()` repeatedly splits the
parameter direction with the larger projected control-hull variation until
`bounds().max_extent() <= maximum_extent`. Throw
`Bezier acceleration subdivision exceeded depth 48` instead of returning a
leaf above the requested extent.

- [ ] **Step 6: Run GREEN and commit**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test --parallel 2
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
git diff --check
git add src/geometry/rational_bezier_surface_3d.hpp src/geometry/rational_bezier_surface_3d.cpp src/CMakeLists.txt apps/native_nurbs_surface_3d_test.cpp
git commit -m "feat: extract rational Bezier surface elements"
```

### Task 3: Solve a segment intersection on one Bézier element

**Files:**
- Create: `src/geometry/nurbs_bezier_intersection_3d.hpp`
- Create: `src/geometry/nurbs_bezier_intersection_3d.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Consumes: one `RationalBezierElement3D`, its source `NurbsSurfacePatch3D`, and a finite segment.
- Produces: all locally isolated `(u,v,t)` roots plus conservative-rejection/Newton diagnostics; it does not deduplicate different elements or patches.

- [ ] **Step 1: Add failing analytic root tests**

Add three focused cases:

```cpp
const kfbim::geometry3d::NurbsSurfaceModel3D plane_model(
    {kfbim::geometry3d::NurbsSurfacePatch3D::make_unit_square_xy()},
    {0}, {});
const auto plane_element =
    kfbim::geometry3d::extract_rational_bezier_elements_3d(
        plane_model).front();
kfbim::geometry3d::NurbsElementIntersectionOptions3D options;
options.geometry_tolerance = 1e-12;
const auto plane = kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
    plane_element, plane_model.patch(0),
    {0.25, 0.75, -1.0}, {0.25, 0.75, 1.0}, options);
require(plane.roots.size() == 1
            && std::abs(plane.roots[0].u - 0.25) < 2e-12
            && std::abs(plane.roots[0].v - 0.75) < 2e-12
            && std::abs(plane.roots[0].t - 0.5) < 2e-12,
        "unit-plane native root");

const kfbim::geometry3d::NurbsSurfaceModel3D cylinder_model(
    {kfbim::geometry3d::NurbsSurfacePatch3D::
         make_quarter_cylinder_patch(1.0, 0.0, 1.0)},
    {0}, {});
const auto cylinder_element =
    kfbim::geometry3d::extract_rational_bezier_elements_3d(
        cylinder_model).front();
const auto cylinder = kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
    cylinder_element, cylinder_model.patch(0),
    {0.8, 0.5, 0.5}, {0.95, 0.5, 0.5}, options);
require(cylinder.roots.size() == 1
            && std::abs(cylinder.roots[0].point.x() - std::sqrt(0.75))
                   < 2e-12
            && cylinder.roots[0].residual < 2e-12,
        "quarter-cylinder native root");
require(cylinder.diagnostics.triangle_seed_hits == 0
            && cylinder.diagnostics.roots_recovered_without_triangle_seed == 1,
        "curved root survives triangle-seed miss");
```

Add this conservative-rejection assertion:

```cpp
const auto miss = kfbim::geometry3d::intersect_nurbs_bezier_element_3d(
    plane_element, plane_model.patch(0),
    {2.0, 2.0, -1.0}, {2.0, 2.0, 1.0}, options);
require(miss.roots.empty() && miss.diagnostics.newton_attempts == 0
            && miss.diagnostics.conservative_rejections > 0,
        "control hull rejects an impossible line before Newton");
```

- [ ] **Step 2: Run RED**

Expected: compilation fails because the element-intersection API is absent.

- [ ] **Step 3: Add the local solver types**

```cpp
struct NurbsElementRoot3D {
    int patch_index = -1;
    int component = -1;
    double u = 0.0;
    double v = 0.0;
    double t = 0.0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double residual = 0.0;
    double transversality = 0.0;
};

struct NurbsElementIntersectionDiagnostics3D {
    int subdivision_boxes = 0;
    int conservative_rejections = 0;
    int triangle_seed_hits = 0;
    int newton_attempts = 0;
    int newton_iterations = 0;
    int roots_recovered_without_triangle_seed = 0;
    int unresolved_boxes = 0;
};

struct NurbsElementIntersectionResult3D {
    std::vector<NurbsElementRoot3D> roots;
    NurbsElementIntersectionDiagnostics3D diagnostics;
    bool overlap_detected = false;
};

struct NurbsElementIntersectionOptions3D {
    double geometry_tolerance = 1e-12;
    double parameter_tolerance = 1e-12;
    int max_subdivision_depth = 36;
    int max_newton_iterations = 24;
    bool use_triangle_seed = true;
};

NurbsElementIntersectionResult3D intersect_nurbs_bezier_element_3d(
    const RationalBezierElement3D& element,
    const NurbsSurfacePatch3D& patch,
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_end,
    const NurbsElementIntersectionOptions3D& options);
```

- [ ] **Step 4: Implement conservative line rejection**

Construct an orthonormal pair perpendicular to the segment direction. Project
every positive rational Bézier control point onto both transverse axes and the
longitudinal segment axis. Reject a parameter box if either transverse range
does not contain the segment line within `geometry_tolerance`, or if its
longitudinal range does not overlap `[0, segment_length]`.

Try the two corner triangles only to obtain a seed. Whether they hit or miss,
recursively split every still-ambiguous Bézier box along the parameter
direction with larger projected variation.

Before root isolation, certify the two overlap forms present in the supported
geometries. For a planar element, require all projected controls to lie in one
plane within `geometry_tolerance`, require the segment line to lie in that
plane, and project a strict interior segment point to strict interior
`(u,v)`. Regularity plus that interior point certifies a positive-length local
intersection. For a ruled element, require every homogeneous control row (or
column) in the ruling direction to project to a straight line parallel to the
segment, solve the transverse control curve, and require the recovered fixed
parameter and a strict interior ruling interval. Only after one of these
control-net certificates succeeds set `overlap_detected=true`; three sampled
coincidences alone are not an overlap proof. Any other persistent singular box
stops as an unresolved conservative candidate.

- [ ] **Step 5: Implement safeguarded native Newton validation**

For a seed `(u,v,t)`, solve

```text
S(u,v) - (a + t*(b-a)) = 0
```

with Jacobian columns `[S_u, S_v, -(b-a)]`. Use analytic derivatives, solve
the `3x3` system with Eigen full-pivot LU, and halve the step until `(u,v,t)`
stays in its Bézier box and the full residual decreases.

Accept only when:

```cpp
root.residual <= options.geometry_tolerance
&& root.t >= -options.parameter_tolerance
&& root.t <= 1.0 + options.parameter_tolerance;
```

Normalize `S_u cross S_v`, store
`abs(normal.dot(normalized_segment_direction))` as transversality, and dedupe
roots from sibling boxes when physical point, `t`, and same-patch parameters
agree within `8 *` their respective tolerances.

If a conservative box reaches `max_subdivision_depth` without a validated root
or a proof of rejection, increment `unresolved_boxes` and throw
`unresolved conservative NURBS intersection candidate` after the traversal.

- [ ] **Step 6: Run GREEN and commit**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test --parallel 2
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
git diff --check
git add src/geometry/nurbs_bezier_intersection_3d.hpp src/geometry/nurbs_bezier_intersection_3d.cpp src/CMakeLists.txt apps/native_nurbs_surface_3d_test.cpp
git commit -m "feat: intersect segments with native NURBS elements"
```

### Task 4: Build the complete multi-patch NURBS query engine

**Files:**
- Create: `src/geometry/nurbs_surface_intersector_3d.hpp`
- Create: `src/geometry/nurbs_surface_intersector_3d.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Consumes: a validated `NurbsSurfaceModel3D` and Task 2/3 element operations.
- Produces: canonical multi-patch segment roots, strict grid-edge queries, global component parity, model bounds, and `2h` conservative leaves.

- [ ] **Step 1: Add failing surface-level query tests**

For each benchmark model, construct `NurbsSurfaceIntersector3D` and require its
model bounds contain dense patch samples. Add these behavioral tests:

```cpp
const NativeNurbsSurface3D torus =
    make_native_nurbs_surface_3d(GeometryKind3D::Torus);
const kfbim::geometry3d::NurbsSurfaceIntersector3D torus_intersector(
    torus.geometry_model());
const auto leaves = torus_intersector.acceleration_leaves(0.05);
require(std::all_of(leaves.begin(), leaves.end(), [](const auto& leaf) {
            return leaf.bounds().max_extent() <= 0.05 + 1e-13;
        }), "all acceleration leaves satisfy requested extent");

const auto seam_hit = torus_intersector.intersect_segment(
    {0.80, -0.04, 0.03}, {0.84, -0.04, 0.03});
require(seam_hit.crossings.size() == 1
            && seam_hit.diagnostics.seam_deduplications >= 1,
        "periodic torus seam root is canonicalized once");

const NativeNurbsSurface3D cylinder =
    make_native_nurbs_surface_3d(GeometryKind3D::HollowCylinder);
const kfbim::geometry3d::NurbsSurfaceIntersector3D cylinder_intersector(
    cylinder.geometry_model());
const auto smooth_hit = cylinder_intersector.intersect_segment(
    {0.06, 0.48, 0.0}, {0.06, 0.52, 0.0});
require(smooth_hit.crossings.size() == 1
            && smooth_hit.diagnostics.seam_deduplications >= 1,
        "smooth cylinder-quarter seam root is canonicalized once");
```

Use the Cartesian-edge query from `(-1,-0.04,0.03)` to
`(1,-0.04,0.03)` and require `multiple crossings on Cartesian edge`. For the
L-prism, query from `(0,-0.3,0.67)` to `(0,-0.3,0.8)` and require
`surface intersects Cartesian node`; query the top-face segment from
`(-0.2,-0.3,0.67)` to `(-0.1,-0.3,0.67)` and require
`surface overlaps Cartesian edge`. For the cylinder query from
`(0.61,-0.10,0)` to `(0.61,0,0)` and require `tangential Cartesian edge
contact`. Query through the outer top rim from `(0.59,-0.05,0.67)` to
`(0.63,-0.05,0.67)` and require `non-G1 feature-edge alignment`.

- [ ] **Step 2: Run RED**

Expected: compilation fails because `NurbsSurfaceIntersector3D` is undefined.

- [ ] **Step 3: Add the public surface-query API**

```cpp
struct NurbsSurfaceCrossing3D {
    int patch_index = -1;
    int component = -1;
    double u = 0.0;
    double v = 0.0;
    double edge_parameter = 0.0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double residual = 0.0;
    double transversality = 0.0;
};

struct NurbsSurfaceIntersectionDiagnostics3D {
    int candidate_elements = 0;
    int triangle_seed_hits = 0;
    int triangle_seed_misses_recovered = 0;
    int subdivision_boxes = 0;
    int newton_attempts = 0;
    int newton_iterations = 0;
    int same_patch_deduplications = 0;
    int seam_deduplications = 0;
    int unresolved_candidates = 0;
};

struct NurbsSurfaceIntersectionResult3D {
    std::vector<NurbsSurfaceCrossing3D> crossings;
    NurbsSurfaceIntersectionDiagnostics3D diagnostics;
    bool overlap_detected = false;
};

struct NurbsCartesianEdgeQuery3D {
    int axis = -1;
    int i = -1;
    int j = -1;
    int k = -1;
    Eigen::Vector3d start = Eigen::Vector3d::Zero();
    Eigen::Vector3d end = Eigen::Vector3d::Zero();
};

struct NurbsSurfaceIntersectorOptions3D {
    bool use_triangle_seeds = true;
    int bvh_leaf_size = 8;
};

class NurbsSurfaceIntersector3D {
public:
    explicit NurbsSurfaceIntersector3D(
        NurbsSurfaceModel3D model,
        NurbsSurfaceIntersectorOptions3D options = {});
    const NurbsSurfaceModel3D& model() const;
    const NurbsAabb3D& bounds() const;
    double geometry_tolerance() const;
    NurbsSurfaceIntersectionResult3D intersect_segment(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end) const;
    std::optional<NurbsSurfaceCrossing3D> intersect_cartesian_edge(
        const NurbsCartesianEdgeQuery3D& edge,
        NurbsSurfaceIntersectionDiagnostics3D* diagnostics = nullptr) const;
    std::vector<int> containing_components(
        const Eigen::Vector3d& point) const;
    std::vector<RationalBezierElement3D> acceleration_leaves(
        double maximum_extent) const;
};
```

- [ ] **Step 4: Implement BVH candidate collection and canonicalization**

Build a private median-split AABB BVH over the extracted knot-span elements;
split on the longest node axis and make leaves at eight elements. A segment
query visits only overlapping BVH nodes and calls Task 3 for each candidate.

At construction compute
`tol_x = max(1e-12 * bounds().diameter(), 1e-14)` and pass it as the local
solver's `geometry_tolerance`. Map
`NurbsSurfaceIntersectorOptions3D::use_triangle_seeds` to the local
`use_triangle_seed` option on every call. Aggregate every local diagnostic and
propagate `overlap_detected` with logical OR; an unresolved local candidate
remains an exception and is never converted to an empty result.

First merge same-patch roots by `(u,v,t,point)`. For cross-patch roots, find
whether each root lies on a connected interval and map its edge parameter to
the mate. Merge only declared G1/periodic seam duplicates. A root on a declared
non-G1 connection throws `non-G1 feature-edge alignment`; coincident roots on
unconnected patches throw `coincident roots on unrelated NURBS patches`.

- [ ] **Step 5: Enforce strict Cartesian-edge degeneracy rules**

`intersect_cartesian_edge()` validates the canonical result in this order:

```cpp
std::string cartesian_edge_diagnostic(
    const char* prefix,
    const NurbsCartesianEdgeQuery3D& edge,
    const std::vector<NurbsSurfaceCrossing3D>& roots)
{
    std::ostringstream out;
    out << prefix << " axis=" << edge.axis
        << " (" << edge.i << ',' << edge.j << ',' << edge.k << ')';
    for (const auto& root : roots) {
        out << " root=(patch=" << root.patch_index
            << ",u=" << root.u << ",v=" << root.v
            << ",t=" << root.edge_parameter << ')';
    }
    return out.str();
}

if (result.overlap_detected)
    throw std::runtime_error(cartesian_edge_diagnostic(
        "surface overlaps Cartesian edge", edge, roots));
if (endpoint_root)
    throw std::runtime_error(cartesian_edge_diagnostic(
        "surface intersects Cartesian node", edge, roots));
if (root.transversality <= 1e-10)
    throw std::runtime_error(cartesian_edge_diagnostic(
        "tangential Cartesian edge contact", edge, roots));
if (roots.size() > 1)
    throw std::runtime_error(cartesian_edge_diagnostic(
        "multiple crossings on Cartesian edge", edge, roots));
```

Every message includes `axis`, `(i,j,k)`, and every available
`(patch,u,v,t)`. Zero roots returns `std::nullopt`; one root returns it.

- [ ] **Step 6: Implement global parity classification**

For `containing_components(point)`, extend a ray beyond `bounds()` and use
`intersect_segment()`, where multiple roots are allowed. Try directions in
this exact order:

```cpp
{1.0, 0.3713906763541037, 0.6947465906068658}
{0.6180339887498948, 1.0, 0.4142135623730950}
{0.2718281828459045, 0.5772156649015329, 1.0}
```

Normalize each direction. Reject a direction with an endpoint, overlap,
feature, or tangent degeneracy and try the next; catch only those four named
degeneracy diagnostics, never an unresolved-candidate error. Toggle parity by
canonical crossing component. Return sorted odd component IDs; if all
directions degenerate, throw
`unable to classify point with three deterministic NURBS rays`.

- [ ] **Step 7: Run GREEN and commit**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test --parallel 2
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
git diff --check
git add src/geometry/nurbs_surface_intersector_3d.hpp src/geometry/nurbs_surface_intersector_3d.cpp src/CMakeLists.txt apps/native_nurbs_surface_3d_test.cpp
git commit -m "feat: query complete native NURBS surfaces"
```

### Task 5: Build Cartesian barriers and flood-filled domain labels

**Files:**
- Create: `src/geometry/nurbs_cartesian_domain_3d.hpp`
- Create: `src/geometry/nurbs_cartesian_domain_3d.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`

**Interfaces:**
- Consumes: a node-layout `CartesianGrid3D` and validated `NurbsSurfaceModel3D`.
- Produces: immutable node labels, dense byte barriers, sparse native crossing records, component diagnostics, and endpoint-pair crossing lookup.

- [ ] **Step 1: Add the failing N128-aligned L-prism regression**

Use the N128 spacing in a cropped shifted box:

```cpp
void test_nurbs_cartesian_l_prism_labels()
{
    constexpr double h = 3.0 / 128.0;
    const NativeNurbsSurface3D surface =
        make_native_nurbs_surface_3d(GeometryKind3D::LPrism);
    const kfbim::CartesianGrid3D grid(
        {-0.5625, -0.703125, -0.6328125},
        {h, h, h}, {54, 54, 58}, kfbim::DofLayout3D::Node);

    const kfbim::geometry3d::NurbsCartesianDomain3D domain(
        grid, surface.geometry_model());
    require(domain.label(grid.index(27, 27, 4)) == 1,
            "L-prism reentrant lower-arm node is inside");
    require(domain.label(grid.index(28, 28, 4)) == 0,
            "L-prism missing quadrant is outside");

    int mismatches = 0;
    for (int node = 0; node < grid.num_dofs(); ++node) {
        const auto x = grid.coord(node);
        const Eigen::Vector3d point(x[0], x[1], x[2]);
        const bool exact = surface.exact_inside(point);
        mismatches += (domain.label(node) > 0) != exact ? 1 : 0;
    }
    require(mismatches == 0, "L-prism cropped-grid labels are exact");
    require(domain.diagnostics().barrier_edge_counts[0]
                + domain.diagnostics().barrier_edge_counts[1]
                + domain.diagnostics().barrier_edge_counts[2] > 0,
            "native crossings create Cartesian barriers");
}
```

Call it from `main()` after the surface-intersector tests.

- [ ] **Step 2: Add curved-target and triangle-independence tests**

Use the shifted grid

```cpp
{-0.781, -0.841, -0.721}, {0.05, 0.05, 0.05}, {34, 33, 31}
```

for torus and hollow-cylinder models. Require node `(17,16,15)` exterior for
both, node `(28,16,15)` inside the torus, and node `(25,16,15)` inside the
cylinder. Build each domain twice with `use_triangle_seeds=true` and `false`;
require identical labels, crossing points, and barrier counts. The domain API
accepts no external triangle mesh, so visualization-mesh refinement and
nonconforming triangle seams cannot affect either construction.

Create a two-component fixture by copying the complete L-prism model, adding
`(1.6,0,0)` to every control point of the copy, offsetting copied patch indices
in every connection, and assigning copied patches component `1`. Use:

```cpp
NurbsSurfacePatch3D translated_patch(const NurbsSurfacePatch3D& patch,
                                     const Eigen::Vector3d& offset)
{
    auto controls = patch.control_net();
    for (auto& row : controls)
        for (Eigen::Vector3d& point : row)
            point += offset;
    return NurbsSurfacePatch3D(
        patch.basis_u(), patch.basis_v(), controls, patch.weights());
}

NurbsSurfaceModel3D two_translated_components(
    const NurbsSurfaceModel3D& source,
    const Eigen::Vector3d& offset)
{
    std::vector<NurbsSurfacePatch3D> patches = source.patches();
    std::vector<int> components(
        static_cast<std::size_t>(source.num_patches()), 0);
    std::vector<NurbsPatchEdgeConnection3D> connections =
        source.connections();
    const int patch_offset = source.num_patches();
    for (int patch = 0; patch < source.num_patches(); ++patch) {
        patches.push_back(translated_patch(source.patch(patch), offset));
        components.push_back(1);
    }
    for (auto connection : source.connections()) {
        connection.first.patch += patch_offset;
        connection.second.patch += patch_offset;
        connections.push_back(connection);
    }
    return NurbsSurfaceModel3D(
        std::move(patches), std::move(components),
        std::move(connections));
}
```

Append original patches/connections first and translated patches/connections
second. Increase both patch indices in every copied connection by the original
patch count. On

```cpp
CartesianGrid3D({-0.7,-0.8,-0.7}, {0.05,0.05,0.05},
                {64,30,30}, DofLayout3D::Node)
```

require `grid.index(14,10,14)` at `(0,-0.3,0)` has label `1`,
`grid.index(46,10,14)` at `(1.6,-0.3,0)` has label `2`, and
`grid.index(30,10,14)` at `(0.8,-0.3,0)` has label `0`.

- [ ] **Step 3: Run RED**

Expected: compilation fails because `NurbsCartesianDomain3D` is absent.

- [ ] **Step 4: Add the public grid-domain API**

```cpp
struct NurbsCartesianDomainOptions3D {
    bool use_triangle_seeds = true;
};

struct NurbsCartesianDomainDiagnostics3D {
    int nurbs_patch_count = 0;
    int bezier_element_count = 0;
    int acceleration_leaf_count = 0;
    std::size_t candidate_grid_edge_count = 0;
    std::array<std::size_t, 3> barrier_edge_counts{{0, 0, 0}};
    int grid_component_count = 0;
    int box_exterior_component_count = 0;
    int representative_query_count = 0;
    std::vector<std::size_t> component_sizes;
    NurbsSurfaceIntersectionDiagnostics3D intersections;
    double maximum_root_residual = 0.0;
};

class NurbsCartesianDomain3D {
public:
    NurbsCartesianDomain3D(
        const CartesianGrid3D& grid,
        NurbsSurfaceModel3D model,
        NurbsCartesianDomainOptions3D options = {});

    int label(int node) const;
    bool has_barrier_between(int node_a, int node_b) const;
    const NurbsSurfaceCrossing3D& crossing_between(
        int node_a, int node_b) const;
    const std::vector<int>& labels() const;
    const NurbsCartesianDomainDiagnostics3D& diagnostics() const;
    const NurbsAabb3D& surface_bounds() const;
    double geometry_tolerance() const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};
```

Require `DofLayout3D::Node`, positive spacings, and a strict positive surface
margin from all six box faces. Include the offending bounds in
`NURBS surface must lie strictly inside Cartesian box`.

- [ ] **Step 5: Enumerate and solve conservative candidate edges**

Ask the intersector for leaves of maximum extent
`2 * max(hx,hy,hz)`. For each leaf and each axis, enumerate grid edges whose
closed segment AABB overlaps the leaf AABB expanded by `tol_x`.

Construct that intersector with
`NurbsSurfaceIntersectorOptions3D::use_triangle_seeds =
options.use_triangle_seeds`, and define `tol_x` in this section as
`intersector.geometry_tolerance()`.

For an axis coordinate, use:

```cpp
start_min = ceil((lower - origin - tol_x) / h) - 1;
start_max = floor((upper - origin + tol_x) / h);
```

clamped to `[0, dims[axis]-2]`. For each fixed transverse node coordinate use
`ceil((lower-origin-tol_x)/h)` through
`floor((upper-origin+tol_x)/h)`, clamped to its node range.

Encode and deduplicate candidate keys as

```cpp
(std::uint64_t(axis) << 62) | std::uint64_t(start_node)
```

before querying. Use exact barrier storage:

```cpp
x[(k * ny + j) * (nx - 1) + i];
y[(k * (ny - 1) + j) * nx + i];
z[(k * ny + j) * nx + i];
```

Store barriers as `std::uint8_t`. Store accepted crossings in one vector and a
sparse `unordered_map<uint64_t,int>` from edge key to crossing index. A grid
edge not in the candidate set remains unblocked only because conservative
Bézier bounds proved it cannot intersect.

- [ ] **Step 6: Flood-fill node graph components**

Initialize graph-component IDs to `-1`. Use a vector-backed BFS queue and
traverse a six-neighbor edge only if its barrier byte is zero. Record size,
representative node, and box-face contact for every component.

Classify exactly as follows:

```cpp
if (touches_box_boundary) {
    domain_label = 0;
    ++diagnostics.box_exterior_component_count;
} else {
    ++diagnostics.representative_query_count;
    const std::vector<int> containing =
        intersector.containing_components(representative_point);
    if (containing.empty())
        domain_label = 0;
    else if (containing.size() == 1)
        domain_label = containing.front() + 1;
    else
        throw std::runtime_error("overlapping NURBS solid components");
}
```

Map graph-component labels to every node in a final linear pass. Then scan all
neighbor pairs and require every binary label change has a barrier and every
barrier has a crossing record. Require the two endpoints of every barrier to
have opposite binary inside/outside labels; otherwise throw
`NURBS barrier does not separate opposite sides`.

- [ ] **Step 7: Run GREEN and commit**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test --parallel 2
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
git diff --check
git add src/geometry/nurbs_cartesian_domain_3d.hpp src/geometry/nurbs_cartesian_domain_3d.cpp src/CMakeLists.txt apps/native_nurbs_surface_3d_test.cpp
git commit -m "feat: label Cartesian grids from native NURBS barriers"
```

### Task 6: Integrate native crossings into GridPair3D and the 3D solver

**Files:**
- Modify: `src/geometry/grid_pair_3d.hpp`
- Modify: `src/geometry/grid_pair_3d.cpp`
- Modify: `apps/native_nurbs_surface_3d_test.cpp`
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`

**Interfaces:**
- Consumes: `shared_ptr<const NurbsCartesianDomain3D>` from Task 5.
- Produces: NURBS-backed `GridPair3D` labels and `P2CrossingOwner3D` values carrying authoritative native parameters.

- [ ] **Step 1: Add failing GridPair native-route assertions**

Extend the L-prism cropped-grid test:

```cpp
auto domain = std::make_shared<const
    kfbim::geometry3d::NurbsCartesianDomain3D>(
        grid, surface.geometry_model());
const auto triangulation =
    kfbim::geometry3d::triangulate_nurbs_surface_patches_3d(
        surface.patches, h);
kfbim::GridPair3D pair(grid,
                       triangulation.interface,
                       triangulation.geometry_interface,
                       domain);
require(pair.domain_label(grid.index(27,27,4)) == 1,
        "GridPair uses NURBS barrier labels");
```

For every unique binary label-changing grid edge, require the owner has
`status == ExactIntersection`, `nurbs_patch_index >= 0`, finite `(u,v)`, root
residual below the domain tolerance, and a physical point on both the grid
edge and `surface.patches[nurbs_patch_index]`.

- [ ] **Step 2: Run RED**

Expected: compilation fails because the four-argument constructor and native
owner fields do not exist.

- [ ] **Step 3: Extend crossing-owner data and GridPair construction**

Add to `P2CrossingOwner3D`:

```cpp
int nurbs_patch_index = -1;
Eigen::Vector2d nurbs_parameter = Eigen::Vector2d::Zero();
Eigen::Vector3d crossing_point = Eigen::Vector3d::Zero();
Eigen::Vector3d crossing_normal = Eigen::Vector3d::Zero();
int surface_component = -1;
double crossing_residual = 0.0;
```

Add this constructor and accessors:

```cpp
GridPair3D(
    const CartesianGrid3D& grid,
    const Interface3D& correction_interface,
    const Interface3D& crossing_geometry,
    std::shared_ptr<const geometry3d::NurbsCartesianDomain3D> nurbs_domain);

bool has_nurbs_domain() const noexcept;
const geometry3d::NurbsCartesianDomainDiagnostics3D&
    nurbs_domain_diagnostics() const;
```

The existing constructors delegate with a null domain and preserve legacy
behavior.

- [ ] **Step 4: Route labels and crossing owners through native records**

When the native domain is present:

- verify its label count equals `grid.num_dofs()`;
- copy its labels into `domain_label_vec`;
- do not build the complete triangle `crossing_mesh`/`crossing_tree`;
- do not call `build_p2_nearest_center_domain_labels`;
- retain P2 center hashes and all projection/restrict support;
- for `p2_crossing_owner_between(a,b)`, require a stored native crossing when
  the endpoint binary labels differ and copy all native fields into the owner;
- recompute `owner.edge_parameter` from `owner.crossing_point` projected onto
  the requested directed segment `grid.coord(a) -> grid.coord(b)`, rather than
  copying the domain record's positive-axis parameter;
- retain the existing endpoint-based P2 center selection only for the
  correction center fields;
- never return `GapFallback` on the native route.

After labels are installed, scan all neighbor pairs and throw
`label-changing edge has no native NURBS crossing` if the invariant fails.

- [ ] **Step 5: Use native `(patch,u,v)` in the harmonic-jet app**

In `PanelCenterHarmonicJetKFBI3D::surface_dof_for_crossing()`, make the native
route authoritative:

```cpp
int patch = owner.nurbs_patch_index;
Eigen::Vector2d uv = owner.nurbs_parameter;
if (patch < 0) {
    // Existing triangle/barycentric legacy fallback remains unchanged.
}
const auto candidates = app3d::parameter_dof_candidates_2x2(
    native_surface_, cloud_, patch, uv.x(), uv.y());
```

Keep the existing normal filter and Euclidean nearest selection over those
four candidates. In `build_crossing_rows()`, use `owner.crossing_point` for a
native owner and use edge interpolation only for the legacy route.

In `run_readiness_case()`, build one shared native domain from
`geometry.native_surface.geometry_model()` and pass it to the new GridPair
constructor. Compute box margin from `domain->surface_bounds()` rather than
triangle sample points.

- [ ] **Step 6: Run GREEN and commit**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test neumann_exterior_zero_trace_3d --parallel 2
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe l_prism 16
git diff --check
git add src/geometry/grid_pair_3d.hpp src/geometry/grid_pair_3d.cpp apps/native_nurbs_surface_3d_test.cpp apps/neumann_exterior_zero_trace_3d.cpp
git commit -m "feat: use native NURBS crossings in GridPair3D"
```

Expected: both 3D formulations converge and every native crossing owner is
exact.

### Task 7: Expose diagnostics and enforce CI smoke coverage

**Files:**
- Modify: `apps/neumann_exterior_zero_trace_3d.cpp`
- Modify: `.github/workflows/build-and-smoke.yml`

**Interfaces:**
- Consumes: `GridPair3D::nurbs_domain_diagnostics()`.
- Produces: stable console/CSV evidence and Linux smoke assertions for the production route.

- [ ] **Step 1: Add failing CI output assertions**

After `three_d_output` is captured, add:

```bash
printf '%s\n' "$three_d_output" | grep -Fq \
  'domain_label_mode=nurbs_barrier_components'
printf '%s\n' "$three_d_output" | grep -Fq \
  'gap_crossings=0'
printf '%s\n' "$three_d_output" | grep -Fq \
  'triangle_fallback_crossings=0'
```

Run the current N=16 output locally and verify RED because the new stable line
is not printed yet.

- [ ] **Step 2: Add exact readiness fields**

Extend `ReadinessResult` and `geometry_readiness.csv` with:

```text
nurbs_patches,bezier_elements,acceleration_leaves,candidate_grid_edges,
triangle_seed_hits,triangle_seed_misses_recovered,subdivision_boxes,
newton_attempts,newton_iterations,seam_deduplications,
barrier_x,barrier_y,barrier_z,grid_components,box_exterior_components,
representative_queries,nurbs_geometry_tolerance,nurbs_root_residual_max,
triangle_fallback_crossings
```

Populate every field from the native domain diagnostics and crossing-owner
scan. Do not recompute intersections in the app.

- [ ] **Step 3: Print one stable line per geometry/level**

Use this exact prefix and field order:

```text
domain_label_mode=nurbs_barrier_components barriers=x/y/z components=c/e queries=q root_residual=r gap_crossings=g triangle_fallback_crossings=f
```

Retain existing label mismatch and exact/gap/endpoint owner output. On the
native route, count `ExactIntersection` owners as exact, require gap and
triangle fallback counts zero, and throw if they are nonzero.

- [ ] **Step 4: Verify all geometries and commit**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test neumann_exterior_zero_trace_3d --parallel 2
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
& .\build\apps\Release\neumann_exterior_zero_trace_3d.exe all 16
git diff --check
git add apps/neumann_exterior_zero_trace_3d.cpp .github/workflows/build-and-smoke.yml
git commit -m "test: cover native NURBS 3d domain labels"
```

Expected: torus, cylinder, and L-prism complete both formulations; every
stable line reports `gap_crossings=0 triangle_fallback_crossings=0`.

### Task 8: Run the L-prism N32/64/128 numerical acceptance study

**Files:**
- Create: `docs/superpowers/results/2026-07-21-3d-nurbs-native-domain-labeling.md`
- Outputs: `output/neumann_exterior_zero_trace_3d/same_patch/v32_n20/*.csv`

**Interfaces:**
- Consumes: the completed native NURBS route and cubic `same_patch 32/20` configuration.
- Produces: numerical acceptance evidence and the decision whether `N=256` is allowed.

- [ ] **Step 1: Run the exact three-level command**

```powershell
$env:KFBIM_3D_CAUCHY_POLICY='same_patch'
$env:KFBIM_3D_CAUCHY_VALUE_COUNT='32'
$env:KFBIM_3D_CAUCHY_NORMAL_COUNT='20'
try {
    & .\build\apps\Release\neumann_exterior_zero_trace_3d.exe `
        l_prism 32 64 128
} finally {
    Remove-Item Env:KFBIM_3D_CAUCHY_POLICY,
                Env:KFBIM_3D_CAUCHY_VALUE_COUNT,
                Env:KFBIM_3D_CAUCHY_NORMAL_COUNT `
        -ErrorAction SilentlyContinue
}
```

- [ ] **Step 2: Enforce the CSV gates**

```powershell
$dir='output\neumann_exterior_zero_trace_3d\same_patch\v32_n20'
$g=Import-Csv "$dir\geometry_readiness.csv"
$d=Import-Csv "$dir\dirichlet_normal_results.csv"
if($g.Count -ne 3 -or $d.Count -ne 3) {
    throw 'acceptance CSVs must each contain N=32,64,128'
}
if(($g.N -join ',') -ne '32,64,128' -or
   ($d.N -join ',') -ne '32,64,128') {
    throw 'acceptance CSV levels or ordering are not N=32,64,128'
}
if(($g | Where-Object {[int]$_.label_mismatches -ne 0}).Count) {
    throw 'L-prism label mismatch remains'
}
if(($g | Where-Object {[int]$_.gap_crossings -ne 0 -or
                                [int]$_.triangle_fallback_crossings -ne 0}).Count) {
    throw 'non-native crossing fallback remains'
}
if(($g | Where-Object {
        [double]$_.nurbs_root_residual_max -gt
        [double]$_.nurbs_geometry_tolerance}).Count) {
    throw 'accepted NURBS root exceeds geometry tolerance'
}
if(($d | Where-Object {[int]$_.converged -ne 1}).Count) {
    throw 'Dirichlet solve did not converge'
}
if([double]$d[2].density_order_linf -lt 2.0 -or
   [double]$d[2].interior_order_linf -lt 2.0) {
    throw 'N64-to-N128 order gate failed'
}
if([double]$d[1].density_linf -ge [double]$d[0].density_linf -or
   [double]$d[2].density_linf -ge [double]$d[1].density_linf -or
   [double]$d[1].interior_linf -ge [double]$d[0].interior_linf -or
   [double]$d[2].interior_linf -ge [double]$d[1].interior_linf) {
    throw 'Dirichlet errors did not decrease on both refinements'
}
```

- [ ] **Step 3: Record numerical and geometry-query results**

The result document includes:

- topology, Bézier-element, acceleration-leaf, and candidate-edge counts;
- triangle-seed hits and misses recovered by NURBS isolation;
- subdivision/Newton work and maximum accepted root residual;
- x/y/z barriers, graph components, and representative queries;
- label mismatch and exact/gap/fallback crossing counts;
- Dirichlet and Neumann GMRES iterations, residuals, and runtime;
- density/interior errors and observed orders;
- comparison with the failed nearest-normal `N=128` result: `15` label
  mismatches, `74` gap crossings, Dirichlet density `6.2212368e-6`, and `42`
  iterations;
- `N=256 allowed: yes/no`, derived only from the gates above.

- [ ] **Step 4: Run complete verification and request review**

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test neumann_exterior_zero_trace_3d --parallel 2
& .\build\apps\Release\native_nurbs_surface_3d_test.exe
ctest --test-dir build -C Release --output-on-failure
git diff --check
git status --short --branch
```

Request an independent review of the complete implementation range. Fix every
Critical or Important finding and rerun the covering verification.

- [ ] **Step 5: Commit the result document**

```powershell
git add docs/superpowers/results/2026-07-21-3d-nurbs-native-domain-labeling.md
git commit -m "test: verify native NURBS 3d domain labels"
```

Do not run `N=256` unless every Task 8 gate passes.
