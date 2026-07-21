# NURBS–Bézier 3D Module Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the mixed NURBS-to-Bézier implementation into focused element, extraction, and subdivision modules without changing any numerical behavior.

**Architecture:** `RationalBezierElement3D` and its intrinsic operations live in the element module. Exact homogeneous knot insertion and span extraction live in the extraction module, while extent-driven recursive subdivision lives in the subdivision module. The old header remains as a forwarding compatibility layer, and the old mixed `.cpp` is removed.

**Tech Stack:** C++17, Eigen, CMake, MSVC Release build.

## Global Constraints

- Do not alter any numerical formula, tolerance, public function signature, field, exception text, or depth limit.
- Treat Task 3 commit `539c8ff` as an immutable baseline; do not alter its intersection algorithms or tests except for the test file's include migration.
- Keep `rational_bezier_surface_3d.hpp` as a source-compatible forwarding header.
- Delete `rational_bezier_surface_3d.cpp` after all implementations have moved.
- Work directly on `main`, as previously requested by the user.

---

### Task 1: Split the mixed module by responsibility

**Files:**
- Create: `src/geometry/rational_bezier_element_3d.hpp`
- Create: `src/geometry/rational_bezier_element_3d.cpp`
- Create: `src/geometry/nurbs_bezier_extraction_3d.hpp`
- Create: `src/geometry/nurbs_bezier_extraction_3d.cpp`
- Create: `src/geometry/rational_bezier_subdivision_3d.hpp`
- Create: `src/geometry/rational_bezier_subdivision_3d.cpp`
- Modify: `src/geometry/rational_bezier_surface_3d.hpp`
- Delete: `src/geometry/rational_bezier_surface_3d.cpp`
- Modify: `src/geometry/nurbs_bezier_intersection_3d.hpp`
- Modify: `src/CMakeLists.txt`
- Modify: `apps/native_nurbs_surface_3d_test.cpp` (include block only)

**Interfaces:**
- Consumes: `NurbsSurfaceModel3D`, `NurbsSurfacePatch3D`, and `NurbsAabb3D` from `nurbs_surface_model_3d.hpp`.
- Produces: the unchanged `RationalBezierElement3D` API, `extract_rational_bezier_elements_3d(const NurbsSurfaceModel3D&)`, and `subdivide_rational_bezier_elements_to_extent_3d(const std::vector<RationalBezierElement3D>&, double)`.

- [ ] **Step 1: Record the characterization baseline**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test
.\build\apps\Release\native_nurbs_surface_3d_test.exe
git status --short
```

Expected: the target builds, the executable prints `native NURBS model tests passed`, and the working tree is clean before new files are created.

- [ ] **Step 2: Create the element interface**

Create `src/geometry/rational_bezier_element_3d.hpp` with the unchanged element API:

```cpp
#pragma once

#include "nurbs_surface_model_3d.hpp"

#include <Eigen/Dense>

#include <array>
#include <vector>

namespace kfbim::geometry3d {

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

} // namespace kfbim::geometry3d
```

- [ ] **Step 3: Move intrinsic element implementation unchanged**

Create `src/geometry/rational_bezier_element_3d.cpp`. Move these private helpers and methods verbatim from the old implementation:

```text
midpoint_if_representable
expected_control_count
validate_element
evaluate_curve
split_curve
RationalBezierElement3D::u0/u1/v0/v1
RationalBezierElement3D::control
RationalBezierElement3D::evaluate
RationalBezierElement3D::bounds
RationalBezierElement3D::split_u
RationalBezierElement3D::split_v
```

The new implementation starts with:

```cpp
#include "rational_bezier_element_3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
```

Do not copy knot insertion, model extraction, projected variation, or batch extent subdivision into this file.

- [ ] **Step 4: Create the exact extraction module**

Create `src/geometry/nurbs_bezier_extraction_3d.hpp`:

```cpp
#pragma once

#include "nurbs_surface_model_3d.hpp"
#include "rational_bezier_element_3d.hpp"

#include <vector>

namespace kfbim::geometry3d {

std::vector<RationalBezierElement3D>
extract_rational_bezier_elements_3d(const NurbsSurfaceModel3D& model);

} // namespace kfbim::geometry3d
```

Create `src/geometry/nurbs_bezier_extraction_3d.cpp`. Move these functions and aliases verbatim from the old implementation:

```text
HomogeneousNet
validate_net
insert_knot_u_once
transpose
refine_active_knots_u
nonzero_spans
extract_rational_bezier_elements_3d
```

The old extraction function's final private `validate_element(element)` call
becomes `(void)element.bounds()` so it invokes the same element validation
through the unchanged public API without exposing a new cross-module helper.

Use these includes:

```cpp
#include "nurbs_bezier_extraction_3d.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>
```

No knot insertion formula, target multiplicity, span indexing, or exception text may change.

- [ ] **Step 5: Create the extent-subdivision module**

Create `src/geometry/rational_bezier_subdivision_3d.hpp`:

```cpp
#pragma once

#include "rational_bezier_element_3d.hpp"

#include <vector>

namespace kfbim::geometry3d {

std::vector<RationalBezierElement3D>
subdivide_rational_bezier_elements_to_extent_3d(
    const std::vector<RationalBezierElement3D>& elements,
    double maximum_extent);

} // namespace kfbim::geometry3d
```

Create `src/geometry/rational_bezier_subdivision_3d.cpp`. Move
`has_representable_midpoint`, `projected_control_variation`, and
`subdivide_rational_bezier_elements_to_extent_3d`. Replace the private
`validate_element(element)` call with `(void)element.bounds()` so eager input
validation is preserved through the unchanged element API. Preserve the exact
depth-48 failure text and the current U/V direction-selection rule.

Use these includes:

```cpp
#include "rational_bezier_subdivision_3d.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>
```

- [ ] **Step 6: Replace the mixed interface with compatibility forwarding**

Replace `src/geometry/rational_bezier_surface_3d.hpp` with:

```cpp
#pragma once

#include "nurbs_bezier_extraction_3d.hpp"
#include "rational_bezier_element_3d.hpp"
#include "rational_bezier_subdivision_3d.hpp"
```

Delete `src/geometry/rational_bezier_surface_3d.cpp` after confirming every function in it exists exactly once in one new `.cpp`.

- [ ] **Step 7: Narrow dependencies and update CMake**

Change `src/geometry/nurbs_bezier_intersection_3d.hpp` from:

```cpp
#include "rational_bezier_surface_3d.hpp"
```

to:

```cpp
#include "rational_bezier_element_3d.hpp"
```

Change the test include block to:

```cpp
#include "src/geometry/nurbs_bezier_extraction_3d.hpp"
#include "src/geometry/rational_bezier_element_3d.hpp"
#include "src/geometry/rational_bezier_subdivision_3d.hpp"
```

In `src/CMakeLists.txt`, replace:

```cmake
geometry/rational_bezier_surface_3d.cpp
```

with:

```cmake
geometry/nurbs_bezier_extraction_3d.cpp
geometry/rational_bezier_element_3d.cpp
geometry/rational_bezier_subdivision_3d.cpp
```

- [ ] **Step 8: Run focused and full verification**

Run:

```powershell
cmake --build build --config Release --target native_nurbs_surface_3d_test
.\build\apps\Release\native_nurbs_surface_3d_test.exe
cmake --build build --config Release --parallel 2
git diff --check
```

Expected: both builds exit zero, the test prints `native NURBS model tests passed`, and `git diff --check` emits no output.

- [ ] **Step 9: Verify behavior-preserving ownership and dirty-worktree isolation**

Run:

```powershell
rg -n "RationalBezierElement3D::|insert_knot_u_once|extract_rational_bezier_elements_3d|subdivide_rational_bezier_elements_to_extent_3d" src/geometry
git diff 539c8ff -- src/geometry/nurbs_bezier_intersection_3d.cpp
git diff -- apps/native_nurbs_surface_3d_test.cpp
```

Expected:

- every moved implementation has exactly one definition in its responsibility-specific file;
- `nurbs_bezier_intersection_3d.cpp` has no changes relative to Task 3 commit `539c8ff`;
- the test file contains all Task 3 tests unchanged plus only the three-line include migration from this refactor.

- [ ] **Step 10: Stage only the refactor and commit**

Stage the responsibility split, CMake update, compatibility header, narrow include changes, and the test include migration:

```powershell
git add -- apps/native_nurbs_surface_3d_test.cpp src/CMakeLists.txt src/geometry/nurbs_bezier_extraction_3d.hpp src/geometry/nurbs_bezier_extraction_3d.cpp src/geometry/rational_bezier_element_3d.hpp src/geometry/rational_bezier_element_3d.cpp src/geometry/rational_bezier_subdivision_3d.hpp src/geometry/rational_bezier_subdivision_3d.cpp src/geometry/rational_bezier_surface_3d.hpp src/geometry/rational_bezier_surface_3d.cpp src/geometry/nurbs_bezier_intersection_3d.hpp
git diff --cached --check
git diff --cached --name-only
git commit -m "refactor: split NURBS Bezier geometry modules"
```

Before committing, the staged diff for `apps/native_nurbs_surface_3d_test.cpp` must contain only the include replacement, and `src/geometry/nurbs_bezier_intersection_3d.cpp` must be absent from the staged file list. After committing, `git status --short` must be clean.
