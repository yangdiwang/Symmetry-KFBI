# NURBS–Bézier 3D 模块职责拆分设计

## 背景

当前 `rational_bezier_surface_3d.hpp/.cpp` 同时承担三种职责：

1. 有理 Bézier 单元的数据表示与基本操作；
2. 从原生 NURBS patch 精确提取 Bézier 单元；
3. 按空间尺度递归细分 Bézier 单元。

文件名只表达了数据类型，无法反映其中的提取和细分算法。本次重构按功能拆分代码，不改变公开数据含义、数值算法或计算结果。

## 目标

- 每个文件只承担一种明确职责。
- 求交代码仅依赖 Bézier 单元表示，不依赖 NURBS 提取实现。
- NURBS 提取代码继续使用齐次 knot 插入，保持几何精确不变。
- 现有调用保持兼容。
- 原样保留当前工作树中尚未提交的 Task 3 求交修改。

## 非目标

- 不修改 Bézier 提取、求值、AABB 或 de Casteljau 算法。
- 不改变 `RationalBezierElement3D` 的字段和函数签名。
- 不修改 Task 3 的求交逻辑。
- 不新增数值功能或精度策略。

## 文件与职责

### `rational_bezier_element_3d.hpp/.cpp`

包含：

- `RationalBezierElement3D`；
- 参数区间访问；
- 控制点访问；
- 有理 Bézier 求值；
- 保守 AABB；
- `split_u()` 和 `split_v()`。

该模块只依赖 NURBS 几何中的 `NurbsAabb3D` 类型，不依赖提取器。

### `nurbs_bezier_extraction_3d.hpp/.cpp`

公开：

- `extract_rational_bezier_elements_3d()`。

内部包含：

- 齐次控制网构造；
- 单次 knot 插入；
- 活动 knot 细化；
- U/V 方向的精确 span 提取。

该模块依赖 `NurbsSurfaceModel3D` 和 `RationalBezierElement3D`。

### `rational_bezier_subdivision_3d.hpp/.cpp`

公开：

- `subdivide_rational_bezier_elements_to_extent_3d()`。

内部包含：

- 控制网空间变化量比较；
- 细分方向选择；
- 深度限制和叶单元收集。

该模块只依赖 `RationalBezierElement3D`。

### 兼容头文件

保留 `rational_bezier_surface_3d.hpp`，但将其改为仅转发包含：

```cpp
#include "nurbs_bezier_extraction_3d.hpp"
#include "rational_bezier_element_3d.hpp"
#include "rational_bezier_subdivision_3d.hpp"
```

删除原 `rational_bezier_surface_3d.cpp`，避免继续形成混合职责实现文件。

## 依赖方向

```text
NurbsSurfaceModel3D
        │
        ▼
nurbs_bezier_extraction_3d
        │
        ▼
rational_bezier_element_3d ◄── rational_bezier_subdivision_3d
        ▲
        │
nurbs_bezier_intersection_3d
```

`nurbs_bezier_intersection_3d.hpp` 改为直接包含 `rational_bezier_element_3d.hpp`，从而不再间接依赖提取和批量细分算法。

## 迁移与构建

- `src/CMakeLists.txt` 删除旧 `.cpp`，加入三个新 `.cpp`。
- 内部调用改用最窄的直接头文件。
- 测试文件显式包含元素、提取和细分三个头文件。
- 旧聚合头文件继续支持现有外部调用。
- 对已修改的 Task 3 文件只调整必要的 include，不改动其未提交算法内容。

## 验证

重构必须满足：

1. 全量 Release 构建通过；
2. `native_nurbs_surface_3d_test` 通过；
3. NURBS 提取、非夹持 knot、AABB、细分和求交测试结果不变；
4. `git diff --check` 通过；
5. 提交只包含模块拆分文件、必要 include/CMake 修改和本设计说明，不吞并当前未提交的 Task 3 算法改动。
