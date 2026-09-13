# 三维几何模型

这个模块集中管理算例的实体构造。每个模型的控制点、权重、patch 顺序和拓扑连接在本模块定义；网格三角化、交点认证、密度离散与求解在调用方完成。

## 从哪里开始

- `catalog_3d.hpp`：枚举 NURBS 模型，按明确的模型 ID 构造 `NurbsSurfaceModel3D`。
- `native_surface_3d.hpp`：保留 Native 构造接口；`native` 指 NURBS patch 自身的 `(u,v)` 参数，可在这些参数上构造密度空间。
- `bicubic_nurbs_models_3d.hpp`：构造 `BicubicNurbsModel3D`，并返回与 patch ID 对齐的 `SurfaceAnalysisChart3D` 参数元数据。
- `industrial/models_3d.hpp`：工业模型的独立构造接口。
- `analytic/cap_surface_geometry_3d.hpp`：椭球/花形的解析参数面、隐式函数、法向和内外判定。
- `rigid_transform_3d.hpp`：共用刚体变换。

```cpp
#include "src/geometry/models3d/catalog_3d.hpp"

const auto model =
    kfbim::geometry3d::make_geometry_model_3d("bicubic_nurbs/u_prism");
const auto topology = model.validate_closed();
// 后续再把 model 交给网格/求交预处理；模型本身不绑定 N 或 h。
```

需要独立的曲面分析参数时使用专门工厂：

```cpp
#include "src/geometry/models3d/bicubic_nurbs_models_3d.hpp"

using namespace kfbim::app3d;
const BicubicNurbsModel3D geometry =
    make_bicubic_nurbs_model_3d(BicubicNurbsShape3D::UPrism);
// geometry.surface 与 SurfaceAnalysisChart3D 使用同一套 patch 编号；
// SurfaceAnalysisChartKind3D 明确每个 chart 的实际参数含义。
```

Native 构造接口及既有研究入口继续可用。模型类型当前位于 `kfbim::app3d` 命名空间，实现属于纯几何模块。旧 support 和 industrial 头文件转发几何类型，support 的 DOF/Cauchy 邻域操作继续留在原模块。

## 按几何体管理

| 目录 | 模型 | 构造方式 |
| --- | --- | --- |
| `native/` | torus、hollow_cylinder、l_prism、u_prism | 使用 NURBS 自身参数；L/U 柱平面片为双线性 patch，圆环与空心圆柱的圆周为精确有理 NURBS |
| `bicubic_nurbs/` | box、solid_cylinder、l_prism、u_prism | 提供独立的曲面分析 chart；平面实体复用拉伸和封面，实心圆柱保留精确有理几何与角度 chart 的区别 |
| `industrial/` | sleeve、bracket、flange、impeller | 每个实体一个 cpp，保留原来的闭合边匹配与构造顺序 |
| `analytic/` | ellipsoid、flower | 共享纯几何参数与查询，由 cap 密度和应用复用 |

`native` 与 `bicubic_nurbs` 是不同的模型配方。例如 `native/u_prism` 有 18 个 patch，`bicubic_nurbs/u_prism` 有 22 个 patch；两者尺寸、分片、编号及参数用途都不同，不能仅凭“U 柱”名称互换。`bicubic_nurbs` 的 catalog ID 为 `bicubic_nurbs/box`、`bicubic_nurbs/solid_cylinder`、`bicubic_nurbs/l_prism` 和 `bicubic_nurbs/u_prism`。

`SurfaceAnalysisChart3D` 描述求值与密度分析所用的参数坐标，`SurfaceAnalysisChartKind3D` 区分仿射平面、圆柱侧面和圆柱盖环等实际 chart。名称中的 bicubic 指该模型族的 NURBS 表示约定，不表示每个分析 chart 都是双三次多项式；圆柱侧面使用角度 chart。

NURBS 目录中的模型可供几何检查和展示使用；某个密度布局或修正后端是否支持它，由对应算例适配器决定。解析 cap 通过自身强类型接口构造，不伪装成 NURBS patch 集合。

## 添加一个几何体

目录、类型、函数和模型 ID 应描述几何形状、表示方式或实际职责。参考脚本编号、实验批次等来源信息写在研究记录中，不作为模型名称。

1. 在适合的模型族目录新增独立 `xxx_3d.cpp` 和必要的强类型参数/工厂声明；已有几何的默认参数代表既定基准，保持原义。
2. 复用已有构造工具。按稳定顺序追加 patches，给出名称、外法向方向及完整的拓扑连接。长边与多条短边相接时保留区间连接，不能强行改成整边一对一。
3. 在 `sources.cmake` 增加实现源文件；需加入 NURBS 模型目录时，在 `catalog_3d.cpp` 登记一个唯一 ID 与工厂。
4. 加几何测试：解析形状/边界位置、法向、闭合拓扑以及对科研有意义的编号/参数约定。无需运行完整 PDE 才能检查实体。
5. 在算例中调用工厂并设置边界数据。若使用特殊 analysis chart，再提供显式适配；制造解、边界均值、网格尺度和 GMRES 不放进几何工厂。

首次拆分保留各模型族自己的拓扑匹配容差与算法。进一步合并构造原语时，应单独验证参数方向和 patch 输出，避免把组织调整变成数值算法变化。

## 构建

本模块随 `kfbim_core` 的三维部分构建，源清单只有 `sources.cmake` 一处。工业示例只消费该清单中的工业子集，继续支持 `examples/industrial_nurbs` 的 Eigen-only 独立构建。

迁移等价性审阅、几何测试和已知认证失败见 [验证报告](../../../docs/Geometry_Model_Module_Validation.md)。
