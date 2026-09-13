# 三维几何模型构造模块

## 目标

将三维算例的实体构造集中到 `src/geometry/models3d/`，每种实体有独立实现，公共 patch/拓扑构造复用；应用只选择模型并装配网格与求解。当前以三维为范围，二维不在这轮迁移中。

## 边界

- 模型提供 NURBS patches、稳定 patch 名称/编号、拓扑、解析域判定（已有时）及几何参数元数据。
- NURBS 几何评价/求交算法继续使用现有 `src/geometry` 基础设施。
- CartesianGrid、按 h 三角化、GridPair、密度空间、制造解、边界均值、GMRES 与输出不属于实体模型。
- 使用 NURBS 自身参数的模型与双三次 NURBS 模型保留各自的尺寸及参数化；双线性 U 柱的 18 patches 与双三次 U 柱的 22 patches 不合并。
- 双三次 NURBS 模型保留 analysis/native 参数区别；几何定义产物带 analysis metadata，算例层设置刚体变换与制造解。
- 工业 NURBS 模型迁入同一模块，旧头文件保留转发，独立 Eigen-only 示例继续构建。
- general-cap 的椭球/花形实体提取为共享解析几何，应用与密度模块复用；密度离散与网格求交装配保持原有职责。

## 接口与依赖

既有算例入口通过适配调用新几何模块。纯几何头文件不依赖 `src/support`、Poisson 或密度空间。模型实现随 `kfbim_core` 构建；模块目录提供 README 和集中源码清单。

命名按几何含义与参数化方式确定，避免来源编号。双三次 NURBS 模型位于 `bicubic_nurbs/`，工厂为 `make_bicubic_nurbs_model_3d(BicubicNurbsShape3D)`，形状枚举明确区分 `Box`、`SolidCylinder`、`LPrism`、`UPrism`；分析坐标元数据使用 `SurfaceAnalysisChart3D`。

新增 NURBS 模型目录 API `available_geometry_models_3d()` 与 `make_geometry_model_3d(id)`，返回几何目录描述和 `NurbsSurfaceModel3D`，用于独立几何检查与发现。目录不承诺每个求解路线都支持所有几何。

## 验证

检查目录工厂、不同模型族的 patch 布局、拓扑闭合与非法名称。保留并运行现有 Trace93 几何/密度、TraceFirst 几何、Native U、工业模型、shared-field fixture 相关测试。编译相关应用及独立工业示例。迁移前已有 torus N64 rigid-transform correction-safe 失败单独记录，不改变数值认证算法来掩盖该失败。
