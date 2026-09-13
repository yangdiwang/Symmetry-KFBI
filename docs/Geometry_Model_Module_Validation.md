# 三维几何模型模块验证（2026-09-13）

本次调整将实体构造移入 `src/geometry/models3d`，保持原有默认尺寸、patch 编号、拓扑与参数约定。范围为三维 Native、Trace93、工业 NURBS 及解析 cap；二维与数值求解算法没有纳入本轮重构。

## 迁移等价性审阅

- Native 的 57 处拓扑连接按原有顺序比较一致，包含 L/U 的分段边区间、方向和 G1 标记；patch 命名、面积公式、inside 谓词及默认尺寸保持一致。
- 刚体变换函数体除 include 外与迁移前逐行一致；旧头文件转发到新模块。
- Trace93 保留 box、L、U、cylinder 的配方、patch 顺序及 analysis/native 参数区别。算例通过公开几何/chart 接口获取元数据，不依赖内部构造工具。
- 工业模型实现除 include/路径外保持一致，独立示例从统一源码清单取得工业子集。
- 解析 cap 保留 14 个 chart 的顺序和公式。密度空间、general-cap 应用共享纯几何查询，制造解、密度约束和网格求交装配留在原层。
- 模型模块未发现对 `src/support`、密度或求解器的 include 依赖。

## 测试范围

新增模型目录测试检查明确的模型 ID、描述、关键模型的 patch 数、闭合拓扑及未知 ID 拒绝。

新增解析几何测试检查椭球/花形全部 14 个参数片上的 level set、单位法向、法切正交及 `locate → evaluate` 回映，并以有限差分检查代表性参数片的切向。刚体测试使用非零旋转中心、旋转和平移。

集成测试还覆盖现有 Trace93 几何/密度、TraceFirst 几何、Native 几何与 U 柱以及 Shared Field 固定参考数据。测试通过不等同于重新完成所有网格层级的 PDE 收敛实验。

## 已确认结果

- 新测试首先因缺少预期的新几何头文件编译失败，之后实现模型 API。
- 九个相关测试目标均编译成功，分两组运行后共七项通过、两项失败，结果如下。
- `kfbi_trace93_study_3d`、`kfbi_trace_first_study_3d`、`kfbi_general_cap_exterior_trace_3d`、`neumann_exterior_zero_trace_3d` 均编译成功。
- 独立工业示例的 gallery 和 test 构建成功；CTest 1/1 通过。

| 测试 | 结果 |
| --- | --- |
| `geometry_model_catalog_3d_test` | 通过 |
| `cap_surface_geometry_3d_test` | 通过，含两个形状的全部 14 个参数片 |
| `trace_first_case_3d_test` | 通过 |
| `trace93_case_3d_test` | 通过 |
| `trace93_density_layout_3d_test` | 通过 |
| `u_prism_geometry_3d_test` | 通过 |
| `shared_field_python_fixture_3d_test` | 通过 |
| `native_nurbs_surface_3d_test` | torus N=128 ambiguous-edge 认证失败；原始工厂独立对照复现相同失败 |
| `native_nurbs_surface_transform_3d_test` | 与迁移前相同的 rotated torus N=64 correction-safe 认证失败 |

迁移前执行已有几何测试时，`native_nurbs_surface_transform_3d_test` 已在 rotated torus N=64 的 domain-edge correction-safe 认证处失败。该问题单独记录，不通过修改数值认证算法消除。

Native 全量测试在迁移前尚无可执行程序，因此对新增观测到的 N=128 失败补做了隔离对照：从提交 `12e3f0120448f833d5c02c5c6e16a7c249b11a79` 取得原始 `native_nurbs_surface_3d.cpp`，与未改动的测试源码及同一套 Release support/core/zfft/GMP/MPFR 库链接。新旧程序均以退出码 1 报出完全相同的 `N=128 torus resolves every ambiguous edge, including same-label edges`。Linker map 确认基线工厂来自 `old_native_factory.obj`，没有载入新 Native 模型/工厂对象。该对照排除了本次工厂提取作为此失败的来源；并未验证完整历史 checkout 的所有代码。

隔离对照工程位于 `tmp/geometry_module_baseline`，程序为 `build/Release/native_nurbs_surface_3d_baseline.exe`，链接映射为 `build/native_nurbs_surface_3d_baseline.map`。两项认证失败均保留，未修改认证逻辑。`git diff --check` 通过。

## 复现环境与命令

本机使用 Visual Studio 2017、Release、C++17，三维及实验构建开关均开启。GMP 运行时需要在 PATH 中；本机目录为 `D:/CGAL/CGAL-5.2-beta1/auxiliary/gmp/lib`。编译中的 Eigen/既有头文件 C4819 编码警告未作为本次重构的问题处理。

```powershell
cmake --build build-3d --config Release --target geometry_model_catalog_3d_test cap_surface_geometry_3d_test trace93_case_3d_test trace93_density_layout_3d_test trace_first_case_3d_test native_nurbs_surface_3d_test u_prism_geometry_3d_test native_nurbs_surface_transform_3d_test shared_field_python_fixture_3d_test --parallel 1 -- /p:PreferredToolArchitecture=x64 /nodeReuse:false
ctest --test-dir build-3d -C Release --output-on-failure -R '^(geometry_model_catalog_3d_test|cap_surface_geometry_3d_test|trace93_case_3d_test|trace93_density_layout_3d_test|trace_first_case_3d_test|native_nurbs_surface_3d_test|u_prism_geometry_3d_test|native_nurbs_surface_transform_3d_test|shared_field_python_fixture_3d_test)$'

cmake --build build-3d --config Release --target kfbi_trace93_study_3d kfbi_trace_first_study_3d kfbi_general_cap_exterior_trace_3d neumann_exterior_zero_trace_3d --parallel 1 -- /p:PreferredToolArchitecture=x64 /nodeReuse:false

cmake --build build-industrial-nurbs --config Release --target industrial_nurbs_models_3d_test industrial_nurbs_gallery_3d --parallel 1 -- /p:PreferredToolArchitecture=x64 /nodeReuse:false
ctest --test-dir build-industrial-nurbs -C Release --output-on-failure
```

本机构建日志位于 `tmp/geometry_module_build.log`、`tmp/geometry_module_extra_build.log` 与 `tmp/geometry_module_industrial_build.log`；两组回归结果位于 `tmp/geometry_module_regression.log`、`tmp/geometry_module_extra_regression.log`。

## 几何语义命名修正

按用户反馈，几何模块的目录、类型、工厂及模型 ID 改用真实的几何含义。双三次 NURBS 配方位于 `bicubic_nurbs/`；公开接口为 `BicubicNurbsModel3D` 和 `make_bicubic_nurbs_model_3d(BicubicNurbsShape3D)`。形状用 `Box`、`SolidCylinder`、`LPrism`、`UPrism` 明确区分；分析坐标元数据为 `SurfaceAnalysisChart3D`。目录 ID 对应 `bicubic_nurbs/box`、`bicubic_nurbs/solid_cylinder`、`bicubic_nurbs/l_prism`、`bicubic_nurbs/u_prism`。

新模块不保留来源编号命名的别名；仓内调用和示例已同步更新。既有数值研究入口通过适配将原 CLI 选择器映射到形状枚举。重命名前后的配方对比确认控制网公式、尺寸、patch 顺序、inside、面积和拓扑均未改变。

命名修正后，相关研究程序重新编译成功，以下四项 CTest 全部通过：`geometry_model_catalog_3d_test`、`trace93_case_3d_test`、`trace93_density_layout_3d_test`、`shared_field_python_fixture_3d_test`。本轮未重复执行上文两个已单独对照的 Native 认证测试。日志为 `tmp/geometry_semantic_names_build.log` 和 `tmp/geometry_semantic_names_tests.log`。
