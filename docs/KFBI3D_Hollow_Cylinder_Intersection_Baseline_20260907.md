# 空心圆柱 NURBS 求交基准与热点记录（2026-09-07）

本记录使用当前工作区实际构建和运行的空心圆柱 `baseline`，不是引用旧版本性能数据。算法、容差和默认求交路线未作优化。原始源码仍保留当前目录整理的未提交改动。

**最重要的结论：完整主程序的长耗时发生在 Q27 all-event 支撑线段求交，不能用仅约 1 秒的网格几何域构造来代表。该长耗时阶段的 30 秒主线程采样中，89.83% 的样本包含曲面位置/导数求值，55.00% 包含 `operator new`。应优先处理求值中的重复基函数计算和临时容器分配。**

## 测量范围

首先测量 `NurbsCartesianDomain3D` 的几何域构造，以及复用该 domain 的 `GridPair3D` 构造。几何为内外半径 0.25/0.55、轴心 (0.06,-0.05)、z 范围 [-0.63,0.67] 的原生 NURBS 空心圆柱，计算盒 [-1.5,1.5]^3，h=3/N。每个 N 预热 1 次、正式重复 3 次，报告中位数。

benchmark 的公共几何/三角化准备和完整准确性扫描在这些计时之外；进程总墙钟还包括参考构造、预热和校验，因此不可直接当作一次预处理时间。几何域计时也不等于后续 Q27/Q64 支撑点到曲面迹点的全部预处理时间。

## 原始基准结果

| 项目 | N=32 | N=64 |
|---|---:|---:|
| 几何域构造 | 0.868773 s | 2.554375 s |
| 其中：边求交 | 0.852736 s | 2.486026 s |
| 边求交 / 几何域构造 | 98.154% | 97.324% |
| 求交器构建 | 0.007450 s | 0.028959 s |
| 候选边枚举 | 0.000255 s | 0.000991 s |
| GridPair 构造 | 0.058606 s | 0.757295 s |
| 几何域 + GridPair | 0.927379 s | 3.323510 s |

各列分别取中位数，不能要求各分项中位数之和严格等于总时间中位数。几何域时间的变异系数分别为 1.585% 和 1.255%。最大已确认阶段为边求交；候选枚举只占几何域的约 0.03%/0.04%。

| 边求交诊断计数（每次运行一致） | N=32 | N=64 |
|---|---:|---:|
| 候选元素处理数 | 3,386 | 11,448 |
| 细分盒数 | 14,034 | 46,128 |
| Newton 尝试数 | 10,810 | 36,583 |
| Newton 迭代数 | 105,346 | 347,551 |
| 最近点辅助尝试数 | 5,872 | 19,400 |
| 最近点辅助迭代数 | 70,479 | 206,151 |
| 未决候选 / targeted retry | 0 / 0 | 0 / 0 |

两组程序均 exit 0，各输出 3 条 raw、1 条 summary；未产生 mismatch 文件。最大根残差分别为 7.109e-15、1.873e-15。

## 求交内部归因方法

本机 VS2017 的原生 CPU 采样工具要求真正提升的 Windows 令牌。因此本轮采用独立临时源副本上的聚合计时，不修改生产源码：四个几何编译单元添加 14 个函数 scope、9 个曲面求值 scope，原有静态库提供其余实现。

scope 使用线程局部栈，记录调用数、包含子调用的时间和扣除子调用的时间。报告内部占比只使用后者，避免将 Newton/最近点内部的曲面求值重复累计。启用区间严格限定在 domain 构造；参考构造、预热和 3 次正式构造分别保存，内部汇总仅取正式构造。

`surface_eval` 包含曲面位置和导数求值，其内部仍包含基函数计算、临时分配、有理加权与导数装配，不能把它直接称为纯基函数时间。内部计时还覆盖 domain 中少量代表点分类射线，因此其调用数与仅限网格边的诊断数略有差别。细分计时也包含几何域构造时触发的少量预细分。

已验证 N32/N64 的插桩运行与原始运行：每份 raw CSV 共 154 列，排除耗时列后，所有 3 次重复的字段完全一致。

### 求交内部结果

| 内部工作（扣除已单独计时的子调用） | N32 中位秒数 | N32 占比 | N64 中位秒数 | N64 占比 |
|---|---:|---:|---:|---:|
| 曲面位置与导数求值 | 0.674358 | 81.83% | 2.126107 | 81.30% |
| Newton 其余工作 | 0.055254 | 6.70% | 0.180588 | 6.97% |
| 最近点辅助其余工作 | 0.039402 | 4.78% | 0.121044 | 4.63% |
| 单个元素求交其余工作 | 0.015396 | 1.86% | 0.050128 | 1.92% |
| 种子选择与三角形初值 | 0.010826 | 1.31% | 0.035966 | 1.38% |
| 唯一根认证 | 0.008027 | 0.97% | 0.029738 | 1.15% |
| 解析/重合检查 | 0.008753 | 1.05% | 0.026738 | 1.02% |
| 查询层其余工作 | 0.007253 | 0.88% | 0.025417 | 0.98% |
| 控制网细分 | 0.004735 | 0.57% | 0.015572 | 0.60% |
| 终端认证其余工作 | 0.000285 | 0.03% | 0.001091 | 0.04% |

占比分母是本轮所有已插桩 scope 的 exclusive 时间之和，不是整个程序墙钟。每项占比先在各次运行内计算，再取中位数；因此表格中位数的总和可能因取中位数及四舍五入略偏离 100%。原始基准阶段时间使用未插桩程序，内部归因时间使用插桩程序，不将两者拼接成同一条精确时间轴。

每次 domain 构造分别记录 387,133 / 1,228,547 次曲面求值。从调用链看，N64 的 Newton 包含子调用时间为 1.513962 s，最近点辅助包含子调用时间为 0.915447 s；两者的主要共同成本是表面求值。这里不能再把 2.126107 s 的表面求值加到这两个包含时间上。

当前最大可测细分热点明确为 `NurbsSurfacePatch3D::evaluate_with_derivatives()` 等曲面求值调用，而不是候选枚举、BVH 构建或控制网细分本身。建议下一轮优化首先围绕同点值/导数复用、回溯中无用导数、基值与导数联合装配和临时缓冲区；本轮没有进一步拆分基函数与内存分配的各自比例。

### 插桩扰动检查

对同一个插桩可执行文件，以关闭、开启、开启、关闭的顺序分别跑 N64，每组仍预热 1 次、重复 3 次。合并得到关闭/开启各 6 个边求交时间样本，中位数比值显示开启计时增加约 **3.02%**。全部非时间字段仍与原基准一致。这仅量化同一可执行文件启用记录的增量，不包含临时源副本、禁用的 scope 包装和编译布局可能产生的全部影响。内部占比是带此扰动的归因估计，不对每个类别机械减去 3.02%。

运行记录位于原始数据目录下的 `overhead_off_1`、`overhead_on_1`、`overhead_on_2`、`overhead_off_2`；具体样本与比值见 `analysis.json`。

## 主程序后续 Q27 支撑路径：真正的长耗时阶段

另用临时 driver 副本保持原 topology target 的默认配置，执行 `hollow_cylinder 32`，显式指定 `KFBIM_3D_SOLVE_SELECTION=neumann_only`、`KFBIM_3D_SUPPORT_PATH=legacy`。输出确认 baseline 姿态、Q27-cover3 all-event、topology-affine local SVD、mean-free pivot elimination 等原有配置。

本机 VS2017 不能编译原主程序的不可复制投影对象工厂初始化。仅在临时副本中将该投影成员改为 `unique_ptr` 并保留相同的两种直接构造分支，绕过编译器限制；该密度投影构造位于本次预处理停止点之后，没有在观察中执行。副本在 harmonic pipeline 构造后以专门的停止标记返回，不运行后续 GMRES。生产源码没有修改。

### 阶段观察

首次有效观察中，domain 完成于 1.030315 s，domain 到 harmonic 构造之间的工作为 0.191378 s。进程在 **120.020330 s** 的观察预算结束时仍停留在 harmonic pipeline 的支撑模板构造，CPU 时间 **119.906250 s**，未进入 GMRES。

该进程是由性能观察包装器主动停止，记录中的 exit 1 是强制停止结果，**不是程序自然报错、GMRES 发散或完整预处理耗时**。因此这里能确认长耗时阶段和 CPU 持续工作状态，但不能给出该阶段的最终总秒数，也不能判断是同一条困难线段还是多条查询累计。

### 主线程调用栈采样

第二次同配置运行中，仅对本任务进程的主线程采样，每 25 ms 一次，持续 30.0154 s，得到 1,200 个样本，0 次失败、0 次栈深度截断。累计暂停目标线程 0.637869 s，约占观察窗口的 2.13%。名称解析在恢复目标线程后执行。采样结束后，该次运行仍未完成支撑构造，包装器在 65 秒预算处停止。

这是主线程周期调用栈采样，不是全系统或 CPU-on-core ETW 计时。下表是某个函数出现在样本栈中的比例，包含其子调用，**各行不能相加**；它也不是该函数自身指令的独占 CPU 比例。

| 函数或调用链 | 样本数 / 1,200 | 包含调用的采样占比 |
|---|---:|---:|
| Q27 `build_all_event_trace_corrections` → NURBS segment/element 求交 | 1,200 | 100.00% |
| `NurbsSurfacePatch3D::evaluate_with_derivatives` | 1,078 | **89.83%** |
| `analytic_evaluation_and_partials` | 1,064 | 88.67% |
| 最近点辅助 `closest_point_nurbs_bezier_element_to_segment_3d` | 667 | 55.58% |
| 普通根求解 `native_newton` | 525 | 43.75% |
| `NurbsBasis1D::evaluate_nonzero_first_derivatives` | 509 | 42.42% |
| `NurbsBasis1D::evaluate_nonzero` | 432 | 36.00% |
| `operator new`（内存分配及其子调用） | 660 | **55.00%** |
| `std::vector<double>` 构造 | 648 | 54.00% |
| `free_base`（内存释放及其子调用） | 220 | 18.33% |

主程序调用链和已解析的项目/CRT 符号支持以下定位：**支撑线段求交反复进入 Newton 与最近点辅助，而曲面基值/导数计算中的临时 vector 构造及堆分配成为突出成本。** 源码对应 [曲面解析求值](../src/geometry/nurbs_surface_3d.hpp#L503)、[基函数导数求值](../src/geometry/nurbs_basis.hpp#L155) 和 [Newton 回溯](../src/geometry/nurbs_bezier_intersection_3d.cpp#L824)。下一轮应优先测量和优化可复用的基函数工作区、同点值/导数复用、回溯中不需要的导数。

由于未加载系统私有 PDB，部分系统内部地址只能映射到邻近导出符号；例如 CSV 中的 `RtlCreateTimer` 不能据此解释为程序真的在创建定时器。本结论使用有项目/CRT符号和源码信息支持的祖先调用链，不依据此类系统内部名称单独归因。优化内联也可能隐藏部分栈帧。

本轮没有保存每个 support 查询的起终点、持续时间和细分盒数分布，因此尚未定位到单条最坏线段，也不能仅由调用栈证明其病态几何原因。没有放宽容差、删减认证或切换到 `native_certified`。

证据：[120 秒观察记录](../output/nurbs_intersection_baseline_20260907/production_preprocess_N32_valid/completion.json)、[运行配置与阶段日志](../output/nurbs_intersection_baseline_20260907/production_preprocess_N32_valid/stdout.log)、[采样 CSV](../output/nurbs_intersection_baseline_20260907/production_preprocess_N32_sampled/stack_samples.csv)、[采样器记录](../output/nurbs_intersection_baseline_20260907/production_preprocess_N32_sampled/sampler.log)。

## 环境与原始文件

- Windows 10 build 19045；Intel Core i7-10700K @ 3.80GHz，16 个逻辑处理器。
- CMake 3.24.2，Visual Studio 2017 / v141，MSVC 19.16.27045 x64；Release `/O2 /Ob2 /DNDEBUG`，Eigen 3.4.0、CGAL 5.6.2。
- 基准可执行文件 SHA256：`78EE12CBE6ED65812FC923820ADA2AC7D4428834784E75198E96F95817FAE967`。
- 原始数据目录：[output/nurbs_intersection_baseline_20260907](../output/nurbs_intersection_baseline_20260907)。该目录为本地运行产物，默认不纳入 Git。
- [N32 原始 CSV](../output/nurbs_intersection_baseline_20260907/N32/nurbs_geometry_preprocess_raw.csv)、[N64 原始 CSV](../output/nurbs_intersection_baseline_20260907/N64/nurbs_geometry_preprocess_raw.csv)。
- [内部归因 CSV](../output/nurbs_intersection_baseline_20260907/intersection_breakdown.csv)、[机器可读分析与等价性检查](../output/nurbs_intersection_baseline_20260907/analysis.json)。
- [可执行文件指纹及环境元数据](../output/nurbs_intersection_baseline_20260907/run_metadata.json)、[临时插桩生成器/采样器/构建与运行日志副本](../output/nurbs_intersection_baseline_20260907/reproduction)。恢复生成器到仓库 `.cache/intersection-profile/` 后可按原构建配置重现；这些是性能观察工具，不是生产算法修改。
- Git HEAD 为 `42c76c980cb6d3ea24b972cf60fb6ba425dc3518`；核心源文件及生成副本指纹见 `reproduction/manifest.json`，临时 driver 差异说明见 `reproduction/driver_manifest.json`。

内部归因的 N32/N64 是分别启动的两个进程，各自 build 1 为参考、2 为预热、3–5 为正式测量。生成器的构造编号跨 N 不重置；若改为同一进程同时运行多个 N，分析脚本的编号筛选也必须同步调整。

复现基准（需要先配置好 Eigen/CGAL；多配置构建示例）：

```powershell
cmake -S . -B build-layout -DKFBIM_BUILD_EXPERIMENTAL_3D=ON
cmake --build build-layout --config Release --target nurbs_geometry_preprocess_benchmark_3d --parallel 1 -- /nr:false
$env:PATH = 'D:\CGAL\CGAL-5.2-beta1\auxiliary\gmp\lib;' + $env:PATH
.\build-layout\apps\Release\nurbs_geometry_preprocess_benchmark_3d.exe --backend baseline --geometry cylinder --N 32 64 --warmup 1 --reps 3 --out output\nurbs_intersection_baseline_repeat
```

这里的 DLL 路径是本机依赖路径；其他机器应使用自己的依赖安装位置。计时期间不并行构建或运行其他基准。
