# Spread / Restrict 资源复用实现说明（2026-09-08）

## 1. 本次实现的定位与边界

本说明对应新增的 C++ `near_surface_anchor_mixed` 路线。它依据提供的技术文档实现 P3 Spread、P2 跳跃延拓、3+3 法向恢复以及 setup 阶段的数据复用；**不是把旧支撑路径求交算法原封不动换成缓存**，也不是已认证的 all-event Restrict。

- 原 `legacy`、`native_certified`、`closest_point` 路线继续保留。不设置 `KFBIM_3D_SUPPORT_PATH` 时仍走原有默认 `legacy`，没有把新 mixed 路线设为默认。
- mixed 路线仅显式选择时启用。元数据报告 `path_certified=false`，目前没有自动识别多根支撑路径并切换 all-event 的 fallback。
- 网格 Spread 仍消费原有 `LaplaceCorrectionSupport3D::crossing_ops`，逐事件累加；本次没有把同一网格边的多个已记录事件缩成单事件。这里的网格事件保留与“支撑路径具有认证的完整事件集”是两件不同的事。
- 不改变原有 NURBS 几何控制点、权重、节点向量、patch 及几何拓扑；没有改为三角片几何，也没有在本次引入插节点或密度自由度自适应。
- 新的局部 P3/P2 Cauchy 闭合针对调和 jump。未新增非调和 jump 的体 forcing 或 forcing-jet 扩展。
- 本次实现依据技术文档及本地代码，没有可逐行核对的对应 Python 实现包。因此不声称与 Python 算子逐位等价，也不在完成实际对照前声称加速倍数或二阶收敛。

## 2. 模块与调用位置

| 文件 | 作用 |
|---|---|
| `src/support/cauchy/direct_coefficient_cubic_cauchy_3d.hpp/.cpp` | 原生 NURBS 三阶几何 jet、解析密度 jet、P3 调和 Cauchy 闭合以及显式低阶选择 |
| `src/support/trace/restrict_resource_plan_3d.hpp/.cpp` | 同 smooth-sheet 锚点索引、错侧访问资源规划、3+3 共享 Q27/Q64 cover |
| `src/support/trace/resource_restrict_assembly_3d.hpp/.cpp` | 将延拓与恢复固化成 `Rg`、`Rc`、已知向量，并分别保留内外迹 |
| `src/support/trace/spread_restrict_resource_3d.hpp/.cpp` | 串接几何计划、P3 Spread、交点 P2 保留、端点行缓存、BVP 算子构造和计时 |
| `apps/laplace/3d/neumann_exterior_zero_trace_3d.cpp` | 显式选择新模式，绑定系数密度与已知数据，在迭代中应用固定算子 |

主流程为：

```text
几何 / 网格 / 原有网格 crossing_ops
  → ResourceGeometryPlan3D：trace / crossing 锚点、两侧 cover、错侧请求
  → 绑定本次 BVP 的密度空间与解析已知数据
  → 构造 P3 Spread；只保留 Restrict 需要的交点 P2 数据和端点 P2 行
  → 构造 P2 Restrict、Rg / Rc / b
  → 释放 setup 临时中心及几何计划
  → GMRES：系数线性映射 + 稀疏矩阵乘法 + 原 bulk solve
```

`ResourceGeometryPlan3D` 和 BVP 算子的生命周期属于单次 operator setup，没有跨几何或跨实验的全局静态缓存。更换几何、网格、密度分配、边界类型或已知数据需要重新构造。

## 3. 几何身份、参数和资源请求

### 3.1 几何身份

每个 `RestrictResourceAnchor3D` 保留物理点、patch、原生参数 `u/v`、`sheet_id`；Spread 锚点另有局部 crossing ID。`sheet_id` 由 `smooth_patch_component()` 的 G1 邻接分组得到，**不是封闭连通分量编号**。因此位于同一封闭实体、却隔着 C0 特征棱边的两个光滑面不会共享同一个 sheet 请求。

`SurfaceDofCloud3D` 中 `u/v` 是原生 knot 域参数。资源绑定时显式转换为密度 API 使用的归一化参数：

\[
\hat u=\frac{u-u_{\min}}{u_{\max}-u_{\min}},\qquad
\hat v=\frac{v-v_{\min}}{v_{\max}-v_{\min}}.
\]

转换仅做一次。不用小数取整或容差量化合并不同原生参数。交点中心复用使用本次模型内精确的 `(patch, u, v)` 键；请求使用整数 `(sheet_id, grid_full_id)` 键。

### 3.2 错侧访问

一个支撑访问记录 trace 中心、需要的侧、节点完整 Cartesian ID、物理点、实际侧。实际侧与需要侧相同时，`visit_to_request=-1`，不创建 jump 请求。

错侧时，reference 模式为每次访问建立请求；reuse 模式仅为每个 `(sheet_id, grid_full_id)` 建立一次请求，但保留原来的访问顺序及每次访问的权重。完整网格 ID 不与 bulk 内部自由度编号混用；当前 C++ bulk 向量包含盒边界节点。

同一整数节点 ID 若对应不同物理坐标或不同实际侧，规划器直接拒绝，不依靠近似坐标比较掩盖不一致。

### 3.3 混合锚点选择规则

对于错侧节点 `x`：

1. 查询同一 smooth sheet 上最近的 trace 锚点 `q`。
2. 仅当 `|x-q| > 1.5h` 时，查询同 sheet 的 Spread 交点锚点。
3. 只有交点锚点严格更近才替换 trace 锚点。
4. 距离相同保留确定性的原输入次序；trace 与 crossing 距离相同时不切换。
5. 最终距离 `> 2.25h` 仅记为 `too_far`，不是剪裁距离或几何认证条件。

查询使用按 sheet 的 KDTree，不是对每次访问线性扫描全部锚点。还记录另一 sheet 的最近 trace 锚点是否比选中锚点至少近 `0.15h`，作为 `competing_sheet` 风险提示。

独立规划 API 提供 guarded 模式并标记 `fallback_required`；装配器要求这种请求提交已独立解决的 fallback 行。**主 driver 当前仅开放 reference/reuse，没有启用 guarded，也没有实现其自动 fallback。** 未触发距离提示不能解释为路径已获认证。

## 4. P3 Spread 与 P2 数据复用

记 jump 约定为 `J0 = u_inside - u_outside`、`J1 = ∂n u_inside - ∂n u_outside`，法向为外法向。在资源 BVP 中：

| BVP | 已知数据 | 系数未知量 |
|---|---|---|
| Neumann | `J1 = gN`，保留原兼容性均值处理 | `J0 = mu(c)` |
| Dirichlet | `J0 = gD` | `J1 = mu(c)` |

密度值与导数由 B 样条系数解析得到；已知数据由解析环境函数的值、梯度、Hessian 和三阶导数提供。此处没有从离散密度样本重新拟合未知 jump。

坐标闭合延续本地 C++ 的物理切平面 graph 坐标 `(s,t,r)`，包含图函数曲率及三阶导数；没有直接照搬文档的管状参数坐标矩阵。两者的展开阶数相同不意味着截断多项式逐项相同，因此本次 A/B 是本地 reference/reuse 之间的检验，不是 Python/C++ 数值等价声明。

对每个 Spread 交点中心建立一次三阶几何/密度 jet 和 P3 Cauchy plan，再遍历属于该中心的网格修正操作。每条操作保持原 `side_delta * stencil_weight` 与目标节点关系。组装得到：

\[
f_{\Delta}(c)=S c+b_s.
\]

本项目已有 bulk solver 使用 `-Delta_h` 约定，因此代码传入的是 `-(S*c + b_s)`；不能把这个符号照抄成正号。

### 4.1 只保留需要的交点低阶资源

Restrict 规划先给出 `required_crossing_ids`。reuse 引擎在这些交点 P3 plan 仍存在时保留 P2 所需的低阶信息，随后释放该中心的三阶临时数据；未被 Restrict 选中的交点不保留完整中心。

低阶数据通过 `lower_value_plan()`、`lower_normal_plan()` 提取。当前表面导数顺序分别为：

```text
J0：value, s, t, ss, st, tt, sss, sst, stt, ttt
J1：value, s, t, ss, st, tt
```

这与环境 P3 多项式系数的单项式顺序不是同一概念。环境多项式的 P3→P2 选择由 `cubic_to_quadratic_selection_3d()` 根据指数标签显式建立，即仅保留总次数不超过 2 的相同单项式；不能取 P3 系数向量的前十项代替。

reference 引擎不利用 Spread 的 P3 plan 提取交点 P2，而是独立构造所需 P2 中心，并为每次错侧访问计算行。reference 和 reuse 使用相同锚点选择准则、相同 stencil、相同访问累加次序；区别是 setup 资源复用，不是不同的数学边界条件。

### 4.2 端点行复用

如果某个被选中的交点对应的 Restrict 目标节点，恰好也出现在 Spread 修正目标中，就在 Spread 遍历时计算并保留该目标的 **P2** 仿射评价行。键为 `(crossing_id, grid_full_id)`。

Restrict 遇到该请求时直接使用已有 P2 行。这里没有把用于 Spread 的 P3 行直接冒充 Restrict 的 P2 行，也没有重新求交或最近点投影。

## 5. 3+3 共享 cover 和双侧迹恢复

新 mixed 路线采用固定法向层：

\[
\rho=(-1.5,-0.75,-0.5,\;0.5,0.75,1.5),\qquad
x_i=q+h\rho_i n(q).
\]

同一侧的三个采样点共享一个 Cartesian 张量 cover。Neumann 推荐配置为 Q27-cover3，Dirichlet 为 Q64-cover4；旧的单点 tensor-product cover API 未被重写。

- Q27 在每轴使用 3 个节点，构造张量 Q2 插值。
- Q64 在每轴使用 4 个节点，构造张量 Q3 插值。
- 每轴优先选择包围该侧三个采样点的节点区间，并尽量围绕三点坐标范围的中点居中；索引限制在网格节点盒中。
- 采样点自身若超出盒边界则报错，不能静默把采样位置夹回盒内。

文档没有给出可复刻的全部 Python tie 细节；上述居中与裁剪准则是本地明确实现的确定性规则。

设六点 Vandermonde 矩阵为 `V[i,:] = (1,rho_i,rho_i^2,rho_i^3)`。`V+` 的第 0 行恢复值，第 1 行除以 `h` 恢复物理法向导数。利用对称层的奇偶分块，代码以两个 2×2 Gram 块的闭式公式构造固定伪逆，不对每个 trace 中心重复做 SVD。

对每侧支撑节点，先作错侧延拓：

\[
U^{\mathrm{side}}(x)
=U_h(x)+\bigl(\chi_{\mathrm{desired}}-\chi_{\mathrm{actual}}\bigr)J_{a(x)}^{P2}(x).
\]

插值到六个法向采样点后：

- 恢复外迹时，对负侧三个采样点减去 `q` 中心的 P2 jump；
- 恢复内迹时，对正侧三个采样点加上 `q` 中心的 P2 jump。

随后用同一个六点 cubic 恢复矩阵提取值和法向导数。这里的固定局部多项式恢复与“用样本拟合未知密度自由度”不同，未知密度仍是解析系数路线。

## 6. 仿射算子以及不能遗漏的 Rc 项

setup 将所有权重与 jump 行装配为：

\[
t_v=R_{g,v}U_h+R_{c,v}c+b_v,\qquad
t_n=R_{g,n}U_h+R_{c,n}c+b_n.
\]

`Rg` 作用于完整 Cartesian 向量，`Rc` 作用于 Base/C0 系数；内外迹分别保留各自的 `Rc` 和已知向量。法向矩阵已经包含 `1/h`，应用阶段不能再除一次 `h`。

若拓扑/边界相容约束给出仿射密度空间：

\[
c=c_p+Zy,
\]

则完整迹为：

\[
t(y)=\underbrace{R_gU_0+R_cc_p+b}_{\text{已知仿射部分}}
      +\underbrace{R_gU_y+R_cZy}_{\text{待迭代线性部分}}.
\]

因此，GMRES 线性作用必须包含 `Rc*Z*y`，不能只留下 `Rg*U_y`。当前资源路线将每次展开后的完整 C0 系数传入固定 Spread/Restrict 矩阵，`Rc` 不因外层自由度降维而删除。

`ResourceBvpOperators3D::trace_basis` 同样从解析 P2 值 jet 的零阶行构建，**只是矩阵 A/B 与 dump 所用的审计副本，不被 driver 的 projector 消费**。driver 在审计与导出完成后将该副本清空并释放容量，不把它带入 GMRES。资源 matvec 也不再分配旧的面板系数零矩阵。

现有 `NativeDensityTransfer3D` 已经缓存 `sample_stencils_` 和 `c0_design_`；原有 Neumann/Dirichlet 仿射 projector 复用 `c0_design_`，QR 在 setup 中构造一次，GMRES 不会重做。这项已有优化无需本轮复刻，不能计为本次资源路线新增的性能收益。

## 7. 计数与时间口径

每个 BVP 写出 `neumann_resource_setup.json` 或 `dirichlet_resource_setup.json`，包括访问数、错侧访问数、请求数、Spread 中心数、保留交点中心数、P2 中心数、P3→P2 复用数、端点行命中数、P2 行评价数及距离风险计数。

时间字段：

| 字段 | 含义 |
|---|---|
| `planning_seconds` | 锚点、sheet、共享 cover、访问及请求规划 |
| `spread_seconds` | P3 Spread 装配，包含本阶段顺带执行的低阶保留与端点 P2 行计算 |
| `restrict_seconds` | Spread 之后剩余的 Restrict 中心/行与矩阵装配 |
| `retained_p2_seconds` | `spread_seconds` 内低阶中心保留工作的子时间 |
| `endpoint_p2_seconds` | `spread_seconds` 内端点 P2 行构造工作的子时间 |

资源模块 `operator_setup_seconds` 记录已计时阶段之和：

\[
T_{\mathrm{setup}}=T_{\mathrm{planning}}+T_{\mathrm{spread}}+T_{\mathrm{restrict}}.
\]

为避免把提前到 Spread 的工作误报为 Restrict 加速，等价 Restrict 口径为：

\[
T_{\mathrm{restrict,eq}}=T_{\mathrm{planning}}+T_{\mathrm{restrict}}
 +T_{\mathrm{retained\ P2}}+T_{\mathrm{endpoint\ P2}}.
\]

后两项已经包含在 Spread 中，不能再加入总 setup 重复计费。上述模块时间也不等于完整算例墙钟时间；网格/域构造、外层约束与投影器、bulk solve、GMRES、输出等仍需另外统计。

该阶段和也不是严格包围整个 setup 调用的墙钟计时：计时段之外的少量集合准备、函数返回时局部析构、driver 清空审计 B 和释放几何计划等不在其中。完整进程耗时以 runner 的 `elapsed_seconds` 为准。`temporary_centers_after_build=0` 是依据对象生命周期报告的逻辑计数，不是进程 RSS 或分配器内存归零的测量。

reference 的 `requests` 等于错侧访问次数，reuse 的 `requests` 是唯一资源键数。`too_far_requests` 和 `competing_sheet_requests` 也采用各自的请求口径，因此不能把两引擎的风险计数直接相减并解释为几何风险下降。

矩阵导出位于模块 setup 计时之外。显式 A/B 检查会额外建立另一个引擎的算子，其检查成本不能伪装成正常单引擎求解成本，也不能用于计算正常运行加速比。

## 8. 开关、对照与运行入口

| 配置 | 作用 |
|---|---|
| `KFBIM_3D_SUPPORT_PATH=near_surface_anchor_mixed` | 显式选择新的未认证 mixed 路线 |
| `KFBIM_3D_RESOURCE_ENGINE=reference` | 同一数学策略的逐访问参考装配 |
| `KFBIM_3D_RESOURCE_ENGINE=reuse` | 请求、交点低阶与端点行复用；mixed 内默认值 |
| `KFBIM_3D_RESOURCE_CHECK_AB=1` | 在一次运行内额外构造另一引擎并比较算子 |
| `KFBIM_3D_RESOURCE_DUMP=1` | 输出 Matrix Market 矩阵与已知向量 |
| `KFBIM_3D_PREPROCESS_ONLY=1` | 只执行预处理检查；不能当成已得到 PDE 数值解 |

A/B 检查比较 `S`、已知 Spread、解析 trace basis、值/法向 `Rg`、内外迹的 `Rc` 和已知向量；当前 driver 使用最大绝对差 `5e-10` 作为失败门限。通过该检查说明 reference/reuse 实现一致，不说明 mixed 与旧 certified 路线等价，也不证明空间收敛阶。

配套入口为 `scripts/validation/run_resource_reuse_3d.ps1`。默认 cylinder、N=32、reuse、Neumann/Dirichlet 两类问题、每方向 4 个密度系数；`-DensityCoefficients 0` 保留原自动分配。支持 `-Engine reference`、`-CompareEngines`、`-PreprocessOnly`、`-DumpMatrices`。脚本为每次运行创建全新结果目录，拒绝覆盖既有结果；仅设置子进程环境，保存参数、stdout/stderr 和包含 A/B 开销的整体墙钟时间。N=16 仍可指定，但不保证曲面采样足以支撑三次密度基；原有可观测性检查不会因此放宽。

```powershell
./scripts/validation/run_resource_reuse_3d.ps1 -Geometry cylinder -Level 32 -CompareEngines -PreprocessOnly
./scripts/validation/run_resource_reuse_3d.ps1 -Geometry cylinder -Level 32 -Engine reuse -SkipBuild
./scripts/validation/run_resource_reuse_3d.ps1 -Geometry cylinder -Level 32 -Engine reference -SkipBuild
```

刚体变换通过脚本 `-RigidCase rot_axis123_17deg_t_xyz_1` 选择，不使用硬编码旧 restrict 的 `--rigid-study` 入口。资源 JSON 和矩阵分别位于本次结果目录的 `<geometry>_N<N>_resource/` 子目录，避免多网格、多几何运行互相覆盖。

## 9. 实际验证记录

### 9.1 版本、构建和测试

测试基于本地 `main` 的 `1ff60b20c21183ea85b3fccf493d480f6b28f512` 加本次未提交修改。使用 Windows / MinGW GCC 16.1、CMake Ninja Release、`BUILD_TESTING=ON`、MPFR ON，重新编译了库、测试及 `kfbi_topology_affine_exterior_trace_3d`，没有使用旧可执行文件冒充新版本结果。

| 新增测试 | 状态 | CTest 耗时 |
|---|---|---:|
| `restrict_resource_plan_3d_test` | 通过 | 0.74 s |
| `resource_restrict_assembly_3d_test` | 通过 | 0.60 s |
| `direct_coefficient_cubic_cauchy_3d_test` | 通过 | 1.37 s |

三项总时间 2.86 s。覆盖非单位参数域、非单位有理权重、三阶几何/系数 jet、P3 调和再现、P3/P2 低阶一致性、刚体变换、唯一请求与 KDTree 边界、Q27/Q64 六点恢复、内外迹、非零仿射特解以及不能遗漏的 `Rc*Z*y`。断言使用运行期检查，Release 构建不会关闭它们。最终日志：`output/resource_reuse_20260908/build_ctest_resource_final.log`。

初次 cubic 单测曾因以开放单 patch 构造要求封闭曲面的密度空间而失败；已修正测试夹具，并提供纯 patch 的三阶几何 jet 重载，没有放宽生产代码的封闭拓扑校验。上表为修正后重新构建的结果。

六项既有路线回归的最终结果另在本节末记录。PowerShell 5.1 语法检查和 DryRun 已通过，含默认 N=32、显式 N=16、固定密度 4、自动密度 0 的参数分支。

### 9.2 端到端算例和资源 A/B

本轮完整求解使用本地 **空心圆柱 cylinder、N=32、baseline（无刚体变换）**；几何由本地工厂生成，保留 16 个原生 NURBS patch。固定每参数方向 4 个密度系数，panel-center 外迹点数 1160；Neumann 使用 Q27-cover3，Dirichlet 使用 Q64-cover4、`analytic_j0_affine_j1` 和 `broken_sheets`。GMRES 上限 80、容差 `2e-10`；BLAS/OMP 线程设为 1。没有修改拓扑约束或降低求解容差。

| 结果目录 | 运行内容 | 退出状态 | 整体墙钟 |
|---|---|---|---:|
| `o/rr16a` | N=16、reuse、A/B、导出 | 可观测性检查拒绝，非完整数值结果 | 6.647 s |
| `o/rr32a` | N=32、reuse、A/B、导出、两类完整求解 | 成功 | 34.126 s |
| `o/rr32b` | N=32、reference、无 A/B/矩阵导出、两类完整求解 | 成功 | 15.485 s |
| `o/rr32c` | N=32、reuse、无 A/B/矩阵导出、两类完整求解 | 成功 | 15.374 s |

N=16 在常数探针通过后，原 guard 报 `surface sampling cannot observe a cubic density basis`，未得到 PDE 误差或 GMRES 结果；因此改用 N=32，没有关闭检查。

N=32 的 reference/reuse 算子最大绝对差：Neumann **2.220446e-16**，Dirichlet **2.775558e-17**。常数 `[u]=1` 探针的外值迹误差为 **1.693090e-14**、内值迹误差为 **1.687539e-14**。这些检查支持资源复用未改变本地 reference 的离散算子，不证明与旧路线或 Python 算子完全相同。

### 9.3 数值误差、残差和 DOF

以下取干净 reuse 运行 `o/rr32c`；reference 的误差在表列有效数字上相同，GMRES 次数相同。对两次 CSV 的 GMRES 残差、投影残差、完整外迹、密度与内部解误差逐项比较，最大绝对差为 Neumann `4.309387e-17`、Dirichlet `1.110223e-16`。

| 指标 | Neumann | Dirichlet |
|---|---:|---:|
| 原始系数 DOF | 256 | 256 |
| Base/C0 DOF | 144 | 192 |
| 最终迭代 DOF | 31 | 128 |
| 外迹点数 | 1160 | 1160 |
| 外迹点数减最终 DOF | 1129 | 1032 |
| GMRES 次数 | 16 | 29 |
| GMRES 相对残差 | 1.465359e-10 | 1.769251e-10 |
| 投影算子残差 L∞ | 5.858328e-12 | 1.248377e-11 |
| 完整未投影外迹 L∞ | 1.097354e-3（值迹） | 2.536753e-3（法向迹） |
| 密度误差 L∞ | 2.587712e-3 | 5.589595e-3 |
| 密度误差 L2 | 1.292917e-3 | 2.294862e-3 |
| 内部解误差 L∞ | 1.857854e-3 | 2.966254e-4 |
| 内部解误差 L2 | 7.480344e-4 | 1.303661e-4 |
| CSV 求解阶段 `seconds` | 4.921642 s | 4.625894 s |

两类问题均满足外迹点数严格大于最终自由度。Neumann 的 144 个 Base 自由度经秩 112 的拓扑约束降至 32，再由均值约束降至 31；Dirichlet 的 192 个 Base 自由度经秩 64 的已有 smooth-seam 约束降至 128。本轮没有引入额外 DOF 消元。Dirichlet 特征棱边保持 `broken_sheets`，不应把此结果说成启用了跨 C0 特征边的梯度相容约束。

CSV 的 `converged=1` 仅表示投影后的 GMRES 系统收敛；两类 `physical_converged=0`，完整外迹仍为有限离散误差，**不能说所有表面采样条件已达到 GMRES 容差**。Dirichlet 的投影泄漏相对 L2 为 1，表示剩余法向迹几乎完全位于投影未控制的分量；不能由小投影残差推断完整法向迹同样很小。Neumann 对应 CSV 字段为 0，本轮不把它当作完整 Neumann 泄漏诊断。

表中求解阶段 `seconds` 包括该 BVP 的约束、投影器准备及迭代等，不是纯 GMRES 内核时间，也不含前面的资源矩阵 setup。未运行 N=64/128，**本轮没有 32→64 收敛阶**。未运行其他几何或刚体变换下的完整 PDE 对照；刚体变换目前仅有局部 jet 单测。

### 9.4 资源和耗时前后对照

使用 `o/rr32b` 的 reference 与 `o/rr32c` 的 reuse，两次串行运行；均不含 A/B 或矩阵导出，也没有与本轮编译/其他 PDE 并发。下表为单次测量，不是重复统计的稳定加速比。

| 项目 | Neumann reference → reuse | Dirichlet reference → reuse |
|---|---:|---:|
| 错侧资源请求 | 11808 → 2006（减少 83.01%） | 41664 → 4279（减少 89.73%） |
| P2 行评价次数 | 18768 → 8966 | 48624 → 11239 |
| P3→P2 保留中心数 | 0 → 123 | 0 → 357 |
| 端点 P2 缓存命中数 | 0 → 0 | 0 → 0 |
| planning | 0.255215 → 0.120265 s | 0.666090 → 0.210556 s |
| spread（含低阶保留） | 0.235293 → 0.422633 s | 0.297375 → 0.255471 s |
| restrict 后续装配 | 0.890280 → 0.550409 s | 1.296737 → 0.704405 s |
| 等价 restrict 时间 | 1.145494 → 0.705071 s | 1.962828 → 0.919857 s |
| 资源 setup 阶段和 | 1.380787 → 1.093307 s | 2.260203 → 1.170432 s |
| CSV 求解阶段 | 4.596335 → 4.921642 s | 3.794736 → 4.625894 s |

setup 阶段和的单次比值约为 Neumann **1.26×**、Dirichlet **1.93×**；已把提前到 Spread 的低阶保留计入等价 Restrict，未把它当成消失的工作。当前算例端点缓存实际命中为零，不能将该功能计作本算例加速来源。

整次两类问题的墙钟仅 **15.485→15.374 s**，差约 0.7%，同时求解阶段耗时有波动；因此可以确认重复请求和行评价明显减少，但**目前不能宣称端到端有显著或稳定加速**。`o/rr32a` 的 A/B 与矩阵导出额外耗时不参加上述加速比较。

reuse 中的风险诊断为 Neumann `too_far=133`、`competing_sheet=714`，Dirichlet `too_far=653`、`competing_sheet=1498`；这些请求仍按文档的 mixed 近锚点规则处理，并没有因此获得路径认证。未认证支撑路径和未启用自动 fallback 仍是本路线的明确边界。

### 9.5 结果文件定位

各运行目录根部保存 `environment.json`、`run.json`、`stdout.log`、`stderr.log`。例如干净 reuse 的 CSV 在：

```text
o/rr32c/resource_mixed/reuse/kfbi_topology_affine_3d/
  q27_c3/d_q64_c4/nm_pc_tm_mf/topology_affine/
    neumann_results.csv
    dirichlet_normal_results.csv
    cylinder_N32_resource/
      neumann_resource_setup.json
      dirichlet_resource_setup.json
```

`o/rr32a` 对应目录另含 Matrix Market 的 `S_delta`、`B`、值/法向 `Rg`、外迹 `Rc` 及已知向量，便于独立检查。reference/reuse 的 comparison 同时检查内外迹；dump 的 `Rc` 文件为外迹版本。

### 9.6 既有路线回归

在上述 PDE 计时全部结束后，单独重新编译六项既有测试并串行执行，没有把并发编译干扰计入数值计时。

| 既有测试 | 状态 | CTest 耗时 |
|---|---|---:|
| `native_nurbs_density_space_3d_test` | 通过 | 9.24 s |
| `tensor_product_cover_restrict_3d_test` | 通过 | 22.71 s |
| `topology_affine_reduction_3d_test` | 通过 | 4.45 s |
| `topology_mean_free_reduction_3d_test` | 通过 | 0.88 s |
| `topology_trace_projector_3d_test` | 通过 | 2.57 s |
| `direct_coefficient_cauchy_3d_test`（原 P2） | 通过 | 6.48 s |

六项 CTest 总时间 46.49 s，增量构建 340.867 s，退出码均为 0。日志：`output/resource_reuse_20260908/build_legacy_regression.log`、`output/resource_reuse_20260908/build_ctest_legacy_regression.log`。加上第 9.1 节的三项新增测试，本次检查的 **9 项测试全部通过**；未执行整个仓库的完整测试集合，不将此记录表述为全仓库回归通过。

本次仅修改本地工作树，未提交或推送 Git。
