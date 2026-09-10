# Trace-first C++ 实现说明（2026-09-09）

本文说明本轮新增代码的结构、数学接口和适用边界，不包含本地数值结果，也不把新增测试文件等同于测试已经通过。实际误差、GMRES、耗时和验收状态应查对应运行目录中的记录及单独的数值报告。

本轮增加独立研究入口 `kfbi_trace_first_study_3d`，用于对照提供的 Python 最新圆环基准；没有替换原有多几何主程序，也没有把圆环专用的约束构造推广为通用 C0 特征棱边/vertex 算法。

## 1. 算例与入口：先固定比较对象

代码入口：

- [trace_first_study_3d.cpp](../apps/laplace/3d/trace_first_study_3d.cpp)：选项、几何准备、N/D 仿射求解、诊断与 JSON 输出。
- [trace_first_case_3d.cpp](../src/support/geometry/trace_first_case_3d.cpp)：`make_trace_first_python_torus_case_3d()`、制造解、刚体变换和密度加密级别。
- [python_torus.json](../tests/cases/trace_first_3d/python_torus.json)：算例配置说明。

当前对照对象是大半径 `R=0.46`、小半径 `r=0.18` 的圆环，使用 4×4 共 16 张精确有理 **双二次 NURBS**，每张 3×3 控制点。这里不是双三次几何，也没有实施几何升阶或改变控制网；三次的是独立的密度 B 样条空间。既有其他算例的几何表示没有因此改变。

默认 `rotate` 是绕 x/y/z 的 17°、−11°、13°，组合为 `Rz Ry Rx`，无平移。另有 `translate`、`rotate_same_translate`、`rotate_translate` 可选；这些选项可运行不代表已经完成所有姿态的数值验证。

以变换前坐标 \(\xi=(x,y,z)\) 表示，制造解为

\[
u(\xi)=e^{ax}\cos(by)\cos(cz)
 +0.18(x^2-y^2)+0.11xyz+0.07x-0.05z-\overline u,
\quad b=0.72,\ c=0.43,\ a=\sqrt{b^2+c^2}.
\]

所有项均调和。值、梯度、Hessian 和三阶导解析计算，再进行刚体坐标转换；没有用差分或样本拟合制造解导数。\(\overline u\) 按 Python 的规则，使用每张几何 patch 的 8×8 Gauss 积分计算。这与后面密度空间的面积均值积分是两项不同的工作。

背景网格为 `[-1,1]^3`，每方向 N 个区间，\(h=2/N\)。驱动以原生 NURBS 结果建立标签与交点；解析圆环 `exact_inside` 只作标签核验，标签不一致时停止，不用解析标签替换数值标签。

## 2. 密度、约束及精确嵌入

主要代码：[python_torus_density_layout_3d.cpp](../src/support/density/python_torus_density_layout_3d.cpp) 中的 `cubic_midpoint_insertion_matrix_3d()` 和 `build_python_torus_density_layout_3d()`。

### 2.1 按 Python 配置建立嵌套分析空间

几何控制网固定，密度使用非有理双三次 B 样条。N=32 时每张 patch 的分析区间为 4×2，网格加倍时两个方向的分析区间都加倍：

| N | 每 patch 分析区间 | 原始密度系数 | Dirichlet 最终维数 | Neumann 最终维数 | 外迹点数量 |
|---:|---|---:|---:|---:|---:|
| 32 | 4×2 | 560 | 240 | 239 | 2048 |
| 64 | 8×4 | 1232 | 720 | 719 | 8192 |

此表是配置公式给出的计数，不是数值计算的通过结果。外迹点为每个密度分析单元 4×4 Gauss 点，必须严格多于最终未知维数。

### 2.2 周期 C0/C1 是当前圆环 atlas 的专用精确关系

每个参数方向由四个四分之一圆环区段组成。相邻区段约束端点系数相同，并约束端点相邻系数差相同。对于这里区间数量一致、端点一阶参数导匹配的 atlas，这些系数关系精确表达周期 C0/C1。

代码先求两个一维约束矩阵的零空间 \(Z_u,Z_v\)，再取张量积并重排到 patch-major 顺序。没有在整个圆环上重新构造一个大型稠密约束矩阵并整体消元。

这一可分离关系依赖当前 atlas，不能直接用于任意重参数化接缝、特征棱边或 vertex。原有通用拓扑仿射局部 SVD、弱棱边相容性和 vertex 处理仍保留在原路线中，本独立研究驱动不声称替代它们。

### 2.3 方形 carrier 不增加实际迭代自由度

现有 `NativeNurbsDensitySpace3D` 使用每 patch 相同的方形三次基。为保留通用密度和 Cauchy 接口，本轮没有重写其索引规则，而使用精确的系数嵌入：

1. u 方向使用 Python 所需的细分数 \(n_u\)。
2. v 方向将 \(n_v=n_u/2\) 的三次样条，通过 Boehm 中点插结精确延拓到 \(n_u\) 的 carrier。
3. 将系数重排为本地 `patch * n*n + i*n + j`。
4. 按原有 `local_to_c0` 合并全边 C0 系数；重复映射必须给出一致的行，否则停止。

插结只改变表示坐标，不改变密度函数。最终映射可记为

\[
Z_D=\mathcal E\,Z_{uv},
\]

其中 \(Z_{uv}\) 为 Python 各向异性空间的周期零空间，\(\mathcal E\) 包含精确插结、重排和兼容 C0 合并。carrier 中多出的坐标不单独参加 GMRES，因此这不是把 Python 4×2 空间近似替换成 4×4 空间。

Neumann 另以每个密度分析单元的 8×8 Gauss 规则形成面积均值行 \(m\)，通过 Householder 变换从 \(Z_D\) 中去除 \(mZ_D\) 的一个方向。当前光滑圆环条件齐次，\(c_p=0\)；求解接口仍保留一般形式 \(c=c_p+Zy\)。

这不是“自由度不足后自动加密直到精度足够”的自适应算法，而是与 Python 对照一致的预定 dyadic 加密规则。

## 3. 仿射方程及 N/D 对应

采用外法向，jump 为内部减外部：

| 问题 | 已知 jump | 未知密度 | 主外迹方程 |
|---|---|---|---|
| Neumann | \(J_1=g_N\) | \(J_0=\mu(c)\) | 外部值迹为零 |
| Dirichlet | \(J_0=g_D\) | \(J_1=\mu(c)\) | 外部法向迹为零 |

未知密度及其参数导数直接由基函数系数取得；边界数据及其导数由制造解取得。新研究驱动的 Neumann 通量均值移除参数传入 `0.0`，与本次 Python 调和算例一致，不能据此推断其他数据无需通量相容性检查。

组装固定算子后：

\[
C(c)=Sc+b_s,\qquad
t=R_gU+R_cc+b_r.
\]

本地 bulk 接口解 \(\Delta_hU=f\)，而 Python 的正特征值 Poisson 实现解 \(-\Delta_hU=f\)。因此本地调用

```cpp
rhs = S * c + known_spread;
poisson.solve(-rhs, U);
```

才与 Python `P.solve(S@c+b)` 同号。不能根据函数名遗漏这个负号。

定义 \(U_p=-\Delta_h^{-1}(Sc_p+b_s)\)。GMRES 方程为

\[
\Pi\left(-R_g\Delta_h^{-1}S+R_c\right)Zy
=-\Pi\left(R_gU_p+R_cc_p+b_r\right).
\]

代码的齐次 matvec 明确保留 `Rc*c`，其中 `c=Z*y`；最终恢复 `c=cp+Z*y`，重新加入全部已知项。不能只迭代网格场的 `Rg*U`。

本独立驱动按 Python 形式构造

\[
M=Z^TB^TWBZ,\qquad
\Pi t=M^{-1}Z^TB^TWt,
\]

使用稠密 Cholesky 和 native reduced 坐标。它不是旧通用主程序的 trace-mass/QR 路线。外迹数量检查、Cholesky 成功检查和三角因子主元比检查都保留；主元比是诊断量，不等同于完整条件数。

GMRES 默认容差 `2e-10`、重启长度 `min(100, reduced_dofs)`、总内迭代上限1000。对当前32/64基准，相当于最多10个长度100的重启周期；同时计算最终完整仿射迹的真实投影相对残差，不只依赖迭代历史中的估计值。

## 4. 外迹优先的 P3/P2 复用

主要代码：[trace_first_resource_3d.hpp](../src/support/trace/trace_first_resource_3d.hpp)、[trace_first_resource_3d.cpp](../src/support/trace/trace_first_resource_3d.cpp)。建议先读 `TracePolynomialCatalog3D`，再读 `build_trace_first_geometry_plan_3d()`，最后读 `build_trace_first_bvp_operators_3d()`。

### 4.1 先选中心，再合并请求

默认 `--policy trace` 下：

- Spread：由网格交点 owner 确定局部候选 patch 集合，在该集合的已有外迹点中选择近锚点；目标节点距锚点不超过 `2.75h` 才使用 trace P3，否则保留事件中心 P3 回退。
- Restrict：错侧节点距当前外迹点 \(q\) 不超过 `2.25h` 时直接使用 \(q\)，否则查询该 patch 及直接声明邻 patch 的已有外迹点；超过 `2.75h` 的选择计入远距离诊断，而不是当成已经认证。

当前候选分支是 patch 邻接近似。当前圆环这些拼接是光滑的；不能把该规则当成任意 C0 特征几何中已经证明正确的 physical-sheet/局部分支分类。

不同访问可先选择不同锚点；选择完成后才共享 `(anchor_id, grid_id)` 请求。最终评价行缓存键为

\[
(\text{degree},\text{source},\text{anchor index},\text{full grid node ID}).
\]

`source` 区分 trace 与事件中心，`degree` 区分 P2 与 P3。同一键若被传入不同空间点，代码抛错。缓存的是系数仿射行 \(\ell c+k\)，不是某次迭代的密度数值。法向样本若没有网格 ID，不错误地按网格节点缓存。

默认 trace-first 不执行此前讨论的 DDA cut-cell 分支搜索、唯一性判据或自动 all-event 回退，也没有证明同侧路径一定无交点。`GuardedReuseBySheetNode` 在此新规划器中显式拒绝，不能把尚未实现的保护策略当成已启用。

### 4.2 P3、P2 与 tubular chart

代码：[direct_coefficient_cubic_cauchy_3d.cpp](../src/support/cauchy/direct_coefficient_cubic_cauchy_3d.cpp)、[extended_tubular_cauchy_3d.cpp](../src/support/cauchy/extended_tubular_cauchy_3d.cpp)。

P3 使用 \(J_0\) 至三阶、\(J_1\) 至二阶，结合调和方程闭合。未知 jet 来自三次密度基的解析导数；几何导数也是解析有理导数。

`--chart extended` 将冻结物理坐标中的多项式与

\[
X(u_0+\delta u,v_0+\delta v)+r\,n(u_0+\delta u,v_0+\delta v)
\]

做 Taylor 复合，得到 tubular 多项式。评价目标时，从该中心初始化局部法向投影：最大20步、参数步长上限0.25、步长停止阈值 `2e-12`。边界外使用首末 rational knot span 的延拓，不把目标参数 clamp 到 `[0,1]`。这里依然存在几何投影，只是固定锚点和评价行的复用减少重复计算；不能称“完全不做最近点计算”。

投影迭代只求一阶几何信息，返回残差与是否满足停止判据。它不是全局最近点或路径分支认证；未停止的投影被记录，非有限或奇异情况抛错。

P2 复用低阶 jet/映射，并按总阶截断，不能直接截取现有 P3 系数数组前10项，因为本地单项式顺序不同。当前 extended adapter 为获得低阶映射，P2-only 中心也会构造一次 cubic transfer，这项工作如实计入 P3 中心统计；这仍有进一步轻量化空间。`--chart physical` 保留冻结物理图坐标评价，二者是不同的评价策略。未实现 `clamped` 开关。

### 4.3 Restrict 的两层延拓仍完整保留

对每侧三个法向采样点，先以 P2 修正插值支撑内的错侧网格值，再做张量插值；随后用当前外迹点中心的 P2，把六个采样值统一到指定内/外分支。

六个法向位置为

\[
\rho=(-1.5,-0.75,-0.5,0.5,0.75,1.5),\qquad x_k=q+h\rho_kn(q).
\]

Neumann 使用 Q27-cover3，Dirichlet 使用 Q64-cover4。每侧三个样本共享空间 cover，最后用固定六点三次恢复矩阵取得值迹及带 `1/h` 的法向迹。六点三次恢复是固定的最小二乘恢复，不是未知密度拟合。

上述权重预装配为 `Rg/Rc/known`，通过已有 [resource_restrict_assembly_3d.cpp](../src/support/trace/resource_restrict_assembly_3d.cpp) 完成。中心、规划器与行缓存只属于 setup 生命周期，不在 GMRES 内重复建立。

### 4.4 显式保留对照开关

| `--policy` | Spread 中心 | Restrict 路线 |
|---|---|---|
| `trace`（默认） | 外迹优先，远距离回退事件 | 外迹优先 |
| `event` | 事件 | 事件路径 |
| `trace-spread` | 外迹优先，远距离回退事件 | 事件路径 |
| `event-spread` | 事件 | 外迹优先 |

`--event-mode all` 默认保留原生网格边全部事件。事件路径 Restrict 会保留认证路径全部转移，并以必要的 q 端项形成带符号修正组合；同侧端点不能预先清空该组合。装配器识别 `signed_extension`，不会再乘一次端点 label 差把多事件贡献抹掉。

`--event-mode python` 是显式比较模式：Spread 过滤同侧端点，并为每个有向邻接选靠目标端点的事件；事件式 Restrict 也采用端点邻近单事件规则。它是为了隔离与包内策略的差异，不能称比 all-event 更一般或更安全。

只有事件路径 Restrict 且使用 `all` 时，本规划器才把 `support_paths_certified` 标为真。默认 trace-first 即使 Spread 保留全部网格事件，也不意味着 Restrict 支撑路径已认证。

## 5. whole-line 几何复用与安全回退

### 5.1 整线根集复用

主要代码：[nurbs_cartesian_domain_3d.cpp](../src/geometry/nurbs_cartesian_domain_3d.cpp) 的原生网格准备流程，以及其中调用的 `certified_grid_line_can_partition_3d()`、`extract_grid_line_edge_3d()`。

启用 `--grid-lines on` 后，对同一轴向网格线只首次求一次整线交点，再将通过复用门控的完整根集分配给各条短边。索引包括轴向与线起点，不混合不同网格线；只缓存根，不为一条线预先存储 N 份大型边记录。

整线结果只有在可认证地分配给短边、且满足短边尺度的横截可靠性要求时才复用。不能因路径变长而放宽近相切判据。整线无法解析、门控失败或求交抛出相应运行时错误时，回到原有逐边求交及 targeted retry 路线；没有把有歧义的整线结果当成“无根”。

逐边流程仍检查完整根集和物理事件认证；构造原生修正支撑时，只要存在未认证候选边即停止。解析圆环公式只用于核验标签，不用于偷偷修复根集。

该优化是一次 setup 内的几何复用，不是 Python `roots+labels.npz` 的持久磁盘缓存。N/D 两个问题可以共享同一个网格几何准备；不同 N 的目录仍需各自建立。

### 5.2 N=64 暴露的空候选边认证顺序问题

本轮 N=64 的几何准备诊断中，实际返回的 4928 个交点事件全部已经认证，但另有 4 条未返回交点的候选边仍为 `unresolved`，因而被修正支撑的完整认证守卫阻止进入后续求解。这里必须区分“已找到的交点全部认证”和“所有候选边的根集完整性全部认证”：前者成立不意味着后者成立；没有返回根也不能直接解释为认证无根。

问题位于 [nurbs_bezier_intersection_3d.cpp](../src/geometry/nurbs_bezier_intersection_3d.cpp) 的 `classify_terminal_box()`。原逻辑先调用最近点迭代；一旦该迭代未收敛或距离不是有限数，就直接标为 `unresolved` 并返回，尚未尝试独立的 `certifies_terminal_separation()`。因此，迭代辅助步骤失败阻断了本可继续尝试的几何排除证明，并非这 4928 个事件中已经发现错误交点。

诊断期间，单纯把 terminal certificate 深度从 6 提到 16 没有解决这 4 条边；改用 h 尺度的几何细分只解决其中 2 条。前者尤其不能修复“到达证书调用之前已经返回”的控制流问题，因此本轮修复不以增加全局预算或放宽容差为方案。

修复保留最近点作为辅助工具，但不允许其失败否决独立的无根证书：

```text
最近点未收敛 / 距离非有限
  → 记录 closest_point_failures
  → 若当前 box 尚无已找到的根，尝试原有完整分离证书
      → 证书成立：该 box 认证为空
      → 证书不成立：仍标 unresolved
  → 若已有 box_root：不经此路径将其改判为空
```

证书仍采用原有正权有理 Bézier 控制凸包性质：曲面片位于欧氏控制点的凸包内。如果存在一个方向，使所有“控制点减线段两个端点”的投影都严格为正，并大于接触容差与 roundoff 保护量，那么整个控制凸包与整条线段分离，故该曲面片不可能与线段相交。当前接触容差为 `8 * geometry_tolerance`，投影另保留原有 `1024 * epsilon * projection_scale` 保护量；必要时仍在原定深度内细分，要求两个子片都能排除。这不是用最近点的距离估计、端点标签相同或“没有找到根”作为证明。

失败最近点给出的方向至多作为候选分离方向；非有限方向清零后，证书仍可从控制凸包自身生成其他候选方向。只有整套排除检查成立才返回认证空结果；证书预算用尽或无法分离仍保留 `unresolved`。有 `box_root` 的盒子不会被这条恢复路径清空，原有多根、相切和根集完整性守卫不变。

实施状态：四条空候选边与邻近双根控制回归已验证。在默认 `2h` 几何细分尺度、分离证书深度 6 下，四条边均返回 `certified_empty=true`、`unresolved=0`，各自的 `closest_point_failures=1` 仍如实保留；这表明消除的是无根认证的控制流障碍，并非把最近点迭代失败改写成成功。邻近控制边通过完整 BVH 候选查询，仍返回完整的两个横截根，参数为 `0.2802921791890603` 和 `0.70691717680697819`，没有被新路径误删。`nurbs_polar_intersection_3d_test` 与 `grid_edge_event_3d_test` 均已直接运行通过。

完整 N=64 的 Python 对照配置（`rotate / trace / extended / event-mode python`）现已完成端到端回归：进程退出码为 0，Neumann 与 Dirichlet 的 GMRES 均达到所设容差。实际记录见 [summary.json](../tests/cases/trace_first_3d/results/python64_certfix_20260909/summary.json)；[geometry.json](../tests/cases/trace_first_3d/results/python64_certfix_20260909/N64/geometry.json) 确认标签不一致数及未认证边数均为 0、4928 个横截事件全部认证、2396 条整线查询全部被接受，且没有 targeted retry。由此确认本配置的几何准备阻断已解除，具体误差、迭代和耗时以该运行目录及单独数值报告为准。

这一结果不等同于所有局部 tubular 投影或 Restrict 支撑路径均已认证；相应诊断仍按第 4、6 节解释。默认 `event-mode all` 的 N=32/64 最终计算及 N=64 整线开关对照现也已完成；具体误差、两网格观测阶、迭代、耗时与最终 11 项直接测试记录见 [数值报告](TraceFirst_Numerical_Report_20260909.md)。本节不把这些特定配置的结果扩展为所有几何或所有网格的收敛保证。

## 6. 数值诊断、结果口径与操作入口

### 6.1 运算前后检查

Neumann 在 GMRES 前以完整 C0 carrier 的常数系数测试 \(J_0=1,J_1=0\)：要求网格场复现内部指示函数、外部值迹接近零。探针不通过均值自由度映射，也不带制造解已知项，这是对 Spread/Poisson/Restrict 符号的一致性检查。Dirichlet 不把 \(J_1=1\) 错当作相同常跳跃测试，并明确记录未检查。

`--compare-cache` 对同一个几何及数学配置分别构造按访问计算和复用版本，比较固定矩阵及已知向量；这是缓存等价性检查，不是 C++ 与 Python 算子的整体等价性证明。`--dump-matrices` 可输出矩阵用于进一步审计。

结果记录包括：

- 原始参考/实际 carrier/最终约束自由度、外迹点数、C0 嵌入与均值残差；
- GMRES 次数及真实投影相对残差；
- 完整未投影的外部值迹、外部法向迹及投影泄漏；
- 内部最大误差、最大误差网格位置、trace Gauss 点上的密度误差；
- P3/P2 中心数、低阶复用数、行缓存命中数、远锚点及事件回退数；
- 投影最大残差和未满足停止条件的投影次数；
- 整线查询/接收/回退/复用边数和逐边查询数。

`interior_l2` 是内部网格点 RMS，`density_l2` 是面积权重归一化 RMS，不是未归一化积分范数。当前本地 `density_linf` 只对应 trace Gauss 点；Python 的独立非 Gauss 密度误差和 best-approximation 诊断没有在此驱动中全部移植。

### 6.2 时间口径

- `shared_geometry_seconds`：可由 N/D 共享的网格几何与修正支撑准备。
- `density_seconds`、`planning_seconds`、`spread_seconds`、`restrict_seconds`、`projection_setup_seconds`：对应各 setup 阶段。
- `affine_rhs_seconds`：已知部分/特解形成右端的工作。
- `gmres_seconds`：GMRES 调用自身耗时。
- `poisson_seconds` 和 `matvec_seconds`：累计内部计时，互相包含且与 GMRES 重叠，不能相加求总耗时。Neumann 的 Poisson 计数还包含常跳跃探针。
- `bvp_total_seconds`：单个 BVP 内的总计时，不含共享几何；会包括开启的缓存 A/B 与矩阵 dump 开销。完整进程时间另由 runner 记录。

### 6.3 文件与命令

[run.ps1](../tests/cases/trace_first_3d/run.ps1) 可按指定级别顺序运行；它不是常驻任务。示例仅说明入口，不表示这些运行已经完成：

```powershell
& ./tests/cases/trace_first_3d/run.ps1 `
  -Levels @(32,64) -Bvp both -Transform rotate -Policy trace `
  -Chart extended -EventMode all -GridLines on
```

以上命令在 PowerShell 会话中执行。若从其他宿主启动独立 PowerShell 进程，需另行正确传递数组参数，不能假设 `-File ... -Levels 32,64` 与原生 PowerShell 数组等价。

[python_reference_rotate.json](../tests/cases/trace_first_3d/python_reference_rotate.json) 保存包内四例既有记录，明确标注 `from_archive=true` 和 `geometry_cache_hit=true`。包内几何目录是热缓存，不能与本地冷启动直接计算“整体加速比”。

[compare_runs.ps1](../tests/cases/trace_first_3d/compare_runs.ps1) 只读现有运行结果，输出误差比与迭代次数差，并提示配置差异；不会自动重算或修改结果：

```powershell
& ./tests/cases/trace_first_3d/compare_runs.ps1 -InputPath <已有运行目录>
```

新增单元测试涉及 tubular Cauchy、精确插结/圆环密度布局、trace catalog 及整线复用等模块。实际编译、测试、数值运行及收敛阶结论以运行日志和单独报告为准；本文件不预先宣称通过，也不承诺所有几何二阶收敛。
