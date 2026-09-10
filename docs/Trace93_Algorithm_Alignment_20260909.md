# Trace93：新包算法对齐与代码设计说明

日期：2026-09-09。本文记录实现设计与代码对应关系，不作为本轮本地测试或数值收敛已经通过的证明。
参考包为 `KFBI3D-TracePointFirst-93-SourceAndSummary-20260909.zip`。
重点是圆柱、L 柱、U 柱的 Python 分析空间、仿射约束和 event-owner 修正中心策略。
旧 torus 独立入口和历史通用几何入口保留；新代码不是把所有旧算法的默认值整体替换。

## 1. 从哪些代码开始阅读

| 层次 | 代码与主要入口 | 职责 |
|---|---|---|
| 算例 | [trace93_case_3d.hpp](../src/support/geometry/trace93_case_3d.hpp)、[实现](../src/support/geometry/trace93_case_3d.cpp)；`make_trace93_case_3d` | NURBS 几何、analysis chart、刚体变换、制造解 |
| 密度 | [trace93_density_layout_3d.hpp](../src/support/density/trace93_density_layout_3d.hpp)、[实现](../src/support/density/trace93_density_layout_3d.cpp)；`build_trace93_density_layout_3d` | raw B 样条系数、约束、特解与零空间、迹点、投影附加行 |
| 事件映射 | [trace_first_resource_3d.cpp](../src/support/trace/trace_first_resource_3d.cpp)；`build_trace_first_geometry_plan_3d` | 从认证事件选择修正中心并生成有符号修正组合 |
| 多项式 | [trace93_resource_3d.cpp](../src/support/trace/trace93_resource_3d.cpp)；`Trace93PolynomialCatalog3D` | 原始系数直接生成 P3/P2，分析坐标评价与缓存 |
| 装配 | 同上；`build_trace93_bvp_operators_3d` | 固定 Spread、Restrict 和已知项 |
| 求解 | [trace93_study_3d.cpp](../apps/laplace/3d/trace93_study_3d.cpp)；`solve_case` | 仿射右端、投影、GMRES、真实残差与误差输出 |
| 批量运行 | [run_trace93.ps1](../tests/cases/trace_first_3d/run_trace93.ps1) | 统一参数、独立目录、限时、日志、档案对照 |

源码中的 `Trace93DensityLayout3D::traces[].u/v` 是 analysis 参数。
`RestrictResourceAnchor3D::u/v` 则是 native NURBS 参数；这两类数据不能混用。

## 2. 新的修正中心选择：交点先定 owner

旧 torus `TracePolynomialFirst` 路线在 Restrict 中先检查当前迹点 q 与网格点的距离：
距离不超过 `2.25h` 时优先 q，否则搜索 q 所属 patch 及其直接相邻 patch 的迹点。
它不能证明候选迹点属于实际穿过的几何分支；该旧路线不认证每条 support path。
新 `OwnerValidatedTraceFirst` 是另一条分支，而不是调小旧距离阈值。

新流程是：认证真实事件 → 保留该事件的 owner patch → 在该 patch 上尝试复用 trace 中心。
判定函数为 `owner_validated_trace_candidate_3d`，默认同时要求：

1. 候选迹点与事件的 `patch_id` 完全相同。
2. 迹点到事件的距离不超过 `1.75h`。
3. 迹点到被修正目标点的距离不超过 `3.25h`。

Spread 还要求对 Cartesian 边的两个端点都满足第三条。
Restrict 优先尝试当前 q；不满足时查询事件 owner patch 中距离事件最近的迹点，并再次检查全部条件。
若仍不满足，保留真实事件本身作为修正中心；不会换到邻 patch 或仅凭同一 smooth sheet 放行。
因此被替换的只是多项式展开中心，不是事件的位置、owner、符号或几何拓扑。
这两个距离条件是复用的局部性规则，不是 Taylor 截断误差上界或新的求交证书。

## 3. 新算例的身份与参数

以下尺寸均指施加刚体变换之前的局部坐标，三种柱体高度均为 `z ∈ [-0.5,0.5]`。

| 算例 | 物理区域 | patch 构成 |
|---|---|---|
| 实心圆柱 | 半径 `R=0.54`，高度 1；帽面中心正方形半边长 `a=0.22` 仅用于分片 | 4 个侧面 + 每个帽面的 1 个中心片、4 个环形分片，共 14 片 |
| L 柱 | `[-0.5,0.5]^2` 内满足 `x<-0.1` 或 `y<-0.1` 的截面 | 8 个侧片 + 上下各 3 个矩形，共 14 片 |
| U 柱 | `[-0.55,0.55]^2` 内满足 `y<-0.1` 或 `x<-0.22` 或 `x>0.22` 的截面 | 12 个侧片 + 上下各 5 个矩形，共 22 片 |

这里的 cylinder 是实心圆柱，不是此前的空心圆柱；L/U 也必须按这些新尺寸识别。
不能把旧同名几何的误差表当作本轮相同算例的结果。
圆柱 patch 顺序保持 Python：侧面 `0..3`，顶中心 `4`、顶环 `5..8`，底中心 `9`、底环 `10..13`。
side 的角度起点是 `-π/4 + quarter·π/2`，圆周方向为 u，轴向为 v。
帽面环片的 u 从内边走向外圆，v 沿圆弧；底帽反向保证外法向一致。

`rotate` 使用角度 `(17,-11,13)` 度，组合顺序为 `Rz Ry Rx`。
平移为 `(0.037,-0.029,0.041)`；`rotate_same_translate` 在上述旋转后加此平移。
`rotate_translate` 使用 `(31,19,-23)` 度并加相同平移；`translate` 只平移。
几何、法向和制造解一起变换，而不是在旋转后的几何上继续套未旋转的方程数据。

## 4. 双三次 NURBS 与角度 analysis chart 分离

原生几何工厂使用齐次 Bézier 升阶，把平面、圆柱侧面和帽面精确构造成 `(3,3)` NURBS。
`bicubic` 执行逐方向齐次升阶；不是从曲面采样点拟合控制点。
圆弧采用正权有理圆表示；帽面用内直边与外有理圆之间的精确平面参数化。
`connect_complete_edges` 匹配完整边、核对方向并建立 smooth/feature 拓扑。

但相同物理圆周不意味着相同参数函数：Python 使用线性角度和三角函数，NURBS 使用有理参数。
三次 B 样条在两种参数下定义的密度函数空间一般不同，不能通过有限次插结点精确互换。
本轮因此不把圆柱 Python 密度强行嵌入原有 `NativeNurbsDensitySpace3D`。

新布局在 Python analysis chart 上直接保存原始系数；native geometry 只负责几何查询与事件 owner。
`analysis_at` 给出位置及最高四阶解析参数导数，侧面保留线性角度，帽环保留 Python 的直边/三角圆弧插值。
`world_to_analysis_uv` 用世界点恢复分析坐标；侧面用角度展开，帽环用带残差检查的参数反算。
`analysis_to_native_uv` 为已有 native-endpoint 求交接口提供真实 NURBS 坐标。
L/U 为仿射 chart，analysis 与 native 参数相同；圆柱明确标记 `native_parameters_match_analysis=false`。
这里的参数反算不是未知密度拟合，也不替代原生求交认证。

## 5. raw 三次密度、细化与外迹采样

每片使用开区间、均匀结点、三次张量积 B 样条：
`μ_p(u,v)=Σ c[p,i,j] B_i(u) B_j(v)`；原始编号为 `patch_offset + j*ncu + i`。
`reference_raw_dofs` 指施加任何约束之前的系数总数，不是旧 Native C0 合并后的数量。
基函数值及一至三阶导数由系数直接解析计算；没有从邻近密度样本拟合 jet。

平面片采用每方向 `max(2,ceil(L_direction/(8h)))` 个 spans。
圆柱按 analysis chart 的物理方向长度选择 dyadic spans：10 点 Gauss 积分、7 条横向测线取最大长度。
再按连接边的参数方向关联块同步细化数，确保两侧边 B 样条结点一致。
恰好整数的长度比减去机器舍入量，避免刚体旋转产生几个 ulp 就意外多细化一层。
这一配置复现新包的细化规则；它不是“约束后 DOF 太少就继续插结点”的自动自适应循环。

每个 density cell 使用 `4×4` Gauss 外迹点，权重乘 analysis chart 的面积 Jacobian。
必须有 `trace_count > reduced_dofs`，求解器还检查迹质量矩阵的正定性与 Cholesky 主元比例。
数量不等式只是必要检查，并不单独证明迹空间可观测或精度足够。

下表是包内 rotate 配置的对齐目标，不是本轮本地已经通过的计数结果：

| 几何 | N | raw DOF | D reduced | N reduced | 外迹点 |
|---|---:|---:|---:|---:|---:|
| L | 32 | 350 | 122 | 67 | 896 |
| L | 64 | 478 | 202 | 127 | 1792 |
| U | 32 | 550 | 190 | 109 | 1408 |
| U | 64 | 718 | 294 | 189 | 2560 |
| cylinder | 32 | 350 | 126 | 101 | 896 |
| cylinder | 64 | 574 | 270 | 229 | 2560 |

## 6. 仿射约束：Dirichlet 与 Neumann 分开

原始离散约束记为 `C c=d`，迭代坐标为 `c=c_p+Zy`；`Z` 列数才是 GMRES 未知数数量。
不能因此宣称原始等式一定精确可行：Python 对不相容的右端使用最小二乘特解。
令 `δ=C cp-d`，实际迭代空间保持 `C c=C cp=d+δ`，而不是在 GMRES 中消除这项固定离散缺陷。
同一 smooth sheet 内的分片接口保留 C0 与一阶相容条件；特征棱边不能不分 BVP 地复制相同约束。

Dirichlet 已知 `J0=gD`、未知 `J1=μ`；不强制两个特征面上的 μ 相等。
由 `∇Γa gD + μa na = ∇Γb gD + μb nb` 求两侧所需的边界密度值。
L/U 把该已知边数据在 Greville 点离散后解方阵，转为边系数的仿射已知部分。
这一步是已知边界数据的样条插值，不是用算出的场去拟合未知密度；非多项式数据仍有离散误差。
各特征边独立插值后的端部导数不必与所有 smooth C1 条件精确相容；这在参考源中已经存在。
包内 rotate 的 L32 D `||Ccp-d||∞=1.388067457558373e-6`，L64 D 为 `4.967766834840237e-7`。
U32/U64 D 也分别记录 `4.8586834837660575e-6`、`1.7897361606067363e-6`，不能当作舍入零。
这些值是档案证据，不是完整 PDE 误差；单测核对档案缺陷及 `Cᵀδ≈0`、`CZ≈0`，不靠放宽精确可行阈值掩盖。
圆柱采用包内 `junction_defect`：特征边仅保留内部 Greville 点的两侧密度值条件，跳过两个端点。
这不是添加额外 vertex 连续性，也不是给 D 增加 Polar-Star 或 edge/vertex 投影权重。

Neumann 已知 `J1=gN`、未知 `J0=μ`；essential 部分包括 smooth C0/C1、feature C0 和均值约束。
feature 梯度相容性采用 Polar-Star 弱约束，不沿用旧的全部点值特征导数行。
在统一边参数上，令 `Q` 的行依次为公共单位切向 τ 与两侧外法向 na、nb。
`inward` 是带符号的参数方向 Xu 或 Xv 归一化，不是先去除边方向后的 co-normal。
若 `ca=maᵀQ⁻¹`，则一侧残差行为 `Da-c_a0 Dτ`，已知项为 `c_a1 gNa+c_a2 gNb`。
另一侧同理；其中切向项在帽环非正交参数下不能删掉。
将这些残差乘公共边三次 B 样条测试函数并积分；L/U 每 cell 用 10 点，圆柱用 12 点 Gauss。

先约化 essential 条件，得到 `(c0,Z0)`，再对 `Cp Z0` 做 SVD。
数值秩内若存在至少 10 倍的最大相邻奇异值间隙，保留间隙之前的左奇异方向；否则保留全部数值秩。
被保留的弱约束与 essential 条件合并，再计算最终 `(cp,Z)`；这不是所有弱行无条件硬塞入系统。
日志输出 essential rank、polar rank、保留数、gap、left-null RHS 和最终约束残差。

约束行先归一化，过小行按源代码规则跳过。
`reduce` 对 `Cᵀ` 做列主元 QR，以 `max(m,n)·eps·|R00|·factor` 判秩并取得零空间。
factor 为最终平面 30、圆柱 40，Polar 的 essential 预约化使用 80。
特解按 SVD 最小范数约束最小二乘解取得，不是密度样本拟合；若 `δ≠0`，原始边数据等式并非全满足。
特别保留 Python 的实际容差复用：它将上述 QR 的 tol 原样传给 `lstsq(cond=tol)`。
SciPy cond 和 Eigen `setThreshold(tol)` 都相对最大奇异值；本地不擅自改成 `tol/σmax`。
因此 QR 判秩门限与特解 SVD 的绝对门限不是同一个表达式，这是源实现本身的选择。

## 7. 投影、仿射方程和 GMRES

设 `B` 为 trace density basis，`Br=BZ`；附加的边/角密度矩阵约化为 `Ber`、`Bvr`。
Neumann 默认 edge-star 权重为 0.1；L/U 的 vertex-star 接口保留，但包内默认权重为 0。
圆柱当前不加入 vertex-star；Dirichlet 不加入 edge/vertex 投影项。
`edge_trace`、`vertex_trace` 从已有 4×4 Gauss trace 值作局部三次外推，不引入新的未知密度拟合。
权重数组已经包含各自系数，不能在求解器中再次乘 0.1 或额外乘 h。

质量矩阵为 `M=BrᵀWBr+BerᵀWeBer+BvrᵀWvBvr`。
对完整外迹 t 的投影为 `Π(t)=M⁻¹[BrᵀWt+BerᵀWe(edge_trace t)+BvrᵀWv(vertex_trace t)]`。
按 Python 规则，若检测到非正特征值则记录并施加对角修正，再做 Cholesky；不是默认改成 QR 投影。
表面制造解的常数由每 geometry patch 的 8×8 analysis 积分去均值。
Neumann 密度 gauge 则来自 density cell 的 4×4 trace 面积积分；两种积分不能混为一谈。

记 `L⁻¹` 为实际离散 Poisson 求解调用，定义 `U(c)=L⁻¹[-(Sc+bS)]`。
先计算 `U0=U(cp)`，仿射右端为 `-Π(Rg U0+Rc cp+br)`。
GMRES 算子为 `y ↦ Π(Rg L⁻¹[-SZy]+Rc Zy)`。
最终用 `c=cp+Zy` 重新计算完整场和完整外迹；已知项不进入线性 matvec 的重复累加。
N 使用外侧值迹方程，D 使用外侧法向迹方程；两种完整外迹都另外输出供诊断。

默认 GMRES 相对容差 `2e-10`；L/U Dirichlet restart `min(100,nred)`、最多 8 周期。
Neumann 和圆柱 Dirichlet restart `min(120,nred)`、最多 12 周期。
本地总迭代上限换算为 restart×周期，并单独重算真实投影残差，不只相信递推残差。
Neumann 在 GMRES 前有常数 jump 探针：raw 系数全 1 应恢复域内指示函数及零外侧值迹。

## 8. P3/P2 修正与缓存边界

Spread 使用 P3，Restrict 使用 P2；N 为 Q27-cover3，D 为 Q64-cover4，均保留 3+3 法向采样。
法向样本位置为 `h·{-1.5,-0.75,-0.5,0.5,0.75,1.5}`。
新 93 入口对空间 cover 的整数取整启用 `1e-13` 偏移对齐；旧 torus 默认 0 不变。
support 越出背景盒应失败，不通过截断或外推网格索引掩盖；小权重也不做 `<1e-16` 删除。
平面使用物理切平面多项式；圆柱侧面使用角度/轴向/径向 chart，不代换为旧 rational-NURBS tube 投影。
`cauchy_plan` 从解析密度 jet 和几何 jet 构造物理 Cauchy closure，再解析复合到所需 chart。
同一中心只构造一次 transfer，P2 取其总次数不超过二的部分，因而可复用 P3 建设成本。
部分仅用于 P2 的中心也会构造该 transfer；`p3_centers` 不能解释为 Spread 独占中心数。

先完成实际事件/中心选择，再按 `(degree,source,anchor_id,grid_id)` 合并多项式评价行。
多事件组合的请求键还包括所有项的身份与符号，不能简化成 `(sheet,grid)`。
路径查询按 `(q_id,grid_id)` 复用；法向 sample 无 grid ID，不混入网格行缓存。
缓存保存系数到修正值的固定映射，不保存某次迭代的密度数值。
setup 后释放几何计划、中心和临时行缓存，GMRES 只访问固定矩阵与 Poisson 求解器。

## 9. 事件模式、认证与安全回退

`--event-mode python93` 是明确的算法对照模式，不是宣称复制 Python 的所有求根容差细节。
Spread 对正轴方向无向边采用最后一个事件，并将同一个事件用于两个差分方向；仅处理端点异侧边。
Restrict 对错侧访问采用最靠近网格点的开放事件；没有开放事件时按参考规则保留 q 修正。
本地仍先认证从网格点到 native q 端点的完整路径，不把“没有找到事件”默认为已认证无根。
`--event-mode all` 则保留完整有符号开放事件组合，并补上目标侧与路径入射侧差值对应的 q 项。
相同端点 label 的多交点路径不能靠 parity 预筛掉；all 模式也检查这些路径。

原生边/路径求交的完整性证书、owner 和 transition 符号检查继续生效；未认证路径停止 setup。
整条 Cartesian 轴线的根集合仅在原有完整性、近节点、近切触、feature 防护通过时复用。
不能接受的整线退回原逐边算法，不能将整线缓存误称为新增的显式根区间证书。
已有“最近点失败后仍尝试独立正权控制凸包分离证书”的安全分支继续保留。
只有完整分离证书成功才认证空集合；有 box root 时不能走空恢复，无证仍 unresolved。
本轮没有复制固定端点裁剪、仅按 root 距离合并或近根猜测来换取表面上的 Python 数字一致。
也没有实现 DDA/cut-cell 拓扑捷径；同 owner 加距离限制不能冒充省掉准确事件查询的证明。

## 10. 统一运行、档案与验证状态

新目标为 `kfbi_trace93_study_3d`；统一脚本默认三种几何、N32/64、rotate、两种 BVP、trace/python93。
运行入口：`./tests/cases/trace_first_3d/run_trace93.ps1 -Geometries cylinder,l,u -Levels 32,64`。
可用 `-Policy event`、`-EventMode all`、`-GridLines off` 和 `-CompareCache` 做单因素对照。
脚本使用独立进程和唯一结果目录，保留 stdout/stderr、每 BVP result 与汇总 summary，不覆盖已有结果。
旧 `kfbi_trace_first_study_3d` 和 [run.ps1](../tests/cases/trace_first_3d/run.ps1) 仍用于旧 torus 方案。

[python_reference_trace93.json](../tests/cases/trace_first_3d/python_reference_trace93.json) 保存包内 72 条记录。
来源是压缩包 `results/fresh_run_manifest.json`，不是本地重新运行 Python：3 几何×2 BVP×4 变换×3 网格。
网格为 32/64/128，四变换为 rotate、translate、rotate_same_translate、rotate_translate。
包内没有完整 fresh_runs 目录；不能声称已获得全部原始矩阵、详细日志或独立验证点数据。
部分 planar Neumann result 缺 bvp、h、trace_points、crossings 等字段：身份取 job，未记录指标保留 null。
`null` 不是零，不据此计算计数差；档案误差与时间也不冒称本地新测。

C++ 的 Spread 命中/回退按有向 crossing operation 计数，Python 通常按无向边计数，两者常差两倍。
应结合 `active_directed_spread_operations` 和真实事件数解读，不能单凭两倍计数认定算法不一致。
本地输出 DOF、约束残差、真实投影残差、完整外迹、密度/域内误差、最大误差点和分项耗时。
表面密度误差是在 trace Gauss 点上的检查，不是独立验证点误差；仅一对网格只能给出观测阶。
几何认证更严格、平台与线程/缓存状态不同，包内时间不能直接用于宣称 C++ 加速比。

专用单测位于 [geometry](../tests/geometry/trace93_case_3d_test.cpp)、[density](../tests/density/trace93_density_layout_3d_test.cpp)、[trace](../tests/trace/trace93_resource_3d_test.cpp)。
它们分别检查几何/参数映射、DOF/约束/解析 jet/投影再现、修正与资源接口；存在测试源码不等于已经通过。
本说明不填写本轮本地 PASS 状态或数值表，后续只依据实际完成的日志与 JSON 单独形成结果报告。
