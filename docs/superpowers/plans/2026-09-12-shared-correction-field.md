# Shared Correction Field KFBI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 C++ Trace93 入口和公共 KFBI 流程中实现共享修正场后端，并完整区分原始外迹与半跳跃闭合后的方程目标。

**Architecture:** 原几何、密度约束、投影、Poisson 和 GMRES 保留；公共仿射流程调用统一修正接口。旧事件多项式后端包装已有矩阵，新共享场后端以固定稀疏拟合分解和联合 S/R 评价实现，避免独立求解程序和稠密总映射。

**Tech Stack:** C++17、Eigen 3.4、现有 CGAL 5.6+、zFFT、CMake 3.20+、CTest；Python 仅作经过审阅的对照导出和报告工具。

**Spec:** [完整设计](../specs/2026-09-12-shared-correction-field-design.md)。执行前同时阅读设计与本计划。

**Status:** T1–T8 的实现和规定验证已完成：36 组对照全部执行，32 组通过，4 组复现已知 N32 raw 研究失败；12 组半跳跃候选全部通过。21 项相关 C++ 测试中 20 项通过，唯一 torus 原有失败已在基点独立复现；12 项 Python 验证脚本测试通过。最终证据见 [验证报告](../../Shared_Field_KFBI_Validation.md)。以下细项保留原计划，合并测试、工作区回退和验证替代的具体范围以报告及 ledger 为准，不将历史未勾选项当作当前运行状态。

## Global Constraints

- 未指定新后端时仍是现有 direct_cauchy，保留当前所有默认行为。
- 首期实现范围是现有 `kfbi_trace93_study_3d` 入口、U 柱、Laplace、Dirichlet/Neumann、两种包内姿态、N32/64/128。
- 新共享场数值文件组成内部 target `kfbim_3d_shared_correction`，只在 `KFBIM_BUILD_EXPERIMENTAL_3D=ON` 时构建。
- 所有公共修正后端返回 `rhs_for_delta`，可直接传入现有 Delta bulk solver；公共求解流程不追加负号。
- 保持密度→拟合因子求解→场系数→共享评价→S/R 的因式应用；禁止形成稠密总映射。
- 内层矩阵、正则值、采样、伪逆截断和两次 refinement 在 setup 固定；matvec 不查询几何和精确解。
- 同一次 forward 只拟合一次共享场。requested/fitted jump、raw/equation trace 分别保留。
- 不将 `tmp/` 审阅副本用作构建、测试或运行依赖；来源包 hash 固定为 `23cfa03489e1d1004a57f39703e1f4237d83afb6da096fed7cc7033ecb884ce5`。
- 本计划不启用 moment 保矩实验，不扩展 production topology 默认算法，不修改现有生产几何认证语义。

## 0. 执行准备与结构图

开始实施时先按工作区规则建立隔离分支，建议名 `codex/shared-correction-field`。不得删除用户现有的未跟踪 `experiments/`、`tmp/`。

目录与所有权：

```text
src/support/
  correction/
    shared_field_geometry_3d.{hpp,cpp}    # U 柱参考几何、拟合采样
    shared_field_space_3d.{hpp,cpp}       # 共享 C2 体样条
    shared_field_extension_3d.{hpp,cpp}   # 固定内层 LS
    shared_field_backend_3d.{hpp,cpp}     # 原密度 -> shared 修正接口
  trace/
    kfbi_correction_backend_3d.{hpp,cpp} # 公共契约、旧矩阵适配器
    affine_trace_coordinates_3d.{hpp,cpp}# Trace93 lift/project
    shared_field_transfer_3d.{hpp,cpp}   # 共享 Spread/Restrict
    jump_identity_closure_3d.{hpp,cpp}   # raw -> equation
  solver/
    affine_kfbi_solve_3d.{hpp,cpp}       # 公共外层流程
tests/
  correction/                          # 空间、拟合、对照数据
  trace/                               # 传递、闭合、坐标回归
  solvers/                             # 仿射流程和集成
tests/cases/shared_field_3d/             # 清单、少量固定 fixture、运行入口
scripts/validation/                    # 独立对照导出与核验
```

任务依赖：T1→T5；T2→T3→T4→T5；T6 导出器可在 T2/T3 期间准备，对照消费 T5；T7 消费 T6；T8 消费全部结果。T1 与 T2 可由不同执行者并行，T3 与 T4 只在空间接口固定后并行。

本计划的接口代码是契约示意，完整类型以设计第 10.3 节及各任务定义为准；不要求把文档里的片段作为独立程序保存。

数值比较的统一约定：除明确给出来源绝对误差门槛外，向量归一化差为 `||a-b||inf / max(1,||a||inf,||b||inf)`，导数先乘对应的 h 或 h²；多项线性/仿射恒等式的分母额外包括各项范数。矩阵比较使用相应诱导范数并记录所用范数。内层后向残差为 `||Kz-r||inf / (||K||inf*||z||inf+||r||inf)`，其中 `r=Aᵀ(q-Mlift)`；分母为零且分子为零时记0，否则记失败。GMRES真实相对残差仍按原来的2范数与右端2范数定义，不用max(1,...)替换。

本地旧 C++ baseline 与包内 Python baseline 是两个独立身份。T1只验收本地抽取前后不变；T6/T7先核对几何、density、投影、中心/event及局部坐标，再决定是否作同算子数值匹配。尚未对齐的baseline必须标为不同离散化对照。

## T1. 抽取公共流程并保持旧算法结果

**Files:**
- Create: `src/support/trace/kfbi_correction_backend_3d.{hpp,cpp}`。
- Create: `src/support/trace/affine_trace_coordinates_3d.{hpp,cpp}`。
- Create: `src/support/trace/jump_identity_closure_3d.{hpp,cpp}`（本阶段实现 RawTrace 分支及接口校验）。
- Create: `src/support/solver/affine_kfbi_solve_3d.{hpp,cpp}`。
- Modify: `apps/laplace/3d/trace93_study_3d.cpp`、`src/support/CMakeLists.txt`、`tests/CMakeLists.txt`。
- Test: `tests/trace/affine_trace_coordinates_3d_test.cpp`、`tests/solvers/affine_kfbi_solve_3d_test.cpp`。

**Interfaces:** 消费已有 `ResourceBvpOperators3D`、`Trace93DensityLayout3D`、`ILaplaceBulkSolver3D`、GMRES；产出设计中 `ICorrectionBackend3D`、`IAffineTraceCoordinates3D`、`TracePair3D`、`CorrectionEvaluation3D` 和 `close_exterior_target_3d`。

同时落实设计中的 `CorrectionBackendDescriptor3D` 与 `CorrectionDiagnosticSnapshot3D`：setup检查尺寸、布局ID和半跳跃能力；每次evaluation带实例内ID，最终快照绑定该ID且不重拟合。

公共 solver 定义 `AffineKfbiEvaluation3D`：`grid_solution/raw_trace/equation_trace`；定义 `AffineKfbiSolveOptions3D`：`neumann/target/relative_tolerance/restart/max_iterations`；定义 `AffineKfbiSolveResult3D`：最终 raw 系数、reduced 坐标、evaluation、真实残差与 GMRES 状态。使用 `evaluate_affine_kfbi_3d(backend, bulk, c, part, target)` 与 `solve_affine_kfbi_3d(backend, coordinates, bulk, options)`。

- [ ] 先保存当前 L32 两 BVP 的一次受控基线，连同配置、commit、raw trace、projected residual 和内误差；未完成运行保留状态，不拿历史档案替代此次回归。
- [ ] 编写小矩阵后端/盒子算子的可计算夹具，验证符号、调用顺序、基场及 known 一次性加入。示例逻辑：

```cpp
auto full = evaluate_affine_kfbi_3d(backend, bulk, cp + c1,
    ApplyPart3D::WithPrescribedData, ExteriorTarget3D::RawTrace);
auto base = evaluate_affine_kfbi_3d(backend, bulk, cp,
    ApplyPart3D::WithPrescribedData, ExteriorTarget3D::RawTrace);
auto inc = evaluate_affine_kfbi_3d(backend, bulk, c1,
    ApplyPart3D::Homogeneous, ExteriorTarget3D::RawTrace);
assert((full.grid_solution - base.grid_solution - inc.grid_solution).norm() < 1e-12);
```

- [ ] 实现旧适配器：`rhs_for_delta=-(S*c+known_spread)`；raw corrections 和 Rg 按旧矩阵直接计算。
- [ ] 原样抽取 Trace93 Gram/edge/vertex 投影；保留权重、正则诊断和 reduced 坐标，不改成 topology 的 mass 坐标。
- [ ] 增加不相同layout ID、raw/trace尺寸错配、零homogeneous输入和 `observe_grid()` 不受上次evaluate影响的测试；验证旧后端快照不含共享场系数。
- [ ] 将原 app 的基场、matvec 和最终恢复接入公共 solver。应用层保留几何准备、选项和报告，公共 solver 不获得制造解回调。
- [ ] 运行新测试、`trace93_resource_3d_test`、`trace93_density_layout_3d_test`、`resource_restrict_assembly_3d_test`，再比受控 L32 基线。归一化算子差暂定 ≤1e-12；若不满足，定位抽取变化再继续。
- [ ] 检查 diff 中没有新默认行为或其他程序重构，记录本任务独立提交。

## T2. 共享体样条空间与几何采样

**Files:**
- Create: `src/support/correction/shared_field_geometry_3d.{hpp,cpp}`、`shared_field_space_3d.{hpp,cpp}`。
- Create: `tests/correction/shared_field_basis_3d_test.cpp`。
- Modify: `src/support/CMakeLists.txt`、`tests/CMakeLists.txt`、`docs/architecture/README.md`。

**Interfaces:** 定义 `SharedFieldGeometry3D`（刚体变换、参考矩形、包围范围），`SharedFieldSurfaceSamples3D`（reference/world 点、reference/world 法向、正权重、patch、analysis uv），`SharedFieldSpaceOptions3D`（h、ratio、width、体求积阶）。`make_trace93_shared_field_geometry_3d(problem)` 只接受首期 U 柱仿射片。

`SharedFieldSpace3D` 提供 `coefficient_count()`、`contains(reference_points)`、`evaluation_matrix(reference_points, derivative)`、`coefficient_lattice()`、曲面/体采样访问器。`SharedFieldDerivative3D` 明确定义 Value、Dx/Dy/Dz、Dxx/Dyy/Dzz、Dxy/Dxz/Dyz、Laplacian。点数组为 `std::vector<Eigen::Vector3d>`；评价矩阵统一 CSR。

- [ ] 编写基函数分片表达、负坐标 floor、张量索引顺序、支持集合、共享编号和边界覆盖的失败测试。
- [ ] 将包内 cardinal 表达及 H 尺度解析移植；所有活动 cell 收集 64 个系数索引后排序去重，严禁按面分别编号。
- [ ] 实现参考矩形上的 H 格点切分和 4×4 面求积、活动体单元上的 3×3×3 求积。按面积/体积分别归一化留给拟合层，原始物理权重同时保存。
- [ ] 验证旋转下位置/法向/梯度转换以及参考空间不随物体姿态意外改变。测试至少包含负参考坐标、结点面两侧、两面交边和顶点邻域。
- [ ] 验证 `sum B=1`、`H sum B'=0`、`H² sum B''=0`，无量纲误差暂定 ≤1e-13；用同一随机系数从不同访问顺序评价，值与一二阶导数一致。
- [ ] 注册实验库 `kfbim_3d_shared_correction` 和单测 `shared_field_basis_3d_test`；实验开关关闭时不编译该库，2D 配置不被新增依赖影响。
- [ ] 记录空间/索引结果后独立提交。场自由度 U32/64/128 的来源目标为 1243/3365/13687，计数不符先查支撑和参考包围盒。

## T3. 固定线性 Cauchy 拟合

**Files:**
- Create: `src/support/correction/shared_field_extension_3d.{hpp,cpp}`。
- Create: `tests/correction/shared_field_fit_3d_test.cpp`。
- Modify: 对应 support/test CMake 源列表。

**Interfaces:** 定义 `SharedFieldFitMatrices3D`：`trace/normal/laplacian`、面/体权重、曲面20项多项式矩阵、系数格点lift矩阵；`SharedFieldFitOptions3D`：三项权重、ridge、lift伪逆截断、固定refinement次数。`SharedFieldExtension3D(matrices, options)` 在构造时固定分解，`solve(value_jump, normal_jump)` 返回 `Eigen::VectorXd` 场系数，`diagnostics(coef,a,b)` 返回归一化最优性与拟合残差。

- [ ] 写常数、线性、调和二次和调和三次的界面 Cauchy 输入测试。测试必须调用真正的 `solve`，不能只把已知多项式系数交给下游。
- [ ] 按设计第4节生成归一化 M、列尺度、20项 lift 伪逆、K。确保正则惩罚围绕 lift，而非误写成 `lambda*||alpha||²`。
- [ ] 采用 Eigen `SimplicialLDLT`/AMD 固定分解；检查数值状态、有限性与正主元，执行两次规定的正则化残差改进。
- [ ] 以 h 缩放一阶、h²缩放二阶量，调和 P0–P3 拟合再现暂定 ≤1e-10；正则系统归一化后向残差暂定 ≤1e-11。若与 Python 分解结果不符，优先比较矩阵、右端、最优性与场值，不直接要求系数位级一致。
- [ ] 独立测试线性组合：`solve(.7*a1-.3*a2,.7*b1-.3*b2)` 与 `.7*solve(a1,b1)-.3*solve(a2,b2)` 的归一化差暂定 ≤1e-10。
- [ ] 对不相容 Cauchy 数据确认正常返回非零 fit 残差，不伪称逐点满足；对非法权重、缺失支撑和分解失败返回明确错误。
- [ ] 确认一组固定矩阵只分解一次，参数变化必须重建；记录矩阵/因子nnz与内存估计后独立提交。

## T4. 共享 Spread/Restrict 传递计划

**Files:**
- Create: `src/support/trace/shared_field_transfer_3d.{hpp,cpp}`。
- Create: `tests/trace/shared_field_transfer_3d_test.cpp`。
- Modify: support/test CMake。

**Interfaces:** `SharedFieldRestrictMode3D { Staged, Direct }`；`SharedFieldTransfer3D` 消费完整 grid、labels、外层迹点、空间评价器和纯 cover stencil。`evaluate(field_coefficients)` 返回 `SharedFieldTransferResult3D { rhs_for_delta, raw_correction }`；`observe_grid(full_solution)` 返回 `TracePair3D`。矩阵和索引在 setup 固定。

- [ ] 先写给定共享系数的常数与调和 P2 测试，验证 `U=chi*D`、零外迹和旋转后梯度方向；该测试与 T3 的拟合测试分开。
- [ ] 用 `grid.index/coord` 构造所有有向标签变化边，直接生成 Delta 右端符号；验证完整盒子边界行是零，零 Dirichlet边界不是普通内部未知量。
- [ ] 复用 `build_shared_side_cover_restrict_stencil_3d` 的几何权重，核对包内 cover/snap/取整规则；不调用 owner/path 总规划器。
- [ ] 合并 Spread 与 Restrict 网格请求为一个有序唯一集合，构造 Eg、Sg、Jv、Jn；staged 再构造负法向 Eminus、Nv、Nn。一次 evaluate 只计算一次 Eg*coef。
- [ ] 对显式多次穿越但端点同标签的夹具验证净符号为零，并检查所有贡献评价的是同一个 D(xj)；加入错误按面分配系数应失败的独立断言。
- [ ] N32 给定调和 P2 的网格/值迹误差 ≤2e-12，法向 ≤2e-11；cover索引与 Python 精确一致，Rg值 ≤2e-14、Rg法向 ≤5e-13。更细网格另外报告 h 缩放误差。
- [ ] 覆盖不足和盒外查询必须失败；装配后删除几何事件回调也能 evaluate；独立提交。

## T5. 共享后端、半跳跃闭合和现有入口

**Files:**
- Create: `src/support/correction/shared_field_backend_3d.{hpp,cpp}`。
- Modify: `src/support/trace/jump_identity_closure_3d.{hpp,cpp}`、`apps/laplace/3d/trace93_study_3d.cpp`、`apps/CMakeLists.txt`。
- Create: `tests/trace/shared_field_half_jump_3d_test.cpp`、`tests/solvers/shared_field_kfbi_3d_test.cpp`。
- Modify: support/test CMake。

**Interfaces:** `SharedFieldCorrectionBackend3D` 实现 T1 的修正接口；消费 geometry/space/extension/transfer、拟合点与外层迹点的原密度采样矩阵、setup 固定的规定数据数组。factory `make_trace93_shared_field_backend_3d(problem,layout,grid,labels,options)` 只在实验 target 内提供。

- [ ] 为 requested/fitted 两组 jump 与 raw/equation 两组迹编写独立测试；`InputJumpHalf` 缺少任一 jump 数组、维数不符或在未支持迹点类别使用时必须失败。
- [ ] 从原 `basis_stencil(patch,u_analysis,v_analysis)` 组装拟合密度矩阵，不从离散密度样本再次拟合密度；已知数据仅在 setup 评价。
- [ ] 同一次 evaluate 调一次 extension.solve，再一次 transfer.evaluate，并用同一 alpha 评价 fitted jump；保存最终场系数用于诊断，禁止不同模块自行重拟合。
- [ ] 在最终forward后立即保存同ID诊断快照；测试之后做其他matvec不改变已保存快照，对过期ID请求快照必须失败。核对 case/layout/context 生命周期，禁止悬垂引用。
- [ ] 实现闭合 `equation=raw+0.5*(fitted-requested)`。逐点检查值、法向恒等式，归一化缺陷暂定 ≤1e-12；确认 raw数组和U未被原地覆盖。
- [ ] 接入公共仿射流程，检查 `F(cp+Zy,true)-F(cp,true)-F(Zy,false)` ≤1e-10；用假的禁止访问精确解/事件回调，验证 matvec 不读取它们。
- [ ] 增加设计第12节 CLI 与明确组合校验。保留旧默认；shared显式启用时默认候选半跳跃目标，输出完整解析配置。单独保留 raw目标和 staged/direct恢复的两个维度。
- [ ] 条件链接新实验库；分别验证实验开关 ON/OFF 的构建和旧入口行为；完整输出 raw/equation/fit 三组诊断后独立提交。

## T6. 来源固定与 Python/C++ 分层对照

**Files:**
- Create: `tests/cases/shared_field_3d/README.md`、`cases.json`、`provenance.json`、`fixtures/` 下小规模固定文件。
- Create: `scripts/validation/export_shared_field_reference.py`、`compare_shared_field_reference.py`。
- Create: `tests/correction/shared_field_python_fixture_3d_test.cpp`。
- Modify: tests CMake。

**数据契约:** `cases.json` 固定 U几何尺寸、姿态矩阵、N、BVP、density因子、field配置、target和恢复模式。`provenance.json` 保存本来源包和每个被用源文件的 SHA256、导出脚本版本、Python/NumPy/SciPy/BLAS版本、排序/坐标/标签约定、fixture hash。禁止把旧72条 Trace93 manifest 当共享场参考。

- [ ] 从来源包主表的 source 字段选择 `baseline_fresh/baseline_pose2/validated_small/raw/remaining128`，排除 prototype、upstream历史、moment、H2和开发重复运行。
- [ ] 独立审阅并实现导出脚本；仅导出小规模曲面/体采样、basis、lift、拟合右端、场评价、S/R输出及最终参考摘要，避免把大批历史数据放入源码树。
- [ ] 固定 seed 和完整 raw 密度方向，导出真正通过拟合的结果。C++读取小文本/MatrixMarket夹具；不新增运行时 Python/Numpy 依赖。
- [ ] 比较原生标签与包内解析标签；真实集成仍使用 native labels。先按物理点和raw系数序核对，再比较空间和operator。
- [ ] 约化基底不同则先检验维数、约束子空间和 ZZ^T，再用同一合法 raw方向比较；不得直接逐元素比较未对齐的 Z 或 reduced矩阵。
- [ ] 继承可用矩阵门槛；共享拟合场/随机方向的归一化 Python/C++差暂定 ≤1e-8，内层最优性还须满足 T3。阈值首次对照后可有证据地修订，不能仅为通过而放宽。
- [ ] 保存对照失败的阶段和差值分布；只有分层通过才进入大规模 PDE；独立提交。

## T7. 真实外层矩阵、收敛与资源验证

**Files:**
- Create: `tests/cases/shared_field_3d/run_shared_field.ps1`。
- Create: `scripts/validation/summarize_shared_field_3d.py`。
- Modify: `tests/cases/shared_field_3d/README.md`。
- Generate at runtime only: `output/shared_field_3d/<unique-run>/`。

- [ ] 先运行 U32 两BVP×两姿态×三方法=12配置；显式配置 baseline、shared/raw、shared/input_jump_half，输出真实残差和完整迹。
- [ ] N32可逐列组装reduced矩阵做SVD，仅用于诊断。坐标对齐后矩阵差暂定 ≤1e-8；baseline/shared_jump等良态矩阵的条件数偏差暂定≤1%。已知近奇异的shared_direct报告最小奇异值、矩阵扰动尺度和病态量级，不要求条件数匹配1%。候选条件数 <100 是本测试的粗筛，不是一般稳定性保证。
- [ ] 候选 `shared_jump` 要求全部GMRES达到设置的 `2e-10` 目标，并重算真实残差。沿用当前C++ `3e-10` 的最终真实残差认证界限时必须分别记录迭代设定值和认证界限，不能只读递推标志。
- [ ] `shared_direct` 的 N32 D 上限失败与 N32 N 错误内解作为已知研究结果保留；流程正常退出不等于数值验收通过，禁止自动切换候选结果替代失败行。
- [ ] NaN/Inf、真实残差超界、缺失配置或重放不一致均判对应验收失败。病态direct在不同实现中不必恰好复现同一步数/退出码；可记录“GMRES收敛、物理解验收失败”，并调查谱与误差。
- [ ] 再顺序运行N64/128，完成候选12配置及完整36配置。每次保存density、field、reduced系数及配置，可用一次forward重放结果。
- [ ] 包内内误差匹配至5%、GMRES步数差≤2作为初期偏差调查线，不作为普遍精度定理。分别逐姿态报告两段阶数，允许忠实复现包内粗段低于2，禁止混用姿态拼阶。
- [ ] 记录setup/fit/transfer/bulk/GMRES时间，区分包含关系；记录峰值内存及因子nnz。首期同时仅保留一套大规模fit因子，不并发N128。若对照4GiB预算，则3.2GiB预警、达到4GiB停止并保留资源失败记录。
- [ ] 完成现有 Trace93/geometry/topology/restrict 相关回归，报告实际运行范围；独立提交脚本和摘要，运行大文件不提交。

## T8. 集成文档与后续扩展入口

**Files:**
- Modify: `README.md`、`docs/architecture/README.md`、`tests/cases/shared_field_3d/README.md`。
- Create: `docs/Shared_Field_KFBI_Implementation.md`、`docs/Shared_Field_KFBI_Validation.md`。

- [ ] 将真正已实现的类型/路径和选项回填到实现说明；区分设计目标、已完成代码、实测验证，不直接复制本计划为完成报告。
- [ ] 记录旧默认未变、共享场是实验后端、raw/equation的字段定义、来源包hash和矩阵对照结果。
- [ ] 文档中明确 C2而非C3、软拟合、面内1/2、N128资源和粗层观测阶限制。
- [ ] 为未来 L/曲面拓展列出三个现有接口接点：几何/窄带求积、原密度采样、lift/project。曲面接口需要研究样条结点面与曲面求积相交问题，不能复用矩形距离函数冒充实现。
- [ ] 为未来topology适配明确使用其mass坐标lift/project配对，不修改其约束；只有独立满足生产推广门槛后才讨论默认切换。
- [ ] 对照设计第2–14节逐项检查有代码/测试/报告覆盖，确认不存在第二个独立KFBI主循环；独立提交最终文档。

## 验证命令与执行说明

以下 target 已实现；使用与现有 build-3d 一致的编译器和依赖配置新目录。半跳跃测试已合并到公共 closure/solver 和 shared 集成测试。实际本机命令还可参阅实现说明。

```powershell
cmake -S . -B build-shared-field -DKFBIM_BUILD_3D=ON -DKFBIM_BUILD_EXPERIMENTAL_3D=ON -DBUILD_TESTING=ON
$kfbiTargets = @(
  'kfbi_trace93_study_3d', 'shared_field_basis_3d_test', 'shared_field_fit_3d_test',
  'shared_field_transfer_3d_test', 'jump_identity_closure_3d_test',
  'shared_field_python_fixture_3d_test', 'shared_field_kfbi_3d_test',
  'affine_trace_coordinates_3d_test', 'affine_kfbi_solve_3d_test',
  'trace93_case_3d_test', 'trace93_density_layout_3d_test', 'trace93_resource_3d_test',
  'resource_restrict_assembly_3d_test'
)
cmake --build build-shared-field --config Release --target $kfbiTargets --parallel 1 -- /p:PreferredToolArchitecture=x64 /nodeReuse:false
ctest --test-dir build-shared-field -C Release --output-on-failure -R 'shared_field|affine_trace_coordinates|affine_kfbi_solve|trace93|resource_restrict_assembly'
```

实际 runner 接口为 `./tests/cases/shared_field_3d/run_shared_field.ps1 -Levels 32 -Bvps dirichlet,neumann -Transforms rotate,rotate_translate -Variants baseline,shared_direct,shared_jump -BuildDirectory build-shared-field`。N64/N128在N32分层验收通过后使用同一接口，默认顺序运行；输出目录必须新建且唯一。上述 MSBuild 附加参数只适用于 Visual Studio。

## 计划自检

- 算法完整性：密度/场两空间、拟合/lift/正则、Spread、staged/direct Restrict、半跳跃、投影/affine/GMRES、原始诊断均有对应任务。
- 架构完整性：所有新增数值能力通过现有入口和统一后端契约进入，无独立大程序；实验依赖没有反向进入核心公共库。
- 验证完整性：包内绕过拟合的P2测试缺口由T3补足；逐层operator测试先于全PDE。
- 来源完整性：临时副本不进入构建；固定fixture与来源记录独立；历史报告不计作本地运行。
- 实施证据：本地 `.cpp/.hpp/CMakeLists.txt` 已集成，原细项保留为设计记录；实际测试映射、构建、PDE 和资源结果见验证报告。由于 `.git` 写入受限，使用原工作区回退，未执行各任务中的独立提交步骤。
