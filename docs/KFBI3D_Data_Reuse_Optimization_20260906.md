# KFBI 3D 数据复用优化说明（2026-09-06）

本批修改将已有计算结果传给后续使用者，减少重复求值和重复分解。离散空间、约束、边界数据和求解流程保持原定义。三个单元测试及 U 柱 N=32 的前后数值对照已通过；局部 CPU 计时显示收益，但当前单次端到端墙钟对照不能确定稳定的整体加速倍数。

## 1. 范围

本批仅包含以下四项等价复用：

| 修改 | 原调用方式 | 新调用方式 |
|---|---|---|
| 密度 P2 参数 jet 批量装配 | 六次单行调用，重复取得一维基函数及其导数 | 取得两方向的基函数及零至二阶导数后，装配六行 |
| direct Cauchy plan 复用几何 jet | 调用方已取得几何 jet，plan builder 再求一次 | 调用方把同一个几何 jet 传入新 overload |
| Cauchy recovery 复用 SVD | 条件数诊断一次 SVD，伪逆再分解一次 | 同一次 thin U/V 分解用于诊断及伪逆 |
| sample stencil 复用装配 design | 保存样点 stencil 后，装配稀疏 design 时重新求值 | 从已经保存的 stencil 按原顺序生成 COO 条目 |

本批不包含 P3 Cauchy 改造、新的 C0/C1/feature 约束、Polar-Star 模式选择、密度分辨率调整或结点细化。已有 all-event catalog、求交事件及其他计划缓存的复用不算作本批新增优化；本批的计时收益不能归因于那些既有机制。

## 2. 接口与数值不变性

### 2.1 批量密度参数 jet

接口位于 [native_nurbs_density_space_3d.hpp](../src/support/density/native_nurbs_density_space_3d.hpp)，实现位于 [native_nurbs_density_space_3d.cpp](../src/support/density/native_nurbs_density_space_3d.cpp)：

```cpp
std::array<NativeDensityC0Stencil3D, 6>
c0_parameter_jet_stencils(int patch, double u, double v) const;
```

返回顺序固定为 `value, u, v, uu, uv, vv`。令密度的原始张量积基为

\[
B_{ij}(u,v)=N_i(u)N_j(v),
\]

六行仍分别使用 \(N_i^{(a)}(u)N_j^{(b)}(v)\)，其中
\((a,b)=(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)\)。改变的是公共中间结果的计算次数，而不是导数公式。

旧的 `c0_parameter_derivative_stencil(...)` 保留。新实现沿用每行原有的 `i/j` 遍历顺序、零乘积跳过规则和重复 C0 索引累加顺序；不会另加容差裁剪。每次调用使用自己的局部数据，不引入可变的浮点键缓存或跨线程共享状态。

### 2.2 复用已经计算的几何 jet

[direct_coefficient_cauchy_3d.hpp](../src/support/cauchy/direct_coefficient_cauchy_3d.hpp) 为 value/normal plan builder 增加带有 `NativeSurfaceParameterJet3D` 参数的 overload。旧接口保留，并在自行计算几何后转调新接口。

[主 driver](../apps/laplace/3d/neumann_exterior_zero_trace_3d.cpp) 在 direct value/normal plan 的绑定过程中，传入已经计算的 `cached.geometry`。该数据来自同一 density surface、同一 patch 及相同的归一化参数 `(u,v)`。法向定向、局部 frame、参数到切平面的变换、graph Hessian 和 Cauchy 行组合公式不变。

新 overload 的契约是调用方提供正确位置及参数坐标下的几何 jet；不能用邻近点、其他 patch 或其他刚体姿态的数据替代。密度 P2 jet 的批量接口用于 value plan，normal plan 的所需导数阶次不因此改变。

### 2.3 同一次 SVD 用于诊断与伪逆

[harmonic_polynomial_space_3d.hpp](../src/support/cauchy/harmonic_polynomial_space_3d.hpp) 新增：

```cpp
Eigen::MatrixXd svd_pseudoinverse_from_decomposition_3d(
    const Eigen::JacobiSVD<Eigen::MatrixXd>& decomposition,
    double relative_cutoff);
```

若加权 design 为 \(A=U\Sigma V^T\)，计算仍为

\[
A^+=V\,\operatorname{diag}(\sigma_i^+)\,U^T,
\qquad
\sigma_i^+=
\begin{cases}
1/\sigma_i,&\sigma_i>\tau\sigma_0,\\
0,&\text{否则}.
\end{cases}
\]

driver 的两处 recovery 使用相同的 \(\tau=3\times10^{-12}\) 和严格大于判据，保留原伪逆乘法顺序与后续列缩放顺序。thin 分解沿用原矩阵表达式；full 分解用于矩形矩阵时，仅取与奇异值对应的列。只包含奇异值而没有 U/V 的分解被拒绝。

`direct_crossing_svd_count_` 对每个受该统计覆盖的 direct crossing plan 由两次改记一次；该字段不是全程序所有 SVD 的总计数。其他分解算法、秩阈值和 GMRES 设置不作调整。

### 2.4 已保存样点 stencil 直接装配 design

[主 driver](../apps/laplace/3d/neumann_exterior_zero_trace_3d.cpp) 的 `NativeDensityTransfer3D` 先初始化 `sample_stencils_`，再用它构造 `c0_design_`。每行仍按 stencil 原有条目顺序生成 `(row, column, weight)`，随后调用相同的 `setFromTriplets` 和 `makeCompressed`。

此项不更改采样点、权重、C0 列编号或投影矩阵的数学内容，也不改变 projector 的满秩和条件数检查。

## 3. 数据依赖与失效规则

以下规则用于限定本批复用，也为后续可能引入的密度结点自适应提供边界；结点自适应本身不在本批实现范围内。

| 数据 | 依赖 | 需要重新构造的情形 |
|---|---|---|
| 几何 jet | 几何模型、刚体姿态、patch、归一化 `(u,v)` | 以上任一项改变；新增采样位置也需重新求值 |
| 一维密度基值及导数 | 次数、结点向量、参数位置、导数阶次 | 密度基或参数位置改变 |
| C0 stencil / sample design | 密度基、raw 编号、C0 映射、采样点 | 密度插结点、编号/接口映射改变或重新采样 |
| direct coefficient plan | 密度 stencil、几何 jet、frame、Cauchy 组合规则 | 任一相关依赖改变 |
| recovery SVD | 加权 design 的全部条目、维度、分解选项 | design 或权重改变；不能复用另一个矩阵的分解 |
| 仿射空间及最终 projector | 约束矩阵/目标、齐次映射、采样 design/权重 | 结点、约束、采样或相关已知数据改变 |

若仅加密密度且几何和网格不变，原位置的几何数据可继续使用；密度相关行、约束、仿射映射及最终 projector 必须按新空间重建。任何扩大缓存生命周期的后续修改都应携带这些依赖，不能仅用旧的整数编号命中缓存。

## 4. 单元测试覆盖

本轮在 `build-3d` 的 Release 构建下执行了以下三个测试程序，均正常退出（exit code 0）：

- [native_nurbs_density_space_3d_test.cpp](../tests/density/native_nurbs_density_space_3d_test.cpp)：圆柱、L 柱、U 柱；`ncoef=4/6`；ValueTrace 与 NormalTrace 两种 C0 映射；各 patch 的端点、内部点、内部结点及左右相邻浮点值。六行逐一与旧单行 API 比较 count、完整索引数组和系数数组，并检查非法 patch。
- [direct_coefficient_cauchy_3d_test.cpp](../tests/cauchy/direct_coefficient_cauchy_3d_test.cpp)：平面与曲面、原姿态与刚体变换姿态、两种密度场，在相同 frame 输入下比较旧接口与几何复用 overload 的 stencil、graph Hessian、诊断量及预组合 Cauchy 行。
- [harmonic_polynomial_space_3d_test.cpp](../tests/cauchy/harmonic_polynomial_space_3d_test.cpp)：矩阵接口与复用分解接口的伪逆一致性、秩截断、values-only 分解拒绝，以及矩形矩阵的 thin/full U/V 支持。

这些测试用于检查所改局部代数的等价性。完整算例仍需比较已有基线与新实现的误差、残差、自由度和 GMRES 迭代记录；不能以局部测试替代完整数值结果。

## 5. 复现命令与计时口径

以下 PowerShell 命令使用本轮实际采用的、已经配置好的 `build-3d`（Release，`-O3 -DNDEBUG`）。需要可用的 Eigen、CGAL 和相同的编译器依赖配置；如使用其他构建目录，应保持同一配置。本轮没有使用旧的 `build-topology-final` 可执行文件作为性能基线。

```powershell
$reuseBuildDir = 'build-3d'
$env:Path = 'C:\tools\msys64\mingw64\bin;C:\Strawberry\c\bin;' + $env:Path
& 'C:\Program Files\CMake\bin\cmake.exe' --build $reuseBuildDir --target native_nurbs_density_space_3d_test direct_coefficient_cauchy_3d_test harmonic_polynomial_space_3d_test kfbi_topology_affine_exterior_trace_3d data_reuse_3d_benchmark --parallel 2
& "$reuseBuildDir/apps/native_nurbs_density_space_3d_test.exe"
& "$reuseBuildDir/apps/direct_coefficient_cauchy_3d_test.exe"
& "$reuseBuildDir/apps/harmonic_polynomial_space_3d_test.exe"
& "$reuseBuildDir/apps/data_reuse_3d_benchmark.exe" --iterations 10000
```

[data_reuse_3d_benchmark.cpp](../benchmarks/3d/data_reuse_3d_benchmark.cpp) 不求解 PDE；对局部工作先做等价性检查，再输出 CSV。`cpu_seconds` 为本进程 CPU 时间，`wall_seconds` 为经过时间；样本几何及矩阵的公共准备在计时之前完成。两边都执行相同的 checksum 工作以保留可观察输出。

`density_jet6` 比较六次单行求值与一次批量装配。`value_plan` 和 `normal_plan` 仅比较重复计算几何与传入已有几何：其中旧 value overload 已使用本批的批量密度 jet，因此 `value_plan` 的比率只表示几何复用的局部收益，不是整个 value builder 从旧版本到新版本的总收益。`svd_48x16` 在固定矩形矩阵上比较两次分解与一次分解。

局部微基准比率不能相乘后当作整体提速，也不能代替完整算例端到端计时。微基准当前按固定顺序测量各变体；正式汇总宜在相同编译器、Release 选项和线程设置下进行多轮交替测量，报告中位数及波动。计时时避免同时编译或执行其他高负载任务。

完整 `full-after` 算例的单次 wall 时间会受到操作系统调度、并发负载和运行环境影响，不能仅凭该单次记录断言性能回退或固定加速倍数。before/after 应同时记录配置、计时边界和相应数值结果；只比较相同范围的耗时。局部复用和端到端耗时分别报告。

## 6. U 柱 N=32：完整数值与计时对照

### 6.1 配置与正确性

前后均运行 `kfbi_topology_affine_exterior_trace_3d.exe u_prism 32`，选择 Both；Neumann 为 `q27_cover3_all_event_cauchy`，Dirichlet 为 `q64_cover4_all_event_cauchy`，采用已有 topology affine local SVD 路线，Dirichlet 保持 `broken_sheets`。密度每方向系数数目为 4，网格间距为 0.09375。`OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 均为 1，内部 FFT 线程数为 2。性能计时期间不并行编译或运行其他本轮数值任务。

共有 10 份 CSV，比较了 95,908 个数值单元和 936 个文本单元；排除耗时和 SVD 次数等性能字段后，全部共同字段精确一致，最大数值差为 0，NaN 等非有限状态也一致。没有新增或缺失的文件、行、列。这里的“精确一致”是保存到 CSV 的数值一致，不扩展为未导出内存数据的逐位证明。

| 指标 | Neumann 优化前 | Neumann 优化后 | Dirichlet 优化前 | Dirichlet 优化后 |
|---|---:|---:|---:|---:|
| 最终 DOF | 15 | 15 | 224 | 224 |
| 外迹采样点 | 832 | 832 | 832 | 832 |
| 外迹点数 / 最终 DOF | 55.4667 | 55.4667 | 3.71429 | 3.71429 |
| 拓扑约束行数 / 秩 | 344 / 176 | 344 / 176 | 32 / 32 | 32 / 32 |
| GMRES 迭代数 | 15 | 15 | 23 | 23 |
| GMRES 相对残差 | 1.013010445389308e-15 | 1.013010445389308e-15 | 7.836291807395449e-11 | 7.836291807395449e-11 |
| 密度最大误差 | 6.514731612558555e-5 | 6.514731612558555e-5 | 1.538127041619125e-4 | 1.538127041619125e-4 |
| 密度 L2 误差 | 2.801643638510894e-5 | 2.801643638510894e-5 | 5.197763161882507e-5 | 5.197763161882507e-5 |
| 内域最大误差 | 5.986122784895631e-5 | 5.986122784895631e-5 | 1.725614460457159e-5 | 1.725614460457159e-5 |
| 内域 L2 误差 | 2.061444743806595e-5 | 2.061444743806595e-5 | 6.328804864356531e-6 | 6.328804864356531e-6 |
| 完整外迹条件 Linf | 3.076809569266270e-5 | 3.076809569266270e-5 | 8.536259359742678e-6 | 8.536259359742678e-6 |
| converged | 1 | 1 | 1 | 1 |
| physical_converged | 0 | 0 | 0 | 0 |

两类问题仍满足外迹点数大于最终 DOF。Neumann 的完整外迹条件指外侧值迹，Dirichlet 指外侧法向迹。本轮没有修改物理收敛判据：两者 `physical_converged=0` 的既有状态保留；Dirichlet 的相对投影泄漏仍为 0.99999999998583722。因此这里验证的是等价性和复用收益，不是修复原有物理残差问题。

本轮没有运行 N=64/128，也没有生成新的收敛阶；单一网格不能估计收敛阶。

### 6.2 完整运行耗时

| 计时范围 | 优化前（秒） | 优化后（秒） |
|---|---:|---:|
| 完整进程墙钟时间 | 302.9726315 | 76.3891442 |
| Neumann 求解阶段墙钟时间 | 11.9279075 | 6.4286490 |
| Dirichlet 求解阶段墙钟时间 | 12.3471489 | 6.5699609 |
| 其余阶段墙钟差额 | 278.6975751 | 63.3905343 |
| 完整进程 CPU 时间 | 未获得 | 51.1093750 |

完整墙钟使用外部 Stopwatch，覆盖进程启动至退出，不含工具返回等待。CSV 的 `seconds` 则是完整 `solve_*` 调用范围，包含仿射约束/投影器设置、已知 jump 对应的网格求解、GMRES、最终场重建和求解器内部的迹/残差诊断，并非纯 GMRES 时间。Neumann 计时起止位于主 driver 的 9099–9109 行，Dirichlet 位于 9394–9400 行。其余阶段是总墙钟减去两个 solve 时间，包含公共几何/网格/求交/算子准备、调用后的误差统计、I/O 和进程开销，不能称为单独的 setup 测量。

本次墙钟观测值约为 3.97 倍差异，但前后主机调度状态明显不同，且 before 没有最终 CPU 记录，不能把这个差异归因于代码优化，也不能宣称稳定的 3.97 倍整体加速。当前证据支持局部计算量和 CPU 耗时减少；整体稳定加速率仍需同等负载下多轮交替测量。

AFTER 正常退出，exit code 为 0。BEFORE 的测量包装器未保留进程句柄，未能取得退出码及最终 CPU 时间，原记录保持未知；其标准输出正常结束，两种求解结果及完整 CSV 均已生成。本轮修正了包装器的句柄保留方式，没有补造 before 的退出码。

`crossing_cauchy_svd_count` 从 1664 降到 832，符合每个受统计覆盖的 plan 从两次分解改为一次；这不是全程序全部 SVD 的总数。

### 6.3 数据与可执行文件记录

- [BEFORE 测量记录](../output/perf_reuse_before_20260906/u32_run1/measurement.json)
- [AFTER 测量记录](../output/perf_reuse_after_20260906/u32_run1/measurement.json)
- [逐字段比较及结果汇总](../output/perf_reuse_after_20260906/comparison.json)
- [AFTER Neumann 原始结果](../output/kfbi_topology_affine_3d/q27_c3/d_q64_c4/nm_pc_tm_mf/topology_affine/perf_reuse_after_20260906/u32_run1/neumann_results.csv)
- [AFTER Dirichlet 原始结果](../output/kfbi_topology_affine_3d/q27_c3/d_q64_c4/nm_pc_tm_mf/topology_affine/perf_reuse_after_20260906/u32_run1/dirichlet_normal_results.csv)

可执行文件 SHA256：

```text
before: 87DF847FAFDF80EDFB7925A980143150F4C4F30D8DFD1BED8B6C5609AB583E29
after:  075695E42424196D1970D52D6C8B6A5B8E0265C859088C2CDB9EC6A01DAD707E
```

## 7. 局部复用微基准

主程序运行结束后，单独执行 `data_reuse_3d_benchmark --iterations 10000`，正常退出（exit code 0）。密度和 plan 使用空心圆柱的 32 个 patch/参数样本，密度每方向 7 个系数、BaseOnly reduction；SVD 使用固定 48×16 满秩矩阵。这与 U32 完整算例不是同一工作负载，不能用局部比率替代整体比率。

每个变体计时 10,000 次调用。四组等价性最大差均为 0，每对 checksum 完全一致。下表的倍数仅为这一次测量的旧/新 CPU 时间比。

| 局部工作 | 原 CPU 秒 | 复用后 CPU 秒 | CPU 时间比 | 原墙钟秒 | 复用后墙钟秒 |
|---|---:|---:|---:|---:|---:|
| 六行密度参数 jet：单行调用 → 批量 | 0.234375 | 0.125000 | 1.88 | 0.2926083 | 0.1786543 |
| value plan：重复几何 → 复用几何 | 0.437500 | 0.218750 | 2.00 | 0.5242637 | 0.2150857 |
| normal plan：重复几何 → 复用几何 | 0.265625 | 0.125000 | 2.13 | 0.2723498 | 0.1268561 |
| recovery：两次 SVD → 一次分解复用 | 5.640625 | 4.296875 | 1.31 | 6.3924023 | 5.7527617 |

本次四项 CPU 时间分别降低约 46.7%、50.0%、52.9%、23.8%。微基准变体固定顺序各测一次，没有统计置信区间；较短测试还受 Windows CPU 计时粒度影响，倍数仅作初测。sample stencil 直接装配 design 的收益未单独计时。未把各局部比率相乘，也未把旧 overload 误作完整旧版本的 value plan。

原始数据：[microbenchmark_10000.csv](../output/perf_reuse_after_20260906/microbenchmark_10000.csv)。微基准可执行文件 SHA256：`B99383529C4E1D11411D9A410BE0B5C2344291D14294C1D6AE3C76D333048BDA`。

## 8. 当前完成边界

本批完成四项局部数据复用、兼容接口、单元测试、微基准以及 U32 两类边值问题的完整前后验收。没有修改现有几何表示、试验空间、约束或 Q27/Q64 路线，也没有降低 GMRES 容差。

跨求解共享更长生命周期的几何上下文、完全按需构造 legacy plan、密度插结点自适应以及 P3/约束算法变更仍不在本批实现中。后续若继续扩大缓存范围，应以第 3 节的依赖失效规则为基础，并继续保留数值等价验收。
