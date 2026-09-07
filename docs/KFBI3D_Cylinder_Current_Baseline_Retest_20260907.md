# 当前主力方案：空心圆柱旧异常算例复测

## 结论

三组实际复测均在 all-event restrict 的根集/精确端点检查处自然异常退出，退出码均为 1。未进入 GMRES，没有产生新的内部误差、密度误差、最终 DOF、完整物理残差或收敛阶。不能把本次失败写成 GMRES 发散，也不能宣称旧低阶问题已经修复。

## 配置与可追溯性

- 源码提交：`275368eada4b23d35e8dd256ee86e21e63c49ac7`。本轮未修改求解器源码。
- 可执行文件：`build-3d/apps/kfbi_topology_affine_exterior_trace_3d.exe`；CMake Release `-O3 -DNDEBUG`，构建检查返回 `ninja: no work to do`。
- SHA256：`075695E42424196D1970D52D6C8B6A5B8E0265C859088C2CDB9EC6A01DAD707E`。
- 保持原 C++ 空心圆柱：外/内半径 0.55/0.25，轴心 (0.06,-0.05)，z 属于 [-0.63,0.67]，计算盒 [-1.5,1.5]^3，h=3/N。
- 保持原调和制造解及姿态；Neumann 使用 baseline，Dirichlet 使用 Tx=(0.137,0,0)。每方向密度系数分别为 4、4、7。
- Neumann 使用 Q27-cover3、topology-affine、mean-free pivot elimination；Dirichlet 使用 Q64-cover4、analytic J0/affine J1、broken sheets。
- GMRES 相对容差 2e-10，上限 80；未放宽求交认证或启用旧路线回退。OMP/OpenBLAS/MKL 各1线程，内部FFT2线程。

## 实测状态与耗时

| 问题与姿态 | N | 退出前状态 | 启动至失败墙钟/秒 | 进程CPU/秒 | GMRES/误差/阶数 |
|---|---:|---|---:|---:|---|
| Neumann baseline | 32 | IncompleteRootSet；1个歧义根簇 | 1221.874970 | 703.796875 | 未进入/未产生/不可计算 |
| Dirichlet Tx | 32 | IncompleteRootSet；1个歧义根簇 | 25.730051 | 19.375000 | 未进入/未产生/不可计算 |
| Dirichlet Tx | 64 | MissingExactEndpoint；无歧义根簇 | 51.097620 | 38.250000 | 未进入/未产生/不可计算 |

Neumann 墙钟包含一次 GDB 调用栈采样导致的暂挂，不是干净性能基准。所有时间均为整个进程启动至异常退出，不能与旧 CSV 的完整 solve_* 调用耗时直接比较。原始记录由独立包装器保存，三例均自然返回应用错误；拟议的20分钟保护停止没有终止任何进程。

## 可复现失败位置

| 算例 | center | side | stencil_slot | grid_node | partition_kind | unresolved | ambiguous_clusters |
|---|---:|---|---:|---:|---:|---:|---:|
| Neumann N32 | 507 | interior | 7 | 13519 | 2 | 0 | 1 |
| Dirichlet N32 | 17 | interior | 6 | 12531 | 2 | 0 | 1 |
| Dirichlet N64 | 29 | interior | 50 | 95013 | 3 | 0 | 0 |

三个报错记录的 endpoint_extension 均为 `1.2974806356936508e-10`。

`apps/restrict_crossing_selector_3d.hpp:100` 定义 2=IncompleteRootSet、3=MissingExactEndpoint。对应 cpp 第455行在歧义根簇非零时直接拒绝根集，所以前两例不能具体归结为端点丢失；第三例未识别到同时匹配原生owner/参数和空间位置的精确端点，尚不能区分漏根与匹配问题。

错误文字中的 shared quadratic 是共用 `build_all_event_trace_corrections` 函数保留的历史名称（主driver第4230行）；Q27/Q64均调用此函数，不代表回退到Q10。

约16分钟时的实际调用栈为 all-event support correction → NURBS segment intersection → closest-point assistance → NURBS derivatives，证明采样时仍在求交预处理，不是GMRES。该专用求交器的细分深度为48；单次局部迭代有上限，但所检查路径没有实用的总细分盒数/墙钟预算。不能仅凭一次采样断言是单条病态路径还是大量查询累计耗时。

## 与旧记录的关系

旧 Neumann N32 shared-quadratic/c0_one_sided 实验曾返回内部误差 0.1606652293、40次GMRES且未收敛。旧 Q10 Dirichlet Tx 的 N32/N64 内部误差为 0.00265451317668、0.000890429431358，阶数1.575874、GMRES19/20。这些旧数字不能填入本轮主力方案结果栏。

当前需要先定位并最小重放上述支撑路径，区分近根歧义、真实端点漏检与owner/参数匹配问题，再恢复完整数值验证。此次没有改动算法，也没有运行N128。

## 原始文件

以下路径为仓库内仅保留在本地的运行产物，未随源码提交。后续修复及验证见[实现说明](KFBI3D_Native_Endpoint_Implementation_20260907.md)；三条失败路径的最小复现输入已固定在 `apps/native_endpoint_path_3d_test.cpp`，不依赖这些本地产物。

- 运行脚本：`output/cylinder_retest_20260907/run_cases.ps1`
- 全部测量记录：`output/cylinder_retest_20260907/measurements.json`
- Neumann N32 错误：`output/cylinder_retest_20260907/n32_baseline/stderr.log`
- Dirichlet N32 错误：`output/cylinder_retest_20260907/d32_tx/stderr.log`
- Dirichlet N64 错误：`output/cylinder_retest_20260907/d64_tx/stderr.log`
- 调用栈采样说明：`output/cylinder_retest_20260907/diagnostic_notes.md`
