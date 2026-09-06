# Neumann 特征棱 C1 环境梯度方案：现状、差异与实现报告

日期：2026-09-03

## 1. 当前几何与密度表示

边界几何由多个 NURBS 曲面片组成。每个曲面片通过参数映射

\[
X_i(u,v):\widehat\Omega_i\rightarrow\mathbb R^3
\]

提供位置、两个参数切向量和单位法向量。Neumann 问题的未知值跳跃
\(J_0=\mu\) 在每个曲面片上用三次张量积 B 样条表示：

\[
\mu_i(u,v)=\sum_A c_{iA}B_{iA}(u,v).
\]

完整公共边上的 C0 系数由 Union-Find 精确合并；部分公共边的 C0
条件使用公共 knot-overlay 单元上的 P0--P3 矩约束。G1 光滑连接已经
使用公共物理余法向导数条件。非 G1 的物理特征棱上，ValueTrace 保持
C0，而表示已知 \(J_1=g_N\) 的 NormalTrace 在两侧保持 broken，避免
错误地令不同面法向下的两个标量值相等。

## 2. 原有特征棱 C1 实现

令公共定向单位切向为 \(t\)，并定义

\[
m_a=n_a\times t,\qquad m_b=n_b\times t,
\qquad c=n_a\cdot n_b,\qquad s=n_b\cdot m_a.
\]

原实现装配

\[
\partial_{m_a}\mu_a=\frac{g_{N,b}-c g_{N,a}}{s},
\qquad
\partial_{m_b}\mu_b=\frac{c g_{N,b}-g_{N,a}}{s}.
\]

这两式并不是独立的经验条件。它们正是先假设两侧存在同一个空间
梯度，再把该梯度的两个横向分量消元后得到的结果。因此原实现与新
思路在连续层面具有相同的物理含义。

原实现的主要数值特征是：

- 每个 knot-overlay 单元、每个 P0--P3 模式装配两行；
- 左端每行只含一侧 \(\mu\) 的余法向导数，右端含两侧 \(g_N\)；
- 显式使用 \(1/s\)，当两个面接近光滑时会放大舍入误差和边界数据
  表示误差；
- 当 \(|s|\) 小于阈值时直接终止 setup；
- Vertex/T-star 只是端点单元矩行的消元分组，并不是额外的顶点点值
  或点梯度方程。

## 3. 新增的直接环境梯度实现

新后端在曲面片 \(i\) 上先由 B 样条系数计算曲面梯度：

\[
\nabla_{\Gamma_i}\mu_i
=J_i(J_i^TJ_i)^{-1}
\begin{bmatrix}\partial_u\mu_i\\\partial_v\mu_i\end{bmatrix},
\qquad J_i=[X_{i,u},X_{i,v}].
\]

再用 Neumann 条件补回法向分量：

\[
G_i=\nabla_{\Gamma_i}\mu_i+g_{N,i}n_i.
\]

最终直接约束

\[
G_a=G_b.
\]

实现中没有盲目加入三个 Cartesian 方程。已有 C0 边迹意味着沿边
切向导数相同，所以 \(t\cdot(G_a-G_b)=0\) 已经包含在 C0 空间内。
代码从两侧法向的角平分方向构造 \(t^\perp\) 内的刚体协变正交基
\(r_0,r_1\)，只装配两个独立条件：

\[
r_k\cdot(G_a-G_b)=0,\qquad k=0,1.
\]

对应的系数方程为

\[
\left[r_k\cdot\nabla_{\Gamma_a}B_a
-r_k\cdot\nabla_{\Gamma_b}B_b\right]c
=\left[r_k\cdot n_b B_b-r_k\cdot n_a B_a\right]g_{N,h}.
\]

因此它仍然是标准仿射约束

\[
C_{\nabla}c=d_{\nabla},\qquad c=c_p+Gz,
\]

GMRES 只迭代齐次坐标 \(z\)。该装配没有二面角正弦除法，而且不会
因为重复加入沿边分量而额外降低自由度。

## 4. 两种实现的具体差别

| 项目 | 原 solved-conormal 实现 | 新 ambient-gradient 实现 |
|---|---|---|
| 连续物理条件 | 单一空间梯度 | 单一空间梯度 |
| 离散未知量 | B 样条值跳跃系数 | 相同 |
| 每个矩模式的行数 | 2 | 2 |
| 比较方向 | 两侧各自余法向 | 公共横截平面的两个正交方向 |
| 左端单行支持 | 通常只涉及一侧，最多 16 个系数 | 同时涉及两侧，最多 32 个系数 |
| 二面角处理 | 显式除以 \(s\) | 无 \(1/s\) |
| 接近光滑的行为 | 条件变差或按阈值拒绝 | 系数保持有界，并自然显示秩退化 |
| U/L 柱直棱 | 与新实现的 C0 商空间行空间相同 | 与原实现相同 |
| 曲特征棱 | P0--P3 余法向矩 | P0--P3 移动横截基矩；有限维行空间不保证完全相同 |

## 5. 生产数据链中仍然存在的差别

新后端已经直接实现了几何上的梯度相容，但当前 Neumann 生产链仍然
保留原来的两层数据处理：

1. 原始 `prescribed_normal_jump` 样本先投影到 feature-broken 的
   NormalTrace B 样条空间，得到 `normal_c0`；
2. 由 `normal_c0` 形成的 feature 右端随后投影到已有 C0/G1 ValueTrace
   空间能够达到的范围。

因此当前新后端严格满足的是投影后的离散仿射右端，而不一定是原始
解析 \(g_N\) 在每个 feature 积分点上的值。物理 spread、Cauchy jet 和
restrict 仍继续使用原始 Neumann 数据，只有 feature 约束右端被修改。
现有诊断 `edge_normal_fit_linf` 和 `feature_target_projection_*` 分别记录
这两层差异。

如果下一步要求“解析 \(g_N\) 完全不拟合、不投影”，还需要增加严格
analytic-\(g_N\) 后端：在 feature Gauss 点直接调用解析边界条件，并在
右端不可达时停止 setup，而不是修改右端。本次实现没有把这一行为
混入 A/B 几何后端，以免改变现有 Neumann 基准算法。

## 6. 代码入口与选择方式

核心装配位于：

- `apps/topology_density_constraints_3d.hpp`
- `apps/topology_density_constraints_3d.cpp`

生产接线位于：

- `apps/neumann_exterior_zero_trace_3d.cpp`

基准实现保持默认不变：

```text
KFBIM_3D_NEUMANN_EDGE_JUMP_JET=topology_affine_local_svd
```

新实现通过以下值显式启用：

```text
KFBIM_3D_NEUMANN_EDGE_JUMP_JET=topology_ambient_gradient_local_svd
```

两个后端共用相同的 C0/G1 topology、Vertex→Edge staged SVD、全局均值
消元、trace projector 和 GMRES 流程。因此 A/B 结果的主要变量只有
feature-C1 的几何离散形式。

## 7. 当前验证结果

已完成以下 focused 验证：

- 原 cylinder/L-prism solved-conormal 回归继续通过；
- U-prism 的 32 条物理 feature connection 全部生成 ambient-gradient
  P0--P3 矩约束；
- 常数值密度仍被导数矩阵消去；
- 一个公共仿射空间梯度 \(q\) 产生的
  \(\mu=q\cdot x+c_0\)、\(g_N=q\cdot n\) 同时满足所有新约束；
- U-prism 上，新旧左端矩阵在已有 C0 topology 商空间中具有相同秩，
  两者合并后秩不增加，说明新实现没有额外消耗自由度；
- U-prism 经过绕任意轴旋转和平移后，feature 数、矩阵尺寸及归一化后的
  ValueTrace/NormalTrace 算子保持不变，并能再现同步旋转的仿射梯度；
- 新行的局部支持上界为 ValueTrace 32、NormalTrace 8，联合归一化正确；
- `topology_density_constraints_3d_test` focused 可执行文件通过；
- 生产 driver 翻译单元通过 GCC 语法检查。

仓库已有的一个 Visual Studio 构建目录混入了 MinGW 系统头，完整 CMake
重建会在未进入本次修改代码前失败；这是构建环境问题。为避免把该
问题误判为算法回归，本次使用仓库既有 GCC focused 对象链重新编译并
运行了相关测试。

## 8. 结论

新增后端正确实现了“曲面梯度 + Neumann 法向分量 = 空间梯度，再令
两侧空间梯度相同”的方案。它与当前公式具有相同的连续约束强度，在
U 柱直棱离散空间中也没有额外降低自由度；主要改进是去掉了
\(1/s\) 和侧相关的解式。当前仍需独立评估的是解析 \(g_N\) 严格输入
模式以及曲特征棱上的有限矩离散收敛性。
