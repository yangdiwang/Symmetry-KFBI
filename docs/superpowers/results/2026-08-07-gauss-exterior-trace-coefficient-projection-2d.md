# NSC-ET-GCP-BG：Gauss 外迹系数投影与 bordered Neumann 约束

## 正式方案

本方案命名为 **NSC-ET-GCP-BG**（NURBS-Spline-Coefficient
Exterior-Trace with Gauss Coefficient Projection and Bordered Gauge）。密度
未知量仍是连续性对齐的 B 样条系数；边界点只承担外迹评价，不再从
`Interface2D` 中选出与未知量相同数量的方形配点。

对未知密度空间的每个非零 span，先在普通 NURBS knot 处进一步分裂，
使每个积分子区间上的几何度量光滑。默认在每个子区间使用四点
Gauss--Legendre：

\[
 \xi_q=\frac{\xi_l+\xi_r}{2}
       +\frac{\xi_r-\xi_l}{2}\widehat\xi_q,
 \qquad
 w_q=\frac{\xi_r-\xi_l}{2}\widehat w_q
       \lVert\gamma_\xi(\xi_q)\rVert.
\]

每个参考点保存 branch、density span、参数、精确 NURBS 点、法向和物理
权重。Gauss 点均严格位于光滑子区间内部，因此角点只出现单侧迹，不会
采到几何导数不唯一的端点。

## 一次性缓存

设未知密度的约化基评价矩阵为

\[
 B_{qi}=N_i(\xi_q),\qquad W=\operatorname{diag}(w_q).
\]

构造函数中仅做一次加权薄 SVD，并显式保存

\[
 P=(\sqrt W B)^+\sqrt W.
\]

代码不形成法方程。预处理必须认证矩阵满列秩并满足

\[
 PB\simeq I.
\]

`trace_projection_build_count` 在整个求解中恒为 1。每次 GMRES matvec
只完成以下操作：

1. 由当前密度系数构造 crossing/参考点局部 Cauchy P2；
2. 解一次 bulk 问题；
3. 用预缓存的六点空间 P2、两侧三层法向样本和统一法向 P2 得到所有
   Gauss 点外迹 `e`；
4. 做一次固定矩阵乘法 `P*e`，返回密度系数坐标下的残差。

Gauss restrict 的空间节点、插值权重和 crossing 认证都在构造阶段缓存。
最终内迹直接由 jump 关系恢复：

\[
 u_{\rm int}=u_{\rm ext}+\phi,
 \qquad
 \partial_n u_{\rm int}=\partial_nu_{\rm ext}+\psi,
\]

不再重复运行一遍 interior restrict。

## Neumann bordered 系统

Neumann 不再同时用 mean-zero 基约化 trial 和 test，因为非圆几何的左零
模一般不等于常数。保留完整的 P3 密度系数，并解平衡尺度的 bordered
系统：

\[
 \begin{bmatrix}
   P\mathcal E & \widehat z\\
   \widehat m^T & 0
 \end{bmatrix}
 \begin{bmatrix}c\\\lambda\end{bmatrix}
 =
 \begin{bmatrix}-Pe_{\rm fixed}\\0\end{bmatrix}.
\]

其中 `z` 是常数密度系数方向，`m=B^TW1` 是物理质量行；用于 bordered
运算的 `z`、`m` 分别按二范数归一化。另保留
`m/(m^T z)`，只用于报告真实密度均值。这样完整外迹模态没有被删除，
均值约束又能直接达到 Krylov 容差。

## 残差含义

结果分别记录：

- 全部 Gauss 点的原始外迹加权相对误差与 `Linf`；
- 未经乘子抵消的真实投影外迹 `||Pe||`；
- bordered 方程残差 `||[Pe+z lambda, m^Tc]||`；
- 原始外迹的物理加权均值、密度均值和乘子；
- `cond(sqrt(W)B)`、`||PB-I||` 和投影构造次数；
- 只属于 Gauss reference plan 的 crossing/fallback/stencil 诊断。

因此 GMRES 收敛不再被误解释为逐点外迹已经为零。

## 验证入口

```text
build/apps/laplace_nsc_et_bvp_2d_test
build/apps/laplace_nsc_et_benchmark_2d \
  --geometry circle --bvp both --density-method arclength-fd \
  --trace-gauss-order 4 --neumann-gauge bordered 32 64 128
```

smoke test 覆盖矩阵尺寸、正权重、光滑子区间内点、边界长度求积、
`PB=I`、确定性系数重构、缓存不变性、border 行列单位范数、密度均值和
Gauss-reference-only fallback。

## 数值结果

以下结果使用周期弧长密度、采样差分、四点 Gauss reference 和默认
bordered Neumann 系统。误差阶按相邻两层的 `Linf` 误差计算。

### 圆

| BVP | N | GMRES | boundary `Linf` | 阶 | bulk `Linf` | 阶 |
|---|---:|---:|---:|---:|---:|---:|
| Dirichlet | 32 | 7 | 1.594e-3 | -- | 1.483e-3 | -- |
| Dirichlet | 64 | 7 | 3.681e-4 | 2.114 | 3.630e-4 | 2.030 |
| Dirichlet | 128 | 7 | 8.967e-5 | 2.037 | 8.854e-5 | 2.036 |
| Neumann | 32 | 9 | 8.474e-3 | -- | 3.247e-3 | -- |
| Neumann | 64 | 9 | 2.008e-3 | 2.077 | 8.992e-4 | 1.852 |
| Neumann | 128 | 8 | 6.786e-4 | 1.565 | 2.155e-4 | 2.061 |

### 椭圆

| BVP | N | GMRES | boundary `Linf` | 阶 | bulk `Linf` | 阶 |
|---|---:|---:|---:|---:|---:|---:|
| Dirichlet | 32 | 8 | 1.856e-3 | -- | 1.796e-3 | -- |
| Dirichlet | 64 | 7 | 5.591e-4 | 1.731 | 5.341e-4 | 1.750 |
| Dirichlet | 128 | 7 | 1.365e-4 | 2.034 | 1.364e-4 | 1.969 |
| Neumann | 32 | 9 | 1.139e-2 | -- | 3.796e-3 | -- |
| Neumann | 64 | 9 | 1.936e-3 | 2.557 | 1.072e-3 | 1.824 |
| Neumann | 128 | 8 | 5.566e-4 | 1.799 | 2.427e-4 | 2.143 |

所有 12 个算例中，`projection_build_count=1`，投影恒等认证误差为
`O(1e-15)`，Gauss reference 的 unresolved/endpoint fallback、stencil
relocation 和 candidate rejection 均为 0。Neumann 的未补偿 `||Pe||`
不等同于 Krylov 残差；例如椭圆三层分别为
`4.33e-4/1.47e-4/1.61e-5`，而对应完整 bordered 残差为
`5.35e-10/1.85e-10/5.82e-10`。

和旧的“选择与未知量同数目的界面点组成方阵 + N-1 正交 gauge”相比，
新方案在 Dirichlet 光滑问题上保持相近的二阶误差；在最敏感的
`N=128` Neumann 问题上，圆的 GMRES 从 11 降到 8、bulk 误差从
`3.820e-4` 降到 `2.155e-4`，椭圆的 GMRES 从 12 降到 8、bulk 误差从
`1.422e-3` 降到 `2.427e-4`。

原始数据：

- `output/nsc_et_gauss_circle_32_64_128.csv`
- `output/nsc_et_gauss_ellipse_32_64_128.csv`
- `output/nsc_et_three_way_arclength_fd_20260807.csv`（旧方案）

### 复杂几何 N=32 smoke

这些单层结果只检查求解稳定性，不作为收敛阶结论：

| 几何 | BVP | GMRES | boundary `Linf` | bulk `Linf` | unresolved/endpoint | relocate/reject |
|---|---:|---:|---:|---:|---:|---:|
| 花形 | Dirichlet | 12 | 7.146e-3 | 4.733e-3 | 0/0 | 0/0 |
| 花形 | Neumann | 14 | 3.355e-1 | 1.684e-2 | 0/0 | 0/0 |
| 心形 | Dirichlet | 11 | 2.309e-3 | 2.318e-3 | 0/0 | 2/12 |
| 心形 | Neumann | 16 | 1.543e-2 | 6.209e-3 | 0/0 | 2/12 |
| L 形 | Dirichlet | 14 | 1.647e-3 | 1.241e-3 | 0/0 | 0/0 |
| L 形 | Neumann | 21 | 8.653e-3 | 5.087e-3 | 0/0 | 0/0 |

心形的 2 次 relocation 和 12 次 candidate rejection 表示初选的完整空间
P2 stencil 不合格后，预处理改选了同侧 stencil；最终没有进入
unresolved/endpoint crossing fallback。花形 N=32 的 Dirichlet raw trace/
density 误差和 Neumann boundary 误差仍偏大，因此这里只将其记为 smoke
通过。L 形 Neumann 相比旧方案的 GMRES 从 23 降至 21，bulk `Linf` 从
`1.703e-1` 降至 `5.087e-3`。

复杂几何原始数据：

- `output/nsc_et_gauss_flower_smoke32.csv`
- `output/nsc_et_gauss_heart_smoke32.csv`
- `output/nsc_et_gauss_lshape_smoke32.csv`
