# NSC-ET-PCD-CJ：NURBS 参数协变差分 crossing jet

日期：2026-08-07

## 正式方案

新方案命名为 **NSC-ET-PCD-CJ**（NURBS-Spline-Coefficient Exterior-Trace
KFBI with Parameter-Covariant Difference Crossing Jets）。它是第三条独立
crossing-jet 路径，不替换以下两条已有路径：

1. `PhysicalArclength + AnalyticBasis`：弧长 B 样条解析求导；
2. `PhysicalArclength + SampledFiniteDifference`：弧长邻点采样差分；
3. `NurbsParameter + CovariantParameterFiniteDifference`：本次新增的参数协变差分。

旧 `cubic_harmonic` 方法是面板中心 4 个值条件加 3 个法向条件的三次调和
多项式最小二乘拟合。PCD-CJ 不做该拟合；它先从密度系数场得到真实 crossing
处的 Cauchy jet，再由曲率和 PDE 唯一闭合 crossing-local P2 空间多项式。

## 任意 NURBS 参数上的微分

设精确 NURBS 曲线为 `gamma(xi)`，

```text
J     = |gamma_xi|,
J_xi  = (gamma_xi dot gamma_xixi) / J.
```

密度样条直接定义在 `xi` 上。在 crossing `xi_c` 周围只评价密度值，先形成

```text
rho_xi, rho_xixi,
```

再使用精确几何作协变转换：

```text
rho_s  = rho_xi / J,
rho_ss = rho_xixi / J^2 - J_xi rho_xi / J^3.
```

因此算法不要求曲线预先弧长参数化，也不需要为每个采样点反解全局弧长映射。
弧长仍用于控制离散分辨率 `H=L/M`，但不作为密度基函数或差分坐标。

## 参数 chart 与物理连续性

物理光滑不等于 NURBS 参数高阶光滑。标准有理二次圆在 quarter knots 处物理
曲率连续，但 `J_xi` 可以有左右极限。PCD-CJ 因而把几何内部 unique NURBS
knots 视为参数 chart 边界：

- P3 `phi` 在 chart knot 使用重数 3，参数样条天然 C0，再约束物理
  `phi_s`、`phi_ss`；
- P2 `psi` 在 chart knot 使用重数 2，参数样条天然 C0，再约束物理 `psi_s`；
- 闭合 seam 和不同几何 branch 之间仍按 BVP/几何类型使用原有
  C0/C1/C2/Discontinuous 连接要求；
- 所有约束通过 `raw_coefficients = R * reduced_coefficients` 消元，GMRES 只使用
  reduced coefficients。

设一个几何 branch 内有 `K` 个参数 charts。总密度 span 数固定为

```text
M = max(K, ceil(L / target_spacing)).
```

`M` 按各 chart 的物理长度作最大余数分配，每个 chart 至少一个 span，再在 chart
内部按等物理弧长反求参数 break。加入 chart 边界只增加 raw coefficients 和约束，
不会增加 reduced DOF，也不会让 GMRES 系统过密。

参数 chart 空间不能直接沿用物理弧长周期空间的 uniform trace-row 快捷选择。
在圆的诊断中，该方阵条件数从弧长空间的约 `2` 增长到参数空间
`N=64` 的 `1.82e5` 和 `N=128` 的 `3.82e9`，虽仍被数值秩测试判为满秩，
却无法控制 chart 附近的导数模态。正式参数路径因此使用全部候选值行的物理
子空间 SVD，再用列主元 QR 选取 square trace rows；弧长路径继续保留其均匀点
策略。修正后圆的参数配点条件数约为 `2.8`，细层 GMRES 和二阶收敛恢复。

## crossing 差分格式

参数协变路径的默认物理目标步长为 `Delta_s=0.125 H`，在 crossing 处映射为

```text
Delta_xi = Delta_s / J(xi_c).
```

chart 内部使用三点中心格式。若中心 stencil 会越过 chart、几何 feature 或闭合
seam，则使用 owner chart 内四点单边格式：

```text
D_xi rho = sigma*(-11 rho0 + 18 rho1 - 9 rho2 + 2 rho3)/(6 Delta_xi),
D_xixi rho = (2 rho0 - 5 rho1 + 4 rho2 - rho3)/Delta_xi^2.
```

`sigma` 是参数前进/后退方向。单边步长还限制到最近 density break 距离的三分
之一，使全部四个值位于同一个 P3/P2 多项式段内。参数 seam 不做周期 wrap，
因为一般 NURBS 不保证两端参数 chart 的尺度是同一个仿射坐标。

步长扫描显示，椭圆 Neumann 在 `0.5H` 时 P2 `psi_xi` 的截断误差会被 trace
算子放大；`0.25H` 的细层 bulk 阶约 `1.45`，`0.125H` 恢复到约 `1.98`，且
GMRES 均为 8--9 次。因此 PCD-CJ 独立采用 `0.125H`，原弧长 sampled-FD
仍使用其已校准的 `0.5H`。

## crossing-local P2 闭合

协变转换得到 `phi, phi_s, phi_ss, psi, psi_s` 后，直接使用真实 crossing 的精确
切向、法向和曲率。仓库的曲率约定给出

```text
H_tt = phi_ss + kappa psi,
H_tn = psi_s - kappa phi_s,
H_nn = alpha phi - [f] - phi_ss - kappa psi.
```

随后 spread 与 crossing-owner joint restrict 评价同一个局部空间 P2 多项式；
没有新增空间 stencil fallback。

## 实现入口

- `NurbsDensityDerivativeScheme2D::CovariantParameterFiniteDifference`
- `NurbsDensitySpace2D::covariant_parameter_finite_difference_rows(...)`
- `NurbsDensitySpace2D::evaluate_crossing_covariant_parameter_finite_difference(...)`
- benchmark：`--density-method covariant-parameter-fd`
- 三方案：`--density-method all`

复现命令：

```powershell
.\build\apps\laplace_nsc_et_benchmark_2d.exe `
  --geometry all --bvp both --density-method all `
  --density-difference-step-over-span 0.5 `
  --covariant-difference-step-over-span 0.125 `
  --output-csv output\nsc_et_three_way_20260807.csv `
  32 64 128
```

## 验证

`nurbs_density_space_2d_test` 覆盖：

- 有理圆上显式重组 `J_xi` 协变修正，并确认遗漏该项会产生非零差异；
- 三个 quarter chart knots 和闭合 seam 的单边 jet；
- 参数区间从 `[0,1]` 仿射变换到 `[2,7]` 后的物理 jet 不变性；
- L 形角点 owner branch 内的 P3 单边二次函数再现；
- 加入 chart 后 reduced DOF 仍等于目标 span 数。

`laplace_nsc_et_bvp_2d_test` 覆盖三种 jet 方法乘 Dirichlet/Neumann 两类 BVP。

## 五几何数值结果

统一设置为 `N=32,64,128`、`density_spacing_over_h=1.5`、GMRES
相对容差 `1e-9`；三条路径使用同一精确 NURBS 几何、同一外迹方程、同一
crossing-owner spread/restrict 和同一组制造解。仅改变密度坐标及 crossing jet
求导方式。PCD-CJ 使用 `Delta_s/H=0.125`，弧长 sampled-FD 仍使用
`Delta_s/H=0.5`。

PCD-CJ 每层的 bulk `L_inf` 误差、相邻层收敛阶和 GMRES 迭代数如下。首层
没有可定义的收敛阶，以 `--` 表示。

| 几何 | BVP | N=32：误差 / GMRES | N=64：误差 / 阶 / GMRES | N=128：误差 / 阶 / GMRES |
|---|---|---:|---:|---:|
| 圆 | D | `2.1169e-3 / 8` | `3.8651e-4 / 2.453 / 7` | `9.6491e-5 / 2.002 / 7` |
| 圆 | N | `4.6223e-3 / 9` | `9.1139e-4 / 2.342 / 8` | `2.1457e-4 / 2.087 / 8` |
| 椭圆 | D | `1.6773e-3 / 8` | `4.9541e-4 / 1.759 / 8` | `1.3853e-4 / 1.838 / 7` |
| 椭圆 | N | `1.5260e-2 / 9` | `9.9941e-3 / 0.611 / 9` | `2.5254e-3 / 1.985 / 9` |
| 花形 | D | `9.4796e-3 / 13` | `1.3103e-3 / 2.855 / 12` | `1.0243e-4 / 3.677 / 11` |
| 花形 | N | `1.3300e-2 / 13` | `1.6151e-2 / -0.280 / 14` | `2.6743e-4 / 5.916 / 13` |
| 心脏形 | D | `2.6240e-3 / 11` | `6.6021e-4 / 1.991 / 12` | `1.1981e-4 / 2.462 / 13` |
| 心脏形 | N | `6.5132e-3 / 16` | `6.3544e-3 / 0.036 / 14` | `1.0054e-3 / 2.660 / 16` |
| L 形 | D | `1.7833e-3 / 19` | `5.0825e-4 / 1.811 / 16` | `5.9159e-5 / 3.103 / 17` |
| L 形 | N | `3.9509e-2 / 23` | `7.5706e-3 / 2.384 / 22` | `2.4963e-4 / 4.923 / 26` |

细层 `N=128` 的三路径比较如下。表项为 `bulk L_inf / p_64-128 / GMRES`。
这里的 analytic 基线是弧长 B 样条的解析基函数导数，不是旧的
`cubic_harmonic` 局部最小二乘拟合。

| 几何 | BVP | 弧长解析导数 | 弧长采样差分 | PCD-CJ 参数协变差分 |
|---|---|---:|---:|---:|
| 圆 | D | `9.1783e-5 / 2.048 / 7` | `8.0335e-5 / 2.045 / 7` | `9.6491e-5 / 2.002 / 7` |
| 圆 | N | `2.3006e-4 / 2.324 / 12` | `3.8200e-4 / 3.887 / 11` | `2.1457e-4 / 2.087 / 8` |
| 椭圆 | D | `1.3956e-4 / 2.090 / 7` | `1.3546e-4 / 2.058 / 7` | `1.3853e-4 / 1.838 / 7` |
| 椭圆 | N | `4.0755e-3 / 0.345 / 14` | `1.4218e-3 / 2.008 / 12` | `2.5254e-3 / 1.985 / 9` |
| 花形 | D | `3.8438e-4 / 1.175 / 11` | `2.9443e-4 / 1.582 / 11` | `1.0243e-4 / 3.677 / 11` |
| 花形 | N | `5.5445e-4 / 3.202 / 16` | `5.8540e-4 / 3.771 / 16` | `2.6743e-4 / 5.916 / 13` |
| 心脏形 | D | `1.2157e-4 / 2.476 / 13` | `1.2373e-4 / 2.367 / 14` | `1.1981e-4 / 2.462 / 13` |
| 心脏形 | N | `4.4666e-4 / 1.619 / 15` | `1.1112e-3 / 0.059 / 16` | `1.0054e-3 / 2.660 / 16` |
| L 形 | D | `5.9157e-5 / 3.104 / 17` | `5.9150e-5 / 3.083 / 17` | `5.9159e-5 / 3.103 / 17` |
| L 形 | N | `5.0538e-4 / 3.708 / 25` | `7.9716e-4 / 4.947 / 24` | `2.4963e-4 / 4.923 / 26` |

三种方法各 30 个算例都通过 GMRES 收敛，`unresolved_gap_fallback` 和
`endpoint_fallback` 均为 0。PCD-CJ 在 `N=128` 上的平均 GMRES 次数为
`12.7`，弧长解析和弧长 sampled-FD 分别为 `13.7`、`13.5`。PCD-CJ 的最大
密度配点条件数为 `481.75`，出现在 L 形 Neumann、`N=128`；仍没有产生
fallback。心脏形累计发生 6 次合法 stencil relocation 和 60 次候选拒绝，
它们是认证选择过程，不是降阶 fallback。

花形 Neumann 和心脏形 Neumann 在 `32 -> 64` 区间仍表现出明显的预渐近
波动，因此不能只用这一层判断阶数；两者在 `64 -> 128` 分别恢复到
`5.916` 和 `2.660`。就这组网格而言，PCD-CJ 的主要结论不是每个算例都比
已有路径误差更小，而是：在完全不采用弧长作为密度坐标或差分坐标的条件下，
五种几何和两类 BVP 都保持收敛，且迭代数没有系统性增加。

数据文件：

- `output/nsc_et_three_way_20260807.csv`：三种方法的全部 90 行结果；
- `output/nsc_et_three_way_summary_20260807.csv`：`N=64,128` 的紧凑比较；
- `output/nsc_et_three_way_covariant_parameter_fd_20260807.csv`：PCD-CJ 原始
  30 行结果。

旧 `cubic_harmonic` 路径改变的是 spread 中局部多项式的构造以及采样
stencil，而本节三路径只改变同一个 NSC-ET crossing jet 的密度坐标/求导
方式。因此旧实验数据不能在不改变其他变量的情况下直接并入上表；该旧路径
仍保留在代码中，PCD-CJ 没有替换或调用它。
