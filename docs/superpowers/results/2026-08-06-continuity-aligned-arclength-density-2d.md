# NSC-ET-CAS：连续性对齐弧长密度样条

日期：2026-08-06

> 2026-08-07 更新：本文记录的数值表使用解析 B 样条导数。当前生产路径已改为弧长邻点
> 采样差分；密度系数空间与连续性约化保持不变。新实现和 A/B 结果见
> `2026-08-07-sampled-arclength-density-differences-2d.md`。

## 正式方案

本次实现将 NSC-ET-KFBI 中的密度表示正式记为
**NSC-ET-CAS**（Continuity-Aligned Arclength Spline）。核心原则是：
几何基函数与密度基函数不必相同，但密度空间必须使用物理弧长导数，并且其数值连续性要与
NURBS 几何的连续性分段对齐。

- 几何仍由原始精确 NURBS provider 给出；求交、切向、法向和曲率不经过密度样条近似。
- 在单条光滑闭曲线上，`phi=[u]` 使用原生周期 P3/C2 B 样条，
  `psi=[u_n]` 使用原生周期 P2/C1 B 样条。周期系数直接按模折叠，不再通过开区间样条加
  seam 约束得到周期空间。
- 每个密度 knot span 按目标物理弧长 `H` 布置。NURBS 参数只用于寻找对应几何点，不作为
  密度的微分参数。
- 在多分支边界上，每个光滑分支使用局部物理弧长样条。连接规则为：G2 对齐
  `phi-C2/psi-C1`，G1 对齐 `phi-C1/psi-C0`，G0 对齐
  `phi-C0/psi-discontinuous`。
- crossing 处直接从系数计算 `phi, phi_s, phi_ss, psi, psi_s`；这些量与精确 NURBS 的
  切向、法向、曲率一起构造局部空间 P2 Cauchy 多项式。
- Dirichlet 固定 `phi`、迭代 P2 `psi`；Neumann 固定 `psi`、迭代 P3 `phi`。
- Neumann 的零均值条件使用正交零空间基。多分支外迹行从 trial function subspace 的正交基
  中选择，避免直接对系数坐标做 QR 而产生参数化依赖。

旧方案保留为 `LegacyNurbsParameter`，用于 A/B；生产默认值为 `PhysicalArclength`。

## 连续性与插值验证

圆的四段有理二次 NURBS 在 CAD quarter knot 处，拟合同一光滑密度后得到：

| 表示 | `phi_ss` 最大跳跃 |
|---|---:|
| 旧 NURBS 参数样条 | 2.35154 |
| NSC-ET-CAS 弧长周期样条 | 2.84217e-14 |

这说明旧方案的系数约束只保证了参数导数连续，不能保证物理弧长二阶导数连续；新方案在机器
精度下满足 P3/C2 周期连续性。

椭圆固定 Neumann 数据的 P2 拟合误差也随网格加密下降：

| N | 旧参数样条 | 弧长样条 |
|---:|---:|---:|
| 32 | 1.18270e-2 | 1.50303e-2 |
| 64 | 1.54379e-3 | 1.45013e-3 |
| 128 | 4.09791e-4 | 1.66887e-4 |

## A/B 实验设置

- 几何：圆、椭圆、花形、心脏形、L 形；均使用其精确 NURBS 表示。
- BVP：Interior Dirichlet 与 Interior Neumann。
- 网格：`N=16,32,64,128`，目标密度间距 `H=1.5h`。
- 制造解：`u=x^3-3xy^2+0.35(x^2-y^2)+0.20x-0.15y+0.10`。
- GMRES 相对容差 `1e-9`，restart 80，最大 160 次。
- 除密度坐标外，几何、crossing、spread、restrict、外迹点规则和 gauge 完全相同。

下表给出 N=128 的内部解 `bulk L_inf`、64 到 128 的相邻阶，以及 N=128 的 GMRES
迭代数。百分比为弧长方案相对旧参数方案的误差变化，负值表示改善。

| 几何 | BVP | 旧参数：误差 / 阶 / it | 弧长 CAS：误差 / 阶 / it | 误差变化 |
|---|---|---:|---:|---:|
| circle | D | 1.37035e-4 / 2.386 / 7 | 9.17831e-5 / 2.048 / 7 | -33.02% |
| circle | N | 2.46181e-4 / 2.759 / 12 | 2.30059e-4 / 2.324 / 12 | -6.55% |
| ellipse | D | 2.17559e-4 / 1.365 / 7 | 1.39563e-4 / 2.090 / 7 | -35.85% |
| ellipse | N | 1.52778e-3 / 2.998 / 13 | 4.07545e-3 / 0.345 / 14 | +166.76% |
| flower | D | 3.04491e-4 / 1.505 / 11 | 3.84377e-4 / 1.175 / 11 | +26.24% |
| flower | N | 4.58103e-4 / 3.171 / 16 | 5.54445e-4 / 3.202 / 16 | +21.03% |
| heart | D | 1.19691e-4 / 2.469 / 13 | 1.21567e-4 / 2.476 / 13 | +1.57% |
| heart | N | 1.01730e-3 / 2.577 / 16 | 4.46660e-4 / 1.619 / 15 | -56.09% |
| L-shape | D | 5.91568e-5 / 3.104 / 17 | 5.91568e-5 / 3.104 / 17 | 0.00% |
| L-shape | N | 4.06464e-4 / 4.023 / 26 | 5.05385e-4 / 3.708 / 25 | +24.34% |

四层 `log(error)`--`log(h)` 最小二乘拟合阶为：

| 几何 | D：旧 / CAS | N：旧 / CAS |
|---|---:|---:|
| circle | 2.052 / 2.157 | 1.893 / 2.014 |
| ellipse | 1.874 / 2.154 | 1.738 / 0.731 |
| flower | 2.688 / 2.819 | 3.304 / 3.720 |
| heart | 2.062 / 2.451 | 1.862 / 2.078 |
| L-shape | 2.688 / 2.688 | 2.781 / 2.691 |

超二阶的观测值来自制造多项式、粗层前渐近和网格相位，不作为高于二阶的理论主张。

## 稳定性

- 两种表示各 40 组，共 80/80 组 GMRES 全部收敛。
- 平均 GMRES 次数均为 13.43；最大次数从旧方案的 26 变为 CAS 的 25。
- unresolved-gap fallback 和 endpoint fallback 均为 0。
- stencil relocation/rejection 两种方案完全相同，仅 Heart 每行出现 1/10；没有进入 crossing
  fallback。
- Circle、Ellipse、Flower 和 L-shape 的 relocation/rejection 均为 0。

## 结论与已知限制

CAS 解决的是表示层面的确定性问题：密度导数现在就是物理弧长导数，P3/P2 的连续性与几何
feature 对齐，且 crossing 移动时密度 jet 连续变化。它不保证在每一个有限网格上都降低误差
常数；Flower 与 L-shape/Neumann 的 N=128 常数略大，但 GMRES 稳定性没有恶化。

唯一明显未解决的问题是 Ellipse/Neumann：固定 P2 `psi` 的独立拟合是收敛的，所有 fallback
为零，外迹残差约 `1e-10`，但未知 P3 `phi` 的细层误差出现平台。现有证据指向外迹点配系统
对高频 `phi_ss` 模式控制不足，而不是周期接缝、弧长映射、GMRES 未收敛或 restrict fallback。
因此生产默认仍采用 CAS；Ellipse/Neumann 的高频稳定化应作为独立的离散外迹问题继续处理，
不应退回到参数导数不连续的旧表示。

## 数据文件

- `output/nsc_et_CONT_FINAL_circle.csv`
- `output/nsc_et_CONT_FINAL_ellipse.csv`
- `output/nsc_et_CONT_FINAL_flower.csv`
- `output/nsc_et_CONT_FINAL_heart.csv`
- `output/nsc_et_CONT_FINAL_lshape.csv`

每个文件均含 16 行：两种密度表示、两类 BVP、四层网格，并包含误差、相邻阶、GMRES、真实
外迹残差和全部 fallback 计数。
