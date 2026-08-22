# NSC-ET-CAS-FD：弧长邻点采样差分密度 jet

日期：2026-08-07

## 改动范围

密度函数仍是物理弧长上的 P3/P2 B 样条系数场，knot、系数、连续性约化和 GMRES 未知量均
不变。本次只替换 crossing Cauchy 多项式所需的
`phi_s, phi_ss, psi_s`：生产路径不再直接使用解析 `B_s/B_ss`，而是在 crossing 周围评价
密度值后做差分。解析导数保留为 A/B 基线和连续性约束构造工具。

正式方案名为 **NSC-ET-CAS-FD**（Continuity-Aligned Arclength Spline with
Finite-Difference Jets）。

## 差分定义

对分支长度 `L_b`、density span 数 `M_b`，定义平均 span

\[
H_b=L_b/M_b,
\qquad \delta=0.5H_b.
\]

在可以使用双侧邻点时：

\[
D_s\rho(s_c)=\frac{\rho(s_c+\delta)-\rho(s_c-\delta)}{2\delta},
\]

\[
D_{ss}\rho(s_c)=
\frac{\rho(s_c+\delta)-2\rho(s_c)+\rho(s_c-\delta)}{\delta^2}.
\]

光滑闭曲线的采样位置按总长度周期回绕。多分支曲线不跨越 feature；若中心 stencil 离开所属
分支，则在可用一侧取 `f_j=rho(s_c+sigma*j*delta)`，`sigma=+1/-1`，并使用

\[
D_s\rho=
\sigma\frac{-11f_0+18f_1-9f_2+2f_3}{6\delta},
\]

\[
D_{ss}\rho=
\frac{2f_0-5f_1+4f_2-f_3}{\delta^2}.
\]

若分支剩余长度不足，`delta` 只缩短到能容纳三个单侧间隔。P3 `phi` 消费值、一阶和二阶
差分；P2 `psi` 只消费值和一阶差分。密度采样值仍由同一个约化系数向量直接评价，不进行
新的拟合。

## 步长选择

以最敏感的 Ellipse/Neumann、N=64/128 扫描 `delta/H`：

| `delta/H` | N=64 bulk L_inf | N=128 bulk L_inf | 64->128 阶 | N=128 GMRES |
|---:|---:|---:|---:|---:|
| 0.125 | 2.5536e-3 | 5.3133e-3 | -1.057 | 14 |
| 0.25 | 1.7896e-3 | 1.1654e-2 | -2.703 | 14 |
| 0.50 | 5.7175e-3 | 1.4218e-3 | 2.008 | 12 |
| 1.00 | 6.4017e-3 | 2.6763e-3 | 1.258 | 14 |

半 span 是该扫描中唯一同时恢复约二阶细层收敛并降低 N=128 误差和迭代数的选择，因此设为
默认值。CLI 可用 `--density-difference-step-over-span` 修改。

## 五种几何 A/B

设置与解析 CAS 实验相同：精确 NURBS，`H约等于1.5h`，N=16/32/64/128，Dirichlet 与
Neumann，GMRES 相对容差 `1e-9`。下表给出 N=128 bulk `L_inf`、64 到 128 的相邻阶和
GMRES 次数。变化率为 FD 相对解析导数；负值表示 FD 误差更小。

| 几何 | BVP | 解析导数：误差 / 阶 / it | 采样差分：误差 / 阶 / it | 误差变化 |
|---|---|---:|---:|---:|
| Circle | D | 9.17831e-5 / 2.048 / 7 | 8.03352e-5 / 2.045 / 7 | -12.5% |
| Circle | N | 2.30059e-4 / 2.324 / 12 | 3.81997e-4 / 3.887 / 11 | +66.0% |
| Ellipse | D | 1.39563e-4 / 2.090 / 7 | 1.35455e-4 / 2.058 / 7 | -2.9% |
| Ellipse | N | 4.07545e-3 / 0.345 / 14 | 1.42178e-3 / 2.008 / 12 | -65.1% |
| Flower | D | 3.84377e-4 / 1.175 / 11 | 2.94432e-4 / 1.582 / 11 | -23.4% |
| Flower | N | 5.54445e-4 / 3.202 / 16 | 5.85398e-4 / 3.771 / 16 | +5.6% |
| Heart | D | 1.21567e-4 / 2.476 / 13 | 1.23726e-4 / 2.367 / 14 | +1.8% |
| Heart | N | 4.46660e-4 / 1.619 / 15 | 1.11121e-3 / 0.059 / 16 | +148.8% |
| L-shape | D | 5.91568e-5 / 3.104 / 17 | 5.91501e-5 / 3.083 / 17 | 0.0% |
| L-shape | N | 5.05385e-4 / 3.708 / 25 | 7.97158e-4 / 4.947 / 24 | +57.7% |

四层 `log(error)`--`log(h)` 最小二乘拟合阶：

| 几何 | D：解析 / FD | N：解析 / FD |
|---|---:|---:|
| Circle | 2.157 / 2.130 | 2.014 / 1.637 |
| Ellipse | 2.154 / 2.114 | 0.731 / 1.629 |
| Flower | 2.819 / 2.903 | 3.719 / 3.161 |
| Heart | 2.451 / 1.869 | 2.078 / 1.439 |
| L-shape | 2.688 / 2.675 | 2.691 / 2.238 |

超二阶值来自制造多项式、网格相位与前渐近，不作为高于二阶的理论主张。

## 稳定性与结论

- 解析与 FD 各 40 组，共 80/80 组 GMRES 全部收敛。
- 平均 GMRES：解析 13.425，FD 12.925；最大次数：25 与 24。
- 两者 unresolved-gap/endpoint fallback 均为 0。
- relocation/rejection 均为 8/80，说明导数方案没有改变几何与 stencil fallback 路径。

采样差分显著修复了 Ellipse/Neumann 的细层高频平台，也改善了 Circle、Ellipse、Flower 的
Dirichlet 误差；但它并非对所有几何占优。Heart/Neumann 在 N=128 出现平台，L-shape 与
Circle/Neumann 的细层误差常数也增大。因此当前代码按用户指定将 FD 设为默认，但保留
`AnalyticBasis`，并把“光滑周期分支使用 FD、feature 分支采用解析或局部自适应差分”的混合
策略列为下一步稳定化方向。

## 复现与数据

```powershell
.\build\apps\laplace_nsc_et_benchmark_2d.exe `
  --geometry all --bvp both `
  --density-coordinate arclength `
  --density-derivative both `
  --density-difference-step-over-span 0.5 `
  16 32 64 128
```

统一的 80 行结果位于：

- `output/nsc_et_CAS_FD_FINAL_comparison.csv`
