# ALS-CJ: Arc-Length B-Spline Crossing Jet

ALS-CJ（弧长 B 样条交点射流）是二维 P2 界面 crossing-owner 修正的正式方案。
它把界面自由度、连续界面插值和交点局部空间多项式分成三个层次：

1. 对每个光滑闭合分量，按面板拓扑排列 P2 端点/中点自由度，并用保留的光滑几何积分得到唯一的周期物理弧长坐标 `l`。
2. 用全部界面自由度建立周期插值系统：`phi=[u]` 使用三次、`C2` 的 B 样条，`psi=[u_n]` 使用二次、`C1` 的 B 样条。几何、弧长、稀疏插值矩阵及其分解只在预处理阶段建立；每次 GMRES 应用只求当前密度的样条系数。
3. 对任意精确 crossing，在其全局弧长位置计算
   `(phi, phi_l, phi_ll, psi, psi_l)`，再结合 crossing 处的切向、法向、曲率、`[f]` 和 PDE 闭合出完整二次局部空间多项式。spread 与 restrict 共享同一份样条系数状态。

因此，当 crossing 穿过面板端点或 B 样条区间边界时，所用射流不会因自由度窗口切换而跳变。P3 的 `phi_ll` 和 P2 的 `psi_l` 都具有目标 `O(h^2)` 插值误差。

在 `LaplaceBvpOptions2D` 中使用：

```cpp
LaplaceBvpOptions2D options;
options.correction_method = LaplaceCorrectionMethod2D::CrossingOwner;
options.crossing_jet_scheme =
    LaplaceCrossingJetScheme2D::ArcLengthBSplineCrossingJet;
```

BVP 决定激活的射流块：Dirichlet 使用 `PhiP3`，Neumann 使用 `PsiP2`，外迹 Cauchy/传输问题使用 `PhiP3PsiP2`。没有周期光滑几何的开放、带角点或仅有位置型 P2 分量仍使用原局部弧长 Lagrange 兼容路径；非精确 crossing 保持既有中心多项式 fallback。

`ExteriorTraceCauchy` 的 `boundary_data` 按已经定向的 jump density 解释：Dirichlet 直接作为 `phi=[u]`，Neumann 直接作为 `psi=[u_n]`。内部和外部 BVP 都不再对这个固定密度块乘物理侧符号；物理侧符号只用于从平均迹和 jump 中选择 physical/ghost trace。
