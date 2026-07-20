# Dirichlet 双格式并存设计

## 背景

当前 `dirichlet_harmonic_jet_python_compatible_2d` 已实现稳定的
`normal_jump_first_kind` 格式：固定值 jump 为 Dirichlet 数据，求解法向 jump，
并令内部迹残差为零。该格式已经在椭圆、花形、圆和 L 形上计算到 `N=512`，因此
必须保持默认行为、数值结果和输出兼容性。

本次新增标准双层势二类格式，并允许在同一可执行程序中分别运行或直接比较两种
格式。两者必须共享现有 apps 计算框架、几何、自由度、spread、restrict 和制造解。

## 目标

1. 保留当前一类格式及其默认行为。
2. 新增以 value jump 为未知量的正确二类 Dirichlet 方程。
3. 支持单独运行任一格式或在同一命令中运行两者。
4. 对四种几何计算 `N=32,64,128,256,512`，比较 GMRES、无穷范数误差、收敛阶、
   耗时及局部条件数。
5. 不改变现有 Neumann 路径。

## 非目标

- 本次不实现 Calderón/hypersingular 预条件。
- 本次不为二类格式重新调参或改变 L 形角点 stencil；先使用完全相同的 4+3 与
  角点 5+4 配置测量结果。
- 本次不改变 bulk solver、外边界条件或制造解。

## 数学格式

### 现有一类格式

设 `[u]=u^- - u^+`，固定

```text
[u] = g_D,
```

以 `q=[u_n]` 为未知量，求解

```text
R^- P(g_D, q) - g_D = 0.
```

离散算子只作用于法向 jump：

```text
A_1 q = R^- P(0, q),
b_1   = g_D - R^- P(g_D, 0).
```

该格式命名为 `normal_jump_first_kind`。

### 新增二类格式

令

```text
[u_n] = 0,
```

以 `mu=[u]` 为未知双层密度，直接求解

```text
A_2 mu = R^- P(mu, 0) = g_D.
```

按当前 jump 约定，该算子对应

```text
A_2 = 1/2 I + K_h,
```

因此具有二类方程的单位阵主部。该格式命名为 `value_jump_second_kind`。

## 程序接口

新增环境变量：

```text
KFBIM_PYJET_DIRICHLET_FORMULATION=
  normal_jump_first_kind | value_jump_second_kind | compare
```

- 默认值为 `normal_jump_first_kind`，确保当前稳定结果不变。
- `value_jump_second_kind` 只运行二类格式。
- `compare` 对每个几何、网格和 restrict/dof 组合依次运行两种格式。
- Neumann 可执行程序忽略该变量，继续使用现有路径。

运行日志、汇总表和 CSV 每行增加 `dirichlet_formulation`。CSV 保持现有列顺序，
只在行尾追加新列。文件名增加所选格式后缀，避免覆盖既有一类结果；`compare` 使用
一个带 `_compare` 后缀、同时包含两种格式行的文件。

## 数值统计

两种格式统一报告：

- GMRES 迭代数和收敛标志；
- 内部网格解的 `Linf`、`L2` 误差与逐层收敛阶；
- 内迹边界残差；
- setup、solve 和 total 时间；
- restrict 与 spread 局部条件数。

一类格式继续报告 `q` 对制造法向数据的密度误差。二类密度 `mu` 不是给定边界值
也没有直接可用的制造密度，因此其 `density_linf/density_l2` 写为 `NaN`，避免把
解密度误判为 Dirichlet 数据。二类格式仍报告密度均值，便于诊断迭代解。

收敛阶按同一几何、同一格式、相邻网格的 `Linf` 误差计算；`compare` 模式必须先按
`dirichlet_formulation` 分组，不能把两种格式的相邻行交叉计算。终端汇总也按格式分组。

## 实现边界

- 复用当前 bulk solve、spread、restrict、迹计算和 GMRES 实现，不复制一套求解器。
- 一类算子保持现有代码路径：算子输入只写入法向 jump，值 jump 为零。
- 二类算子输入只写入值 jump，法向 jump 为零，右端直接使用 `g_D`。
- 两种格式完成 GMRES 后均重新构造势，并用内部迹 `R^-` 计算实际边界残差。
- Neumann 目标不解析 Dirichlet 格式变量，也不改变其 CSV 语义。
- 无效的 Dirichlet 格式值应给出清楚错误并返回非零退出码。

## 测试与验收

采用测试先行：先扩展 CI smoke，要求二类模式输出
`dirichlet_formulation=value_jump_second_kind`、GMRES 收敛且误差和边界残差有限；
在实现前该测试必须因缺少格式分派而失败。

实现后按以下层次验证：

1. 构建 apps 两个目标并运行现有 Neumann smoke，确认未回归。
2. 不设置环境变量运行 Dirichlet smoke，确认默认仍是
   `normal_jump_first_kind`，并满足当前误差阈值。
3. 分别运行二类模式和 `compare` smoke，确认格式标签、CSV 分组、有限数值及 GMRES
   收敛。
4. 对 ellipse、flower、circle、lshape 运行 `N=32,64,128,256,512`，直接用 Markdown
   表格对比 GMRES、`Linf`、收敛阶、耗时和局部条件数。

验收重点是“两种格式能独立选择且默认格式结果不变”。二类格式在 L 形角点若因密度
奇异性表现变差，本轮不通过改 stencil 掩盖，而是在结果中明确报告，留作后续优化。

## 预期文件改动

- `apps/neumann_harmonic_jet_python_compatible_2d.cpp`：格式枚举、选择、算子分派、统计。
- `.github/workflows/build-and-smoke.yml`：双格式 smoke 验证。
- `README.md`：数学定义、环境变量和运行方式。

不新增第二份求解器源文件；两个 Dirichlet 格式继续由同一 apps 可执行程序维护。
