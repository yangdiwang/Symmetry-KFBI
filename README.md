# Symmetry-KFBI

## 3D Trace93 外迹点优先入口（2026-09-09）

最新 Python 93 算法包的 C++ 对照入口是 `kfbi_trace93_study_3d`，统一算例、
包内参考记录及运行脚本在 [tests/cases/trace_first_3d](tests/cases/trace_first_3d/README.md)。
该入口使用精确双三次 NURBS 几何、独立 analysis 参数密度、Polar-Star Neumann
约束和 Dirichlet 仿射分解；P3 Spread / P2 Restrict 复用外迹中心，
Neumann 为 Q27-cover3，Dirichlet 为 Q64-cover4。

新增对齐对象是 L 柱（14 片）、U 柱（22 片）和半径 0.54 的实心圆柱（14 片），
不是旧空心圆柱。旧 torus 的 `kfbi_trace_first_study_3d` 及其他历史入口保留，
没有将其结果混入新包基准。算法差异、仍保留的认证求交及 Dirichlet 固定约束缺陷见
[算法对齐说明](docs/Trace93_Algorithm_Alignment_20260909.md)；实际完成的验证范围见
[数值验证报告](docs/Trace93_Numerical_Report_20260909.md)。

```powershell
cmake --build build-3d --target kfbi_trace93_study_3d --parallel 2
./tests/cases/trace_first_3d/run_trace93.ps1 -Geometries l,u,cylinder -Levels 32,64 -SkipBuild
```

完整 native support-path 认证仍可能耗时很长；脚本每个几何/网格进程默认限时
20 分钟，可通过 `-TimeoutMinutes` 调整。结果保存在统一算例目录下的独立
`results/` 子目录；包内档案、单测通过和本地完整 PDE 求解分别记录，不互相替代。

## 2D NURBS same-parameter route

The L-shape exterior-trace app now uses the NURBS Same-Parameter Cauchy Jet
(NSP-CJ) by default: exact NURBS geometry, cubic B-spline reconstruction of
`phi=[u]`, quadratic B-spline reconstruction of `psi=[u_n]`, and one shared
NURBS parameter at every crossing. The formulation, tests, convergence table,
and rigid-transform comparison are documented in
[`docs/nsp_cj_2d.md`](docs/nsp_cj_2d.md).

## 历史 3D native-NURBS route

The 3D app constructs the torus, hollow cylinder, and L prism from native
NURBS patches. Interface DOFs are uniform native-parameter cell midpoints;
grid crossings use triangle-interpolated `(u,v)` and exactly `2x2` candidate
DOFs. Crossing ownership may traverse only periodic/G1 edge maps. The 48/28
local Cauchy stencil instead starts from a complete topological patch ring,
including non-G1 neighbors, and ranks that local pool by physical distance;
it therefore cannot jump between spatially close but topologically unrelated
surface sheets. Detailed `N=16,32,64` convergence results are recorded in
[`docs/superpowers/results/2026-07-21-3d-harmonic-jet-results.md`](docs/superpowers/results/2026-07-21-3d-harmonic-jet-results.md).

空心圆柱的 Q27 预处理新增可选的 `KFBIM_3D_SUPPORT_PATH=closest_point` 分支：
使用网格节点到真实 NURBS 边界的最近点构造局部跳跃延拓，并缓存节点投影。
运行入口、适用条件和验证结果见[最近点延拓说明](docs/KFBI3D_Closest_Point_Extension_20260907.md)。

这是从原 KFBI 工作区中独立整理出的二维与三维算例仓库。可执行入口按 Laplace
（`apps/laplace/2d/`、`apps/laplace/3d/`）、传输问题（`apps/transmission/`）和形状优化
（`apps/shape_optimization/`）分组；回归测试、性能基准和工具脚本分别位于 `tests/`、
`benchmarks/` 与 `scripts/`，共享但不公开安装的支撑代码位于 `src/support/`。完整目录职责、
构建开关和逐文件迁移索引见[仓库布局与迁移说明](docs/architecture/README.md)。构建不依赖
原代码库的相对路径。

## 依赖与构建

需要 CMake 3.20 以上版本和支持 C++17 的编译器。所有算例都使用 Eigen 3.4：
CMake 优先使用系统安装的 Eigen，找不到时会自动下载固定的 3.4.0 版本。三维
算例还需要 CGAL 5.6 或更新版本，以及 CGAL 所需的 Boost、GMP 和 MPFR；zFFT
已经放在 `third_party/zfft` 中。

Ubuntu 24.04 可安装完整构建依赖：

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build g++ libeigen3-dev libcgal-dev libgmp-dev libmpfr-dev
```

Windows 可使用 vcpkg 安装 Eigen 和 CGAL，然后把 vcpkg toolchain 传给 CMake：

```powershell
vcpkg install eigen3:x64-windows cgal:x64-windows
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --parallel 2
```

`KFBIM_BUILD_3D` 默认为 `ON`，因此默认配置会检查 CGAL 并确保三维目标确实生成。
只需二维算例时，可以显式配置 `-DKFBIM_BUILD_3D=OFF`。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
```

Windows 使用 Ninja 时可写成：

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
```

首次 Release 构建会实例化较多 Eigen 模板。内存较小的机器建议把并行数保持在
`2`。所有生成数据都写入仓库内的 `output/`，该目录不会提交到 Git。

Python 可视化依赖可用以下命令安装：

```bash
python -m pip install -r requirements.txt
```

## 算例

| 程序 | 用途 | 快速运行 |
| --- | --- | --- |
| `shape_opt_transmission_2d` | 二维传输问题形状优化 | `build/apps/shape_opt_transmission_2d --N 24 --iters 1` |
| `transmission_center_perturb_2d` | 随机中心扰动收敛实验 | `build/apps/transmission_center_perturb_2d --samples 1 --levels 24` |
| `neumann_exterior_trace_circle_2d` | 光滑圆周新旧 Neumann 格式比较 | `build/apps/neumann_exterior_trace_circle_2d 24` |
| `neumann_exterior_trace_lshape_2d` | L 形界面外侧 trace 格式 | `build/apps/neumann_exterior_trace_lshape_2d 24` |
| `neumann_harmonic_jet_case_2d` | 原 P2 harmonic-jet 算例 | `build/apps/neumann_harmonic_jet_case_2d ellipse 24` |
| `neumann_harmonic_jet_python_compatible_2d` | 新自由度与 restrict 格式比较，支持椭圆、花瓣和偏心圆 | `build/apps/neumann_harmonic_jet_python_compatible_2d circle 24` |
| `dirichlet_harmonic_jet_python_compatible_2d` | 同一 spread/restrict 框架、并存一类/二类格式的 Dirichlet BVP | `build/apps/dirichlet_harmonic_jet_python_compatible_2d all 32 64 128 256 512` |
| `neumann_exterior_zero_trace_3d` | 圆环、空心圆柱和 L 棱柱的三维 Neumann 一类/Dirichlet 法向二类收敛实验 | `build/apps/neumann_exterior_zero_trace_3d all 16 32 64` |
| `kfbi_topology_affine_exterior_trace_3d` | 三维 topology-affine 正式基准：Neumann Q27-cover3 值迹与 Dirichlet Q64-cover4 法向迹 | `build/apps/kfbi_topology_affine_exterior_trace_3d all 32` |

在 Windows 上可执行文件名带 `.exe`。研究计算可省略末尾的 `24`，使用程序内置的
多层网格；快速命令只用于检查完整运行链路。

三维入口在参数面板中心建立曲面自由度，使用同一套三次调和 Cauchy 多项式完成
spread 和 restrict。restrict 采用 4×4×4 三次网格插值与法向三次联合拟合，并在
恢复法向导数时除以 `h`。程序同时执行 Neumann 外侧值迹为零的一类格式，以及
Dirichlet 外侧法向迹为零、以法向 jump 为未知量的二类格式；多层误差、GMRES
残差和观测阶分别写入 `output/neumann_exterior_zero_trace_3d/` 下的
`neumann_results.csv` 与 `dirichlet_normal_results.csv`。
三维 Cauchy 邻域默认使用 `g1_nearest`，条件点仅来自本曲面片及其周期/G1 光滑
邻域，不跨越非 G1 尖边。需要研究跨尖边条件点时，可显式设置
`KFBIM_3D_CAUCHY_POLICY=topological_nearest`；该策略沿完整曲面拓扑扩展，但不会
跳到空间上接近而拓扑无关的曲面片。Cauchy 策略同时作用于 Neumann 与程序附带的
Dirichlet-normal 格式。
Neumann 外侧值迹 restrict 默认使用 `joint_tricubic_crossing_owner`：插值模板中的
错误侧节点若可靠地穿过外部 non-G1 曲面片，其 Cauchy 修正会重新归属到穿越面；
可设置 `KFBIM_3D_NEUMANN_TRACE_RESTRICT=joint_tricubic_cauchy` 恢复旧路线。
Dirichlet 外侧法向迹 restrict 也默认使用同一 crossing-owner 路线，并同时作用于
算子项 `W_h(0,\sigma)` 和右端项 `W_h(f,0)`；可设置
`KFBIM_3D_DIRICHLET_NORMAL_RESTRICT=joint_tricubic_cauchy` 恢复旧路线。
以上 crossing-owner 默认仅描述通用 `neumann_exterior_zero_trace_3d` 入口；独立的
`kfbi_topology_affine_exterior_trace_3d` 正式默认采用两条 direct trace 路线：Neumann
设置 `KFBIM_3D_NEUMANN_TRACE_RESTRICT=q27_cover3_all_event_cauchy`，在每个迹点使用
一个 3×3×3 的 Q2 Cartesian tensor-product cover，把 27 个支撑节点沿各自的完整有序
求交事件序列延拓到指定内/外分支后，直接计算界面值迹；Dirichlet 设置
`KFBIM_3D_DIRICHLET_NORMAL_RESTRICT=q64_cover4_all_event_cauchy`，相同地延拓一个
4×4×4 的 Q3 cover，并在迹点直接计算解析插值多项式的外法向导数
`n·∇Q3`。两条路线都不使用离界面的法向采样层，也不做 `a1/h` 法向拟合恢复。
`shared_q10_cubic_gridline_cauchy` 仍可通过上述两个 selector 显式选择，用作
A/B/legacy comparison；它不再是 topology-affine 的基准主路线，也不会作为
未认证事件的隐式回退。
拓扑仿射可执行目标默认设置
`KFBIM_3D_DIRICHLET_JUMP_SPACE=analytic_j0_affine_j1`：已知值 jump
`J_0=g_D` 直接由解析边界值、梯度和 Hessian 构造 crossing jet，未知法向 jump
写成满足拓扑约束的仿射空间 `J_1=c_p+Gz`，不再对已知 jump 做面板样本拟合。
非光滑 feature 上默认使用
`KFBIM_3D_DIRICHLET_FEATURE_COUPLING=broken_sheets`；当解析数据确实来自单一环境
梯度时，可显式选择 `ambient_gradient_affine_mortar`，在 GMRES 前以只依赖左端算子的
unisolvent 行抽取消去特征边与 Vertex/T-star 相容自由度；解析右端不会被拟合或投影。
拓扑 trace projector 还强制正权外迹采样点数严格大于最终自由度数，
否则在 GMRES 前终止并报告 samples/coordinates。

`neumann_harmonic_jet_python_compatible_2d` 的 `circle` 几何与
`neumann_exterior_trace_circle_2d` 使用同一个圆：圆心 `(0.07, -0.04)`、半径
`0.5`、计算盒 `[-1,1]^2`，并使用相同的四次调和制造解，因此可直接比较误差。
未设置相关环境变量时，所有几何的主力方案均默认使用 `uniform_midpoint` 自由度、
`cubic_harmonic` 4+3 spread 和 `bicubic_cubic` restrict；后者使用双三次网格插值
和法向三次联合拟合。显式环境变量仍可覆盖这些预设。均匀界面段数默认取网格
crossing 数量的 `0.75` 倍，以降低界面未知量规模。

Dirichlet 入口并存三个迭代格式。稳定的默认格式 `normal_jump_first_kind` 固定
`[u]=g_D`，以 `q=[u_n]` 为未知量，求解
`R^-P(0,q)=g_D-R^-P(g_D,0)`。新增的正确二类格式
`value_jump_second_kind` 固定 `[u_n]=0`，以 `mu=[u]` 为未知量，直接求解
`R^-P(mu,0)=g_D`；按本程序 jump 约定，该离散算子对应 `1/2 I+K_h`。新的
`normal_jump_second_kind` 仍固定 `[u]=g_D`、以 `q=[u_n]` 为未知量，但求解
`R_n^+P(0,q)=-R_n^+P(g_D,0)`，即令外侧法向迹为零；该算子对应
`+/-1/2 I+K'_h`。三个格式每次
算子应用都依次执行 spread、FFT bulk solve 和 restrict，并以实际内迹残差
`u^- - g_D` 检查边界精度。该入口与 Neumann 入口共享全部自由度、4+3 spread、
角点 5+4 三次调和 Cauchy 拟合以及双二次/法向二次两层 restrict 实现。命令中的
`all` 依次运行椭圆、花形、圆和 L 形。

两个 normal-jump 格式的 `density_linf/density_l2` 用制造法向数据检验 `q`。
value-jump 二类密度 `mu`
没有直接可用的制造密度参考，因此这两列有意写为 `NaN`，不表示 GMRES 或势函数
计算失败；二类结果仍报告密度均值、内迹残差和区域解误差。

同一程序也接受 `lshape`：固定 L 形顶点为 `(-0.93,-1.04)`、
`(1.07,-1.04)`、`(1.07,-0.04)`、`(0.07,-0.04)`、`(0.07,0.96)`、
`(-0.93,0.96)`，计算盒为 `[-1.5,1.5]^2`，制造解使用非多项式调和函数
`exp(0.42*x)*cos(0.42*y)`。不光滑角点本身不设置界面自由度；每条直边独立使用
等弧长区间中点。普通点使用同边 4+3 三次 Cauchy 拟合；若某界面点按距离选出的
原始 4 点跨过角点，则改用跨角点的 5 个值加 4 个法向数据的三次调和拟合。
RHS spread 和 normal restrict 共用这套逐自由度多项式及其系数。

形状优化和中心扰动结果可分别绘图：

```bash
python scripts/visualization/visualize_shape_opt_2d.py output/shape_opt_transmission_2d
python scripts/visualization/visualize_transmission_center_perturb_2d.py output/transmission_center_perturb_2d
```

## 新自由度与 restrict 分支

`neumann_harmonic_jet_python_compatible_2d` 通过环境变量选择比较方式：

- `KFBIM_PYJET_DOF_MODE=crossing|uniform_midpoint|compare`：交点自由度、沿曲线
  近似等弧长区间中点自由度，或两者都运行。区间中点避开参数段端点。
- `KFBIM_PYJET_DIRICHLET_FORMULATION=normal_jump_first_kind|value_jump_second_kind|normal_jump_second_kind|compare`：
  仅供 Dirichlet 入口选择一类格式、两个二类格式或三者都运行；默认值为
  `normal_jump_first_kind`。Neumann 入口忽略该变量。
- `KFBIM_PYJET_UNIFORM_DOF_RATIO=<正数>`：设置 `uniform_midpoint` 段数相对于
  crossing 数量的比例；默认值为 `0.75`。
- `KFBIM_PYJET_SPREAD_MODE=harmonic_jet|crossing_density|quadratic_harmonic|cubic_harmonic`：
  选择完整 harmonic jet、交点密度，或二次/三次局部调和 Cauchy 多项式；后两者
  将跳跃修正 spread 到 crossing 两端网格点。
- `KFBIM_PYJET_QUADRATIC_SPREAD_NEIGHBORS=3` 和
  `KFBIM_PYJET_QUADRATIC_SPREAD_DERIVATIVE_NEIGHBORS=3`：二次 spread 默认让
  中心自由度及其两侧邻点同时提供函数值和法向数据，形成 `6×5` 加权最小二乘
  `3+3` 重构。把导数样本数显式设为 `2` 可恢复仅使用两侧邻点法向数据的对称
  `3+2` 重构。
- `KFBIM_PYJET_CUBIC_SPREAD_NEIGHBORS=4` 和
  `KFBIM_PYJET_CUBIC_SPREAD_DERIVATIVE_NEIGHBORS=3`：三次 spread 默认使用
  4 个函数值样本和其中 3 个法向导数样本，形成 7 项调和基重构。
- `KFBIM_PYJET_CORNER_FIT_DEGREE=3`、`KFBIM_PYJET_CORNER_FIT_NEIGHBORS=5`
  和 `KFBIM_PYJET_CORNER_FIT_DERIVATIVE_NEIGHBORS=4`：仅控制 L 形跨角点自由度
  的三次调和 Cauchy 拟合；完整 harmonic-jet 路径仍保持原来的四次 20+12。
- `KFBIM_PYJET_RESTRICT_MODE=bicubic_cubic`：每侧四个法向点，双三次网格插值，
  法向三次联合拟合。
- `KFBIM_PYJET_RESTRICT_MODE=bicubic_quadratic`：每侧三个法向点，双三次网格
  插值，法向二次联合拟合。
- `KFBIM_PYJET_RESTRICT_MODE=biquadratic_cubic`：每侧四个法向点，双二次网格
  插值，法向三次联合拟合。
- `KFBIM_PYJET_RESTRICT_MODE=biquadratic_quadratic`：每侧三个法向点，双二次
  网格插值，法向二次联合拟合。
- `KFBIM_PYJET_RESTRICT_MODE=biquadratic_quadratic_two_layer`：每侧两个对称法向
  点，双二次网格插值，法向二次联合拟合。
- `normal_jump_second_kind` 从同一联合拟合的无量纲一次项提取外侧法向迹，并除以
  网格间距 `h`。该格式支持上述五种联合法向 restrict，不支持仅恢复外侧函数值的
  `six_point_quadratic_exterior`。
- 法向 restrict 将内侧采样值换算到外侧分支时，主力方案使用
  `[u] + rho [u_n]` 的界面一阶关系；这一步属于最后的一维法向联合拟合，
  与网格插值模板中的局部 Cauchy 修正是两个不同环节。
- 当 spread 为默认的 `cubic_harmonic` 且使用联合法向 restrict 时，插值模板跨界
  节点的 Cauchy 修正与 RHS spread 共用同一个逐自由度三次多项式；双二次/双三次
  只改变网格插值权重，法向二次/三次只改变最后的一维联合拟合。six-point 外侧值
  restrict 保持其独立的二次 Cauchy 修正。
- `KFBIM_PYJET_RESTRICT_MODE=six_point_quadratic_exterior`：最近网格点、上下左右
  四点和朝界面方向的最近斜点组成六点二次多项式模板。
- `KFBIM_PYJET_RESTRICT_MODE=normal_compare|compare`：分别比较原有三种法向格式，
  或保持原有行为、比较原有四种格式。
- `KFBIM_PYJET_RESTRICT_MODE=degree_compare`：固定其他参数，仅比较双二次/双三次
  网格插值与法向二次/三次拟合的四种组合。L 形加密结果见
  [共享 Cauchy 非多项式比较](docs/superpowers/results/2026-07-21-lshape-shared-cauchy-nonpolynomial.md)。

进入插值模板的跨界网格值先使用相应界面自由度的局部 Cauchy 多项式统一调整到
外侧值。每个 crossing 使用最近界面自由度对应的模板计算修正。

PowerShell 直接比较三个 Dirichlet 格式：

```powershell
$env:KFBIM_PYJET_DIRICHLET_FORMULATION = "compare"
$env:KFBIM_PYJET_DOF_MODE = "uniform_midpoint"
$env:KFBIM_PYJET_RESTRICT_MODE = "biquadratic_quadratic_two_layer"
.\build\apps\Release\dirichlet_harmonic_jet_python_compatible_2d.exe all 32 64 128 256 512
```

把第一行改为 `normal_jump_first_kind`、`value_jump_second_kind` 或
`normal_jump_second_kind` 可单独运行对应格式。

PowerShell 比较原有四种 restrict 格式的例子：

```powershell
$env:KFBIM_PYJET_DOF_MODE = "uniform_midpoint"
$env:KFBIM_PYJET_RESTRICT_MODE = "compare"
.\build\apps\neumann_harmonic_jet_python_compatible_2d.exe ellipse 24 48
```

Linux/macOS 使用：

```bash
KFBIM_PYJET_DOF_MODE=uniform_midpoint \
KFBIM_PYJET_RESTRICT_MODE=compare \
./build/apps/neumann_harmonic_jet_python_compatible_2d ellipse 24 48
```

## 本地验证结果

本仓库以 Release 模式完整构建，并在 `N=24` 上运行七个二维 C++ 程序、在 `N=16`
上运行三维程序的全部三种几何，同时运行两个 Python 绘图程序，所有命令退出码均为
0。新增的均匀中点自由度在椭圆测试中得到以下快速比较；这是低分辨率 smoke test，
不应替代多层网格收敛结论。

| restrict 格式 | GMRES | Linf | L2 |
| --- | ---: | ---: | ---: |
| 双三次 + 每侧四点三次法向拟合 | 10 | `3.579266e-4` | `1.711335e-4` |
| 双二次 + 每侧三点二次法向拟合 | 9 | `4.269918e-4` | `1.647600e-4` |
| 双二次 + 每侧两点二次法向拟合 | 10 | `3.503595e-4` | `1.313030e-4` |
| 六网格点外侧二次拟合 | 12 | `3.618789e-4` | `1.520921e-4` |

三维 `all 16` readiness 中，圆环、空心圆柱和 L 棱柱的网格标签不一致数均为 0；
常数 jump 探针的最大无穷范数误差为 `1.77e-14`。

GitHub Actions 会在每次 push 和 pull request 时安装依赖、重新构建默认启用的目标、
运行 C++ smoke 矩阵和验证脚本自测，并执行两个可视化脚本。

第三方依赖说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
