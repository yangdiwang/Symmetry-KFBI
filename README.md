# Symmetry-KFBI

这是从原 KFBI 工作区中独立整理出的二维算例仓库。仓库包含 `apps` 下的全部
七个 C++ 算例、两个 Python 可视化脚本，以及它们实际依赖的 KFBI 核心源码和
zFFT。构建不依赖原代码库的相对路径。

## 依赖与构建

需要 CMake 3.20 以上版本和支持 C++17 的编译器。Eigen 3.4 是唯一的外部 C++
依赖：CMake 优先使用系统安装的 Eigen，找不到时会自动下载固定的 3.4.0 版本。
zFFT 已放在 `third_party/zfft` 中。

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

在 Windows 上可执行文件名带 `.exe`。研究计算可省略末尾的 `24`，使用程序内置的
多层网格；快速命令只用于检查完整运行链路。

`neumann_harmonic_jet_python_compatible_2d` 的 `circle` 几何与
`neumann_exterior_trace_circle_2d` 使用同一个圆：圆心 `(0.07, -0.04)`、半径
`0.5`、计算盒 `[-1,1]^2`，并使用相同的四次调和制造解，因此可直接比较误差。
未设置相关环境变量时，所有几何的主力方案均默认使用 `uniform_midpoint` 自由度、
`cubic_harmonic` 4+3 spread 和 `biquadratic_quadratic_two_layer` restrict；后者在
每侧使用两个对称法向点。显式环境变量仍可覆盖这些预设。均匀界面段数默认取网格
crossing 数量的 `0.75` 倍，以降低界面未知量规模。

Dirichlet 入口并存两个迭代格式。稳定的默认格式 `normal_jump_first_kind` 固定
`[u]=g_D`，以 `q=[u_n]` 为未知量，求解
`R^-P(0,q)=g_D-R^-P(g_D,0)`。新增的正确二类格式
`value_jump_second_kind` 固定 `[u_n]=0`，以 `mu=[u]` 为未知量，直接求解
`R^-P(mu,0)=g_D`；按本程序 jump 约定，该离散算子对应 `1/2 I+K_h`。两者每次
算子应用都依次执行 spread、FFT bulk solve 和 restrict，并以实际内迹残差
`u^- - g_D` 检查边界精度。该入口与 Neumann 入口共享全部自由度、4+3 spread、
角点 5+4 三次调和 Cauchy 拟合以及双二次/法向二次两层 restrict 实现。命令中的
`all` 依次运行椭圆、花形、圆和 L 形。

一类格式的 `density_linf/density_l2` 用制造法向数据检验 `q`。二类密度 `mu`
没有直接可用的制造密度参考，因此这两列有意写为 `NaN`，不表示 GMRES 或势函数
计算失败；二类结果仍报告密度均值、内迹残差和区域解误差。

同一程序也接受 `lshape`：固定 L 形顶点为 `(-0.93,-1.04)`、
`(1.07,-1.04)`、`(1.07,-0.04)`、`(0.07,-0.04)`、`(0.07,0.96)`、
`(-0.93,0.96)`，计算盒为 `[-1.5,1.5]^2`，制造解使用三次调和多项式。
不光滑角点本身不设置界面自由度；每条直边独立使用等弧长区间中点。restrict 的
局部 Cauchy stencil 只从同一条边选取邻点。spread 通常使用同边 4+3 插值；若某
界面点按距离选出的原始 4 点跨过角点，则改用 5 个值加 4 个法向数据的三次调和
Cauchy 加权最小二乘拟合。该 `9×7` 超定组合由 L 形加密实验筛选得到。

形状优化和中心扰动结果可分别绘图：

```bash
python apps/visualize_shape_opt_2d.py output/shape_opt_transmission_2d
python apps/visualize_transmission_center_perturb_2d.py output/transmission_center_perturb_2d
```

## 新自由度与 restrict 分支

`neumann_harmonic_jet_python_compatible_2d` 通过环境变量选择比较方式：

- `KFBIM_PYJET_DOF_MODE=crossing|uniform_midpoint|compare`：交点自由度、沿曲线
  近似等弧长区间中点自由度，或两者都运行。区间中点避开参数段端点。
- `KFBIM_PYJET_DIRICHLET_FORMULATION=normal_jump_first_kind|value_jump_second_kind|compare`：
  仅供 Dirichlet 入口选择稳定一类格式、正确二类格式或两者都运行；默认值为
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
- `KFBIM_PYJET_RESTRICT_MODE=biquadratic_quadratic`：每侧三个法向点，双二次
  网格插值，法向二次联合拟合。
- `KFBIM_PYJET_RESTRICT_MODE=biquadratic_quadratic_two_layer`：每侧两个对称法向
  点，双二次网格插值，法向二次联合拟合。
- 法向 restrict 将内侧采样值换算到外侧分支时，主力方案使用
  `[u] + rho [u_n]` 的线性延拓。数值对比表明，在当前每侧两点的法向二次联合
  拟合中，直接使用三次 Cauchy jump 多项式没有带来一致的精度改善。
- `KFBIM_PYJET_RESTRICT_MODE=six_point_quadratic_exterior`：最近网格点、上下左右
  四点和朝界面方向的最近斜点组成六点二次多项式模板。
- `KFBIM_PYJET_RESTRICT_MODE=normal_compare|compare`：分别比较前三种法向格式，
  或比较全部四种格式。

进入插值模板的跨界网格值先使用相应界面自由度的局部 Cauchy 多项式统一调整到
外侧值。每个 crossing 使用最近界面自由度对应的模板计算修正。

PowerShell 直接比较两个 Dirichlet 格式：

```powershell
$env:KFBIM_PYJET_DIRICHLET_FORMULATION = "compare"
$env:KFBIM_PYJET_DOF_MODE = "uniform_midpoint"
$env:KFBIM_PYJET_RESTRICT_MODE = "biquadratic_quadratic_two_layer"
.\build\apps\Release\dirichlet_harmonic_jet_python_compatible_2d.exe all 32 64 128 256 512
```

把第一行改为 `normal_jump_first_kind` 或 `value_jump_second_kind` 可单独运行对应格式。

PowerShell 比较全部 restrict 格式的例子：

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

本仓库整理时以 Release 模式完整构建，并在 `N=24` 上实际运行了七个 C++ 程序和
两个 Python 绘图程序，所有命令退出码均为 0。新增的均匀中点自由度在椭圆测试中
得到以下快速比较；这是低分辨率 smoke test，不应替代多层网格收敛结论。

| restrict 格式 | GMRES | Linf | L2 |
| --- | ---: | ---: | ---: |
| 双三次 + 每侧四点三次法向拟合 | 10 | `3.579266e-4` | `1.711335e-4` |
| 双二次 + 每侧三点二次法向拟合 | 9 | `4.269918e-4` | `1.647600e-4` |
| 双二次 + 每侧两点二次法向拟合 | 10 | `3.503595e-4` | `1.313030e-4` |
| 六网格点外侧二次拟合 | 12 | `3.618789e-4` | `1.520921e-4` |

GitHub Actions 会在每次 push 和 pull request 时重新构建全部目标、运行七个快速
C++ 算例，并执行两个可视化脚本。

第三方依赖说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
