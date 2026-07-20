# Symmetry-KFBI

这是从原 KFBI 工作区中独立整理出的二维算例仓库。仓库包含 `apps` 下的全部
六个 C++ 算例、两个 Python 可视化脚本，以及它们实际依赖的 KFBI 核心源码和
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
| `neumann_harmonic_jet_python_compatible_2d` | 新自由度与 restrict 格式比较 | `build/apps/neumann_harmonic_jet_python_compatible_2d ellipse 24` |

在 Windows 上可执行文件名带 `.exe`。研究计算可省略末尾的 `24`，使用程序内置的
多层网格；快速命令只用于检查完整运行链路。

形状优化和中心扰动结果可分别绘图：

```bash
python apps/visualize_shape_opt_2d.py output/shape_opt_transmission_2d
python apps/visualize_transmission_center_perturb_2d.py output/transmission_center_perturb_2d
```

## 新自由度与 restrict 分支

`neumann_harmonic_jet_python_compatible_2d` 通过环境变量选择比较方式：

- `KFBIM_PYJET_DOF_MODE=crossing|uniform_midpoint|compare`：交点自由度、沿曲线
  近似等弧长区间中点自由度，或两者都运行。区间中点避开参数段端点。
- `KFBIM_PYJET_RESTRICT_MODE=bicubic_cubic`：每侧四个法向点，双三次网格插值，
  法向三次联合拟合。
- `KFBIM_PYJET_RESTRICT_MODE=biquadratic_quadratic`：每侧三个法向点，双二次
  网格插值，法向二次联合拟合。
- `KFBIM_PYJET_RESTRICT_MODE=biquadratic_quadratic_two_layer`：每侧两个对称法向
  点，双二次网格插值，法向二次联合拟合。
- `KFBIM_PYJET_RESTRICT_MODE=six_point_quadratic_exterior`：最近网格点、上下左右
  四点和朝界面方向的最近斜点组成六点二次多项式模板。
- `KFBIM_PYJET_RESTRICT_MODE=normal_compare|compare`：分别比较前三种法向格式，
  或比较全部四种格式。

进入插值模板的跨界网格值先使用相应界面自由度的局部 Cauchy 多项式统一调整到
外侧值。每个 crossing 使用最近界面自由度对应的模板计算修正。

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

本仓库整理时以 Release 模式完整构建，并在 `N=24` 上实际运行了六个 C++ 程序和
两个 Python 绘图程序，所有命令退出码均为 0。新增的均匀中点自由度在椭圆测试中
得到以下快速比较；这是低分辨率 smoke test，不应替代多层网格收敛结论。

| restrict 格式 | GMRES | Linf | L2 |
| --- | ---: | ---: | ---: |
| 双三次 + 每侧四点三次法向拟合 | 10 | `3.579266e-4` | `1.711335e-4` |
| 双二次 + 每侧三点二次法向拟合 | 9 | `4.269918e-4` | `1.647600e-4` |
| 双二次 + 每侧两点二次法向拟合 | 10 | `3.503595e-4` | `1.313030e-4` |
| 六网格点外侧二次拟合 | 12 | `3.618789e-4` | `1.520921e-4` |

GitHub Actions 会在每次 push 和 pull request 时重新构建全部目标、运行六个快速
C++ 算例，并执行两个可视化脚本。

第三方依赖说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
