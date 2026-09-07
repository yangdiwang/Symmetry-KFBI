# 仓库布局

本仓库按“可发布核心、仓库内部支撑、可执行程序、测试、基准与工具脚本”划分目录。
随着算例路线和诊断能力增加，程序入口、测试程序与共享 helper 曾混放在 `apps/`；这次
整理按功能拆分路径和构建组织，并增加可独立控制的构建选项与 CTest 注册，不改变数值
算法行为。已有 CMake target 名称、生成的可执行文件名和 C++ namespace 保留历史名称，
便于继续使用现有命令、结果目录和研究记录。逐文件迁移关系见
[file-moves.tsv](file-moves.tsv)。

## 目录职责

| 目录 | 内容与边界 |
| --- | --- |
| `include/kfbim/` | 面向使用者的公开头文件。 |
| `src/` | `kfbim_core` 的实现与随核心安装的内部头文件。 |
| `src/support/` | 算例、测试和基准共享的仓库内部支撑代码，按 `cauchy`、`density`、`diagnostics`、`geometry`、`topology`、`trace` 分类。这里的 targets 不安装，也不属于公开 API。 |
| `apps/laplace/2d/`、`apps/laplace/3d/` | Laplace 方程的二维、三维算例和求解入口。构建后的程序仍位于 `build/apps/`；多配置生成器在其下增加 `Release/`、`Debug/` 等配置目录。 |
| `apps/transmission/` | 传输问题及中心扰动入口。 |
| `apps/shape_optimization/` | 形状优化入口。 |
| `benchmarks/2d/`、`benchmarks/3d/` | 性能、收敛和数据复用基准。 |
| `tests/` | 按 `cauchy`、`density`、`diagnostics`、`geometry`、`solvers`、`topology`、`trace` 分组的回归测试。CMake 注册的测试可统一交给 CTest 运行。 |
| `scripts/validation/` | 验证矩阵运行、结果合并、报告生成和审计脚本。 |
| `scripts/visualization/` | 读取 `output/` 结果并生成图形的 Python 脚本。 |
| `docs/architecture/` | 布局说明与机器可读的文件迁移索引。 |
| `output/` | 运行时生成的数据；不作为源码提交。 |

`src/support/` 解决的是仓库内部多个程序共用实现的问题。它与 `kfbim_core` 分开，避免把
实验路线、诊断格式和算例装配细节扩展成已承诺的安装接口。不要从外部项目依赖这些
targets 或头文件；若某项能力需要成为公开 API，应先迁入 `include/kfbim/` 与相应的核心
实现，并明确安装和兼容性约束。

## 构建开关

| CMake 选项 | 首次配置默认值 | 作用 |
| --- | --- | --- |
| `KFBIM_BUILD_APPS` | `ON` | 构建独立算例和求解程序。 |
| `KFBIM_BUILD_BENCHMARKS` | 跟随 `KFBIM_BUILD_APPS` | 构建性能与收敛基准。 |
| `BUILD_TESTING` | 跟随 `KFBIM_BUILD_APPS` | 构建并向 CTest 注册回归测试。 |
| `KFBIM_BUILD_3D` | `ON` | 构建依赖 CGAL 的三维核心、支撑代码及相关目标。 |
| `KFBIM_BUILD_EXPERIMENTAL_3D` | `OFF` | 启用已有的三维实验算法、研究测试与基准。 |
| `KFBIM_BUILD_REMOTE_SOLVER_ROUTE_TEST` | `OFF` | 启用旧的远程求解路线测试；仍要求同时开启 `KFBIM_BUILD_EXPERIMENTAL_3D`。 |

关闭 apps 并不禁止单独构建测试或基准。例如，可显式使用
`-DKFBIM_BUILD_APPS=OFF -DBUILD_TESTING=ON`。只要 apps、测试或基准任一类别启用，CMake
就会构建它们需要的内部 `src/support/` targets。历史上的 legacy 与 experimental 路线
仍由原有选项和运行时选择器控制；目录移动没有改变默认算法路线。

上表默认值只在新 build 目录首次配置时写入 CMake cache；重新配置已有 build 目录不会
覆盖缓存值，需要用 `-D<选项>=ON/OFF` 显式修改。旧 build 目录若缓存了
`BUILD_TESTING=OFF`，应在重新配置时传入 `-DBUILD_TESTING=ON` 才会构建并注册测试。

## 常用命令

单配置生成器可执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

Visual Studio 等多配置生成器应显式给出配置：

```powershell
cmake -S . -B build
cmake --build build --config Release --parallel 2
ctest --test-dir build -C Release --output-on-failure
```

验证和绘图脚本从新目录直接调用，例如：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/validation/run_kfbi_3d_full_validation.ps1 -DryRun
python scripts/visualization/visualize_shape_opt_2d.py output/shape_opt_transmission_2d
```

已安装 PowerShell 7 时，也可将 `powershell` 换成 `pwsh`。

程序 target 名称和 `build/apps/` 下的运行路径保持不变；脚本自身的位置已经迁入
`scripts/`。需要查找旧路径对应的新位置时，以 [file-moves.tsv](file-moves.tsv) 为准。
