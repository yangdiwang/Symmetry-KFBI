# KFBI 3D Dirichlet / Neumann 数值验证报告

> **历史 Q10 对照报告（2026-08-20 快照）。** 本文数据早于当前生产基准，不能用于验证
> Neumann `Q27-cover3` 或 Dirichlet `Q64-cover4`。其中关于“单事件数据结构”的描述也已
> 过时；当前生产路线使用认证的 all-event support-path 延拓。新的 Q27/Q64 数值结论应只
> 从带 `trace_restrict_mode` 列的重新生成报告中引用。

- 数据源：`output/kfbi_3d_full_validation/preliminary0820/merged_all_results.csv`
- 记录数：150；后端/几何组合数：7；重复主键组：0
- 生成时间：2026-08-20 20:22:25 +08:00
- 本文档仅由合并 CSV 生成；生成过程不运行求解器，也不改写源 CSV。
- 输入完整性模式：AllowPartial（未强制固定目录）

## 方法与判据说明

- 两类后端均求解三维 Laplace 方程 `Delta u=0`，使用同一调和制造解 `u*(x,y,z)=exp(0.35x) cos(0.21y) cos(0.28z)`；`0.35^2=0.21^2+0.28^2`。
- Cartesian 计算盒统一为 `[-1.5,1.5]^3`，网格宽度 `h=3/N`，本报告的收敛层为 `N=32,64,128`。GMRES 相对容差为 `2e-10`，最大迭代数为 80。
- `topology_native` 与 `general_cap` 是两条不同的几何/迹离散后端，表格保留后端标签，不把两者的自由度定义混为一谈。
- Neumann 问题以未知值跳 `J0` 和已知法向跳 `gN` 驱动界面问题，并令外侧值迹为零；在合法零均值未知空间中直接消去一个自由度，再把外迹残差投影到同一最终空间。Dirichlet 问题以已知值跳和未知法向跳驱动界面问题，并令外侧法向迹为零；它保留完整密度坐标。
- 三层刚体收敛序列按后端选择：`general_cap` 使用绕 `(1,2,3)` 轴旋转 17 度再平移的 `R17+T1`，`topology_native` 使用纯平移 `Tx=(0.137,0,0)`。所有七个几何在 `N=32` 仍完整测试八种姿态。
- 姿态常量：旋转轴为 `(1,2,3)/sqrt(14)`，旋转中心为 `(0.07,-0.07,0.02)`；`Ty=(0,-0.083,0)`、`Tz=(0,0,0.061)`、`T1=(0.137,-0.083,0.061)`、`T2=(-0.109,0.151,-0.047)`。
- `topology_native` 未把 `R17+T1` 用作三层收敛姿态：U 柱在该姿态的 `N=64` 网格上出现一条含三个已认证物理交点的 label-changing Cartesian edge，而当前 correction/spread 数据结构严格要求单交点。测试因此保持 fail-closed，并改用三层均可认证的 `Tx`；没有放宽求交容差或挑选其中一个根。
- 收敛阶使用同一“后端/几何/方程/姿态”内相邻网格的 `p=log(E_prev/E_cur)/log(h_prev/h_cur)`；本报告显示合并器写入的 order 字段，不自行伪造缺失阶。
- 代数收敛严格读取 `gmres_converged`。物理收敛严格读取 `physical_converged`；其阈值由求解程序定义，CSV 中未编码阈值时本报告不反推。
- `topology_native` 的 `physical_converged` 还要求未投影外迹条件误差通过约 `10*GMRES tolerance=2e-9` 的严格阈值；有限网格上的离散误差可令该标志为假，即使 GMRES 和投影算子残差已经收敛。`general_cap` 未输出该布尔标志，报告保留为“未提供”。
- 误差列：`Eint=interior_linf`，`Eρ=density_linf`，`Eext=exterior_condition_linf`，`Ebc=boundary_residual_linf`；`relres` 和 `op-res` 分别为 GMRES 相对残差与算子残差。
- CSV 中出现的 `zero_space_solver`：`mean_free_pivot_elimination`、`not_applicable`。

## 基准姿态

范围：全部后端；姿态 ID：`baseline`。以下阶数由同一后端、几何、方程和姿态的相邻网格层计算；首个有效层没有前驱，阶通常为 NaN。

### Dirichlet

| 后端 / 几何 | N | h | Eint | pint | Eρ | pρ | Eext | pext | Ebc | pbc | GMRES it | relres | op-res | ncoef | DOF（pre→final） | 代数收敛 | physical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| topology_native / torus | 32 | 9.375E-02 | 8.663E-03 | NaN | 1.050E-01 | NaN | 1.333E-02 | NaN | 1.404E-02 | NaN | 13 | 4.245E-11 | 3.764E-12 | 4 | 64→64 | 是 | 否 |
| topology_native / cylinder | 32 | 9.375E-02 | 7.754E-03 | NaN | 7.131E-02 | NaN | 1.309E-02 | NaN | 1.042E-02 | NaN | 19 | 7.021E-11 | 7.695E-12 | 4 | 128→128 | 是 | 否 |
| topology_native / l_prism | 32 | 9.375E-02 | 4.662E-04 | NaN | 5.893E-03 | NaN | 1.575E-04 | NaN | 4.949E-04 | NaN | 18 | 1.022E-10 | 6.068E-12 | 5 | 260→260 | 是 | 否 |
| topology_native / u_prism | 32 | 9.375E-02 | 6.512E-04 | NaN | 5.640E-03 | NaN | 3.754E-04 | NaN | 7.247E-04 | NaN | 20 | 1.689E-10 | 1.094E-11 | 4 | 224→224 | 是 | 否 |
| general_cap / sphere | 32 | 9.375E-02 | 2.079E-04 | NaN | 4.563E-03 | NaN | 2.231E-03 | NaN | NaN | NaN | 10 | 3.053E-11 | 2.051E-11 | 6 | 184→184 | 是 | 未提供 |
| general_cap / sphere | 64 | 4.688E-02 | 6.468E-06 | 5.006 | 3.657E-04 | 3.641 | 1.014E-04 | 4.459 | NaN | NaN | 9 | 1.771E-10 | 4.067E-10 | 9 | 646→646 | 是 | 未提供 |
| general_cap / sphere | 128 | 2.344E-02 | 5.301E-07 | 3.609 | 4.676E-05 | 2.967 | 4.355E-06 | 4.542 | NaN | NaN | 10 | 7.128E-11 | 2.172E-10 | 16 | 2704→2704 | 是 | 未提供 |
| general_cap / ellipsoid | 32 | 9.375E-02 | 4.281E-04 | NaN | 1.208E-02 | NaN | 7.784E-03 | NaN | NaN | NaN | 10 | 1.035E-10 | 6.303E-11 | 6 | 184→184 | 是 | 未提供 |
| general_cap / ellipsoid | 64 | 4.688E-02 | 1.175E-05 | 5.187 | 1.141E-03 | 3.403 | 3.339E-04 | 4.543 | NaN | NaN | 10 | 1.539E-10 | 1.883E-10 | 9 | 646→646 | 是 | 未提供 |
| general_cap / ellipsoid | 128 | 2.344E-02 | 5.489E-07 | 4.420 | 1.640E-04 | 2.799 | 1.812E-05 | 4.204 | NaN | NaN | 11 | 4.898E-11 | 1.105E-10 | 15 | 2326→2326 | 是 | 未提供 |
| general_cap / flower | 32 | 9.375E-02 | 3.644E-03 | NaN | 7.726E-02 | NaN | 4.137E-02 | NaN | NaN | NaN | 12 | 4.622E-11 | 3.276E-11 | 6 | 184→184 | 是 | 未提供 |
| general_cap / flower | 64 | 4.688E-02 | 3.997E-04 | 3.188 | 8.638E-03 | 3.161 | 4.656E-03 | 3.152 | NaN | NaN | 12 | 8.059E-11 | 6.889E-11 | 9 | 646→646 | 是 | 未提供 |
| general_cap / flower | 128 | 2.344E-02 | 1.712E-05 | 4.546 | 8.597E-04 | 3.329 | 5.131E-04 | 3.182 | NaN | NaN | 13 | 5.951E-11 | 1.899E-10 | 16 | 2704→2704 | 是 | 未提供 |

### Neumann

| 后端 / 几何 | N | h | Eint | pint | Eρ | pρ | Eext | pext | Ebc | pbc | GMRES it | relres | op-res | ncoef | DOF（pre→final） | 代数收敛 | physical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| topology_native / torus | 32 | 9.375E-02 | 2.560E-03 | NaN | 5.106E-03 | NaN | 1.642E-03 | NaN | NaN | NaN | 16 | 1.507E-10 | 1.775E-12 | 4 | 64→63 | 是 | 否 |
| topology_native / cylinder | 32 | 9.375E-02 | 1.568E-03 | NaN | 2.327E-03 | NaN | 1.282E-03 | NaN | NaN | NaN | 16 | 1.704E-10 | 4.663E-12 | 4 | 32→31 | 是 | 否 |
| topology_native / l_prism | 32 | 9.375E-02 | 2.434E-05 | NaN | 2.365E-05 | NaN | 4.069E-06 | NaN | NaN | NaN | 19 | 6.377E-11 | 1.934E-12 | 5 | 50→49 | 是 | 否 |
| topology_native / u_prism | 32 | 9.375E-02 | 5.327E-05 | NaN | 5.741E-05 | NaN | 3.000E-05 | NaN | NaN | NaN | 15 | 1.021E-15 | 3.123E-17 | 4 | 16→15 | 是 | 否 |
| general_cap / sphere | 32 | 9.375E-02 | 1.818E-03 | NaN | 3.148E-03 | NaN | 1.427E-03 | NaN | NaN | NaN | 9 | 1.438E-10 | 6.729E-12 | 6 | 184→183 | 是 | 未提供 |
| general_cap / sphere | 64 | 4.688E-02 | 1.516E-04 | 3.584 | 2.158E-04 | 3.867 | 5.900E-05 | 4.596 | NaN | NaN | 8 | 1.078E-10 | 3.248E-12 | 9 | 646→645 | 是 | 未提供 |
| general_cap / sphere | 128 | 2.344E-02 | 2.952E-05 | 2.360 | 3.333E-05 | 2.695 | 2.600E-06 | 4.504 | NaN | NaN | 8 | 3.578E-11 | 1.046E-12 | 16 | 2704→2703 | 是 | 未提供 |
| general_cap / ellipsoid | 32 | 9.375E-02 | 1.753E-03 | NaN | 3.661E-03 | NaN | 1.844E-03 | NaN | NaN | NaN | 10 | 1.154E-10 | 3.839E-12 | 6 | 184→183 | 是 | 未提供 |
| general_cap / ellipsoid | 64 | 4.688E-02 | 2.503E-04 | 2.808 | 3.064E-04 | 3.579 | 7.335E-05 | 4.652 | NaN | NaN | 10 | 2.350E-11 | 5.327E-13 | 9 | 646→645 | 是 | 未提供 |
| general_cap / ellipsoid | 128 | 2.344E-02 | 5.223E-05 | 2.261 | 5.561E-05 | 2.462 | 3.953E-06 | 4.214 | NaN | NaN | 9 | 9.848E-11 | 1.620E-12 | 15 | 2326→2325 | 是 | 未提供 |
| general_cap / flower | 32 | 9.375E-02 | 8.505E-03 | NaN | 2.069E-02 | NaN | 1.065E-02 | NaN | NaN | NaN | 12 | 1.866E-11 | 1.059E-12 | 6 | 184→183 | 是 | 未提供 |
| general_cap / flower | 64 | 4.688E-02 | 5.114E-04 | 4.056 | 9.623E-04 | 4.426 | 4.802E-04 | 4.471 | NaN | NaN | 12 | 2.777E-11 | 1.202E-12 | 9 | 646→645 | 是 | 未提供 |
| general_cap / flower | 128 | 2.344E-02 | 6.418E-05 | 2.994 | 1.030E-04 | 3.223 | 2.966E-05 | 4.017 | NaN | NaN | 12 | 2.705E-11 | 7.074E-13 | 16 | 2704→2703 | 是 | 未提供 |

## general_cap 收敛刚体变换姿态：R17+T1

范围：后端 `general_cap`；姿态 ID：`rot_axis123_17deg_t_xyz_1`。以下阶数由同一后端、几何、方程和姿态的相邻网格层计算；首个有效层没有前驱，阶通常为 NaN。

### Dirichlet

| 后端 / 几何 | N | h | Eint | pint | Eρ | pρ | Eext | pext | Ebc | pbc | GMRES it | relres | op-res | ncoef | DOF（pre→final） | 代数收敛 | physical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| general_cap / sphere | 32 | 9.375E-02 | 2.130E-04 | NaN | 4.562E-03 | NaN | 2.296E-03 | NaN | NaN | NaN | 10 | 1.728E-10 | 2.065E-10 | 6 | 184→184 | 是 | 未提供 |
| general_cap / sphere | 64 | 4.688E-02 | 6.881E-06 | 4.952 | 3.611E-04 | 3.659 | 1.051E-04 | 4.449 | NaN | NaN | 11 | 4.628E-11 | 9.875E-11 | 9 | 646→646 | 是 | 未提供 |
| general_cap / sphere | 128 | 2.344E-02 | 3.762E-07 | 4.193 | 4.667E-05 | 2.952 | 4.448E-06 | 4.563 | NaN | NaN | 12 | 6.841E-11 | 4.609E-10 | 16 | 2704→2704 | 是 | 未提供 |
| general_cap / ellipsoid | 32 | 9.375E-02 | 4.419E-04 | NaN | 1.209E-02 | NaN | 7.740E-03 | NaN | NaN | NaN | 12 | 3.523E-11 | 3.705E-11 | 6 | 184→184 | 是 | 未提供 |
| general_cap / ellipsoid | 64 | 4.688E-02 | 1.659E-05 | 4.736 | 1.152E-03 | 3.391 | 3.402E-04 | 4.508 | NaN | NaN | 11 | 1.585E-10 | 3.716E-10 | 9 | 646→646 | 是 | 未提供 |
| general_cap / ellipsoid | 128 | 2.344E-02 | 5.583E-07 | 4.893 | 1.623E-04 | 2.828 | 1.846E-05 | 4.204 | NaN | NaN | 11 | 1.842E-10 | 8.686E-10 | 15 | 2326→2326 | 是 | 未提供 |
| general_cap / flower | 32 | 9.375E-02 | 3.877E-03 | NaN | 7.918E-02 | NaN | 4.172E-02 | NaN | NaN | NaN | 13 | 2.929E-11 | 2.969E-11 | 6 | 184→184 | 是 | 未提供 |
| general_cap / flower | 64 | 4.688E-02 | 4.365E-04 | 3.151 | 8.619E-03 | 3.199 | 4.711E-03 | 3.147 | NaN | NaN | 12 | 1.689E-10 | 4.885E-10 | 9 | 646→646 | 是 | 未提供 |
| general_cap / flower | 128 | 2.344E-02 | 2.136E-05 | 4.353 | 8.647E-04 | 3.317 | 5.133E-04 | 3.198 | NaN | NaN | 13 | 1.133E-10 | 4.841E-10 | 16 | 2704→2704 | 是 | 未提供 |

### Neumann

| 后端 / 几何 | N | h | Eint | pint | Eρ | pρ | Eext | pext | Ebc | pbc | GMRES it | relres | op-res | ncoef | DOF（pre→final） | 代数收敛 | physical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| general_cap / sphere | 32 | 9.375E-02 | 1.605E-03 | NaN | 3.332E-03 | NaN | 1.555E-03 | NaN | NaN | NaN | 11 | 1.378E-10 | 7.206E-12 | 6 | 184→183 | 是 | 未提供 |
| general_cap / sphere | 64 | 4.688E-02 | 1.631E-04 | 3.299 | 2.129E-04 | 3.968 | 6.121E-05 | 4.667 | NaN | NaN | 11 | 4.035E-11 | 2.367E-12 | 9 | 646→645 | 是 | 未提供 |
| general_cap / sphere | 128 | 2.344E-02 | 2.999E-05 | 2.443 | 3.266E-05 | 2.704 | 2.574E-06 | 4.572 | NaN | NaN | 10 | 1.236E-10 | 3.946E-12 | 16 | 2704→2703 | 是 | 未提供 |
| general_cap / ellipsoid | 32 | 9.375E-02 | 2.437E-03 | NaN | 4.032E-03 | NaN | 1.823E-03 | NaN | NaN | NaN | 12 | 1.259E-10 | 5.046E-12 | 6 | 184→183 | 是 | 未提供 |
| general_cap / ellipsoid | 64 | 4.688E-02 | 2.445E-04 | 3.317 | 3.021E-04 | 3.738 | 7.493E-05 | 4.604 | NaN | NaN | 12 | 8.319E-11 | 3.090E-12 | 9 | 646→645 | 是 | 未提供 |
| general_cap / ellipsoid | 128 | 2.344E-02 | 5.055E-05 | 2.274 | 5.479E-05 | 2.463 | 3.994E-06 | 4.229 | NaN | NaN | 12 | 3.503E-11 | 1.100E-12 | 15 | 2326→2325 | 是 | 未提供 |
| general_cap / flower | 32 | 9.375E-02 | 1.167E-02 | NaN | 2.107E-02 | NaN | 1.066E-02 | NaN | NaN | NaN | 15 | 3.283E-11 | 1.636E-12 | 6 | 184→183 | 是 | 未提供 |
| general_cap / flower | 64 | 4.688E-02 | 5.270E-04 | 4.469 | 9.725E-04 | 4.437 | 4.823E-04 | 4.466 | NaN | NaN | 14 | 1.302E-10 | 4.476E-12 | 9 | 646→645 | 是 | 未提供 |
| general_cap / flower | 128 | 2.344E-02 | 6.976E-05 | 2.917 | 1.017E-04 | 3.257 | 2.954E-05 | 4.029 | NaN | NaN | 14 | 1.159E-10 | 2.278E-12 | 16 | 2704→2703 | 是 | 未提供 |

## topology_native 收敛刚体变换姿态：Tx

范围：后端 `topology_native`；姿态 ID：`tx_p0137`。以下阶数由同一后端、几何、方程和姿态的相邻网格层计算；首个有效层没有前驱，阶通常为 NaN。

### Dirichlet

| 后端 / 几何 | N | h | Eint | pint | Eρ | pρ | Eext | pext | Ebc | pbc | GMRES it | relres | op-res | ncoef | DOF（pre→final） | 代数收敛 | physical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| topology_native / torus | 32 | 9.375E-02 | 6.459E-03 | NaN | 9.407E-02 | NaN | 1.214E-02 | NaN | 1.042E-02 | NaN | 13 | 4.484E-11 | 4.108E-12 | 4 | 64→64 | 是 | 否 |
| topology_native / torus | 64 | 4.688E-02 | 1.457E-03 | 2.148 | 1.937E-02 | 2.280 | 2.870E-03 | 2.081 | 1.785E-03 | 2.545 | 12 | 1.546E-10 | 8.786E-12 | 6 | 256→256 | 是 | 否 |
| topology_native / cylinder | 32 | 9.375E-02 | 2.655E-03 | NaN | 4.955E-02 | NaN | 1.263E-02 | NaN | 4.745E-03 | NaN | 19 | 9.822E-11 | 7.380E-12 | 4 | 128→128 | 是 | 否 |
| topology_native / cylinder | 64 | 4.688E-02 | 8.904E-04 | 1.576 | 1.368E-02 | 1.856 | 4.206E-03 | 1.586 | 1.174E-03 | 2.015 | 20 | 6.962E-11 | 4.059E-12 | 7 | 560→560 | 是 | 否 |
| topology_native / cylinder | 128 | 2.344E-02 | 3.370E-04 | 1.402 | 4.297E-03 | 1.671 | 2.675E-04 | 3.975 | 3.761E-04 | 1.643 | 21 | 5.473E-11 | 1.997E-12 | 11 | 1584→1584 | 是 | 否 |
| topology_native / l_prism | 32 | 9.375E-02 | 4.363E-04 | NaN | 6.007E-03 | NaN | 1.792E-04 | NaN | 5.435E-04 | NaN | 18 | 6.954E-11 | 5.988E-12 | 5 | 260→260 | 是 | 否 |
| topology_native / l_prism | 64 | 4.688E-02 | 2.290E-04 | 0.930 | 3.444E-03 | 0.803 | 1.104E-04 | 0.699 | 2.384E-04 | 1.189 | 20 | 5.802E-11 | 5.249E-12 | 7 | 532→532 | 是 | 否 |
| topology_native / l_prism | 128 | 2.344E-02 | 1.171E-04 | 0.968 | 2.329E-03 | 0.565 | 5.845E-05 | 0.918 | 1.221E-04 | 0.965 | 19 | 1.549E-10 | 7.236E-12 | 12 | 1632→1632 | 是 | 否 |
| topology_native / u_prism | 32 | 9.375E-02 | 6.311E-04 | NaN | 5.373E-03 | NaN | 3.946E-04 | NaN | 7.388E-04 | NaN | 20 | 1.093E-10 | 6.342E-12 | 4 | 224→224 | 是 | 否 |
| topology_native / u_prism | 64 | 4.688E-02 | 3.587E-04 | 0.815 | 3.735E-03 | 0.525 | 1.711E-04 | 1.206 | 3.976E-04 | 0.894 | 20 | 1.986E-10 | 1.408E-11 | 6 | 552→552 | 是 | 否 |
| topology_native / u_prism | 128 | 2.344E-02 | 1.408E-04 | 1.349 | 2.116E-03 | 0.820 | 8.715E-05 | 0.973 | 1.458E-04 | 1.448 | 20 | 1.333E-10 | 7.303E-12 | 10 | 1640→1640 | 是 | 否 |

### Neumann

| 后端 / 几何 | N | h | Eint | pint | Eρ | pρ | Eext | pext | Ebc | pbc | GMRES it | relres | op-res | ncoef | DOF（pre→final） | 代数收敛 | physical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| topology_native / torus | 32 | 9.375E-02 | 2.558E-03 | NaN | 4.430E-03 | NaN | 1.690E-03 | NaN | NaN | NaN | 17 | 1.227E-10 | 1.426E-12 | 4 | 64→63 | 是 | 否 |
| topology_native / torus | 64 | 4.688E-02 | 1.161E-04 | 4.461 | 2.211E-04 | 4.325 | 8.195E-05 | 4.366 | NaN | NaN | 17 | 8.536E-11 | 6.533E-13 | 6 | 256→255 | 是 | 否 |
| topology_native / cylinder | 32 | 9.375E-02 | 1.825E-03 | NaN | 2.561E-03 | NaN | 1.327E-03 | NaN | NaN | NaN | 17 | 3.927E-11 | 1.122E-12 | 4 | 32→31 | 是 | 否 |
| topology_native / cylinder | 64 | 4.688E-02 | 3.711E-05 | 5.620 | 6.764E-05 | 5.243 | 2.866E-05 | 5.533 | NaN | NaN | 18 | 1.269E-10 | 2.165E-12 | 7 | 320→319 | 是 | 否 |
| topology_native / cylinder | 128 | 2.344E-02 | 3.224E-06 | 3.525 | 3.884E-06 | 4.122 | 1.276E-06 | 4.490 | NaN | NaN | 20 | 7.223E-11 | 1.175E-12 | 11 | 1152→1151 | 是 | 否 |
| topology_native / l_prism | 32 | 9.375E-02 | 1.519E-05 | NaN | 1.525E-05 | NaN | 3.638E-06 | NaN | NaN | NaN | 19 | 8.659E-11 | 3.452E-12 | 5 | 50→49 | 是 | 否 |
| topology_native / l_prism | 64 | 4.688E-02 | 5.250E-06 | 1.532 | 4.892E-06 | 1.640 | 4.791E-07 | 2.925 | NaN | NaN | 21 | 6.967E-11 | 2.983E-12 | 7 | 198→197 | 是 | 否 |
| topology_native / l_prism | 128 | 2.344E-02 | 8.719E-07 | 2.590 | 8.201E-07 | 2.577 | 3.710E-08 | 3.691 | NaN | NaN | 21 | 1.899E-10 | 4.417E-12 | 12 | 988→987 | 是 | 否 |
| topology_native / u_prism | 32 | 9.375E-02 | 4.905E-05 | NaN | 6.347E-05 | NaN | 2.524E-05 | NaN | NaN | NaN | 14 | 5.411E-11 | 1.722E-12 | 4 | 16→15 | 是 | 否 |
| topology_native / u_prism | 64 | 4.688E-02 | 1.073E-05 | 2.192 | 1.059E-05 | 2.584 | 7.067E-07 | 5.158 | NaN | NaN | 23 | 4.751E-11 | 1.037E-12 | 6 | 168→167 | 是 | 否 |
| topology_native / u_prism | 128 | 2.344E-02 | 1.721E-06 | 2.641 | 1.717E-06 | 2.624 | 1.642E-07 | 2.105 | NaN | NaN | 25 | 6.357E-11 | 7.925E-13 | 10 | 904→903 | 是 | 否 |

## N=32 全姿态逐行明细

每条 N=32 记录单独列出；`Dirichlet boundary_residual` 对 Neumann 或未提供该指标的 Dirichlet 记录显示为 `N/A`。

| 后端 / geometry | formulation | pose | interior_linf | density_linf | exterior_condition_linf | Dirichlet boundary_residual | GMRES it | relres | operator residual | DOF（pre→final） | algebraic | physical |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| topology_native / torus | dirichlet | baseline | 8.663E-03 | 1.050E-01 | 1.333E-02 | 1.404E-02 | 13 | 4.245E-11 | 3.764E-12 | 64→64 | 是 | 否 |
| topology_native / torus | dirichlet | tx_p0137 | 6.459E-03 | 9.407E-02 | 1.214E-02 | 1.042E-02 | 13 | 4.484E-11 | 4.108E-12 | 64→64 | 是 | 否 |
| topology_native / torus | dirichlet | ty_m0083 | 8.400E-03 | 1.024E-01 | 1.537E-02 | 1.340E-02 | 12 | 8.397E-11 | 8.984E-12 | 64→64 | 是 | 否 |
| topology_native / torus | dirichlet | tz_p0061 | 8.233E-03 | 1.059E-01 | 1.147E-02 | 1.400E-02 | 12 | 1.035E-10 | 7.808E-12 | 64→64 | 是 | 否 |
| topology_native / torus | dirichlet | t_xyz_1 | 6.596E-03 | 9.908E-02 | 1.328E-02 | 1.032E-02 | 12 | 1.112E-10 | 7.963E-12 | 64→64 | 是 | 否 |
| topology_native / torus | dirichlet | t_xyz_2 | 8.409E-03 | 1.041E-01 | 1.260E-02 | 1.259E-02 | 13 | 4.049E-11 | 3.165E-12 | 64→64 | 是 | 否 |
| topology_native / torus | dirichlet | rot_axis123_17deg | 9.171E-03 | 1.025E-01 | 2.269E-02 | 1.218E-02 | 13 | 5.898E-11 | 3.518E-12 | 64→64 | 是 | 否 |
| topology_native / torus | dirichlet | rot_axis123_17deg_t_xyz_1 | 7.862E-03 | 9.345E-02 | 1.995E-02 | 1.338E-02 | 12 | 1.845E-10 | 1.447E-11 | 64→64 | 是 | 否 |
| topology_native / torus | neumann | baseline | 2.560E-03 | 5.106E-03 | 1.642E-03 | N/A | 16 | 1.507E-10 | 1.775E-12 | 64→63 | 是 | 否 |
| topology_native / torus | neumann | tx_p0137 | 2.558E-03 | 4.430E-03 | 1.690E-03 | N/A | 17 | 1.227E-10 | 1.426E-12 | 64→63 | 是 | 否 |
| topology_native / torus | neumann | ty_m0083 | 2.116E-03 | 4.286E-03 | 1.592E-03 | N/A | 17 | 1.014E-10 | 1.309E-12 | 64→63 | 是 | 否 |
| topology_native / torus | neumann | tz_p0061 | 3.722E-03 | 4.981E-03 | 1.630E-03 | N/A | 17 | 1.219E-10 | 1.375E-12 | 64→63 | 是 | 否 |
| topology_native / torus | neumann | t_xyz_1 | 3.619E-03 | 4.766E-03 | 1.719E-03 | N/A | 17 | 1.912E-10 | 2.024E-12 | 64→63 | 是 | 否 |
| topology_native / torus | neumann | t_xyz_2 | 2.952E-03 | 5.151E-03 | 1.553E-03 | N/A | 18 | 3.365E-11 | 3.931E-13 | 64→63 | 是 | 否 |
| topology_native / torus | neumann | rot_axis123_17deg | 2.466E-03 | 4.314E-03 | 1.637E-03 | N/A | 17 | 1.547E-10 | 1.691E-12 | 64→63 | 是 | 否 |
| topology_native / torus | neumann | rot_axis123_17deg_t_xyz_1 | 2.894E-03 | 4.243E-03 | 1.786E-03 | N/A | 18 | 1.765E-11 | 2.458E-13 | 64→63 | 是 | 否 |
| topology_native / cylinder | dirichlet | baseline | 7.754E-03 | 7.131E-02 | 1.309E-02 | 1.042E-02 | 19 | 7.021E-11 | 7.695E-12 | 128→128 | 是 | 否 |
| topology_native / cylinder | dirichlet | tx_p0137 | 2.655E-03 | 4.955E-02 | 1.263E-02 | 4.745E-03 | 19 | 9.822E-11 | 7.380E-12 | 128→128 | 是 | 否 |
| topology_native / cylinder | dirichlet | ty_m0083 | 7.892E-03 | 7.376E-02 | 1.387E-02 | 1.046E-02 | 19 | 7.531E-11 | 8.006E-12 | 128→128 | 是 | 否 |
| topology_native / cylinder | dirichlet | tz_p0061 | 7.713E-03 | 7.239E-02 | 1.384E-02 | 1.046E-02 | 19 | 6.194E-11 | 4.763E-12 | 128→128 | 是 | 否 |
| topology_native / cylinder | dirichlet | t_xyz_1 | 3.639E-03 | 5.662E-02 | 1.322E-02 | 5.949E-03 | 19 | 9.034E-11 | 8.271E-12 | 128→128 | 是 | 否 |
| topology_native / cylinder | dirichlet | t_xyz_2 | 3.236E-03 | 4.367E-02 | 1.408E-02 | 5.275E-03 | 19 | 1.508E-10 | 9.709E-12 | 128→128 | 是 | 否 |
| topology_native / cylinder | dirichlet | rot_axis123_17deg | 4.747E-03 | 5.413E-02 | 1.525E-02 | 6.433E-03 | 19 | 1.128E-10 | 8.073E-12 | 128→128 | 是 | 否 |
| topology_native / cylinder | dirichlet | rot_axis123_17deg_t_xyz_1 | 4.306E-03 | 4.933E-02 | 1.659E-02 | 5.770E-03 | 19 | 9.189E-11 | 6.179E-12 | 128→128 | 是 | 否 |
| topology_native / cylinder | neumann | baseline | 1.568E-03 | 2.327E-03 | 1.282E-03 | N/A | 16 | 1.704E-10 | 4.663E-12 | 32→31 | 是 | 否 |
| topology_native / cylinder | neumann | tx_p0137 | 1.825E-03 | 2.561E-03 | 1.327E-03 | N/A | 17 | 3.927E-11 | 1.122E-12 | 32→31 | 是 | 否 |
| topology_native / cylinder | neumann | ty_m0083 | 1.597E-03 | 2.410E-03 | 1.314E-03 | N/A | 17 | 6.896E-11 | 2.381E-12 | 32→31 | 是 | 否 |
| topology_native / cylinder | neumann | tz_p0061 | 1.623E-03 | 2.343E-03 | 1.283E-03 | N/A | 16 | 1.700E-10 | 6.003E-12 | 32→31 | 是 | 否 |
| topology_native / cylinder | neumann | t_xyz_1 | 2.045E-03 | 2.648E-03 | 1.262E-03 | N/A | 17 | 4.720E-11 | 1.756E-12 | 32→31 | 是 | 否 |
| topology_native / cylinder | neumann | t_xyz_2 | 1.743E-03 | 2.505E-03 | 1.282E-03 | N/A | 17 | 3.979E-11 | 1.244E-12 | 32→31 | 是 | 否 |
| topology_native / cylinder | neumann | rot_axis123_17deg | 1.797E-03 | 2.662E-03 | 1.356E-03 | N/A | 17 | 6.314E-11 | 1.865E-12 | 32→31 | 是 | 否 |
| topology_native / cylinder | neumann | rot_axis123_17deg_t_xyz_1 | 1.429E-03 | 2.620E-03 | 1.317E-03 | N/A | 17 | 8.145E-11 | 1.905E-12 | 32→31 | 是 | 否 |
| topology_native / l_prism | dirichlet | baseline | 4.662E-04 | 5.893E-03 | 1.575E-04 | 4.949E-04 | 18 | 1.022E-10 | 6.068E-12 | 260→260 | 是 | 否 |
| topology_native / l_prism | dirichlet | tx_p0137 | 4.363E-04 | 6.007E-03 | 1.792E-04 | 5.435E-04 | 18 | 6.954E-11 | 5.988E-12 | 260→260 | 是 | 否 |
| topology_native / l_prism | dirichlet | ty_m0083 | 4.647E-04 | 5.877E-03 | 1.530E-04 | 4.936E-04 | 19 | 6.466E-11 | 3.923E-12 | 260→260 | 是 | 否 |
| topology_native / l_prism | dirichlet | tz_p0061 | 4.661E-04 | 5.858E-03 | 1.590E-04 | 4.949E-04 | 18 | 8.353E-11 | 7.604E-12 | 260→260 | 是 | 否 |
| topology_native / l_prism | dirichlet | t_xyz_1 | 4.397E-04 | 5.879E-03 | 1.678E-04 | 5.432E-04 | 18 | 9.659E-11 | 6.223E-12 | 260→260 | 是 | 否 |
| topology_native / l_prism | dirichlet | t_xyz_2 | 4.598E-04 | 5.263E-03 | 1.567E-04 | 5.596E-04 | 18 | 1.525E-10 | 1.203E-11 | 260→260 | 是 | 否 |
| topology_native / l_prism | dirichlet | rot_axis123_17deg | 4.559E-04 | 5.730E-03 | 3.374E-04 | 4.962E-04 | 19 | 6.755E-11 | 6.084E-12 | 260→260 | 是 | 否 |
| topology_native / l_prism | dirichlet | rot_axis123_17deg_t_xyz_1 | 4.692E-04 | 5.535E-03 | 2.504E-04 | 4.867E-04 | 18 | 6.991E-11 | 5.987E-12 | 260→260 | 是 | 否 |
| topology_native / l_prism | neumann | baseline | 2.434E-05 | 2.365E-05 | 4.069E-06 | N/A | 19 | 6.377E-11 | 1.934E-12 | 50→49 | 是 | 否 |
| topology_native / l_prism | neumann | tx_p0137 | 1.519E-05 | 1.525E-05 | 3.638E-06 | N/A | 19 | 8.659E-11 | 3.452E-12 | 50→49 | 是 | 否 |
| topology_native / l_prism | neumann | ty_m0083 | 2.413E-05 | 2.343E-05 | 4.391E-06 | N/A | 19 | 7.695E-11 | 2.919E-12 | 50→49 | 是 | 否 |
| topology_native / l_prism | neumann | tz_p0061 | 2.474E-05 | 2.417E-05 | 4.192E-06 | N/A | 19 | 6.803E-11 | 2.234E-12 | 50→49 | 是 | 否 |
| topology_native / l_prism | neumann | t_xyz_1 | 1.518E-05 | 1.539E-05 | 3.666E-06 | N/A | 19 | 9.333E-11 | 3.875E-12 | 50→49 | 是 | 否 |
| topology_native / l_prism | neumann | t_xyz_2 | 3.554E-05 | 3.808E-05 | 4.340E-06 | N/A | 19 | 7.023E-11 | 2.228E-12 | 50→49 | 是 | 否 |
| topology_native / l_prism | neumann | rot_axis123_17deg | 2.146E-05 | 2.195E-05 | 3.743E-06 | N/A | 19 | 5.498E-11 | 2.357E-12 | 50→49 | 是 | 否 |
| topology_native / l_prism | neumann | rot_axis123_17deg_t_xyz_1 | 2.037E-05 | 2.064E-05 | 3.586E-06 | N/A | 19 | 5.776E-11 | 1.957E-12 | 50→49 | 是 | 否 |
| topology_native / u_prism | dirichlet | baseline | 6.512E-04 | 5.640E-03 | 3.754E-04 | 7.247E-04 | 20 | 1.689E-10 | 1.094E-11 | 224→224 | 是 | 否 |
| topology_native / u_prism | dirichlet | tx_p0137 | 6.311E-04 | 5.373E-03 | 3.946E-04 | 7.388E-04 | 20 | 1.093E-10 | 6.342E-12 | 224→224 | 是 | 否 |
| topology_native / u_prism | dirichlet | ty_m0083 | 6.535E-04 | 5.585E-03 | 3.839E-04 | 7.288E-04 | 21 | 7.099E-11 | 4.078E-12 | 224→224 | 是 | 否 |
| topology_native / u_prism | dirichlet | tz_p0061 | 6.529E-04 | 5.649E-03 | 3.749E-04 | 7.263E-04 | 20 | 1.501E-10 | 1.162E-11 | 224→224 | 是 | 否 |
| topology_native / u_prism | dirichlet | t_xyz_1 | 6.353E-04 | 5.101E-03 | 3.719E-04 | 7.489E-04 | 20 | 9.388E-11 | 1.016E-11 | 224→224 | 是 | 否 |
| topology_native / u_prism | dirichlet | t_xyz_2 | 5.588E-04 | 5.583E-03 | 2.947E-04 | 6.579E-04 | 20 | 6.289E-11 | 4.079E-12 | 224→224 | 是 | 否 |
| topology_native / u_prism | dirichlet | rot_axis123_17deg | 5.800E-04 | 5.406E-03 | 4.992E-04 | 6.121E-04 | 19 | 1.006E-10 | 9.605E-12 | 224→224 | 是 | 否 |
| topology_native / u_prism | dirichlet | rot_axis123_17deg_t_xyz_1 | 5.880E-04 | 5.287E-03 | 4.235E-04 | 5.898E-04 | 19 | 1.039E-10 | 7.057E-12 | 224→224 | 是 | 否 |
| topology_native / u_prism | neumann | baseline | 5.327E-05 | 5.741E-05 | 3.000E-05 | N/A | 15 | 1.021E-15 | 3.123E-17 | 16→15 | 是 | 否 |
| topology_native / u_prism | neumann | tx_p0137 | 4.905E-05 | 6.347E-05 | 2.524E-05 | N/A | 14 | 5.411E-11 | 1.722E-12 | 16→15 | 是 | 否 |
| topology_native / u_prism | neumann | ty_m0083 | 5.483E-05 | 5.819E-05 | 2.990E-05 | N/A | 14 | 1.533E-10 | 4.626E-12 | 16→15 | 是 | 否 |
| topology_native / u_prism | neumann | tz_p0061 | 5.522E-05 | 5.891E-05 | 2.995E-05 | N/A | 15 | 5.484E-16 | 1.865E-17 | 16→15 | 是 | 否 |
| topology_native / u_prism | neumann | t_xyz_1 | 4.992E-05 | 6.341E-05 | 2.543E-05 | N/A | 15 | 1.210E-15 | 4.077E-17 | 16→15 | 是 | 否 |
| topology_native / u_prism | neumann | t_xyz_2 | 4.070E-05 | 5.699E-05 | 2.801E-05 | N/A | 14 | 1.181E-10 | 4.703E-12 | 16→15 | 是 | 否 |
| topology_native / u_prism | neumann | rot_axis123_17deg | 4.749E-05 | 6.001E-05 | 2.873E-05 | N/A | 15 | 6.990E-16 | 1.735E-17 | 16→15 | 是 | 否 |
| topology_native / u_prism | neumann | rot_axis123_17deg_t_xyz_1 | 3.929E-05 | 5.982E-05 | 2.803E-05 | N/A | 15 | 1.300E-15 | 5.551E-17 | 16→15 | 是 | 否 |
| general_cap / sphere | dirichlet | baseline | 2.079E-04 | 4.563E-03 | 2.231E-03 | N/A | 10 | 3.053E-11 | 2.051E-11 | 184→184 | 是 | 未提供 |
| general_cap / sphere | dirichlet | tx_p0137 | 2.225E-04 | 4.624E-03 | 2.233E-03 | N/A | 10 | 3.080E-11 | 2.852E-11 | 184→184 | 是 | 未提供 |
| general_cap / sphere | dirichlet | ty_m0083 | 2.020E-04 | 4.576E-03 | 2.231E-03 | N/A | 10 | 1.156E-10 | 7.860E-11 | 184→184 | 是 | 未提供 |
| general_cap / sphere | dirichlet | tz_p0061 | 2.353E-04 | 4.553E-03 | 2.226E-03 | N/A | 10 | 9.456E-11 | 7.917E-11 | 184→184 | 是 | 未提供 |
| general_cap / sphere | dirichlet | t_xyz_1 | 2.301E-04 | 4.613E-03 | 2.235E-03 | N/A | 11 | 4.213E-11 | 4.117E-11 | 184→184 | 是 | 未提供 |
| general_cap / sphere | dirichlet | t_xyz_2 | 2.008E-04 | 4.542E-03 | 2.287E-03 | N/A | 11 | 4.919E-11 | 5.674E-11 | 184→184 | 是 | 未提供 |
| general_cap / sphere | dirichlet | rot_axis123_17deg | 2.331E-04 | 4.575E-03 | 2.270E-03 | N/A | 10 | 1.010E-10 | 1.077E-10 | 184→184 | 是 | 未提供 |
| general_cap / sphere | dirichlet | rot_axis123_17deg_t_xyz_1 | 2.130E-04 | 4.562E-03 | 2.296E-03 | N/A | 10 | 1.728E-10 | 2.065E-10 | 184→184 | 是 | 未提供 |
| general_cap / sphere | neumann | baseline | 1.818E-03 | 3.148E-03 | 1.427E-03 | N/A | 9 | 1.438E-10 | 6.729E-12 | 184→183 | 是 | 未提供 |
| general_cap / sphere | neumann | tx_p0137 | 1.639E-03 | 3.101E-03 | 1.744E-03 | N/A | 9 | 1.379E-10 | 7.563E-12 | 184→183 | 是 | 未提供 |
| general_cap / sphere | neumann | ty_m0083 | 1.671E-03 | 3.345E-03 | 1.371E-03 | N/A | 10 | 8.476E-11 | 3.870E-12 | 184→183 | 是 | 未提供 |
| general_cap / sphere | neumann | tz_p0061 | 1.838E-03 | 3.165E-03 | 1.436E-03 | N/A | 10 | 1.046E-10 | 4.142E-12 | 184→183 | 是 | 未提供 |
| general_cap / sphere | neumann | t_xyz_1 | 1.696E-03 | 3.101E-03 | 1.750E-03 | N/A | 12 | 1.863E-11 | 8.638E-13 | 184→183 | 是 | 未提供 |
| general_cap / sphere | neumann | t_xyz_2 | 1.705E-03 | 3.600E-03 | 1.543E-03 | N/A | 11 | 1.130E-10 | 4.436E-12 | 184→183 | 是 | 未提供 |
| general_cap / sphere | neumann | rot_axis123_17deg | 1.644E-03 | 3.533E-03 | 1.362E-03 | N/A | 10 | 1.661E-10 | 9.186E-12 | 184→183 | 是 | 未提供 |
| general_cap / sphere | neumann | rot_axis123_17deg_t_xyz_1 | 1.605E-03 | 3.332E-03 | 1.555E-03 | N/A | 11 | 1.378E-10 | 7.206E-12 | 184→183 | 是 | 未提供 |
| general_cap / ellipsoid | dirichlet | baseline | 4.281E-04 | 1.208E-02 | 7.784E-03 | N/A | 10 | 1.035E-10 | 6.303E-11 | 184→184 | 是 | 未提供 |
| general_cap / ellipsoid | dirichlet | tx_p0137 | 5.104E-04 | 1.210E-02 | 7.746E-03 | N/A | 11 | 2.666E-11 | 1.723E-11 | 184→184 | 是 | 未提供 |
| general_cap / ellipsoid | dirichlet | ty_m0083 | 4.470E-04 | 1.207E-02 | 7.789E-03 | N/A | 11 | 5.634E-11 | 4.093E-11 | 184→184 | 是 | 未提供 |
| general_cap / ellipsoid | dirichlet | tz_p0061 | 3.685E-04 | 1.201E-02 | 7.790E-03 | N/A | 10 | 1.351E-10 | 1.093E-10 | 184→184 | 是 | 未提供 |
| general_cap / ellipsoid | dirichlet | t_xyz_1 | 4.985E-04 | 1.203E-02 | 7.744E-03 | N/A | 11 | 1.016E-10 | 7.940E-11 | 184→184 | 是 | 未提供 |
| general_cap / ellipsoid | dirichlet | t_xyz_2 | 4.624E-04 | 1.206E-02 | 7.720E-03 | N/A | 11 | 6.492E-11 | 4.173E-11 | 184→184 | 是 | 未提供 |
| general_cap / ellipsoid | dirichlet | rot_axis123_17deg | 6.880E-04 | 1.220E-02 | 7.623E-03 | N/A | 11 | 1.026E-10 | 1.450E-10 | 184→184 | 是 | 未提供 |
| general_cap / ellipsoid | dirichlet | rot_axis123_17deg_t_xyz_1 | 4.419E-04 | 1.209E-02 | 7.740E-03 | N/A | 12 | 3.523E-11 | 3.705E-11 | 184→184 | 是 | 未提供 |
| general_cap / ellipsoid | neumann | baseline | 1.753E-03 | 3.661E-03 | 1.844E-03 | N/A | 10 | 1.154E-10 | 3.839E-12 | 184→183 | 是 | 未提供 |
| general_cap / ellipsoid | neumann | tx_p0137 | 2.592E-03 | 4.116E-03 | 1.551E-03 | N/A | 11 | 4.527E-11 | 1.412E-12 | 184→183 | 是 | 未提供 |
| general_cap / ellipsoid | neumann | ty_m0083 | 1.794E-03 | 3.694E-03 | 1.926E-03 | N/A | 12 | 5.077E-11 | 1.777E-12 | 184→183 | 是 | 未提供 |
| general_cap / ellipsoid | neumann | tz_p0061 | 2.414E-03 | 3.959E-03 | 1.853E-03 | N/A | 10 | 8.998E-11 | 2.628E-12 | 184→183 | 是 | 未提供 |
| general_cap / ellipsoid | neumann | t_xyz_1 | 2.642E-03 | 4.190E-03 | 1.786E-03 | N/A | 12 | 4.908E-11 | 1.904E-12 | 184→183 | 是 | 未提供 |
| general_cap / ellipsoid | neumann | t_xyz_2 | 2.239E-03 | 4.421E-03 | 1.841E-03 | N/A | 12 | 2.849E-11 | 9.788E-13 | 184→183 | 是 | 未提供 |
| general_cap / ellipsoid | neumann | rot_axis123_17deg | 2.423E-03 | 3.897E-03 | 1.890E-03 | N/A | 12 | 5.921E-11 | 2.105E-12 | 184→183 | 是 | 未提供 |
| general_cap / ellipsoid | neumann | rot_axis123_17deg_t_xyz_1 | 2.437E-03 | 4.032E-03 | 1.823E-03 | N/A | 12 | 1.259E-10 | 5.046E-12 | 184→183 | 是 | 未提供 |
| general_cap / flower | dirichlet | baseline | 3.644E-03 | 7.726E-02 | 4.137E-02 | N/A | 12 | 4.622E-11 | 3.276E-11 | 184→184 | 是 | 未提供 |
| general_cap / flower | dirichlet | tx_p0137 | 3.256E-03 | 7.658E-02 | 4.136E-02 | N/A | 12 | 4.481E-11 | 3.000E-11 | 184→184 | 是 | 未提供 |
| general_cap / flower | dirichlet | ty_m0083 | 3.810E-03 | 7.810E-02 | 4.139E-02 | N/A | 12 | 1.609E-10 | 1.237E-10 | 184→184 | 是 | 未提供 |
| general_cap / flower | dirichlet | tz_p0061 | 3.641E-03 | 7.728E-02 | 4.139E-02 | N/A | 12 | 1.044E-10 | 6.886E-11 | 184→184 | 是 | 未提供 |
| general_cap / flower | dirichlet | t_xyz_1 | 4.670E-03 | 7.703E-02 | 4.154E-02 | N/A | 12 | 1.711E-10 | 1.112E-10 | 184→184 | 是 | 未提供 |
| general_cap / flower | dirichlet | t_xyz_2 | 3.936E-03 | 7.847E-02 | 4.178E-02 | N/A | 13 | 4.188E-11 | 2.323E-11 | 184→184 | 是 | 未提供 |
| general_cap / flower | dirichlet | rot_axis123_17deg | 4.103E-03 | 7.873E-02 | 4.084E-02 | N/A | 12 | 1.785E-10 | 2.903E-10 | 184→184 | 是 | 未提供 |
| general_cap / flower | dirichlet | rot_axis123_17deg_t_xyz_1 | 3.877E-03 | 7.918E-02 | 4.172E-02 | N/A | 13 | 2.929E-11 | 2.969E-11 | 184→184 | 是 | 未提供 |
| general_cap / flower | neumann | baseline | 8.505E-03 | 2.069E-02 | 1.065E-02 | N/A | 12 | 1.866E-11 | 1.059E-12 | 184→183 | 是 | 未提供 |
| general_cap / flower | neumann | tx_p0137 | 1.128E-02 | 2.592E-02 | 9.004E-03 | N/A | 12 | 2.880E-11 | 1.421E-12 | 184→183 | 是 | 未提供 |
| general_cap / flower | neumann | ty_m0083 | 8.376E-03 | 2.142E-02 | 1.125E-02 | N/A | 13 | 3.339E-11 | 1.715E-12 | 184→183 | 是 | 未提供 |
| general_cap / flower | neumann | tz_p0061 | 8.232E-03 | 2.070E-02 | 1.065E-02 | N/A | 13 | 1.176E-10 | 5.907E-12 | 184→183 | 是 | 未提供 |
| general_cap / flower | neumann | t_xyz_1 | 1.159E-02 | 2.704E-02 | 9.051E-03 | N/A | 14 | 1.003E-10 | 4.884E-12 | 184→183 | 是 | 未提供 |
| general_cap / flower | neumann | t_xyz_2 | 7.684E-03 | 1.983E-02 | 1.246E-02 | N/A | 14 | 1.380E-10 | 8.152E-12 | 184→183 | 是 | 未提供 |
| general_cap / flower | neumann | rot_axis123_17deg | 1.092E-02 | 2.030E-02 | 1.255E-02 | N/A | 14 | 1.477E-10 | 7.029E-12 | 184→183 | 是 | 未提供 |
| general_cap / flower | neumann | rot_axis123_17deg_t_xyz_1 | 1.167E-02 | 2.107E-02 | 1.066E-02 | N/A | 15 | 3.283E-11 | 1.636E-12 | 184→183 | 是 | 未提供 |

## N=32 八姿态刚体变换矩阵

每个单元为 `Eint (GMRES it)`；上标 `†` 表示 `gmres_converged=False`，破折号表示缺失记录。

| 简写 | 完整姿态 ID |
|---|---|
| base | `baseline` |
| Tx | `tx_p0137` |
| Ty | `ty_m0083` |
| Tz | `tz_p0061` |
| T1 | `t_xyz_1` |
| T2 | `t_xyz_2` |
| R17 | `rot_axis123_17deg` |
| R17+T1 | `rot_axis123_17deg_t_xyz_1` |

### Dirichlet：Eint（GMRES it）

| 后端 / 几何 | base | Tx | Ty | Tz | T1 | T2 | R17 | R17+T1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| topology_native / torus | 8.663E-03 (13) | 6.459E-03 (13) | 8.400E-03 (12) | 8.233E-03 (12) | 6.596E-03 (12) | 8.409E-03 (13) | 9.171E-03 (13) | 7.862E-03 (12) |
| topology_native / cylinder | 7.754E-03 (19) | 2.655E-03 (19) | 7.892E-03 (19) | 7.713E-03 (19) | 3.639E-03 (19) | 3.236E-03 (19) | 4.747E-03 (19) | 4.306E-03 (19) |
| topology_native / l_prism | 4.662E-04 (18) | 4.363E-04 (18) | 4.647E-04 (19) | 4.661E-04 (18) | 4.397E-04 (18) | 4.598E-04 (18) | 4.559E-04 (19) | 4.692E-04 (18) |
| topology_native / u_prism | 6.512E-04 (20) | 6.311E-04 (20) | 6.535E-04 (21) | 6.529E-04 (20) | 6.353E-04 (20) | 5.588E-04 (20) | 5.800E-04 (19) | 5.880E-04 (19) |
| general_cap / sphere | 2.079E-04 (10) | 2.225E-04 (10) | 2.020E-04 (10) | 2.353E-04 (10) | 2.301E-04 (11) | 2.008E-04 (11) | 2.331E-04 (10) | 2.130E-04 (10) |
| general_cap / ellipsoid | 4.281E-04 (10) | 5.104E-04 (11) | 4.470E-04 (11) | 3.685E-04 (10) | 4.985E-04 (11) | 4.624E-04 (11) | 6.880E-04 (11) | 4.419E-04 (12) |
| general_cap / flower | 3.644E-03 (12) | 3.256E-03 (12) | 3.810E-03 (12) | 3.641E-03 (12) | 4.670E-03 (12) | 3.936E-03 (13) | 4.103E-03 (12) | 3.877E-03 (13) |

### Neumann：Eint（GMRES it）

| 后端 / 几何 | base | Tx | Ty | Tz | T1 | T2 | R17 | R17+T1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| topology_native / torus | 2.560E-03 (16) | 2.558E-03 (17) | 2.116E-03 (17) | 3.722E-03 (17) | 3.619E-03 (17) | 2.952E-03 (18) | 2.466E-03 (17) | 2.894E-03 (18) |
| topology_native / cylinder | 1.568E-03 (16) | 1.825E-03 (17) | 1.597E-03 (17) | 1.623E-03 (16) | 2.045E-03 (17) | 1.743E-03 (17) | 1.797E-03 (17) | 1.429E-03 (17) |
| topology_native / l_prism | 2.434E-05 (19) | 1.519E-05 (19) | 2.413E-05 (19) | 2.474E-05 (19) | 1.518E-05 (19) | 3.554E-05 (19) | 2.146E-05 (19) | 2.037E-05 (19) |
| topology_native / u_prism | 5.327E-05 (15) | 4.905E-05 (14) | 5.483E-05 (14) | 5.522E-05 (15) | 4.992E-05 (15) | 4.070E-05 (14) | 4.749E-05 (15) | 3.929E-05 (15) |
| general_cap / sphere | 1.818E-03 (9) | 1.639E-03 (9) | 1.671E-03 (10) | 1.838E-03 (10) | 1.696E-03 (12) | 1.705E-03 (11) | 1.644E-03 (10) | 1.605E-03 (11) |
| general_cap / ellipsoid | 1.753E-03 (10) | 2.592E-03 (11) | 1.794E-03 (12) | 2.414E-03 (10) | 2.642E-03 (12) | 2.239E-03 (12) | 2.423E-03 (12) | 2.437E-03 (12) |
| general_cap / flower | 8.505E-03 (12) | 1.128E-02 (12) | 8.376E-03 (13) | 8.232E-03 (13) | 1.159E-02 (14) | 7.684E-03 (14) | 1.092E-02 (14) | 1.167E-02 (15) |

## 收敛状态统计

`gmres_converged` 是代数收敛的唯一统计来源；`physical_converged` 为空时记为“未提供”，不会被当作失败。

| 后端 | 方程 | 行数 | 代数：是 | 代数：否 | 代数：未提供 | physical：是 | physical：否 | physical：未提供 | GMRES it 范围 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| general_cap | dirichlet | 36 | 36 | 0 | 0 | 0 | 0 | 36 | 9–13 |
| general_cap | neumann | 36 | 36 | 0 | 0 | 0 | 0 | 36 | 8–15 |
| topology_native | dirichlet | 39 | 39 | 0 | 0 | 0 | 39 | 0 | 12–21 |
| topology_native | neumann | 39 | 39 | 0 | 0 | 0 | 39 | 0 | 14–25 |

## 覆盖完整性

目标矩阵为：`baseline` 含 N=32/64/128；`general_cap` 的收敛刚体姿态为 `R17+T1`，`topology_native` 的收敛刚体姿态为 `Tx`，各含 N=32/64/128；全部八个姿态均含 N=32。

| 后端 / 几何 | 方程 | baseline 层 | 收敛刚体姿态 | 收敛刚体层 | N=32 姿态 | 缺失组合数 |
|---|---|---|---|---|---:|---:|
| topology_native / torus | dirichlet | 32 | Tx (`tx_p0137`) | 32/64 | 8/8 | 3 |
| topology_native / torus | neumann | 32 | Tx (`tx_p0137`) | 32/64 | 8/8 | 3 |
| topology_native / cylinder | dirichlet | 32 | Tx (`tx_p0137`) | 32/64/128 | 8/8 | 2 |
| topology_native / cylinder | neumann | 32 | Tx (`tx_p0137`) | 32/64/128 | 8/8 | 2 |
| topology_native / l_prism | dirichlet | 32 | Tx (`tx_p0137`) | 32/64/128 | 8/8 | 2 |
| topology_native / l_prism | neumann | 32 | Tx (`tx_p0137`) | 32/64/128 | 8/8 | 2 |
| topology_native / u_prism | dirichlet | 32 | Tx (`tx_p0137`) | 32/64/128 | 8/8 | 2 |
| topology_native / u_prism | neumann | 32 | Tx (`tx_p0137`) | 32/64/128 | 8/8 | 2 |
| general_cap / sphere | dirichlet | 32/64/128 | R17+T1 (`rot_axis123_17deg_t_xyz_1`) | 32/64/128 | 8/8 | 0 |
| general_cap / sphere | neumann | 32/64/128 | R17+T1 (`rot_axis123_17deg_t_xyz_1`) | 32/64/128 | 8/8 | 0 |
| general_cap / ellipsoid | dirichlet | 32/64/128 | R17+T1 (`rot_axis123_17deg_t_xyz_1`) | 32/64/128 | 8/8 | 0 |
| general_cap / ellipsoid | neumann | 32/64/128 | R17+T1 (`rot_axis123_17deg_t_xyz_1`) | 32/64/128 | 8/8 | 0 |
| general_cap / flower | dirichlet | 32/64/128 | R17+T1 (`rot_axis123_17deg_t_xyz_1`) | 32/64/128 | 8/8 | 0 |
| general_cap / flower | neumann | 32/64/128 | R17+T1 (`rot_axis123_17deg_t_xyz_1`) | 32/64/128 | 8/8 | 0 |

## 失败、缺失值与 NaN

- `NaN` 阶通常表示该组的首个网格层，或相邻误差/网格尺度非有限、非正，因而不能合法计算对数阶。
- 空白（表中显示为“—”）表示源 CSV 没有给出该指标；这与数值 `NaN` 不同。general-cap 的 `physical_converged` 空值按“未提供”处理。
- `physical_converged=False` 是源程序的物理判据未通过，不等价于 GMRES 代数未收敛；两类状态分别列出。

| 指标列 | 有限 | NaN/Inf | 空白 | 无法解析 |
|---|---:|---:|---:|---:|
| `interior_linf` | 150 | 0 | 0 | 0 |
| `interior_order_linf` | 38 | 112 | 0 | 0 |
| `density_linf` | 150 | 0 | 0 | 0 |
| `density_order_linf` | 38 | 112 | 0 | 0 |
| `exterior_condition_linf` | 150 | 0 | 0 | 0 |
| `exterior_condition_order_linf` | 38 | 112 | 0 | 0 |
| `boundary_residual_linf` | 39 | 111 | 0 | 0 |
| `boundary_residual_order_linf` | 7 | 143 | 0 | 0 |
| `gmres_relative_residual` | 150 | 0 | 0 | 0 |
| `operator_residual_linf` | 150 | 0 | 0 | 0 |

### 明确未通过的记录

| case_id | 后端 / 几何 | 方程 | 姿态 | N | 代数 | physical | it | relres | op-res | Eint |
|---|---|---|---|---:|:---:|:---:|---:|---:|---:|---:|
| b_tt | topology_native / torus | dirichlet | baseline | 32 | 是 | 否 | 13 | 4.245E-11 | 3.764E-12 | 8.663E-03 |
| x_tt | topology_native / torus | dirichlet | tx_p0137 | 32 | 是 | 否 | 13 | 4.484E-11 | 4.108E-12 | 6.459E-03 |
| x_tt | topology_native / torus | dirichlet | tx_p0137 | 64 | 是 | 否 | 12 | 1.546E-10 | 8.786E-12 | 1.457E-03 |
| y_tt | topology_native / torus | dirichlet | ty_m0083 | 32 | 是 | 否 | 12 | 8.397E-11 | 8.984E-12 | 8.400E-03 |
| z_tt | topology_native / torus | dirichlet | tz_p0061 | 32 | 是 | 否 | 12 | 1.035E-10 | 7.808E-12 | 8.233E-03 |
| 1_tt | topology_native / torus | dirichlet | t_xyz_1 | 32 | 是 | 否 | 12 | 1.112E-10 | 7.963E-12 | 6.596E-03 |
| 2_tt | topology_native / torus | dirichlet | t_xyz_2 | 32 | 是 | 否 | 13 | 4.049E-11 | 3.165E-12 | 8.409E-03 |
| r_tt | topology_native / torus | dirichlet | rot_axis123_17deg | 32 | 是 | 否 | 13 | 5.898E-11 | 3.518E-12 | 9.171E-03 |
| q_tt | topology_native / torus | dirichlet | rot_axis123_17deg_t_xyz_1 | 32 | 是 | 否 | 12 | 1.845E-10 | 1.447E-11 | 7.862E-03 |
| b_tt | topology_native / torus | neumann | baseline | 32 | 是 | 否 | 16 | 1.507E-10 | 1.775E-12 | 2.560E-03 |
| x_tt | topology_native / torus | neumann | tx_p0137 | 32 | 是 | 否 | 17 | 1.227E-10 | 1.426E-12 | 2.558E-03 |
| x_tt | topology_native / torus | neumann | tx_p0137 | 64 | 是 | 否 | 17 | 8.536E-11 | 6.533E-13 | 1.161E-04 |
| y_tt | topology_native / torus | neumann | ty_m0083 | 32 | 是 | 否 | 17 | 1.014E-10 | 1.309E-12 | 2.116E-03 |
| z_tt | topology_native / torus | neumann | tz_p0061 | 32 | 是 | 否 | 17 | 1.219E-10 | 1.375E-12 | 3.722E-03 |
| 1_tt | topology_native / torus | neumann | t_xyz_1 | 32 | 是 | 否 | 17 | 1.912E-10 | 2.024E-12 | 3.619E-03 |
| 2_tt | topology_native / torus | neumann | t_xyz_2 | 32 | 是 | 否 | 18 | 3.365E-11 | 3.931E-13 | 2.952E-03 |
| r_tt | topology_native / torus | neumann | rot_axis123_17deg | 32 | 是 | 否 | 17 | 1.547E-10 | 1.691E-12 | 2.466E-03 |
| q_tt | topology_native / torus | neumann | rot_axis123_17deg_t_xyz_1 | 32 | 是 | 否 | 18 | 1.765E-11 | 2.458E-13 | 2.894E-03 |
| b_tn | topology_native / cylinder | dirichlet | baseline | 32 | 是 | 否 | 19 | 7.021E-11 | 7.695E-12 | 7.754E-03 |
| x_tn | topology_native / cylinder | dirichlet | tx_p0137 | 32 | 是 | 否 | 19 | 9.822E-11 | 7.380E-12 | 2.655E-03 |
| x_tn | topology_native / cylinder | dirichlet | tx_p0137 | 64 | 是 | 否 | 20 | 6.962E-11 | 4.059E-12 | 8.904E-04 |
| x_tn | topology_native / cylinder | dirichlet | tx_p0137 | 128 | 是 | 否 | 21 | 5.473E-11 | 1.997E-12 | 3.370E-04 |
| y_tn | topology_native / cylinder | dirichlet | ty_m0083 | 32 | 是 | 否 | 19 | 7.531E-11 | 8.006E-12 | 7.892E-03 |
| z_tn | topology_native / cylinder | dirichlet | tz_p0061 | 32 | 是 | 否 | 19 | 6.194E-11 | 4.763E-12 | 7.713E-03 |
| 1_tn | topology_native / cylinder | dirichlet | t_xyz_1 | 32 | 是 | 否 | 19 | 9.034E-11 | 8.271E-12 | 3.639E-03 |
| 2_tn | topology_native / cylinder | dirichlet | t_xyz_2 | 32 | 是 | 否 | 19 | 1.508E-10 | 9.709E-12 | 3.236E-03 |
| r_tn | topology_native / cylinder | dirichlet | rot_axis123_17deg | 32 | 是 | 否 | 19 | 1.128E-10 | 8.073E-12 | 4.747E-03 |
| q_tn | topology_native / cylinder | dirichlet | rot_axis123_17deg_t_xyz_1 | 32 | 是 | 否 | 19 | 9.189E-11 | 6.179E-12 | 4.306E-03 |
| b_tn | topology_native / cylinder | neumann | baseline | 32 | 是 | 否 | 16 | 1.704E-10 | 4.663E-12 | 1.568E-03 |
| x_tn | topology_native / cylinder | neumann | tx_p0137 | 32 | 是 | 否 | 17 | 3.927E-11 | 1.122E-12 | 1.825E-03 |
| x_tn | topology_native / cylinder | neumann | tx_p0137 | 64 | 是 | 否 | 18 | 1.269E-10 | 2.165E-12 | 3.711E-05 |
| x_tn | topology_native / cylinder | neumann | tx_p0137 | 128 | 是 | 否 | 20 | 7.223E-11 | 1.175E-12 | 3.224E-06 |
| y_tn | topology_native / cylinder | neumann | ty_m0083 | 32 | 是 | 否 | 17 | 6.896E-11 | 2.381E-12 | 1.597E-03 |
| z_tn | topology_native / cylinder | neumann | tz_p0061 | 32 | 是 | 否 | 16 | 1.700E-10 | 6.003E-12 | 1.623E-03 |
| 1_tn | topology_native / cylinder | neumann | t_xyz_1 | 32 | 是 | 否 | 17 | 4.720E-11 | 1.756E-12 | 2.045E-03 |
| 2_tn | topology_native / cylinder | neumann | t_xyz_2 | 32 | 是 | 否 | 17 | 3.979E-11 | 1.244E-12 | 1.743E-03 |
| r_tn | topology_native / cylinder | neumann | rot_axis123_17deg | 32 | 是 | 否 | 17 | 6.314E-11 | 1.865E-12 | 1.797E-03 |
| q_tn | topology_native / cylinder | neumann | rot_axis123_17deg_t_xyz_1 | 32 | 是 | 否 | 17 | 8.145E-11 | 1.905E-12 | 1.429E-03 |
| b_tn | topology_native / l_prism | dirichlet | baseline | 32 | 是 | 否 | 18 | 1.022E-10 | 6.068E-12 | 4.662E-04 |
| x_tn | topology_native / l_prism | dirichlet | tx_p0137 | 32 | 是 | 否 | 18 | 6.954E-11 | 5.988E-12 | 4.363E-04 |
| x_tn | topology_native / l_prism | dirichlet | tx_p0137 | 64 | 是 | 否 | 20 | 5.802E-11 | 5.249E-12 | 2.290E-04 |
| x_tn | topology_native / l_prism | dirichlet | tx_p0137 | 128 | 是 | 否 | 19 | 1.549E-10 | 7.236E-12 | 1.171E-04 |
| y_tn | topology_native / l_prism | dirichlet | ty_m0083 | 32 | 是 | 否 | 19 | 6.466E-11 | 3.923E-12 | 4.647E-04 |
| z_tn | topology_native / l_prism | dirichlet | tz_p0061 | 32 | 是 | 否 | 18 | 8.353E-11 | 7.604E-12 | 4.661E-04 |
| 1_tn | topology_native / l_prism | dirichlet | t_xyz_1 | 32 | 是 | 否 | 18 | 9.659E-11 | 6.223E-12 | 4.397E-04 |
| 2_tn | topology_native / l_prism | dirichlet | t_xyz_2 | 32 | 是 | 否 | 18 | 1.525E-10 | 1.203E-11 | 4.598E-04 |
| r_tn | topology_native / l_prism | dirichlet | rot_axis123_17deg | 32 | 是 | 否 | 19 | 6.755E-11 | 6.084E-12 | 4.559E-04 |
| q_tn | topology_native / l_prism | dirichlet | rot_axis123_17deg_t_xyz_1 | 32 | 是 | 否 | 18 | 6.991E-11 | 5.987E-12 | 4.692E-04 |
| b_tn | topology_native / l_prism | neumann | baseline | 32 | 是 | 否 | 19 | 6.377E-11 | 1.934E-12 | 2.434E-05 |
| x_tn | topology_native / l_prism | neumann | tx_p0137 | 32 | 是 | 否 | 19 | 8.659E-11 | 3.452E-12 | 1.519E-05 |
| x_tn | topology_native / l_prism | neumann | tx_p0137 | 64 | 是 | 否 | 21 | 6.967E-11 | 2.983E-12 | 5.250E-06 |
| x_tn | topology_native / l_prism | neumann | tx_p0137 | 128 | 是 | 否 | 21 | 1.899E-10 | 4.417E-12 | 8.719E-07 |
| y_tn | topology_native / l_prism | neumann | ty_m0083 | 32 | 是 | 否 | 19 | 7.695E-11 | 2.919E-12 | 2.413E-05 |
| z_tn | topology_native / l_prism | neumann | tz_p0061 | 32 | 是 | 否 | 19 | 6.803E-11 | 2.234E-12 | 2.474E-05 |
| 1_tn | topology_native / l_prism | neumann | t_xyz_1 | 32 | 是 | 否 | 19 | 9.333E-11 | 3.875E-12 | 1.518E-05 |
| 2_tn | topology_native / l_prism | neumann | t_xyz_2 | 32 | 是 | 否 | 19 | 7.023E-11 | 2.228E-12 | 3.554E-05 |
| r_tn | topology_native / l_prism | neumann | rot_axis123_17deg | 32 | 是 | 否 | 19 | 5.498E-11 | 2.357E-12 | 2.146E-05 |
| q_tn | topology_native / l_prism | neumann | rot_axis123_17deg_t_xyz_1 | 32 | 是 | 否 | 19 | 5.776E-11 | 1.957E-12 | 2.037E-05 |
| b_tn | topology_native / u_prism | dirichlet | baseline | 32 | 是 | 否 | 20 | 1.689E-10 | 1.094E-11 | 6.512E-04 |
| x_tn | topology_native / u_prism | dirichlet | tx_p0137 | 32 | 是 | 否 | 20 | 1.093E-10 | 6.342E-12 | 6.311E-04 |
| x_tn | topology_native / u_prism | dirichlet | tx_p0137 | 64 | 是 | 否 | 20 | 1.986E-10 | 1.408E-11 | 3.587E-04 |
| x_tn | topology_native / u_prism | dirichlet | tx_p0137 | 128 | 是 | 否 | 20 | 1.333E-10 | 7.303E-12 | 1.408E-04 |
| y_tn | topology_native / u_prism | dirichlet | ty_m0083 | 32 | 是 | 否 | 21 | 7.099E-11 | 4.078E-12 | 6.535E-04 |
| z_tn | topology_native / u_prism | dirichlet | tz_p0061 | 32 | 是 | 否 | 20 | 1.501E-10 | 1.162E-11 | 6.529E-04 |
| 1_tn | topology_native / u_prism | dirichlet | t_xyz_1 | 32 | 是 | 否 | 20 | 9.388E-11 | 1.016E-11 | 6.353E-04 |
| 2_tn | topology_native / u_prism | dirichlet | t_xyz_2 | 32 | 是 | 否 | 20 | 6.289E-11 | 4.079E-12 | 5.588E-04 |
| r_tn | topology_native / u_prism | dirichlet | rot_axis123_17deg | 32 | 是 | 否 | 19 | 1.006E-10 | 9.605E-12 | 5.800E-04 |
| q_tn | topology_native / u_prism | dirichlet | rot_axis123_17deg_t_xyz_1 | 32 | 是 | 否 | 19 | 1.039E-10 | 7.057E-12 | 5.880E-04 |
| b_tn | topology_native / u_prism | neumann | baseline | 32 | 是 | 否 | 15 | 1.021E-15 | 3.123E-17 | 5.327E-05 |
| x_tn | topology_native / u_prism | neumann | tx_p0137 | 32 | 是 | 否 | 14 | 5.411E-11 | 1.722E-12 | 4.905E-05 |
| x_tn | topology_native / u_prism | neumann | tx_p0137 | 64 | 是 | 否 | 23 | 4.751E-11 | 1.037E-12 | 1.073E-05 |
| x_tn | topology_native / u_prism | neumann | tx_p0137 | 128 | 是 | 否 | 25 | 6.357E-11 | 7.925E-13 | 1.721E-06 |
| y_tn | topology_native / u_prism | neumann | ty_m0083 | 32 | 是 | 否 | 14 | 1.533E-10 | 4.626E-12 | 5.483E-05 |
| z_tn | topology_native / u_prism | neumann | tz_p0061 | 32 | 是 | 否 | 15 | 5.484E-16 | 1.865E-17 | 5.522E-05 |
| 1_tn | topology_native / u_prism | neumann | t_xyz_1 | 32 | 是 | 否 | 15 | 1.210E-15 | 4.077E-17 | 4.992E-05 |
| 2_tn | topology_native / u_prism | neumann | t_xyz_2 | 32 | 是 | 否 | 14 | 1.181E-10 | 4.703E-12 | 4.070E-05 |
| r_tn | topology_native / u_prism | neumann | rot_axis123_17deg | 32 | 是 | 否 | 15 | 6.990E-16 | 1.735E-17 | 4.749E-05 |
| q_tn | topology_native / u_prism | neumann | rot_axis123_17deg_t_xyz_1 | 32 | 是 | 否 | 15 | 1.300E-15 | 5.551E-17 | 3.929E-05 |

## 解释边界

该报告忠实汇总 CSV 已记录的数据。若某个误差、阶或 physical 状态未由后端输出，报告保留为 `NaN`、空白或“未提供”，不以零替代，也不据 GMRES 状态推断物理状态。
