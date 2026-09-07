# 原生 NURBS 端点路径认证：第一阶段实现说明

日期：2026-09-07。

状态：第一阶段代码已完成 Release 构建、15 项路径回归、5 组精确几何基础检查及 4 个既有 restrict 回归。**本轮没有运行完整 PDE，不提供新的 PDE 误差、收敛阶、GMRES 迭代数或整体加速比。** 默认生产路线仍为 legacy。

## 1. 本批修改解决什么问题

Q27-cover3 / Q64-cover4 的 restrict 需要查询从 Cartesian 支撑节点到曲面迹点的路径。旧路径通过已舍入的曲面点构造线段，再延长端点、求交、识别哪些根属于迹点；这一流程可能在端点附近出现根身份或完整性无法认证的问题。

本批增加显式可选的 `native_certified` 路线：曲面端点由 `(patch_index, u, v)` 定义，使用原始 NURBS 数据的精确有理求值确定端点，而不是把已经舍入的三维坐标重新当作曲面上的精确点。已知端点的存在性和普通交点的存在性采用不同证明。

第一阶段只替换这一类 **support-to-trace 几何查询及其事件适配**。它不是整个 spread/restrict 求交系统的全面替换，也不是新的 Dirichlet/Neumann 离散算法：

- 默认仍为 `legacy`，必须显式启用新路线。
- 新路线只接入 Q27/Q64 cover；不借此恢复或推广 Q10。
- 旧 `--restrict-probe` 诊断入口使用另一套 joint-tricubic 路线；与 `native_certified` 同时选择时会在几何 setup 前拒绝，不能用旧 probe 冒充新路径验证。
- Cartesian 网格边目录、spread 装配以及既有网格交点数据不由这个新内核重新实现。
- 不修改密度次数、结点分辨率、C0/C1/vertex 约束、仿射特解与零空间、最终 DOF 或 GMRES 容差。
- 不把“一个几何路径返回 Certified”写成“整体 PDE 二阶收敛已证明”。

## 2. 阅读入口与分工

| 文件 | 本批职责 |
| --- | --- |
| [native_nurbs_exact_geometry_3d.hpp](../src/geometry/native_nurbs_exact_geometry_3d.hpp)、[.cpp](../src/geometry/native_nurbs_exact_geometry_3d.cpp) | 原始 NURBS 的精确齐次 Bézier 提取、精确求值、保守 AABB |
| [native_endpoint_path_3d.hpp](../src/geometry/native_endpoint_path_3d.hpp)、[.cpp](../src/geometry/native_endpoint_path_3d.cpp) | 根证明、参数盒覆盖、排序、方向与代表精度认证；预算及失败信息 |
| [nurbs_surface_intersector_3d.hpp](../src/geometry/nurbs_surface_intersector_3d.hpp)、[.cpp](../src/geometry/nurbs_surface_intersector_3d.cpp) | `intersect_segment_to_native_endpoint()` 入口及每个 intersector 的懒初始化精确几何缓存 |
| [native_endpoint_path_adapter_3d.hpp](../apps/native_endpoint_path_adapter_3d.hpp)、[.cpp](../apps/native_endpoint_path_adapter_3d.cpp) | 将完整认证路径转换为现有 all-event correction 接口；按迹侧条件加入端点项 |
| [neumann_exterior_zero_trace_3d.cpp](../apps/neumann_exterior_zero_trace_3d.cpp) | 环境变量、输出隔离、`build_all_event_trace_corrections()` 接入和 setup 内路径缓存 |
| [native_nurbs_exact_geometry_3d_test.cpp](../apps/native_nurbs_exact_geometry_3d_test.cpp) | 精确提取基础层回归 |
| [native_endpoint_path_3d_test.cpp](../apps/native_endpoint_path_3d_test.cpp) | 有界几何路径与适配器回归，不求解 PDE |

推荐顺序是：先看公共结果类型，再看 `make_element()`、`known_endpoint_attempt()`、`ordinary_attempt()`、`root_relation()`、`run_pass()`，最后看适配器和 driver 接入。

## 3. 精确几何的含义

这里的“精确”指 **相对于当前模型中原始 IEEE binary64 控制点、权重、结点数据精确**；并不把模型替换为某个理想圆柱或未舍入的理想刚体变换。

对于原始控制点坐标 `x` 和权重 `w`，分别执行 `CGAL::Gmpq(x)`、`CGAL::Gmpq(w)`，再形成齐次坐标 `Q(x) * Q(w)`。不能先用 double 计算 `x*w`，再声称 `Q(x*w)` 是原始数据的精确乘积。

两个参数方向的结点插入都采用有理运算，包含结点差和插入系数的有理除法。每个 `ExactNativeBezierElement3D` 保存：patch/component、两个次数、原 patch 参数区间、按 u-major 排列的四维有理控制网。提取不改变参数方向和法向约定。

目前要求原始数据有限、权重严格为正、活动结点区间端点 clamped；支持原生基函数允许的非均匀内部结点及其重数。非 clamped 端点不能静默进入认证。

`evaluate_native_bezier_homogeneous()` 接收原 patch 参数，而不是自动归一化的局部参数；不裁剪越域输入。它是正权几何求值器，**不能拿来求值第四分量可能为零或负数的导数控制网**。内核的齐次导数另用多项式 de Casteljau 求值。

正权保证有理曲面处于欧氏控制点凸包内。AABB 由精确有理商的向外舍入区间构造，不以未扩张的普通 double 控制点求值作为认证边界。

## 4. 查询、方程和根证明

### 4.1 原生端点定义查询

设支撑节点为原始 double 向量 `P`，曲面端点为

\[
q = S_{\mathrm{patch}}(u_*,v_*),\qquad D=q-P,
\qquad x(t)=P+tD,\quad 0\leq t\leq1.
\]

`P` 的各分量精确转换为有理数；`q` 使用上述精确 NURBS 几何求值。因此 `t=1` 的端点身份是构造定义，不需要“离端点足够近”的容差识别。

精确端点的保守坐标区间与 `P` 构成查询 AABB，再与精确 Bézier 元素的保守 AABB 筛选候选。第一阶段是对缓存元素边界的遍历筛选，**没有新增一个完整的精确几何 BVH**。

### 4.2 将三维线面求交化为两个参数方程

设元素齐次曲面为 `H=(H_x,H_y,H_z,W)`。选择 `|D_k|` 最大的非零分量，另外两个方向记为 `l`，构造

\[
G_l(u,v)
=D_k\bigl(H_l-P_lW\bigr)
-D_l\bigl(H_k-P_kW\bigr)=0.
\]

两式系数由有理运算得到，再转为向外舍入的区间控制网。每个方程可独立作正比例尺度归一化，这不改变其零集。路径参数使用

\[
t=\frac{H_k-P_kW}{WD_k}
\]

的区间包围确定，不用数值残差近似代替参数范围证明。

### 4.3 区域排除和完整覆盖

待处理单元始于每个候选元素的完整参数盒。可以通过 Bernstein 控制网包围排除某个方程为零，或通过 `t` 区间证明根不在 `[0,1]`。有效 Krawczyk 包围也可提供无根排除。

不能排除或完成根证明时，沿参数方向细分。即使分割点是在 binary64 中选取，其局部比例也使用精确有理差与商转换为区间，保证控制网与新参数盒对应。

根区域只在证明覆盖**整个当前单元**时结束处理；剩余待处理盒不允许因为找到了一个“最近根”而被丢弃。所有候选区域处理完后才进入完整性门。

### 4.4 普通根：存在性与唯一性分别证明

普通 Newton 迭代只产生候选 `(u,v)`，不产生认证根。令 `X` 为候选附近参数盒，`C` 为中点 Jacobian 的数值逆，使用区间计算

\[
K(X)=x_0-CG(x_0)+(I-CJ(X))(X-x_0).
\]

需要有效的完整区间线性化，并证明 `K(X)` 严格位于 `X` 内及收缩界小于 1，才可得到存在且唯一的根。线性化中途失败，不能消费已经填写的部分误差矩阵来排除区域。

此外，内核对“当前单元与隔离盒的公共包围盒”证明至多一根。这一步把小盒内的存在性扩展为对整个当前单元的完整处理，防止仅保留候选附近的根而遗漏同一单元里的其它根。

普通根最后必须满足严格区间关系 `0 < t_lo <= t_hi < 1`。与端点或起点无法分离时，不强行归类。

### 4.5 已知端点：精确存在性加局部唯一性

端点对应参数已经给出，存在性由精确求值定义。若端点位于同 patch 的多个 Bézier 元素闭边界上，先精确验证各表示位置一致、法向正则且同向、端点非切向。

随后在包含当前单元和端点参数的公共盒上证明 `||I-CJ||_∞ < 1`，得到该公共盒至多一个根。结合端点的精确存在性，生成 `KnownEndpointUnique`，其 `t_interval` 严格为 `[1,1]`。

这与普通根 `ExistsUnique` 不同；两者都不同于仅有小残差的 `Candidate`。

### 4.6 合并、排序与方向

同根关系使用已知端点身份、唯一性盒对根包围盒的包含关系，或同元素公共盒的唯一性证明。两个 `t` 区间严格分离可证明是不同根。无法证明同一或不同，返回 `Unresolved`；不按 `abs(t_i-t_j)<tol` 合并。

合并后，相邻物理事件的 `t` 包围必须严格分离。`proof_id` 区分证明记录，`event_id` 在本次查询内区分规范化事件；它们不是跨全部网格边的全局事件编号。

事件方向来自曲面有向法向与路径方向的严格点积区间：负号为 entering `+1`，正号为 leaving `-1`。要求输入几何的参数法向遵守应用的外法向约定。期望迹侧不能改写方向；支撑节点的 inside 标签只用于核对由事件链推导出的起始状态。

其中 `oriented_dot_lower/upper` 使用与外法向同向的、未单位化的齐次导数叉积；其正比例尺度不影响符号证明，但不能将它的数值直接解释为单位法向夹角。单位法向及夹角诊断存于 representative。路径局部的 `native_event_id/native_proof_id` 经适配器保留到 correction；跨路径审计时必须同时携带 center/node 路径身份。

## 5. 四道认证条件与严格失败

`Certified` 要求同时满足：

1. 候选区域完整覆盖，没有未解决区域。
2. 不同物理根身份与严格顺序已证明。
3. 每个事件横截方向已证明，inside/outside 转移链一致。
4. 点、参数、法向代表值满足所要求的误差上界。

`select_native_endpoint_path_events_3d()` 不只检查状态枚举，还检查这四个标志、未解决区域计数、根证明类型、收缩界、有限区间、有限代表值、事件标识和符号一致性。它保留全部开区间交点，仅在目标迹侧与端点来向侧不同时加入 `t=1` 的端点跳跃项。

失败通过 `Unresolved`、`BudgetExceeded`、`UnsupportedEndpoint`、`DegenerateSegment` 或 `InvalidInput` 显式返回。应用层携带 center、grid node、stencil slot、迹侧、失败原因及查询 dump 抛出异常，停止 operator setup。**不会自动回退 legacy、忽略未认证根或把期望的 parity 写成几何证据。**

区间运算采用 binary64 向外舍入，并在配置了 MPFR 后允许升级至更高精度。NaN、无效区间和未完成的线性化不是排除证据。精度升级仍受逐路径工作预算约束，不是无限重试；没有 MPFR 后端时，不能把要求更高精度的查询标为已完成。

dump 用于重放与审计，包含查询数据、模型指纹、预算及证明诊断；模型指纹本身不是几何正确性的数学证明。重放还需要相应版本的原始模型。

## 6. 精度口径

最终代码对三维点和法向先计算分量最大误差，再用向外舍入的 `sqrt(3)` 系数转换为**欧氏范数上界**；两个参数坐标仍使用 **L∞ 口径**：

\[
\|\widehat x-x_*\|_2\leq e_x,\qquad
\|(\widehat u,\widehat v)-(u_*,v_*)\|_\infty\leq e_{uv},\qquad
\|\widehat n-n_*\|_2\leq e_n.
\]

这里使用 `||z||₂ <= sqrt(3) ||z||∞` 的保守换算，不是直接求出误差的精确欧氏范数。早期审查时点与法向尚为 L∞ 界；本文按最后加入该换算的代码记录，不能混用旧口径。

公共查询默认点与参数容差为 `2e-12`，法向容差为 `2e-10`；driver 将点容差设为当前 intersector 的 `geometry_tolerance()`，其余使用查询默认值。端点参数来自原始输入，因此该端点的参数误差界为零，但其三维点和法向仍需通过代表精度检查。

这些是几何查询的误差界，不是 PDE 解的最大误差、密度误差或收敛阶。数值残差和 transversality 代表值保留为诊断，不能替代存在性、唯一性与严格方向证明。

## 7. 缓存范围与未改变的 Cauchy 搬移

有两层新增缓存：

- 每个 intersector 的 `call_once` 精确几何缓存：从当前不可变模型提取一次有理 Bézier 元素，并同步保存它们的保守 AABB。后续路径不重复精确结点插入和全体 AABB 转换。
- 当前 operator setup 的 `(center, grid_node)` 路径缓存：两种迹侧以及 Q27/Q64 的重叠支撑节点复用同一完整几何路径。迹侧事件选择在适配器中独立执行，不污染几何缓存。

两类 cover 的模板构造结束后，driver 清空路径及中间 Cauchy 恢复缓存，保留已经装配的算子行。成功路径不长期保存大量 verbose dump；失败信息在抛出前保留。

模型控制点、权重、结点、刚体变换、网格、迹点参数或查询容差/预算发生改变时，应重建相应 intersector/setup。缓存不是脱离模型版本的全局浮点坐标表。公开的底层函数所接收的精确元素及可选边界必须来自同一当前模型；生产入口负责建立这一对应关系。

最近网格交点 Cauchy 搬移仍保留。当前装配在所选事件序列只有一个事件、非 feature contact、单 owner 等既有条件下，仍可在同一 G1 sheet 上选择最近网格交点作为恢复中心；不满足时使用原有精确事件 owner 路径。

因此不能把本批描述为“所有 correction 都改成精确事件中心”，也不能把该项搬移误差归为已经消除。几何事件认证和 Cauchy 恢复中心选择是两个不同环节。

## 8. 启用、预算和输出隔离

PowerShell 中可显式选择：

```powershell
$env:KFBIM_3D_SUPPORT_PATH = 'native_certified'
```

未设置、空字符串或 `legacy` 均走旧路线；其它值报错。新路线要求当前算子使用 Q27/Q64 cover。恢复旧路线可执行：

```powershell
$env:KFBIM_3D_SUPPORT_PATH = 'legacy'
```

可选的正整数预算环境变量如下；不设置时使用公共类型中的默认值：

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `KFBIM_3D_NATIVE_PATH_MAX_CANDIDATES` | 4096 | 候选元素上限 |
| `KFBIM_3D_NATIVE_PATH_MAX_BOXES` | 32768 | 跨精度重试累计的细分节点预算 |
| `KFBIM_3D_NATIVE_PATH_MAX_NEWTON` | 16384 | 跨精度重试累计的 Newton 步预算 |
| `KFBIM_3D_NATIVE_PATH_MAX_BITS` | 512 | 允许的精度位数上限 |

公共 API 还提供细分深度、同根关系尝试次数及精度升级次数等预算。它们是确定的工作量限制，不应表述为墙钟时间上限。第一阶段不调用最近点优化来修补证明，closest-point 预算默认且实际用途为零。

`support_path_output_root_3d()` 在启用新路线时，为应用输出根追加 `native_endpoint`。默认输出因此隔离在 `output/native_endpoint/...`；若设置 `KFBIM_3D_OUTPUT_ROOT`，则在该自定义根下使用 `native_endpoint/...`。不要把新目录中的试验记录覆盖或冒充已有 legacy 基准。

## 9. 第一阶段尚未支持的情形

- 端点位于 patch 边界，包括 C0 feature edge、vertex 和一般 patch 接缝：明确拒绝，不借容差挪入内部。
- 同 patch 内部 Bézier 分界上的端点：仅在精确位置一致、两侧正则法向同向及唯一性可证明时处理；真正非光滑内部结点不视为 smooth endpoint。
- 普通未知根恰位于 Bézier piece 边界或多个 patch 的共同边界：严格内部 Krawczyk 证明可能无法建立；目前没有完整的跨 piece/跨 patch feature-root 认证器，应返回未解决或预算耗尽，而非声称全部支持。
- 切触、线面重合、奇异法向和多 owner feature/vertex 事件尚无完整专用证明与合并路线。
- 当前 API 只表示“普通空间起点到原生曲面端点”。尚未提供对称的原生起点/普通终点反向 API；反向一致性不能仅通过现有签名交换两个三维向量来宣称完成。
- `endpoint_extension_ratio != 0` 明确不支持，`post_endpoint_roots` 在第一阶段为空。延长查询及端点后根的完整认证留待后续。
- 未新增完整 BVH、真正批量 interval 求交和全局统一 grid-edge/support-path 事件编号体系。

这些限制属于本阶段实现范围，不应通过默认回退或放宽认证门限隐藏。

## 10. 实际验证结果

以下检查已于 2026-09-07 执行。环境为 `build-3d`、Release、MinGW GCC 16.1.0，启用 MPFR。完整路径测试输出见[验证日志](KFBI3D_Native_Endpoint_Validation_20260907.md)。

精确几何基础层：

- 原始 double 分别有理化后相乘，与先舍入乘积的区别。
- 非均匀内部结点、两个方向结点插入、不同合法重数。
- 独立精确 de Boor 对照、解析曲面和常数次数案例。
- 内部点、元素边界、紧邻边界的参数点和向外舍入 AABB。
- 越域、非有限参数、坏权重、坏控制网、退化区间与非 clamped 端点的拒绝。

路径与适配器：

- 平面端点、多个普通根、同 patch 分块边界的已知端点。
- 路径外的小残差根、端点前极近但独立的普通根。
- 原始失败记录中的圆柱 N32、Tx D32、Tx D64 参数位模式；不以打印小数替代原始参数。
- 四道条件逐项失败、错误起始标签、候选根冒充认证根、低预算、无效或不支持的输入。
- 重复查询及刚体变换一致性。
- 实际 `NurbsSurfaceIntersector3D` 成员入口的首次查询、复用精确几何与 bounds、复制只读 intersector 后查询，三次结果和工作量一致。

已有构建目录可使用以下命令构建两个几何测试目标；具体编译器环境须与该目录的 CMake 配置一致：

```powershell
cmake --build build-3d --target native_nurbs_exact_geometry_3d_test native_endpoint_path_3d_test
& .\build-3d\apps\native_nurbs_exact_geometry_3d_test.exe
& .\build-3d\apps\native_endpoint_path_3d_test.exe
```

路径测试支持 `--list` 和 `--case SUBSTRING`，只运行有界几何查询，不启动 PDE。若构建生成器使用配置子目录，应按实际生成位置调整可执行文件路径。

| 验证项 | 状态 | 实际记录 |
| --- | --- | --- |
| 统一编译 | 通过 | 主程序、两个新增测试及下列既有测试，退出码 0；另有新内核独立语法检查通过 |
| 精确几何基础回归 | 5 组通过 | `native_nurbs_exact_geometry_3d_test`，退出码 0 |
| 原生路径与适配器回归 | 15/15 通过 | `native_endpoint_path_3d_test`，`SUMMARY selected=15 failures=0` |
| 实际成员入口及缓存 | 通过 | D32 Tx 的首次/重复/复制实例三次查询均 Certified，均访问 95 个区域，证书与代表值一致 |
| 既有 restrict 单元回归 | 4/4 通过 | crossing selector、path-state、tensor-product cover、shared quadratic 四个测试均退出 0 |
| 主程序启动检查 | 通过 | 启用 `native_certified` 环境后执行 `--help`，退出码 0；未启动 PDE |
| 旧诊断入口互斥检查 | 通过 | `native_certified` 配合 `--restrict-probe 16` 在 setup 前以预期退出码 1 拒绝 |
| 新旧路线完整 PDE A/B | 未执行 | 没有新增 setup 总耗时、解误差、收敛阶、GMRES 或 DOF 数值 |

最后再次构建全部上述目标，返回 `ninja: no work to do.`，确认报告对应的源码与构建产物一致。

负向测试的 PASS 表示符合预期地拒绝了不合法输入、错误标签、非有限区间、未完成证书或低预算，而不是把这些查询判成 Certified。既有 shared-quadratic 单测仅用于保留比较路线的回归，不改变主力 Neumann Q27 / Dirichlet Q64 选择。

### 10.1 三条历史失败路径的回放

这里的 N32/D32/D64 指原先失败记录所属的网格和边界条件；本次只回放对应的支撑节点到迹点路径，**不是重新计算三套 PDE**。时间采用最后一次完整 15 项测试中各 case 的 `steady_clock` 墙钟时间，包含 fixture 构造、几何查询和断言；单次测量会受机器负载影响，不能作为完整 setup 耗时或加速比。

| 历史路径 | 回放结果 | 开段普通根 | 原生端点 | 本次访问区域 | 累计 Newton 步 | 最高区间精度 | case 墙钟时间/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 圆柱 Neumann N32，center 507 / node 13519 | Certified，双侧适配通过 | 0 | 1，enter `+1` | 251 | 2202 | 128 bit | 2.3980575 |
| 圆柱 Dirichlet Tx N32，center 17 / node 12531 | Certified，双侧适配通过 | 0 | 1，enter `+1` | 95 | 364 | 53 bit | 0.0920243 |
| 圆柱 Dirichlet Tx N64，center 29 / node 95013 | Certified，双侧适配通过 | 1，enter `+1` | 1，leave `-1` | 109 | 460 | 53 bit | 0.2846784 |

三条路径均 `unresolved_regions=0`，四道条件全部通过。精度列是区间计算使用的位数，Newton 步是求交器的工作量，**两者都不是 PDE 精度或 GMRES 次数**。访问区域是最终成功精度 pass 的计数，而 Newton 工作量跨精度重试累计，因此不应把两列误作同一次 pass 的平均成本。

D64 保留的开段根满足测试参考 `t≈0.925171384642`。恢复内侧时使用这个开段根、不添加端点；恢复外侧时使用开段根及 `t=1` 的离开端点。这正是条件端点处理，而不是无条件把端点追加到两个迹侧。

此外，相距 `2^-40` 的“开段真根＋原生端点”均被保留，case 时间为 0.003904 s；极端权重造成非有限区间时返回 `Unresolved / NonFiniteBernsteinInterval`，没有静默排除候选。

### 10.2 构建一致性说明

开发中增加两个事件 ID 字段后，首次构建过程中曾混用旧 path-state 对象与新事件结构，导致有开段事件的适配检查失败；这不是求交根丢失。已完整增量重编该头文件的消费者、重新链接，并在此之后取得上述 15/15 结果。不要只替换部分旧对象或沿用该中间可执行文件。

PDE 对比必须固定物理几何、刚体变换、方程、网格范围与步长、密度与约束配置、Q27/Q64 路线以及 GMRES 参数。若在 setup 阶段失败，应报告失败阶段及其耗时，不能填写虚构的解误差或把尚未开始的 GMRES 写成已收敛。
