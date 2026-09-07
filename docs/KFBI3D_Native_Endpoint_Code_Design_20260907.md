# KFBI3D 原生端点路径认证：代码设计

日期：2026-09-07。状态：设计稿，未实现、未编译、未运行 PDE。

本文只规定代码边界、数据合同和实施验收；数学推导见配套的[算法文件](KFBI3D_Native_Endpoint_Algorithm_20260907.md)。

历史设计快照：本文“未实现”等表述描述设计编写时的状态；当前第一阶段实现、验证与尚未支持范围见[实现说明](KFBI3D_Native_Endpoint_Implementation_20260907.md)，测试原始输出见[验证记录](KFBI3D_Native_Endpoint_Validation_20260907.md)。

文中的新增文件、类型、函数和字段均为**拟议设计**，不是仓库中已经存在、可直接编译的实现。
源码定位基于本次只读检查，后续修改后以函数名为准；不得将本文的上线门槛理解为已通过的测试结果。

## 1. 目标、范围与非目标

目标是消除以下接口错配：局部接受小残差候选，上层按固定 uv/t 距离识别根，再按启发式不确定区间判断歧义，最后以更窄阈值寻找已知端点。
需要提供从原生几何路径到应用层事件序列的可审计合同，而不是单独放宽某个 tolerance。

第一期明确支持：

- 端点位于单个 physical smooth sheet 的 patch 内部；局部参数化正则，分母严格为正，法向方向可验证。
- 原生端点对查询路径的横截性可证明，起点可证明不在边界上。
- 普通光滑横截根具有存在性、唯一性和位置包围证书。
- 所有候选区域被覆盖，所有交点和端点的顺序、侧别转换与代表点精度均通过验收。

第一期不承诺：

- feature/vertex 端点、patch 边界端点、切触端点、退化参数化或重叠线段自动成功。
- 普通多 owner 根仅凭拓扑标签或一个 `exact_inside` 布尔值就获得严格事件转换证书。
- 任意近根在固定精度和固定工作量下都能分离。

明确不支持的端点返回 `UnsupportedEndpoint`；可能可解但证据不足返回 `Unresolved`；耗尽预算返回 `BudgetExceeded`。
起点在边界上第一期也返回 `UnsupportedEndpoint`，诊断原因标为 `start-on-boundary`，不能悄悄采用普通网格 label。
普通 open root 若涉及 feature，多 owner 和现有 side classifier 可以复用，但缺少可验证根身份/侧别证据时必须拒绝放行。

本设计不改变 Neumann/Dirichlet 仿射密度空间、不新增 PDE 策略、不改变 P2/P3 路线、Q27/Q64 stencil、3+3 法向层、GMRES 容差或算子迭代公式。

## 2. 当前修改点与职责迁移

| 当前代码位置 | 当前职责/限制 | 拟议修改 |
|---|---|---|
| `src/geometry/nurbs_bezier_intersection_3d.hpp:14`，`NurbsElementRoot3D` | 只传近似 uv/t、point、residual、transversality | 候选与可选证书分开保存 |
| 同文件 `:83`，`NurbsElementSegmentCertificate3D` | 单个类别加 optional root，无隔离证明对象 | 增加可复核证书和覆盖摘要，禁止候选冒充证书 |
| `nurbs_bezier_intersection_3d.cpp:442`，`certifies_unique_transverse_root` | 区间 Jacobian 收缩证明至多一个零点 | 保留为 at-most-one 原语；与存在证组合 |
| 同文件 `:704`，`native_newton` | 几何残差验收、t clamp、浮点 polishing | 只生成候选和精度诊断，不自行赋予存在证 |
| 同文件 `:860`，`roots_agree` | 固定 uv/t/空间阈值去重 | 仅作候选筛选；正式合并改用根关系证明 |
| 同文件 `:2304、2432` | 唯一性成立后结束当前区域搜索 | 必须同时存在已认证根，并登记区域覆盖证据 |
| `src/geometry/nurbs_surface_intersector_3d.hpp:57` | 跨层 root 不携带认证盒 | 传递证书引用、owner 来源、查询身份 |
| `nurbs_surface_intersector_3d.cpp:208、772` | 同片及接缝 canonicalization | 引入 `RootRelation3D`，保存合并证明链 |
| 同文件 `:962、977` | residual/alpha 不确定度与近根簇 | 降为调度启发式；证明状态由认证盒决定 |
| 同文件 `:1933` | filtered 查询用启发式区间统计独立根 | 改用认证区间和明确的覆盖/早停合同 |
| `apps/restrict_crossing_selector_3d.cpp:406` | 固定 uv 与空间距离识别端点 | 使用已知端点 proof/event 身份及 t 区间 |
| `apps/neumann_exterior_zero_trace_3d.cpp:4159–4224` | `tie`、固定 `4*tie` 延拓、端点划分 | 增加原生端点查询入口，保留有条件延拓 |

现有 BVH、Bezier 分解、dominant-component 齐次直线方程、区间 Jacobian 运算、all-event 目录和 fail-closed 行为应尽量复用。
特别是齐次直线方程不改回基于舍入单位正交轴的方程，避免另引入几何定义偏差。

## 3. 拟议文件边界

| 拟议新增文件 | 单一职责 |
|---|---|
| `src/geometry/native_endpoint_path_3d.hpp/.cpp` | 原生端点、路径语义、查询身份、总流程和状态 |
| `src/geometry/nurbs_root_certificate_3d.hpp/.cpp` | 区间根证书、存在/唯一/位置认证、根关系证明 |
| `src/geometry/path_certification_diagnostics_3d.hpp/.cpp` | 确定性计数、结构化诊断与可复现 dump 数据 |

现有局部求交文件保留候选生成和细分；现有 surface intersector 负责候选覆盖与聚合。
若实现阶段发现独立 `.cpp` 导致内部区间工具重复，应先抽取已有区间工具，不要复制第二套不同舍入规则。
日志文件写出可放在 app 层；几何核心只提供可序列化的数据，不在每次求交时隐式写磁盘。

## 4. 数据模型与不变量

以下 C++17 风格代码用于说明接口，不是完整头文件；`Interval`、身份类型和序列化类型均待实现。

```cpp
enum class RootProofKind3D {
    Candidate,
    ExistsUnique,
    KnownEndpointUnique
};

enum class RootRelation3D {
    SameRoot,
    DistinctRoots,
    Unresolved
};

enum class PathStatus3D {
    Certified,
    Unresolved,
    BudgetExceeded,
    UnsupportedEndpoint,
    DegenerateSegment,
    InvalidInput
};

struct NativeSurfaceEndpoint3D {
    GeometryModelIdentity3D model;
    int patch_index;
    double u;
    double v;
    std::optional<Eigen::Vector3d> rounded_point_hint;
};

struct RootCertificate3D {
    RootProofKind3D kind;
    RootProofId3D proof_id;
    PathIdentity3D path;
    NativeChartIdentity3D chart;
    ParameterBox2D uniqueness_box;
    ParameterBox2D root_enclosure;
    Interval t_interval;
    RootProofEvidence3D evidence;
    PrecisionPolicyIdentity3D precision_policy;
};

struct CertifiedPathRoot3D {
    NurbsSurfaceCrossing3D representative;
    RootCertificate3D certificate;
    std::vector<NativeRootOwnerEvidence3D> owners;
    std::optional<PhysicalEventId3D> event_id;
    EventTransitionEvidence3D transition;
    RepresentativeAccuracyEvidence3D accuracy;
};
```

必须维护以下区分：

1. `Candidate` 表示数值候选状态，不是可供应用层使用的根证书；正式证书工厂只能构造后两种有效证明。
2. `uniqueness_box` 是证明至多一个根的区域；`root_enclosure` 是证明该根所在的区域，两者不必相同。
3. 已知端点的原生参数可提供退化为单点的根位置证据，但其邻域唯一性仍需单独证明。
4. `proof_id` 标识证明对象/来源；`event_id` 标识跨 owner 合并后的物理事件，二者不能互代。
5. `transversality` 数值估计、`reliable_transversality_tolerance` 和严格横截性下界不是同一个字段。
6. 证书须绑定模型、路径和精度策略；不能把相似位置上的旧证明搬到新模型或新方向。
7. 原生参数、模型参数和 double hint 都必须检查有限性；不允许 NaN 进入排序、缓存或区间运算。

`RootProofEvidence3D` 至少记录证明方法、有效域、W 正性、非零 D 分量、Jacobian 收缩界、存在性见证和精度级别。
“记录证据”不要求生产环境保留全部庞大区间网，但必须能定位原生数据和按相同策略复核；debug/replay 模式保存完整局部证据。

### 4.1 路径结果

```cpp
struct NativeEndpointPathResult3D {
    PathStatus3D status;
    PathIdentity3D path;
    std::vector<CertifiedPathRoot3D> open_roots;
    std::optional<CertifiedPathRoot3D> endpoint;
    std::vector<CertifiedPathRoot3D> post_endpoint_roots;
    PathCoverage3D coverage;
    PathCertificationDiagnostics3D diagnostics;
};
```

`Certified` 交付 app 必须同时通过四道 gate：

| gate | 必须成立的条件 |
|---|---|
| 根覆盖 | 查询范围内全部候选被排除、认证或细分覆盖；无未解决区域 |
| 根顺序 | 不同事件的认证位置可严格排序；端点身份唯一明确 |
| 事件转换 | 起点状态、各根 entering/leaving/contact 及端点 incoming 状态具有一致证据 |
| 代表点精度 | 实际传给 correction 的 point/uv/normal 满足应用层精度合同 |

单独的 `coverage.complete=true` 不意味着 `status=Certified`。
失败结果可以携带已经证明的 partial roots，但 app 不得把这些 partial roots 当作完整 correction 目录。

## 5. 原生端点与路径的精确语义

`NativeSurfaceEndpoint3D` 的几何定义是原生 NURBS 在给定参数处的点，而不是 `rounded_point_hint`。
模型中存储的控制数据、结点、权重及参数的二进制值应固定解释；认证运算证明的是这一个声明的原生模型。
如果加速用 Bezier 数据经过浮点提取，认证所用系数必须精确对应原生模型或以向外包围明确覆盖提取误差。
不能只认证一个舍入后的替代 Bezier 面，却声称认证了原生端点恒等式。

设查询起点为 p，端点的齐次值为 `(H_q,W_q)`，其中 `W_q>0`，定义：

\[
q=H_q/W_q,\qquad D=H_q-pW_q,\qquad x(t)=p+tD/W_q.
\]

原生端点恒为 `t=1`，无论是否启用端点延拓。
用可证明非零的 dominant 分量 `D_k`，复用现有齐次直线方程：

\[
G_j(u,v)=D_k(H_j-p_jW)-D_j(H_k-p_kW)=0,\quad j\ne k.
\]

普通根的沿线参数包围采用：

\[
t(X)=\frac{W_q\,(H_k(X)-p_kW(X))}{W(X)D_k}.
\]

实现时必须验证 `W_q>0`、`W(X)>0` 和 `0∉D_k`，每一步向外舍入。
若当前区间精度无法判断 D 的某个分量非零，按预算升精度；不能由浮点 `abs(D_k)>0` 代替证明。
若路径确实退化返回 `DegenerateSegment`；尚不能判定退化与否返回 `Unresolved` 或 `BudgetExceeded`。

已知端点处 `G_j(u_q,v_q)=0` 来自相同齐次定义的代数恒等式；不能用“小于 geometry tolerance”替代该等式。
有限精度区间运算中恒等式可能只表现为包含零的窄区间，应保留结构化原生见证，而不是把“区间含零”当成存在性证明。
`rounded_point_hint` 只用于一致性诊断、显示及向后兼容；不一致时报告其误差，禁止反向修改原生定义去迁就 hint。

现有 `intersect_segment(start, rounded_end)` 保持“这两个 double 端点定义的线段”语义。
新接口不是该旧接口的无条件别名：两种路径在浮点上可能差一个舍入量，必须在 dump 和缓存 key 中区分。

## 6. 拟议接口

```cpp
NativeEndpointPathResult3D intersect_segment_to_native_endpoint_3d(
    const NurbsSurfaceIntersector3D& intersector,
    const Eigen::Vector3d& start,
    const NativeSurfaceEndpoint3D& endpoint,
    const NativeEndpointQueryOptions3D& options,
    const PathCertificationBudget3D& budget);

RootCertificationAttempt3D certify_ordinary_root_3d(
    const NativeLineEquationContext3D& line,
    const NativeChartBox3D& box,
    const NurbsElementRoot3D& candidate,
    PathCertificationWorkState3D& work);

RootCertificationAttempt3D certify_known_endpoint_3d(
    const NativeLineEquationContext3D& line,
    const NativeChartBox3D& box,
    const NativeSurfaceEndpoint3D& endpoint,
    PathCertificationWorkState3D& work);

RootRelationResult3D resolve_root_relation_3d(
    const RootCertificate3D& first,
    const RootCertificate3D& second,
    PathCertificationWorkState3D& work);
```

`RootRelationResult3D` 包含三态值和关系证据；不是只有一个无从审计的枚举。
`NativeEndpointQueryOptions3D` 应包含模式版本、代表点精度要求、是否允许延拓、延拓策略、精度阶梯和诊断选项。
预算单独传入，避免把“提高几何精度”和“无限增加工作量”混成一个参数。

上述自由函数应只是薄 facade；实际候选调度入口拟增为 `NurbsSurfaceIntersector3D::intersect_segment_to_native_endpoint(...)` 成员函数。
该成员访问私有 BVH/element 数据并调用局部认证原语，不公开可变 `elements_`，也不绕过现有 candidate owner 校验。
现有 `certify_candidate_segment(candidate,start,end)` 仍采用两个 double 端点语义，不能把它直接当成原生端点认证接口。
`nurbs_root_certificate_3d.hpp` 仅依赖区间/身份/参数盒等基础类型，不依赖 intersector；intersector 头文件前向声明端点、预算和结果类型。
完整路径结果在 `native_endpoint_path_3d.hpp` 定义，成员定义所在 `.cpp` 再包含完整类型，避免新增循环 include。

## 7. 局部求根：候选、存在性、唯一性、精度

### 7.1 Newton 和 closest point 只生成候选

保留现有 triangle/sample seed、Newton 和 closest-point assistance。
`native_newton` 可以继续返回最佳近似和残差，但 clamp 到 t=0/1 的小残差点不直接形成真根。
禁止仅因为 residual 下降到 `64*epsilon*scale`，就认为已满足固定 uv 身份阈值或下游 correction 精度。
候选的数量与已认证根的数量分别计数。

### 7.2 普通横截根：完整存在唯一证

在原生参数盒 X 上用两维齐次方程 `G=(G_1,G_2)`，构造：

\[
K(X)=x_0-CG(x_0)+(I-CJ_G(X))(X-x_0).
\]

`G(x0)`、`J_G(X)` 及所有乘加均以同一原生模型向外包围。
当 `K(X)⊂int(X)` 给出存在性，且已有 at-most-one 证成立，才产生 `ExistsUnique`。
这里 C 的可逆性不能省略：本方案同时验证 `||I-CJ_G(X)||<1`，由此保证所用 C 可逆；若拆分存在证原语，须显式检查同一前提。
随后计算 t 包围区间并验证有限线段范围；证明无限直线上有根，不等于有限线段内有根。
认证失败时细分或升精度；最终未证实不等于“没有根”。

根落在人工细分边界时，可以构造跨相邻子盒、仍属于同一有效 patch 图册的认证区域。
真正的 patch 边界根不能靠越界延拓参数化强行满足内点包含；需边界感知证书，否则一期返回未解决。
真切触/奇异根保留独立诊断，不能用横截性 floor 改造成 transverse。

### 7.3 已知原生端点：恒等存在＋at-most-one

在包含 `(u_q,v_q)` 的局部盒内，以原生路径恒等式给出存在性，再调用现有区间 Jacobian at-most-one 原语。
两者成立才产生 `KnownEndpointUnique`，其原生参数位置和 `t=1` 不依赖 Newton 搜索精度。
唯一性失败时继续分裂盒；不能因为盒中有端点就跳过该盒的其它可能根。
盒内多个候选若均被共同唯一性证明涵盖，只是同一端点的多个近似，不构成多个事件。

### 7.4 代表点精度仍需独立检查

证书保证一个根存在，不自动保证选择的 representative 足够精确。
对普通根继续收缩已认证盒或对该根升精度求值，使 point/uv/normal 的包围宽度满足应用要求。
对已知端点从原生参数直接求值；法向和外向约定也需单独验证。
`residual/alpha` 只能指导初始盒大小、优先级或精度预算，不能直接写进 `certified t_interval`。

## 8. 根关系、近根保护和物理事件身份

正式 canonicalization 使用 `RootRelation3D`，旧的固定距离比较只作廉价候选筛选。

- `SameRoot`：例如，两份存在证所包围的根都落在同一个可验证至多一根的区域；保留共同唯一性或包含关系证据。
- `DistinctRoots`：例如，两个真实根的认证 t 区间分离，或者另有严格几何证据。
- `Unresolved`：无法证明相同或不同，继续收缩/细分/升精度，预算用尽则失败。

两个 root enclosure 相交不证明同根：各自真实根可能位于交集之外。
两个 t 区间重叠也不证明同根；它只是需要进一步判断的信号。
同 patch、同 sheet 或同 component 都不足以单独证明同一物理事件。
跨 patch 合并必须用已声明拓扑、参数映射及公共物理根的可验证等价关系，不能只比较点距离。
合并保存全部 owner 和证明来源；代表点选择按确定性规则，不让线程顺序决定结果。

已有 `root_edge_parameter_uncertainty` 和 stationary witness 可以继续辅助选择细分方向或检查可疑根簇。
它们不再直接解除 ambiguity、决定完整根数或赋予 `event_id`。
不同真根即使小于旧的16倍 geometry tolerance，仍须保留；如果当前精度无法排序，结果应是未解决，不是合并。

## 9. 全候选覆盖与查询流程

拟议流程：

```text
validate model/path/endpoint/start-side
  -> construct native homogeneous line context and enclosing query bounds
  -> deterministic BVH candidate coverage
  -> per-region exclusion / candidate generation / existence+uniqueness proof
  -> unresolved regions: local subdivision or bounded precision escalation
  -> relation proofs and canonical physical roots
  -> root order + endpoint identity + transition + representative accuracy
  -> Certified result or explicit non-success status
```

`PathCoverage3D` 至少保存：声明查询区间、原始候选目录身份、每个候选的状态、细分覆盖树摘要及未解决纵向区间。
原始候选区域只有以下出口：证明无根；证明根集合且覆盖其余区域；完整划分为已登记子区域。
只证明某个小盒存在唯一根，不能据此宣布整个原始 Bezier element 处理完毕。
已知端点 seed 必须进入调度，但不能代替 BVH 对其它候选的枚举。
BVH query bounds 要包含原生端点的可靠空间包围；不得只用可能漏掉真实原生点的 rounded hint 构造无 padding bounds。

所有 seed、候选和子盒按稳定顺序处理；并行实现必须在归并前确定排序键。
开始标签相同、期望侧别相同、或目前只找到端点，都不能作为终止全候选求交的理由。
保留已认证 partial roots 和 unresolved 区域作为诊断，但不把 incomplete coverage 交给 spread/restrict 装配。

## 10. 端点延拓与应用层事件转换

### 10.1 延拓保留，但必须有条件

原生闭合查询默认声明 `t∈[0,1]`；已知端点存在性不依赖人为延拓。
第一期默认关闭延拓；本节其余条款是显式可选的后续能力合同，不是第一期必须启用的求交捷径。
若某种认证/侧别取样需要延拓，采用同一原生直线 `t∈[0,1+lambda]`，其中 lambda 明确为正并进入查询身份。
不要重新定义一条经过 independently rounded `q+delta*dhat` 的直线。
延拓长度选择可以参考横截性和已有认证邻域，但固定 `4*tie` 或 `geometry_tolerance/alpha` 都不是成功证书。
每次选定延拓查询后，应认证其完整声明范围；延拓后端点仍固定为 t=1，post roots 单独返回。
extension 内的第二个真根必须保存；延拓内 unresolved 区域不得静默忽略。
如设计另一次较短查询，必须记录为新查询及独立覆盖证明，不能把失败长查询裁剪后直接标为成功。

### 10.2 partition 消费证明而不是猜测身份

端点通过已知根 proof/event 关系识别，不再以 `abs(u-uq)<16e-12` 作为决定性条件。
普通根满足 `0<t_lo<=t_hi<1` 才进入 open roots；满足 `1<t_lo<=t_hi<=1+lambda` 且有限查询范围已认证，才进入 post roots。
完全处于声明查询区间之外的根应排除但保留覆盖依据；区间与0相交且身份未明时必须精化，不能提前作为 open event。
普通根区间跨越0或查询终点而未判定归属时不能交付 `Certified`；证实起点在边界则按一期范围返回 unsupported。
区间跨越1且尚未证明是端点，应继续收缩/判定；不能仅按代表点 t 的严格大小分类。
用于 `[start,q)` 的 open roots 不包含端点，但端点证明保留在结果中供 continuation 使用。
`MissingExactEndpoint` 的旧诊断需要细化为：无候选、存在性未证、身份未证、唯一性未证、精度未达标等原因。

### 10.3 几何 sign 不能由 desired side 反推

对支持的光滑端点，设 `b=n_out(q)·(q-p)`；应证明 b 的符号，不仅检查数值估计。
`b<0` 表示 entering，sign=+1，incoming_inside=0；`b>0` 表示 leaving，sign=-1，incoming_inside=1。
实现中可用带已验证外向符号的未归一化叉积计算同号量，避免不必要的平方根。
open events 推进后的状态必须与该 incoming 证据一致，同时核对传入的 `node_inside`。
`desired_inside` 只选择应用层需要恢复的迹侧，不产生根、不决定真实 crossing sign、不修复不一致的几何序列。
多 owner open roots沿用现有目录和 classifier 接口时，必须附可复核拓扑/侧别证据；单个布尔 callback 不是严格认证。

拟议 app 接入伪代码如下；函数名和证据访问器均为示意，不是现成可编译实现：

```cpp
auto path = intersector.intersect_segment_to_native_endpoint(
    node_point, native_endpoint, options, budget);
require(path.status == PathStatus3D::Certified);
require(path.endpoint.has_value());
require_all_four_gates(path);
require_matches_certified_start_state(node_inside, path);
auto state = apply_all_open_events(node_inside, path.open_roots);
const auto& endpoint = *path.endpoint;
require(state == endpoint.transition.incoming_inside);
assemble_all_open_crossing_terms(path.open_roots);
if (desired_inside != state) {
    require(desired_inside == endpoint.transition.outgoing_inside);
    assemble_endpoint_term(endpoint); // sign 来自几何证据，不来自 desired
}
require_final_branch_and_telescoping_consistency(path, desired_inside);
```

实际 default-center/net-change 系数仍按现有 all-event telescoping 公式装配；上例只展示几何验收及端点是否参与的控制流，不另立 PDE 公式。

## 11. 接口兼容与 filtered 早停

旧的双 double 端点接口不改变几何语义，保留旧行为供 A/B 和已有调用者使用。
新增证书字段时检查所有聚合初始化位置，尤其 `as_surface_crossing()`，避免字段位移导致静默错误。
旧结果可以经适配器变成新候选，但不能自动升级为新 `Certified`。
新接口到旧消费者的适配仅在四道 gate 已通过后进行，且必须保留 event/owner 审计信息。

filtered API 仍可以在明确的消费合同下早停，但必须满足：

- “已找到至少 k 个独立真根”以认证根和关系证明计数，不能用启发式不确定区间代替。
- 早停结果明确 `all_candidates_processed=false`、`coverage.complete=false`。
- 最近根查询只有证明所有未处理区域都严格位于已选根之后时，才能宣称最近根已认证。
- nearest/tie 所需代表点精度与排序要求要单独通过，不能把浮点最小 t 当作最近根证据。
- all-event/native-endpoint 查询禁用按根数量早停，不得复用 filtered 的 partial result 冒充完整目录。

`maximum_independent_crossings` 不是总根数上限；不得因此截断实际多事件。
旧测试中人工构造 root 的 fixture，应明确构造测试证据或走 legacy API，不能依靠空 diagnostics 自动获得新认证状态。

## 12. 缓存身份与复用边界

当前 app 的 `(center,node)` key 只在一个固定 pipeline 实例内具有隐含几何上下文。
新设计应把这些隐含条件显式纳入 key 或由不可变对象生命周期强制保证：

| 身份组成 | 必须区分的变化 |
|---|---|
| 原生模型 | 控制数据、结点、权重、拓扑、法向约定及刚体变换 |
| 路径语义 | double-endpoint / native-endpoint，起点精确位模式，patch/u/v |
| 方向 | 正向/反向；逆向复用必须显式变换 t 区间、顺序与 sign |
| 查询范围 | 主区间、是否延拓、lambda 与延拓版本 |
| 认证策略 | 区间/精确 backend、精度阶梯、证书算法版本 |
| 代表点要求 | point/uv/normal 的目标误差合同 |
| 失败预算 | budget profile 及已用工作量；较小预算失败不能阻止较大预算重试 |

不要把仅 surface class、component、sheet 或中心空间坐标作为完整缓存身份。
同一原生路径在 N/D、Q27/Q64 和 interior/exterior 分支之间可以复用几何证书；这些应用选择不应重复构造同一原始路径。
但 density coefficients、jump jets、PDE 状态和 desired side 不应写进几何根证书，也不能影响其内容。
逆向路径若用另一种端点语义定义，不能仅因两个 rounded 点交换就自动命中缓存。
无向 key 只有在两个端点的原生语义可验证交换、并实现向外舍入区间反变换时才允许使用。

提高精度可使旧证书继续有效，但是否可直接交付仍需核对新代表点要求和算法版本；禁止无条件降级复用。
partial/失败缓存只能作为继续计算的材料，不能在新请求中转换成成功。

## 13. 确定性工作量预算与诊断

```cpp
struct PathCertificationBudget3D {
    std::uint64_t max_candidate_regions;
    std::uint64_t max_subdivision_nodes;
    unsigned max_subdivision_depth;
    std::uint64_t max_newton_steps;
    std::uint64_t max_closest_point_evaluations;
    std::uint64_t max_relation_refinements;
    unsigned max_precision_escalations;
    unsigned max_precision_bits;
};
```

各字段需显式配置并检查有效性；不使用“0表示无限”的隐式无限预算。
默认 profile 的具体数值须由几何单测与单路径回放校准，本文不虚构已经可用的数值。
调度顺序固定，计数在执行工作前检查；所有子查询和延拓尝试共享父级总预算，不能重试时清零逃避上限。
wall/CPU 时间是性能诊断，不是决定算法分支的唯一预算，否则相同输入在不同机器上可能给出不同状态。
closest-point 预算须计入内部实际函数/导数评估，不能只计算外层调用次数；失败重试和嵌套 assistance 同样消耗统一预算。
取消请求单独记录，不伪装成几何失败；统计溢出、非有限区间和非法模型不能继续传播。

拟议精度阶梯为 binary64 向外舍入快路，再按预算选择可选 MPFR 定向舍入的128/256/512 bit 局部路径。
这不是本轮新增依赖或已存在 backend；缺少所需 backend 时返回 `Unresolved`，原因 `PrecisionBackendUnavailable`。
每次晋级须从原始 IEEE 位模式的控制数据、结点、权重及原生端点参数重新构造方程与保守 Bezier 提取/细分。
仅把已经舍入的 double root 或旧宽区间转换到高精度，不会恢复原始几何信息，不能算完成精度晋级。
不能假设 `long double` 等价于可用的严格高精度 backend；精度/舍入支持必须显式验证。

最低诊断字段：

- 路径/模型/策略/预算 ID，最终状态与细分原因码。
- 原始候选数、数值候选数、存在唯一证数量、已知端点证数量、物理事件数。
- 同根证明次数、真近根保留次数、关系未解决次数、未处理与 unresolved 区域数量。
- 原生 q 与 rounded hint 的差、q 的 uv、起点、D、Wq、dominant 分量和延拓 lambda。
- 每个根的 uv/t 代表点、residual、认证盒、横截性证据与代表点精度。
- coverage/order/transition/accuracy 四个 gate 的结果和具体失败位置。
- 各阶段 CPU/wall 时间，标明测量口径，不声称跨平台的 `std::clock` 一定是 CPU 时间。

dump 使用可往返的十六进制浮点或足够17位十进制，并保存原生模型可重建配方及指纹。
可复现性不能只依赖裸内存地址、运行时 center 序号或日志里四舍五入的坐标。
推荐记录稳定 candidate/box/proof ID、策略版本、编译浮点选项和精度 backend。
输出大小受独立日志上限控制；若截断，应显式标记，并仍保留重放路径与模型所需的最小数据。

## 14. 测试挂载与验证阶段

### 14.1 复用现有 target

| 文件及入口 | 增补内容 |
|---|---|
| `apps/native_nurbs_surface_3d_test.cpp:1726` | candidate 与 ExistsUnique/KnownEndpointUnique 分层 |
| 同文件 `:1829` | 真根在线段/盒外但残差小的候选不得认证；全区域覆盖 |
| 同文件 `:1939` | 两个真近根的分离、预算不足不合并 |
| 同文件 `:2099` | 子盒边界同根以共同唯一盒合并，而非距离碰巧接近 |
| `apps/restrict_crossing_feature_side_classifier_3d_test.cpp:166、194` | 原生端点身份、近根保留、四道 gate 失败 |
| `apps/restrict_crossing_selector_3d_test.cpp:180` | 独立真根即使距离小于旧 tolerance 仍保留 |

现有端点 partition 测试采用人工 root、0.1延拓距离及1e-10参数阈值，未覆盖本次约1.3e-10物理延拓的真实近切路径。
必须加入真实 NURBS intersector 到 partition 的集成单路径测试，不能只再写一组理想人工点。
三条固定圆柱 fixture 的模型、姿态和原生参数见[算法文件第9.1节](KFBI3D_Native_Endpoint_Algorithm_20260907.md)。
fixture 须保存原 CSV 的 round-trip/IEEE 位模式；例如 D32 的 u 是 `0.15000000000000002`，不能换成字面量 `0.15` 或理想有理数。
表中的代表 q 只供显示与校验，不反向定义原生测试路径。

### 14.2 必要的新回归矩阵

- 本次失败圆柱 q/node 路径：多种 seed、多个 query leaf、正反向回放，物理根数一致。
- 一个真端点＋多个可接受小残差近似：同根有证据地合并；不产生假 `t≈1` 事件。
- 两个真实近根：间距跨越旧 tie/uv 阈值，证据足够时全部保留，预算不足时明确失败。
- 仅盒外/段外存在零点：clamp 后残差小也不能获得存在证。
- endpoint 的 rounded hint 扰动：原生路径语义不变；旧 double 路径作为不同查询处理。
- 正权重、W区间含零、退化 D、横截性区间含零及非法参数分别覆盖。
- 端点在 feature/vertex/patch 边界、起点在边界：一期返回明确 unsupported，而非悄悄回退。
- 声明延拓内存在第二根或 unresolved region：不得只交付主路径已知端点后忽略剩余范围。
- owner/seam等价、无法证明的多 owner侧别、模型变换/精度策略缓存失配。
- 全部权重乘同一正比例因子后的几何不变性；使用保持比例关系的精确模型构造，避免将独立舍入后的不同模型误称为相同几何。
- 人为降低预算：确定性 `BudgetExceeded`、partial evidence保留、同预算回放状态一致。

### 14.3 CMake 和后续数值阶段

现有几何源注册在 `src/CMakeLists.txt:49–53`；拟议新增源只在对应 geometry 构建条件下加入同一库。
已有测试 target 挂载在 `apps/CMakeLists.txt:126、182、188`，链接段 `:413、495、500`。
可拟设独立 CMake feature 控制 MPFR 认证 backend；只有后续获得实现授权时才加入依赖探测和链接，未启用时须按上述缺 backend 合同返回状态。
先扩展这些 target；若新增独立 replay 工具，应单独建 target，默认不运行大网格或 PDE。
不能因测试 fixture 暂未适配，就删除现有 all-event、逆向查询、feature owner 或 fail-closed 回归。

验证依次进行：文档/接口审查 → 编译与单测 → 固定单路径 replay → 仅几何目录 setup → 经授权的小网格 PDE A/B → 经授权的32/64/128完整数值报告。
每阶段分别记录：状态/根数/身份与覆盖，随后才是 setup耗时、场误差、收敛阶和GMRES次数。
在几何 gate 未通过时，不得调整 GMRES 参数试图掩盖 setup 失败。
本文没有执行这些阶段，也没有声称任何精度、性能或收敛测试已经通过。

## 15. 分期实施和上线门槛

建议以小范围、可回滚提交推进：

1. 固定失败路径输入、诊断格式和现有基线；加入不会自动跑 PDE 的 replay/回归。
2. 引入拟议数据类型和只读诊断适配，旧求交路径行为保持不变。
3. 实现原生端点路径语义、已知端点存在唯一证和光滑内部端点支持。
4. 接入普通根存在证、共同唯一盒关系判断和完整覆盖账本。
5. 接入证书驱动 partition、四道 gate、几何缓存身份及 filtered 合同。
6. 通过上述测试后，才以显式配置启用新路径；在经过授权的数值 A/B 完成前不替换默认生产基线。

上线必须同时满足：

- `Certified` 只来自四道 gate；没有 unresolved 区域、未确认事件顺序或未达精度的代表点。
- 本次真实失败路径不再依靠调大固定 tolerance 偶然成功，dump 能说明每个事件为何存在、为何合并/保留。
- 真近根和多事件不减少；正反向事件顺序/sign一致；已知端点不会吞掉近邻真根。
- 不支持情形和预算用尽均在 operator setup 前明确停止，不静默切换 nearest-root 或跳过路径。
- 缓存跨N/D/Q27/Q64复用不改变几何结果；跨模型、路径语义、方向和精度失配不会误命中。
- 已有生产数值指标的变化经过独立报告，不将更少失败、更少候选或更短耗时直接称为算法精度提升。

回滚保留旧 API 和显式模式开关；新路径失败时默认报告失败，而不是在同一次算子装配中无提示混用旧路径。
feature/vertex 端点和一般切触完整支持另立后续设计与验收，不能由本期光滑端点通过推断已经解决。

## 16. 设计结论

本次修改的核心不是“让求交更容易通过”，而是把**数值候选、存在唯一证、物理事件身份和应用可用精度**分成可验证的层次。
原生端点是可利用的几何见证，但只有在路径语义一致、局部唯一性和全路径覆盖同时成立时，才能安全地减少重复搜索及误判。
配套[算法文件](KFBI3D_Native_Endpoint_Algorithm_20260907.md)规定这些证书的数学含义；实现不得以残差阈值、区间重叠或期望侧别代替其中任何一项证明。
