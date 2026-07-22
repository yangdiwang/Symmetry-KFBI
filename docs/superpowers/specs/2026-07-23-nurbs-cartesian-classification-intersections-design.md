# 面向 Cartesian 内外判定的 NURBS 求交设计

日期：2026-07-23
状态：设计已讨论，等待用户最终审查

## 1. 目标

本模块为 KFBI 三维 Cartesian 网格的点内外判定和界面记录服务，
不构造通用、完备的线面求交器。设计目标是：

- 保存一条网格边上发现的全部物理交点；
- 正确得到边两端内外状态是否翻转；
- 不因多交点、切触或高次面元直接终止几何构造；
- 用随网格尺度变化的局部剖分获得充足初值；
- 避免将两个很近的真实交点错误合并；
- 对无法确认根数的少量退化边，使用端点独立分类得到奇偶性。

非目标：

- 证明每个原始 NURBS span 只有一个根；
- 计算切触阶数或根的代数重数；
- 为任意高次 NURBS 面元构造昂贵的完整凸包求交算法；
- 强迫后续 KFBI 界面格式同时处理一条边上的全部交点。

## 2. 核心语义

必须区分两个概念：

1. `has_interface_crossing`：网格边上存在一个或多个几何交点；
2. `changes_inside_outside`：边两端的内外状态不同。

若所有交点都是明确横截，则

```text
changes_inside_outside = transverse_crossing_count % 2 == 1
```

上述公式需要按 component 分别计算。若当前 Domain 保留 component 标签，
则一条边只要有任一 component 的奇偶状态变化就必须阻断 flood；只有在纯二值
inside/outside 模式下，才只比较两个端点的二值状态。例如一条边从 component A
内部经过外部到达 component B 内部，总横截数为偶数，但 component 标签已经改变。

切触不改变奇偶性。存在近切触、近根簇或其他歧义时，不计算切触阶数，
而是独立判断两个端点：

```text
changes_inside_outside = inside(endpoint_a) != inside(endpoint_b)
```

当前保留 component 标签的 Flood labeling 使用 `changes_component_membership`
作为 barrier；纯二值模式使用 `changes_inside_outside`。同一 component 上的偶数个
横截交点不建立 label barrier，但不同 component 的奇偶状态同时变化时仍需阻断。
无论是否建立 barrier，只要存在交点，该边都是 interface edge。

## 3. 结果数据结构

底层 `NurbsElementIntersectionResult3D::roots` 和曲面层
`NurbsSurfaceIntersectionResult3D::crossings` 继续保存数组。
Cartesian edge 接口不再返回单个 `optional<Crossing>`，而返回摘要：

```cpp
struct NurbsCartesianEdgeIntersections3D {
    std::vector<NurbsSurfaceCrossing3D> crossings;
    std::vector<AmbiguousRootCluster3D> ambiguous_clusters;
    std::vector<int> toggled_components;
    int confirmed_transverse_count = 0;
    bool has_near_tangent_candidate = false;
    bool root_count_known = true;
    bool parity_known_from_roots = true;
    bool changes_inside_outside = false;
    bool changes_component_membership = false;
};
```

`crossings` 只保存已经确认的物理根；不能确认是一个根还是多个根的候选保存在
`ambiguous_clusters`，不能把候选数量伪装成物理根数。即使 `root_count_known` 为假，
端点独立分类仍可使 point classification 成功。

Domain 采用连续交点存储，并为每条边保存范围：

```cpp
struct EdgeCrossingRange3D {
    int begin = 0;
    int count = 0;
    int confirmed_transverse_count = 0;
    bool changes_inside_outside = false;
    bool changes_component_membership = false;
};
```

每条边还保存稀疏的 `toggled_components`。当前按 component 标记的 Domain 使用
`changes_component_membership` 建立 flood barrier；纯二值 Domain 使用
`changes_inside_outside`。

现有单交点查询改为交点范围查询。若旧的下游算法明确要求单交点，
兼容接口只能在 `count == 1` 时返回该交点，不能静默选择第一个根。

## 4. 随网格尺度变化的曲面局部化

保留当前精确的有理 de Casteljau 细分：

- 原始 NURBS 先转换为有理 Bézier span；
- BVH 查询叶面的最大物理尺寸为 `2h`；
- 细分只改变参数覆盖，不近似或改变原始 NURBS 几何；
- 每个叶面保留 patch、原始 element、局部参数盒和拓扑来源。

`2h` 的职责是提供与网格匹配的搜索尺度和 Newton 初值覆盖，
不用于假设一个叶面只能含有一个交点。

## 5. 分层初筛

对每条网格边和候选叶面依次执行：

1. BVH/AABB 排除；
2. 控制点在线方向和两个正交方向上的投影区间排除；
3. 控制点数不超过 16 时，使用现有完整控制凸包分离检查；
4. 仍不能排除时进入采样和局部求解。

所有排除条件必须是单向安全的：允许保留无交候选，不允许删除真实有交候选。
AABB 相交和局部最近点正距离都不是无交证明。

对于超过 16 个控制点的高次面元，不做组合式完整凸包检查。
采用投影初筛、局部细分以及最终端点分类备用路线，保证计算量有界。

## 6. 每个 `2h` 叶面的等参数预采样

在建立 BVH 叶面时预计算固定 `4 x 4` 等参数采样：

```text
u_i = u_0 + i (u_1-u_0) / 3
v_j = v_0 + j (v_1-v_0) / 3
```

每个采样保存 `(u, v, point)`。对具体网格边只执行点到有限线段的投影：

```text
t = clamp(((point-a) dot d) / (d dot d), 0, 1)
r2 = |point - (a+t d)|^2
```

投影开销远低于 Newton；曲面采样点可被所有网格边复用。

不能简单选全局最近的若干点，因为它们可能集中在同一个极值附近。
种子选择规则为：

- 找出采样网格上的离散局部极值候选；
- 按投影距离排序；
- 在 `(u,v)` 中执行非极大值抑制，保证种子参数分散；
- 每个叶面最多保留 4 个采样种子；
- 若没有离散极值，至少保留全局最近采样点；
- 三角初筛给出的交点种子始终保留并具有最高优先级。

## 7. 局部求解与细分

每个种子执行线段投影后的 `(u,v)` 局部求解。结果分为：

- 距离小于接触容差：作为交点/接触候选，再由 Newton 修正；
- 距离为正：仅表示该局部种子没有发现交点，不能排除整个叶面；
- 迭代失败或病态：保留最佳状态作为子盒种子。

候选盒的主局部细分深度为 4。达到深度 4 后，如果局部平面性差、
多个种子结果不一致或控制范围仍无法排除，只对该困难盒补充细分最多 2 层。
其他盒不增加深度。

找到一个根后不能结束原始 span 或其他候选盒；所有候选盒继续处理，
最后在曲面层统一汇总根。

删除“完整原始 span 单根证明”作为 Cartesian 分类的前置要求，包括：

- 原始 span 上的切触唯一性证明；
- 找到一个唯一横截根后跳过其他候选区域的假设；
- 多个交点直接报告网格分辨率不足的策略。

保留控制凸包分离证书，因为它仍用于安全排除无交盒。

## 8. 记录线面距离驻点

为保护两个很近的真实交点，不能只记录局部最小距离点。
对投影后的残差 `R = S(u,v)-L(t(u,v))`，记录满足

```text
R dot S_u = 0
R dot S_v = 0
```

的驻点。在线段内部，投影同时满足 `R dot d = 0`。
驻点可以是局部最小值、局部最大值或鞍点。

内部结构为：

```cpp
struct LineSurfaceStationaryPoint3D {
    int patch_index;
    int component;
    double u, v, t;
    Eigen::Vector3d surface_point;
    double distance;
    double tangent_measure; // abs(n dot unit_d)
    ParameterBox2D source_box;
};
```

若同一光滑曲面分支上存在两个交点，局部带符号距离由 Rolle 定理在二者之间
具有驻点。对于疑似重复的近根对，在两根之间用参数中点补做驻点求解。
满足以下条件的驻点是“两个根确实不同”的强证据：

```text
t1 < t_stationary < t2
abs(n dot unit_d) 接近零
stationary distance 明显大于几何容差
```

驻点不存在不能单独证明两个候选是同一个根。

## 9. 根聚类和去重

根按 component 和线段参数 `t` 排序。每个根根据残差和局部条件数建立数值
不确定区间，而不是使用 `C h` 作为合并容差。推荐的横截根参数误差尺度为：

```text
delta_t ~ residual / (segment_length * abs(n dot unit_d))
```

实际实现需加入机器精度下限和近切触保护。

去重顺序：

1. 不确定区间不重叠：保留为不同根；
2. 区间重叠但两根之间存在可分辨驻点：保留为不同根；
3. 同一叶面不同种子收敛到相同参数根：合并；
4. 同一 patch 且位于共享细分边界的同一参数根：合并；
5. 不同 patch 只有在拓扑共享边/顶点、物理位置和 `t` 一致时合并；
6. 不同 component 永不合并；
7. 仍无法确认时形成歧义交点簇，不强制删除候选。

合并时保留残差最小的数值代表，并保存全部来源信息。

### 非 G1 边

非 G1 边上带符号距离不可微，不能使用光滑驻点判据：

- 映射到同一共享拓扑边参数、物理位置和 `t` 的双侧报告是同一交点；
- 位于不同单侧面且误差区间可分离的结果是不同交点；
- 不能确认时保留歧义簇，并使用端点分类确定奇偶性；
- 双侧法向不平均，交点标记为 feature-edge contact。

## 10. 切触和奇偶性

不计算切触阶数。`abs(n dot unit_d)` 只用于发现明显横截和触发歧义备用路线，
不单独决定根的代数重数。

- 明显横截根参与横截数量奇偶计数；
- 近切触候选保留在几何交点集合中；
- 存在近切触或歧义根簇时，独立分类边的两个端点；
- 端点分类使用已有的多条确定性非退化射线，遇到切触则更换射线；
- 端点落在曲面上仍是明确的输入退化，需要报告。

端点分类结果应按 Cartesian node 缓存，避免多条歧义边重复发射射线。

## 11. 容差策略

搜索尺度随 `h` 变化：

- 最大 BVH 叶面尺寸 `2h`；
- 固定 `4 x 4` 叶面采样对应的物理采样间距；
- 候选盒范围和局部补充细分覆盖。

几何判定容差不随 `h` 变化：

- Newton 根残差；
- 接触距离；
- patch seam 几何一致性；
- 驻点收敛；
- 根的不确定区间和去重。

统一物理几何容差使用模型尺度 `L` 和可配置的 CAD/模型容差：

```text
epsilon_x = max(model_tolerance,
                64 * machine_epsilon * L)
```

默认 `model_tolerance` 可取 `1e-12 * L`，但不能覆盖调用者提供的更大 CAD 容差。

局部参数容差由 `epsilon_x` 和曲面 Jacobian 换算。`h` 只决定“搜索多细”，
不能决定“两个根是否相同”。

## 12. 失败和备用语义

以下情况不再使整条 Cartesian edge 失败：

- 多个交点；
- 偶数个横截交点；
- 可记录的切触候选；
- 高次面元未获得完整凸包分离证书；
- 根候选数量存在歧义，但两个端点可独立分类。

只有以下情况报告无法分类：

- 网格节点落在曲面上；
- surface 与网格边发生区间重合；
- 两个端点经过全部备用射线仍无法分类；
- NURBS 拓扑或 component 定义自相矛盾。

## 13. 诊断量

新增或调整诊断：

- 每条边原始根候选数、最终根簇数和保存交点数；
- 多交点边数、偶数交点边数和奇数交点边数；
- `4 x 4` 采样种子数、去重种子数和每叶面最大种子数；
- 驻点求解次数及由驻点保护的近根对数量；
- 同叶面、同 patch、G1 seam 和非 G1 seam 去重次数；
- 近切触/歧义边数及端点分类备用次数；
- 高次面元候选数、投影排除数和备用分类数；
- 主深度与困难盒补充深度的最大值。

## 14. 测试和验收标准

至少覆盖：

1. 一条边零个、一个、两个和三个横截交点，验证交点保存及奇偶 barrier；
   其中同时覆盖“同一 component 两根”和“不同 component 各一根”；
2. 光滑曲面上的两个近根，中间驻点阻止错误合并；
3. 相邻叶面重复报告同一根，只保留一个物理交点；
4. 不同 component 的近根不合并；
5. G1 seam 上同一根合并；
6. 非 G1 seam 上同点双侧报告合并、两个近根不误合并；
7. 二阶切触和高阶零导数横截都通过端点标签获得正确奇偶性，
   不计算切触阶数；
8. 双三次完整凸包快速路径和双四次以上的有界工作路径；
9. triangle seed 开关前后交点集合及节点标签一致；
10. `N=32,64,128` 加密时节点内外标签稳定，去重结果不因 `h` 容差改变；
11. L 形、圆柱、环面以及包含非 G1 边的现有三维几何回归；
12. 所有交点残差满足固定几何容差，主深度不超过 4，补充深度不超过 6。

## 15. 预计修改范围

- `src/geometry/nurbs_bezier_intersection_3d.*`
  - 多种子、驻点记录、候选盒处理和根来源信息；
- `src/geometry/nurbs_bezier_segment_closest_point_3d.*`
  - 从仅返回最佳最近点扩展为可记录多个驻点结果；
- `src/geometry/nurbs_surface_intersector_3d.*`
  - `4 x 4` 叶面预采样、全部根汇总、拓扑感知聚类和 edge 摘要；
- `src/geometry/nurbs_cartesian_domain_3d.*`
  - interface edge 与 parity barrier 分离、多交点范围存储、端点标签缓存；
- `apps/native_nurbs_surface_3d_test.cpp`
  - 上述单元和回归测试；
- `apps/neumann_exterior_zero_trace_3d.cpp`
  - 新诊断输出及 CSV 字段。
