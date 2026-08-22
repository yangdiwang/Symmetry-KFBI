# KFBI3D 拓扑自由度、局部约束消元与外迹残差投影：Codex 实现规格

**文档版本：** 1.0
**日期：** 2026-08-15
**目标读者：** Codex / 负责实现 KFBI3D 密度空间、约束消元、外迹测试与 GMRES 接口的开发者
**规范关键字：** 本文中的 **MUST / SHOULD / MAY** 分别表示必须、建议和可选。

---

## 0. Codex 执行契约

Codex 在实现本规格时必须遵守以下边界：

1. **只改变密度自由度空间、约束预处理和残差投影接口。** 不得改变 Cartesian Poisson 离散、direct coefficient Cauchy 的物理公式、spread/restrict 的符号约定。
2. **生产路径不得构造或存储全局稠密零空间。** 允许在小规模测试中构造 global-SVD oracle。
3. **局部块使用完整稠密 SVD；全局约束、基映射与矩阵乘法保持稀疏。**
4. **所有几何分支在 GMRES 前固定。** Patch owner、first-hit、same-sheet nearest crossing、Q10 stencil、法向采样点均不得依赖 Krylov 向量。
5. **每一个 Krylov 方向必须满足齐次约束。** 必须验证 `||C @ E||_inf`。
6. **仿射特解必须单独进入右端。** GMRES 的 `matvec` 中不得重复加入 `p` 或已知边界 jump。
7. **完整物理外迹必须保留并报告。** 不得只报告投影后的代数残差。
8. **任何约束不相容都必须显式报告。** 禁止静默丢弃“零行但非零右端”或用无标记最小二乘掩盖物理冲突。
9. **Neumann 与 Dirichlet 必须共享同一套 `AffineReduction` 和 `TraceProjector` API；** 两者只在未知 jump、约束类型、观测外迹和 gauge 上不同。
10. Codex 完成后必须输出：修改文件、测试命令、关键数值诊断、与 reference 路径的 A/B 结果。

---

## 1. 依据的当前代码与实现状态

### 1.1 当前已实现并应复用

| 功能 | 当前文件 / 函数 |
|---|---|
| T 型长边-短边拓扑 | `src/tjunction_topology.py`: `EdgeUse`, `JunctionSegment` |
| Broken patch 系数空间 | `build_broken_atlas()` |
| T-aware 约束组装 | `build_tjunction_constraints()` |
| Vertex/T-star 与 edge 分块 | `tjunction_blocks()` |
| 顺序局部仿射 SVD 消元 | `affine_eliminate_local_svd()` |
| T-junction Neumann 驱动 | `integrated_tjunction_neumann_driver.py` |
| 非 T 局部 edge/vertex 对照 | `integrated_local_neumann_driver.py` |
| 外迹投影 | `ReducedProjector` |
| Neumann GMRES | `solve_reduction()` |
| 未知密度解析 jet | `direct_coefficient_cauchy.py`: `density_jet_arrays()` |
| 已知 Neumann 数据 jet | `known_neumann_jets()` |
| shared-side Q10 restrict | `shared_side_restrict_study.py` |
| 3+3 法向恢复 | `normal_sample_count_study.py`; `rho/h = ±{0.5,0.75,1.5}` |
| first-hit 后复用 same-sheet 最近 spread crossing | `remap_firsthits_to_nearest_spread()` |

### 1.2 当前代码真实状态

当前 T-junction 路径具有以下特征：

- 已完整接入 **Neumann**；Dirichlet T-space 尚未实现。
- `C` 在组装后仍为 dense NumPy 数组；局部计算时临时转 CSR。
- `E`、`H` 使用 sparse CSR；局部 `A_loc`、`Z_loc` 使用 dense NumPy。
- Vertex/T-star block 实际先于 edge-interior block 执行。
- 齐次空间由局部 SVD 得到；若 `p` 的约束残差较大，当前驱动使用一次 sparse global LSMR 修正特解。
- 当前 Neumann direct coefficient Cauchy 已完成；Dirichlet 仍可复用现有 legacy operator，后续再切换为 direct coefficient jet。

### 1.3 本规格要求 Codex 完成的目标

Codex 应在保持现有数值行为的基础上完成：

- 将 `C` 改为真正的 sparse CSR 组装；
- 抽象统一的 `AffineReduction`；
- 抽象统一的 `TraceProjector`；
- 保留 T-junction、Vertex/T-star、smooth seam、feature edge；
- 增加 Dirichlet 的 smooth T-junction 密度空间；
- 将非齐次特解的全局 LSMR fallback 改为“约束支撑图 connected component 局部求解”，全局 LSMR仅保留诊断模式；
- 增加完整的约束相容性、rank、投影 rank、线性性和 A/B 测试。

---

## 2. 数学对象与统一记号

### 2.1 Patch 局部基函数与 broken coefficient

第 `p` 个 patch 上使用局部三次 spline/NURBS 基函数：

```text
R^(p)_{i,j}(u,v),     0 <= i < n_u(p), 0 <= j < n_v(p)
```

局部密度：

```text
mu_h^(p)(u,v) = sum_{i,j} c^(p)_{i,j} R^(p)_{i,j}(u,v)
```

所有 patch 完全独立时，形成 broken coefficient 向量：

```text
c_full in R^(N_full)
```

当前统一 `ncoef = n` 的代码使用：

```text
global_index(p,i,j) = p*n*n + i*n + j
```

Codex 必须把索引封装成 `PatchDofLayout`，禁止在新代码中散落 `p*n*n+i*n+j`。

### 2.2 可选的基础展开 A0

为兼容两种现有路径，定义：

```text
c_full = A0 * y0
```

- T-junction broken atlas：`A0 = I`；
- 现有 conforming atlas：`A0 = smooth_map(at)`，先包含 union-find/C1 reduction。

局部约束消元在 `y0` 空间中产生：

```text
y0 = p + E*z
```

因此最终完整 patch 系数为：

```text
c_full = c_p + G*z
c_p    = A0*p
G      = A0*E
```

后续 direct jet、spread、trace basis 和最终密度都必须使用 `c_p` 与 `G`，不得混用 `p/E` 和 full coefficient 空间。

### 2.3 最终密度基函数

设 broken 原始基函数统一编号为 `phi_i(x)`。最终合法密度写成：

```text
mu_h(x; z) = mu_p(x) + sum_alpha z_alpha Psi_alpha(x)
```

其中：

```text
mu_p(x)     = sum_i (c_p)_i phi_i(x)
Psi_alpha(x)= sum_i G_{i,alpha} phi_i(x)
```

`G` 的每一列就是一个最终合法全局分片基函数。

---

## 3. 两类边界条件的未知空间

### 3.1 Neumann

使用 exterior-zero formulation：

```text
J0 = [u]       = mu_N        unknown
J1 = [d_n u]   = g_N         known
```

目标条件：

```text
gamma_h^+ E_h(mu_N, g_N) = 0
```

即外侧值迹为零。

Neumann 的密度是值 jump，因此：

- 在所有参数 seam 上必须保持物理 C0；
- 在 smooth sheet seam 上还施加物理 C1；
- 在物理 C0 feature edge 上施加两侧 Neumann-compatible transverse jet；
- 合法集合一般是仿射空间，`p != 0`；
- 需要一个 scalar mean/gauge 条件。

### 3.2 Dirichlet

使用 exterior-zero formulation：

```text
J0 = [u]       = g_D         known
J1 = [d_n u]   = mu_D        unknown
```

目标条件：

```text
N_h^+ E_h(g_D, mu_D) = 0
```

即外侧法向迹为零。

Dirichlet 未知是相对于各面法向定义的法向 jump，因此：

- 仅在同一 physical smooth sheet 的 seam/T-junction 上连接；
- smooth seam 上施加 density 的物理 C0 与需要的物理 C1；
- **不得**跨真实 C0 feature edge 强制 `mu_D` 相等；两侧保持独立；
- feature vertex 只按 smooth-sheet 分组，不做跨折面的 value/derivative coupling；
- 当前多数 Dirichlet 约束为齐次，通常 `p = 0`；
- 不需要 Neumann mean/gauge。

> 当前源代码只实现了 conforming Dirichlet 与 exterior normal trace。Dirichlet T-junction 是本规格的目标扩展，必须按上述 smooth-sheet 规则实现。

---

## 4. 必须实现的数据结构

```python
@dataclass(frozen=True)
class PatchDofLayout:
    patch_id: int
    n_u: int
    n_v: int
    offset: int

    def global_index(self, i: int, j: int) -> int: ...
    def decode(self, global_index: int) -> tuple[int, int]: ...
```

```python
@dataclass(frozen=True)
class EdgeUse:
    patch: int
    edge: str              # u0, u1, v0, v1
    s0: float
    s1: float
    reverse: bool

    def local_t(self, s: float) -> float: ...
    def macro_s(self, t: float) -> float: ...
    @property
    def scale_dt_ds(self) -> float: ...
```

```python
@dataclass(frozen=True)
class JunctionSegment:
    macro: str
    label: str
    a: EdgeUse
    b: EdgeUse
    s0: float
    s1: float
    smooth: bool
```

```python
@dataclass(frozen=True)
class ConstraintMeta:
    kind: str              # c0, smooth_c1, feature_jet, ...
    macro: str
    segment: str
    cell: tuple[float, float]
    mode: int
    star: tuple[float,float,float] | None
    side: str              # both, a, b
```

```python
@dataclass
class ConstraintSystem:
    C: scipy.sparse.csr_matrix
    d: numpy.ndarray
    meta: list[ConstraintMeta]
```

```python
@dataclass
class ConstraintBlock:
    kind: Literal['vertex', 'edge']
    key: object
    row_ids: numpy.ndarray
```

```python
@dataclass
class AffineReduction:
    p: numpy.ndarray                    # base-coordinate particular
    E: scipy.sparse.csr_matrix          # base-coordinate homogeneous map
    C: scipy.sparse.csr_matrix
    d: numpy.ndarray
    records: list[dict]
    particular_residual_linf: float
    homogeneous_residual_linf: float
```

```python
@dataclass
class TraceProjector:
    Bz: scipy.sparse.csr_matrix | numpy.ndarray
    sqrt_w: numpy.ndarray
    # small path: Q,R; scalable path: sparse factorization of M=Bz.T W Bz

    def project(self, trace: numpy.ndarray) -> numpy.ndarray: ...
```

---

## 5. 物理拓扑与分块顺序

### 5.1 拓扑实体

实现必须区分：

- `smooth_seam`：参数 patch 不同，但属于同一 physical smooth sheet；
- `feature_edge`：真实 C0 折边或凹棱；
- `macro_edge`：一个物理接口，可由一条长边和多条短边表示；
- `smooth_T`：长边-短边的光滑 T 点；
- `feature_T`：长 feature edge 与多个短 feature edge 汇合；
- `physical_corner`：普通多面体角点；
- `face_interior`：不触及 skeleton 约束的 patch 内部。

### 5.2 Block 顺序

生产实现必须使用：

```text
1. 所有 physical Vertex / T-star blocks
2. 所有 edge-interior blocks
```

原因：共享 corner/first-strip 系数必须先由一个 star 联合处理；随后 edge 中段只能在 star 合法空间中继续消元。

### 5.3 Vertex/T-star 行收集

对 overlay segment 两端附近的 `star_layers` 个 cell：

- 根据实际物理坐标 `round(X, tol_digits)` 生成 star key；
- 来自所有入射 segment、patch、C0、smooth C1、feature jet 的行进入同一个 block；
- 不允许一条 edge 单独处理到端点再由另一条 edge 覆盖同一 coefficient；
- 默认不添加独立 point row，除非诊断表明 moment rows 无法控制端点极限。

### 5.4 支撑图改进（目标实现）

固定 `star_layers` 只能作为初始候选。Codex SHOULD 增加 constraint-support graph：

- 约束行为一类节点；
- 当前 reduced basis column 为另一类节点；
- 若 `A[row,col] != 0`，则连边；
- 对共享 active columns 的 block 自动合并为 connected component；
- 确保任何共享 `c11`/corner strip 的约束进入同一局部 SVD。

---

## 6. T-junction overlay 与积分矩约束

### 6.1 Macro 参数

`EdgeUse` 将局部 `t in [0,1]` 映射到物理 macro 坐标 `s in [s0,s1]`。一条长边可覆盖 `[0,1]`，两个短边分别覆盖 `[0,1/2]`、`[1/2,1]`。

### 6.2 Overlay cell

将两侧局部 knot break 映射到 macro `s`，取并集：

```text
K_overlay = union(K_a_in_macro, K_b_in_macro)
I_r = [s_r, s_{r+1}]
```

每个 cell 上使用 `P_0,...,P_p` Legendre modes。当前 cubic 使用 `p=3`，积分可使用 5 点 Gauss-Legendre。

### 6.3 C0 行

对每个 segment/cell/mode：

```text
integral_{I_r} P_k(s_hat) [mu_a(X(s)) - mu_b(X(s))] |X'(s)| ds = 0
```

每个基函数系数为该基函数在积分泛函下的值。组装时只写入涉及 patch 的 global columns。

### 6.4 Smooth C1 行

公共物理切向：

```text
tau = (dX/ds) / |dX/ds|
```

参考侧法向 `n_a` 给出公共 co-normal：

```text
nu = normalize(cross(n_a, tau))
```

对 patch 参数化 `J=[X_u, X_v]`：

```text
xi = solve(J.T @ J, J.T @ nu)
D_nu R = xi_u R_u + xi_v R_v
```

约束：

```text
integral P_k [D_nu mu_a - D_nu mu_b] |X'| ds = 0
```

禁止直接比较 raw `u` 或 `v` 导数。

### 6.5 Neumann feature-edge 非齐次行

已知：

```text
n_a, n_b, g_a, g_b
N = [n_a, n_b]
G_perp = N (N.T N)^(-1) [g_a, g_b]^T
q_a = m_a . G_perp
q_b = m_b . G_perp
```

约束：

```text
integral P_k [D_{m_a} mu_a - q_a] |X'| ds = 0
integral P_k [D_{m_b} mu_b - q_b] |X'| ds = 0
```

必须检查 `cond(N.T @ N)`；近乎平行法向时必须报错或切换到 smooth-seam 逻辑，不得继续求解病态 2x2 系统。

### 6.6 Dirichlet T-junction 约束

Dirichlet 的未知 `mu_D=J1`：

- smooth T：组装 C0 与 physical C1；
- feature T：不跨 feature edge 组装 C0 或 transverse jet；各 smooth sheet 的子边独立；
- feature T 点处按 `sheet_id` 分组建立多个独立 star，不允许将不同面法向对应的 `mu_D` 合并。

---

## 7. 稀疏约束组装要求

当前代码先创建 full dense row。目标实现 MUST 直接组装 COO/CSR：

```python
row_ids: list[int]
col_ids: list[int]
values: list[float]
rhs: list[float]
```

每次局部 patch row 只输出：

```python
(local_col_ids, local_values)
```

然后加 patch offset，写入全局 COO。

接口：

```python
def assemble_constraints(...) -> ConstraintSystem:
    ...
    C = scipy.sparse.coo_matrix(
        (values, (row_ids, col_ids)),
        shape=(n_rows, n_base_coords),
    ).tocsr()
    C.sum_duplicates()
    C.eliminate_zeros()
    return ConstraintSystem(C=C, d=np.asarray(rhs), meta=meta)
```

零行规则：

- `||row|| <= zero_tol` 且 `|rhs| <= rhs_tol`：删除；
- `||row|| <= zero_tol` 且 `|rhs| > rhs_tol`：立即抛出 `ConstraintIncompatibility`。

---

## 8. 顺序局部仿射 SVD 消元

### 8.1 始终维护的表达式

```text
y0 = p + E*z
```

初始化：

```python
p = zeros(n_base)
E = sparse_identity(n_base)
```

### 8.2 处理一个 block

当前 block 原约束：

```text
C_B y0 = d_B
```

代入当前仿射表达：

```text
A z = b
A = C_B E
b = d_B - C_B p
```

代码必须保持 `C_B @ E` 为 sparse，先从 sparse 结果提取 active columns，再将局部矩阵转 dense：

```python
A_sp = C[row_ids] @ E
cand = np.unique(A_sp.indices)
A_loc = A_sp[:, cand].toarray()
b = d[row_ids] - C[row_ids] @ p
```

### 8.3 零行与自由度耗尽检查

对局部行：

- `||A_row|| <= tol` 且 `|b_row| <= tol_rhs`：该约束已被前序 block 满足，删除；
- `||A_row|| <= tol` 且 `|b_row| > tol_rhs`：当前剩余空间无法满足该约束，必须触发 incompatibility fallback；不得只记录后继续。

若 `cand` 为空且 `||b||>tol_rhs`，同样是无解。

### 8.4 行缩放与 dense SVD

```text
D_rr = ||A_loc[r,:]||_2
A_s = D^{-1} A_loc
b_s = D^{-1} b
A_s = U Sigma V^T
```

数值秩：

```text
rank = count(sigma_i > sigma_0 * rank_tol)
```

局部最小范数特解：

```text
x_p = V_r Sigma_r^{-1} U_r^T b_s
```

局部完整零空间：

```text
Z_loc = V_0
```

必须记录：`rows`, `active_rows`, `support_cols`, `rank`, `local_nullity`, `residual`, `homogeneous_residual`, `remaining`。

### 8.5 构造局部坐标变换 H

设当前坐标分为 `untouched` 和 `cand`：

```text
y = w_p + H z_new
```

其中：

- untouched 行为 identity；
- active 行由 `Z_loc` 填入；
- `w_p[cand] = x_p`。

`H` 必须是 sparse CSR。

### 8.6 特解与零空间复合

```text
p_new = p_old + E_old w_p
E_new = E_old H
```

这两个公式是实现的核心不变量：

- 新特解只能沿旧零空间移动，因此不破坏旧非齐次约束；
- 新零空间是旧零空间的子空间，因此不重新引入非法方向。

### 8.7 Block 结束后的不变量

处理完前 `k` 个 block 后必须满足：

```text
C_processed p ~= d_processed
C_processed E ~= 0
rank(E) = number_of_columns(E)
```

---

## 9. 不相容与自由度不足的处理策略

定义归一化相容残差：

```text
eta = ||A_loc x_p - b||_2 / (||b||_2 + eps)
```

### 9.1 相容且有零空间

```text
eta <= consistency_tol, rank < n_active
```

正常保留 `n_active-rank` 个局部自由度。

### 9.2 相容且局部自由度全部确定

```text
eta <= consistency_tol, rank == n_active
```

接受唯一局部特解，`Z_loc` 为空。这不是错误。

### 9.3 不相容

```text
eta > consistency_tol
```

按以下顺序处理：

1. 合并共享 active columns 的相邻 edge/star blocks；
2. 扩大 vertex support / star layers；
3. 将多个 `g_N` 投影到公共 edge/vertex jet 空间后重新生成 `d`；
4. 对 density trace 做 knot insertion / 增加模态；
5. 按优先级放松约束：保留 C0，保留沿边 C1，最后放松 transverse feature jet；
6. 只有显式开启 `allow_weighted_least_squares=True` 时才返回近似特解，并在结果中标记 `constraints_exact=False`。

生产模式不得静默使用 global LSMR。当前 global LSMR 仅保留为 `reference_particular_solver`。

---

## 10. AffineReduction 输出与基函数解释

预处理输出：

```text
p in R^(n_base)
E in R^(n_base x n_red)
```

完整系数：

```text
c_p = A0 p
G   = A0 E
c_full(z) = c_p + G z
```

最终第 `alpha` 个密度基函数：

```text
Psi_alpha = sum_i G[i,alpha] phi_i
```

Codex 必须提供诊断 API：

```python
def reduced_basis_support(G, alpha, tol=1e-13) -> list[tuple[int,float]]: ...

def full_coefficient_formula(c_p, G, i, tol=1e-13) -> dict:
    # c_i = c_p[i] + sum_j G[i,j] z_j
    ...
```

---

## 11. Direct coefficient Cauchy 接口

### 11.1 Neumann

完整系数：

```text
c = c_p + G z
```

在 crossing 处直接评价：

```text
mu, mu_u, mu_v, mu_uu, mu_uv, mu_vv
```

再经几何 Jacobian/二阶链式法则得到：

```text
J0, J0_s, J0_t, J0_ss, J0_st, J0_tt
```

已知：

```text
J1 = g_N, J1_s, J1_t
```

结合曲率与 PDE 闭合空间二阶 Cauchy P2。不得从 surface samples 重新拟合未知 density jet。

### 11.2 Dirichlet

当前可先复用 legacy operator。若 Codex 同时实现 direct Dirichlet jet，则：

```text
J0 = g_D             known, need tangential derivatives through second order
J1 = mu_D            unknown, need value and tangential first derivatives
```

仍使用 `c=c_p+Gz` 解析评价未知 `J1`。

---

## 12. Restrict 与完整外迹恢复

生产默认：

```text
shared-side Q10
+ rho/h = ±{0.5,0.75,1.5}
+ cubic least-squares normal recovery
+ parity / first-hit
+ same-sheet nearest spread-crossing Cauchy reuse
```

六个法向样本拟合：

```text
p(rho) = a0 + a1 rho + a2 rho^2 + a3 rho^3
```

恢复：

```text
value trace       = a0
normal derivative = a1 / h
```

Neumann 观测：外侧值迹 `t_v = u_h^+(q_i)`。
Dirichlet 观测：外侧法向迹 `t_n = d_n u_h^+(q_i)`。

完整外迹向量必须保留：

```text
t in R^(n_test)
```

不得在 restrict 内直接压缩为 GMRES 维数。

---

## 13. 最终外迹/法向迹投影到迭代空间

### 13.1 构造 reduced density basis 的测试矩阵

设：

```text
B_test : full patch coefficient basis at independent geometric test points
B0     = B_test A0
Bz     = B0 E = B_test G
```

`Bz[:,alpha]` 是最终合法密度基函数 `Psi_alpha` 在全部测试点的值。

**注意：** `Bz` 不是“每个密度基函数经过 KFBI 后的外迹响应矩阵”。KFBI 响应保持 matrix-free。

### 13.2 加权最小二乘投影

测试权重：

```text
W = diag(w_i)
```

寻找 `r_z`：

```text
r_z = argmin_r || W^(1/2) (Bz r - t) ||_2
```

小中规模当前实现：

```text
W^(1/2) Bz = Q R
r_z = R^(-1) Q^T W^(1/2) t
```

必须检查：

```text
rank(Bz) == n_red
cond(R) <= configured_limit
```

大规模 SHOULD 使用：

```text
M = Bz^T W Bz
M r_z = Bz^T W t
```

配合 sparse Cholesky / SPD factorization；不得显式形成 dense `Bz` 或 dense `M`。

### 13.3 Projector API

```python
class TraceProjector:
    def project(self, full_trace: np.ndarray) -> np.ndarray:
        """Return coordinates in the same reduced density basis used by GMRES."""
```

同一个 projector 用于：

- Neumann 的 exterior value trace；
- Dirichlet 的 exterior normal trace。

区别只在传入的 `full_trace` 内容。

---

## 14. Neumann GMRES 的精确实现

定义 operator 数据：

```text
S           unknown J0 spread
known_j1    known g_N spread
Rv          grid potential -> exterior value trace
T0v         unknown J0 direct correction -> value trace
known_j1_v  known J1 direct correction -> value trace
```

### 14.1 固定仿射基场

```python
c_p = A0 @ p
base_potential = poisson.solve(S @ c_p + known_j1)
base_trace = Rv @ base_potential + T0v @ c_p + known_j1_v
rhs_trace = -projector.project(base_trace)
```

### 14.2 齐次 matrix-free matvec

```python
def apply_neumann_homogeneous(z):
    c = G @ z
    potential = poisson.solve(S @ c)
    full_trace = Rv @ potential + T0v @ c
    return projector.project(full_trace)
```

### 14.3 Mean/gauge augmentation

```text
mean_p = mean_vec^T c_p
mean_z = G^T mean_vec
const_projected = projector.project(ones(n_test))
```

线性系统：

```text
[ P T_v G     P 1 ] [z]       [ -P t_p ]
[ mean_z^T     0  ] [lambda] = [ -mean_p ]
```

最终恢复：

```python
c_final = c_p + G @ z
potential = poisson.solve(S @ c_final + known_j1)
full_trace = Rv @ potential + T0v @ c_final + known_j1_v
```

必须同时报告 `||projected_residual||`、`||full_trace||_inf`、mean residual。

---

## 15. Dirichlet GMRES 的精确实现

定义 operator 数据：

```text
S           unknown J1 spread
known_j0    known g_D spread
Rn          grid potential -> exterior normal trace
T1n         unknown J1 direct correction -> normal trace
known_j0_n  known J0 direct correction -> normal trace
```

### 15.1 固定仿射基场

```python
c_p = A0 @ p
base_potential = poisson.solve(S @ c_p + known_j0)
base_normal_trace = Rn @ base_potential + T1n @ c_p + known_j0_n
rhs = -projector.project(base_normal_trace)
```

### 15.2 齐次 matrix-free matvec

```python
def apply_dirichlet_homogeneous(z):
    c = G @ z
    potential = poisson.solve(S @ c)
    full_normal_trace = Rn @ potential + T1n @ c
    return projector.project(full_normal_trace)
```

无 mean augmentation：

```text
P T_n G z = -P t_{n,p}
```

最终恢复并报告未投影完整法向迹。

---

## 16. 推荐模块划分

```text
kfbi3d/dof_layout.py
    PatchDofLayout
    base/full coefficient conversion

kfbi3d/topology.py
    EdgeUse, JunctionSegment, physical vertex/sheet IDs
    macro-edge and T-junction construction

kfbi3d/constraints.py
    sparse C0/C1/feature rows
    overlay quadrature
    constraint metadata

kfbi3d/reduction.py
    ConstraintBlock
    build_blocks
    affine_eliminate_local_svd
    component-local particular solver
    diagnostics

kfbi3d/density_basis.py
    AffineReduction
    lift(z), lift_homogeneous(z), full_formula(), basis_support()

kfbi3d/trace_projector.py
    TraceProjector
    dense QR backend
    sparse SPD backend

kfbi3d/gmres_operators.py
    NeumannTraceOperator
    DirichletNormalTraceOperator

kfbi3d/diagnostics.py
    pointwise constraints
    full trace diagnostics
    space comparison
```

现有函数名可暂时保留，但必须通过上述对象封装，避免 driver 直接操作散乱的 `p/E/C/d`。

---

## 17. 实现顺序（Codex 任务清单）

### Task 1: Sparse constraint assembly

- 将 `_global_row()` 替换为 sparse triplet 写入；
- `ConstraintSystem.C` 返回 CSR；
- 保持当前数值 rows/rank 不变。

### Task 2: AffineReduction abstraction

- 包装 `p/E/C/d/records`；
- 提供 `lift_base(z)`, `lift_full(z, A0)`, `particular_full(A0)`；
- 添加 shape/rank 检查。

### Task 3: Local SVD refactor

- 输入 CSR；
- `A_sp=C[ids]@E` 保持 sparse；
- 只将 `A_sp[:,cand]` 转 dense；
- 添加 inconsistency exception/fallback policy。

### Task 4: Support-graph components

- 先构建 physical star/edge blocks；
- 再按共享 active reduced columns 合并；
- 替代固定 star_layers 的遗漏风险。

### Task 5: Particular solve localization

- 对每个 connected component 求 affine particular；
- global LSMR 仅作为 reference；
- 输出每个 component 的 `eta`。

### Task 6: Unified TraceProjector

- 输入 `B_test`, `A0`, `E`, `weights`；
- 构造 `Bz`；
- 实现 QR 与 sparse-SPD 两后端；
- 加 rank/condition diagnostics。

### Task 7: Unified GMRES operators

- Neumann：value trace + gauge；
- Dirichlet：normal trace，无 gauge；
- base/right-hand side 与 homogeneous matvec 严格分离。

### Task 8: Dirichlet T-space

- smooth macro-edge/T-star：C0+C1；
- feature edge：不耦合不同 smooth sheets；
- 使用同一 `AffineReduction` 与 `TraceProjector`。

### Task 9: Regression and A/B

- local vs global SVD oracle；
- T vs conforming atlas；
- sparse vs old dense constraint assembly；
- QR projector vs sparse-SPD projector。

---

## 18. 必须通过的验收测试

### 18.1 索引测试

- 随机 `(patch,i,j)` 编码/解码 round-trip；
- 约束 row 非零 columns 必须只落在目标 patch slices。

### 18.2 约束组装测试

- C0 行直接评价与数值积分一致；
- smooth C1 使用同一个 world-space co-normal；
- feature `q_a/q_b` 与 90 度解析关系一致；
- T macro-edge 长边在两个 child segments 中引用同一 full coefficient columns。

### 18.3 消元不变量

每个 block 后检查：

```text
||C_processed p - d_processed||
||C_processed E||
number_of_columns(E_new) = number_of_columns(E_old) - rank(block)
```

### 18.4 空间等价性

小规模构造：

```text
p_global = lstsq(C,d)
E_global = null_space(C)
```

检查：

```text
principal_angle(range(E_local), range(E_global)) < 1e-8
particular distance modulo range(E_global) < 1e-8
```

### 18.5 投影测试

对随机 reduced vector `r`：

```text
t = Bz r
project(t) ~= r
```

检查 projector rank 和 condition。

### 18.6 GMRES 线性性

对随机 `z1,z2,a,b`：

```text
A(a z1 + b z2) ~= a A(z1) + b A(z2)
```

Neumann base field不得进入该测试的 homogeneous matvec。

### 18.7 完整外迹测试

- projected residual 达到容差；
- 未投影 full trace 同时下降；
- 禁止仅凭 projected residual 判定收敛。

### 18.8 数值 reference

至少保持当前量级：

- L-prism/local 与 global reference GMRES 相同或差不超过 1；
- T-junction N=32/64 迭代数不随 DOF 系统增长；
- `||C E||_inf` 接近机器精度；
- 刚体变换下 projector condition、GMRES 与场误差保持稳定。

---

## 19. 运行时诊断与结果 schema

每个运行必须输出：

```json
{
  "full_coefficients": 0,
  "reduced_coordinates": 0,
  "constraint_rows": 0,
  "constraint_rank": 0,
  "particular_solver": "component_local_svd",
  "particular_residual_linf": 0.0,
  "homogeneous_residual_linf": 0.0,
  "projector_rank": 0,
  "projector_condition": 0.0,
  "gmres_iterations": 0,
  "gmres_final_projected_residual": 0.0,
  "full_trace_linf": 0.0,
  "full_trace_meanfree_linf": 0.0,
  "mean_residual": 0.0,
  "block_records": []
}
```

Dirichlet 用 `full_normal_trace_linf` 替代 `full_trace_meanfree_linf`；无需 `mean_residual`。

---

## 20. 当前代码到目标代码的关键迁移规则

| 当前行为 | 目标行为 |
|---|---|
| `C=np.asarray(rows)` | 直接 COO/CSR 组装 |
| `C[ids]` dense 后临时 CSR | `C` 原生 CSR |
| 完整 `A=(C_B@E).toarray()` | 先 sparse 找 `cand`，仅 `A[:,cand]` dense |
| fixed `star_layers` | star + support graph component closure |
| global sparse LSMR 修正 `p` | component-local compatible particular；global 仅 reference |
| `ReducedProjector` 内 `E.toarray()` | dense QR 小规模；sparse SPD 大规模 |
| Neumann-only T-space | 增加 smooth-sheet Dirichlet T-space |
| driver 直接拼 `p/E` | `AffineReduction` 与 operator class |

---

## 21. 明确禁止的实现

Codex 不得：

- 对整个 `C` 做 production dense SVD；
- 分别对两条入射 edge 的端点做 SVD，再在 Vertex 覆盖 coefficient；
- 将 Vertex 零空间和 edge 零空间按列拼接 `[Z_V, Z_E]`；正确复合是 `E_new=E_old@H`；
- 在后续约束中继续使用已消元的原始自由度；正确转换是 `A=C_block@E`；
- 在 GMRES matvec 中加入 `p`、`known_j0` 或 `known_j1`；这些只进入右端；
- 用 density basis evaluation `Bz` 冒充 KFBI trace response matrix；
- 在 feature edge 上把 Dirichlet normal-density 跨不同法向面强行相等；
- 静默接受 nonzero RHS 的 zero constraint row；
- 只报告投影残差而不报告完整外迹。

---

## 22. 最终端到端流程

```text
Patch-local spline/NURBS basis
    -> broken full coefficient indexing
    -> physical topology: sheets / smooth seams / feature edges / T stars
    -> sparse C0 / smooth-C1 / feature-jet constraints
    -> Vertex/T-star blocks
    -> edge-interior blocks
    -> sequential local dense SVD inside sparse global maps
    -> y0 = p + E z
    -> c_full = c_p + G z
    -> direct coefficient density jets
    -> Cauchy closure + spread + Poisson solve
    -> shared-side Q10 + 3+3 + first-hit/nearest crossing restrict
    -> full exterior value trace (Neumann)
       or full exterior normal trace (Dirichlet)
    -> Bz = B_test G
    -> weighted trace projection into z coordinates
    -> matrix-free GMRES
    -> recover c_final and report full, unprojected trace
```

该流程的最终不变量是：

```text
C p ~= d
C E ~= 0
c_full(z) = A0 (p + E z)
projected residual dimension == number of columns(E)
full physical trace remains independently observable
```

---

## 附录 A：当前源代码映射

| 规范步骤 | 当前实现位置 |
|---|---|
| EdgeUse / macro interval | `src/tjunction_topology.py`, `EdgeUse` |
| Overlay breaks | `_overlay_breaks()` |
| 物理方向 derivative row | `_direction_row()` |
| C0/C1/feature rows | `build_tjunction_constraints()` |
| Vertex first block order | `tjunction_blocks()` 实际循环 |
| 局部 SVD | `affine_eliminate_local_svd()` |
| 特解复合 | `p = p + Eold @ wp` |
| 零空间复合 | `E = Eold @ H` |
| T reduction driver | `make_tjunction_reductions()` |
| Projector | `ReducedProjector` |
| Neumann base/matvec | `solve_reduction()` |
| Dirichlet normal-trace reference | `kfbi_feature_aware_pose_corrected.py`, Dirichlet solve block |
| Direct coefficient Neumann jet | `direct_coefficient_cauchy.py` |

## 附录 B：最小伪代码

```python
def preprocess(atlas, boundary_condition):
    A0 = build_base_expansion(atlas)
    cs = assemble_sparse_constraints(atlas, boundary_condition)
    blocks = build_vertex_then_edge_blocks(cs.meta)
    red = affine_eliminate_local_svd(cs.C, cs.d, blocks)
    assert max_abs(cs.C @ red.E) < homogeneous_tol
    projector = TraceProjector(atlas.Btest @ A0 @ red.E,
                               atlas.test_weights)
    return A0, red, projector
```

```python
def eliminate_block(C, d, block, p, E):
    A_sp = C[block.row_ids] @ E
    b = d[block.row_ids] - C[block.row_ids] @ p
    A_sp, b = check_and_drop_satisfied_zero_rows(A_sp, b)
    cand = unique(A_sp.indices)
    A_loc = A_sp[:, cand].toarray()
    x_p, Z_loc, diag = dense_scaled_svd(A_loc, b)
    H, w_p = embed_local_map(E.shape[1], cand, x_p, Z_loc)
    return p + E @ w_p, (E @ H).tocsr(), diag
```

```python
def neumann_matvec(z, G, ops, projector):
    c = G @ z
    u = ops.poisson.solve(ops.S @ c)
    trace = ops.Rv @ u + ops.T0v @ c
    return projector.project(trace)
```

```python
def dirichlet_matvec(z, G, ops, projector):
    c = G @ z
    u = ops.poisson.solve(ops.S @ c)
    normal_trace = ops.Rn @ u + ops.T1n @ c
    return projector.project(normal_trace)
```

---

**完成定义：** 当 Codex 完成本规格后，生产代码应在不构造全局稠密零空间的前提下，为 conforming、T-junction、smooth seam、Neumann feature edge 和 Dirichlet smooth T-space 统一生成 `c_full=c_p+Gz`，并将完整 exterior value/normal trace 稳定投影回同一 reduced density basis，形成可验证的 matrix-free GMRES 系统。
