# 3D KFBI 几何预处理后端比较

## 实验范围

- 后端：`baseline`（原 BVH 认证求交）、`pure`（稳定候选映射 + early unique-root certificate）、`hybrid`（平面解析/光滑面最近点认证 + 自动退回 pure）。
- 几何：torus、hollow cylinder、L-prism。
- 网格：`N=32,64,128`，节点布局；每格 1 次 warmup、3 次 Release 计时，repetition 间轮换后端顺序。
- 计时范围：native NURBS domain、`GridPair3D`、二者 combined。全节点、全结构边、交点和 owner 的准确性扫描在计时区间外。
- 复现：

```powershell
.\build\apps\Release\nurbs_geometry_preprocess_benchmark_3d.exe `
  --backend all --geometry all --N 32 64 128 `
  --warmup 1 --reps 3 `
  --out output\nurbs_geometry_preprocess_benchmark
```

原始数据：`output/nurbs_geometry_preprocess_benchmark/nurbs_geometry_preprocess_raw.csv`；汇总数据：`output/nurbs_geometry_preprocess_benchmark/nurbs_geometry_preprocess_summary.csv`。

## 准确性

最终运行写出 81 行 raw、27 行 summary，程序 exit 0，且未生成 mismatch 文件。

| 检查项 | 结果 |
|---|---:|
| analytic / baseline node-label mismatch | 0 |
| barrier/interface edge mismatch | 0 |
| edge classification / crossing-count mismatch | 0 |
| crossing-field / correction-crossing mismatch | 0 |
| `GridPair3D` correction-owner mismatch | 0 |
| unsafe label-changing edge | 0 |
| targeted-retry unsafe | 0 |
| 9 个 geometry/N 组的跨后端 topology checksum | 全部一致 |
| 最大跨后端物理点差 | `2.653e-11` |
| 最大 native root residual | `2.150e-12` |

最大点差来自 torus/N=64 的浅交角根。Hybrid residual 为 `3.52e-18`，baseline residual 为 `1.31e-12`；位置差落在由 `residual/(transversality-reliability)` 给出的条件误差界内。比较器只对可靠横截根的位置量使用该条件界；patch/component/count、残差、法向、reliability 和 owner 对自身 crossing 的检查仍保持原阈值，并有 `transversality >= 2*reliability` 与 `0.25h` 上限。

`used_targeted_retry` 是后端控制流元数据，不参与跨后端几何分类相等或 topology checksum；它仍单独写入 route diagnostics。

Torus/N=128 的初始求交记录包含 baseline/pure/hybrid = `5/5/3` 个 unresolved element candidates，对应 `5/5/4` 条 ambiguous-parity edges。Targeted retry 解决其中 `2/2/1` 条 label-changing ambiguous edges；剩余三条非 label-changing ambiguous edges 采用 endpoint-parity fallback。所有 label-changing edges 均 correction-safe；targeted retry 结果为 baseline `2/2/0`、pure `2/2/0`、hybrid `1/1/0`（总数/成功/unsafe）。

## 速度

下表均为 3 次计时的中位数；括号内为相对 baseline 的 combined 加速比。

| Geometry | N | Baseline combined (s) | Pure combined (s) | Hybrid combined (s) | Hybrid domain speedup |
|---|---:|---:|---:|---:|---:|
| cylinder | 32 | 0.8747 | 0.8597 (1.02×) | 0.5208 (1.68×) | 1.77× |
| cylinder | 64 | 3.4990 | 3.3325 (1.05×) | 1.8461 (1.90×) | 2.60× |
| cylinder | 128 | 19.5826 | 19.3601 (1.01×) | 13.3375 (1.47×) | 2.83× |
| L-prism | 32 | 0.0861 | 0.0781 (1.10×) | 0.0723 (1.19×) | 1.59× |
| L-prism | 64 | 0.4741 | 0.4254 (1.11×) | 0.4193 (1.13×) | 1.45× |
| L-prism | 128 | 3.5890 | 3.4316 (1.05×) | 3.3975 (1.06×) | 1.34× |
| torus | 32 | 1.8168 | 2.2259 (0.82×) | 0.6130 (2.96×) | 3.36× |
| torus | 64 | 5.3830 | 5.3622 (1.00×) | 1.7771 (3.03×) | 4.37× |
| torus | 128 | 30.2421 | 30.0224 (1.01×) | 15.5990 (1.94×) | 4.83× |

解释：

- `pure` 删除了重复 BVH candidate discovery，并提前接受部分唯一根，但大多数曲面元素仍进入原认证工作，combined 通常只改善约 0–11%。Torus/N=32 主矩阵存在较明显的系统负载/离群值波动；额外 7-rep 复跑得到 pure `1.007×`、hybrid `2.823×` combined，但 combined CV 仍为 `23.2%/16.1%`，因此稳定结论以 N=64/128 为主。
- `hybrid` 对 L-prism/N=128 的 15,122 次 candidate-element 处理全部走平面解析快路；但 `GridPair3D` 占 combined 的大部分，所以最终只有 `1.06×`。
- Cylinder/N=128 的 41,436 次 smooth closest-point 尝试中 41,412 次被直接认证，24 次 closest prefilter 安全回退；另有 3,329 次 non-G1 feature element 未进入 closest 路线，因此总 `certified_fallback_elements=3,353`。combined 为 `1.47×`。
- Torus/N=128 的 42,444 次 closest-point 尝试中 42,169 次被直接认证、275 次回退；domain 达 `4.83×`，但约 11.7 s 的 `GridPair3D` 固定成本把 combined 限制到 `1.94×`。

## 结论

`hybrid` 是三种生产几何上最有效的可选预处理路线：在稳定的 N=64/128 数据中，domain 加速为 `1.34–4.83×`，combined 加速为 `1.06–3.03×`，且 KFBI 实际消费的标签、barrier、交点和 owner 与 baseline 一致。`pure` 保留为低风险对照/回退路线；默认后端仍是 `baseline`，现有应用行为不变。
