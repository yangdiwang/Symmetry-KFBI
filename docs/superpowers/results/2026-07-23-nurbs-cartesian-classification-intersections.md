# NURBS Cartesian Classification and Intersection Results

## Implemented scope

- Cartesian edges now return all confirmed NURBS crossings plus component-parity metadata and unresolved close-root clusters.
- Every rational Bezier BVH leaf stores a 4-by-4 parameter sample table. Each edge query retains at most four separated projected-distance seeds; triangle seeds remain optional accelerators.
- A bounded line-surface stationary-point solver protects close physical roots when a positive-distance minimum lies between them. Missing proof produces an ambiguous cluster instead of deleting a candidate.
- Root canonicalization follows declared patch topology. G1 and non-G1 shared-edge duplicates merge within one component; non-G1 representatives preserve one-sided normal data and are marked as feature contacts. Unrelated coincident patches remain fatal.
- The Cartesian domain stores contiguous crossing ranges, separates geometric interface edges from flood barriers, applies parity independently per solid component, and caches endpoint component queries for ambiguous parity.
- Console and CSV diagnostics cover sampling, stationary solves, close-root clusters, topology merges, high-degree fallbacks, interface/barrier counts, multi-crossing parity, and endpoint fallback.

The PDE correction, restriction, Cauchy fitting, and GMRES algorithms were not changed by this geometry-classification work.

## Regression coverage

- Analytic ruled graphs: two roots, three roots, quadratic contact, and zero-derivative odd crossing.
- Close roots at `0.5 +/- 1e-3` with a positive stationary witness, checked at scales corresponding to `N=32,64,128`.
- Split-leaf duplicate root, periodic seam, G1 seam, non-G1 feature seam, unrelated coincident patches, and bounded high-degree fallback.
- Under-resolved torus multi-crossing edges, same-component even parity, two-component membership changes, and triangle-seed-independent crossing ranges.
- Full torus, hollow-cylinder, and L-prism runs at `N=32,64,128`.

## Geometry classification

All nine production runs had zero analytic node-label mismatches and zero triangle-fallback crossings.

| Geometry | N | Interface edges | Barriers | Multi / even | Ambiguous fallback | Max root residual |
|---|---:|---:|---:|---:|---:|---:|
| Torus | 32 | 688 | 686 | 2 / 2 | 0 | 2.06e-12 |
| Torus | 64 | 2760 | 2758 | 2 / 2 | 0 | 2.15e-12 |
| Torus | 128 | 11405 | 11404 | 1 / 1 | 5 | 2.12e-12 |
| Hollow cylinder | 32 | 1174 | 1160 | 14 / 14 | 0 | 1.63e-12 |
| Hollow cylinder | 64 | 4544 | 4544 | 0 / 0 | 0 | 7.08e-13 |
| Hollow cylinder | 128 | 17812 | 17812 | 0 / 0 | 0 | 1.78e-12 |
| L-prism | 32 | 982 | 982 | 0 / 0 | 0 | 3.33e-16 |
| L-prism | 64 | 3926 | 3926 | 0 / 0 | 0 | 2.72e-16 |
| L-prism | 128 | 15122 | 15122 | 0 / 0 | 0 | 2.72e-16 |

The main subdivision depth stayed at or below 4, terminal certification stayed at or below 6, and no edge/leaf accepted more than two sample seeds in these runs (the enforced limit is four).

## PDE results

Interior `L_inf` errors and observed orders:

| Geometry | Formulation | N=32 | N=64 (order) | N=128 (order) | GMRES iterations |
|---|---|---:|---:|---:|---:|
| Torus | Neumann exterior trace | 1.090e-6 | 2.882e-7 (1.919) | 7.353e-8 (1.970) | 23, 22, 21 |
| Torus | Dirichlet exterior normal | 4.060e-7 | 7.566e-8 (2.424) | 1.569e-8 (2.270) | 15, 13, 12 |
| Hollow cylinder | Neumann exterior trace | 3.687e-6 | 7.375e-7 (2.322) | 2.315e-7 (1.672) | 24, 23, 26 |
| Hollow cylinder | Dirichlet exterior normal | 2.303e-6 | 1.834e-7 (3.650) | 2.921e-8 (2.650) | 31, 25, 24 |
| L-prism | Neumann exterior trace | 2.865e-6 | 6.219e-7 (2.204) | 2.136e-7 (1.542) | 28, 24, 31 |
| L-prism | Dirichlet exterior normal | 1.959e-6 | 4.018e-7 (2.286) | 1.074e-6 (-1.418) | 54, 82, 99 |

The L-prism Dirichlet-normal deterioration at `N=128` is not caused by the new inside/outside classifier: that run has zero label mismatches, machine-level planar root residual, no multi-crossing ambiguity, and identical interface/barrier counts. It remains a corner-sensitive Cauchy/restriction/GMRES issue in the PDE route.

## Artifacts

- `output/neumann_exterior_zero_trace_3d/geometry_readiness.csv`
- `output/neumann_exterior_zero_trace_3d/neumann_results.csv`
- `output/neumann_exterior_zero_trace_3d/dirichlet_normal_results.csv`
