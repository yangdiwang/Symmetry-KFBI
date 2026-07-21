# 3D L-prism Cauchy Stencil Policy Comparison

## Configuration

- Active cubic 3D harmonic Cauchy space: 16 coefficients.
- Requested samples: 48 values and 28 normal derivatives.
- Dirichlet formulation: normal-jump second-kind with exterior normal trace
  zero.
- GMRES relative tolerance: `2e-10`.
- Native NURBS L-prism, `N=16,32,64`.

Commands used the unchanged positional CLI and set
`KFBIM_3D_CAUCHY_POLICY` to `topological_nearest`, `same_patch`, or
`balanced_patches`.

## Dirichlet normal-jump results

| policy | N | GMRES | density Linf | order | interior Linf | order | exterior normal | condition median/p95/max | radius max/h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| topological nearest | 16 | 33 | 1.803430e-4 | - | 1.698062e-5 | - | 1.343555e-10 | 8.62 / 12.80 / 14.68 | 3.758 |
| topological nearest | 32 | 54 | 4.409306e-5 | 2.032 | 1.959255e-6 | 3.116 | 1.081341e-10 | 7.10 / 11.08 / 15.49 | 3.962 |
| topological nearest | 64 | 82 | 4.751953e-6 | 3.214 | 4.017826e-7 | 2.286 | 4.233364e-10 | 7.57 / 9.92 / 15.01 | 4.177 |
| same patch | 16 | 38 | 1.671067e-4 | - | 1.111374e-5 | - | 1.009287e-10 | 65.55 / 360.40 / 360.40 | 7.498 |
| same patch | 32 | 26 | 6.189984e-5 | 1.433 | 3.590472e-6 | 1.630 | 1.876760e-10 | 16.92 / 79.23 / 107.27 | 7.141 |
| same patch | 64 | 24 | 3.862144e-6 | 4.002 | 3.047223e-7 | 3.559 | 1.174301e-10 | 9.41 / 65.33 / 100.51 | 7.003 |
| balanced patches | 16 | 33 | 2.269332e-4 | - | 1.509811e-5 | - | 9.331957e-11 | 11.14 / 24.02 / 31.12 | 5.530 |
| balanced patches | 32 | 53 | 3.879748e-5 | 2.548 | 1.726132e-6 | 3.129 | 1.420083e-10 | 10.17 / 19.38 / 32.56 | 5.893 |
| balanced patches | 64 | 83 | 1.086391e-5 | 1.836 | 3.942835e-7 | 2.130 | 1.387198e-10 | 8.64 / 16.80 / 35.71 | 6.892 |

All nine Dirichlet solves converged. Density and interior errors decrease at
both refinements for every policy. The same-patch route changes the iteration
sequence from `33,54,82` to `38,26,24`, while retaining positive density and
interior orders. At `N=64` it is also slightly more accurate than the control.

Patch balancing enforces a maximum value/normal sample-count imbalance of
`1/1`, but its iterations are `33,53,83`, indistinguishable from the control.
It therefore does not remove the refinement-dependent Dirichlet iteration
growth.

## Neumann regression guard

| policy | N=16 | N=32 | N=64 |
|---|---:|---:|---:|
| topological nearest | 26 (converged) | 28 (converged) | 24 (converged) |
| same patch | 200 (not converged) | 38 (converged) | 26 (converged) |
| balanced patches | 27 (converged) | 29 (converged) | 24 (converged) |

The coarse same-patch Neumann solve stops at the 200-iteration limit with
relative residual `2.673425e-6`. Thus strict same-patch sampling is not a
universal replacement for both boundary conditions.

## Conclusion

For the Dirichlet exterior-normal equation, strict same-patch Cauchy fitting
is the clearly more stable of the two proposed alternatives and preserves
convergence. Equalizing sample counts across incident patches does not help.

The result also separates two effects: same-patch fits have substantially
larger local least-squares condition numbers and radii, yet much better global
Dirichlet GMRES behavior. Local Cauchy conditioning is therefore not the main
cause of the original iteration growth. The decisive change is removal of
Cauchy normal data that cross non-G1 feature edges.

For a production route, the next refinement should retain same-side sampling
across non-G1 edges while still permitting crossing of genuinely G1 native
patch seams. That keeps the Dirichlet benefit without treating artificial
smooth patch boundaries as physical corners.
