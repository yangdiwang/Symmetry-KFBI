# 3D Cubic Cauchy Sample-count Search Results

## Conclusion

For the L-prism Dirichlet normal-jump second-kind formulation, strict
`same_patch` sampling with requested budgets of `32` value samples and `20`
normal samples is the winner under the predefined ranking. Its Dirichlet GMRES counts are
`37, 28, 24` at `N=16,32,64`; it has the smallest `N=64` density error among
the three iteration-tied finalists and passes both order-two gates.

The search changed only Cauchy sample counts. The Cauchy polynomial remained
cubic, and restrict remained the existing 4x4x4 tricubic Cartesian
interpolation plus the joint two-sided cubic normal fit.

## Configuration

All search runs used:

- geometry: `l_prism`;
- Cauchy policy: `same_patch`;
- Cauchy degree: 3 (16 harmonic coefficients);
- restrict grid/normal degrees: 3/3;
- unchanged GMRES tolerance: `2e-10`.

Exact sweep commands:

```powershell
$exe='.\build\apps\Release\neumann_exterior_zero_trace_3d.exe'
$env:KFBIM_3D_CAUCHY_POLICY='same_patch'

foreach($pair in '10/6','12/8','16/10','20/12','24/16','32/20','48/28') {
    $counts=$pair.Split('/')
    $env:KFBIM_3D_CAUCHY_VALUE_COUNT=$counts[0]
    $env:KFBIM_3D_CAUCHY_NORMAL_COUNT=$counts[1]
    & $exe l_prism 16 32
}

foreach($pair in '24/20','32/16','32/28','48/16','48/20') {
    $counts=$pair.Split('/')
    $env:KFBIM_3D_CAUCHY_VALUE_COUNT=$counts[0]
    $env:KFBIM_3D_CAUCHY_NORMAL_COUNT=$counts[1]
    & $exe l_prism 16 32
}

foreach($pair in '24/20','32/20','48/20','48/28') {
    $counts=$pair.Split('/')
    $env:KFBIM_3D_CAUCHY_VALUE_COUNT=$counts[0]
    $env:KFBIM_3D_CAUCHY_NORMAL_COUNT=$counts[1]
    & $exe l_prism 16 32 64
}

Remove-Item Env:KFBIM_3D_CAUCHY_POLICY,
            Env:KFBIM_3D_CAUCHY_VALUE_COUNT,
            Env:KFBIM_3D_CAUCHY_NORMAL_COUNT
& $exe l_prism 16 32 64
```

## Stage 1: paired sweep

All full-rank pairs converged for the Dirichlet formulation. Orders below are
from `N=16` to `N=32`.

| Counts | Result | Dirichlet iterations | Density Linf at 16/32 (order) | Interior Linf at 16/32 (order) |
|---|---|---:|---:|---:|
| 10/6 | rejected: rank deficient at DOF 0 | - | - | - |
| 12/8 | rejected: rank deficient at DOF 1 | - | - | - |
| 16/10 | rejected: rank deficient at DOF 110 | - | - | - |
| 20/12 | full rank | 44 / 30 | 8.066e-5 / 2.829e-5 (1.512) | 7.645e-6 / 1.455e-6 (2.394) |
| 24/16 | full rank | 39 / 29 | 8.275e-5 / 2.528e-5 (1.711) | 6.461e-6 / 1.353e-6 (2.256) |
| 32/20 | full rank | 37 / 28 | 1.051e-4 / 3.755e-5 (1.485) | 6.674e-6 / 2.313e-6 (1.529) |
| 48/28 | full-rank control | 38 / 26 | 1.671e-4 / 6.190e-5 (1.433) | 1.111e-5 / 3.590e-6 (1.630) |

The theoretical `10/6` minimum is not usable with the actual nearest-point
layout: the 16 conditions do not span the 16-column cubic harmonic system.

## Stage 2: local cross sweep

The best adjacent Stage-1 region was `32/20`--`48/28`; `24/16` was included
as the immediate lower neighbor. The new cross combinations gave:

| Counts | Dirichlet iterations 16/32 | Density Linf at N=32 | Interior Linf at N=32 | Density/interior order |
|---|---:|---:|---:|---:|
| 24/20 | 37 / 28 | 2.192e-5 | 1.349e-6 | 1.937 / 2.238 |
| 32/16 | 39 / 29 | 4.346e-5 | 2.361e-6 | 1.319 / 1.541 |
| 32/28 | 38 / 26 | 3.533e-5 | 2.220e-6 | 1.446 / 1.498 |
| 48/16 | 39 / 29 | 7.428e-5 | 3.837e-6 | 1.414 / 1.784 |
| 48/20 | 37 / 28 | 6.529e-5 | 3.764e-6 | 1.506 / 1.710 |

Normal count `20` produced the smallest maximum iteration count. Increasing
the value count above `24` did not improve the Dirichlet iteration sequence.

The complete `N=16/32` solver audit for every full-rank Stage-1/2 pair is
below. Each entry is `N=16 / N=32`.

| Counts | GMRES relative residual | Exterior-normal residual | Route mismatch | Runtime (s) |
|---|---:|---:|---:|---:|
| 20/12 | 1.475e-10 / 1.221e-10 | 6.777e-11 / 2.018e-10 | 1.586e-14 / 3.077e-14 | 0.0715 / 0.3164 |
| 24/16 | 8.960e-11 / 9.770e-11 | 4.282e-11 / 1.043e-10 | 1.511e-14 / 3.308e-14 | 0.0642 / 0.3116 |
| 32/20 | 1.734e-10 / 8.014e-11 | 1.061e-10 / 7.018e-11 | 1.484e-14 / 3.489e-14 | 0.0622 / 0.3065 |
| 48/28 | 1.076e-10 / 1.174e-10 | 1.009e-10 / 1.877e-10 | 1.394e-14 / 3.203e-14 | 0.0672 / 0.3222 |
| 24/20 | 1.734e-10 / 8.014e-11 | 1.061e-10 / 7.019e-11 | 1.553e-14 / 3.156e-14 | 0.0608 / 0.2978 |
| 32/16 | 8.959e-11 / 9.770e-11 | 4.282e-11 / 1.043e-10 | 1.522e-14 / 3.204e-14 | 0.0638 / 0.3287 |
| 32/28 | 1.076e-10 / 1.174e-10 | 1.009e-10 / 1.877e-10 | 1.445e-14 / 3.384e-14 | 0.0769 / 0.2950 |
| 48/16 | 8.959e-11 / 9.770e-11 | 4.282e-11 / 1.043e-10 | 1.462e-14 / 3.275e-14 | 0.0672 / 0.3050 |
| 48/20 | 1.733e-10 / 8.014e-11 | 1.060e-10 / 7.017e-11 | 1.442e-14 / 3.268e-14 | 0.0706 / 0.3196 |

## Stage 3: N=64 verification

| Policy/counts | N | GMRES | Density Linf (order) | Interior Linf (order) | Condition p95 | Radius max/h |
|---|---:|---:|---:|---:|---:|---:|
| same 24/20 | 16 | 37 | 8.394408e-5 | 6.365785e-6 | 2.422e2 | 5.503 |
| | 32 | 28 | 2.192000e-5 (1.937) | 1.349312e-6 (2.238) | 6.090e1 | 4.934 |
| | 64 | 24 | 2.797150e-6 (2.970) | 1.806489e-7 (2.901) | 5.392e1 | 4.934 |
| same 32/20 | 16 | 37 | 1.051176e-4 | 6.673964e-6 | 3.604e2 | 6.409 |
| | 32 | 28 | 3.754507e-5 (1.485) | 2.312716e-6 (1.529) | 5.952e1 | 5.750 |
| | 64 | 24 | 2.721994e-6 (3.786) | 2.584010e-7 (3.162) | 5.202e1 | 5.750 |
| same 48/20 | 16 | 37 | 1.854923e-4 | 1.231365e-5 | 3.604e2 | 7.498 |
| | 32 | 28 | 6.528754e-5 (1.506) | 3.763690e-6 (1.710) | 7.923e1 | 7.141 |
| | 64 | 24 | 3.917537e-6 (4.059) | 3.038775e-7 (3.631) | 6.533e1 | 7.003 |
| same 48/28 | 16 | 38 | 1.671067e-4 | 1.111374e-5 | 3.604e2 | 7.498 |
| | 32 | 26 | 6.189984e-5 (1.433) | 3.590472e-6 (1.630) | 7.923e1 | 7.141 |
| | 64 | 24 | 3.862144e-6 (4.002) | 3.047223e-7 (3.559) | 6.533e1 | 7.003 |
| topological 48/28 | 16 | 33 | 1.803430e-4 | 1.698062e-5 | 1.280e1 | 3.758 |
| | 32 | 54 | 4.409306e-5 (2.032) | 1.959255e-6 (3.116) | 1.108e1 | 3.962 |
| | 64 | 82 | 4.751953e-6 (3.214) | 4.017826e-7 (2.286) | 9.921 | 4.177 |

Every same-patch finalist passed the required `N=32` to `N=64` order-two
gate. `24/20`, `32/20`, and `48/20` tie on the first two ranking criteria:
all have `37,28,24` iterations and decreasing `N=32` to `N=64` counts. The
next listed criterion selects `32/20`, whose `N=64` density error is
`2.721994e-6`, versus `2.797150e-6` for `24/20` and `3.917537e-6` for
`48/20`. `24/20` remains a useful lighter alternative: its requested budget
uses eight fewer samples and has a 30% smaller `N=64` interior error, but it
is not the formal winner under the fixed ranking.

Relative to same-patch `48/28`, `32/20` reduces the total requested sample
budget by 31.6%, reduces the maximum Dirichlet iteration count from 38 to 37,
and reduces the `N=64` density/interior errors by about 29.5%/15.2%. At
`N=16`, finite patch capacity gives actual value min/max `16/32` and normal
min/max `16/20`; at `N=32,64`, the actual counts are exactly `32/32` and
`20/20`. Relative to the original topological `48/28` route, it removes the
`33,54,82` iteration growth and reduces the fine-grid density/interior errors
by about 42.7%/35.7%.

Final residual and runtime audit:

| Policy/counts | N | GMRES relative residual | Exterior-normal residual | Route mismatch | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| same 24/20 | 16 | 1.734e-10 | 1.061e-10 | 1.553e-14 | 0.0608 |
| | 32 | 8.014e-11 | 7.019e-11 | 3.156e-14 | 0.2978 |
| | 64 | 1.512e-10 | 1.271e-10 | 6.994e-14 | 1.7284 |
| same 32/20 | 16 | 1.734e-10 | 1.061e-10 | 1.484e-14 | 0.0622 |
| | 32 | 8.014e-11 | 7.018e-11 | 3.489e-14 | 0.3065 |
| | 64 | 1.512e-10 | 1.272e-10 | 6.962e-14 | 1.8089 |
| same 48/20 | 16 | 1.733e-10 | 1.060e-10 | 1.442e-14 | 0.0706 |
| | 32 | 8.014e-11 | 7.017e-11 | 3.268e-14 | 0.3196 |
| | 64 | 1.512e-10 | 1.271e-10 | 6.734e-14 | 1.8323 |
| same 48/28 | 16 | 1.076e-10 | 1.009e-10 | 1.394e-14 | 0.0672 |
| | 32 | 1.174e-10 | 1.877e-10 | 3.203e-14 | 0.3222 |
| | 64 | 1.420e-10 | 1.174e-10 | 7.032e-14 | 1.7910 |
| topological 48/28 | 16 | 1.586e-10 | 1.344e-10 | 1.381e-14 | 0.0581 |
| | 32 | 9.223e-11 | 1.081e-10 | 3.416e-14 | 0.5255 |
| | 64 | 1.815e-10 | 4.233e-10 | 6.970e-14 | 5.2387 |

## Neumann diagnostic

Strict same-patch sampling is not suitable as a universal Neumann default:
for all finalists the `N=16` Neumann solve reached the 200-step limit. For
`32/20`, the Neumann status/iterations were `failed/200`, `passed/43`, and
`passed/26` at `N=16,32,64`. This was recorded but was not an acceptance gate
for the Dirichlet-normal optimization. The topological `48/28` Neumann
control converged in `26,28,24` steps.

## Recommendation

Use cubic `same_patch 32/20` for the L-prism Dirichlet normal-jump route. Keep
the current topological `48/28` configuration for Neumann or mixed workloads.
The production defaults remain unchanged by this experiment; the new
environment variables select the recommended Dirichlet configuration without
overwriting other result directories.
