# 3D Cauchy Degree Audit

## Goal

Verify the polynomial degree used by the active app-local three-dimensional
harmonic Cauchy fit before changing the L-prism stencil policy.

## Audit result

`PanelCenterCauchyFit3D` retains a legacy default argument of degree four,
but the only active construction in `PanelCenterHarmonicJetKFBI3D` explicitly
passes degree three:

```cpp
fit_(cloud, stencils, h_, 3)
```

Commit `f92cd73` changed that active argument from four to three. All current
native-NURBS 3D convergence results were generated after that commit and
therefore already use the 16-coefficient degree-three harmonic space.

No production-code degree change is required. Treating the current results
as degree-four results would be incorrect.

## Verification

Build and run the native NURBS test and the current 3D app for
`l_prism 16 32 64` to reconfirm the already-active cubic route.

Report the cubic results using:

- GMRES iterations and relative residual;
- maximum local Cauchy condition number;
- density and interior `Linf` errors and their observed orders;
- exterior-normal condition residual and direct/jump route mismatch.

A valid degree-three versus degree-four comparison would require a new
controlled run of both degrees on the same current native-NURBS geometry and
stencils. Historical pre-`f92cd73` results are not a clean baseline because
the surrounding 3D route subsequently changed.
