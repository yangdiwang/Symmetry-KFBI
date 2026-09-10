# Industrial NURBS geometry examples

Goal: construct and display a sleeve, U bracket, four-hole flange, and impeller-like closed domain entirely from in-code NURBS control nets. These are solver test geometries, not manufacturing CAD models or completed PDE experiments.

Architecture: use the existing `NurbsSurfacePatch3D` and `NurbsSurfaceModel3D`. Add an Eigen-only geometry library, a numerical validation/export executable, and a renderer. Exports are visualization/results only; construction does not read CAD or mesh files. Sharp edges are intentional and recorded separately from smooth seams.

Specification: positive weights, nondegenerate patches, exactly paired edges, consistent outward orientation, a single connected closed boundary, correct genus, and positive volume. Preserve existing untracked experiments and temporary files.

Tasks:
1. Establish shared model interface and failing geometry checks.
2. Construct sleeve and U bracket locally; delegate flange and impeller as independent factory files.
3. Compile against existing Eigen and native NURBS code; validate topology, sampled Jacobians, seam gaps, and volume.
4. Sample the actual native evaluator, produce four views and a contact sheet, inspect rendered results.
5. Document reproducible commands, model parameters, intentional simplifications, and remaining solver integration work.

Validation: native `validate_closed`, explicit connectedness and Euler characteristic, dense parameter sampling including edges, integrated outward volume, analytic volumes where available, negative tests for missing/reversed faces. Geometry checks do not constitute a global self-intersection proof or PDE convergence validation.

Progress: complete. Initial empty sleeve factory failed the geometry test, then the implemented factory passed. Standalone MSVC Release build returned 0; all four native geometry checks and an independent CTest rerun passed. PNGs were generated from actual native evaluations and visually inspected. Independent code review found no substantive defect; corrected documentation about the degrees of planar caps. HTML data and syntax were checked, but browser runtime verification was blocked by local URL security policy; no alternate route was attempted. Branch: `codex/industrial-nurbs-models`. Existing untracked experiments are preserved. No PDE experiment or full solver build is claimed.

Follow-up refinement (user found v1 too simple): completed four substantially richer factories. Sleeve: 64 patches with flange, groove, chamfers and toroidal radii. Bracket: 82 patches with upright bored ears and rounded tops. Flange: 256 patches with eight bolt bores, entry/exit chamfers, raised neck/sealing lip and toroidal root. Impeller: 192 patches with eight swept upright vanes integrated into a bored hub and circular disk. Original visual outputs are preserved in `output/industrial_nurbs_v1`.

Refinement validation: feature tests failed against the v1 native library, then passed against v2. Independent code review and native tests found no geometry defect. Added direct checks of the sixteen curved vane flanks and planar orientation independent of axis. Final MSVC build and CTest passed; fresh export is byte-identical to the data used for the verified PNGs. Maximum seam gap 7.022e-16, maximum relative analytic-volume error 3.148e-14. PNG overview and all four individual views were inspected. HTML was regenerated, without another attempt to bypass the previously blocked local browser URL.
