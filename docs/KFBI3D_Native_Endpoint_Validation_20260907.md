# 原生端点路径验证输出

以下保存最后一次完整路径回归输出，以及另行执行的测试结果。未运行 PDE。

```text
KFBI3D native endpoint validation, 2026-09-07
Configuration: build-3d, Release, MinGW GCC 16.1.0, MPFR enabled.
Command: build-3d/apps/native_endpoint_path_3d_test.exe
No PDE solve was run. PASS includes expected fail-closed negative tests.

BEGIN fixture_metadata
 fixture=cylinder_n32 patch=3 u=0.65000000000000002 v=0.25 endpoint_dot=-3.3704988784862e-06
 fixture=cylinder_d32_tx patch=0 u=0.15000000000000002 v=0.25 endpoint_dot=-0.00051231155579184451
 fixture=cylinder_d64_tx patch=0 u=0.078947368421052627 v=0.053571428571428568 endpoint_dot=0.00024402224332696881
PASS fixture_metadata wall_seconds=0.0010353
BEGIN plane_endpoint
 status=Certified open=0 endpoint=1 visited=1 unresolved=0 newton=0 precision_bits=53 reason=
PASS plane_endpoint wall_seconds=0.0024309
BEGIN two_ordinary_roots
 status=Certified open=2 endpoint=1 visited=3 unresolved=0 newton=2 precision_bits=53 reason=
PASS two_ordinary_roots wall_seconds=0.0044988
BEGIN split_boundary_endpoint
 status=Certified open=0 endpoint=1 visited=2 unresolved=0 newton=0 precision_bits=53 reason=
PASS split_boundary_endpoint wall_seconds=0.002458
BEGIN outside_small_residual
 status=Certified open=0 endpoint=1 visited=1 unresolved=0 newton=0 precision_bits=53 reason=
PASS outside_small_residual wall_seconds=0.0028473
BEGIN close_independent_root_before_endpoint
 status=Certified open=1 endpoint=1 visited=2 unresolved=0 newton=1 precision_bits=53 reason=
PASS close_independent_root_before_endpoint wall_seconds=0.003904
BEGIN nonfinite_interval_cannot_be_excluded
 status=Unresolved open=0 endpoint=0 visited=1 unresolved=1 newton=0 precision_bits=53 reason=NonFiniteBernsteinInterval
 message=NonFiniteBernsteinInterval
 replay=native_endpoint_path_v1 model_fnv=4c446d82c8b7069a status=Unresolved reason=NonFiniteBernsteinInterval
start=0x1p-1,0x1p-1,0x0p+0 endpoint=2,0x1p-1,0x1p-1 tolerances=0x1.19799812dea11p-39,0x1.19799812dea11p-39,0x1.b7cdfd9d7bdbbp-33 budget=1024,64,60,64,0,1024,64,128
coverage=3,1,0,0,0 gates=0,0,0,0 precision=53 newton=0
patch 0 component 0 ku 0x0p+0 0x0p+0 0x1p+0 0x1p+0 kv 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 0 0 0x0p+0 0x1p+0 0x1p-2 0x1.56e1fc2f8f359p-997
cw 0 1 0x0p+0 0x0p+0 0x1p-2 0x1.7e43c8800759cp+996
cw 1 0 0x1p+0 0x1p+0 0x1p-2 0x1.7e43c8800759cp+996
cw 1 1 0x1p+0 0x0p+0 0x1p-2 0x1.56e1fc2f8f359p-997
patch 1 component 0 ku 0x0p+0 0x0p+0 0x1p+0 0x1p+0 kv 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 0 0 0x0p+0 0x0p+0 0x1.8p-1 0x1.56e1fc2f8f359p-997
cw 0 1 0x0p+0 0x1p+0 0x1.8p-1 0x1.7e43c8800759cp+996
cw 1 0 0x1p+0 0x0p+0 0x1.8p-1 0x1.7e43c8800759cp+996
cw 1 1 0x1p+0 0x1p+0 0x1.8p-1 0x1.56e1fc2f8f359p-997
patch 2 component 0 ku 0x0p+0 0x0p+0 0x1p+0 0x1p+0 kv 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 0 0 0x0p+0 0x1p+0 0x1p+0 0x1p+0
cw 0 1 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 1 0 0x1p+0 0x1p+0 0x1p+0 0x1p+0
cw 1 1 0x1p+0 0x0p+0 0x1p+0 0x1p+0

PASS nonfinite_interval_cannot_be_excluded wall_seconds=0.0043177
BEGIN adapter_rejects_incomplete_or_wrong_state
 status=Certified open=0 endpoint=1 visited=1 unresolved=0 newton=0 precision_bits=53 reason=
PASS adapter_rejects_incomplete_or_wrong_state wall_seconds=0.0027555
BEGIN low_budget
 status=BudgetExceeded open=0 endpoint=0 visited=0 unresolved=1 newton=0 precision_bits=53 reason=CandidateBudgetExceeded
 message=CandidateBudgetExceeded
 replay=native_endpoint_path_v1 model_fnv=491f8f1f234e40f2 status=BudgetExceeded reason=CandidateBudgetExceeded
start=0x1p-1,0x1p-1,0x0p+0 endpoint=2,0x1p-1,0x1p-1 tolerances=0x1.19799812dea11p-39,0x1.19799812dea11p-39,0x1.b7cdfd9d7bdbbp-33 budget=1,1,60,8192,0,1024,64,512
coverage=0,0,0,0,0 gates=0,0,0,0 precision=53 newton=0
patch 0 component 0 ku 0x0p+0 0x0p+0 0x1p+0 0x1p+0 kv 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 0 0 0x0p+0 0x1p+0 0x1p-2 0x1p+0
cw 0 1 0x0p+0 0x0p+0 0x1p-2 0x1p+0
cw 1 0 0x1p+0 0x1p+0 0x1p-2 0x1p+0
cw 1 1 0x1p+0 0x0p+0 0x1p-2 0x1p+0
patch 1 component 0 ku 0x0p+0 0x0p+0 0x1p+0 0x1p+0 kv 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 0 0 0x0p+0 0x0p+0 0x1.8p-1 0x1p+0
cw 0 1 0x0p+0 0x1p+0 0x1.8p-1 0x1p+0
cw 1 0 0x1p+0 0x0p+0 0x1.8p-1 0x1p+0
cw 1 1 0x1p+0 0x1p+0 0x1.8p-1 0x1p+0
patch 2 component 0 ku 0x0p+0 0x0p+0 0x1p+0 0x1p+0 kv 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 0 0 0x0p+0 0x1p+0 0x1p+0 0x1p+0
cw 0 1 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 1 0 0x1p+0 0x1p+0 0x1p+0 0x1p+0
cw 1 1 0x1p+0 0x0p+0 0x1p+0 0x1p+0

PASS low_budget wall_seconds=0.0021074
BEGIN invalid_and_unsupported
 status=InvalidInput open=0 endpoint=0 visited=0 unresolved=1 newton=0 precision_bits=53 reason=InvalidNativeEndpointQuery
 message=InvalidNativeEndpointQuery
 replay=native_endpoint_path_v1 model_fnv=33d6ab3209fee39a status=InvalidInput reason=InvalidNativeEndpointQuery
start=0x1p-1,0x1p-1,0x0p+0 endpoint=0,nan,0x1p-1 tolerances=0x1.19799812dea11p-39,0x1.19799812dea11p-39,0x1.b7cdfd9d7bdbbp-33 budget=1024,8192,60,8192,0,1024,64,512
coverage=0,0,0,0,0 gates=0,0,0,0 precision=53 newton=0
patch 0 component 0 ku 0x0p+0 0x0p+0 0x1p+0 0x1p+0 kv 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 0 0 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 0 1 0x0p+0 0x1p+0 0x1p+0 0x1p+0
cw 1 0 0x1p+0 0x0p+0 0x1p+0 0x1p+0
cw 1 1 0x1p+0 0x1p+0 0x1p+0 0x1p+0

 status=UnsupportedEndpoint open=0 endpoint=0 visited=0 unresolved=1 newton=0 precision_bits=53 reason=EndpointOnPatchBoundary
 message=EndpointOnPatchBoundary
 replay=native_endpoint_path_v1 model_fnv=33d6ab3209fee39a status=UnsupportedEndpoint reason=EndpointOnPatchBoundary
start=0x0p+0,0x1p-1,0x0p+0 endpoint=0,0x0p+0,0x1p-1 tolerances=0x1.19799812dea11p-39,0x1.19799812dea11p-39,0x1.b7cdfd9d7bdbbp-33 budget=1024,8192,60,8192,0,1024,64,512
coverage=0,0,0,0,0 gates=0,0,0,0 precision=53 newton=0
patch 0 component 0 ku 0x0p+0 0x0p+0 0x1p+0 0x1p+0 kv 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 0 0 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 0 1 0x0p+0 0x1p+0 0x1p+0 0x1p+0
cw 1 0 0x1p+0 0x0p+0 0x1p+0 0x1p+0
cw 1 1 0x1p+0 0x1p+0 0x1p+0 0x1p+0

 status=DegenerateSegment open=0 endpoint=0 visited=0 unresolved=1 newton=0 precision_bits=53 reason=CoincidentNativeEndpoints
 message=CoincidentNativeEndpoints
 replay=native_endpoint_path_v1 model_fnv=33d6ab3209fee39a status=DegenerateSegment reason=CoincidentNativeEndpoints
start=0x1p-1,0x1p-1,0x1p+0 endpoint=0,0x1p-1,0x1p-1 tolerances=0x1.19799812dea11p-39,0x1.19799812dea11p-39,0x1.b7cdfd9d7bdbbp-33 budget=1024,8192,60,8192,0,1024,64,512
coverage=0,0,0,0,0 gates=0,0,0,0 precision=53 newton=0
patch 0 component 0 ku 0x0p+0 0x0p+0 0x1p+0 0x1p+0 kv 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 0 0 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 0 1 0x0p+0 0x1p+0 0x1p+0 0x1p+0
cw 1 0 0x1p+0 0x0p+0 0x1p+0 0x1p+0
cw 1 1 0x1p+0 0x1p+0 0x1p+0 0x1p+0

 status=UnsupportedEndpoint open=0 endpoint=0 visited=0 unresolved=1 newton=0 precision_bits=53 reason=TangentNativeEndpoint
 message=TangentNativeEndpoint
 replay=native_endpoint_path_v1 model_fnv=33d6ab3209fee39a status=UnsupportedEndpoint reason=TangentNativeEndpoint
start=0x0p+0,0x1p-1,0x1p+0 endpoint=0,0x1p-1,0x1p-1 tolerances=0x1.19799812dea11p-39,0x1.19799812dea11p-39,0x1.b7cdfd9d7bdbbp-33 budget=1024,8192,60,8192,0,1024,64,512
coverage=0,0,0,0,0 gates=0,0,0,0 precision=53 newton=0
patch 0 component 0 ku 0x0p+0 0x0p+0 0x1p+0 0x1p+0 kv 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 0 0 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 0 1 0x0p+0 0x1p+0 0x1p+0 0x1p+0
cw 1 0 0x1p+0 0x0p+0 0x1p+0 0x1p+0
cw 1 1 0x1p+0 0x1p+0 0x1p+0 0x1p+0

 status=Unresolved open=0 endpoint=1 visited=1 unresolved=1 newton=0 precision_bits=53 reason=StartStateMismatch
 message=StartStateMismatch
 replay=native_endpoint_path_v1 model_fnv=33d6ab3209fee39a status=Unresolved reason=StartStateMismatch
start=0x1p-1,0x1p-1,0x0p+0 endpoint=0,0x1p-1,0x1p-1 tolerances=0x1.19799812dea11p-39,0x1.19799812dea11p-39,0x1.b7cdfd9d7bdbbp-33 budget=1024,8192,60,8192,0,1024,64,512
coverage=1,1,0,1,0 gates=0,1,0,0 precision=53 newton=0
root proof=1 event=1 kind=2 patch=0 uv=0.5,0.5 T=1,1 residual=0 contraction=7.7715611723760978e-16 sign=-1
patch 0 component 0 ku 0x0p+0 0x0p+0 0x1p+0 0x1p+0 kv 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 0 0 0x0p+0 0x0p+0 0x1p+0 0x1p+0
cw 0 1 0x0p+0 0x1p+0 0x1p+0 0x1p+0
cw 1 0 0x1p+0 0x0p+0 0x1p+0 0x1p+0
cw 1 1 0x1p+0 0x1p+0 0x1p+0 0x1p+0

PASS invalid_and_unsupported wall_seconds=0.0062843
BEGIN repeated_query_and_rigid_transform
 status=Certified open=0 endpoint=1 visited=1 unresolved=0 newton=0 precision_bits=53 reason=
 status=Certified open=0 endpoint=1 visited=1 unresolved=0 newton=0 precision_bits=53 reason=
 status=Certified open=0 endpoint=1 visited=1 unresolved=0 newton=0 precision_bits=53 reason=
PASS repeated_query_and_rigid_transform wall_seconds=0.0071681
BEGIN intersector_cached_native_query
 cache_phase=initial_call_once status=Certified open=0 visited=95 reason=
 cache_phase=reused_geometry_and_bounds status=Certified open=0 visited=95 reason=
 cache_phase=copied_immutable_intersector status=Certified open=0 visited=95 reason=
PASS intersector_cached_native_query wall_seconds=0.2945676
BEGIN cylinder_n32
 fixture=cylinder_n32 patch=3 u=0.65000000000000002 v=0.25 endpoint_dot=-3.3704988784862e-06
 status=Certified open=0 endpoint=1 visited=251 unresolved=0 newton=2202 precision_bits=128 reason=
PASS cylinder_n32 wall_seconds=2.3980575
BEGIN cylinder_d32_tx
 fixture=cylinder_d32_tx patch=0 u=0.15000000000000002 v=0.25 endpoint_dot=-0.00051231155579184451
 status=Certified open=0 endpoint=1 visited=95 unresolved=0 newton=364 precision_bits=53 reason=
PASS cylinder_d32_tx wall_seconds=0.0920243
BEGIN cylinder_d64_tx
 fixture=cylinder_d64_tx patch=0 u=0.078947368421052627 v=0.053571428571428568 endpoint_dot=0.00024402224332696881
 status=Certified open=1 endpoint=1 visited=109 unresolved=0 newton=460 precision_bits=53 reason=
PASS cylinder_d64_tx wall_seconds=0.2846784
SUMMARY selected=15 failures=0

Other separately executed checks (exit code 0):
native_nurbs_exact_geometry_3d_test: all checks passed
restrict crossing selector 3D tests passed
restrict crossing path-state 3D tests passed
tensor-product cover restrict 3D tests passed
3D shared quadratic restrict tests passed
kfbi_topology_affine_exterior_trace_3d.exe --help: exit code 0 (native_certified environment selected)

Additional CLI guard check after final main-program rebuild:
Command: KFBIM_3D_SUPPORT_PATH=native_certified; kfbi_topology_affine_exterior_trace_3d.exe --restrict-probe 16
error: --restrict-probe is a legacy joint-tricubic diagnostic; native_certified requires the Q27/Q64 production routes
PASS native_certified rejects legacy --restrict-probe before setup (expected exit 1).

Final build freshness check for all seven targets:
ninja: no work to do.

```
