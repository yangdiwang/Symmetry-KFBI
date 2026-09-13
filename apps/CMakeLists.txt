# Source ownership follows the functional repository layout.
# Target names and per-target options are preserved for compatibility.

if(KFBIM_BUILD_3D)
    add_executable(industrial_nurbs_gallery_3d
        ${PROJECT_SOURCE_DIR}/apps/geometry/industrial_nurbs_gallery_3d.cpp)
    target_link_libraries(industrial_nurbs_gallery_3d PRIVATE kfbim_core)

    add_executable(kfbi_trace93_study_3d
        ${PROJECT_SOURCE_DIR}/apps/laplace/3d/trace93_study_3d.cpp)
    target_link_libraries(kfbi_trace93_study_3d PRIVATE kfbim_3d_app_geometry kfbim_core)
    execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        OUTPUT_VARIABLE KFBIM_TRACE93_BUILD_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(KFBIM_TRACE93_BUILD_REVISION)
        target_compile_definitions(kfbi_trace93_study_3d PRIVATE
            KFBIM_BUILD_REVISION="${KFBIM_TRACE93_BUILD_REVISION}")
    endif()
    if(KFBIM_BUILD_EXPERIMENTAL_3D)
        target_link_libraries(kfbi_trace93_study_3d PRIVATE kfbim_3d_shared_correction)
        target_compile_definitions(kfbi_trace93_study_3d PRIVATE KFBIM_HAS_SHARED_FIELD_3D=1)
    endif()
    target_include_directories(kfbi_trace93_study_3d PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_features(kfbi_trace93_study_3d PRIVATE cxx_std_17)
    set_target_properties(kfbi_trace93_study_3d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/apps")
    add_executable(kfbi_trace_first_study_3d
        ${PROJECT_SOURCE_DIR}/apps/laplace/3d/trace_first_study_3d.cpp)
    target_link_libraries(kfbi_trace_first_study_3d PRIVATE kfbim_3d_app_geometry kfbim_core)
    target_include_directories(kfbi_trace_first_study_3d PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_features(kfbi_trace_first_study_3d PRIVATE cxx_std_17)
    set_target_properties(kfbi_trace_first_study_3d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/apps")
endif()

add_executable(shape_opt_transmission_2d
    ${PROJECT_SOURCE_DIR}/apps/shape_optimization/shape_opt_transmission_2d.cpp
)

add_executable(transmission_center_perturb_2d
    ${PROJECT_SOURCE_DIR}/apps/transmission/transmission_center_perturb_2d.cpp
)

add_executable(neumann_exterior_trace_lshape_2d
    ${PROJECT_SOURCE_DIR}/apps/laplace/2d/neumann_exterior_trace_lshape_2d.cpp
)

add_executable(neumann_exterior_trace_lshape_slow_points_2d
    ${PROJECT_SOURCE_DIR}/apps/laplace/2d/neumann_exterior_trace_lshape_slow_points_2d.cpp
)

add_executable(neumann_exterior_trace_circle_2d
    ${PROJECT_SOURCE_DIR}/apps/laplace/2d/neumann_exterior_trace_circle_2d.cpp
)

add_executable(neumann_harmonic_jet_case_2d
    ${PROJECT_SOURCE_DIR}/apps/laplace/2d/neumann_harmonic_jet_case_2d.cpp
)

add_executable(laplace_smooth_boundary_iteration_2d
    ${PROJECT_SOURCE_DIR}/apps/laplace/2d/laplace_smooth_boundary_iteration_2d.cpp
)

add_executable(laplace_exterior_trace_flower_convergence_2d
    ${PROJECT_SOURCE_DIR}/apps/laplace/2d/laplace_exterior_trace_flower_convergence_2d.cpp
)

add_executable(neumann_harmonic_jet_python_compatible_2d
    ${PROJECT_SOURCE_DIR}/apps/laplace/2d/neumann_harmonic_jet_python_compatible_2d.cpp
)

add_executable(dirichlet_harmonic_jet_python_compatible_2d
    ${PROJECT_SOURCE_DIR}/apps/laplace/2d/neumann_harmonic_jet_python_compatible_2d.cpp
)

if(KFBIM_BUILD_3D)
    add_executable(neumann_exterior_zero_trace_3d
        ${PROJECT_SOURCE_DIR}/apps/laplace/3d/neumann_exterior_zero_trace_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/cauchy/crossing_cauchy_plan_3d.cpp
    )

    add_executable(neumann_exterior_zero_trace_3d_all_quartic
        ${PROJECT_SOURCE_DIR}/apps/laplace/3d/neumann_exterior_zero_trace_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/cauchy/crossing_cauchy_plan_3d.cpp
    )

    add_executable(kfbi_native_c0_exterior_trace_3d
        ${PROJECT_SOURCE_DIR}/apps/laplace/3d/neumann_exterior_zero_trace_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/cauchy/crossing_cauchy_plan_3d.cpp
    )

    add_executable(kfbi_topology_affine_exterior_trace_3d
        ${PROJECT_SOURCE_DIR}/apps/laplace/3d/neumann_exterior_zero_trace_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/cauchy/crossing_cauchy_plan_3d.cpp
    )

    add_executable(kfbi_general_cap_exterior_trace_3d
        ${PROJECT_SOURCE_DIR}/apps/laplace/3d/kfbi_general_cap_exterior_trace_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/density/cap_atlas_density_space_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/cauchy/crossing_cauchy_plan_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/mean_free_trace_mass_coordinates_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/reduced_trace_projection_3d.cpp
    )

endif()
target_link_libraries(shape_opt_transmission_2d PRIVATE
    kfbim_core
)

target_link_libraries(transmission_center_perturb_2d PRIVATE
    kfbim_core
)

target_link_libraries(neumann_exterior_trace_lshape_2d PRIVATE
    kfbim_core
)

target_link_libraries(neumann_exterior_trace_lshape_slow_points_2d PRIVATE
    kfbim_core
)

target_link_libraries(neumann_exterior_trace_circle_2d PRIVATE
    kfbim_core
)

target_link_libraries(neumann_harmonic_jet_case_2d PRIVATE
    kfbim_core
)

target_link_libraries(laplace_smooth_boundary_iteration_2d PRIVATE
    kfbim_core
)

target_link_libraries(laplace_exterior_trace_flower_convergence_2d PRIVATE
    kfbim_core
)

target_compile_definitions(laplace_exterior_trace_flower_convergence_2d PRIVATE
    KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
)

target_link_libraries(neumann_harmonic_jet_python_compatible_2d PRIVATE
    kfbim_core
)

target_link_libraries(dirichlet_harmonic_jet_python_compatible_2d PRIVATE
    kfbim_core
)

if(KFBIM_BUILD_3D)
    target_link_libraries(neumann_exterior_zero_trace_3d PRIVATE
        kfbim_3d_app_geometry
        kfbim_core
    )

    target_link_libraries(neumann_exterior_zero_trace_3d_all_quartic PRIVATE
        kfbim_3d_app_geometry
        kfbim_core
    )

    target_link_libraries(kfbi_native_c0_exterior_trace_3d PRIVATE
        kfbim_3d_app_geometry
        kfbim_core
    )

    target_link_libraries(kfbi_topology_affine_exterior_trace_3d PRIVATE
        kfbim_3d_app_geometry
        kfbim_core
    )

    target_link_libraries(kfbi_general_cap_exterior_trace_3d PRIVATE
        kfbim_core
    )

endif()
target_compile_definitions(shape_opt_transmission_2d PRIVATE
    KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
)

target_compile_definitions(transmission_center_perturb_2d PRIVATE
    KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
)

target_compile_definitions(neumann_exterior_trace_lshape_2d PRIVATE
    KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
)

target_compile_definitions(neumann_exterior_trace_lshape_slow_points_2d PRIVATE
    KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
)

target_compile_definitions(neumann_exterior_trace_circle_2d PRIVATE
    KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
)

target_compile_definitions(neumann_harmonic_jet_case_2d PRIVATE
    KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
)

target_compile_definitions(laplace_smooth_boundary_iteration_2d PRIVATE
    KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
)

target_compile_definitions(neumann_harmonic_jet_python_compatible_2d PRIVATE
    KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
)

target_compile_definitions(dirichlet_harmonic_jet_python_compatible_2d PRIVATE
    KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
    KFBIM_PYJET_DEFAULT_BVP_DIRICHLET=1
)

if(KFBIM_BUILD_3D)
    target_compile_definitions(neumann_exterior_zero_trace_3d PRIVATE
        KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
    )

    target_compile_definitions(
        neumann_exterior_zero_trace_3d_all_quartic PRIVATE
        KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
        KFBIM_3D_CAUCHY_POLYNOMIAL_DEGREE=4
        KFBIM_3D_RESTRICT_GRID_DEGREE=4
        KFBIM_3D_RESTRICT_NORMAL_DEGREE=4
    )

    target_compile_definitions(kfbi_native_c0_exterior_trace_3d PRIVATE
        KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
        KFBIM_3D_NATIVE_COEFFICIENT_DEFAULT=1
    )

    target_compile_definitions(kfbi_topology_affine_exterior_trace_3d PRIVATE
        KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
        KFBIM_3D_NATIVE_COEFFICIENT_DEFAULT=1
        KFBIM_3D_TOPOLOGY_AFFINE_DEFAULT=1
    )

    target_compile_definitions(kfbi_general_cap_exterior_trace_3d PRIVATE
        KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
    )

endif()
target_compile_features(shape_opt_transmission_2d PRIVATE cxx_std_17)

target_compile_features(transmission_center_perturb_2d PRIVATE cxx_std_17)

target_compile_features(neumann_exterior_trace_lshape_2d PRIVATE cxx_std_17)

target_compile_features(neumann_exterior_trace_circle_2d PRIVATE cxx_std_17)

target_compile_features(neumann_harmonic_jet_case_2d PRIVATE cxx_std_17)

target_compile_features(laplace_smooth_boundary_iteration_2d PRIVATE cxx_std_17)

target_compile_features(laplace_exterior_trace_flower_convergence_2d PRIVATE cxx_std_17)

target_compile_features(neumann_harmonic_jet_python_compatible_2d PRIVATE cxx_std_17)

target_compile_features(dirichlet_harmonic_jet_python_compatible_2d PRIVATE cxx_std_17)

if(KFBIM_BUILD_3D)
    target_compile_features(neumann_exterior_zero_trace_3d PRIVATE cxx_std_17)

    target_compile_features(
        neumann_exterior_zero_trace_3d_all_quartic PRIVATE cxx_std_17)

    target_compile_features(
        kfbi_native_c0_exterior_trace_3d PRIVATE cxx_std_17)

    target_compile_features(
        kfbi_topology_affine_exterior_trace_3d PRIVATE cxx_std_17)

    target_compile_features(
        kfbi_general_cap_exterior_trace_3d PRIVATE cxx_std_17)

endif()

foreach(target IN ITEMS
    shape_opt_transmission_2d
    transmission_center_perturb_2d
    neumann_exterior_trace_lshape_2d
    neumann_exterior_trace_lshape_slow_points_2d
    neumann_exterior_trace_circle_2d
    neumann_harmonic_jet_case_2d
    laplace_smooth_boundary_iteration_2d
    laplace_exterior_trace_flower_convergence_2d
    neumann_harmonic_jet_python_compatible_2d
    dirichlet_harmonic_jet_python_compatible_2d
    neumann_exterior_zero_trace_3d
    neumann_exterior_zero_trace_3d_all_quartic
    kfbi_native_c0_exterior_trace_3d
    kfbi_topology_affine_exterior_trace_3d
    kfbi_general_cap_exterior_trace_3d
)
    if(TARGET ${target})
        target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR})
        # Keep existing scripts and CLI examples valid on all generators.
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/apps")
    endif()
endforeach()
