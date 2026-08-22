add_executable(shape_opt_transmission_2d
    shape_opt_transmission_2d.cpp
)

add_executable(transmission_center_perturb_2d
    transmission_center_perturb_2d.cpp
)

add_library(kfbim_2d_benchmark_geometry STATIC
    benchmark_nurbs_geometries_2d.cpp
)

add_executable(benchmark_nurbs_geometries_2d_test
    benchmark_nurbs_geometries_2d_test.cpp
)

add_executable(laplace_nsc_et_benchmark_2d
    laplace_nsc_et_benchmark_2d.cpp
)

add_executable(laplace_nsc_et_bvp_2d_test
    laplace_nsc_et_bvp_2d_test.cpp
)

add_executable(neumann_exterior_trace_lshape_2d
    neumann_exterior_trace_lshape_2d.cpp
)

add_executable(neumann_exterior_trace_lshape_slow_points_2d
    neumann_exterior_trace_lshape_slow_points_2d.cpp
)

add_executable(neumann_exterior_trace_circle_2d
    neumann_exterior_trace_circle_2d.cpp
)

add_executable(neumann_harmonic_jet_case_2d
    neumann_harmonic_jet_case_2d.cpp
)

add_executable(curve_resampler_2d_test
    curve_resampler_2d_test.cpp
)

add_executable(laplace_crossing_local_polynomial_2d_test
    laplace_crossing_local_polynomial_2d_test.cpp
)

add_executable(nurbs_density_space_2d_test
    nurbs_density_space_2d_test.cpp
)

add_executable(laplace_smooth_boundary_iteration_2d
    laplace_smooth_boundary_iteration_2d.cpp
)

add_executable(laplace_exterior_trace_flower_convergence_2d
    laplace_exterior_trace_flower_convergence_2d.cpp
)

add_executable(neumann_harmonic_jet_python_compatible_2d
    neumann_harmonic_jet_python_compatible_2d.cpp
)

add_executable(dirichlet_harmonic_jet_python_compatible_2d
    neumann_harmonic_jet_python_compatible_2d.cpp
)

option(KFBIM_BUILD_EXPERIMENTAL_3D
    "Build supplementary remote 3D algorithms, studies, and benchmarks" OFF)
option(KFBIM_BUILD_REMOTE_SOLVER_ROUTE_TEST
    "Build the legacy remote solver route test (requires an API adapter)" OFF)

if(KFBIM_BUILD_REMOTE_SOLVER_ROUTE_TEST AND NOT KFBIM_BUILD_EXPERIMENTAL_3D)
    message(FATAL_ERROR
        "KFBIM_BUILD_REMOTE_SOLVER_ROUTE_TEST requires "
        "KFBIM_BUILD_EXPERIMENTAL_3D=ON")
endif()

if(KFBIM_BUILD_3D)
    add_library(kfbim_3d_app_geometry STATIC
        crossing_owner_restrict_3d.cpp
        direct_coefficient_cauchy_3d.cpp
        restrict_crossing_selector_3d.cpp
        restrict_crossing_path_state_3d.cpp
        edge_jump_jet_reduction_3d.cpp
        exterior_only_cubic_normal_restrict_3d.cpp
        geometric_feature_trace_fit_3d.cpp
        harmonic_polynomial_space_3d.cpp
        mean_free_trace_mass_coordinates_3d.cpp
        native_density_resolution_3d.cpp
        native_nurbs_density_space_3d.cpp
        native_nurbs_surface_3d.cpp
        native_nurbs_surface_transform_3d.cpp
        reduced_trace_projection_3d.cpp
        dirichlet_rigid_transform_study_3d.cpp
        shared_quadratic_restrict_3d.cpp
        topology_affine_reduction_3d.cpp
        topology_density_constraints_3d.cpp
        topology_mean_free_reduction_3d.cpp
        topology_reachable_target_projection_3d.cpp
        topology_trace_projector_3d.cpp
    )

    if(KFBIM_BUILD_EXPERIMENTAL_3D)
        find_package(Threads REQUIRED)

        # Supplementary A/B routes stay outside the baseline library so an
        # ordinary build retains the reference algorithm and dependencies.
        add_library(kfbim_3d_experimental_algorithms STATIC
            harmonic_trace_correction_3d.cpp
            harmonic_cauchy_fit_3d.cpp
            restrict_owner_geometry_preprocessor_3d.cpp
            neumann_edge_continuity_3d.cpp
            neumann_rigid_transform_study_3d.cpp
            neumann_edge_augmented_cauchy_3d.cpp
            neumann_edge_cauchy_study_3d.cpp
        )

        add_library(kfbim_phase_profile_3d STATIC
            kfbi_phase_profile_3d.cpp
        )
    endif()

    add_executable(native_nurbs_surface_3d_test
        native_nurbs_surface_3d_test.cpp
    )

    add_executable(grid_edge_event_3d_test
        grid_edge_event_3d_test.cpp
    )

    add_executable(u_prism_geometry_3d_test
        u_prism_geometry_3d_test.cpp
    )

    add_executable(smooth_t_topology_3d_test
        smooth_t_topology_3d_test.cpp
    )

    add_executable(native_density_resolution_3d_test
        native_density_resolution_3d_test.cpp
    )

    add_executable(native_nurbs_density_space_3d_test
        native_nurbs_density_space_3d_test.cpp
    )

    add_executable(edge_jump_jet_reduction_3d_test
        edge_jump_jet_reduction_3d_test.cpp
    )

    add_executable(native_nurbs_surface_transform_3d_test
        native_nurbs_surface_transform_3d_test.cpp
    )

    add_executable(dirichlet_rigid_transform_study_3d_test
        dirichlet_rigid_transform_study_3d_test.cpp
    )

    add_executable(harmonic_polynomial_space_3d_test
        harmonic_polynomial_space_3d_test.cpp
    )

    add_executable(mean_free_trace_mass_coordinates_3d_test
        mean_free_trace_mass_coordinates_3d_test.cpp
    )

    add_executable(geometric_feature_trace_fit_3d_test
        geometric_feature_trace_fit_3d_test.cpp
    )

    add_executable(exterior_only_cubic_normal_restrict_3d_test
        exterior_only_cubic_normal_restrict_3d_test.cpp
    )

    add_executable(crossing_owner_restrict_3d_test
        crossing_owner_restrict_3d_test.cpp
    )

    add_executable(restrict_crossing_selector_3d_test
        restrict_crossing_selector_3d_test.cpp
    )

    # Pure synthetic feature-event regression.  It links the selector directly
    # so exact-side classification can be exercised without the PDE stack.
    add_executable(restrict_crossing_feature_side_classifier_3d_test
        restrict_crossing_feature_side_classifier_3d_test.cpp
        restrict_crossing_selector_3d.cpp
    )

    # Pure state-machine regression: deliberately links no geometry/PDE
    # library so all-event continuation can be tested in isolation.
    add_executable(restrict_crossing_path_state_3d_test
        restrict_crossing_path_state_3d_test.cpp
        restrict_crossing_path_state_3d.cpp
    )

    add_executable(shared_quadratic_restrict_3d_test
        shared_quadratic_restrict_3d_test.cpp
    )

    add_executable(topology_affine_reduction_3d_test
        topology_affine_reduction_3d_test.cpp
    )

    add_executable(topology_density_constraints_3d_test
        topology_density_constraints_3d_test.cpp
    )

    add_executable(topology_mean_free_reduction_3d_test
        topology_mean_free_reduction_3d_test.cpp
    )

    add_executable(topology_reachable_target_projection_3d_test
        topology_reachable_target_projection_3d_test.cpp
    )

    add_executable(topology_trace_projector_3d_test
        topology_trace_projector_3d_test.cpp
    )

    add_executable(direct_coefficient_cauchy_3d_test
        direct_coefficient_cauchy_3d_test.cpp
        crossing_cauchy_plan_3d.cpp
    )

    if(KFBIM_BUILD_EXPERIMENTAL_3D)
    add_executable(harmonic_cauchy_fit_3d_test
        harmonic_cauchy_fit_3d_test.cpp
    )

    add_executable(neumann_edge_continuity_3d_test
        neumann_edge_continuity_3d_test.cpp
    )

    add_executable(neumann_edge_augmented_cauchy_3d_test
        neumann_edge_augmented_cauchy_3d_test.cpp
    )

    add_executable(neumann_edge_cauchy_study_3d_test
        neumann_edge_cauchy_study_3d_test.cpp
    )

    add_executable(neumann_rigid_transform_study_3d_test
        neumann_rigid_transform_study_3d_test.cpp
    )

    add_executable(restrict_owner_geometry_preprocessor_3d_test
        restrict_owner_geometry_preprocessor_3d_test.cpp
    )

    add_executable(harmonic_trace_correction_3d_test
        harmonic_trace_correction_3d_test.cpp
    )

    add_executable(kfbim_phase_profile_3d_test
        kfbi_phase_profile_3d_test.cpp
    )

    add_executable(nurbs_geometry_preprocess_benchmark_3d_test
        nurbs_geometry_preprocess_benchmark_3d.cpp
        nurbs_geometry_preprocess_benchmark_3d_test.cpp
    )

    add_executable(nurbs_geometry_preprocess_benchmark_3d
        nurbs_geometry_preprocess_benchmark_3d.cpp
    )

    endif()

    add_executable(neumann_exterior_zero_trace_3d
        neumann_exterior_zero_trace_3d.cpp
        crossing_cauchy_plan_3d.cpp
    )

    add_executable(neumann_exterior_zero_trace_3d_all_quartic
        neumann_exterior_zero_trace_3d.cpp
        crossing_cauchy_plan_3d.cpp
    )

    add_executable(kfbi_native_c0_exterior_trace_3d
        neumann_exterior_zero_trace_3d.cpp
        crossing_cauchy_plan_3d.cpp
    )

    add_executable(kfbi_topology_affine_exterior_trace_3d
        neumann_exterior_zero_trace_3d.cpp
        crossing_cauchy_plan_3d.cpp
    )

    add_executable(kfbi_general_cap_exterior_trace_3d
        kfbi_general_cap_exterior_trace_3d.cpp
        cap_atlas_density_space_3d.cpp
        crossing_cauchy_plan_3d.cpp
        mean_free_trace_mass_coordinates_3d.cpp
        reduced_trace_projection_3d.cpp
    )

    if(KFBIM_BUILD_EXPERIMENTAL_3D
       AND KFBIM_BUILD_REMOTE_SOLVER_ROUTE_TEST)
        add_executable(neumann_exterior_zero_trace_3d_route_test
            neumann_exterior_zero_trace_3d_route_test.cpp
        )
    endif()
endif()

target_link_libraries(shape_opt_transmission_2d PRIVATE
    kfbim_core
)

target_link_libraries(transmission_center_perturb_2d PRIVATE
    kfbim_core
)

target_link_libraries(kfbim_2d_benchmark_geometry PUBLIC
    kfbim_core
)

target_link_libraries(benchmark_nurbs_geometries_2d_test PRIVATE
    kfbim_2d_benchmark_geometry
)

target_link_libraries(laplace_nsc_et_benchmark_2d PRIVATE
    kfbim_2d_benchmark_geometry
    kfbim_core
)

target_link_libraries(laplace_nsc_et_bvp_2d_test PRIVATE
    kfbim_2d_benchmark_geometry
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

target_link_libraries(curve_resampler_2d_test PRIVATE
    kfbim_core
)

target_link_libraries(laplace_crossing_local_polynomial_2d_test PRIVATE
    kfbim_core
)

target_link_libraries(nurbs_density_space_2d_test PRIVATE
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

target_compile_definitions(laplace_nsc_et_benchmark_2d PRIVATE
    KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
)

target_link_libraries(neumann_harmonic_jet_python_compatible_2d PRIVATE
    kfbim_core
)

target_link_libraries(dirichlet_harmonic_jet_python_compatible_2d PRIVATE
    kfbim_core
)

if(KFBIM_BUILD_3D)
    target_link_libraries(kfbim_3d_app_geometry PUBLIC
        kfbim_core
    )

    if(KFBIM_BUILD_EXPERIMENTAL_3D)
        target_link_libraries(kfbim_3d_experimental_algorithms PUBLIC
            kfbim_3d_app_geometry
        )
        target_link_libraries(kfbim_phase_profile_3d PUBLIC
            Threads::Threads
        )
        if(WIN32)
            target_link_libraries(kfbim_phase_profile_3d PUBLIC Psapi)
        endif()
    endif()

    target_link_libraries(native_nurbs_surface_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(grid_edge_event_3d_test PRIVATE
        kfbim_core
    )

    target_link_libraries(u_prism_geometry_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(smooth_t_topology_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(native_density_resolution_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(native_nurbs_density_space_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(edge_jump_jet_reduction_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    if(KFBIM_BUILD_EXPERIMENTAL_3D)
    target_link_libraries(harmonic_cauchy_fit_3d_test PRIVATE
        kfbim_3d_experimental_algorithms
    )
    target_compile_features(harmonic_cauchy_fit_3d_test PRIVATE cxx_std_17)

    target_link_libraries(neumann_edge_continuity_3d_test PRIVATE
        kfbim_3d_experimental_algorithms
    )

    target_link_libraries(neumann_edge_augmented_cauchy_3d_test PRIVATE
        kfbim_3d_experimental_algorithms
    )

    target_link_libraries(neumann_edge_cauchy_study_3d_test PRIVATE
        kfbim_3d_experimental_algorithms
    )

    endif()

    target_link_libraries(native_nurbs_surface_transform_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(dirichlet_rigid_transform_study_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    if(KFBIM_BUILD_EXPERIMENTAL_3D)
        target_link_libraries(neumann_rigid_transform_study_3d_test PRIVATE
            kfbim_3d_experimental_algorithms
        )
    endif()

    target_link_libraries(harmonic_polynomial_space_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(mean_free_trace_mass_coordinates_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(geometric_feature_trace_fit_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(exterior_only_cubic_normal_restrict_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(crossing_owner_restrict_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(restrict_crossing_selector_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(
        restrict_crossing_feature_side_classifier_3d_test PRIVATE
        Eigen3::Eigen
    )
    target_include_directories(
        restrict_crossing_feature_side_classifier_3d_test PRIVATE
        ${PROJECT_SOURCE_DIR}
    )

    target_link_libraries(restrict_crossing_path_state_3d_test PRIVATE
        Eigen3::Eigen
    )

    target_link_libraries(shared_quadratic_restrict_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(topology_affine_reduction_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(topology_density_constraints_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(topology_mean_free_reduction_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(
        topology_reachable_target_projection_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(topology_trace_projector_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(direct_coefficient_cauchy_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    if(KFBIM_BUILD_EXPERIMENTAL_3D)
    target_link_libraries(restrict_owner_geometry_preprocessor_3d_test PRIVATE
        kfbim_3d_experimental_algorithms
    )

    target_link_libraries(harmonic_trace_correction_3d_test PRIVATE
        kfbim_3d_experimental_algorithms
    )

    target_link_libraries(kfbim_phase_profile_3d_test PRIVATE
        kfbim_phase_profile_3d
    )

    target_link_libraries(nurbs_geometry_preprocess_benchmark_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(nurbs_geometry_preprocess_benchmark_3d PRIVATE
        kfbim_3d_app_geometry
    )

    endif()

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

    if(KFBIM_BUILD_EXPERIMENTAL_3D
       AND KFBIM_BUILD_REMOTE_SOLVER_ROUTE_TEST)
        target_link_libraries(neumann_exterior_zero_trace_3d_route_test PRIVATE
            kfbim_3d_experimental_algorithms
            kfbim_3d_app_geometry
            kfbim_core
            kfbim_phase_profile_3d
            Threads::Threads
        )
        if(MSVC)
            target_compile_options(
                neumann_exterior_zero_trace_3d_route_test PRIVATE /bigobj)
        endif()
    endif()
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
    if(KFBIM_BUILD_EXPERIMENTAL_3D)
        target_compile_definitions(
            nurbs_geometry_preprocess_benchmark_3d PRIVATE
            KFBIM_NURBS_GEOMETRY_PREPROCESS_BENCHMARK_3D_MAIN=1
        )
    endif()

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
target_compile_features(kfbim_2d_benchmark_geometry PUBLIC cxx_std_17)
target_compile_features(benchmark_nurbs_geometries_2d_test PRIVATE cxx_std_17)
target_compile_features(laplace_nsc_et_benchmark_2d PRIVATE cxx_std_17)
target_compile_features(laplace_nsc_et_bvp_2d_test PRIVATE cxx_std_17)
target_compile_features(neumann_exterior_trace_lshape_2d PRIVATE cxx_std_17)
target_compile_features(neumann_exterior_trace_circle_2d PRIVATE cxx_std_17)
target_compile_features(neumann_harmonic_jet_case_2d PRIVATE cxx_std_17)
target_compile_features(curve_resampler_2d_test PRIVATE cxx_std_17)
target_compile_features(laplace_crossing_local_polynomial_2d_test PRIVATE cxx_std_17)
target_compile_features(nurbs_density_space_2d_test PRIVATE cxx_std_17)
target_compile_features(laplace_smooth_boundary_iteration_2d PRIVATE cxx_std_17)
target_compile_features(laplace_exterior_trace_flower_convergence_2d PRIVATE cxx_std_17)
target_compile_features(neumann_harmonic_jet_python_compatible_2d PRIVATE cxx_std_17)
target_compile_features(dirichlet_harmonic_jet_python_compatible_2d PRIVATE cxx_std_17)
if(KFBIM_BUILD_3D)
    target_compile_features(kfbim_3d_app_geometry PUBLIC cxx_std_17)
    target_compile_features(native_nurbs_surface_3d_test PRIVATE cxx_std_17)
    target_compile_features(grid_edge_event_3d_test PRIVATE cxx_std_17)
    target_compile_features(u_prism_geometry_3d_test PRIVATE cxx_std_17)
    target_compile_features(smooth_t_topology_3d_test PRIVATE cxx_std_17)
    target_compile_features(native_density_resolution_3d_test PRIVATE cxx_std_17)
    target_compile_features(native_nurbs_density_space_3d_test PRIVATE cxx_std_17)
    target_compile_features(edge_jump_jet_reduction_3d_test PRIVATE cxx_std_17)
    target_compile_features(native_nurbs_surface_transform_3d_test PRIVATE cxx_std_17)
    target_compile_features(dirichlet_rigid_transform_study_3d_test PRIVATE cxx_std_17)
    target_compile_features(harmonic_polynomial_space_3d_test PRIVATE cxx_std_17)
    target_compile_features(
        mean_free_trace_mass_coordinates_3d_test PRIVATE cxx_std_17)
    target_compile_features(
        geometric_feature_trace_fit_3d_test PRIVATE cxx_std_17)
    target_compile_features(exterior_only_cubic_normal_restrict_3d_test PRIVATE cxx_std_17)
    target_compile_features(crossing_owner_restrict_3d_test PRIVATE cxx_std_17)
    target_compile_features(restrict_crossing_selector_3d_test PRIVATE cxx_std_17)
    target_compile_features(
        restrict_crossing_feature_side_classifier_3d_test PRIVATE cxx_std_17)
    target_compile_features(restrict_crossing_path_state_3d_test PRIVATE cxx_std_17)
    target_compile_features(shared_quadratic_restrict_3d_test PRIVATE cxx_std_17)
    target_compile_features(topology_affine_reduction_3d_test PRIVATE cxx_std_17)
    target_compile_features(topology_density_constraints_3d_test PRIVATE cxx_std_17)
    target_compile_features(topology_mean_free_reduction_3d_test PRIVATE cxx_std_17)
    target_compile_features(
        topology_reachable_target_projection_3d_test PRIVATE cxx_std_17)
    target_compile_features(topology_trace_projector_3d_test PRIVATE cxx_std_17)
    target_compile_features(direct_coefficient_cauchy_3d_test PRIVATE cxx_std_17)
    target_compile_features(neumann_exterior_zero_trace_3d PRIVATE cxx_std_17)
    target_compile_features(
        neumann_exterior_zero_trace_3d_all_quartic PRIVATE cxx_std_17)
    target_compile_features(
        kfbi_native_c0_exterior_trace_3d PRIVATE cxx_std_17)
    target_compile_features(
        kfbi_topology_affine_exterior_trace_3d PRIVATE cxx_std_17)
    target_compile_features(
        kfbi_general_cap_exterior_trace_3d PRIVATE cxx_std_17)

    if(KFBIM_BUILD_EXPERIMENTAL_3D)
        target_compile_features(
            kfbim_3d_experimental_algorithms PUBLIC cxx_std_17)
        target_compile_features(kfbim_phase_profile_3d PUBLIC cxx_std_17)
        target_compile_features(harmonic_cauchy_fit_3d_test PRIVATE cxx_std_17)
        target_compile_features(neumann_edge_continuity_3d_test PRIVATE cxx_std_17)
        target_compile_features(
            neumann_edge_augmented_cauchy_3d_test PRIVATE cxx_std_17)
        target_compile_features(neumann_edge_cauchy_study_3d_test PRIVATE cxx_std_17)
        target_compile_features(
            neumann_rigid_transform_study_3d_test PRIVATE cxx_std_17)
        target_compile_features(kfbim_phase_profile_3d_test PRIVATE cxx_std_17)
        target_compile_features(
            restrict_owner_geometry_preprocessor_3d_test PRIVATE cxx_std_17)
        target_compile_features(harmonic_trace_correction_3d_test PRIVATE cxx_std_17)
        target_compile_features(
            nurbs_geometry_preprocess_benchmark_3d_test PRIVATE cxx_std_17)
        target_compile_features(
            nurbs_geometry_preprocess_benchmark_3d PRIVATE cxx_std_17)
    endif()

    if(KFBIM_BUILD_EXPERIMENTAL_3D
       AND KFBIM_BUILD_REMOTE_SOLVER_ROUTE_TEST)
        target_compile_definitions(
            neumann_exterior_zero_trace_3d_route_test PRIVATE
            KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
        )
        target_compile_features(
            neumann_exterior_zero_trace_3d_route_test PRIVATE cxx_std_17)
    endif()
endif()
