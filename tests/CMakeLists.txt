# Source ownership follows the functional repository layout.
# Target names and per-target options are preserved for compatibility.

if(KFBIM_BUILD_3D)
    foreach(case IN ITEMS extended_tubular_cauchy_3d python_torus_density_layout_3d trace_first_case_3d nurbs_grid_line_cache_3d trace_first_resource_3d nurbs_torus_empty_edge_replay_3d trace93_case_3d trace93_density_layout_3d trace93_resource_3d)
        if(case STREQUAL "extended_tubular_cauchy_3d")
            set(case_dir cauchy)
        elseif(case STREQUAL "python_torus_density_layout_3d" OR case STREQUAL "trace93_density_layout_3d")
            set(case_dir density)
        elseif(case STREQUAL "trace_first_resource_3d" OR case STREQUAL "trace93_resource_3d")
            set(case_dir trace)
        else()
            set(case_dir geometry)
        endif()
        add_executable(${case}_test ${PROJECT_SOURCE_DIR}/tests/${case_dir}/${case}_test.cpp)
        target_link_libraries(${case}_test PRIVATE kfbim_3d_app_geometry kfbim_core)
        target_include_directories(${case}_test PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_features(${case}_test PRIVATE cxx_std_17)
        set_target_properties(${case}_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/apps")
        add_test(NAME ${case}_test COMMAND $<TARGET_FILE:${case}_test>)
        set_tests_properties(${case}_test PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endforeach()
endif()

add_executable(benchmark_nurbs_geometries_2d_test
    ${PROJECT_SOURCE_DIR}/tests/geometry/benchmark_nurbs_geometries_2d_test.cpp
)

add_executable(laplace_nsc_et_bvp_2d_test
    ${PROJECT_SOURCE_DIR}/tests/solvers/laplace_nsc_et_bvp_2d_test.cpp
)

add_executable(curve_resampler_2d_test
    ${PROJECT_SOURCE_DIR}/tests/geometry/curve_resampler_2d_test.cpp
)

add_executable(laplace_crossing_local_polynomial_2d_test
    ${PROJECT_SOURCE_DIR}/tests/cauchy/laplace_crossing_local_polynomial_2d_test.cpp
)

add_executable(nurbs_density_space_2d_test
    ${PROJECT_SOURCE_DIR}/tests/density/nurbs_density_space_2d_test.cpp
)

if(KFBIM_BUILD_3D)
    add_executable(direct_coefficient_cubic_cauchy_3d_test
        ${PROJECT_SOURCE_DIR}/tests/cauchy/direct_coefficient_cubic_cauchy_3d_test.cpp
        ${PROJECT_SOURCE_DIR}/src/support/cauchy/crossing_cauchy_plan_3d.cpp)
    target_link_libraries(direct_coefficient_cubic_cauchy_3d_test PRIVATE kfbim_3d_app_geometry)
    target_compile_features(direct_coefficient_cubic_cauchy_3d_test PRIVATE cxx_std_17)

    add_executable(restrict_resource_plan_3d_test
        ${PROJECT_SOURCE_DIR}/tests/trace/restrict_resource_plan_3d_test.cpp)
    target_link_libraries(restrict_resource_plan_3d_test PRIVATE kfbim_3d_app_geometry)
    target_compile_features(restrict_resource_plan_3d_test PRIVATE cxx_std_17)

    add_executable(resource_restrict_assembly_3d_test
        ${PROJECT_SOURCE_DIR}/tests/trace/resource_restrict_assembly_3d_test.cpp)
    target_link_libraries(resource_restrict_assembly_3d_test PRIVATE kfbim_3d_app_geometry)
    target_compile_features(resource_restrict_assembly_3d_test PRIVATE cxx_std_17)

    add_executable(closest_point_trace_extension_3d_test
        ${PROJECT_SOURCE_DIR}/tests/trace/closest_point_trace_extension_3d_test.cpp
        ${PROJECT_SOURCE_DIR}/src/support/cauchy/crossing_cauchy_plan_3d.cpp)
    target_link_libraries(closest_point_trace_extension_3d_test PRIVATE
        kfbim_3d_app_geometry kfbim_core CGAL::CGAL)
    target_compile_features(closest_point_trace_extension_3d_test PRIVATE cxx_std_17)

    add_executable(nurbs_surface_closest_point_3d_test
        ${PROJECT_SOURCE_DIR}/tests/geometry/nurbs_surface_closest_point_3d_test.cpp)
    target_link_libraries(nurbs_surface_closest_point_3d_test PRIVATE
        kfbim_core CGAL::CGAL)
    target_compile_features(nurbs_surface_closest_point_3d_test PRIVATE cxx_std_17)

    add_executable(nurbs_patch_polar_evaluator_3d_test
        ${PROJECT_SOURCE_DIR}/tests/geometry/nurbs_patch_polar_evaluator_3d_test.cpp)
    target_link_libraries(nurbs_patch_polar_evaluator_3d_test PRIVATE
        kfbim_core CGAL::CGAL)
    target_compile_features(nurbs_patch_polar_evaluator_3d_test PRIVATE cxx_std_17)

    add_executable(nurbs_polar_intersection_3d_test
        ${PROJECT_SOURCE_DIR}/tests/geometry/nurbs_polar_intersection_3d_test.cpp)
    target_link_libraries(nurbs_polar_intersection_3d_test PRIVATE
        kfbim_3d_app_geometry kfbim_core CGAL::CGAL)
    target_compile_features(nurbs_polar_intersection_3d_test PRIVATE cxx_std_17)

    add_executable(native_nurbs_surface_3d_test
        ${PROJECT_SOURCE_DIR}/tests/geometry/native_nurbs_surface_3d_test.cpp
    )

    add_executable(native_endpoint_path_3d_test
        ${PROJECT_SOURCE_DIR}/tests/geometry/native_endpoint_path_3d_test.cpp
    )

    target_link_libraries(native_endpoint_path_3d_test PRIVATE
        kfbim_3d_app_geometry kfbim_core CGAL::CGAL)

    target_compile_features(native_endpoint_path_3d_test PRIVATE cxx_std_17)

    add_executable(native_nurbs_exact_geometry_3d_test
        ${PROJECT_SOURCE_DIR}/tests/geometry/native_nurbs_exact_geometry_3d_test.cpp
    )

    target_link_libraries(native_nurbs_exact_geometry_3d_test PRIVATE
        kfbim_core CGAL::CGAL)

    target_compile_features(native_nurbs_exact_geometry_3d_test PRIVATE cxx_std_17)

    add_executable(grid_edge_event_3d_test
        ${PROJECT_SOURCE_DIR}/tests/geometry/grid_edge_event_3d_test.cpp
    )

    add_executable(u_prism_geometry_3d_test
        ${PROJECT_SOURCE_DIR}/tests/geometry/u_prism_geometry_3d_test.cpp
    )

    add_executable(smooth_t_topology_3d_test
        ${PROJECT_SOURCE_DIR}/tests/topology/smooth_t_topology_3d_test.cpp
    )

    add_executable(native_density_resolution_3d_test
        ${PROJECT_SOURCE_DIR}/tests/density/native_density_resolution_3d_test.cpp
    )

    add_executable(native_nurbs_density_space_3d_test
        ${PROJECT_SOURCE_DIR}/tests/density/native_nurbs_density_space_3d_test.cpp
    )

    add_executable(edge_jump_jet_reduction_3d_test
        ${PROJECT_SOURCE_DIR}/tests/topology/edge_jump_jet_reduction_3d_test.cpp
    )

    add_executable(native_nurbs_surface_transform_3d_test
        ${PROJECT_SOURCE_DIR}/tests/geometry/native_nurbs_surface_transform_3d_test.cpp
    )

    add_executable(dirichlet_rigid_transform_study_3d_test
        ${PROJECT_SOURCE_DIR}/tests/diagnostics/dirichlet_rigid_transform_study_3d_test.cpp
    )

    add_executable(harmonic_polynomial_space_3d_test
        ${PROJECT_SOURCE_DIR}/tests/cauchy/harmonic_polynomial_space_3d_test.cpp
    )

    add_executable(mean_free_trace_mass_coordinates_3d_test
        ${PROJECT_SOURCE_DIR}/tests/trace/mean_free_trace_mass_coordinates_3d_test.cpp
    )

    add_executable(geometric_feature_trace_fit_3d_test
        ${PROJECT_SOURCE_DIR}/tests/trace/geometric_feature_trace_fit_3d_test.cpp
    )

    add_executable(exterior_only_cubic_normal_restrict_3d_test
        ${PROJECT_SOURCE_DIR}/tests/trace/exterior_only_cubic_normal_restrict_3d_test.cpp
    )

    add_executable(crossing_owner_restrict_3d_test
        ${PROJECT_SOURCE_DIR}/tests/trace/crossing_owner_restrict_3d_test.cpp
    )

    add_executable(restrict_crossing_selector_3d_test
        ${PROJECT_SOURCE_DIR}/tests/trace/restrict_crossing_selector_3d_test.cpp
    )

    add_executable(restrict_crossing_feature_side_classifier_3d_test
        ${PROJECT_SOURCE_DIR}/tests/trace/restrict_crossing_feature_side_classifier_3d_test.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/restrict_crossing_selector_3d.cpp
    )

    add_executable(restrict_crossing_path_state_3d_test
        ${PROJECT_SOURCE_DIR}/tests/trace/restrict_crossing_path_state_3d_test.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/restrict_crossing_path_state_3d.cpp
    )

    add_executable(shared_quadratic_restrict_3d_test
        ${PROJECT_SOURCE_DIR}/tests/trace/shared_quadratic_restrict_3d_test.cpp
    )

    add_executable(tensor_product_cover_restrict_3d_test
        ${PROJECT_SOURCE_DIR}/tests/trace/tensor_product_cover_restrict_3d_test.cpp
    )

    add_executable(topology_affine_reduction_3d_test
        ${PROJECT_SOURCE_DIR}/tests/topology/topology_affine_reduction_3d_test.cpp
    )

    add_executable(topology_density_constraints_3d_test
        ${PROJECT_SOURCE_DIR}/tests/topology/topology_density_constraints_3d_test.cpp
    )

    add_executable(topology_mean_free_reduction_3d_test
        ${PROJECT_SOURCE_DIR}/tests/topology/topology_mean_free_reduction_3d_test.cpp
    )

    add_executable(topology_reachable_target_projection_3d_test
        ${PROJECT_SOURCE_DIR}/tests/topology/topology_reachable_target_projection_3d_test.cpp
    )

    add_executable(topology_trace_projector_3d_test
        ${PROJECT_SOURCE_DIR}/tests/topology/topology_trace_projector_3d_test.cpp
    )

    add_executable(direct_coefficient_cauchy_3d_test
        ${PROJECT_SOURCE_DIR}/tests/cauchy/direct_coefficient_cauchy_3d_test.cpp
        ${PROJECT_SOURCE_DIR}/src/support/cauchy/crossing_cauchy_plan_3d.cpp
    )

endif()
if(KFBIM_BUILD_3D)
    if(KFBIM_BUILD_EXPERIMENTAL_3D)
        add_executable(harmonic_cauchy_fit_3d_test
            ${PROJECT_SOURCE_DIR}/tests/cauchy/harmonic_cauchy_fit_3d_test.cpp
        )

        add_executable(neumann_edge_continuity_3d_test
            ${PROJECT_SOURCE_DIR}/tests/topology/neumann_edge_continuity_3d_test.cpp
        )

        add_executable(neumann_edge_augmented_cauchy_3d_test
            ${PROJECT_SOURCE_DIR}/tests/cauchy/neumann_edge_augmented_cauchy_3d_test.cpp
        )

        add_executable(neumann_edge_cauchy_study_3d_test
            ${PROJECT_SOURCE_DIR}/tests/diagnostics/neumann_edge_cauchy_study_3d_test.cpp
        )

        add_executable(neumann_rigid_transform_study_3d_test
            ${PROJECT_SOURCE_DIR}/tests/diagnostics/neumann_rigid_transform_study_3d_test.cpp
        )

        add_executable(restrict_owner_geometry_preprocessor_3d_test
            ${PROJECT_SOURCE_DIR}/tests/trace/restrict_owner_geometry_preprocessor_3d_test.cpp
        )

        add_executable(harmonic_trace_correction_3d_test
            ${PROJECT_SOURCE_DIR}/tests/trace/harmonic_trace_correction_3d_test.cpp
        )

        add_executable(kfbim_phase_profile_3d_test
            ${PROJECT_SOURCE_DIR}/tests/diagnostics/kfbi_phase_profile_3d_test.cpp
        )

        add_executable(nurbs_geometry_preprocess_benchmark_3d_test
            ${PROJECT_SOURCE_DIR}/benchmarks/3d/nurbs_geometry_preprocess_benchmark_3d.cpp
            ${PROJECT_SOURCE_DIR}/tests/diagnostics/nurbs_geometry_preprocess_benchmark_3d_test.cpp
        )

    endif()
endif()
if(KFBIM_BUILD_3D)
    if(KFBIM_BUILD_EXPERIMENTAL_3D
        AND KFBIM_BUILD_REMOTE_SOLVER_ROUTE_TEST)
        add_executable(neumann_exterior_zero_trace_3d_route_test
            ${PROJECT_SOURCE_DIR}/tests/solvers/neumann_exterior_zero_trace_3d_route_test.cpp
        )

    endif()
endif()
target_link_libraries(benchmark_nurbs_geometries_2d_test PRIVATE
    kfbim_2d_benchmark_geometry
)

target_link_libraries(laplace_nsc_et_bvp_2d_test PRIVATE
    kfbim_2d_benchmark_geometry
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

if(KFBIM_BUILD_3D)
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

endif()
if(KFBIM_BUILD_3D)
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
endif()
if(KFBIM_BUILD_3D)
    target_link_libraries(native_nurbs_surface_transform_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(dirichlet_rigid_transform_study_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

endif()
if(KFBIM_BUILD_3D)
    if(KFBIM_BUILD_EXPERIMENTAL_3D)
        target_link_libraries(neumann_rigid_transform_study_3d_test PRIVATE
            kfbim_3d_experimental_algorithms
        )

    endif()
endif()
if(KFBIM_BUILD_3D)
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

    target_link_libraries(tensor_product_cover_restrict_3d_test PRIVATE
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

endif()
if(KFBIM_BUILD_3D)
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

    endif()
endif()
if(KFBIM_BUILD_3D)
    if(KFBIM_BUILD_EXPERIMENTAL_3D
        AND KFBIM_BUILD_REMOTE_SOLVER_ROUTE_TEST)
        target_link_libraries(neumann_exterior_zero_trace_3d_route_test PRIVATE
            kfbim_3d_experimental_algorithms
            kfbim_3d_app_geometry
            kfbim_core
            kfbim_phase_profile_3d
            Threads::Threads
        )

    endif()
endif()
if(KFBIM_BUILD_3D)
    if(KFBIM_BUILD_EXPERIMENTAL_3D
        AND KFBIM_BUILD_REMOTE_SOLVER_ROUTE_TEST)
        if(MSVC)
            target_compile_options(
                neumann_exterior_zero_trace_3d_route_test PRIVATE /bigobj)

        endif()
    endif()
endif()
target_compile_features(benchmark_nurbs_geometries_2d_test PRIVATE cxx_std_17)

target_compile_features(laplace_nsc_et_bvp_2d_test PRIVATE cxx_std_17)

target_compile_features(curve_resampler_2d_test PRIVATE cxx_std_17)

target_compile_features(laplace_crossing_local_polynomial_2d_test PRIVATE cxx_std_17)

target_compile_features(nurbs_density_space_2d_test PRIVATE cxx_std_17)

if(KFBIM_BUILD_3D)
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

    target_compile_features(tensor_product_cover_restrict_3d_test PRIVATE cxx_std_17)

    target_compile_features(topology_affine_reduction_3d_test PRIVATE cxx_std_17)

    target_compile_features(topology_density_constraints_3d_test PRIVATE cxx_std_17)

    target_compile_features(topology_mean_free_reduction_3d_test PRIVATE cxx_std_17)

    target_compile_features(
        topology_reachable_target_projection_3d_test PRIVATE cxx_std_17)

    target_compile_features(topology_trace_projector_3d_test PRIVATE cxx_std_17)

    target_compile_features(direct_coefficient_cauchy_3d_test PRIVATE cxx_std_17)

endif()
if(KFBIM_BUILD_3D)
    if(KFBIM_BUILD_EXPERIMENTAL_3D)
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

    endif()
endif()
if(KFBIM_BUILD_3D)
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

foreach(target IN ITEMS
    restrict_resource_plan_3d_test
    resource_restrict_assembly_3d_test
    direct_coefficient_cubic_cauchy_3d_test
    benchmark_nurbs_geometries_2d_test
    laplace_nsc_et_bvp_2d_test
    curve_resampler_2d_test
    laplace_crossing_local_polynomial_2d_test
    nurbs_density_space_2d_test
    nurbs_surface_closest_point_3d_test
    closest_point_trace_extension_3d_test
    nurbs_patch_polar_evaluator_3d_test
    nurbs_polar_intersection_3d_test
    native_nurbs_surface_3d_test
    native_endpoint_path_3d_test
    native_nurbs_exact_geometry_3d_test
    grid_edge_event_3d_test
    u_prism_geometry_3d_test
    smooth_t_topology_3d_test
    native_density_resolution_3d_test
    native_nurbs_density_space_3d_test
    edge_jump_jet_reduction_3d_test
    native_nurbs_surface_transform_3d_test
    dirichlet_rigid_transform_study_3d_test
    harmonic_polynomial_space_3d_test
    mean_free_trace_mass_coordinates_3d_test
    geometric_feature_trace_fit_3d_test
    exterior_only_cubic_normal_restrict_3d_test
    crossing_owner_restrict_3d_test
    restrict_crossing_selector_3d_test
    restrict_crossing_feature_side_classifier_3d_test
    restrict_crossing_path_state_3d_test
    shared_quadratic_restrict_3d_test
    tensor_product_cover_restrict_3d_test
    topology_affine_reduction_3d_test
    topology_density_constraints_3d_test
    topology_mean_free_reduction_3d_test
    topology_reachable_target_projection_3d_test
    topology_trace_projector_3d_test
    direct_coefficient_cauchy_3d_test
    harmonic_cauchy_fit_3d_test
    neumann_edge_continuity_3d_test
    neumann_edge_augmented_cauchy_3d_test
    neumann_edge_cauchy_study_3d_test
    neumann_rigid_transform_study_3d_test
    restrict_owner_geometry_preprocessor_3d_test
    harmonic_trace_correction_3d_test
    kfbim_phase_profile_3d_test
    nurbs_geometry_preprocess_benchmark_3d_test
    neumann_exterior_zero_trace_3d_route_test
)
    if(TARGET ${target})
        target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR})
        # Keep existing scripts and CLI examples valid on all generators.
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/apps")
        add_test(NAME ${target} COMMAND $<TARGET_FILE:${target}>)
        set_tests_properties(${target} PROPERTIES
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endif()
endforeach()
