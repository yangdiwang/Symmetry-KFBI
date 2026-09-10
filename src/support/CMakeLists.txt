# Source ownership follows the functional repository layout.
# Target names and per-target options are preserved for compatibility.

add_library(kfbim_2d_benchmark_geometry STATIC
    ${PROJECT_SOURCE_DIR}/src/support/geometry/benchmark_nurbs_geometries_2d.cpp
)

if(KFBIM_BUILD_3D)
    add_library(kfbim_3d_app_geometry STATIC
        ${PROJECT_SOURCE_DIR}/src/support/geometry/trace93_case_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/density/trace93_density_layout_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/trace93_resource_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/geometry/trace_first_case_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/density/python_torus_density_layout_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/cauchy/extended_tubular_cauchy_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/cauchy/crossing_cauchy_plan_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/trace_first_resource_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/closest_point_trace_extension_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/crossing_owner_restrict_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/cauchy/direct_coefficient_cauchy_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/cauchy/direct_coefficient_cubic_cauchy_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/restrict_resource_plan_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/resource_restrict_assembly_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/spread_restrict_resource_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/restrict_crossing_selector_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/restrict_crossing_path_state_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/topology/edge_jump_jet_reduction_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/exterior_only_cubic_normal_restrict_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/geometric_feature_trace_fit_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/cauchy/harmonic_polynomial_space_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/mean_free_trace_mass_coordinates_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/density/native_density_resolution_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/density/native_nurbs_density_space_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/geometry/native_nurbs_surface_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/geometry/native_nurbs_surface_transform_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/native_endpoint_path_adapter_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/reduced_trace_projection_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/diagnostics/dirichlet_rigid_transform_study_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/shared_quadratic_restrict_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/trace/tensor_product_cover_restrict_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/topology/topology_affine_reduction_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/topology/topology_density_constraints_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/topology/topology_mean_free_reduction_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/topology/topology_reachable_target_projection_3d.cpp
        ${PROJECT_SOURCE_DIR}/src/support/topology/topology_trace_projector_3d.cpp
    )

endif()
if(KFBIM_BUILD_3D)
    if(KFBIM_BUILD_EXPERIMENTAL_3D)
        find_package(Threads REQUIRED)

        add_library(kfbim_3d_experimental_algorithms STATIC
            ${PROJECT_SOURCE_DIR}/src/support/trace/harmonic_trace_correction_3d.cpp
            ${PROJECT_SOURCE_DIR}/src/support/cauchy/harmonic_cauchy_fit_3d.cpp
            ${PROJECT_SOURCE_DIR}/src/support/trace/restrict_owner_geometry_preprocessor_3d.cpp
            ${PROJECT_SOURCE_DIR}/src/support/topology/neumann_edge_continuity_3d.cpp
            ${PROJECT_SOURCE_DIR}/src/support/diagnostics/neumann_rigid_transform_study_3d.cpp
            ${PROJECT_SOURCE_DIR}/src/support/cauchy/neumann_edge_augmented_cauchy_3d.cpp
            ${PROJECT_SOURCE_DIR}/src/support/diagnostics/neumann_edge_cauchy_study_3d.cpp
        )

        add_library(kfbim_phase_profile_3d STATIC
            ${PROJECT_SOURCE_DIR}/src/support/diagnostics/kfbi_phase_profile_3d.cpp
        )

    endif()
endif()
target_link_libraries(kfbim_2d_benchmark_geometry PUBLIC
    kfbim_core
)

if(KFBIM_BUILD_3D)
    target_link_libraries(kfbim_3d_app_geometry PUBLIC
        kfbim_core
    )

endif()
if(KFBIM_BUILD_3D)
    if(KFBIM_BUILD_EXPERIMENTAL_3D)
        target_link_libraries(kfbim_3d_experimental_algorithms PUBLIC
            kfbim_3d_app_geometry
        )

        target_link_libraries(kfbim_phase_profile_3d PUBLIC
            Threads::Threads
        )

    endif()
endif()
if(KFBIM_BUILD_3D)
    if(KFBIM_BUILD_EXPERIMENTAL_3D)
        if(WIN32)
            target_link_libraries(kfbim_phase_profile_3d PUBLIC Psapi)

        endif()
    endif()
endif()
target_compile_features(kfbim_2d_benchmark_geometry PUBLIC cxx_std_17)

if(KFBIM_BUILD_3D)
    target_compile_features(kfbim_3d_app_geometry PUBLIC cxx_std_17)

endif()
if(KFBIM_BUILD_3D)
    if(KFBIM_BUILD_EXPERIMENTAL_3D)
        target_compile_features(
            kfbim_3d_experimental_algorithms PUBLIC cxx_std_17)

        target_compile_features(kfbim_phase_profile_3d PUBLIC cxx_std_17)

    endif()
endif()

foreach(target IN ITEMS
    kfbim_2d_benchmark_geometry
    kfbim_3d_app_geometry
    kfbim_3d_experimental_algorithms
    kfbim_phase_profile_3d
)
    if(TARGET ${target})
        target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR})
    endif()
endforeach()
