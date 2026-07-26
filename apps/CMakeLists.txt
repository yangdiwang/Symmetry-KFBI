add_executable(shape_opt_transmission_2d
    shape_opt_transmission_2d.cpp
)

add_executable(transmission_center_perturb_2d
    transmission_center_perturb_2d.cpp
)

add_executable(neumann_exterior_trace_lshape_2d
    neumann_exterior_trace_lshape_2d.cpp
)

add_executable(neumann_exterior_trace_circle_2d
    neumann_exterior_trace_circle_2d.cpp
)

add_executable(neumann_harmonic_jet_case_2d
    neumann_harmonic_jet_case_2d.cpp
)

add_executable(neumann_harmonic_jet_python_compatible_2d
    neumann_harmonic_jet_python_compatible_2d.cpp
)

add_executable(dirichlet_harmonic_jet_python_compatible_2d
    neumann_harmonic_jet_python_compatible_2d.cpp
)

if(KFBIM_BUILD_3D)
    find_package(Threads REQUIRED)

    add_library(kfbim_3d_app_geometry STATIC
        harmonic_trace_correction_3d.cpp
        harmonic_cauchy_fit_3d.cpp
        crossing_owner_restrict_3d.cpp
        exterior_only_cubic_normal_restrict_3d.cpp
        harmonic_polynomial_space_3d.cpp
        native_nurbs_surface_3d.cpp
        native_nurbs_surface_transform_3d.cpp
        restrict_owner_geometry_preprocessor_3d.cpp
        dirichlet_rigid_transform_study_3d.cpp
        neumann_edge_continuity_3d.cpp
        neumann_rigid_transform_study_3d.cpp
        neumann_edge_augmented_cauchy_3d.cpp
        neumann_edge_cauchy_study_3d.cpp
    )

    add_library(kfbim_phase_profile_3d STATIC
        kfbi_phase_profile_3d.cpp
    )

    add_executable(native_nurbs_surface_3d_test
        native_nurbs_surface_3d_test.cpp
    )

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

    add_executable(native_nurbs_surface_transform_3d_test
        native_nurbs_surface_transform_3d_test.cpp
    )

    add_executable(dirichlet_rigid_transform_study_3d_test
        dirichlet_rigid_transform_study_3d_test.cpp
    )

    add_executable(neumann_rigid_transform_study_3d_test
        neumann_rigid_transform_study_3d_test.cpp
    )

    add_executable(harmonic_polynomial_space_3d_test
        harmonic_polynomial_space_3d_test.cpp
    )

    add_executable(exterior_only_cubic_normal_restrict_3d_test
        exterior_only_cubic_normal_restrict_3d_test.cpp
    )

    add_executable(crossing_owner_restrict_3d_test
        crossing_owner_restrict_3d_test.cpp
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

    add_executable(neumann_exterior_zero_trace_3d
        neumann_exterior_zero_trace_3d.cpp
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

target_link_libraries(neumann_exterior_trace_circle_2d PRIVATE
    kfbim_core
)

target_link_libraries(neumann_harmonic_jet_case_2d PRIVATE
    kfbim_core
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

    target_link_libraries(native_nurbs_surface_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(harmonic_cauchy_fit_3d_test PRIVATE
        kfbim_3d_app_geometry
    )
    target_compile_features(harmonic_cauchy_fit_3d_test PRIVATE cxx_std_17)

    target_link_libraries(neumann_edge_continuity_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(neumann_edge_augmented_cauchy_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(neumann_edge_cauchy_study_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(native_nurbs_surface_transform_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(dirichlet_rigid_transform_study_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(neumann_rigid_transform_study_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(harmonic_polynomial_space_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(exterior_only_cubic_normal_restrict_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(crossing_owner_restrict_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(restrict_owner_geometry_preprocessor_3d_test PRIVATE
        kfbim_3d_app_geometry
    )

    target_link_libraries(harmonic_trace_correction_3d_test PRIVATE
        kfbim_3d_app_geometry
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

    target_link_libraries(neumann_exterior_zero_trace_3d PRIVATE
        kfbim_3d_app_geometry
        kfbim_core
        kfbim_phase_profile_3d
        Threads::Threads
    )
    if(WIN32)
        target_link_libraries(neumann_exterior_zero_trace_3d PRIVATE Psapi)
    endif()
    if(MSVC)
        target_compile_options(neumann_exterior_zero_trace_3d PRIVATE /bigobj)
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

target_compile_definitions(neumann_exterior_trace_circle_2d PRIVATE
    KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
)

target_compile_definitions(neumann_harmonic_jet_case_2d PRIVATE
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
    target_compile_definitions(nurbs_geometry_preprocess_benchmark_3d PRIVATE
        KFBIM_NURBS_GEOMETRY_PREPROCESS_BENCHMARK_3D_MAIN=1
    )

    target_compile_definitions(neumann_exterior_zero_trace_3d PRIVATE
        KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
    )
endif()

target_compile_features(shape_opt_transmission_2d PRIVATE cxx_std_17)
target_compile_features(transmission_center_perturb_2d PRIVATE cxx_std_17)
target_compile_features(neumann_exterior_trace_lshape_2d PRIVATE cxx_std_17)
target_compile_features(neumann_exterior_trace_circle_2d PRIVATE cxx_std_17)
target_compile_features(neumann_harmonic_jet_case_2d PRIVATE cxx_std_17)
target_compile_features(neumann_harmonic_jet_python_compatible_2d PRIVATE cxx_std_17)
target_compile_features(dirichlet_harmonic_jet_python_compatible_2d PRIVATE cxx_std_17)
if(KFBIM_BUILD_3D)
    target_compile_features(kfbim_3d_app_geometry PUBLIC cxx_std_17)
    target_compile_features(kfbim_phase_profile_3d PUBLIC cxx_std_17)
    target_compile_features(neumann_edge_continuity_3d_test PRIVATE cxx_std_17)
    target_compile_features(neumann_edge_augmented_cauchy_3d_test PRIVATE cxx_std_17)
    target_compile_features(neumann_edge_cauchy_study_3d_test PRIVATE cxx_std_17)
    target_compile_features(native_nurbs_surface_3d_test PRIVATE cxx_std_17)
    target_compile_features(native_nurbs_surface_transform_3d_test PRIVATE cxx_std_17)
    target_compile_features(dirichlet_rigid_transform_study_3d_test PRIVATE cxx_std_17)
    target_compile_features(neumann_rigid_transform_study_3d_test PRIVATE cxx_std_17)
    target_compile_features(harmonic_polynomial_space_3d_test PRIVATE cxx_std_17)
    target_compile_features(exterior_only_cubic_normal_restrict_3d_test PRIVATE cxx_std_17)
    target_compile_features(crossing_owner_restrict_3d_test PRIVATE cxx_std_17)
    target_compile_features(kfbim_phase_profile_3d_test PRIVATE cxx_std_17)
    target_compile_features(restrict_owner_geometry_preprocessor_3d_test PRIVATE cxx_std_17)
    target_compile_features(harmonic_trace_correction_3d_test PRIVATE cxx_std_17)
    target_compile_features(nurbs_geometry_preprocess_benchmark_3d_test PRIVATE cxx_std_17)
    target_compile_features(nurbs_geometry_preprocess_benchmark_3d PRIVATE cxx_std_17)
    target_compile_features(neumann_exterior_zero_trace_3d PRIVATE cxx_std_17)
endif()
