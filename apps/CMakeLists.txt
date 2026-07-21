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
    add_library(kfbim_3d_app_geometry STATIC
        native_nurbs_surface_3d.cpp
    )

    add_executable(native_nurbs_surface_3d_test
        native_nurbs_surface_3d_test.cpp
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

    target_link_libraries(neumann_exterior_zero_trace_3d PRIVATE
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
    target_compile_features(native_nurbs_surface_3d_test PRIVATE cxx_std_17)
    target_compile_features(neumann_exterior_zero_trace_3d PRIVATE cxx_std_17)
endif()
