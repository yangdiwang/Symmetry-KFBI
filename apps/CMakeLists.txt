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

target_compile_features(shape_opt_transmission_2d PRIVATE cxx_std_17)
target_compile_features(transmission_center_perturb_2d PRIVATE cxx_std_17)
target_compile_features(neumann_exterior_trace_lshape_2d PRIVATE cxx_std_17)
target_compile_features(neumann_exterior_trace_circle_2d PRIVATE cxx_std_17)
target_compile_features(neumann_harmonic_jet_case_2d PRIVATE cxx_std_17)
target_compile_features(neumann_harmonic_jet_python_compatible_2d PRIVATE cxx_std_17)
