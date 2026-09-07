# Source ownership follows the functional repository layout.
# Target names and per-target options are preserved for compatibility.

add_executable(laplace_nsc_et_benchmark_2d
    ${PROJECT_SOURCE_DIR}/benchmarks/2d/laplace_nsc_et_benchmark_2d.cpp
)

if(KFBIM_BUILD_3D)
    add_executable(data_reuse_3d_benchmark
        ${PROJECT_SOURCE_DIR}/benchmarks/3d/data_reuse_3d_benchmark.cpp
        ${PROJECT_SOURCE_DIR}/src/support/cauchy/crossing_cauchy_plan_3d.cpp
    )

endif()
if(KFBIM_BUILD_3D)
    if(KFBIM_BUILD_EXPERIMENTAL_3D)
        add_executable(nurbs_geometry_preprocess_benchmark_3d
            ${PROJECT_SOURCE_DIR}/benchmarks/3d/nurbs_geometry_preprocess_benchmark_3d.cpp
        )

    endif()
endif()
target_link_libraries(laplace_nsc_et_benchmark_2d PRIVATE
    kfbim_2d_benchmark_geometry
    kfbim_core
)

target_compile_definitions(laplace_nsc_et_benchmark_2d PRIVATE
    KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
)

if(KFBIM_BUILD_3D)
    target_link_libraries(data_reuse_3d_benchmark PRIVATE
        kfbim_3d_app_geometry
    )

endif()
if(KFBIM_BUILD_3D)
    if(KFBIM_BUILD_EXPERIMENTAL_3D)
        target_link_libraries(nurbs_geometry_preprocess_benchmark_3d PRIVATE
            kfbim_3d_app_geometry
        )

        target_compile_definitions(
            nurbs_geometry_preprocess_benchmark_3d PRIVATE
            KFBIM_NURBS_GEOMETRY_PREPROCESS_BENCHMARK_3D_MAIN=1
        )

    endif()
endif()
target_compile_features(laplace_nsc_et_benchmark_2d PRIVATE cxx_std_17)

if(KFBIM_BUILD_3D)
    target_compile_features(data_reuse_3d_benchmark PRIVATE cxx_std_17)

endif()
if(KFBIM_BUILD_3D)
    if(KFBIM_BUILD_EXPERIMENTAL_3D)
        target_compile_features(
            nurbs_geometry_preprocess_benchmark_3d PRIVATE cxx_std_17)

    endif()
endif()

foreach(target IN ITEMS
    laplace_nsc_et_benchmark_2d
    data_reuse_3d_benchmark
    nurbs_geometry_preprocess_benchmark_3d
)
    if(TARGET ${target})
        target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR})
        # Keep existing scripts and CLI examples valid on all generators.
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/apps")
    endif()
endforeach()
