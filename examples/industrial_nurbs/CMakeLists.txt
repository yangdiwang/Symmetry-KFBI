cmake_minimum_required(VERSION 3.20)
project(IndustrialNurbsExamples LANGUAGES CXX)

# Standalone Eigen-only build of the same geometry used by kfbim_core.
# This gallery does not require the CGAL crossing pipeline or a PDE solver.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
get_filename_component(KFBI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
include("${KFBI_ROOT}/src/geometry/models3d/sources.cmake")
add_library(industrial_nurbs_geometry STATIC
    "${KFBI_ROOT}/src/geometry/nurbs_surface_model_3d.cpp"
    ${KFBIM_INDUSTRIAL_MODEL_SOURCES_3D})
target_include_directories(industrial_nurbs_geometry PUBLIC "${KFBI_ROOT}")
target_link_libraries(industrial_nurbs_geometry PUBLIC Eigen3::Eigen)
if(MSVC)
    target_compile_options(industrial_nurbs_geometry PUBLIC /utf-8)
endif()
add_executable(industrial_nurbs_gallery_3d "${KFBI_ROOT}/apps/geometry/industrial_nurbs_gallery_3d.cpp")
target_link_libraries(industrial_nurbs_gallery_3d PRIVATE industrial_nurbs_geometry)
add_executable(industrial_nurbs_models_3d_test "${KFBI_ROOT}/tests/geometry/industrial_nurbs_models_3d_test.cpp")
target_link_libraries(industrial_nurbs_models_3d_test PRIVATE industrial_nurbs_geometry)
enable_testing()
add_test(NAME industrial_nurbs_models_3d_test COMMAND industrial_nurbs_models_3d_test)
