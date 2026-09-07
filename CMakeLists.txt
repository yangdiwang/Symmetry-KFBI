cmake_minimum_required(VERSION 3.20)
project(SymmetryKFBI VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MSVC and MinGW both hide the non-standard M_PI constant in strict language
# mode unless this compatibility definition is present.  zFFT uses M_PI.
add_compile_definitions(_USE_MATH_DEFINES)

option(KFBIM_BUILD_APPS "Build standalone solver examples" ON)
option(KFBIM_BUILD_BENCHMARKS "Build performance and convergence benchmarks" ${KFBIM_BUILD_APPS})
option(BUILD_TESTING "Build and register regression tests" ${KFBIM_BUILD_APPS})
option(KFBIM_BUILD_EXPERIMENTAL_3D
    "Build supplementary remote 3D algorithms, studies, and benchmarks" OFF)
option(KFBIM_BUILD_REMOTE_SOLVER_ROUTE_TEST
    "Build the legacy remote solver route test (requires an API adapter)" OFF)
if(KFBIM_BUILD_REMOTE_SOLVER_ROUTE_TEST AND NOT KFBIM_BUILD_EXPERIMENTAL_3D)
    message(FATAL_ERROR
        "KFBIM_BUILD_REMOTE_SOLVER_ROUTE_TEST requires KFBIM_BUILD_EXPERIMENTAL_3D=ON")
endif()

# Eigen is required by all targets. Prefer an installed package and fetch the
# fixed 3.4.0 release when it is not available.
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 3.4 was not found locally; fetching Eigen 3.4.0")
    include(FetchContent)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    set(_kfbim_build_testing ${BUILD_TESTING})
    set(BUILD_TESTING OFF)
    FetchContent_MakeAvailable(Eigen3)
    set(BUILD_TESTING ${_kfbim_build_testing})
    unset(_kfbim_build_testing)
endif()

# The 3D interface/crossing pipeline uses CGAL. Keep it enabled by default so
# a normal build really produces every advertised app; 2D-only users may opt
# out explicitly instead of silently losing the 3D executable.
option(KFBIM_BUILD_3D "Build the CGAL-dependent 3D core and apps" ON)
if(KFBIM_BUILD_3D)
    find_package(CGAL 5.6 REQUIRED)
endif()

add_subdirectory(third_party/zfft)
add_subdirectory(src)

if(KFBIM_BUILD_APPS OR KFBIM_BUILD_BENCHMARKS OR BUILD_TESTING)
    add_subdirectory(src/support)
endif()

if(KFBIM_BUILD_APPS)
    add_subdirectory(apps)
endif()
if(KFBIM_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

install(EXPORT SymmetryKFBITargets
    NAMESPACE SymmetryKFBI::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SymmetryKFBI
)
