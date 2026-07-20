cmake_minimum_required(VERSION 3.20)
project(SymmetryKFBI VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MSVC and MinGW both hide the non-standard M_PI constant in strict language
# mode unless this compatibility definition is present.  zFFT uses M_PI.
add_compile_definitions(_USE_MATH_DEFINES)

# Eigen is the only external dependency.  Prefer an installed package and
# fetch the fixed 3.4.0 release when it is not available.
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
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen3)
endif()

add_subdirectory(third_party/zfft)
add_subdirectory(src)

option(KFBIM_BUILD_APPS "Build all standalone examples" ON)
if(KFBIM_BUILD_APPS)
    add_subdirectory(apps)
endif()

install(EXPORT SymmetryKFBITargets
    NAMESPACE SymmetryKFBI::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SymmetryKFBI
)
