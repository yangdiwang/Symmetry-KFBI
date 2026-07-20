add_library(kfbim_core STATIC
    # Layer 0 - Grid, interface, and geometry
    grid/cartesian_grid_2d.cpp
    interface/interface_2d.cpp
    geometry/grid_pair_2d.cpp
    geometry/corner_patch_2d.cpp
    geometry/p2_projection_2d.cpp
    geometry/curve_resampler_2d.cpp

    # Layer 1 - Transfer operators and local Cauchy reconstruction
    transfer/laplace_spread_2d.cpp
    transfer/laplace_restrict_2d.cpp

    # Layer 2 - Bundled zFFT bulk solver
    bulk_solvers/laplace_zfft_bulk_solver_2d.cpp

    # Layer 3 - Potential pipeline and problem wrappers
    potentials/laplace_potential.cpp
    operators/laplace_bvp_2d.cpp
    operators/laplace_neumann_exterior_trace_2d.cpp
    operators/laplace_transmission_2d.cpp

    # Layer 4 - Outer solver
    gmres/gmres.cpp
)

target_include_directories(kfbim_core PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(kfbim_core PUBLIC Eigen3::Eigen zfft)
target_compile_features(kfbim_core PUBLIC cxx_std_17)

if(MINGW)
    target_compile_options(kfbim_core PRIVATE -Wa,-mbig-obj)
endif()

install(TARGETS kfbim_core
    EXPORT SymmetryKFBITargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/kfbim
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/src
    FILES_MATCHING PATTERN "*.hpp"
)
