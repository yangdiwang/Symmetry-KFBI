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

if(KFBIM_BUILD_3D)
    target_sources(kfbim_core PRIVATE
        grid/cartesian_grid_3d.cpp
        grid/mac_grid_3d.cpp
        interface/interface_3d.cpp
        geometry/grid_pair_3d.cpp
        geometry/edge_patch_3d.cpp
        geometry/p2_projection_3d.cpp
        geometry/nurbs_patch_triangulator_3d.cpp
        geometry/nurbs_bezier_intersection_3d.cpp
        geometry/nurbs_cartesian_domain_3d.cpp
        geometry/nurbs_surface_intersector_3d.cpp
        geometry/nurbs_bezier_extraction_3d.cpp
        geometry/rational_bezier_element_3d.cpp
        geometry/rational_bezier_subdivision_3d.cpp
        geometry/nurbs_surface_model_3d.cpp
        local_cauchy/edge_patch_laplacian_3d.cpp
        local_cauchy/edge_singular_bspline_fit_3d.cpp
        local_cauchy/edge_singular_fit_3d.cpp
        transfer/laplace_spread_3d.cpp
        transfer/laplace_restrict_3d.cpp
        bulk_solvers/laplace_zfft_bulk_solver_3d.cpp
        operators/laplace_bvp_3d.cpp
        operators/laplace_transmission_3d.cpp
    )
    target_link_libraries(kfbim_core PRIVATE CGAL::CGAL)
    target_compile_definitions(kfbim_core PUBLIC KFBIM_HAS_CGAL)
endif()

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
