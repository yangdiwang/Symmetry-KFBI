add_library(kfbim_core STATIC
    # Layer 0 - Grid, interface, and geometry
    grid/cartesian_grid_2d.cpp
    interface/interface_2d.cpp
    geometry/grid_pair_2d.cpp
    geometry/corner_patch_2d.cpp
    geometry/p2_projection_2d.cpp
    geometry/curve_resampler_2d.cpp
    geometry/nurbs_boundary_2d.cpp

    # Layer 1 - Transfer operators and local Cauchy reconstruction
    transfer/laplace_arc_length_bspline_crossing_jet_2d.cpp
    transfer/nurbs_density_space_2d.cpp
    transfer/laplace_nurbs_density_trace_state_2d.cpp
    transfer/laplace_spread_2d.cpp
    transfer/laplace_restrict_2d.cpp

    # Layer 2 - Bundled zFFT bulk solver
    bulk_solvers/laplace_zfft_bulk_solver_2d.cpp

    # Layer 3 - Potential pipeline and problem wrappers
    potentials/laplace_potential.cpp
    operators/laplace_bvp_2d.cpp
    operators/laplace_neumann_exterior_trace_2d.cpp
    operators/laplace_nsc_et_bvp_2d.cpp
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
        geometry/nurbs_bezier_segment_closest_point_3d.cpp
        geometry/nurbs_bezier_segment_stationary_points_3d.cpp
        geometry/nurbs_cartesian_domain_3d.cpp
        geometry/nurbs_surface_intersector_3d.cpp
        geometry/native_nurbs_exact_geometry_3d.cpp
        geometry/native_endpoint_path_3d.cpp
        geometry/nurbs_bezier_extraction_3d.cpp
        geometry/nurbs_patch_polar_evaluator_3d.cpp
        geometry/nurbs_surface_closest_point_3d.cpp
        geometry/rational_bezier_element_3d.cpp
        geometry/rational_bezier_subdivision_3d.cpp
        geometry/nurbs_surface_model_3d.cpp
        geometry/industrial_nurbs_models_3d.cpp
        geometry/industrial_nurbs_sleeve_3d.cpp
        geometry/industrial_nurbs_bracket_3d.cpp
        geometry/industrial_nurbs_flange_3d.cpp
        geometry/industrial_nurbs_impeller_3d.cpp
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

    option(KFBIM_NATIVE_ENDPOINT_MPFR
        "Enable directed-rounding MPFR refinement for native endpoint certificates" ON)
    if(KFBIM_NATIVE_ENDPOINT_MPFR)
        if(MPFR_INCLUDE_DIR AND MPFR_LIBRARIES)
            target_include_directories(kfbim_core PRIVATE ${MPFR_INCLUDE_DIR})
            target_link_libraries(kfbim_core PRIVATE ${MPFR_LIBRARIES})
            set_source_files_properties(geometry/native_endpoint_path_3d.cpp
                PROPERTIES COMPILE_DEFINITIONS KFBIM_NATIVE_ENDPOINT_HAS_MPFR=1)
        else()
            message(WARNING
                "MPFR was not found: native endpoint refinement will fail closed "
                "when binary64 cannot certify a query")
        endif()
    endif()

    # A certificate must not be compiled with transformations that discard
    # the directed/outward rounding used by its arithmetic backend.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(
            geometry/native_nurbs_exact_geometry_3d.cpp
            geometry/native_endpoint_path_3d.cpp
            geometry/nurbs_surface_closest_point_3d.cpp
            PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    elseif(MSVC)
        set_source_files_properties(
            geometry/native_nurbs_exact_geometry_3d.cpp
            geometry/native_endpoint_path_3d.cpp
            geometry/nurbs_surface_closest_point_3d.cpp
            PROPERTIES COMPILE_OPTIONS "/fp:strict")
    endif()
endif()

if(KFBIM_BUILD_3D)
    target_sources(kfbim_core PRIVATE geometry/nurbs_grid_line_cache_3d.cpp)
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
    PATTERN "support" EXCLUDE
)
