add_library(zfft STATIC
    src/zfft.cc
    src/zfft_ext.cc
)

target_include_directories(zfft PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Optional OpenMP support
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND AND NOT MSVC)
    target_link_libraries(zfft PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_definitions(zfft PRIVATE _USE_MATH_DEFINES)
endif()

target_compile_features(zfft PUBLIC cxx_std_17)

# zFFT uses the conventional M_PI extension throughout its imported sources.
# GNU dialects expose it on GCC/Clang; MSVC uses _USE_MATH_DEFINES from the
# top-level build.
set_target_properties(zfft PROPERTIES CXX_EXTENSIONS ON)

install(TARGETS zfft
    EXPORT SymmetryKFBITargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
