# A single source inventory for the core and the standalone industrial gallery.
set(KFBIM_INDUSTRIAL_MODEL_SOURCES_3D
    ${CMAKE_CURRENT_LIST_DIR}/industrial/models_3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/industrial/sleeve_3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/industrial/bracket_3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/industrial/flange_3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/industrial/impeller_3d.cpp)

set(KFBIM_GEOMETRY_MODEL_SOURCES_3D
    ${CMAKE_CURRENT_LIST_DIR}/catalog_3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native_surface_3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rigid_transform_3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/construction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/torus_3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/hollow_cylinder_3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/l_prism_3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native/u_prism_3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bicubic_nurbs_models_3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bicubic_nurbs/construction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bicubic_nurbs/box_3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bicubic_nurbs/solid_cylinder_3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bicubic_nurbs/l_prism_3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bicubic_nurbs/u_prism_3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/analytic/cap_surface_geometry_3d.cpp
    ${KFBIM_INDUSTRIAL_MODEL_SOURCES_3D})
