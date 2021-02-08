set (
    SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/camera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/entity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/light.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/texture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/transform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/volume.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nvisii.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nvisii.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/nvisii_import_scene.cpp
    PARENT_SCOPE
)

set (
    SRC_CU
    ${CMAKE_CURRENT_SOURCE_DIR}/devicecode/path_tracer.cu
    PARENT_SCOPE
)