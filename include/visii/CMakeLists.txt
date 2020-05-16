add_subdirectory(utilities)

set(HDR 
    ${Utilities_HDR}
    ${CMAKE_CURRENT_SOURCE_DIR}/entity_struct.h
    ${CMAKE_CURRENT_SOURCE_DIR}/entity.h
    ${CMAKE_CURRENT_SOURCE_DIR}/material_struct.h
    ${CMAKE_CURRENT_SOURCE_DIR}/material.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_struct.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/transform_struct.h
    ${CMAKE_CURRENT_SOURCE_DIR}/transform.h
    ${CMAKE_CURRENT_SOURCE_DIR}/visii.h
    PARENT_SCOPE) 