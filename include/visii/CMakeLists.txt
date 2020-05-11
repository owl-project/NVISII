add_subdirectory(utilities)

set(HDR 
    ${Utilities_HDR}
    ${CMAKE_CURRENT_SOURCE_DIR}/entity_struct.h
    ${CMAKE_CURRENT_SOURCE_DIR}/entity.h
    ${CMAKE_CURRENT_SOURCE_DIR}/transform_struct.h
    ${CMAKE_CURRENT_SOURCE_DIR}/transform.h
    PARENT_SCOPE) 