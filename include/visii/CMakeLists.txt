add_subdirectory(utilities)

set(HDR 
    ${Utilities_HDR}
    ${CMAKE_CURRENT_SOURCE_DIR}/entity_struct.h
    ${CMAKE_CURRENT_SOURCE_DIR}/entity.h
    PARENT_SCOPE) 