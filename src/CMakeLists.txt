add_subdirectory(externals)
add_subdirectory(visii)

set(Externals_HDR ${Externals_HDR} PARENT_SCOPE)
set(SRC ${SRC} ${Externals_SRC} PARENT_SCOPE)
set(SRC_CU ${SRC_CU} PARENT_SCOPE)
