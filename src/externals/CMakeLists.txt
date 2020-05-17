set(
    Externals_HDR
    ${CMAKE_CURRENT_SOURCE_DIR}/glm_bindings/constants.i
    ${CMAKE_CURRENT_SOURCE_DIR}/glm_bindings/functions.i
    ${CMAKE_CURRENT_SOURCE_DIR}/glm_bindings/glm.i
    ${CMAKE_CURRENT_SOURCE_DIR}/glm_bindings/ivec2.i
    ${CMAKE_CURRENT_SOURCE_DIR}/glm_bindings/ivec3.i
    ${CMAKE_CURRENT_SOURCE_DIR}/glm_bindings/ivec4.i
    ${CMAKE_CURRENT_SOURCE_DIR}/glm_bindings/u16vec2.i
    ${CMAKE_CURRENT_SOURCE_DIR}/glm_bindings/u16vec3.i
    ${CMAKE_CURRENT_SOURCE_DIR}/glm_bindings/u16vec4.i
    ${CMAKE_CURRENT_SOURCE_DIR}/glm_bindings/vec2.i
    ${CMAKE_CURRENT_SOURCE_DIR}/glm_bindings/vec3.i
    ${CMAKE_CURRENT_SOURCE_DIR}/glm_bindings/vec4.i
    ${CMAKE_CURRENT_SOURCE_DIR}/glfw_implementation/glfw.h
    PARENT_SCOPE
)

set (
    Externals_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/glfw_implementation/glfw.cpp
    PARENT_SCOPE
)