/* File shared by both host and device */
#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#define GLM_FORCE_RIGHT_HANDED
#include <stdint.h>
#include <glm/glm.hpp>
using namespace glm;

struct TextureStruct
{
    int32_t width = -1;
    int32_t height = -1;
    vec2 scale = vec2(1.f, 1.f); 
};
