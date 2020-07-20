/* File shared by both host and device */
#pragma once

#define MAX_TEXTURES 32768

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#define GLM_FORCE_RIGHT_HANDED
#include <stdint.h>
#include <glm/glm.hpp>
using namespace glm;

struct TextureStruct
{
    int32_t width;
    int32_t height;
};
