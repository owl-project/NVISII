/* File shared by both host and device */
#pragma once

#include <stdint.h>
#include <glm/glm.hpp>
using namespace glm;

struct LightStruct {
    float r = 0.f;
    float g = 0.f;
    float b = 0.f;
    float intensity = 1.f;
    float exposure = 0.f;
    float falloff = 2.f;
    int32_t color_texture_id = -1;
    bool use_surface_area = false;
};
