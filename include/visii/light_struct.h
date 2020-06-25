/* File shared by both host and device */
#pragma once

#define MAX_LIGHTS 100
#include <stdint.h>
#include <glm/glm.hpp>
using namespace glm;

// #ifndef LIGHT_FLAGS
// #define LIGHT_FLAGS
// #define LIGHT_FLAGS_DOUBLE_SIDED (1<<0)
// #define LIGHT_FLAGS_SHOW_END_CAPS (1<<1)
// #define LIGHT_FLAGS_CAST_SHADOWS (1<<2)
// #define LIGHT_FLAGS_USE_VSM (1<<3)
// #define LIGHT_FLAGS_DISABLED (1<<4)
// #define LIGHT_FLAGS_POINT (1<<5)
// #define LIGHT_FLAGS_SPHERE (1<<6)
// #define LIGHT_FLAGS_DISK (1<<7)
// #define LIGHT_FLAGS_ROD (1<<8)
// #define LIGHT_FLAGS_PLANE (1<<9)
// #endif

struct LightStruct {
    float r = 0.f;
    float g = 0.f;
    float b = 0.f;
    float intensity = 0.f;
    int32_t color_texture_id = -1;
    uint32_t flags = 0;
};
