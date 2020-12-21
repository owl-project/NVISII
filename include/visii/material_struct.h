/* File shared by both host and device */
#pragma once

#define NUM_MAT_PARAMS 19
#include <stdint.h>
#include <glm/glm.hpp>
using namespace glm;

/* Follows the disney BSDF */
struct MaterialStruct {    
    int32_t transmission_roughness_texture_id = -1; // 112
    int32_t base_color_texture_id = -1; // 120
    int32_t roughness_texture_id = -1; // 124
    int32_t alpha_texture_id = -1; // 132
    int32_t normal_map_texture_id = -1; // 136
    int32_t subsurface_color_texture_id = -1; // 140
    int32_t subsurface_radius_texture_id = -1; // 144
    int32_t subsurface_texture_id = -1; // 148
    int32_t metallic_texture_id = -1; // 152
    int32_t specular_texture_id = -1; // 156
    int32_t specular_tint_texture_id = -1; // 160
    int32_t anisotropic_texture_id = -1; // 164
    int32_t anisotropic_rotation_texture_id = -1; // 168
    int32_t sheen_texture_id = -1; // 172
    int32_t sheen_tint_texture_id = -1; // 176
    int32_t clearcoat_texture_id = -1; // 180
    int32_t clearcoat_roughness_texture_id = -1; // 184
    int32_t ior_texture_id = -1; // 188
    int32_t transmission_texture_id = -1; // 192

    int8_t transmission_roughness_texture_channel = 0; // 112
    int8_t roughness_texture_channel = 0; // 124
    int8_t alpha_texture_channel = 0; // 132
    int8_t normal_map_texture_channel = 0; // 136
    int8_t subsurface_texture_channel = 0; // 148
    int8_t metallic_texture_channel = 01; // 152
    int8_t specular_texture_channel = 0; // 156
    int8_t specular_tint_texture_channel = 0; // 160
    int8_t anisotropic_texture_channel = 0; // 164
    int8_t anisotropic_rotation_texture_channel = 0; // 168
    int8_t sheen_texture_channel = 0; // 172
    int8_t sheen_tint_texture_channel = 0; // 176
    int8_t clearcoat_texture_channel = 0; // 180
    int8_t clearcoat_roughness_texture_channel = 0; // 184
    int8_t ior_texture_channel = 0; // 188
    int8_t transmission_texture_channel = 0; // 192
};
