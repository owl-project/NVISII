/* File shared by both host and device */
#pragma once

#define MAX_MATERIALS 100000
#include <stdint.h>
#include <glm/glm.hpp>
using namespace glm;

/* Follows the disney BSDF */
struct MaterialStruct {
    vec4 base_color = vec4(0.f); // 16 // Note: also contains alpha
    vec4 subsurface_radius = vec4(0.f); // 32
    vec4 subsurface_color = vec4(0.f); //48
    
    float subsurface = 0.f; // 52
    float metallic = 0.f; // 56
    float specular = 0.f; // 60
    float specular_tint = 0.f; // 64
    float roughness = 0.f; // 68
    float anisotropic = 0.f; // 72
    float anisotropic_rotation = 0.f; // 76
    float sheen = 0.f; // 80
    float sheen_tint = 0.f; // 84
    float clearcoat = 0.f; // 88
    float clearcoat_roughness = 0.f; // 92
    float ior = 0.f; // 96
    float transmission = 0.f; // 100
    float transmission_roughness = 0.f; // 104
    
    int16_t transmission_roughness_texture_id = -1; // 112
    int16_t transmission_roughness_texture_channel = 0; // 112
    int16_t base_color_texture_id = -1; // 120
    // int16_t base_color_texture_channel = 0; // 120
    int16_t roughness_texture_id = -1; // 124
    int16_t roughness_texture_channel = 0; // 124
    int16_t alpha_texture_id = -1; // 132
    int16_t alpha_texture_channel = 0; // 132
    int16_t normal_map_texture_id = -1; // 136
    int16_t normal_map_texture_channel = 0; // 136
    int16_t subsurface_color_texture_id = -1; // 140
    // int16_t subsurface_color_texture_channel = -1; // 140
    int16_t subsurface_radius_texture_id = -1; // 144
    // int16_t subsurface_radius_texture_channel = -1; // 144
    int16_t subsurface_texture_id = -1; // 148
    int16_t subsurface_texture_channel = 0; // 148
    int16_t metallic_texture_id = -1; // 152
    int16_t metallic_texture_channel = 01; // 152
    int16_t specular_texture_id = -1; // 156
    int16_t specular_texture_channel = 0; // 156
    int16_t specular_tint_texture_id = -1; // 160
    int16_t specular_tint_texture_channel = 0; // 160
    int16_t anisotropic_texture_id = -1; // 164
    int16_t anisotropic_texture_channel = 0; // 164
    int16_t anisotropic_rotation_texture_id = -1; // 168
    int16_t anisotropic_rotation_texture_channel = 0; // 168
    int16_t sheen_texture_id = -1; // 172
    int16_t sheen_texture_channel = 0; // 172
    int16_t sheen_tint_texture_id = -1; // 176
    int16_t sheen_tint_texture_channel = 0; // 176
    int16_t clearcoat_texture_id = -1; // 180
    int16_t clearcoat_texture_channel = 0; // 180
    int16_t clearcoat_roughness_texture_id = -1; // 184
    int16_t clearcoat_roughness_texture_channel = 0; // 184
    int16_t ior_texture_id = -1; // 188
    int16_t ior_texture_channel = 0; // 188
    int16_t transmission_texture_id = -1; // 192
    int16_t transmission_texture_channel = 0; // 192
};
