/* File shared by both host and device */
#pragma once

#define MAX_MATERIALS 100000
#include <stdint.h>
#include <glm/glm.hpp>
using namespace glm;

/* Follows the disney BSDF */
struct MaterialStruct {
    vec4 base_color; // 16 // Note: also contains alpha
    vec4 subsurface_radius; // 32
    vec4 subsurface_color; //48
    
    float subsurface; // 52
    float metallic; // 56
    float specular; // 60
    float specular_tint; // 64
    float roughness; // 68
    float anisotropic; // 72
    float anisotropic_rotation; // 76
    float sheen; // 80
    float sheen_tint; // 84
    float clearcoat; // 88
    float clearcoat_roughness; // 92
    float ior; // 96
    float transmission; // 100
    float transmission_roughness; // 104
    
    int32_t flags; // 116
    int32_t volume_texture_id; // 108
    int32_t transmission_roughness_texture_id; // 112
    int32_t base_color_texture_id; // 120
    int32_t roughness_texture_id; // 124
    int32_t occlusion_texture_id; // 128


    /* Addresses for texture mapped parameters */
    int32_t alpha_texture_id; // 132
    int32_t bump_texture_id; // 136
    int32_t subsurface_color_texture_id; // 140
    int32_t subsurface_radius_texture_id; // 144
    int32_t subsurface_texture_id; // 148
    int32_t metallic_texture_id; // 152
    int32_t specular_texture_id; // 156
    int32_t specular_tint_texture_id; // 160

    int32_t anisotropic_texture_id; // 164
    int32_t anisotropic_rotation_texture_id; // 168
    int32_t sheen_texture_id; // 172
    int32_t sheen_tint_texture_id; // 176
    int32_t clearcoat_texture_id; // 180
    int32_t clearcoat_roughness_texture_id; // 184
    int32_t ior_texture_id; // 188
    int32_t transmission_texture_id; // 192
    // int32_t ph8_id; // 196
    // int32_t ph8_id; // 200
    // int32_t ph8_id; // 184
    // int32_t ph8_id; // 184
};
