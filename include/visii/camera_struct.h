/* File shared by both host and device */
#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#define GLM_FORCE_RIGHT_HANDED
#include <glm/glm.hpp>
using namespace glm;

/* This could be split up to allow for better GPU memory reads */
struct CameraStruct
{
    mat4 view;
    mat4 proj;
    mat4 viewinv;
    mat4 projinv;
    mat4 viewproj;
    // float far_pos;
    float fov;
    float focalDistance;
    float apertureDiameter;
    int tex_id;
};
