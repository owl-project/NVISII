/* File shared by both host and device */
#pragma once

#define MAX_MESHES 100000
#include <glm/glm.hpp>
using namespace glm;

/* Note: the purpose of many of these parameters is for occlusion queries in a potential rasterizer framework */
struct MeshStruct {
    mat4 bb_local_to_parent;

    /* The last computed average of all mesh positions */
    vec4 center; // 16
    
    /* Minimum and maximum bounding box coordinates */
    vec4 bbmin; // 32
    vec4 bbmax; // 48
    
    /* The radius of a sphere centered at the centroid which contains the mesh */
    float bounding_sphere_radius; // 52
    int32_t show_bounding_box; // 56
    int32_t ph2; // 60
    int32_t ph3; // 64
};
