/* File shared by both host and device */
#pragma once

#define MAX_TRANSFORMS 1000000
#include <glm/glm.hpp>
using namespace glm;

/* This could probably be split up to allow for better GPU memory reads */
struct TransformStruct
{
    /* 64 bytes */
    mat4 worldToLocal;
    mat4 localToWorld;
    // vec3 translation;
    // quat rotation;
    // vec3 scale;
    // mat4 worldToLocalRotation;
    // mat4 localToWorldRotation;
    // mat4 worldToLocalTranslation;
    // mat4 localToWorldTranslation;
    
    /* 128 bytes, for temporal reprojection */
    // mat4 worldToLocalPrev;
    // mat4 localToWorldPrev;
    // mat4 worldToLocalRotationPrev;
    // mat4 worldToLocalTranslationPrev;
};
