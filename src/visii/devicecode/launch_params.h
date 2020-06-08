#pragma once

#include <owl/owl.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/norm.hpp>

#include <visii/entity_struct.h>
#include <visii/transform_struct.h>
#include <visii/material_struct.h>
#include <visii/camera_struct.h>
#include <visii/mesh_struct.h>
#include <visii/light_struct.h>

struct LaunchParams {
    glm::ivec2 frameSize;
    uint64_t frameID = 0;
    glm::vec4 *frameBuffer;
    glm::vec4 *albedoBuffer;
    glm::vec4 *normalBuffer;
    glm::vec4 *accumPtr;
    OptixTraversableHandle world;
    float domeLightIntensity = 1.f;

    EntityStruct    cameraEntity;
    EntityStruct    *entities = nullptr;
    TransformStruct *transforms = nullptr;
    MaterialStruct  *materials = nullptr;
    CameraStruct    *cameras = nullptr;
    MeshStruct      *meshes = nullptr;
    LightStruct     *lights = nullptr;
    uint32_t        *lightEntities = nullptr;
    uint32_t        *instanceToEntityMap = nullptr;
    uint32_t         numLightEntities = 0;

    vec4 **vertexLists = nullptr;
    ivec3 **indexLists = nullptr;

    bool environmentMapSet = false;
    cudaTextureObject_t environmentMap;

    cudaTextureObject_t GGX_E_AVG_LOOKUP;
    cudaTextureObject_t GGX_E_LOOKUP;
};
