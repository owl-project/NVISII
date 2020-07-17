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
#include <visii/texture_struct.h>

struct LaunchParams {
    glm::ivec2 frameSize;
    uint64_t frameID = 0;
    glm::vec4 *frameBuffer;
    glm::vec4 *albedoBuffer;
    glm::vec4 *normalBuffer;
    glm::vec4 *accumPtr;
    OptixTraversableHandle world;
    float domeLightIntensity = 1.f;
    float directClamp = 100.f; 
    float indirectClamp = 100.f; 

    EntityStruct    cameraEntity;
    EntityStruct    *entities = nullptr;
    TransformStruct *transforms = nullptr;
    MaterialStruct  *materials = nullptr;
    CameraStruct    *cameras = nullptr;
    MeshStruct      *meshes = nullptr;
    LightStruct     *lights = nullptr;
    TextureStruct   *textures = nullptr;
    uint32_t        *lightEntities = nullptr;
    uint32_t        *instanceToEntityMap = nullptr;
    uint32_t         numLightEntities = 0;

    vec4 **vertexLists = nullptr;
    vec4 **normalLists = nullptr;
    vec2 **texCoordLists = nullptr;
    ivec3 **indexLists = nullptr;

    int32_t environmentMapID = -1;
    cudaTextureObject_t *textureObjects = nullptr;

    cudaTextureObject_t GGX_E_AVG_LOOKUP;
    cudaTextureObject_t GGX_E_LOOKUP;

    // Used to extract metadata from the renderer.
    uint32_t renderDataMode = 0;
    uint32_t renderDataBounce = 0;
};

enum RenderDataFlags : uint32_t { 
  NONE = 0, 
  DEPTH = 1, 
  POSITION = 2,
  NORMAL = 3,
  ENTITY_ID = 4,
  DENOISE_NORMAL = 5,
  DENOISE_ALBEDO = 6,
};