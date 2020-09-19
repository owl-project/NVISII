#pragma once

#include <owl/owl.h>
#include <owl/owl_device_buffer.h>
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
    glm::vec4 *scratchBuffer;
    glm::vec4 *mvecBuffer;
    glm::vec4 *accumPtr;
    OptixTraversableHandle world;
    float domeLightIntensity = 1.f;
    glm::vec3 domeLightColor = glm::vec3(-1.f);
    float directClamp = 100.f; 
    float indirectClamp = 100.f; 
    uint32_t maxSpecularBounceDepth = 8;
    uint32_t maxDiffuseBounceDepth = 2;
    uint32_t numLightSamples = 1;
    uint32_t seed = 0;
    vec2 xPixelSamplingInterval = vec2(0.f,1.f);
    vec2 yPixelSamplingInterval = vec2(0.f,1.f);
    vec2 timeSamplingInterval = vec2(0.f,1.f);
    mat4 proj;
    mat4 viewT0;
    mat4 viewT1;

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

    owl::device::Buffer vertexLists;
    owl::device::Buffer normalLists;
    owl::device::Buffer texCoordLists;
    owl::device::Buffer indexLists;

    int32_t environmentMapID = -1;
    glm::quat environmentMapRotation = glm::quat(1,0,0,0);
    float* environmentMapRows = nullptr;
    float* environmentMapCols = nullptr;
    int environmentMapWidth = 0;
    int environmentMapHeight = 0;
    cudaTextureObject_t proceduralSkyTexture = 0;
    cudaTextureObject_t *textureObjects = nullptr;

    cudaTextureObject_t GGX_E_AVG_LOOKUP;
    cudaTextureObject_t GGX_E_LOOKUP;

    // Used to extract metadata from the renderer.
    uint32_t renderDataMode = 0;
    uint32_t renderDataBounce = 0;

    glm::vec3 sceneBBMin = glm::vec3(0.f);
    glm::vec3 sceneBBMax = glm::vec3(0.f);
};

enum RenderDataFlags : uint32_t { 
  NONE = 0, 
  DEPTH = 1, 
  POSITION = 2,
  NORMAL = 3,
  ENTITY_ID = 4,
  DENOISE_NORMAL = 5,
  DENOISE_ALBEDO = 6,
  DIFFUSE_MOTION_VECTORS = 7,
  BASE_COLOR = 8,
  DIFFUSE_COLOR = 9,
  DIFFUSE_DIRECT_LIGHTING = 10,
  DIFFUSE_INDIRECT_LIGHTING = 11,
  GLOSSY_COLOR = 12,
  GLOSSY_DIRECT_LIGHTING = 13,
  GLOSSY_INDIRECT_LIGHTING = 14,
  TRANSMISSION_COLOR = 15,
  TRANSMISSION_DIRECT_LIGHTING = 16,
  TRANSMISSION_INDIRECT_LIGHTING = 17,
  RAY_DIRECTION = 18,
  HEATMAP = 19,
};

#define MAX_LIGHT_SAMPLES 10

// #define REPROJECT true
