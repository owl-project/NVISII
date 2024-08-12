#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <owl/owl.h>
#include <owl/owl_device_buffer.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/norm.hpp>

#include <nvisii/entity_struct.h>
#include <nvisii/transform_struct.h>
#include <nvisii/material_struct.h>
#include <nvisii/camera_struct.h>
#include <nvisii/mesh_struct.h>
#include <nvisii/light_struct.h>
#include <nvisii/texture_struct.h>
#include <nvisii/volume_struct.h>

#include "./buffer.h"

struct LaunchParams {
    Buffer<float> assignmentBuffer;

    glm::ivec2 frameSize;
    uint64_t frameID = 0;
    glm::vec4 *frameBuffer;
    uchar4 *albedoBuffer;
    glm::vec4 *normalBuffer;
    glm::vec4 *scratchBuffer;
    glm::vec4 *mvecBuffer;
    glm::vec4 *accumPtr;
    OptixTraversableHandle IAS;
    float domeLightIntensity = 1.f;
    float domeLightExposure = 0.f;
    glm::vec3 domeLightColor = glm::vec3(-1.f);
    float directClamp = 100.f; 
    float indirectClamp = 100.f; 
    uint32_t maxDiffuseDepth = 2;
    uint32_t maxGlossyDepth = 2;
    uint32_t maxTransparencyDepth = 8;
    uint32_t maxTransmissionDepth = 12;
    uint32_t maxVolumeDepth = 2;
    uint32_t numLightSamples = 1;
    uint32_t seed = 0;
    vec2 xPixelSamplingInterval = vec2(0.f,1.f);
    vec2 yPixelSamplingInterval = vec2(0.f,1.f);
    vec2 timeSamplingInterval = vec2(0.f,1.f);
    mat4 proj;
    mat4 viewT0;
    mat4 viewT1;

    EntityStruct    cameraEntity;
    Buffer<EntityStruct> entities;
    Buffer<TransformStruct> transforms;
    Buffer<MaterialStruct> materials;
    Buffer<CameraStruct> cameras;
    Buffer<MeshStruct> meshes;
    Buffer<LightStruct> lights;
    Buffer<TextureStruct> textures;
    Buffer<VolumeStruct> volumes;
    Buffer<uint32_t> lightEntities;
    Buffer<uint32_t> instanceToEntity;
    uint32_t         numLightEntities = 0;

    Buffer<Buffer<float3>> vertexLists;
    Buffer<Buffer<float4>> normalLists;
    Buffer<Buffer<float4>> tangentLists;
    Buffer<Buffer<float2>> texCoordLists;
    Buffer<Buffer<int3>> indexLists;

    int32_t environmentMapID = -1;
    glm::quat environmentMapRotation = glm::quat(1,0,0,0);
    float* environmentMapRows = nullptr;
    float* environmentMapCols = nullptr;
    int environmentMapWidth = 0;
    int environmentMapHeight = 0;
    cudaTextureObject_t proceduralSkyTexture = 0;
    Buffer<cudaTextureObject_t> textureObjects; //cudaTextureObject_t
    Buffer<Buffer<uint8_t>> volumeHandles; //nanovdb::GridHandle<>

    cudaTextureObject_t GGX_E_AVG_LOOKUP;
    cudaTextureObject_t GGX_E_LOOKUP;

    // Used to extract metadata from the renderer.
    uint32_t renderDataMode = 0;
    uint32_t renderDataBounce = 0;

    glm::vec3 sceneBBMin = glm::vec3(0.f);
    glm::vec3 sceneBBMax = glm::vec3(0.f);

    bool enableDomeSampling = true;
};

enum RenderDataFlags : uint32_t { 
  NONE = 0, 
  DEPTH = 1, 
  POSITION = 2,
  NORMAL = 3,
  ENTITY_ID = 4,
  SCREEN_SPACE_NORMAL = 5,
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
  TEXTURE_COORDINATES = 20,
  DEVICE_ID = 21,
  TANGENT = 22
};

#define MAX_LIGHT_SAMPLES 10

// #define REPROJECT true
