#define GLM_FORCE_CUDA
#define PMEVENT( x ) asm volatile("pmevent " #x ";")

#include <stdint.h>
#include "launch_params.h"
#include "types.h"
#include "path_tracer.h"
#include "disney_bsdf.h"
#include "lights.h"
#include "math.h"
#include <optix_device.h>
#include <owl/common/math/random.h>
#include <owl/common/math/box.h>

#include "visii/utilities/procedural_sky.h"

#include <glm/gtx/matrix_interpolation.hpp>

typedef owl::common::LCG<4> Random;

extern "C" __constant__ LaunchParams optixLaunchParams;

struct RayPayload {
    int instanceID = -1;
    int primitiveID = -1;
    float2 barycentrics;
    float tHit = -1.f;
    float localToWorld[12];
    float localToWorldT0[12];
    float localToWorldT1[12];
};

__device__
vec2 toUV(vec3 n)
{
    n.z = -n.z;
    n.x = -n.x;
    vec2 uv;

    uv.x = approx_atan2f(float(-n.x), float(n.y));
    uv.x = (uv.x + M_PI / 2.0f) / (M_PI * 2.0f) + M_PI * (28.670f / 360.0f);

    uv.y = ::clamp(float(acosf(n.z) / M_PI), .001f, .999f);

    return uv;
}

// Uv range: [0, 1]
__device__
vec3 toPolar(vec2 uv)
{
    float theta = 2.0 * M_PI * uv.x + - M_PI / 2.0;
    float phi = M_PI * uv.y;

    vec3 n;
    n.x = __cosf(theta) * __sinf(phi);
    n.y = __sinf(theta) * __sinf(phi);
    n.z = __cosf(phi);

    n.z = -n.z;
    n.x = -n.x;
    return n;
}

__device__
cudaTextureObject_t getEnvironmentTexture()
{
    auto &LP = optixLaunchParams;
    cudaTextureObject_t tex = 0;
    if (LP.environmentMapID >= 0) {
        return LP.textureObjects.get(LP.environmentMapID, __LINE__);
    } else if ((LP.environmentMapID == -2) && (LP.proceduralSkyTexture != 0)) {
        return LP.proceduralSkyTexture;
    }
    return tex;    
}

inline __device__
float3 missColor(const float3 n_dir, cudaTextureObject_t &tex)
{
    auto &LP = optixLaunchParams;
    vec3 rayDir = LP.environmentMapRotation * make_vec3(n_dir);
    if (tex)
    {
        vec2 tc = toUV(vec3(rayDir.x, rayDir.y, rayDir.z));
        float4 texColor = tex2D<float4>(tex, tc.x,tc.y);
        return make_float3(texColor);
    }
    
    if (glm::any(glm::greaterThanEqual(LP.domeLightColor, glm::vec3(0.f)))) return make_float3(LP.domeLightColor);
    float t = 0.5f*(rayDir.z + 1.0f);
    float3 c = (1.0f - t) * make_float3(pow(vec3(1.0f), vec3(2.2f))) + t * make_float3( pow(vec3(0.5f, 0.7f, 1.0f), vec3(2.2f)) );
    return c;
}

inline __device__
float3 missColor(const owl::Ray &ray, cudaTextureObject_t &tex)
{
    return missColor(ray.direction, tex);
}


OPTIX_MISS_PROGRAM(miss)()
{
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
    auto &LP = optixLaunchParams;
    RayPayload &prd = owl::getPRD<RayPayload>();
    prd.instanceID = optixGetInstanceIndex();
    prd.tHit = optixGetRayTmax();
    prd.barycentrics = optixGetTriangleBarycentrics();
    prd.primitiveID = optixGetPrimitiveIndex();

    // const OptixTraversableHandle handle = optixGetTransformListHandle(prd.instanceID);
    // const OptixTransformType type = optixGetTransformTypeFromHandle( handle );
    // if (type == OPTIX_TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM) {
    //     const OptixMatrixMotionTransform* transformData = optixGetMatrixMotionTransformFromHandle( handle );
    //     memcpy(prd.localToWorld, &transformData->transform[0][0], 12 * sizeof(float));
    // }
    // else if (type == OPTIX_TRANSFORM_TYPE_SRT_MOTION_TRANSFORM) {
    //     const OptixSRTMotionTransform* transformData = optixGetMatrixMotionTransformFromHandle( handle );
    //     memcpy(prd.localToWorld, &transformData->transform[0][0], 12 * sizeof(float));
    // }
    // optixGetInterpolatedTransformation( trf0, trf1, trf2, transformData, time );
    // const float4* transform = (const float4*)( &transformData->transform[key][0] );
    // const float4* transform = (const float4*)( &transformData->transform[key][0] );

    // This seems to cause most of the stalls in san miguel scene.
    optixGetObjectToWorldTransformMatrix(prd.localToWorld);
    // const int entityID = LP.surfaceInstanceToEntity.get(prd.instanceID, __LINE__);
    // EntityStruct entity = LP.entities.get(entityID, __LINE__);
    // TransformStruct transform = LP.transforms.get(entity.transform_id, __LINE__);
    // prd.localToWorld[0]  = glm::row(transform.localToWorld, 0).x;
    // prd.localToWorld[1]  = glm::row(transform.localToWorld, 0).y;
    // prd.localToWorld[2]  = glm::row(transform.localToWorld, 0).z;
    // prd.localToWorld[3]  = glm::row(transform.localToWorld, 0).w;
    // prd.localToWorld[4]  = glm::row(transform.localToWorld, 1).x;
    // prd.localToWorld[5]  = glm::row(transform.localToWorld, 1).y;
    // prd.localToWorld[6]  = glm::row(transform.localToWorld, 1).z;
    // prd.localToWorld[7]  = glm::row(transform.localToWorld, 1).w;
    // prd.localToWorld[8]  = glm::row(transform.localToWorld, 2).x;
    // prd.localToWorld[9]  = glm::row(transform.localToWorld, 2).y;
    // prd.localToWorld[10] = glm::row(transform.localToWorld, 2).z;
    // prd.localToWorld[11] = glm::row(transform.localToWorld, 2).w;
    // to_optix_tfm(transform.localToWorld, prd.localToWorld);
    
    // If we don't need motion vectors, (or in the future if an object 
    // doesn't have motion blur) then return.
    if (LP.renderDataMode == RenderDataFlags::NONE) return;
   
    OptixTraversableHandle handle = optixGetTransformListHandle(prd.instanceID);
    float4 trf00, trf01, trf02;
    float4 trf10, trf11, trf12;
    
    optix_impl::optixGetInterpolatedTransformationFromHandle( trf00, trf01, trf02, handle, /* time */ 0.f, true );
    optix_impl::optixGetInterpolatedTransformationFromHandle( trf10, trf11, trf12, handle, /* time */ 1.f, true );
    memcpy(&prd.localToWorldT0[0], &trf00, sizeof(trf00));
    memcpy(&prd.localToWorldT0[4], &trf01, sizeof(trf01));
    memcpy(&prd.localToWorldT0[8], &trf02, sizeof(trf02));
    memcpy(&prd.localToWorldT1[0], &trf10, sizeof(trf10));
    memcpy(&prd.localToWorldT1[4], &trf11, sizeof(trf11));
    memcpy(&prd.localToWorldT1[8], &trf12, sizeof(trf12));
}

OPTIX_CLOSEST_HIT_PROGRAM(ShadowRay)()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    prd.instanceID = optixGetInstanceIndex();
    prd.tHit = optixGetRayTmax();
}

OPTIX_CLOSEST_HIT_PROGRAM(VolumeMesh)()
{   
    auto &LP = optixLaunchParams;
    RayPayload &prd = owl::getPRD<RayPayload>();
    prd.instanceID = optixGetInstanceIndex();
    prd.tHit = optixGetRayTmax();
    prd.barycentrics = optixGetTriangleBarycentrics();
    prd.primitiveID = optixGetPrimitiveIndex();
}

OPTIX_CLOSEST_HIT_PROGRAM(VolumeShadowRay)()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    prd.tHit = 1.0f;//optixGetRayTmax();
}

OPTIX_INTERSECT_PROGRAM(VolumeIntersection)()
{
    float tmax      = optixGetRayTmax();
    optixReportIntersection(tmax, 0);
}

OPTIX_BOUNDS_PROGRAM(VolumeBounds)(
    const void  *geomData,
    owl::common::box3f &primBounds,
    const int    primID)
{
    const VolumeGeomData &self = *(const VolumeGeomData*)geomData;
    primBounds = owl::common::box3f();
    primBounds.lower.x = self.bbmin.x;
    primBounds.lower.y = self.bbmin.y;
    primBounds.lower.z = self.bbmin.z;
    primBounds.upper.x = self.bbmax.x;
    primBounds.upper.y = self.bbmax.y;
    primBounds.upper.z = self.bbmax.z;
}

inline __device__
bool loadCamera(EntityStruct &cameraEntity, CameraStruct &camera, TransformStruct &transform)
{
    auto &LP = optixLaunchParams;
    cameraEntity = LP.cameraEntity;
    if (!cameraEntity.initialized) return false;
    if ((cameraEntity.transform_id < 0) || (cameraEntity.transform_id >= LP.transforms.count)) return false;
    if ((cameraEntity.camera_id < 0) || (cameraEntity.camera_id >= LP.cameras.count)) return false;
    camera = LP.cameras.get(cameraEntity.camera_id, __LINE__);
    transform = LP.transforms.get(cameraEntity.transform_id, __LINE__);
    return true;
}

inline __device__ 
float3 sampleTexture(int32_t textureId, float2 texCoord, float3 defaultVal) {
    auto &LP = optixLaunchParams;
    if (textureId < 0 || textureId >= (LP.textures.count + LP.materials.count * NUM_MAT_PARAMS)) return defaultVal;
    cudaTextureObject_t tex = LP.textureObjects.get(textureId, __LINE__);
    if (!tex) return defaultVal;
    TextureStruct texInfo = LP.textures.get(textureId, __LINE__);
    texCoord.x = texCoord.x / texInfo.scale.x;
    texCoord.y = texCoord.y / texInfo.scale.y;
    return make_float3(tex2D<float4>(tex, texCoord.x, texCoord.y));
}

inline __device__ 
float sampleTexture(int32_t textureId, float2 texCoord, int8_t channel, float defaultVal) {
    auto &LP = optixLaunchParams;
    if (textureId < 0 || textureId >= (LP.textures.count + LP.materials.count * NUM_MAT_PARAMS)) return defaultVal;
    cudaTextureObject_t tex = LP.textureObjects.get(textureId, __LINE__);
    if (!tex) return defaultVal;
    TextureStruct texInfo = LP.textures.get(textureId, __LINE__);
    texCoord.x = texCoord.x / texInfo.scale.x;
    texCoord.y = texCoord.y / texInfo.scale.y;
    if (channel == 0) return tex2D<float4>(tex, texCoord.x, texCoord.y).x;
    if (channel == 1) return tex2D<float4>(tex, texCoord.x, texCoord.y).y;
    if (channel == 2) return tex2D<float4>(tex, texCoord.x, texCoord.y).z;
    if (channel == 3) return tex2D<float4>(tex, texCoord.x, texCoord.y).w;
    return defaultVal;
}

__device__
void loadMeshTriIndices(int meshID, int numIndices, int primitiveID, int3 &triIndices)
{
    auto &LP = optixLaunchParams;
    auto indices = LP.indexLists.get(meshID, __LINE__);
    triIndices = indices.get(primitiveID, __LINE__);   
}

__device__
void loadMeshVertexData(int meshID, int numVertices, int3 indices, float2 barycentrics, float3 &position, float3 &geometricNormal, float3 &edge1, float3 &edge2)
{
    auto &LP = optixLaunchParams;
    auto vertices = LP.vertexLists.get(meshID, __LINE__);
    const float3 A = vertices.get(indices.x, __LINE__);
    const float3 B = vertices.get(indices.y, __LINE__);
    const float3 C = vertices.get(indices.z, __LINE__);
    edge1 = B - A;
    edge2 = C - A;
    position = A * (1.f - (barycentrics.x + barycentrics.y)) + B * barycentrics.x + C * barycentrics.y;
    geometricNormal = normalize(cross(B-A,C-A));
}

__device__
void loadMeshUVData(int meshID, int numTexCoords, int3 indices, float2 barycentrics, float2 &uv, float2 &edge1, float2 &edge2)
{
    auto &LP = optixLaunchParams;
    auto texCoords = LP.texCoordLists.get(meshID, __LINE__);
    const float2 &A = texCoords.get(indices.x, __LINE__);
    const float2 &B = texCoords.get(indices.y, __LINE__);
    const float2 &C = texCoords.get(indices.z, __LINE__);
    edge1 = B - A;
    edge2 = C - A;
    uv = A * (1.f - (barycentrics.x + barycentrics.y)) + B * barycentrics.x + C * barycentrics.y;
}

__device__
void loadMeshNormalData(int meshID, int numNormals, int3 indices, float2 barycentrics, float3 &normal)
{
    auto &LP = optixLaunchParams;
    auto normals = LP.normalLists.get(meshID, __LINE__);
    const float3 &A = make_float3(normals.get(indices.x, __LINE__));
    const float3 &B = make_float3(normals.get(indices.y, __LINE__));
    const float3 &C = make_float3(normals.get(indices.z, __LINE__));
    normal = A * (1.f - (barycentrics.x + barycentrics.y)) + B * barycentrics.x + C * barycentrics.y;
}

__device__
void loadMeshTangentData(int meshID, int numTangents, int3 indices, float2 barycentrics, float3 &tangent)
{
    auto &LP = optixLaunchParams;
    auto tangents = LP.tangentLists.get(meshID, __LINE__);
    const float3 &A = make_float3(tangents.get(indices.x, __LINE__));
    const float3 &B = make_float3(tangents.get(indices.y, __LINE__));
    const float3 &C = make_float3(tangents.get(indices.z, __LINE__));
    tangent = A * (1.f - (barycentrics.x + barycentrics.y)) + B * barycentrics.x + C * barycentrics.y;
}

__device__ 
void loadDisneyMaterial(const MaterialStruct &p, float2 uv, DisneyMaterial &mat, float roughnessMinimum) {
    mat.base_color = sampleTexture(p.base_color_texture_id, uv, make_float3(.8f, .8f, .8f));
    mat.metallic = sampleTexture(p.metallic_texture_id, uv, p.metallic_texture_channel, .0f);
    mat.specular = sampleTexture(p.specular_texture_id, uv, p.specular_texture_channel, .5f);
    mat.roughness = sampleTexture(p.roughness_texture_id, uv, p.roughness_texture_channel, .5f);
    mat.specular_tint = sampleTexture(p.specular_tint_texture_id, uv, p.specular_tint_texture_channel, 0.f);
    mat.anisotropy = sampleTexture(p.anisotropic_texture_id, uv, p.anisotropic_texture_channel, 0.f);
    mat.sheen = sampleTexture(p.sheen_texture_id, uv, p.sheen_texture_channel, 0.f);
    mat.sheen_tint = sampleTexture(p.sheen_tint_texture_id, uv, p.sheen_tint_texture_channel, 0.5f);
    mat.clearcoat = sampleTexture(p.clearcoat_texture_id, uv, p.clearcoat_texture_channel, 0.f);
    float clearcoat_roughness = sampleTexture(p.clearcoat_roughness_texture_id, uv, p.clearcoat_roughness_texture_channel, 0.3f);
    mat.ior = sampleTexture(p.ior_texture_id, uv, p.ior_texture_channel, 1.45f);
    mat.specular_transmission = sampleTexture(p.transmission_texture_id, uv, p.transmission_texture_channel, 0.f);
    mat.flatness = sampleTexture(p.subsurface_texture_id, uv, p.subsurface_texture_channel, 0.f);
    mat.subsurface_color = sampleTexture(p.subsurface_color_texture_id, uv, make_float3(0.8f, 0.8f, 0.8f));
    mat.transmission_roughness = sampleTexture(p.transmission_roughness_texture_id, uv, p.transmission_roughness_texture_channel, 0.f);
    mat.alpha = sampleTexture(p.alpha_texture_id, uv, p.alpha_texture_channel, 1.f);
    
    mat.transmission_roughness = max(max(mat.transmission_roughness, MIN_ROUGHNESS), roughnessMinimum);
    mat.roughness = max(max(mat.roughness, MIN_ROUGHNESS), roughnessMinimum);
    clearcoat_roughness = max(clearcoat_roughness, roughnessMinimum);
    mat.clearcoat_gloss = 1.0 - clearcoat_roughness * clearcoat_roughness;
}

__device__
float sampleTime(float xi) {
    auto &LP = optixLaunchParams;
    return  LP.timeSamplingInterval[0] + 
           (LP.timeSamplingInterval[1] - 
            LP.timeSamplingInterval[0]) * xi;
}

inline __device__
owl::Ray generateRay(const CameraStruct &camera, const TransformStruct &transform, ivec2 pixelID, ivec2 frameSize, LCGRand &rng, float time)
{
    auto &LP = optixLaunchParams;
    /* Generate camera rays */    
    glm::quat r0 = glm::quat_cast(LP.viewT0);
    glm::quat r1 = glm::quat_cast(LP.viewT1);
    glm::vec4 p0 = glm::column(LP.viewT0, 3);
    glm::vec4 p1 = glm::column(LP.viewT1, 3);

    glm::vec4 pos = glm::mix(p0, p1, time);
    glm::quat rot = (glm::all(glm::equal(r0, r1))) ? r0 : glm::slerp(r0, r1, time);
    glm::mat4 camLocalToWorld = glm::mat4_cast(rot);
    camLocalToWorld = glm::column(camLocalToWorld, 3, pos);

    mat4 projinv = glm::inverse(LP.proj);
    mat4 viewinv = glm::inverse(camLocalToWorld);
    vec2 aa =  vec2(LP.xPixelSamplingInterval[0], LP.yPixelSamplingInterval[0])
            + (vec2(LP.xPixelSamplingInterval[1], LP.yPixelSamplingInterval[1]) 
            -  vec2(LP.xPixelSamplingInterval[0], LP.yPixelSamplingInterval[0])
            ) * vec2(lcg_randomf(rng),lcg_randomf(rng));

    vec2 inUV = (vec2(pixelID.x, pixelID.y) + aa) / vec2(frameSize);
    vec3 right = normalize(glm::column(viewinv, 0));
    vec3 up = normalize(glm::column(viewinv, 1));
    vec3 origin = glm::column(viewinv, 3);
    
    float cameraLensRadius = camera.apertureDiameter;

    vec3 p(0.f);
    if (cameraLensRadius > 0.0) {
        do {
            p = 2.0f*vec3(lcg_randomf(rng),lcg_randomf(rng),0.f) - vec3(1.f,1.f,0.f);
        } while (dot(p,p) >= 1.0f);
    }

    vec3 rd = cameraLensRadius * p;
    vec3 lens_offset = (right * rd.x) / float(frameSize.x) + (up * rd.y) / float(frameSize.y);

    origin = origin + lens_offset;
    vec2 dir = inUV * 2.f - 1.f; dir.y *= -1.f;
    vec4 t = (projinv * vec4(dir.x, dir.y, -1.f, 1.f));
    vec3 target = vec3(t) / float(t.w);
    vec3 direction = normalize(vec3(viewinv * vec4(target, 0.f))) * camera.focalDistance;
    direction = normalize(direction - lens_offset);

    owl::Ray ray;
    ray.tmin = .001f;
    ray.tmax = 1e20f;//10000.0f;
    ray.origin = make_float3(origin) ;
    ray.direction = make_float3(direction);
    
    return ray;
}

__device__
void initializeRenderData(float3 &renderData)
{
    auto &LP = optixLaunchParams;
    // these might change in the future...
    if (LP.renderDataMode == RenderDataFlags::NONE) {
        renderData = make_float3(FLT_MAX);
    }
    else if (LP.renderDataMode == RenderDataFlags::DEPTH) {
        renderData = make_float3(FLT_MAX);
    }
    else if (LP.renderDataMode == RenderDataFlags::POSITION) {
        renderData = make_float3(FLT_MAX);
    }
    else if (LP.renderDataMode == RenderDataFlags::NORMAL) {
        renderData = make_float3(FLT_MAX);
    }
    else if (LP.renderDataMode == RenderDataFlags::SCREEN_SPACE_NORMAL) {
        renderData = make_float3(0.0f);
    }
    else if (LP.renderDataMode == RenderDataFlags::ENTITY_ID) {
        renderData = make_float3(FLT_MAX);
    }
    else if (LP.renderDataMode == RenderDataFlags::BASE_COLOR) {
        renderData = make_float3(0.0, 0.0, 0.0);
    }
    else if (LP.renderDataMode == RenderDataFlags::TEXTURE_COORDINATES) {
        renderData = make_float3(0.0, 0.0, 0.0);
    }
    else if (LP.renderDataMode == RenderDataFlags::DIFFUSE_MOTION_VECTORS) {
        renderData = make_float3(0.0, 0.0, -1.0);
    }
    else if (LP.renderDataMode == RenderDataFlags::HEATMAP) {
        renderData = make_float3(0.0, 0.0, 0.0);
    }
}

__device__
void saveLightingColorRenderData (
    float3 &renderData, int bounce,
    float3 w_n, float3 w_o, float3 w_i, 
    DisneyMaterial &mat
)
{
    auto &LP = optixLaunchParams;
    if (LP.renderDataMode == RenderDataFlags::NONE) return;
    if (bounce != LP.renderDataBounce) return;
    
    // Note, dillum and iillum are expected to change outside this function depending on the 
    // render data flags.
    if (LP.renderDataMode == RenderDataFlags::DIFFUSE_COLOR) {
        renderData = disney_diffuse_color(mat, w_n, w_o, w_i, normalize(w_o + w_i));  
    }
    else if (LP.renderDataMode == RenderDataFlags::GLOSSY_COLOR) {
        renderData = disney_microfacet_reflection_color(mat, w_n, w_o, w_i, normalize(w_o + w_i));
    }
    else if (LP.renderDataMode == RenderDataFlags::TRANSMISSION_COLOR) {
        renderData = disney_microfacet_transmission_color(mat, w_n, w_o, w_i, normalize(w_o + w_i));
    }
}

__device__
void saveLightingIrradianceRenderData(
    float3 &renderData, int bounce,
    float3 dillum, float3 iillum,
    int sampledBsdf)
{
    auto &LP = optixLaunchParams;
    if (LP.renderDataMode == RenderDataFlags::NONE) return;
    if (bounce != LP.renderDataBounce) return;
    
    // Note, dillum and iillum are expected to change outside this function depending on the 
    // render data flags.
    if (LP.renderDataMode == RenderDataFlags::DIFFUSE_DIRECT_LIGHTING) {
        renderData = dillum;
    }
    else if (LP.renderDataMode == RenderDataFlags::DIFFUSE_INDIRECT_LIGHTING) {
        renderData = iillum;
    }
    else if (LP.renderDataMode == RenderDataFlags::GLOSSY_DIRECT_LIGHTING) {
        renderData = dillum;
    }
    else if (LP.renderDataMode == RenderDataFlags::GLOSSY_INDIRECT_LIGHTING) {
        renderData = iillum;
    }
    else if (LP.renderDataMode == RenderDataFlags::TRANSMISSION_DIRECT_LIGHTING) {
        renderData = dillum;
    }
    else if (LP.renderDataMode == RenderDataFlags::TRANSMISSION_INDIRECT_LIGHTING) {
        renderData = iillum;
    }
}

__device__
void saveMissRenderData(
    float3 &renderData, 
    int bounce,
    float3 mvec)
{
    auto &LP = optixLaunchParams;
    if (LP.renderDataMode == RenderDataFlags::NONE) return;
    if (bounce != LP.renderDataBounce) return;

    if (LP.renderDataMode == RenderDataFlags::DIFFUSE_MOTION_VECTORS) {
        renderData = mvec;
    }
}


__device__
void saveGeometricRenderData(
    float3 &renderData, 
    int bounce, float depth, 
    float3 w_p, float3 w_n, float3 w_o, float2 uv, 
    int entity_id, float3 diffuse_mvec, float time,
    DisneyMaterial &mat)
{
    auto &LP = optixLaunchParams;
    if (LP.renderDataMode == RenderDataFlags::NONE) return;
    if (bounce != LP.renderDataBounce) return;

    if (LP.renderDataMode == RenderDataFlags::DEPTH) {
        renderData = make_float3(depth);
    }
    else if (LP.renderDataMode == RenderDataFlags::POSITION) {
        renderData = w_p;
    }
    else if (LP.renderDataMode == RenderDataFlags::NORMAL) {
        renderData = w_n;
    }
    else if (LP.renderDataMode == RenderDataFlags::SCREEN_SPACE_NORMAL) {
        glm::quat r0 = glm::quat_cast(LP.viewT0);
        glm::quat r1 = glm::quat_cast(LP.viewT1);
        glm::quat rot = (glm::all(glm::equal(r0, r1))) ? r0 : glm::slerp(r0, r1, time);
        vec3 tmp = normalize(glm::mat3_cast(rot) * make_vec3(w_n));
        tmp = normalize(vec3(LP.proj * vec4(tmp, 0.f)));
        renderData.x = tmp.x;
        renderData.y = tmp.y;
        renderData.z = tmp.z;
    }
    else if (LP.renderDataMode == RenderDataFlags::ENTITY_ID) {
        renderData = make_float3(float(entity_id));
    }
    else if (LP.renderDataMode == RenderDataFlags::DIFFUSE_MOTION_VECTORS) {
        renderData = diffuse_mvec;
    }
    else if (LP.renderDataMode == RenderDataFlags::BASE_COLOR) {
        renderData = mat.base_color;
    }
    else if (LP.renderDataMode == RenderDataFlags::TEXTURE_COORDINATES) {
        renderData = make_float3(uv.x, uv.y, 0.0);
    }
    else if (LP.renderDataMode == RenderDataFlags::RAY_DIRECTION) {
        renderData = -w_o;
    }
}

__device__
void saveHeatmapRenderData(
    float3 &renderData, 
    int bounce,
    uint64_t start_clock
)
{
    auto &LP = optixLaunchParams;
    if (LP.renderDataMode != RenderDataFlags::HEATMAP) return;
    // if (bounce < LP.renderDataBounce) return;

    uint64_t absClock = clock()-start_clock;
    float relClock = /*global.clockScale **/ absClock / 10000000.f;
    relClock = min(relClock, 1.f);
    renderData = make_float3(relClock);
}

__device__
float3 faceNormalForward(const float3 &w_o, const float3 &gn, const float3 &n)
{
    float3 new_n = n;
    // if (dot(w_o, new_n) < 0.f) {
    //     // prevents differences from geometric and shading normal from creating black artifacts
    //     new_n = reflect(-new_n, gn); 
    // }
    if (dot(w_o, new_n) < 0.f) {
        new_n = -new_n;
    }
    return new_n;
}

__device__
bool debugging() {
    #ifndef DEBUGGING
    return false;
    #endif
    auto &LP = optixLaunchParams;
    auto pixelID = ivec2(owl::getLaunchIndex()[0], owl::getLaunchIndex()[1]);
    return glm::all(glm::equal(pixelID, ivec2(LP.frameSize.x / 2, LP.frameSize.y / 2)));
}

__device__
glm::mat4 test_interpolate(glm::mat4& _mat1, glm::mat4& _mat2, float _time)
{
    glm::quat rot0 = glm::quat_cast(_mat1);
    glm::quat rot1= glm::quat_cast(_mat2);

    glm::quat finalRot = glm::slerp(rot0, rot1, _time);

    glm::mat4 finalMat = glm::mat4_cast(finalRot);

    finalMat[3] = _mat1[3] * (1 - _time) + _mat2[3] * _time;
    
    return finalMat;
}

/// Taken and modified from Algorithm 2 in "Pixar's Production Volume Rendering" paper.
/// \param x The origin of the ray.
/// \param w The direction of the light (opposite of ray direction). 
/// \param d The distance along the ray to the boundary.
/// \param t The returned hit distance.
/// \param event Will be updated to represent the event that occured during tracking.
///              0 means the boundary was hit
///              1 means an absorption/emission occurred
///              2 means a scattering collision occurred
///              3 means a null collision occurred
template<typename AccT>
__device__
void SampleDeltaTracking(
    LCGRand &rng, 
    AccT& acc, 
    float majorant_extinction, 
    float linear_attenuation_unit, 
    float absorption_, 
    float scattering_, 
    vec3 x, 
    vec3 w, 
    float d, 
    float &t, 
    int &event
) {
    float rand1 = lcg_randomf(rng);
    float rand2 = lcg_randomf(rng);
    
    // Set new t for the current x.
    t = (-log(1.0f - rand1) / majorant_extinction) * linear_attenuation_unit;
    
    // A boundary has been hit
    if (t >= d) {
    	event = 0;
        t = d;
        return;
    }
    
    // Update current position
    x = x - t * w;
    auto coord_pos = nanovdb::Coord::Floor( nanovdb::Vec3f(x.x, x.y, x.z) );
    float densityValue = acc.getValue(coord_pos);

   	float absorption = densityValue * absorption_; //sample_volume_absorption(x);
    float scattering = densityValue * scattering_; //sample_volume_scattering(x);
    float extinction = absorption + scattering;
    //float null_collision = 1.f - extinction;
    float null_collision = majorant_extinction - extinction;
    
    //extinction = extinction / majorant_extinction;
    absorption = absorption / majorant_extinction;
    scattering = scattering / majorant_extinction;
    null_collision = null_collision / majorant_extinction;
    
        
    // An absorption/emission collision occured
    if (rand2 < absorption) 
    {
    	event = 1;
        return;
    }
    
    // A scattering collision occurred
    else if (rand2 < (absorption + scattering)) {
    //else if (rand2 < (1.f - null_collision)) {
        event = 2;
        return;
    }
    
    // A null collision occurred
    else {
        event = 3;
        return;    	
    }
}

/// Taken and modified from Algorithm 2 in "Pixar's Production Volume Rendering" paper.
/// Implements the top level delta tracking algorithm, returning radiance.
/// \param seed The seed to use by the random number generator.
/// \param x The origin of the ray.
/// \param w The direction of the light (opposite of ray direction). 
/// \param d The distance along the ray to the boundary.
template<typename AccT>
__device__
vec3 DeltaTracking(
    LCGRand &rng, 
    AccT& acc, 
    cudaTextureObject_t &envTex,
    float majorant_extinction, 
    float linear_attenuation_unit, 
    float absorption, 
    float scattering, 
    vec3 x, 
    vec3 w
) {
    auto &LP = optixLaunchParams;

    auto bbox = acc.root().bbox();
    #define MAX_VOLUME_DEPTH 10000
    float t0, t1;
    
    auto wRay = nanovdb::Ray<float>(
        reinterpret_cast<const nanovdb::Vec3f&>( x ),
        reinterpret_cast<const nanovdb::Vec3f&>( -w )
    );
    wRay.setTimes(EPSILON, 1e20f);
    bool hit = wRay.clip(bbox);
    if (!hit) return make_vec3(missColor(make_float3(-w), envTex) * LP.domeLightIntensity * pow(2.f, LP.domeLightExposure));
    t0 = wRay.t0();
    t1 = wRay.t1();
    
    // Move ray to volume boundary
    x = x - t0 * w;
    t1 = t1 - t0;
    t0 = 0.f;
        
    // Note: original algorithm had unlimited bounces. 
    vec3 throughput = vec3(1.f);
    for (int i = 0; i < MAX_VOLUME_DEPTH; ++i) {
        int event = 0;
        float t = 0.f;
        SampleDeltaTracking(rng, acc, majorant_extinction, linear_attenuation_unit, absorption, scattering, x, w, t1, t, event);
        x = x - t * w;
        
        // A boundary has been hit. Sample the background.
        if (event == 0) return throughput * make_vec3(missColor(make_float3(-w), envTex) * LP.domeLightIntensity * pow(2.f, LP.domeLightExposure));
        
        // An absorption / emission occurred.
        if (event == 1) return throughput * vec3(0.f);//vec3(sample_volume_emission(x));
        
        // A scattering collision occurred.
        if (event == 2) {            
            float rand1 = lcg_randomf(rng);
            float rand2 = lcg_randomf(rng);
            // Sample isotropic phase function to get new ray direction           
            float phi = 2.0f * M_PI * rand1;
            float cos_theta = 1.0f - 2.0f * rand2;
            float sin_theta = sqrt (1.0f - cos_theta * cos_theta);
            w = -vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
            
            // Compute updated boundary
            auto wRay = nanovdb::Ray<float>(
                reinterpret_cast<const nanovdb::Vec3f&>( x ),
                reinterpret_cast<const nanovdb::Vec3f&>( -w )
            );
            bool hit = wRay.clip(bbox);
            if (!hit) 
                return vec3(0.f,1.f,0.f);//throughput * make_vec3(missColor(make_float3(-w), envTex) * LP.domeLightIntensity * pow(2.f, LP.domeLightExposure));
            t0 = wRay.t0();
            t1 = wRay.t1();
        }
        
        // A null collision occurred.
        if (event == 3) {
            // update boundary in relation to the new collision x, w does not change.
            t1 = t1 - t;
        }
    }
    
    // If we got stuck in the volume
    return vec3(1.f, 0.f, 0.f);
    return throughput * make_vec3(missColor(make_float3(-w), envTex) * LP.domeLightIntensity * pow(2.f, LP.domeLightExposure));
}

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
    const RayGenData &self = owl::getProgramData<RayGenData>();
    
    auto &LP = optixLaunchParams;
    auto launchIndex = optixGetLaunchIndex().x;
    auto launchDim = optixGetLaunchDimensions().x;
    auto pixelID = ivec2(launchIndex % LP.frameSize.x, launchIndex / LP.frameSize.x);

    /* compute who is repsonible for a given group of pixels */
    /* and if it's not us, just return. */
    /* (some other device will compute these pixels) */
    // int deviceThatIsResponsible = (pixelID.x>>5) % self.deviceCount;
    int deviceThatIsResponsible = (pixelID.x>>5) % self.deviceCount;
    if (self.deviceIndex != deviceThatIsResponsible) {
        // auto fbOfs = pixelID.x+LP.frameSize.x * ((LP.frameSize.y - 1) -  pixelID.y);
        // float4* accumPtr = (float4*) LP.accumPtr;
        // float4* fbPtr = (float4*) LP.frameBuffer;
        // accumPtr[fbOfs] = make_float4(0.f);
        // fbPtr[fbOfs] = make_float4(0.f);
        return;
    }

    auto dims = ivec2(LP.frameSize.x, LP.frameSize.x);
    uint64_t start_clock = clock();
    int numLights = LP.numLightEntities;
    int numLightSamples = LP.numLightSamples;
    bool enableDomeSampling = LP.enableDomeSampling;
    
    LCGRand rng = get_rng(LP.frameID + LP.seed * 10007, make_uint2(pixelID.x, pixelID.y), make_uint2(dims.x, dims.y));
    float time = sampleTime(lcg_randomf(rng));

    // If no camera is in use, just display some random noise...
    owl::Ray ray;
    {
        EntityStruct    camera_entity;
        TransformStruct camera_transform;
        CameraStruct    camera;
        if (!loadCamera(camera_entity, camera, camera_transform)) {
            auto fbOfs = pixelID.x+LP.frameSize.x * ((LP.frameSize.y - 1) -  pixelID.y);
            LP.frameBuffer[fbOfs] = vec4(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng), 1.f);
            return;
        }
        
        // Trace an initial ray through the scene
        ray = generateRay(camera, camera_transform, pixelID, LP.frameSize, rng, time);
    }

    cudaTextureObject_t envTex = getEnvironmentTexture();

    float3 accum_illum = make_float3(0.f);
    float3 pathThroughput = make_float3(1.f);
    float3 renderData = make_float3(0.f);
    float3 primaryAlbedo = make_float3(0.f);
    float3 primaryNormal = make_float3(0.f);
    initializeRenderData(renderData);

    uint8_t bounce = 0;
    uint8_t diffuseBounce = 0;
    uint8_t specularBounce = 0;
    uint8_t visibilitySkips = 0;

    // direct here is used for final image clamping
    float3 directIllum = make_float3(0.f);
    float3 illum = make_float3(0.f);
    
    RayPayload payload;
    payload.tHit = -1.f;
    ray.time = time;
    
    // temporary volume rendering test code
    RayPayload volPayload;
    volPayload.tHit = -1.f;
    owl::Ray volRay = ray;

    owl::traceRay(  /*accel to trace against*/ LP.surfacesIAS,
                    /*the ray to trace*/ ray,
                    /*prd*/ payload,
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT);

    owl::traceRay(  /*accel to trace against*/ LP.volumesIAS,
                /*the ray to trace*/ volRay,
                    /*prd*/ volPayload,
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT);
    
    if (volPayload.tHit != -1.f) {
        const int entityID = LP.volumeInstanceToEntity.get(volPayload.instanceID, __LINE__);
        EntityStruct entity = LP.entities.get(entityID, __LINE__);
        VolumeStruct volume = LP.volumes.get(entity.volume_id, __LINE__);
        TransformStruct transform = LP.transforms.get(entity.transform_id, __LINE__);

        // for now, assuming transform is identity.
        uint8_t *hdl = (uint8_t*)LP.volumeHandles.get(0, __LINE__).data;
        const auto grid = reinterpret_cast<const nanovdb::FloatGrid*>(hdl);
        const auto& tree = grid->tree();
        auto acc = tree.getAccessor();
        float majorant = volume.majorant;//tree.root().valueMax();
        float linear_attenuation_unit = volume.scale;//max(max(grid->voxelSize()[0], grid->voxelSize()[1]), grid->voxelSize()[2]);
        float absorption = volume.absorption;
        float scattering = volume.scattering;
        vec3 color = DeltaTracking(
            rng, acc, envTex, 
            majorant, linear_attenuation_unit, absorption, scattering,
            make_vec3(volRay.origin), -make_vec3(volRay.direction));

        auto fbOfs = pixelID.x+LP.frameSize.x * ((LP.frameSize.y - 1) -  pixelID.y);
        float4* accumPtr = (float4*) LP.accumPtr;
        float4* fbPtr = (float4*) LP.frameBuffer;
        float4* normalPtr = (float4*) LP.normalBuffer;
        float4* albedoPtr = (float4*) LP.albedoBuffer;

        float t0 = volRay.tmin;
        float t1 = volRay.tmax;
        auto bbox = acc.root().bbox();
        auto wRay = nanovdb::Ray<float>(
            reinterpret_cast<const nanovdb::Vec3f&>( volRay.origin ),
            reinterpret_cast<const nanovdb::Vec3f&>( volRay.direction )
        );
        wRay.setTimes(t0, t1);
        bool hit = wRay.clip(bbox);
        if (
            (pixelID.x == int(LP.frameSize.x / 2)) && 
            (pixelID.y == int(LP.frameSize.y / 2))
        ) {
            float3 mn = make_float3(bbox.min()[0], bbox.min()[1], bbox.min()[2]);
            float3 mx = make_float3(bbox.max()[0], bbox.max()[1], bbox.max()[2]);
            // printf("hit %d bb %f %f %f, %f %f %f\n", hit, mn.x, mn.y, mn.z, mx.x, mx.y, mx.z);
        }

        float4 fbcolor = make_float4(color, 1.f);
        float4 prev_color = accumPtr[fbOfs];
        float4 accum_color = make_float4((make_float3(fbcolor) + float(LP.frameID) * make_float3(prev_color)) / float(LP.frameID + 1), 1.0f);

        // if (lockout < 0) color = make_float4(1.f, 0.f, 0.f, 1.f);
        accumPtr[fbOfs] = accum_color;
        fbPtr[fbOfs] = accum_color;
        albedoPtr[fbOfs] = accum_color;
        normalPtr[fbOfs] = accum_color;
        return;

        // uint64_t checksum = grid->checksum();
        // auto bbox = grid->tree().bbox().asReal<float>();

        //((void**)LP.volumeHandles.data)[0];// ((void*)LP.volumeHandles.data)[0];//getPtr(entity.volume_id, __LINE__);
        // bool valid = grid->isValid();
        
        // return;


        
        // // {
        //     // auto *data = reinterpret_cast<const typename GridT::DataType*>(grid);
        //     // if (data->mMagic != NANOVDB_MAGIC_NUMBER) {
        //     //     printf("Incorrect magic number: Expected %d, but read %d\n)", NANOVDB_MAGIC_NUMBER, data->mMagic);
        //     //     // mErrorStr = ss.str();
        //     // } 
        //     // else if (!validateChecksum(*mGrid, detailed ? ChecksumMode::Full : ChecksumMode::Partial)) {
        //     //     mErrorStr.assign("Mis-matching checksum");
        //     // } else if (data->mMajor != NANOVDB_MAJOR_VERSION_NUMBER) {
        //     //     ss << "Invalid major version number: Expected " << NANOVDB_MAJOR_VERSION_NUMBER << ", but read " << data->mMajor;
        //     //     mErrorStr = ss.str();
        //     // } else if (data->mGridClass >= GridClass::End) {
        //     //     mErrorStr.assign("Invalid Grid Class");
        //     // } else if (data->mGridType != mapToGridType<ValueT>()) {
        //     //     mErrorStr.assign("Invalid Grid Type");
        //     // } else if ( (const void*)(&(mGrid->tree())) != (const void*)(mGrid+1) ) {
        //     //     mErrorStr.assign("Invalid Tree pointer");
        //     // }
        // // }
        // // float3 mn = make_float3(bbox.min()[0], bbox.min()[1], bbox.min()[2]);
        // // float3 mx = make_float3(bbox.max()[0], bbox.max()[1], bbox.max()[2]);
        // // printf("bb %f %f %f, %f %f %f\n", mn.x, mn.y, mn.z, mx.x, mx.y, mx.z);
        // // printf("t0 %f, t1 %f\n", iRay.t0(), iRay.t1());

        
        
        
        // // nanovdb::isValid(*grid, true, true);
        
        // // printf("Checksum: %llu\n", checksum);
        

        // auto wRay = nanovdb::Ray<float>(
        //     reinterpret_cast<const nanovdb::Vec3f&>( volRay.origin ),
        //     reinterpret_cast<const nanovdb::Vec3f&>( volRay.direction )
        // );
        // nanovdb::Ray<float> iRay = wRay;//wRay.worldToIndexF(*grid);
        // iRay.setTimes(t0, t1);
        
        // const float voxelSize = 1.f;//static_cast<float>(grid->voxelSize()[0]);
        // bool hit = iRay.clip(bbox);
        // float transmittance = 1.f;
        // if (hit) {
            


        //     float opacity = .125f;
        //     const float dt = 1.f;//voxelSize;
        //     // int lockout = 1000;
        //     for (float t = iRay.t0(); t < iRay.t1(); t += dt) {
        //         auto  densityValue = acc.getValue(nanovdb::Coord::Floor(iRay(t)));
        //         float densityScalar = densityValue * opacity;
        //         transmittance *= expf( -densityScalar * dt );
        //         // lockout--;
        //         // if (lockout < 0) {
        //         //     break;
        //         // }
        //     }
        //     // return transmittance;


            
        // }

    }

    // Shade each hit point on a path using NEE with MIS
    do {     
        float alpha = 0.f;

        // If ray misses, terminate the ray
        if (payload.tHit <= 0.f) {
            // Compute lighting from environment
            if (bounce == 0) {
                float3 col = missColor(ray, envTex);
                illum = illum + pathThroughput * (col * LP.domeLightIntensity);
                directIllum = illum;
                primaryAlbedo = col;
            }
            else if (enableDomeSampling)
                illum = illum + pathThroughput * (missColor(ray, envTex) * LP.domeLightIntensity * pow(2.f, LP.domeLightExposure));
            
            const float envDist = 10000.0f; // large value
            /* Compute miss motion vector */
            float3 mvec;
            // Point far away
            float3 pFar = ray.origin + ray.direction * envDist;
            // TODO: account for motion from rotating dome light
            vec4 tmp1 = LP.proj * LP.viewT0 * /*xfmt0 **/ make_vec4(pFar, 1.0f);
            float3 pt0 = make_float3(tmp1 / tmp1.w) * .5f;
            vec4 tmp2 = LP.proj * LP.viewT1 * /*xfmt1 **/ make_vec4(pFar, 1.0f);
            float3 pt1 = make_float3(tmp2 / tmp2.w) * .5f;
            mvec = pt1 - pt0;
            saveMissRenderData(renderData, bounce, mvec);
            break;
        }

        // Otherwise, load the object we hit.
        const int entityID = LP.surfaceInstanceToEntity.get(payload.instanceID, __LINE__);
        EntityStruct entity = LP.entities.get(entityID, __LINE__);
        MeshStruct mesh = LP.meshes.get(entity.mesh_id, __LINE__);
        TransformStruct transform = LP.transforms.get(entity.transform_id, __LINE__);

        // Skip forward if the hit object is invisible for this ray type, skip it.
        if (((entity.flags & ENTITY_VISIBILITY_CAMERA_RAYS) == 0)) {
            ray.origin = ray.origin + ray.direction * (payload.tHit + EPSILON);
            payload.tHit = -1.f;
            ray.time = time;
            owl::traceRay( LP.surfacesIAS, ray, payload, OPTIX_RAY_FLAG_DISABLE_ANYHIT);
            visibilitySkips++;
            if (visibilitySkips > 10) break; // avoid locking up.
            continue;
        }

        // Set new outgoing light direction and hit position.
        const float3 w_o = -ray.direction;
        float3 hit_p = ray.origin + payload.tHit * ray.direction;

        // Load geometry data for the hit object
        float3 mp, p, v_x, v_y, v_z, v_gz, p_e1, p_e2; 
        float2 uv, uv_e1, uv_e2; 
        int3 indices;
        float3 diffuseMotion;
        loadMeshTriIndices(entity.mesh_id, mesh.numTris, payload.primitiveID, indices);
        loadMeshVertexData(entity.mesh_id, mesh.numVerts, indices, payload.barycentrics, mp, v_gz, p_e1, p_e2); // todo, remomve pe1,pe2
        loadMeshUVData(entity.mesh_id, mesh.numVerts, indices, payload.barycentrics, uv, uv_e1, uv_e2); // todo, remove e1, e2
        loadMeshNormalData(entity.mesh_id, mesh.numVerts, indices, payload.barycentrics, v_z);
        loadMeshTangentData(entity.mesh_id, mesh.numVerts, indices, payload.barycentrics, v_x);

        // Load material data for the hit object
        DisneyMaterial mat; MaterialStruct entityMaterial;
        if (entity.material_id >= 0 && entity.material_id < LP.materials.count) {
            entityMaterial = LP.materials.get(entity.material_id, __LINE__);
            loadDisneyMaterial(entityMaterial, uv, mat, MIN_ROUGHNESS);
        }
       
        // Compute tangent and bitangent based on UVs
        // {
        //     float f = 1.0f / (uv_e1.x * uv_e2.y - uv_e2.x * uv_e1.y);
        //     v_x.x = f * (uv_e2.y * p_e1.x - uv_e1.y * p_e2.x);
        //     v_x.y = f * (uv_e2.y * p_e1.y - uv_e1.y * p_e2.y);
        //     v_x.z = f * (uv_e2.y * p_e1.z - uv_e1.y * p_e2.z);
        // }
            
        // Transform geometry data into world space
        {
            // both glm::interpolate and my own interpolation functions cause artifacts... 
            // optix transform seems to work though
            // glm::mat4 xfm = test_interpolate(transform.localToWorld, transform.localToWorld, time);
            glm::mat4 xfm = to_mat4(payload.localToWorld);
            p = make_float3(xfm * make_vec4(mp, 1.0f));
            hit_p = p;
            glm::mat3 nxfm = transpose(glm::inverse(glm::mat3(xfm)));
            v_gz = make_float3(normalize(nxfm * make_vec3(v_gz)));
            v_z = make_float3(normalize(nxfm * make_vec3(v_z)));
            v_x = make_float3(normalize(nxfm * make_vec3(v_x)));

            
            v_y = -cross(v_z, v_x);
            v_x = -cross(v_y, v_z);

            if (LP.renderDataMode != RenderDataFlags::NONE) {
                // glm::mat4 xfmt0 = transform.localToWorldPrev;
                // glm::mat4 xfmt1 = transform.localToWorld;
                glm::mat4 xfmt0 = to_mat4(payload.localToWorldT0);
                glm::mat4 xfmt1 = to_mat4(payload.localToWorldT1);
                vec4 tmp1 = LP.proj * LP.viewT0 * xfmt0 * make_vec4(mp, 1.0f);
                vec4 tmp2 = LP.proj * LP.viewT1 * xfmt1 * make_vec4(mp, 1.0f);
                float3 pt0 = make_float3(tmp1 / tmp1.w) * .5f;
                float3 pt1 = make_float3(tmp2 / tmp2.w) * .5f;
                diffuseMotion = pt1 - pt0;
            } else {
                diffuseMotion = make_float3(0.f, 0.f, 0.f);
            }
        }    
        
        // Fallback for tangent and bitangent if UVs result in degenerate vectors.
        if (
            all(lessThan(abs(make_vec3(v_x)), vec3(EPSILON))) || 
            all(lessThan(abs(make_vec3(v_y)), vec3(EPSILON))) ||
            any(isnan(make_vec3(v_x))) || 
            any(isnan(make_vec3(v_y)))
        ) {
            ortho_basis(v_x, v_y, v_z);
        }

        // Construct TBN matrix, sample normal map
        {
            glm::mat3 tbn;
            tbn = glm::column(tbn, 0, make_vec3(v_x) );
            tbn = glm::column(tbn, 1, make_vec3(v_y) );
            tbn = glm::column(tbn, 2, make_vec3(v_z) );   
            float3 dN;
            if (entity.light_id >= 0 && entity.light_id < LP.lights.count) {
                dN = make_float3(0.5f, .5f, 1.f);
            } else {
                dN = sampleTexture(entityMaterial.normal_map_texture_id, uv, make_float3(0.5f, .5f, 1.f));
            }

            dN = normalize( (dN * make_float3(2.0f)) - make_float3(1.f) );   
            v_z = make_float3(tbn * make_vec3(dN));
        }

        // // TEMP CODE
        // auto fbOfs = pixelID.x+LP.frameSize.x * ((LP.frameSize.y - 1) -  pixelID.y);
        // LP.frameBuffer[fbOfs] = make_vec4(v_z, 1.f);
        // return;

        // If we didn't hit glass, flip the surface normal to face forward.
        if ((mat.specular_transmission == 0.f) && (entity.light_id == -1)) {
            v_z = faceNormalForward(w_o, v_gz, v_z);
        }

        // For segmentations, save geometric metadata
        saveGeometricRenderData(renderData, bounce, payload.tHit, hit_p, v_z, w_o, uv, entityID, diffuseMotion, time, mat);
        if (bounce == 0) {
            primaryAlbedo = mat.base_color;
            primaryNormal = v_z;
        }

        // Potentially skip forward if the hit object is transparent 
        if ((entity.light_id == -1) && (mat.alpha < 1.f)) {
            float alpha_rnd = lcg_randomf(rng);

            if (alpha_rnd > mat.alpha) {
                ray.origin = ray.origin + ray.direction * (payload.tHit + EPSILON);
                payload.tHit = -1.f;
                ray.time = time;
                owl::traceRay( LP.surfacesIAS, ray, payload, OPTIX_RAY_FLAG_DISABLE_ANYHIT);
                ++bounce;     
                specularBounce++; // counting transparency as a specular bounce for now
                continue;
            }
        }

        // If the entity we hit is a light, terminate the path.
        // Note that NEE/MIS will also potentially terminate the path, preventing double-counting.
        if (entity.light_id >= 0 && entity.light_id < LP.lights.count) {
            float dotNWi = max(dot(ray.direction, v_z), 0.f);
            if ((dotNWi > EPSILON) && (bounce != 0)) break;

            LightStruct entityLight = LP.lights.get(entity.light_id, __LINE__);
            float3 lightEmission;
            if (entityLight.color_texture_id == -1) lightEmission = make_float3(entityLight.r, entityLight.g, entityLight.b);
            else lightEmission = sampleTexture(entityLight.color_texture_id, uv, make_float3(0.f, 0.f, 0.f));
            float dist = payload.tHit;
            lightEmission = (lightEmission * entityLight.intensity);
            if (bounce != 0) lightEmission = (lightEmission * pow(2.f, entityLight.exposure)) / (dist * dist);
            float3 contribution = pathThroughput * lightEmission;
            illum = illum + contribution;
            if (bounce == 0) directIllum = illum;
            break;
        }

        // Next, we'll be sampling direct light sources
        int32_t sampledLightIDs[MAX_LIGHT_SAMPLES] = {-2};
        float lightPDFs[MAX_LIGHT_SAMPLES] = {0.f};
        float3 irradiance = make_float3(0.f);

        // note, rdForcedBsdf is -1 by default
        int forcedBsdf = -1;

        // First, sample the BRDF so that we can use the sampled direction for MIS
        float3 w_i;
        float bsdfPDF;
        int sampledBsdf = -1;
        float3 bsdf, bsdfColor;
        sample_disney_brdf(mat, v_z, w_o, v_x, v_y, rng, w_i, bsdfPDF, sampledBsdf, bsdf, bsdfColor, forcedBsdf);

        // Next, sample the light source by importance sampling the light
        const uint32_t occlusion_flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;
        for (uint32_t lid = 0; lid < numLightSamples; ++lid) 
        {
            uint32_t randmax = (enableDomeSampling) ? numLights + 1 : numLights;
            uint32_t randomID = uint32_t(min(lcg_randomf(rng) * randmax, float(randmax-1)));
            float dotNWi;
            float3 bsdf, bsdfColor;
            float3 lightEmission;
            float3 lightDir;
            float lightDistance = 1e20f;
            float falloff = 2.0f;
            int numTris;

            // sample background
            if (randomID == numLights) {
                sampledLightIDs[lid] = -1;
                if (
                    (LP.environmentMapWidth != 0) && (LP.environmentMapHeight != 0) &&
                    (LP.environmentMapRows != nullptr) && (LP.environmentMapCols != nullptr) 
                ) 
                {
                    // Reduces noise for strangely noisy dome light textures, but at the expense 
                    // of a highly uncoalesced binary search through a 2D CDF.
                    // disabled by default to avoid the hit to performance
                    float rx = lcg_randomf(rng);
                    float ry = lcg_randomf(rng);
                    float* rows = LP.environmentMapRows;
                    float* cols = LP.environmentMapCols;
                    int width = LP.environmentMapWidth;
                    int height = LP.environmentMapHeight;
                    float invjacobian = width * height / float(4 * M_PI);
                    float row_pdf, col_pdf;
                    unsigned x, y;
                    ry = sample_cdf(rows, height, ry, &y, &row_pdf);
                    y = max(min(y, height - 1), 0);
                    rx = sample_cdf(cols + y * width, width, rx, &x, &col_pdf);
                    lightDir = make_float3(toPolar(vec2((x /*+ rx*/) / float(width), (y/* + ry*/)/float(height))));
                    lightDir = glm::inverse(LP.environmentMapRotation) * lightDir;
                    lightPDFs[lid] = row_pdf * col_pdf * invjacobian;
                } 
                else 
                {            
                    glm::mat3 tbn;
                    tbn = glm::column(tbn, 0, make_vec3(v_x) );
                    tbn = glm::column(tbn, 1, make_vec3(v_y) );
                    tbn = glm::column(tbn, 2, make_vec3(v_z) );            
                    const float3 hemi_dir = (cos_sample_hemisphere(make_float2(lcg_randomf(rng), lcg_randomf(rng))));
                    lightDir = make_float3(tbn * make_vec3(hemi_dir));
                    lightPDFs[lid] = 1.f / float(2.0 * M_PI);
                }

                numTris = 1.f;
                lightEmission = (missColor(lightDir, envTex) * LP.domeLightIntensity * pow(2.f, LP.domeLightExposure));
            }
            // sample light sources
            else 
            {
                if (numLights == 0) continue;
                sampledLightIDs[lid] = LP.lightEntities.get(randomID, __LINE__);
                EntityStruct light_entity = LP.entities.get(sampledLightIDs[lid], __LINE__);
                LightStruct light_light = LP.lights.get(light_entity.light_id, __LINE__);
                TransformStruct transform = LP.transforms.get(light_entity.transform_id, __LINE__);
                MeshStruct mesh = LP.meshes.get(light_entity.mesh_id, __LINE__);
                uint32_t random_tri_id = uint32_t(min(lcg_randomf(rng) * mesh.numTris, float(mesh.numTris - 1)));
                auto indices = LP.indexLists.get(light_entity.mesh_id, __LINE__);
                auto vertices = LP.vertexLists.get(light_entity.mesh_id, __LINE__);
                auto normals = LP.normalLists.get(light_entity.mesh_id, __LINE__);
                auto texCoords = LP.texCoordLists.get(light_entity.mesh_id, __LINE__);
                int3 triIndex = indices.get(random_tri_id, __LINE__);
                
                // Sample the light to compute an incident light ray to this point
                auto &ltw = transform.localToWorld;
                float3 dir; float2 uv;
                float3 pos = hit_p;
                 // Might be a bug here with normal transform...
                float3 n1 = make_float3(ltw * normals.get(triIndex.x, __LINE__));
                float3 n2 = make_float3(ltw * normals.get(triIndex.y, __LINE__));
                float3 n3 = make_float3(ltw * normals.get(triIndex.z, __LINE__));
                float3 v1 = make_float3(ltw * make_float4(vertices.get(triIndex.x, __LINE__), 1.0f));
                float3 v2 = make_float3(ltw * make_float4(vertices.get(triIndex.y, __LINE__), 1.0f));
                float3 v3 = make_float3(ltw * make_float4(vertices.get(triIndex.z, __LINE__), 1.0f));
                float2 uv1 = texCoords.get(triIndex.x, __LINE__);
                float2 uv2 = texCoords.get(triIndex.y, __LINE__);
                float2 uv3 = texCoords.get(triIndex.z, __LINE__);
                sampleTriangle(pos, n1, n2, n3, v1, v2, v3, uv1, uv2, uv3, 
                    lcg_randomf(rng), lcg_randomf(rng), dir, lightDistance, lightPDFs[lid], uv, 
                    /*double_sided*/ false, /*use surface area*/ light_light.use_surface_area);
                
                falloff = light_light.falloff;
                numTris = mesh.numTris;
                lightDir = make_float3(dir.x, dir.y, dir.z);
                if (light_light.color_texture_id == -1) lightEmission = make_float3(light_light.r, light_light.g, light_light.b) * (light_light.intensity * pow(2.f, light_light.exposure));
                else lightEmission = sampleTexture(light_light.color_texture_id, uv, make_float3(0.f, 0.f, 0.f)) * (light_light.intensity * pow(2.f, light_light.exposure));
            }

            disney_brdf(mat, v_z, w_o, lightDir, normalize(w_o + lightDir), v_x, v_y, bsdf, bsdfColor, forcedBsdf);
            dotNWi = max(dot(lightDir, v_z), 0.f);
            lightPDFs[lid] *= (1.f / float(numLights + 1.f)) * (1.f / float(numTris));
            if ((lightPDFs[lid] > 0.0) && (dotNWi > EPSILON)) {
                RayPayload payload; payload.instanceID = -2;
                owl::RayT</*type*/1, /*prd*/1> ray; // shadow ray
                ray.tmin = EPSILON * 10.f; ray.tmax = lightDistance + EPSILON; // needs to be distance to light, else anyhit logic breaks.
                ray.origin = hit_p; ray.direction = lightDir;
                ray.time = time;
                owl::traceRay( LP.surfacesIAS, ray, payload, occlusion_flags);
                bool visible = (randomID == numLights) ?
                    (payload.instanceID == -2) : 
                    ((payload.instanceID == -2) || (LP.surfaceInstanceToEntity.get(payload.instanceID, __LINE__) == sampledLightIDs[lid]));
                if (visible) {
                    if (randomID != numLights) lightEmission = lightEmission / pow(payload.tHit, falloff);
                    float w = power_heuristic(1.f, lightPDFs[lid], 1.f, bsdfPDF);
                    float3 Li = (lightEmission * w) / lightPDFs[lid];
                    irradiance = irradiance + (bsdf * bsdfColor * Li);
                }
            }
        }

        // For segmentations, save lighting metadata
        saveLightingColorRenderData(renderData, bounce, v_z, w_o, w_i, mat);

        // Terminate the path if the bsdf probability is impossible, or if the bsdf filters out all light
        if (bsdfPDF < EPSILON || all_zero(bsdf) || all_zero(bsdfColor)) {
            float3 contribution = pathThroughput * irradiance;
            illum = illum + contribution;
            break;
        }

        // Next, sample a light source using the importance sampled BDRF direction.
        ray.origin = hit_p;
        ray.direction = w_i;
        ray.tmin = EPSILON;//* 100.f;
        payload.instanceID = -1;
        payload.tHit = -1.f;
        ray.time = sampleTime(lcg_randomf(rng));
        owl::traceRay(LP.surfacesIAS, ray, payload, OPTIX_RAY_FLAG_DISABLE_ANYHIT);

        // Check if we hit any of the previously sampled lights
        bool hitLight = false;
        for (uint32_t lid = 0; lid < numLightSamples; ++lid)
        {
            if (lightPDFs[lid] > EPSILON) 
            {
                // if by sampling the brdf we also hit the dome light...
                if ((payload.instanceID == -1) && (sampledLightIDs[lid] == -1) && enableDomeSampling) {
                    // Case where we hit the background, and also previously sampled the background   
                    float w = power_heuristic(1.f, bsdfPDF, 1.f, lightPDFs[lid]);
                    float3 lightEmission = missColor(ray, envTex) * LP.domeLightIntensity * pow(2.f, LP.domeLightExposure);
                    float3 Li = (lightEmission * w) / bsdfPDF;
                    float dotNWi = max(dot(ray.direction, v_gz), 0.f);  // geometry term
                    if (dotNWi > 0.f) {
                        irradiance = irradiance + (bsdf * bsdfColor * Li);
                    }
                    hitLight = true;
                }
                // else if by sampling the brdf we also hit an area light
                else if (payload.instanceID != -1) {
                    int entityID = LP.surfaceInstanceToEntity.get(payload.instanceID, __LINE__);
                    bool visible = (entityID == sampledLightIDs[lid]);
                    // We hit the light we sampled previously
                    if (visible) {
                        int3 indices; float3 p, p_e1, p_e2; float3 lv_gz; 
                        float2 uv, uv_e1, uv_e2;
                        EntityStruct light_entity = LP.entities.get(sampledLightIDs[lid], __LINE__);
                        MeshStruct light_mesh = LP.meshes.get(light_entity.mesh_id, __LINE__);
                        LightStruct light_light = LP.lights.get(light_entity.light_id, __LINE__);
                        loadMeshTriIndices(light_entity.mesh_id, light_mesh.numTris, payload.primitiveID, indices);
                        loadMeshUVData(light_entity.mesh_id, light_mesh.numVerts, indices, payload.barycentrics, uv, uv_e1, uv_e2);

                        float dist = payload.tHit;
                        float dotNWi = max(dot(ray.direction, v_gz), 0.f); // geometry term

                        float3 lightEmission;
                        if (light_light.color_texture_id == -1) lightEmission = make_float3(light_light.r, light_light.g, light_light.b) * (light_light.intensity * pow(2.f, light_light.exposure));
                        else lightEmission = sampleTexture(light_light.color_texture_id, uv, make_float3(0.f, 0.f, 0.f)) * (light_light.intensity * pow(2.f, light_light.exposure));
                        lightEmission = lightEmission / pow(dist, light_light.falloff);

                        if (dotNWi > EPSILON) 
                        {
                            float w = power_heuristic(1.f, bsdfPDF, 1.f, lightPDFs[lid]);
                            float3 Li = (lightEmission * w) / bsdfPDF;
                            irradiance = irradiance + (bsdf * bsdfColor * Li);
                        }
                        hitLight = true;
                    }
                }
            }
        }
        irradiance = irradiance / float(numLightSamples);

        // Accumulate radiance (ie pathThroughput * irradiance), and update the path throughput using the sampled BRDF
        float3 contribution = pathThroughput * irradiance;
        illum = illum + contribution;
        pathThroughput = (pathThroughput * bsdf * bsdfColor) / bsdfPDF;
        if (bounce == 0) directIllum = illum;

        // Avoid double counting light sources by terminating here if we hit a light sampled thorugh NEE/MIS
        if (hitLight) break;

        // Russian Roulette
        // Randomly terminate a path with a probability inversely equal to the throughput
        float pmax = max(pathThroughput.x, max(pathThroughput.y, pathThroughput.z));
        if (lcg_randomf(rng) > pmax) {
            break;
        }

        // // Do path regularization to reduce fireflies
        // // Note, .35f was chosen emperically, but could be exposed as a parameter later on.
        // EDIT: finding that path regularization doesn't generalize well with transmissive objects...
        // if (sampledSpecular) {
        //     roughnessMinimum = min((roughnessMinimum + .35f), 1.f);
        // }

        // if the bounce count is less than the max bounce count, potentially add on radiance from the next hit location.
        ++bounce;     
        if (sampledBsdf == 0) diffuseBounce++;
        else specularBounce++;
    } while (diffuseBounce < LP.maxDiffuseBounceDepth && specularBounce < LP.maxSpecularBounceDepth);   

    // For segmentations, save heatmap metadata
    saveHeatmapRenderData(renderData, bounce, start_clock);

    // clamp out any extreme fireflies
    glm::vec3 gillum = vec3(illum.x, illum.y, illum.z);
    glm::vec3 dillum = vec3(directIllum.x, directIllum.y, directIllum.z);
    glm::vec3 iillum = gillum - dillum;

    // For segmentations, indirect/direct lighting metadata extraction
    // float3 aovGIllum = aovIllum;
    // aovIndirectIllum = aovGIllum - aovDirectIllum;
    // saveLightingIrradianceRenderData(renderData, bounce, aovDirectIllum, aovIndirectIllum, rdSampledBsdf);

    if (LP.indirectClamp > 0.f)
        iillum = clamp(iillum, vec3(0.f), vec3(LP.indirectClamp));
    if (LP.directClamp > 0.f)
        dillum = clamp(dillum, vec3(0.f), vec3(LP.directClamp));

    gillum = dillum + iillum;

    // just in case we get inf's or nans, remove them.
    if (glm::any(glm::isnan(gillum))) gillum = vec3(0.f);
    if (glm::any(glm::isinf(gillum))) gillum = vec3(0.f);
    illum = make_float3(gillum.r, gillum.g, gillum.b);

    // accumulate the illumination from this sample into what will be an average illumination from all samples in this pixel
    accum_illum = illum;

    /* Write to AOVs, progressively refining results */
    auto fbOfs = pixelID.x+LP.frameSize.x * ((LP.frameSize.y - 1) -  pixelID.y);
    float4* accumPtr = (float4*) LP.accumPtr;
    float4* fbPtr = (float4*) LP.frameBuffer;
    float4* normalPtr = (float4*) LP.normalBuffer;
    float4* albedoPtr = (float4*) LP.albedoBuffer;

    float4 prev_color = accumPtr[fbOfs];
    float4 prev_normal = normalPtr[fbOfs];
    float4 prev_albedo = albedoPtr[fbOfs];
    float4 accum_color;

    if (LP.renderDataMode == RenderDataFlags::NONE) 
    {
        accum_color = make_float4((accum_illum + float(LP.frameID) * make_float3(prev_color)) / float(LP.frameID + 1), 1.0f);
    }
    else {
        // Override framebuffer output if user requested to render metadata
        accum_illum = make_float3(renderData.x, renderData.y, renderData.z);
        accum_color = make_float4((accum_illum + float(LP.frameID) * make_float3(prev_color)) / float(LP.frameID + 1), 1.0f);
    }
    
    
    // compute screen space normal / albedo
    vec4 oldAlbedo = make_vec4(prev_albedo);
    vec4 oldNormal = make_vec4(prev_normal);
    if (any(isnan(oldAlbedo))) oldAlbedo = vec4(0.f);
    if (any(isnan(oldNormal))) oldNormal = vec4(0.f);
    vec4 newAlbedo = vec4(primaryAlbedo.x, primaryAlbedo.y, primaryAlbedo.z, 1.f);
    vec4 accumAlbedo = (newAlbedo + float(LP.frameID) * oldAlbedo) / float(LP.frameID + 1);
    vec4 newNormal = vec4(make_vec3(primaryNormal), 1.f);
    if (!all(equal(make_vec3(primaryNormal), vec3(0.f, 0.f, 0.f)))) {
        glm::quat r0 = glm::quat_cast(LP.viewT0);
        glm::quat r1 = glm::quat_cast(LP.viewT1);
        glm::quat rot = (glm::all(glm::equal(r0, r1))) ? r0 : glm::slerp(r0, r1, time);
        vec3 tmp = normalize(glm::mat3_cast(rot) * make_vec3(primaryNormal));
        tmp = normalize(vec3(LP.proj * vec4(tmp, 0.f)));
        newNormal = vec4(tmp, 1.f);
    }
    vec4 accumNormal = (newNormal + float(LP.frameID) * oldNormal) / float(LP.frameID + 1);

    // save data to frame buffers
    accumPtr[fbOfs] = accum_color;
    fbPtr[fbOfs] = accum_color;
    albedoPtr[fbOfs] = make_float4(accumAlbedo);
    normalPtr[fbOfs] = make_float4(accumNormal);    
}
