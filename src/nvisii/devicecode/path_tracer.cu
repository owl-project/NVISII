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

#include "nvisii/utilities/procedural_sky.h"

#include <glm/gtx/matrix_interpolation.hpp>

typedef owl::common::LCG<4> Random;

extern "C" __constant__ LaunchParams optixLaunchParams;

struct RayPayload {
    int instanceID = -1;
    int primitiveID = -1;
    float2 barycentrics;
    
    // for volumes
    float3 objectSpaceRayOrigin;
    float3 objectSpaceRayDirection;
    float t0;
    float t1;
    int eventID = -1;
    float3 gradient;
    float3 mp;
    float density;

    float tHit = -1.f;
    float localToWorld[12];
    float localToWorldT0[12];
    float localToWorldT1[12];
    LCGRand rng;
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
        GET(tex, cudaTextureObject_t, LP.textureObjects, LP.environmentMapID);
        return tex;
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
    
    optixGetObjectToWorldTransformMatrix(prd.localToWorld);
    
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
    int &event,
    bool debug = false
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
    x = x + t * w;
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

// bool debug = (prd.primitiveID == -2);
// if (debug) {
//     if (!  ((mn[0] < x[0]) && (x[0] < mx[0]) && 
//             (mn[1] < x[1]) && (x[1] < mx[1]) && 
//             (mn[2] < x[2]) && (x[2] < mx[2]))
//     ) {
//         printf("X");
//     } else {
//         printf("O");
//     }
// }
// if (debug) {
//     printf("\n");
// }

OPTIX_CLOSEST_HIT_PROGRAM(VolumeMesh)()
{   
    auto &LP = optixLaunchParams;
    RayPayload &prd = owl::getPRD<RayPayload>();
    // const auto &self = owl::getProgramData<VolumeGeomData>();
    // LCGRand rng = prd.rng;

    // // Load the volume we hit
    // GET(VolumeStruct volume, VolumeStruct, LP.volumes, self.volumeID);
    // uint8_t *hdl = (uint8_t*)LP.volumeHandles.get(self.volumeID, __LINE__).data;
    // const auto grid = reinterpret_cast<const nanovdb::FloatGrid*>(hdl);
    // const auto& tree = grid->tree();
    // auto acc = tree.getAccessor();

    // auto bbox = acc.root().bbox();    
    // auto mx = bbox.max();
    // auto mn = bbox.min();
    // glm::vec3 offset = glm::vec3(mn[0], mn[1], mn[2]) + 
    //             (glm::vec3(mx[0], mx[1], mx[2]) - 
    //             glm::vec3(mn[0], mn[1], mn[2])) * .5f;

    // float majorant_extinction = acc.root().valueMax();
    // float gradient_factor = volume.gradient_factor;
    // float linear_attenuation_unit = volume.scale;
    // float absorption = volume.absorption;
    // float scattering = volume.scattering;

    // vec3 x = make_vec3(prd.objectSpaceRayOrigin) + offset;
    // vec3 w = make_vec3(prd.objectSpaceRayDirection);

    // linear_attenuation_unit /= length(w);

    // // Move ray to volume boundary
    // float t0 = prd.t0, t1 = prd.t1;
    // x = x + t0 * w;
    // t1 = t1 - t0;
    // t0 = 0.f;

    // // Sample the free path distance to see if our ray makes it to the boundary
    // float t;
    // int event;
    // bool hitVolume = false;
    // #define MAX_NULL_COLLISIONS 10000
    // for (int dti = 0; dti < MAX_NULL_COLLISIONS; ++dti) {
    //     SampleDeltaTracking(rng, acc, majorant_extinction, linear_attenuation_unit, 
    //         absorption, scattering, x, w, t1, t, event);
    //     x = x + t * w;

    //     // The boundary was hit
    //     if (event == 0) {
    //         break;
    //     }

    //     // An absorption / emission event occurred
    //     if (event == 1) {
    //         hitVolume = true;
    //         break;
    //     }

    //     // A scattering event occurred
    //     if (event == 2) {
    //         hitVolume = true;
    //         break;
    //     }

    //     // A null collision occurred.
    //     if (event == 3) {
    //         // update boundary in relation to the new collision x, w does not change.
    //         t1 = t1 - t;
    //     }
    // }

    optixGetObjectToWorldTransformMatrix(prd.localToWorld);

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

OPTIX_CLOSEST_HIT_PROGRAM(VolumeShadowRay)()
{
    // auto &LP = optixLaunchParams;
    // const auto &self = owl::getProgramData<VolumeGeomData>();
    // RayPayload &prd = owl::getPRD<RayPayload>();
    // LCGRand rng = prd.rng;

    // GET(VolumeStruct volume, VolumeStruct, LP.volumes, self.volumeID);
    // uint8_t *hdl = (uint8_t*)LP.volumeHandles.get(self.volumeID, __LINE__).data;
    // const auto grid = reinterpret_cast<const nanovdb::FloatGrid*>(hdl);
    // const auto& tree = grid->tree();
    // auto acc = tree.getAccessor();

    // auto bbox = acc.root().bbox();    
    // auto mx = bbox.max();
    // auto mn = bbox.min();
    // glm::vec3 offset = glm::vec3(mn[0], mn[1], mn[2]) + 
    //             (glm::vec3(mx[0], mx[1], mx[2]) - 
    //             glm::vec3(mn[0], mn[1], mn[2])) * .5f;

    // float majorant_extinction = acc.root().valueMax();
    // float gradient_factor = volume.gradient_factor;
    // float linear_attenuation_unit = volume.scale;
    // float absorption = volume.absorption;
    // float scattering = volume.scattering;

    // vec3 x = make_vec3(prd.objectSpaceRayOrigin) + offset;
    // vec3 w = make_vec3(prd.objectSpaceRayDirection);

    // linear_attenuation_unit /= length(w);

    // // Move ray to volume boundary
    // float t0 = prd.t0, t1 = prd.t1;
    // x = x + t0 * w;
    // t1 = t1 - t0;
    // t0 = 0.f;

    // // Sample the free path distance to see if our ray makes it to the boundary
    // float t;
    // int event;
    // bool hitVolume = false;
    // #define MAX_NULL_COLLISIONS 10000
    // for (int dti = 0; dti < MAX_NULL_COLLISIONS; ++dti) {
    //     SampleDeltaTracking(rng, acc, majorant_extinction, linear_attenuation_unit, 
    //         absorption, scattering, x, w, t1, t, event);
    //     x = x + t * w;

    //     // The boundary was hit
    //     if (event == 0) {
    //         break;
    //     }

    //     // An absorption / emission event occurred
    //     if (event == 1) {
    //         hitVolume = true;
    //         break;
    //     }

    //     // A scattering event occurred
    //     if (event == 2) {
    //         hitVolume = true;
    //         break;
    //     }

    //     // A null collision occurred.
    //     if (event == 3) {
    //         // update boundary in relation to the new collision x, w does not change.
    //         t1 = t1 - t;
    //     }
    // }

    // if (!hitVolume) {
    //     prd.tHit = -1.f;
    // }
    // else {
    //     prd.instanceID = optixGetInstanceIndex();
    //     prd.eventID = event;
    //     prd.tHit = t;
    // }
}

OPTIX_INTERSECT_PROGRAM(VolumeIntersection)()
{
    // float old_tmax      = optixGetRayTmax();

    // const int primID = optixGetPrimitiveIndex();
    auto &LP = optixLaunchParams;
    const auto &self = owl::getProgramData<VolumeGeomData>();
    RayPayload &prd = owl::getPRD<RayPayload>();
    float3 origin = optixGetObjectRayOrigin();

    // note, this is _not_ normalized. Useful for computing world space tmin/mmax
    float3 direction = optixGetObjectRayDirection();

    float3 lb = make_float3(self.bbmin.x, self.bbmin.y, self.bbmin.z);
    float3 rt = make_float3(self.bbmax.x, self.bbmax.y, self.bbmax.z);

    // typical ray AABB intersection test
    float3 dirfrac;

    // direction is unit direction vector of ray
    dirfrac.x = 1.0f / direction.x;
    dirfrac.y = 1.0f / direction.y;
    dirfrac.z = 1.0f / direction.z;

    // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
    // origin is origin of ray
    float t1 = (lb.x - origin.x)*dirfrac.x;
    float t2 = (rt.x - origin.x)*dirfrac.x;
    float t3 = (lb.y - origin.y)*dirfrac.y;
    float t4 = (rt.y - origin.y)*dirfrac.y;
    float t5 = (lb.z - origin.z)*dirfrac.z;
    float t6 = (rt.z - origin.z)*dirfrac.z;

    float thit0 = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float thit1 = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
    if (thit1 < 0) { return; }

    // if tmin > tmax, ray doesn't intersect AABB
    if (thit0 >= thit1) { return; }

    // clip hit to near position
    thit0 = max(thit0, optixGetRayTmin());

    // Load the volume we hit
    GET(VolumeStruct volume, VolumeStruct, LP.volumes, self.volumeID);
    uint8_t *hdl = (uint8_t*)LP.volumeHandles.get(self.volumeID, __LINE__).data;
    const auto grid = reinterpret_cast<const nanovdb::FloatGrid*>(hdl);
    const auto& tree = grid->tree();
    auto acc = tree.getAccessor();
    auto nvdbSampler = nanovdb::SampleFromVoxels<nanovdb::DefaultReadAccessor<float>, 
        /*Interpolation Degree*/1, /*UseCache*/false>(acc);

    float majorant_extinction = acc.root().valueMax();
    float gradient_factor = volume.gradient_factor;
    float linear_attenuation_unit = volume.scale;
    float absorption = volume.absorption;
    float scattering = volume.scattering;

    auto bbox = acc.root().bbox();    
    auto mx = bbox.max();
    auto mn = bbox.min();
    float3 offset = make_float3(glm::vec3(mn[0], mn[1], mn[2]) + 
                (glm::vec3(mx[0], mx[1], mx[2]) - 
                glm::vec3(mn[0], mn[1], mn[2])) * .5f);

    // Sample the free path distance to see if our ray makes it to the boundary
    float t = thit0;
    int event;
    bool hitVolume = false;
    float unit = volume.scale / length(direction);
    #define MAX_NULL_COLLISIONS 1000
    for (int i = 0; i < MAX_NULL_COLLISIONS; ++i) {
        // Sample a distance
        t = t - (log(1.0f - lcg_randomf(prd.rng)) / majorant_extinction) * unit; 

        // A boundary has been hit, no intersection
        if (t >= thit1) return;

        // Update current position
        float3 x = offset + origin + t * direction;

        // Sample heterogeneous media
        float densityValue = nvdbSampler(nanovdb::Vec3f(x.x, x.y, x.z));

        float a = densityValue * absorption;
        float s = densityValue * scattering;
        float e = a + s;
        float n = majorant_extinction - e;

        a = a / majorant_extinction;
        s = s / majorant_extinction;
        n = n / majorant_extinction;

        float event = lcg_randomf(prd.rng);
        // An absorption/emission collision occured
        if (event < (a + s)) {
            if (optixReportIntersection(t, /* hit kind */ 0)) {
                auto g = nvdbSampler.gradient(nanovdb::Vec3f(x.x, x.y, x.z)); 
                prd.objectSpaceRayOrigin = origin;
                prd.objectSpaceRayDirection = direction;
                prd.eventID = (event < a) ? 1 : 2;
                prd.instanceID = optixGetInstanceIndex();
                prd.tHit = t;
                prd.mp = x - offset; // not super confident about this offset...
                prd.gradient = make_float3(g[0], g[1], g[2]);// TEMPORARY FOR BUNNY
                prd.density = densityValue;
            }
            return;
        }

        // A null collision occurred
        else {
            event = 3;
            continue;    	
        }
    }
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
    GET(camera, CameraStruct, LP.cameras, cameraEntity.camera_id);
    GET(transform, TransformStruct, LP.transforms, cameraEntity.transform_id);
    return true;
}

inline __device__ 
float3 sampleTexture(int32_t textureId, float2 texCoord, float3 defaultVal) {
    auto &LP = optixLaunchParams;
    if (textureId < 0 || textureId >= (LP.textures.count + LP.materials.count * NUM_MAT_PARAMS)) return defaultVal;
    GET(cudaTextureObject_t tex, cudaTextureObject_t, LP.textureObjects, textureId);
    if (!tex) return defaultVal;
    GET(TextureStruct texInfo, TextureStruct, LP.textures, textureId);
    texCoord.x = texCoord.x / texInfo.scale.x;
    texCoord.y = texCoord.y / texInfo.scale.y;
    return make_float3(tex2D<float4>(tex, texCoord.x, texCoord.y));
}

inline __device__ 
float sampleTexture(int32_t textureId, float2 texCoord, int8_t channel, float defaultVal) {
    auto &LP = optixLaunchParams;
    if (textureId < 0 || textureId >= (LP.textures.count + LP.materials.count * NUM_MAT_PARAMS)) return defaultVal;
    GET(cudaTextureObject_t tex, cudaTextureObject_t, LP.textureObjects, textureId);
    if (!tex) return defaultVal;
    GET(TextureStruct texInfo, TextureStruct, LP.textures, textureId);
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
    GET(Buffer<int3> indices, Buffer<int3>, LP.indexLists, meshID);
    GET(triIndices, int3, indices, primitiveID);
}

__device__
void loadMeshVertexData(int meshID, int numVertices, int3 indices, float2 barycentrics, float3 &position, float3 &geometricNormal)
{
    auto &LP = optixLaunchParams;
    GET(Buffer<float3> vertices, Buffer<float3>, LP.vertexLists, meshID);
    GET(const float3 A, float3, vertices, indices.x);
    GET(const float3 B, float3, vertices, indices.y);
    GET(const float3 C, float3, vertices, indices.z);
    position = A * (1.f - (barycentrics.x + barycentrics.y)) + B * barycentrics.x + C * barycentrics.y;
    geometricNormal = normalize(cross(B-A,C-A));
}

__device__
void loadMeshUVData(int meshID, int numTexCoords, int3 indices, float2 barycentrics, float2 &uv)
{
    auto &LP = optixLaunchParams;
    GET(Buffer<float2> texCoords, Buffer<float2>, LP.texCoordLists, meshID);
    GET(const float2 A, float2, texCoords, indices.x);
    GET(const float2 B, float2, texCoords, indices.y);
    GET(const float2 C, float2, texCoords, indices.z);
    uv = A * (1.f - (barycentrics.x + barycentrics.y)) + B * barycentrics.x + C * barycentrics.y;
}

__device__
void loadMeshNormalData(int meshID, int numNormals, int3 indices, float2 barycentrics, float3 &normal)
{
    auto &LP = optixLaunchParams;
    GET(Buffer<float4> normals, Buffer<float4>, LP.normalLists, meshID);
    GET(const float4 A, float4, normals, indices.x);
    GET(const float4 B, float4, normals, indices.y);
    GET(const float4 C, float4, normals, indices.z);
    normal = make_float3(A) * (1.f - (barycentrics.x + barycentrics.y)) + make_float3(B) * barycentrics.x + make_float3(C) * barycentrics.y;
}

__device__
void loadMeshTangentData(int meshID, int numTangents, int3 indices, float2 barycentrics, float3 &tangent)
{
    auto &LP = optixLaunchParams;
    GET(Buffer<float4> tangents, Buffer<float4>, LP.tangentLists, meshID);
    GET(const float4 A, float4, tangents, indices.x);
    GET(const float4 B, float4, tangents, indices.y);
    GET(const float4 C, float4, tangents, indices.z);
    tangent = make_float3(A) * (1.f - (barycentrics.x + barycentrics.y)) + make_float3(B) * barycentrics.x + make_float3(C) * barycentrics.y;
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
owl::Ray generateRay(const CameraStruct &camera, const TransformStruct &transform, int2 pixelID, float2 frameSize, LCGRand &rng, float time)
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

    vec2 inUV = (vec2(pixelID.x, pixelID.y) + aa) / make_vec2(frameSize);
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
void saveDeviceAssignment(
    float3 &renderData, 
    int bounce,
    uint32_t deviceIndex
)
{
    auto &LP = optixLaunchParams;
    if (LP.renderDataMode != RenderDataFlags::DEVICE_ID) return;
    renderData = make_float3(deviceIndex);
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

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
    const RayGenData &self = owl::getProgramData<RayGenData>();
    auto &LP = optixLaunchParams;
    auto launchIndex = optixGetLaunchIndex().x;
    auto launchDim = optixGetLaunchDimensions().x;
    auto pixelID = make_int2(launchIndex % LP.frameSize.x, launchIndex / LP.frameSize.x);

    // Terminate thread if current pixel not assigned to this device
    GET(float start, float, LP.assignmentBuffer, self.deviceIndex);
    GET(float stop, float, LP.assignmentBuffer, self.deviceIndex + 1);
    start *= (LP.frameSize.x * LP.frameSize.y);
    stop *= (LP.frameSize.x * LP.frameSize.y);

    // if (launchIndex == 0) {
    //     printf("device %d start %f stop %f\n", self.deviceIndex, start, stop);
    // }

    if( pixelID.x > LP.frameSize.x-1 || pixelID.y > LP.frameSize.y-1 ) return;
    if( (launchIndex < start) || (stop <= launchIndex) ) return;
    // if (self.deviceIndex == 1) return;
    
    cudaTextureObject_t envTex = getEnvironmentTexture();    
    bool debug = (pixelID.x == int(LP.frameSize.x / 2) && pixelID.y == int(LP.frameSize.y / 2));
    float tmax = 1e20f; //todo: customize depending on scene bounds //glm::distance(LP.sceneBBMin, LP.sceneBBMax);

    auto dims = ivec2(LP.frameSize.x, LP.frameSize.x);
    uint64_t start_clock = clock();
    int numLights = LP.numLightEntities;
    int numLightSamples = LP.numLightSamples;
    bool enableDomeSampling = LP.enableDomeSampling;
    
    LCGRand rng = get_rng(LP.frameID + LP.seed * 10007, make_uint2(pixelID.x, pixelID.y), make_uint2(dims.x, dims.y));
    float time = sampleTime(lcg_randomf(rng));

    // If no camera is in use, just display some random noise...
    owl::Ray ray;
    EntityStruct    camera_entity;
    TransformStruct camera_transform;
    CameraStruct    camera;
    if (!loadCamera(camera_entity, camera, camera_transform)) {
        auto fbOfs = pixelID.x+LP.frameSize.x * ((LP.frameSize.y - 1) -  pixelID.y);
        LP.frameBuffer[fbOfs] = vec4(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng), 1.f);
        return;
    }
    
    // Trace an initial ray through the scene
    ray = generateRay(camera, camera_transform, pixelID, make_float2(LP.frameSize), rng, time);
    ray.tmax = tmax;

    float3 accum_illum = make_float3(0.f);
    float3 pathThroughput = make_float3(1.f);
    float3 renderData = make_float3(0.f);
    float3 primaryAlbedo = make_float3(0.f);
    float3 primaryNormal = make_float3(0.f);
    initializeRenderData(renderData);

    uint8_t depth = 0;
    uint8_t diffuseDepth = 0;
    uint8_t glossyDepth = 0;
    uint8_t transparencyDepth = 0;
    uint8_t transmissionDepth = 0;
    uint8_t volumeDepth = 0;
    int sampledBsdf = -1;
    bool useBRDF = true;

    // direct here is used for final image clamping
    float3 directIllum = make_float3(0.f);
    float3 illum = make_float3(0.f);
    
    RayPayload payload;
    payload.tHit = -1.f;
    ray.time = time;
    ray.visibilityMask = ENTITY_VISIBILITY_CAMERA_RAYS;
    owl::traceRay(  /*accel to trace against*/ LP.IAS,
                    /*the ray to trace*/ ray,
                    /*prd*/ payload,
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT);
    
    // Shade each hit point on a path using NEE with MIS
    do {     
        float alpha = 0.f;
        
        // If ray misses, terminate the ray
        if (payload.tHit <= 0.f) {
            // Compute lighting from environment
            if (depth == 0) {
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
            saveMissRenderData(renderData, depth, mvec);
            break;
        }


        // Load the object we hit.
        GET(int entityID, int, LP.instanceToEntity, payload.instanceID);
        GET(EntityStruct entity, EntityStruct, LP.entities, entityID);
        GET(TransformStruct transform, TransformStruct, LP.transforms, entity.transform_id);

        bool isVolume = (entity.volume_id != -1);
        MeshStruct mesh;  
        VolumeStruct volume;  
        if (!isVolume) { GET(mesh, MeshStruct, LP.meshes, entity.mesh_id); }
        else { GET(volume, VolumeStruct, LP.volumes, entity.volume_id); }
        
        // Set new outgoing light direction and hit position.
        const float3 w_o = -ray.direction;
        float3 hit_p = ray.origin + payload.tHit * ray.direction;

        // Load geometry data for the hit object
        float3 mp, p, v_x, v_y, v_z, v_gz, v_bz; 
        float2 uv; 
        float3 diffuseMotion;
        if (isVolume) {
            v_x = v_y = make_float3(0.f); // Perhaps I could use divergence / curl here?
            v_z = v_gz = normalize(payload.gradient);
            if (any(isnan(make_vec3(v_z)))) v_z = v_gz = make_float3(0.f);
            mp = payload.mp;
            uv = make_float2(payload.density, length(payload.gradient));
        }
        else {
            int3 indices;
            loadMeshTriIndices(entity.mesh_id, mesh.numTris, payload.primitiveID, indices);
            loadMeshVertexData(entity.mesh_id, mesh.numVerts, indices, payload.barycentrics, mp, v_gz);
            loadMeshUVData(entity.mesh_id, mesh.numVerts, indices, payload.barycentrics, uv);
            loadMeshNormalData(entity.mesh_id, mesh.numVerts, indices, payload.barycentrics, v_z);
            loadMeshTangentData(entity.mesh_id, mesh.numVerts, indices, payload.barycentrics, v_x);
        }

        // Load material data for the hit object
        DisneyMaterial mat; MaterialStruct entityMaterial;
        if (entity.material_id >= 0 && entity.material_id < LP.materials.count) {
            GET(entityMaterial, MaterialStruct, LP.materials, entity.material_id);
            loadDisneyMaterial(entityMaterial, uv, mat, MIN_ROUGHNESS);
        }
      
        // Transform geometry data into world space
        {
            glm::mat4 xfm = to_mat4(payload.localToWorld);
            p = make_float3(xfm * make_vec4(mp, 1.0f));
            hit_p = p;
            glm::mat3 nxfm = transpose(glm::inverse(glm::mat3(xfm)));
            v_gz = make_float3(normalize(nxfm * make_vec3(v_gz)));
            v_z = make_float3(normalize(nxfm * make_vec3(v_z)));
            v_x = make_float3(normalize(nxfm * make_vec3(v_x)));
            v_y = cross(v_z, v_x);
            v_x = cross(v_y, v_z);

            if (LP.renderDataMode != RenderDataFlags::NONE) {
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
                dN = sampleTexture(entityMaterial.normal_map_texture_id, uv, make_float3(0.5f, .5f, 0.f));
                GET(TextureStruct tex, TextureStruct, LP.textures, entityMaterial.normal_map_texture_id);
                // For DirectX normal maps. 
                // if (!tex.rightHanded) {
                //     dN.y = 1.f - dN.y;
                // }
            }

            dN = normalize( (dN * make_float3(2.0f)) - make_float3(1.f) );   
            v_z = make_float3(tbn * make_vec3(dN));

            // make sure geometric and shading normal face the same direction.
            if (dot(v_z, v_gz) < 0.f) {
                v_z = -v_z;
            }

            v_bz = v_z;
        }

        // // TEMP CODE
        // auto fbOfs = pixelID.x+LP.frameSize.x * ((LP.frameSize.y - 1) -  pixelID.y);
        // LP.frameBuffer[fbOfs] = make_vec4(v_z, 1.f);
        // return;

        // If we didn't hit glass, flip the surface normal to face forward.
        if ((mat.specular_transmission == 0.f) && (entity.light_id == -1)) {
            if (dot(w_o, v_gz) < 0.f) {
                v_z = -v_z;
                v_gz = -v_gz;
            }

            // compute bent normal
            float3 r = reflect(-w_o, v_z);
            float a = dot(v_gz, r);
            v_bz = v_z;
            if (a < 0.f) {
                float b = max(0.001f, dot(v_z, v_gz));
                v_bz = normalize(w_o + normalize(r - v_z * a / b));
            }
        }

        if (any(isnan(make_vec3(v_z)))) {
            // Since gradient can be 0, normalizing can cause nans. 
            // Doesn't really matter, since 0 length normals result in a phase function (no surface present).
            v_z = v_x = v_y = make_float3(0.f);
        }

        // For segmentations, save geometric metadata
        saveGeometricRenderData(renderData, depth, payload.tHit, hit_p, v_z, w_o, uv, entityID, diffuseMotion, time, mat);
        if (depth == 0) {
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
                // ray.visibilityMask reuses the last visibility mask here
                owl::traceRay( LP.IAS, ray, payload, OPTIX_RAY_FLAG_DISABLE_ANYHIT);                
                ++depth;     
                transparencyDepth++;
                continue;
            }
        }

        // If the entity we hit is a light, terminate the path.
        // Note that NEE/MIS will also potentially terminate the path, preventing double-counting.
        // todo: account for volumetric emission here...
        if (entity.light_id >= 0 && entity.light_id < LP.lights.count) {
            float dotNWi = max(dot(ray.direction, v_z), 0.f);
            if ((dotNWi > EPSILON) && (depth != 0)) break;

            GET(LightStruct entityLight, LightStruct, LP.lights, entity.light_id);
            float3 lightEmission;
            if (entityLight.color_texture_id == -1) lightEmission = make_float3(entityLight.r, entityLight.g, entityLight.b);
            else lightEmission = sampleTexture(entityLight.color_texture_id, uv, make_float3(0.f, 0.f, 0.f));
            float dist = payload.tHit;
            lightEmission = (lightEmission * entityLight.intensity);
            if (depth != 0) lightEmission = (lightEmission * pow(2.f, entityLight.exposure)) / max((dist * dist), 1.f);
            float3 contribution = pathThroughput * lightEmission;
            illum = illum + contribution;
            if (depth == 0) directIllum = illum;
            break;
        }

        // Next, we'll be sampling direct light sources
        int32_t sampledLightID = -2;
        float lightPDF = 0.f;
        float3 irradiance = make_float3(0.f);

        // If we hit a volume, use hybrid scattering to determine whether or not to use a BRDF or a phase function.
        if (isVolume) {
            float opacity = mat.alpha; // would otherwise be sampled from a transfer function
            float grad_len = uv.y;
            float p_brdf = opacity * (1.f - exp(-25.f * pow(volume.gradient_factor, 3.f) * grad_len));
            float pdf;
            float rand_brdf = lcg_randomf(rng);
            
            if (rand_brdf < p_brdf) {
                useBRDF = true;
            } else {
                useBRDF = false;
            }
        }

        // First, sample the BRDF / phase function so that we can use the sampled direction for MIS
        float3 w_i;
        float bsdfPDF;
        float3 bsdf;
        if (useBRDF) {
            sample_disney_brdf(
                mat, rng, v_gz, v_z, v_bz, v_x, v_y, w_o, // inputs
                w_i, bsdfPDF, sampledBsdf, bsdf);         // outputs
        } else {
            /* a scatter event occurred */
            if (payload.eventID == 2) {
                // currently isotropic. Todo: implement henyey greenstien...
                float rand1 = lcg_randomf(rng);
                float rand2 = lcg_randomf(rng);

                // Sample isotropic phase function to get new ray direction           
                float phi = 2.0f * M_PI * rand1;
                float cos_theta = 1.0f - 2.0f * rand2;
                float sin_theta = sqrt (1.0f - cos_theta * cos_theta);
                
                bsdfPDF = 1.f / (4.0 * M_PI);
                bsdf = make_float3(1.f / (4.0 * M_PI));
                w_i = make_float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
            } 

            /* An absorption / emission event occurred */ 
            if (payload.eventID == 1) {
                bsdfPDF = 1.f / (4.0 * M_PI);
                bsdf = make_float3(1.f / (4.0 * M_PI));
                w_i = -w_o;
            }

            // For all events, modify throughput by base color.
            bsdf = bsdf * mat.base_color;
        }

        // At this point, if we are refracting and we ran out of transmission bounces, skip forward.
        // This avoids creating black regions on glass objects due to bounce limits
        if (sampledBsdf == DISNEY_TRANSMISSION_BRDF && transmissionDepth >= LP.maxTransmissionDepth) {
            ray.origin = ray.origin + ray.direction * (payload.tHit + EPSILON);
            payload.tHit = -1.f;
            ray.time = time;
            // ray.visibilityMask reuses the last visibility mask here
            owl::traceRay( LP.IAS, ray, payload, OPTIX_RAY_FLAG_DISABLE_ANYHIT);
            
            // Count this as a "transparent" bounce.
            ++depth;     
            transparencyDepth++;
            continue;
        }

        // Next, sample the light source by importance sampling the light
        const uint32_t occlusion_flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;
        
        uint32_t randmax = (enableDomeSampling) ? numLights + 1 : numLights;
        uint32_t randomID = uint32_t(min(lcg_randomf(rng) * randmax, float(randmax-1)));
        float dotNWi  = 0.f;
        float3 l_bsdf = make_float3(0.f);
        float3 lightEmission = make_float3(0.f);
        float3 lightDir = make_float3(0.f);
        float lightDistance = 1e20f;
        float falloff = 2.0f;
        int numTris = 0;

        // sample background
        if (randomID == numLights) {
            sampledLightID = -1;
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
                lightPDF = row_pdf * col_pdf * invjacobian;
            } 
            else 
            {            
                glm::mat3 tbn;
                tbn = glm::column(tbn, 0, make_vec3(v_x) );
                tbn = glm::column(tbn, 1, make_vec3(v_y) );
                tbn = glm::column(tbn, 2, make_vec3(v_z) );            
                const float3 hemi_dir = (cos_sample_hemisphere(make_float2(lcg_randomf(rng), lcg_randomf(rng))));
                lightDir = make_float3(tbn * make_vec3(hemi_dir));
                lightPDF = 1.f / float(2.0 * M_PI);
            }

            numTris = 1.f;
            lightEmission = (missColor(lightDir, envTex) * LP.domeLightIntensity * pow(2.f, LP.domeLightExposure));
        }
        // sample light sources
        else 
        {
            if (numLights == 0) continue;
            GET( sampledLightID, int, LP.lightEntities, randomID );
            GET( EntityStruct light_entity, EntityStruct, LP.entities, sampledLightID );
            GET( LightStruct light_light, LightStruct, LP.lights, light_entity.light_id );
            GET( TransformStruct transform, TransformStruct, LP.transforms, light_entity.transform_id );
            GET( MeshStruct mesh, MeshStruct, LP.meshes, light_entity.mesh_id );
            uint32_t random_tri_id = uint32_t(min(lcg_randomf(rng) * mesh.numTris, float(mesh.numTris - 1)));
            GET( Buffer<int3> indices, Buffer<int3>, LP.indexLists, light_entity.mesh_id );
            GET( Buffer<float3> vertices, Buffer<float3>, LP.vertexLists, light_entity.mesh_id );
            GET( Buffer<float4> normals, Buffer<float4>, LP.normalLists, light_entity.mesh_id );
            GET( Buffer<float2> texCoords, Buffer<float2>, LP.texCoordLists, light_entity.mesh_id );
            GET( int3 triIndex, int3, indices, random_tri_id );
            
            // Sample the light to compute an incident light ray to this point
            auto &ltw = transform.localToWorld;
            float3 dir; float2 uv;
            float3 pos = hit_p;

            GET(float3 n1, float3, normals, triIndex.x );
            GET(float3 n2, float3, normals, triIndex.y );
            GET(float3 n3, float3, normals, triIndex.z );
            GET(float3 v1, float3, vertices, triIndex.x );
            GET(float3 v2, float3, vertices, triIndex.y );
            GET(float3 v3, float3, vertices, triIndex.z );
            GET(float2 uv1, float2, texCoords, triIndex.x );
            GET(float2 uv2, float2, texCoords, triIndex.y );
            GET(float2 uv3, float2, texCoords, triIndex.z );

            // Might be a bug here with normal transform...
            n1 = make_float3(ltw * make_float4(n1, 0.0f));
            n2 = make_float3(ltw * make_float4(n2, 0.0f));
            n3 = make_float3(ltw * make_float4(n3, 0.0f));
            v1 = make_float3(ltw * make_float4(v1, 1.0f));
            v2 = make_float3(ltw * make_float4(v2, 1.0f));
            v3 = make_float3(ltw * make_float4(v3, 1.0f));
            sampleTriangle(pos, n1, n2, n3, v1, v2, v3, uv1, uv2, uv3, 
                lcg_randomf(rng), lcg_randomf(rng), dir, lightDistance, lightPDF, uv, 
                /*double_sided*/ false, /*use surface area*/ light_light.use_surface_area);
            
            falloff = light_light.falloff;
            numTris = mesh.numTris;
            lightDir = make_float3(dir.x, dir.y, dir.z);
            if (light_light.color_texture_id == -1) lightEmission = make_float3(light_light.r, light_light.g, light_light.b) * (light_light.intensity * pow(2.f, light_light.exposure));
            else lightEmission = sampleTexture(light_light.color_texture_id, uv, make_float3(0.f, 0.f, 0.f)) * (light_light.intensity * pow(2.f, light_light.exposure));
        }

        if (useBRDF) {
            disney_brdf(
                mat, v_gz, v_z, v_bz, v_x, v_y,
                w_o, lightDir, normalize(w_o + lightDir), l_bsdf
            );
            dotNWi = max(dot(lightDir, v_z), 0.f);

            // auto fbOfs = pixelID.x+LP.frameSize.x * ((LP.frameSize.y - 1) -  pixelID.y);
            // LP.frameBuffer[fbOfs] = vec4(l_bsdf.x, l_bsdf.y, l_bsdf.z, 1.f);
            // return;
        } else {
            // currently isotropic. Todo: implement henyey greenstien...
            l_bsdf = make_float3(1.f / (4.0 * M_PI)) * mat.base_color;
            dotNWi = 1.f; // no geom term for phase function
        }
        lightPDF *= (1.f / float(numLights + 1.f)) * (1.f / float(numTris));
        if ((lightPDF > 0.0) && (dotNWi > EPSILON)) {
            RayPayload payload; payload.instanceID = -2;
            RayPayload volPayload = payload;
            owl::RayT</*type*/1, /*prd*/1> ray; // shadow ray
            ray.tmin = EPSILON * 10.f; ray.tmax = lightDistance + EPSILON; // needs to be distance to light, else anyhit logic breaks.
            ray.origin = hit_p; ray.direction = lightDir;
            ray.time = time;
            ray.visibilityMask = ENTITY_VISIBILITY_SHADOW_RAYS;
            owl::traceRay( LP.IAS, ray, payload, occlusion_flags);
            ray.tmax = (payload.instanceID == -2) ? ray.tmax : payload.tHit;
            bool visible;
            if (randomID == numLights) {
                //  If we sampled the dome light, just check to see if we hit anything
                visible = (payload.instanceID == -2);
            } else {
                // If we sampled a light source, then check to see if we hit something other than the light
                int surfEntity;
                if (payload.instanceID == -2) surfEntity = -1;
                else { GET(surfEntity, int, LP.instanceToEntity, payload.instanceID); }
                visible = (payload.instanceID == -2 || surfEntity == sampledLightID);
            }
            if (visible) {
                if (randomID != numLights) lightEmission = lightEmission / max(pow(payload.tHit, falloff),1.f);
                float w = power_heuristic(1.f, lightPDF, 1.f, bsdfPDF);
                float3 Li = (lightEmission * w) / lightPDF;
                irradiance = irradiance + (l_bsdf * Li);
            }
        }

        // For segmentations, save lighting metadata
        saveLightingColorRenderData(renderData, depth, v_z, w_o, w_i, mat);

        // Terminate the path if the bsdf probability is impossible, or if the bsdf filters out all light
        if (bsdfPDF < EPSILON || all_zero(bsdf)) {
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
        if (isVolume) ray.visibilityMask = ENTITY_VISIBILITY_VOLUME_SCATTER_RAYS;
        else if (sampledBsdf == DISNEY_TRANSMISSION_BRDF) ray.visibilityMask = ENTITY_VISIBILITY_TRANSMISSION_RAYS;
        else if (sampledBsdf == DISNEY_DIFFUSE_BRDF) ray.visibilityMask = ENTITY_VISIBILITY_DIFFUSE_RAYS;
        else if (sampledBsdf == DISNEY_GLOSSY_BRDF) ray.visibilityMask = ENTITY_VISIBILITY_GLOSSY_RAYS;
        else if (sampledBsdf == DISNEY_CLEARCOAT_BRDF) ray.visibilityMask = ENTITY_VISIBILITY_GLOSSY_RAYS;
        owl::traceRay(LP.IAS, ray, payload, OPTIX_RAY_FLAG_DISABLE_ANYHIT);

        // Check if we hit any of the previously sampled lights
        bool hitLight = false;
        if (lightPDF > EPSILON) 
        {
            float dotNWi = (useBRDF) ? max(dot(ray.direction, v_gz), 0.f) : 1.f;  // geometry term

            // if by sampling the brdf we also hit the dome light...
            if ((payload.instanceID == -1) && (sampledLightID == -1) && enableDomeSampling) {
                // Case where we hit the background, and also previously sampled the background   
                float w = power_heuristic(1.f, bsdfPDF, 1.f, lightPDF);
                float3 lightEmission = missColor(ray, envTex) * LP.domeLightIntensity * pow(2.f, LP.domeLightExposure);
                float3 Li = (lightEmission * w) / bsdfPDF;
                
                if (dotNWi > 0.f) {
                    irradiance = irradiance + (bsdf * Li);
                }
                hitLight = true;
            }
            // else if by sampling the brdf we also hit an area light
            // TODO: consider hitting emissive voxels?
            else if (payload.instanceID != -1) {
                GET(int entityID, int, LP.instanceToEntity, payload.instanceID);
                bool visible = (entityID == sampledLightID);
                // We hit the light we sampled previously
                if (visible) {
                    int3 indices; float3 p; float3 lv_gz; float2 uv;
                    GET(EntityStruct light_entity, EntityStruct, LP.entities, sampledLightID);
                    GET(MeshStruct light_mesh, MeshStruct, LP.meshes, light_entity.mesh_id);
                    GET(LightStruct light_light, LightStruct, LP.lights, light_entity.light_id);
                    loadMeshTriIndices(light_entity.mesh_id, light_mesh.numTris, payload.primitiveID, indices);
                    loadMeshUVData(light_entity.mesh_id, light_mesh.numVerts, indices, payload.barycentrics, uv);

                    float dist = payload.tHit;
                    
                    float3 lightEmission;
                    if (light_light.color_texture_id == -1) lightEmission = make_float3(light_light.r, light_light.g, light_light.b) * (light_light.intensity * pow(2.f, light_light.exposure));
                    else lightEmission = sampleTexture(light_light.color_texture_id, uv, make_float3(0.f, 0.f, 0.f)) * (light_light.intensity * pow(2.f, light_light.exposure));
                    lightEmission = lightEmission / max(pow(dist, light_light.falloff), 1.f);

                    if (dotNWi > EPSILON) 
                    {
                        float w = power_heuristic(1.f, bsdfPDF, 1.f, lightPDF);
                        float3 Li = (lightEmission * w) / bsdfPDF;
                        irradiance = irradiance + (bsdf * Li);
                    }
                    hitLight = true;
                }
            }
        }
        
        // Accumulate radiance (ie pathThroughput * irradiance), and update the path throughput using the sampled BRDF
        float3 contribution = pathThroughput * irradiance;
        illum = illum + contribution;
        pathThroughput = (pathThroughput * bsdf) / bsdfPDF;
        if (depth == 0) directIllum = illum;

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
        ++depth;     
        if (!useBRDF) volumeDepth++;
        else if (sampledBsdf == DISNEY_DIFFUSE_BRDF) diffuseDepth++;
        else if (sampledBsdf == DISNEY_GLOSSY_BRDF) glossyDepth++;
        else if (sampledBsdf == DISNEY_CLEARCOAT_BRDF) glossyDepth++;
        else if (sampledBsdf == DISNEY_TRANSMISSION_BRDF) transmissionDepth++;
        // transparency depth handled earlier
        
        // for transmission, once we hit the limit, we'll stop refracting instead 
        // of terminating, just so that we don't get black regions in our glass
        if (transmissionDepth >= LP.maxTransmissionDepth) continue;
    } while (
        // Terminate the path if the sampled BRDF's corresponding bounce depth exceeds the max bounce for that bounce type minus the overall path depth.
        // This prevents long tails that can otherwise occur from mixing BRDF events
        (!(sampledBsdf == DISNEY_DIFFUSE_BRDF && diffuseDepth > (LP.maxDiffuseDepth - (depth - 1)))) &&
        (!(sampledBsdf == DISNEY_GLOSSY_BRDF && glossyDepth > LP.maxGlossyDepth - (depth - 1)) ) &&
        (!(sampledBsdf == DISNEY_CLEARCOAT_BRDF && glossyDepth > LP.maxGlossyDepth - (depth - 1)) ) &&
        (!(useBRDF == false && volumeDepth > LP.maxVolumeDepth - (depth - 1))) &&
        (!(transparencyDepth > LP.maxTransparencyDepth - (depth - 1)))
        // (!(sampledBsdf == DISNEY_TRANSMISSION_BRDF && transmissionDepth < LP.maxTransmissionDepth - (depth - 1)) ) && // see comment above
    );   

    // For segmentations, save heatmap metadata
    saveHeatmapRenderData(renderData, depth, start_clock);

    // Device assignment data
    saveDeviceAssignment(renderData, depth, self.deviceIndex);

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
