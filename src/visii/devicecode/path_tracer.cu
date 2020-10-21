#define GLM_FORCE_CUDA
#define PMEVENT( x ) asm volatile("pmevent " #x ";")

#include <stdint.h>
template<class T>
__device__
T read(T* buf, size_t addr, size_t size, uint32_t line) {
    if (buf == nullptr) {::printf("Device Side Error on Line %d: buffer was nullptr.\n", line); asm("trap;");}
    if (addr >= size) {::printf("Device Side Error on Line %d: out of bounds access (addr: %d, size %d).\n", line, uint32_t(addr), uint32_t(size)); asm("trap;");}
    return buf[addr];
}

#include "launch_params.h"
#include "types.h"
#include "path_tracer.h"
#include "disney_bsdf.h"
#include "lights.h"
#include "math.h"
#include <optix_device.h>
#include <owl/common/math/random.h>

#include "visii/utilities/procedural_sky.h"

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
        return read((cudaTextureObject_t*)LP.textureObjects.data, LP.environmentMapID, LP.textureObjects.count, __LINE__);
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

inline __device__
bool loadCamera(EntityStruct &cameraEntity, CameraStruct &camera, TransformStruct &transform)
{
    auto &LP = optixLaunchParams;
    cameraEntity = LP.cameraEntity;
    if (!cameraEntity.initialized) return false;
    if ((cameraEntity.transform_id < 0) || (cameraEntity.transform_id >= LP.transforms.count)) return false;
    if ((cameraEntity.camera_id < 0) || (cameraEntity.camera_id >= LP.cameras.count)) return false;
    camera = read((CameraStruct*)LP.cameras.data, cameraEntity.camera_id, LP.cameras.count, __LINE__);
    transform = read((TransformStruct*)LP.transforms.data, cameraEntity.transform_id, LP.transforms.count, __LINE__);
    return true;
}

inline __device__ 
float3 sampleTexture(int32_t textureId, float2 texCoord, float3 defaultVal) {
    auto &LP = optixLaunchParams;
    if (textureId < 0 || textureId >= (LP.textures.count + LP.materials.count * NUM_MAT_PARAMS)) return defaultVal;
    cudaTextureObject_t tex = read((cudaTextureObject_t*)LP.textureObjects.data, textureId, LP.textureObjects.count, __LINE__);
    if (!tex) return defaultVal;
    return make_float3(tex2D<float4>(tex, texCoord.x, texCoord.y));
}

inline __device__ 
float sampleTexture(int32_t textureId, float2 texCoord, int8_t channel, float defaultVal) {
    auto &LP = optixLaunchParams;
    if (textureId < 0 || textureId >= (LP.textures.count + LP.materials.count * NUM_MAT_PARAMS)) return defaultVal;
    cudaTextureObject_t tex = read((cudaTextureObject_t*)LP.textureObjects.data, textureId, LP.textureObjects.count, __LINE__);
    if (!tex) return defaultVal;
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
    auto *indexLists = (owl::device::Buffer *)LP.indexLists.data;
    int3 *indices = (int3*) read(indexLists, meshID, LP.indexLists.count, __LINE__).data;
    triIndices = read(indices, primitiveID, numIndices, __LINE__);   
}

__device__
void loadMeshVertexData(int meshID, int numVertices, int3 indices, float2 barycentrics, float3 &position, float3 &geometricNormal, float3 &edge1, float3 &edge2)
{
    auto &LP = optixLaunchParams;
    owl::device::Buffer *vertexLists = (owl::device::Buffer *)LP.vertexLists.data;
    float3 *vertices = (float3*) read(vertexLists, meshID, LP.vertexLists.count, __LINE__).data;
    const float3 A = read(vertices, indices.x, numVertices, __LINE__);
    const float3 B = read(vertices, indices.y, numVertices, __LINE__);
    const float3 C = read(vertices, indices.z, numVertices, __LINE__);
    edge1 = B - A;
    edge2 = C - A;
    position = A * (1.f - (barycentrics.x + barycentrics.y)) + B * barycentrics.x + C * barycentrics.y;
    geometricNormal = normalize(cross(B-A,C-A));
}

__device__
void loadMeshUVData(int meshID, int numTexCoords, int3 indices, float2 barycentrics, float2 &uv, float2 &edge1, float2 &edge2)
{
    auto &LP = optixLaunchParams;
    owl::device::Buffer *texCoordLists = (owl::device::Buffer *)LP.texCoordLists.data;
    float2 *texCoords = (float2*) read(texCoordLists, meshID, LP.texCoordLists.count, __LINE__).data;
    const float2 &A = read(texCoords, indices.x, numTexCoords, __LINE__);
    const float2 &B = read(texCoords, indices.y, numTexCoords, __LINE__);
    const float2 &C = read(texCoords, indices.z, numTexCoords, __LINE__);
    edge1 = B - A;
    edge2 = C - A;
    uv = A * (1.f - (barycentrics.x + barycentrics.y)) + B * barycentrics.x + C * barycentrics.y;
}

__device__
void loadMeshNormalData(int meshID, int numNormals, int3 indices, float2 barycentrics, float2 uv, float3 &normal)
{
    auto &LP = optixLaunchParams;
    owl::device::Buffer *normalLists = (owl::device::Buffer *)LP.normalLists.data;
    float4 *normals = (float4*) read(normalLists, meshID, LP.normalLists.count, __LINE__).data;
    const float3 &A = make_float3(read(normals, indices.x, numNormals, __LINE__));
    const float3 &B = make_float3(read(normals, indices.y, numNormals, __LINE__));
    const float3 &C = make_float3(read(normals, indices.z, numNormals, __LINE__));
    normal = A * (1.f - (barycentrics.x + barycentrics.y)) + B * barycentrics.x + C * barycentrics.y;
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
    glm::quat rot = glm::slerp(r0, r1, time);
    glm::mat4 camLocalToWorld = glm::mat4_cast(rot);
    camLocalToWorld = glm::column(camLocalToWorld, 3, pos);

    mat4 projinv = glm::inverse(LP.proj);
    mat4 viewinv = glm::inverse(camLocalToWorld);
    vec2 aa =  vec2(LP.xPixelSamplingInterval[0], LP.yPixelSamplingInterval[0])
            + (vec2(LP.xPixelSamplingInterval[1], LP.yPixelSamplingInterval[1]) 
            -  vec2(LP.xPixelSamplingInterval[0], LP.yPixelSamplingInterval[0])
            ) * vec2(lcg_randomf(rng),lcg_randomf(rng));

    vec2 inUV = (vec2(pixelID.x, pixelID.y) + aa) / vec2(LP.frameSize);
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
    int entity_id, float3 diffuse_mvec,
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
    if (dot(w_o, new_n) < 0.f) {
        // prevents differences from geometric and shading normal from creating black artifacts
        new_n = reflect(-new_n, gn); 
    }
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

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
    auto &LP = optixLaunchParams;
    auto launchIndex = optixGetLaunchIndex().x;
    auto launchDim = optixGetLaunchDimensions().x;
    auto pixelID = ivec2(launchIndex % LP.frameSize.x, launchIndex / LP.frameSize.x);
    auto dims = ivec2(LP.frameSize.x, LP.frameSize.x);
    uint64_t start_clock = clock();
    
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
    owl::traceRay(  /*accel to trace against*/ LP.world,
                    /*the ray to trace*/ ray,
                    /*prd*/ payload,
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT);

    // Shade each hit point on a path using NEE with MIS
    do {     
        float alpha = 0.f;

        // If ray misses, terminate the ray
        if (payload.tHit <= 0.f) {
            // Compute lighting from environment
            if (bounce == 0) {
                illum = illum + pathThroughput * (missColor(ray, envTex) * LP.domeLightIntensity);
                directIllum = illum;
            }
            else 
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
        const int entityID = read((uint32_t*)LP.instanceToEntityMap.data, payload.instanceID, LP.instanceToEntityMap.count, __LINE__);
        EntityStruct entity = read((EntityStruct*)LP.entities.data, entityID, LP.entities.count, __LINE__);
        MeshStruct mesh = read((MeshStruct*)LP.meshes.data, entity.mesh_id, LP.meshes.count, __LINE__);

        // Skip forward if the hit object is invisible for this ray type, skip it.
        if (((entity.flags & ENTITY_VISIBILITY_CAMERA_RAYS) == 0)) {
            ray.origin = ray.origin + ray.direction * (payload.tHit + EPSILON);
            payload.tHit = -1.f;
            ray.time = time;
            owl::traceRay( LP.world, ray, payload, OPTIX_RAY_FLAG_DISABLE_ANYHIT);
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
        loadMeshVertexData(entity.mesh_id, mesh.numVerts, indices, payload.barycentrics, mp, v_gz, p_e1, p_e2);
        loadMeshUVData(entity.mesh_id, mesh.numVerts, indices, payload.barycentrics, uv, uv_e1, uv_e2);
        loadMeshNormalData(entity.mesh_id, mesh.numVerts, indices, payload.barycentrics, uv, v_z);

        // Load material data for the hit object
        DisneyMaterial mat; MaterialStruct entityMaterial;
        if (entity.material_id >= 0 && entity.material_id < LP.materials.count) {
            entityMaterial = read((MaterialStruct*)LP.materials.data, entity.material_id, LP.materials.count, __LINE__);
            loadDisneyMaterial(entityMaterial, uv, mat, MIN_ROUGHNESS);
        }
       
        // Compute tangent and bitangent based on UVs
        {
            float f = 1.0f / (uv_e1.x * uv_e2.y - uv_e2.x * uv_e1.y);
            v_x.x = f * (uv_e2.y * p_e1.x - uv_e1.y * p_e2.x);
            v_x.y = f * (uv_e2.y * p_e1.y - uv_e1.y * p_e2.y);
            v_x.z = f * (uv_e2.y * p_e1.z - uv_e1.y * p_e2.z);
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
            v_y = -cross(v_z, v_x);
            v_x = -cross(v_y, v_z);

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
        ) 
        {
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

        // Potentially skip forward if the hit object is transparent 
        if ((entity.light_id == -1) && (mat.alpha < 1.f)) {
            float alpha_rnd = lcg_randomf(rng);

            if (alpha_rnd > mat.alpha) {
                ray.origin = ray.origin + ray.direction * (payload.tHit + EPSILON);
                payload.tHit = -1.f;
                ray.time = time;
                owl::traceRay( LP.world, ray, payload, OPTIX_RAY_FLAG_DISABLE_ANYHIT);
                ++bounce;     
                specularBounce++; // counting transparency as a specular bounce for now
                continue;
            }
        }

        // If we didn't hit glass, flip the surface normal to face forward.
        if ((mat.specular_transmission == 0.f) && (entity.light_id == -1)) {
            v_z = faceNormalForward(w_o, v_gz, v_z);
        }

        // For segmentations, save geometric metadata
        saveGeometricRenderData(renderData, bounce, payload.tHit, hit_p, v_z, w_o, uv, entityID, diffuseMotion, mat);

        // If the entity we hit is a light, terminate the path.
        // Note that NEE/MIS will also potentially terminate the path, preventing double-counting.
        if (entity.light_id >= 0 && entity.light_id < LP.lights.count) {
            float dotNWi = max(dot(ray.direction, v_z), 0.f);
            if ((dotNWi > EPSILON) && (bounce != 0)) break;

            LightStruct entityLight = ((LightStruct*)LP.lights.data)[entity.light_id];
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
        int numLights = LP.numLightEntities;
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
        owl::device::Buffer *vertexLists = (owl::device::Buffer *)LP.vertexLists.data;
        owl::device::Buffer *normalLists = (owl::device::Buffer *)LP.normalLists.data;
        owl::device::Buffer *texCoordLists = (owl::device::Buffer *)LP.texCoordLists.data;
        const uint32_t occlusion_flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;
        for (uint32_t lid = 0; lid < LP.numLightSamples; ++lid) 
        {
            uint32_t randomID = uint32_t(min(lcg_randomf(rng) * (numLights+1), float(numLights)));
            float dotNWi;
            float3 bsdf, bsdfColor;
            float3 lightEmission;
            float3 lightDir;
            float lightDistance = 1e20f;
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
                    // Unused by default to avoid the hit to performance
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
                sampledLightIDs[lid] = read((uint32_t*)LP.lightEntities.data, randomID, LP.lightEntities.count, __LINE__);
                EntityStruct light_entity = read((EntityStruct*)LP.entities.data, sampledLightIDs[lid], LP.entities.count, __LINE__);
                LightStruct light_light = read((LightStruct*)LP.lights.data, light_entity.light_id, LP.lights.count, __LINE__);
                TransformStruct transform = read((TransformStruct*)LP.transforms.data, light_entity.transform_id, LP.transforms.count, __LINE__);
                MeshStruct mesh = read((MeshStruct*)LP.meshes.data, light_entity.mesh_id, LP.meshes.count, __LINE__);
                uint32_t random_tri_id = uint32_t(min(lcg_randomf(rng) * mesh.numTris, float(mesh.numTris - 1)));
                owl::device::Buffer *indexLists = (owl::device::Buffer *)LP.indexLists.data;
                ivec3 *indices = (ivec3*) read(indexLists, light_entity.mesh_id, LP.indexLists.count, __LINE__).data;
                float3 *vertices = (float3*) read(vertexLists, light_entity.mesh_id, LP.vertexLists.count, __LINE__).data;
                vec4 *normals = (vec4*) read(normalLists, light_entity.mesh_id, LP.normalLists.count, __LINE__).data;
                vec2 *texCoords = (vec2*) read(texCoordLists, light_entity.mesh_id, LP.texCoordLists.count, __LINE__).data;
                ivec3 triIndex = read(indices, random_tri_id, mesh.numTris, __LINE__);   
                
                // Sample the light to compute an incident light ray to this point
                auto &ltw = transform.localToWorld;
                vec3 dir; vec2 uv;
                vec3 pos = vec3(hit_p.x, hit_p.y, hit_p.z);
                 // Might be a bug here with normal transform...
                vec4 n1 = ltw * read(normals, triIndex.x, mesh.numVerts, __LINE__);
                vec4 n2 = ltw * read(normals, triIndex.y, mesh.numVerts, __LINE__);
                vec4 n3 = ltw * read(normals, triIndex.z, mesh.numVerts, __LINE__);
                vec4 v1 = ltw * vec4(make_vec3(read(vertices, triIndex.x, mesh.numVerts, __LINE__)), 1.0f);
                vec4 v2 = ltw * vec4(make_vec3(read(vertices, triIndex.y, mesh.numVerts, __LINE__)), 1.0f);
                vec4 v3 = ltw * vec4(make_vec3(read(vertices, triIndex.z, mesh.numVerts, __LINE__)), 1.0f);
                vec2 uv1 = read(texCoords, triIndex.x, mesh.numVerts, __LINE__);
                vec2 uv2 = read(texCoords, triIndex.y, mesh.numVerts, __LINE__);
                vec2 uv3 = read(texCoords, triIndex.z, mesh.numVerts, __LINE__);
                sampleTriangle(pos, n1, n2, n3, v1, v2, v3, uv1, uv2, uv3, 
                    lcg_randomf(rng), lcg_randomf(rng), dir, lightDistance, lightPDFs[lid], uv, 
                    /*double_sided*/ false, /*use surface area*/ light_light.use_surface_area);
                
                numTris = mesh.numTris;
                lightDir = make_float3(dir.x, dir.y, dir.z);
                if (light_light.color_texture_id == -1) lightEmission = make_float3(light_light.r, light_light.g, light_light.b) * (light_light.intensity * pow(2.f, light_light.exposure));
                else lightEmission = sampleTexture(light_light.color_texture_id, make_float2(uv), make_float3(0.f, 0.f, 0.f)) * (light_light.intensity * pow(2.f, light_light.exposure));
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
                owl::traceRay( LP.world, ray, payload, occlusion_flags);
                bool visible = (randomID == numLights) ?
                    (payload.instanceID == -2) : 
                    ((payload.instanceID == -2) || (read((uint32_t*)LP.instanceToEntityMap.data, payload.instanceID, LP.instanceToEntityMap.count, __LINE__) == sampledLightIDs[lid]));
                if (visible) {
                    if (randomID != numLights) lightEmission = lightEmission / (payload.tHit * payload.tHit);
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
        ray.tmin = EPSILON * 100.f;
        payload.instanceID = -1;
        payload.tHit = -1.f;
        ray.time = sampleTime(lcg_randomf(rng));
        owl::traceRay(LP.world, ray, payload, OPTIX_RAY_FLAG_DISABLE_ANYHIT);

        // Check if we hit any of the previously sampled lights
        bool hitLight = false;
        for (uint32_t lid = 0; lid < LP.numLightSamples; ++lid)
        {
            if (lightPDFs[lid] > EPSILON) 
            {
                // if by sampling the brdf we also hit the light source...
                if ((payload.instanceID == -1) && (sampledLightIDs[lid] == -1)) {
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
                else if (payload.instanceID != -1) {
                    // Case where we hit the light, and also previously sampled the same light
                    int entityID = read((uint32_t*) LP.instanceToEntityMap.data, payload.instanceID, LP.instanceToEntityMap.count, __LINE__);
                    bool visible = (entityID == sampledLightIDs[lid]);
                    // We hit the light we sampled previously
                    if (visible) {
                        int3 indices; float3 p, p_e1, p_e2; float3 lv_gz; 
                        float2 uv, uv_e1, uv_e2;
                        EntityStruct light_entity = read((EntityStruct*)LP.entities.data, sampledLightIDs[lid], LP.entities.count, __LINE__);
                        MeshStruct light_mesh = read((MeshStruct*)LP.meshes.data, light_entity.mesh_id, LP.meshes.count, __LINE__);
                        LightStruct light_light = read((LightStruct*)LP.lights.data, light_entity.light_id, LP.lights.count, __LINE__);
                        loadMeshTriIndices(light_entity.mesh_id, light_mesh.numTris, payload.primitiveID, indices);
                        loadMeshUVData(light_entity.mesh_id, light_mesh.numVerts, indices, payload.barycentrics, uv, uv_e1, uv_e2);

                        float dist = payload.tHit;
                        float dotNWi = max(dot(ray.direction, v_gz), 0.f); // geometry term

                        float3 lightEmission;
                        if (light_light.color_texture_id == -1) lightEmission = make_float3(light_light.r, light_light.g, light_light.b) * (light_light.intensity * pow(2.f, light_light.exposure));
                        else lightEmission = sampleTexture(light_light.color_texture_id, uv, make_float3(0.f, 0.f, 0.f)) * (light_light.intensity * pow(2.f, light_light.exposure));
                        lightEmission = lightEmission / (dist * dist);

                        if (dotNWi > 0.f) 
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
        irradiance = irradiance / float(LP.numLightSamples);

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

    float4 prev_color = accumPtr[fbOfs];
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
    
    // uncoalesced writes here are expensive...
    // perhaps swap this out for a texture?
    accumPtr[fbOfs] = accum_color;
    fbPtr[fbOfs] = accum_color;
    // vec4 oldAlbedo = LP.albedoBuffer[fbOfs];
    // vec4 oldNormal = LP.normalBuffer[fbOfs];
    // if (any(isnan(oldAlbedo))) oldAlbedo = vec4(1.f);
    // if (any(isnan(oldNormal))) oldNormal = vec4(1.f);
    // vec4 newAlbedo = vec4(primaryAlbedo.x, primaryAlbedo.y, primaryAlbedo.z, 1.f);
    // vec4 newNormal = normalize(VP * vec4(primaryNormal.x, primaryNormal.y, primaryNormal.z, 0.f));
    // newNormal.a = 1.f;
    // vec4 accumAlbedo = (newAlbedo + float(LP.frameID) * oldAlbedo) / float(LP.frameID + 1);
    // vec4 accumNormal = (newNormal + float(LP.frameID) * oldNormal) / float(LP.frameID + 1);
    // LP.albedoBuffer[fbOfs] = accumAlbedo;
    // LP.normalBuffer[fbOfs] = accumNormal;    
}
