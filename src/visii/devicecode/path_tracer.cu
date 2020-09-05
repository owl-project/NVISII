#include "path_tracer.h"
#include "disney_bsdf.h"
#include "lights.h"
#include "launch_params.h"
#include "types.h"
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

    uv.x = atan(-n.x, n.y);
    uv.x = (uv.x + M_PI / 2.0) / (M_PI * 2.0) + M_PI * (28.670 / 360.0);

    uv.y = acos(n.z) / M_PI;

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

    //n = normalize(n);
    n.z = -n.z;
    n.x = -n.x;
    return n;
}

// inline __device__
// vec2 toSpherical(vec3 dir) {
//     dir = normalize(dir);
//     float u = atan(dir.z, dir.x) / (2.0f * 3.1415926535897932384626433832795f) + .5f;
//     float v = asin(dir.y) / 3.1415926535897932384626433832795f + .5f;
//     return vec2(u, (1.0f - v));
// }

// inline __device__
// vec3 toDirectional(vec2 coords) {
//     dir = normalize(dir);
//     float u = atan(dir.z, dir.x) / (2.0f * 3.1415926535897932384626433832795f) + .5f;
//     float v = asin(dir.y) / 3.1415926535897932384626433832795f + .5f;
//     return vec2(u, (1.0f - v));
// }

// Dual2<Vec3> map(float x, float y) const {
//     // pixel coordinates of entry (x,y)
//     Dual2<float> u = Dual2<float>(x, 1, 0) * invres;
//     Dual2<float> v = Dual2<float>(y, 0, 1) * invres;
//     Dual2<float> theta   = u * float(2 * M_PI);
//     Dual2<float> st, ct;
//     fast_sincos(theta, &st, &ct);
//     Dual2<float> cos_phi = 1.0f - 2.0f * v;
//     Dual2<float> sin_phi = sqrt(1.0f - cos_phi * cos_phi);
//     return make_Vec3(sin_phi * ct,
//                      sin_phi * st,
//                      cos_phi);
// }

inline __device__
float3 missColor(const float3 dir)
{
    vec3 rayDir = optixLaunchParams.environmentMapRotation * make_vec3(normalize(dir));
    if (optixLaunchParams.environmentMapID >= 0) 
    {
        vec2 tc = toUV(vec3(rayDir.x, rayDir.y, rayDir.z));
        cudaTextureObject_t tex = optixLaunchParams.textureObjects[optixLaunchParams.environmentMapID];
        if (!tex) return make_float3(1.f, 0.f, 1.f);

        float4 texColor = tex2D<float4>(tex, tc.x,tc.y);
        return make_float3(texColor);
    }
    if ((optixLaunchParams.environmentMapID == -2) && (optixLaunchParams.proceduralSkyTexture != 0)) {
        vec2 tc = toUV(vec3(rayDir.x, rayDir.y, rayDir.z));
        cudaTextureObject_t tex = optixLaunchParams.proceduralSkyTexture;
        if (!tex) return make_float3(1.f, 0.f, 1.f);

        float4 texColor = tex2D<float4>(tex, tc.x,tc.y);
        return make_float3(texColor);
    }
    
    if (glm::any(glm::greaterThanEqual(optixLaunchParams.domeLightColor, glm::vec3(0.f)))) return make_float3(optixLaunchParams.domeLightColor);

    float t = 0.5f*(rayDir.z + 1.0f);
    float3 c = (1.0f - t) * make_float3(pow(vec3(1.0f), vec3(2.2f))) + t * make_float3( pow(vec3(0.5f, 0.7f, 1.0f), vec3(2.2f)) );
    return c;
}

inline __device__
float3 missColor(const owl::Ray &ray)
{
    return missColor(ray.direction);
}


OPTIX_MISS_PROGRAM(miss)()
{
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    bool shadowray = prd.instanceID == -2;
    prd.instanceID = optixGetInstanceIndex();
    prd.tHit = optixGetRayTmax();
    if (shadowray) return;

    prd.barycentrics = optixGetTriangleBarycentrics();
    prd.primitiveID = optixGetPrimitiveIndex();
    optixGetObjectToWorldTransformMatrix(prd.localToWorld);
    
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

inline __device__
bool loadCamera(EntityStruct &cameraEntity, CameraStruct &camera, TransformStruct &transform)
{
    cameraEntity = optixLaunchParams.cameraEntity;
    if (!cameraEntity.initialized) return false;
    if ((cameraEntity.transform_id < 0) || (cameraEntity.transform_id >= MAX_TRANSFORMS)) return false;
    if ((cameraEntity.camera_id < 0) || (cameraEntity.camera_id >= MAX_CAMERAS)) return false;
    camera = optixLaunchParams.cameras[cameraEntity.camera_id];
    transform = optixLaunchParams.transforms[cameraEntity.transform_id];
    return true;
}

inline __device__ 
float3 sampleTexture(int32_t textureId, float2 texCoord) {
    if (textureId < 0 || textureId >= (MAX_TEXTURES + MAX_MATERIALS * NUM_MAT_PARAMS)) return make_float3(0.f,0.f,0.f);
    cudaTextureObject_t tex = optixLaunchParams.textureObjects[textureId];
    if (!tex) return make_float3(0.f,0.f,0.f);
    return make_float3(tex2D<float4>(tex, texCoord.x, texCoord.y));
}

inline __device__ 
float sampleTexture(int32_t textureId, float2 texCoord, int8_t channel) {
    if (textureId < 0 || textureId >= (MAX_TEXTURES + MAX_MATERIALS * NUM_MAT_PARAMS)) return 0.f;
    cudaTextureObject_t tex = optixLaunchParams.textureObjects[textureId];
    if (!tex) return 0.f;
    return make_vec4(tex2D<float4>(tex, texCoord.x, texCoord.y))[channel];
}

__device__
void loadMeshTriIndices(int meshID, int primitiveID, int3 &triIndices)
{
    owl::device::Buffer *indexLists = (owl::device::Buffer *)optixLaunchParams.indexLists.data;
    int3 *indices = (int3*) indexLists[meshID].data;
    triIndices = indices[primitiveID];   
}

__device__
void loadMeshVertexData(int meshID, int3 indices, float2 barycentrics, float3 &position, float3 &geometricNormal, float3 &edge1, float3 &edge2)
{
    owl::device::Buffer *vertexLists = (owl::device::Buffer *)optixLaunchParams.vertexLists.data;
    float4 *vertices = (float4*) vertexLists[meshID].data;
    const float3 A = make_float3(vertices[indices.x]);
    const float3 B = make_float3(vertices[indices.y]);
    const float3 C = make_float3(vertices[indices.z]);
    edge1 = B - A;
    edge2 = C - A;
    position = A * (1.f - (barycentrics.x + barycentrics.y)) + B * barycentrics.x + C * barycentrics.y;
    geometricNormal = normalize(cross(B-A,C-A));
}

__device__
void loadMeshUVData(int meshID, int3 indices, float2 barycentrics, float2 &uv, float2 &edge1, float2 &edge2)
{
    owl::device::Buffer *texCoordLists = (owl::device::Buffer *)optixLaunchParams.texCoordLists.data;
    float2 *texCoords = (float2*) texCoordLists[meshID].data;
    const float2 &A = texCoords[indices.x];
    const float2 &B = texCoords[indices.y];
    const float2 &C = texCoords[indices.z];
    edge1 = B - A;
    edge2 = C - A;
    uv = A * (1.f - (barycentrics.x + barycentrics.y)) + B * barycentrics.x + C * barycentrics.y;
}

__device__
void loadMeshNormalData(int meshID, int3 indices, float2 barycentrics, float2 uv, float3 &normal)
{
    owl::device::Buffer *normalLists = (owl::device::Buffer *)optixLaunchParams.normalLists.data;
    float4 *normals = (float4*) normalLists[meshID].data;
    const float3 &A = make_float3(normals[indices.x]);
    const float3 &B = make_float3(normals[indices.y]);
    const float3 &C = make_float3(normals[indices.z]);
    normal = A * (1.f - (barycentrics.x + barycentrics.y)) + B * barycentrics.x + C * barycentrics.y;
}

__device__ 
void loadDisneyMaterial(const MaterialStruct &p, float2 uv, DisneyMaterial &mat, float roughnessMinimum) {
    mat.base_color = sampleTexture(p.base_color_texture_id, uv);
    mat.metallic = sampleTexture(p.metallic_texture_id, uv, p.metallic_texture_channel);
    mat.specular = sampleTexture(p.specular_texture_id, uv, p.specular_texture_channel);
    mat.roughness = max(max(sampleTexture(p.roughness_texture_id, uv, p.roughness_texture_channel), MIN_ROUGHNESS), roughnessMinimum);
    mat.specular_tint = sampleTexture(p.specular_tint_texture_id, uv, p.specular_tint_texture_channel);
    mat.anisotropy = sampleTexture(p.anisotropic_texture_id, uv, p.anisotropic_texture_channel);
    mat.sheen = sampleTexture(p.sheen_texture_id, uv, p.sheen_texture_channel);
    mat.sheen_tint = sampleTexture(p.sheen_tint_texture_id, uv, p.sheen_tint_texture_channel);
    mat.clearcoat = sampleTexture(p.clearcoat_texture_id, uv, p.clearcoat_texture_channel);
    float clearcoat_roughness = max(sampleTexture(p.clearcoat_roughness_texture_id, uv, p.clearcoat_roughness_texture_channel), roughnessMinimum);
    mat.clearcoat_gloss = 1.0 - clearcoat_roughness * clearcoat_roughness;
    mat.ior = sampleTexture(p.ior_texture_id, uv, p.ior_texture_channel);
    mat.specular_transmission = sampleTexture(p.transmission_texture_id, uv, p.transmission_texture_channel);
    mat.flatness = sampleTexture(p.subsurface_texture_id, uv, p.subsurface_texture_channel);
    mat.subsurface_color = sampleTexture(p.subsurface_color_texture_id, uv);
    mat.transmission_roughness = max(max(sampleTexture(p.transmission_roughness_texture_id, uv, p.transmission_roughness_texture_channel), MIN_ROUGHNESS), roughnessMinimum);
}

__device__
float sampleTime(float xi) {
    return  optixLaunchParams.timeSamplingInterval[0] + 
           (optixLaunchParams.timeSamplingInterval[1] - 
            optixLaunchParams.timeSamplingInterval[0]) * xi;
}

inline __device__
owl::Ray generateRay(const CameraStruct &camera, const TransformStruct &transform, ivec2 pixelID, ivec2 frameSize, LCGRand &rng, float time)
{
    /* Generate camera rays */    
    glm::quat r0 = glm::quat_cast(optixLaunchParams.viewT0);
    glm::quat r1 = glm::quat_cast(optixLaunchParams.viewT1);
    glm::vec4 p0 = glm::column(optixLaunchParams.viewT0, 3);
    glm::vec4 p1 = glm::column(optixLaunchParams.viewT1, 3);

    glm::vec4 pos = glm::mix(p0, p1, time);
    glm::quat rot = glm::slerp(r0, r1, time);
    glm::mat4 camLocalToWorld = glm::mat4_cast(rot);
    camLocalToWorld = glm::column(camLocalToWorld, 3, pos);

    mat4 projinv = glm::inverse(optixLaunchParams.proj);
    mat4 viewinv = glm::inverse(camLocalToWorld);
    vec2 aa =  vec2(optixLaunchParams.xPixelSamplingInterval[0], optixLaunchParams.yPixelSamplingInterval[0])
            + (vec2(optixLaunchParams.xPixelSamplingInterval[1], optixLaunchParams.yPixelSamplingInterval[1]) 
            -  vec2(optixLaunchParams.xPixelSamplingInterval[0], optixLaunchParams.yPixelSamplingInterval[0])
            ) * vec2(lcg_randomf(rng),lcg_randomf(rng));

    vec2 inUV = (vec2(pixelID.x, pixelID.y) + aa) / vec2(optixLaunchParams.frameSize);
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
    ray.origin = owl::vec3f(origin.x, origin.y, origin.z) ;
    ray.direction = owl::vec3f(direction.x, direction.y, direction.z);
    ray.direction = normalize(owl::vec3f(direction.x, direction.y, direction.z));
    
    return ray;
}

__device__
void initializeRenderData(float3 &renderData)
{
    // these might change in the future...
    if (optixLaunchParams.renderDataMode == RenderDataFlags::NONE) {
        renderData = make_float3(FLT_MAX);
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::DEPTH) {
        renderData = make_float3(FLT_MAX);
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::POSITION) {
        renderData = make_float3(FLT_MAX);
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::NORMAL) {
        renderData = make_float3(FLT_MAX);
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::ENTITY_ID) {
        renderData = make_float3(FLT_MAX);
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::BASE_COLOR) {
        renderData = make_float3(0.0, 0.0, 0.0);
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::DIFFUSE_MOTION_VECTORS) {
        renderData = make_float3(0.0, 0.0, -1.0);
    }
}

__device__
void saveLightingColorRenderData (
    float3 &renderData, int bounce,
    float3 w_n, float3 w_o, float3 w_i, 
    DisneyMaterial &mat
)
{
    if (optixLaunchParams.renderDataMode == RenderDataFlags::NONE) return;
    if (bounce != optixLaunchParams.renderDataBounce) return;
    
    // Note, dillum and iillum are expected to change outside this function depending on the 
    // render data flags.
    if (optixLaunchParams.renderDataMode == RenderDataFlags::DIFFUSE_COLOR) {
        renderData = disney_diffuse_color(mat, w_n, w_o, w_i); 
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::GLOSSY_COLOR) {
        renderData = disney_microfacet_reflection_color(mat, w_n, w_o, w_i);
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::TRANSMISSION_COLOR) {
        renderData = disney_microfacet_transmission_color(mat, w_n, w_o, w_i);
    }
}

__device__
void saveLightingIrradianceRenderData(
    float3 &renderData, int bounce,
    float3 dillum, float3 iillum,
    int sampledBsdf)
{
    if (optixLaunchParams.renderDataMode == RenderDataFlags::NONE) return;
    if (bounce != optixLaunchParams.renderDataBounce) return;
    
    // Note, dillum and iillum are expected to change outside this function depending on the 
    // render data flags.
    if (optixLaunchParams.renderDataMode == RenderDataFlags::DIFFUSE_DIRECT_LIGHTING) {
        renderData = dillum;
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::DIFFUSE_INDIRECT_LIGHTING) {
        renderData = iillum;
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::GLOSSY_DIRECT_LIGHTING) {
        renderData = dillum;
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::GLOSSY_INDIRECT_LIGHTING) {
        renderData = iillum;
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::TRANSMISSION_DIRECT_LIGHTING) {
        renderData = dillum;
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::TRANSMISSION_INDIRECT_LIGHTING) {
        renderData = iillum;
    }
}

__device__
void saveMissRenderData(
    float3 &renderData, 
    int bounce,
    float3 mvec)
{
    if (optixLaunchParams.renderDataMode == RenderDataFlags::NONE) return;
    if (bounce != optixLaunchParams.renderDataBounce) return;

    if (optixLaunchParams.renderDataMode == RenderDataFlags::DIFFUSE_MOTION_VECTORS) {
        renderData = mvec;
    }
}


__device__
void saveGeometricRenderData(
    float3 &renderData, 
    int bounce, float depth, 
    float3 w_p, float3 w_n, float3 w_o,
    int entity_id, float3 diffuse_mvec,
    DisneyMaterial &mat)
{
    if (optixLaunchParams.renderDataMode == RenderDataFlags::NONE) return;
    if (bounce != optixLaunchParams.renderDataBounce) return;

    if (optixLaunchParams.renderDataMode == RenderDataFlags::DEPTH) {
        renderData = make_float3(depth);
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::POSITION) {
        renderData = w_p;
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::NORMAL) {
        renderData = w_n;
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::ENTITY_ID) {
        renderData = make_float3(float(entity_id));
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::DIFFUSE_MOTION_VECTORS) {
        renderData = diffuse_mvec;
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::BASE_COLOR) {
        renderData = mat.base_color;
    }
    else if (optixLaunchParams.renderDataMode == RenderDataFlags::RAY_DIRECTION) {
        renderData = -w_o;
    }
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
    auto pixelID = ivec2(owl::getLaunchIndex()[0], owl::getLaunchIndex()[1]);
    return glm::all(glm::equal(pixelID, ivec2(optixLaunchParams.frameSize.x / 2, optixLaunchParams.frameSize.y / 2)));
}

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
    auto pixelID = ivec2(owl::getLaunchIndex()[0], owl::getLaunchIndex()[1]);
    float start_clock, stop_clock;
    start_clock = clock();
    
    LCGRand rng = get_rng(optixLaunchParams.frameID + optixLaunchParams.seed * 10007);
    float time = sampleTime(lcg_randomf(rng));

    // If no camera is in use, just display some random noise...
    owl::Ray ray;
    {
        EntityStruct    camera_entity;
        TransformStruct camera_transform;
        CameraStruct    camera;
        if (!loadCamera(camera_entity, camera, camera_transform)) {
            auto fbOfs = pixelID.x+optixLaunchParams.frameSize.x * ((optixLaunchParams.frameSize.y - 1) -  pixelID.y);
            optixLaunchParams.frameBuffer[fbOfs] = vec4(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng), 1.f);
            return;
        }
        
        // Trace an initial ray through the scene
        ray = generateRay(camera, camera_transform, pixelID, optixLaunchParams.frameSize, rng, time);
    }

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
    owl::traceRay(  /*accel to trace against*/ optixLaunchParams.world,
                    /*the ray to trace*/ ray,
                    /*prd*/ payload,
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT);

    stop_clock = clock();
    if (debugging()) {
        printf("dClock : %f\n", stop_clock - start_clock);
    }

    // Shade each hit point on a path using NEE with MIS
    do {     
        // If ray misses
        if (payload.tHit <= 0.f) {
            illum = illum + pathThroughput * (missColor(ray) * optixLaunchParams.domeLightIntensity);
            if (bounce == 0) {
                // primaryNormal = make_float3(0.f, 0.f, 1.f);
                // primaryAlbedo = illum;
                directIllum = illum;
            }
            
            const float envDist = 10000.0f; // large value
            /* Compute miss motion vector */
            float3 mvec;
            // Point far away
            float3 pFar = ray.origin + ray.direction * envDist;
            // TODO: account for motion from rotating dome light
            vec4 tmp1 = optixLaunchParams.proj * optixLaunchParams.viewT0 * /*xfmt0 **/ make_vec4(pFar, 1.0f);
            float3 pt0 = make_float3(tmp1 / tmp1.w) * .5f;
            vec4 tmp2 = optixLaunchParams.proj * optixLaunchParams.viewT1 * /*xfmt1 **/ make_vec4(pFar, 1.0f);
            float3 pt1 = make_float3(tmp2 / tmp2.w) * .5f;
            mvec = pt1 - pt0;
            saveMissRenderData(renderData, bounce, mvec);
            break;
        }

        // Otherwise, load common position, vectors, and material data used for shading...
        const int entityID = optixLaunchParams.instanceToEntityMap[payload.instanceID];
        EntityStruct entity = optixLaunchParams.entities[entityID];

        // Skip forward if the hit object is invisible for this ray type
        if ((bounce == 0) && ((entity.visibilityFlags & ENTITY_VISIBILITY_CAMERA_RAYS) == 0)) {
            ray.origin = ray.origin + ray.direction * (payload.tHit + EPSILON);
            payload.tHit = -1.f;
            ray.time = time;
            owl::traceRay( optixLaunchParams.world, ray, payload, OPTIX_RAY_FLAG_DISABLE_ANYHIT);
            visibilitySkips++;
            if (visibilitySkips > 10) break; // avoid locking up.

            // If ray misses
            if (payload.tHit <= 0.f) {
                illum = missColor(ray) * optixLaunchParams.domeLightIntensity;
                // primaryNormal = make_float3(0.f, 0.f, 1.f);
                // primaryAlbedo = illum;
                directIllum = illum;
            }
            continue;
        }

        DisneyMaterial mat; MaterialStruct entityMaterial; LightStruct entityLight;
        if (entity.material_id >= 0 && entity.material_id < MAX_MATERIALS) {
            entityMaterial = optixLaunchParams.materials[entity.material_id];
        }
        
        const float3 w_o = -ray.direction;
        float3 hit_p = ray.origin + payload.tHit * ray.direction;
        float3 mp, p, v_x, v_y, v_z, v_gz, p_e1, p_e2; 
        float2 uv, uv_e1, uv_e2; 
        int3 indices;
        float3 diffuseMotion;
        
        loadMeshTriIndices(entity.mesh_id, payload.primitiveID, indices);
        loadMeshVertexData(entity.mesh_id, indices, payload.barycentrics, mp, v_gz, p_e1, p_e2);
        loadMeshUVData(entity.mesh_id, indices, payload.barycentrics, uv, uv_e1, uv_e2);
        loadMeshNormalData(entity.mesh_id, indices, payload.barycentrics, uv, v_z);
        loadDisneyMaterial(entityMaterial, uv, mat, MIN_ROUGHNESS);
        
        bool shouldNormalFaceForward = (mat.specular_transmission == 0.f);
        {
            float f = 1.0f / (uv_e1.x * uv_e2.y - uv_e2.x * uv_e1.y);
            v_x.x = f * (uv_e2.y * p_e1.x - uv_e1.y * p_e2.x);
            v_x.y = f * (uv_e2.y * p_e1.y - uv_e1.y * p_e2.y);
            v_x.z = f * (uv_e2.y * p_e1.z - uv_e1.y * p_e2.z);
            v_x = normalize(v_x);
            v_z = normalize(v_z);            
        }
            
        // Transform data into world space
        {
            glm::mat4 xfm = to_mat4(payload.localToWorld);
            glm::mat4 xfmt0 = to_mat4(payload.localToWorldT0);
            glm::mat4 xfmt1 = to_mat4(payload.localToWorldT1);
            glm::mat3 nxfm = transpose(glm::inverse(glm::mat3(xfm)));
            p = make_float3(xfm * make_vec4(mp, 1.0f));
            vec4 tmp1 = optixLaunchParams.proj * optixLaunchParams.viewT0 * xfmt0 * make_vec4(mp, 1.0f);
            float3 pt0 = make_float3(tmp1 / tmp1.w) * .5f;
            vec4 tmp2 = optixLaunchParams.proj * optixLaunchParams.viewT1 * xfmt1 * make_vec4(mp, 1.0f);
            float3 pt1 = make_float3(tmp2 / tmp2.w) * .5f;
            diffuseMotion = pt1 - pt0;
            hit_p = p;
            v_gz = make_float3(normalize(nxfm * make_vec3(v_gz)));
            v_z = make_float3(normalize(nxfm * make_vec3(v_z)));
            v_x = make_float3(normalize(nxfm * make_vec3(v_x)));
            v_y = cross(v_z, v_x);
            v_x = cross(v_y, v_z);
        }       

        if (
            all(lessThan(abs(make_vec3(v_x)), vec3(EPSILON))) || 
            all(lessThan(abs(make_vec3(v_y)), vec3(EPSILON))) ||
            any(isnan(make_vec3(v_x))) || 
            any(isnan(make_vec3(v_y)))
        ) 
        {
            ortho_basis(v_x, v_y, v_z);
        }

        {
            glm::mat3 tbn;
            tbn = glm::column(tbn, 0, make_vec3(v_x) );
            tbn = glm::column(tbn, 1, make_vec3(v_y) );
            tbn = glm::column(tbn, 2, make_vec3(v_z) );   
            float3 dN = sampleTexture(entityMaterial.normal_map_texture_id, uv); //vec4(0.5f, .5f, 1.f, 0.f)
            dN = (dN * make_float3(2.0f)) - make_float3(1.f);   
            v_z = make_float3(normalize(tbn * normalize(make_vec3(dN))) );
        }

        if (shouldNormalFaceForward) {
            v_z = faceNormalForward(w_o, v_gz, v_z);
        }

        // For segmentations, geometric metadata extraction dependent on the hit object
        saveGeometricRenderData(renderData, bounce, payload.tHit, hit_p, v_z, w_o, entityID, diffuseMotion, mat);
                    
        // If this is the first hit, keep track of primary albedo and normal for denoising.
        // if (bounce == 0) {
            // primaryNormal = v_z;
            // primaryAlbedo = mat.base_color;
        // }

        // If the entity we hit is a light, terminate the path.
        // First hits are colored by the light. All other light hits are handled by NEE/MIS 
        if (entity.light_id >= 0 && entity.light_id < MAX_LIGHTS) {
            if (bounce == 0) 
            {
                entityLight = optixLaunchParams.lights[entity.light_id];
                float3 light_emission;
                if (entityLight.color_texture_id == -1) light_emission = make_float3(entityLight.r, entityLight.g, entityLight.b) * entityLight.intensity;
                else light_emission = sampleTexture(entityLight.color_texture_id, uv); // * intensity; temporarily commenting out to show texture for bright lights in LDR
                illum = light_emission; 
                directIllum = illum;
            }
            break;
        }
                    
        // Sample a light source
        int32_t sampledLightIDs[MAX_LIGHT_SAMPLES] = {-2};
        float lightPDFs[MAX_LIGHT_SAMPLES] = {0.f};
        
        int numLights = optixLaunchParams.numLightEntities;
        float3 irradiance = make_float3(0.f);

        // note, rdForcedBsdf is -1 by default
        int forcedBsdf = -1;//rdForcedBsdf;//(bounce == optixLaunchParams.renderDataBounce) ? rdForcedBsdf : -1; 

        // first, sample the BRDF so that we can use the sampled direction for MIS
        float3 w_i;
        float bsdfPDF;
        int sampledBsdf = -1;
        float3 bsdf, bsdfColor;
        sample_disney_brdf(mat, v_z, w_o, v_x, v_y, rng, w_i, bsdfPDF, sampledBsdf, bsdf, bsdfColor, forcedBsdf);

        // next, sample the light source by importance sampling the light
        owl::device::Buffer *vertexLists = (owl::device::Buffer *)optixLaunchParams.vertexLists.data;
        owl::device::Buffer *normalLists = (owl::device::Buffer *)optixLaunchParams.normalLists.data;
        owl::device::Buffer *texCoordLists = (owl::device::Buffer *)optixLaunchParams.texCoordLists.data;
        auto &i2e = optixLaunchParams.instanceToEntityMap;
        const uint32_t occlusion_flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;
        // for (uint32_t lid = 0; lid < optixLaunchParams.numLightSamples; ++lid) 
        {
            uint32_t lid = 0;
            uint32_t randomID = uint32_t(min(lcg_randomf(rng) * (numLights+1), float(numLights)));
            float dotNWi;
            float3 bsdf, bsdfColor;
            float3 lightEmission;
            float3 lightDir;
            int numTris;
            
            // sample background
            if (randomID == numLights)
            {
                sampledLightIDs[lid] = -1;

                if (
                    (optixLaunchParams.environmentMapWidth != 0) && (optixLaunchParams.environmentMapHeight != 0) &&
                    (optixLaunchParams.environmentMapRows != nullptr) && (optixLaunchParams.environmentMapCols != nullptr)
                ) 
                {
                    // significant bottleneck here
                    // Vec3fa color = m_background->sample(dg, wi, tMax, RandomSampler_get2D(sampler));
                    float rx = lcg_randomf(rng);
                    float ry = lcg_randomf(rng);
                    float* rows = optixLaunchParams.environmentMapRows;
                    float* cols = optixLaunchParams.environmentMapCols;
                    int width = optixLaunchParams.environmentMapWidth;
                    int height = optixLaunchParams.environmentMapHeight;
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
                    const float3 hemi_dir = normalize(cos_sample_hemisphere(make_float2(lcg_randomf(rng), lcg_randomf(rng))));
                    lightDir = make_float3(normalize(tbn * normalize(make_vec3(hemi_dir))) );
                    lightPDFs[lid] = 1.f;
                }

                numTris = 1.f;
                dotNWi = fabs(dot(lightDir, v_z)); // for now, making all lights double sided.
                lightEmission = (missColor(ray) * optixLaunchParams.domeLightIntensity);
                disney_brdf(mat, v_z, w_o, lightDir, v_x, v_y, bsdf, bsdfColor, forcedBsdf);
            }
            // sample light sources
            else 
            {
                if (numLights == 0) continue;
                sampledLightIDs[lid] = optixLaunchParams.lightEntities[randomID];
                EntityStruct light_entity = optixLaunchParams.entities[sampledLightIDs[lid]];
                LightStruct light_light = optixLaunchParams.lights[light_entity.light_id];
                TransformStruct transform = optixLaunchParams.transforms[light_entity.transform_id];
                MeshStruct mesh = optixLaunchParams.meshes[light_entity.mesh_id];

                uint32_t random_tri_id = uint32_t(min(lcg_randomf(rng) * mesh.numTris, float(mesh.numTris - 1)));
                owl::device::Buffer *indexLists = (owl::device::Buffer *)optixLaunchParams.indexLists.data;
                ivec3 *indices = (ivec3*) indexLists[light_entity.mesh_id].data;
                vec4 *vertices = (vec4*) vertexLists[light_entity.mesh_id].data;
                vec4 *normals = (vec4*) normalLists[light_entity.mesh_id].data;
                vec2 *texCoords = (vec2*) texCoordLists[light_entity.mesh_id].data;
                ivec3 triIndex = indices[random_tri_id];   
                
                // Sample the light to compute an incident light ray to this point
                auto &ltw = transform.localToWorld;
                vec3 dir; vec2 uv;
                vec3 pos = vec3(hit_p.x, hit_p.y, hit_p.z);
                sampleTriangle(pos, 
                    ltw * normals[triIndex.x], ltw * normals[triIndex.y], ltw * normals[triIndex.z], 
                    ltw * vertices[triIndex.x], ltw * vertices[triIndex.y], ltw * vertices[triIndex.z], // Might be a bug here with normal transform...
                    texCoords[triIndex.x], texCoords[triIndex.y], texCoords[triIndex.z], 
                    lcg_randomf(rng), lcg_randomf(rng), dir, lightPDFs[lid], uv, /*double_sided*/ false);
                vec3 normal = glm::vec3(v_z.x, v_z.y, v_z.z);
                
                dotNWi = abs(dot(dir, normal));
                numTris = mesh.numTris;
                lightDir = make_float3(dir.x, dir.y, dir.z);
                if (light_light.color_texture_id == -1) lightEmission = make_float3(light_light.r, light_light.g, light_light.b) * light_light.intensity;
                else lightEmission = sampleTexture(light_light.color_texture_id, make_float2(uv)) * light_light.intensity;
                disney_brdf(mat, v_z, w_o, lightDir, v_x, v_y, bsdf, bsdfColor, forcedBsdf);
            }

            lightPDFs[lid] *= (1.f / (numLights + 1)) * (1.f / (numTris));
            if ((lightPDFs[lid] > 0.0) && (dotNWi > EPSILON)) {
                RayPayload payload; payload.instanceID = -2;
                owl::Ray ray;
                ray.tmin = EPSILON * 10.f; ray.tmax = 1e20f;
                ray.origin = hit_p; ray.direction = lightDir;
                ray.time = time;
                owl::traceRay( optixLaunchParams.world, ray, payload, occlusion_flags);
                bool visible = (randomID == numLights) ?
                    (payload.instanceID == -2) : (payload.instanceID >= 0 && i2e[payload.instanceID] == sampledLightIDs[lid]);
                if (visible) {
                    if (randomID != numLights) lightEmission = lightEmission / (payload.tHit * payload.tHit);
                    float w = power_heuristic(1.f, lightPDFs[lid], 1.f, bsdfPDF);
                    float3 Li = (lightEmission * w) / lightPDFs[lid];
                    irradiance = irradiance + (bsdf * bsdfColor * Li * fabs(dotNWi));
                }
            }
        }

        // For segmentations, lighting metadata extraction dependent on sampling the BSDF
        saveLightingColorRenderData(renderData, bounce, v_z, w_o, w_i, mat);

        // terminate if the bsdf probability is impossible, or if the bsdf filters out all light
        if (bsdfPDF < EPSILON || all_zero(bsdf) || all_zero(bsdfColor)) {
            float3 contribution = pathThroughput * irradiance;
            illum = illum + contribution;
            break;
        }

        // sample a light source by importance sampling the BDRF
        // trace the next ray along that sampled BRDF direction
        ray.origin = hit_p;
        ray.direction = w_i;
        ray.tmin = EPSILON * 100.f;
        payload.instanceID = -1;
        payload.tHit = -1.f;
        ray.time = sampleTime(lcg_randomf(rng));
        owl::traceRay(optixLaunchParams.world, ray, payload, OPTIX_RAY_FLAG_DISABLE_ANYHIT);

        // for (uint32_t lid = 0; lid < optixLaunchParams.numLightSamples; ++lid)
        {
            uint32_t lid = 0;
            if (lightPDFs[lid] > EPSILON) 
            {
                // if by sampling the brdf we also hit the light source...
                if ((payload.instanceID == -1) && (sampledLightIDs[lid] == -1)) {
                    // Case where we hit the background, and also previously sampled the background   
                    float w = power_heuristic(1.f, bsdfPDF, 1.f, lightPDFs[lid]);
                    float3 lightEmission = missColor(ray) * optixLaunchParams.domeLightIntensity;
                    float3 Li = (lightEmission * w) / bsdfPDF;
                    float dotNWi = dot(v_gz, ray.direction);  // geometry term
                    if (dotNWi > 0.f) {
                        irradiance = irradiance + (bsdf * bsdfColor * Li * fabs(dotNWi));
                    }
                }
                else if (payload.instanceID != -1) {
                    // Case where we hit the light, and also previously sampled the same light
                    int entityID = optixLaunchParams.instanceToEntityMap[payload.instanceID];
                    bool visible = (entityID == sampledLightIDs[lid]);
                    // We hit the light we sampled previously
                    if (visible) {
                        int3 indices; float3 p, p_e1, p_e2; float3 lv_gz; 
                        float2 uv, uv_e1, uv_e2;
                        EntityStruct light_entity = optixLaunchParams.entities[sampledLightIDs[lid]];
                        LightStruct light_light = optixLaunchParams.lights[light_entity.light_id];
                        loadMeshTriIndices(light_entity.mesh_id, payload.primitiveID, indices);
                        // loadMeshVertexData(light_entity.mesh_id, indices, payload.barycentrics, p, lv_gz, p_e1, p_e2);
                        loadMeshUVData(light_entity.mesh_id, indices, payload.barycentrics, uv, uv_e1, uv_e2);

                        float dist = payload.tHit;// distance(vec3(p.x, p.y, p.z), vec3(ray.origin.x, ray.origin.y, ray.origin.z)); // should I be using this?
                        float dotNWi = dot(v_gz, ray.direction); // geometry term

                        float3 lightEmission;
                        if (light_light.color_texture_id == -1) lightEmission = make_float3(light_light.r, light_light.g, light_light.b) * light_light.intensity;
                        else lightEmission = sampleTexture(light_light.color_texture_id, uv) * light_light.intensity;
                        lightEmission = lightEmission / (dist * dist);

                        // float dotWiN = dot(-lv_gz, ray.direction); // is light facing towards us? // Seems like this calculation isn't needed.
                        if ((dotNWi > 0.f) /*&& (dotWiN > 0.f)*/) 
                        {
                            float w = power_heuristic(1.f, bsdfPDF, 1.f, lightPDFs[lid]);
                            float3 Li = (lightEmission * w) / bsdfPDF;
                            irradiance = irradiance + (bsdf * bsdfColor * Li * fabs(dotNWi)); // missing r^2 falloff?
                        }
                    }
                }
            }
        }

        // irradiance = irradiance / float(optixLaunchParams.numLightSamples);

        // accumulate any radiance (ie pathThroughput * irradiance), and update the path throughput using the sampled BRDF
        float3 contribution = pathThroughput * irradiance;
        illum = illum + contribution;
        pathThroughput = (pathThroughput * bsdf * bsdfColor) / bsdfPDF;

        if (bounce == 0) {
            directIllum = illum;
        }

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
    } while (diffuseBounce < optixLaunchParams.maxDiffuseBounceDepth && specularBounce < optixLaunchParams.maxSpecularBounceDepth);

    // clamp out any extreme fireflies
    glm::vec3 gillum = vec3(illum.x, illum.y, illum.z);
    glm::vec3 dillum = vec3(directIllum.x, directIllum.y, directIllum.z);
    glm::vec3 iillum = gillum - dillum;

    // For segmentations, indirect/direct lighting metadata extraction
    // float3 aovGIllum = aovIllum;
    // aovIndirectIllum = aovGIllum - aovDirectIllum;
    // saveLightingIrradianceRenderData(renderData, bounce, aovDirectIllum, aovIndirectIllum, rdSampledBsdf);

    if (optixLaunchParams.indirectClamp > 0.f)
        iillum = clamp(iillum, vec3(0.f), vec3(optixLaunchParams.indirectClamp));
    if (optixLaunchParams.directClamp > 0.f)
        dillum = clamp(dillum, vec3(0.f), vec3(optixLaunchParams.directClamp));

    gillum = dillum + iillum;

    // just in case we get inf's or nans, remove them.
    if (glm::any(glm::isnan(gillum))) gillum = vec3(0.f);
    if (glm::any(glm::isinf(gillum))) gillum = vec3(0.f);
    illum = make_float3(gillum.r, gillum.g, gillum.b);

    // accumulate the illumination from this sample into what will be an average illumination from all samples in this pixel
    accum_illum = illum;

    /* Write to AOVs, progressively refining results */
    auto fbOfs = pixelID.x+optixLaunchParams.frameSize.x * ((optixLaunchParams.frameSize.y - 1) -  pixelID.y);
    float4 &prev_color = (float4&) optixLaunchParams.accumPtr[fbOfs];
    float4 accum_color = make_float4((accum_illum + float(optixLaunchParams.frameID) * make_float3(prev_color)) / float(optixLaunchParams.frameID + 1), 1.0f);
    optixLaunchParams.accumPtr[fbOfs] = vec4(
        accum_color.x, 
        accum_color.y, 
        accum_color.z, 
        accum_color.w
    );

    float3 color = make_float3(accum_color);
    optixLaunchParams.frameBuffer[fbOfs] = vec4(
        color.x,
        color.y,
        color.z,
        1.0f
    );
    
    // vec4 oldAlbedo = optixLaunchParams.albedoBuffer[fbOfs];
    // vec4 oldNormal = optixLaunchParams.normalBuffer[fbOfs];
    // if (any(isnan(oldAlbedo))) oldAlbedo = vec4(1.f);
    // if (any(isnan(oldNormal))) oldNormal = vec4(1.f);
    // vec4 newAlbedo = vec4(primaryAlbedo.x, primaryAlbedo.y, primaryAlbedo.z, 1.f);
    // vec4 newNormal = normalize(VP * vec4(primaryNormal.x, primaryNormal.y, primaryNormal.z, 0.f));
    // newNormal.a = 1.f;
    // vec4 accumAlbedo = (newAlbedo + float(optixLaunchParams.frameID) * oldAlbedo) / float(optixLaunchParams.frameID + 1);
    // vec4 accumNormal = (newNormal + float(optixLaunchParams.frameID) * oldNormal) / float(optixLaunchParams.frameID + 1);
    // optixLaunchParams.albedoBuffer[fbOfs] = accumAlbedo;
    // optixLaunchParams.normalBuffer[fbOfs] = accumNormal;

    // Override framebuffer output if user requested to render metadata
    if (optixLaunchParams.renderDataMode != RenderDataFlags::NONE) {
        optixLaunchParams.frameBuffer[fbOfs] = vec4( renderData.x, renderData.y, renderData.z, 1.0f);
    }
}
