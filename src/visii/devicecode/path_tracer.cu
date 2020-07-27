#include "path_tracer.h"
#include "disney_bsdf.h"
#include "lights.h"
#include "launch_params.h"
#include "types.h"
#include <optix_device.h>
#include <owl/common/math/random.h>

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

inline __device__
vec2 toSpherical(vec3 dir) {
    dir = normalize(dir);
    float u = atan(dir.z, dir.x) / (2.0f * 3.1415926535897932384626433832795f) + .5f;
    float v = asin(dir.y) / 3.1415926535897932384626433832795f + .5f;
    return vec2(u, (1.0f - v));
}

inline __device__
float3 missColor(const owl::Ray &ray)
{
    auto pixelID = owl::getLaunchIndex();

    vec3 rayDir = optixLaunchParams.environmentMapRotation * make_vec3(normalize(ray.direction));
    if (optixLaunchParams.environmentMapID != -1) 
    {
        vec2 tc = toSpherical(vec3(rayDir.x, -rayDir.z, rayDir.y));
        cudaTextureObject_t tex = optixLaunchParams.textureObjects[optixLaunchParams.environmentMapID];
        if (!tex) return make_float3(1.f, 0.f, 1.f);

        float4 texColor = tex2D<float4>(tex, tc.x,tc.y);
        return make_float3(texColor);
    }

    float t = 0.5f*(rayDir.z + 1.0f);
    float3 c = (1.0f - t) * make_float3(pow(vec3(1.0f), vec3(2.2f))) + t * make_float3( pow(vec3(0.5f, 0.7f, 1.0f), vec3(2.2f)) );
    return c;
}

OPTIX_MISS_PROGRAM(miss)()
{
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    prd.barycentrics = optixGetTriangleBarycentrics();
    prd.instanceID = optixGetInstanceIndex();
    prd.primitiveID = optixGetPrimitiveIndex();
    prd.tHit = optixGetRayTmax();
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
vec4 sampleTexture(int32_t textureId, vec2 texCoord, vec4 defaultValue) {
    if (textureId < 0 || textureId >= MAX_TEXTURES) return defaultValue;
    cudaTextureObject_t tex = optixLaunchParams.textureObjects[textureId];
    if (!tex) return defaultValue;
    return make_vec4(tex2D<float4>(tex, texCoord.x, texCoord.y));
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
void loadDisneyMaterial(const MaterialStruct &p, vec2 uv, DisneyMaterial &mat, float roughnessMinimum) {
    mat.base_color = make_float3(sampleTexture(p.base_color_texture_id, uv, vec4(p.base_color.r, p.base_color.g, p.base_color.b, 1.f)));
    mat.metallic = sampleTexture(p.metallic_texture_id, uv, vec4(p.metallic))[p.metallic_texture_channel];
    mat.specular = sampleTexture(p.specular_texture_id, uv, vec4(p.specular))[p.specular_texture_channel];
    mat.roughness = max(max(sampleTexture(p.roughness_texture_id, uv, vec4(p.roughness))[p.roughness_texture_channel], MIN_ROUGHNESS), roughnessMinimum);
    mat.specular_tint = sampleTexture(p.specular_tint_texture_id, uv, vec4(p.specular_tint))[p.specular_tint_texture_channel];
    mat.anisotropy = sampleTexture(p.anisotropic_texture_id, uv, vec4(p.anisotropic))[p.anisotropic_texture_channel];
    mat.sheen = sampleTexture(p.sheen_texture_id, uv, vec4(p.sheen))[p.sheen_texture_channel];
    mat.sheen_tint = sampleTexture(p.sheen_tint_texture_id, uv, vec4(p.sheen_tint))[p.sheen_tint_texture_channel];
    mat.clearcoat = sampleTexture(p.clearcoat_texture_id, uv, vec4(p.clearcoat))[p.clearcoat_texture_channel];
    float clearcoat_roughness = max(sampleTexture(p.clearcoat_roughness_texture_id, uv, vec4(p.clearcoat_roughness))[p.clearcoat_roughness_texture_channel], roughnessMinimum);
    mat.clearcoat_gloss = 1.0 - clearcoat_roughness * clearcoat_roughness;
    mat.ior = sampleTexture(p.ior_texture_id, uv, vec4(p.ior))[p.ior_texture_channel];
    mat.specular_transmission = sampleTexture(p.transmission_texture_id, uv, vec4(p.transmission))[p.transmission_texture_channel];
    mat.flatness = sampleTexture(p.subsurface_texture_id, uv, vec4(p.subsurface))[p.subsurface_texture_channel];
    mat.transmission_roughness = max(max(sampleTexture(p.transmission_roughness_texture_id, uv, vec4(p.transmission_roughness))[p.transmission_roughness_texture_channel], MIN_ROUGHNESS), roughnessMinimum);
}

__device__
float sampleTime(float xi) {
    return  optixLaunchParams.timeSamplingInterval[0] + 
           (optixLaunchParams.timeSamplingInterval[1] - 
            optixLaunchParams.timeSamplingInterval[0]) * xi;
}

inline __device__
owl::Ray generateRay(const CameraStruct &camera, const TransformStruct &transform, ivec2 pixelID, ivec2 frameSize, LCGRand &rng)
{
    /* Generate camera rays */    
    glm::quat r0 = glm::quat_cast(optixLaunchParams.viewT0);
    glm::quat r1 = glm::quat_cast(optixLaunchParams.viewT1);
    glm::vec4 p0 = glm::column(optixLaunchParams.viewT0, 3);
    glm::vec4 p1 = glm::column(optixLaunchParams.viewT1, 3);
    float time = sampleTime(lcg_randomf(rng));

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
void saveRenderData(float3 &renderData, int bounce, float depth, float3 w_p, float3 w_n, int entity_id, float3 diffuse_mvec, float3 base_color)
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
        renderData = base_color;
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

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
    auto pixelID = ivec2(owl::getLaunchIndex()[0], owl::getLaunchIndex()[1]);
    auto fbOfs = pixelID.x+optixLaunchParams.frameSize.x* ((optixLaunchParams.frameSize.y - 1) -  pixelID.y);
    LCGRand rng = get_rng(optixLaunchParams.frameID + optixLaunchParams.seed * 10007);

    // If no camera is in use, just display some random noise...
    EntityStruct    camera_entity;
    TransformStruct camera_transform;
    CameraStruct    camera;
    if (!loadCamera(camera_entity, camera, camera_transform)) {
        optixLaunchParams.frameBuffer[fbOfs] = vec4(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng), 1.f);
        return;
    }

    mat4 VP = camera.proj * camera_transform.worldToLocal;

    float3 accum_illum = make_float3(0.f);
    float3 primaryAlbedo = make_float3(0.f);
    float3 primaryNormal = make_float3(0.f);
    float3 primaryDiffuseMotion = make_float3(0.f);
    
    float3 renderData = make_float3(0.f);
    initializeRenderData(renderData);
    
    // For potentially several samples per pixel... 
    // Update: for machine learning applications, it's important there be only 1SPP.
    // Metadata like depth or IDs dont work with multiple SPP.
    #define SPP 1
    for (uint32_t rid = 0; rid < SPP; ++rid) 
    {

        // Trace an initial ray through the scene
        owl::Ray ray = generateRay(camera, camera_transform, pixelID, optixLaunchParams.frameSize, rng);

        DisneyMaterial mat;
        int bounce = 0;
        int visibilitySkips = 0;
        float3 directIllum = make_float3(0.f);
        float3 illum = make_float3(0.f);
        float3 path_throughput = make_float3(1.f);
        uint16_t ray_count = 0;
        float roughnessMinimum = 0.f;
        RayPayload payload;
        payload.tHit = -1.f;
        ray.time = sampleTime(lcg_randomf(rng));
        owl::traceRay(  /*accel to trace against*/ optixLaunchParams.world,
                        /*the ray to trace*/ ray,
                        /*prd*/ payload);

        // If ray misses
        if (payload.tHit <= 0.f) {
            illum = missColor(ray) * optixLaunchParams.domeLightIntensity;
            primaryNormal = make_float3(0.f, 0.f, 1.f);
            primaryAlbedo = illum;
            directIllum = illum;
        }

        // If we hit something, shade each hit point on a path using NEE with MIS
        else do {     
            // Load common position, vectors, and material data used for shading...
            const int entityID = optixLaunchParams.instanceToEntityMap[payload.instanceID];
            EntityStruct entity = optixLaunchParams.entities[entityID];

            // Skip forward if the hit object is invisible for this ray type
            if ((bounce == 0) && ((entity.visibilityFlags & ENTITY_VISIBILITY_CAMERA_RAYS) == 0)) {
                ray.origin = ray.origin + ray.direction * (payload.tHit + EPSILON);
                payload.tHit = -1.f;
                ray.time = sampleTime(lcg_randomf(rng));
                owl::traceRay( optixLaunchParams.world, ray, payload);
                visibilitySkips++;
                if (visibilitySkips > 10) break; // avoid locking up.

                // If ray misses
                if (payload.tHit <= 0.f) {
                    illum = missColor(ray) * optixLaunchParams.domeLightIntensity;
                    primaryNormal = make_float3(0.f, 0.f, 1.f);
                    primaryAlbedo = illum;
                    directIllum = illum;
                }
                continue;
            }

            TransformStruct entityTransform = optixLaunchParams.transforms[entity.transform_id];
            MaterialStruct entityMaterial; LightStruct entityLight;
            if (entity.material_id >= 0 && entity.material_id < MAX_MATERIALS) {
                entityMaterial = optixLaunchParams.materials[entity.material_id];
            }
            
            const float3 w_o = -ray.direction;
            float3 hit_p = ray.origin + payload.tHit * ray.direction;
            float3 mp, p, pt0, pt1, v_x, v_y, v_z, v_gz, p_e1, p_e2; float2 uv, uv_e1, uv_e2; int3 indices;
            bool shouldNormalFaceForward = (entityMaterial.transmission == 0.f);
            
            loadMeshTriIndices(entity.mesh_id, payload.primitiveID, indices);
            loadMeshVertexData(entity.mesh_id, indices, payload.barycentrics, mp, v_gz, p_e1, p_e2);
            loadMeshUVData(entity.mesh_id, indices, payload.barycentrics, uv, uv_e1, uv_e2);
            loadMeshNormalData(entity.mesh_id, indices, payload.barycentrics, uv, v_z);
            loadDisneyMaterial(entityMaterial, make_vec2(uv), mat, roughnessMinimum);
            
            glm::mat4 xfm;
            xfm = glm::column(xfm, 0, vec4(payload.localToWorld[0], payload.localToWorld[4],  payload.localToWorld[8], 0.0f));
            xfm = glm::column(xfm, 1, vec4(payload.localToWorld[1], payload.localToWorld[5],  payload.localToWorld[9], 0.0f));
            xfm = glm::column(xfm, 2, vec4(payload.localToWorld[2], payload.localToWorld[6],  payload.localToWorld[10], 0.0f));
            xfm = glm::column(xfm, 3, vec4(payload.localToWorld[3], payload.localToWorld[7],  payload.localToWorld[11], 1.0f));
            glm::mat4 xfmt0;
            xfmt0 = glm::column(xfmt0, 0, vec4(payload.localToWorldT0[0], payload.localToWorldT0[4],  payload.localToWorldT0[8], 0.0f));
            xfmt0 = glm::column(xfmt0, 1, vec4(payload.localToWorldT0[1], payload.localToWorldT0[5],  payload.localToWorldT0[9], 0.0f));
            xfmt0 = glm::column(xfmt0, 2, vec4(payload.localToWorldT0[2], payload.localToWorldT0[6],  payload.localToWorldT0[10], 0.0f));
            xfmt0 = glm::column(xfmt0, 3, vec4(payload.localToWorldT0[3], payload.localToWorldT0[7],  payload.localToWorldT0[11], 1.0f));
            glm::mat4 xfmt1;
            xfmt1 = glm::column(xfmt1, 0, vec4(payload.localToWorldT1[0], payload.localToWorldT1[4],  payload.localToWorldT1[8], 0.0f));
            xfmt1 = glm::column(xfmt1, 1, vec4(payload.localToWorldT1[1], payload.localToWorldT1[5],  payload.localToWorldT1[9], 0.0f));
            xfmt1 = glm::column(xfmt1, 2, vec4(payload.localToWorldT1[2], payload.localToWorldT1[6],  payload.localToWorldT1[10], 0.0f));
            xfmt1 = glm::column(xfmt1, 3, vec4(payload.localToWorldT1[3], payload.localToWorldT1[7],  payload.localToWorldT1[11], 1.0f));
            glm::mat3 nxfm = transpose(glm::inverse(glm::mat3(xfm)));

            // If the material has a normal map, load it. 
            float f = 1.0f / (uv_e1.x * uv_e2.y - uv_e2.x * uv_e1.y);
            vec3 tangent, binormal;
            tangent.x = f * (uv_e2.y * p_e1.x - uv_e1.y * p_e2.x);
            tangent.y = f * (uv_e2.y * p_e1.y - uv_e1.y * p_e2.y);
            tangent.z = f * (uv_e2.y * p_e1.z - uv_e1.y * p_e2.z);
            tangent = normalize(tangent);
            v_z = normalize(v_z);            
            
            // Transform data into world space
            p = make_float3(xfm * make_vec4(mp, 1.0f));
            vec4 tmp1 = optixLaunchParams.proj * optixLaunchParams.viewT0 * xfmt0 * make_vec4(mp, 1.0f);
            pt0 = make_float3(tmp1 / tmp1.w) * .5f;

            vec4 tmp2 = optixLaunchParams.proj * optixLaunchParams.viewT1 * xfmt1 * make_vec4(mp, 1.0f);
            pt1 = make_float3(tmp2 / tmp2.w) * .5f;
            
            float3 diffuseMotion = pt1 - pt0;
            // diffuseMotion = make_float3(diffuseMotion.x, diffuseMotion.z, diffuseMotion.y);
            if (bounce == 0) primaryDiffuseMotion = diffuseMotion;
            // float3 test = make_float3(abs(make_vec3(diffuseMotion)));
            // test.z = 0.f;
            // illum = test;
            // break;

            hit_p = p;
            v_gz = make_float3(normalize(nxfm * make_vec3(v_gz)));
            v_z = make_float3(normalize(nxfm * make_vec3(v_z)));
            v_x = make_float3(normalize(nxfm * tangent));
            v_y = cross(v_z, v_x);
            v_x = cross(v_y, v_z);

            if (
                all(lessThan(abs(make_vec3(v_x)), vec3(EPSILON))) || 
                all(lessThan(abs(make_vec3(v_y)), vec3(EPSILON))) ||
                any(isnan(make_vec3(v_x))) || 
                any(isnan(make_vec3(v_y)))
            ) {
                ortho_basis(v_x, v_y, v_z);
            }

            glm::mat3 tbn;
            tbn = glm::column(tbn, 0, make_vec3(v_x) );
            tbn = glm::column(tbn, 1, make_vec3(v_y) );
            tbn = glm::column(tbn, 2, make_vec3(v_z) );
            
            float3 dN = make_float3(sampleTexture(entityMaterial.normal_map_texture_id, make_vec2(uv), vec4(0.5f, .5f, 1.f, 0.f)));
            dN = (dN * make_float3(2.0f)) - make_float3(1.f);   

            v_z = make_float3(normalize(tbn * normalize(make_vec3(dN))) );

            if (shouldNormalFaceForward) {
                v_z = faceNormalForward(w_o, v_gz, v_z);
            }

            // For segmentations, metadata extraction for applications like denoising or ML training
            saveRenderData(renderData, bounce, payload.tHit, hit_p, v_z, entityID, diffuseMotion, mat.base_color);
                        
            // If this is the first hit, keep track of primary albedo and normal for denoising.
            if (bounce == 0) {
                primaryNormal = v_z;
                primaryAlbedo = mat.base_color;
            }

            // If the entity we hit is a light, terminate the path.
            // First hits are colored by the light. All other light hits are handled by NEE/MIS 
            if (entity.light_id >= 0 && entity.light_id < MAX_LIGHTS) {
                if (bounce == 0) 
                {
                    entityLight = optixLaunchParams.lights[entity.light_id];
                    float3 light_emission;
                    if (entityLight.color_texture_id == -1) light_emission = make_float3(entityLight.r, entityLight.g, entityLight.b) * entityLight.intensity;
                    else light_emission = make_float3(sampleTexture(entityLight.color_texture_id, make_vec2(uv), vec4(entityLight.r, entityLight.g, entityLight.b, 1.f))); // * intensity; temporarily commenting out to show texture for bright lights in LDR
                    illum = light_emission; 
                    directIllum = illum;
                }
                break;
            }
                        
            // Sample a light source
            uint32_t sampledLightID = -1;
            int numLights = optixLaunchParams.numLightEntities;
            // float3 lightEmission = make_float3(0.f);
            float3 irradiance = make_float3(0.f);
            float light_pdf = 0.f;

            EntityStruct light_entity;
            MaterialStruct light_material;
            LightStruct light_light;
            light_material.base_color_texture_id = -1;
            
            float3 n_l = v_z; //faceNormalForward(w_o, v_gz, v_z);
            // if (dot(w_o, n_l) < 0.f) {
                // n_l = -n_l;
            // }

            // first, sample the light source by importance sampling the light
            do {
                if (numLights == 0) break;

                // Disable NEE for transmission events 
                // bool entering = dot(w_o, v_z) < 0.f;
                // if (entering) break;

                
                uint32_t random_id = uint32_t(min(lcg_randomf(rng) * numLights, float(numLights - 1)));
                random_id = min(random_id, numLights - 1);
                sampledLightID = optixLaunchParams.lightEntities[random_id];
                light_entity = optixLaunchParams.entities[sampledLightID];
                
                // shouldn't happen, but just in case...
                if ((light_entity.light_id < 0) || (light_entity.light_id > MAX_LIGHTS)) break;
                if ((light_entity.transform_id < 0) || (light_entity.transform_id > MAX_TRANSFORMS)) break;
            
                light_light = optixLaunchParams.lights[light_entity.light_id];
                TransformStruct transform = optixLaunchParams.transforms[light_entity.transform_id];
                MeshStruct mesh;
                
                bool is_area_light = false;
                if ((light_entity.mesh_id >= 0) && (light_entity.mesh_id < MAX_MESHES)) {
                    mesh = optixLaunchParams.meshes[light_entity.mesh_id];
                    is_area_light = true;
                };

                const uint32_t occlusion_flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
                    // | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;
                    // | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
            
                if (!is_area_light) break;

                uint32_t random_tri_id = uint32_t(min(lcg_randomf(rng) * mesh.numTris, float(mesh.numTris - 1)));
                owl::device::Buffer *indexLists = (owl::device::Buffer *)optixLaunchParams.indexLists.data;
                ivec3 *indices = (ivec3*) indexLists[light_entity.mesh_id].data;
                ivec3 triIndex = indices[random_tri_id];   
                
                // Sample the light to compute an incident light ray to this point
                {    
                    owl::device::Buffer *vertexLists = (owl::device::Buffer *)optixLaunchParams.vertexLists.data;
                    owl::device::Buffer *texCoordLists = (owl::device::Buffer *)optixLaunchParams.texCoordLists.data;
                    vec4 *vertices = (vec4*) vertexLists[light_entity.mesh_id].data;
                    vec2 *texCoords = (vec2*) texCoordLists[light_entity.mesh_id].data;
                    vec3 dir; 
                    vec2 uv;
                    vec3 pos = vec3(hit_p.x, hit_p.y, hit_p.z);
                    vec3 v1 = transform.localToWorld * vertices[triIndex.x];
                    vec3 v2 = transform.localToWorld * vertices[triIndex.y];
                    vec3 v3 = transform.localToWorld * vertices[triIndex.z];
                    vec2 uv1 = texCoords[triIndex.x];
                    vec2 uv2 = texCoords[triIndex.y];
                    vec2 uv3 = texCoords[triIndex.z];
                    vec3 N = normalize(cross( normalize(v2 - v1), normalize(v3 - v1)));
                    sampleTriangle(pos, N, v1, v2, v3, uv1, uv2, uv3, lcg_randomf(rng), lcg_randomf(rng), dir, light_pdf, uv);
                    vec3 normal = glm::vec3(n_l.x, n_l.y, n_l.z);
                    float dotNWi = fabs(dot(dir, normal)); // for now, making all lights double sided.
                    light_pdf = abs(light_pdf);
                    
                    float4 default_light_emission = make_float4(light_light.r, light_light.g, light_light.b, 0.f);
                    float3 lightEmission = make_float3(sampleTexture(light_light.color_texture_id, uv, make_vec4(default_light_emission))) * light_light.intensity;
        
                    if ((light_pdf > EPSILON) && (dotNWi > EPSILON)) {
                        float3 light_dir = make_float3(dir.x, dir.y, dir.z);
                        light_dir = normalize(light_dir);
                        float bsdf_pdf = disney_pdf(mat, n_l, w_o, light_dir, v_x, v_y);
                        if (bsdf_pdf > EPSILON) {
                            RayPayload payload;
                            owl::Ray ray;
                            ray.tmin = EPSILON * 10.f;
                            ray.tmax = 1e20f;
                            ray.origin = hit_p;
                            ray.direction = light_dir;
                            payload.tHit = -1.f;
                            ray.time = sampleTime(lcg_randomf(rng));
                            owl::traceRay( optixLaunchParams.world, ray, payload, occlusion_flags);
                            if (payload.instanceID == -1) continue;
                            int entityID = optixLaunchParams.instanceToEntityMap[payload.instanceID];
                            bool visible = ((entityID == sampledLightID) || (entityID == -1));
                            if (visible) {
                                float w = power_heuristic(1.f, light_pdf, 1.f, bsdf_pdf);
                                float3 bsdf = disney_brdf(mat, n_l, w_o, light_dir, v_x, v_y, optixLaunchParams.GGX_E_LOOKUP, optixLaunchParams.GGX_E_AVG_LOOKUP);
                                float3 Li = lightEmission * w / light_pdf;
                                irradiance = (bsdf * Li * fabs(dotNWi));
                            }
                        }
                    }
                }
            } while (false);

            // next, sample a light source by importance sampling the BDRF
            float3 w_i;
            float bsdf_pdf;
            bool sampledSpecular;
            float3 bsdf = sample_disney_brdf(mat, v_z, w_o, v_x, v_y, rng, w_i, bsdf_pdf, sampledSpecular, optixLaunchParams.GGX_E_LOOKUP, optixLaunchParams.GGX_E_AVG_LOOKUP);
            if (bsdf_pdf < EPSILON || all_zero(bsdf)) {
                break;
            }

            // trace the next ray along that sampled BRDF direction
            if (dot(w_o, v_z) < 0.f) {
                v_z = -v_z;
            }
            ray.origin = hit_p;// + v_z * .1;
            ray.direction = w_i;
            ray.tmin = EPSILON * 100.f;
            payload.tHit = -1.f;
            ray.time = sampleTime(lcg_randomf(rng));
            owl::traceRay(optixLaunchParams.world, ray, payload);

            if (light_pdf > EPSILON) 
            {
                // if by sampling the brdf we also hit the light source...
                if (payload.instanceID == -1) continue;
                int entityID = optixLaunchParams.instanceToEntityMap[payload.instanceID];
                bool visible = (entityID == sampledLightID);
                if (visible) {
                    int3 indices; float3 p, p_e1, p_e2; float3 v_gz; float2 uv, uv_e1, uv_e2;
                    loadMeshTriIndices(light_entity.mesh_id, payload.primitiveID, indices);
                    loadMeshVertexData(light_entity.mesh_id, indices, payload.barycentrics, p, v_gz, p_e1, p_e2);
                    loadMeshUVData(light_entity.mesh_id, indices, payload.barycentrics, uv, uv_e1, uv_e2);

                    // Transform data into world space
                    glm::mat4 xfm;
                    xfm = glm::column(xfm, 0, vec4(payload.localToWorld[0], payload.localToWorld[4],  payload.localToWorld[8], 0.0f));
                    xfm = glm::column(xfm, 1, vec4(payload.localToWorld[1], payload.localToWorld[5],  payload.localToWorld[9], 0.0f));
                    xfm = glm::column(xfm, 2, vec4(payload.localToWorld[2], payload.localToWorld[6],  payload.localToWorld[10], 0.0f));
                    xfm = glm::column(xfm, 3, vec4(payload.localToWorld[3], payload.localToWorld[7],  payload.localToWorld[11], 1.0f));
                    glm::mat3 nxfm = transpose(glm::inverse(glm::mat3(xfm)));
                    p = make_float3(xfm * make_vec4(p, 1.0f));
                    v_gz = make_float3(normalize(nxfm * normalize(make_vec3(v_gz))));

                    float4 default_light_emission = make_float4(light_light.r, light_light.g, light_light.b, 0.f);
                    float3 lightEmission = make_float3(sampleTexture(light_light.color_texture_id, make_vec2(uv), make_vec4(default_light_emission))) * light_light.intensity;

                    float dist = distance(vec3(p.x, p.y, p.z), vec3(ray.origin.x, ray.origin.y, ray.origin.z)); // should I be using this?
                    float dotNWi = fabs(dot(-v_gz, ray.direction)); // for now, making all lights double sided.
                    if (dotNWi > 0.f){
                        float w = power_heuristic(1.f, bsdf_pdf, 1.f, light_pdf);
                        float3 Li = lightEmission * w / bsdf_pdf;
                        irradiance = irradiance + (bsdf * Li * fabs(dotNWi)); // missing r^2 falloff?
                    }
                }
            }

            // accumulate any radiance (ie path_throughput * irradiance), and update the path throughput using the sampled BRDF
            illum = illum + path_throughput * irradiance;
            path_throughput = path_throughput * bsdf / bsdf_pdf;

            // If ray misses, interpret normal as "miss color" assigned by miss program and move on to the next sample
            if (payload.tHit <= 0.f) {
                illum = illum + path_throughput * missColor(ray) * optixLaunchParams.domeLightIntensity;
            }
            
            if (bounce == 0) {
                directIllum = illum;
            }

            if ((payload.tHit <= 0.0f) || (path_throughput.x < EPSILON && path_throughput.y < EPSILON && path_throughput.z < EPSILON)) {
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
        } while (bounce < optixLaunchParams.maxBounceDepth);

        // clamp out any extreme fireflies
        glm::vec3 gillum = vec3(illum.x, illum.y, illum.z);
        glm::vec3 dillum = vec3(directIllum.x, directIllum.y, directIllum.z);
        glm::vec3 iillum = gillum - dillum;

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
        accum_illum = accum_illum + illum;
    }
    accum_illum = accum_illum / float(SPP);


    /* Write to AOVs, progressively refining results */
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
    
    vec4 oldAlbedo = optixLaunchParams.albedoBuffer[fbOfs];
    vec4 oldNormal = optixLaunchParams.normalBuffer[fbOfs];
    if (any(isnan(oldAlbedo))) oldAlbedo = vec4(1.f);
    if (any(isnan(oldNormal))) oldNormal = vec4(1.f);
    vec4 newAlbedo = vec4(primaryAlbedo.x, primaryAlbedo.y, primaryAlbedo.z, 1.f);
    vec4 newNormal = normalize(VP * vec4(primaryNormal.x, primaryNormal.y, primaryNormal.z, 0.f));
    newNormal.a = 1.f;
    vec4 accumAlbedo = (newAlbedo + float(optixLaunchParams.frameID) * oldAlbedo) / float(optixLaunchParams.frameID + 1);
    vec4 accumNormal = (newNormal + float(optixLaunchParams.frameID) * oldNormal) / float(optixLaunchParams.frameID + 1);
    optixLaunchParams.albedoBuffer[fbOfs] = accumAlbedo;
    optixLaunchParams.normalBuffer[fbOfs] = accumNormal;

    // Override framebuffer output if user requested to render metadata
    if (optixLaunchParams.renderDataMode != RenderDataFlags::NONE) {
        optixLaunchParams.frameBuffer[fbOfs] = vec4( renderData.x, renderData.y, renderData.z, 1.0f);
    }
}
