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
    uint32_t entityID;
    float2 uv;
    float tHit;
    float3 normal;
    float3 gnormal;
    // float pad;
};

inline __device__
float3 missColor(const owl::Ray &ray)
{
  auto pixelID = owl::getLaunchIndex();

  float3 rayDir = normalize(ray.direction);
  float t = 0.5f*(rayDir.z + 1.0f);
  float3 c = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
  return c;
}

OPTIX_MISS_PROGRAM(miss)()
{
    RayPayload &payload = get_payload<RayPayload>();
    payload.tHit = -1.f;
    payload.entityID = -1;
    owl::Ray ray;
    ray.direction = optixGetWorldRayDirection();
    payload.normal = missColor(ray) * optixLaunchParams.domeLightIntensity;
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
    const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
    
    const float2 bc    = optixGetTriangleBarycentrics();
    const int instID   = optixGetInstanceIndex();
    const int primID   = optixGetPrimitiveIndex();
    const int entityID = optixLaunchParams.instanceToEntityMap[instID];
    const ivec3 index  = self.index[primID];
    
    // compute position: (actually not needed. implicit via tMax )
    // vec3 V;
    // {
    //     const vec3 &A      = self.vertex[index.x];
    //     const vec3 &B      = self.vertex[index.y];
    //     const vec3 &C      = self.vertex[index.z];
    //     V = A * (1.f - (bc.x + bc.y)) + B * bc.x + C * bc.y;
    // }

    // compute normal:
    float3 N, GN;

    const float3 &A      = (float3&) self.vertex[index.x];
    const float3 &B      = (float3&) self.vertex[index.y];
    const float3 &C      = (float3&) self.vertex[index.z];
    GN = normalize(cross(B-A,C-A));
    
    if (self.normals) {
        const float3 &A = (float3&) self.normals[index.x];
        const float3 &B = (float3&) self.normals[index.y];
        const float3 &C = (float3&) self.normals[index.z];
        N = normalize(A * (1.f - (bc.x + bc.y)) + B * bc.x + C * bc.y);
    } else {
        N = GN;
    }

    GN = normalize(optixTransformNormalFromObjectToWorldSpace(GN));
    N = normalize(optixTransformNormalFromObjectToWorldSpace(N));
    // normalize(transpose(mat3(gl_WorldToObjectNV)) * payload.m_n);
    // N  = normalize(transpose(mat3(gl_WorldToObjectNV)) * payload.m_n);

    // compute uv:
    float2 UV;
    if (self.texcoords) {
        const float2 &A = (float2&) self.texcoords[index.x];
        const float2 &B = (float2&) self.texcoords[index.y];
        const float2 &C = (float2&) self.texcoords[index.z];
        UV = A * (1.f - (bc.x + bc.y)) + B * bc.x + C * bc.y;
    } else {
        UV = bc;
    }

    // store data in payload
    RayPayload &prd = owl::getPRD<RayPayload>();
    prd.entityID = entityID;
    prd.uv = UV;
    prd.tHit = optixGetRayTmax();
    prd.normal = N;
    prd.gnormal = GN;
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

__device__ 
void loadMaterial(const MaterialStruct &p, float2 uv, DisneyMaterial &mat) {

    // uint32_t mask = __float_as_int(p.base_color.x);
    // if (IS_TEXTURED_PARAM(mask)) {
    //     const uint32_t tex_id = GET_TEXTURE_ID(mask);
    //     mat.base_color = make_float3(tex2D<float4>(launch_params.textures[tex_id], uv.x, uv.y));
    // } else {
        mat.base_color = make_float3(p.base_color.x, p.base_color.y, p.base_color.z);
    // }

    mat.metallic = /*textured_scalar_param(*/p.metallic/*, uv)*/;
    mat.specular = /*textured_scalar_param(*/p.specular/*, uv)*/;
    mat.roughness = /*textured_scalar_param(*/p.roughness/*, uv)*/;
    mat.specular_tint = /*textured_scalar_param(*/p.specular_tint/*, uv)*/;
    mat.anisotropy = /*textured_scalar_param(*/p.anisotropic/*, uv)*/;
    mat.sheen = /*textured_scalar_param(*/p.sheen/*, uv)*/;
    mat.sheen_tint = /*textured_scalar_param(*/p.sheen_tint/*, uv)*/;
    mat.clearcoat = /*textured_scalar_param(*/p.clearcoat/*, uv)*/;
    mat.clearcoat_gloss = /*textured_scalar_param(*/1.0 - p.clearcoat_roughness/*, uv)*/;
    mat.ior = /*textured_scalar_param(*/p.ior/*, uv)*/;
    mat.specular_transmission = /*textured_scalar_param(*/p.transmission/*, uv)*/;
    mat.flatness = p.subsurface;
}

inline __device__
owl::Ray generateRay(const CameraStruct &camera, const TransformStruct &transform, ivec2 pixelID, ivec2 frameSize, LCGRand &rng)
{
    /* Generate camera rays */    
    mat4 camWorldToLocal = transform.worldToLocal;
    mat4 projinv = camera.projinv;//glm::inverse(glm::perspective(.785398, 1.0, .1, 1000));//camera.projinv;
    mat4 viewinv = camera.viewinv * camWorldToLocal;
    vec2 aa = vec2(lcg_randomf(rng),lcg_randomf(rng)) - vec2(.5f,.5f);
    vec2 inUV = (vec2(pixelID.x, pixelID.y) + aa) / vec2(optixLaunchParams.frameSize);
    vec3 right = normalize(glm::vec3(viewinv[0]));
    vec3 up = normalize(glm::vec3(viewinv[1]));
    
    float cameraLensRadius = camera.apertureDiameter;

    vec3 p(0.f);
    if (cameraLensRadius > 0.0) {
        do {
            p = 2.0f*vec3(lcg_randomf(rng),lcg_randomf(rng),0.f) - vec3(1.f,1.f,0.f);
        } while (dot(p,p) >= 1.0f);
    }

    vec3 rd = cameraLensRadius * p;
    vec3 lens_offset = (right * rd.x) / float(frameSize.x) + (up * rd.y) / float(frameSize.y);

    vec3 origin = vec3(viewinv * vec4(0.f,0.f,0.f,1.f)) + lens_offset;
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

__device__ float3 sample_direct_light(const DisneyMaterial &mat, const float3 &hit_p,
    const float3 &n, const float3 &v_x, const float3 &v_y, const float3 &w_o,
    const LightStruct *lights, const EntityStruct *entities, const TransformStruct *transforms,
    const uint32_t* light_entities, const uint32_t num_lights, 
    uint16_t &ray_count, LCGRand &rng)
{
    float3 illum = make_float3(0.f);
    
    if (num_lights == 0) return illum;

    uint32_t random_id = lcg_randomf(rng) * num_lights;
    random_id = min(random_id, num_lights - 1);
    uint32_t light_entity_id = light_entities[random_id];
    EntityStruct light_entity = entities[light_entity_id];
    
    // shouldn't happen, but just in case...
    if ((light_entity.light_id < 0) || (light_entity.light_id > MAX_LIGHTS)) return illum;
    if ((light_entity.transform_id < 0) || (light_entity.transform_id > MAX_LIGHTS)) return illum;
    
    LightStruct light = lights[light_entity.light_id];
    TransformStruct transform = transforms[light_entity.transform_id];
    float3 light_emission = make_float3(light.r, light.g, light.b) * light.intensity;

    const uint32_t occlusion_flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT
        | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
        | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;

    // Sample the light to compute an incident light ray to this point
    {
        float3 light_pos = make_float3(
            transform.localToWorld[3][0], 
            transform.localToWorld[3][1], 
            transform.localToWorld[3][2]);
                // sample_quad_light_position(light,
                // make_float2(lcg_randomf(rng), lcg_randomf(rng)));
        float3 light_dir = light_pos - hit_p;
        float light_dist = length(light_dir);
        light_dir = normalize(light_dir);

        float light_pdf = 1.f; //quad_light_pdf(light, light_pos, hit_p, light_dir);
        float bsdf_pdf = disney_pdf(mat, n, w_o, light_dir, v_x, v_y);

        // uint32_t shadow_hit = 1;
        RayPayload payload;
        payload.entityID = light_entity_id;
        owl::Ray ray;
        ray.tmin = EPSILON * 10.f;
        ray.tmax = light_dist;
        ray.origin = owl::vec3f(hit_p.x, hit_p.y, hit_p.z) ;
        ray.direction = owl::vec3f(light_dir.x, light_dir.y, light_dir.z);
        owl::traceRay(  /*accel to trace against*/ optixLaunchParams.world,
                        /*the ray to trace*/ ray,
                        /*prd*/ payload,
                        occlusion_flags);
                            
    //     optixTrace(launch_params.scene, hit_p, light_dir, EPSILON, light_dist, 0.f,
    //             0xff, occlusion_flags, PRIMARY_RAY, 1, OCCLUSION_RAY,
    //             shadow_hit);
    // #ifdef REPORT_RAY_STATS
    //     ++ray_count;
    // #endif
        if (light_pdf >= EPSILON && bsdf_pdf >= EPSILON && payload.entityID != light_entity_id) {
            float3 bsdf = disney_brdf(mat, n, w_o, light_dir, v_x, v_y);
            float w = 1.f; // power_heuristic(1.f, light_pdf, 1.f, bsdf_pdf); // NOTE: TEMPORARILY DIABLING POWER HEURISTIC SINCE FOR NOW ONLY DOING POINT LIGHTS
            illum = bsdf * light_emission * fabs(dot(light_dir, n)) * w / light_pdf;
        }
    }

    // Sample the BRDF to compute a light sample as well
    // {
    //     float3 w_i;
    //     float bsdf_pdf;
    //     float3 bsdf = sample_disney_brdf(mat, n, w_o, v_x, v_y, rng, w_i, bsdf_pdf);

    //     float light_dist;
    //     float3 light_pos;
    //     if (!all_zero(bsdf) && bsdf_pdf >= EPSILON && quad_intersect(light, hit_p, w_i, light_dist, light_pos)) {
    //         float light_pdf = quad_light_pdf(light, light_pos, hit_p, w_i);
    //         if (light_pdf >= EPSILON) {
    //             float w = power_heuristic(1.f, bsdf_pdf, 1.f, light_pdf);
    //             uint32_t shadow_hit = 1;
    //             optixTrace(launch_params.scene, hit_p, w_i, EPSILON, light_dist, 0.f,
    //                     0xff, occlusion_flags, PRIMARY_RAY, 1, OCCLUSION_RAY,
    //                     shadow_hit);
    // // #ifdef REPORT_RAY_STATS
    // //             ++ray_count;
    // // #endif
    //             if (!shadow_hit) {
    //                 illum = illum + bsdf * light_emission * fabs(dot(w_i, n)) * w / bsdf_pdf;
    //             }
    //         }
    //     }
    // }
    return illum;
}

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
    auto pixelID = ivec2(owl::getLaunchIndex()[0], owl::getLaunchIndex()[1]);
    auto fbOfs = pixelID.x+optixLaunchParams.frameSize.x* ((optixLaunchParams.frameSize.y - 1) -  pixelID.y);
    LCGRand rng = get_rng(optixLaunchParams.frameID);

    EntityStruct    camera_entity;
    TransformStruct camera_transform;
    CameraStruct    camera;
    if (!loadCamera(camera_entity, camera, camera_transform)) {
        optixLaunchParams.fbPtr[fbOfs] = vec4(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng), 1.f);
        return;
    }


    float3 accum_illum = make_float3(0.f);
    #define SPP 4
    for (uint32_t rid = 0; rid < SPP; ++rid) {

        owl::Ray ray = generateRay(camera, camera_transform, pixelID, optixLaunchParams.frameSize, rng);

        DisneyMaterial mat;
        int bounce = 0;
        float3 illum = make_float3(0.f);
        float3 path_throughput = make_float3(1.f);
        uint16_t ray_count = 0;

        do {
            RayPayload payload;
            owl::traceRay(  /*accel to trace against*/ optixLaunchParams.world,
                            /*the ray to trace*/ ray,
                            /*prd*/ payload);
            #ifdef REPORT_RAY_STATS
                ++ray_count;
            #endif

            // if ray misses, interpret normal as "miss color" assigned by miss program
            if (payload.tHit <= 0.f) {
                illum = illum + path_throughput * payload.normal;
                break;
            }

            EntityStruct entity = optixLaunchParams.entities[payload.entityID];
            MaterialStruct entityMaterial = optixLaunchParams.materials[entity.material_id];
            TransformStruct entityTransform = optixLaunchParams.transforms[entity.transform_id];
            loadMaterial(entityMaterial, payload.uv, mat);

            const float3 w_o = -ray.direction;
            const float3 hit_p = ray.origin + payload.tHit * ray.direction;
            float3 v_x, v_y;
            float3 v_z = payload.normal;
            float3 v_gz = payload.gnormal;
            if (mat.specular_transmission == 0.f && dot(w_o, v_z) < 0.f) {
                // prevents differences from geometric and shading normal from creating black artifacts
                v_z = reflect(-v_z, v_gz); 
            }
            if (mat.specular_transmission == 0.f && dot(w_o, v_z) < 0.f) {
                v_z = -v_z;
            }
            ortho_basis(v_x, v_y, v_z);

            illum = illum + path_throughput * 
                sample_direct_light(mat, hit_p, v_z, v_x, v_y, w_o,
                    optixLaunchParams.lights, 
                    optixLaunchParams.entities, 
                    optixLaunchParams.transforms, 
                    optixLaunchParams.lightEntities,
                    optixLaunchParams.numLightEntities, 
                    ray_count, rng);

            float3 w_i;
            float pdf;
            float3 bsdf = sample_disney_brdf(mat, v_z, w_o, v_x, v_y, rng, w_i, pdf);
            if (pdf < EPSILON || all_zero(bsdf)) {
                break;
            }
            path_throughput = path_throughput * bsdf / pdf;

            if (path_throughput.x < EPSILON && path_throughput.y < EPSILON && path_throughput.z < EPSILON) {
                break;
            }

            // vec3 offset = payload.normal * .001f;
            ray.origin = hit_p;// + make_float3(offset.x, offset.y, offset.z);
            ray.direction = w_i;

            ++bounce;

            // if (tprd.tHit > 0.f) {
            //     finalColor = vec3(tprd.normal.x, tprd.normal.y, tprd.normal.z);
            // }
        } while (bounce < MAX_PATH_DEPTH);
        accum_illum = accum_illum + illum;
    }
    accum_illum = accum_illum / float(SPP);


    // finalColor = vec3(ray.direction.x, ray.direction.y, ray.direction.z);
    /* Write AOVs */
    float4 &prev_color = (float4&) optixLaunchParams.accumPtr[fbOfs];
    float4 accum_color = make_float4((accum_illum + float(optixLaunchParams.frameID) * make_float3(prev_color)) / float(optixLaunchParams.frameID + 1), 1.0f);
    optixLaunchParams.accumPtr[fbOfs] = vec4(
        accum_color.x, 
        accum_color.y, 
        accum_color.z, 
        accum_color.w
    );
    optixLaunchParams.fbPtr[fbOfs] = vec4(
        linear_to_srgb(accum_color.x),
        linear_to_srgb(accum_color.y),
        linear_to_srgb(accum_color.z),
        1.0f
    );
}

