#include "deviceCode.h"
#include "launchParams.h"
#include <optix_device.h>

extern "C" __constant__ LaunchParams optixLaunchParams;

struct TriMeshPayload {
    float r = -1.f, g = -1.f, b = -1.f, tmax = -1.f;
};
OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
    TriMeshPayload &prd = owl::getPRD<TriMeshPayload>();

    const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
    float2 bc = optixGetTriangleBarycentrics();

    // compute normal:
    const int   primID = optixGetPrimitiveIndex();
    const ivec3 index  = self.index[primID];
    const vec3 &A     = self.vertex[index.x];
    const vec3 &B     = self.vertex[index.y];
    const vec3 &C     = self.vertex[index.z];
    const vec3 &ACol = vec3(1.0, 0.0, 0.0); ///(self.colors == nullptr) ? vec3(optixLaunchParams.tri_mesh_color) : self.colors[index.x];
    const vec3 &BCol = vec3(1.0, 0.0, 0.0); ///(self.colors == nullptr) ? vec3(optixLaunchParams.tri_mesh_color) : self.colors[index.y];
    const vec3 &CCol = vec3(1.0, 0.0, 0.0); ///(self.colors == nullptr) ? vec3(optixLaunchParams.tri_mesh_color) : self.colors[index.z];
    const vec3 Ng     = normalize(cross(B-A,C-A));

    auto rayDir = optixGetWorldRayDirection();
    vec3 dir = vec3(rayDir.x, rayDir.y, rayDir.z);

    vec3 vcol = ACol * (1.f - (bc.x + bc.y)) + BCol * bc.x + CCol * bc.y;

    vec3 color = (.2f + .8f*fabs(dot(dir,Ng)))*vcol;
    prd.r = color.x;
    prd.g = color.y;
    prd.b = color.z;
    prd.tmax = optixGetRayTmax();
}

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
    auto pixelID = owl::getLaunchIndex();
    auto fbOfs = pixelID.x+optixLaunchParams.frameSize.x* ((optixLaunchParams.frameSize.y - 1) -  pixelID.y);
    optixLaunchParams.fbPtr[fbOfs] = glm::vec4(1.0, 0.0, 1.0, 1.f);
}

OPTIX_MISS_PROGRAM(miss)()
{

}

