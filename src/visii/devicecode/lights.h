#pragma once

#include "cuda_utils.h"

// Quad-shaped light source
struct QuadLight {
	float3 emission;
	float pad1;

	float3 position;
	float pad2;

	float3 normal;
	float pad3;

	float3 v_x;
	float width;

	float3 v_y;
	float height;
};

__device__ vec4 sample_quad_light_position(vec4 q0, vec4 q1, vec4 q2, vec4 q3, float2 samples) 
{
	vec4 t0 = glm::mix(q0,q1, samples.x);
	vec4 t1 = glm::mix(q3,q2, samples.x);
	return mix(t0, t1, samples.y);
	
	// return samples.x * light.v_x * light.width
		// + samples.y * light.v_y * light.height + light.position;
}

/* Compute the PDF of sampling the sampled point p light with the ray specified by orig and dir,
 * assuming the light is not occluded
 */
__device__ float quad_light_pdf(
	vec4 q0, vec4 q1, vec4 q2, vec4 q3,
	const float3 &p, const float3 &orig, const float3 &dir) 
{
	float width = glm::distance(q0, q1);
	float height = glm::distance(q0, q2);
	// vec3 normal = glm::normalize(glm::cross( glm::normalize(glm::vec3(q1 - q0)), glm::normalize(glm::vec3(q2 - q0)) ));
	float surface_area = width * height;
	float3 to_pt = p - dir;
	float dist_sqr = dot(to_pt, to_pt);
	// float n_dot_w = abs(dot(normal, -glm::vec3(dir.x, dir.y, dir.z)));
	// if (n_dot_w < EPSILON) {
		// return 0.f;
	// }
	return dist_sqr / ( /*n_dot_w **/ surface_area);
}

__device__ bool quad_intersect(const QuadLight &light, const float3 &orig, const float3 &dir,
	float &t, float3 &light_pos)
{
	float denom = dot(dir, light.normal);
	if (denom >= EPSILON) {
		t = dot(light.position - orig, light.normal) / denom;
		if (t < 0.f) {
			return false;
		}

		// It's a finite plane so now see if the hit point is actually inside the plane
		light_pos = orig + dir * t;
		float3 hit_v = light_pos - light.position;
		if (fabs(dot(hit_v, light.v_x)) < light.width && fabs(dot(hit_v, light.v_y)) < light.height) {
			return true;
		}
	}
	return false;
}

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;
struct SphQuad {
    vec3 o, x, y, z; // local reference system ’R’
    float z0, z0sq; //
    float x0, y0, y0sq; // rectangle coords in ’R’
    float x1, y1, y1sq; //
    float b0, b1, b0sq, k; // misc precomputed constants
    float S; // solid angle of ’Q’
};

__device__
void SphQuadInit(vec3 s, vec3 ex, vec3 ey, vec3 o, SphQuad &squad) {
    squad.o = o;
    float exl = length(ex), eyl = length(ey);
    // compute local reference system ’R’
    squad.x = ex / exl;
    squad.y = ey / eyl;
    squad.z = cross(squad.x, squad.y);
    // compute rectangle coords in local reference system
    vec3 d = s - o;
    squad.z0 = dot(d, squad.z);
    // flip ’z’ to make it point against ’Q’
    if (squad.z0 > 0.) {
		squad.z *= -1.;
		squad.z0 *= -1.;
    }
    squad.z0sq = squad.z0 * squad.z0;
    squad.x0 = dot(d, squad.x);
    squad.y0 = dot(d, squad.y);
    squad.x1 = squad.x0 + exl;
    squad.y1 = squad.y0 + eyl;
    squad.y0sq = squad.y0 * squad.y0;
    squad.y1sq = squad.y1 * squad.y1;
    // create vectors to four vertices
    vec3 v00 = vec3(squad.x0, squad.y0, squad.z0);
    vec3 v01 = vec3(squad.x0, squad.y1, squad.z0);
    vec3 v10 = vec3(squad.x1, squad.y0, squad.z0);
    vec3 v11 = vec3(squad.x1, squad.y1, squad.z0);
    // compute normals to edges
    vec3 n0 = normalize(cross(v00, v10));
    vec3 n1 = normalize(cross(v10, v11));
    vec3 n2 = normalize(cross(v11, v01));
    vec3 n3 = normalize(cross(v01, v00));
    // compute internal angles (gamma_i)
    float g0 = acos(-dot(n0,n1));
    float g1 = acos(-dot(n1,n2));
    float g2 = acos(-dot(n2,n3));
    float g3 = acos(-dot(n3,n0));
    // compute predefined constants
    squad.b0 = n0.z;
    squad.b1 = n2.z;
    squad.b0sq = squad.b0 * squad.b0;
    squad.k = 2.*M_PI - g2 - g3;
    // compute solid angle from internal angles
    squad.S = g0 + g1 - squad.k;
}

__device__
void SphQuadSample(vec3 x, SphQuad squad, float u, float v, vec3 &w, float &pdf) {
    // 1. compute ’cu’
    float au = u * squad.S + squad.k;
    float fu = (cos(au) * squad.b0 - squad.b1) / sin(au);
    float cu = 1./sqrt(fu*fu + squad.b0sq) * (fu>0. ? +1. : -1.);
    cu = glm::clamp(cu, -1.f, 1.f); // avoid NaNs
    // 2. compute ’xu’
    float xu = -(cu * squad.z0) / sqrt(1. - cu*cu);
    xu = glm::clamp(xu, squad.x0, squad.x1); // avoid Infs
    // 3. compute ’yv’
    float d = sqrt(xu*xu + squad.z0sq);
    float h0 = squad.y0 / sqrt(d*d + squad.y0sq);
    float h1 = squad.y1 / sqrt(d*d + squad.y1sq);
    float hv = h0 + v * (h1-h0), hv2 = hv*hv;
    float yv = (hv2 < 1.-EPSILON) ? (hv*d)/sqrt(1.-hv2) : squad.y1;
    // 4. transform (xu,yv,z0) to world coords
    vec3 p = (squad.o + xu*squad.x + yv*squad.y + squad.z0*squad.z);
    w = normalize(p - x);
    pdf = 1. / squad.S;
}

__device__
void sampleDirectLight( vec3 pos,
                       	vec3 normal,
                        float rand1,
                        float rand2, 
                        float rand3, 
						mat4 lightTransform,
						mat4 lightTransformInv,
						vec3 bbmin,
						vec3 bbmax,
                       	vec3 &dir,
                       	float &pdf ) 
{	
	// pick one of the three AABB orientations to sample from.
	// glm::mat4 rotation, rotationInv;
	// ivec4 q[2];
	
	// vec3 p[8] = {
	// 	vec3(bbmin.x, bbmin.y, bbmin.z),
	// 	vec3(bbmax.x, bbmin.y, bbmin.z),
	// 	vec3(bbmin.x, bbmax.y, bbmin.z),
	// 	vec3(bbmax.x, bbmax.y, bbmin.z),
	// 	vec3(bbmin.x, bbmin.y, bbmax.z),
	// 	vec3(bbmax.x, bbmin.y, bbmax.z),
	// 	vec3(bbmin.x, bbmax.y, bbmax.z),
	// 	vec3(bbmax.x, bbmax.y, bbmax.z)
	// };

	pos = vec3( lightTransformInv * vec4(pos, 1.0) );
	bool minCloser = (distance(bbmin , pos) < distance(bbmax , pos));

	float nPlanes = 6;

	vec3 e1, e2;
	rand3 *= nPlanes;
	vec3 s;
	if (0.f <= rand3 && rand3 < 1.f) 
	{
		e1 = vec3(0., bbmax.y - bbmin.y, 0.);
		e2 = vec3(0., 0., bbmax.z - bbmin.z);
		s = bbmin;
		{
			vec3 n = cross(normalize(e1), normalize(e2));
			if (abs(dot(n, pos - s)) < EPSILON) {
				return;
			}
		}
	}
	if (1.f <= rand3 && rand3 < 2.f) 
	{
		e1 = vec3(bbmax.x - bbmin.x, 0., 0.);
		e2 = vec3(0., 0., bbmax.z - bbmin.z);
		s = bbmin;
		{
			vec3 n = cross(normalize(e1), normalize(e2));
			if (abs(dot(n, pos - s)) < EPSILON) {
				return;
			}
		}
	}
	if (2.f <= rand3 && rand3 < 3.f) 
	{
		e1 = vec3(bbmax.x - bbmin.x, 0., 0.);
		e2 = vec3(0., bbmax.y - bbmin.y, 0.);
		s = bbmin;
		{
			vec3 n = cross(normalize(e1), normalize(e2));
			if (abs(dot(n, pos - s)) < EPSILON) {
				return;
			}
		}
	}
	if (3.f <= rand3 && rand3 < 4.f) 
	{
		e1 = -vec3(0., bbmax.y - bbmin.y, 0.);
		e2 = -vec3(0., 0., bbmax.z - bbmin.z);
		s = bbmax;
		{
			vec3 n = cross(normalize(e1), normalize(e2));
			if (abs(dot(n, pos - s)) < EPSILON) {
				return;
			}
		}
	}
	if (4.f <= rand3 && rand3 < 5.f) 
	{
		e1 = -vec3(bbmax.x - bbmin.x, 0., 0.);
		e2 = -vec3(0., 0., bbmax.z - bbmin.z);
		s = bbmax;
		{
			vec3 n = cross(normalize(e1), normalize(e2));
			if (abs(dot(n, pos - s)) < EPSILON) {
				return;
			}
		}
	}
	if (5.f <= rand3 && rand3 < 6.f) 
	{
		e1 = -vec3(bbmax.x - bbmin.x, 0., 0.);
		e2 = -vec3(0., bbmax.y - bbmin.y, 0.);
		s = bbmax;
		{
			vec3 n = cross(normalize(e1), normalize(e2));
			if (abs(dot(n, pos - s)) < EPSILON) {
				return;
			}
		}
	}

	SphQuad squad;
	SphQuadInit(s, e1, e2, pos, squad);
	SphQuadSample(pos, squad, rand1, rand2, dir, pdf);
	pdf /= float(nPlanes);

	//convert dir to world space
    dir = normalize(vec3(lightTransform * vec4(dir,0.0) ));
}

__device__
void sampleDirectLightPDF( vec3 pos,
						vec3 bbmin,
						vec3 bbmax,
                       	float &pdf ) 
{
	vec2 pmin = vec2(bbmin.x /* min x */, bbmin.y /* min y */);
	vec2 pmax = vec2(bbmax.x /* max x */, bbmax.y /* max y */);
	vec3 s = vec3(pmin, 0.0);
	vec3 ex = vec3(pmax.x - pmin.x, 0., 0.);
	vec3 ey = vec3(0., pmax.y - pmin.y, 0.);
	SphQuad squad;
	SphQuadInit(s, ex, ey, pos, squad);

	// pdf is 1 over solid angle of spherical quad
	pdf = 1.0f / squad.S;
}