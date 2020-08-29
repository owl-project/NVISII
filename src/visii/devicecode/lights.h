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
    vec3 d = (s - o);
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
                        float rand4, 
						mat4 lightTransform,
						mat4 lightTransformInv,
						vec3 bbmin,
						vec3 bbmax,
                       	vec3 &dir,
                       	float &pdf ) 
{	
	pos = vec3( lightTransformInv * vec4(pos, 1.0) );
	normal = vec3( lightTransformInv * vec4(normal, 0.0) );
	bool minCloser = (distance(bbmin , pos) < distance(bbmax , pos));

	float nPlanes = 6;

	vec3 e1, e2;
	rand3 *= nPlanes;
	vec3 s1, s2, s;
	if (0.f <= rand3 && rand3 < 1.f) 
	{
		e1 = vec3(0., bbmax.y - bbmin.y, 0.);
		e2 = vec3(0., 0., bbmax.z - bbmin.z);
		s1 = bbmin;
		s2 = bbmin + vec3(bbmax.x - bbmin.x, 0.0, 0.0);
		s = (rand4 < .5) ? s1 : s2;
	}
	if (1.f <= rand3 && rand3 < 2.f) 
	{
		e1 = vec3(bbmax.x - bbmin.x, 0., 0.);
		e2 = vec3(0., 0., bbmax.z - bbmin.z);
		s1 = bbmin;
		s2 = bbmin + vec3(0.0, bbmax.y - bbmin.y, 0.0);
		s = (rand4 < .5) ? s1 : s2;
	}
	if (2.f <= rand3 && rand3 < 3.f) 
	{
		e1 = vec3(bbmax.x - bbmin.x, 0., 0.);
		e2 = vec3(0., bbmax.y - bbmin.y, 0.);
		s1 = bbmin;
		s2 = bbmin + vec3(0.0, 0.0, bbmax.z - bbmin.z);
		s = (rand4 < .5) ? s1 : s2;
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

//Gram-Schmidt method
__device__
vec3 orthogonalize(const vec3 &a, const vec3 &b) {
    //we assume that a is normalized
	return normalize(b - dot(a,b)*a);
}

__device__
vec3 slerp(vec3 start, vec3 end, float percent)
{
	// Dot product - the cosine of the angle between 2 vectors.
	float cosTheta = dot(start, end);
	// Clamp it to be in the range of Acos()
	// This may be unnecessary, but floating point
	// precision can be a fickle mistress.
	cosTheta = glm::clamp(cosTheta, -1.0f, 1.0f);
	// Acos(dot) returns the angle between start and end,
	// And multiplying that by percent returns the angle between
	// start and the final result.
	float theta = acos(cosTheta)*percent;
	vec3 RelativeVec = normalize(end - start*cosTheta);
     // Orthonormal basis
								 // The final result.
	return ((start*cos(theta)) + (RelativeVec*sin(theta)));
}

//Function which does triangle sampling proportional to their solid angle.
//You can find more information and pseudocode here:
// * Stratified Sampling of Spherical Triangles. J Arvo - ‎1995
// * Stratified sampling of 2d manifolds. J Arvo - ‎2001
__device__
void sampleSphericalTriangle(const vec3 &A, const vec3 &B, const vec3 &C, float rand1, float rand2, vec3 &w, float &wPdf) {
	//calculate internal angles of spherical triangle: alpha, beta and gamma
	vec3 BA = orthogonalize(A, B-A);
	vec3 CA = orthogonalize(A, C-A);
	vec3 AB = orthogonalize(B, A-B);
	vec3 CB = orthogonalize(B, C-B);
	vec3 BC = orthogonalize(C, B-C);
	vec3 AC = orthogonalize(C, A-C);
	float alpha = acos(glm::clamp(dot(BA, CA), -1.0f, 1.0f));
	float beta = acos(glm::clamp(dot(AB, CB), -1.0f, 1.0f));
	float gamma = acos(glm::clamp(dot(BC, AC), -1.0f, 1.0f));

	//calculate arc lengths for edges of spherical triangle
	float a = acos(glm::clamp(dot(B, C), -1.0f, 1.0f));
	float b = acos(glm::clamp(dot(C, A), -1.0f, 1.0f));
	float c = acos(glm::clamp(dot(A, B), -1.0f, 1.0f));

	float area = alpha + beta + gamma - M_PI;

	//Use one random variable to select the new area.
	float area_S = rand1*area;

	//Save the sine and cosine of the angle delta
	float p = sin(area_S - alpha);
	float q = cos(area_S - alpha);

	// Compute the pair(u; v) that determines sin(beta_s) and cos(beta_s)
	float u = q - cos(alpha);
	float v = p + sin(alpha)*cos(c);

	//Compute the s coordinate as normalized arc length from A to C_s.
	float s = (1.0 / b)*acos(glm::clamp(((v*q - u*p)*cos(alpha) - v) / ((v*p + u*q)*sin(alpha)), -1.0f, 1.0f));

	//Compute the third vertex of the sub - triangle.
	vec3 C_s = slerp(A, C, s);

	//Compute the t coordinate using C_s and rand2
	float t = acos(1.0 - rand2*(1.0f - dot(C_s, B))) / acos(dot(C_s, B));

	//Construct the corresponding point on the sphere.
	vec3 P = slerp(B, C_s, t);

	w = P;
	wPdf = 1.0 / area;
}

// Converting PDF between from Area to Solid angle
inline __device__
float PdfAtoW( float aPdfA, float aDist2, float aCosThere ){
    float absCosTheta = abs(aCosThere);
    if( absCosTheta < EPSILON )
        return 0.0;
    
    return aPdfA * aDist2 / absCosTheta;
}

inline __device__
vec3 uniformPointWithinTriangle( const vec3 &v1, const vec3 &v2, const vec3 &v3, float rand1, float rand2 ) {
    rand1 = sqrt(rand1);
    return (1.0f - rand1)* v1 + rand1 * (1.0f-rand2) * v2 + rand1 * rand2 * v3;
}

inline __device__
vec2 uniformUVWithinTriangle( const vec2 &uv1, const vec2 &uv2, const vec2 &uv3, float rand1, float rand2 ) {
    rand1 = sqrt(rand1);
    return (1.0f - rand1)* uv1 + rand1 * (1.0f-rand2) * uv2 + rand1 * rand2 * uv3;
}

inline __device__
void sampleTriangle(const vec3 &pos, 
					const vec3 &n1, const vec3 &n2, const vec3 &n3, 
					const vec3 &v1, const vec3 &v2, const vec3 &v3, 
					const vec2 &uv1, const vec2 &uv2, const vec2 &uv3, 
					float rand1, float rand2, vec3 &dir, float &pdf, vec2 &uv,
					bool double_sided)
{
	vec3 p = uniformPointWithinTriangle( v1, v2, v3, rand1, rand2 );
	vec3 n = uniformPointWithinTriangle( n1, n2, n3, rand1, rand2 );
	uv = uniformUVWithinTriangle( uv1, uv2, uv3, rand1, rand2 );
	float triangleArea = fabs(length(cross(v1-v2, v3-v2)) * 0.5);
	float pdfA = triangleArea;//1.0 / triangleArea;
	dir = p - pos;
	float d2 = dot(dir, dir); 
	float d = sqrt(d2); // linear
	dir /= d;
	float aCosThere = max(0.0, (double_sided) ? fabs(dot(-dir,n)) : dot(-dir,n));
	pdf = PdfAtoW( pdfA, d2 + 1.0, aCosThere ); // adding 1 to remove singularity at 0
}

__device__
const float* upper_bound (const float* first, const float* last, const float& val)
{
  const float* it;
//   iterator_traits<const float*>::difference_type count, step;
  int count, step;
//   count = std::distance(first,last);
  count = (last-first);
  while (count > 0)
  {
    it = first; 
    step=count/2; 
    // std::advance (it,step);
    it = it + step;
    if ( ! (val < *it))                 // or: if (!comp(val,*it)), for version (2)
    { 
        first=++it; 
        count-=step+1;  
    }
    else count=step;
  }
  return first;
}

__device__ float sample_cdf(const float* data, unsigned int n, float x, unsigned int *idx, float* pdf) 
{
    *idx = upper_bound(data, data + n, x) - data;
    float scaled_sample;
    if (*idx == 0) {
        *pdf = data[0];
        scaled_sample = x / data[0];
    } else {
        *pdf = data[*idx] - data[*idx - 1];
        scaled_sample = (x - data[*idx - 1]) / (data[*idx] - data[*idx - 1]);
    }
    // keep result in [0,1)
    return min(scaled_sample, 0.99999994f);
}