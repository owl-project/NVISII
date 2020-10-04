#pragma once

#include "cuda_utils.h"

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
					bool double_sided, bool use_surface_area)
{
	vec3 p = uniformPointWithinTriangle( v1, v2, v3, rand1, rand2 );
	vec3 n = uniformPointWithinTriangle( n1, n2, n3, rand1, rand2 );
	uv = uniformUVWithinTriangle( uv1, uv2, uv3, rand1, rand2 );
	float pdfA;
	if (use_surface_area) {
		float triangleArea = fabs(length(cross(v1-v2, v3-v2)) * 0.5);
		float pdfA = 1.0f / triangleArea;
	} else{
		pdfA = 1.0f;
	}
	dir = p - pos;
	float d2 = dot(dir, dir); 
	float d = sqrt(d2); // linear
	dir /= d;
	float aCosThere = max(0.0, (double_sided) ? fabs(dot(-dir,n)) : dot(-dir,n));
	pdf = PdfAtoW( pdfA, d2, aCosThere );
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