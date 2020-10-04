#pragma once

#include <math_constants.h>
#include <optix.h>
#include "float3.h"
#include "types.h"

// Tone Mapping
// From http://filmicgames.com/archives/75
__device__ float3 uncharted_2_tonemap(float3 x)
{
	if (x.x < 0) x.x = 0;
	if (x.y < 0) x.y = 0;
	if (x.z < 0) x.z = 0;
	float A = 0.15f;
	float B = 0.50f;
	float C = 0.10f;
	float D = 0.20f;
	float E_ = 0.02f;
	float F = 0.30f;
	float3 result = ((x*(A*x+C*B)+D*E_)/(x*(A*x+B)+D*F)) -E_/F;
	if (result.x < 0) result.x = 0;
	if (result.y < 0) result.y = 0;
	if (result.z < 0) result.z = 0;
	return result;
}

__device__ float linear_to_srgb(float x) {
	if (x <= 0.0031308f) {
		return 12.92f * x;
	}
	return 1.055f * pow(x, 1.f/2.4f) - 0.055f;
}

__device__ float luminance(const float3 &c) {
	return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ float pow2(float x) {
	return x * x;
}

// code from [Frisvad2012]
__device__ void ortho_basis(float3 &b1, float3 &b2, float3 n)
{
    if (n.z < -0.9999999f)
    {
        b1 = make_float3( 0.0f, -1.0f, 0.0f);
        b2 = make_float3(-1.0f,  0.0f, 0.0f);
        return;
    }
    float a = 1.0f / (1.0f + n.z);
    float b = -n.x*n.y*a;
    b1 = make_float3(1.0 - n.x*n.x*a, b, -n.x);
    b2 = make_float3(b, 1.0 - n.y*n.y*a, -n.y);
}

template<typename T>
__device__ T clamp(const T &x, const T &lo, const T &hi) {
	if (x < lo) {
		return lo;
	}
	if (x > hi) {
		return hi;
	}
	return x;
}

__device__ float lerp(float x, float y, float s) {
	return x * (1.f - s) + y * s;
}

__device__ float3 lerp(float3 x, float3 y, float s) {
	return x * (1.f - s) + y * s;
}

__device__ float3 reflect(const float3 &i, const float3 &n) {
	return i - 2.f * n * dot(i, n);
}

__device__ float3 refract( float3 i, float3 n, float eta )
{
  if (eta == 1.f) return i;
  if (eta <= 0.f) return make_float3(0.f);
  if (isnan(eta)) return make_float3(0.f);
  if (isinf(eta)) return make_float3(0.f);
  float cosi = dot(-i, n);
  float cost2 = 1.0f - eta * eta * (1.0f - cosi*cosi);
  float3 t = eta*i + ((eta*cosi - sqrt(abs(cost2))) * n);
  return t * ((cost2 > 0.f) ? make_float3(1.f) : make_float3(0.f));
}

__device__ float3 refract_ray(const float3 &i, const float3 &n, float eta) {
	float n_dot_i = dot(n, i);
	float k = 1.f - eta * eta * (1.f - n_dot_i * n_dot_i);
	if (k < 0.f) {
		return make_float3(0.f);
	}
	return eta * i - (eta * n_dot_i + sqrt(k)) * n;
}

__device__ float component(const float4 &v, const uint32_t i) {
    switch (i) {
    case 0: return v.x;
    case 1: return v.y;
    case 2: return v.z;
    case 3: return v.w;
    default: return CUDART_NAN_F;
    }
}

__device__ void* unpack_ptr(uint32_t hi, uint32_t lo) {
	const uint64_t val = static_cast<uint64_t>(hi) << 32 | lo;
	return reinterpret_cast<void*>(val);
}

__device__ void pack_ptr(void *ptr, uint32_t &hi, uint32_t &lo) {
	const uint64_t val = reinterpret_cast<uint64_t>(ptr);
	hi = val >> 32;
	lo = val & 0x00000000ffffffff;
}

template<typename T>
__device__ T& get_payload() {
	return *reinterpret_cast<T*>(unpack_ptr(optixGetPayload_0(), optixGetPayload_1()));
}

template<typename T>
__device__ const T& get_shader_params() {
	return *reinterpret_cast<const T*>(optixGetSbtDataPointer());
}
