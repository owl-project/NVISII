#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include "types.h"

__device__ float4 make_float4(float c) {
	return make_float4(c, c, c, c);
}

__device__ float4 make_float4(float3 v, float c) {
	return make_float4(v.x, v.y, v.z, c);
}

__device__ float4 make_float4(glm::vec3 v, float c) {
	return make_float4(v.x, v.y, v.z, c);
}

__device__ float4 make_float4(glm::vec4 v) {
	return make_float4(v.x, v.y, v.z, v.w);
}

__device__ float3 make_float3(float c) {
	return make_float3(c, c, c);
}

__device__ float3 make_float3(float4 v) {
	return make_float3(v.x, v.y, v.z);
}

__device__ float3 make_float3(glm::vec4 v) {
	return make_float3(v.x, v.y, v.z);
}

__device__ float3 make_float3(glm::vec3 v) {
	return make_float3(v.x, v.y, v.z);
}

__device__ float2 make_float2(float c) {
	return make_float2(c, c);
}

__device__ float2 make_float2(uint2 v) {
	return make_float2(v.x, v.y);
}

__device__ float2 make_float2(glm::vec2 v) {
	return make_float2(v.x, v.y);
}

__device__ glm::vec4 make_vec4(float4 v) {
	return glm::vec4(v.x, v.y, v.z, v.w);
}

__device__ glm::vec4 make_vec4(float3 v, float c) {
	return glm::vec4(v.x, v.y, v.z, c);
}

__device__ glm::vec3 make_vec3(float4 v) {
	return glm::vec3(v.x, v.y, v.z);
}

__device__ glm::vec3 make_vec3(float3 v) {
	return glm::vec3(v.x, v.y, v.z);
}

__device__ glm::vec2 make_vec2(float2 v) {
	return glm::vec2(v.x, v.y);
}

__device__ glm::mat4 to_mat4(float xfm_[12])
{
    glm::mat4 xfm;
    xfm = glm::column(xfm, 0, vec4(xfm_[0], xfm_[4],  xfm_[8], 0.0f));
    xfm = glm::column(xfm, 1, vec4(xfm_[1], xfm_[5],  xfm_[9], 0.0f));
    xfm = glm::column(xfm, 2, vec4(xfm_[2], xfm_[6],  xfm_[10], 0.0f));
    xfm = glm::column(xfm, 3, vec4(xfm_[3], xfm_[7],  xfm_[11], 1.0f));
	return xfm;
}

__device__ float length(const float3 &v) {
	// return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	return __fsqrt_rn(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ float3 normalize(const float3 &v) {
	// float l = length(v);
	// if (l < 0.f) {
	// 	l = 0.0001f;
	// }
	const float c = __frsqrt_rn(v.x * v.x + v.y * v.y + v.z * v.z);  //1.f / length(v);
	return make_float3(v.x * c, v.y * c, v.z * c);
}

__device__ float3 cross(const float3 &a, const float3 &b) {
	float3 c;
	c.x = a.y * b.z - a.z * b.y;
	c.y = a.z * b.x - a.x * b.z;
	c.z = a.x * b.y - a.y * b.x;
	return c;
}

__device__ float3 neg(const float3 &a) {
	return make_float3(-a.x, -a.y, -a.z);
}

__device__ bool all_zero(const float3 &v) {
	return v.x == 0.f && v.y == 0.f && v.z == 0.f;
}

__device__ float dot(const float3 a, const float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float4 operator*(const uint32_t s, const float4 &v) {
	return make_float4(s * v.x, s * v.y, s * v.z, s * v.w);
}

__device__ float4 operator*(const float4 &v, const uint32_t s) {
	return s * v;
}

__device__ float4 operator+(const float4 &a, const float4 &b) {
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__device__ float4 operator/(const float4 &a, const uint32_t s) {
	const float x = 1.f / s;
	return x * a;
}

__device__ float3 operator-(const float3 &a, const float3 &b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator-(const float3 &a, const float s) {
	return make_float3(a.x - s, a.y - s, a.z - s);
}

__device__ float3 operator-(const float s, const float3 &a) {
	return make_float3(s - a.x, s - a.y, s - a.z);
}

__device__ float3 operator-(const float3 &a) {
	return make_float3(-a.x, -a.y, -a.z);
}

__device__ float3 operator+(const float3 &a, const float3 &b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator+(const float3 &a, const float s) {
	return make_float3(a.x + s, a.y + s, a.z + s);
}

__device__ float3 operator+(const float s, const float3 &a) {
	return a + s;
}

__device__ float3 operator*(const float3 &a, const float s) {
	return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ float3 operator*(const float s, const float3 &a) {
	return a * s;
}

__device__ float3 operator*(const float3 &a, const float3 &b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ float3 operator/(const float3 &a, const float s) {
	return make_float3(a.x / s, a.y / s, a.z / s);
}

__device__ float3 operator/(const float s, const float3 &a) {
	return make_float3(a.x / s, a.y / s, a.z / s);
}

__device__ float3 operator/(const float3 &a, const float3 &b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ float2 operator-(const float2 &a, const float2 &b) {
	return make_float2(a.x - b.x, a.y - b.y);
}

__device__ float2 operator-(const float2 &a, const float s) {
	return make_float2(a.x - s, a.y - s);
}

__device__ float2 operator-(const float s, const float2 &a) {
	return make_float2(s - a.x, s - a.y);
}

__device__ float2 operator-(const float2 &a) {
	return make_float2(-a.x, -a.y);
}

__device__ float2 operator+(const float2 &a, const float2 &b) {
	return make_float2(a.x + b.x, a.y + b.y);
}

__device__ float2 operator+(const float2 &a, const float s) {
	return make_float2(a.x + s, a.y + s);
}

__device__ float2 operator+(const float s, const float2 &a) {
	return a + s;
}

__device__ float2 operator*(const float2 &a, const float s) {
	return make_float2(a.x * s, a.y * s);
}

__device__ float2 operator*(const float s, const float2 &a) {
	return a * s;
}

__device__ float2 operator/(const float2 &a, const float2 &b) {
	return make_float2(a.x / b.x, a.y / b.y);
}

__device__
float approx_acosf(float x) {
    return (-0.69813170079773212f * x * x - 0.87266462599716477f) * x + 1.5707963267948966f;
}

// Polynomial approximating arctangenet on the range -1,1.
// Max error < 0.005 (or 0.29 degrees)
__device__
float approx_atanf(float z)
{
    const float n1 = 0.97239411f;
    const float n2 = -0.19194795f;
    return (n1 + n2 * z * z) * z;
}

__device__
float approx_atan2f(float y, float x)
{
    if (x != 0.0f)
    {
        if (fabsf(x) > fabsf(y))
        {
            const float z = y / x;
            if (x > 0.0)
            {
                // atan2(y,x) = atan(y/x) if x > 0
                return approx_atanf(z);
            }
            else if (y >= 0.0)
            {
                // atan2(y,x) = atan(y/x) + PI if x < 0, y >= 0
                return approx_atanf(z) + M_PI;
            }
            else
            {
                // atan2(y,x) = atan(y/x) - PI if x < 0, y < 0
                return approx_atanf(z) - M_PI;
            }
        }
        else // Use property atan(y/x) = PI/2 - atan(x/y) if |y/x| > 1.
        {
            const float z = x / y;
            if (y > 0.0)
            {
                // atan2(y,x) = PI/2 - atan(x/y) if |y/x| > 1, y > 0
                return -approx_atanf(z) + M_PI_2;
            }
            else
            {
                // atan2(y,x) = -PI/2 - atan(x/y) if |y/x| > 1, y < 0
                return -approx_atanf(z) - M_PI_2;
            }
        }
    }
    else
    {
        if (y > 0.0f) // x = 0, y > 0
        {
            return M_PI_2;
        }
        else if (y < 0.0f) // x = 0, y < 0
        {
            return -M_PI_2;
        }
    }
    return 0.0f; // x,y = 0. Could return NaN instead.
}
