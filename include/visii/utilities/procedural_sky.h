#pragma once

#ifdef __CUDACC__
#include "cuda_utils.h"
#ifndef CUDA_DECORATOR
#define CUDA_DECORATOR __both__
#endif
#else
#ifndef CUDA_DECORATOR
#define CUDA_DECORATOR
#endif
#endif


#include <glm/glm.hpp>
using namespace glm;

inline CUDA_DECORATOR
float Scale(float inCos)
{
	float x = 1.0f - inCos;
	return 0.25f * exp(-0.00287f + x*(0.459f + x*(3.83f + x*(-6.80f + x*5.25f))));
}

inline CUDA_DECORATOR
vec3 ProceduralSkybox(
    vec3 rd, 
    vec3 sunPos, 
    vec3 sunColor = vec3(1.0f,1.0f,0.5f),
    float sunIntensity = 10.f, 
    float sunSize = .04f, 
    float sunSizeConvergence = 100.0f, 
    vec3 skyTint = vec3(.5f, .5f, .5f), 
    float atmosphereThickness = 1.0f
)
{
    #define OUTER_RADIUS 1.025f
    #define kRAYLEIGH (mix(0.0f, 0.0025f, pow(atmosphereThickness,2.5f))) 
    #define kMIE 0.0010f 
    #define kSUN_BRIGHTNESS 20.0f 
    #define kMAX_SCATTER 50.0f 
    #define MIE_G (-0.990f) 
    #define MIE_G2 0.9801f 
    const vec3 ScatteringWavelength = vec3(.65f, .57f, .475f);
    const vec3 ScatteringWavelengthRange = vec3(.15f, .15f, .15f);    
    const float kOuterRadius = OUTER_RADIUS; 
    const float kOuterRadius2 = OUTER_RADIUS*OUTER_RADIUS;
    const float kInnerRadius = 1.0f;
    const float kInnerRadius2 = 1.0f;
    const float kCameraHeight = 0.0001f;
    const float kHDSundiskIntensityFactor = 15.0f;
    const float kSunScale = 400.0f * kSUN_BRIGHTNESS;
    const float kKmESun = kMIE * kSUN_BRIGHTNESS;
    const float kKm4PI = kMIE * 4.0f * 3.14159265f;
    const float kScale = 1.0 / (OUTER_RADIUS - 1.0f);
    const float kScaleDepth = 0.25f;
    const float kScaleOverScaleDepth = (1.0f / (OUTER_RADIUS - 1.0f)) / 0.25f;
    const float kSamples = 2.0f;

    vec3 kSkyTintInGammaSpace = skyTint;
    vec3 kScatteringWavelength = mix(ScatteringWavelength-ScatteringWavelengthRange,ScatteringWavelength+ScatteringWavelengthRange,vec3(1.f,1.f,1.f) - kSkyTintInGammaSpace);
    vec3 kInvWavelength = 1.0f / (pow(kScatteringWavelength, vec3(4.0f)));
    float kKrESun = kRAYLEIGH * kSUN_BRIGHTNESS;
    float kKr4PI = kRAYLEIGH * 4.0f * 3.14159265f;
    vec3 cameraPos = vec3(0.f,kInnerRadius + kCameraHeight,0.f);
    vec3 eyeRay = rd;
    eyeRay.y = abs(eyeRay.y);
    float _far = 0.0f;
    vec3 cIn, cOut;

    _far = sqrt(kOuterRadius2 + kInnerRadius2 * eyeRay.y * eyeRay.y - kInnerRadius2) - kInnerRadius * eyeRay.y;
    vec3 pos = cameraPos + _far * eyeRay;
    float height = kInnerRadius + kCameraHeight;
    float depth = exp(kScaleOverScaleDepth * (-kCameraHeight));
    float startAngle = dot(eyeRay, cameraPos) / height;
    float startOffset = depth*Scale(startAngle);
    float sampleLength = _far / kSamples;
    float scaledLength = sampleLength * kScale;
    vec3 sampleRay = eyeRay * sampleLength;
    vec3 samplePoint = cameraPos + sampleRay * 0.5f;
    vec3 frontColor = vec3(0.0f, 0.0f, 0.0f);
    for (int i=0; i<2; i++)
    {
        float height = length(samplePoint);
        float depth = exp(kScaleOverScaleDepth * (kInnerRadius - height));
        float lightAngle = dot(normalize(sunPos), samplePoint) / height;
        float cameraAngle = dot(eyeRay, samplePoint) / height;
        float scatter = (startOffset + depth*(Scale(lightAngle) - Scale(cameraAngle)));
        vec3 attenuate = exp(-glm::clamp(scatter, 0.0f, kMAX_SCATTER) * (kInvWavelength * kKr4PI + kKm4PI));
        frontColor += attenuate * (depth * scaledLength);
        samplePoint += sampleRay;
    }
    cIn = frontColor * (kInvWavelength * kKrESun);
    cOut = frontColor * kKmESun;
    
    vec3 skyColor = (cIn * (0.75f + 0.75f * dot(normalize(sunPos), -eyeRay) * dot(normalize(sunPos), -eyeRay))); 
    float lightColorIntensity = glm::clamp(length(sunColor), 0.25f, 1.0f);
    vec3 _sunColor = kHDSundiskIntensityFactor * glm::clamp(cOut,0.0f,1.0f) * sunColor / lightColorIntensity;	    
    vec3 ray = -rd;
    vec3 color = skyColor;

    float EyeCos = pow(glm::clamp(dot(normalize(sunPos), -ray),0.0f,1.0f), sunSizeConvergence);		
	float temp = pow(1.0f + MIE_G2 - 2.0f * MIE_G * (-EyeCos), pow(sunSize,0.65f) * 10.f);
    float sunAttenuation = (1.5f * ((1.0f - MIE_G2) / (2.0f + MIE_G2)) * (1.0 + EyeCos * EyeCos) / glm::max(temp,1.0e-4f));	
    color += _sunColor * sunAttenuation * sunIntensity;
    return color;
}