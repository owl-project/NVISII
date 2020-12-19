/* File shared by both host and device */
#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#define GLM_FORCE_RIGHT_HANDED
#include <stdint.h>
#include <glm/glm.hpp>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/SampleFromVoxels.h>

struct VolumeStruct
{
    float gradient_factor = 0.5f;
    float scale = 1.f;
    float absorption = 0.5f;
    float scattering = 0.5f;
};
