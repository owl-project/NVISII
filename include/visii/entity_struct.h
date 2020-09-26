/* File shared by both host and device */
#pragma once

#include <glm/glm.hpp>
#include <stdint.h>
#define MAX_ENTITIES 1000000

#ifndef ENTITY_VISIBILITY_FLAGS
#define ENTITY_VISIBILITY_FLAGS
#define ENTITY_VISIBILITY_CAMERA_RAYS (1<<0)
// #define LIGHT_FLAGS_SHOW_END_CAPS (1<<1)
// #define LIGHT_FLAGS_CAST_SHADOWS (1<<2)
// #define LIGHT_FLAGS_USE_VSM (1<<3)
// #define LIGHT_FLAGS_DISABLED (1<<4)
// #define LIGHT_FLAGS_POINT (1<<5)
// #define LIGHT_FLAGS_SPHERE (1<<6)
// #define LIGHT_FLAGS_DISK (1<<7)
// #define LIGHT_FLAGS_ROD (1<<8)
// #define LIGHT_FLAGS_PLANE (1<<9)
#endif

struct EntityStruct {
	int32_t initialized = 0;
	int32_t transform_id = -1;
	int32_t camera_id = -1;
	int32_t material_id = -1;
	int32_t light_id = -1;
	int32_t mesh_id = -1;
	int32_t visibilityFlags = 1;
	glm::vec4 bbmin = glm::vec4(0.f);
	glm::vec4 bbmax = glm::vec4(0.f);
};