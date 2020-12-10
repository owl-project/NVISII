/* File shared by both host and device */
#pragma once

#include <glm/glm.hpp>
#include <stdint.h>

#ifndef ENTITY_VISIBILITY_FLAGS
#define ENTITY_VISIBILITY_FLAGS
#define ENTITY_VISIBILITY_CAMERA_RAYS (1<<0)
#endif

struct EntityStruct {
	int32_t initialized = 0;
	int32_t transform_id = -1;
	int32_t camera_id = -1;
	int32_t material_id = -1;
	int32_t light_id = -1;
	int32_t mesh_id = -1;
	int32_t volume_id = -1;
	int32_t flags = 1;
	glm::vec4 bbmin = glm::vec4(0.f);
	glm::vec4 bbmax = glm::vec4(0.f);
};