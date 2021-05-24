/* File shared by both host and device */
#pragma once

#include <glm/glm.hpp>
#include <stdint.h>

#ifndef ENTITY_VISIBILITY_FLAGS
#define ENTITY_VISIBILITY_FLAGS
#define ENTITY_VISIBILITY_CAMERA_RAYS (1<<0) // object is visible to direct camera rays
#define ENTITY_VISIBILITY_DIFFUSE_RAYS (1<<1) // object is visible to diffuse rays
#define ENTITY_VISIBILITY_GLOSSY_RAYS (1<<2) // object is visible to glossy rays
#define ENTITY_VISIBILITY_TRANSMISSION_RAYS (1<<3) // object is visible to transmission rays
#define ENTITY_VISIBILITY_VOLUME_SCATTER_RAYS (1<<4) // object is visible to multiple-scattering volume rays
#define ENTITY_VISIBILITY_SHADOW_RAYS (1<<5) // object is visible to shadow rays (ie, casts shadows)
#endif

struct EntityStruct {
	int32_t initialized = 0;
	int32_t transform_id = -1;
	int32_t camera_id = -1;
	int32_t material_id = -1;
	int32_t light_id = -1;
	int32_t mesh_id = -1;
	int32_t volume_id = -1;
	uint32_t flags = (uint32_t)-1;
	glm::vec4 bbmin = glm::vec4(0.f);
	glm::vec4 bbmax = glm::vec4(0.f);
};