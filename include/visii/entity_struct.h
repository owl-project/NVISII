/* File shared by both host and device */
#pragma once

#include <stdint.h>
#define MAX_ENTITIES 1024

struct EntityStruct {
	int32_t initialized;
	int32_t transform_id;
	int32_t camera_id;
	int32_t material_id;
	int32_t light_id;
	int32_t mesh_id;
	int32_t rigid_body_id;
	int32_t collider_id;
};