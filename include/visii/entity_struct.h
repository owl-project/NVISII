/* File shared by both host and device */
#pragma once

#include <stdint.h>
#define MAX_ENTITIES 1024

struct EntityStruct {
	int32_t initialized = 0;
	int32_t transform_id = -1;
	int32_t camera_id = -1;
	int32_t material_id = -1;
	int32_t light_id = -1;
	int32_t mesh_id = -1;
	int32_t rigid_body_id = -1;
	int32_t collider_id = -1;
};