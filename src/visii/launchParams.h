#pragma once

#include <owl/owl.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_access.hpp>

#include <visii/entity_struct.h>

struct LaunchParams {
    glm::ivec2 frameSize;
    glm::vec4 *fbPtr;
    uint32_t *accumPtr;
    OptixTraversableHandle world;

    EntityStruct cameraEntity;
};