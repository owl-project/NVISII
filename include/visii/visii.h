#pragma once
#include <visii/entity.h>
#include <visii/transform.h>
#include <visii/material.h>
#include <visii/mesh.h>
#include <visii/camera.h>
#include <visii/light.h>
#include <visii/texture.h>

/**
   Initializes various backend systems required to render scene data.
*/
void initializeInteractive(bool window_on_top = false);

/**
   Initializes various backend systems required to render scene data.
*/
void initializeHeadless();

/**
   Cleans up any allocated resources, closes windows and shuts down any running backend systems.
*/
void cleanup();

/** Tells the renderer which camera entity to use for rendering. The transform 
 * component of this camera entity places the camera into the world, and the
 * camera component of this camera entity describes the perspective to use, the 
 * field of view, the depth of field, and other "analog" camera properties.
 * \param camera_entity The entity containing a camera and transform component, to use for rendering. */
void setCameraEntity(Entity* camera_entity);


/** Sets the intensity, or brightness, that the dome light (aka environment light) will emit it's color.
 * \param intensity How powerful the dome light is in emitting light
 */ 
void setDomeLightIntensity(float intensity);

/** Sets the texture used to color the dome light (aka the environment). 
 * Textures are sampled using a 2D to 3D latitude/longitude strategy.
 * \param texture The texture to sample for the dome light.
 */ 
void setDomeLightTexture(Texture* texture);

/**
   If using interactive mode, resizes the window to the specified dimentions.
*/
void resizeWindow(uint32_t width, uint32_t height);

std::vector<float> readFrameBuffer();

std::vector<float> render(uint32_t width, uint32_t height, uint32_t samples_per_pixel);

void enableDenoiser();
void disableDenoiser();
void renderToHDR(uint32_t width, uint32_t height, uint32_t samples_per_pixel, std::string image_path);
void renderToPNG(uint32_t width, uint32_t height, uint32_t samples_per_pixel, std::string image_path);

std::vector<Entity*> importOBJ(std::string name_prefix, std::string filepath, std::string mtl_base_dir, 
        glm::vec3 position = glm::vec3(0.0f), 
        glm::vec3 scale = glm::vec3(1.0f),
        glm::quat rotation = glm::angleAxis(0.0f, glm::vec3(1.0f, 0.0f, 0.0f)));