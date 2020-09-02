#pragma once
#include <visii/entity.h>
#include <visii/transform.h>
#include <visii/material.h>
#include <visii/mesh.h>
#include <visii/camera.h>
#include <visii/light.h>
#include <visii/texture.h>

/**
  * Initializes various backend systems required to render scene data.
  * 
  * @param window_on_top Keeps the window opened during an interactive session on top of any other windows.
  * @param verbose If false, visii will avoid outputing any unneccessary text
*/
void initializeInteractive(bool window_on_top = false, bool verbose = true);

/**
  * Initializes various backend systems required to render scene data.
  * 
  * This call avoids using any OpenGL resources, to enable 
  * @param verbose If false, visii will avoid outputing any unneccessary text
*/
void initializeHeadless(bool verbose = true);

/**
  * Cleans up any allocated resources
*/
void clearAll();

/**
  * closes windows and shuts down any running backend systems.
*/
void deinitialize();

/** 
 * Tells the renderer which camera entity to use for rendering. The transform 
 * component of this camera entity places the camera into the world, and the
 * camera component of this camera entity describes the perspective to use, the 
 * field of view, the depth of field, and other "analog" camera properties.
 * 
 * @param camera_entity The entity containing a camera and transform component, to use for rendering. 
 */
void setCameraEntity(Entity* camera_entity);


/** 
 * Sets the intensity, or brightness, that the dome light (aka environment light) will emit it's color.
 * 
 * @param intensity How powerful the dome light is in emitting light
 */ 
void setDomeLightIntensity(float intensity);

/** 
 * Sets the color which this dome light will emit.
 * 
 * @param The RGB color emitted that this dome light should emit.
 */ 
void setDomeLightColor(glm::vec3 color);

/** 
 * Configures the procedural sky for the dome light (aka the environment).
 * @param sun_position The position of the sun relative to [0,0,0]. As the sun 
 * goes down (in Z), Rayleigh scattering will cause the sky to change colors.
 * 
 * @param sky_tint controls the relative color of the sky before Rayleigh scattering.
 * @param atmosphere_thickness effects Rayleigh scattering. Thin atmospheres look more 
 * like space, and thick atmospheres see more Rayleigh scattering.
 */ 
void setDomeLightSky(
    vec3 sun_position, 
    vec3 sky_tint = vec3(.5f, .5f, .5f), 
    float atmosphere_thickness = 1.0f);

/** 
 * Sets the texture used to color the dome light (aka the environment). 
 * Textures are sampled using a 2D to 3D latitude/longitude strategy.
 * 
 * @param texture The texture to sample for the dome light.
 */ 
void setDomeLightTexture(Texture* texture);

/** Disconnects the dome light texture, reverting back to any existing constant dome light color */
void clearDomeLightTexture();

/** 
 * Sets the rotation to apply to the dome light (aka the environment). 
 * 
 * @param rotation The rotation to apply to the dome light
 */ 
void setDomeLightRotation(glm::quat rotation);

/** 
 * Clamps the indirect light intensity during progressive image refinement. 
 * This reduces fireflies from indirect lighting, but also removes energy, and biases the resulting image.
 * 
 * @param clamp The maximum intensity that indirect lighting can contribute per frame. A value of 0 disables indirect light clamping.
 */ 
void setIndirectLightingClamp(float clamp);

/** 
 * Clamps the direct light intensity during progressive image refinement. 
 * This reduces fireflies from direct lighting, but also removes energy, and biases the resulting image.
 * 
 * @param clamp The maximum intensity that direct lighting can contribute per frame. A value of 0 disables direct light clamping.
 */ 
void setDirectLightingClamp(float clamp);

/** 
 * Sets the maximum number of times that a ray originating from the camera can bounce through the scene to accumulate light.
 * For scenes containing only rough surfaces, this max bounce depth can be set to lower values.
 * For scenes containing complex transmissive or reflective objects like glass or metals, this 
 * max bounce depth might need to be increased to accurately render these objects. Specular and diffuse
 * max bounce depth is separated to optimize these scenes.
 * 
 * @param diffuse_depth The maximum number of diffuse bounces allowed per ray.
 * @param specular_depth The maximum number of specular (reflection/refraction) bounces allowed per ray.
 */ 
void setMaxBounceDepth(uint32_t diffuse_depth, uint32_t specular_depth);

/**
 * Sets the number of light samples to take per path vertex. A higher number of samples will reduce noise per frame, but
 * also reduces frames per second.
 * 
 * @param count The number of light samples to take per path vertex. Currently constrained to a maximum of 10 samples per vertex.
 */
void setLightSampleCount(uint32_t count);

/** 
 * Sets the region of the pixel where rays should sample. By default, rays sample the entire
 * pixel area between [0,1]. Rays can instead sample a specific location of the pixel, like the pixel center,
 * by specifying a specific location within the pixel area, eg [.5, .5]. 
 * This allows for enabling or disabling antialiasing, possibly at the cost of noise in intermediate data buffers. 
 * 
 * @param x_sample_interval The interval to sample rays within along the x axis. A value of [0,1] will sample the entire pixel x axis.
 * @param y_sample_interval The interval to sample rays within along the y axis. A value of [0,1] will sample the entire pixel y axis.
 */ 
void samplePixelArea(vec2 x_sample_interval = vec2(0.f, 1.f), vec2 y_sample_interval = vec2(0.f, 1.f));

/** 
 * Sets the interval of time that rays should sample. By default, rays sample the entire
 * time interval befween the current frame and the next, [0,1]. Rays can instead sample a specific point in time,
 * like the end-of-frame time, by specifying a specific location within the time interval, eg [1.0, 1.0] or [0.0, 0.0]. 
 * This allows for enabling or disabling motion blur, while still preserving motion vectors. 
 * 
 * @param time_sample_interval The interval to sample rays within along in time. A value of [0,1] will result in motion blur across the entire frame.
 */ 
void sampleTimeInterval(vec2 time_sample_interval = vec2(0.f, 1.f));

/**
  * If using interactive mode, resizes the window to the specified dimensions.
  * 
  * @param width The width to resize the window to
  * @param height The height to resize the window to
*/
void resizeWindow(uint32_t width, uint32_t height);

/** Enables the Optix denoiser. */
void enableDenoiser();

/** Disables the Optix denoiser. */
void disableDenoiser();

/** 
 * Renders the current scene, returning the resulting framebuffer back to the user directly.
 * 
 * @param width The width of the image to render
 * @param height The height of the image to render
 * @param samples_per_pixel The number of rays to trace and accumulate per pixel.
 * @param seed A seed used to initialize the random number generator.
*/
std::vector<float> render(uint32_t width, uint32_t height, uint32_t samples_per_pixel, uint32_t seed = 0);

/** 
 * Renders the current scene, saving the resulting framebuffer to an HDR image on disk.
 * 
 * @param width The width of the image to render
 * @param height The height of the image to render
 * @param samples_per_pixel The number of rays to trace and accumulate per pixel.
 * @param image_path The path to use to save the HDR file, including the extension.
 * @param seed A seed used to initialize the random number generator.
*/
void renderToHDR(uint32_t width, uint32_t height, uint32_t samples_per_pixel, std::string image_path, uint32_t seed = 0);

/** 
 * Renders the current scene, saving the resulting framebuffer to a PNG image on disk.
 * 
 * @param width The width of the image to render
 * @param height The height of the image to render
 * @param samples_per_pixel The number of rays to trace and accumulate per pixel.
 * @param image_path The path to use to save the PNG file, including the extension.
 * @param seed A seed used to initialize the random number generator.
*/
void renderToPNG(uint32_t width, uint32_t height, uint32_t samples_per_pixel, std::string image_path, uint32_t seed = 0);

/** 
 * Renders out metadata used to render the current scene, returning the resulting framebuffer back to the user directly.
 * 
 * @param width The width of the image to render
 * @param height The height of the image to render
 * @param start_frame The start seed to feed into the random number generator
 * @param frame_count The number of frames to accumulate the resulting framebuffers by. For ID data, this should be set to 0.
 * @param bounce The number of bounces required to reach the vertex whose metadata result should come from. A value of 0
 * would save data for objects directly visible to the camera, a value of 1 would save reflections/refractions, etc.
 * @param options Indicates the data to return. Current possible values include 
 * "none" for rendering out raw path traced data, "depth" to render the distance between the previous path vertex to the current one,
 * "position" for rendering out the world space position of the path vertex, "normal" for rendering out the world space normal of the 
 * path vertex, "entity_id" for rendering out the entity ID whose surface the path vertex hit, "denoise_normal" for rendering out
 * the normal buffer supplied to the Optix denoiser, and "denoise_albedo" for rendering out the albedo supplied to the Optix denoiser.   
 * @param seed A seed used to initialize the random number generator.
*/
std::vector<float> renderData(
  uint32_t width, uint32_t height, uint32_t start_frame, uint32_t frame_count, uint32_t bounce, std::string options, uint32_t seed = 0);

/**
 * Imports an OBJ containing scene data. 
 * First, any materials described by the mtl file are used to generate Material components.
 * Next, any textures required by those materials will be loaded. 
 * After that, all shapes will be separated by material.
 * For each separated shape, an entity is created to attach a transform, mesh, and material component together.
 * These shapes are then translated so that the transform component is centered at the centroid of the shape.
 * Finally, any specified position, scale, and/or rotation are applied to the generated transforms.
 * 
 * @param name_prefix A string used to uniquely prefix any generated component names by.
 * @param filepath The path for the OBJ file to load
 * @param mtl_base_dir The path to the directory containing the corresponding .mtl file for the OBJ being loaded
 * @param position A change in position to apply to all entities generated by this function
 * @param position A change in scale to apply to all entities generated by this function
 * @param position A change in rotation to apply to all entities generated by this function
*/
std::vector<Entity*> importOBJ(std::string name_prefix, std::string file_path, std::string mtl_base_dir, 
        glm::vec3 position = glm::vec3(0.0f), 
        glm::vec3 scale = glm::vec3(1.0f),
        glm::quat rotation = glm::angleAxis(0.0f, glm::vec3(1.0f, 0.0f, 0.0f)));

// This is for internal testing purposes. Don't call this unless you know what you're doing.
void __test__(std::vector<std::string> args);
