#pragma once
#include <visii/entity.h>
#include <visii/transform.h>
#include <visii/material.h>
#include <visii/mesh.h>
#include <visii/camera.h>
#include <visii/light.h>
#include <visii/texture.h>
#include <visii/volume.h>

/**
  * Deprecated. Please use initialize() instead.
*/
void initializeInteractive(
  bool window_on_top = false, 
  bool verbose = false,
  uint32_t max_entities = 10000,
  uint32_t max_cameras = 10,
  uint32_t max_transforms = 10000,
  uint32_t max_meshes = 10000,
  uint32_t max_materials = 10000,
  uint32_t max_lights = 100,
  uint32_t max_textures = 1000,
  uint32_t max_volumes = 1000);

/**
  * Deprecated. Please use initialize(headless = True) instead.
*/
void initializeHeadless(
  bool verbose = false,
  uint32_t max_entities = 10000,
  uint32_t max_cameras = 10,
  uint32_t max_transforms = 10000,
  uint32_t max_meshes = 10000,
  uint32_t max_materials = 10000,
  uint32_t max_lights = 100,
  uint32_t max_textures = 1000,
  uint32_t max_volumes = 1000);

/**
  * Initializes various backend systems required to render scene data.
  * 
  * @param headless If true, avoids using any OpenGL resources, to enable use on systems without displays.
  * @param window_on_top Keeps the window opened during an interactive session on top of any other windows. (assuming headless is False)
  * @param lazy_updates If True, visii will only upload components to the GPU on call to 
  * render/render_to_png/render_data for better scene editing performance. (assuming headless is False. Always on when headless is True)
  * @param verbose If false, visii will avoid outputing any unneccessary text
  * @param max_entities The max number of creatable Entity components.
  * @param max_cameras The max number of creatable Camera components.
  * @param max_transforms The max number of creatable Transform components.
  * @param max_meshes The max number of creatable Mesh components.
  * @param max_materials The max number of creatable Material components.
  * @param max_lights The max number of creatable Light components.
  * @param max_textures The max number of creatable Texture components.
*/
void initialize(
  bool headless = false, 
  bool window_on_top = false, 
  bool lazy_updates = false, 
  bool verbose = false,
  uint32_t max_entities = 10000,
  uint32_t max_cameras = 10,
  uint32_t max_transforms = 10000,
  uint32_t max_meshes = 10000,
  uint32_t max_materials = 10000,
  uint32_t max_lights = 100,
  uint32_t max_textures = 1000,
  uint32_t max_volumes = 1000);

/**
  * Removes any allocated components but keeps visii initialized.
  * Call this if you would like to clear the current scene.
*/
void clearAll();

/**
  * Closes the interactive window, and shuts down any running backend systems.
  * Call this function at the end of your script.
*/
void deinitialize();

/* Deprecated. Please use the "register_callback" function instead. */
void registerPreRenderCallback(std::function<void()> callback);

/**
 * Registers a callback which is called on the render thread before each rendered 
 * frame. This mechanism is useful for implementing camera controls and other 
 * routines dependent on cursor and button clicks. To disable the callback, pass 
 * nullptr/None here.
 */
void registerCallback(std::function<void()> callback);

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
 * Modifies the intensity, or brightness, that the dome light (aka environment light) will emit it's color.
 * Increasing the exposure by 1 will double the energy emitted by the light. 
 * An exposure of 0 produces an unmodified intensity.
 * An exposure of -1 cuts the intensity of the light in half.
 * light_intensity = intensity * pow(2, exposureExposure)
 * 
 * @param exposure How powerful the light source is in emitting light.
 */ 
void setDomeLightExposure(float exposure);

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
 * @param saturation causes the sky to appear more or less "vibrant"
 */ 
void setDomeLightSky(
    glm::vec3 sun_position, 
    glm::vec3 sky_tint = vec3(.5f, .5f, .5f), 
    float atmosphere_thickness = 1.0f,
    float saturation = 1.0f);

/** 
 * Sets the texture used to color the dome light (aka the environment). 
 * Textures are sampled using a 2D to 3D latitude/longitude strategy.
 * 
 * @param texture The texture to sample for the dome light.
 * @param enable_cdf If True, reduces noise of sampling a dome light texture, 
 * but at the expense of frame rate. Useful for dome lights with bright lights 
 * that should cast shadows.
 */ 
void setDomeLightTexture(Texture* texture, bool enable_cdf = false);

/** Disconnects the dome light texture, reverting back to any existing constant dome light color */
void clearDomeLightTexture();

/** 
 * Sets the rotation to apply to the dome light (aka the environment). 
 * 
 * @param rotation The rotation to apply to the dome light
 */ 
void setDomeLightRotation(glm::quat rotation);

/** If enabled, objects will be lit by the dome light. */
void enableDomeLightSampling();

/** If disabled, objects will not be lit by the dome light. 
 * Instead, the dome light will only effect the background color. */
void disableDomeLightSampling();

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
 * @param volume_depth The maximum number of volume bounces allowed per ray.
 */ 
void setMaxBounceDepth(uint32_t diffuse_depth = 2, uint32_t specular_depth = 8, uint32_t volume_depth = 2);

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
void samplePixelArea(glm::vec2 x_sample_interval = glm::vec2(0.f, 1.f), glm::vec2 y_sample_interval = glm::vec2(0.f, 1.f));

/** 
 * Sets the interval of time that rays should sample. By default, rays sample the entire
 * time interval befween the current frame and the next, [0,1]. Rays can instead sample a specific point in time,
 * like the end-of-frame time, by specifying a specific location within the time interval, eg [1.0, 1.0] or [0.0, 0.0]. 
 * This allows for enabling or disabling motion blur, while still preserving motion vectors. 
 * 
 * @param time_sample_interval The interval to sample rays within along in time. A value of [0,1] will result in motion blur across the entire frame.
 */ 
void sampleTimeInterval(glm::vec2 time_sample_interval = glm::vec2(0.f, 1.f));

/** Enables the Optix denoiser. */
void enableDenoiser();

/** Disables the Optix denoiser. */
void disableDenoiser();

/**
 * Controls what guides and modes are used to denoise the image.
 * @param use_albedo_guide If True, uses albedo to guide the denoiser. Useful for scenes with 
 * textures or large uniformly colored sections. Can cause issues when denoising motion blur.
 * @param use_normal_guide If True, uses surface normals to guide the denoiser. Useful for 
 * scenes where geometrically complex objects do not have distinct albedo (eg walls, uniformly colored objects, etc)
 * @param use_kernel_prediction If True, uses the OptiX kernel prediction model for denoising, which avoids intensity 
 * shifts and false color prediction by instead predicting a normalized kernel. 
*/
void configureDenoiser(bool use_albedo_guide = true, bool use_normal_guide = true, bool use_kernel_prediction = true);

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
 * Deprecated. Please use renderToFile. 
*/
void renderToHDR(uint32_t width, uint32_t height, uint32_t samples_per_pixel, std::string image_path, uint32_t seed = 0);

/** 
 * Deprecated. Please use renderToFile. 
*/
void renderToPNG(uint32_t width, uint32_t height, uint32_t samples_per_pixel, std::string image_path, uint32_t seed = 0);

/** 
 * Renders the current scene, saving the resulting framebuffer to an image on disk.
 * 
 * @param width The width of the image to render
 * @param height The height of the image to render
 * @param samples_per_pixel The number of rays to trace and accumulate per pixel.
 * @param file_path The path to use to save the file, including the extension. Supported extensions include EXR, HDR, and PNG 
 * @param seed A seed used to initialize the random number generator.
*/
void renderToFile(uint32_t width, uint32_t height, uint32_t samples_per_pixel, std::string file_path, uint32_t seed = 0);

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
 * @param file_path The path to use to save the file, including the extension. Supported extensions are EXR, HDR, and PNG
 * @param seed A seed used to initialize the random number generator.
*/
void renderDataToFile(uint32_t width, uint32_t height, uint32_t start_frame, uint32_t frame_count, uint32_t bounce, std::string options, std::string file_path, uint32_t seed = 0);

/**
 * An object containing a list of components that together represent a scene
*/
struct Scene {
  std::vector<Entity*> entities;
  std::vector<Transform*> transforms;
  std::vector<Texture*> textures;
  std::vector<Material*> materials;
  std::vector<Mesh*> meshes;
  std::vector<Light*> lights;
  std::vector<Camera*> cameras;
};

/**
 * Imports a file containing scene data. 
 * 
 * Supported file formats include: AMF 3DS AC ASE ASSBIN B3D BVH COLLADA DXF 
 * CSM HMP IRRMESH IRR LWO LWS M3D MD2 MD3 MD5 MDC MDL NFF NDO OFF OBJ OGRE 
 * OPENGEX PLY MS3D COB BLEND IFC XGL FBX Q3D Q3BSP RAW SIB SMD STL 
 * TERRAGEN 3D X X3D GLTF 3MF MMD
 * 
 * First, any materials described by the file are used to generate Material components.
 * Next, any textures required by those materials will be loaded. 
 * After that, all shapes will be separated by material.
 * For each separated shape, an entity is created to attach a transform, mesh, and material component together.
 * These shapes are then translated so that the transform component is centered at the centroid of the shape.
 * Finally, any specified position, scale, and/or rotation are applied to the generated transforms.
 * 
 * @param filepath The path for the file to load
 * @param position A change in position to apply to all entities generated by this function
 * @param position A change in scale to apply to all entities generated by this function
 * @param position A change in rotation to apply to all entities generated by this function
 * @param args A list of optional arguments that can effect the importer. 
 * Possible options include: 
 * "verbose" - print out information related to loading the scene.
*/
Scene importScene(
        std::string file_path,
        glm::vec3 position = glm::vec3(0.0f), 
        glm::vec3 scale = glm::vec3(1.0f),
        glm::quat rotation = glm::angleAxis(0.0f, glm::vec3(1.0f, 0.0f, 0.0f)),
        std::vector<std::string> args = std::vector<std::string>());

/** @returns the minimum axis aligned bounding box position for the axis aligned bounding box containing all scene geometry*/
glm::vec3 getSceneMinAabbCorner();

/** @returns the maximum axis aligned bounding box position for the axis aligned bounding box containing all scene geometry*/
glm::vec3 getSceneMaxAabbCorner();

/** @returns the center of the aligned bounding box for the axis aligned bounding box containing all scene geometry*/
glm::vec3 getSceneAabbCenter();

// This is for internal purposes. Forces the scene bounds to update.
void updateSceneAabb(Entity* entity);

/** 
 * If enabled, the interactive window image will change asynchronously as scene components are altered.
 * However, bulk component edits will slow down, as each component edit will individually cause the renderer to 
 * temporarily lock components while being uploaded to the GPU.
 */
void enableUpdates();

/** 
 * If disabled, the interactive window image will only show scene changes on call to render, render_to_png, and render_data.
 * Bulk component edits will be much faster when disabled, as all component edits can be done without the renderer 
 * locking them for upload to the GPU. 
 */
void disableUpdates();

/*** If in interactive mode, returns true if updates are enabled, and false otherwise */
bool areUpdatesEnabled();

/**
  * If using interactive mode, resizes the window to the specified dimensions.
  * 
  * @param width The width to resize the window to
  * @param height The height to resize the window to
*/
void resizeWindow(uint32_t width, uint32_t height);

/** 
 * If in interactive mode, returns True if the specified button is pressed but not held.
 * @param The button to check. Not case sensitive. Possible options include:
 * SPACE, APOSTROPHE, COMMA, MINUS, PERIOD, SLASH, SEMICOLON, EQUAL, UP, DOWN, LEFT, RIGHT
 * 0-9, A->Z, [, ], \\, `, ESCAPE, ENTER, TAB, BACKSPACE, INSERT, DELETE, PAGE_UP, PAGE_DOWN, HOME, 
 * CAPS_LOCK, SCROLL_LOCK, NUM_LOCK, PRINT_SCREEN, PAUSE, F1 -> F25, KP_0 -> KP_9,
 * KP_DECIMAL, KP_DIVIDE, KP_MULTIPLY, KP_SUBTRACT, KP_ADD, KP_ENTER, KP_EQUAL, 
 * LEFT_SHIFT, LEFT_CONTROL, LEFT_ALT, LEFT_SUPER, RIGHT_SHIFT, RIGHT_CONTROL, RIGHT_ALT, RIGHT_SUPER,
 * MOUSE_LEFT, MOUSE_MIDDLE, MOUSE_RIGHT
*/
bool isButtonPressed(std::string button);

/** 
 * If in interactive mode, returns True if the specified button is held down.
 * @param The button to check. Not case sensitive. Possible options include:
 * SPACE, APOSTROPHE, COMMA, MINUS, PERIOD, SLASH, SEMICOLON, EQUAL, UP, DOWN, LEFT, RIGHT
 * 0-9, A->Z, [, ], \\, `, ESCAPE, ENTER, TAB, BACKSPACE, INSERT, DELETE, PAGE_UP, PAGE_DOWN, HOME, 
 * CAPS_LOCK, SCROLL_LOCK, NUM_LOCK, PRINT_SCREEN, PAUSE, F1 -> F25, KP_0 -> KP_9,
 * KP_DECIMAL, KP_DIVIDE, KP_MULTIPLY, KP_SUBTRACT, KP_ADD, KP_ENTER, KP_EQUAL, 
 * LEFT_SHIFT, LEFT_CONTROL, LEFT_ALT, LEFT_SUPER, RIGHT_SHIFT, RIGHT_CONTROL, RIGHT_ALT, RIGHT_SUPER,
 * MOUSE_LEFT, MOUSE_MIDDLE, MOUSE_RIGHT
*/
bool isButtonHeld(std::string button);

/** If in interactive mode, returns the position of the cursor relative to the window. */
glm::vec2 getCursorPos();

/** 
 * If in interactive mode, sets the mode of the cursor.
 * @param mode Can be one of the following:
 * NORMAL - makes the cursor visible and beaving normally
 * HIDDEN makes the cursor invisible when it is over the content area of the window, 
 * but does not restrict the cursor from leaving.
 * DISABLED - hides and grabs the cursor, providing virtual and unlimited cursor movement. 
 * This is useful for implementing for example 3D camera controls.
 */
void setCursorMode(std::string mode);

/** If in interactive mode, returns size of the window */
glm::ivec2 getWindowSize();

/** If in interactive mode, returns true if the close button on the window was clicked. */
bool shouldWindowClose();

// This is for internal testing purposes. Don't call this unless you know what you're doing.
void __test__(std::vector<std::string> args);
