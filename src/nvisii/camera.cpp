#include <nvisii/camera.h>
#include <glm/gtx/matrix_transform_2d.hpp>

namespace nvisii {

std::vector<Camera> Camera::cameras;
std::vector<CameraStruct> Camera::cameraStructs;
std::map<std::string, uint32_t> Camera::lookupTable;
std::shared_ptr<std::recursive_mutex> Camera::editMutex;
bool Camera::factoryInitialized = false;
bool Camera::anyDirty = true;
// int32_t Camera::minRenderOrder = 0;
// int32_t Camera::maxRenderOrder = 0;
// #include <algorithm>
// #include <limits>

// #define GLM_FORCE_DEPTH_ZERO_TO_ONE
// #define GLM_DEPTH_ZERO_TO_ONE
// using namespace Libraries;

Camera::Camera() {
	initialized = false;
}

Camera::Camera(std::string name, uint32_t id) {
	initialized = true;
	this->name = name;
	this->id = id;

    cameraStructs[id].apertureDiameter = 0.0;
    cameraStructs[id].focalDistance = 1.0;
	// this->render_complete_mutex = std::make_shared<std::recursive_mutex>();
	// this->cv = std::make_shared<std::condition_variable>();
}

std::string Camera::toString()
{
	std::string output;
	output += "{\n";
	output += "\ttype: \"Camera\",\n";
	output += "\tname: \"" + name + "\",\n";
	output += "}";
	return output;
}

CameraStruct Camera::getStruct() {
	return cameraStructs[id];
}

void Camera::initializeFactory(uint32_t max_components)
{
	if (isFactoryInitialized()) return;
    cameras.resize(max_components);
    cameraStructs.resize(max_components);
	editMutex = std::make_shared<std::recursive_mutex>();
	factoryInitialized = true;
}

bool Camera::isFactoryInitialized()
{
    return factoryInitialized;
}

bool Camera::isInitialized()
{
	return initialized;
}

bool Camera::areAnyDirty() {
    return anyDirty;
}

void Camera::markDirty() {
    dirty = true;
    anyDirty = true;
};

void Camera::updateComponents()
{
    for (int i = 0; i < cameras.size(); ++i) {
		if (cameras[i].isDirty()) {
            cameras[i].markClean();
        }
	};
	anyDirty = false;


	// if (!anyDirty) return;
	// anyDirty = false;
    // 	auto vulkan = Libraries::Vulkan::Get();
    //     auto device = vulkan->get_device();

    //     if (SSBOMemory == vk::DeviceMemory()) return;
    //     if (stagingSSBOMemory == vk::DeviceMemory()) return;
        
    //     auto bufferSize = cameras.size() * sizeof(CameraStruct);

    // 	/* Pin the buffer */
    // 	pinnedMemory = (CameraStruct*) device.mapMemory(stagingSSBOMemory, 0, bufferSize);
    // 	if (pinnedMemory == nullptr) return;
        
    // 	/* TODO: remove this for loop */
    // 	for (uint32_t i = 0; i < cameras.size(); ++i) {
    // 		if (!cameras[i].is_initialized()) continue;
    // 		pinnedMemory[i] = cameras[i].camera_struct;

    // 		for (uint32_t j = 0; j < cameras[i].maxViews; ++j) {
    // 			pinnedMemory[i].multiviews[j].viewinv = glm::inverse(pinnedMemory[i].multiviews[j].view);
    // 			pinnedMemory[i].multiviews[j].projinv = glm::inverse(pinnedMemory[i].multiviews[j].proj);
    // 			pinnedMemory[i].multiviews[j].viewproj = pinnedMemory[i].multiviews[j].proj * pinnedMemory[i].multiviews[j].view;
    // 		}
    // 	};

    // 	device.unmapMemory(stagingSSBOMemory);

    // 	vk::BufferCopy copyRegion;
    // 	copyRegion.size = bufferSize;
    //     command_buffer.copyBuffer(stagingSSBO, SSBO, copyRegion);
    // }

    // void Camera::setup(uint32_t tex_width, uint32_t tex_height, uint32_t msaa_samples, uint32_t max_views, bool use_depth_prepass, bool use_multiview)
    // {
    // 	auto rs = Systems::RenderSystem::Get();
    // 	if ((tex_width == 0) || (tex_height == 0)) {
    // 		throw std::runtime_error("Error: width and height of camera " + name + " texture must be greater than zero!");
    // 	}

    // 	#if DISABLE_MULTIVIEW
    // 	this->use_multiview = false;
    // 	#else
    // 	this->use_multiview = use_multiview;
    // 	#endif

    // 	this->use_depth_prepass = use_depth_prepass;
    // 	if (max_views > MAX_MULTIVIEW) {
    // 		throw std::runtime_error( std::string("Error: Camera component cannot render to more than " + std::to_string(MAX_MULTIVIEW) + " layers simultaneously."));
    //     }

    // 	this->maxViews = max_views;
    // 	set_view(glm::mat4(1.0), 0);
    // 	// set_orthographic_projection(-1, 1, -1, 1, -1, 1);
    // 	this->msaa_samples = msaa_samples;
        
    // 	/* If the number of views is 6, assume the user might want to use the camera as a cubemap. */
    // 	if (max_views == 6) {
    // 		renderTexture = Texture::CreateCubemapGBuffers(name, tex_width, tex_height, msaa_samples);
    // 		if (msaa_samples != 1)
    // 			resolveTexture = Texture::CreateCubemapGBuffers(name, tex_width, tex_height, 1);
    // 	}
    // 	// Otherwise, just create G Buffer Resources
    // 	else {
    // 		renderTexture = Texture::Create2DGBuffers(name, tex_width, tex_height, msaa_samples, max_views);
    // 		if (msaa_samples != 1)
    // 			resolveTexture = Texture::Create2DGBuffers(name + "_resolve", tex_width, tex_height, 1, max_views);
    // 	}

    // 	// Is there a better way to do this? Seems like tex_id is the same for each view...
    // 	for (uint32_t i = 0; i < max_views; ++i) {
    // 		camera_struct.multiviews[i].tex_id = (msaa_samples != 1) ? resolveTexture->get_id() : renderTexture->get_id();
    // 	}

    // 	rs->create_camera_resources(get_id());

    // 	this->max_visibility_distance = std::numeric_limits<float>::max();
    // 	mark_dirty();
}

void Camera::clearAll()
{
	if (!isFactoryInitialized()) return;

	for (auto &camera : cameras) {
		if (camera.initialized) {
			Camera::remove(camera.name);
		}
	}
}

/* Static Factory Implementations */

Camera* Camera::create(std::string name, float fieldOfView, float aspect)
{
	auto camera = StaticFactory::create(editMutex, name, "Camera", lookupTable, cameras.data(), cameras.size());
	try {
        camera->setFOV(fieldOfView, aspect);
        return camera;
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Camera", lookupTable, cameras.data(), cameras.size());
		throw;
	}
}

Camera* Camera::createPerspectiveFromFOV(std::string name, float fieldOfView, float aspect)
{
    static bool createPerspectiveFromFOVDeprecatedShown = false;
    if (createPerspectiveFromFOVDeprecatedShown == false) {
        std::cout<<"Warning, create_perspective_from_fov is deprecated and will be removed in a subsequent release. Please switch to create_from_fov." << std::endl;
        createPerspectiveFromFOVDeprecatedShown = true;
    }
    return createFromFOV(name, fieldOfView, aspect);
}

Camera* Camera::createFromFOV(std::string name, float fieldOfView, float aspect)
{
	auto camera = StaticFactory::create(editMutex, name, "Camera", lookupTable, cameras.data(), cameras.size());
	try {
        camera->setFOV(fieldOfView, aspect);
        return camera;
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Camera", lookupTable, cameras.data(), cameras.size());
		throw;
	}
}

Camera* Camera::createPerspectiveFromFocalLength(std::string name, float focalLength, float sensorWidth, float sensorHeight)
{
    static bool createPerspectiveFromFocalLength = false;
    if (createPerspectiveFromFocalLength == false) {
        std::cout<<"Warning, create_perspective_from_focal_length is deprecated and will be removed in a subsequent release. Please switch to create_from_focal_length." << std::endl;
        createPerspectiveFromFocalLength = true;
    }
    return createFromFocalLength(name, focalLength, sensorWidth, sensorHeight);
}

Camera* Camera::createFromFocalLength(std::string name, float focalLength, float sensorWidth, float sensorHeight)
{
	auto camera = StaticFactory::create(editMutex, name, "Camera", lookupTable, cameras.data(), cameras.size());
	try {
        camera->setFocalLength(focalLength, sensorWidth, sensorHeight);
        return camera;
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Camera", lookupTable, cameras.data(), cameras.size());
		throw;
	}
}

Camera* Camera::createFromIntrinsics(std::string name, float fx, float fy, float cx, float cy, float width, float height, float near, float far)
{
	auto camera = StaticFactory::create(editMutex, name, "Camera", lookupTable, cameras.data(), cameras.size());
	try {
        camera->setIntrinsics(fx, fy, cx, cy, width, height, near, far);
        return camera;
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Camera", lookupTable, cameras.data(), cameras.size());
		throw;
	}
}

std::shared_ptr<std::recursive_mutex> Camera::getEditMutex()
{
	return editMutex;
}

Camera* Camera::get(std::string name) {
	return StaticFactory::get(editMutex, name, "Camera", lookupTable, cameras.data(), cameras.size());
}

void Camera::remove(std::string name) {
	StaticFactory::remove(editMutex, name, "Camera", lookupTable, cameras.data(), cameras.size());
	anyDirty = true;
}

CameraStruct* Camera::getFrontStruct() {
	return cameraStructs.data();
}

Camera* Camera::getFront() {
	return cameras.data();
}

uint32_t Camera::getCount() {
	return cameras.size();
}

std::string Camera::getName()
{
    return name;
}

std::map<std::string, uint32_t> Camera::getNameToIdMap()
{
	return lookupTable;
}



// void Camera::update_used_views(uint32_t multiview) {
// 	usedViews = (usedViews >= multiview) ? usedViews : multiview;
// }

// void Camera::check_multiview_index(uint32_t multiview) {
// 	if (multiview >= maxViews)
//     {
// 		throw std::runtime_error( std::string("Error: multiview index is larger than " + std::to_string(maxViews) ));
//     }
// }

// // glm::mat4 MakeInfReversedZOrthoRH(float left, float right, float bottom, float top, float zNear)
// // {
// // 	return glm::mat4 (
// // 		2.0f / (right - left), 0.0f,  0.0f,  0.0f,
// // 		0.0f,    2.0f / (top - bottom),  0.0f,  0.0f,
// // 		0.0f, 0.0f,  0.0, -1.0f,
// // 		0.0, 0.0, zNear,  0.0f
// // 	);
// // }

// // bool Camera::set_orthographic_projection(float left, float right, float bottom, float top, float near_pos, uint32_t multiview)
// // {
// // 	check_multiview_index(multiview);
// // 	camera_struct.multiviews[multiview].near_pos = near_pos;
// // 	camera_struct.multiviews[multiview].proj = MakeInfReversedZOrthoRH(left, right, bottom, top, near_pos);
// // 	camera_struct.multiviews[multiview].projinv = glm::inverse(camera_struct.multiviews[multiview].proj);
// // 	return true;
// // };

glm::mat4 makeInfReversedZProjRH(float fovY_radians, float aspectWbyH, float zNear)
{
    float f = 1.0f / tan(fovY_radians / 2.0f);
    return glm::mat4(
        f / aspectWbyH, 0.0f,  0.0f,  0.0f,
                  0.0f,    f,  0.0f,  0.0f,
                  0.0f, 0.0f,  0.0f, -1.0f,
                  0.0f, 0.0f, zNear,  0.0f);
}

glm::mat4 makeProjRH(float fovY_radians, float aspectWbyH, float zNear)
{
	auto proj = glm::perspectiveFov(fovY_radians, aspectWbyH, 1.0f, zNear, 1000.0f);
	return proj;
}

// void Camera::set_perspective_projection(float fov_in_radians, float width, float height, float near_pos, uint32_t multiview)
// {
// 	check_multiview_index(multiview);
// 	camera_struct.multiviews[multiview].near_pos = near_pos;
// 	#ifndef DISABLE_REVERSE_Z
// 	camera_struct.multiviews[multiview].proj = MakeInfReversedZProjRH(fov_in_radians, width / height, near_pos);
// 	set_clear_depth(0.0);
// 	#else
// 	camera_struct.multiviews[multiview].proj = MakeProjRH(fov_in_radians, width / height, near_pos);
// 	set_clear_depth(1.0);
// 	#endif
// 	camera_struct.multiviews[multiview].projinv = glm::inverse(camera_struct.multiviews[multiview].proj);
// 	mark_dirty();
// };

void Camera::setFOV(float fieldOfView, float aspect)
{
    cameraStructs[id].proj = glm::perspective(fieldOfView, aspect, 1.f, 1000.f);
    cameraStructs[id].projinv = glm::inverse(cameraStructs[id].proj);
    markDirty();
}

void Camera::setFocalLength(float focalLength, float sensorWidth, float sensorHeight)
{
    float aspect = sensorWidth / sensorHeight;
    float fovy = 2.f*atan(0.5f*sensorHeight / focalLength);
    cameraStructs[id].proj = glm::perspective(fovy, aspect, 1.f, 1000.f);
    cameraStructs[id].projinv = glm::inverse(cameraStructs[id].proj);
    markDirty();
}

// void Camera::set_custom_projection(glm::mat4 custom_projection, float near_pos, uint32_t multiview)
// {
// 	check_multiview_index(multiview);
// 	camera_struct.multiviews[multiview].near_pos = near_pos;
// 	camera_struct.multiviews[multiview].proj = custom_projection;
// 	camera_struct.multiviews[multiview].projinv = glm::inverse(custom_projection);
// 	mark_dirty();
// }

// float Camera::get_near_pos(uint32_t multiview) { 
// 	check_multiview_index(multiview);
// 	return camera_struct.multiviews[multiview].near_pos; 
// }

void Camera::setFocalDistance(float distance)
{
    cameraStructs[id].focalDistance = distance;
    markDirty();
}

void Camera::setApertureDiameter(float diameter)
{
    cameraStructs[id].apertureDiameter = diameter;
    markDirty();
}

// void Camera::set_render_order(int32_t order) {
// 	renderOrder = order;
	
// 	/* Update min/max render orders */
// 	for (auto &camera : cameras) {
// 		if (!camera.is_initialized()) continue;
// 		minRenderOrder = std::min(minRenderOrder, camera.get_render_order());
// 		maxRenderOrder = std::max(maxRenderOrder, camera.get_render_order());
// 	}
// }

// void Camera::set_max_visible_distance(float max_distance)
// {
// 	this->max_visibility_distance = std::max(max_distance, 0.0f);
// }

// float Camera::get_max_visible_distance()
// {
// 	return this->max_visibility_distance;
// }

// int32_t Camera::get_render_order()
// {
// 	return renderOrder;
// }

// int32_t Camera::GetMinRenderOrder() {
// 	return minRenderOrder;
// }

// int32_t Camera::GetMaxRenderOrder() {
// 	return maxRenderOrder;
// }

glm::mat3 Camera::getIntrinsicMatrix(float width, float height) { 
    glm::mat3 intrinsics;
    intrinsics = glm::column(intrinsics, 0, glm::vec3(glm::column(cameraStructs[id].proj, 0)));
    intrinsics = glm::column(intrinsics, 1, glm::vec3(glm::column(cameraStructs[id].proj, 1)));
    intrinsics = glm::column(intrinsics, 2, glm::vec3(0.f, 0.f, 1.f));

    // perspective is from -1 to 1. Make it go from 0 to 2.
    glm::mat3 translation = glm::translate(glm::mat3(1.f), glm::vec2(.5f, .5f));
    // from 0 to 2 to now 0 to 1
    glm::mat3 scale1 = glm::scale(glm::mat3(1.f), glm::vec2(.5f, .5f));
    // from 0 to 1 to now 0 to width/height
    glm::mat3 scale2 = glm::scale(glm::mat3(1.f), glm::vec2(width, height));
    
	intrinsics = scale2 * scale1 * translation * intrinsics; 
    
    // finding we need these multiplies to match opencv...
    intrinsics[2][0] *= 2;
    intrinsics[2][1] *= 2;
    return intrinsics;
};

glm::mat4 Camera::getProjection() {
	return cameraStructs[id].proj; 
};


void Camera::setProjection(glm::mat4 projection)
{
    cameraStructs[id].proj = projection;
    markDirty();
}
		
void Camera::setIntrinsics(float fx, float fy, float cx, float cy, float width, float height, float near, float far)
{
    auto P = glm::mat4();
    P[0][0] = 2.0 * fx / width;
    P[1][1] = 2.0 * fy / height;
    P[2][0] = 1.0 - 2.0 * cx / (width - 1.0);
    P[2][1] = 2.0 * cy / (height - 1.0) - 1.0;
    P[2][3] = -1.0;

    float n = near;
    float f = far;
    P[2][2] = (f + n) / (n - f);
    P[3][2] = (2.f * f * n) / (n - f);
    cameraStructs[id].proj = P;
    markDirty();
}

// Texture* Camera::get_texture()
// {
// 	if (msaa_samples == 1) return renderTexture;
// 	else return resolveTexture;
// }

// /* TODO: Explain this */
// uint32_t Camera::get_max_views()
// {
// 	return maxViews;
// }

// void Camera::pause_visibility_testing()
// {
// 	visibilityTestingPaused = true;
// }

// void Camera::resume_visibility_testing()
// {
// 	visibilityTestingPaused = false;
// }

// // vk::Fence Camera::get_fence(uint32_t frame_idx) {
// // 	return fences[frame_idx];
// // }

// void Camera::set_clear_color(float r, float g, float b, float a) {
// 	clearColor = glm::vec4(r, g, b, a);
// }

// void Camera::set_clear_stencil(uint32_t stencil) {
// 	clearStencil = stencil;
// }

// void Camera::set_clear_depth(float depth) {
// 	clearDepth = depth;
// }



// vk::Buffer Camera::GetSSBO()
// {
// 	return SSBO;
// }

// uint32_t Camera::GetSSBOSize()
// {
// 	return cameras.size() * sizeof(CameraStruct);
// }

// std::vector<Camera *> Camera::GetCamerasByOrder(uint32_t order)
// {
// 	/* Todo: improve the performance of this. */
// 	std::vector<Camera *> selected_cameras;
// 	for (uint32_t i = 0; i < cameras.size(); ++i) {
// 		if (!cameras[i].is_initialized()) continue;

// 		if (cameras[i].renderOrder == order) {
// 			selected_cameras.push_back(&cameras[i]);
// 		}
// 	}
// 	return selected_cameras;
// }

// void Camera::cleanup()
// {
// 	if (!initialized) return;

// 	auto vulkan = Vulkan::Get();
// 	auto device = vulkan->get_device();
	
// 	uint32_t max_frames_in_flight = 2;
// }

// void Camera::force_render_mode(RenderMode rendermode)
// {
// 	renderModeOverride = rendermode;
// }

// RenderMode Camera::get_rendermode_override() {
// 	return renderModeOverride;
// }

// bool Camera::should_record_depth_prepass()
// {
// 	return use_depth_prepass;
// }

// bool Camera::should_use_multiview()
// {
// 	return use_multiview;
// }

// void Camera::mark_render_as_complete()
// {
// 	{
// 		auto m = render_complete_mutex.get();
// 		std::lock_guard<std::recursive_mutex> lk(*m);
// 		render_ready = true;
// 	}
// 	cv->notify_one();
// }

// void Camera::wait_for_render_complete()
// {
// 	// Wait until main() sends data
// 	auto m = render_complete_mutex.get();
//     std::unique_lock<std::recursive_mutex> lk(*m);
// 	render_ready = false;
//     cv->wait(lk, [this]{return render_ready;});
// 	render_ready = false;
// 	lk.unlock();
// }

};
