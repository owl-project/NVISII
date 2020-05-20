#include <visii/camera.h>
#include <visii/entity.h>
#include <visii/transform.h>
#include <visii/material.h>
#include <visii/mesh.h>

// #include "RTXBigUMesh/Camera/Camera.hxx"
// #include "RTXBigUMesh/Light/Light.hxx"
// #include "RTXBigUMesh/Mesh/Mesh.hxx"
// #include "RTXBigUMesh/RigidBody/RigidBody.hxx"
// #include "RTXBigUMesh/Collider/Collider.hxx"

// #include "RTXBigUMesh/Systems/PhysicsSystem/PhysicsSystem.hxx"

Entity Entity::entities[MAX_ENTITIES];
EntityStruct Entity::entityStructs[MAX_ENTITIES];
std::map<std::string, uint32_t> Entity::lookupTable;
std::shared_ptr<std::mutex> Entity::creationMutex;
bool Entity::factoryInitialized = false;
bool Entity::anyDirty = true;

Entity::Entity() {
	this->initialized = false;
}

Entity::Entity(std::string name, uint32_t id) {
	this->initialized = true;
	this->name = name;
	this->id = id;
	entityStructs[id].initialized = true;
	entityStructs[id].transform_id = -1;
	entityStructs[id].camera_id = -1;
	entityStructs[id].material_id = -1;
	entityStructs[id].light_id = -1;
	entityStructs[id].mesh_id = -1;
	entityStructs[id].rigid_body_id = -1;
	entityStructs[id].collider_id = -1;
}

std::string Entity::toString()
{
	std::string output;
	output += "{\n";
	output += "\ttype: \"Entity\",\n";
	output += "\tname: \"" + name + "\",\n";
	output += "\tid: \"" + std::to_string(id) + "\",\n";
	output += "\ttransform_id: " + std::to_string(entityStructs[id].transform_id) + "\n";
	output += "\tcamera_id: " + std::to_string(entityStructs[id].camera_id) + "\n";
	output += "\tmaterial_id: " + std::to_string(entityStructs[id].material_id) + "\n";
	output += "\tlight_id: " + std::to_string(entityStructs[id].light_id) + "\n";
	output += "\tmesh_id: " + std::to_string(entityStructs[id].mesh_id) + "\n";
	output += "}";
	return output;
}


EntityStruct Entity::getStruct() {
	return entityStructs[id];
}

// void Entity::set_collider(int32_t collider_id) 
// {
// 	if (collider_id < -1) 
// 		throw std::runtime_error( std::string("Collider id must be greater than or equal to -1"));
// 	if (collider_id >= MAX_COLLIDERS)
// 		throw std::runtime_error( std::string("Collider id must be less than max colliders"));
// 	auto ps = Systems::PhysicsSystem::get();
// 	auto edit_mutex = ps->get_edit_mutex();
//     auto edit_lock = std::lock_guard<std::mutex>(*edit_mutex.get());
// 	this->entity_struct.collider_id = collider_id;
// 	markDirty();
// }

// void Entity::set_collider(Collider* collider) 
// {
// 	if (!collider) 
// 		throw std::runtime_error( std::string("Invalid rigid body handle."));
// 	if (!collider->isFactoryInitialized())
// 		throw std::runtime_error("Error, collider not initialized");
// 	auto ps = Systems::PhysicsSystem::get();
// 	auto edit_mutex = ps->get_edit_mutex();
//     auto edit_lock = std::lock_guard<std::mutex>(*edit_mutex.get());
// 	this->entity_struct.collider_id = collider->getId();
// 	markDirty();
// }

// void Entity::clear_collider()
// {
// 	auto ps = Systems::PhysicsSystem::get();
// 	auto edit_mutex = ps->get_edit_mutex();
//     auto edit_lock = std::lock_guard<std::mutex>(*edit_mutex.get());
// 	this->entity_struct.collider_id = -1;
// 	markDirty();
// }

// int32_t Entity::get_collider_id() 
// {
// 	return this->entity_struct.collider_id;
// }

// Collider* Entity::get_collider()
// {
// 	if ((this->entity_struct.collider_id < 0) || (this->entity_struct.collider_id >= MAX_COLLIDERS)) 
// 		return nullptr;
// 	auto collider = Collider::get(this->entity_struct.collider_id); 
// 	if (!collider->isFactoryInitialized())
// 		return nullptr;
// 	return collider;
// }

// void Entity::set_rigid_body(int32_t rigid_body_id) 
// {
// 	if (rigid_body_id < -1) 
// 		throw std::runtime_error( std::string("RigidBody id must be greater than or equal to -1"));
// 	if (rigid_body_id >= MAX_RIGIDBODIES)
// 		throw std::runtime_error( std::string("RigidBody id must be less than max rigid bodies"));
// 	this->entity_struct.rigid_body_id = rigid_body_id;
// 	markDirty();
// }

// void Entity::set_rigid_body(RigidBody* rigid_body) 
// {
// 	if (!rigid_body) 
// 		throw std::runtime_error( std::string("Invalid rigid body handle."));
// 	if (!rigid_body->isFactoryInitialized())
// 		throw std::runtime_error("Error, rigid body not initialized");
// 	this->entity_struct.rigid_body_id = rigid_body->getId();
// 	markDirty();
// }

// void Entity::clear_rigid_body()
// {
// 	this->entity_struct.rigid_body_id = -1;
// 	markDirty();
// }

// int32_t Entity::get_rigid_body_id() 
// {
// 	return this->entity_struct.rigid_body_id;
// }

// RigidBody* Entity::get_rigid_body()
// {
// 	if ((this->entity_struct.rigid_body_id < 0) || (this->entity_struct.rigid_body_id >= MAX_RIGIDBODIES)) 
// 		return nullptr;
// 	auto rigid_body = RigidBody::get(this->entity_struct.rigid_body_id);
// 	if (!rigid_body->isFactoryInitialized())
// 		return nullptr;
// 	return rigid_body;
// }

void Entity::setTransform(int32_t transform_id) 
{
	if (transform_id < -1) 
		throw std::runtime_error( std::string("Transform id must be greater than or equal to -1"));
	if (transform_id >= MAX_TRANSFORMS)
		throw std::runtime_error( std::string("Transform id must be less than max transforms"));
	entityStructs[id].transform_id = transform_id;
	markDirty();
}

void Entity::setTransform(Transform* transform) 
{
	if (!transform) 
		throw std::runtime_error( std::string("Invalid transform handle."));
	if (!transform->isFactoryInitialized())
		throw std::runtime_error("Error, transform not initialized");
	entityStructs[id].transform_id = transform->getId();
	markDirty();
}

void Entity::clearTransform()
{
	entityStructs[id].transform_id = -1;
	markDirty();
}

int32_t Entity::getTransformId() 
{
	return entityStructs[id].transform_id;
}

Transform* Entity::getTransform()
{
	if ((entityStructs[id].transform_id < 0) || (entityStructs[id].transform_id >= MAX_TRANSFORMS)) 
		return nullptr;
	auto transform = Transform::get(entityStructs[id].transform_id); 
	if (!transform->isFactoryInitialized())
		return nullptr;
	return transform;
}

void Entity::setCamera(int32_t camera_id) 
{
	if (camera_id < -1) 
		throw std::runtime_error( std::string("Camera id must be greater than or equal to -1"));
	if (camera_id >= MAX_CAMERAS)
		throw std::runtime_error( std::string("Camera id must be less than max cameras"));
	entityStructs[id].camera_id = camera_id;
	markDirty();
}

void Entity::setCamera(Camera *camera) 
{
	if (!camera)
		throw std::runtime_error( std::string("Invalid camera handle."));
	if (!camera->isFactoryInitialized())
		throw std::runtime_error("Error, camera not initialized");
	entityStructs[id].camera_id = camera->getId();
	markDirty();
}

void Entity::clearCamera()
{
	entityStructs[id].camera_id = -1;
	markDirty();
}

int32_t Entity::getCameraId() 
{
	return entityStructs[id].camera_id;
}

Camera* Entity::getCamera()
{
	if ((entityStructs[id].camera_id < 0) || (entityStructs[id].camera_id >= MAX_CAMERAS)) 
		return nullptr;
	auto camera = Camera::get(entityStructs[id].camera_id); 
	if (!camera->isFactoryInitialized())
		return nullptr;
	return camera;
}

void Entity::setMaterial(int32_t material_id) 
{
	if (material_id < -1) 
		throw std::runtime_error( std::string("Material id must be greater than or equal to -1"));
	if (material_id >= MAX_MATERIALS)
		throw std::runtime_error( std::string("Material id must be less than max materials"));
	entityStructs[id].material_id = material_id;
	markDirty();
}

void Entity::setMaterial(Material *material) 
{
	if (!material)
		throw std::runtime_error( std::string("Invalid material handle."));
	if (!material->isFactoryInitialized())
		throw std::runtime_error("Error, material not initialized");
	entityStructs[id].material_id = material->getId();
	markDirty();
}

void Entity::clearMaterial()
{
	entityStructs[id].material_id = -1;
	markDirty();
}

int32_t Entity::getMaterialId() 
{
	return entityStructs[id].material_id;
}

Material* Entity::getMaterial()
{
	if ((entityStructs[id].material_id < 0) || (entityStructs[id].material_id >= MAX_MATERIALS)) 
		return nullptr;
	auto material = Material::get(entityStructs[id].material_id);
	if (!material->isFactoryInitialized()) return nullptr;
	return material;
}

// void Entity::set_light(int32_t light_id) 
// {
// 	if (light_id < -1) 
// 		throw std::runtime_error( std::string("Light id must be greater than or equal to -1"));
// 	if (light_id >= MAX_LIGHTS)
// 		throw std::runtime_error( std::string("Light id must be less than max lights"));
// 	this->entity_struct.light_id = light_id;
// 	markDirty();
// }

// void Entity::set_light(Light* light) 
// {
// 	if (!light) 
// 		throw std::runtime_error( std::string("Invalid light handle."));
// 	if (!light->isFactoryInitialized())
// 		throw std::runtime_error("Error, light not initialized");
// 	this->entity_struct.light_id = light->getId();
// 	markDirty();
// }

// void Entity::clear_light()
// {
// 	this->entity_struct.light_id = -1;
// 	markDirty();
// }

// int32_t Entity::get_light_id() 
// {
// 	return this->entity_struct.light_id;
// }

// Light* Entity::get_light()
// {
// 	if ((this->entity_struct.light_id < 0) || (this->entity_struct.light_id >= MAX_LIGHTS)) 
// 		return nullptr;
// 	auto light = Light::get(this->entity_struct.light_id); 
// 	if (!light->isFactoryInitialized())
// 		return nullptr;
// 	return light;
// }

void Entity::setMesh(int32_t mesh_id) 
{
	if (mesh_id < -1) 
		throw std::runtime_error( std::string("Mesh id must be greater than or equal to -1"));
	if (mesh_id >= MAX_MESHES)
		throw std::runtime_error( std::string("Mesh id must be less than max meshes"));
	this->entityStructs[id].mesh_id = mesh_id;

	// auto rs = Systems::RenderSystem::get();
	// rs->enqueue_bvh_rebuild();
	markDirty();
}

void Entity::setMesh(Mesh* mesh) 
{
	if (!mesh) 
		throw std::runtime_error( std::string("Invalid mesh handle."));
	if (!mesh->isFactoryInitialized())
		throw std::runtime_error("Error, mesh not initialized");
	this->entityStructs[id].mesh_id = mesh->getId();

	// auto rs = Systems::RenderSystem::get();
	// rs->enqueue_bvh_rebuild();
	markDirty();
}

void Entity::clearMesh()
{
	this->entityStructs[id].mesh_id = -1;

	// auto rs = Systems::RenderSystem::get();
	// rs->enqueue_bvh_rebuild();
	markDirty();
}

int32_t Entity::getMeshId() 
{
	return this->entityStructs[id].mesh_id;
}

Mesh* Entity::getMesh()
{
	if ((this->entityStructs[id].mesh_id < 0) || (this->entityStructs[id].mesh_id >= MAX_MESHES)) 
		return nullptr;
	auto mesh = Mesh::get(this->entityStructs[id].mesh_id);
	if (!mesh->isFactoryInitialized()) 
		return nullptr;
	return mesh;
}

/* SSBO logic */
void Entity::initializeFactory()
{
	if (isFactoryInitialized()) return;

	// auto vulkan = Libraries::Vulkan::get();
	// auto device = vulkan->get_device();
	// auto physical_device = vulkan->get_physical_device();

	// {
	// 	vk::BufferCreateInfo bufferInfo = {};
	// 	bufferInfo.size = MAX_ENTITIES * sizeof(EntityStruct);
	// 	bufferInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
	// 	bufferInfo.sharingMode = vk::SharingMode::eExclusive;
	// 	stagingSSBO = device.createBuffer(bufferInfo);

	// 	vk::MemoryRequirements memReqs = device.getBufferMemoryRequirements(stagingSSBO);
	// 	vk::MemoryAllocateInfo allocInfo = {};
	// 	allocInfo.allocationSize = memReqs.size;

	// 	vk::PhysicalDeviceMemoryProperties memProperties = physical_device.getMemoryProperties();
	// 	vk::MemoryPropertyFlags properties = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
	// 	allocInfo.memoryTypeIndex = vulkan->find_memory_type(memReqs.memoryTypeBits, properties);

	// 	stagingSSBOMemory = device.allocateMemory(allocInfo);
	// 	device.bindBufferMemory(stagingSSBO, stagingSSBOMemory, 0);
	// }

	// {
	// 	vk::BufferCreateInfo bufferInfo = {};
	// 	bufferInfo.size = MAX_ENTITIES * sizeof(EntityStruct);
	// 	bufferInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc;
	// 	bufferInfo.sharingMode = vk::SharingMode::eExclusive;
	// 	SSBO = device.createBuffer(bufferInfo);

	// 	vk::MemoryRequirements memReqs = device.getBufferMemoryRequirements(SSBO);
	// 	vk::MemoryAllocateInfo allocInfo = {};
	// 	allocInfo.allocationSize = memReqs.size;

	// 	vk::PhysicalDeviceMemoryProperties memProperties = physical_device.getMemoryProperties();
	// 	vk::MemoryPropertyFlags properties = vk::MemoryPropertyFlagBits::eDeviceLocal;
	// 	allocInfo.memoryTypeIndex = vulkan->find_memory_type(memReqs.memoryTypeBits, properties);

	// 	SSBOMemory = device.allocateMemory(allocInfo);
	// 	device.bindBufferMemory(SSBO, SSBOMemory, 0);
	// }

	creationMutex = std::make_shared<std::mutex>();

	factoryInitialized = true;

}

bool Entity::isFactoryInitialized()
{
	return factoryInitialized;
}

bool Entity::isInitialized()
{
	return initialized;
}

bool Entity::areAnyDirty()
{
	return anyDirty;
}

void Entity::updateComponents()
{
	if (!areAnyDirty()) return;
	
	for (uint32_t eid = 0; eid < Entity::getCount(); ++eid) {
		if (entities[eid].isDirty()) 
			entities[eid].markClean();
	}
}

// void Entity::UploadSSBO(vk::CommandBuffer command_buffer)
// {
// 	if (!Dirty) return;
// 	Dirty = false;
// 	auto vulkan = Libraries::Vulkan::get();
// 	auto device = vulkan->get_device();

// 	if (SSBOMemory == vk::DeviceMemory()) return;
// 	if (stagingSSBOMemory == vk::DeviceMemory()) return;

// 	auto bufferSize = MAX_ENTITIES * sizeof(EntityStruct);

// 	/* Pin the buffer */
// 	pinnedMemory = (EntityStruct*) device.mapMemory(stagingSSBOMemory, 0, bufferSize);

// 	if (pinnedMemory == nullptr) return;
// 	EntityStruct entityStructs[MAX_ENTITIES];
	
// 	/* TODO: remove this for loop */
// 	for (int i = 0; i < MAX_ENTITIES; ++i) {
// 		// if (!entities[i].isFactoryInitialized()) continue;
// 		/* TODO: account for parent transforms */
// 		entityStructs[i] = entities[i].entity_struct;
// 	};

// 	/* Copy to GPU mapped memory */
// 	memcpy(pinnedMemory, entityStructs, sizeof(entityStructs));

// 	device.unmapMemory(stagingSSBOMemory);

// 	vk::BufferCopy copyRegion;
// 	copyRegion.size = bufferSize;
// 	command_buffer.copyBuffer(stagingSSBO, SSBO, copyRegion);
// }

// vk::Buffer Entity::getSsbo()
// {
// 	if ((SSBO != vk::Buffer()) && (SSBOMemory != vk::DeviceMemory()))
// 		return SSBO;
// 	else return vk::Buffer();
// }

// uint32_t Entity::getSsboSize()
// {
// 	return MAX_ENTITIES * sizeof(EntityStruct);
// }

void Entity::cleanUp()
{
	if (!isFactoryInitialized()) return;

	for (auto &entity : entities) {
		if (entity.initialized) {
			Entity::remove(entity.id);
		}
	}

	// auto vulkan = Libraries::Vulkan::get();
	// if (!vulkan->isFactoryInitialized())
	// 	throw std::runtime_error( std::string("Vulkan library is not initialized"));
	// auto device = vulkan->get_device();
	// if (device == vk::Device())
	// 	throw std::runtime_error( std::string("Invalid vulkan device"));
	
	// device.destroyBuffer(SSBO);
	// device.freeMemory(SSBOMemory);

	// device.destroyBuffer(stagingSSBO);
	// device.freeMemory(stagingSSBOMemory);

	// SSBO = vk::Buffer();
	// SSBOMemory = vk::DeviceMemory();
	// stagingSSBO = vk::Buffer();
	// stagingSSBOMemory = vk::DeviceMemory();

	factoryInitialized = false;
}	

/* Static Factory Implementations */
Entity* Entity::create(
	std::string name, 
	Transform* transform, 
	Material* material, 
	Mesh* mesh, 
	Camera* camera//, 
	// Light* light, 
	// RigidBody* rigid_body,
	// Collider* collider
    )
{
	auto entity =  StaticFactory::create(creationMutex, name, "Entity", lookupTable, entities, MAX_ENTITIES);
	try {
		if (transform) entity->setTransform(transform);
		if (material) entity->setMaterial(material);
		if (camera) entity->setCamera(camera);
		if (mesh) entity->setMesh(mesh);
		// if (light) entity->setLight(light);
		// if (rigid_body) entity->setRigidBody(rigidBody);
		// if (collider) entity->setCollider(collider);
		return entity;
	} catch (...) {
		StaticFactory::removeIfExists(creationMutex, name, "Entity", lookupTable, entities, MAX_ENTITIES);
		throw;
	}
}

Entity* Entity::get(std::string name) {
	return StaticFactory::get(creationMutex, name, "Entity", lookupTable, entities, MAX_ENTITIES);
}

Entity* Entity::get(uint32_t id) {
	return StaticFactory::get(creationMutex, id, "Entity", lookupTable, entities, MAX_ENTITIES);
}

void Entity::remove(std::string name) {
	StaticFactory::remove(creationMutex, name, "Entity", lookupTable, entities, MAX_ENTITIES);
	anyDirty = true;
}

void Entity::remove(uint32_t id) {
	StaticFactory::remove(creationMutex, id, "Entity", lookupTable, entities, MAX_ENTITIES);
	anyDirty = true;
}

EntityStruct* Entity::getFrontStruct() {
	return entityStructs;
}

Entity* Entity::getFront() {
	return entities;
}

uint32_t Entity::getCount() {
	return MAX_ENTITIES;
}
