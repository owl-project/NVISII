#include <visii/entity.h>
#include <visii/transform.h>
#include <visii/material.h>

// #include "RTXBigUMesh/Camera/Camera.hxx"
// #include "RTXBigUMesh/Light/Light.hxx"
// #include "RTXBigUMesh/Mesh/Mesh.hxx"
// #include "RTXBigUMesh/RigidBody/RigidBody.hxx"
// #include "RTXBigUMesh/Collider/Collider.hxx"

// #include "RTXBigUMesh/Systems/PhysicsSystem/PhysicsSystem.hxx"

Entity Entity::entities[MAX_ENTITIES];
EntityStruct Entity::entity_structs[MAX_ENTITIES];
std::map<std::string, uint32_t> Entity::lookupTable;
std::shared_ptr<std::mutex> Entity::creation_mutex;
bool Entity::Initialized = false;
bool Entity::Dirty = true;

Entity::Entity() {
	this->initialized = false;
}

Entity::Entity(std::string name, uint32_t id) {
	this->initialized = true;
	this->name = name;
	this->id = id;
	entity_structs[id].initialized = true;
	entity_structs[id].transform_id = -1;
	entity_structs[id].camera_id = -1;
	entity_structs[id].material_id = -1;
	entity_structs[id].light_id = -1;
	entity_structs[id].mesh_id = -1;
	entity_structs[id].rigid_body_id = -1;
	entity_structs[id].collider_id = -1;
}

std::string Entity::to_string()
{
	std::string output;
	output += "{\n";
	output += "\ttype: \"Entity\",\n";
	output += "\tname: \"" + name + "\",\n";
	output += "\tid: \"" + std::to_string(id) + "\",\n";
	output += "\ttransform_id: " + std::to_string(entity_structs[id].transform_id) + "\n";
	output += "\tcamera_id: " + std::to_string(entity_structs[id].camera_id) + "\n";
	output += "\tmaterial_id: " + std::to_string(entity_structs[id].material_id) + "\n";
	output += "\tlight_id: " + std::to_string(entity_structs[id].light_id) + "\n";
	output += "\tmesh_id: " + std::to_string(entity_structs[id].mesh_id) + "\n";
	output += "}";
	return output;
}


EntityStruct Entity::get_struct() {
	return entity_structs[id];
}

// void Entity::set_collider(int32_t collider_id) 
// {
// 	if (collider_id < -1) 
// 		throw std::runtime_error( std::string("Collider id must be greater than or equal to -1"));
// 	if (collider_id >= MAX_COLLIDERS)
// 		throw std::runtime_error( std::string("Collider id must be less than max colliders"));
// 	auto ps = Systems::PhysicsSystem::Get();
// 	auto edit_mutex = ps->get_edit_mutex();
//     auto edit_lock = std::lock_guard<std::mutex>(*edit_mutex.get());
// 	this->entity_struct.collider_id = collider_id;
// 	mark_dirty();
// }

// void Entity::set_collider(Collider* collider) 
// {
// 	if (!collider) 
// 		throw std::runtime_error( std::string("Invalid rigid body handle."));
// 	if (!collider->is_initialized())
// 		throw std::runtime_error("Error, collider not initialized");
// 	auto ps = Systems::PhysicsSystem::Get();
// 	auto edit_mutex = ps->get_edit_mutex();
//     auto edit_lock = std::lock_guard<std::mutex>(*edit_mutex.get());
// 	this->entity_struct.collider_id = collider->get_id();
// 	mark_dirty();
// }

// void Entity::clear_collider()
// {
// 	auto ps = Systems::PhysicsSystem::Get();
// 	auto edit_mutex = ps->get_edit_mutex();
//     auto edit_lock = std::lock_guard<std::mutex>(*edit_mutex.get());
// 	this->entity_struct.collider_id = -1;
// 	mark_dirty();
// }

// int32_t Entity::get_collider_id() 
// {
// 	return this->entity_struct.collider_id;
// }

// Collider* Entity::get_collider()
// {
// 	if ((this->entity_struct.collider_id < 0) || (this->entity_struct.collider_id >= MAX_COLLIDERS)) 
// 		return nullptr;
// 	auto collider = Collider::Get(this->entity_struct.collider_id); 
// 	if (!collider->is_initialized())
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
// 	mark_dirty();
// }

// void Entity::set_rigid_body(RigidBody* rigid_body) 
// {
// 	if (!rigid_body) 
// 		throw std::runtime_error( std::string("Invalid rigid body handle."));
// 	if (!rigid_body->is_initialized())
// 		throw std::runtime_error("Error, rigid body not initialized");
// 	this->entity_struct.rigid_body_id = rigid_body->get_id();
// 	mark_dirty();
// }

// void Entity::clear_rigid_body()
// {
// 	this->entity_struct.rigid_body_id = -1;
// 	mark_dirty();
// }

// int32_t Entity::get_rigid_body_id() 
// {
// 	return this->entity_struct.rigid_body_id;
// }

// RigidBody* Entity::get_rigid_body()
// {
// 	if ((this->entity_struct.rigid_body_id < 0) || (this->entity_struct.rigid_body_id >= MAX_RIGIDBODIES)) 
// 		return nullptr;
// 	auto rigid_body = RigidBody::Get(this->entity_struct.rigid_body_id);
// 	if (!rigid_body->is_initialized())
// 		return nullptr;
// 	return rigid_body;
// }

void Entity::set_transform(int32_t transform_id) 
{
	if (transform_id < -1) 
		throw std::runtime_error( std::string("Transform id must be greater than or equal to -1"));
	if (transform_id >= MAX_TRANSFORMS)
		throw std::runtime_error( std::string("Transform id must be less than max transforms"));
	entity_structs[id].transform_id = transform_id;
	mark_dirty();
}

void Entity::set_transform(Transform* transform) 
{
	if (!transform) 
		throw std::runtime_error( std::string("Invalid transform handle."));
	if (!transform->is_initialized())
		throw std::runtime_error("Error, transform not initialized");
	entity_structs[id].transform_id = transform->get_id();
	mark_dirty();
}

void Entity::clear_transform()
{
	entity_structs[id].transform_id = -1;
	mark_dirty();
}

int32_t Entity::get_transform_id() 
{
	return entity_structs[id].transform_id;
}

Transform* Entity::get_transform()
{
	if ((entity_structs[id].transform_id < 0) || (entity_structs[id].transform_id >= MAX_TRANSFORMS)) 
		return nullptr;
	auto transform = Transform::Get(entity_structs[id].transform_id); 
	if (!transform->is_initialized())
		return nullptr;
	return transform;
}

// void Entity::set_camera(int32_t camera_id) 
// {
// 	if (camera_id < -1) 
// 		throw std::runtime_error( std::string("Camera id must be greater than or equal to -1"));
// 	if (camera_id >= MAX_CAMERAS)
// 		throw std::runtime_error( std::string("Camera id must be less than max cameras"));
// 	this->entity_struct.camera_id = camera_id;
// 	mark_dirty();
// }

// void Entity::set_camera(Camera *camera) 
// {
// 	if (!camera)
// 		throw std::runtime_error( std::string("Invalid camera handle."));
// 	if (!camera->is_initialized())
// 		throw std::runtime_error("Error, camera not initialized");
// 	this->entity_struct.camera_id = camera->get_id();
// 	mark_dirty();
// }

// void Entity::clear_camera()
// {
// 	this->entity_struct.camera_id = -1;
// 	mark_dirty();
// }

// int32_t Entity::get_camera_id() 
// {
// 	return this->entity_struct.camera_id;
// }

// Camera* Entity::get_camera()
// {
// 	if ((this->entity_struct.camera_id < 0) || (this->entity_struct.camera_id >= MAX_CAMERAS)) 
// 		return nullptr;
// 	auto camera = Camera::Get(this->entity_struct.camera_id); 
// 	if (!camera->is_initialized())
// 		return nullptr;
// 	return camera;
// }

void Entity::set_material(int32_t material_id) 
{
	if (material_id < -1) 
		throw std::runtime_error( std::string("Material id must be greater than or equal to -1"));
	if (material_id >= MAX_MATERIALS)
		throw std::runtime_error( std::string("Material id must be less than max materials"));
	entity_structs[id].material_id = material_id;
	mark_dirty();
}

void Entity::set_material(Material *material) 
{
	if (!material)
		throw std::runtime_error( std::string("Invalid material handle."));
	if (!material->is_initialized())
		throw std::runtime_error("Error, material not initialized");
	entity_structs[id].material_id = material->get_id();
	mark_dirty();
}

void Entity::clear_material()
{
	entity_structs[id].material_id = -1;
	mark_dirty();
}

int32_t Entity::get_material_id() 
{
	return entity_structs[id].material_id;
}

Material* Entity::get_material()
{
	if ((entity_structs[id].material_id < 0) || (entity_structs[id].material_id >= MAX_MATERIALS)) 
		return nullptr;
	auto material = Material::Get(entity_structs[id].material_id);
	if (!material->is_initialized()) return nullptr;
	return material;
}

// void Entity::set_light(int32_t light_id) 
// {
// 	if (light_id < -1) 
// 		throw std::runtime_error( std::string("Light id must be greater than or equal to -1"));
// 	if (light_id >= MAX_LIGHTS)
// 		throw std::runtime_error( std::string("Light id must be less than max lights"));
// 	this->entity_struct.light_id = light_id;
// 	mark_dirty();
// }

// void Entity::set_light(Light* light) 
// {
// 	if (!light) 
// 		throw std::runtime_error( std::string("Invalid light handle."));
// 	if (!light->is_initialized())
// 		throw std::runtime_error("Error, light not initialized");
// 	this->entity_struct.light_id = light->get_id();
// 	mark_dirty();
// }

// void Entity::clear_light()
// {
// 	this->entity_struct.light_id = -1;
// 	mark_dirty();
// }

// int32_t Entity::get_light_id() 
// {
// 	return this->entity_struct.light_id;
// }

// Light* Entity::get_light()
// {
// 	if ((this->entity_struct.light_id < 0) || (this->entity_struct.light_id >= MAX_LIGHTS)) 
// 		return nullptr;
// 	auto light = Light::Get(this->entity_struct.light_id); 
// 	if (!light->is_initialized())
// 		return nullptr;
// 	return light;
// }

// void Entity::set_mesh(int32_t mesh_id) 
// {
// 	if (mesh_id < -1) 
// 		throw std::runtime_error( std::string("Mesh id must be greater than or equal to -1"));
// 	if (mesh_id >= MAX_MESHES)
// 		throw std::runtime_error( std::string("Mesh id must be less than max meshes"));
// 	this->entity_struct.mesh_id = mesh_id;

// 	auto rs = Systems::RenderSystem::Get();
// 	rs->enqueue_bvh_rebuild();
// 	mark_dirty();
// }

// void Entity::set_mesh(Mesh* mesh) 
// {
// 	if (!mesh) 
// 		throw std::runtime_error( std::string("Invalid mesh handle."));
// 	if (!mesh->is_initialized())
// 		throw std::runtime_error("Error, mesh not initialized");
// 	this->entity_struct.mesh_id = mesh->get_id();

// 	auto rs = Systems::RenderSystem::Get();
// 	rs->enqueue_bvh_rebuild();
// 	mark_dirty();
// }

// void Entity::clear_mesh()
// {
// 	this->entity_struct.mesh_id = -1;

// 	auto rs = Systems::RenderSystem::Get();
// 	rs->enqueue_bvh_rebuild();
// 	mark_dirty();
// }

// int32_t Entity::get_mesh_id() 
// {
// 	return this->entity_struct.mesh_id;
// }

// Mesh* Entity::get_mesh()
// {
// 	if ((this->entity_struct.mesh_id < 0) || (this->entity_struct.mesh_id >= MAX_MESHES)) 
// 		return nullptr;
// 	auto mesh = Mesh::Get(this->entity_struct.mesh_id);
// 	if (!mesh->is_initialized()) 
// 		return nullptr;
// 	return mesh;
// }

/* SSBO logic */
void Entity::Initialize()
{
	if (IsInitialized()) return;

	// auto vulkan = Libraries::Vulkan::Get();
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

	creation_mutex = std::make_shared<std::mutex>();

	Initialized = true;

}

bool Entity::IsInitialized()
{
	return Initialized;
}

// void Entity::UpdateComponents()
// {
// 	/* TODO: remove this for loop */
// 	for (int i = 0; i < MAX_ENTITIES; ++i) {
// 		if (!entities[i].is_initialized()) continue;
// 		/* TODO: account for parent transforms */
// 		entity_structs[i] = entities[i].entity_struct;
// 	};
// }

// void Entity::UploadSSBO(vk::CommandBuffer command_buffer)
// {
// 	if (!Dirty) return;
// 	Dirty = false;
// 	auto vulkan = Libraries::Vulkan::Get();
// 	auto device = vulkan->get_device();

// 	if (SSBOMemory == vk::DeviceMemory()) return;
// 	if (stagingSSBOMemory == vk::DeviceMemory()) return;

// 	auto bufferSize = MAX_ENTITIES * sizeof(EntityStruct);

// 	/* Pin the buffer */
// 	pinnedMemory = (EntityStruct*) device.mapMemory(stagingSSBOMemory, 0, bufferSize);

// 	if (pinnedMemory == nullptr) return;
// 	EntityStruct entity_structs[MAX_ENTITIES];
	
// 	/* TODO: remove this for loop */
// 	for (int i = 0; i < MAX_ENTITIES; ++i) {
// 		// if (!entities[i].is_initialized()) continue;
// 		/* TODO: account for parent transforms */
// 		entity_structs[i] = entities[i].entity_struct;
// 	};

// 	/* Copy to GPU mapped memory */
// 	memcpy(pinnedMemory, entity_structs, sizeof(entity_structs));

// 	device.unmapMemory(stagingSSBOMemory);

// 	vk::BufferCopy copyRegion;
// 	copyRegion.size = bufferSize;
// 	command_buffer.copyBuffer(stagingSSBO, SSBO, copyRegion);
// }

// vk::Buffer Entity::GetSSBO()
// {
// 	if ((SSBO != vk::Buffer()) && (SSBOMemory != vk::DeviceMemory()))
// 		return SSBO;
// 	else return vk::Buffer();
// }

// uint32_t Entity::GetSSBOSize()
// {
// 	return MAX_ENTITIES * sizeof(EntityStruct);
// }

void Entity::CleanUp()
{
	if (!IsInitialized()) return;

	for (auto &entity : entities) {
		if (entity.initialized) {
			Entity::Delete(entity.id);
		}
	}

	// auto vulkan = Libraries::Vulkan::Get();
	// if (!vulkan->is_initialized())
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

	Initialized = false;
}	

/* Static Factory Implementations */
Entity* Entity::Create(
	std::string name, 
	Transform* transform, 
	Material* material//, 
	// Camera* camera, 
	// Light* light, 
	// Mesh* mesh, 
	// RigidBody* rigid_body,
	// Collider* collider
    )
{
	auto entity =  StaticFactory::Create(creation_mutex, name, "Entity", lookupTable, entities, MAX_ENTITIES);
	try {
		if (transform) entity->set_transform(transform);
		if (material) entity->set_material(material);
		// if (camera) entity->set_camera(camera);
		// if (light) entity->set_light(light);
		// if (mesh) entity->set_mesh(mesh);
		// if (rigid_body) entity->set_rigid_body(rigid_body);
		// if (collider) entity->set_collider(collider);
		return entity;
	} catch (...) {
		StaticFactory::DeleteIfExists(creation_mutex, name, "Entity", lookupTable, entities, MAX_ENTITIES);
		throw;
	}
}

Entity* Entity::Get(std::string name) {
	return StaticFactory::Get(creation_mutex, name, "Entity", lookupTable, entities, MAX_ENTITIES);
}

Entity* Entity::Get(uint32_t id) {
	return StaticFactory::Get(creation_mutex, id, "Entity", lookupTable, entities, MAX_ENTITIES);
}

void Entity::Delete(std::string name) {
	StaticFactory::Delete(creation_mutex, name, "Entity", lookupTable, entities, MAX_ENTITIES);
	Dirty = true;
}

void Entity::Delete(uint32_t id) {
	StaticFactory::Delete(creation_mutex, id, "Entity", lookupTable, entities, MAX_ENTITIES);
	Dirty = true;
}

EntityStruct* Entity::GetFrontStruct() {
	return entity_structs;
}

Entity* Entity::GetFront() {
	return entities;
}

uint32_t Entity::GetCount() {
	return MAX_ENTITIES;
}
