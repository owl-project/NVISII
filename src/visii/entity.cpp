#include <visii/camera.h>
#include <visii/entity.h>
#include <visii/light.h>
#include <visii/transform.h>
#include <visii/material.h>
#include <visii/mesh.h>

Entity Entity::entities[MAX_ENTITIES];
EntityStruct Entity::entityStructs[MAX_ENTITIES];
std::map<std::string, uint32_t> Entity::lookupTable;
std::shared_ptr<std::mutex> Entity::editMutex;
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

void Entity::setTransform(Transform* transform) 
{
	if (!transform) 
		throw std::runtime_error( std::string("Invalid transform handle."));
	if (!transform->isFactoryInitialized())
		throw std::runtime_error("Error, transform not initialized");
	entityStructs[id].transform_id = transform->getId();
	transform->entities.insert(id);
	markDirty();
}

void Entity::clearTransform()
{
	auto transforms = Transform::getFront();
	if (entityStructs[id].transform_id != -1) {
		transforms[entityStructs[id].transform_id].entities.erase(id);
	}
	entityStructs[id].transform_id = -1;
	markDirty();
}

Transform* Entity::getTransform()
{
	if ((entityStructs[id].transform_id < 0) || (entityStructs[id].transform_id >= MAX_TRANSFORMS)) 
		return nullptr;
	auto transforms = Transform::getFront(); 
	if (!transforms[entityStructs[id].transform_id].isInitialized())
		return nullptr;
	return &transforms[entityStructs[id].transform_id];
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

Camera* Entity::getCamera()
{
	if ((entityStructs[id].camera_id < 0) || (entityStructs[id].camera_id >= MAX_CAMERAS)) 
		return nullptr;
	auto cameras = Camera::getFront(); 
	if (!cameras[entityStructs[id].camera_id].isInitialized())
		return nullptr;
	return &cameras[entityStructs[id].camera_id];
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

Material* Entity::getMaterial()
{
	if ((entityStructs[id].material_id < 0) || (entityStructs[id].material_id >= MAX_MATERIALS)) 
		return nullptr;
	auto &material = Material::getFront()[entityStructs[id].material_id];
	if (!material.isInitialized()) return nullptr;
	return &material;
}

void Entity::setLight(Light* light) 
{
	if (!light) 
		throw std::runtime_error( std::string("Invalid light handle."));
	if (!light->isFactoryInitialized())
		throw std::runtime_error("Error, light not initialized");
	entityStructs[id].light_id = light->getId();
	markDirty();
}

void Entity::clearLight()
{
	entityStructs[id].light_id = -1;
	markDirty();
}

Light* Entity::getLight()
{
	if ((entityStructs[id].light_id < 0) || (entityStructs[id].light_id >= MAX_LIGHTS)) 
		return nullptr;
	auto light = Light::get(entityStructs[id].light_id); 
	if (!light->isFactoryInitialized())
		return nullptr;
	return light;
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

Mesh* Entity::getMesh()
{
	if ((this->entityStructs[id].mesh_id < 0) || (this->entityStructs[id].mesh_id >= MAX_MESHES)) 
		return nullptr;
	auto mesh = Mesh::get(this->entityStructs[id].mesh_id);
	if (!mesh->isFactoryInitialized()) 
		return nullptr;
	return mesh;
}

void Entity::initializeFactory()
{
	if (isFactoryInitialized()) return;
	editMutex = std::make_shared<std::mutex>();
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

void Entity::markDirty() {
	dirty = true;
	anyDirty = true;
};

void Entity::updateComponents()
{
	if (!areAnyDirty()) return;
	
	for (uint32_t eid = 0; eid < Entity::getCount(); ++eid) {
		if (entities[eid].isDirty()) 
			entities[eid].markClean();
	}
	anyDirty = false;
}

void Entity::clearAll()
{
	if (!isFactoryInitialized()) return;
	for (auto &entity : entities) {
		if (entity.initialized) {
			Entity::remove(entity.name);
		}
	}
}

/* Static Factory Implementations */
Entity* Entity::create(
	std::string name, 
	Transform* transform, 
	Material* material, 
	Mesh* mesh, 
	Light* light, 
	Camera* camera
    )
{
	auto entity =  StaticFactory::create(editMutex, name, "Entity", lookupTable, entities, MAX_ENTITIES);
	try {
		if (transform) entity->setTransform(transform);
		if (material) entity->setMaterial(material);
		if (camera) entity->setCamera(camera);
		if (mesh) entity->setMesh(mesh);
		if (light) entity->setLight(light);
		return entity;
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Entity", lookupTable, entities, MAX_ENTITIES);
		throw;
	}
}

std::shared_ptr<std::mutex> Entity::getEditMutex()
{
	return editMutex;
}

Entity* Entity::get(std::string name) {
	return StaticFactory::get(editMutex, name, "Entity", lookupTable, entities, MAX_ENTITIES);
}

void Entity::remove(std::string name) {
	StaticFactory::remove(editMutex, name, "Entity", lookupTable, entities, MAX_ENTITIES);
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
