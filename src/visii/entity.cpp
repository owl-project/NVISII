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
	auto &entity = getStruct();
	entity.initialized = true;
	entity.transform_id = -1;
	entity.camera_id = -1;
	entity.material_id = -1;
	entity.light_id = -1;
	entity.mesh_id = -1;
}

std::string Entity::toString()
{
	std::string output;
	output += "{\n";
	output += "\ttype: \"Entity\",\n";
	output += "\tname: \"" + name + "\",\n";
	output += "\tid: \"" + std::to_string(id) + "\",\n";
	output += "\ttransform_id: " + std::to_string(getStruct().transform_id) + "\n";
	output += "\tcamera_id: " + std::to_string(getStruct().camera_id) + "\n";
	output += "\tmaterial_id: " + std::to_string(getStruct().material_id) + "\n";
	output += "\tlight_id: " + std::to_string(getStruct().light_id) + "\n";
	output += "\tmesh_id: " + std::to_string(getStruct().mesh_id) + "\n";
	output += "}";
	return output;
}

EntityStruct &Entity::getStruct() {
	if (!isInitialized()) throw std::runtime_error("Error: entity is uninitialized.");
	return entityStructs[id];
}

void Entity::setTransform(Transform* transform) 
{
	auto &entity = getStruct();
	if (!transform) throw std::runtime_error( std::string("Invalid transform handle."));
	if (!transform->isFactoryInitialized()) throw std::runtime_error("Error, transform not initialized");
	entity.transform_id = transform->getId();
	transform->entities.insert(id);
	markDirty();
}

void Entity::clearTransform()
{
	auto &entity = getStruct();
	auto transforms = Transform::getFront();
	if (entity.transform_id != -1) transforms[entity.transform_id].entities.erase(id);
	entity.transform_id = -1;
	markDirty();
}

Transform* Entity::getTransform()
{
	auto &entity = getStruct();
	if ((entity.transform_id < 0) || (entity.transform_id >= MAX_TRANSFORMS)) return nullptr;
	auto transforms = Transform::getFront(); 
	if (!transforms[entity.transform_id].isInitialized()) return nullptr;
	return &transforms[entity.transform_id];
}

void Entity::setCamera(Camera *camera) 
{
	auto &entity = getStruct();
	if (!camera) throw std::runtime_error( std::string("Invalid camera handle."));
	if (!camera->isFactoryInitialized()) throw std::runtime_error("Error, camera not initialized");
	entity.camera_id = camera->getId();
	camera->entities.insert(id);
	markDirty();
}

void Entity::clearCamera()
{
	auto &entity = getStruct();
	auto cameras = Camera::getFront();
	if (entity.camera_id != -1) cameras[entity.camera_id].entities.erase(id);
	entity.camera_id = -1;
	markDirty();
}

Camera* Entity::getCamera()
{
	auto &entity = getStruct();
	if ((entity.camera_id < 0) || (entity.camera_id >= MAX_CAMERAS)) return nullptr;
	auto cameras = Camera::getFront(); 
	if (!cameras[entity.camera_id].isInitialized()) return nullptr;
	return &cameras[entity.camera_id];
}

void Entity::setMaterial(Material *material) 
{
	auto &entity = getStruct();
	if (!material) throw std::runtime_error( std::string("Invalid material handle."));
	if (!material->isFactoryInitialized()) throw std::runtime_error("Error, material not initialized");
	entity.material_id = material->getId();
	material->entities.insert(id);
	markDirty();
}

void Entity::clearMaterial()
{
	auto &entity = getStruct();
	auto materials = Material::getFront();
	if (entity.material_id != -1) materials[entity.material_id].entities.erase(id);
	entity.material_id = -1;
	markDirty();
}

Material* Entity::getMaterial()
{
	auto &entity = getStruct();
	if ((entity.material_id < 0) || (entity.material_id >= MAX_MATERIALS)) return nullptr;
	auto &material = Material::getFront()[entity.material_id];
	if (!material.isInitialized()) return nullptr;
	return &material;
}

void Entity::setLight(Light* light) 
{
	auto &entity = getStruct();
	if (!light) throw std::runtime_error( std::string("Invalid light handle."));
	if (!light->isFactoryInitialized()) throw std::runtime_error("Error, light not initialized");
	entity.light_id = light->getId();
	light->entities.insert(id);
	markDirty();
}

void Entity::clearLight()
{
	auto &entity = getStruct();
	auto lights = Light::getFront();
	if (entity.light_id != -1) lights[entity.light_id].entities.erase(id);
	entity.light_id = -1;
	markDirty();
}

Light* Entity::getLight()
{
	auto &entity = getStruct();
	if ((entity.light_id < 0) || (entity.light_id >= MAX_LIGHTS)) return nullptr;
	auto &light = Light::getFront()[entity.light_id];
	if (!light.isInitialized()) return nullptr;
	return &light;
}

void Entity::setMesh(Mesh* mesh) 
{
	auto &entity = getStruct();
	if (!mesh) throw std::runtime_error( std::string("Invalid mesh handle."));
	if (!mesh->isFactoryInitialized()) throw std::runtime_error("Error, mesh not initialized");
	entity.mesh_id = mesh->getId();
	mesh->entities.insert(id);
	markDirty();
}

void Entity::clearMesh()
{
	auto &entity = getStruct();
	auto meshes = Mesh::getFront();
	if (entity.mesh_id != -1) meshes[entity.mesh_id].entities.erase(id);
	entity.mesh_id = -1;
	markDirty();
}

Mesh* Entity::getMesh()
{
	auto &entity = getStruct();
	if ((entity.mesh_id < 0) || (entity.mesh_id >= MAX_MESHES))  return nullptr;
	auto mesh = Mesh::get(entity.mesh_id);
	if (!mesh->isFactoryInitialized()) return nullptr;
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
	auto entity = Entity::get(name);
	entity->clearCamera();
	entity->clearLight();
	entity->clearMaterial();
	entity->clearMesh();
	entity->clearTransform();
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
