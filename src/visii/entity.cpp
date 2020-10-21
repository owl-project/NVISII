#include <visii/camera.h>
#include <visii/entity.h>
#include <visii/light.h>
#include <visii/transform.h>
#include <visii/material.h>
#include <visii/mesh.h>
#include <visii/visii.h>

std::vector<Entity> Entity::entities;
std::vector<EntityStruct> Entity::entityStructs;
std::map<std::string, uint32_t> Entity::lookupTable;
std::shared_ptr<std::recursive_mutex> Entity::editMutex;
bool Entity::factoryInitialized = false;
std::set<Entity*> Entity::dirtyEntities;
std::set<Entity*> Entity::renderableEntities;

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
	std::lock_guard<std::recursive_mutex> lock(*Entity::getEditMutex().get());

	auto &entity = getStruct();
	if (!transform) throw std::runtime_error( std::string("Invalid transform handle."));
	if (!transform->isFactoryInitialized()) throw std::runtime_error("Error, transform not initialized");
	entity.transform_id = transform->getId();
	transform->entities.insert(id);
	markDirty();
}

void Entity::clearTransform()
{
	std::lock_guard<std::recursive_mutex> lock(*Entity::getEditMutex().get());
	
	auto &entity = getStruct();
	auto transforms = Transform::getFront();
	if (entity.transform_id != -1) transforms[entity.transform_id].entities.erase(id);
	entity.transform_id = -1;
	markDirty();
}

Transform* Entity::getTransform()
{
	auto &entity = getStruct();
	if ((entity.transform_id < 0) || (entity.transform_id >= Transform::getCount())) return nullptr;
	auto transforms = Transform::getFront(); 
	if (!transforms[entity.transform_id].isInitialized()) return nullptr;
	return &transforms[entity.transform_id];
}

void Entity::setCamera(Camera *camera) 
{
	std::lock_guard<std::recursive_mutex> lock(*Entity::getEditMutex().get());

	auto &entity = getStruct();
	if (!camera) throw std::runtime_error( std::string("Invalid camera handle."));
	if (!camera->isFactoryInitialized()) throw std::runtime_error("Error, camera not initialized");
	entity.camera_id = camera->getId();
	camera->entities.insert(id);
	markDirty();
}

void Entity::clearCamera()
{
	std::lock_guard<std::recursive_mutex> lock(*Entity::getEditMutex().get());

	auto &entity = getStruct();
	auto cameras = Camera::getFront();
	if (entity.camera_id != -1) cameras[entity.camera_id].entities.erase(id);
	entity.camera_id = -1;
	markDirty();
}

Camera* Entity::getCamera()
{
	auto &entity = getStruct();
	if ((entity.camera_id < 0) || (entity.camera_id >= Camera::getCount())) return nullptr;
	auto cameras = Camera::getFront(); 
	if (!cameras[entity.camera_id].isInitialized()) return nullptr;
	return &cameras[entity.camera_id];
}

void Entity::setMaterial(Material *material) 
{
	std::lock_guard<std::recursive_mutex> lock(*Entity::getEditMutex().get());

	auto &entity = getStruct();
	if (!material) throw std::runtime_error( std::string("Invalid material handle."));
	if (!material->isFactoryInitialized()) throw std::runtime_error("Error, material not initialized");
	entity.material_id = material->getId();
	material->entities.insert(id);
	markDirty();
}

void Entity::clearMaterial()
{
	std::lock_guard<std::recursive_mutex> lock(*Entity::getEditMutex().get());

	auto &entity = getStruct();
	auto materials = Material::getFront();
	if (entity.material_id != -1) materials[entity.material_id].entities.erase(id);
	entity.material_id = -1;
	markDirty();
}

Material* Entity::getMaterial()
{
	auto &entity = getStruct();
	if ((entity.material_id < 0) || (entity.material_id >= Material::getCount())) return nullptr;
	auto &material = Material::getFront()[entity.material_id];
	if (!material.isInitialized()) return nullptr;
	return &material;
}

void Entity::setLight(Light* light) 
{
	std::lock_guard<std::recursive_mutex> lock(*Entity::getEditMutex().get());

	auto &entity = getStruct();
	if (!light) throw std::runtime_error( std::string("Invalid light handle."));
	if (!light->isFactoryInitialized()) throw std::runtime_error("Error, light not initialized");
	entity.light_id = light->getId();
	light->entities.insert(id);
	markDirty();
}

void Entity::clearLight()
{
	std::lock_guard<std::recursive_mutex> lock(*Entity::getEditMutex().get());

	auto &entity = getStruct();
	auto lights = Light::getFront();
	if (entity.light_id != -1) lights[entity.light_id].entities.erase(id);
	entity.light_id = -1;
	markDirty();
}

Light* Entity::getLight()
{
	auto &entity = getStruct();
	if ((entity.light_id < 0) || (entity.light_id >= Light::getCount())) return nullptr;
	auto &light = Light::getFront()[entity.light_id];
	if (!light.isInitialized()) return nullptr;
	return &light;
}

void Entity::setMesh(Mesh* mesh) 
{
	std::lock_guard<std::recursive_mutex> lock(*Entity::getEditMutex().get());

	auto &entity = getStruct();
	if (!mesh) throw std::runtime_error( std::string("Invalid mesh handle."));
	if (!mesh->isInitialized()) throw std::runtime_error("Error, mesh not initialized");
	entity.mesh_id = mesh->getId();
	mesh->entities.insert(id);
	markDirty();
}

void Entity::clearMesh()
{
	std::lock_guard<std::recursive_mutex> lock(*Entity::getEditMutex().get());
	
	auto &entity = getStruct();
	auto meshes = Mesh::getFront();
	if (entity.mesh_id != -1) meshes[entity.mesh_id].entities.erase(id);
	entity.mesh_id = -1;
	markDirty();
}

Mesh* Entity::getMesh()
{
	auto &entity = getStruct();
	if ((entity.mesh_id < 0) || (entity.mesh_id >= Mesh::getCount()))  return nullptr;
	auto &mesh = Mesh::getFront()[entity.mesh_id];
	if (!mesh.isInitialized()) return nullptr;
	return &mesh;
}

void Entity::setVisibility(bool camera)
{
	std::lock_guard<std::recursive_mutex> lock(*Entity::getEditMutex().get());
	
	auto &entity = getStruct();
	if (camera) {
		entity.flags |= ENTITY_VISIBILITY_CAMERA_RAYS;
	} else {
		entity.flags &= (~ENTITY_VISIBILITY_CAMERA_RAYS);
	}
	markDirty();
}

glm::vec3 Entity::getMinAabbCorner()
{
	return entityStructs[id].bbmin;
}

glm::vec3 Entity::getMaxAabbCorner()
{
	return entityStructs[id].bbmax;
}

glm::vec3 Entity::getAabbCenter()
{
	return entityStructs[id].bbmin + (entityStructs[id].bbmax - entityStructs[id].bbmin) * .5f;
}

void Entity::initializeFactory()
{
	if (isFactoryInitialized()) return;
	entities.resize(1000000);
	entityStructs.resize(1000000);
	editMutex = std::make_shared<std::recursive_mutex>();
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
	return dirtyEntities.size() > 0;
}

std::set<Entity*> Entity::getDirtyEntities()
{
	return dirtyEntities;
}

std::set<Entity*> Entity::getRenderableEntities()
{
	return renderableEntities;
}

void Entity::markDirty() {
	dirtyEntities.insert(this);
	// if (transformChanged || meshChanged || materialAssigned || lightAssigned)
	// todo, optimize this...
	{
		updateRenderables();
		computeAabb();
	}
};

void Entity::computeAabb()
{
	if ((getMesh() == nullptr) || (getTransform() == nullptr)) {
		entityStructs[id].bbmin = entityStructs[id].bbmax = vec4(0.f);
	}
	else {
		mat4 ltw = getTransform()->getLocalToWorldMatrix();
		vec3 lbbmin = vec3(ltw * vec4(getMesh()->getMinAabbCorner(), 1.f));
		vec3 lbbmax = vec3(ltw * vec4(getMesh()->getMaxAabbCorner(), 1.f));
		vec3 p[8];
		p[0] = vec3(lbbmin.x, lbbmin.y, lbbmin.z);
		p[1] = vec3(lbbmin.x, lbbmin.y, lbbmax.z);
		p[2] = vec3(lbbmin.x, lbbmax.y, lbbmin.z);
		p[3] = vec3(lbbmin.x, lbbmax.y, lbbmax.z);
		p[4] = vec3(lbbmax.x, lbbmin.y, lbbmin.z);
		p[5] = vec3(lbbmax.x, lbbmin.y, lbbmax.z);
		p[6] = vec3(lbbmax.x, lbbmax.y, lbbmin.z);
		p[7] = vec3(lbbmax.x, lbbmax.y, lbbmax.z);
		vec3 bbmin = p[0], bbmax = p[0];
		for (int i = 1; i < 8; ++i) {
			bbmin = glm::min(bbmin, p[i]);
			bbmax = glm::max(bbmax, p[i]);
		}
		entityStructs[id].bbmin = vec4(bbmin, 1.f);
		entityStructs[id].bbmax = vec4(bbmax, 1.f);
	}

	updateSceneAabb(this);
}

void Entity::updateRenderables() 
{
	if (!isInitialized()) renderableEntities.erase(this);
	else if (!getTransform()) renderableEntities.erase(this);
	else if (!getMesh()) renderableEntities.erase(this);
	else if (!(getMaterial() || getLight())) renderableEntities.erase(this);
	else renderableEntities.insert(this);
}

void Entity::updateComponents()
{
	if (dirtyEntities.size() == 0) return;
	// for (auto &e : dirtyEntities) {
	// 	if (!e->isInitialized()) continue;
	// 	//
	// }
	dirtyEntities.clear();
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
	auto createEntity = [transform, material, mesh, light, camera] (Entity* entity) {
		entity->setVisibility(true);
		if (transform) entity->setTransform(transform);
		if (material) entity->setMaterial(material);
		if (camera) entity->setCamera(camera);
		if (mesh) entity->setMesh(mesh);
		if (light) entity->setLight(light);
		dirtyEntities.insert(entity);
	};
	try {
		return StaticFactory::create<Entity>(editMutex, name, "Entity", lookupTable, entities.data(), entities.size(), createEntity);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Entity", lookupTable, entities.data(), entities.size());
		throw;
	}
}

std::shared_ptr<std::recursive_mutex> Entity::getEditMutex()
{
	return editMutex;
}

Entity* Entity::get(std::string name) {
	return StaticFactory::get(editMutex, name, "Entity", lookupTable, entities.data(), entities.size());
}

void Entity::remove(std::string name) {
	auto entity = Entity::get(name);
	if (!entity) return;
	entity->clearCamera();
	entity->clearLight();
	entity->clearMaterial();
	entity->clearMesh();
	entity->clearTransform();
	int32_t oldID = entity->getId();
	StaticFactory::remove(editMutex, name, "Entity", lookupTable, entities.data(), entities.size());
	dirtyEntities.insert(&entities[oldID]);
}

EntityStruct* Entity::getFrontStruct() {
	return entityStructs.data();
}

Entity* Entity::getFront() {
	return entities.data();
}

uint32_t Entity::getCount() {
	return entities.size();
}

std::string Entity::getName()
{
    return name;
}

int32_t Entity::getId()
{
    return id;
}

std::map<std::string, uint32_t> Entity::getNameToIdMap()
{
	return lookupTable;
}