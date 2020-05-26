#include <visii/transform.h>
#include <visii/entity.h>

Transform Transform::transforms[MAX_TRANSFORMS];
TransformStruct Transform::transformStructs[MAX_TRANSFORMS];
std::map<std::string, uint32_t> Transform::lookupTable;

std::shared_ptr<std::mutex> Transform::creationMutex;
bool Transform::factoryInitialized = false;
bool Transform::anyDirty = true;

void Transform::initializeFactory()
{
	if (isFactoryInitialized()) return;
	creationMutex = std::make_shared<std::mutex>();
	factoryInitialized = true;
}

bool Transform::isFactoryInitialized()
{
	return factoryInitialized;
}

bool Transform::areAnyDirty()
{
	return anyDirty;
}

void Transform::markDirty() {
	dirty = true;
	anyDirty = true;
	for (auto &eid : entities) {
		Entity::get(eid)->markDirty();
	}
};

void Transform::updateComponents() 
{
	for (int i = 0; i < MAX_TRANSFORMS; ++i) {
		if (!transforms[i].isFactoryInitialized()) continue;
		transformStructs[i].worldToLocalPrev = transformStructs[i].worldToLocal;
		transformStructs[i].localToWorldPrev = transformStructs[i].localToWorld;
		transformStructs[i].worldToLocalRotationPrev = transformStructs[i].worldToLocalRotation;
		transformStructs[i].worldToLocalTranslationPrev = transformStructs[i].worldToLocalTranslation;

		transformStructs[i].worldToLocal = transforms[i].getWorldToLocalMatrix();
		transformStructs[i].localToWorld = transforms[i].getLocalToWorldMatrix();
		transformStructs[i].worldToLocalRotation = transforms[i].getWorldToLocalRotationMatrix();
		transformStructs[i].worldToLocalTranslation = transforms[i].getWorldToLocalTranslationMatrix();
		transforms[i].markClean();
	};
	anyDirty = false;
}

void Transform::cleanUp() 
{
	if (!isFactoryInitialized()) return;

	for (auto &transform : transforms) {
		if (transform.initialized) {
			Transform::remove(transform.id);
		}
	}

	factoryInitialized = false;
}


/* Static Factory Implementations */
Transform* Transform::create(std::string name) {
	auto t = StaticFactory::create(creationMutex, name, "Transform", lookupTable, transforms, MAX_TRANSFORMS);
	anyDirty = true;
	return t;
}

Transform* Transform::get(std::string name) {
	return StaticFactory::get(creationMutex, name, "Transform", lookupTable, transforms, MAX_TRANSFORMS);
}

Transform* Transform::get(uint32_t id) {
	return StaticFactory::get(creationMutex, id, "Transform", lookupTable, transforms, MAX_TRANSFORMS);
}

void Transform::remove(std::string name) {
	StaticFactory::remove(creationMutex, name, "Transform", lookupTable, transforms, MAX_TRANSFORMS);
}

void Transform::remove(uint32_t id) {
	StaticFactory::remove(creationMutex, id, "Transform", lookupTable, transforms, MAX_TRANSFORMS);
}

TransformStruct* Transform::getFrontStruct()
{
	return transformStructs;
}

Transform* Transform::getFront() {
	return transforms;
}

uint32_t Transform::getCount() {
	return MAX_TRANSFORMS;
}

Transform::Transform() { 
	initialized = false;
}

Transform::Transform(std::string name, uint32_t id) {
	initialized = true; this->name = name; this->id = id;
}

std::string Transform::toString()
{
	std::string output;
	output += "{\n";
	output += "\ttype: \"Transform\",\n";
	output += "\tname: \"" + name + "\",\n";
	output += "\tid: \"" + std::to_string(id) + "\",\n";
	output += "\tscale: " + glm::to_string(getScale()) + "\n";
	output += "\tposition: " + glm::to_string(getPosition()) + "\n";
	output += "\trotation: " + glm::to_string(getRotation()) + "\n";
	output += "\tright: " + glm::to_string(right) + "\n";
	output += "\tup: " + glm::to_string(up) + "\n";
	output += "\tforward: " + glm::to_string(forward) + "\n";
	output += "\tlocal_to_parent_matrix: " + glm::to_string(getLocalToParentMatrix()) + "\n";
	output += "\tparent_to_local_matrix: " + glm::to_string(getParentToLocalMatrix()) + "\n";
	output += "}";
	return output;
}

vec3 Transform::transformDirection(vec3 direction)
{
	return vec3(localToParentRotation * vec4(direction, 0.0));
}

vec3 Transform::transformPoint(vec3 point)
{
	return vec3(localToParentMatrix * vec4(point, 1.0));
}

vec3 Transform::transformVector(vec3 vector)
{
	return vec3(localToParentMatrix * vec4(vector, 0.0));
}

vec3 Transform::inverseTransformDirection(vec3 direction)
{
	return vec3(parentToLocalRotation * vec4(direction, 0.0));
}

vec3 Transform::inverseTransformPoint(vec3 point)
{
	return vec3(parentToLocalMatrix * vec4(point, 1.0));
}

vec3 Transform::inverseTransformVector(vec3 vector)
{
	return vec3(localToParentMatrix * vec4(vector, 0.0));
}

/*
Rotates the transform so the forward vector points at the target's current position.
Then it rotates the transform to point its up direction vector in the direction hinted at 
by the parentUp vector.
*/

// void Transform::look_at(vec3 point)
// {
// 	if (glm::distance2(point, position) <= 1e-10f) return;
// 	glm::vec3 to = glm::normalize(point - position);
// 	if (glm::distance2(forward, to) <= 1e-10f) return;
// 	glm::vec3 axis = glm::normalize(glm::cross(forward, to));
// 	float amount = glm::dot(forward, to);
// 	add_rotation(amount, axis);
// }

void Transform::rotateAround(vec3 point, float angle, vec3 axis)
{
	glm::vec3 direction = point - getPosition();
	glm::vec3 newPosition = getPosition() + direction;
	glm::quat newRotation = glm::angleAxis(angle, axis) * getRotation();
	newPosition = newPosition - direction * glm::angleAxis(-angle, axis);

	rotation = glm::normalize(newRotation);
	localToParentRotation = glm::toMat4(rotation);
	parentToLocalRotation = glm::inverse(localToParentRotation);

	position = newPosition;
	localToParentTranslation = glm::translate(glm::mat4(1.0), position);
	parentToLocalTranslation = glm::translate(glm::mat4(1.0), -position);

	updateMatrix();
	markDirty();
}

void Transform::rotateAround(vec3 point, glm::quat rot)
{
	glm::vec3 direction = point - getPosition();
	glm::vec3 newPosition = getPosition() + direction;
	glm::quat newRotation = rot * getRotation();
	newPosition = newPosition - direction * glm::inverse(rot);

	rotation = glm::normalize(newRotation);
	localToParentRotation = glm::toMat4(rotation);
	parentToLocalRotation = glm::inverse(localToParentRotation);

	position = newPosition;
	localToParentTranslation = glm::translate(glm::mat4(1.0), position);
	parentToLocalTranslation = glm::translate(glm::mat4(1.0), -position);

	updateMatrix();
	markDirty();
}

void Transform::setTransform(glm::mat4 transformation, bool decompose)
{
	if (decompose)
	{
		glm::vec3 scale;
		glm::quat rotation;
		glm::vec3 translation;
		glm::vec3 skew;
		glm::vec4 perspective;
		glm::decompose(transformation, scale, rotation, translation, skew, perspective);

		/* Decomposition can return negative scales. We make the assumption this is impossible.*/

		if (scale.x < 0.0) scale.x *= -1;
		if (scale.y < 0.0) scale.y *= -1;
		if (scale.z < 0.0) scale.z *= -1;
		scale = glm::max(scale, glm::vec3(.0001f));
		
		if (!(glm::any(glm::isnan(translation))))
			setPosition(translation);
		if (!(glm::any(glm::isnan(scale))))
			setScale(scale);
		if (!(glm::any(glm::isnan(rotation))))
			setRotation(rotation);
	}
	else {
		this->localToParentTransform = transformation;
		this->parentToLocalTransform = glm::inverse(transformation);
		updateMatrix();
	}
	markDirty();
}

quat Transform::getRotation()
{
	return rotation;
}

void Transform::setRotation(quat newRotation)
{
	rotation = glm::normalize(newRotation);
	updateRotation();
	markDirty();
}

void Transform::setRotation(float angle, vec3 axis)
{
	setRotation(glm::angleAxis(angle, axis));
	markDirty();
}

void Transform::addRotation(quat additionalRotation)
{
	setRotation(getRotation() * additionalRotation);
	updateRotation();
	markDirty();
}

void Transform::addRotation(float angle, vec3 axis)
{
	addRotation(glm::angleAxis(angle, axis));
	markDirty();
}

void Transform::updateRotation()
{
	localToParentRotation = glm::toMat4(rotation);
	parentToLocalRotation = glm::inverse(localToParentRotation);
	updateMatrix();
	markDirty();
}

vec3 Transform::getPosition()
{
	return position;
}

vec3 Transform::getRight()
{
	return right;
}

vec3 Transform::getUp()
{
	return up;
}

vec3 Transform::getForward()
{
	return forward;
}

void Transform::setPosition(vec3 newPosition)
{
	position = newPosition;
	updatePosition();
	markDirty();
}

void Transform::addPosition(vec3 additionalPosition)
{
	setPosition(getPosition() + additionalPosition);
	updatePosition();
	markDirty();
}

void Transform::setPosition(float x, float y, float z)
{
	setPosition(glm::vec3(x, y, z));
	markDirty();
}

void Transform::addPosition(float dx, float dy, float dz)
{
	addPosition(glm::vec3(dx, dy, dz));
	markDirty();
}

void Transform::updatePosition()
{
	localToParentTranslation = glm::translate(glm::mat4(1.0), position);
	parentToLocalTranslation = glm::translate(glm::mat4(1.0), -position);
	updateMatrix();
	markDirty();
}

vec3 Transform::getScale()
{
	return scale;
}

void Transform::setScale(vec3 newScale)
{
	scale = newScale;
	updateScale();
	markDirty();
}

void Transform::setScale(float newScale)
{
	scale = vec3(newScale, newScale, newScale);
	updateScale();
	markDirty();
}

void Transform::addScale(vec3 additionalScale)
{
	setScale(getScale() + additionalScale);
	updateScale();
	markDirty();
}

void Transform::setScale(float x, float y, float z)
{
	setScale(glm::vec3(x, y, z));
	markDirty();
}

void Transform::addScale(float dx, float dy, float dz)
{
	addScale(glm::vec3(dx, dy, dz));
	markDirty();
}

void Transform::addScale(float ds)
{
	addScale(glm::vec3(ds, ds, ds));
	markDirty();
}

void Transform::updateScale()
{
	localToParentScale = glm::scale(glm::mat4(1.0), scale);
	parentToLocalScale = glm::scale(glm::mat4(1.0), glm::vec3(1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z));
	updateMatrix();
	markDirty();
}

void Transform::updateMatrix()
{
	localToParentMatrix = (localToParentTransform * localToParentTranslation * localToParentRotation * localToParentScale);
	parentToLocalMatrix = (parentToLocalScale * parentToLocalRotation * parentToLocalTranslation * parentToLocalTransform);

	right = glm::vec3(localToParentMatrix[0]);
	forward = glm::vec3(localToParentMatrix[1]);
	up = glm::vec3(localToParentMatrix[2]);
	position = glm::vec3(localToParentMatrix[3]);

	updateChildren();
	markDirty();
}

glm::mat4 Transform::computeWorldToLocalMatrix()
{
	glm::mat4 parentMatrix = glm::mat4(1.0);
	if (parent != -1) {
		parentMatrix = transforms[parent].computeWorldToLocalMatrix();
		return getParentToLocalMatrix() * parentMatrix;
	}
	else return getParentToLocalMatrix();
}

void Transform::updateWorldMatrix()
{
	if (parent == -1) {
		worldToLocalMatrix = parentToLocalMatrix;
		localToWorldMatrix = localToParentMatrix;

		worldScale = scale;
		worldTranslation = position;
		worldRotation = rotation;
		worldSkew = glm::vec3(0.f, 0.f, 0.f);
		worldPerspective = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f); // not sure what this should default to...
	} else {
		worldToLocalMatrix = computeWorldToLocalMatrix();
		localToWorldMatrix = glm::inverse(worldToLocalMatrix); 
		glm::decompose(localToWorldMatrix, worldScale, worldRotation, worldTranslation, worldSkew, worldPerspective);
	}
	markDirty();
}

glm::mat4 Transform::getParentToLocalMatrix()
{
	return /*(interpolation >= 1.0 ) ?*/ parentToLocalMatrix /*: glm::interpolate(glm::mat4(1.0), parentToLocalMatrix, interpolation)*/;
}

glm::mat4 Transform::getLocalToParentMatrix()
{
	return /*(interpolation >= 1.0 ) ?*/ localToParentMatrix /*: glm::interpolate(glm::mat4(1.0), localToParentMatrix, interpolation)*/;
}

glm::mat4 Transform::getLocalToParentTranslationMatrix()
{
	return localToParentTranslation;
}

glm::mat4 Transform::getLocalToParentScaleMatrix()
{
	return localToParentScale;
}

glm::mat4 Transform::getLocalToParentRotationMatrix()
{
	return localToParentRotation;
}

glm::mat4 Transform::getParentToLocalTranslationMatrix()
{
	return parentToLocalTranslation;
}

glm::mat4 Transform::getParentToLocalScaleMatrix()
{
	return parentToLocalScale;
}

glm::mat4 Transform::getParentToLocalRotationMatrix()
{
	return parentToLocalRotation;
}

void Transform::setParent(uint32_t parent) {
	if ((parent < 0) || (parent >= MAX_TRANSFORMS))
		throw std::runtime_error(std::string("Error: parent must be between 0 and ") + std::to_string(MAX_TRANSFORMS));
	
	if (parent == this->getId())
		throw std::runtime_error(std::string("Error: a component cannot be the parent of itself"));

	this->parent = parent;
	transforms[parent].children.insert(this->id);
	updateChildren();
	markDirty();
}

void Transform::clearParent()
{
	if ((parent < 0) || (parent >= MAX_TRANSFORMS)){
		parent = -1;
		return;
	}
	
	transforms[parent].children.erase(this->id);
	this->parent = -1;
	updateChildren();
	markDirty();
}

void Transform::addChild(uint32_t object) {
	if ((object < 0) || (object >= MAX_TRANSFORMS))
		throw std::runtime_error(std::string("Error: child must be between 0 and ") + std::to_string(MAX_TRANSFORMS));
	
	if (object == this->getId())
		throw std::runtime_error(std::string("Error: a component cannot be it's own child"));

	children.insert(object);
	transforms[object].parent = this->id;
	transforms[object].updateWorldMatrix();
	transforms[object].markDirty();
}

void Transform::removeChild(uint32_t object) {
	if ((object < 0) || (object >= MAX_TRANSFORMS))
		throw std::runtime_error(std::string("Error: child must be between 0 and ") + std::to_string(MAX_TRANSFORMS));
	
	if (object == this->getId())
		throw std::runtime_error(std::string("Error: a component cannot be it's own child"));

	if (children.find(object) == children.end()) 
		throw std::runtime_error(std::string("Error: child does not exist"));

	children.erase(object);
	transforms[object].parent = -1;
	transforms[object].updateWorldMatrix();
	transforms[object].markDirty();
}

glm::mat4 Transform::getWorldToLocalMatrix() {
	return worldToLocalMatrix;
}

glm::mat4 Transform::getLocalToWorldMatrix() {
	return localToWorldMatrix;
}

glm::quat Transform::getWorldRotation() {
	return worldRotation;
}

glm::vec3 Transform::getWorldTranslation() {
	return worldTranslation;
}

glm::mat4 Transform::getWorldToLocalRotationMatrix()
{
	return glm::toMat4(glm::inverse(worldRotation));
}

glm::mat4 Transform::getLocalToWorldRotationMatrix()
{
	return glm::toMat4(worldRotation);
}

glm::mat4 Transform::getWorldToLocalTranslationMatrix()
{
	glm::mat4 m(1.0);
	m = glm::translate(m, -1.0f * worldTranslation);
	return m;
}

glm::mat4 Transform::getLocalToWorldTranslationMatrix()
{
	glm::mat4 m(1.0);
	m = glm::translate(m, worldTranslation);
	return m;
}

glm::mat4 Transform::getWorldToLocalScaleMatrix()
{
	glm::mat4 m(1.0);
	m = glm::scale(m, 1.0f / worldScale);
	return m;
}

glm::mat4 Transform::getLocalToWorldScaleMatrix()
{
	glm::mat4 m(1.0);
	m = glm::scale(m, worldScale);
	return m;
}

void Transform::updateChildren()
{
	for (auto &c : children) {
		auto &t = transforms[c];
		t.updateChildren();
	}

	updateWorldMatrix();
	markDirty();
}

TransformStruct &Transform::getStruct()
{
	return transformStructs[id];
}