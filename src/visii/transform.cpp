#include <visii/transform.h>
#include <visii/entity.h>

Transform Transform::transforms[MAX_TRANSFORMS];
TransformStruct Transform::transformStructs[MAX_TRANSFORMS];
std::map<std::string, uint32_t> Transform::lookupTable;

std::shared_ptr<std::mutex> Transform::editMutex;
bool Transform::factoryInitialized = false;
bool Transform::anyDirty = true;

void Transform::initializeFactory()
{
	if (isFactoryInitialized()) return;
	editMutex = std::make_shared<std::mutex>();
	factoryInitialized = true;
}

bool Transform::isFactoryInitialized()
{
	return factoryInitialized;
}

bool Transform::isInitialized()
{
	return initialized;
}

bool Transform::areAnyDirty()
{
	return anyDirty;
}

void Transform::markDirty() {
	dirty = true;
	anyDirty = true;
	auto entityPointers = Entity::getFront();
	for (auto &eid : entities) {
		entityPointers[eid].markDirty();
	}
};

void Transform::updateComponents() 
{
	for (int i = 0; i < MAX_TRANSFORMS; ++i) {
		if (!transforms[i].isFactoryInitialized()) continue;
		transformStructs[i].worldToLocal = transforms[i].getWorldToLocalMatrix();
		transformStructs[i].localToWorld = transforms[i].getLocalToWorldMatrix();
		transforms[i].markClean();
	};
	anyDirty = false;
}

void Transform::clearAll() 
{
	if (!isFactoryInitialized()) return;

	for (auto &transform : transforms) {
		if (transform.initialized) {
			Transform::remove(transform.name);
		}
	}
}


/* Static Factory Implementations */
Transform* Transform::create(std::string name, 
	vec3 scale, quat rotation, vec3 position) 
{
	auto t = StaticFactory::create(editMutex, name, "Transform", lookupTable, transforms, MAX_TRANSFORMS);
	t->setPosition(position);
	t->setRotation(rotation);
	t->setScale(scale);
	anyDirty = true;
	return t;
}

std::shared_ptr<std::mutex> Transform::getEditMutex()
{
	return editMutex;
}

Transform* Transform::get(std::string name) {
	return StaticFactory::get(editMutex, name, "Transform", lookupTable, transforms, MAX_TRANSFORMS);
}

void Transform::remove(std::string name) {
	StaticFactory::remove(editMutex, name, "Transform", lookupTable, transforms, MAX_TRANSFORMS);
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

std::map<std::string, uint32_t> Transform::getNameToIdMap()
{
	return lookupTable;
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
	// output += "\tid: \"" + std::to_string(id) + "\",\n";
	// output += "\tscale: " + glm::to_string(getScale()) + "\n";
	// output += "\tposition: " + glm::to_string(getPosition()) + "\n";
	// output += "\trotation: " + glm::to_string(getRotation()) + "\n";
	// output += "\tright: " + glm::to_string(right) + "\n";
	// output += "\tup: " + glm::to_string(up) + "\n";
	// output += "\tforward: " + glm::to_string(forward) + "\n";
	// output += "\tlocal_to_parent_matrix: " + glm::to_string(getLocalToParentMatrix()) + "\n";
	// output += "\tparent_to_local_matrix: " + glm::to_string(getParentToLocalMatrix()) + "\n";
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

glm::quat safeQuatLookAt(
    glm::vec3 const& lookFrom,
    glm::vec3 const& lookTo,
    glm::vec3 const& up,
    glm::vec3 const& alternativeUp)
{
    glm::vec3  direction       = lookTo - lookFrom;
    float      directionLength = glm::length(direction);

    // Check if the direction is valid; Also deals with NaN
    if(!(directionLength > 0.0001))
        return glm::quat(1, 0, 0, 0); // Just return identity

    // Normalize direction
    direction /= directionLength;

    // Is the normal up (nearly) parallel to direction?
    if(glm::abs(glm::dot(direction, up)) > .9999f) {
        // Use alternative up
        return glm::quatLookAt(direction, alternativeUp);
    }
    else {
        return glm::quatLookAt(direction, up);
    }
}

void Transform::lookAt(vec3 at, vec3 up, vec3 eye)
{
	if (glm::any(glm::isnan(eye))) {
		eye = this->position;
	} else {
		setPosition(eye);
	}
	up = normalize(up);
	glm::vec3 forward = glm::normalize(at - eye);
	glm::quat rotation = safeQuatLookAt(eye, at, up, up);
	setRotation(rotation);
}

// void Transform::rotateAround(vec3 point, float angle, vec3 axis)
// {
// 	glm::vec3 direction = point - getPosition();
// 	glm::vec3 newPosition = getPosition() + direction;
// 	glm::quat newRotation = glm::angleAxis(angle, axis) * getRotation();
// 	newPosition = newPosition - direction * glm::angleAxis(-angle, axis);

// 	rotation = glm::normalize(newRotation);
// 	localToParentRotation = glm::toMat4(rotation);
// 	parentToLocalRotation = glm::inverse(localToParentRotation);

// 	position = newPosition;
// 	localToParentTranslation = glm::translate(glm::mat4(1.0), position);
// 	parentToLocalTranslation = glm::translate(glm::mat4(1.0), -position);

// 	updateMatrix();
// 	markDirty();
// }

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
		// rotation = glm::conjugate(rotation);

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

// void Transform::setRotation(float angle, vec3 axis)
// {
// 	setRotation(glm::angleAxis(angle, axis));
// 	markDirty();
// }

void Transform::addRotation(quat additionalRotation)
{
	setRotation(getRotation() * additionalRotation);
	updateRotation();
	markDirty();
}

// void Transform::addRotation(float angle, vec3 axis)
// {
// 	addRotation(glm::angleAxis(angle, axis));
// 	markDirty();
// }

void Transform::updateRotation()
{
	localToParentRotation = glm::toMat4(rotation);
	parentToLocalRotation = glm::inverse(localToParentRotation);

	nextLocalToParentRotation = glm::toMat4(angularVelocity * rotation);
	nextParentToLocalRotation = glm::inverse(localToParentRotation);
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

void Transform::setLinearVelocity(vec3 newLinearVelocity, float framesPerSecond, float mix)
{
	mix = glm::clamp(mix, 0.f, 1.f);
	newLinearVelocity /= framesPerSecond;
	linearVelocity = glm::mix(newLinearVelocity, linearVelocity, mix);
	updatePosition();
	markDirty();
}

void Transform::setAngularVelocity(quat newAngularVelocity, float framesPerSecond, float mix)
{
	mix = glm::clamp(mix, 0.f, 1.f);
	newAngularVelocity[0] = newAngularVelocity[0] / framesPerSecond;
	newAngularVelocity[1] = newAngularVelocity[1] / framesPerSecond;
	newAngularVelocity[2] = newAngularVelocity[2] / framesPerSecond;
	angularVelocity = glm::lerp(newAngularVelocity, angularVelocity, mix);
	updateRotation();
	markDirty();
}

void Transform::setScalarVelocity(vec3 newScalarVelocity, float framesPerSecond, float mix)
{
	mix = glm::clamp(mix, 0.f, 1.f);
	newScalarVelocity /= framesPerSecond;
	scalarVelocity = glm::mix(newScalarVelocity, scalarVelocity, mix);
	updateScale();
	markDirty();
}

// void Transform::setPosition(float x, float y, float z)
// {
// 	setPosition(glm::vec3(x, y, z));
// 	markDirty();
// }

// void Transform::addPosition(float dx, float dy, float dz)
// {
// 	addPosition(glm::vec3(dx, dy, dz));
// 	markDirty();
// }

void Transform::updatePosition()
{
	localToParentTranslation = glm::translate(glm::mat4(1.0), position);
	parentToLocalTranslation = glm::translate(glm::mat4(1.0), -position);
	nextLocalToParentTranslation = glm::translate(glm::mat4(1.0), position + linearVelocity);
	nextParentToLocalTranslation = glm::translate(glm::mat4(1.0), -position + linearVelocity);
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

// void Transform::setScale(float newScale)
// {
// 	scale = vec3(newScale, newScale, newScale);
// 	updateScale();
// 	markDirty();
// }

void Transform::addScale(vec3 additionalScale)
{
	setScale(getScale() + additionalScale);
	updateScale();
	markDirty();
}

// void Transform::setScale(float x, float y, float z)
// {
// 	setScale(glm::vec3(x, y, z));
// 	markDirty();
// }

// void Transform::addScale(float dx, float dy, float dz)
// {
// 	addScale(glm::vec3(dx, dy, dz));
// 	markDirty();
// }

// void Transform::addScale(float ds)
// {
// 	addScale(glm::vec3(ds, ds, ds));
// 	markDirty();
// }

void Transform::updateScale()
{
	localToParentScale = glm::scale(glm::mat4(1.0), scale);
	parentToLocalScale = glm::scale(glm::mat4(1.0), glm::vec3(1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z));
	nextLocalToParentScale = glm::scale(glm::mat4(1.0), scale + scalarVelocity);
	nextParentToLocalScale = glm::scale(glm::mat4(1.0), glm::vec3(1.0 / (scale.x + scalarVelocity.x), 1.0 / (scale.y + scalarVelocity.y), 1.0 / (scale.z + scalarVelocity.z)));
	updateMatrix();
	markDirty();
}

void Transform::updateMatrix()
{
	localToParentMatrix = (localToParentTransform * localToParentTranslation * localToParentRotation * localToParentScale);
	parentToLocalMatrix = (parentToLocalScale * parentToLocalRotation * parentToLocalTranslation * parentToLocalTransform);

	nextLocalToParentMatrix = (localToParentTransform * nextLocalToParentTranslation * nextLocalToParentRotation * nextLocalToParentScale);
	nextParentToLocalMatrix = (nextParentToLocalScale * nextParentToLocalRotation * nextParentToLocalTranslation * parentToLocalTransform);

	right = glm::vec3(localToParentMatrix[0]);
	up = glm::vec3(localToParentMatrix[1]);
	forward = glm::vec3(localToParentMatrix[2]);
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

glm::mat4 Transform::computeNextWorldToLocalMatrix()
{
	glm::mat4 parentMatrix = glm::mat4(1.0);
	if (parent != -1) {
		parentMatrix = transforms[parent].computeNextWorldToLocalMatrix();
		return getNextParentToLocalMatrix() * parentMatrix;
	}
	else return getNextParentToLocalMatrix();
}

void Transform::updateWorldMatrix()
{
	if (parent == -1) {
		worldToLocalMatrix = parentToLocalMatrix;
		localToWorldMatrix = localToParentMatrix;
		nextWorldToLocalMatrix = nextParentToLocalMatrix;
		nextLocalToWorldMatrix = nextLocalToParentMatrix;

		worldScale = scale;
		worldTranslation = position;
		worldRotation = rotation;
		worldSkew = glm::vec3(0.f, 0.f, 0.f);
		worldPerspective = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f); // not sure what this should default to...
	} else {
		worldToLocalMatrix = computeWorldToLocalMatrix();
		nextWorldToLocalMatrix = computeNextWorldToLocalMatrix();
		localToWorldMatrix = glm::inverse(worldToLocalMatrix); 
		nextLocalToWorldMatrix = glm::inverse(worldToLocalMatrix); 
		glm::decompose(localToWorldMatrix, worldScale, worldRotation, worldTranslation, worldSkew, worldPerspective);
		// glm::decompose(nextLocalToWorldMatrix, worldScale, worldRotation, worldTranslation, worldSkew, worldPerspective);
	}
	markDirty();
}

glm::mat4 Transform::getParentToLocalMatrix()
{
	return parentToLocalMatrix;
}

glm::mat4 Transform::getNextParentToLocalMatrix()
{
	return nextParentToLocalMatrix;
}

glm::mat4 Transform::getLocalToParentMatrix()
{
	return localToParentMatrix;
}

glm::mat4 Transform::getNextLocalToParentMatrix()
{
	return nextLocalToParentMatrix;
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

void Transform::setParent(Transform *parent) {
	if (!parent)
		throw std::runtime_error(std::string("Error: parent is empty"));

	if (!parent->isInitialized())
		throw std::runtime_error(std::string("Error: parent is uninitialized"));
	
	if (parent->getId() == this->getId())
		throw std::runtime_error(std::string("Error: a transform cannot be the parent of itself"));

	this->parent = parent->getId();
	transforms[parent->getId()].children.insert(this->id);
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

void Transform::addChild(Transform *object) {
	if (!object)
		throw std::runtime_error(std::string("Error: child is empty"));

	if (!object->isInitialized())
		throw std::runtime_error(std::string("Error: child is uninitialized"));
	
	if (object->getId() == this->getId())
		throw std::runtime_error(std::string("Error: a transform cannot be the child of itself"));

	children.insert(object->getId());
	transforms[object->getId()].parent = this->id;
	transforms[object->getId()].updateWorldMatrix();
	transforms[object->getId()].markDirty();
}

void Transform::removeChild(Transform *object) {
	if (!object)
		throw std::runtime_error(std::string("Error: child is empty"));

	if (!object->isInitialized())
		throw std::runtime_error(std::string("Error: child is uninitialized"));
	
	if (object->getId() == this->getId())
		throw std::runtime_error(std::string("Error: a transform cannot be the child of itself"));

	children.erase(object->getId());
	transforms[object->getId()].parent = -1;
	transforms[object->getId()].updateWorldMatrix();
	transforms[object->getId()].markDirty();
}

glm::mat4 Transform::getWorldToLocalMatrix() {
	return worldToLocalMatrix;
}

glm::mat4 Transform::getLocalToWorldMatrix() {
	return localToWorldMatrix;
}

glm::mat4 Transform::getNextLocalToWorldMatrix() {
	return nextLocalToWorldMatrix;
}

glm::quat Transform::getWorldRotation() {
	return worldRotation;
}

glm::vec3 Transform::getWorldTranslation() {
	return worldTranslation;
}

glm::vec3 Transform::getWorldScale() {
	return worldScale;
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