#include <visii/transform.h>
#include <visii/entity.h>
#include <glm/gtx/matrix_decompose.hpp>

Transform Transform::transforms[MAX_TRANSFORMS];
TransformStruct Transform::transformStructs[MAX_TRANSFORMS];
std::map<std::string, uint32_t> Transform::lookupTable;

std::shared_ptr<std::mutex> Transform::editMutex;
bool Transform::factoryInitialized = false;
std::set<Transform*> Transform::dirtyTransforms;
static bool mutexAcquired = false;

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

bool Transform::areAnyDirty() {
	return dirtyTransforms.size() > 0;
};

std::set<Transform*> Transform::getDirtyTransforms()
{
	return dirtyTransforms;
}

void Transform::updateComponents() 
{
	if (dirtyTransforms.size() == 0) return;
	for (auto &t : dirtyTransforms) {
		if (!t->isInitialized()) continue;
		transformStructs[t->id].worldToLocal = t->getWorldToLocalMatrix();
		transformStructs[t->id].localToWorld = t->getLocalToWorldMatrix();
	}
	dirtyTransforms.clear();
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
	auto createTransform = [scale, rotation, position] (Transform* transform) {
		dirtyTransforms.insert(transform);
		transform->setPosition(position);
		transform->setRotation(rotation);
		transform->setScale(scale);
	};

	try {
		return StaticFactory::create<Transform>(editMutex, name, "Transform", lookupTable, transforms, MAX_TRANSFORMS, createTransform);
	}
	catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Transform", lookupTable, transforms, MAX_TRANSFORMS);
		throw;
	}
}

Transform* Transform::createFromMatrix(std::string name, mat4 xfm) 
{
	auto createTransform = [xfm] (Transform* transform) {
		dirtyTransforms.insert(transform);
		transform->setTransform(xfm);
	};

	try {
		return StaticFactory::create<Transform>(editMutex, name, "Transform", lookupTable, transforms, MAX_TRANSFORMS, createTransform);
	}
	catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Transform", lookupTable, transforms, MAX_TRANSFORMS);
		throw;
	}
}

std::shared_ptr<std::mutex> Transform::getEditMutex()
{
	return editMutex;
}

Transform* Transform::get(std::string name) {
	return StaticFactory::get(editMutex, name, "Transform", lookupTable, transforms, MAX_TRANSFORMS);
}

void Transform::remove(std::string name) {
	auto t = get(name);
	if (!t) return;
	int32_t oldID = t->getId();
	StaticFactory::remove(editMutex, name, "Transform", lookupTable, transforms, MAX_TRANSFORMS);
	dirtyTransforms.insert(&transforms[oldID]);
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

std::string Transform::getName()
{
    return name;
}

int32_t Transform::getId()
{
    return id;
}

std::map<std::string, uint32_t> Transform::getNameToIdMap()
{
	return lookupTable;
}

void Transform::markDirty() {
	dirtyTransforms.insert(this);
	auto entityPointers = Entity::getFront();
	for (auto &eid : entities) {
		entityPointers[eid].markDirty();
	}
};

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

vec3 Transform::transformDirection(vec3 direction, bool previous)
{
	return vec3(getLocalToParentRotationMatrix(previous) * vec4(direction, 0.0));
}

vec3 Transform::transformPoint(vec3 point, bool previous)
{
	return vec3(getLocalToParentMatrix(previous) * vec4(point, 1.0));
}

vec3 Transform::transformVector(vec3 vector, bool previous)
{
	return vec3(getLocalToParentMatrix(previous) * vec4(vector, 0.0));
}

vec3 Transform::inverseTransformDirection(vec3 direction, bool previous)
{
	return vec3(getParentToLocalRotationMatrix(previous) * vec4(direction, 0.0));
}

vec3 Transform::inverseTransformPoint(vec3 point, bool previous)
{
	return vec3(getParentToLocalMatrix(previous) * vec4(point, 1.0));
}

vec3 Transform::inverseTransformVector(vec3 vector, bool previous)
{
	return vec3(getLocalToParentMatrix(previous) * vec4(vector, 0.0));
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

void Transform::lookAt(vec3 at, vec3 up, vec3 eye, bool previous)
{
	// if (!mutexAcquired) {
	// 	std::lock_guard<std::mutex>lock(*editMutex.get());
	// }
	if (previous) {
		useRelativeAngularMotionBlur = false;
	}
	if (glm::any(glm::isnan(eye))) {
		eye = (previous) ? this->prevPosition : this->position;
	} else {
		if (previous) {
			useRelativeLinearMotionBlur = false;
		}
		setPosition(eye, previous);
	}
	up = normalize(up);
	glm::vec3 forward = glm::normalize(at - eye);
	glm::quat rotation = safeQuatLookAt(eye, at, up, up);
	setRotation(rotation, previous);
}

// void Transform::nextLookAt(vec3 at, vec3 up, vec3 eye)
// {
// 	if (glm::any(glm::isnan(eye))) {
// 		eye = position;
// 	}
// 	up = normalize(up);
// 	// glm::vec3 forward = glm::normalize(at - eye);
// 	// glm::quat rotation = safeQuatLookAt(eye, at, up, up);
	

// 	nextWorldToLocalMatrix = glm::lookAt(eye, at, up);
// 	glm::mat4 dNext = nextWorldToLocalMatrix * localToWorldMatrix;
// 	glm::quat rot = glm::quat_cast(dNext);
// 	glm::vec4 vel = glm::column(dNext, 3);

// 	setLinearVelocity(vec3(vel));
// 	setAngularVelocity(rot);
// 	// nextLocalToWorldMatrix = inverse(nextWorldToLocalMatrix);
// 	// setAngularVelocity(rot * glm::inverse(rotation));
// 	// markDirty();

// }

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

void Transform::rotateAround(vec3 point, glm::quat rot, bool previous)
{
	if (previous) useRelativeAngularMotionBlur = false;
	glm::vec3 direction = point - getPosition(previous);
	glm::vec3 newPosition = getPosition(previous) + direction;
	glm::quat newRotation = rot * getRotation(previous);
	newPosition = newPosition - direction * glm::inverse(rot);

	glm::quat &r = (previous) ? prevRotation : rotation;
	glm::vec3 &t = (previous) ? prevPosition : position;

	// glm::mat4 &ltpr = (previous) ? prevLocalToParentRotation : localToParentRotation;
	// glm::mat4 &ptlr = (previous) ? prevParentToLocalRotation : parentToLocalRotation;
	// glm::mat4 &ltpt = (previous) ? prevLocalToParentTranslation : localToParentTranslation;
	// glm::mat4 &ptlt = (previous) ? prevParentToLocalTranslation : parentToLocalTranslation;
	r = glm::normalize(newRotation);
	// ltpr = glm::toMat4(rotation);
	// ptlr = glm::inverse(ltpr);

	t = newPosition;
	// ltpt = glm::translate(glm::mat4(1.0), t);
	// ptlt = glm::translate(glm::mat4(1.0), -t);

	updateMatrix();
	markDirty();
}

void Transform::setTransform(glm::mat4 transformation, bool decompose, bool previous)
{
	if (previous) {
		useRelativeLinearMotionBlur = false;
		useRelativeAngularMotionBlur = false;
		useRelativeScalarMotionBlur = false;
	}
	if (decompose)
	{
		// glm::vec3 scale;
		// glm::quat rotation;
		// glm::vec3 translation;

		// translation = glm::vec3(glm::column(transformation, 3));
		// glm::mat3 rot = glm::mat3(transformation);
		// float det = glm::determinant(rot);
		// // if (abs(det) < glm::epsilon<float>()) {
		// // 	throw std::runtime_error("Error: upper left 3x3 determinant must be non-zero");
		// // }
		// scale.x = length(glm::column(rot, 0));
		// scale.y = length(glm::column(rot, 1));
		// scale.z = length(glm::column(rot, 2));
		// if (det < 0) {
		// 	scale *= -1.f;
		// 	rot *= -1.f;
		// }
		// rotation = glm::normalize(glm::quat_cast(rot));

		glm::vec3 scale;
		glm::quat rotation;
		glm::vec3 translation;
		glm::vec3 skew;
		glm::vec4 perspective;
		bool worked = glm::decompose(transformation, scale, rotation, translation, skew, perspective);
		
		/* If the above didn't work, throw an exception */
		if (!worked) {
			throw std::runtime_error( 
				std::string("Decomposition failed! Is the product of the 4x4 with the determinant of the upper left 3x3 nonzero?")
				+ std::string("See Graphics Gems II: Decomposing a Matrix into Simple Transformations"));
			// setScale(vec3(1.f), previous);
			// setPosition(vec3(0.f), previous);
			// setRotation(quat(1.f, 0.f, 0.f, 0.f), previous);
			// setTransform(transformation, false, previous);
			// return;
		}

		if (glm::length(skew) > .0001f) {
			throw std::runtime_error( 
				std::string("Decomposition failed! Skew detected in the upper left 3x3.")
			);
			return;
		}
			
		/* Decomposition can return negative scales. We make the assumption this is impossible.*/
		if (scale.x < 0.0) scale.x *= -1;
		if (scale.y < 0.0) scale.y *= -1;
		if (scale.z < 0.0) scale.z *= -1;
		scale = glm::max(scale, glm::vec3(.0001f));
		
		
		if (!(glm::any(glm::isnan(translation))))
			setPosition(translation, previous);
		if (!(glm::any(glm::isnan(scale))))
			setScale(scale, previous);
		if (!(glm::any(glm::isnan(rotation))))
			setRotation(rotation, previous);
	}
	else {
		if (previous) {
			this->prevLocalToParentTransform = transformation;
			// this->prevParentToLocalTransform = glm::inverse(transformation);
		}
		else {
			this->localToParentTransform = transformation;
			// this->parentToLocalTransform = glm::inverse(transformation);
		}
		updateMatrix();
	}
	markDirty();
}

quat Transform::getRotation(bool previous)
{
	if (previous) return prevRotation;
	else return rotation;
}

void Transform::setRotation(quat newRotation, bool previous)
{
	if (previous) useRelativeAngularMotionBlur = false;
	auto &r = (previous) ? prevRotation : rotation;
	r = glm::normalize(newRotation);
	updateRotation();
	markDirty();
}

// void Transform::setRotation(float angle, vec3 axis)
// {
// 	setRotation(glm::angleAxis(angle, axis));
// 	markDirty();
// }

void Transform::addRotation(quat additionalRotation, bool previous)
{
	if (previous) useRelativeAngularMotionBlur = false;
	setRotation(getRotation(previous) * additionalRotation, previous);
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
	// localToParentRotation = glm::toMat4(rotation);
	// parentToLocalRotation = glm::inverse(localToParentRotation);

	// if (useRelativeMotionBlur) {
	// 	prevLocalToParentRotation = glm::toMat4(angularVelocity * rotation);
	// } else {
	// 	prevLocalToParentRotation = glm::toMat4(prevRotation);
	// }
	// prevParentToLocalRotation = glm::inverse(prevLocalToParentRotation);
	updateMatrix();
	markDirty();
}

vec3 Transform::getPosition(bool previous)
{
	if (previous) return prevPosition;
	else return position;
}

vec3 Transform::getRight(bool previous)
{
	if (previous) return glm::normalize(glm::vec3(glm::column(prevLocalToParentMatrix, 0))); 
	else return glm::normalize(glm::vec3(glm::column(localToParentMatrix, 0))); 
}

vec3 Transform::getUp(bool previous)
{
	if (previous) return glm::normalize(glm::vec3(glm::column(prevLocalToParentMatrix, 1))); 
	else return glm::normalize(glm::vec3(glm::column(localToParentMatrix, 1))); 
}

vec3 Transform::getForward(bool previous)
{
	if (previous) return glm::normalize(glm::vec3(glm::column(prevLocalToParentMatrix, 2))); 
	else return glm::normalize(glm::vec3(glm::column(localToParentMatrix, 2))); 
}

vec3 Transform::getWorldPosition(bool previous)
{
	if (previous) return glm::vec3(glm::column(prevLocalToWorldMatrix, 3)); 
	else return glm::vec3(glm::column(localToWorldMatrix, 3)); 
}

vec3 Transform::getWorldRight(bool previous)
{
	if (previous) return glm::normalize(glm::vec3(glm::column(prevLocalToWorldMatrix, 0))); 
	else return glm::normalize(glm::vec3(glm::column(localToWorldMatrix, 0))); 
}

vec3 Transform::getWorldUp(bool previous)
{
	if (previous) return glm::normalize(glm::vec3(glm::column(prevLocalToWorldMatrix, 1))); 
	else return glm::normalize(glm::vec3(glm::column(localToWorldMatrix, 1))); 
}

vec3 Transform::getWorldForward(bool previous)
{
	if (previous) return glm::normalize(glm::vec3(glm::column(prevLocalToWorldMatrix, 2))); 
	else return glm::normalize(glm::vec3(glm::column(localToWorldMatrix, 2))); 
}

void Transform::setPosition(vec3 newPosition, bool previous)
{
	if (previous) useRelativeLinearMotionBlur = false;
	auto &p = (previous) ? prevPosition : position;
	p = newPosition;
	updatePosition();
	markDirty();
}

void Transform::addPosition(vec3 additionalPosition, bool previous)
{
	if (previous) useRelativeLinearMotionBlur = false;
	setPosition(getPosition(previous) + additionalPosition, previous);
	updatePosition();
	markDirty();
}

void Transform::setLinearVelocity(vec3 newLinearVelocity, float framesPerSecond, float mix)
{
	useRelativeLinearMotionBlur = true;
	mix = glm::clamp(mix, 0.f, 1.f);
	newLinearVelocity /= framesPerSecond;
	linearMotion = glm::mix(newLinearVelocity, linearMotion, mix);
	updatePosition();
	markDirty();
}

void Transform::setAngularVelocity(quat newAngularVelocity, float framesPerSecond, float mix)
{
	useRelativeAngularMotionBlur = true;
	mix = glm::clamp(mix, 0.f, 1.f);
	newAngularVelocity[0] = newAngularVelocity[0] / framesPerSecond;
	newAngularVelocity[1] = newAngularVelocity[1] / framesPerSecond;
	newAngularVelocity[2] = newAngularVelocity[2] / framesPerSecond;
	angularMotion = glm::lerp(newAngularVelocity, angularMotion, mix);
	updateRotation();
	markDirty();
}

void Transform::setScalarVelocity(vec3 newScalarVelocity, float framesPerSecond, float mix)
{
	useRelativeScalarMotionBlur = true;
	mix = glm::clamp(mix, 0.f, 1.f);
	newScalarVelocity /= framesPerSecond;
	scalarMotion = glm::mix(newScalarVelocity, scalarMotion, mix);
	updateScale();
	markDirty();
}

void Transform::clearMotion()
{
	useRelativeLinearMotionBlur = true;
	useRelativeAngularMotionBlur = true;
	useRelativeScalarMotionBlur = true;
	scalarMotion = glm::vec3(0.f);
	angularMotion = glm::quat(1.f, 0.f, 0.f, 0.f);
	linearMotion = glm::vec3(0.f);
	updateMatrix();
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
	// localToParentTranslation = glm::translate(glm::mat4(1.0), position);
	// parentToLocalTranslation = glm::translate(glm::mat4(1.0), -position);
	// if (useRelativeMotionBlur) {
	// 	prevLocalToParentTranslation = glm::translate(glm::mat4(1.0), position + linearVelocity);
	// 	prevParentToLocalTranslation = glm::translate(glm::mat4(1.0), -position + linearVelocity);
	// } else {
	// 	prevLocalToParentTranslation = glm::translate(glm::mat4(1.0), prevPosition);
	// 	prevParentToLocalTranslation = glm::translate(glm::mat4(1.0), -prevPosition);
	// }
	updateMatrix();
	markDirty();
}

vec3 Transform::getScale(bool previous)
{
	if (previous) return prevScale;
	else return scale;
}

void Transform::setScale(vec3 newScale, bool previous)
{
	if (previous) useRelativeScalarMotionBlur = false;
	auto &s = (previous) ? prevScale : scale;
	s = newScale;
	updateScale();
	markDirty();
}

// void Transform::setScale(float newScale)
// {
// 	scale = vec3(newScale, newScale, newScale);
// 	updateScale();
// 	markDirty();
// }

void Transform::addScale(vec3 additionalScale, bool previous)
{
	if (previous) useRelativeScalarMotionBlur = false;
	setScale(getScale(previous) + additionalScale, previous);
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
	// localToParentScale = glm::scale(glm::mat4(1.0), scale);
	// parentToLocalScale = glm::scale(glm::mat4(1.0), glm::vec3(1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z));
	// if (useRelativeMotionBlur) {
	// 	prevLocalToParentScale = glm::scale(glm::mat4(1.0), scale + scalarVelocity);
	// 	prevParentToLocalScale = glm::scale(glm::mat4(1.0), glm::vec3(1.0 / (scale.x + scalarVelocity.x), 1.0 / (scale.y + scalarVelocity.y), 1.0 / (scale.z + scalarVelocity.z)));
	// } else {
	// 	prevLocalToParentScale = glm::scale(glm::mat4(1.0), prevScale);
	// 	prevParentToLocalScale = glm::scale(glm::mat4(1.0), glm::vec3(1.0 / prevScale.x, 1.0 / prevScale.y, 1.0 / prevScale.z));
	// }
	updateMatrix();
	markDirty();
}

void Transform::updateMatrix()
{
	localToParentMatrix = (localToParentTransform * getLocalToParentTranslationMatrix(false) * getLocalToParentRotationMatrix(false) * getLocalToParentScaleMatrix(false));
	parentToLocalMatrix = (getParentToLocalScaleMatrix(false) * getParentToLocalRotationMatrix(false) * getParentToLocalTranslationMatrix(false) * glm::inverse(localToParentTransform));

	prevLocalToParentMatrix = (prevLocalToParentTransform * getLocalToParentTranslationMatrix(true) * getLocalToParentRotationMatrix(true) * getLocalToParentScaleMatrix(true));
	prevParentToLocalMatrix = (getParentToLocalScaleMatrix(true) * getParentToLocalRotationMatrix(true) * getParentToLocalTranslationMatrix(true) * glm::inverse(prevLocalToParentTransform));

	// right = glm::vec3(localToParentMatrix[0]);
	// up = glm::vec3(localToParentMatrix[1]);
	// forward = glm::vec3(localToParentMatrix[2]);
	// position = glm::vec3(localToParentMatrix[3]);

	// prevRight = glm::vec3(prevLocalToParentMatrix[0]);
	// prevUp = glm::vec3(prevLocalToParentMatrix[1]);
	// prevForward = glm::vec3(prevLocalToParentMatrix[2]);
	// prevPosition = glm::vec3(prevLocalToParentMatrix[3]);

	updateChildren();
	markDirty();
}

glm::mat4 Transform::computeWorldToLocalMatrix(bool previous)
{
	glm::mat4 parentMatrix = glm::mat4(1.0);
	if (parent != -1) {
		parentMatrix = transforms[parent].computeWorldToLocalMatrix(previous);
		return getParentToLocalMatrix(previous) * parentMatrix;
	}
	else return getParentToLocalMatrix(previous);
}

// glm::mat4 Transform::computeNextWorldToLocalMatrix(bool previous)
// {
// 	glm::mat4 parentMatrix = glm::mat4(1.0);
// 	if (parent != -1) {
// 		parentMatrix = transforms[parent].computeNextWorldToLocalMatrix();
// 		return getNextParentToLocalMatrix() * parentMatrix;
// 	}
// 	else return getNextParentToLocalMatrix();
// }

void Transform::updateWorldMatrix()
{
	if (parent == -1) {
		worldToLocalMatrix = parentToLocalMatrix;
		localToWorldMatrix = localToParentMatrix;
		prevWorldToLocalMatrix = prevParentToLocalMatrix;
		prevLocalToWorldMatrix = prevLocalToParentMatrix;

		// worldScale = scale;
		// worldTranslation = position;
		// worldRotation = rotation;
		// worldSkew = glm::vec3(0.f, 0.f, 0.f);
		// worldPerspective = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f); // not sure what this should default to...

		// prevWorldScale = prevScale;
		// prevWorldTranslation = prevPosition;
		// prevWorldRotation = prevRotation;
		// prevWorldSkew = glm::vec3(0.f, 0.f, 0.f);
		// prevWorldPerspective = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f); // not sure what this should default to...
	} else {
		worldToLocalMatrix = computeWorldToLocalMatrix(/*previous=*/false);
		prevWorldToLocalMatrix = computeWorldToLocalMatrix(/*previous=*/true);
		localToWorldMatrix = glm::inverse(worldToLocalMatrix); 
		prevLocalToWorldMatrix = glm::inverse(prevWorldToLocalMatrix); 
		// glm::decompose(localToWorldMatrix, worldScale, worldRotation, worldTranslation, worldSkew, worldPerspective);
		// glm::decompose(prevLocalToWorldMatrix, prevWorldScale, prevWorldRotation, prevWorldTranslation, prevWorldSkew, prevWorldPerspective);
		// glm::decompose(nextLocalToWorldMatrix, worldScale, worldRotation, worldTranslation, worldSkew, worldPerspective);
	}
	markDirty();
}

glm::mat4 Transform::getParentToLocalMatrix(bool previous)
{
	if (previous) return prevParentToLocalMatrix;
	else return parentToLocalMatrix;
}

// glm::mat4 Transform::getNextParentToLocalMatrix()
// {
// 	return nextParentToLocalMatrix;
// }

glm::mat4 Transform::getLocalToParentMatrix(bool previous)
{
	if (previous) return prevLocalToParentMatrix;
	else return localToParentMatrix;
}

// glm::mat4 Transform::getNextLocalToParentMatrix()
// {
// 	return nextLocalToParentMatrix;
// }

glm::mat4 Transform::getLocalToParentTranslationMatrix(bool previous)
{
	if ((previous) && (useRelativeLinearMotionBlur)) return glm::translate(glm::mat4(1.0), position - linearMotion);
	else if (previous) return glm::translate(glm::mat4(1.0), prevPosition);
	else return glm::translate(glm::mat4(1.0), position);
}

glm::mat4 Transform::getLocalToParentScaleMatrix(bool previous)
{
	if ((previous) && (useRelativeScalarMotionBlur)) return glm::scale(glm::mat4(1.0), scale - scalarMotion);
	else if (previous) return glm::scale(glm::mat4(1.0), prevScale);
	else return glm::scale(glm::mat4(1.0), scale);
}

glm::mat4 Transform::getLocalToParentRotationMatrix(bool previous)
{
	if ((previous) && (useRelativeAngularMotionBlur)) return glm::toMat4(angularMotion * rotation);
	else if (previous) return glm::toMat4(prevRotation);
	else return glm::toMat4(rotation);
}

glm::mat4 Transform::getParentToLocalTranslationMatrix(bool previous)
{
	if ((previous) && (useRelativeLinearMotionBlur)) return glm::translate(glm::mat4(1.0), -(position - linearMotion));
	else if (previous) return glm::translate(glm::mat4(1.0), -prevPosition);
	else return glm::translate(glm::mat4(1.0), -position);
}

glm::mat4 Transform::getParentToLocalScaleMatrix(bool previous)
{
	if ((previous) && (useRelativeScalarMotionBlur)) return glm::scale(glm::mat4(1.0), glm::vec3(1.0 / (scale - scalarMotion).x, 1.0 / (scale - scalarMotion).y, 1.0 / (scale - scalarMotion).z));
	else if (previous) return glm::scale(glm::mat4(1.0), glm::vec3(1.0 / prevScale.x, 1.0 / prevScale.y, 1.0 / prevScale.z));
	else return glm::scale(glm::mat4(1.0), glm::vec3(1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z));
}

glm::mat4 Transform::getParentToLocalRotationMatrix(bool previous)
{
	if ((previous) && (useRelativeAngularMotionBlur)) return glm::toMat4(glm::inverse(angularMotion * rotation));
	else if (previous) return glm::toMat4(glm::inverse(prevRotation));
	else return glm::toMat4(glm::inverse(rotation));
}

Transform* Transform::getParent() {
	if ((this->parent < 0) || (this->parent >= MAX_TRANSFORMS)) return nullptr;
	return &transforms[this->parent];
}

std::vector<Transform*> Transform::getChildren() {
	std::vector<Transform*> children_list;
	for (auto &cid : this->children){
		// in theory I don't need to do this, but better safe than sorry.
		if ((cid < 0) || (cid >= MAX_TRANSFORMS)) continue;
		children_list.push_back(&transforms[cid]);
	}
	return children_list;
}

void Transform::setParent(Transform *parent) {
	if (!parent)
		throw std::runtime_error(std::string("Error: parent is empty"));

	if (!parent->isInitialized())
		throw std::runtime_error(std::string("Error: parent is uninitialized"));
	
	if (parent->getId() == this->getId())
		throw std::runtime_error(std::string("Error: a transform cannot be the parent of itself"));

	// check for circular relationships
	auto tmp = parent;
	while (tmp->getParent() != nullptr) {
		if (tmp->getParent()->getId() == this->getId()) {
			throw std::runtime_error(std::string("Error: circular dependency detected"));
		}
		tmp = tmp->getParent();
	}

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

	object->setParent(&transforms[getId()]);
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

glm::mat4 Transform::getWorldToLocalMatrix(bool previous) {
	if (previous) return prevWorldToLocalMatrix;
	else return worldToLocalMatrix;
}

glm::mat4 Transform::getLocalToWorldMatrix(bool previous) {
	if (previous) return prevLocalToWorldMatrix;
	else return localToWorldMatrix;
}

// glm::mat4 Transform::getNextWorldToLocalMatrix() {
// 	return nextWorldToLocalMatrix;
// }

// glm::mat4 Transform::getNextLocalToWorldMatrix() {
// 	return nextLocalToWorldMatrix;
// }

// glm::quat Transform::getWorldRotation(bool previous) {
// 	if (previous) return prevWorldRotation;
// 	else return worldRotation;
// }

// glm::vec3 Transform::getWorldTranslation(bool previous) {
// 	if (previous) return prevWorldTranslation;
// 	else return worldTranslation;
// }

// glm::vec3 Transform::getWorldScale(bool previous) {
// 	if (previous) return prevWorldScale;
// 	else return worldScale;
// }

// glm::mat4 Transform::getWorldToLocalRotationMatrix(bool previous)
// {
// 	return glm::toMat4(glm::inverse(worldRotation));
// }

// glm::mat4 Transform::getLocalToWorldRotationMatrix(bool previous)
// {
// 	if (previous) glm::toMat4(prevWorldRotation);
// 	return glm::toMat4(worldRotation);
// }

// glm::mat4 Transform::getWorldToLocalTranslationMatrix(bool previous)
// {
// 	glm::mat4 m(1.0);
// 	m = glm::translate(m, -1.0f * (previous) ? prevWorldTranslation : worldTranslation);
// 	return m;
// }

// glm::mat4 Transform::getLocalToWorldTranslationMatrix(bool previous)
// {
// 	glm::mat4 m(1.0);
// 	m = glm::translate(m, (previous) ? prevWorldTranslation : worldTranslation);
// 	return m;
// }

// glm::mat4 Transform::getWorldToLocalScaleMatrix(bool previous)
// {
// 	glm::mat4 m(1.0);
// 	m = glm::scale(m, 1.0f / ((previous) ? prevWorldScale : worldScale));
// 	return m;
// }

// glm::mat4 Transform::getLocalToWorldScaleMatrix(bool previous)
// {
// 	glm::mat4 m(1.0);
// 	m = glm::scale(m, (previous) ? prevWorldScale : worldScale);
// 	return m;
// }


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
