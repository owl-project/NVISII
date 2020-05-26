#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#define GLM_FORCE_RIGHT_HANDED

#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/common.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_interpolation.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <map>
#include <mutex>

#include <visii/utilities/static_factory.h>
#include <visii/transform_struct.h>

using namespace glm;
using namespace std;

/**
 * The "Transform" component places an entity into the scene.
 * These transform components represent a scale, a rotation, and a translation, in that order.
*/
class Transform : public StaticFactory
{
    friend class StaticFactory;
    friend class Entity;
  private:
    /* Scene graph information */
    int32_t parent = -1;
	  std::set<int32_t> children;

    /* Local <=> Parent */
    vec3 scale = vec3(1.0);
    vec3 position = vec3(0.0);
    quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

    vec3 right = vec3(1.0, 0.0, 0.0);
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 forward = vec3(0.0, 0.0, 1.0);

    mat4 localToParentTransform = mat4(1);
    mat4 localToParentRotation = mat4(1);
    mat4 localToParentTranslation = mat4(1);
    mat4 localToParentScale = mat4(1);

    mat4 parentToLocalTransform = mat4(1);
    mat4 parentToLocalRotation = mat4(1);
    mat4 parentToLocalTranslation = mat4(1);
    mat4 parentToLocalScale = mat4(1);

    mat4 localToParentMatrix = mat4(1);
    mat4 parentToLocalMatrix = mat4(1);

    /* Local <=> World */
    mat4 localToWorldMatrix = mat4(1);
    mat4 worldToLocalMatrix = mat4(1);

    // from local to world decomposition. 
    // May only approximate the localToWorldMatrix
    glm::vec3 worldScale;
    glm::quat worldRotation;
    glm::vec3 worldTranslation;
    glm::vec3 worldSkew;
    glm::vec4 worldPerspective;
    
    // float interpolation = 1.0;

    /* TODO */
	static std::shared_ptr<std::mutex> creationMutex;
    static bool factoryInitialized;

    static Transform transforms[MAX_TRANSFORMS];
    static TransformStruct transformStructs[MAX_TRANSFORMS];
    static std::map<std::string, uint32_t> lookupTable;
    
    /* Updates cached rotation values */
    void updateRotation();

    /* Updates cached position values */
    void updatePosition();

    /* Updates cached scale values */
    void updateScale();

    /* Updates cached final local to parent matrix values */
    void updateMatrix();

    /* Updates cached final local to world matrix values */
    void updateWorldMatrix();

    /* updates all childrens cached final local to world matrix values */
    void updateChildren();

    /* updates the struct for this transform which can be uploaded to the GPU. */
    void updateStruct();
    
    /* traverses from the current transform up through its ancestors, 
    computing a final world to local matrix */
    glm::mat4 computeWorldToLocalMatrix();

    Transform();
    Transform(std::string name, uint32_t id);

    /* Indicates that one of the components has been edited */
    static bool anyDirty;

    /* Indicates this component has been edited */
    bool dirty = true;

  public:
    /** Constructs a transform with the given name.
     * \returns a reference to a transform component
     * \param name A unique name for this transform.
    */
    static Transform* create(std::string name);

    /** Gets a transform by name 
     * \returns a transform who's primary name key matches \p name 
     * \param name A unique name used to lookup this transform. */
    static Transform* get(std::string name);

    /** Gets a transform by id 
     * \returns a transform who's primary id key matches \p id 
     * \param id A unique id used to lookup this transform. */
    static Transform* get(uint32_t id);

    /** \returns a pointer to the table of TransformStructs required for rendering*/
    static TransformStruct* getFrontStruct();

    /** \returns a pointer to the table of transform components */
    static Transform* getFront();

    /** \returns the number of allocated transforms */
	  static uint32_t getCount();

    /** Deletes the transform who's primary name key matches \p name 
     * \param name A unique name used to lookup the transform for deletion.*/
    static void remove(std::string name);

    /** Deletes the transform who's primary id key matches \p id 
     * \param id A unique id used to lookup the transform for deletion.*/
    static void remove(uint32_t id);

    /** Allocates the tables used to store all transform components */
    static void initializeFactory();

    /** \return True if the tables used to store all transform components have been allocated, and False otherwise */
    static bool isFactoryInitialized();

    /** Iterates through all transform components, computing transform metadata for rendering purposes. */
    static void updateComponents();

    /** Frees any tables used to store transform components */
    static void cleanUp();

    /** \return True if the Transform has been modified since the previous frame, and False otherwise */
	  bool isDirty() { return dirty; }

	  static bool areAnyDirty();


    /** \return True if the Transform has not been modified since the previous frame, and False otherwise */
	  bool isClean() { return !dirty; }

    /** Tags the current component as being modified since the previous frame. */
	  void markDirty();

    /** Tags the current component as being unmodified since the previous frame. */
	  void markClean() { dirty = false; }

    /** \return a json string representation of the current component */
    std::string toString();

    /** Transforms direction from local to parent.
     *  This operation is not affected by scale or position of the transform.
     *  The returned vector has the same length as the input direction.
     *  \param direction The direction to apply the transform to.
     *  \return The transformed direction.
    */
    vec3 transformDirection(vec3 direction);

    /** Transforms position from local to parent. Note, affected by scale.
      * The opposite conversion, from parent to local, can be done with Transform.inverse_transform_point
      * \param point The point to apply the transform to.
      * \return The transformed point. 
    */
    vec3 transformPoint(vec3 point);

    /** Transforms vector from local to parent.
      * This is not affected by position of the transform, but is affected by scale.
      * The returned vector may have a different length that the input vector.
      * \param vector The vector to apply the transform to.
      * \return The transformed vector.
    */
    vec3 transformVector(vec3 vector);

    /** Transforms a direction from parent space to local space.
      * The opposite of Transform.transform_direction.
      * This operation is unaffected by scale.
      * \param point The direction to apply the inverse transform to.
      * \return The transformed direction.
    */
    vec3 inverseTransformDirection(vec3 direction);

    /** Transforms position from parent space to local space.
      * Essentially the opposite of Transform.transform_point.
      * Note, affected by scale.
      * \param point The point to apply the inverse transform to.
      * \return The transformed point.
    */
    vec3 inverseTransformPoint(vec3 point);

    /** Transforms a vector from parent space to local space.
      * The opposite of Transform.transform_vector.
      * This operation is affected by scale.
      * \param point The vector to apply the inverse transform to.
      * \return The transformed vector.
    */
    vec3 inverseTransformVector(vec3 vector);

    /**
    Rotates the transform so the forward vector points at the target's current position.
    Then it rotates the transform to point its up direction vector in the direction hinted at 
    by the parentUp vector.
    */
    // void look_at(Transform target, vec3 parentUp);
    // void look_at(vec3 point);

    /**
    Applies a rotation of eulerAngles.z degrees around the z axis, eulerAngles.x degrees around 
    the x axis, and eulerAngles.y degrees around the y axis (in that order).
    If relativeTo is not specified, rotation is relative to local space.
    */
    // void rotate(vec3 eularAngles, Space = Space::Local);

    /** Rotates the transform about the provided axis, passing through the provided point in parent 
      * coordinates by the provided angle in degrees.
      * This modifies both the position and rotation of the transform.
      * \param point The pivot point in space to rotate around.
      * \param angle The angle (in radians) to rotate.
      * \param axis  The axis to rotate about.
    */
    void rotateAround(vec3 point, float angle, vec3 axis);

    /** Rotates the transform through the provided quaternion, passing through the provided point in parent 
      * coordinates.
      * This modifies both the position and rotation of the transform.
      * \param point The pivot point in space to rotate around.
      * \param rot   The quaternion to use for rotation.
    */
    void rotateAround(vec3 point, glm::quat rot);

    /** Sets an optional additional transform, useful for representing normally unsupported transformations
      * like sheers and projections. 
      * \param transformation  a 4 by 4 column major transformation matrix
      * \param decompose       attempts to use singular value decomposition to decompose the provided transform into a translation, rotation, and scale 
    */
    void setTransform(glm::mat4 transformation, bool decompose = true);

    /** \return A quaternion rotating the transform from local to parent */
    quat getRotation();

    /** Sets the rotation of the transform from local to parent via a quaternion 
      * \param newRotation The new rotation quaternion to set the current transform quaternion to.
    */
    void setRotation(quat newRotation);

    /** Sets the rotation of the transform from local to parent using an axis 
      * in local space to rotate about, and an angle in radians to drive the rotation. 
      * \param angle The angle (in radians) to rotate.
      * \param axis  The axis to rotate about.
    */
    void setRotation(float angle, vec3 axis);

    /** Adds a rotation to the existing transform rotation from local to parent 
      * via a quaternion. 
      * \param additionalRotation The rotation quaternion apply to the existing transform quaternion.
    */
    void addRotation(quat additionalRotation);

    /** Adds a rotation to the existing transform rotation from local to parent 
      * using an axis in local space to rotate about, and an angle in radians to 
      * drive the rotation 
      * \param angle The angle (in radians) to rotate the current transform quaterion by.
      * \param axis  The axis to rotate about.
    */
    void addRotation(float angle, vec3 axis);

    /** \returns a position vector describing where this transform will be translated to in its parent space. */
    vec3 getPosition();

    /** \returns a vector pointing right relative to the current transform placed in its parent's space. */
    vec3 getRight();

    /** \returns a vector pointing up relative to the current transform placed in its parent's space. */
    vec3 getUp();

    /** \returns a vector pointing forward relative to the current transform placed in its parent's space. */
    vec3 getForward();

    /** Sets the position vector describing where this transform should be translated to when placed in its 
      * parent space. 
      * \param newPosition The new position to set the current transform position to.
    */
    void setPosition(vec3 newPosition);

    /** Adds to the current the position vector describing where this transform should be translated to 
      * when placed in its parent space. 
      * \param additionalPosition The position (interpreted as a vector) to add onto the current transform position.
    */
    void addPosition(vec3 additionalPosition);

    /** Sets the position vector describing where this transform should be translated to when placed in its 
      * parent space. 
      * \param x The x component of the new position.
      * \param y The y component of the new position.
      * \param z The z component of the new position.
    */
    void setPosition(float x, float y, float z);

    /** Adds to the current the position vector describing where this transform should be translated to 
      * when placed in its parent space. 
      * \param dx The change in x to add onto the current transform position.
      * \param dy The change in y to add onto the current transform position.
      * \param dz The change in z to add onto the current transform position.
    */
    void addPosition(float dx, float dy, float dz);

    /** \returns the scale of this transform from local to parent space along its right, up, and forward 
      * directions respectively */
    vec3 getScale();

    /** Sets the scale of this transform from local to parent space along its right, up, and forward 
      * directions respectively. 
      * \param newScale The new scale to set the current transform scale to.
    */
    void setScale(vec3 newScale);

    /** Sets the scale of this transform from local to parent space along its right, up, and forward 
      * directions simultaneously.
      * \param newScale The new uniform scale to set the current transform scale to.
    */
    void setScale(float newScale);

    /** Adds to the current the scale of this transform from local to parent space along its right, up, 
      * and forward directions respectively 
      * \param additionalScale The scale to add onto the current transform scale.
    */
    void addScale(vec3 additionalScale);

    /** Sets the scale of this transform from local to parent space along its right, up, and forward 
      * directions respectively.
      * \param x The x component of the new scale.
      * \param y The y component of the new scale.
      * \param z The z component of the new scale.
    */
    void setScale(float x, float y, float z);

    /** Adds to the current the scale of this transform from local to parent space along its right, up, 
      * and forward directions respectively 
      * \param dx The change in x to add onto the current transform scale.
      * \param dy The change in y to add onto the current transform scale.
      * \param dz The change in z to add onto the current transform scale.
    */
    void addScale(float dx, float dy, float dz);

    /** Adds to the scale of this transform from local to parent space along its right, up, and forward 
      * directions simultaneously 
      * \param ds The change in scale to uniformly add onto all components of the current transform scale.
    */
    void addScale(float ds);

    /** \returns the final matrix transforming this object from it's parent coordinate space to it's 
      * local coordinate space */
    glm::mat4 getParentToLocalMatrix();

    /** \returns the final matrix transforming this object from it's local coordinate space to it's 
      * parents coordinate space */
    glm::mat4 getLocalToParentMatrix();

    /** \returns the final matrix translating this object from it's local coordinate space to it's 
      * parent coordinate space */
    glm::mat4 getLocalToParentTranslationMatrix();

    /** \returns the final matrix translating this object from it's local coordinate space to it's 
      * parent coordinate space */
    glm::mat4 getLocalToParentScaleMatrix();

    /** \returns the final matrix rotating this object in it's local coordinate space to it's 
      * parent coordinate space */
    glm::mat4 getLocalToParentRotationMatrix();

    /** \returns the final matrix translating this object from it's parent coordinate space to it's 
      * local coordinate space */
    glm::mat4 getParentToLocalTranslationMatrix();

    /** \returns the final matrix scaling this object from it's parent coordinate space to it's 
      * local coordinate space */
    glm::mat4 getParentToLocalScaleMatrix();

    /** \returns the final matrix rotating this object from it's parent coordinate space to it's 
      * local coordinate space */
    glm::mat4 getParentToLocalRotationMatrix();

    /** Set the parent of this transform, whose transformation will be applied after the current
      * transform. 
      * \param parent The primary id key of the transform component to constrain the current transform to. Any existing parent constraint is replaced.
    */
    void setParent(uint32_t parent);

    /** Removes the parent-child relationship affecting this node. */
    void clearParent();

    /** Add a child to this transform, whose transformation will be applied before the current
      * transform. 
      * \param child The primary id key of the child transform component to constrain to the current transform. Any existing parent constraint is replaced.
    */
	void addChild(uint32_t child);

    /** Removes a child transform previously added to the current transform. 
      * \param child The primary id key of the constrained child transform component to un-constrain from the current transform. Any existing parent constraint is replaced.
    */
	void removeChild(uint32_t child);

    /** \returns a matrix transforming this component from world space to its local space, taking all 
      * parent transforms into account. */
    glm::mat4 getWorldToLocalMatrix();

    /** \returns a matrix transforming this component from its local space to world space, taking all 
      * parent transforms into account. */
	glm::mat4 getLocalToWorldMatrix();

    /** \returns a (possibly approximate) rotation rotating the current transform from 
      * local space to world space, taking all parent transforms into account */
	glm::quat getWorldRotation();

    /** \returns a (possibly approximate) translation moving the current transform from 
      * local space to world space, taking all parent transforms into account */
    glm::vec3 getWorldTranslation();

    /** \returns a (possibly approximate) rotation matrix rotating the current transform from 
      * local space to world space, taking all parent transforms into account */
    glm::mat4 getWorldToLocalRotationMatrix();

    /** \returns a (possibly approximate) rotation matrix rotating the current transform from 
      * world space to local space, taking all parent transforms into account */
    glm::mat4 getLocalToWorldRotationMatrix();

    /** \returns a (possibly approximate) translation matrix translating the current transform from 
      * local space to world space, taking all parent transforms into account */
    glm::mat4 getWorldToLocalTranslationMatrix();

    /** \returns a (possibly approximate) translation matrix rotating the current transform from 
      * world space to local space */
    glm::mat4 getLocalToWorldTranslationMatrix();

    /** \returns a (possibly approximate) scale matrix scaling the current transform from 
      * local space to world space, taking all parent transforms into account */
    glm::mat4 getWorldToLocalScaleMatrix();

    /** \returns a (possibly approximate) scale matrix scaling the current transform from 
      * world space to local space, taking all parent transforms into account */
    glm::mat4 getLocalToWorldScaleMatrix();

    /** \returns a struct with only essential data */
    TransformStruct &getStruct();
};
