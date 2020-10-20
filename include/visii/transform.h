#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#define GLM_FORCE_RIGHT_HANDED

#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
// #include <glm/gtx/string_cast.hpp>
#include <glm/common.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_interpolation.hpp>
// #include <glm/gtx/matrix_decompose.hpp>
#include <map>
#include <mutex>

#include <visii/utilities/static_factory.h>
#include <visii/transform_struct.h>

using namespace std;

/**
 * The "Transform" component places an entity into the scene.
 * These transform components represent a scale, a rotation, and a translation, in that order.
 * These transform components also keep track of the previous frame scale, rotation, and translation, which 
 * can optionally be used for creating motion blur and for temporal effects like reprojection.
*/
class Transform : public StaticFactory
{
    friend class StaticFactory;
    friend class Entity;

  private:
    bool useRelativeLinearMotionBlur = true;
    bool useRelativeAngularMotionBlur = true;
    bool useRelativeScalarMotionBlur = true;

    /* Scene graph information */
    int32_t parent = -1;
	  std::set<int32_t> children;

    /* Local <=> Parent */
    glm::vec3 scale = glm::vec3(1.0);
    glm::vec3 position = glm::vec3(0.0);
    glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

    glm::vec3 prevScale = glm::vec3(1.0);
    glm::vec3 prevPosition = glm::vec3(0.0);
    glm::quat prevRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

    glm::vec3 linearMotion = glm::vec3(0.0);
    glm::quat angularMotion = glm::quat(1.f,0.f,0.f,0.f);
    glm::vec3 scalarMotion = glm::vec3(0.0);

    glm::mat4 localToParentTransform = glm::mat4(1);
    glm::mat4 localToParentMatrix = glm::mat4(1);
    glm::mat4 parentToLocalMatrix = glm::mat4(1);

    glm::mat4 prevLocalToParentTransform = glm::mat4(1);
    glm::mat4 prevLocalToParentMatrix = glm::mat4(1);
    glm::mat4 prevParentToLocalMatrix = glm::mat4(1);

    /* Local <=> World */
    glm::mat4 localToWorldMatrix = glm::mat4(1);
    glm::mat4 worldToLocalMatrix = glm::mat4(1);

    glm::mat4 prevLocalToWorldMatrix = glm::mat4(1);
    glm::mat4 prevWorldToLocalMatrix = glm::mat4(1);

  	static std::shared_ptr<std::recursive_mutex> editMutex;
    static bool factoryInitialized;

    static std::vector<Transform> transforms;
    static std::vector<TransformStruct> transformStructs;
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
    glm::mat4 computeWorldToLocalMatrix(bool previous);
    // glm::mat4 computePrevWorldToLocalMatrix(bool previous);


    static std::set<Transform*> dirtyTransforms;

  public:
    /**
      * Instantiates a null Transform. Used to mark a row in the table as null. 
      * Note: for internal use only. 
     */
    Transform();
/**
      * Instantiates a Transform with the given name and ID. Used to mark a row in the table as null. 
      * Note: for internal use only.
     */
    
    Transform(std::string name, uint32_t id);
    
    /**
     * Constructs a transform with the given name.
     * 
     * @param name A unique name for this transform.
     * @param scale The initial scale of the transform, applied first. 
     * @param rotation The initial scale of the transform, applied after scale.
     * @param position The initial position of the transform, applied after rotation.
     * @returns a reference to a transform component
    */
    static Transform* create(std::string name, 
      glm::vec3 scale = glm::vec3(1.0f), 
      glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
      glm::vec3 position = glm::vec3(0.f) 
    );

    /**
     * Constructs a transform with the given name, initializing with the given matrix.
     * 
     * @param name A unique name for this transform.
     * @param matrix The initial local to world transformation to be applied
     * @returns a reference to a transform component
    */
    static Transform* createFromMatrix(std::string name, 
      glm::mat4 transform = glm::mat4(1.0f)
    );

    /** 
     * @param name The name of the transform to get
     * @returns a transform who's name matches the given name 
     */
    static Transform* get(std::string name);

    /** @returns a pointer to the table of TransformStructs required for rendering*/
    static TransformStruct* getFrontStruct();

    /** @returns a pointer to the table of transform components */
    static Transform* getFront();

    /** @returns the number of allocated transforms */
	  static uint32_t getCount();

    /** @returns the name of this component */
	  std::string getName();

    /** @returns the unique integer ID for this component */
	  int32_t getId();

    // For internal use
    int32_t getAddress();

    /** @returns A map whose key is a transform name and whose value is the ID for that transform */
	  static std::map<std::string, uint32_t> getNameToIdMap();

    /** @param name The name of the transform to remove */
    static void remove(std::string name);

    /** Allocates the tables used to store all transform components */
    static void initializeFactory();

    /** @returns True if the tables used to store all transform components have been allocated, and False otherwise */
    static bool isFactoryInitialized();

    /** @returns True the current transform is a valid, initialized transform, and False if the transform was cleared or removed. */
	  bool isInitialized();

    /** Iterates through all transform components, computing transform metadata for rendering purposes. */
    static void updateComponents();

    /** Clears any existing transform components. */
    static void clearAll();

    /** @return True if any the transform has been modified since the previous frame, and False otherwise */
	  static bool areAnyDirty();

    /** @returns a list of transforms that have been modified since the previous frame */
    static std::set<Transform*> getDirtyTransforms();

    /** Tags the current component as being modified since the previous frame. */
	  void markDirty();

    /** For internal use. Returns the mutex used to lock transforms for processing by the renderer. */
    static std::shared_ptr<std::recursive_mutex> getEditMutex();

    /** @returns a json string representation of the current component */
    std::string toString();

    /** 
     *  Transforms direction from local to parent.
     *  This operation is not affected by scale or position of the transform.
     *  The returned vector has the same length as the input direction.
     *  
     *  @param direction The direction to apply the transform to.
     *  @param previous If true, uses the previous transform as the transform to apply.
     *  @return The transformed direction.
    */
    glm::vec3 transformDirection(glm::vec3 direction, bool previous = false);

    /** 
     * Transforms position from local to parent. Note, affected by scale.
     * The opposite conversion, from parent to local, can be done with Transform.inverse_transform_point
     * 
     * @param point The point to apply the transform to.
     * @param previous If true, uses the previous transform as the transform to apply.
     * @return The transformed point. 
    */
    glm::vec3 transformPoint(glm::vec3 point, bool previous = false);

    /** 
     * Transforms vector from local to parent.
     * This is not affected by position of the transform, but is affected by scale.
     * The returned vector may have a different length that the input vector.
     * 
     * @param vector The vector to apply the transform to.
     * @param previous If true, uses the previous transform as the transform to apply.
     * @return The transformed vector.
    */
    glm::vec3 transformVector(glm::vec3 vector, bool previous = false);

    /** 
     * Transforms a direction from parent space to local space.
     * The opposite of Transform.transform_direction.
     * This operation is unaffected by scale.
     * 
     * @param point The direction to apply the inverse transform to.
     * @param previous If true, uses the previous transform as the transform to apply.
     * @return The transformed direction.
    */
    glm::vec3 inverseTransformDirection(glm::vec3 direction, bool previous = false);

    /** 
     * Transforms position from parent space to local space.
     * Essentially the opposite of Transform.transform_point.
     * Note, affected by scale.
     * 
     * @param point The point to apply the inverse transform to.
     * @param previous If true, uses the previous transform as the transform to apply.
     * @return The transformed point.
    */
    glm::vec3 inverseTransformPoint(glm::vec3 point, bool previous = false);

    /** 
     * Transforms a vector from parent space to local space.
     * The opposite of Transform.transform_vector.
     * This operation is affected by scale.
     * 
     * @param point The vector to apply the inverse transform to.
     * @param previous If true, uses the previous transform as the transform to apply.
     * @return The transformed vector.
    */
    glm::vec3 inverseTransformVector(glm::vec3 vector, bool previous = false);

    /**
     *  Rotates the transform so the forward vector points at the target's current position.
     *  Then it rotates the transform to point its up direction vector in the direction hinted at 
     *  by the parentUp vector.
     * 
     * @param at The position to point the transform towards
     * @param up The unit direction pointing upwards
     * @param eye (optional) The position to place the object
     * @param previous If true, edits the previous translation and/or rotation.
    */
    void lookAt(glm::vec3 at, glm::vec3 up, glm::vec3 eye = glm::vec3(NAN), bool previous = false);

    // /**
    //  *  For motion blur. Rotates the prev transform so the forward vector points at the target's current position.
    //  *  Then it rotates the transform to point its up direction vector in the direction hinted at 
    //  *  by the parentUp vector.
    //  * 
    //  * @param at The position to point the transform towards
    //  * @param up The unit direction pointing upwards
    //  * @param eye (optional) The position to place the object
    // */
    // void prevLookAt(glm::vec3 at, glm::vec3 up, glm::vec3 eye = glm::vec3(NAN));

    // /**
    // Applies a rotation of eulerAngles.z degrees around the z axis, eulerAngles.x degrees around 
    // the x axis, and eulerAngles.y degrees around the y axis (in that order).
    // If relativeTo is not specified, rotation is relative to local space.
    // */
    // void rotate(glm::vec3 eularAngles, Space = Space::Local);

    // /** 
    //  * Rotates the transform about the provided axis, passing through the provided point in parent 
    //  * coordinates by the provided angle in degrees.
    //  * This modifies both the position and rotation of the transform.
    //  * 
    //  * @param point The pivot point in space to rotate around.
    //  * @param angle The angle (in radians) to rotate.
    //  * @param axis  The axis to rotate about.
    // */
    // void rotateAround(glm::vec3 point, float angle, glm::vec3 axis);

    /** 
     * Rotates the transform through the provided quaternion, passing through the provided point in parent 
     * coordinates.
     * This modifies both the position and rotation of the transform.
     * 
     * @param point The pivot point in space to rotate around.
     * @param quaternion The quaternion to use for rotation.
     * @param previous If true, edits the previous translation and rotation.
    */
    void rotateAround(glm::vec3 point, glm::quat quaternion, bool previous = false);

    /** 
     * Sets an optional additional transform, useful for representing normally unsupported transformations
     * like sheers and projections. 
     * 
     * @param transformation a 4 by 4 column major transformation matrix
     * @param decompose attempts to use the technique described in "Graphics Gems II: Decomposing a Matrix Into Simple Transformations" 
     * to represent the transform as a user controllable translation, rotation, and scale.
     * If a sheer is detected, or if the decomposition failed, this will fall back to a non-decomposed transformation, and user 
     * controllable translation, rotation, and scale will be set to identity values.
     * @param previous If true, edits the previous translation, rotation, and scale.
    */
    void setTransform(glm::mat4 transformation, bool decompose = true, bool previous = false);

    /** 
     * @param previous If true, returns the previous rotation.
     * @return A quaternion rotating the transform from local to parent 
     */
    glm::quat getRotation(bool previous = false);

    /** 
     * Sets the rotation of the transform from local to parent via a quaternion 
     * 
     * @param newRotation The new rotation quaternion to set the current transform quaternion to.
     * @param previous If true, edits the previous rotation.
    */
    void setRotation(glm::quat newRotation, bool previous = false);

    // /** 
    //  * Sets the rotation of the transform from local to parent using an axis 
    //  * in local space to rotate about, and an angle in radians to drive the rotation. 
    //  * 
    //  * @param angle The angle (in radians) to rotate.
    //  * @param axis  The axis to rotate about.
    // */
    // void setRotation(float angle, glm::vec3 axis);

    /** 
     * Adds a rotation to the existing transform rotation from local to parent 
     * via a quaternion. 
     * 
     * @param additionalRotation The rotation quaternion apply to the existing transform quaternion.
     * @param previous If true, edits the previous rotation.
    */
    void addRotation(glm::quat additionalRotation, bool previous = false);

    // /** 
    //  * Adds a rotation to the existing transform rotation from local to parent 
    //  * using an axis in local space to rotate about, and an angle in radians to 
    //  * drive the rotation
    //  *  
    //  * @param angle The angle (in radians) to rotate the current transform quaterion by.
    //  * @param axis  The axis to rotate about.
    // */
    // void addRotation(float angle, glm::vec3 axis);

    /** 
     * @param previous If true, returns the previous parent-space position.
     * @returns a position vector describing where this transform will be translated to in its' parent's space. 
     */
    glm::vec3 getPosition(bool previous = false);

    /** 
     * @param previous If true, returns the previous parent-space right vector.
     * @returns a vector pointing right relative to the current transform placed in its' parent's space. 
     */
    glm::vec3 getRight(bool previous = false);

    /** 
     * @param previous If true, returns the previous parent-space up vector.
     * @returns a vector pointing up relative to the current transform placed in its' parent's space. 
     */
    glm::vec3 getUp(bool previous = false);

    /** 
     * @param previous If true, returns the previous parent-space forward vector.
     * @returns a vector pointing forward relative to the current transform placed in its' parent's space. 
     */
    glm::vec3 getForward(bool previous = false);

    /** 
     * @param previous If true, returns the previous world-space position.
     * @returns a position vector describing where this transform will be translated to in world-space. 
     */
    glm::vec3 getWorldPosition(bool previous = false);

    /** 
     * @param previous If true, returns the previous world-space right vector.
     * @returns a vector pointing right relative to the current transform placed in world-space. 
     */
    glm::vec3 getWorldRight(bool previous = false);

    /** 
     * @param previous If true, returns the previous world-space up vector.
     * @returns a vector pointing up relative to the current transform placed in world-space. 
     */
    glm::vec3 getWorldUp(bool previous = false);

    /** 
     * @param previous If true, returns the previous world-space forward vector.
     * @returns a vector pointing forward relative to the current transform placed in world-space. 
     */
    glm::vec3 getWorldForward(bool previous = false);

    /** 
     * Sets the position vector describing where this transform should be translated to when placed in its 
     * parent space. 
     * 
     * @param newPosition The new position to set the current transform position to.
     * @param previous If true, edits the previous position.
    */
    void setPosition(glm::vec3 newPosition, bool previous = false);

    /** 
     * Adds to the current the position vector describing where this transform should be translated to 
     * when placed in its parent space. 
     * 
     * @param additionalPosition The position (interpreted as a vector) to add onto the current transform position.
     * @param previous If true, edits the previous position.
    */
    void addPosition(glm::vec3 additionalPosition, bool previous = false);

    // /**
    //  * Sets the position vector describing where this transform should be translated to when placed in its 
    //  * parent space. 
    //  * 
    //  * @param x The x component of the new position.
    //  * @param y The y component of the new position.
    //  * @param z The z component of the new position.
    // */
    // void setPosition(float x, float y, float z);

    // /**
    //  * Adds to the current the position vector describing where this transform should be translated to 
    //  * when placed in its parent space. 
    //  * 
    //  * @param dx The change in x to add onto the current transform position.
    //  * @param dy The change in y to add onto the current transform position.
    //  * @param dz The change in z to add onto the current transform position.
    // */
    // void addPosition(float dx, float dy, float dz);

    /** 
     * @param previous If true, returns the previous scale.
     * @returns the scale of this transform from local to parent space along its right, up, and forward 
     * directions respectively 
     */
    glm::vec3 getScale(bool previous = false);

    /** 
     * Sets the scale of this transform from local to parent space along its right, up, and forward 
     * directions respectively. 
     * 
     * @param newScale The new scale to set the current transform scale to.
     * @param previous If true, edits the previous scale.
    */
    void setScale(glm::vec3 newScale, bool previous = false);

    // /** 
    //  * Sets the scale of this transform from local to parent space along its right, up, and forward 
    //  * directions simultaneously.
    //  * 
    //  * @param newScale The new uniform scale to set the current transform scale to.
    // */
    // void setScale(float newScale);

    /** 
     * Adds to the current the scale of this transform from local to parent space along its right, up, 
     * and forward directions respectively 
     * 
     * @param additionalScale The scale to add onto the current transform scale.
     * @param previous If true, edits the previous scale.
    */
    void addScale(glm::vec3 additionalScale, bool previous = false);

    // /** 
    //  * Sets the scale of this transform from local to parent space along its right, up, and forward 
    //  * directions respectively.
    //  * 
    //  * @param x The x component of the new scale.
    //  * @param y The y component of the new scale.
    //  * @param z The z component of the new scale.
    // */
    // void setScale(float x, float y, float z);

    // /** 
    //  * Adds to the current the scale of this transform from local to parent space along its right, up, 
    //  * and forward directions respectively 
    //  * 
    //  * @param dx The change in x to add onto the current transform scale.
    //  * @param dy The change in y to add onto the current transform scale.
    //  * @param dz The change in z to add onto the current transform scale.
    // */
    // void addScale(float dx, float dy, float dz);

    // /** 
    //  * Adds to the scale of this transform from local to parent space along its right, up, and forward 
    //  * directions simultaneously 
    //  * 
    //  * @param ds The change in scale to uniformly add onto all components of the current transform scale.
    // */
    // void addScale(float ds);

    /** 
     * Sets the linear velocity vector describing how fast this transform is translating within its 
     * parent space. Causes motion blur.
     * 
     * @param velocity The new linear velocity to set the current transform linear velocity to, in meters per second.
     * @param frames_per_second Used to convert meters per second into meters per frame. Useful for animations.
    */
    void setLinearVelocity(glm::vec3 velocity, float frames_per_second = 1.0f, float mix = 0.0f);

    /** 
     * Sets the angular velocity vector describing how fast this transform is rotating within its 
     * parent space. Causes motion blur.
     * 
     * @param velocity The new angular velocity to set the current transform angular velocity to, in radians per second.
     * @param frames_per_second Used to convert radians per second into scale per frame. Useful for animations.
    */
    void setAngularVelocity(glm::quat velocity, float frames_per_second = 1.0f, float mix = 0.0f);

    /** 
     * Sets the scalar velocity vector describing how fast this transform is scaling within its 
     * parent space. Causes motion blur.
     * 
     * @param velocity The new scalar velocity to set the current transform scalar velocity to, in additional scale per second
     * @param frames_per_second Used to convert additional scale per second into additional scale per frame. Useful for animations.
    */
    void setScalarVelocity(glm::vec3 velocity, float frames_per_second = 1.0f, float mix = 0.0f);

    /**
     * Resets any "previous" transform data, effectively clearing any current motion blur.
     */
    void clearMotion();

    /** 
     * @param previous If true, returns the previous parent-to-local matrix.
     * @returns the final matrix transforming this object from it's parent coordinate space to it's 
     * local coordinate space 
     */
    glm::mat4 getParentToLocalMatrix(bool previous = false);

    // /** 
    //  * @returns the final matrix transforming this object from it's parent coordinate space to it's 
    //  * local coordinate space, accounting for linear and angular velocities.
    //  */
    // glm::mat4 getPrevParentToLocalMatrix(bool previous = false);

    /** 
     * @param previous If true, returns the previous local-to-parent matrix.
     * @returns the final matrix transforming this object from it's local coordinate space to it's 
     * parents coordinate space 
    */
    glm::mat4 getLocalToParentMatrix(bool previous = false);

    /** 
     * @param previous If true, returns the previous local-to-parent translation matrix.
     * @returns the final matrix translating this object from it's local coordinate space to it's 
     * parent coordinate space 
     */
    glm::mat4 getLocalToParentTranslationMatrix(bool previous = false);

    /** 
     * @param previous If true, returns the previous local-to-parent scale matrix.
     * @returns the final matrix translating this object from it's local coordinate space to it's 
     * parent coordinate space 
     */
    glm::mat4 getLocalToParentScaleMatrix(bool previous = false);

    /** 
     * @param previous If true, returns the previous local-to-parent rotation matrix.
     * @returns the final matrix rotating this object in it's local coordinate space to it's 
     * parent coordinate space 
     */
    glm::mat4 getLocalToParentRotationMatrix(bool previous = false);

    /** 
     * @param previous If true, returns the previous parent-to-local translation matrix.
     * @returns the final matrix translating this object from it's parent coordinate space to it's 
     * local coordinate space 
     */
    glm::mat4 getParentToLocalTranslationMatrix(bool previous = false);

    /** 
     * @param previous If true, returns the previous parent-to-local scale matrix.
     * @returns the final matrix scaling this object from it's parent coordinate space to it's 
     * local coordinate space 
     */
    glm::mat4 getParentToLocalScaleMatrix(bool previous = false);

    /** 
     * @param previous If true, returns the previous parent-to-local rotation matrix.
     * @returns the final matrix rotating this object from it's parent coordinate space to it's 
     * local coordinate space 
    * */
    glm::mat4 getParentToLocalRotationMatrix(bool previous = false);

    /** 
     * Set the parent of this transform, whose transformation will be applied after the current
     * transform. 
     * 
     * @param parent The transform component to constrain the current transform to. Any existing parent constraint is replaced.
    */
    void setParent(Transform * parent);
    
    Transform* getParent();
    std::vector<Transform*> getChildren();

    /** Removes the parent-child relationship affecting this node. */
    void clearParent();

    /** 
     * Add a child to this transform, whose transformation will be applied before the current
     * transform. 
     * 
     * @param child The child transform component to constrain to the current transform. Any existing parent constraint is replaced.
    */
	  void addChild(Transform*  child);

    /** 
     * Removes a child transform previously added to the current transform. 
     * 
     * @param child The constrained child transform component to un-constrain from the current transform. Any existing parent constraint is replaced.
    */
	  void removeChild(Transform* child);

    /** 
     * @param previous If true, returns the previous world-to-local matrix.
     * @returns a matrix transforming this component from world space to its local space, taking all 
     * parent transforms into account. 
     */
    glm::mat4 getWorldToLocalMatrix(bool previous = false);

    /** 
     * @param previous If true, returns the previous local-to-world matrix.
     * @returns a matrix transforming this component from its local space to world space, taking all 
     * parent transforms into account. 
     */
	  glm::mat4 getLocalToWorldMatrix(bool previous = false);

    // /** 
    //  * @param previous If true, returns the previous world space scale.
    //  * @returns a (possibly approximate) scale scaling the current transform from 
    //    * local space to world space, taking all parent transforms into account 
    //    */
    // glm::vec3 getWorldScale(bool previous = false);

    // /** 
    //  * @param previous If true, returns the previous world space rotation.
    //  * @returns a (possibly approximate) rotation rotating the current transform from 
    //  * local space to world space, taking all parent transforms into account 
    //  */
	  // glm::quat getWorldRotation(bool previous = false);

    // /** 
    //  * @param previous If true, returns the previous world space translation.
    //  * @returns a (possibly approximate) translation moving the current transform from 
    //  * local space to world space, taking all parent transforms into account 
    //  */
    // glm::vec3 getWorldTranslation(bool previous = false);

    // /** 
    //  * @param previous If true, returns the previous world-to-local rotation matrix.
    //  * @returns a (possibly approximate) rotation matrix rotating the current transform from 
    //  * local space to world space, taking all parent transforms into account 
    //  */
    // glm::mat4 getWorldToLocalRotationMatrix(bool previous = false);

    // /** 
    //  * @param previous If true, returns the previous local-to-world rotation matrix.
    //  * @returns a (possibly approximate) rotation matrix rotating the current transform from 
    //  * world space to local space, taking all parent transforms into account 
    //  */
    // glm::mat4 getLocalToWorldRotationMatrix(bool previous = false);

    // /** 
    //  * @param previous If true, returns the previous world-to-local translation matrix.
    //  * @returns a (possibly approximate) translation matrix translating the current transform from 
    //  * local space to world space, taking all parent transforms into account 
    //  */
    // glm::mat4 getWorldToLocalTranslationMatrix(bool previous = false);

    // /** 
    //  * @param previous If true, returns the previous local-to-world translation matrix.
    //  * @returns a (possibly approximate) translation matrix rotating the current transform from 
    //  * world space to local space 
    //  */
    // glm::mat4 getLocalToWorldTranslationMatrix(bool previous = false);

    // /** 
    //  * @param previous If true, returns the previous world-to-local scale matrix.
    //  * @returns a (possibly approximate) scale matrix scaling the current transform from 
    //  * local space to world space, taking all parent transforms into account 
    //  */
    // glm::mat4 getWorldToLocalScaleMatrix(bool previous = false);

    // /** 
    //  * @param previous If true, returns the previous local-to-world scale matrix.
    //  * @returns a (possibly approximate) scale matrix scaling the current transform from 
    //  * world space to local space, taking all parent transforms into account 
    //  */
    // glm::mat4 getLocalToWorldScaleMatrix(bool previous = false);

    /** @returns a struct with only essential data */
    TransformStruct &getStruct();
};
