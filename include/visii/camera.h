#pragma once

#include <mutex>
#include <condition_variable>

#include <visii/utilities/static_factory.h>
#include <visii/camera_struct.h>

/**
 * The "Camera" component describes the perspective of an entity.
 * It lens properties, like depth of field, focal length, field of view, and so on.
 * It also describes the target render resolution.
 * By connecting a camera component to an entity with a transform, that entity
 * can be used to render the scene from a position in space.
*/
class Camera : public StaticFactory
{
	friend class StaticFactory;
    friend class Entity;
private:
  	/** Prevents multiple components from simultaneously being added and/or removed from the component list */
		static std::shared_ptr<std::recursive_mutex> editMutex;

		/** Marks that the StaticFactory has allocated the table of components */
		static bool factoryInitialized;

		/** The table of Camera components */
		static std::vector<Camera> cameras;

		/** The table of Camera structs */
		static std::vector<CameraStruct> cameraStructs;

		/* A lookup table of name to camera id */
		static std::map<std::string, uint32_t> lookupTable;

    /* Indicates that one of the components has been edited */
    static bool anyDirty;

    /* Indicates this component has been edited */
    bool dirty = true;
	
  public:
  	/**
		 * Instantiates a null Camera. Used to mark a row in the table as null. 
     * Note: for internal use only. 
		*/
		Camera();

		/**
		 * Instantiates a Camera with the given name and ID. Used to mark a row in the table as null. 
		 * Note: for internal use only.
		 */
		Camera(std::string name, uint32_t id);

		/** 
		 * Constructs a camera component.
		 * @param name A unique name for this camera.
		 * @param field_of_view Specifies the field of view angle in the y direction. Expressed in radians.
		 * @param aspect Specifies the aspect ratio that determines the field of view in the x direction. The aspect ratio is a ratio of x (width) to y (height)
		 * @returns a reference to a camera component
		*/
		static Camera *create(std::string name, float field_of_view = 0.785398, float aspect = 1.0);

		/** Deprecated in favor of either create or create_from_fov */
		static Camera *createPerspectiveFromFOV(std::string name, float field_of_view, float aspect);

		/** 
		 * Constructs a camera component from a field of view.
		 * The field of view controls the amount of zoom, i.e. the amount of the scene 
		 * which is visible all at once. A smaller field of view results in a longer focal length (more zoom),
		 * while larger field of view allow you to see more of the scene at once (shorter focal length, less zoom)
		 * 
		 * @param name A unique name for this camera.
		 * @param field_of_view Specifies the field of view angle in the y direction. Expressed in radians.
		 * @param aspect Specifies the aspect ratio that determines the field of view in the x direction. The aspect ratio is a ratio of x (width) to y (height)
		 * @returns a reference to a camera component
		*/
		static Camera *createFromFOV(std::string name, float field_of_view, float aspect);

		/** Deprecated in favor of create_from_focal_length */
		static Camera *createPerspectiveFromFocalLength(std::string name, float focal_length, float sensor_width, float sensor_height);

		/** 
		 * Constructs a camera component from a focal length.
		 * The focal length controls the amount of zoom, i.e. the amount of the scene 
		 * which is visible all at once. Longer focal lengths result in a smaller field of view (more zoom),
		 * while short focal lengths allow you to see more of the scene at once (larger FOV, less zoom)
		 * 
		 * @param name A unique name for this camera.
		 * @param focal_length Specifies the focal length of the camera lens (in millimeters).
		 * @param sensor_width Specifies the width of the camera sensor (in millimeters). 
		 * @param sensor_height Specifies the height of the camera sensor (in millimeters). 
		 * @returns a reference to a camera component
		*/
		static Camera *createFromFocalLength(std::string name, float focal_length, float sensor_width, float sensor_height);

		/** 
		 * @param name The name of the camera to get
			 * @returns a Camera who's name matches the given name 
		*/
		static Camera *get(std::string name);

		/** @returns a pointer to the table of CameraStructs */
		static CameraStruct *getFrontStruct();

		/** @returns a pointer to the list of Camera components. */
		static Camera *getFront();

		/** @returns the number of allocated cameras. */
		static uint32_t getCount();

		/** @returns the name of this component */
		std::string getName();

		/** @returns A map whose key is a camera name and whose value is the ID for that camera */
		static std::map<std::string, uint32_t> getNameToIdMap();

		/** @param name The name of the camera to remove */
		static void remove(std::string name);

		/** Allocates the tables used to store all Camera components */
		static void initializeFactory(uint32_t max_components);

		/** @returns True if the tables used to store all Camera components have been allocated, and False otherwise */
		static bool isFactoryInitialized();

		/** @returns True the current camera is a valid, initialized camera, and False if the camera was cleared or removed. */
		bool isInitialized();

		/** Iterates through all components, updating any component struct fields and marking components as clean. */
			static void updateComponents();

		/** Clears any existing camera components. */
		static void clearAll();

		/** @returns a string representation of the current component */
			std::string toString();

		/** Indicates whether or not any cameras are "out of date" and need to be updated through the "update components" function*/
		static bool areAnyDirty();

    /** @returns True if the camera has been modified since the previous frame, and False otherwise */
    bool isDirty() { return dirty; }

    /** @returns True if the camera has not been modified since the previous frame, and False otherwise */
    bool isClean() { return !dirty; }

    /** Tags the current component as being modified since the previous frame. */
    void markDirty();

    /** Tags the current component as being unmodified since the previous frame. */
    void markClean() { dirty = false; }

		/** Returns the simplified struct used to represent the current component */
		CameraStruct getStruct();

		/** 
		 * Tells the current camera component to use a projection matrix constructed from a field of view.
		 * The field of view controls the amount of zoom, i.e. the amount of the scene 
		 * which is visible all at once. A smaller field of view results in a longer focal length (more zoom),
		 * while larger field of view allow you to see more of the scene at once (shorter focal length, less zoom)
		 * 
		 * @param field_of_view Specifies the field of view angle in the y direction. Expressed in radians.
		 * @param aspect Specifies the aspect ratio that determines the field of view in the x direction. The aspect ratio is a ratio of x (width) to y (height)
		*/
		void setFOV(float field_of_view, float aspect);

		//  * @param near Specifies the distance from the viewer to the near clipping plane (always positive) 

		/** 
		 * Tells the current camera component to use a projection matrix constructed from a focal length.
		 * The focal length controls the amount of zoom, i.e. the amount of the scene 
		 * which is visible all at once. Longer focal lengths result in a smaller field of view (more zoom),
		 * while short focal lengths allow you to see more of the scene at once (larger FOV, less zoom)
		 * 
		 * @param focal_length Specifies the focal length of the camera lens (in millimeters).
		 * @param sensor_width Specifies the width of the camera sensor (in millimeters). 
		 * @param sensor_height Specifies the height of the camera sensor (in millimeters). 
		 * NOTE: not to be confused with set_focal_distance
		*/
		void setFocalLength(float focal_length, float sensor_width, float sensor_height);

		/** 
		 * Real-world cameras transmit light through a lens that bends and focuses it onto the sensor. 
		 * Because of this, objects that are a certain distance away are in focus, but objects in front 
		 * and behind that are blurred.
		 * 
		 * @param distance The distance to the camera focal position. Note that this is different from focal length, 
		 * and has no effect on the perspective of a camera.
		 * 
		 * NOTE: not to be confused with set_focal_length
		 */
		void setFocalDistance(float distance);

		/** 
		 * Real-world cameras transmit light through a lens that bends and focuses it onto the sensor. 
		 * Because of this, objects that are a certain distance away are in focus, but objects in front 
		 * and behind that are blurred.
		 * 
		 * @param diameter Defines the amount of blurring by setting the diameter of the aperture (in millimeters).
		 */
		void setApertureDiameter(float diameter);
		
		/** @returns the camera to projection matrix.
			This transform can be used to achieve perspective (eg a vanishing point), or for scaling
			an orthographic view. */
		glm::mat4 getProjection();

		/**
		 * The intrinsic matrix is a 3x3 matrix that transforms 3D (non-homogeneous) cooordinates in camera space into 2D (homogeneous) image coordinates. 
		 * These types of matrices are commonly used for computer vision applications, but are less common in computer graphics.
		 * @param width The width of the image (not tracked internally by the camera)
		 * @param height The height of the image (not tracked internally by the camera)
		 * @returns An intrinsic matrix representation of the camera's perspective.
		*/
		glm::mat3 getIntrinsicMatrix(uint32_t width, uint32_t height);

		/** For internal use. Returns the mutex used to lock cameras for processing by the renderer. */
		static std::shared_ptr<std::recursive_mutex> getEditMutex();
};
