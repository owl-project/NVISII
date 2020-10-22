#pragma once

#include <visii/utilities/static_factory.h>
#include <visii/entity_struct.h>

class Camera;
class Light;
class Transform;
class Material;
class Mesh;

/**
 * An "Entity" is a component that is used to connect other component types together. 
 * 
 * If you'd like to place an object in the scene, an Entity would be used to 
 * connect a Mesh component, a Transform component, and a Material component together.
 * 
 * Only one component of a given type can be connected to an entity at any given point 
 * in time.
 * 
 * Multiple entities can "share" the same component type. For example, a sphere mesh
 * component can be referenced by many entities, with each entity having a unique 
 * material. This is essentially a form of "instancing". Alternatively, many 
 * different entities can share the same Material component, such that any adjustments
 * to that material component effect a collection of objects instead of just one.
 */
class Entity : public StaticFactory {
	friend class StaticFactory;
private:
	/** If an entity isn't active, its callbacks aren't called */
	bool active = true;

	/** Prevents multiple components from simultaneously being added and/or removed from the component list */
	static std::shared_ptr<std::recursive_mutex> editMutex;
	
    /** Marks that the StaticFactory has allocated the table of components */
	static bool factoryInitialized;
	
    /** The table of Entity components */
	static std::vector<Entity> entities;

    /** The table of Entity structs */
	static std::vector<EntityStruct> entityStructs;

    /** A lookup table where, given the name of a component, returns the primary key of that component */
	static std::map<std::string, uint32_t> lookupTable;
	
	static std::set<Entity*> dirtyEntities;
	static std::set<Entity*> renderableEntities;

public:
	/**
	 * Instantiates a null Entity. Used to mark a row in the table as null. 
     * Note: for internal use only. 
	 */
	Entity();

    /**
	 * Instantiates an Entity with the given name and ID. Used to mark a row in the table as null. 
     * Note: for internal use only.
	 */
	Entity(std::string name, uint32_t id);

    /**
	 * Constructs an Entity with the given name.
	 * 
	 * @param transform (optional) A transform component places the entity into the scene
	 * @param material (optional) A material component describes how an entity should look when rendered.
	 * @param mesh (optional) A mesh component describes the geometry of the entity to be rendered. 
	 * @param light (optional) A light component indicates that any connected geometry should act like a light source.
	 * @param camera (optional) A camera component indicates that the current entity can be used to view into the scene.
     * @returns a reference to an Entity
	 */
	static Entity* create(std::string name, 
		Transform* transform = nullptr, 
		Material* material = nullptr,
		Mesh* mesh = nullptr,
		Light* light = nullptr,
		Camera* camera = nullptr
	);

	/**
     * @param name The name of the entity to get
	 * @returns an Entity who's name matches the given name 
	 */
	static Entity* get(std::string name);

    /** @returns a pointer to the table of EntityStructs */
	static EntityStruct* getFrontStruct();

    /** @returns a pointer to the table of Entity components */
	static Entity* getFront();

    /** @returns the number of allocated entities */
	static uint32_t getCount();

	/** @returns the name of this component */
	std::string getName();

	/** @returns the unique integer ID for this component */
	int32_t getId();

	/** @returns A map whose key is an entity name and whose value is the ID for that entity */
	static std::map<std::string, uint32_t> getNameToIdMap();

    /** @param name The name of the Entity to remove */
	static void remove(std::string name);
	
    /** Allocates the tables used to store all Entity components */
    static void initializeFactory(uint32_t max_components);

    /** @returns True if the tables used to store all Entity components have been allocated, and False otherwise */
	static bool isFactoryInitialized();

	/** @returns True the current entity is a valid, initialized entity, and False if the entity was cleared or removed. */
	bool isInitialized();

	/** Iterates through all components, updating any component struct fields and marking components as clean. */
    static void updateComponents();

    /** Clears any existing entity components. This function can be used to reset a scene. */
    static void clearAll();	

    /** @returns a string representation of the current component */
	std::string toString();

	/** @return True if any the entities has been modified since the previous frame, and False otherwise */
	static bool areAnyDirty();

    /** @returns a list of entities that have been modified since the previous frame */
    static std::set<Entity*> getDirtyEntities();

	/** @returns a list of entities that are renderable (ie, can be seen) by the camera. (note, currently ignores visibility) */
    static std::set<Entity*> getRenderableEntities();

    /** Tags the current component as being modified since the previous frame. */
	void markDirty();

    /** Returns the simplified struct used to represent the current component */
	EntityStruct &getStruct();

    /** Connects a transform component to the current entity */
	void setTransform(Transform* transform);

    /** Disconnects any transform component from the current entity */
	void clearTransform();
    
    /** @returns a reference to the connected transform component, or None/nullptr if no component is connected. */
	Transform* getTransform();

	/** Connects a camera component to the current entity */
	void setCamera(Camera *camera);

	/** Disconnects any camera component from the current entity */
	void clearCamera();

	/** @returns a reference to the connected camera component, or None/nullptr if no component is connected. */
	Camera* getCamera();
	
	/** Connects a material component to the current entity */
    void setMaterial(Material *material);

	/** Disconnects any material component from the current entity */
    void clearMaterial();

	/** @returns a reference to the connected material component, or None/nullptr if no component is connected. */
    Material* getMaterial();

	/** Connects a light component to the current entity */
	void setLight(Light* light);
	
	/** Disconnects any light component from the current entity */
	void clearLight();
	
	/** @returns a reference to the connected light component, or None/nullptr if no component is connected. */
	Light* getLight();
	
	/** Connects a mesh component to the current entity */
	void setMesh(Mesh* mesh);
	
	/** Disconnects any mesh component from the current entity */
	void clearMesh();
	
	/** @returns a reference to the connected mesh component, or None/nullptr if no component is connected. */
	Mesh* getMesh();

	/**
	 * Objects can be set to be invisible to particular ray types:
	 * @param camera Makes the object visible to camera rays
	*/
	void setVisibility(bool camera = true);

	/** @returns the minimum axis aligned bounding box position. Requires a transform and mesh component to be attached. */
	glm::vec3 getMinAabbCorner();
	
	/** @returns the maximum axis aligned bounding box position. Requires a transform and mesh component to be attached. */
	glm::vec3 getMaxAabbCorner();

	/** @returns the center of the aligned bounding box. Requires a transform and mesh component to be attached. */
	glm::vec3 getAabbCenter();

	/** For internal use. Returns the mutex used to lock entities for processing by the renderer. */
	static std::shared_ptr<std::recursive_mutex> getEditMutex();

	/** For internal use. */
	void computeAabb();

	/** For internal use. */
	void updateRenderables();
};
