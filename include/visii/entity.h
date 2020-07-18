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
	static std::shared_ptr<std::mutex> editMutex;
	
    /** Marks that the StaticFactory has allocated the table of components */
	static bool factoryInitialized;
	
    /** The table of Entity components */
	static Entity entities[MAX_ENTITIES];

    /** The table of Entity structs */
	static EntityStruct entityStructs[MAX_ENTITIES];

    /** A lookup table where, given the name of a component, returns the primary key of that component */
	static std::map<std::string, uint32_t> lookupTable;

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

	/** Indicates that one of the components has been edited */
    static bool anyDirty;

    /** Indicates this component has been edited */
    bool dirty = true;

public:
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

	/** @returns A map whose key is an entity name and whose value is the ID for that entity */
	static std::map<std::string, uint32_t> getNameToIdMap();

    /** @param name The name of the Entity to remove */
	static void remove(std::string name);
	
    /** Allocates the tables used to store all Entity components */
    static void initializeFactory();

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

	/** Indicates whether or not any entities are "out of date" and need to be updated through the "update components" function */
	static bool areAnyDirty();

    /** @returns True if the Entity has been modified since the previous frame, and False otherwise */
	bool isDirty() { return dirty; }

    /** @returns True if the Entity has not been modified since the previous frame, and False otherwise */
	bool isClean() { return !dirty; }

    /** Tags the current component as being modified, and in need of updating. */
	void markDirty();

    /** Tags the current component as being unmodified, or updated. */
	void markClean() { dirty = false; }

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

	/** For internal use. Returns the mutex used to lock entities for processing by the renderer. */
	static std::shared_ptr<std::mutex> getEditMutex();
};
