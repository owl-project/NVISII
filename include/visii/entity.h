#pragma once

#include <visii/utilities/static_factory.h>
#include <visii/entity_struct.h>

class Camera;
class Transform;
class Material;
class Mesh;

/**
 * The "Entity" component is the most basic component in a scene.
 * They can be thought of as a "join table" in a data base, connecting
 * different objects together through a collection of primary keys.
 * Currently, only one component of a given type can be connected to an entity 
 * at any given point in time.
 * In order to place an Entity into a scene, connect to a Transform component.
 * For an Entity to be visible, connect both a Mesh component and a Material
 * component.
 * 
*/
class Entity : public StaticFactory {
	friend class StaticFactory;
private:
	/** If an entity isn't active, its callbacks aren't called */
	bool active = true;

	//std::shared_ptr<Callbacks> callbacks;
	//std::map<std::type_index, std::vector<std::shared_ptr<Component>>> components;
	
	/** Prevents multiple components from simultaneously being added and/or removed from the component list */
	static std::shared_ptr<std::mutex> creationMutex;
	
    /** Marks that the StaticFactory has allocated the table of components */
	static bool factoryInitialized;
	
    /** The table of Entity components */
	static Entity entities[MAX_ENTITIES];

    /** The table of Entity Structs */
	static EntityStruct entityStructs[MAX_ENTITIES];

    /** A lookup table where, given the name of a component, returns the primary key of that component */
	static std::map<std::string, uint32_t> lookupTable;
    
    /** If an entity has a camera component, a window can be uniquely mapped to a specific entity.
     * Once mapped, the window will show the output image rendered by the camera. */
	static std::map<std::string, uint32_t> windowToEntity;

    /** If a window is mapped to a particular entity, this mapping can be used to determine the window connected to a given entity. */
	static std::map<uint32_t, std::string> entityToWindow;

    /** Instantiates a null Entity. Used to mark a row in the table as null. 
     * Note: for internal use only. */
	Entity();

    /** Instantiates an Entity with the given name and ID. Used to mark a row in the table as null. 
     * Note: for internal use only.
    */
	Entity(std::string name, uint32_t id);

	/** Indicates that one of the components has been edited */
    static bool anyDirty;

    /** Indicates this component has been edited */
    bool dirty = true;

public:
    /** Constructs an Entity with the given name.
     * \return an Entity allocated by the renderer. */
	static Entity* create(std::string name, 
		Transform* transform = nullptr, 
		Material* material = nullptr,
		Mesh* mesh = nullptr,
		Camera* camera = nullptr//,
		// Light* light = nullptr,
		// RigidBody* rigid_body = nullptr,
		// Collider* collider = nullptr
	);

    /** Gets an Entity by name 
     * \return an Entity who's primary name key matches \p name */
	static Entity* get(std::string name);
    
    /** Gets an Entity by id 
     * \return an Entity who's primary id key matches \p id */
	static Entity* get(uint32_t id);

    /** \return a pointer to the table of EntityStructs */
	static EntityStruct* getFrontStruct();

    /** \return a pointer to the table of Entity components */
	static Entity* getFront();

    /** \return the number of allocated entities */
	static uint32_t getCount();

    /** Deletes the Entity who's primary name key matches \p name */
	static void remove(std::string name);

    /** Deletes the Entity who's primary id key matches \p id */
	static void remove(uint32_t id);
	
    /** Allocates the tables used to store all Entity components */
    static void initializeFactory();

    /** \return True if the tables used to store all Entity components have been allocated, and False otherwise */
	static bool isFactoryInitialized();

    // static void UpdateComponents(); // remove this... 

    // static void UploadSSBO(vk::CommandBuffer command_buffer);
    // static vk::Buffer GetSSBO();
	// static uint32_t GetSSBOSize();

    /** Frees any tables used to store Entity components */
    static void cleanUp();	

    /** \return a string representation of the current component */
	std::string toString();
	
	// void set_rigid_body(int32_t rigid_body_id);
	// void set_rigid_body(RigidBody* rigid_body);
	// void clear_rigid_body();
	// int32_t get_rigid_body_id();
	// RigidBody* get_rigid_body();

	// void set_collider(int32_t collider_id);
	// void set_collider(Collider* collider);
	// void clear_collider();
	// int32_t get_collider_id();
	// Collider* get_collider();

    /** \return True if the Entity has been modified since the previous frame, and False otherwise */
	bool isDirty() { return dirty; }

    /** \return True if the Entity has not been modified since the previous frame, and False otherwise */
	bool isClean() { return !dirty; }

    /** Tags the current component as being modified since the previous frame. */
	void markDirty() {
		// Dirty = true;
		dirty = true;
	};

    /** Tags the current component as being unmodified since the previous frame. */
	void markClean() { dirty = false; }

    /** Returns the simplified struct used to represent the current component */
	EntityStruct getStruct();

    /** Connects a transform component to the current entity by primary id key */
	void setTransform(int32_t transform_id);

    /** Connects a transform component to the current entity */
	void setTransform(Transform* transform);

    /** Disconnects any transform component from the current entity */
	void clearTransform();
    
    /** \return the primary id key of the connected transform component, or -1 if no component is connected. */
	int32_t getTransformId();

    /** \return a reference to the connected transform component, or None/nullptr if no component is connected. */
	Transform* getTransform();

	/** Connects a camera component to the current entity by primary id key */
	void setCamera(int32_t camera_id);

	/** Connects a camera component to the current entity */
	void setCamera(Camera *camera);

	/** Disconnects any camera component from the current entity */
	void clearCamera();

	/** \return the primary id key of the connected camera component, or -1 if no component is connected. */
	int32_t getCameraId();

	/** \return a reference to the connected camera component, or None/nullptr if no component is connected. */
	Camera* getCamera();
	
	/** Connects a material component to the current entity by primary id key */
    void setMaterial(int32_t material_id);

	/** Connects a material component to the current entity */
    void setMaterial(Material *material);

	/** Disconnects any material component from the current entity */
    void clearMaterial();

	/** \return the primary id key of the connected material component, or -1 if no component is connected. */
    int32_t getMaterialId();

	/** \return a reference to the connected material component, or None/nullptr if no component is connected. */
    Material* getMaterial();


	// void set_light(int32_t light_id);
	// void set_light(Light* light);
	// void clear_light();
	// int32_t get_light_id();
	// Light* get_light();
	
	/** Connects a mesh component to the current entity by primary id key */
	void setMesh(int32_t mesh_id);
	
	/** Connects a mesh component to the current entity */
	void setMesh(Mesh* mesh);
	
	/** Disconnects any mesh component from the current entity */
	void clearMesh();
	
	/** \return the primary id key of the connected mesh component, or -1 if no component is connected. */
	int32_t getMeshId();
	
	/** \return a reference to the connected mesh component, or None/nullptr if no component is connected. */
	Mesh* getMesh();
};
