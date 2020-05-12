// ┌──────────────────────────────────────────────────────────────────┐
// │  Entity                                                          │
// └──────────────────────────────────────────────────────────────────┘

#pragma once

#include <visii/utilities/static_factory.h>
#include <visii/entity_struct.h>

// class Camera;
class Transform;
class Material;
// class Light;
// class Mesh;
// class RigidBody;
// class Collider;


/**
 * The "Entity" component is the most basic component in a scene.
 * They can be thought of as a "join table" in a data base, connecting
 * different objects together through a collection of primary keys.
 * 
 * Currently, only one component of a given type can be connected to an entity 
 * at any given point in time.
 * 
 * In order to place an Entity into a scene, connect to a Transform component.
 * 
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
	static std::shared_ptr<std::mutex> creation_mutex;
	
    /** Marks that the StaticFactory has allocated the table of components */
	static bool Initialized;
	
    /** The table of Entity components */
	static Entity entities[MAX_ENTITIES];

    /** The table of Entity Structs */
	static EntityStruct entity_structs[MAX_ENTITIES];

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
    static bool Dirty;

    /** Indicates this component has been edited */
    bool dirty = true;

public:
    /** Constructs an Entity with the given name.
     * \return an Entity allocated by the renderer. */
	static Entity* Create(std::string name, 
		Transform* transform = nullptr, 
		Material* material = nullptr//,
		// Camera* camera = nullptr,
		// Light* light = nullptr,
		// Mesh* mesh = nullptr,
		// RigidBody* rigid_body = nullptr,
		// Collider* collider = nullptr
	);

    /** Gets an Entity by name 
     * \return an Entity who's primary name key matches \p name */
	static Entity* Get(std::string name);
    
    /** Gets an Entity by id 
     * \return an Entity who's primary id key matches \p id */
	static Entity* Get(uint32_t id);

    /** \return a pointer to the table of EntityStructs */
	static EntityStruct* GetFrontStruct();

    /** \return a pointer to the table of Entity components */
	static Entity* GetFront();

    /** \return the number of allocated entities */
	static uint32_t GetCount();

    /** Deletes the Entity who's primary name key matches \p name */
	static void Delete(std::string name);

    /** Deletes the Entity who's primary id key matches \p id */
	static void Delete(uint32_t id);
	
    /** Allocates the tables used to store all Entity components */
    static void Initialize();

    /** \return True if the tables used to store all Entity components have been allocated, and False otherwise */
	static bool IsInitialized();

    // static void UpdateComponents(); // remove this... 

    // static void UploadSSBO(vk::CommandBuffer command_buffer);
    // static vk::Buffer GetSSBO();
	// static uint32_t GetSSBOSize();

    /** Frees any tables used to store Entity components */
    static void CleanUp();	

    /** \return a string representation of the current component */
	std::string to_string();
	
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
	bool is_dirty() { return dirty; }

    /** \return True if the Entity has not been modified since the previous frame, and False otherwise */
	bool is_clean() { return !dirty; }

    /** Tags the current component as being modified since the previous frame. */
	void mark_dirty() {
		// Dirty = true;
		dirty = true;
	};

    /** Tags the current component as being unmodified since the previous frame. */
	void mark_clean() { dirty = false; }

    /** Returns the simplified struct used to represent the current component */
	EntityStruct get_struct();

    /** Connects a transform component to the current entity by primary id key */
	void set_transform(int32_t transform_id);

    /** Connects a transform component to the current entity */
	void set_transform(Transform* transform);

    /** Disconnects any transform component from the current entity */
	void clear_transform();
    
    /** \return the primary id key of the connected transform component, or -1 if no component is connected. */
	int32_t get_transform_id();

    /** \return a reference to the connected transform component, or None/nullptr if no component is connected. */
	Transform* get_transform();

	// void set_camera(int32_t camera_id);
	// void set_camera(Camera *camera);
	// void clear_camera();
	// int32_t get_camera_id();
	// Camera* get_camera();

	/** Connects a material component to the current entity by primary id key */
    void set_material(int32_t material_id);

	/** Connects a material component to the current entity */
    void set_material(Material *material);

	/** Disconnects any material component from the current entity */
    void clear_material();

	/** \return the primary id key of the connected material component, or -1 if no component is connected. */
    int32_t get_material_id();

	/** \return a reference to the connected material component, or None/nullptr if no component is connected. */
    Material* get_material();


	// void set_light(int32_t light_id);
	// void set_light(Light* light);
	// void clear_light();
	// int32_t get_light_id();
	// Light* get_light();

	// void set_mesh(int32_t mesh_id);
	// void set_mesh(Mesh* mesh);
	// void clear_mesh();
	// int32_t get_mesh_id();
	// Mesh* get_mesh();
};
