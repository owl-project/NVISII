#pragma once

#include <mutex>
#include <condition_variable>

#include <visii/utilities/static_factory.h>
#include <visii/texture_struct.h>

/**
 * The "Texture" component describes a 2D pattern used to drive the "Material" component's parameters.
*/
class Texture : public StaticFactory
{
	friend class StaticFactory;
  public:
	
	/* Creates a texture component by name. */
	static Texture *create(std::string name);

	/* Retrieves a texture component by name. */
	static Texture *get(std::string name);

	/* Retrieves a texture component by id. */
	static Texture *get(uint32_t id);

	static TextureStruct *getFrontStruct();

	/* Returns a pointer to the list of texture components. */
	static Texture *getFront();

	/* Returns the total number of reserved textures. */
	static uint32_t getCount();

	/* Deallocates the texture with the given name. */
	static void remove(std::string name);

	/* Deallocates the texture with the given id. */
	static void remove(uint32_t id);

	/* Initializes the Texture factory. Loads any default components. */
	static void initializeFactory();

	/* TODO: Explain this */
	static bool isFactoryInitialized();

    static void updateComponents();

	/* Releases vulkan resources */
	static void cleanUp();

    /** \return True if the material has been modified since the previous frame, and False otherwise */
    bool isDirty() { return dirty; }

	static bool areAnyDirty();

    /** \return True if the material has not been modified since the previous frame, and False otherwise */
    bool isClean() { return !dirty; }

    /** Tags the current component as being modified since the previous frame. */
    void markDirty();

    /** Tags the current component as being unmodified since the previous frame. */
    void markClean() { dirty = false; }

    /** Returns a json string representation of the current component */
    std::string toString();

  private:
  	/* Creates an uninitialized texture. Useful for preallocation. */
	Texture();

	/* Creates a texture with the given name and id. */
	Texture(std::string name, uint32_t id);

  	/* TODO */
	static std::shared_ptr<std::mutex> creationMutex;

	/* TODO */
	static bool factoryInitialized;
	
    /* A list of the camera components, allocated statically */
	static Texture textures[MAX_TEXTURES];
	static TextureStruct textureStructs[MAX_TEXTURES];
	
	/* A lookup table of name to camera id */
	static std::map<std::string, uint32_t> lookupTable;

    /* Indicates that one of the components has been edited */
    static bool anyDirty;

    /* Indicates this component has been edited */
    bool dirty = true;
};
