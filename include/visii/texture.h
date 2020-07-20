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
	
	/** 
	 * Constructs a Texture with the given name.
     * @return a Texture allocated by the renderer. 
	*/
	static Texture *create(std::string name);

	/** 
	 * Constructs a Texture with the given name from a image located on the filesystem. 
	 * Supported formats include JPEG, PNG, TGA, BMP, PSD, GIF, HDR, PIC, and PNM
	 * @param path The path to the image.
	 * @param linear Indicates the image to load should not be gamma corrected.
     * @returns a Texture allocated by the renderer. 
	*/
	static Texture *createFromImage(std::string name, std::string path, bool linear = false);

	/** 
	 * Constructs a Texture with the given name from custom user data.
	 * @param width The width of the image.
	 * @param height The height of the image.
	 * @param data A row major flattened vector of RGBA texels. The length of this vector should be 4 * width * height.
     * @returns a Texture allocated by the renderer. 
	*/
	static Texture *createFromData(std::string name, uint32_t width, uint32_t height, std::vector<float> data);

    /**
     * @param name The name of the Texture to get
	 * @returns a Texture who's name matches the given name 
	 */
	static Texture *get(std::string name);

    /** @returns a pointer to the table of TextureStructs */
	static TextureStruct *getFrontStruct();

	/** @returns a pointer to the table of Texture components */
	static Texture *getFront();

	/** @returns the number of allocated textures */
	static uint32_t getCount();

	/** @returns the name of this component */
	std::string getName();

	/** @returns A map whose key is a texture name and whose value is the ID for that texture */
	static std::map<std::string, uint32_t> getNameToIdMap();

	/** @param name The name of the Texture to remove */
	static void remove(std::string name);

	/** Allocates the tables used to store all Texture components */
	static void initializeFactory();

	/** @returns True if the tables used to store all Texture components have been allocated, and False otherwise */
	static bool isFactoryInitialized();
    
    /** @returns True the current Texture is a valid, initialized Texture, and False if the Texture was cleared or removed. */
	bool isInitialized();

	/** Iterates through all components, updating any component struct fields and marking components as clean. */
    static void updateComponents();

	/** Clears any existing Texture components. */
    static void clearAll();	

    /** @returns True if the material has been modified since the previous frame, and False otherwise */
    bool isDirty() { return dirty; }

	/** Indicates whether or not any entities are "out of date" and need to be updated through the "update components" function */
	static bool areAnyDirty();

    /** @returns True if the material has not been modified since the previous frame, and False otherwise */
    bool isClean() { return !dirty; }

    /** Tags the current component as being modified since the previous frame. */
    void markDirty();

    /** Tags the current component as being unmodified since the previous frame. */
    void markClean() { dirty = false; }

	/** For internal use. Returns the mutex used to lock entities for processing by the renderer. */
	static std::shared_ptr<std::mutex> getEditMutex();

    /** @returns a json string representation of the current component */
    std::string toString();

    /** @returns a flattened list of texels */
    std::vector<vec4> getTexels();

    /** @returns the width of the texture in texels */
    uint32_t getWidth();
    
    /** @returns the height of the texture in texels */
    uint32_t getHeight();

  private:
  	/** Creates an uninitialized texture. Useful for preallocation. */
	Texture();

	/** Creates a texture with the given name and id. */
	Texture(std::string name, uint32_t id);

  	/* TODO */
	static std::shared_ptr<std::mutex> editMutex;

	/* TODO */
	static bool factoryInitialized;
	
    /** A list of the camera components, allocated statically */
	static Texture textures[MAX_TEXTURES];
	static TextureStruct textureStructs[MAX_TEXTURES];
	
	/** A lookup table of name to camera id */
	static std::map<std::string, uint32_t> lookupTable;

    /** Indicates that one of the components has been edited */
    static bool anyDirty;

    /** Indicates this component has been edited */
    bool dirty = true;

    /** The texels of the texture */
    std::vector<vec4> texels;
};
