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
	friend class Material;
	friend class Light;
  public:
	/**
      * Instantiates a null Texture. Used to mark a row in the table as null. 
      * Note: for internal use only. 
     */
    Texture();
    
    /**
    * Instantiates a Texture with the given name and ID. Used to mark a row in the table as null. 
    * Note: for internal use only.
    */
    Texture(std::string name, uint32_t id);

	/**
    * Destructs a Texture.
    * Note: for internal use only.
    */
	~Texture();

	/** 
	 * Constructs a Texture with the given name.
	 * @param name The name of the texture to create.
     * @return a Texture allocated by the renderer. 
	*/
	static Texture *create(std::string name);

	/** 
	 * Deprecated. Please use createFromFile. 
	*/
	static Texture *createFromImage(std::string name, std::string path, bool linear = false);

	/** 
	 * Constructs a Texture with the given name from a file. 
	 * @param name The name of the texture to create.
	 * Supported formats include JPEG, PNG, TGA, BMP, PSD, GIF, HDR, PIC, and PNM
	 * @param path The path to the image.
	 * @param linear Indicates the image to load should not be gamma corrected.
     * @returns a Texture allocated by the renderer. 
	*/
	static Texture *createFromFile(std::string name, std::string path, bool linear = false);

	/** 
	 * Constructs a Texture with the given name from custom user data.
	 * @param name The name of the texture to create.
	 * @param width The width of the image.
	 * @param height The height of the image.
	 * @param data A row major flattened vector of RGBA texels. The length of this vector should be 4 * width * height.
     * @returns a Texture allocated by the renderer. 
	*/
	static Texture *createFromData(std::string name, uint32_t width, uint32_t height, const float* data, uint32_t length);
	
	/** 
	 * Constructs a Texture with the given name that mixes two different textures together.
	 * @param name The name of the texture to create.
	 * @param a The first of two textures to mix. 
	 * @param b The second of two textures to mix. 
	 * @param mix A value between 0 and 1 used to mix between the first and second textures.
     * @returns a Texture allocated by the renderer. 
	*/
	static Texture *createMix(std::string name, Texture* a, Texture* b, float mix);

	/** 
	 * Constructs a Texture with the given name that adds two different textures together.
	 * @param name The name of the texture to create.
	 * @param a The first of two textures to add. 
	 * @param b The second of two textures to add. 
	 * @returns a Texture allocated by the renderer. 
	*/
	static Texture *createAdd(std::string name, Texture* a, Texture* b);

	/** 
	 * Constructs a Texture with the given name that multiplies two different textures together.
	 * @param name The name of the texture to create.
	 * @param a The first of two textures to multiply. 
	 * @param b The second of two textures to multiply. 
	 * @returns a Texture allocated by the renderer. 
	*/
	static Texture *createMultiply(std::string name, Texture* a, Texture* b);

	/** 
	 * Constructs a Texture with the given name by applying a color transformation on the HSV space to an existing texture.
	 * @param name The name of the texture to create.
	 * @param t The texture to take pixels from
	 * @param hue Specifies the hue rotation of the image. 360 degrees are mapped to [0,1]. 
	 * The hue shifts of 0 (-180) and 1 (180) have the same result.
	 * @param saturation A saturation of 0 removes hues from the image, resulting in a grayscale image. 
	 * A shift greater than 1.0 increases saturation.
	 * @param value is the overall brightness of the image. De/Increasing values shift an image darker/lighter.
	 * @param mix A value between 0 and 1 used to mix between the original input and the HSV transformed image. 
     * @returns a Texture allocated by the renderer. 
	*/
	static Texture* createHSV(std::string name, Texture* tex, float hue, float saturation, float value, float mix);

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
	static void initializeFactory(uint32_t max_components);

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
	static std::shared_ptr<std::recursive_mutex> getEditMutex();

    /** @returns a json string representation of the current component */
    std::string toString();

    /** @returns a flattened list of texels */
    std::vector<vec4> getTexels();

    /** @returns a flattened list of 8-bit texels */
	std::vector<i8vec4> get8BitTexels();

	/**
	 * Sample the texture at the given texture coordinates
	 * @param uv A pair of values between [0,0] and [1,1]
	 * @returns a sampled texture value
	*/
	vec4 sample(vec2 uv);

    /** @returns the width of the texture in texels */
    uint32_t getWidth();
    
    /** @returns the height of the texture in texels */
    uint32_t getHeight();

	/**
	 * Sets the "scale" of a texture. Useful for patterns that repeat, eg tiles.
	 * Under the hood, this scales the texture coordinates of the object this texture influences.
	 * @param scale The scale of the texture. A value of [.5,.5] will cause a texture to take 
	 * up half the footprint in UV space of the original texture, effectively causing the texture 
	 * to repeat in a pattern. Textures can be flipped in either U and/or V using negative scales.
	*/
	void setScale(glm::vec2 scale);

	/** @returns True if the texture contains any values above 1 */
    bool isHDR();

  private:
  	/* TODO */
	static std::shared_ptr<std::recursive_mutex> editMutex;

	/* TODO */
	static bool factoryInitialized;
	
    /** A list of the camera components, allocated statically */
	static std::vector<Texture> textures;
	static std::vector<TextureStruct> textureStructs;
	
	/** A lookup table of name to camera id */
	static std::map<std::string, uint32_t> lookupTable;

    /** Indicates that one of the components has been edited */
    static bool anyDirty;

    /** Indicates this component has been edited */
    bool dirty = true;

    /** The texels of the texture */
    std::vector<vec4> texels;
};
