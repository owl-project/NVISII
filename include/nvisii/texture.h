#pragma once

#include <mutex>
#include <condition_variable>

#include <nvisii/utilities/static_factory.h>
#include <nvisii/texture_struct.h>

namespace nvisii {

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
	 * Deprecated. Please use createFromFile. 
	*/
	static Texture *createFromImage(std::string name, std::string path, bool linear = false);

	/** 
	 * Constructs a Texture with the given name from a file. 
	 * @param name The name of the texture to create.
	 * Supported formats include JPEG, PNG, TGA, BMP, PSD, GIF, HDR, PIC, PNM, KTX, and DDS
	 * @param path The path to the image.
	 * @param linear Indicates the image is already linear and should not be gamma corrected. Ignored for KTX, DDS, and HDR formats.
     * @returns a Texture allocated by the renderer. 
	*/
	static Texture *createFromFile(std::string name, std::string path, bool linear = false);

	/** 
	 * Constructs a Texture with the given name from custom user data.
	 * @param name The name of the texture to create.
	 * @param width The width of the image.
	 * @param height The height of the image.
	 * @param data A row major flattened vector of RGBA texels. The length of this vector should be 4 * width * height.
	 * @param linear Indicates the image is already linear and should not be gamma corrected. Note, defaults to True for this function.
	 * @param hdr If true, represents the channels of the texture using 32 bit floats. Otherwise, textures are stored natively using 8 bits per channel.
     * @returns a Texture allocated by the renderer. 
	*/
	static Texture *createFromData(std::string name, uint32_t width, uint32_t height, const float* data, uint32_t length, bool linear = true, bool hdr = false);
	
	/** 
	 * Constructs a Texture with the given name that mixes two different textures together.
	 * @param name The name of the texture to create.
	 * @param a The first of two textures to mix. 
	 * @param b The second of two textures to mix. 
	 * @param mix A value between 0 and 1 used to mix between the first and second textures.
	 * @param hdr If true, represents the channels of the texture using 32 bit floats. Otherwise, textures are stored natively using 8 bits per channel.
     * @returns a Texture allocated by the renderer. 
	*/
	static Texture *createMix(std::string name, Texture* a, Texture* b, float mix = 1.0, bool hdr = false);

	/** 
	 * Constructs a Texture with the given name that adds two different textures together.
	 * @param name The name of the texture to create.
	 * @param a The first of two textures to add. 
	 * @param b The second of two textures to add. 
	 * @param hdr If true, represents the channels of the texture using 32 bit floats. Otherwise, textures are stored natively using 8 bits per channel.
	 * @returns a Texture allocated by the renderer. 
	*/
	static Texture *createAdd(std::string name, Texture* a, Texture* b, bool hdr = false);

	/** 
	 * Constructs a Texture with the given name that multiplies two different textures together.
	 * @param name The name of the texture to create.
	 * @param a The first of two textures to multiply. 
	 * @param b The second of two textures to multiply.
	 * @param hdr If true, represents the channels of the texture using 32 bit floats. Otherwise, textures are stored natively using 8 bits per channel. 
	 * @returns a Texture allocated by the renderer. 
	*/
	static Texture *createMultiply(std::string name, Texture* a, Texture* b, bool hdr = false);

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
	 * @param hdr If true, represents the channels of the texture using 32 bit floats. Otherwise, textures are stored natively using 8 bits per channel. 
     * @returns a Texture allocated by the renderer. 
	*/
	static Texture* createHSV(std::string name, Texture* tex, float hue, float saturation, float value, float mix = 1.0, bool hdr = false);

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

	/** @returns the unique integer ID for this component */
	int32_t getId();

	// for internal use
	int32_t getAddress();

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

	/** Indicates whether or not any entities are "out of date" and need to be updated through the "update components" function */
	static bool areAnyDirty();

	/** @returns a list of textures that have been modified since the previous frame */
    static std::set<Texture*> getDirtyTextures();

    /** Tags the current component as being modified since the previous frame. */
    void markDirty();

	/** For internal use. Returns the mutex used to lock entities for processing by the renderer. */
	static std::shared_ptr<std::recursive_mutex> getEditMutex();

    /** @returns a json string representation of the current component */
    std::string toString();

    /** @returns a flattened list of 32-bit float texels */
    std::vector<vec4> getFloatTexels();

    /** @returns a flattened list of 8-bit texels */
	std::vector<u8vec4> getByteTexels();

	/**
	 * Sample the texture at the given texture coordinates
	 * @param uv A pair of values between [0,0] and [1,1]
	 * @returns a sampled texture value
	*/
	vec4 sampleFloatTexels(vec2 uv);
	
	/**
	 * Sample the texture at the given texture coordinates
	 * @param uv A pair of values between [0,0] and [1,1]
	 * @returns a sampled texture value
	*/
	u8vec4 sampleByteTexels(vec2 uv);

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

	/** @returns True if the texture is represented linearly. Otherwise, the texture is in sRGB space */
    bool isLinear();

  private:
  	/* TODO */
	static std::shared_ptr<std::recursive_mutex> editMutex;

	/* TODO */
	static bool factoryInitialized;
	
    /** A list of the texture components, allocated statically */
	static std::vector<Texture> textures;
	static std::vector<TextureStruct> textureStructs;
	
	/** A lookup table of name to texture id */
	static std::map<std::string, uint32_t> lookupTable;

	static std::set<Texture*> dirtyTextures;

    /** The texels of the texture */
    std::vector<vec4> floatTexels;
    std::vector<u8vec4> byteTexels;
	bool linear = false;
};

};
