#pragma once

#include <mutex>
#include <condition_variable>

#include <visii/utilities/static_factory.h>
#include <visii/volume_struct.h>

/**
 * The "Volume" component is essentially the dual of a mesh component. 
 * As a result, entities can have a mesh component or a volume component attached,
 * but not both.
 *  
 * With a mesh component, surfaces are explicitly defined using triangles, and 
 * the volumes separating that surface are implicit (eg air outside vs glass inside). 
 * With a Volume component, that's reversed. Voxels are used to explicitly represent 
 * the density of particles in space, and surfaces are implicitly defined in areas 
 * where particles are dense.
*/
class Volume : public StaticFactory
{
	friend class StaticFactory;
  public:
	/**
      * Instantiates a null Volume. Used to mark a row in the table as null. 
      * Note: for internal use only. 
     */
    Volume();
    
    /**
    * Instantiates a Volume with the given name and ID. Used to mark a row in 
	* the table as null. 
    * Note: for internal use only.
    */
    Volume(std::string name, uint32_t id);

	/**
    * Destructs a Volume.
    * Note: for internal use only.
    */
	~Volume();

	/** 
	 * Constructs a Volume with the given name from a file. 
	 * @param name The name of the volume to create.
	 * Supported formats include NanoVDB (.nvdb)
	 * @param path The path to the file.
	 * @returns a Volume allocated by the renderer. 
	*/
	static Volume *createFromFile(std::string name, std::string path);
	
    /**
     * @param name The name of the Volume to get
	 * @returns a Volume who's name matches the given name 
	 */
	static Volume *get(std::string name);

    /** @returns a pointer to the table of VolumeStructs */
	static VolumeStruct *getFrontStruct();

	/** @returns a pointer to the table of Volume components */
	static Volume *getFront();

	/** @returns the number of allocated volumes */
	static uint32_t getCount();

	/** @returns the name of this component */
	std::string getName();

	/** @returns the unique integer ID for this component */
	int32_t getId();

	// for internal use
	int32_t getAddress();

	/** 
	 * @returns A map whose key is a volume name and whose value is the ID for 
	 * that volume 
	 */
	static std::map<std::string, uint32_t> getNameToIdMap();

	/** @param name The name of the Volume to remove */
	static void remove(std::string name);

	/** Allocates the tables used to store all Volume components */
	static void initializeFactory(uint32_t max_components);

	/** 
	 * @returns True if the tables used to store all Volume components have been 
	 * allocated, and False otherwise 
	 */
	static bool isFactoryInitialized();
    
    /** 
	 * @returns True the current Volume is a valid, initialized Volume, and 
	 * False if the Volume was cleared or removed. 
	 */
	bool isInitialized();

	/** 
	 * Iterates through all components, updating any component struct fields 
	 * and marking components as clean. 
	 */
    static void updateComponents();

	/** Clears any existing Volume components. */
    static void clearAll();	

	/** 
	 * Indicates whether or not any entities are "out of date" and need to be 
	 * updated through the "update components" function 
	 */
	static bool areAnyDirty();

	/** 
	 * @returns a list of volumes that have been modified since the previous 
	 * frame 
	 */
    static std::set<Volume*> getDirtyVolumes();

    /** Tags the current component as being modified since the previous frame. */
    void markDirty();

	/** 
	 * For internal use. Returns the mutex used to lock entities for processing 
	 * by the renderer. 
	 */
	static std::shared_ptr<std::recursive_mutex> getEditMutex();

    /** @returns a json string representation of the current component */
    std::string toString();

  private:

  	/* TODO */
	static std::shared_ptr<std::recursive_mutex> editMutex;

	/* TODO */
	static bool factoryInitialized;
	
    /** A list of the volume components, allocated statically */
	static std::vector<Volume> volumes;
	static std::vector<VolumeStruct> volumeStructs;
	
	/** A lookup table of name to volume id */
	static std::map<std::string, uint32_t> lookupTable;

	static std::set<Volume*> dirtyVolumes;

    /** Private volume data here... */
    
};
