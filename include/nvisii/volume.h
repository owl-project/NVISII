#pragma once

#include <mutex>
#include <condition_variable>

#include <nvisii/utilities/static_factory.h>
#include <nvisii/volume_struct.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridValidator.h>

namespace nvisii {

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
	friend class Entity;
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
	 * Creates a sparse fog volume of a sphere such that the exterior
	 * is 0 and inactive, the interior is active with values varying
	 * smoothly from 0 at the surface of the sphere to 1 at the half 
	 * width and interior of the sphere.
	*/
	static Volume *createSphere(std::string name);
	
	/**
	 * Creates a sparse fog volume of a torus in the xz-plane such
	 * that the exterior is 0 and inactive, the interior is active with
	 * values varying smoothly from 0 at the surface of the torus to 1
	 * at the half width and interior of the torus.
	*/
	static Volume *createTorus(std::string name);
	
	/**
	 * Creates a sparse fog volume of a box such that the exterior 
	 * is 0 and inactive, the interior is active with values varying
	 * smoothly from 0 at the surface of the box to 1 at the half width
	 * and interior of the box.
	 * @param name The name of the volume to create.
	 * @param size The width, height, and depth of the box in local units.
	 * @param center The center of the box in local units
	 * @param half_width The half-width of the narrow band in voxel units
	 */
	static Volume *createBox(std::string name, 
		glm::vec3 size = glm::vec3(100.f), 
		glm::vec3 center = glm::vec3(0.f), 
		float half_width = 3.f);
	
	/**
	 * Creates a sparse fog volume of an octahedron such that the exterior
	 * is 0 and inactive, the interior is active with values varying
	 * smoothly from 0 at the surface of the octahedron to 1 at the half width
	 * and interior of the octahedron
	 */
	static Volume *createOctahedron(std::string name);

	/**
	 * Constructs a Volume with the given name from custom user data. 
	 * @param name The name of the volume to create.
	 * @param width The width of the volume.
	 * @param height The height of the volume.
	 * @param depth The depth of the volume.
	 * @param data A row major flattened vector of single-scalar voxels. 
	 * The length of this vector should be width * height * depth.
	 * @param background If a voxel matches this value, that voxel is considered
	 * as "empty". This is used to "sparcify" the volume and save memory.
	 */
	static Volume *createFromData(
		std::string name, 
		uint32_t width, 
		uint32_t height, 
		uint32_t depth, 
		const float* data, 
		uint32_t length,
		float background);

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

	/** @returns the type of the volume's scalar field */
	std::string getGridType();

	/** 
	 * @param level The level of nodes being referenced (0->3 = leaf -> root)
	 * @returns the number of nodes, or subvolumes, used to store a volume.
	 * (See the illustration of the NanoVDB data structure for more details)
	 */
	uint32_t getNodeCount(uint32_t level);

	/**
	 * @param level The level of nodes being referenced (0->3 = leaf -> root).
	 * @param node_idx The index of the node within the selected level. 
	 * @returns the minimum node axis aligned bounding box position 
	 */
	glm::vec3 getMinAabbCorner(uint32_t level, uint32_t node_idx);

	/** 
	 * @param level The level of nodes being referenced (0->3 = leaf -> root).
	 * @param node_idx The index of the node within the selected level.
	 * @returns the maximum node axis aligned bounding box position 
	 */
	glm::vec3 getMaxAabbCorner(uint32_t level, uint32_t node_idx);

	/** 
	 * @param level The level of nodes being referenced (0->3 = leaf -> root).
	 * @param node_idx The index of the node within the selected level.
	 * @returns the center of the aligned bounding box for a node 
	 */
	glm::vec3 getAabbCenter(uint32_t level, uint32_t node_idx);

	/** @returns the handle to the nanovdb grid. For internal purposes. */
	std::shared_ptr<nanovdb::GridHandle<>> getNanoVDBGridHandle();

	/** todo... document */
	void setScale(float units);

	/** todo... document */
	void setScattering(float scattering);
	
	/** todo... document */
	void setAbsorption(float absorption);

	/** todo... document */
	void setGradientFactor(float factor);

	/** todo... document */
	float getMax(uint32_t level, uint32_t node_idx);

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
	std::shared_ptr<nanovdb::GridHandle<>> gridHdlPtr;
};

};
