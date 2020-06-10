
#pragma once

#include <visii/utilities/static_factory.h>
#include <visii/light_struct.h>

/**
 * A "Light" component illuminates objects in a scene. Light components must 
 * be added to an entity with a transform component to have a visible      
 * impact on the scene.                                                    
*/
class Light : public StaticFactory {
    friend class StaticFactory;
public:
    /** Constructs a light with the given name.
     * \returns a reference to a light component
     * \param name A unique name for this light
     */
    static Light* create(std::string name);

    /** Constructs a light component which emits a plausible light color based on standard temperature measurement. 
     * \returns a reference to a light component
     * \param name A unique name for this light
     * \param kelvin The temperature of the black body light. Typical values range from 1000K (very warm) to 12000K (very cold).
     * \param intensity How powerful the light source is in emitting light
    */
    static Light* createFromTemperature(std::string name, float kelvin, float intensity);
    
    /** Constructs a light component which emits a given light color.
     * \param name A unique name for this light.
     * \param color An RGB color to emit. Values should range between 0 and 1. 
     * \param intensity How powerful the light source is in emitting light 
    */
    static Light* createFromRGB(std::string name, glm::vec3 color, float intensity);

    /** Gets a light by name 
     * \returns a light who's primary name key matches \p name 
     * \param name A unique name used to lookup this light. */
    static Light* get(std::string name);

    /** Gets a light by id 
     * \returns a light who's primary id key matches \p id 
     * \param id A unique id used to lookup this light. */
    static Light* get(uint32_t id);

    /** \returns a pointer to the table of LightStructs required for rendering */
    static LightStruct* getFrontStruct();

    /** \returns a pointer to the table of light components */
    static Light* getFront();

    /** \returns the number of allocated lights */
    static uint32_t getCount();

    /** Deletes the light who's primary name key matches \p name 
     * \param name A unique name used to lookup the light for deletion.*/
    static void remove(std::string name);

    /** Deletes the light who's primary id key matches \p id 
     * \param id A unique id used to lookup the light for deletion.*/
    static void remove(uint32_t id);

    /** Allocates the tables used to store all light components */
    static void initializeFactory();

    /** \return True if the tables used to store all light components have been allocated, and False otherwise */
    static bool isFactoryInitialized();
    
    /** \return True the current light is a valid, initialized light, and False if the light was cleared or removed. */
	bool isInitialized();

    /** Iterates through all light components, computing light metadata for rendering purposes. */
    static void updateComponents();

    /** Frees any tables used to store light components */
    static void cleanUp();

    /** \return True if this lightmaterial has been modified since the previous frame, and False otherwise */
    bool isDirty() { return dirty; }
    
    /** \return True if any the light has been modified since the previous frame, and False otherwise */
    static bool areAnyDirty();

    /** \return True if the light has not been modified since the previous frame, and False otherwise */
    bool isClean() { return !dirty; }

    /** Tags the current component as being modified since the previous frame. */
    void markDirty();

    /** Tags the current component as being unmodified since the previous frame. */
    void markClean() { dirty = false; }

    /** Returns a json string representation of the current component */
    std::string toString();

    /** Sets the color which this light component will emit.
     * \param r The first of three scalars used to describe the color emitted by this light.
     * \param g The second of three scalars used to describe the color emitted by this light.
     * \param b The third of three scalars used to describe the color emitted by this light.
     */
    void setColor(float r, float g, float b);

    /** Sets the color which this light component will emit. 
     * \param The RGB color emitted that this light should emit.
    */
    void setColor(glm::vec3 color);

    /** Sets a realistic emission color via a temperature.
     * \param kelvin The temperature of the black body light. Typical values range from 1000K (very warm) to 12000K (very cold).
    */
    void setTemperature(float kelvin);

    /** Sets the intensity, or brightness, that this light component will emit it's color. 
     * \param intensity How powerful the light source is in emitting light 
    */
    void setIntensity(float intensity);
    
    // /** In the case of disk and quad area lights, enables or disables light from emitting from one or both sides. 
    //  * \param double_sided If True, emits light from both front and back sides. Otherwise, light will only emit in the forward direction.
    // */
    // void doubleSided(bool double_sided);
    
    // /** In the case of rod area lights, enables or disables light from coming from the ends of the rod. 
    //  * \param show_end_caps If True, emits light from the end caps of a rod area light. Otherwise, end caps do not emit light.
    // */
    // void showEndCaps(bool show_end_caps);

    /** Enables or disables casting shadows from the current light.
     * \aram cast_shadows If True, the light can be occluded, and will cast shadows. Otherwise, light cannot be occluded, and shadows will not be cast.
    */
    // void castShadows(bool cast_shadows);

    // /** Toggles the light on or off 
    //  * \param disabled If True, disables the current light component from emitting light. Otherwise, the light will illuminate objects like normal.
    // */
    // void disabled(bool disabled);

    // /* Sets a cone angle, which creates a spot light like effect 
    // * \param angle The angle, in radians, of the cone used to occlude the light source. 
    // */
    // void setConeAngle(float angle);

    // /* Sets the transition rate of a spot light shadow */
    // void setConeSoftness(float softness);

    /** Make the light act as a point light */
    // void usePoint();

    // /** Make the light act as a plane (or rectangle) area light */
    // void usePlane();

    // /** Make the light act as a disk area light */
    // void useDisk();

    // /** Make the light act as a rod area light */
    // void useRod();

    // /** Make the light act as a sphere area light */
    // void useSphere();


private:
    /* Creates an uninitialized light. Useful for preallocation. */
    Light();
    
    /* Creates a light with the given name and id */
    Light(std::string name, uint32_t id);

    /* A mutex used to make component access and modification thread safe */
    static std::shared_ptr<std::mutex> creationMutex;

    /* Flag indicating that static resources were created */
    static bool factoryInitialized;

    /* A list of light components, allocated statically */
    static Light lights[MAX_LIGHTS];
    static LightStruct lightStructs[MAX_LIGHTS];

    /* A lookup table of name to light id */
    static std::map<std::string, uint32_t> lookupTable;

    /* Indicates that one of the components has been edited */
    static bool anyDirty;

    /* Indicates this component has been edited */
    bool dirty = true;
};
