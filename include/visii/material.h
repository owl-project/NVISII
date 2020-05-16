#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#define GLM_FORCE_RIGHT_HANDED

#include <glm/glm.hpp>
#include <glm/common.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

#include <iostream>
#include <map>
#include <unordered_map>
#include <mutex>
#include <string>

#include <visii/utilities/static_factory.h>
#include <visii/material_struct.h>

// class Texture;

enum MaterialFlags : uint32_t { 
  MATERIAL_FLAGS_HIDDEN = 1, 
  MATERIAL_FLAGS_SHOW_SKYBOX = 2
};

/**
 * The "Material" component describes the surface properties of an entity.
 * This material follows a physically based workflow, more specifically the 
 * Blender principled shader, and very similar to the Disney material model. 
*/
class Material : public StaticFactory
{
  friend class StaticFactory;
  public:
    /** Constructs a material with the given name.
     * \returns a reference to a material component
     * \param name A unique name for this material.
    */
    static Material* create(std::string name);

    /** Gets a material by name 
     * \returns a material who's primary name key matches \p name 
     * \param name A unique name used to lookup this material. */
    static Material* get(std::string name);

    /** Gets a material by id 
     * \returns a material who's primary id key matches \p id 
     * \param id A unique id used to lookup this material. */
    static Material* get(uint32_t id);

    /** \returns a pointer to the table of MaterialStructs required for rendering */
    static MaterialStruct* getFrontStruct();

    /** \returns a pointer to the table of material components */
    static Material* getFront();

    /** \returns the number of allocated materials */
	  static uint32_t getCount();

    /** Deletes the material who's primary name key matches \p name 
     * \param name A unique name used to lookup the material for deletion.*/
    static void remove(std::string name);

    /** Deletes the material who's primary id key matches \p id 
     * \param id A unique id used to lookup the material for deletion.*/
    static void remove(uint32_t id);

    /** Allocates the tables used to store all material components */
    static void initializeFactory();

    /** \return True if the tables used to store all material components have been allocated, and False otherwise */
    static bool isFactoryInitialized();

    /** Iterates through all material components, computing material metadata for rendering purposes. */
    static void updateComponents();

    /** Frees any tables used to store material components */
    static void cleanUp();

    /** \return True if the material has been modified since the previous frame, and False otherwise */
    bool isDirty() { return dirty; }

    /** \return True if the material has not been modified since the previous frame, and False otherwise */
    bool isClean() { return !dirty; }

    /** Tags the current component as being modified since the previous frame. */
    void markDirty() {
        // Dirty = true;
        dirty = true;
    };

    /** Tags the current component as being unmodified since the previous frame. */
    void markClean() { dirty = false; }

    /** Returns a json string representation of the current component */
    std::string toString();

    // /* This method prevents an entity from rendering. */
    // void hidden(bool hide);

    /* Accessors / Mutators */
    
    /** The diffuse or metal surface color. Ignored if a base color texture is set.
      * \param color a red, green, blue color intensity vector, usually between 0 and 1 */
    void setBaseColor(vec3 color);
    
    /** The diffuse or metal surface color. Ignored if a base color texture is set.
      * \param r red intensity, usually between 0 and 1
      * \param g green intensity, usually between 0 and 1
      * \param b blue intensity, usually between 0 and 1 */
    void setBaseColor(float r, float g, float b);

    /** The diffuse or metal surface color. Ignored if a base color texture is set.
      * \returns the color intensity vector */
    vec3 getBaseColor();

    /** Mix between diffuse and subsurface scattering. 
      * \param subsurface Rather than being a simple mix between Diffuse and Subsurface Scattering, 
      * this value controls a multiplier for the Subsurface Radius. */
    void setSubsurface(float subsurface);

    /** Mix between diffuse and subsurface scattering. 
     * \returns the current subsurface radius multiplier. */
    float getSubsurface();

    /** Average distance that light scatters below the surface. Higher radius gives a softer appearance, 
      *  as light bleeds into shadows and through the object. The scattering distance is specified separately 
      *  for the RGB channels, to render materials such as skin where red light scatters deeper. 
      *  \param subsurface_radius control the subsurface radius. The X, Y and Z values of this vector are mapped to the R, G and B radius values, respectively. */
    void setSubsurfaceRadius(vec3 subsurfaceRadius);

    /** Average distance that light scatters below the surface. Higher radius gives a softer appearance, 
      *  as light bleeds into shadows and through the object. The scattering distance is specified separately 
      *  for the RGB channels, to render materials such as skin where red light scatters deeper. 
      *  \param r control the red subsurface radius 
      *  \param g control the green subsurface radius
      *  \param b control the blue subsurface radius */
    void setSubsurfaceRadius(float r, float g, float b);

    /** Average distance that light scatters below the surface. Higher radius gives a softer appearance, 
      *  as light bleeds into shadows and through the object. 
      * \returns The subsurface scattering distance is specified separately for the RGB channels. */
    vec3 getSubsurfaceRadius();

    /** The subsurface scattering base color. 
     * \param color the color intensity vector, usually between 0 and 1 */
    void setSubsurfaceColor(vec3 color);

    /** The subsurface scattering base color. 
     * \param r the red subsurface color intensity 
     * \param g the green subsurface color intensity 
     * \param b the blue subsurface color intensity */
    void setSubsurfaceColor(float r, float g, float b);

    /** The subsurface scattering base color.
     * \returns the color intensity vector, usually between 0 and 1 */
    vec3 getSubsurfaceColor();    
    
    /** Blends between a non-metallic and metallic material model. 
      * \param metallic A value of 1.0 gives 
      * a fully specular reflection tinted with the base color, without diffuse reflection 
      * or transmission. At 0.0 the material consists of a diffuse or transmissive base layer, 
      * with a specular reflection layer on top. */
    void setMetallic(float metallic);

    /** Blends between a non-metallic and metallic material model. 
      * \returns the current metallic value. */
    float getMetallic();

    /** The amount of dielectric specular reflection. 
     * \param specular Specifies facing (along normal) reflectivity in the most common 0 - 8% range. Since materials with reflectivity above 8% do exist, the field allows values above 1.*/
    void setSpecular(float specular);

    /** The amount of dielectric specular reflection. 
      * \returns the current dielectric specular reflection value. */
    float getSpecular();

    /** Tints the facing specular reflection using the base color, while glancing reflection remains white.
      * Normal dielectrics have colorless reflection, so this parameter is not technically physically correct 
      * and is provided for faking the appearance of materials with complex surface structure. 
      * \param specular_tint a value between 0 and 1, enabling/disabling specular tint */
    void setSpecularTint(float specularTint);

    /** Tints the facing specular reflection using the base color, while glancing reflection remains white.
      * \returns the current specular tint value, between 0 and 1 */
    float getSpecularTint();

    /** Microfacet roughness of the surface for diffuse and specular reflection. 
      * \param roughness Specifies the surface microfacet roughness value, between 0 and 1 */
    void setRoughness(float roughness);

    /** Microfacet roughness of the surface for diffuse and specular reflection. 
      * \returns the current surface microfacet roughness value, between 0 and 1 */
    float getRoughness();

    /** The transparency of the surface, independent of transmission.
      * \param a Controls the transparency of the surface, with 1.0 being fully opaque. */
    void setAlpha(float a);

    /** The transparency of the surface, independent of transmission.
      * \returns the current surface transparency, with 1.0 being fully opaque and 0.0 being fully transparent. */
    float getAlpha();

    /** The amount of anisotropy for specular reflection.
      * \param anistropic The amount of anisotropy for specular reflection. Higher values give elongated highlights along the tangent direction; negative values give highlights shaped perpendicular to the tangent direction. */
    void setAnisotropic(float anisotropic);

    /** The amount of anisotropy for specular reflection.
      * \returns The current amount of anisotropy for specular reflection. */
    float getAnisotropic();

    /** The direction of anisotropy.
      * \param anisotropic_rotation Rotates the direction of anisotropy, with 1.0 going full circle. */
    void setAnisotropicRotation(float anisotropicRotation);

    /** The direction of anisotropy.
      * \returns the current the direction of anisotropy, between 0 and 1. */
    float getAnisotropicRotation();

    /** Amount of soft velvet like reflection near edges, for simulating materials such as cloth. 
     * \param sheen controls the amount of sheen, between 0 and 1 */
    void setSheen(float sheen);
    
    /** Amount of soft velvet like reflection near edges, for simulating materials such as cloth. 
     * \returns the current sheen amount, between 0 and 1 */
    float getSheen();

    /** Mix between white and using base color for sheen reflection. 
     * \param sheen_tint controls the mix between white and base color for sheen reflection. */
    void setSheenTint(float sheenTint);
    
    /** Mix between white and using base color for sheen reflection. 
     * \returns the current value used to mix between white and base color for sheen reflection. */
    float getSheenTint();

    /** Extra white specular layer on top of others. This is useful for materials like car paint and the like. 
     * \param clearcoat controls the influence of clear coat, between 0 and 1 */
    void setClearcoat(float clearcoat);

    /** Extra white specular layer on top of others. This is useful for materials like car paint and the like. 
     * \returns the current clear coat influence */
    float getClearcoat();
    
    /** Microfacet surface roughness of clearcoat specular. 
     * \param clearcoat_roughness the roughness of the microfacet distribution influencing the clearcoat, between 0 and 1 */
    void setClearcoatRoughness(float clearcoatRoughness);

    /** Microfacet surface roughness of clearcoat specular. 
     * \returns the current clearcoat microfacet roughness value, between 0 and 1 */
    float getClearcoatRoughness();
    
    /** Index of refraction used for transmission events. 
     * \param ior the index of refraction. A value of 1 results in no refraction. For reference, the IOR of water is roughly 1.33, and for glass is roughly 1.57. */
    void setIor(float ior);

    /** Index of refraction used for transmission events. 
     * \returns the current index of refraction. */
    float getIor();
    
    /** Controls how much the surface looks like glass. Note, metallic takes precedence.
     * \param transmission Mixes between a fully opaque surface at zero to fully glass like transmissions at one. */
    void setTransmission(float transmission);

    /** Controls how much the surface looks like glass. Note, metallic takes precedence.
     * \returns the current specular transmission of the surface. */
    float getTransmission();
    
    /** The roughness of the interior surface used for transmitted light. 
     * \param transmission_roughness Controls the roughness value used for transmitted light. */
    void setTransmissionRoughness(float transmissionRoughness);

    /** The roughness of the interior surface used for transmitted light. 
     * \returns the current roughness value used for transmitted light. */
    float getTransmissionRoughness();
    
    

    // /* Certain constant material properties can be replaced with texture lookups. */
    // void set_base_color_texture(uint32_t texture_id);
    // void set_base_color_texture(Texture *texture);
    // void clear_base_color_texture();

    // void set_roughness_texture(uint32_t texture_id);
    // void set_roughness_texture(Texture *texture);
    // void clear_roughness_texture();

    // void set_bump_texture(uint32_t texture_id);
    // void set_bump_texture(Texture *texture);
    // void clear_bump_texture();

    // void set_alpha_texture(uint32_t texture_id);
    // void set_alpha_texture(Texture *texture);
    // void clear_alpha_texture();

    // /* A uniform base color can be replaced with per-vertex colors as well. */
    // void use_vertex_colors(bool use);

    // /* The volume texture to be used by volume type materials */
    // void set_volume_texture(uint32_t texture_id);
    // void set_volume_texture(Texture *texture);

    // void set_transfer_function_texture(uint32_t texture_id);
    // void set_transfer_function_texture(Texture *texture);
    // void clear_transfer_function_texture();

    // bool contains_transparency();
    // bool should_show_skybox();
    // bool is_hidden();

  private:
    /* Creates an uninitialized material. Useful for preallocation. */
    Material();

    /* Creates a material with the given name and id. */
    Material(std::string name, uint32_t id);

    /* TODO */
    static std::shared_ptr<std::mutex> creationMutex;

    /* TODO */
    static bool factoryInitialized;

    /*  A list of the material components, allocated statically */
    static Material materials[MAX_MATERIALS];
    static MaterialStruct materialStructs[MAX_MATERIALS];

    /* A lookup table of name to material id */
    static std::map<std::string, uint32_t> lookupTable;
    
    /* Indicates that one of the components has been edited */
    static bool anyDirty;

    /* Indicates this component has been edited */
    bool dirty = true;
};
