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
// #include <glm/gtx/string_cast.hpp>

#include <iostream>
#include <map>
#include <unordered_map>
#include <mutex>
#include <string>

#include <visii/utilities/static_factory.h>
#include <visii/material_struct.h>

class Texture;

// enum MaterialFlags : uint32_t { 
//   MATERIAL_FLAGS_HIDDEN = 1, 
//   MATERIAL_FLAGS_SHOW_SKYBOX = 2
// };

/**
 * The "Material" component describes the surface properties of an entity.
 * This material follows a physically based workflow, more specifically the 
 * Blender principled shader, and is very similar to the Disney material model. 
*/
class Material : public StaticFactory
{
  friend class StaticFactory;
  friend class Entity;
  public:
    /**
     * Constructs a material with the given name.
     * 
     * @returns a reference to a material component
     * @param name A unique name for this material.
     * @param base_color The diffuse or metal surface color.
     * @param roughness Microfacet roughness of the surface for diffuse and specular reflection. 
     * @param metallic Blends between a non-metallic and metallic material model. 
     * @param specular The amount of dielectric specular reflection. 
     * @param specular_tint Tints the facing specular reflection using the base color, while glancing reflection remains white.
     * @param transmission Controls how much the surface looks like glass. Note, metallic takes precedence.
     * @param transmission_roughness The roughness of the interior surface used for transmitted light. 
     * @param ior Index of refraction used for transmission events.
     * @param alpha The transparency of the surface, independent of transmission.
     * @param subsurface_radius Average distance that light scatters below the surface
     * @param subsurface_color The subsurface scattering base color. 
     * @param subsurface Mix between diffuse and subsurface scattering. 
     * @param anisotropic The amount of anisotropy for specular reflection.
     * @param anisotropic_rotation The angle of anisotropy.
     * @param sheen Amount of soft velvet like reflection near edges, for simulating materials such as cloth. 
     * @param sheen_tint Mix between white and using base color for sheen reflection. 
     * @param clearcoat Extra white specular layer on top of others.
     * @param clearcoat_roughness Microfacet surface roughness of clearcoat specular. 
    */
    static Material* create(std::string name,
      vec3  base_color = vec3(.8f, .8f, .8f),
      float roughness = .5f,
      float metallic = 0.f, 
      float specular = .5f,
      float specular_tint = 0.f,
      float transmission = 0.f, 
      float transmission_roughness = 0.f, 
      float ior = 1.45f, 
      float alpha = 1.0f, 
      vec3  subsurface_radius = vec3(1.0, .2, .1),
      vec3  subsurface_color = vec3(0.8f, 0.8f, 0.8f),
      float subsurface = 0.f,
      float anisotropic = 0.f, 
      float anisotropic_rotation = 0.f,
      float sheen = 0.f,
      float sheen_tint = 0.5f, 
      float clearcoat = 0.f,
      float clearcoat_roughness = .03f);

    /**
     * Gets a material by name 
     * 
     * @returns a material who's primary name key matches \p name 
     * @param name A unique name used to lookup this material. 
    */
    static Material* get(std::string name);

    /** @returns a pointer to the table of MaterialStructs */
    static MaterialStruct* getFrontStruct();

    /** @returns a pointer to the table of Material components */
    static Material* getFront();

    /** @returns the number of allocated materials */
	  static uint32_t getCount();

    /** @returns the name of this component */
	  std::string getName();

    /** @returns A map whose key is a material name and whose value is the ID for that material */
	  static std::map<std::string, uint32_t> getNameToIdMap();

    /** @param name The name of the material to remove */
    static void remove(std::string name);

    /** Allocates the tables used to store all material components */
    static void initializeFactory();

    /** @returns True if the tables used to store all material components have been allocated, and False otherwise */
    static bool isFactoryInitialized();

    /** @returns True the current material is a valid, initialized material, and False if the material was cleared or removed. */
	  bool isInitialized();

    /** Iterates through all material components, computing material metadata for rendering purposes. */
    static void updateComponents();

    /** Clears any existing material components.*/
    static void clearAll();	

    /** @return True if this material has been modified since the previous frame, and False otherwise */
    bool isDirty() { return dirty; }
    
    /** @return True if any the material has been modified since the previous frame, and False otherwise */
    static bool areAnyDirty();

    /** @return True if the material has not been modified since the previous frame, and False otherwise */
    bool isClean() { return !dirty; }

    /** Tags the current component as being modified since the previous frame. */
    void markDirty();

    /** Tags the current component as being unmodified since the previous frame. */
    void markClean() { dirty = false; }

    /** For internal use. Returns the mutex used to lock entities for processing by the renderer. */
    static std::shared_ptr<std::mutex> getEditMutex();

    /** Returns a json string representation of the current component */
    std::string toString();

    // /* This method prevents an entity from rendering. */
    // void hidden(bool hide);

    /* Accessors / Mutators */
    
    /** 
     * The diffuse or metal surface color. Ignored if a base color texture is set.
     * 
     * @param color a red, green, blue color intensity vector, usually between 0 and 1 
    */
    void setBaseColor(vec3 color);
    
    /** 
     * The diffuse or metal surface color. Texture is expected to be RGB. Overrides any existing constant base color. 
     * 
     * @param texture An RGB texture component whose values range between 0 and 1. Alpha channel is ignored.
    */
    void setBaseColorTexture(Texture *texture);

    /** Disconnects the base color texture, reverting back to any existing constant base color*/
    void clearBaseColorTexture();

    /** 
     * The diffuse or metal surface color. Ignored if a base color texture is set.
     *
     * @returns the color intensity vector 
    */
    vec3 getBaseColor();

    /** 
     * Mix between diffuse and subsurface scattering. 
     * 
     * @param subsurface Rather than being a simple mix between Diffuse and Subsurface Scattering, 
     * this value controls a multiplier for the Subsurface Radius. 
    */
    void setSubsurface(float subsurface);

    /** 
     * Mix between diffuse and subsurface scattering. Overrides any existing constant subsurface 
     * 
     * @param texture A grayscale texture component containing subsurface radius multipliers.
     * @param channel A value between 0 and 3 indicating the channel to use for this parameter.
     */
    void setSubsurfaceTexture(Texture *texture, int channel = 0);

    /** Disconnects the subsurface texture, reverting back to any existing constant subsurface */
    void clearSubsurfaceTexture();

    /** 
     * Mix between diffuse and subsurface scattering. 
     * 
     * @returns the current subsurface radius multiplier. 
    */
    float getSubsurface();

    /** 
     * Average distance that light scatters below the surface. Higher radius gives a softer appearance, 
     * as light bleeds into shadows and through the object. The scattering distance is specified separately 
     * for the RGB channels, to render materials such as skin where red light scatters deeper. 
     *
     * @param subsurface_radius control the subsurface radius. The X, Y and Z values of this vector are mapped to the R, G and B radius values, respectively. 
    */
    void setSubsurfaceRadius(vec3 subsurfaceRadius);

    /** 
     * Average distance that light scatters below the surface. Higher radius gives a softer appearance, 
     * as light bleeds into shadows and through the object. Overrides any existing constant subsurface radius 
     * 
     * @param texture An RGB texture component controlling the subsurface radius in x, y, and z. Alpha channel is ignored.
    */
    void setSubsurfaceRadiusTexture(Texture *texture);

    /** Disconnects the subsurface radius texture, reverting back to any existing constant subsurface radius */
    void clearSubsurfaceRadiusTexture();

    /** 
     * Average distance that light scatters below the surface. Higher radius gives a softer appearance, 
     * as light bleeds into shadows and through the object. 
     * 
     * @returns The subsurface scattering distance is specified separately for the RGB channels. 
    */
    vec3 getSubsurfaceRadius();

    /**
     * The subsurface scattering base color. 
     * 
     * @param color the color intensity vector, usually between 0 and 1 
    */
    void setSubsurfaceColor(vec3 color);

    /** 
     * The subsurface scattering base color. Overrides any existing constant subsurface color 
     * 
     * @param texture An RGB texture whose values range between 0 and 1. Alpha channel is ignored.
     */
    void setSubsurfaceColorTexture(Texture *texture);

    /** Disconnects the subsurface color texture, reverting back to any existing constant subsurface color */
    void clearSubsurfaceColorTexture();

    /* 
     * The subsurface scattering base color.
     * 
     * @returns the color intensity vector, usually between 0 and 1 
    */
    vec3 getSubsurfaceColor();    

    /** 
     * Blends between a non-metallic and metallic material model. 
     * 
     * @param metallic A value of 1.0 gives 
     * a fully specular reflection tinted with the base color, without diffuse reflection 
     * or transmission. At 0.0 the material consists of a diffuse or transmissive base layer, 
     * with a specular reflection layer on top. 
    */
    void setMetallic(float metallic);

    /** 
     * Blends between a non-metallic and metallic material model. Overrides any existing constant metallic 
     * 
     * @param texture A grayscale texture, where texel values of 1 give a fully specular reflection tinted with 
     * the base color, without diffuse reflection or transmission. When texel values equal 0.0 the material 
     * consists of a diffuse or transmissive base layer, with a specular reflection layer on top. 
     * @param channel A value between 0 and 3 indicating the channel to use for this parameter.
     */
    void setMetallicTexture(Texture *texture, int channel = 0);

    /** Disconnects the metallic texture, reverting back to any existing constant metallic */
    void clearMetallicTexture();

    /** 
     * Blends between a non-metallic and metallic material model. 
     * 
     * @returns the current metallic value. 
    */
    float getMetallic();

    /** 
     * The amount of dielectric specular reflection. 
     * 
     * @param specular Specifies facing (along normal) reflectivity in the most common 0 - 8% range. Since materials with reflectivity above 8% do exist, the field allows values above 1.
    */
    void setSpecular(float specular);

    /** 
     * The amount of dielectric specular reflection. Overrides any existing constant specular 
     * 
     * @param texture A grayscale texture containing dielectric specular reflection values.
     * @param channel A value between 0 and 3 indicating the channel to use for this parameter.
     */
    void setSpecularTexture(Texture *texture, int channel = 0);

    /** Disconnects the specular texture, reverting back to any existing constant specular */
    void clearSpecularTexture();

    /** 
     * The amount of dielectric specular reflection. 
     * 
     * @returns the current dielectric specular reflection value. 
    */
    float getSpecular();

    /** 
     * Tints the facing specular reflection using the base color, while glancing reflection remains white.
     * Normal dielectrics have colorless reflection, so this parameter is not technically physically correct 
     * and is provided for faking the appearance of materials with complex surface structure. 
     * 
     * @param specular_tint a value between 0 and 1, enabling/disabling specular tint 
    */
    void setSpecularTint(float specularTint);

    /** 
     * Tints the facing specular reflection using the base color, while glancing reflection remains white. Overrides any existing constant specular tint 
     *
     * @param texture A grayscale texture containing specular tint values, between 0 and 1.
     * @param channel A value between 0 and 3 indicating the channel to use for this parameter.
    */
    void setSpecularTintTexture(Texture *texture, int channel = 0);

    /** Disconnects the specular tint texture, reverting back to any existing constant specular tint */
    void clearSpecularTintTexture();

    /** 
     * Tints the facing specular reflection using the base color, while glancing reflection remains white.
     * 
     * @returns the current specular tint value, between 0 and 1 
    */
    float getSpecularTint();

    /** 
     * Microfacet roughness of the surface for diffuse and specular reflection. 
     * 
     * @param roughness Specifies the surface microfacet roughness value, between 0 and 1 
    */
    void setRoughness(float roughness);

    /** 
     * Microfacet roughness of the surface for diffuse and specular reflection. Overrides any existing constant roughness 
     *
     * @param texture A grayscale texture containing microfacet roughness values, between 0 and 1.
     * @param channel A value between 0 and 3 indicating the channel to use for this parameter.
     */
    void setRoughnessTexture(Texture *texture, int channel = 0);

    /** Disconnects the roughness texture, reverting back to any existing constant roughness */
    void clearRoughnessTexture();

    /** 
     * Microfacet roughness of the surface for diffuse and specular reflection. 
     * 
     * @returns the current surface microfacet roughness value, between 0 and 1 
    */
    float getRoughness();

    /** 
     * The transparency of the surface, independent of transmission.
     * 
     * @param a Controls the transparency of the surface, with 1.0 being fully opaque. 
    */
    void setAlpha(float a);

    /** 
     * The transparency of the surface, independent of transmission. Overrides any existing constant alpha 
     * 
     * @param texture A grayscale texture containing surface transparency values, with 1.0 being fully opaque and 0.0 
     * being fully transparent.
     * @param channel A value between 0 and 3 indicating the channel to use for this parameter.
     */
    void setAlphaTexture(Texture *texture, int channel = 0);

    /** Disconnects the alpha texture, reverting back to any existing constant alpha */
    void clearAlphaTexture();

    /** 
     * The transparency of the surface, independent of transmission.
     * 
     * @returns the current surface transparency, with 1.0 being fully opaque and 0.0 being fully transparent. 
    */
    float getAlpha();

    /** 
     * The amount of anisotropy for specular reflection.
     * 
     * @param anistropic The amount of anisotropy for specular reflection. Higher values give elongated highlights along the tangent direction; negative values give highlights shaped perpendicular to the tangent direction. 
    */
    void setAnisotropic(float anisotropic);

    /** 
     * The amount of anisotropy for specular reflection. Overrides any existing constant anisotropy 
     * 
     * @param texture A grayscale texture containing amounts of anisotropy for specular reflection. G, B, and A channels are ignored.
     * @param channel A value between 0 and 3 indicating the channel to use for this parameter.
     */
    void setAnisotropicTexture(Texture *texture, int channel = 0);

    /** Disconnects the anisotropic texture, reverting back to any existing constant anisotropy */
    void clearAnisotropicTexture();

    /** 
     * The amount of anisotropy for specular reflection.
     * 
     * @returns The current amount of anisotropy for specular reflection. 
    */
    float getAnisotropic();

    /** 
     * The angle of anisotropy.
     * @param anisotropic_rotation Rotates the angle of anisotropy, with 1.0 going full circle. 
    */
    void setAnisotropicRotation(float anisotropicRotation);

    /** 
     * The angle of anisotropy. Overrides any existing constant anisotropic rotation 
     * 
     * @param texture A grayscale texture containing the angle of anisotropy, between 0 and 1.
     * @param channel A value between 0 and 3 indicating the channel to use for this parameter.
     */
    void setAnisotropicRotationTexture(Texture *texture, int channel = 0);
    
    /** Disconnects the anisotropic rotation texture, reverting back to any existing constant anisotropic rotation */
    void clearAnisotropicRotationTexture();

    /** 
     * The angle of anisotropy.
     * 
     * @returns the current the angle of anisotropy, between 0 and 1. 
    */
    float getAnisotropicRotation();

    /** 
     * Amount of soft velvet like reflection near edges, for simulating materials such as cloth. 
     * 
     * @param sheen controls the amount of sheen, between 0 and 1 
    */
    void setSheen(float sheen);
    
    /** 
     * Amount of soft velvet like reflection near edges, for simulating materials such as cloth. Overrides any existing constant sheen 
     * 
     * @param texture A grayscale texture containing amounts of sheen, between 0 and 1.
     * @param channel A value between 0 and 3 indicating the channel to use for this parameter.
     */
    void setSheenTexture(Texture *texture, int channel = 0);

    /** Disconnects the sheen texture, reverting back to any existing constant sheen */
    void clearSheenTexture();

    /** 
     * Amount of soft velvet like reflection near edges, for simulating materials such as cloth. 
     * 
     * @returns the current sheen amount, between 0 and 1 
    */
    float getSheen();

    /** 
     * Mix between white and using base color for sheen reflection. 
     * 
     * @param sheen_tint controls the mix between white and base color for sheen reflection. 
    */
    void setSheenTint(float sheenTint);
    
    /** 
     * Mix between white and using base color for sheen reflection. Overrides any existing constant sheen tint 
     * 
     * @param texture A grayscale texture containing values used to mix between white and base color for sheen reflection. 
     * @param channel A value between 0 and 3 indicating the channel to use for this parameter.     
    */
    void setSheenTintTexture(Texture *texture, int channel = 0);

    /** Disconnects the sheen tint texture, reverting back to any existing constant sheen tint */
    void clearSheenTintTexture();

    /** 
     * Mix between white and using base color for sheen reflection. 
     * 
     * @returns the current value used to mix between white and base color for sheen reflection. 
    */
    float getSheenTint();

    /** 
     * Extra white specular layer on top of others. This is useful for materials like car paint and the like. 
     * 
     * @param clearcoat controls the influence of clear coat, between 0 and 1 
    */
    void setClearcoat(float clearcoat);

    /** 
     * Extra white specular layer on top of others. Overrides any existing constant clearcoat 
     * 
     * @param texture A grayscale texture controlling the influence of clear coat, between 0 and 1.
     * @param channel A value between 0 and 3 indicating the channel to use for this parameter.
     */
    void setClearcoatTexture(Texture *texture, int channel = 0);

    /** Disconnects the clearcoat texture, reverting back to any existing constant clearcoat */
    void clearClearcoatTexture();

    /** 
     * Extra white specular layer on top of others. This is useful for materials like car paint and the like. 
     * 
     * @returns the current clear coat influence 
    */
    float getClearcoat();

    /** 
     * Microfacet surface roughness of clearcoat specular. 
     * 
     * @param clearcoat_roughness the roughness of the microfacet distribution influencing the clearcoat, between 0 and 1 
    */
    void setClearcoatRoughness(float clearcoatRoughness);

    /** 
     * Microfacet surface roughness of clearcoat specular. Overrides any existing constant clearcoat roughness 
     * 
     * @param texture the roughness of the microfacet distribution influencing the clearcoat, between 0 and 1 
     * @param channel A value between 0 and 3 indicating the channel to use for this parameter.
     */
    void setClearcoatRoughnessTexture(Texture *texture, int channel = 0);

    /** Disconnects the clearcoat roughness texture, reverting back to any existing constant clearcoat roughness */
    void clearClearcoatRoughnessTexture();

    /** 
     * Microfacet surface roughness of clearcoat specular. 
     * 
     * @returns the current clearcoat microfacet roughness value, between 0 and 1 
    */
    float getClearcoatRoughness();
    
    /** 
     * Index of refraction used for transmission events. 
     * 
     * @param ior the index of refraction. A value of 1 results in no refraction. For reference, the IOR of water is roughly 1.33, and for glass is roughly 1.57. 
    */
    void setIor(float ior);

    /** 
     * Index of refraction used for transmission events. Overrides any existing constant ior. 
     * 
     * @param texture the index of refraction. A value of 1 results in no refraction. For reference, the IOR of water is roughly 1.33, and for glass is roughly 1.57. 
     * @param channel A value between 0 and 3 indicating the channel to use for this parameter.
     */
    void setIorTexture(Texture *texture, int channel = 0);

    /** Disconnects the ior texture, reverting back to any existing constant ior */
    void clearIorTexture();

    /** 
     * Index of refraction used for transmission events. 
     * 
     * @returns the current index of refraction. 
    */
    float getIor();

    /** 
     * Controls how much the surface looks like glass. Note, metallic takes precedence.
     * 
     * @param transmission Mixes between a fully opaque surface at zero to fully glass like transmissions at one. 
    */
    void setTransmission(float transmission);


    /** 
     * Controls how much the surface looks like glass. Note, metallic takes precedence. Overrides any existing constant transmission. 
     * 
     * @param texture A grayscale texture containing the specular transmission of the surface.
     * @param channel A value between 0 and 3 indicating the channel to use for this parameter.
    */
    void setTransmissionTexture(Texture *texture, int channel = 0);
    
    /** Disconnects the transmission texture, reverting back to any existing constant transmission */
    void clearTransmissionTexture();
    
    /** 
     * Controls how much the surface looks like glass. Note, metallic takes precedence.
     * 
     * @returns the current specular transmission of the surface. 
    */
    float getTransmission();

    /** 
     * The roughness of the interior surface used for transmitted light. 
     * 
     * @param transmission_roughness Controls the roughness value used for transmitted light. 
    */
    void setTransmissionRoughness(float transmissionRoughness);

    /** 
     * The roughness of the interior surface used for transmitted light. Overrides any existing constant transmission roughness 
     * 
     * @param texture Controls the roughness value used for transmitted light. 
     * @param channel A value between 0 and 3 indicating the channel to use for this parameter.
    */
    void setTransmissionRoughnessTexture(Texture *texture, int channel = 0);
    
    /** Disconnects the TransmissionRoughness texture, reverting back to any existing constant TransmissionRoughness */
    void clearTransmissionRoughnessTexture();
    
    /** 
     * The roughness of the interior surface used for transmitted light. 
     * 
     * @returns the current roughness value used for transmitted light. 
    */
    float getTransmissionRoughness();

    /** 
     * A normal map texture used to displace surface normals. 
     * 
     * @param texture A texture containing a surface normal displacement between 0 and 1. A channel is ignored.
    */
    void setNormalMapTexture(Texture *texture);
    
    /** Disconnects the normal map texture */
    void clearNormalMapTexture();

    // /* A uniform base color can be replaced with per-vertex colors as well. */
    // void use_vertex_colors(bool use);

    // /* The volume texture to be used by volume type materials */
    // bool should_show_skybox();
    // bool is_hidden();

  private:
    /* Creates an uninitialized material. Useful for preallocation. */
    Material();

    /* Creates a material with the given name and id. */
    Material(std::string name, uint32_t id);

    /* TODO */
    static std::shared_ptr<std::mutex> editMutex;

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
