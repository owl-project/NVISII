#include <visii/material.h>
#include <visii/texture.h>

Material Material::materials[MAX_MATERIALS];
MaterialStruct Material::materialStructs[MAX_MATERIALS];
std::map<std::string, uint32_t> Material::lookupTable;
std::shared_ptr<std::recursive_mutex> Material::editMutex;
bool Material::factoryInitialized = false;
bool Material::anyDirty = true;

Material::Material() {
	initialized = false;
}

Material::Material(std::string name, uint32_t id)
{
	this->initialized = true;
	this->name = name;
	this->id = id;

	/* Working off blender's principled BSDF */
	this->base_color = vec4(.8, .8, .8, 1.0);
	this->subsurface_radius = vec4(1.0, .2, .1, 1.0);
	this->subsurface_color = vec4(.8, .8, .8, 1.0);
	this->subsurface = 0.0;
	this->metallic = 0.0;
	this->specular = .5;
	this->specular_tint = 0.0;
	this->roughness = .5;
	this->anisotropic = 0.0;
	this->anisotropic_rotation = 0.0;
	this->sheen = 0.0;
	this->sheen_tint = 0.5;
	this->clearcoat = 0.0;
	this->clearcoat_roughness = .03f;
	this->ior = 1.45f;
	this->transmission = 0.0;
	this->transmission_roughness = 0.0;
	materialStructs[id].transmission_roughness_texture_id = -1;
	materialStructs[id].base_color_texture_id = -1;
	materialStructs[id].roughness_texture_id = -1;
	materialStructs[id].alpha_texture_id = -1;
	materialStructs[id].normal_map_texture_id = -1;
	materialStructs[id].subsurface_color_texture_id = -1;
	materialStructs[id].subsurface_radius_texture_id = -1;
	materialStructs[id].subsurface_texture_id = -1;
	materialStructs[id].metallic_texture_id = -1;
	materialStructs[id].specular_texture_id = -1;
	materialStructs[id].specular_tint_texture_id = -1;
	materialStructs[id].anisotropic_texture_id = -1;
	materialStructs[id].anisotropic_rotation_texture_id = -1;
	materialStructs[id].sheen_texture_id = -1;
	materialStructs[id].clearcoat_texture_id = -1;
	materialStructs[id].clearcoat_roughness_texture_id = -1;
	materialStructs[id].ior_texture_id = -1;
	materialStructs[id].transmission_texture_id = -1;
}

std::string Material::toString() {
	std::string output;
	output += "{\n";
	output += "\ttype: \"Material\",\n";
	output += "\tname: \"" + name + "\"\n";
	output += "}";
	return output;
}

void Material::initializeFactory()
{
	if (isFactoryInitialized()) return;
	editMutex = std::make_shared<std::recursive_mutex>();
	factoryInitialized = true;
}

bool Material::isFactoryInitialized()
{
	return factoryInitialized;
}

bool Material::isInitialized()
{
	return initialized;
}

bool Material::areAnyDirty()
{
	return anyDirty;
}

void Material::markDirty() {
	dirty = true;
	anyDirty = true;
};

void Material::updateComponents()
{
	if (!anyDirty) return;

	for (int i = 0; i < MAX_MATERIALS; ++i) {
		if (materials[i].isDirty()) {
            materials[i].markClean();
        }
	};
	anyDirty = false;
} 

void Material::clearAll()
{
	if (!isFactoryInitialized()) return;

	for (auto &material : materials) {
		if (material.initialized) {
			Material::remove(material.name);
		}
	}
}	

/* Static Factory Implementations */
Material* Material::create(std::string name,
	vec3  base_color,
	float roughness,
	float metallic,
	float specular,
	float specular_tint,
	float transmission,
	float transmission_roughness,
	float ior,
	float alpha,
	vec3  subsurface_radius,
	vec3  subsurface_color,
	float subsurface,
	float anisotropic,
	float anisotropic_rotation,
	float sheen,
	float sheen_tint,
	float clearcoat,
	float clearcoat_roughness)
{
	auto mat = StaticFactory::create(editMutex, name, "Material", lookupTable, materials, MAX_MATERIALS);
	mat->setBaseColor(base_color);
	mat->setRoughness(roughness);
	mat->setMetallic(metallic);
	mat->setSpecular(specular);
	mat->setSpecularTint(specular_tint);
	mat->setTransmission(transmission);
	mat->setTransmissionRoughness(transmission_roughness);
	mat->setIor(ior);
	mat->setAlpha(alpha);
	mat->setSubsurfaceRadius(subsurface_radius);
	mat->setSubsurfaceColor(subsurface_color);
	mat->setSubsurface(subsurface);
	mat->setAnisotropic(anisotropic);
	mat->setAnisotropicRotation(anisotropic_rotation);
	mat->setSheen(sheen);
	mat->setSheenTint(sheen_tint);
	mat->setClearcoat(clearcoat);
	mat->setClearcoatRoughness(clearcoat_roughness);
	anyDirty = true;
	return mat;
}

std::shared_ptr<std::recursive_mutex> Material::getEditMutex()
{
	return editMutex;
}

Material* Material::get(std::string name) {
	return StaticFactory::get(editMutex, name, "Material", lookupTable, materials, MAX_MATERIALS);
}

void Material::remove(std::string name) {
	StaticFactory::remove(editMutex, name, "Material", lookupTable, materials, MAX_MATERIALS);
	anyDirty = true;
}

MaterialStruct* Material::getFrontStruct()
{
	return materialStructs;
}

Material* Material::getFront() {
	return materials;
}

uint32_t Material::getCount() {
	return MAX_MATERIALS;
}

std::string Material::getName()
{
    return name;
}

std::map<std::string, uint32_t> Material::getNameToIdMap()
{
	return lookupTable;
}

void Material::setBaseColor(glm::vec3 color) {
	base_color.r = color.r;
	base_color.g = color.g;
	base_color.b = color.b;
	markDirty();
}

glm::vec3 Material::getBaseColor() {
	return vec3(base_color.r, base_color.g, base_color.b);
}

void Material::setBaseColorTexture(Texture *texture) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].base_color_texture_id = texture->getId();
	markDirty();
}

void Material::clearBaseColorTexture() {
	materialStructs[id].base_color_texture_id = -1;
	markDirty();
}

void Material::setSubsurfaceColor(glm::vec3 color) {
	subsurface_color.r = color.r;
	subsurface_color.g = color.g;
	subsurface_color.b = color.b;
	markDirty();
}

glm::vec3 Material::getSubsurfaceColor() {
	return glm::vec3(subsurface_color.r, subsurface_color.g, subsurface_color.b);
}

void Material::setSubsurfaceColorTexture(Texture *texture) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].subsurface_color_texture_id = texture->getId();
	markDirty();
}

void Material::clearSubsurfaceColorTexture() {
	materialStructs[id].subsurface_color_texture_id = -1;
	markDirty();
}

void Material::setSubsurfaceRadius(glm::vec3 radius) {
	subsurface_radius = glm::vec4(radius.x, radius.y, radius.z, 0.0);
	markDirty();
}

glm::vec3 Material::getSubsurfaceRadius() {
	return glm::vec3(subsurface_radius.x, subsurface_radius.y, subsurface_radius.z);
}

void Material::setSubsurfaceRadiusTexture(Texture *texture) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].subsurface_radius_texture_id = texture->getId();
	markDirty();
}

void Material::clearSubsurfaceRadiusTexture() {
	materialStructs[id].subsurface_radius_texture_id = -1;
	markDirty();
}

void Material::setAlpha(float a) 
{
	base_color.a = a;
	markDirty();
}

float Material::getAlpha()
{
	return base_color.a;
}

void Material::setAlphaTexture(Texture *texture, int channel) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].alpha_texture_id = texture->getId();
	materialStructs[id].alpha_texture_channel = clamp(channel, 0, 3);
	markDirty();
}

void Material::clearAlphaTexture() {
	materialStructs[id].alpha_texture_id = -1;
	markDirty();
}

void Material::setSubsurface(float subsurface) {
	this->subsurface = subsurface;
	markDirty();
}

float Material::getSubsurface() {
	return this->subsurface;
}

void Material::setSubsurfaceTexture(Texture *texture, int channel) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].subsurface_texture_id = texture->getId();
	materialStructs[id].subsurface_texture_channel = clamp(channel, 0, 3);
	markDirty();
}

void Material::clearSubsurfaceTexture() {
	materialStructs[id].subsurface_texture_id = -1;
	markDirty();
}

void Material::setMetallic(float metallic) {
	this->metallic = metallic;
	markDirty();
}

float Material::getMetallic() {
	return metallic;
}

void Material::setMetallicTexture(Texture *texture, int channel) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].metallic_texture_id = texture->getId();
	materialStructs[id].metallic_texture_channel = clamp(channel, 0, 3);
	markDirty();
}

void Material::clearMetallicTexture() {
	materialStructs[id].metallic_texture_id = -1;
	markDirty();
}

void Material::setSpecular(float specular) {
	this->specular = specular;
	markDirty();
}

float Material::getSpecular() {
	return specular;
}

void Material::setSpecularTexture(Texture *texture, int channel) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].specular_texture_id = texture->getId();
	materialStructs[id].specular_texture_channel = clamp(channel, 0, 3);
	markDirty();
}

void Material::clearSpecularTexture() {
	materialStructs[id].specular_texture_id = -1;
	markDirty();
}

void Material::setSpecularTint(float specular_tint) {
	this->specular_tint = specular_tint;
	markDirty();
}

float Material::getSpecularTint() {
	return specular_tint;
}

void Material::setSpecularTintTexture(Texture *texture, int channel) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].specular_tint_texture_id = texture->getId();
	materialStructs[id].specular_tint_texture_channel = clamp(channel, 0, 3);
	markDirty();
}

void Material::clearSpecularTintTexture() {
	materialStructs[id].specular_tint_texture_id = -1;
	markDirty();
}

void Material::setRoughness(float roughness) {
	this->roughness = roughness;
	markDirty();
}

float Material::getRoughness() {
	return roughness;
}

void Material::setRoughnessTexture(Texture *texture, int channel) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].roughness_texture_id = texture->getId();
	materialStructs[id].roughness_texture_channel = clamp(channel, 0, 3);
	markDirty();
}

void Material::clearRoughnessTexture() {
	materialStructs[id].roughness_texture_id = -1;
	markDirty();
}

void Material::setAnisotropic(float anisotropic) {
	this->anisotropic = anisotropic;
	markDirty();
}

float Material::getAnisotropic() {
	return anisotropic;
}

void Material::setAnisotropicTexture(Texture *texture, int channel) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].anisotropic_texture_id = texture->getId();
	materialStructs[id].anisotropic_texture_channel = clamp(channel, 0, 3);
	markDirty();
}

void Material::clearAnisotropicTexture() {
	materialStructs[id].anisotropic_texture_id = -1;
	markDirty();
}

void Material::setAnisotropicRotation(float anisotropic_rotation) {
	this->anisotropic_rotation = anisotropic_rotation;
	markDirty();
}

float Material::getAnisotropicRotation() {
	return anisotropic_rotation;
}

void Material::setAnisotropicRotationTexture(Texture *texture, int channel) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].anisotropic_rotation_texture_id = texture->getId();
	materialStructs[id].anisotropic_rotation_texture_channel = clamp(channel, 0, 3);
	markDirty();
}

void Material::clearAnisotropicRotationTexture() {
	materialStructs[id].anisotropic_rotation_texture_id = -1;
	markDirty();
}

void Material::setSheen(float sheen) {
	this->sheen = sheen;
	markDirty();
}

float Material::getSheen() {
	return sheen;
}

void Material::setSheenTexture(Texture *texture, int channel) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].sheen_texture_id = texture->getId();
	materialStructs[id].sheen_texture_channel = clamp(channel, 0, 3);
	markDirty();
}

void Material::clearSheenTexture() {
	materialStructs[id].sheen_texture_id = -1;
	markDirty();
}

void Material::setSheenTint(float sheen_tint) {
	this->sheen_tint = sheen_tint;
	markDirty();
}

float Material::getSheenTint() {
	return sheen_tint;
}

void Material::setSheenTintTexture(Texture *texture, int channel) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].sheen_tint_texture_id = texture->getId();
	materialStructs[id].sheen_tint_texture_channel = clamp(channel, 0, 3);
	markDirty();
}

void Material::clearSheenTintTexture() {
	materialStructs[id].sheen_tint_texture_id = -1;
	markDirty();
}

void Material::setClearcoat(float clearcoat) {
	this->clearcoat = clearcoat;
	markDirty();
}

float Material::getClearcoat() {
	return clearcoat;
}

void Material::setClearcoatTexture(Texture *texture, int channel) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].clearcoat_texture_id = texture->getId();
	materialStructs[id].clearcoat_texture_channel = clamp(channel, 0, 3);
	markDirty();
}

void Material::clearClearcoatTexture() {
	materialStructs[id].clearcoat_texture_id = -1;
	markDirty();
}

void Material::setClearcoatRoughness(float clearcoat_roughness) {
	this->clearcoat_roughness = clearcoat_roughness;
	markDirty();
}

float Material::getClearcoatRoughness() {
	return clearcoat_roughness;
}

void Material::setClearcoatRoughnessTexture(Texture *texture, int channel) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].clearcoat_roughness_texture_id = texture->getId();
	materialStructs[id].clearcoat_roughness_texture_channel = clamp(channel, 0, 3);
	markDirty();
}

void Material::clearClearcoatRoughnessTexture() {
	materialStructs[id].clearcoat_roughness_texture_id = -1;
	markDirty();
}

void Material::setIor(float ior) {
	this->ior = ior;
	markDirty();
}

float Material::getIor() {
	return ior;
}

void Material::setIorTexture(Texture *texture, int channel) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].ior_texture_id = texture->getId();
	materialStructs[id].ior_texture_channel = clamp(channel, 0, 3);
	markDirty();
}

void Material::clearIorTexture() {
	materialStructs[id].ior_texture_id = -1;
	markDirty();
}

void Material::setTransmission(float transmission) {
	this->transmission = transmission;
	markDirty();
}

float Material::getTransmission() {
	return transmission;
}

void Material::setTransmissionTexture(Texture *texture, int channel) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].transmission_texture_id = texture->getId();
	materialStructs[id].transmission_texture_channel = clamp(channel, 0, 3);
	markDirty();
}

void Material::clearTransmissionTexture() {
	materialStructs[id].transmission_texture_id = -1;
	markDirty();
}

void Material::setTransmissionRoughness(float transmission_roughness) {
	this->transmission_roughness = transmission_roughness;
	markDirty();
}

float Material::getTransmissionRoughness() {
	return transmission_roughness;
}

void Material::setTransmissionRoughnessTexture(Texture *texture, int channel) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].transmission_roughness_texture_id = texture->getId();
	materialStructs[id].transmission_roughness_texture_channel = clamp(channel, 0, 3);
	markDirty();
}

void Material::clearTransmissionRoughnessTexture() {
	materialStructs[id].transmission_roughness_texture_id = -1;
	markDirty();
}

// bool Material::containsTransparency() {
// 	/* We can expand this to other transparency cases if needed */
// 	if ((this->material_struct.flags & (1 << MaterialFlags::MATERIAL_FLAGS_HIDDEN)) != 0) return true;
// 	if (this->material_struct.alpha_texture_id != -1) return true;
// 	if (this->material_struct.base_color.a < 1.0f) return true;
// 	// if (this->renderMode == RENDER_MODE_VOLUME) return true;
// 	return false;
// }

// bool Material::shouldShowSkybox()
// {
// 	return ((this->material_struct.flags & (1 << MaterialFlags::MATERIAL_FLAGS_SHOW_SKYBOX)) != 0);
// }

// bool Material::isHidden()
// {
// 	return ((this->material_struct.flags & (1 << MaterialFlags::MATERIAL_FLAGS_HIDDEN)) != 0);
// }

void Material::setNormalMapTexture(Texture *texture)
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
	materialStructs[id].normal_map_texture_id = texture->getId();
	markDirty();
}

void Material::clearNormalMapTexture()
{
	materialStructs[id].normal_map_texture_id = -1;
	markDirty();
}

// void Material::useVertexColors(bool use)
// {
// 	if (use) {
// 		this->material_struct.flags |= (1 << 0);
// 	} else {
// 		this->material_struct.flags &= ~(1 << 0);
// 	}
// 	markDirty();
// }

// void Material::setVolumeTexture(uint32_t texture_id)
// {
// 	this->material_struct.volume_texture_id = texture_id;
// 	markDirty();
// }

// void Material::setVolumeTexture(Texture *textur, int channele)
// {
// 	if (!texture) 
// 		throw std::runtime_error( std::string("Invalid texture handle"));
// 	this->material_struct.volume_texture_id = texture->get_id();
// 	markDirty();
// }

// void Material::setTransferFunctionTexture(uint32_t texture_id)
// {
// 	this->material_struct.transfer_function_texture_id = texture_id;
// 	markDirty();
// }

// void Material::setTransferFunctionTexture(Texture *textur, int channele)
// {
// 	if (!texture) 
// 		throw std::runtime_error( std::string("Invalid texture handle"));
// 	this->material_struct.transfer_function_texture_id = texture->get_id();
// 	markDirty();
// }

// void Material::clearTransferFunctionTexture()
// {
// 	this->material_struct.transfer_function_texture_id = -1;
// 	markDirty();
// }

// void Material::showEnvironment(bool show) {
// 	if (show) {
// 		this->material_struct.flags |= (1 << MaterialFlags::MATERIAL_FLAGS_SHOW_SKYBOX);
// 	}
// 	else {
// 		this->material_struct.flags &= ~(1 << MaterialFlags::MATERIAL_FLAGS_SHOW_SKYBOX);
// 	}
// 	markDirty();
// }

// void Material::hidden(bool hide) {
// 	if (hide) {
// 		this->material_struct.flags |= (1 << MaterialFlags::MATERIAL_FLAGS_HIDDEN);
// 	}
// 	else {
// 		this->material_struct.flags &= ~(1 << MaterialFlags::MATERIAL_FLAGS_HIDDEN);
// 	}
// }
