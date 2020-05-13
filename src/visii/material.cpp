#include <visii/material.h>

Material Material::materials[MAX_MATERIALS];
MaterialStruct Material::material_structs[MAX_MATERIALS];
std::map<std::string, uint32_t> Material::lookupTable;

std::shared_ptr<std::mutex> Material::creation_mutex;
bool Material::Initialized = false;
bool Material::Dirty = true;

Material::Material() {
	initialized = false;
}

Material::Material(std::string name, uint32_t id)
{
	this->initialized = true;
	this->name = name;
	this->id = id;

	/* Working off blender's principled BSDF */
	material_structs[id].base_color = vec4(.8, .8, .8, 1.0);
	material_structs[id].subsurface_radius = vec4(1.0, .2, .1, 1.0);
	material_structs[id].subsurface_color = vec4(.8, .8, .8, 1.0);
	material_structs[id].subsurface = 0.0;
	material_structs[id].metallic = 0.0;
	material_structs[id].specular = .5;
	material_structs[id].specular_tint = 0.0;
	material_structs[id].roughness = .5;
	material_structs[id].anisotropic = 0.0;
	material_structs[id].anisotropic_rotation = 0.0;
	material_structs[id].sheen = 0.0;
	material_structs[id].sheen_tint = 0.5;
	material_structs[id].clearcoat = 0.0;
	material_structs[id].clearcoat_roughness = .03f;
	material_structs[id].ior = 1.45f;
	material_structs[id].transmission = 0.0;
	material_structs[id].transmission_roughness = 0.0;
	material_structs[id].volume_texture_id = -1;
	material_structs[id].flags = 0;
	material_structs[id].base_color_texture_id = -1;
	material_structs[id].roughness_texture_id = -1;
	material_structs[id].occlusion_texture_id = -1;
	material_structs[id].transfer_function_texture_id = -1;
	material_structs[id].bump_texture_id = -1;
	material_structs[id].alpha_texture_id = -1;
}

std::string Material::to_string() {
	std::string output;
	output += "{\n";
	output += "\ttype: \"Material\",\n";
	output += "\tname: \"" + name + "\"\n";
	output += "\tbase_color: \"" + glm::to_string(material_structs[id].base_color) + "\"\n";
	output += "\tsubsurface: \"" + std::to_string(material_structs[id].subsurface) + "\"\n";
	output += "\tsubsurface_radius: \"" + glm::to_string(material_structs[id].subsurface_radius) + "\"\n";
	output += "\tsubsurface_color: \"" + glm::to_string(material_structs[id].subsurface_color) + "\"\n";
	output += "\tmetallic: \"" + std::to_string(material_structs[id].metallic) + "\"\n";
	output += "\tspecular: \"" + std::to_string(material_structs[id].specular) + "\"\n";
	output += "\tspecular_tint: \"" + std::to_string(material_structs[id].specular_tint) + "\"\n";
	output += "\troughness: \"" + std::to_string(material_structs[id].roughness) + "\"\n";
	output += "\tanisotropic: \"" + std::to_string(material_structs[id].anisotropic) + "\"\n";
	output += "\tanisotropic_rotation: \"" + std::to_string(material_structs[id].anisotropic_rotation) + "\"\n";
	output += "\tsheen: \"" + std::to_string(material_structs[id].sheen) + "\"\n";
	output += "\tsheen_tint: \"" + std::to_string(material_structs[id].sheen_tint) + "\"\n";
	output += "\tclearcoat: \"" + std::to_string(material_structs[id].clearcoat) + "\"\n";
	output += "\tclearcoat_roughness: \"" + std::to_string(material_structs[id].clearcoat_roughness) + "\"\n";
	output += "\tior: \"" + std::to_string(material_structs[id].ior) + "\"\n";
	output += "\ttransmission: \"" + std::to_string(material_structs[id].transmission) + "\"\n";
	output += "\ttransmission_roughness: \"" + std::to_string(material_structs[id].transmission_roughness) + "\"\n";
	output += "}";
	return output;
}

void Material::Initialize()
{
	if (IsInitialized()) return;
	creation_mutex = std::make_shared<std::mutex>();
	Initialized = true;
}

bool Material::IsInitialized()
{
	return Initialized;
}

void Material::UpdateComponents()
{

} 

void Material::CleanUp()
{
	if (!IsInitialized()) return;

	for (auto &material : materials) {
		if (material.initialized) {
			Material::Delete(material.id);
		}
	}

	Initialized = false;
}	

/* Static Factory Implementations */
Material* Material::Create(std::string name) {
	auto mat = StaticFactory::Create(creation_mutex, name, "Material", lookupTable, materials, MAX_MATERIALS);
	Dirty = true;
	return mat;
}

Material* Material::Get(std::string name) {
	return StaticFactory::Get(creation_mutex, name, "Material", lookupTable, materials, MAX_MATERIALS);
}

Material* Material::Get(uint32_t id) {
	return StaticFactory::Get(creation_mutex, id, "Material", lookupTable, materials, MAX_MATERIALS);
}

void Material::Delete(std::string name) {
	StaticFactory::Delete(creation_mutex, name, "Material", lookupTable, materials, MAX_MATERIALS);
	Dirty = true;
}

void Material::Delete(uint32_t id) {
	StaticFactory::Delete(creation_mutex, id, "Material", lookupTable, materials, MAX_MATERIALS);
	Dirty = true;
}

MaterialStruct* Material::GetFrontStruct()
{
	return material_structs;
}

Material* Material::GetFront() {
	return materials;
}

uint32_t Material::GetCount() {
	return MAX_MATERIALS;
}

// void Material::set_base_color_texture(uint32_t texture_id) 
// {
// 	if (texture_id > MAX_TEXTURES)
// 		throw std::runtime_error( std::string("Invalid texture handle"));
// 	this->material_struct.base_color_texture_id = texture_id;
// 	mark_dirty();
// }

// void Material::set_base_color_texture(Texture *texture) 
// {
// 	if (!texture) 
// 		throw std::runtime_error( std::string("Invalid texture handle"));
// 	this->material_struct.base_color_texture_id = texture->get_id();
// 	mark_dirty();
// }

// void Material::clear_base_color_texture() {
// 	this->material_struct.base_color_texture_id = -1;
// 	mark_dirty();
// }

// void Material::set_roughness_texture(uint32_t texture_id) 
// {
// 	if (texture_id > MAX_TEXTURES)
// 		throw std::runtime_error( std::string("Invalid texture handle"));
// 	this->material_struct.roughness_texture_id = texture_id;
// 	mark_dirty();
// }

// void Material::set_roughness_texture(Texture *texture) 
// {
// 	if (!texture) 
// 		throw std::runtime_error( std::string("Invalid texture handle"));
// 	this->material_struct.roughness_texture_id = texture->get_id();
// 	mark_dirty();
// }

// void Material::set_bump_texture(uint32_t texture_id)
// {
// 	if (texture_id > MAX_TEXTURES)
// 		throw std::runtime_error( std::string("Invalid texture handle"));
// 	this->material_struct.bump_texture_id = texture_id;
// 	mark_dirty();
// }

// void Material::set_bump_texture(Texture *texture)
// {
// 	if (!texture)
// 		throw std::runtime_error( std::string("Invalid texture handle"));
// 	this->material_struct.bump_texture_id = texture->get_id();
// 	mark_dirty();
// }

// void Material::clear_bump_texture()
// {
// 	this->material_struct.bump_texture_id = -1;
// 	mark_dirty();
// }

// void Material::set_alpha_texture(uint32_t texture_id)
// {
// 	if (texture_id > MAX_TEXTURES)
// 		throw std::runtime_error( std::string("Invalid texture handle"));
// 	this->material_struct.alpha_texture_id = texture_id;
// 	mark_dirty();
// }

// void Material::set_alpha_texture(Texture *texture)
// {
// 	if (!texture)
// 		throw std::runtime_error( std::string("Invalid texture handle"));
// 	this->material_struct.alpha_texture_id = texture->get_id();
// 	mark_dirty();
// }

// void Material::clear_alpha_texture()
// {
// 	this->material_struct.alpha_texture_id = -1;
// 	mark_dirty();
// }

// void Material::use_vertex_colors(bool use)
// {
// 	if (use) {
// 		this->material_struct.flags |= (1 << 0);
// 	} else {
// 		this->material_struct.flags &= ~(1 << 0);
// 	}
// 	mark_dirty();
// }

// void Material::set_volume_texture(uint32_t texture_id)
// {
// 	this->material_struct.volume_texture_id = texture_id;
// 	mark_dirty();
// }

// void Material::set_volume_texture(Texture *texture)
// {
// 	if (!texture) 
// 		throw std::runtime_error( std::string("Invalid texture handle"));
// 	this->material_struct.volume_texture_id = texture->get_id();
// 	mark_dirty();
// }

// void Material::clear_roughness_texture() {
// 	this->material_struct.roughness_texture_id = -1;
// 	mark_dirty();
// }

// void Material::set_transfer_function_texture(uint32_t texture_id)
// {
// 	this->material_struct.transfer_function_texture_id = texture_id;
// 	mark_dirty();
// }

// void Material::set_transfer_function_texture(Texture *texture)
// {
// 	if (!texture) 
// 		throw std::runtime_error( std::string("Invalid texture handle"));
// 	this->material_struct.transfer_function_texture_id = texture->get_id();
// 	mark_dirty();
// }

// void Material::clear_transfer_function_texture()
// {
// 	this->material_struct.transfer_function_texture_id = -1;
// 	mark_dirty();
// }

// void Material::show_environment(bool show) {
// 	if (show) {
// 		this->material_struct.flags |= (1 << MaterialFlags::MATERIAL_FLAGS_SHOW_SKYBOX);
// 	}
// 	else {
// 		this->material_struct.flags &= ~(1 << MaterialFlags::MATERIAL_FLAGS_SHOW_SKYBOX);
// 	}
// 	mark_dirty();
// }

// void Material::hidden(bool hide) {
// 	if (hide) {
// 		this->material_struct.flags |= (1 << MaterialFlags::MATERIAL_FLAGS_HIDDEN);
// 	}
// 	else {
// 		this->material_struct.flags &= ~(1 << MaterialFlags::MATERIAL_FLAGS_HIDDEN);
// 	}
// }

void Material::set_base_color(glm::vec3 color) {
	material_structs[id].base_color.r = color.r;
	material_structs[id].base_color.g = color.g;
	material_structs[id].base_color.b = color.b;
	mark_dirty();
}

void Material::set_base_color(float r, float g, float b) {
	material_structs[id].base_color.r = r;
	material_structs[id].base_color.g = g;
	material_structs[id].base_color.b = b;
	mark_dirty();
}

glm::vec3 Material::get_base_color() {
	return vec3(material_structs[id].base_color.r, 
				material_structs[id].base_color.g, 
				material_structs[id].base_color.b);
}

void Material::set_subsurface_color(glm::vec3 color) {
	material_structs[id].subsurface_color.r = color.r;
	material_structs[id].subsurface_color.g = color.g;
	material_structs[id].subsurface_color.b = color.b;
	mark_dirty();
}

void Material::set_subsurface_color(float r, float g, float b) {
	material_structs[id].subsurface_color.r = r;
	material_structs[id].subsurface_color.g = g;
	material_structs[id].subsurface_color.b = b;
	mark_dirty();
}

glm::vec3 Material::get_subsurface_color() {
	return glm::vec3(material_structs[id].subsurface_color.r, 
					 material_structs[id].subsurface_color.g, 
					 material_structs[id].subsurface_color.b);
}

void Material::set_subsurface_radius(glm::vec3 radius) {
	material_structs[id].subsurface_radius = glm::vec4(radius.x, radius.y, radius.z, 0.0);
	mark_dirty();
}

void Material::set_subsurface_radius(float x, float y, float z) {
	material_structs[id].subsurface_radius = glm::vec4(x, y, z, 0.0);
	mark_dirty();
}

glm::vec3 Material::get_subsurface_radius() {
	return glm::vec3(material_structs[id].subsurface_radius.x, 
					 material_structs[id].subsurface_radius.y, 
					 material_structs[id].subsurface_radius.z);
}

void Material::set_alpha(float a) 
{
	material_structs[id].base_color.a = a;
	mark_dirty();
}

float Material::get_alpha()
{
	return material_structs[id].base_color.a;
}

void Material::set_subsurface(float subsurface) {
	material_structs[id].subsurface = subsurface;
	mark_dirty();
}

float Material::get_subsurface() {
	return material_structs[id].subsurface;
}

void Material::set_metallic(float metallic) {
	material_structs[id].metallic = metallic;
	mark_dirty();
}

float Material::get_metallic() {
	return material_structs[id].metallic;
}

void Material::set_specular(float specular) {
	material_structs[id].specular = specular;
	mark_dirty();
}

float Material::get_specular() {
	return material_structs[id].specular;
}

void Material::set_specular_tint(float specular_tint) {
	material_structs[id].specular_tint = specular_tint;
	mark_dirty();
}

float Material::get_specular_tint() {
	return material_structs[id].specular_tint;
}

void Material::set_roughness(float roughness) {
	material_structs[id].roughness = roughness;
	mark_dirty();
}

float Material::get_roughness() {
	return material_structs[id].roughness;
}

void Material::set_anisotropic(float anisotropic) {
	material_structs[id].anisotropic = anisotropic;
	mark_dirty();
}

float Material::get_anisotropic() {
	return material_structs[id].anisotropic;
}

void Material::set_anisotropic_rotation(float anisotropic_rotation) {
	material_structs[id].anisotropic_rotation = anisotropic_rotation;
	mark_dirty();
}

float Material::get_anisotropic_rotation() {
	return material_structs[id].anisotropic_rotation;
}

void Material::set_sheen(float sheen) {
	material_structs[id].sheen = sheen;
	mark_dirty();
}

float Material::get_sheen() {
	return material_structs[id].sheen;
}

void Material::set_sheen_tint(float sheen_tint) {
	material_structs[id].sheen_tint = sheen_tint;
	mark_dirty();
}

float Material::get_sheen_tint() {
	return material_structs[id].sheen_tint;
}

void Material::set_clearcoat(float clearcoat) {
	material_structs[id].clearcoat = clearcoat;
	mark_dirty();
}

float Material::get_clearcoat() {
	return material_structs[id].clearcoat;
}

void Material::set_clearcoat_roughness(float clearcoat_roughness) {
	material_structs[id].clearcoat_roughness = clearcoat_roughness;
	mark_dirty();
}

float Material::get_clearcoat_roughness() {
	return material_structs[id].clearcoat_roughness;
}

void Material::set_ior(float ior) {
	material_structs[id].ior = ior;
	mark_dirty();
}

float Material::get_ior() {
	return material_structs[id].ior;
}

void Material::set_transmission(float transmission) {
	material_structs[id].transmission = transmission;
	mark_dirty();
}

float Material::get_transmission() {
	return material_structs[id].transmission;
}

void Material::set_transmission_roughness(float transmission_roughness) {
	material_structs[id].transmission_roughness = transmission_roughness;
	mark_dirty();
}

float Material::get_transmission_roughness() {
	return material_structs[id].transmission_roughness;
}

// bool Material::contains_transparency() {
// 	/* We can expand this to other transparency cases if needed */
// 	if ((this->material_struct.flags & (1 << MaterialFlags::MATERIAL_FLAGS_HIDDEN)) != 0) return true;
// 	if (this->material_struct.alpha_texture_id != -1) return true;
// 	if (this->material_struct.base_color.a < 1.0f) return true;
// 	// if (this->renderMode == RENDER_MODE_VOLUME) return true;
// 	return false;
// }

// bool Material::should_show_skybox()
// {
// 	return ((this->material_struct.flags & (1 << MaterialFlags::MATERIAL_FLAGS_SHOW_SKYBOX)) != 0);
// }

// bool Material::is_hidden()
// {
// 	return ((this->material_struct.flags & (1 << MaterialFlags::MATERIAL_FLAGS_HIDDEN)) != 0);
// }