#include <nvisii/light.h>
#include <nvisii/texture.h>

namespace nvisii {

std::vector<Light> Light::lights;
std::vector<LightStruct> Light::lightStructs;
std::map<std::string, uint32_t> Light::lookupTable;
std::shared_ptr<std::recursive_mutex> Light::editMutex;
bool Light::factoryInitialized = false;
bool Light::anyDirty = true;

Light::Light()
{
    this->initialized = false;
}

Light::Light(std::string name, uint32_t id)
{
    this->initialized = true;
    this->name = name;
    this->id = id;
    
    this->lightStructs[id].r = 1.f;
    this->lightStructs[id].g = 1.f;
    this->lightStructs[id].b = 1.f;
    this->lightStructs[id].intensity = 1.0;
    this->lightStructs[id].exposure = 0.0;
    this->lightStructs[id].color_texture_id = -1;
}

std::string Light::toString() {
    std::string output;
    output += "{\n";
    output += "\ttype: \"Light\",\n";
    output += "\tname: \"" + name + "\"\n";
    output += "}";
    return output;
}

LightStruct &Light::getStruct() {
	if (!isInitialized()) throw std::runtime_error("Error: light is uninitialized.");
	return lightStructs[id];
}

void Light::setColor(glm::vec3 color)
{
    auto &light = getStruct();
    light.r = max(0.f, min(color.r, 1.f));
    light.g = max(0.f, min(color.g, 1.f));
    light.b = max(0.f, min(color.b, 1.f));
    markDirty();
}

glm::vec3 Light::getColor()
{
    return glm::vec3(lightStructs[id].r, lightStructs[id].g, lightStructs[id].b);
}

void Light::setColorTexture(Texture *texture) 
{
	if (!texture) throw std::runtime_error( std::string("Invalid texture handle"));
    auto &light = getStruct();
	light.color_texture_id = texture->getId();
    texture->lights.insert(id);
	markDirty();
}

void Light::clearColorTexture() {
    auto &light = getStruct();
    auto textures = Texture::getFront();
    if (light.color_texture_id != -1) 
        textures[light.color_texture_id].lights.erase(id);
	light.color_texture_id = -1;
	markDirty();
}

void Light::setTemperature(float kelvin)
{
    kelvin = max(kelvin, 0.f);

    float temp = kelvin / 100.0f;

    float red, green, blue;

    if ( temp <= 66 ){ 
        red = 255;         
        green = temp;
        green = 99.4708025861f * logf(green) - 161.1195681661f;

        if( temp <= 19){
            blue = 0;
        } else {
            blue = temp-10.f;
            blue = 138.5177312231f * logf(blue) - 305.0447927307f;
        }
    } else {
        red = temp - 60.f;
        red = 329.698727446f * powf(red, -0.1332047592f);
        
        green = temp - 60;
        green = 288.1221695283f * powf(green, -0.0755148492f );

        blue = 255;
    }

    auto &light = getStruct();
    light.r = red / 255.f;
    light.g = green / 255.f;
    light.b = blue / 255.f;
    markDirty();
}

void Light::setIntensity(float intensity)
{
    auto &light = getStruct();
    light.intensity = intensity;
    markDirty();
}

float Light::getIntensity()
{
    auto &light = getStruct();
    return light.intensity;
}

void Light::setExposure(float exposure)
{
    auto &light = getStruct();
    light.exposure = exposure;
    markDirty();
}

float Light::getExposure()
{
    auto &light = getStruct();
    return light.exposure;
}

void Light::setFalloff(float falloff)
{
    auto &light = getStruct();
    light.falloff = falloff;
    markDirty();
}

float Light::getFalloff()
{
    auto &light = getStruct();
    return light.falloff;
}

void Light::useSurfaceArea(bool use) 
{
    auto &light = getStruct();
    light.use_surface_area = use;
}

/* SSBO logic */
void Light::initializeFactory(uint32_t max_components)
{
    if (isFactoryInitialized()) return;
    lights.resize(max_components);
    lightStructs.resize(max_components);
    editMutex = std::make_shared<std::recursive_mutex>();
    factoryInitialized = true;
}

bool Light::isFactoryInitialized()
{
    return factoryInitialized;
}

bool Light::isInitialized()
{
	return initialized;
}

bool Light::areAnyDirty()
{
	return anyDirty;
}

void Light::markDirty() {
	dirty = true;
    anyDirty = true;
};

void Light::updateComponents()
{
	if (!anyDirty) return;

	for (int i = 0; i < lights.size(); ++i) {
		if (lights[i].isDirty()) {
            lights[i].markClean();
        }
	};
	anyDirty = false;
} 

void Light::clearAll()
{
    if (!isFactoryInitialized()) return;

    for (auto &light : lights) {
		if (light.initialized) {
			Light::remove(light.name);
		}
	}
}	

/* Static Factory Implementations */
Light* Light::create(std::string name) {
    auto l = StaticFactory::create(editMutex, name, "Light", lookupTable, lights.data(), lights.size());
    anyDirty = true;
    return l;
}

Light* Light::createFromTemperature(std::string name, float kelvin, float intensity) {
    auto light = StaticFactory::create(editMutex, name, "Light", lookupTable, lights.data(), lights.size());
    light->setTemperature(kelvin);
    light->setIntensity(intensity);
    return light;
}

Light* Light::createFromRGB(std::string name, glm::vec3 color, float intensity) {
    auto light = StaticFactory::create(editMutex, name, "Light", lookupTable, lights.data(), lights.size());
    light->setColor(color);
    light->setIntensity(intensity);
    return light;
}

std::shared_ptr<std::recursive_mutex> Light::getEditMutex()
{
	return editMutex;
}

Light* Light::get(std::string name) {
    return StaticFactory::get(editMutex, name, "Light", lookupTable, lights.data(), lights.size());
}

void Light::remove(std::string name) {
    StaticFactory::remove(editMutex, name, "Light", lookupTable, lights.data(), lights.size());
    anyDirty = true;
}

Light* Light::getFront() {
    return lights.data();
}

LightStruct* Light::getFrontStruct() {
    return lightStructs.data();
}

uint32_t Light::getCount() {
    return uint32_t(lights.size());
}

std::string Light::getName()
{
    return name;
}

std::map<std::string, uint32_t> Light::getNameToIdMap()
{
	return lookupTable;
}

};
