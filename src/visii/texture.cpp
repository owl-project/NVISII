#include <visii/texture.h>

Texture Texture::textures[MAX_TEXTURES];
TextureStruct Texture::textureStructs[MAX_TEXTURES];
std::map<std::string, uint32_t> Texture::lookupTable;
std::shared_ptr<std::mutex> Texture::creationMutex;
bool Texture::factoryInitialized = false;
bool Texture::anyDirty = true;

Texture::Texture()
{
    this->initialized = false;
}

Texture::Texture(std::string name, uint32_t id)
{
    this->initialized = true;
    this->name = name;
    this->id = id;
    
    // TODO: INITIALIZE STRUCT
}

std::string Texture::toString() {
    std::string output;
    output += "{\n";
    output += "\ttype: \"Texture\",\n";
    output += "\tname: \"" + name + "\"\n";
    output += "}";
    return output;
}

/* SSBO logic */
void Texture::initializeFactory()
{
    if (isFactoryInitialized()) return;
    creationMutex = std::make_shared<std::mutex>();
    factoryInitialized = true;
}

bool Texture::isFactoryInitialized()
{
    return factoryInitialized;
}

bool Texture::areAnyDirty()
{
	return anyDirty;
}

void Texture::markDirty() {
	dirty = true;
    anyDirty = true;
};

void Texture::updateComponents()
{
	if (!anyDirty) return;

	for (int i = 0; i < MAX_TEXTURES; ++i) {
		if (textures[i].isDirty()) {
            textures[i].markClean();
        }
	};
	anyDirty = false;
} 

void Texture::cleanUp()
{
    if (!isFactoryInitialized()) return;

    for (auto &light : textures) {
		if (light.initialized) {
			Texture::remove(light.id);
		}
	}

    factoryInitialized = false;
}	

/* Static Factory Implementations */
Texture* Texture::create(std::string name) {
    auto l = StaticFactory::create(creationMutex, name, "Texture", lookupTable, textures, MAX_TEXTURES);
    anyDirty = true;
    return l;
}

Texture* Texture::get(std::string name) {
    return StaticFactory::get(creationMutex, name, "Texture", lookupTable, textures, MAX_TEXTURES);
}

Texture* Texture::get(uint32_t id) {
    return StaticFactory::get(creationMutex, id, "Texture", lookupTable, textures, MAX_TEXTURES);
}

void Texture::remove(std::string name) {
    StaticFactory::remove(creationMutex, name, "Texture", lookupTable, textures, MAX_TEXTURES);
    anyDirty = true;
}

void Texture::remove(uint32_t id) {
    StaticFactory::remove(creationMutex, id, "Texture", lookupTable, textures, MAX_TEXTURES);
    anyDirty = true;
}

Texture* Texture::getFront() {
    return textures;
}

TextureStruct* Texture::getFrontStruct() {
    return textureStructs;
}

uint32_t Texture::getCount() {
    return MAX_TEXTURES;
}
