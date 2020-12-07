#include <visii/volume.h>

#include <cstring>
#include <algorithm>

#include <glm/gtc/color_space.hpp>

std::vector<Volume> Volume::volumes;
std::vector<VolumeStruct> Volume::volumeStructs;
std::map<std::string, uint32_t> Volume::lookupTable;
std::shared_ptr<std::recursive_mutex> Volume::editMutex;
bool Volume::factoryInitialized = false;
std::set<Volume*> Volume::dirtyVolumes;

Volume::Volume()
{
    this->initialized = false;
}

Volume::~Volume()
{
    
}

Volume::Volume(std::string name, uint32_t id)
{
    this->initialized = true;
    this->name = name;
    this->id = id;
}

std::string Volume::toString() {
    std::string output;
    output += "{\n";
    output += "\ttype: \"Volume\",\n";
    output += "\tname: \"" + name + "\"\n";
    output += "}";
    return output;
}

/* SSBO logic */
void Volume::initializeFactory(uint32_t max_components)
{
    if (isFactoryInitialized()) return;
    volumes.resize(max_components);
    volumeStructs.resize(max_components);
    editMutex = std::make_shared<std::recursive_mutex>();
    factoryInitialized = true;
}

bool Volume::isFactoryInitialized()
{
    return factoryInitialized;
}

bool Volume::isInitialized()
{
	return initialized;
}

bool Volume::areAnyDirty()
{
    return dirtyVolumes.size() > 0;
}

void Volume::markDirty() {
    if (getAddress() < 0 || getAddress() >= volumes.size()) {
        throw std::runtime_error("Error, volume not allocated in list");
    }
	dirtyVolumes.insert(this);
};

std::set<Volume*> Volume::getDirtyVolumes()
{
	return dirtyVolumes;
}

void Volume::updateComponents()
{
    if (dirtyVolumes.size() == 0) return;
	dirtyVolumes.clear();
} 

void Volume::clearAll()
{
    if (!isFactoryInitialized()) return;

    for (auto &volume : volumes) {
		if (volume.initialized) {
			Volume::remove(volume.name);
		}
	}
}

/* Static Factory Implementations */
Volume* Volume::createFromFile(std::string name, std::string path) {
    auto create = [path] (Volume* v) {
        v->markDirty();
    };

    try {
        return StaticFactory::create<Volume>(editMutex, name, "Volume", lookupTable, volumes.data(), volumes.size(), create);
    } catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Volume", lookupTable, volumes.data(), volumes.size());
		throw;
	}
}

std::shared_ptr<std::recursive_mutex> Volume::getEditMutex()
{
	return editMutex;
}

Volume* Volume::get(std::string name) {
    return StaticFactory::get(editMutex, name, "Volume", lookupTable, volumes.data(), volumes.size());
}

void Volume::remove(std::string name) {
    auto t = get(name);
	if (!t) return;
    int32_t oldID = t->getId();
	StaticFactory::remove(editMutex, name, "Volume", lookupTable, volumes.data(), volumes.size());
	dirtyVolumes.insert(&volumes[oldID]);
}

Volume* Volume::getFront() {
    return volumes.data();
}

VolumeStruct* Volume::getFrontStruct() {
    return volumeStructs.data();
}

uint32_t Volume::getCount() {
    return volumes.size();
}

std::string Volume::getName()
{
    return name;
}

int32_t Volume::getId()
{
    return id;
}

int32_t Volume::getAddress()
{
	return (this - volumes.data());
}

std::map<std::string, uint32_t> Volume::getNameToIdMap()
{
	return lookupTable;
}
