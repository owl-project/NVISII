#include <visii/volume.h>

#include <cstring>
#include <algorithm>

#include <glm/gtc/color_space.hpp>

#include <nanovdb/util/GridChecksum.h>
#include <nanovdb/util/Primitives.h>

namespace visii {

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

/**
 * Check if a file exists
 * @return true if and only if the file exists, false else
 */
bool fileExists(const char* file) {
    struct stat buf;
    return (stat(file, &buf) == 0);
}

/* Static Factory Implementations */
Volume* Volume::createFromFile(std::string name, std::string path) {
    auto create = [path] (Volume* v) {
        if (!fileExists(path.c_str())) {
            throw std::runtime_error(std::string("Error: file does not exist ") + path);
        }

        // first, check the extension
        std::string extension = std::string(strrchr(path.c_str(), '.'));
        std::transform(extension.data(), extension.data() + extension.size(), 
        std::addressof(extension[0]), [](unsigned char c){ return std::tolower(c); });

        if (extension.compare(".nvdb") == 0) {
            auto list = nanovdb::io::readGridMetaData(path);
            std::cerr << "Opened file " << path << std::endl;
            std::cerr << "    grids:" << std::endl;
            for (auto& m : list) {
                std::cerr << "        " << m.gridName << std::endl;
            }
            assert( list.size() > 0 );

            nanovdb::GridHandle<> gridHdl;
            if (list[0].gridName.length() > 0)
                gridHdl = nanovdb::io::readGrid<>(path, list[0].gridName);
            else
                gridHdl = nanovdb::io::readGrid<>(path);
            
            if (!gridHdl) {
                throw std::runtime_error("Error: unable to read nvdb grid!");
            }

            v->gridHdlPtr = std::make_shared<nanovdb::GridHandle<>>(std::move(gridHdl));
        }
        else {
            throw std::runtime_error(std::string("Error: unsupported format ") + 
                extension);
        }
        v->markDirty();
    };

    try {
        return StaticFactory::create<Volume>(editMutex, name, "Volume", lookupTable, volumes.data(), volumes.size(), create);
    } catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Volume", lookupTable, volumes.data(), volumes.size());
		throw;
	}
}

Volume *Volume::createSphere(std::string name)
{
    auto create = [] (Volume* v) {
        nanovdb::GridHandle<> gridHdl = nanovdb::createFogVolumeSphere();
        v->gridHdlPtr = std::make_shared<nanovdb::GridHandle<>>(std::move(gridHdl));
        v->markDirty();
    };

    try {
        return StaticFactory::create<Volume>(editMutex, name, "Volume", lookupTable, volumes.data(), volumes.size(), create);
    } catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Volume", lookupTable, volumes.data(), volumes.size());
		throw;
	}
}

Volume *Volume::createTorus(std::string name)
{
    auto create = [] (Volume* v) {
        nanovdb::GridHandle<> gridHdl = nanovdb::createFogVolumeTorus();
        v->gridHdlPtr = std::make_shared<nanovdb::GridHandle<>>(std::move(gridHdl));
        v->markDirty();
    };

    try {
        return StaticFactory::create<Volume>(editMutex, name, "Volume", lookupTable, volumes.data(), volumes.size(), create);
    } catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Volume", lookupTable, volumes.data(), volumes.size());
		throw;
	}
}

Volume *Volume::createBox(std::string name)
{
    auto create = [] (Volume* v) {
        nanovdb::GridHandle<> gridHdl = nanovdb::createFogVolumeBox();
        v->gridHdlPtr = std::make_shared<nanovdb::GridHandle<>>(std::move(gridHdl));
        v->markDirty();
    };

    try {
        return StaticFactory::create<Volume>(editMutex, name, "Volume", lookupTable, volumes.data(), volumes.size(), create);
    } catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Volume", lookupTable, volumes.data(), volumes.size());
		throw;
	}
}

Volume *Volume::createOctahedron(std::string name)
{
    auto create = [] (Volume* v) {
        nanovdb::GridHandle<> gridHdl = nanovdb::createFogVolumeOctahedron();
        v->gridHdlPtr = std::make_shared<nanovdb::GridHandle<>>(std::move(gridHdl));
        v->markDirty();
    };

    try {
        return StaticFactory::create<Volume>(editMutex, name, "Volume", lookupTable, volumes.data(), volumes.size(), create);
    } catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Volume", lookupTable, volumes.data(), volumes.size());
		throw;
	}
}

Volume *Volume::createFromData(
    std::string name, 
    uint32_t width, 
    uint32_t height, 
    uint32_t depth, 
    const float* data, 
    uint32_t length,
    float background
)
{
    if (length != (width * height * depth)) { throw std::runtime_error("Error: width * height * depth does not equal length of data!"); }
    if (width == 0) { throw std::runtime_error("Error: width must be greater than 0!"); }
    if (height == 0) { throw std::runtime_error("Error: height must be greater than 0!"); }
    if (depth == 0) { throw std::runtime_error("Error: depth must be greater than 0!"); }

    auto create = [width, height, depth, data, background] (Volume* v) {
        nanovdb::GridBuilder<float> builder(background);
        auto acc = builder.getAccessor();

        for (uint32_t z = 0; z < depth; ++z) {
            for (uint32_t y = 0; y < height; ++y) {
                for (uint32_t x = 0; x < width; ++x) {
                    acc.setValue(nanovdb::Coord(x + 1, y + 1, z + 1), data[x + y * width + z * width * height]);
                }
            }
        }

        nanovdb::GridHandle<> gridHdl = builder.getHandle<>();
        v->gridHdlPtr = std::make_shared<nanovdb::GridHandle<>>(std::move(gridHdl));
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

std::string Volume::getGridType()
{
    const nanovdb::GridMetaData* metadata = gridHdlPtr.get()->gridMetaData();
    auto type = metadata->gridType();
    switch(type) {
        case nanovdb::GridType::Float : return "float";
        case nanovdb::GridType::Double : return "double";
        case nanovdb::GridType::Int16 : return "int16";
        case nanovdb::GridType::Int32 : return "int32";
        case nanovdb::GridType::Int64 : return "int64";
        case nanovdb::GridType::Vec3f : return "vec3f";
        case nanovdb::GridType::Vec3d : return "vec3d";
        case nanovdb::GridType::Mask : return "mask";
        case nanovdb::GridType::FP16 : return "fp16";
        case nanovdb::GridType::UInt32 : return "uint32";
        default : return "unknown";
    };
}
 
uint32_t Volume::getNodeCount(uint32_t level)
{
    const nanovdb::GridMetaData* metadata = gridHdlPtr.get()->gridMetaData();
    if (metadata->gridType() != nanovdb::GridType::Float) 
        throw std::runtime_error("Error, unsupported grid format!");
    nanovdb::FloatGrid* gridPtr = reinterpret_cast<nanovdb::FloatGrid*>(gridHdlPtr.get()->data());
    return gridPtr->tree().nodeCount(level);
}

glm::vec3 Volume::getMinAabbCorner(uint32_t level, uint32_t node_idx)
{
    const nanovdb::GridMetaData* metadata = gridHdlPtr.get()->gridMetaData();
    if (metadata->gridType() != nanovdb::GridType::Float) 
        throw std::runtime_error("Error, unsupported grid format!");
    nanovdb::FloatGrid* gridPtr = 
        reinterpret_cast<nanovdb::FloatGrid*>(gridHdlPtr.get()->data());
    auto &tree = gridPtr->tree();
    auto root = tree.getNode<3>(0);
    auto mx = root->bbox().max();
    auto mn = root->bbox().min();
    glm::vec3 offset = glm::vec3(mn[0], mn[1], mn[2]) + 
                (glm::vec3(mx[0], mx[1], mx[2]) - 
                glm::vec3(mn[0], mn[1], mn[2])) * .5f;
    if (level == 0) {
        auto node = tree.getNode<0>(node_idx);
        auto m = node->bbox().min();
        return glm::vec3(m[0], m[1], m[2])-offset;
    }
    if (level == 1) {
        auto node = tree.getNode<1>(node_idx);
        auto m = node->bbox().min();
        return glm::vec3(m[0], m[1], m[2])-offset;
    }
    if (level == 2) {
        auto node = tree.getNode<2>(node_idx);
        auto m = node->bbox().min();
        return glm::vec3(m[0], m[1], m[2])-offset;
    }
    if (level == 3) {
        auto node = tree.getNode<3>(node_idx);
        auto m = node->bbox().min();
        return glm::vec3(m[0], m[1], m[2])-offset;
    }    
    return glm::vec3(NAN);
}

glm::vec3 Volume::getMaxAabbCorner(uint32_t level, uint32_t node_idx)
{
    const nanovdb::GridMetaData* metadata = gridHdlPtr.get()->gridMetaData();
    if (metadata->gridType() != nanovdb::GridType::Float) 
        throw std::runtime_error("Error, unsupported grid format!");
    nanovdb::FloatGrid* gridPtr = 
        reinterpret_cast<nanovdb::FloatGrid*>(gridHdlPtr.get()->data());
    auto &tree = gridPtr->tree();
    auto root = tree.getNode<3>(0);
    auto mx = root->bbox().max();
    auto mn = root->bbox().min();
    glm::vec3 offset = glm::vec3(mn[0], mn[1], mn[2]) + 
                (glm::vec3(mx[0], mx[1], mx[2]) - 
                glm::vec3(mn[0], mn[1], mn[2])) * .5f;
    if (level == 0) {
        auto node = tree.getNode<0>(node_idx);
        auto m = node->bbox().max();
        return glm::vec3(m[0], m[1], m[2])-offset;
    }
    if (level == 1) {
        auto node = tree.getNode<1>(node_idx);
        auto m = node->bbox().max();
        return glm::vec3(m[0], m[1], m[2])-offset;
    }
    if (level == 2) {
        auto node = tree.getNode<2>(node_idx);
        auto m = node->bbox().max();
        return glm::vec3(m[0], m[1], m[2])-offset;
    }
    if (level == 3) {
        auto node = tree.getNode<3>(node_idx);
        auto m = node->bbox().max();
        return glm::vec3(m[0], m[1], m[2])-offset;
    }    
    return glm::vec3(NAN);
}

glm::vec3 Volume::getAabbCenter(uint32_t level, uint32_t node_idx)
{
    nanovdb::FloatGrid* gridPtr = 
        reinterpret_cast<nanovdb::FloatGrid*>(gridHdlPtr.get()->data());
    auto &tree = gridPtr->tree();
    auto root = tree.getNode<3>(0);
    auto mx = root->bbox().max();
    auto mn = root->bbox().min();
    glm::vec3 offset = glm::vec3(mn[0], mn[1], mn[2]) + 
                (glm::vec3(mx[0], mx[1], mx[2]) - 
                glm::vec3(mn[0], mn[1], mn[2])) * .5f;

    return -offset + getMinAabbCorner(level, node_idx) + 
        (getMaxAabbCorner(level, node_idx) - 
        getMinAabbCorner(level, node_idx)) * .5f;
}

float Volume::getMax(uint32_t level, uint32_t node_idx)
{
    const nanovdb::GridMetaData* metadata = gridHdlPtr.get()->gridMetaData();
    if (metadata->gridType() != nanovdb::GridType::Float) 
        throw std::runtime_error("Error, unsupported grid format!");
    nanovdb::FloatGrid* gridPtr = 
        reinterpret_cast<nanovdb::FloatGrid*>(gridHdlPtr.get()->data());
    
    auto &tree = gridPtr->tree();
    if (level == 0) {
        auto node = tree.getNode<0>(node_idx);
        auto m = node->valueMax();
        return m;
    }
    if (level == 1) {
        auto node = tree.getNode<1>(node_idx);
        auto m = node->valueMax();
        return m;
    }
    if (level == 2) {
        auto node = tree.getNode<2>(node_idx);
        auto m = node->valueMax();
        return m;
    }
    if (level == 3) {
        auto node = tree.getNode<3>(node_idx);
        auto m = node->valueMax();
        return m;
    }    
    return NAN;
}


std::shared_ptr<nanovdb::GridHandle<>> Volume::getNanoVDBGridHandle()
{
    return gridHdlPtr;
}

void Volume::setScale(float units)
{
    this->volumeStructs[id].scale = units;
    markDirty();
}

void Volume::setScattering(float scattering)
{
    this->volumeStructs[id].scattering = scattering;
    markDirty();
}

void Volume::setAbsorption(float absorption)
{
    this->volumeStructs[id].absorption = absorption;
    markDirty();
}

void Volume::setGradientFactor(float factor)
{
    this->volumeStructs[id].gradient_factor = factor;
    markDirty();
}

};
