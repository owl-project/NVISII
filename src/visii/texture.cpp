#include <visii/texture.h>

#include <stb_image.h>
#include <stb_image_write.h>
#include <cstring>

std::vector<Texture> Texture::textures;
std::vector<TextureStruct> Texture::textureStructs;
std::map<std::string, uint32_t> Texture::lookupTable;
std::shared_ptr<std::recursive_mutex> Texture::editMutex;
bool Texture::factoryInitialized = false;
bool Texture::anyDirty = true;

Texture::Texture()
{
    this->initialized = false;
    markDirty();
}

Texture::~Texture()
{
    std::vector<glm::vec4>().swap(this->texels);
}

Texture::Texture(std::string name, uint32_t id)
{
    this->initialized = true;
    this->name = name;
    this->id = id;

    // ------------------------------------------------------------------
    // create a 4x4 checkerboard texture
    // ------------------------------------------------------------------
    glm::ivec2 texSize = glm::ivec2(4);
    std::vector<vec4> texels;
    for (int iy=0;iy<texSize.y;iy++) {
        for (int ix=0;ix<texSize.x;ix++) {
            texels.push_back(((ix ^ iy)&1) ?
                            vec4(0,1,0,0) :
                            vec4(1));
        }
    }

    textureStructs[id].width = 4;
    textureStructs[id].height = 4;
    this->texels = texels;
    
    markDirty();
}

std::string Texture::toString() {
    std::string output;
    output += "{\n";
    output += "\ttype: \"Texture\",\n";
    output += "\tname: \"" + name + "\"\n";
    output += "}";
    return output;
}

std::vector<vec4> Texture::getTexels() {
    return texels;
}

uint32_t Texture::getWidth() {
    return textureStructs[id].width;
}

uint32_t Texture::getHeight() {
    return textureStructs[id].height;
}

/* SSBO logic */
void Texture::initializeFactory()
{
    if (isFactoryInitialized()) return;
    textures.resize(32768);
    textureStructs.resize(32768);
    editMutex = std::make_shared<std::recursive_mutex>();
    factoryInitialized = true;
}

bool Texture::isFactoryInitialized()
{
    return factoryInitialized;
}

bool Texture::isInitialized()
{
	return initialized;
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

	for (int i = 0; i < textures.size(); ++i) {
		if (textures[i].isDirty()) {
            textures[i].markClean();
        }
	};
	anyDirty = false;
} 

void Texture::clearAll()
{
    if (!isFactoryInitialized()) return;

    for (auto &texture : textures) {
		if (texture.initialized) {
			Texture::remove(texture.name);
		}
	}
}	

/* Static Factory Implementations */
Texture* Texture::create(std::string name) {
    auto l = StaticFactory::create(editMutex, name, "Texture", lookupTable, textures.data(), textures.size());
    anyDirty = true;
    return l;
}

Texture* Texture::createFromImage(std::string name, std::string path, bool linear) {
    static bool createFromImageDeprecatedShown = false;
    if (createFromImageDeprecatedShown == false) {
        std::cout<<"Warning, create_from_image is deprecated and will be removed in a subsequent release. Please switch to create_from_file." << std::endl;
        createFromImageDeprecatedShown = true;
    }
    return createFromFile(name, path, linear);
}

Texture* Texture::createFromFile(std::string name, std::string path, bool linear) {
    auto create = [path, linear] (Texture* l) {
        int x, y, num_channels;
        stbi_set_flip_vertically_on_load(true);
        if (linear) {
            stbi_ldr_to_hdr_gamma(1.0f);
        } else {
            stbi_ldr_to_hdr_gamma(2.2f);
        }
        float* pixels = stbi_loadf(path.c_str(), &x, &y, &num_channels, STBI_rgb_alpha);
        if (!pixels) { 
            std::string reason (stbi_failure_reason());
            throw std::runtime_error(std::string("Error: failed to load texture image \"") + path + std::string("\". Reason: ") + reason); 
        }
        l->texels.resize(x * y);
        memcpy(l->texels.data(), pixels, x * y * 4 * sizeof(float));
        textureStructs[l->getId()].width = x;
        textureStructs[l->getId()].height = y;
        l->markDirty();
        stbi_image_free(pixels);
    };

    try {
        return StaticFactory::create<Texture>(editMutex, name, "Texture", lookupTable, textures.data(), textures.size(), create);
    } catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Texture", lookupTable, textures.data(), textures.size());
		throw;
	}
}

Texture* Texture::createFromData(std::string name, uint32_t width, uint32_t height, const float* data, uint32_t length)
{
    if (length != (width * height * 4)) { throw std::runtime_error("Error: width * height * 4 does not equal length of data!"); }
    if (width == 0) { throw std::runtime_error("Error: width must be greater than 0!"); }
    if (height == 0) { throw std::runtime_error("Error: height must be greater than 0!"); }

    auto create = [width, height, length, data] (Texture* l) {
        l->texels.resize(width * height);
        memcpy(l->texels.data(), data, width * height * 4 * sizeof(float));
        textureStructs[l->getId()].width = width;
        textureStructs[l->getId()].height = height;
        l->markDirty();
    };

    try {
        return StaticFactory::create<Texture>(editMutex, name, "Texture", lookupTable, textures.data(), textures.size(), create);
    } catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Texture", lookupTable, textures.data(), textures.size());
		throw;
	}
}

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.b, c.g, K.w, K.z), vec4(c.g, c.b, K.x, K.y), step(c.b, c.g));
    vec4 q = mix(vec4(p.x, p.y, p.w, c.r), vec4(c.r, p.y, p.z, p.x), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10f;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0f, 2.0f / 3.0f, 1.0f / 3.0f, 3.0f);
    vec3 p = abs(fract(vec3(c.x) + vec3(K)) * 6.0f - vec3(K.w));
    return c.z * mix(vec3(K.x), clamp(p - vec3(K.x), 0.0f, 1.0f), c.y);
}

Texture* Texture::createHSV(std::string name, Texture* tex, float hue, float sat, float val, float alpha)
{
    auto create = [tex, hue, sat, val, alpha] (Texture* l) {
        if (!tex || !tex->isInitialized()) throw std::runtime_error(std::string("Error: input texture is null/uninitialized!")); 

        uint32_t width = tex->getWidth();
        uint32_t height = tex->getHeight();
        l->texels.resize(width * height);
        textureStructs[l->getId()].width = width;
        textureStructs[l->getId()].height = height;

        float dh = (hue * 2.f) - 1.0f;
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                vec2 off = vec2(1 / float(width), 1 / float(height)); 
                vec2 uv = vec2(x / float(width), y / float(height)) + .5f * off;
                vec4 c = tex->sample(uv);
                vec3 rgb = vec3(c);
                vec3 hsv = rgb2hsv(rgb);
                hsv.x = (hsv.x + dh) - (long)(hsv.x + dh);
                hsv.y = clamp(sat * hsv.y, 0.f, 1.f); hsv.z = clamp(hsv.z * val, 0.f, 1.f);
                rgb = mix(rgb, hsv2rgb(hsv), alpha);
                l->texels[y * width + x] = vec4( rgb.r, rgb.g, rgb.b, c.a); // todo, transform in HSV space...
            }
        }
        
        l->markDirty();
    };

    try {
        return StaticFactory::create<Texture>(editMutex, name, "Texture", lookupTable, textures.data(), textures.size(), create);
    } catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Texture", lookupTable, textures.data(), textures.size());
		throw;
	}
}

Texture* Texture::createMix(std::string name, Texture* a, Texture* b, float mix)
{
    auto create = [a, b, mix] (Texture* l) {
        if (!a || !a->isInitialized()) throw std::runtime_error(std::string("Error: Texture A is null/uninitialized!")); 
        if (!b || !b->isInitialized()) throw std::runtime_error(std::string("Error: Texture B is null/uninitialized!")); 

        uint32_t width = ::max(a->getWidth(), b->getWidth());
        uint32_t height = ::max(a->getHeight(), b->getHeight());
        l->texels.resize(width * height);
        textureStructs[l->getId()].width = width;
        textureStructs[l->getId()].height = height;

        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                vec2 off = vec2(1 / float(width), 1 / float(height)); 
                vec2 uv = vec2(x / float(width), y / float(height)) + .5f * off;
                l->texels[y * width + x] = glm::mix(a->sample(uv), b->sample(uv), mix);
            }
        }
        
        l->markDirty();
    };

    try {
        return StaticFactory::create<Texture>(editMutex, name, "Texture", lookupTable, textures.data(), textures.size(), create);
    } catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Texture", lookupTable, textures.data(), textures.size());
		throw;
	}
}

Texture* Texture::createAdd(std::string name, Texture* a, Texture* b)
{
    auto create = [a, b] (Texture* l) {
        if (!a || !a->isInitialized()) throw std::runtime_error(std::string("Error: Texture A is null/uninitialized!")); 
        if (!b || !b->isInitialized()) throw std::runtime_error(std::string("Error: Texture B is null/uninitialized!")); 

        uint32_t width = ::max(a->getWidth(), b->getWidth());
        uint32_t height = ::max(a->getHeight(), b->getHeight());
        l->texels.resize(width * height);
        textureStructs[l->getId()].width = width;
        textureStructs[l->getId()].height = height;

        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                vec2 off = vec2(1 / float(width), 1 / float(height)); 
                vec2 uv = vec2(x / float(width), y / float(height)) + .5f * off;
                l->texels[y * width + x] = a->sample(uv) + b->sample(uv);
            }
        }
        
        l->markDirty();
    };

    try {
        return StaticFactory::create<Texture>(editMutex, name, "Texture", lookupTable, textures.data(), textures.size(), create);
    } catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Texture", lookupTable, textures.data(), textures.size());
		throw;
	}
}

Texture* Texture::createMultiply(std::string name, Texture* a, Texture* b)
{
    auto create = [a, b] (Texture* l) {
        if (!a || !a->isInitialized()) throw std::runtime_error(std::string("Error: Texture A is null/uninitialized!")); 
        if (!b || !b->isInitialized()) throw std::runtime_error(std::string("Error: Texture B is null/uninitialized!")); 

        uint32_t width = ::max(a->getWidth(), b->getWidth());
        uint32_t height = ::max(a->getHeight(), b->getHeight());
        l->texels.resize(width * height);
        textureStructs[l->getId()].width = width;
        textureStructs[l->getId()].height = height;

        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                vec2 off = vec2(1 / float(width), 1 / float(height)); 
                vec2 uv = vec2(x / float(width), y / float(height)) + .5f * off;
                l->texels[y * width + x] = a->sample(uv) * b->sample(uv);
            }
        }
        
        l->markDirty();
    };

    try {
        return StaticFactory::create<Texture>(editMutex, name, "Texture", lookupTable, textures.data(), textures.size(), create);
    } catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Texture", lookupTable, textures.data(), textures.size());
		throw;
	}
}

vec4 Texture::sample(vec2 uv) {
    uint32_t width = textureStructs[id].width;
    uint32_t height = textureStructs[id].height;
    vec2 coord = uv * vec2(width-1, height-1);
    ivec2 coord_floor = glm::ivec2(glm::floor(coord));
    ivec2 coord_ceil = glm::ivec2(glm::ceil(coord));
    return texels[coord_floor.y * width + coord_floor.x]; // todo, interpolate four surrouding pixels
    // vec2 remainder = coord - vec2(coord_floor);
    // return glm::mix(
    //     texels[coord_floor.y * width + coord_floor.x],
    //     texels[coord_ceil.y * width + coord_ceil.x],
    //     remainder);
}

std::shared_ptr<std::recursive_mutex> Texture::getEditMutex()
{
	return editMutex;
}

Texture* Texture::get(std::string name) {
    return StaticFactory::get(editMutex, name, "Texture", lookupTable, textures.data(), textures.size());
}

void Texture::remove(std::string name) {
    auto t = get(name);
	if (!t) return;
    std::vector<glm::vec4>().swap(t->texels);
    StaticFactory::remove(editMutex, name, "Texture", lookupTable, textures.data(), textures.size());
    anyDirty = true;
}

Texture* Texture::getFront() {
    return textures.data();
}

TextureStruct* Texture::getFrontStruct() {
    return textureStructs.data();
}

uint32_t Texture::getCount() {
    return textures.size();
}

std::string Texture::getName()
{
    return name;
}

std::map<std::string, uint32_t> Texture::getNameToIdMap()
{
	return lookupTable;
}