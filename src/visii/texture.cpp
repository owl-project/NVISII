#include <visii/texture.h>

#include <stb_image.h>
#include <stb_image_write.h>
#include <cstring>

#include <algorithm>

#include <gli/gli.hpp>
#include <gli/convert.hpp>
#include <gli/core/s3tc.hpp>

#include <glm/gtc/color_space.hpp>
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
    std::vector<glm::vec4>().swap(this->floatTexels);
    std::vector<glm::u8vec4>().swap(this->byteTexels);
}

Texture::Texture(std::string name, uint32_t id)
{
    this->initialized = true;
    this->name = name;
    this->id = id;

    textureStructs[id].width = -1;
    textureStructs[id].height = -1;
    this->floatTexels = std::vector<vec4>();
    this->byteTexels = std::vector<u8vec4>();
    
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

std::vector<vec4> Texture::getFloatTexels() {
    // If natively represented as 32f, return that. 
    // otherwise, cast 8uc to 32f.
    if (floatTexels.size() > 0) return floatTexels;
    std::vector<vec4> floatTexels(byteTexels.size());
    for (uint32_t i = 0; i < byteTexels.size(); ++i) {
        floatTexels[i] = vec4(byteTexels[i]) / 255.0f;
    }
    return floatTexels;
}

std::vector<u8vec4> Texture::getByteTexels() {
    // If natively represented as 8uc, return that. 
    // otherwise, cast 32f to 8uc.
    if (byteTexels.size() > 0) return byteTexels;
    std::vector<u8vec4> texels8(floatTexels.size());
    for (uint32_t i = 0; i < floatTexels.size(); ++i) {
        texels8[i] = u8vec4(floatTexels[i] * 255.0f);
    }
    return texels8;
}

uint32_t Texture::getWidth() {
    return textureStructs[id].width;
}

uint32_t Texture::getHeight() {
    return textureStructs[id].height;
}

void Texture::setScale(glm::vec2 scale)
{
    textureStructs[id].scale = scale;
    markDirty();
}

bool Texture::isHDR()
{
    // if the texture is natively represented as a 32 bit-per-channel texture, it's HDR.
    return (floatTexels.size() > 0);
}

bool Texture::isLinear() {
    return linear;
}

/* SSBO logic */
void Texture::initializeFactory(uint32_t max_components)
{
    if (isFactoryInitialized()) return;
    textures.resize(max_components);
    textureStructs.resize(max_components);
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
        // first, check the extension
        std::string extension = std::string(strrchr(path.c_str(), '.'));
        std::transform(extension.data(), extension.data() + extension.size(), 
            std::addressof(extension[0]), [](unsigned char c){ return std::tolower(c); });
        
        if ((extension.compare(".dds") == 0) || (extension.compare(".ktx") == 0)) {
            auto texture = gli::load(path);
            if (texture.target() != gli::target::TARGET_2D) {
                std::string reason = "Currently only 2D textures supported!";
                throw std::runtime_error(std::string("Error: failed to load texture image \"") + 
                    path + std::string("\". Reason: ") + reason); 
            }

            auto tex2D = gli::texture2d(texture);
            auto format = tex2D.format();
            if (tex2D.empty())
                throw std::runtime_error( std::string("Error: image " + path + " is empty"));

            // gli detects whether or not a texture is srgb. Ignore "linear" parameter above.
            l->linear = (!gli::is_srgb(format));

            if (gli::is_compressed(format)) {
                if ((format != gli::FORMAT_RGBA_DXT1_UNORM_BLOCK8) &&
                    (format != gli::FORMAT_RGBA_DXT5_UNORM_BLOCK16) &&
                    (format != gli::FORMAT_R_ATI1N_UNORM_BLOCK8) &&
                    (format != gli::FORMAT_RG_ATI2N_UNORM_BLOCK16)
                )
                throw std::runtime_error(std::string("Error: image " + path + " is compressed using an unsupported S3TC format. " + 
                        "Supported formats are " + 
                        "FORMAT_RGBA32_SFLOAT_PACK32, " +
                        "FORMAT_RGBA8_SRGB_PACK8, " +
                        "FORMAT_R32_SFLOAT_PACK32, " +
                        "FORMAT_R8_SRGB_PACK8, " +
                        "FORMAT_RG32_SFLOAT_PACK32, " +
                        "FORMAT_RG8_SRGB_PACK8" + 
                        "FORMAT_RGBA_DXT1_UNORM_BLOCK8, " +
                        "FORMAT_RGBA_DXT5_UNORM_BLOCK16, " +
                        "FORMAT_R_ATI1N_UNORM_BLOCK8, " +
                        "FORMAT_RG_ATI2N_UNORM_BLOCK16")); 

                // decompress RGBA DXT1
                gli::texture2d &TextureCompressed = tex2D;
                gli::texture2d TextureLocalDecompressed(
                    gli::FORMAT_RGBA32_SFLOAT_PACK32, 
                    TextureCompressed.extent(), 
                    TextureCompressed.levels(), 
                    TextureCompressed.swizzles());
                gli::extent2d BlockExtent;
                {
                    gli::extent3d TempExtent = gli::block_extent(format);
                    BlockExtent.x = TempExtent.x;
                    BlockExtent.y = TempExtent.y;
                }

                for(size_t Level = 0; Level < TextureCompressed.levels(); ++Level) {
                    gli::extent2d TexelCoord, BlockCoord;
                    gli::extent2d LevelExtent = TextureCompressed.extent(Level);
                    gli::extent2d LevelExtentInBlocks = glm::max(gli::extent2d(1, 1), LevelExtent / BlockExtent);
                    gli::extent2d DecompressedBlockCoord;
                    for(BlockCoord.y = 0, TexelCoord.y = 0; BlockCoord.y < LevelExtentInBlocks.y; ++BlockCoord.y, TexelCoord.y += BlockExtent.y) {
                        for(BlockCoord.x = 0, TexelCoord.x = 0; BlockCoord.x < LevelExtentInBlocks.x; ++BlockCoord.x, TexelCoord.x += BlockExtent.x) {
                            if (format == gli::FORMAT_RGBA_DXT1_UNORM_BLOCK8) {
                                const gli::detail::dxt1_block *DXT1Block = TextureCompressed.data<gli::detail::dxt1_block>(0, 0, Level) + (BlockCoord.y * LevelExtentInBlocks.x + BlockCoord.x);
                                const gli::detail::texel_block4x4 DecompressedBlock = gli::detail::decompress_dxt1_block(*DXT1Block);
                                for(DecompressedBlockCoord.y = 0; DecompressedBlockCoord.y < glm::min(4, LevelExtent.y); ++DecompressedBlockCoord.y) {
                                    for(DecompressedBlockCoord.x = 0; DecompressedBlockCoord.x < glm::min(4, LevelExtent.x); ++DecompressedBlockCoord.x) {
                                        TextureLocalDecompressed.store(TexelCoord + DecompressedBlockCoord, Level, DecompressedBlock.Texel[DecompressedBlockCoord.y][DecompressedBlockCoord.x]);
                                    }
                                }
                            }
                            else if (format == gli::FORMAT_RGBA_DXT5_UNORM_BLOCK16) {
                                const gli::detail::dxt5_block *DXT5Block = TextureCompressed.data<gli::detail::dxt5_block>(0, 0, Level) + (BlockCoord.y * LevelExtentInBlocks.x + BlockCoord.x);
                                const gli::detail::texel_block4x4 DecompressedBlock = gli::detail::decompress_dxt5_block(*DXT5Block);
                                for(DecompressedBlockCoord.y = 0; DecompressedBlockCoord.y < glm::min(4, LevelExtent.y); ++DecompressedBlockCoord.y) {
                                    for(DecompressedBlockCoord.x = 0; DecompressedBlockCoord.x < glm::min(4, LevelExtent.x); ++DecompressedBlockCoord.x) {
                                        TextureLocalDecompressed.store(TexelCoord + DecompressedBlockCoord, Level, DecompressedBlock.Texel[DecompressedBlockCoord.y][DecompressedBlockCoord.x]);
                                    }
                                }
                            }
                            else if (format == gli::FORMAT_R_ATI1N_UNORM_BLOCK8) {
                                const gli::detail::bc4_block *BC4Block = TextureCompressed.data<gli::detail::bc4_block>(0, 0, Level) + (BlockCoord.y * LevelExtentInBlocks.x + BlockCoord.x);
						        const gli::detail::texel_block4x4 DecompressedBlock = gli::detail::decompress_bc4unorm_block(*BC4Block);
                                for(DecompressedBlockCoord.y = 0; DecompressedBlockCoord.y < glm::min(4, LevelExtent.y); ++DecompressedBlockCoord.y) {
                                    for(DecompressedBlockCoord.x = 0; DecompressedBlockCoord.x < glm::min(4, LevelExtent.x); ++DecompressedBlockCoord.x) {
                                        TextureLocalDecompressed.store(TexelCoord + DecompressedBlockCoord, Level, DecompressedBlock.Texel[DecompressedBlockCoord.y][DecompressedBlockCoord.x]);
                                    }
                                }
                            }
                            else if (format == gli::FORMAT_RG_ATI2N_UNORM_BLOCK16) {
                                const gli::detail::bc5_block *BC5Block = TextureCompressed.data<gli::detail::bc5_block>(0, 0, Level) + (BlockCoord.y * LevelExtentInBlocks.x + BlockCoord.x);
                                const gli::detail::texel_block4x4 DecompressedBlock = gli::detail::decompress_bc5unorm_block(*BC5Block);
                                for(DecompressedBlockCoord.y = 0; DecompressedBlockCoord.y < glm::min(4, LevelExtent.y); ++DecompressedBlockCoord.y) {
                                    for(DecompressedBlockCoord.x = 0; DecompressedBlockCoord.x < glm::min(4, LevelExtent.x); ++DecompressedBlockCoord.x) {
                                        TextureLocalDecompressed.store(TexelCoord + DecompressedBlockCoord, Level, DecompressedBlock.Texel[DecompressedBlockCoord.y][DecompressedBlockCoord.x]);
                                    }
                                }
                            }
                        }
                    }
                }
                
                TextureLocalDecompressed = gli::flip(TextureLocalDecompressed);
                
                int lvl = 0;
                textureStructs[l->getId()].width = (uint32_t)(TextureLocalDecompressed.extent(lvl).x);
                textureStructs[l->getId()].height = (uint32_t)(TextureLocalDecompressed.extent(lvl).y);
                auto image = TextureLocalDecompressed[lvl]; // get mipmap 0
                if (gli::is_float(format)) {
                    l->floatTexels.resize(textureStructs[l->getId()].width * textureStructs[l->getId()].height);
                    memcpy(l->floatTexels.data(), image.data(), (uint32_t)image.size());
                } else {
                    l->byteTexels.resize(textureStructs[l->getId()].width * textureStructs[l->getId()].height);
                    std::vector<vec4> temp(textureStructs[l->getId()].width * textureStructs[l->getId()].height);
                    memcpy(temp.data(), image.data(), (uint32_t)image.size());
                    for (uint32_t i = 0; i < temp.size(); ++i) l->byteTexels[i] = u8vec4(temp[i] * 255.f);
                }
                l->markDirty();                
            }
            else {
                tex2D = gli::flip(tex2D);
                textureStructs[l->getId()].width = (uint32_t)(tex2D.extent().x);
                textureStructs[l->getId()].height = (uint32_t)(tex2D.extent().y);
                auto image = tex2D[0]; // get mipmap 0
                if (format == gli::FORMAT_RGBA32_SFLOAT_PACK32) {
                    l->floatTexels.resize(textureStructs[l->getId()].width * textureStructs[l->getId()].height);
                    memcpy(l->floatTexels.data(), image.data(), (uint32_t)image.size());
                }
                else if (format == gli::FORMAT_RGBA8_SRGB_PACK8) {
                    l->byteTexels.resize(textureStructs[l->getId()].width * textureStructs[l->getId()].height);
                    memcpy(l->byteTexels.data(), image.data(), (uint32_t)image.size());
                }
                if (format == gli::FORMAT_R32_SFLOAT_PACK32) {
                    tex2D = gli::convert(tex2D, gli::format::FORMAT_RGBA32_SFLOAT_PACK32);
                    l->floatTexels.resize(textureStructs[l->getId()].width * textureStructs[l->getId()].height);
                    memcpy(l->floatTexels.data(), image.data(), (uint32_t)image.size());
                }
                else if (format == gli::FORMAT_R8_SRGB_PACK8) {
                    tex2D = gli::convert(tex2D, gli::format::FORMAT_RGBA8_SRGB_PACK8);
                    l->byteTexels.resize(textureStructs[l->getId()].width * textureStructs[l->getId()].height);
                    memcpy(l->byteTexels.data(), image.data(), (uint32_t)image.size());
                }
                if (format == gli::FORMAT_RG32_SFLOAT_PACK32) {
                    tex2D = gli::convert(tex2D, gli::format::FORMAT_RGBA32_SFLOAT_PACK32);
                    l->floatTexels.resize(textureStructs[l->getId()].width * textureStructs[l->getId()].height);
                    memcpy(l->floatTexels.data(), image.data(), (uint32_t)image.size());
                }
                else if (format == gli::FORMAT_RG8_SRGB_PACK8) {
                    tex2D = gli::convert(tex2D, gli::format::FORMAT_RGBA8_SRGB_PACK8);
                    l->byteTexels.resize(textureStructs[l->getId()].width * textureStructs[l->getId()].height);
                    memcpy(l->byteTexels.data(), image.data(), (uint32_t)image.size());
                }
                else {
                    throw std::runtime_error(std::string("Error: image " + path + " uses an unsupported format. " + 
                        "Supported formats are " + 
                        "FORMAT_RGBA32_SFLOAT_PACK32, " +
                        "FORMAT_RGBA8_SRGB_PACK8, " +
                        "FORMAT_R32_SFLOAT_PACK32, " +
                        "FORMAT_R8_SRGB_PACK8, " +
                        "FORMAT_RG32_SFLOAT_PACK32, " +
                        "FORMAT_RG8_SRGB_PACK8" + 
                        "FORMAT_RGBA_DXT1_UNORM_BLOCK8, " +
                        "FORMAT_RGBA_DXT5_UNORM_BLOCK16, " +
                        "FORMAT_R_ATI1N_UNORM_BLOCK8, " +
                        "FORMAT_RG_ATI2N_UNORM_BLOCK16"));
                }
                l->markDirty();
            }
        }
        else {
            if (extension.compare(".hdr") == 0) {
                int x, y, num_channels;
                stbi_set_flip_vertically_on_load(true);
                l->linear = true; // Since we convert HDR images from srgb to linear, srgb is always false here.
                float* pixels = stbi_loadf(path.c_str(), &x, &y, &num_channels, STBI_rgb_alpha);
                if (!pixels) { 
                    std::string reason (stbi_failure_reason());
                    throw std::runtime_error(std::string("Error: failed to load texture image \"") + path + std::string("\". Reason: ") + reason); 
                }
                l->floatTexels.resize(x * y);
                memcpy(l->floatTexels.data(), pixels, x * y * 4 * sizeof(float));
                textureStructs[l->getId()].width = x;
                textureStructs[l->getId()].height = y;
                l->markDirty();
                stbi_image_free(pixels);
            }
            else {
                l->linear = linear; // if linear is true, treat the texture contents as if it were not sRGB.
                int x, y, num_channels;
                stbi_set_flip_vertically_on_load(true);
                stbi_uc* pixels = stbi_load(path.c_str(), &x, &y, &num_channels, STBI_rgb_alpha);
                if (!pixels) { 
                    std::string reason (stbi_failure_reason());
                    throw std::runtime_error(std::string("Error: failed to load texture image \"") + path + std::string("\". Reason: ") + reason); 
                }
                l->byteTexels.resize(x * y);
                memcpy(l->byteTexels.data(), pixels, x * y * 4 * sizeof(stbi_uc));
                textureStructs[l->getId()].width = x;
                textureStructs[l->getId()].height = y;
                l->markDirty();
                stbi_image_free(pixels);
            }
        }

    };

    try {
        return StaticFactory::create<Texture>(editMutex, name, "Texture", lookupTable, textures.data(), textures.size(), create);
    } catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Texture", lookupTable, textures.data(), textures.size());
		throw;
	}
}

Texture* Texture::createFromData(std::string name, uint32_t width, uint32_t height, const float* data, uint32_t length, bool linear, bool hdr)
{
    if (length != (width * height * 4)) { throw std::runtime_error("Error: width * height * 4 does not equal length of data!"); }
    if (width == 0) { throw std::runtime_error("Error: width must be greater than 0!"); }
    if (height == 0) { throw std::runtime_error("Error: height must be greater than 0!"); }

    auto create = [width, height, length, data, linear, hdr] (Texture* l) {
        // user must specify if texture should be srgb or linear. 
        // we default to linear here as SRGB is more common for images that are stored / loaded from disk.
        l->linear = linear; 
        if (hdr) {
            l->floatTexels.resize(width * height);
            memcpy(l->floatTexels.data(), data, width * height * 4 * sizeof(float));
        } else {
            // gotta convert from float to byte. 
            // TODO: update function signature to accept void* instead of just float*
            l->byteTexels.resize(width * height);
            for (uint32_t i = 0; i < length; ++i) {
                l->byteTexels[i] = u8vec4(vec4(
                    data[i * 4 + 0], 
                    data[i * 4 + 1], 
                    data[i * 4 + 2], 
                    data[i * 4 + 3]) * 255.f);
            }
        }
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

Texture* Texture::createHSV(std::string name, Texture* tex, float hue, float sat, float val, float alpha, bool hdr)
{
    auto create = [tex, hue, sat, val, alpha, hdr] (Texture* l) {
        if (!tex || !tex->isInitialized()) throw std::runtime_error(std::string("Error: input texture is null/uninitialized!")); 

        uint32_t width = tex->getWidth();
        uint32_t height = tex->getHeight();
        if (hdr) l->floatTexels.resize(width * height);
        else l->byteTexels.resize(width * height);
        textureStructs[l->getId()].width = width;
        textureStructs[l->getId()].height = height;

        float dh = (hue * 2.f) - 1.0f;
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                vec2 off = vec2(1 / float(width), 1 / float(height)); 
                vec2 uv = vec2(x / float(width), y / float(height)) + .5f * off;
                vec4 c = tex->sampleFloatTexels(uv);
                if (!tex->isLinear()) c = glm::convertSRGBToLinear(c);
                vec3 rgb = vec3(c);
                vec3 hsv = rgb2hsv(rgb);
                hsv.x = (hsv.x + dh) - (long)(hsv.x + dh);
                hsv.y = clamp(sat * hsv.y, 0.f, 1.f); hsv.z = clamp(hsv.z * val, 0.f, 1.f);
                rgb = mix(rgb, hsv2rgb(hsv), alpha);
                vec4 result = vec4( rgb.r, rgb.g, rgb.b, c.a);
                if (!tex->isLinear()) result = glm::convertLinearToSRGB(result);
                if (hdr) l->floatTexels[y * width + x] = vec4( rgb.r, rgb.g, rgb.b, c.a);
                else l->byteTexels[y * width + x] = u8vec4(vec4(rgb.r, rgb.g, rgb.b, c.a) * 255.f);
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

Texture* Texture::createMix(std::string name, Texture* a, Texture* b, float mix, bool hdr)
{
    auto create = [a, b, mix, hdr] (Texture* l) {
        if (!a || !a->isInitialized()) throw std::runtime_error(std::string("Error: Texture A is null/uninitialized!")); 
        if (!b || !b->isInitialized()) throw std::runtime_error(std::string("Error: Texture B is null/uninitialized!")); 

        uint32_t width = ::max(a->getWidth(), b->getWidth());
        uint32_t height = ::max(a->getHeight(), b->getHeight());
        if (hdr) l->floatTexels.resize(width * height);
        else l->byteTexels.resize(width * height);
        textureStructs[l->getId()].width = width;
        textureStructs[l->getId()].height = height;

        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                vec2 off = vec2(1 / float(width), 1 / float(height)); 
                vec2 uv = vec2(x / float(width), y / float(height)) + .5f * off;
                vec4 ac = a->sampleFloatTexels(uv);
                vec4 bc = b->sampleFloatTexels(uv);
                if (!a->isLinear()) ac = glm::convertSRGBToLinear(ac);
                if (!b->isLinear()) bc = glm::convertSRGBToLinear(bc);
                vec4 result = glm::mix(ac, bc, mix);
                if (!a->isLinear() && !b->isLinear()) result = glm::convertLinearToSRGB(result);
                if (hdr) l->floatTexels[y * width + x] = result;
                else l->byteTexels[y * width + x] = u8vec4(result * 255.f);
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

Texture* Texture::createAdd(std::string name, Texture* a, Texture* b, bool hdr)
{
    auto create = [a, b, hdr] (Texture* l) {
        if (!a || !a->isInitialized()) throw std::runtime_error(std::string("Error: Texture A is null/uninitialized!")); 
        if (!b || !b->isInitialized()) throw std::runtime_error(std::string("Error: Texture B is null/uninitialized!")); 

        uint32_t width = ::max(a->getWidth(), b->getWidth());
        uint32_t height = ::max(a->getHeight(), b->getHeight());
        if (hdr) l->floatTexels.resize(width * height);
        else l->byteTexels.resize(width * height);
        textureStructs[l->getId()].width = width;
        textureStructs[l->getId()].height = height;

        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                vec2 off = vec2(1 / float(width), 1 / float(height)); 
                vec2 uv = vec2(x / float(width), y / float(height)) + .5f * off;
                vec4 ac = a->sampleFloatTexels(uv);
                vec4 bc = b->sampleFloatTexels(uv);
                if (!a->isLinear()) ac = glm::convertSRGBToLinear(ac);
                if (!b->isLinear()) bc = glm::convertSRGBToLinear(bc);
                vec4 result = ac + bc;
                if (!a->isLinear() && !b->isLinear()) result = glm::convertLinearToSRGB(result);
                if (hdr) l->floatTexels[y * width + x] = result;
                else l->byteTexels[y * width + x] = u8vec4(result * 255.f);
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

Texture* Texture::createMultiply(std::string name, Texture* a, Texture* b, bool hdr)
{
    auto create = [a, b, hdr] (Texture* l) {
        if (!a || !a->isInitialized()) throw std::runtime_error(std::string("Error: Texture A is null/uninitialized!")); 
        if (!b || !b->isInitialized()) throw std::runtime_error(std::string("Error: Texture B is null/uninitialized!")); 

        uint32_t width = ::max(a->getWidth(), b->getWidth());
        uint32_t height = ::max(a->getHeight(), b->getHeight());
        if (hdr) l->floatTexels.resize(width * height);
        else l->byteTexels.resize(width * height);
        textureStructs[l->getId()].width = width;
        textureStructs[l->getId()].height = height;

        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                vec2 off = vec2(1 / float(width), 1 / float(height)); 
                vec2 uv = vec2(x / float(width), y / float(height)) + .5f * off;
                vec4 ac = a->sampleFloatTexels(uv);
                vec4 bc = b->sampleFloatTexels(uv);
                if (!a->isLinear()) ac = glm::convertSRGBToLinear(ac);
                if (!b->isLinear()) bc = glm::convertSRGBToLinear(bc);
                vec4 result = ac * bc;
                if (!a->isLinear() && !b->isLinear()) result = glm::convertLinearToSRGB(result);
                if (hdr) l->floatTexels[y * width + x] = result;
                else l->byteTexels[y * width + x] = u8vec4(result * 255.f);
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

vec4 Texture::sampleFloatTexels(vec2 uv) {
    uint32_t width = textureStructs[id].width;
    uint32_t height = textureStructs[id].height;
    vec2 coord = uv * vec2(width-1, height-1);
    ivec2 coord_floor = glm::ivec2(glm::floor(coord));
    ivec2 coord_ceil = glm::ivec2(glm::ceil(coord));

    // todo, interpolate four surrouding pixels
    if (floatTexels.size() > 0)
        return floatTexels[coord_floor.y * width + coord_floor.x]; 
    else 
        return vec4(byteTexels[coord_floor.y * width + coord_floor.x]) / 255.f; 
}

u8vec4 Texture::sampleByteTexels(vec2 uv) {
    uint32_t width = textureStructs[id].width;
    uint32_t height = textureStructs[id].height;
    vec2 coord = uv * vec2(width-1, height-1);
    ivec2 coord_floor = glm::ivec2(glm::floor(coord));
    ivec2 coord_ceil = glm::ivec2(glm::ceil(coord));
    // todo, interpolate four surrouding pixels
    if (byteTexels.size() > 0)
        return floatTexels[coord_floor.y * width + coord_floor.x]; 
    else 
        return u8vec4(floatTexels[coord_floor.y * width + coord_floor.x] * 255.f); 
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
    std::vector<glm::vec4>().swap(t->floatTexels);
    std::vector<glm::u8vec4>().swap(t->byteTexels);
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