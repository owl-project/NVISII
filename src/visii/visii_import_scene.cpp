#include <visii/visii.h>
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <set>

class Vertex
{
  public:
	glm::vec4 point = glm::vec4(0.0);
	glm::vec4 color = glm::vec4(1, 0, 1, 1);
	glm::vec4 normal = glm::vec4(0.0);
	glm::vec2 texcoord = glm::vec2(0.0);

	std::vector<glm::vec4> wnormals = {}; // For computing normals

	bool operator==(const Vertex &other) const
	{
		bool result =
			(point == other.point && color == other.color && normal == other.normal && texcoord == other.texcoord);
		return result;
	}
};

struct TextureInfo {
    std::string path = "";
    bool is_bump = false;
};

struct TextureInfoCompare {
    bool operator() (const TextureInfo& lhs, const TextureInfo& rhs) const
    {
        return lhs.path < rhs.path;
    }
};

inline glm::mat4 aiMatrix4x4ToGlm(const aiMatrix4x4* from)
{
    glm::mat4 to;

    to[0][0] = (float)from->a1; to[0][1] = (float)from->b1;  to[0][2] = (float)from->c1; to[0][3] = (float)from->d1;
    to[1][0] = (float)from->a2; to[1][1] = (float)from->b2;  to[1][2] = (float)from->c2; to[1][3] = (float)from->d2;
    to[2][0] = (float)from->a3; to[2][1] = (float)from->b3;  to[2][2] = (float)from->c3; to[2][3] = (float)from->d3;
    to[3][0] = (float)from->a4; to[3][1] = (float)from->b4;  to[3][2] = (float)from->c4; to[3][3] = (float)from->d4;

    return to;
}

std::string dirnameOf(const std::string& fname)
{
     size_t pos = fname.find_last_of("\\/");
     return (std::string::npos == pos)
         ? ""
         : fname.substr(0, pos);
}

Scene importScene(std::string path, glm::vec3 position, glm::vec3 scale, glm::quat rotation)
{
    std::string directory = dirnameOf(path);
    
    Scene visiiScene;

    // temporary
    auto material = Material::create("default");
    visiiScene.materials.push_back(material);

    // Check and validate the specified model file extension.
    const char* extension = strrchr(path.c_str(), '.');
    if (!extension)
        throw std::runtime_error(
            std::string("Error: \"") + path + 
            std::string(" \" provide a file with a valid extension."));

    if (AI_FALSE == aiIsExtensionSupported(extension))
        throw std::runtime_error(
            std::string("Error: \"") + path + 
            std::string(" \"The specified model file extension \"") 
            + std::string(extension) + std::string("\" is currently unsupported."));

    auto scene = aiImportFile(path.c_str(), 
        aiProcessPreset_TargetRealtime_MaxQuality | 
        aiProcess_Triangulate |
        aiProcess_PreTransformVertices );
    
    if (!scene) {
        std::string err = std::string(aiGetErrorString());
        throw std::runtime_error(
            std::string("Error: \"") + path + std::string("\"") + err);
    }

    if (scene->mNumMeshes <= 0) 
        throw std::runtime_error(
            std::string("Error: \"") + path + 
            std::string("\" positions must be greater than 1!"));
    
    std::set<TextureInfo, TextureInfoCompare> texture_paths;
    std::map<std::string, Texture*> texture_map;
    
    // load materials
    for (uint32_t materialIdx = 0; materialIdx < scene->mNumMaterials; ++materialIdx) {
        auto &material = scene->mMaterials[materialIdx];
        auto name = std::string(material->GetName().C_Str());
        std::cout<<"Creating material " << name << std::endl;
        auto mat = Material::create(name);
        visiiScene.materials.push_back(mat);
        
        // todo, add texture paths to map above, load later and connect
        if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
            aiString Path;
            if (material->GetTexture(aiTextureType_DIFFUSE, 0, &Path, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS) {
                std::string path = directory + "/" + std::string(Path.C_Str());
                std::replace(path.begin(), path.end(), '\\', '/');
                texture_paths.insert({path, /* is bump */false});
            }
        }
    }

    // load textures
    for (auto &tex : texture_paths)
    {
        // std::cout<<"Texture path " << tex.path << std::endl;
        // auto &texture = scene->mTextures[textureIdx];
        // auto texturePath = std::string(texture->mFilename.C_Str());

        std::string textureName = tex.path;
        int duplicateCount = 0;
        while (Texture::get(textureName) != nullptr) {
            duplicateCount += 1;
            textureName += std::to_string(duplicateCount);
        }
        std::cout<<"Creating texture " << textureName << std::endl;
        auto texture = Texture::createFromFile(textureName, tex.path);
        visiiScene.textures.push_back(texture);
        texture_map[tex.path] = texture;
    }

    // assign textures to materials
    for (uint32_t materialIdx = 0; materialIdx < scene->mNumMaterials; ++materialIdx) {
        auto &material = scene->mMaterials[materialIdx];
        auto name = std::string(material->GetName().C_Str());
        auto mat = visiiScene.materials[materialIdx];
        
        // todo, add texture paths to map above, load later and connect
        if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
            aiString Path;
            if (material->GetTexture(aiTextureType_DIFFUSE, 0, &Path, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS) {
                std::string path = directory + "/" + std::string(Path.C_Str());
                std::replace(path.begin(), path.end(), '\\', '/');
                mat->setBaseColorTexture(texture_map[path]);
            }
        }
    }

    // load objects
    for (uint32_t meshIdx = 0; meshIdx < scene->mNumMeshes; ++meshIdx) {
        auto &aiMesh = scene->mMeshes[meshIdx];
        auto &aiVertices = aiMesh->mVertices;
        auto &aiNormals = aiMesh->mNormals;
        auto &aiFaces = aiMesh->mFaces;
        auto &aiTextureCoords = aiMesh->mTextureCoords;

        std::vector<float> positions;
        std::vector<float> normals;
        std::vector<float> texCoords;
        std::vector<uint32_t> indices;

        // mesh at the very least needs positions...
        if (!aiMesh->HasPositions()) continue;

        // note that we triangulated the meshes above
        for (uint32_t vid = 0; vid < aiMesh->mNumVertices; ++vid) {
            Vertex v;
            if (aiMesh->HasPositions()) {
                auto vert = aiVertices[vid];
                v.point.x = vert.x;
                v.point.y = vert.y;
                v.point.z = vert.z;
            }
            if (aiMesh->HasNormals()) {
                auto normal = aiNormals[vid];
                v.normal.x = normal.x;
                v.normal.y = normal.y;
                v.normal.z = normal.z;
            }
            if (aiMesh->HasTextureCoords(0)) {
                // just try to take the first texcoord
                auto texCoord = aiTextureCoords[0][vid];						
                v.texcoord.x = texCoord.x;
                v.texcoord.y = texCoord.y;
            }
            positions.push_back(v.point.x);
            positions.push_back(v.point.y);
            positions.push_back(v.point.z);
            normals.push_back(v.normal.x);
            normals.push_back(v.normal.y);
            normals.push_back(v.normal.z);
            texCoords.push_back(v.texcoord.x);
            texCoords.push_back(v.texcoord.y);
        }

        for (uint32_t faceIdx = 0; faceIdx < aiMesh->mNumFaces; ++faceIdx) {
            // faces must have only 3 indices
            auto &aiFace = aiFaces[faceIdx];			
            if (aiFace.mNumIndices != 3) continue;
            indices.push_back(aiFace.mIndices[0]);
            indices.push_back(aiFace.mIndices[1]);
            indices.push_back(aiFace.mIndices[2]);
            if (((aiFace.mIndices[0]) >= positions.size()) || 
                ((aiFace.mIndices[1]) >= positions.size()) || 
                ((aiFace.mIndices[2]) >= positions.size()))
                throw std::runtime_error(
                    std::string("Error: \"") + path +
                    std::string("\" invalid mesh index detected!"));
        }

        std::string meshName = std::string(aiMesh->mName.C_Str());
        int duplicateCount = 0;
        while (Mesh::get(meshName) != nullptr) {
            duplicateCount += 1;
            meshName += std::to_string(duplicateCount);
        }
        auto mesh = Mesh::createFromData(
            meshName, 
            positions, 3,
            normals, 3,
            /*colors*/{}, 3,
            texCoords, 2,
            indices
        );
        visiiScene.meshes.push_back(mesh);
        std::cout<<meshName<<std::endl;
    }

    std::function<void(aiNode*, Transform*)> addNode;
    addNode = [&scene, &visiiScene, &addNode, position, rotation, scale](aiNode* node, Transform* parentTransform) {
        // Create the transform to represent this node
        std::string transformName = std::string(node->mName.C_Str());
        std::cout<<transformName<<std::endl;
        int duplicateCount = 0;
        while (Transform::get(transformName) != nullptr) {
            duplicateCount += 1;
            transformName += std::to_string(duplicateCount);
        }
        auto transform = Transform::create(transformName);
        transform->setTransform(aiMatrix4x4ToGlm(&node->mTransformation));
        if (parentTransform == nullptr) {
            transform->setScale(transform->getScale() * scale);
            transform->addRotation(rotation);
            transform->addPosition(position);
        } 
        else transform->setParent(parentTransform);

        visiiScene.transforms.push_back(transform);
        
        // Create entities for each mesh that is associated with this node
        for (uint32_t mid = 0; mid < node->mNumMeshes; ++mid) {
            uint32_t meshIndex = node->mMeshes[mid];
            auto mesh = visiiScene.meshes[meshIndex];
            auto &aiMesh = scene->mMeshes[meshIndex];
            auto material = visiiScene.materials[aiMesh->mMaterialIndex];
            auto entity = Entity::create(transformName + "_" + mesh->getName());
            entity->setMesh(mesh);
            entity->setMaterial(material);
            entity->setTransform(transform);
        }

        for (uint32_t cid = 0; cid < node->mNumChildren; ++cid) 
            addNode(node->mChildren[cid], transform);
    };

    addNode(scene->mRootNode, nullptr);

    // mesh->computeMetadata();

    aiReleaseImport(scene);
    // dirtyMeshes.insert(mesh);

    return visiiScene;
}