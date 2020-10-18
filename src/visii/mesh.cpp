#ifndef TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#endif

#ifndef TINYOBJ_LOADER_OPT_IMPLEMENTATION
#define TINYOBJ_LOADER_OPT_IMPLEMENTATION
#endif

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_FS
#define TINYGLTF_NO_STB_IMAGE_WRITE

#include <sys/types.h>
#include <sys/stat.h>
#include <functional>
#include <limits>
#include <fcntl.h>
#ifndef WIN32
#include <unistd.h>
#endif

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#define GLM_FORCE_RIGHT_HANDED

#include <glm/gtx/vector_angle.hpp>

// #include "tetgen.h"

#include <visii/mesh.h>
#include <visii/entity.h>

// #include "Foton/Tools/Options.hxx"
#include <visii/utilities/hash_combiner.h>
#include <tiny_obj_loader.h>
// #include <tiny_obj_loader_opt.h>
#include <tiny_stl.h>
#include <tiny_gltf.h>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <generator/generator.hpp>

// // For some reason, windows is defining MemoryBarrier as something else, preventing me 
// // from using the vulkan MemoryBarrier type...
// #ifdef WIN32
// #undef MemoryBarrier
// #endif

Mesh Mesh::meshes[MAX_MESHES];
MeshStruct Mesh::meshStructs[MAX_MESHES];
std::map<std::string, uint32_t> Mesh::lookupTable;
std::shared_ptr<std::recursive_mutex> Mesh::editMutex;
bool Mesh::factoryInitialized = false;
std::set<Mesh*> Mesh::dirtyMeshes;

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

void buildOrthonormalBasis(glm::vec3 n, glm::vec3 &b1, glm::vec3 &b2)
{
    if (n.z < -0.9999999)
    {
        b1 = glm::vec3( 0.0, -1.0, 0.0);
        b2 = glm::vec3(-1.0,  0.0, 0.0);
        return;
    }
    float a = 1.0f / (1.0f + n.z);
    float b = -n.x*n.y*a;
    b1 = glm::vec3(1.0 - n.x*n.x*a, b, -n.x);
    b2 = glm::vec3(b, 1.0 - n.y*n.y*a, -n.y);
}

namespace std
{
template <>
struct hash<Vertex>
{
	size_t operator()(const Vertex &k) const
	{
		std::size_t h = 0;
		hash_combine(h, k.point.x, k.point.y, k.point.z,
					 k.color.x, k.color.y, k.color.z, k.color.a,
					 k.normal.x, k.normal.y, k.normal.z,
					 k.texcoord.x, k.texcoord.y);
		return h;
	}
};
} // namespace std

Mesh::Mesh() {
	this->initialized = false;
}

Mesh::Mesh(std::string name, uint32_t id)
{
	this->initialized = true;
	this->name = name;
	this->id = id;
	this->meshStructs[id].show_bounding_box = 0;
	this->meshStructs[id].numTris = 0;
	this->meshStructs[id].numVerts = 0;
}

std::string Mesh::toString() {
	std::string output;
	output += "{\n";
	output += "\ttype: \"Mesh\",\n";
	output += "\tname: \"" + name + "\",\n";
	// output += "\tnum_positions: \"" + std::to_string(positions.size()) + "\",\n";
	// output += "\tnum_edge_indices: \"" + std::to_string(edge_indices.size()) + "\",\n";
	// output += "\tnum_triangleIndices: \"" + std::to_string(triangleIndices.size()) + "\",\n";
	// output += "\tnum_tetrahedra_indices: \"" + std::to_string(tetrahedra_indices.size()) + "\",\n";
	output += "}";
	return output;
}

bool Mesh::areAnyDirty() { 
	return dirtyMeshes.size() > 0; 
}

std::set<Mesh*> Mesh::getDirtyMeshes()
{
	return dirtyMeshes;
}

void Mesh::markDirty() {
	dirtyMeshes.insert(this);
	auto entityPointers = Entity::getFront();
	for (auto &eid : entities) {
		entityPointers[eid].markDirty();
	}
};

// std::vector<float> Mesh::getVertices(uint32_t vertex_dimensions) {

// 	std::array<float, 4> test;
// 	std::cout<<sizeof(test)<<std::endl;
// 	if ((vertex_dimensions != 3) && (vertex_dimensions != 4)) {
// 		throw std::runtime_error("Error, vertex dimensions must be either 3 or 4");
// 	}

// 	if (vertex_dimensions == 4) {
// 		std::vector<float> verts(positions.size() * 4);
// 		memcpy(verts.data(), positions.data(), positions.size() * 4 * sizeof(float));
// 		return verts;
// 	}

// 	if (vertex_dimensions == 3) {
// 		std::vector<float> verts(positions.size() * 3);
// 		for (size_t i = 0; i < positions.size(); ++i) {
// 			verts[i * 3 + 0] = positions[i][0];
// 			verts[i * 3 + 1] = positions[i][1];
// 			verts[i * 3 + 2] = positions[i][2];
// 		}
// 		return verts;
// 	}
// }

std::vector<std::array<float, 3>> Mesh::getVertices() {
	return positions;
}

std::vector<glm::vec4> Mesh::getColors() {
	return colors;
}

std::vector<glm::vec4> Mesh::getNormals() {
	return normals;
}

std::vector<glm::vec2> Mesh::getTexCoords() {
	return texCoords;
}

// std::vector<uint32_t> Mesh::get_edge_indices() {
// 	return edge_indices;
// }

std::vector<uint32_t> Mesh::getTriangleIndices() {
	return triangleIndices;
}

// std::vector<uint32_t> Mesh::get_tetrahedra_indices() {
// 	return tetrahedra_indices;
// }

// vk::Buffer Mesh::get_point_buffer()
// {
// 	return pointBuffer;
// }

// vk::Buffer Mesh::get_color_buffer()
// {
// 	return colorBuffer;
// }

// vk::Buffer Mesh::get_triangle_index_buffer()
// {
// 	return triangleIndexBuffer;
// }

// vk::Buffer Mesh::get_normal_buffer()
// {
// 	return normalBuffer;
// }

// vk::Buffer Mesh::get_texcoord_buffer()
// {
// 	return texCoordBuffer;
// }

// uint32_t Mesh::get_total_edge_indices()
// {
// 	return (uint32_t)edge_indices.size();
// }

// uint32_t Mesh::get_total_triangleIndices()
// {
// 	return (uint32_t)triangleIndices.size();
// }

// uint32_t Mesh::get_total_tetrahedra_indices()
// {
// 	return (uint32_t)tetrahedra_indices.size();
// }

// uint32_t Mesh::get_index_bytes()
// {
// 	return sizeof(uint32_t);
// }

void Mesh::computeMetadata()
{
	// Compute AABB and center
	glm::vec4 s(0.0);
	meshStructs[id].bbmin = glm::vec4(std::numeric_limits<float>::max());
	meshStructs[id].bbmax = glm::vec4( std::numeric_limits<float>::lowest());
	meshStructs[id].bbmax.w = 0.f;
	meshStructs[id].bbmin.w = 0.f;
	for (int i = 0; i < positions.size(); i += 1)
	{	
		auto p = glm::vec3(positions[i][0], positions[i][1], positions[i][2]);
		s += glm::vec4(p[0], p[1], p[2], 0.0f);
		meshStructs[id].bbmin = glm::vec4(glm::min(p, glm::vec3(meshStructs[id].bbmin)), 0.0);
		meshStructs[id].bbmax = glm::vec4(glm::max(p, glm::vec3(meshStructs[id].bbmax)), 0.0);
	}
	s /= (float)positions.size();
	meshStructs[id].center = s;

	// Bounding Sphere
	meshStructs[id].bounding_sphere_radius = 0.0;
	for (int i = 0; i < positions.size(); i += 1) {
		glm::vec3 p = glm::vec3(positions[i][0], positions[i][1], positions[i][2]);
		meshStructs[id].bounding_sphere_radius = std::max(meshStructs[id].bounding_sphere_radius, 
			glm::distance(glm::vec4(p.x, p.y, p.z, 0.0f), meshStructs[id].center));
	}

	this->meshStructs[id].numTris = uint32_t(triangleIndices.size()) / 3;
	this->meshStructs[id].numVerts = uint32_t(positions.size());
}

// void Mesh::save_tetrahedralization(float quality_bound, float maximum_volume)
// {
// 	try  {
// 		/* NOTE, POSSIBLY LEAKING MEMORY */
// 		tetgenio in, out;
// 		tetgenio::facet *f;
// 		tetgenio::polygon *p;
// 		int i;

// 		// All indices start from 1.
// 		in.firstnumber = 1;

// 		in.numberofpoints = this->positions.size();
// 		in.pointlist = new REAL[in.numberofpoints * 3];
// 		for (uint32_t i = 0; i < this->positions.size(); ++i) {
// 			in.pointlist[i * 3 + 0] = this->positions[i].x;
// 			in.pointlist[i * 3 + 1] = this->positions[i].y;
// 			in.pointlist[i * 3 + 2] = this->positions[i].z;
// 		}

// 		in.numberoffacets = this->triangleIndices.size() / 3; 
// 		in.facetlist = new tetgenio::facet[in.numberoffacets];
// 		in.facetmarkerlist = new int[in.numberoffacets];

// 		for (uint32_t i = 0; i < this->triangleIndices.size() / 3; ++i) {
// 			f = &in.facetlist[i];
// 			f->numberofpolygons = 1;
// 			f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
// 			f->numberofholes = 0;
// 			f->holelist = NULL;
// 			p = &f->polygonlist[0];
// 			p->numberofvertices = 3;
// 			p->vertexlist = new int[p->numberofvertices];
// 			// Note, tetgen indices start at one.
// 			p->vertexlist[0] = triangleIndices[i * 3 + 0] + 1; 
// 			p->vertexlist[1] = triangleIndices[i * 3 + 1] + 1; 
// 			p->vertexlist[2] = triangleIndices[i * 3 + 2] + 1; 
// 			in.facetmarkerlist[i] = 0; // ?
// 		}

// 		// // Set 'in.facetmarkerlist'

// 		// in.facetmarkerlist[0] = -1;
// 		// in.facetmarkerlist[2] = 0;
// 		// in.facetmarkerlist[3] = 0;
// 		// in.facetmarkerlist[4] = 0;
// 		// in.facetmarkerlist[5] = 0;

// 		// Output the PLC to files 'barin.node' and 'barin.poly'.
// 		// in.save_nodes("barin");
// 		// in.save_poly("barin");

// 		// Tetrahedralize the PLC. Switches are chosen to read a PLC (p),
// 		//   do quality mesh generation (q) with a specified quality bound
// 		//   (1.414), and apply a maximum volume constraint (a0.1).

// 		std::string flags = "pq";
// 		flags += std::to_string(quality_bound);
// 		flags += "a";
// 		flags += std::to_string(maximum_volume);
// 		::tetrahedralize((char*)flags.c_str(), &in, &out);

// 		// // Output mesh to files 'barout.node', 'barout.ele' and 'barout.face'.
// 		out.save_nodes((char*)this->name.c_str());
// 		out.save_elements((char*)this->name.c_str());
// 		// out.save_faces((char*)this->name.c_str());
// 	}
// 	catch (...)
// 	{
// 		throw std::runtime_error("Error: failed to tetrahedralize mesh");
// 	}
// }

glm::vec3 Mesh::getCentroid()
{
	return vec3(meshStructs[id].center);
}

float Mesh::getBoundingSphereRadius()
{
	return meshStructs[id].bounding_sphere_radius;
}

glm::vec3 Mesh::getMinAabbCorner()
{
	return meshStructs[id].bbmin;
}

glm::vec3 Mesh::getMaxAabbCorner()
{
	return meshStructs[id].bbmax;
}

glm::vec3 Mesh::getAabbCenter()
{
	return meshStructs[id].bbmin + (meshStructs[id].bbmax - meshStructs[id].bbmin) * .5f;
}


// void Mesh::cleanup()
// {
// 	// auto vulkan = Libraries::Vulkan::Get();
// 	// if (!vulkan->is_initialized())
// 	// 	throw std::runtime_error( std::string("Vulkan library is not initialized"));
// 	// auto device = vulkan->get_device();
// 	// if (device == vk::Device())
// 	// 	throw std::runtime_error( std::string("Invalid vulkan device"));

// 	// /* Destroy index buffer */
// 	// device.destroyBuffer(triangleIndexBuffer);
// 	// device.freeMemory(triangleIndexBufferMemory);

// 	// /* Destroy vertex buffer */
// 	// device.destroyBuffer(pointBuffer);
// 	// device.freeMemory(pointBufferMemory);

// 	// /* Destroy vertex color buffer */
// 	// device.destroyBuffer(colorBuffer);
// 	// device.freeMemory(colorBufferMemory);

// 	// /* Destroy normal buffer */
// 	// device.destroyBuffer(normalBuffer);
// 	// device.freeMemory(normalBufferMemory);

// 	// /* Destroy uv buffer */
// 	// device.destroyBuffer(texCoordBuffer);
// 	// device.freeMemory(texCoordBufferMemory);
// }

void Mesh::clearAll()
{
	if (!isFactoryInitialized()) return;

	for (auto &mesh : meshes) {
		if (mesh.initialized) {
			// mesh.cleanup();
			Mesh::remove(mesh.name);
		}
	}
}

void Mesh::initializeFactory() {
	if (isFactoryInitialized()) return;
	editMutex = std::make_shared<std::recursive_mutex>();
	factoryInitialized = true;
}

bool Mesh::isFactoryInitialized()
{
    return factoryInitialized;
}

bool Mesh::isInitialized()
{
    return initialized;
}

void Mesh::updateComponents()
{
	if (dirtyMeshes.size() == 0) return;
	for (auto &m : dirtyMeshes) {
		if (!m->isInitialized()) continue;
		// m->computeMetadata();
	}
	dirtyMeshes.clear();
} 

// void Mesh::UploadSSBO(vk::CommandBuffer command_buffer)
// {
// 	if (!Dirty) return;
// 	Dirty = false;
//     auto vulkan = Libraries::Vulkan::Get();
//     auto device = vulkan->get_device();

//     if (SSBOMemory == vk::DeviceMemory()) return;
//     if (stagingSSBOMemory == vk::DeviceMemory()) return;

//     auto bufferSize = MAX_MESHES * sizeof(MeshStruct);

//     /* Pin the buffer */
// 	auto pinnedMemory = (MeshStruct*) device.mapMemory(stagingSSBOMemory, 0, bufferSize);
// 	if (pinnedMemory == nullptr) return;
	
// 	for (uint32_t i = 0; i < MAX_MESHES; ++i) {
// 		if (!meshes[i].is_initialized()) continue;
// 		pinnedMemory[i] = meshes[i].mesh_struct;
// 	};

// 	device.unmapMemory(stagingSSBOMemory);

//     vk::BufferCopy copyRegion;
// 	copyRegion.size = bufferSize;
//     command_buffer.copyBuffer(stagingSSBO, SSBO, copyRegion);
// }

// vk::Buffer Mesh::GetSSBO()
// {
//     if ((SSBO != vk::Buffer()) && (SSBOMemory != vk::DeviceMemory()))
//         return SSBO;
//     else return vk::Buffer();
// }

// uint32_t Mesh::GetSSBOSize()
// {
//     return MAX_MESHES * sizeof(MeshStruct);
// }

// std::vector<vk::Buffer> Mesh::GetPositionSSBOs()
// {
// 	Mesh *DefaultMesh = Get("DefaultMesh");
// 	std::vector<vk::Buffer> ssbos(MAX_MESHES);
// 	for (int i = 0; i < MAX_MESHES; ++i) {
// 		ssbos[i] = (meshes[i].initialized) ? meshes[i].pointBuffer : DefaultMesh->pointBuffer;
// 	}
// 	return ssbos;
// }

// std::vector<uint32_t> Mesh::GetPositionSSBOSizes()
// {
// 	Mesh *DefaultMesh = Get("DefaultMesh");
// 	std::vector<uint32_t> ssbo_sizes(MAX_MESHES);
// 	for (int i = 0; i < MAX_MESHES; ++i) {
// 		ssbo_sizes[i] = (meshes[i].initialized) ? meshes[i].pointBufferSize : DefaultMesh->pointBufferSize;
// 	}
// 	return ssbo_sizes;
// }

// std::vector<vk::Buffer> Mesh::GetNormalSSBOs()
// {
// 	Mesh *DefaultMesh = Get("DefaultMesh");
// 	std::vector<vk::Buffer> ssbos(MAX_MESHES);
// 	for (int i = 0; i < MAX_MESHES; ++i) {
// 		ssbos[i] = (meshes[i].initialized) ? meshes[i].normalBuffer : DefaultMesh->normalBuffer;
// 	}
// 	return ssbos;
// }

// std::vector<uint32_t> Mesh::GetNormalSSBOSizes()
// {
// 	Mesh *DefaultMesh = Get("DefaultMesh");
// 	std::vector<uint32_t> ssbo_sizes(MAX_MESHES);
// 	for (int i = 0; i < MAX_MESHES; ++i) {
// 		ssbo_sizes[i] = (meshes[i].initialized) ? meshes[i].normalBufferSize : DefaultMesh->normalBufferSize;
// 	}
// 	return ssbo_sizes;
// }

// std::vector<vk::Buffer> Mesh::GetColorSSBOs()
// {
// 	Mesh *DefaultMesh = Get("DefaultMesh");
// 	std::vector<vk::Buffer> ssbos(MAX_MESHES);
// 	for (int i = 0; i < MAX_MESHES; ++i) {
// 		ssbos[i] = (meshes[i].initialized) ? meshes[i].colorBuffer : DefaultMesh->colorBuffer;
// 	}
// 	return ssbos;
// }

// std::vector<uint32_t> Mesh::GetColorSSBOSizes()
// {
// 	Mesh *DefaultMesh = Get("DefaultMesh");
// 	std::vector<uint32_t> ssbo_sizes(MAX_MESHES);
// 	for (int i = 0; i < MAX_MESHES; ++i) {
// 		ssbo_sizes[i] = (meshes[i].initialized) ? meshes[i].colorBufferSize : DefaultMesh->colorBufferSize;
// 	}
// 	return ssbo_sizes;
// }

// std::vector<vk::Buffer> Mesh::GetTexCoordSSBOs()
// {
// 	Mesh *DefaultMesh = Get("DefaultMesh");
// 	std::vector<vk::Buffer> ssbos(MAX_MESHES);
// 	for (int i = 0; i < MAX_MESHES; ++i) {
// 		ssbos[i] = (meshes[i].initialized) ? meshes[i].texCoordBuffer : DefaultMesh->texCoordBuffer;
// 	}
// 	return ssbos;
// }

// std::vector<uint32_t> Mesh::GetTexCoordSSBOSizes()
// {
// 	Mesh *DefaultMesh = Get("DefaultMesh");
// 	std::vector<uint32_t> ssbo_sizes(MAX_MESHES);
// 	for (int i = 0; i < MAX_MESHES; ++i) {
// 		ssbo_sizes[i] = (meshes[i].initialized) ? meshes[i].texCoordBufferSize : DefaultMesh->texCoordBufferSize;
// 	}
// 	return ssbo_sizes;
// }

// std::vector<vk::Buffer> Mesh::GetIndexSSBOs()
// {
// 	Mesh *DefaultMesh = Get("DefaultMesh");
// 	std::vector<vk::Buffer> ssbos(MAX_MESHES);
// 	for (int i = 0; i < MAX_MESHES; ++i) {
// 		ssbos[i] = (meshes[i].initialized) ? meshes[i].triangleIndexBuffer : DefaultMesh->triangleIndexBuffer;
// 	}
// 	return ssbos;
// }

// std::vector<uint32_t> Mesh::GetIndexSSBOSizes()
// {
// 	Mesh *DefaultMesh = Get("DefaultMesh");
// 	std::vector<uint32_t> ssbo_sizes(MAX_MESHES);
// 	for (int i = 0; i < MAX_MESHES; ++i) {
// 		ssbo_sizes[i] = (meshes[i].initialized) ? meshes[i].triangleIndexBufferSize : DefaultMesh->triangleIndexBufferSize;
// 	}
// 	return ssbo_sizes;
// }

// const char *mmap_file(size_t *len, const char* filename)
// {
//   (*len) = 0;
// #ifdef _WIN32
//   HANDLE file = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
//   assert(file != INVALID_HANDLE_VALUE);

//   HANDLE fileMapping = CreateFileMapping(file, NULL, PAGE_READONLY, 0, 0, NULL);
//   assert(fileMapping != INVALID_HANDLE_VALUE);

//   LPVOID fileMapView = MapViewOfFile(fileMapping, FILE_MAP_READ, 0, 0, 0);
//   auto fileMapViewChar = (const char*)fileMapView;
//   assert(fileMapView != NULL);

//   LARGE_INTEGER fileSize;
//   fileSize.QuadPart = 0;
//   GetFileSizeEx(file, &fileSize);

//   (*len) = static_cast<size_t>(fileSize.QuadPart);
//   return fileMapViewChar;

// #else

//   FILE* f = fopen(filename, "rb" );
//   if (!f) {
//     fprintf(stderr, "Failed to open file : %s\n", filename);
//     return nullptr;
//   }
//   fseek(f, 0, SEEK_END);
//   long fileSize = ftell(f);
//   fclose(f);

//   if (fileSize < 16) {
//     fprintf(stderr, "Empty or invalid .obj : %s\n", filename);
//     return nullptr;
//   }

//   struct stat sb;
//   char *p;
//   int fd;

//   fd = open (filename, O_RDONLY);
//   if (fd == -1) {
//     perror ("open");
//     return nullptr;
//   }

//   if (fstat (fd, &sb) == -1) {
//     perror ("fstat");
//     return nullptr;
//   }

//   if (!S_ISREG (sb.st_mode)) {
//     fprintf (stderr, "%s is not a file\n", filename);
//     return nullptr;
//   }

//   p = (char*)mmap (0, fileSize, PROT_READ, MAP_SHARED, fd, 0);

//   if (p == MAP_FAILED) {
//     perror ("mmap");
//     return nullptr;
//   }

//   if (close (fd) == -1) {
//     perror ("close");
//     return nullptr;
//   }

//   (*len) = fileSize;

//   return p;

// #endif
// }

// const char* get_file_data(size_t *len, const char* filename)
// {
//     const char *ext = strrchr(filename, '.');
//     size_t data_len = 0;
//     const char* data = nullptr;
// 	data = mmap_file(&data_len, filename);
//     (*len) = data_len;
//     return data;
// }

// void CalcNormal(float N[3], float v0[3], float v1[3], float v2[3]) {
//   float v10[3];
//   v10[0] = v1[0] - v0[0];
//   v10[1] = v1[1] - v0[1];
//   v10[2] = v1[2] - v0[2];

//   float v20[3];
//   v20[0] = v2[0] - v0[0];
//   v20[1] = v2[1] - v0[1];
//   v20[2] = v2[2] - v0[2];

//   N[0] = v20[1] * v10[2] - v20[2] * v10[1];
//   N[1] = v20[2] * v10[0] - v20[0] * v10[2];
//   N[2] = v20[0] * v10[1] - v20[1] * v10[0];

//   float len2 = N[0] * N[0] + N[1] * N[1] + N[2] * N[2];
//   if (len2 > 0.0f) {
//     float len = sqrtf(len2);

//     N[0] /= len;
//     N[1] /= len;
//   }
// }

// void Mesh::loadObj(std::string objPath)
// {
// 	tinyobj_opt::attrib_t attrib;
// 	std::vector<tinyobj_opt::shape_t> shapes;
// 	std::vector<tinyobj_opt::material_t> materials;
// 	int num_threads = -1;

// 	struct stat st;
// 	if (stat(objPath.c_str(), &st) != 0)
// 		throw std::runtime_error(std::string(objPath + " does not exist!"));

// 	size_t data_len = 0;
//   	const char* data = get_file_data(&data_len, objPath.c_str());

// 	tinyobj_opt::LoadOption option;
// 	option.req_num_threads = num_threads;
// 	option.verbose = false;//verbose;
// 	option.triangulate = true;
// 	bool ret = parseObj(&attrib, &shapes, &materials, data, data_len, option);

// 	if (!ret) {
// 		throw std::runtime_error( std::string("Error: Failed to parse " + objPath));
// 	}

// 	std::vector<Vertex> vertices;

// 	bool has_normals = false;

// 	{
// 		size_t face_offset = 0;
// 		for (size_t v = 0; v < attrib.face_num_verts.size(); ++v) {
// 			if (attrib.face_num_verts[v] % 3 != 0) {
// 				throw std::runtime_error( std::string("Error: Found non-triangular face in " + objPath));
// 			}
// 			for (size_t f = 0; f < attrib.face_num_verts[v] / 3; f++) {
// 				tinyobj_opt::index_t idx0 = attrib.indices[face_offset+3*f+0];
// 				tinyobj_opt::index_t idx1 = attrib.indices[face_offset+3*f+1];
// 				tinyobj_opt::index_t idx2 = attrib.indices[face_offset+3*f+2];

// 				Vertex v[3] = {Vertex(), Vertex(), Vertex()};
// 				for (int k = 0; k < 3; k++) {
// 					int f0 = idx0.vertex_index;
// 					int f1 = idx1.vertex_index;
// 					int f2 = idx2.vertex_index;
// 					assert(f0 >= 0);
// 					assert(f1 >= 0);
// 					assert(f2 >= 0);

// 					v[0].point[k] = attrib.vertices[3*f0+k];
// 					v[1].point[k] = attrib.vertices[3*f1+k];
// 					v[2].point[k] = attrib.vertices[3*f2+k];
// 				}

// 				if (attrib.normals.size() > 0) {
// 					int nf0 = idx0.normal_index;
// 					int nf1 = idx1.normal_index;
// 					int nf2 = idx2.normal_index;

// 					if (nf0 >= 0 && nf1 >= 0 && nf2 >= 0) {
// 						assert(3*nf0+2 < attrib.normals.size());
// 						assert(3*nf1+2 < attrib.normals.size());
// 						assert(3*nf2+2 < attrib.normals.size());
// 						for (int k = 0; k < 3; k++) {
// 							v[0].normal[k] = attrib.normals[3*nf0+k];
// 							v[1].normal[k] = attrib.normals[3*nf1+k];
// 							v[2].normal[k] = attrib.normals[3*nf2+k];
// 						}
// 					} else {
// 						// compute geometric normal
// 						CalcNormal(&v[0].normal.x, &v[0].point.x, &v[1].point.x, &v[2].point.x);
// 						v[1].normal[0] = v[0].normal[0]; v[1].normal[1] = v[0].normal[1]; v[1].normal[2] = v[0].normal[2];
// 						v[2].normal[0] = v[0].normal[0]; v[2].normal[1] = v[0].normal[1]; v[2].normal[2] = v[0].normal[2];
// 					}
// 				} else {
// 					// compute geometric normal
// 					CalcNormal(&v[0].normal.x, &v[0].point.x, &v[1].point.x, &v[2].point.x);
// 					v[1].normal[0] = v[0].normal[0]; v[1].normal[1] = v[0].normal[1]; v[1].normal[2] = v[0].normal[2];
// 					v[2].normal[0] = v[0].normal[0]; v[2].normal[1] = v[0].normal[1]; v[2].normal[2] = v[0].normal[2];
// 				}

// 				if (attrib.texcoords.size() > 0) {
// 					int tcf0 = idx0.texcoord_index;
// 					int tcf1 = idx1.texcoord_index;
// 					int tcf2 = idx2.texcoord_index;

// 					if (tcf0 >= 0 && tcf1 >= 0 && tcf2 >= 0) {
// 						assert(2*tcf0+2 < attrib.texcoords.size());
// 						assert(2*tcf1+2 < attrib.texcoords.size());
// 						assert(2*tcf2+2 < attrib.texcoords.size());
// 						for (int k = 0; k < 2; k++) {
// 							v[0].texcoord[k] = attrib.texcoords[2*tcf0+k];
// 							v[1].texcoord[k] = attrib.texcoords[2*tcf1+k];
// 							v[2].texcoord[k] = attrib.texcoords[2*tcf2+k];
// 						}
// 					}
// 				}
// 				vertices.push_back(v[0]);
// 				vertices.push_back(v[1]);
// 				vertices.push_back(v[2]);
// 			}
// 			face_offset += attrib.face_num_verts[v];
// 		}
// 	}

// 	/* Map vertices to buffers */
// 	triangleIndices.resize(vertices.size());
// 	positions.resize(vertices.size());
// 	normals.resize(vertices.size());
// 	colors.resize(vertices.size());
// 	texCoords.resize(vertices.size());
// 	for (int i = 0; i < vertices.size(); ++i)
// 	{
// 		Vertex v = vertices[i];
// 		triangleIndices[i] = i;
// 		positions[i] = {v.point.x, v.point.y, v.point.z};
// 		colors[i] = v.color;
// 		normals[i] = v.normal;
// 		texCoords[i] = v.texcoord;
// 	}

// 	if (triangleIndices.size() < 3)
// 		throw std::runtime_error(std::string("Error, OBJ ") + std::string(objPath) + std::string(" has no triangles!"));

// 	computeMetadata();
// }



void Mesh::loadObj(std::string objPath)
{
	struct stat st;
	if (stat(objPath.c_str(), &st) != 0)
		throw std::runtime_error(std::string(objPath + " does not exist!"));

	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;

	tinyobj::attrib_t attrib;
	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, objPath.c_str()))
		throw std::runtime_error( std::string("Error: Unable to load " + objPath));

	std::vector<Vertex> vertices;

	bool has_normals = false;

	/* If the mesh has a set of shapes, merge them all into one */
	if (shapes.size() > 0)
	{
		for (const auto &shape : shapes)
		{
			for (const auto &index : shape.mesh.indices)
			{
				Vertex vertex = Vertex();
				vertex.point = {
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2],
					1.0f};
				if (attrib.colors.size() != 0)
				{
					vertex.color = {
						attrib.colors[3 * index.vertex_index + 0],
						attrib.colors[3 * index.vertex_index + 1],
						attrib.colors[3 * index.vertex_index + 2],
						1.f};
				}
				if (attrib.normals.size() != 0)
				{
					if (index.normal_index == -1) {
						vertex.normal = {0.f, 0.f, 0.f, 0.f};
					}
					else {
						vertex.normal = {
							attrib.normals[3 * index.normal_index + 0],
							attrib.normals[3 * index.normal_index + 1],
							attrib.normals[3 * index.normal_index + 2],
							0.0f};
						has_normals = true;
					}
				}
				if (attrib.texcoords.size() != 0)
				{
					if (index.texcoord_index == -1) {
						vertex.texcoord = {0.f, 0.f};
					}
					else {
						vertex.texcoord = {
							attrib.texcoords[2 * index.texcoord_index + 0],
							attrib.texcoords[2 * index.texcoord_index + 1]};
					}
				}
				vertices.push_back(vertex);
			}
		}
	}

	/* If the obj has no shapes, eg polylines, then try looking for per vertex data */
	else if (shapes.size() == 0)
	{
		for (int idx = 0; idx < attrib.vertices.size() / 3; ++idx)
		{
			Vertex v = Vertex();
			v.point = glm::vec4(attrib.vertices[(idx * 3)], attrib.vertices[(idx * 3) + 1], attrib.vertices[(idx * 3) + 2], 1.0f);
			if (attrib.normals.size() != 0)
			{
				v.normal = glm::vec4(attrib.normals[(idx * 3)], attrib.normals[(idx * 3) + 1], attrib.normals[(idx * 3) + 2], 0.0f);
				has_normals = true;
			}
			if (attrib.colors.size() != 0)
			{
				v.normal = glm::vec4(attrib.colors[(idx * 3)], attrib.colors[(idx * 3) + 1], attrib.colors[(idx * 3) + 2], 0.0f);
			}
			if (attrib.texcoords.size() != 0)
			{
				v.texcoord = glm::vec2(attrib.texcoords[(idx * 2)], attrib.texcoords[(idx * 2) + 1]);
			}
			vertices.push_back(v);
		}
	}

	/* Eliminate duplicate positions */
	std::unordered_map<Vertex, uint32_t> uniqueVertexMap = {};
	std::vector<Vertex> uniqueVertices;
	for (int i = 0; i < vertices.size(); ++i)
	{
		Vertex vertex = vertices[i];
		if (uniqueVertexMap.count(vertex) == 0)
		{
			uniqueVertexMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
			uniqueVertices.push_back(vertex);
		}
		triangleIndices.push_back(uniqueVertexMap[vertex]);
	}

	/* Map vertices to buffers */
	for (int i = 0; i < uniqueVertices.size(); ++i)
	{
		Vertex v = uniqueVertices[i];
		positions.push_back({v.point.x, v.point.y, v.point.z});
		colors.push_back(v.color);
		normals.push_back(v.normal);
		texCoords.push_back(v.texcoord);
	}

	if (!has_normals) {
		generateSmoothNormals();
	}

	computeMetadata();
}




// void Mesh::load_stl(std::string stlPath) {
// 	allowEdits = allow_edits;

// 	struct stat st;
// 	if (stat(stlPath.c_str(), &st) != 0)
// 		throw std::runtime_error( std::string(stlPath + " does not exist!"));

// 	std::vector<float> p;
// 	std::vector<float> n;

// 	if (!read_stl(stlPath, p, n) )
// 		throw std::runtime_error( std::string("Error: Unable to load " + stlPath));

// 	std::vector<Vertex> vertices;

// 	/* STLs only have positions and face normals, so generate colors and UVs */
// 	for (uint32_t i = 0; i < p.size() / 3; ++i) {
// 		Vertex vertex = Vertex();
// 		vertex.point = {
// 			p[i * 3 + 0],
// 			p[i * 3 + 1],
// 			p[i * 3 + 2],
// 			1.0f
// 		};
// 		vertex.normal = {
// 			n[i * 3 + 0],
// 			n[i * 3 + 1],
// 			n[i * 3 + 2],
// 			0.0f
// 		};
// 		vertices.push_back(vertex);
// 	}

// 	/* Eliminate duplicate positions */
// 	std::unordered_map<Vertex, uint32_t> uniqueVertexMap = {};
// 	std::vector<Vertex> uniqueVertices;
// 	for (int i = 0; i < vertices.size(); ++i)
// 	{
// 		Vertex vertex = vertices[i];
// 		if (uniqueVertexMap.count(vertex) == 0)
// 		{
// 			uniqueVertexMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
// 			uniqueVertices.push_back(vertex);
// 		}
// 		triangleIndices.push_back(uniqueVertexMap[vertex]);
// 	}

// 	/* Map vertices to buffers */
// 	for (int i = 0; i < uniqueVertices.size(); ++i)
// 	{
// 		Vertex v = uniqueVertices[i];
// 		positions.push_back(v.point);
// 		colors.push_back(v.color);
// 		normals.push_back(v.normal);
// 		texcoords.push_back(v.texcoord);
// 	}

// 	cleanup();
// 	createPointBuffer(allow_edits, submit_immediately);
// 	createColorBuffer(allow_edits, submit_immediately);
// 	createTriangleIndexBuffer(allow_edits, submit_immediately);
// 	createNormalBuffer(allow_edits, submit_immediately);
// 	createTexCoordBuffer(allow_edits, submit_immediately);
// 	compute_metadata(submit_immediately);
// }

// void Mesh::load_glb(std::string glbPath)
// {
// 	allowEdits = allow_edits;
// 	struct stat st;
// 	if (stat(glbPath.c_str(), &st) != 0)
// 	{
// 		throw std::runtime_error(std::string("Error: " + glbPath + " does not exist"));
// 	}

// 	// read file
// 	unsigned char *file_buffer = NULL;
// 	uint32_t file_size = 0;
// 	{
// 		FILE *fp = fopen(glbPath.c_str(), "rb");
// 		if (!fp) {
// 			throw std::runtime_error( std::string(glbPath + " does not exist!"));
// 		}
// 		assert(fp);
// 		fseek(fp, 0, SEEK_END);
// 		file_size = (uint32_t)ftell(fp);
// 		rewind(fp);
// 		file_buffer = (unsigned char *)malloc(file_size);
// 		assert(file_buffer);
// 		fread(file_buffer, 1, file_size, fp);
// 		fclose(fp);
// 	}

// 	tinygltf::Model model;
// 	tinygltf::TinyGLTF loader;

// 	std::string err, warn;
// 	if (!loader.LoadBinaryFromMemory(&model, &err, &warn, file_buffer, file_size, "", tinygltf::REQUIRE_ALL))
// 		throw std::runtime_error( std::string("Error: Unable to load " + glbPath + " " + err));

// 	std::vector<Vertex> vertices;

// 	for (const auto &mesh : model.meshes) {
// 		for (const auto &primitive : mesh.primitives)
// 		{
// 			const auto &idx_accessor = model.accessors[primitive.indices];
// 			const auto &pos_accessor = model.accessors[primitive.attributes.find("POSITION")->second];
// 			const auto &nrm_accessor = model.accessors[primitive.attributes.find("NORMAL")->second];
// 			const auto &tex_accessor = model.accessors[primitive.attributes.find("TEXCOORD_0")->second];

// 			const auto &idx_bufferView = model.bufferViews[idx_accessor.bufferView];
// 			const auto &pos_bufferView = model.bufferViews[pos_accessor.bufferView];
// 			const auto &nrm_bufferView = model.bufferViews[nrm_accessor.bufferView];
// 			const auto &tex_bufferView = model.bufferViews[tex_accessor.bufferView];

// 			const auto &idx_buffer = model.buffers[idx_bufferView.buffer]; 
// 			const auto &pos_buffer = model.buffers[pos_bufferView.buffer]; 
// 			const auto &nrm_buffer = model.buffers[nrm_bufferView.buffer]; 
// 			const auto &tex_buffer = model.buffers[tex_bufferView.buffer]; 

// 			const float *pos = (const float *) pos_buffer.data.data();
// 			const float *nrm = (const float *) nrm_buffer.data.data();
// 			const float *tex = (const float *) tex_buffer.data.data();
// 			const char* idx  = (const char *) &idx_buffer.data[idx_bufferView.byteOffset];

// 			/* For each vertex */
// 			for (int i = 0; i < idx_accessor.count; ++ i) {
// 				unsigned int index = -1;
// 				if (idx_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
// 					index = (unsigned int) ((unsigned int*)idx)[i];
// 				else 
// 					index = (unsigned int) ((unsigned short*)idx)[i];
				
// 				Vertex vertex = Vertex();
// 				vertex.point = {
// 					pos[3 * index + 0],
// 					pos[3 * index + 1],
// 					pos[3 * index + 2],
// 					1.0f};

// 				vertex.normal = {
// 					nrm[3 * index + 0],
// 					nrm[3 * index + 1],
// 					nrm[3 * index + 2],
// 					0.0f};

// 				vertex.texcoord = {
// 					tex[2 * index + 0],
// 					tex[2 * index + 1]};
				
// 				vertices.push_back(vertex);
// 			}
// 		}
// 	}

// 	/* Eliminate duplicate positions */
// 	std::unordered_map<Vertex, uint32_t> uniqueVertexMap = {};
// 	std::vector<Vertex> uniqueVertices;
// 	for (int i = 0; i < vertices.size(); ++i)
// 	{
// 		Vertex vertex = vertices[i];
// 		if (uniqueVertexMap.count(vertex) == 0)
// 		{
// 			uniqueVertexMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
// 			uniqueVertices.push_back(vertex);
// 		}
// 		triangleIndices.push_back(uniqueVertexMap[vertex]);
// 	}

// 	/* Map vertices to buffers */
// 	for (int i = 0; i < uniqueVertices.size(); ++i)
// 	{
// 		Vertex v = uniqueVertices[i];
// 		positions.push_back(v.point);
// 		colors.push_back(v.color);
// 		normals.push_back(v.normal);
// 		texcoords.push_back(v.texcoord);
// 	}

// 	cleanup();
// 	createPointBuffer(allow_edits, submit_immediately);
// 	createColorBuffer(allow_edits, submit_immediately);
// 	createTriangleIndexBuffer(allow_edits, submit_immediately);
// 	createNormalBuffer(allow_edits, submit_immediately);
// 	createTexCoordBuffer(allow_edits, submit_immediately);
// 	compute_metadata(submit_immediately);
// }

// struct Triangle {
// 	uint32_t idx[3];

// 	bool operator==(const Triangle &other) const
// 	{ 
// 		uint32_t s_idx1[3];
// 		uint32_t s_idx2[3];

// 		std::copy(std::begin(idx), std::end(idx), std::begin(s_idx1));
// 		std::copy(std::begin(other.idx), std::end(other.idx), std::begin(s_idx2));

// 		std::sort(std::begin(s_idx1), std::end(s_idx1));
// 		std::sort(std::begin(s_idx2), std::end(s_idx2));

// 		return (s_idx1[0] == s_idx2[0]
// 			&& s_idx1[1] == s_idx2[1]
// 			&& s_idx1[2] == s_idx2[2]);
// 	}
// };

// namespace std {
// 	template <>
// 	struct hash<Triangle>
// 	{
// 		std::size_t operator()(const Triangle& t) const
// 		{
// 			using std::size_t;
// 			using std::hash;
// 			uint32_t s_idx[3];
// 			std::copy(std::begin(t.idx), std::end(t.idx), &s_idx[0]);
// 			std::sort(std::begin(s_idx), std::end(s_idx));
			
// 			return ((hash<uint32_t>()(s_idx[0])
// 					^ (hash<uint32_t>()(s_idx[1]) << 1)) >> 1)
// 					^ (hash<uint32_t>()(s_idx[2]) << 1);
// 		}
// 	};
// }

// void Mesh::load_tetgen(std::string path)
// {
// 	struct stat st;
// 	allowEdits = allow_edits;
	
// 	size_t lastindex = path.find_last_of("."); 
// 	std::string rawname = path.substr(0, lastindex); 

// 	std::string nodePath = rawname + ".node";
// 	std::string elePath = rawname + ".node";
	
// 	if (stat(nodePath.c_str(), &st) != 0)
// 		throw std::runtime_error(std::string("Error: " + nodePath + " does not exist"));

// 	if (stat(elePath.c_str(), &st) != 0)
// 		throw std::runtime_error(std::string("Error: " + elePath + " does not exist"));

// 	// Somehow here, verify the node and ele files are in the same directory...
// 	tetgenio in;

// 	in.load_tetmesh((char*)rawname.c_str());

// 	if (in.mesh_dim != 3) 
// 		throw std::runtime_error(std::string("Error: Node dimension must be 3"));

// 	if (in.numberoftetrahedra <= 0)
// 		throw std::runtime_error(std::string("Error: number of tetrahedra must be more than 0"));

// 	std::vector<Vertex> tri_vertices;
// 	std::vector<Vertex> tet_vertices;
// 	for (uint32_t i = 0; i < (uint32_t)in.numberoftetrahedra; ++i) {
// 		uint32_t i1 = in.tetrahedronlist[i * 4 + 0] - 1;
// 		uint32_t i2 = in.tetrahedronlist[i * 4 + 1] - 1;
// 		uint32_t i3 = in.tetrahedronlist[i * 4 + 2] - 1;
// 		uint32_t i4 = in.tetrahedronlist[i * 4 + 3] - 1;

// 		Vertex v1, v2, v3, v4;
// 		v1.point = glm::vec4(in.pointlist[i1 * 3 + 0], in.pointlist[i1 * 3 + 1], in.pointlist[i1 * 3 + 2], 1.0f);
// 		v2.point = glm::vec4(in.pointlist[i2 * 3 + 0], in.pointlist[i2 * 3 + 1], in.pointlist[i2 * 3 + 2], 1.0f);
// 		v3.point = glm::vec4(in.pointlist[i3 * 3 + 0], in.pointlist[i3 * 3 + 1], in.pointlist[i3 * 3 + 2], 1.0f);
// 		v4.point = glm::vec4(in.pointlist[i4 * 3 + 0], in.pointlist[i4 * 3 + 1], in.pointlist[i4 * 3 + 2], 1.0f);

// 		tri_vertices.push_back(v1); tri_vertices.push_back(v4); tri_vertices.push_back(v3);
// 		tri_vertices.push_back(v1); tri_vertices.push_back(v2); tri_vertices.push_back(v4);
// 		tri_vertices.push_back(v2); tri_vertices.push_back(v3); tri_vertices.push_back(v4);
// 		tri_vertices.push_back(v1); tri_vertices.push_back(v3); tri_vertices.push_back(v2);

// 		tet_vertices.push_back(v1); tet_vertices.push_back(v2); tet_vertices.push_back(v3); tet_vertices.push_back(v4);
// 	}

// 	/* Eliminate duplicate positions */
// 	std::unordered_map<Vertex, uint32_t> uniqueVertexMap = {};
// 	std::vector<Vertex> uniqueVertices;
// 	std::vector<uint32_t> all_triangleIndices;
// 	for (int i = 0; i < tri_vertices.size(); ++i)
// 	{
// 		Vertex vertex = tri_vertices[i];
// 		if (uniqueVertexMap.count(vertex) == 0)
// 		{
// 			uniqueVertexMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
// 			uniqueVertices.push_back(vertex);
// 		}
// 		all_triangleIndices.push_back(uniqueVertexMap[vertex]);
// 	}

// 	/* Only add outside triangles to triangle indices. 
// 	We do this by removing all triangles which contain duplicates. */
// 	std::unordered_map<Triangle, uint32_t> triangleCount = {};
// 	for (int i = 0; i < all_triangleIndices.size(); i+=3)
// 	{
// 		Triangle t;
// 		t.idx[0] = all_triangleIndices[i + 0];
// 		t.idx[1] = all_triangleIndices[i + 1];
// 		t.idx[2] = all_triangleIndices[i + 2];
// 		if (triangleCount.find(t) != triangleCount.end())
// 			triangleCount[t]++;
// 		else
// 			triangleCount[t] = 1;
// 	}

// 	for (auto &item : triangleCount) {
// 		if (item.second == 1) {
// 			triangleIndices.push_back(item.first.idx[0]);
// 			triangleIndices.push_back(item.first.idx[1]);
// 			triangleIndices.push_back(item.first.idx[2]);
// 		}
// 	}

// 	/* Since tetrahedra vertices are the triangle indices, we can reuse the 
// 		same unique map constructed above */
// 	for (int i = 0; i < tet_vertices.size(); ++i)
// 	{
// 		Vertex vertex = tet_vertices[i];
// 		tetrahedra_indices.push_back(uniqueVertexMap[vertex]);		
// 	}

// 	/* Map vertices to buffers */
// 	for (int i = 0; i < uniqueVertices.size(); ++i)
// 	{
// 		Vertex v = uniqueVertices[i];
// 		positions.push_back(v.point);
// 		colors.push_back(v.color);
// 		normals.push_back(v.normal);
// 		texcoords.push_back(v.texcoord);
// 	}

// 	cleanup();
// 	createPointBuffer(allow_edits, submit_immediately);
// 	createColorBuffer(allow_edits, submit_immediately);
// 	createTriangleIndexBuffer(allow_edits, submit_immediately);
// 	createNormalBuffer(allow_edits, submit_immediately);
// 	createTexCoordBuffer(allow_edits, submit_immediately);
// 	compute_metadata(submit_immediately);
// }

void Mesh::loadData(
	std::vector<float> &positions_, 
	uint32_t position_dimensions,
	std::vector<float> &normals_,
	uint32_t normal_dimensions, 
	std::vector<float> &colors_, 
	uint32_t color_dimensions,
	std::vector<float> &texcoords_, 
	uint32_t texcoord_dimensions,
	std::vector<uint32_t> indices_
)
{
	bool readingNormals = normals_.size() > 0;
	bool readingColors = colors_.size() > 0;
	bool readingTexCoords = texcoords_.size() > 0;
	bool readingIndices = indices_.size() > 0;

	if ((position_dimensions != 3) && (position_dimensions != 4)) 
		throw std::runtime_error( std::string("Error, invalid position dimensions. Possible position dimensions are 3 or 4."));
	
	if ((normal_dimensions != 3) && (normal_dimensions != 4)) 
		throw std::runtime_error( std::string("Error, invalid normal dimensions. Possible normal dimensions are 3 or 4."));

	if ((color_dimensions != 3) && (color_dimensions != 4)) 
		throw std::runtime_error( std::string("Error, invalid color dimensions. Possible color dimensions are 3 or 4."));

	if (texcoord_dimensions != 2) 
		throw std::runtime_error( std::string("Error, invalid texcoord dimensions. Possible position dimensions are 2."));

	if (positions_.size() == 0)
		throw std::runtime_error( std::string("Error, no positions supplied. "));

	if ((!readingIndices) && (((positions_.size() / position_dimensions) % 3) != 0))
		throw std::runtime_error( std::string("Error: No indices provided, and length of positions (") + std::to_string(positions_.size()) + std::string(") is not a multiple of 3."));

	if ((readingIndices) && ((indices_.size() % 3) != 0))
		throw std::runtime_error( std::string("Error: Length of indices (") + std::to_string(indices_.size()) + std::string(") is not a multiple of 3."));
	
	if (readingNormals && ((normals_.size() / normal_dimensions) != (positions_.size() / position_dimensions)))
		throw std::runtime_error( std::string("Error, length mismatch. Total normals: " + std::to_string(normals_.size() / normal_dimensions) + " does not equal total positions: " + std::to_string(positions_.size() / position_dimensions)));

	if (readingColors && ((colors_.size() / color_dimensions) != (positions_.size() / position_dimensions)))
		throw std::runtime_error( std::string("Error, length mismatch. Total colors: " + std::to_string(colors_.size() / color_dimensions) + " does not equal total positions: " + std::to_string(positions_.size() / position_dimensions)));
		
	if (readingTexCoords && ((texcoords_.size() / texcoord_dimensions) != (positions_.size() / position_dimensions)))
		throw std::runtime_error( std::string("Error, length mismatch. Total texcoords: " + std::to_string(texcoords_.size() / texcoord_dimensions) + " does not equal total positions: " + std::to_string(positions_.size() / position_dimensions)));
	
	if (readingIndices) {
		for (uint32_t i = 0; i < indices_.size(); ++i) {
			if (indices_[i] >= positions_.size())
				throw std::runtime_error( std::string("Error, index out of bounds. Index " + std::to_string(i) + " is greater than total positions: " + std::to_string(positions_.size() / position_dimensions)));
		}
	}
		
	std::vector<Vertex> vertices;

	/* For each vertex */
	for (int i = 0; i < positions_.size() / position_dimensions; ++ i) {
		Vertex vertex = Vertex();
		vertex.point.x = positions_[i * position_dimensions + 0];
		vertex.point.y = positions_[i * position_dimensions + 1];
		vertex.point.z = positions_[i * position_dimensions + 2];
		vertex.point.w = (position_dimensions == 4) ? positions_[i * position_dimensions + 3] : 1.f;
		if (readingNormals) {
			vertex.normal.x = normals_[i * normal_dimensions + 0];
			vertex.normal.y = normals_[i * normal_dimensions + 1];
			vertex.normal.z = normals_[i * normal_dimensions + 2];
			vertex.normal.w = (normal_dimensions == 4) ? normals_[i * normal_dimensions + 3] : 0.f;
		}
		if (readingColors) {
			vertex.color.x = colors_[i * color_dimensions + 0];
			vertex.color.y = colors_[i * color_dimensions + 1];
			vertex.color.z = colors_[i * color_dimensions + 2];
			vertex.color.w = (color_dimensions == 4) ? colors_[i * color_dimensions + 3] : 1.f;
		}
		if (readingTexCoords) {
			vertex.texcoord.x = texcoords_[i * texcoord_dimensions + 0];      
			vertex.texcoord.y = texcoords_[i * texcoord_dimensions + 1];      
		}  
		vertices.push_back(vertex);
	}

	/* Eliminate duplicate positions */
	std::unordered_map<Vertex, uint32_t> uniqueVertexMap = {};
	std::vector<Vertex> uniqueVertices;

	/* Don't bin positions as unique when editing, since it's unexpected for a user to lose positions */
	bool allow_edits = false; // temporary...
	if (readingIndices) {
		this->triangleIndices = indices_;
		uniqueVertices = vertices;
	}
	/* If indices werent supplied and editing isn't allowed, optimize by binning unique verts */
	else {    
		for (int i = 0; i < vertices.size(); ++i)
		{
			Vertex vertex = vertices[i];
			if (uniqueVertexMap.count(vertex) == 0)
			{
				uniqueVertexMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
				uniqueVertices.push_back(vertex);
			}
			this->triangleIndices.push_back(uniqueVertexMap[vertex]);
		}
	}

	/* Map vertices to buffers */
	this->positions.resize(uniqueVertices.size());
	this->colors.resize(uniqueVertices.size());
	this->normals.resize(uniqueVertices.size());
	this->texCoords.resize(uniqueVertices.size());
	for (int i = 0; i < uniqueVertices.size(); ++i)
	{
		Vertex v = uniqueVertices[i];
		this->positions[i] = {v.point.x, v.point.y, v.point.z};
		this->colors[i] = v.color;
		this->normals[i] = v.normal;
		this->texCoords[i] = v.texcoord;
	}

	if (!readingNormals) {
		generateSmoothNormals();
	}

	computeMetadata();
}

// void Mesh::edit_position(uint32_t index, glm::vec4 new_position)
// {
// 	auto vulkan = Libraries::Vulkan::Get();
// 	if (!vulkan->is_initialized())
// 		throw std::runtime_error("Error: Vulkan is not initialized");
// 	auto device = vulkan->get_device();

// 	if (!allowEdits)
// 		throw std::runtime_error("Error: editing this component is not allowed. \
// 			Edits can be enabled during creation.");
	
// 	if (index >= this->positions.size())
// 		throw std::runtime_error("Error: index out of bounds. Max index is " + std::to_string(this->positions.size() - 1));
	
// 	positions[index] = new_position;
// 	compute_metadata();

// 	void *data = device.mapMemory(pointBufferMemory, (index * sizeof(glm::vec4)), sizeof(glm::vec4), vk::MemoryMapFlags());
// 	memcpy(data, &new_position, sizeof(glm::vec4));
// 	device.unmapMemory(pointBufferMemory);
// }

// void Mesh::edit_positions(uint32_t index, std::vector<glm::vec4> new_positions)
// {
// 	auto vulkan = Libraries::Vulkan::Get();
// 	if (!vulkan->is_initialized())
// 		throw std::runtime_error("Error: Vulkan is not initialized");
// 	auto device = vulkan->get_device();

// 	if (!allowEdits)
// 		throw std::runtime_error("Error: editing this component is not allowed. \
// 			Edits can be enabled during creation.");
	
// 	if (index >= this->positions.size())
// 		throw std::runtime_error("Error: index out of bounds. Max index is " + std::to_string(this->positions.size() - 1));
	
// 	if ((index + new_positions.size()) > this->positions.size())
// 		throw std::runtime_error("Error: too many positions for given index, out of bounds. Max index is " + std::to_string(this->positions.size() - 1));
	
// 	memcpy(&positions[index], new_positions.data(), new_positions.size() * sizeof(glm::vec4));
// 	compute_metadata();

// 	void *data = device.mapMemory(pointBufferMemory, (index * sizeof(glm::vec4)), sizeof(glm::vec4) * new_positions.size(), vk::MemoryMapFlags());
// 	memcpy(data, new_positions.data(), sizeof(glm::vec4) * new_positions.size());
// 	device.unmapMemory(pointBufferMemory);
// }

// void Mesh::edit_normal(uint32_t index, glm::vec4 new_normal)
// {
// 	auto vulkan = Libraries::Vulkan::Get();
// 	if (!vulkan->is_initialized())
// 		throw std::runtime_error("Error: Vulkan is not initialized");
// 	auto device = vulkan->get_device();

// 	if (!allowEdits)
// 		throw std::runtime_error("Error: editing this component is not allowed. \
// 			Edits can be enabled during creation.");
	
// 	if (index >= this->normals.size())
// 		throw std::runtime_error("Error: index out of bounds. Max index is " + std::to_string(this->normals.size() - 1));
	
// 	void *data = device.mapMemory(normalBufferMemory, (index * sizeof(glm::vec4)), sizeof(glm::vec4), vk::MemoryMapFlags());
// 	memcpy(data, &new_normal, sizeof(glm::vec4));
// 	device.unmapMemory(normalBufferMemory);
// }

// void Mesh::edit_normals(uint32_t index, std::vector<glm::vec4> new_normals)
// {
// 	auto vulkan = Libraries::Vulkan::Get();
// 	if (!vulkan->is_initialized())
// 		throw std::runtime_error("Error: Vulkan is not initialized");
// 	auto device = vulkan->get_device();

// 	if (!allowEdits)
// 		throw std::runtime_error("Error: editing this component is not allowed. \
// 			Edits can be enabled during creation.");
	
// 	if (index >= this->normals.size())
// 		throw std::runtime_error("Error: index out of bounds. Max index is " + std::to_string(this->normals.size() - 1));
	
// 	if ((index + new_normals.size()) > this->normals.size())
// 		throw std::runtime_error("Error: too many normals for given index, out of bounds. Max index is " + std::to_string(this->normals.size() - 1));
	
// 	void *data = device.mapMemory(normalBufferMemory, (index * sizeof(glm::vec4)), sizeof(glm::vec4) * new_normals.size(), vk::MemoryMapFlags());
// 	memcpy(data, new_normals.data(), sizeof(glm::vec4) * new_normals.size());
// 	device.unmapMemory(normalBufferMemory);
// }

void Mesh::generateSmoothNormals()
{
	std::vector<std::vector<glm::vec4>> w_normals(positions.size());

	for (uint32_t f = 0; f < triangleIndices.size(); f += 3)
	{
		uint32_t i1 = triangleIndices[f + 0];
		uint32_t i2 = triangleIndices[f + 1];
		uint32_t i3 = triangleIndices[f + 2];

		// p1, p2 and p3 are the positions in the face (f)
		auto p1 = glm::vec3(positions[i1][0], positions[i1][1], positions[i1][2]);
		auto p2 = glm::vec3(positions[i2][0], positions[i2][1], positions[i2][2]);
		auto p3 = glm::vec3(positions[i3][0], positions[i3][1], positions[i3][2]);

		// calculate facet normal of the triangle  using cross product;
		// both components are "normalized" against a common point chosen as the base
		auto n = glm::cross((p2 - p1), (p3 - p1));    // p1 is the 'base' here

		// get the angle between the two other positions for each point;
		// the starting point will be the 'base' and the two adjacent positions will be normalized against it
		auto a1 = glm::angle(glm::normalize(p2 - p1), glm::normalize(p3 - p1));    // p1 is the 'base' here
		auto a2 = glm::angle(glm::normalize(p3 - p2), glm::normalize(p1 - p2));    // p2 is the 'base' here
		auto a3 = glm::angle(glm::normalize(p1 - p3), glm::normalize(p2 - p3));    // p3 is the 'base' here

		// normalize the initial facet normals if you want to ignore surface area
		// if (!area_weighting)
		// {
		//    n = glm::normalize(n);
		// }

		// store the weighted normal in an structured array
		auto wn1 = n * a1;
		auto wn2 = n * a2;
		auto wn3 = n * a3;
		w_normals[i1].push_back(glm::vec4(wn1.x, wn1.y, wn1.z, 0.f));
		w_normals[i2].push_back(glm::vec4(wn2.x, wn2.y, wn2.z, 0.f));
		w_normals[i3].push_back(glm::vec4(wn3.x, wn3.y, wn3.z, 0.f));
	}
	for (uint32_t v = 0; v < w_normals.size(); v++)
	{
		glm::vec4 N = glm::vec4(0.0);

		// run through the normals in each vertex's array and interpolate them
		// vertex(v) here fetches the data of the vertex at index 'v'
		for (uint32_t n = 0; n < w_normals[v].size(); n++)
		{
			N += w_normals[v][n];
		}

		// normalize the final normal
		normals[v] = glm::normalize(glm::vec4(N.x, N.y, N.z, 0.0f));
	}

	markDirty();
}

// void Mesh::edit_vertex_color(uint32_t index, glm::vec4 new_color)
// {
// 	auto vulkan = Libraries::Vulkan::Get();
// 	if (!vulkan->is_initialized())
// 		throw std::runtime_error("Error: Vulkan is not initialized");
// 	auto device = vulkan->get_device();

// 	if (!allowEdits)
// 		throw std::runtime_error("Error: editing this component is not allowed. \
// 			Edits can be enabled during creation.");
	
// 	if (index >= this->colors.size())
// 		throw std::runtime_error("Error: index out of bounds. Max index is " + std::to_string(this->colors.size() - 1));
	
// 	void *data = device.mapMemory(colorBufferMemory, (index * sizeof(glm::vec4)), sizeof(glm::vec4), vk::MemoryMapFlags());
// 	memcpy(data, &new_color, sizeof(glm::vec4));
// 	device.unmapMemory(colorBufferMemory);
// }

// void Mesh::edit_vertex_colors(uint32_t index, std::vector<glm::vec4> new_colors)
// {
// 	auto vulkan = Libraries::Vulkan::Get();
// 	if (!vulkan->is_initialized())
// 		throw std::runtime_error("Error: Vulkan is not initialized");
// 	auto device = vulkan->get_device();

// 	if (!allowEdits)
// 		throw std::runtime_error("Error: editing this component is not allowed. \
// 			Edits can be enabled during creation.");
	
// 	if (index >= this->colors.size())
// 		throw std::runtime_error("Error: index out of bounds. Max index is " + std::to_string(this->colors.size() - 1));
	
// 	if ((index + new_colors.size()) > this->colors.size())
// 		throw std::runtime_error("Error: too many colors for given index, out of bounds. Max index is " + std::to_string(this->colors.size() - 1));
	
// 	void *data = device.mapMemory(colorBufferMemory, (index * sizeof(glm::vec4)), sizeof(glm::vec4) * new_colors.size(), vk::MemoryMapFlags());
// 	memcpy(data, new_colors.data(), sizeof(glm::vec4) * new_colors.size());
// 	device.unmapMemory(colorBufferMemory);
// }

// void Mesh::edit_texture_coordinate(uint32_t index, glm::vec2 new_texcoord)
// {
// 	auto vulkan = Libraries::Vulkan::Get();
// 	if (!vulkan->is_initialized())
// 		throw std::runtime_error("Error: Vulkan is not initialized");
// 	auto device = vulkan->get_device();

// 	if (!allowEdits)
// 		throw std::runtime_error("Error: editing this component is not allowed. \
// 			Edits can be enabled during creation.");
	
// 	if (index >= this->texcoords.size())
// 		throw std::runtime_error("Error: index out of bounds. Max index is " + std::to_string(this->texcoords.size() - 1));
	
// 	void *data = device.mapMemory(texCoordBufferMemory, (index * sizeof(glm::vec2)), sizeof(glm::vec2), vk::MemoryMapFlags());
// 	memcpy(data, &new_texcoord, sizeof(glm::vec2));
// 	device.unmapMemory(texCoordBufferMemory);
// }

// void Mesh::edit_texture_coordinates(uint32_t index, std::vector<glm::vec2> new_texcoords)
// {
// 	auto vulkan = Libraries::Vulkan::Get();
// 	if (!vulkan->is_initialized())
// 		throw std::runtime_error("Error: Vulkan is not initialized");
// 	auto device = vulkan->get_device();

// 	if (!allowEdits)
// 		throw std::runtime_error("Error: editing this component is not allowed. \
// 			Edits can be enabled during creation.");
	
// 	if (index >= this->texcoords.size())
// 		throw std::runtime_error("Error: index out of bounds. Max index is " + std::to_string(this->texcoords.size() - 1));
	
// 	if ((index + new_texcoords.size()) > this->texcoords.size())
// 		throw std::runtime_error("Error: too many texture coordinates for given index, out of bounds. Max index is " + std::to_string(this->texcoords.size() - 1));
	
// 	void *data = device.mapMemory(texCoordBufferMemory, (index * sizeof(glm::vec2)), sizeof(glm::vec2) * new_texcoords.size(), vk::MemoryMapFlags());
// 	memcpy(data, new_texcoords.data(), sizeof(glm::vec2) * new_texcoords.size());
// 	device.unmapMemory(texCoordBufferMemory);
// }

// void Mesh::build_low_level_bvh(bool submit_immediately)
// {
// 	auto vulkan = Libraries::Vulkan::Get();
// 	if (!vulkan->is_initialized()) throw std::runtime_error("Error: vulkan is not initialized");

// 	if (!vulkan->is_ray_tracing_enabled()) 
// 		throw std::runtime_error("Error: Vulkan device extension VK_NVX_raytracing is not currently enabled.");
	
// 	auto dldi = vulkan->get_dldi();
// 	auto device = vulkan->get_device();
// 	if (!device) 
// 		throw std::runtime_error("Error: vulkan device not initialized");



// 	/* ----- Make geometry handle ----- */
// 	vk::GeometryDataNV geoData;

// 	{
// 		vk::GeometryTrianglesNV tris;
// 		tris.vertexData = this->pointBuffer;
// 		tris.vertexOffset = 0;
// 		tris.vertexCount = (uint32_t) this->positions.size();
// 		tris.vertexStride = sizeof(glm::vec4);
// 		tris.vertexFormat = vk::Format::eR32G32B32Sfloat;
// 		tris.indexData = this->triangleIndexBuffer;
// 		tris.indexOffset = 0;
// 		tris.indexCount = this->get_total_triangleIndices();
// 		tris.indexType = vk::IndexType::eUint32;
// 		tris.transformData = vk::Buffer();
// 		tris.transformOffset = 0;

// 		geoData.triangles = tris;
// 		geometry.geometryType = vk::GeometryTypeNV::eTriangles;
// 		geometry.geometry = geoData;
// 		// geometry.flags = vk::GeometryFlagBitsNV::eOpaque;
// 	}
	


// 	/* ----- Create the bottom level acceleration structure ----- */
// 	// Bottom level acceleration structures correspond to the geometry

// 	auto CreateAccelerationStructure = [&](vk::AccelerationStructureTypeNV type, uint32_t geometryCount,
// 		vk::GeometryNV* geometries, uint32_t instanceCount, vk::AccelerationStructureNV& AS, vk::DeviceMemory& memory)
// 	{
// 		vk::AccelerationStructureCreateInfoNV accelerationStructureInfo;
// 		accelerationStructureInfo.compactedSize = 0;
// 		accelerationStructureInfo.info.type = type;
// 		accelerationStructureInfo.info.instanceCount = instanceCount;
// 		accelerationStructureInfo.info.geometryCount = geometryCount;
// 		accelerationStructureInfo.info.pGeometries = geometries;

// 		AS = device.createAccelerationStructureNV(accelerationStructureInfo, nullptr, dldi);

// 		vk::AccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo;
// 		memoryRequirementsInfo.accelerationStructure = AS;
// 		memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeNV::eObject;

// 		vk::MemoryRequirements2 memoryRequirements;
// 		memoryRequirements = device.getAccelerationStructureMemoryRequirementsNV(memoryRequirementsInfo, dldi);

// 		vk::MemoryAllocateInfo memoryAllocateInfo;
// 		memoryAllocateInfo.allocationSize = memoryRequirements.memoryRequirements.size;
// 		memoryAllocateInfo.memoryTypeIndex = vulkan->find_memory_type(memoryRequirements.memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

// 		memory = device.allocateMemory(memoryAllocateInfo);
		
// 		vk::BindAccelerationStructureMemoryInfoNV bindInfo;
// 		bindInfo.accelerationStructure = AS;
// 		bindInfo.memory = memory;
// 		bindInfo.memoryOffset = 0;
// 		bindInfo.deviceIndexCount = 0;
// 		bindInfo.pDeviceIndices = nullptr;

// 		device.bindAccelerationStructureMemoryNV({bindInfo}, dldi);
// 	};

// 	CreateAccelerationStructure(vk::AccelerationStructureTypeNV::eBottomLevel,
// 		1, &geometry, 0, lowAS, lowASMemory);


// 	/* Build low level BVH */
// 	auto GetScratchBufferSize = [&](vk::AccelerationStructureNV handle)
// 	{
// 		vk::AccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo;
// 		memoryRequirementsInfo.accelerationStructure = handle;
// 		memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeNV::eBuildScratch;

// 		vk::MemoryRequirements2 memoryRequirements;
// 		memoryRequirements = device.getAccelerationStructureMemoryRequirementsNV( memoryRequirementsInfo, dldi);

// 		vk::DeviceSize result = memoryRequirements.memoryRequirements.size;
// 		return result;
// 	};

// 	{
// 		vk::DeviceSize scratchBufferSize = GetScratchBufferSize(lowAS);

// 		vk::BufferCreateInfo bufferInfo;
// 		bufferInfo.size = scratchBufferSize;
// 		bufferInfo.usage = vk::BufferUsageFlagBits::eRayTracingNV;
// 		bufferInfo.sharingMode = vk::SharingMode::eExclusive;
// 		vk::Buffer accelerationStructureScratchBuffer = device.createBuffer(bufferInfo);
		
// 		vk::MemoryRequirements scratchBufferRequirements;
// 		scratchBufferRequirements = device.getBufferMemoryRequirements(accelerationStructureScratchBuffer);
		
// 		vk::MemoryAllocateInfo scratchMemoryAllocateInfo;
// 		scratchMemoryAllocateInfo.allocationSize = scratchBufferRequirements.size;
// 		scratchMemoryAllocateInfo.memoryTypeIndex = vulkan->find_memory_type(scratchBufferRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

// 		vk::DeviceMemory accelerationStructureScratchMemory = device.allocateMemory(scratchMemoryAllocateInfo);
// 		device.bindBufferMemory(accelerationStructureScratchBuffer, accelerationStructureScratchMemory, 0);

// 		/* Now we can build our acceleration structure */
// 		vk::MemoryBarrier memoryBarrier;
// 		memoryBarrier.srcAccessMask  = vk::AccessFlagBits::eAccelerationStructureWriteNV;
// 		memoryBarrier.srcAccessMask |= vk::AccessFlagBits::eAccelerationStructureReadNV;
// 		memoryBarrier.dstAccessMask  = vk::AccessFlagBits::eAccelerationStructureWriteNV;
// 		memoryBarrier.dstAccessMask |= vk::AccessFlagBits::eAccelerationStructureReadNV;

// 		auto cmd = vulkan->begin_one_time_graphics_command();

// 		{
// 			vk::AccelerationStructureInfoNV asInfo;
// 			asInfo.type = vk::AccelerationStructureTypeNV::eBottomLevel;
// 			asInfo.instanceCount = 0;
// 			asInfo.geometryCount = 1;// (uint32_t)geometries.size();
// 			asInfo.pGeometries = &geometry;//&geometries[0];

// 			cmd.buildAccelerationStructureNV(&asInfo, 
// 				vk::Buffer(), 0, VK_FALSE, 
// 				lowAS, vk::AccelerationStructureNV(),
// 				accelerationStructureScratchBuffer, 0, dldi);
// 		}
		
// 		cmd.pipelineBarrier(
// 		    vk::PipelineStageFlagBits::eAccelerationStructureBuildNV, 
// 		    vk::PipelineStageFlagBits::eAccelerationStructureBuildNV, 
// 		    vk::DependencyFlags(), {memoryBarrier}, {}, {});

// 		if (submit_immediately)
// 			vulkan->end_one_time_graphics_command_immediately(cmd, "build acceleration structure", true);
// 		else
// 			vulkan->end_one_time_graphics_command(cmd, "build acceleration structure", true);
// 	}

// 	/* Get a handle to the acceleration structure */
// 	device.getAccelerationStructureHandleNV(lowAS, sizeof(uint64_t), &ASHandle, dldi);

// 	/* Might need a fence here */
// 	lowBVHBuilt = true;
// }

// vk::AccelerationStructureNV Mesh::get_low_level_bvh()
// {
// 	return lowAS;
// }

// uint64_t Mesh::get_low_level_bvh_handle()
// {
// 	return ASHandle;
// }

// vk::GeometryNV Mesh::get_nv_geometry()
// {
// 	return geometry;
// }

std::shared_ptr<std::recursive_mutex> Mesh::getEditMutex()
{
	return editMutex;
}

/* Static Factory Implementations */
Mesh* Mesh::get(std::string name) {
	return StaticFactory::get(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
}

Mesh* Mesh::createBox(std::string name, glm::vec3 size, glm::ivec3 segments)
{
	auto create = [&] (Mesh* mesh) {
		dirtyMeshes.insert(mesh);
		generator::BoxMesh gen_mesh{size, segments};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
	};

	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createCappedCone(std::string name, float radius, float size, int slices, int segments, int rings, float start, float sweep)
{
	auto create = [&] (Mesh* mesh) {
		generator::CappedConeMesh gen_mesh{radius, size, slices, segments, rings, start, sweep};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createCappedCylinder(std::string name, float radius, float size, int slices, int segments, int rings, float start, float sweep)
{
	auto create = [&] (Mesh* mesh) {
		generator::CappedCylinderMesh gen_mesh{radius, size, slices, segments, rings, start, sweep};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {		
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createCappedTube(std::string name, float radius, float innerRadius, float size, int slices, int segments, int rings, float start, float sweep)
{
	auto create = [&] (Mesh* mesh) {
		generator::CappedTubeMesh gen_mesh{radius, innerRadius, size, slices, segments, rings, start, sweep};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createCapsule(std::string name, float radius, float size, int slices, int segments, int rings, float start, float sweep)
{
	auto create = [&] (Mesh* mesh) {
		generator::CapsuleMesh gen_mesh{radius, size, slices, segments, rings, start, sweep};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
} 

Mesh* Mesh::createCone(std::string name, float radius, float size, int slices, int segments, float start, float sweep)
{
	auto create = [&] (Mesh* mesh) {
		generator::ConeMesh gen_mesh{radius, size, slices, segments, start, sweep};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}
 
Mesh* Mesh::createConvexPolygonFromCircle(std::string name, float radius, int sides, int segments, int rings)
{
	auto create = [&] (Mesh* mesh) {
		generator::ConvexPolygonMesh gen_mesh{radius, sides, segments, rings};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createConvexPolygon(std::string name, std::vector<glm::vec2> vertices, int segments, int rings)
{
	auto create = [&] (Mesh* mesh) {
		std::vector<dvec2> verts;
		for (uint32_t i = 0; i < vertices.size(); ++i) verts.push_back(dvec2(vertices[i]));
		generator::ConvexPolygonMesh gen_mesh{verts, segments, rings};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createCylinder(std::string name, float radius, float size, int slices, int segments, float start, float sweep)
{
	auto create = [&] (Mesh* mesh) {
		generator::CylinderMesh gen_mesh{radius, size, slices, segments, start, sweep};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createDisk(std::string name, float radius, float innerRadius, int slices, int rings, float start, float sweep)
{
	auto create = [&] (Mesh* mesh) {
		generator::DiskMesh gen_mesh{radius, innerRadius, slices, rings, start, sweep};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createDodecahedron(std::string name, float radius, int segments, int rings)
{
	auto create = [&] (Mesh* mesh) {
		generator::DodecahedronMesh gen_mesh{radius, segments, rings};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createPlane(std::string name, vec2 size, ivec2 segments, bool flipZ)
{
	auto create = [&] (Mesh* mesh) {
		generator::PlaneMesh gen_mesh{size, segments};
		mesh->generateProcedural(gen_mesh, flipZ);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createIcosahedron(std::string name, float radius, int segments)
{
	auto create = [&] (Mesh* mesh) {
		generator::IcosahedronMesh gen_mesh{radius, segments};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createIcosphere(std::string name, float radius, int segments)
{
	auto create = [&] (Mesh* mesh) {
		generator::IcoSphereMesh gen_mesh{radius, segments};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

/* Might add this later. Requires a callback which defines a function mapping R2->R */
// Mesh* Mesh::createParametricMesh(std::string name, uint32_t x_segments = 16, uint32_t y_segments = 16)
// {
//     if (!mesh) return nullptr;
//     auto gen_mesh = generator::ParametricMesh( , glm::ivec2(x_segments, y_segments));
//     mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		// return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
// }

Mesh* Mesh::createRoundedBox(std::string name, float radius, vec3 size, int slices, ivec3 segments)
{
	auto create = [&] (Mesh* mesh) {
		generator::RoundedBoxMesh gen_mesh{
			radius, size, slices, segments
		};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createSphere(std::string name, float radius, int slices, int segments, float sliceStart, float sliceSweep, float segmentStart, float segmentSweep)
{
	auto create = [&] (Mesh* mesh) {
		generator::SphereMesh gen_mesh{radius, slices, segments, sliceStart, sliceSweep, segmentStart, segmentSweep};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createSphericalCone(std::string name, float radius, float size, int slices, int segments, int rings, float start, float sweep)
{
	auto create = [&] (Mesh* mesh) {
		generator::SphericalConeMesh gen_mesh{radius, size, slices, segments, rings, start, sweep};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createSphericalTriangleFromSphere(std::string name, float radius, int segments)
{
	auto create = [&] (Mesh* mesh) {
		generator::SphericalTriangleMesh gen_mesh{radius, segments};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createSphericalTriangleFromTriangle(std::string name, vec3 v0, vec3 v1, vec3 v2, int segments)
{
	auto create = [&] (Mesh* mesh) {
		generator::SphericalTriangleMesh gen_mesh{v0, v1, v2, segments};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createSpring(std::string name, float minor, float major, float size, int slices, int segments, float minorStart, float minorSweep, float majorStart, float majorSweep)
{
	auto create = [&] (Mesh* mesh) {
		generator::SpringMesh gen_mesh{minor, major, size, slices, segments, minorStart, minorSweep, majorStart, majorSweep};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createTeapotahedron(std::string name, int segments)
{
	auto create = [&] (Mesh* mesh) {
		generator::TeapotMesh gen_mesh(segments);
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createTorus(std::string name, float minor, float major, int slices, int segments, float minorStart, float minorSweep, float majorStart, float majorSweep)
{
	auto create = [&] (Mesh* mesh) {
		generator::TorusMesh gen_mesh{minor, major, slices, segments, minorStart, minorSweep, majorStart, majorSweep};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createTorusKnot(std::string name, int p, int q, int slices, int segments)
{
	auto create = [&] (Mesh* mesh) {
		generator::TorusKnotMesh gen_mesh{p, q, slices, segments};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createTriangleFromCircumscribedCircle(std::string name, float radius, int segments)
{
	auto create = [&] (Mesh* mesh) {
		generator::TriangleMesh gen_mesh{radius, segments};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createTriangle(std::string name, vec3 v0, vec3 v1, vec3 v2, int segments)
{
	auto create = [&] (Mesh* mesh) {
		generator::TriangleMesh gen_mesh{v0, v1, v2, segments};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createTube(std::string name, float radius, float innerRadius, float size, int slices, int segments, float start, float sweep)
{
	auto create = [&] (Mesh* mesh) {
		generator::TubeMesh gen_mesh{radius, innerRadius, size, slices, segments, start, sweep};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createLine(std::string name, glm::vec3 start, glm::vec3 stop, float radius, int segments)
{
	auto create = [&] (Mesh* mesh) {
		using namespace generator;
		ParametricPath parametricPath {
			[start, stop](double t) {
				std::cout<<t<<std::endl;
				PathVertex vertex;				
				vertex.position = (stop * float(t)) + (start * (1.0f - float(t)));
				glm::vec3 tangent = glm::normalize(stop - start);
				glm::vec3 B1;
				glm::vec3 B2;
				buildOrthonormalBasis(tangent, B1, B2);
				vertex.tangent = tangent;
				vertex.normal = B1;
				vertex.texCoord = t;
				return vertex;
			},
			((int32_t) 1) // number of segments
		} ;
		CircleShape circle_shape(radius, segments);
		ExtrudeMesh<generator::CircleShape, generator::ParametricPath> extrude_mesh(circle_shape, parametricPath);
		mesh->generateProcedural(extrude_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	try {		
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createTubeFromPolyline(std::string name, std::vector<glm::vec3> positions, float radius, int segments)
{
	if (positions.size() <= 1)
		throw std::runtime_error("Error: positions must be greater than 1!");

	auto create = [&] (Mesh* mesh) {
		using namespace generator;
		ParametricPath parametricPath {
			[positions](double t) {
				t = t * .999f;
				
				// t is 1.0 / positions.size() - 1 and goes from 0 to 1.0
				float t_scaled = ((float)t * (((float)positions.size()) - 1.0f));
				uint32_t p1_idx = (uint32_t) floor(t_scaled);
				uint32_t p2_idx = min(p1_idx + 1, uint32_t(positions.size() - 1));

				float t_segment = t_scaled - floor(t_scaled);

				glm::vec3 p1 = positions[p1_idx];
				glm::vec3 p2 = positions[p2_idx];

				PathVertex vertex;
				
				vertex.position = (p2 * t_segment) + (p1 * (1.0f - t_segment));

				glm::vec3 next = (p2 * glm::clamp((t_segment + .01f), 0.f, 1.0f)) + (p1 * glm::clamp((1.0f - (t_segment + .01f)), 0.f, 1.f));
				glm::vec3 prev = (p2 * glm::clamp((t_segment - .01f), 0.f, 1.0f)) + (p1 * glm::clamp((1.0f - (t_segment - .01f)), 0.f, 1.f));

				glm::vec3 tangent = glm::normalize(next - prev);
				glm::vec3 B1;
				glm::vec3 B2;
				buildOrthonormalBasis(tangent, B1, B2);
				vertex.tangent = tangent;
				vertex.normal = B1;
				vertex.texCoord = t;

				return vertex;
			},
			((int32_t) positions.size() - 1) // number of segments
		} ;
		CircleShape circle_shape(radius, segments);
		ExtrudeMesh<generator::CircleShape, generator::ParametricPath> extrude_mesh(circle_shape, parametricPath);
		mesh->generateProcedural(extrude_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	
	try {		
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createRoundedRectangleTubeFromPolyline(std::string name, std::vector<glm::vec3> positions, float radius, vec2 size, int slices, ivec2 segments)
{
	if (positions.size() <= 1)
		throw std::runtime_error("Error: positions must be greater than 1!");

	auto create = [&] (Mesh* mesh) {
		using namespace generator;
		ParametricPath parametricPath {
			[positions](double t) {
				t = t * .999f;
				// t is 1.0 / positions.size() - 1 and goes from 0 to 1.0
				float t_scaled = (float)t * ((float)(positions.size()) - 1.0f);
				uint32_t p1_idx = (uint32_t) floor(t_scaled);
				uint32_t p2_idx = min(p1_idx + 1, uint32_t(positions.size() - 1));

				float t_segment = t_scaled - floor(t_scaled);

				glm::vec3 p1 = positions[p1_idx];
				glm::vec3 p2 = positions[p2_idx];

				PathVertex vertex;
				
				vertex.position = (p2 * t_segment) + (p1 * (1.0f - t_segment));

				glm::vec3 next = (p2 * (t_segment + .01f)) + (p1 * (1.0f - (t_segment + .01f)));
				glm::vec3 prev = (p2 * (t_segment - .01f)) + (p1 * (1.0f - (t_segment - .01f)));

				glm::vec3 tangent = glm::normalize(next - prev);
				glm::vec3 B1;
				glm::vec3 B2;
				buildOrthonormalBasis(tangent, B1, B2);
				vertex.tangent = tangent;
				vertex.normal = B1;
				vertex.texCoord = t;

				return vertex;
			},
			((int32_t) positions.size() - 1) // number of segments
		} ;
		RoundedRectangleShape rounded_rectangle_shape(radius, size, slices, segments);
		ExtrudeMesh<generator::RoundedRectangleShape, generator::ParametricPath> extrude_mesh(rounded_rectangle_shape, parametricPath);
		mesh->generateProcedural(extrude_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createRectangleTubeFromPolyline(std::string name, std::vector<glm::vec3> positions, vec2 size, ivec2 segments)
{
	if (positions.size() <= 1)
		throw std::runtime_error("Error: positions must be greater than 1!");
	
	auto create = [&] (Mesh* mesh) {
		using namespace generator;
		ParametricPath parametricPath {
			[positions](double t) {
				t = t * .999f;
				// t is 1.0 / positions.size() - 1 and goes from 0 to 1.0
				float t_scaled = (float)t * ((float)(positions.size()) - 1.0f);
				uint32_t p1_idx = (uint32_t) floor(t_scaled);
				uint32_t p2_idx = min(p1_idx + 1, uint32_t(positions.size() - 1));

				float t_segment = t_scaled - floor(t_scaled);

				glm::vec3 p1 = positions[p1_idx];
				glm::vec3 p2 = positions[p2_idx];

				PathVertex vertex;
				
				vertex.position = (p2 * t_segment) + (p1 * (1.0f - t_segment));

				glm::vec3 next = (p2 * (t_segment + .01f)) + (p1 * (1.0f - (t_segment + .01f)));
				glm::vec3 prev = (p2 * (t_segment - .01f)) + (p1 * (1.0f - (t_segment - .01f)));

				glm::vec3 tangent = glm::normalize(next - prev);
				glm::vec3 B1;
				glm::vec3 B2;
				buildOrthonormalBasis(tangent, B1, B2);
				vertex.tangent = tangent;
				vertex.normal = B1;
				vertex.texCoord = t;

				return vertex;
			},
			((int32_t) positions.size() - 1) // number of segments
		} ;
		RectangleShape rectangle_shape(size, segments);
		ExtrudeMesh<generator::RectangleShape, generator::ParametricPath> extrude_mesh(rectangle_shape, parametricPath);
		mesh->generateProcedural(extrude_mesh, /* flip z = */ false);
		dirtyMeshes.insert(mesh);
	};
	
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

Mesh* Mesh::createWireframeBoundingBox(
			std::string name, vec3 mn, vec3 mx, float width)
{
	auto create = [mn, mx, width] (Mesh* mesh) {
		dirtyMeshes.insert(mesh);
		
		// First, start off with a normal box
		std::vector<glm::vec2> uvs = {
			{.666f, 0.f}, {1.00f, 0.f}, {.666f, .333f}, {1.00f, .333f}, // X
			{0.f, .333f}, {.333f, .333f}, {0.f, .666f}, {.333f, .666f}, // Y
			{.000f, 0.f}, {.333f, 0.f}, {000.f, .333f}, {.333f, .333f},  // Z
			{.333f, 0.f}, {.666f, 0.f}, {.333f, .333f}, {.666f, .333f},  // -X
			{0.f, .666f}, {.333f, .666f}, {0.f, 1.f}, {.333f, 1.f},  // -Y
			{.333f, .333f}, {.666f, .333f}, {.333f, .666f}, {.666f, .666f}, // - Z
		};
		
		// std::vector<glm::vec3> verts = {
		// 	{mx[0], mn[1], mx[2]}, {mx[0], mn[1], mn[2]}, {mn[0], mn[1], mx[2]}, {mn[0], mn[1], mx[2]}, // X
		// 	{mx[0], mx[1], mn[2]}, {mn[0], mn[1], mx[2]}, {mx[0], mx[1], mx[2]}, {mn[0], mx[1], mx[2]}, // Y
		// 	{mx[0], mx[1], mx[2]}, {mn[0], mx[1], mn[2]}, {mx[0], mn[1], mx[2]}, {mn[0], mn[1], mx[2]}, // Z
		// 	{mn[0], mx[1], mx[2]}, {mx[0], mn[1], mx[2]}, {mn[0], mn[1], mx[2]}, {mx[0], mx[1], mn[2]}, // -X
		// 	{mn[0], mn[1], mx[2]}, {mn[0], mn[1], mn[2]}, {mx[0], mn[1], mx[2]}, {mx[0], mn[1], mn[2]}, // -Y
		// 	{mn[0], mx[1], mn[2]}, {mx[0], mx[1], mx[2]}, {mn[0], mn[1], mn[2]}, {mx[0], mn[1], mn[2]}, // -Z
		// };

		std::vector<glm::vec3> verts = {
			{mx[0], mn[1], mn[2]}, {mx[0], mx[1], mn[2]}, {mx[0], mn[1], mx[2]}, {mx[0], mx[1], mx[2]}, // X
			{mn[0], mx[1], mn[2]}, {mx[0], mx[1], mn[2]}, {mn[0], mx[1], mx[2]}, {mx[0], mx[1], mx[2]}, // Y
			{mn[0], mn[1], mx[2]}, {mx[0], mn[1], mx[2]}, {mn[0], mx[1], mx[2]}, {mx[0], mx[1], mx[2]}, // Z
			{mn[0], mn[1], mn[2]}, {mn[0], mx[1], mn[2]}, {mn[0], mn[1], mx[2]}, {mn[0], mx[1], mx[2]}, // -X
			{mn[0], mn[1], mn[2]}, {mx[0], mn[1], mn[2]}, {mn[0], mn[1], mx[2]}, {mx[0], mn[1], mx[2]}, // -Y
			{mn[0], mn[1], mn[2]}, {mx[0], mn[1], mn[2]}, {mn[0], mx[1], mn[2]}, {mx[0], mx[1], mn[2]}, // -Z
		};

		std::vector<glm::vec3> normals = {
			{1.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, // X
			{0.f, 1.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 1.f, 0.f}, // Y
			{0.f, 0.f, 1.f}, {0.f, 0.f, 1.f}, {0.f, 0.f, 1.f}, {0.f, 0.f, 1.f}, // Z
			{-1.f, 0.f, 0.f}, {-1.f, 0.f, 0.f}, {-1.f, 0.f, 0.f}, {-1.f, 0.f, 0.f}, // -X
			{0.f, -1.f, 0.f}, {0.f, -1.f, 0.f}, {0.f, -1.f, 0.f}, {0.f, -1.f, 0.f}, // -Y
			{0.f, 0.f, -1.f}, {0.f, 0.f, -1.f}, {0.f, 0.f, -1.f}, {0.f, 0.f, -1.f}, // -Z
		};

		std::vector<glm::ivec2> edges = { 
			{0 + (0 * 4), 1 + (0 * 4)}, {1 + (0 * 4), 3 + (0 * 4)}, {2 + (0 * 4), 3 + (0 * 4)}, {0 + (0 * 4), 2 + (0 * 4)}, // X
			{0 + (1 * 4), 1 + (1 * 4)}, {1 + (1 * 4), 3 + (1 * 4)}, {2 + (1 * 4), 3 + (1 * 4)}, {0 + (1 * 4), 2 + (1 * 4)}, // Y
			{0 + (2 * 4), 1 + (2 * 4)}, {1 + (2 * 4), 3 + (2 * 4)}, {2 + (2 * 4), 3 + (2 * 4)}, {0 + (2 * 4), 2 + (2 * 4)}, // Z
			{0 + (3 * 4), 1 + (3 * 4)}, {1 + (3 * 4), 3 + (3 * 4)}, {2 + (3 * 4), 3 + (3 * 4)}, {0 + (3 * 4), 2 + (3 * 4)}, // -X
			{0 + (4 * 4), 1 + (4 * 4)}, {1 + (4 * 4), 3 + (4 * 4)}, {2 + (4 * 4), 3 + (4 * 4)}, {0 + (4 * 4), 2 + (4 * 4)}, // -Y
			{0 + (5 * 4), 1 + (5 * 4)}, {1 + (5 * 4), 3 + (5 * 4)}, {2 + (5 * 4), 3 + (5 * 4)}, {0 + (5 * 4), 2 + (5 * 4)}, // -Z			
		};


		// Now, for each edge of that box, create a sub-box which will act as an edge for our wireframe box
		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;
		uint32_t ioff = 0;
		for (uint32_t eid = 0; eid < edges.size(); ++eid) {
			glm::vec3 p0 = verts[edges[eid].x];//glm::all(glm::lessThanEqual(verts[edges[eid].x], verts[edges[eid].y])) ? verts[edges[eid].x] : verts[edges[eid].y];
			glm::vec3 p1 = verts[edges[eid].y];//glm::all(glm::greaterThan(verts[edges[eid].x], verts[edges[eid].y])) ? verts[edges[eid].x] : verts[edges[eid].y];
			glm::vec3 mn = glm::vec3(p0) - glm::vec3(width * .5f);
			glm::vec3 mx = glm::vec3(p1) + glm::vec3(width * .5f);

			std::vector<glm::vec3> edgeVerts = {
				{mx[0], mn[1], mn[2]}, {mx[0], mx[1], mn[2]}, {mx[0], mn[1], mx[2]}, {mx[0], mx[1], mx[2]}, // X
				{mn[0], mx[1], mn[2]}, {mx[0], mx[1], mn[2]}, {mn[0], mx[1], mx[2]}, {mx[0], mx[1], mx[2]}, // Y
				{mn[0], mn[1], mx[2]}, {mx[0], mn[1], mx[2]}, {mn[0], mx[1], mx[2]}, {mx[0], mx[1], mx[2]}, // Z
				{mn[0], mn[1], mn[2]}, {mn[0], mx[1], mn[2]}, {mn[0], mn[1], mx[2]}, {mn[0], mx[1], mx[2]}, // -X
				{mn[0], mn[1], mn[2]}, {mx[0], mn[1], mn[2]}, {mn[0], mn[1], mx[2]}, {mx[0], mn[1], mx[2]}, // -Y
				{mn[0], mn[1], mn[2]}, {mx[0], mn[1], mn[2]}, {mn[0], mx[1], mn[2]}, {mx[0], mx[1], mn[2]}, // -Z
			};

			// For all faces
			for (uint32_t i = 0; i < 6; ++i) {
				bool even = ((i % 2) == 0);
				int face = -1;
				if (i < 2) face = (even) ? 0 : 3; 
				else if (i < 4) face = (even) ? 1 : 4; 
				else if (i < 6) face = (even) ? 2 : 5; 
				Vertex v1, v2, v3, v4;
				v1.point = vec4(edgeVerts[face * 4 + 0], 1.f); v2.point = vec4(edgeVerts[face * 4 + 1], 1.f); 
				v3.point = vec4(edgeVerts[face * 4 + 2], 1.f); v4.point = vec4(edgeVerts[face * 4 + 3], 1.f);
				v1.texcoord = uvs[face * 4 + 0]; v2.texcoord = uvs[face * 4 + 1]; 
				v3.texcoord = uvs[face * 4 + 2]; v4.texcoord = uvs[face * 4 + 3];
				v1.normal = vec4(normals[face * 4 + 0], 0.f); v2.normal = vec4(normals[face * 4 + 1], 0.f);
				v3.normal = vec4(normals[face * 4 + 2], 0.f); v4.normal = vec4(normals[face * 4 + 3], 0.f);
				vertices.push_back(v1); vertices.push_back(v2); vertices.push_back(v3); vertices.push_back(v4);
				indices.push_back(ioff + 0); indices.push_back(ioff + 1); indices.push_back(ioff + 2); // T0
				indices.push_back(ioff + 1); indices.push_back(ioff + 3); indices.push_back(ioff + 2); // T1
				ioff += 4; // add 4, since we added 4 new vertices.
			}
		}

		for (auto &v : vertices) {
			mesh->positions.push_back({v.point.x, v.point.y, v.point.z});
			mesh->colors.push_back(v.color);
			mesh->normals.push_back(v.normal);
			mesh->texCoords.push_back(v.texcoord);
		}
		mesh->triangleIndices = indices;
		mesh->computeMetadata();
	};
	
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}


Mesh* Mesh::createFromObj(std::string name, std::string path)
{
	static bool createFromImageDeprecatedShown = false;
    if (createFromImageDeprecatedShown == false) {
        std::cout<<"Warning, create_from_obj is deprecated and will be removed in a subsequent release. Please switch to create_from_file." << std::endl;
        createFromImageDeprecatedShown = true;
    }
	return createFromFile(name, path);
}

Mesh* Mesh::createFromFile(std::string name, std::string path)
{
	auto create = [path, name] (Mesh* mesh) {
		// Check and validate the specified model file extension.
		const char* extension = strrchr(path.c_str(), '.');
		if (!extension)
			throw std::runtime_error(
				std::string("Error: \"") + name + 
				std::string(" \" provide a file with a valid extension."));

		if (AI_FALSE == aiIsExtensionSupported(extension))
			throw std::runtime_error(
				std::string("Error: \"") + name + 
				std::string(" \"The specified model file extension \"") 
				+ std::string(extension) + std::string("\" is currently unsupported."));

		auto scene = aiImportFile(path.c_str(), 
			aiProcessPreset_TargetRealtime_MaxQuality | 
			aiProcess_Triangulate |
			aiProcess_PreTransformVertices );
		
		if (!scene) {
			std::string err = std::string(aiGetErrorString());
			throw std::runtime_error(
				std::string("Error: \"") + name + std::string("\"") + err);
		}

		if (scene->mNumMeshes <= 0) 
			throw std::runtime_error(
				std::string("Error: \"") + name + 
				std::string("\" positions must be greater than 1!"));
		
		mesh->positions.clear();
		mesh->colors.clear();
		mesh->texCoords.clear();
		mesh->normals.clear();
		mesh->triangleIndices.clear();

		uint32_t off = 0;
		for (uint32_t meshIdx = 0; meshIdx < scene->mNumMeshes; ++meshIdx) {
			auto &aiMesh = scene->mMeshes[meshIdx];
			auto &aiVertices = aiMesh->mVertices;
			auto &aiNormals = aiMesh->mNormals;
			auto &aiFaces = aiMesh->mFaces;
			auto &aiTextureCoords = aiMesh->mTextureCoords;

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
				mesh->positions.push_back({v.point.x, v.point.y, v.point.z});
				mesh->normals.push_back({v.normal.x, v.normal.y, v.normal.z, 0.f});
				mesh->texCoords.push_back({v.texcoord.x, v.texcoord.y});
			}

			for (uint32_t faceIdx = 0; faceIdx < aiMesh->mNumFaces; ++faceIdx) {
				// faces must have only 3 indices
				auto &aiFace = aiFaces[faceIdx];			
				if (aiFace.mNumIndices != 3) continue;
				 
				mesh->triangleIndices.push_back(aiFace.mIndices[0] + off);
				mesh->triangleIndices.push_back(aiFace.mIndices[1] + off);
				mesh->triangleIndices.push_back(aiFace.mIndices[2] + off);

				if (((aiFace.mIndices[0] + off) >= mesh->positions.size()) || 
					((aiFace.mIndices[1] + off) >= mesh->positions.size()) || 
					((aiFace.mIndices[2] + off) >= mesh->positions.size()))
					throw std::runtime_error(
						std::string("Error: \"") + name +
						std::string("\" invalid mesh index detected!"));
			}
			off += aiMesh->mNumVertices;
		}

		mesh->computeMetadata();

		aiReleaseImport(scene);
		dirtyMeshes.insert(mesh);
	};
	
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

// Mesh* Mesh::createFromStl(std::string name, std::string stlPath)
// {
// 	try {
// 		mesh->load_stl(stlPath, allow_edits, submit_immediately);
// 		dirtyMeshes.insert(mesh);
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::createFromGlb(std::string name, std::string glbPath)
// {
// 	try {
// 		mesh->load_glb(glbPath, allow_edits, submit_immediately);
// 		dirtyMeshes.insert(mesh);
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::createFromTetgen(std::string name, std::string path)
// {
// 	try {
// 		mesh->load_tetgen(path, allow_edits, submit_immediately);
// 		dirtyMeshes.insert(mesh);
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

Mesh* Mesh::createFromData(
	std::string name,
	std::vector<float> positions_, 
	uint32_t position_dimensions,
	std::vector<float> normals_, 
	uint32_t normal_dimensions, 
	std::vector<float> colors_, 
	uint32_t color_dimensions, 
	std::vector<float> texcoords_, 
	uint32_t texcoord_dimensions, 
	std::vector<uint32_t> indices_
) {
	auto create = [&positions_, position_dimensions, &normals_, normal_dimensions, 
				   &colors_, color_dimensions, &texcoords_, texcoord_dimensions, &indices_] 
				   (Mesh* mesh) 
	{
		mesh->loadData(positions_, position_dimensions, normals_, normal_dimensions, 
			colors_, color_dimensions, texcoords_, texcoord_dimensions, indices_);
		dirtyMeshes.insert(mesh);
	};
	
	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES, create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

void Mesh::remove(std::string name) {
	auto m = get(name);
	if (!m) return;
	std::vector<std::array<float, 3>>().swap(m->positions);
	std::vector<glm::vec4>().swap(m->normals);
	std::vector<glm::vec4>().swap(m->colors);
	std::vector<glm::vec2>().swap(m->texCoords);
	std::vector<uint32_t>().swap(m->triangleIndices);
	int32_t oldID = m->getId();
	StaticFactory::remove(editMutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
	dirtyMeshes.insert(&meshes[oldID]);
}

MeshStruct* Mesh::getFrontStruct()
{
	return meshStructs;
}

Mesh* Mesh::getFront() {
	return meshes;
}

uint32_t Mesh::getCount() {
	return MAX_MESHES;
}

std::string Mesh::getName()
{
    return name;
}

int32_t Mesh::getId()
{
    return id;
}

int32_t Mesh::getAddress()
{
	return (this - meshes);
}

std::map<std::string, uint32_t> Mesh::getNameToIdMap()
{
	return lookupTable;
}

// uint64_t Mesh::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer &buffer, vk::DeviceMemory &bufferMemory)
// {
// 	auto vulkan = Libraries::Vulkan::Get();
// 	auto device = vulkan->get_device();

// 	/* To create a VBO, we need to use this struct: */
// 	vk::BufferCreateInfo bufferInfo;
// 	bufferInfo.size = size;
// 	bufferInfo.usage = usage;
// 	bufferInfo.sharingMode = vk::SharingMode::eExclusive;

// 	/* Now create the buffer */
// 	buffer = device.createBuffer(bufferInfo);

// 	/* Identify the memory requirements for the vertex buffer */
// 	vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);

// 	/* Look for a suitable type that meets our property requirements */
// 	vk::MemoryAllocateInfo allocInfo;
// 	allocInfo.allocationSize = memRequirements.size;
// 	allocInfo.memoryTypeIndex = vulkan->find_memory_type(memRequirements.memoryTypeBits, properties);

// 	/* Now, allocate the memory for that buffer */
// 	bufferMemory = device.allocateMemory(allocInfo);

// 	/* Associate the allocated memory with the VBO handle */
// 	device.bindBufferMemory(buffer, bufferMemory, 0);

// 	return memRequirements.size;
// }

// void Mesh::createPointBuffer(bool allow_edits, bool submit_immediately)
// {
// 	auto vulkan = Libraries::Vulkan::Get();
// 	auto device = vulkan->get_device();

// 	vk::DeviceSize bufferSize = positions.size() * sizeof(glm::vec4);
// 	pointBufferSize = bufferSize;
// 	vk::Buffer stagingBuffer;
// 	vk::DeviceMemory stagingBufferMemory;
// 	createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

// 	/* Map the memory to a pointer on the host */
// 	void *data = device.mapMemory(stagingBufferMemory, 0, bufferSize,  vk::MemoryMapFlags());

// 	/* Copy over our vertex data, then unmap */
// 	memcpy(data, positions.data(), (size_t)bufferSize);
// 	device.unmapMemory(stagingBufferMemory);

// 	vk::MemoryPropertyFlags memoryProperties;
// 	if (!allowEdits) memoryProperties = vk::MemoryPropertyFlagBits::eDeviceLocal;
// 	else {
// 		memoryProperties = vk::MemoryPropertyFlagBits::eHostVisible;
// 		memoryProperties |= vk::MemoryPropertyFlagBits::eHostCoherent;
// 	}
// 	createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc | 
// 							 vk::BufferUsageFlagBits::eTransferDst | 
// 							 vk::BufferUsageFlagBits::eVertexBuffer | 
// 							 vk::BufferUsageFlagBits::eStorageBuffer, memoryProperties, pointBuffer, pointBufferMemory);
	
// 	auto cmd = vulkan->begin_one_time_graphics_command();
// 	vk::BufferCopy copyRegion;
// 	copyRegion.size = bufferSize;
// 	cmd.copyBuffer(stagingBuffer, pointBuffer, copyRegion);

// 	if (submit_immediately)
// 		vulkan->end_one_time_graphics_command_immediately(cmd, "copy point buffer", true);
// 	else
// 		vulkan->end_one_time_graphics_command(cmd, "copy point buffer", true);

// 	/* Clean up the staging buffer */
// 	device.destroyBuffer(stagingBuffer);
// 	device.freeMemory(stagingBufferMemory);
// }

// void Mesh::createColorBuffer(bool allow_edits, bool submit_immediately)
// {
// 	auto vulkan = Libraries::Vulkan::Get();
// 	auto device = vulkan->get_device();

// 	vk::DeviceSize bufferSize = colors.size() * sizeof(glm::vec4);
// 	colorBufferSize = bufferSize;
// 	vk::Buffer stagingBuffer;
// 	vk::DeviceMemory stagingBufferMemory;
// 	createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

// 	/* Map the memory to a pointer on the host */
// 	void *data = device.mapMemory(stagingBufferMemory, 0, bufferSize,  vk::MemoryMapFlags());

// 	/* Copy over our vertex data, then unmap */
// 	memcpy(data, colors.data(), (size_t)bufferSize);
// 	device.unmapMemory(stagingBufferMemory);

// 	vk::MemoryPropertyFlags memoryProperties;
// 	if (!allowEdits) memoryProperties = vk::MemoryPropertyFlagBits::eDeviceLocal;
// 	else {
// 		memoryProperties = vk::MemoryPropertyFlagBits::eHostVisible;
// 		memoryProperties |= vk::MemoryPropertyFlagBits::eHostCoherent;
// 	}
// 	createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc | 
// 							 vk::BufferUsageFlagBits::eTransferDst | 
// 							 vk::BufferUsageFlagBits::eVertexBuffer | 
// 							 vk::BufferUsageFlagBits::eStorageBuffer, memoryProperties, colorBuffer, colorBufferMemory);
	
// 	auto cmd = vulkan->begin_one_time_graphics_command();
// 	vk::BufferCopy copyRegion;
// 	copyRegion.size = bufferSize;
// 	cmd.copyBuffer(stagingBuffer, colorBuffer, copyRegion);

// 	if (submit_immediately)
// 		vulkan->end_one_time_graphics_command_immediately(cmd, "copy point color buffer", true);
// 	else
// 		vulkan->end_one_time_graphics_command(cmd, "copy point color buffer", true);

// 	/* Clean up the staging buffer */
// 	device.destroyBuffer(stagingBuffer);
// 	device.freeMemory(stagingBufferMemory);
// }

// void Mesh::createTriangleIndexBuffer(bool allow_edits, bool submit_immediately)
// {
// 	auto vulkan = Libraries::Vulkan::Get();
// 	auto device = vulkan->get_device();

// 	vk::DeviceSize bufferSize = triangleIndices.size() * sizeof(uint32_t);
// 	triangleIndexBufferSize = bufferSize;
// 	vk::Buffer stagingBuffer;
// 	vk::DeviceMemory stagingBufferMemory;
// 	createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

// 	void *data = device.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags());
// 	memcpy(data, triangleIndices.data(), (size_t)bufferSize);
// 	device.unmapMemory(stagingBufferMemory);

// 	vk::MemoryPropertyFlags memoryProperties;
// 	// if (!allowEdits) memoryProperties = vk::MemoryPropertyFlagBits::eDeviceLocal;
// 	// else {
// 		memoryProperties = vk::MemoryPropertyFlagBits::eHostVisible;
// 		memoryProperties |= vk::MemoryPropertyFlagBits::eHostCoherent;
// 	// }
// 	// Why cant I create a device local index buffer?..
// 	createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc | 
// 							 vk::BufferUsageFlagBits::eTransferDst | 
// 							 vk::BufferUsageFlagBits::eIndexBuffer | 
// 							 vk::BufferUsageFlagBits::eStorageBuffer, memoryProperties, triangleIndexBuffer, triangleIndexBufferMemory);
	
// 	auto cmd = vulkan->begin_one_time_graphics_command();
// 	vk::BufferCopy copyRegion;
// 	copyRegion.size = bufferSize;
// 	cmd.copyBuffer(stagingBuffer, triangleIndexBuffer, copyRegion);

// 	if (submit_immediately)
// 		vulkan->end_one_time_graphics_command_immediately(cmd, "copy point index buffer", true);
// 	else
// 		vulkan->end_one_time_graphics_command(cmd, "copy point index buffer", true);

// 	device.destroyBuffer(stagingBuffer);
// 	device.freeMemory(stagingBufferMemory);
// }

// void Mesh::createNormalBuffer(bool allow_edits, bool submit_immediately)
// {
// 	auto vulkan = Libraries::Vulkan::Get();
// 	auto device = vulkan->get_device();

// 	vk::DeviceSize bufferSize = normals.size() * sizeof(glm::vec4);
// 	normalBufferSize = bufferSize;
// 	vk::Buffer stagingBuffer;
// 	vk::DeviceMemory stagingBufferMemory;
// 	createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

// 	/* Map the memory to a pointer on the host */
// 	void *data = device.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags());

// 	/* Copy over our normal data, then unmap */
// 	memcpy(data, normals.data(), (size_t)bufferSize);
// 	device.unmapMemory(stagingBufferMemory);

// 	vk::MemoryPropertyFlags memoryProperties;
// 	if (!allowEdits) memoryProperties = vk::MemoryPropertyFlagBits::eDeviceLocal;
// 	else {
// 		memoryProperties = vk::MemoryPropertyFlagBits::eHostVisible;
// 		memoryProperties |= vk::MemoryPropertyFlagBits::eHostCoherent;
// 	}
// 	createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc |
// 							 vk::BufferUsageFlagBits::eTransferDst | 
// 							 vk::BufferUsageFlagBits::eVertexBuffer | 
// 							 vk::BufferUsageFlagBits::eStorageBuffer, memoryProperties, normalBuffer, normalBufferMemory);
	
// 	auto cmd = vulkan->begin_one_time_graphics_command();
// 	vk::BufferCopy copyRegion;
// 	copyRegion.size = bufferSize;
// 	cmd.copyBuffer(stagingBuffer, normalBuffer, copyRegion);

// 	if (submit_immediately)
// 		vulkan->end_one_time_graphics_command_immediately(cmd, "copy point normal buffer", true);
// 	else
// 		vulkan->end_one_time_graphics_command(cmd, "copy point normal buffer", true);

// 	/* Clean up the staging buffer */
// 	device.destroyBuffer(stagingBuffer);
// 	device.freeMemory(stagingBufferMemory);
// }

// void Mesh::createTexCoordBuffer(bool allow_edits, bool submit_immediately)
// {
// 	auto vulkan = Libraries::Vulkan::Get();
// 	auto device = vulkan->get_device();

// 	vk::DeviceSize bufferSize = texcoords.size() * sizeof(glm::vec2);
// 	texCoordBufferSize = bufferSize;
// 	vk::Buffer stagingBuffer;
// 	vk::DeviceMemory stagingBufferMemory;
// 	createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

// 	/* Map the memory to a pointer on the host */
// 	void *data = device.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags());

// 	/* Copy over our normal data, then unmap */
// 	memcpy(data, texcoords.data(), (size_t)bufferSize);
// 	device.unmapMemory(stagingBufferMemory);

// 	vk::MemoryPropertyFlags memoryProperties;
// 	if (!allowEdits) memoryProperties = vk::MemoryPropertyFlagBits::eDeviceLocal;
// 	else {
// 		memoryProperties = vk::MemoryPropertyFlagBits::eHostVisible;
// 		memoryProperties |= vk::MemoryPropertyFlagBits::eHostCoherent;
// 	}
// 	createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc | 
// 							 vk::BufferUsageFlagBits::eTransferDst | 
// 							 vk::BufferUsageFlagBits::eVertexBuffer | 
// 							 vk::BufferUsageFlagBits::eStorageBuffer, memoryProperties, texCoordBuffer, texCoordBufferMemory);
	
// 	auto cmd = vulkan->begin_one_time_graphics_command();
// 	vk::BufferCopy copyRegion;
// 	copyRegion.size = bufferSize;
// 	cmd.copyBuffer(stagingBuffer, texCoordBuffer, copyRegion);

// 	if (submit_immediately)
// 		vulkan->end_one_time_graphics_command_immediately(cmd, "copy point texcoord buffer", true);
// 	else
// 		vulkan->end_one_time_graphics_command(cmd, "copy point texcoord buffer", true);

// 	/* Clean up the staging buffer */
// 	device.destroyBuffer(stagingBuffer);
// 	device.freeMemory(stagingBufferMemory);
// }

// /* TODO */
// void Mesh::showBoundingBox(bool should_show)
// {
// 	this->mesh_struct.show_bounding_box = should_show;
// }

// bool Mesh::shouldShowBoundingBox()
// {
// 	return this->mesh_struct.show_bounding_box;
// }