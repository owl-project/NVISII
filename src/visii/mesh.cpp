#ifndef TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#endif

#define TINYGLTF_IMPLEMENTATION
// #define TINYGLTF_NO_FS
// #define TINYGLTF_NO_STB_IMAGE_WRITE

#include <sys/types.h>
#include <sys/stat.h>
#include <functional>
#include <limits>

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#define GLM_FORCE_RIGHT_HANDED

#include <glm/gtx/vector_angle.hpp>

// #include "tetgen.h"

#include <visii/mesh.h>

// #include "Foton/Tools/Options.hxx"
// #include "Foton/Tools/HashCombiner.hxx"
// #include <tiny_stl.h>
// #include <tiny_gltf.h>

#include <generator/generator.hpp>

// // For some reason, windows is defining MemoryBarrier as something else, preventing me 
// // from using the vulkan MemoryBarrier type...
// #ifdef WIN32
// #undef MemoryBarrier
// #endif

Mesh Mesh::meshes[MAX_MESHES];
MeshStruct Mesh::mesh_structs[MAX_MESHES];
std::map<std::string, uint32_t> Mesh::lookupTable;
std::shared_ptr<std::mutex> Mesh::creation_mutex;
bool Mesh::Initialized = false;
bool Mesh::Dirty = true;

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

// void buildOrthonormalBasis(glm::vec3 n, glm::vec3 &b1, glm::vec3 &b2)
// {
//     if (n.z < -0.9999999)
//     {
//         b1 = glm::vec3( 0.0, -1.0, 0.0);
//         b2 = glm::vec3(-1.0,  0.0, 0.0);
//         return;
//     }
//     float a = 1.0f / (1.0f + n.z);
//     float b = -n.x*n.y*a;
//     b1 = glm::vec3(1.0 - n.x*n.x*a, b, -n.x);
//     b2 = glm::vec3(b, 1.0 - n.y*n.y*a, -n.y);
// }

// namespace std
// {
// template <>
// struct hash<Vertex>
// {
// 	size_t operator()(const Vertex &k) const
// 	{
// 		std::size_t h = 0;
// 		hash_combine(h, k.point.x, k.point.y, k.point.z,
// 					 k.color.x, k.color.y, k.color.z, k.color.a,
// 					 k.normal.x, k.normal.y, k.normal.z,
// 					 k.texcoord.x, k.texcoord.y);
// 		return h;
// 	}
// };
// } // namespace std

Mesh::Mesh() {
	this->initialized = false;
}

Mesh::Mesh(std::string name, uint32_t id)
{
	this->initialized = true;
	this->name = name;
	this->id = id;
	this->mesh_structs[id].show_bounding_box = 0;
	this->mesh_structs[id].bb_local_to_parent = glm::mat4(1.0);
}

std::string Mesh::to_string() {
	std::string output;
	output += "{\n";
	output += "\ttype: \"Mesh\",\n";
	output += "\tname: \"" + name + "\",\n";
	// output += "\tnum_positions: \"" + std::to_string(positions.size()) + "\",\n";
	// output += "\tnum_edge_indices: \"" + std::to_string(edge_indices.size()) + "\",\n";
	// output += "\tnum_triangle_indices: \"" + std::to_string(triangle_indices.size()) + "\",\n";
	// output += "\tnum_tetrahedra_indices: \"" + std::to_string(tetrahedra_indices.size()) + "\",\n";
	output += "}";
	return output;
}


// std::vector<glm::vec4> Mesh::get_positions() {
// 	return positions;
// }

// std::vector<glm::vec4> Mesh::get_colors() {
// 	return colors;
// }

// std::vector<glm::vec4> Mesh::get_normals() {
// 	return normals;
// }

// std::vector<glm::vec2> Mesh::get_texcoords() {
// 	return texcoords;
// }

// std::vector<uint32_t> Mesh::get_edge_indices() {
// 	return edge_indices;
// }

// std::vector<uint32_t> Mesh::get_triangle_indices() {
// 	return triangle_indices;
// }

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

// uint32_t Mesh::get_total_triangle_indices()
// {
// 	return (uint32_t)triangle_indices.size();
// }

// uint32_t Mesh::get_total_tetrahedra_indices()
// {
// 	return (uint32_t)tetrahedra_indices.size();
// }

// uint32_t Mesh::get_index_bytes()
// {
// 	return sizeof(uint32_t);
// }

void Mesh::compute_metadata()
{
	glm::vec4 s(0.0);
	mesh_structs[id].bbmin = glm::vec4(std::numeric_limits<float>::max());
	mesh_structs[id].bbmax = glm::vec4( std::numeric_limits<float>::lowest());
	mesh_structs[id].bbmax.w = 0.f;
	mesh_structs[id].bbmin.w = 0.f;
	for (int i = 0; i < positions.size(); i += 1)
	{
		s += glm::vec4(positions[i].x, positions[i].y, positions[i].z, 0.0f);
		mesh_structs[id].bbmin = glm::vec4(glm::min(glm::vec3(positions[i]), glm::vec3(mesh_structs[id].bbmin)), 0.0);
		mesh_structs[id].bbmax = glm::vec4(glm::max(glm::vec3(positions[i]), glm::vec3(mesh_structs[id].bbmax)), 0.0);
	}
	s /= (float)positions.size();
	mesh_structs[id].center = s;

	mesh_structs[id].bb_local_to_parent = glm::mat4(1.0);
	mesh_structs[id].bb_local_to_parent = glm::translate(mesh_structs[id].bb_local_to_parent, glm::vec3(mesh_structs[id].bbmax + mesh_structs[id].bbmin) * .5f);
	mesh_structs[id].bb_local_to_parent = glm::scale(mesh_structs[id].bb_local_to_parent, glm::vec3(mesh_structs[id].bbmax - mesh_structs[id].bbmin) * .5f);

	mesh_structs[id].bounding_sphere_radius = 0.0;
	for (int i = 0; i < positions.size(); i += 1) {
		mesh_structs[id].bounding_sphere_radius = std::max(mesh_structs[id].bounding_sphere_radius, 
			glm::distance(glm::vec4(positions[i].x, positions[i].y, positions[i].z, 0.0f), mesh_structs[id].center));
	}
	
	// auto vulkan = Libraries::Vulkan::Get();
	// if (vulkan->is_ray_tracing_enabled()) {
	// 	build_low_level_bvh(submit_immediately);
	// }
	mark_dirty();
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

// 		in.numberoffacets = this->triangle_indices.size() / 3; 
// 		in.facetlist = new tetgenio::facet[in.numberoffacets];
// 		in.facetmarkerlist = new int[in.numberoffacets];

// 		for (uint32_t i = 0; i < this->triangle_indices.size() / 3; ++i) {
// 			f = &in.facetlist[i];
// 			f->numberofpolygons = 1;
// 			f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
// 			f->numberofholes = 0;
// 			f->holelist = NULL;
// 			p = &f->polygonlist[0];
// 			p->numberofvertices = 3;
// 			p->vertexlist = new int[p->numberofvertices];
// 			// Note, tetgen indices start at one.
// 			p->vertexlist[0] = triangle_indices[i * 3 + 0] + 1; 
// 			p->vertexlist[1] = triangle_indices[i * 3 + 1] + 1; 
// 			p->vertexlist[2] = triangle_indices[i * 3 + 2] + 1; 
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

// glm::vec3 Mesh::get_centroid()
// {
// 	return vec3(mesh_struct.center);
// }

// float Mesh::get_bounding_sphere_radius()
// {
// 	return mesh_struct.bounding_sphere_radius;
// }

// glm::vec3 Mesh::get_min_aabb_corner()
// {
// 	return mesh_struct.bbmin;
// }

// glm::vec3 Mesh::get_max_aabb_corner()
// {
// 	return mesh_struct.bbmax;
// }


void Mesh::cleanup()
{
	// auto vulkan = Libraries::Vulkan::Get();
	// if (!vulkan->is_initialized())
	// 	throw std::runtime_error( std::string("Vulkan library is not initialized"));
	// auto device = vulkan->get_device();
	// if (device == vk::Device())
	// 	throw std::runtime_error( std::string("Invalid vulkan device"));

	// /* Destroy index buffer */
	// device.destroyBuffer(triangleIndexBuffer);
	// device.freeMemory(triangleIndexBufferMemory);

	// /* Destroy vertex buffer */
	// device.destroyBuffer(pointBuffer);
	// device.freeMemory(pointBufferMemory);

	// /* Destroy vertex color buffer */
	// device.destroyBuffer(colorBuffer);
	// device.freeMemory(colorBufferMemory);

	// /* Destroy normal buffer */
	// device.destroyBuffer(normalBuffer);
	// device.freeMemory(normalBufferMemory);

	// /* Destroy uv buffer */
	// device.destroyBuffer(texCoordBuffer);
	// device.freeMemory(texCoordBufferMemory);
}

void Mesh::CleanUp()
{
	if (!IsInitialized()) return;

	for (auto &mesh : meshes) {
		if (mesh.initialized) {
			mesh.cleanup();
			Mesh::Delete(mesh.id);
		}
	}

	Initialized = false;
}

void Mesh::Initialize() {
	if (IsInitialized()) return;
	creation_mutex = std::make_shared<std::mutex>();
	Initialized = true;
}

bool Mesh::IsInitialized()
{
    return Initialized;
}

void Mesh::UpdateComponents()
{

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

// void Mesh::load_obj(std::string objPath, bool allow_edits, bool submit_immediately)
// {
// 	allowEdits = allow_edits;

// 	struct stat st;
// 	if (stat(objPath.c_str(), &st) != 0)
// 		throw std::runtime_error( std::string(objPath + " does not exist!"));

// 	std::vector<tinyobj::shape_t> shapes;
// 	std::vector<tinyobj::material_t> materials;
// 	std::string err;

// 	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, objPath.c_str()))
// 		throw std::runtime_error( std::string("Error: Unable to load " + objPath));

// 	std::vector<Vertex> vertices;

// 	bool has_normals = false;

// 	/* If the mesh has a set of shapes, merge them all into one */
// 	if (shapes.size() > 0)
// 	{
// 		for (const auto &shape : shapes)
// 		{
// 			for (const auto &index : shape.mesh.indices)
// 			{
// 				Vertex vertex = Vertex();
// 				vertex.point = {
// 					attrib.vertices[3 * index.vertex_index + 0],
// 					attrib.vertices[3 * index.vertex_index + 1],
// 					attrib.vertices[3 * index.vertex_index + 2],
// 					1.0f};
// 				if (attrib.colors.size() != 0)
// 				{
// 					vertex.color = {
// 						attrib.colors[3 * index.vertex_index + 0],
// 						attrib.colors[3 * index.vertex_index + 1],
// 						attrib.colors[3 * index.vertex_index + 2],
// 						1.f};
// 				}
// 				if (attrib.normals.size() != 0)
// 				{
// 					vertex.normal = {
// 						attrib.normals[3 * index.normal_index + 0],
// 						attrib.normals[3 * index.normal_index + 1],
// 						attrib.normals[3 * index.normal_index + 2],
// 						0.0f};
// 					has_normals = true;
// 				}
// 				if (attrib.texcoords.size() != 0)
// 				{
// 					vertex.texcoord = {
// 						attrib.texcoords[2 * index.texcoord_index + 0],
// 						attrib.texcoords[2 * index.texcoord_index + 1]};
// 				}
// 				vertices.push_back(vertex);
// 			}
// 		}
// 	}

// 	/* If the obj has no shapes, eg polylines, then try looking for per vertex data */
// 	else if (shapes.size() == 0)
// 	{
// 		for (int idx = 0; idx < attrib.vertices.size() / 3; ++idx)
// 		{
// 			Vertex v = Vertex();
// 			v.point = glm::vec4(attrib.vertices[(idx * 3)], attrib.vertices[(idx * 3) + 1], attrib.vertices[(idx * 3) + 2], 1.0f);
// 			if (attrib.normals.size() != 0)
// 			{
// 				v.normal = glm::vec4(attrib.normals[(idx * 3)], attrib.normals[(idx * 3) + 1], attrib.normals[(idx * 3) + 2], 0.0f);
// 				has_normals = true;
// 			}
// 			if (attrib.colors.size() != 0)
// 			{
// 				v.normal = glm::vec4(attrib.colors[(idx * 3)], attrib.colors[(idx * 3) + 1], attrib.colors[(idx * 3) + 2], 0.0f);
// 			}
// 			if (attrib.texcoords.size() != 0)
// 			{
// 				v.texcoord = glm::vec2(attrib.texcoords[(idx * 2)], attrib.texcoords[(idx * 2) + 1]);
// 			}
// 			vertices.push_back(v);
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
// 		triangle_indices.push_back(uniqueVertexMap[vertex]);
// 	}

// 	if (!has_normals) {
// 		compute_smooth_normals(false);
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


// void Mesh::load_stl(std::string stlPath, bool allow_edits, bool submit_immediately) {
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
// 		triangle_indices.push_back(uniqueVertexMap[vertex]);
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

// void Mesh::load_glb(std::string glbPath, bool allow_edits, bool submit_immediately)
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
// 		triangle_indices.push_back(uniqueVertexMap[vertex]);
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

// void Mesh::load_tetgen(std::string path, bool allow_edits, bool submit_immediately)
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
// 	std::vector<uint32_t> all_triangle_indices;
// 	for (int i = 0; i < tri_vertices.size(); ++i)
// 	{
// 		Vertex vertex = tri_vertices[i];
// 		if (uniqueVertexMap.count(vertex) == 0)
// 		{
// 			uniqueVertexMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
// 			uniqueVertices.push_back(vertex);
// 		}
// 		all_triangle_indices.push_back(uniqueVertexMap[vertex]);
// 	}

// 	/* Only add outside triangles to triangle indices. 
// 	We do this by removing all triangles which contain duplicates. */
// 	std::unordered_map<Triangle, uint32_t> triangleCount = {};
// 	for (int i = 0; i < all_triangle_indices.size(); i+=3)
// 	{
// 		Triangle t;
// 		t.idx[0] = all_triangle_indices[i + 0];
// 		t.idx[1] = all_triangle_indices[i + 1];
// 		t.idx[2] = all_triangle_indices[i + 2];
// 		if (triangleCount.find(t) != triangleCount.end())
// 			triangleCount[t]++;
// 		else
// 			triangleCount[t] = 1;
// 	}

// 	for (auto &item : triangleCount) {
// 		if (item.second == 1) {
// 			triangle_indices.push_back(item.first.idx[0]);
// 			triangle_indices.push_back(item.first.idx[1]);
// 			triangle_indices.push_back(item.first.idx[2]);
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

// void Mesh::load_raw(
// 	std::vector<glm::vec4> &positions_, 
// 	std::vector<glm::vec4> &normals_, 
// 	std::vector<glm::vec4> &colors_, 
// 	std::vector<glm::vec2> &texcoords_, 
// 	std::vector<uint32_t> indices_,
// 	bool allow_edits, bool submit_immediately
// )
// {
// 	allowEdits = allow_edits;
// 	bool reading_normals = normals_.size() > 0;
// 	bool reading_colors = colors_.size() > 0;
// 	bool reading_texcoords = texcoords_.size() > 0;
// 	bool reading_indices = indices_.size() > 0;

// 	if (positions_.size() == 0)
// 		throw std::runtime_error( std::string("Error, no positions supplied. "));

// 	if ((!reading_indices) && ((positions_.size() % 3) != 0))
// 		throw std::runtime_error( std::string("Error: No indices provided, and length of positions (") + std::to_string(positions_.size()) + std::string(") is not a multiple of 3."));

// 	if ((reading_indices) && ((indices_.size() % 3) != 0))
// 		throw std::runtime_error( std::string("Error: Length of indices (") + std::to_string(triangle_indices.size()) + std::string(") is not a multiple of 3."));
	
// 	if (reading_normals && (normals_.size() != positions_.size()))
// 		throw std::runtime_error( std::string("Error, length mismatch. Total normals: " + std::to_string(normals_.size()) + " does not equal total positions: " + std::to_string(positions_.size())));

// 	if (reading_colors && (colors_.size() != positions_.size()))
// 		throw std::runtime_error( std::string("Error, length mismatch. Total colors: " + std::to_string(colors_.size()) + " does not equal total positions: " + std::to_string(positions_.size())));
		
// 	if (reading_texcoords && (texcoords_.size() != positions_.size()))
// 		throw std::runtime_error( std::string("Error, length mismatch. Total texcoords: " + std::to_string(texcoords_.size()) + " does not equal total positions: " + std::to_string(positions_.size())));
	
// 	if (reading_indices) {
// 		for (uint32_t i = 0; i < indices_.size(); ++i) {
// 			if (indices_[i] >= positions_.size())
// 				throw std::runtime_error( std::string("Error, index out of bounds. Index " + std::to_string(i) + " is greater than total positions: " + std::to_string(positions_.size())));
// 		}
// 	}
		
// 	std::vector<Vertex> vertices;

// 	/* For each vertex */
// 	for (int i = 0; i < positions_.size(); ++ i) {
// 		Vertex vertex = Vertex();
// 		vertex.point = positions_[i];
// 		if (reading_normals) vertex.normal = normals_[i];
// 		if (reading_colors) vertex.color = colors_[i];
// 		if (reading_texcoords) vertex.texcoord = texcoords_[i];        
// 		vertices.push_back(vertex);
// 	}

// 	/* Eliminate duplicate positions */
// 	std::unordered_map<Vertex, uint32_t> uniqueVertexMap = {};
// 	std::vector<Vertex> uniqueVertices;

// 	/* Don't bin positions as unique when editing, since it's unexpected for a user to lose positions */
// 	if ((allow_edits && !reading_indices) || (!reading_normals)) {
// 		uniqueVertices = vertices;
// 		for (int i = 0; i < vertices.size(); ++i) {
// 			triangle_indices.push_back(i);
// 		}
// 	}
// 	else if (reading_indices) {
// 		triangle_indices = indices_;
// 		uniqueVertices = vertices;
// 	}
// 	/* If indices werent supplied and editing isn't allowed, optimize by binning unique verts */
// 	else {    
// 		for (int i = 0; i < vertices.size(); ++i)
// 		{
// 			Vertex vertex = vertices[i];
// 			if (uniqueVertexMap.count(vertex) == 0)
// 			{
// 				uniqueVertexMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
// 				uniqueVertices.push_back(vertex);
// 			}
// 			triangle_indices.push_back(uniqueVertexMap[vertex]);
// 		}
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

// 	if (!reading_normals) {
// 		compute_smooth_normals(false);
// 	}

// 	cleanup();
// 	createPointBuffer(allow_edits, submit_immediately);
// 	createColorBuffer(allow_edits, submit_immediately);
// 	createTriangleIndexBuffer(allow_edits, submit_immediately);
// 	createNormalBuffer(allow_edits, submit_immediately);
// 	createTexCoordBuffer(allow_edits, submit_immediately);
// 	compute_metadata(submit_immediately);
// }

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

// void Mesh::compute_smooth_normals(bool upload)
// {
// 	std::vector<std::vector<glm::vec4>> w_normals(positions.size());

// 	for (uint32_t f = 0; f < triangle_indices.size(); f += 3)
// 	{
// 		uint32_t i1 = triangle_indices[f + 0];
// 		uint32_t i2 = triangle_indices[f + 1];
// 		uint32_t i3 = triangle_indices[f + 2];

// 		// p1, p2 and p3 are the positions in the face (f)
// 		auto p1 = glm::vec3(positions[i1]);
// 		auto p2 = glm::vec3(positions[i2]);
// 		auto p3 = glm::vec3(positions[i3]);

// 		// calculate facet normal of the triangle  using cross product;
// 		// both components are "normalized" against a common point chosen as the base
// 		auto n = glm::cross((p2 - p1), (p3 - p1));    // p1 is the 'base' here

// 		// get the angle between the two other positions for each point;
// 		// the starting point will be the 'base' and the two adjacent positions will be normalized against it
// 		auto a1 = glm::angle(glm::normalize(p2 - p1), glm::normalize(p3 - p1));    // p1 is the 'base' here
// 		auto a2 = glm::angle(glm::normalize(p3 - p2), glm::normalize(p1 - p2));    // p2 is the 'base' here
// 		auto a3 = glm::angle(glm::normalize(p1 - p3), glm::normalize(p2 - p3));    // p3 is the 'base' here

// 		// normalize the initial facet normals if you want to ignore surface area
// 		// if (!area_weighting)
// 		// {
// 		//    n = glm::normalize(n);
// 		// }

// 		// store the weighted normal in an structured array
// 		auto wn1 = n * a1;
// 		auto wn2 = n * a2;
// 		auto wn3 = n * a3;
// 		w_normals[i1].push_back(glm::vec4(wn1.x, wn1.y, wn1.z, 0.f));
// 		w_normals[i2].push_back(glm::vec4(wn2.x, wn2.y, wn2.z, 0.f));
// 		w_normals[i3].push_back(glm::vec4(wn3.x, wn3.y, wn3.z, 0.f));
// 	}
// 	for (uint32_t v = 0; v < w_normals.size(); v++)
// 	{
// 		glm::vec4 N = glm::vec4(0.0);

// 		// run through the normals in each vertex's array and interpolate them
// 		// vertex(v) here fetches the data of the vertex at index 'v'
// 		for (uint32_t n = 0; n < w_normals[v].size(); n++)
// 		{
// 			N += w_normals[v][n];
// 		}

// 		// normalize the final normal
// 		normals[v] = glm::normalize(glm::vec4(N.x, N.y, N.z, 0.0f));
// 	}

// 	if (upload) {
// 		edit_normals(0, normals);
// 	}
// }

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
// 		tris.indexCount = this->get_total_triangle_indices();
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

/* Static Factory Implementations */
Mesh* Mesh::Get(std::string name) {
	return StaticFactory::Get(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
}

Mesh* Mesh::Get(uint32_t id) {
	return StaticFactory::Get(creation_mutex, id, "Mesh", lookupTable, meshes, MAX_MESHES);
}

Mesh* Mesh::CreateBox(std::string name)
{
	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
	try {
		generator::BoxMesh gen_mesh{{1, 1, 1}, {1, 1, 1}};
		mesh->generate_procedural(gen_mesh, /* flip z = */ false);
		Dirty = true;
		return mesh;
	} catch (...) {
		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
		throw;
	}
}

// Mesh* Mesh::CreateCappedCone(std::string name, bool allow_edits, bool submit_immediately, float radius, float height)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::CappedConeMesh gen_mesh{radius, height * .5};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateCappedCylinder(std::string name, float radius, float size, int slices, int segments, int rings, float start, float sweep, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {		
// 		generator::CappedCylinderMesh gen_mesh{radius, size, slices, segments, rings, start, sweep};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateCappedTube(std::string name, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::CappedTubeMesh gen_mesh{};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateCapsule(std::string name, float radius, float size, int slices, int segments, int rings, float start, float sweep, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::CapsuleMesh gen_mesh{radius, size, slices, segments, rings, start, sweep};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateCone(std::string name, bool allow_edits, bool submit_immediately, float radius, float height)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::ConeMesh gen_mesh{radius, height * .5};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreatePentagon(std::string name, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::ConvexPolygonMesh gen_mesh{};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateCylinder(std::string name, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::CylinderMesh gen_mesh{};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateDisk(std::string name, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::DiskMesh gen_mesh{};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateDodecahedron(std::string name, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::DodecahedronMesh gen_mesh{};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreatePlane(std::string name, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::PlaneMesh gen_mesh{{1, 1}, {1, 1}};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateIcosahedron(std::string name, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::IcosahedronMesh gen_mesh{};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateIcosphere(std::string name, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::IcoSphereMesh gen_mesh{};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// /* Might add this later. Requires a callback which defines a function mapping R2->R */
// // Mesh* Mesh::CreateParametricMesh(std::string name, uint32_t x_segments = 16, uint32_t y_segments = 16, bool allow_edits, bool submit_immediately)
// // {
// //     auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// //     if (!mesh) return nullptr;
// //     auto gen_mesh = generator::ParametricMesh( , glm::ivec2(x_segments, y_segments));
// //     mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// //     return mesh;
// // }

// Mesh* Mesh::CreateRoundedBox(std::string name, float radius, glm::vec3 size, int slices, glm::ivec3 segments, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::RoundedBoxMesh gen_mesh{
// 			radius, size, slices, segments
// 		};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateSphere(std::string name, float radius, int slices, int segments, float slice_start, float slice_sweep, float segment_start, float segment_sweep, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::SphereMesh gen_mesh{radius, slices, segments, slice_start, slice_sweep, segment_start, segment_sweep};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateSphericalCone(std::string name, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::SphericalConeMesh gen_mesh{};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateSphericalTriangle(std::string name, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::SphericalTriangleMesh gen_mesh{};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateSpring(std::string name, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::SpringMesh gen_mesh{};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateTeapotahedron(std::string name, uint32_t segments, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::TeapotMesh gen_mesh(segments);
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateTorus(std::string name, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::TorusMesh gen_mesh{};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateTorusKnot(std::string name, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::TorusKnotMesh gen_mesh{};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateTriangle(std::string name, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::TriangleMesh gen_mesh{};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateTube(std::string name, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		generator::TubeMesh gen_mesh{};
// 		mesh->generate_procedural(gen_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateTubeFromPolyline(std::string name, std::vector<glm::vec3> positions, float radius, uint32_t segments, bool allow_edits, bool submit_immediately)
// {
// 	if (positions.size() <= 1)
// 		throw std::runtime_error("Error: positions must be greater than 1!");
	
// 	using namespace generator;
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
		
// 		ParametricPath parametricPath {
// 			[positions](double t) {
// 				// t is 1.0 / positions.size() - 1 and goes from 0 to 1.0
// 				float t_scaled = (float)t * (((float)positions.size()) - 1.0f);
// 				uint32_t p1_idx = (uint32_t) floor(t_scaled);
// 				uint32_t p2_idx = p1_idx + 1;

// 				float t_segment = t_scaled - floor(t_scaled);

// 				glm::vec3 p1 = positions[p1_idx];
// 				glm::vec3 p2 = positions[p2_idx];

// 				PathVertex vertex;
				
// 				vertex.position = (p2 * t_segment) + (p1 * (1.0f - t_segment));

// 				glm::vec3 next = (p2 * (t_segment + .01f)) + (p1 * (1.0f - (t_segment + .01f)));
// 				glm::vec3 prev = (p2 * (t_segment - .01f)) + (p1 * (1.0f - (t_segment - .01f)));

// 				glm::vec3 tangent = glm::normalize(next - prev);
// 				glm::vec3 B1;
// 				glm::vec3 B2;
// 				buildOrthonormalBasis(tangent, B1, B2);
// 				vertex.tangent = tangent;
// 				vertex.normal = B1;
// 				vertex.texCoord = t;

// 				return vertex;
// 			},
// 			((int32_t) positions.size() - 1) // number of segments
// 		} ;
// 		CircleShape circle_shape(radius, segments);
// 		ExtrudeMesh extrude_mesh(circle_shape, parametricPath);
// 		mesh->generate_procedural(extrude_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateRoundedRectangleTubeFromPolyline(std::string name, std::vector<glm::vec3> positions, float radius, float size_x, float size_y, bool allow_edits, bool submit_immediately)
// {
// 	if (positions.size() <= 1)
// 		throw std::runtime_error("Error: positions must be greater than 1!");
	
// 	using namespace generator;
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
		
// 		ParametricPath parametricPath {
// 			[positions](double t) {
// 				// t is 1.0 / positions.size() - 1 and goes from 0 to 1.0
// 				float t_scaled = (float)t * ((float)(positions.size()) - 1.0f);
// 				uint32_t p1_idx = (uint32_t) floor(t_scaled);
// 				uint32_t p2_idx = p1_idx + 1;

// 				float t_segment = t_scaled - floor(t_scaled);

// 				glm::vec3 p1 = positions[p1_idx];
// 				glm::vec3 p2 = positions[p2_idx];

// 				PathVertex vertex;
				
// 				vertex.position = (p2 * t_segment) + (p1 * (1.0f - t_segment));

// 				glm::vec3 next = (p2 * (t_segment + .01f)) + (p1 * (1.0f - (t_segment + .01f)));
// 				glm::vec3 prev = (p2 * (t_segment - .01f)) + (p1 * (1.0f - (t_segment - .01f)));

// 				glm::vec3 tangent = glm::normalize(next - prev);
// 				glm::vec3 B1;
// 				glm::vec3 B2;
// 				buildOrthonormalBasis(tangent, B1, B2);
// 				vertex.tangent = tangent;
// 				vertex.normal = B1;
// 				vertex.texCoord = t;

// 				return vertex;
// 			},
// 			((int32_t) positions.size() - 1) // number of segments
// 		} ;
// 		RoundedRectangleShape rounded_rectangle_shape(radius, {size_x, size_y}, 4, {1, 1});
// 		ExtrudeMesh extrude_mesh(rounded_rectangle_shape, parametricPath);
// 		mesh->generate_procedural(extrude_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateRectangleTubeFromPolyline(std::string name, std::vector<glm::vec3> positions, float size_x, float size_y, bool allow_edits, bool submit_immediately)
// {
// 	if (positions.size() <= 1)
// 		throw std::runtime_error("Error: positions must be greater than 1!");
	
// 	using namespace generator;
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		ParametricPath parametricPath {
// 			[positions](double t) {
// 				// t is 1.0 / positions.size() - 1 and goes from 0 to 1.0
// 				float t_scaled = (float)t * ((float)(positions.size()) - 1.0f);
// 				uint32_t p1_idx = (uint32_t) floor(t_scaled);
// 				uint32_t p2_idx = p1_idx + 1;

// 				float t_segment = t_scaled - floor(t_scaled);

// 				glm::vec3 p1 = positions[p1_idx];
// 				glm::vec3 p2 = positions[p2_idx];

// 				PathVertex vertex;
				
// 				vertex.position = (p2 * t_segment) + (p1 * (1.0f - t_segment));

// 				glm::vec3 next = (p2 * (t_segment + .01f)) + (p1 * (1.0f - (t_segment + .01f)));
// 				glm::vec3 prev = (p2 * (t_segment - .01f)) + (p1 * (1.0f - (t_segment - .01f)));

// 				glm::vec3 tangent = glm::normalize(next - prev);
// 				glm::vec3 B1;
// 				glm::vec3 B2;
// 				buildOrthonormalBasis(tangent, B1, B2);
// 				vertex.tangent = tangent;
// 				vertex.normal = B1;
// 				vertex.texCoord = t;

// 				return vertex;
// 			},
// 			((int32_t) positions.size() - 1) // number of segments
// 		} ;
// 		RectangleShape rectangle_shape({size_x, size_y}, {1, 1});
// 		ExtrudeMesh extrude_mesh(rectangle_shape, parametricPath);
// 		mesh->generate_procedural(extrude_mesh, allow_edits, false, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }


// Mesh* Mesh::CreateFromOBJ(std::string name, std::string objPath, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		mesh->load_obj(objPath, allow_edits, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateFromSTL(std::string name, std::string stlPath, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		mesh->load_stl(stlPath, allow_edits, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateFromGLB(std::string name, std::string glbPath, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		mesh->load_glb(glbPath, allow_edits, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateFromTetgen(std::string name, std::string path, bool allow_edits, bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		mesh->load_tetgen(path, allow_edits, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

// Mesh* Mesh::CreateFromRaw (
// 	std::string name,
// 	std::vector<glm::vec4> positions, 
// 	std::vector<glm::vec4> normals, 
// 	std::vector<glm::vec4> colors, 
// 	std::vector<glm::vec2> texcoords, 
// 	std::vector<uint32_t> indices, 
// 	bool allow_edits, 
// 	bool submit_immediately)
// {
// 	auto mesh = StaticFactory::Create(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 	try {
// 		mesh->load_raw(positions, normals, colors, texcoords, indices, allow_edits, submit_immediately);
// 		Dirty = true;
// 		return mesh;
// 	} catch (...) {
// 		StaticFactory::DeleteIfExists(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
// 		throw;
// 	}
// }

void Mesh::Delete(std::string name) {
	StaticFactory::Delete(creation_mutex, name, "Mesh", lookupTable, meshes, MAX_MESHES);
	Dirty = true;
}

void Mesh::Delete(uint32_t id) {
	StaticFactory::Delete(creation_mutex, id, "Mesh", lookupTable, meshes, MAX_MESHES);
	Dirty = true;
}

MeshStruct* Mesh::GetFrontStruct()
{
	return mesh_structs;
}

Mesh* Mesh::GetFront() {
	return meshes;
}

uint32_t Mesh::GetCount() {
	return MAX_MESHES;
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

// 	vk::DeviceSize bufferSize = triangle_indices.size() * sizeof(uint32_t);
// 	triangleIndexBufferSize = bufferSize;
// 	vk::Buffer stagingBuffer;
// 	vk::DeviceMemory stagingBufferMemory;
// 	createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

// 	void *data = device.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags());
// 	memcpy(data, triangle_indices.data(), (size_t)bufferSize);
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
// void Mesh::show_bounding_box(bool should_show)
// {
// 	this->mesh_struct.show_bounding_box = should_show;
// }

// bool Mesh::should_show_bounding_box()
// {
// 	return this->mesh_struct.show_bounding_box;
// }