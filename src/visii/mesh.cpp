#include <sys/types.h>
#include <sys/stat.h>
#include <functional>
#include <limits>
#include <fcntl.h>
#include <unordered_map>
#ifndef WIN32
#include <unistd.h>
#endif

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#define GLM_FORCE_RIGHT_HANDED

#include <glm/gtx/vector_angle.hpp>

#include <visii/mesh.h>
#include <visii/entity.h>
#include <visii/utilities/hash_combiner.h>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <generator/generator.hpp>

std::vector<Mesh> Mesh::meshes;
std::vector<MeshStruct> Mesh::meshStructs;
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

std::vector<uint32_t> Mesh::getTriangleIndices() {
	return triangleIndices;
}

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

void Mesh::initializeFactory(uint32_t max_components) {
	if (isFactoryInitialized()) return;
	meshes.resize(max_components);
	meshStructs.resize(max_components);
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

std::shared_ptr<std::recursive_mutex> Mesh::getEditMutex()
{
	return editMutex;
}

/* Static Factory Implementations */
Mesh* Mesh::get(std::string name) {
	return StaticFactory::get(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
}

Mesh* Mesh::createBox(std::string name, glm::vec3 size, glm::ivec3 segments)
{
	auto create = [&] (Mesh* mesh) {
		dirtyMeshes.insert(mesh);
		generator::BoxMesh gen_mesh{size, segments};
		mesh->generateProcedural(gen_mesh, /* flip z = */ false);
	};

	try {
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
		throw;
	}
}

/* Might add this later. Requires a callback which defines a function mapping R2->R */
// Mesh* Mesh::createParametricMesh(std::string name, uint32_t x_segments = 16, uint32_t y_segments = 16)
// {
//     if (!mesh) return nullptr;
//     auto gen_mesh = generator::ParametricMesh( , glm::ivec2(x_segments, y_segments));
//     mesh->generateProcedural(gen_mesh, /* flip z = */ false);
		// return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
		throw;
	}
}


Mesh* Mesh::createFromObj(std::string name, std::string path)
{
	static bool createFromFileDeprecatedShown = false;
    if (createFromFileDeprecatedShown == false) {
        std::cout<<"Warning, create_from_obj is deprecated and will be removed in a subsequent release. Please switch to create_from_file." << std::endl;
        createFromFileDeprecatedShown = true;
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
			aiProcessPreset_TargetRealtime_Fast | 
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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
		throw;
	}
}

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
		return StaticFactory::create<Mesh>(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size(), create);
	} catch (...) {
		StaticFactory::removeIfExists(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
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
	StaticFactory::remove(editMutex, name, "Mesh", lookupTable, meshes.data(), meshes.size());
	dirtyMeshes.insert(&meshes[oldID]);
}

MeshStruct* Mesh::getFrontStruct()
{
	return meshStructs.data();
}

Mesh* Mesh::getFront() {
	return meshes.data();
}

uint32_t Mesh::getCount() {
	return meshes.size();
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
	return (this - meshes.data());
}

std::map<std::string, uint32_t> Mesh::getNameToIdMap()
{
	return lookupTable;
}
