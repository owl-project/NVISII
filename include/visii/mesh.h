// ┌───────────────────────────────────────────────────────────────────────────┐
// │  Mesh Component                                                           │
// │                                                                           │
// └───────────────────────────────────────────────────────────────────────────┘
/* A mesh contains vertex information that has been loaded to the GPU. */

#pragma once

/* System includes */
#include <iostream>
#include <map>
#include <mutex>

/* External includes */
#include <glm/glm.hpp>
// #include <tiny_obj_loader.h>

/* Project includes */
// #include "Foton/Tools/Options.hxx"
#include <visii/utilities/static_factory.h>
#include <visii/mesh_struct.h>

/* Forward declarations */
class Vertex;

/* Class declaration */
class Mesh : public StaticFactory
{
	friend class StaticFactory;
	public:
		/* Creates a mesh component from a procedural box */
		static Mesh* createBox(std::string name);
		
		/* Creates a mesh component from a procedural cone capped on the bottom */
		static Mesh* createCappedCone(std::string name, float radius = 1.0, float height = 1.0);
		
		/* Creates a mesh component from a procedural cylinder capped on the bottom */
		static Mesh* createCappedCylinder(std::string name, float radius = 1.0f, float size = 1.f, int slices = 32, int segments = 1, int rings = 1, float start = 0.0f, float sweep = 6.28319f);
		
		/* Creates a mesh component from a procedural tube capped on both ends */
		static Mesh* createCappedTube(std::string name);
		
		/* Creates a mesh component from a procedural capsule */
		static Mesh* createCapsule(std::string name, float radius = 1.0, float size = 0.5, int slices = 32, int segments = 4, int rings = 8, float start = 0.0, float sweep = 6.28319f);
		
		/* Creates a mesh component from a procedural cone */
		static Mesh* createCone(std::string name, float radius = 1.0, float height = 1.0);
		
		/* Creates a mesh component from a procedural pentagon */
		static Mesh* createPentagon(std::string name);
		
		/* Creates a mesh component from a procedural cylinder (uncapped) */
		static Mesh* createCylinder(std::string name);

		/* Creates a mesh component from a procedural disk */
		static Mesh* createDisk(std::string name);

		/* Creates a mesh component from a procedural dodecahedron */
		static Mesh* createDodecahedron(std::string name);

		/* Creates a mesh component from a procedural plane */
		static Mesh* createPlane(std::string name);

		/* Creates a mesh component from a procedural icosahedron */
		static Mesh* createIcosahedron(std::string name);

		/* Creates a mesh component from a procedural icosphere */
		static Mesh* createIcosphere(std::string name);

		/* Creates a mesh component from a procedural parametric mesh. (TODO: accept a callback which given an x and y position 
			returns a Z hightfield) */
		// static Mesh* createParametricMesh(std::string name);

		/* Creates a mesh component from a procedural box with rounded edges */
		static Mesh* createRoundedBox(std::string name, float radius = .25, glm::vec3 size = glm::vec3(.75f, .75f, .75f), int slices=4, glm::ivec3 segments=glm::ivec3(1, 1, 1));
	
		/* Creates a mesh component from a procedural sphere */
		static Mesh* createSphere(std::string name, float radius = 1.0f, int slices = 16, int segments = 16, float slice_start = 0.f, float sliceSweep = 6.28319f, float segmentStart = 0.f, float segmentSweep = 6.28319f);

		/* Creates a mesh component from a procedural cone with a rounded cap */
		static Mesh* createSphericalCone(std::string name);

		/* Creates a mesh component from a procedural quarter-hemisphere */
		static Mesh* createSphericalTriangle(std::string name);

		/* Creates a mesh component from a procedural spring */
		static Mesh* createSpring(std::string name);

		/* Creates a mesh component from a procedural utah teapot */
		static Mesh* createTeapotahedron(std::string name, uint32_t segments = 8);

		/* Creates a mesh component from a procedural torus */
		static Mesh* createTorus(std::string name);

		/* Creates a mesh component from a procedural torus knot */
		static Mesh* createTorusKnot(std::string name);

		/* Creates a mesh component from a procedural triangle */
		static Mesh* createTriangle(std::string name);

		/* Creates a mesh component from a procedural tube (uncapped) */
		static Mesh* createTube(std::string name);

		/* Creates a mesh component from a procedural tube (uncapped) generated from a polyline */
		static Mesh* createTubeFromPolyline(std::string name, std::vector<glm::vec3> positions, float radius = 1.0, uint32_t segments = 16);

		/* Creates a mesh component from a procedural rounded rectangle tube (uncapped) generated from a polyline */
		static Mesh* createRoundedRectangleTubeFromPolyline(std::string name, std::vector<glm::vec3> positions, float radius = 1.0, float sizeX = .75, float sizeY = .75);

		/* Creates a mesh component from a procedural rectangle tube (uncapped) generated from a polyline */
		static Mesh* createRectangleTubeFromPolyline(std::string name, std::vector<glm::vec3> positions, float sizeX = 1.0, float sizeY = 1.0);

		// /* Creates a mesh component from an OBJ file (ignores .mtl files) */
		// static Mesh* createFromObj(std::string name, std::string objPath);

		// /* Creates a mesh component from an ASCII STL file */
		// static Mesh* createFromStl(std::string name, std::string stlPath);

		// /* Creates a mesh component from a GLB file (material properties are ignored) */
		// static Mesh* createFromGlb(std::string name, std::string glbPath);

		// /* Creates a mesh component from TetGen node/element files (Can be made using "Mesh::tetrahedrahedralize") */
		// static Mesh* createFromTetgen(std::string name, std::string path);

		// /* Creates a mesh component from a set of positions, optional normals, optional colors, optional texture coordinates, 
		// 	and optional indices. If anything other than positions is supplied (eg normals), that list must be the same length
		// 	as the point list. If indicies are supplied, indices must be a multiple of 3 (triangles). Otherwise, all other
		// 	supplied per vertex data must be a multiple of 3 in length. */
		// static Mesh* createFromRaw(
		// 	std::string name,
		// 	std::vector<glm::vec4> positions, 
		// 	std::vector<glm::vec4> normals = std::vector<glm::vec4>(), 
		// 	std::vector<glm::vec4> colors = std::vector<glm::vec4>(), 
		// 	std::vector<glm::vec2> texcoords = std::vector<glm::vec2>(), 
		// 	std::vector<uint32_t> indices = std::vector<uint32_t>(),
		// 	bool allow_edits = false, bool submit_immediately = false);

        /** Gets a mesh by name 
         * \returns a mesh who's primary name key matches \p name 
         * \param name A unique name used to lookup this mesh. */
        static Mesh* get(std::string name);

        /** Gets a mesh by id 
         * \returns a mesh who's primary id key matches \p id 
         * \param id A unique id used to lookup this mesh. */
        static Mesh* get(uint32_t id);

        /** \returns a pointer to the table of MeshStructs required for rendering */
        static MeshStruct* getFrontStruct();

        /** \returns a pointer to the table of mesh components */
        static Mesh* getFront();

        /** \returns the number of allocated meshes */
        static uint32_t getCount();

        /** Deletes the mesh who's primary name key matches \p name 
         * \param name A unique name used to lookup the mesh for deletion.*/
        static void remove(std::string name);

        /** Deletes the mesh who's primary id key matches \p id 
         * \param id A unique id used to lookup the mesh for deletion.*/
        static void remove(uint32_t id);

        /** Allocates the tables used to store all mesh components */
        static void initializeFactory();

        /** \return True if the tables used to store all mesh components have been allocated, and False otherwise */
        static bool isFactoryInitialized();

        /** Iterates through all mesh components, computing mesh metadata for rendering purposes. */
        static void updateComponents();

        /** Frees any tables used to store mesh components */
        static void cleanUp();

        /** \return True if the mesh has been modified since the previous frame, and False otherwise */
        bool isDirty() { return dirty; }

        /** \return True if the mesh has not been modified since the previous frame, and False otherwise */
        bool isClean() { return !dirty; }

        /** Tags the current component as being modified since the previous frame. */
        void markDirty() {
            // Dirty = true;
            dirty = true;
        };

        /** Tags the current component as being unmodified since the previous frame. */
        void markClean() { dirty = false; }
		
		// /* TODO EXPLAIN THIS*/
		// static std::vector<vk::Buffer> GetPositionSSBOs();
		
		// /* TODO EXPLAIN THIS*/
		// static std::vector<uint32_t> GetPositionSSBOSizes();
		
		// /* TODO EXPLAIN THIS*/
		// static std::vector<vk::Buffer> GetNormalSSBOs();
		
		// /* TODO EXPLAIN THIS*/
		// static std::vector<uint32_t> GetNormalSSBOSizes();
		
		// /* TODO EXPLAIN THIS*/
		// static std::vector<vk::Buffer> GetColorSSBOs();
		
		// /* TODO EXPLAIN THIS*/
		// static std::vector<uint32_t> GetColorSSBOSizes();
		
		// /* TODO EXPLAIN THIS*/
		// static std::vector<vk::Buffer> GetTexCoordSSBOs();
		
		// /* TODO EXPLAIN THIS*/
		// static std::vector<uint32_t> GetTexCoordSSBOSizes();
		
		// /* TODO EXPLAIN THIS*/
		// static std::vector<vk::Buffer> GetIndexSSBOs();
		
		// /* TODO EXPLAIN THIS*/
		// static std::vector<uint32_t> GetIndexSSBOSizes();

		/** Returns a json string representation of the current component */
		std::string toString();
		
		// /* If editing is enabled, returns a list of per vertex positions */
		// std::vector<glm::vec4> get_positions();

		// /* If editing is enabled, returns a list of per vertex colors */
		// std::vector<glm::vec4> get_colors();

		// /* If editing is enabled, returns a list of per vertex normals */
		// std::vector<glm::vec4> get_normals();

		// /* If editing is enabled, returns a list of per vertex texture coordinates */
		// std::vector<glm::vec2> get_texcoords();

		// /* If editing is enabled, returns a list of edge indices */
		// std::vector<uint32_t> get_edge_indices();

		// /* If editing is enabled, returns a list of triangle indices */
		// std::vector<uint32_t> get_triangle_indices();

		// /* If editing is enabled, returns a list of tetrahedra indices */
		// std::vector<uint32_t> get_tetrahedra_indices();		

		// /* Returns the handle to the position buffer */
		// vk::Buffer get_point_buffer();

		// /* Returns the handle to the per vertex colors buffer */
		// vk::Buffer get_color_buffer();

		// /* Returns the handle to the triangle indices buffer */
		// vk::Buffer get_triangle_index_buffer();

		// /* Returns the handle to the per vertex normals buffer */
		// vk::Buffer get_normal_buffer();

		// /* Returns the handle to the per vertex texcoords buffer */
		// vk::Buffer get_texcoord_buffer();

		// /* Returns the total number of edge indices used by this mesh. 
		// 	Divide by 2 to get the number of edges.  */
		// uint32_t get_total_edge_indices();

		// /* Returns the total number of triangle indices used by this mesh. 
		// 	Divide by 3 to get the number of triangles.  */
		// uint32_t get_total_triangle_indices();

		// /* Returns the total number of tetrahedral indices used by this mesh. 
		// 	Divide by 4 to get the number of tetrahedra.  */
		// uint32_t get_total_tetrahedra_indices();

		// /* Returns the total number of bytes per index */
		// uint32_t get_index_bytes();

		/* Computes the average of all vertex positions. (centroid) 
			as well as min/max bounds and bounding sphere data. */
		void computeMetadata();

		// /* TODO: Explain this */
		// void save_tetrahedralization(float quality_bound, float maximum_volume);

		// /* Returns the last computed centroid. */
		// glm::vec3 get_centroid();

		// /* Returns the minimum axis aligned bounding box position */
		// glm::vec3 get_min_aabb_corner();

		// /* Returns the maximum axis aligned bounding box position */
		// glm::vec3 get_max_aabb_corner();

		// /* Returns the radius of a sphere centered at the centroid which completely contains the mesh */
		// float get_bounding_sphere_radius();

		// /* If mesh editing is enabled, replaces the position at the given index with a new position */
		// void edit_position(uint32_t index, glm::vec4 new_position);

		// /* If mesh editing is enabled, replaces the set of positions starting at the given index with a new set of positions */
		// void edit_positions(uint32_t index, std::vector<glm::vec4> new_positions);

		// /* If mesh editing is enabled, replaces the normal at the given index with a new normal */
		// void edit_normal(uint32_t index, glm::vec4 new_normal);

		// /* If mesh editing is enabled, replaces the set of normals starting at the given index with a new set of normals */
		// void edit_normals(uint32_t index, std::vector<glm::vec4> new_normals);

		// /* TODO: EXPLAIN THIS */
		// void compute_smooth_normals(bool upload = true);

		// /* If mesh editing is enabled, replaces the vertex color at the given index with a new vertex color */
		// void edit_vertex_color(uint32_t index, glm::vec4 new_color);

		// /* If mesh editing is enabled, replaces the set of vertex colors starting at the given index with a new set of vertex colors */
		// void edit_vertex_colors(uint32_t index, std::vector<glm::vec4> new_colors);

		// /* If mesh editing is enabled, replaces the texture coordinate at the given index with a new texture coordinate */
		// void edit_texture_coordinate(uint32_t index, glm::vec2 new_texcoord);

		// /* If mesh editing is enabled, replaces the set of texture coordinates starting at the given index with a new set of texture coordinates */
		// void edit_texture_coordinates(uint32_t index, std::vector<glm::vec2> new_texcoords);
		
		// /* If RTX Raytracing is enabled, builds a low level BVH for this mesh. */
		// void build_low_level_bvh(bool submit_immediately = false);

		// /* TODO */
		// vk::AccelerationStructureNV get_low_level_bvh();
		
		// uint64_t get_low_level_bvh_handle();

		// vk::GeometryNV get_nv_geometry();

		// /* TODO */
		// void show_bounding_box(bool should_show);

		// /* TODO */
		// bool should_show_bounding_box();

	private:
		/* Creates an uninitialized mesh. Useful for preallocation. */
		Mesh();

		/* Creates a mesh with the given name and id. */
		Mesh(std::string name, uint32_t id);

		/* TODO */
		static std::shared_ptr<std::mutex> creationMutex;
		
		/* TODO */
		static bool factoryInitialized;
		
		/* A list of the mesh components, allocated statically */
		static Mesh meshes[MAX_MESHES];
		static MeshStruct meshStructs[MAX_MESHES];

		/* A lookup table of name to mesh id */
		static std::map<std::string, uint32_t> lookupTable;

		// /* Lists of per vertex data. These might not match GPU memory if editing is disabled. */
		std::vector<glm::vec4> positions;
		std::vector<glm::vec4> normals;
		std::vector<glm::vec4> colors;
		std::vector<glm::vec2> texCoords;
		// std::vector<uint32_t> tetrahedra_indices;
		std::vector<uint32_t> triangleIndices;
		// std::vector<uint32_t> edge_indices;

		// /* A handle to the attributes loaded from tiny obj */
		// tinyobj::attrib_t attrib;

		// /* A handle to the buffer containing per vertex positions */
		// vk::Buffer pointBuffer;
		// vk::DeviceMemory pointBufferMemory;
		// uint32_t pointBufferSize;

		// /* A handle to the buffer containing per vertex colors */
		// vk::Buffer colorBuffer;
		// vk::DeviceMemory colorBufferMemory;
		// uint32_t colorBufferSize;

		// /* A handle to the buffer containing per vertex normals */
		// vk::Buffer normalBuffer;
		// vk::DeviceMemory normalBufferMemory;
		// uint32_t normalBufferSize;

		// /* A handle to the buffer containing per vertex texture coordinates */
		// vk::Buffer texCoordBuffer;
		// vk::DeviceMemory texCoordBufferMemory;
		// uint32_t texCoordBufferSize;

		// /* A handle to the buffer containing triangle indices */		
		// vk::Buffer triangleIndexBuffer;
		// vk::DeviceMemory triangleIndexBufferMemory;
		// uint32_t triangleIndexBufferSize;

		// /* An RTX geometry handle */
		// vk::GeometryNV geometry;

		// /* An RTX handle to the low level acceleration structure */
		// vk::AccelerationStructureNV lowAS;
		// vk::DeviceMemory lowASMemory;
		// uint64_t ASHandle;

		// /* True if the low level BVH was built. (TODO: make false if mesh edits were submitted) */
		// bool lowBVHBuilt = false;
		
		// /* True if this mesh component supports editing. If false, indices are automatically generated. */
		// bool allowEdits = false;

		/* Frees any resources this mesh component may have allocated */
		// void cleanup();

		// /* Creates a generic vertex buffer object */
		// uint64_t createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer &buffer, vk::DeviceMemory &bufferMemory);

		// /* Creates a position buffer, and uploads position data stored in the positions list */
		// void createPointBuffer(bool allow_edits, bool submit_immediately);

		// /* Creates a per vertex color buffer, and uploads per vertex color data stored in the colors list */
		// void createColorBuffer(bool allow_edits, bool submit_immediately);

		// /* Creates a normal buffer, and uploads normal data stored in the normals list */
		// void createNormalBuffer(bool allow_edits, bool submit_immediately);

		// /* Creates a texture coordinate buffer, and uploads texture coordinate data stored in the texture coordinates list */
		// void createTexCoordBuffer(bool allow_edits, bool submit_immediately);

		// /* Creates an index buffer, and uploads index data stored in the indices list */
		// void createTriangleIndexBuffer(bool allow_edits, bool submit_immediately);

		// /* Loads in an OBJ mesh and copies per vertex data to the GPU */
		// void load_obj(std::string objPath, bool allow_edits, bool submit_immediately);

		// /* Loads in an STL mesh and copies per vertex data to the GPU */
		// void load_stl(std::string stlPath, bool allow_edits, bool submit_immediately);

		// /* Loads in a GLB mesh and copies per vertex data to the GPU */
		// void load_glb(std::string glbPath, bool allow_edits, bool submit_immediately);

		// /* TODO: Explain this */
		// void load_tetgen(std::string path, bool allow_edits, bool submit_immediately);

		// /* Copies per vertex data to the GPU */
		// void load_raw (
		// 	std::vector<glm::vec4> &positions, 
		// 	std::vector<glm::vec4> &normals, 
		// 	std::vector<glm::vec4> &colors, 
		// 	std::vector<glm::vec2> &texcoords,
		// 	std::vector<uint32_t> indices,
		// 	bool allow_edits,
		// 	bool submit_immediately
		// );
		
		/* Creates a procedural mesh from the given mesh generator, and copies per vertex to the GPU */
		template <class Generator>
		void generateProcedural(Generator &mesh, bool flip_z)
		{
			std::vector<Vertex> vertices;

			auto genVerts = mesh.vertices();
			while (!genVerts.done()) {
				auto vertex = genVerts.generate();
				positions.push_back(glm::vec4(vertex.position.x, vertex.position.y, vertex.position.z, 1.0f));
				if (flip_z)
					normals.push_back(glm::vec4(-vertex.normal.x, -vertex.normal.y, -vertex.normal.z, 0.0f));
				else
					normals.push_back(glm::vec4(vertex.normal.x, vertex.normal.y, vertex.normal.z, 0.0f));
				texCoords.push_back(vertex.texCoord);
				colors.push_back(glm::vec4(0.0, 0.0, 0.0, 0.0));
				genVerts.next();
			}

			auto genTriangles = mesh.triangles();
			while (!genTriangles.done()) {
				auto triangle = genTriangles.generate();
				triangleIndices.push_back(triangle.vertices[0]);
				triangleIndices.push_back(triangle.vertices[1]);
				triangleIndices.push_back(triangle.vertices[2]);
				genTriangles.next();
			}

			// cleanup();
			// createPointBuffer();
			// createColorBuffer();
			// createNormalBuffer();
			// createTexCoordBuffer();
			// createTriangleIndexBuffer();
			computeMetadata();
		}

		/* Indicates that one of the components has been edited */
		static bool anyDirty;

		/* Indicates this component has been edited */
		bool dirty = true;
};
