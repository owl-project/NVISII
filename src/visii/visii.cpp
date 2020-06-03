#include <visii/visii.h>

#include <glfw_implementation/glfw.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <ImGuizmo.h>
#include <visii/utilities/colors.h>
#include <owl/owl.h>
#include <cuda_gl_interop.h>

#include <devicecode/launch_params.h>
#include <devicecode/path_tracer.h>

#include <thread>
#include <future>
#include <queue>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

// #define __optix_optix_function_table_h__
#include <optix_stubs.h>
// OptixFunctionTable g_optixFunctionTable;

// extern optixDenoiserSetModel;
std::promise<void> exitSignal;
std::thread renderThread;
static bool initialized = false;
static bool close = true;

static struct WindowData {
    GLFWwindow* window = nullptr;
    ivec2 currentSize, lastSize;
} WindowData;

/* Embedded via cmake */
extern "C" char ptxCode[];

struct MeshData {
    OWLBuffer vertices;
    OWLBuffer colors;
    OWLBuffer normals;
    OWLBuffer texCoords;
    OWLBuffer indices;
    OWLGeom geom;
    OWLGroup blas;
};

static struct OptixData {
    OWLContext context;
    OWLModule module;
    OWLLaunchParams launchParams;
    LaunchParams LP;
    GLuint imageTexID = -1;
    cudaGraphicsResource_t cudaResourceTex;
    OWLBuffer frameBuffer;
    OWLBuffer accumBuffer;

    OWLBuffer entityBuffer;
    OWLBuffer transformBuffer;
    OWLBuffer cameraBuffer;
    OWLBuffer materialBuffer;
    OWLBuffer meshBuffer;
    OWLBuffer lightBuffer;
    OWLBuffer lightEntitiesBuffer;
    OWLBuffer instanceToEntityMapBuffer;
    OWLBuffer vertexListsBuffer;
    OWLBuffer indexListsBuffer;

    uint32_t numLightEntities;

    OWLRayGen rayGen;
    OWLMissProg missProg;
    OWLGeomType trianglesGeomType;
    MeshData meshes[MAX_MESHES];
    OWLGroup tlas;

    std::vector<uint32_t> lightEntities;

    cudaStream_t stream;
    OptixDenoiser denoiser;
    OWLBuffer denoiserScratchBuffer;
    OWLBuffer denoiserStateBuffer;
} OptixData;

static struct ViSII {
    struct Command {
        std::function<void()> function;
        std::shared_ptr<std::promise<void>> promise;
    };

    std::thread::id render_thread_id;
    std::condition_variable cv;
    std::mutex qMutex;
    std::queue<Command> commandQueue = {};
    bool headlessMode;
} ViSII;

void applyStyle()
{
	ImGuiStyle* style = &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	colors[ImGuiCol_Text]                   = ImVec4(1.000f, 1.000f, 1.000f, 1.000f);
	colors[ImGuiCol_TextDisabled]           = ImVec4(0.500f, 0.500f, 0.500f, 1.000f);
	colors[ImGuiCol_WindowBg]               = ImVec4(0.180f, 0.180f, 0.180f, 1.000f);
	colors[ImGuiCol_ChildBg]                = ImVec4(0.280f, 0.280f, 0.280f, 0.000f);
	colors[ImGuiCol_PopupBg]                = ImVec4(0.313f, 0.313f, 0.313f, 1.000f);
	colors[ImGuiCol_Border]                 = ImVec4(0.266f, 0.266f, 0.266f, 1.000f);
	colors[ImGuiCol_BorderShadow]           = ImVec4(0.000f, 0.000f, 0.000f, 0.000f);
	colors[ImGuiCol_FrameBg]                = ImVec4(0.160f, 0.160f, 0.160f, 1.000f);
	colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.200f, 0.200f, 0.200f, 1.000f);
	colors[ImGuiCol_FrameBgActive]          = ImVec4(0.280f, 0.280f, 0.280f, 1.000f);
	colors[ImGuiCol_TitleBg]                = ImVec4(0.148f, 0.148f, 0.148f, 1.000f);
	colors[ImGuiCol_TitleBgActive]          = ImVec4(0.148f, 0.148f, 0.148f, 1.000f);
	colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.148f, 0.148f, 0.148f, 1.000f);
	colors[ImGuiCol_MenuBarBg]              = ImVec4(0.195f, 0.195f, 0.195f, 1.000f);
	colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.160f, 0.160f, 0.160f, 1.000f);
	colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.277f, 0.277f, 0.277f, 1.000f);
	colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.300f, 0.300f, 0.300f, 1.000f);
	colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_CheckMark]              = ImVec4(1.000f, 1.000f, 1.000f, 1.000f);
	colors[ImGuiCol_SliderGrab]             = ImVec4(0.391f, 0.391f, 0.391f, 1.000f);
	colors[ImGuiCol_SliderGrabActive]       = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_Button]                 = ImVec4(1.000f, 1.000f, 1.000f, 0.000f);
	colors[ImGuiCol_ButtonHovered]          = ImVec4(1.000f, 1.000f, 1.000f, 0.156f);
	colors[ImGuiCol_ButtonActive]           = ImVec4(1.000f, 1.000f, 1.000f, 0.391f);
	colors[ImGuiCol_Header]                 = ImVec4(0.313f, 0.313f, 0.313f, 1.000f);
	colors[ImGuiCol_HeaderHovered]          = ImVec4(0.469f, 0.469f, 0.469f, 1.000f);
	colors[ImGuiCol_HeaderActive]           = ImVec4(0.469f, 0.469f, 0.469f, 1.000f);
	colors[ImGuiCol_Separator]              = colors[ImGuiCol_Border];
	colors[ImGuiCol_SeparatorHovered]       = ImVec4(0.391f, 0.391f, 0.391f, 1.000f);
	colors[ImGuiCol_SeparatorActive]        = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_ResizeGrip]             = ImVec4(1.000f, 1.000f, 1.000f, 0.250f);
	colors[ImGuiCol_ResizeGripHovered]      = ImVec4(1.000f, 1.000f, 1.000f, 0.670f);
	colors[ImGuiCol_ResizeGripActive]       = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_Tab]                    = ImVec4(0.098f, 0.098f, 0.098f, 1.000f);
	colors[ImGuiCol_TabHovered]             = ImVec4(0.352f, 0.352f, 0.352f, 1.000f);
	colors[ImGuiCol_TabActive]              = ImVec4(0.195f, 0.195f, 0.195f, 1.000f);
	colors[ImGuiCol_TabUnfocused]           = ImVec4(0.098f, 0.098f, 0.098f, 1.000f);
	colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(0.195f, 0.195f, 0.195f, 1.000f);
	// colors[ImGuiCol_DockingPreview]         = ImVec4(1.000f, 0.391f, 0.000f, 0.781f);
	// colors[ImGuiCol_DockingEmptyBg]         = ImVec4(0.180f, 0.180f, 0.180f, 1.000f);
	colors[ImGuiCol_PlotLines]              = ImVec4(0.469f, 0.469f, 0.469f, 1.000f);
	colors[ImGuiCol_PlotLinesHovered]       = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_PlotHistogram]          = ImVec4(0.586f, 0.586f, 0.586f, 1.000f);
	colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_TextSelectedBg]         = ImVec4(1.000f, 1.000f, 1.000f, 0.156f);
	colors[ImGuiCol_DragDropTarget]         = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_NavHighlight]           = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.000f, 0.000f, 0.000f, 0.586f);
	colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.000f, 0.000f, 0.000f, 0.586f);

	style->ChildRounding = 4.0f;
	style->FrameBorderSize = 1.0f;
	style->FrameRounding = 2.0f;
	style->GrabMinSize = 7.0f;
	style->PopupRounding = 2.0f;
	style->ScrollbarRounding = 12.0f;
	style->ScrollbarSize = 13.0f;
	style->TabBorderSize = 1.0f;
	style->TabRounding = 0.0f;
	style->WindowRounding = 4.0f;
}

void resetAccumulation() {
    OptixData.LP.frameID = 0;
}

void setCameraEntity(Entity* camera_entity)
{
    if (!camera_entity) throw std::runtime_error("Error: camera entity was nullptr/None");
    if (!camera_entity->isInitialized()) throw std::runtime_error("Error: camera entity is uninitialized");

    OptixData.LP.cameraEntity = camera_entity->getStruct();
    resetAccumulation();
}

void setDomeLightIntensity(float intensity)
{
    intensity = std::max(float(intensity), float(0.f));
    OptixData.LP.domeLightIntensity = intensity;
    resetAccumulation();
}

void initializeFrameBuffer(int fbWidth, int fbHeight) {
    auto &OD = OptixData;
    if (OD.imageTexID != -1) {
        cudaGraphicsUnregisterResource(OD.cudaResourceTex);
    }
    
    // Enable Texturing
    glEnable(GL_TEXTURE_2D);
    // Generate a Texture ID for the framebuffer
    glGenTextures(1, &OD.imageTexID);
    // Make this teh current texture
    glBindTexture(GL_TEXTURE_2D, OD.imageTexID);
    // Allocate the texture memory. The last parameter is NULL since we only 
    // want to allocate memory, not initialize it.
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, fbWidth, fbHeight);
    
    // Must set the filter mode, GL_LINEAR enables interpolation when scaling
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    //Registration with CUDA
    cudaGraphicsGLRegisterImage(&OD.cudaResourceTex, OD.imageTexID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
}

void resizeOptixFrameBuffer(uint32_t width, uint32_t height)
{
    OptixData.LP.frameSize.x = width;
    OptixData.LP.frameSize.y = height;
    owlBufferResize(OptixData.frameBuffer, width * height);
    owlBufferResize(OptixData.accumBuffer, width * height);
    resetAccumulation();
}

void updateFrameBuffer()
{
    glfwGetFramebufferSize(WindowData.window, &WindowData.currentSize.x, &WindowData.currentSize.y);

    if ((WindowData.currentSize.x != WindowData.lastSize.x) || (WindowData.currentSize.y != WindowData.lastSize.y))  {
        WindowData.lastSize.x = WindowData.currentSize.x; WindowData.lastSize.y = WindowData.currentSize.y;
        initializeFrameBuffer(WindowData.currentSize.x, WindowData.currentSize.y);
        resizeOptixFrameBuffer(WindowData.currentSize.x, WindowData.currentSize.y);
        resetAccumulation();
    }
}


void initializeOptix(bool headless)
{
    using namespace glm;
    auto &OD = OptixData;
    OD.context = owlContextCreate(/*requested Device IDs*/ nullptr, /* Num Devices */ 0);
    // owlContextSetRayTypeCount(context, 2); // for both "feeler" and query rays on the same accel.
    OD.module = owlModuleCreate(OD.context, ptxCode);
    
    /* Setup Optix Launch Params */
    OWLVarDecl launchParamVars[] = {
        { "frameSize",           OWL_USER_TYPE(glm::ivec2),         OWL_OFFSETOF(LaunchParams, frameSize)},
        { "frameID",             OWL_USER_TYPE(uint64_t),           OWL_OFFSETOF(LaunchParams, frameID)},
        { "fbPtr",               OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, fbPtr)},
        { "accumPtr",            OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, accumPtr)},
        { "world",               OWL_GROUP,                         OWL_OFFSETOF(LaunchParams, world)},
        { "cameraEntity",        OWL_USER_TYPE(EntityStruct),       OWL_OFFSETOF(LaunchParams, cameraEntity)},
        { "entities",            OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, entities)},
        { "transforms",          OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, transforms)},
        { "cameras",             OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, cameras)},
        { "materials",           OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, materials)},
        { "meshes",              OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, meshes)},
        { "lights",              OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, lights)},
        { "lightEntities",       OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, lightEntities)},
        { "vertexLists",         OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, vertexLists)},
        { "indexLists",          OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, indexLists)},
        { "numLightEntities",    OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, numLightEntities)},
        { "instanceToEntityMap", OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, instanceToEntityMap)},
        { "domeLightIntensity",  OWL_USER_TYPE(float),              OWL_OFFSETOF(LaunchParams, domeLightIntensity)},
        { /* sentinel to mark end of list */ }
    };
    OD.launchParams = owlLaunchParamsCreate(OD.context, sizeof(LaunchParams), launchParamVars, -1);
    
    /* Create AOV Buffers */
    if (!headless) {
        initializeFrameBuffer(512, 512);
    }

    OD.frameBuffer = owlManagedMemoryBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
    OD.accumBuffer = owlDeviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
    OD.LP.frameSize = glm::ivec2(512, 512);
    owlLaunchParamsSetBuffer(OD.launchParams, "fbPtr", OD.frameBuffer);
    owlLaunchParamsSetBuffer(OD.launchParams, "accumPtr", OD.accumBuffer);
    owlLaunchParamsSetRaw(OD.launchParams, "frameSize", &OD.LP.frameSize);

    /* Create Component Buffers */
    OD.entityBuffer    = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(EntityStruct),    MAX_ENTITIES,   nullptr);
    OD.transformBuffer = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(TransformStruct), MAX_TRANSFORMS, nullptr);
    OD.cameraBuffer    = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(CameraStruct),    MAX_CAMERAS,    nullptr);
    OD.materialBuffer  = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(MaterialStruct),  MAX_MATERIALS,  nullptr);
    OD.meshBuffer      = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(MeshStruct),      MAX_MESHES,     nullptr);
    OD.lightBuffer     = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(LightStruct),     MAX_LIGHTS,     nullptr);
    OD.lightEntitiesBuffer            = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(uint32_t),        1,     nullptr);
    OD.instanceToEntityMapBuffer      = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(uint32_t),        1,     nullptr);
    OD.vertexListsBuffer = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(vec4*), MAX_MESHES, nullptr);
    OD.indexListsBuffer =  owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(ivec3*), MAX_MESHES, nullptr);
    owlLaunchParamsSetBuffer(OD.launchParams, "entities",   OD.entityBuffer);
    owlLaunchParamsSetBuffer(OD.launchParams, "transforms", OD.transformBuffer);
    owlLaunchParamsSetBuffer(OD.launchParams, "cameras",    OD.cameraBuffer);
    owlLaunchParamsSetBuffer(OD.launchParams, "materials",  OD.materialBuffer);
    owlLaunchParamsSetBuffer(OD.launchParams, "meshes",     OD.meshBuffer);
    owlLaunchParamsSetBuffer(OD.launchParams, "lights",     OD.lightBuffer);
    owlLaunchParamsSetBuffer(OD.launchParams, "lightEntities", OD.lightEntitiesBuffer);
    owlLaunchParamsSetBuffer(OD.launchParams, "instanceToEntityMap", OD.instanceToEntityMapBuffer);
    owlLaunchParamsSetBuffer(OD.launchParams, "vertexLists", OD.vertexListsBuffer);
    owlLaunchParamsSetBuffer(OD.launchParams, "indexLists", OD.indexListsBuffer);

    OD.LP.numLightEntities = uint32_t(OD.lightEntities.size());
    owlLaunchParamsSetRaw(OD.launchParams, "numLightEntities", &OD.LP.numLightEntities);
    owlLaunchParamsSetRaw(OD.launchParams, "domeLightIntensity", &OD.LP.domeLightIntensity);

    OWLVarDecl trianglesGeomVars[] = {
        { "index",      OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
        { "vertex",     OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
        { "colors",     OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,colors)},
        { "normals",    OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,normals)},
        { "texcoords",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,texcoords)},
        {/* sentinel to mark end of list */}
    };
    OD.trianglesGeomType
        = owlGeomTypeCreate(OD.context, OWL_GEOM_TRIANGLES, sizeof(TrianglesGeomData), trianglesGeomVars,-1);
    
    /* Temporary test code */
    const int NUM_VERTICES = 1;
    vec3 vertices[NUM_VERTICES] = {{ 0.f, 0.f, 0.f }};
    const int NUM_INDICES = 1;
    ivec3 indices[NUM_INDICES] = {{ 0, 0, 0 }};
    owlGeomTypeSetClosestHit(OD.trianglesGeomType, /*ray type */ 0, OD.module,"TriangleMesh");
    
    OWLBuffer vertexBuffer = owlDeviceBufferCreate(OD.context,OWL_FLOAT3,NUM_VERTICES,vertices);
    OWLBuffer indexBuffer = owlDeviceBufferCreate(OD.context,OWL_INT3,NUM_INDICES,indices);
    OWLGeom trianglesGeom = owlGeomCreate(OD.context,OD.trianglesGeomType);
    owlTrianglesSetVertices(trianglesGeom,vertexBuffer,NUM_VERTICES,sizeof(vec3),0);
    owlTrianglesSetIndices(trianglesGeom,indexBuffer, NUM_INDICES,sizeof(ivec3),0);
    owlGeomSetBuffer(trianglesGeom,"vertex",nullptr);
    owlGeomSetBuffer(trianglesGeom,"index",nullptr);
    owlGeomSetBuffer(trianglesGeom,"colors",nullptr);
    owlGeomSetBuffer(trianglesGeom,"normals",nullptr);
    owlGeomSetBuffer(trianglesGeom,"texcoords",nullptr);
    OWLGroup trianglesGroup = owlTrianglesGeomGroupCreate(OD.context,1,&trianglesGeom);
    owlGroupBuildAccel(trianglesGroup);
    OWLGroup world = owlInstanceGroupCreate(OD.context, 1);
    owlInstanceGroupSetChild(world, 0, trianglesGroup); 
    owlGroupBuildAccel(world);
    owlLaunchParamsSetGroup(OD.launchParams, "world", world);

    // Setup miss prog 
    OWLVarDecl missProgVars[] = {{ /* sentinel to mark end of list */ }};
    OD.missProg = owlMissProgCreate(OD.context,OD.module,"miss",sizeof(MissProgData),missProgVars,-1);
    
    // Setup ray gen program
    OWLVarDecl rayGenVars[] = {{ /* sentinel to mark end of list */ }};
    OD.rayGen = owlRayGenCreate(OD.context,OD.module,"rayGen", sizeof(RayGenData), rayGenVars,-1);

    // Build *SBT* required to trace the groups   
    owlBuildPrograms(OD.context);
    owlBuildPipeline(OD.context);
    owlBuildSBT(OD.context);

    // // Setup denoiser
    // OptixDenoiserOptions options;
    // options.inputKind = OPTIX_DENOISER_INPUT_RGB; // TODO, add albedo and normal
    // options.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    // auto spcontext = dynamic_cast<owl::APIContext::SP&>(OD.context);//->optixContext;
    // optixDenoiserCreate(OD.context.optixContext, &options, &OD.denoiser);
    // OptixDenoiserModelKind kind = OPTIX_DENOISER_MODEL_KIND_HDR;
    // if (!OD.denoiser) throw std::runtime_error("ERROR: denoiser unavailable!");
    
    // optixDenoiserSetModel(OD.denoiser, kind, /*data*/ nullptr, /*sizeInBytes*/ 0);

    // // TODO, reallocate resources on window size change
    // OptixDenoiserSizes denoiserSizes;
    // optixDenoiserComputeMemoryResources(OD.denoiser, OD.LP.frameSize.x, OD.LP.frameSize.y, &denoiserSizes);
    // OD.denoiserScratchBuffer = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(void*), 
    //     denoiserSizes.recommendedScratchSizeInBytes, nullptr);
    // OD.denoiserStateBuffer = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(void*), 
    //     denoiserSizes.stateSizeInBytes, nullptr);
    
    // cudaStreamCreate(&OD.stream);
    // optixDenoiserSetup (
    //     OD.denoiser, 
    //     (cudaStream_t) OD.stream, 
    //     (unsigned int) OD.LP.frameSize.x, 
    //     (unsigned int) OD.LP.frameSize.y, 
    //     (CUdeviceptr) owlBufferGetPointer(OD.denoiserStateBuffer, 0), 
    //     denoiserSizes.stateSizeInBytes,
    //     (CUdeviceptr) owlBufferGetPointer(OD.denoiserScratchBuffer, 0), 
    //     denoiserSizes.recommendedScratchSizeInBytes
    // );

}

void updateComponents()
{
    auto &OD = OptixData;

    if (Mesh::areAnyDirty()) resetAccumulation();
    if (Material::areAnyDirty()) resetAccumulation();
    if (Camera::areAnyDirty()) resetAccumulation();
    if (Transform::areAnyDirty()) resetAccumulation();
    if (Entity::areAnyDirty()) resetAccumulation();
    if (Light::areAnyDirty()) resetAccumulation();

    // Build / Rebuild BLAS
    if (Mesh::areAnyDirty()) {
        Mesh* meshes = Mesh::getFront();
        for (uint32_t mid = 0; mid < Mesh::getCount(); ++mid) {
            if (!meshes[mid].isDirty()) continue;
            if (!meshes[mid].isInitialized()) continue;
            OD.meshes[mid].vertices  = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(vec4), meshes[mid].getVertices().size(), meshes[mid].getVertices().data());
            OD.meshes[mid].colors    = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(vec4), meshes[mid].getColors().size(), meshes[mid].getColors().data());
            OD.meshes[mid].normals   = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(vec4), meshes[mid].getNormals().size(), meshes[mid].getNormals().data());
            OD.meshes[mid].texCoords = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(vec2), meshes[mid].getTexCoords().size(), meshes[mid].getTexCoords().data());
            OD.meshes[mid].indices   = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(uint32_t), meshes[mid].getTriangleIndices().size(), meshes[mid].getTriangleIndices().data());
            OD.meshes[mid].geom      = owlGeomCreate(OD.context, OD.trianglesGeomType);
            owlTrianglesSetVertices(OD.meshes[mid].geom, OD.meshes[mid].vertices, meshes[mid].getVertices().size(), sizeof(vec4), 0);
            owlTrianglesSetIndices(OD.meshes[mid].geom, OD.meshes[mid].indices, meshes[mid].getTriangleIndices().size() / 3, sizeof(ivec3), 0);
            owlGeomSetBuffer(OD.meshes[mid].geom,"vertex", OD.meshes[mid].vertices);
            owlGeomSetBuffer(OD.meshes[mid].geom,"index", OD.meshes[mid].indices);
            owlGeomSetBuffer(OD.meshes[mid].geom,"colors", OD.meshes[mid].colors);
            owlGeomSetBuffer(OD.meshes[mid].geom,"normals", OD.meshes[mid].normals);
            owlGeomSetBuffer(OD.meshes[mid].geom,"texcoords", OD.meshes[mid].texCoords);
            OD.meshes[mid].blas = owlTrianglesGeomGroupCreate(OD.context, 1, &OD.meshes[mid].geom);
            owlGroupBuildAccel(OD.meshes[mid].blas);          
        }

        std::vector<vec4*> vertexLists(Mesh::getCount(), nullptr);
        std::vector<ivec3*> indexLists(Mesh::getCount(), nullptr);
        for (uint32_t mid = 0; mid < Mesh::getCount(); ++mid) {
            if (!meshes[mid].isInitialized()) continue;
            vertexLists[mid] = ((vec4*) owlBufferGetPointer(OD.meshes[mid].vertices, /* device */ 0));
            indexLists[mid] = ((ivec3*) owlBufferGetPointer(OD.meshes[mid].indices, /* device */ 0));
        }
        owlBufferUpload(OD.vertexListsBuffer, vertexLists.data());
        owlBufferUpload(OD.indexListsBuffer, indexLists.data());
    }

    // Build / Rebuild TLAS
    if (Entity::areAnyDirty()) {
        std::vector<OWLGroup> instances;
        std::vector<glm::mat4> instanceTransforms;
        std::vector<uint32_t> instanceToEntityMap;
        Entity* entities = Entity::getFront();
        for (uint32_t eid = 0; eid < Entity::getCount(); ++eid) {
            // if (!entities[eid].isDirty()) continue; // if any entities are dirty, need to rebuild entire TLAS
            if (!entities[eid].isInitialized()) continue;
            if (!entities[eid].getTransform()) continue;
            if (!entities[eid].getMesh()) continue;
            if (!entities[eid].getMaterial() && !entities[eid].getLight()) continue;

            OWLGroup blas = OD.meshes[entities[eid].getMesh()->getId()].blas;
            if (!blas) return;
            glm::mat4 localToWorld = entities[eid].getTransform()->getLocalToWorldMatrix();
            instances.push_back(blas);
            instanceTransforms.push_back(localToWorld);            
            instanceToEntityMap.push_back(eid);
        }

        OD.tlas = owlInstanceGroupCreate(OD.context, instances.size());
        for (uint32_t iid = 0; iid < instances.size(); ++iid) {
            owlInstanceGroupSetChild(OD.tlas, iid, instances[iid]); 
            glm::mat4 m44xfm = instanceTransforms[iid];
            owl4x3f xfm = {
                {m44xfm[0][0], m44xfm[0][1], m44xfm[0][2]}, 
                {m44xfm[1][0], m44xfm[1][1], m44xfm[1][2]}, 
                {m44xfm[2][0], m44xfm[2][1], m44xfm[2][2]},
                {m44xfm[3][0], m44xfm[3][1], m44xfm[3][2]}};
            owlInstanceGroupSetTransform(OD.tlas, iid, xfm);
        }
        owlBufferResize(OD.instanceToEntityMapBuffer, instanceToEntityMap.size());
        owlBufferUpload(OD.instanceToEntityMapBuffer, instanceToEntityMap.data());
        owlGroupBuildAccel(OD.tlas);
        owlLaunchParamsSetGroup(OD.launchParams, "world", OD.tlas);
        owlBuildSBT(OD.context);
    }

    if (Entity::areAnyDirty()) {
        Entity* entities = Entity::getFront();
        OD.lightEntities.resize(0);
        for (uint32_t eid = 0; eid < Entity::getCount(); ++eid) {
            if (!entities[eid].isInitialized()) continue;
            if (!entities[eid].getTransform()) continue;
            if (!entities[eid].getLight()) continue;
            OD.lightEntities.push_back(eid);
        }
        owlBufferResize(OptixData.lightEntitiesBuffer, OD.lightEntities.size());
        owlBufferUpload(OptixData.lightEntitiesBuffer, OD.lightEntities.data());
        OD.LP.numLightEntities = uint32_t(OD.lightEntities.size());
        owlLaunchParamsSetRaw(OD.launchParams, "numLightEntities", &OD.LP.numLightEntities);
    }
    
    Entity::updateComponents();
    Transform::updateComponents();
    Camera::updateComponents();
    Mesh::updateComponents();
    Material::updateComponents();
    Light::updateComponents();

    // For now, just copy everything each frame. Later we can check if any components are dirty, and be more conservative in uploading data
    owlBufferUpload(OptixData.entityBuffer,    Entity::getFrontStruct());
    owlBufferUpload(OptixData.cameraBuffer,    Camera::getFrontStruct());
    owlBufferUpload(OptixData.meshBuffer,      Mesh::getFrontStruct());
    owlBufferUpload(OptixData.materialBuffer,  Material::getFrontStruct());
    owlBufferUpload(OptixData.transformBuffer, Transform::getFrontStruct());
    owlBufferUpload(OptixData.lightBuffer,     Light::getFrontStruct());
}

void updateLaunchParams()
{
    // glfwGetFramebufferSize(window, &curr_frame_size.x, &curr_frame_size.y);
    // const vec3f lookFrom(-4.f,-3.f,-2.f);
    // const vec3f lookAt(0.f,0.f,0.f);
    // const vec3f lookUp(0.f,1.f,0.f);
    // const float cosFovy = 0.66f;
    // LP.view = camera_controls.transform();//glm::lookAt(glm::vec3(-4.f, -3.f, -2.f), glm::vec3(0.f), glm::vec3(0.f, 1.f, 0.f));
    // LP.proj = glm::perspective(glm::radians(45.f), float(curr_frame_size.x) / float(curr_frame_size.y), 1.0f, 1000.0f);
    // LP.viewinv = glm::inverse(LP.view);
    // LP.projinv = glm::inverse(LP.proj);
    // // owlBufferUpload(frameStateBuffer, &LP);

    // auto cam = camera->get_struct();
    // auto cam_transform = camera_transform->get_struct();
    // owlLaunchParamsSetRaw(launchParams,"camera_entity",&cam);
    // owlLaunchParamsSetRaw(launchParams,"camera_transform",&cam_transform);
    // // owlLaunchParamsSetBuffer(launchParams,"entities",entitiesBuffer);
    // // owlLaunchParamsSetBuffer(launchParams,"transforms",transformsBuffer);

    // owlLaunchParamsSetRaw(launchParams,"view",&LP.view);
    // owlLaunchParamsSetRaw(launchParams,"proj",&LP.proj);
    // owlLaunchParamsSetRaw(launchParams,"viewinv",&LP.viewinv);
    // owlLaunchParamsSetRaw(launchParams,"projinv",&LP.projinv);
    // owlLaunchParamsSetRaw(launchParams,"frame_size",&LP.frame_size);
    // owlLaunchParamsSetRaw(launchParams,"frame",&LP.frame); 
    // owlLaunchParamsSetRaw(launchParams,"startFrame",&LP.startFrame); 
    // owlLaunchParamsSetRaw(launchParams,"reset",&LP.reset); 
    // owlLaunchParamsSetRaw(launchParams,"enable_pathtracer",&LP.enable_pathtracer); 
    // owlLaunchParamsSetRaw(launchParams,"enable_space_skipping",&LP.enable_space_skipping); 
    // owlLaunchParamsSetRaw(launchParams,"enable_adaptive_sampling",&LP.enable_adaptive_sampling); 
    // owlLaunchParamsSetRaw(launchParams,"enable_id_colors",&LP.enable_id_colors); 
    // owlLaunchParamsSetRaw(launchParams,"mirror",&LP.mirror); 
    // owlLaunchParamsSetRaw(launchParams,"zoom",&LP.zoom); 
    // owlLaunchParamsSetRaw(launchParams,"min_step_size",&LP.min_step_size); 
    // owlLaunchParamsSetRaw(launchParams,"max_step_size",&LP.max_step_size); 
    // owlLaunchParamsSetRaw(launchParams,"adaptive_power",&LP.adaptive_power); 
    // owlLaunchParamsSetRaw(launchParams,"attenuation",&LP.attenuation); 
    // owlLaunchParamsSetRaw(launchParams,"opacity",&LP.opacity); 
    // owlLaunchParamsSetRaw(launchParams,"volume_type",&LP.volume_type); 
    // owlLaunchParamsSetRaw(launchParams,"show_time_heatmap",&LP.show_time_heatmap); 
    // owlLaunchParamsSetRaw(launchParams,"show_samples_heatmap",&LP.show_samples_heatmap); 
    // owlLaunchParamsSetRaw(launchParams,"empty_threshold",&LP.empty_threshold);        
    // owlLaunchParamsSetRaw(launchParams,"time_min",&LP.time_min);        
    // owlLaunchParamsSetRaw(launchParams,"time_max",&LP.time_max);        
    // owlLaunchParamsSetRaw(launchParams,"tri_mesh_color",&LP.tri_mesh_color);        
    // owlLaunchParamsSetRaw(launchParams,"background_color",&LP.background_color);    
    // owlLaunchParamsSetRaw(launchParams, "transferFunctionMin", &LP.transferFunctionMin);
    // owlLaunchParamsSetRaw(launchParams, "transferFunctionMax", &LP.transferFunctionMax);
    // owlLaunchParamsSetRaw(launchParams, "transferFunctionWidth", &LP.transferFunctionWidth);
    owlLaunchParamsSetRaw(OptixData.launchParams, "frameID", &OptixData.LP.frameID);
    owlLaunchParamsSetRaw(OptixData.launchParams, "frameSize", &OptixData.LP.frameSize);
    owlLaunchParamsSetRaw(OptixData.launchParams, "cameraEntity", &OptixData.LP.cameraEntity);
    owlLaunchParamsSetRaw(OptixData.launchParams, "domeLightIntensity", &OptixData.LP.domeLightIntensity);
    // auto bumesh_transform_struct = bumesh_transform->get_struct();
    // owlLaunchParamsSetRaw(launchParams,"bumesh_transform",&bumesh_transform_struct);

    // auto tri_mesh_transform_struct = tri_mesh_transform->get_struct();
    // owlLaunchParamsSetRaw(launchParams,"tri_mesh_transform",&tri_mesh_transform_struct);
    
    OptixData.LP.frameID ++;
}

void traceRays()
{
    auto &OD = OptixData;
    
    /* Trace Rays */
    owlParamsLaunch2D(OD.rayGen, OD.LP.frameSize.x, OD.LP.frameSize.y, OD.launchParams);
}

void drawFrameBufferToWindow()
{
    auto &OD = OptixData;
    for (int i = 0; i < owlGetDeviceCount(OD.context); i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        cudaError_t err = cudaPeekAtLastError();
        if (err != 0) {
            std::cout<< "ERROR: " << cudaGetErrorString(err)<<std::endl;
            throw std::runtime_error("ERROR");
        }
    }

    cudaGraphicsMapResources(1, &OD.cudaResourceTex);
    const void* fbdevptr = owlBufferGetPointer(OD.frameBuffer,0);
    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(&array, OD.cudaResourceTex, 0, 0);
    cudaMemcpyToArray(array, 0, 0, fbdevptr, OD.LP.frameSize.x *  OD.LP.frameSize.y  * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &OD.cudaResourceTex);

    for (int i = 0; i < owlGetDeviceCount(OD.context); i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        cudaError_t err = cudaPeekAtLastError();
        if (err != 0) {
            std::cout<< "ERROR: " << cudaGetErrorString(err)<<std::endl;
            throw std::runtime_error("ERROR");
        }
    }

    // Draw pixels from optix frame buffer
    glViewport(0, 0, OD.LP.frameSize.x, OD.LP.frameSize.y);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
        
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
            
    glDisable(GL_DEPTH_TEST);
    
    glBindTexture(GL_TEXTURE_2D, OD.imageTexID);

    // This is incredibly slow, but does not require interop
    // glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, windowSize.x, windowSize.y, GL_RGBA, GL_UNSIGNED_BYTE, imageData);
    
    // Draw texture to screen via immediate mode
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, OD.imageTexID);

    glBegin(GL_QUADS);
    glTexCoord2f( 0.0f, 0.0f );
    glVertex2f  ( 0.0f, 0.0f );

    glTexCoord2f( 1.0f, 0.0f );
    glVertex2f  ( 1.0f, 0.0f );

    glTexCoord2f( 1.0f, 1.0f );
    glVertex2f  ( 1.0f, 1.0f );

    glTexCoord2f( 0.0f, 1.0f );
    glVertex2f  ( 0.0f, 1.0f );
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // LP.frame = LP.frame % 1000;
    // LP.frame++;
    // LP.startFrame++;
}

void drawGUI()
{
    auto &io  = ImGui::GetIO();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::ShowDemoWindow();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Update and Render additional Platform Windows
    // (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
    //  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
}

std::future<void> enqueueCommand(std::function<void()> function)
{
    if (ViSII.render_thread_id != std::this_thread::get_id()) 
        std::lock_guard<std::mutex> lock(ViSII.qMutex);

    ViSII::Command c;
    c.function = function;
    c.promise = std::make_shared<std::promise<void>>();
    auto new_future = c.promise->get_future();
    ViSII.commandQueue.push(c);
    // cv.notify_one();
    return new_future;
}

void processCommandQueue()
{
    std::lock_guard<std::mutex> lock(ViSII.qMutex);
    while (!ViSII.commandQueue.empty()) {
        auto item = ViSII.commandQueue.front();
        item.function();
        try {
            item.promise->set_value();
        }
        catch (std::future_error& e) {
            if (e.code() == std::make_error_condition(std::future_errc::promise_already_satisfied))
                std::cout << "ViSII: [promise already satisfied]\n";
            else
                std::cout << "ViSII: [unknown exception]\n";
        }
        ViSII.commandQueue.pop();
    }
}

void resizeWindow(uint32_t width, uint32_t height)
{
    if (ViSII.headlessMode) return;

    auto resizeWindow = [width, height] () {
        using namespace Libraries;
        auto glfw = GLFW::Get();
        glfw->resize_window("ViSII", width, height);
    };

    auto future = enqueueCommand(resizeWindow);
    future.wait();
}

std::vector<float> readFrameBuffer() {
    std::vector<float> frameBuffer(OptixData.LP.frameSize.x * OptixData.LP.frameSize.y * 4);

    auto readFrameBuffer = [&frameBuffer] () {
        int num_devices = owlGetDeviceCount(OptixData.context);
        for (int i = 0; i < num_devices; ++i) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
        cudaSetDevice(0);

        const glm::vec4 *fb = (const glm::vec4*)owlBufferGetPointer(OptixData.frameBuffer,0);
        for (uint32_t test = 0; test < frameBuffer.size(); test += 4) {
            frameBuffer[test + 0] = fb[test / 4].r;
            frameBuffer[test + 1] = fb[test / 4].g;
            frameBuffer[test + 2] = fb[test / 4].b;
            frameBuffer[test + 3] = fb[test / 4].a;
        }

        // memcpy(frameBuffer.data(), fb, frameBuffer.size() * sizeof(float));
    };

    auto future = enqueueCommand(readFrameBuffer);
    future.wait();

    return frameBuffer;
}

std::vector<float> render(uint32_t width, uint32_t height, uint32_t samplesPerPixel) {
    std::vector<float> frameBuffer(width * height * 4);

    auto readFrameBuffer = [&frameBuffer, width, height, samplesPerPixel] () {
        if (!ViSII.headlessMode) {
            using namespace Libraries;
            auto glfw = GLFW::Get();
            glfw->resize_window("ViSII", width, height);
            initializeFrameBuffer(width, height);
        }
        
        resizeOptixFrameBuffer(width, height);
        resetAccumulation();
        updateComponents();

        for (uint32_t i = 0; i < samplesPerPixel; ++i) {
            if (!ViSII.headlessMode) {
                auto glfw = Libraries::GLFW::Get();
                glfw->poll_events();
                glfw->swap_buffers("ViSII");
            }

            updateLaunchParams();
            traceRays();

            if (!ViSII.headlessMode) {
                drawFrameBufferToWindow();
            }
        }        

        int num_devices = owlGetDeviceCount(OptixData.context);
        for (int i = 0; i < num_devices; ++i) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
        cudaSetDevice(0);

        const glm::vec4 *fb = (const glm::vec4*) owlBufferGetPointer(OptixData.frameBuffer,0);
        for (uint32_t test = 0; test < frameBuffer.size(); test += 4) {
            frameBuffer[test + 0] = fb[test / 4].r;
            frameBuffer[test + 1] = fb[test / 4].g;
            frameBuffer[test + 2] = fb[test / 4].b;
            frameBuffer[test + 3] = fb[test / 4].a;
        }
    };

    auto future = enqueueCommand(readFrameBuffer);
    future.wait();

    return frameBuffer;
}

void renderToHDR(uint32_t width, uint32_t height, uint32_t samplesPerPixel, std::string imagePath)
{
    std::vector<float> framebuffer = render(width, height, samplesPerPixel);
    stbi_flip_vertically_on_write(true);
    stbi_write_hdr(imagePath.c_str(), width, height, /* num channels*/ 4, framebuffer.data());
    // stbi_write_png(path.c_str(), curr_frame_size.x, curr_frame_size.y, /* num channels*/ 4, frameBuffer.data(), /* stride in bytes */ curr_frame_size.x * 4);
    // memcpy(frameBuffer.data(), fb, frameBuffer.size() * sizeof(float));
}

void renderToPNG(uint32_t width, uint32_t height, uint32_t samplesPerPixel, std::string imagePath)
{
    std::vector<float> fb = render(width, height, samplesPerPixel);
    std::vector<uint8_t> colors(4 * width * height);
    for (size_t i = 0; i < (width * height); ++i) {       
        colors[i * 4 + 0] = uint8_t(glm::clamp(fb[i * 4 + 0] * 255.f, 0.f, 255.f));
        colors[i * 4 + 1] = uint8_t(glm::clamp(fb[i * 4 + 1] * 255.f, 0.f, 255.f));
        colors[i * 4 + 2] = uint8_t(glm::clamp(fb[i * 4 + 2] * 255.f, 0.f, 255.f));
        colors[i * 4 + 3] = uint8_t(glm::clamp(fb[i * 4 + 3] * 255.f, 0.f, 255.f));
    }
    stbi_flip_vertically_on_write(true);
    stbi_write_png(imagePath.c_str(), width, height, /* num channels*/ 4, colors.data(), /* stride in bytes */ width * 4);
}

void initializeInteractive(bool windowOnTop)
{
    // don't initialize more than once
    if (initialized == true) return;

    initialized = true;
    close = false;
    Camera::initializeFactory();
    Entity::initializeFactory();
    Transform::initializeFactory();
    Material::initializeFactory();
    Mesh::initializeFactory();
    Light::initializeFactory();

    auto loop = [windowOnTop]() {
        ViSII.render_thread_id = std::this_thread::get_id();
        ViSII.headlessMode = false;

        auto glfw = Libraries::GLFW::Get();
        WindowData.window = glfw->create_window("ViSII", 512, 512, windowOnTop, true, true);
        WindowData.currentSize = WindowData.lastSize = ivec2(512, 512);
        glfw->make_context_current("ViSII");
        glfw->poll_events();

        initializeOptix(/*headless = */ false);

        ImGui::CreateContext();
        auto &io  = ImGui::GetIO();
        // ImGui::StyleColorsDark()
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
        applyStyle();
        ImGui_ImplGlfw_InitForOpenGL(WindowData.window, true);
        const char* glsl_version = "#version 130";
        ImGui_ImplOpenGL3_Init(glsl_version);

        while (!close)
        {
            /* Poll events from the window */
            glfw->poll_events();
            glfw->swap_buffers("ViSII");

            updateFrameBuffer();
            updateComponents();
            updateLaunchParams();

            static double start=0;
            static double stop=0;
            start = glfwGetTime();
            traceRays();
            drawFrameBufferToWindow();
            stop = glfwGetTime();
            glfwSetWindowTitle(WindowData.window, std::to_string(1.f / (stop - start)).c_str());
            drawGUI();

            processCommandQueue();
            if (close) break;
        }

        ImGui::DestroyContext();
        if (glfw->does_window_exist("ViSII")) glfw->destroy_window("ViSII");
    };

    renderThread = thread(loop);
}

void initializeHeadless()
{
    // don't initialize more than once
    if (initialized == true) return;

    initialized = true;
    close = false;
    Camera::initializeFactory();
    Entity::initializeFactory();
    Transform::initializeFactory();
    Material::initializeFactory();
    Mesh::initializeFactory();
    Light::initializeFactory();

    auto loop = []() {
        ViSII.render_thread_id = std::this_thread::get_id();
        ViSII.headlessMode = true;

        initializeOptix(/*headless = */ true);

        while (!close)
        {
            updateComponents();
            updateLaunchParams();
            traceRays();
            processCommandQueue();
            if (close) break;
        }
    };

    renderThread = thread(loop);
}

void cleanup()
{
    if (initialized == true) {
        /* cleanup window if open */
        if (close == false) {
            close = true;
            renderThread.join();
        }
        // optixDenoiserDestroy(OptixData.denoiser);
    }
    initialized = false;
}
