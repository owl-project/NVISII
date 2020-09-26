#include <visii/visii.h>

#include <algorithm>

#include <glfw_implementation/glfw.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <ImGuizmo.h>
#include <visii/utilities/colors.h>
#include <owl/owl.h>
#include <owl/helper/optix.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include <devicecode/launch_params.h>
#include <devicecode/path_tracer.h>

#define PBRLUT_IMPLEMENTATION
#include <visii/utilities/ggx_lookup_tables.h>
#include <visii/utilities/procedural_sky.h>

#include <thread>
#include <future>
#include <queue>
#include <algorithm>
#include <cctype>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

#define USE_OPTIX70
#undef USE_OPTIX71

// #define __optix_optix_function_table_h__
#include <optix_stubs.h>
// OptixFunctionTable g_optixFunctionTable;

// #include <thrust/reduce.h>
// #include <thrust/execution_policy.h>
// #include <thrust/device_vector.h>
// #include <thrust/device_ptr.h>

// extern optixDenoiserSetModel;
std::promise<void> exitSignal;
std::thread renderThread;
static bool initialized = false;
static bool stopped = true;
static bool verbose = true;

static struct WindowData {
    GLFWwindow* window = nullptr;
    ivec2 currentSize, lastSize;
} WindowData;

/* Embedded via cmake */
extern "C" char ptxCode[];

// struct MeshData {
//     OWLBuffer vertices;
//     OWLBuffer colors;
//     OWLBuffer normals;
//     OWLBuffer texCoords;
//     OWLBuffer indices;
//     OWLGeom geom;
//     OWLGroup blas;
// };

static struct OptixData {
    OWLContext context;
    OWLModule module;
    OWLLaunchParams launchParams;
    LaunchParams LP;
    GLuint imageTexID = -1;
    cudaGraphicsResource_t cudaResourceTex;
    OWLBuffer frameBuffer;
    OWLBuffer normalBuffer;
    OWLBuffer albedoBuffer;
    OWLBuffer scratchBuffer;
    OWLBuffer mvecBuffer;
    OWLBuffer accumBuffer;

    OWLBuffer entityBuffer;
    OWLBuffer transformBuffer;
    OWLBuffer cameraBuffer;
    OWLBuffer materialBuffer;
    OWLBuffer meshBuffer;
    OWLBuffer lightBuffer;
    OWLBuffer textureBuffer;
    OWLBuffer lightEntitiesBuffer;
    OWLBuffer instanceToEntityMapBuffer;
    OWLBuffer vertexListsBuffer;
    OWLBuffer normalListsBuffer;
    OWLBuffer texCoordListsBuffer;
    OWLBuffer indexListsBuffer;
    OWLBuffer textureObjectsBuffer;

    OWLTexture textureObjects[MAX_TEXTURES + NUM_MAT_PARAMS * MAX_MATERIALS];

    uint32_t numLightEntities;

    OWLRayGen rayGen;
    OWLMissProg missProg;
    OWLGeomType trianglesGeomType;

    OWLBuffer vertexLists[MAX_MESHES];
    OWLBuffer normalLists[MAX_MESHES];
    OWLBuffer texCoordLists[MAX_MESHES];
    OWLBuffer indexLists[MAX_MESHES];
    OWLGeom geomList[MAX_MESHES];
    OWLGroup blasList[MAX_MESHES];

    OWLGroup tlas = nullptr;

    std::vector<uint32_t> lightEntities;

    bool enableDenoiser = false;
    OptixDenoiserSizes denoiserSizes;
    OptixDenoiser denoiser;
    OWLBuffer denoiserScratchBuffer;
    OWLBuffer denoiserStateBuffer;
    OWLBuffer hdrIntensityBuffer;

    Texture* domeLightTexture = nullptr;

    OWLBuffer environmentMapRowsBuffer;
    OWLBuffer environmentMapColsBuffer;
    OWLTexture proceduralSkyTexture;

    MaterialStruct materialStructs[MAX_MATERIALS];

    OWLBuffer placeholder;

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

int getDeviceCount() {
    return owlGetDeviceCount(OptixData.context);
}

OWLContext contextCreate()
{
    OWLContext context = owlContextCreate(/*requested Device IDs*/ nullptr, /* Num Devices */  0);
    owlEnableMotionBlur(context);
    cudaSetDevice(0); // OWL leaves the device as num_devices - 1 after the context is created. set it back to 0.
    return context;
}

OWLModule moduleCreate(OWLContext context, const char* ptxCode)
{
    return owlModuleCreate(context, ptxCode);
}

OWLBuffer managedMemoryBufferCreate(OWLContext context, OWLDataType type, size_t count, void* init)
{
    return owlManagedMemoryBufferCreate(context, type, count, init);
}

OWLBuffer deviceBufferCreate(OWLContext context, OWLDataType type, size_t count, void* init)
{
    return owlDeviceBufferCreate(context, type, count, init);
}

void bufferDestroy(OWLBuffer buffer)
{
    owlBufferDestroy(buffer);
}

void bufferResize(OWLBuffer buffer, size_t newItemCount) {
    owlBufferResize(buffer, newItemCount);
}

const void* bufferGetPointer(OWLBuffer buffer, int deviceId)
{
    return owlBufferGetPointer(buffer, deviceId);
}

void bufferUpload(OWLBuffer buffer, const void *hostPtr)
{
    owlBufferUpload(buffer, hostPtr);
}

CUstream getStream(OWLContext context, int deviceId)
{
    return owlContextGetStream(context, deviceId);
}

OptixDeviceContext getOptixContext(OWLContext context, int deviceID)
{
    return owlContextGetOptixContext(context, deviceID);
}

void buildPrograms(OWLContext context) {
    owlBuildPrograms(context);
}

void buildPipeline(OWLContext context) {
    owlBuildPipeline(context);
}

void buildSBT(OWLContext context) {
    owlBuildSBT(context);
}

OWLMissProg missProgCreate(OWLContext context, OWLModule module, const char *programName, size_t sizeOfVarStruct, OWLVarDecl *vars, size_t numVars)
{
    return owlMissProgCreate(context, module, programName, sizeOfVarStruct, vars, numVars);
}

OWLRayGen rayGenCreate(OWLContext context, OWLModule module, const char *programName, size_t sizeOfVarStruct, OWLVarDecl *vars, size_t numVars) 
{
    return owlRayGenCreate(context, module, programName, sizeOfVarStruct, vars, numVars);
}

OWLGeomType geomTypeCreate(OWLContext context, OWLGeomKind kind, size_t sizeOfVarStruct, OWLVarDecl *vars, size_t numVars)
{
    return owlGeomTypeCreate(context, kind, sizeOfVarStruct, vars, numVars);
}

void geomTypeSetClosestHit(OWLGeomType type, int rayType, OWLModule module, const char *progName)
{
    owlGeomTypeSetClosestHit(type, rayType, module, progName);
}

OWLGeom geomCreate(OWLContext context, OWLGeomType type)
{
    return owlGeomCreate(context, type);
}

void trianglesSetVertices(OWLGeom triangles, OWLBuffer vertices, size_t count, size_t stride, size_t offset)
{
    owlTrianglesSetVertices(triangles,vertices,count,stride,offset);
}

void trianglesSetIndices(OWLGeom triangles, OWLBuffer indices, size_t count, size_t stride, size_t offset)
{
    owlTrianglesSetIndices(triangles, indices, count, stride, offset);
}

void geomSetBuffer(OWLGeom object, const char *varName, OWLBuffer buffer)
{
    owlGeomSetBuffer(object, varName, buffer);
}

OWLGroup trianglesGeomGroupCreate(OWLContext context, size_t numGeometries, OWLGeom *initValues)
{
    return owlTrianglesGeomGroupCreate(context, numGeometries, initValues);
}

OWLGroup instanceGroupCreate(OWLContext context, size_t numInstances, const OWLGroup *initGroups = (const OWLGroup *)nullptr, 
                            const uint32_t *initInstanceIDs = (const uint32_t *)nullptr, const float *initTransforms = (const float *)nullptr, 
                            OWLMatrixFormat matrixFormat = OWL_MATRIX_FORMAT_OWL)
{
    return owlInstanceGroupCreate(context, numInstances, initGroups, initInstanceIDs, initTransforms, matrixFormat);
}

void groupBuildAccel(OWLGroup group)
{
    owlGroupBuildAccel(group);
}

void instanceGroupSetChild(OWLGroup group, int whichChild, OWLGroup child)
{
    owlInstanceGroupSetChild(group, whichChild, child); 
}

void instanceGroupSetTransform(OWLGroup group, size_t childID, glm::mat4 m44xfm)
{
    owl4x3f xfm = {
        {m44xfm[0][0], m44xfm[0][1], m44xfm[0][2]}, 
        {m44xfm[1][0], m44xfm[1][1], m44xfm[1][2]}, 
        {m44xfm[2][0], m44xfm[2][1], m44xfm[2][2]},
        {m44xfm[3][0], m44xfm[3][1], m44xfm[3][2]}};
    owlInstanceGroupSetTransform(group, childID, xfm);
}


OWLLaunchParams launchParamsCreate(OWLContext context, size_t size, OWLVarDecl *vars, size_t numVars)
{
    return owlParamsCreate(context, size, vars, numVars);
}

void launchParamsSetBuffer(OWLLaunchParams params, const char* varName, OWLBuffer buffer)
{
    owlParamsSetBuffer(params, varName, buffer);
}

void launchParamsSetRaw(OWLLaunchParams params, const char* varName, const void* data)
{
    owlParamsSetRaw(params, varName, data);
}

void launchParamsSetTexture(OWLLaunchParams params, const char* varName, OWLTexture texture)
{
    owlParamsSetTexture(params, varName, texture);
}

void launchParamsSetGroup(OWLLaunchParams params, const char *varName, OWLGroup group) {
    owlParamsSetGroup(params, varName, group);
}

void paramsLaunch2D(OWLRayGen rayGen, int dims_x, int dims_y, OWLLaunchParams launchParams)
{
    owlLaunch2D(rayGen, dims_x, dims_y, launchParams);
}

void synchronizeDevices()
{
    for (int i = 0; i < getDeviceCount(); i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        cudaError_t err = cudaPeekAtLastError();
        if (err != 0) {
            std::cout<< "ERROR: " << cudaGetErrorString(err)<<std::endl;
            throw std::runtime_error(std::string("ERROR: ") + cudaGetErrorString(err));
        }
    }
    cudaSetDevice(0);
}

void initializeFrameBuffer(int fbWidth, int fbHeight) {
    fbWidth = glm::max(fbWidth, 1);
    fbHeight = glm::max(fbHeight, 1);
    synchronizeDevices();

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
    
    synchronizeDevices();
}

void resizeOptixFrameBuffer(uint32_t width, uint32_t height)
{
    auto &OD = OptixData;

    OD.LP.frameSize.x = width;
    OD.LP.frameSize.y = height;
    bufferResize(OD.frameBuffer, width * height);
    bufferResize(OD.normalBuffer, width * height);
    bufferResize(OD.albedoBuffer, width * height);
    bufferResize(OD.scratchBuffer, width * height);
    bufferResize(OD.mvecBuffer, width * height);    
    bufferResize(OD.accumBuffer, width * height);
    
    // Reconfigure denoiser
    optixDenoiserComputeMemoryResources(OD.denoiser, OD.LP.frameSize.x, OD.LP.frameSize.y, &OD.denoiserSizes);
    uint64_t scratchSizeInBytes;
    #ifdef USE_OPTIX70
    scratchSizeInBytes = OD.denoiserSizes.recommendedScratchSizeInBytes;
    #else
    scratchSizeInBytes = OD.denoiserSizes.withOverlapScratchSizeInBytes;
    #endif
    bufferResize(OD.denoiserScratchBuffer, scratchSizeInBytes);
    bufferResize(OD.denoiserStateBuffer, OD.denoiserSizes.stateSizeInBytes);
    
    auto cudaStream = getStream(OD.context, 0);
    optixDenoiserSetup (
        OD.denoiser, 
        (cudaStream_t) cudaStream, 
        (unsigned int) OD.LP.frameSize.x, 
        (unsigned int) OD.LP.frameSize.y, 
        (CUdeviceptr) bufferGetPointer(OD.denoiserStateBuffer, 0), 
        OD.denoiserSizes.stateSizeInBytes,
        (CUdeviceptr) bufferGetPointer(OD.denoiserScratchBuffer, 0), 
        scratchSizeInBytes
    );

    resetAccumulation();
}

void updateFrameBuffer()
{
    glfwGetFramebufferSize(WindowData.window, &WindowData.currentSize.x, &WindowData.currentSize.y);

    // window is minimized
    if ((WindowData.currentSize.x == 0) || (WindowData.currentSize.y == 0)) return;

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
    OD.context = contextCreate();   
    OD.module = moduleCreate(OD.context, ptxCode);
    
    /* Setup Optix Launch Params */
    OWLVarDecl launchParamVars[] = {
        { "frameSize",               OWL_USER_TYPE(glm::ivec2),         OWL_OFFSETOF(LaunchParams, frameSize)},
        { "frameID",                 OWL_USER_TYPE(uint64_t),           OWL_OFFSETOF(LaunchParams, frameID)},
        { "frameBuffer",             OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, frameBuffer)},
        { "normalBuffer",            OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, normalBuffer)},
        { "albedoBuffer",            OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, albedoBuffer)},
        { "scratchBuffer",           OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, scratchBuffer)},
        { "mvecBuffer",              OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, mvecBuffer)},
        { "accumPtr",                OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, accumPtr)},
        { "world",                   OWL_GROUP,                         OWL_OFFSETOF(LaunchParams, world)},
        { "cameraEntity",            OWL_USER_TYPE(EntityStruct),       OWL_OFFSETOF(LaunchParams, cameraEntity)},
        { "entities",                OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, entities)},
        { "transforms",              OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, transforms)},
        { "cameras",                 OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, cameras)},
        { "materials",               OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, materials)},
        { "meshes",                  OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, meshes)},
        { "lights",                  OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, lights)},
        { "textures",                OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, textures)},
        { "lightEntities",           OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, lightEntities)},
        { "vertexLists",             OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, vertexLists)},
        { "normalLists",             OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, normalLists)},
        { "texCoordLists",           OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, texCoordLists)},
        { "indexLists",              OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, indexLists)},
        { "numLightEntities",        OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, numLightEntities)},
        { "instanceToEntityMap",     OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, instanceToEntityMap)},
        { "domeLightIntensity",      OWL_USER_TYPE(float),              OWL_OFFSETOF(LaunchParams, domeLightIntensity)},
        { "domeLightColor",          OWL_USER_TYPE(glm::vec3),          OWL_OFFSETOF(LaunchParams, domeLightColor)},
        { "directClamp",             OWL_USER_TYPE(float),              OWL_OFFSETOF(LaunchParams, directClamp)},
        { "indirectClamp",           OWL_USER_TYPE(float),              OWL_OFFSETOF(LaunchParams, indirectClamp)},
        { "maxDiffuseBounceDepth",   OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, maxDiffuseBounceDepth)},
        { "maxSpecularBounceDepth",  OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, maxSpecularBounceDepth)},
        { "numLightSamples",         OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, numLightSamples)},
        { "seed",                    OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, seed)},
        { "xPixelSamplingInterval",  OWL_USER_TYPE(glm::vec2),          OWL_OFFSETOF(LaunchParams, xPixelSamplingInterval)},
        { "yPixelSamplingInterval",  OWL_USER_TYPE(glm::vec2),          OWL_OFFSETOF(LaunchParams, yPixelSamplingInterval)},
        { "timeSamplingInterval",    OWL_USER_TYPE(glm::vec2),          OWL_OFFSETOF(LaunchParams, timeSamplingInterval)},
        { "proj",                    OWL_USER_TYPE(glm::mat4),          OWL_OFFSETOF(LaunchParams, proj)},
        { "viewT0",                  OWL_USER_TYPE(glm::mat4),          OWL_OFFSETOF(LaunchParams, viewT0)},
        { "viewT1",                  OWL_USER_TYPE(glm::mat4),          OWL_OFFSETOF(LaunchParams, viewT1)},
        { "environmentMapID",        OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, environmentMapID)},
        { "environmentMapRotation",  OWL_USER_TYPE(glm::quat),          OWL_OFFSETOF(LaunchParams, environmentMapRotation)},
        { "environmentMapRows",      OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, environmentMapRows)},
        { "environmentMapCols",      OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, environmentMapCols)},
        { "environmentMapWidth",     OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, environmentMapWidth)},
        { "environmentMapHeight",    OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, environmentMapHeight)},
        { "textureObjects",          OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, textureObjects)},
        { "proceduralSkyTexture",    OWL_TEXTURE,                       OWL_OFFSETOF(LaunchParams, proceduralSkyTexture)},
        { "GGX_E_AVG_LOOKUP",        OWL_TEXTURE,                       OWL_OFFSETOF(LaunchParams, GGX_E_AVG_LOOKUP)},
        { "GGX_E_LOOKUP",            OWL_TEXTURE,                       OWL_OFFSETOF(LaunchParams, GGX_E_LOOKUP)},
        { "renderDataMode",          OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, renderDataMode)},
        { "renderDataBounce",        OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, renderDataBounce)},
        { "sceneBBMin",              OWL_USER_TYPE(glm::vec3),          OWL_OFFSETOF(LaunchParams, sceneBBMin)},
        { "sceneBBMax",              OWL_USER_TYPE(glm::vec3),          OWL_OFFSETOF(LaunchParams, sceneBBMax)},
        { /* sentinel to mark end of list */ }
    };
    OD.launchParams = launchParamsCreate(OD.context, sizeof(LaunchParams), launchParamVars, -1);
    
    /* Create AOV Buffers */
    if (!headless) {
        initializeFrameBuffer(512, 512);        
    }

    OD.frameBuffer = managedMemoryBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
    OD.accumBuffer = deviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
    OD.normalBuffer = deviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
    OD.albedoBuffer = deviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
    OD.scratchBuffer = deviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
    OD.mvecBuffer = deviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
    OD.LP.frameSize = glm::ivec2(512, 512);
    launchParamsSetBuffer(OD.launchParams, "frameBuffer", OD.frameBuffer);
    launchParamsSetBuffer(OD.launchParams, "normalBuffer", OD.normalBuffer);
    launchParamsSetBuffer(OD.launchParams, "albedoBuffer", OD.albedoBuffer);
    launchParamsSetBuffer(OD.launchParams, "scratchBuffer", OD.scratchBuffer);
    launchParamsSetBuffer(OD.launchParams, "mvecBuffer", OD.mvecBuffer);
    launchParamsSetBuffer(OD.launchParams, "accumPtr", OD.accumBuffer);
    launchParamsSetRaw(OD.launchParams, "frameSize", &OD.LP.frameSize);

    /* Create Component Buffers */
    // note, extra textures reserved for internal use
    OD.entityBuffer              = deviceBufferCreate(OD.context, OWL_USER_TYPE(EntityStruct),        MAX_ENTITIES,   nullptr);
    OD.transformBuffer           = deviceBufferCreate(OD.context, OWL_USER_TYPE(TransformStruct),     MAX_TRANSFORMS, nullptr);
    OD.cameraBuffer              = deviceBufferCreate(OD.context, OWL_USER_TYPE(CameraStruct),        MAX_CAMERAS,    nullptr);
    OD.materialBuffer            = deviceBufferCreate(OD.context, OWL_USER_TYPE(MaterialStruct),      MAX_MATERIALS,  nullptr);
    OD.meshBuffer                = deviceBufferCreate(OD.context, OWL_USER_TYPE(MeshStruct),          MAX_MESHES,     nullptr);
    OD.lightBuffer               = deviceBufferCreate(OD.context, OWL_USER_TYPE(LightStruct),         MAX_LIGHTS,     nullptr);
    OD.textureBuffer             = deviceBufferCreate(OD.context, OWL_USER_TYPE(TextureStruct),       MAX_TEXTURES + NUM_MAT_PARAMS * MAX_MATERIALS,   nullptr);
    OD.lightEntitiesBuffer       = deviceBufferCreate(OD.context, OWL_USER_TYPE(uint32_t),            1,              nullptr);
    OD.instanceToEntityMapBuffer = deviceBufferCreate(OD.context, OWL_USER_TYPE(uint32_t),            1,              nullptr);
    OD.vertexListsBuffer         = deviceBufferCreate(OD.context, OWL_BUFFER,                         MAX_MESHES,     nullptr);
    OD.normalListsBuffer         = deviceBufferCreate(OD.context, OWL_BUFFER,                         MAX_MESHES,     nullptr);
    OD.texCoordListsBuffer       = deviceBufferCreate(OD.context, OWL_BUFFER,                         MAX_MESHES,     nullptr);
    OD.indexListsBuffer          = deviceBufferCreate(OD.context, OWL_BUFFER,                         MAX_MESHES,     nullptr);
    OD.textureObjectsBuffer      = deviceBufferCreate(OD.context, OWL_TEXTURE,                        MAX_TEXTURES + NUM_MAT_PARAMS * MAX_MATERIALS,   nullptr);

    

    launchParamsSetBuffer(OD.launchParams, "entities",             OD.entityBuffer);
    launchParamsSetBuffer(OD.launchParams, "transforms",           OD.transformBuffer);
    launchParamsSetBuffer(OD.launchParams, "cameras",              OD.cameraBuffer);
    launchParamsSetBuffer(OD.launchParams, "materials",            OD.materialBuffer);
    launchParamsSetBuffer(OD.launchParams, "meshes",               OD.meshBuffer);
    launchParamsSetBuffer(OD.launchParams, "lights",               OD.lightBuffer);
    launchParamsSetBuffer(OD.launchParams, "textures",             OD.textureBuffer);
    launchParamsSetBuffer(OD.launchParams, "lightEntities",        OD.lightEntitiesBuffer);
    launchParamsSetBuffer(OD.launchParams, "instanceToEntityMap",  OD.instanceToEntityMapBuffer);
    launchParamsSetBuffer(OD.launchParams, "vertexLists",          OD.vertexListsBuffer);
    launchParamsSetBuffer(OD.launchParams, "normalLists",          OD.normalListsBuffer);
    launchParamsSetBuffer(OD.launchParams, "texCoordLists",        OD.texCoordListsBuffer);
    launchParamsSetBuffer(OD.launchParams, "indexLists",           OD.indexListsBuffer);
    launchParamsSetBuffer(OD.launchParams, "textureObjects",       OD.textureObjectsBuffer);

    OD.LP.environmentMapID = -1;
    OD.LP.environmentMapRotation = glm::quat(1,0,0,0);
    launchParamsSetRaw(OD.launchParams, "environmentMapID", &OD.LP.environmentMapID);
    launchParamsSetRaw(OD.launchParams, "environmentMapRotation", &OD.LP.environmentMapRotation);

    launchParamsSetBuffer(OD.launchParams, "environmentMapRows", OD.environmentMapRowsBuffer);
    launchParamsSetBuffer(OD.launchParams, "environmentMapCols", OD.environmentMapColsBuffer);
    launchParamsSetRaw(OD.launchParams, "environmentMapWidth", &OD.LP.environmentMapWidth);
    launchParamsSetRaw(OD.launchParams, "environmentMapHeight", &OD.LP.environmentMapHeight);

                            
    OWLTexture GGX_E_AVG_LOOKUP = owlTexture2DCreate(OD.context,
                            OWL_TEXEL_FORMAT_R32F,
                            GGX_E_avg_size,1,
                            GGX_E_avg,
                            OWL_TEXTURE_LINEAR,
                            OWL_TEXTURE_CLAMP);
    OWLTexture GGX_E_LOOKUP = owlTexture2DCreate(OD.context,
                            OWL_TEXEL_FORMAT_R32F,
                            GGX_E_size[0],GGX_E_size[1],
                            GGX_E,
                            OWL_TEXTURE_LINEAR,
                            OWL_TEXTURE_CLAMP);
    launchParamsSetTexture(OD.launchParams, "GGX_E_AVG_LOOKUP", GGX_E_AVG_LOOKUP);
    launchParamsSetTexture(OD.launchParams, "GGX_E_LOOKUP",     GGX_E_LOOKUP);
    launchParamsSetTexture(OD.launchParams, "proceduralSkyTexture",     0);
    
    OD.LP.numLightEntities = uint32_t(OD.lightEntities.size());
    launchParamsSetRaw(OD.launchParams, "numLightEntities", &OD.LP.numLightEntities);
    launchParamsSetRaw(OD.launchParams, "domeLightIntensity", &OD.LP.domeLightIntensity);
    launchParamsSetRaw(OD.launchParams, "domeLightColor", &OD.LP.domeLightColor);
    launchParamsSetRaw(OD.launchParams, "directClamp", &OD.LP.directClamp);
    launchParamsSetRaw(OD.launchParams, "indirectClamp", &OD.LP.indirectClamp);
    launchParamsSetRaw(OD.launchParams, "maxDiffuseBounceDepth", &OD.LP.maxDiffuseBounceDepth);
    launchParamsSetRaw(OD.launchParams, "maxSpecularBounceDepth", &OD.LP.maxSpecularBounceDepth);
    launchParamsSetRaw(OD.launchParams, "numLightSamples", &OD.LP.numLightSamples);
    launchParamsSetRaw(OD.launchParams, "seed", &OD.LP.seed);
    launchParamsSetRaw(OD.launchParams, "xPixelSamplingInterval", &OD.LP.xPixelSamplingInterval);
    launchParamsSetRaw(OD.launchParams, "yPixelSamplingInterval", &OD.LP.yPixelSamplingInterval);
    launchParamsSetRaw(OD.launchParams, "timeSamplingInterval", &OD.LP.timeSamplingInterval);

    OWLVarDecl trianglesGeomVars[] = {{/* sentinel to mark end of list */}};
    OD.trianglesGeomType = geomTypeCreate(OD.context, OWL_GEOM_TRIANGLES, sizeof(TrianglesGeomData), trianglesGeomVars,-1);
    
    /* Temporary test code */
    const int NUM_VERTICES = 1;
    vec3 vertices[NUM_VERTICES] = {{ 0.f, 0.f, 0.f }};
    const int NUM_INDICES = 1;
    ivec3 indices[NUM_INDICES] = {{ 0, 0, 0 }};
    geomTypeSetClosestHit(OD.trianglesGeomType, /*ray type */ 0, OD.module,"TriangleMesh");
    
    OWLBuffer vertexBuffer = deviceBufferCreate(OD.context,OWL_FLOAT4,NUM_VERTICES,vertices);
    OWLBuffer indexBuffer = deviceBufferCreate(OD.context,OWL_INT3,NUM_INDICES,indices);
    OWLGeom trianglesGeom = geomCreate(OD.context,OD.trianglesGeomType);
    trianglesSetVertices(trianglesGeom,vertexBuffer,NUM_VERTICES,sizeof(vec4),0);
    trianglesSetIndices(trianglesGeom,indexBuffer, NUM_INDICES,sizeof(ivec3),0);
    OWLGroup trianglesGroup = trianglesGeomGroupCreate(OD.context,1,&trianglesGeom);
    groupBuildAccel(trianglesGroup);
    OWLGroup world = instanceGroupCreate(OD.context, 1);
    instanceGroupSetChild(world, 0, trianglesGroup); 
    groupBuildAccel(world);
    launchParamsSetGroup(OD.launchParams, "world", world);

    // Setup miss prog 
    OWLVarDecl missProgVars[] = {{ /* sentinel to mark end of list */ }};
    OD.missProg = missProgCreate(OD.context,OD.module,"miss",sizeof(MissProgData),missProgVars,-1);
    
    // Setup ray gen program
    OWLVarDecl rayGenVars[] = {{ /* sentinel to mark end of list */ }};
    OD.rayGen = rayGenCreate(OD.context,OD.module,"rayGen", sizeof(RayGenData), rayGenVars,-1);

    // Build *SBT* required to trace the groups   
    buildPrograms(OD.context);
    buildPipeline(OD.context);
    buildSBT(OD.context);

    // Setup denoiser
    OptixDenoiserOptions options;
    options.inputKind = OPTIX_DENOISER_INPUT_RGB;//_ALBEDO;//_NORMAL;
    // options.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    auto optixContext = getOptixContext(OD.context, 0);
    auto cudaStream = getStream(OD.context, 0);
    OPTIX_CHECK(optixDenoiserCreate(optixContext, &options, &OD.denoiser));
    OptixDenoiserModelKind kind = OPTIX_DENOISER_MODEL_KIND_HDR;
    
    OPTIX_CHECK(optixDenoiserSetModel(OD.denoiser, kind, /*data*/ nullptr, /*sizeInBytes*/ 0));

    OPTIX_CHECK(optixDenoiserComputeMemoryResources(OD.denoiser, OD.LP.frameSize.x, OD.LP.frameSize.y, &OD.denoiserSizes));
    uint64_t scratchSizeInBytes;
    #ifdef USE_OPTIX70
    scratchSizeInBytes = OD.denoiserSizes.recommendedScratchSizeInBytes;
    #else
    scratchSizeInBytes = OD.denoiserSizes.withOverlapScratchSizeInBytes;
    #endif

    OD.denoiserScratchBuffer = deviceBufferCreate(OD.context, OWL_USER_TYPE(void*), 
        scratchSizeInBytes, nullptr);
    OD.denoiserStateBuffer = deviceBufferCreate(OD.context, OWL_USER_TYPE(void*), 
        OD.denoiserSizes.stateSizeInBytes, nullptr);
    OD.hdrIntensityBuffer = deviceBufferCreate(OD.context, OWL_USER_TYPE(float),
        1, nullptr);

    OPTIX_CHECK(optixDenoiserSetup (
        OD.denoiser, 
        (cudaStream_t) cudaStream, 
        (unsigned int) OD.LP.frameSize.x, 
        (unsigned int) OD.LP.frameSize.y, 
        (CUdeviceptr) bufferGetPointer(OD.denoiserStateBuffer, 0), 
        OD.denoiserSizes.stateSizeInBytes,
        (CUdeviceptr) bufferGetPointer(OD.denoiserScratchBuffer, 0), 
        scratchSizeInBytes
    ));

    OD.placeholder = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(void*), 1, nullptr);

    setDomeLightSky(glm::vec3(0,0,10));

    OptixData.LP.sceneBBMin = OptixData.LP.sceneBBMax = glm::vec3(0.f);
}

void initializeImgui()
{
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

void setCameraEntity(Entity* camera_entity)
{
    if (!camera_entity) {
        OptixData.LP.cameraEntity = EntityStruct();
        OptixData.LP.cameraEntity.initialized = false;        
        resetAccumulation();
    }
    else {
        if (!camera_entity->isInitialized()) throw std::runtime_error("Error: camera entity is uninitialized");
        OptixData.LP.cameraEntity = camera_entity->getStruct();
    }
    resetAccumulation();
}

void setDomeLightIntensity(float intensity)
{
    intensity = std::max(float(intensity), float(0.f));
    OptixData.LP.domeLightIntensity = intensity;
    resetAccumulation();
}

void setDomeLightColor(vec3 color)
{
    clearDomeLightTexture();
    color.r = glm::max(0.f, glm::min(color.r, 1.f));
    color.g = glm::max(0.f, glm::min(color.g, 1.f));
    color.b = glm::max(0.f, glm::min(color.b, 1.f));
    OptixData.LP.domeLightColor = color;
    resetAccumulation();
}

void clearDomeLightTexture()
{
    auto func = [] () {
        OptixData.LP.environmentMapID = -1;
        if (OptixData.environmentMapRowsBuffer) owlBufferRelease(OptixData.environmentMapRowsBuffer);
        if (OptixData.environmentMapColsBuffer) owlBufferRelease(OptixData.environmentMapColsBuffer);
        OptixData.environmentMapRowsBuffer = nullptr;
        OptixData.environmentMapColsBuffer = nullptr;
        OptixData.LP.environmentMapWidth = -1;
        OptixData.LP.environmentMapHeight = -1;  
    };

    resetAccumulation();
    auto future = enqueueCommand(func);
    if (ViSII.render_thread_id != std::this_thread::get_id()) 
        future.wait();
}

void generateDomeCDF()
{

}

// Uv range: [0, 1]
vec3 toPolar(vec2 uv)
{
    float theta = 2.0 * M_PI * uv.x + - M_PI / 2.0;
    float phi = M_PI * uv.y;

    vec3 n;
    n.x = cos(theta) * sin(phi);
    n.y = sin(theta) * sin(phi);
    n.z = cos(phi);

    //n = normalize(n);
    // n.z = -n.z;
    // n.x = -n.x;
    return n;
}

void setDomeLightSky(vec3 sunPos, vec3 skyTint, float atmosphereThickness, float saturation)
{
    auto func = [sunPos, skyTint, atmosphereThickness, saturation] () {
        /* Generate procedural sky */
        uint32_t width = 1024/2;
        uint32_t height = 512/2;
        std::vector<glm::vec4> texels(width * height);
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                glm::vec2 uv = glm::vec2(x / float(width), y / float(height));
                glm::vec3 dir = toPolar(uv);
                glm::vec3 c = ProceduralSkybox(glm::vec3(dir.x, -dir.z, dir.y), glm::vec3(sunPos.x, sunPos.z, sunPos.y), skyTint, atmosphereThickness, saturation);
                texels[x + y * width] = glm::vec4(c.r, c.g, c.b, 1.0f);
            }
        }

        //debug
        // stbi_write_hdr("./proceduralSky.hdr", width, height, 4, (float*)texels.data());

        OptixData.LP.environmentMapID = -2;
        if (OptixData.proceduralSkyTexture) {
            owlTexture2DDestroy(OptixData.proceduralSkyTexture);
        }
        OptixData.proceduralSkyTexture = owlTexture2DCreate(OptixData.context, OWL_TEXEL_FORMAT_RGBA32F, width, height, texels.data());
        owlParamsSetTexture(OptixData.launchParams, "proceduralSkyTexture", OptixData.proceduralSkyTexture);

        float invWidth = 1.f / float(width);
        float invHeight = 1.f / float(height);
        float invjacobian = width * height / float(4 * M_PI);

        auto rows = std::vector<float>(height);
        auto cols = std::vector<float>(width * height);
        for (int y = 0, i = 0; y < height; y++) {
            for (int x = 0; x < width; x++, i++) {
                cols[i] = std::max(texels[i].r, std::max(texels[i].g, texels[i].b)) + ((x > 0) ? cols[i - 1] : 0.f);
            }
            rows[y] = cols[i - 1] + ((y > 0) ? rows[y - 1] : 0.0f);
            // normalize the pdf for this scanline (if it was non-zero)
            if (cols[i - 1] > 0) {
                for (int x = 0; x < width; x++) {
                    cols[i - width + x] /= cols[i - 1];
                }
            }
        }

        // normalize the pdf across all scanlines
        for (int y = 0; y < height; y++)
            rows[y] /= rows[height - 1];
        
        // both eval and sample below return a "weight" that is
        // value[i] / row*col_pdf, so might as well bake it into the table
        for (int y = 0, i = 0; y < height; y++) {
            float row_pdf = rows[y] - (y > 0 ? rows[y - 1] : 0.0f);
            for (int x = 0; x < width; x++, i++) {
                float col_pdf = cols[i] - (x > 0 ? cols[i - 1] : 0.0f);
                texels[i].r /= row_pdf * col_pdf * invjacobian;
                texels[i].g /= row_pdf * col_pdf * invjacobian;
                texels[i].b /= row_pdf * col_pdf * invjacobian;
            }
        }

        if (OptixData.environmentMapRowsBuffer) owlBufferRelease(OptixData.environmentMapRowsBuffer);
        if (OptixData.environmentMapColsBuffer) owlBufferRelease(OptixData.environmentMapColsBuffer);
        OptixData.environmentMapRowsBuffer = owlDeviceBufferCreate(OptixData.context, OWL_USER_TYPE(float), height, rows.data());
        OptixData.environmentMapColsBuffer = owlDeviceBufferCreate(OptixData.context, OWL_USER_TYPE(float), width * height, cols.data());
        OptixData.LP.environmentMapWidth = width;
        OptixData.LP.environmentMapHeight = height;  
        resetAccumulation();
    };
    auto future = enqueueCommand(func);
    if (ViSII.render_thread_id != std::this_thread::get_id()) 
        future.wait();
}

void setDomeLightTexture(Texture* texture)
{
    auto func = [texture] () {
        OptixData.LP.environmentMapID = texture->getId();
        std::vector<glm::vec4> texels = texture->getTexels();
        int width = texture->getWidth();
        int height = texture->getHeight();

        float invWidth = 1.f / float(width);
        float invHeight = 1.f / float(height);
        float invjacobian = width * height / float(4 * M_PI);

        auto rows = std::vector<float>(height);
        auto cols = std::vector<float>(width * height);
        for (int y = 0, i = 0; y < height; y++) {
            for (int x = 0; x < width; x++, i++) {
                cols[i] = std::max(texels[i].r, std::max(texels[i].g, texels[i].b)) + ((x > 0) ? cols[i - 1] : 0.f);
            }
            rows[y] = cols[i - 1] + ((y > 0) ? rows[y - 1] : 0.0f);
            // normalize the pdf for this scanline (if it was non-zero)
            if (cols[i - 1] > 0) {
                for (int x = 0; x < width; x++) {
                    cols[i - width + x] /= cols[i - 1];
                }
            }
        }

        // normalize the pdf across all scanlines
        for (int y = 0; y < height; y++)
            rows[y] /= rows[height - 1];
        
        // both eval and sample below return a "weight" that is
        // value[i] / row*col_pdf, so might as well bake it into the table
        for (int y = 0, i = 0; y < height; y++) {
            float row_pdf = rows[y] - (y > 0 ? rows[y - 1] : 0.0f);
            for (int x = 0; x < width; x++, i++) {
                float col_pdf = cols[i] - (x > 0 ? cols[i - 1] : 0.0f);
                texels[i].r /= row_pdf * col_pdf * invjacobian;
                texels[i].g /= row_pdf * col_pdf * invjacobian;
                texels[i].b /= row_pdf * col_pdf * invjacobian;
            }
        }

        if (OptixData.environmentMapRowsBuffer) owlBufferRelease(OptixData.environmentMapRowsBuffer);
        if (OptixData.environmentMapColsBuffer) owlBufferRelease(OptixData.environmentMapColsBuffer);
        OptixData.environmentMapRowsBuffer = owlDeviceBufferCreate(OptixData.context, OWL_USER_TYPE(float), height, rows.data());
        OptixData.environmentMapColsBuffer = owlDeviceBufferCreate(OptixData.context, OWL_USER_TYPE(float), width * height, cols.data());
        OptixData.LP.environmentMapWidth = width;
        OptixData.LP.environmentMapHeight = height;  
        resetAccumulation();        
    };
    auto future = enqueueCommand(func);
    if (ViSII.render_thread_id != std::this_thread::get_id()) 
        future.wait();
}

void setDomeLightRotation(glm::quat rotation)
{
    OptixData.LP.environmentMapRotation = rotation;
    resetAccumulation();
}

void setIndirectLightingClamp(float clamp)
{
    clamp = std::max(float(clamp), float(0.f));
    OptixData.LP.indirectClamp = clamp;
    launchParamsSetRaw(OptixData.launchParams, "indirectClamp", &OptixData.LP.indirectClamp);
    resetAccumulation();
}

void setDirectLightingClamp(float clamp)
{
    clamp = std::max(float(clamp), float(0.f));
    OptixData.LP.directClamp = clamp;
    launchParamsSetRaw(OptixData.launchParams, "directClamp", &OptixData.LP.directClamp);
    resetAccumulation();
}

void setMaxBounceDepth(uint32_t diffuseDepth, uint32_t specularDepth)
{
    OptixData.LP.maxDiffuseBounceDepth = diffuseDepth;
    OptixData.LP.maxSpecularBounceDepth = specularDepth;
    launchParamsSetRaw(OptixData.launchParams, "maxDiffuseBounceDepth", &OptixData.LP.maxDiffuseBounceDepth);
    launchParamsSetRaw(OptixData.launchParams, "maxSpecularBounceDepth", &OptixData.LP.maxSpecularBounceDepth);
    resetAccumulation();
}

void setLightSampleCount(uint32_t count)
{
    if (count > MAX_LIGHT_SAMPLES) 
        throw std::runtime_error(
            std::string("Error: max number of light samples is ") 
            + std::to_string(MAX_LIGHT_SAMPLES));
    if (count == 0) 
        throw std::runtime_error(
            std::string("Error: number of light samples must be between 1 and ") 
            + std::to_string(MAX_LIGHT_SAMPLES));
    OptixData.LP.numLightSamples = count;
    launchParamsSetRaw(OptixData.launchParams, "numLightSamples", &OptixData.LP.numLightSamples);
    resetAccumulation();
}

void samplePixelArea(vec2 xSampleInterval, vec2 ySampleInterval)
{
    OptixData.LP.xPixelSamplingInterval = xSampleInterval;
    OptixData.LP.yPixelSamplingInterval = ySampleInterval;
    launchParamsSetRaw(OptixData.launchParams, "xPixelSamplingInterval", &OptixData.LP.xPixelSamplingInterval);
    launchParamsSetRaw(OptixData.launchParams, "yPixelSamplingInterval", &OptixData.LP.yPixelSamplingInterval);
    resetAccumulation();
}

void sampleTimeInterval(vec2 sampleTimeInterval)
{
    OptixData.LP.timeSamplingInterval = sampleTimeInterval;
    launchParamsSetRaw(OptixData.launchParams, "timeSamplingInterval", &OptixData.LP.timeSamplingInterval);
    resetAccumulation();
}

void updateComponents()
{
    auto &OD = OptixData;

    // If any of the components are dirty, reset accumulation
    if (Mesh::areAnyDirty()) resetAccumulation();
    if (Material::areAnyDirty()) resetAccumulation();
    if (Camera::areAnyDirty()) resetAccumulation();
    if (Transform::areAnyDirty()) resetAccumulation();
    if (Entity::areAnyDirty()) resetAccumulation();
    if (Light::areAnyDirty()) resetAccumulation();
    if (Texture::areAnyDirty()) resetAccumulation();

    // Manage Meshes: Build / Rebuild BLAS
    auto dirtyMeshes = Mesh::getDirtyMeshes();
    if (dirtyMeshes.size() > 0) {
        auto mutex = Mesh::getEditMutex();
        std::lock_guard<std::mutex> lock(*mutex.get());
        for (auto &m : dirtyMeshes) {
            if (OD.vertexLists[m->getId()]) { owlBufferRelease(OD.vertexLists[m->getId()]); OD.vertexLists[m->getId()] = nullptr; }
            if (OD.normalLists[m->getId()]) { owlBufferRelease(OD.normalLists[m->getId()]); OD.normalLists[m->getId()] = nullptr; }
            if (OD.texCoordLists[m->getId()]) { owlBufferRelease(OD.texCoordLists[m->getId()]); OD.texCoordLists[m->getId()] = nullptr; }
            if (OD.indexLists[m->getId()]) { owlBufferRelease(OD.indexLists[m->getId()]); OD.indexLists[m->getId()] = nullptr; }
            if (OD.geomList[m->getId()]) { owlGeomRelease(OD.geomList[m->getId()]); OD.geomList[m->getId()] = nullptr; }
            if (OD.blasList[m->getId()]) { owlGroupRelease(OD.blasList[m->getId()]); OD.blasList[m->getId()] = nullptr; }
            if (!m->isInitialized()) continue;
            if (m->getTriangleIndices().size() == 0) {
                throw std::runtime_error("ERROR: indices is 0");
            }

            OD.vertexLists[m->getId()]  = deviceBufferCreate(OD.context, OWL_USER_TYPE(vec4), m->getVertices().size(), m->getVertices().data());
            OD.normalLists[m->getId()]   = deviceBufferCreate(OD.context, OWL_USER_TYPE(vec4), m->getNormals().size(), m->getNormals().data());
            OD.texCoordLists[m->getId()] = deviceBufferCreate(OD.context, OWL_USER_TYPE(vec2), m->getTexCoords().size(), m->getTexCoords().data());
            OD.indexLists[m->getId()]   = deviceBufferCreate(OD.context, OWL_USER_TYPE(uint32_t), m->getTriangleIndices().size(), m->getTriangleIndices().data());
            OD.geomList[m->getId()]      = geomCreate(OD.context, OD.trianglesGeomType);
            trianglesSetVertices(OD.geomList[m->getId()], OD.vertexLists[m->getId()], m->getVertices().size(), sizeof(vec4), 0);
            trianglesSetIndices(OD.geomList[m->getId()], OD.indexLists[m->getId()], m->getTriangleIndices().size() / 3, sizeof(ivec3), 0);
            OD.blasList[m->getId()] = trianglesGeomGroupCreate(OD.context, 1, &OD.geomList[m->getId()]);
            groupBuildAccel(OD.blasList[m->getId()]);          
        }

        bufferUpload(OD.vertexListsBuffer, OD.vertexLists);
        bufferUpload(OD.texCoordListsBuffer, OD.texCoordLists);
        bufferUpload(OD.indexListsBuffer, OD.indexLists);
        bufferUpload(OD.normalListsBuffer, OD.normalLists);
        Mesh::updateComponents();
        bufferUpload(OptixData.meshBuffer, Mesh::getFrontStruct());
    }    

    // Manage Entities: Build / Rebuild TLAS
    auto dirtyEntities = Entity::getDirtyEntities();
    if (dirtyEntities.size() > 0) {
        auto mutex = Entity::getEditMutex();
        std::lock_guard<std::mutex> lock(*mutex.get());

        std::vector<OWLGroup> instances;
        std::vector<glm::mat4> t0InstanceTransforms;
        std::vector<glm::mat4> t1InstanceTransforms;
        std::vector<uint32_t> instanceToEntityMap;
        Entity* entities = Entity::getFront();
        for (uint32_t eid = 0; eid < Entity::getCount(); ++eid) {
            // if (!entities[eid].isDirty()) continue; // if any entities are dirty, need to rebuild entire TLAS
            if (!entities[eid].isInitialized()) continue;
            if (!entities[eid].getTransform()) continue;
            if (!entities[eid].getMesh()) continue;
            if (!entities[eid].getMaterial() && !entities[eid].getLight()) continue;

            OWLGroup blas = OD.blasList[entities[eid].getMesh()->getId()];
            if (!blas) {
                // Not sure why, but the mesh this entity references hasn't been constructed yet.
                // Mark it as dirty. It should be available in a subsequent frame
                entities[eid].getMesh()->markDirty();
                return; 
                // throw std::runtime_error("ERROR: entity missing BLAS");
            }
            glm::mat4 prevLocalToWorld = entities[eid].getTransform()->getLocalToWorldMatrix(/*previous = */true);
            glm::mat4 localToWorld = entities[eid].getTransform()->getLocalToWorldMatrix(/*previous = */false);
            t0InstanceTransforms.push_back(prevLocalToWorld);            
            t1InstanceTransforms.push_back(localToWorld);            
            instances.push_back(blas);
            instanceToEntityMap.push_back(eid);
        }

        std::vector<owl4x3f>     t0Transforms;
        std::vector<owl4x3f>     t1Transforms;
        auto oldTLAS = OD.tlas;
        // not sure why, but if I release this TLAS, I get the following error
        // python3d: /home/runner/work/ViSII/ViSII/externals/owl/owl/ObjectRegistry.cpp:83: 
        //   owl::RegisteredObject* owl::ObjectRegistry::getPtr(int): Assertion `objects[ID]' failed.
        OD.tlas = instanceGroupCreate(OD.context, instances.size());
        for (uint32_t iid = 0; iid < instances.size(); ++iid) {
            instanceGroupSetChild(OD.tlas, iid, instances[iid]); 
            glm::mat4 xfm0 = t0InstanceTransforms[iid];
            glm::mat4 xfm1 = t1InstanceTransforms[iid];
            
            owl4x3f oxfm0 = {
                {xfm0[0][0], xfm0[0][1], xfm0[0][2]}, 
                {xfm0[1][0], xfm0[1][1], xfm0[1][2]}, 
                {xfm0[2][0], xfm0[2][1], xfm0[2][2]},
                {xfm0[3][0], xfm0[3][1], xfm0[3][2]}};
            t0Transforms.push_back(oxfm0);

            owl4x3f oxfm1 = {
                {xfm1[0][0], xfm1[0][1], xfm1[0][2]}, 
                {xfm1[1][0], xfm1[1][1], xfm1[1][2]}, 
                {xfm1[2][0], xfm1[2][1], xfm1[2][2]},
                {xfm1[3][0], xfm1[3][1], xfm1[3][2]}};
            t1Transforms.push_back(oxfm1);
        }
        
        owlInstanceGroupSetTransforms(OD.tlas,0,(const float*)t0Transforms.data());
        owlInstanceGroupSetTransforms(OD.tlas,1,(const float*)t1Transforms.data());

        bufferResize(OD.instanceToEntityMapBuffer, instanceToEntityMap.size());
        bufferUpload(OD.instanceToEntityMapBuffer, instanceToEntityMap.data());
        groupBuildAccel(OD.tlas);
        launchParamsSetGroup(OD.launchParams, "world", OD.tlas);
        buildSBT(OD.context);

        if (oldTLAS) {owlGroupRelease(oldTLAS);}
    
        OD.lightEntities.resize(0);
        for (uint32_t eid = 0; eid < Entity::getCount(); ++eid) {
            if (!entities[eid].isInitialized()) continue;
            if (!entities[eid].getTransform()) continue;
            if (!entities[eid].getLight()) continue;
            if (!entities[eid].getMesh()) continue;
            OD.lightEntities.push_back(eid);
        }
        bufferResize(OptixData.lightEntitiesBuffer, OD.lightEntities.size());
        bufferUpload(OptixData.lightEntitiesBuffer, OD.lightEntities.data());
        OD.LP.numLightEntities = uint32_t(OD.lightEntities.size());
        launchParamsSetRaw(OD.launchParams, "numLightEntities", &OD.LP.numLightEntities);

        Entity::updateComponents();
        bufferUpload(OptixData.entityBuffer,    Entity::getFrontStruct());
    }

    // Manage textures and materials
    if (Texture::areAnyDirty() || Material::areAnyDirty()) {
        auto mutex = Texture::getEditMutex();
        std::lock_guard<std::mutex> lock(*mutex.get());

        // Allocate cuda textures for all texture components
        Texture* textures = Texture::getFront();
        for (uint32_t tid = 0; tid < MAX_TEXTURES; ++tid) {
            if (!textures[tid].isDirty()) continue;
            if (OD.textureObjects[tid]) { owlTexture2DDestroy(OD.textureObjects[tid]); OD.textureObjects[tid] = nullptr; }
            if (!textures[tid].isInitialized()) continue;
            OD.textureObjects[tid] = owlTexture2DCreate(
                OD.context, OWL_TEXEL_FORMAT_RGBA32F,
                textures[tid].getWidth(), textures[tid].getHeight(), textures[tid].getTexels().data(),
                OWL_TEXTURE_LINEAR, OWL_TEXTURE_WRAP);
        }

        // Create additional cuda textures for material constants

        // Manage materials
        {
            auto mutex = Material::getEditMutex();
            std::lock_guard<std::mutex> lock(*mutex.get());
            Material* materials = Material::getFront();
            MaterialStruct* matStructs = Material::getFrontStruct();
            
            for (uint32_t mid = 0; mid < MAX_MATERIALS; ++mid) {
                if (!materials[mid].isInitialized()) continue;
                if (!materials[mid].isDirty()) continue;

                OptixData.materialStructs[mid] = matStructs[mid];

                auto genSTex = [&OD](int index, float s) {
                    if (OD.textureObjects[index]) { owlTexture2DDestroy(OD.textureObjects[index]); OD.textureObjects[index] = nullptr; }
                    OD.textureObjects[index] = owlTexture2DCreate(
                        OD.context, OWL_TEXEL_FORMAT_R32F,
                        1,1, &s, OWL_TEXTURE_LINEAR, OWL_TEXTURE_WRAP);
                };

                auto genRGBATex = [&OD](int index, vec4 c) {
                    if (OD.textureObjects[index]) { owlTexture2DDestroy(OD.textureObjects[index]); OD.textureObjects[index] = nullptr; }
                    OD.textureObjects[index] = owlTexture2DCreate(
                        OD.context, OWL_TEXEL_FORMAT_RGBA32F,
                        1,1, &c, OWL_TEXTURE_LINEAR, OWL_TEXTURE_WRAP);
                };

                int off = MAX_TEXTURES + mid * NUM_MAT_PARAMS;
                if (matStructs[mid].transmission_roughness_texture_id == -1) { genRGBATex(off + 0, vec4(materials[mid].getTransmissionRoughness())); }
                if (matStructs[mid].base_color_texture_id == -1)             { genRGBATex(off + 1, vec4(materials[mid].getBaseColor(), 1.f)); }
                if (matStructs[mid].roughness_texture_id == -1)              { genRGBATex(off + 2, vec4(materials[mid].getRoughness())); }
                if (matStructs[mid].alpha_texture_id == -1)                  { genRGBATex(off + 3, vec4(materials[mid].getAlpha())); }
                if (matStructs[mid].normal_map_texture_id == -1)             { genRGBATex(off + 4, vec4(0.5f, .5f, 1.f, 0.f)); }
                if (matStructs[mid].subsurface_color_texture_id == -1)       { genRGBATex(off + 5, vec4(materials[mid].getSubsurfaceColor(), 1.f)); }
                if (matStructs[mid].subsurface_radius_texture_id == -1)      { genRGBATex(off + 6, vec4(materials[mid].getSubsurfaceRadius(), 1.f)); }
                if (matStructs[mid].subsurface_texture_id == -1)             { genRGBATex(off + 7, vec4(materials[mid].getSubsurface())); }
                if (matStructs[mid].metallic_texture_id == -1)               { genRGBATex(off + 8, vec4(materials[mid].getMetallic())); }
                if (matStructs[mid].specular_texture_id == -1)               { genRGBATex(off + 9, vec4(materials[mid].getSpecular())); }
                if (matStructs[mid].specular_tint_texture_id == -1)          { genRGBATex(off + 10, vec4(materials[mid].getSpecularTint())); }
                if (matStructs[mid].anisotropic_texture_id == -1)            { genRGBATex(off + 11, vec4(materials[mid].getAnisotropic())); }
                if (matStructs[mid].anisotropic_rotation_texture_id == -1)   { genRGBATex(off + 12, vec4(materials[mid].getAnisotropicRotation())); }
                if (matStructs[mid].sheen_texture_id == -1)                  { genRGBATex(off + 13, vec4(materials[mid].getSheen())); }
                if (matStructs[mid].sheen_tint_texture_id == -1)             { genRGBATex(off + 14, vec4(materials[mid].getSheenTint())); }
                if (matStructs[mid].clearcoat_texture_id == -1)              { genRGBATex(off + 15, vec4(materials[mid].getClearcoat())); }
                if (matStructs[mid].clearcoat_roughness_texture_id == -1)    { genRGBATex(off + 16, vec4(materials[mid].getClearcoatRoughness())); }
                if (matStructs[mid].ior_texture_id == -1)                    { genRGBATex(off + 17, vec4(materials[mid].getIor())); }
                if (matStructs[mid].transmission_texture_id == -1)           { genRGBATex(off + 18, vec4(materials[mid].getTransmission())); }
                
                if (matStructs[mid].transmission_roughness_texture_id == -1) { OptixData.materialStructs[mid].transmission_roughness_texture_id = off + 0; }
                if (matStructs[mid].base_color_texture_id == -1)             { OptixData.materialStructs[mid].base_color_texture_id = off + 1; }
                if (matStructs[mid].roughness_texture_id == -1)              { OptixData.materialStructs[mid].roughness_texture_id = off + 2; }
                if (matStructs[mid].alpha_texture_id == -1)                  { OptixData.materialStructs[mid].alpha_texture_id = off + 3; }
                if (matStructs[mid].normal_map_texture_id == -1)             { OptixData.materialStructs[mid].normal_map_texture_id = off + 4; }
                if (matStructs[mid].subsurface_color_texture_id == -1)       { OptixData.materialStructs[mid].subsurface_color_texture_id = off + 5; }
                if (matStructs[mid].subsurface_radius_texture_id == -1)      { OptixData.materialStructs[mid].subsurface_radius_texture_id = off + 6; }
                if (matStructs[mid].subsurface_texture_id == -1)             { OptixData.materialStructs[mid].subsurface_texture_id = off + 7; }
                if (matStructs[mid].metallic_texture_id == -1)               { OptixData.materialStructs[mid].metallic_texture_id = off + 8; }
                if (matStructs[mid].specular_texture_id == -1)               { OptixData.materialStructs[mid].specular_texture_id = off + 9; }
                if (matStructs[mid].specular_tint_texture_id == -1)          { OptixData.materialStructs[mid].specular_tint_texture_id = off + 10; }
                if (matStructs[mid].anisotropic_texture_id == -1)            { OptixData.materialStructs[mid].anisotropic_texture_id = off + 11; }
                if (matStructs[mid].anisotropic_rotation_texture_id == -1)   { OptixData.materialStructs[mid].anisotropic_rotation_texture_id = off + 12; }
                if (matStructs[mid].sheen_texture_id == -1)                  { OptixData.materialStructs[mid].sheen_texture_id = off + 13; }
                if (matStructs[mid].sheen_tint_texture_id == -1)             { OptixData.materialStructs[mid].sheen_tint_texture_id = off + 14; }
                if (matStructs[mid].clearcoat_texture_id == -1)              { OptixData.materialStructs[mid].clearcoat_texture_id = off + 15; }
                if (matStructs[mid].clearcoat_roughness_texture_id == -1)    { OptixData.materialStructs[mid].clearcoat_roughness_texture_id = off + 16; }
                if (matStructs[mid].ior_texture_id == -1)                    { OptixData.materialStructs[mid].ior_texture_id = off + 17; }
                if (matStructs[mid].transmission_texture_id == -1)           { OptixData.materialStructs[mid].transmission_texture_id = off + 18; }
            }


            Material::updateComponents();
            bufferUpload(OptixData.materialBuffer, OptixData.materialStructs);
        }
        
        bufferUpload(OD.textureObjectsBuffer, OD.textureObjects);
        Texture::updateComponents();
        bufferUpload(OptixData.textureBuffer, Texture::getFrontStruct());
    }
    
    // Manage transforms
    auto dirtyTransforms = Transform::getDirtyTransforms();
    if (dirtyTransforms.size() > 0) {
        auto mutex = Transform::getEditMutex();
        std::lock_guard<std::mutex> lock(*mutex.get());

        Transform::updateComponents();
        
        // for each device
        for (uint32_t id = 0; id < owlGetDeviceCount(OptixData.context); ++id)
        {
            cudaSetDevice(id);

            TransformStruct* devTransforms = (TransformStruct*)owlBufferGetPointer(OptixData.transformBuffer, id);
            TransformStruct* transformStructs = Transform::getFrontStruct();
            for (auto &t : dirtyTransforms) {
                if (!t->isInitialized()) continue;
                CUDA_CHECK(cudaMemcpy(&devTransforms[t->getId()], &transformStructs[t->getId()], sizeof(TransformStruct), cudaMemcpyHostToDevice));
            }
        }

        cudaSetDevice(0);
        // bufferUpload(OptixData.transformBuffer, Transform::getFrontStruct());
    }   

    // Manage Cameras
    if (Camera::areAnyDirty()) {
        auto mutex = Camera::getEditMutex();
        std::lock_guard<std::mutex> lock(*mutex.get());

        Camera::updateComponents();
        bufferUpload(OptixData.cameraBuffer,    Camera::getFrontStruct());
    }    

    // Manage lights
    if (Light::areAnyDirty()) {
        auto mutex = Light::getEditMutex();
        std::lock_guard<std::mutex> lock(*mutex.get());

        Light::updateComponents();
        bufferUpload(OptixData.lightBuffer,     Light::getFrontStruct());
    }

    if (OptixData.LP.cameraEntity.initialized) {
        auto transform = Transform::getFront()[OptixData.LP.cameraEntity.transform_id];
        auto camera = Camera::getFront()[OptixData.LP.cameraEntity.camera_id];
        OptixData.LP.proj = camera.getProjection();
        OptixData.LP.viewT0 = transform.getWorldToLocalMatrix(/*previous = */ true);
        OptixData.LP.viewT1 = transform.getWorldToLocalMatrix(/*previous = */ false);
    }   
}

void updateLaunchParams()
{
    launchParamsSetRaw(OptixData.launchParams, "frameID", &OptixData.LP.frameID);
    launchParamsSetRaw(OptixData.launchParams, "frameSize", &OptixData.LP.frameSize);
    launchParamsSetRaw(OptixData.launchParams, "cameraEntity", &OptixData.LP.cameraEntity);
    launchParamsSetRaw(OptixData.launchParams, "domeLightIntensity", &OptixData.LP.domeLightIntensity);
    launchParamsSetRaw(OptixData.launchParams, "domeLightColor", &OptixData.LP.domeLightColor);
    launchParamsSetRaw(OptixData.launchParams, "renderDataMode", &OptixData.LP.renderDataMode);
    launchParamsSetRaw(OptixData.launchParams, "renderDataBounce", &OptixData.LP.renderDataBounce);
    launchParamsSetRaw(OptixData.launchParams, "seed", &OptixData.LP.seed);
    launchParamsSetRaw(OptixData.launchParams, "proj", &OptixData.LP.proj);
    launchParamsSetRaw(OptixData.launchParams, "viewT0", &OptixData.LP.viewT0);
    launchParamsSetRaw(OptixData.launchParams, "viewT1", &OptixData.LP.viewT1);

    launchParamsSetRaw(OptixData.launchParams, "environmentMapID", &OptixData.LP.environmentMapID);
    launchParamsSetRaw(OptixData.launchParams, "environmentMapRotation", &OptixData.LP.environmentMapRotation);
    launchParamsSetBuffer(OptixData.launchParams, "environmentMapRows", OptixData.environmentMapRowsBuffer);
    launchParamsSetBuffer(OptixData.launchParams, "environmentMapCols", OptixData.environmentMapColsBuffer);
    launchParamsSetRaw(OptixData.launchParams, "environmentMapWidth", &OptixData.LP.environmentMapWidth);
    launchParamsSetRaw(OptixData.launchParams, "environmentMapHeight", &OptixData.LP.environmentMapHeight);
    launchParamsSetRaw(OptixData.launchParams, "sceneBBMin", &OptixData.LP.sceneBBMin);
    launchParamsSetRaw(OptixData.launchParams, "sceneBBMax", &OptixData.LP.sceneBBMax);

    OptixData.LP.frameID ++;
}

void traceRays()
{
    auto &OD = OptixData;
    
    /* Trace Rays */
    paramsLaunch2D(OD.rayGen, OD.LP.frameSize.x, OD.LP.frameSize.y, OD.launchParams);
}

void denoiseImage() {
    synchronizeDevices();

    auto &OD = OptixData;
    auto cudaStream = getStream(OD.context, 0);

    CUdeviceptr frameBuffer = (CUdeviceptr) bufferGetPointer(OD.frameBuffer, 0);

    std::vector<OptixImage2D> inputLayers;
    OptixImage2D colorLayer;
    colorLayer.width = OD.LP.frameSize.x;
    colorLayer.height = OD.LP.frameSize.y;
    colorLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    colorLayer.pixelStrideInBytes = 4 * sizeof(float);
    colorLayer.rowStrideInBytes   = OD.LP.frameSize.x * 4 * sizeof(float);
    colorLayer.data   = (CUdeviceptr) bufferGetPointer(OD.frameBuffer, 0);
    inputLayers.push_back(colorLayer);

    OptixImage2D albedoLayer;
    albedoLayer.width = OD.LP.frameSize.x;
    albedoLayer.height = OD.LP.frameSize.y;
    albedoLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    albedoLayer.pixelStrideInBytes = 4 * sizeof(float);
    albedoLayer.rowStrideInBytes   = OD.LP.frameSize.x * 4 * sizeof(float);
    albedoLayer.data   = (CUdeviceptr) bufferGetPointer(OD.albedoBuffer, 0);
    // inputLayers.push_back(albedoLayer);

    OptixImage2D normalLayer;
    normalLayer.width = OD.LP.frameSize.x;
    normalLayer.height = OD.LP.frameSize.y;
    normalLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    normalLayer.pixelStrideInBytes = 4 * sizeof(float);
    normalLayer.rowStrideInBytes   = OD.LP.frameSize.x * 4 * sizeof(float);
    normalLayer.data   = (CUdeviceptr) bufferGetPointer(OD.normalBuffer, 0);
    // inputLayers.push_back(normalLayer);

    OptixImage2D outputLayer = colorLayer; // can I get away with this?

    uint64_t scratchSizeInBytes;
    #ifdef USE_OPTIX70
    scratchSizeInBytes = OD.denoiserSizes.recommendedScratchSizeInBytes;
    #else
    scratchSizeInBytes = OD.denoiserSizes.withOverlapScratchSizeInBytes;
    #endif

    // compute average pixel intensity for hdr denoising
    OPTIX_CHECK(optixDenoiserComputeIntensity(
        OD.denoiser, 
        cudaStream, 
        &inputLayers[0], 
        (CUdeviceptr) bufferGetPointer(OD.hdrIntensityBuffer, 0),
        (CUdeviceptr) bufferGetPointer(OD.denoiserScratchBuffer, 0),
        scratchSizeInBytes));

    OptixDenoiserParams params;
    params.denoiseAlpha = 0;    // Don't touch alpha.
    params.blendFactor  = 0.0f; // Show the denoised image only.
    params.hdrIntensity = (CUdeviceptr) bufferGetPointer(OD.hdrIntensityBuffer, 0);
    
    OPTIX_CHECK(optixDenoiserInvoke(
        OD.denoiser,
        cudaStream,
        &params,
        (CUdeviceptr) bufferGetPointer(OD.denoiserStateBuffer, 0),
        OD.denoiserSizes.stateSizeInBytes,
        inputLayers.data(),
        inputLayers.size(),
        /* inputOffsetX */ 0,
        /* inputOffsetY */ 0,
        &outputLayer,
        (CUdeviceptr) bufferGetPointer(OD.denoiserScratchBuffer, 0),
        scratchSizeInBytes
    ));

    synchronizeDevices();
}

void drawFrameBufferToWindow()
{
    auto &OD = OptixData;
    synchronizeDevices();

    cudaGraphicsMapResources(1, &OD.cudaResourceTex);
    const void* fbdevptr = bufferGetPointer(OD.frameBuffer,0);
    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(&array, OD.cudaResourceTex, 0, 0);
    cudaMemcpyToArray(array, 0, 0, fbdevptr, OD.LP.frameSize.x *  OD.LP.frameSize.y  * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &OD.cudaResourceTex);

    
    // Draw pixels from optix frame buffer
    glEnable(GL_FRAMEBUFFER_SRGB); 
    glViewport(0, 0, OD.LP.frameSize.x, OD.LP.frameSize.y);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
        
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
            
    glDisable(GL_DEPTH_TEST);    
    glBindTexture(GL_TEXTURE_2D, OD.imageTexID);
    
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
}

void drawGUI()
{
    // auto &io  = ImGui::GetIO();
    // ImGui_ImplOpenGL3_NewFrame();
    // ImGui_ImplGlfw_NewFrame();
    // ImGui::NewFrame();

    // ImGui::ShowDemoWindow();

    // ImGui::Render();
    // ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // // Update and Render additional Platform Windows
    // // (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
    // //  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
    // if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    // {
    //     GLFWwindow* backup_current_context = glfwGetCurrentContext();
    //     ImGui::UpdatePlatformWindows();
    //     ImGui::RenderPlatformWindowsDefault();
    //     glfwMakeContextCurrent(backup_current_context);
    // }
}

void resizeWindow(uint32_t width, uint32_t height)
{
    width = (width <= 0) ? 1 : width;
    height = (height <= 0) ? 1 : height;
    if (ViSII.headlessMode) return;

    auto resizeWindow = [width, height] () {
        using namespace Libraries;
        auto glfw = GLFW::Get();
        glfw->resize_window("ViSII", width, height);

        glViewport(0,0,width,height);
    };

    auto future = enqueueCommand(resizeWindow);
    future.wait();
}

void enableDenoiser() 
{

    auto enableDenoiser = [] () {
        // int num_devices = getDeviceCount();
        // if (num_devices > 1) {
        //     throw std::runtime_error("ERROR: OptiX denoiser currently only supported for single GPU OptiX contexts");
        // }
        OptixData.enableDenoiser = true;
        // resetAccumulation(); // reset not required, just effects final framebuffer
    };
    enqueueCommand(enableDenoiser).wait();
}

void disableDenoiser()
{
    auto disableDenoiser = [] () {
        OptixData.enableDenoiser = false;
        // resetAccumulation(); // reset not required, just effects final framebuffer
    };
    enqueueCommand(disableDenoiser).wait();
}

std::vector<float> readFrameBuffer() {
    std::vector<float> frameBuffer(OptixData.LP.frameSize.x * OptixData.LP.frameSize.y * 4);

    auto readFrameBuffer = [&frameBuffer] () {
        int num_devices = getDeviceCount();
        synchronizeDevices();

        const glm::vec4 *fb = (const glm::vec4*)bufferGetPointer(OptixData.frameBuffer,0);
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

std::vector<float> render(uint32_t width, uint32_t height, uint32_t samplesPerPixel, uint32_t seed) {
    if ((width < 1) || (height < 1)) throw std::runtime_error("Error, invalid width/height");
    std::vector<float> frameBuffer(width * height * 4);

    auto readFrameBuffer = [&frameBuffer, width, height, samplesPerPixel, seed] () {
        if (!ViSII.headlessMode) {
            using namespace Libraries;
            auto glfw = GLFW::Get();
            glfw->resize_window("ViSII", width, height);
            initializeFrameBuffer(width, height);
        }
        
        OptixData.LP.seed = seed;

        resizeOptixFrameBuffer(width, height);
        resetAccumulation();
        updateComponents();

        for (uint32_t i = 0; i < samplesPerPixel; ++i) {
            // std::cout<<i<<std::endl;
            if (!ViSII.headlessMode) {
                auto glfw = Libraries::GLFW::Get();
                glfw->poll_events();
                glfw->swap_buffers("ViSII");
                glClearColor(1,1,1,1);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            }

            updateLaunchParams();
            traceRays();
            if (OptixData.enableDenoiser)
            {
                denoiseImage();
            }

            if (!ViSII.headlessMode) {
                drawFrameBufferToWindow();
                glfwSetWindowTitle(WindowData.window, 
                    (std::to_string(i) + std::string("/") + std::to_string(samplesPerPixel)).c_str());
            }

            if (verbose) {
                std::cout<< "\r" << i << "/" << samplesPerPixel;
            }
        }      
        if (!ViSII.headlessMode) {
            glfwSetWindowTitle(WindowData.window, 
                (std::to_string(samplesPerPixel) + std::string("/") + std::to_string(samplesPerPixel) 
                + std::string(" - done!")).c_str());
        }
        
        if (verbose) {
            std::cout<<"\r "<< samplesPerPixel << "/" << samplesPerPixel <<" - done!" << std::endl;
        }

        synchronizeDevices();

        const glm::vec4 *fb = (const glm::vec4*) bufferGetPointer(OptixData.frameBuffer,0);
        for (uint32_t test = 0; test < frameBuffer.size(); test += 4) {
            frameBuffer[test + 0] = fb[test / 4].r;
            frameBuffer[test + 1] = fb[test / 4].g;
            frameBuffer[test + 2] = fb[test / 4].b;
            frameBuffer[test + 3] = fb[test / 4].a;
        }

        synchronizeDevices();
    };

    auto future = enqueueCommand(readFrameBuffer);
    future.wait();

    return frameBuffer;
}

std::string trim(const std::string& line)
{
    const char* WhiteSpace = " \t\v\r\n";
    std::size_t start = line.find_first_not_of(WhiteSpace);
    std::size_t end = line.find_last_not_of(WhiteSpace);
    return start == end ? std::string() : line.substr(start, end - start + 1);
}

std::vector<float> renderData(uint32_t width, uint32_t height, uint32_t startFrame, uint32_t frameCount, uint32_t bounce, std::string _option, uint32_t seed)
{
    std::vector<float> frameBuffer(width * height * 4);

    auto readFrameBuffer = [&frameBuffer, width, height, startFrame, frameCount, bounce, _option, seed] () {
        if (!ViSII.headlessMode) {
            using namespace Libraries;
            auto glfw = GLFW::Get();
            glfw->resize_window("ViSII", width, height);
            initializeFrameBuffer(width, height);
        }

        // remove trailing whitespace from option, convert to lowercase
        std::string option = trim(_option);
        std::transform(option.begin(), option.end(), option.begin(),
            [](unsigned char c){ return std::tolower(c); });

        if (option == std::string("none")) {
            OptixData.LP.renderDataMode = RenderDataFlags::NONE;
        }
        else if (option == std::string("depth")) {
            OptixData.LP.renderDataMode = RenderDataFlags::DEPTH;
        }
        else if (option == std::string("ray_direction")) {
            OptixData.LP.renderDataMode = RenderDataFlags::RAY_DIRECTION;
        }
        else if (option == std::string("position")) {
            OptixData.LP.renderDataMode = RenderDataFlags::POSITION;
        }
        else if (option == std::string("normal")) {
            OptixData.LP.renderDataMode = RenderDataFlags::NORMAL;
        }
        else if (option == std::string("entity_id")) {
            OptixData.LP.renderDataMode = RenderDataFlags::ENTITY_ID;
        }
        else if (option == std::string("base_color")) {
            OptixData.LP.renderDataMode = RenderDataFlags::BASE_COLOR;
        }
        else if (option == std::string("denoise_normal")) {
            OptixData.LP.renderDataMode = RenderDataFlags::DENOISE_NORMAL;
        }
        else if (option == std::string("denoise_albedo")) {
            OptixData.LP.renderDataMode = RenderDataFlags::DENOISE_ALBEDO;
        }
        else if (option == std::string("diffuse_color")) {
            OptixData.LP.renderDataMode = RenderDataFlags::DIFFUSE_COLOR;
        }
        else if (option == std::string("diffuse_direct_lighting")) {
            OptixData.LP.renderDataMode = RenderDataFlags::DIFFUSE_DIRECT_LIGHTING;
        }
        else if (option == std::string("diffuse_indirect_lighting")) {
            OptixData.LP.renderDataMode = RenderDataFlags::DIFFUSE_INDIRECT_LIGHTING;
        }
        else if (option == std::string("glossy_color")) {
            OptixData.LP.renderDataMode = RenderDataFlags::GLOSSY_COLOR;
        }
        else if (option == std::string("glossy_direct_lighting")) {
            OptixData.LP.renderDataMode = RenderDataFlags::GLOSSY_DIRECT_LIGHTING;
        }
        else if (option == std::string("glossy_indirect_lighting")) {
            OptixData.LP.renderDataMode = RenderDataFlags::GLOSSY_INDIRECT_LIGHTING;
        }
        else if (option == std::string("transmission_color")) {
            OptixData.LP.renderDataMode = RenderDataFlags::TRANSMISSION_COLOR;
        }
        else if (option == std::string("transmission_direct_lighting")) {
            OptixData.LP.renderDataMode = RenderDataFlags::TRANSMISSION_DIRECT_LIGHTING;
        }
        else if (option == std::string("transmission_indirect_lighting")) {
            OptixData.LP.renderDataMode = RenderDataFlags::TRANSMISSION_INDIRECT_LIGHTING;
        }
        else if (option == std::string("diffuse_motion_vectors")) {
            OptixData.LP.renderDataMode = RenderDataFlags::DIFFUSE_MOTION_VECTORS;
        }
        else if (option == std::string("heatmap")) {
            OptixData.LP.renderDataMode = RenderDataFlags::HEATMAP;
        }
        else {
            throw std::runtime_error(std::string("Error, unknown option : \"") + _option + std::string("\". ")
            + std::string("See documentation for available options"));
        }
        
        resizeOptixFrameBuffer(width, height);
        OptixData.LP.frameID = startFrame;
        OptixData.LP.renderDataBounce = bounce;
        OptixData.LP.seed = seed;
        updateComponents();

        for (uint32_t i = startFrame; i < frameCount; ++i) {
            // std::cout<<i<<std::endl;
            if (!ViSII.headlessMode) {
                auto glfw = Libraries::GLFW::Get();
                glfw->poll_events();
                glfw->swap_buffers("ViSII");
                glClearColor(1,1,1,1);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            }

            updateLaunchParams();
            traceRays();
            // Dont run denoiser to raw data rendering
            // if (OptixData.enableDenoiser)
            // {
            //     denoiseImage();
            // }

            if (!ViSII.headlessMode) {
                drawFrameBufferToWindow();
            }
        }

        synchronizeDevices();

        const glm::vec4 *fb = (const glm::vec4*) bufferGetPointer(OptixData.frameBuffer,0);
        for (uint32_t test = 0; test < frameBuffer.size(); test += 4) {
            frameBuffer[test + 0] = fb[test / 4].r;
            frameBuffer[test + 1] = fb[test / 4].g;
            frameBuffer[test + 2] = fb[test / 4].b;
            frameBuffer[test + 3] = fb[test / 4].a;
        }

        synchronizeDevices();

        OptixData.LP.renderDataMode = 0;
        OptixData.LP.renderDataBounce = 0;
        updateLaunchParams();
    };

    auto future = enqueueCommand(readFrameBuffer);
    future.wait();

    return frameBuffer;
}

// void renderDataToHDR(uint32_t width, uint32_t height, uint32_t startFrame, uint32_t frameCount, uint32_t bounce, std::string field, std::string imagePath)
// {
//     std::vector<float> fb = renderData(width, height, startFrame, frameCount, bounce, field);
//     stbi_flip_vertically_on_write(true);
//     stbi_write_hdr(imagePath.c_str(), width, height, /* num channels*/ 4, fb.data());
// }

void renderToHDR(uint32_t width, uint32_t height, uint32_t samplesPerPixel, std::string imagePath, uint32_t seed)
{
    std::vector<float> fb = render(width, height, samplesPerPixel, seed);
    stbi_flip_vertically_on_write(true);
    stbi_write_hdr(imagePath.c_str(), width, height, /* num channels*/ 4, fb.data());
}

float linearToSRGB(float x) {
    if (x <= 0.0031308f) {
		return 12.92f * x;
	}
	return 1.055f * pow(x, 1.f/2.4f) - 0.055f;
}

vec3 linearToSRGB(vec3 x) {
	x.r = linearToSRGB(x.r);
	x.g = linearToSRGB(x.g);
	x.b = linearToSRGB(x.b);
    return x;
}

// Tone Mapping
// From http://filmicgames.com/archives/75
vec3 Uncharted2Tonemap(vec3 x)
{
    x = max(x, vec3(0));
	float A = 0.15f;
	float B = 0.50f;
	float C = 0.10f;
	float D = 0.20f;
	float E_ = 0.02f;
	float F = 0.30f;
	return max(vec3(0.0f), ((x*(A*x+C*B)+D*E_)/(x*(A*x+B)+D*F))-E_/F);
}

void renderToPNG(uint32_t width, uint32_t height, uint32_t samplesPerPixel, std::string imagePath, uint32_t seed)
{
    // float exposure = 2.f; // TODO: expose as a parameter

    std::vector<float> fb = render(width, height, samplesPerPixel, seed);
    std::vector<uint8_t> colors(4 * width * height);
    for (size_t i = 0; i < (width * height); ++i) {     
        vec3 color = vec3(fb[i * 4 + 0], fb[i * 4 + 1], fb[i * 4 + 2]);
        float alpha = fb[i * 4 + 3];

        // color = Uncharted2Tonemap(color * exposure);
        // color = color * (1.0f / Uncharted2Tonemap(vec3(11.2f)));

        color = linearToSRGB(color);

        colors[i * 4 + 0] = uint8_t(glm::clamp(color.r * 255.f, 0.f, 255.f));
        colors[i * 4 + 1] = uint8_t(glm::clamp(color.g * 255.f, 0.f, 255.f));
        colors[i * 4 + 2] = uint8_t(glm::clamp(color.b * 255.f, 0.f, 255.f));
        colors[i * 4 + 3] = uint8_t(glm::clamp(alpha * 255.f, 0.f, 255.f));
    }
    stbi_flip_vertically_on_write(true);
    stbi_write_png(imagePath.c_str(), width, height, /* num channels*/ 4, colors.data(), /* stride in bytes */ width * 4);
}

// void renderDataToPNG(uint32_t width, uint32_t height, uint32_t startFrame, uint32_t frameCount, uint32_t bounce, std::string field, std::string imagePath)
// {
//     std::vector<float> fb = renderData(width, height, startFrame, frameCount, bounce, field);
//     std::vector<uint8_t> colors(4 * width * height);
//     for (size_t i = 0; i < (width * height); ++i) {       
//         colors[i * 4 + 0] = uint8_t(glm::clamp(fb[i * 4 + 0] * 255.f, 0.f, 255.f));
//         colors[i * 4 + 1] = uint8_t(glm::clamp(fb[i * 4 + 1] * 255.f, 0.f, 255.f));
//         colors[i * 4 + 2] = uint8_t(glm::clamp(fb[i * 4 + 2] * 255.f, 0.f, 255.f));
//         colors[i * 4 + 3] = uint8_t(glm::clamp(fb[i * 4 + 3] * 255.f, 0.f, 255.f));
//     }
//     stbi_flip_vertically_on_write(true);
//     stbi_write_png(imagePath.c_str(), width, height, /* num channels*/ 4, colors.data(), /* stride in bytes */ width * 4);
// }

void initializeComponentFactories()
{
    Camera::initializeFactory();
    Entity::initializeFactory();
    Transform::initializeFactory();
    Texture::initializeFactory();
    Material::initializeFactory();
    Mesh::initializeFactory();
    Light::initializeFactory();
}

void reproject(glm::vec4 *samplesBuffer, glm::vec4 *t0AlbedoBuffer, glm::vec4 *t1AlbedoBuffer, glm::vec4 *mvecBuffer, glm::vec4 *scratchBuffer, glm::vec4 *imageBuffer, int width, int height);

void initializeInteractive(bool windowOnTop, bool _verbose)
{
    // don't initialize more than once
    if (initialized == true) {
        throw std::runtime_error("Error: already initialized!");
    }

    initialized = true;
    stopped = false;
    verbose = _verbose;
    initializeComponentFactories();

    auto loop = [windowOnTop]() {
        ViSII.render_thread_id = std::this_thread::get_id();
        ViSII.headlessMode = false;

        auto glfw = Libraries::GLFW::Get();
        WindowData.window = glfw->create_window("ViSII", 512, 512, windowOnTop, true, true);
        WindowData.currentSize = WindowData.lastSize = ivec2(512, 512);
        glfw->make_context_current("ViSII");
        glfw->poll_events();

        initializeOptix(/*headless = */ false);

        initializeImgui();

        while (!stopped)
        {
            /* Poll events from the window */
            glfw->poll_events();
            glfw->swap_buffers("ViSII");
            glClearColor(1,1,1,1);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            updateFrameBuffer();
            updateComponents();
            updateLaunchParams();

            static double start=0;
            static double stop=0;
            start = glfwGetTime();
            traceRays();   

            if (OptixData.enableDenoiser)
            {
                denoiseImage();
            }        

            // glm::vec4* samplePtr = (glm::vec4*) bufferGetPointer(OptixData.accumBuffer,0);
            // glm::vec4* mvecPtr = (glm::vec4*) bufferGetPointer(OptixData.mvecBuffer,0);
            // glm::vec4* t0AlbPtr = (glm::vec4*) bufferGetPointer(OptixData.scratchBuffer,0);
            // glm::vec4* t1AlbPtr = (glm::vec4*) bufferGetPointer(OptixData.albedoBuffer,0);
            // glm::vec4* fbPtr = (glm::vec4*) bufferGetPointer(OptixData.frameBuffer,0);
            // glm::vec4* sPtr = (glm::vec4*) bufferGetPointer(OptixData.normalBuffer,0);
            // int width = OptixData.LP.frameSize.x;
            // int height = OptixData.LP.frameSize.y;
            // reproject(samplePtr, t0AlbPtr, t1AlbPtr, mvecPtr, sPtr, fbPtr, width, height);

            

            drawFrameBufferToWindow();
            stop = glfwGetTime();
            glfwSetWindowTitle(WindowData.window, std::to_string(1.f / (stop - start)).c_str());
            drawGUI();

            processCommandQueue();
            if (stopped) break;
        }

        ImGui::DestroyContext();
        if (glfw->does_window_exist("ViSII")) glfw->destroy_window("ViSII");
    };

    renderThread = thread(loop);

    // Waits for the render thread to start before returning
    auto wait = [] () {};
    auto future = enqueueCommand(wait);
    future.wait();
}

void initializeHeadless(bool _verbose)
{
    // don't initialize more than once
    if (initialized == true) {
        throw std::runtime_error("Error: already initialized!");
    }

    initialized = true;
    stopped = false;
    verbose = _verbose;

    initializeComponentFactories();

    auto loop = []() {
        ViSII.render_thread_id = std::this_thread::get_id();
        ViSII.headlessMode = true;

        initializeOptix(/*headless = */ true);

        while (!stopped)
        {
            processCommandQueue();
            if (stopped) break;
        }
    };

    renderThread = thread(loop);

    // Waits for the render thread to start before returning
    auto wait = [] () {};
    auto future = enqueueCommand(wait);
    future.wait();
}

void initialize(bool headless, bool windowOnTop, bool verbose) {
    if (headless) initializeHeadless(verbose);
    else initializeInteractive(windowOnTop, verbose);
}

void clearAll()
{
    setCameraEntity(nullptr);
    Entity::clearAll();
    Transform::clearAll();
    Material::clearAll();
    Texture::clearAll();
    Mesh::clearAll();
    Camera::clearAll();
    Light::clearAll();
}

glm::vec3 getSceneMinAabbCorner() {
    return OptixData.LP.sceneBBMin;
}

glm::vec3 getSceneMaxAabbCorner() {
    return OptixData.LP.sceneBBMax;
}

glm::vec3 getSceneAabbCenter() {
    return OptixData.LP.sceneBBMin + (OptixData.LP.sceneBBMax - OptixData.LP.sceneBBMin) * .5f;
}

void updateSceneAabb(Entity* entity)
{
    // If updated entity AABB lies within scene AABB, return. 
    glm::vec3 bbmin = entity->getMinAabbCorner();
    glm::vec3 bbmax = entity->getMaxAabbCorner();

    if (glm::all(glm::greaterThan(bbmin, glm::vec3(OptixData.LP.sceneBBMin))) && 
        glm::all(glm::lessThan(bbmax, glm::vec3(OptixData.LP.sceneBBMax)))) return;

    // otherwise, recompute scene AABB
    bool first = true;
    auto entities = Entity::getRenderableEntities();
    for (auto &e : entities) {
        OptixData.LP.sceneBBMin = (first) ? e->getMinAabbCorner() : 
          glm::min(OptixData.LP.sceneBBMin, e->getMinAabbCorner());
        OptixData.LP.sceneBBMax = (first) ? e->getMaxAabbCorner() : 
          glm::max(OptixData.LP.sceneBBMax, e->getMaxAabbCorner());
        first = false;
    }
}

#ifdef __unix__
# include <unistd.h>
#elif defined _WIN32
# include <windows.h>
#define sleep(x) Sleep(1000 * (x))
#endif

void deinitialize()
{
    if (initialized == true) {
        /* cleanup window if open */
        if (stopped == false) {
            stopped = true;
            renderThread.join();
        }
        if (OptixData.denoiser)
            OPTIX_CHECK(optixDenoiserDestroy(OptixData.denoiser));
        clearAll();
    }
    else {
        throw std::runtime_error("Error: already deinitialized!");
    }
    initialized = false;
    // sleeping here. 
    // Some strange bug with python where deinitialize immediately before interpreter exit
    // on windows causes lockup. The sleep here fixes that lockup, suggesting some race condition...
    sleep(1); 
}

void __test__(std::vector<std::string> args) {
    if (args.size() != 1) return;

    std::string option = args[0];

    if (option == std::string("none")) {
        OptixData.LP.renderDataMode = RenderDataFlags::NONE;
    }
    else if (option == std::string("depth")) {
        OptixData.LP.renderDataMode = RenderDataFlags::DEPTH;
    }
    else if (option == std::string("ray_direction")) {
        OptixData.LP.renderDataMode = RenderDataFlags::RAY_DIRECTION;
    }
    else if (option == std::string("position")) {
        OptixData.LP.renderDataMode = RenderDataFlags::POSITION;
    }
    else if (option == std::string("normal")) {
        OptixData.LP.renderDataMode = RenderDataFlags::NORMAL;
    }
    else if (option == std::string("entity_id")) {
        OptixData.LP.renderDataMode = RenderDataFlags::ENTITY_ID;
    }
    else if (option == std::string("base_color")) {
        OptixData.LP.renderDataMode = RenderDataFlags::BASE_COLOR;
    }
    else if (option == std::string("denoise_normal")) {
        OptixData.LP.renderDataMode = RenderDataFlags::DENOISE_NORMAL;
    }
    else if (option == std::string("denoise_albedo")) {
        OptixData.LP.renderDataMode = RenderDataFlags::DENOISE_ALBEDO;
    }
    else if (option == std::string("diffuse_color")) {
        OptixData.LP.renderDataMode = RenderDataFlags::DIFFUSE_COLOR;
    }
    else if (option == std::string("diffuse_direct_lighting")) {
        OptixData.LP.renderDataMode = RenderDataFlags::DIFFUSE_DIRECT_LIGHTING;
    }
    else if (option == std::string("diffuse_indirect_lighting")) {
        OptixData.LP.renderDataMode = RenderDataFlags::DIFFUSE_INDIRECT_LIGHTING;
    }
    else if (option == std::string("glossy_color")) {
        OptixData.LP.renderDataMode = RenderDataFlags::GLOSSY_COLOR;
    }
    else if (option == std::string("glossy_direct_lighting")) {
        OptixData.LP.renderDataMode = RenderDataFlags::GLOSSY_DIRECT_LIGHTING;
    }
    else if (option == std::string("glossy_indirect_lighting")) {
        OptixData.LP.renderDataMode = RenderDataFlags::GLOSSY_INDIRECT_LIGHTING;
    }
    else if (option == std::string("transmission_color")) {
        OptixData.LP.renderDataMode = RenderDataFlags::TRANSMISSION_COLOR;
    }
    else if (option == std::string("transmission_direct_lighting")) {
        OptixData.LP.renderDataMode = RenderDataFlags::TRANSMISSION_DIRECT_LIGHTING;
    }
    else if (option == std::string("transmission_indirect_lighting")) {
        OptixData.LP.renderDataMode = RenderDataFlags::TRANSMISSION_INDIRECT_LIGHTING;
    }
    else if (option == std::string("diffuse_motion_vectors")) {
        OptixData.LP.renderDataMode = RenderDataFlags::DIFFUSE_MOTION_VECTORS;
    }
    else if (option == std::string("heatmap")) {
        OptixData.LP.renderDataMode = RenderDataFlags::HEATMAP;
    }
    else {
        throw std::runtime_error(std::string("Error, unknown option : \"") + option + std::string("\". ")
        + std::string("See documentation for available options"));
    }

    resetAccumulation();
}