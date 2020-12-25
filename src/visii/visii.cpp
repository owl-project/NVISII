#undef NDEBUG

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

#include <glm/gtc/color_space.hpp>

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
#include <functional>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

// Assimp already seems to define this
#ifndef TINYEXR_IMPLEMENTATION
#define MINIZ_HEADER_FILE_ONLY
// #define TINYEXR_USE_MINIZ 0
// #include "zlib.h"
#define TINYEXR_IMPLEMENTATION
#endif
#include <tinyexr.h>

#define USE_AOV
#define USE_OPTIX72
// #undef USE_OPTIX71

// #define __optix_optix_function_table_h__
#include <optix_stubs.h>
// OptixFunctionTable g_optixFunctionTable;

// #include <thrust/reduce.h>
// #include <thrust/execution_policy.h>
// #include <thrust/device_vector.h>
// #include <thrust/device_ptr.h>

namespace visii {

// extern optixDenoiserSetModel;
std::promise<void> exitSignal;
std::thread renderThread;
static bool initialized = false;
static bool stopped = true;
static bool lazyUpdatesEnabled = false;
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
    OWLBuffer volumeBuffer;
    OWLBuffer lightEntitiesBuffer;
    OWLBuffer surfaceInstanceToEntityBuffer;
    OWLBuffer volumeInstanceToEntityBuffer;
    OWLBuffer vertexListsBuffer;
    OWLBuffer normalListsBuffer;
    OWLBuffer tangentListsBuffer;
    OWLBuffer texCoordListsBuffer;
    OWLBuffer indexListsBuffer;
    OWLBuffer textureObjectsBuffer;
    OWLBuffer volumeHandlesBuffer;

    std::vector<OWLTexture> textureObjects;
    std::vector<TextureStruct> textureStructs;

    std::vector<OWLBuffer> volumeHandles;

    uint32_t numLightEntities;

    OWLRayGen rayGen;
    OWLMissProg missProg;
    OWLGeomType trianglesGeomType;
    OWLGeomType volumeGeomType;
    OWLGeomType cdfTrianglesGeomType;

    std::vector<OWLBuffer> vertexLists;
    std::vector<OWLBuffer> normalLists;
    std::vector<OWLBuffer> tangentLists;
    std::vector<OWLBuffer> texCoordLists;
    std::vector<OWLBuffer> indexLists;
    std::vector<OWLGeom> surfaceGeomList;
    std::vector<OWLGroup> surfaceBlasList;
    std::vector<OWLGeom> volumeGeomList;
    std::vector<OWLGroup> volumeBlasList;

    OWLGroup surfacesIAS = nullptr;
    OWLGroup volumesIAS = nullptr;

    std::vector<uint32_t> lightEntities;

    bool enableDenoiser = false;
    bool enableKernelPrediction = false;
    bool enableAlbedoGuide = true;
    bool enableNormalGuide = true;
    OptixDenoiserSizes denoiserSizes;
    OptixDenoiser denoiser;
    OWLBuffer denoiserScratchBuffer;
    OWLBuffer denoiserStateBuffer;
    OWLBuffer hdrIntensityBuffer;
    OWLBuffer colorAvgBuffer;

    Texture* domeLightTexture = nullptr;

    OWLBuffer environmentMapRowsBuffer;
    OWLBuffer environmentMapColsBuffer;
    OWLTexture proceduralSkyTexture;

    std::vector<MaterialStruct> materialStructs;

    OWLBuffer placeholder;
    OWLGroup placeholderGroup;
    OWLGroup placeholderUserGroup;

    OWLBuffer environmentMapCDFIndices;
    OWLBuffer environmentMapCDFVertices;
    OWLGeom environmentMapCDFGeom;
    OWLGroup environmentMapCDF;
    OWLGroup environmentMapCDFIAS;
} OptixData;

static struct ViSII {
    struct Command {
        std::function<void()> function;
        std::shared_ptr<std::promise<void>> promise;
    };

    std::thread::id render_thread_id;
    std::condition_variable cv;
    std::recursive_mutex qMutex;
    std::queue<Command> commandQueue = {};
    bool headlessMode;
    std::function<void()> callback;
    std::recursive_mutex callbackMutex;

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

owl4x3f glmToOWL(glm::mat4 &xfm){
    owl4x3f oxfm = {
        {xfm[0][0], xfm[0][1], xfm[0][2]}, 
        {xfm[1][0], xfm[1][1], xfm[1][2]}, 
        {xfm[2][0], xfm[2][1], xfm[2][2]},
        {xfm[3][0], xfm[3][1], xfm[3][2]}};
    return oxfm;
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

void checkForErrors()
{
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }    
}

void initializeFrameBuffer(int fbWidth, int fbHeight) {
    fbWidth = glm::max(fbWidth, 1);
    fbHeight = glm::max(fbHeight, 1);
    synchronizeDevices();

    auto &OD = OptixData;
    if (OD.imageTexID != -1) {
        cudaGraphicsUnregisterResource(OD.cudaResourceTex);
        glDeleteTextures(1, &OD.imageTexID);
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
    OD.context = owlContextCreate(/*requested Device IDs*/ nullptr, /* Num Devices */  0);

    int numGPUsFound = owlGetDeviceCount(OD.context);
    if (verbose) {
        std::cout<<"Found " << numGPUsFound << " GPUs available for rendering."<<std::endl;
    }

    owlEnableMotionBlur(OD.context);
    owlContextSetRayTypeCount(OD.context, 2);
    cudaSetDevice(0); // OWL leaves the device as num_devices - 1 after the context is created. set it back to 0.
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
        { "surfacesIAS",             OWL_GROUP,                         OWL_OFFSETOF(LaunchParams, surfacesIAS)},
        { "volumesIAS",              OWL_GROUP,                         OWL_OFFSETOF(LaunchParams, volumesIAS)},
        { "cameraEntity",            OWL_USER_TYPE(EntityStruct),       OWL_OFFSETOF(LaunchParams, cameraEntity)},
        { "entities",                OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, entities)},
        { "transforms",              OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, transforms)},
        { "cameras",                 OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, cameras)},
        { "materials",               OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, materials)},
        { "meshes",                  OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, meshes)},
        { "lights",                  OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, lights)},
        { "textures",                OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, textures)},
        { "volumes",                 OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, volumes)},
        { "lightEntities",           OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, lightEntities)},
        { "vertexLists",             OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, vertexLists)},
        { "normalLists",             OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, normalLists)},
        { "tangentLists",            OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, tangentLists)},
        { "texCoordLists",           OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, texCoordLists)},
        { "indexLists",              OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, indexLists)},
        { "numLightEntities",        OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, numLightEntities)},
        { "surfaceInstanceToEntity", OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, surfaceInstanceToEntity)},
        { "volumeInstanceToEntity",  OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, volumeInstanceToEntity)},
        { "domeLightIntensity",      OWL_USER_TYPE(float),              OWL_OFFSETOF(LaunchParams, domeLightIntensity)},
        { "domeLightExposure",       OWL_USER_TYPE(float),              OWL_OFFSETOF(LaunchParams, domeLightExposure)},
        { "domeLightColor",          OWL_USER_TYPE(glm::vec3),          OWL_OFFSETOF(LaunchParams, domeLightColor)},
        { "directClamp",             OWL_USER_TYPE(float),              OWL_OFFSETOF(LaunchParams, directClamp)},
        { "indirectClamp",           OWL_USER_TYPE(float),              OWL_OFFSETOF(LaunchParams, indirectClamp)},
        { "maxDiffuseBounceDepth",   OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, maxDiffuseBounceDepth)},
        { "maxSpecularBounceDepth",  OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, maxSpecularBounceDepth)},
        { "maxVolumeBounceDepth",    OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, maxVolumeBounceDepth)},
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
        { "environmentMapCDFIAS",    OWL_GROUP,                         OWL_OFFSETOF(LaunchParams, environmentMapCDFIAS)},
        { "textureObjects",          OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, textureObjects)},
        { "volumeHandles",           OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, volumeHandles)},
        { "proceduralSkyTexture",    OWL_TEXTURE,                       OWL_OFFSETOF(LaunchParams, proceduralSkyTexture)},
        { "GGX_E_AVG_LOOKUP",        OWL_TEXTURE,                       OWL_OFFSETOF(LaunchParams, GGX_E_AVG_LOOKUP)},
        { "GGX_E_LOOKUP",            OWL_TEXTURE,                       OWL_OFFSETOF(LaunchParams, GGX_E_LOOKUP)},
        { "renderDataMode",          OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, renderDataMode)},
        { "renderDataBounce",        OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, renderDataBounce)},
        { "sceneBBMin",              OWL_USER_TYPE(glm::vec3),          OWL_OFFSETOF(LaunchParams, sceneBBMin)},
        { "sceneBBMax",              OWL_USER_TYPE(glm::vec3),          OWL_OFFSETOF(LaunchParams, sceneBBMax)},
        { "enableDomeSampling", OWL_USER_TYPE(bool),               OWL_OFFSETOF(LaunchParams, enableDomeSampling)},
        { /* sentinel to mark end of list */ }
    };
    OD.launchParams = launchParamsCreate(OD.context, sizeof(LaunchParams), launchParamVars, -1);
    
    /* Create AOV Buffers */
    if (!headless) {
        initializeFrameBuffer(512, 512);        
    }

    if (numGPUsFound > 1) {
        OD.frameBuffer = managedMemoryBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
        OD.accumBuffer = managedMemoryBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
        OD.normalBuffer = managedMemoryBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
        OD.albedoBuffer = managedMemoryBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
        OD.scratchBuffer = managedMemoryBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
        OD.mvecBuffer = managedMemoryBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
    } else {
        OD.frameBuffer = deviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
        OD.accumBuffer = deviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
        OD.normalBuffer = deviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
        OD.albedoBuffer = deviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
        OD.scratchBuffer = deviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
        OD.mvecBuffer = deviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
    }
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
    OD.entityBuffer              = deviceBufferCreate(OD.context, OWL_USER_TYPE(EntityStruct),        Entity::getCount(),   nullptr);
    OD.transformBuffer           = deviceBufferCreate(OD.context, OWL_USER_TYPE(TransformStruct),     Transform::getCount(), nullptr);
    OD.cameraBuffer              = deviceBufferCreate(OD.context, OWL_USER_TYPE(CameraStruct),        Camera::getCount(),    nullptr);
    OD.materialBuffer            = deviceBufferCreate(OD.context, OWL_USER_TYPE(MaterialStruct),      Material::getCount(),  nullptr);
    OD.meshBuffer                = deviceBufferCreate(OD.context, OWL_USER_TYPE(MeshStruct),          Mesh::getCount(),     nullptr);
    OD.lightBuffer               = deviceBufferCreate(OD.context, OWL_USER_TYPE(LightStruct),         Light::getCount(),     nullptr);
    OD.textureBuffer             = deviceBufferCreate(OD.context, OWL_USER_TYPE(TextureStruct),       Texture::getCount() + NUM_MAT_PARAMS * Material::getCount(),   nullptr);
    OD.volumeBuffer              = deviceBufferCreate(OD.context, OWL_USER_TYPE(VolumeStruct),        Volume::getCount(),   nullptr);
    OD.volumeHandlesBuffer       = deviceBufferCreate(OD.context, OWL_BUFFER,                         Volume::getCount(),   nullptr);
    OD.lightEntitiesBuffer       = deviceBufferCreate(OD.context, OWL_USER_TYPE(uint32_t),            1,              nullptr);
    OD.surfaceInstanceToEntityBuffer = deviceBufferCreate(OD.context, OWL_USER_TYPE(uint32_t),            1,              nullptr);
    OD.volumeInstanceToEntityBuffer = deviceBufferCreate(OD.context, OWL_USER_TYPE(uint32_t),            1,              nullptr);
    OD.vertexListsBuffer         = deviceBufferCreate(OD.context, OWL_BUFFER,                         Mesh::getCount(),     nullptr);
    OD.normalListsBuffer         = deviceBufferCreate(OD.context, OWL_BUFFER,                         Mesh::getCount(),     nullptr);
    OD.tangentListsBuffer        = deviceBufferCreate(OD.context, OWL_BUFFER,                         Mesh::getCount(),     nullptr);
    OD.texCoordListsBuffer       = deviceBufferCreate(OD.context, OWL_BUFFER,                         Mesh::getCount(),     nullptr);
    OD.indexListsBuffer          = deviceBufferCreate(OD.context, OWL_BUFFER,                         Mesh::getCount(),     nullptr);
    OD.textureObjectsBuffer      = deviceBufferCreate(OD.context, OWL_TEXTURE,                        Texture::getCount() + NUM_MAT_PARAMS * Material::getCount(),   nullptr);

    launchParamsSetBuffer(OD.launchParams, "entities",             OD.entityBuffer);
    launchParamsSetBuffer(OD.launchParams, "transforms",           OD.transformBuffer);
    launchParamsSetBuffer(OD.launchParams, "cameras",              OD.cameraBuffer);
    launchParamsSetBuffer(OD.launchParams, "materials",            OD.materialBuffer);
    launchParamsSetBuffer(OD.launchParams, "meshes",               OD.meshBuffer);
    launchParamsSetBuffer(OD.launchParams, "lights",               OD.lightBuffer);
    launchParamsSetBuffer(OD.launchParams, "textures",             OD.textureBuffer);
    launchParamsSetBuffer(OD.launchParams, "volumes",              OD.volumeBuffer);
    launchParamsSetBuffer(OD.launchParams, "lightEntities",        OD.lightEntitiesBuffer);
    launchParamsSetBuffer(OD.launchParams, "surfaceInstanceToEntity",  OD.surfaceInstanceToEntityBuffer);
    launchParamsSetBuffer(OD.launchParams, "volumeInstanceToEntity",  OD.volumeInstanceToEntityBuffer);
    launchParamsSetBuffer(OD.launchParams, "vertexLists",          OD.vertexListsBuffer);
    launchParamsSetBuffer(OD.launchParams, "normalLists",          OD.normalListsBuffer);
    launchParamsSetBuffer(OD.launchParams, "tangentLists",          OD.tangentListsBuffer);
    launchParamsSetBuffer(OD.launchParams, "texCoordLists",        OD.texCoordListsBuffer);
    launchParamsSetBuffer(OD.launchParams, "indexLists",           OD.indexListsBuffer);
    launchParamsSetBuffer(OD.launchParams, "textureObjects",       OD.textureObjectsBuffer);
    launchParamsSetBuffer(OD.launchParams, "volumeHandles",       OD.volumeHandlesBuffer);

    uint32_t meshCount = Mesh::getCount();
    OD.vertexLists.resize(meshCount);
    OD.normalLists.resize(meshCount);
    OD.tangentLists.resize(meshCount);
    OD.texCoordLists.resize(meshCount);
    OD.indexLists.resize(meshCount);
    OD.surfaceGeomList.resize(meshCount);
    OD.surfaceBlasList.resize(meshCount);
    
    uint32_t volumeCount = Volume::getCount();
    OD.volumeGeomList.resize(volumeCount);
    OD.volumeBlasList.resize(volumeCount);

    uint32_t materialCount = Material::getCount();
    OD.textureObjects.resize(Texture::getCount() + NUM_MAT_PARAMS * materialCount, nullptr);        
    OD.textureStructs.resize(Texture::getCount() + NUM_MAT_PARAMS * materialCount);
    OD.materialStructs.resize(materialCount);

    OD.volumeHandles.resize(Volume::getCount());

    OD.LP.environmentMapID = -1;
    OD.LP.environmentMapRotation = glm::quat(1,0,0,0);
    launchParamsSetRaw(OD.launchParams, "environmentMapID", &OD.LP.environmentMapID);
    launchParamsSetRaw(OD.launchParams, "environmentMapRotation", &OD.LP.environmentMapRotation);

    launchParamsSetBuffer(OD.launchParams, "environmentMapRows", OD.environmentMapRowsBuffer);
    launchParamsSetBuffer(OD.launchParams, "environmentMapCols", OD.environmentMapColsBuffer);
    launchParamsSetRaw(OD.launchParams, "environmentMapWidth", &OD.LP.environmentMapWidth);
    launchParamsSetRaw(OD.launchParams, "environmentMapHeight", &OD.LP.environmentMapHeight);

    // OWLTexture GGX_E_AVG_LOOKUP = owlTexture2DCreate(OD.context,
    //                         OWL_TEXEL_FORMAT_R32F,
    //                         GGX_E_avg_size,1,
    //                         GGX_E_avg,
    //                         OWL_TEXTURE_LINEAR,
    //                         OWL_COLOR_SPACE_LINEAR,
    //                         OWL_TEXTURE_CLAMP);
    // OWLTexture GGX_E_LOOKUP = owlTexture2DCreate(OD.context,
    //                         OWL_TEXEL_FORMAT_R32F,
    //                         GGX_E_size[0],GGX_E_size[1],
    //                         GGX_E,
    //                         OWL_TEXTURE_LINEAR,
    //                         OWL_TEXTURE_CLAMP,
    //                         OWL_COLOR_SPACE_LINEAR);
    // launchParamsSetTexture(OD.launchParams, "GGX_E_AVG_LOOKUP", GGX_E_AVG_LOOKUP);
    // launchParamsSetTexture(OD.launchParams, "GGX_E_LOOKUP",     GGX_E_LOOKUP);
    
    OD.LP.numLightEntities = uint32_t(OD.lightEntities.size());
    launchParamsSetRaw(OD.launchParams, "numLightEntities", &OD.LP.numLightEntities);
    launchParamsSetRaw(OD.launchParams, "domeLightIntensity", &OD.LP.domeLightIntensity);
    launchParamsSetRaw(OD.launchParams, "domeLightExposure", &OD.LP.domeLightExposure);
    launchParamsSetRaw(OD.launchParams, "domeLightColor", &OD.LP.domeLightColor);
    launchParamsSetRaw(OD.launchParams, "directClamp", &OD.LP.directClamp);
    launchParamsSetRaw(OD.launchParams, "indirectClamp", &OD.LP.indirectClamp);
    launchParamsSetRaw(OD.launchParams, "maxDiffuseBounceDepth", &OD.LP.maxDiffuseBounceDepth);
    launchParamsSetRaw(OD.launchParams, "maxSpecularBounceDepth", &OD.LP.maxSpecularBounceDepth);
    launchParamsSetRaw(OD.launchParams, "maxVolumeBounceDepth", &OD.LP.maxVolumeBounceDepth);
    launchParamsSetRaw(OD.launchParams, "numLightSamples", &OD.LP.numLightSamples);
    launchParamsSetRaw(OD.launchParams, "seed", &OD.LP.seed);
    launchParamsSetRaw(OD.launchParams, "xPixelSamplingInterval", &OD.LP.xPixelSamplingInterval);
    launchParamsSetRaw(OD.launchParams, "yPixelSamplingInterval", &OD.LP.yPixelSamplingInterval);
    launchParamsSetRaw(OD.launchParams, "timeSamplingInterval", &OD.LP.timeSamplingInterval);

    OWLVarDecl trianglesGeomVars[] = {{/* sentinel to mark end of list */}};
    OD.trianglesGeomType = geomTypeCreate(OD.context, OWL_GEOM_TRIANGLES, sizeof(TrianglesGeomData), trianglesGeomVars,-1);
    OWLVarDecl volumeGeomVars[] = {
        { "bbmin", OWL_USER_TYPE(glm::vec4), OWL_OFFSETOF(VolumeGeomData, bbmin)},
        { "bbmax", OWL_USER_TYPE(glm::vec4), OWL_OFFSETOF(VolumeGeomData, bbmax)},
        { "volumeID", OWL_USER_TYPE(uint32_t), OWL_OFFSETOF(VolumeGeomData, volumeID)},
        {/* sentinel to mark end of list */}
    };
    OD.volumeGeomType = owlGeomTypeCreate(OD.context, OWL_GEOM_USER, sizeof(VolumeGeomData), volumeGeomVars, -1);
    OD.cdfTrianglesGeomType = owlGeomTypeCreate(OD.context, OWL_GEOM_TRIANGLES, sizeof(TrianglesGeomData), trianglesGeomVars,-1);

    geomTypeSetClosestHit(OD.trianglesGeomType, /*ray type */ 0, OD.module,"TriangleMesh");
    geomTypeSetClosestHit(OD.trianglesGeomType, /*ray type */ 1, OD.module,"ShadowRay");
    geomTypeSetClosestHit(OD.cdfTrianglesGeomType, /*ray type */ 0, OD.module,"ShadowRay"); // might change
    geomTypeSetClosestHit(OD.cdfTrianglesGeomType, /*ray type */ 1, OD.module,"ShadowRay"); // might change
    owlGeomTypeSetClosestHit(OD.volumeGeomType, /*ray type */ 0, OD.module,"VolumeMesh");
    owlGeomTypeSetClosestHit(OD.volumeGeomType, /*ray type */ 1, OD.module,"VolumeShadowRay");
    owlGeomTypeSetIntersectProg(OD.volumeGeomType, /*ray type */ 0, OD.module,"VolumeIntersection");
    owlGeomTypeSetIntersectProg(OD.volumeGeomType, /*ray type */ 1, OD.module,"VolumeIntersection");
    owlGeomTypeSetBoundsProg(OD.volumeGeomType, OD.module, "VolumeBounds");

    // Setup miss prog 
    OWLVarDecl missProgVars[] = {{ /* sentinel to mark end of list */ }};
    OD.missProg = missProgCreate(OD.context,OD.module,"miss",sizeof(MissProgData),missProgVars,-1);
    
    // Setup ray gen program
    OWLVarDecl rayGenVars[] = {
        { "deviceIndex",   OWL_DEVICE, OWL_OFFSETOF(RayGenData, deviceIndex)}, // this var is automatically set
        { "deviceCount",   OWL_INT,    OWL_OFFSETOF(RayGenData, deviceCount)},
        { /* sentinel to mark end of list */ }};
    OD.rayGen = rayGenCreate(OD.context,OD.module,"rayGen", sizeof(RayGenData), rayGenVars,-1);
    owlRayGenSet1i(OD.rayGen, "deviceCount",  numGPUsFound);

    buildPrograms(OD.context);
    
    /* Temporary GAS. Required for certain older driver versions. */
    const int NUM_VERTICES = 1;
    vec3 vertices[NUM_VERTICES] = {{ 0.f, 0.f, 0.f }};
    const int NUM_INDICES = 1;
    ivec3 indices[NUM_INDICES] = {{ 0, 0, 0 }};    
    OWLBuffer vertexBuffer = deviceBufferCreate(OD.context,OWL_FLOAT4,NUM_VERTICES,vertices);
    OWLBuffer indexBuffer = deviceBufferCreate(OD.context,OWL_INT3,NUM_INDICES,indices);
    OWLGeom trianglesGeom = geomCreate(OD.context,OD.trianglesGeomType);
    trianglesSetVertices(trianglesGeom,vertexBuffer,NUM_VERTICES,sizeof(vec4),0);
    trianglesSetIndices(trianglesGeom,indexBuffer, NUM_INDICES,sizeof(ivec3),0);
    OD.placeholderGroup = trianglesGeomGroupCreate(OD.context,1,&trianglesGeom);
    groupBuildAccel(OD.placeholderGroup);

    // build IAS
    OWLGroup surfacesIAS = instanceGroupCreate(OD.context, 1);
    instanceGroupSetChild(surfacesIAS, 0, OD.placeholderGroup); 
    groupBuildAccel(surfacesIAS);
    launchParamsSetGroup(OD.launchParams, "surfacesIAS", surfacesIAS);

    OWLGeom userGeom = owlGeomCreate(OD.context, OD.volumeGeomType);
    owlGeomSetPrimCount(userGeom, 1);
    glm::vec4 tmpbbmin(1.f), tmpbbmax(-1.f); // unhittable
    owlGeomSetRaw(userGeom, "bbmin", &tmpbbmin);
    owlGeomSetRaw(userGeom, "bbmax", &tmpbbmax);
    OD.placeholderUserGroup = owlUserGeomGroupCreate(OD.context, 1, &userGeom);
    groupBuildAccel(OD.placeholderUserGroup);

    OWLGroup volumesIAS = instanceGroupCreate(OD.context, 1);
    instanceGroupSetChild(volumesIAS, 0, OD.placeholderUserGroup); 
    groupBuildAccel(volumesIAS);
    launchParamsSetGroup(OD.launchParams, "volumesIAS", volumesIAS);

    // TESTING importance sample using RT core CDF idea
    launchParamsSetGroup(OD.launchParams, "environmentMapCDFIAS", surfacesIAS); // placeholder CDF accel

    // Build *SBT* required to trace the groups   
    buildPipeline(OD.context);
    buildSBT(OD.context);

    // Setup denoiser
    OptixDenoiserOptions options;
    options.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;
    #ifndef USE_OPTIX72
    options.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    #endif
    auto optixContext = getOptixContext(OD.context, 0);
    auto cudaStream = getStream(OD.context, 0);
    OPTIX_CHECK(optixDenoiserCreate(optixContext, &options, &OD.denoiser));
    
    OptixDenoiserModelKind kind = OPTIX_DENOISER_MODEL_KIND_HDR; // having troubles with the AOV denoiser again...
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
    OD.colorAvgBuffer = deviceBufferCreate(OD.context, OWL_USER_TYPE(float),
        4, nullptr);
        

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
    // if (ViSII.render_thread_id != std::this_thread::get_id()) 
    std::lock_guard<std::recursive_mutex> lock(ViSII.qMutex);

    ViSII::Command c;
    c.function = function;
    c.promise = std::make_shared<std::promise<void>>();
    auto new_future = c.promise->get_future();
    ViSII.commandQueue.push(c);
    // cv.notify_one();
    return new_future;
}

void enqueueCommandAndWait(std::function<void()> function)
{
    if (ViSII.render_thread_id != std::this_thread::get_id()) {
        if (ViSII.callback) {
            throw std::runtime_error(
                std::string("Error: calling a blocking function while callback set, which would otherwise result in a ")
                + std::string("deadlock. To work around this issue, either temporarily clear the callback, or ")
                + std::string("alternatively call this function from within the callback.")
            );
        }
        enqueueCommand(function).wait();
    } else {
        function();
    }
}

void processCommandQueue()
{
    std::lock_guard<std::recursive_mutex> lock(ViSII.qMutex);
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

void setDomeLightExposure(float exposure)
{
    OptixData.LP.domeLightExposure = exposure;
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
    resetAccumulation();
    enqueueCommand([] () {
        OptixData.LP.environmentMapID = -1;
        if (OptixData.environmentMapRowsBuffer) owlBufferRelease(OptixData.environmentMapRowsBuffer);
        if (OptixData.environmentMapColsBuffer) owlBufferRelease(OptixData.environmentMapColsBuffer);
        OptixData.environmentMapRowsBuffer = nullptr;
        OptixData.environmentMapColsBuffer = nullptr;
        OptixData.LP.environmentMapWidth = -1;
        OptixData.LP.environmentMapHeight = -1;  
    });
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
    enqueueCommand([sunPos, skyTint, atmosphereThickness, saturation] () {
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

        // float invWidth = 1.f / float(width);
        // float invHeight = 1.f / float(height);
        // float invjacobian = width * height / float(4 * M_PI);

        // auto rows = std::vector<float>(height);
        // auto cols = std::vector<float>(width * height);
        // for (int y = 0, i = 0; y < height; y++) {
        //     for (int x = 0; x < width; x++, i++) {
        //         cols[i] = std::max(texels[i].r, std::max(texels[i].g, texels[i].b)) + ((x > 0) ? cols[i - 1] : 0.f);
        //     }
        //     rows[y] = cols[i - 1] + ((y > 0) ? rows[y - 1] : 0.0f);
        //     // normalize the pdf for this scanline (if it was non-zero)
        //     if (cols[i - 1] > 0) {
        //         for (int x = 0; x < width; x++) {
        //             cols[i - width + x] /= cols[i - 1];
        //         }
        //     }
        // }

        // // normalize the pdf across all scanlines
        // for (int y = 0; y < height; y++)
        //     rows[y] /= rows[height - 1];
        
        // if (OptixData.environmentMapRowsBuffer) owlBufferRelease(OptixData.environmentMapRowsBuffer);
        // if (OptixData.environmentMapColsBuffer) owlBufferRelease(OptixData.environmentMapColsBuffer);
        // OptixData.environmentMapRowsBuffer = owlDeviceBufferCreate(OptixData.context, OWL_USER_TYPE(float), height, rows.data());
        // OptixData.environmentMapColsBuffer = owlDeviceBufferCreate(OptixData.context, OWL_USER_TYPE(float), width * height, cols.data());
        // OptixData.LP.environmentMapWidth = width;
        // OptixData.LP.environmentMapHeight = height;  

        OptixData.LP.environmentMapWidth = 0;
        OptixData.LP.environmentMapHeight = 0;  
        resetAccumulation();
    });
}

void setDomeLightTexture(Texture* texture, bool enableCDF)
{
    enqueueCommand([texture, enableCDF] () {
        OptixData.LP.environmentMapID = texture->getId();
        if (enableCDF) {
            std::vector<glm::vec4> texels = texture->getFloatTexels();

            int width = texture->getWidth();
            int height = texture->getHeight();
            int cdfWidth = width;
            int cdfHeight = height;

            float invWidth = 1.f / float(cdfWidth);
            float invHeight = 1.f / float(cdfHeight);
            float invjacobian = cdfWidth * cdfHeight / float(4 * M_PI);

            auto rows = std::vector<float>(cdfHeight);
            auto cols = std::vector<float>(cdfWidth * cdfHeight);
            for (int y = 0, i = 0; y < cdfHeight; y++) {
                for (int x = 0; x < cdfWidth; x++, i++) {
                    glm::vec4 texel = texels[i];
                    cols[i] = std::max(texel.r, std::max(texel.g, texel.b)) + ((x > 0) ? cols[i - 1] : 0.f);
                }
                rows[y] = cols[i - 1] + ((y > 0) ? rows[y - 1] : 0.0f);
                // normalize the pdf for this scanline (if it was non-zero)
                if (cols[i - 1] > 0) {
                    for (int x = 0; x < cdfWidth; x++) {
                        cols[i - cdfWidth + x] /= cols[i - 1];
                    }
                }
            }

            // normalize the pdf across all scanlines
            for (int y = 0; y < cdfHeight; y++) rows[y] /= rows[cdfHeight - 1];

            if (OptixData.environmentMapRowsBuffer) owlBufferRelease(OptixData.environmentMapRowsBuffer);
            if (OptixData.environmentMapColsBuffer) owlBufferRelease(OptixData.environmentMapColsBuffer);
            OptixData.environmentMapRowsBuffer = owlDeviceBufferCreate(OptixData.context, OWL_USER_TYPE(float), cdfHeight, rows.data());
            OptixData.environmentMapColsBuffer = owlDeviceBufferCreate(OptixData.context, OWL_USER_TYPE(float), cdfWidth * cdfHeight, cols.data());
            OptixData.LP.environmentMapWidth = cdfWidth;
            OptixData.LP.environmentMapHeight = cdfHeight;  

            if (OptixData.environmentMapCDF) { owlGroupRelease(OptixData.environmentMapCDF); OptixData.environmentMapCDF = nullptr; }
            if (OptixData.environmentMapCDFIAS) { owlGroupRelease(OptixData.environmentMapCDFIAS); OptixData.environmentMapCDFIAS = nullptr; }
            if (OptixData.environmentMapCDFIndices) {owlBufferDestroy(OptixData.environmentMapCDFIndices); OptixData.environmentMapCDFIndices = nullptr; }
            if (OptixData.environmentMapCDFVertices) {owlBufferDestroy(OptixData.environmentMapCDFVertices); OptixData.environmentMapCDFVertices = nullptr; }
            if (OptixData.environmentMapCDFGeom) {owlGeomRelease(OptixData.environmentMapCDFGeom); OptixData.environmentMapCDFGeom = nullptr; }
            
            std::vector<glm::vec3> cdfVertices;
            std::vector<glm::ivec3> cdfIndices;
            std::vector<OWLGeom> cdfGeoms;
            uint32_t offset = 0;
            for (int y = 0, i = 0; y < cdfHeight; y++) {
                for (int x = 0; x < cdfWidth; x++, i++) {
                    float height = cols[i];
                    // XZ plane tris
                    glm::vec3 v00 = glm::vec3(x - .5f, y, 0.f);
                    glm::vec3 v01 = glm::vec3(x + .5f, y, 0.f);
                    glm::vec3 v02 = glm::vec3(x - .5f, y, height);
                    glm::vec3 v03 = glm::vec3(x + .5f, y, height);
                    cdfVertices.push_back(v00);
                    cdfVertices.push_back(v01);
                    cdfVertices.push_back(v02);
                    cdfVertices.push_back(v03);
                    cdfIndices.push_back(glm::ivec3(0, 1, 2) + glm::ivec3(offset));
                    cdfIndices.push_back(glm::ivec3(0, 2, 3) + glm::ivec3(offset));
                    offset += 4;
                }
            }

            for (int y = 0; y < cdfHeight; y++) {
                float height = rows[y];
                // YZ plane tris
                glm::vec3 v0 = glm::vec3(cdfWidth - 1, y - .5f, 0.f);
                glm::vec3 v1 = glm::vec3(cdfWidth - 1, y + .5f, 0.f);
                glm::vec3 v2 = glm::vec3(cdfWidth - 1, y - .5f, height);
                glm::vec3 v3 = glm::vec3(cdfWidth - 1, y + .5f, height);
                cdfVertices.push_back(v0);
                cdfVertices.push_back(v1);
                cdfVertices.push_back(v2);
                cdfVertices.push_back(v3);
                cdfIndices.push_back(glm::ivec3(0, 1, 2) + glm::ivec3(offset));
                cdfIndices.push_back(glm::ivec3(0, 2, 3) + glm::ivec3(offset));
                offset += 4;
            }
            
            std::cout<< "Creating " << cdfVertices.size() << " vertices" << std::endl;
            std::cout<< "Creating " << cdfIndices.size() << " indices" << std::endl;
            OptixData.environmentMapCDFVertices = owlDeviceBufferCreate(OptixData.context, OWL_USER_TYPE(glm::vec3), cdfVertices.size(), 0);
            OptixData.environmentMapCDFIndices = owlDeviceBufferCreate(OptixData.context, OWL_USER_TYPE(glm::ivec3), cdfIndices.size(), 0);
            OptixData.environmentMapCDFGeom = owlGeomCreate(OptixData.context, OptixData.cdfTrianglesGeomType);
            owlTrianglesSetVertices(OptixData.environmentMapCDFGeom, OptixData.environmentMapCDFVertices, cdfVertices.size(), sizeof(vec3), 0);
            owlTrianglesSetIndices(OptixData.environmentMapCDFGeom, OptixData.environmentMapCDFIndices, cdfIndices.size(), sizeof(ivec3), 0);
            OptixData.environmentMapCDF = owlTrianglesGeomGroupCreate(OptixData.context, 1, &OptixData.environmentMapCDFGeom);
            owlGroupBuildAccel(OptixData.environmentMapCDF);
            OptixData.environmentMapCDFIAS = owlInstanceGroupCreate(OptixData.context, 1);
            owlInstanceGroupSetChild(OptixData.environmentMapCDFIAS, 0, OptixData.environmentMapCDF);
            owlGroupBuildAccel(OptixData.environmentMapCDFIAS);
            owlParamsSetGroup(OptixData.launchParams, "environmentMapCDFIAS", OptixData.environmentMapCDFIAS);

            owlBuildSBT(OptixData.context);
        }
        else {
            OptixData.LP.environmentMapWidth = 0;
            OptixData.LP.environmentMapHeight = 0;  
        }
        resetAccumulation();        
    });
}

void setDomeLightRotation(glm::quat rotation)
{
    OptixData.LP.environmentMapRotation = rotation;
    resetAccumulation();
}

void enableDomeLightSampling()
{
    OptixData.LP.enableDomeSampling = true;
    resetAccumulation();
}

void disableDomeLightSampling()
{
    OptixData.LP.enableDomeSampling = false;
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

void setMaxBounceDepth(uint32_t diffuseDepth, uint32_t specularDepth, uint32_t volumeDepth)
{
    OptixData.LP.maxDiffuseBounceDepth = diffuseDepth;
    OptixData.LP.maxSpecularBounceDepth = specularDepth;
    OptixData.LP.maxVolumeBounceDepth = volumeDepth;
    launchParamsSetRaw(OptixData.launchParams, "maxDiffuseBounceDepth", &OptixData.LP.maxDiffuseBounceDepth);
    launchParamsSetRaw(OptixData.launchParams, "maxSpecularBounceDepth", &OptixData.LP.maxSpecularBounceDepth);
    launchParamsSetRaw(OptixData.launchParams, "maxVolumeBounceDepth", &OptixData.LP.maxVolumeBounceDepth);
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
    
    if (OptixData.LP.cameraEntity.initialized) {
        auto transform = Transform::getFront()[OptixData.LP.cameraEntity.transform_id];
        auto camera = Camera::getFront()[OptixData.LP.cameraEntity.camera_id];
        OptixData.LP.proj = camera.getProjection();
        OptixData.LP.viewT0 = transform.getWorldToLocalMatrix(/*previous = */ true);
        OptixData.LP.viewT1 = transform.getWorldToLocalMatrix(/*previous = */ false);
    }

    // If any of the components are dirty, reset accumulation
    bool anyUpdated = false;
    anyUpdated |= Mesh::areAnyDirty();
    anyUpdated |= Material::areAnyDirty();
    anyUpdated |= Camera::areAnyDirty();
    anyUpdated |= Transform::areAnyDirty();
    anyUpdated |= Light::areAnyDirty();
    anyUpdated |= Texture::areAnyDirty();
    anyUpdated |= Entity::areAnyDirty();
    anyUpdated |= Volume::areAnyDirty();

    if (!anyUpdated) return;
    resetAccumulation();
    
    std::recursive_mutex dummyMutex;
    std::lock_guard<std::recursive_mutex> mesh_lock(Mesh::areAnyDirty()           ? *Mesh::getEditMutex().get() : dummyMutex);
    std::lock_guard<std::recursive_mutex> camera_lock(Camera::areAnyDirty()       ? *Camera::getEditMutex().get() : dummyMutex);
    std::lock_guard<std::recursive_mutex> transform_lock(Transform::areAnyDirty() ? *Transform::getEditMutex().get() : dummyMutex);
    std::lock_guard<std::recursive_mutex> entity_lock(Entity::areAnyDirty()       ? *Entity::getEditMutex().get() : dummyMutex);
    std::lock_guard<std::recursive_mutex> light_lock(Light::areAnyDirty()         ? *Light::getEditMutex().get() : dummyMutex);
    std::lock_guard<std::recursive_mutex> texture_lock(Texture::areAnyDirty()     ? *Texture::getEditMutex().get() : dummyMutex);
    std::lock_guard<std::recursive_mutex> volume_lock(Volume::areAnyDirty()       ? *Volume::getEditMutex().get() : dummyMutex);

    // Manage Meshes: Build / Rebuild BLAS
    auto dirtyMeshes = Mesh::getDirtyMeshes();
    if (dirtyMeshes.size() > 0) {
        for (auto &m : dirtyMeshes) {
            // First, release any resources from a previous, stale mesh.
            if (OD.vertexLists[m->getAddress()]) { 
                owlBufferRelease(OD.vertexLists[m->getAddress()]); 
                OD.vertexLists[m->getAddress()] = nullptr; 
            }
            if (OD.normalLists[m->getAddress()]) { owlBufferRelease(OD.normalLists[m->getAddress()]); OD.normalLists[m->getAddress()] = nullptr; }
            if (OD.tangentLists[m->getAddress()]) { owlBufferRelease(OD.tangentLists[m->getAddress()]); OD.tangentLists[m->getAddress()] = nullptr; }
            if (OD.texCoordLists[m->getAddress()]) { owlBufferRelease(OD.texCoordLists[m->getAddress()]); OD.texCoordLists[m->getAddress()] = nullptr; }
            if (OD.indexLists[m->getAddress()]) { owlBufferRelease(OD.indexLists[m->getAddress()]); OD.indexLists[m->getAddress()] = nullptr; }
            if (OD.surfaceGeomList[m->getAddress()]) { owlGeomRelease(OD.surfaceGeomList[m->getAddress()]); OD.surfaceGeomList[m->getAddress()] = nullptr; }
            if (OD.surfaceBlasList[m->getAddress()]) { owlGroupRelease(OD.surfaceBlasList[m->getAddress()]); OD.surfaceBlasList[m->getAddress()] = nullptr; }
            
            // At this point, if the mesh no longer exists, move to the next dirty mesh.
            if (!m->isInitialized()) continue;
            if (m->getTriangleIndices().size() == 0) throw std::runtime_error("ERROR: indices is 0");

            // Next, allocate resources for the new mesh.
            OD.vertexLists[m->getAddress()]  = deviceBufferCreate(OD.context, OWL_USER_TYPE(vec3), m->getVertices().size(), m->getVertices().data());
            OD.normalLists[m->getAddress()]   = deviceBufferCreate(OD.context, OWL_USER_TYPE(vec4), m->getNormals().size(), m->getNormals().data());
            OD.tangentLists[m->getAddress()]   = deviceBufferCreate(OD.context, OWL_USER_TYPE(vec4), m->getTangents().size(), m->getTangents().data());
            OD.texCoordLists[m->getAddress()] = deviceBufferCreate(OD.context, OWL_USER_TYPE(vec2), m->getTexCoords().size(), m->getTexCoords().data());
            OD.indexLists[m->getAddress()]   = deviceBufferCreate(OD.context, OWL_USER_TYPE(uint32_t), m->getTriangleIndices().size(), m->getTriangleIndices().data());
            
            // Create geometry and build BLAS
            OD.surfaceGeomList[m->getAddress()] = geomCreate(OD.context, OD.trianglesGeomType);
            trianglesSetVertices(OD.surfaceGeomList[m->getAddress()], OD.vertexLists[m->getAddress()], m->getVertices().size(), sizeof(std::array<float, 3>), 0);
            trianglesSetIndices(OD.surfaceGeomList[m->getAddress()], OD.indexLists[m->getAddress()], m->getTriangleIndices().size() / 3, sizeof(ivec3), 0);
            OD.surfaceBlasList[m->getAddress()] = trianglesGeomGroupCreate(OD.context, 1, &OD.surfaceGeomList[m->getAddress()]);
            groupBuildAccel(OD.surfaceBlasList[m->getAddress()]);
        }

        bufferUpload(OD.vertexListsBuffer, OD.vertexLists.data());
        bufferUpload(OD.texCoordListsBuffer, OD.texCoordLists.data());
        bufferUpload(OD.indexListsBuffer, OD.indexLists.data());
        bufferUpload(OD.normalListsBuffer, OD.normalLists.data());
        bufferUpload(OD.tangentListsBuffer, OD.tangentLists.data());
        Mesh::updateComponents();
        bufferUpload(OptixData.meshBuffer, Mesh::getFrontStruct());
    }    

    // Manage Volumes: Build / Rebuild BLAS
    auto dirtyVolumes = Volume::getDirtyVolumes();
    if (dirtyVolumes.size() > 0) {
        for (auto &v : dirtyVolumes) {
            // First, release any resources from a previous, stale volume
            if (OD.volumeHandles[v->getAddress()]) owlBufferDestroy(OD.volumeHandles[v->getAddress()]);
            if (OD.volumeGeomList[v->getAddress()]) { owlGeomRelease(OD.volumeGeomList[v->getAddress()]); OD.volumeGeomList[v->getAddress()] = nullptr; }
            if (OD.volumeBlasList[v->getAddress()]) { owlGroupRelease(OD.volumeBlasList[v->getAddress()]); OD.volumeBlasList[v->getAddress()] = nullptr; }

            // At this point, if the volume no longer exists, move to the next dirty volume.
            if (!v->isInitialized()) continue;
            
            // Next, allocate resources for the new volume.
            auto gridHdlPtr = v->getNanoVDBGridHandle();
            const nanovdb::FloatGrid* grid = reinterpret_cast<nanovdb::FloatGrid*>(gridHdlPtr.get()->data());
            std::cout<<grid->checksum()<<std::endl;
            nanovdb::isValid(*grid, true, true);

            // auto acc = grid->tree().getAccessor();
            // auto bbox = tree.root().bbox();
            auto bbox = grid->tree().bbox().asReal<float>();
            // int nodecount = grid->tree().nodeCount(3);
            // std::cout<<nodecount<<std::endl;
            std::cout<<bbox.min()[0]<<bbox.min()[1]<<bbox.min()[2]<<bbox.max()[0]<<bbox.max()[1]<<bbox.max()[2]<<std::endl;

            OD.volumeHandles[v->getAddress()] = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(uint8_t), gridHdlPtr.get()->size(), nullptr);
            owlBufferUpload(OD.volumeHandles[v->getAddress()], gridHdlPtr.get()->data());
            printf("%hhx\n",gridHdlPtr.get()->data()[0]);
            const void* d_gridData = owlBufferGetPointer(OD.volumeHandles[v->getAddress()], 0);
            uint8_t first_byte;
            cudaMemcpy((void*)&first_byte, d_gridData, 1, cudaMemcpyDeviceToHost);
            printf("%hhx\n",first_byte);


            // Create geometry and build BLAS
            uint32_t volumeID = v->getAddress();
            OD.volumeGeomList[v->getAddress()] = geomCreate(OD.context, OD.volumeGeomType);
            owlGeomSetPrimCount(OD.volumeGeomList[v->getAddress()], 1); // for now, only one prim per volume. This might change...
            glm::vec4 tmpbbmin = glm::vec4(v->getMinAabbCorner(3, 0), 1.f);
            glm::vec4 tmpbbmax = glm::vec4(v->getMaxAabbCorner(3, 0), 1.f);
            owlGeomSetRaw(OD.volumeGeomList[v->getAddress()], "bbmin", &tmpbbmin);
            owlGeomSetRaw(OD.volumeGeomList[v->getAddress()], "bbmax", &tmpbbmax);
            owlGeomSetRaw(OD.volumeGeomList[v->getAddress()], "volumeID", &volumeID);
            OD.volumeBlasList[v->getAddress()] = owlUserGeomGroupCreate(OD.context, 1, &OD.volumeGeomList[v->getAddress()]);
            groupBuildAccel(OD.volumeBlasList[v->getAddress()]);    
        }
        Volume::updateComponents();
        owlBufferUpload(OptixData.volumeBuffer, Volume::getFrontStruct());
        owlBufferUpload(OD.volumeHandlesBuffer, OD.volumeHandles.data());
    }

    // Manage Entities: Build / Rebuild TLAS
    auto dirtyEntities = Entity::getDirtyEntities();
    if (dirtyEntities.size() > 0) {
        // Surface instances
        std::vector<OWLGroup> surfaceInstances;
        std::vector<glm::mat4> t0SurfaceTransforms;
        std::vector<glm::mat4> t1SurfaceTransforms;
        std::vector<uint32_t> surfaceInstanceToEntity;
        
        // Volume instances
        std::vector<OWLGroup> volumeInstances;
        std::vector<glm::mat4> t0VolumeTransforms;
        std::vector<glm::mat4> t1VolumeTransforms;
        std::vector<uint32_t> volumeInstanceToEntity;

        // Todo: curves...

        // Aggregate instanced geometry and transformations 
        Entity* entities = Entity::getFront();
        for (uint32_t eid = 0; eid < Entity::getCount(); ++eid) {
            // if (!entities[eid].isDirty()) continue; // if any entities are dirty, need to rebuild entire TLAS 

            // For an entity to go into a TLAS, it needs:
            // 1. a transform, to place it into the TLAS.
            // 2. geometry, either a mesh or a volume.
            // 3. a material or a light, to control surface appearance
            if (!entities[eid].isInitialized()) continue;
            if (!entities[eid].getTransform()) continue;
            if (!(entities[eid].getMesh() || entities[eid].getVolume())) continue;
            if (!entities[eid].getMaterial() && !entities[eid].getLight()) continue;

            // Get instance transformation
            glm::mat4 prevLocalToWorld = entities[eid].getTransform()->getLocalToWorldMatrix(/*previous = */true);
            glm::mat4 localToWorld = entities[eid].getTransform()->getLocalToWorldMatrix(/*previous = */false);

            // Add any instanced mesh geometry to the list
            if (entities[eid].getMesh()) {
                uint32_t address = entities[eid].getMesh()->getAddress();
                OWLGroup blas = OD.surfaceBlasList[address];
                if (!blas) {
                    // Not sure why, but the mesh this entity references hasn't been constructed yet.
                    // Mark it as dirty. It should be available in a subsequent frame
                    entities[eid].getMesh()->markDirty(); return; 
                }
                surfaceInstances.push_back(blas);
                surfaceInstanceToEntity.push_back(eid);
                t0SurfaceTransforms.push_back(prevLocalToWorld);
                t1SurfaceTransforms.push_back(localToWorld);
            }
            
            // Add any instanced volume geometry to the list
            if (entities[eid].getVolume()) {
                uint32_t address = entities[eid].getVolume()->getAddress();
                OWLGroup blas = OD.volumeBlasList[address];
                if (!blas) {
                    // Same as meshes, if BLAS doesn't exist, force BLAS build and try again.
                    entities[eid].getMesh()->markDirty(); return; 
                }
                volumeInstances.push_back(blas);
                volumeInstanceToEntity.push_back(eid);
                t0VolumeTransforms.push_back(prevLocalToWorld);
                t1VolumeTransforms.push_back(localToWorld);
            }     
        }

        std::vector<owl4x3f>     t0OwlSurfaceTransforms;
        std::vector<owl4x3f>     t1OwlSurfaceTransforms;
        std::vector<owl4x3f>     t0OwlVolumeTransforms;
        std::vector<owl4x3f>     t1OwlVolumeTransforms;
        auto oldSurfaceIAS = OD.surfacesIAS;
        auto oldVolumeIAS = OD.volumesIAS;
        
        // If no surfaces instanced, insert an unhittable placeholder.
        // (required for certain older driver versions)
        if (surfaceInstances.size() == 0) {
            OD.surfacesIAS = instanceGroupCreate(OD.context, 1);
            instanceGroupSetChild(OD.surfacesIAS, 0, OD.placeholderGroup); 
            groupBuildAccel(OD.surfacesIAS);
        }

        // If no volumes instanced, insert an unhittable placeholder.
        // (required for certain older driver versions)
        if (volumeInstances.size() == 0) {
            OD.volumesIAS = instanceGroupCreate(OD.context, 1);
            instanceGroupSetChild(OD.volumesIAS, 0, OD.placeholderUserGroup); 
            groupBuildAccel(OD.volumesIAS);
        }

        // Set surface transforms to IAS, upload surface instance to entity map
        if (surfaceInstances.size() > 0) {
            OD.surfacesIAS = instanceGroupCreate(OD.context, surfaceInstances.size());
            for (uint32_t iid = 0; iid < surfaceInstances.size(); ++iid) {
                instanceGroupSetChild(OD.surfacesIAS, iid, surfaceInstances[iid]);                 
                t0OwlSurfaceTransforms.push_back(glmToOWL(t0SurfaceTransforms[iid]));
                t1OwlSurfaceTransforms.push_back(glmToOWL(t1SurfaceTransforms[iid]));
            }            
            owlInstanceGroupSetTransforms(OD.surfacesIAS,0,(const float*)t0OwlSurfaceTransforms.data());
            owlInstanceGroupSetTransforms(OD.surfacesIAS,1,(const float*)t1OwlSurfaceTransforms.data());
            bufferResize(OD.surfaceInstanceToEntityBuffer, surfaceInstanceToEntity.size());
            bufferUpload(OD.surfaceInstanceToEntityBuffer, surfaceInstanceToEntity.data());
        }       

        // Set volume transforms to IAS, upload volume instance to entity map
        if (volumeInstances.size() > 0) {
            OD.volumesIAS = instanceGroupCreate(OD.context, volumeInstances.size());
            for (uint32_t iid = 0; iid < volumeInstances.size(); ++iid) {
                instanceGroupSetChild(OD.volumesIAS, iid, volumeInstances[iid]);                 
                t0OwlVolumeTransforms.push_back(glmToOWL(t0VolumeTransforms[iid]));
                t1OwlVolumeTransforms.push_back(glmToOWL(t1VolumeTransforms[iid]));
            }            
            owlInstanceGroupSetTransforms(OD.volumesIAS,0,(const float*)t0OwlVolumeTransforms.data());
            owlInstanceGroupSetTransforms(OD.volumesIAS,1,(const float*)t1OwlVolumeTransforms.data());
            bufferResize(OD.volumeInstanceToEntityBuffer, volumeInstanceToEntity.size());
            bufferUpload(OD.volumeInstanceToEntityBuffer, volumeInstanceToEntity.data());
        }

        // Build IAS
        groupBuildAccel(OD.volumesIAS);
        launchParamsSetGroup(OD.launchParams, "volumesIAS", OD.volumesIAS);
        groupBuildAccel(OD.surfacesIAS);
        launchParamsSetGroup(OD.launchParams, "surfacesIAS", OD.surfacesIAS);

        // Now that IAS have changed, we need to rebuild SBT
        buildSBT(OD.context);

        // Release any old IAS (TODO, don't rebuild if entity edit doesn't effect IAS...)
        if (oldSurfaceIAS) {owlGroupRelease(oldSurfaceIAS);}
        if (oldVolumeIAS) {owlGroupRelease(oldVolumeIAS);}
    
        // Aggregate entities that are light sources (todo: consider emissive volumes...)
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

        // Finally, upload entity structs to the GPU.
        Entity::updateComponents();
        bufferUpload(OptixData.entityBuffer,    Entity::getFrontStruct());
    }

    // Manage textures and materials
    if (Texture::areAnyDirty() || Material::areAnyDirty()) {
        std::lock_guard<std::recursive_mutex> material_lock(Material::areAnyDirty()   ? *Material::getEditMutex().get() : dummyMutex);

        // Allocate cuda textures for all texture components
        auto dirtyTextures = Texture::getDirtyTextures();
        for (auto &texture : dirtyTextures) {
            int tid = texture->getAddress();
            if (OD.textureObjects[tid]) { 
                owlTexture2DDestroy(OD.textureObjects[tid]); 
                OD.textureObjects[tid] = 0; 
            }
            if (!texture->isInitialized()) continue;
            bool isHDR = texture->isHDR();
            bool isLinear = texture->isLinear();
            uint32_t width = texture->getWidth();
            uint32_t height = texture->getHeight();
            OWLTexelFormat format = ((isHDR) ? OWL_TEXEL_FORMAT_RGBA32F : OWL_TEXEL_FORMAT_RGBA8);
            OWLTextureColorSpace colorSpace = ((isLinear) ? OWL_COLOR_SPACE_LINEAR: OWL_COLOR_SPACE_SRGB);
            if (width < 1 || height < 1 || 
                (isHDR && texture->getFloatTexels().size() != width * height) || 
                (!isHDR && texture->getByteTexels().size() != width * height)) 
            {
                std::cout<<"Internal error: corrupt texture. Attempting to recover..." <<std::endl;
                return; 
            }
            if (isHDR) {
                auto texels = texture->getFloatTexels();
                OD.textureObjects[tid] = owlTexture2DCreate(
                    OD.context, 
                    format,
                    width, height, texels.data(),
                    OWL_TEXTURE_LINEAR, 
                    OWL_TEXTURE_WRAP,
                    colorSpace
                );
            }
            else {
                auto texels = texture->getByteTexels();
                OD.textureObjects[tid] = owlTexture2DCreate(
                    OD.context, 
                    format,
                    width, height, texels.data(),
                    OWL_TEXTURE_LINEAR, 
                    OWL_TEXTURE_WRAP,
                    colorSpace
                );
            }
            
        }

        // Create additional cuda textures for material constants

        // Manage materials
        {
            Material* materials = Material::getFront();
            MaterialStruct* matStructs = Material::getFrontStruct();
            
            for (uint32_t mid = 0; mid < Material::getCount(); ++mid) {
                if (!materials[mid].isInitialized()) continue;
                if (!materials[mid].isDirty()) continue;

                OptixData.materialStructs[mid] = matStructs[mid];

                auto genRGBATex = [&OD](int index, vec4 c, vec4 defaultVal) {
                    if (OD.textureObjects[index]) { 
                        owlTexture2DDestroy(OD.textureObjects[index]); 
                        OD.textureObjects[index] = 0; 
                    }
                    if (glm::all(glm::equal(c, defaultVal))) return;
                    OD.textureObjects[index] = owlTexture2DCreate(
                        OD.context, OWL_TEXEL_FORMAT_RGBA32F,
                        1,1, &c, OWL_TEXTURE_LINEAR, OWL_TEXTURE_WRAP, OWL_COLOR_SPACE_LINEAR);
                    OptixData.textureStructs[index] = TextureStruct();
                    OptixData.textureStructs[index].width = 1;
                    OptixData.textureStructs[index].height = 1;
                };

                int off = Texture::getCount() + mid * NUM_MAT_PARAMS;
                auto &m = materials[mid];
                auto &ms = matStructs[mid];
                auto &odms = OptixData.materialStructs[mid];
                if (ms.transmission_roughness_texture_id == -1) { genRGBATex(off + 0, vec4(m.getTransmissionRoughness()), vec4(0.f)); }
                if (ms.base_color_texture_id == -1)             { genRGBATex(off + 1, vec4(m.getBaseColor(), 1.f), vec4(.8f, .8f, .8f, 1.f)); }
                if (ms.roughness_texture_id == -1)              { genRGBATex(off + 2, vec4(m.getRoughness()), vec4(.5f)); }
                if (ms.alpha_texture_id == -1)                  { genRGBATex(off + 3, vec4(m.getAlpha()), vec4(1.f)); }
                if (ms.normal_map_texture_id == -1)             { genRGBATex(off + 4, vec4(0.5f, .5f, 1.f, 0.f), vec4(0.5f, .5f, 1.f, 0.f)); }
                if (ms.subsurface_color_texture_id == -1)       { genRGBATex(off + 5, vec4(m.getSubsurfaceColor(), 1.f), glm::vec4(0.8f, 0.8f, 0.8f, 1.f)); }
                if (ms.subsurface_radius_texture_id == -1)      { genRGBATex(off + 6, vec4(m.getSubsurfaceRadius(), 1.f), glm::vec4(1.0f, .2f, .1f, 1.f)); }
                if (ms.subsurface_texture_id == -1)             { genRGBATex(off + 7, vec4(m.getSubsurface()), glm::vec4(0.f)); }
                if (ms.metallic_texture_id == -1)               { genRGBATex(off + 8, vec4(m.getMetallic()), glm::vec4(0.f)); }
                if (ms.specular_texture_id == -1)               { genRGBATex(off + 9, vec4(m.getSpecular()), glm::vec4(.5f)); }
                if (ms.specular_tint_texture_id == -1)          { genRGBATex(off + 10, vec4(m.getSpecularTint()), glm::vec4(0.f)); }
                if (ms.anisotropic_texture_id == -1)            { genRGBATex(off + 11, vec4(m.getAnisotropic()), glm::vec4(0.f)); }
                if (ms.anisotropic_rotation_texture_id == -1)   { genRGBATex(off + 12, vec4(m.getAnisotropicRotation()), glm::vec4(0.f)); }
                if (ms.sheen_texture_id == -1)                  { genRGBATex(off + 13, vec4(m.getSheen()), glm::vec4(0.f)); }
                if (ms.sheen_tint_texture_id == -1)             { genRGBATex(off + 14, vec4(m.getSheenTint()), glm::vec4(0.5f)); }
                if (ms.clearcoat_texture_id == -1)              { genRGBATex(off + 15, vec4(m.getClearcoat()), glm::vec4(0.f)); }
                if (ms.clearcoat_roughness_texture_id == -1)    { genRGBATex(off + 16, vec4(m.getClearcoatRoughness()), glm::vec4(0.3f)); }
                if (ms.ior_texture_id == -1)                    { genRGBATex(off + 17, vec4(m.getIor()), glm::vec4(1.45f)); }
                if (ms.transmission_texture_id == -1)           { genRGBATex(off + 18, vec4(m.getTransmission()), glm::vec4(0.f)); }
                
                if (ms.transmission_roughness_texture_id == -1) { odms.transmission_roughness_texture_id = off + 0; }
                if (ms.base_color_texture_id == -1)             { odms.base_color_texture_id = off + 1; }
                if (ms.roughness_texture_id == -1)              { odms.roughness_texture_id = off + 2; }
                if (ms.alpha_texture_id == -1)                  { odms.alpha_texture_id = off + 3; }
                if (ms.normal_map_texture_id == -1)             { odms.normal_map_texture_id = off + 4; }
                if (ms.subsurface_color_texture_id == -1)       { odms.subsurface_color_texture_id = off + 5; }
                if (ms.subsurface_radius_texture_id == -1)      { odms.subsurface_radius_texture_id = off + 6; }
                if (ms.subsurface_texture_id == -1)             { odms.subsurface_texture_id = off + 7; }
                if (ms.metallic_texture_id == -1)               { odms.metallic_texture_id = off + 8; }
                if (ms.specular_texture_id == -1)               { odms.specular_texture_id = off + 9; }
                if (ms.specular_tint_texture_id == -1)          { odms.specular_tint_texture_id = off + 10; }
                if (ms.anisotropic_texture_id == -1)            { odms.anisotropic_texture_id = off + 11; }
                if (ms.anisotropic_rotation_texture_id == -1)   { odms.anisotropic_rotation_texture_id = off + 12; }
                if (ms.sheen_texture_id == -1)                  { odms.sheen_texture_id = off + 13; }
                if (ms.sheen_tint_texture_id == -1)             { odms.sheen_tint_texture_id = off + 14; }
                if (ms.clearcoat_texture_id == -1)              { odms.clearcoat_texture_id = off + 15; }
                if (ms.clearcoat_roughness_texture_id == -1)    { odms.clearcoat_roughness_texture_id = off + 16; }
                if (ms.ior_texture_id == -1)                    { odms.ior_texture_id = off + 17; }
                if (ms.transmission_texture_id == -1)           { odms.transmission_texture_id = off + 18; }
            }

            Material::updateComponents();
            bufferUpload(OptixData.materialBuffer, OptixData.materialStructs.data());
        }
        
        bufferUpload(OD.textureObjectsBuffer, OD.textureObjects.data());
        Texture::updateComponents();
        memcpy(OptixData.textureStructs.data(), Texture::getFrontStruct(), Texture::getCount() * sizeof(TextureStruct));
        bufferUpload(OptixData.textureBuffer, OptixData.textureStructs.data());
    }
    
    // Manage transforms
    auto dirtyTransforms = Transform::getDirtyTransforms();
    if (dirtyTransforms.size() > 0) {
        Transform::updateComponents();
        
        // // for each device
        // for (uint32_t id = 0; id < owlGetDeviceCount(OptixData.context); ++id)
        // {
        //     cudaSetDevice(id);

        //     TransformStruct* devTransforms = (TransformStruct*)owlBufferGetPointer(OptixData.transformBuffer, id);
        //     TransformStruct* transformStructs = Transform::getFrontStruct();
        //     for (auto &t : dirtyTransforms) {
        //         if (!t->isInitialized()) continue;
        //         CUDA_CHECK(cudaMemcpy(&devTransforms[t->getAddress()], &transformStructs[t->getAddress()], sizeof(TransformStruct), cudaMemcpyHostToDevice));
        //     }
        // }

        // cudaSetDevice(0);
        owlBufferUpload(OptixData.transformBuffer, Transform::getFrontStruct());
    }   

    // Manage Cameras
    if (Camera::areAnyDirty()) {
        Camera::updateComponents();
        bufferUpload(OptixData.cameraBuffer,    Camera::getFrontStruct());
    }    

    // Manage lights
    if (Light::areAnyDirty()) {
        Light::updateComponents();
        bufferUpload(OptixData.lightBuffer,     Light::getFrontStruct());
    }
}

void updateLaunchParams()
{
    launchParamsSetRaw(OptixData.launchParams, "frameID", &OptixData.LP.frameID);
    launchParamsSetRaw(OptixData.launchParams, "frameSize", &OptixData.LP.frameSize);
    launchParamsSetRaw(OptixData.launchParams, "cameraEntity", &OptixData.LP.cameraEntity);
    launchParamsSetRaw(OptixData.launchParams, "domeLightIntensity", &OptixData.LP.domeLightIntensity);
    launchParamsSetRaw(OptixData.launchParams, "domeLightExposure", &OptixData.LP.domeLightExposure);
    launchParamsSetRaw(OptixData.launchParams, "domeLightColor", &OptixData.LP.domeLightColor);
    launchParamsSetRaw(OptixData.launchParams, "renderDataMode", &OptixData.LP.renderDataMode);
    launchParamsSetRaw(OptixData.launchParams, "renderDataBounce", &OptixData.LP.renderDataBounce);
    launchParamsSetRaw(OptixData.launchParams, "enableDomeSampling", &OptixData.LP.enableDomeSampling);
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
    if (OD.enableAlbedoGuide) inputLayers.push_back(albedoLayer);

    OptixImage2D normalLayer;
    normalLayer.width = OD.LP.frameSize.x;
    normalLayer.height = OD.LP.frameSize.y;
    normalLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    normalLayer.pixelStrideInBytes = 4 * sizeof(float);
    normalLayer.rowStrideInBytes   = OD.LP.frameSize.x * 4 * sizeof(float);
    normalLayer.data   = (CUdeviceptr) bufferGetPointer(OD.normalBuffer, 0);
    if (OD.enableNormalGuide) inputLayers.push_back(normalLayer);

    OptixImage2D outputLayer = colorLayer; // can I get away with this?

    uint64_t scratchSizeInBytes;
    #ifdef USE_OPTIX70
    scratchSizeInBytes = OD.denoiserSizes.recommendedScratchSizeInBytes;
    #else
    scratchSizeInBytes = OD.denoiserSizes.withOverlapScratchSizeInBytes;
    #endif

    OptixDenoiserParams params;

    if (!OD.enableKernelPrediction) {
        OPTIX_CHECK(optixDenoiserComputeIntensity(
            OD.denoiser, 
            cudaStream, 
            &inputLayers[0], 
            (CUdeviceptr) bufferGetPointer(OD.hdrIntensityBuffer, 0),
            (CUdeviceptr) bufferGetPointer(OD.denoiserScratchBuffer, 0),
            scratchSizeInBytes));
    }

    if (!OD.enableKernelPrediction) {
        OPTIX_CHECK(optixDenoiserComputeAverageColor(
            OD.denoiser, 
            cudaStream, 
            &inputLayers[0], 
            (CUdeviceptr) bufferGetPointer(OD.colorAvgBuffer, 0),
            (CUdeviceptr) bufferGetPointer(OD.denoiserScratchBuffer, 0),
            scratchSizeInBytes));
    }

    params.denoiseAlpha = 0;    // Don't touch alpha.
    params.blendFactor  = 0.0f; // Show the denoised image only.
    params.hdrIntensity = (CUdeviceptr) bufferGetPointer(OD.hdrIntensityBuffer, 0);
    #ifdef USE_OPTIX72
    params.hdrAverageColor = (CUdeviceptr) bufferGetPointer(OD.colorAvgBuffer, 0);
    #endif
    
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
    synchronizeDevices();
    glFlush();

    auto &OD = OptixData;
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

    enqueueCommand([width, height] () {
        using namespace Libraries;
        auto glfw = GLFW::Get();
        glfw->resize_window("ViSII", width, height);
        glViewport(0,0,width,height);
    });
}

void enableDenoiser() 
{
    enqueueCommand([] () { OptixData.enableDenoiser = true; });
}

void disableDenoiser()
{
    enqueueCommand([] () { OptixData.enableDenoiser = false; });
}

void configureDenoiser(bool useAlbedoGuide, bool useNormalGuide, bool useKernelPrediction)
{
    if (useNormalGuide && (!useAlbedoGuide)) {
        throw std::runtime_error("Error, unsupported denoiser configuration."
            "If normal guide is enabled, albedo guide must also be enabled.");
    }

    enqueueCommand([useAlbedoGuide, useNormalGuide, useKernelPrediction](){
        OptixData.enableAlbedoGuide = useAlbedoGuide;
        OptixData.enableNormalGuide = useNormalGuide;
        OptixData.enableKernelPrediction = useKernelPrediction;

        // Reconfigure denoiser

        // Setup denoiser
        OptixDenoiserOptions options;
        if ((!useAlbedoGuide) && (!useNormalGuide)) options.inputKind = OPTIX_DENOISER_INPUT_RGB;
        else if (!useNormalGuide) options.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;
        else options.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;

        #ifndef USE_OPTIX72
        options.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
        #endif

        if (OptixData.denoiser) optixDenoiserDestroy(OptixData.denoiser);
        
        auto optixContext = getOptixContext(OptixData.context, 0);
        auto cudaStream = getStream(OptixData.context, 0);
        OPTIX_CHECK(optixDenoiserCreate(optixContext, &options, &OptixData.denoiser));

        OptixDenoiserModelKind kind;
        if (OptixData.enableKernelPrediction) kind = OPTIX_DENOISER_MODEL_KIND_AOV;
        else kind = OPTIX_DENOISER_MODEL_KIND_HDR;
        
        OPTIX_CHECK(
            optixDenoiserSetModel(OptixData.denoiser, kind, 
            /*data*/ nullptr, /*sizeInBytes*/ 0));

        optixDenoiserComputeMemoryResources(OptixData.denoiser, 
            OptixData.LP.frameSize.x, OptixData.LP.frameSize.y, 
            &OptixData.denoiserSizes);

        uint64_t scratchSizeInBytes;
        #ifdef USE_OPTIX70
        scratchSizeInBytes = OptixData.denoiserSizes.recommendedScratchSizeInBytes;
        #else
        scratchSizeInBytes = OptixData.denoiserSizes.withOverlapScratchSizeInBytes;
        #endif
        bufferResize(OptixData.denoiserScratchBuffer, scratchSizeInBytes);
        bufferResize(OptixData.denoiserStateBuffer, OptixData.denoiserSizes.stateSizeInBytes);
        
        optixDenoiserSetup (
            OptixData.denoiser, 
            (cudaStream_t) cudaStream, 
            (unsigned int) OptixData.LP.frameSize.x, 
            (unsigned int) OptixData.LP.frameSize.y, 
            (CUdeviceptr) bufferGetPointer(OptixData.denoiserStateBuffer, 0), 
            OptixData.denoiserSizes.stateSizeInBytes,
            (CUdeviceptr) bufferGetPointer(OptixData.denoiserScratchBuffer, 0), 
            scratchSizeInBytes
        );
    });
}

std::vector<float> readFrameBuffer() {
    std::vector<float> frameBuffer(OptixData.LP.frameSize.x * OptixData.LP.frameSize.y * 4);

    enqueueCommandAndWait([&frameBuffer] () {
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
    });
    return frameBuffer;
}

std::vector<float> render(uint32_t width, uint32_t height, uint32_t samplesPerPixel, uint32_t seed) {
    if ((width < 1) || (height < 1)) throw std::runtime_error("Error, invalid width/height");
    std::vector<float> frameBuffer(width * height * 4);

    enqueueCommandAndWait([&frameBuffer, width, height, samplesPerPixel, seed] () {
        if (!ViSII.headlessMode) {
            if ((width != WindowData.currentSize.x) || (height != WindowData.currentSize.y))
            {
                using namespace Libraries;
                auto glfw = GLFW::Get();
                glfw->resize_window("ViSII", width, height);
                initializeFrameBuffer(width, height);
            }
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
            owlLaunch2D(OptixData.rayGen, OptixData.LP.frameSize.x * OptixData.LP.frameSize.y, 1, OptixData.launchParams);
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
        cudaMemcpyAsync(frameBuffer.data(), fb, width * height * sizeof(glm::vec4), cudaMemcpyDeviceToHost);

        synchronizeDevices();
    });

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

    enqueueCommandAndWait([&frameBuffer, width, height, startFrame, frameCount, bounce, _option, seed] () {
        if (!ViSII.headlessMode) {
            if ((width != WindowData.currentSize.x) || (height != WindowData.currentSize.y))
            {
                using namespace Libraries;
                auto glfw = GLFW::Get();
                glfw->resize_window("ViSII", width, height);
                initializeFrameBuffer(width, height);
            }
        }

        // remove trailing whitespace from option, convert to lowercase
        std::string option = trim(_option);
        std::transform(option.data(), option.data() + option.size(), std::addressof(option[0]), [](unsigned char c){ return std::tolower(c); });

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
        else if (option == std::string("texture_coordinates")) {
            OptixData.LP.renderDataMode = RenderDataFlags::TEXTURE_COORDINATES;
        }
        else if (option == std::string("screen_space_normal")) {
            OptixData.LP.renderDataMode = RenderDataFlags::SCREEN_SPACE_NORMAL;
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
            owlLaunch2D(OptixData.rayGen, OptixData.LP.frameSize.x * OptixData.LP.frameSize.y, 1, OptixData.launchParams);
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
        cudaMemcpyAsync(frameBuffer.data(), fb, width * height * sizeof(glm::vec4), cudaMemcpyDeviceToHost);

        OptixData.LP.renderDataMode = 0;
        OptixData.LP.renderDataBounce = 0;
        updateLaunchParams();
    });

    return frameBuffer;
}

std::string getFileExtension(const std::string &filename) {
  if (filename.find_last_of(".") != std::string::npos)
    return filename.substr(filename.find_last_of(".") + 1);
  return "";
}

void renderDataToFile(uint32_t width, uint32_t height, uint32_t startFrame, uint32_t frameCount, uint32_t bounce, std::string field, std::string imagePath, uint32_t seed)
{
    std::vector<float> fb = renderData(width, height, startFrame, frameCount, bounce, field);
    std::string extension = getFileExtension(imagePath);
    if ((extension.compare("exr") == 0) || (extension.compare("EXR") == 0)) {
        std::vector<float> colors(4 * width * height);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {     
                vec4 color = vec4(
                    fb[(((height - y) - 1) * width + x) * 4 + 0], 
                    fb[(((height - y) - 1) * width + x) * 4 + 1], 
                    fb[(((height - y) - 1) * width + x) * 4 + 2], 
                    fb[(((height - y) - 1) * width + x) * 4 + 3]);
                colors[(y * width + x) * 4 + 0] = color.r;
                colors[(y * width + x) * 4 + 1] = color.g;
                colors[(y * width + x) * 4 + 2] = color.b;
                colors[(y * width + x) * 4 + 3] = color.a;
            }
        }

        const char* err = nullptr;
        int ret = SaveEXR(colors.data(), width, height, /*components*/4, /*gp16*/0, imagePath.c_str(), &err);
        if (TINYEXR_SUCCESS != ret) {
            throw std::runtime_error(std::string("Error saving EXR : \"") + imagePath + std::string("\". ")
                + std::string(err));
        }
    }
    else if ((extension.compare("hdr") == 0) || (extension.compare("HDR") == 0)) {
        stbi_flip_vertically_on_write(true);
        stbi_write_hdr(imagePath.c_str(), width, height, /* num channels*/ 4, fb.data());
    }
    else if ((extension.compare("png") == 0) || (extension.compare("PNG") == 0)) {
        std::vector<uint8_t> colors(4 * width * height);
        for (size_t i = 0; i < (width * height); ++i) {     
            vec3 color = vec3(fb[i * 4 + 0], fb[i * 4 + 1], fb[i * 4 + 2]);
            float alpha = fb[i * 4 + 3];
            color = glm::convertLinearToSRGB(color);
            colors[i * 4 + 0] = uint8_t(glm::clamp(color.r * 255.f, 0.f, 255.f));
            colors[i * 4 + 1] = uint8_t(glm::clamp(color.g * 255.f, 0.f, 255.f));
            colors[i * 4 + 2] = uint8_t(glm::clamp(color.b * 255.f, 0.f, 255.f));
            colors[i * 4 + 3] = uint8_t(glm::clamp(alpha * 255.f, 0.f, 255.f));
        }
        stbi_flip_vertically_on_write(true);
        stbi_write_png(imagePath.c_str(), width, height, /* num channels*/ 4, colors.data(), /* stride in bytes */ width * 4);
    }
}

static bool renderToHDRDeprecatedShown = false;
void renderToHDR(uint32_t width, uint32_t height, uint32_t samplesPerPixel, std::string imagePath, uint32_t seed)
{
    if (renderToHDRDeprecatedShown == false) {
        std::cout<<"Warning, render_to_hdr is deprecated and will be removed in a subsequent release. Please switch to render_to_file." << std::endl;
        renderToHDRDeprecatedShown = true;
    }

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

static bool renderToPNGDeprecatedShown = false;
void renderToPNG(uint32_t width, uint32_t height, uint32_t samplesPerPixel, std::string imagePath, uint32_t seed)
{
    if (renderToPNGDeprecatedShown == false) {
        std::cout<<"Warning, render_to_png is deprecated and will be removed in a subsequent release. Please switch to render_to_file." << std::endl;
        renderToPNGDeprecatedShown = true;
    }

    // float exposure = 2.f; // TODO: expose as a parameter

    std::vector<float> fb = render(width, height, samplesPerPixel, seed);
    std::vector<uint8_t> colors(4 * width * height);
    for (size_t i = 0; i < (width * height); ++i) {     
        vec3 color = vec3(fb[i * 4 + 0], fb[i * 4 + 1], fb[i * 4 + 2]);
        float alpha = fb[i * 4 + 3];

        // color = Uncharted2Tonemap(color * exposure);
        // color = color * (1.0f / Uncharted2Tonemap(vec3(11.2f)));

        color = glm::convertLinearToSRGB(color);

        colors[i * 4 + 0] = uint8_t(glm::clamp(color.r * 255.f, 0.f, 255.f));
        colors[i * 4 + 1] = uint8_t(glm::clamp(color.g * 255.f, 0.f, 255.f));
        colors[i * 4 + 2] = uint8_t(glm::clamp(color.b * 255.f, 0.f, 255.f));
        colors[i * 4 + 3] = uint8_t(glm::clamp(alpha * 255.f, 0.f, 255.f));
    }
    stbi_flip_vertically_on_write(true);
    stbi_write_png(imagePath.c_str(), width, height, /* num channels*/ 4, colors.data(), /* stride in bytes */ width * 4);
}

void renderToFile(uint32_t width, uint32_t height, uint32_t samplesPerPixel, std::string imagePath, uint32_t seed)
{
    std::vector<float> fb = render(width, height, samplesPerPixel, seed);
    std::string extension = getFileExtension(imagePath);
    if ((extension.compare("exr") == 0) || (extension.compare("EXR") == 0)) {
        std::vector<float> colors(4 * width * height);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {     
                vec4 color = vec4(
                    fb[(((height - y) - 1) * width + x) * 4 + 0], 
                    fb[(((height - y) - 1) * width + x) * 4 + 1], 
                    fb[(((height - y) - 1) * width + x) * 4 + 2], 
                    fb[(((height - y) - 1) * width + x) * 4 + 3]);
                colors[(y * width + x) * 4 + 0] = color.r;
                colors[(y * width + x) * 4 + 1] = color.g;
                colors[(y * width + x) * 4 + 2] = color.b;
                colors[(y * width + x) * 4 + 3] = color.a;
            }
        }

        const char* err = nullptr;
        int ret = SaveEXR(colors.data(), width, height, /*components*/4, /*gp16*/0, imagePath.c_str(), &err);
        if (TINYEXR_SUCCESS != ret) {
            throw std::runtime_error(std::string("Error saving EXR : \"") + imagePath + std::string("\". ")
                + std::string(err));
        }
    }
    else if ((extension.compare("hdr") == 0) || (extension.compare("HDR") == 0)) {
        stbi_flip_vertically_on_write(true);
        stbi_write_hdr(imagePath.c_str(), width, height, /* num channels*/ 4, fb.data());
    }
    else if ((extension.compare("png") == 0) || (extension.compare("PNG") == 0)) {
        std::vector<uint8_t> colors(4 * width * height);
        for (size_t i = 0; i < (width * height); ++i) {     
            vec3 color = vec3(fb[i * 4 + 0], fb[i * 4 + 1], fb[i * 4 + 2]);
            float alpha = fb[i * 4 + 3];
            color = glm::convertLinearToSRGB(color);
            colors[i * 4 + 0] = uint8_t(glm::clamp(color.r * 255.f, 0.f, 255.f));
            colors[i * 4 + 1] = uint8_t(glm::clamp(color.g * 255.f, 0.f, 255.f));
            colors[i * 4 + 2] = uint8_t(glm::clamp(color.b * 255.f, 0.f, 255.f));
            colors[i * 4 + 3] = uint8_t(glm::clamp(alpha * 255.f, 0.f, 255.f));
        }
        stbi_flip_vertically_on_write(true);
        stbi_write_png(imagePath.c_str(), width, height, /* num channels*/ 4, colors.data(), /* stride in bytes */ width * 4);
    }
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

void initializeComponentFactories(
    uint32_t maxEntities, 
    uint32_t maxCameras, 
    uint32_t maxTransforms, 
    uint32_t maxMeshes, 
    uint32_t maxMaterials, 
    uint32_t maxLights,
    uint32_t maxTextures,
    uint32_t maxVolumes
) {
    Entity::initializeFactory(maxEntities);
    Camera::initializeFactory(maxCameras);
    Transform::initializeFactory(maxTransforms);
    Mesh::initializeFactory(maxMeshes);
    Material::initializeFactory(maxMaterials);
    Light::initializeFactory(maxLights);
    Texture::initializeFactory(maxTextures);
    Volume::initializeFactory(maxVolumes);
}

void reproject(glm::vec4 *samplesBuffer, glm::vec4 *t0AlbedoBuffer, glm::vec4 *t1AlbedoBuffer, glm::vec4 *mvecBuffer, glm::vec4 *scratchBuffer, glm::vec4 *imageBuffer, int width, int height);


static bool initializeInteractiveDeprecatedShown = false;
static bool initializeHeadlessDeprecatedShown = false;
void initializeInteractive(
    bool windowOnTop, 
    bool _verbose,
    uint32_t maxEntities,
    uint32_t maxCameras,
    uint32_t maxTransforms,
    uint32_t maxMeshes,
    uint32_t maxMaterials,
    uint32_t maxLights,
    uint32_t maxTextures,
    uint32_t maxVolumes)
{
    if (initializeInteractiveDeprecatedShown == false) {
        std::cout<<"Warning, initialize_interactive is deprecated and will be removed in a subsequent release. Please switch to initialize." << std::endl;
        initializeInteractiveDeprecatedShown = true;
    }

    // don't initialize more than once
    if (initialized == true) {
        throw std::runtime_error("Error: already initialized!");
    }

    initialized = true;
    stopped = false;
    verbose = _verbose;
    ViSII.callback = nullptr;

    initializeComponentFactories(maxEntities, maxCameras, maxTransforms, maxMeshes, maxMaterials, maxLights, maxTextures, maxVolumes);

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

            if (ViSII.callback && ViSII.callbackMutex.try_lock()) {
                ViSII.callback();
                ViSII.callbackMutex.unlock();
            }

            static double start=0;
            static double stop=0;
            start = glfwGetTime();

            if (!lazyUpdatesEnabled) {
                updateFrameBuffer();
                updateComponents();
                updateLaunchParams();
                owlLaunch2D(OptixData.rayGen, OptixData.LP.frameSize.x * OptixData.LP.frameSize.y, 1, OptixData.launchParams);
                if (OptixData.enableDenoiser)
                {
                    denoiseImage();
                }        
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
            checkForErrors();
            if (stopped) break;
        }

        if (OptixData.denoiser)
            OPTIX_CHECK(optixDenoiserDestroy(OptixData.denoiser));

        if (OptixData.imageTexID != -1) {
            cudaGraphicsUnregisterResource(OptixData.cudaResourceTex);
            glDeleteTextures(1, &OptixData.imageTexID);
        }

        ImGui::DestroyContext();
        if (glfw->does_window_exist("ViSII")) glfw->destroy_window("ViSII");

        owlContextDestroy(OptixData.context);
    };

    renderThread = std::thread(loop);

    // Waits for the render thread to start before returning
    enqueueCommandAndWait([] () {});
}

void initializeHeadless(
    bool _verbose, 
    uint32_t maxEntities,
    uint32_t maxCameras,
    uint32_t maxTransforms,
    uint32_t maxMeshes,
    uint32_t maxMaterials,
    uint32_t maxLights,
    uint32_t maxTextures,
    uint32_t maxVolumes)
{
    if (initializeHeadlessDeprecatedShown == false) {
        std::cout<<"Warning, initialize_headless is deprecated and will be removed in a subsequent release. Please switch to initialize(headless = True)." << std::endl;
        initializeHeadlessDeprecatedShown = true;
    }

    // don't initialize more than once
    if (initialized == true) {
        throw std::runtime_error("Error: already initialized!");
    }

    initialized = true;
    stopped = false;
    verbose = _verbose;
    ViSII.callback = nullptr;

    initializeComponentFactories(maxEntities, maxCameras, maxTransforms, maxMeshes, maxMaterials, maxLights, maxTextures, maxVolumes);

    auto loop = []() {
        ViSII.render_thread_id = std::this_thread::get_id();
        ViSII.headlessMode = true;

        initializeOptix(/*headless = */ true);

        while (!stopped)
        {
            if(ViSII.callback){
                ViSII.callback();
            }
            processCommandQueue();
            if (stopped) break;
        }

        if (OptixData.denoiser)
            OPTIX_CHECK(optixDenoiserDestroy(OptixData.denoiser));
        
        owlContextDestroy(OptixData.context);
    };

    renderThread = std::thread(loop);

    // Waits for the render thread to start before returning
    enqueueCommandAndWait([] () {});
}

void initialize(
    bool headless, 
    bool windowOnTop, 
    bool _lazyUpdatesEnabled, 
    bool verbose,
    uint32_t maxEntities,
    uint32_t maxCameras,
    uint32_t maxTransforms,
    uint32_t maxMeshes,
    uint32_t maxMaterials,
    uint32_t maxLights,
    uint32_t maxTextures,
    uint32_t maxVolumes) 
{
    lazyUpdatesEnabled = _lazyUpdatesEnabled;
    // prevents deprecated warning from showing
    initializeInteractiveDeprecatedShown = true;
    initializeHeadlessDeprecatedShown = true;

    lazyUpdatesEnabled = _lazyUpdatesEnabled;
    if (headless) 
        initializeHeadless(
            verbose, maxEntities, maxCameras, maxTransforms, maxMeshes, 
            maxMaterials, maxLights, maxTextures, maxVolumes);
    else 
        initializeInteractive(
            windowOnTop, verbose, maxEntities, maxCameras, maxTransforms, 
            maxMeshes, maxMaterials, maxLights, maxTextures, maxVolumes);
}

static bool registerPreRenderCallbackDeprecatedShown = false;
void registerPreRenderCallback(std::function<void()> callback){
    if (registerPreRenderCallbackDeprecatedShown == false) {
        std::cout<<"Warning, register_pre_render_callback is deprecated and will be removed in a subsequent release. Please switch to register_callback." << std::endl;
        registerPreRenderCallbackDeprecatedShown = true;
    }
    registerCallback(callback);
}

void registerCallback(std::function<void()> callback){
    ViSII.callback = callback;
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
    Volume::clearAll();
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

void enableUpdates()
{
    enqueueCommand([] () { lazyUpdatesEnabled = false; });
}

void disableUpdates()
{
    enqueueCommand([] () { lazyUpdatesEnabled = true; });
}

bool areUpdatesEnabled()
{
    return lazyUpdatesEnabled == false;
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
        clearAll();
    }
    initialized = false;
    checkForErrors();
}

bool isButtonPressed(std::string button) {
    if (ViSII.headlessMode) return false;
    auto glfw = Libraries::GLFW::Get();
    std::transform(button.data(), button.data() + button.size(), 
        std::addressof(button[0]), [](unsigned char c){ return std::toupper(c); });
    bool pressed, prevPressed;
    if (button.compare("MOUSE_LEFT") == 0) {
        pressed = glfw->get_button_action("ViSII", 0) == 1;
        prevPressed = glfw->get_button_action_prev("ViSII", 0) == 1;
    }
    else if (button.compare("MOUSE_RIGHT") == 0) {
        pressed = glfw->get_button_action("ViSII", 1) == 1;
        prevPressed = glfw->get_button_action_prev("ViSII", 1) == 1;
    }
    else if (button.compare("MOUSE_MIDDLE") == 0) {
        pressed = glfw->get_button_action("ViSII", 2) == 1;
        prevPressed = glfw->get_button_action_prev("ViSII", 2) == 1;
    }
    else {
        pressed = glfw->get_key_action("ViSII", glfw->get_key_code(button)) == 1;
        prevPressed = glfw->get_key_action_prev("ViSII", glfw->get_key_code(button)) == 1;
    }
    
    return pressed && !prevPressed;
}

bool isButtonHeld(std::string button) {
    if (ViSII.headlessMode) return false;
    auto glfw = Libraries::GLFW::Get();
    std::transform(button.data(), button.data() + button.size(), 
        std::addressof(button[0]), [](unsigned char c){ return std::toupper(c); });
    if (button.compare("MOUSE_LEFT") == 0) return glfw->get_button_action("ViSII", 0) >= 1;
    if (button.compare("MOUSE_RIGHT") == 0) return glfw->get_button_action("ViSII", 1) >= 1;
    if (button.compare("MOUSE_MIDDLE") == 0) return glfw->get_button_action("ViSII", 2) >= 1;
    return glfw->get_key_action("ViSII", glfw->get_key_code(button)) >= 1;
}

vec2 getCursorPos()
{
    if (ViSII.headlessMode) return vec2(NAN, NAN);
    auto glfw = Libraries::GLFW::Get();
    auto pos = glfw->get_cursor_pos("ViSII");
    return vec2(pos[0], pos[1]);
}

void setCursorMode(std::string mode)
{
    if (ViSII.headlessMode) return;
    enqueueCommand([mode] () {
        std::string mode_ = mode;
        std::transform(mode_.data(), mode_.data() + mode_.size(), 
            std::addressof(mode_[0]), [](unsigned char c){ return std::toupper(c); });
        int value = GLFW_CURSOR_NORMAL;
        if (mode_.compare("NORMAL") == 0) value = GLFW_CURSOR_NORMAL;
        if (mode_.compare("HIDDEN") == 0) value = GLFW_CURSOR_HIDDEN;
        if (mode_.compare("DISABLED") == 0) value = GLFW_CURSOR_DISABLED;
        glfwSetInputMode(WindowData.window, GLFW_CURSOR, value);
    });
}

ivec2 getWindowSize()
{
    if (ViSII.headlessMode) return ivec2(NAN, NAN);
    return WindowData.currentSize;
}

bool shouldWindowClose()
{
    if (ViSII.headlessMode) return false;
    auto glfw = Libraries::GLFW::Get();
    return glfw->should_close("ViSII");
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
    else if (option == std::string("texture_coordinates")) {
        OptixData.LP.renderDataMode = RenderDataFlags::TEXTURE_COORDINATES;
    }
    else if (option == std::string("screen_space_normal")) {
        OptixData.LP.renderDataMode = RenderDataFlags::SCREEN_SPACE_NORMAL;
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

};
