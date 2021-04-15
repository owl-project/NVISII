#undef NDEBUG

#include <nvisii/nvisii.h>

#include <algorithm>

#include <glfw_implementation/glfw.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <ImGuizmo.h>
#include <nvisii/utilities/colors.h>
#include <owl/owl.h>
#include <owl/helper/optix.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include <glm/gtc/color_space.hpp>

#include <devicecode/launch_params.h>
#include <devicecode/path_tracer.h>

#define PBRLUT_IMPLEMENTATION
#include <nvisii/utilities/ggx_lookup_tables.h>
#include <nvisii/utilities/procedural_sky.h>

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

// #define __optix_optix_function_table_h__
#include <optix_stubs.h>
// OptixFunctionTable g_optixFunctionTable;

// #include <thrust/reduce.h>
// #include <thrust/execution_policy.h>
// #include <thrust/device_vector.h>
// #include <thrust/device_ptr.h>

namespace nvisii {

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
    bool resourceSharingSuccessful = true;
    OWLBuffer assignmentBuffer;

    OWLBuffer frameBuffer;
    OWLBuffer normalBuffer;
    OWLBuffer albedoBuffer;
    OWLBuffer scratchBuffer;
    OWLBuffer mvecBuffer;
    OWLBuffer accumBuffer;

    OWLBuffer combinedFrameBuffer;
    OWLBuffer combinedNormalBuffer;
    OWLBuffer combinedAlbedoBuffer;

    OWLBuffer entityBuffer;
    OWLBuffer transformBuffer;
    OWLBuffer cameraBuffer;
    OWLBuffer materialBuffer;
    OWLBuffer meshBuffer;
    OWLBuffer lightBuffer;
    OWLBuffer textureBuffer;
    OWLBuffer volumeBuffer;
    OWLBuffer lightEntitiesBuffer;
    OWLBuffer instanceToEntityBuffer;
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

    std::vector<OWLBuffer> vertexLists;
    std::vector<OWLBuffer> normalLists;
    std::vector<OWLBuffer> tangentLists;
    std::vector<OWLBuffer> texCoordLists;
    std::vector<OWLBuffer> indexLists;
    std::vector<OWLGeom> surfaceGeomList;
    std::vector<OWLGroup> surfaceBlasList;
    std::vector<OWLGeom> volumeGeomList;
    std::vector<OWLGroup> volumeBlasList;

    OWLGroup IAS = nullptr;

    std::vector<uint32_t> lightEntities;

    bool enableDenoiser = false;
    #if USE_OPTIX72
    bool enableKernelPrediction = true;
    #else
    bool enableKernelPrediction = false;
    #endif
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

} OptixData;

static struct NVISII {
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

    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> events;
    std::vector<float> times;
    std::vector<float> weights;
} NVISII;

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

void synchronizeDevices(std::string error_string = "")
{
    for (int i = 0; i < getDeviceCount(); i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        cudaError_t err = cudaPeekAtLastError();
        if (err != 0) {
            std::cout<< "ERROR " << error_string << ": " << cudaGetErrorString(err)<<std::endl;
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
    cudaSetDevice(0);

    fbWidth = glm::max(fbWidth, 1);
    fbHeight = glm::max(fbHeight, 1);
    synchronizeDevices();

    auto &OD = OptixData;
    if (OD.imageTexID != -1) {
        if (OptixData.cudaResourceTex && OptixData.resourceSharingSuccessful) {
            cudaGraphicsUnregisterResource(OptixData.cudaResourceTex);
            OptixData.cudaResourceTex = 0;
        }
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
    static bool renderToHDRDeprecatedShown = false;
    cudaError_t rc = cudaGraphicsGLRegisterImage(&OD.cudaResourceTex, OD.imageTexID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
    if (rc != cudaSuccess) {
        std::string err = cudaGetErrorString(cudaGetLastError());
        if (verbose && !renderToHDRDeprecatedShown) {
            std::cout
                  << "Warning: Could not do CUDA graphics resource sharing "
                  << "for the display buffer texture ("
                  << err
                  << ")... falling back to slower path"
                  << std::endl;
            renderToHDRDeprecatedShown = true;
        }
        OD.resourceSharingSuccessful = false;
        if (OD.cudaResourceTex) {
          cudaGraphicsUnregisterResource(OD.cudaResourceTex);
          OD.cudaResourceTex = 0;
        }
    } else {
        OD.resourceSharingSuccessful = true;
    }
    synchronizeDevices();
}

void resizeOptixFrameBuffer(uint32_t width, uint32_t height)
{
    auto &OD = OptixData;
    OD.LP.frameSize.x = width;
    OD.LP.frameSize.y = height;
    owlBufferResize(OD.frameBuffer, width * height);
    owlBufferResize(OD.normalBuffer, width * height);
    owlBufferResize(OD.albedoBuffer, width * height);
    owlBufferResize(OD.scratchBuffer, width * height);
    owlBufferResize(OD.mvecBuffer, width * height);    
    owlBufferResize(OD.accumBuffer, width * height);

    owlBufferResize(OD.combinedFrameBuffer, width * height);
    owlBufferResize(OD.combinedNormalBuffer, width * height);
    owlBufferResize(OD.combinedAlbedoBuffer, width * height);

    // Reconfigure denoiser
    optixDenoiserComputeMemoryResources(OD.denoiser, OD.LP.frameSize.x, OD.LP.frameSize.y, &OD.denoiserSizes);
    uint64_t scratchSizeInBytes;
    #ifdef USE_OPTIX70
    scratchSizeInBytes = OD.denoiserSizes.recommendedScratchSizeInBytes;
    #else
    scratchSizeInBytes = OD.denoiserSizes.withOverlapScratchSizeInBytes;
    #endif
    owlBufferResize(OD.denoiserScratchBuffer, scratchSizeInBytes);
    owlBufferResize(OD.denoiserStateBuffer, OD.denoiserSizes.stateSizeInBytes);
    
    auto cudaStream = owlContextGetStream(OD.context, 0);
    optixDenoiserSetup (
        OD.denoiser, 
        (cudaStream_t) cudaStream, 
        (unsigned int) OD.LP.frameSize.x, 
        (unsigned int) OD.LP.frameSize.y, 
        (CUdeviceptr) owlBufferGetPointer(OD.denoiserStateBuffer, 0), 
        OD.denoiserSizes.stateSizeInBytes,
        (CUdeviceptr) owlBufferGetPointer(OD.denoiserScratchBuffer, 0), 
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
    OD.module = owlModuleCreate(OD.context, ptxCode);
    
    /* Setup Optix Launch Params */
    OWLVarDecl launchParamVars[] = {
        { "assignmentBuffer",        OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, assignmentBuffer)},
        { "frameSize",               OWL_USER_TYPE(glm::ivec2),         OWL_OFFSETOF(LaunchParams, frameSize)},
        { "frameID",                 OWL_USER_TYPE(uint64_t),           OWL_OFFSETOF(LaunchParams, frameID)},
        { "frameBuffer",             OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, frameBuffer)},
        { "normalBuffer",            OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, normalBuffer)},
        { "albedoBuffer",            OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, albedoBuffer)},
        { "scratchBuffer",           OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, scratchBuffer)},
        { "mvecBuffer",              OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, mvecBuffer)},
        { "accumPtr",                OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, accumPtr)},
        { "IAS",                     OWL_GROUP,                         OWL_OFFSETOF(LaunchParams, IAS)},
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
        { "instanceToEntity",        OWL_BUFFER,                        OWL_OFFSETOF(LaunchParams, instanceToEntity)},
        { "domeLightIntensity",      OWL_USER_TYPE(float),              OWL_OFFSETOF(LaunchParams, domeLightIntensity)},
        { "domeLightExposure",       OWL_USER_TYPE(float),              OWL_OFFSETOF(LaunchParams, domeLightExposure)},
        { "domeLightColor",          OWL_USER_TYPE(glm::vec3),          OWL_OFFSETOF(LaunchParams, domeLightColor)},
        { "directClamp",             OWL_USER_TYPE(float),              OWL_OFFSETOF(LaunchParams, directClamp)},
        { "indirectClamp",           OWL_USER_TYPE(float),              OWL_OFFSETOF(LaunchParams, indirectClamp)},
        { "maxDiffuseDepth",         OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, maxDiffuseDepth)},
        { "maxGlossyDepth",          OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, maxGlossyDepth)},
        { "maxTransparencyDepth",    OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, maxTransparencyDepth)},
        { "maxTransmissionDepth",    OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, maxTransmissionDepth)},
        { "maxVolumeDepth",          OWL_USER_TYPE(uint32_t),           OWL_OFFSETOF(LaunchParams, maxVolumeDepth)},
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
    OD.launchParams = owlParamsCreate(OD.context, sizeof(LaunchParams), launchParamVars, -1);
    
    /* Create AOV Buffers */
    if (!headless) {
        initializeFrameBuffer(512, 512);        
    }

    OD.assignmentBuffer = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(float), owlGetDeviceCount(OD.context) + 1, nullptr);
    owlParamsSetBuffer(OD.launchParams, "assignmentBuffer", OD.assignmentBuffer);

    // If we only have one GPU, framebuffer pixels can stay on device 0. 
    if (numGPUsFound == 1) {
        OD.frameBuffer = owlDeviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
        OD.accumBuffer = owlDeviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
        OD.normalBuffer = owlDeviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
        OD.albedoBuffer = owlDeviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
        OD.scratchBuffer = owlDeviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
        OD.mvecBuffer = owlDeviceBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
    }
    // Otherwise, multiple GPUs must use host pinned memory to merge partial framebuffers together
    else {
        OD.frameBuffer = owlHostPinnedBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512);
        OD.accumBuffer = owlHostPinnedBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512);
        OD.normalBuffer = owlHostPinnedBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512);
        OD.albedoBuffer = owlHostPinnedBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512);
        OD.scratchBuffer = owlHostPinnedBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512);
        OD.mvecBuffer = owlHostPinnedBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512);
    }

    // For multiGPU denoising, its best to denoise using something other than zero-copy memory.
    OD.combinedFrameBuffer = owlManagedMemoryBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
    OD.combinedNormalBuffer = owlManagedMemoryBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);
    OD.combinedAlbedoBuffer = owlManagedMemoryBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),512*512, nullptr);

    OD.LP.frameSize = glm::ivec2(512, 512);
    owlParamsSetBuffer(OD.launchParams, "frameBuffer", OD.frameBuffer);
    owlParamsSetBuffer(OD.launchParams, "normalBuffer", OD.normalBuffer);
    owlParamsSetBuffer(OD.launchParams, "albedoBuffer", OD.albedoBuffer);
    owlParamsSetBuffer(OD.launchParams, "scratchBuffer", OD.scratchBuffer);
    owlParamsSetBuffer(OD.launchParams, "mvecBuffer", OD.mvecBuffer);
    owlParamsSetBuffer(OD.launchParams, "accumPtr", OD.accumBuffer);
    owlParamsSetRaw(OD.launchParams, "frameSize", &OD.LP.frameSize);

    /* Create Component Buffers */
    // note, extra textures reserved for internal use
    OD.entityBuffer              = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(EntityStruct),        Entity::getCount(),   nullptr);
    OD.transformBuffer           = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(TransformStruct),     Transform::getCount(), nullptr);
    OD.cameraBuffer              = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(CameraStruct),        Camera::getCount(),    nullptr);
    OD.materialBuffer            = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(MaterialStruct),      Material::getCount(),  nullptr);
    OD.meshBuffer                = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(MeshStruct),          Mesh::getCount(),     nullptr);
    OD.lightBuffer               = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(LightStruct),         Light::getCount(),     nullptr);
    OD.textureBuffer             = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(TextureStruct),       Texture::getCount() + NUM_MAT_PARAMS * Material::getCount(),   nullptr);
    OD.volumeBuffer              = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(VolumeStruct),        Volume::getCount(),   nullptr);
    OD.volumeHandlesBuffer       = owlDeviceBufferCreate(OD.context, OWL_BUFFER,                         Volume::getCount(),   nullptr);
    OD.lightEntitiesBuffer       = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(uint32_t),            1,              nullptr);
    OD.instanceToEntityBuffer    = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(uint32_t),            1,              nullptr);
    OD.vertexListsBuffer         = owlDeviceBufferCreate(OD.context, OWL_BUFFER,                         Mesh::getCount(),     nullptr);
    OD.normalListsBuffer         = owlDeviceBufferCreate(OD.context, OWL_BUFFER,                         Mesh::getCount(),     nullptr);
    OD.tangentListsBuffer        = owlDeviceBufferCreate(OD.context, OWL_BUFFER,                         Mesh::getCount(),     nullptr);
    OD.texCoordListsBuffer       = owlDeviceBufferCreate(OD.context, OWL_BUFFER,                         Mesh::getCount(),     nullptr);
    OD.indexListsBuffer          = owlDeviceBufferCreate(OD.context, OWL_BUFFER,                         Mesh::getCount(),     nullptr);
    OD.textureObjectsBuffer      = owlDeviceBufferCreate(OD.context, OWL_TEXTURE,                        Texture::getCount() + NUM_MAT_PARAMS * Material::getCount(),   nullptr);

    owlParamsSetBuffer(OD.launchParams, "entities",             OD.entityBuffer);
    owlParamsSetBuffer(OD.launchParams, "transforms",           OD.transformBuffer);
    owlParamsSetBuffer(OD.launchParams, "cameras",              OD.cameraBuffer);
    owlParamsSetBuffer(OD.launchParams, "materials",            OD.materialBuffer);
    owlParamsSetBuffer(OD.launchParams, "meshes",               OD.meshBuffer);
    owlParamsSetBuffer(OD.launchParams, "lights",               OD.lightBuffer);
    owlParamsSetBuffer(OD.launchParams, "textures",             OD.textureBuffer);
    owlParamsSetBuffer(OD.launchParams, "volumes",              OD.volumeBuffer);
    owlParamsSetBuffer(OD.launchParams, "lightEntities",        OD.lightEntitiesBuffer);
    owlParamsSetBuffer(OD.launchParams, "instanceToEntity",     OD.instanceToEntityBuffer);
    owlParamsSetBuffer(OD.launchParams, "vertexLists",          OD.vertexListsBuffer);
    owlParamsSetBuffer(OD.launchParams, "normalLists",          OD.normalListsBuffer);
    owlParamsSetBuffer(OD.launchParams, "tangentLists",         OD.tangentListsBuffer);
    owlParamsSetBuffer(OD.launchParams, "texCoordLists",        OD.texCoordListsBuffer);
    owlParamsSetBuffer(OD.launchParams, "indexLists",           OD.indexListsBuffer);
    owlParamsSetBuffer(OD.launchParams, "textureObjects",       OD.textureObjectsBuffer);
    owlParamsSetBuffer(OD.launchParams, "volumeHandles",        OD.volumeHandlesBuffer);

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
    owlParamsSetRaw(OD.launchParams, "environmentMapID", &OD.LP.environmentMapID);
    owlParamsSetRaw(OD.launchParams, "environmentMapRotation", &OD.LP.environmentMapRotation);

    owlParamsSetBuffer(OD.launchParams, "environmentMapRows", OD.environmentMapRowsBuffer);
    owlParamsSetBuffer(OD.launchParams, "environmentMapCols", OD.environmentMapColsBuffer);
    owlParamsSetRaw(OD.launchParams, "environmentMapWidth", &OD.LP.environmentMapWidth);
    owlParamsSetRaw(OD.launchParams, "environmentMapHeight", &OD.LP.environmentMapHeight);
   
    OD.LP.numLightEntities = uint32_t(OD.lightEntities.size());
    owlParamsSetRaw(OD.launchParams, "numLightEntities", &OD.LP.numLightEntities);
    owlParamsSetRaw(OD.launchParams, "domeLightIntensity", &OD.LP.domeLightIntensity);
    owlParamsSetRaw(OD.launchParams, "domeLightExposure", &OD.LP.domeLightExposure);
    owlParamsSetRaw(OD.launchParams, "domeLightColor", &OD.LP.domeLightColor);
    owlParamsSetRaw(OD.launchParams, "directClamp", &OD.LP.directClamp);
    owlParamsSetRaw(OD.launchParams, "indirectClamp", &OD.LP.indirectClamp);
    owlParamsSetRaw(OD.launchParams, "maxDiffuseDepth", &OD.LP.maxDiffuseDepth);
    owlParamsSetRaw(OD.launchParams, "maxGlossyDepth", &OD.LP.maxGlossyDepth);
    owlParamsSetRaw(OD.launchParams, "maxTransparencyDepth", &OD.LP.maxTransparencyDepth);
    owlParamsSetRaw(OD.launchParams, "maxTransmissionDepth", &OD.LP.maxTransmissionDepth);
    owlParamsSetRaw(OD.launchParams, "maxVolumeDepth", &OD.LP.maxVolumeDepth);
    owlParamsSetRaw(OD.launchParams, "numLightSamples", &OD.LP.numLightSamples);
    owlParamsSetRaw(OD.launchParams, "seed", &OD.LP.seed);
    owlParamsSetRaw(OD.launchParams, "xPixelSamplingInterval", &OD.LP.xPixelSamplingInterval);
    owlParamsSetRaw(OD.launchParams, "yPixelSamplingInterval", &OD.LP.yPixelSamplingInterval);
    owlParamsSetRaw(OD.launchParams, "timeSamplingInterval", &OD.LP.timeSamplingInterval);

    OWLVarDecl trianglesGeomVars[] = {{/* sentinel to mark end of list */}};
    OD.trianglesGeomType = geomTypeCreate(OD.context, OWL_GEOM_TRIANGLES, sizeof(TrianglesGeomData), trianglesGeomVars,-1);
    OWLVarDecl volumeGeomVars[] = {
        { "bbmin", OWL_USER_TYPE(glm::vec4), OWL_OFFSETOF(VolumeGeomData, bbmin)},
        { "bbmax", OWL_USER_TYPE(glm::vec4), OWL_OFFSETOF(VolumeGeomData, bbmax)},
        { "volumeID", OWL_USER_TYPE(uint32_t), OWL_OFFSETOF(VolumeGeomData, volumeID)},
        {/* sentinel to mark end of list */}
    };
    OD.volumeGeomType = owlGeomTypeCreate(OD.context, OWL_GEOM_USER, sizeof(VolumeGeomData), volumeGeomVars, -1);
    geomTypeSetClosestHit(OD.trianglesGeomType, /*ray type */ 0, OD.module,"TriangleMesh");
    geomTypeSetClosestHit(OD.trianglesGeomType, /*ray type */ 1, OD.module,"ShadowRay");
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

    owlBuildPrograms(OD.context);
    
    /* Temporary GAS. Required for certain older driver versions. */
    const int NUM_VERTICES = 1;
    vec3 vertices[NUM_VERTICES] = {{ 0.f, 0.f, 0.f }};
    const int NUM_INDICES = 1;
    ivec3 indices[NUM_INDICES] = {{ 0, 0, 0 }};    
    OWLBuffer vertexBuffer = owlDeviceBufferCreate(OD.context,OWL_FLOAT4,NUM_VERTICES,vertices);
    OWLBuffer indexBuffer = owlDeviceBufferCreate(OD.context,OWL_INT3,NUM_INDICES,indices);
    OWLGeom trianglesGeom = geomCreate(OD.context,OD.trianglesGeomType);
    trianglesSetVertices(trianglesGeom,vertexBuffer,NUM_VERTICES,sizeof(vec4),0);
    trianglesSetIndices(trianglesGeom,indexBuffer, NUM_INDICES,sizeof(ivec3),0);
    OD.placeholderGroup = trianglesGeomGroupCreate(OD.context,1,&trianglesGeom);
    groupBuildAccel(OD.placeholderGroup);

    // build IAS
    OWLGroup IAS = instanceGroupCreate(OD.context, 1);
    instanceGroupSetChild(IAS, 0, OD.placeholderGroup); 
    groupBuildAccel(IAS);
    owlParamsSetGroup(OD.launchParams, "IAS", IAS);

    OWLGeom userGeom = owlGeomCreate(OD.context, OD.volumeGeomType);
    owlGeomSetPrimCount(userGeom, 1);
    glm::vec4 tmpbbmin(1.f), tmpbbmax(-1.f); // unhittable
    owlGeomSetRaw(userGeom, "bbmin", &tmpbbmin);
    owlGeomSetRaw(userGeom, "bbmax", &tmpbbmax);
    OD.placeholderUserGroup = owlUserGeomGroupCreate(OD.context, 1, &userGeom);
    groupBuildAccel(OD.placeholderUserGroup);

    // Build *SBT* required to trace the groups   
    owlBuildPipeline(OD.context);
    owlBuildSBT(OD.context);

    // Setup denoiser
    configureDenoiser(OD.enableAlbedoGuide, OD.enableNormalGuide, OD.enableKernelPrediction);

    OD.placeholder = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(void*), 1, nullptr);

    setDomeLightSky(glm::vec3(0,0,10));

    OptixData.LP.sceneBBMin = OptixData.LP.sceneBBMax = glm::vec3(0.f);

    // To measure how long each card takes to trace for load balancing
    int numGPUs = owlGetDeviceCount(OptixData.context);
    for (uint32_t deviceID = 0; deviceID < numGPUs; deviceID++) {
        cudaSetDevice(deviceID);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        NVISII.events.push_back({start, stop});
        NVISII.times.push_back(1.f);
        NVISII.weights.push_back(1.f / float(numGPUs));
    }
    cudaSetDevice(0);
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
    // if (NVISII.render_thread_id != std::this_thread::get_id()) 
    std::lock_guard<std::recursive_mutex> lock(NVISII.qMutex);

    NVISII::Command c;
    c.function = function;
    c.promise = std::make_shared<std::promise<void>>();
    auto new_future = c.promise->get_future();
    NVISII.commandQueue.push(c);
    // cv.notify_one();
    return new_future;
}

void enqueueCommandAndWait(std::function<void()> function)
{
    if (NVISII.render_thread_id != std::this_thread::get_id()) {
        if (NVISII.callback) {
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
    std::lock_guard<std::recursive_mutex> lock(NVISII.qMutex);
    while (!NVISII.commandQueue.empty()) {
        auto item = NVISII.commandQueue.front();
        item.function();
        try {
            item.promise->set_value();
        }
        catch (std::future_error& e) {
            if (e.code() == std::make_error_condition(std::future_errc::promise_already_satisfied))
                std::cout << "NVISII: [promise already satisfied]\n";
            else
                std::cout << "NVISII: [unknown exception]\n";
        }
        NVISII.commandQueue.pop();
    }
}

void updateGPUWeights()
{
    int num_gpus = owlGetDeviceCount(OptixData.context);
    float target = 1.f / float(num_gpus);
    
    std::vector<float> signals(num_gpus);
    float total_time = 0.f;
    for (uint32_t i = 0; i < num_gpus; ++i) total_time += NVISII.times[i];
    for (uint32_t i = 0; i < num_gpus; ++i) signals[i] = NVISII.times[i] / float(total_time);

    std::vector<float> p_error(num_gpus);
    for (uint32_t i = 0; i < num_gpus; ++i) p_error[i] = target - signals[i];

    // update weights 
    float pK = 1.f;
    for (uint32_t i = 0; i < num_gpus; ++i) {
        NVISII.weights[i] = max(NVISII.weights[i] + p_error[i], .001f);
    }

    std::vector<float> scan;
    for (size_t i = 0; i <= num_gpus; ++i) {
        if (i == 0) scan.push_back(0.f);
        else scan.push_back(scan[i - 1] + NVISII.weights[i - 1]);
    }

    // std::cout<<"Scan: ";
    for (size_t i = 0; i <= num_gpus; ++i) {
        scan[i] /= scan[num_gpus];
        // std::cout<<scan[i] << " ";
    }
    // std::cout<<std::endl;

    owlBufferUpload(OptixData.assignmentBuffer, scan.data());
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
    owlParamsSetRaw(OptixData.launchParams, "indirectClamp", &OptixData.LP.indirectClamp);
    resetAccumulation();
}

void setDirectLightingClamp(float clamp)
{
    clamp = std::max(float(clamp), float(0.f));
    OptixData.LP.directClamp = clamp;
    owlParamsSetRaw(OptixData.launchParams, "directClamp", &OptixData.LP.directClamp);
    resetAccumulation();
}

void setMaxBounceDepth(
    uint32_t diffuseDepth,
    uint32_t glossyDepth,
    uint32_t transparencyDepth,
    uint32_t transmissionDepth,
    uint32_t volumeDepth
) {
    OptixData.LP.maxDiffuseDepth = diffuseDepth;
    OptixData.LP.maxGlossyDepth = glossyDepth;
    OptixData.LP.maxTransparencyDepth = transparencyDepth;
    OptixData.LP.maxTransmissionDepth = transmissionDepth;
    OptixData.LP.maxVolumeDepth = volumeDepth;
    
    owlParamsSetRaw(OptixData.launchParams, "maxDiffuseDepth", &OptixData.LP.maxDiffuseDepth);
    owlParamsSetRaw(OptixData.launchParams, "maxGlossyDepth", &OptixData.LP.maxGlossyDepth);
    owlParamsSetRaw(OptixData.launchParams, "maxTransparencyDepth", &OptixData.LP.maxTransparencyDepth);
    owlParamsSetRaw(OptixData.launchParams, "maxTransmissionDepth", &OptixData.LP.maxTransmissionDepth);
    owlParamsSetRaw(OptixData.launchParams, "maxVolumeDepth", &OptixData.LP.maxVolumeDepth);
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
    owlParamsSetRaw(OptixData.launchParams, "numLightSamples", &OptixData.LP.numLightSamples);
    resetAccumulation();
}

void samplePixelArea(vec2 xSampleInterval, vec2 ySampleInterval)
{
    OptixData.LP.xPixelSamplingInterval = xSampleInterval;
    OptixData.LP.yPixelSamplingInterval = ySampleInterval;
    owlParamsSetRaw(OptixData.launchParams, "xPixelSamplingInterval", &OptixData.LP.xPixelSamplingInterval);
    owlParamsSetRaw(OptixData.launchParams, "yPixelSamplingInterval", &OptixData.LP.yPixelSamplingInterval);
    resetAccumulation();
}

void sampleTimeInterval(vec2 sampleTimeInterval)
{
    OptixData.LP.timeSamplingInterval = sampleTimeInterval;
    owlParamsSetRaw(OptixData.launchParams, "timeSamplingInterval", &OptixData.LP.timeSamplingInterval);
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
            OD.vertexLists[m->getAddress()]  = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(vec3), m->getVertices().size(), m->getVertices().data());
            OD.normalLists[m->getAddress()]   = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(vec4), m->getNormals().size(), m->getNormals().data());
            OD.tangentLists[m->getAddress()]   = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(vec4), m->getTangents().size(), m->getTangents().data());
            OD.texCoordLists[m->getAddress()] = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(vec2), m->getTexCoords().size(), m->getTexCoords().data());
            OD.indexLists[m->getAddress()]   = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(uint32_t), m->getTriangleIndices().size(), m->getTriangleIndices().data());
            
            // Create geometry and build BLAS
            OD.surfaceGeomList[m->getAddress()] = geomCreate(OD.context, OD.trianglesGeomType);
            trianglesSetVertices(OD.surfaceGeomList[m->getAddress()], OD.vertexLists[m->getAddress()], m->getVertices().size(), sizeof(std::array<float, 3>), 0);
            trianglesSetIndices(OD.surfaceGeomList[m->getAddress()], OD.indexLists[m->getAddress()], m->getTriangleIndices().size() / 3, sizeof(ivec3), 0);
            OD.surfaceBlasList[m->getAddress()] = trianglesGeomGroupCreate(OD.context, 1, &OD.surfaceGeomList[m->getAddress()]);
            groupBuildAccel(OD.surfaceBlasList[m->getAddress()]);          
        }

        owlBufferUpload(OD.vertexListsBuffer, OD.vertexLists.data());
        owlBufferUpload(OD.texCoordListsBuffer, OD.texCoordLists.data());
        owlBufferUpload(OD.indexListsBuffer, OD.indexLists.data());
        owlBufferUpload(OD.normalListsBuffer, OD.normalLists.data());
        owlBufferUpload(OD.tangentListsBuffer, OD.tangentLists.data());
        Mesh::updateComponents();
        owlBufferUpload(OptixData.meshBuffer, Mesh::getFrontStruct());
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
            nanovdb::isValid(*grid, true, true);

            // auto acc = grid->tree().getAccessor();
            // auto bbox = tree.root().bbox();
            auto bbox = grid->tree().bbox().asReal<float>();
            // int nodecount = grid->tree().nodeCount(3);

            OD.volumeHandles[v->getAddress()] = owlDeviceBufferCreate(OD.context, OWL_USER_TYPE(uint8_t), gridHdlPtr.get()->size(), nullptr);
            owlBufferUpload(OD.volumeHandles[v->getAddress()], gridHdlPtr.get()->data());
            // printf("%hhx\n",gridHdlPtr.get()->data()[0]);
            const void* d_gridData = owlBufferGetPointer(OD.volumeHandles[v->getAddress()], 0);
            uint8_t first_byte;
            cudaMemcpy((void*)&first_byte, d_gridData, 1, cudaMemcpyDeviceToHost);
            // printf("%hhx\n",first_byte);

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
        std::vector<OWLGroup> instances;
        std::vector<glm::mat4> t0Transforms;
        std::vector<glm::mat4> t1Transforms;
        std::vector<uint8_t> masks;
        std::vector<uint32_t> instanceToEntity;

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
            t0Transforms.push_back(prevLocalToWorld);
            t1Transforms.push_back(localToWorld);

            // Get instance mask
            masks.push_back(entities[eid].getStruct().flags);

            // Indirection from instance back to entity ID
            instanceToEntity.push_back(eid);

            // Add any instanced mesh geometry to the list
            if (entities[eid].getMesh()) {
                uint32_t address = entities[eid].getMesh()->getAddress();
                OWLGroup blas = OD.surfaceBlasList[address];
                if (!blas) {
                    // Not sure why, but the mesh this entity references hasn't been constructed yet.
                    // Mark it as dirty. It should be available in a subsequent frame
                    entities[eid].getMesh()->markDirty(); return; 
                }
                instances.push_back(blas);
            }
            
            // Add any instanced volume geometry to the list
            else if (entities[eid].getVolume()) {
                uint32_t address = entities[eid].getVolume()->getAddress();
                OWLGroup blas = OD.volumeBlasList[address];
                if (!blas) {
                    // Same as meshes, if BLAS doesn't exist, force BLAS build and try again.
                    entities[eid].getMesh()->markDirty(); return; 
                }
                instances.push_back(blas);
            } 
            
            else {
                throw std::runtime_error("Internal Error, renderable entity has no mesh or volume components!?");
            }   
        }

        std::vector<uint8_t>     owlVisibilityMasks;
        std::vector<owl4x3f>     t0OwlTransforms;
        std::vector<owl4x3f>     t1OwlTransforms;
        auto oldIAS = OD.IAS;
        
        // If no objects are instanced, insert an unhittable placeholder.
        // (required for certain older driver versions)
        if (instances.size() == 0) {
            OD.IAS = instanceGroupCreate(OD.context, 1);
            instanceGroupSetChild(OD.IAS, 0, OD.placeholderGroup); 
            groupBuildAccel(OD.IAS);
        }

        // Set instance transforms and masks, upload instance to entity map
        if (instances.size() > 0) {
            OD.IAS = instanceGroupCreate(OD.context, instances.size());
            for (uint32_t iid = 0; iid < instances.size(); ++iid) {
                instanceGroupSetChild(OD.IAS, iid, instances[iid]);                 
                t0OwlTransforms.push_back(glmToOWL(t0Transforms[iid]));
                t1OwlTransforms.push_back(glmToOWL(t1Transforms[iid]));
                owlVisibilityMasks.push_back(masks[iid]);
            }            
            owlInstanceGroupSetTransforms(OD.IAS,0,(const float*)t0OwlTransforms.data());
            owlInstanceGroupSetTransforms(OD.IAS,1,(const float*)t1OwlTransforms.data());
            owlInstanceGroupSetVisibilityMasks(OD.IAS, owlVisibilityMasks.data());
            owlBufferResize(OD.instanceToEntityBuffer, instanceToEntity.size());
            owlBufferUpload(OD.instanceToEntityBuffer, instanceToEntity.data());
        }       

        // Build IAS
        groupBuildAccel(OD.IAS);
        owlParamsSetGroup(OD.launchParams, "IAS", OD.IAS);
        
        // Now that IAS have changed, we need to rebuild SBT
        owlBuildSBT(OD.context);

        // Release any old IAS (TODO, don't rebuild if entity edit doesn't effect IAS...)
        if (oldIAS) {owlGroupRelease(oldIAS);}
    
        // Aggregate entities that are light sources (todo: consider emissive volumes...)
        OD.lightEntities.resize(0);
        for (uint32_t eid = 0; eid < Entity::getCount(); ++eid) {
            if (!entities[eid].isInitialized()) continue;
            if (!entities[eid].getTransform()) continue;
            if (!entities[eid].getLight()) continue;
            if (!entities[eid].getMesh()) continue;
            OD.lightEntities.push_back(eid);
        }
        owlBufferResize(OptixData.lightEntitiesBuffer, OD.lightEntities.size());
        owlBufferUpload(OptixData.lightEntitiesBuffer, OD.lightEntities.data());
        OD.LP.numLightEntities = uint32_t(OD.lightEntities.size());
        owlParamsSetRaw(OD.launchParams, "numLightEntities", &OD.LP.numLightEntities);

        // Finally, upload entity structs to the GPU.
        Entity::updateComponents();
        owlBufferUpload(OptixData.entityBuffer,    Entity::getFrontStruct());
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
            owlBufferUpload(OptixData.materialBuffer, OptixData.materialStructs.data());
        }
        
        owlBufferUpload(OD.textureObjectsBuffer, OD.textureObjects.data());
        Texture::updateComponents();
        memcpy(OptixData.textureStructs.data(), Texture::getFrontStruct(), Texture::getCount() * sizeof(TextureStruct));
        owlBufferUpload(OptixData.textureBuffer, OptixData.textureStructs.data());
    }
    
    // Manage transforms
    auto dirtyTransforms = Transform::getDirtyTransforms();
    if (dirtyTransforms.size() > 0) {
        Transform::updateComponents();

        // cudaSetDevice(0);
        owlBufferUpload(OptixData.transformBuffer, Transform::getFrontStruct());
    }   

    // Manage Cameras
    if (Camera::areAnyDirty()) {
        Camera::updateComponents();
        owlBufferUpload(OptixData.cameraBuffer,    Camera::getFrontStruct());
    }    

    // Manage lights
    if (Light::areAnyDirty()) {
        Light::updateComponents();
        owlBufferUpload(OptixData.lightBuffer,     Light::getFrontStruct());
    }
}

void updateLaunchParams()
{
    owlParamsSetRaw(OptixData.launchParams, "frameID", &OptixData.LP.frameID);
    owlParamsSetRaw(OptixData.launchParams, "frameSize", &OptixData.LP.frameSize);
    owlParamsSetRaw(OptixData.launchParams, "cameraEntity", &OptixData.LP.cameraEntity);
    owlParamsSetRaw(OptixData.launchParams, "domeLightIntensity", &OptixData.LP.domeLightIntensity);
    owlParamsSetRaw(OptixData.launchParams, "domeLightExposure", &OptixData.LP.domeLightExposure);
    owlParamsSetRaw(OptixData.launchParams, "domeLightColor", &OptixData.LP.domeLightColor);
    owlParamsSetRaw(OptixData.launchParams, "renderDataMode", &OptixData.LP.renderDataMode);
    owlParamsSetRaw(OptixData.launchParams, "renderDataBounce", &OptixData.LP.renderDataBounce);
    owlParamsSetRaw(OptixData.launchParams, "enableDomeSampling", &OptixData.LP.enableDomeSampling);
    owlParamsSetRaw(OptixData.launchParams, "seed", &OptixData.LP.seed);
    owlParamsSetRaw(OptixData.launchParams, "proj", &OptixData.LP.proj);
    owlParamsSetRaw(OptixData.launchParams, "viewT0", &OptixData.LP.viewT0);
    owlParamsSetRaw(OptixData.launchParams, "viewT1", &OptixData.LP.viewT1);

    owlParamsSetRaw(OptixData.launchParams, "environmentMapID", &OptixData.LP.environmentMapID);
    owlParamsSetRaw(OptixData.launchParams, "environmentMapRotation", &OptixData.LP.environmentMapRotation);
    owlParamsSetBuffer(OptixData.launchParams, "environmentMapRows", OptixData.environmentMapRowsBuffer);
    owlParamsSetBuffer(OptixData.launchParams, "environmentMapCols", OptixData.environmentMapColsBuffer);
    owlParamsSetRaw(OptixData.launchParams, "environmentMapWidth", &OptixData.LP.environmentMapWidth);
    owlParamsSetRaw(OptixData.launchParams, "environmentMapHeight", &OptixData.LP.environmentMapHeight);
    owlParamsSetRaw(OptixData.launchParams, "sceneBBMin", &OptixData.LP.sceneBBMin);
    owlParamsSetRaw(OptixData.launchParams, "sceneBBMax", &OptixData.LP.sceneBBMax);

    OptixData.LP.frameID ++;
}

// Update: This is still prohibitively slow. Official OptiX samples use host pinned memory. 
// Moving to that approach...
// // Different GPUs have different local framebuffers.
// // This function combines those framebuffers on the CPU, then uploads results to device 0.
void mergeFrameBuffers() {
    // For multigpu setups, we currently render to zero-copy memory to merge on the host.
    // So for now, just upload those results to device 0's combined unified frame buffers on the device
    owlBufferUpload(OptixData.combinedFrameBuffer, owlBufferGetPointer(OptixData.frameBuffer, 0));
    
    if (OptixData.enableAlbedoGuide) {
        owlBufferUpload(OptixData.combinedAlbedoBuffer, owlBufferGetPointer(OptixData.albedoBuffer, 0));
    }

    if (OptixData.enableNormalGuide) {
        owlBufferUpload(OptixData.combinedNormalBuffer, owlBufferGetPointer(OptixData.normalBuffer, 0));
    }
}

void denoiseImage() {
    synchronizeDevices();

    auto &OD = OptixData;
    auto cudaStream = owlContextGetStream(OD.context, 0);

    std::vector<OptixImage2D> inputLayers;
    OptixImage2D colorLayer;
    colorLayer.width = OD.LP.frameSize.x;
    colorLayer.height = OD.LP.frameSize.y;
    colorLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    colorLayer.pixelStrideInBytes = 4 * sizeof(float);
    colorLayer.rowStrideInBytes   = OD.LP.frameSize.x * 4 * sizeof(float);
    colorLayer.data   = (CUdeviceptr) owlBufferGetPointer(OD.combinedFrameBuffer, 0);
    inputLayers.push_back(colorLayer);

    OptixImage2D albedoLayer;
    albedoLayer.width = OD.LP.frameSize.x;
    albedoLayer.height = OD.LP.frameSize.y;
    albedoLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    albedoLayer.pixelStrideInBytes = 4 * sizeof(float);
    albedoLayer.rowStrideInBytes   = OD.LP.frameSize.x * 4 * sizeof(float);
    albedoLayer.data   = (CUdeviceptr) owlBufferGetPointer(OD.combinedAlbedoBuffer, 0);
    if (OD.enableAlbedoGuide) inputLayers.push_back(albedoLayer);

    OptixImage2D normalLayer;
    normalLayer.width = OD.LP.frameSize.x;
    normalLayer.height = OD.LP.frameSize.y;
    normalLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    normalLayer.pixelStrideInBytes = 4 * sizeof(float);
    normalLayer.rowStrideInBytes   = OD.LP.frameSize.x * 4 * sizeof(float);
    normalLayer.data   = (CUdeviceptr) owlBufferGetPointer(OD.combinedNormalBuffer, 0);
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
            (CUdeviceptr) owlBufferGetPointer(OD.hdrIntensityBuffer, 0),
            (CUdeviceptr) owlBufferGetPointer(OD.denoiserScratchBuffer, 0),
            scratchSizeInBytes));
    }

    #if !defined(USE_OPTIX70) && !defined(USE_OPTIX71)
    if (OD.enableKernelPrediction) {
        OPTIX_CHECK(optixDenoiserComputeAverageColor(
            OD.denoiser, 
            cudaStream, 
            &inputLayers[0], 
            (CUdeviceptr) owlBufferGetPointer(OD.colorAvgBuffer, 0),
            (CUdeviceptr) owlBufferGetPointer(OD.denoiserScratchBuffer, 0),
            scratchSizeInBytes));
    }
    #endif

    params.denoiseAlpha = 0;    // Don't touch alpha.
    params.blendFactor  = 0.0f; // Show the denoised image only.
    params.hdrIntensity = (CUdeviceptr) owlBufferGetPointer(OD.hdrIntensityBuffer, 0);
    #ifdef USE_OPTIX72
    params.hdrAverageColor = (CUdeviceptr) owlBufferGetPointer(OD.colorAvgBuffer, 0);
    #endif
    
    OPTIX_CHECK(optixDenoiserInvoke(
        OD.denoiser,
        cudaStream,
        &params,
        (CUdeviceptr) owlBufferGetPointer(OD.denoiserStateBuffer, 0),
        OD.denoiserSizes.stateSizeInBytes,
        inputLayers.data(),
        inputLayers.size(),
        /* inputOffsetX */ 0,
        /* inputOffsetY */ 0,
        &outputLayer,
        (CUdeviceptr) owlBufferGetPointer(OD.denoiserScratchBuffer, 0),
        scratchSizeInBytes
    ));
}

inline const char* getGLErrorString( GLenum error )
{
    switch( error )
    {
    case GL_NO_ERROR:            return "No error";
    case GL_INVALID_ENUM:        return "Invalid enum";
    case GL_INVALID_VALUE:       return "Invalid value";
    case GL_INVALID_OPERATION:   return "Invalid operation";
        //case GL_STACK_OVERFLOW:      return "Stack overflow";
        //case GL_STACK_UNDERFLOW:     return "Stack underflow";
    case GL_OUT_OF_MEMORY:       return "Out of memory";
        //case GL_TABLE_TOO_LARGE:     return "Table too large";
    default:                     return "Unknown GL error";
    }
}

#define DO_GL_CHECK
#ifdef DO_GL_CHECK
#    define GL_CHECK( call )                                            \
    do                                                                  \
      {                                                                 \
        call;                                                           \
        GLenum err = glGetError();                                      \
        if( err != GL_NO_ERROR )                                        \
          {                                                             \
            std::stringstream ss;                                       \
            ss << "GL error " <<  getGLErrorString( err ) << " at "     \
               << __FILE__  << "(" <<  __LINE__  << "): " << #call      \
               << std::endl;                                            \
            std::cerr << ss.str() << std::endl;                         \
            throw std::runtime_error( ss.str().c_str() );               \
          }                                                             \
      }                                                                 \
    while (0)


#    define GL_CHECK_ERRORS( )                                          \
    do                                                                  \
      {                                                                 \
        GLenum err = glGetError();                                      \
        if( err != GL_NO_ERROR )                                        \
          {                                                             \
            std::stringstream ss;                                       \
            ss << "GL error " <<  getGLErrorString( err ) << " at "     \
               << __FILE__  << "(" <<  __LINE__  << ")";                \
            std::cerr << ss.str() << std::endl;                         \
            throw std::runtime_error( ss.str().c_str() );               \
          }                                                             \
      }                                                                 \
    while (0)

#else
#    define GL_CHECK( call )   do { call; } while(0)
#    define GL_CHECK_ERRORS( ) do { ;     } while(0)
#endif

void drawFrameBufferToWindow()
{
    synchronizeDevices();
    glFlush();
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    auto &OD = OptixData;
    const void* fbdevptr = owlBufferGetPointer(OD.combinedFrameBuffer,0);

    if (OD.resourceSharingSuccessful) {
        cudaGraphicsMapResources(1, &OD.cudaResourceTex);
        cudaArray_t array;
        cudaGraphicsSubResourceGetMappedArray(&array, OD.cudaResourceTex, 0, 0);
        cudaMemcpyToArray(array, 0, 0, fbdevptr, OD.LP.frameSize.x *  OD.LP.frameSize.y  * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &OD.cudaResourceTex);
    } else {
        GL_CHECK(glBindTexture(GL_TEXTURE_2D, OD.imageTexID));
        glEnable(GL_TEXTURE_2D);
        GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D,0,
                                    0, 0,
                                    OD.LP.frameSize.x, OD.LP.frameSize.y,
                                    GL_RGBA, GL_FLOAT, fbdevptr));    
    }

    // Draw pixels from optix frame buffer
    glEnable(GL_FRAMEBUFFER_SRGB); 
    glViewport(0, 0, OD.LP.frameSize.x, OD.LP.frameSize.y);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
        
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
            
    glDisable(GL_DEPTH_TEST);    
    
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
    if (NVISII.headlessMode) return;

    enqueueCommand([width, height] () {
        using namespace Libraries;
        auto glfw = GLFW::Get();
        glfw->resize_window("NVISII", width, height);
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

    enqueueCommandAndWait([useAlbedoGuide, useNormalGuide, useKernelPrediction](){
        OptixData.enableAlbedoGuide = useAlbedoGuide;
        OptixData.enableNormalGuide = useNormalGuide;
        #ifdef USE_OPTIX70
        if (useKernelPrediction) {
            throw std::runtime_error("Error: the current build of NVISII uses optix"
            " 7.0, which does not support the kernel prediction denoiser.");
        }
        #endif
        OptixData.enableKernelPrediction = useKernelPrediction;

        // Reconfigure denoiser
        // Allocate required buffers
        if (!OptixData.hdrIntensityBuffer)
            OptixData.hdrIntensityBuffer = owlDeviceBufferCreate(OptixData.context, OWL_USER_TYPE(float), 1, nullptr);
        if (!OptixData.colorAvgBuffer)
            OptixData.colorAvgBuffer = owlDeviceBufferCreate(OptixData.context, OWL_USER_TYPE(float), 4, nullptr);
        if (!OptixData.denoiserScratchBuffer)
            OptixData.denoiserScratchBuffer = owlDeviceBufferCreate(OptixData.context, OWL_USER_TYPE(void*), 1, nullptr);
        if (!OptixData.denoiserStateBuffer)
            OptixData.denoiserStateBuffer = owlDeviceBufferCreate(OptixData.context, OWL_USER_TYPE(void*), 1, nullptr);
        
        // Setup denoiser
        OptixDenoiserOptions options;
        if ((!useAlbedoGuide) && (!useNormalGuide)) options.inputKind = OPTIX_DENOISER_INPUT_RGB;
        else if (!useNormalGuide) options.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;
        else options.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;

        #ifdef USE_OPTIX70
        options.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
        #endif

        if (OptixData.denoiser) optixDenoiserDestroy(OptixData.denoiser);
        
        auto optixContext = owlContextGetOptixContext(OptixData.context, 0);
        auto cudaStream = owlContextGetStream(OptixData.context, 0);
        OPTIX_CHECK(optixDenoiserCreate(optixContext, &options, &OptixData.denoiser));

        OptixDenoiserModelKind kind;
        #if defined(USE_OPTIX70) || defined(USE_OPTIX71)
            kind = OPTIX_DENOISER_MODEL_KIND_HDR;
        #else
            if (OptixData.enableKernelPrediction) kind = OPTIX_DENOISER_MODEL_KIND_AOV;
            else kind = OPTIX_DENOISER_MODEL_KIND_HDR;
        #endif
        
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
        owlBufferResize(OptixData.denoiserScratchBuffer, scratchSizeInBytes);
        owlBufferResize(OptixData.denoiserStateBuffer, OptixData.denoiserSizes.stateSizeInBytes);
        
        optixDenoiserSetup (
            OptixData.denoiser, 
            (cudaStream_t) cudaStream, 
            (unsigned int) OptixData.LP.frameSize.x, 
            (unsigned int) OptixData.LP.frameSize.y, 
            (CUdeviceptr) owlBufferGetPointer(OptixData.denoiserStateBuffer, 0), 
            OptixData.denoiserSizes.stateSizeInBytes,
            (CUdeviceptr) owlBufferGetPointer(OptixData.denoiserScratchBuffer, 0), 
            scratchSizeInBytes
        );
    });
}

std::vector<float> readFrameBuffer() {
    std::vector<float> frameBuffer(OptixData.LP.frameSize.x * OptixData.LP.frameSize.y * 4);

    enqueueCommandAndWait([&frameBuffer] () {
        int num_devices = getDeviceCount();
        synchronizeDevices();

        const glm::vec4 *fb = (const glm::vec4*)owlBufferGetPointer(OptixData.combinedFrameBuffer,0);
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
        if (!NVISII.headlessMode) {
            if ((width != WindowData.currentSize.x) || (height != WindowData.currentSize.y))
            {
                using namespace Libraries;
                auto glfw = GLFW::Get();
                glfw->resize_window("NVISII", width, height);
                initializeFrameBuffer(width, height);
            }
        }
        
        OptixData.LP.seed = seed;

        resizeOptixFrameBuffer(width, height);
        resetAccumulation();
        updateComponents();
        int numGPUs = owlGetDeviceCount(OptixData.context);

        for (uint32_t i = 0; i < samplesPerPixel; ++i) {
            // std::cout<<i<<std::endl;
            if (!NVISII.headlessMode) {
                auto glfw = Libraries::GLFW::Get();
                glfw->poll_events();
                glfw->swap_buffers("NVISII");
                glClearColor(1,1,1,1);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            }

            updateLaunchParams();
            for (uint32_t deviceID = 0; deviceID < numGPUs; deviceID++) {
                cudaSetDevice(deviceID);
                cudaEventRecord(NVISII.events[deviceID].first);
                owlAsyncLaunch2DOnDevice(OptixData.rayGen, OptixData.LP.frameSize.x * OptixData.LP.frameSize.y, 1, deviceID, OptixData.launchParams);
                cudaEventRecord(NVISII.events[deviceID].second);
            }
            for (uint32_t deviceID = 0; deviceID < numGPUs; deviceID++) {
                cudaEventSynchronize(NVISII.events[deviceID].second);
                cudaEventElapsedTime(&NVISII.times[deviceID], NVISII.events[deviceID].first, NVISII.events[deviceID].second);
            }
            updateGPUWeights();
            mergeFrameBuffers();

            if (!NVISII.headlessMode) {
                if (OptixData.enableDenoiser)
                {
                    denoiseImage();
                }

                drawFrameBufferToWindow();
                glfwSetWindowTitle(WindowData.window, 
                    (std::to_string(i) + std::string("/") + std::to_string(samplesPerPixel)).c_str());
            }

            if (verbose) {
                std::cout<< "\r" << i << "/" << samplesPerPixel;
            }
        }      
        if (!NVISII.headlessMode) {
            glfwSetWindowTitle(WindowData.window, 
                (std::to_string(samplesPerPixel) + std::string("/") + std::to_string(samplesPerPixel) 
                + std::string(" - done!")).c_str());
        }
        
        if (verbose) {
            std::cout<<"\r "<< samplesPerPixel << "/" << samplesPerPixel <<" - done!" << std::endl;
        }

        if (OptixData.enableDenoiser)
        {
            denoiseImage();
        }

        synchronizeDevices();
        const glm::vec4 *fb = (const glm::vec4*) owlBufferGetPointer(OptixData.combinedFrameBuffer,0);
        cudaMemcpyAsync(frameBuffer.data(), fb, width * height * sizeof(glm::vec4), cudaMemcpyDeviceToHost);
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

    enqueueCommandAndWait([](){});

    enqueueCommandAndWait([&frameBuffer, width, height, startFrame, frameCount, bounce, _option, seed] () {
        if (!NVISII.headlessMode) {
            if ((width != WindowData.currentSize.x) || (height != WindowData.currentSize.y))
            {
                using namespace Libraries;
                auto glfw = GLFW::Get();
                glfw->resize_window("NVISII", width, height);
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
        else if (option == std::string("device_id")) {
            OptixData.LP.renderDataMode = RenderDataFlags::DEVICE_ID;
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
        int numGPUs = owlGetDeviceCount(OptixData.context);

        for (uint32_t i = startFrame; i < frameCount; ++i) {
            // std::cout<<i<<std::endl;
            if (!NVISII.headlessMode) {
                auto glfw = Libraries::GLFW::Get();
                glfw->poll_events();
                glfw->swap_buffers("NVISII");
                glClearColor(1,1,1,1);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            }

            updateLaunchParams();

            for (uint32_t deviceID = 0; deviceID < numGPUs; deviceID++) {
                cudaSetDevice(deviceID);
                cudaEventRecord(NVISII.events[deviceID].first);
                owlAsyncLaunch2DOnDevice(OptixData.rayGen, OptixData.LP.frameSize.x * OptixData.LP.frameSize.y, 1, deviceID, OptixData.launchParams);
                cudaEventRecord(NVISII.events[deviceID].second);
            }
            for (uint32_t deviceID = 0; deviceID < numGPUs; deviceID++) {
                cudaEventSynchronize(NVISII.events[deviceID].second);
                cudaEventElapsedTime(&NVISII.times[deviceID], NVISII.events[deviceID].first, NVISII.events[deviceID].second);
            }
            updateGPUWeights();
            mergeFrameBuffers();
            
            // Dont run denoiser to raw data rendering
            // if (OptixData.enableDenoiser)
            // {
            //     denoiseImage();
            // }

            if (!NVISII.headlessMode) {
                drawFrameBufferToWindow();
            }
        }

        synchronizeDevices();

        const glm::vec4 *fb = (const glm::vec4*) owlBufferGetPointer(OptixData.combinedFrameBuffer,0);
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
    NVISII.callback = nullptr;

    initializeComponentFactories(maxEntities, maxCameras, maxTransforms, maxMeshes, maxMaterials, maxLights, maxTextures, maxVolumes);

    auto loop = [windowOnTop]() {
        NVISII.render_thread_id = std::this_thread::get_id();
        NVISII.headlessMode = false;

        auto glfw = Libraries::GLFW::Get();
        WindowData.window = glfw->create_window("NVISII", 512, 512, windowOnTop, true, true);
        WindowData.currentSize = WindowData.lastSize = ivec2(512, 512);
        glfw->make_context_current("NVISII");
        glfw->poll_events();

        initializeOptix(/*headless = */ false);
        initializeImgui();

        int numGPUs = owlGetDeviceCount(OptixData.context);

        while (!stopped)
        {
            /* Poll events from the window */
            glfw->poll_events();
            glfw->swap_buffers("NVISII");
            glClearColor(1,1,1,1);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            if (NVISII.callback && NVISII.callbackMutex.try_lock()) {
                NVISII.callback();
                NVISII.callbackMutex.unlock();
            }

            static double start=0;
            static double stop=0;
            start = glfwGetTime();

            if (!lazyUpdatesEnabled) {
                updateFrameBuffer();
                updateComponents();
                updateLaunchParams();

                for (uint32_t deviceID = 0; deviceID < numGPUs; deviceID++) {
                    cudaSetDevice(deviceID);
                    cudaEventRecord(NVISII.events[deviceID].first, owlParamsGetCudaStream(OptixData.launchParams, deviceID));
                    owlAsyncLaunch2DOnDevice(OptixData.rayGen, OptixData.LP.frameSize.x * OptixData.LP.frameSize.y, 1, deviceID, OptixData.launchParams);
                    cudaEventRecord(NVISII.events[deviceID].second, owlParamsGetCudaStream(OptixData.launchParams, deviceID));
                }
                owlLaunchSync(OptixData.launchParams);
                for (uint32_t deviceID = 0; deviceID < numGPUs; deviceID++) {
                    cudaEventElapsedTime(&NVISII.times[deviceID], NVISII.events[deviceID].first, NVISII.events[deviceID].second);
                }
                updateGPUWeights();
                mergeFrameBuffers();

                if (OptixData.enableDenoiser) {
                    denoiseImage();
                }
            }
            // glm::vec4* samplePtr = (glm::vec4*) owlBufferGetPointer(OptixData.accumBuffer,0);
            // glm::vec4* mvecPtr = (glm::vec4*) owlBufferGetPointer(OptixData.mvecBuffer,0);
            // glm::vec4* t0AlbPtr = (glm::vec4*) owlBufferGetPointer(OptixData.scratchBuffer,0);
            // glm::vec4* t1AlbPtr = (glm::vec4*) owlBufferGetPointer(OptixData.albedoBuffer,0);
            // glm::vec4* fbPtr = (glm::vec4*) owlBufferGetPointer(OptixData.frameBuffer,0);
            // glm::vec4* sPtr = (glm::vec4*) owlBufferGetPointer(OptixData.normalBuffer,0);
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
            if (OptixData.cudaResourceTex) {
                cudaGraphicsUnregisterResource(OptixData.cudaResourceTex);
                OptixData.cudaResourceTex = 0;
            }
            glDeleteTextures(1, &OptixData.imageTexID);
        }

        ImGui::DestroyContext();
        if (glfw->does_window_exist("NVISII")) glfw->destroy_window("NVISII");

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
    NVISII.callback = nullptr;

    initializeComponentFactories(maxEntities, maxCameras, maxTransforms, maxMeshes, maxMaterials, maxLights, maxTextures, maxVolumes);

    auto loop = []() {
        NVISII.render_thread_id = std::this_thread::get_id();
        NVISII.headlessMode = true;

        initializeOptix(/*headless = */ true);

        while (!stopped)
        {
            if(NVISII.callback){
                NVISII.callback();
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
    NVISII.callback = callback;
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
    enqueueCommandAndWait([] () { lazyUpdatesEnabled = false; });
}

void disableUpdates()
{
    enqueueCommandAndWait([] () { lazyUpdatesEnabled = true; });
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
    if (NVISII.headlessMode) return false;
    auto glfw = Libraries::GLFW::Get();
    std::transform(button.data(), button.data() + button.size(), 
        std::addressof(button[0]), [](unsigned char c){ return std::toupper(c); });
    bool pressed, prevPressed;
    if (button.compare("MOUSE_LEFT") == 0) {
        pressed = glfw->get_button_action("NVISII", 0) == 1;
        prevPressed = glfw->get_button_action_prev("NVISII", 0) == 1;
    }
    else if (button.compare("MOUSE_RIGHT") == 0) {
        pressed = glfw->get_button_action("NVISII", 1) == 1;
        prevPressed = glfw->get_button_action_prev("NVISII", 1) == 1;
    }
    else if (button.compare("MOUSE_MIDDLE") == 0) {
        pressed = glfw->get_button_action("NVISII", 2) == 1;
        prevPressed = glfw->get_button_action_prev("NVISII", 2) == 1;
    }
    else {
        pressed = glfw->get_key_action("NVISII", glfw->get_key_code(button)) == 1;
        prevPressed = glfw->get_key_action_prev("NVISII", glfw->get_key_code(button)) == 1;
    }
    
    return pressed && !prevPressed;
}

bool isButtonHeld(std::string button) {
    if (NVISII.headlessMode) return false;
    auto glfw = Libraries::GLFW::Get();
    std::transform(button.data(), button.data() + button.size(), 
        std::addressof(button[0]), [](unsigned char c){ return std::toupper(c); });
    if (button.compare("MOUSE_LEFT") == 0) return glfw->get_button_action("NVISII", 0) >= 1;
    if (button.compare("MOUSE_RIGHT") == 0) return glfw->get_button_action("NVISII", 1) >= 1;
    if (button.compare("MOUSE_MIDDLE") == 0) return glfw->get_button_action("NVISII", 2) >= 1;
    return glfw->get_key_action("NVISII", glfw->get_key_code(button)) >= 1;
}

vec2 getCursorPos()
{
    if (NVISII.headlessMode) return vec2(NAN, NAN);
    auto glfw = Libraries::GLFW::Get();
    auto pos = glfw->get_cursor_pos("NVISII");
    return vec2(pos[0], pos[1]);
}

void setCursorMode(std::string mode)
{
    if (NVISII.headlessMode) return;
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
    if (NVISII.headlessMode) return ivec2(NAN, NAN);
    return WindowData.currentSize;
}

bool shouldWindowClose()
{
    if (NVISII.headlessMode) return false;
    auto glfw = Libraries::GLFW::Get();
    return glfw->should_close("NVISII");
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
    else if (option == std::string("device_id")) {
        OptixData.LP.renderDataMode = RenderDataFlags::DEVICE_ID;
    }
    else {
        throw std::runtime_error(std::string("Error, unknown option : \"") + option + std::string("\". ")
        + std::string("See documentation for available options"));
    }

    resetAccumulation();
}

};
