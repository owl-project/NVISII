#include <visii/visii.h>

#include <glfw_implementation/glfw.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <ImGuizmo.h>
#include <visii/utilities/colors.h>
#include <owl/owl.h>
#include <cuda_gl_interop.h>

#include <launchParams.h>
#include <deviceCode.h>

#include <thread>
#include <future>

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

static struct OptixData {
    OWLContext context;
    OWLModule module;
    OWLLaunchParams launchParams;
    LaunchParams LP;
    GLuint imageTexID = -1;
    cudaGraphicsResource_t cudaResourceTex;
    OWLBuffer frameBuffer;
    OWLBuffer accumBuffer;
    OWLRayGen rayGen;
    OWLMissProg missProg;
    OWLGeomType trianglesGeomType;
} OptixData;

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

void setCameraEntity(Entity* camera_entity)
{
    if (!camera_entity) throw std::runtime_error("Error: camera entity was nullptr/None");
    if (!camera_entity->isInitialized()) throw std::runtime_error("Error: camera entity is uninitialized");

    OptixData.LP.cameraEntity = camera_entity->getStruct();
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

void updateFrameBuffer()
{
    glfwGetFramebufferSize(WindowData.window, &WindowData.currentSize.x, &WindowData.currentSize.y);

    if ((WindowData.currentSize.x != WindowData.lastSize.x) || (WindowData.currentSize.y != WindowData.lastSize.y))  {
        WindowData.lastSize.x = WindowData.currentSize.x; WindowData.lastSize.y = WindowData.currentSize.y;
        OptixData.LP.frameSize = WindowData.currentSize;
        initializeFrameBuffer(WindowData.currentSize.x, WindowData.currentSize.y);
        owlBufferResize(OptixData.frameBuffer, WindowData.currentSize.x*WindowData.currentSize.y);
        owlBufferResize(OptixData.accumBuffer, WindowData.currentSize.x*WindowData.currentSize.y);
    }
}

void initializeOptix()
{
    using namespace glm;
    auto &OD = OptixData;
    OD.context = owlContextCreate(/*requested Device IDs*/ nullptr, /* Num Devices */ 0);
    // owlContextSetRayTypeCount(context, 2); // for both "feeler" and query rays on the same accel.
    OD.module = owlModuleCreate(OD.context, ptxCode);
    
    /* Setup Optix Launch Params */
    OWLVarDecl launchParamVars[] = {
        { "frameSize",         OWL_USER_TYPE(glm::ivec2),         OWL_OFFSETOF(LaunchParams, frameSize)},
        { "fbPtr",             OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, fbPtr)},
        { "accumPtr",          OWL_BUFPTR,                        OWL_OFFSETOF(LaunchParams, accumPtr)},
        { "world",             OWL_GROUP,                         OWL_OFFSETOF(LaunchParams, world)},
        { "cameraEntity",      OWL_USER_TYPE(EntityStruct),       OWL_OFFSETOF(LaunchParams, cameraEntity)},
        { /* sentinel to mark end of list */ }
    };
    OD.launchParams = owlLaunchParamsCreate(OD.context, sizeof(LaunchParams), launchParamVars, -1);
    
    initializeFrameBuffer(1024, 1024);
    OD.frameBuffer = owlManagedMemoryBufferCreate(OD.context,OWL_USER_TYPE(glm::vec4),1024*1024, nullptr);
    OD.accumBuffer = owlDeviceBufferCreate(OD.context,OWL_INT,1024*1024, nullptr);
    OD.LP.frameSize = glm::ivec2(1024, 1024);
    owlLaunchParamsSetBuffer(OD.launchParams, "fbPtr", OD.frameBuffer);
    owlLaunchParamsSetBuffer(OD.launchParams, "accumPtr", OD.accumBuffer);
    owlLaunchParamsSetRaw(OD.launchParams, "frameSize", &OD.LP.frameSize);


    /* Temporary test code */
    const int NUM_VERTICES = 8;
    vec3 vertices[NUM_VERTICES] =
    {
        { -1.f,-1.f,-.1f },
        { +1.f,-1.f,-.1f },
        { -1.f,+1.f,-.1f },
        { +1.f,+1.f,-.1f },
        { -1.f,-1.f,+.1f },
        { +1.f,-1.f,+.1f },
        { -1.f,+1.f,+.1f },
        { +1.f,+1.f,+.1f }
    };

    const int NUM_INDICES = 12;
    ivec3 indices[NUM_INDICES] =
    {
        { 0,1,3 }, { 2,3,0 },
        { 5,7,6 }, { 5,6,4 },
        { 0,4,5 }, { 0,5,1 },
        { 2,3,7 }, { 2,7,6 },
        { 1,5,7 }, { 1,7,3 },
        { 4,0,2 }, { 4,2,6 }
    };

    OWLVarDecl trianglesGeomVars[] = {
        { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
        { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
        { "colors",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,colors)}
        // { "color",  OWL_FLOAT3, OWL_OFFSETOF(TrianglesGeomData,color)}
    };
    OD.trianglesGeomType
        = owlGeomTypeCreate(OD.context,
                            OWL_GEOM_TRIANGLES,
                            sizeof(TrianglesGeomData),
                            trianglesGeomVars,3);
    owlGeomTypeSetClosestHit(OD.trianglesGeomType, /*ray type */ 0, OD.module,"TriangleMesh");
    
    OWLBuffer vertexBuffer = owlDeviceBufferCreate(OD.context,OWL_FLOAT3,NUM_VERTICES,vertices);
    OWLBuffer indexBuffer = owlDeviceBufferCreate(OD.context,OWL_INT3,NUM_INDICES,indices);
    OWLGeom trianglesGeom = owlGeomCreate(OD.context,OD.trianglesGeomType);
    owlTrianglesSetVertices(trianglesGeom,vertexBuffer,NUM_VERTICES,sizeof(vec3),0);
    owlTrianglesSetIndices(trianglesGeom,indexBuffer, NUM_INDICES,sizeof(ivec3),0);
    owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
    owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);
    owlGeomSetBuffer(trianglesGeom,"colors",nullptr);
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
}

void updateComponents()
{

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
    owlLaunchParamsSetRaw(OptixData.launchParams, "frameSize", &OptixData.LP.frameSize);
    owlLaunchParamsSetRaw(OptixData.launchParams, "cameraEntity", &OptixData.LP.cameraEntity);

    // auto bumesh_transform_struct = bumesh_transform->get_struct();
    // owlLaunchParamsSetRaw(launchParams,"bumesh_transform",&bumesh_transform_struct);

    // auto tri_mesh_transform_struct = tri_mesh_transform->get_struct();
    // owlLaunchParamsSetRaw(launchParams,"tri_mesh_transform",&tri_mesh_transform_struct);
}

void traceRays()
{
    auto &OD = OptixData;
    
    static double start=0;
    static double stop=0;

    /* Trace Rays */
    start = glfwGetTime();
    owlParamsLaunch2D(OD.rayGen, OD.LP.frameSize.x, OD.LP.frameSize.y, OD.launchParams);
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
    stop = glfwGetTime();
    glfwSetWindowTitle(WindowData.window, std::to_string(1.f / (stop - start)).c_str());

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

void initializeInteractive()
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

    auto loop = []() {
        auto glfw = Libraries::GLFW::Get();
        WindowData.window = glfw->create_window("ViSII", 1024, 1024, false, true, true);
        WindowData.currentSize = WindowData.lastSize = ivec2(1024, 1024);
        glfw->make_context_current("ViSII");
        glfw->poll_events();

        initializeOptix();

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

            auto newColor = Colors::hsvToRgb({float(glfwGetTime() * .1f), 1.0f, 1.0f});
            glClearColor(newColor[0],newColor[1],newColor[2],1);
            glClear(GL_COLOR_BUFFER_BIT);

            updateFrameBuffer();
            updateComponents();
            updateLaunchParams();
            traceRays();
            drawGUI();

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
    Camera::initializeFactory();
    Entity::initializeFactory();
    Transform::initializeFactory();
    Material::initializeFactory();
    Mesh::initializeFactory();

    // auto loop = []() {
    //     while (!close)
    //     {
    //         if (close) break;
    //     }
    // };

    // renderThread = thread(loop);
}

void cleanup()
{
    if (initialized == true) {
        /* cleanup window if open */
        if (close == false) {
            close = true;
            renderThread.join();
        }
    }
    initialized = false;
}