#include <visii/visii.h>
#include <glfw_implementation/glfw.h>
#include <visii/utilities/colors.h>

#include <thread>
#include <future>

std::promise<void> exitSignal;
std::thread renderThread;
GLFWwindow* window = nullptr;

bool close = true;
void initialize()
{
    // don't initialize more than once
    if (close == false) return;

    close = false;
    Camera::initializeFactory();
    Entity::initializeFactory();
    Transform::initializeFactory();
    Material::initializeFactory();
    Mesh::initializeFactory();

    auto loop = []() {
        auto glfw = Libraries::GLFW::Get();
        window = glfw->create_window("ViSII", 1024, 1024, false, true, true);
        glfw->make_context_current("ViSII");
        glfw->poll_events();
        while (!close)
        {
            /* Poll events from the window */
            glfw->poll_events();
            glfw->swap_buffers("ViSII");

            auto newColor = Colors::hsvToRgb({float(glfwGetTime() * .1f), 1.0f, 1.0f});
            glClearColor(newColor[0],newColor[1],newColor[2],1);
            glClear(GL_COLOR_BUFFER_BIT);

            if (close) break;
        }

        if (glfw->does_window_exist("ViSII")) glfw->destroy_window("ViSII");
    };

    renderThread = thread(loop);
}

void cleanup()
{
    if (close == false) {
        close = true;
        renderThread.join();
    }
}