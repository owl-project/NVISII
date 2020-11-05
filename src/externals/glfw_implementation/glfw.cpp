#include "glfw.h"
#include <algorithm>
#include <cctype>
#include <string>

int windowed_xpos, windowed_ypos, windowed_width, windowed_height;

void resize_window_callback(GLFWwindow * window, int width, int height) {
    
}

void close_window_callback(GLFWwindow *window)
{
    auto window_key = Libraries::GLFW::Get()->get_key_from_ptr(window);
    if (window_key.size() > 0) {
        Libraries::GLFW::Get()->set_should_close(window_key, true);
    }    
    
    // disable closing the window for now
    glfwSetWindowShouldClose(window, GLFW_FALSE);
}

void cursor_position_callback(GLFWwindow * window, double xpos, double ypos) {
    auto window_key = Libraries::GLFW::Get()->get_key_from_ptr(window);
    if (window_key.size() > 0) {
        Libraries::GLFW::Get()->set_cursor_pos(window_key, xpos, ypos);
    }
}

void scroll_callback(GLFWwindow * window, double xoffset, double yoffset) {
    auto window_key = Libraries::GLFW::Get()->get_key_from_ptr(window);
    if (window_key.size() > 0) {
        Libraries::GLFW::Get()->set_scroll(window_key, xoffset, yoffset);
    }
}

void mouse_button_callback(GLFWwindow * window, int button, int action, int mods)
{
    auto window_key = Libraries::GLFW::Get()->get_key_from_ptr(window);
    if (window_key.size() > 0) {
        Libraries::GLFW::Get()->set_button_data(window_key, button, action, mods);
    }
}

void key_callback( GLFWwindow* window, int key, int scancode, int action, int mods )
{
    auto window_key = Libraries::GLFW::Get()->get_key_from_ptr(window);
    if (window_key.size() > 0) {
        Libraries::GLFW::Get()->set_key_data(window_key, key, scancode, action, mods);
    }

    if (action == GLFW_PRESS && ((key == GLFW_KEY_ENTER && mods == GLFW_MOD_ALT) ||
        (key == GLFW_KEY_F11 && mods == GLFW_MOD_ALT)))
    {
        // if (!Libraries::GLFW::Get()->is_swapchain_out_of_date(window_key))
        {
            Libraries::GLFW::Get()->toggle_fullscreen(window_key);
        }
    }
}

namespace Libraries {
    GLFW::GLFW() { }
    GLFW::~GLFW() { }

    GLFW* GLFW::Get() {
        static GLFW instance;
        if (!instance.is_initialized()) instance.initialize();
        return &instance;
    }
    
    bool GLFW::initialize() {
        auto result = glfwInit();
        if (!result)
            throw std::runtime_error( std::string("Error: Failed to initialize " + std::to_string(result)));

        window_mutex = std::make_shared<std::mutex>();
        initialized = true;
        return true;
    }

    unordered_map<string, GLFW::Window> &GLFW::Windows()
    {
        static unordered_map<string, Window> windows;
        return windows;
    }

    GLFWwindow* GLFW::create_window(string key, uint32_t width, uint32_t height, bool floating, bool resizable, bool decorated) {
        /* If uninitialized, or if window already exists, return false */
        if (initialized == false)
            throw std::runtime_error( std::string("Error: uninitialized, cannot create window."));
        auto ittr = Windows().find(key);
        if ( ittr != Windows().end() )
            throw std::runtime_error( std::string("Error: window already exists, cannot create window"));

        // For Vulkan, if we were using it...
        // glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
        glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);
        glfwWindowHint(GLFW_DECORATED, (decorated) ? GLFW_TRUE : GLFW_FALSE);
        glfwWindowHint(GLFW_RESIZABLE, (resizable) ? GLFW_TRUE : GLFW_FALSE);
        glfwWindowHint(GLFW_FLOATING, (floating) ? GLFW_TRUE : GLFW_FALSE);
        // glfwWindowHint( GLFW_DOUBLEBUFFER, GL_FALSE );

        Window window = {};
        auto ptr = glfwCreateWindow(width, height, key.c_str(), NULL, NULL);
        if (!ptr)
            throw std::runtime_error( std::string("Error: Failed to create OpenGL window. Minimum OpenGL version is 4.3."));

        glfwSetWindowSizeCallback(ptr, &resize_window_callback);
        glfwSetScrollCallback(ptr, &scroll_callback);
        glfwSetCursorPosCallback(ptr, &cursor_position_callback);
        glfwSetMouseButtonCallback(ptr, &mouse_button_callback);
        glfwSetWindowCloseCallback(ptr, &close_window_callback);
        glfwSetKeyCallback(ptr, &key_callback);
        glfwSetWindowSizeLimits(ptr, 1, 1, GLFW_DONT_CARE, GLFW_DONT_CARE);
        window.ptr = ptr;
        // window.swapchain_out_of_date = true;
        Windows()[key] = window;
        return window.ptr;
    }

    bool GLFW::make_context_current(std::string key) {
        /* If uninitialized, or if window already exists, return false */
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot make context current."));
        auto ittr = Windows().find(key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot make context current."));
        
        auto mutex = window_mutex.get();
        std::lock_guard<std::mutex> lock(*mutex);
        
        auto window = Windows()[key];
        glfwMakeContextCurrent(window.ptr);
        glfwSwapInterval( 0 );

        if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
        {
            std::cout << "Failed to initialize OpenGL context" << std::endl;
            throw std::runtime_error( std::string("Failed to initialize OpenGL context"));
        }

        // gladLoadGL();

        return true;
    }

    void GLFW::swap_buffers(string key) {
        /* If uninitialized, or if window already exists, return false */
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot make context current."));
        auto ittr = Windows().find(key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot make context current."));
        
        auto window = Windows()[key];
        glfwSwapBuffers(window.ptr);
    }

    bool GLFW::resize_window(std::string key, uint32_t width, uint32_t height) {
        /* If uninitialized, or if window already exists, return false */
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot resize window."));
        auto ittr = Windows().find(key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot resize window."));

        auto mutex = window_mutex.get();
        std::lock_guard<std::mutex> lock(*mutex);
        
        auto window = Windows()[key];
        glfwSetWindowSize(window.ptr, width, height);
        // Libraries::GLFW::Get()->set_swapchain_out_of_date(key);
        return true;
    }

    bool GLFW::toggle_fullscreen(std::string key) {
        /* If uninitialized, or if window already exists, return false */
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot resize window."));
        auto ittr = Windows().find(key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot resize window."));

        auto mutex = window_mutex.get();
        std::lock_guard<std::mutex> lock(*mutex);

        auto window = Windows()[key];

        if (glfwGetWindowMonitor(window.ptr))
        {
            glfwSetWindowMonitor(window.ptr, NULL,
                                windowed_xpos, windowed_ypos,
                                windowed_width, windowed_height, 0);
        }
        else
        {
            GLFWmonitor* monitor = glfwGetPrimaryMonitor();
            if (monitor)
            {
                const GLFWvidmode* mode = glfwGetVideoMode(monitor);
                glfwGetWindowPos(window.ptr, &windowed_xpos, &windowed_ypos);
                glfwGetWindowSize(window.ptr, &windowed_width, &windowed_height);
                glfwSetWindowMonitor(window.ptr, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
            }
        }
        
        // Libraries::GLFW::Get()->set_swapchain_out_of_date(key);
        
        return true;
    }

    bool GLFW::set_window_visibility(std::string key, bool visible) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot set window visibility."));
        auto ittr = Windows().find(key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot set window visibility."));

        auto mutex = window_mutex.get();
        std::lock_guard<std::mutex> lock(*mutex);

        auto window = Windows()[key];
        if (visible)
            glfwShowWindow(window.ptr);
        else 
            glfwHideWindow(window.ptr);
            
        // Libraries::GLFW::Get()->set_swapchain_out_of_date(key);
        return true;
    }

    bool GLFW::set_window_pos(std::string key, uint32_t x, uint32_t y) {
        /* If uninitialized, or if window already exists, return false */
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot set window pos."));

        auto ittr = Windows().find(key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exists, cannot set window pos."));
        
        auto window = Windows()[key];
        glfwSetWindowPos(window.ptr, x, y);
        return true;
    }

    bool GLFW::destroy_window(string key) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot destroy window."));

        auto mutex = window_mutex.get();
        std::lock_guard<std::mutex> lock(*mutex);

        auto ittr = Windows().find(key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot destroy window."));
        auto window = Windows()[key];

        glfwDestroyWindow(window.ptr);

        Windows().erase(key);
        return true;
    }

    GLFWwindow* GLFW::get_ptr(std::string key) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot get window ptr from key."));
        
        for (auto &window : Windows()) {
            if (window.first.compare(key) == 0)
                return window.second.ptr;
        }
        return nullptr;
    }

    std::string GLFW::get_key_from_ptr(GLFWwindow* ptr) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot get window key from ptr."));

        for (auto &window : Windows()) {
            if (window.second.ptr == ptr)
                return window.first;
        }
        return "";
    }

    std::vector<std::string> GLFW::get_window_keys() {
        std::vector<std::string> result;

        /* If not initialized, or window doesnt exist, return false. */
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot get window keys."));
     
        for (auto &window : Windows()) {
            result.push_back(window.first);
        }
        return result;
    }

    bool GLFW::does_window_exist(string key) {
        /* If not initialized, or window doesnt exist, return false. */
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot query window existance."));
           
        auto window = Windows().find(key);
        return ( window != Windows().end() );
    }

    bool GLFW::poll_events() {
        /* If not initialized, or window doesnt exist, return false. */
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot poll events."));

        if ((Windows().size() == 0) || (initialized == false)) return false;
 
        for (auto &i : Windows()) {
            if (glfwWindowShouldClose(i.second.ptr)) {
                destroy_window(i.first);
                /* Break here required. Erase modifies iterator used by for loop. */
                break;
            }
            // reset scroll
            set_scroll(i.first, 0, 0);
            // copy events
            std::memcpy(i.second.keysPrev, i.second.keys, sizeof(i.second.keysPrev));
            std::memcpy(i.second.buttonsPrev, i.second.buttons, sizeof(i.second.buttonsPrev));
        }
        glfwPollEvents();
        
        return true;
    }

    bool GLFW::wait_events() {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot wait for vents."));
            
        glfwWaitEvents();
        return true;
    }

    bool GLFW::post_empty_event() {
        /* If not initialized, or window doesnt exist, return false. */
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot post an empty event."));

        glfwPostEmptyEvent();
        return true;
    }

    bool GLFW::set_should_close(std::string window_key, bool should_close)
    {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot set should close."));

        auto ittr = Windows().find(window_key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot set should close."));

        auto window = &Windows()[window_key];
        window->shouldClose = should_close;
        return true;
    }

    bool GLFW::should_close(std::string window_key) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized."));
        
        auto ittr = Windows().find(window_key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist."));

        auto window = &Windows()[window_key];
        return window->shouldClose;
    }

    std::shared_ptr<std::mutex> GLFW::get_mutex()
    {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, can't get window mutex."));
        
        return window_mutex;
    }

    std::array<float, 2> GLFW::get_size(std::string window_key)
    {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot get window size."));
        
        auto ittr = Windows().find(window_key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot get window size."));

        auto window = &Windows()[window_key];
        return window->extent;
    }

    bool GLFW::set_scroll(std::string key, double xoffset, double yoffset) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot set cursor position."));

        auto ittr = Windows().find(key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot set cursor position."));

        auto window = &Windows()[key];
        window->xscroll = xoffset;
        window->yscroll = yoffset;
        return true;
    }

    std::array<float, 2> GLFW::get_scroll_offset(std::string key) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot get cursor position."));

        auto ittr = Windows().find(key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot get cursor position."));
        auto window = &Windows()[key];
        return {float(window->xscroll), float(window->yscroll)};
    }

    bool GLFW::set_cursor_pos(std::string key, double xpos, double ypos) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot set cursor position."));

        auto ittr = Windows().find(key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot set cursor position."));

        auto window = &Windows()[key];
        window->xpos = xpos;
        window->ypos = ypos;
        return true;
    }

    std::array<float, 2> GLFW::get_cursor_pos(std::string key) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot get cursor position."));

        auto ittr = Windows().find(key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot get cursor position."));
        auto window = &Windows()[key];
        return {float(window->xpos), float(window->ypos)};
    }

    bool GLFW::set_button_data(std::string key, int button, int action, int mods) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot set button data."));
        auto ittr = Windows().find(key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot set button data"));
        
        if ((button >= 7) || (button < 0))
            throw std::runtime_error( std::string("Error: Button must be between 0 and 7."));

        auto window = &Windows()[key];
        window->buttons[button].action = action;
        window->buttons[button].mods = mods;
        return true;
    }

    int GLFW::get_button_action(std::string key, int button) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot get button action."));

        auto ittr = Windows().find(key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot get button action."));
        
        if ((button >= 7) || (button < 0))
            throw std::runtime_error( std::string("Error: Button must be between 0 and 7."));

        auto window = &Windows()[key];
        return window->buttons[button].action;
    }

    int GLFW::get_button_action_prev(std::string key, int button) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot get button action."));

        auto ittr = Windows().find(key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot get button action."));
        
        if ((button >= 7) || (button < 0))
            throw std::runtime_error( std::string("Error: Button must be between 0 and 7."));

        auto window = &Windows()[key];
        return window->buttonsPrev[button].action;
    }

    int GLFW::get_button_mods(std::string key, int button) {
        if (initialized == false) {
            throw std::runtime_error( std::string("Error: Uninitialized, cannot get button mods."));
            std::cout << "GLFW: "<<std::endl;
            return false;
        }
        auto ittr = Windows().find(key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot get button mods."));
        
        if ((button >= 7) || (button < 0))
            throw std::runtime_error( std::string("Error: Button must be between 0 and 7."));

        auto window = &Windows()[key];
        return window->buttons[button].mods;
    }

    int GLFW::get_button_mods_prev(std::string key, int button) {
        if (initialized == false) {
            throw std::runtime_error( std::string("Error: Uninitialized, cannot get button mods."));
            std::cout << "GLFW: "<<std::endl;
            return false;
        }
        auto ittr = Windows().find(key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot get button mods."));
        
        if ((button >= 7) || (button < 0))
            throw std::runtime_error( std::string("Error: Button must be between 0 and 7."));

        auto window = &Windows()[key];
        return window->buttonsPrev[button].mods;
    }

    bool GLFW::set_key_data(std::string window_key, int key, int scancode, int action, int mods) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot set key data."));

        auto ittr = Windows().find(window_key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot set key data."));
        
        if ((key >= 348) || (key < 0))
            throw std::runtime_error( std::string("Error: Button must be between 0 and 348."));

        auto window = &Windows()[window_key];
        window->keys[key].scancode = mods;
        window->keys[key].action = action;
        window->keys[key].mods = mods;
        return true;
    }

    int GLFW::get_key_action(std::string window_key, int key) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot get button mods."));

        auto ittr = Windows().find(window_key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot get button mods."));

        if ((key >= 348) || (key < 0))
            throw std::runtime_error( std::string("Error: Button must be between 0 and 348."));

        auto window = &Windows()[window_key];
        return window->keys[key].action;
    }

    int GLFW::get_key_action_prev(std::string window_key, int key) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot get button mods."));

        auto ittr = Windows().find(window_key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot get button mods."));

        if ((key >= 348) || (key < 0))
            throw std::runtime_error( std::string("Error: Button must be between 0 and 348."));

        auto window = &Windows()[window_key];
        return window->keysPrev[key].action;
    }

    int GLFW::get_key_scancode(std::string window_key, int key) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot get button mods."));

        auto ittr = Windows().find(window_key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot get button mods."));
        
        if ((key >= 348) || (key < 0))
            throw std::runtime_error( std::string("Error: Button must be between 0 and 348."));

        auto window = &Windows()[window_key];
        return window->keys[key].scancode;
    }

    int GLFW::get_key_mods(std::string window_key, int key) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot get key mods."));

        auto ittr = Windows().find(window_key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot get key mods."));
        
        if ((key >= 348) || (key < 0))
            throw std::runtime_error( std::string("Error: Button must be between 0 and 348."));

        auto window = &Windows()[window_key];
        return window->keys[key].mods;
    }

    int GLFW::get_key_mods_prev(std::string window_key, int key) {
        if (initialized == false)
            throw std::runtime_error( std::string("Error: Uninitialized, cannot get key mods."));

        auto ittr = Windows().find(window_key);
        if ( ittr == Windows().end() )
            throw std::runtime_error( std::string("Error: window does not exist, cannot get key mods."));
        
        if ((key >= 348) || (key < 0))
            throw std::runtime_error( std::string("Error: Button must be between 0 and 348."));

        auto window = &Windows()[window_key];
        return window->keysPrev[key].mods;
    }

    double GLFW::get_time() {
        return glfwGetTime();
    }

    int GLFW::get_key_code(std::string key) {
        std::transform(key.begin(), key.end(), key.begin(),
            [](unsigned char c){ return std::toupper(c); });    
        if (key.compare("SPACE") == 0) return GLFW_KEY_SPACE;
        else if (key.compare("APOSTROPHE") == 0) return GLFW_KEY_APOSTROPHE;
        else if (key.compare("COMMA") == 0) return GLFW_KEY_COMMA;
        else if (key.compare("MINUS") == 0) return GLFW_KEY_MINUS;
        else if (key.compare("PERIOD") == 0) return GLFW_KEY_PERIOD;
        else if (key.compare("SLASH") == 0) return GLFW_KEY_SLASH;
        else if (key.compare("0") == 0) return GLFW_KEY_0;
        else if (key.compare("1") == 0) return GLFW_KEY_1;
        else if (key.compare("2") == 0) return GLFW_KEY_2;
        else if (key.compare("3") == 0) return GLFW_KEY_3;
        else if (key.compare("4") == 0) return GLFW_KEY_4;
        else if (key.compare("5") == 0) return GLFW_KEY_5;
        else if (key.compare("6") == 0) return GLFW_KEY_6;
        else if (key.compare("7") == 0) return GLFW_KEY_7;
        else if (key.compare("8") == 0) return GLFW_KEY_8;
        else if (key.compare("9") == 0) return GLFW_KEY_9;
        else if (key.compare("SEMICOLON") == 0) return GLFW_KEY_SEMICOLON;
        else if (key.compare("EQUAL") == 0) return GLFW_KEY_EQUAL;
        else if (key.compare("A") == 0) return GLFW_KEY_A;
        else if (key.compare("B") == 0) return GLFW_KEY_B;
        else if (key.compare("C") == 0) return GLFW_KEY_C;
        else if (key.compare("D") == 0) return GLFW_KEY_D;
        else if (key.compare("E") == 0) return GLFW_KEY_E;
        else if (key.compare("F") == 0) return GLFW_KEY_F;
        else if (key.compare("G") == 0) return GLFW_KEY_G;
        else if (key.compare("H") == 0) return GLFW_KEY_H;
        else if (key.compare("I") == 0) return GLFW_KEY_I;
        else if (key.compare("J") == 0) return GLFW_KEY_J;
        else if (key.compare("K") == 0) return GLFW_KEY_K;
        else if (key.compare("L") == 0) return GLFW_KEY_L;
        else if (key.compare("M") == 0) return GLFW_KEY_M;
        else if (key.compare("N") == 0) return GLFW_KEY_N;
        else if (key.compare("O") == 0) return GLFW_KEY_O;
        else if (key.compare("P") == 0) return GLFW_KEY_P;
        else if (key.compare("Q") == 0) return GLFW_KEY_Q;
        else if (key.compare("R") == 0) return GLFW_KEY_R;
        else if (key.compare("S") == 0) return GLFW_KEY_S;
        else if (key.compare("T") == 0) return GLFW_KEY_T;
        else if (key.compare("U") == 0) return GLFW_KEY_U;
        else if (key.compare("V") == 0) return GLFW_KEY_V;
        else if (key.compare("W") == 0) return GLFW_KEY_W;
        else if (key.compare("X") == 0) return GLFW_KEY_X;
        else if (key.compare("Y") == 0) return GLFW_KEY_Y;
        else if (key.compare("Z") == 0) return GLFW_KEY_Z;
        else if (key.compare("LEFT_BRACKET") == 0) return GLFW_KEY_LEFT_BRACKET;
        else if (key.compare("[") == 0) return GLFW_KEY_LEFT_BRACKET;
        else if (key.compare("BACKSLASH") == 0) return GLFW_KEY_BACKSLASH;
        else if (key.compare("\\") == 0) return GLFW_KEY_BACKSLASH;
        else if (key.compare("RIGHT_BRACKET") == 0) return GLFW_KEY_RIGHT_BRACKET;
        else if (key.compare("]") == 0) return GLFW_KEY_RIGHT_BRACKET;
        else if (key.compare("GRAVE_ACCENT") == 0) return GLFW_KEY_GRAVE_ACCENT;
        else if (key.compare("`") == 0) return GLFW_KEY_GRAVE_ACCENT;
        else if (key.compare("WORLD_1") == 0) return GLFW_KEY_WORLD_1;
        else if (key.compare("WORLD_2") == 0) return GLFW_KEY_WORLD_2;
        else if (key.compare("ESCAPE") == 0) return GLFW_KEY_ESCAPE;
        else if (key.compare("ENTER") == 0) return GLFW_KEY_ENTER;
        else if (key.compare("TAB") == 0) return GLFW_KEY_TAB;
        else if (key.compare("BACKSPACE") == 0) return GLFW_KEY_BACKSPACE;
        else if (key.compare("INSERT") == 0) return GLFW_KEY_INSERT;
        else if (key.compare("DELETE") == 0) return GLFW_KEY_DELETE;
        else if (key.compare("RIGHT") == 0) return GLFW_KEY_RIGHT;
        else if (key.compare("LEFT") == 0) return GLFW_KEY_LEFT;
        else if (key.compare("DOWN") == 0) return GLFW_KEY_DOWN;
        else if (key.compare("UP") == 0) return GLFW_KEY_UP;
        else if (key.compare("PAGE_UP") == 0) return GLFW_KEY_PAGE_UP;
        else if (key.compare("PAGE_DOWN") == 0) return GLFW_KEY_PAGE_DOWN;
        else if (key.compare("HOME") == 0) return GLFW_KEY_HOME;
        else if (key.compare("END") == 0) return GLFW_KEY_END;
        else if (key.compare("CAPS_LOCK") == 0) return GLFW_KEY_CAPS_LOCK;
        else if (key.compare("SCROLL_LOCK") == 0) return GLFW_KEY_SCROLL_LOCK;
        else if (key.compare("NUM_LOCK") == 0) return GLFW_KEY_NUM_LOCK;
        else if (key.compare("PRINT_SCREEN") == 0) return GLFW_KEY_PRINT_SCREEN;
        else if (key.compare("PAUSE") == 0) return GLFW_KEY_PAUSE;
        else if (key.compare("F1") == 0) return GLFW_KEY_F1;
        else if (key.compare("F2") == 0) return GLFW_KEY_F;
        else if (key.compare("F3") == 0) return GLFW_KEY_F3;
        else if (key.compare("F4") == 0) return GLFW_KEY_F4;
        else if (key.compare("F5") == 0) return GLFW_KEY_F5;
        else if (key.compare("F6") == 0) return GLFW_KEY_F6;
        else if (key.compare("F7") == 0) return GLFW_KEY_F7;
        else if (key.compare("F8") == 0) return GLFW_KEY_F8;
        else if (key.compare("F9") == 0) return GLFW_KEY_F9;
        else if (key.compare("F10") == 0) return GLFW_KEY_F10;
        else if (key.compare("F11") == 0) return GLFW_KEY_F11;
        else if (key.compare("F12") == 0) return GLFW_KEY_F12;
        else if (key.compare("F13") == 0) return GLFW_KEY_F13;
        else if (key.compare("F14") == 0) return GLFW_KEY_F14;
        else if (key.compare("F15") == 0) return GLFW_KEY_F15;
        else if (key.compare("F16") == 0) return GLFW_KEY_F16;
        else if (key.compare("F17") == 0) return GLFW_KEY_F17;
        else if (key.compare("F18") == 0) return GLFW_KEY_F18;
        else if (key.compare("F19") == 0) return GLFW_KEY_F19;
        if (key.compare("F20") == 0) return GLFW_KEY_F20;
        else if (key.compare("F21") == 0) return GLFW_KEY_F21;
        else if (key.compare("F22") == 0) return GLFW_KEY_F22;
        else if (key.compare("F23") == 0) return GLFW_KEY_F23;
        else if (key.compare("F24") == 0) return GLFW_KEY_F24;
        else if (key.compare("F25") == 0) return GLFW_KEY_F25;
        else if (key.compare("KP_0") == 0) return GLFW_KEY_KP_0;
        else if (key.compare("KP_1") == 0) return GLFW_KEY_KP_1;
        else if (key.compare("KP_2") == 0) return GLFW_KEY_KP_2;
        else if (key.compare("KP_3") == 0) return GLFW_KEY_KP_3;
        else if (key.compare("KP_4") == 0) return GLFW_KEY_KP_4;
        else if (key.compare("KP_5") == 0) return GLFW_KEY_KP_5;
        else if (key.compare("KP_6") == 0) return GLFW_KEY_KP_6;
        else if (key.compare("KP_7") == 0) return GLFW_KEY_KP_7;
        else if (key.compare("KP_8") == 0) return GLFW_KEY_KP_8;
        else if (key.compare("KP_9") == 0) return GLFW_KEY_KP_9;
        else if (key.compare("KP_DECIMAL") == 0) return GLFW_KEY_KP_DECIMAL;
        else if (key.compare("KP_DIVIDE") == 0) return GLFW_KEY_KP_DIVIDE;
        else if (key.compare("KP_MULTIPLY") == 0) return GLFW_KEY_KP_MULTIPLY;
        else if (key.compare("KP_SUBTRACT") == 0) return GLFW_KEY_KP_SUBTRACT;
        else if (key.compare("KP_ADD") == 0) return GLFW_KEY_KP_ADD;
        else if (key.compare("KP_ENTER") == 0) return GLFW_KEY_KP_ENTER;
        else if (key.compare("KP_EQUAL") == 0) return GLFW_KEY_KP_EQUAL;
        else if (key.compare("LEFT_SHIFT") == 0) return GLFW_KEY_LEFT_SHIFT;
        else if (key.compare("LEFT_CONTROL") == 0) return GLFW_KEY_LEFT_CONTROL;
        else if (key.compare("LEFT_ALT") == 0) return GLFW_KEY_LEFT_ALT;
        else if (key.compare("LEFT_SUPER") == 0) return GLFW_KEY_LEFT_SUPER;
        else if (key.compare("RIGHT_SHIFT") == 0) return GLFW_KEY_RIGHT_SHIFT;
        else if (key.compare("RIGHT_CONTROL") == 0) return GLFW_KEY_RIGHT_CONTROL;
        else if (key.compare("RIGHT_ALT") == 0) return GLFW_KEY_RIGHT_ALT;
        else if (key.compare("RIGHT_SUPER") == 0) return GLFW_KEY_RIGHT_SUPER;
        else if (key.compare("MENU") == 0) return GLFW_KEY_MENU;
        else if (key.compare("LAST") == 0) return GLFW_KEY_LAST;

        return -1;
    }
}
