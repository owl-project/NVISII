#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include <thread>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <future>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <map>
#include <utility>
#include <array>
#include <string.h>

#include <visii/utilities/singleton.h>

namespace Libraries {
    using namespace std;
    class GLFW : public Singleton
    {
    public:
        static GLFW* Get();
        bool initialize();
        GLFWwindow* create_window(string key, uint32_t width = 512, uint32_t height = 512, bool floating = true, bool resizable = true, bool decorated = true);
        bool make_context_current(std::string key);
        bool resize_window(std::string key, uint32_t width, uint32_t height);
        bool set_window_visibility(std::string key, bool visible);
        bool toggle_fullscreen(std::string key);
        bool set_window_pos(std::string key, uint32_t x, uint32_t y);
        bool destroy_window(string key);
        std::vector<std::string> get_window_keys();
        bool poll_events();
        bool wait_events();
        bool does_window_exist(string key);
        bool post_empty_event();
        bool should_close(string key);
        void swap_buffers(string key);
        std::string get_key_from_ptr(GLFWwindow* ptr);
        bool set_scroll(std::string key, double xoffset, double yoffset);
        std::array<float, 2> get_scroll_offset(std::string key);
        bool set_cursor_pos(std::string key, double xpos, double ypos);
        std::array<float, 2> get_cursor_pos(std::string key);
        bool set_button_data(std::string key, int button, int action, int mods);
        int get_button_action(std::string key, int button);
        int get_button_action_prev(std::string key, int button);
        int get_button_mods(std::string key, int button);
        int get_button_mods_prev(std::string key, int button);

        bool set_key_data(std::string window_key, int key, int scancode, int action, int mods);
        int get_key_scancode(std::string window_key, int key);
        int get_key_action(std::string window_key, int key);
        int get_key_action_prev(std::string window_key, int key);
        int get_key_mods(std::string window_key, int key);
        int get_key_mods_prev(std::string window_key, int key);

        bool set_should_close(std::string window_key, bool should_close);

        static int get_key_code(std::string key);
        GLFWwindow* get_ptr(std::string key);
        std::shared_ptr<std::mutex> get_mutex();
        double get_time();

        std::array<float, 2> get_size(std::string window_key);
        
        private:
        GLFW();
        ~GLFW();    

        struct Button {
            unsigned char action;
            unsigned char mods;
        };

        struct Key {
            int scancode;
            unsigned char action;
            unsigned char mods;
        };

        // mutex used to guarantee exclusive access to windows
        std::shared_ptr<std::mutex> window_mutex;
        
        struct Window {
            uint32_t current_image_index;
            GLFWwindow* ptr;
            std::array<float,2> extent;
            double xscroll = 0.f;
            double yscroll = 0.f;
            double xpos;
            double ypos;
            bool shouldClose = false;
             // This is a bit inefficient, but allows lookup by GLFW key
            Button buttons[8];
            Button buttonsPrev[8];
            Key keys[349];
            Key keysPrev[349];
        };

        static unordered_map<string, Window> &Windows();
    };
}
