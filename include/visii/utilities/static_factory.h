// ┌──────────────────────────────────────────────────────────────────────────┐
// | Copyright 2018-2020 Nathan Morrical                                      | 
// |                                                                          | 
// | Licensed under the Apache License, Version 2.0 (the "License");          | 
// | you may not use this file except in compliance with the License.         | 
// | You may obtain a copy of the License at                                  | 
// |                                                                          | 
// |     http://www.apache.org/licenses/LICENSE-2.0                           | 
// |                                                                          | 
// | Unless required by applicable law or agreed to in writing, software      | 
// | distributed under the License is distributed on an "AS IS" BASIS,        | 
// | WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. |
// | See the License for the specific language governing permissions and      | 
// | limitations under the License.                                           | 
// └──────────────────────────────────────────────────────────────────────────┘

#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#define GLM_FORCE_RIGHT_HANDED
#define SF_VERBOSE 0

#include <glm/glm.hpp>
#include <glm/common.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
// #include <glm/gtx/string_cast.hpp>

#include <iostream>
#include <string>
#include <set>
#include <map>
#include <vector>
#include <memory>
#include <typeindex>

#include <exception>
#include <mutex>
#include <thread>
#include <future>

class StaticFactory {
    public:

    /* items must be printable. */
    virtual std::string toString() = 0;

    /* returns the name (or key) of the current item. This can be used to look up this components id. */
    virtual std::string getName() { return name; };

    /* returns the id of the current item. */
    virtual int32_t getId() { return id; };
    
    /* Returns whether or not a key exists in the lookup table. */
    static bool doesItemExist(std::map<std::string, uint32_t> &lookupTable, std::string name)
    {
        auto it = lookupTable.find(name);
        return (it != lookupTable.end());
    }

    /* Returns the first index where an item of type T is uninitialized. */
    template<class T>
    static int32_t findAvailableID(T *items, uint32_t max_items) 
    {
        for (uint32_t i = 0; i < max_items; ++i)
            if (items[i].initialized == false)
                return (int32_t)i;
        return -1;
    }
    
    /* Reserves a location in items and adds an entry in the lookup table */
    template<class T>
    static T* create(std::shared_ptr<std::mutex> factory_mutex, std::string name, std::string type, std::map<std::string, uint32_t> &lookupTable, T* items, uint32_t maxItems, std::function<void(T*)> function = nullptr) 
    {
        auto mutex = factory_mutex.get();
        std::lock_guard<std::mutex> lock(*mutex);
        if (doesItemExist(lookupTable, name))
            throw std::runtime_error(std::string("Error: " + type + " \"" + name + "\" already exists."));

        int32_t id = findAvailableID(items, maxItems);

        if (id < 0) 
            throw std::runtime_error(std::string("Error: max " + type + " limit reached."));

        // TODO: make this only output if verbose
        #if SF_VERBOSE
        std::cout << "Adding " << type << " \"" << name << "\"" << std::endl;
        #endif
        items[id] = T(name, id);
        lookupTable[name] = id;

        // callback for creation before releasing mutex
        if (function != nullptr) function(&items[id]);

        return &items[id];
    }

    /* Retrieves an element with a lookup table indirection */
    template<class T>
    static T* get(std::shared_ptr<std::mutex> factory_mutex, std::string name, std::string type, std::map<std::string, uint32_t> &lookupTable, T* items, uint32_t maxItems) 
    {
        auto mutex = factory_mutex.get();
        std::lock_guard<std::mutex> lock(*mutex);
        if (doesItemExist(lookupTable, name)) {
            uint32_t id = lookupTable[name];
            if (!items[id].initialized) return nullptr;
            return &items[id];
        }

        throw std::runtime_error(std::string("Error: " + type + " \"" + name + "\" does not exist."));
        return nullptr;
    }

    /* Retrieves an element by ID directly */
    template<class T>
    static T* get(std::shared_ptr<std::mutex> factory_mutex, uint32_t id, std::string type, std::map<std::string, uint32_t> &lookupTable, T* items, uint32_t maxItems) 
    {
        auto mutex = factory_mutex.get();
        std::lock_guard<std::mutex> lock(*mutex);
        if (id >= maxItems) 
            throw std::runtime_error(std::string("Error: id greater than max " + type));

        else if (!items[id].initialized) return nullptr;
            //throw std::runtime_error(std::string("Error: " + type + " with id " + std::to_string(id) + " does not exist"));
         
        return &items[id];
    }

    /* Removes an element with a lookup table indirection, removing from both items and the lookup table */
    template<class T>
    static void remove(std::shared_ptr<std::mutex> factory_mutex, std::string name, std::string type, std::map<std::string, uint32_t> &lookupTable, T* items, uint32_t maxItems)
    {
        auto mutex = factory_mutex.get();
        std::lock_guard<std::mutex> lock(*mutex);
        if (!doesItemExist(lookupTable, name))
            throw std::runtime_error(std::string("Error: " + type + " \"" + name + "\" does not exist."));

        items[lookupTable[name]] = T();
        lookupTable.erase(name);
    }

    /* If it exists, removes an element with a lookup table indirection, removing from both items and the lookup table */
    template<class T>
    static void removeIfExists(std::shared_ptr<std::mutex> factory_mutex, std::string name, std::string type, std::map<std::string, uint32_t> &lookupTable, T* items, uint32_t maxItems)
    {
        auto mutex = factory_mutex.get();
        std::lock_guard<std::mutex> lock(*mutex);
        if (!doesItemExist(lookupTable, name)) return;
        items[lookupTable[name]] = T();
        lookupTable.erase(name);
    }

    /* Removes an element by ID directly, removing from both items and the lookup table */
    template<class T>
    static void remove(std::shared_ptr<std::mutex> factory_mutex, uint32_t id, std::string type, std::map<std::string, uint32_t> &lookupTable, T* items, uint32_t maxItems)
    {
        auto mutex = factory_mutex.get();
        std::lock_guard<std::mutex> lock(*mutex);
        if (id >= maxItems)
            throw std::runtime_error(std::string("Error: id greater than max " + type));

        if (!items[id].initialized)
            throw std::runtime_error(std::string("Error: " + type + " with id " + std::to_string(id) + " does not exist"));

        lookupTable.erase(items[id].name);
        items[id] = T();
    }

    protected:

    /* Inheriting factories should set this field to true when a component is considered initialied. */
    bool initialized = false;
    
    /* Inheriting factories should set these fields when a component is created. */
    std::string name = "";
    uint32_t id = -1;

    /* All items keep track of the entities which use them. */
    std::set<uint32_t> entities;
};
#undef SF_VERBOSE