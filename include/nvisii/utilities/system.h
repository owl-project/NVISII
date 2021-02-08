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

#include <thread>
#include <future>
#include <vector>

#include "include/utilities/singleton.h"

namespace Systems {
    using namespace std;
    class System : public Singleton 
    {
    public:
        bool running = false;
        virtual bool start() {return false;}
        virtual bool stop() {return false;}
    protected:
        System() {}
        ~System() {}
        System(const System&) = delete;
        System& operator=(const System&) = delete;
        System(System&&) = delete;
        System& operator=(System&&) = delete;
        
        std::promise<void> exitSignal;
        std::thread eventThread;
    };
}
