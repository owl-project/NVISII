#pragma once
#include <visii/entity.h>

namespace visii 
{

/**
   Initializes various backend systems required to render scene data.
*/
void Initialize()
{
    Entity::Initialize();
}

};