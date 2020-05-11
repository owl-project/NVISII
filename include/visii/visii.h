#pragma once
#include <visii/entity.h>
#include <visii/transform.h>

namespace visii 
{

/**
   Initializes various backend systems required to render scene data.
*/
void Initialize()
{
    Entity::Initialize();
    Transform::Initialize();
}

};