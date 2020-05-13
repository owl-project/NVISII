#pragma once
#include <visii/entity.h>
#include <visii/transform.h>
#include <visii/material.h>

namespace visii 
{

/**
   Initializes various backend systems required to render scene data.
*/
void Initialize()
{
    Entity::Initialize();
    Transform::Initialize();
    Material::Initialize();
}

};