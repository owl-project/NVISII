#pragma once
#include <visii/entity.h>
#include <visii/transform.h>
#include <visii/material.h>
#include <visii/mesh.h>
#include <visii/camera.h>

/**
   Initializes various backend systems required to render scene data.
*/
void initializeInteractive();

/**
   Initializes various backend systems required to render scene data.
*/
void initializeHeadless();

/**
   Cleans up any allocated resources, closes windows and shuts down any running backend systems.
*/
void cleanup();