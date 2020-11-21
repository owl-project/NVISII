#pragma once

#include <owl/owl.h>
#include <glm/glm.hpp>

using namespace glm;

/* variables for the triangle mesh geometry */
struct TrianglesGeomData
{int placeholder;};

/* variables for the ray generation program */
struct RayGenData
{
    /*  index of the device that the current programs are running on;
        filled in automatically by owl as long as it is exported as a
        variable of type OWL_DEVICE */
    int deviceIndex;
    /*  number of devices that are available in the context; set by
        the app by querying owlGetDeviceCount and then assigning this
        via a int variable. */
    int deviceCount;
};

/* variables for the miss program */
struct MissProgData
{int placeholder;};
