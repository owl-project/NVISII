#pragma once

#include <owl/owl.h>
#include <glm/glm.hpp>

using namespace glm;

/* variables for the triangle mesh geometry */
struct TrianglesGeomData
{
  /*! base color we use for the entire mesh */
  vec4 *colors = nullptr;
  /*! array/buffer of vertex positions */
  vec4 *vertex = nullptr;
  /*! array/buffer of vertex positions */
  vec4 *normals = nullptr;
  /*! array/buffer of vertex positions */
  vec2 *texcoords = nullptr;
  /*! array/buffer of vertex indices */
  ivec3 *index = nullptr;
};

/* variables for the ray generation program */
struct RayGenData
{int placeholder;};

/* variables for the miss program */
struct MissProgData
{int placeholder;};
