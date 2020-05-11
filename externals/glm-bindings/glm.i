
%{

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#define GLM_FORCE_RIGHT_HANDED

#include <stdexcept>
#include <string>
#include <sstream>
#include <stdint.h>

// included in math/ofVectorMath.h
#include <glm/glm.hpp>
#include <glm/fwd.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/functions.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/perpendicular.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/spline.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <glm/gtx/scalar_multiplication.hpp>
#include <glm/gtx/string_cast.hpp>

// extras
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/fast_square_root.hpp>
#include <glm/gtx/rotate_vector.hpp>
%}


// FMA isn't linking for some reason. ignoring for now.
%ignore glm::fma;

// ----- C++ -----


%include <std_except.i>
%include <std_string.i>
%include <stdint.i>

// expanded primitives
%typedef unsigned int std::size_t;

// ----- Bindings------

namespace glm {

#ifdef SWIGLUA
%rename(add) operator+;
%rename(sub) operator-;
%rename(mul) operator*;
%rename(div) operator/;
%rename(eq) operator==;
#endif

%typedef int length_t;

%include "./ivec2.i"
%include "./ivec3.i"
%include "./ivec4.i"
%include "./vec2.i"
%include "./vec3.i"
%include "./vec4.i"
%include "./u16vec2.i"
%include "./u16vec3.i"
%include "./u16vec4.i"
%include "./mat3.i"
%include "./mat4.i"
%include "./quat.i"
%include "./constants.i"
%include "./functions.i"
} // namespace


/* Representations */
%extend glm::vec2 {
    %feature("python:slot", "tp_repr", functype="reprfunc") __repr__;
    std::string __repr__() { return glm::to_string(*$self); }
}

%extend glm::vec3 {
    %feature("python:slot", "tp_repr", functype="reprfunc") __repr__;
    std::string __repr__() { return glm::to_string(*$self); }
}

%extend glm::vec4 {
    %feature("python:slot", "tp_repr", functype="reprfunc") __repr__;
    std::string __repr__() { return glm::to_string(*$self); }
}

%extend glm::ivec2 {
    %feature("python:slot", "tp_repr", functype="reprfunc") __repr__;
    std::string __repr__() { return glm::to_string(*$self); }
}

%extend glm::ivec3 {
    %feature("python:slot", "tp_repr", functype="reprfunc") __repr__;
    std::string __repr__() { return glm::to_string(*$self); }
}

%extend glm::ivec4 {
    %feature("python:slot", "tp_repr", functype="reprfunc") __repr__;
    std::string __repr__() { return glm::to_string(*$self); }
}

%extend glm::mat3 {
    %feature("python:slot", "tp_repr", functype="reprfunc") __repr__;
    std::string __repr__() { return glm::to_string(*$self); }
}

%extend glm::mat4 {
    %feature("python:slot", "tp_repr", functype="reprfunc") __repr__;
    std::string __repr__() { return glm::to_string(*$self); }
}

%extend glm::quat {
    %feature("python:slot", "tp_repr", functype="reprfunc") __repr__;
    std::string __repr__() { return glm::to_string(*$self); }
}

/* Vectors */
%include "std_vector.i"
namespace std {
    %template(Vec2Vector) vector<glm::vec2>;
    %template(Vec3Vector) vector<glm::vec3>;
    %template(Vec4Vector) vector<glm::vec4>;
    
    %template(IVec2Vector) vector<glm::ivec2>;
    %template(IVec3Vector) vector<glm::ivec3>;
    %template(IVec4Vector) vector<glm::ivec4>;

    %template(U16Vec2Vector) vector<glm::u16vec2>;
    %template(U16Vec3Vector) vector<glm::u16vec3>;
    %template(U16Vec4Vector) vector<glm::u16vec4>;

    %template(Vec2Vector2D) vector<vector<glm::vec2>>;
    %template(Vec3Vector2D) vector<vector<glm::vec3>>;
    %template(Vec4Vector2D) vector<vector<glm::vec4>>;

    %template(IVec2Vector2D) vector<vector<glm::ivec2>>;
    %template(IVec3Vector2D) vector<vector<glm::ivec3>>;
    %template(IVec4Vector2D) vector<vector<glm::ivec4>>;

    %template(U16Vec2Vector2D) vector<vector<glm::u16vec2>>;
    %template(U16Vec3Vector2D) vector<vector<glm::u16vec3>>;
    %template(U16Vec4Vector2D) vector<vector<glm::u16vec4>>;
};