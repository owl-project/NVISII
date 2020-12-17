%module visii

/* -------- Python Version Check --------------*/
#if Python_VERSION_MAJOR == 3
%pythonbegin %{_built_major_version = 3%}
#else
%pythonbegin %{_built_major_version = 2%}
#endif
#if Python_VERSION_MINOR == 8
%pythonbegin %{_built_minor_version = 8%}
#elif Python_VERSION_MINOR == 7
%pythonbegin %{_built_minor_version = 7%}
#elif Python_VERSION_MINOR == 6
%pythonbegin %{_built_minor_version = 6%}
#elif Python_VERSION_MINOR == 5
%pythonbegin %{_built_minor_version = 5%}
#endif
%pythonbegin %{
from sys import version_info as _import_version_info
if _import_version_info < (_built_major_version, _built_minor_version, 0):
    raise RuntimeError("This module was built for Python " + str(_built_major_version) + "." + str(_built_minor_version) 
        + " but current interpreter is Python " + str(_import_version_info[0]) + "." + str(_import_version_info[1]) )
%}

/* -------- Debug Build Check --------------*/
#ifdef Python_DEBUG
%pythonbegin %{
import sys
if not hasattr(sys, 'gettotalrefcount'):
   raise RuntimeError("This module was built in debug mode; however, the current interpreter was built in release mode.")
%}
#else
%pythonbegin %{
import sys
if hasattr(sys, 'gettotalrefcount'):
   raise RuntimeError("This module was built in release mode; however, the current interpreter was built in debug mode.")
%}
#endif

/* -------- Path Stuff --------------*/
%pythonbegin %{

import os, sys, platform, math
try:
    from . import version
    __version__ = version.__version__
except ImportError:
    pass

__this_dir__= os.path.dirname(os.path.abspath(__file__))

WIN32=platform.system()=="Windows" or platform.system()=="win32"
if WIN32:
	def AddSysPath(value):
		os.environ['PATH'] = value + os.pathsep + os.environ['PATH']
		sys.path.insert(0, value)
		if hasattr(os,'add_dll_directory'): 
			os.add_dll_directory(value) # this is needed for python 38  

	AddSysPath(__this_dir__)

else:
	sys.path.append(__this_dir__)

%}

%begin %{
#define SWIG_PYTHON_CAST_MODE
%}

/* -------- Features --------------*/
%include "exception.i"
%exception {
  try {
	$action
  } catch (const std::exception& e) {
	SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

// numpy stuff
%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}


%apply (float* INPLACE_ARRAY_FLAT, int DIM_FLAT) {(const float* data, uint32_t length)};


/* -------- GLM Vector Math Library --------------*/
%feature("autodoc","2");
%include "glm.i"
%feature("autodoc", "");

%{
#include <vector>
#include <array>
#include "visii/visii.h"
#include "visii/camera.h"
#include "visii/entity.h"
#include "visii/light.h"
#include "visii/texture.h"
#include "visii/transform.h"
#include "visii/material.h"
#include "visii/mesh.h"
%}

/* STD Vectors */
%include "std_array.i"
%include "std_vector.i"
namespace std {
  %template(Float3) array<float, 3>;
  %template(Float4) std::array<float, 4>;

  %template(FloatVector) vector<float>;
  %template(Float3Vector) vector<array<float, 3>>;
  %template(Float4Vector) vector<array<float, 4>>;
  %template(UINT32Vector) vector<uint32_t>;
  %template(StringVector) vector<string>;
  %template(EntityVector) vector<Entity*>;
  %template(TransformVector) vector<Transform*>;
  %template(MeshVector) vector<Mesh*>;
  %template(CameraVector) vector<Camera*>;
  %template(TextureVector) vector<Texture*>;
  %template(LightVector) vector<Light*>;
  %template(MaterialVector) vector<Material*>;
}

/* STD Maps */
%include "std_map.i"
namespace std {
  %template(StringToUINT32Map) map<string, uint32_t>;
}

/* -------- Ignores --------------*/
%ignore Entity::Entity();
%ignore Entity::Entity(std::string name, uint32_t id);
%ignore Entity::initializeFactory();
%ignore Entity::getFront();
%ignore Entity::getFrontStruct();
%ignore Entity::isFactoryInitialized();
%ignore Entity::updateComponents();
%ignore Entity::getStruct();
%ignore Entity::getEditMutex();
%ignore Entity::isDirty();
%ignore Entity::isClean();
%ignore Entity::markDirty();
%ignore Entity::markClean();

%ignore Transform::Transform();
%ignore Transform::Transform(std::string name, uint32_t id);
%ignore Transform::initializeFactory();
%ignore Transform::getFront();
%ignore Transform::getFrontStruct();
%ignore Transform::isFactoryInitialized();
%ignore Transform::updateComponents();
%ignore Transform::getStruct();
%ignore Transform::getEditMutex();
%ignore Transform::isDirty();
%ignore Transform::isClean();
%ignore Transform::markDirty();
%ignore Transform::markClean();

%ignore Material::Material();
%ignore Material::Material(std::string name, uint32_t id);
%ignore Material::initializeFactory();
%ignore Material::getFront();
%ignore Material::getFrontStruct();
%ignore Material::isFactoryInitialized();
%ignore Material::updateComponents();
%ignore Material::getStruct();
%ignore Material::getEditMutex();
%ignore Material::isDirty();
%ignore Material::isClean();
%ignore Material::markDirty();
%ignore Material::markClean();

%ignore Camera::Camera();
%ignore Camera::Camera(std::string name, uint32_t id);

%ignore Mesh::Mesh();
%ignore Mesh::Mesh(std::string name, uint32_t id);

%ignore Light::Light();
%ignore Light::Light(std::string name, uint32_t id);

%ignore Texture::Texture();
%ignore Texture::Texture(std::string name, uint32_t id);
%ignore Texture::~Texture();

/* -------- Renames --------------*/
%rename("%(undercase)s",%$isfunction) "";
%rename("%(undercase)s",%$isclass) "";

%feature("kwargs");
//  transform;
// // %feature("kwargs") camera;
// // %feature("kwargs") texture;
// // %feature("kwargs") entity;
// // %feature("kwargs") light;
// // %feature("kwargs") material;
// // %feature("kwargs") mesh;

%typemap(in) std::function<void()> (void *argp = 0, int res = 0) {
  if ($input == Py_None) {
    $1 = nullptr;
  }
  else if (!PyFunction_Check($input)) {
    PyErr_SetString(PyExc_ValueError, "in method '" "$symname" "', argument " "$argnum" " Expected a function");
    return NULL;
  }
  else {
    #if Python_VERSION_MAJOR==3
    PyObject* fc = PyObject_GetAttrString($input, "__code__");
    #else
    PyObject* fc = PyObject_GetAttrString($input, "func_code");
    #endif
    if (!fc) {
      #if Python_VERSION_MAJOR==3
      PyErr_SetString(PyExc_ValueError, "in method '" "$symname" "', argument " "$argnum" " function has no \"__code__\" member ");
      #else
      PyErr_SetString(PyExc_ValueError, "in method '" "$symname" "', argument " "$argnum" " function has no \"func_code\" member");
      #endif
      return NULL;
    }

    PyObject* ac = PyObject_GetAttrString(fc, "co_argcount");
    if(!ac) {
      PyErr_SetString(PyExc_ValueError, "in method '" "$symname" "', argument " "$argnum" " function has no \"co_argcount\" member");
      Py_DECREF(fc);
      return NULL;
    }

    // we now have the argument count, do something with this function
    const int count = PyInt_AsLong(ac);
    Py_DECREF(ac);
    Py_DECREF(fc);

    if (count != 0) {
      PyErr_SetString(PyExc_ValueError, "in method '" "$symname" "', argument " "$argnum" " function has an unexpected argument");
      return NULL;
    }
    $1 = [$input]() {
      PyGILState_STATE gstate;
      gstate = PyGILState_Ensure();
      PyEval_CallObject($input, NULL);
      PyGILState_Release(gstate);
    };
  }
}

%include "visii/visii.h"
%include "visii/utilities/static_factory.h"
%include "visii/camera.h"
%include "visii/entity.h"
%include "visii/light.h"
%include "visii/texture.h"
%include "visii/transform.h"
%include "visii/material.h"
%include "visii/mesh.h"

// void registerPreRenderCallback(std::function<void()> callback);
// %feature("director") CallBack;

// %inline %{
// struct CallBack {
//   virtual void handle() = 0;
//   virtual ~CallBack() {}
// };
// %}

// %{
// static CallBack *handler_ptr = NULL;
// static void handler_helper() {
//   // Make the call up to the target language when handler_ptr
//   // is an instance of a target language director class
//   handler_ptr->handle();
// }
// // If desired, handler_ptr above could be changed to a thread-local variable in order to make thread-safe
// %}

// %inline %{
// int binary_op_wrapper(int a, int b, BinaryOp *handler) {
//   handler_ptr = handler;
//   int result = binary_op(a, b, &handler_helper);
//   handler = NULL;
//   return result;
// }
// %}


// Cleanup on exit
%init %{
  atexit(deinitialize);
%}