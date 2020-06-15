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

/* -------- Features --------------*/
%include "exception.i"
%exception {
  try {
	$action
  } catch (const std::exception& e) {
	SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

/* STD Vectors */
%include "std_vector.i"
namespace std {
  %template(FloatVector) vector<float>;
}

/* -------- GLM Vector Math Library --------------*/
%feature("autodoc","2");
%include "glm.i"
%feature("autodoc", "");

%{
#include "visii/visii.h"

#include "visii/camera.h"
#include "visii/entity.h"
#include "visii/light.h"
#include "visii/texture.h"
#include "visii/transform.h"
#include "visii/material.h"
#include "visii/mesh.h"
%}

/* -------- Ignores --------------*/
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

%include "visii/visii.h"
%include "visii/utilities/static_factory.h"
%include "visii/camera.h"
%include "visii/entity.h"
%include "visii/light.h"
%include "visii/texture.h"
%include "visii/transform.h"
%include "visii/material.h"
%include "visii/mesh.h"



// Cleanup on exit
// %init %{
//     atexit(cleanup);
// %}

%init %{
  atexit(cleanup);
%}