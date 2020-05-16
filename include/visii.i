%module visii

/* -------- Python Version Check --------------*/
#if Python3_VERSION_MAJOR == 3
%pythonbegin %{_built_major_version = 3%}
#else
%pythonbegin %{_built_major_version = 2%}
#endif
#if Python3_VERSION_MINOR == 8
%pythonbegin %{_built_minor_version = 3%}
#elif Python3_VERSION_MINOR == 7
%pythonbegin %{_built_minor_version = 7%}
#elif Python3_VERSION_MINOR == 6
%pythonbegin %{_built_minor_version = 6%}
#endif
%pythonbegin %{
from sys import version_info as _import_version_info
if _import_version_info < (_built_major_version, _built_minor_version, 0):
    raise RuntimeError("This module was built for Python " + str(_built_major_version) + "." + str(_built_minor_version) 
        + " but current interpreter is Python " + str(_import_version_info[0]) + "." + str(_import_version_info[1]) )
%}

/* -------- Debug Build Check --------------*/
#ifdef Python3_DEBUG
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

/* -------- GLM Vector Math Library --------------*/
%feature("autodoc","2");
%include "glm-bindings/glm.i"
%feature("autodoc", "");

// %rename("Entity") "entity";
// %rename("Transform") "transform";
// %rename("Material") "material";
// %rename("Mesh") "mesh";

// %feature("doxygen:ignore:transferfull");
// %feature("doxygen:ignore:compileroptions", range="line");
// %feature("doxygen:ignore:forcpponly", range="end");
// %feature("doxygen:ignore:beginPythonOnly", range="end:endPythonOnly", contents="parse");

%{
#include "visii/visii.h"

#include "visii/camera.h"
#include "visii/entity.h"
#include "visii/transform.h"
#include "visii/material.h"
#include "visii/mesh.h"
%}

%rename("%(undercase)s",%$isfunction) "";
%rename("%(undercase)s",%$isclass) "";

%feature("kwargs") camera;
%feature("kwargs") entity;
%feature("kwargs") transform;
%feature("kwargs") material;
%feature("kwargs") mesh;

%include "visii/visii.h"
%include "visii/utilities/static_factory.h"
%include "visii/camera.h"
%include "visii/entity.h"
%include "visii/transform.h"
%include "visii/material.h"
%include "visii/mesh.h"

