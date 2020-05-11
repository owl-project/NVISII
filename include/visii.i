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

%include "externals/glm-bindings/glm.i"

