# NOTE: Run INSTALL target to generate and build python bindings

# ┌──────────────────────────────────────────────────────────────────┐
# │  Projects Settings                                               │
# └──────────────────────────────────────────────────────────────────┘
cmake_minimum_required (VERSION 3.13)
cmake_policy( VERSION 3.13...3.14 )
project(ViSII CXX C)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Force 64 bit builds
if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  MESSAGE(FATAL_ERROR "32 bit build detected. Please switch to a 64 bit build")
endif()

# Throw exception if Apple detected (Currently unsupported platform)
if(APPLE)
  MESSAGE(FATAL_ERROR "Apple platform detected, but not currently supported.")
endif(APPLE)

# If on windows, use c++14 (oldest available by modern msvc)
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++14")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif(MSVC)

if(UNIX)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
endif(UNIX)

# Set the default install prefix path
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install/" CACHE PATH "..." FORCE)
endif()

# clear postfixes
set(CMAKE_DEBUG_POSTFIX "" CACHE STRING "" FORCE)
set(CMAKE_RELWITHDEBINFO_POSTFIX "" CACHE STRING "" FORCE)
set(CMAKE_MINSIZEREL_POSTFIX "" CACHE STRING "" FORCE)

# Force binaries to go to build folder, not subfolders
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin )
    set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib )
    set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib )
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )

# Default RPATH settings for *NIX distros
# set(RPATHS "./;${CMAKE_INSTALL_PREFIX};${CMAKE_INSTALL_PREFIX}/visii;${CMAKE_INSTALL_PREFIX}/visii/lib;")
# set(CMAKE_INSTALL_RPATH "${RPATHS}")
set(CMAKE_INSTALL_RPATH $ORIGIN)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_INSTALL_NAME_DIR "${RPATHS}")
set(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_INSTALL_NAME_DIR @executable_path)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

set(MY_DEFINITION 
    $<$<CONFIG:Debug>:definition_for_debug>
    $<$<CONFIG:RelWithDebInfo>:definition_for_rel_with_debug>
    $<$<CONFIG:Release>:definition_for_release> 
    $<$<CONFIG:MinSizeRel>:definition_for_tight_release>
    )

# ┌──────────────────────────────────────────────────────────────────┐
# │  Build options                                                   │
# └──────────────────────────────────────────────────────────────────┘

# Build options go here... Things like "Build Tests", or "Generate documentation"...
option(NVCC_VERBOSE "verbose cuda -> ptx -> embedded build" OFF)

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_C_COMPILER_ID MATCHES "Clang")
	# Enable c++11 and hide symbols which shouldn't be visible
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

# ┌──────────────────────────────────────────────────────────────────┐
# │  Definitions                                                     │
# └──────────────────────────────────────────────────────────────────┘

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-DNOMINMAX)
add_definitions(-DGENERATOR_USE_GLM)
add_definitions(-D_WIN64)

# ┌──────────────────────────────────────────────────────────────────┐
# │  External Dependencies                                           │
# └──────────────────────────────────────────────────────────────────┘

# Git (for auto submodule cloning)
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Update submodules as needed
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
      message(STATUS "Submodule update")
      execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                      RESULT_VARIABLE GIT_SUBMOD_RESULT)
      if(NOT GIT_SUBMOD_RESULT EQUAL "0")
          message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
      endif()
  endif()
endif()

# swig
set(SWIG_FOUND true)
set(SWIG_VERSION 4.0.1)
if(MSVC)
  set(SWIG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/swig-4.0.1/win64/Lib/")
  set(SWIG_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/externals/swig-4.0.1/win64/swig.exe")
  # include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/externals/swig-4.0.1/win64/Lib/)
  # include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/externals/swig-4.0.1/win64/Lib/python)
endif()
if(UNIX)
  if (NOT SWIG_DIR)
    set(SWIG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/swig-4.0.1/linux64/share/swig/4.0.1/")
  endif()
  if (NOT SWIG_EXECUTABLE)
    set(SWIG_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/externals/swig-4.0.1/linux64/bin/swig")
  endif()
endif()
find_package(SWIG 4.0.1)
include(${SWIG_USE_FILE})

# python
if (EXISTS ${Python_ROOT_DIR})
  set(Python_FIND_STRATEGY "LOCATION")
  message(WARN "${Python_ROOT_DIR}") 
endif()

set(PYTHON_VERSION "" CACHE STRING "Optional user-selected Python version")
if(PYTHON_VERSION)
  find_package(Python ${PYTHON_VERSION} EXACT COMPONENTS Development REQUIRED)
else()
  find_package(Python  COMPONENTS Development REQUIRED)
endif()

if (UNIX AND NOT Python_FOUND)
  message(FATAL_ERROR "Python development libraries missing!"
    "You probably dont have python 2.7 or later development libraries installed on your system. It is possible to recover "
    "by running \"sudo apt install python3-dev\" (or equivalent on your unix distro)")
endif()
if (MSVC AND NOT Python_FOUND)
  message(FATAL_ERROR "Python development libraries missing!"
    "You probably dont have python 2.7 or later development libraries installed on your system. It is possible to recover "
    "by downloading python from https://www.python.org/downloads/. NOTE: for debug builds, make sure to check \"Download debug binaries\" "
    "in the \"Advanced Options\" tab of the Python installation wizard, and to run with python_d.exe")
endif()
include_directories(SYSTEM ${Python_INCLUDE_DIRS})

# # glm
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/externals/glm)

# geometry generator
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/externals/generator)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/externals/generator/include)
set_property(TARGET generator PROPERTY FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/externals/generator")

# glad
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/externals/glad/include)
set(GLAD_SRC ${CMAKE_CURRENT_SOURCE_DIR}/externals/glad/src/glad.c)

# glfw
#    note: on linux, xorg-dev might be required
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/externals/glfw/CMakeLists.txt")
  message(FATAL_ERROR "The GLFW submodule directory is empty! "
    "You probably did not clone the project with --recursive. It is possible to recover "
    "by running \"git submodule update --init --recursive\"")
endif()
if (UNIX AND NOT APPLE)
  find_package(X11 REQUIRED)
  if (NOT X11_Xrandr_FOUND OR NOT X11_Xinerama_FOUND OR NOT X11_Xkb_FOUND OR NOT X11_Xcursor_FOUND)
    message(FATAL_ERROR "Some X11 development libraries are missing! "
      "You probably dont have xorg-dev installed on your system. It is possible to recover "
      "by running \"sudo apt install xorg-dev\" (or equivalent on your unix distro)")
  endif()  
endif()
set(GLFW_BUILD_DOCS OFF CACHE STRING "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE STRING "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE STRING "" FORCE)
set(GLFW_INSTALL        OFF CACHE STRING "" FORCE)
add_subdirectory(externals/glfw)
include_directories(SYSTEM externals/glfw/include)
set_property(TARGET glfw PROPERTY FOLDER "externals/GLFW3" )

# OpenGL
find_package(OpenGL)

# stb
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/externals/stb/)

# tiny things
include_directories(SYSTEM externals/tiny/)

# ImGUI
add_definitions(-D IMGUI_IMPL_OPENGL_LOADER_GLAD)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/externals/imgui/)
set(IMGUI_SRC 
  ${CMAKE_CURRENT_SOURCE_DIR}/externals/imgui/imgui.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/externals/imgui/imgui_demo.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/externals/imgui/imgui_draw.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/externals/imgui/imgui_widgets.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/externals/imgui/imgui_impl_glfw.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/externals/imgui/imgui_impl_opengl3.cpp
  # IMGUIZMO
  ${CMAKE_CURRENT_SOURCE_DIR}/externals/imgui/ImCurveEdit.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/externals/imgui/ImGradient.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/externals/imgui/ImGuizmo.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/externals/imgui/ImSequencer.cpp
  # TFN EDITOR
  # ${CMAKE_CURRENT_SOURCE_DIR}/externals/imgui/gl_core_4_5.c
  # ${CMAKE_CURRENT_SOURCE_DIR}/externals/imgui/transfer_function_widget.cpp
  # More addons
  # ${CMAKE_CURRENT_SOURCE_DIR}/externals/imgui/imguivariouscontrols.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/externals/imgui/imgui_rangeslider.cpp
)

# # tbb
# find_package(TBB REQUIRED)
# include_directories(${TBB_INCLUDE_DIR})

# cuda
if (${CUDA_TOOLKIT_ROOT_DIR})
message(INFO " CUDA_TOOLKIT_DIR set to ${CUDA_TOOLKIT_ROOT_DIR}")
endif()

find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_INCLUDE})
find_program(BIN2C bin2c 
  HINTS 
  ${CUDA_TOOLKIT_ROOT_DIR}/bin/
  /usr/local/cuda/bin)

# optix 7
set(OptiX_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/externals/optix-7.0.0/)
find_package(OptiX REQUIRED VERSION 7)
include_directories(${OptiX_INCLUDE})

# owl
set(SET_UP_CONFIGURATIONS_DONE 1)
set(OWL_BUILD_SAMPLES OFF CACHE STRING "" FORCE)
set(OWL_BUILD_ADVANCED_TESTS OFF CACHE STRING "" FORCE)
set(OWL_USE_TBB OFF CACHE STRING "" FORCE)
add_subdirectory(externals/owl)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/externals/owl/)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/externals/owl/owl/include)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/externals/owl/owl/common/cmake)
include(configure_cuda)

# add libraries to a list for linking
set (
  LIBRARIES
  generator
  ${GLFW_LIBRARIES}
  glfw
  ${OPENGL_gl_LIBRARY}
  # ${TBB_LIBRARIES}
  owl
  )

set (
  PY_LIBRARY
  
)

if (WIN32)
set(LIBRARIES ${LIBRARIES} ${Python_LIBRARIES})

# If debug lib for python exists, add it to the list.
if (${Python_LIBRARY_DEBUG})
set (
  LIBRARIES
  ${LIBRARIES}
  ${Python_LIBRARY_DEBUG}
)
endif()
if (${Python_LIBRARY_RELEASE})
set (
  LIBRARIES
  ${LIBRARIES}
  ${Python_LIBRARY_RELEASE}
)
endif()

# If on linux, request support for atomics
if(NOT WIN32)
  set (
    LIBRARIES
    ${LIBRARIES}
    -latomic
  )
endif()

endif()

# ┌──────────────────────────────────────────────────────────────────┐
# │  Definitions                                                     │
# └──────────────────────────────────────────────────────────────────┘
# message(WARN ${Python3_VERSION_MAJOR})
# add_definitions(-DPython3_VERSION_MAJOR=${Python3_VERSION_MAJOR})
# add_definitions(-DPython3_VERSION_MINOR=${Python3_VERSION_MINOR})
# add_definitions(-DPython3_VERSION_PATCH=${Python3_VERSION_PATCH})


# # ┌──────────────────────────────────────────────────────────────────┐
# # │  Macros                                                          │
# # └──────────────────────────────────────────────────────────────────┘

# For compiling cuda kernels and embedding them as ptx
if (NVCC_VERBOSE)
set (NVCC_VERBOSE_FLAG --verbose)
endif()

macro(cuda_compile_and_embed output_var cuda_file)
  set(var_name ${output_var})
  cuda_compile_ptx(ptx_files ${cuda_file} OPTIONS -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --keep ${NVCC_VERBOSE_FLAG})
  list(GET ptx_files 0 ptx_file)
  set(embedded_file ${ptx_file}_embedded.c)
  add_custom_command(
    OUTPUT ${embedded_file}
    COMMAND ${BIN2C} -c --padd 0 --type char --name ${var_name} ${ptx_file}  > ${embedded_file}
    DEPENDS ${ptx_file}
    COMMENT "compiling (and embedding ptx from) ${cuda_file}"
    )
  set(${output_var} ${embedded_file})
endmacro()

# ┌──────────────────────────────────────────────────────────────────┐
# │  Add source files                                                │
# └──────────────────────────────────────────────────────────────────┘
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/externals)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/src/externals/glm_bindings)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/src/externals/)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/src/visii/)

# defines a global ${HDR} variable containing a list of all project headers
add_subdirectory(include) 

# defines a global ${SRC} variable containing a list of all .cpp files
add_subdirectory(src)

set(SRC ${SRC} ${GLAD_SRC} ${IMGUI_SRC})
set(HDR ${HDR} ${Externals_HDR})

# ┌──────────────────────────────────────────────────────────────────┐
# │  CUDA PTX                                                        │
# └──────────────────────────────────────────────────────────────────┘
# OPTIONS -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
cuda_compile_and_embed(ptxCode ${SRC_CU})

# ┌──────────────────────────────────────────────────────────────────┐
# │  ViSII Library                                                   │
# └──────────────────────────────────────────────────────────────────┘
cuda_add_library(visii_lib SHARED ${SRC} ${HDR} ${ptxCode} OPTIONS -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored)
target_link_libraries(visii_lib ${LIBRARIES})
set_target_properties(visii_lib PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS true)
install(TARGETS visii_lib 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/visii/
    RENAME "visii_lib"
)

# ┌──────────────────────────────────────────────────────────────────┐
# │  Setup Targets                                                   │
# └──────────────────────────────────────────────────────────────────┘
set(VISII_INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/visii.i)
set_property(SOURCE ${VISII_INTERFACE} PROPERTY CPLUSPLUS ON)
set_property(SOURCE ${VISII_INTERFACE} PROPERTY USE_TARGET_INCLUDE_DIRECTORIES TRUE)
set_property(SOURCE ${VISII_INTERFACE} PROPERTY SWIG_FLAGS "-doxygen") # -builtin seems to remove kwargs...
swig_add_library(visii TYPE SHARED LANGUAGE python SOURCES ${VISII_INTERFACE})
if (WIN32) 
set_property(TARGET visii PROPERTY SWIG_GENERATED_COMPILE_OPTIONS -bigobj)
endif()
set_property(TARGET ${SWIG_MODULE_visii_REAL_NAME} PROPERTY SWIG_DEPENDS ${HDR} ${GLM_HDR})
target_link_libraries(${SWIG_MODULE_visii_REAL_NAME} PUBLIC ${LIBRARIES} visii_lib)
set(SWIG_DEFS 
  Python_VERSION_MAJOR=${Python_VERSION_MAJOR} 
  Python_VERSION_MINOR=${Python_VERSION_MINOR})
set(SWIG_DEFS ${SWIG_DEFS} $<$<CONFIG:Debug>:Python_DEBUG>)
set_property(TARGET ${SWIG_MODULE_visii_REAL_NAME} PROPERTY SWIG_COMPILE_DEFINITIONS ${SWIG_DEFS})
if(UNIX)
set_target_properties(${SWIG_MODULE_visii_REAL_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

# ┌──────────────────────────────────────────────────────────────────┐
# │  Install                                                         │
# └──────────────────────────────────────────────────────────────────┘

# Install
install(FILES ${CMAKE_BINARY_DIR}/visii.py DESTINATION ${CMAKE_INSTALL_PREFIX}/visii/ RENAME __init__.py)
install(FILES ${CMAKE_BINARY_DIR}/bin/_visii.pyd DESTINATION ${CMAKE_INSTALL_PREFIX}/visii/ OPTIONAL)
install(FILES ${CMAKE_BINARY_DIR}/lib/_visii.so DESTINATION ${CMAKE_INSTALL_PREFIX}/visii/ OPTIONAL)
install(FILES ${CMAKE_BINARY_DIR}/bin/_visii.pyd CONFIGURATIONS Debug DESTINATION ${CMAKE_INSTALL_PREFIX}/visii/ OPTIONAL RENAME _visii_d.pyd)
install(FILES ${CMAKE_BINARY_DIR}/bin/_visii.pdb CONFIGURATIONS Debug DESTINATION ${CMAKE_INSTALL_PREFIX}/visii/ OPTIONAL RENAME _visii_d.pdb)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/setup.py DESTINATION ${CMAKE_INSTALL_PREFIX})

# # TBB DLL for windows
# if (WIN32) 
# install(FILES ${PROJECT_SOURCE_DIR}/external/tbb/vc14/tbb.dll DESTINATION ${CMAKE_INSTALL_PREFIX})
# install(FILES ${PROJECT_SOURCE_DIR}/external/tbb/vc14/tbb_debug.dll DESTINATION ${CMAKE_INSTALL_PREFIX})
# endif(WIN32)
 