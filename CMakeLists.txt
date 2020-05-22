# NOTE: Run INSTALL target to generate and build python bindings

# ┌──────────────────────────────────────────────────────────────────┐
# │  Projects Settings                                               │
# └──────────────────────────────────────────────────────────────────┘
cmake_minimum_required (VERSION 3.13)
cmake_policy( VERSION 3.13...3.14 )
project(ViSII CXX C)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Use c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Force 64 bit builds
if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  MESSAGE(FATAL_ERROR "32 bit build detected. Please switch to a 64 bit build")
endif()

# Throw exception if Apple detected (Currently unsupported platform)
if(APPLE)
  MESSAGE(FATAL_ERROR "Apple platform detected, but not currently supported.")
endif(APPLE)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif(MSVC)

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
set(RPATHS "${CMAKE_INSTALL_PREFIX};${CMAKE_INSTALL_PREFIX}/visii;${CMAKE_INSTALL_PREFIX}/visii/lib;")
set(CMAKE_INSTALL_RPATH "${RPATHS}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_INSTALL_NAME_DIR "${RPATHS}")
# set(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_INSTALL_NAME_DIR @executable_path)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

# ┌──────────────────────────────────────────────────────────────────┐
# │  Build options                                                   │
# └──────────────────────────────────────────────────────────────────┘

# Build options go here... Things like "Build Tests", or "Generate documentation"...
option(NVCC_VERBOSE "verbose cuda -> ptx -> embedded build" ON)

# ┌──────────────────────────────────────────────────────────────────┐
# │  Definitions                                                     │
# └──────────────────────────────────────────────────────────────────┘

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-DNOMINMAX)
add_definitions(-DGENERATOR_USE_GLM)

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
  set(SWIG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/swig-4.0.1/linux64/share/swig/4.0.1/")
  set(SWIG_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/externals/swig-4.0.1/linux64/bin/swig")
endif()
find_package(SWIG 4.0.1)
include(${SWIG_USE_FILE})

# python
find_package(PythonLibs)
find_package(Python3 3.6 COMPONENTS Development REQUIRED)
if (UNIX AND NOT Python3_FOUND)
  message(FATAL_ERROR "Python 3 development libraries missing!"
    "You probably dont have python 3.6 or later development libraries installed on your system. It is possible to recover "
    "by running \"sudo apt install python3-dev\" (or equivalent on your unix distro)")
endif()
if (MSVC AND NOT Python3_FOUND)
  message(FATAL_ERROR "Python 3 development libraries missing!"
    "You probably dont have python 3.6 or later development libraries installed on your system. It is possible to recover "
    "by downloading python from https://www.python.org/downloads/. NOTE: for debug builds, make sure to check \"Download debug binaries\" "
    "in the \"Advanced Options\" tab of the Python installation wizard, and to run with python_d.exe")
endif()
include_directories(SYSTEM ${Python3_INCLUDE_DIRS})

# glm
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

# # stb
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/stb/)

# # tiny obj
# include_directories(SYSTEM external/tinyobj)

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
find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_INCLUDE})

# optix 7
set(OptiX_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/externals/optix-7.0.0/)
find_package(OptiX REQUIRED VERSION 7)
include_directories(${OptiX_INCLUDE})

# owl
set(OWL_BUILD_SAMPLES OFF CACHE STRING "" FORCE)
set(OWL_BUILD_ADVANCED_TESTS OFF CACHE STRING "" FORCE)
set(OWL_USE_TBB OFF CACHE STRING "" FORCE)
add_subdirectory(externals/owl)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/externals/owl/owl/include)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/externals/owl/owl/common/cmake)
include(configure_cuda)

# include(configure_optix)
# include(configure_build_type)

# add libraries to a list for linking
set (
  LIBRARIES
  ${PYTHON_LIBRARY}
  generator
  ${GLFW_LIBRARIES}
  glfw
  ${OPENGL_gl_LIBRARY}
  # ${TBB_LIBRARIES}
  owl
  )
  
if(NOT WIN32)
  set (
    LIBRARIES
    ${LIBRARIES}
    -latomic
  )
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
set(VERBOSE 1)
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
add_library(visii_lib SHARED ${SRC} ${HDR} ${ptxCode})
target_link_libraries(visii_lib PUBLIC ${LIBRARIES})
set_target_properties(visii_lib PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS true)
install(TARGETS visii_lib 
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    RENAME "visii_lib"
)

# ┌──────────────────────────────────────────────────────────────────┐
# │  Setup Targets                                                   │
# └──────────────────────────────────────────────────────────────────┘
set(VISII_INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/visii.i)
set_property(SOURCE ${VISII_INTERFACE} PROPERTY CPLUSPLUS ON)
set_property(SOURCE ${VISII_INTERFACE} PROPERTY USE_TARGET_INCLUDE_DIRECTORIES TRUE)
set_property(SOURCE ${VISII_INTERFACE} PROPERTY SWIG_FLAGS "-doxygen" "-py3") # -builtin seems to remove kwargs...
swig_add_library(visii TYPE SHARED LANGUAGE python SOURCES ${VISII_INTERFACE})
set_property(TARGET ${SWIG_MODULE_visii_REAL_NAME}  PROPERTY SWIG_DEPENDS ${HDR} ${GLM_HDR})
target_link_libraries(${SWIG_MODULE_visii_REAL_NAME} PUBLIC ${LIBRARIES} visii_lib)
set(SWIG_DEFS 
  Python3_VERSION_MAJOR=${Python3_VERSION_MAJOR} 
  Python3_VERSION_MINOR=${Python3_VERSION_MINOR})
if(CMAKE_BUILD_TYPE MATCHES Debug)
set(SWIG_DEFS ${SWIG_DEFS} Python3_DEBUG)
endif()
set_property(TARGET ${SWIG_MODULE_visii_REAL_NAME} PROPERTY SWIG_COMPILE_DEFINITIONS ${SWIG_DEFS})


# ┌──────────────────────────────────────────────────────────────────┐
# │  Install                                                         │
# └──────────────────────────────────────────────────────────────────┘

# Install
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_BINARY_DIR}/visii.py DESTINATION ${CMAKE_INSTALL_PREFIX})
# install(FILES ${CMAKE_BINARY_DIR}/glm.py DESTINATION ${CMAKE_INSTALL_PREFIX})

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    # GLM debug module
    if (WIN32) 
    install(FILES ${CMAKE_BINARY_DIR}/bin/_visii.pyd DESTINATION ${CMAKE_INSTALL_PREFIX} RENAME _visii_d.pyd)
    install(FILES ${CMAKE_BINARY_DIR}/bin/_visii.pdb DESTINATION ${CMAKE_INSTALL_PREFIX} RENAME _visii_d.pdb OPTIONAL)
    else()
    install(FILES ${CMAKE_BINARY_DIR}/lib/_visii.so DESTINATION ${CMAKE_INSTALL_PREFIX} RENAME _visii.so)
    endif()
ELSE(CMAKE_BUILD_TYPE MATCHES Debug)
    # GLM module
    if (WIN32) 
    install(FILES ${CMAKE_BINARY_DIR}/bin/_visii.pyd DESTINATION ${CMAKE_INSTALL_PREFIX})
    install(FILES ${CMAKE_BINARY_DIR}/bin/_visii.pdb DESTINATION ${CMAKE_INSTALL_PREFIX} OPTIONAL)
    else()
    install(FILES ${CMAKE_BINARY_DIR}/lib/_visii.so DESTINATION ${CMAKE_INSTALL_PREFIX})
    endif()
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

# # TBB DLL for windows
# if (WIN32) 
# install(FILES ${PROJECT_SOURCE_DIR}/external/tbb/vc14/tbb.dll DESTINATION ${CMAKE_INSTALL_PREFIX})
# install(FILES ${PROJECT_SOURCE_DIR}/external/tbb/vc14/tbb_debug.dll DESTINATION ${CMAKE_INSTALL_PREFIX})
# endif(WIN32)
