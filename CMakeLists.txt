# NOTE: Run INSTALL target to generate and build python bindings

# ┌──────────────────────────────────────────────────────────────────┐
# │  Projects Settings                                               │
# └──────────────────────────────────────────────────────────────────┘
cmake_minimum_required (VERSION 3.10)
cmake_policy( VERSION 3.10...3.14 )
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

# ┌──────────────────────────────────────────────────────────────────┐
# │  Definitions                                                     │
# └──────────────────────────────────────────────────────────────────┘

# Global compiler definitions go here... 

# For MSVC
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

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
  set(SWIG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/swig-4.0.1/linux64/bin/")
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
    "in the \"Advanced Options\" tab of the Python installation wizard (debug build of python also required).")
endif()
include_directories(SYSTEM ${Python3_INCLUDE_DIRS})

# glm
include_directories(SYSTEM externals/glm)

# # glad
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include)
# set(GLAD_SRC ${CMAKE_CURRENT_SOURCE_DIR}/external/glad/src/glad.c)

# # glfw
# #    note: on linux, xorg-dev might be required
# if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/glfw/CMakeLists.txt")
#   message(FATAL_ERROR "The GLFW submodule directory is empty! "
#     "You probably did not clone the project with --recursive. It is possible to recover "
#     "by running \"git submodule update --init --recursive\"")
# endif()
# if (UNIX AND NOT APPLE)
#   find_package(X11 REQUIRED)
#   if (NOT X11_Xrandr_FOUND OR NOT X11_Xinerama_FOUND OR NOT X11_Xkb_FOUND OR NOT X11_Xcursor_FOUND)
#     message(FATAL_ERROR "Some X11 development libraries are missing! "
#       "You probably dont have xorg-dev installed on your system. It is possible to recover "
#       "by running \"sudo apt install xorg-dev\" (or equivalent on your unix distro)")
#   endif()  
# endif()
# set(GLFW_BUILD_DOCS OFF CACHE STRING "" FORCE)
# set(GLFW_BUILD_EXAMPLES OFF CACHE STRING "" FORCE)
# set(GLFW_BUILD_TESTS    OFF CACHE STRING "" FORCE)
# set(GLFW_INSTALL        OFF CACHE STRING "" FORCE)
# # set(GLFW_VULKAN_STATIC  ON CACHE STRING "" FORCE)
# add_subdirectory(external/glfw)
# include_directories(SYSTEM external/glfw/include)
# set_property(TARGET glfw PROPERTY FOLDER "external/GLFW3" )

# # OpenGL
# find_package(OpenGL)

# # stb
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/stb/)

# # tiny obj
# include_directories(SYSTEM external/tinyobj)

# # ImGUI
# add_definitions(-D IMGUI_IMPL_OPENGL_LOADER_GLAD)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/)
# set(IMGUI_SRC 
#   ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_demo.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_draw.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_widgets.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_impl_glfw.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_impl_opengl3.cpp
#   # IMGUIZMO
#   ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/ImCurveEdit.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/ImGradient.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/ImGuizmo.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/ImSequencer.cpp
#   # TFN EDITOR
#   # ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/gl_core_4_5.c
#   ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/transfer_function_widget.cpp
#   # More addons
#   ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imguivariouscontrols.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_rangeslider.cpp
#   )

# # tbb
# find_package(TBB REQUIRED)
# include_directories(${TBB_INCLUDE_DIR})

# # owl
# set(OWL_BUILD_SAMPLES OFF CACHE STRING "" FORCE)
# set(OWL_BUILD_TESTS OFF CACHE STRING "" FORCE)
# add_subdirectory(external/owl)
# include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/external/owl/owl/include)
# list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external/owl/owl/common/cmake)
# include(configure_cuda)
# include(configure_optix)
# # include(configure_build_type)

# add libraries to a list for linking
set (
  LIBRARIES
  ${PYTHON_LIBRARY}
  # ${GLFW_LIBRARIES}
  # ${OPENGL_gl_LIBRARY}
  # ${TBB_LIBRARIES}
  # owl
  )
  
if(NOT WIN32)
  set (
    LIBRARIES
    ${LIBRARIES}
    -latomic
  )
endif()

# # ┌──────────────────────────────────────────────────────────────────┐
# # │  Macros                                                          │
# # └──────────────────────────────────────────────────────────────────┘

# # For compiling cuda kernels and embedding them as ptx
# macro(cuda_compile_and_embed output_var cuda_file)
#   set(var_name ${output_var})
#   cuda_compile_ptx(ptx_files ${cuda_file} OPTIONS -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored)
#   list(GET ptx_files 0 ptx_file)
#   set(embedded_file ${ptx_file}_embedded.c)
#   add_custom_command(
#     OUTPUT ${embedded_file}
#     COMMAND ${BIN2C} -c --padd 0 --type char --name ${var_name} ${ptx_file}  > ${embedded_file}
#     DEPENDS ${ptx_file}
#     COMMENT "compiling (and embedding ptx from) ${cuda_file}"
#     )
#   set(${output_var} ${embedded_file})
# endmacro()

# ┌──────────────────────────────────────────────────────────────────┐
# │  Add source files                                                │
# └──────────────────────────────────────────────────────────────────┘
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR})
# add_subdirectory(RTXBigUMesh)


set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/include/visii.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/include/visii.i PROPERTY USE_TARGET_INCLUDE_DIRECTORIES TRUE)
swig_add_library(visii TYPE SHARED LANGUAGE python SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/include/visii.i)
set_property(TARGET ${SWIG_MODULE_visii_REAL_NAME}  PROPERTY SWIG_DEPENDS ${GLM_HDR})
set_property(TARGET ${SWIG_MODULE_visii_REAL_NAME} PROPERTY FOLDER "swig_libraries/visii")
target_link_libraries(${SWIG_MODULE_visii_REAL_NAME} PUBLIC ${LIBRARIES})
# target_link_libraries(${SWIG_MODULE_visii_REAL_NAME} PUBLIC ${LIBRARIES} RTXBigUMeshLib)

# ┌──────────────────────────────────────────────────────────────────┐
# │  Misc Install                                                    │
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
