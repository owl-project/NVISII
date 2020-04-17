# ViSII - A VIrtual Scene Imaging Interface

This library provides a simple, primarily python-user targetted, interface to rendering images of a virtual scene. Its key cornerstones are:

a) a relatively simple, RTX/OptiX-accelerated path tracer, and
b) a interface (available in both python and C) for declaring a scene, doing basic modifications to it, and rendering images

The two primary goals of this lirary are
a) ease of use (in particular, for non-expert users, and from languages like python), and
b) ease of deployment (ie, allowing headless rendering, minimal dependenies, etc).
To be clear: There will be more sophisitcated renderers out there, as well as faster ones, better ones, etc;
the goal of _this_ project is to offer something that's easy to get started with.

## Code Structure

- submodules/ : external git submodule dependencies to build this
- visii/ : the (static) library that provides the renderer
    - visii/scene/ : code that maintains the visii "scene graph"
    - visii/render/ : the actual renderer(s) provided in this library
- cAPI/ : a extern "C" shared library/DLL interface for this library
- python/ : python interface for this library
- (?) tools/ : importer tools, as required for samples

## Building

todo

## Samples

todo: need (at least) the following samples

- load an OBJ file, declare camera and light, render an image, save as ppm

- same as before, but do simple omdification of scene (ie, rotate it)

- same as before, but two scene (probably need way of "naming" objects when loading), with one rotating around the other

- same as before, but also render depth, and primID
