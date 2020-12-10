# ViSII - A VIrtual Scene Imaging Interface

![ViSII examples](https://i.imgur.com/A3MDFzy.png)

ViSII is a python-enabled ray tracing based renderer built on top of NVIDIA OptiX (C++/CUDA backend). 
The tool allows you to define complex scenes: 3d meshes, object materials, lights, loading textures, _etc._, and render 
them using ray tracing techniques. 
A scene can be described in many ways, you can use our simple mesh definitions, use multiple obj files, or 
load a complex scene defined in an obj file.
ViSII can be used to export metadata about the scene, _e.g._, object segmentation. 

[Documentation](https://owl-project.github.io/ViSII/).

<!--
This library provides a simple, primarily python-user targeted, interface to rendering images of a virtual scene. Its key cornerstones are:

a) a relatively simple, RTX/OptiX-accelerated path tracer, and
b) a interface (available in both python and C) for declaring a scene, doing basic modifications to it, and rendering images

 The two primary goals of this lirary are
a) ease of use (in particular, for non-expert users, and from languages like python), and
b) ease of deployment (ie, allowing headless rendering, minimal dependenies, etc).
To be clear: This is an academic and research renderer. There will be more sophisticated renderers out there, as well as faster ones, better ones, etc;
the goal of _this_ project is to offer something that's easy to get started with.
 -->
## Installing

We highly recommend that you use the pre-built wheels for python as follow: 
```
pip install visii
```
Also make sure your NVIDIA drivers are up to date (R435 or above).
If you need more information about how to install NVIDIA drivers on Ubuntu please consult
[this](https://ingowald.blog/installing-the-latest-nvidia-driver-cuda-and-optix-on-linux-ubuntu-18-04/).


## Getting Started 

We wrote different examples covering most of the functionalities of ViSII, [here](examples/). 
You can also find more extensive documentation [here](https://owl-project.github.io/ViSII/).

## Building 

Exact commands used to ViSII can be found in .github/manylinux.yml and .github/windows.yml.
More information on how to build will be added in the near future. 

<!-- Although we do not recommend building visii from scratch. Here are the rudimentary 
requirements: 
-->

## Docker

Here are the steps to build a docker image for ViSII. 

```
cd docker
sudo sh get_nvidia_libs.sh
```

Since the CUDA docker image includes limited libs, this script adds the missing one for ViSII to run. 
This could potentially cause problems if the docker image is deployed on a different system, 
please make sure the NVIDIA drivers match on all your systems. 

```
docker build . -t visii:07.20
```

You can run an example like follow, 
make sure you change `/PATH/TO/ViSII/` to your path to the root of this repo.   
```
docker run --gpus 1 -v /PATH/TO/ViSII/:/code visii:07.20 python examples/01.simple_scene.py
```
This will save a `tmp.png` in the root folder. 

## Citation

If you use this tool in your research project, please cite as follows:

    @misc{Morrical20visii,
    author = {Nathan Morrical and Jonathan Tremblay and Stan Birchfield and Ingo Wald},
    note= {\url{ https://github.com/owl-project/ViSII/ }},
    title = {{ViSII}: VIrtual Scene Imaging Interface},
    Year = 2020
    }

## Extra examples

[Falling teapots](https://imgur.com/Fzjg7ZQ)

[Falling objects dans une salle de bain](https://imgur.com/BqSKTO7)

[Random camera pose around objects](https://imgur.com/79eMgUv)

<!-- ## Code Structure

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

- same as before, but do simple modification of scene (ie, rotate it)

- same as before, but two scene (probably need way of "naming" objects when loading), with one rotating around the other

- same as before, but also render depth, and primID -->
