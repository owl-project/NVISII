# ViSii examples

This folder contains different examples showing different capabilities for visii. 
The first example uses interactive mode, whereas the rest of the examples 
use headless mode and save local images (look for `tmp.png` in the `examples/` 
folder). 
We also provide baseline outputs for what visii should be producing. 
You can download them through `sh download_results.sh`. 

There is also a python `requirements.txt` for different packages needed to run all the examples. 
Some examples have minimal requirements, e.g., `visii` only whereas others use packages that 
makes the final task easier. 

Please note that if you have more than one GPU, you might run into problems if they 
are not the same. Please use only one GPU with the following command `CUDA_VISIBLE_DEVICES=0`. 
For example you could run: 
```
CUDA_VISIBLE_DEVICES=0 python 1.simple_scene.py
```


## 0.helloworld.py
This is simple script that opens up a window with gaussian noise displayed. 


## 1.simple_scene.py
This shows a sphere on top of a mirror floor. You should see how to set up a simple scene 
with visii, the light is provided by the dome. 


## 2.random_scene.py
This shows how to generate a randomized scene with different meshes and how to randomize 
the materials of the object. This is similar to how domain randomization look like. 


## 3.pybullet.py
This example creates a video in a folder, you are going to need `ffmpeg` to generate 
the final video, which is called `output.mp4` and it is located in the `examples/` folder.


## 4.load_obj_file.py
This script assumes you have downloaded the content for the demos using `sh download_content.sh`. 
You can also use your own obj file, see the arguments to the script. 


## 5.lights.py
This script presents how to add lights to your scenes. Lights in visii are meshes, 
so we show how to do different lights in this example. You can also check how the different 
materials react to each other, e.g., the cube is transmissive (it is transparent). 


## 6.textures.py
This script assumes you have downloaded the content for the demos using `sh download_content.sh`. 
Moreover the textures can be used to set specific material propreties, such as 
roughness (see `7.procedural_texture.py`). 


## 7.procedural_texture.py
This script shows how to use procedurally loading a texture and applying it to a roughness 
material property. The final picture looks like a mirror with spots on them. 


## 8.obj_scene_loader.py
This scene uses downloaded content, so make sure to run `sh download_content.sh` to get 
the obj scene. This scene is a bathroom showing how you can change specific material or objects
from a large obj scene.   


## 9.meta_data_exporting.py
This is a very simple scene showing different export that visii does naturally, e.g., normals. 


## 10.light_texture.py
This is the same scene to `5.lights.py` but we added textures to the lights. Check out 
how the color light bounces on the objects and the floor. 

## 11.instance_motion_blur.py
In this example, we demonstrate how to use motion blur for linear, angular, and scalar motion.
The end results are four dragons with different types of motion blur.

## 12.pybullet_motion_blur.py
Many physics frameworks expose linear and angular velocity. In this example, we 
demonstrate how to account for these linear and angular velocities through motion blur 
when generating an animation.

## 13.reprojection.py
A common technique for denoising ray traced images is to use "temporal reprojection". 
Here, we use diffuse motion vectors to reproject an image from frame to frame, accounting
for disocclusions.

## 14.normal_map.py
A simple script that loads a texture, normal and roughness map and apply them to a flat surface. 

## 15.camera_motion_car_blur.py
An interactive demo for controlling the camera similar to to a first person shooter control using `w,a,s,d` and `q` and `e` for up and down. Using the left click on the mouse you can rotate the camera. Here is an [example](https://imgur.com/VYda2UF) of the sort of motion you can generate.

## 16.create_mesh_from_data.py
This example shows how to create a mesh using an external package (in this case, `open3d`) and render it with visii. 
This is the exact same scene as `4.load_obj_file.py`. 

## 17.alpha_transparency.py
Although not entirely realistic, alpha transparency is a useful feature for making objects "see-through" without necessarily bending rays due to refraction. This is commonly used to handle alpha cutouts (eg. for leaves) but is also used frequently when "visualizing" scientific data. Here, we use PySide2 to demonstrate the difference between "alpha transparency" and "transmission.

## 18.materials.py
An example showing the different materials and how they impact the mesh. 

## 19.depth_map_to_point_cloud.py
This example shows how you can go from a depth map to a point cloud. Please consult [this](https://dsp.stackexchange.com/questions/26373/what-is-the-difference-between-a-range-image-and-a-depth-map) for more informantion between computer vision and graphics depth map. 

## 20.motion_vectors.py
An example to show you how to export motion vectors, they are very similar optical flow in computer vision, but it will break with reflective materials. The script outputs frame 0, moves an object, and export frame 1 as well as the motion vector between the two frames. 


## Notes
All these examples were developed and tested on Ubuntu 18.04 with cuda 11.0, NVIDIA drivers
450.36.06 and using a NVIDIA TITAN RTX . 


<!-- ## TODO
- exporting data is missing segmentation id for objects -->
