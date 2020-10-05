import sys, os, math, colorsys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys, os, math, colorsys
# os.add_dll_directory(os.path.join(os.getcwd(), '..', 'install'))
sys.path.append(os.path.join(os.getcwd(), "..", "install"))

import visii
import noise
import random
import argparse
import numpy as np 
import math
import time
import subprocess
import os

parser = argparse.ArgumentParser()

parser.add_argument('--spp', 
                    default=400,
                    type=int,
                    help = "number of sample per pixel, higher the more costly")
parser.add_argument('--width', 
                    default=500,
                    type=int,
                    help = 'image output width')
parser.add_argument('--height', 
                    default=500,
                    type=int,
                    help = 'image output height')
parser.add_argument('--noise',
                    action='store_true',
                    default=False,
                    help = "if added the output of the ray tracing is not sent to optix's denoiser")
parser.add_argument('--outf',
                    default='normal_map_outf',
                    help = 'folder to output the images')

opt = parser.parse_args()

# # # # # # # # # # # # # # # # # # # # # # # # #
if os.path.isdir(opt.outf):
    print(f'folder {opt.outf}/ exists')
else:
    os.mkdir(opt.outf)
    print(f'created folder {opt.outf}/')
    
# # # # # # # # # # # # # # # # # # # # # # # # #
visii.initialize(headless = False, verbose = True)

if not opt.noise is True: 
    visii.enable_denoiser()

camera = visii.entity.create(
    name = "camera",
    transform = visii.transform.create("camera"),
    camera = visii.camera.create_perspective_from_fov(
        name = "camera", 
        field_of_view = 0.785398, 
        aspect = float(opt.width)/float(opt.height)
    )
)

camera.get_transform().look_at(
    at = (0,0,0),
    up = (0,0,1),
    eye = (0,0,3),
)
visii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

visii.set_dome_light_intensity(0)

# third light 
obj_entity = visii.entity.create(
    name="light",
    mesh = visii.mesh.create_plane('light', flip_z = True),
    transform = visii.transform.create("light"),
)
obj_entity.set_light(
    visii.light.create('light')
)
obj_entity.get_light().set_intensity(10)

obj_entity.get_light().set_temperature(5000)

obj_entity.get_transform().set_scale((0.2, 0.2, 0.2))
obj_entity.get_transform().set_position((1,0,2))
obj_entity.get_transform().look_at(
    at = (0,0,0),
    up = (0,0,1),
)


# Lets set some objects in the scene
entity = visii.entity.create(
    name = "floor",
    mesh = visii.mesh.create_plane("mesh_floor"),
    transform = visii.transform.create("transform_floor"),
    material = visii.material.create("material_floor")
)


entity.get_transform().set_scale(visii.vec3(2))

mat = visii.material.get("material_floor")
mat.set_metallic(0)
mat.set_roughness(1)

# # # # # # # # # # # # # # # # # # # # # # # # #

# load the texture 
color_tex = visii.texture.create_from_image("color",'content/Bricks051_2K_Color.jpg')
normal_tex = visii.texture.create_from_image("normal",'content/Bricks051_2K_Normal.jpg', linear = True)
rough_tex = visii.texture.create_from_image("rough",'content/Bricks051_2K_Roughness.jpg', linear = True)

mat.set_base_color_texture(color_tex)
mat.set_normal_map_texture(normal_tex)
mat.set_roughness_texture(rough_tex)

# # # # # # # # # # # # # # # # # # # # # # # # #

for i in range(100):
    obj_entity.get_transform().look_at(at = (0,0,0), up = (0,0,1), eye = (math.sin(math.pi * 2.0 * (i / 100.0)), math.cos(math.pi * 2.0 * (i / 100.0)),1))
    entity.get_transform().set_rotation(visii.angleAxis(math.pi * 2.0 * (i / 100.0), (0,0,1)))
    # time.sleep(.1)
    visii.render_to_png(
        width=int(opt.width), 
        height=int(opt.height), 
        samples_per_pixel=int(opt.spp),
        image_path=f"{opt.outf}/{str(i).zfill(5)}.png"
    )

# let's clean up the GPU
visii.deinitialize()

subprocess.call(['ffmpeg', '-y', '-framerate', '30', '-i', r"%05d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '../output.mp4'], cwd=os.path.realpath(opt.outf))