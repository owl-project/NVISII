import visii
import noise
import random
import argparse
import numpy as np 
import math
import time
import subprocess
import os


opt = lambda: None
opt.spp = 128 
opt.width = 1000
opt.height = 1000 
opt.noise = False
opt.outf = '14_normal_map_outf'

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
    camera = visii.camera.create(
        name = "camera", 
        aspect = float(opt.width)/float(opt.height)
    )
)

camera.get_transform().look_at(
    at = (0,0,0.5),
    up = (0,0,1),
    eye = (0,2,2),
)
visii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

visii.set_dome_light_intensity(0)

# third light 
light_entity = visii.entity.create(
    name="light",
    mesh = visii.mesh.create_plane('light', flip_z = True),
    transform = visii.transform.create("light"),
)
light_entity.set_light(
    visii.light.create('light')
)
light_entity.get_light().set_intensity(3)
light_entity.get_light().set_falloff(0)
light_entity.get_light().set_temperature(4000)

light_entity.get_transform().set_scale((0.2, 0.2, 0.2))
light_entity.get_transform().set_position((1,0,2))
light_entity.get_transform().look_at(
    at = (0,0,0),
    up = (0,0,1),
)


# Lets set some objects in the scene
test_plane = visii.entity.create(
    name = "test_plane",
    mesh = visii.mesh.create_plane("test_plane"),
    transform = visii.transform.create("test_plane"),
    material = visii.material.create("test_plane")
)
test_plane.get_transform().set_scale((.5,.5,.5))
test_plane.get_transform().set_position((0,0,.5))

mat = visii.material.get("test_plane")
mat.set_metallic(0)
mat.set_roughness(1)
mat.set_specular(0)

# Normal maps must be made using linear = True.
test_normal_tex = visii.texture.create_from_file("test_normal_map",'../data/TestNormalMap.png', linear = True)
mat.set_normal_map_texture(test_normal_tex)

brick_plane = visii.entity.create(
    name = "brick_plane",
    mesh = visii.mesh.create_plane("brick_plane"),
    transform = visii.transform.create("brick_plane"),
    material = visii.material.create("brick_plane")
)
brick_plane.get_transform().set_scale((10.0,10.0,10.0))

mat = visii.material.get("brick_plane")
mat.set_metallic(0)
mat.set_roughness(1)
mat.set_specular(0)

# load an example brick texture 
color_tex = visii.texture.create_from_file("color",'content/Bricks051_2K_Color.jpg')
normal_tex = visii.texture.create_from_file("normal",'content/Bricks051_2K_Normal.jpg', linear = True)
rough_tex = visii.texture.create_from_file("rough",'content/Bricks051_2K_Roughness.jpg', linear = True)

color_tex.set_scale((.1,.1))
normal_tex.set_scale((.1,.1))
rough_tex.set_scale((.1,.1))

mat.set_base_color_texture(color_tex)
mat.set_normal_map_texture(normal_tex)
mat.set_roughness_texture(rough_tex)

# # # # # # # # # # # # # # # # # # # # # # # # #

for i in range(100):
    light_entity.get_transform().look_at(at = (0,0,0), up = (0,0,1), eye = (math.cos(math.pi * 2.0 * (i / 100.0)), math.sin(math.pi * 2.0 * (i / 100.0)),1))
    test_plane.get_transform().set_rotation(visii.angleAxis(-math.pi * 2.0 * (i / 100.0), (0,0,1)))
    # time.sleep(.1)
    visii.render_to_file(
        width=int(opt.width), 
        height=int(opt.height), 
        samples_per_pixel=int(opt.spp),
        file_path=f"{opt.outf}/{str(i).zfill(5)}.png"
    )

# let's clean up the GPU
visii.deinitialize()

subprocess.call(['ffmpeg', '-y', '-framerate', '24', '-i', r"%05d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '../output.mp4'], cwd=os.path.realpath(opt.outf))