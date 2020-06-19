# import sys, os
# os.add_dll_directory(os.path.join(os.getcwd(), '..', 'install'))
# sys.path.append(os.path.join(os.getcwd(), "..", "install"))

import visii
import numpy as np 
from PIL import Image 
import PIL
import randomcolor
from utils import * 
import argparse

parser = argparse.ArgumentParser()
   
parser.add_argument('--spp', 
                    default=10,
                    type=int)
parser.add_argument('--width', 
                    default=500,
                    type=int)
parser.add_argument('--height', 
                    default=500,
                    type=int)
parser.add_argument('--noise',
                    action='store_true',
                    default=False)
parser.add_argument('--outf', 
                    default="outf")

opt = parser.parse_args()



SAMPLES_PER_PIXEL = opt.spp
# SAMPLES_PER_PIXEL = 100

# WIDTH = 1920 
# HEIGHT = 1080

WIDTH = opt.width
HEIGHT = opt.height

visii.initialize_headless()


if not opt.noise is True: 
    visii.enable_denoiser()
#%%
camera_entity = visii.entity.create(
    name="my_camera_entity",
    transform=visii.transform.create("my_camera_transform"),
    camera=visii.camera.create_perspective_from_fov(name = "my_camera", 
        field_of_view = 0.785398, 
        aspect = opt.width/float(opt.height),
        near = .1))



visii.set_camera_entity(camera_entity)
# camera_entity.get_transform().set_position(0, 0.0, -5.)
camera_entity.get_camera().use_perspective_from_fov(0.785398, 1.0, .01)
camera_entity.get_camera().set_view(
    visii.lookAt(
        visii.vec3(4,0,0.5),
        visii.vec3(0,0,1),
        visii.vec3(0,0,1),
    )
)
visii.vec3(0,0,5), # camera_origin
visii.vec3(0,0,0), # look at (world coordinate)
visii.vec3(1,0,0), # up vector

# load the 2d texture

dome = visii.texture.create_from_image("dome", "textures/abandoned_tank_farm_01_1k.hdr")
visii.set_dome_light_texture(dome)


floor = visii.entity.create(
    name="floor",
    mesh = visii.mesh.create_plane("floor"),
    transform = visii.transform.create("floor"),
    material = visii.material.create("floor")
)

floor.get_transform().set_scale(1000)
floor.get_material().set_roughness(1.0)


# LIGHT


areaLight1 = visii.entity.create(
    name="areaLight1",
    light = visii.light.create("areaLight1"),
    transform = visii.transform.create("areaLight1"),
    mesh = visii.mesh.create_teapotahedron("areaLight1"),
)
areaLight1.get_light().set_intensity(10000.)
areaLight1.get_transform().set_position(0, 0, 5)
areaLight1.get_light().set_temperature(4000)



obj_mesh = visii.mesh.create_from_obj("obj", "models/Mayo/google_16k/textured.obj")
obj_texture = visii.texture.create_from_image('obj',"models/Mayo/google_16k/texture_map.png")

mesh1 = visii.entity.create(
    name="mesh1",
    # mesh = visii.mesh.create_sphere("mesh1", 1, 128, 128),
    mesh = soup_mesh,
    transform = visii.transform.create("mesh1"),
    material = visii.material.create("mesh1")
)

random_material('mesh1')

# mesh1.get_material().set_metallic(0)  # should 0 or 1      
# mesh1.get_material().set_transmission(random.uniform(0.9,1))  # should 0 or 1   
# mesh1.get_material().set_roughness(random.uniform(0,.1)) # default is 1

mesh1.get_material().set_metallic(0)  # should 0 or 1      
mesh1.get_material().set_transmission(0)  # should 0 or 1      
mesh1.get_material().set_roughness(random.uniform(0,1)) # default is 1  
mesh1.get_material().set_base_color_texture(soup_texture)

mesh1.get_transform().set_position(0.0, 0.0, 1.0)
mesh1.get_transform().set_scale(0.01)


print(mesh1.get_id())





#%%
# visii.enable_denoiser()

i_frame = 0 

visii.render_data_to_hdr(
                    width=int(opt.width), 
                    height=int(opt.height), 
                    frame=int(0),
                    bounce=int(0),
                    options="depth",
                    image_path=f"{opt.outf}/{str(i_frame).zfill(5)}_depth.hdr")

visii.render_data_to_hdr(
                    width=int(opt.width), 
                    height=int(opt.height), 
                    frame=int(0),
                    bounce=int(0),
                    options="normal",
                    image_path=f"{opt.outf}/{str(i_frame).zfill(5)}_normal.hdr")

visii.render_data_to_hdr(
                    width=int(opt.width), 
                    height=int(opt.height), 
                    frame=int(0),
                    bounce=int(0),
                    options="entity_id",
                    image_path=f"{opt.outf}/{str(i_frame).zfill(5)}_segmentation.hdr")

visii.render_data_to_hdr(
                    width=int(opt.width), 
                    height=int(opt.height), 
                    frame=int(0),
                    bounce=int(0),
                    options="position",
                    image_path=f"{opt.outf}/{str(i_frame).zfill(5)}_position.hdr")

visii.render_to_png(
                    width=int(opt.width), 
                    height=int(opt.height), 
                    samples_per_pixel = int(opt.spp),   
                    # frame=int(0),
                    # bounce=int(0),
                    # options="none",
                    image_path=f"{opt.outf}/{str(i_frame).zfill(5)}.png")

visii.cleanup()