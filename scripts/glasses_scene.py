import pybullet as p
import pybullet_data
import time

import visii
import numpy as np 
from PIL import Image 
import PIL
import randomcolor
from utils import * 
import argparse

parser = argparse.ArgumentParser()
   
parser.add_argument('--spp', 
                    default=30,
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
                    default='out_physics')
opt = parser.parse_args()


try:
    os.mkdir(opt.outf)
    print(f'created {opt.outf}/ folder')
except:
    print(f'{opt.outf}/ exists')




visii.initialize_headless()

if not opt.noise is True: 
    visii.enable_denoiser()


camera_entity = visii.entity.create(
    name="camera",
    transform=visii.transform.create("camera"),
    camera=visii.camera.create_perspective_from_fov(name = "camera", 
        field_of_view = 0.785398, 
        aspect = opt.width/float(opt.height),
        near = .1))



visii.set_camera_entity(camera_entity)
# camera_entity.get_transform().set_position(0, 0.0, -5.)
camera_entity.get_transform().set_position(visii.vec3(4,0,1.8))
camera_entity.get_transform().look_at(
    visii.vec3(0,0,0.5),
    visii.vec3(0,0,1),
    # visii.vec3(5,0,2),
)

print(camera_entity.get_transform().get_position())

light = visii.entity.create(
    name="light",
    mesh = visii.mesh.create_plane("light"),
    transform = visii.transform.create("light"),
    material = visii.material.create("light"),
    light = visii.light.create('light')
)
light.get_light().set_intensity(1000)
light.get_light().set_temperature(5000)
# light.get_transform().set_position(0,0,-0.1)
light.get_transform().set_position(visii.vec3(-1.5,1.5,2))
light.get_transform().set_scale(visii.vec3(0.1))
light.get_transform().set_rotation(visii.quat(0,0,1,0))

light.get_transform().look_at(
    visii.vec3(0,0,0),
    visii.vec3(0,0,1),
    )

light.get_transform().add_rotation(visii.quat(0,0,1,0))


# light2 = visii.entity.create(
#     name="light2",
#     mesh = visii.mesh.create_plane("light2"),
#     transform = visii.transform.create("light2"),
#     material = visii.material.create("light2"),
#     light = visii.light.create('light2')
# )
# light2.get_light().set_intensity(1000)
# light2.get_light().set_temperature(5000)
# # light2.get_transform().set_position(0,0,-0.1)
# light2.get_transform().set_position(visii.vec3(3,-1,4))
# light2.get_transform().set_scale(visii.vec3(0.4))
# light2.get_transform().set_rotation(visii.quat(0,0,1,0))
# light2.get_light().set_color(visii.vec3(1,0,0))

# light2.get_transform().look_at(
#     visii.vec3(0,0,0),
#     visii.vec3(0,0,1),
#     )


dome = visii.texture.create_from_image("dome", "textures/abandoned_tank_farm_01_1k.hdr")
visii.set_dome_light_texture(dome)
visii.set_dome_light_intensity(0.1)
# Physics init 
physicsClient = p.connect(p.DIRECT) # or p.GUI or p.DIRECT for non-graphical version
p.setGravity(0,0,-10)


floor = visii.entity.create(
    name="floor",
    mesh = visii.mesh.create_plane("floor"),
    transform = visii.transform.create("floor"),
    material = visii.material.create("floor")
)
# floor.get_transform().set_position(0,0,-0.1)
floor.get_transform().set_position(visii.vec3(0,0,0))
floor.get_transform().set_scale(visii.vec3(10))
# floor.get_material().set_roughness(1.0)

# perlin = visii.texture.create_from_image("perlin", "tex.png")
# floor.get_material().set_roughness_texture(perlin)

# random_material("floor")
floor.get_material().set_transmission(0)
floor.get_material().set_metallic(1.0)
floor.get_material().set_roughness(0)

floor.get_material().set_base_color(visii.vec3(1,0.3,1))

# cube_visii = add_random_obj(name='cube',obj_id=3) # force to create a cube

# LOAD SOME OBJECTS 


import glob 

objects_dict = {}

base_rot = visii.quat(0.7071,0.7071,0,0)*visii.quat(0.7071,0,0.7071,0)
# base_rot = visii.quat(1,0,0,0)

folders = glob.glob("models/*")


scale = 0.1
name='ikea'
print(f"loading {name}")
obj_entity = create_obj(
    name = name,
    path_obj = 'turbosquid/LE_V1_0131_Ikea_Glass_Pokal.obj',
    path_tex = None,
    scale = scale,
    )

obj = visii.entity.get(name)
obj.get_transform().set_rotation(base_rot)
obj_mat = obj.get_material()
# obj.get_material().set_roughness(0)
# obj.get_material().set_base_color(visii.vec3(1,1,1))
obj.get_material().set_base_color(visii.vec3(1,1,1))

obj_mat.set_metallic(0)  # should 0 or 1      
obj_mat.set_transmission(1)  # should 0 or 1      
obj_mat.set_transmission_roughness(0)  # should 0 or 1      
obj_mat.set_roughness(0) # default is 1  

obj_mat.set_sheen(0)  # degault is 0     
# obj_mat.set_clearcoat(1)  # degault is 0     
obj_mat.set_clearcoat(0)  # degault is 0     
obj_mat.set_specular(0)  # degault is 0     

obj_mat.set_anisotropic(0)  # degault is 0     

# obj = visii.import_obj("bottle",
#     "turbosquid/green_bottle/Beer_Bottle_Green_001.obj",
#     'turbosquid/green_bottle/',
#     visii.vec3(1,0,0), # translation here
#     visii.vec3(0.1), # scale here
#     visii.angleAxis(3.14 * .5, visii.vec3(1,0,0)) #rotation here)
# )



visii.render_to_png(
            width=int(opt.width), 
            height=int(opt.height), 
            samples_per_pixel=int(opt.spp),
            image_path=f"tmp.png"
    )

p.disconnect()
visii.cleanup()