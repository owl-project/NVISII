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
        visii.vec3(6,0,1),
        visii.vec3(0,0,2),
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

#%%
mesh1 = visii.entity.create(
    name="mesh1",
    mesh = visii.mesh.create_sphere("mesh1", 1, 128, 128),
    transform = visii.transform.create("mesh1"),
    material = visii.material.create("mesh1")
)

random_material('mesh1')

mesh1.get_material().set_metallic(0)  # should 0 or 1      
mesh1.get_material().set_transmission(random.uniform(0.9,1))  # should 0 or 1   
mesh1.get_material().set_roughness(random.uniform(0,.1)) # default is 1




mesh1.get_transform().set_position(0.0, 0.0, 1.0)


areaLight1 = visii.entity.create(
    name="areaLight1",
    light = visii.light.create("areaLight1"),
    transform = visii.transform.create("areaLight1"),
    mesh = visii.mesh.create_teapotahedron("areaLight1"),
)
areaLight1.get_light().set_intensity(10000.)
areaLight1.get_transform().set_position(0, 0, 5)
areaLight1.get_light().set_temperature(4000)

visii.set_dome_light_intensity(0.1)
#%%

floor.get_transform().set_scale(1000)
floor.get_material().set_roughness(1.0)


#%%
# visii.enable_denoiser()

# Read and save the image 
x = visii.render(width=WIDTH, height=HEIGHT, samples_per_pixel=SAMPLES_PER_PIXEL)
# x = np.array(x).reshape(WIDTH,HEIGHT,4)
x = np.array(x).reshape(HEIGHT,WIDTH,4)

# make sure the image is clamped 
x[x>1.0] = 1.0
x[x<0] = 0

img = Image.fromarray((x*255).astype(np.uint8)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
img.save("tmp.png")

visii.cleanup()