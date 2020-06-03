# import sys, os
# os.add_dll_directory(os.path.join(os.getcwd(), '..', 'install'))
# sys.path.append(os.path.join(os.getcwd(), "..", "install"))

import visii
import numpy as np 
from PIL import Image 
import PIL
import randomcolor


SAMPLES_PER_PIXEL = 4000
# SAMPLES_PER_PIXEL = 100

# WIDTH = 1920 
# HEIGHT = 1080

WIDTH = 512 
HEIGHT = 512


#%%
visii.initialize_headless()
camera_entity = visii.entity.create(
    name="my_camera_entity",
    transform=visii.transform.create("my_camera_transform"),
    camera=visii.camera.create_perspective_from_fov(name = "my_camera", field_of_view = 0.785398, aspect = 1., near = .1))



visii.set_camera_entity(camera_entity)
camera_entity.get_transform().set_position(0, 0.0, -5.)
camera_entity.get_camera().use_perspective_from_fov(0.785398, 1.0, .01)
camera_entity.get_camera().set_view(
    visii.lookAt(
        visii.vec3(3,3,2),
        visii.vec3(0,0,.5),
        visii.vec3(0,0,1),
    )
)

floor = visii.entity.create(
    name="floor",
    mesh = visii.mesh.create_plane("floor"),
    transform = visii.transform.create("floor"),
    material = visii.material.create("floor")
)

#%%
mesh1 = visii.entity.create(
    name="mesh1",
    mesh = visii.mesh.create_sphere("sphere", 1, 128, 128),
    transform = visii.transform.create("mesh1"),
    material = visii.material.create("mesh1")
)

rcolor = randomcolor.RandomColor()
c = eval(str(rcolor.generate(luminosity='bright',format_='rgb')[0])[3:])
mesh1.get_material().set_base_color(
    c[0]/255.0,
    c[1]/255.0,
    c[2]/255.0
    )    

mesh1.get_material().set_roughness(np.random.uniform(0,1)) # default is 1  
mesh1.get_material().set_metallic(np.random.uniform(0,1))  # degault is 0     
mesh1.get_material().set_transmission(np.random.uniform(0,1))  # degault is 0     
mesh1.get_material().set_sheen(np.random.uniform(0,1))  # degault is 0     
mesh1.get_material().set_clearcoat(np.random.uniform(0,1))  # degault is 0     
mesh1.get_material().set_specular(np.random.uniform(0,1))  # degault is 0     
mesh1.get_material().set_anisotropic(np.random.uniform(0,1))  # degault is 0   

mesh1.get_transform().set_position(0.0, 0.0, 1.0)


areaLight1 = visii.entity.create(
    name="areaLight1",
    light = visii.light.create("areaLight1"),
    transform = visii.transform.create("areaLight1"),
    mesh = visii.mesh.create_teapotahedron("areaLight1"),
)
areaLight1.get_light().set_intensity(10000.)
areaLight1.get_transform().set_position(0, 0, 4)
areaLight1.get_light().set_temperature(4000)

visii.set_dome_light_intensity(0.1)
#%%

floor.get_transform().set_scale(1000)
floor.get_material().set_roughness(1.0)


#%%

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