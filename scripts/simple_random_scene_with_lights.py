import sys, os
os.add_dll_directory(os.path.join(os.getcwd(), '..', 'install'))
sys.path.append(os.path.join(os.getcwd(), "..", "install"))

import visii 
import numpy as np 
from PIL import Image 
import PIL
import time 
from pyquaternion import Quaternion
import randomcolor


NB_OBJS = 10000
NB_LIGHTS = 20

SAMPLES_PER_PIXEL = 256

# WIDTH = 1920 
# HEIGHT = 1080

WIDTH = 1000 
HEIGHT = 500


visii.initialize_headless()

# time to initialize this is a bug

# Create a camera
camera_entity = visii.entity.create(
    name="my_camera",
    transform=visii.transform.create("my_camera"),
    camera=visii.camera.create_perspective_from_fov(name = "my_camera",
        field_of_view = 0.785398,
        # aspect = 1.,
        aspect = WIDTH/HEIGHT,
        near = .1
        )
    )

# This is to set the camera internal parameters
# camera_entity.get_camera().set_aperture_diameter(20)
# camera_entity.get_camera().set_focal_distance(3.5)

# Change the dome light intensity
visii.set_dome_light_intensity(0.1)

# set the view camera transform
camera_entity.get_camera().set_view(
    visii.lookAt(
        visii.vec3(0,0,5), # camera_origin
        visii.vec3(0,0,0), # look at (world coordinate)
        visii.vec3(1,0,0), # up vector
    )
)

# set the camera
visii.set_camera_entity(camera_entity)
meshes = [] 

rcolor = randomcolor.RandomColor()

def add_random_light(name = 'name'):
    global rcolor
    obj= visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        mesh = visii.mesh.create_sphere(name),
        light = visii.light.create(name)
    )
    obj.get_transform().set_scale(0.5)


    obj.get_light().set_intensity(1000000000000000)
    obj.get_light().set_temperature(5000)

    # obj.get_transform().set_position(
    #     np.random.uniform(-1,1),
    #     np.random.uniform(-1,1),
    #     np.random.uniform(6,7)
    #     )
    # obj.get_transform().set_position(
    #     0,
    #     0,
    #     2.5
    #     )

    obj.get_transform().set_position(
        np.random.uniform(-5,5),
        np.random.uniform(-5,5),
        np.random.uniform(-10,3)
        )
def add_random_obj(name = "name"):
    global rcolor
    obj= visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        material = visii.material.create(name),
    )

    obj_id = np.random.randint(0,16)

    mesh = None
    if obj_id == 0:
        mesh = visii.mesh.create_sphere(name)
    if obj_id == 1:
        mesh = visii.mesh.create_torus_knot(name, 
            np.random.randint(2,6), 
            np.random.randint(4,10))
    if obj_id == 2:
        mesh = visii.mesh.create_teapotahedron(name)
    if obj_id == 3:
        mesh = visii.mesh.create_box(name)
    if obj_id == 4:
        mesh = visii.mesh.create_capped_cone(name)
    if obj_id == 5:
        mesh = visii.mesh.create_capped_cylinder(name)
    if obj_id == 6:
        mesh = visii.mesh.create_capsule(name)
    if obj_id == 7:
        mesh = visii.mesh.create_cylinder(name)
    if obj_id == 8:
        mesh = visii.mesh.create_disk(name)
    if obj_id == 9:
        mesh = visii.mesh.create_dodecahedron(name)
    if obj_id == 10:
        mesh = visii.mesh.create_icosahedron(name)
    if obj_id == 11:
        mesh = visii.mesh.create_icosphere(name)
    if obj_id == 12:
        mesh = visii.mesh.create_rounded_box(name)
    if obj_id == 13:
        mesh = visii.mesh.create_spring(name)
    if obj_id == 14:
        mesh = visii.mesh.create_torus(name)
    if obj_id == 15:
        mesh = visii.mesh.create_tube(name)

    obj.set_mesh(mesh)
    obj.get_transform().set_position(
        np.random.uniform(-5,5),
        np.random.uniform(-5,5),
        np.random.uniform(-10,3)
        )
    q = Quaternion.random()
    obj.get_transform().set_rotation(
        visii.quat(q.w,q.x,q.y,q.z)
        )
    obj.get_transform().set_scale(np.random.uniform(0.01,0.2))
    
    c = eval(str(rcolor.generate(luminosity='bright',format_='rgb')[0])[3:])
    obj.get_material().set_base_color(
        c[0]/255.0,
        c[1]/255.0,
        c[2]/255.0)  

    obj.get_material().set_roughness(np.random.uniform(0,1)) # default is 1  
    obj.get_material().set_metallic(np.random.uniform(0,1))  # degault is 0     
    obj.get_material().set_transmission(np.random.uniform(0,1))  # degault is 0     
    obj.get_material().set_sheen(np.random.uniform(0,1))  # degault is 0     
    obj.get_material().set_clearcoat(np.random.uniform(0,1))  # degault is 0     
    obj.get_material().set_specular(np.random.uniform(0,1))  # degault is 0     
    obj.get_material().set_anisotropic(np.random.uniform(0,1))  # degault is 0     


# create a random scene, the function defines the values
for i in range(NB_OBJS):
    add_random_obj(str(i))

for i in range(NB_LIGHTS):
    add_random_light("l"+str(i))

################################################################

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
