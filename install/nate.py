import visii 
import numpy as np 
from PIL import Image 
import PIL
import time 
from pyquaternion import Quaternion
import randomcolor


NB_OBJS = 1000
SLEEP_TIME = 12


visii.initialize_headless()

# time to initialize this is a bug

# Create a camera
camera_entity = visii.entity.create(
    name="my_camera",
    transform=visii.transform.create("my_camera"),
    camera=visii.camera.create_perspective_from_fov(name = "my_camera",
        field_of_view = 0.785398,
        aspect = 1.,
        near = .1
        )
    )

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

def add_random_obj(name = "name"):
    global rcolor
    obj= visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
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
        np.random.uniform(-10,0.5)
        )
    q = Quaternion.random()
    obj.get_transform().set_rotation(
        visii.quat(q.w,q.x,q.y,q.z)
        )
    obj.get_transform().set_scale(np.random.uniform(0.01,0.2))
    
    # obj.get_transform().set_scale(
    #     np.random.uniform(0.01,0.2),
    #     np.random.uniform(0.01,0.2),
    #     np.random.uniform(0.01,0.2)
    # )

    # material random

    # obj.get_material().set_base_color(
    #     np.random.uniform(0,1),
    #     np.random.uniform(0,1),
    #     np.random.uniform(0,1)) 

    c = eval(str(rcolor.generate(format_='rgb')[0])[3:])
    obj.get_material().set_base_color(
        c[0]/255.0,
        c[1]/255.0,
        c[2]/255.0)  


    obj.get_material().set_roughness(np.random.uniform(0,1)) # default is 1  
    obj.get_material().set_metallic(np.random.uniform(0,1))  # degault is 0     
    obj.get_material().set_transmission(np.random.uniform(0,1))  # degault is 0     
    obj.get_material().set_sheen(np.random.uniform(0,1))  # degault is 0     


def move_around(obj_id):


# create a random scene, the function defines the values
for i in range(NB_OBJS):
    add_random_obj(str(i))

################################################################

time.sleep(SLEEP_TIME)

# Read and save the image 
x = np.array(visii.read_frame_buffer()).reshape(512,512,4)
img = Image.fromarray((x*255).astype(np.uint8)).transpose(PIL.Image.FLIP_TOP_BOTTOM)

# You should see a noise image, like gaussian noise. 
img.save("tmp.png")

visii.cleanup()
