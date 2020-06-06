# import sys, os
# os.add_dll_directory(os.path.join(os.getcwd(), '..', 'install'))
# sys.path.append(os.path.join(os.getcwd(), "..", "install"))

# input()

import random
import visii 
import time 
import randomcolor


NB_OBJS = 1
NB_LIGHTS = 20

SAMPLES_PER_PIXEL = 1000

# WIDTH = 1920 
# HEIGHT = 1080

WIDTH =  1000
HEIGHT = 500


visii.initialize_headless()
# visii.initialize_interactive()

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
visii.set_dome_light_intensity(0.4)

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
    obj.get_transform().set_scale(2)


    obj.get_light().set_intensity(random.uniform(50000,100000))
    # obj.get_light().set_temperature(np.random.randint(100,9000))

    c = eval(str(rcolor.generate(luminosity='bright',format_='rgb')[0])[3:])
    obj.get_light().set_color(
        c[0]/255.0,
        c[1]/255.0,
        c[2]/255.0)  
 
    obj.get_light().set_temperature(4000)
    # obj.get_light().set_intensity(10000.)

    obj.get_transform().set_position(
            random.uniform(-10,10),
            random.uniform(-10,10),
            random.uniform(5,30)
        )
def add_random_obj(name = "name"):
    global rcolor
    obj= visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        material = visii.material.create(name),
    )

    obj_id = random.randint(0,15)

    mesh = None
    if obj_id == 0:
        mesh = visii.mesh.create_sphere(name)
    if obj_id == 1:
        mesh = visii.mesh.create_torus_knot(name, 
            random.randint(2,6),
            random.randint(4,10))
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
        random.uniform(-5,5),
        random.uniform(-5,5),
        random.uniform(-10,3)
        )
    obj.get_transform().set_rotation(
        visii.quat(1.0 ,random.random(), random.random(), random.random()) 
        )
    obj.get_transform().set_scale(random.uniform(0.01,0.2))
    
    c = eval(str(rcolor.generate(luminosity='bright',format_='rgb')[0])[3:])
    obj.get_material().set_base_color(
        c[0]/255.0,
        c[1]/255.0,
        c[2]/255.0)  

    obj.get_material().set_roughness(random.uniform(0,1)) # default is 1  
    obj.get_material().set_metallic(random.uniform(0,1))  # degault is 0     
    obj.get_material().set_transmission(random.uniform(0,1))  # degault is 0     
    obj.get_material().set_sheen(random.uniform(0,1))  # degault is 0     
    obj.get_material().set_clearcoat(random.uniform(0,1))  # degault is 0     
    obj.get_material().set_specular(random.uniform(0,1))  # degault is 0     
    obj.get_material().set_anisotropic(random.uniform(0,1))  # degault is 0     


# create a random scene, the function defines the values

print('creating objects')
for i in range(NB_OBJS):
    add_random_obj(str(i))

print('creating lights')
for i in range(NB_LIGHTS):
    add_random_light("l"+str(i))


################################################################
print('rendering')

print('denoiser')
visii.enable_denoiser()
visii.render_to_png(width=WIDTH, 
                    height=HEIGHT, 
                    samples_per_pixel=SAMPLES_PER_PIXEL,
                    image_path="denoise.png")

print('noise')
visii.disable_denoiser()
visii.render_to_png(width=WIDTH, 
                    height=HEIGHT, 
                    samples_per_pixel=SAMPLES_PER_PIXEL,
                    image_path="noise.png")

# visii.cleanup()

