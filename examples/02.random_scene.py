import visii
import random
import colorsys

NB_OBJS = 10000
SAMPLES_PER_PIXEL = 16
WIDTH = 1920 
HEIGHT = 1080
USE_DENOISER = True
FILE_NAME = "tmp.png"

# # # # # # # # # # # # # # # # # # # # # # # # #

# visii uses components to create a scene. 
# We can increase the max component limit here if necessary.
# In this case, we'll need 16 meshes, a material for each object,
# and finally a transform for each object as well as one more for the camera.
visii.initialize(headless = True, verbose = True, 
    max_entities = NB_OBJS + 1,
    max_transforms = NB_OBJS + 1,  
    max_materials = NB_OBJS,
    max_meshes = 16
    # these are also available
    # max_lights, max_textures, & max_cameras
)

if USE_DENOISER is True: 
    visii.enable_denoiser()

# Create a camera
# Lets create an entity that will serve as our camera. 
camera = visii.entity.create(
    name = "camera",
    transform = visii.transform.create("camera"),
    camera = visii.camera.create(
        name = "camera",  
        aspect = float(WIDTH)/float(HEIGHT)
    )
)

# set the view camera transform
camera.get_transform().look_at(
    at = (0,0,0), # look at (world coordinate)
    up = (1,0,0), # up vector
    eye = (0,0,5), # camera_origin    
)

# set the camera
visii.set_camera_entity(camera)

# Change the dome light intensity
visii.set_dome_light_intensity(5)
visii.set_dome_light_sky(sun_position=(5,-5,5))

# # # # # # # # # # # # # # # # # # # # # # # # #

# lets create a random scene, first lets pre load some mesh
# we are going to use the mesh.get() to retrieve the meshes

visii.mesh.create_sphere('mesh_0')
visii.mesh.create_torus_knot('mesh_1', 
    random.randint(2,6), 
    random.randint(4,10))
visii.mesh.create_teapotahedron('mesh_2')
visii.mesh.create_box('mesh_3')
visii.mesh.create_capped_cone('mesh_4')
visii.mesh.create_capped_cylinder('mesh_5')
visii.mesh.create_capsule('mesh_6')
visii.mesh.create_cylinder('mesh_7')
visii.mesh.create_disk('mesh_8')
visii.mesh.create_dodecahedron('mesh_9')
visii.mesh.create_icosahedron('mesh_10')
visii.mesh.create_icosphere('mesh_11')
visii.mesh.create_rounded_box('mesh_12')
visii.mesh.create_spring('mesh_13')
visii.mesh.create_torus('mesh_14')
visii.mesh.create_tube('mesh_15')

def add_random_obj(name = "name"):
    # this function adds a random object from the pre loaded meshes
    # it will give it a random pose and an random material

    obj= visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
    )

    obj_id = random.randint(0,15)

    # set the mesh, the meshes can be shared
    mesh = visii.mesh.get(f'mesh_{obj_id}')
    obj.set_mesh(mesh)

    obj.get_transform().set_position((
        random.uniform(-5,5),
        random.uniform(-5,5),
        random.uniform(-1,3)
    ))

    obj.get_transform().set_rotation((
        random.uniform(0,1),
        random.uniform(0,1),
        random.uniform(0,1),
        random.uniform(0,1)
    ))

    s = random.uniform(0.05,0.15)
    obj.get_transform().set_scale((
        s,s,s
    ))  

    rgb = colorsys.hsv_to_rgb(
        random.uniform(0,1),
        random.uniform(0.7,1),
        random.uniform(0.7,1)
    )

    obj.get_material().set_base_color(rgb)

    obj_mat = obj.get_material()
    r = random.randint(0,2)

    # This is a simple logic for more natural random materials, e.g.,  
    # mirror or glass like objects
    if r == 0:  
        # Plastic / mat
        obj_mat.set_metallic(0)  # should 0 or 1      
        obj_mat.set_transmission(0)  # should 0 or 1      
        obj_mat.set_roughness(random.uniform(0,1)) # default is 1  
    if r == 1:  
        # metallic
        obj_mat.set_metallic(random.uniform(0.9,1))  # should 0 or 1      
        obj_mat.set_transmission(0)  # should 0 or 1      
    if r == 2:  
        # glass
        obj_mat.set_metallic(0)  # should 0 or 1      
        obj_mat.set_transmission(random.uniform(0.9,1))  # should 0 or 1      

    if r > 0: # for metallic and glass
        r2 = random.randint(0,1)
        if r2 == 1: 
            obj_mat.set_roughness(random.uniform(0,.1)) # default is 1  
        else:
            obj_mat.set_roughness(random.uniform(0.9,1)) # default is 1  

    obj_mat.set_sheen(random.uniform(0,1))  # degault is 0     
    obj_mat.set_clearcoat(random.uniform(0,1))  # degault is 0     
    obj_mat.set_specular(random.uniform(0,1))  # degault is 0     

    r = random.randint(0,1)
    if r == 0:
        obj_mat.set_anisotropic(random.uniform(0,0.1))  # degault is 0     
    else:
        obj_mat.set_anisotropic(random.uniform(0.9,1))  # degault is 0     


# create a random scene, the values are hard coded in the function defines the values
for i in range(NB_OBJS):
    add_random_obj(str(i))
    print("\rcreating random object", i, end="")
print(" - done!")

# # # # # # # # # # # # # # # # # # # # # # # # #
visii.render_to_png(
    width = WIDTH, 
    height = HEIGHT, 
    samples_per_pixel = SAMPLES_PER_PIXEL,   
    image_path = FILE_NAME
)

# let's clean up the GPU
visii.deinitialize()