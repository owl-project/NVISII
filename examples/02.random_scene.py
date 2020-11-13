# 02.random_scene.py
#
# This shows how to generate a randomized scene using a couple built-in mesh 
# types and some randomized materials. 

import visii
from random import *
import colorsys

opt = lambda: None
opt.nb_objs = 10000 
opt.spp = 16 
opt.width = 1920
opt.height = 1080 
opt.out = '02_random_scene.png'

# visii uses sets of components to represent a scene. 
# We can increase the max component limit here if necessary.
# In this case, we'll need 16 meshes, a material for each object,
# and finally a transform for each object as well as one more for the camera.
visii.initialize(
    headless = True, 
    verbose = True, 
    lazy_updates = True,
    max_entities = opt.nb_objs + 1,
    max_transforms = opt.nb_objs + 1,  
    max_materials = opt.nb_objs,
    max_meshes = 16
    # these are also available
    # max_lights, max_textures, & max_cameras
)

# Turn on the denoiser
visii.enable_denoiser()

# Create a camera
camera = visii.entity.create(
    name = "camera",
    transform = visii.transform.create("camera"),
    camera = visii.camera.create(
        name = "camera",  
        aspect = float(opt.width)/float(opt.height)
    )
)
camera.get_transform().look_at(at = (0,0,0), up = (1,0,0), eye = (0,0,5))
visii.set_camera_entity(camera)

# Lets create a random scene. 

# First lets pre-load some mesh components.
visii.mesh.create_sphere('m_0')
visii.mesh.create_torus_knot('m_1')
visii.mesh.create_teapotahedron('m_2')
visii.mesh.create_box('m_3')
visii.mesh.create_capped_cone('m_4')
visii.mesh.create_capped_cylinder('m_5')
visii.mesh.create_capsule('m_6')
visii.mesh.create_cylinder('m_7')
visii.mesh.create_disk('m_8')
visii.mesh.create_dodecahedron('m_9')
visii.mesh.create_icosahedron('m_10')
visii.mesh.create_icosphere('m_11')
visii.mesh.create_rounded_box('m_12')
visii.mesh.create_spring('m_13')
visii.mesh.create_torus('m_14')
visii.mesh.create_tube('m_15')

def add_random_obj(name = "name"):
    # this function adds a random object that uses one of the pre-loaded mesh
    # components, assigning a random pose and random material to that object.

    obj = visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
    )

    mesh_id = randint(0,15)

    # set the mesh. (Note that meshes can be shared, saving memory)
    mesh = visii.mesh.get(f'm_{mesh_id}')
    obj.set_mesh(mesh)

    obj.get_transform().set_position((
        uniform(-5,5),
        uniform(-5,5),
        uniform(-1,3)
    ))

    obj.get_transform().set_rotation((
        uniform(0,1), # X 
        uniform(0,1), # Y
        uniform(0,1), # Z
        uniform(0,1)  # W
    ))

    s = uniform(0.05,0.15)
    obj.get_transform().set_scale((
        s,s,s
    ))  

    rgb = colorsys.hsv_to_rgb(
        uniform(0,1),
        uniform(0.7,1),
        uniform(0.7,1)
    )

    obj.get_material().set_base_color(rgb)

    mat = obj.get_material()
    
    # Some logic to generate "natural" random materials
    material_type = randint(0,2)
    
    # Glossy / Matte Plastic
    if material_type == 0:  
        if randint(0,2): mat.set_roughness(uniform(.9, 1))
        else           : mat.set_roughness(uniform(.0,.1))
    
    # Metallic
    if material_type == 1:  
        mat.set_metallic(uniform(0.9,1))
        if randint(0,2): mat.set_roughness(uniform(.9, 1))
        else           : mat.set_roughness(uniform(.0,.1))
    
    # Glass
    if material_type == 2:  
        mat.set_transmission(uniform(0.9,1))
        
        # controls outside roughness
        if randint(0,2): mat.set_roughness(uniform(.9, 1))
        else           : mat.set_roughness(uniform(.0,.1))
        
        # controls inside roughness
        if randint(0,2): mat.set_transmission_roughness(uniform(.9, 1))
        else           : mat.set_transmission_roughness(uniform(.0,.1))

    mat.set_sheen(uniform(0,1)) # <- soft velvet like reflection near edges
    mat.set_clearcoat(uniform(0,1)) # <- Extra, white, shiny layer. Good for car paint.    
    if randint(0,1): mat.set_anisotropic(uniform(0.9,1)) # elongates highlights 

    # (lots of other material parameters are listed in the docs)

# Now, use the above function to make a bunch of random objects
for i in range(opt.nb_objs):
    add_random_obj(str(i))
    print("\rcreating random object", i, end="")
print(" - done!")

visii.render_to_file(
    width = opt.width, 
    height = opt.height, 
    samples_per_pixel = opt.spp,   
    file_path = opt.out
)

visii.deinitialize()