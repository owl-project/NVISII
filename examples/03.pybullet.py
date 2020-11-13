import os 
import visii
import random
import colorsys
import subprocess 
import math
import pybullet as p 
import numpy as np

opt = lambda : None
opt.nb_objects = 50
opt.spp = 64 
opt.width = 500
opt.height = 500 
opt.noise = False
opt.frame_freq = 8
opt.nb_frames = 300
opt.outf = '03_pybullet'


# # # # # # # # # # # # # # # # # # # # # # # # #
if os.path.isdir(opt.outf):
    print(f'folder {opt.outf}/ exists')
else:
    os.mkdir(opt.outf)
    print(f'created folder {opt.outf}/')
# # # # # # # # # # # # # # # # # # # # # # # # #

# show an interactive window, and use "lazy" updates for faster object creation time 
visii.initialize(headless=False, lazy_updates=True)

if not opt.noise is True: 
    visii.enable_denoiser()

# Create a camera
camera = visii.entity.create(
    name = "camera",
    transform = visii.transform.create("camera"),
    camera = visii.camera.create_from_fov(
        name = "camera", 
        field_of_view = 0.85,
        aspect = float(opt.width)/float(opt.height)
    )
)
camera.get_transform().look_at(
    at = (0,0,0),
    up = (0,0,1),
    eye = (10,0,4),
)
visii.set_camera_entity(camera)

# Setup bullet physics stuff
seconds_per_step = 1.0 / 240.0
frames_per_second = 30.0
physicsClient = p.connect(p.GUI) # non-graphical version
p.setGravity(0,0,-10)

# Lets set the scene

# Change the dome light intensity
visii.set_dome_light_intensity(1.0)

# atmospheric thickness makes the sky go orange, almost like a sunset
visii.set_dome_light_sky(sun_position=(10,10,10), atmosphere_thickness=1.0, saturation=1.0)

# Lets add a sun light
sun = visii.entity.create(
    name = "sun",
    mesh = visii.mesh.create_sphere("sphere"),
    transform = visii.transform.create("sun"),
    light = visii.light.create("sun")
)
sun.get_transform().set_position((10,10,10))
sun.get_light().set_temperature(5780)
sun.get_light().set_intensity(1000)

floor = visii.entity.create(
    name="floor",
    mesh = visii.mesh.create_plane("floor"),
    transform = visii.transform.create("floor"),
    material = visii.material.create("floor")
)
floor.get_transform().set_position((0,0,0))
floor.get_transform().set_scale((10, 10, 10))
floor.get_material().set_roughness(0.1)
floor.get_material().set_base_color((0.5,0.5,0.5))

# Set the collision with the floor mesh
# first lets get the vertices 
vertices = floor.get_mesh().get_vertices()

# get the position of the object
pos = floor.get_transform().get_position()
pos = [pos[0],pos[1],pos[2]]
scale = floor.get_transform().get_scale()
scale = [scale[0],scale[1],scale[2]]
rot = floor.get_transform().get_rotation()
rot = [rot[0],rot[1],rot[2],rot[3]]

# create a collision shape that is a convex hull
obj_col_id = p.createCollisionShape(
    p.GEOM_MESH,
    vertices = vertices,
    meshScale = scale,
)

# create a body without mass so it is static
p.createMultiBody(
    baseCollisionShapeIndex = obj_col_id,
    basePosition = pos,
    baseOrientation= rot,
)    

# lets create a bunch of objects 
mesh = visii.mesh.create_teapotahedron('mesh')

# set up for pybullet - here we will use indices for 
# objects with holes 
vertices = mesh.get_vertices()
indices = mesh.get_triangle_indices()

ids_pybullet_and_visii_names = []

for i in range(opt.nb_objects):
    name = f"mesh_{i}"
    obj= visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
    )
    obj.set_mesh(mesh)

    # transforms
    pos = visii.vec3(
        random.uniform(-4,4),
        random.uniform(-4,4),
        random.uniform(2,5)
    )
    rot = visii.normalize(visii.quat(
        random.uniform(-1,1),
        random.uniform(-1,1),
        random.uniform(-1,1),
        random.uniform(-1,1),
    ))
    s = random.uniform(0.2,0.5)
    scale = (s,s,s)

    obj.get_transform().set_position(pos)
    obj.get_transform().set_rotation(rot)
    obj.get_transform().set_scale(scale)

    # pybullet setup 
    pos = [pos[0],pos[1],pos[2]]
    rot = [rot[0],rot[1],rot[2],rot[3]]
    scale = [scale[0],scale[1],scale[2]]

    obj_col_id = p.createCollisionShape(
        p.GEOM_MESH,
        vertices = vertices,
        meshScale = scale,
        # if you have static object like a bowl
        # this allows you to have concave objects, but 
        # for non concave object, using indices is 
        # suboptimal, you can uncomment if you want to test
        # indices =  indices,  
    )
    
    p.createMultiBody(
        baseCollisionShapeIndex = obj_col_id,
        basePosition = pos,
        baseOrientation= rot,
        baseMass = random.uniform(0.5,2)
    )       

    # to keep track of the ids and names 
    ids_pybullet_and_visii_names.append(
        {
            "pybullet_id":obj_col_id, 
            "visii_id":name
        }
    )

    # Material setting
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

# Lets run the simulation for a few steps. 
for i in range (int(opt.nb_frames)):
    steps_per_frame = math.ceil( 1.0 / (seconds_per_step * frames_per_second) )
    for j in range(steps_per_frame):
        p.stepSimulation()

    # Lets update the pose of the objects in visii 
    for ids in ids_pybullet_and_visii_names:

        # get the pose of the objects
        pos, rot = p.getBasePositionAndOrientation(ids['pybullet_id'])

        # get the visii entity for that object
        obj_entity = visii.entity.get(ids['visii_id'])
        obj_entity.get_transform().set_position(pos)

        # visii quat expects w as the first argument
        obj_entity.get_transform().set_rotation(rot)
    print(f'rendering frame {str(i).zfill(5)}/{str(opt.nb_frames).zfill(5)}')
    visii.render_to_file(
        width=int(opt.width), 
        height=int(opt.height), 
        samples_per_pixel=int(opt.spp),
        file_path=f"{opt.outf}/{str(i).zfill(5)}.png"
    )

p.disconnect()
visii.deinitialize()

subprocess.call(['ffmpeg', '-y', '-framerate', '30', '-i', r"%05d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '../output.mp4'], cwd=os.path.realpath(opt.outf))
