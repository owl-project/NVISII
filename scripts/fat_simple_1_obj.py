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
        visii.vec3(5,0,2),
        visii.vec3(0,0,0),
        visii.vec3(0,0,1),
    )
)

# areaLight1 = visii.entity.create(
#     name="areaLight1",
#     light = visii.light.create("areaLight1"),
#     transform = visii.transform.create("areaLight1"),
#     mesh = visii.mesh.create_teapotahedron("areaLight1"),
# )
# areaLight1.get_light().set_intensity(10000.)
# areaLight1.get_transform().set_position(4, 3, 3)
# areaLight1.get_light().set_temperature(4000)

dome = visii.texture.create_from_image("dome", "textures/abandoned_tank_farm_01_1k.hdr")
visii.set_dome_light_texture(dome)

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
floor.get_transform().set_position(0,0,0)
floor.get_transform().set_scale(10)
floor.get_material().set_roughness(1.0)

# random_material("floor")
floor.get_material().set_transmission(0)
floor.get_material().set_metallic(1.0)
floor.get_material().set_roughness(0)

floor.get_material().set_base_color(visii.vec3(0.5,0.5,0.5))

# Set the collision of the floor
plane_col_id = p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(baseCollisionShapeIndex = plane_col_id,
                    basePosition = [0,0,0],
                    # baseOrientation= p.getQuaternionFromEuler([0,0,0])
                    )

# cube_visii = add_random_obj(name='cube',obj_id=3) # force to create a cube

def create_obj(
    name = 'name',
    path_obj = "",
    path_tex = "",
    scale = 1, 
    rot_base = None
    ):

    
    # This is for YCB like dataset
    obj_mesh = visii.mesh.create_from_obj(name, path_obj)
    obj_texture = visii.texture.create_from_image(name,path_tex)

    obj_entity = visii.entity.create(
        name=name,
        # mesh = visii.mesh.create_sphere("mesh1", 1, 128, 128),
        mesh = obj_mesh,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
    )

    # should randomize
    obj_entity.get_material().set_metallic(0)  # should 0 or 1      
    obj_entity.get_material().set_transmission(0)  # should 0 or 1      
    obj_entity.get_material().set_roughness(random.uniform(0,1)) # default is 1  
    obj_entity.get_material().set_roughness(1) # default is 1  

    obj_entity.get_material().set_base_color_texture(obj_texture)

    obj_entity.get_transform().set_scale(scale)

    return obj_entity

def create_physics(
    aabb,             # list two vec3 (min,max) as visii vec3
    base_position = [0,0,0],    # list x,y,z
    base_orientation = [0,0,0,1], # list x,y,z,w 
    base_rot = None,  # visii quat to rotate the frame of the object
    type_collision = p.GEOM_MESH, # so far that is the only one
    scale = 1,        # scale in all directions
    mass = 1,         # mass in kg
    mesh_path=""      # path to the obj mesh
    ):

    if not base_rot is None:    
        min_vec4 = visii.vec4(aabb[0][0],aabb[0][1],aabb[0][2],1)
        max_vec4 = visii.vec4(aabb[1][0],aabb[1][1],aabb[1][2],1)

        rot_min = base_rot * min_vec4
        rot_max = base_rot * max_vec4
        
        aabb = [
                visii.vec3(rot_min[0],rot_min[1],rot_min[2]),
                visii.vec3(rot_max[0],rot_max[1],rot_max[2]),
                ]
    # if type_collision == p.GEOM_BOX:
    if type_collision == p.GEOM_MESH:
        obj_col_id = p.createCollisionShape(
            type_collision,
            meshScale = [scale,scale,scale],
            fileName = mesh_path
            )

    if type_collision == p.GEOM_CAPSULE:
        radius = max(aabb[1][0],aabb[1][1])
        height = aabb[1][2] * 2 

        obj_col_id = p.createCollisionShape(
            type_collision,
            radius = radius * scale,
            height = height * scale
        )


    obj_id = p.createMultiBody(  
                        baseMass = 1, 
                        baseCollisionShapeIndex = obj_col_id,
                        basePosition = base_position,
                        baseOrientation= base_orientation,
                        )

    return obj_id


def update_pose(obj_dict):
    pos, rot = p.getBasePositionAndOrientation(obj_dict['bullet_id'])


    obj_entity = visii.entity.get(obj_dict['visii_id'])
    obj_entity.get_transform().set_position(visii.vec3(
                                            pos[0],
                                            pos[1],
                                            pos[2]
                                            )
                                        )
    if not obj_dict['base_rot'] is None: 
        obj_entity.get_transform().set_rotation(visii.quat(
                                                rot[3],
                                                rot[0],
                                                rot[1],
                                                rot[2]
                                                ) * obj_dict['base_rot']   
                                            )
    else:
        obj_entity.get_transform().set_rotation(visii.quat(
                                                rot[3],
                                                rot[0],
                                                rot[1],
                                                rot[2]
                                                )   
                                            )

def render(i_frame):
    visii.render_to_png(
                width=int(opt.width), 
                height=int(opt.height), 
                samples_per_pixel=int(opt.spp),
                image_path=f"{opt.outf}/{str(i_frame).zfill(5)}.png")


# LOAD SOME OBJECTS 


import glob 

objects_dict = {}

base_rot = visii.quat(0.7071,0.7071,0,0)*visii.quat(0.7071,0,0.7071,0)
base_rot = visii.quat(1,0,0,0)

for folder in glob.glob("models/*"):
    name  = folder.replace("models/","")
    path_obj = f"{folder}/google_16k/textured.obj"
    path_tex = f"{folder}/google_16k/texture_map.png"
    # if "0" in name:
    #     scale = 0.1 
    # else:
    #     scale = 0.01
    scale = 0.1
    print(f"loading {name}")
    obj_entity = create_obj(
        name = name,
        path_obj = path_obj,
        path_tex = path_tex,
        scale = scale,
        )

    bullet_id = create_physics(
        # base_position = [pos[0],pos[1],pos[2]],
        # base_orientation = [rot_random[0],rot_random[1],rot_random[2],rot_random[3]],
        base_rot = base_rot,
        aabb = [obj_entity.get_mesh().get_min_aabb_corner(), 
                obj_entity.get_mesh().get_max_aabb_corner()],
        scale = scale,
        mesh_path = path_obj
        )

    objects_dict[name] = {        
        "visii_id": name,
        "bullet_id": bullet_id, 
        'base_rot': base_rot
    }
    # break
    if len(objects_dict.keys())>2:
        break 

print('loaded')
from pyquaternion import Quaternion 

for key in objects_dict:
    pos_rand = [
        np.random.uniform(-2,2),
        np.random.uniform(-2,2),
        np.random.uniform(2,4),
    ]
    rq = Quaternion.random()
    rot_random = visii.quat(rq.w,rq.x,rq.y,rq.z)

    # update physics.

    p.resetBasePositionAndOrientation(
        objects_dict[key]['bullet_id'],
        pos_rand,
        [rot_random[0],rot_random[1],rot_random[2],rot_random[3]]
    )


for i in range (1000):
    p.stepSimulation()
    # if i % 20 == 0:
    if True:
        # time.sleep(1)
        print(i)
        for key in objects_dict:
            update_pose(objects_dict[key])


        # camera_entity.get_camera().set_view(
        #     visii.lookAt(
        #         visii.vec3(6,0,2),
        #         visii.vec3(cubePos[0],cubePos[1],cubePos[2]),
        #         visii.vec3(0,0,1),
        #     )
        # )
        render(i)

p.disconnect()
visii.cleanup()