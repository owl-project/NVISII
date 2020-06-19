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
# Physics init 
physicsClient = p.connect(p.DIRECT) # or p.GUI or p.DIRECT for non-graphical version
p.setGravity(0,0,-10)


if not opt.noise is True: 
    visii.enable_denoiser()


camera_entity = visii.entity.create(
    name="my_camera_entity",
    transform=visii.transform.create("my_camera_transform"),
    camera=visii.camera.create_perspective_from_fov(name = "my_camera", 
        field_of_view = 0.785398, 
        aspect = opt.width/float(opt.height),
        near = .1))


camera_pos = visii.vec3(0,-1.5,1.3)
camera_pos = visii.vec3(1,-1.5,1.8)
visii.set_camera_entity(camera_entity)
# camera_entity.get_transform().set_position(0, 0.0, -5.)
camera_entity.get_camera().use_perspective_from_fov(0.785398, 1.0, .01)
camera_entity.get_camera().set_view(
    visii.lookAt(
        camera_pos,
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

# dome = visii.texture.create_from_image("dome", "textures/abandoned_tank_farm_01_1k.hdr")
# visii.set_dome_light_texture(dome)

test = visii.texture.create_from_image("test", "scenes/salle_de_bain/textures/WoodFloor_BaseColor.jpg")

# initialize the salle de bain
obj_scale = 0.1
# sdb = visii.import_obj("sdb",
#     "scenes/salle_de_bain/salle_de_bain.obj",
#     'scenes/salle_de_bain/',
#     visii.vec3(1,0,0), # translation here
#     visii.vec3(obj_scale), # scale here
#     visii.angleAxis(3.14 * .5, visii.vec3(1,0,0)) #rotation here)
# )

sdb = visii.import_obj("sdb",
    "scenes/salle_de_bain_separated/salle_de_bain_separated.obj",
    'scenes/salle_de_bain_separated/',
    visii.vec3(1,0,0), # translation here
    visii.vec3(obj_scale), # scale here
    visii.angleAxis(3.14 * .5, visii.vec3(1,0,0)) #rotation here)
)


# obj_scale = 0.2
# sdb = visii.import_obj("sdb",
#     "scenes/simple_place/simple_place.obj",
#     'scenes/simple_place/',
#     visii.vec3(0,0,0), # translation here
#     visii.vec3(obj_scale), # scale here
#     # visii.angleAxis(, visii.vec3(0,0,1))
#     # visii.quat(0,0,0,1) #rotation here
#     visii.angleAxis(3.14 * .15, visii.vec3(0,1,0))  * visii.angleAxis(3.14 * .5, visii.vec3(1,0,0))  #rotation here)
#     # visii.angleAxis(3.14 * .5, visii.vec3(1,0,0)) * visii.angleAxis(3.14 * .25, visii.vec3(0,1,0)) #rotation here)
#     # visii.angleAxis(0, visii.vec3(1,0,0) ) #rotation here)

# )

# obj_scale = 0.1
# sdb = visii.import_obj("sdb",
#     "scenes/bowl/bowl.obj",
#     'scenes/bowl/',
#     visii.vec3(0), # translation here
#     visii.vec3(obj_scale), # scale here
#     visii.angleAxis(3.14 * .5, visii.vec3(1,0,0)) #rotation here)
# )

mirror = visii.material.get('sdbMirror')

mirror.set_roughness(0)
mirror.set_metallic(1)
mirror.set_base_color(visii.vec3(1))

# light = visii.entity.get("sbdLight")
# print(light)
# light.set_light(visii.light.create('light'))
# light.get_light().set_intensity(10000)
# light.get_light().set_temperature(5000)


# create collision meshes
light = None
for i_s, s in enumerate(sdb):

    if "light" in s.get_name().lower():
        print ("light")
        s.set_light(visii.light.create('light'))
        s.get_light().set_intensity(100)
        s.get_light().set_temperature(5000)
        s.clear_material()
        light = s
    vertices = []
    indices = s.get_mesh().get_triangle_indices()

    for v in s.get_mesh().get_vertices():
        vertices.append([v[0],v[1],v[2]])
        pos = s.get_transform().get_position()
        pos = [pos[0],pos[1],pos[2]]
        rot = s.get_transform().get_rotation()
        rot = [rot[0],rot[1],rot[2],rot[3]]

    try:    

        obj_col_id = p.createCollisionShape(
            p.GEOM_MESH,
            vertices = vertices,
            meshScale = [obj_scale,obj_scale,obj_scale],
            indices =  indices,
            )
        
        p.createMultiBody(
            baseCollisionShapeIndex = obj_col_id,
            basePosition = pos,
            baseOrientation= rot,
        )    
        print(f"added collision with indices for {s.get_name()}, at {pos}, {rot}")
        # print('min,max')
        # print(s.get_mesh().get_min_aabb_corner(),s.get_mesh().get_max_aabb_corner())
        # print('distance')
        # print(distance(s.get_mesh().get_min_aabb_corner(),s.get_mesh().get_max_aabb_corner()))
    except:
        try:
            obj_col_id = p.createCollisionShape(
                p.GEOM_MESH,
                vertices = vertices,
                meshScale = [obj_scale,obj_scale,obj_scale],
                )
            
            p.createMultiBody(
                baseCollisionShapeIndex = obj_col_id,
                basePosition = pos,
                baseOrientation= rot,
            )    
            print(f"added collision for {s.get_name()}, at {pos}, {rot}")
        except:
            print(f"failed to generate collision for {s.get_name()}")
        # print(distance(s.get_mesh().get_min_aabb_corner(),s.get_mesh().get_max_aabb_corner()))

# raise()

    # raise()
    # a = visii.import_obj("sdb",
    #     "scenes/teapot/teapot.obj",
    #     'scenes/teapot/',
    #     visii.vec3(0), # translation here
    #     visii.vec3(.1), # scale here
    #     visii.angleAxis(3.14 * .5, visii.vec3(1,0,0)) #rotation here)
    # )
    # a[0].get_transform().set_scale(.1)





# floor = visii.entity.create(
#     name="floor",
#     mesh = visii.mesh.create_plane("floor"),
#     transform = visii.transform.create("floor"),
#     material = visii.material.create("floor"),
#     light = visii.light.create('floor')
# )
# floor.get_light().set_intensity(10000)
# floor.get_light().set_temperature(5000)
# # floor.get_transform().set_position(0,0,-0.1)
# floor.get_transform().set_position(visii.vec3(0,0,3))
# floor.get_transform().set_scale(visii.vec3(0.5))
# floor.get_transform().set_rotation(visii.quat(0,0,1,0))

# Set the collision of the floor
# plane_col_id = p.createCollisionShape(p.GEOM_PLANE)
# p.createMultiBody(
#     baseCollisionShapeIndex = plane_col_id,
#     basePosition = [0,0,0],
#     # baseOrientation= p.getQuaternionFromEuler([0,0,0])
# )

# cube_visii = add_random_obj(name='cube',obj_id=3) # force to create a cube



# LOAD SOME OBJECTS 

def render(i_frame):
    visii.render_to_png(
                width=int(opt.width), 
                height=int(opt.height), 
                samples_per_pixel=int(opt.spp),
                image_path=f"{opt.outf}/{str(i_frame).zfill(5)}.png")

import glob 

objects_dict = {}

base_rot = visii.quat(0.7071,0.7071,0,0)*visii.quat(0.7071,0,0.7071,0)
base_rot = visii.quat(1,0,0,0)

for folder in glob.glob("models/*"):
    
    # break

    name  = folder.replace("models/","")
    path_obj = f"{folder}/google_16k/textured.obj"
    path_tex = f"{folder}/google_16k/texture_map.png"
    # if "0" in name:
    #     scale = 0.1 
    # else:
    #     scale = 0.01
    scale = 0.02
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
        mesh_path = path_obj,
        name = name
        )

    objects_dict[name] = {        
        "visii_id": name,
        "bullet_id": bullet_id, 
        'base_rot': base_rot
    }
    # break
    if len(objects_dict.keys())>10:
        break 

print('loaded')
from pyquaternion import Quaternion 

for key in objects_dict:
    pos_rand = [
        np.random.uniform(-0.1,0.1),
        np.random.uniform(-0.1,0.1),
        np.random.uniform(1.8,3),
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
    # if i % 2 == 0:
    if True:
        # time.sleep(1)
        print(i)
        for key in objects_dict:
            update_pose(objects_dict[key])

        pos, rot = p.getBasePositionAndOrientation(objects_dict['MacaroniAndCheese']['bullet_id'])   
        camera_entity.get_camera().set_view(
            visii.lookAt(
                camera_pos,
                visii.vec3(pos[0],pos[1],pos[2]),
                # visii.vec3(-0.5,2,2),
                # light.get_transform().get_position(),
                visii.vec3(0,0,1),
            )
        )
        render(i)

p.disconnect()
visii.cleanup()