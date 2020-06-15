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
                    default=15,
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
        visii.vec3(6,0,2),
        visii.vec3(0,0,0),
        visii.vec3(0,0,1),
    )
)

areaLight1 = visii.entity.create(
    name="areaLight1",
    light = visii.light.create("areaLight1"),
    transform = visii.transform.create("areaLight1"),
    mesh = visii.mesh.create_teapotahedron("areaLight1"),
)
areaLight1.get_light().set_intensity(10000.)
areaLight1.get_transform().set_position(4, 3, 3)
areaLight1.get_light().set_temperature(4000)

dome = visii.texture.create_from_image("dome", "textures/abandoned_tank_farm_01_1k.hdr")
visii.set_dome_light_texture(dome)

# Physics init 
physicsClient = p.connect(p.DIRECT) # or p.GUI or p.DIRECT for non-graphical version
p.setGravity(0,0,-10)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setAdditionalSearchPath("/home/jtremblay/code/bullet3/data/") #optionally


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
name = 'cube'

soup_mesh = visii.mesh.create_from_obj("soup", "models/alphabet_soup/textured.obj")
soup_texture = visii.texture.create_from_image('soup',"models/alphabet_soup/texture_map.png")

soup_entity = visii.entity.create(
    name="soup_entity",
    # mesh = visii.mesh.create_sphere("mesh1", 1, 128, 128),
    mesh = soup_mesh,
    transform = visii.transform.create("soup_entity"),
    material = visii.material.create("soup_entity")
)

soup_entity.get_material().set_metallic(0)  # should 0 or 1      
soup_entity.get_material().set_transmission(0)  # should 0 or 1      
soup_entity.get_material().set_roughness(random.uniform(0,1)) # default is 1  
soup_entity.get_material().set_roughness(1) # default is 1  
soup_entity.get_material().set_base_color_texture(soup_texture)

soup_entity.get_transform().set_position(0.0, 0.0, 2.0)
# soup_entity.get_transform().set_rotation(visii.quat(0.7071,0,0.7071,0))
# soup_entity.get_transform().set_rotation(visii.quat(0.7071,0,0,0.7071))
base_rot = visii.quat(0.7071,0.7071,0,0)*visii.quat(0.7071,0,0.7071,0)

from pyquaternion import Quaternion 
rq = Quaternion.random()
rot_random = visii.quat(rq.w,rq.x,rq.y,rq.z)

soup_entity.get_transform().set_rotation(base_rot*rot_random)
soup_entity.get_transform().set_scale(0.01)

print(soup_mesh.get_min_aabb_corner(), soup_mesh.get_max_aabb_corner())

soup_col_id = p.createCollisionShape(p.GEOM_CAPSULE,
    radius = 0.3,
    height = 0.82
    )


soup_id = p.createMultiBody(  
                    baseMass = 1, 
                    baseCollisionShapeIndex = soup_col_id,
                    basePosition = [0,0,2.0],
                    baseOrientation= [rot_random.x,rot_random.y,rot_random.z,rot_random.w]
                    )



def render(i_frame):
    visii.render_to_png(
                width=int(opt.width), 
                height=int(opt.height), 
                samples_per_pixel=int(opt.spp),
                image_path=f"{opt.outf}/{str(i_frame).zfill(5)}.png")

for i in range (200):
    p.stepSimulation()
    # if i % 20 == 0:
    if True:
        # time.sleep(1)
        print(i)
        cubePos, cubeOrn = p.getBasePositionAndOrientation(soup_id)


        soup_entity.get_transform().set_position(visii.vec3(
                                                    cubePos[0],
                                                    cubePos[1],
                                                    cubePos[2]
                                                    )
                                                )

        soup_entity.get_transform().set_rotation(visii.quat(
                                                    cubeOrn[3],
                                                    cubeOrn[0],
                                                    cubeOrn[1],
                                                    cubeOrn[2]
                                                    )*base_rot   
                                                )

        camera_entity.get_camera().set_view(
            visii.lookAt(
                visii.vec3(6,0,2),
                visii.vec3(cubePos[0],cubePos[1],cubePos[2]),
                visii.vec3(0,0,1),
            )
        )
        render(i)

p.disconnect()
visii.cleanup()