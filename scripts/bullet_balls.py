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


dome = visii.texture.create_from_image("dome", "textures/abandoned_tank_farm_01_1k.hdr")
visii.set_dome_light_texture(dome)




# Physics init 
physicsClient = p.connect(p.DIRECT) # or p.GUI or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setAdditionalSearchPath("/home/jtremblay/code/bullet3/data/") #optionally

p.setGravity(0,0,-10)

planeId = p.loadURDF("plane.urdf")
print(p.getAABB(planeId))


floor = visii.entity.create(
    name="floor",
    mesh = visii.mesh.create_plane("floor"),
    transform = visii.transform.create("floor"),
    material = visii.material.create("floor")
)
# floor.get_transform().set_position(0,0,-0.1)
floor.get_transform().set_position(0,0,0)
floor.get_transform().set_scale(100)
floor.get_material().set_roughness(1.0)

print(floor.get_mesh().get_min_aabb_corner(), floor.get_mesh().get_max_aabb_corner())

# random_material("floor")
floor.get_material().set_transmission(0)
floor.get_material().set_metallic(1.0)
floor.get_material().set_roughness(0)

floor.get_material().set_base_color(visii.vec3(0.5,0.5,0.5))

cubeStartPos = [0,0,2]
cubeStartOrientation = p.getQuaternionFromEuler([1,1,0])

# cube_visii = add_random_obj(name='cube',obj_id=3) # force to create a cube
name = 'cube'
cube_visii = visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        material = visii.material.create(name),
        mesh = visii.mesh.create_box(name, visii.vec3(0.5,0.5,0.5))
    )

random_material('cube')


boxId = p.loadURDF("cube.urdf",cubeStartPos, cubeStartOrientation)
print("box",p.getAABB(boxId))

print(p.getCollisionShapeData(boxId,-1))

def render(i_frame):
    visii.render_to_png(
                width=int(opt.width), 
                height=int(opt.height), 
                samples_per_pixel=int(opt.spp),
                image_path=f"{opt.outf}/{str(i_frame).zfill(5)}.png")

for i in range (500):
    p.stepSimulation()
    # if i % 30 == 0:
    if True:
        time.sleep(1)
        print(i)
        cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)


        cube_visii.get_transform().set_position(visii.vec3(
                                                    cubePos[0],
                                                    cubePos[1],
                                                    cubePos[2]
                                                    )
                                                )

        cube_visii.get_transform().set_rotation(visii.quat(
                                                    cubeOrn[3],
                                                    cubeOrn[0],
                                                    cubeOrn[1],
                                                    cubeOrn[2]
                                                    )   
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