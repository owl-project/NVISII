import sys, os

import visii 
from PIL import Image 
import PIL
import time 
from pyquaternion import Quaternion
import randomcolor
import argparse

from utils import *



parser = argparse.ArgumentParser()
   
parser.add_argument('--outf', 
                    default="out_visii")
parser.add_argument('--nbobjects', 
                    default=1500)
parser.add_argument('--distractors', 
                    default=25)
parser.add_argument('--nblights', 
                    default=10)
parser.add_argument('--spp', 
                    default=10)
parser.add_argument('--nbframes',
                    type=int, 
                    default=10)
parser.add_argument('--width', 
                    default=500)
parser.add_argument('--height', 
                    default=500)

opt = parser.parse_args()

try:
    os.mkdir(opt.outf)
    print(f'created {opt.outf}/ folder')
except:
    print(f'{opt.outf}/ exists')




visii.initialize_headless()

# time to initialize this is a bug

# Create a camera
camera_entity = visii.entity.create(
    name="my_camera",
    transform=visii.transform.create("my_camera"),
    camera=visii.camera.create_perspective_from_fov(name = "my_camera",
        field_of_view = 0.785398,
        # aspect = 1.,
        aspect = opt.width/float(opt.height),
        near = .1
        )
    )

# This is to set the camera internal parameters
# camera_entity.get_camera().set_aperture_diameter(20)
# camera_entity.get_camera().set_focal_distance(3.5)

# Change the dome light intensity
visii.set_dome_light_intensity(0.3)

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


# create a random scene, the function defines the values
for i in range(opt.nbobjects):
    add_random_obj(str(i),
        scale_lim = [0.1,0.5],
        x_lim = [-5, 5],
        y_lim = [-5, 5],
        z_lim = [-10, 0]
    )
    random_material(str(i))

for i in range(opt.distractors):
    add_random_obj("d"+str(i),
        scale_lim = [0.1,0.2],
        x_lim = [-2, 2],
        y_lim = [-2, 2],
        z_lim = [0, 3]
    )
    random_material(str(i))


for i in range(opt.nblights):
    add_random_obj("L"+str(i),
        scale_lim = [0.5,1],
        x_lim = [-5, 5],
        y_lim = [-5, 5],
        z_lim = [8, 20]
    )
    random_light("L"+str(i),
        intensity_lim = [50000,100000]
        )    


# create red teapot
scale = 0.2
teapot = visii.entity.create(
            name = 'teapot',
            transform = visii.transform.create('teapot'),
            material = visii.material.create('teapot')
)
teapot.set_mesh(add_random_obj.create_teapotahedron)


# add the cuboid transform
# mainly, mesh now has a get_min_aabb_corner, a get_max_aabb_corner, a get_aabb_center, a get_centroid, 
# and then camera has a get_projection and a get_view

add_cuboid('teapot')

random_material('teapot',color=[1,0,0])

teapot.get_transform().set_scale(scale)
teapot.get_transform().set_position(
    random.uniform(-0.1,0.1),
    random.uniform(-0.1,0.1),
    random.uniform(1,2)        
)

teapot.get_transform().set_rotation(
    visii.quat(1.0 ,random.random(), random.random(), random.random()) 
)    

# for i_t in range(9):
#     trans = visii.transform.get(f"teapot_{i_t}")

#     corner = visii.entity.create(
#         name=f"teapot_{i_t}",
#         mesh = visii.mesh.create_sphere(f"teapot_{i_t}", 1, 128, 128),
#         transform = trans,
#         material = visii.material.create(f"teapot_{i_t}")
#     )
#     corner.get_transform().set_scale(0.1)


################################################################

visii.enable_denoiser()

for i_frame in range(opt.nbframes): 

    print(f"{opt.outf}/{str(i_frame).zfill(4)}.png")

    # randomize the lights
    for i in range(opt.nblights):
        random_intensity("L"+str(i),
            intensity_lim  = [50000, 100000]
            )
        random_color("L"+str(i))

    # randomize the objects     
    for obj_id in range(opt.nbobjects): 
        random_translation(obj_id,
            x_lim = [-5,5],
            y_lim = [-5,5],
            z_lim = [-10,0])
        random_scale(obj_id,
            # scale_lim = [0.1,0.5],
            speed_lim = [0.001,0.005],
            x_lim = [0.3,0.5],
            y_lim = [0.3,0.5],
            z_lim = [0.3,0.5]
            )
        random_rotation(obj_id)
        random_color(obj_id,
            speed_lim = [0.02,0.25])

    for i in range(opt.distractors):
        random_translation("d"+str(i),
            x_lim = [-2, 2],
            y_lim = [-2, 2],
            z_lim = [0, 3],
            speed_lim = [0.02,0.1]
        )
        random_scale("d"+str(i),
            # scale_lim = [0.1,0.5],
            speed_lim = [0.001,0.01],
            x_lim = [0.1,0.2],
            y_lim = [0.1,0.2],
            z_lim = [0.1,0.2]
            )
        random_color("d"+str(i),
            speed_lim = [0.02,0.25])
        random_rotation("d"+str(i))

    # Move object of interest
    random_translation('teapot',
            x_lim = [-1.8,1.8],
            y_lim = [-1.8,1.8],
            z_lim = [1,3],
            speed_lim = [0.05,0.1])
    random_scale('teapot', scale_lim = [scale,scale+0.1], speed_lim = [0.001,0.003])
    random_rotation('teapot')
    # random_color("teapot")

    # Find the positions in image space of the teapot
    # get_projection and a get_view

    # get_cuboid_image_space('teapot')
    export_to_ndds_file(
        filename = f"{opt.outf}/{str(i_frame).zfill(5)}.json",
        obj_names= ['teapot'],
        width=int(opt.width), 
        height=int(opt.height),
        )
    visii.render_to_png(
                    width=int(opt.width), 
                    height=int(opt.height), 
                    samples_per_pixel=int(opt.spp),
                    image_path=f"{opt.outf}/{str(i_frame).zfill(5)}.png")
    
    # import PIL.Image as Image
    # import PIL.ImageDraw as ImageDraw

    # img = Image.open(f"{opt.outf}/{str(i_frame).zfill(4)}.png")
    # draw = ImageDraw.Draw(img)

    # for p in points:
    #     draw.ellipse((int(p[0]*opt.width)-3,int(p[1]*opt.height)-3,int(p[0]*opt.width)+3,int(p[1]*opt.height)+3),fill='blue')

    # img.save(f"{opt.outf}/{str(i_frame).zfill(4)}_out.png")

visii.cleanup()