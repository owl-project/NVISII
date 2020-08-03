import visii
import noise
import random
import argparse
import numpy as np 

parser = argparse.ArgumentParser()

parser.add_argument('--spp', 
                    default=100,
                    type=int,
                    help = "number of sample per pixel, higher the more costly")
parser.add_argument('--width', 
                    default=500,
                    type=int,
                    help = 'image output width')
parser.add_argument('--height', 
                    default=500,
                    type=int,
                    help = 'image output height')
parser.add_argument('--noise',
                    action='store_true',
                    default=False,
                    help = "if added the output of the ray tracing is not sent to optix's denoiser")
parser.add_argument('--out',
                    default='tmp.png',
                    help = "output filename")

opt = parser.parse_args()

# # # # # # # # # # # # # # # # # # # # # # # # #
visii.initialize_headless()

if not opt.noise is True: 
    visii.enable_denoiser()

camera = visii.entity.create(
    name = "camera",
    transform = visii.transform.create("camera"),
    camera = visii.camera.create_perspective_from_fov(
        name = "camera", 
        field_of_view = 0.785398, 
        aspect = float(opt.width)/float(opt.height)
    )
)

camera.get_transform().look_at(
    visii.vec3(0,0,0), # look at (world coordinate)
    visii.vec3(0,0,1), # up vector
    visii.vec3(0,0,3), # camera_origin    
)
visii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

visii.set_dome_light_intensity(0)

# third light 
obj_entity = visii.entity.create(
    name="light",
    mesh = visii.mesh.create_plane('light'),
    transform = visii.transform.create("light"),
)
obj_entity.set_light(
    visii.light.create('light')
)
obj_entity.get_light().set_intensity(10000)

obj_entity.get_light().set_temperature(5000)

obj_entity.get_transform().set_scale(
    visii.vec3(0.2)
)
obj_entity.get_transform().set_position(
    visii.vec3(1,0,2)
)
obj_entity.get_transform().look_at(
    at =  visii.vec3(0,0,0), # look at (world coordinate)
    up = visii.vec3(0,0,1), # up vector
)
obj_entity.get_transform().add_rotation(visii.quat(0,0,1,0))


# Lets set some objects in the scene
entity = visii.entity.create(
    name = "floor",
    mesh = visii.mesh.create_plane("mesh_floor"),
    transform = visii.transform.create("transform_floor"),
    material = visii.material.create("material_floor")
)


entity.get_transform().set_scale(visii.vec3(2))

mat = visii.material.get("material_floor")
mat.set_metallic(0)
mat.set_roughness(1)

# # # # # # # # # # # # # # # # # # # # # # # # #

# load the texture 
color_tex = visii.texture.create_from_image("color",'content/Bricks051_2K_Color.jpg')
normal_tex = visii.texture.create_from_image("normal",'content/Bricks051_2K_Normal.jpg')
rough_tex = visii.texture.create_from_image("rough",'content/Bricks051_2K_Roughness.jpg')

mat.set_base_color_texture(color_tex)
mat.set_normal_map_texture(normal_tex)
mat.set_roughness_texture(rough_tex)


# # # # # # # # # # # # # # # # # # # # # # # # #


visii.render_to_png(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    image_path=f"{opt.out}"
)
visii.render_to_hdr(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    image_path=f"{(opt.out).replace('png', 'hdr')}"
)

# let's clean up the GPU
visii.deinitialize()