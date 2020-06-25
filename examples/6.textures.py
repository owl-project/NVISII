import visii
import random
import argparse

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
    visii.vec3(0,0,2), # look at (world coordinate)
    visii.vec3(0,0,1), # up vector
    visii.vec3(-2,0,2), # camera_origin    
)
visii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

visii.set_dome_light_intensity(1)

# load the textures
dome = visii.texture.create_from_image("dome", "content/kiara_4_mid-morning_4k.hdr")
floor_tex = visii.texture.create_from_image("floor",'content/photos_2020_5_11_fst_gray-wall-grunge.jpg')

# we can add HDR images to act as dome
visii.set_dome_light_texture(dome)


# Lets set some objects in the scene
entity = visii.entity.create(
    name = "floor",
    mesh = visii.mesh.create_plane("mesh_floor"),
    transform = visii.transform.create("transform_floor"),
    material = visii.material.create("material_floor")
)
entity.get_transform().set_scale(visii.vec3(5))
mat = visii.material.get("material_floor")

mat.set_roughness(1)

mat.set_base_color_texture(floor_tex)

# # # # # # # # # # # # # # # # # # # # # # # # #


sphere = visii.entity.create(
    name="sphere",
    mesh = visii.mesh.create_sphere("sphere"),
    transform = visii.transform.create("sphere"),
    material = visii.material.create("sphere")
)
sphere.get_transform().set_position(
    visii.vec3(0,0,2))
sphere.get_transform().set_scale(
    visii.vec3(0.2))
sphere.get_material().set_base_color(
    visii.vec3(1,1,1))  
sphere.get_material().set_roughness(0)   
sphere.get_material().set_metallic(1)   



#%%
# # # # # # # # # # # # # # # # # # # # # # # # #

visii.render_to_png(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    image_path=f"{opt.out}"
)

# let's clean up the GPU
visii.cleanup()