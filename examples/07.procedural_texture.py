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
visii.initialize(headless=True, verbose=True)

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
visii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

# Lets use noise package to create a 2D noise texture
img_shape = (20,20,4)
img = np.zeros(img_shape)
for i in range(img_shape[0]):
    for j in range(img_shape[1]):
        for c in range(4):
            img[i][j][c] = noise.snoise_inv(  
                i, j, 
                octaves=10, 
                persistence=.25, 
                lacunarity=1.0,
                base=c
            )

# Going to create two textures, one for the noise,
# and one for 1 - the noise
noise = visii.texture.create_from_data(
    'noise',
    img_shape[0],
    img_shape[1],
    1.0 - img.astype(np.float32)
)    

noise_inv = visii.texture.create_from_data(
    'noise_inv',
    img_shape[0],
    img_shape[1],
    img.astype(np.float32)
)    

# Set the sky
dome = visii.texture.create_from_image("dome", "content/teatro_massimo_2k.hdr")
visii.set_dome_light_intensity(1)
visii.set_dome_light_texture(dome)

# Lets make some objects for our scene
cylinder = visii.entity.create(
    name = "cylinder",
    mesh = visii.mesh.create_capped_cylinder("mesh_cylinder"),
    transform = visii.transform.create("transform_cylinder"),
    material = visii.material.create("material_cylinder")
)
cylinder.get_transform().set_scale((1.0, 1.0, .3))
cylinder.get_transform().set_position((0.0, 0.0, -.3))

# going to use the noise texture to create a marble-like base color
cylinder.get_material().set_roughness_texture(noise)   

teapot = visii.entity.create(
    name="teapot",
    mesh = visii.mesh.create_teapotahedron("teapot", segments = 64),
    transform = visii.transform.create("teapot"),
    material = visii.material.create("teapot")
)
teapot.get_transform().set_position((0,0,0.0))
teapot.get_transform().set_scale((0.2, 0.2, 0.2))
teapot.get_material().set_base_color((0.8,.1,0.1))  

# Use the noise to alternate between smooth metal and rough plastic
teapot.get_material().set_metallic_texture(noise)   
teapot.get_material().set_roughness_texture(noise_inv)

# Make the camera look at the center of the object
camera.get_transform().look_at(
    at = teapot.get_aabb_center(), # look at (world coordinate)
    up = (0,0,1), # up vector
    eye = (1.8,1.8,1.0), # camera_origin    
)

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