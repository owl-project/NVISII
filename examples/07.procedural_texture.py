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
    visii.vec3(-2,0,2), # camera_origin    
)
visii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

# Lets use noise package to create a 2d perlin noise 
# texture

shape = (1024,1024)
img = np.zeros(shape)

for i in range(shape[0]):
    for j in range(shape[1]):
        img[i][j] = noise.pnoise2(  
            i/100.0, 
            j/100.0, 
            octaves=6, 
            persistence=0.5, 
            lacunarity=2.0, 
            repeatx=1024, 
            repeaty=1024, 
            base=0  
        )

data = np.concatenate([
        img.reshape(shape[0],shape[1],1),
        img.reshape(shape[0],shape[1],1),
        img.reshape(shape[0],shape[1],1),
        img.reshape(shape[0],shape[1],1),
    ]
)

noise = visii.texture.create_from_data(
    'noise',
    shape[0],
    shape[1],
    data.reshape(shape[0]*shape[1],4).astype(np.float32)
)    

visii.set_dome_light_intensity(1)

# Lets set some objects in the scene
entity = visii.entity.create(
    name = "floor",
    mesh = visii.mesh.create_plane("mesh_floor"),
    transform = visii.transform.create("transform_floor"),
    material = visii.material.create("material_floor")
)
entity.get_transform().set_scale(visii.vec3(2))

mat = visii.material.get("material_floor")
mat.set_metallic(1)
mat.set_roughness(0)

mat.set_roughness_texture(noise)
# # # # # # # # # # # # # # # # # # # # # # # # #


sphere = visii.entity.create(
    name="sphere",
    mesh = visii.mesh.create_sphere("sphere"),
    transform = visii.transform.create("sphere"),
    material = visii.material.create("sphere")
)
sphere.get_transform().set_position(
    visii.vec3(0,0,0.5))
sphere.get_transform().set_scale(
    visii.vec3(0.3))
sphere.get_material().set_base_color(
    visii.vec3(0.2,1,0.1))  
sphere.get_material().set_roughness(1)   
sphere.get_material().set_metallic(0)   



#%%
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