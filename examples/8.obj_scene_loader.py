import visii
import noise
import random
import argparse
import numpy as np 

parser = argparse.ArgumentParser()

parser.add_argument('--spp', 
                    default=400,
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
    visii.vec3(0,0,1.2), # look at (world coordinate)
    visii.vec3(0,0,1), # up vector
    visii.vec3(1,-1.5,1.8)
)
visii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

# This function loads the 
sdb = visii.import_obj(
    "sdb", # prefix name
    'content/salle_de_bain_separated/salle_de_bain_separated.obj', #obj path
    'content/salle_de_bain_separated/', # mtl folder 
    visii.vec3(1,0,0), # translation 
    visii.vec3(0.1), # scale here
    visii.angleAxis(3.14 * .5, visii.vec3(1,0,0)) #rotation here
)

# visii loads each obj model as its own entity
# you can find them by name where a prefix is added
# to the obj name defined is added. 

# visii does the same thing for the different material defined 
# in the mtl file

# since obj/mtl do not have definition for metallic propreties 
# lets add them manually to the material
mirror = visii.material.get('sdbMirror')

mirror.set_roughness(0)
mirror.set_metallic(1)
mirror.set_base_color(visii.vec3(1))

# When loading the obj scene, visii returns a list of entities 
# you can loop them to find specific objects or add physical 
# collisions. 

# Since obj/mtl do not define lights, lets add one to the mesh 
# named light
for i_s, s in enumerate(sdb):
    if "light" in s.get_name().lower():
        s.set_light(visii.light.create('light'))
        s.get_light().set_intensity(100)
        s.get_light().set_temperature(5000)
        s.clear_material()

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
visii.cleanup()