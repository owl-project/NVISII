import os
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
parser.add_argument('--outf',
                    default='metadata',
                    help = 'folder to output the images')
opt = parser.parse_args()

# # # # # # # # # # # # # # # # # # # # # # # # #
if os.path.isdir(opt.outf):
    print(f'folder {opt.outf}/ exists')
else:
    os.mkdir(opt.outf)
    print(f'created folder {opt.outf}/')
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
    visii.vec3(0,1,1)
)
visii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

floor = visii.entity.create(
    name="floor",
    mesh = visii.mesh.create_plane("floor"),
    transform = visii.transform.create("floor"),
    material = visii.material.create("floor")
)

floor.get_transform().set_scale(visii.visii.vec3(100))
floor.get_material().set_roughness(1.0)

areaLight1 = visii.entity.create(
    name="areaLight1",
    light = visii.light.create("areaLight1"),
    transform = visii.transform.create("areaLight1"),
    mesh = visii.mesh.create_teapotahedron("areaLight1"),
)
areaLight1.get_light().set_intensity(10000.)
areaLight1.get_light().set_temperature(4000)
areaLight1.get_transform().set_position(
    visii.vec3(0, 0, 5))

mesh1 = visii.entity.create(
    name="mesh1",
    mesh = visii.mesh.create_teapotahedron("mesh1"),
    transform = visii.transform.create("mesh1"),
    material = visii.material.create("mesh1")
)

mesh1.get_material().set_metallic(0)  # should 0 or 1      
mesh1.get_material().set_transmission(0.8)  # should 0 or 1      
mesh1.get_material().set_roughness(0.2) # default is 1  
mesh1.get_material().set_base_color(
    visii.vec3(0.9, 0.2, 0.7))

mesh1.get_transform().set_position(
    visii.vec3(0.0, 0.0, 0))
mesh1.get_transform().set_scale(
    visii.vec3(0.1))

# # # # # # # # # # # # # # # # # # # # # # # # #

# visii offers different ways to export meta data
# these are exported in HDR which offers very good 
# storage for values.

visii.render_data_to_hdr(
    width=int(opt.width), 
    height=int(opt.height), 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="depth",
    image_path=f"{opt.outf}/depth.hdr"
)

visii.render_data_to_hdr(
    width=int(opt.width), 
    height=int(opt.height), 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="normal",
    image_path=f"{opt.outf}/normal.hdr"
)

# the entities are stored with an id, 
# visii.entity.get_id(), this is used to 
# do the segmentation. 
# ids = visii.texture.get_ids_names()
# index = ids.indexof('soup')

# visii.texture.get('soup').get_id()

['soup','can']
visii.render_data_to_hdr(
    width=int(opt.width), 
    height=int(opt.height), 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="entity_id",
    image_path=f"{opt.outf}/segmentation.hdr"
)

visii.render_data_to_hdr(
    width=int(opt.width), 
    height=int(opt.height), 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="position",
    image_path=f"{opt.outf}/position.hdr"
)

visii.render_to_png(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    image_path=f"{opt.outf}/img.png"

)

visii.render_to_hdr(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    image_path=f"{opt.outf}/img.hdr"
)

# let's clean up the GPU
visii.cleanup()