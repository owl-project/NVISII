import os
import visii
import noise
import random
import argparse
import numpy as np 
import PIL
from PIL import Image 

parser = argparse.ArgumentParser()

parser.add_argument('--spp', 
                    default=1024,
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

visii.initialize(headless=False, verbose=True, lazy_updates = True)

if not opt.noise is True: 
    visii.enable_denoiser()

camera = visii.entity.create(
    name = "camera",
    transform = visii.transform.create("camera"),
    camera = visii.camera.create(
        name = "camera", 
        aspect = float(opt.width)/float(opt.height)
    )
)

camera.get_transform().look_at(
    at = (0,0,0),
    up = (0,0,1),
    eye = (0,1,1)
)
visii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

# Create a scene to use for exporting segmentations
floor = visii.entity.create(
    name="floor",
    mesh = visii.mesh.create_plane("floor"),
    transform = visii.transform.create("floor"),
    material = visii.material.create("floor")
)

floor.get_transform().set_scale((2,2,2))
floor.get_material().set_roughness(1.0)
areaLight1 = visii.entity.create(
    name="areaLight1",
    light = visii.light.create("areaLight1"),
    transform = visii.transform.create("areaLight1"),
    mesh = visii.mesh.create_plane("areaLight1"),
)
areaLight1.get_transform().set_rotation(visii.angleAxis(3.14, (1,0,0)))
areaLight1.get_light().set_intensity(1)
areaLight1.get_light().set_exposure(-3)
areaLight1.get_light().set_temperature(8000)
areaLight1.get_transform().set_position((0, 0, .6))
areaLight1.get_transform().set_scale((.2, .2, .2))

mesh1 = visii.entity.create(
    name="mesh1",
    mesh = visii.mesh.create_teapotahedron("mesh1", segments=64),
    transform = visii.transform.create("mesh1"),
    material = visii.material.create("mesh1")
)

brick_base_color = visii.texture.create_from_file("bricks_base_color", "./content/Bricks051_2K_Color.jpg")
brick_normal = visii.texture.create_from_file("bricks_normal", "./content/Bricks051_2K_Normal.jpg", linear=True)
brick_roughness = visii.texture.create_from_file("bricks_roughness", "./content/Bricks051_2K_Roughness.jpg", linear=True)
mesh1.get_material().set_roughness_texture(brick_roughness)
mesh1.get_material().set_normal_map_texture(brick_normal)
mesh1.get_material().set_base_color_texture(brick_base_color)

mesh1.get_transform().set_position((0.0, 0.0, 0))
mesh1.get_transform().set_scale((0.12, 0.12, 0.12))

visii.set_dome_light_intensity(0)
# # # # # # # # # # # # # # # # # # # # # # # # #

# visii offers different ways to export meta data
# these are exported as raw arrays of numbers

# for many segmentations, it might be beneficial to only 
# sample pixel centers instead of the whole pixel area.
# to do so, call this function
visii.sample_pixel_area(
    x_sample_interval = (.5,.5), 
    y_sample_interval = (.5, .5))

depth_array = visii.render_data(
    width=int(opt.width), 
    height=int(opt.height), 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="depth"
)
depth_array = np.array(depth_array).reshape(opt.width,opt.height,4)
depth_array[...,:-1] = depth_array[...,:-1] / np.max(depth_array[...,:-1])
depth_array[...,:-1] = depth_array[...,:-1] - np.min(depth_array[...,:-1])

# save the segmentation image
img = Image.fromarray((depth_array*255).astype(np.uint8)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
img.save(f"{opt.outf}/depth.png")


normals_array = visii.render_data(
    width=int(opt.width), 
    height=int(opt.height), 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="normal"
)

# transform normals to be between 0 and 1
normals_array = np.array(normals_array).reshape(opt.width, opt.height,4)
normals_array = (normals_array * .5) + .5

# save the segmentation image
img = Image.fromarray((normals_array*255).astype(np.uint8)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
img.save(f"{opt.outf}/normals.png")

texture_coords_array = visii.render_data(
    width=int(opt.width), 
    height=int(opt.height), 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="texture_coordinates"
)
# save the segmentation image
texture_coords_array = np.array(texture_coords_array).reshape(opt.width, opt.height,4)
img = Image.fromarray((texture_coords_array*255).astype(np.uint8)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
img.save(f"{opt.outf}/texture_coordinates.png")
print("done")

# the entities are stored with an id, 
# visii.entity.get_id(), this is used to 
# do the segmentation. 
# ids = visii.texture.get_ids_names()
# index = ids.indexof('soup')
# visii.texture.get('soup').get_id()

segmentation_array = visii.render_data(
    width=int(opt.width), 
    height=int(opt.height), 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="entity_id"
)
segmentation_array = np.array(segmentation_array).reshape(opt.width,opt.height,4)

# set the background as 0. Normalize to make segmentation visible
segmentation_array[segmentation_array>3.0] = 0 
segmentation_array /= 3.0

# save the segmentation image
img = Image.fromarray((segmentation_array*255).astype(np.uint8)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
img.save(f"{opt.outf}/segmentation.png")
    
position_array = visii.render_data(
    width=int(opt.width), 
    height=int(opt.height), 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="position"
)
position_array = np.array(position_array).reshape(opt.width,opt.height,4)
position_array[...,:-1] = position_array[...,:-1] / (np.max(position_array[...,:-1]) - np.min(position_array[...,:-1]))
position_array[...,:-1] = position_array[...,:-1] - np.min(position_array[...,:-1])

# save the segmentation image
img = Image.fromarray((position_array*255).astype(np.uint8)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
img.save(f"{opt.outf}/positions.png")

# motion vectors can be useful for reprojection

# induce motion, sample only at T=1
mesh1.get_transform().set_angular_velocity(visii.angleAxis(0.5, (0,0,1)))
visii.sample_time_interval((1,1))
motion_vectors_array = visii.render_data(
    width=int(opt.width), 
    height=int(opt.height), 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="diffuse_motion_vectors"
)

# transform vectors to be between 0 and 1
motion_vectors_array = np.array(motion_vectors_array).reshape(opt.width,opt.height,4)
motion_vectors_array[...,:-1] = motion_vectors_array[...,:-1] / (np.max(motion_vectors_array[...,:-1]) - np.min(motion_vectors_array[...,:-1]))
motion_vectors_array[...,:-1] = motion_vectors_array[...,:-1] - np.min(motion_vectors_array[...,:-1])

# save the segmentation image
img = Image.fromarray((motion_vectors_array*255).astype(np.uint8)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
img.save(f"{opt.outf}/diffuse_motion_vectors.png")

# for the final image, sample the entire pixel area to anti-alias the result
visii.sample_pixel_area(
    x_sample_interval = (0.0, 1.0), 
    y_sample_interval = (0.0, 1.0)
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
visii.deinitialize()