import os
import math
import nvisii
import noise
import random
import numpy as np 
import PIL
from PIL import Image
from scipy.ndimage import map_coordinates

opt = lambda: None
opt.spp = 400 
opt.width = 500
opt.height = 500 
opt.out = '13_reprojection.png'
opt.outf = '13_reprojection'

# # # # # # # # # # # # # # # # # # # # # # # # #
if os.path.isdir(opt.outf):
    print(f'folder {opt.outf}/ exists')
else:
    os.mkdir(opt.outf)
    print(f'created folder {opt.outf}/')

# # # # # # # # # # # # # # # # # # # # # # # # #
nvisii.initialize(headless=False, verbose=True)

camera = nvisii.entity.create(
    name = "camera",
    transform = nvisii.transform.create("camera"),
    camera = nvisii.camera.create(
        name = "camera", 
        aspect = float(opt.width)/float(opt.height)
    )
)

# Add some motion to the camera
angle = 0
camera.get_transform().look_at(
    at = (0,0,.1),
    up = (0,0,1),
    eye = (math.sin(angle), math.cos(angle),.2),
    previous = True
)

angle = -nvisii.pi() * .05
camera.get_transform().look_at(
    at = (0,0,.1),
    up = (0,0,1),
    eye = (math.sin(angle), math.cos(angle),.2),
    previous = False
)

nvisii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

floor = nvisii.entity.create(
    name="floor",
    mesh = nvisii.mesh.create_plane("floor"),
    transform = nvisii.transform.create("floor"),
    material = nvisii.material.create("floor")
)

floor.get_material().set_roughness(1.0)

mesh1 = nvisii.entity.create(
    name="mesh1",
    mesh = nvisii.mesh.create_teapotahedron("mesh1"),
    transform = nvisii.transform.create("mesh1"),
    material = nvisii.material.create("mesh1")
)

mesh1.get_material().set_roughness(1.0)
mesh1.get_material().set_base_color((1.0, 0.0, 0.0))

mesh1.get_transform().set_position((-0.05, 0.0, 0), previous=True)
mesh1.get_transform().set_scale((0.1, 0.1, 0.1), previous = False)
mesh1.get_transform().set_scale((0.1, 0.1, 0.1), previous = True)
mesh1.get_transform().set_position((0.05, 0.0, 0), previous=False)

tex = nvisii.texture.create_from_file("dome", "./content/teatro_massimo_2k.hdr")
nvisii.set_dome_light_texture(tex, enable_cdf=True)
nvisii.set_dome_light_intensity(0.8)

nvisii.set_direct_lighting_clamp(10.0)
nvisii.set_indirect_lighting_clamp(10.0)
nvisii.set_max_bounce_depth(0, 0)
nvisii.sample_pixel_area((.5, .5), (.5, .5))

# # # # # # # # # # # # # # # # # # # # # # # # #
# First, let's render out the scene with motion blur to understand
# how the object is moving
nvisii.sample_time_interval((0.0, 1.0))
nvisii.render_to_file(width=opt.width, height=opt.height, samples_per_pixel=opt.spp,
    file_path=f"{opt.outf}/motion_blur.png"
)

def save_image(data, name):
    img = Image.fromarray(np.clip((np.abs(data) ** (1.0 / 2.2))*255, 0, 255).astype(np.uint8)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
    img.save(name)

# Now let's render out the where the object is at time = 0 and time = 1
nvisii.sample_time_interval((0.0, 0.0)) # only sample at t = 0
t0_array = nvisii.render(
    width=opt.width, 
    height=opt.height, 
    samples_per_pixel=opt.spp, 
    seed = 0
)
t0_array = np.array(t0_array).reshape(opt.height,opt.width,4)
save_image(t0_array, f"{opt.outf}/t0.png")

nvisii.sample_time_interval((1.0, 1.0)) # only sample at t = 1
t1_array = nvisii.render(
    width=opt.width, 
    height=opt.height, 
    samples_per_pixel=8, 
    seed = 1
)
t1_array = np.array(t1_array).reshape(opt.height,opt.width,4)
save_image(t1_array, f"{opt.outf}/t1.png")

# Next, let's obtain segmentation data for both
# these timesteps to do the reprojection
nvisii.sample_time_interval((0.0, 0.0))
t0_base_colors_array = nvisii.render_data(
    width=opt.width, 
    height=opt.height, 
    start_frame=0,
    frame_count=1, 
    bounce=0, 
    options="base_color"
)
t0_base_colors_array = np.array(t0_base_colors_array).reshape(opt.height,opt.width,4)
save_image(t0_base_colors_array, f"{opt.outf}/t0_base_color.png")

nvisii.sample_time_interval((1.0, 1.0))
t1_base_colors_array = nvisii.render_data(
    width=opt.width, 
    height=opt.height, 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="base_color"
)
t1_base_colors_array = np.array(t1_base_colors_array).reshape(opt.height,opt.width,4)
save_image(t1_base_colors_array, f"{opt.outf}/t1_base_color.png")

# After that, get diffuse motion vectors at T1 to drive the reprojection
nvisii.sample_time_interval((1.0, 1.0))
t1_motion_vectors_array = nvisii.render_data_to_file(
    width=opt.width, 
    height=opt.height, 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="diffuse_motion_vectors",
    file_path= f"{opt.outf}/t1_motion_vectors.exr"
)
t1_motion_vectors_array = nvisii.render_data(
    width=opt.width, 
    height=opt.height, 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="diffuse_motion_vectors"
)
t1_motion_vectors_array = np.array(t1_motion_vectors_array).reshape(opt.height,opt.width,4)
save_image(t1_motion_vectors_array, f"{opt.outf}/t1_motion_vectors.png")

# Transform those motion vectors into image space
t1_motion_vectors_array[0][0][0] = 1
t1_motion_vectors_array[0][0][1] = 1
t1_motion_vectors_array[:,:,0] *= opt.width
t1_motion_vectors_array[:,:,1] *= opt.height

# Add relative coordinates to convert the vector field to a lookup table
idx = np.stack(np.meshgrid(np.arange(opt.height), np.arange(opt.width), indexing='ij'),axis=-1)
lut = idx - np.flip(t1_motion_vectors_array[...,:2], axis=2)

# Reproject base color
t0_reproj_base_colors = np.empty((opt.height, opt.width, 4))
for i in range(4):
    t0_reproj_base_colors[...,i] = map_coordinates(t0_base_colors_array[...,i], np.moveaxis(lut, -1, 0), order=0)
save_image(t0_reproj_base_colors, f"{opt.outf}/t0_reproj_base_colors.png")

# Reproject samples
t0_reproj = np.empty((opt.height, opt.width, 4))
for i in range(4):
    t0_reproj[...,i] = map_coordinates(t0_array[...,i], np.moveaxis(lut, -1, 0), order=3)
save_image(t0_reproj, f"{opt.outf}/t0_reproj.png")

# calculate a mixing 'mask' for how much of the reprojected image should be included in the final result
mixing_mask = (t0_reproj_base_colors == t1_base_colors_array).all(axis=-1) * .9
save_image(mixing_mask, f"{opt.outf}/mixing_mask.png")

# combine the reprojected t0 image with the t1 image using the mixing mask
mixed_img = t0_reproj * mixing_mask[:,:,None] + (1-mixing_mask[:,:,None]) * t1_array
save_image(mixed_img, f"{opt.outf}/mixed_img.png")

# let's clean up the GPU
nvisii.deinitialize()
