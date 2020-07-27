import os
import visii
import noise
import random
import argparse
import numpy as np 
import PIL
from PIL import Image
from scipy.ndimage import map_coordinates
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
parser.add_argument('--out',
                    default='tmp.png',
                    help = "output filename")
parser.add_argument('--outf',
                    default='reprojection',
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

floor.get_transform().set_scale(visii.vec3(100))
floor.get_material().set_roughness(1.0)

areaLight1 = visii.entity.create(
    name="areaLight1",
    light = visii.light.create("areaLight1"),
    transform = visii.transform.create("areaLight1"),
    mesh = visii.mesh.create_teapotahedron("areaLight1"),
)
areaLight1.get_light().set_intensity(10000.)
areaLight1.get_light().set_temperature(8000)
areaLight1.get_transform().set_position(
    visii.vec3(0, 0, 5))

mesh1 = visii.entity.create(
    name="mesh1",
    mesh = visii.mesh.create_teapotahedron("mesh1"),
    transform = visii.transform.create("mesh1"),
    material = visii.material.create("mesh1")
)

mesh1.get_material().set_roughness(1.0)
mesh1.get_material().set_base_color(
    visii.vec3(1.0, 0.0, 0.0))

mesh1.get_transform().set_position(visii.vec3(-0.05, 0.0, 0))
mesh1.get_transform().set_scale(visii.vec3(0.1))
mesh1.get_transform().set_linear_velocity(visii.vec3(0.1, 0.0, 0.0))

visii.set_dome_light_intensity(1)
visii.set_direct_lighting_clamp(10.0)
visii.set_indirect_lighting_clamp(10.0)
visii.set_max_bounce_depth(0)
visii.sample_pixel_area(visii.vec2(.5), visii.vec2(.5))
# # # # # # # # # # # # # # # # # # # # # # # # #

# First, let's render out the scene with motion blur to understand
# how the object is moving
visii.sample_time_interval(visii.vec2(0.0, 1.0))
visii.render_to_png(width=opt.width, height=opt.height, samples_per_pixel=opt.spp,
    image_path=f"{opt.outf}/motion_blur.png"
)

def save_image(data, name):
    img = Image.fromarray(np.clip((np.abs(data) ** (1.0 / 2.2))*255, 0, 255).astype(np.uint8)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
    img.save(name)

# Now let's render out the where the object is at time = 0 and time = 1
visii.sample_time_interval(visii.vec2(0.0, 0.0)) # only sample at t = 0
t0_array = visii.render(
    width=opt.width, 
    height=opt.height, 
    samples_per_pixel=opt.spp, 
    seed = 0
)
t0_array = np.array(t0_array).reshape(opt.height,opt.width,4)
save_image(t0_array, f"{opt.outf}/t0.png")

visii.sample_time_interval(visii.vec2(1.0, 1.0)) # only sample at t = 1
t1_array = visii.render(
    width=opt.width, 
    height=opt.height, 
    samples_per_pixel=1, 
    seed = 1
)
t1_array = np.array(t1_array).reshape(opt.height,opt.width,4)
save_image(t1_array, f"{opt.outf}/t1.png")

# Next, let's obtain segmentation data for both
# these timesteps to do the reprojection
visii.sample_time_interval(visii.vec2(0.0, 0.0))
t0_base_colors_array = visii.render_data(
    width=opt.width, 
    height=opt.height, 
    start_frame=0,
    frame_count=1, 
    bounce=0, 
    options="base_color"
)
t0_base_colors_array = np.array(t0_base_colors_array).reshape(opt.height,opt.width,4)
save_image(t0_base_colors_array, f"{opt.outf}/t0_base_color.png")

visii.sample_time_interval(visii.vec2(1.0, 1.0))
t1_base_colors_array = visii.render_data(
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
visii.sample_time_interval(visii.vec2(1.0, 1.0))
t1_motion_vectors_array = visii.render_data(
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
visii.deinitialize()