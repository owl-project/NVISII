# For more information please read: 
# https://dsp.stackexchange.com/questions/26373/what-is-the-difference-between-a-range-image-and-a-depth-map

import os
import nvisii
import noise
import random
import numpy as np 
import PIL
from PIL import Image 
import math 

opt = lambda: None
opt.spp = 1024 
opt.width = 500
opt.height = 500 
opt.noise = False

# # # # # # # # # # # # # # # # # # # # # # # # #

nvisii.initialize(headless=False, verbose=True)
nvisii.set_dome_light_intensity(1)
nvisii.set_dome_light_color(nvisii.vec3(1,1,1))

if not opt.noise is True: 
    nvisii.enable_denoiser()

camera = nvisii.entity.create(
    name = "camera",
    transform = nvisii.transform.create("camera"),
    camera = nvisii.camera.create(
        name = "camera", 
        aspect = float(opt.width)/float(opt.height)
    )
)

camera.get_transform().look_at(
    at = (0,0,0),
    up = (0,0,1),
    eye = (0,1,1)
)
nvisii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

# Create a scene to use for exporting segmentations
floor = nvisii.entity.create(
    name="floor",
    mesh = nvisii.mesh.create_plane("floor"),
    transform = nvisii.transform.create("floor"),
    material = nvisii.material.create("floor")
)

floor.get_transform().set_scale((2,2,2))
floor.get_material().set_roughness(1.0)

mesh1 = nvisii.entity.create(
    name="mesh1",
    mesh = nvisii.mesh.create_teapotahedron("mesh1", segments=64),
    transform = nvisii.transform.create("mesh1"),
    material = nvisii.material.create("mesh1")
)

brick_base_color = nvisii.texture.create_from_file("bricks_base_color", "./content/Bricks051_2K_Color.jpg")
brick_normal = nvisii.texture.create_from_file("bricks_normal", "./content/Bricks051_2K_Normal.jpg", linear=True)
brick_roughness = nvisii.texture.create_from_file("bricks_roughness", "./content/Bricks051_2K_Roughness.jpg", linear=True)
mesh1.get_material().set_roughness_texture(brick_roughness)
mesh1.get_material().set_normal_map_texture(brick_normal)
mesh1.get_material().set_base_color_texture(brick_base_color)

mesh1.get_transform().set_position((0.0, 0.0, 0))
mesh1.get_transform().set_scale((0.12, 0.12, 0.12))

# # # # # # # # # # # # # # # # # # # # # # # # #

# nvisii offers different ways to export meta data
# these are exported as raw arrays of numbers

# for many segmentations, it might be beneficial to only 
# sample pixel centers instead of the whole pixel area.
# to do so, call this function
nvisii.sample_pixel_area(
    x_sample_interval = (.5,.5), 
    y_sample_interval = (.5, .5)
)

depth_array = nvisii.render_data(
    width=int(opt.width), 
    height=int(opt.height), 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="depth"
)
depth_array = np.array(depth_array).reshape(opt.height,opt.width,4)
depth_array = np.flipud(depth_array)
# save the segmentation image

def convert_from_uvd(u, v, d,fx,fy,cx,cy):
    # d *= self.pxToMetre
    x_over_z = (cx - u) / fx
    y_over_z = (cy - v) / fy
    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z
    return x, y, z

xyz = []
intrinsics = camera.get_camera().get_intrinsic_matrix(opt.width,opt.height)

# Use Open3D to render a point cloud from the distance metadata
for i in range(opt.height):
    for j in range(opt.width):
        x,y,z = convert_from_uvd(i,j, depth_array[i,j,0], 
            intrinsics[0][0], intrinsics[1][1], intrinsics[2][0],intrinsics[2][1])
        xyz.append([x,y,z])
import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
view_ctl = vis.get_view_control()
view_ctl.set_front((1, 1, 0))
view_ctl.set_up((0, -1, -1))
view_ctl.set_lookat(pcd.get_center())
vis.run()
vis.destroy_window()

# let's clean up the GPU
nvisii.deinitialize()