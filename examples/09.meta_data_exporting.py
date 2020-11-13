import os
import visii
import noise
import random

opt = lambda : None
opt.spp = 512 
opt.width = 500
opt.height = 500 
opt.outf = "09_metadata"

# # # # # # # # # # # # # # # # # # # # # # # # #
if os.path.isdir(opt.outf):
    print(f'folder {opt.outf}/ exists')
else:
    os.mkdir(opt.outf)
    print(f'created folder {opt.outf}/')
# # # # # # # # # # # # # # # # # # # # # # # # #

visii.initialize(headless=False, verbose=True, lazy_updates = True)

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

visii.render_data_to_file(
    width=opt.width, 
    height=opt.height, 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="depth",
    file_path = f"{opt.outf}/depth.exr"
)

visii.render_data_to_file(
    width=opt.width, 
    height=opt.height, 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="normal",
    file_path = f"{opt.outf}/normal.exr"
)

visii.render_data_to_file(
    width=opt.width, 
    height=opt.height, 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="texture_coordinates",
    file_path = f"{opt.outf}/texture_coordinates.exr"
)

# the entities are stored with an id, 
# visii.entity.get_id(), this is used to 
# do the segmentation. 
# ids = visii.texture.get_ids_names()
# index = ids.indexof('soup')
# visii.texture.get('soup').get_id()
visii.render_data_to_file(
    width=opt.width, 
    height=opt.height, 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="entity_id",
    file_path = f"{opt.outf}/entity_id.exr"
)
    
visii.render_data_to_file(
    width=opt.width, 
    height=opt.height, 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="position",
    file_path = f"{opt.outf}/position.exr"
)

# motion vectors can be useful for reprojection

# induce motion, sample only at T=1
mesh1.get_transform().set_angular_velocity(visii.angleAxis(0.5, (0,0,1)))
visii.sample_time_interval((1,1))
visii.render_data_to_file(
    width=opt.width, 
    height=opt.height, 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="diffuse_motion_vectors",
    file_path = f"{opt.outf}/diffuse_motion_vectors.exr"
)

# for the final image, sample the entire pixel area to anti-alias the result
visii.sample_pixel_area(
    x_sample_interval = (0.0, 1.0), 
    y_sample_interval = (0.0, 1.0)
)

visii.render_to_file(
    width=opt.width, 
    height=opt.height, 
    samples_per_pixel=opt.spp,
    file_path=f"{opt.outf}/img.png"
)

visii.render_to_file(
    width=opt.width, 
    height=opt.height, 
    samples_per_pixel=opt.spp,
    file_path=f"{opt.outf}/img.exr"
)

visii.render_to_file(
    width=opt.width, 
    height=opt.height, 
    samples_per_pixel=opt.spp,
    file_path=f"{opt.outf}/img.hdr"
)

# let's clean up the GPU
visii.deinitialize()