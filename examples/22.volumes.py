#%%

# 22.volumes.py
#
# This shows an example of several volumes. Some volume uses the NanoVDB format,
# others use a raw volume, and then some are generated procedurally.
# This scene tests how volumes can be lit up with light sources, and how they can 
# overlap. 

# Note, the API here is subject to change with future versions...

import nvisii
import numpy as np
opt = lambda: None
opt.spp = 512 
opt.width = 1024
opt.height = 1024 
opt.out = '22_volumes.png'

nvisii.initialize(headless = False, verbose = True, window_on_top = True)
nvisii.enable_denoiser()

# Configuring the denoiser here to not use albedo and normal guides, which are 
# noisy for volumes
nvisii.configure_denoiser(False, False, True)

# Make a camera...
camera = nvisii.entity.create(name = "camera")
camera.set_transform(nvisii.transform.create(name = "camera_transform"))
camera.set_camera(
    nvisii.camera.create_from_fov(
        name = "camera_camera", 
        field_of_view = 0.785398, # note, this is in radians
        aspect = opt.width / float(opt.height)
    )
)
nvisii.set_camera_entity(camera)
camera.get_transform().look_at(at = (0, 0, .5), up = (0, 0, 1), eye = (0, 5, 2))

# Make a dome light
env_tex = nvisii.texture.create_from_file("env_tex", "./content/kiara_4_mid-morning_4k.hdr")
nvisii.enable_dome_light_sampling()
nvisii.set_dome_light_texture(env_tex, enable_cdf=True)
nvisii.set_dome_light_exposure(-2.0)


# Make a textured floor
floor = nvisii.entity.create(
    name = "floor",
    mesh = nvisii.mesh.create_plane("mesh_floor"),
    transform = nvisii.transform.create("transform_floor"),
    material = nvisii.material.create("material_floor")
)
mat = floor.get_material()
floor_tex = nvisii.texture.create_from_file("floor_tex", "./content/salle_de_bain_separated/textures/WoodFloor_BaseColor.jpg")
mat.set_base_color_texture(floor_tex) 
trans = floor.get_transform()
trans.set_scale((5,5,1))

# Make a procedural torus volume 
torus = nvisii.entity.create(
    name="torus",
    volume = nvisii.volume.create_torus("torus"),
    transform = nvisii.transform.create("torus"),
    material = nvisii.material.create("torus")
)
torus.get_transform().set_position((0.8,2,.2))
torus.get_transform().set_scale((0.003, 0.003, 0.003))
torus.get_transform().set_angle_axis(nvisii.pi() * .5, (1,0,0))
torus.get_material().set_base_color((1.,1.,1.0))  
# The gradient factor here controls how "surface like" the volume is. 
# Higher values mean "more surface like" in areas where there is a strong 
# gradient in the scalar field of the volume (which occurs near surfaces defined 
# by high density regions)
torus.get_volume().set_gradient_factor(10) 

# Absorption controls the probability of light being absorbed by the volume
torus.get_volume().set_absorption(1.)
# Absorption controls the probability of light bouncing off one of the particles in the volume
torus.get_volume().set_scattering(.0)
# The scale here controls how "big" a voxel is, where "1" means a voxel is 1cm wide.
# Larger scales result in particles being distributed over longer distances, 
# causing the volume to appear less dense
torus.get_volume().set_scale(100)

# Create a procedural octahedron
octahedron = nvisii.entity.create(
    name="octahedron",
    volume = nvisii.volume.create_octahedron("octahedron"),
    transform = nvisii.transform.create("octahedron"),
    material = nvisii.material.create("octahedron")
)
octahedron.get_transform().set_position((.80,2.0,0.2)) # Note that this octahedron is inside the torus
octahedron.get_transform().set_scale((0.01, 0.01, 0.01))
octahedron.get_transform().set_angle_axis(nvisii.pi() * .25, (0,0,1))
octahedron.get_material().set_base_color((1.0,0.0,0))  
octahedron.get_volume().set_gradient_factor(10)
octahedron.get_volume().set_absorption(0)
octahedron.get_volume().set_scattering(1)
octahedron.get_volume().set_scale(15)

# Create a procedural sphere
sphere = nvisii.entity.create(
    name="sphere",
    volume = nvisii.volume.create_sphere("sphere"),
    transform = nvisii.transform.create("sphere"),
    material = nvisii.material.create("sphere")
)
sphere.get_transform().set_position((-1.0,2,0.25))
sphere.get_transform().set_scale((0.0025, 0.0025, 0.0025))
sphere.get_material().set_base_color((0.2,0.2,1.0))  
sphere.get_volume().set_gradient_factor(10)
sphere.get_volume().set_absorption(0)
sphere.get_volume().set_scattering(1)
sphere.get_volume().set_scale(100)

# Create a procedural box
box = nvisii.entity.create(
    name="box",
    volume = nvisii.volume.create_box("box"),
    transform = nvisii.transform.create("box"),
    material = nvisii.material.create("box")
)
box.get_transform().set_position((-1.0,2,0.25))
box.get_transform().set_scale((0.005, 0.005, 0.005))
box.get_transform().set_angle_axis(.3, (0,0,1))
box.get_material().set_base_color((1.0,1.0,1.0))  
box.get_volume().set_gradient_factor(10)
box.get_volume().set_absorption(0)
box.get_volume().set_scattering(1)
box.get_volume().set_scale(100)

# Create a cloudy bunny using a nanovdb file
bunny = nvisii.entity.create(
    name="bunny",
    volume = nvisii.volume.create_from_file("bunny", "./content/bunny_cloud.nvdb"),
    transform = nvisii.transform.create("bunny"),
    material = nvisii.material.create("bunny")
)
bunny.get_transform().set_position((-.8,.5,0.75))
bunny.get_transform().set_scale((0.003, 0.003, 0.003))
bunny.get_material().set_base_color((0.1,0.9,0.08))  
bunny.get_material().set_roughness(0.7)   
bunny.get_volume().set_gradient_factor(10)
bunny.get_volume().set_absorption(1)
bunny.get_volume().set_scattering(0)
bunny.get_volume().set_scale(4)
bunny.get_transform().set_angle_axis(nvisii.pi() * .5, (1,0,0))
bunny.get_transform().add_angle_axis(nvisii.pi(), (0,1,0))

# Create a boston teapot using a raw CT scanned volume
voxels = np.fromfile("./content/boston_teapot_256x256x178_uint8.raw", dtype=np.uint8).astype(np.float32)
teapot = nvisii.entity.create(
    name="teapot",
    volume = nvisii.volume.create_from_data("teapot", width = 256, height = 256, depth = 178, data = voxels, background = 0.0),
    transform = nvisii.transform.create("teapot"),
    material = nvisii.material.create("teapot")
)
teapot.get_transform().set_position((1,0,0.7))
teapot.get_transform().set_scale((0.005, 0.005, 0.005))
teapot.get_material().set_base_color((1.0,1.0,1.0))  
teapot.get_material().set_roughness(0.0)
teapot.get_material().set_metallic(1.0)
teapot.get_volume().set_gradient_factor(100)
teapot.get_volume().set_absorption(1)
teapot.get_volume().set_scattering(0)
teapot.get_volume().set_scale(250)
teapot.get_transform().set_angle_axis(-nvisii.pi() * .5, (1,0,0))
teapot.get_transform().add_angle_axis(nvisii.pi() * 1.1, (0,1,0))

# Volumes can be lit up using light sources
light = nvisii.entity.create(
    name="light",
    mesh = nvisii.mesh.create_sphere("light"),
    transform = nvisii.transform.create("light"),
    light = nvisii.light.create("light")
)
light.get_transform().set_position((0,1,2.5))
light.get_transform().set_scale((.2,.2,.2))
light.get_light().set_temperature(4000)
light.get_light().set_intensity(20)

# Render out the image
print("rendering to", "22_volumes.png")
nvisii.render_to_file(
    width = opt.width, 
    height = opt.height, 
    samples_per_pixel = opt.spp,   
    file_path = "22_volumes.png"
)

nvisii.deinitialize()
