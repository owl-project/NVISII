#%%

# 22.volumes.py
#
# This shows an example of two volumes. One volume uses the NanoVDB format,
# and the other is a raw volume.

import nvisii
import numpy as np
opt = lambda: None
opt.spp = 50 
opt.width = 512
opt.height = 512 
opt.out = '22_volumes.png'

# headless - no window
# verbose - output number of frames rendered, etc..
nvisii.initialize(headless = False, verbose = True, window_on_top = True)

#%%
# Use a neural network to denoise ray traced
nvisii.enable_denoiser()

# First, lets create an entity that will serve as our camera.
camera = nvisii.entity.create(name = "camera")

# To place the camera into our scene, we'll add a "transform" component.
# (All nvisii objects have a "name" that can be used for easy lookup later.)
camera.set_transform(nvisii.transform.create(name = "camera_transform"))

# To make our camera entity act like a "camera", we'll add a camera component
camera.set_camera(
    nvisii.camera.create_from_fov(
        name = "camera_camera", 
        field_of_view = 0.785398, # note, this is in radians
        aspect = opt.width / float(opt.height)
    )
)

# Finally, we'll select this entity to be the current camera entity.
# (nvisii can only use one camera at the time)
nvisii.set_camera_entity(camera)

# Lets set the camera to look at an object. 
# We'll do this by editing the transform component.
camera.get_transform().look_at(at = (0, 0, .9), up = (0, 0, 1), eye = (0, 5, 1))

# Next, lets at an object (a floor).
floor = nvisii.entity.create(
    name = "floor",
    mesh = nvisii.mesh.create_plane("mesh_floor"),
    transform = nvisii.transform.create("transform_floor"),
    material = nvisii.material.create("material_floor")
)

# Lets make our floor act as a mirror
mat = floor.get_material()
# mat = nvisii.material.get("material_floor") # <- this also works
#%%
# Mirrors are smooth and "metallic".
mat.set_base_color((1.,1.,1.)) 
mat.set_metallic(0) 
mat.set_roughness(1)

# Make the floor large by scaling it
trans = floor.get_transform()
trans.set_scale((5,5,1))

#%%
# Let's also add a sphere
torus = nvisii.entity.create(
    name="torus",
    volume = nvisii.volume.create_torus("torus"),
    transform = nvisii.transform.create("torus"),
    material = nvisii.material.create("torus")
)
#%%
torus.get_transform().set_position((.5,2,0.35))
torus.get_transform().set_scale((0.003, 0.003, 0.003))
torus.get_transform().set_angle_axis(3.14 * .25, (1,0,0))
torus.get_material().set_base_color((.0,0.0,1))  
torus.get_material().set_roughness(0.0)
torus.get_material().set_transmission(0.0)
torus.get_volume().set_gradient_factor(10)
torus.get_volume().set_absorption(0)
torus.get_volume().set_scattering(1)
torus.get_volume().set_scale(50)

#%%
# Let's also add a bunny
bunny = nvisii.entity.create(
    name="bunny",
    volume = nvisii.volume.create_from_file("bunny", "./content/bunny_cloud.nvdb"),
    transform = nvisii.transform.create("bunny"),
    material = nvisii.material.create("bunny")
)
#%%
bunny.get_transform().set_position((-1,0,0.75))
bunny.get_transform().set_scale((0.003, 0.003, 0.003))
bunny.get_material().set_base_color((0.1,0.9,0.08))  
bunny.get_material().set_roughness(0.7)   
bunny.get_volume().set_gradient_factor(10)
bunny.get_volume().set_absorption(0)
bunny.get_volume().set_scattering(1)
bunny.get_volume().set_scale(10)
bunny.get_transform().set_angle_axis(nvisii.pi() * .5, (1,0,0))
bunny.get_transform().add_angle_axis(nvisii.pi(), (0,1,0))

#%%
voxels = np.fromfile("./content/boston_teapot_256x256x178_uint8.raw", dtype=np.uint8).astype(np.float32)



#%%
# Let's also add a teapot
teapot = nvisii.entity.create(
    name="teapot",
    volume = nvisii.volume.create_from_data("teapot", width = 256, height = 256, depth = 178, data = voxels, background = 0.0),
    transform = nvisii.transform.create("teapot"),
    material = nvisii.material.create("teapot")
)
#%%
teapot.get_transform().set_position((1,0,0.7))
teapot.get_transform().set_scale((0.005, 0.005, 0.005))
teapot.get_material().set_base_color((1.0,0.0,0.0))  
teapot.get_material().set_roughness(0.0)
teapot.get_material().set_metallic(1.0)
teapot.get_volume().set_gradient_factor(100)
teapot.get_volume().set_absorption(1)
teapot.get_volume().set_scattering(0)
teapot.get_volume().set_scale(250)
teapot.get_transform().set_angle_axis(-nvisii.pi() * .5, (1,0,0))
teapot.get_transform().add_angle_axis(nvisii.pi() * 1.1, (0,1,0))

#%%
#%%
# Now that we have a simple scene, let's render it 
print("rendering to", "01_simple_scene.png")
nvisii.render_to_file(
    width = opt.width, 
    height = opt.height, 
    samples_per_pixel = opt.spp,   
    file_path = "01_simple_scene.png"
)

nvisii.deinitialize()
