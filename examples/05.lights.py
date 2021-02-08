import nvisii
import random

opt = lambda : None
opt.spp = 1000
opt.width = 500
opt.height = 500 
opt.out = "05_lights.png"

# # # # # # # # # # # # # # # # # # # # # # # # #
nvisii.initialize(headless=True, verbose=True)

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
    at = (0,0,.5),
    up = (0,0,1),
    eye = (-2,-2,1.5),
)
nvisii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

# lets turn off the ambiant lights 
nvisii.set_dome_light_intensity(0)
nvisii.disable_dome_light_sampling()

# add a light to an entity
obj_entity = nvisii.entity.create(
    name="light_1",
    mesh = nvisii.mesh.create_sphere('light_1'),
    transform = nvisii.transform.create("light_1"),
)

# a light is an entity with a light added to it. 
obj_entity.set_light(
    nvisii.light.create('light_1')
)
obj_entity.get_light().set_intensity(1)
obj_entity.get_light().set_temperature(8000)

#lets set the size and placement of the light
obj_entity.get_transform().set_scale((0.2, 0.2, 0.2))

#light above the scene
obj_entity.get_transform().set_position((0,0,1.5))

# Second light 
obj_entity = nvisii.entity.create(
    name="light_2",
    mesh = nvisii.mesh.create_teapotahedron('light_2'),
    transform = nvisii.transform.create("light_2"),
)
# a light is an entity with a light added to it. 
obj_entity.set_light(
    nvisii.light.create('light_2')
)
obj_entity.get_light().set_intensity(2)

# you can also set the light color manually
obj_entity.get_light().set_color((1.,.0,.0))

#lets set the size and placement of the light
obj_entity.get_transform().set_scale((0.1, 0.1, 0.1))
obj_entity.get_transform().set_position((0.2,-0.7,0.10))
obj_entity.get_transform().set_rotation(nvisii.angleAxis(90, (0,0,1)))

# third light 
obj_entity = nvisii.entity.create(
    name="light_3",
    mesh = nvisii.mesh.create_plane('light_3', flip_z = True),
    transform = nvisii.transform.create("light_3"),
)
obj_entity.set_light(
    nvisii.light.create('light_3')
)
# Intensity effects the appearance of the light in 
# addition to what intensity that light emits.
obj_entity.get_light().set_intensity(1)

# Exposure does not effect direct appearance of the light,
# but does effect the relative power of the light in illuminating
# other objects.
obj_entity.get_light().set_exposure(4)

# Light power can also be controlled by surface area (with larger lights emitting more) 
# This has more impact for larger area lights, but is off by default to make lights easier
# to control.
# obj_entity.get_light().use_surface_area(True)

obj_entity.get_light().set_color((0,.5,1))
obj_entity.get_transform().set_scale((0.2, 0.2, 0.2))
obj_entity.get_transform().look_at(
    at = (-1,-1,0),
    up = (0,0,1),
    eye = (-0.5,0.5,1.0)
)

# # # # # # # # # # # # # # # # # # # # # # # # #

# Lets set some objects in the scene

# Create a box that'll act like a room for the objects
room = nvisii.entity.create(
    name="room",
    mesh = nvisii.mesh.create_box('room'),
    transform = nvisii.transform.create("room"),
    material = nvisii.material.create("room"),
)
room.get_transform().set_scale((2,2,2))
room.get_transform().set_position((0,0,2))
mat = nvisii.material.get("room")
mat.set_base_color(nvisii.vec3(0.19,0.16,0.19)) 
mat.set_roughness(1)

sphere = nvisii.entity.create(
    name="sphere",
    mesh = nvisii.mesh.create_sphere("sphere"),
    transform = nvisii.transform.create("sphere"),
    material = nvisii.material.create("sphere")
)
sphere.get_transform().set_position(
    nvisii.vec3(0.4,0,0.2))
sphere.get_transform().set_scale(
    nvisii.vec3(0.2))
sphere.get_material().set_base_color(
    nvisii.vec3(0.1,0.96,0.4))  
sphere.get_material().set_roughness(0.7)   
sphere.get_material().set_specular(1)   

sphere2 = nvisii.entity.create(
    name="sphere2",
    mesh = nvisii.mesh.create_sphere("sphere2"),
    transform = nvisii.transform.create("sphere2"),
    material = nvisii.material.create("sphere2")
)
sphere2.get_transform().set_position(
    nvisii.vec3(-0.5,-0.1,0.1))
sphere2.get_transform().set_scale(
    nvisii.vec3(0.1))
sphere2.get_material().set_base_color(
    nvisii.vec3(0.1,0.56,1))  
sphere2.get_material().set_roughness(0)   
sphere2.get_material().set_specular(0)   

disk = nvisii.entity.create(
    name="disk",
    mesh = nvisii.mesh.create_capped_cylinder("disk"),
    transform = nvisii.transform.create("disk"),
    material = nvisii.material.create("disk")
)
disk.get_transform().set_scale((.4,.4,.01))
disk.get_transform().set_position((0.2,-0.7,0.01))
disk.get_material().set_roughness(0)
disk.get_material().set_metallic(1)
disk.get_material().set_base_color((1,1,1))

cone = nvisii.entity.create(
    name="cone",
    mesh = nvisii.mesh.create_cone("cone"),
    transform = nvisii.transform.create("cone"),
    material = nvisii.material.create("cone")
)
# lets set the cone up
cone.get_transform().set_position(
    nvisii.vec3(0.08,0.35,0.2))
cone.get_transform().set_scale(
    nvisii.vec3(0.3))
cone.get_material().set_base_color(
    nvisii.vec3(245/255, 230/255, 66/255))  
cone.get_material().set_roughness(1)   
cone.get_material().set_specular(0)   
cone.get_material().set_metallic(0)   

box1 = nvisii.entity.create(
    name="box1",
    mesh = nvisii.mesh.create_box("box1"),
    transform = nvisii.transform.create("box1"),
    material = nvisii.material.create("box1")
)
# lets set the box1 up
box1.get_transform().set_position(
    nvisii.vec3(-0.5,0.4,0.2))
box1.get_transform().set_scale(
    nvisii.vec3(0.2))
box1.get_material().set_base_color(
    nvisii.vec3(1,1,1))  
box1.get_material().set_roughness(0)   
box1.get_material().set_specular(0)   
box1.get_material().set_metallic(1)   

#%%
# # # # # # # # # # # # # # # # # # # # # # # # #

nvisii.render_to_file(
    width=opt.width, 
    height=opt.height, 
    samples_per_pixel=opt.spp,
    file_path=opt.out
)

# let's clean up the GPU
nvisii.deinitialize()