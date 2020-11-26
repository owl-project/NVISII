import visii
import random

opt = lambda : None
opt.spp = 256 
opt.width = 500
opt.height = 500 
opt.out = "05_lights.png"

# # # # # # # # # # # # # # # # # # # # # # # # #
visii.initialize(headless=True, verbose=True)

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
    at = (0,0,.5),
    up = (0,0,1),
    eye = (-2,-2,1),
)
visii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

# lets turn off the ambiant lights 
visii.set_dome_light_intensity(0)

# add a light to an entity
obj_entity = visii.entity.create(
    name="light_1",
    mesh = visii.mesh.create_sphere('light_1'),
    transform = visii.transform.create("light_1"),
)

# a light is an entity with a light added to it. 
obj_entity.set_light(
    visii.light.create('light_1')
)
obj_entity.get_light().set_intensity(1)
obj_entity.get_light().set_temperature(8000)

#lets set the size and placement of the light
obj_entity.get_transform().set_scale((0.2, 0.2, 0.2))

#light above the scene
obj_entity.get_transform().set_position((0,0,1.5))

# Second light 
obj_entity = visii.entity.create(
    name="light_2",
    mesh = visii.mesh.create_teapotahedron('light_2'),
    transform = visii.transform.create("light_2"),
)
# a light is an entity with a light added to it. 
obj_entity.set_light(
    visii.light.create('light_2')
)
obj_entity.get_light().set_intensity(1)

# you can also set the light color manually
obj_entity.get_light().set_color((1,0,0))

#lets set the size and placement of the light
obj_entity.get_transform().set_scale((0.1, 0.1, 0.1))
obj_entity.get_transform().set_position((0,-0.6,0.4))
obj_entity.get_transform().set_rotation(visii.angleAxis(90, (0,0,1)))

# third light 
obj_entity = visii.entity.create(
    name="light_3",
    mesh = visii.mesh.create_plane('light_3', flip_z = True),
    transform = visii.transform.create("light_3"),
)
obj_entity.set_light(
    visii.light.create('light_3')
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
room = visii.entity.create(
    name="room",
    mesh = visii.mesh.create_box('room'),
    transform = visii.transform.create("room"),
    material = visii.material.create("room"),
)
room.get_transform().set_scale((2,2,2))
room.get_transform().set_position((0,0,2))
mat = visii.material.get("room")
mat.set_base_color(visii.vec3(0.19,0.16,0.19)) 
mat.set_roughness(1)

sphere = visii.entity.create(
    name="sphere",
    mesh = visii.mesh.create_sphere("sphere"),
    transform = visii.transform.create("sphere"),
    material = visii.material.create("sphere")
)
sphere.get_transform().set_position(
    visii.vec3(0.4,0,0.2))
sphere.get_transform().set_scale(
    visii.vec3(0.2))
sphere.get_material().set_base_color(
    visii.vec3(0.1,0.96,0.4))  
sphere.get_material().set_roughness(0.7)   
sphere.get_material().set_specular(1)   

sphere2 = visii.entity.create(
    name="sphere2",
    mesh = visii.mesh.create_sphere("sphere2"),
    transform = visii.transform.create("sphere2"),
    material = visii.material.create("sphere2")
)
sphere2.get_transform().set_position(
    visii.vec3(-0.5,-0.1,0.1))
sphere2.get_transform().set_scale(
    visii.vec3(0.1))
sphere2.get_material().set_base_color(
    visii.vec3(0.1,0.56,1))  
sphere2.get_material().set_roughness(0)   
sphere2.get_material().set_specular(0)   

sphere3 = visii.entity.create(
    name="sphere3",
    mesh = visii.mesh.create_sphere("sphere3"),
    transform = visii.transform.create("sphere3"),
    material = visii.material.create("sphere3")
)
sphere3.get_transform().set_position(
    visii.vec3(0.6,-0.6,0.16))
sphere3.get_transform().set_scale(
    visii.vec3(0.16))
sphere3.get_material().set_base_color(
    visii.vec3(0.5,0.8,0.5))  
sphere3.get_material().set_roughness(0)   
sphere3.get_material().set_specular(1)   
sphere3.get_material().set_metallic(1)

cone = visii.entity.create(
    name="cone",
    mesh = visii.mesh.create_cone("cone"),
    transform = visii.transform.create("cone"),
    material = visii.material.create("cone")
)
# lets set the cone up
cone.get_transform().set_position(
    visii.vec3(0.08,0.35,0.2))
cone.get_transform().set_scale(
    visii.vec3(0.3))
cone.get_material().set_base_color(
    visii.vec3(245/255, 230/255, 66/255))  
cone.get_material().set_roughness(1)   
cone.get_material().set_specular(0)   
cone.get_material().set_metallic(0)   

box1 = visii.entity.create(
    name="box1",
    mesh = visii.mesh.create_box("box1"),
    transform = visii.transform.create("box1"),
    material = visii.material.create("box1")
)
# lets set the box1 up
box1.get_transform().set_position(
    visii.vec3(-0.5,0.4,0.2))
box1.get_transform().set_scale(
    visii.vec3(0.2))
box1.get_material().set_base_color(
    visii.vec3(1,1,1))  
box1.get_material().set_roughness(0)   
box1.get_material().set_specular(0)   
box1.get_material().set_metallic(1)   

#%%
# # # # # # # # # # # # # # # # # # # # # # # # #

visii.render_to_file(
    width=opt.width, 
    height=opt.height, 
    samples_per_pixel=opt.spp,
    file_path=opt.out
)

# let's clean up the GPU
visii.deinitialize()