import nvisii
import random

opt = lambda: None
opt.spp = 400 
opt.width = 500
opt.height = 500 
opt.noise = False
opt.out = '10_light_texture.png'

# # # # # # # # # # # # # # # # # # # # # # # # #
nvisii.initialize(headless = True, verbose = True)

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
    nvisii.vec3(0,0,0), # look at (world coordinate)
    nvisii.vec3(0,0,1), # up vector
    nvisii.vec3(-2,0,1), # camera_origin    
)
nvisii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

# lets turn off the ambiant lights 
nvisii.set_dome_light_intensity(0)
nvisii.disable_dome_light_sampling()

tex = nvisii.texture.create_from_file("tex", "content/gradient.png")


obj_entity = nvisii.entity.create(
    name="light",
    mesh = nvisii.mesh.create_plane('light'),
    transform = nvisii.transform.create("light"),
)
obj_entity.set_light(
    nvisii.light.create('light')
)

# Intensity effects the appearance of the light in 
# addition to what intensity that light emits.
obj_entity.get_light().set_intensity(2)

# lets set the color texture as the color of the light
obj_entity.get_light().set_color_texture(tex)

obj_entity.get_transform().set_scale((0.6,0.6,0.2))
obj_entity.get_transform().set_position((0.5,-0.4,0.7))

obj_entity.get_transform().look_at(
    at = (0,0,0),
    up = (0,0,1),
)
obj_entity.get_transform().add_rotation((0,1,0,0))


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
obj_entity.get_light().set_color_texture(tex)

#lets set the size and placement of the light
obj_entity.get_transform().set_scale((0.1, 0.1, 0.1))
obj_entity.get_transform().set_position((-0.5,0.4,0))
obj_entity.get_transform().set_rotation(
    nvisii.angleAxis(90, (0,0,1))
)

# # # # # # # # # # # # # # # # # # # # # # # # #

# Lets set some objects in the scene
room = nvisii.entity.create(
    name="room",
    mesh = nvisii.mesh.create_box('room'),
    transform = nvisii.transform.create("room"),
    material = nvisii.material.create("room"),
)
room.get_transform().set_scale((2.0,2.0,2.0))
room.get_transform().set_position((0,0,2.0))
mat = nvisii.material.get("room")
mat.set_base_color(nvisii.vec3(0.19,0.16,0.19)) 
mat.set_roughness(1)

sphere = nvisii.entity.create(
    name="sphere",
    mesh = nvisii.mesh.create_sphere("sphere"),
    transform = nvisii.transform.create("sphere"),
    material = nvisii.material.create("sphere")
)
sphere.get_transform().set_position((0.4,0,0.2))
sphere.get_transform().set_scale((0.2, 0.2, 0.2))
sphere.get_material().set_base_color((0.1,0.96,0.4))  
sphere.get_material().set_roughness(0.7)   
sphere.get_material().set_specular(1)   

sphere2 = nvisii.entity.create(
    name="sphere2",
    mesh = nvisii.mesh.create_sphere("sphere2"),
    transform = nvisii.transform.create("sphere2"),
    material = nvisii.material.create("sphere2")
)
sphere2.get_transform().set_position((-0.5,-0.1,0.1))
sphere2.get_transform().set_scale((0.1, 0.1, 0.1))
sphere2.get_material().set_base_color((0.1,0.56,1))  
sphere2.get_material().set_roughness(0)   
sphere2.get_material().set_specular(0)   

sphere3 = nvisii.entity.create(
    name="sphere3",
    mesh = nvisii.mesh.create_sphere("sphere3"),
    transform = nvisii.transform.create("sphere3"),
    material = nvisii.material.create("sphere3")
)
sphere3.get_transform().set_position((0.6,-0.5,0.16))
sphere3.get_transform().set_scale((0.16, 0.16, 0.16))
sphere3.get_material().set_base_color((0.5,0.8,0.5))  
sphere3.get_material().set_roughness(0)   
sphere3.get_material().set_specular(1)   
sphere3.get_material().set_metallic(1)   

cone = nvisii.entity.create(
    name="cone",
    mesh = nvisii.mesh.create_cone("cone"),
    transform = nvisii.transform.create("cone"),
    material = nvisii.material.create("cone")
)
# lets set the cone up
cone.get_transform().set_position((0.08,0.35,0.2))
cone.get_transform().set_scale((0.3, 0.3, 0.3))
cone.get_material().set_base_color((245/255, 230/255, 66/255))  
cone.get_material().set_roughness(1)   
cone.get_material().set_specular(0)   
cone.get_material().set_metallic(0)   

# # # # # # # # # # # # # # # # # # # # # # # # #

nvisii.render_to_file(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    file_path=f"{opt.out}"
)

# let's clean up the GPU
nvisii.deinitialize()