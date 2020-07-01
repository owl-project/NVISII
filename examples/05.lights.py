import visii
import random
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--spp', 
                    default=100,
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

opt = parser.parse_args()

# # # # # # # # # # # # # # # # # # # # # # # # #
visii.initialize_headless()

if not opt.noise is True: 
    visii.enable_denoiser()

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
    visii.vec3(-2,0,1), # camera_origin    
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
obj_entity.get_light().set_intensity(200000)
obj_entity.get_light().set_temperature(8000)

#lets set the size and placement of the light
obj_entity.get_transform().set_scale(
    visii.vec3(0.2)
)

#light above the scene
obj_entity.get_transform().set_position(
    visii.vec3(0,0,2)
)

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
obj_entity.get_light().set_intensity(70000)

# you can also set the light color manually
obj_entity.get_light().set_color(
    visii.vec3(1,0,0)
)
#lets set the size and placement of the light
obj_entity.get_transform().set_scale(
    visii.vec3(0.1)
)
obj_entity.get_transform().set_position(
    visii.vec3(0,-0.6,0.4)
)
obj_entity.get_transform().set_rotation(
    visii.angleAxis(90, visii.vec3(0,0,1))
)

# third light 
obj_entity = visii.entity.create(
    name="light_3",
    mesh = visii.mesh.create_plane('light_3'),
    transform = visii.transform.create("light_3"),
)
obj_entity.set_light(
    visii.light.create('light_3')
)
obj_entity.get_light().set_intensity(10000)

obj_entity.get_light().set_color(
    visii.vec3(0,1,1)
)
obj_entity.get_transform().set_scale(
    visii.vec3(0.2)
)
obj_entity.get_transform().set_position(
    visii.vec3(0.5,0.5,0.7)
)

# # # # # # # # # # # # # # # # # # # # # # # # #

# Lets set some objects in the scene
entity = visii.entity.create(
    name = "floor",
    mesh = visii.mesh.create_plane("mesh_floor"),
    transform = visii.transform.create("transform_floor"),
    material = visii.material.create("material_floor")
)
entity.get_transform().set_scale(visii.vec3(100))
mat = visii.material.get("material_floor")
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
    visii.vec3(0.6,-0.5,0.1))
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
box1.get_material().set_metallic(0)   
box1.get_material().set_transmission(1)   

# lets make the plane light look at the box
obj_entity = visii.entity.get("light_3")
obj_entity.get_transform().look_at(
    box1.get_transform().get_position(),
    # visii.vec3(-0.5,0.5,0.1),
    visii.vec3(0,0,1),
)

# The plane has to be flipped
obj_entity.get_transform().add_rotation(visii.quat(0,0,1,0))


#%%
# # # # # # # # # # # # # # # # # # # # # # # # #

visii.render_to_png(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    image_path=f"{opt.out}"
)

# let's clean up the GPU
visii.cleanup()