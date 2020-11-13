import visii
import noise
import random
import numpy as np 

opt = lambda : None
opt.spp = 512 
opt.width = 1024
opt.height = 1024 
opt.out = "08_obj_scene_loader.png"

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
    at = (-5,0,12), # look at (world coordinate)
    up = (0,0,1), # up vector
    eye = (5,-15,18)
)
visii.set_camera_entity(camera)

visii.disable_dome_light_sampling()

# # # # # # # # # # # # # # # # # # # # # # # # #

sdb = visii.import_scene(
    file_path = 'content/salle_de_bain_separated/salle_de_bain_separated.obj',
    position = (1,0,0),
    scale = (1.0, 1.0, 1.0),
    rotation = visii.angleAxis(3.14 * .5, (1,0,0)),
    args = ["verbose"] # list assets as they are loaded
)

# Using the above function, 
# visii loads each obj model as its own entity.
# You can find them by name (with an optional prefix added
# to front of each generated component name)

# visii generates the same naming pattern for the different 
# materials defined in the mtl file

# since obj/mtl do not have definition for metallic properties 
# lets add them manually to the material
mirror = visii.material.get('Mirror')

mirror.set_roughness(0)
mirror.set_metallic(1)
mirror.set_base_color((1,1,1))

# When loading the obj scene, visii returns a list of entities 
# you can loop them to find specific objects or add physical 
# collisions. 

# Since obj/mtl do not define lights, lets add one to the mesh 
# named light
for i_s, s in enumerate(sdb.entities):
    if "light" in s.get_name().lower():
        s.set_light(visii.light.create('light'))
        s.get_light().set_intensity(10)
        s.get_light().set_exposure(18)
        s.get_light().set_temperature(5000)
        s.clear_material()

# # # # # # # # # # # # # # # # # # # # # # # # #

visii.render_to_file(
    width=opt.width, 
    height=opt.height, 
    samples_per_pixel=opt.spp,
    file_path=opt.out
)

# let's clean up the GPU
visii.deinitialize()