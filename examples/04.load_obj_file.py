import visii

opt = lambda : None
opt.nb_objects = 50
opt.spp = 256 
opt.width = 500
opt.height = 500 
opt.out = "04_load_obj_file.png" 

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
    at = (0,0.1,0.1),
    up = (0,0,1),
    eye = (1,0.7,0.2),
)
visii.set_camera_entity(camera)

visii.set_dome_light_sky(sun_position = (10, 10, 1), saturation = 2)
visii.set_dome_light_exposure(1)

# # # # # # # # # # # # # # # # # # # # # # # # #

# This function loads a signle obj mesh. It ignores 
# the associated .mtl file
mesh = visii.mesh.create_from_file("obj", "./content/dragon/dragon.obj")

obj_entity = visii.entity.create(
    name="obj_entity",
    mesh = mesh,
    transform = visii.transform.create("obj_entity"),
    material = visii.material.create("obj_entity")
)

# lets set the obj_entity up
obj_entity.get_transform().set_rotation( 
    (0.7071, 0, 0, 0.7071)
)
obj_entity.get_material().set_base_color(
    (0.9,0.12,0.08)
)  
obj_entity.get_material().set_roughness(0.7)   
obj_entity.get_material().set_specular(1)   
obj_entity.get_material().set_sheen(1)


# # # # # # # # # # # # # # # # # # # # # # # # #

visii.render_to_file(
    width=opt.width, 
    height=opt.height, 
    samples_per_pixel=opt.spp,
    file_path=opt.out 
)

# let's clean up GPU resources
visii.deinitialize()