import visii

opt = lambda: None
opt.spp = 100 
opt.width = 1280
opt.height = 720 
opt.noise = False
opt.path_obj = 'content/dragon/dragon.obj'
opt.out = '11_instance_motion_blur.png'

# # # # # # # # # # # # # # # # # # # # # # # # #
visii.initialize(headless=True, verbose=True)

if not opt.noise is True: 
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
    eye = (0,3.0,0.2),
)
visii.set_camera_entity(camera)

visii.set_dome_light_sky(sun_position = (10, 10, 1), saturation = 2)
visii.set_dome_light_intensity(1.5)

# # # # # # # # # # # # # # # # # # # # # # # # #

floor = visii.entity.create(
    name = "floor",
    mesh = visii.mesh.create_plane("floor", size = (10,10)),
    material = visii.material.create("floor", base_color = (.5, .5, .5), roughness = 0.0, metallic = 1.0),
    transform = visii.transform.create("floor", position = (0,0,-.3))
)

# Next, let's load an obj
mesh = visii.mesh.create_from_file("obj", opt.path_obj)

# Now, lets make three instances of that mesh
obj1 = visii.entity.create(
    name="obj1",
    mesh = mesh,
    transform = visii.transform.create("obj1"),
    material = visii.material.create("obj1")
)

obj2 = visii.entity.create(
    name="obj2",
    mesh = mesh,
    transform = visii.transform.create("obj2"),
    material = visii.material.create("obj2")
)

obj3 = visii.entity.create(
    name="obj3",
    mesh = mesh,
    transform = visii.transform.create("obj3"),
    material = visii.material.create("obj3")
)

obj4 = visii.entity.create(
    name="obj4",
    mesh = mesh,
    transform = visii.transform.create("obj4"),
    material = visii.material.create("obj4")
)

# place those objects into the scene

# lets set the obj_entity up
obj1.get_transform().set_position((-1.5, 0, 0))
obj1.get_transform().set_rotation((0.7071, 0, 0, 0.7071))
obj1.get_material().set_base_color((1,0,0))  
obj1.get_material().set_roughness(0.7)   
obj1.get_material().set_specular(1)   
obj1.get_material().set_sheen(1)

obj2.get_transform().set_position((-.5, 0, 0))
obj2.get_transform().set_rotation((0.7071, 0, 0, 0.7071))
obj2.get_material().set_base_color((0,1,0))  
obj2.get_material().set_roughness(0.7)   
obj2.get_material().set_specular(1)   
obj2.get_material().set_sheen(1)

obj3.get_transform().set_position((.5, 0, 0))
obj3.get_transform().set_rotation((0.7071, 0, 0, 0.7071))
obj3.get_material().set_base_color((0,0,1))  
obj3.get_material().set_roughness(0.7)   
obj3.get_material().set_specular(1)   
obj3.get_material().set_sheen(1)

obj4.get_transform().set_position((1.5, 0, 0))
obj4.get_transform().set_rotation((0.7071, 0, 0, 0.7071))
obj4.get_material().set_base_color((.5,.5,.5))  
obj4.get_material().set_roughness(0.7)   
obj4.get_material().set_specular(1)   
obj4.get_material().set_sheen(1)


# Use linear motion blur on the first object...
obj1.get_transform().set_linear_velocity((.0, .0, .2))

# angular motion blur on the second object...
obj2.get_transform().set_angular_velocity((visii.pi() / 16, visii.pi() / 16, visii.pi() / 16, 1))

# and scalar motion blur on the third object
obj3.get_transform().set_scalar_velocity((-.5, -.5, -.5))

# # # # # # # # # # # # # # # # # # # # # # # # #

visii.render_to_png(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    image_path=f"{opt.out}"
)

# let's clean up the GPU
visii.deinitialize()