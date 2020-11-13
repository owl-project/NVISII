import visii
import random

opt = lambda : None
opt.spp = 256 
opt.width = 500
opt.height = 500 
opt.out = "06_textures.png"

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
    eye = (-2,0,.5),
)
visii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

visii.set_dome_light_intensity(3)

# load the textures
dome = visii.texture.create_from_file("dome", "content/kiara_4_mid-morning_4k.hdr")
tex = visii.texture.create_from_file("tex",'content/photos_2020_5_11_fst_gray-wall-grunge.jpg')

# Textures can be mixed and altered. 
# Checkout create_hsv, create_add, create_multiply, and create_mix
floor_tex = visii.texture.create_hsv("floor", tex, 
    hue = 0, saturation = .5, value = 1.0, mix = 1.0)

# we can add HDR images to act as a dome that lights up our scene

# use "enable_cdf" for dome light textures that contain 
# bright objects that cast shadows (like the sun). Note
# that this has a significant impact on rendering performance,
# and is disabled by default.
visii.set_dome_light_texture(dome, enable_cdf = True)
visii.set_dome_light_rotation(visii.angleAxis(visii.pi() * .1, (0,0,1)))

# Lets set some objects in the scene
entity = visii.entity.create(
    name = "floor",
    mesh = visii.mesh.create_plane("mesh_floor"),
    transform = visii.transform.create("transform_floor"),
    material = visii.material.create("material_floor")
)
entity.get_transform().set_scale((1,1,1))
mat = visii.material.get("material_floor")

mat.set_roughness(.5)

# Lets set the base color and roughness of the object to use a texture. 
# but the textures could also be used to set other
# material propreties
mat.set_base_color_texture(floor_tex)
mat.set_roughness_texture(tex)

# # # # # # # # # # # # # # # # # # # # # # # # #

knot = visii.entity.create(
    name="knot",
    mesh = visii.mesh.create_torus_knot("knot"),
    transform = visii.transform.create("knot"),
    material = visii.material.create("knot")
)
knot.get_transform().set_position((0,0,.5))
knot.get_transform().set_scale((0.2, 0.2, 0.2))
knot.get_material().set_base_color((1,1,1))  
knot.get_material().set_roughness(0)   
knot.get_material().set_metallic(1)   

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