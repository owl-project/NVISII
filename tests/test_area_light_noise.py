#%%
import sys, os, math
os.add_dll_directory(os.path.join(os.getcwd(), '..', 'install'))
sys.path.append(os.path.join(os.getcwd(), "..", "install"))

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import visii as v
#%%
v.initialize_interactive(window_on_top = True)

camera_entity = v.entity.create(
    name="my_camera_entity",
    transform=v.transform.create("my_camera_transform"),
    camera=v.camera.create_perspective_from_fov(name = "my_camera", field_of_view = 0.785398, aspect = 1.))
v.set_camera_entity(camera_entity)

#%%
from ipywidgets import interact
def moveCamera(x=3,y=3,z=2):
    camera_entity.get_transform().look_at(
        v.vec3(0,0,2.0),
        v.vec3(0,0,1),
        v.vec3(x,y,z),
    )
    # camera_entity.get_transform().set_position(0, 0.0, x)
interact(moveCamera, x=(-10, 10, .001), y=(-10, 10, .001), z=(-10, 10, .001))

#%%
tex = v.texture.create_from_image("texture", "../data/abandoned_tank_farm_01_1k.hdr")
v.set_dome_light_texture(tex)

#%%
floor = v.entity.create(
    name="floor",
    mesh = v.mesh.create_plane("floor"),
    transform = v.transform.create("floor"),
    material = v.material.create("floor")
)
#%%
mesh1 = v.entity.create(
    name="mesh1",
    mesh = v.mesh.create_from_obj("mesh2", "../data/dragon.obj"),
    transform = v.transform.create("mesh1"),
    material = v.material.create("mesh1")
)

#%%
mesh2 = v.entity.create(
    name="mesh2",
    mesh = v.mesh.create_sphere("sphere2"),
    transform = v.transform.create("mesh2"),
    material = v.material.create("mesh2")
)

# %%
mesh1.get_transform().set_scale(v.vec3(4))
mesh1.get_transform().set_rotation(v.angleAxis(1.57, v.vec3(1,0,0)))
mesh1.get_transform().set_position(v.vec3(0,0,1.0))

#%%
areaLight1 = v.entity.create(
    name="areaLight1",
    light = v.light.create("areaLight1"),
    transform = v.transform.create("areaLight1"),
    mesh = v.mesh.create_sphere("areaLight1"),
)
# %%
areaLight1.get_transform().set_scale(v.vec3(.25))
#%%
floor.get_transform().set_scale(v.vec3(1000))
mesh1.get_transform().set_scale(v.vec3(.5))
areaLight1.get_transform().set_position(v.vec3(0, 0, 4))
areaLight1.get_transform().set_scale(v.vec3(.5))
floor.get_material().set_roughness(1.0)
mesh1.get_material().set_base_color(v.vec3(1.0, 1.0, 1.0))
mesh2.get_material().set_base_color(v.vec3(1.0, 1.0, 1.0))
mesh1.get_transform().set_position(v.vec3(-1.0, 1.0, 0.0))
mesh2.get_transform().set_position(v.vec3(-1.0, 3.0, 1.0))

mesh1.get_transform().set_scale(v.vec3(4))
mesh1.get_transform().set_rotation(v.angleAxis(1.57, v.vec3(1,0,0)))
mesh1.get_transform().set_position(v.vec3(0,0,1.0))

#%%
#%%
def changeRoughness(roughness=0): 
    mesh1.get_material().set_roughness(roughness)
    mesh2.get_material().set_roughness(roughness)
def changeTransmission(transmission=1): 
    mesh1.get_material().set_transmission(transmission)    
    mesh2.get_material().set_transmission(transmission)    
def changeIor(ior=1.57): 
    mesh1.get_material().set_ior(ior)
    mesh2.get_material().set_ior(ior)
def changeSheen(sheen=0): 
    mesh1.get_material().set_sheen(sheen)
    mesh2.get_material().set_sheen(sheen)
def changeClearCoat(clearcoat=0): 
    mesh1.get_material().set_clearcoat(clearcoat)
    mesh2.get_material().set_clearcoat(clearcoat)
def changeClearCoatRoughness(clearcoat_roughness=0): 
    mesh1.get_material().set_clearcoat_roughness(clearcoat_roughness)
    mesh2.get_material().set_clearcoat_roughness(clearcoat_roughness)
def changeMetallic(metallic=0): 
    mesh1.get_material().set_metallic(metallic)
    mesh2.get_material().set_metallic(metallic)
def changeSpecularTint(specular_tint=0): 
    mesh1.get_material().set_specular_tint(specular_tint)
    mesh2.get_material().set_specular_tint(specular_tint)
def changeSpecular(specular=1): 
    mesh1.get_material().set_specular(specular)
    mesh2.get_material().set_specular(specular)
def changeSubsurface(subsurface=0): 
    mesh1.get_material().set_subsurface(subsurface)
    mesh2.get_material().set_subsurface(subsurface)
def changeTransmissionRoughess(transmission_roughness=0): 
    mesh1.get_material().set_transmission_roughness(transmission_roughness)
    mesh2.get_material().set_transmission_roughness(transmission_roughness)
interact(changeRoughness, roughness=(0.0, 1.0, .001))
interact(changeTransmission, transmission=(0.0, 1.0, .001))
interact(changeIor, ior=(0.0, 2.0, .001))
interact(changeSheen, sheen=(0.0, 1.0, .001))
interact(changeClearCoat, clearcoat=(0.0, 1.0, .001))
interact(changeClearCoatRoughness, clearcoat_roughness=(0.0, 1.0, .001))
interact(changeMetallic, metallic=(0.0, 1.0, .001))
interact(changeSpecularTint, specular_tint=(0.0, 1.0, .001))
interact(changeSpecular, specular=(0.0, 2.0, .001))
interact(changeSubsurface, subsurface=(0.0, 1.0, .001))
interact(changeTransmissionRoughess, transmission_roughness=(0.0, 1.0, .001))
#%%
def changeDomeLightIntensity(dome_intensity=0): v.set_dome_light_intensity(dome_intensity)
interact(changeDomeLightIntensity, dome_intensity=(0.0, 1.0, .001))
#%%
areaLight1.get_light().set_intensity(10000.)
#%%
def moveLight(x = 0, y = 0, z = 3): areaLight1.get_transform().set_position(v.vec3(x,y,z))
interact(moveLight, x=(0.0, 5.0, .001), y=(0.0, 5.0, .001), z=(-5.0, 5.0, .001))
def scaleLight(sx = 1, sy = 1., sz = 1): areaLight1.get_transform().set_scale(v.vec3(sx, sy, sz))
interact(scaleLight, sx=(0.0001, 1.0, .001), sy=(0.0001, 1.0, .001), sz=(0.0001, 1.0, .001))
def rotateLight(rx = 0, ry = 0., rz = 0): 
    areaLight1.get_transform().set_rotation(v.angleAxis(rx, v.vec3(1,0,0)))
    areaLight1.get_transform().add_rotation(v.angleAxis(ry, v.vec3(0,1,0)))
    areaLight1.get_transform().add_rotation(v.angleAxis(rz, v.vec3(0,0,1)))
interact(rotateLight, rx=(-3.14, 3.14, .001), ry=(-3.14, 3.14, .001), rz=(-3.14, 3.14, .001))

areaLight1.get_transform().set_scale(v.vec3(.25))
floor.get_transform().set_scale(v.vec3(100))
areaLight1.get_light().set_temperature(4000)
# %%

# %%
v.render_to_png(512,512,1024,"area_light_3.png")

# %%
v.enable_denoiser()

# %%


# %%

# %%
floor.get_material().set_base_color(v.vec3(1.0))
mesh1.get_material().set_base_color(v.vec3(1.0))
mesh2.get_material().set_base_color(v.vec3(1.0))

# %%
mesh1.get_material().set_base_color(v.vec3(1,0,0))

# %%
mesh2.get_material().set_base_color(v.vec3(0,1.,0))


# %%


teapot = v.entity.create(
    name="teapot",
    mesh = v.mesh.create_teapotahedron("teapot"),
    transform = v.transform.create("teapot"),
    material = v.material.create("teapot")
)

teapot.get_material().set_base_color(...)
teapot.get_material().set_metallic(...)
teapot.get_material().set_transmission(...)
teapot.get_material().set_roughness(...)
...


#%%
camera_entity.get_camera().set_aperture_diameter(10)
camera_entity.get_camera().set_focal_distance(6)


# %%
teapot = v.mesh.create_teapotahedron("test")

#%%
areaLight1.set_mesh(teapot)

# %%
tex2 = v.texture.create_from_image("grid", "../data/grid.jpg")

# %%
mat = v.material.create("test")

# %%
areaLight1.set_material(mat)

# %%
mat.set_base_color_texture(tex2)

# %%
pl = v.mesh.create_plane("temp")

# %%
areaLight1.set_mesh(pl)

# %%
#%%
import sys, os, math
os.add_dll_directory(os.path.join(os.getcwd(), '..', 'install'))
sys.path.append(os.path.join(os.getcwd(), "..", "install"))
# %%

import visii as v
v.initialize_interactive() # Or headless

camera_entity = v.entity.create(
    name = "my camera",
    transform = v.transform.create("my camera transform"),
    camera = v.camera.create_perspective_from_fov("my perspective",  field_of_view = 0.785398, aspect = 1.0)
)
camera_entity.get_transform().look_at(
    eye = v.vec3(3,3,3), at = v.vec3(0), up = v.vec3(0,0,1)
)

v.set_camera_entity(camera_entity)

my_mesh = v.entity.create(
    name = "my mesh",
    transform = v.transform.create("my mesh transform"),
    mesh = v.mesh.create_sphere("my mesh"),
    material = v.material.create("my material")
)

my_mesh.get_material().set_base_color(v.vec3(1, 0, 0))

v.render_to_png(width = 512, height = 512, samples_per_pixel = 1024, image_path = "my.png")




# %%
import visii

# %%
v.render_data_to_png(width = 512, height = 512, start_frame = 0, frame_count = 64, bounce = 0, options = "denoise_normal", image_path = "test.png")

# %%
v.render_data_to_png(width = 512, height = 512, start_frame = 0, frame_count = 64, bounce = 0, options = "denoise_albedo", image_path = "test.png")


# %%
