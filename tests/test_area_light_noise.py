#%%
<<<<<<< Updated upstream
import sys, os, math
=======
import sys, os, math, colorsys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
tex = v.texture.create_from_image("texture", "../data/abandoned_tank_farm_01_1k.hdr")
v.set_dome_light_texture(tex)
=======
def moveCameraNext(x=0,y=3,z=2):
    camera_entity.get_transform().next_look_at(
        v.vec3(0,0,0.0),
        v.vec3(0,0,1),
        v.vec3(x,y,z),
    )
    # camera_entity.get_transform().set_position(0, 0.0, x)
interact(moveCameraNext, x=(-10, 10, .001), y=(-10, 10, .001), z=(-10, 10, .001))

#%%
# tex = v.texture.create_from_image("texture", "../data/dome.hdr")
# v.set_dome_light_texture(tex)
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
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
=======
def changeColor(hue=0, sat=0, val=1): 
    rgb = colorsys.hsv_to_rgb(hue, sat, val)
    outer.get_material().set_base_color(v.vec3(rgb[0], rgb[1], rgb[2]))
def changeRoughness(roughness=1): outer.get_material().set_roughness(roughness)
def changeTransmission(transmission=0): outer.get_material().set_transmission(transmission)    
def changeIor(ior=1.57): outer.get_material().set_ior(ior)
def changeSheen(sheen=0): outer.get_material().set_sheen(sheen)
def changeClearCoat(clearcoat=0): outer.get_material().set_clearcoat(clearcoat)
def changeClearCoatRoughness(clearcoat_roughness=0): outer.get_material().set_clearcoat_roughness(clearcoat_roughness)
def changeMetallic(metallic=0): outer.get_material().set_metallic(metallic)
def changeSpecularTint(specular_tint=0): outer.get_material().set_specular_tint(specular_tint)
def changeSpecular(specular=1): outer.get_material().set_specular(specular)
def changeSubsurface(subsurface=0): outer.get_material().set_subsurface(subsurface)
def changeTransmissionRoughess(transmission_roughness=0): outer.get_material().set_transmission_roughness(transmission_roughness)
def changeAnisotropy(anisotropy=0): outer.get_material().set_anisotropic(anisotropy)
interact(changeColor, hue=(0.0, 1.0, .001), sat=(0.0, 1.0, .001), val=(0.0, 1.0, .001))
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
interact(changeAnisotropy, anisotropy=(0.0, 1.0, .001))
#%%
def changeColor(hue=0, sat=0, val=1): 
    rgb = colorsys.hsv_to_rgb(hue, sat, val)
    inner.get_material().set_base_color(v.vec3(rgb[0], rgb[1], rgb[2]))
def changeRoughness(roughness=1): inner.get_material().set_roughness(roughness)
def changeTransmission(transmission=0): inner.get_material().set_transmission(transmission)    
def changeIor(ior=1.57): inner.get_material().set_ior(ior)
def changeSheen(sheen=0): inner.get_material().set_sheen(sheen)
def changeClearCoat(clearcoat=0): inner.get_material().set_clearcoat(clearcoat)
def changeClearCoatRoughness(clearcoat_roughness=0): inner.get_material().set_clearcoat_roughness(clearcoat_roughness)
def changeMetallic(metallic=0): inner.get_material().set_metallic(metallic)
def changeSpecularTint(specular_tint=0): inner.get_material().set_specular_tint(specular_tint)
def changeSpecular(specular=1): inner.get_material().set_specular(specular)
def changeSubsurface(subsurface=0): inner.get_material().set_subsurface(subsurface)
def changeTransmissionRoughess(transmission_roughness=0): inner.get_material().set_transmission_roughness(transmission_roughness)
def changeAnisotropy(anisotropy=0): inner.get_material().set_anisotropic(anisotropy)
interact(changeColor, hue=(0.0, 1.0, .001), sat=(0.0, 1.0, .001), val=(0.0, 1.0, .001))
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
#%%
def changeDomeLightIntensity(dome_intensity=0): v.set_dome_light_intensity(dome_intensity)
interact(changeDomeLightIntensity, dome_intensity=(0.0, 1.0, .001))
#%%
areaLight1.get_light().set_intensity(10000.)
=======
interact(changeAnisotropy, anisotropy=(0.0, 1.0, .001))
interact(changeColor, hue=(0.0, 1.0, .001), sat=(0.0, 1.0, .001), val=(0.0, 1.0, .001))


#%%
def changeLinearVelocity(lx=0, ly=0, lz=0): 
    inner.get_transform().set_linear_velocity(v.vec3(lx,ly,lz))
    outer.get_transform().set_linear_velocity(v.vec3(lx,ly,lz))
interact(changeLinearVelocity, lx=(-1.0, 1.0, .001), ly=(-1.0, 1.0, .001), lz=(-1.0, 1.0, .001))

def changeScalarVelocity(sx=0, sy=0, sz=0): 
    inner.get_transform().set_scalar_velocity(v.vec3(sx,sy,sz))
    outer.get_transform().set_scalar_velocity(v.vec3(sx,sy,sz))
interact(changeScalarVelocity, sx=(-1.0, 1.0, .001), sy=(-1.0, 1.0, .001), sz=(-1.0, 1.0, .001))

def changeAngularVelocity(ax=0, ay=0, az=0): 
    q = v.quat(1,0,0,0)
    q = v.angleAxis(ax, v.vec3(1,0,0)) * q
    q = v.angleAxis(ay, v.vec3(0,1,0)) * q
    q = v.angleAxis(az, v.vec3(0,0,1)) * q
    inner.get_transform().set_angular_velocity(q)
    outer.get_transform().set_angular_velocity(q)
interact(changeAngularVelocity, ax=(-1.0, 1.0, .001), ay=(-1.0, 1.0, .001), az=(-1.0, 1.0, .001))

def changePosition(x=0, y=0, z=0): 
    inner.get_transform().set_position(v.vec3(x,y,z))
    outer.get_transform().set_position(v.vec3(x,y,z))
interact(changePosition, x=(-1.0, 1.0, .001), y=(-1.0, 1.0, .001), z=(-1.0, 1.0, .001))

def changeScale(sx=1, sy=1, sz=1): 
    inner.get_transform().set_scale(v.vec3(sx,sy,sz))
    outer.get_transform().set_scale(v.vec3(sx,sy,sz))
interact(changeScale, sx=(-1.0, 1.0, .001), sy=(-1.0, 1.0, .001), sz=(-1.0, 1.0, .001))

def changeRotation(ax=3.14 * .5, ay=0, az=0): 
    q = v.quat(1,0,0,0)
    q = v.angleAxis(ax, v.vec3(1,0,0)) * q
    q = v.angleAxis(ay, v.vec3(0,1,0)) * q
    q = v.angleAxis(az, v.vec3(0,0,1)) * q
    inner.get_transform().set_rotation(q)
    outer.get_transform().set_rotation(q)
interact(changeRotation, ax=(-3.14, 3.14, .001), ay=(-3.14, 3.14, .001), az=(-3.14, 3.14, .001))

# v.sample_time_interval(v.vec2(1))
# v.sample_pixel_area(v.vec2(.5), v.vec2(.5))
#%%
def changeDomeLightIntensity(dome_intensity=1): v.set_dome_light_intensity(dome_intensity)
interact(changeDomeLightIntensity, dome_intensity=(0.0, 4.0, .001))
#%%
def changeLightIntensity(intensity=100): light.get_light().set_intensity(intensity)
interact(changeLightIntensity, intensity=(0.0, 1000.0, .001))

>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======
glasses = v.entity.create(
    name = "glasses",
    mesh = v.mesh.create_from_obj(name = "glasses", path="C:/Users/natevm/3D Objects/glasses/LE_V1_0131_Ikea_Glass_Pokal.obj"),
    material = v.material.create(name = "glasses", base_color = v.vec3(1), roughness = 0, transmission = 1),
    transform = v.transform.create(name = "glasses")
)

# %%
glasses.get_transform().set_position(v.vec3(0,0,0.1))
glasses.get_transform().set_rotation(v.angleAxis(3.14 * .5, v.vec3(1,0,0)))
glasses.get_transform().set_scale(v.vec3(.2))
glasses.get_material().set_base_color(v.vec3(1, 1, 1))
# %%
mesh1.get_transform().set_position(v.vec3(-8, 2, 1))

# %%
floor.get_material().set_roughness(1.0)
floor.get_material().set_metallic(0)



# %%
mesh2.get_material().set_normal_map_texture(normtex)
floor.get_material().set_normal_map_texture(normtex)


# %%
v.set_indirect_lighting_clamp(0.001)

# %%
mesh2.get_transform().set_rotation(v.angleAxis(3.14, v.vec3(1,0,0)))

# %%
floor.get_material().clear_normal_map_texture()

# %%
inner.set_light(light.get_light())

# %%
inner.clear_light()

# %%
v.set_direct_lighting_clamp(10.0)
v.set_indirect_lighting_clamp(10.0)
# %%
orm = v.texture.create_from_image("ORM", "../data/RustedMetal_ORM.png", linear=True)

# %%
norm = v.texture.create_from_image("N", "../data/RustedMetal_N.png", linear=True)


# %%
base = v.texture.create_from_image("Base", "../data/RustedMetal_BaseColor.png", linear=True)

# %%
outer.get_material().set_base_color_texture(base)
outer.get_material().set_normal_map_texture(norm2)
outer.get_material().set_roughness_texture(orm, channel=1)
outer.get_material().set_metallic_texture(orm, channel=2)

# %%

# %%

# %%
outer.get_material().clear_normal_map_texture()

# %%
norm2 = v.texture.create_from_image("N2", "../data/normal.png", linear=True)

# %%
norm2 = v.texture.create_from_image("N4", "../data/RustedMetal_N.png", linear=True)

# %%
outer.get_material().set_normal_map_texture(norm2)

# %%
>>>>>>> Stashed changes
