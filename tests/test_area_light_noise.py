#%%
import sys, os, math
os.add_dll_directory(os.path.join(os.getcwd(), '..', 'install'))
sys.path.append(os.path.join(os.getcwd(), "..", "install"))

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import visii
#%%
visii.initialize_interactive(window_on_top = True)

camera_entity = visii.entity.create(
    name="my_camera_entity",
    transform=visii.transform.create("my_camera_transform"),
    camera=visii.camera.create_perspective_from_fov(name = "my_camera", field_of_view = 0.785398, aspect = 1., near = .1))
visii.set_camera_entity(camera_entity)
camera_entity.get_transform().set_position(0, 0.0, -5.)
camera_entity.get_camera().use_perspective_from_fov(0.785398, 1.0, .01)
camera_entity.get_camera().set_view(
    visii.lookAt(
        visii.vec3(2,2,2),
        visii.vec3(0,0,.5),
        visii.vec3(0,0,1),
    )
)
#%%
from ipywidgets import interact
def moveCamera(x=3,y=3,z=2):
    camera_entity.get_camera().set_view(
        visii.lookAt(
            visii.vec3(x,y,z),
            visii.vec3(0,0,.5),
            visii.vec3(0,0,1),
        )
    )
    # camera_entity.get_transform().set_position(0, 0.0, x)
interact(moveCamera, x=(-10, 10, .001), y=(-10, 10, .001), z=(-10, 10, .001))

#%%
floor = visii.entity.create(
    name="floor",
    mesh = visii.mesh.create_plane("floor"),
    transform = visii.transform.create("floor"),
    material = visii.material.create("floor")
)
#%%
mesh1 = visii.entity.create(
    name="mesh1",
    mesh = visii.mesh.create_sphere("sphere1", 1, 128, 128),
    transform = visii.transform.create("mesh1"),
    material = visii.material.create("mesh1")
)

mesh2 = visii.entity.create(
    name="mesh2",
    mesh = visii.mesh.create_sphere("sphere2", 1, 128, 128),
    transform = visii.transform.create("mesh2"),
    material = visii.material.create("mesh2")
)

#%%
areaLight1 = visii.entity.create(
    name="areaLight1",
    light = visii.light.create("areaLight1"),
    transform = visii.transform.create("areaLight1"),
    mesh = visii.mesh.create_sphere("areaLight1"),
)
# %%
areaLight1.get_transform().set_scale(.25)
#%%
floor.get_transform().set_scale(1000)
# mesh1.get_transform().set_scale(.5)
areaLight1.get_transform().set_position(0, 0, 4)
areaLight1.get_transform().set_scale(.5)
floor.get_material().set_roughness(1.0)
mesh1.get_material().set_base_color(1.0, 1.0, 1.0)
mesh2.get_material().set_base_color(1.0, 1.0, 1.0)
mesh1.get_transform().set_position(-1.0, 1.0, 1.0)
mesh2.get_transform().set_position(1.0, -1.0, 1.0)
def changeRoughness(roughness=0): 
    mesh1.get_material().set_roughness(roughness)
    mesh2.get_material().set_roughness(roughness)
def changeTransmission(transmission=0): 
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
#%%
def changeDomeLightIntensity(dome_intensity=0): visii.set_dome_light_intensity(dome_intensity)
interact(changeDomeLightIntensity, dome_intensity=(0.0, 1.0, .001))
#%%
areaLight1.get_light().set_intensity(100000.)
#%%
def moveLight(x = 0, y = 0, z = 3): areaLight1.get_transform().set_position(x,y,z)
interact(moveLight, x=(0.0, 5.0, .001), y=(0.0, 5.0, .001), z=(-5.0, 5.0, .001))
def scaleLight(sx = 1, sy = 1., sz = 1): areaLight1.get_transform().set_scale(sx, sy, sz)
interact(scaleLight, sx=(0.0001, 1.0, .001), sy=(0.0001, 1.0, .001), sz=(0.0001, 1.0, .001))
def rotateLight(rx = 0, ry = 0., rz = 0): 
    areaLight1.get_transform().set_rotation(rx, visii.vec3(1,0,0))
    areaLight1.get_transform().add_rotation(ry, visii.vec3(0,1,0))
    areaLight1.get_transform().add_rotation(rz, visii.vec3(0,0,1))
interact(rotateLight, rx=(-3.14, 3.14, .001), ry=(-3.14, 3.14, .001), rz=(-3.14, 3.14, .001))

areaLight1.get_transform().set_scale(.25)
floor.get_transform().set_scale(100)
areaLight1.get_light().set_temperature(4000)
# %%

# %%
visii.render_to_png(1024,1024,4096,"test_area_light_noise.png")

# %%
