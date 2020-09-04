#%%
import sys, os, math, colorsys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys, os, math, colorsys
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

#%%
tex = v.texture.create_from_image("dome", "../data/dome.hdr")
v.set_dome_light_texture(tex)
v.set_dome_light_intensity(5)

#%%
entities_obj = v.import_obj(
    "", # prefix name
    'C:/Users/natevm/3D Objects/CrytekSponza/sponza_fixed.obj', #obj path
    'C:/Users/natevm/3D Objects/CrytekSponza/', # mtl folder 
    v.vec3(0,0,0), # translation 
    v.vec3(1), # scale here
    v.angleAxis(3.14 * .5, v.vec3(1,0,0)) #rotation here
)

#%%
# from ipywidgets import interact
offset = 0
zoffset = 2
def moveCamera(x=10,y=offset,z=zoffset):
    camera_entity.get_transform().look_at(
        v.vec3(0,offset,zoffset),
        v.vec3(0,0,1),
        v.vec3(x,y,z),
        previous = False
    )
    camera_entity.get_transform().clear_motion()
    # camera_entity.get_transform().set_position(0, 0.0, x)
moveCamera()
interact(moveCamera, x=(-10, 10, .001), y=(-10, 10, .001), z=(-10, 10, .001))
#%%
camera_entity.get_camera().use_perspective_from_fov(field_of_view = 0.785398, aspect = 1920.0/1080.0)

# for i in range(len(entities_obj)):
#     if i == 5: continue
#     entities_obj[i].clear_material()


#%%
v.set_dome_light_intensity(10)
v.set_max_bounce_depth(2)
v.set_direct_lighting_clamp(1000)
v.set_indirect_lighting_clamp(1000)
# v.render_to_png(512, 512, 10000, "is.png")

#%%
v.enable_denoiser()

#%%
# # Light 
# light = entities_obj[0]
# light.set_light(v.light.create("areaLight1"),)

# #%% Light 
# floor = entities_obj[1]

# #%% LTE Logo 
# lteLogo = entities_obj[2]

# #%% outer
# outer = entities_obj[3]

# #%% inner
# inner = entities_obj[4]

# #%%
# def changeColor(hue=0, sat=1, val=1): 
#     rgb = colorsys.hsv_to_rgb(hue, sat, val)
#     inner.get_material().set_base_color(v.vec3(rgb[0], rgb[1], rgb[2]))
# def changeRoughness(roughness=0): inner.get_material().set_roughness(roughness)
# def changeTransmission(transmission=0): inner.get_material().set_transmission(transmission)    
# def changeIor(ior=1.57): inner.get_material().set_ior(ior)
# def changeSheen(sheen=0): inner.get_material().set_sheen(sheen)
# def changeClearCoat(clearcoat=0): inner.get_material().set_clearcoat(clearcoat)
# def changeClearCoatRoughness(clearcoat_roughness=0): inner.get_material().set_clearcoat_roughness(clearcoat_roughness)
# def changeMetallic(metallic=1): inner.get_material().set_metallic(metallic)
# def changeSpecularTint(specular_tint=0): inner.get_material().set_specular_tint(specular_tint)
# def changeSpecular(specular=1): inner.get_material().set_specular(specular)
# def changeSubsurface(subsurface=0): inner.get_material().set_subsurface(subsurface)
# def changeTransmissionRoughess(transmission_roughness=0): inner.get_material().set_transmission_roughness(transmission_roughness)
# def changeAnisotropy(anisotropy=0): inner.get_material().set_anisotropic(anisotropy)
# interact(changeColor, hue=(0.0, 1.0, .001), sat=(0.0, 1.0, .001), val=(0.0, 1.0, .001))
# interact(changeRoughness, roughness=(0.0, 1.0, .001))
# interact(changeTransmission, transmission=(0.0, 1.0, .001))
# interact(changeIor, ior=(0.0, 2.0, .001))
# interact(changeSheen, sheen=(0.0, 1.0, .001))
# interact(changeClearCoat, clearcoat=(0.0, 1.0, .001))
# interact(changeClearCoatRoughness, clearcoat_roughness=(0.0, 1.0, .001))
# interact(changeMetallic, metallic=(0.0, 1.0, .001))
# interact(changeSpecularTint, specular_tint=(0.0, 1.0, .001))
# interact(changeSpecular, specular=(0.0, 2.0, .001))
# interact(changeSubsurface, subsurface=(0.0, 1.0, .001))
# interact(changeTransmissionRoughess, transmission_roughness=(0.0, 1.0, .001))
# interact(changeAnisotropy, anisotropy=(0.0, 1.0, .001))
# #%%
# def changeColor(hue=0, sat=1, val=1): 
#     rgb = colorsys.hsv_to_rgb(hue, sat, val)
#     outer.get_material().set_base_color(v.vec3(rgb[0], rgb[1], rgb[2]))
# def changeRoughness(roughness=1): outer.get_material().set_roughness(roughness)
# def changeTransmission(transmission=0): outer.get_material().set_transmission(transmission)    
# def changeIor(ior=1.57): outer.get_material().set_ior(ior)
# def changeSheen(sheen=0): outer.get_material().set_sheen(sheen)
# def changeClearCoat(clearcoat=0): outer.get_material().set_clearcoat(clearcoat)
# def changeClearCoatRoughness(clearcoat_roughness=0): outer.get_material().set_clearcoat_roughness(clearcoat_roughness)
# def changeMetallic(metallic=0): outer.get_material().set_metallic(metallic)
# def changeSpecularTint(specular_tint=0): outer.get_material().set_specular_tint(specular_tint)
# def changeSpecular(specular=1): outer.get_material().set_specular(specular)
# def changeSubsurface(subsurface=0): outer.get_material().set_subsurface(subsurface)
# def changeTransmissionRoughess(transmission_roughness=0): outer.get_material().set_transmission_roughness(transmission_roughness)
# def changeAnisotropy(anisotropy=0): outer.get_material().set_anisotropic(anisotropy)
# interact(changeColor, hue=(0.0, 1.0, .001), sat=(0.0, 1.0, .001), val=(0.0, 1.0, .001))
# interact(changeRoughness, roughness=(0.0, 1.0, .001))
# interact(changeTransmission, transmission=(0.0, 1.0, .001))
# interact(changeIor, ior=(0.0, 2.0, .001))
# interact(changeSheen, sheen=(0.0, 1.0, .001))
# interact(changeClearCoat, clearcoat=(0.0, 1.0, .001))
# interact(changeClearCoatRoughness, clearcoat_roughness=(0.0, 1.0, .001))
# interact(changeMetallic, metallic=(0.0, 1.0, .001))
# interact(changeSpecularTint, specular_tint=(0.0, 1.0, .001))
# interact(changeSpecular, specular=(0.0, 2.0, .001))
# interact(changeSubsurface, subsurface=(0.0, 1.0, .001))
# interact(changeTransmissionRoughess, transmission_roughness=(0.0, 1.0, .001))
# interact(changeAnisotropy, anisotropy=(0.0, 1.0, .001))
# #%%
# def changeLinearVelocity(lx=0, ly=0, lz=0): 
#     inner.get_transform().set_linear_velocity(v.vec3(lx,ly,lz))
#     outer.get_transform().set_linear_velocity(v.vec3(lx,ly,lz))
# interact(changeLinearVelocity, lx=(-1.0, 1.0, .001), ly=(-1.0, 1.0, .001), lz=(-1.0, 1.0, .001))

# def changeScalarVelocity(sx=0, sy=0, sz=0): 
#     inner.get_transform().set_scalar_velocity(v.vec3(sx,sy,sz))
#     outer.get_transform().set_scalar_velocity(v.vec3(sx,sy,sz))
# interact(changeScalarVelocity, sx=(-1.0, 1.0, .001), sy=(-1.0, 1.0, .001), sz=(-1.0, 1.0, .001))

# def changeAngularVelocity(ax=0, ay=0, az=0): 
#     q = v.quat(1,0,0,0)
#     q = v.angleAxis(ax, v.vec3(1,0,0)) * q
#     q = v.angleAxis(ay, v.vec3(0,1,0)) * q
#     q = v.angleAxis(az, v.vec3(0,0,1)) * q
#     inner.get_transform().set_angular_velocity(q)
#     outer.get_transform().set_angular_velocity(q)
# interact(changeAngularVelocity, ax=(-1.0, 1.0, .001), ay=(-1.0, 1.0, .001), az=(-1.0, 1.0, .001))

# #%%
# def changeDomeLightIntensity(dome_intensity=1): v.set_dome_light_intensity(dome_intensity)
# interact(changeDomeLightIntensity, dome_intensity=(0.0, 4.0, .001))
# #%%
# def changeLightIntensity(intensity=0): light.get_light().set_intensity(intensity)
# interact(changeLightIntensity, intensity=(0.0, 1000.0, .001))

# #%%
# def moveLight(x = 0, y = 0, z = 3): light.get_transform().set_position(v.vec3(x,y,z))
# interact(moveLight, x=(-5.0, 5.0, .001), y=(-5.0, 5.0, .001), z=(-5.0, 5.0, .001))
# def scaleLight(sx = 1, sy = 1., sz = 1): light.get_transform().set_scale(v.vec3(sx, sy, sz))
# interact(scaleLight, sx=(0.0001, 1.0, .001), sy=(0.0001, 1.0, .001), sz=(0.0001, 1.0, .001))
# def rotateLight(rx = 1.57, ry = 0., rz = 0): 
#     light.get_transform().set_rotation(v.angleAxis(rx, v.vec3(1,0,0)))
#     light.get_transform().add_rotation(v.angleAxis(ry, v.vec3(0,1,0)))
#     light.get_transform().add_rotation(v.angleAxis(rz, v.vec3(0,0,1)))
# interact(rotateLight, rx=(-3.14, 3.14, .001), ry=(-3.14, 3.14, .001), rz=(-3.14, 3.14, .001))

# # light.get_transform().set_scale(v.vec3(.25))
# # floor.get_transform().set_scale(v.vec3(100))
# light.get_light().set_temperature(8000)


input()