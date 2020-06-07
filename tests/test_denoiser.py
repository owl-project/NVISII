#%%
import sys, os, math
os.add_dll_directory(os.path.join(os.getcwd(), '..', 'install'))
sys.path.append(os.path.join(os.getcwd(), "..", "install"))

import visii

#%%
visii.initialize_interactive(window_on_top = True)
# visii.initialize_headless()

camera_entity = visii.entity.create(
    name="my_camera_entity",
    transform=visii.transform.create("my_camera_transform"),
    camera=visii.camera.create_perspective_from_fov(name = "my_camera", field_of_view = 0.785398, aspect = 1., near = .1))
visii.set_camera_entity(camera_entity)

#%%
camera_entity.get_transform().set_position(0, 0.0, -5.)
camera_entity.get_camera().use_perspective_from_fov(0.785398, 1.0, .01)
camera_entity.get_camera().set_view(
    visii.lookAt(
        visii.vec3(2,2,4),
        visii.vec3(0,0,.5),
        visii.vec3(0,0,1),
    )
)
camera_entity.get_camera().set_focal_distance(8.5)
camera_entity.get_camera().set_aperture_diameter(300)

#%%
floor = visii.entity.create(
    name="floor",
    mesh = visii.mesh.create_plane("floor"),
    transform = visii.transform.create("floor"),
    material = visii.material.create("floor")
)

#%%
sphere_mesh = visii.mesh.create_sphere("sphere1", 1, 128, 128)
mesh1 = visii.entity.create(
    name="mesh1",
    mesh = sphere_mesh,
    transform = visii.transform.create("mesh1"),
    material = visii.material.create("mesh1")
)

mesh2 = visii.entity.create(
    name="mesh2",
    mesh = sphere_mesh,
    transform = visii.transform.create("mesh2"),
    material = visii.material.create("mesh2")
)

mesh3 = visii.entity.create(
    name="mesh3",
    mesh = sphere_mesh,
    transform = visii.transform.create("mesh3"),
    material = visii.material.create("mesh3")
)
#%%

floor.get_material().set_roughness(1.0)
mesh2.get_material().set_roughness(0.0)
mesh3.get_material().set_roughness(0.0)
mesh1.get_material().set_roughness(0.0)
mesh3.get_material().set_metallic(1.0)
mesh2.get_material().set_transmission(1.0)

mesh1.get_material().set_base_color(1.0, 1.0, 1.0)
mesh2.get_material().set_base_color(1.0, 1.0, 1.0)
mesh3.get_material().set_base_color(1.0, 1.0, 1.0)


#%%
areaLight1 = visii.entity.create(
    name="areaLight1",
    light = visii.light.create("areaLight1"),
    transform = visii.transform.create("areaLight1"),
    mesh = sphere_mesh
)

#%%
areaLight1.get_transform().set_scale(.25)
floor.get_transform().set_scale(1000)
areaLight1.get_transform().set_position(0, 0, 4)
mesh1.get_transform().set_position(-1.0, 1.0, 1.0)
mesh2.get_transform().set_position(1.0, -1.0, 1.0)
mesh3.get_transform().set_position(1.0, 1.0, 1.0)
areaLight1.get_transform().set_scale(.3)


# %%
visii.set_dome_light_intensity(0)
areaLight1.get_light().set_intensity(1000000.)
areaLight1.get_light().set_temperature(5000)

# %%
visii.disable_denoiser()
visii.render_to_png(1080,1080,64,"test_denoiser_off.png")

# %%
visii.enable_denoiser()
visii.render_to_png(1080,1080,64,"test_denoiser_on.png")


# %%
visii.cleanup()