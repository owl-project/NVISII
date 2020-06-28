#%%
import sys, os, math
os.add_dll_directory(os.path.join(os.getcwd(), '..', 'install'))
sys.path.append(os.path.join(os.getcwd(), "..", "install"))

import visii

# %%
visii.initialize_interactive()

# %%
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

# %%
tex = visii.texture.create("texture")

# %%
visii.set_dome_light_texture(tex)

# %%