#%%
import sys, os
os.add_dll_directory(os.path.join(os.getcwd(), '..', 'install'))
sys.path.append(os.path.join(os.getcwd(), "..", "install"))

import visii

SAMPLES_PER_PIXEL = 1024
WIDTH = 1024 
HEIGHT = 1024

# input()

#%%
visii.initialize_headless()
# visii.initialize_interactive()
camera_entity = visii.entity.create(
    name="my_camera_entity",
    transform=visii.transform.create("my_camera_transform"),
    camera=visii.camera.create_perspective_from_fov(name = "my_camera", field_of_view = 0.785398, aspect = 1., near = .1))


visii.set_camera_entity(camera_entity)
camera_entity.get_transform().set_position(0, 0.0, -5.)
camera_entity.get_camera().use_perspective_from_fov(0.785398, 1.0, .01)
camera_entity.get_camera().set_view(
    visii.lookAt(
        visii.vec3(0,0,500),
        visii.vec3(0,0,0),
        visii.vec3(1,0,0),
    )
)

visii.set_dome_light_intensity(0)

# Create floor
floor = visii.entity.create(
    name="floor",
    mesh = visii.mesh.create_plane("floor"),
    transform = visii.transform.create("floor"),
    material = visii.material.create("floor")
)
floor.get_transform().set_scale(1000)
floor.get_material().set_roughness(1.0)


def create_sphere_light(name, position, intensity):
    L = visii.entity.create(
        name=name,
        light = visii.light.create(name),
        transform = visii.transform.create(name),
        mesh = visii.mesh.create_sphere(name),
    )
    L.get_light().set_intensity(intensity)
    L.get_transform().set_position(position)
    L.get_light().set_temperature(4000)


for i in range(0, 10):
    create_sphere_light("l" + str(i), visii.vec3(-50, -50, 0) + visii.vec3(i * 10, i * 10, 20*20), 1000000)

# %%
# Read and save the image 
for i in range(0, 10):
    floor.get_transform().set_position(0, 0, i * 20)
    visii.render_to_png(width=WIDTH, height=HEIGHT, samples_per_pixel=SAMPLES_PER_PIXEL, image_path="test_area_light" + str(i) +".png")

# # make sure the image is clamped 
# x[x>1.0] = 1.0
# x[x<0] = 0

# img = Image.fromarray((x*255).astype(np.uint8)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
# img.save("test_area_light.png")
