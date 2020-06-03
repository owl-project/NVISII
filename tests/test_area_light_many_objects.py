#%%
import sys, os, math
os.add_dll_directory(os.path.join(os.getcwd(), '..', 'install'))
sys.path.append(os.path.join(os.getcwd(), "..", "install"))

import visii

SAMPLES_PER_PIXEL = 128
WIDTH = 1024 
HEIGHT = 1024

NUM_OBJECTS = 90
NUM_LIGHTS = 20
L_DIST = 10
O_DIST = 5

visii.initialize_interactive()
# visii.initialize_headless()
camera_entity = visii.entity.create(
    name="my_camera_entity",
    transform=visii.transform.create("my_camera_transform"),
    camera=visii.camera.create_perspective_from_fov(name = "my_camera", field_of_view = 0.785398, aspect = 1., near = .1))

visii.set_camera_entity(camera_entity)
camera_entity.get_transform().set_position(0, 0.0, -5.)
camera_entity.get_camera().use_perspective_from_fov(0.785398, 1.0, .01)
camera_entity.get_camera().set_view(
    visii.lookAt(
        visii.vec3(0,0,30),
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


def create_sphere_object(name, position):
    L = visii.entity.create(
        name=name,
        material = visii.material.create(name),
        transform = visii.transform.create(name),
        mesh = visii.mesh.create_sphere(name),
    )
    L.get_transform().set_position(position)


for i in range(0, NUM_OBJECTS):
    twopi = math.pi * 2
    create_sphere_object("o" + str(i), visii.vec3(O_DIST * math.sin((i / NUM_OBJECTS) * twopi), O_DIST * math.cos((i / NUM_OBJECTS) * twopi), 1))

def create_sphere_light(name, position, intensity):
    L = visii.entity.create(
        name=name,
        light = visii.light.create(name),
        transform = visii.transform.create(name),
        mesh = visii.mesh.create_sphere(name),
    )
    L.get_light().set_intensity(intensity)
    L.get_transform().set_position(position)
    L.get_transform().set_scale(.2)
    L.get_light().set_temperature(4000)


for i in range(0, NUM_LIGHTS):
    twopi = math.pi * 2
    create_sphere_light("l" + str(i), visii.vec3(L_DIST * math.sin((i / NUM_LIGHTS) * twopi), L_DIST * math.cos((i / NUM_LIGHTS) * twopi), 1), 100000)

visii.render_to_png(width=WIDTH, height=HEIGHT, samples_per_pixel=SAMPLES_PER_PIXEL, image_path="test_area_light_many_objects.png")

input()