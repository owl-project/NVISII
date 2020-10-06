import visii
import random
import time

# this will create a window where you should 
# only see gaussian noise pattern
visii.initialize()

camera = visii.entity.create(name = "camera")
camera.set_transform(visii.transform.create(name = "camera_transform"))
camera.set_camera(
    visii.camera.create_perspective_from_fov(
        name = "camera_camera", 
        field_of_view = 0.785398, # note, this is in radians
        aspect = 1.0
    )
)
visii.set_camera_entity(camera)
camera.get_transform().look_at(
    at = (0, 0, 0.9), # at position
    up = (0, 0, 1),   # up vector
    eye = (0, 5, 1)   # eye position
)

sphere = visii.entity.create(
    name="sphere",
    mesh = visii.mesh.create_sphere("sphere"),
    transform = visii.transform.create("sphere"),
    material = visii.material.create("sphere")
)
sphere.get_transform().set_scale((0.4, 0.4, 0.4))
sphere.get_material().set_base_color((0.1,0.9,0.08))  
sphere.get_material().set_roughness(0.7)   
sphere.get_material().set_specular(1)
sphere.get_transform().set_position((0,0,0.41))

print("[Internal] transform limit is: " + str(visii.transform.get_count()))

for i in range(1000):
    print("Removing and recreating sphere " + str(i))
    visii.mesh.remove("sphere")
    sphere.set_mesh(visii.mesh.create_sphere("sphere", random.random() * 2.0))
    time.sleep(0.01)

print("Finished successfully")
visii.deinitialize()