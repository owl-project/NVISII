import visii 
import numpy as np 
from PIL import Image 
import PIL
import time 


visii.initialize_headless()

# time to initialize this is a bug

# Create a camera
camera_entity = visii.entity.create(
    name="my_camera_entity",
    transform=visii.transform.create("my_camera_transform"),
    camera=visii.camera.create_perspective_from_fov(name = "my_camera",
    	field_of_view = 0.785398,
    	aspect = 1.,
    	near = .1
    	)
    )

# set the view camera transform
camera_entity.get_camera().set_view(
    visii.lookAt(
        visii.vec3(2,2,2), # camera_origin
        visii.vec3(0,0,0), # up vector (world coordinate)
        visii.vec3(0,0,1), # look at z is up
    )
)

# set the camera
visii.set_camera_entity(camera_entity)

# add floor
floor = visii.entity.create(
    name = "floor",
    mesh = visii.mesh.create_plane("floor"),
    transform = visii.transform.create("floor"),
    material = visii.material.create("floor")
)


sphere = visii.entity.create(
    name="sphere",
    mesh = visii.mesh.create_sphere("sphere"),
    transform = visii.transform.create("sphere"),
    material = visii.material.create("sphere")
)
sphere.get_transform().set_position(0.5,0,0.2)
sphere.get_transform().set_scale(0.4)
sphere.get_material().set_base_color(0,1,0) #normalize! 
sphere.get_material().set_roughness(0) # default is 1  
sphere.get_material().set_metallic(0.8)  # degault is 0 

knot = visii.entity.create(
    name="knot",
    mesh = visii.mesh.create_torus_knot("knot", 5, 8),
    transform = visii.transform.create("knot"),
    material = visii.material.create("knot")
)
knot.get_transform().set_position(-0.5,0,0.2)
knot.get_transform().set_scale(0.3)
knot.get_material().set_base_color(1,0,0)
# knot.get_material().diffuse(1,0,0)

teapotahedron = visii.entity.create(
    name="teapotahedron",
    mesh = visii.mesh.create_teapotahedron("teapotahedron"),
    transform = visii.transform.create("teapotahedron"),
    material = visii.material.create("teapotahedron")
)
teapotahedron.get_transform().set_position(0.1,1.2,0)
teapotahedron.get_transform().set_scale(0.1)
teapotahedron.get_material().set_base_color(0.9,0.9,1)
teapotahedron.get_material().set_transmission(0)
teapotahedron.get_material().set_roughness(0.2)



################################################################

time.sleep(3)

# Read and save the image 
x = np.array(visii.read_frame_buffer()).reshape(512,512,4)
img = Image.fromarray((x*255).astype(np.uint8)).transpose(PIL.Image.FLIP_TOP_BOTTOM)

# You should see a noise image, like gaussian noise. 
img.save("tmp.png")

visii.cleanup()



