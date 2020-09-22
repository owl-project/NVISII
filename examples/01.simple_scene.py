import visii


SAMPLES_PER_PIXEL = 50
WIDTH = 500 
HEIGHT = 500 
USE_DENOISER = True
FILE_NAME = "tmp.png"

# # # # # # # # # # # # # # # # # # # # # # # # #

visii.initialize_headless()

if USE_DENOISER is True: 
    visii.enable_denoiser()

# Lets create an entity that will serve as our camera.

# All visii objects have a name that can be used for easy lookup later.
camera = visii.entity.create(name = "camera")

# To place the camera into our scene, we'll add a "transform" component 
camera.set_transform(visii.transform.create(name = "camera_transform"))

# To make our camera entity act like a "camera", we'll add a camera component
camera.set_camera(
    visii.camera.create_perspective_from_fov(
        name = "camera_camera", 
        field_of_view = 0.785398, # note, this is in radians
        aspect = float(WIDTH)/float(HEIGHT)
    )
)

# Finally, we'll select this entity to be the current camera entity.
# (visii can only use one camera at the time)
visii.set_camera_entity(camera)

# Lets place our camera in the scene to look at an object

# All positions and vectors are defined through a sequence of three numbers.
# That sequence can be specified using lists, tuples, through numpy, 
# or using the built in visii.vec3 type:
# transform.set_position([x, y, z]) <- lists
# transform.set_position((x, y, z)) <- tuples
# transform.set_position(np.array([x, y, z])) <- numpy arrays
# transform.set_position(visii.vec3(x, y, z)) <- visii vec3 object

# Lets set the camera to look at an object. 
# We'll do this by editing the transform component, 
# which functionally acts like the camera's "view" transform.
# (Note that any of the below three vectors can match any of 
# the above mentioned patterns)
camera.get_transform().look_at(
    at = (0, 0, 0.9), # at position
    up = (0, 0, 1),   # up vector
    eye = (0, 5, 1)   # eye position
)

# Next, lets at an object (a floor).
# For an entity to be visible to a camera, that entity
# must have a mesh component, a transform component, and a 
# material component.
visii.entity.create(
    name = "floor",
    mesh = visii.mesh.create_plane("mesh_floor"),
    transform = visii.transform.create("transform_floor"),
    material = visii.material.create("material_floor")
)

# Lets make our floor act as a mirror
# we first get the material associated with our floor entity
# if you do not have a direct handler to the entity, you
# search for a specific entity, material, transform, etc. 
# name. 
mat = visii.material.get("material_floor")

# Lets change the color
# the colors are RGB and the values are expected to be between 
# 0 and 1.  
mat.set_base_color((0.19,0.16,0.19)) 
# Lets now change the metallic propreties for shinyness
mat.set_metallic(1) 
# to make sure we get a perfect mirror lets change the roughness
mat.set_roughness(0)

# we want to make sure our floor is large so let's update the 
# scale of the object.  
trans = visii.transform.get("transform_floor")

# the scale takes as input a vector of 3 numbers
trans.set_scale((5,5,1))

# Let's also add a sphere
sphere = visii.entity.create(
    name="sphere",
    mesh = visii.mesh.create_sphere("sphere"),
    transform = visii.transform.create("sphere"),
    material = visii.material.create("sphere")
)
# lets set the sphere up
sphere.get_transform().set_position((0,0,0.41))
sphere.get_transform().set_scale((0.4, 0.4, 0.4))
sphere.get_material().set_base_color((0.1,0.9,0.08))  
sphere.get_material().set_roughness(0.7)   
sphere.get_material().set_specular(1)   


# # # # # # # # # # # # # # # # # # # # # # # # #
# now that we have a simple scene set up let's render it 
print("rendering to", FILE_NAME)
visii.render_to_png(
    width = WIDTH, 
    height = HEIGHT, 
    samples_per_pixel = SAMPLES_PER_PIXEL,   
    image_path = FILE_NAME
)

# let's clean up the GPU
visii.deinitialize()
print("done!")