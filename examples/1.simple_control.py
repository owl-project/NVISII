import visii
import argparse

parser = argparse.ArgumentParser()
   
parser.add_argument('--spp', 
                    default=50,
                    type=int,
                    help = "number of sample per pixel, higher the more costly")
parser.add_argument('--width', 
                    default=500,
                    type=int,
                    help = 'image output width')
parser.add_argument('--height', 
                    default=500,
                    type=int,
                    help = 'image output height')
parser.add_argument('--noise',
                    action='store_true',
                    default=False,
                    help = "if added the output of the ray tracing is not sent to optix's denoiser")
parser.add_argument('--out',
                    default='tmp.png',
                    help = 'the file output name, e.g., the image - it has to be a png')
opt = parser.parse_args()


# # # # # # # # # # # # # # # # # # # # # # # # #

visii.initialize_headless()

if not opt.noise is True: 
    visii.enable_denoiser()

# Lets create an entity that will serve as our camera. 
camera = visii.entity.create(
    name = "camera",
    transform=visii.transform.create("camera_transform"),
)

# Lets add a camera component to the entity
camera.set_camera(
    visii.camera.create_perspective_from_fov(
        name = "camera_camera", 
        field_of_view = 0.785398, 
        aspect = opt.width/float(opt.height)
    )
)

# visii can only use one camera at the time
visii.set_camera_entity(camera)

# Lets place our camera to look at the scene
# all the position are defined by visii.vector3  
camera.get_transform().set_position(
    visii.vec3(0, 5, 1)
)

# Lets set the view camera, we only offer look_at
camera.get_transform().look_at(
    visii.vec3(0,0,0), # at
    visii.vec3(0,0,1), # up vector
)

# Lets at a floor
visii.entity.create(
    name = "floor",
    mesh = visii.mesh.create_plane("mesh_floor"),
    transform = visii.transform.create("transform_floor"),
    material = visii.material.create("material_floor")
)

# Lets make our floor act as a purple mirror
# we first get the material associated with our floor entity
# if you do not have a direct handler to the entity, you
# search for a specific entity, material, transform, etc. 
# name. 
mat = visii.material.get("material_floor")

# Lets change the color
# the colors are RGB and the values are expected to be between 
# 0 and 1.  
mat.set_base_color(visii.vec3(0.9,0,0.7)) 

# Lets now change the metallic propreties for shinyness
mat.set_metallic(1) 
# to make sure we get a perfect mirror lets change the roughness
mat.set_roughness(0)

# we want to make sure our floor is large so let's update the 
# scale of the object.  
trans = visii.transform.get("transform_floor")

# the scale takes as input a vector 3
trans.set_scale(visii.vec3(5,5,1))

# Let's also add a sphere
sphere = visii.entity.create(
    name="sphere",
    mesh = visii.mesh.create_sphere("sphere"),
    transform = visii.transform.create("sphere"),
    material = visii.material.create("sphere")
)
# lets set the sphere up
sphere.get_transform().set_position(
    visii.vec3(0,0,0.2))
sphere.get_transform().set_scale(
    visii.vec3(0.4))
sphere.get_material().set_base_color(
    visii.vec3(0,1,0))  
sphere.get_material().set_roughness(1)   


# # # # # # # # # # # # # # # # # # # # # # # # #
# now that we have a simple scene set up let's render it 

visii.render_to_png(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel = int(opt.spp),   
    image_path=f"{opt.out}")

# let's clean up the GPU
visii.cleanup()