import visii
import numpy as np 
from math import acos
from math import sqrt
from math import pi  
import colorsys
import cv2 

opt = lambda: None
opt.spp = 100 
opt.width = 1280
opt.height = 720 
opt.noise = False
opt.path_obj = 'content/dragon/dragon.obj'

# # # # # # # # # # # # # # # # # # # # # # # # #
visii.initialize(headless=True, verbose=True)

if not opt.noise is True: 
    visii.enable_denoiser()

camera = visii.entity.create(
    name = "camera",
    transform = visii.transform.create("camera"),
    camera = visii.camera.create(
        name = "camera",  
        aspect = float(opt.width)/float(opt.height)
    )
)

camera.get_transform().look_at(
    at = (0,0.1,0.1),
    up = (0,0,1),
    eye = (0,3.0,0.2),
)
visii.set_camera_entity(camera)

visii.set_dome_light_sky(sun_position = (10, 10, 1), saturation = 2)
visii.set_dome_light_intensity(1.5)

# # # # # # # # # # # # # # # # # # # # # # # # #

floor = visii.entity.create(
    name = "floor",
    mesh = visii.mesh.create_plane("floor", size = (10,10)),
    material = visii.material.create("floor", base_color = (.5, .5, .5), roughness = 0.0, metallic = 1.0),
    transform = visii.transform.create("floor", position = (0,0,-.3))
)

# Next, let's load an obj
mesh = visii.mesh.create_from_file("obj", opt.path_obj)

# Now, lets make three instances of that mesh
obj1 = visii.entity.create(
    name="obj1",
    mesh = mesh,
    transform = visii.transform.create("obj1"),
    material = visii.material.create("obj1")
)

obj2 = visii.entity.create(
    name="obj2",
    mesh = mesh,
    transform = visii.transform.create("obj2"),
    material = visii.material.create("obj2")
)

obj3 = visii.entity.create(
    name="obj3",
    mesh = mesh,
    transform = visii.transform.create("obj3"),
    material = visii.material.create("obj3")
)

obj4 = visii.entity.create(
    name="obj4",
    mesh = mesh,
    transform = visii.transform.create("obj4"),
    material = visii.material.create("obj4")
)

# place those objects into the scene

# lets set the obj_entity up
obj1.get_transform().set_position((-1.5, 0, 0))
obj1.get_transform().set_rotation((0.7071, 0, 0, 0.7071))
obj1.get_material().set_base_color((1,0,0))  
obj1.get_material().set_roughness(0.7)   
obj1.get_material().set_specular(1)   
obj1.get_material().set_sheen(1)

obj2.get_transform().set_position((-.5, 0, 0))
obj2.get_transform().set_rotation((0.7071, 0, 0, 0.7071))
obj2.get_material().set_base_color((0,1,0))  
obj2.get_material().set_roughness(0.7)   
obj2.get_material().set_specular(1)   
obj2.get_material().set_sheen(1)

obj3.get_transform().set_position((.5, 0, 0))
obj3.get_transform().set_rotation((0.7071, 0, 0, 0.7071))
obj3.get_material().set_base_color((0,0,1))  
obj3.get_material().set_roughness(0.7)   
obj3.get_material().set_specular(1)   
obj3.get_material().set_sheen(1)

obj4.get_transform().set_position((1.5, 0, 0))
obj4.get_transform().set_rotation((0.7071, 0, 0, 0.7071))
obj4.get_material().set_base_color((.5,.5,.5))  
obj4.get_material().set_roughness(0.7)   
obj4.get_material().set_specular(1)   
obj4.get_material().set_sheen(1)

# # # # # # # # # # # # # # # # # # # # # # # # #

# MOTION VECTORS section

# need to remove the motion blur that adding previous transform will cause. 
# We also want the motion from frame 0 to 1
visii.sample_time_interval((0,0))

# make sure that raw sample the middle the of the pixel 
# without this there will be noise on the motion segmentation

visii.sample_pixel_area(
    x_sample_interval = (.5,.5), 
    y_sample_interval = (.5, .5)
)

def length(v):
    return np.sqrt(v[0]**2+v[1]**2)

def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]

def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees

def py_ang(A, B=(1,0)):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner

def generate_image_from_motion_vector(motion_vectors_array, use_magnitude=False):
    image = np.zeros(motion_vectors_array.shape)
    image[:,:,3] =1 
    indices = np.abs(motion_vectors_array[:,:,0]) + np.abs(motion_vectors_array[:,:,1])
    indices = np.nonzero(indices > 0)

    for i_indice in range(len(indices[0])):
        i,j = indices[0][i_indice], indices[1][i_indice]
        angle_vector = np.array([
                        motion_vectors_array[i,j,0],
                        motion_vectors_array[i,j,1]]
                    )
        magnitude = length(angle_vector)
        
        # Use the hsv to apply color as a function of the angle
        c = [0,0,0]
        if magnitude > 0.000001:
            angle=py_ang(angle_vector)
            if use_magnitude:
                c = colorsys.hsv_to_rgb(angle/360,1,magnitude)
            else:
                c = colorsys.hsv_to_rgb(angle/360,1,1)
        # for i_c in range(3):
        image[i,j,0:3] = c
    return image



visii.render_to_file(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    file_path=f"20_frame1.png"
)

obj1.get_transform().set_position(obj1.get_transform().get_position(),previous=True)
obj1.get_transform().add_position(visii.vec3(0,0.5,0))

obj2.get_transform().set_position(obj2.get_transform().get_position(),previous=True)
obj2.get_transform().add_position(visii.vec3(0,0,0.5))

obj3.get_transform().set_rotation(obj3.get_transform().get_rotation(),previous=True)
obj3.get_transform().add_rotation(visii.quat(0,-1,0,0))

motion_vectors_array = visii.render_data(
    width=int(opt.width), 
    height=int(opt.height), 
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="diffuse_motion_vectors"
)

motion_vectors_array = np.array(motion_vectors_array).reshape(opt.height,opt.width,4) * -1
motion_vectors_array = np.flipud(motion_vectors_array)
image = generate_image_from_motion_vector(motion_vectors_array)
cv2.imwrite("20_motion_from_1_to_2.png",image*255)


# frame now has to be set at 1 to have the current image, e.g., the transformed one
visii.sample_time_interval((1,1))
visii.render_to_file(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    file_path=f"20_frame2.png"
)



# let's clean up the GPU
visii.deinitialize()