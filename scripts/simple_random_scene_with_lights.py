# import sys, os
# os.add_dll_directory(os.path.join(os.getcwd(), '..', 'install'))
# sys.path.append(os.path.join(os.getcwd(), "..", "install"))

# input()

import random
import visii 
import time 
import randomcolor


# NB_OBJS = 16999
NB_OBJS = 10999
# NB_OBJS = 1000
NB_LIGHTS = 20

SAMPLES_PER_PIXEL = 200

# WIDTH = 1920 
# HEIGHT = 1080

WIDTH =  1000	
HEIGHT = 500


visii.initialize_headless()
# visii.initialize_interactive()

# time to initialize this is a bug

# Create a camera
camera_entity = visii.entity.create(
    name="my_camera",
    transform=visii.transform.create("my_camera"),
    camera=visii.camera.create_perspective_from_fov(name = "my_camera",
        field_of_view = 0.785398,
        # aspect = 1.,
        aspect = WIDTH/HEIGHT,
        near = .1
        )
    )

# This is to set the camera internal parameters
# camera_entity.get_camera().set_aperture_diameter(20)
# camera_entity.get_camera().set_focal_distance(3.5)

# Change the dome light intensity
visii.set_dome_light_intensity(0.5)

# set the view camera transform
camera_entity.get_camera().set_view(
    visii.lookAt(
        visii.vec3(0,0,5), # camera_origin
        visii.vec3(0,0,0), # look at (world coordinate)
        visii.vec3(1,0,0), # up vector
    )
)

# set the camera
visii.set_camera_entity(camera_entity)
meshes = [] 

rcolor = randomcolor.RandomColor()

def add_random_light(name = 'name'):
    global rcolor
    obj= visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        mesh = visii.mesh.create_sphere(name),
        light = visii.light.create(name)
    )
    obj.get_transform().set_scale(2)


    obj.get_light().set_intensity(random.uniform(50000,100000))
    # obj.get_light().set_temperature(np.random.randint(100,9000))

    c = eval(str(rcolor.generate(luminosity='bright',format_='rgb')[0])[3:])
    obj.get_light().set_color(
        c[0]/255.0,
        c[1]/255.0,
        c[2]/255.0)  
 
    obj.get_light().set_temperature(4000)
    # obj.get_light().set_intensity(10000.)

    obj.get_transform().set_position(
            random.uniform(-10,10),
            random.uniform(-10,10),
            random.uniform(5,30)
        )
    
def add_random_obj(name = "name"):
    global rcolor
    obj= visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        material = visii.material.create(name),
    )

    obj_id = random.randint(0,15)

    mesh = None
    if obj_id == 0:
        if add_random_obj.create_sphere is None:
            add_random_obj.create_sphere = visii.mesh.create_sphere(name) 
        mesh = add_random_obj.create_sphere
    if obj_id == 1:
        if add_random_obj.create_torus_knot is None:
            add_random_obj.create_torus_knot = visii.mesh.create_torus_knot(name, 
                random.randint(2,6),
                random.randint(4,10))
        mesh = add_random_obj.create_torus_knot
    if obj_id == 2:
        if add_random_obj.create_teapotahedron is None:
            add_random_obj.create_teapotahedron = visii.mesh.create_teapotahedron(name) 
        mesh = add_random_obj.create_teapotahedron
    if obj_id == 3:
        if add_random_obj.create_box is None:
            add_random_obj.create_box = visii.mesh.create_box(name) 
        mesh = add_random_obj.create_box
    if obj_id == 4:
        if add_random_obj.create_capped_cone is None:
            add_random_obj.create_capped_cone = visii.mesh.create_capped_cone(name) 
        mesh = add_random_obj.create_capped_cone
    if obj_id == 5:
        if add_random_obj.create_capped_cylinder is None:
            add_random_obj.create_capped_cylinder = visii.mesh.create_capped_cylinder(name) 
        mesh = add_random_obj.create_capped_cylinder
    if obj_id == 6:
        if add_random_obj.create_capsule is None:
            add_random_obj.create_capsule = visii.mesh.create_capsule(name) 
        mesh = add_random_obj.create_capsule
    if obj_id == 7:
        if add_random_obj.create_cylinder is None:
            add_random_obj.create_cylinder = visii.mesh.create_cylinder(name) 
        mesh = add_random_obj.create_cylinder
    if obj_id == 8:
        if add_random_obj.create_disk is None:
            add_random_obj.create_disk = visii.mesh.create_disk(name) 
        mesh = add_random_obj.create_disk
    if obj_id == 9:
        if add_random_obj.create_dodecahedron is None:
            add_random_obj.create_dodecahedron = visii.mesh.create_dodecahedron(name) 
        mesh = add_random_obj.create_dodecahedron
    if obj_id == 10:
        if add_random_obj.create_icosahedron is None:
            add_random_obj.create_icosahedron = visii.mesh.create_icosahedron(name) 
        mesh = add_random_obj.create_icosahedron
    if obj_id == 11:
        if add_random_obj.create_icosphere is None:
            add_random_obj.create_icosphere = visii.mesh.create_icosphere(name) 
        mesh = add_random_obj.create_icosphere
    if obj_id == 12:
        if add_random_obj.create_rounded_box is None:
            add_random_obj.create_rounded_box = visii.mesh.create_rounded_box(name) 
        mesh = add_random_obj.create_rounded_box
    if obj_id == 13:
        if add_random_obj.create_spring is None:
            add_random_obj.create_spring = visii.mesh.create_spring(name) 
        mesh = add_random_obj.create_spring
    if obj_id == 14:
        if add_random_obj.create_torus is None:
            add_random_obj.create_torus = visii.mesh.create_torus(name) 
        mesh = add_random_obj.create_torus
    if obj_id == 15:
        if add_random_obj.create_tube is None:
            add_random_obj.create_tube = visii.mesh.create_tube(name) 
        mesh = add_random_obj.create_tube

    obj.set_mesh(mesh)
    obj.get_transform().set_position(
        random.uniform(-5,5),
        random.uniform(-5,5),
        random.uniform(-10,3)
        )
    obj.get_transform().set_rotation(
        visii.quat(1.0 ,random.random(), random.random(), random.random()) 
        )
    obj.get_transform().set_scale(random.uniform(0.01,0.2))
    
    c = eval(str(rcolor.generate(luminosity='bright',format_='rgb')[0])[3:])
    obj.get_material().set_base_color(
        c[0]/255.0,
        c[1]/255.0,
        c[2]/255.0)  

    obj.get_material().set_roughness(random.uniform(0,1)) # default is 1  
    obj.get_material().set_metallic(random.uniform(0,1))  # degault is 0     
    obj.get_material().set_transmission(random.uniform(0,1))  # degault is 0     
    obj.get_material().set_sheen(random.uniform(0,1))  # degault is 0     
    obj.get_material().set_clearcoat(random.uniform(0,1))  # degault is 0     
    obj.get_material().set_specular(random.uniform(0,1))  # degault is 0     
    obj.get_material().set_anisotropic(random.uniform(0,1))  # degault is 0     

#create the meshes place holders 
add_random_obj.create_sphere = None
add_random_obj.create_torus_knot = None
add_random_obj.create_teapotahedron = None
add_random_obj.create_box = None
add_random_obj.create_capped_cone = None
add_random_obj.create_capped_cylinder = None
add_random_obj.create_capsule = None
add_random_obj.create_cylinder = None
add_random_obj.create_disk = None
add_random_obj.create_dodecahedron = None
add_random_obj.create_icosahedron = None
add_random_obj.create_icosphere = None
add_random_obj.create_rounded_box = None
add_random_obj.create_spring = None
add_random_obj.create_torus = None
add_random_obj.create_tube = None

# create a random scene, the function defines the values

print('creating objects')
for i in range(NB_OBJS):
    add_random_obj(str(i))

print('creating lights')
for i in range(NB_LIGHTS):
    add_random_light("l"+str(i))


################################################################
print('rendering')



# for i in range(0,2000,50):
# 	if i is 0: 
# 		i = 1
# 	print(f' denoiser {i}')
# 	visii.enable_denoiser()
# 	visii.render_to_png(width=WIDTH, 
# 	                    height=HEIGHT, 
# 	                    samples_per_pixel=i,
# 	                    image_path=f"denoise/out_denoise_{str(i).zfill(3)}.png")
# 	visii.disable_denoiser()

print(' denoise')
visii.enable_denoiser()
visii.render_to_png(width=WIDTH, 
                    height=HEIGHT, 
                    samples_per_pixel=SAMPLES_PER_PIXEL,
                    image_path="out_denoise.png")


print(' noise')
visii.disable_denoiser()
visii.render_to_png(width=WIDTH, 
                    height=HEIGHT, 
                    samples_per_pixel=SAMPLES_PER_PIXEL,
                    image_path="out_noise.png")

visii.cleanup()

