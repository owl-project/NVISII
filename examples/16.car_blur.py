import visii
import noise
import random
import argparse
import numpy as np 

parser = argparse.ArgumentParser()

parser.add_argument('--spp', 
                    default=400,
                    type=int,
                    help = "number of sample per pixel, higher the more costly")
parser.add_argument('--width', 
                    default=1000,
                    type=int,
                    help = 'image output width')
parser.add_argument('--height', 
                    default=1000,
                    type=int,
                    help = 'image output height')
parser.add_argument('--noise',
                    action='store_true',
                    default=False,
                    help = "if added the output of the ray tracing is not sent to optix's denoiser")
parser.add_argument('--out',
                    default='tmp.png',
                    help = "output filename")

opt = parser.parse_args()

# # # # # # # # # # # # # # # # # # # # # # # # #
visii.initialize_interactive()
visii.resize_window(1000,1000)
visii.set_max_bounce_depth(50)
visii.set_dome_light_intensity(.5)
# # # # # # # # # # # # # # # # # # # # # # # # #

# load the textures
dome = visii.texture.create_from_image("dome", "content/teatro_massimo_2k.hdr")

# we can add HDR images to act as dome
visii.set_dome_light_texture(dome)
visii.set_dome_light_rotation(visii.angleAxis(visii.pi() * .5, visii.vec3(0, 0, 1)))

game_running = True
rotate_camera = False
speed_camera = .01
camera_movement = [0,0]

car_speed = 0
car_speed_x = car_speed
car_speed_y = -2 * car_speed

x_rot = 0
y_rot = 0 
camera_height = 80
# # # # # # # # # # # # # # # # # # # # # # # # #

if not opt.noise is True: 
    visii.enable_denoiser()

camera = visii.entity.create(
    name = "camera",
    transform = visii.transform.create("camera"),
    camera = visii.camera.create_perspective_from_fov(
        name = "camera", 
        field_of_view = 0.785398, 
        aspect = float(opt.width)/float(opt.height)
    )
)

camera.get_transform().look_at(
    at = visii.vec3(-50,0,camera_height) , # look at (world coordinate)
    up = visii.vec3(0,0,1), # up vector
    eye = visii.vec3(-500,500,100 + camera_height),
    previous = False
)

camera.get_transform().look_at(
    at = visii.vec3(-50,0,camera_height) + visii.vec3(car_speed_x, car_speed_y, .0) , # look at (world coordinate)
    up = visii.vec3(0,0,1), # up vector
    eye = visii.vec3(-500,500,100 + camera_height),
    previous = True
)

camera.get_camera().set_aperture_diameter(5000)
camera.get_camera().set_focal_distance(500)

# init_rot = camera.get_transform().get_rotation()

# rot = visii.angleAxis( 
#     x_rot, 
#     visii.vec3(0,1,0)
# )        
# rot = rot * visii.angleAxis( 
#     y_rot, 
#     visii.vec3(1,0,0)
# ) 
# camera.get_transform().add_rotation(rot)

visii.set_camera_entity(camera)

floor = visii.entity.create(
    name = "floor",
    mesh = visii.mesh.create_plane("plane"),
    material = visii.material.create("plane"),
    transform = visii.transform.create("plane")
)
floor.get_transform().set_scale(visii.vec3(10000))
floor.get_transform().set_position(visii.vec3(0, 0, -5))
floor.get_material().set_base_color(visii.vec3(.0))
floor.get_material().set_roughness(1)
floor.get_material().set_specular(0)

# # # # # # # # # # # # # # # # # # # # # # # # #

# This function loads the 
sdb = visii.import_obj(
    "sdb", # prefix name
    'content/bmw_alt.obj', #obj path
    'content/', # mtl folder 
    visii.vec3(0,0,0), # translation 
    visii.vec3(1), # scale here
    visii.angleAxis(3.14 * .5, visii.vec3(1,0,0)) #rotation here
)

# mirror = visii.material.get('sdbMirror')

# mirror.set_roughness(0)
# mirror.set_metallic(1)
# mirror.set_base_color(visii.vec3(1))

hl = visii.light.create("headlights")
tl = visii.light.create("taillights")
tl.set_color(visii.vec3(1,0,0))
hl.set_intensity(10000)
tl.set_intensity(10000)

for i_s, s in enumerate(sdb):
    # print(s.get_name())
    # if 'car' in s.get_name():
    #     print(s.get_name())
    s.get_transform().set_linear_velocity(visii.vec3(car_speed_x, car_speed_y, .0))

    print(s.get_name())
    if "carshell" in s.get_name().lower():
        s.get_material().set_clearcoat(1)
        s.get_material().set_clearcoat_roughness(0)
        s.get_material().set_roughness(.1)

    if "angeleye" in s.get_name().lower():
        s.set_light(hl) 

    if "lightsbulb" in s.get_name().lower():
        s.set_light(hl)  

    # if "taillight" in s.get_name().lower():
        # s.set_light(tl)  

    if "lightsglass" in s.get_name().lower() or "window" in s.get_name().lower():
        print(s.get_name())
        s.clear_material()
        s.set_material(visii.material.create(s.get_name().lower()))
        s.get_material().set_ior(1.0)
        # s.get_material().set_base_color(visii.vec3(1,1,1))
        s.get_material().set_transmission(1)
        s.get_material().set_roughness(0)
        s.get_material().set_metallic(0)
    # elif 'light' in s.get_name().lower():
    #     print(s.get_name())
    #     s.set_light(visii.light.create('light' + str(i_s)))
    #     s.get_light().set_intensity(20)
    #     s.get_light().set_temperature(5000)
    # if 'tire' in s.get_name().lower():
        # s.get_transform().set_angular_velocity(visii.angleAxis(3.14 * .05, visii.vec3(1,0,0)))

# visii.entity.get("sdbcarShell_1").get_material().set_base_color(visii.vec3(1,0,0))

visii.render_to_png(1024, 1024, 1000, "motion_blur_3")


# # # # # # # # # # # # # # # # # # # # # # # # #
import pygame 
pygame.init()
screen = pygame.display.set_mode((400, 400))

while game_running:

    # visii camera matrix 
    cam_matrix = camera.get_transform().get_local_to_world_matrix()
    to_add = visii.vec4(0,0,0,0)

    keys = pygame.key.get_pressed()
    mouse = pygame.mouse.get_pressed()
    mouseRel = pygame.mouse.get_rel()
    
    # print(mouseRel)

    # camera control
    # Forward and backward
    if keys[pygame.K_w]:
        to_add[2] = 1 * speed_camera * -1
    if keys[pygame.K_s]:
        to_add[2] = 1 * speed_camera

    # left and right 
    if keys[pygame.K_a]:
        to_add[0] = 1 * speed_camera * -1
    if keys[pygame.K_d]:
        to_add[0] = 1 * speed_camera

    # up and down
    if keys[pygame.K_q]:
        to_add[1] = 1 * speed_camera * -1
    if keys[pygame.K_e]:
        to_add[1] = 1 * speed_camera 

    # camera rotation
    if mouse[0]:
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        camera_movement = mouseRel
        rotate_camera = True
    else:
        if rotate_camera:
            pygame.mouse.set_pos([200,200])
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        rotate_camera = False
            
    for event in pygame.event.get():
        # print(event)

        # Game is running check for quit
        if event.type == pygame.QUIT:
            game_running = False
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                game_running = False

            # change speed movement
            if event.key == pygame.K_UP:
                speed_camera *= 0.5 
                print('decrease speed camera')
            if event.key == pygame.K_DOWN:
                speed_camera /= 0.5
                print('increase speed camera')

    # set mouse in the middle when moving

    if rotate_camera:
        x_rot -= (camera_movement[0]) * 0.001
        y_rot -= (camera_movement[1]) * 0.001

        init_rot = visii.angleAxis(
            visii.pi() * .5,
            visii.vec3(1,0,0)
        )

        rot_x_to_apply = visii.angleAxis( 
            x_rot + visii.pi() * 1.25, 
            # camera.get_transform().get_up()
            visii.vec3(0,1,0)
        )        

        rot_y_to_apply = visii.angleAxis( 
            y_rot, 
            visii.vec3(1,0,0)
        ) 
       
        camera.get_transform().set_rotation(init_rot * rot_x_to_apply * rot_y_to_apply)
        camera.get_transform().clear_motion()
 

    # control the camera
    if abs(to_add[0]) > 0.0 or \
       abs(to_add[1]) > 0.0 or \
       abs(to_add[2]) > 0.0:

        to_add_world = cam_matrix * to_add

        camera.get_transform().add_position(
            visii.vec3(
                to_add_world[0],
                to_add_world[1],
                to_add_world[2]
            )
        )

# let's clean up the GPU
visii.deinitialize()