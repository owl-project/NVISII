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
visii.set_max_bounce_depth(2)
visii.set_dome_light_intensity(0)
# # # # # # # # # # # # # # # # # # # # # # # # #


game_running = True
rotate_camera = False
speed_camera = 0.1 
camera_movement_pos_old = [200,200]
camera_movement_pos_now = [200,200]

x_rot = 0
y_rot = 0 

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
    at = visii.vec3(1,-1.5,1.8) + visii.vec3(0,1,0), # look at (world coordinate)
    up = visii.vec3(0,0,1), # up vector
    eye = visii.vec3(1,-1.5,1.8)
)

init_rot = camera.get_transform().get_rotation()

rot = visii.angleAxis( 
    x_rot, 
    visii.vec3(0,1,0)
)        
rot = rot * visii.angleAxis( 
    y_rot, 
    visii.vec3(1,0,0)
) 
camera.get_transform().add_rotation(rot)

visii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

# This function loads the 
sdb = visii.import_obj(
    "sdb", # prefix name
    'content/salle_de_bain_separated/salle_de_bain_separated.obj', #obj path
    'content/salle_de_bain_separated/', # mtl folder 
    visii.vec3(0,0,1), # translation 
    visii.vec3(0.1), # scale here
    visii.angleAxis(3.14 * .5, visii.vec3(1,0,0)) #rotation here
)

mirror = visii.material.get('sdbMirror')

mirror.set_roughness(0)
mirror.set_metallic(1)
mirror.set_base_color(visii.vec3(1))

for i_s, s in enumerate(sdb):
    if "light" in s.get_name().lower():
        s.set_light(visii.light.create('light'))
        s.get_light().set_intensity(50)
        s.get_light().set_temperature(5000)
        s.clear_material()

# # # # # # # # # # # # # # # # # # # # # # # # #
import pygame 
pygame.init()
screen = pygame.display.set_mode((400, 400))

while game_running:

    # visii camera matrix 
    cam_matrix = camera.get_transform().get_local_to_world_matrix()
    to_add = visii.vec4(0,0,0,0)
    
    for event in pygame.event.get():
        print(event)

        # Game is running check for quit
        if event.type == pygame.QUIT:
            game_running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                game_running = False
            # camera control
            # Forward and backward
            if event.key == pygame.K_w:
                print('hello')
                to_add[2] = 1 * speed_camera * -1
            if event.key == pygame.K_s:
                to_add[2] = 1 * speed_camera

            # left and right 
            if event.key == pygame.K_a:
                to_add[0] = 1 * speed_camera * -1
            if event.key == pygame.K_d:
                to_add[0] = 1 * speed_camera

            # up and down
            if event.key == pygame.K_q:
                to_add[1] = 1 * speed_camera * -1
            if event.key == pygame.K_e:
                to_add[1] = 1 * speed_camera 

            # change speed movement
            if event.key == pygame.K_UP:
                speed_camera *= 0.5 
                print('decrease speed camera')
            if event.key == pygame.K_DOWN:
                speed_camera /= 0.5
                print('increase speed camera')

        # camera rotation
        if event.type == pygame.MOUSEMOTION and rotate_camera:
            print('cam is moving')
            camera_movement_pos_old = camera_movement_pos_now
            camera_movement_pos_now = event.pos

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # left click grows radius 
                rotate_camera = True
                print('rotation')

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1: # left click grows radius 
                rotate_camera = False
                print('no rotation')
                camera_movement_pos_old = [200,200]
                camera_movement_pos_now = [200,200]

    # set mouse in the middle when moving

    if rotate_camera:
        x_rot -= (camera_movement_pos_old[0] - camera_movement_pos_now[0]) * 0.0001
        y_rot -= (camera_movement_pos_old[1] - camera_movement_pos_now[1]) * 0.0001

        # camera.get_transform().set_rotation(init_rot)

        rot_x_to_apply = visii.angleAxis( 
            x_rot, 
            # camera.get_transform().get_up()
            visii.vec3(0,1,0)
        )        
        # camera.get_transform().add_rotation(rot)


        rot_y_to_apply = visii.angleAxis( 
            y_rot, 
            # camera.get_transform().get_right()
            visii.vec3(1,0,0)
        ) 
        # camera.get_transform().look_at(
        #     at = camera.get_transform().get_position() + visii.vec3(0,1,0), # look at (world coordinate)
        #     up = visii.vec3(0,0,1), # up vector
        # )        

        camera.get_transform().set_rotation(init_rot* rot_x_to_apply * rot_y_to_apply)

        # camera.get_transform().set_rotation(rot * init_rot)
        
        pygame.mouse.set_pos([200,200])

    # control the camera
    if to_add[0]**2 > 0.001 or \
       to_add[1]**2 > 0.001 or \
       to_add[2]**2 > 0.001:

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