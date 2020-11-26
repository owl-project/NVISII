import visii
import noise
import random
import numpy as np 

opt = lambda: None
opt.spp = 400 
opt.width = 1920
opt.height = 1080 
opt.noise = False
opt.out = '15_camera_motion_car_blur.png'
opt.control = True

# # # # # # # # # # # # # # # # # # # # # # # # #
visii.initialize()
visii.set_dome_light_intensity(.8)
visii.resize_window(int(opt.width), int(opt.height))
# # # # # # # # # # # # # # # # # # # # # # # # #

# load the textures
dome = visii.texture.create_from_file("dome", "content/teatro_massimo_2k.hdr")

# we can add HDR images to act as dome
visii.set_dome_light_texture(dome)
visii.set_dome_light_rotation(visii.angleAxis(visii.pi() * .5, visii.vec3(0, 0, 1)))

car_speed = 0
car_speed_x = car_speed
car_speed_y = -2 * car_speed

camera_height = 80
# # # # # # # # # # # # # # # # # # # # # # # # #

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
    'content/bmw/bmw.obj', #obj path
    'content/bmw/', # mtl folder 
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
hl.set_intensity(1000)
tl.set_intensity(1000)

#%%
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
    #     s.set_light(tl)  

    if "tirerim" in s.get_name().lower():
        s.clear_material()
        s.set_material(visii.material.create(s.get_name().lower()))
        # s.get_material().set_ior(1.0)
        s.get_material().set_base_color(visii.vec3(0.6,0.6,0.6))
        s.get_material().set_transmission(0)
        s.get_material().set_roughness(0.1)
        s.get_material().set_metallic(1)

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

# visii.render_to_png(1024, 1024, 1000, "motion_blur_3")

if opt.control:
    camera.get_transform().clear_motion()

    cursor = visii.vec4()
    speed_camera = 4.0
    rot = visii.vec2(visii.pi() * 1.25, 0)
    def interact():
        global speed_camera
        global cursor
        global rot

        # visii camera matrix 
        cam_matrix = camera.get_transform().get_local_to_world_matrix()
        dt = visii.vec4(0,0,0,0)

        # translation
        if visii.is_button_held("W"): dt[2] = -speed_camera
        if visii.is_button_held("S"): dt[2] =  speed_camera
        if visii.is_button_held("A"): dt[0] = -speed_camera
        if visii.is_button_held("D"): dt[0] =  speed_camera
        if visii.is_button_held("Q"): dt[1] = -speed_camera
        if visii.is_button_held("E"): dt[1] =  speed_camera 

        # control the camera
        if visii.length(dt) > 0.0:
            w_dt = cam_matrix * dt
            camera.get_transform().add_position(visii.vec3(w_dt))

        # camera rotation
        cursor[2] = cursor[0]
        cursor[3] = cursor[1]
        cursor[0] = visii.get_cursor_pos().x
        cursor[1] = visii.get_cursor_pos().y
        if visii.is_button_held("MOUSE_LEFT"):
            visii.set_cursor_mode("DISABLED")
            rotate_camera = True
        else:
            visii.set_cursor_mode("NORMAL")
            rotate_camera = False

        if rotate_camera:
            rot.x -= (cursor[0] - cursor[2]) * 0.001
            rot.y -= (cursor[1] - cursor[3]) * 0.001
            init_rot = visii.angleAxis(visii.pi() * .5, (1,0,0))
            yaw = visii.angleAxis(rot.x, (0,1,0))
            pitch = visii.angleAxis(rot.y, (1,0,0)) 
            camera.get_transform().set_rotation(init_rot * yaw * pitch)

        # change speed movement
        if visii.is_button_pressed("UP"):
            speed_camera *= 0.5 
            print('decrease speed camera', speed_camera)
        if visii.is_button_pressed("DOWN"):
            speed_camera /= 0.5
            print('increase speed camera', speed_camera)
            
    visii.register_pre_render_callback(interact)
    import time
    while not visii.should_window_close(): 
        time.sleep(.1)

visii.render_to_file(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    file_path=f"{opt.out}"
)

# let's clean up the GPU
visii.deinitialize()