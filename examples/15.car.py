import nvisii
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
nvisii.initialize()
nvisii.set_dome_light_intensity(.8)
nvisii.resize_window(int(opt.width), int(opt.height))
# # # # # # # # # # # # # # # # # # # # # # # # #

# load the textures
dome = nvisii.texture.create_from_file("dome", "content/teatro_massimo_2k.hdr")

# we can add HDR images to act as dome
nvisii.set_dome_light_texture(dome, enable_cdf=True)
nvisii.set_dome_light_rotation(nvisii.angleAxis(nvisii.pi() * .5, nvisii.vec3(0, 0, 1)))

camera_height = 80
# # # # # # # # # # # # # # # # # # # # # # # # #

if not opt.noise is True: 
    nvisii.enable_denoiser()

camera = nvisii.entity.create(
    name = "camera",
    transform = nvisii.transform.create("camera"),
    camera = nvisii.camera.create(
        name = "camera", 
        aspect = float(opt.width)/float(opt.height)
    )
)

camera.get_transform().look_at(
    at = nvisii.vec3(-50,0,camera_height) , # look at (world coordinate)
    up = nvisii.vec3(0,0,1), # up vector
    eye = nvisii.vec3(-500,500,100 + camera_height)
)

nvisii.set_camera_entity(camera)

floor = nvisii.entity.create(
    name = "floor",
    mesh = nvisii.mesh.create_plane("plane"),
    material = nvisii.material.create("plane"),
    transform = nvisii.transform.create("plane")
)
floor.get_transform().set_scale(nvisii.vec3(10000))
floor.get_transform().set_position(nvisii.vec3(0, 0, -5))
floor.get_material().set_base_color(nvisii.vec3(1.0))
floor.get_material().set_roughness(0)
floor.get_material().set_specular(0)

# # # # # # # # # # # # # # # # # # # # # # # # #

# This function loads the 
sdb = nvisii.import_scene(
    'content/bmw/bmw.obj', #obj path
    nvisii.vec3(0,0,0), # translation 
    nvisii.vec3(1), # scale here
    nvisii.angleAxis(3.14 * .5, nvisii.vec3(1,0,0)), #rotation here
    args=["verbose"]
)

# mirror = nvisii.material.get('sdbMirror')

# mirror.set_roughness(0)
# mirror.set_metallic(1)
# mirror.set_base_color(nvisii.vec3(1))

hl = nvisii.light.create("headlights")
tl = nvisii.light.create("taillights")
tl.set_color(nvisii.vec3(1,0,0))
hl.set_intensity(1000)
tl.set_intensity(1000)

#%%
for i_s, s in enumerate(sdb.entities):
    # print(s.get_name())
    # if 'car' in s.get_name():
    #     print(s.get_name())
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
        s.set_material(nvisii.material.create(s.get_name().lower()))
        # s.get_material().set_ior(1.0)
        s.get_material().set_base_color(nvisii.vec3(0.6,0.6,0.6))
        s.get_material().set_transmission(0)
        s.get_material().set_roughness(0.1)
        s.get_material().set_metallic(1)

    if "lightsglass" in s.get_name().lower() or "window" in s.get_name().lower():
        print(s.get_name())
        s.clear_material()
        s.set_material(nvisii.material.create(s.get_name().lower()))
        s.get_material().set_ior(1.0)
        # s.get_material().set_base_color(nvisii.vec3(1,1,1))
        s.get_material().set_transmission(1)
        s.get_material().set_roughness(0)
        s.get_material().set_metallic(0)
        s.set_visibility(shadow = False)
    
    if "interior" in s.get_name().lower():
        s.get_material().set_base_color((1,1,1))

    # elif 'light' in s.get_name().lower():
    #     print(s.get_name())
    #     s.set_light(nvisii.light.create('light' + str(i_s)))
    #     s.get_light().set_intensity(20)
    #     s.get_light().set_temperature(5000)
    # if 'tire' in s.get_name().lower():
        # s.get_transform().set_angular_velocity(nvisii.angleAxis(3.14 * .05, nvisii.vec3(1,0,0)))

# nvisii.entity.get("sdbcarShell_1").get_material().set_base_color(nvisii.vec3(1,0,0))

# nvisii.render_to_png(1024, 1024, 1000, "motion_blur_3")

if opt.control:
    camera.get_transform().clear_motion()

    cursor = nvisii.vec4()
    speed_camera = 4.0
    rot = nvisii.vec2(nvisii.pi() * 1.25, 0)
    def interact():
        global speed_camera
        global cursor
        global rot

        # nvisii camera matrix 
        cam_matrix = camera.get_transform().get_local_to_world_matrix()
        dt = nvisii.vec4(0,0,0,0)

        # translation
        if nvisii.is_button_held("W"): dt[2] = -speed_camera
        if nvisii.is_button_held("S"): dt[2] =  speed_camera
        if nvisii.is_button_held("A"): dt[0] = -speed_camera
        if nvisii.is_button_held("D"): dt[0] =  speed_camera
        if nvisii.is_button_held("Q"): dt[1] = -speed_camera
        if nvisii.is_button_held("E"): dt[1] =  speed_camera 

        # control the camera
        if nvisii.length(dt) > 0.0:
            w_dt = cam_matrix * dt
            camera.get_transform().add_position(nvisii.vec3(w_dt))

        # camera rotation
        cursor[2] = cursor[0]
        cursor[3] = cursor[1]
        cursor[0] = nvisii.get_cursor_pos().x
        cursor[1] = nvisii.get_cursor_pos().y
        if nvisii.is_button_held("MOUSE_LEFT"):
            nvisii.set_cursor_mode("DISABLED")
            rotate_camera = True
        else:
            nvisii.set_cursor_mode("NORMAL")
            rotate_camera = False

        if rotate_camera:
            rot.x -= (cursor[0] - cursor[2]) * 0.001
            rot.y -= (cursor[1] - cursor[3]) * 0.001
            init_rot = nvisii.angleAxis(nvisii.pi() * .5, (1,0,0))
            yaw = nvisii.angleAxis(rot.x, (0,1,0))
            pitch = nvisii.angleAxis(rot.y, (1,0,0)) 
            camera.get_transform().set_rotation(init_rot * yaw * pitch)

        # change speed movement
        if nvisii.is_button_pressed("UP"):
            speed_camera *= 0.5 
            print('decrease speed camera', speed_camera)
        if nvisii.is_button_pressed("DOWN"):
            speed_camera /= 0.5
            print('increase speed camera', speed_camera)
            
    nvisii.register_pre_render_callback(interact)
    import time
    while not nvisii.should_window_close(): 
        time.sleep(.1)

nvisii.render_to_file(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    file_path=f"{opt.out}"
)

# let's clean up the GPU
nvisii.deinitialize()