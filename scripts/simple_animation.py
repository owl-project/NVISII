# import sys, os
# os.add_dll_directory(os.path.join(os.getcwd(), '..', 'install'))
# sys.path.append(os.path.join(os.getcwd(), "..", "install"))

import visii 
import numpy as np 
from PIL import Image 
import PIL
import time 
from pyquaternion import Quaternion
import randomcolor
import subprocess
import random 
from utils import * 

NB_OBJS = 1000
NB_LIGHTS = 20

SAMPLES_PER_PIXEL = 100
NB_FRAMES = 300

# WIDTH = 1920
# HEIGHT = 1080

WIDTH = 1000
HEIGHT = 1000


visii.initialize_headless()

visii.set_dome_light_intensity(0.5)



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
        visii.vec3(0,0,5), # camera_origin
        visii.vec3(0,0,0), # look at (world coordinate)
        visii.vec3(1,0,0), # up vector
    )
)

# set the camera
visii.set_camera_entity(camera_entity)




# create a random scene, the function defines the values
for i in range(NB_OBJS):
    add_random_obj(str(i),
        scale_lim = [0.01,0.1],
        x_lim = [-0.1, 0.1],
        y_lim = [-0.1, 0.1],
        z_lim = [-0.1, 0.1]
    )
    random_material(str(i))

for i in range(NB_LIGHTS):
    add_random_obj("L"+str(i),
        scale_lim = [0.5,1],
        x_lim = [-5, 5],
        y_lim = [-5, 5],
        z_lim = [8, 20]
    )
    random_light("L"+str(i))    

################################################################

visii.enable_denoiser()

for i in range(NB_FRAMES): 

    print(f"outf/{str(i).zfill(4)}.png")
    for obj_id in range(NB_OBJS): 
        random_translation(obj_id,
            x_lim = [-5,5],
            y_lim = [-5,5],
            z_lim = [-5,5])

    # time.sleep(SLEEP_TIME)
    # a = [512*512*4]
    # visii.get_buffer_width(), visii.get_buffer_height()
    # x = np.array(visii.read_frame_buffer()).reshape(512,512,4)
    # x = visii.render(width=WIDTH, height=HEIGHT, samples_per_pixel=SAMPLE_PER_PIXEL)
    # x = np.array(x).reshape(HEIGHT,WIDTH,4)

    # img = Image.fromarray((x*255).astype(np.uint8)).transpose(PIL.Image.FLIP_TOP_BOTTOM)

    # img.save(f"outf/{str(i).zfill(4)}.png")
    # time.sleep(0.5)
    visii.render_to_png(width=WIDTH, 
                    height=HEIGHT, 
                    samples_per_pixel=SAMPLES_PER_PIXEL,
                    image_path=f"outf/{str(i).zfill(4)}.png")

visii.cleanup()
# ffmpeg -y -framerate 15 -pattern_type glob -i "outf/*.png" output.mp4
subprocess.call(['ffmpeg', '-y',\
    '-framerate', '15', '-pattern_type', 'glob', '-i',\
    "outf/*.png", 'output.mp4'])
