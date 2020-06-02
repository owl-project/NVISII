import visii 
import numpy as np 
from PIL import Image 
import PIL
import time 
from pyquaternion import Quaternion
import randomcolor
import subprocess

NB_OBJS = 100000
NB_TRACE_PER_PIXEL = 320
NB_FRAMES = 300

# WIDTH = 1920
# HEIGHT = 1080

WIDTH = 1000
HEIGHT = 1000


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
        visii.vec3(0,0,5), # camera_origin
        visii.vec3(0,0,0), # look at (world coordinate)
        visii.vec3(1,0,0), # up vector
    )
)

# set the camera
visii.set_camera_entity(camera_entity)
meshes = [] 

rcolor = randomcolor.RandomColor()

def add_random_obj(name = "name"):
    global rcolor
    obj= visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
    )

    obj_id = np.random.randint(0,16)

    mesh = None
    if obj_id == 0:
        mesh = visii.mesh.create_sphere(name)
    if obj_id == 1:
        mesh = visii.mesh.create_torus_knot(name, 
            np.random.randint(2,6), 
            np.random.randint(4,10))
    if obj_id == 2:
        mesh = visii.mesh.create_teapotahedron(name)
    if obj_id == 3:
        mesh = visii.mesh.create_box(name)
    if obj_id == 4:
        mesh = visii.mesh.create_capped_cone(name)
    if obj_id == 5:
        mesh = visii.mesh.create_capped_cylinder(name)
    if obj_id == 6:
        mesh = visii.mesh.create_capsule(name)
    if obj_id == 7:
        mesh = visii.mesh.create_cylinder(name)
    if obj_id == 8:
        mesh = visii.mesh.create_disk(name)
    if obj_id == 9:
        mesh = visii.mesh.create_dodecahedron(name)
    if obj_id == 10:
        mesh = visii.mesh.create_icosahedron(name)
    if obj_id == 11:
        mesh = visii.mesh.create_icosphere(name)
    if obj_id == 12:
        mesh = visii.mesh.create_rounded_box(name)
    if obj_id == 13:
        mesh = visii.mesh.create_spring(name)
    if obj_id == 14:
        mesh = visii.mesh.create_torus(name)
    if obj_id == 15:
        mesh = visii.mesh.create_tube(name)

    obj.set_mesh(mesh)
    obj.get_transform().set_position(
        # np.random.uniform(-5,5),
        # np.random.uniform(-5,5),
        # np.random.uniform(-5,5)
        np.random.uniform(-0.5,0.5),
        np.random.uniform(-0.5,0.5),
        np.random.uniform(-0.5,0.5)
        
        )
    q = Quaternion.random()
    obj.get_transform().set_rotation(
        visii.quat(q.w,q.x,q.y,q.z)
        )
    obj.get_transform().set_scale(np.random.uniform(0.01,0.2))
    
    # obj.get_transform().set_scale(
    #     np.random.uniform(0.01,0.2),
    #     np.random.uniform(0.01,0.2),
    #     np.random.uniform(0.01,0.2)
    # )

    # material random

    # obj.get_material().set_base_color(
    #     np.random.uniform(0,1),
    #     np.random.uniform(0,1),
    #     np.random.uniform(0,1)) 

    c = eval(str(rcolor.generate(format_='rgb')[0])[3:])
    obj.get_material().set_base_color(
        c[0]/255.0,
        c[1]/255.0,
        c[2]/255.0)  


    obj.get_material().set_roughness(np.random.uniform(0,1)) # default is 1  
    obj.get_material().set_metallic(np.random.uniform(0,1))  # degault is 0     
    obj.get_material().set_transmission(np.random.uniform(0,1))  # degault is 0     
    obj.get_material().set_sheen(np.random.uniform(0,1))  # degault is 0     
    obj.get_material().set_clearcoat(np.random.uniform(0,1))  # degault is 0     
    obj.get_material().set_specular(np.random.uniform(0,1))  # degault is 0     
    obj.get_material().set_anisotropic(np.random.uniform(0,1))  # degault is 0     


def move_around(obj_id):
    trans = visii.transform.get(str(obj_id))
    # trans = visii.entity.get(str(obj_id)).get_transform()
    # trans.add_position(
    #     np.random.uniform(-0.5,0.5),
    #     np.random.uniform(-0.5,0.5),
    #     np.random.uniform(-0.5,0.5)
    # )

    def normalized(a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2==0] = 1
        return a / np.expand_dims(l2, axis)

    # Position    
    if not str(obj_id) in move_around.destination['pos'].keys() :
        move_around.destination['pos'][str(obj_id)] = [
            np.random.uniform(-5,5),
            np.random.uniform(-5,5),
            np.random.uniform(-5,5)
        ]
    else:
        goal = move_around.destination['pos'][str(obj_id)]
        pos = trans.get_world_translation()
        if np.linalg.norm(np.array(goal) - np.array([pos[0],pos[1],pos[2]])) < 0.1:
            move_around.destination['pos'][str(obj_id)] = [
                np.random.uniform(-5,5),
                np.random.uniform(-5,5),
                np.random.uniform(-5,5)
            ]    
            goal = move_around.destination['pos'][str(obj_id)]

        dir_vec = normalized(np.array(goal) - np.array([pos[0],pos[1],pos[2]]))[0] * 0.005
        trans.add_position(dir_vec[0],dir_vec[1],dir_vec[2])

    # Rotation
    if not str(obj_id) in move_around.destination['rot'].keys() :
        move_around.destination['rot'][str(obj_id)] = Quaternion.random()
    else:
        goal = move_around.destination['rot'][str(obj_id)]
        rot = trans.get_rotation()
        rot = Quaternion(rot.w,rot.x,rot.y,rot.z)
        if Quaternion.sym_distance(goal, rot) < 0.1:
            move_around.destination['rot'][str(obj_id)] = Quaternion.random()    
            goal = move_around.destination['rot'][str(obj_id)]
        dir_vec = Quaternion.slerp(rot,goal,0.01)
        q = visii.quat()
        q.w,q.x,q.y,q.z = dir_vec.w,dir_vec.x,dir_vec.y,dir_vec.z
        trans.set_rotation(q)

    # color
    if not str(obj_id) in move_around.destination['color'].keys() :
        c = eval(str(rcolor.generate(format_='rgb')[0])[3:])
        move_around.destination['color'][str(obj_id)] = np.array(c)/255.0

    else:
        goal = move_around.destination['color'][str(obj_id)]
        current = visii.material.get(str(obj_id)).get_base_color()
        current = np.array([current[0],current[1],current[2]])

        if np.linalg.norm(goal - current) < 0.1:
            c = eval(str(rcolor.generate(format_='rgb')[0])[3:])
            move_around.destination['color'][str(obj_id)] = np.array(c)/255
            goal = move_around.destination['color'][str(obj_id)]


        dir_vec = normalized(np.array(goal) - current)[0] * 0.01
        color = current + dir_vec
        color[color>1]=1
        color[color<0]=0

        visii.material.get(str(obj_id)).set_base_color(
            color[0],
            color[1],
            color[2]
        )

    # Materials - roughness
    if not str(obj_id) in move_around.destination['roughness'].keys() :
        move_around.destination['roughness'][str(obj_id)] = np.random.uniform(0,1)

    else:
        goal = move_around.destination['roughness'][str(obj_id)]
        current = visii.material.get(str(obj_id)).get_roughness()

        if np.abs(goal-current) < 0.01:
            move_around.destination['roughness'][str(obj_id)] = np.random.uniform(0,1)
            goal = move_around.destination['roughness'][str(obj_id)]

        interval = 0.001
        dir_vec = (goal - current)
        if dir_vec > 0:
            to_set = current + interval
        else:
            to_set = current - interval
        if to_set>1:
            to_set = 1
        if to_set<0:
            to_set = 0

        visii.material.get(str(obj_id)).set_roughness(to_set)

    # Materials - metallic
    if not str(obj_id) in move_around.destination['metallic'].keys() :
        move_around.destination['metallic'][str(obj_id)] = np.random.uniform(0,1)

    else:
        goal = move_around.destination['metallic'][str(obj_id)]
        current = visii.material.get(str(obj_id)).get_metallic()

        if np.abs(goal-current) < 0.01:
            move_around.destination['metallic'][str(obj_id)] = np.random.uniform(0,1)
            goal = move_around.destination['metallic'][str(obj_id)]

        interval = 0.001
        dir_vec = (goal - current)
        if dir_vec > 0:
            to_set = current + interval
        else:
            to_set = current - interval
        if to_set>1:
            to_set = 1
        if to_set<0:
            to_set = 0

        visii.material.get(str(obj_id)).set_metallic(to_set)

    # Materials - transmission
    if not str(obj_id) in move_around.destination['transmission'].keys() :
        move_around.destination['transmission'][str(obj_id)] = np.random.uniform(0,1)

    else:
        goal = move_around.destination['transmission'][str(obj_id)]
        current = visii.material.get(str(obj_id)).get_transmission()

        if np.abs(goal-current) < 0.01:
            move_around.destination['transmission'][str(obj_id)] = np.random.uniform(0,1)
            goal = move_around.destination['transmission'][str(obj_id)]

        interval = 0.001
        dir_vec = (goal - current)
        if dir_vec > 0:
            to_set = current + interval
        else:
            to_set = current - interval
        if to_set>1:
            to_set = 1
        if to_set<0:
            to_set = 0

        visii.material.get(str(obj_id)).set_transmission(to_set)

    # Materials - sheen
    if not str(obj_id) in move_around.destination['sheen'].keys() :
        move_around.destination['sheen'][str(obj_id)] = np.random.uniform(0,1)

    else:
        goal = move_around.destination['sheen'][str(obj_id)]
        current = visii.material.get(str(obj_id)).get_sheen()

        if np.abs(goal-current) < 0.01:
            move_around.destination['sheen'][str(obj_id)] = np.random.uniform(0,1)
            goal = move_around.destination['sheen'][str(obj_id)]

        interval = 0.001
        dir_vec = (goal - current)
        if dir_vec > 0:
            to_set = current + interval
        else:
            to_set = current - interval
        if to_set>1:
            to_set = 1
        if to_set<0:
            to_set = 0

        visii.material.get(str(obj_id)).set_sheen(to_set)
    
    # Materials - clearcoat
    if not str(obj_id) in move_around.destination['clearcoat'].keys() :
        move_around.destination['clearcoat'][str(obj_id)] = np.random.uniform(0,1)

    else:
        goal = move_around.destination['clearcoat'][str(obj_id)]
        current = visii.material.get(str(obj_id)).get_clearcoat()

        if np.abs(goal-current) < 0.01:
            move_around.destination['clearcoat'][str(obj_id)] = np.random.uniform(0,1)
            goal = move_around.destination['clearcoat'][str(obj_id)]

        interval = 0.001
        dir_vec = (goal - current)
        if dir_vec > 0:
            to_set = current + interval
        else:
            to_set = current - interval
        if to_set>1:
            to_set = 1
        if to_set<0:
            to_set = 0

        visii.material.get(str(obj_id)).set_clearcoat(to_set)

    # Materials - specular
    if not str(obj_id) in move_around.destination['specular'].keys() :
        move_around.destination['specular'][str(obj_id)] = np.random.uniform(0,1)

    else:
        goal = move_around.destination['specular'][str(obj_id)]
        current = visii.material.get(str(obj_id)).get_specular()

        if np.abs(goal-current) < 0.01:
            move_around.destination['specular'][str(obj_id)] = np.random.uniform(0,1)
            goal = move_around.destination['specular'][str(obj_id)]

        interval = 0.001
        dir_vec = (goal - current)
        if dir_vec > 0:
            to_set = current + interval
        else:
            to_set = current - interval
        if to_set>1:
            to_set = 1
        if to_set<0:
            to_set = 0

        visii.material.get(str(obj_id)).set_specular(to_set)


    # Materials - anisotropic
    if not str(obj_id) in move_around.destination['anisotropic'].keys() :
        move_around.destination['anisotropic'][str(obj_id)] = np.random.uniform(0,1)

    else:
        goal = move_around.destination['anisotropic'][str(obj_id)]
        current = visii.material.get(str(obj_id)).get_anisotropic()

        if np.abs(goal-current) < 0.01:
            move_around.destination['anisotropic'][str(obj_id)] = np.random.uniform(0,1)
            goal = move_around.destination['anisotropic'][str(obj_id)]

        interval = 0.001
        dir_vec = (goal - current)
        if dir_vec > 0:
            to_set = current + interval
        else:
            to_set = current - interval
        if to_set>1:
            to_set = 1
        if to_set<0:
            to_set = 0

        visii.material.get(str(obj_id)).set_anisotropic(to_set)



# create a random scene, the function defines the values
for i in range(NB_OBJS):
    add_random_obj(str(i))

################################################################
# Animation
move_around.destination = {
    "pos":{},
    "rot":{},
    'color':{},
    'roughness':{},
    'metallic':{},
    'transmission':{},
    'sheen':{},
    'clearcoat':{},
    'specular':{},
    'anisotropic':{},
    } 

for i in range(NB_FRAMES): 

    print(f"outf/{str(i).zfill(4)}.png")
    for obj_id in range(NB_OBJS): 
        move_around(obj_id)

    # time.sleep(SLEEP_TIME)
    # a = [512*512*4]
    # visii.get_buffer_width(), visii.get_buffer_height()
    # x = np.array(visii.read_frame_buffer()).reshape(512,512,4)
    x = visii.render(width=WIDTH, height=HEIGHT, samples_per_pixel=NB_TRACE_PER_PIXEL)
    x = np.array(x).reshape(HEIGHT,WIDTH,4)

    img = Image.fromarray((x*255).astype(np.uint8)).transpose(PIL.Image.FLIP_TOP_BOTTOM)

    img.save(f"outf/{str(i).zfill(4)}.png")



visii.cleanup()
# ffmpeg -y -framerate 15 -pattern_type glob -i "outf/*.png" output.mp4
subprocess.call(['ffmpeg', '-y',\
    '-framerate', '15', '-pattern_type', 'glob', '-i',\
    "outf/*.png", 'output.mp4'])
