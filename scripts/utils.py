
import random 
import visii
import randomcolor
import math 

def add_random_obj(name = "name",
    x_lim = [-1,1],
    y_lim = [-1,1],
    z_lim = [-1,1],
    scale_lim = [0.01,1],
    obj_id = None,
    ):
    
    # obj = visii.entity.get(name)
    # if obj is None:
    obj= visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
    )
    if obj_id is None:
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
        random.uniform(x_lim[0],x_lim[1]),
        random.uniform(y_lim[0],y_lim[1]),
        random.uniform(z_lim[0],z_lim[1])        
    )
    
    obj.get_transform().set_rotation(
        visii.quat(1.0 ,random.random(), random.random(), random.random()) 
    )    

    obj.get_transform().set_scale(random.uniform(scale_lim[0],scale_lim[1]))
    
    return obj
add_random_obj.rcolor = randomcolor.RandomColor()
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



def random_material(obj_id,
    color = None, # list of 3 numbers between [0..1]
    ):
    obj_mat = visii.material.get(str(obj_id))
    
    if color is None:
        c = eval(str(add_random_obj.rcolor.generate(format_='rgb',luminosity='bright')[0])[3:])
        obj_mat.set_base_color(
            c[0]/255.0,
            c[1]/255.0,
            c[2]/255.0)  
    else:
        obj_mat.set_base_color(color[0],color[1],color[2])  
        
    r = random.randint(0,2)

    if r == 0:  
        # Plastic / mat
        obj_mat.set_metallic(0)  # should 0 or 1      
        obj_mat.set_transmission(0)  # should 0 or 1      
        obj_mat.set_roughness(random.uniform(0,1)) # default is 1  
    if r == 1:  
        # metallic
        obj_mat.set_metallic(random.uniform(0.9,1))  # should 0 or 1      
        obj_mat.set_transmission(0)  # should 0 or 1      
    if r == 2:  
        # glass
        obj_mat.set_metallic(0)  # should 0 or 1      
        obj_mat.set_transmission(random.uniform(0.9,1))  # should 0 or 1      

    if r > 0: # for metallic and glass
        r2 = random.randint(0,1)
        if r2 == 1: 
            obj_mat.set_roughness(random.uniform(0,.1)) # default is 1  
        else:
            obj_mat.set_roughness(random.uniform(0.9,1)) # default is 1  

    obj_mat.set_sheen(random.uniform(0,1))  # degault is 0     
    obj_mat.set_clearcoat(random.uniform(0,1))  # degault is 0     
    obj_mat.set_specular(random.uniform(0,1))  # degault is 0     

    r = random.randint(0,1)
    if r == 0:
        obj_mat.set_anisotropic(random.uniform(0,0.1))  # degault is 0     
    else:
        obj_mat.set_anisotropic(random.uniform(0.9,1))  # degault is 0     



########################################
# ANIMATION RANDOMIZATION 
########################################



def distance(v0,v1=[0,0,0]):
    l2 = 0
    try:
        for i in range(len(v0)):
            l2 += (v0[i]-v1[i])**2
    except:
        for i in range(3):
            l2 += (v0[i]-v1[i])**2
    return math.sqrt(l2)

def normalize(v):
    l2 = distance(v)
    return [v[0]/l2,v[1]/l2,v[2]/l2]

def random_translation(obj_id,
    x_lim = [-1,1],
    y_lim = [-1,1],
    z_lim = [-1,1],
    speed_lim = [0.01,0.05]
    ):
    trans = visii.transform.get(str(obj_id))

    # Position    
    if not str(obj_id) in random_translation.destinations.keys() :
        random_translation.destinations[str(obj_id)] = [
            random.uniform(x_lim[0],x_lim[1]),
            random.uniform(y_lim[0],y_lim[1]),
            random.uniform(z_lim[0],z_lim[1])
        ]
        random_translation.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
    else:
        goal = random_translation.destinations[str(obj_id)]
        pos = trans.get_world_translation()

        if distance(goal,pos) < 0.1:
            random_translation.destinations[str(obj_id)] = [
                random.uniform(x_lim[0],x_lim[1]),
                random.uniform(y_lim[0],y_lim[1]),
                random.uniform(z_lim[0],z_lim[1])
            ]
            random_translation.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])    
            goal = random_translation.destinations[str(obj_id)]

        dir_vec = normalize(
            [
                goal[0] - pos[0],
                goal[1] - pos[1],
                goal[2] - pos[2]
            ]   
        )
        
        trans.add_position(
            dir_vec[0] * random_translation.speeds[str(obj_id)],
            dir_vec[1] * random_translation.speeds[str(obj_id)],
            dir_vec[2] * random_translation.speeds[str(obj_id)]
        )


random_translation.destinations = {}
random_translation.speeds = {}


def random_rotation(obj_id,
    speed_lim = [0.01,0.05]
    ):
    from pyquaternion import Quaternion

    trans = visii.transform.get(str(obj_id))    

    # Rotation
    if not str(obj_id) in random_rotation.destinations.keys() :
        random_rotation.destinations[str(obj_id)] = Quaternion.random()
        random_rotation.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])

    else:
        goal = random_rotation.destinations[str(obj_id)]
        rot = trans.get_rotation()
        rot = Quaternion(rot.w,rot.x,rot.y,rot.z)
        if Quaternion.sym_distance(goal, rot) < 0.1:
            random_rotation.destinations[str(obj_id)] = Quaternion.random()    
            random_rotation.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
            goal = random_rotation.destinations[str(obj_id)]
        dir_vec = Quaternion.slerp(rot,goal,random_rotation.speeds[str(obj_id)])
        q = visii.quat()
        q.w,q.x,q.y,q.z = dir_vec.w,dir_vec.x,dir_vec.y,dir_vec.z
        trans.set_rotation(q)

random_rotation.destinations = {}
random_rotation.speeds = {}

def random_scale(obj_id,
    scale_lim = [0.01,0.2],
    speed_lim = [0.01,0.02],
    x_lim = None,
    y_lim = None,
    z_lim = None
    ):
    # This assumes only one dimensions gets scale

    trans = visii.transform.get(str(obj_id))    

    limit = min(speed_lim)*2
    
    if not x_lim is None:
        limit = [min(y_lim)*2,min(x_lim)*2,min(z_lim)*2]
    
    # Rotation
    if not str(obj_id) in random_scale.destinations.keys() :
        if not x_lim is None:
            random_scale.destinations[str(obj_id)] = [
                                        random.uniform(x_lim[0],x_lim[1]),
                                        random.uniform(y_lim[0],y_lim[1]),
                                        random.uniform(z_lim[0],z_lim[1])
                                        ]
        else:    
            random_scale.destinations[str(obj_id)] = random.uniform(scale_lim[0],scale_lim[1])

        random_scale.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])

    else:
        goal = random_scale.destinations[str(obj_id)]

        if x_lim is None:
            current = trans.get_scale()[0]
        else:
            current = trans.get_scale()
        
        if x_lim is None:

            if abs(goal-current) < limit:
                random_scale.destinations[str(obj_id)] = random.uniform(scale_lim[0],scale_lim[1])
                random_scale.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
                goal = random_scale.destinations[str(obj_id)]
            if goal>current:
                q = random_scale.speeds[str(obj_id)]
            else:
                q = -random_scale.speeds[str(obj_id)]
            trans.set_scale(current + q)
        else:
            limits  = [x_lim,y_lim,z_lim]
            q = [0,0,0]
            for i in range(3):
                if abs(goal[i]-current[i]) < limit[i]:
                    random_scale.destinations[str(obj_id)][i] = random.uniform(limits[i][0],limits[i][1])
                    random_scale.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])                    
                    goal = random_scale.destinations[str(obj_id)]
                if goal[i]>current[i]:
                    q[i] = random_scale.speeds[str(obj_id)]
                else:
                    q[i] = -random_scale.speeds[str(obj_id)]
            trans.set_scale(current + visii.vec3(q[0],q[1],q[2]))

random_scale.destinations = {}
random_scale.speeds = {}


def random_color(obj_id,
    speed_lim = [0.01,0.1]
    ):

    # color
    if not str(obj_id) in random_color.destinations.keys() :
        c = eval(str(random_color.rcolor.generate(luminosity='bright',format_='rgb')[0])[3:])
        random_color.destinations[str(obj_id)] = visii.vec3(c[0]/255.0, c[1]/255.0, c[2]/255.0)
        random_color.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])

    else:
        goal = random_color.destinations[str(obj_id)]
        current = visii.material.get(str(obj_id)).get_base_color()

        if distance(goal,current) < 0.1:
            random_color.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
            c = eval(str(random_color.rcolor.generate(luminosity='bright',format_='rgb')[0])[3:])
            random_color.destinations[str(obj_id)] = visii.vec3(c[0]/255.0, c[1]/255.0, c[2]/255.0)
            goal = random_color.destinations[str(obj_id)]

        target = visii.mix(current,goal,
            visii.vec3( random_color.speeds[str(obj_id)],
                        random_color.speeds[str(obj_id)],
                        random_color.speeds[str(obj_id)]
                        )
        ) 

        visii.material.get(str(obj_id)).set_base_color(target)

random_color.destinations = {}
random_color.speeds = {}
random_color.rcolor = randomcolor.RandomColor()

######## RANDOM LIGHTS ############

def random_light(obj_id,
    intensity_lim = [5000,10000],
    color = None,
    temperature_lim  = [100,10000],
    ):

    obj = visii.entity.get(str(obj_id))
    obj.set_light(visii.light.create(str(obj_id)))

    obj.get_light().set_intensity(random.uniform(intensity_lim[0],intensity_lim[1]))
    # obj.get_light().set_temperature(np.random.randint(100,9000))


    if not color is None:
        obj.get_material().set_base_color(color[0],color[1],color[2])  
        # c = eval(str(rcolor.generate(luminosity='bright',format_='rgb')[0])[3:])
        # obj.get_light().set_color(
        #     c[0]/255.0,
        #     c[1]/255.0,
        #     c[2]/255.0)  
       
    else:
        obj.get_light().set_temperature(random.uniform(temperature_lim[0],temperature_lim[1]))

def random_intensity(obj_id,
    intensity_lim = [5000,10000],
    speed_lim = [100,1000]
    ):

    obj = visii.entity.get(str(obj_id)).get_light()

    if not str(obj_id) in random_intensity.destinations.keys() :
        random_intensity.destinations[str(obj_id)] = random.uniform(intensity_lim[0],intensity_lim[1])
        random_intensity.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
        random_intensity.current[str(obj_id)] = random_intensity.destinations[str(obj_id)]
        obj.set_intensity(random_intensity.current[str(obj_id)]) 
    else:
        goal = random_intensity.destinations[str(obj_id)]
        current = random_intensity.current[str(obj_id)]
        
        if abs(goal-current) < min(speed_lim)*2:
            random_intensity.destinations[str(obj_id)] = random.uniform(intensity_lim[0],intensity_lim[1])
            random_intensity.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
            goal = random_intensity.destinations[str(obj_id)]
        if goal>current:
            q = random_intensity.speeds[str(obj_id)]
        else:
            q = -random_intensity.speeds[str(obj_id)]
        obj.set_intensity(random_intensity.current[str(obj_id)] + q)
        random_intensity.current[str(obj_id)] = random_intensity.current[str(obj_id)] + q

random_intensity.destinations = {}
random_intensity.current = {}
random_intensity.speeds = {}

######## NDDS ##########
def add_cuboid(name):
    obj = visii.entity.get(name)

    min_obj = obj.get_mesh().get_min_aabb_corner()
    max_obj = obj.get_mesh().get_max_aabb_corner()
    centroid_obj = obj.get_mesh().get_aabb_center()

    # TODO CHECK WHICH POINT IS WHICH
    cuboid = [
        visii.vec3(max_obj[0], max_obj[1], max_obj[2]),
        visii.vec3(min_obj[0], max_obj[1], max_obj[2]),
        visii.vec3(max_obj[0], min_obj[1], max_obj[2]),
        visii.vec3(max_obj[0], max_obj[1], min_obj[2]),
        visii.vec3(min_obj[0], min_obj[1], max_obj[2]),
        visii.vec3(max_obj[0], min_obj[1], min_obj[2]),
        visii.vec3(min_obj[0], max_obj[1], min_obj[2]),
        visii.vec3(min_obj[0], min_obj[1], min_obj[2]),
        visii.vec3(centroid_obj[0], centroid_obj[1], centroid_obj[2]), 
    ]

    for i_p, p in enumerate(cuboid):
        child_transform = visii.transform.create(f"{name}_cuboid_{i_p}")
        child_transform.set_position(p)
        child_transform.set_parent(obj.get_transform().get_id())

def get_cuboid_image_space(obj_id, camera_name = 'my_camera'):
    # return cubdoid + centroid projected to the image, values [0..1]
    # This assumes that only the cam_view is used.

    # cam_matrix = camera_entity.transform().get_world_to_local_matrix()

    cam_view_matrix = visii.entity.get(camera_name).get_camera().get_view()
    cam_proj_matrix = visii.entity.get(camera_name).get_camera().get_projection()

    points = []
    for i_t in range(9):
        trans = visii.transform.get(f"{obj_id}_cuboid_{i_t}")
        pos_m = visii.vec4(
            trans.get_world_translation()[0],
            trans.get_world_translation()[1],
            trans.get_world_translation()[2],
            1)
      
        p_image = cam_proj_matrix * (cam_view_matrix * pos_m) 
        p_image = visii.vec2(p_image) / p_image.w
        p_image = p_image * visii.vec2(1,-1)
        p_image = (p_image + visii.vec2(1,1)) * 0.5
        points.append([p_image[0],p_image[1]])

    return points


def export_to_ndds_file(
    filename = "tmp.json", #this has to include path as well
    obj_names = [], # this is a list of ids to load and export
    height = 500, 
    width = 500,
    camera_name = 'my_camera',
    ):
    # To do export things in the camera frame, e.g., pose and quaternion

    import simplejson as json

    # assume we only use the view camera
    cam_view_matrix = visii.entity.get(camera_name).get_camera().get_view()
    cam_world_translation = [cam_view_matrix[3][0],cam_view_matrix[3][1],cam_view_matrix[3][2]] 
    cam_world_quaternion = visii.quat_cast(cam_view_matrix)

    dict_out = {
                    "camera_data" : {
                        "width" : width,
                        'height' : height,
                        'location_world':cam_world_translation,
                        'quaternion_world_xyzw':[
                            cam_world_quaternion[0],
                            cam_world_quaternion[1],
                            cam_world_quaternion[2],
                            cam_world_quaternion[3],
                            ],
                    }, 
                    "objects" : []
                }

    for obj_name in obj_names: 

        projected_keypoints = get_cuboid_image_space(obj_name,camera_name=camera_name)

        # put them in the image space. 
        for i_p, p in enumerate(projected_keypoints):
            projected_keypoints[i_p] = [p[0]*width, p[1]*height]

        # Get the location and rotation of the object in the camera frame 

        trans = visii.transform.get(obj_name)
        obj_matrix = cam_view_matrix * trans.get_local_to_world_matrix()
        quaternion_xyzw = visii.quat_cast(obj_matrix)
        translation = [obj_matrix[3][0],obj_matrix[3][1],obj_matrix[3][2]] 


        dict_out['objects'].append({
            'class':obj_name,
            'location':translation,
            'quaternion_xyzw':[
                    quaternion_xyzw[0],
                    quaternion_xyzw[1],
                    quaternion_xyzw[2],
                    quaternion_xyzw[3],
                ],
            'projected_cuboid':projected_keypoints,
            
        })
    
    # if os.path.exists(filename):
    #     with open(path_json) as f:
    #         data = json.load(f)
    #     dict_out['objects'] = data['objects'] + dict_out['objects']
    
    with open(filename, 'w+') as fp:
        json.dump(dict_out, fp, indent=4, sort_keys=True)

#################### BULLET THINGS ##############################

import pybullet as p


def create_obj(
    name = 'name',
    path_obj = "",
    path_tex = "",
    scale = 1, 
    rot_base = None
    ):

    
    # This is for YCB like dataset
    obj_mesh = visii.mesh.create_from_obj(name, path_obj)
    obj_texture = visii.texture.create_from_image(name,path_tex)

    obj_entity = visii.entity.create(
        name=name,
        # mesh = visii.mesh.create_sphere("mesh1", 1, 128, 128),
        mesh = obj_mesh,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
    )

    # should randomize
    obj_entity.get_material().set_metallic(0)  # should 0 or 1      
    obj_entity.get_material().set_transmission(0)  # should 0 or 1      
    obj_entity.get_material().set_roughness(random.uniform(0,1)) # default is 1  
    obj_entity.get_material().set_roughness(1) # default is 1  

    obj_entity.get_material().set_base_color_texture(obj_texture)

    obj_entity.get_transform().set_scale(visii.vec3(scale))

    return obj_entity

def create_physics(
    aabb,             # list two vec3 (min,max) as visii vec3
    base_position = [0,0,0],    # list x,y,z
    base_orientation = [0,0,0,1], # list x,y,z,w 
    base_rot = None,  # visii quat to rotate the frame of the object
    type_collision = p.GEOM_MESH, # so far that is the only one
    scale = 1,        # scale in all directions
    mass = 1,         # mass in kg
    mesh_path="",      # path to the obj mesh
    name="",
    ):

    if not base_rot is None:    
        min_vec4 = visii.vec4(aabb[0][0],aabb[0][1],aabb[0][2],1)
        max_vec4 = visii.vec4(aabb[1][0],aabb[1][1],aabb[1][2],1)

        rot_min = base_rot * min_vec4
        rot_max = base_rot * max_vec4
        
        aabb = [
                visii.vec3(rot_min[0],rot_min[1],rot_min[2]),
                visii.vec3(rot_max[0],rot_max[1],rot_max[2]),
                ]
    # if type_collision == p.GEOM_BOX:
    # print(visii.mesh.get(name).get_vertices())

    vertices = []
    for v in visii.mesh.get(name).get_vertices():
        vertices.append([v[0],v[1],v[2]])
    
    print(len(vertices))

    if type_collision == p.GEOM_MESH:
        obj_col_id = p.createCollisionShape(
            type_collision,
            meshScale = [scale,scale,scale],
            # fileName = mesh_path
            vertices = vertices,
            # vertices = [
            # [0,0,0],
            # [0,0,1],
            # [0,1,0],
            # [1,0,0],
            # [1,0,1],
            # [1,1,0],
            # [0,1,1],
            # [1,1,1],
            # ]
            )

    # if type_collision == p.GEOM_CAPSULE:
    #     radius = max(aabb[1][0],aabb[1][1])
    #     height = aabb[1][2] * 2 

    #     obj_col_id = p.createCollisionShape(
    #         type_collision,
    #         radius = radius * scale,
    #         height = height * scale
    #     )


    obj_id = p.createMultiBody(  
                        baseMass = 1, 
                        baseCollisionShapeIndex = obj_col_id,
                        basePosition = base_position,
                        baseOrientation= base_orientation,
                        )

    return obj_id


def update_pose(obj_dict):
    pos, rot = p.getBasePositionAndOrientation(obj_dict['bullet_id'])


    obj_entity = visii.entity.get(obj_dict['visii_id'])
    obj_entity.get_transform().set_position(visii.vec3(
                                            pos[0],
                                            pos[1],
                                            pos[2]
                                            )
                                        )
    if not obj_dict['base_rot'] is None: 
        obj_entity.get_transform().set_rotation(visii.quat(
                                                rot[3],
                                                rot[0],
                                                rot[1],
                                                rot[2]
                                                ) * obj_dict['base_rot']   
                                            )
    else:
        obj_entity.get_transform().set_rotation(visii.quat(
                                                rot[3],
                                                rot[0],
                                                rot[1],
                                                rot[2]
                                                )   
                                            )
