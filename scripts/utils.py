import random 
import visii
import randomcolor
import math 

def add_random_obj(name = "name",
    x_lim = [-1,1],
    y_lim = [-1,1],
    z_lim = [-1,1],
    scale_lim = [0.01,1]
    ):

    obj= visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
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
        random.uniform(x_lim[0],x_lim[1]),
        random.uniform(y_lim[0],y_lim[1]),
        random.uniform(z_lim[0],z_lim[1])        
    )
    
    obj.get_transform().set_rotation(
        visii.quat(1.0 ,random.random(), random.random(), random.random()) 
    )    

    obj.get_transform().set_scale(random.uniform(scale_lim[0],scale_lim[1]))
    
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

def random_light(obj_id,
    intensity_lim = [50000,100000],
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
        

    obj_mat.set_roughness(random.uniform(0,1)) # default is 1  
    obj_mat.set_metallic(random.uniform(0,1))  # degault is 0     
    obj_mat.set_transmission(random.uniform(0,1))  # degault is 0     
    obj_mat.set_sheen(random.uniform(0,1))  # degault is 0     
    obj_mat.set_clearcoat(random.uniform(0,1))  # degault is 0     
    obj_mat.set_specular(random.uniform(0,1))  # degault is 0     
    obj_mat.set_anisotropic(random.uniform(0,1))  # degault is 0     

def distance(v0,v1=[0,0,0]):
    l2 = 0
    for i in range(len(v0)):
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





    # # Rotation
    # if not str(obj_id) in move_around.destination['rot'].keys() :
    #     move_around.destination['rot'][str(obj_id)] = Quaternion.random()
    # else:
    #     goal = move_around.destination['rot'][str(obj_id)]
    #     rot = trans.get_rotation()
    #     rot = Quaternion(rot.w,rot.x,rot.y,rot.z)
    #     if Quaternion.sym_distance(goal, rot) < 0.1:
    #         move_around.destination['rot'][str(obj_id)] = Quaternion.random()    
    #         goal = move_around.destination['rot'][str(obj_id)]
    #     dir_vec = Quaternion.slerp(rot,goal,0.01)
    #     q = visii.quat()
    #     q.w,q.x,q.y,q.z = dir_vec.w,dir_vec.x,dir_vec.y,dir_vec.z
    #     trans.set_rotation(q)

    # # color
    # if not str(obj_id) in move_around.destination['color'].keys() :
    #     c = eval(str(rcolor.generate(format_='rgb')[0])[3:])
    #     move_around.destination['color'][str(obj_id)] = np.array(c)/255.0

    # else:
    #     goal = move_around.destination['color'][str(obj_id)]
    #     current = visii.material.get(str(obj_id)).get_base_color()
    #     current = np.array([current[0],current[1],current[2]])

    #     if np.linalg.norm(goal - current) < 0.1:
    #         c = eval(str(rcolor.generate(format_='rgb')[0])[3:])
    #         move_around.destination['color'][str(obj_id)] = np.array(c)/255
    #         goal = move_around.destination['color'][str(obj_id)]


    #     dir_vec = normalized(np.array(goal) - current)[0] * 0.01
    #     color = current + dir_vec
    #     color[color>1]=1
    #     color[color<0]=0

    #     visii.material.get(str(obj_id)).set_base_color(
    #         color[0],
    #         color[1],
    #         color[2]
    #     )

    # # Materials - roughness
    # if not str(obj_id) in move_around.destination['roughness'].keys() :
    #     move_around.destination['roughness'][str(obj_id)] = np.random.uniform(0,1)

    # else:
    #     goal = move_around.destination['roughness'][str(obj_id)]
    #     current = visii.material.get(str(obj_id)).get_roughness()

    #     if np.abs(goal-current) < 0.01:
    #         move_around.destination['roughness'][str(obj_id)] = np.random.uniform(0,1)
    #         goal = move_around.destination['roughness'][str(obj_id)]

    #     interval = 0.001
    #     dir_vec = (goal - current)
    #     if dir_vec > 0:
    #         to_set = current + interval
    #     else:
    #         to_set = current - interval
    #     if to_set>1:
    #         to_set = 1
    #     if to_set<0:
    #         to_set = 0

    #     visii.material.get(str(obj_id)).set_roughness(to_set)

    # # Materials - metallic
    # if not str(obj_id) in move_around.destination['metallic'].keys() :
    #     move_around.destination['metallic'][str(obj_id)] = np.random.uniform(0,1)

    # else:
    #     goal = move_around.destination['metallic'][str(obj_id)]
    #     current = visii.material.get(str(obj_id)).get_metallic()

    #     if np.abs(goal-current) < 0.01:
    #         move_around.destination['metallic'][str(obj_id)] = np.random.uniform(0,1)
    #         goal = move_around.destination['metallic'][str(obj_id)]

    #     interval = 0.001
    #     dir_vec = (goal - current)
    #     if dir_vec > 0:
    #         to_set = current + interval
    #     else:
    #         to_set = current - interval
    #     if to_set>1:
    #         to_set = 1
    #     if to_set<0:
    #         to_set = 0

    #     visii.material.get(str(obj_id)).set_metallic(to_set)

    # # Materials - transmission
    # if not str(obj_id) in move_around.destination['transmission'].keys() :
    #     move_around.destination['transmission'][str(obj_id)] = np.random.uniform(0,1)

    # else:
    #     goal = move_around.destination['transmission'][str(obj_id)]
    #     current = visii.material.get(str(obj_id)).get_transmission()

    #     if np.abs(goal-current) < 0.01:
    #         move_around.destination['transmission'][str(obj_id)] = np.random.uniform(0,1)
    #         goal = move_around.destination['transmission'][str(obj_id)]

    #     interval = 0.001
    #     dir_vec = (goal - current)
    #     if dir_vec > 0:
    #         to_set = current + interval
    #     else:
    #         to_set = current - interval
    #     if to_set>1:
    #         to_set = 1
    #     if to_set<0:
    #         to_set = 0

    #     visii.material.get(str(obj_id)).set_transmission(to_set)

    # # Materials - sheen
    # if not str(obj_id) in move_around.destination['sheen'].keys() :
    #     move_around.destination['sheen'][str(obj_id)] = np.random.uniform(0,1)

    # else:
    #     goal = move_around.destination['sheen'][str(obj_id)]
    #     current = visii.material.get(str(obj_id)).get_sheen()

    #     if np.abs(goal-current) < 0.01:
    #         move_around.destination['sheen'][str(obj_id)] = np.random.uniform(0,1)
    #         goal = move_around.destination['sheen'][str(obj_id)]

    #     interval = 0.001
    #     dir_vec = (goal - current)
    #     if dir_vec > 0:
    #         to_set = current + interval
    #     else:
    #         to_set = current - interval
    #     if to_set>1:
    #         to_set = 1
    #     if to_set<0:
    #         to_set = 0

    #     visii.material.get(str(obj_id)).set_sheen(to_set)
    
    # # Materials - clearcoat
    # if not str(obj_id) in move_around.destination['clearcoat'].keys() :
    #     move_around.destination['clearcoat'][str(obj_id)] = np.random.uniform(0,1)

    # else:
    #     goal = move_around.destination['clearcoat'][str(obj_id)]
    #     current = visii.material.get(str(obj_id)).get_clearcoat()

    #     if np.abs(goal-current) < 0.01:
    #         move_around.destination['clearcoat'][str(obj_id)] = np.random.uniform(0,1)
    #         goal = move_around.destination['clearcoat'][str(obj_id)]

    #     interval = 0.001
    #     dir_vec = (goal - current)
    #     if dir_vec > 0:
    #         to_set = current + interval
    #     else:
    #         to_set = current - interval
    #     if to_set>1:
    #         to_set = 1
    #     if to_set<0:
    #         to_set = 0

    #     visii.material.get(str(obj_id)).set_clearcoat(to_set)

    # # Materials - specular
    # if not str(obj_id) in move_around.destination['specular'].keys() :
    #     move_around.destination['specular'][str(obj_id)] = np.random.uniform(0,1)

    # else:
    #     goal = move_around.destination['specular'][str(obj_id)]
    #     current = visii.material.get(str(obj_id)).get_specular()

    #     if np.abs(goal-current) < 0.01:
    #         move_around.destination['specular'][str(obj_id)] = np.random.uniform(0,1)
    #         goal = move_around.destination['specular'][str(obj_id)]

    #     interval = 0.001
    #     dir_vec = (goal - current)
    #     if dir_vec > 0:
    #         to_set = current + interval
    #     else:
    #         to_set = current - interval
    #     if to_set>1:
    #         to_set = 1
    #     if to_set<0:
    #         to_set = 0

    #     visii.material.get(str(obj_id)).set_specular(to_set)


    # # Materials - anisotropic
    # if not str(obj_id) in move_around.destination['anisotropic'].keys() :
    #     move_around.destination['anisotropic'][str(obj_id)] = np.random.uniform(0,1)

    # else:
    #     goal = move_around.destination['anisotropic'][str(obj_id)]
    #     current = visii.material.get(str(obj_id)).get_anisotropic()

    #     if np.abs(goal-current) < 0.01:
    #         move_around.destination['anisotropic'][str(obj_id)] = np.random.uniform(0,1)
    #         goal = move_around.destination['anisotropic'][str(obj_id)]

    #     interval = 0.001
    #     dir_vec = (goal - current)
    #     if dir_vec > 0:
    #         to_set = current + interval
    #     else:
    #         to_set = current - interval
    #     if to_set>1:
    #         to_set = 1
    #     if to_set<0:
    #         to_set = 0

    #     visii.material.get(str(obj_id)).set_anisotropic(to_set)
