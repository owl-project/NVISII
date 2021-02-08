import nvisii

opt = lambda : None
opt.spp = 100 
opt.width = 500
opt.height = 500 
opt.out = "21_ndds_export.png" 

# # # # # # # # # # # # # # # # # # # # # # # # #
# Function to add cuboid information to an object using 
def add_cuboid(name, debug=False):
    """
    Add cuboid children to the transform tree to a given object for exporting

    :param name: string name of the nvisii entity to add a cuboid
    :param debug:   bool - add sphere on the nvisii entity to make sure the  
                    cuboid is located at the right place. 

    :return: return a list of cuboid in canonical space of the object. 
    """

    obj = nvisii.entity.get(name)

    min_obj = obj.get_mesh().get_min_aabb_corner()
    max_obj = obj.get_mesh().get_max_aabb_corner()
    centroid_obj = obj.get_mesh().get_aabb_center()

    cuboid = [
        nvisii.vec3(max_obj[0], max_obj[1], max_obj[2]),
        nvisii.vec3(min_obj[0], max_obj[1], max_obj[2]),
        nvisii.vec3(max_obj[0], min_obj[1], max_obj[2]),
        nvisii.vec3(max_obj[0], max_obj[1], min_obj[2]),
        nvisii.vec3(min_obj[0], min_obj[1], max_obj[2]),
        nvisii.vec3(max_obj[0], min_obj[1], min_obj[2]),
        nvisii.vec3(min_obj[0], max_obj[1], min_obj[2]),
        nvisii.vec3(min_obj[0], min_obj[1], min_obj[2]),
        nvisii.vec3(centroid_obj[0], centroid_obj[1], centroid_obj[2]), 
    ]

    # change the ids to be like ndds / DOPE
    cuboid = [  cuboid[2],cuboid[0],cuboid[3],
                cuboid[5],cuboid[4],cuboid[1],
                cuboid[6],cuboid[7],cuboid[-1]]

    cuboid.append(nvisii.vec3(centroid_obj[0], centroid_obj[1], centroid_obj[2]))
        
    for i_p, p in enumerate(cuboid):
        child_transform = nvisii.transform.create(f"{name}_cuboid_{i_p}")
        child_transform.set_position(p)
        child_transform.set_scale(nvisii.vec3(0.3))
        child_transform.set_parent(obj.get_transform())
        if debug: 
            nvisii.entity.create(
                name = f"{name}_cuboid_{i_p}",
                mesh = nvisii.mesh.create_sphere(f"{name}_cuboid_{i_p}"),
                transform = child_transform, 
                material = nvisii.material.create(f"{name}_cuboid_{i_p}")
            )
    
    for i_v, v in enumerate(cuboid):
        cuboid[i_v]=[v[0], v[1], v[2]]

    return cuboid

def get_cuboid_image_space(obj_id, camera_name = 'camera'):
    """
    reproject the 3d points into the image space for a given object. 
    It assumes you already added the cuboid to the object 

    :obj_id: string for the name of the object of interest
    :camera_name: string representing the camera name in nvisii

    :return: cubdoid + centroid projected to the image, values [0..1]
    """

    cam_matrix = nvisii.entity.get(camera_name).get_transform().get_world_to_local_matrix()
    cam_proj_matrix = nvisii.entity.get(camera_name).get_camera().get_projection()

    points = []
    points_cam = []
    for i_t in range(9):
        trans = nvisii.transform.get(f"{obj_id}_cuboid_{i_t}")
        mat_trans = trans.get_local_to_world_matrix()
        pos_m = nvisii.vec4(
            mat_trans[3][0],
            mat_trans[3][1],
            mat_trans[3][2],
            1)
        
        p_cam = cam_matrix * pos_m 

        p_image = cam_proj_matrix * (cam_matrix * pos_m) 
        p_image = nvisii.vec2(p_image) / p_image.w
        p_image = p_image * nvisii.vec2(1,-1)
        p_image = (p_image + nvisii.vec2(1,1)) * 0.5

        points.append([p_image[0],p_image[1]])
        points_cam.append([p_cam[0],p_cam[1],p_cam[2]])
    return points, points_cam

# function to export meta data about the scene and about the objects 
# of interest. Everything gets saved into a json file.
def export_to_ndds_file(
    filename = "tmp.json", #this has to include path as well
    obj_names = [], # this is a list of ids to load and export
    height = 500, 
    width = 500,
    camera_name = 'camera',
    camera_struct = None,
    visibility_percentage = False, 
    ):
    """
    Method that exports the meta data like NDDS. This includes all the scene information in one 
    scene. 

    :filename: string for the json file you want to export, you have to include the extension
    :obj_names: [string] each entry is a nvisii entity that has the cuboids attached to, these
                are the objects that are going to be exported. 
    :height: int height of the image size 
    :width: int width of the image size 
    :camera_name: string for the camera name nvisii entity
    :camera_struct: dictionary of the camera look at information. Expecting the following 
                    entries: 'at','eye','up'. All three has to be floating arrays of three entries.
                    This is an optional export. 
    :visibility_percentage: bool if you want to export the visibility percentage of the object. 
                            Careful this can be costly on a scene with a lot of objects. 

    :return nothing: 
    """


    import simplejson as json

    # assume we only use the view camera
    cam_matrix = nvisii.entity.get(camera_name).get_transform().get_world_to_local_matrix()
    
    cam_matrix_export = []
    for row in cam_matrix:
        cam_matrix_export.append([row[0],row[1],row[2],row[3]])
    
    cam_world_location = nvisii.entity.get(camera_name).get_transform().get_position()
    cam_world_quaternion = nvisii.entity.get(camera_name).get_transform().get_rotation()
    # cam_world_quaternion = nvisii.quat_cast(cam_matrix)

    cam_intrinsics = nvisii.entity.get(camera_name).get_camera().get_intrinsic_matrix(width, height)

    if camera_struct is None:
        camera_struct = {
            'at': [0,0,0,],
            'eye': [0,0,0,],
            'up': [0,0,0,]
        }

    dict_out = {
                "camera_data" : {
                    "width" : width,
                    'height' : height,
                    'camera_look_at':
                    {
                        'at': [
                            camera_struct['at'][0],
                            camera_struct['at'][1],
                            camera_struct['at'][2],
                        ],
                        'eye': [
                            camera_struct['eye'][0],
                            camera_struct['eye'][1],
                            camera_struct['eye'][2],
                        ],
                        'up': [
                            camera_struct['up'][0],
                            camera_struct['up'][1],
                            camera_struct['up'][2],
                        ]
                    },
                    'camera_view_matrix':cam_matrix_export,
                    'location_world':
                    [
                        cam_world_location[0],
                        cam_world_location[1],
                        cam_world_location[2],
                    ],
                    'quaternion_world_xyzw':[
                        cam_world_quaternion[0],
                        cam_world_quaternion[1],
                        cam_world_quaternion[2],
                        cam_world_quaternion[3],
                    ],
                    'intrinsics':{
                        'fx':cam_intrinsics[0][0],
                        'fy':cam_intrinsics[1][1],
                        'cx':cam_intrinsics[2][0],
                        'cy':cam_intrinsics[2][1]
                    }
                }, 
                "objects" : []
            }

    # Segmentation id to export
    id_keys_map = nvisii.entity.get_name_to_id_map()

    for obj_name in obj_names: 

        projected_keypoints, _ = get_cuboid_image_space(obj_name, camera_name=camera_name)

        # put them in the image space. 
        for i_p, p in enumerate(projected_keypoints):
            projected_keypoints[i_p] = [p[0]*width, p[1]*height]

        # Get the location and rotation of the object in the camera frame 

        trans = nvisii.transform.get(obj_name)
        quaternion_xyzw = nvisii.inverse(cam_world_quaternion) * trans.get_rotation()

        object_world = nvisii.vec4(
            trans.get_position()[0],
            trans.get_position()[1],
            trans.get_position()[2],
            1
        ) 
        pos_camera_frame = cam_matrix * object_world

        #check if the object is visible
        visibility = -1
        bounding_box = [-1,-1,-1,-1]

        segmentation_mask = nvisii.render_data(
            width=int(width), 
            height=int(height), 
            start_frame=0,
            frame_count=1,
            bounce=int(0),
            options="entity_id",
        )
        segmentation_mask = np.array(segmentation_mask).reshape(width,height,4)[:,:,0]
        
        if visibility_percentage == True and int(id_keys_map [obj_name]) in np.unique(segmentation_mask.astype(int)): 
            transforms_to_keep = {}
            
            for name in id_keys_map.keys():
                if 'camera' in name.lower() or obj_name in name:
                    continue
                trans_to_keep = nvisii.entity.get(name).get_transform()
                transforms_to_keep[name]=trans_to_keep
                nvisii.entity.get(name).clear_transform()

            # Percentage visibility through full segmentation mask. 
            segmentation_unique_mask = nvisii.render_data(
                width=int(width), 
                height=int(height), 
                start_frame=0,
                frame_count=1,
                bounce=int(0),
                options="entity_id",
            )

            segmentation_unique_mask = np.array(segmentation_unique_mask).reshape(width,height,4)[:,:,0]

            values_segmentation = np.where(segmentation_mask == int(id_keys_map[obj_name]))[0]
            values_segmentation_full = np.where(segmentation_unique_mask == int(id_keys_map[obj_name]))[0]
            visibility = len(values_segmentation)/float(len(values_segmentation_full))
            
            # set back the objects from remove
            for entity_name in transforms_to_keep.keys():
                nvisii.entity.get(entity_name).set_transform(transforms_to_keep[entity_name])
        else:

            if int(id_keys_map[obj_name]) in np.unique(segmentation_mask.astype(int)): 
                #
                visibility = 1
                y,x = np.where(segmentation_mask == int(id_keys_map[obj_name]))
                bounding_box = [int(min(x)),int(max(x)),height-int(max(y)),height-int(min(y))]
            else:
                visibility = 0

        # Final export
        dict_out['objects'].append({
            'class':obj_name.split('_')[0],
            'name':obj_name,
            'provenance':'nvisii',
            # TODO check the location
            'location_camera': [
                pos_camera_frame[0],
                pos_camera_frame[1],
                pos_camera_frame[2]
            ],
            'location_world': [
                trans.get_position()[0],
                trans.get_position()[1],
                trans.get_position()[2]
            ],
            'quaternion_xyzw_camera':[
                quaternion_xyzw[0],
                quaternion_xyzw[1],
                quaternion_xyzw[2],
                quaternion_xyzw[3],
            ],
            'quaternion_xyzw_world':[
                trans.get_rotation()[0],
                trans.get_rotation()[1],
                trans.get_rotation()[2],
                trans.get_rotation()[3]
            ],
            'projected_cuboid_image':projected_keypoints,
            'segmentation_id':id_keys_map[obj_name],
            'visibility_image':visibility,
            'bounding_box_minx_maxx_miny_maxy_image':bounding_box
        })
        
    with open(filename, 'w+') as fp:
        json.dump(dict_out, fp, indent=4, sort_keys=True)

# # # # # # # # # # # # # # # # # # # # # # # # #
nvisii.initialize(headless=True, verbose=True)

nvisii.enable_denoiser()

camera = nvisii.entity.create(
    name = "camera",
    transform = nvisii.transform.create("camera"),
    camera = nvisii.camera.create(
        name = "camera",  
        aspect = float(opt.width)/float(opt.height)
    )
)

# # # # # # # # # # # # # # # # # # # # # # # # #


# lets store the camera look at information so we can export it
camera_struct_look_at = {
    'at':[0,0.1,0.1],
    'up':[0,0,1],
    'eye':[1,0.7,0.2]
}


# # # # # # # # # # # # # # # # # # # # # # # # #

camera.get_transform().look_at(
    at = camera_struct_look_at['at'],
    up = camera_struct_look_at['up'],
    eye = camera_struct_look_at['eye']
)


nvisii.set_camera_entity(camera)

nvisii.set_dome_light_sky(sun_position = (10, 10, 1), saturation = 2)
nvisii.set_dome_light_exposure(1)

mesh = nvisii.mesh.create_from_file("obj", "./content/dragon/dragon.obj")

obj_entity = nvisii.entity.create(
    name="obj_entity",
    mesh = mesh,
    transform = nvisii.transform.create("obj_entity"),
    material = nvisii.material.create("obj_entity")
)

# # # # # # # # # # # # # # # # # # # # # # # # #
# Lets add the cuboid to the object we want to export

add_cuboid("obj_entity")

# lets keep track of the entities we want to export
entities_to_export = ["obj_entity"]

# lets set the obj_entity up
obj_entity.get_transform().set_rotation( 
    (0.7071, 0, 0, 0.7071)
)
obj_entity.get_material().set_base_color(
    (0.9,0.12,0.08)
)  
obj_entity.get_material().set_roughness(0.7)   
obj_entity.get_material().set_specular(1)   
obj_entity.get_material().set_sheen(1)


# # # # # # # # # # # # # # # # # # # # # # # # #

nvisii.render_to_file(
    width=opt.width, 
    height=opt.height, 
    samples_per_pixel=opt.spp,
    file_path=opt.out 
)

export_to_ndds_file(
    filename = opt.out.replace('png','json'),
    obj_names = entities_to_export,
    width=opt.width, 
    height=opt.height, 
    camera_struct = camera_struct_look_at
    )

# let's clean up GPU resources
nvisii.deinitialize()