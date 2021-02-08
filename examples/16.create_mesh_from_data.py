import nvisii

import numpy as np 
import open3d as o3d


opt = lambda: None
opt.spp = 100 
opt.width = 500
opt.height = 500 
opt.noise = False
opt.out = '16_create_mesh_from_data.png'
opt.path_obj = 'content/dragon/dragon.obj'

# # # # # # # # # # # # # # # # # # # # # # # # #
nvisii.initialize(headless = True, verbose = True)

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
    at = (0,0,0),
    up = (0,0,1),
    eye = (0.2,0.2,0.2),
)
nvisii.set_camera_entity(camera)

nvisii.set_dome_light_intensity(1)

# # # # # # # # # # # # # # # # # # # # # # # # #

# Although NVISII has official support for stl files through mesh.create_from_file,
# let's open the STL from another library, and use the create_from_data interface. 

# let load the object using open3d
mesh = o3d.io.read_triangle_mesh("content/dragon.stl")

if not mesh.has_vertex_normals():
    mesh = mesh.compute_vertex_normals()

normals = np.array(mesh.vertex_normals).flatten().tolist()
vertices = np.array(mesh.vertices).flatten().tolist()

mesh = nvisii.mesh.create_from_data(
    'stl_mesh',
    positions=vertices,
    normals=normals
)

# # # # # # # # # # # # # # # # # # # # # # # # #

obj_entity = nvisii.entity.create(
    name="obj_entity",
    mesh = mesh,
    transform = nvisii.transform.create("obj_entity",
        scale=(0.3, 0.3, 0.3)
    ),
    material = nvisii.material.create("obj_entity")
)

obj_entity.get_material().set_base_color((0.9,0.12,0.08))  
obj_entity.get_material().set_roughness(0.7)   
obj_entity.get_material().set_specular(1)   
obj_entity.get_material().set_sheen(1)


# # # # # # # # # # # # # # # # # # # # # # # # #

nvisii.render_to_file(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    file_path=f"{opt.out}"
)

# let's clean up the GPU
nvisii.deinitialize()