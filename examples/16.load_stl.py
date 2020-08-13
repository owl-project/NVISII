import visii
import argparse

import numpy as np 
import open3d as o3d

parser = argparse.ArgumentParser()

parser.add_argument('--spp', 
                    default=100,
                    type=int,
                    help = "number of sample per pixel, higher the more costly")
parser.add_argument('--width', 
                    default=500,
                    type=int,
                    help = 'image output width')
parser.add_argument('--height', 
                    default=500,
                    type=int,
                    help = 'image output height')
parser.add_argument('--noise',
                    action='store_true',
                    default=False,
                    help = "if added the output of the ray tracing is not sent to optix's denoiser")
parser.add_argument('--path_obj',
                    default='content/dragon/dragon.obj',
                    help = "path to the obj mesh you want to load")
parser.add_argument('--out',
                    default='tmp.png',
                    help = "output filename")

opt = parser.parse_args()

# # # # # # # # # # # # # # # # # # # # # # # # #
visii.initialize_headless()

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
    visii.vec3(0,0,0), # look at (world coordinate)
    visii.vec3(0,0,1), # up vector
    visii.vec3(0.2,0.2,0.2), # camera_origin    
)
visii.set_camera_entity(camera)

visii.set_dome_light_intensity(1)

# # # # # # # # # # # # # # # # # # # # # # # # #

# let load the object using open3d
mesh = o3d.io.read_triangle_mesh("content/dragon.stl")

if not mesh.has_vertex_normals():
    mesh = mesh.compute_vertex_normals()

normals = np.array(mesh.vertex_normals).flatten().tolist()
vertices = np.array(mesh.vertices).flatten().tolist()

mesh = visii.mesh.create_from_data(
    'stl_mesh',
    positions=vertices,
    normals=normals
)

# # # # # # # # # # # # # # # # # # # # # # # # #

obj_entity = visii.entity.create(
    name="obj_entity",
    mesh = mesh,
    transform = visii.transform.create("obj_entity",
        scale=visii.vec3(0.3)
    ),
    material = visii.material.create("obj_entity")
)

obj_entity.get_material().set_base_color(
    visii.vec3(0.9,0.12,0.08)
)  
obj_entity.get_material().set_roughness(0.7)   
obj_entity.get_material().set_specular(1)   
obj_entity.get_material().set_sheen(1)


# # # # # # # # # # # # # # # # # # # # # # # # #

visii.render_to_png(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    image_path=f"{opt.out}"
)

# let's clean up the GPU
visii.deinitialize()