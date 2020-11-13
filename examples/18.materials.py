# 18.materials.py
#
# This example demonstrates the effects of different material parameters.
# It also shows how to implement basic camera controls

import visii
from colorsys import *
import time

opt = lambda: None
opt.spp = 1024 
opt.width = 1920
opt.height = 1080 
opt.out = '18_materials.png'

visii.initialize(headless = False, verbose = True)

# Use a neural network to denoise ray traced
visii.enable_denoiser()

# This is new, have the dome light use a texture
dome = visii.texture.create_from_file("dome", "content/teatro_massimo_2k.hdr")
visii.set_dome_light_texture(dome, enable_cdf=True)
visii.resize_window(1920, 1080)

# # Make a wall
# wall = visii.entity.create(
#     name = "wall",
#     mesh = visii.mesh.create_plane("mesh_wall"),
#     transform = visii.transform.create("transform_wall"),
#     material = visii.material.create("material_wall")
# )
# wall.get_transform().set_scale((50,50,1))
# wall.get_transform().set_rotation(visii.angleAxis(visii.pi() * .5, (1,0,0)))
# wall.get_transform().set_position((0,-.5,0))

# Make a sphere mesh that we'll make several instances of
# sphere_mesh = visii.mesh.create_teapotahedron("sphere")
sphere_mesh = visii.mesh.create_sphere("sphere", radius=1)
box_mesh = visii.mesh.create_rounded_box("box")

for x in range(20):
    for y in range(20):
        name = str(x) + "_" + str(y)
        m = box_mesh
        if y % 4 < 2:
            m = sphere_mesh

        visii.entity.create(
            name = name,
            mesh = m,
            transform = visii.transform.create(
                name = name, 
                # position = (x * 1.3 + .33 * pow(-1, y), 0, y * .35),
                position = (x * .25, 0, y * .25 * .7 + .02 * pow(-1, y)),
                # scale = (.15, .15, .15),
                scale = ((y % 2) * .01 + .09, 
                         (y % 2) * .01 + .09, 
                         (y % 2) * .01 + .09),
                rotation = visii.angleAxis(-.78, (1,0,0))
            ),
            material = visii.material.create(
                name = name
            )
        )
        mat = visii.material.get(name)
        
        # The diffuse, metal, or glass surface color
        # mat.set_base_color(...)
        mat.set_base_color(hsv_to_rgb(y / 60.0, y % 2, 1.0))

        # Specifies the microfacet roughness of the surface for diffuse 
        # or specular reflection
        if y == 0 or y == 1: 
            mat.set_roughness(x / 20.0)

        # Blends between a non-metallic and a metallic material model.
        if y == 2 or y == 3: 
            mat.set_roughness(0.0)
            mat.set_metallic(x / 20.0)

        # Blends between a fully opaque surface to a fully glass like 
        # transmission one
        if y == 4 or y == 5: 
            mat.set_transmission(x / 20.0)
            mat.set_roughness(0.0)
        
        # Controls the roughness used for transmitted light
        if y == 6 or y == 7: 
            mat.set_transmission_roughness(x / 20.0)
            mat.set_roughness(0.0)
            mat.set_transmission(1.0)

        # Gives a velvet like shine at glancing angles
        # This effect is subtle, but useful for fabric materials
        if y == 8 or y == 9: 
            mat.set_roughness(1.0)
            mat.set_sheen(x / 20.0)

        # Extra white specular layer on top of other material layers.
        # This effect is pretty subtle, but useful for materials like
        # car paint.
        if y == 10 or y == 11: 
            mat.set_roughness(0.2)
            mat.set_clearcoat(x / 20.0)

        # Elongates the highlights of glossy materials
        if y == 12 or y == 13: 
            mat.set_roughness(.5)
            mat.set_metallic(1.0)
            mat.set_anisotropic(x / 20.0)

        # Interpolates between a surface and subsurface color.
        if y == 14 or y == 15: 
            # The subsurface scattering base color
            mat.set_roughness(1)
            mat.set_subsurface_color((1,0,0))
            mat.set_subsurface(x / 20.0)

        # Controls how much incluence that specular reflection occurs at head-on
        # reflections
        if y == 16 or y == 17: 
            mat.set_roughness(0.0)
            mat.set_specular(x / 20.0)

        # Controls the probability of a ray passing through the material
        if y == 18 or y == 19: 
            mat.set_alpha(x / 20.0)
        
# Create a camera
center = visii.get_scene_aabb_center()
camera = visii.entity.create(
    name = "camera",
    transform = visii.transform.create(name = "camera_transform"),
    camera = visii.camera.create(name = "camera_camera", aspect=opt.width / opt.height)
)
camera.get_transform().look_at(at = (center.x, 0, center.z), up = (0, 0, 1), eye = (center.x, -5, center.z))
visii.set_camera_entity(camera)

# Render out the final image
print("rendering to", opt.out)
visii.render_to_file(
    width = opt.width, 
    height = opt.height, 
    samples_per_pixel = opt.spp,   
    file_path = opt.out
)

visii.deinitialize()
