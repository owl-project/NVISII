import visii
import random
import time
import numpy as np

# this will create a window where you should 
# only see gaussian noise pattern
visii.initialize()

camera = visii.entity.create(name = "camera")
camera.set_transform(visii.transform.create(name = "camera_transform"))
camera.set_camera(
    visii.camera.create_perspective_from_fov(
        name = "camera_camera", 
        field_of_view = 0.785398, # note, this is in radians
        aspect = 1.0
    )
)
visii.set_camera_entity(camera)
camera.get_transform().look_at(
    at = (0, 0, 0.9), # at position
    up = (0, 0, 1),   # up vector
    eye = (0, 5, 1)   # eye position
)

# sphere = visii.entity.create(
#     name="sphere",
#     mesh = visii.mesh.create_sphere("sphere"),
#     transform = visii.transform.create("sphere"),
#     material = visii.material.create("sphere")
# )
# sphere.get_transform().set_scale((0.4, 0.4, 0.4))
# sphere.get_material().set_base_color((0.1,0.9,0.08))  
# sphere.get_material().set_roughness(0.7)   
# sphere.get_material().set_specular(1)
# sphere.get_transform().set_position((0,0,0.41))

def create_sharp_mesh(name, vertices, indices):
    assert((len(indices) % 3) == 0 and len(indices) > 0)
    positions = []
    normals = []
    colors = []
    texcoords = []
    indices_out = []
    for t in range(len(indices)//3):
        def LoadVertex(i):
            assert(3*(i+1) <= len(vertices))
            return [vertices[3*i], vertices[3*i+1], vertices[3*i+2]]
        v0 = np.array(LoadVertex(indices[3*t]))
        v1 = np.array(LoadVertex(indices[3*t+1]))
        v2 = np.array(LoadVertex(indices[3*t+2]))
        nrml = np.cross(v1-v0, v2-v0)
        l = np.linalg.norm(nrml)
        if l < 1e-6:
            print('Degenerate triangle')
            continue
        nrml *= 1.0 / l
        positions += v0.tolist() + v1.tolist() + v2.tolist()
        normals += nrml.tolist() * 3
        colors += [1.0, 1.0, 1.0, 1.0] * 3
        texcoords += [0.0, 0.0] * 3
        indices_out += [3*t, 3*t+1, 3*t+2]
    return visii.mesh.create_from_data(name, positions, 3, normals, 3, colors, 4, texcoords, 2, indices_out)

class TriVis:
    def __init__(self):
        self.namectr = 0
        self.nullxfrm = visii.transform.create("TriVis_nullxfrm")
        self.mesh = visii.mesh.create_box("TriVis_mesh_0")
        self.matl = visii.material.create("TriVis_matl", (0.4, 0.2, 0.8))
        self.ent = visii.entity.create("TriVis", self.nullxfrm, self.matl, self.mesh)
    
    def update(self, verts):
        assert(len(verts) == 9)
        visii.mesh.remove("TriVis_mesh_" + str(self.namectr))
        self.namectr += 1
        self.mesh = create_sharp_mesh("TriVis_mesh_" + str(self.namectr), verts, [0,1,2])
        self.ent.set_mesh(self.mesh)

print("[Internal] transform limit is: " + str(visii.transform.get_count()))

tvis = TriVis()
tvis.update(np.random.rand(9).tolist())

for i in range(10000):
    print("Removing and recreating triangle " + str(i))
    tvis.update(np.random.rand(9).tolist())
    # visii.mesh.remove("sphere")
    # sphere.set_mesh(visii.mesh.create_sphere("sphere", random.random() * 2.0))
    time.sleep(0.01)

print("Finished successfully")
visii.deinitialize()