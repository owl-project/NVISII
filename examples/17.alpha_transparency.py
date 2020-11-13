import visii
import math
import PySide2
from PySide2.QtCore import *
from PySide2.QtWidgets import *

visii.initialize()
visii.set_max_bounce_depth(diffuse_depth=2, specular_depth=20)
visii.enable_denoiser()

# Set the sky
dome = visii.texture.create_from_file("dome", "content/teatro_massimo_2k.hdr")
visii.set_dome_light_intensity(.5)
visii.set_dome_light_texture(dome)

# Set camera
camera = visii.entity.create(
    name = "camera",
    transform = visii.transform.create(name = "camera_transform"),
    camera = visii.camera.create(
        name = "camera_camera", 
        aspect = 1.0
    )
)
camera.get_transform().look_at(
    at = (0, 0, 0.9), # at position
    up = (0, 0, 1),   # up vector
    eye = (0, 5, 2)   # eye position
)
visii.set_camera_entity(camera)

# Floor
floor = visii.entity.create(
    name = "floor",
    mesh = visii.mesh.create_plane("mesh_floor"),
    transform = visii.transform.create("transform_floor"),
    material = visii.material.create("material_floor")
)
floor.get_material().set_base_color((0.19,0.16,0.19)) 
floor.get_material().set_metallic(0) 
floor.get_material().set_roughness(1)
floor.get_transform().set_scale((5,5,1))

# Teapot
teapotahedron = visii.entity.create(
    name="teapotahedron",
    mesh = visii.mesh.create_teapotahedron("teapotahedron", segments = 32),
    transform = visii.transform.create("teapotahedron"),
    material = visii.material.create("teapotahedron")
)
teapotahedron.get_transform().set_rotation(visii.angleAxis(visii.pi() / 4.0, (0,0,1)))
teapotahedron.get_transform().set_position((0,0,0.41))
teapotahedron.get_transform().set_scale((0.4, 0.4, 0.4))
teapotahedron.get_material().set_base_color((1.0,1.0,1.0))  
teapotahedron.get_material().set_roughness(0.0)  

# Objects can be made to be "alpha transparent", which simulates little holes in the
# mesh that let light through. The smaller the alpha, the more little holes.
teapotahedron.get_material().set_alpha(0.5)   

# Make a QT window to demonstrate the difference between alpha transparency and transmission
app = QApplication([]) # Start an application.
window = QWidget() # Create a window.
layout = QVBoxLayout() # Create a layout.

def rotateCamera(value):
    value = value / 100.0
    camera.get_transform().look_at(
        at = (0, 0, 0.9), # at position
        up = (0, 0, 1),   # up vector
        eye = (5 * math.cos(value * 2 * visii.pi()), 5 * math.sin(value * 2 * visii.pi()), 2)   # eye position
    )
dial = QDial() 
dial.setWrapping(True)
dial.valueChanged[int].connect(rotateCamera)
layout.addWidget(QLabel('Camera rotation')) 
layout.addWidget(dial) 

def setTransmission(value):
    value = value / 100.0
    teapotahedron.get_material().set_transmission(value)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setTransmission)
slider.setValue(0)
layout.addWidget(QLabel('Transmission')) 
layout.addWidget(slider) 

def setAlpha(value):
    value = value / 100.0
    teapotahedron.get_material().set_alpha(value)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setAlpha)
slider.setValue(50)
layout.addWidget(QLabel('Alpha')) 
layout.addWidget(slider) 

window.setLayout(layout)
window.show() 
app.exec_() 

visii.deinitialize()