import nvisii
import math
import PySide2
import colorsys
from PySide2.QtCore import *
from PySide2.QtWidgets import *

nvisii.initialize()
nvisii.resize_window(1000,1000)
nvisii.enable_denoiser()

# Set the sky
dome = nvisii.texture.create_from_file("dome", "content/teatro_massimo_2k.hdr")
nvisii.set_dome_light_texture(dome)

# Set camera
camera = nvisii.entity.create(
    name = "camera",
    transform = nvisii.transform.create(name = "camera_transform"),
    camera = nvisii.camera.create(
        name = "camera_camera", 
        aspect = 1.0
    )
)
camera.get_transform().look_at(
    at = (0, 0, 0.9), # at position
    up = (0, 0, 1),   # up vector
    eye = (0, 5, 2)   # eye position
)
nvisii.set_camera_entity(camera)

# Floor
floor = nvisii.entity.create(
    name = "floor",
    mesh = nvisii.mesh.create_plane("mesh_floor"),
    transform = nvisii.transform.create("transform_floor"),
    material = nvisii.material.create("material_floor")
)
floor.get_material().set_base_color((0.19,0.16,0.19)) 
floor.get_material().set_metallic(0) 
floor.get_material().set_roughness(1)
floor.get_transform().set_scale((5,5,1))

# Teapot
teapotahedron = nvisii.entity.create(
    name="teapotahedron",
    mesh = nvisii.mesh.create_teapotahedron("teapotahedron", segments = 32),
    transform = nvisii.transform.create("teapotahedron"),
    material = nvisii.material.create("teapotahedron")
)
teapotahedron.get_transform().set_rotation(nvisii.angleAxis(nvisii.pi() / 4.0, (0,0,1)))
teapotahedron.get_transform().set_position((0,0,0.41))
teapotahedron.get_transform().set_scale((0.4, 0.4, 0.4))
teapotahedron.get_material().set_base_color((1.0,1.0,1.0))  
teapotahedron.get_material().set_roughness(0.0)  

# Objects can be made to be "alpha transparent", which simulates little holes in the
# mesh that let light through. The smaller the alpha, the more little holes.
teapotahedron.get_material().set_alpha(1.0)   

# Make a QT window to demonstrate the difference between alpha transparency and transmission
app = QApplication([]) # Start an application.
window = QWidget() # Create a window.
layout = QVBoxLayout() # Create a layout.

def rotateCamera(value):
    value = value / 100.0
    cam_pos = camera.get_transform().get_position()

    camera.get_transform().look_at(
        at = (0, 0, 0.9), # at position
        up = (0, 0, 1),   # up vector
        eye = (5 * math.cos(value * 2 * nvisii.pi()), 5 * math.sin(value * 2 * nvisii.pi()), cam_pos[2])   # eye position
    )
rotateCamera(0)
dial = QDial() 
dial.setWrapping(True)
dial.valueChanged[int].connect(rotateCamera)
layout.addWidget(QLabel('Camera rotation')) 
layout.addWidget(dial) 

def rotateCameraElevation(value):
    # print(value)
    value = value / 100
    cam_pos = camera.get_transform().get_position()
    camera.get_transform().look_at(
        at = (0, 0, 0.9), # at position
        up = (0, 0, 1),   # up vector
        eye = (cam_pos[0], cam_pos[1], 0.1 + 6*value)   # eye position
    )
    # print(value, 2 * math.cos(value * 2 * nvisii.pi()))

slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(rotateCameraElevation)
slider.setValue(40)
layout.addWidget(QLabel('Camera Elevation')) 
layout.addWidget(slider) 

color = [0,1.0,1.0]

def setColorHue(value):
    value = value / 100.0
    color[0] = value
    rgb = colorsys.hsv_to_rgb(color[0],color[1],color[2])
    teapotahedron.get_material().set_base_color(nvisii.vec3(rgb[0],rgb[1],rgb[2]))

def setColorSaturation(value):
    value = value / 100.0
    color[1] = value
    rgb = colorsys.hsv_to_rgb(color[0],color[1],color[2])
    teapotahedron.get_material().set_base_color(nvisii.vec3(rgb[0],rgb[1],rgb[2]))

def setColorValue(value):
    value = value / 100.0
    color[2] = value
    rgb = colorsys.hsv_to_rgb(color[0],color[1],color[2])
    teapotahedron.get_material().set_base_color(nvisii.vec3(rgb[0],rgb[1],rgb[2]))

setColorHue(int(color[0]*100))
setColorSaturation(int(color[1]*100))
setColorValue(int(color[2]*100))

slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setColorHue)
slider.setValue(int(color[0]*100))
layout.addWidget(QLabel('Color Hue')) 
layout.addWidget(slider) 

slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setColorSaturation)
slider.setValue(int(color[1]*100))
layout.addWidget(QLabel('Color Saturation')) 
layout.addWidget(slider) 

slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setColorValue)
slider.setValue(int(color[2]*100))
layout.addWidget(QLabel('Color Value')) 
layout.addWidget(slider) 


def setAlpha(value):
    value = value / 100.0
    teapotahedron.get_material().set_alpha(value)
setAlpha(100)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setAlpha)
slider.setValue(100)
layout.addWidget(QLabel('Alpha')) 
layout.addWidget(slider) 

def setTransmission(value):
    value = value / 100.0
    teapotahedron.get_material().set_transmission(value)
setTransmission(0)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setTransmission)
slider.setValue(0)
layout.addWidget(QLabel('Transmission')) 
layout.addWidget(slider) 

def setRoughness(value):
    value = value / 100.0
    teapotahedron.get_material().set_roughness(value)
setRoughness(0)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setRoughness)
slider.setValue(0)
layout.addWidget(QLabel('Roughness')) 
layout.addWidget(slider) 

def setTransmissionRoughness(value):
    value = value / 100.0
    teapotahedron.get_material().set_transmission_roughness(value)
setTransmissionRoughness(0)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setTransmissionRoughness)
slider.setValue(0)
layout.addWidget(QLabel('Transmission Roughness')) 
layout.addWidget(slider) 

def setMetal(value):
    value = value / 100.0
    teapotahedron.get_material().set_metallic(value)
setMetal(0)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setMetal)
slider.setValue(0)
layout.addWidget(QLabel('Metallic')) 
layout.addWidget(slider) 

def setAnisotropic(value):
    value = value / 100.0
    teapotahedron.get_material().set_anisotropic(value)
setAnisotropic(0)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setAnisotropic)
slider.setValue(0)
layout.addWidget(QLabel('Anisotropy')) 
layout.addWidget(slider) 

def setSpecular(value):
    value = value / 100.0
    teapotahedron.get_material().set_specular(value)
setSpecular(50)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setSpecular)
slider.setValue(50)
layout.addWidget(QLabel('Specular')) 
layout.addWidget(slider) 

def setSheen(value):
    value = value / 100.0
    teapotahedron.get_material().set_sheen(value)
setSheen(0)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setSheen)
slider.setValue(0)
layout.addWidget(QLabel('Sheen')) 
layout.addWidget(slider) 

window.setLayout(layout)
window.show() 
app.exec_() 

nvisii.deinitialize()