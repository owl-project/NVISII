import visii
import math
import PySide2
import colorsys
from PySide2.QtCore import *
from PySide2.QtWidgets import *

visii.initialize()
visii.resize_window(1000,1000)
visii.set_max_bounce_depth(diffuse_depth=2, specular_depth=20)
visii.enable_denoiser()

# Set the sky
dome = visii.texture.create_from_file("dome", "content/teatro_massimo_2k.hdr")
visii.set_dome_light_intensity(1.15)
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
    cam_pos = camera.get_transform().get_position()

    camera.get_transform().look_at(
        at = (0, 0, 0.9), # at position
        up = (0, 0, 1),   # up vector
        eye = (5 * math.cos(value * 2 * visii.pi()), 5 * math.sin(value * 2 * visii.pi()), cam_pos[2])   # eye position
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
    # print(value, 2 * math.cos(value * 2 * visii.pi()))

slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(rotateCameraElevation)
slider.setValue(40)
layout.addWidget(QLabel('Camera Elevation')) 
layout.addWidget(slider) 

color = [1,0.1,0.5]

def setColorHue(value):
    value = value / 100.0
    color[0] = value
    rgb = colorsys.hsv_to_rgb(color[0],color[1],color[2])
    teapotahedron.get_material().set_base_color(visii.vec3(rgb[0],rgb[1],rgb[2]))

def setColorSaturation(value):
    value = value / 100.0
    color[1] = value
    rgb = colorsys.hsv_to_rgb(color[0],color[1],color[2])
    teapotahedron.get_material().set_base_color(visii.vec3(rgb[0],rgb[1],rgb[2]))

def setColorValue(value):
    value = value / 100.0
    color[2] = value
    rgb = colorsys.hsv_to_rgb(color[0],color[1],color[2])
    teapotahedron.get_material().set_base_color(visii.vec3(rgb[0],rgb[1],rgb[2]))

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
slider.setValue(int(color[1]*100))
layout.addWidget(QLabel('Color Value')) 
layout.addWidget(slider) 


def setAlpha(value):
    value = value / 100.0
    teapotahedron.get_material().set_alpha(value)
setAlpha(50)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setAlpha)
slider.setValue(50)
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

def setTransmissionRoughness(value):
    value = value / 100.0
    teapotahedron.get_material().set_transmission_roughness(value)
setTransmissionRoughness(0)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setTransmissionRoughness)
slider.setValue(0)
layout.addWidget(QLabel('Transmission Roughness')) 
layout.addWidget(slider) 


def setIor(value):
    value = (value / 100.0)*3
    teapotahedron.get_material().set_ior(value)
setIor(1.5)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setIor)
slider.setValue(50)
layout.addWidget(QLabel('Transmission Ior')) 
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

def setMetal(value):
    value = value / 100.0
    teapotahedron.get_material().set_metallic(value)
setMetal(0)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setMetal)
slider.setValue(0)
layout.addWidget(QLabel('Metallic')) 
layout.addWidget(slider) 

def setSpecular(value):
    value = value / 100.0
    teapotahedron.get_material().set_specular(value)
setSpecular(0)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setSpecular)
slider.setValue(0)
layout.addWidget(QLabel('Specular')) 
layout.addWidget(slider) 

def setSpecularTint(value):
    value = value / 100.0
    teapotahedron.get_material().set_specular_tint(value)
setSpecularTint(0)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setSpecularTint)
slider.setValue(0)
layout.addWidget(QLabel('Specular Tint')) 
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

def setSheenTint(value):
    value = value / 100.0
    teapotahedron.get_material().set_sheen_tint(value)
setSheenTint(0)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setSheenTint)
slider.setValue(0)
layout.addWidget(QLabel('Sheen Tint')) 
layout.addWidget(slider) 

def setAnisotropy(value):
    value = value / 100.0
    teapotahedron.get_material().set_anisotropic(value)
setAnisotropy(0)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setAnisotropy)
slider.setValue(0)
layout.addWidget(QLabel('Anisotropy')) 
layout.addWidget(slider) 

def setClearCoat(value):
    value = value / 100.0
    teapotahedron.get_material().set_clearcoat(value)
setClearCoat(0)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setClearCoat)
slider.setValue(0)
layout.addWidget(QLabel('Clearcoat')) 
layout.addWidget(slider)

def setClearCoatRough(value):
    value = value / 100.0
    teapotahedron.get_material().set_clearcoat_roughness(value)
setClearCoatRough(0)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setClearCoatRough)
slider.setValue(0)
layout.addWidget(QLabel('Clearcoat Roughness')) 
layout.addWidget(slider)

color_subsurface = [0.5,1,1 ]

def setColorHueSub(value):
    value = value / 100.0
    color_subsurface[0] = value
    rgb = colorsys.hsv_to_rgb(color_subsurface[0],color_subsurface[1],color_subsurface[2])
    teapotahedron.get_material().set_subsurface_color(visii.vec3(rgb[0],rgb[1],rgb[2]))

def setColorSaturationSub(value):
    value = value / 100.0
    color_subsurface[1] = value
    rgb = colorsys.hsv_to_rgb(color_subsurface[0],color_subsurface[1],color_subsurface[2])
    teapotahedron.get_material().set_subsurface_color(visii.vec3(rgb[0],rgb[1],rgb[2]))

def setColorValueSub(value):
    value = value / 100.0
    color_subsurface[2] = value
    rgb = colorsys.hsv_to_rgb(color_subsurface[0],color_subsurface[1],color_subsurface[2])
    teapotahedron.get_material().set_subsurface_color(visii.vec3(rgb[0],rgb[1],rgb[2]))

setColorHueSub(int(color_subsurface[0]*100))
setColorSaturationSub(int(color_subsurface[1]*100))
setColorValueSub(int(color_subsurface[2]*100))

slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setColorHueSub)
slider.setValue(int(color_subsurface[0]*100))
layout.addWidget(QLabel('Sub Color Hue')) 
layout.addWidget(slider) 

slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setColorSaturationSub)
slider.setValue(int(color_subsurface[1]*100))
layout.addWidget(QLabel('Sub Color Saturation')) 
layout.addWidget(slider) 

slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setColorValueSub)
slider.setValue(int(color_subsurface[1]*100))
layout.addWidget(QLabel('Sub Color Value')) 
layout.addWidget(slider) 

def setSubSurface(value):
    value = value / 100.0
    teapotahedron.get_material().set_subsurface(value)
setSubSurface(0)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setSubSurface)
slider.setValue(0)
layout.addWidget(QLabel('Subsurface mix')) 
layout.addWidget(slider) 

window.setLayout(layout)
window.show() 
app.exec_() 

visii.deinitialize()