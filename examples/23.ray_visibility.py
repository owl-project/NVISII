import nvisii
import math
import PySide2
import colorsys
from PySide2.QtCore import *
from PySide2.QtWidgets import *

nvisii.initialize()
nvisii.resize_window(1000,1000)
nvisii.enable_denoiser()
# nvisii.configure_denoiser(False, False, True)
nvisii.set_max_bounce_depth(diffuse_depth=2, glossy_depth = 8, transparency_depth = 8, transmission_depth = 12, volume_depth = 2)

# Set the sky
nvisii.disable_dome_light_sampling()
nvisii.set_dome_light_color((0,0,0))

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
    at = (0, 0, 0.5), # at position
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

# Mirror 1
mirror1 = nvisii.entity.create(
    name = "mirror1",
    mesh = nvisii.mesh.create_box("mesh_mirror1"),
    transform = nvisii.transform.create("transform_mirror1"),
    material = nvisii.material.create("material_mirror1")
)
mirror1.get_transform().look_at(eye = (-1.5, -1.5, .5), at = (0,0,.7), up = (0,0,1))
mirror1.get_material().set_base_color((1.,1.,1.)) 
mirror1.get_material().set_metallic(1) 
mirror1.get_material().set_roughness(0)
mirror1.get_transform().set_scale((.7,.7,.1))

# Glass 1
glass1 = nvisii.entity.create(
    name = "glass1",
    mesh = nvisii.mesh.create_box("mesh_glass1"),
    transform = nvisii.transform.create("transform_glass1"),
    material = nvisii.material.create("material_glass1")
)
glass1.get_transform().look_at(eye = (1.5, 1.5, .5), at = (0,0,.7), up = (0,0,1))
glass1.get_material().set_base_color((1.,1.,1.)) 
glass1.get_material().set_transmission(1) 
glass1.get_material().set_roughness(0)
glass1.get_transform().set_scale((.7,.7,.1))

# Mirror 2
mirror2 = nvisii.entity.create(
    name = "mirror2",
    mesh = nvisii.mesh.create_box("mesh_mirror2"),
    transform = nvisii.transform.create("transform_mirror2"),
    material = nvisii.material.create("material_mirror2")
)
mirror2.get_transform().look_at(eye = (1.5, -1.5, .5), at = (0,0,.7), up = (0,0,1))
mirror2.get_material().set_base_color((1.,1.,1.)) 
mirror2.get_material().set_metallic(1) 
mirror2.get_material().set_roughness(0)
mirror2.get_transform().set_scale((.7,.7,.1))

# Glass 2
glass2 = nvisii.entity.create(
    name = "glass2",
    mesh = nvisii.mesh.create_box("mesh_glass2"),
    transform = nvisii.transform.create("transform_glass2"),
    material = nvisii.material.create("material_glass2")
)
glass2.get_transform().look_at(eye = (-1.5, 1.5, .5), at = (0,0,.7), up = (0,0,1))
glass2.get_material().set_base_color((1.,1.,1.)) 
glass2.get_material().set_transmission(1) 
glass2.get_material().set_roughness(0)
glass2.get_transform().set_scale((.7,.7,.1))

# Fog
fog = nvisii.entity.create(
    name = "fog",
    volume = nvisii.volume.create_box("mesh_fog"),
    transform = nvisii.transform.create("transform_fog"),
    material = nvisii.material.create("material_fog")
)
fog.get_material().set_base_color((1.,1.,1.)) 
fog.get_material().set_transmission(1) 
fog.get_material().set_roughness(0)
fog.get_volume().set_scale(100)

# Light
light = nvisii.entity.create(
    name = "light",
    light = nvisii.light.create("light"),
    transform = nvisii.transform.create("light"),
    mesh = nvisii.mesh.create_sphere("light")
)
light.get_transform().set_position((0,0,5))
light.get_transform().set_scale((.1,.1,.1))
light.get_light().set_exposure(7)

# Light blocker
blocker = nvisii.entity.create(
    name = "blocker",
    mesh = nvisii.mesh.create_capped_tube("blocker", innerRadius = .04),
    transform = nvisii.transform.create("blocker"),
    material = nvisii.material.create("blocker")
)
blocker.get_transform().set_scale((10,10,.01))
blocker.get_transform().set_position((0,0,3.0))

# Teapot
teapotahedron = nvisii.entity.create(
    name="teapotahedron",
    mesh = nvisii.mesh.create_teapotahedron("teapotahedron", segments = 32),
    transform = nvisii.transform.create("teapotahedron"),
    material = nvisii.material.create("teapotahedron")
)
teapotahedron.get_transform().set_rotation(nvisii.angleAxis(nvisii.pi() / 4.0, (0,0,1)))
teapotahedron.get_transform().set_position((0,0,0))
teapotahedron.get_transform().set_scale((0.4, 0.4, 0.4))
teapotahedron.get_material().set_base_color((255.0 / 255.0, 100.0 / 255.0, 2.0 / 256.0))  
teapotahedron.get_material().set_roughness(0.0)  
teapotahedron.get_material().set_specular(1.0) 
teapotahedron.get_material().set_metallic(1.0) 

# Make a QT window to demonstrate the difference between alpha transparency and transmission
app = QApplication([]) # Start an application.
window = QWidget() # Create a window.
layout = QVBoxLayout() # Create a layout.

def rotateCamera(value):
    value = value / 100.0
    cam_pos = camera.get_transform().get_position()

    camera.get_transform().look_at(
        at = (0, 0, 0.5), # at position
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
        at = (0, 0, 0.5), # at position
        up = (0, 0, 1),   # up vector
        eye = (cam_pos[0], cam_pos[1], 0.1 + 2.5*value)   # eye position
    )
    # print(value, 2 * math.cos(value * 2 * nvisii.pi()))

slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(rotateCameraElevation)
slider.setValue(40)
layout.addWidget(QLabel('Camera Elevation')) 
layout.addWidget(slider) 

# Add some toggles to demonstrate how the set_visibility function works

camera_visibility = True
diffuse_visibility = True
glossy_visibility = True
transmission_visibility = True
scatter_visibility = True
shadow_visibility = True
def updateVisibility():
    global camera_visibility
    global diffuse_visibility
    global glossy_visibility
    global transmission_visibility
    global scatter_visibility
    global shadow_visibility

    teapotahedron.set_visibility(
        camera = camera_visibility, 
        diffuse = diffuse_visibility, 
        glossy = glossy_visibility, 
        transmission = transmission_visibility, 
        volume_scatter = scatter_visibility, 
        shadow = shadow_visibility)

def toggleCamera():
    global camera_visibility
    camera_visibility = not camera_visibility
    updateVisibility()
button = QPushButton("toggleCamera")
button.clicked.connect(toggleCamera)
layout.addWidget(button) 

def toggleDiffuse():
    global diffuse_visibility
    diffuse_visibility = not diffuse_visibility
    updateVisibility()
button = QPushButton("toggleDiffuse")
button.clicked.connect(toggleDiffuse)
layout.addWidget(button) 

def toggleGlossy():
    global glossy_visibility
    glossy_visibility = not glossy_visibility
    updateVisibility()
button = QPushButton("toggleGlossy")
button.clicked.connect(toggleGlossy)
layout.addWidget(button) 

def toggleTransmission():
    global transmission_visibility
    transmission_visibility = not transmission_visibility
    updateVisibility()
button = QPushButton("toggleTransmission")
button.clicked.connect(toggleTransmission)
layout.addWidget(button) 

def toggleScattering():
    global scatter_visibility
    scatter_visibility = not scatter_visibility
    updateVisibility()
button = QPushButton("toggleScattering")
button.clicked.connect(toggleScattering)
layout.addWidget(button) 

def toggleShadows():
    global shadow_visibility
    shadow_visibility = not shadow_visibility
    updateVisibility()
button = QPushButton("toggleShadows")
button.clicked.connect(toggleShadows)
layout.addWidget(button) 

def setFogStrength(value):
    value = (100 - value) * 2 + 10
    fog.get_volume().set_scale(value)
setFogStrength(100)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setFogStrength)
slider.setValue(100)
layout.addWidget(QLabel('Fog Strength')) 
layout.addWidget(slider) 


def setLightHeight(value):
    value = value / 100.0
    light.get_transform().set_position((0,0,3 + value * 2))
setLightHeight(50)
slider = QSlider(Qt.Horizontal) 
slider.valueChanged[int].connect(setLightHeight)
slider.setValue(50)
layout.addWidget(QLabel('Light Height')) 
layout.addWidget(slider) 


window.setLayout(layout)
window.show() 
app.exec_() 

nvisii.deinitialize()