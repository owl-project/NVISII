import os

import pybullet as p
import pybullet_data
import math

import nvisii as nv


nv.initialize(window_on_top = True)

camera = nv.entity.create(
  name = "camera",
  camera = nv.camera.create("camera"),
  transform = nv.transform.create("camera")
)
nv.set_camera_entity(camera)

nv.enable_denoiser()
nv.disable_dome_light_sampling()

nv.set_dome_light_color((0,0,0))
light = nv.entity.create(
  name = "light",
  transform = nv.transform.create("light"),
  mesh = nv.mesh.create_plane("light", flip_z = True),
  light = nv.light.create("light")
)
light.get_transform().set_position((0,0,1.5))
light.get_transform().set_scale((.25,.25,.25))
light.get_light().set_temperature(4500)
light.get_light().set_exposure(2)

# Enable nvisii interactions
camera.get_transform().look_at(
    eye = (-2., 2., .4),
    at = (0, 0, .4),
    up = (0,0,1)
)
cursor = nv.vec4()
speed_camera = .20
rot = nv.vec2(0, 0)
init_rot = camera.get_transform().get_rotation()
prev_window_size = nv.vec2(0,0)

i = 0
def interact():
    global prev_window_size
    global speed_camera
    global cursor
    global init_rot
    global rot
    global i

    window_size = nv.vec2(nv.get_window_size().x, nv.get_window_size().y)
    if (nv.length(window_size - prev_window_size) > 0):
        camera.get_camera().set_fov(.8, window_size.x / float(window_size.y))
    prev_window_size = window_size

    # nvisii camera matrix 
    cam_matrix = camera.get_transform().get_local_to_world_matrix()
    dt = nv.vec4(0,0,0,0)

    # translation
    if nv.is_button_held("W"): dt[2] = -speed_camera
    if nv.is_button_held("S"): dt[2] =  speed_camera
    if nv.is_button_held("A"): dt[0] = -speed_camera
    if nv.is_button_held("D"): dt[0] =  speed_camera
    if nv.is_button_held("Q"): dt[1] = -speed_camera
    if nv.is_button_held("E"): dt[1] =  speed_camera 

    # control the camera
    if nv.length(dt) > 0.0:
        w_dt = cam_matrix * dt
        camera.get_transform().add_position(nv.vec3(w_dt))

    # camera rotation
    cursor[2] = cursor[0]
    cursor[3] = cursor[1]
    cursor[0] = nv.get_cursor_pos().x
    cursor[1] = nv.get_cursor_pos().y
    if nv.is_button_held("MOUSE_LEFT"):
        rotate_camera = True
    else:
        rotate_camera = False

    if rotate_camera:
        rot.x -= (cursor[0] - cursor[2]) * 0.001
        rot.y -= (cursor[1] - cursor[3]) * 0.001
        # init_rot = nv.angleAxis(nv.pi() * .5, (1,0,0))
        yaw = nv.angleAxis(rot.x, (0,1,0))
        pitch = nv.angleAxis(rot.y, (1,0,0)) 
        camera.get_transform().set_rotation(init_rot * yaw * pitch)

    # change speed movement
    if nv.is_button_pressed("UP"):
        speed_camera *= 0.5 
        print('decrease speed camera', speed_camera)

    if nv.is_button_pressed("DOWN"):
        speed_camera /= 0.5
        print('increase speed camera', speed_camera)
    
    # Render out an image
    if nv.is_button_pressed("SPACE"):
        i = i + 1
        nv.render_to_file(nv.get_window_size().x, nv.get_window_size().y, 256, str(i) + ".png")
nv.register_callback(interact)

# This function translates the state of all PyBullet visual objects into
# nvisii scene components
def update_visual_objects(object_ids, pkg_path, nv_objects=None):
    # object ids are in pybullet engine
    # pkg_path is for loading the object geometries
    # nv_objects refers to the already entities loaded, otherwise it is going 
    # to load the geometries and create entities. 
    if nv_objects is None:
        nv_objects = { }
    for object_id in object_ids:
        for idx, visual in enumerate(p.getVisualShapeData(object_id)):
            # Extract visual data from pybullet
            objectUniqueId = visual[0]
            linkIndex = visual[1]
            visualGeometryType = visual[2]
            dimensions = visual[3]
            meshAssetFileName = visual[4]
            local_visual_frame_position = visual[5]
            local_visual_frame_orientation = visual[6]
            rgbaColor = visual[7]

            if linkIndex == -1:
                dynamics_info = p.getDynamicsInfo(object_id,-1)
                inertial_frame_position = dynamics_info[3]
                inertial_frame_orientation = dynamics_info[4]
                base_state = p.getBasePositionAndOrientation(objectUniqueId)
                world_link_frame_position = base_state[0]
                world_link_frame_orientation = base_state[1]    
                m1 = nv.translate(nv.mat4(1), nv.vec3(inertial_frame_position[0], inertial_frame_position[1], inertial_frame_position[2]))
                m1 = m1 * nv.mat4_cast(nv.quat(inertial_frame_orientation[3], inertial_frame_orientation[0], inertial_frame_orientation[1], inertial_frame_orientation[2]))
                m2 = nv.translate(nv.mat4(1), nv.vec3(world_link_frame_position[0], world_link_frame_position[1], world_link_frame_position[2]))
                m2 = m2 * nv.mat4_cast(nv.quat(world_link_frame_orientation[3], world_link_frame_orientation[0], world_link_frame_orientation[1], world_link_frame_orientation[2]))
                m = nv.inverse(m1) * m2
                q = nv.quat_cast(m)
                world_link_frame_position = m[3]
                world_link_frame_orientation = q
            else:
                linkState = p.getLinkState(objectUniqueId, linkIndex)
                world_link_frame_position = linkState[4]
                world_link_frame_orientation = linkState[5]
            
            # Name to use for components
            object_name = f"{objectUniqueId}_{linkIndex}_{idx}"

            meshAssetFileName = meshAssetFileName.decode('UTF-8')
            if object_name not in nv_objects:
                # Create mesh component if not yet made
                if visualGeometryType == p.GEOM_MESH:
                    try:
                        nv_objects[object_name] = nv.import_scene(
                            pkg_path + "/" + meshAssetFileName
                        )
                    except Exception as e:
                        print(e)
                elif visualGeometryType == p.GEOM_BOX:
                    assert len(meshAssetFileName) == 0
                    nv_objects[object_name] = nv.entity.create(
                        name=object_name,
                        mesh=nv.mesh.create_box(
                            name=object_name,
                            # half dim in NVISII v.s. pybullet
                            size=nv.vec3(dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2)
                        ),
                        transform=nv.transform.create(object_name),
                        material=nv.material.create(object_name),
                    )
                elif visualGeometryType == p.GEOM_CYLINDER:
                    assert len(meshAssetFileName) == 0
                    length = dimensions[0]
                    radius = dimensions[1]
                    nv_objects[object_name] = nv.entity.create(
                        name=object_name,
                        mesh=nv.mesh.create_cylinder(
                            name=object_name,
                            radius=radius,
                            size=length / 2,    # size in nvisii is half of the length in pybullet
                        ),
                        transform=nv.transform.create(object_name),
                        material=nv.material.create(object_name),
                    )
                elif visualGeometryType == p.GEOM_SPHERE:
                    assert len(meshAssetFileName) == 0
                    nv_objects[object_name] = nv.entity.create(
                        name=object_name,
                        mesh=nv.mesh.create_sphere(
                            name=object_name,
                            radius=dimensions[0],
                        ),
                        transform=nv.transform.create(object_name),
                        material=nv.material.create(object_name),
                    )
                else:
                    # other primitive shapes currently not supported
                    continue

            if object_name not in nv_objects: continue

            # Link transform
            m1 = nv.translate(nv.mat4(1), nv.vec3(world_link_frame_position[0], world_link_frame_position[1], world_link_frame_position[2]))
            m1 = m1 * nv.mat4_cast(nv.quat(world_link_frame_orientation[3], world_link_frame_orientation[0], world_link_frame_orientation[1], world_link_frame_orientation[2]))

            # Visual frame transform
            m2 = nv.translate(nv.mat4(1), nv.vec3(local_visual_frame_position[0], local_visual_frame_position[1], local_visual_frame_position[2]))
            m2 = m2 * nv.mat4_cast(nv.quat(local_visual_frame_orientation[3], local_visual_frame_orientation[0], local_visual_frame_orientation[1], local_visual_frame_orientation[2]))

            # import scene directly with mesh files
            if isinstance(nv_objects[object_name], nv.scene):
                # Set root transform of visual objects collection to above transform
                nv_objects[object_name].transforms[0].set_transform(m1 * m2)
                nv_objects[object_name].transforms[0].set_scale(dimensions)

                for m in nv_objects[object_name].materials:
                    m.set_base_color((rgbaColor[0] ** 2.2, rgbaColor[1] ** 2.2, rgbaColor[2] ** 2.2))
            # for entities created for primitive shapes
            else:
                nv_objects[object_name].get_transform().set_transform(m1 * m2)
                nv_objects[object_name].get_material().set_base_color(
                    (
                        rgbaColor[0] ** 2.2,
                        rgbaColor[1] ** 2.2,
                        rgbaColor[2] ** 2.2,
                    )
                )
            # print(visualGeometryType)
    return nv_objects

# The below code is taken and modified from PyBullet's included examples.
# Look for the calls to "update_visual_objects" for nvisii specifics:
#clid = p.connect(p.SHARED_MEMORY)


p.connect(p.GUI)
# p.setAdditionalSearchPath("./content/urdf/")
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf", [0, 0, .0], useFixedBase=True)
kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
kukaEndEffectorIndex = 6
numJoints = p.getNumJoints(kukaId)
if (numJoints != 7):
  exit()

cubeId1 = p.loadURDF("cube.urdf", [1, 1, 2])
cubeId2 = p.loadURDF("cube.urdf", [-1, -1, 2])
cubeId3 = p.loadURDF("cube.urdf", [1, -1, 2])

# Keep track of the cube objects
nv_objects = update_visual_objects([planeId, kukaId, cubeId1, cubeId2, cubeId3], "")

#lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
#upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
#joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
#restposes for null space
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
#joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

for i in range(numJoints):
  p.resetJointState(kukaId, i, rp[i])

p.setGravity(0, 0, -10)
t = 0.
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 0

count = 0
useOrientation = 0
useSimulation = 1

#trailDuration is duration (in seconds) after debug lines will be removed automatically
#use 0 for no-removal
trailDuration = 15

while 1:
  t = t + 0.01

  # Periodically update nvisii components to match pybullet
  update_visual_objects([planeId, kukaId, cubeId1, cubeId2, cubeId3], ".", nv_objects)

  if (useSimulation):
    p.stepSimulation()    

  for i in range(1):
    pos = [.3 * math.sin(t * .8 * 5), 0.3 * math.cos(t * 5), 0.3 * math.sin(t * .9 * 5) + .5]
    #end effector points down, not up (in case useOrientation==1)
    orn = p.getQuaternionFromEuler([0, -math.pi, 0])

    if (useNullSpace == 1):
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos, orn, ll, ul,
                                                  jr, rp)
      else:
        jointPoses = p.calculateInverseKinematics(kukaId,
                                                  kukaEndEffectorIndex,
                                                  pos,
                                                  lowerLimits=ll,
                                                  upperLimits=ul,
                                                  jointRanges=jr,
                                                  restPoses=rp)
    else:
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(kukaId,
                                                  kukaEndEffectorIndex,
                                                  pos,
                                                  orn,
                                                  jointDamping=jd)
      else:
        jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos)

    if (useSimulation):
      for i in range(numJoints):
        p.setJointMotorControl2(bodyIndex=kukaId,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.03,
                                velocityGain=1)
    else:
      #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
      for i in range(numJoints):
        p.resetJointState(kukaId, i, jointPoses[i])

  ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
  if (hasPrevPose):
    p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, trailDuration)
    p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
  prevPose = pos
  prevPose1 = ls[4]
  hasPrevPose = 1
