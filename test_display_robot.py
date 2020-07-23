
import sys
import time
import pybullet as p

p.connect(p.GUI)
p.setRealTimeSimulation(False)

p.loadURDF("urdf/plane/plane.urdf", [0, 0, -0.1])
# panda
p.loadURDF("urdf/franka_panda/panda.urdf", [0.0, 0.0, 0.0])
# iiwa
p.loadURDF("urdf/kuka_iiwa/model.urdf", [0.0, 1.0, 0.0])
# xarm
p.loadURDF("urdf/xarm/xarm6_robot.urdf", [0.0, 2.0, 0.0])

time.sleep(15.0)

p.disconnect()
