import sys
import time
import numpy as np
import pybullet as p

from utils.modify_urdf import UrdfDesigner
from utils.manipulator import Manipulator

p.connect(p.GUI)
p.setPhysicsEngineParameter(enableFileCaching=0)
p.setRealTimeSimulation(False)

tool_pose = [0, 0, 0, 0, 0, 0]
#tool_pose = [0.1, 0.1, -0.2, 0, 0, 0]
base_pose = [0, 0, 0, 0, 0, 0]
tcp_pose = [0.4, 0, 0.3, 0, 0, 0]

mp = Manipulator(tool_pose, base_pose)

mp.reset_joint_angle([0.0, 0.0, 0.0, -0.5*np.pi, 0.0, 0.5*np.pi, 0.0])

mp.reset_tcp_pose(tcp_pose)

time.sleep(5)

mp.remove()

tool_pose = [0.1, 0.1, -0.2, 0, 0, 0]
base_pose = [0, 0, 0, 0, 0, 0]
tcp_pose = [0.4, 0, 0.3, 0, 0, 0]

mp.set_tool_pose(tool_pose)
mp.load(base_pose)

mp.reset_tcp_pose(tcp_pose)

time.sleep(10)

p.disconnect()