import sys
import time
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from env.robot import Manipulator
from env.work import Work

if __name__ == '__main__':

    p.connect(p.GUI)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setRealTimeSimulation(False)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)

    # Plane
    plane_pos = [0, 0, -0.1]
    p.loadURDF("urdf/plane/plane.urdf", plane_pos)

    robot_base_pose = [0, 0, 0, 0, 0, 0]
    robot_tool_pose = [0, 0, -0.05, 0, 0, 0]

    robot = Manipulator(tool_pose=robot_tool_pose, base_pose=robot_base_pose)


    time.sleep(3)