import sys
import time
import numpy as np
import pybullet as p

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

    base_pose = [0, 0, 0, 0, 0, 0]
    orn_q = p.getQuaternionFromEuler(base_pose[3:6])
    robot_id = p.loadURDF("urdf/kuka_iiwa_6/basic_model.urdf", basePosition=base_pose[:3], baseOrientation=orn_q)



