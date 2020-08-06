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
    #robot_tool_pose = [0, 0, 0, -0.3, 0.4, 0.2]
    robot_tool_pose = [0, 0, -0.1, 0.0, 0.0, 0.0]

    robot = Manipulator(tool_pose=robot_tool_pose, base_pose=robot_base_pose)

    # Reset joint position
    basic_joint = np.array([0.0, -0.5*np.pi, 1.0*np.pi, 0.0*np.pi, 0.5*np.pi, 0.0*np.pi])
    add_joint = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    robot.reset_joint(basic_joint + add_joint)

    tcp_pose, force, joint_pos = robot.get_state()
    print(tcp_pose)

    robot.move_to_pose(np.array(tcp_pose) + np.array([0.0, 0, -0.3, 0.3, 0, 0]), mode='direct')

    tcp_pose, force, joint_pos = robot.get_state()
    print(tcp_pose)

    robot.move_to_pose(np.array(tcp_pose) + np.array([0.0, 0, -0.3, 0.3, 0, 0]), mode='trajectory')

    tcp_pose, force, joint_pos = robot.get_state()
    print(tcp_pose)

    robot.move_to_pose(np.array(tcp_pose) + np.array([0.2, 0, +0.3, -0.3, 0, 0]), mode='trajectory')

    tcp_pose, force, joint_pos = robot.get_state()
    print(tcp_pose)

    time.sleep(1)