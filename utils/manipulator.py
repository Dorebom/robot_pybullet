'''
Update 2020.7.18.
Author dorebom.b
'''

import time
import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.modify_urdf import UrdfDesigner

class Manipulator():
    def __init__(self, tool_pose, base_pose):
        self.set_tool_pose(tool_pose)
        self.load(base_pose)

    def load(self, base_pose):
        #p.resetSimulation()
        orn_q = p.getQuaternionFromEuler(base_pose[3:6])
        self.robot_id = p.loadURDF("urdf/kuka_iiwa/modified_model.urdf", basePosition=base_pose[:3], baseOrientation=orn_q)
        print('robot joint num:',p.getNumJoints(self.robot_id))
        self.reset_joint_angle([0.0, 0.0, 0.0, -0.5*np.pi, 0.0, 0.5*np.pi, 0.0])


    def remove(self):
        p.removeBody(self.robot_id)

    def set_tool_pose(self, tool_pose):
        # 必ず、removeすること
        self.ud = UrdfDesigner()
        self.ud.load('robot')
        self.ud.modify_tcp_pose(tool_pose)
        self.ud.export()

    def reset_tcp_pose(self, tcp_pose):
        self.desire_joint_pos = p.calculateInverseKinematics(self.robot_id, 10, tcp_pose[:3], p.getQuaternionFromEuler(tcp_pose[3:6]))
        self.reset_joint_angle(self.desire_joint_pos)

    def reset_joint_angle(self, angle):
        for i in range(7):
            p.resetJointState(self.robot_id, i, targetValue=angle[i])
        p.stepSimulation()

