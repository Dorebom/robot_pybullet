'''
Add: 2020.7.18.
Author: dorebom.b
'''

import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.modify_urdf import UrdfDesigner

class Manipulator():
    '''
    1. Function list
    __init__
    load
    remove
    set_tool_pose
    reset_joint
    reset_pose
    move_to_joint
    move_to_pose
    calc_ik
    '''

    def __init__(self, tool_pose=[0, 0, 0, 0, 0, 0], base_pose=[0, 0, 0, 0, 0, 0]):
        self.sim_times = 30
        self.set_tool_pose(tool_pose)
        self.load(base_pose)

    def load(self, base_pose):
        #p.resetSimulation()
        orn_q = p.getQuaternionFromEuler(base_pose[3:6])
        self.robot_id = p.loadURDF("urdf/kuka_iiwa/modified_model.urdf", basePosition=base_pose[:3], baseOrientation=orn_q)
        print('robot joint num:',p.getNumJoints(self.robot_id))
        self.reset_joint([0.0, 0.0, 0.0, -0.5*np.pi, 0.0, 0.5*np.pi, 0.0])

    def remove(self):
        p.removeBody(self.robot_id)

    def set_tool_pose(self, tool_pose):
        # 必ず、removeすること
        self.ud = UrdfDesigner()
        self.ud.load('robot')
        self.ud.modify_tcp_pose(tool_pose)
        self.ud.export()

    def reset_pose(self, tcp_pose):
        self.desire_joint_pos = self.calc_ik(tcp_pose)
        self.reset_joint(self.desire_joint_pos)

    def reset_joint(self, angle):
        for i in range(7):
            p.resetJointState(self.robot_id, i, targetValue=angle[i])
        p.stepSimulation()

    def move_to_joint(self, cmd_joint_pos):
        p_gain = 1.0 # 2.2
        v_gain = 0.6 # 0.6

        p.setJointMotorControlArray(self.robot_id, \
            [0, 1, 2, 3, 4, 5, 6], \
            p.POSITION_CONTROL, \
            targetPositions = cmd_joint_pos, \
            positionGains = np.ones(6) * p_gain, \
            velocityGains = np.ones(6) * v_gain)

        for step in range(self.sim_times):
            p.stepSimulation()

    def move_to_pose(self, tcp_pose):
        cmd_joint_pos = self.calc_ik(tcp_pose)
        self.move_to_joint(cmd_joint_pos)

    def calc_ik(self, tcp_pose):
        return p.calculateInverseKinematics(self.robot_id, 10, tcp_pose[:3], p.getQuaternionFromEuler(tcp_pose[3:6]))