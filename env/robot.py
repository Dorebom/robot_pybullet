'''
Add: 2020.7.18.
Author: dorebom.b
'''

import time
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
        self.end_effector_link_index = 9
        self.force_joint_index = 7
        self.sim_times = 30
        self.act_tcp_pose = np.zeros(6)
        self.act_wrist_force = np.zeros(6)
        self.sim_interval = 0.004 # [sec]

        self.ud = UrdfDesigner()
        self.set_tool_pose(tool_pose)
        self.load(base_pose)

    def load(self, base_pose):
        #p.resetSimulation()
        orn_q = p.getQuaternionFromEuler(base_pose[3:6])
        self.robot_id = p.loadURDF("urdf/kuka_iiwa/modified_model.urdf", basePosition=base_pose[:3], baseOrientation=orn_q)
        print('robot joint num:',p.getNumJoints(self.robot_id))
        p.enableJointForceTorqueSensor(self.robot_id, self.force_joint_index)
        self.reset_joint([0.0, 0.0, 0.0, -0.5*np.pi, 0.0, 0.5*np.pi, 0.0])
        p.stepSimulation()

    def remove(self):
        p.removeBody(self.robot_id)

    def set_tool_pose(self, tool_pose):
        # 必ず、removeすること
        self.ud.load('robot')
        self.ud.modify_tcp_pose(tool_pose)
        self.ud.export()

    def reset_pose(self, tcp_pose):
        iter_count = 0
        act_pos = p.getLinkState(self.robot_id, self.end_effector_link_index)[0]
        diff_pos = np.linalg.norm(np.array(tcp_pose[:3]) - np.array(act_pos))
        while diff_pos > 0.001 and iter_count < 50:
            self.desire_joint_pos = self.calc_ik(tcp_pose)
            self.reset_joint(self.desire_joint_pos)
            act_pos = p.getLinkState(self.robot_id, self.end_effector_link_index)[0]
            diff_pos = np.linalg.norm(np.array(tcp_pose[:3]) - np.array(act_pos))
            iter_count += 1
        print(act_pos)

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
            positionGains = np.ones(7) * p_gain, \
            velocityGains = np.ones(7) * v_gain)

        for step in range(self.sim_times):
            p.stepSimulation()
            time.sleep(self.sim_interval)

    def move_to_pose(self, tcp_pose, mode='trajectory'):

        if mode == 'direct':
            cmd_joint_pos = self.calc_ik(tcp_pose)
            self.move_to_joint(cmd_joint_pos)
        elif mode == 'trajectory':
            accel_coeff = 100
            max_velocity = 50
            update_interval_time = self.sim_interval
            # [Points to be improved]
            # Manipulatorのサイズで、回転角に掛ける係数を調整したほうがいいけど、
            # 今は時間がないため、後回し
            diff_pose = np.array(tcp_pose) - self.act_tcp_pose
            norm_diff_pose = np.linalg.norm(diff_pose)

            if norm_diff_pose > 0.001:
                if norm_diff_pose < (max_velocity ** 2 / accel_coeff):
                    time_1 = np.sqrt(norm_diff_pose / accel_coeff)
                    time_2 = time_1
                    max_velocity = np.sqrt(norm_diff_pose * accel_coeff)
                    form = 'triangle'
                else:
                    time_1 = max_velocity / accel_coeff
                    time_2 = norm_diff_pose / max_velocity
                    form = 'trapezoid'

                max_count = int((time_1 + time_2) / update_interval_time) + 10

                for t in range(max_count):
                    cmd_time = t * update_interval_time
                    if form == 'triangle':
                        if cmd_time < time_1:
                            cmd_l = (0.5 * accel_coeff * np.power(cmd_time, 2)) / norm_diff_pose
                        elif time_1 <= cmd_time and cmd_time <= 2 * time_1:
                            cmd_l = (0.5 * accel_coeff * np.power(time_1, 2) + max_velocity * (cmd_time - time_1) - 0.5 * accel_coeff * np.power((cmd_time - time_1), 2)) / norm_diff_pose
                        else:
                            cmd_l = 1.0
                    if form == 'trapezoid':
                        if cmd_time < time_1:
                            cmd_l = (0.5 * accel_coeff * np.power(cmd_time, 2)) / norm_diff_pose
                        elif time_1 <= cmd_time and cmd_time < time_2:
                            cmd_l = (0.5 * accel_coeff * np.power(time_1, 2) + max_velocity * (cmd_time - time_1)) / norm_diff_pose
                        elif time_2 <= cmd_time and cmd_time < time_1 + time_2:
                            cmd_l = (0.5 * accel_coeff * np.power(time_1, 2) + max_velocity * (time_2 - time_1) \
                                    + max_velocity * (cmd_time - time_2) - 0.5 * accel_coeff * np.power((cmd_time - time_2), 2)) / norm_diff_pose
                        else:
                            cmd_l = 1.0
                    cmd_tcp_pose = self.act_tcp_pose + cmd_l * diff_pose
                    cmd_joint_pos = self.calc_ik(cmd_tcp_pose)
                    self.move_to_joint(cmd_joint_pos)

    def calc_ik(self, tcp_pose):
        return p.calculateInverseKinematics(bodyUniqueId = self.robot_id, \
                                            endEffectorLinkIndex = 9, \
                                            targetPosition = tcp_pose[:3], \
                                            targetOrientation = p.getQuaternionFromEuler(tcp_pose[3:6]))

    def reset(self, tcp_pose = [0, 0, 0, 0, 0, 0], \
                    base_pose = [0, 0, 0, 0, 0, 0], \
                    tool_pose = [0, 0, 0, 0, 0, 0]):
        #self.remove()
        self.set_tool_pose(tool_pose = tool_pose)
        self.load(base_pose = base_pose)
        self.reset_pose(tcp_pose = tcp_pose)


    def get_state(self):
        states = p.getLinkState(self.robot_id, self.end_effector_link_index)
        self.act_tcp_pose[:3] = states[0]
        self.act_tcp_pose[3:6] = p.getEulerFromQuaternion(states[1])

        force_torque = np.array(p.getJointState(self.robot_id, self.force_joint_index)[2])
        self.act_wrist_force = force_torque/6500

        return self.act_tcp_pose, self.act_wrist_force

    def _update_move_trajectory(self):
        pass