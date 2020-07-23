'''
Update 2020.7.18.
Author dorebom.b
'''

import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R

class Kinematics():
    def __init__(self, robot_id, robot_param, tool_pose, current_angle):
        self.robot_id = robot_id
        self.robot_param = robot_param
        self.tool_pose = tool_pose
        self.prev_joint_angle = current_angle

    def reset_tool_pose(self, tool_pose):
        self.tool_pose = tool_pose

    def calc_inverse_kinematics(self, tcp_pose, current_angle):

        a1 = self.robot_param[0]
        a2 = self.robot_param[1]
        b  = self.robot_param[2]
        c1 = self.robot_param[3]
        c2 = self.robot_param[4]
        c3 = self.robot_param[5]
        c4 = self.robot_param[6]

        wrist_pos, wrist_orn_m = self._calc_wrist_pose(tcp_pose)

        c0 = wrist_pos
        nx1 = np.sqrt(c0[0] ** 2 + c0[1] ** 2) - a1
        s1_sq = nx1 ** 2 + (c0[2] - c1) ** 2
        s2_sq = (nx1 + 2 * a1) ** 2 + (c0[2] - c1) ** 2
        k_sq = a2 ** 2 + c3 ** 2

        q1_ = []
        q1_.append(np.arctan2(c0[1], c0[0]))
        if (q1_[0] < 0):
            q1_.append(np.arctan2(c0[1], c0[0]) + np.pi)
        else:
            q1_.append(np.arctan2(c0[1], c0[0]) - np.pi)
        # Select an angle q1 that is closer to the current joint value
        q1_idx = self._getNearestValue(q1_, current_angle[0])
        q1 = q1_[q1_idx]

        q2_ = []
        if q1_idx == 0:
            tmp = (s1_sq + c2 ** 2 - k_sq) / (2 * np.sqrt(s1_sq) * c2)
            if abs(tmp) > 1:
                #print('desired position is impossible')
                return current_angle, 0, False
            q2_.append(- np.arccos(tmp) + np.arctan2(nx1, c0[2] - c1) - np.pi / 2)
            q2_.append(np.arccos(tmp) + np.arctan2(nx1, c0[2] - c1) - np.pi / 2)
        else:
            tmp = (s2_sq + c2 ** 2 - k_sq) / (2 * np.sqrt(s2_sq) * c2)
            if abs(tmp) > 1:
                #print('desired position is impossible')
                return current_angle, 0, False
            q2_.append(- np.arccos(tmp) - np.arctan2(nx1 + 2 * a1, c0[2] - c1) - np.pi / 2)
            q2_.append(np.arccos(tmp) - np.arctan2(nx1 + 2 * a1, c0[2] - c1) - np.pi / 2)
        # Select an angle q2 that is closer to the current joint value
        q2_idx = self._getNearestValue(q2_, current_angle[1])
        q2 = q2_[q2_idx]

        q3_ = []
        if q1_idx == 0:
            tmp = (s1_sq - c2 ** 2 - k_sq) / (2 * c2 * np.sqrt(k_sq))
            if abs(tmp) > 1:
                #print('desired position is impossible')
                return current_angle, 0, False
            q3_.append(np.arccos(tmp) - np.arctan2(a2, c3) + np.pi / 2)
            q3_.append(- np.arccos(tmp) - np.arctan2(a2, c3) + np.pi / 2)
        else:
            tmp = (s2_sq - c2 ** 2 - k_sq) / (2 * c2 * np.sqrt(k_sq))
            if abs(tmp) > 1:
                #print('desired position is impossible')
                return current_angle, 0, False
            q3_.append(np.arccos(tmp) - np.arctan2(a2, c3) + np.pi / 2)
            q3_.append(- np.arccos(tmp) - np.arctan2(a2, c3) + np.pi / 2)
        q3 = q3_[q2_idx]

        e11 = wrist_orn_m[0][0]
        e12 = wrist_orn_m[0][1]
        e13 = wrist_orn_m[0][2]
        e21 = wrist_orn_m[1][0]
        e22 = wrist_orn_m[1][1]
        e23 = wrist_orn_m[1][2]
        e31 = wrist_orn_m[2][0]
        e32 = wrist_orn_m[2][1]
        e33 = wrist_orn_m[2][2]

        s1p = np.sin(q1)
        s23p = np.sin(q2 + q3)
        c1p = np.cos(q1)
        c23p = np.cos(q2 + q3)
        mp = e13 * s23p * c1p + e23 * s23p * s1p + e33 * c23p

        q4_p = np.arctan2(e23 * c1p - e13 * s1p, e13 * c23p * c1p + e23 * c23p * s1p - e33 * s23p)
        if abs(self.prev_joint_angle[3] - q4_p) > np.pi * 1.9:
            if q4_p > 0:
                q4_p = -np.pi * 2.0 + q4_p
            else:
                q4_p = np.pi * 2.0 + q4_p
        if q4_p > 0:
            q4_q = q4_p - np.pi
        elif q4_p <= 0:
            q4_q = q4_p + np.pi

        q5_p = np.arctan2(np.sqrt(1 - mp ** 2), mp)
        q5_q = - q5_p

        q6_p = np.arctan2(e12 * s23p * c1p + e22 * s23p * s1p + e32 * c23p,
                          -e11 * s23p * c1p - e21 * s23p * s1p - e31 * c23p)

        if abs(self.prev_joint_angle[5] - q6_p) > np.pi * 1.9:
            if q6_p > 0:
                q6_p = -np.pi * 2.0 + q6_p
            else:
                q6_p = np.pi * 2.0 + q6_p

        if q6_p > 0:
            q6_q = q6_p - np.pi
        elif q6_p <= 0:
            q6_q = q6_p + np.pi

        # Select an angle q4 that is closer to the current joint value
        if abs(current_angle[3] - q4_p) + abs(current_angle[5] - q6_p) <= abs(current_angle[3] - q4_q) + abs(current_angle[5] - q6_p):
            q4 = q4_p
            q5 = q5_p
            q6 = q6_p
            q4_idx = 0
        else:
            q4 = q4_q
            q5 = q5_q
            q6 = q6_q
            q4_idx = 1

        if abs(q4) > np.pi * (190.0/180.0):
            #print('desired position is impossible')
            return current_angle, 0, False

        if q1_idx == 0 and q2_idx == 0 and q4_idx == 0:
            mode = 1
        elif q1_idx == 0 and q2_idx == 1 and q4_idx == 0:
            mode = 2
        elif q1_idx == 1 and q2_idx == 0 and q4_idx == 0:
            mode = 3
        elif q1_idx == 1 and q2_idx == 1 and q4_idx == 0:
            mode = 4
        elif q1_idx == 0 and q2_idx == 0 and q4_idx == 1:
            mode = 5
        elif q1_idx == 0 and q2_idx == 1 and q4_idx == 1:
            mode = 6
        elif q1_idx == 1 and q2_idx == 0 and q4_idx == 1:
            mode = 7
        elif q1_idx == 1 and q2_idx == 1 and q4_idx == 1:
            mode = 8

        self.prev_joint_angle = [q1, q2, q3, q4, q5, q6]
        self.prevMode = mode

        return [q1, q2, q3, q4, q5, q6], mode, True

    def _calc_wrist_pose(self, tcp_pose):

        c4 = self.robot_param[6]

        tcp_pos = np.array(tcp_pose[:3])
        tcp_orn_m = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(tcp_pose[3:6]))).reshape([3,3])

        tmp_1 = self.tool_pose[:3] + np.array([0.0, 0.0, c4])
        tmp_1[1] *= -1.0

        wrist_pos = tcp_pos + np.dot(tcp_orn_m, tmp_1)
        inv_uc_orn_M = R.from_matrix(np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(self.tool_pose[3:6]) )).reshape([3,3])).inv().as_matrix()
        wrist_orn_m = np.dot(inv_uc_orn_M, tcp_orn_m)

        rot = R.from_dcm(wrist_orn_m)
        rot_e = np.array(rot.as_euler('XYZ'))
        rot_e[0] *= -1.0
        rot_e[2] *= -1.0
        rot_3 = R.from_euler('XYZ', rot_e)
        wrist_orn_m2 = rot_3.as_dcm()

        return wrist_pos, wrist_orn_m2

    def _getNearestValue(self, datalist, num):
        idx = np.abs(np.asarray(datalist) - num).argmin()
        return idx

    def reset_joint_angle(self, angle):
        q = [angle[0], angle[1], 0.0, angle[2], angle[3], angle[4], angle[5]]
        for i in range(7):
            p.resetJointState(self.robot_id, i, targetValue=q[i])
        p.stepSimulation()


